import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange

def timestep_embedding(timesteps, dim, max_period=10000):
    """Create sinusoidal timestep embeddings with no fixed max horizon."""
    half = dim // 2
    freqs = torch.exp(
        -torch.log(torch.tensor(max_period, device=timesteps.device, dtype=torch.float32))
        * torch.arange(half, device=timesteps.device, dtype=torch.float32)
        / max(half, 1)
    )
    args = timesteps.float()[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


def modulate(x, shift, scale):
    """AdaLN-zero modulation"""
    return x * (1 + scale) + shift


class MLPBlock(nn.Module):
    """Transformer MLP without an internal norm."""

    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class SelfAttention(nn.Module):
    """Multi-head self-attention with an optional keep mask."""

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = heads * dim_head
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head**-0.5
        self.dropout = dropout
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x, mask=None):
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = (
            rearrange(t, "b n (h d) -> b h n d", h=self.heads)
            for t in (q, k, v)
        )
        scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        if mask is not None:
            scores = scores.masked_fill(~mask.bool()[None, None], -torch.finfo(scores.dtype).max)
        attn = scores.softmax(dim=-1)
        attn = F.dropout(attn, p=self.dropout, training=self.training)
        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class CrossAttention(nn.Module):
    """Multi-head cross-attention with an optional keep mask."""

    def __init__(
        self,
        query_dim,
        context_dim=None,
        heads=8,
        dim_head=64,
        dropout=0.0,
    ):
        super().__init__()
        context_dim = context_dim or query_dim
        inner_dim = heads * dim_head
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head**-0.5
        self.dropout = dropout
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(dropout))

    def forward(self, x, context, mask=None):
        q = self.to_q(x)
        k, v = self.to_kv(context).chunk(2, dim=-1)
        q = rearrange(q, "b n (h d) -> b h n d", h=self.heads)
        k = rearrange(k, "b n (h d) -> b h n d", h=self.heads)
        v = rearrange(v, "b n (h d) -> b h n d", h=self.heads)
        scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        if mask is not None:
            scores = scores.masked_fill(~mask.bool()[None, None], -torch.finfo(scores.dtype).max)
        attn = scores.softmax(dim=-1)
        attn = F.dropout(attn, p=self.dropout, training=self.training)
        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)

class SIGReg(torch.nn.Module):
    """Sketch Isotropic Gaussian Regularizer (single-GPU!)"""

    def __init__(self, knots=17, num_proj=1024):
        super().__init__()
        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj):
        """
        proj: (T, B, D)
        """
        # sample random projections
        A = torch.randn(proj.size(-1), self.num_proj, device=proj.device)
        A = A.div_(A.norm(p=2, dim=0))
        # compute the epps-pulley statistic
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean() # average over projections and time
    
class Embedder(nn.Module):
    def __init__(
        self,
        input_dim=10,
        smoothed_dim=10,
        emb_dim=10,
        mlp_scale=4,
    ):
        super().__init__()
        self.patch_embed = nn.Conv1d(input_dim, smoothed_dim, kernel_size=1, stride=1)
        self.embed = nn.Sequential(
            nn.Linear(smoothed_dim, mlp_scale * emb_dim),
            nn.SiLU(),
            nn.Linear(mlp_scale * emb_dim, emb_dim),
        )

    def forward(self, x):
        """
        x: (B, T, D)
        """
        x = x.float()
        x = x.permute(0, 2, 1)
        x = self.patch_embed(x)
        x = x.permute(0, 2, 1)
        x = self.embed(x)
        return x


class MLP(nn.Module):
    """Simple MLP with optional normalization and activation"""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim=None,
        norm_fn=nn.LayerNorm,
        act_fn=nn.GELU,
    ):
        super().__init__()
        norm_fn = norm_fn(hidden_dim) if norm_fn is not None else nn.Identity()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            norm_fn,
            act_fn(),
            nn.Linear(hidden_dim, output_dim or input_dim),
        )

    def forward(self, x):
        """
        x: (B*T, D)
        """
        return self.net(x)


class RMSNormMLP(nn.Module):
    """Token-local nonlinear projection without batch-dependent statistics."""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim=None,
        eps=1e-6,
        act_fn=nn.GELU,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.RMSNorm(hidden_dim, eps=eps),
            act_fn(),
            nn.Linear(hidden_dim, output_dim or input_dim),
        )

    def forward(self, x):
        return self.net(x)


class FlowTrajectoryBlock(nn.Module):
    """AdaLN-conditioned block for noisy future latent tokens."""

    def __init__(
        self,
        dim,
        heads,
        dim_head,
        mlp_dim,
        dropout=0.0,
    ):
        super().__init__()
        self.self_attn = SelfAttention(
            dim, heads=heads, dim_head=dim_head, dropout=dropout
        )
        self.cross_attn = CrossAttention(
            dim, heads=heads, dim_head=dim_head, dropout=dropout
        )
        self.mlp = MLPBlock(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm3 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, 9 * dim, bias=True)
        )

        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(
        self,
        x,
        context,
        t_cond,
        self_mask=None,
        cross_mask=None,
    ):
        (
            shift_sa,
            scale_sa,
            gate_sa,
            shift_ca,
            scale_ca,
            gate_ca,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = self.adaLN_modulation(t_cond).chunk(9, dim=-1)
        sa_output = self.self_attn(
            modulate(self.norm1(x), shift_sa, scale_sa),
            mask=self_mask,
        )
        x = x + gate_sa * sa_output
        ca_output = self.cross_attn(
            modulate(self.norm2(x), shift_ca, scale_ca),
            context,
            mask=cross_mask,
        )
        x = x + gate_ca * ca_output
        mlp_output = self.mlp(
            modulate(self.norm3(x), shift_mlp, scale_mlp)
        )
        x = x + gate_mlp * mlp_output
        return x


class FlowTrajectoryPredictor(nn.Module):
    """Flow-matching predictor for variable-horizon latent trajectories."""

    def __init__(
        self,
        *,
        depth,
        heads,
        mlp_dim,
        input_dim,
        hidden_dim,
        action_dim=None,
        output_dim=None,
        output_proj_hidden_dim=None,
        causal_self_attention=True,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
    ):
        super().__init__()
        action_dim = action_dim or input_dim
        output_dim = output_dim or input_dim
        self.hidden_dim = hidden_dim
        self.causal_self_attention = bool(causal_self_attention)
        self.input_proj = (
            nn.Linear(input_dim, hidden_dim)
            if input_dim != hidden_dim
            else nn.Identity()
        )
        self.context_proj = (
            nn.Linear(input_dim, hidden_dim)
            if input_dim != hidden_dim
            else nn.Identity()
        )
        self.action_proj = nn.Linear(action_dim, hidden_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.dropout = nn.Dropout(emb_dropout)
        self.layers = nn.ModuleList(
            [
                FlowTrajectoryBlock(
                    hidden_dim,
                    heads,
                    dim_head,
                    mlp_dim,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.output_proj = RMSNormMLP(
            hidden_dim,
            output_proj_hidden_dim or 4 * hidden_dim,
            output_dim,
        )

    def _positional_embedding(self, positions, dtype):
        return timestep_embedding(positions, self.hidden_dim).to(dtype=dtype)

    def _build_cross_mask(self, hist_len, pred_len, action_len, query_num, device):
        query_future_idx = torch.arange(pred_len, device=device).repeat_interleave(
            query_num
        )
        hist_mask = torch.ones(
            pred_len * query_num,
            hist_len * query_num,
            device=device,
            dtype=torch.bool,
        )
        action_idx = torch.arange(action_len, device=device)
        if pred_len == 1 and action_len > 1:
            action_mask = torch.ones(
                pred_len * query_num, action_len, device=device, dtype=torch.bool
            )
        else:
            action_mask = action_idx[None, :] <= query_future_idx[:, None]
        return torch.cat([hist_mask, action_mask], dim=-1)

    def _build_causal_self_mask(self, frame_count, query_num, device):
        """Build a causal mask over latent frames.

        Tokens within the same latent frame attend bidirectionally. A frame
        attends only to itself and earlier frames, preventing later predicted
        states from leaking into earlier ones.
        """
        frame_index = torch.arange(
            frame_count, device=device
        ).repeat_interleave(query_num)
        return frame_index[None, :] <= frame_index[:, None]

    def _build_future_self_mask(self, pred_len, query_num, device):
        if not self.causal_self_attention:
            return None
        return self._build_causal_self_mask(pred_len, query_num, device)

    def forward(self, noisy_emb, hist_emb, act_emb, t):
        """
        noisy_emb: (B, P, K, D)
        hist_emb: (B, H, K, D)
        act_emb: (B, F, A)
        t: (B,)
        """
        b, pred_len, query_num, _ = noisy_emb.shape
        hist_len = hist_emb.size(1)
        action_len = act_emb.size(1)
        dtype = noisy_emb.dtype
        device = noisy_emb.device

        action_pos = torch.arange(
            hist_len,
            hist_len + action_len,
            device=device,
            dtype=torch.float32,
        )
        if pred_len == action_len:
            pred_pos = action_pos
        elif pred_len == 1:
            pred_pos = action_pos[-1:].clone()
        else:
            pred_pos = action_pos[:pred_len]
        pred_pos_emb = self._positional_embedding(pred_pos, dtype)
        action_pos_emb = self._positional_embedding(action_pos, dtype)

        x = self.input_proj(noisy_emb)
        x = x + pred_pos_emb[None, :, None, :]
        x = rearrange(x, "b p k d -> b (p k) d")
        x = self.dropout(x)

        hist_ctx = self.context_proj(hist_emb)
        hist_ctx = rearrange(hist_ctx, "b h k d -> b (h k) d")

        act_ctx = self.action_proj(act_emb)
        act_ctx = act_ctx + action_pos_emb[None, :, :]
        context = torch.cat([hist_ctx, act_ctx], dim=1)

        t_emb = timestep_embedding(t, self.hidden_dim).to(dtype=dtype)
        t_cond = self.time_mlp(t_emb)[:, None, :]
        self_mask = self._build_future_self_mask(
            pred_len,
            query_num,
            device,
        )
        cross_mask = self._build_cross_mask(
            hist_len, pred_len, action_len, query_num, device=device
        )

        for layer in self.layers:
            x = layer(
                x,
                context,
                t_cond,
                self_mask=self_mask,
                cross_mask=cross_mask,
            )

        x = self.norm(x)
        x = self.output_proj(x)
        return rearrange(
            x,
            "b (p k) d -> b p k d",
            p=pred_len,
            k=query_num,
        )

import numpy as np


def add_noise(images, patches, clip):
    low, high = clip
    noisy = images.astype(np.float32) + patches[:, None, ...]
    return np.clip(noisy, low, high).astype(images.dtype)


def _as_xy(points):
    if points is None:
        return np.empty((0, 2), dtype=np.float32)
    points = np.asarray(points)
    if points.size == 0:
        return np.empty((0, 2), dtype=np.float32)
    points = points.reshape(-1, points.shape[-1])
    return points[:, :2].astype(np.float32)


def _append_key_points(points, src, key, env_idx, image_shape):
    if key not in src:
        return
    value = np.asarray(src[key])
    if value.shape[0] <= env_idx:
        return
    xy = _as_xy(value[env_idx])
    height, width = image_shape
    valid = (
        np.isfinite(xy).all(axis=1)
        & (xy[:, 0] >= 0)
        & (xy[:, 0] < width)
        & (xy[:, 1] >= 0)
        & (xy[:, 1] < height)
    )
    if valid.any():
        points.append(xy[valid])


def _ring_points(center, radius):
    offsets = np.array(
        [
            [0.0, 0.0],
            [radius, 0.0],
            [-radius, 0.0],
            [0.0, radius],
            [0.0, -radius],
            [radius, radius],
            [radius, -radius],
            [-radius, radius],
            [-radius, -radius],
        ],
        dtype=np.float32,
    )
    return center[None, :] + offsets


def _subsample_coords(coords, max_points=1600):
    if len(coords) <= max_points:
        return coords
    step = int(np.ceil(len(coords) / max_points))
    return coords[::step]


def _mask_points(mask):
    coords = np.argwhere(mask)
    if coords.size == 0:
        return np.empty((0, 2), dtype=np.float32)
    coords = _subsample_coords(coords)
    return coords[:, [1, 0]].astype(np.float32)


def _sample_centers(rng, protected_points, image_shape, radius, min_distance):
    height, width = image_shape
    margin = min(radius, max(0.0, (min(height, width) - 1) / 2.0))
    low = np.array([margin, margin], dtype=np.float32)
    high = np.array([width - 1 - margin, height - 1 - margin], dtype=np.float32)
    high = np.maximum(high, low)

    centers = np.zeros((len(protected_points), 2), dtype=np.float32)
    for i, points in enumerate(protected_points):
        points = _as_xy(points)
        if points.size == 0:
            centers[i] = rng.uniform(low, high)
            continue

        for _ in range(1000):
            candidate = rng.uniform(low, high)
            dist = np.linalg.norm(points - candidate[None, :], axis=1)
            if dist.min() > min_distance:
                centers[i] = candidate
                break
        else:
            xs = np.linspace(low[0], high[0], num=max(2, int(width // 8)))
            ys = np.linspace(low[1], high[1], num=max(2, int(height // 8)))
            xx, yy = np.meshgrid(xs, ys)
            grid = np.stack([xx.ravel(), yy.ravel()], axis=1)
            dist = np.linalg.norm(grid[:, None, :] - points[None, :, :], axis=2)
            centers[i] = grid[np.argmax(dist.min(axis=1))]
    return centers


def make_noise_patches(rng, perturb_cfg, protected_points, image_shape):
    noise_cfg = perturb_cfg.gaussian_noise
    if noise_cfg.center != "random_non_agent":
        raise ValueError(
            "Only gaussian_noise.center=random_non_agent is currently supported."
        )

    n, height, width = image_shape[:3]
    radius = float(noise_cfg.radius)
    std = float(noise_cfg.std)
    min_distance = radius + float(noise_cfg.get("protected_margin", 14.0))
    centers = _sample_centers(
        rng,
        protected_points,
        (height, width),
        radius=radius,
        min_distance=min_distance,
    )

    yy, xx = np.mgrid[0:height, 0:width]
    patches = np.empty((n, height, width, 3), dtype=np.float32)
    for i, center in enumerate(centers):
        dist2 = (xx - center[0]) ** 2 + (yy - center[1]) ** 2
        spatial = np.exp(-dist2 / (2.0 * radius * radius))[..., None]
        patches[i] = rng.normal(0.0, std, size=(height, width, 3)) * spatial
    return patches, centers


class GaussianNoisePerturbation:
    def __init__(self, perturb_cfg, rng, protected_points_fn):
        self.cfg = perturb_cfg
        self.rng = rng
        self.protected_points_fn = protected_points_fn
        self.noise = None
        self.centers = None
        self.clip = None

    def setup(self, world, init_state, goal_state):
        image_shape = world.infos["pixels"].shape[0:1] + world.infos["pixels"].shape[2:]
        protected_points = self.protected_points_fn(
            world,
            init_state,
            goal_state,
            image_shape[1:3],
        )
        self.noise, self.centers = make_noise_patches(
            self.rng,
            self.cfg,
            protected_points,
            image_shape,
        )
        self.clip = tuple(self.cfg.gaussian_noise.clip)
        if "goal" in goal_state:
            goal_state["goal"] = add_noise(
                goal_state["goal"][:, None, ...],
                self.noise,
                self.clip,
            )[:, 0]

    def apply_to_policy_info(self, infos):
        policy_info = infos.copy()
        policy_info["pixels"] = add_noise(infos["pixels"], self.noise, self.clip)
        return policy_info


def tworoom_protected_points(world, init_state, goal_state, image_shape):
    protected = []
    for i in range(world.num_envs):
        points = []
        for key in ("proprio", "pos_agent", "state", "pos_target"):
            _append_key_points(points, init_state, key, i, image_shape)
        for key in (
            "goal_proprio",
            "goal_pos_agent",
            "goal_state",
            "goal_pos_target",
        ):
            _append_key_points(points, goal_state, key, i, image_shape)
        protected.append(
            np.concatenate(points, axis=0)
            if points
            else np.empty((0, 2), dtype=np.float32)
        )
    return protected


def _pusht_to_pixel(xy, image_shape, flip_y):
    xy = _as_xy(xy)
    height, width = image_shape
    out = np.empty_like(xy, dtype=np.float32)
    out[:, 0] = xy[:, 0] / 512.0 * width
    y = 512.0 - xy[:, 1] if flip_y else xy[:, 1]
    out[:, 1] = y / 512.0 * height
    return out


def pusht_protected_points(world, init_state, goal_state, image_shape):
    protected = []
    for i in range(world.num_envs):
        points = []
        for src in (init_state, goal_state):
            key = "goal_state" if src is goal_state else "state"
            if key not in src:
                continue
            state = np.asarray(src[key][i], dtype=np.float32)
            anchors = np.stack([state[:2], state[2:4]], axis=0)
            for flip_y in (True, False):
                pix = _pusht_to_pixel(anchors, image_shape, flip_y)
                points.append(_ring_points(pix[0], 10.0))
                points.append(_ring_points(pix[1], 28.0))
        protected.append(
            np.concatenate(points, axis=0)
            if points
            else np.empty((0, 2), dtype=np.float32)
        )
    return protected


def _reacher_segmentation_points(env, image_shape):
    unwrapped = env.unwrapped
    height, width = image_shape
    try:
        physics = unwrapped.env.physics
        seg = physics.render(
            height,
            width,
            camera_id=unwrapped.camera_id,
            segmentation=True,
        )
        model = physics.model
        ids = []
        for name in ("target", "arm", "hand", "finger"):
            try:
                ids.append(model.name2id(name, "geom"))
            except Exception:
                pass
        if not ids:
            return np.empty((0, 2), dtype=np.float32)
        return _mask_points(np.isin(seg[..., 0], ids))
    except Exception:
        return np.empty((0, 2), dtype=np.float32)


def reacher_protected_points(world, init_state, goal_state, image_shape):
    protected = []
    for env in world.envs.envs:
        protected.append(_reacher_segmentation_points(env, image_shape))
    return protected


def _cube_color_mask_points(env, image):
    unwrapped = env.unwrapped
    targets = []
    try:
        agent = np.asarray(unwrapped.variation_space["agent"]["color"].value)
        targets.append(agent.reshape(-1, 3)[0])
    except Exception:
        pass
    try:
        cube = np.asarray(unwrapped.variation_space["cube"]["color"].value)
        targets.extend(cube.reshape(-1, 3))
    except Exception:
        pass
    if not targets:
        return np.empty((0, 2), dtype=np.float32)

    target_rgb = np.asarray(targets, dtype=np.float32)
    if target_rgb.max() <= 1.0:
        target_rgb = target_rgb * 255.0
    image_f = image.astype(np.float32)
    dist = np.linalg.norm(image_f[..., None, :] - target_rgb[None, None, :, :], axis=-1)
    mask = dist.min(axis=-1) < 95.0
    return _mask_points(mask)


def _cube_state_fallback_points(init_state, goal_state, env_idx, image_shape):
    height, width = image_shape
    points = []
    src_keys = (
        (init_state, ("proprio_effector_pos", "privileged_block_0_pos")),
        (
            goal_state,
            ("goal_proprio_effector_pos", "goal_privileged_block_0_pos"),
        ),
    )
    for src, keys in src_keys:
        for key in keys:
            if key not in src:
                continue
            xyz = np.asarray(src[key][env_idx], dtype=np.float32).reshape(-1)
            # Coarse front-camera fallback for the single-cube OGBench layout.
            x = np.clip((xyz[1] + 0.35) / 0.70 * width, 0, width - 1)
            y = np.clip((0.65 - xyz[0]) / 0.40 * height, 0, height - 1)
            points.append(np.array([[x, y]], dtype=np.float32))
    return (
        np.concatenate(points, axis=0)
        if points
        else np.empty((0, 2), dtype=np.float32)
    )


def cube_protected_points(world, init_state, goal_state, image_shape):
    protected = []
    for i, env in enumerate(world.envs.envs):
        points = []
        image = world.infos["pixels"][i, 0]
        color_points = _cube_color_mask_points(env, image)
        if len(color_points):
            points.append(color_points)
        fallback = _cube_state_fallback_points(
            init_state,
            goal_state,
            i,
            image_shape,
        )
        if len(fallback):
            points.append(fallback)
        protected.append(
            np.concatenate(points, axis=0)
            if points
            else np.empty((0, 2), dtype=np.float32)
        )
    return protected


PROTECTED_POINT_PROVIDERS = {
    "swm/TwoRoom-v1": tworoom_protected_points,
    "swm/PushT-v1": pusht_protected_points,
    "swm/ReacherDMControl-v0": reacher_protected_points,
    "swm/OGBCube-v0": cube_protected_points,
}


def build_gaussian_perturbation(perturb_cfg, env_name, seed):
    if env_name not in PROTECTED_POINT_PROVIDERS:
        raise ValueError(f"No gaussian_noise perturbation registered for env: {env_name}")
    rng = np.random.default_rng(int(perturb_cfg.get("seed", seed)))
    return GaussianNoisePerturbation(
        perturb_cfg,
        rng,
        PROTECTED_POINT_PROVIDERS[env_name],
    )

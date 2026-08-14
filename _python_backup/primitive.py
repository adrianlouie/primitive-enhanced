"""
primitive.py — Hill Climbing Image Reconstruction (Optimized)

Reconstructs a target image using geometric shapes (triangles, ellipses,
rotated ellipses, rectangles, rotated rectangles, lines) via hill climbing.
Each step evaluates many candidates and keeps the one that most improves
the perceptual match to the target.

Optimizations over baseline:
  - CIE Lab perceptual scoring
  - Analytical optimal color + alpha solve
  - Rotated ellipses, rotated rectangles, line shapes
  - Edge-aware shape placement (Sobel map)
  - Parallel candidate evaluation (multiprocessing)
  - Early exit when a great candidate is found
  - Simulated annealing fallback acceptance
  - Adaptive shape-type weighting
  - Incremental error map (only recompute changed pixels)

Usage:
    python primitive.py <image_path>
    python primitive.py <image_path> <num_shapes>
"""

import sys
import os
import time
import math
import random
from pathlib import Path
from datetime import datetime
import multiprocessing as mp

import numpy as np
from PIL import Image, ImageDraw

# ── Configuration ─────────────────────────────────────────────────────────────

NUM_SHAPES      = 1000   # total shapes to place (more = better quality, slower)
CANDIDATES      = 500    # random shapes tried per step (more = better, slower)
MUTATIONS       = 14     # mutations tried per candidate
ALPHA           = 140    # default shape opacity: 0 (invisible) – 255 (opaque)
MAX_DIM         = 600    # resize target to this on its longest side (for speed)
SAVE_GIF        = True   # save an animated GIF showing the build process
GIF_FRAME_EVERY = 7      # add a GIF frame every N accepted shapes
USE_PARALLEL    = True   # use multiprocessing for candidate evaluation

# Relative probability of each shape type (higher = chosen more often)
SHAPE_WEIGHTS = {
    "triangle":  5,
    "ellipse":   3,
    "rectangle": 2,
    "line":      2,
}

_SHAPE_POOL = [k for k, w in SHAPE_WEIGHTS.items() for _ in range(w)]


# ── LAB color space ───────────────────────────────────────────────────────────

def rgb_to_lab(rgb_f32: np.ndarray) -> np.ndarray:
    """
    Convert (N, 3) float32 RGB [0–255] to CIE Lab.
    Uses the standard sRGB → XYZ D65 → Lab pipeline.
    """
    rgb = rgb_f32 / 255.0
    # Linearize sRGB
    mask = rgb > 0.04045
    rgb  = np.where(mask, ((rgb + 0.055) / 1.055) ** 2.4, rgb / 12.92)
    # sRGB → XYZ (D65 illuminant)
    M = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ], dtype=np.float32)
    xyz = rgb @ M.T
    # Normalize by D65 white point
    xyz /= np.array([0.95047, 1.00000, 1.08883], dtype=np.float32)
    # XYZ → Lab
    eps   = 0.008856
    kappa = 903.3
    f = np.where(xyz > eps, xyz ** (1.0 / 3.0), (kappa * xyz + 16.0) / 116.0)
    L = 116.0 * f[:, 1] - 16.0
    a = 500.0 * (f[:, 0] - f[:, 1])
    b = 200.0 * (f[:, 1] - f[:, 2])
    return np.stack([L, a, b], axis=1)


# ── Scoring ───────────────────────────────────────────────────────────────────

def score_delta(
    canvas:     np.ndarray,
    target:     np.ndarray,
    shape_rgba: np.ndarray,
    target_lab: np.ndarray = None,
) -> float:
    """
    Compute total error change from compositing shape_rgba onto canvas.
    Uses CIE Lab if target_lab is provided (perceptual), otherwise RGB MSE.
    Negative = improvement (canvas looks more like target).
    """
    mask = shape_rgba[:, :, 3] > 0
    if not mask.any():
        return 0.0

    masked  = shape_rgba[mask]
    a       = masked[:, 3:4] / 255.0
    s_rgb   = masked[:, :3].astype(np.float32)
    c_rgb   = canvas[mask].astype(np.float32)
    new_rgb = a * s_rgb + (1.0 - a) * c_rgb

    if target_lab is not None:
        t_lab   = target_lab[mask]
        c_lab   = rgb_to_lab(c_rgb)
        new_lab = rgb_to_lab(new_rgb)
        old_err = np.sum((c_lab   - t_lab) ** 2)
        new_err = np.sum((new_lab - t_lab) ** 2)
    else:
        t_rgb   = target[mask].astype(np.float32)
        old_err = np.sum((c_rgb  - t_rgb) ** 2)
        new_err = np.sum((new_rgb - t_rgb) ** 2)

    return float(new_err - old_err)


def apply_shape(canvas: np.ndarray, shape_rgba: np.ndarray) -> np.ndarray:
    """Alpha-composite shape_rgba onto canvas, return new uint8 RGB array."""
    a      = shape_rgba[:, :, 3:4] / 255.0
    s_rgb  = shape_rgba[:, :, :3].astype(np.float32)
    result = a * s_rgb + (1.0 - a) * canvas.astype(np.float32)
    return result.astype(np.uint8)


def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2))


def error_weighted_point(err_map: np.ndarray) -> tuple:
    """Return (x, y) sampled proportional to err_map values."""
    flat  = err_map.flatten()
    total = flat.sum()
    if total < 1e-9:
        idx = np.random.randint(len(flat))
    else:
        flat = flat / total
        idx  = np.random.choice(len(flat), p=flat)
    H, W = err_map.shape
    return idx % W, idx // W


def compute_edge_map(target: np.ndarray) -> np.ndarray:
    """Sobel edge magnitude map from target, normalized to [0, 1]. Returns float32 (H, W)."""
    gray = target.mean(axis=2).astype(np.float32)
    p    = np.pad(gray, 1, mode='edge')
    gx = (
        -p[:-2, :-2] + p[:-2, 2:]
        - 2 * p[1:-1, :-2] + 2 * p[1:-1, 2:]
        - p[2:, :-2]  + p[2:, 2:]
    )
    gy = (
        -p[:-2, :-2] - 2 * p[:-2, 1:-1] - p[:-2, 2:]
        + p[2:, :-2]  + 2 * p[2:, 1:-1]  + p[2:, 2:]
    )
    mag = np.sqrt(gx ** 2 + gy ** 2)
    return mag / (mag.max() + 1e-8)


# ── Shape utilities ───────────────────────────────────────────────────────────

def shape_bbox(shape: dict, W: int, H: int) -> tuple:
    """Return (x1, y1, x2, y2) bounding box clamped to image bounds."""
    t = shape["type"]
    if t == "triangle":
        xs = [p[0] for p in shape["pts"]]
        ys = [p[1] for p in shape["pts"]]
        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)
    elif t == "ellipse":
        r  = max(shape["rx"], shape["ry"])
        x1, x2 = shape["x"] - r, shape["x"] + r
        y1, y2 = shape["y"] - r, shape["y"] + r
    elif t == "line":
        th = shape["thickness"]
        x1 = min(shape["x1"], shape["x2"]) - th
        x2 = max(shape["x1"], shape["x2"]) + th
        y1 = min(shape["y1"], shape["y2"]) - th
        y2 = max(shape["y1"], shape["y2"]) + th
    else:  # rectangle — use full diagonal to cover rotation
        cx = shape["x"] + shape["w"] // 2
        cy = shape["y"] + shape["h"] // 2
        r  = int(math.sqrt(shape["w"] ** 2 + shape["h"] ** 2) / 2) + 2
        x1, x2 = cx - r, cx + r
        y1, y2 = cy - r, cy + r
    return (max(0, int(x1)), max(0, int(y1)),
            min(W, int(x2) + 1), min(H, int(y2) + 1))


def sample_color(target: np.ndarray, x: int, y: int, radius: int = 10) -> tuple:
    H, W = target.shape[:2]
    x    = int(np.clip(x, 0, W - 1))
    y    = int(np.clip(y, 0, H - 1))
    x1, x2 = max(0, x - radius), min(W, x + radius + 1)
    y1, y2 = max(0, y - radius), min(H, y + radius + 1)
    region  = target[y1:y2, x1:x2].reshape(-1, 3)
    if len(region) == 0:
        region = target.reshape(-1, 3)
    c = region[random.randrange(len(region))]
    return (int(c[0]), int(c[1]), int(c[2]), ALPHA)


def solve_color_alpha(
    canvas:    np.ndarray,
    target:    np.ndarray,
    shape_arr: np.ndarray,
) -> tuple:
    """
    Analytically solve for the optimal RGB color AND optimal alpha for a shape.

    Step 1 — optimal color given alpha:
        C = (target - (1-a)*canvas) / a
    Step 2 — optimal alpha given color C:
        a* = Σ(t-c)·(C-c) / Σ(C-c)²   (closed-form MSE minimizer)

    Returns (r, g, b, alpha_int).
    """
    mask = shape_arr[:, :, 3] > 0
    if not mask.any():
        return (128, 128, 128, ALPHA)

    a0 = ALPHA / 255.0
    t  = target[mask].astype(np.float32)
    c  = canvas[mask].astype(np.float32)

    # Step 1: optimal color at initial alpha
    opt_C = np.clip((t - (1.0 - a0) * c) / a0, 0.0, 255.0)
    C_bar = opt_C.mean(axis=0)   # (3,)

    # Step 2: optimal alpha given C_bar
    diff = C_bar - c             # (M, 3)
    num  = float(np.sum((t - c) * diff))
    den  = float(np.sum(diff ** 2))
    a_opt = float(np.clip(num / den, 0.05, 1.0)) if den > 1e-6 else a0

    color = np.clip(C_bar, 0, 255).astype(np.uint8)
    return (int(color[0]), int(color[1]), int(color[2]), int(a_opt * 255))


# ── Shape generation ──────────────────────────────────────────────────────────

def random_shape(
    target:     np.ndarray,
    max_size:   int,
    focus:      tuple = None,
    edge_focus: tuple = None,
    shape_pool: list  = None,
) -> dict:
    """
    Generate a random shape, optionally biased toward error/edge hotspots.
    - edge_focus: 15% probability of placing near a detected edge
    - focus: 60% probability of placing near the highest-error region
    """
    if shape_pool is None:
        shape_pool = _SHAPE_POOL
    H, W = target.shape[:2]
    kind  = random.choice(shape_pool)
    r     = random.random()

    if edge_focus is not None and r < 0.15:
        fx, fy = edge_focus
        spread = max(max_size, 20)
        x = int(np.clip(fx + random.randint(-spread, spread), 0, W - 1))
        y = int(np.clip(fy + random.randint(-spread, spread), 0, H - 1))
    elif focus is not None and r < 0.75:
        fx, fy = focus
        spread = max(max_size, 20)
        x = int(np.clip(fx + random.randint(-spread, spread), 0, W - 1))
        y = int(np.clip(fy + random.randint(-spread, spread), 0, H - 1))
    else:
        x = random.randint(0, W - 1)
        y = random.randint(0, H - 1)

    # 20% chance: small detail shape regardless of current annealing phase
    if random.random() < 0.20:
        size = random.randint(3, max(4, max_size // 3))
    else:
        size = random.randint(max(4, max_size // 4), max_size)

    color = sample_color(target, x, y)
    angle = random.uniform(0.0, 360.0)

    if kind == "triangle":
        def pt():
            return (x + random.randint(-size, size), y + random.randint(-size, size))
        return {"type": "triangle", "pts": [pt(), pt(), pt()], "color": color}

    elif kind == "ellipse":
        rx = random.randint(max(2, size // 4), size)
        ry = random.randint(max(2, size // 4), size)
        return {"type": "ellipse", "x": x, "y": y, "rx": rx, "ry": ry,
                "angle": angle, "color": color}

    elif kind == "line":
        angle_rad = math.radians(angle)
        length    = random.randint(max(4, size // 2), size * 2)
        thickness = random.randint(1, max(2, size // 8))
        x2 = int(x + length * math.cos(angle_rad))
        y2 = int(y + length * math.sin(angle_rad))
        return {"type": "line", "x1": x, "y1": y, "x2": x2, "y2": y2,
                "thickness": thickness, "color": color}

    else:  # rectangle
        w = random.randint(max(2, size // 2), size)
        h = random.randint(max(2, size // 2), size)
        return {"type": "rectangle", "x": x, "y": y, "w": w, "h": h,
                "angle": angle, "color": color}


def mutate(shape: dict, target: np.ndarray, max_size: int) -> dict:
    """Return a mutated copy of shape — geometry, color resample, or color nudge."""
    s     = {k: (list(v) if isinstance(v, list) else v) for k, v in shape.items()}
    delta = max(2, max_size // 3)
    move  = random.randint(0, 2)

    if move == 0:  # adjust geometry
        t = s["type"]
        if t == "triangle":
            i = random.randint(0, 2)
            s["pts"][i] = (
                s["pts"][i][0] + random.randint(-delta, delta),
                s["pts"][i][1] + random.randint(-delta, delta),
            )
        elif t == "ellipse":
            s["x"]    += random.randint(-delta, delta)
            s["y"]    += random.randint(-delta, delta)
            s["rx"]    = max(2, s["rx"] + random.randint(-delta // 2, delta // 2))
            s["ry"]    = max(2, s["ry"] + random.randint(-delta // 2, delta // 2))
            s["angle"] = (s["angle"] + random.uniform(-15, 15)) % 360
        elif t == "line":
            s["x1"] += random.randint(-delta, delta)
            s["y1"] += random.randint(-delta, delta)
            s["x2"] += random.randint(-delta, delta)
            s["y2"] += random.randint(-delta, delta)
            s["thickness"] = max(1, s["thickness"] + random.randint(-1, 1))
        else:  # rectangle
            s["x"]    += random.randint(-delta, delta)
            s["y"]    += random.randint(-delta, delta)
            s["w"]     = max(2, s["w"] + random.randint(-delta // 2, delta // 2))
            s["h"]     = max(2, s["h"] + random.randint(-delta // 2, delta // 2))
            s["angle"] = (s["angle"] + random.uniform(-15, 15)) % 360

    elif move == 1:  # resample color from target
        t = s["type"]
        if t == "triangle":
            cx, cy = s["pts"][0]
        elif t == "line":
            cx, cy = s["x1"], s["y1"]
        else:
            cx, cy = s["x"], s["y"]
        s["color"] = sample_color(target, int(cx), int(cy))

    else:  # nudge color slightly
        c = list(s["color"])
        c[0] = int(np.clip(c[0] + random.randint(-25, 25), 0, 255))
        c[1] = int(np.clip(c[1] + random.randint(-25, 25), 0, 255))
        c[2] = int(np.clip(c[2] + random.randint(-25, 25), 0, 255))
        s["color"] = tuple(c)

    return s


def rasterize(shape: dict, size: tuple) -> np.ndarray:
    """Draw shape onto transparent RGBA canvas. Supports rotation for ellipse/rect."""
    t = shape["type"]

    if t == "triangle":
        img  = Image.new("RGBA", size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        draw.polygon([(p[0], p[1]) for p in shape["pts"]], fill=shape["color"])
        return np.array(img)

    elif t == "ellipse":
        rx, ry = shape["rx"], shape["ry"]
        angle  = shape.get("angle", 0.0)
        pad    = int(math.sqrt(rx ** 2 + ry ** 2)) + 4
        tmp    = Image.new("RGBA", (rx * 2 + pad * 2, ry * 2 + pad * 2), (0, 0, 0, 0))
        d      = ImageDraw.Draw(tmp)
        d.ellipse([pad, pad, pad + rx * 2, pad + ry * 2], fill=shape["color"])
        tmp    = tmp.rotate(angle, expand=True)
        canvas = Image.new("RGBA", size, (0, 0, 0, 0))
        ox     = shape["x"] - tmp.width  // 2
        oy     = shape["y"] - tmp.height // 2
        canvas.paste(tmp, (ox, oy), tmp)
        return np.array(canvas)

    elif t == "rectangle":
        w, h   = shape["w"], shape["h"]
        angle  = shape.get("angle", 0.0)
        cx     = shape["x"] + w // 2
        cy     = shape["y"] + h // 2
        pad    = int(math.sqrt(w ** 2 + h ** 2) // 2) + 4
        tmp    = Image.new("RGBA", (w + pad * 2, h + pad * 2), (0, 0, 0, 0))
        d      = ImageDraw.Draw(tmp)
        d.rectangle([pad, pad, pad + w, pad + h], fill=shape["color"])
        tmp    = tmp.rotate(angle, expand=True)
        canvas = Image.new("RGBA", size, (0, 0, 0, 0))
        ox     = cx - tmp.width  // 2
        oy     = cy - tmp.height // 2
        canvas.paste(tmp, (ox, oy), tmp)
        return np.array(canvas)

    else:  # line
        img  = Image.new("RGBA", size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        draw.line(
            [(shape["x1"], shape["y1"]), (shape["x2"], shape["y2"])],
            fill=shape["color"],
            width=shape["thickness"],
        )
        return np.array(img)


def _recolor(arr: np.ndarray, color: tuple, orig_alpha: int) -> None:
    """
    Update arr's RGB and alpha channels in-place after solve_color_alpha.
    Scales the alpha proportionally so anti-aliased edge pixels (from PIL
    rotation) retain their gradient rather than being snapped to a flat value.
    """
    r, g, b, a_new = color
    mask = arr[:, :, 3] > 0
    arr[mask, 0] = r
    arr[mask, 1] = g
    arr[mask, 2] = b
    scale = a_new / max(orig_alpha, 1)
    arr[:, :, 3] = np.clip(arr[:, :, 3].astype(np.float32) * scale, 0, 255).astype(np.uint8)


# ── Parallel worker ───────────────────────────────────────────────────────────

def _eval_batch(args):
    """
    Worker function for multiprocessing.Pool.
    Evaluates n_cands candidates (each + mutations) and returns the best found.
    Returns (best_delta, best_shape).
    """
    (canvas_bytes, canvas_shape, canvas_dtype,
     target_bytes, target_shape, target_dtype,
     target_lab_bytes, tlab_shape, tlab_dtype,
     size, max_size, focus, edge_focus,
     shape_pool, n_mutations, n_cands) = args

    canvas     = np.frombuffer(canvas_bytes,     dtype=canvas_dtype).reshape(canvas_shape)
    target     = np.frombuffer(target_bytes,     dtype=target_dtype).reshape(target_shape)
    target_lab = np.frombuffer(target_lab_bytes, dtype=tlab_dtype).reshape(tlab_shape)

    best_delta = 0.0
    best_shape = None

    for _ in range(n_cands):
        shape          = random_shape(target, max_size, focus=focus,
                                      edge_focus=edge_focus, shape_pool=shape_pool)
        shape_arr      = rasterize(shape, size)
        orig_a         = shape["color"][3]
        shape["color"] = solve_color_alpha(canvas, target, shape_arr)
        _recolor(shape_arr, shape["color"], orig_a)
        d              = score_delta(canvas, target, shape_arr, target_lab)

        if d < best_delta:
            best_delta = d
            best_shape = shape

        for _ in range(n_mutations):
            cand          = mutate(shape, target, max_size)
            cand_arr      = rasterize(cand, size)
            orig_a        = cand["color"][3]
            cand["color"] = solve_color_alpha(canvas, target, cand_arr)
            _recolor(cand_arr, cand["color"], orig_a)
            d             = score_delta(canvas, target, cand_arr, target_lab)

            if d < best_delta:
                best_delta = d
                best_shape = cand

    return (best_delta, best_shape)


# ── Hill climbing ─────────────────────────────────────────────────────────────

def hill_climb(target_path: str, num_shapes: int = NUM_SHAPES) -> None:
    # ── Load & resize ──────────────────────────────────────────────────────────
    target_img = Image.open(target_path).convert("RGB")
    W, H = target_img.size

    if max(W, H) > MAX_DIM:
        scale      = MAX_DIM / max(W, H)
        W, H       = int(W * scale), int(H * scale)
        target_img = target_img.resize((W, H), Image.LANCZOS)

    target = np.array(target_img)
    size   = (W, H)

    # ── Precompute perceptual and structural maps ───────────────────────────────
    target_lab = rgb_to_lab(
        target.reshape(-1, 3).astype(np.float32)
    ).reshape(H, W, 3).astype(np.float32)
    edge_map = compute_edge_map(target)

    # ── Init canvas ────────────────────────────────────────────────────────────
    avg_color = tuple(int(v) for v in target.mean(axis=(0, 1)))
    canvas    = np.full((H, W, 3), avg_color, dtype=np.uint8)

    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)
    stem   = Path(target_path).stem
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    frames = [Image.fromarray(canvas)]

    initial_score = mse(canvas, target)

    n_workers        = os.cpu_count() or 4
    cands_per_worker = max(1, CANDIDATES // n_workers)

    print(f"\n  Primitive — Hill Climbing Image Reconstruction")
    print(f"  {'─' * 50}")
    print(f"  Target    : {Path(target_path).name}  ({W}×{H})")
    print(f"  Shapes    : {num_shapes}   Candidates/step: {CANDIDATES}")
    print(f"  Workers   : {n_workers if USE_PARALLEL else 1}")
    print(f"  Output    : {output_dir.resolve()}/")
    print(f"  {'─' * 50}\n")

    t0         = time.time()
    focus      = None
    edge_focus = None
    err_map    = None           # cached per-pixel error map

    # Pre-serialize arrays that don't change between steps
    target_bytes     = target.tobytes()
    target_lab_bytes = target_lab.tobytes()

    pool = mp.Pool(n_workers) if USE_PARALLEL else None

    try:
        for i in range(1, num_shapes + 1):
            progress = i / num_shapes

            # ── 3-phase annealing ─────────────────────────────────────────────
            if progress < 0.40:
                t_       = progress / 0.40
                max_size = int(max(W, H) * (0.60 - 0.35 * t_))   # 60% → 25%
            elif progress < 0.75:
                t_       = (progress - 0.40) / 0.35
                max_size = int(max(W, H) * (0.25 - 0.15 * t_))   # 25% → 10%
            else:
                t_       = (progress - 0.75) / 0.25
                max_size = int(max(W, H) * (0.10 - 0.07 * t_))   # 10% → 3%
            max_size = max(4, max_size)

            # ── Update focus points (every 2 shapes) ──────────────────────────
            if i % 2 == 0 or err_map is None:
                if err_map is None:
                    err_map = np.mean(
                        (canvas.astype(np.float32) - target.astype(np.float32)) ** 2,
                        axis=2,
                    )
                focus      = error_weighted_point(err_map)
                edge_focus = error_weighted_point(edge_map)

            # ── Candidate evaluation (parallel or serial) ─────────────────────
            canvas_bytes = canvas.tobytes()
            batch_arg = (
                canvas_bytes, canvas.shape, canvas.dtype,
                target_bytes, target.shape, target.dtype,
                target_lab_bytes, target_lab.shape, target_lab.dtype,
                size, max_size, focus, edge_focus,
                _SHAPE_POOL, MUTATIONS, cands_per_worker,
            )

            if USE_PARALLEL and pool is not None:
                results = pool.map(_eval_batch, [batch_arg] * n_workers)
            else:
                # Serial fallback: run all CANDIDATES in a single batch
                serial_arg = tuple(list(batch_arg[:-1]) + [CANDIDATES])
                results = [_eval_batch(serial_arg)]

            best_delta = 0.0
            best_shape = None

            for bd, bs in results:
                if bs is not None and bd < best_delta:
                    best_delta = bd
                    best_shape = bs

            # ── Winner refinement (30 extra mutations on best) ────────────────
            if best_shape is not None:
                for _ in range(30):
                    cand          = mutate(best_shape, target, max_size)
                    cand_arr      = rasterize(cand, size)
                    orig_a        = cand["color"][3]
                    cand["color"] = solve_color_alpha(canvas, target, cand_arr)
                    _recolor(cand_arr, cand["color"], orig_a)
                    d             = score_delta(canvas, target, cand_arr, target_lab)
                    if d < best_delta:
                        best_delta = d
                        best_shape = cand

            # ── Accept shape & update state ───────────────────────────────────
            if best_shape is not None:
                canvas = apply_shape(canvas, rasterize(best_shape, size))

                # Incremental error map: only recompute bbox of accepted shape
                if err_map is not None:
                    x1, y1, x2, y2 = shape_bbox(best_shape, W, H)
                    patch_c = canvas[y1:y2, x1:x2].astype(np.float32)
                    patch_t = target[y1:y2, x1:x2].astype(np.float32)
                    err_map[y1:y2, x1:x2] = np.mean((patch_c - patch_t) ** 2, axis=2)

            if i % 10 == 0 or i == 1 or i == num_shapes:
                cur_score   = mse(canvas, target)
                improvement = (1.0 - cur_score / initial_score) * 100.0
                elapsed     = time.time() - t0
                eta         = (elapsed / i) * (num_shapes - i)
                print(
                    f"  [{i:4d}/{num_shapes}]  "
                    f"score: {cur_score:7.2f}  "
                    f"improvement: {improvement:5.1f}%  "
                    f"elapsed: {elapsed:5.1f}s  "
                    f"ETA: {eta:5.1f}s"
                )

            if SAVE_GIF and i % GIF_FRAME_EVERY == 0:
                frames.append(Image.fromarray(canvas))

    finally:
        if pool is not None:
            pool.close()
            pool.join()

    # ── Save outputs ───────────────────────────────────────────────────────────
    out_png = output_dir / f"{stem}_primitive_{run_id}.png"
    Image.fromarray(canvas).save(out_png)
    print(f"\n  Saved  → {out_png}")

    comparison = Image.new("RGB", (W * 2 + 4, H), (40, 40, 40))
    comparison.paste(target_img, (0, 0))
    comparison.paste(Image.fromarray(canvas), (W + 4, 0))
    out_cmp = output_dir / f"{stem}_comparison_{run_id}.png"
    comparison.save(out_cmp)
    print(f"  Saved  → {out_cmp}  (side-by-side: original | reconstruction)")

    if SAVE_GIF and len(frames) > 1:
        frames.append(Image.fromarray(canvas))
        out_gif = output_dir / f"{stem}_primitive_{run_id}.gif"
        frames[0].save(
            out_gif,
            save_all=True,
            append_images=frames[1:],
            duration=80,
            loop=0,
        )
        print(f"  Saved  → {out_gif}  (animated build process)")

    elapsed = time.time() - t0
    print(f"\n  Done in {elapsed:.1f}s  |  {num_shapes / elapsed:.1f} shapes/sec\n")


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("\nUsage:  python primitive.py <image_path> [num_shapes]")
        print("        python primitive.py photo.jpg")
        print("        python primitive.py photo.jpg 500\n")
        sys.exit(1)

    image_path = sys.argv[1]
    n_shapes   = int(sys.argv[2]) if len(sys.argv) > 2 else NUM_SHAPES
    hill_climb(image_path, n_shapes)

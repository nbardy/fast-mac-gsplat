"""
CPU reference for spacetime Gaussian rasterization.

This is intentionally simple and orthographic. It validates the math and the
spacetime tile-time indexing strategy before writing Metal kernels.

Run:
    python st_raster_ref.py

It will render a small toy sequence to ./st_out if Pillow is installed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import math
import numpy as np

EPS = 1e-8


@dataclass
class STGaussian:
    """4D spacetime Gaussian primitive."""

    mu: np.ndarray       # shape (4,), world spacetime center: x,y,z,t
    Sigma: np.ndarray    # shape (4,4), SPD covariance
    alpha: float         # density scale
    color: np.ndarray    # shape (3,), RGB in [0,1]

    @property
    def Q(self) -> np.ndarray:
        return np.linalg.inv(self.Sigma)

    @property
    def log_norm(self) -> float:
        sign, logdet = np.linalg.slogdet(self.Sigma)
        if sign <= 0:
            raise ValueError("Sigma must be SPD")
        return math.log(max(self.alpha, EPS)) - 2.0 * math.log(2.0 * math.pi) - 0.5 * logdet


@dataclass
class ProjectedGaussian:
    """3D sensor-time Gaussian footprint after integrating out ray depth."""

    mean_y: np.ndarray   # shape (3,), pixel_x, pixel_y, time
    H: np.ndarray        # shape (3,3), precision in sensor-time
    cov_y: np.ndarray    # shape (3,3), inverse of H
    log_amp: float
    color: np.ndarray    # shape (3,)
    depth0: float        # approximate depth at footprint center


def project_gaussian_affine(
    g: STGaussian,
    b: np.ndarray,
    A: np.ndarray,
    v: np.ndarray,
    regularize: float = 1e-6,
) -> ProjectedGaussian:
    """
    Project a 4D Gaussian through affine spacetime rays.

    gamma_y(s) = b + A @ y + s * v
    y = (pixel_x, pixel_y, tau)
    """
    Q = g.Q
    c = b - g.mu
    a = float(v @ Q @ v)
    if a <= EPS:
        raise ValueError("Ray direction has near-zero precision length")

    Qv = Q @ v
    Q_perp = Q - np.outer(Qv, Qv) / a

    H = A.T @ Q_perp @ A
    # Numerical regularization. In a production system, clamp primitive covariances upstream.
    H = 0.5 * (H + H.T) + regularize * np.eye(3)

    h = A.T @ Q_perp @ c
    cov_y = np.linalg.inv(H)
    mean_y = -cov_y @ h

    const = float(c @ Q_perp @ c - h @ cov_y @ h)
    log_amp = g.log_norm + 0.5 * math.log((2.0 * math.pi) / a) - 0.5 * const

    # Depth of maximum contribution at mean_y.
    depth0 = -float(v @ Q @ (c + A @ mean_y)) / a

    return ProjectedGaussian(
        mean_y=mean_y,
        H=H,
        cov_y=cov_y,
        log_amp=log_amp,
        color=g.color,
        depth0=depth0,
    )


def make_ortho_camera_affine(
    width: int,
    height: int,
    world_scale: float,
    time0: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Orthographic camera rays:
        x = (px - width/2) * world_scale
        y = (py - height/2) * world_scale
        z = s
        t = tau

    gamma(px,py,tau,s) = b + A @ [px,py,tau] + s*v
    """
    b = np.array([-0.5 * width * world_scale, -0.5 * height * world_scale, 0.0, time0], dtype=np.float64)
    A = np.array(
        [
            [world_scale, 0.0, 0.0],
            [0.0, world_scale, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    v = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float64)
    return b, A, v


def gaussian_aabb_sensor(pg: ProjectedGaussian, R: float) -> Tuple[np.ndarray, np.ndarray]:
    """Conservative axis-aligned bound for d^2 <= R^2."""
    radii = R * np.sqrt(np.maximum(np.diag(pg.cov_y), 0.0))
    return pg.mean_y - radii, pg.mean_y + radii


def build_tile_time_index(
    pgs: Sequence[ProjectedGaussian],
    width: int,
    height: int,
    tile_w: int,
    tile_h: int,
    time0: float,
    time1: float,
    time_bins: int,
    R: float = 3.0,
) -> Tuple[List[List[int]], Tuple[int, int, int]]:
    """
    Build a compact-ish Python list index over (tile_x,tile_y,time_bin).
    Metal implementation should use count -> prefix -> fill buffers.
    """
    nx = (width + tile_w - 1) // tile_w
    ny = (height + tile_h - 1) // tile_h
    nt = time_bins
    cells: List[List[int]] = [[] for _ in range(nx * ny * nt)]
    dt = (time1 - time0) / max(nt, 1)

    def cell_id(cx: int, cy: int, ct: int) -> int:
        return (ct * ny + cy) * nx + cx

    for pid, pg in enumerate(pgs):
        lo, hi = gaussian_aabb_sensor(pg, R)
        min_x = max(0, int(math.floor(lo[0] / tile_w)))
        max_x = min(nx - 1, int(math.floor(hi[0] / tile_w)))
        min_y = max(0, int(math.floor(lo[1] / tile_h)))
        max_y = min(ny - 1, int(math.floor(hi[1] / tile_h)))
        min_t = max(0, int(math.floor((lo[2] - time0) / dt)))
        max_t = min(nt - 1, int(math.floor((hi[2] - time0) / dt)))
        if max_x < min_x or max_y < min_y or max_t < min_t:
            continue
        for ct in range(min_t, max_t + 1):
            for cy in range(min_y, max_y + 1):
                for cx in range(min_x, max_x + 1):
                    cells[cell_id(cx, cy, ct)].append(pid)

    # Optional: sort by approximate depth once per cell.
    for lst in cells:
        lst.sort(key=lambda pid: pgs[pid].depth0)
    return cells, (nx, ny, nt)


def render_sequence_order_independent(
    pgs: Sequence[ProjectedGaussian],
    cells: Sequence[Sequence[int]],
    grid_shape: Tuple[int, int, int],
    width: int,
    height: int,
    frames: int,
    tile_w: int,
    tile_h: int,
    time0: float,
    time1: float,
    R: float = 3.0,
) -> np.ndarray:
    """
    Render using order-independent volume compositing:
        tau = sum mass_i
        rgb = weighted_mean_color * (1 - exp(-tau))
    """
    nx, ny, nt = grid_shape
    out = np.zeros((frames, height, width, 4), dtype=np.float32)
    dt_frame = (time1 - time0) / max(frames - 1, 1)
    dt_bin = (time1 - time0) / max(nt, 1)
    R2 = R * R

    def cell_id(cx: int, cy: int, ct: int) -> int:
        return (ct * ny + cy) * nx + cx

    for f in range(frames):
        tau_f = time0 + f * dt_frame
        ct = min(nt - 1, max(0, int(math.floor((tau_f - time0) / dt_bin))))
        for py in range(height):
            cy = py // tile_h
            for px in range(width):
                cx = px // tile_w
                y = np.array([px + 0.5, py + 0.5, tau_f], dtype=np.float64)
                accum_tau = 0.0
                accum_rgb = np.zeros(3, dtype=np.float64)
                for pid in cells[cell_id(cx, cy, ct)]:
                    pg = pgs[pid]
                    d = y - pg.mean_y
                    d2 = float(d @ pg.H @ d)
                    if d2 > R2:
                        continue
                    mass = math.exp(pg.log_amp - 0.5 * d2)
                    accum_tau += mass
                    accum_rgb += mass * pg.color
                if accum_tau > EPS:
                    alpha = 1.0 - math.exp(-accum_tau)
                    rgb = accum_rgb / accum_tau * alpha
                    out[f, py, px, :3] = np.clip(rgb, 0.0, 1.0)
                    out[f, py, px, 3] = alpha
    return out


def render_sequence_sorted_alpha(
    pgs: Sequence[ProjectedGaussian],
    cells: Sequence[Sequence[int]],
    grid_shape: Tuple[int, int, int],
    width: int,
    height: int,
    frames: int,
    tile_w: int,
    tile_h: int,
    time0: float,
    time1: float,
    R: float = 3.0,
) -> np.ndarray:
    """Approximate sorted-alpha compositing using per-cell depth-sorted lists."""
    nx, ny, nt = grid_shape
    out = np.zeros((frames, height, width, 4), dtype=np.float32)
    dt_frame = (time1 - time0) / max(frames - 1, 1)
    dt_bin = (time1 - time0) / max(nt, 1)
    R2 = R * R

    def cell_id(cx: int, cy: int, ct: int) -> int:
        return (ct * ny + cy) * nx + cx

    for f in range(frames):
        tau_f = time0 + f * dt_frame
        ct = min(nt - 1, max(0, int(math.floor((tau_f - time0) / dt_bin))))
        for py in range(height):
            cy = py // tile_h
            for px in range(width):
                cx = px // tile_w
                y = np.array([px + 0.5, py + 0.5, tau_f], dtype=np.float64)
                T = 1.0
                rgb = np.zeros(3, dtype=np.float64)
                for pid in cells[cell_id(cx, cy, ct)]:
                    pg = pgs[pid]
                    d = y - pg.mean_y
                    d2 = float(d @ pg.H @ d)
                    if d2 > R2:
                        continue
                    mass = math.exp(pg.log_amp - 0.5 * d2)
                    a = 1.0 - math.exp(-mass)
                    rgb += T * a * pg.color
                    T *= (1.0 - a)
                    if T < 1e-4:
                        break
                out[f, py, px, :3] = np.clip(rgb, 0.0, 1.0)
                out[f, py, px, 3] = 1.0 - T
    return out


def make_constant_velocity_gaussian(
    x0: np.ndarray,
    v_xyz: np.ndarray,
    t0: float,
    sigma_x: float,
    sigma_t: float,
    alpha: float,
    color: np.ndarray,
) -> STGaussian:
    """
    Construct the exact 4D covariance for a 3D isotropic Gaussian moving at constant velocity.

    Density proportional to:
        exp(-||x - x0 - v*(t-t0)||^2/(2 sigma_x^2) - (t-t0)^2/(2 sigma_t^2))
    """
    v = np.asarray(v_xyz, dtype=np.float64)
    Q = np.zeros((4, 4), dtype=np.float64)
    Q[:3, :3] = np.eye(3) / (sigma_x * sigma_x)
    Q[:3, 3] = -v / (sigma_x * sigma_x)
    Q[3, :3] = -v / (sigma_x * sigma_x)
    Q[3, 3] = 1.0 / (sigma_t * sigma_t) + float(v @ v) / (sigma_x * sigma_x)
    Sigma = np.linalg.inv(Q)
    mu = np.array([x0[0], x0[1], x0[2], t0], dtype=np.float64)
    return STGaussian(mu=mu, Sigma=Sigma, alpha=alpha, color=np.asarray(color, dtype=np.float64))


def demo() -> np.ndarray:
    width, height, frames = 128, 96, 24
    time0, time1 = 0.0, 1.0
    world_scale = 1.0 / 48.0
    tile_w, tile_h = 8, 8
    time_bins = 8

    b, A, ray_v = make_ortho_camera_affine(width, height, world_scale, time0=time0)

    prims = [
        make_constant_velocity_gaussian(
            x0=np.array([-0.55, -0.2, 2.0]),
            v_xyz=np.array([0.95, 0.35, 0.0]),
            t0=0.5,
            sigma_x=0.045,
            sigma_t=0.36,
            alpha=220.0,
            color=np.array([1.0, 0.25, 0.10]),
        ),
        make_constant_velocity_gaussian(
            x0=np.array([0.42, 0.28, 1.5]),
            v_xyz=np.array([-0.75, -0.28, 0.0]),
            t0=0.5,
            sigma_x=0.055,
            sigma_t=0.32,
            alpha=180.0,
            color=np.array([0.1, 0.65, 1.0]),
        ),
    ]

    pgs = [project_gaussian_affine(g, b=b, A=A, v=ray_v) for g in prims]
    cells, shape = build_tile_time_index(
        pgs, width, height, tile_w, tile_h, time0, time1, time_bins, R=4.0
    )

    total_refs = sum(len(c) for c in cells)
    nonempty = sum(1 for c in cells if c)
    print(f"projected={len(pgs)} cells={len(cells)} nonempty={nonempty} totalRefs={total_refs}")

    return render_sequence_order_independent(
        pgs, cells, shape, width, height, frames, tile_w, tile_h, time0, time1, R=4.0
    )


if __name__ == "__main__":
    imgs = demo()
    out_dir = Path("st_out")
    out_dir.mkdir(exist_ok=True)
    try:
        from PIL import Image
    except Exception:
        print("Pillow not installed; returning array only.")
    else:
        for i, img in enumerate(imgs):
            rgb = (np.clip(img[..., :3], 0.0, 1.0) * 255.0).astype(np.uint8)
            Image.fromarray(rgb).save(out_dir / f"frame_{i:03d}.png")
        print(f"wrote {len(imgs)} frames to {out_dir.resolve()}")

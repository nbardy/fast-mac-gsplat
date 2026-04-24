from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from torch_gsplat_bridge_v73.reference import (  # noqa: E402
    capture_frontk_state_dense,
    dense_reference_forward,
    manual_backward_from_state,
    reference_forward_state,
    sort_projected_inputs,
    tiled_scan_work,
    unpack_front_count,
    unpack_front_overflow,
)


def make_case(B: int, G: int, H: int, W: int):
    torch.manual_seed(1234 + B * 17 + G * 3 + H + W)
    means2d = torch.rand(B, G, 2, dtype=torch.float32)
    means2d[..., 0] *= float(W - 1)
    means2d[..., 1] *= float(H - 1)

    base = torch.rand(B, G, 2, 2, dtype=torch.float32)
    mats = torch.matmul(base.transpose(-1, -2), base) + 0.15 * torch.eye(2)
    conics = torch.stack([mats[..., 0, 0], mats[..., 0, 1], mats[..., 1, 1]], dim=-1)
    colors = torch.rand(B, G, 3, dtype=torch.float32)
    opacities = torch.rand(B, G, dtype=torch.float32) * 0.9 + 0.05
    depths = torch.rand(B, G, dtype=torch.float32)
    return means2d, conics, colors, opacities, depths


def run_case(B: int, G: int, H: int, W: int, K: int, tile_size: int):
    alpha_threshold = 1.0 / 255.0
    background = (0.03, 0.04, 0.05)
    eps = 1e-8

    means2d, conics, colors, opacities, depths = make_case(B, G, H, W)
    means2d_s, conics_s, colors_s, opacities_s, depths_s, _ = sort_projected_inputs(
        means2d, conics, colors, opacities, depths
    )

    state = reference_forward_state(
        means2d_s,
        conics_s,
        colors_s,
        opacities_s,
        depths_s,
        height=H,
        width=W,
        front_k=K,
        tile_size=tile_size,
        alpha_threshold=alpha_threshold,
        background=background,
        eps=eps,
    )

    dense_ids, dense_raw, dense_count, dense_overflow = capture_frontk_state_dense(
        means2d_s,
        conics_s,
        opacities_s,
        height=H,
        width=W,
        front_k=K,
        alpha_threshold=alpha_threshold,
    )

    tiled_count = torch.empty_like(dense_count)
    tiled_overflow = torch.empty_like(dense_overflow)
    for b in range(B):
        for y in range(H):
            for x in range(W):
                meta = int(state.front_meta[b, y, x])
                tiled_count[b, y, x] = unpack_front_count(meta)
                tiled_overflow[b, y, x] = 1 if unpack_front_overflow(meta) else 0

    assert torch.equal(dense_count, tiled_count), "front_count mismatch"
    assert torch.equal(dense_overflow, tiled_overflow), "overflow mismatch"

    for b in range(B):
        for y in range(H):
            for x in range(W):
                n = int(dense_count[b, y, x])
                if n == 0:
                    continue
                assert torch.equal(
                    dense_ids[b, y, x, :n].to(torch.int32),
                    state.front_ids[b, y, x, :n].to(torch.int32),
                ), f"front_ids mismatch at {(b, y, x)}"
                assert torch.allclose(
                    dense_raw[b, y, x, :n],
                    state.front_raw_alpha[b, y, x, :n],
                    atol=1e-7,
                    rtol=0.0,
                ), f"front_raw_alpha mismatch at {(b, y, x)}"

    dense_work = H * W * G * B
    sparse_work = tiled_scan_work(state.tile_offsets, height=H, width=W, tile_size=tile_size)
    assert sparse_work <= dense_work, "tiled work should not exceed dense work for these random cases"

    means_req = means2d_s.clone().requires_grad_(True)
    conics_req = conics_s.clone().requires_grad_(True)
    colors_req = colors_s.clone().requires_grad_(True)
    opacities_req = opacities_s.clone().requires_grad_(True)
    image = dense_reference_forward(
        means_req,
        conics_req,
        colors_req,
        opacities_req,
        H,
        W,
        alpha_threshold,
        background,
    )
    grad_seed = torch.randn_like(image)
    loss = torch.sum(image * grad_seed)
    g_means_ref, g_conics_ref, g_colors_ref, g_opacity_ref = torch.autograd.grad(
        loss,
        (means_req, conics_req, colors_req, opacities_req),
        allow_unused=False,
    )

    g_means, g_conics, g_colors, g_opacity, g_depths = manual_backward_from_state(
        grad_seed,
        means2d_s,
        conics_s,
        colors_s,
        opacities_s,
        height=H,
        width=W,
        tile_size=tile_size,
        alpha_threshold=alpha_threshold,
        background=background,
        eps=eps,
        state=state,
    )

    print(
        f"case B={B} G={G} H={H} W={W} K={K} tile={tile_size} "
        f"dense_work={dense_work} tiled_work={sparse_work} ratio={dense_work/max(sparse_work,1):.2f}x"
    )
    print("  max |g_means - ref|   =", float((g_means - g_means_ref).abs().max()))
    print("  max |g_conics - ref|  =", float((g_conics - g_conics_ref).abs().max()))
    print("  max |g_colors - ref|  =", float((g_colors - g_colors_ref).abs().max()))
    print("  max |g_opacity - ref| =", float((g_opacity - g_opacity_ref).abs().max()))
    print("  max |g_depths|        =", float(g_depths.abs().max()))

    assert torch.allclose(g_means, g_means_ref, atol=2e-6, rtol=0.0)
    assert torch.allclose(g_conics, g_conics_ref, atol=2e-6, rtol=0.0)
    assert torch.allclose(g_colors, g_colors_ref, atol=2e-6, rtol=0.0)
    assert torch.allclose(g_opacity, g_opacity_ref, atol=2e-6, rtol=0.0)
    assert float(g_depths.abs().max()) == 0.0


def main():
    run_case(B=1, G=8, H=7, W=9, K=2, tile_size=4)
    run_case(B=1, G=12, H=8, W=8, K=2, tile_size=4)
    run_case(B=2, G=10, H=6, W=7, K=4, tile_size=4)
    print("reference_tiled_exactness: all checks passed")


if __name__ == "__main__":
    main()

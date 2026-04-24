from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from torch_gsplat_bridge_v72.reference import build_tile_bins, tiled_scan_work  # noqa: E402


def make_clustered_scene(G: int, H: int, W: int, cluster_count: int = 8):
    torch.manual_seed(2026)
    centers = torch.rand(cluster_count, 2)
    centers[:, 0] *= W - 1
    centers[:, 1] *= H - 1
    assignments = torch.randint(0, cluster_count, (G,))
    means = centers[assignments] + 6.0 * torch.randn(G, 2)
    means[:, 0].clamp_(0, W - 1)
    means[:, 1].clamp_(0, H - 1)

    scales = 0.04 + 0.08 * torch.rand(G, 2)
    a = 1.0 / (scales[:, 0] ** 2)
    c = 1.0 / (scales[:, 1] ** 2)
    b = torch.zeros_like(a)
    conics = torch.stack([a, b, c], dim=-1)
    opacities = 0.35 + 0.55 * torch.rand(G)
    return means.unsqueeze(0), conics.unsqueeze(0), opacities.unsqueeze(0)


def make_uniform_scene(G: int, H: int, W: int):
    torch.manual_seed(2027)
    means = torch.rand(G, 2)
    means[:, 0] *= W - 1
    means[:, 1] *= H - 1

    scales = 0.04 + 0.08 * torch.rand(G, 2)
    a = 1.0 / (scales[:, 0] ** 2)
    c = 1.0 / (scales[:, 1] ** 2)
    b = torch.zeros_like(a)
    conics = torch.stack([a, b, c], dim=-1)
    opacities = 0.35 + 0.55 * torch.rand(G)
    return means.unsqueeze(0), conics.unsqueeze(0), opacities.unsqueeze(0)


def report(name: str, means: torch.Tensor, conics: torch.Tensor, opacities: torch.Tensor, H: int, W: int, tile_size: int):
    offsets, ids = build_tile_bins(
        means,
        conics,
        opacities,
        height=H,
        width=W,
        tile_size=tile_size,
        alpha_threshold=1.0 / 255.0,
    )
    dense_work = H * W * means.shape[1]
    sparse_work = tiled_scan_work(offsets, height=H, width=W, tile_size=tile_size)
    bins = offsets[0, 1:] - offsets[0, :-1]
    print(f"{name}:")
    print(f"  dense pixel*splat work    = {dense_work:,}")
    print(f"  tiled pixel*bin work      = {sparse_work:,}")
    print(f"  reduction                 = {dense_work / max(sparse_work, 1):.2f}x")
    print(f"  total tile refs           = {int(ids.numel()):,}")
    print(f"  mean contributors / tile  = {float(bins.float().mean()):.2f}")
    print(f"  p95 contributors / tile   = {float(torch.quantile(bins.float(), 0.95)):.2f}")
    print(f"  max contributors / tile   = {int(bins.max())}")


def main():
    H, W, G = 512, 512, 6000
    tile_size = 16
    report("clustered_512_6k", *make_clustered_scene(G, H, W), H, W, tile_size)
    report("uniform_512_6k", *make_uniform_scene(G, H, W), H, W, tile_size)


if __name__ == "__main__":
    main()

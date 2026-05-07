from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import torch

VARIANT_ROOT = Path(__file__).resolve().parents[1]
BENCH_DIR = Path(__file__).resolve().parent
for path in (VARIANT_ROOT, BENCH_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from benchmark_mps import make_case
from torch_gsplat_bridge_v6_refined_features_f32_reduce import RasterConfig, get_runtime_shader_config
import torch_gsplat_bridge_v6_refined_features_f32_reduce.rasterize as rz


def _q(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    tensor = torch.tensor(values, dtype=torch.float32)
    return float(torch.quantile(tensor, q).item())


def _summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
    tensor = torch.tensor(values, dtype=torch.float32)
    return {
        "mean": float(tensor.mean().item()),
        "p50": float(torch.quantile(tensor, 0.50).item()),
        "p95": float(torch.quantile(tensor, 0.95).item()),
        "max": float(tensor.max().item()),
    }


def _sample_contributors(
    means2d: torch.Tensor,
    conics: torch.Tensor,
    opacities: torch.Tensor,
    tile_counts: torch.Tensor,
    tile_offsets: torch.Tensor,
    binned_ids: torch.Tensor,
    cfg: RasterConfig,
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    rng = random.Random(seed)
    B = int(means2d.shape[0])
    H = int(cfg.height)
    W = int(cfg.width)
    tile_size = int(cfg.tile_size)
    tiles_x = (W + tile_size - 1) // tile_size
    tiles_y = (H + tile_size - 1) // tile_size
    tiles_per_image = tiles_x * tiles_y

    means_cpu = means2d.reshape(-1, 2).detach().cpu()
    conics_cpu = conics.reshape(-1, 3).detach().cpu()
    opacities_cpu = opacities.reshape(-1).detach().cpu()
    counts_cpu = tile_counts.detach().cpu().to(torch.long)
    offsets_cpu = tile_offsets.detach().cpu().to(torch.long)
    ids_cpu = binned_ids.detach().cpu().to(torch.long)

    checked_prefixes: list[float] = []
    contributors: list[float] = []
    tile_candidates: list[float] = []
    final_transmittance: list[float] = []
    alpha_mass: list[float] = []
    stopped_early = 0
    overflow_samples = 0

    for _ in range(samples):
        pix = rng.randrange(B * H * W)
        batch = pix // (H * W)
        rem = pix - batch * H * W
        y = rem // W
        x = rem - y * W
        tile_x = x // tile_size
        tile_y = y // tile_size
        tile = batch * tiles_per_image + tile_y * tiles_x + tile_x
        count = int(counts_cpu[tile].item())
        tile_candidates.append(float(count))
        if count <= 0:
            checked_prefixes.append(0.0)
            contributors.append(0.0)
            final_transmittance.append(1.0)
            alpha_mass.append(0.0)
            continue
        if count > int(cfg.max_fast_pairs):
            overflow_samples += 1

        start = int(offsets_cpu[tile].item())
        ids = torch.sort(ids_cpu[start : start + count]).values
        m = means_cpu.index_select(0, ids)
        q = conics_cpu.index_select(0, ids)
        o = opacities_cpu.index_select(0, ids)
        dx = float(x) + 0.5 - m[:, 0]
        dy = float(y) + 0.5 - m[:, 1]
        power = -0.5 * (q[:, 0] * dx * dx + 2.0 * q[:, 1] * dx * dy + q[:, 2] * dy * dy)
        raw_alpha = o * torch.exp(power)
        alpha = torch.minimum(raw_alpha, torch.tensor(0.99, dtype=raw_alpha.dtype))
        alpha = torch.where((power <= 0.0) & (alpha >= float(cfg.alpha_threshold)), alpha, torch.zeros_like(alpha))
        trans = torch.cumprod(1.0 - alpha, dim=0)
        stop_hits = torch.nonzero(trans <= float(cfg.transmittance_threshold), as_tuple=False)
        if stop_hits.numel():
            prefix = int(stop_hits[0].item()) + 1
            stopped_early += 1
        else:
            prefix = count
        active_alpha = alpha[:prefix]
        checked_prefixes.append(float(prefix))
        contributors.append(float((active_alpha > 0.0).sum().item()))
        final_transmittance.append(float(trans[prefix - 1].item()) if prefix > 0 else 1.0)
        alpha_mass.append(float(active_alpha.sum().item()))

    return {
        "sampled_pixels": int(samples),
        "sample_seed": int(seed),
        "overflow_sample_count": int(overflow_samples),
        "early_stop_fraction": float(stopped_early / max(samples, 1)),
        "tile_candidates": _summary(tile_candidates),
        "checked_depth_prefix": _summary(checked_prefixes),
        "alpha_passing_contributors": _summary(contributors),
        "final_transmittance": _summary(final_transmittance),
        "sum_alpha_before_stop": _summary(alpha_mass),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--gaussians", type=int, default=8192)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--feature-dim", type=int, default=32)
    p.add_argument("--case", type=str, default="medium_sigma_3_8")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--samples", type=int, default=2048)
    p.add_argument("--sample-seed", type=int, default=123)
    p.add_argument("--batch-strategy", type=str, default="flatten")
    p.add_argument("--max-fast-pairs", type=int, default=-1)
    p.add_argument("--alpha-threshold", type=float, default=1.0 / 255.0)
    p.add_argument("--transmittance-threshold", type=float, default=1.0e-4)
    p.add_argument("--json", action="store_true")
    args = p.parse_args()

    if not torch.backends.mps.is_available():
        raise SystemExit("MPS is not available.")
    device = torch.device("mps")
    means2d, conics, colors, opacities, depths = make_case(
        args.case,
        args.batch_size,
        args.gaussians,
        args.height,
        args.width,
        args.feature_dim,
        device,
        args.seed,
    )

    rt = get_runtime_shader_config()
    cfg = RasterConfig(
        height=args.height,
        width=args.width,
        tile_size=rt.tile_size,
        max_fast_pairs=args.max_fast_pairs if args.max_fast_pairs > 0 else rt.fast_cap,
        batch_strategy=args.batch_strategy,
        active_policy="off",
        alpha_threshold=args.alpha_threshold,
        transmittance_threshold=args.transmittance_threshold,
    )
    rz._check_inputs(means2d, conics, colors, opacities, depths)
    means2d_b, conics_b, colors_b, opacities_b, depths_b, _was_batched = rz._normalize_inputs(
        means2d, conics, colors, opacities, depths
    )
    B, G = means2d_b.shape[:2]
    F = colors_b.shape[-1]
    rz._runtime_validate(cfg, F)
    tiles_y = (cfg.height + cfg.tile_size - 1) // cfg.tile_size
    tiles_x = (cfg.width + cfg.tile_size - 1) // cfg.tile_size
    chunk_b = rz._choose_batch_chunk_size(cfg, B, G, tiles_y * tiles_x)
    if chunk_b != B:
        raise SystemExit("profile_contributors currently requires a flatten batch chunk; pass --batch-strategy flatten")

    _, m_s_b, q_s_b, _c_s_b, o_s_b = rz._maybe_sort_inputs_by_depth(
        means2d_b,
        conics_b,
        colors_b,
        opacities_b,
        depths_b,
        inputs_sorted_by_depth=False,
    )
    m_s = m_s_b.reshape(-1, 2)
    q_s = q_s_b.reshape(-1, 3)
    c_dummy = colors_b.reshape(-1, F)
    o_s = o_s_b.reshape(-1)
    meta_i32, meta_f32 = rz._make_meta(cfg, means2d.device, B, G, F)
    tile_counts, tile_offsets, binned_ids = torch.ops.gsplat_metal_v6_refined_features_f32_reduce.bin(
        m_s, q_s, c_dummy, o_s, meta_i32, meta_f32
    )
    torch.mps.synchronize()

    counts_cpu = tile_counts.detach().cpu().to(torch.float32)
    contributor_stats = _sample_contributors(
        m_s_b,
        q_s_b,
        o_s_b,
        tile_counts,
        tile_offsets,
        binned_ids,
        cfg,
        samples=args.samples,
        seed=args.sample_seed,
    )
    result = {
        "case": args.case,
        "seed": args.seed,
        "height": args.height,
        "width": args.width,
        "gaussians": args.gaussians,
        "feature_dim": args.feature_dim,
        "batch_size": args.batch_size,
        "batch_strategy": args.batch_strategy,
        "alpha_threshold": args.alpha_threshold,
        "transmittance_threshold": args.transmittance_threshold,
        "runtime_tile_size": rt.tile_size,
        "runtime_chunk": rt.chunk_size,
        "runtime_fast_cap": rt.fast_cap,
        "tiles": int(counts_cpu.numel()),
        "total_pairs": int(counts_cpu.sum().item()) if counts_cpu.numel() else 0,
        "mean_pairs_per_tile": float(counts_cpu.mean().item()) if counts_cpu.numel() else 0.0,
        "p50_pairs_per_tile": _q(counts_cpu.tolist(), 0.50),
        "p95_pairs_per_tile": _q(counts_cpu.tolist(), 0.95),
        "max_pairs_per_tile": int(counts_cpu.max().item()) if counts_cpu.numel() else 0,
        "overflow_tile_count": int((counts_cpu > int(cfg.max_fast_pairs)).sum().item()) if counts_cpu.numel() else 0,
        "contributors": contributor_stats,
    }
    if args.json:
        print(json.dumps(result, sort_keys=True))
    else:
        print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

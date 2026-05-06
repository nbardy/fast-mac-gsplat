from __future__ import annotations

import argparse
import gc
import json
import sys
import threading
import time
from pathlib import Path
from statistics import mean, median
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from torch_gsplat_bridge_v6_feature_lookup_experiment import (
    RasterConfig,
    rasterize_projected_gaussians,
    rasterize_projected_gaussians_feature_lookup,
)


def _device() -> torch.device:
    if not torch.backends.mps.is_available():
        raise SystemExit("MPS is not available")
    return torch.device("mps")


def _sync() -> None:
    torch.mps.synchronize()


def _clear_mps() -> None:
    gc.collect()
    torch.mps.empty_cache()
    _sync()


def _memory_stats() -> dict[str, int]:
    return {
        "current_allocated_bytes": int(torch.mps.current_allocated_memory()),
        "driver_allocated_bytes": int(torch.mps.driver_allocated_memory()),
    }


class MemorySampler:
    def __init__(self, interval_s: float):
        self.interval_s = float(interval_s)
        self.stop_event = threading.Event()
        self.max_current_allocated_bytes = 0
        self.max_driver_allocated_bytes = 0
        self.samples = 0
        self.thread: threading.Thread | None = None

    def _sample_once(self) -> None:
        stats = _memory_stats()
        self.max_current_allocated_bytes = max(
            self.max_current_allocated_bytes,
            stats["current_allocated_bytes"],
        )
        self.max_driver_allocated_bytes = max(
            self.max_driver_allocated_bytes,
            stats["driver_allocated_bytes"],
        )
        self.samples += 1

    def _run(self) -> None:
        while not self.stop_event.is_set():
            self._sample_once()
            self.stop_event.wait(self.interval_s)
        self._sample_once()

    def __enter__(self) -> "MemorySampler":
        self._sample_once()
        self.thread = threading.Thread(target=self._run, name="mps-memory-sampler", daemon=True)
        self.thread.start()
        return self

    def __exit__(self, _exc_type, _exc, _tb) -> None:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join()
        self._sample_once()

    def stats(self) -> dict[str, int]:
        return {
            "sampled_peak_current_allocated_bytes": int(self.max_current_allocated_bytes),
            "sampled_peak_driver_allocated_bytes": int(self.max_driver_allocated_bytes),
            "memory_sample_count": int(self.samples),
        }


def _make_case(*, batch: int, gaussians: int, compact_dim: int, feature_dim: int, height: int, width: int, seed: int):
    torch.manual_seed(seed)
    device = _device()
    means = torch.rand(batch, gaussians, 2, device=device) * float(min(height, width) - 12) + 6.0
    sigmas = torch.rand(batch, gaussians, 2, device=device) * 3.0 + 2.0
    conics = torch.stack(
        [
            1.0 / sigmas[..., 0].square(),
            torch.zeros(batch, gaussians, device=device),
            1.0 / sigmas[..., 1].square(),
        ],
        dim=-1,
    )
    weights = torch.randn(batch, gaussians, compact_dim, device=device) * 0.25
    lookup = torch.randn(compact_dim, feature_dim, device=device) * 0.5
    opacities = torch.rand(batch, gaussians, device=device) * 0.35 + 0.35
    depths = torch.rand(batch, gaussians, device=device)
    return means, conics, weights, lookup, opacities, depths


def _clone_leaf(value: torch.Tensor) -> torch.Tensor:
    return value.detach().clone().requires_grad_(True)


def _time_one(
    *,
    mode: str,
    means: torch.Tensor,
    conics: torch.Tensor,
    weights: torch.Tensor,
    lookup: torch.Tensor,
    opacities: torch.Tensor,
    depths: torch.Tensor,
    config: RasterConfig,
    memory_sample_interval_ms: float,
) -> tuple[float, dict[str, int], float]:
    _clear_mps()
    means_i = _clone_leaf(means)
    conics_i = _clone_leaf(conics)
    weights_i = _clone_leaf(weights)
    lookup_i = _clone_leaf(lookup)
    opacities_i = _clone_leaf(opacities)

    _sync()
    sampler = MemorySampler(memory_sample_interval_ms / 1000.0)
    with sampler:
        t0 = time.perf_counter()
        if mode == "direct":
            full_features = weights_i @ lookup_i
            features, alpha = rasterize_projected_gaussians(
                means_i,
                conics_i,
                full_features,
                opacities_i,
                depths,
                config,
            )
        elif mode == "lookup":
            result = rasterize_projected_gaussians_feature_lookup(
                means_i,
                conics_i,
                weights_i,
                lookup_i,
                opacities_i,
                depths,
                config,
            )
            features = result.features
            alpha = result.alpha
        else:
            raise ValueError(f"unsupported mode: {mode}")
        loss = features.square().mean() + 0.17 * alpha.square().mean()
        loss.backward()
        _sync()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
    mem = {**_memory_stats(), **sampler.stats()}
    loss_value = float(loss.detach().cpu())

    del means_i, conics_i, weights_i, lookup_i, opacities_i, features, alpha, loss
    if mode == "direct":
        del full_features
    _clear_mps()
    return elapsed_ms, mem, loss_value


def _summarize(times: list[float], memories: list[dict[str, int]]) -> dict[str, Any]:
    return {
        "mean_ms": float(mean(times)),
        "median_ms": float(median(times)),
        "min_ms": float(min(times)),
        "max_ms": float(max(times)),
        "max_current_allocated_bytes": int(max(m["current_allocated_bytes"] for m in memories)),
        "max_driver_allocated_bytes": int(max(m["driver_allocated_bytes"] for m in memories)),
        "sampled_peak_current_allocated_bytes": int(
            max(m["sampled_peak_current_allocated_bytes"] for m in memories)
        ),
        "sampled_peak_driver_allocated_bytes": int(max(m["sampled_peak_driver_allocated_bytes"] for m in memories)),
        "memory_sample_count_total": int(sum(m["memory_sample_count"] for m in memories)),
    }


def run_case(args: argparse.Namespace, compact_dim: int, mode: str) -> dict[str, Any]:
    means, conics, weights, lookup, opacities, depths = _make_case(
        batch=args.batch_size,
        gaussians=args.gaussians,
        compact_dim=compact_dim,
        feature_dim=args.feature_dim,
        height=args.height,
        width=args.width,
        seed=args.seed + compact_dim,
    )
    config = RasterConfig(
        height=args.height,
        width=args.width,
        max_fast_pairs=args.max_fast_pairs,
        background=(0.0,) * args.feature_dim,
        active_policy=args.active_policy,
        batch_strategy=args.batch_strategy,
        enable_overflow_fallback=not args.no_overflow_fallback,
        inputs_sorted_by_depth=False,
    )

    for _ in range(args.warmup):
        _time_one(
            mode=mode,
            means=means,
            conics=conics,
            weights=weights,
            lookup=lookup,
            opacities=opacities,
            depths=depths,
            config=config,
            memory_sample_interval_ms=args.memory_sample_interval_ms,
        )

    times: list[float] = []
    memories: list[dict[str, int]] = []
    losses: list[float] = []
    for _ in range(args.iters):
        elapsed_ms, mem, loss_value = _time_one(
            mode=mode,
            means=means,
            conics=conics,
            weights=weights,
            lookup=lookup,
            opacities=opacities,
            depths=depths,
            config=config,
            memory_sample_interval_ms=args.memory_sample_interval_ms,
        )
        times.append(elapsed_ms)
        memories.append(mem)
        losses.append(loss_value)

    return {
        "mode": mode,
        "height": int(args.height),
        "width": int(args.width),
        "batch_size": int(args.batch_size),
        "gaussians": int(args.gaussians),
        "compact_dim": int(compact_dim),
        "feature_dim": int(args.feature_dim),
        "warmup": int(args.warmup),
        "iters": int(args.iters),
        "active_policy": str(args.active_policy),
        "batch_strategy": str(args.batch_strategy),
        "max_fast_pairs": int(args.max_fast_pairs),
        "loss_first": float(losses[0]),
        **_summarize(times, memories),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark compact-basis lookup rasterization against direct F-channel rasterization.")
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gaussians", type=int, default=2048)
    parser.add_argument("--feature-dim", type=int, default=32)
    parser.add_argument("--compact-dims", type=str, default="4,8,16")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-fast-pairs", type=int, default=2048)
    parser.add_argument("--active-policy", choices=("off", "on", "auto"), default="off")
    parser.add_argument("--batch-strategy", choices=("auto", "flatten", "serial"), default="flatten")
    parser.add_argument("--no-overflow-fallback", action="store_true")
    parser.add_argument("--memory-sample-interval-ms", type=float, default=0.5)
    parser.add_argument("--jsonl-output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    compact_dims = [int(item) for item in args.compact_dims.split(",") if item.strip()]
    rows = []
    for compact_dim in compact_dims:
        for mode in ("direct", "lookup"):
            row = run_case(args, compact_dim, mode)
            rows.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
    if args.jsonl_output is not None:
        args.jsonl_output.parent.mkdir(parents=True, exist_ok=True)
        with args.jsonl_output.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()

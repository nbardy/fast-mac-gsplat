from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from statistics import median

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from torch_gsplat_bridge_v6 import RasterConfig, get_runtime_shader_config, rasterize_projected_gaussians  # noqa: E402
from benchmarks.benchmark_mps import make_case  # noqa: E402


def clear_grads(xs):
    for x in xs:
        if x.grad is not None:
            x.grad.zero_()


def sync():
    torch.mps.synchronize()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--height", type=int, default=4096)
    p.add_argument("--width", type=int, default=4096)
    p.add_argument("--gaussians", type=int, default=65536)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--case", type=str, default="uniform_random")
    p.add_argument("--trace-file", type=str, default="")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--iters", type=int, default=7)
    p.add_argument("--backward", action="store_true")
    p.add_argument("--shuffle-order", action="store_true")
    args = p.parse_args()

    if not torch.backends.mps.is_available():
        raise SystemExit("MPS is not available.")
    device = torch.device("mps")
    rt = get_runtime_shader_config()

    base = make_case(args.case, args.batch_size, args.gaussians, args.height, args.width, device, args.seed, args.trace_file)

    names = ["direct_off", "policy_auto", "active_on"]
    inputs_by_name = {}
    for name in names:
        xs = [t.clone().contiguous() for t in base]
        if args.backward:
            for t in xs[:-1]:
                t.requires_grad_(True)
        inputs_by_name[name] = tuple(xs)

    cfgs = {
        "direct_off": RasterConfig(
            height=args.height,
            width=args.width,
            tile_size=rt.tile_size,
            max_fast_pairs=rt.fast_cap,
            active_policy="off",
            use_active_tiles=None,
        ),
        "policy_auto": RasterConfig(
            height=args.height,
            width=args.width,
            tile_size=rt.tile_size,
            max_fast_pairs=rt.fast_cap,
            active_policy="auto",
            use_active_tiles=None,
        ),
        "active_on": RasterConfig(
            height=args.height,
            width=args.width,
            tile_size=rt.tile_size,
            max_fast_pairs=rt.fast_cap,
            active_policy="on",
            use_active_tiles=None,
        ),
    }

    def run_one(name):
        m, q, c, o, d = inputs_by_name[name]
        cfg = cfgs[name]
        if args.backward:
            t0 = time.perf_counter()
            out = rasterize_projected_gaussians(m, q, c, o, d, cfg)
            sync()
            t1 = time.perf_counter()
            out.square().mean().backward()
            sync()
            t2 = time.perf_counter()
            clear_grads((m, q, c, o))
            return (t2 - t0) * 1000.0, (t1 - t0) * 1000.0, (t2 - t1) * 1000.0
        t0 = time.perf_counter()
        _ = rasterize_projected_gaussians(m, q, c, o, d, cfg)
        sync()
        t1 = time.perf_counter()
        return (t1 - t0) * 1000.0, (t1 - t0) * 1000.0, 0.0

    for name in names:
        for _ in range(args.warmup):
            run_one(name)

    rows = defaultdict(list)
    order = names[:]
    for i in range(args.iters):
        if args.shuffle_order:
            random.Random(1000 + i).shuffle(order)
        for name in order:
            rows[name].append(run_one(name))

    result = {}
    for name in names:
        total = [x[0] for x in rows[name]]
        fwd = [x[1] for x in rows[name]]
        bwd = [x[2] for x in rows[name]]
        result[name] = {
            "total_mean": sum(total) / len(total),
            "total_median": float(median(total)),
            "forward_mean": sum(fwd) / len(fwd),
            "backward_mean": sum(bwd) / len(bwd),
            "min": min(total),
            "max": max(total),
        }

    print(json.dumps({
        "height": args.height,
        "width": args.width,
        "batch_size": args.batch_size,
        "gaussians": args.gaussians,
        "case": args.case,
        "backward": bool(args.backward),
        "results": result,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

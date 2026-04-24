
from __future__ import annotations

import argparse
import itertools
import json
import os
import random
import subprocess
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--height", type=int, default=4096)
    p.add_argument("--width", type=int, default=4096)
    p.add_argument("--gaussians", type=int, default=65536)
    p.add_argument("--batch-sizes", type=str, default="1,2,4,8")
    p.add_argument("--strategies", type=str, default="auto,flatten,serial")
    p.add_argument("--tile-sizes", type=str, default="16")
    p.add_argument("--chunks", type=str, default="64")
    p.add_argument("--caps", type=str, default="2048")
    p.add_argument("--stop-count-modes", type=str, default="adaptive,never,always")
    p.add_argument("--dense-thresholds", type=str, default="32,64,128")
    p.add_argument("--active-policies", type=str, default="off,auto,on")
    p.add_argument("--max-pairs-per-launches", type=str, default="0")
    p.add_argument("--trace-file", type=str, default="")
    p.add_argument("--active-tile-modes", type=str, default="")
    p.add_argument("--sort-active-tile-modes", type=str, default="on")
    p.add_argument("--seeds", type=str, default="0,1,2")
    p.add_argument("--cases", type=str, default="microbench_uniform_random,sparse_screen,clustered_hot_tiles,layered_depth,overflow_adversarial")
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--iters", type=int, default=3)
    p.add_argument("--backward", action="store_true")
    p.add_argument("--shuffle-order", action="store_true")
    args = p.parse_args()

    batch_sizes = [int(x) for x in args.batch_sizes.split(",") if x]
    strategies = [x for x in args.strategies.split(",") if x]
    tile_sizes = [int(x) for x in args.tile_sizes.split(",") if x]
    chunks = [int(x) for x in args.chunks.split(",") if x]
    caps = [int(x) for x in args.caps.split(",") if x]
    stop_modes = [x for x in args.stop_count_modes.split(",") if x]
    dense_thresholds = [int(x) for x in args.dense_thresholds.split(",") if x]
    if args.active_tile_modes:
        active_policies = ["on" if x in ("on", "true", "1", "yes") else "off" for x in args.active_tile_modes.split(",") if x]
    else:
        active_policies = [x for x in args.active_policies.split(",") if x]
    max_pairs_per_launches = [int(x) for x in args.max_pairs_per_launches.split(",") if x]
    sort_active_tile_modes = [x for x in args.sort_active_tile_modes.split(",") if x]
    seeds = [int(x) for x in args.seeds.split(",") if x]
    cases = [x for x in args.cases.split(",") if x]

    bench = Path(__file__).resolve().parent / "benchmark_mps.py"
    jobs = list(
        itertools.product(
            cases,
            seeds,
            batch_sizes,
            strategies,
            tile_sizes,
            chunks,
            caps,
            stop_modes,
            dense_thresholds,
            active_policies,
            max_pairs_per_launches,
            sort_active_tile_modes,
        )
    )
    if args.shuffle_order:
        random.Random(0).shuffle(jobs)

    rows = []
    for case, seed, batch_size, strategy, tile, chunk, cap, stop_mode, dense_threshold, active_policy, max_pairs_per_launch, sort_active_tile_mode in jobs:
        env = os.environ.copy()
        env["GSP_TILE_SIZE"] = str(tile)
        env["GSP_CHUNK"] = str(chunk)
        env["GSP_FAST_CAP"] = str(cap)
        cmd = [
            sys.executable,
            str(bench),
            "--height", str(args.height),
            "--width", str(args.width),
            "--gaussians", str(args.gaussians),
            "--batch-size", str(batch_size),
            "--batch-strategy", strategy,
            "--warmup", str(args.warmup),
            "--iters", str(args.iters),
            "--case", case,
            "--seed", str(seed),
            "--stop-count-mode", stop_mode,
            "--dense-threshold", str(dense_threshold),
            "--active-policy", active_policy,
            "--max-pairs-per-launch", str(max_pairs_per_launch),
            "--profile",
            "--json",
        ]
        if args.trace_file:
            cmd.extend(["--trace-file", args.trace_file])
        if sort_active_tile_mode in ("off", "false", "0", "no"):
            cmd.append("--no-sort-active-tiles")
        if args.backward:
            cmd.append("--backward")
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True, check=True)
        line = proc.stdout.strip().splitlines()[-1]
        row = json.loads(line)
        rows.append(row)
        print(
            f"case={case:16s} seed={seed} B={batch_size} strat={strategy:7s} stop={stop_mode:8s} dense={dense_threshold:3d} "
            f"policy={active_policy:4s} pairs_launch={max_pairs_per_launch:8d} sort_active={sort_active_tile_mode:3s} "
            f"tile={tile:2d} chunk={chunk:3d} cap={cap:4d} "
            f"mean={row['mean_ms']:8.3f} med={row['median_ms']:8.3f} fwd={row.get('forward_ms', 0.0):8.3f} "
            f"bwd={row.get('backward_ms', 0.0):8.3f} active={row.get('profile_active_tile_count', 0):6d} "
            f"dense_active={row.get('profile_dense_active_tile_count', 0):6d} overflow={row.get('profile_overflow_tile_count', 0)}"
        )

    print("\nJSON rows:")
    print(json.dumps(rows, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

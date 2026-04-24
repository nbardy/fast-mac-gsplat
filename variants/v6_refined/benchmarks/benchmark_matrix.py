from __future__ import annotations

import argparse
import itertools
import json
import os
import random
import subprocess
import sys
from pathlib import Path


def _csv_ints(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x]


def _csv_strs(s: str) -> list[str]:
    return [x for x in s.split(",") if x]


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
    p.add_argument("--seeds", type=str, default="0,1,2")
    p.add_argument(
        "--cases",
        type=str,
        default="uniform_random,sparse_screen,clustered_hot_tiles,layered_depth,overflow_adversarial,temporal_adjacent",
    )
    p.add_argument("--active-policies", type=str, default="off,auto,on")
    p.add_argument("--active-overrides", type=str, default="none")
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--iters", type=int, default=3)
    p.add_argument("--backward", action="store_true")
    p.add_argument("--shuffle-order", action="store_true")
    p.add_argument("--trace-file", type=str, default="")
    args = p.parse_args()

    batch_sizes = _csv_ints(args.batch_sizes)
    strategies = _csv_strs(args.strategies)
    tile_sizes = _csv_ints(args.tile_sizes)
    chunks = _csv_ints(args.chunks)
    caps = _csv_ints(args.caps)
    stop_modes = _csv_strs(args.stop_count_modes)
    dense_thresholds = _csv_ints(args.dense_thresholds)
    seeds = _csv_ints(args.seeds)
    cases = _csv_strs(args.cases)
    active_policies = _csv_strs(args.active_policies)
    active_overrides = _csv_strs(args.active_overrides)

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
            active_overrides,
        )
    )
    if args.shuffle_order:
        random.Random(0).shuffle(jobs)

    rows = []
    for case, seed, batch_size, strategy, tile, chunk, cap, stop_mode, dense_threshold, active_policy, active_override in jobs:
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
            "--profile",
            "--json",
        ]
        if args.trace_file:
            cmd += ["--trace-file", args.trace_file]
        if active_override != "none":
            if active_override == "on":
                cmd.append("--active-tiles")
            elif active_override == "off":
                cmd.append("--no-active-tiles")
            else:
                raise ValueError(f"unknown active override: {active_override}")
        if args.backward:
            cmd.append("--backward")
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True, check=True)
        line = proc.stdout.strip().splitlines()[-1]
        row = json.loads(line)
        rows.append(row)
        print(
            f"case={case:18s} seed={seed} B={batch_size} strat={strategy:7s} stop={stop_mode:8s} dense={dense_threshold:3d} "
            f"tile={tile:2d} chunk={chunk:3d} cap={cap:4d} active_policy={active_policy:4s} active_override={active_override:4s} "
            f"mean={row['mean_ms']:8.3f} med={row['median_ms']:8.3f} fwd={row.get('forward_ms', 0.0):8.3f} "
            f"bwd={row.get('backward_ms', 0.0):8.3f} active_frac={row.get('profile_mean_active_tile_fraction', 0.0):6.3f} "
            f"overflow={row.get('profile_overflow_tile_count', 0)} reason={row.get('profile_selected_active_reason', '')}"
        )

    print("\nJSON rows:")
    print(json.dumps(rows, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

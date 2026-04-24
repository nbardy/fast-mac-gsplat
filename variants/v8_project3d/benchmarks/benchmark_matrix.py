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
    p.add_argument("--batch-sizes", type=str, default="1,2,4")
    p.add_argument("--strategies", type=str, default="auto,flatten,serial")
    p.add_argument("--tile-sizes", type=str, default="8,16,32")
    p.add_argument("--chunks", type=str, default="32,64,128")
    p.add_argument("--caps", type=str, default="1024,2048")
    p.add_argument("--seeds", type=str, default="0,1,2")
    p.add_argument("--cases", type=str, default="medium_sigma_3_8,overflow_stress")
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
    seeds = [int(x) for x in args.seeds.split(",") if x]
    cases = [x for x in args.cases.split(",") if x]

    bench = Path(__file__).resolve().parent / "benchmark_mps.py"
    jobs = list(itertools.product(cases, seeds, batch_sizes, strategies, tile_sizes, chunks, caps))
    if args.shuffle_order:
        random.Random(0).shuffle(jobs)

    rows = []
    for case, seed, batch_size, strategy, tile, chunk, cap in jobs:
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
            "--profile",
            "--json",
        ]
        if args.backward:
            cmd.append("--backward")
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True, check=True)
        line = proc.stdout.strip().splitlines()[-1]
        row = json.loads(line)
        rows.append(row)
        print(
            f"case={case:16s} seed={seed} B={batch_size} strat={strategy:7s} "
            f"tile={tile:2d} chunk={chunk:3d} cap={cap:4d} "
            f"mean={row['mean_ms']:8.3f} fwd={row.get('forward_ms', 0.0):8.3f} "
            f"bwd={row.get('backward_ms', 0.0):8.3f} "
            f"p95_pairs={row.get('profile_p95_pairs_per_tile', 0.0):8.2f} "
            f"overflow={row.get('profile_overflow_tile_count', 0)} "
            f"chunk_sel={row.get('profile_chosen_batch_chunk', 0)}"
        )

    print("\nJSON rows:")
    print(json.dumps(rows, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

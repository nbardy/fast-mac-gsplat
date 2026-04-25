from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from torch_gsplat_bridge_v9_hw_tile_exact import probe_hw_interop, render_constant_rgba, render_constant_rgba_direct


def parse_size(raw: str) -> tuple[int, int]:
    h, w = raw.lower().split("x", 1)
    return int(h), int(w)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", default="64x64,512x512,1080x1920")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--paths", default="blit,direct")
    parser.add_argument("--jsonl", type=Path)
    args = parser.parse_args()

    if not torch.backends.mps.is_available():
        raise SystemExit("MPS is not available.")

    status = probe_hw_interop(compile_pipelines=True, compile_advanced=True, run_render_probe=True)
    if not status.render_to_mps_tensor_validated:
        raise SystemExit(f"render interop probe did not validate: {status.as_dict()}")

    rows = []
    path_fns = {
        "blit": ("render_rgba32f_private_texture_gpu_blit_to_torch_mps_buffer", render_constant_rgba),
        "direct": ("render_rgba32f_direct_to_torch_mps_buffer_backed_texture", render_constant_rgba_direct),
    }
    for path_name in [p.strip() for p in args.paths.split(",") if p.strip()]:
        if path_name not in path_fns:
            raise SystemExit(f"unknown path {path_name!r}; expected one of {sorted(path_fns)}")
        path_label, fn = path_fns[path_name]
        for raw_size in args.sizes.split(","):
            height, width = parse_size(raw_size.strip())
            if path_name == "direct" and (width * 16) % 256 != 0:
                print(json.dumps({
                    "height": height,
                    "width": width,
                    "path": path_label,
                    "status": "skipped",
                    "reason": "width*16 is not 256-byte aligned for buffer-backed texture rows",
                }, sort_keys=True))
                continue
            for _ in range(args.warmup):
                fn(height, width)
            torch.mps.synchronize()

            samples = []
            for _ in range(args.iters):
                t0 = time.perf_counter()
                out = fn(height, width)
                torch.mps.synchronize()
                samples.append((time.perf_counter() - t0) * 1000.0)
                del out

            row = {
                "height": height,
                "width": width,
                "pixels": height * width,
                "iters": args.iters,
                "warmup": args.warmup,
                "min_ms": min(samples),
                "median_ms": statistics.median(samples),
                "mean_ms": statistics.fmean(samples),
                "max_ms": max(samples),
                "path": path_label,
            }
            rows.append(row)
            print(json.dumps(row, sort_keys=True))

    if args.jsonl is not None:
        args.jsonl.parent.mkdir(parents=True, exist_ok=True)
        with args.jsonl.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()

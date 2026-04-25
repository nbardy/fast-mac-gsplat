from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from torch_gsplat_bridge_v9_hw_eval_parity import (
    probe_hw_interop,
    render_constant_rgba,
    render_constant_rgba_direct,
    render_gaussian_eval_rgba,
)


def parse_size(raw: str) -> tuple[int, int]:
    h, w = raw.lower().split("x", 1)
    return int(h), int(w)


def make_gaussian_inputs(height: int, width: int, count: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if count <= 0:
        raise ValueError("gaussians must be positive")
    xs = torch.linspace(width * 0.25, width * 0.75, count, dtype=torch.float32)
    ys = torch.linspace(height * 0.25, height * 0.75, count, dtype=torch.float32)
    means = torch.stack((xs, ys), dim=1).to("mps")
    conics = torch.tensor([[1.0 / 16.0, 0.0, 1.0 / 16.0]], dtype=torch.float32).repeat(count, 1).to("mps")
    ramp = torch.linspace(0.25, 0.85, count, dtype=torch.float32)
    colors = torch.stack((ramp, 1.0 - ramp * 0.5, torch.full_like(ramp, 0.35)), dim=1).to("mps")
    opacities = torch.full((count,), 0.75, dtype=torch.float32, device="mps")
    return means, conics, colors, opacities


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", default="64x64,512x512,1080x1920")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--paths", default="blit,direct,gaussian-direct")
    parser.add_argument("--gaussians", type=int, default=1)
    parser.add_argument("--jsonl", type=Path)
    args = parser.parse_args()

    if not torch.backends.mps.is_available():
        raise SystemExit("MPS is not available.")

    status = probe_hw_interop(compile_pipelines=True, compile_advanced=True, run_render_probe=True)
    if not status.render_to_mps_tensor_validated:
        raise SystemExit(f"render interop probe did not validate: {status.as_dict()}")

    rows = []
    path_fns = {
        "blit": "render_rgba32f_private_texture_gpu_blit_to_torch_mps_buffer",
        "direct": "render_rgba32f_direct_to_torch_mps_buffer_backed_texture",
        "gaussian-blit": "gaussian_eval_rgba32f_private_texture_gpu_blit_to_torch_mps_buffer",
        "gaussian-direct": "gaussian_eval_rgba32f_direct_to_torch_mps_buffer_backed_texture",
    }
    for path_name in [p.strip() for p in args.paths.split(",") if p.strip()]:
        if path_name not in path_fns:
            raise SystemExit(f"unknown path {path_name!r}; expected one of {sorted(path_fns)}")
        path_label = path_fns[path_name]
        if path_name.startswith("gaussian") and not status.gaussian_eval_rgba_validated:
            raise SystemExit(f"Gaussian eval probe did not validate: {status.as_dict()}")
        for raw_size in args.sizes.split(","):
            height, width = parse_size(raw_size.strip())
            if path_name in {"direct", "gaussian-direct"} and (width * 16) % 256 != 0:
                print(json.dumps({
                    "height": height,
                    "width": width,
                    "path": path_label,
                    "status": "skipped",
                    "reason": "width*16 is not 256-byte aligned for buffer-backed texture rows",
                }, sort_keys=True))
                continue
            if path_name == "blit":
                fn = render_constant_rgba
            elif path_name == "direct":
                fn = render_constant_rgba_direct
            else:
                gaussian_inputs = make_gaussian_inputs(height, width, args.gaussians)
                direct = path_name == "gaussian-direct"

                def fn(h: int, w: int, inputs=gaussian_inputs, direct=direct) -> torch.Tensor:
                    return render_gaussian_eval_rgba(*inputs, h, w, direct=direct)

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
                "gaussians": args.gaussians if path_name.startswith("gaussian") else 0,
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

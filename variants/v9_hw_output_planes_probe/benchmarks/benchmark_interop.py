from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from torch_gsplat_bridge_v9_hw_output_planes import (
    DIRECT_OUTPUT_FORMATS,
    GAUSSIAN_OUTPUT_FORMATS,
    direct_width_multiple,
    probe_hw_interop,
    render_constant_direct_format,
    render_constant_rgba,
    render_constant_rgba_direct,
    render_gaussian_eval_format,
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
    parser.add_argument("--paths", default="blit,direct,formats,gaussian-direct-rgba32f,gaussian-direct-rgba16f")
    parser.add_argument("--formats", default="rgba32f,rgba16f,r32f,rg32f")
    parser.add_argument("--gaussian-formats", default="rgba32f,rgba16f")
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
        "blit": (
            "render_rgba32f_private_texture_gpu_blit_to_torch_mps_buffer",
            "rgba32f",
            None,
            render_constant_rgba,
        ),
        "direct": (
            "render_rgba32f_direct_to_torch_mps_buffer_backed_texture",
            "rgba32f",
            direct_width_multiple("rgba32f"),
            render_constant_rgba_direct,
        ),
    }
    for path_name in [p.strip() for p in args.paths.split(",") if p.strip()]:
        if path_name == "gaussian-direct":
            path_name = "gaussian-direct-rgba32f"
        if path_name == "gaussian-blit":
            path_name = "gaussian-blit-rgba32f"
        if path_name not in path_fns and path_name not in {"formats", "gaussian-direct-formats", "gaussian-blit-formats"}:
            if not (path_name.startswith("gaussian-direct-") or path_name.startswith("gaussian-blit-")):
                expected = sorted(path_fns) + [
                    "formats",
                    "gaussian-direct",
                    "gaussian-blit",
                    "gaussian-direct-rgba32f",
                    "gaussian-direct-rgba16f",
                    "gaussian-direct-formats",
                    "gaussian-blit-rgba32f",
                    "gaussian-blit-rgba16f",
                    "gaussian-blit-formats",
                ]
                raise SystemExit(f"unknown path {path_name!r}; expected one of {expected}")

        work_items = []
        if path_name == "formats":
            for output_format in [f.strip() for f in args.formats.split(",") if f.strip()]:
                if output_format not in DIRECT_OUTPUT_FORMATS:
                    raise SystemExit(f"unknown format {output_format!r}; expected one of {DIRECT_OUTPUT_FORMATS}")
                work_items.append(
                    (
                        f"render_{output_format}_direct_to_torch_mps_buffer_backed_texture",
                        output_format,
                        direct_width_multiple(output_format),
                        lambda h, w, fmt=output_format: render_constant_direct_format(fmt, h, w),
                        0,
                    )
                )
        elif path_name in {"gaussian-direct-formats", "gaussian-blit-formats"}:
            direct = path_name == "gaussian-direct-formats"
            for output_format in [f.strip() for f in args.gaussian_formats.split(",") if f.strip()]:
                if output_format not in GAUSSIAN_OUTPUT_FORMATS:
                    raise SystemExit(f"unknown Gaussian format {output_format!r}; expected one of {GAUSSIAN_OUTPUT_FORMATS}")
                width_multiple = direct_width_multiple(output_format) if direct else None
                label_mode = "direct_to_torch_mps_buffer_backed_texture" if direct else "private_texture_gpu_blit_to_torch_mps_buffer"
                work_items.append(
                    (
                        f"gaussian_eval_{output_format}_{label_mode}",
                        output_format,
                        width_multiple,
                        None,
                        args.gaussians,
                    )
                )
        elif path_name.startswith("gaussian-direct-") or path_name.startswith("gaussian-blit-"):
            direct = path_name.startswith("gaussian-direct-")
            output_format = path_name.rsplit("-", 1)[1]
            if output_format not in GAUSSIAN_OUTPUT_FORMATS:
                raise SystemExit(f"unknown Gaussian format {output_format!r}; expected one of {GAUSSIAN_OUTPUT_FORMATS}")
            width_multiple = direct_width_multiple(output_format) if direct else None
            label_mode = "direct_to_torch_mps_buffer_backed_texture" if direct else "private_texture_gpu_blit_to_torch_mps_buffer"
            work_items.append(
                (
                    f"gaussian_eval_{output_format}_{label_mode}",
                    output_format,
                    width_multiple,
                    None,
                    args.gaussians,
                )
            )
        else:
            path_label, output_format, width_multiple, fn = path_fns[path_name]
            work_items.append((path_label, output_format, width_multiple, fn, 0))

        for path_label, output_format, width_multiple, fn_template, gaussian_count in work_items:
            if gaussian_count and not status.gaussian_eval_rgba_validated:
                raise SystemExit(f"Gaussian eval probe did not validate: {status.as_dict()}")
            if output_format == "rgba16f" and gaussian_count and not status.gaussian_eval_rgba16_validated:
                raise SystemExit(f"Gaussian RGBA16F eval probe did not validate: {status.as_dict()}")
            for raw_size in args.sizes.split(","):
                height, width = parse_size(raw_size.strip())
                if width_multiple is not None and width % width_multiple != 0:
                    print(
                        json.dumps(
                            {
                                "height": height,
                                "width": width,
                                "output_format": output_format,
                                "path": path_label,
                                "status": "skipped",
                                "reason": (
                                    f"width is not a multiple of {width_multiple} "
                                    "for 256-byte buffer-backed texture rows"
                                ),
                            },
                            sort_keys=True,
                        )
                    )
                    continue
                if gaussian_count:
                    gaussian_inputs = make_gaussian_inputs(height, width, gaussian_count)
                    direct = "direct_to_torch" in path_label

                    def fn(h: int, w: int, inputs=gaussian_inputs, fmt=output_format, direct=direct) -> torch.Tensor:
                        return render_gaussian_eval_format(fmt, *inputs, h, w, direct=direct)

                else:
                    fn = fn_template

                assert fn is not None
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
                    "gaussians": gaussian_count,
                    "min_ms": min(samples),
                    "median_ms": statistics.median(samples),
                    "mean_ms": statistics.fmean(samples),
                    "max_ms": max(samples),
                    "path": path_label,
                    "output_format": output_format,
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

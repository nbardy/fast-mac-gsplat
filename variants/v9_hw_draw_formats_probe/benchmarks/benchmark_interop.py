from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from torch_gsplat_bridge_v9_hw_draw_formats import (
    DIRECT_OUTPUT_FORMATS,
    direct_width_multiple,
    probe_hw_interop,
    render_constant_direct_format,
    render_constant_rgba,
    render_constant_rgba_direct,
)


def parse_size(raw: str) -> tuple[int, int]:
    h, w = raw.lower().split("x", 1)
    return int(h), int(w)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", default="64x64,512x512,1080x1920")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--paths", default="blit,direct,formats")
    parser.add_argument("--formats", default="rgba32f,rgba16f,r32f,rg32f")
    parser.add_argument("--jsonl", type=Path)
    args = parser.parse_args()

    if not torch.backends.mps.is_available():
        raise SystemExit("MPS is not available.")

    status = probe_hw_interop(compile_pipelines=True, compile_advanced=True, run_render_probe=True)
    if not status.render_to_mps_tensor_validated:
        raise SystemExit(f"render interop probe did not validate: {status.as_dict()}")

    rows = []
    path_fns = {
        "blit": ("render_rgba32f_private_texture_gpu_blit_to_torch_mps_buffer", "rgba32f", None, render_constant_rgba),
        "direct": (
            "render_rgba32f_direct_to_torch_mps_buffer_backed_texture",
            "rgba32f",
            direct_width_multiple("rgba32f"),
            render_constant_rgba_direct,
        ),
    }
    for path_name in [p.strip() for p in args.paths.split(",") if p.strip()]:
        if path_name == "icb":
            print(
                json.dumps(
                    {
                        "path": "render_rgba32f_direct_to_torch_mps_buffer_backed_texture_icb_execute",
                        "status": "skipped",
                        "reason": "disabled: known AGX executeCommandsInBufferCommon crash risk",
                    },
                    sort_keys=True,
                )
            )
            continue
        if path_name not in path_fns and path_name != "formats":
            raise SystemExit(f"unknown path {path_name!r}; expected one of {sorted(path_fns) + ['formats', 'icb']}")
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
                    )
                )
        else:
            work_items.append(path_fns[path_name])

        for path_label, output_format, width_multiple, fn in work_items:
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

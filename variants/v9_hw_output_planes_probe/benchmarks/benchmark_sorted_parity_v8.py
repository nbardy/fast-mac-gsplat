from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Callable

import torch
from torch import Tensor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from torch_gsplat_bridge_v9_hw_output_planes import (  # noqa: E402
    GAUSSIAN_OUTPUT_FORMATS,
    direct_width_multiple,
    probe_hw_interop,
    render_gaussian_eval_format,
    render_gaussian_eval_format_sorted,
)


def _load_v8_api():
    v8_root = ROOT.parent / "v8_hw_eval"
    if str(v8_root) not in sys.path:
        sys.path.insert(0, str(v8_root))
    from torch_gsplat_bridge_v8_hw_eval import (  # noqa: PLC0415
        RasterConfig,
        get_runtime_shader_config,
        rasterize_projected_gaussians,
    )
    from torch_gsplat_bridge_v8_hw_eval.rasterize import _native_ops_registered  # noqa: PLC0415

    if not _native_ops_registered():
        raise RuntimeError(
            "v8_hw_eval native ops are not registered; run "
            "`cd variants/v8_hw_eval && python3 setup.py build_ext --inplace` "
            "before this benchmark"
        )

    return RasterConfig, get_runtime_shader_config, rasterize_projected_gaussians


def parse_size(raw: str) -> tuple[int, int]:
    h, w = raw.lower().split("x", 1)
    return int(h), int(w)


def parse_str_list(raw: str) -> tuple[str, ...]:
    return tuple(x.strip() for x in raw.split(",") if x.strip())


def parse_int_list(raw: str) -> tuple[int, ...]:
    return tuple(int(x.strip()) for x in raw.split(",") if x.strip())


def _sync_mps() -> None:
    if torch.backends.mps.is_available():
        torch.mps.synchronize()


def _time_ms(fn: Callable[[], Tensor], *, warmup: int, iters: int) -> tuple[dict[str, float], Tensor]:
    last = fn()
    _sync_mps()
    for _ in range(warmup):
        last = fn()
        _sync_mps()

    samples: list[float] = []
    for _ in range(iters):
        t0 = time.perf_counter()
        last = fn()
        _sync_mps()
        samples.append((time.perf_counter() - t0) * 1000.0)

    return (
        {
            "min_ms": float(min(samples)),
            "median_ms": float(statistics.median(samples)),
            "mean_ms": float(statistics.fmean(samples)),
            "max_ms": float(max(samples)),
        },
        last,
    )


def _make_overlap_stack(
    *,
    height: int,
    width: int,
    gaussians: int,
    device: torch.device | str = "mps",
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    if gaussians <= 0:
        raise ValueError("gaussians must be positive")
    idx = torch.arange(gaussians, dtype=torch.float32)
    means = torch.full((gaussians, 2), 0.0, dtype=torch.float32)
    means[:, 0] = width * 0.5 + 0.5
    means[:, 1] = height * 0.5 + 0.5
    sigma = 3.5 + idx.remainder(5) * 0.2
    conics = torch.stack((1.0 / sigma.square(), torch.zeros_like(sigma), 1.0 / sigma.square()), dim=1)
    denom = float(max(gaussians - 1, 1))
    ramp = idx / denom
    colors = torch.stack((0.15 + 0.75 * ramp, 0.80 - 0.45 * ramp, 0.20 + 0.55 * (1.0 - ramp)), dim=1)
    opacities = torch.full((gaussians,), 0.20, dtype=torch.float32)
    if gaussians == 2:
        colors = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float32)
        opacities = torch.tensor([0.5, 0.5], dtype=torch.float32)
        conics = torch.tensor([[1.0, 0.0, 1.0], [1.0, 0.0, 1.0]], dtype=torch.float32)
    depths = idx.contiguous()
    return (
        means.to(device=device, dtype=torch.float32).contiguous(),
        conics.to(device=device, dtype=torch.float32).contiguous(),
        colors.to(device=device, dtype=torch.float32).contiguous(),
        opacities.to(device=device, dtype=torch.float32).contiguous(),
        depths.to(device=device, dtype=torch.float32).contiguous(),
    )


def _render_v8(
    means2d: Tensor,
    conics: Tensor,
    colors: Tensor,
    opacities: Tensor,
    depths: Tensor,
    *,
    height: int,
    width: int,
) -> Tensor:
    RasterConfig, get_runtime_shader_config, rasterize_projected_gaussians = _load_v8_api()
    rt = get_runtime_shader_config()
    cfg = RasterConfig(
        height=int(height),
        width=int(width),
        tile_size=rt.tile_size,
        max_fast_pairs=rt.fast_cap,
        background=(0.0, 0.0, 0.0),
        enable_overflow_fallback=True,
        use_active_tiles=False,
        active_policy="off",
        stop_count_mode="adaptive",
        use_hardware_eval=False,
        hardware_eval_policy="off",
    )
    return rasterize_projected_gaussians(means2d, conics, colors, opacities, depths, cfg)


def _render_v9_order(
    order: str,
    output_format: str,
    means2d: Tensor,
    conics: Tensor,
    colors: Tensor,
    opacities: Tensor,
    depths: Tensor,
    *,
    height: int,
    width: int,
    direct: bool,
) -> Tensor:
    if order == "input":
        return render_gaussian_eval_format(
            output_format,
            means2d,
            conics,
            colors,
            opacities,
            int(height),
            int(width),
            direct=direct,
        )
    if order == "ascending":
        return render_gaussian_eval_format_sorted(
            output_format,
            means2d,
            conics,
            colors,
            opacities,
            depths,
            int(height),
            int(width),
            direct=direct,
            descending=False,
        )
    if order == "descending":
        return render_gaussian_eval_format_sorted(
            output_format,
            means2d,
            conics,
            colors,
            opacities,
            depths,
            int(height),
            int(width),
            direct=direct,
            descending=True,
        )
    raise ValueError("order must be input, ascending, or descending")


def _compare(v8_rgb: Tensor, v9_rgba: Tensor) -> dict[str, float]:
    v8_cpu = v8_rgb.detach().cpu().float()
    v9_cpu = v9_rgba.detach().cpu().float()
    rgb_diff = (v9_cpu[..., :3] - v8_cpu).abs()
    alpha = v9_cpu[..., 3]
    return {
        "rgb_max_abs_err": float(rgb_diff.max().item()),
        "rgb_mean_abs_err": float(rgb_diff.mean().item()),
        "v8_rgb_max": float(v8_cpu.max().item()),
        "v9_rgb_max": float(v9_cpu[..., :3].max().item()),
        "v9_alpha_max": float(alpha.max().item()),
        "v9_alpha_mean": float(alpha.mean().item()),
    }


def run_case(
    *,
    height: int,
    width: int,
    gaussians: int,
    output_format: str,
    order: str,
    warmup: int,
    iters: int,
    direct: bool,
) -> dict[str, object]:
    means2d, conics, colors, opacities, depths = _make_overlap_stack(
        height=int(height),
        width=int(width),
        gaussians=int(gaussians),
    )

    with torch.no_grad():
        v8_stats, v8_rgb = _time_ms(
            lambda: _render_v8(means2d, conics, colors, opacities, depths, height=height, width=width),
            warmup=warmup,
            iters=iters,
        )
        v9_stats, v9_rgba = _time_ms(
            lambda: _render_v9_order(
                order,
                output_format,
                means2d,
                conics,
                colors,
                opacities,
                depths,
                height=height,
                width=width,
                direct=direct,
            ),
            warmup=warmup,
            iters=iters,
        )
        compare = _compare(v8_rgb, v9_rgba)

    v9_med = v9_stats["median_ms"]
    return {
        "status": "ok",
        "case": "overlap_stack",
        "height": int(height),
        "width": int(width),
        "pixels": int(height) * int(width),
        "gaussians": int(gaussians),
        "output_format": output_format,
        "order": order,
        "direct": bool(direct),
        "warmup": int(warmup),
        "iters": int(iters),
        "v8_path": "v8_forward_eval_black_background",
        "v9_path": f"v9_output_planes_{output_format}_{order}",
        "order_note": (
            "descending is the reverse painter-order candidate for fixed source-over color parity; "
            "input and ascending are diagnostics"
        ),
        "v9_limitations": [
            "eval-only",
            "fixed-function source-over blending",
            "no final_T/stop_count/stopped-prefix state",
            "no backward",
            "black-background RGB comparison only",
        ],
        "validation_uses_cpu_readback": True,
        "native_op_uses_cpu_readback": False,
        "rgb_within_1e_5": bool(compare["rgb_max_abs_err"] <= 1.0e-5),
        "v8_over_v9_median_speedup": float(v8_stats["median_ms"] / v9_med) if v9_med > 0.0 else float("inf"),
        **{f"v8_{k}": v for k, v in v8_stats.items()},
        **{f"v9_{k}": v for k, v in v9_stats.items()},
        **compare,
    }


def markdown_report(rows: list[dict[str, object]]) -> str:
    lines = [
        "# V9 Output-Planes Sorted Parity Diagnostic",
        "",
        "This compares output-planes fixed eval against v8 forward eval on black-background overlap stacks.",
        "Reverse/depth-descending order is the only fixed-blend candidate expected to match v8 color.",
        "Even a color match does not produce `final_T`, `stop_count`, or backward replay state.",
        "",
        "| status | size | G | format | order | <=1e-5 | max err | mean err | v8 ms | v9 ms | v8/v9 |",
        "| --- | ---: | ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        if row.get("status") != "ok":
            lines.append(
                "| skipped | {height}x{width} | {gaussians} | {output_format} | {order} |  |  |  |  |  |  |".format(
                    **row
                )
            )
            continue
        lines.append(
            "| ok | {height}x{width} | {gaussians} | {output_format} | {order} | {rgb_within_1e_5} | "
            "{rgb_max_abs_err:.6g} | {rgb_mean_abs_err:.6g} | {v8_median_ms:.3f} | "
            "{v9_median_ms:.3f} | {v8_over_v9_median_speedup:.3f} |".format(**row)
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare output-planes sorted eval orders against v8 forward.")
    parser.add_argument("--sizes", default="16x32,64x64")
    parser.add_argument("--gaussians", default="2,16")
    parser.add_argument("--orders", default="input,ascending,descending")
    parser.add_argument("--output-formats", default="rgba32f")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--direct", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--jsonl", type=Path)
    parser.add_argument("--markdown", type=Path)
    args = parser.parse_args()

    if not torch.backends.mps.is_available():
        raise SystemExit("MPS is not available; this benchmark requires MPS tensors.")

    status = probe_hw_interop(compile_pipelines=True, compile_advanced=False, run_render_probe=False)
    if not status.native_extension_loaded or not status.gaussian_eval_rgba_op_available:
        raise SystemExit(f"v9 Gaussian eval op unavailable: {status.as_dict()}")

    rows: list[dict[str, object]] = []
    for height, width in [parse_size(raw) for raw in parse_str_list(args.sizes)]:
        for gaussians in parse_int_list(args.gaussians):
            for output_format in parse_str_list(args.output_formats):
                if output_format not in GAUSSIAN_OUTPUT_FORMATS:
                    raise SystemExit(f"unknown output format {output_format!r}; expected one of {GAUSSIAN_OUTPUT_FORMATS}")
                for order in parse_str_list(args.orders):
                    if args.direct and width % direct_width_multiple(output_format) != 0:
                        row = {
                            "status": "skipped",
                            "height": int(height),
                            "width": int(width),
                            "gaussians": int(gaussians),
                            "output_format": output_format,
                            "order": order,
                            "reason": "width is not aligned for direct buffer-backed render target",
                        }
                    else:
                        row = run_case(
                            height=height,
                            width=width,
                            gaussians=gaussians,
                            output_format=output_format,
                            order=order,
                            warmup=args.warmup,
                            iters=args.iters,
                            direct=args.direct,
                        )
                    rows.append(row)
                    print(json.dumps(row, sort_keys=True))

    if args.jsonl is not None:
        args.jsonl.parent.mkdir(parents=True, exist_ok=True)
        with args.jsonl.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, sort_keys=True) + "\n")

    if args.markdown is not None:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(markdown_report(rows), encoding="utf-8")


if __name__ == "__main__":
    main()

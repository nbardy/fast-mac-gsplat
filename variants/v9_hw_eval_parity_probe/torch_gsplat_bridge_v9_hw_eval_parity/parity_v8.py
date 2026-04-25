from __future__ import annotations

import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import torch
from torch import Tensor

from .interop import probe_hw_interop, render_gaussian_eval_rgba


CURRENT_V9_LIMITATIONS = (
    "eval-only; no backward path",
    "batch size 1 only",
    "expects already projected pixel-space means2d and conics",
    "no depth sort; multi-splat hardware blend order is not v8-equivalent",
    "no tile/imageblock path",
    "no transmittance early termination",
    "black transparent clear only; compare RGB against v8 black background",
    "direct path requires width * 16 bytes to be 256-byte aligned",
)


@dataclass(frozen=True)
class ProjectedInputs:
    case: str
    requested_gaussians: int
    means2d: Tensor
    conics: Tensor
    colors: Tensor
    opacities: Tensor
    depths: Tensor
    comparable_to_v8: bool
    notes: tuple[str, ...]

    @property
    def gaussians(self) -> int:
        return int(self.means2d.shape[0])


def _variant_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _ensure_local_v8_path() -> None:
    v8_root = _variant_root().parent / "v8_hw_eval"
    if str(v8_root) not in sys.path:
        sys.path.insert(0, str(v8_root))


def _load_v8_api():
    _ensure_local_v8_path()
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
            "before the v9 parity harness"
        )

    return RasterConfig, get_runtime_shader_config, rasterize_projected_gaussians


def parse_size(raw: str) -> tuple[int, int]:
    h, w = raw.lower().split("x", 1)
    return int(h), int(w)


def parse_int_list(raw: str) -> tuple[int, ...]:
    return tuple(int(x.strip()) for x in raw.split(",") if x.strip())


def parse_str_list(raw: str) -> tuple[str, ...]:
    return tuple(x.strip() for x in raw.split(",") if x.strip())


def direct_width_aligned(width: int) -> bool:
    return (int(width) * 16) % 256 == 0


def _cpu_generator(seed: int) -> torch.Generator:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    return gen


def _linspace_grid(height: int, width: int, count: int) -> Tensor:
    side = int(torch.ceil(torch.sqrt(torch.tensor(float(count)))).item())
    xs = torch.linspace(max(0.5, width * 0.18), max(0.5, width * 0.82), side)
    ys = torch.linspace(max(0.5, height * 0.18), max(0.5, height * 0.82), side)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    means = torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=1)[:count]
    means[:, 0].clamp_(0.5, max(0.5, width - 0.5))
    means[:, 1].clamp_(0.5, max(0.5, height - 0.5))
    return means


def make_projected_inputs(
    case: str,
    *,
    height: int,
    width: int,
    gaussians: int,
    seed: int = 0,
    device: torch.device | str = "mps",
) -> ProjectedInputs:
    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive")
    if gaussians <= 0:
        raise ValueError("gaussians must be positive")

    device = torch.device(device)
    gen = _cpu_generator(seed)
    notes: list[str] = []
    comparable = True

    if case == "tiny_single":
        count = 1
        means = torch.tensor([[width // 2 + 0.5, height // 2 + 0.5]], dtype=torch.float32)
        conics = torch.tensor([[0.25, 0.0, 0.25]], dtype=torch.float32)
        colors = torch.tensor([[0.25, 0.5, 0.75]], dtype=torch.float32)
        opacities = torch.tensor([0.65], dtype=torch.float32)
        notes.append("single Gaussian; v8 depth sort is a no-op")
    elif case == "grid_ordered":
        count = int(gaussians)
        means = _linspace_grid(height, width, count)
        idx = torch.arange(count, dtype=torch.float32)
        sx = 2.0 + idx.remainder(5) * 0.35
        sy = 2.4 + idx.remainder(7) * 0.30
        conics = torch.stack((1.0 / sx.square(), torch.zeros_like(sx), 1.0 / sy.square()), dim=1)
        colors = torch.stack(
            (
                0.15 + idx.remainder(3) * 0.25,
                0.25 + idx.remainder(5) * 0.12,
                0.20 + idx.remainder(7) * 0.08,
            ),
            dim=1,
        ).clamp_(0.0, 1.0)
        opacities = (0.25 + idx.remainder(4) * 0.08).to(torch.float32)
        notes.append("depths are monotonic with input order; black background")
        if count > 1:
            comparable = False
            notes.append("multi-splat diagnostic; current v9 blend order is not a v8 order guarantee")
    elif case in {"overlap_ordered", "depth_mismatch"}:
        count = int(gaussians)
        center = torch.tensor([width * 0.5, height * 0.5], dtype=torch.float32)
        scale = torch.tensor([max(width * 0.10, 1.0), max(height * 0.10, 1.0)], dtype=torch.float32)
        means = center + torch.randn((count, 2), generator=gen, dtype=torch.float32) * scale
        means[:, 0].clamp_(0.5, max(0.5, width - 0.5))
        means[:, 1].clamp_(0.5, max(0.5, height - 0.5))
        sig = torch.rand((count, 2), generator=gen, dtype=torch.float32) * 3.0 + 3.0
        conics = torch.stack((1.0 / sig[:, 0].square(), torch.zeros(count), 1.0 / sig[:, 1].square()), dim=1)
        colors = torch.rand((count, 3), generator=gen, dtype=torch.float32) * 0.75 + 0.10
        opacities = torch.rand((count,), generator=gen, dtype=torch.float32) * 0.20 + 0.12
        if case == "depth_mismatch":
            comparable = False
            notes.append("intentional order diagnostic: v8 sorts by depth while v9 has no v8 order contract")
        else:
            notes.append("overlapping splats with depths monotonic in input order")
            if count > 1:
                comparable = False
                notes.append("multi-splat diagnostic; current v9 blend order is not a v8 order guarantee")
    else:
        raise ValueError("unknown case; expected tiny_single, grid_ordered, overlap_ordered, or depth_mismatch")

    depths = torch.arange(count, dtype=torch.float32) / float(max(count - 1, 1))
    if case == "depth_mismatch":
        depths = torch.flip(depths, dims=(0,))

    return ProjectedInputs(
        case=case,
        requested_gaussians=int(gaussians),
        means2d=means.to(device=device, dtype=torch.float32).contiguous(),
        conics=conics.to(device=device, dtype=torch.float32).contiguous(),
        colors=colors.to(device=device, dtype=torch.float32).contiguous(),
        opacities=opacities.to(device=device, dtype=torch.float32).contiguous(),
        depths=depths.to(device=device, dtype=torch.float32).contiguous(),
        comparable_to_v8=comparable,
        notes=tuple(notes),
    )


def render_v8_direct_rgb(inputs: ProjectedInputs, *, height: int, width: int) -> Tensor:
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
    return rasterize_projected_gaussians(
        inputs.means2d,
        inputs.conics,
        inputs.colors,
        inputs.opacities,
        inputs.depths,
        cfg,
    )


def render_v9_fixed_eval_rgba(inputs: ProjectedInputs, *, height: int, width: int, direct: bool = True) -> Tensor:
    return render_gaussian_eval_rgba(
        inputs.means2d,
        inputs.conics,
        inputs.colors,
        inputs.opacities,
        int(height),
        int(width),
        direct=bool(direct),
    )


def _sync_mps() -> None:
    if torch.backends.mps.is_available():
        torch.mps.synchronize()


def _time_ms(fn: Callable[[], Tensor], *, warmup: int, iters: int) -> tuple[dict[str, float], Tensor]:
    if warmup < 0:
        raise ValueError("warmup must be non-negative")
    if iters <= 0:
        raise ValueError("iters must be positive")

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


def _prefix_stats(prefix: str, stats: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}_{k}": v for k, v in stats.items()}


def _compare_rgb(v8_rgb: Tensor, v9_rgba: Tensor) -> dict[str, float]:
    v8_cpu = v8_rgb.detach().cpu()
    v9_cpu = v9_rgba.detach().cpu()
    diff = (v9_cpu[..., :3] - v8_cpu).abs()
    alpha = v9_cpu[..., 3]
    return {
        "rgb_max_abs_err": float(diff.max().item()),
        "rgb_mean_abs_err": float(diff.mean().item()),
        "v8_rgb_max": float(v8_cpu.max().item()),
        "v9_rgb_max": float(v9_cpu[..., :3].max().item()),
        "v9_alpha_max": float(alpha.max().item()),
        "v9_alpha_mean": float(alpha.mean().item()),
    }


def run_parity_case(
    case: str,
    *,
    height: int,
    width: int,
    gaussians: int,
    seed: int = 0,
    warmup: int = 3,
    iters: int = 10,
    v9_direct: bool = True,
) -> dict[str, Any]:
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is not available")
    if v9_direct and not direct_width_aligned(width):
        raise ValueError("v9 direct render requires width * 16 bytes to be 256-byte aligned")

    status = probe_hw_interop(compile_pipelines=True, compile_advanced=False, run_render_probe=False)
    if not status.native_extension_loaded or not status.gaussian_eval_rgba_op_available:
        raise RuntimeError(f"v9 Gaussian eval op unavailable: {status.as_dict()}")

    inputs = make_projected_inputs(
        case,
        height=int(height),
        width=int(width),
        gaussians=int(gaussians),
        seed=int(seed),
        device="mps",
    )

    with torch.no_grad():
        v8_fn = lambda: render_v8_direct_rgb(inputs, height=height, width=width)
        v9_fn = lambda: render_v9_fixed_eval_rgba(inputs, height=height, width=width, direct=v9_direct)
        v8_stats, v8_rgb = _time_ms(v8_fn, warmup=warmup, iters=iters)
        v9_stats, v9_rgba = _time_ms(v9_fn, warmup=warmup, iters=iters)
        compare = _compare_rgb(v8_rgb, v9_rgba)

    v8_med = v8_stats["median_ms"]
    v9_med = v9_stats["median_ms"]
    speedup = v8_med / v9_med if v9_med > 0.0 else float("inf")
    rgb_within_1e_5 = compare["rgb_max_abs_err"] <= 1.0e-5
    return {
        "status": "ok",
        "case": inputs.case,
        "height": int(height),
        "width": int(width),
        "pixels": int(height) * int(width),
        "requested_gaussians": int(inputs.requested_gaussians),
        "gaussians": int(inputs.gaussians),
        "seed": int(seed),
        "warmup": int(warmup),
        "iters": int(iters),
        "v8_path": "v8_forward_eval_direct_tiles_active_off_black_bg",
        "v9_path": "v9_fixed_eval_rgba_direct" if v9_direct else "v9_fixed_eval_rgba_private_texture_blit",
        "v9_direct": bool(v9_direct),
        "direct_width_aligned": bool(direct_width_aligned(width)),
        "comparable_to_v8": bool(inputs.comparable_to_v8),
        "assumptions": [
            "B=1",
            "v8 background is black",
            "compare v9 premultiplied RGB only; alpha is diagnostic",
            "single-G rows are parity candidates; multi-G rows are ordering diagnostics",
        ],
        "case_notes": list(inputs.notes),
        "v9_limitations": list(CURRENT_V9_LIMITATIONS),
        "validation_uses_cpu_readback": True,
        "native_op_uses_cpu_readback": False,
        "rgb_within_1e_5": bool(rgb_within_1e_5),
        "v8_over_v9_median_speedup": float(speedup),
        **_prefix_stats("v8", v8_stats),
        **_prefix_stats("v9", v9_stats),
        **compare,
    }


def skipped_row(
    *,
    case: str,
    height: int,
    width: int,
    gaussians: int,
    seed: int,
    warmup: int,
    iters: int,
    reason: str,
) -> dict[str, Any]:
    return {
        "status": "skipped",
        "case": case,
        "height": int(height),
        "width": int(width),
        "pixels": int(height) * int(width),
        "requested_gaussians": int(gaussians),
        "gaussians": 0,
        "seed": int(seed),
        "warmup": int(warmup),
        "iters": int(iters),
        "reason": reason,
        "v9_limitations": list(CURRENT_V9_LIMITATIONS),
    }


def markdown_report(rows: Iterable[dict[str, Any]]) -> str:
    rows = list(rows)
    lines = [
        "# V9 HW Eval Parity vs V8 Forward",
        "",
        "Comparison target: v9 fixed eval RGBA premultiplied RGB against v8 forward-eval RGB.",
        "Validation readback happens after each native op returns; the native v9 render op itself does not read GPU data on CPU.",
        "",
        "## Current v9 limitations",
        "",
    ]
    lines.extend(f"- {item}" for item in CURRENT_V9_LIMITATIONS)
    lines.extend(
        [
            "",
            "## Rows",
            "",
            "| status | case | size | G | comparable | <=1e-5 | max err | mean err | v8 median ms | v9 median ms | v8/v9 | notes |",
            "| --- | --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in rows:
        if row.get("status") != "ok":
            lines.append(
                "| skipped | {case} | {height}x{width} | {requested_gaussians} | no |  |  |  |  |  |  | {reason} |".format(
                    **row
                )
            )
            continue
        notes = "; ".join(row.get("case_notes", []))
        lines.append(
            "| {status} | {case} | {height}x{width} | {gaussians} | {comparable_to_v8} | {rgb_within_1e_5} | "
            "{rgb_max_abs_err:.6g} | {rgb_mean_abs_err:.6g} | {v8_median_ms:.3f} | "
            "{v9_median_ms:.3f} | {v8_over_v9_median_speedup:.3f} | {notes} |".format(
                notes=notes,
                **row,
            )
        )
    lines.append("")
    return "\n".join(lines)

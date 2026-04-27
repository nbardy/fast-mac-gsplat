from __future__ import annotations

import argparse
import itertools
import json
import platform
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from statistics import median, stdev
from typing import Any, Callable

import torch


ROOT = Path(__file__).resolve().parents[1]
V5_ROOT = ROOT / "variants" / "v5"
V8_ROOT = ROOT / "variants" / "v8"
DEFAULT_BG = (0.0, 0.0, 0.0)
DEFAULT_RENDERERS = "v5_default,v5_presorted,v8_direct"
DEFAULT_MODES = "forward,forward_backward"


TensorTuple = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


@dataclass(frozen=True)
class Renderer:
    name: str
    fn: Callable[..., torch.Tensor]
    config: Any
    input_order: str


@dataclass(frozen=True)
class Case:
    height: int
    width: int
    gaussians: int
    batch_size: int
    distribution: str
    seed: int
    role: str = "timing"

    @property
    def case_id(self) -> str:
        return f"{self.width}x{self.height}_B{self.batch_size}_G{self.gaussians}_{self.distribution}_seed{self.seed}"


def ensure_path(path: Path) -> None:
    raw = str(path)
    if raw not in sys.path:
        sys.path.insert(0, raw)


def csv_str(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def csv_int(raw: str) -> list[int]:
    return [int(part) for part in csv_str(raw)]


def parse_resolutions(raw: str) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for item in csv_str(raw):
        width_raw, height_raw = item.lower().split("x", 1)
        out.append((int(height_raw), int(width_raw)))
    return out


def sync_mps() -> None:
    if torch.backends.mps.is_available():
        torch.mps.synchronize()


def git_head() -> str:
    proc = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False)
    return proc.stdout.strip() if proc.returncode == 0 else "unknown"


def make_inputs(case: Case, device: torch.device) -> TensorTuple:
    torch.manual_seed(case.seed)
    B, G, H, W = case.batch_size, case.gaussians, case.height, case.width
    if case.distribution in ("uniform_random", "microbench_uniform_random", "medium_sigma_3_8"):
        means2d = torch.rand(B, G, 2, device=device, dtype=torch.float32)
        means2d[..., 0] *= W
        means2d[..., 1] *= H
        sig = torch.rand(B, G, 2, device=device, dtype=torch.float32) * 5.0 + 3.0
        depths = torch.rand(B, G, device=device, dtype=torch.float32)
    elif case.distribution == "sparse_screen":
        centers = torch.tensor(
            [[0.22 * W, 0.24 * H], [0.76 * W, 0.30 * H], [0.55 * W, 0.76 * H], [0.24 * W, 0.68 * H]],
            device=device,
            dtype=torch.float32,
        )
        choices = torch.randint(0, centers.shape[0], (B, G), device=device)
        means2d = centers.index_select(0, choices.reshape(-1)).view(B, G, 2)
        jitter = torch.tensor([0.025 * W, 0.025 * H], device=device, dtype=torch.float32)
        means2d = means2d + torch.randn(B, G, 2, device=device, dtype=torch.float32) * jitter
        means2d[..., 0].clamp_(0, W - 1)
        means2d[..., 1].clamp_(0, H - 1)
        sig = torch.rand(B, G, 2, device=device, dtype=torch.float32) * 4.0 + 2.0
        depths = torch.rand(B, G, device=device, dtype=torch.float32)
    elif case.distribution == "clustered_hot_tiles":
        centers = torch.tensor([[0.50 * W, 0.50 * H], [0.53 * W, 0.48 * H]], device=device, dtype=torch.float32)
        choices = torch.randint(0, centers.shape[0], (B, G), device=device)
        means2d = centers.index_select(0, choices.reshape(-1)).view(B, G, 2)
        jitter = torch.tensor([0.018 * W, 0.018 * H], device=device, dtype=torch.float32)
        means2d = means2d + torch.randn(B, G, 2, device=device, dtype=torch.float32) * jitter
        means2d[..., 0].clamp_(0, W - 1)
        means2d[..., 1].clamp_(0, H - 1)
        sig = torch.rand(B, G, 2, device=device, dtype=torch.float32) * 8.0 + 4.0
        depths = torch.rand(B, G, device=device, dtype=torch.float32)
    elif case.distribution == "layered_depth":
        centers = torch.tensor(
            [[0.38 * W, 0.40 * H], [0.58 * W, 0.55 * H], [0.48 * W, 0.70 * H]],
            device=device,
            dtype=torch.float32,
        )
        choices = torch.randint(0, centers.shape[0], (B, G), device=device)
        means2d = centers.index_select(0, choices.reshape(-1)).view(B, G, 2)
        jitter = torch.tensor([0.045 * W, 0.045 * H], device=device, dtype=torch.float32)
        means2d = means2d + torch.randn(B, G, 2, device=device, dtype=torch.float32) * jitter
        means2d[..., 0].clamp_(0, W - 1)
        means2d[..., 1].clamp_(0, H - 1)
        sig = torch.rand(B, G, 2, device=device, dtype=torch.float32) * 10.0 + 4.0
        bands = torch.linspace(0.05, 0.95, 6, device=device, dtype=torch.float32)
        band_ids = torch.arange(G, device=device).remainder(bands.numel()).view(1, G).expand(B, G)
        depths = bands.index_select(0, band_ids.reshape(-1)).view(B, G)
        depths = (depths + torch.randn(B, G, device=device, dtype=torch.float32) * 0.01).clamp_(0.0, 1.0)
    else:
        raise ValueError(f"unknown distribution: {case.distribution}")

    conics = torch.stack(
        [
            1.0 / torch.clamp(sig[..., 0].square(), min=1e-4),
            torch.zeros(B, G, device=device, dtype=torch.float32),
            1.0 / torch.clamp(sig[..., 1].square(), min=1e-4),
        ],
        dim=-1,
    ).contiguous()
    colors = torch.rand(B, G, 3, device=device, dtype=torch.float32).contiguous()
    opacities = torch.rand(B, G, device=device, dtype=torch.float32).mul_(0.7).add_(0.1).contiguous()
    return means2d.contiguous(), conics, colors, opacities, depths.contiguous()


def gather_2d(values: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
    return values.gather(1, perm.unsqueeze(-1).expand(-1, -1, values.shape[-1]))


def gather_1d(values: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
    return values.gather(1, perm)


def sort_inputs_by_depth(inputs: TensorTuple) -> tuple[TensorTuple, torch.Tensor]:
    means2d, conics, colors, opacities, depths = inputs
    perm = torch.argsort(depths.detach(), dim=1, stable=True)
    sorted_inputs = (
        gather_2d(means2d, perm).contiguous(),
        gather_2d(conics, perm).contiguous(),
        gather_2d(colors, perm).contiguous(),
        gather_1d(opacities, perm).contiguous(),
        gather_1d(depths, perm).contiguous(),
    )
    return sorted_inputs, perm


def gather_grad_by_perm(grad: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
    if grad.ndim == 3:
        return gather_2d(grad, perm)
    if grad.ndim == 2:
        return gather_1d(grad, perm)
    raise ValueError(f"unexpected grad rank: {grad.ndim}")


def clone_inputs(inputs: TensorTuple, *, backward: bool) -> TensorTuple:
    cloned = []
    for i, tensor in enumerate(inputs):
        item = tensor.detach().clone().contiguous()
        item.requires_grad_(backward and i < 4)
        cloned.append(item)
    return tuple(cloned)  # type: ignore[return-value]


def clear_grads(inputs: TensorTuple) -> None:
    for tensor in inputs[:4]:
        if tensor.grad is not None:
            tensor.grad.zero_()


def load_renderer(name: str, height: int, width: int) -> Renderer:
    if name in ("v5_default", "v5_presorted"):
        ensure_path(V5_ROOT)
        from torch_gsplat_bridge_v5 import RasterConfig, get_runtime_shader_config, rasterize_projected_gaussians

        rt = get_runtime_shader_config()
        cfg = RasterConfig(
            height=height,
            width=width,
            tile_size=rt.tile_size,
            max_fast_pairs=rt.fast_cap,
            background=DEFAULT_BG,
            inputs_sorted_by_depth=(name == "v5_presorted"),
        )
        return Renderer(name=name, fn=rasterize_projected_gaussians, config=cfg, input_order="depth_presorted" if name == "v5_presorted" else "generated")

    if name == "v8_direct":
        ensure_path(V8_ROOT)
        from torch_gsplat_bridge_v8 import RasterConfig, get_runtime_shader_config, rasterize_projected_gaussians

        rt = get_runtime_shader_config()
        cfg = RasterConfig(
            height=height,
            width=width,
            tile_size=rt.tile_size,
            max_fast_pairs=rt.fast_cap,
            background=DEFAULT_BG,
            active_policy="off",
        )
        return Renderer(name=name, fn=rasterize_projected_gaussians, config=cfg, input_order="generated")

    raise ValueError(f"unknown renderer: {name}")


def inputs_for_renderer(renderer: Renderer, base_inputs: TensorTuple, sorted_inputs: TensorTuple) -> TensorTuple:
    return sorted_inputs if renderer.input_order == "depth_presorted" else base_inputs


def time_renderer(
    renderer: Renderer,
    case: Case,
    mode: str,
    base_inputs: TensorTuple,
    sorted_inputs: TensorTuple,
    *,
    warmup: int,
    iters: int,
) -> dict[str, Any]:
    backward = mode == "forward_backward"
    source_inputs = inputs_for_renderer(renderer, base_inputs, sorted_inputs)
    run_inputs = clone_inputs(source_inputs, backward=backward)
    sync_mps()

    def step() -> tuple[float, float]:
        if backward:
            clear_grads(run_inputs)
        sync_mps()
        if backward:
            t0 = time.perf_counter()
            out = renderer.fn(*run_inputs, renderer.config)
            sync_mps()
            t1 = time.perf_counter()
            out.square().mean().backward()
            sync_mps()
            t2 = time.perf_counter()
            clear_grads(run_inputs)
            return (t1 - t0) * 1000.0, (t2 - t1) * 1000.0

        with torch.no_grad():
            t0 = time.perf_counter()
            _ = renderer.fn(*run_inputs, renderer.config)
            sync_mps()
            t1 = time.perf_counter()
        return (t1 - t0) * 1000.0, 0.0

    for _ in range(warmup):
        step()

    forward_times: list[float] = []
    backward_times: list[float] = []
    total_times: list[float] = []
    for _ in range(iters):
        f_ms, b_ms = step()
        forward_times.append(f_ms)
        backward_times.append(b_ms)
        total_times.append(f_ms + b_ms)

    return {
        "kind": "timing",
        "status": "ok",
        "renderer": renderer.name,
        "mode": mode,
        "height": case.height,
        "width": case.width,
        "batch_size": case.batch_size,
        "gaussians": case.gaussians,
        "distribution": case.distribution,
        "seed": case.seed,
        "case_id": case.case_id,
        "case_role": case.role,
        "input_order": renderer.input_order,
        "presort_time_excluded": renderer.name == "v5_presorted",
        "warmup": warmup,
        "iters": iters,
        "mean_ms": sum(total_times) / len(total_times),
        "median_ms": float(median(total_times)),
        "min_ms": min(total_times),
        "max_ms": max(total_times),
        "stddev_ms": float(stdev(total_times)) if len(total_times) > 1 else 0.0,
        "forward_mean_ms": sum(forward_times) / len(forward_times),
        "forward_median_ms": float(median(forward_times)),
        "backward_mean_ms": sum(backward_times) / len(backward_times),
        "backward_median_ms": float(median(backward_times)),
        "samples_ms": total_times,
    }


def run_with_grads(renderer: Renderer, inputs: TensorTuple) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
    run_inputs = clone_inputs(inputs, backward=True)
    out = renderer.fn(*run_inputs, renderer.config)
    sync_mps()
    out.square().mean().backward()
    sync_mps()
    grads = tuple(t.grad.detach().clone() for t in run_inputs[:4] if t.grad is not None)
    return out.detach().clone(), grads


def parity_v5_default_presorted(
    case: Case,
    base_inputs: TensorTuple,
    sorted_inputs: TensorTuple,
    perm: torch.Tensor,
    *,
    threshold: float,
) -> dict[str, Any]:
    default_renderer = load_renderer("v5_default", case.height, case.width)
    presorted_renderer = load_renderer("v5_presorted", case.height, case.width)
    out_default, grads_default = run_with_grads(default_renderer, base_inputs)
    out_presorted, grads_presorted = run_with_grads(presorted_renderer, sorted_inputs)
    image_abs = (out_default - out_presorted).detach().abs()

    grad_names = ("means2d", "conics", "colors", "opacities")
    grad_max_values: list[float] = []
    result: dict[str, Any] = {
        "kind": "parity",
        "comparison": "v5_default_vs_v5_presorted",
        "height": case.height,
        "width": case.width,
        "batch_size": case.batch_size,
        "gaussians": case.gaussians,
        "distribution": case.distribution,
        "seed": case.seed,
        "case_id": case.case_id,
        "case_role": case.role,
        "threshold": threshold,
        "image_max_abs_error": float(image_abs.max().item()),
        "image_mean_abs_error": float(image_abs.mean().item()),
    }
    for name, default_grad, presorted_grad in zip(grad_names, grads_default, grads_presorted):
        default_sorted_grad = gather_grad_by_perm(default_grad, perm)
        grad_abs = (default_sorted_grad - presorted_grad).detach().abs()
        max_err = float(grad_abs.max().item())
        result[f"grad_{name}_max_abs_error"] = max_err
        result[f"grad_{name}_mean_abs_error"] = float(grad_abs.mean().item())
        grad_max_values.append(max_err)

    result["grad_max_abs_error"] = max(grad_max_values) if grad_max_values else 0.0
    result["max_abs_error"] = max(float(result["image_max_abs_error"]), float(result["grad_max_abs_error"]))
    result["status"] = "ok" if float(result["max_abs_error"]) <= threshold else "failed"
    return result


def group_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("height"),
        row.get("width"),
        row.get("batch_size"),
        row.get("gaussians"),
        row.get("distribution"),
        row.get("mode"),
        row.get("seed"),
    )


def add_comparisons(rows: list[dict[str, Any]], *, noise_threshold_pct: float) -> None:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("kind") == "timing":
            groups.setdefault(group_key(row), []).append(row)

    for group_rows in groups.values():
        ok_rows = [row for row in group_rows if row.get("status") == "ok"]
        if not ok_rows:
            continue
        best = min(ok_rows, key=lambda row: float(row["median_ms"]))
        default = next((row for row in ok_rows if row.get("renderer") == "v5_default"), None)
        presorted = next((row for row in ok_rows if row.get("renderer") == "v5_presorted"), None)
        for row in group_rows:
            row["best_renderer"] = best.get("renderer")
            if row.get("status") != "ok":
                continue
            row["delta_vs_best_pct"] = (float(row["median_ms"]) / float(best["median_ms"]) - 1.0) * 100.0
            if default is not None and row is not default:
                row["delta_vs_v5_default_pct"] = (float(row["median_ms"]) / float(default["median_ms"]) - 1.0) * 100.0
        if default is not None and presorted is not None:
            delta = (float(presorted["median_ms"]) / float(default["median_ms"]) - 1.0) * 100.0
            speedup = (float(default["median_ms"]) / float(presorted["median_ms"]) - 1.0) * 100.0
            if abs(delta) <= noise_threshold_pct:
                verdict = "noisy_flat"
            elif delta < 0.0:
                verdict = "faster"
            else:
                verdict = "slower"
            for row in group_rows:
                row["v5_presorted_verdict"] = verdict
                row["v5_presorted_delta_pct"] = delta
                row["v5_presorted_speedup_pct"] = speedup


def fmt_ms(value: Any) -> str:
    return "" if not isinstance(value, (int, float)) else f"{float(value):.3f}"


def fmt_pct(value: Any) -> str:
    return "" if not isinstance(value, (int, float)) else f"{float(value):+.1f}%"


def fmt_err(value: Any) -> str:
    return "" if not isinstance(value, (int, float)) else f"{float(value):.3e}"


def write_jsonl(rows: list[dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")


def write_markdown(rows: list[dict[str, Any]], out_path: Path, args: argparse.Namespace, metadata: dict[str, Any]) -> None:
    timing_rows = [row for row in rows if row.get("kind") == "timing"]
    parity_rows = [row for row in rows if row.get("kind") == "parity"]

    lines = [
        "# V5/V8 Common-Tensor Benchmark",
        "",
        f"Generated: {metadata['generated_at']}",
        "",
        "V5 presorted receives the same generated Gaussian set after a stable depth sort and sets `inputs_sorted_by_depth=True`; the depth-sort time is intentionally excluded to measure the renderer-side win from avoiding redundant sort/unsort.",
        "",
        "## Settings",
        "",
        f"- git HEAD: `{metadata['git_head']}`",
        f"- python: `{metadata['python']}`",
        f"- torch: `{metadata['torch']}`",
        f"- platform: `{metadata['platform']}`",
        f"- warmup: `{args.warmup}`",
        f"- iters: `{args.iters}`",
        f"- renderers: `{args.renderers}`",
        f"- modes: `{args.modes}`",
        f"- noise threshold: `{args.noise_threshold_pct:.1f}%`",
        f"- command: `{metadata['command']}`",
        "",
    ]

    if parity_rows:
        lines += [
            "## V5 Default vs Presorted Parity",
            "",
            "| Case | Status | Image Max Err | Grad Max Err | Max Err | Threshold |",
            "|---|---|---:|---:|---:|---:|",
        ]
        for row in parity_rows:
            lines.append(
                f"| {row.get('case_id')} | {row.get('status')} | {fmt_err(row.get('image_max_abs_error'))} | "
                f"{fmt_err(row.get('grad_max_abs_error'))} | {fmt_err(row.get('max_abs_error'))} | {fmt_err(row.get('threshold'))} |"
            )
        lines.append("")

    presorted_rows = [
        row
        for row in timing_rows
        if row.get("renderer") == "v5_presorted" and row.get("status") == "ok" and row.get("v5_presorted_verdict")
    ]
    if presorted_rows:
        lines += [
            "## V5 Presorted Verdict",
            "",
            "| Case | Mode | V5 Default ms | V5 Presorted ms | Delta | Speedup | Verdict |",
            "|---|---|---:|---:|---:|---:|---|",
        ]
        for row in presorted_rows:
            default = next(
                (
                    candidate
                    for candidate in timing_rows
                    if candidate.get("renderer") == "v5_default"
                    and candidate.get("status") == "ok"
                    and group_key(candidate) == group_key(row)
                ),
                None,
            )
            lines.append(
                f"| {row.get('case_id')} | {row.get('mode')} | {fmt_ms(default.get('median_ms') if default else None)} | "
                f"{fmt_ms(row.get('median_ms'))} | {fmt_pct(row.get('v5_presorted_delta_pct'))} | "
                f"{fmt_pct(row.get('v5_presorted_speedup_pct'))} | {row.get('v5_presorted_verdict')} |"
            )
        lines.append("")

    lines += [
        "## Timing Results",
        "",
        "| Case | Mode | Renderer | Status | Median ms | Mean ms | Fwd Median ms | Bwd Median ms | Stddev ms | Delta vs V5 Default | Best | Notes |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in timing_rows:
        notes = []
        if row.get("presort_time_excluded"):
            notes.append("presort excluded")
        if row.get("error"):
            notes.append(str(row["error"]).replace("|", "/"))
        lines.append(
            f"| {row.get('case_id')} | {row.get('mode')} | {row.get('renderer')} | {row.get('status')} | "
            f"{fmt_ms(row.get('median_ms'))} | {fmt_ms(row.get('mean_ms'))} | {fmt_ms(row.get('forward_median_ms'))} | "
            f"{fmt_ms(row.get('backward_median_ms'))} | {fmt_ms(row.get('stddev_ms'))} | "
            f"{fmt_pct(row.get('delta_vs_v5_default_pct'))} | {row.get('best_renderer', '')} | {'; '.join(notes)} |"
        )
    lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def build_cases(args: argparse.Namespace) -> list[Case]:
    cases = [
        Case(height=height, width=width, gaussians=gaussians, batch_size=batch_size, distribution=distribution, seed=args.seed)
        for (height, width), gaussians, batch_size, distribution in itertools.product(
            parse_resolutions(args.resolutions),
            csv_int(args.splats),
            csv_int(args.batch_sizes),
            csv_str(args.distributions),
        )
    ]
    if args.include_accuracy_case:
        acc_height, acc_width = parse_resolutions(args.accuracy_resolution)[0]
        accuracy_case = Case(
            height=acc_height,
            width=acc_width,
            gaussians=args.accuracy_splats,
            batch_size=args.accuracy_batch_size,
            distribution=args.accuracy_distribution,
            seed=args.accuracy_seed,
            role="accuracy",
        )
        if all(case.case_id != accuracy_case.case_id for case in cases):
            cases.append(accuracy_case)
    return cases


def run(args: argparse.Namespace) -> list[dict[str, Any]]:
    if not torch.backends.mps.is_available():
        raise SystemExit("MPS is not available.")

    device = torch.device("mps")
    rows: list[dict[str, Any]] = []
    for case in build_cases(args):
        print(f"case {case.case_id}", flush=True)
        base_inputs = make_inputs(case, device)
        sorted_inputs, perm = sort_inputs_by_depth(base_inputs)

        if case.role == "accuracy":
            try:
                parity_row = parity_v5_default_presorted(case, base_inputs, sorted_inputs, perm, threshold=args.parity_threshold)
            except Exception as exc:
                parity_row = {
                    "kind": "parity",
                    "status": "error",
                    "comparison": "v5_default_vs_v5_presorted",
                    "case_id": case.case_id,
                    "error": str(exc),
                    "traceback": traceback.format_exc(limit=8),
                }
            rows.append(parity_row)

        for mode in csv_str(args.modes):
            if mode not in ("forward", "forward_backward"):
                raise ValueError(f"unknown mode: {mode}")
            for renderer_name in csv_str(args.renderers):
                print(f"  {mode} {renderer_name}", flush=True)
                try:
                    renderer = load_renderer(renderer_name, case.height, case.width)
                    row = time_renderer(
                        renderer,
                        case,
                        mode,
                        base_inputs,
                        sorted_inputs,
                        warmup=args.warmup,
                        iters=args.iters,
                    )
                except Exception as exc:
                    row = {
                        "kind": "timing",
                        "status": "error",
                        "renderer": renderer_name,
                        "mode": mode,
                        "height": case.height,
                        "width": case.width,
                        "batch_size": case.batch_size,
                        "gaussians": case.gaussians,
                        "distribution": case.distribution,
                        "seed": case.seed,
                        "case_id": case.case_id,
                        "case_role": case.role,
                        "error": str(exc),
                        "traceback": traceback.format_exc(limit=8),
                    }
                rows.append(row)

    add_comparisons(rows, noise_threshold_pct=args.noise_threshold_pct)
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description="Compare V5 default, V5 presorted, and V8 direct on common generated projected tensors.")
    p.add_argument("--resolutions", type=str, default="512x512")
    p.add_argument("--splats", type=str, default="6000")
    p.add_argument("--batch-sizes", type=str, default="1,4")
    p.add_argument("--distributions", type=str, default="microbench_uniform_random")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--renderers", type=str, default=DEFAULT_RENDERERS)
    p.add_argument("--modes", type=str, default=DEFAULT_MODES)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--iters", type=int, default=7)
    p.add_argument("--include-accuracy-case", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--accuracy-resolution", type=str, default="64x64")
    p.add_argument("--accuracy-splats", type=int, default=128)
    p.add_argument("--accuracy-batch-size", type=int, default=2)
    p.add_argument("--accuracy-distribution", type=str, default="layered_depth")
    p.add_argument("--accuracy-seed", type=int, default=17)
    p.add_argument("--parity-threshold", type=float, default=1e-5)
    p.add_argument("--noise-threshold-pct", type=float, default=5.0)
    p.add_argument("--output-md", type=str, default="benchmarks/v5_v8_common_tensor_compare.md")
    p.add_argument("--output-jsonl", type=str, default="benchmarks/v5_v8_common_tensor_compare.jsonl")
    args = p.parse_args()

    metadata = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "git_head": git_head(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "platform": platform.platform(),
        "command": " ".join([sys.executable, *sys.argv]),
    }
    rows = run(args)

    out_jsonl = Path(args.output_jsonl)
    if not out_jsonl.is_absolute():
        out_jsonl = ROOT / out_jsonl
    out_md = Path(args.output_md)
    if not out_md.is_absolute():
        out_md = ROOT / out_md
    write_jsonl(rows, out_jsonl)
    write_markdown(rows, out_md, args, metadata)
    print(f"wrote {out_jsonl}")
    print(f"wrote {out_md}")


if __name__ == "__main__":
    main()

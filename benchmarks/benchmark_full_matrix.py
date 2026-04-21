from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path
from statistics import median
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
V3_ROOT = ROOT / "variants" / "v3"
V5_ROOT = ROOT / "variants" / "v5"
V6_ROOT = ROOT / "variants" / "v6"
V7_ROOT = ROOT / "variants" / "v7"
for path in (ROOT, V3_ROOT, V5_ROOT, V6_ROOT, V7_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from torch_gsplat_bridge_fast import RasterConfig as RasterConfigV2
from torch_gsplat_bridge_fast import rasterize_projected_gaussians as rasterize_v2
from torch_gsplat_bridge_v3 import RasterConfig as RasterConfigV3
from torch_gsplat_bridge_v3 import rasterize_projected_gaussians as rasterize_v3
from torch_gsplat_bridge_v5 import RasterConfig as RasterConfigV5
from torch_gsplat_bridge_v5 import rasterize_projected_gaussians as rasterize_v5
from torch_gsplat_bridge_v6 import RasterConfig as RasterConfigV6
from torch_gsplat_bridge_v6 import rasterize_projected_gaussians as rasterize_v6
from torch_gsplat_bridge_v7 import RasterConfig as RasterConfigV7
from torch_gsplat_bridge_v7 import rasterize_projected_gaussians as rasterize_v7


DEFAULT_BG = (0.0, 0.0, 0.0)
DEFAULT_RENDERERS = "torch_direct,v2_fastpath,v3_candidate,v5_batched,v6_direct,v6_auto,v7_hardware"
DEFAULT_DISTS = "microbench_uniform_random,sparse_screen,clustered_hot_tiles,layered_depth"


def sync() -> None:
    torch.mps.synchronize()


def csv_str(raw: str) -> list[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def csv_int(raw: str) -> list[int]:
    return [int(x) for x in csv_str(raw)]


def parse_resolutions(raw: str) -> list[tuple[int, int]]:
    out = []
    for item in csv_str(raw):
        width_raw, height_raw = item.lower().split("x", 1)
        out.append((int(height_raw), int(width_raw)))
    return out


def make_inputs(case: str, B: int, G: int, H: int, W: int, seed: int, device: torch.device):
    torch.manual_seed(seed)
    if case in ("uniform_random", "microbench_uniform_random", "medium_sigma_3_8"):
        means2d = torch.rand(B, G, 2, device=device, dtype=torch.float32)
        means2d[..., 0] *= W
        means2d[..., 1] *= H
        sig = torch.rand(B, G, 2, device=device) * 5.0 + 3.0
        depths = torch.rand(B, G, device=device, dtype=torch.float32)
    elif case == "sparse_screen":
        centers = torch.tensor(
            [[0.22 * W, 0.24 * H], [0.76 * W, 0.30 * H], [0.55 * W, 0.76 * H], [0.24 * W, 0.68 * H]],
            device=device,
            dtype=torch.float32,
        )
        choices = torch.randint(0, centers.shape[0], (B, G), device=device)
        means2d = centers.index_select(0, choices.reshape(-1)).view(B, G, 2)
        means2d = means2d + torch.randn(B, G, 2, device=device, dtype=torch.float32) * torch.tensor([0.025 * W, 0.025 * H], device=device)
        means2d[..., 0].clamp_(0, W - 1)
        means2d[..., 1].clamp_(0, H - 1)
        sig = torch.rand(B, G, 2, device=device) * 4.0 + 2.0
        depths = torch.rand(B, G, device=device, dtype=torch.float32)
    elif case == "clustered_hot_tiles":
        centers = torch.tensor([[0.50 * W, 0.50 * H], [0.53 * W, 0.48 * H]], device=device, dtype=torch.float32)
        choices = torch.randint(0, centers.shape[0], (B, G), device=device)
        means2d = centers.index_select(0, choices.reshape(-1)).view(B, G, 2)
        means2d = means2d + torch.randn(B, G, 2, device=device, dtype=torch.float32) * torch.tensor([0.018 * W, 0.018 * H], device=device)
        means2d[..., 0].clamp_(0, W - 1)
        means2d[..., 1].clamp_(0, H - 1)
        sig = torch.rand(B, G, 2, device=device) * 8.0 + 4.0
        depths = torch.rand(B, G, device=device, dtype=torch.float32)
    elif case == "layered_depth":
        centers = torch.tensor([[0.38 * W, 0.40 * H], [0.58 * W, 0.55 * H], [0.48 * W, 0.70 * H]], device=device, dtype=torch.float32)
        choices = torch.randint(0, centers.shape[0], (B, G), device=device)
        means2d = centers.index_select(0, choices.reshape(-1)).view(B, G, 2)
        means2d = means2d + torch.randn(B, G, 2, device=device, dtype=torch.float32) * torch.tensor([0.045 * W, 0.045 * H], device=device)
        means2d[..., 0].clamp_(0, W - 1)
        means2d[..., 1].clamp_(0, H - 1)
        sig = torch.rand(B, G, 2, device=device) * 10.0 + 4.0
        bands = torch.linspace(0.05, 0.95, 6, device=device, dtype=torch.float32)
        band_ids = torch.arange(G, device=device).remainder(bands.numel()).view(1, G).expand(B, G)
        depths = bands.index_select(0, band_ids.reshape(-1)).view(B, G)
        depths = (depths + torch.randn(B, G, device=device, dtype=torch.float32) * 0.01).clamp_(0.0, 1.0)
    elif case == "overflow_adversarial":
        means2d = torch.randn(B, G, 2, device=device, dtype=torch.float32) * torch.tensor([W * 0.01, H * 0.01], device=device)
        means2d = means2d + torch.tensor([W * 0.5, H * 0.5], device=device)
        means2d[..., 0].clamp_(0, W - 1)
        means2d[..., 1].clamp_(0, H - 1)
        sig = torch.rand(B, G, 2, device=device) * 48.0 + 24.0
        depths = torch.rand(B, G, device=device, dtype=torch.float32)
    else:
        raise ValueError(f"unknown distribution: {case}")

    conics = torch.stack(
        [
            1.0 / torch.clamp(sig[..., 0].square(), min=1e-4),
            torch.zeros(B, G, device=device),
            1.0 / torch.clamp(sig[..., 1].square(), min=1e-4),
        ],
        dim=-1,
    ).contiguous()
    colors = torch.rand(B, G, 3, device=device, dtype=torch.float32).contiguous()
    opacities = torch.rand(B, G, device=device, dtype=torch.float32).mul_(0.7).add_(0.1).contiguous()
    return means2d.contiguous(), conics, colors, opacities, depths.contiguous()


def dense_torch_reference(means2d, conics, colors, opacities, depths, height: int, width: int):
    outs = []
    bg = torch.tensor(DEFAULT_BG, dtype=means2d.dtype, device=means2d.device)
    ys = torch.arange(height, dtype=means2d.dtype, device=means2d.device) + 0.5
    xs = torch.arange(width, dtype=means2d.dtype, device=means2d.device) + 0.5
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    for b in range(means2d.shape[0]):
        perm = torch.argsort(depths[b].detach(), dim=0, stable=True)
        m = means2d[b].index_select(0, perm)
        q = conics[b].index_select(0, perm)
        c = colors[b].index_select(0, perm)
        o = opacities[b].index_select(0, perm)
        out = torch.zeros((height, width, 3), dtype=means2d.dtype, device=means2d.device)
        trans = torch.ones((height, width), dtype=means2d.dtype, device=means2d.device)
        for i in range(m.shape[0]):
            dx = xx - m[i, 0]
            dy = yy - m[i, 1]
            power = -0.5 * (q[i, 0] * dx * dx + 2.0 * q[i, 1] * dx * dy + q[i, 2] * dy * dy)
            alpha = torch.clamp(o[i] * torch.exp(power), max=0.99)
            alpha = torch.where((power <= 0.0) & (alpha >= 1.0 / 255.0), alpha, torch.zeros_like(alpha))
            weight = trans * alpha
            out = out + weight[..., None] * c[i]
            trans = trans * (1.0 - alpha)
        outs.append(out + trans[..., None] * bg)
    return torch.stack(outs, dim=0)


def clone_inputs(inputs, backward: bool):
    out = []
    for i, tensor in enumerate(inputs):
        cloned = tensor.detach().clone().contiguous()
        cloned.requires_grad_(backward and i < 4)
        out.append(cloned)
    return tuple(out)


def clear_grads(inputs) -> None:
    for tensor in inputs[:4]:
        if tensor.grad is not None:
            tensor.grad.zero_()


def loop_single(fn, inputs, cfg):
    means2d, conics, colors, opacities, depths = inputs
    outs = []
    for b in range(means2d.shape[0]):
        outs.append(fn(means2d[b], conics[b], colors[b], opacities[b], depths[b], cfg))
    return torch.stack(outs, dim=0)


def run_renderer(renderer: str, inputs, height: int, width: int):
    if renderer == "torch_direct":
        return dense_torch_reference(*inputs, height, width)
    if renderer == "v2_fastpath":
        return loop_single(rasterize_v2, inputs, RasterConfigV2(height=height, width=width, background=DEFAULT_BG))
    if renderer == "v3_candidate":
        return loop_single(rasterize_v3, inputs, RasterConfigV3(height=height, width=width, background=DEFAULT_BG))
    if renderer == "v5_batched":
        return rasterize_v5(*inputs, RasterConfigV5(height=height, width=width, background=DEFAULT_BG))
    if renderer == "v6_direct":
        return rasterize_v6(*inputs, RasterConfigV6(height=height, width=width, background=DEFAULT_BG, active_policy="off"))
    if renderer == "v6_auto":
        return rasterize_v6(*inputs, RasterConfigV6(height=height, width=width, background=DEFAULT_BG, active_policy="auto"))
    if renderer == "v7_hardware":
        return rasterize_v7(*inputs, RasterConfigV7(height=height, width=width, background=DEFAULT_BG))
    raise ValueError(f"unknown renderer: {renderer}")


def time_one(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.backends.mps.is_available():
        return {"status": "skipped", "reason": "MPS unavailable"}

    mode = args.mode
    backward = mode == "forward_backward"
    work_items = args.batch_size * args.height * args.width * args.gaussians
    if args.renderer == "torch_direct" and work_items > args.torch_max_work_items:
        return {"status": "skipped", "reason": f"torch work_items {work_items} exceeds limit {args.torch_max_work_items}"}

    device = torch.device("mps")
    base_inputs = make_inputs(args.distribution, args.batch_size, args.gaussians, args.height, args.width, args.seed, device)

    def step():
        run_inputs = clone_inputs(base_inputs, backward=backward)
        t0 = time.perf_counter()
        out = run_renderer(args.renderer, run_inputs, args.height, args.width)
        sync()
        t1 = time.perf_counter()
        if backward:
            out.square().mean().backward()
            sync()
        t2 = time.perf_counter()
        clear_grads(run_inputs)
        return (t1 - t0) * 1000.0, (t2 - t1) * 1000.0

    for _ in range(args.warmup):
        step()
    forward_times = []
    backward_times = []
    total_times = []
    for _ in range(args.iters):
        f_ms, b_ms = step()
        forward_times.append(f_ms)
        backward_times.append(b_ms)
        total_times.append(f_ms + b_ms)

    return {
        "status": "ok",
        "renderer": args.renderer,
        "mode": mode,
        "height": args.height,
        "width": args.width,
        "batch_size": args.batch_size,
        "gaussians": args.gaussians,
        "distribution": args.distribution,
        "seed": args.seed,
        "warmup": args.warmup,
        "iters": args.iters,
        "mean_ms": sum(total_times) / len(total_times),
        "median_ms": float(median(total_times)),
        "min_ms": min(total_times),
        "max_ms": max(total_times),
        "forward_ms": sum(forward_times) / len(forward_times),
        "backward_ms": sum(backward_times) / len(backward_times),
    }


def run_one_json(args: argparse.Namespace) -> None:
    try:
        row = time_one(args)
    except Exception as exc:
        row = {
            "status": "error",
            "renderer": args.renderer,
            "mode": args.mode,
            "height": args.height,
            "width": args.width,
            "batch_size": args.batch_size,
            "gaussians": args.gaussians,
            "distribution": args.distribution,
            "seed": args.seed,
            "error": str(exc),
            "traceback": traceback.format_exc(limit=8),
        }
    print(json.dumps(row, sort_keys=True))


def fmt_ms(value: Any) -> str:
    if not isinstance(value, (float, int)):
        return ""
    return f"{float(value):.3f}"


def fmt_pct(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:+.1f}%"


def key_for(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("height"),
        row.get("width"),
        row.get("batch_size"),
        row.get("gaussians"),
        row.get("distribution"),
        row.get("mode"),
    )


def add_relative_columns(rows: list[dict[str, Any]]) -> None:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(key_for(row), []).append(row)

    for group_rows in groups.values():
        ok_rows = [r for r in group_rows if r.get("status") == "ok"]
        if not ok_rows:
            continue
        best = min(ok_rows, key=lambda r: float(r["mean_ms"]))
        torch_row = next((r for r in ok_rows if r.get("renderer") == "torch_direct"), None)
        v6_row = next((r for r in ok_rows if r.get("renderer") == "v6_direct"), None)
        for row in group_rows:
            row["best_renderer"] = best.get("renderer")
            if row.get("status") != "ok":
                continue
            mean = float(row["mean_ms"])
            row["slower_than_best_pct"] = (mean / float(best["mean_ms"]) - 1.0) * 100.0
            if torch_row is not None and torch_row is not row:
                row["speedup_vs_torch_pct"] = (float(torch_row["mean_ms"]) / mean - 1.0) * 100.0
            if v6_row is not None and v6_row is not row:
                row["time_delta_vs_v6_direct_pct"] = (mean / float(v6_row["mean_ms"]) - 1.0) * 100.0


def write_markdown(rows: list[dict[str, Any]], out_path: Path, args: argparse.Namespace) -> None:
    add_relative_columns(rows)
    ok_rows = [r for r in rows if r.get("status") == "ok"]
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in ok_rows:
        groups.setdefault(key_for(row), []).append(row)

    lines = [
        "# Full Rasterizer Benchmark",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "Lower milliseconds are better. `% vs torch` is speedup over direct Torch when Torch was small enough to run. `% vs v6 direct` is time delta, so negative means faster than v6 direct.",
        "",
        "## Settings",
        "",
        f"- warmup: `{args.warmup}`",
        f"- iters: `{args.iters}`",
        f"- seed: `{args.seed}`",
        f"- timeout seconds per cell: `{args.timeout_sec}`",
        "",
        "## Winners",
        "",
        "| Resolution | B | Splats | Distribution | Mode | Best Renderer | Mean ms |",
        "|---|---:|---:|---|---|---|---:|",
    ]
    for key, group_rows in sorted(groups.items()):
        best = min(group_rows, key=lambda r: float(r["mean_ms"]))
        h, w, b, g, dist, mode = key
        lines.append(f"| {w}x{h} | {b} | {g} | {dist} | {mode} | {best['renderer']} | {fmt_ms(best['mean_ms'])} |")

    lines += [
        "",
        "## Full Results",
        "",
        "| Resolution | B | Splats | Distribution | Mode | Renderer | Status | Mean ms | Median ms | Fwd ms | Bwd ms | Slower Than Best | % vs Torch | % vs v6 Direct | Notes |",
        "|---|---:|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        status = row.get("status", "")
        h = row.get("height")
        w = row.get("width")
        note = row.get("reason") or row.get("error") or ""
        lines.append(
            "| "
            f"{w}x{h} | {row.get('batch_size')} | {row.get('gaussians')} | {row.get('distribution')} | {row.get('mode')} | "
            f"{row.get('renderer')} | {status} | {fmt_ms(row.get('mean_ms'))} | {fmt_ms(row.get('median_ms'))} | "
            f"{fmt_ms(row.get('forward_ms'))} | {fmt_ms(row.get('backward_ms'))} | "
            f"{fmt_pct(row.get('slower_than_best_pct'))} | {fmt_pct(row.get('speedup_vs_torch_pct'))} | "
            f"{fmt_pct(row.get('time_delta_vs_v6_direct_pct'))} | {str(note).replace('|', '/')} |"
        )
    lines.append("")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def parent_main(args: argparse.Namespace) -> None:
    resolutions = parse_resolutions(args.resolutions)
    splats = csv_int(args.splats)
    batch_sizes = csv_int(args.batch_sizes)
    distributions = csv_str(args.distributions)
    renderers = csv_str(args.renderers)
    modes = csv_str(args.modes)

    jobs = list(itertools.product(resolutions, splats, batch_sizes, distributions, modes, renderers))
    if args.limit_cells > 0:
        jobs = jobs[: args.limit_cells]

    rows = []
    script = Path(__file__).resolve()
    for i, ((height, width), g, b, dist, mode, renderer) in enumerate(jobs, start=1):
        cmd = [
            sys.executable,
            str(script),
            "--run-one-json",
            "--height", str(height),
            "--width", str(width),
            "--gaussians", str(g),
            "--batch-size", str(b),
            "--distribution", dist,
            "--mode", mode,
            "--renderer", renderer,
            "--warmup", str(args.warmup),
            "--iters", str(args.iters),
            "--seed", str(args.seed),
            "--torch-max-work-items", str(args.torch_max_work_items),
        ]
        label = f"[{i}/{len(jobs)}] {width}x{height} B={b} G={g} {dist} {mode} {renderer}"
        print(label, flush=True)
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=args.timeout_sec, check=False)
        except subprocess.TimeoutExpired:
            rows.append({
                "status": "timeout",
                "renderer": renderer,
                "mode": mode,
                "height": height,
                "width": width,
                "batch_size": b,
                "gaussians": g,
                "distribution": dist,
                "reason": f"timeout after {args.timeout_sec}s",
            })
            continue
        line = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            row = {
                "status": "error",
                "renderer": renderer,
                "mode": mode,
                "height": height,
                "width": width,
                "batch_size": b,
                "gaussians": g,
                "distribution": dist,
                "error": (proc.stderr or proc.stdout)[-1000:],
            }
        rows.append(row)

    out_md = Path(args.output_md)
    if not out_md.is_absolute():
        out_md = ROOT / out_md
    write_markdown(rows, out_md, args)
    if args.output_jsonl:
        out_jsonl = Path(args.output_jsonl)
        if not out_jsonl.is_absolute():
            out_jsonl = ROOT / out_jsonl
        out_jsonl.parent.mkdir(parents=True, exist_ok=True)
        out_jsonl.write_text("\n".join(json.dumps(r, sort_keys=True) for r in rows) + "\n", encoding="utf-8")
    print(f"wrote {out_md}")


def main() -> None:
    p = argparse.ArgumentParser(description="Run a full rasterizer matrix and write a Markdown report.")
    p.add_argument("--run-one-json", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--renderer", type=str, default="v6_direct")
    p.add_argument("--mode", type=str, default="forward", choices=("forward", "forward_backward"))
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--gaussians", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--distribution", type=str, default="microbench_uniform_random")
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--iters", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--torch-max-work-items", type=int, default=64_000_000)
    p.add_argument("--resolutions", type=str, default="512x512,1024x512")
    p.add_argument("--splats", type=str, default="512,2048")
    p.add_argument("--batch-sizes", type=str, default="1,4")
    p.add_argument("--distributions", type=str, default=DEFAULT_DISTS)
    p.add_argument("--renderers", type=str, default=DEFAULT_RENDERERS)
    p.add_argument("--modes", type=str, default="forward,forward_backward")
    p.add_argument("--timeout-sec", type=float, default=120.0)
    p.add_argument("--limit-cells", type=int, default=0)
    p.add_argument("--output-md", type=str, default="benchmarks/full_rasterizer_benchmark.md")
    p.add_argument("--output-jsonl", type=str, default="")
    args = p.parse_args()

    if args.run_one_json:
        run_one_json(args)
    else:
        parent_main(args)


if __name__ == "__main__":
    main()

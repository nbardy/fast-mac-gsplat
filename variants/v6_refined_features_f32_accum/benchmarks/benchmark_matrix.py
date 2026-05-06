from __future__ import annotations

import argparse
import itertools
import json
import os
import random
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
THIS_BENCH = HERE / "benchmark_mps.py"
STABLE_BENCH = HERE.parents[1] / "v6_refined_features" / "benchmarks" / "benchmark_mps.py"


def _csv_ints(value: str) -> list[int]:
    return [int(part) for part in value.split(",") if part]


def _csv_strs(value: str) -> list[str]:
    return [part for part in value.split(",") if part]


def _command_text(cmd: list[str], env_overrides: dict[str, str]) -> str:
    env_text = " ".join(f"{key}={shlex.quote(value)}" for key, value in sorted(env_overrides.items()))
    cmd_text = " ".join(shlex.quote(part) for part in cmd)
    if not env_text:
        return cmd_text
    return f"{env_text} {cmd_text}"


def _jsonl_write(path: Path | None, row: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


def _last_json_line(stdout: str) -> dict[str, Any]:
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    raise ValueError("benchmark did not emit a JSON line")


def _tail(text: str, limit: int = 4000) -> str:
    if len(text) <= limit:
        return text
    return text[-limit:]


def _run_job(
    *,
    variant: str,
    bench: Path,
    env: dict[str, str],
    env_overrides: dict[str, str],
    cmd: list[str],
    dry_run: bool,
    timeout_s: float,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "variant": variant,
        "benchmark": str(bench),
        "command": cmd,
        "env": env_overrides,
        "status": "dry_run" if dry_run else "pending",
        "runtime_s": 0.0,
    }
    print(f"$ {_command_text(cmd, env_overrides)}", flush=True)
    if dry_run:
        return row

    started = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(HERE),
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        row.update(
            {
                "status": "timeout",
                "runtime_s": time.perf_counter() - started,
                "error": f"timed out after {timeout_s:.1f}s",
                "stdout": _tail(exc.stdout or ""),
                "stderr": _tail(exc.stderr or ""),
            }
        )
        return row

    row["runtime_s"] = time.perf_counter() - started
    row["returncode"] = proc.returncode
    if proc.returncode != 0:
        row.update({"status": "error", "stdout": _tail(proc.stdout), "stderr": _tail(proc.stderr)})
        return row

    try:
        result = _last_json_line(proc.stdout)
    except ValueError as exc:
        row.update({"status": "error", "error": str(exc), "stdout": _tail(proc.stdout), "stderr": _tail(proc.stderr)})
        return row

    row.update({"status": "ok", "result": result, "stderr": _tail(proc.stderr)})
    return row


def _build_command(
    *,
    bench: Path,
    args: argparse.Namespace,
    case: str,
    seed: int,
    batch_size: int,
    strategy: str,
) -> list[str]:
    cmd = [
        sys.executable,
        str(bench),
        "--height",
        str(args.height),
        "--width",
        str(args.width),
        "--gaussians",
        str(args.gaussians),
        "--batch-size",
        str(batch_size),
        "--feature-dim",
        str(args.feature_dim),
        "--batch-strategy",
        strategy,
        "--warmup",
        str(args.warmup),
        "--iters",
        str(args.iters),
        "--case",
        case,
        "--seed",
        str(seed),
        "--json",
    ]
    if args.backward:
        cmd.append("--backward")
    if args.alpha_loss:
        cmd.append("--alpha-loss")
    if args.profile:
        cmd.append("--profile")
    return cmd


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Safe sequential benchmark matrix for the v6_refined_features_f32_accum fork. "
            "Defaults are intentionally small and avoid overflow_stress."
        )
    )
    p.add_argument("--height", type=int, default=128)
    p.add_argument("--width", type=int, default=128)
    p.add_argument("--gaussians", type=int, default=1024)
    p.add_argument("--feature-dim", type=int, default=32)
    p.add_argument("--batch-sizes", type=str, default="1")
    p.add_argument("--strategies", type=str, default="auto")
    p.add_argument("--tile-sizes", type=str, default="16")
    p.add_argument("--chunks", type=str, default="32")
    p.add_argument("--caps", type=str, default="512")
    p.add_argument("--seeds", type=str, default="0")
    p.add_argument("--cases", type=str, default="sparse_sigma_1_5,medium_sigma_3_8")
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--iters", type=int, default=2)
    p.add_argument("--timeout-s", type=float, default=60.0)
    p.add_argument("--backward", action="store_true")
    p.add_argument("--alpha-loss", action="store_true")
    p.add_argument("--profile", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--output-jsonl", type=Path, default=None)
    p.add_argument("--include-stable-baseline", action="store_true")
    p.add_argument("--shuffle-order", action="store_true")
    args = p.parse_args()

    batch_sizes = _csv_ints(args.batch_sizes)
    strategies = _csv_strs(args.strategies)
    tile_sizes = _csv_ints(args.tile_sizes)
    chunks = _csv_ints(args.chunks)
    caps = _csv_ints(args.caps)
    seeds = _csv_ints(args.seeds)
    cases = _csv_strs(args.cases)
    variants = [("f32_accum", THIS_BENCH)]
    if args.include_stable_baseline:
        variants.append(("stable_v6_refined_features", STABLE_BENCH))

    jobs = list(itertools.product(variants, cases, seeds, batch_sizes, strategies, tile_sizes, chunks, caps))
    if args.shuffle_order:
        random.Random(0).shuffle(jobs)

    ok_count = 0
    for (variant, bench), case, seed, batch_size, strategy, tile, chunk, cap in jobs:
        env = os.environ.copy()
        env_overrides = {
            "GSP_TILE_SIZE": str(tile),
            "GSP_CHUNK": str(chunk),
            "GSP_FAST_CAP": str(cap),
        }
        env.update(env_overrides)
        cmd = _build_command(
            bench=bench,
            args=args,
            case=case,
            seed=seed,
            batch_size=batch_size,
            strategy=strategy,
        )
        row = _run_job(
            variant=variant,
            bench=bench,
            env=env,
            env_overrides=env_overrides,
            cmd=cmd,
            dry_run=args.dry_run,
            timeout_s=args.timeout_s,
        )
        row.update(
            {
                "case": case,
                "seed": seed,
                "batch_size": batch_size,
                "strategy": strategy,
                "tile_size": tile,
                "chunk": chunk,
                "cap": cap,
            }
        )
        _jsonl_write(args.output_jsonl, row)
        if row["status"] == "ok":
            ok_count += 1
            result = row["result"]
            print(
                f"{variant:25s} case={case:18s} seed={seed} B={batch_size} "
                f"strat={strategy:7s} tile={tile:2d} chunk={chunk:3d} cap={cap:4d} "
                f"mean={result['mean_ms']:8.3f} fwd={result.get('forward_ms', 0.0):8.3f} "
                f"bwd={result.get('backward_ms', 0.0):8.3f}",
                flush=True,
            )
        else:
            message = row.get("error") or row.get("stderr") or row["status"]
            print(f"{variant:25s} case={case:18s} status={row['status']} {message}", flush=True)

    print(f"completed {len(jobs)} jobs: ok={ok_count} non_ok={len(jobs) - ok_count}", flush=True)


if __name__ == "__main__":
    main()

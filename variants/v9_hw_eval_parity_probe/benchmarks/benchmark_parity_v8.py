from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from torch_gsplat_bridge_v9_hw_eval_parity.parity_v8 import (  # noqa: E402
    direct_width_aligned,
    markdown_report,
    parse_int_list,
    parse_size,
    parse_str_list,
    run_parity_case,
    skipped_row,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare v9 fixed eval RGBA against v8 forward eval RGB.")
    parser.add_argument("--sizes", default="16x16,64x64")
    parser.add_argument("--gaussians", default="1,16")
    parser.add_argument("--cases", default="tiny_single,grid_ordered,overlap_ordered")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--v9-direct", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--jsonl", type=Path)
    parser.add_argument("--markdown", type=Path)
    args = parser.parse_args()

    if not torch.backends.mps.is_available():
        raise SystemExit("MPS is not available; v8/v9 parity requires MPS tensors.")

    rows: list[dict] = []
    sizes = [parse_size(s) for s in parse_str_list(args.sizes)]
    requested_gaussians = parse_int_list(args.gaussians)
    cases = parse_str_list(args.cases)

    for case in cases:
        if case == "tiny_single":
            case_gaussians = (1,)
        elif case == "depth_mismatch":
            case_gaussians = tuple(g for g in requested_gaussians if g > 1) or (2,)
        else:
            case_gaussians = requested_gaussians
        for height, width in sizes:
            for gaussians in case_gaussians:
                if args.v9_direct and not direct_width_aligned(width):
                    row = skipped_row(
                        case=case,
                        height=height,
                        width=width,
                        gaussians=gaussians,
                        seed=args.seed,
                        warmup=args.warmup,
                        iters=args.iters,
                        reason="v9 direct path requires width * 16 bytes to be 256-byte aligned",
                    )
                else:
                    row = run_parity_case(
                        case,
                        height=height,
                        width=width,
                        gaussians=gaussians,
                        seed=args.seed,
                        warmup=args.warmup,
                        iters=args.iters,
                        v9_direct=args.v9_direct,
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

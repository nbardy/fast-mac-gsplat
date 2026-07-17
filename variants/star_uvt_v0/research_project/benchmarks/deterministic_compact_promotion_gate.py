from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


SCRIPT_PATH = Path(__file__).resolve()
VARIANT_ROOT = SCRIPT_PATH.parents[2]
HARNESS = VARIANT_ROOT / "research_project" / "trainer_harness"
for _path in (VARIANT_ROOT, HARNESS):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

try:
    from research_project.trainer_harness.tile_metal_autograd import (
        BACKWARD_POLICY_NAMES,
        DETERMINISTIC_COMPACT_BACKWARD_POLICY,
        KNOWN_NONDETERMINISTIC_COMPACT_SAMPLE_EMISSION_MODES,
        deterministic_compact_backward_cli_args,
        resolve_deterministic_compact_backward_policy,
        resolve_backward_policy,
        validate_deterministic_compact_backward_modes,
        validate_backward_policy,
    )
except ImportError:  # pragma: no cover - direct script execution fallback.
    from tile_metal_autograd import (
        BACKWARD_POLICY_NAMES,
        DETERMINISTIC_COMPACT_BACKWARD_POLICY,
        KNOWN_NONDETERMINISTIC_COMPACT_SAMPLE_EMISSION_MODES,
        deterministic_compact_backward_cli_args,
        resolve_deterministic_compact_backward_policy,
        resolve_backward_policy,
        validate_deterministic_compact_backward_modes,
        validate_backward_policy,
    )


def _collect_repeatability_max_abs(value: Any) -> list[float]:
    found: list[float] = []
    if isinstance(value, dict):
        max_abs = value.get("max_abs")
        if isinstance(max_abs, (int, float)):
            found.append(float(max_abs))
        for key, child in value.items():
            if key in {"grad_delta_vs_first", "delta_vs_first"} and isinstance(child, list):
                for row in child:
                    if isinstance(row, dict) and isinstance(row.get("max_abs"), (int, float)):
                        found.append(float(row["max_abs"]))
            found.extend(_collect_repeatability_max_abs(child))
    elif isinstance(value, list):
        for child in value:
            found.extend(_collect_repeatability_max_abs(child))
    return found


def run_gate(args: argparse.Namespace) -> dict[str, Any]:
    policy = resolve_backward_policy(args.policy)
    failures: list[str] = []
    compact_policy = resolve_deterministic_compact_backward_policy(DETERMINISTIC_COMPACT_BACKWARD_POLICY)
    try:
        validate_backward_policy(
            policy,
            require_deterministic=args.require_deterministic,
            require_compact=args.require_compact,
            require_promotion_contract=args.require_promotion_contract,
        )
    except ValueError as exc:
        failures.append(str(exc))
    try:
        validate_deterministic_compact_backward_modes(policy.reduction_mode, policy.sample_emission_mode)
    except ValueError as exc:
        failures.append(f"policy {policy.name} is not deterministic-compact promotable: {exc}")
    if policy.name == "deterministic_compact":
        if policy.sample_emission_mode != compact_policy["sample_emission_mode"]:
            failures.append(
                f"deterministic_compact sample mode {policy.sample_emission_mode} does not match "
                f"{compact_policy['sample_emission_mode']}"
            )
        if policy.reduction_mode != compact_policy["reduction_mode"]:
            failures.append(
                f"deterministic_compact reduction mode {policy.reduction_mode} does not match "
                f"{compact_policy['reduction_mode']}"
            )

    rejected_modes: dict[str, str] = {}
    for sample_emission_mode in KNOWN_NONDETERMINISTIC_COMPACT_SAMPLE_EMISSION_MODES:
        try:
            validate_deterministic_compact_backward_modes("index_add", sample_emission_mode)
        except ValueError as exc:
            rejected_modes[sample_emission_mode] = str(exc)
        else:
            failures.append(f"{sample_emission_mode} was not rejected by deterministic compact validation")

    rejected_pairings: dict[str, str] = {}
    for reduction_mode, sample_emission_mode in (
        ("index_add", "tile_pair"),
        ("key_sort_scan_metal", "tile_pair_reduced"),
        ("key_sort_scan_metal", "direct_fixedpoint"),
    ):
        key = f"{reduction_mode}+{sample_emission_mode}"
        try:
            validate_deterministic_compact_backward_modes(reduction_mode, sample_emission_mode)
        except ValueError as exc:
            rejected_pairings[key] = str(exc)
        else:
            failures.append(f"{key} was not rejected by deterministic compact validation")

    quality: dict[str, Any] = {"status": "not_loaded"}
    if args.quality_json is not None:
        data = json.loads(args.quality_json.read_text())
        meta = data.get("meta", {}) if isinstance(data, dict) else {}
        quality_reduction_mode = meta.get("uvt_reduction_mode") if isinstance(meta, dict) else None
        quality_sample_emission_mode = meta.get("uvt_sample_emission_mode") if isinstance(meta, dict) else None
        repeatability = _collect_repeatability_max_abs(data)
        max_repeatability_delta = max(repeatability) if repeatability else None
        quality = {
            "status": "loaded",
            "path": str(args.quality_json),
            "reduction_mode": quality_reduction_mode,
            "sample_emission_mode": quality_sample_emission_mode,
            "repeatability_delta_count": len(repeatability),
            "max_repeatability_delta": max_repeatability_delta,
        }
        if quality_reduction_mode is not None and quality_reduction_mode != policy.reduction_mode:
            failures.append(
                f"quality JSON reduction mode {quality_reduction_mode} does not match policy {policy.reduction_mode}"
            )
        if quality_sample_emission_mode is not None and quality_sample_emission_mode != policy.sample_emission_mode:
            failures.append(
                "quality JSON sample emission mode "
                f"{quality_sample_emission_mode} does not match policy {policy.sample_emission_mode}"
            )
        if args.max_repeatability_delta is not None:
            if max_repeatability_delta is None:
                failures.append("quality JSON did not contain repeatability max_abs deltas")
            elif max_repeatability_delta > args.max_repeatability_delta:
                failures.append(
                    f"repeatability delta {max_repeatability_delta:.9g} exceeds "
                    f"{args.max_repeatability_delta:.9g}"
                )
    elif args.max_repeatability_delta is not None:
        failures.append("--max-repeatability-delta requires --quality-json")

    return {
        "benchmark": "deterministic_compact_promotion_gate",
        "policy": policy.as_dict(),
        "deterministic_compact_policy": compact_policy,
        "deterministic_compact_cli_args": list(deterministic_compact_backward_cli_args()),
        "requirements": {
            "deterministic": bool(args.require_deterministic),
            "compact": bool(args.require_compact),
            "promotion_contract": bool(args.require_promotion_contract),
            "max_repeatability_delta": args.max_repeatability_delta,
        },
        "quality": quality,
        "rejected_nondeterministic_modes": rejected_modes,
        "rejected_invalid_pairings": rejected_pairings,
        "failures": failures,
        "pass": not failures,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Static STAR backward-policy promotion gate.")
    parser.add_argument("--policy", choices=BACKWARD_POLICY_NAMES, default="deterministic_compact")
    parser.add_argument("--require-deterministic", action="store_true")
    parser.add_argument("--require-compact", action="store_true")
    parser.add_argument("--require-promotion-contract", action="store_true")
    parser.add_argument("--quality-json", type=Path)
    parser.add_argument("--max-repeatability-delta", type=float)
    parser.add_argument("--out-json", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_gate(args)
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(text)
    print(text, end="")
    if not result["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

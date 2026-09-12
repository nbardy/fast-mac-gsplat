from __future__ import annotations

from dataclasses import dataclass
import math
import sys
from pathlib import Path

import torch
from torch import Tensor


@dataclass(frozen=True)
class BackwardPolicy:
    name: str
    sample_emission_mode: str
    reduction_mode: str
    deterministic: bool
    compact: bool
    promotion_contract: bool
    note: str

    def as_dict(self) -> dict[str, bool | str]:
        return {
            "name": self.name,
            "sample_emission_mode": self.sample_emission_mode,
            "reduction_mode": self.reduction_mode,
            "deterministic": self.deterministic,
            "compact": self.compact,
            "promotion_contract": self.promotion_contract,
            "note": self.note,
        }


BACKWARD_POLICIES: dict[str, BackwardPolicy] = {
    "deterministic_quality": BackwardPolicy(
        name="deterministic_quality",
        sample_emission_mode="tile_pair",
        reduction_mode="key_sort_scan_metal",
        deterministic=True,
        compact=False,
        promotion_contract=True,
        note="Exact tile-pair sample emission with keyed deterministic reduction; quality reference for static STAR promotion.",
    ),
    "deterministic_compact": BackwardPolicy(
        name="deterministic_compact",
        sample_emission_mode="tile_pair",
        reduction_mode="key_sort_scan_metal",
        deterministic=True,
        compact=True,
        promotion_contract=True,
        note="Zero-pruned tile-pair emission with keyed deterministic reduction; current static STAR promotion gate.",
    ),
    "deterministic_suffix_segmented": BackwardPolicy(
        name="deterministic_suffix_segmented",
        sample_emission_mode="tile_pair_suffix",
        reduction_mode="key_sort_segmented_metal",
        deterministic=True,
        compact=True,
        promotion_contract=False,
        note="Suffix tile-pair emission with keyed segmented deterministic reduction; comparison candidate, not the current gate.",
    ),
    "deterministic_sharedsort": BackwardPolicy(
        name="deterministic_sharedsort",
        sample_emission_mode="tile_pair_sharedsort",
        reduction_mode="key_sort_scan_metal",
        deterministic=True,
        compact=False,
        promotion_contract=False,
        note="Shared-sort diagnostic path for reducer comparisons; deterministic but not the compact promotion target.",
    ),
    "fast_exploration": BackwardPolicy(
        name="fast_exploration",
        sample_emission_mode="direct_atomic",
        reduction_mode="index_add",
        deterministic=False,
        compact=True,
        promotion_contract=False,
        note="Fast direct atomic exploration path; useful for throughput probes, not promotable.",
    ),
}

BACKWARD_POLICY_NAMES = tuple(BACKWARD_POLICIES)


def resolve_backward_policy(name: str) -> BackwardPolicy:
    try:
        return BACKWARD_POLICIES[name]
    except KeyError as exc:
        valid = ", ".join(BACKWARD_POLICY_NAMES)
        raise ValueError(f"unknown backward policy {name!r}; expected one of: {valid}") from exc


def validate_backward_policy(
    policy: BackwardPolicy,
    *,
    require_deterministic: bool = False,
    require_compact: bool = False,
    require_promotion_contract: bool = False,
) -> None:
    if require_deterministic and not policy.deterministic:
        raise ValueError(f"{policy.name} is not deterministic")
    if require_compact and not policy.compact:
        raise ValueError(f"{policy.name} is not compact")
    if require_promotion_contract and not policy.promotion_contract:
        raise ValueError(f"{policy.name} does not satisfy the static STAR promotion contract")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from torch_gsplat_bridge_star_uvt import (  # noqa: E402
    ProjectiveTraceCellAtlasBudgetReport,
    ProjectiveTraceCellAtlasComplexityStats,
    ProjectiveTraceCellAtlasCoverageReport,
    ProjectiveTraceCellAtlasFallbackStats,
    ProjectiveTraceCellAtlasSupportMarginReport,
    ProjectiveTraceCellAtlasVisibilityReport,
    UVTRenderConfig,
    direct_atomic_backward,
    direct_atomic_backward_gated,
    direct_backward_projective_trace_cell_interval_atlas_metal,
    direct_fixedpoint_backward,
    direct_split_fixedpoint_backward,
    eval_projective_trace_cell_torch,
    ProjectiveTraceCellTraceAtlas,
    direct_serial_backward,
    mark_projective_trace_cell_visibility_fallbacks,
    pack_projective_trace_tile_time_bins,
    projective_trace_cell_atlas_budget_report,
    projective_trace_cell_atlas_complexity_stats,
    projective_trace_cell_atlas_coverage_report,
    projective_trace_cell_atlas_fallback_stats,
    projective_trace_cell_atlas_fallback_tile_sample_mask,
    projective_trace_cell_atlas_support_margin_report,
    projective_trace_cell_atlas_visibility_report,
    rebin_projective_trace_cell_atlas_support_events,
    render_projective_trace_cell_atlas_reference,
    render_projective_trace_cell_interval_atlas_metal,
    split_projective_trace_cell_atlas_fallback_cells,
    stratify_projective_trace_cell_atlas_visibility_events,
    stratify_projective_trace_cell_atlas_visibility,
    tile_pair_atomic_backward,
    tile_pair_fixedpoint_backward,
    reduce_sample_bundle_scan,
    reduce_sample_bundle_scan_compensated,
    reduce_sample_bundle_sorted_segments,
    render_uvt_tubes,
    render_uvt_tubes_gated,
    stable_backward_samples,
    stable_backward_samples_with_keys,
    tile_pair_backward_samples,
    tile_pair_backward_samples_compensated,
    tile_pair_grouped_backward_samples,
    tile_pair_parallel_backward_samples,
    tile_pair_scanline_backward_samples,
    tile_pair_sharedsort_backward_samples,
    tile_pair_suffix_backward_samples,
    tile_pair_suffix_reduced_backward,
    tile_pair_target_bounds_backward_samples,
    tile_pair_reduced_backward,
    tile_pair_reduced_parallel_backward,
)

KEYED_REDUCTION_MODES = (
    "key_sort_scan_metal",
    "key_sort_compensated_scan_metal",
    "key_sort_segmented_metal",
)
KEYED_SAMPLE_EMISSION_MODES = (
    "with_keys",
    "tile_pair",
    "tile_pair_compensated",
    "tile_pair_grouped",
    "tile_pair_parallel",
    "tile_pair_scanline",
    "tile_pair_sharedsort",
    "tile_pair_target_bounds",
    "tile_pair_suffix",
)
DIRECT_REDUCED_SAMPLE_EMISSION_MODES = (
    "direct_atomic",
    "direct_fixedpoint",
    "direct_split_fixedpoint",
    "direct_serial",
    "tile_pair_atomic",
    "tile_pair_fixedpoint",
    "tile_pair_reduced",
    "tile_pair_reduced_parallel",
    "tile_pair_suffix_reduced",
)
SAMPLE_REDUCTION_MODES = (
    "index_add",
    "sorted_cpu",
    "scan_metal",
    "compensated_scan_metal",
    "sort_scan_metal",
    "sort_compensated_scan_metal",
    *KEYED_REDUCTION_MODES,
)
SAMPLE_EMISSION_MODES = (
    "atomic_append",
    *KEYED_SAMPLE_EMISSION_MODES,
    *DIRECT_REDUCED_SAMPLE_EMISSION_MODES,
)

KNOWN_NONDETERMINISTIC_COMPACT_SAMPLE_EMISSION_MODES = (
    "atomic_append",
    "direct_atomic",
    "tile_pair_atomic",
)

DETERMINISTIC_COMPACT_BACKWARD_POLICY = "deterministic_compact_tile_pair_zero_prune_key_sort"
DETERMINISTIC_COMPACT_BACKWARD_PRESETS = {
    DETERMINISTIC_COMPACT_BACKWARD_POLICY: {
        "sample_emission_mode": "tile_pair",
        "reduction_mode": "key_sort_scan_metal",
        "status": "promotion_gate",
        "evidence": "active zero-pruned tile-pair quality branch with keyed deterministic reduction",
    },
    "deterministic_compact_tile_pair_reduced": {
        "sample_emission_mode": "tile_pair_reduced",
        "reduction_mode": "index_add",
        "status": "exact_candidate_not_default",
        "evidence": "exact parity and repeatability, but current full trainer row is slower",
    },
    "deterministic_compact_tile_pair_sharedsort_key_sort": {
        "sample_emission_mode": "tile_pair_sharedsort",
        "reduction_mode": "key_sort_scan_metal",
        "status": "comparison_candidate",
        "evidence": "tile-local shared sort preserves tile-pair semantics but missed current quality/timing gate",
    },
    "deterministic_compact_tile_pair_suffix_key_sort": {
        "sample_emission_mode": "tile_pair_suffix",
        "reduction_mode": "key_sort_scan_metal",
        "status": "comparison_candidate",
        "evidence": "suffix path is repeatable and faster than old tile-pair but missed quality promotion",
    },
}
DETERMINISTIC_COMPACT_BACKWARD_ALIASES = {
    "default": DETERMINISTIC_COMPACT_BACKWARD_POLICY,
    "deterministic_compact": DETERMINISTIC_COMPACT_BACKWARD_POLICY,
    "deterministic_quality": DETERMINISTIC_COMPACT_BACKWARD_POLICY,
    "promotion": DETERMINISTIC_COMPACT_BACKWARD_POLICY,
    "quality": DETERMINISTIC_COMPACT_BACKWARD_POLICY,
    "tile_pair_zero_prune_key_sort": DETERMINISTIC_COMPACT_BACKWARD_POLICY,
    "tile_pair_reduced": "deterministic_compact_tile_pair_reduced",
    "tile_pair_sharedsort": "deterministic_compact_tile_pair_sharedsort_key_sort",
    "tile_pair_suffix": "deterministic_compact_tile_pair_suffix_key_sort",
}


def validate_deterministic_compact_backward_modes(reduction_mode: str, sample_emission_mode: str) -> None:
    if reduction_mode not in SAMPLE_REDUCTION_MODES:
        raise ValueError(f"unknown compact reduction mode: {reduction_mode}")
    if sample_emission_mode not in SAMPLE_EMISSION_MODES:
        raise ValueError(f"unknown compact sample emission mode: {sample_emission_mode}")
    if sample_emission_mode in KNOWN_NONDETERMINISTIC_COMPACT_SAMPLE_EMISSION_MODES:
        raise ValueError(f"{sample_emission_mode} is not accepted for deterministic compact promotion")
    if reduction_mode == "index_add" and sample_emission_mode not in DIRECT_REDUCED_SAMPLE_EMISSION_MODES:
        raise ValueError("index_add over compact sample rows is not accepted for deterministic compact promotion")
    if reduction_mode in KEYED_REDUCTION_MODES and sample_emission_mode not in KEYED_SAMPLE_EMISSION_MODES:
        raise ValueError(
            "keyed sort reduction requires a keyed sample emission mode "
            f"(got sample_emission_mode={sample_emission_mode})"
        )
    if sample_emission_mode in DIRECT_REDUCED_SAMPLE_EMISSION_MODES and reduction_mode != "index_add":
        raise ValueError(f"{sample_emission_mode} bypasses the reducer and requires reduction_mode=index_add")


def resolve_deterministic_compact_backward_policy(preset: str | None = None) -> dict[str, str]:
    policy_name = preset or DETERMINISTIC_COMPACT_BACKWARD_POLICY
    policy_name = DETERMINISTIC_COMPACT_BACKWARD_ALIASES.get(policy_name, policy_name)
    if policy_name not in DETERMINISTIC_COMPACT_BACKWARD_PRESETS:
        known = sorted((*DETERMINISTIC_COMPACT_BACKWARD_PRESETS, *DETERMINISTIC_COMPACT_BACKWARD_ALIASES))
        raise ValueError(f"unknown deterministic compact backward preset {preset!r}; known presets: {', '.join(known)}")
    policy = DETERMINISTIC_COMPACT_BACKWARD_PRESETS[policy_name]
    reduction_mode = policy["reduction_mode"]
    sample_emission_mode = policy["sample_emission_mode"]
    validate_deterministic_compact_backward_modes(reduction_mode, sample_emission_mode)
    return {
        "preset": policy_name,
        "reduction_mode": reduction_mode,
        "sample_emission_mode": sample_emission_mode,
        "status": policy["status"],
        "evidence": policy["evidence"],
    }


def deterministic_compact_backward_cli_args(preset: str | None = None) -> tuple[str, str, str, str]:
    policy = resolve_deterministic_compact_backward_policy(preset)
    return (
        "--uvt-reduction-mode",
        policy["reduction_mode"],
        "--uvt-sample-emission-mode",
        policy["sample_emission_mode"],
    )


def _reduce_samples(ids: Tensor, samples: Tensor, tube_count: int, trailing: int | None) -> Tensor:
    flat_ids = ids.reshape(-1)
    if trailing is None:
        sample_rows = samples.reshape(-1)
    else:
        sample_rows = samples.reshape(-1, trailing)
    if sample_rows.shape[0] != flat_ids.shape[0]:
        shared = min(int(sample_rows.shape[0]), int(flat_ids.shape[0]))
        flat_ids = flat_ids[:shared]
        sample_rows = sample_rows[:shared]
    valid_positions = torch.nonzero((flat_ids >= 0) & (flat_ids < tube_count), as_tuple=False).flatten()
    valid_ids = flat_ids.index_select(0, valid_positions).to(torch.int64)
    valid_samples = sample_rows.index_select(0, valid_positions)
    if trailing is None:
        out = torch.zeros((tube_count,), dtype=torch.float32, device=samples.device)
        out.index_add_(0, valid_ids, valid_samples)
        return out
    out = torch.zeros((tube_count, trailing), dtype=torch.float32, device=samples.device)
    out.index_add_(0, valid_ids, valid_samples)
    return out


def _reduce_sample_bundle_index_add(
    ids: Tensor,
    grad_ma_samples: Tensor,
    grad_q_samples: Tensor,
    grad_opacity_samples: Tensor,
    grad_color_samples: Tensor,
    tube_count: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    flat_ids = ids.reshape(-1).to(torch.int64)
    grad_ma_rows = grad_ma_samples.reshape(-1, 3)
    grad_q_rows = grad_q_samples.reshape(-1, 6)
    grad_opacity_rows = grad_opacity_samples.reshape(-1, 1)
    grad_color_rows = grad_color_samples.reshape(-1, 3)
    if not (
        flat_ids.shape[0]
        == grad_ma_rows.shape[0]
        == grad_q_rows.shape[0]
        == grad_opacity_rows.shape[0]
        == grad_color_rows.shape[0]
    ):
        raise ValueError("compact sample ids and gradient rows must have the same length")
    valid_positions = torch.nonzero((flat_ids >= 0) & (flat_ids < tube_count), as_tuple=False).flatten()
    flat_ids = flat_ids.index_select(0, valid_positions)
    sample_rows = torch.cat(
        (
            grad_ma_rows,
            grad_q_rows,
            grad_opacity_rows,
            grad_color_rows,
        ),
        dim=-1,
    ).index_select(0, valid_positions)
    out = torch.zeros((tube_count, 13), dtype=torch.float32, device=grad_ma_samples.device)
    out.index_add_(0, flat_ids, sample_rows)
    return out[:, :3], out[:, 3:9], out[:, 9], out[:, 10:13]


def _reduce_sample_bundle_sorted_cpu(
    ids: Tensor,
    grad_ma_samples: Tensor,
    grad_q_samples: Tensor,
    grad_opacity_samples: Tensor,
    grad_color_samples: Tensor,
    tube_count: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    device = grad_ma_samples.device
    flat_ids = ids.reshape(-1).detach().cpu().to(torch.int64)
    grad_ma_rows = grad_ma_samples.reshape(-1, 3).detach().cpu().to(torch.float32)
    grad_q_rows = grad_q_samples.reshape(-1, 6).detach().cpu().to(torch.float32)
    grad_opacity_rows = grad_opacity_samples.reshape(-1, 1).detach().cpu().to(torch.float32)
    grad_color_rows = grad_color_samples.reshape(-1, 3).detach().cpu().to(torch.float32)
    if not (
        flat_ids.shape[0]
        == grad_ma_rows.shape[0]
        == grad_q_rows.shape[0]
        == grad_opacity_rows.shape[0]
        == grad_color_rows.shape[0]
    ):
        raise ValueError("compact sample ids and gradient rows must have the same length")
    valid_positions = torch.nonzero((flat_ids >= 0) & (flat_ids < tube_count), as_tuple=False).flatten()
    flat_ids = flat_ids.index_select(0, valid_positions)
    sample_rows = torch.cat(
        (
            grad_ma_rows,
            grad_q_rows,
            grad_opacity_rows,
            grad_color_rows,
        ),
        dim=-1,
    ).index_select(0, valid_positions)
    out = torch.zeros((tube_count, 13), dtype=torch.float32)
    if flat_ids.numel() == 0:
        return out[:, :3].to(device), out[:, 3:9].to(device), out[:, 9].to(device), out[:, 10:13].to(device)

    order = torch.argsort(flat_ids, stable=True)
    sorted_ids = flat_ids.index_select(0, order)
    sorted_rows = sample_rows.index_select(0, order)
    unique_ids, counts = torch.unique_consecutive(sorted_ids, return_counts=True)
    start = 0
    for tube_id, count in zip(unique_ids.tolist(), counts.tolist(), strict=True):
        end = start + int(count)
        out[int(tube_id)] = sorted_rows[start:end].to(torch.float64).sum(dim=0).to(torch.float32)
        start = end
    return out[:, :3].to(device), out[:, 3:9].to(device), out[:, 9].to(device), out[:, 10:13].to(device)


def _reduce_sample_bundle_scan_metal(
    ids: Tensor,
    grad_ma_samples: Tensor,
    grad_q_samples: Tensor,
    grad_opacity_samples: Tensor,
    grad_color_samples: Tensor,
    tube_count: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    return reduce_sample_bundle_scan(
        ids,
        grad_ma_samples,
        grad_q_samples,
        grad_opacity_samples,
        grad_color_samples,
        tube_count,
    )


def _reduce_sample_bundle_compensated_scan_metal(
    ids: Tensor,
    grad_ma_samples: Tensor,
    grad_q_samples: Tensor,
    grad_opacity_samples: Tensor,
    grad_color_samples: Tensor,
    tube_count: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    return reduce_sample_bundle_scan_compensated(
        ids,
        grad_ma_samples,
        grad_q_samples,
        grad_opacity_samples,
        grad_color_samples,
        tube_count,
    )


def _reduce_sample_bundle_sort_scan_metal(
    ids: Tensor,
    grad_ma_samples: Tensor,
    grad_q_samples: Tensor,
    grad_opacity_samples: Tensor,
    grad_color_samples: Tensor,
    tube_count: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    order = torch.argsort(ids.reshape(-1), stable=True)
    return reduce_sample_bundle_scan(
        ids.reshape(-1).index_select(0, order).contiguous(),
        grad_ma_samples.reshape(-1, 3).index_select(0, order).contiguous(),
        grad_q_samples.reshape(-1, 6).index_select(0, order).contiguous(),
        grad_opacity_samples.reshape(-1).index_select(0, order).contiguous(),
        grad_color_samples.reshape(-1, 3).index_select(0, order).contiguous(),
        tube_count,
    )


def _reduce_sample_bundle_sort_compensated_scan_metal(
    ids: Tensor,
    grad_ma_samples: Tensor,
    grad_q_samples: Tensor,
    grad_opacity_samples: Tensor,
    grad_color_samples: Tensor,
    tube_count: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    order = torch.argsort(ids.reshape(-1), stable=True)
    return reduce_sample_bundle_scan_compensated(
        ids.reshape(-1).index_select(0, order).contiguous(),
        grad_ma_samples.reshape(-1, 3).index_select(0, order).contiguous(),
        grad_q_samples.reshape(-1, 6).index_select(0, order).contiguous(),
        grad_opacity_samples.reshape(-1).index_select(0, order).contiguous(),
        grad_color_samples.reshape(-1, 3).index_select(0, order).contiguous(),
        tube_count,
    )


def _reduce_sample_bundle_key_sort_scan_metal(
    ids: Tensor,
    keys: Tensor | None,
    grad_ma_samples: Tensor,
    grad_q_samples: Tensor,
    grad_opacity_samples: Tensor,
    grad_color_samples: Tensor,
    tube_count: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    if keys is None:
        raise ValueError("key_sort_scan_metal requires sample keys")
    flat_ids = ids.reshape(-1)
    flat_keys = keys.reshape(-1)
    valid_positions = torch.nonzero((flat_ids >= 0) & (flat_ids < tube_count), as_tuple=False).flatten()
    valid_ids = flat_ids.index_select(0, valid_positions)
    valid_keys = flat_keys.index_select(0, valid_positions)
    combined = valid_ids.to(torch.int64) * 2147483648 + valid_keys.to(torch.int64)
    order = torch.argsort(combined, stable=True)
    ordered_positions = valid_positions.index_select(0, order)
    return reduce_sample_bundle_scan(
        flat_ids.index_select(0, ordered_positions).contiguous(),
        grad_ma_samples.reshape(-1, 3).index_select(0, ordered_positions).contiguous(),
        grad_q_samples.reshape(-1, 6).index_select(0, ordered_positions).contiguous(),
        grad_opacity_samples.reshape(-1).index_select(0, ordered_positions).contiguous(),
        grad_color_samples.reshape(-1, 3).index_select(0, ordered_positions).contiguous(),
        tube_count,
    )


def _reduce_sample_bundle_key_sort_compensated_scan_metal(
    ids: Tensor,
    keys: Tensor | None,
    grad_ma_samples: Tensor,
    grad_q_samples: Tensor,
    grad_opacity_samples: Tensor,
    grad_color_samples: Tensor,
    tube_count: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    if keys is None:
        raise ValueError("key_sort_compensated_scan_metal requires sample keys")
    flat_ids = ids.reshape(-1)
    flat_keys = keys.reshape(-1)
    valid_positions = torch.nonzero((flat_ids >= 0) & (flat_ids < tube_count), as_tuple=False).flatten()
    valid_ids = flat_ids.index_select(0, valid_positions)
    valid_keys = flat_keys.index_select(0, valid_positions)
    combined = valid_ids.to(torch.int64) * 2147483648 + valid_keys.to(torch.int64)
    order = torch.argsort(combined, stable=True)
    ordered_positions = valid_positions.index_select(0, order)
    return reduce_sample_bundle_scan_compensated(
        flat_ids.index_select(0, ordered_positions).contiguous(),
        grad_ma_samples.reshape(-1, 3).index_select(0, ordered_positions).contiguous(),
        grad_q_samples.reshape(-1, 6).index_select(0, ordered_positions).contiguous(),
        grad_opacity_samples.reshape(-1).index_select(0, ordered_positions).contiguous(),
        grad_color_samples.reshape(-1, 3).index_select(0, ordered_positions).contiguous(),
        tube_count,
    )


def _reduce_sample_bundle_key_sort_segmented_metal(
    ids: Tensor,
    keys: Tensor | None,
    grad_ma_samples: Tensor,
    grad_q_samples: Tensor,
    grad_opacity_samples: Tensor,
    grad_color_samples: Tensor,
    tube_count: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    if keys is None:
        raise ValueError("key_sort_segmented_metal requires sample keys")
    flat_ids = ids.reshape(-1)
    flat_keys = keys.reshape(-1)
    valid_positions = torch.nonzero((flat_ids >= 0) & (flat_ids < tube_count), as_tuple=False).flatten()
    valid_ids = flat_ids.index_select(0, valid_positions)
    valid_keys = flat_keys.index_select(0, valid_positions)
    combined = valid_ids.to(torch.int64) * 2147483648 + valid_keys.to(torch.int64)
    order = torch.argsort(combined, stable=True)
    ordered_positions = valid_positions.index_select(0, order)
    return reduce_sample_bundle_sorted_segments(
        flat_ids.index_select(0, ordered_positions).contiguous(),
        grad_ma_samples.reshape(-1, 3).index_select(0, ordered_positions).contiguous(),
        grad_q_samples.reshape(-1, 6).index_select(0, ordered_positions).contiguous(),
        grad_opacity_samples.reshape(-1).index_select(0, ordered_positions).contiguous(),
        grad_color_samples.reshape(-1, 3).index_select(0, ordered_positions).contiguous(),
        tube_count,
    )


def _reduce_sample_bundle(
    ids: Tensor,
    grad_ma_samples: Tensor,
    grad_q_samples: Tensor,
    grad_opacity_samples: Tensor,
    grad_color_samples: Tensor,
    tube_count: int,
    *,
    mode: str = "index_add",
    keys: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    if mode == "index_add":
        return _reduce_sample_bundle_index_add(
            ids,
            grad_ma_samples,
            grad_q_samples,
            grad_opacity_samples,
            grad_color_samples,
            tube_count,
        )
    if mode == "sorted_cpu":
        return _reduce_sample_bundle_sorted_cpu(
            ids,
            grad_ma_samples,
            grad_q_samples,
            grad_opacity_samples,
            grad_color_samples,
            tube_count,
        )
    if mode == "scan_metal":
        return _reduce_sample_bundle_scan_metal(
            ids,
            grad_ma_samples,
            grad_q_samples,
            grad_opacity_samples,
            grad_color_samples,
            tube_count,
        )
    if mode == "compensated_scan_metal":
        return _reduce_sample_bundle_compensated_scan_metal(
            ids,
            grad_ma_samples,
            grad_q_samples,
            grad_opacity_samples,
            grad_color_samples,
            tube_count,
        )
    if mode == "sort_scan_metal":
        return _reduce_sample_bundle_sort_scan_metal(
            ids,
            grad_ma_samples,
            grad_q_samples,
            grad_opacity_samples,
            grad_color_samples,
            tube_count,
        )
    if mode == "sort_compensated_scan_metal":
        return _reduce_sample_bundle_sort_compensated_scan_metal(
            ids,
            grad_ma_samples,
            grad_q_samples,
            grad_opacity_samples,
            grad_color_samples,
            tube_count,
        )
    if mode == "key_sort_scan_metal":
        return _reduce_sample_bundle_key_sort_scan_metal(
            ids,
            keys,
            grad_ma_samples,
            grad_q_samples,
            grad_opacity_samples,
            grad_color_samples,
            tube_count,
        )
    if mode == "key_sort_compensated_scan_metal":
        return _reduce_sample_bundle_key_sort_compensated_scan_metal(
            ids,
            keys,
            grad_ma_samples,
            grad_q_samples,
            grad_opacity_samples,
            grad_color_samples,
            tube_count,
        )
    if mode == "key_sort_segmented_metal":
        return _reduce_sample_bundle_key_sort_segmented_metal(
            ids,
            keys,
            grad_ma_samples,
            grad_q_samples,
            grad_opacity_samples,
            grad_color_samples,
            tube_count,
        )
    raise ValueError(
        "reduction mode must be one of: index_add, sorted_cpu, scan_metal, compensated_scan_metal, sort_scan_metal, sort_compensated_scan_metal, key_sort_scan_metal, key_sort_compensated_scan_metal, key_sort_segmented_metal"
    )


class _MetalTileBackward(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        ma: Tensor,
        q_uvt: Tensor,
        depth0: Tensor,
        depth_beta: Tensor,
        opacity: Tensor,
        color: Tensor,
        config: UVTRenderConfig,
        reduction_mode: str,
        sample_emission_mode: str,
    ) -> Tensor:
        ctx.config = config
        ctx.reduction_mode = reduction_mode
        ctx.sample_emission_mode = sample_emission_mode
        ctx.tube_count = int(ma.shape[0])
        ctx.save_for_backward(ma, q_uvt, depth0, depth_beta, opacity, color)
        return render_uvt_tubes(ma, q_uvt, depth0, depth_beta, opacity, color, config)

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> tuple[Tensor | None, ...]:
        ma, q_uvt, depth0, depth_beta, opacity, color = ctx.saved_tensors
        keys = None
        if ctx.sample_emission_mode in (
            "direct_atomic",
            "direct_fixedpoint",
            "direct_split_fixedpoint",
            "direct_serial",
            "tile_pair_atomic",
            "tile_pair_fixedpoint",
            "tile_pair_reduced",
            "tile_pair_reduced_parallel",
            "tile_pair_suffix_reduced",
        ):
            direct_backward = {
                "direct_atomic": direct_atomic_backward,
                "direct_fixedpoint": direct_fixedpoint_backward,
                "direct_split_fixedpoint": direct_split_fixedpoint_backward,
                "direct_serial": direct_serial_backward,
                "tile_pair_atomic": tile_pair_atomic_backward,
                "tile_pair_fixedpoint": tile_pair_fixedpoint_backward,
                "tile_pair_reduced": tile_pair_reduced_backward,
                "tile_pair_reduced_parallel": tile_pair_reduced_parallel_backward,
                "tile_pair_suffix_reduced": tile_pair_suffix_reduced_backward,
            }[ctx.sample_emission_mode]
            grad_ma, grad_q, grad_opacity, grad_color, _tile_unstable = direct_backward(
                ma.detach(),
                q_uvt.detach(),
                depth0.detach(),
                depth_beta.detach(),
                opacity.detach(),
                color.detach(),
                grad_output.contiguous(),
                ctx.config,
            )
            grad_depth0 = torch.zeros_like(depth0)
            grad_depth_beta = torch.zeros_like(depth_beta)
            return grad_ma, grad_q, grad_depth0, grad_depth_beta, grad_opacity, grad_color, None, None, None
        if ctx.sample_emission_mode == "with_keys":
            ids, grad_ma_samples, grad_q_samples, grad_opacity_samples, grad_color_samples, keys, _tile_unstable = (
                stable_backward_samples_with_keys(
                    ma.detach(),
                    q_uvt.detach(),
                    depth0.detach(),
                    depth_beta.detach(),
                    opacity.detach(),
                    color.detach(),
                    grad_output.contiguous(),
                    ctx.config,
                )
            )
        elif ctx.sample_emission_mode == "atomic_append":
            ids, grad_ma_samples, grad_q_samples, grad_opacity_samples, grad_color_samples, _tile_unstable = stable_backward_samples(
                ma.detach(),
                q_uvt.detach(),
                depth0.detach(),
                depth_beta.detach(),
                opacity.detach(),
                color.detach(),
                grad_output.contiguous(),
                ctx.config,
            )
        elif ctx.sample_emission_mode in (
            "tile_pair",
            "tile_pair_compensated",
            "tile_pair_grouped",
            "tile_pair_parallel",
            "tile_pair_scanline",
            "tile_pair_sharedsort",
            "tile_pair_target_bounds",
            "tile_pair_suffix",
        ):
            tile_pair_fn = {
                "tile_pair": tile_pair_backward_samples,
                "tile_pair_compensated": tile_pair_backward_samples_compensated,
                "tile_pair_grouped": tile_pair_grouped_backward_samples,
                "tile_pair_parallel": tile_pair_parallel_backward_samples,
                "tile_pair_scanline": tile_pair_scanline_backward_samples,
                "tile_pair_sharedsort": tile_pair_sharedsort_backward_samples,
                "tile_pair_target_bounds": tile_pair_target_bounds_backward_samples,
                "tile_pair_suffix": tile_pair_suffix_backward_samples,
            }[ctx.sample_emission_mode]
            ids, grad_ma_samples, grad_q_samples, grad_opacity_samples, grad_color_samples, keys, _tile_unstable = tile_pair_fn(
                ma.detach(),
                q_uvt.detach(),
                depth0.detach(),
                depth_beta.detach(),
                opacity.detach(),
                color.detach(),
                grad_output.contiguous(),
                ctx.config,
            )
        else:
            raise ValueError(
                "sample emission mode must be one of: atomic_append, with_keys, tile_pair, tile_pair_compensated, tile_pair_grouped, tile_pair_parallel, tile_pair_scanline, tile_pair_sharedsort, tile_pair_target_bounds, tile_pair_suffix, direct_atomic, direct_fixedpoint, direct_split_fixedpoint, direct_serial, tile_pair_atomic, tile_pair_fixedpoint, tile_pair_reduced, tile_pair_reduced_parallel, tile_pair_suffix_reduced"
            )
        tube_count = ctx.tube_count
        grad_ma, grad_q, grad_opacity, grad_color = _reduce_sample_bundle(
            ids,
            grad_ma_samples,
            grad_q_samples,
            grad_opacity_samples,
            grad_color_samples,
            tube_count,
            mode=ctx.reduction_mode,
            keys=keys,
        )
        grad_depth0 = torch.zeros_like(depth0)
        grad_depth_beta = torch.zeros_like(depth_beta)
        return grad_ma, grad_q, grad_depth0, grad_depth_beta, grad_opacity, grad_color, None, None, None


def render_uvt_tubes_metal_tile_backward(
    ma: Tensor,
    q_uvt: Tensor,
    depth0: Tensor,
    depth_beta: Tensor,
    opacity: Tensor,
    color: Tensor,
    config: UVTRenderConfig,
    reduction_mode: str = "index_add",
    sample_emission_mode: str = "atomic_append",
) -> Tensor:
    """Use Metal forward and Metal per-sample backward with an opt-in diagnostic reducer."""

    return _MetalTileBackward.apply(
        ma, q_uvt, depth0, depth_beta, opacity, color, config, reduction_mode, sample_emission_mode
    )


def full_active_intervals(tube_count: int, frames: int, device: torch.device | str) -> tuple[Tensor, Tensor]:
    """Return whole-video interval gates for trainer-selected q-UVT backends."""

    if tube_count <= 0:
        raise ValueError("tube_count must be positive")
    if frames <= 0:
        raise ValueError("frames must be positive")
    dev = torch.device(device)
    active_start = torch.zeros((int(tube_count),), dtype=torch.int32, device=dev)
    active_stop = torch.full((int(tube_count),), int(frames), dtype=torch.int32, device=dev)
    return active_start, active_stop


@dataclass(frozen=True)
class ProjectiveCellIntervalStaticAtlas:
    cells: list
    source_window_indices: tuple[int, ...]
    source_primitive_ids: tuple[int, ...]
    active_start: tuple[int, ...]
    active_stop: tuple[int, ...]
    spatial_precision_uv: Tensor | None = None
    depth_affine_uv: Tensor | None = None
    depth_reference_uvt: Tensor | None = None
    opacity_time_centered: bool = False


@dataclass(frozen=True)
class ProjectiveCellIntervalAtlasRefresh:
    atlas: ProjectiveTraceCellTraceAtlas
    before: ProjectiveTraceCellAtlasCoverageReport
    after: ProjectiveTraceCellAtlasCoverageReport
    support_margin_before: ProjectiveTraceCellAtlasSupportMarginReport
    support_margin_after: ProjectiveTraceCellAtlasSupportMarginReport
    support_tail_alpha_bound_before: float
    support_tail_alpha_bound_after: float
    visibility_before: ProjectiveTraceCellAtlasVisibilityReport
    visibility_after: ProjectiveTraceCellAtlasVisibilityReport
    budget_after: ProjectiveTraceCellAtlasBudgetReport
    rebinned: bool
    visibility_stratified: bool
    fallback_marked: bool


@dataclass
class ProjectiveCellIntervalTrainerState:
    atlas: ProjectiveTraceCellTraceAtlas
    times: Tensor
    config: UVTRenderConfig
    sigma_px: float
    image_width: int
    image_height: int
    tile_size: int
    uv_padding: float = 0.0
    support_uv_padding: float | None = None
    budget_support_guard: bool = False
    support_guard_policy: str = "fixed"
    support_guard_bisect_steps: int = 8
    support_stale_overshoot_epsilon: float = 0.0
    support_stale_tail_alpha_epsilon: float = 0.0
    depth_padding: float = 0.0
    depth_epsilon: float = 1.0e-6
    refresh_every: int = 1
    check_visibility: bool = True
    allow_ambiguous_fallback: bool = False
    fallback_render_mode: str = "error"
    enforce_complexity_budget: bool = False
    max_interval_to_dense_trace_sample_ratio: float = 1.0
    max_fallback_fraction: float = 0.20
    max_cells_per_active_set_group: int = 16
    step_index: int = 0
    last_refresh: ProjectiveCellIntervalAtlasRefresh | None = None

    def __post_init__(self) -> None:
        if self.image_width <= 0 or self.image_height <= 0:
            raise ValueError("image dimensions must be positive")
        if self.tile_size <= 0:
            raise ValueError("tile_size must be positive")
        if self.sigma_px <= 0.0:
            raise ValueError("sigma_px must be positive")
        if self.uv_padding < 0.0:
            raise ValueError("uv_padding must be non-negative")
        if self.support_uv_padding is not None and self.support_uv_padding < self.uv_padding:
            raise ValueError("support_uv_padding must be None or at least uv_padding")
        if self.support_guard_policy not in {
            "fixed",
            "budgeted",
            "local_budgeted",
            "trace_budgeted",
            "slack_budgeted",
        }:
            raise ValueError(
                "support_guard_policy must be one of: fixed, budgeted, local_budgeted, "
                "trace_budgeted, slack_budgeted"
            )
        if self.support_guard_bisect_steps < 0:
            raise ValueError("support_guard_bisect_steps must be non-negative")
        if self.support_stale_overshoot_epsilon < 0.0:
            raise ValueError("support_stale_overshoot_epsilon must be non-negative")
        if self.support_stale_tail_alpha_epsilon < 0.0:
            raise ValueError("support_stale_tail_alpha_epsilon must be non-negative")
        if self.depth_padding < 0.0:
            raise ValueError("depth_padding must be non-negative")
        if self.depth_epsilon < 0.0:
            raise ValueError("depth_epsilon must be non-negative")
        if self.refresh_every < 1:
            raise ValueError("refresh_every must be positive")
        if self.max_interval_to_dense_trace_sample_ratio < 0.0:
            raise ValueError("max_interval_to_dense_trace_sample_ratio must be non-negative")
        if self.max_fallback_fraction < 0.0:
            raise ValueError("max_fallback_fraction must be non-negative")
        if self.max_cells_per_active_set_group < 1:
            raise ValueError("max_cells_per_active_set_group must be positive")
        if self.fallback_render_mode not in {"error", "mixed", "reference"}:
            raise ValueError("fallback_render_mode must be one of: error, mixed, reference")

    def render(self) -> Tensor:
        if any(cell.fallback for cell in self.atlas.cells):
            if self.fallback_render_mode == "mixed":
                return self.render_mixed_fallback()
            if self.fallback_render_mode == "reference":
                return self.render_reference_with_fallback()
            raise RuntimeError(
                "projective interval Metal render cannot execute fallback cells in strict mode; "
                "set fallback_render_mode='mixed' or use render_reference_with_fallback()"
            )
        return render_projective_cell_interval_atlas_metal_backward(
            self.atlas,
            self.times,
            self.config,
            sigma_px=float(self.sigma_px),
        )

    def render_mixed_fallback(self) -> Tensor:
        """Render fast cells with native Metal autograd and patch fallback cells.

        Fallback is a whole tile/sample visibility replacement. The fallback
        regions use the live-depth Torch reference, so gradients still flow
        through trace opacity, color, and footprint parameters there while
        ordinary cells keep the native interval Metal VJP.
        """

        if not any(cell.fallback for cell in self.atlas.cells):
            return render_projective_cell_interval_atlas_metal_backward(
                self.atlas,
                self.times,
                self.config,
                sigma_px=float(self.sigma_px),
            )

        fast_atlas, _fallback_atlas = split_projective_trace_cell_atlas_fallback_cells(self.atlas)
        if fast_atlas.cells:
            fast = render_projective_cell_interval_atlas_metal_backward(
                fast_atlas,
                self.times,
                self.config,
                sigma_px=float(self.sigma_px),
            )
        else:
            fast = torch.zeros(
                (int(self.config.frames), int(self.image_height), int(self.image_width), int(self.atlas.color.shape[1])),
                dtype=self.atlas.color.dtype,
                device=self.atlas.color.device,
            )
        reference = self.render_reference_with_fallback(fallback_tiles_only=True).to(device=fast.device)
        fallback_mask = projective_trace_cell_atlas_fallback_tile_sample_mask(
            self.atlas,
            frames=int(self.times.numel()),
            image_width=int(self.image_width),
            image_height=int(self.image_height),
            tile_size=int(self.tile_size),
            device=fast.device,
        )
        return _patch_projective_cell_fallback_regions(
            fast,
            reference,
            fallback_mask,
            tile_size=int(self.tile_size),
        )

    def fallback_stats(self) -> ProjectiveTraceCellAtlasFallbackStats:
        return projective_trace_cell_atlas_fallback_stats(self.atlas)

    def complexity_stats(self) -> ProjectiveTraceCellAtlasComplexityStats:
        return projective_trace_cell_atlas_complexity_stats(self.atlas)

    def budget_report(
        self,
        *,
        max_interval_to_dense_trace_sample_ratio: float = 1.0,
        max_fallback_fraction: float = 0.20,
        max_cells_per_active_set_group: int = 16,
    ) -> ProjectiveTraceCellAtlasBudgetReport:
        return projective_trace_cell_atlas_budget_report(
            self.atlas,
            max_interval_to_dense_trace_sample_ratio=max_interval_to_dense_trace_sample_ratio,
            max_fallback_fraction=max_fallback_fraction,
            max_cells_per_active_set_group=max_cells_per_active_set_group,
        )

    def render_reference_with_fallback(self, *, fallback_tiles_only: bool = False) -> Tensor:
        return render_projective_trace_cell_atlas_reference(
            self.atlas,
            self.times,
            image_width=int(self.image_width),
            image_height=int(self.image_height),
            tile_size=int(self.tile_size),
            sigma_px=float(self.sigma_px),
            alpha_cutoff=float(self.config.alpha_threshold),
            transmittance_cutoff=float(self.config.transmittance_threshold),
            allow_fallback_cells=True,
            fallback_sort_live_depth=True,
            fallback_tiles_only=fallback_tiles_only,
        )

    def refresh(self, *, force: bool = False) -> ProjectiveCellIntervalAtlasRefresh:
        refresh = refresh_projective_cell_interval_atlas_if_stale(
            self.atlas,
            self.times,
            image_width=int(self.image_width),
            image_height=int(self.image_height),
            tile_size=int(self.tile_size),
            uv_padding=float(self.uv_padding),
            support_uv_padding=None if self.support_uv_padding is None else float(self.support_uv_padding),
            tile_capacity=int(self.config.tile_capacity),
            budget_support_guard=bool(self.budget_support_guard),
            support_guard_policy=str(self.support_guard_policy),
            support_guard_bisect_steps=int(self.support_guard_bisect_steps),
            support_stale_overshoot_epsilon=float(self.support_stale_overshoot_epsilon),
            support_stale_tail_alpha_epsilon=float(self.support_stale_tail_alpha_epsilon),
            sigma_px=float(self.sigma_px),
            depth_padding=float(self.depth_padding),
            depth_epsilon=float(self.depth_epsilon),
            check_visibility=bool(self.check_visibility),
            allow_ambiguous_fallback=bool(self.allow_ambiguous_fallback),
            enforce_complexity_budget=bool(self.enforce_complexity_budget),
            max_interval_to_dense_trace_sample_ratio=float(self.max_interval_to_dense_trace_sample_ratio),
            max_fallback_fraction=float(self.max_fallback_fraction),
            max_cells_per_active_set_group=int(self.max_cells_per_active_set_group),
            force=force,
        )
        self.atlas = refresh.atlas
        self.last_refresh = refresh
        return refresh

    def after_optimizer_step(self) -> ProjectiveCellIntervalAtlasRefresh | None:
        self.step_index += 1
        if self.step_index % int(self.refresh_every) != 0:
            return None
        return self.refresh()


def _static_projective_cell_atlas(atlas: ProjectiveTraceCellTraceAtlas) -> ProjectiveCellIntervalStaticAtlas:
    return ProjectiveCellIntervalStaticAtlas(
        cells=atlas.cells,
        source_window_indices=atlas.source_window_indices,
        source_primitive_ids=atlas.source_primitive_ids,
        active_start=atlas.active_start,
        active_stop=atlas.active_stop,
        spatial_precision_uv=atlas.spatial_precision_uv,
        depth_affine_uv=atlas.depth_affine_uv,
        depth_reference_uvt=atlas.depth_reference_uvt,
        opacity_time_centered=atlas.opacity_time_centered,
    )


def _patch_projective_cell_fallback_regions(
    fast: Tensor,
    reference: Tensor,
    fallback_mask: Tensor,
    *,
    tile_size: int,
) -> Tensor:
    if fast.shape != reference.shape:
        raise ValueError("fast and reference renders must share shape")
    if fast.ndim != 4:
        raise ValueError("renders must have shape [frames,height,width,channels]")
    if fallback_mask.shape[0] != fast.shape[0]:
        raise ValueError("fallback mask frame count must match render frame count")
    if tile_size <= 0:
        raise ValueError("tile_size must be positive")
    if not bool(torch.any(fallback_mask).item()):
        return fast

    patched = fast.clone()
    height = int(fast.shape[1])
    width = int(fast.shape[2])
    for frame_index, tile_v, tile_u in fallback_mask.detach().cpu().nonzero(as_tuple=False).tolist():
        v0 = int(tile_v) * int(tile_size)
        u0 = int(tile_u) * int(tile_size)
        v1 = min(height, v0 + int(tile_size))
        u1 = min(width, u0 + int(tile_size))
        patched[int(frame_index), v0:v1, u0:u1, :] = reference[int(frame_index), v0:v1, u0:u1, :]
    return patched


def refresh_projective_cell_interval_atlas_if_stale(
    atlas: ProjectiveTraceCellTraceAtlas,
    times: Tensor,
    *,
    image_width: int,
    image_height: int,
    tile_size: int,
    uv_padding: float = 0.0,
    support_uv_padding: float | None = None,
    tile_capacity: int | None = None,
    budget_support_guard: bool = False,
    support_guard_policy: str = "fixed",
    support_guard_bisect_steps: int = 8,
    support_stale_overshoot_epsilon: float = 0.0,
    support_stale_tail_alpha_epsilon: float = 0.0,
    sigma_px: float | None = None,
    depth_padding: float = 0.0,
    depth_epsilon: float = 1.0e-6,
    check_visibility: bool = True,
    allow_ambiguous_fallback: bool = False,
    enforce_complexity_budget: bool = False,
    max_interval_to_dense_trace_sample_ratio: float = 1.0,
    max_fallback_fraction: float = 0.20,
    max_cells_per_active_set_group: int = 16,
    force: bool = False,
) -> ProjectiveCellIntervalAtlasRefresh:
    """Refresh compiled projective cell support when live traces leave it.

    The autograd bridge treats cells/order as static for a single forward/backward
    pass. A trainer can call this between optimizer steps to preserve that local
    contract while coefficients move. Rebinning keeps ``coeffs``, ``opacity``,
    and ``color`` tensors intact and updates only support/depth metadata.
    """

    if support_uv_padding is not None and support_uv_padding < uv_padding:
        raise ValueError("support_uv_padding must be None or at least uv_padding")
    if tile_capacity is not None and tile_capacity <= 0:
        raise ValueError("tile_capacity must be positive")
    if support_guard_policy not in {"fixed", "budgeted", "local_budgeted", "trace_budgeted", "slack_budgeted"}:
        raise ValueError(
            "support_guard_policy must be one of: fixed, budgeted, local_budgeted, "
            "trace_budgeted, slack_budgeted"
        )
    if support_guard_bisect_steps < 0:
        raise ValueError("support_guard_bisect_steps must be non-negative")
    if support_stale_overshoot_epsilon < 0.0:
        raise ValueError("support_stale_overshoot_epsilon must be non-negative")
    if support_stale_tail_alpha_epsilon < 0.0:
        raise ValueError("support_stale_tail_alpha_epsilon must be non-negative")
    if support_stale_tail_alpha_epsilon > 0.0 and (sigma_px is None or sigma_px <= 0.0):
        raise ValueError("positive support_stale_tail_alpha_epsilon requires positive sigma_px")
    effective_support_guard_policy = str(support_guard_policy)
    if effective_support_guard_policy == "fixed" and budget_support_guard:
        effective_support_guard_policy = "budgeted"

    def _atlas_overflows_tile_capacity(current_atlas: ProjectiveTraceCellTraceAtlas) -> bool:
        if tile_capacity is None:
            return False
        bins = pack_projective_trace_tile_time_bins(
            current_atlas.cells,
            image_width=image_width,
            image_height=image_height,
            frames=int(times.numel()),
            tile_x=tile_size,
            tile_y=tile_size,
            tile_t=int(times.numel()),
            tile_capacity=int(tile_capacity),
            allow_fallback_cells=True,
        )
        return bool(torch.any(bins.tile_overflow > 0).item())

    def _overflow_tile_coords(current_atlas: ProjectiveTraceCellTraceAtlas) -> set[tuple[int, int]]:
        if tile_capacity is None:
            return set()
        bins = pack_projective_trace_tile_time_bins(
            current_atlas.cells,
            image_width=image_width,
            image_height=image_height,
            frames=int(times.numel()),
            tile_x=tile_size,
            tile_y=tile_size,
            tile_t=int(times.numel()),
            tile_capacity=int(tile_capacity),
            allow_fallback_cells=True,
        )
        tiles_x = (int(image_width) + int(tile_size) - 1) // int(tile_size)
        tiles_y = (int(image_height) + int(tile_size) - 1) // int(tile_size)
        coords: set[tuple[int, int]] = set()
        for flat_tile_id in bins.tile_overflow.detach().cpu().nonzero(as_tuple=False).reshape(-1).tolist():
            tile_id = int(flat_tile_id) % int(tiles_x * tiles_y)
            coords.add((tile_id % tiles_x, tile_id // tiles_x))
        return coords

    def _atlas_with_cells(
        template_atlas: ProjectiveTraceCellTraceAtlas,
        cells: list[ProjectiveTraceTileTimeCell],
    ) -> ProjectiveTraceCellTraceAtlas:
        return ProjectiveTraceCellTraceAtlas(
            coeffs=template_atlas.coeffs,
            opacity=template_atlas.opacity,
            color=template_atlas.color,
            cells=sorted(cells, key=lambda cell: (cell.start, cell.stop, cell.tile_v, cell.tile_u)),
            source_window_indices=template_atlas.source_window_indices,
            source_primitive_ids=template_atlas.source_primitive_ids,
            active_start=template_atlas.active_start,
            active_stop=template_atlas.active_stop,
            opacity_time_coeffs=template_atlas.opacity_time_coeffs,
            spatial_precision_uv=template_atlas.spatial_precision_uv,
            depth_affine_uv=template_atlas.depth_affine_uv,
            depth_reference_uvt=template_atlas.depth_reference_uvt,
            opacity_time_centered=template_atlas.opacity_time_centered,
        )

    def _mix_target_and_base_tiles(
        *,
        target_atlas: ProjectiveTraceCellTraceAtlas,
        base_atlas: ProjectiveTraceCellTraceAtlas,
        base_tile_coords: set[tuple[int, int]],
    ) -> ProjectiveTraceCellTraceAtlas:
        if not base_tile_coords:
            return target_atlas
        mixed_cells = [
            cell
            for cell in target_atlas.cells
            if (int(cell.tile_u), int(cell.tile_v)) not in base_tile_coords
        ]
        mixed_cells.extend(
            cell
            for cell in base_atlas.cells
            if (int(cell.tile_u), int(cell.tile_v)) in base_tile_coords
        )
        return _atlas_with_cells(target_atlas, mixed_cells)

    def _filter_cell_by_primitive_ids(
        cell: ProjectiveTraceTileTimeCell,
        selected_ids: set[int],
    ) -> ProjectiveTraceTileTimeCell | None:
        kept: list[tuple[int, tuple[float, float]]] = [
            (int(primitive_id), depth_interval)
            for primitive_id, depth_interval in zip(cell.ordered_primitive_ids, cell.depth_intervals)
            if int(primitive_id) in selected_ids
        ]
        if not kept:
            return None
        ordered_ids = tuple(primitive_id for primitive_id, _depth_interval in kept)
        return type(cell)(
            tile_u=cell.tile_u,
            tile_v=cell.tile_v,
            start=cell.start,
            stop=cell.stop,
            primitive_ids=tuple(sorted(set(ordered_ids))),
            ordered_primitive_ids=ordered_ids,
            depth_intervals=tuple(depth_interval for _primitive_id, depth_interval in kept),
            fallback=cell.fallback,
            fallback_reasons=cell.fallback_reasons,
        )

    def _tile_primitive_ids(
        current_atlas: ProjectiveTraceCellTraceAtlas,
        tile_coord: tuple[int, int],
    ) -> set[int]:
        return {
            int(primitive_id)
            for cell in current_atlas.cells
            if (int(cell.tile_u), int(cell.tile_v)) == tile_coord
            for primitive_id in cell.ordered_primitive_ids
        }

    def _tile_support_event_distances(
        current_atlas: ProjectiveTraceCellTraceAtlas,
        tile_coord: tuple[int, int],
        *,
        base_padding: float,
    ) -> dict[int, float]:
        samples = eval_projective_trace_cell_torch(
            current_atlas.coeffs.detach().cpu().contiguous(),
            times.detach().cpu().contiguous(),
        )
        tile_u, tile_v = (int(tile_coord[0]), int(tile_coord[1]))
        tile_left = float(tile_u * int(tile_size))
        tile_right = float((tile_u + 1) * int(tile_size))
        tile_top = float(tile_v * int(tile_size))
        tile_bottom = float((tile_v + 1) * int(tile_size))
        distances: dict[int, float] = {}
        for cell in current_atlas.cells:
            if (int(cell.tile_u), int(cell.tile_v)) != (tile_u, tile_v):
                continue
            start = max(0, int(cell.start))
            stop = min(int(samples.shape[1]), int(cell.stop))
            if start >= stop:
                continue
            for primitive_id in cell.primitive_ids:
                primitive_id = int(primitive_id)
                if primitive_id < 0 or primitive_id >= int(samples.shape[0]):
                    continue
                best = distances.get(primitive_id, float("inf"))
                for sample_index in range(start, stop):
                    u, v, _depth, valid_sign = samples[primitive_id, sample_index]
                    if float(valid_sign.item()) == 0.0:
                        continue
                    u_min = float(u.item()) - float(base_padding)
                    u_max = float(u.item()) + float(base_padding)
                    v_min = float(v.item()) - float(base_padding)
                    v_max = float(v.item()) + float(base_padding)
                    distance = max(
                        tile_left - u_max,
                        u_min - tile_right,
                        tile_top - v_max,
                        v_min - tile_bottom,
                        0.0,
                    )
                    best = min(best, float(distance))
                distances[primitive_id] = best
        return distances

    def _mix_target_and_base_traces(
        *,
        target_atlas: ProjectiveTraceCellTraceAtlas,
        base_atlas: ProjectiveTraceCellTraceAtlas,
        base_tile_coords: set[tuple[int, int]],
        rank_by_support_event_distance: bool = False,
        base_padding: float = 0.0,
    ) -> ProjectiveTraceCellTraceAtlas:
        if not base_tile_coords:
            return target_atlas
        mixed_cells = [
            cell
            for cell in target_atlas.cells
            if (int(cell.tile_u), int(cell.tile_v)) not in base_tile_coords
        ]
        for tile_coord in sorted(base_tile_coords):
            selected_ids = _tile_primitive_ids(base_atlas, tile_coord)
            if len(selected_ids) > int(tile_capacity or 0):
                mixed_cells.extend(
                    cell
                    for cell in base_atlas.cells
                    if (int(cell.tile_u), int(cell.tile_v)) == tile_coord
                )
                continue
            target_ids = _tile_primitive_ids(target_atlas, tile_coord)
            extra_ids = list(target_ids - selected_ids)
            if rank_by_support_event_distance:
                distances = _tile_support_event_distances(target_atlas, tile_coord, base_padding=base_padding)
                extra_ids.sort(
                    key=lambda primitive_id: (distances.get(int(primitive_id), float("inf")), int(primitive_id))
                )
            else:
                extra_ids.sort()
            for primitive_id in extra_ids:
                if len(selected_ids) >= int(tile_capacity or 0):
                    break
                selected_ids.add(int(primitive_id))
            for cell in target_atlas.cells:
                if (int(cell.tile_u), int(cell.tile_v)) != tile_coord:
                    continue
                filtered = _filter_cell_by_primitive_ids(cell, selected_ids)
                if filtered is not None:
                    mixed_cells.append(filtered)
        return _atlas_with_cells(target_atlas, mixed_cells)

    def _rebin_with_padding(padding: float) -> ProjectiveTraceCellTraceAtlas:
        return rebin_projective_trace_cell_atlas_support_events(
            atlas,
            times,
            image_width=image_width,
            image_height=image_height,
            tile_size=tile_size,
            uv_padding=float(padding),
            depth_padding=depth_padding,
            root_epsilon=depth_epsilon,
        )

    def _rebin_with_budgeted_support() -> ProjectiveTraceCellTraceAtlas:
        target_padding = float(uv_padding if support_uv_padding is None else support_uv_padding)
        target_atlas = _rebin_with_padding(target_padding)
        if (
            effective_support_guard_policy == "fixed"
            or tile_capacity is None
            or support_uv_padding is None
            or support_uv_padding <= uv_padding
            or not _atlas_overflows_tile_capacity(target_atlas)
        ):
            return target_atlas

        base_padding = float(uv_padding)
        best_atlas = _rebin_with_padding(base_padding)
        if _atlas_overflows_tile_capacity(best_atlas):
            return best_atlas

        overflow_tile_coords = _overflow_tile_coords(target_atlas)
        if effective_support_guard_policy in {"trace_budgeted", "slack_budgeted"}:
            trace_mixed_atlas = _mix_target_and_base_traces(
                target_atlas=target_atlas,
                base_atlas=best_atlas,
                base_tile_coords=overflow_tile_coords,
                rank_by_support_event_distance=effective_support_guard_policy == "slack_budgeted",
                base_padding=base_padding,
            )
            if not _atlas_overflows_tile_capacity(trace_mixed_atlas):
                return trace_mixed_atlas

        if effective_support_guard_policy in {"local_budgeted", "trace_budgeted", "slack_budgeted"}:
            mixed_atlas = _mix_target_and_base_tiles(
                target_atlas=target_atlas,
                base_atlas=best_atlas,
                base_tile_coords=overflow_tile_coords,
            )
            if not _atlas_overflows_tile_capacity(mixed_atlas):
                return mixed_atlas

        low_guard = 0.0
        high_guard = float(support_uv_padding) - base_padding
        for _ in range(int(support_guard_bisect_steps)):
            mid_guard = 0.5 * (low_guard + high_guard)
            if mid_guard <= low_guard + 1.0e-6:
                break
            candidate_atlas = _rebin_with_padding(base_padding + mid_guard)
            if _atlas_overflows_tile_capacity(candidate_atlas):
                high_guard = mid_guard
                continue
            low_guard = mid_guard
            best_atlas = candidate_atlas
        return best_atlas

    def _budget_report(current_atlas: ProjectiveTraceCellTraceAtlas) -> ProjectiveTraceCellAtlasBudgetReport:
        return projective_trace_cell_atlas_budget_report(
            current_atlas,
            max_interval_to_dense_trace_sample_ratio=max_interval_to_dense_trace_sample_ratio,
            max_fallback_fraction=max_fallback_fraction,
            max_cells_per_active_set_group=max_cells_per_active_set_group,
        )

    def _support_tile_range(
        *,
        u_min: float,
        u_max: float,
        v_min: float,
        v_max: float,
    ) -> tuple[int, int, int, int] | None:
        tile_cols = (int(image_width) + int(tile_size) - 1) // int(tile_size)
        tile_rows = (int(image_height) + int(tile_size) - 1) // int(tile_size)
        tile_u_min = int(math.floor(float(u_min) / float(tile_size)))
        tile_u_max = int(math.floor(float(u_max) / float(tile_size))) + 1
        tile_v_min = int(math.floor(float(v_min) / float(tile_size)))
        tile_v_max = int(math.floor(float(v_max) / float(tile_size))) + 1
        if tile_u_max <= 0 or tile_v_max <= 0 or tile_u_min >= tile_cols or tile_v_min >= tile_rows:
            return None
        return (
            max(0, tile_u_min),
            min(tile_cols, tile_u_max),
            max(0, tile_v_min),
            min(tile_rows, tile_v_max),
        )

    def _opacity_upper_bounds(current_atlas: ProjectiveTraceCellTraceAtlas) -> Tensor:
        opacity = current_atlas.opacity.detach().cpu().to(dtype=torch.float32).clamp(min=0.0)
        if current_atlas.opacity_time_coeffs is None:
            return opacity.clamp(max=1.0)
        coeffs = current_atlas.opacity_time_coeffs.detach().cpu().to(dtype=torch.float32).contiguous()
        times_cpu = times.detach().cpu().to(dtype=torch.float32).contiguous()
        t = times_cpu.reshape(1, -1)
        qv = (
            coeffs[:, 0:1] + coeffs[:, 2:3] * (t - coeffs[:, 1:2]).square()
            if current_atlas.opacity_time_centered
            else coeffs[:, 0:1] + coeffs[:, 1:2] * t + coeffs[:, 2:3] * t.square()
        )
        scale = torch.exp(-0.5 * qv).amax(dim=1)
        return (opacity * scale).clamp(max=1.0)

    def _support_tail_alpha_bound(current_atlas: ProjectiveTraceCellTraceAtlas) -> float:
        if sigma_px is None:
            return 0.0
        coeffs_cpu = current_atlas.coeffs.detach().cpu().contiguous()
        times_cpu = times.detach().cpu().contiguous()
        dense = eval_projective_trace_cell_torch(coeffs_cpu, times_cpu)
        frame_count = int(times_cpu.numel())
        trace_count = int(coeffs_cpu.shape[0])
        covered_by_sample: dict[tuple[int, int], set[tuple[int, int]]] = {}
        for cell in current_atlas.cells:
            start = max(0, int(cell.start))
            stop = min(frame_count, int(cell.stop))
            if start >= stop:
                continue
            for trace_id in cell.primitive_ids:
                if trace_id < 0 or trace_id >= trace_count:
                    continue
                for sample_index in range(start, stop):
                    covered_by_sample.setdefault((int(trace_id), sample_index), set()).add(
                        (int(cell.tile_u), int(cell.tile_v))
                    )

        opacity_bound = _opacity_upper_bounds(current_atlas)
        omitted_alpha_by_tile_sample: dict[tuple[int, int, int], float] = {}

        def _add_omitted_alpha(sample_index: int, tile_u: int, tile_v: int, tail: float) -> None:
            key = (int(sample_index), int(tile_u), int(tile_v))
            omitted_alpha_by_tile_sample[key] = omitted_alpha_by_tile_sample.get(key, 0.0) + float(tail)

        spatial_precision_cpu = (
            None
            if current_atlas.spatial_precision_uv is None
            else current_atlas.spatial_precision_uv.detach().cpu().to(dtype=torch.float32).contiguous()
        )

        def _min_spatial_quadratic_on_tile(
            *,
            trace_id: int,
            center_u: float,
            center_v: float,
            tile_u: int,
            tile_v: int,
        ) -> float:
            if spatial_precision_cpu is None:
                raise RuntimeError("spatial precision metadata is unavailable")
            q_uu = float(spatial_precision_cpu[int(trace_id), 0].item())
            q_uv = float(spatial_precision_cpu[int(trace_id), 1].item())
            q_vv = float(spatial_precision_cpu[int(trace_id), 2].item())
            det = q_uu * q_vv - q_uv * q_uv
            if q_uu <= 0.0 or q_vv <= 0.0 or det <= 0.0:
                return math.inf
            u0 = float(int(tile_u) * int(tile_size))
            u1 = float(min(int(image_width), (int(tile_u) + 1) * int(tile_size)))
            v0 = float(int(tile_v) * int(tile_size))
            v1 = float(min(int(image_height), (int(tile_v) + 1) * int(tile_size)))

            def _clamp(value: float, low: float, high: float) -> float:
                return max(float(low), min(float(high), float(value)))

            def _quad(u_value: float, v_value: float) -> float:
                du = float(u_value) - float(center_u)
                dv = float(v_value) - float(center_v)
                return q_uu * du * du + 2.0 * q_uv * du * dv + q_vv * dv * dv

            candidates: list[tuple[float, float]] = []
            if u0 <= float(center_u) <= u1 and v0 <= float(center_v) <= v1:
                candidates.append((float(center_u), float(center_v)))
            for u_value in (u0, u1):
                du = float(u_value) - float(center_u)
                v_value = float(center_v) - (q_uv / q_vv) * du
                candidates.append((u_value, _clamp(v_value, v0, v1)))
            for v_value in (v0, v1):
                dv = float(v_value) - float(center_v)
                u_value = float(center_u) - (q_uv / q_uu) * dv
                candidates.append((_clamp(u_value, u0, u1), v_value))
            for u_value in (u0, u1):
                for v_value in (v0, v1):
                    candidates.append((u_value, v_value))
            return min(_quad(u_value, v_value) for u_value, v_value in candidates)

        def _omitted_tail_for_tile(trace_id: int, sample_index: int, tile_u: int, tile_v: int, overshoot: float) -> float:
            if spatial_precision_cpu is not None:
                u, v, _depth, _valid_sign = dense[int(trace_id), int(sample_index)]
                q_min = _min_spatial_quadratic_on_tile(
                    trace_id=int(trace_id),
                    center_u=float(u.item()),
                    center_v=float(v.item()),
                    tile_u=int(tile_u),
                    tile_v=int(tile_v),
                )
                return float(opacity_bound[int(trace_id)].item()) * math.exp(-0.5 * float(q_min))
            certified_distance = max(float(uv_padding) - float(overshoot), 0.0)
            return float(opacity_bound[int(trace_id)].item()) * math.exp(
                -0.5 * (certified_distance / float(sigma_px)) ** 2
            )

        for trace_id in range(trace_count):
            active_start = int(current_atlas.active_start[trace_id])
            active_stop = int(current_atlas.active_stop[trace_id])
            if active_start < 0 or active_stop > frame_count or active_start >= active_stop:
                return math.inf
            for sample_index in range(active_start, active_stop):
                u, v, _depth, valid_sign = dense[trace_id, sample_index]
                if float(valid_sign.item()) == 0.0:
                    return math.inf
                u_min = float(u.item()) - float(uv_padding)
                u_max = float(u.item()) + float(uv_padding)
                v_min = float(v.item()) - float(uv_padding)
                v_max = float(v.item()) + float(uv_padding)
                tile_range = _support_tile_range(u_min=u_min, u_max=u_max, v_min=v_min, v_max=v_max)
                if tile_range is None:
                    continue
                covered_tiles = covered_by_sample.get((trace_id, sample_index), set())
                tile_u_min, tile_u_max, tile_v_min, tile_v_max = tile_range
                if not covered_tiles:
                    for tile_u in range(tile_u_min, tile_u_max):
                        for tile_v in range(tile_v_min, tile_v_max):
                            tail = _omitted_tail_for_tile(trace_id, sample_index, tile_u, tile_v, 0.0)
                            _add_omitted_alpha(sample_index, tile_u, tile_v, tail)
                    continue
                covered_u = [tile_u for tile_u, _tile_v in covered_tiles]
                covered_v = [tile_v for _tile_u, tile_v in covered_tiles]
                covered_u_min = min(covered_u)
                covered_u_max = max(covered_u)
                covered_v_min = min(covered_v)
                covered_v_max = max(covered_v)
                for tile_u in range(tile_u_min, tile_u_max):
                    for tile_v in range(tile_v_min, tile_v_max):
                        if (tile_u, tile_v) in covered_tiles:
                            continue
                        u_overshoot = 0.0
                        if tile_u < covered_u_min:
                            u_overshoot = float(covered_u_min * int(tile_size)) - u_min
                        elif tile_u > covered_u_max:
                            u_overshoot = u_max - float((covered_u_max + 1) * int(tile_size))
                        v_overshoot = 0.0
                        if tile_v < covered_v_min:
                            v_overshoot = float(covered_v_min * int(tile_size)) - v_min
                        elif tile_v > covered_v_max:
                            v_overshoot = v_max - float((covered_v_max + 1) * int(tile_size))
                        overshoot = max(0.0, u_overshoot, v_overshoot)
                        tail = _omitted_tail_for_tile(trace_id, sample_index, tile_u, tile_v, overshoot)
                        _add_omitted_alpha(sample_index, tile_u, tile_v, tail)
        return float(max(omitted_alpha_by_tile_sample.values(), default=0.0))

    def _raise_if_budget_failed(report: ProjectiveTraceCellAtlasBudgetReport) -> None:
        if not enforce_complexity_budget or report.within_budget:
            return
        failures = ", ".join(report.failures)
        raise RuntimeError(
            "projective cell atlas exceeded complexity budget "
            f"({failures}; interval_to_dense_trace_sample_ratio="
            f"{report.stats.interval_to_dense_trace_sample_ratio:.6g}, "
            f"fallback_fraction={report.stats.fallback_fraction:.6g}, "
            f"max_cells_per_active_set_group={report.stats.max_cells_per_active_set_group})"
        )

    before = projective_trace_cell_atlas_coverage_report(
        atlas,
        times,
        image_width=image_width,
        image_height=image_height,
        tile_size=tile_size,
        uv_padding=uv_padding,
    )
    support_margin_before = projective_trace_cell_atlas_support_margin_report(
        atlas,
        times,
        image_width=image_width,
        image_height=image_height,
        tile_size=tile_size,
        uv_padding=uv_padding,
    )
    support_tail_alpha_bound_before = _support_tail_alpha_bound(atlas)
    visibility_before = projective_trace_cell_atlas_visibility_report(
        atlas,
        times,
        depth_epsilon=depth_epsilon,
        image_width=image_width,
        image_height=image_height,
        tile_size=tile_size,
    )
    tail_alpha_allows_reuse = (
        support_stale_tail_alpha_epsilon > 0.0
        and support_tail_alpha_bound_before <= float(support_stale_tail_alpha_epsilon)
    )
    support_stale = before.stale and (
        before.invalid_active_samples > 0
        or (
            support_margin_before.max_boundary_overshoot_px > float(support_stale_overshoot_epsilon)
            and not tail_alpha_allows_reuse
        )
    )
    visibility_stale = check_visibility and visibility_before.stale
    if not force and not support_stale and not visibility_stale:
        budget_after = _budget_report(atlas)
        _raise_if_budget_failed(budget_after)
        return ProjectiveCellIntervalAtlasRefresh(
            atlas=atlas,
            before=before,
            after=before,
            support_margin_before=support_margin_before,
            support_margin_after=support_margin_before,
            support_tail_alpha_bound_before=support_tail_alpha_bound_before,
            support_tail_alpha_bound_after=support_tail_alpha_bound_before,
            visibility_before=visibility_before,
            visibility_after=visibility_before,
            budget_after=budget_after,
            rebinned=False,
            visibility_stratified=False,
            fallback_marked=False,
        )

    refreshed_atlas = _rebin_with_budgeted_support()
    after = projective_trace_cell_atlas_coverage_report(
        refreshed_atlas,
        times,
        image_width=image_width,
        image_height=image_height,
        tile_size=tile_size,
        uv_padding=uv_padding,
    )
    visibility_after = projective_trace_cell_atlas_visibility_report(
        refreshed_atlas,
        times,
        depth_epsilon=depth_epsilon,
        image_width=image_width,
        image_height=image_height,
        tile_size=tile_size,
    )
    support_tail_alpha_bound_after = 0.0 if not after.stale else _support_tail_alpha_bound(refreshed_atlas)
    if after.stale:
        raise RuntimeError(
            "projective cell atlas rebin did not repair coverage "
            f"(missing_tile_pairs={after.missing_tile_pairs}, invalid_active_samples={after.invalid_active_samples})"
        )
    visibility_stratified = False

    def _try_visibility_stratifier(
        candidate_atlas: ProjectiveTraceCellTraceAtlas,
        *,
        current_mismatches: int,
    ) -> tuple[ProjectiveTraceCellTraceAtlas, ProjectiveTraceCellAtlasCoverageReport, ProjectiveTraceCellAtlasVisibilityReport, bool]:
        candidate_after = projective_trace_cell_atlas_coverage_report(
            candidate_atlas,
            times,
            image_width=image_width,
            image_height=image_height,
            tile_size=tile_size,
            uv_padding=uv_padding,
        )
        candidate_visibility_after = projective_trace_cell_atlas_visibility_report(
            candidate_atlas,
            times,
            depth_epsilon=depth_epsilon,
            image_width=image_width,
            image_height=image_height,
            tile_size=tile_size,
        )
        accepted = not candidate_after.stale and (
            candidate_visibility_after.order_mismatch_samples < current_mismatches
            or not candidate_visibility_after.stale
        )
        return candidate_atlas, candidate_after, candidate_visibility_after, accepted

    if (
        check_visibility
        and visibility_after.stale
        and visibility_after.order_mismatch_samples > 0
        and visibility_after.invalid_depth_samples == 0
    ):
        event_atlas, event_after, event_visibility_after, event_accepted = _try_visibility_stratifier(
            stratify_projective_trace_cell_atlas_visibility_events(
                refreshed_atlas,
                times,
                root_epsilon=depth_epsilon,
            ),
            current_mismatches=visibility_after.order_mismatch_samples,
        )
        if event_accepted:
            refreshed_atlas = event_atlas
            after = event_after
            visibility_after = event_visibility_after
            visibility_stratified = True

    if (
        check_visibility
        and visibility_after.stale
        and visibility_after.order_mismatch_samples > 0
        and visibility_after.invalid_depth_samples == 0
    ):
        sampled_atlas, sampled_after, sampled_visibility_after, sampled_accepted = _try_visibility_stratifier(
            stratify_projective_trace_cell_atlas_visibility(
                refreshed_atlas,
                times,
                depth_epsilon=depth_epsilon,
            ),
            current_mismatches=visibility_after.order_mismatch_samples,
        )
        if sampled_accepted:
            refreshed_atlas = sampled_atlas
            after = sampled_after
            visibility_after = sampled_visibility_after
            visibility_stratified = True

    fallback_marked = False
    if (
        check_visibility
        and visibility_after.stale
        and allow_ambiguous_fallback
        and visibility_after.order_mismatch_samples == 0
        and visibility_after.invalid_depth_samples == 0
        and visibility_after.ambiguous_depth_samples > 0
    ):
        refreshed_atlas = mark_projective_trace_cell_visibility_fallbacks(
            refreshed_atlas,
            times,
            depth_epsilon=depth_epsilon,
            image_width=image_width,
            image_height=image_height,
            tile_size=tile_size,
        )
        visibility_after = projective_trace_cell_atlas_visibility_report(
            refreshed_atlas,
            times,
            depth_epsilon=depth_epsilon,
            image_width=image_width,
            image_height=image_height,
            tile_size=tile_size,
            mark_ambiguous_stale=False,
        )
        fallback_marked = True

    if check_visibility and visibility_after.stale:
        raise RuntimeError(
            "projective cell atlas rebin did not repair visibility order "
            f"(order_mismatch_samples={visibility_after.order_mismatch_samples}, "
            f"ambiguous_depth_samples={visibility_after.ambiguous_depth_samples}, "
            f"invalid_depth_samples={visibility_after.invalid_depth_samples})"
        )
    budget_after = _budget_report(refreshed_atlas)
    _raise_if_budget_failed(budget_after)
    support_margin_after = projective_trace_cell_atlas_support_margin_report(
        refreshed_atlas,
        times,
        image_width=image_width,
        image_height=image_height,
        tile_size=tile_size,
        uv_padding=uv_padding,
    )
    return ProjectiveCellIntervalAtlasRefresh(
        atlas=refreshed_atlas,
        before=before,
        after=after,
        support_margin_before=support_margin_before,
        support_margin_after=support_margin_after,
        support_tail_alpha_bound_before=support_tail_alpha_bound_before,
        support_tail_alpha_bound_after=support_tail_alpha_bound_after,
        visibility_before=visibility_before,
        visibility_after=visibility_after,
        budget_after=budget_after,
        rebinned=True,
        visibility_stratified=visibility_stratified,
        fallback_marked=fallback_marked,
    )


def _materialize_projective_cell_atlas(
    coeffs: Tensor,
    opacity: Tensor,
    opacity_time_coeffs: Tensor | None,
    spatial_precision_uv: Tensor | None,
    color: Tensor,
    static_atlas: ProjectiveCellIntervalStaticAtlas,
) -> ProjectiveTraceCellTraceAtlas:
    return ProjectiveTraceCellTraceAtlas(
        coeffs=coeffs,
        opacity=opacity,
        color=color,
        cells=static_atlas.cells,
        source_window_indices=static_atlas.source_window_indices,
        source_primitive_ids=static_atlas.source_primitive_ids,
        active_start=static_atlas.active_start,
        active_stop=static_atlas.active_stop,
        opacity_time_coeffs=opacity_time_coeffs,
        spatial_precision_uv=spatial_precision_uv if static_atlas.spatial_precision_uv is not None else None,
        depth_affine_uv=static_atlas.depth_affine_uv,
        depth_reference_uvt=static_atlas.depth_reference_uvt,
        opacity_time_centered=static_atlas.opacity_time_centered,
    )


class _ProjectiveCellIntervalBackward(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        coeffs: Tensor,
        opacity: Tensor,
        opacity_time_coeffs: Tensor,
        spatial_precision_uv: Tensor,
        color: Tensor,
        times: Tensor,
        static_atlas: ProjectiveCellIntervalStaticAtlas,
        config: UVTRenderConfig,
        sigma_px: float,
    ) -> Tensor:
        ctx.static_atlas = static_atlas
        ctx.config = config
        ctx.sigma_px = float(sigma_px)
        ctx.save_for_backward(coeffs, opacity, opacity_time_coeffs, spatial_precision_uv, color, times)
        atlas = _materialize_projective_cell_atlas(
            coeffs,
            opacity,
            opacity_time_coeffs,
            spatial_precision_uv,
            color,
            static_atlas,
        )
        return render_projective_trace_cell_interval_atlas_metal(
            atlas,
            times,
            config,
            sigma_px=float(sigma_px),
        )

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> tuple[Tensor | None, ...]:
        coeffs, opacity, opacity_time_coeffs, spatial_precision_uv, color, times = ctx.saved_tensors
        atlas = _materialize_projective_cell_atlas(
            coeffs.detach(),
            opacity.detach(),
            opacity_time_coeffs.detach(),
            spatial_precision_uv.detach(),
            color.detach(),
            ctx.static_atlas,
        )
        grads = direct_backward_projective_trace_cell_interval_atlas_metal(
            atlas,
            times.detach(),
            grad_output.contiguous(),
            ctx.config,
            sigma_px=ctx.sigma_px,
        )
        return (
            grads.grad_coeffs,
            grads.grad_opacity,
            grads.grad_opacity_time_coeffs,
            grads.grad_spatial_precision_uv if ctx.static_atlas.spatial_precision_uv is not None else None,
            grads.grad_color,
            None,
            None,
            None,
            None,
        )


def render_projective_cell_interval_atlas_metal_backward(
    atlas: ProjectiveTraceCellTraceAtlas,
    times: Tensor,
    config: UVTRenderConfig,
    *,
    sigma_px: float,
) -> Tensor:
    """Use native interval-compressed projective-cell forward/backward.

    This is the trainer-harness bridge for gauge-domain cell atlases. The
    topology, active intervals, and visibility cells are compiled constants;
    ``coeffs``, ``opacity``, and ``color`` remain differentiable tensors.
    """

    opacity_time_coeffs = atlas.opacity_time_coeffs
    if opacity_time_coeffs is None:
        opacity_time_coeffs = torch.zeros(
            (int(atlas.coeffs.shape[0]), 3),
            dtype=atlas.coeffs.dtype,
            device=atlas.coeffs.device,
        )
    spatial_precision_uv = atlas.spatial_precision_uv
    if spatial_precision_uv is None:
        spatial_precision_uv = torch.zeros(
            (int(atlas.coeffs.shape[0]), 3),
            dtype=atlas.coeffs.dtype,
            device=atlas.coeffs.device,
        )

    return _ProjectiveCellIntervalBackward.apply(
        atlas.coeffs,
        atlas.opacity,
        opacity_time_coeffs,
        spatial_precision_uv,
        atlas.color,
        times,
        _static_projective_cell_atlas(atlas),
        config,
        float(sigma_px),
    )


class _MetalIntervalGatedBackward(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        ma: Tensor,
        q_uvt: Tensor,
        depth0: Tensor,
        depth_beta: Tensor,
        opacity: Tensor,
        color: Tensor,
        active_start: Tensor,
        active_stop: Tensor,
        config: UVTRenderConfig,
    ) -> Tensor:
        ctx.config = config
        ctx.save_for_backward(ma, q_uvt, depth0, depth_beta, opacity, color, active_start, active_stop)
        return render_uvt_tubes_gated(
            ma,
            q_uvt,
            depth0,
            depth_beta,
            opacity,
            color,
            active_start,
            active_stop,
            config,
        )

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> tuple[Tensor | None, ...]:
        ma, q_uvt, depth0, depth_beta, opacity, color, active_start, active_stop = ctx.saved_tensors
        grad_ma, grad_q, grad_opacity, grad_color, _tile_unstable = direct_atomic_backward_gated(
            ma.detach(),
            q_uvt.detach(),
            depth0.detach(),
            depth_beta.detach(),
            opacity.detach(),
            color.detach(),
            grad_output.contiguous(),
            active_start.detach(),
            active_stop.detach(),
            ctx.config,
        )
        grad_depth0 = torch.zeros_like(depth0)
        grad_depth_beta = torch.zeros_like(depth_beta)
        return (
            grad_ma,
            grad_q,
            grad_depth0,
            grad_depth_beta,
            grad_opacity,
            grad_color,
            None,
            None,
            None,
        )


def render_uvt_tubes_metal_interval_gated_backward(
    ma: Tensor,
    q_uvt: Tensor,
    depth0: Tensor,
    depth_beta: Tensor,
    opacity: Tensor,
    color: Tensor,
    active_start: Tensor,
    active_stop: Tensor,
    config: UVTRenderConfig,
) -> Tensor:
    """Use native interval-gated q-UVT forward/backward with explicit chart-domain gates."""

    return _MetalIntervalGatedBackward.apply(
        ma,
        q_uvt,
        depth0,
        depth_beta,
        opacity,
        color,
        active_start,
        active_stop,
        config,
    )

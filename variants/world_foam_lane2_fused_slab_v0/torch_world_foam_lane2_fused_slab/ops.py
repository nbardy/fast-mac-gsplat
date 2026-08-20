from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from .certificate_binding import (
    NativeFixedWordP0ContinuousCertificateBinding,
    NativeFixedWordP0RuntimeBinding,
    NativeFixedWordP0TrainingTopologyBinding,
    assert_native_fixed_word_p0_certificate_binding,
    assert_native_fixed_word_p0_runtime_binding,
)

MAX_REALRAY_BOUNDARIES = 128
MAX_REALRAY_FUSED_MSE_BOUNDARIES = 256
MAX_REALRAY_FUSED_MSE_CAP224_BOUNDARIES = 224

_EXTENSION_LOAD_ERROR: Exception | None = None
_EXTENSION_LIBRARY_PATH: Path | None = None

_NATIVE_NAMESPACE = "world_foam_lane2_fused_slab_v0"
_KINETIC_MEMORY_LIGHT_RESOURCE_ATTESTATION_OP_NAME = (
    "kinetic_memory_light_selected_kernel_resource_attestation"
)
_KINETIC_MEMORY_LIGHT_SELECTED_KERNELS = (
    (
        "kinetic_precompiled_length_p0_lie_node_forward_into_launch_only_v1",
        "wf2_kinetic_precompiled_length_p0_lie_node_forward_tensor",
    ),
    (
        "kinetic_ragged_p0_lie_sample_accumulate_loss_only_launch_only",
        "wf2_kinetic_ragged_p0_lie_sample_mse_vjp_accumulate_only_tensor",
    ),
    (
        "kinetic_precompiled_length_p0_lie_material_node_vjp_accumulate_launch_only",
        "wf2_kinetic_precompiled_length_p0_lie_material_node_vjp_tensor",
    ),
)
_KINETIC_MEMORY_LIGHT_COMPILED_SCHEMAS = (
    (
        _KINETIC_MEMORY_LIGHT_RESOURCE_ATTESTATION_OP_NAME,
        "kinetic_memory_light_selected_kernel_resource_attestation(Tensor dispatch_anchor) -> (str[], str[], int[], int[], int[])",
    ),
    (
        "kinetic_precompiled_length_p0_lie_node_forward_launch_only",
        "kinetic_precompiled_length_p0_lie_node_forward_launch_only(Tensor word_offsets_i32, Tensor word_owner_i32, Tensor node_physical_length_f32, Tensor site_rgba_f32, Tensor config_i32, Tensor config_f32, int track_count, int node_count) -> Tensor",
    ),
    (
        "kinetic_precompiled_length_p0_lie_node_forward_into_launch_only_v1",
        "kinetic_precompiled_length_p0_lie_node_forward_into_launch_only_v1(Tensor word_offsets_i32, Tensor word_owner_i32, Tensor node_physical_length_f32, Tensor site_rgba_f32, Tensor config_i32, Tensor config_f32, Tensor(a!) node_chart_out_f32, int track_count, int node_count) -> Tensor(a!)",
    ),
    (
        "kinetic_ragged_p0_lie_sample_accumulate_loss_only_launch_only",
        "kinetic_ragged_p0_lie_sample_accumulate_loss_only_launch_only(Tensor node_chart_f32, Tensor sample_row_i32, Tensor sample_to_node_f32, Tensor target_rgb_f32, Tensor background_rgb_f32, Tensor(a!) loss_f32, Tensor(b!) grad_node_chart_f32, Tensor(c!) cone_diagnostic_i32, Tensor config_i32, Tensor config_f32, int row_count, int node_count, int sample_count) -> (Tensor(a!), Tensor(b!), Tensor(c!))",
    ),
    (
        "kinetic_precompiled_length_p0_lie_node_vjp_accumulate_launch_only",
        "kinetic_precompiled_length_p0_lie_node_vjp_accumulate_launch_only(Tensor word_offsets_i32, Tensor word_owner_i32, Tensor node_physical_length_f32, Tensor site_rgba_f32, Tensor node_chart_f32, Tensor grad_node_chart_f32, Tensor(a!) grad_site_rgba_f32, Tensor config_i32, Tensor config_f32, int track_count, int node_count) -> (Tensor(a!), Tensor)",
    ),
    (
        "kinetic_precompiled_length_p0_lie_material_node_vjp_accumulate_launch_only",
        "kinetic_precompiled_length_p0_lie_material_node_vjp_accumulate_launch_only(Tensor word_offsets_i32, Tensor word_owner_i32, Tensor node_physical_length_f32, Tensor site_rgba_f32, Tensor node_chart_f32, Tensor grad_node_chart_f32, Tensor(a!) grad_site_rgba_f32, Tensor config_i32, Tensor config_f32, int track_count, int node_count) -> Tensor(a!)",
    ),
)

# Exact compiled operators exercised by the two production full-geometry
# reverse paths.  Keep this separate from the material-only memory-light ABI:
# callers must opt in to the stronger contract instead of inferring geometry
# support from Python wrappers with the same public names.
_KINETIC_LAZY_FULL_GEOMETRY_COMPILED_SCHEMAS = (
    (
        "kinetic_precompiled_length_p0_lie_node_vjp_accumulate_launch_only",
        "kinetic_precompiled_length_p0_lie_node_vjp_accumulate_launch_only(Tensor word_offsets_i32, Tensor word_owner_i32, Tensor node_physical_length_f32, Tensor site_rgba_f32, Tensor node_chart_f32, Tensor grad_node_chart_f32, Tensor(a!) grad_site_rgba_f32, Tensor config_i32, Tensor config_f32, int track_count, int node_count) -> (Tensor(a!), Tensor)",
    ),
    (
        "kinetic_fused_direct_full_vjp_accumulate_launch_only_v1",
        "kinetic_fused_direct_full_vjp_accumulate_launch_only_v1(Tensor word_offsets_i32, Tensor word_owner_i32, Tensor source_site_ids_i64, Tensor node_physical_length_f32, Tensor site_rgba_f32, Tensor node_chart_f32, Tensor row_node_time_f32, Tensor row_near_far_f32, Tensor row_ray_coeff_f32, Tensor compact_positions0_f32, Tensor compact_velocities_f32, Tensor compact_weight_coefficients_f32, Tensor grad_node_chart_f32, Tensor(a!) grad_site_rgba_f32, Tensor(b!) grad_global_positions0_f32, Tensor(c!) grad_global_velocities_f32, Tensor(d!) grad_global_weight_coefficients_f32, Tensor config_i32, Tensor config_f32, int row_count, int node_count) -> (Tensor(a!), Tensor(b!), Tensor(c!), Tensor(d!), Tensor)",
    ),
    (
        "kinetic_fused_direct_full_vjp_validate_shared_status_launch_only_v1",
        "kinetic_fused_direct_full_vjp_validate_shared_status_launch_only_v1(Tensor word_offsets_i32, Tensor word_owner_i32, Tensor source_site_ids_i64, Tensor node_physical_length_f32, Tensor site_rgba_f32, Tensor node_chart_f32, Tensor row_node_time_f32, Tensor row_near_far_f32, Tensor row_ray_coeff_f32, Tensor compact_positions0_f32, Tensor compact_velocities_f32, Tensor compact_weight_coefficients_f32, Tensor grad_node_chart_f32, Tensor grad_site_rgba_f32, Tensor grad_global_positions0_f32, Tensor grad_global_velocities_f32, Tensor grad_global_weight_coefficients_f32, Tensor config_i32, Tensor config_f32, Tensor(e!) validation_status_i32, bool validate_shared_global_ledgers, int row_count, int node_count) -> Tensor(e!)",
    ),
    (
        "kinetic_fused_direct_full_vjp_accumulate_shared_status_launch_only_v1",
        "kinetic_fused_direct_full_vjp_accumulate_shared_status_launch_only_v1(Tensor word_offsets_i32, Tensor word_owner_i32, Tensor source_site_ids_i64, Tensor node_physical_length_f32, Tensor site_rgba_f32, Tensor node_chart_f32, Tensor row_node_time_f32, Tensor row_near_far_f32, Tensor row_ray_coeff_f32, Tensor compact_positions0_f32, Tensor compact_velocities_f32, Tensor compact_weight_coefficients_f32, Tensor grad_node_chart_f32, Tensor(a!) grad_site_rgba_f32, Tensor(b!) grad_global_positions0_f32, Tensor(c!) grad_global_velocities_f32, Tensor(d!) grad_global_weight_coefficients_f32, Tensor config_i32, Tensor config_f32, Tensor(e) validation_status_i32, int row_count, int node_count) -> (Tensor(a!), Tensor(b!), Tensor(c!), Tensor(d!), Tensor(e))",
    ),
    (
        "kinetic_fused_direct_full_vjp_finalize_shared_status_launch_only_v1",
        "kinetic_fused_direct_full_vjp_finalize_shared_status_launch_only_v1(Tensor(a) grad_site_rgba_f32, Tensor(b) grad_global_positions0_f32, Tensor(c) grad_global_velocities_f32, Tensor(d) grad_global_weight_coefficients_f32, Tensor(e!) validation_status_i32, bool finalize_shared_global_ledgers) -> (Tensor(a), Tensor(b), Tensor(c), Tensor(d), Tensor(e!))",
    ),
    (
        "kinetic_fused_union_full_vjp_validate_shared_status_launch_only_v2",
        "kinetic_fused_union_full_vjp_validate_shared_status_launch_only_v2(Tensor word_offsets_i32, Tensor word_owner_i32, Tensor source_site_ids_i64, Tensor compact_to_geometry_output_i64, Tensor geometry_output_source_site_ids_i64, Tensor node_physical_length_f32, Tensor site_rgba_f32, Tensor node_chart_f32, Tensor row_node_time_f32, Tensor row_near_far_f32, Tensor row_ray_coeff_f32, Tensor compact_positions0_f32, Tensor compact_velocities_f32, Tensor compact_weight_coefficients_f32, Tensor grad_node_chart_f32, Tensor grad_site_rgba_f32, Tensor grad_union_positions0_f32, Tensor grad_union_velocities_f32, Tensor grad_union_weight_coefficients_f32, Tensor config_i32, Tensor config_f32, Tensor(e!) validation_status_i32, bool validate_shared_union_ledgers, int global_site_count, int union_site_count, int row_count, int node_count) -> Tensor(e!)",
    ),
    (
        "kinetic_fused_union_full_vjp_accumulate_shared_status_launch_only_v2",
        "kinetic_fused_union_full_vjp_accumulate_shared_status_launch_only_v2(Tensor word_offsets_i32, Tensor word_owner_i32, Tensor source_site_ids_i64, Tensor compact_to_geometry_output_i64, Tensor geometry_output_source_site_ids_i64, Tensor node_physical_length_f32, Tensor site_rgba_f32, Tensor node_chart_f32, Tensor row_node_time_f32, Tensor row_near_far_f32, Tensor row_ray_coeff_f32, Tensor compact_positions0_f32, Tensor compact_velocities_f32, Tensor compact_weight_coefficients_f32, Tensor grad_node_chart_f32, Tensor(a!) grad_site_rgba_f32, Tensor(b!) grad_union_positions0_f32, Tensor(c!) grad_union_velocities_f32, Tensor(d!) grad_union_weight_coefficients_f32, Tensor config_i32, Tensor config_f32, Tensor(e) validation_status_i32, int global_site_count, int union_site_count, int row_count, int node_count) -> (Tensor(a!), Tensor(b!), Tensor(c!), Tensor(d!), Tensor(e))",
    ),
    (
        "kinetic_fused_union_full_vjp_finalize_shared_status_launch_only_v2",
        "kinetic_fused_union_full_vjp_finalize_shared_status_launch_only_v2(Tensor(a) grad_site_rgba_f32, Tensor(b) grad_union_positions0_f32, Tensor(c) grad_union_velocities_f32, Tensor(d) grad_union_weight_coefficients_f32, Tensor(e!) validation_status_i32, bool finalize_shared_union_ledgers, int union_site_count) -> (Tensor(a), Tensor(b), Tensor(c), Tensor(d), Tensor(e!))",
    ),
)


def _load_extension_library() -> None:
    global _EXTENSION_LIBRARY_PATH
    candidates = sorted(Path(__file__).resolve().parent.glob("_C*.so"))
    if not candidates:
        return
    if len(candidates) != 1:
        raise RuntimeError(
            f"expected exactly one native extension library, found {len(candidates)}"
        )
    _EXTENSION_LIBRARY_PATH = candidates[0].resolve()
    torch.ops.load_library(str(_EXTENSION_LIBRARY_PATH))


try:
    _load_extension_library()
except Exception as exc:
    _EXTENSION_LOAD_ERROR = exc


def _compiled_source_paths() -> tuple[Path, ...]:
    variant_dir = Path(__file__).resolve().parent.parent
    metal_host = variant_dir / "csrc" / "metal" / "world_foam_lane2_metal.mm"
    sources = [variant_dir / "csrc" / "bindings.cpp", metal_host]
    if metal_host.exists():
        host_source = metal_host.read_text(encoding="utf-8")
        sources.extend(
            variant_dir / "csrc" / "metal" / filename
            for filename in sorted(
                set(
                    re.findall(
                        r'stringByAppendingPathComponent:@"([^\"]+\.metal)"',
                        host_source,
                    )
                )
            )
        )
    return tuple(sources)


def _assert_compiled_schemas_registered(
    schemas: tuple[tuple[str, str], ...],
) -> None:
    for name, unqualified_schema in schemas:
        qualified_name = f"{_NATIVE_NAMESPACE}::{name}"
        try:
            handle = torch._C._dispatch_find_schema_or_throw(qualified_name, "")
        except RuntimeError as exc:
            raise RuntimeError(
                f"required compiled schema is not registered: {qualified_name}"
            ) from exc
        expected_schema = f"{_NATIVE_NAMESPACE}::{unqualified_schema}"
        actual_schema = str(handle.schema())
        if actual_schema != expected_schema:
            raise RuntimeError(
                f"compiled schema mismatch for {qualified_name}: {actual_schema!r}"
            )
        if not torch._C._dispatch_has_kernel_for_dispatch_key(
            qualified_name,
            "CompositeExplicitAutograd",
        ):
            raise RuntimeError(
                f"compiled operator has no CompositeExplicitAutograd kernel: {qualified_name}"
            )


def assert_kinetic_memory_light_compiled_abi_registered() -> None:
    """Fail cold when Python wrappers mask a missing or stale compiled ABI."""

    candidates = tuple(
        sorted(Path(__file__).resolve().parent.glob("_C*.so"))
    )
    if len(candidates) != 1:
        raise RuntimeError(
            f"kinetic memory-light ABI requires one _C library, found {len(candidates)}"
        )
    selected = candidates[0].resolve()
    if _EXTENSION_LOAD_ERROR is not None:
        raise RuntimeError("native extension failed to load") from _EXTENSION_LOAD_ERROR
    if _EXTENSION_LIBRARY_PATH != selected:
        raise RuntimeError("loaded native extension does not match the selected library")

    sources = _compiled_source_paths()
    missing_sources = tuple(path for path in sources if not path.exists())
    if missing_sources:
        raise RuntimeError(f"native compiled sources are missing: {missing_sources}")
    newest_source_mtime = max(path.stat().st_mtime for path in sources)
    if selected.stat().st_mtime < newest_source_mtime:
        raise RuntimeError("native extension is older than its compiled sources")

    _assert_compiled_schemas_registered(_KINETIC_MEMORY_LIGHT_COMPILED_SCHEMAS)


def assert_kinetic_lazy_full_geometry_compiled_abi_registered() -> None:
    """Attest the exact staged and union-v2 full-geometry compiled ABI.

    This deliberately calls the base memory-light attestation first so one
    invocation binds the selected library, source freshness, forward/loss
    operators, and both full-geometry reverse implementations.
    """

    assert_kinetic_memory_light_compiled_abi_registered()
    _assert_compiled_schemas_registered(
        _KINETIC_LAZY_FULL_GEOMETRY_COMPILED_SCHEMAS
    )


@dataclass(frozen=True)
class PowerBoundaryConfig:
    camera_velocity_x: float
    invalid_epsilon: float = 1.0e-7


@dataclass(frozen=True)
class RealRayReplayConfig:
    near: float
    far: float
    invalid_epsilon: float = 1.0e-7
    transmittance_threshold: float = 1.0e-4


@dataclass(frozen=True)
class KineticMemoryLightSelectedMetalKernelResource:
    """Observable pipeline properties for one selected material-only kernel.

    The three integer properties come directly from the compiled
    ``MetalKernelFunction``. Metal/PyTorch exposes no corresponding query for
    register allocation, per-thread private storage, or compiler spills, so
    those fields stay explicitly unobservable instead of being estimated.
    """

    operator_name: str
    metal_function_name: str
    max_threads_per_threadgroup: int
    thread_execution_width: int
    static_threadgroup_memory_length_bytes: int
    queried_from_compiled_metal_kernel_function: bool = True
    static_threadgroup_memory_length_observable: bool = True
    register_bytes_per_thread: None = None
    register_bytes_per_thread_observable: bool = False
    private_memory_bytes_per_thread: None = None
    private_memory_bytes_per_thread_observable: bool = False
    compiler_spill_bytes: None = None
    compiler_spill_bytes_observable: bool = False


@dataclass(frozen=True)
class KineticMemoryLightSelectedKernelResourceAttestation:
    """ABI/source-fresh report for the exact material-only custom kernels."""

    kernels: tuple[KineticMemoryLightSelectedMetalKernelResource, ...]
    abi_namespace: str = _NATIVE_NAMESPACE
    compiled_operator_name: str = _KINETIC_MEMORY_LIGHT_RESOURCE_ATTESTATION_OP_NAME
    selected_execution_path: str = "kinetic_material_only"
    queried_properties: tuple[str, ...] = (
        "MetalKernelFunction::getMaxThreadsPerThreadgroup()",
        "MetalKernelFunction::getThreadExecutionWidth()",
        "MetalKernelFunction::getStaticThreadGroupMemoryLength()",
    )
    compiled_abi_schema_verified: bool = True
    compiled_source_mtime_gate_passed: bool = True
    optional_full_geometry_vjp_included: bool = False
    kernel_execution_verified_by_this_query: bool = False
    native_private_or_spill_bytes_measured: bool = False


def kinetic_memory_light_selected_kernel_resource_attestation(
    dispatch_anchor: Tensor,
) -> KineticMemoryLightSelectedKernelResourceAttestation:
    """Query selected pipeline properties without inventing private-memory data.

    ``dispatch_anchor`` must be an already-live MPS tensor. It selects the
    registered CompositeExplicitAutograd kernel but is otherwise untouched.
    A measurement driver should call this after its cold step (or inside the
    cold-compile phase), so querying the dynamic Metal library does not
    accidentally warm the compile path before that phase is measured.
    """

    if not isinstance(dispatch_anchor, Tensor):
        raise TypeError("dispatch_anchor must be a torch.Tensor")
    if dispatch_anchor.device.type != "mps":
        raise ValueError("dispatch_anchor must be on MPS")
    assert_kinetic_memory_light_compiled_abi_registered()

    raw = getattr(
        getattr(torch.ops, _NATIVE_NAMESPACE),
        _KINETIC_MEMORY_LIGHT_RESOURCE_ATTESTATION_OP_NAME,
    )(dispatch_anchor)
    if not isinstance(raw, tuple) or len(raw) != 5:
        raise RuntimeError("selected-kernel resource attestation returned an invalid ABI tuple")
    (
        raw_operator_names,
        raw_metal_function_names,
        raw_max_threads,
        raw_execution_widths,
        raw_static_threadgroup_bytes,
    ) = raw
    vectors = (
        raw_operator_names,
        raw_metal_function_names,
        raw_max_threads,
        raw_execution_widths,
        raw_static_threadgroup_bytes,
    )
    expected_count = len(_KINETIC_MEMORY_LIGHT_SELECTED_KERNELS)
    if any(len(values) != expected_count for values in vectors):
        raise RuntimeError("selected-kernel resource attestation vector lengths do not match")
    if not all(isinstance(value, str) for value in raw_operator_names):
        raise RuntimeError("selected-kernel operator names must be strings")
    if not all(isinstance(value, str) for value in raw_metal_function_names):
        raise RuntimeError("selected Metal function names must be strings")

    operator_names = tuple(raw_operator_names)
    metal_function_names = tuple(raw_metal_function_names)
    if tuple(zip(operator_names, metal_function_names, strict=True)) != (
        _KINETIC_MEMORY_LIGHT_SELECTED_KERNELS
    ):
        raise RuntimeError("selected-kernel names do not match the sealed material-only path")

    numeric_vectors = tuple(
        tuple(int(value) for value in values)
        for values in (
            raw_max_threads,
            raw_execution_widths,
            raw_static_threadgroup_bytes,
        )
    )
    max_threads, execution_widths, static_threadgroup_bytes = numeric_vectors
    if any(value <= 0 for value in max_threads):
        raise RuntimeError("max threads per threadgroup must be positive")
    if any(value <= 0 for value in execution_widths):
        raise RuntimeError("thread execution width must be positive")
    if any(
        execution_width > maximum
        for maximum, execution_width in zip(
            max_threads,
            execution_widths,
            strict=True,
        )
    ):
        raise RuntimeError("thread execution width exceeds the kernel threadgroup maximum")
    if any(value < 0 for value in static_threadgroup_bytes):
        raise RuntimeError("static threadgroup memory length cannot be negative")

    kernels = tuple(
        KineticMemoryLightSelectedMetalKernelResource(
            operator_name=operator_name,
            metal_function_name=metal_function_name,
            max_threads_per_threadgroup=maximum,
            thread_execution_width=execution_width,
            static_threadgroup_memory_length_bytes=static_bytes,
        )
        for (
            operator_name,
            metal_function_name,
            maximum,
            execution_width,
            static_bytes,
        ) in zip(
            operator_names,
            metal_function_names,
            max_threads,
            execution_widths,
            static_threadgroup_bytes,
            strict=True,
        )
    )
    return KineticMemoryLightSelectedKernelResourceAttestation(kernels=kernels)


@dataclass(frozen=True)
class _CheckedFixedWordP0FusedCall:
    """Checked storage for the legacy single-block fused convenience call.

    This is not the production streaming token: it intentionally bundles one
    chart and one sample block. Production code uses the four separated tokens
    below so a K block never owns or rebuilds topology/world storage.
    """

    boundary_f32: Tensor
    track_ray_coeff_f32: Tensor
    compiler_node_t_f32: Tensor
    word_offsets_i32: Tensor
    word_owner_i32: Tensor
    word_left_incidence_i32: Tensor
    word_right_incidence_i32: Tensor
    track_incidence_offsets_i32: Tensor
    incidence_boundary_i32: Tensor
    site_rgba_f32: Tensor
    sample_to_node_f32: Tensor
    target_rgb_f32: Tensor
    background_rgb_f32: Tensor
    config_i32: Tensor
    config_f32: Tensor
    boundary_count: int
    track_count: int
    node_count: int
    sample_count: int
    site_count: int
    word_count: int
    incidence_count: int


@dataclass(frozen=True)
class FixedWordP0TopologyToken:
    """Immutable fixed-word topology bound to a real CPU acceptance snapshot.

    The bound continuous owner certificate compares every claimed owner with
    every power site at both segment endpoints over the full time interval.
    A native token therefore cannot promote a merely assumed fixed word.
    """

    word_offsets_i32: Tensor
    word_owner_i32: Tensor
    word_left_incidence_i32: Tensor
    word_right_incidence_i32: Tensor
    track_incidence_offsets_i32: Tensor
    incidence_boundary_i32: Tensor
    boundary_site_pairs_i32: Tensor
    active_boundary_site_pairs_i32: Tensor
    certificate_binding: NativeFixedWordP0RuntimeBinding
    continuous_certificate_digest: str
    topology_generation_id: str
    boundary_count: int
    track_count: int
    site_count: int
    word_count: int
    incidence_count: int
    tensor_signatures: tuple[tuple[Any, ...], ...]
    training_binding_digest: str = ""
    binding_mode: str = "strict_frozen_evaluation"
    paper_evidence_eligible: bool = True
    transfer_jacobian_certified: bool = True


@dataclass(frozen=True)
class FixedWordP0WorldRefreshToken:
    """One world refresh with boundaries derived from its exact resident sites."""

    topology: FixedWordP0TopologyToken
    sites_f32: Tensor
    site_rgba_f32: Tensor
    track_ray_coeff_f32: Tensor
    boundary_f32: Tensor
    mobius_coeff_f32: Tensor
    config_i32: Tensor
    config: RealRayReplayConfig
    physical_length_epsilon: float
    cone_tolerance: float
    world_generation_id: str
    tensor_signatures: tuple[tuple[Any, ...], ...]
    binding_mode: str = "strict_frozen_evaluation"
    paper_evidence_eligible: bool = True
    transfer_jacobian_certified: bool = True


@dataclass(frozen=True)
class FixedWordP0ChartToken:
    """Per-chart node storage bound to a sealed continuous certificate."""

    world: FixedWordP0WorldRefreshToken
    compiler_node_t_f32: Tensor
    node_chart_f32: Tensor
    config_i32: Tensor
    config_f32: Tensor
    continuous_certificate_digest: str
    chart_index: int
    chart_generation_id: str
    node_count: int
    tensor_signatures: tuple[tuple[Any, ...], ...]
    training_binding_digest: str = ""
    binding_mode: str = "strict_frozen_evaluation"
    paper_evidence_eligible: bool = True
    transfer_jacobian_certified: bool = True


@dataclass
class _FixedWordP0SampleLedger:
    """Constant-size host lifecycle guard for one deterministic K partition."""

    sample_block_size: int
    expected_block_count: int
    next_global_sample_start: int
    consumed_block_count: int = 0
    accumulated_element_count: int = 0
    finalized: bool = False
    state_tensor_signatures: tuple[tuple[Any, ...], ...] = ()


@dataclass(frozen=True)
class FixedWordP0SampleStateToken:
    """Per-chart accumulation state with one immutable global normalization."""

    chart: FixedWordP0ChartToken
    loss_f32: Tensor
    grad_node_chart_f32: Tensor
    cone_diagnostic_i32: Tensor
    loss_normalization_id: str
    sample_partition_generation_id: str
    global_track_count: int
    global_sample_count: int
    global_sample_start: int
    global_sample_end: int
    global_loss_element_count: int
    expected_local_element_count: int
    global_loss_scale: float
    ledger: _FixedWordP0SampleLedger


@dataclass(frozen=True)
class FixedWordP0SampleBlockToken:
    """Cheap per-K sample data; contains no topology or site-density copies."""

    sample_state: FixedWordP0SampleStateToken
    sample_to_node_f32: Tensor
    target_rgb_f32: Tensor
    background_rgb_f32: Tensor
    config_i32: Tensor
    config_f32: Tensor
    sample_block_id: str
    global_sample_start: int
    global_sample_end: int
    sample_count: int
    element_count: int
    sample_weight_evaluation: str
    sample_weight_linear_interactions: int
    sample_weight_dense_fallback_interactions: int
    sample_weight_exact_node_rows: int
    sample_weight_dense_fallback_rows: int
    tensor_signatures: tuple[tuple[Any, ...], ...]


@dataclass
class _FixedWordP0WorldGradLedger:
    """Host-only reverse lifecycle for one exact world refresh."""

    expected_chart_generation_ids: frozenset[str]
    expected_chart_ranges: tuple[tuple[str, int, int], ...]
    reversed_chart_generation_ids: set[str] = field(default_factory=set)
    boundary_finalized: bool = False
    site_finalized: bool = False
    grad_tensor_signatures: tuple[tuple[Any, ...], ...] = ()


@dataclass
class _FixedWordP0MaterialWorldGradLedger:
    """Host-only reverse guard for the RGBA-only training capability."""

    expected_chart_generation_ids: frozenset[str]
    expected_chart_ranges: tuple[tuple[str, int, int], ...]
    reversed_chart_generation_ids: set[str] = field(default_factory=set)
    finalized: bool = False
    grad_tensor_signatures: tuple[tuple[Any, ...], ...] = ()


@dataclass(frozen=True)
class FixedWordP0WorldGradToken:
    """World-bound shared adjoints; raw buffers never cross token boundaries."""

    world: FixedWordP0WorldRefreshToken
    grad_site_rgba_f32: Tensor
    grad_mobius_coeff_f32: Tensor
    grad_boundary_f32: Tensor
    loss_normalization_id: str
    sample_partition_generation_id: str
    global_track_count: int
    global_sample_count: int
    global_loss_element_count: int
    ledger: _FixedWordP0WorldGradLedger


@dataclass(frozen=True)
class FixedWordP0MaterialWorldGradToken:
    """Shared RGBA adjoint with no geometry-side reverse allocations."""

    world: FixedWordP0WorldRefreshToken
    grad_site_rgba_f32: Tensor
    loss_normalization_id: str
    sample_partition_generation_id: str
    global_track_count: int
    global_sample_count: int
    global_loss_element_count: int
    ledger: _FixedWordP0MaterialWorldGradLedger
    geometry_vjp_executed: bool = False


@dataclass(frozen=True)
class PreparedSparsePowerBoundarySiteVjp:
    """Checked resident topology and sites for the final sparse site scatter."""

    active_boundary_site_pairs_i32: Tensor
    sites_f32: Tensor
    boundary_count: int


@dataclass(frozen=True)
class PreparedKineticRaggedP0LieSampleBlock:
    """Cold-validated ephemeral N-sample block over (track, chart) rows.

    The mutable reducer state is deliberately not stored here.  Dropping this
    block after launch releases all sample-axis tensors; the surviving loss,
    node cotangent, and diagnostics have shapes independent of N.
    """

    node_chart_f32: Tensor
    sample_row_i32: Tensor
    sample_to_node_f32: Tensor
    target_rgb_f32: Tensor
    background_rgb_f32: Tensor
    config_i32: Tensor
    config_f32: Tensor
    row_count: int
    node_count: int
    sample_count: int
    tensor_signatures: tuple[tuple[Any, ...], ...]


@dataclass(frozen=True)
class PreparedKineticFusedDirectFullVjpV1:
    """Raw, unpromoted fixed-camera token for the fused kinetic full VJP.

    This suffixed ABI is intentionally not part of the selected, rebuilt
    kinetic ABI yet.  It must first be rebuilt and checked against the staged
    certified sparse reducer.  The token retains the primal ``[J,W]`` physical
    lengths, but it has no length-cotangent storage: the native reverse carries
    only adjacent ``previous_bar_ell``/``current_bar_ell`` scalars.  This raw
    preparer performs structural validation only; the sealed equal-rank runtime
    adapter owns compiler/world/certificate provenance admission.
    """

    word_offsets_i32: Tensor
    word_owner_i32: Tensor
    source_site_ids_i64: Tensor
    node_physical_length_f32: Tensor
    site_rgba_f32: Tensor
    node_chart_f32: Tensor
    row_node_time_f32: Tensor
    row_near_far_f32: Tensor
    row_ray_coeff_f32: Tensor
    compact_positions0_f32: Tensor
    compact_velocities_f32: Tensor
    compact_weight_coefficients_f32: Tensor
    config_i32: Tensor
    config_f32: Tensor
    row_count: int
    node_count: int
    word_count: int
    compact_site_count: int
    global_site_count: int
    weight_coefficient_count: int
    tensor_owned_by_preparer: tuple[bool, ...]
    retained_logical_tensor_bytes: int
    preparer_owned_logical_tensor_bytes: int
    tensor_signatures: tuple[tuple[Any, ...], ...]
    persistent_frame_tensor_bytes: int = 0
    persistent_sample_tensor_bytes: int = 0
    persistent_target_tensor_bytes: int = 0
    persistent_prediction_tensor_bytes: int = 0
    runtime_status: str = (
        "raw_fixed_camera_source_only_until_native_rebuild_and_sparse_oracle_parity"
    )


@dataclass(frozen=True)
class KineticFusedDirectFullVjpResultV1:
    """Unaccepted fixed-camera bars plus the native validation receipt.

    The four bar tensors remain caller-owned, but they are not consumable until
    :meth:`accepted_bars` fences and accepts the scalar status. Preflight also
    requires every output element to be finite and exactly zero. Invalid
    preflight input rejects before every atomic and leaves the bars unchanged.
    A postwrite nonfinite reason can mean finite atomic contributions overflowed
    a shared destination; in that case the bars are mutated disposable scratch
    and must be quarantined.  Zero proves both the preflight and postwrite
    finiteness gates passed, but not exact floating-point summation. The caller
    must additionally own these tensors as fresh single-use scratch,
    quarantine them after any rejected or interrupted transaction, and defer
    every persistent/optimizer commit until the accepted receipt is returned;
    storage checks cannot prove the absence of hidden aliases.

    This raw/combined convenience result is not the prepared multi-block
    transaction token and does not carry its single-use, abort-settle, or
    quarantine lifetime proof. It therefore remains unpromoted and is not by
    itself optimizer authorization.
    """

    grad_site_rgba_f32: Tensor
    grad_global_positions0_f32: Tensor
    grad_global_velocities_f32: Tensor
    grad_global_weight_coefficients_f32: Tensor
    validation_status_i32: Tensor
    accumulation_enqueued: bool
    finalization_enqueued: bool
    shared_status_reused: bool
    runtime_status: str = (
        "raw_fixed_camera_source_only_until_native_rebuild_and_sparse_oracle_parity"
    )

    @property
    def validation_status_tensor_bytes(self) -> int:
        return self.validation_status_i32.numel() * self.validation_status_i32.element_size()

    def validation_reason_mask(self) -> int:
        """Fence the native stream and read the bounded validation receipt."""

        if (
            not isinstance(self.validation_status_i32, Tensor)
            or self.validation_status_i32.device.type != "mps"
            or self.validation_status_i32.dtype != torch.int32
            or tuple(self.validation_status_i32.shape) != (1,)
            or not self.validation_status_i32.is_contiguous()
            or self.validation_status_tensor_bytes != 4
        ):
            raise RuntimeError("fused kinetic validation receipt has an invalid scalar ABI")
        return int(self.validation_status_i32.item())

    def accepted_bars(self) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Return finalized scratch bars only after accepting a zero mask."""

        if not self.accumulation_enqueued:
            raise RuntimeError(
                "fused kinetic phase receipt has no locally proven accumulation"
            )
        if not self.finalization_enqueued:
            raise RuntimeError(
                "fused kinetic accumulation-only receipt has not passed postwrite finalization"
            )
        reason_mask = self.validation_reason_mask()
        if reason_mask != 0:
            raise RuntimeError(
                f"fused kinetic transaction rejected scratch bars with reason mask 0x{reason_mask:02x}"
            )
        return (
            self.grad_site_rgba_f32,
            self.grad_global_positions0_f32,
            self.grad_global_velocities_f32,
            self.grad_global_weight_coefficients_f32,
        )


@dataclass(frozen=True)
class PreparedKineticFusedUnionFullVjpV2:
    """Cold-bound exact union-index-space factorization of one v1 block.

    ``direct_v1_oracle`` retains the unchanged primal/oracle token. The two
    additional identities are deliberately not folded into its global source
    ids: compact rows name immutable global sources, compact-to-output names
    transaction-local union destinations, and output-source ids prove how that
    union embeds back into the global world. This is source-only until rebuild,
    v1/staged parity, and allocator evidence.

    If a cold CPU identity required an asynchronous device copy, the original
    CPU tensor is retained as a transfer predecessor. It cannot be released
    until a higher-level adapter proves the copy/launch completion fence; this
    raw token intentionally has no early-release operation.
    """

    direct_v1_oracle: PreparedKineticFusedDirectFullVjpV1
    compact_to_geometry_output_i64: Tensor
    geometry_output_source_site_ids_i64: Tensor
    compact_to_geometry_output_transfer_source: Tensor
    geometry_output_source_site_ids_transfer_source: Tensor
    config_i32: Tensor
    global_site_count: int
    union_site_count: int
    mapping_tensor_owned_by_preparer: tuple[bool, bool]
    transfer_predecessor_logical_tensor_bytes: int
    transfer_predecessor_release_requires_proven_fence: bool
    retained_logical_tensor_bytes: int
    preparer_owned_logical_tensor_bytes: int
    tensor_signatures: tuple[tuple[Any, ...], ...]
    geometry_output_index_space: str = "request_union"
    factorization_identity: str = "P_b=P_U*Q_b"
    runtime_status: str = (
        "raw_union_v2_source_only_until_native_rebuild_v1_sparse_parity_and_allocator_evidence"
    )


@dataclass(frozen=True)
class KineticFusedUnionFullVjpResultV2:
    """Aliases from one raw split union-v2 phase and its shared receipt."""

    grad_site_rgba_f32: Tensor
    grad_union_positions0_f32: Tensor
    grad_union_velocities_f32: Tensor
    grad_union_weight_coefficients_f32: Tensor
    validation_status_i32: Tensor
    accumulation_enqueued: bool
    finalization_enqueued: bool
    shared_status_reused: bool = True
    geometry_output_index_space: str = "request_union"
    runtime_status: str = (
        "raw_union_v2_source_only_until_native_rebuild_v1_sparse_parity_and_allocator_evidence"
    )

    def validation_reason_mask(self) -> int:
        if (
            not isinstance(self.validation_status_i32, Tensor)
            or self.validation_status_i32.device.type != "mps"
            or self.validation_status_i32.dtype != torch.int32
            or tuple(self.validation_status_i32.shape) != (1,)
            or not self.validation_status_i32.is_contiguous()
            or self.validation_status_i32.numel()
            * self.validation_status_i32.element_size()
            != 4
        ):
            raise RuntimeError("union-v2 validation receipt has an invalid scalar ABI")
        return int(self.validation_status_i32.item())

    def accepted_bars(self) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        if not self.accumulation_enqueued or not self.finalization_enqueued:
            raise RuntimeError(
                "union-v2 bars require locally proven accumulation and postwrite finalization"
            )
        reason_mask = self.validation_reason_mask()
        if reason_mask != 0:
            raise RuntimeError(
                f"union-v2 transaction rejected scratch bars with reason mask 0x{reason_mask:03x}"
            )
        return (
            self.grad_site_rgba_f32,
            self.grad_union_positions0_f32,
            self.grad_union_velocities_f32,
            self.grad_union_weight_coefficients_f32,
        )


def _tensor_mutation_signature(tensor: Tensor) -> tuple[Any, ...]:
    """Capture metadata/version only; never copy or synchronize tensor values."""
    return (
        tensor.untyped_storage().data_ptr(),
        tensor._version,
        tuple(tensor.shape),
        tuple(tensor.stride()),
        tensor.storage_offset(),
        tensor.dtype,
        tensor.device,
    )


def _same_exact_tensor_view(left: Tensor, right: Tensor) -> bool:
    """Compare one returned alias without reading or copying tensor values."""

    return (
        left.untyped_storage().data_ptr() == right.untyped_storage().data_ptr()
        and left.storage_offset() == right.storage_offset()
        and tuple(left.shape) == tuple(right.shape)
        and tuple(left.stride()) == tuple(right.stride())
        and left.dtype == right.dtype
        and left.device == right.device
    )


def _capture_tensor_signatures(tensors: tuple[Tensor, ...]) -> tuple[tuple[Any, ...], ...]:
    return tuple(_tensor_mutation_signature(tensor) for tensor in tensors)


def _validate_half_open_partition(
    records: tuple[tuple[str, int, int], ...],
    *,
    expected_start: int,
    expected_end: int,
    name: str,
) -> tuple[tuple[str, int, int], ...]:
    """Validate a compact O(blocks) exact tiling without per-sample metadata."""
    if expected_start < 0 or expected_end <= expected_start:
        raise ValueError(f"{name} expected range must be nonempty and half-open")
    if not records:
        raise ValueError(f"{name} must contain at least one range")
    if any(not record_id.strip() for record_id, _, _ in records):
        raise ValueError(f"{name} ids must be nonempty")
    if len({record_id for record_id, _, _ in records}) != len(records):
        raise ValueError(f"{name} ids must be unique")
    ordered = tuple(sorted(records, key=lambda record: (record[1], record[2], record[0])))
    cursor = expected_start
    for _, start, end in ordered:
        if start != cursor or end <= start:
            raise ValueError(f"{name} ranges must tile exactly without gaps or overlaps")
        cursor = end
    if cursor != expected_end:
        raise ValueError(f"{name} ranges must tile exactly without gaps or overlaps")
    return ordered


def _assert_tensor_signatures_current(
    tensors: tuple[Tensor, ...],
    expected: tuple[tuple[Any, ...], ...],
    *,
    token_name: str,
) -> None:
    if _capture_tensor_signatures(tensors) != expected:
        raise ValueError(f"{token_name} tensor storage or mutation version changed; refresh its generation")


def _topology_token_tensors(topology: FixedWordP0TopologyToken) -> tuple[Tensor, ...]:
    return (
        topology.word_offsets_i32,
        topology.word_owner_i32,
        topology.word_left_incidence_i32,
        topology.word_right_incidence_i32,
        topology.track_incidence_offsets_i32,
        topology.incidence_boundary_i32,
        topology.boundary_site_pairs_i32,
        topology.active_boundary_site_pairs_i32,
    )


def _assert_topology_token_current(topology: FixedWordP0TopologyToken) -> None:
    binding = assert_native_fixed_word_p0_runtime_binding(topology.certificate_binding)
    if topology.binding_mode != binding.binding_mode:
        raise ValueError("topology token binding mode changed")
    if topology.paper_evidence_eligible is not binding.paper_evidence_eligible:
        raise ValueError("topology token paper-evidence eligibility changed")
    if topology.transfer_jacobian_certified is not binding.transfer_jacobian_certified:
        raise ValueError("topology token transfer/Jacobian certification changed")
    if type(binding) is NativeFixedWordP0ContinuousCertificateBinding:
        if topology.continuous_certificate_digest != binding.canonical_digest:
            raise ValueError("topology token continuous certificate digest changed")
        if topology.training_binding_digest:
            raise ValueError("strict topology token cannot carry a training binding digest")
    else:
        if topology.continuous_certificate_digest:
            raise ValueError("training topology token cannot claim a continuous transfer certificate")
        if topology.training_binding_digest != binding.canonical_digest:
            raise ValueError("topology token training binding digest changed")
    if topology.topology_generation_id != binding.topology_snapshot_generation:
        raise ValueError("topology token snapshot generation is stale")
    _assert_tensor_signatures_current(
        _topology_token_tensors(topology),
        topology.tensor_signatures,
        token_name="topology token",
    )


def _world_token_tensors(world: FixedWordP0WorldRefreshToken) -> tuple[Tensor, ...]:
    return (
        world.sites_f32,
        world.site_rgba_f32,
        world.track_ray_coeff_f32,
        world.boundary_f32,
        world.mobius_coeff_f32,
        world.config_i32,
    )


def _assert_world_token_current(world: FixedWordP0WorldRefreshToken) -> None:
    _assert_topology_token_current(world.topology)
    if (
        world.binding_mode != world.topology.binding_mode
        or world.paper_evidence_eligible is not world.topology.paper_evidence_eligible
        or world.transfer_jacobian_certified is not world.topology.transfer_jacobian_certified
    ):
        raise ValueError("world refresh token binding capability changed")
    if world.world_generation_id != world.topology.certificate_binding.world_snapshot_generation:
        raise ValueError("world refresh token snapshot generation is stale")
    _assert_tensor_signatures_current(
        _world_token_tensors(world),
        world.tensor_signatures,
        token_name="world refresh token",
    )


def _chart_token_tensors(chart: FixedWordP0ChartToken) -> tuple[Tensor, ...]:
    return (
        chart.compiler_node_t_f32,
        chart.node_chart_f32,
        chart.config_i32,
        chart.config_f32,
    )


def _assert_chart_token_current(chart: FixedWordP0ChartToken) -> None:
    _assert_world_token_current(chart.world)
    binding = chart.world.topology.certificate_binding
    if chart.chart_index < 0 or chart.chart_index >= len(binding.charts):
        raise ValueError("chart token index left the continuous certificate")
    certified_chart = binding.charts[chart.chart_index]
    if (
        chart.binding_mode != binding.binding_mode
        or chart.paper_evidence_eligible is not binding.paper_evidence_eligible
        or chart.transfer_jacobian_certified is not binding.transfer_jacobian_certified
    ):
        raise ValueError("chart token binding capability changed")
    if type(binding) is NativeFixedWordP0ContinuousCertificateBinding:
        if chart.continuous_certificate_digest != binding.canonical_digest:
            raise ValueError("chart token continuous certificate digest changed")
        if chart.training_binding_digest:
            raise ValueError("strict chart token cannot carry a training binding digest")
    else:
        if chart.continuous_certificate_digest:
            raise ValueError("training chart token cannot claim a continuous transfer certificate")
        if chart.training_binding_digest != binding.canonical_digest:
            raise ValueError("chart token training binding digest changed")
    if chart.chart_generation_id != certified_chart.chart_digest:
        raise ValueError("chart token certificate generation is stale")
    _assert_tensor_signatures_current(
        _chart_token_tensors(chart),
        chart.tensor_signatures,
        token_name="chart token",
    )


def _sample_block_tensors(sample_block: FixedWordP0SampleBlockToken) -> tuple[Tensor, ...]:
    return (
        sample_block.sample_to_node_f32,
        sample_block.target_rgb_f32,
        sample_block.background_rgb_f32,
        sample_block.config_i32,
        sample_block.config_f32,
    )


def _assert_sample_block_current(sample_block: FixedWordP0SampleBlockToken) -> None:
    if not sample_block.sample_weight_evaluation.startswith("verified_fit_derived_second_form_barycentric"):
        raise ValueError("sample block token has unverified interpolation-weight provenance")
    if (
        sample_block.sample_weight_linear_interactions
        != sample_block.sample_count * sample_block.sample_state.chart.node_count
        or sample_block.sample_weight_dense_fallback_interactions < 0
        or sample_block.sample_weight_exact_node_rows < 0
        or sample_block.sample_weight_dense_fallback_rows < 0
        or sample_block.sample_weight_exact_node_rows + sample_block.sample_weight_dense_fallback_rows
        > sample_block.sample_count
    ):
        raise ValueError("sample block token interpolation-weight accounting is invalid")
    _assert_tensor_signatures_current(
        _sample_block_tensors(sample_block),
        sample_block.tensor_signatures,
        token_name="sample block token",
    )


def _sample_state_tensors(sample_state: FixedWordP0SampleStateToken) -> tuple[Tensor, ...]:
    return (
        sample_state.loss_f32,
        sample_state.grad_node_chart_f32,
        sample_state.cone_diagnostic_i32,
    )


def _assert_sample_state_current(sample_state: FixedWordP0SampleStateToken) -> None:
    _assert_tensor_signatures_current(
        _sample_state_tensors(sample_state),
        sample_state.ledger.state_tensor_signatures,
        token_name="sample state token",
    )


def _assert_next_sample_block_range(
    sample_state: FixedWordP0SampleStateToken,
    *,
    global_sample_start: int,
    global_sample_end: int,
) -> None:
    ledger = sample_state.ledger
    if ledger.finalized:
        raise ValueError("sample state is already finalized")
    if ledger.consumed_block_count >= ledger.expected_block_count:
        raise ValueError("sample state already consumed every expected K block")
    expected_start = ledger.next_global_sample_start
    expected_end = min(
        expected_start + ledger.sample_block_size,
        sample_state.global_sample_end,
    )
    if (global_sample_start, global_sample_end) != (expected_start, expected_end):
        raise ValueError("sample block range is not the next deterministic K partition")


def _require_sample_state_ready_for_reverse(sample_state: FixedWordP0SampleStateToken) -> None:
    ledger = sample_state.ledger
    if ledger.finalized:
        raise ValueError("sample state was already reversed")
    if (
        ledger.consumed_block_count != ledger.expected_block_count
        or ledger.next_global_sample_start != sample_state.global_sample_end
    ):
        raise ValueError("sample state cannot finalize with missing K blocks")
    if ledger.accumulated_element_count != sample_state.expected_local_element_count:
        raise ValueError("sample state accumulated element count does not match chart-local expectation")


def _world_grad_tensors(world_grad: FixedWordP0WorldGradToken) -> tuple[Tensor, ...]:
    return (
        world_grad.grad_site_rgba_f32,
        world_grad.grad_mobius_coeff_f32,
        world_grad.grad_boundary_f32,
    )


def _assert_world_grad_current(world_grad: FixedWordP0WorldGradToken) -> None:
    _assert_world_token_current(world_grad.world)
    _assert_tensor_signatures_current(
        _world_grad_tensors(world_grad),
        world_grad.ledger.grad_tensor_signatures,
        token_name="world gradient token",
    )


def _assert_material_world_grad_current(
    world_grad: FixedWordP0MaterialWorldGradToken,
) -> None:
    _assert_world_token_current(world_grad.world)
    if world_grad.geometry_vjp_executed:
        raise ValueError("material world gradient token cannot claim a geometry VJP")
    _assert_tensor_signatures_current(
        (world_grad.grad_site_rgba_f32,),
        world_grad.ledger.grad_tensor_signatures,
        token_name="material world gradient token",
    )


def _require_mps_tensor(tensor: Tensor, *, name: str, dtype: torch.dtype, cols: int | None) -> None:
    if tensor.device.type != "mps":
        raise ValueError(f"{name} must be on MPS")
    if tensor.dtype != dtype:
        raise ValueError(f"{name} must have dtype {dtype}")
    if cols is None:
        if tensor.ndim != 1:
            raise ValueError(f"{name} must be rank-1")
    elif tensor.ndim != 2 or tensor.shape[1] != cols:
        raise ValueError(f"{name} must have shape [N,{cols}]")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def _candidate_mask_time_slab_count(candidate_mask_u32: Tensor, *, beam_count: int, name: str) -> int:
    if beam_count <= 0:
        raise ValueError("beam_count must be positive")
    if candidate_mask_u32.shape[0] % beam_count != 0:
        raise ValueError(f"{name} length must be beam_count * time_slab_count")
    time_slab_count = candidate_mask_u32.shape[0] // beam_count
    if time_slab_count <= 0:
        raise ValueError(f"{name} must contain at least one time slab")
    return int(time_slab_count)


def _validate_csr_offsets_cpu(
    row_offsets_cpu: Tensor,
    *,
    candidate_count: int,
    max_boundaries: int = MAX_REALRAY_BOUNDARIES,
    name: str = "candidate_row_offsets_i32",
) -> None:
    if row_offsets_cpu.numel() == 0:
        raise ValueError(f"{name} must contain at least one offset")
    if int(row_offsets_cpu[0].item()) != 0:
        raise ValueError(f"{name}[0] must be 0")
    row_lengths = row_offsets_cpu[1:] - row_offsets_cpu[:-1]
    if bool((row_lengths < 0).any().item()):
        raise ValueError(f"{name} must be monotonic nondecreasing")
    if int(row_offsets_cpu[-1].item()) != int(candidate_count):
        raise ValueError(f"{name}[-1] must match candidate count")
    if row_lengths.numel() and int(row_lengths.max().item()) > max_boundaries:
        raise ValueError(
            f"{name} row contains {int(row_lengths.max().item())} candidates, "
            f"exceeding Metal local boundary cap {max_boundaries}"
        )


def _validate_segment_tape_offsets_cpu(
    segment_offsets_cpu: Tensor,
    *,
    sample_count: int,
    segment_count: int,
    max_segments_per_sample: int | None,
) -> None:
    if segment_offsets_cpu.numel() != sample_count + 1:
        raise ValueError("segment_offsets_i32 length must be track_count * frame_count + 1")
    if int(segment_offsets_cpu[0].item()) != 0:
        raise ValueError("segment_offsets_i32[0] must be 0")
    row_lengths = segment_offsets_cpu[1:] - segment_offsets_cpu[:-1]
    if bool((row_lengths < 0).any().item()):
        raise ValueError("segment_offsets_i32 must be monotonic nondecreasing")
    if int(segment_offsets_cpu[-1].item()) != int(segment_count):
        raise ValueError("segment_offsets_i32[-1] must match segment count")
    if max_segments_per_sample is not None and row_lengths.numel():
        max_row = int(row_lengths.max().item())
        if max_row > max_segments_per_sample:
            raise ValueError(
                f"segment tape row contains {max_row} segments, exceeding Metal replay cap {max_segments_per_sample}"
            )


def _validate_packed_endpoint_records_cpu(
    record_i32: Tensor,
    *,
    name: str,
    site_count: int,
    boundary_count: int,
) -> None:
    records = record_i32.detach().cpu().to(dtype=torch.int64)
    if records.numel() == 0:
        return
    if int(records.min().item()) < 0:
        raise ValueError(f"{name} must contain nonnegative packed endpoint records")
    owner_i64 = records & 255
    if int(owner_i64.max().item()) >= int(site_count):
        raise ValueError(f"{name} owner code must be < site_count")
    for field_name, shift in (("left", 8), ("right", 20)):
        cut_code_i64 = (records >> shift) & 4095
        cut_i64 = cut_code_i64 - 2
        if bool(((cut_code_i64 >= 2) & (cut_i64 >= int(boundary_count))).any().item()):
            raise ValueError(f"{name} {field_name} cut id must be < boundary_count")


def _validate_packed_endpoint_delta_records_cpu(
    *,
    base_record_i32: Tensor,
    change_record_i32: Tensor,
    site_count: int,
    boundary_count: int,
) -> None:
    _validate_packed_endpoint_records_cpu(
        base_record_i32,
        name="base_record_i32",
        site_count=site_count,
        boundary_count=boundary_count,
    )
    _validate_packed_endpoint_records_cpu(
        change_record_i32,
        name="change_record_i32",
        site_count=site_count,
        boundary_count=boundary_count,
    )


def _validate_track_boundary_incidence_csr_cpu(
    track_incidence_offsets_i32: Tensor,
    incidence_boundary_i32: Tensor,
    *,
    track_count: int,
    boundary_count: int,
) -> None:
    offsets = track_incidence_offsets_i32.detach().cpu().to(dtype=torch.int64)
    boundary_ids = incidence_boundary_i32.detach().cpu().to(dtype=torch.int64)
    if offsets.ndim != 1 or offsets.numel() != track_count + 1:
        raise ValueError("track_incidence_offsets_i32 must have shape [track_count + 1]")
    if boundary_ids.ndim != 1:
        raise ValueError("incidence_boundary_i32 must be rank-1")
    if int(offsets[0].item()) != 0:
        raise ValueError("track_incidence_offsets_i32[0] must be 0")
    if bool(((offsets[1:] - offsets[:-1]) < 0).any().item()):
        raise ValueError("track_incidence_offsets_i32 must be monotonic nondecreasing")
    if int(offsets[-1].item()) != boundary_ids.numel():
        raise ValueError("track_incidence_offsets_i32[-1] must match incidence count")
    row_lengths = offsets[1:] - offsets[:-1]
    if row_lengths.numel() and int(row_lengths.max().item()) > 4093:
        raise ValueError("packed row-local incidence codes support at most 4093 incidences per track")
    if boundary_ids.numel() and (
        int(boundary_ids.min().item()) < 0 or int(boundary_ids.max().item()) >= boundary_count
    ):
        raise ValueError("incidence_boundary_i32 values must be in [0, boundary_count)")
    for track_id in range(track_count):
        begin = int(offsets[track_id].item())
        end = int(offsets[track_id + 1].item())
        row = boundary_ids[begin:end]
        if row.numel() > 1 and bool((row[1:] <= row[:-1]).any().item()):
            raise ValueError("each incidence CSR row must contain strictly increasing unique boundary ids")


def _validate_compact_referenced_boundaries_cpu(
    incidence_boundary_i32: Tensor,
    *,
    boundary_count: int,
) -> None:
    """Require B to mean referenced faces, never a dense all-pairs table."""
    referenced_boundary_ids = torch.unique(
        incidence_boundary_i32.detach().cpu().to(dtype=torch.int64),
        sorted=True,
    )
    compact_boundary_ids = torch.arange(boundary_count, dtype=torch.int64)
    if not torch.equal(referenced_boundary_ids, compact_boundary_ids):
        raise ValueError("boundary_site_pairs_i32 must be the exact compact referenced boundary table")


def _validate_fixed_word_incidence_csr_cpu(
    *,
    word_offsets_i32: Tensor,
    word_owner_i32: Tensor,
    word_left_incidence_i32: Tensor,
    word_right_incidence_i32: Tensor,
    track_incidence_offsets_i32: Tensor,
    track_count: int,
    site_count: int,
) -> None:
    offsets = word_offsets_i32.detach().cpu().to(dtype=torch.int64)
    owners = word_owner_i32.detach().cpu().to(dtype=torch.int64)
    left_cuts = word_left_incidence_i32.detach().cpu().to(dtype=torch.int64)
    right_cuts = word_right_incidence_i32.detach().cpu().to(dtype=torch.int64)
    incidence_offsets = track_incidence_offsets_i32.detach().cpu().to(dtype=torch.int64)
    if offsets.ndim != 1 or offsets.numel() != track_count + 1:
        raise ValueError("word_offsets_i32 must have shape [track_count + 1]")
    if incidence_offsets.ndim != 1 or incidence_offsets.numel() != track_count + 1:
        raise ValueError("track_incidence_offsets_i32 must have shape [track_count + 1]")
    word_count = owners.numel()
    if any(tensor.ndim != 1 or tensor.numel() != word_count for tensor in (left_cuts, right_cuts)):
        raise ValueError("fixed-word owner and cut arrays must be rank-1 with identical lengths")
    if int(offsets[0].item()) != 0:
        raise ValueError("word_offsets_i32[0] must be 0")
    if int(offsets[-1].item()) != word_count:
        raise ValueError("word_offsets_i32[-1] must match fixed-word record count")
    row_lengths = offsets[1:] - offsets[:-1]
    if bool((row_lengths <= 0).any().item()):
        raise ValueError("each fixed-word CSR track row must be nonempty and monotonic")
    if owners.numel() and (int(owners.min().item()) < 0 or int(owners.max().item()) >= site_count):
        raise ValueError("fixed-word owner ids must be in [0, site_count)")
    for track_id in range(track_count):
        begin = int(offsets[track_id].item())
        end = int(offsets[track_id + 1].item())
        incidence_row_size = int((incidence_offsets[track_id + 1] - incidence_offsets[track_id]).item())
        if int(left_cuts[begin].item()) != -1:
            raise ValueError("each fixed-word CSR row must start at the near cut (-1)")
        if int(right_cuts[end - 1].item()) != -2:
            raise ValueError("each fixed-word CSR row must end at the far cut (-2)")
        previous_right = -1
        for cursor in range(begin, end):
            left = int(left_cuts[cursor].item())
            right = int(right_cuts[cursor].item())
            if left != previous_right:
                raise ValueError("fixed-word cuts must form one adjacent stable ordered word")
            if left != -1 and not 0 <= left < incidence_row_size:
                raise ValueError("fixed-word left cut must be near or a row-local incidence id")
            if right != -2 and not 0 <= right < incidence_row_size:
                raise ValueError("fixed-word right cut must be far or a row-local incidence id")
            if left == right:
                raise ValueError("fixed-word segments must have distinct endpoint cuts")
            if cursor + 1 != end and right < 0:
                raise ValueError("only the final fixed-word segment may use the far cut")
            previous_right = right


def _packed_record_u32(record: Tensor) -> int:
    return int(record.item()) & 0xFFFFFFFF


def _validate_packed_endpoint_incidence_delta_records_cpu(
    *,
    base_offsets_i16: Tensor,
    base_record_incidence_i32: Tensor,
    track_change_offsets_i16: Tensor,
    change_offsets_i16: Tensor,
    change_record_incidence_i32: Tensor,
    track_incidence_offsets_i32: Tensor,
    track_count: int,
    site_count: int,
) -> None:
    base_offsets = base_offsets_i16.detach().cpu().to(dtype=torch.int64)
    track_change_offsets = track_change_offsets_i16.detach().cpu().to(dtype=torch.int64)
    change_offsets = change_offsets_i16.detach().cpu().to(dtype=torch.int64)
    incidence_offsets = track_incidence_offsets_i32.detach().cpu().to(dtype=torch.int64)
    base_records = base_record_incidence_i32.detach().cpu()
    change_records = change_record_incidence_i32.detach().cpu()

    def validate_row(records: Tensor, begin: int, end: int, incidence_count: int, name: str) -> None:
        for record_id in range(begin, end):
            packed = _packed_record_u32(records[record_id])
            if (packed & 255) >= site_count:
                raise ValueError(f"{name} owner code must be < site_count")
            for field_name, shift in (("left", 8), ("right", 20)):
                code = (packed >> shift) & 4095
                if code >= 2 and code - 2 >= incidence_count:
                    raise ValueError(f"{name} {field_name} row-local incidence id is outside its track CSR row")

    for track_id in range(track_count):
        incidence_count = int((incidence_offsets[track_id + 1] - incidence_offsets[track_id]).item())
        validate_row(
            base_records,
            int(base_offsets[track_id].item()),
            int(base_offsets[track_id + 1].item()),
            incidence_count,
            "base_record_incidence_i32",
        )
        for change_id in range(
            int(track_change_offsets[track_id].item()),
            int(track_change_offsets[track_id + 1].item()),
        ):
            validate_row(
                change_records,
                int(change_offsets[change_id].item()),
                int(change_offsets[change_id + 1].item()),
                incidence_count,
                "change_record_incidence_i32",
            )


def recode_packed_endpoint_delta_records_to_track_incidence(
    track_incidence_offsets_i32: Tensor,
    incidence_boundary_i32: Tensor,
    base_offsets_i16: Tensor,
    base_record_i32: Tensor,
    track_change_offsets_i16: Tensor,
    change_offsets_i16: Tensor,
    change_record_i32: Tensor,
    *,
    track_count: int,
    boundary_count: int,
    site_count: int,
) -> tuple[Tensor, Tensor]:
    """Recode global packed cut ids to row-local sparse-incidence ids.

    The returned tapes preserve owner and near/far codes. Internal cuts use
    their zero-based position in the corresponding sorted CSR row, so the
    Metal replay resolves each endpoint with one offset add and one gather.
    This is a topology-build operation, not part of the per-sample hot path.
    """
    if track_count <= 0 or boundary_count <= 0 or site_count <= 0 or site_count > 256:
        raise ValueError("track_count/boundary_count must be positive and site_count must be in [1, 256]")
    for name, tensor, dtype in (
        ("track_incidence_offsets_i32", track_incidence_offsets_i32, torch.int32),
        ("incidence_boundary_i32", incidence_boundary_i32, torch.int32),
        ("base_offsets_i16", base_offsets_i16, torch.int16),
        ("base_record_i32", base_record_i32, torch.int32),
        ("track_change_offsets_i16", track_change_offsets_i16, torch.int16),
        ("change_offsets_i16", change_offsets_i16, torch.int16),
        ("change_record_i32", change_record_i32, torch.int32),
    ):
        if tensor.ndim != 1 or tensor.dtype != dtype:
            raise ValueError(f"{name} must be a rank-1 {dtype} tensor")
    change_count = change_offsets_i16.numel() - 1
    if change_count < 0:
        raise ValueError("change_offsets_i16 must contain at least one offset")
    _validate_segment_tape_offsets_cpu(
        base_offsets_i16.detach().cpu(),
        sample_count=track_count,
        segment_count=base_record_i32.numel(),
        max_segments_per_sample=129,
    )
    _validate_segment_tape_offsets_cpu(
        track_change_offsets_i16.detach().cpu(),
        sample_count=track_count,
        segment_count=change_count,
        max_segments_per_sample=None,
    )
    _validate_segment_tape_offsets_cpu(
        change_offsets_i16.detach().cpu(),
        sample_count=change_count,
        segment_count=change_record_i32.numel(),
        max_segments_per_sample=129,
    )
    _validate_track_boundary_incidence_csr_cpu(
        track_incidence_offsets_i32,
        incidence_boundary_i32,
        track_count=track_count,
        boundary_count=boundary_count,
    )
    base_offsets = base_offsets_i16.detach().cpu().to(dtype=torch.int64)
    track_change_offsets = track_change_offsets_i16.detach().cpu().to(dtype=torch.int64)
    change_offsets = change_offsets_i16.detach().cpu().to(dtype=torch.int64)
    incidence_offsets = track_incidence_offsets_i32.detach().cpu().to(dtype=torch.int64)
    incidence_boundary = incidence_boundary_i32.detach().cpu().to(dtype=torch.int64)
    base_source = base_record_i32.detach().cpu()
    change_source = change_record_i32.detach().cpu()
    base_recoded = torch.empty(base_source.shape, dtype=torch.int32)
    change_recoded = torch.empty(change_source.shape, dtype=torch.int32)

    def recode_row(source: Tensor, output: Tensor, begin: int, end: int, boundary_to_local: dict[int, int]) -> None:
        for record_id in range(begin, end):
            packed = _packed_record_u32(source[record_id])
            owner = packed & 255
            if owner >= site_count:
                raise ValueError("packed endpoint record owner code must be < site_count")
            codes: list[int] = []
            for field_name, shift in (("left", 8), ("right", 20)):
                code = (packed >> shift) & 4095
                if code < 2:
                    codes.append(code)
                    continue
                boundary_id = code - 2
                if boundary_id >= boundary_count:
                    raise ValueError(f"packed endpoint record {field_name} cut id must be < boundary_count")
                if boundary_id not in boundary_to_local:
                    raise ValueError(
                        f"packed endpoint record {field_name} boundary {boundary_id} is absent from its track incidence row"
                    )
                codes.append(boundary_to_local[boundary_id] + 2)
            recoded_u32 = owner | (codes[0] << 8) | (codes[1] << 20)
            output[record_id] = recoded_u32 if recoded_u32 < (1 << 31) else recoded_u32 - (1 << 32)

    for track_id in range(track_count):
        incidence_begin = int(incidence_offsets[track_id].item())
        incidence_end = int(incidence_offsets[track_id + 1].item())
        boundary_to_local = {
            int(boundary_id.item()): local_id
            for local_id, boundary_id in enumerate(incidence_boundary[incidence_begin:incidence_end])
        }
        recode_row(
            base_source,
            base_recoded,
            int(base_offsets[track_id].item()),
            int(base_offsets[track_id + 1].item()),
            boundary_to_local,
        )
        for change_id in range(
            int(track_change_offsets[track_id].item()),
            int(track_change_offsets[track_id + 1].item()),
        ):
            recode_row(
                change_source,
                change_recoded,
                int(change_offsets[change_id].item()),
                int(change_offsets[change_id + 1].item()),
                boundary_to_local,
            )

    return (
        base_recoded.to(device=base_record_i32.device),
        change_recoded.to(device=change_record_i32.device),
    )


def recode_sparse_power_boundary_site_pairs(
    incidence_boundary_i32: Tensor,
    boundary_site_pairs_i32: Tensor,
    *,
    boundary_count: int,
    site_count: int,
) -> Tensor:
    """Build the fixed sparse boundary-to-site table used by the geometry VJP.

    Repeated boundary ids across track-incidence rows are reduced to one sorted
    row ``[boundary_id, left_site, right_site]``. This is a topology-build
    operation; keep the result resident and reuse it across optimizer steps.
    """
    if boundary_count < 0 or site_count <= 0:
        raise ValueError("boundary_count must be nonnegative and site_count must be positive")
    if incidence_boundary_i32.ndim != 1 or incidence_boundary_i32.dtype != torch.int32:
        raise ValueError("incidence_boundary_i32 must be a rank-1 int32 tensor")
    if (
        boundary_site_pairs_i32.ndim != 2
        or boundary_site_pairs_i32.shape[1] != 2
        or boundary_site_pairs_i32.dtype != torch.int32
    ):
        raise ValueError("boundary_site_pairs_i32 must have shape [boundary_count,2] and dtype int32")
    if boundary_site_pairs_i32.shape[0] != boundary_count:
        raise ValueError("boundary_site_pairs_i32 row count must match boundary_count")

    incidence = incidence_boundary_i32.detach().cpu().to(dtype=torch.int64)
    boundary_pairs = boundary_site_pairs_i32.detach().cpu().to(dtype=torch.int64)
    if incidence.numel() == 0:
        return torch.empty((0, 3), dtype=torch.int32, device=incidence_boundary_i32.device)
    if int(incidence.min().item()) < 0 or int(incidence.max().item()) >= boundary_count:
        raise ValueError("incidence_boundary_i32 values must be in [0, boundary_count)")
    boundary_ids = torch.unique(incidence, sorted=True)
    active_pairs = boundary_pairs.index_select(0, boundary_ids)
    if int(active_pairs.min().item()) < 0 or int(active_pairs.max().item()) >= site_count:
        raise ValueError("active boundary site ids must be in [0, site_count)")
    if bool((active_pairs[:, 0] == active_pairs[:, 1]).any().item()):
        raise ValueError("active power boundaries must connect two distinct sites")
    table = torch.cat((boundary_ids[:, None], active_pairs), dim=1).to(dtype=torch.int32)
    return table.to(device=incidence_boundary_i32.device).contiguous()


def _validate_sparse_power_boundary_site_pairs_cpu(
    active_boundary_site_pairs_i32: Tensor,
    *,
    boundary_count: int,
    site_count: int,
) -> None:
    if (
        active_boundary_site_pairs_i32.ndim != 2
        or active_boundary_site_pairs_i32.shape[1] != 3
        or active_boundary_site_pairs_i32.dtype != torch.int32
    ):
        raise ValueError("active_boundary_site_pairs_i32 must have shape [U,3] and dtype int32")
    active = active_boundary_site_pairs_i32.detach().cpu().to(dtype=torch.int64)
    if active.shape[0] == 0:
        return
    boundary_ids = active[:, 0]
    if int(boundary_ids.min().item()) < 0 or int(boundary_ids.max().item()) >= boundary_count:
        raise ValueError("active boundary ids must be in [0, boundary_count)")
    if boundary_ids.numel() > 1 and bool((boundary_ids[1:] <= boundary_ids[:-1]).any().item()):
        raise ValueError("active boundary ids must be strictly increasing and unique")
    site_ids = active[:, 1:]
    if int(site_ids.min().item()) < 0 or int(site_ids.max().item()) >= site_count:
        raise ValueError("active boundary site ids must be in [0, site_count)")
    if bool((site_ids[:, 0] == site_ids[:, 1]).any().item()):
        raise ValueError("active power boundaries must connect two distinct sites")


def sparse_power_boundary_vjp_to_sites_reference(
    active_boundary_site_pairs_i32: Tensor,
    sites: Tensor,
    grad_boundary: Tensor,
) -> Tensor:
    """CPU/reference pullback from 4D power-bisector bars to site bars."""
    if sites.ndim != 2 or sites.shape[1] != 5 or not sites.is_floating_point():
        raise ValueError("sites must be a floating tensor with shape [site_count,5]")
    if grad_boundary.ndim != 2 or grad_boundary.shape[1] != 5 or not grad_boundary.is_floating_point():
        raise ValueError("grad_boundary must be a floating tensor with shape [boundary_count,5]")
    if sites.device != grad_boundary.device or sites.dtype != grad_boundary.dtype:
        raise ValueError("sites and grad_boundary must share device and dtype")
    _validate_sparse_power_boundary_site_pairs_cpu(
        active_boundary_site_pairs_i32,
        boundary_count=grad_boundary.shape[0],
        site_count=sites.shape[0],
    )
    if active_boundary_site_pairs_i32.shape[0] == 0:
        return torch.zeros_like(sites)
    active = active_boundary_site_pairs_i32.to(device=sites.device, dtype=torch.long)
    boundary_ids, left_ids, right_ids = active.unbind(dim=1)
    bars = grad_boundary.index_select(0, boundary_ids)
    grad_normal = bars[:, :4]
    grad_bias = bars[:, 4:5]
    left = sites.index_select(0, left_ids)
    right = sites.index_select(0, right_ids)
    left_grad = torch.cat((-2.0 * grad_normal + 2.0 * grad_bias * left[:, :4], -grad_bias), dim=1)
    right_grad = torch.cat((2.0 * grad_normal - 2.0 * grad_bias * right[:, :4], grad_bias), dim=1)
    grad_sites = torch.zeros_like(sites)
    grad_sites.index_add_(0, left_ids, left_grad)
    grad_sites.index_add_(0, right_ids, right_grad)
    return grad_sites


def sparse_power_boundary_from_sites_reference(
    boundary_site_pairs_i32: Tensor,
    sites: Tensor,
) -> Tensor:
    """Derive compact 4D power faces from the exact sites used by the VJP."""
    if sites.ndim != 2 or sites.shape[1] != 5 or not sites.is_floating_point():
        raise ValueError("sites must be a floating tensor with shape [site_count,5]")
    pairs = torch.as_tensor(boundary_site_pairs_i32, dtype=torch.long, device=sites.device)
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError("boundary_site_pairs_i32 must have shape [boundary_count,2]")
    if pairs.numel() and (int(pairs.min().item()) < 0 or int(pairs.max().item()) >= sites.shape[0]):
        raise ValueError("boundary site ids must be in [0, site_count)")
    left = sites.index_select(0, pairs[:, 0])
    right = sites.index_select(0, pairs[:, 1])
    normal = 2.0 * (right[:, :4] - left[:, :4])
    bias = left[:, :4].square().sum(dim=1) - right[:, :4].square().sum(dim=1) - left[:, 4] + right[:, 4]
    return torch.cat((normal, bias[:, None]), dim=1)


def scaled_rgb_mse_reference(
    prediction_rgb: Tensor,
    target_rgb: Tensor,
    *,
    loss_scale: float,
) -> tuple[Tensor, Tensor]:
    """Reference scalar/adjoint contract for partition-invariant RGB loss.

    ``loss_scale`` is chosen from the global logical training batch and reused
    unchanged for every disjoint local block. Summing block losses and
    scattering block adjoints then exactly recovers the unsplit result.
    """
    if prediction_rgb.shape != target_rgb.shape or prediction_rgb.shape[-1:] != (3,):
        raise ValueError("prediction_rgb and target_rgb must have matching [...,3] shapes")
    if prediction_rgb.device != target_rgb.device or prediction_rgb.dtype != target_rgb.dtype:
        raise ValueError("prediction_rgb and target_rgb must share device and dtype")
    if not math.isfinite(loss_scale) or loss_scale <= 0.0:
        raise ValueError("loss_scale must be finite and positive")
    residual = prediction_rgb - target_rgb
    return loss_scale * residual.square().sum(), (2.0 * loss_scale) * residual


def _validate_csr_values(
    *,
    row_index_i32: Tensor,
    candidate_row_offsets_i32: Tensor,
    candidate_boundary_ids_i32: Tensor,
    row_count: int,
    candidate_count: int,
    boundary_count: int,
) -> None:
    row_index_cpu = row_index_i32.detach().cpu()
    row_offsets_cpu = candidate_row_offsets_i32.detach().cpu()
    candidate_ids_cpu = candidate_boundary_ids_i32.detach().cpu()
    if int(row_index_cpu.min().item()) < 0 or int(row_index_cpu.max().item()) >= row_count:
        raise ValueError("row_index_i32 values must be in [0, row_count)")
    _validate_csr_offsets_cpu(row_offsets_cpu, candidate_count=candidate_count)
    if candidate_count and (
        int(candidate_ids_cpu.min().item()) < 0 or int(candidate_ids_cpu.max().item()) >= boundary_count
    ):
        raise ValueError("candidate_boundary_ids_i32 values must be in [0, boundary_count)")


def count_power_boundary_events(
    boundary_f32: Tensor,
    beam_f32: Tensor,
    config: PowerBoundaryConfig,
    *,
    boundary_u32: Tensor | None = None,
    beam_u32: Tensor | None = None,
) -> Tensor:
    """Count Gate 0 power-boundary events with the local MPS Metal kernel.

    Layouts:
    - `boundary_f32`: `[B,4]` rows `(nx, nz, nt, b)`.
    - `beam_f32`: `[M,5]` rows `(u_center, t0, t1, near_depth, far_depth)`.
    - `boundary_u32`: optional `[B,4]` rows `(left_site, right_site, 0, 0)`.
    - `beam_u32`: optional `[M,4]` rows `(payload_id, flags, 0, 0)`.

    Returns an int32 `[M,8]` tensor matching `WF2PowerBoundaryCount` field
    order: `(beam_id, payload_id, boundary_event_count,
    invalid_denominator_count, flags, 0, 0, 0)`.
    """
    boundary_f32 = boundary_f32.contiguous()
    beam_f32 = beam_f32.contiguous()
    _require_mps_tensor(boundary_f32, name="boundary_f32", dtype=torch.float32, cols=4)
    _require_mps_tensor(beam_f32, name="beam_f32", dtype=torch.float32, cols=5)

    if boundary_u32 is None:
        boundary_u32 = torch.zeros((boundary_f32.shape[0], 4), device=boundary_f32.device, dtype=torch.int32)
    if beam_u32 is None:
        beam_u32 = torch.zeros((beam_f32.shape[0], 4), device=beam_f32.device, dtype=torch.int32)
        beam_u32[:, 0] = torch.arange(beam_f32.shape[0], device=beam_f32.device, dtype=torch.int32)

    boundary_u32 = boundary_u32.contiguous()
    beam_u32 = beam_u32.contiguous()
    _require_mps_tensor(boundary_u32, name="boundary_u32", dtype=torch.int32, cols=4)
    _require_mps_tensor(beam_u32, name="beam_u32", dtype=torch.int32, cols=4)
    if boundary_u32.shape[0] != boundary_f32.shape[0]:
        raise ValueError("boundary_u32 row count must match boundary_f32")
    if beam_u32.shape[0] != beam_f32.shape[0]:
        raise ValueError("beam_u32 row count must match beam_f32")

    ops = torch.ops.world_foam_lane2_fused_slab_v0
    if not hasattr(ops, "count_power_boundary_events"):
        raise RuntimeError("world_foam_lane2_fused_slab_v0 custom ops not found. Build this variant first.")
    config_i32 = torch.tensor(
        [boundary_f32.shape[0], beam_f32.shape[0]],
        device=boundary_f32.device,
        dtype=torch.int32,
    )
    config_f32 = torch.tensor(
        [config.camera_velocity_x, config.invalid_epsilon],
        device=boundary_f32.device,
        dtype=torch.float32,
    )
    return ops.count_power_boundary_events(
        boundary_f32,
        boundary_u32,
        beam_f32,
        beam_u32,
        config_i32,
        config_f32,
    )


def shared_signal_replay(
    boundary_f32: Tensor,
    candidate_mask_u32: Tensor,
    sites_f32: Tensor,
    site_signal_f32: Tensor,
    beam_f32: Tensor,
    frame_t_f32: Tensor,
    grad_output_f32: Tensor,
    config: PowerBoundaryConfig,
) -> tuple[Tensor, Tensor]:
    """Replay shared Gate 0 slab candidates and emit signal-gradient samples.

    `candidate_mask_u32` is an int32 bit mask per beam/time slab in row-major
    `[beam, slab]` order. A legacy length-`M` mask is treated as one slab.
    Bits `0..30` mark boundary rows that are shared candidates for that slab.
    The kernel emits:

    - output: `[M,T]` scalar segment-integral signal values.
    - grad_samples: `[M,T,N]` per-ray per-site signal-gradient samples.

    This is a fixed-segment site-signal VJP reference. It does not implement
    site-position, weight, topology, or sorting gradients.
    """
    boundary_f32 = boundary_f32.contiguous()
    candidate_mask_u32 = candidate_mask_u32.contiguous()
    sites_f32 = sites_f32.contiguous()
    site_signal_f32 = site_signal_f32.contiguous()
    beam_f32 = beam_f32.contiguous()
    frame_t_f32 = frame_t_f32.contiguous()
    grad_output_f32 = grad_output_f32.contiguous()

    _require_mps_tensor(boundary_f32, name="boundary_f32", dtype=torch.float32, cols=4)
    _require_mps_tensor(candidate_mask_u32, name="candidate_mask_u32", dtype=torch.int32, cols=None)
    _require_mps_tensor(sites_f32, name="sites_f32", dtype=torch.float32, cols=4)
    _require_mps_tensor(site_signal_f32, name="site_signal_f32", dtype=torch.float32, cols=None)
    _require_mps_tensor(beam_f32, name="beam_f32", dtype=torch.float32, cols=5)
    _require_mps_tensor(frame_t_f32, name="frame_t_f32", dtype=torch.float32, cols=None)
    if grad_output_f32.device.type != "mps":
        raise ValueError("grad_output_f32 must be on MPS")
    if grad_output_f32.dtype != torch.float32:
        raise ValueError("grad_output_f32 must be float32")
    if grad_output_f32.ndim != 2:
        raise ValueError("grad_output_f32 must have shape [M,T]")
    if not grad_output_f32.is_contiguous():
        raise ValueError("grad_output_f32 must be contiguous")
    if boundary_f32.shape[0] > 31:
        raise ValueError("shared_signal_replay currently supports at most 31 boundaries")
    if sites_f32.shape[0] > 32:
        raise ValueError("shared_signal_replay currently supports at most 32 sites")
    time_slab_count = _candidate_mask_time_slab_count(
        candidate_mask_u32,
        beam_count=beam_f32.shape[0],
        name="candidate_mask_u32",
    )
    if site_signal_f32.shape[0] != sites_f32.shape[0]:
        raise ValueError("site_signal_f32 length must match sites_f32 rows")
    if grad_output_f32.shape != (beam_f32.shape[0], frame_t_f32.shape[0]):
        raise ValueError("grad_output_f32 shape must be [beam_count, frame_count]")

    ops = torch.ops.world_foam_lane2_fused_slab_v0
    if not hasattr(ops, "shared_signal_replay"):
        raise RuntimeError(
            "world_foam_lane2_fused_slab_v0 shared_signal_replay op not found. Build this variant first."
        )
    config_i32 = torch.tensor(
        [
            boundary_f32.shape[0],
            beam_f32.shape[0],
            sites_f32.shape[0],
            frame_t_f32.shape[0],
            time_slab_count,
        ],
        device=boundary_f32.device,
        dtype=torch.int32,
    )
    config_f32 = torch.tensor([config.camera_velocity_x], device=boundary_f32.device, dtype=torch.float32)
    return ops.shared_signal_replay(
        boundary_f32,
        candidate_mask_u32,
        sites_f32,
        site_signal_f32,
        beam_f32,
        frame_t_f32,
        grad_output_f32,
        config_i32,
        config_f32,
    )


def shared_rgb_replay(
    boundary_f32: Tensor,
    candidate_mask_u32: Tensor,
    sites_f32: Tensor,
    site_rgb_f32: Tensor,
    beam_f32: Tensor,
    frame_t_f32: Tensor,
    grad_output_rgb_f32: Tensor,
    config: PowerBoundaryConfig,
) -> tuple[Tensor, Tensor]:
    """Replay shared Gate 0 slab candidates and emit RGB-gradient samples.

    Returns:
    - output_rgb: `[M,T,3]` RGB segment-integral values.
    - grad_samples_rgb: `[M,T,N,3]` per-ray per-site RGB signal-gradient
      samples.

    `candidate_mask_u32` is row-major `[beam, slab]`; a length-`M` mask is one
    slab. This is still a fixed-segment signal VJP reference. It does not implement
    alpha/depth compositing, site-position, weight, topology, or sorting
    gradients.
    """
    boundary_f32 = boundary_f32.contiguous()
    candidate_mask_u32 = candidate_mask_u32.contiguous()
    sites_f32 = sites_f32.contiguous()
    site_rgb_f32 = site_rgb_f32.contiguous()
    beam_f32 = beam_f32.contiguous()
    frame_t_f32 = frame_t_f32.contiguous()
    grad_output_rgb_f32 = grad_output_rgb_f32.contiguous()

    _require_mps_tensor(boundary_f32, name="boundary_f32", dtype=torch.float32, cols=4)
    _require_mps_tensor(candidate_mask_u32, name="candidate_mask_u32", dtype=torch.int32, cols=None)
    _require_mps_tensor(sites_f32, name="sites_f32", dtype=torch.float32, cols=4)
    _require_mps_tensor(site_rgb_f32, name="site_rgb_f32", dtype=torch.float32, cols=3)
    _require_mps_tensor(beam_f32, name="beam_f32", dtype=torch.float32, cols=5)
    _require_mps_tensor(frame_t_f32, name="frame_t_f32", dtype=torch.float32, cols=None)
    if grad_output_rgb_f32.device.type != "mps":
        raise ValueError("grad_output_rgb_f32 must be on MPS")
    if grad_output_rgb_f32.dtype != torch.float32:
        raise ValueError("grad_output_rgb_f32 must be float32")
    if grad_output_rgb_f32.ndim != 3 or grad_output_rgb_f32.shape[2] != 3:
        raise ValueError("grad_output_rgb_f32 must have shape [M,T,3]")
    if not grad_output_rgb_f32.is_contiguous():
        raise ValueError("grad_output_rgb_f32 must be contiguous")
    if boundary_f32.shape[0] > 31:
        raise ValueError("shared_rgb_replay currently supports at most 31 boundaries")
    if sites_f32.shape[0] > 32:
        raise ValueError("shared_rgb_replay currently supports at most 32 sites")
    time_slab_count = _candidate_mask_time_slab_count(
        candidate_mask_u32,
        beam_count=beam_f32.shape[0],
        name="candidate_mask_u32",
    )
    if site_rgb_f32.shape[0] != sites_f32.shape[0]:
        raise ValueError("site_rgb_f32 row count must match sites_f32 rows")
    if grad_output_rgb_f32.shape[:2] != (beam_f32.shape[0], frame_t_f32.shape[0]):
        raise ValueError("grad_output_rgb_f32 shape must be [beam_count, frame_count, 3]")

    ops = torch.ops.world_foam_lane2_fused_slab_v0
    if not hasattr(ops, "shared_rgb_replay"):
        raise RuntimeError("world_foam_lane2_fused_slab_v0 shared_rgb_replay op not found. Build this variant first.")
    config_i32 = torch.tensor(
        [
            boundary_f32.shape[0],
            beam_f32.shape[0],
            sites_f32.shape[0],
            frame_t_f32.shape[0],
            time_slab_count,
        ],
        device=boundary_f32.device,
        dtype=torch.int32,
    )
    config_f32 = torch.tensor([config.camera_velocity_x], device=boundary_f32.device, dtype=torch.float32)
    return ops.shared_rgb_replay(
        boundary_f32,
        candidate_mask_u32,
        sites_f32,
        site_rgb_f32,
        beam_f32,
        frame_t_f32,
        grad_output_rgb_f32,
        config_i32,
        config_f32,
    )


def shared_rgba_depth_replay(
    boundary_f32: Tensor,
    candidate_mask_u32: Tensor,
    sites_f32: Tensor,
    site_rgba_f32: Tensor,
    beam_f32: Tensor,
    frame_t_f32: Tensor,
    config: PowerBoundaryConfig,
) -> tuple[Tensor, Tensor, Tensor]:
    """Replay shared candidates and composite RGB, alpha, and expected depth.

    Returns:
    - output_rgb: `[M,T,3]` front-to-back RGB.
    - output_alpha: `[M,T]` accumulated alpha.
    - output_depth: `[M,T]` alpha-weighted expected depth, or far depth when
      alpha is zero.

    `candidate_mask_u32` is row-major `[beam, slab]`; a length-`M` mask is one
    slab. This is a forward-only compositor proof. It does not implement
    gradients, site-position updates, topology changes, or a trainer path.
    """
    boundary_f32 = boundary_f32.contiguous()
    candidate_mask_u32 = candidate_mask_u32.contiguous()
    sites_f32 = sites_f32.contiguous()
    site_rgba_f32 = site_rgba_f32.contiguous()
    beam_f32 = beam_f32.contiguous()
    frame_t_f32 = frame_t_f32.contiguous()

    _require_mps_tensor(boundary_f32, name="boundary_f32", dtype=torch.float32, cols=4)
    _require_mps_tensor(candidate_mask_u32, name="candidate_mask_u32", dtype=torch.int32, cols=None)
    _require_mps_tensor(sites_f32, name="sites_f32", dtype=torch.float32, cols=4)
    _require_mps_tensor(site_rgba_f32, name="site_rgba_f32", dtype=torch.float32, cols=4)
    _require_mps_tensor(beam_f32, name="beam_f32", dtype=torch.float32, cols=5)
    _require_mps_tensor(frame_t_f32, name="frame_t_f32", dtype=torch.float32, cols=None)
    if boundary_f32.shape[0] > 31:
        raise ValueError("shared_rgba_depth_replay currently supports at most 31 boundaries")
    if sites_f32.shape[0] > 32:
        raise ValueError("shared_rgba_depth_replay currently supports at most 32 sites")
    time_slab_count = _candidate_mask_time_slab_count(
        candidate_mask_u32,
        beam_count=beam_f32.shape[0],
        name="candidate_mask_u32",
    )
    if site_rgba_f32.shape[0] != sites_f32.shape[0]:
        raise ValueError("site_rgba_f32 row count must match sites_f32 rows")

    ops = torch.ops.world_foam_lane2_fused_slab_v0
    if not hasattr(ops, "shared_rgba_depth_replay"):
        raise RuntimeError(
            "world_foam_lane2_fused_slab_v0 shared_rgba_depth_replay op not found. Build this variant first."
        )
    config_i32 = torch.tensor(
        [
            boundary_f32.shape[0],
            beam_f32.shape[0],
            sites_f32.shape[0],
            frame_t_f32.shape[0],
            time_slab_count,
        ],
        device=boundary_f32.device,
        dtype=torch.int32,
    )
    config_f32 = torch.tensor([config.camera_velocity_x], device=boundary_f32.device, dtype=torch.float32)
    return ops.shared_rgba_depth_replay(
        boundary_f32,
        candidate_mask_u32,
        sites_f32,
        site_rgba_f32,
        beam_f32,
        frame_t_f32,
        config_i32,
        config_f32,
    )


def shared_rgba_depth_vjp(
    boundary_f32: Tensor,
    candidate_mask_u32: Tensor,
    sites_f32: Tensor,
    site_rgba_f32: Tensor,
    beam_f32: Tensor,
    frame_t_f32: Tensor,
    grad_rgb_f32: Tensor,
    grad_alpha_f32: Tensor,
    grad_depth_f32: Tensor,
    config: PowerBoundaryConfig,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Replay shared candidates and emit fixed-segment RGBA VJP samples.

    Returns `output_rgb [M,T,3]`, `output_alpha [M,T]`, `output_depth [M,T]`,
    and `grad_samples_rgba [M,T,N,4]`. `candidate_mask_u32` is row-major
    `[beam, slab]`; a length-`M` mask is one slab. The final channel is the
    density gradient. Boundary cuts, owners, site positions, and topology are
    fixed.
    """
    boundary_f32 = boundary_f32.contiguous()
    candidate_mask_u32 = candidate_mask_u32.contiguous()
    sites_f32 = sites_f32.contiguous()
    site_rgba_f32 = site_rgba_f32.contiguous()
    beam_f32 = beam_f32.contiguous()
    frame_t_f32 = frame_t_f32.contiguous()
    grad_rgb_f32 = grad_rgb_f32.contiguous()
    grad_alpha_f32 = grad_alpha_f32.contiguous()
    grad_depth_f32 = grad_depth_f32.contiguous()

    _require_mps_tensor(boundary_f32, name="boundary_f32", dtype=torch.float32, cols=4)
    _require_mps_tensor(candidate_mask_u32, name="candidate_mask_u32", dtype=torch.int32, cols=None)
    _require_mps_tensor(sites_f32, name="sites_f32", dtype=torch.float32, cols=4)
    _require_mps_tensor(site_rgba_f32, name="site_rgba_f32", dtype=torch.float32, cols=4)
    _require_mps_tensor(beam_f32, name="beam_f32", dtype=torch.float32, cols=5)
    _require_mps_tensor(frame_t_f32, name="frame_t_f32", dtype=torch.float32, cols=None)
    if grad_rgb_f32.device.type != "mps":
        raise ValueError("grad_rgb_f32 must be on MPS")
    if grad_rgb_f32.dtype != torch.float32:
        raise ValueError("grad_rgb_f32 must be float32")
    if grad_rgb_f32.ndim != 3 or grad_rgb_f32.shape[2] != 3:
        raise ValueError("grad_rgb_f32 must have shape [M,T,3]")
    if not grad_rgb_f32.is_contiguous():
        raise ValueError("grad_rgb_f32 must be contiguous")
    if grad_alpha_f32.device.type != "mps":
        raise ValueError("grad_alpha_f32 must be on MPS")
    if grad_alpha_f32.dtype != torch.float32:
        raise ValueError("grad_alpha_f32 must be float32")
    if grad_alpha_f32.ndim != 2:
        raise ValueError("grad_alpha_f32 must have shape [M,T]")
    if not grad_alpha_f32.is_contiguous():
        raise ValueError("grad_alpha_f32 must be contiguous")
    if grad_depth_f32.device.type != "mps":
        raise ValueError("grad_depth_f32 must be on MPS")
    if grad_depth_f32.dtype != torch.float32:
        raise ValueError("grad_depth_f32 must be float32")
    if grad_depth_f32.ndim != 2:
        raise ValueError("grad_depth_f32 must have shape [M,T]")
    if not grad_depth_f32.is_contiguous():
        raise ValueError("grad_depth_f32 must be contiguous")
    if boundary_f32.shape[0] > 31:
        raise ValueError("shared_rgba_depth_vjp currently supports at most 31 boundaries")
    if sites_f32.shape[0] > 32:
        raise ValueError("shared_rgba_depth_vjp currently supports at most 32 sites")
    time_slab_count = _candidate_mask_time_slab_count(
        candidate_mask_u32,
        beam_count=beam_f32.shape[0],
        name="candidate_mask_u32",
    )
    if site_rgba_f32.shape[0] != sites_f32.shape[0]:
        raise ValueError("site_rgba_f32 row count must match sites_f32 rows")
    expected_shape = (beam_f32.shape[0], frame_t_f32.shape[0])
    if grad_rgb_f32.shape[:2] != expected_shape:
        raise ValueError("grad_rgb_f32 shape must be [beam_count, frame_count, 3]")
    if grad_alpha_f32.shape != expected_shape:
        raise ValueError("grad_alpha_f32 shape must be [beam_count, frame_count]")
    if grad_depth_f32.shape != expected_shape:
        raise ValueError("grad_depth_f32 shape must be [beam_count, frame_count]")

    ops = torch.ops.world_foam_lane2_fused_slab_v0
    if not hasattr(ops, "shared_rgba_depth_vjp"):
        raise RuntimeError(
            "world_foam_lane2_fused_slab_v0 shared_rgba_depth_vjp op not found. Build this variant first."
        )
    config_i32 = torch.tensor(
        [
            boundary_f32.shape[0],
            beam_f32.shape[0],
            sites_f32.shape[0],
            frame_t_f32.shape[0],
            time_slab_count,
        ],
        device=boundary_f32.device,
        dtype=torch.int32,
    )
    config_f32 = torch.tensor([config.camera_velocity_x], device=boundary_f32.device, dtype=torch.float32)
    return ops.shared_rgba_depth_vjp(
        boundary_f32,
        candidate_mask_u32,
        sites_f32,
        site_rgba_f32,
        beam_f32,
        frame_t_f32,
        grad_rgb_f32,
        grad_alpha_f32,
        grad_depth_f32,
        config_i32,
        config_f32,
    )


def realray_rgba_depth_replay(
    boundary_f32: Tensor,
    sites_f32: Tensor,
    site_rgba_f32: Tensor,
    rays_f32: Tensor,
    frame_t_f32: Tensor,
    config: RealRayReplayConfig,
) -> tuple[Tensor, Tensor, Tensor]:
    """Render true camera rays through 4D power-cell boundaries.

    Layouts:
    - `boundary_f32`: `[B,5]` rows `(nx, ny, nz, nt, b)`.
    - `sites_f32`: `[S,5]` rows `(x, y, z, t, weight)`.
    - `site_rgba_f32`: `[S,4]` rows `(r, g, b, density)`.
    - `rays_f32`: `[R,6]` rows `(ox, oy, oz, dx, dy, dz)`.
    - `frame_t_f32`: `[R]` normalized time for each ray.

    Returns `output_rgb [R,3]`, `output_alpha [R]`, and `output_depth [R]`.
    This is a forward-only linear per-ray baseline. It does not share work over
    time and does not implement backward gradients.
    """
    boundary_f32 = boundary_f32.contiguous()
    sites_f32 = sites_f32.contiguous()
    site_rgba_f32 = site_rgba_f32.contiguous()
    rays_f32 = rays_f32.contiguous()
    frame_t_f32 = frame_t_f32.contiguous()
    _require_mps_tensor(boundary_f32, name="boundary_f32", dtype=torch.float32, cols=5)
    _require_mps_tensor(sites_f32, name="sites_f32", dtype=torch.float32, cols=5)
    _require_mps_tensor(site_rgba_f32, name="site_rgba_f32", dtype=torch.float32, cols=4)
    _require_mps_tensor(rays_f32, name="rays_f32", dtype=torch.float32, cols=6)
    _require_mps_tensor(frame_t_f32, name="frame_t_f32", dtype=torch.float32, cols=None)
    if boundary_f32.shape[0] > 128:
        raise ValueError("realray_rgba_depth_replay currently supports at most 128 boundaries")
    if sites_f32.shape[0] > 64:
        raise ValueError("realray_rgba_depth_replay currently supports at most 64 sites")
    if site_rgba_f32.shape[0] != sites_f32.shape[0]:
        raise ValueError("site_rgba_f32 row count must match sites_f32 rows")
    if frame_t_f32.shape[0] != rays_f32.shape[0]:
        raise ValueError("frame_t_f32 length must match rays_f32 rows")
    if config.far <= config.near:
        raise ValueError("config.far must be greater than config.near")
    if config.invalid_epsilon <= 0.0:
        raise ValueError("config.invalid_epsilon must be positive")
    if config.transmittance_threshold < 0.0:
        raise ValueError("config.transmittance_threshold must be nonnegative")

    ops = torch.ops.world_foam_lane2_fused_slab_v0
    if not hasattr(ops, "realray_rgba_depth_replay"):
        raise RuntimeError(
            "world_foam_lane2_fused_slab_v0 realray_rgba_depth_replay op not found. Build this variant first."
        )
    config_i32 = torch.tensor(
        [boundary_f32.shape[0], rays_f32.shape[0], sites_f32.shape[0]],
        device=boundary_f32.device,
        dtype=torch.int32,
    )
    config_f32 = torch.tensor(
        [config.near, config.far, config.invalid_epsilon, config.transmittance_threshold],
        device=boundary_f32.device,
        dtype=torch.float32,
    )
    return ops.realray_rgba_depth_replay(
        boundary_f32,
        sites_f32,
        site_rgba_f32,
        rays_f32,
        frame_t_f32,
        config_i32,
        config_f32,
    )


def shared_realray_rgba_depth_replay(
    boundary_f32: Tensor,
    candidate_mask_i32: Tensor,
    sites_f32: Tensor,
    site_rgba_f32: Tensor,
    track_rays_f32: Tensor,
    frame_t_f32: Tensor,
    config: RealRayReplayConfig,
) -> tuple[Tensor, Tensor, Tensor]:
    """Render true camera-ray tracks using shared time-slab candidate masks.

    Layouts:
    - `boundary_f32`: `[B,5]` rows `(nx, ny, nz, nt, b)`.
    - `candidate_mask_i32`: `[R_tracks * time_slabs, W]` bitset words.
      Word `boundary_id // 32`, bit `boundary_id % 32` marks a shared
      boundary candidate for that pixel track and time slab.
    - `sites_f32`: `[S,5]` rows `(x, y, z, t, weight)`.
    - `site_rgba_f32`: `[S,4]` rows `(r, g, b, density)`.
    - `track_rays_f32`: `[R_tracks,6]` rows `(ox, oy, oz, dx, dy, dz)`.
    - `frame_t_f32`: `[T]` normalized frame times.

    Returns `output_rgb [R_tracks,T,3]`, `output_alpha [R_tracks,T]`, and
    `output_depth [R_tracks,T]`. This is forward-only fixed-candidate replay.
    It does not implement backward gradients, topology changes, or training.
    """
    boundary_f32 = boundary_f32.contiguous()
    candidate_mask_i32 = candidate_mask_i32.contiguous()
    sites_f32 = sites_f32.contiguous()
    site_rgba_f32 = site_rgba_f32.contiguous()
    track_rays_f32 = track_rays_f32.contiguous()
    frame_t_f32 = frame_t_f32.contiguous()
    _require_mps_tensor(boundary_f32, name="boundary_f32", dtype=torch.float32, cols=5)
    if candidate_mask_i32.device.type != "mps":
        raise ValueError("candidate_mask_i32 must be on MPS")
    if candidate_mask_i32.dtype != torch.int32:
        raise ValueError("candidate_mask_i32 must have dtype torch.int32")
    if candidate_mask_i32.ndim != 2:
        raise ValueError("candidate_mask_i32 must have shape [track_count * time_slab_count, mask_word_count]")
    if not candidate_mask_i32.is_contiguous():
        raise ValueError("candidate_mask_i32 must be contiguous")
    _require_mps_tensor(sites_f32, name="sites_f32", dtype=torch.float32, cols=5)
    _require_mps_tensor(site_rgba_f32, name="site_rgba_f32", dtype=torch.float32, cols=4)
    _require_mps_tensor(track_rays_f32, name="track_rays_f32", dtype=torch.float32, cols=6)
    _require_mps_tensor(frame_t_f32, name="frame_t_f32", dtype=torch.float32, cols=None)
    if boundary_f32.shape[0] > 128:
        raise ValueError("shared_realray_rgba_depth_replay currently supports at most 128 boundaries")
    if sites_f32.shape[0] > 64:
        raise ValueError("shared_realray_rgba_depth_replay currently supports at most 64 sites")
    if site_rgba_f32.shape[0] != sites_f32.shape[0]:
        raise ValueError("site_rgba_f32 row count must match sites_f32 rows")
    if frame_t_f32.shape[0] <= 0:
        raise ValueError("frame_t_f32 must contain at least one frame")
    if candidate_mask_i32.shape[0] % track_rays_f32.shape[0] != 0:
        raise ValueError("candidate_mask_i32 rows must be track_count * time_slab_count")
    time_slab_count = candidate_mask_i32.shape[0] // track_rays_f32.shape[0]
    if time_slab_count <= 0:
        raise ValueError("candidate_mask_i32 must contain at least one time slab")
    expected_words = (boundary_f32.shape[0] + 31) // 32
    if candidate_mask_i32.shape[1] != expected_words:
        raise ValueError(f"candidate_mask_i32 word count must be ceil(boundary_count / 32) = {expected_words}")
    if config.far <= config.near:
        raise ValueError("config.far must be greater than config.near")
    if config.invalid_epsilon <= 0.0:
        raise ValueError("config.invalid_epsilon must be positive")
    if config.transmittance_threshold < 0.0:
        raise ValueError("config.transmittance_threshold must be nonnegative")

    ops = torch.ops.world_foam_lane2_fused_slab_v0
    if not hasattr(ops, "shared_realray_rgba_depth_replay"):
        raise RuntimeError(
            "world_foam_lane2_fused_slab_v0 shared_realray_rgba_depth_replay op not found. Build this variant first."
        )
    config_i32 = torch.tensor(
        [
            boundary_f32.shape[0],
            track_rays_f32.shape[0],
            sites_f32.shape[0],
            frame_t_f32.shape[0],
            time_slab_count,
            candidate_mask_i32.shape[1],
        ],
        device=boundary_f32.device,
        dtype=torch.int32,
    )
    config_f32 = torch.tensor(
        [config.near, config.far, config.invalid_epsilon, config.transmittance_threshold],
        device=boundary_f32.device,
        dtype=torch.float32,
    )
    return ops.shared_realray_rgba_depth_replay(
        boundary_f32,
        candidate_mask_i32,
        sites_f32,
        site_rgba_f32,
        track_rays_f32,
        frame_t_f32,
        config_i32,
        config_f32,
    )


def shared_realray_rgba_depth_vjp(
    boundary_f32: Tensor,
    candidate_mask_i32: Tensor,
    sites_f32: Tensor,
    site_rgba_f32: Tensor,
    track_rays_f32: Tensor,
    frame_t_f32: Tensor,
    grad_rgb_f32: Tensor,
    grad_alpha_f32: Tensor,
    grad_depth_f32: Tensor,
    config: RealRayReplayConfig,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Render shared true camera-ray tracks and emit fixed-segment RGBA VJP samples.

    Returns `output_rgb [K,T,3]`, `output_alpha [K,T]`, `output_depth [K,T]`,
    and `grad_samples_rgba [K,T,S,4]`. The final gradient channel is density.
    Candidate cuts, segment owners, site positions, weights, and topology are
    fixed; this is not an autograd trainer boundary.
    """
    boundary_f32 = boundary_f32.contiguous()
    candidate_mask_i32 = candidate_mask_i32.contiguous()
    sites_f32 = sites_f32.contiguous()
    site_rgba_f32 = site_rgba_f32.contiguous()
    track_rays_f32 = track_rays_f32.contiguous()
    frame_t_f32 = frame_t_f32.contiguous()
    grad_rgb_f32 = grad_rgb_f32.contiguous()
    grad_alpha_f32 = grad_alpha_f32.contiguous()
    grad_depth_f32 = grad_depth_f32.contiguous()
    _require_mps_tensor(boundary_f32, name="boundary_f32", dtype=torch.float32, cols=5)
    if candidate_mask_i32.device.type != "mps":
        raise ValueError("candidate_mask_i32 must be on MPS")
    if candidate_mask_i32.dtype != torch.int32:
        raise ValueError("candidate_mask_i32 must have dtype torch.int32")
    if candidate_mask_i32.ndim != 2:
        raise ValueError("candidate_mask_i32 must have shape [track_count * time_slab_count, mask_word_count]")
    if not candidate_mask_i32.is_contiguous():
        raise ValueError("candidate_mask_i32 must be contiguous")
    _require_mps_tensor(sites_f32, name="sites_f32", dtype=torch.float32, cols=5)
    _require_mps_tensor(site_rgba_f32, name="site_rgba_f32", dtype=torch.float32, cols=4)
    _require_mps_tensor(track_rays_f32, name="track_rays_f32", dtype=torch.float32, cols=6)
    _require_mps_tensor(frame_t_f32, name="frame_t_f32", dtype=torch.float32, cols=None)
    if grad_rgb_f32.device.type != "mps":
        raise ValueError("grad_rgb_f32 must be on MPS")
    if grad_rgb_f32.dtype != torch.float32:
        raise ValueError("grad_rgb_f32 must be float32")
    if grad_rgb_f32.ndim != 3 or grad_rgb_f32.shape[2] != 3:
        raise ValueError("grad_rgb_f32 must have shape [track_count, frame_count, 3]")
    if not grad_rgb_f32.is_contiguous():
        raise ValueError("grad_rgb_f32 must be contiguous")
    if grad_alpha_f32.device.type != "mps":
        raise ValueError("grad_alpha_f32 must be on MPS")
    if grad_alpha_f32.dtype != torch.float32:
        raise ValueError("grad_alpha_f32 must be float32")
    if grad_alpha_f32.ndim != 2:
        raise ValueError("grad_alpha_f32 must have shape [track_count, frame_count]")
    if not grad_alpha_f32.is_contiguous():
        raise ValueError("grad_alpha_f32 must be contiguous")
    if grad_depth_f32.device.type != "mps":
        raise ValueError("grad_depth_f32 must be on MPS")
    if grad_depth_f32.dtype != torch.float32:
        raise ValueError("grad_depth_f32 must be float32")
    if grad_depth_f32.ndim != 2:
        raise ValueError("grad_depth_f32 must have shape [track_count, frame_count]")
    if not grad_depth_f32.is_contiguous():
        raise ValueError("grad_depth_f32 must be contiguous")
    if boundary_f32.shape[0] > 128:
        raise ValueError("shared_realray_rgba_depth_vjp currently supports at most 128 boundaries")
    if sites_f32.shape[0] > 64:
        raise ValueError("shared_realray_rgba_depth_vjp currently supports at most 64 sites")
    if site_rgba_f32.shape[0] != sites_f32.shape[0]:
        raise ValueError("site_rgba_f32 row count must match sites_f32 rows")
    if candidate_mask_i32.shape[0] % track_rays_f32.shape[0] != 0:
        raise ValueError("candidate_mask_i32 rows must be track_count * time_slab_count")
    time_slab_count = candidate_mask_i32.shape[0] // track_rays_f32.shape[0]
    expected_words = (boundary_f32.shape[0] + 31) // 32
    if candidate_mask_i32.shape[1] != expected_words:
        raise ValueError(f"candidate_mask_i32 word count must be ceil(boundary_count / 32) = {expected_words}")
    expected_shape = (track_rays_f32.shape[0], frame_t_f32.shape[0])
    if grad_rgb_f32.shape[:2] != expected_shape:
        raise ValueError("grad_rgb_f32 shape must be [track_count, frame_count, 3]")
    if grad_alpha_f32.shape != expected_shape:
        raise ValueError("grad_alpha_f32 shape must be [track_count, frame_count]")
    if grad_depth_f32.shape != expected_shape:
        raise ValueError("grad_depth_f32 shape must be [track_count, frame_count]")
    if config.far <= config.near:
        raise ValueError("config.far must be greater than config.near")
    if config.invalid_epsilon <= 0.0:
        raise ValueError("config.invalid_epsilon must be positive")
    if config.transmittance_threshold < 0.0:
        raise ValueError("config.transmittance_threshold must be nonnegative")

    ops = torch.ops.world_foam_lane2_fused_slab_v0
    if not hasattr(ops, "shared_realray_rgba_depth_vjp"):
        raise RuntimeError(
            "world_foam_lane2_fused_slab_v0 shared_realray_rgba_depth_vjp op not found. Build this variant first."
        )
    config_i32 = torch.tensor(
        [
            boundary_f32.shape[0],
            track_rays_f32.shape[0],
            sites_f32.shape[0],
            frame_t_f32.shape[0],
            time_slab_count,
            candidate_mask_i32.shape[1],
        ],
        device=boundary_f32.device,
        dtype=torch.int32,
    )
    config_f32 = torch.tensor(
        [config.near, config.far, config.invalid_epsilon, config.transmittance_threshold],
        device=boundary_f32.device,
        dtype=torch.float32,
    )
    return ops.shared_realray_rgba_depth_vjp(
        boundary_f32,
        candidate_mask_i32,
        sites_f32,
        site_rgba_f32,
        track_rays_f32,
        frame_t_f32,
        grad_rgb_f32,
        grad_alpha_f32,
        grad_depth_f32,
        config_i32,
        config_f32,
    )


def shared_realray_rgba_depth_vjp_reduce(
    boundary_f32: Tensor,
    candidate_mask_i32: Tensor,
    sites_f32: Tensor,
    site_rgba_f32: Tensor,
    track_rays_f32: Tensor,
    frame_t_f32: Tensor,
    grad_rgb_f32: Tensor,
    grad_alpha_f32: Tensor,
    grad_depth_f32: Tensor,
    config: RealRayReplayConfig,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Render shared true camera-ray tracks and reduce fixed-segment RGBA grads.

    Returns `output_rgb [K,T,3]`, `output_alpha [K,T]`, `output_depth [K,T]`,
    and `grad_site_rgba [S,4]`. This avoids materializing the smoke-only
    `[K,T,S,4]` gradient-sample tensor by using an internal chunked partial
    reduction, but it still fixes segment geometry, owner selection, sorting,
    masks, site positions, weights, rays, and topology.
    """
    boundary_f32 = boundary_f32.contiguous()
    candidate_mask_i32 = candidate_mask_i32.contiguous()
    sites_f32 = sites_f32.contiguous()
    site_rgba_f32 = site_rgba_f32.contiguous()
    track_rays_f32 = track_rays_f32.contiguous()
    frame_t_f32 = frame_t_f32.contiguous()
    grad_rgb_f32 = grad_rgb_f32.contiguous()
    grad_alpha_f32 = grad_alpha_f32.contiguous()
    grad_depth_f32 = grad_depth_f32.contiguous()
    _require_mps_tensor(boundary_f32, name="boundary_f32", dtype=torch.float32, cols=5)
    if candidate_mask_i32.device.type != "mps":
        raise ValueError("candidate_mask_i32 must be on MPS")
    if candidate_mask_i32.dtype != torch.int32:
        raise ValueError("candidate_mask_i32 must have dtype torch.int32")
    if candidate_mask_i32.ndim != 2:
        raise ValueError("candidate_mask_i32 must have shape [track_count * time_slab_count, mask_word_count]")
    if not candidate_mask_i32.is_contiguous():
        raise ValueError("candidate_mask_i32 must be contiguous")
    _require_mps_tensor(sites_f32, name="sites_f32", dtype=torch.float32, cols=5)
    _require_mps_tensor(site_rgba_f32, name="site_rgba_f32", dtype=torch.float32, cols=4)
    _require_mps_tensor(track_rays_f32, name="track_rays_f32", dtype=torch.float32, cols=6)
    _require_mps_tensor(frame_t_f32, name="frame_t_f32", dtype=torch.float32, cols=None)
    if grad_rgb_f32.device.type != "mps":
        raise ValueError("grad_rgb_f32 must be on MPS")
    if grad_rgb_f32.dtype != torch.float32:
        raise ValueError("grad_rgb_f32 must be float32")
    if grad_rgb_f32.ndim != 3 or grad_rgb_f32.shape[2] != 3:
        raise ValueError("grad_rgb_f32 must have shape [track_count, frame_count, 3]")
    if not grad_rgb_f32.is_contiguous():
        raise ValueError("grad_rgb_f32 must be contiguous")
    if grad_alpha_f32.device.type != "mps":
        raise ValueError("grad_alpha_f32 must be on MPS")
    if grad_alpha_f32.dtype != torch.float32:
        raise ValueError("grad_alpha_f32 must be float32")
    if grad_alpha_f32.ndim != 2:
        raise ValueError("grad_alpha_f32 must have shape [track_count, frame_count]")
    if not grad_alpha_f32.is_contiguous():
        raise ValueError("grad_alpha_f32 must be contiguous")
    if grad_depth_f32.device.type != "mps":
        raise ValueError("grad_depth_f32 must be on MPS")
    if grad_depth_f32.dtype != torch.float32:
        raise ValueError("grad_depth_f32 must be float32")
    if grad_depth_f32.ndim != 2:
        raise ValueError("grad_depth_f32 must have shape [track_count, frame_count]")
    if not grad_depth_f32.is_contiguous():
        raise ValueError("grad_depth_f32 must be contiguous")
    if track_rays_f32.shape[0] <= 0:
        raise ValueError("track_rays_f32 must contain at least one track")
    if frame_t_f32.shape[0] <= 0:
        raise ValueError("frame_t_f32 must contain at least one frame")
    if sites_f32.shape[0] <= 0:
        raise ValueError("sites_f32 must contain at least one site")
    if boundary_f32.shape[0] > 128:
        raise ValueError("shared_realray_rgba_depth_vjp_reduce currently supports at most 128 boundaries")
    if sites_f32.shape[0] > 64:
        raise ValueError("shared_realray_rgba_depth_vjp_reduce currently supports at most 64 sites")
    if site_rgba_f32.shape[0] != sites_f32.shape[0]:
        raise ValueError("site_rgba_f32 row count must match sites_f32 rows")
    if candidate_mask_i32.shape[0] % track_rays_f32.shape[0] != 0:
        raise ValueError("candidate_mask_i32 rows must be track_count * time_slab_count")
    time_slab_count = candidate_mask_i32.shape[0] // track_rays_f32.shape[0]
    expected_words = (boundary_f32.shape[0] + 31) // 32
    if candidate_mask_i32.shape[1] != expected_words:
        raise ValueError(f"candidate_mask_i32 word count must be ceil(boundary_count / 32) = {expected_words}")
    expected_shape = (track_rays_f32.shape[0], frame_t_f32.shape[0])
    if grad_rgb_f32.shape[:2] != expected_shape:
        raise ValueError("grad_rgb_f32 shape must be [track_count, frame_count, 3]")
    if grad_alpha_f32.shape != expected_shape:
        raise ValueError("grad_alpha_f32 shape must be [track_count, frame_count]")
    if grad_depth_f32.shape != expected_shape:
        raise ValueError("grad_depth_f32 shape must be [track_count, frame_count]")
    if config.far <= config.near:
        raise ValueError("config.far must be greater than config.near")
    if config.invalid_epsilon <= 0.0:
        raise ValueError("config.invalid_epsilon must be positive")
    if config.transmittance_threshold < 0.0:
        raise ValueError("config.transmittance_threshold must be nonnegative")

    ops = torch.ops.world_foam_lane2_fused_slab_v0
    if not hasattr(ops, "shared_realray_rgba_depth_vjp_reduce"):
        raise RuntimeError(
            "world_foam_lane2_fused_slab_v0 shared_realray_rgba_depth_vjp_reduce op not found. Build this variant first."
        )
    config_i32 = torch.tensor(
        [
            boundary_f32.shape[0],
            track_rays_f32.shape[0],
            sites_f32.shape[0],
            frame_t_f32.shape[0],
            time_slab_count,
            candidate_mask_i32.shape[1],
        ],
        device=boundary_f32.device,
        dtype=torch.int32,
    )
    config_f32 = torch.tensor(
        [config.near, config.far, config.invalid_epsilon, config.transmittance_threshold],
        device=boundary_f32.device,
        dtype=torch.float32,
    )
    return ops.shared_realray_rgba_depth_vjp_reduce(
        boundary_f32,
        candidate_mask_i32,
        sites_f32,
        site_rgba_f32,
        track_rays_f32,
        frame_t_f32,
        grad_rgb_f32,
        grad_alpha_f32,
        grad_depth_f32,
        config_i32,
        config_f32,
    )


def shared_realray_rgba_depth_vjp_reduce_csr(
    boundary_f32: Tensor,
    row_index_i32: Tensor,
    candidate_row_offsets_i32: Tensor,
    candidate_boundary_ids_i32: Tensor,
    sites_f32: Tensor,
    site_rgba_f32: Tensor,
    track_rays_f32: Tensor,
    frame_t_f32: Tensor,
    grad_rgb_f32: Tensor,
    grad_alpha_f32: Tensor,
    grad_depth_f32: Tensor,
    config: RealRayReplayConfig,
    *,
    time_slab_count: int,
    row_count: int,
    reduce_chunk_size: int = 4,
    use_direct_atomic: bool = False,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Render shared real-ray tracks and reduce site RGBA grads from CSR candidates.

    `row_index_i32 [K]` maps each track to a CSR row group. Per-track CSR uses
    `row_index_i32[track] = track`; tiled CSR maps many tracks to the same row
    group. The per-slab row is `row_index * time_slab_count + slab_id`.
    """
    boundary_f32 = boundary_f32.contiguous()
    row_index_i32 = row_index_i32.contiguous()
    candidate_row_offsets_i32 = candidate_row_offsets_i32.contiguous()
    candidate_boundary_ids_i32 = candidate_boundary_ids_i32.contiguous()
    sites_f32 = sites_f32.contiguous()
    site_rgba_f32 = site_rgba_f32.contiguous()
    track_rays_f32 = track_rays_f32.contiguous()
    frame_t_f32 = frame_t_f32.contiguous()
    grad_rgb_f32 = grad_rgb_f32.contiguous()
    grad_alpha_f32 = grad_alpha_f32.contiguous()
    grad_depth_f32 = grad_depth_f32.contiguous()
    _require_mps_tensor(boundary_f32, name="boundary_f32", dtype=torch.float32, cols=5)
    _require_mps_tensor(row_index_i32, name="row_index_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(candidate_row_offsets_i32, name="candidate_row_offsets_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(candidate_boundary_ids_i32, name="candidate_boundary_ids_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(sites_f32, name="sites_f32", dtype=torch.float32, cols=5)
    _require_mps_tensor(site_rgba_f32, name="site_rgba_f32", dtype=torch.float32, cols=4)
    _require_mps_tensor(track_rays_f32, name="track_rays_f32", dtype=torch.float32, cols=6)
    _require_mps_tensor(frame_t_f32, name="frame_t_f32", dtype=torch.float32, cols=None)
    if grad_rgb_f32.device.type != "mps":
        raise ValueError("grad_rgb_f32 must be on MPS")
    if grad_rgb_f32.dtype != torch.float32:
        raise ValueError("grad_rgb_f32 must be float32")
    if grad_rgb_f32.ndim != 3 or grad_rgb_f32.shape[2] != 3:
        raise ValueError("grad_rgb_f32 must have shape [track_count, frame_count, 3]")
    if not grad_rgb_f32.is_contiguous():
        raise ValueError("grad_rgb_f32 must be contiguous")
    if grad_alpha_f32.device.type != "mps":
        raise ValueError("grad_alpha_f32 must be on MPS")
    if grad_alpha_f32.dtype != torch.float32:
        raise ValueError("grad_alpha_f32 must be float32")
    if grad_alpha_f32.ndim != 2:
        raise ValueError("grad_alpha_f32 must have shape [track_count, frame_count]")
    if not grad_alpha_f32.is_contiguous():
        raise ValueError("grad_alpha_f32 must be contiguous")
    if grad_depth_f32.device.type != "mps":
        raise ValueError("grad_depth_f32 must be on MPS")
    if grad_depth_f32.dtype != torch.float32:
        raise ValueError("grad_depth_f32 must be float32")
    if grad_depth_f32.ndim != 2:
        raise ValueError("grad_depth_f32 must have shape [track_count, frame_count]")
    if not grad_depth_f32.is_contiguous():
        raise ValueError("grad_depth_f32 must be contiguous")
    if track_rays_f32.shape[0] <= 0:
        raise ValueError("track_rays_f32 must contain at least one track")
    if frame_t_f32.shape[0] <= 0:
        raise ValueError("frame_t_f32 must contain at least one frame")
    if sites_f32.shape[0] <= 0:
        raise ValueError("sites_f32 must contain at least one site")
    if boundary_f32.shape[0] > 128:
        raise ValueError("shared_realray_rgba_depth_vjp_reduce_csr currently supports at most 128 boundaries")
    if sites_f32.shape[0] > 64:
        raise ValueError("shared_realray_rgba_depth_vjp_reduce_csr currently supports at most 64 sites")
    if site_rgba_f32.shape[0] != sites_f32.shape[0]:
        raise ValueError("site_rgba_f32 row count must match sites_f32 rows")
    if row_index_i32.shape[0] != track_rays_f32.shape[0]:
        raise ValueError("row_index_i32 length must match track count")
    if time_slab_count <= 0:
        raise ValueError("time_slab_count must be positive")
    if row_count <= 0:
        raise ValueError("row_count must be positive")
    if reduce_chunk_size <= 0:
        raise ValueError("reduce_chunk_size must be positive")
    expected_offsets = row_count * time_slab_count + 1
    if candidate_row_offsets_i32.shape[0] != expected_offsets:
        raise ValueError("candidate_row_offsets_i32 length must be row_count * time_slab_count + 1")
    _validate_csr_values(
        row_index_i32=row_index_i32,
        candidate_row_offsets_i32=candidate_row_offsets_i32,
        candidate_boundary_ids_i32=candidate_boundary_ids_i32,
        row_count=row_count,
        candidate_count=candidate_boundary_ids_i32.shape[0],
        boundary_count=boundary_f32.shape[0],
    )
    expected_shape = (track_rays_f32.shape[0], frame_t_f32.shape[0])
    if grad_rgb_f32.shape[:2] != expected_shape:
        raise ValueError("grad_rgb_f32 shape must be [track_count, frame_count, 3]")
    if grad_alpha_f32.shape != expected_shape:
        raise ValueError("grad_alpha_f32 shape must be [track_count, frame_count]")
    if grad_depth_f32.shape != expected_shape:
        raise ValueError("grad_depth_f32 shape must be [track_count, frame_count]")
    if config.far <= config.near:
        raise ValueError("config.far must be greater than config.near")
    if config.invalid_epsilon <= 0.0:
        raise ValueError("config.invalid_epsilon must be positive")
    if config.transmittance_threshold < 0.0:
        raise ValueError("config.transmittance_threshold must be nonnegative")

    ops = torch.ops.world_foam_lane2_fused_slab_v0
    if not hasattr(ops, "shared_realray_rgba_depth_vjp_reduce_csr"):
        raise RuntimeError(
            "world_foam_lane2_fused_slab_v0 shared_realray_rgba_depth_vjp_reduce_csr op not found. Build this variant first."
        )
    config_i32 = torch.tensor(
        [
            boundary_f32.shape[0],
            track_rays_f32.shape[0],
            sites_f32.shape[0],
            frame_t_f32.shape[0],
            time_slab_count,
            row_count,
            candidate_boundary_ids_i32.shape[0],
        ],
        device=boundary_f32.device,
        dtype=torch.int32,
    )
    config_f32 = torch.tensor(
        [config.near, config.far, config.invalid_epsilon, config.transmittance_threshold],
        device=boundary_f32.device,
        dtype=torch.float32,
    )
    return ops.shared_realray_rgba_depth_vjp_reduce_csr(
        boundary_f32,
        row_index_i32,
        candidate_row_offsets_i32,
        candidate_boundary_ids_i32,
        sites_f32,
        site_rgba_f32,
        track_rays_f32,
        frame_t_f32,
        grad_rgb_f32,
        grad_alpha_f32,
        grad_depth_f32,
        config_i32,
        config_f32,
    )


def fused_slab_affine_num32_den16_vjp_reduce(
    row_index_i32: Tensor,
    candidate_row_offsets_i32: Tensor,
    candidate_depth_num_f32: Tensor,
    candidate_depth_den_f16: Tensor,
    sites_f32: Tensor,
    site_rgba_f32: Tensor,
    ray_coeff_f32: Tensor,
    frame_t_f32: Tensor,
    grad_rgb_f32: Tensor,
    grad_alpha_f32: Tensor,
    grad_depth_f32: Tensor,
    config: RealRayReplayConfig,
    *,
    time_slab_count: int,
    row_count: int,
    reduce_chunk_size: int = 4,
    use_direct_atomic: bool = False,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Render mixed affine CSR candidates and reduce site RGBA gradients.

    This matches `fused_slab_affine_num32_den16_realray_rgba_depth_replay`
    for forward values, while adding the same frozen-geometry site-RGBA VJP
    contract used by the existing CSR autograd path.
    """
    row_index_i32 = row_index_i32.contiguous()
    candidate_row_offsets_i32 = candidate_row_offsets_i32.contiguous()
    candidate_depth_num_f32 = candidate_depth_num_f32.contiguous()
    candidate_depth_den_f16 = candidate_depth_den_f16.contiguous()
    sites_f32 = sites_f32.contiguous()
    site_rgba_f32 = site_rgba_f32.contiguous()
    ray_coeff_f32 = ray_coeff_f32.contiguous()
    frame_t_f32 = frame_t_f32.contiguous()
    grad_rgb_f32 = grad_rgb_f32.contiguous()
    grad_alpha_f32 = grad_alpha_f32.contiguous()
    grad_depth_f32 = grad_depth_f32.contiguous()
    _require_mps_tensor(row_index_i32, name="row_index_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(candidate_row_offsets_i32, name="candidate_row_offsets_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(candidate_depth_num_f32, name="candidate_depth_num_f32", dtype=torch.float32, cols=2)
    _require_mps_tensor(candidate_depth_den_f16, name="candidate_depth_den_f16", dtype=torch.float16, cols=2)
    _require_mps_tensor(sites_f32, name="sites_f32", dtype=torch.float32, cols=5)
    _require_mps_tensor(site_rgba_f32, name="site_rgba_f32", dtype=torch.float32, cols=4)
    _require_mps_tensor(ray_coeff_f32, name="ray_coeff_f32", dtype=torch.float32, cols=12)
    _require_mps_tensor(frame_t_f32, name="frame_t_f32", dtype=torch.float32, cols=None)
    if grad_rgb_f32.device.type != "mps":
        raise ValueError("grad_rgb_f32 must be on MPS")
    if grad_rgb_f32.dtype != torch.float32:
        raise ValueError("grad_rgb_f32 must be float32")
    if grad_rgb_f32.ndim != 3 or grad_rgb_f32.shape[2] != 3:
        raise ValueError("grad_rgb_f32 must have shape [track_count, frame_count, 3]")
    if not grad_rgb_f32.is_contiguous():
        raise ValueError("grad_rgb_f32 must be contiguous")
    if grad_alpha_f32.device.type != "mps":
        raise ValueError("grad_alpha_f32 must be on MPS")
    if grad_alpha_f32.dtype != torch.float32:
        raise ValueError("grad_alpha_f32 must be float32")
    if grad_alpha_f32.ndim != 2:
        raise ValueError("grad_alpha_f32 must have shape [track_count, frame_count]")
    if not grad_alpha_f32.is_contiguous():
        raise ValueError("grad_alpha_f32 must be contiguous")
    if grad_depth_f32.device.type != "mps":
        raise ValueError("grad_depth_f32 must be on MPS")
    if grad_depth_f32.dtype != torch.float32:
        raise ValueError("grad_depth_f32 must be float32")
    if grad_depth_f32.ndim != 2:
        raise ValueError("grad_depth_f32 must have shape [track_count, frame_count]")
    if not grad_depth_f32.is_contiguous():
        raise ValueError("grad_depth_f32 must be contiguous")
    if ray_coeff_f32.shape[0] <= 0:
        raise ValueError("ray_coeff_f32 must contain at least one track")
    if frame_t_f32.shape[0] <= 0:
        raise ValueError("frame_t_f32 must contain at least one frame")
    if sites_f32.shape[0] <= 0:
        raise ValueError("sites_f32 must contain at least one site")
    if sites_f32.shape[0] > 64:
        raise ValueError("fused_slab_affine_num32_den16_vjp_reduce currently supports at most 64 sites")
    if site_rgba_f32.shape[0] != sites_f32.shape[0]:
        raise ValueError("site_rgba_f32 row count must match sites_f32 rows")
    if row_index_i32.shape[0] != ray_coeff_f32.shape[0]:
        raise ValueError("row_index_i32 length must match track count")
    if candidate_depth_den_f16.shape[0] != candidate_depth_num_f32.shape[0]:
        raise ValueError("candidate_depth_den_f16 row count must match candidate_depth_num_f32 rows")
    if time_slab_count <= 0:
        raise ValueError("time_slab_count must be positive")
    if row_count <= 0:
        raise ValueError("row_count must be positive")
    if reduce_chunk_size <= 0:
        raise ValueError("reduce_chunk_size must be positive")
    expected_offsets = row_count * time_slab_count + 1
    if candidate_row_offsets_i32.shape[0] != expected_offsets:
        raise ValueError("candidate_row_offsets_i32 length must be row_count * time_slab_count + 1")
    row_index_cpu = row_index_i32.detach().cpu()
    row_offsets_cpu = candidate_row_offsets_i32.detach().cpu()
    if int(row_index_cpu.min().item()) < 0 or int(row_index_cpu.max().item()) >= row_count:
        raise ValueError("row_index_i32 values must be in [0, row_count)")
    _validate_csr_offsets_cpu(row_offsets_cpu, candidate_count=candidate_depth_num_f32.shape[0])
    expected_shape = (ray_coeff_f32.shape[0], frame_t_f32.shape[0])
    if grad_rgb_f32.shape[:2] != expected_shape:
        raise ValueError("grad_rgb_f32 shape must be [track_count, frame_count, 3]")
    if grad_alpha_f32.shape != expected_shape:
        raise ValueError("grad_alpha_f32 shape must be [track_count, frame_count]")
    if grad_depth_f32.shape != expected_shape:
        raise ValueError("grad_depth_f32 shape must be [track_count, frame_count]")
    if config.far <= config.near:
        raise ValueError("config.far must be greater than config.near")
    if config.invalid_epsilon <= 0.0:
        raise ValueError("config.invalid_epsilon must be positive")
    if config.transmittance_threshold < 0.0:
        raise ValueError("config.transmittance_threshold must be nonnegative")

    ops = torch.ops.world_foam_lane2_fused_slab_v0
    op_name = (
        "fused_slab_affine_num32_den16_vjp_direct_atomic"
        if use_direct_atomic
        else "fused_slab_affine_num32_den16_vjp_reduce"
    )
    if not hasattr(ops, op_name):
        raise RuntimeError(f"world_foam_lane2_fused_slab_v0 {op_name} op not found. Build this variant first.")
    config_i32 = torch.tensor(
        [
            0,
            ray_coeff_f32.shape[0],
            sites_f32.shape[0],
            frame_t_f32.shape[0],
            time_slab_count,
            row_count,
            candidate_depth_num_f32.shape[0],
            reduce_chunk_size,
        ],
        device=row_index_i32.device,
        dtype=torch.int32,
    )
    config_f32 = torch.tensor(
        [config.near, config.far, config.invalid_epsilon, config.transmittance_threshold],
        device=row_index_i32.device,
        dtype=torch.float32,
    )
    return getattr(ops, op_name)(
        row_index_i32,
        candidate_row_offsets_i32,
        candidate_depth_num_f32,
        candidate_depth_den_f16,
        sites_f32,
        site_rgba_f32,
        ray_coeff_f32,
        frame_t_f32,
        grad_rgb_f32,
        grad_alpha_f32,
        grad_depth_f32,
        config_i32,
        config_f32,
    )


def fused_slab_affine_num32_den16_vjp_direct_atomic(
    row_index_i32: Tensor,
    candidate_row_offsets_i32: Tensor,
    candidate_depth_num_f32: Tensor,
    candidate_depth_den_f16: Tensor,
    sites_f32: Tensor,
    site_rgba_f32: Tensor,
    ray_coeff_f32: Tensor,
    frame_t_f32: Tensor,
    grad_rgb_f32: Tensor,
    grad_alpha_f32: Tensor,
    grad_depth_f32: Tensor,
    config: RealRayReplayConfig,
    *,
    time_slab_count: int,
    row_count: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Render mixed affine CSR candidates and accumulate site RGBA gradients with atomics."""
    return fused_slab_affine_num32_den16_vjp_reduce(
        row_index_i32,
        candidate_row_offsets_i32,
        candidate_depth_num_f32,
        candidate_depth_den_f16,
        sites_f32,
        site_rgba_f32,
        ray_coeff_f32,
        frame_t_f32,
        grad_rgb_f32,
        grad_alpha_f32,
        grad_depth_f32,
        config,
        time_slab_count=time_slab_count,
        row_count=row_count,
        reduce_chunk_size=1,
        use_direct_atomic=True,
    )


def fused_slab_affine_num32_den16_vjp_direct_atomic_grad_only(
    row_index_i32: Tensor,
    candidate_row_offsets_i32: Tensor,
    candidate_depth_num_f32: Tensor,
    candidate_depth_den_f16: Tensor,
    sites_f32: Tensor,
    site_rgba_f32: Tensor,
    ray_coeff_f32: Tensor,
    frame_t_f32: Tensor,
    grad_rgb_f32: Tensor,
    grad_alpha_f32: Tensor,
    grad_depth_f32: Tensor,
    config: RealRayReplayConfig,
    *,
    time_slab_count: int,
    row_count: int,
) -> Tensor:
    """Accumulate mixed affine site RGBA gradients without writing replay outputs."""
    row_index_i32 = row_index_i32.contiguous()
    candidate_row_offsets_i32 = candidate_row_offsets_i32.contiguous()
    candidate_depth_num_f32 = candidate_depth_num_f32.contiguous()
    candidate_depth_den_f16 = candidate_depth_den_f16.contiguous()
    sites_f32 = sites_f32.contiguous()
    site_rgba_f32 = site_rgba_f32.contiguous()
    ray_coeff_f32 = ray_coeff_f32.contiguous()
    frame_t_f32 = frame_t_f32.contiguous()
    grad_rgb_f32 = grad_rgb_f32.contiguous()
    grad_alpha_f32 = grad_alpha_f32.contiguous()
    grad_depth_f32 = grad_depth_f32.contiguous()
    _require_mps_tensor(row_index_i32, name="row_index_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(candidate_row_offsets_i32, name="candidate_row_offsets_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(candidate_depth_num_f32, name="candidate_depth_num_f32", dtype=torch.float32, cols=2)
    _require_mps_tensor(candidate_depth_den_f16, name="candidate_depth_den_f16", dtype=torch.float16, cols=2)
    _require_mps_tensor(sites_f32, name="sites_f32", dtype=torch.float32, cols=5)
    _require_mps_tensor(site_rgba_f32, name="site_rgba_f32", dtype=torch.float32, cols=4)
    _require_mps_tensor(ray_coeff_f32, name="ray_coeff_f32", dtype=torch.float32, cols=12)
    _require_mps_tensor(frame_t_f32, name="frame_t_f32", dtype=torch.float32, cols=None)
    if grad_rgb_f32.device.type != "mps" or grad_rgb_f32.dtype != torch.float32:
        raise ValueError("grad_rgb_f32 must be float32 on MPS")
    if grad_rgb_f32.ndim != 3 or grad_rgb_f32.shape[2] != 3:
        raise ValueError("grad_rgb_f32 must have shape [track_count, frame_count, 3]")
    if grad_alpha_f32.device.type != "mps" or grad_alpha_f32.dtype != torch.float32:
        raise ValueError("grad_alpha_f32 must be float32 on MPS")
    if grad_alpha_f32.ndim != 2:
        raise ValueError("grad_alpha_f32 must have shape [track_count, frame_count]")
    if grad_depth_f32.device.type != "mps" or grad_depth_f32.dtype != torch.float32:
        raise ValueError("grad_depth_f32 must be float32 on MPS")
    if grad_depth_f32.ndim != 2:
        raise ValueError("grad_depth_f32 must have shape [track_count, frame_count]")
    if ray_coeff_f32.shape[0] <= 0 or frame_t_f32.shape[0] <= 0:
        raise ValueError("ray_coeff_f32 and frame_t_f32 must be nonempty")
    if sites_f32.shape[0] <= 0 or sites_f32.shape[0] > 64:
        raise ValueError("sites_f32 row count must be in [1, 64]")
    if site_rgba_f32.shape[0] != sites_f32.shape[0]:
        raise ValueError("site_rgba_f32 row count must match sites_f32")
    if row_index_i32.shape[0] != ray_coeff_f32.shape[0]:
        raise ValueError("row_index_i32 length must match track count")
    if time_slab_count <= 0:
        raise ValueError("time_slab_count must be positive")
    if row_count <= 0:
        raise ValueError("row_count must be positive")
    if candidate_depth_den_f16.shape[0] != candidate_depth_num_f32.shape[0]:
        raise ValueError("candidate numerator and denominator row counts must match")
    if candidate_row_offsets_i32.shape[0] != row_count * time_slab_count + 1:
        raise ValueError("candidate_row_offsets_i32 length must be row_count * time_slab_count + 1")
    row_index_cpu = row_index_i32.detach().cpu()
    row_offsets_cpu = candidate_row_offsets_i32.detach().cpu()
    if int(row_index_cpu.min().item()) < 0 or int(row_index_cpu.max().item()) >= row_count:
        raise ValueError("row_index_i32 values must be in [0, row_count)")
    _validate_csr_offsets_cpu(row_offsets_cpu, candidate_count=candidate_depth_num_f32.shape[0])
    expected_shape = (ray_coeff_f32.shape[0], frame_t_f32.shape[0])
    if grad_rgb_f32.shape[:2] != expected_shape:
        raise ValueError("grad_rgb_f32 shape must be [track_count, frame_count, 3]")
    if grad_alpha_f32.shape != expected_shape or grad_depth_f32.shape != expected_shape:
        raise ValueError("grad_alpha_f32 and grad_depth_f32 must have shape [track_count, frame_count]")
    if config.far <= config.near:
        raise ValueError("config.far must be greater than config.near")
    if config.invalid_epsilon <= 0.0:
        raise ValueError("config.invalid_epsilon must be positive")
    if config.transmittance_threshold < 0.0:
        raise ValueError("config.transmittance_threshold must be nonnegative")

    ops = torch.ops.world_foam_lane2_fused_slab_v0
    op_name = "fused_slab_affine_num32_den16_vjp_direct_atomic_grad_only"
    if not hasattr(ops, op_name):
        raise RuntimeError(f"world_foam_lane2_fused_slab_v0 {op_name} op not found. Build this variant first.")
    config_i32 = torch.tensor(
        [
            0,
            ray_coeff_f32.shape[0],
            sites_f32.shape[0],
            frame_t_f32.shape[0],
            time_slab_count,
            row_count,
            candidate_depth_num_f32.shape[0],
            1,
        ],
        device=row_index_i32.device,
        dtype=torch.int32,
    )
    config_f32 = torch.tensor(
        [config.near, config.far, config.invalid_epsilon, config.transmittance_threshold],
        device=row_index_i32.device,
        dtype=torch.float32,
    )
    return getattr(ops, op_name)(
        row_index_i32,
        candidate_row_offsets_i32,
        candidate_depth_num_f32,
        candidate_depth_den_f16,
        sites_f32,
        site_rgba_f32,
        ray_coeff_f32,
        frame_t_f32,
        grad_rgb_f32,
        grad_alpha_f32,
        grad_depth_f32,
        config_i32,
        config_f32,
    )


def _segment_tape_config_tensors(
    *,
    segment_offsets_i32: Tensor,
    segment_owner_i32: Tensor,
    segment_length_f32: Tensor,
    segment_mid_f32: Tensor,
    site_rgba_f32: Tensor,
    config: RealRayReplayConfig,
    track_count: int,
    frame_count: int,
    max_segments_per_sample: int | None,
) -> tuple[Tensor, Tensor]:
    segment_offsets_i32 = segment_offsets_i32.contiguous()
    segment_owner_i32 = segment_owner_i32.contiguous()
    segment_length_f32 = segment_length_f32.contiguous()
    segment_mid_f32 = segment_mid_f32.contiguous()
    site_rgba_f32 = site_rgba_f32.contiguous()
    _require_mps_tensor(segment_offsets_i32, name="segment_offsets_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(segment_owner_i32, name="segment_owner_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(segment_length_f32, name="segment_length_f32", dtype=torch.float32, cols=None)
    _require_mps_tensor(segment_mid_f32, name="segment_mid_f32", dtype=torch.float32, cols=None)
    _require_mps_tensor(site_rgba_f32, name="site_rgba_f32", dtype=torch.float32, cols=4)
    if track_count <= 0 or frame_count <= 0:
        raise ValueError("track_count and frame_count must be positive")
    if site_rgba_f32.shape[0] <= 0 or site_rgba_f32.shape[0] > 64:
        raise ValueError("site_rgba_f32 row count must be in [1, 64]")
    if (
        segment_owner_i32.shape[0] != segment_length_f32.shape[0]
        or segment_owner_i32.shape[0] != segment_mid_f32.shape[0]
    ):
        raise ValueError("segment owner, length, and mid arrays must have matching lengths")
    if config.far <= config.near:
        raise ValueError("config.far must be greater than config.near")
    if config.transmittance_threshold < 0.0:
        raise ValueError("config.transmittance_threshold must be nonnegative")
    _validate_segment_tape_offsets_cpu(
        segment_offsets_i32.detach().cpu(),
        sample_count=track_count * frame_count,
        segment_count=segment_owner_i32.shape[0],
        max_segments_per_sample=max_segments_per_sample,
    )
    config_i32 = torch.tensor(
        [track_count, frame_count, site_rgba_f32.shape[0], segment_owner_i32.shape[0]],
        device=segment_offsets_i32.device,
        dtype=torch.int32,
    )
    config_f32 = torch.tensor(
        [config.far, config.transmittance_threshold],
        device=segment_offsets_i32.device,
        dtype=torch.float32,
    )
    return config_i32, config_f32


def _segment_tape_rgb_mse_config_tensors(
    *,
    segment_offsets_i32: Tensor,
    segment_owner_i32: Tensor,
    segment_length_f32: Tensor,
    site_rgba_f32: Tensor,
    config: RealRayReplayConfig,
    track_count: int,
    frame_count: int,
    max_segments_per_sample: int | None,
) -> tuple[Tensor, Tensor]:
    segment_offsets_i32 = segment_offsets_i32.contiguous()
    segment_owner_i32 = segment_owner_i32.contiguous()
    segment_length_f32 = segment_length_f32.contiguous()
    site_rgba_f32 = site_rgba_f32.contiguous()
    _require_mps_tensor(segment_offsets_i32, name="segment_offsets_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(segment_owner_i32, name="segment_owner_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(segment_length_f32, name="segment_length_f32", dtype=torch.float32, cols=None)
    _require_mps_tensor(site_rgba_f32, name="site_rgba_f32", dtype=torch.float32, cols=4)
    if track_count <= 0 or frame_count <= 0:
        raise ValueError("track_count and frame_count must be positive")
    if site_rgba_f32.shape[0] <= 0 or site_rgba_f32.shape[0] > 64:
        raise ValueError("site_rgba_f32 row count must be in [1, 64]")
    if segment_owner_i32.shape[0] != segment_length_f32.shape[0]:
        raise ValueError("segment owner and length arrays must have matching lengths")
    if config.far <= config.near:
        raise ValueError("config.far must be greater than config.near")
    if config.transmittance_threshold < 0.0:
        raise ValueError("config.transmittance_threshold must be nonnegative")
    _validate_segment_tape_offsets_cpu(
        segment_offsets_i32.detach().cpu(),
        sample_count=track_count * frame_count,
        segment_count=segment_owner_i32.shape[0],
        max_segments_per_sample=max_segments_per_sample,
    )
    config_i32 = torch.tensor(
        [track_count, frame_count, site_rgba_f32.shape[0], segment_owner_i32.shape[0]],
        device=segment_offsets_i32.device,
        dtype=torch.int32,
    )
    config_f32 = torch.tensor(
        [config.far, config.transmittance_threshold],
        device=segment_offsets_i32.device,
        dtype=torch.float32,
    )
    return config_i32, config_f32


def segment_tape_rgba_depth_replay(
    segment_offsets_i32: Tensor,
    segment_owner_i32: Tensor,
    segment_length_f32: Tensor,
    segment_mid_f32: Tensor,
    site_rgba_f32: Tensor,
    config: RealRayReplayConfig,
    *,
    track_count: int,
    frame_count: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Replay a fixed-geometry compact segment tape on MPS.

    Tape layout:
    - `segment_offsets_i32`: `[track_count * frame_count + 1]`
    - `segment_owner_i32`: `[segment_count]`
    - `segment_length_f32`: `[segment_count]`
    - `segment_mid_f32`: `[segment_count]`

    The tape fixes geometry and ownership only; RGBA/density remains live.
    """
    segment_offsets_i32 = segment_offsets_i32.contiguous()
    segment_owner_i32 = segment_owner_i32.contiguous()
    segment_length_f32 = segment_length_f32.contiguous()
    segment_mid_f32 = segment_mid_f32.contiguous()
    site_rgba_f32 = site_rgba_f32.contiguous()
    config_i32, config_f32 = _segment_tape_config_tensors(
        segment_offsets_i32=segment_offsets_i32,
        segment_owner_i32=segment_owner_i32,
        segment_length_f32=segment_length_f32,
        segment_mid_f32=segment_mid_f32,
        site_rgba_f32=site_rgba_f32,
        config=config,
        track_count=track_count,
        frame_count=frame_count,
        max_segments_per_sample=None,
    )
    ops = torch.ops.world_foam_lane2_fused_slab_v0
    op_name = "segment_tape_rgba_depth_replay"
    if not hasattr(ops, op_name):
        raise RuntimeError(f"world_foam_lane2_fused_slab_v0 {op_name} op not found. Build this variant first.")
    return getattr(ops, op_name)(
        segment_offsets_i32,
        segment_owner_i32,
        segment_length_f32,
        segment_mid_f32,
        site_rgba_f32,
        config_i32,
        config_f32,
    )


def segment_tape_mse_vjp_direct_atomic_rgb_only(
    segment_offsets_i32: Tensor,
    segment_owner_i32: Tensor,
    segment_length_f32: Tensor,
    segment_mid_f32: Tensor,
    site_rgba_f32: Tensor,
    target_rgb_f32: Tensor,
    config: RealRayReplayConfig,
    *,
    track_count: int,
    frame_count: int,
) -> tuple[Tensor, Tensor]:
    """Compute RGB MSE and site RGBA gradients for a compact segment/owner-run tape."""
    segment_offsets_i32 = segment_offsets_i32.contiguous()
    segment_owner_i32 = segment_owner_i32.contiguous()
    segment_length_f32 = segment_length_f32.contiguous()
    segment_mid_f32 = segment_mid_f32.contiguous()
    site_rgba_f32 = site_rgba_f32.contiguous()
    target_rgb_f32 = target_rgb_f32.contiguous()
    config_i32, config_f32 = _segment_tape_config_tensors(
        segment_offsets_i32=segment_offsets_i32,
        segment_owner_i32=segment_owner_i32,
        segment_length_f32=segment_length_f32,
        segment_mid_f32=segment_mid_f32,
        site_rgba_f32=site_rgba_f32,
        config=config,
        track_count=track_count,
        frame_count=frame_count,
        max_segments_per_sample=129,
    )
    if target_rgb_f32.device.type != "mps" or target_rgb_f32.dtype != torch.float32:
        raise ValueError("target_rgb_f32 must be float32 on MPS")
    if target_rgb_f32.shape != (track_count, frame_count, 3):
        raise ValueError("target_rgb_f32 must have shape [track_count, frame_count, 3]")
    ops = torch.ops.world_foam_lane2_fused_slab_v0
    op_name = "segment_tape_mse_vjp_direct_atomic_rgb_only"
    if not hasattr(ops, op_name):
        raise RuntimeError(f"world_foam_lane2_fused_slab_v0 {op_name} op not found. Build this variant first.")
    return getattr(ops, op_name)(
        segment_offsets_i32,
        segment_owner_i32,
        segment_length_f32,
        site_rgba_f32,
        target_rgb_f32,
        config_i32,
        config_f32,
    )


def segment_tape_nomids_mse_vjp_direct_atomic_rgb_only(
    segment_offsets_i32: Tensor,
    segment_owner_i32: Tensor,
    segment_length_f32: Tensor,
    site_rgba_f32: Tensor,
    target_rgb_f32: Tensor,
    config: RealRayReplayConfig,
    *,
    track_count: int,
    frame_count: int,
) -> tuple[Tensor, Tensor]:
    """Compute RGB MSE/VJP for a compact segment tape without a resident mids tensor."""
    segment_offsets_i32 = segment_offsets_i32.contiguous()
    segment_owner_i32 = segment_owner_i32.contiguous()
    segment_length_f32 = segment_length_f32.contiguous()
    site_rgba_f32 = site_rgba_f32.contiguous()
    target_rgb_f32 = target_rgb_f32.contiguous()
    config_i32, config_f32 = _segment_tape_rgb_mse_config_tensors(
        segment_offsets_i32=segment_offsets_i32,
        segment_owner_i32=segment_owner_i32,
        segment_length_f32=segment_length_f32,
        site_rgba_f32=site_rgba_f32,
        config=config,
        track_count=track_count,
        frame_count=frame_count,
        max_segments_per_sample=129,
    )
    if target_rgb_f32.device.type != "mps" or target_rgb_f32.dtype != torch.float32:
        raise ValueError("target_rgb_f32 must be float32 on MPS")
    if target_rgb_f32.shape != (track_count, frame_count, 3):
        raise ValueError("target_rgb_f32 must have shape [track_count, frame_count, 3]")
    ops = torch.ops.world_foam_lane2_fused_slab_v0
    op_name = "segment_tape_mse_vjp_direct_atomic_rgb_only"
    if not hasattr(ops, op_name):
        raise RuntimeError(f"world_foam_lane2_fused_slab_v0 {op_name} op not found. Build this variant first.")
    return getattr(ops, op_name)(
        segment_offsets_i32,
        segment_owner_i32,
        segment_length_f32,
        site_rgba_f32,
        target_rgb_f32,
        config_i32,
        config_f32,
    )


def _endpoint_run_config_tensors(
    *,
    run_offsets_i32: Tensor,
    run_owner_i32: Tensor,
    run_start_f32: Tensor,
    run_end_f32: Tensor,
    site_rgba_f32: Tensor,
    config: RealRayReplayConfig,
    track_count: int,
    frame_count: int,
    max_runs_per_sample: int | None,
) -> tuple[Tensor, Tensor]:
    _require_mps_tensor(run_offsets_i32, name="run_offsets_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(run_owner_i32, name="run_owner_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(run_start_f32, name="run_start_f32", dtype=torch.float32, cols=None)
    _require_mps_tensor(run_end_f32, name="run_end_f32", dtype=torch.float32, cols=None)
    _require_mps_tensor(site_rgba_f32, name="site_rgba_f32", dtype=torch.float32, cols=4)
    if run_start_f32.shape != run_owner_i32.shape:
        raise ValueError("run_start_f32 length must match run_owner_i32")
    if run_end_f32.shape != run_owner_i32.shape:
        raise ValueError("run_end_f32 length must match run_owner_i32")
    if track_count <= 0:
        raise ValueError("track_count must be positive")
    if frame_count <= 0:
        raise ValueError("frame_count must be positive")
    if site_rgba_f32.shape[0] <= 0:
        raise ValueError("site_rgba_f32 must contain at least one site")
    if site_rgba_f32.shape[0] > 64:
        raise ValueError("endpoint run replay supports at most 64 sites")
    if config.far <= config.near:
        raise ValueError("config.far must be greater than config.near")
    if config.transmittance_threshold < 0.0:
        raise ValueError("config.transmittance_threshold must be nonnegative")
    _validate_segment_tape_offsets_cpu(
        run_offsets_i32.detach().cpu(),
        sample_count=track_count * frame_count,
        segment_count=run_owner_i32.shape[0],
        max_segments_per_sample=max_runs_per_sample,
    )
    config_i32 = torch.tensor(
        [track_count, frame_count, site_rgba_f32.shape[0], run_owner_i32.shape[0]],
        device=run_offsets_i32.device,
        dtype=torch.int32,
    )
    config_f32 = torch.tensor(
        [config.far, config.transmittance_threshold],
        device=run_offsets_i32.device,
        dtype=torch.float32,
    )
    return config_i32, config_f32


def endpoint_run_rgba_depth_replay(
    run_offsets_i32: Tensor,
    run_owner_i32: Tensor,
    run_start_f32: Tensor,
    run_end_f32: Tensor,
    site_rgba_f32: Tensor,
    config: RealRayReplayConfig,
    *,
    track_count: int,
    frame_count: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Replay same-owner endpoint runs with continuous absorption depth."""
    run_offsets_i32 = run_offsets_i32.contiguous()
    run_owner_i32 = run_owner_i32.contiguous()
    run_start_f32 = run_start_f32.contiguous()
    run_end_f32 = run_end_f32.contiguous()
    site_rgba_f32 = site_rgba_f32.contiguous()
    config_i32, config_f32 = _endpoint_run_config_tensors(
        run_offsets_i32=run_offsets_i32,
        run_owner_i32=run_owner_i32,
        run_start_f32=run_start_f32,
        run_end_f32=run_end_f32,
        site_rgba_f32=site_rgba_f32,
        config=config,
        track_count=track_count,
        frame_count=frame_count,
        max_runs_per_sample=None,
    )
    ops = torch.ops.world_foam_lane2_fused_slab_v0
    op_name = "endpoint_run_rgba_depth_replay"
    if not hasattr(ops, op_name):
        raise RuntimeError(f"world_foam_lane2_fused_slab_v0 {op_name} op not found. Build this variant first.")
    return getattr(ops, op_name)(
        run_offsets_i32,
        run_owner_i32,
        run_start_f32,
        run_end_f32,
        site_rgba_f32,
        config_i32,
        config_f32,
    )


def endpoint_run_vjp_direct_atomic_grad_only(
    run_offsets_i32: Tensor,
    run_owner_i32: Tensor,
    run_start_f32: Tensor,
    run_end_f32: Tensor,
    site_rgba_f32: Tensor,
    grad_rgb_f32: Tensor,
    grad_alpha_f32: Tensor,
    grad_depth_f32: Tensor,
    config: RealRayReplayConfig,
    *,
    track_count: int,
    frame_count: int,
) -> Tensor:
    """Accumulate site RGBA gradients for continuous-depth endpoint runs."""
    run_offsets_i32 = run_offsets_i32.contiguous()
    run_owner_i32 = run_owner_i32.contiguous()
    run_start_f32 = run_start_f32.contiguous()
    run_end_f32 = run_end_f32.contiguous()
    site_rgba_f32 = site_rgba_f32.contiguous()
    grad_rgb_f32 = grad_rgb_f32.contiguous()
    grad_alpha_f32 = grad_alpha_f32.contiguous()
    grad_depth_f32 = grad_depth_f32.contiguous()
    config_i32, config_f32 = _endpoint_run_config_tensors(
        run_offsets_i32=run_offsets_i32,
        run_owner_i32=run_owner_i32,
        run_start_f32=run_start_f32,
        run_end_f32=run_end_f32,
        site_rgba_f32=site_rgba_f32,
        config=config,
        track_count=track_count,
        frame_count=frame_count,
        max_runs_per_sample=129,
    )
    if grad_rgb_f32.device.type != "mps" or grad_rgb_f32.dtype != torch.float32:
        raise ValueError("grad_rgb_f32 must be float32 on MPS")
    if grad_rgb_f32.shape != (track_count, frame_count, 3):
        raise ValueError("grad_rgb_f32 must have shape [track_count, frame_count, 3]")
    if grad_alpha_f32.device.type != "mps" or grad_alpha_f32.dtype != torch.float32:
        raise ValueError("grad_alpha_f32 must be float32 on MPS")
    if grad_alpha_f32.shape != (track_count, frame_count):
        raise ValueError("grad_alpha_f32 must have shape [track_count, frame_count]")
    if grad_depth_f32.device.type != "mps" or grad_depth_f32.dtype != torch.float32:
        raise ValueError("grad_depth_f32 must be float32 on MPS")
    if grad_depth_f32.shape != (track_count, frame_count):
        raise ValueError("grad_depth_f32 must have shape [track_count, frame_count]")
    ops = torch.ops.world_foam_lane2_fused_slab_v0
    op_name = "endpoint_run_vjp_direct_atomic_grad_only"
    if not hasattr(ops, op_name):
        raise RuntimeError(f"world_foam_lane2_fused_slab_v0 {op_name} op not found. Build this variant first.")
    return getattr(ops, op_name)(
        run_offsets_i32,
        run_owner_i32,
        run_start_f32,
        run_end_f32,
        site_rgba_f32,
        grad_rgb_f32,
        grad_alpha_f32,
        grad_depth_f32,
        config_i32,
        config_f32,
    )


def endpoint_run_mse_vjp_direct_atomic_rgb_only(
    run_offsets_i32: Tensor,
    run_owner_i32: Tensor,
    run_start_f32: Tensor,
    run_end_f32: Tensor,
    site_rgba_f32: Tensor,
    target_rgb_f32: Tensor,
    config: RealRayReplayConfig,
    *,
    track_count: int,
    frame_count: int,
) -> tuple[Tensor, Tensor]:
    """Compute RGB MSE and site RGBA gradients for continuous-depth endpoint runs."""
    run_offsets_i32 = run_offsets_i32.contiguous()
    run_owner_i32 = run_owner_i32.contiguous()
    run_start_f32 = run_start_f32.contiguous()
    run_end_f32 = run_end_f32.contiguous()
    site_rgba_f32 = site_rgba_f32.contiguous()
    target_rgb_f32 = target_rgb_f32.contiguous()
    config_i32, config_f32 = _endpoint_run_config_tensors(
        run_offsets_i32=run_offsets_i32,
        run_owner_i32=run_owner_i32,
        run_start_f32=run_start_f32,
        run_end_f32=run_end_f32,
        site_rgba_f32=site_rgba_f32,
        config=config,
        track_count=track_count,
        frame_count=frame_count,
        max_runs_per_sample=129,
    )
    if target_rgb_f32.device.type != "mps" or target_rgb_f32.dtype != torch.float32:
        raise ValueError("target_rgb_f32 must be float32 on MPS")
    if target_rgb_f32.shape != (track_count, frame_count, 3):
        raise ValueError("target_rgb_f32 must have shape [track_count, frame_count, 3]")
    ops = torch.ops.world_foam_lane2_fused_slab_v0
    op_name = "endpoint_run_mse_vjp_direct_atomic_rgb_only"
    if not hasattr(ops, op_name):
        raise RuntimeError(f"world_foam_lane2_fused_slab_v0 {op_name} op not found. Build this variant first.")
    return getattr(ops, op_name)(
        run_offsets_i32,
        run_owner_i32,
        run_start_f32,
        run_end_f32,
        site_rgba_f32,
        target_rgb_f32,
        config_i32,
        config_f32,
    )


def _endpoint_delta_replace_config_tensors(
    *,
    base_offsets_i32: Tensor,
    base_owner_i32: Tensor,
    base_start_f32: Tensor,
    base_end_f32: Tensor,
    track_change_offsets_i32: Tensor,
    change_frame_i32: Tensor,
    change_offsets_i32: Tensor,
    change_owner_i32: Tensor,
    change_start_f32: Tensor,
    change_end_f32: Tensor,
    site_rgba_f32: Tensor,
    config: RealRayReplayConfig,
    track_count: int,
    frame_count: int,
    max_runs_per_sample: int | None,
) -> tuple[Tensor, Tensor]:
    _require_mps_tensor(base_offsets_i32, name="base_offsets_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(base_owner_i32, name="base_owner_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(base_start_f32, name="base_start_f32", dtype=torch.float32, cols=None)
    _require_mps_tensor(base_end_f32, name="base_end_f32", dtype=torch.float32, cols=None)
    _require_mps_tensor(track_change_offsets_i32, name="track_change_offsets_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(change_frame_i32, name="change_frame_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(change_offsets_i32, name="change_offsets_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(change_owner_i32, name="change_owner_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(change_start_f32, name="change_start_f32", dtype=torch.float32, cols=None)
    _require_mps_tensor(change_end_f32, name="change_end_f32", dtype=torch.float32, cols=None)
    _require_mps_tensor(site_rgba_f32, name="site_rgba_f32", dtype=torch.float32, cols=4)
    if base_start_f32.shape != base_owner_i32.shape or base_end_f32.shape != base_owner_i32.shape:
        raise ValueError("base owner, start, and end arrays must have matching lengths")
    if change_start_f32.shape != change_owner_i32.shape or change_end_f32.shape != change_owner_i32.shape:
        raise ValueError("change owner, start, and end arrays must have matching lengths")
    if track_count <= 0:
        raise ValueError("track_count must be positive")
    if frame_count <= 0:
        raise ValueError("frame_count must be positive")
    if site_rgba_f32.shape[0] <= 0:
        raise ValueError("site_rgba_f32 must contain at least one site")
    if site_rgba_f32.shape[0] > 64:
        raise ValueError("endpoint delta replay supports at most 64 sites")
    if config.far <= config.near:
        raise ValueError("config.far must be greater than config.near")
    if config.transmittance_threshold < 0.0:
        raise ValueError("config.transmittance_threshold must be nonnegative")
    _validate_segment_tape_offsets_cpu(
        base_offsets_i32.detach().cpu(),
        sample_count=track_count,
        segment_count=base_owner_i32.shape[0],
        max_segments_per_sample=max_runs_per_sample,
    )
    _validate_segment_tape_offsets_cpu(
        track_change_offsets_i32.detach().cpu(),
        sample_count=track_count,
        segment_count=change_frame_i32.shape[0],
        max_segments_per_sample=None,
    )
    _validate_segment_tape_offsets_cpu(
        change_offsets_i32.detach().cpu(),
        sample_count=change_frame_i32.shape[0],
        segment_count=change_owner_i32.shape[0],
        max_segments_per_sample=max_runs_per_sample,
    )
    config_i32 = torch.tensor(
        [
            track_count,
            frame_count,
            site_rgba_f32.shape[0],
            base_owner_i32.shape[0],
            change_frame_i32.shape[0],
            change_owner_i32.shape[0],
        ],
        device=base_offsets_i32.device,
        dtype=torch.int32,
    )
    config_f32 = torch.tensor(
        [config.far, config.transmittance_threshold],
        device=base_offsets_i32.device,
        dtype=torch.float32,
    )
    return config_i32, config_f32


def endpoint_delta_replace_rgba_depth_replay(
    base_offsets_i32: Tensor,
    base_owner_i32: Tensor,
    base_start_f32: Tensor,
    base_end_f32: Tensor,
    track_change_offsets_i32: Tensor,
    change_frame_i32: Tensor,
    change_offsets_i32: Tensor,
    change_owner_i32: Tensor,
    change_start_f32: Tensor,
    change_end_f32: Tensor,
    site_rgba_f32: Tensor,
    config: RealRayReplayConfig,
    *,
    track_count: int,
    frame_count: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Replay endpoint rows from first-frame rows plus changed replacement rows."""
    base_offsets_i32 = base_offsets_i32.contiguous()
    base_owner_i32 = base_owner_i32.contiguous()
    base_start_f32 = base_start_f32.contiguous()
    base_end_f32 = base_end_f32.contiguous()
    track_change_offsets_i32 = track_change_offsets_i32.contiguous()
    change_frame_i32 = change_frame_i32.contiguous()
    change_offsets_i32 = change_offsets_i32.contiguous()
    change_owner_i32 = change_owner_i32.contiguous()
    change_start_f32 = change_start_f32.contiguous()
    change_end_f32 = change_end_f32.contiguous()
    site_rgba_f32 = site_rgba_f32.contiguous()
    config_i32, config_f32 = _endpoint_delta_replace_config_tensors(
        base_offsets_i32=base_offsets_i32,
        base_owner_i32=base_owner_i32,
        base_start_f32=base_start_f32,
        base_end_f32=base_end_f32,
        track_change_offsets_i32=track_change_offsets_i32,
        change_frame_i32=change_frame_i32,
        change_offsets_i32=change_offsets_i32,
        change_owner_i32=change_owner_i32,
        change_start_f32=change_start_f32,
        change_end_f32=change_end_f32,
        site_rgba_f32=site_rgba_f32,
        config=config,
        track_count=track_count,
        frame_count=frame_count,
        max_runs_per_sample=None,
    )
    ops = torch.ops.world_foam_lane2_fused_slab_v0
    op_name = "endpoint_delta_replace_rgba_depth_replay"
    if not hasattr(ops, op_name):
        raise RuntimeError(f"world_foam_lane2_fused_slab_v0 {op_name} op not found. Build this variant first.")
    return getattr(ops, op_name)(
        base_offsets_i32,
        base_owner_i32,
        base_start_f32,
        base_end_f32,
        track_change_offsets_i32,
        change_frame_i32,
        change_offsets_i32,
        change_owner_i32,
        change_start_f32,
        change_end_f32,
        site_rgba_f32,
        config_i32,
        config_f32,
    )


def endpoint_delta_replace_vjp_direct_atomic_grad_only(
    base_offsets_i32: Tensor,
    base_owner_i32: Tensor,
    base_start_f32: Tensor,
    base_end_f32: Tensor,
    track_change_offsets_i32: Tensor,
    change_frame_i32: Tensor,
    change_offsets_i32: Tensor,
    change_owner_i32: Tensor,
    change_start_f32: Tensor,
    change_end_f32: Tensor,
    site_rgba_f32: Tensor,
    grad_rgb_f32: Tensor,
    grad_alpha_f32: Tensor,
    grad_depth_f32: Tensor,
    config: RealRayReplayConfig,
    *,
    track_count: int,
    frame_count: int,
) -> Tensor:
    """Accumulate site RGBA gradients for endpoint replacement-delta replay."""
    base_offsets_i32 = base_offsets_i32.contiguous()
    base_owner_i32 = base_owner_i32.contiguous()
    base_start_f32 = base_start_f32.contiguous()
    base_end_f32 = base_end_f32.contiguous()
    track_change_offsets_i32 = track_change_offsets_i32.contiguous()
    change_frame_i32 = change_frame_i32.contiguous()
    change_offsets_i32 = change_offsets_i32.contiguous()
    change_owner_i32 = change_owner_i32.contiguous()
    change_start_f32 = change_start_f32.contiguous()
    change_end_f32 = change_end_f32.contiguous()
    site_rgba_f32 = site_rgba_f32.contiguous()
    grad_rgb_f32 = grad_rgb_f32.contiguous()
    grad_alpha_f32 = grad_alpha_f32.contiguous()
    grad_depth_f32 = grad_depth_f32.contiguous()
    config_i32, config_f32 = _endpoint_delta_replace_config_tensors(
        base_offsets_i32=base_offsets_i32,
        base_owner_i32=base_owner_i32,
        base_start_f32=base_start_f32,
        base_end_f32=base_end_f32,
        track_change_offsets_i32=track_change_offsets_i32,
        change_frame_i32=change_frame_i32,
        change_offsets_i32=change_offsets_i32,
        change_owner_i32=change_owner_i32,
        change_start_f32=change_start_f32,
        change_end_f32=change_end_f32,
        site_rgba_f32=site_rgba_f32,
        config=config,
        track_count=track_count,
        frame_count=frame_count,
        max_runs_per_sample=129,
    )
    if grad_rgb_f32.device.type != "mps" or grad_rgb_f32.dtype != torch.float32:
        raise ValueError("grad_rgb_f32 must be float32 on MPS")
    if grad_rgb_f32.shape != (track_count, frame_count, 3):
        raise ValueError("grad_rgb_f32 must have shape [track_count, frame_count, 3]")
    if grad_alpha_f32.device.type != "mps" or grad_alpha_f32.dtype != torch.float32:
        raise ValueError("grad_alpha_f32 must be float32 on MPS")
    if grad_alpha_f32.shape != (track_count, frame_count):
        raise ValueError("grad_alpha_f32 must have shape [track_count, frame_count]")
    if grad_depth_f32.device.type != "mps" or grad_depth_f32.dtype != torch.float32:
        raise ValueError("grad_depth_f32 must be float32 on MPS")
    if grad_depth_f32.shape != (track_count, frame_count):
        raise ValueError("grad_depth_f32 must have shape [track_count, frame_count]")
    ops = torch.ops.world_foam_lane2_fused_slab_v0
    op_name = "endpoint_delta_replace_vjp_direct_atomic_grad_only"
    if not hasattr(ops, op_name):
        raise RuntimeError(f"world_foam_lane2_fused_slab_v0 {op_name} op not found. Build this variant first.")
    return getattr(ops, op_name)(
        base_offsets_i32,
        base_owner_i32,
        base_start_f32,
        base_end_f32,
        track_change_offsets_i32,
        change_frame_i32,
        change_offsets_i32,
        change_owner_i32,
        change_start_f32,
        change_end_f32,
        site_rgba_f32,
        grad_rgb_f32,
        grad_alpha_f32,
        grad_depth_f32,
        config_i32,
        config_f32,
    )


def _endpoint_record_delta_config_tensors(
    *,
    boundary_f32: Tensor,
    rays_f32: Tensor,
    frame_t_f32: Tensor,
    base_offsets_i32: Tensor,
    base_owner_i32: Tensor,
    base_left_i32: Tensor,
    base_right_i32: Tensor,
    track_change_offsets_i32: Tensor,
    change_frame_i32: Tensor,
    change_offsets_i32: Tensor,
    change_owner_i32: Tensor,
    change_left_i32: Tensor,
    change_right_i32: Tensor,
    site_rgba_f32: Tensor,
    config: RealRayReplayConfig,
    track_count: int,
    frame_count: int,
    max_runs_per_sample: int | None,
) -> tuple[Tensor, Tensor]:
    _require_mps_tensor(boundary_f32, name="boundary_f32", dtype=torch.float32, cols=5)
    _require_mps_tensor(frame_t_f32, name="frame_t_f32", dtype=torch.float32, cols=None)
    _require_mps_tensor(base_offsets_i32, name="base_offsets_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(base_owner_i32, name="base_owner_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(base_left_i32, name="base_left_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(base_right_i32, name="base_right_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(track_change_offsets_i32, name="track_change_offsets_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(change_frame_i32, name="change_frame_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(change_offsets_i32, name="change_offsets_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(change_owner_i32, name="change_owner_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(change_left_i32, name="change_left_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(change_right_i32, name="change_right_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(site_rgba_f32, name="site_rgba_f32", dtype=torch.float32, cols=4)
    if rays_f32.device.type != "mps" or rays_f32.dtype != torch.float32:
        raise ValueError("rays_f32 must be float32 on MPS")
    if rays_f32.shape != (track_count, frame_count, 6):
        raise ValueError("rays_f32 must have shape [track_count, frame_count, 6]")
    if not rays_f32.is_contiguous():
        raise ValueError("rays_f32 must be contiguous")
    if frame_t_f32.shape[0] != frame_count:
        raise ValueError("frame_t_f32 must have shape [frame_count]")
    if base_left_i32.shape != base_owner_i32.shape or base_right_i32.shape != base_owner_i32.shape:
        raise ValueError("base owner, left, and right arrays must have matching lengths")
    if change_left_i32.shape != change_owner_i32.shape or change_right_i32.shape != change_owner_i32.shape:
        raise ValueError("change owner, left, and right arrays must have matching lengths")
    if track_count <= 0:
        raise ValueError("track_count must be positive")
    if frame_count <= 0:
        raise ValueError("frame_count must be positive")
    if boundary_f32.shape[0] <= 0:
        raise ValueError("boundary_f32 must contain at least one boundary")
    if boundary_f32.shape[0] > 32767:
        raise ValueError("endpoint record delta replay supports at most 32767 boundaries")
    if site_rgba_f32.shape[0] <= 0:
        raise ValueError("site_rgba_f32 must contain at least one site")
    if site_rgba_f32.shape[0] > 64:
        raise ValueError("endpoint record delta replay supports at most 64 sites")
    if config.far <= config.near:
        raise ValueError("config.far must be greater than config.near")
    if config.invalid_epsilon <= 0.0:
        raise ValueError("config.invalid_epsilon must be positive")
    if config.transmittance_threshold < 0.0:
        raise ValueError("config.transmittance_threshold must be nonnegative")
    _validate_segment_tape_offsets_cpu(
        base_offsets_i32.detach().cpu(),
        sample_count=track_count,
        segment_count=base_owner_i32.shape[0],
        max_segments_per_sample=max_runs_per_sample,
    )
    _validate_segment_tape_offsets_cpu(
        track_change_offsets_i32.detach().cpu(),
        sample_count=track_count,
        segment_count=change_frame_i32.shape[0],
        max_segments_per_sample=None,
    )
    _validate_segment_tape_offsets_cpu(
        change_offsets_i32.detach().cpu(),
        sample_count=change_frame_i32.shape[0],
        segment_count=change_owner_i32.shape[0],
        max_segments_per_sample=max_runs_per_sample,
    )
    config_i32 = torch.tensor(
        [
            boundary_f32.shape[0],
            track_count,
            frame_count,
            site_rgba_f32.shape[0],
            base_owner_i32.shape[0],
            change_frame_i32.shape[0],
            change_owner_i32.shape[0],
        ],
        device=boundary_f32.device,
        dtype=torch.int32,
    )
    config_f32 = torch.tensor(
        [config.near, config.far, config.invalid_epsilon, config.transmittance_threshold],
        device=boundary_f32.device,
        dtype=torch.float32,
    )
    return config_i32, config_f32


def endpoint_record_delta_replace_rgba_depth_replay(
    boundary_f32: Tensor,
    rays_f32: Tensor,
    frame_t_f32: Tensor,
    base_offsets_i32: Tensor,
    base_owner_i32: Tensor,
    base_left_i32: Tensor,
    base_right_i32: Tensor,
    track_change_offsets_i32: Tensor,
    change_frame_i32: Tensor,
    change_offsets_i32: Tensor,
    change_owner_i32: Tensor,
    change_left_i32: Tensor,
    change_right_i32: Tensor,
    site_rgba_f32: Tensor,
    config: RealRayReplayConfig,
    *,
    track_count: int,
    frame_count: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Replay owner+cut-id endpoint row deltas and recover depths from rays/boundaries."""
    boundary_f32 = boundary_f32.contiguous()
    rays_f32 = rays_f32.contiguous()
    frame_t_f32 = frame_t_f32.contiguous()
    base_offsets_i32 = base_offsets_i32.contiguous()
    base_owner_i32 = base_owner_i32.contiguous()
    base_left_i32 = base_left_i32.contiguous()
    base_right_i32 = base_right_i32.contiguous()
    track_change_offsets_i32 = track_change_offsets_i32.contiguous()
    change_frame_i32 = change_frame_i32.contiguous()
    change_offsets_i32 = change_offsets_i32.contiguous()
    change_owner_i32 = change_owner_i32.contiguous()
    change_left_i32 = change_left_i32.contiguous()
    change_right_i32 = change_right_i32.contiguous()
    site_rgba_f32 = site_rgba_f32.contiguous()
    config_i32, config_f32 = _endpoint_record_delta_config_tensors(
        boundary_f32=boundary_f32,
        rays_f32=rays_f32,
        frame_t_f32=frame_t_f32,
        base_offsets_i32=base_offsets_i32,
        base_owner_i32=base_owner_i32,
        base_left_i32=base_left_i32,
        base_right_i32=base_right_i32,
        track_change_offsets_i32=track_change_offsets_i32,
        change_frame_i32=change_frame_i32,
        change_offsets_i32=change_offsets_i32,
        change_owner_i32=change_owner_i32,
        change_left_i32=change_left_i32,
        change_right_i32=change_right_i32,
        site_rgba_f32=site_rgba_f32,
        config=config,
        track_count=track_count,
        frame_count=frame_count,
        max_runs_per_sample=None,
    )
    ops = torch.ops.world_foam_lane2_fused_slab_v0
    op_name = "endpoint_record_delta_replace_rgba_depth_replay"
    if not hasattr(ops, op_name):
        raise RuntimeError(f"world_foam_lane2_fused_slab_v0 {op_name} op not found. Build this variant first.")
    return getattr(ops, op_name)(
        boundary_f32,
        rays_f32,
        frame_t_f32,
        base_offsets_i32,
        base_owner_i32,
        base_left_i32,
        base_right_i32,
        track_change_offsets_i32,
        change_frame_i32,
        change_offsets_i32,
        change_owner_i32,
        change_left_i32,
        change_right_i32,
        site_rgba_f32,
        config_i32,
        config_f32,
    )


def endpoint_record_delta_replace_vjp_direct_atomic_grad_only(
    boundary_f32: Tensor,
    rays_f32: Tensor,
    frame_t_f32: Tensor,
    base_offsets_i32: Tensor,
    base_owner_i32: Tensor,
    base_left_i32: Tensor,
    base_right_i32: Tensor,
    track_change_offsets_i32: Tensor,
    change_frame_i32: Tensor,
    change_offsets_i32: Tensor,
    change_owner_i32: Tensor,
    change_left_i32: Tensor,
    change_right_i32: Tensor,
    site_rgba_f32: Tensor,
    grad_rgb_f32: Tensor,
    grad_alpha_f32: Tensor,
    grad_depth_f32: Tensor,
    config: RealRayReplayConfig,
    *,
    track_count: int,
    frame_count: int,
) -> Tensor:
    """Accumulate site RGBA gradients for owner+cut-id endpoint row deltas."""
    boundary_f32 = boundary_f32.contiguous()
    rays_f32 = rays_f32.contiguous()
    frame_t_f32 = frame_t_f32.contiguous()
    base_offsets_i32 = base_offsets_i32.contiguous()
    base_owner_i32 = base_owner_i32.contiguous()
    base_left_i32 = base_left_i32.contiguous()
    base_right_i32 = base_right_i32.contiguous()
    track_change_offsets_i32 = track_change_offsets_i32.contiguous()
    change_frame_i32 = change_frame_i32.contiguous()
    change_offsets_i32 = change_offsets_i32.contiguous()
    change_owner_i32 = change_owner_i32.contiguous()
    change_left_i32 = change_left_i32.contiguous()
    change_right_i32 = change_right_i32.contiguous()
    site_rgba_f32 = site_rgba_f32.contiguous()
    grad_rgb_f32 = grad_rgb_f32.contiguous()
    grad_alpha_f32 = grad_alpha_f32.contiguous()
    grad_depth_f32 = grad_depth_f32.contiguous()
    config_i32, config_f32 = _endpoint_record_delta_config_tensors(
        boundary_f32=boundary_f32,
        rays_f32=rays_f32,
        frame_t_f32=frame_t_f32,
        base_offsets_i32=base_offsets_i32,
        base_owner_i32=base_owner_i32,
        base_left_i32=base_left_i32,
        base_right_i32=base_right_i32,
        track_change_offsets_i32=track_change_offsets_i32,
        change_frame_i32=change_frame_i32,
        change_offsets_i32=change_offsets_i32,
        change_owner_i32=change_owner_i32,
        change_left_i32=change_left_i32,
        change_right_i32=change_right_i32,
        site_rgba_f32=site_rgba_f32,
        config=config,
        track_count=track_count,
        frame_count=frame_count,
        max_runs_per_sample=129,
    )
    if grad_rgb_f32.device.type != "mps" or grad_rgb_f32.dtype != torch.float32:
        raise ValueError("grad_rgb_f32 must be float32 on MPS")
    if grad_rgb_f32.shape != (track_count, frame_count, 3):
        raise ValueError("grad_rgb_f32 must have shape [track_count, frame_count, 3]")
    if grad_alpha_f32.device.type != "mps" or grad_alpha_f32.dtype != torch.float32:
        raise ValueError("grad_alpha_f32 must be float32 on MPS")
    if grad_alpha_f32.shape != (track_count, frame_count):
        raise ValueError("grad_alpha_f32 must have shape [track_count, frame_count]")
    if grad_depth_f32.device.type != "mps" or grad_depth_f32.dtype != torch.float32:
        raise ValueError("grad_depth_f32 must be float32 on MPS")
    if grad_depth_f32.shape != (track_count, frame_count):
        raise ValueError("grad_depth_f32 must have shape [track_count, frame_count]")
    ops = torch.ops.world_foam_lane2_fused_slab_v0
    op_name = "endpoint_record_delta_replace_vjp_direct_atomic_grad_only"
    if not hasattr(ops, op_name):
        raise RuntimeError(f"world_foam_lane2_fused_slab_v0 {op_name} op not found. Build this variant first.")
    return getattr(ops, op_name)(
        boundary_f32,
        rays_f32,
        frame_t_f32,
        base_offsets_i32,
        base_owner_i32,
        base_left_i32,
        base_right_i32,
        track_change_offsets_i32,
        change_frame_i32,
        change_offsets_i32,
        change_owner_i32,
        change_left_i32,
        change_right_i32,
        site_rgba_f32,
        grad_rgb_f32,
        grad_alpha_f32,
        grad_depth_f32,
        config_i32,
        config_f32,
    )


def endpoint_record_delta_replace_coeff16_mse_vjp_direct_atomic_rgb_only(
    coeff_f16: Tensor,
    frame_t_f32: Tensor,
    base_offsets_i32: Tensor,
    base_owner_i32: Tensor,
    base_left_i32: Tensor,
    base_right_i32: Tensor,
    track_change_offsets_i32: Tensor,
    change_frame_i32: Tensor,
    change_offsets_i32: Tensor,
    change_owner_i32: Tensor,
    change_left_i32: Tensor,
    change_right_i32: Tensor,
    site_rgba_f32: Tensor,
    target_rgb_f32: Tensor,
    config: RealRayReplayConfig,
    *,
    track_count: int,
    frame_count: int,
    boundary_count: int,
) -> tuple[Tensor, Tensor]:
    """Compute RGB MSE/VJP for delta-replace endpoint records with coeff16 cut depths."""
    coeff_f16 = coeff_f16.contiguous()
    frame_t_f32 = frame_t_f32.contiguous()
    base_offsets_i32 = base_offsets_i32.contiguous()
    base_owner_i32 = base_owner_i32.contiguous()
    base_left_i32 = base_left_i32.contiguous()
    base_right_i32 = base_right_i32.contiguous()
    track_change_offsets_i32 = track_change_offsets_i32.contiguous()
    change_frame_i32 = change_frame_i32.contiguous()
    change_offsets_i32 = change_offsets_i32.contiguous()
    change_owner_i32 = change_owner_i32.contiguous()
    change_left_i32 = change_left_i32.contiguous()
    change_right_i32 = change_right_i32.contiguous()
    site_rgba_f32 = site_rgba_f32.contiguous()
    target_rgb_f32 = target_rgb_f32.contiguous()
    _require_mps_tensor(coeff_f16, name="coeff_f16", dtype=torch.float16, cols=4)
    _require_mps_tensor(frame_t_f32, name="frame_t_f32", dtype=torch.float32, cols=None)
    _require_mps_tensor(base_offsets_i32, name="base_offsets_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(base_owner_i32, name="base_owner_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(base_left_i32, name="base_left_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(base_right_i32, name="base_right_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(track_change_offsets_i32, name="track_change_offsets_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(change_frame_i32, name="change_frame_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(change_offsets_i32, name="change_offsets_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(change_owner_i32, name="change_owner_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(change_left_i32, name="change_left_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(change_right_i32, name="change_right_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(site_rgba_f32, name="site_rgba_f32", dtype=torch.float32, cols=4)
    if track_count <= 0 or frame_count <= 0 or boundary_count <= 0:
        raise ValueError("track_count, frame_count, and boundary_count must be positive")
    if boundary_count > 32767:
        raise ValueError("endpoint record delta replace coeff16 replay supports boundary count <= 32767")
    if coeff_f16.shape[0] != track_count * boundary_count:
        raise ValueError("coeff_f16 row count must be track_count * boundary_count")
    if frame_t_f32.shape[0] != frame_count:
        raise ValueError("frame_t_f32 must have shape [frame_count]")
    if base_left_i32.shape != base_owner_i32.shape or base_right_i32.shape != base_owner_i32.shape:
        raise ValueError("base owner, left, and right arrays must have matching lengths")
    if change_left_i32.shape != change_owner_i32.shape or change_right_i32.shape != change_owner_i32.shape:
        raise ValueError("change owner, left, and right arrays must have matching lengths")
    if target_rgb_f32.device.type != "mps" or target_rgb_f32.dtype != torch.float32:
        raise ValueError("target_rgb_f32 must be float32 on MPS")
    if target_rgb_f32.shape != (track_count, frame_count, 3):
        raise ValueError("target_rgb_f32 must have shape [track_count, frame_count, 3]")
    if site_rgba_f32.shape[0] <= 0 or site_rgba_f32.shape[0] > 64:
        raise ValueError("endpoint record delta replace coeff16 replay supports site count in [1, 64]")
    if config.far <= config.near:
        raise ValueError("config.far must be greater than config.near")
    if config.invalid_epsilon <= 0.0:
        raise ValueError("config.invalid_epsilon must be positive")
    if config.transmittance_threshold < 0.0:
        raise ValueError("config.transmittance_threshold must be nonnegative")
    _validate_segment_tape_offsets_cpu(
        base_offsets_i32.detach().cpu(),
        sample_count=track_count,
        segment_count=base_owner_i32.shape[0],
        max_segments_per_sample=129,
    )
    _validate_segment_tape_offsets_cpu(
        track_change_offsets_i32.detach().cpu(),
        sample_count=track_count,
        segment_count=change_frame_i32.shape[0],
        max_segments_per_sample=None,
    )
    _validate_segment_tape_offsets_cpu(
        change_offsets_i32.detach().cpu(),
        sample_count=change_frame_i32.shape[0],
        segment_count=change_owner_i32.shape[0],
        max_segments_per_sample=129,
    )
    config_i32 = torch.tensor(
        [
            boundary_count,
            track_count,
            frame_count,
            site_rgba_f32.shape[0],
            base_owner_i32.shape[0],
            change_frame_i32.shape[0],
            change_owner_i32.shape[0],
        ],
        device=coeff_f16.device,
        dtype=torch.int32,
    )
    config_f32 = torch.tensor(
        [config.near, config.far, config.invalid_epsilon, config.transmittance_threshold],
        device=coeff_f16.device,
        dtype=torch.float32,
    )
    ops = torch.ops.world_foam_lane2_fused_slab_v0
    op_name = "endpoint_record_delta_replace_coeff16_mse_vjp_direct_atomic_rgb_only"
    if not hasattr(ops, op_name):
        raise RuntimeError(f"world_foam_lane2_fused_slab_v0 {op_name} op not found. Build this variant first.")
    return getattr(ops, op_name)(
        coeff_f16,
        frame_t_f32,
        base_offsets_i32,
        base_owner_i32,
        base_left_i32,
        base_right_i32,
        track_change_offsets_i32,
        change_frame_i32,
        change_offsets_i32,
        change_owner_i32,
        change_left_i32,
        change_right_i32,
        site_rgba_f32,
        target_rgb_f32,
        config_i32,
        config_f32,
    )


def endpoint_record_delta_replace_coeff16_i16x3_mse_vjp_direct_atomic_rgb_only(
    coeff_f16: Tensor,
    frame_t_f32: Tensor,
    base_offsets_i32: Tensor,
    base_record_i16: Tensor,
    track_change_offsets_i32: Tensor,
    change_frame_i32: Tensor,
    change_offsets_i32: Tensor,
    change_record_i16: Tensor,
    site_rgba_f32: Tensor,
    target_rgb_f32: Tensor,
    config: RealRayReplayConfig,
    *,
    track_count: int,
    frame_count: int,
    boundary_count: int,
) -> tuple[Tensor, Tensor]:
    """Compute RGB MSE/VJP for delta-replace endpoint records with coeff16 and i16x3 records."""
    coeff_f16 = coeff_f16.contiguous()
    frame_t_f32 = frame_t_f32.contiguous()
    base_offsets_i32 = base_offsets_i32.contiguous()
    base_record_i16 = base_record_i16.contiguous()
    track_change_offsets_i32 = track_change_offsets_i32.contiguous()
    change_frame_i32 = change_frame_i32.contiguous()
    change_offsets_i32 = change_offsets_i32.contiguous()
    change_record_i16 = change_record_i16.contiguous()
    site_rgba_f32 = site_rgba_f32.contiguous()
    target_rgb_f32 = target_rgb_f32.contiguous()
    _require_mps_tensor(coeff_f16, name="coeff_f16", dtype=torch.float16, cols=4)
    _require_mps_tensor(frame_t_f32, name="frame_t_f32", dtype=torch.float32, cols=None)
    _require_mps_tensor(base_offsets_i32, name="base_offsets_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(base_record_i16, name="base_record_i16", dtype=torch.int16, cols=None)
    _require_mps_tensor(track_change_offsets_i32, name="track_change_offsets_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(change_frame_i32, name="change_frame_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(change_offsets_i32, name="change_offsets_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(change_record_i16, name="change_record_i16", dtype=torch.int16, cols=None)
    _require_mps_tensor(site_rgba_f32, name="site_rgba_f32", dtype=torch.float32, cols=4)
    if track_count <= 0 or frame_count <= 0 or boundary_count <= 0:
        raise ValueError("track_count, frame_count, and boundary_count must be positive")
    if boundary_count > 32765:
        raise ValueError("endpoint record delta replace coeff16 i16x3 replay supports boundary count <= 32765")
    if coeff_f16.shape[0] != track_count * boundary_count:
        raise ValueError("coeff_f16 row count must be track_count * boundary_count")
    if frame_t_f32.shape[0] != frame_count:
        raise ValueError("frame_t_f32 must have shape [frame_count]")
    if base_record_i16.numel() % 3 != 0 or change_record_i16.numel() % 3 != 0:
        raise ValueError("i16x3 record tensors must have length divisible by 3")
    if target_rgb_f32.device.type != "mps" or target_rgb_f32.dtype != torch.float32:
        raise ValueError("target_rgb_f32 must be float32 on MPS")
    if target_rgb_f32.shape != (track_count, frame_count, 3):
        raise ValueError("target_rgb_f32 must have shape [track_count, frame_count, 3]")
    if site_rgba_f32.shape[0] <= 0 or site_rgba_f32.shape[0] > 32767:
        raise ValueError("endpoint record delta replace coeff16 i16x3 replay supports site count in [1, 32767]")
    if config.far <= config.near:
        raise ValueError("config.far must be greater than config.near")
    if config.invalid_epsilon <= 0.0:
        raise ValueError("config.invalid_epsilon must be positive")
    if config.transmittance_threshold < 0.0:
        raise ValueError("config.transmittance_threshold must be nonnegative")
    base_record_count = base_record_i16.numel() // 3
    change_record_count = change_record_i16.numel() // 3
    _validate_segment_tape_offsets_cpu(
        base_offsets_i32.detach().cpu(),
        sample_count=track_count,
        segment_count=base_record_count,
        max_segments_per_sample=129,
    )
    _validate_segment_tape_offsets_cpu(
        track_change_offsets_i32.detach().cpu(),
        sample_count=track_count,
        segment_count=change_frame_i32.shape[0],
        max_segments_per_sample=None,
    )
    _validate_segment_tape_offsets_cpu(
        change_offsets_i32.detach().cpu(),
        sample_count=change_frame_i32.shape[0],
        segment_count=change_record_count,
        max_segments_per_sample=129,
    )
    config_i32 = torch.tensor(
        [
            boundary_count,
            track_count,
            frame_count,
            site_rgba_f32.shape[0],
            base_record_count,
            change_frame_i32.shape[0],
            change_record_count,
        ],
        device=coeff_f16.device,
        dtype=torch.int32,
    )
    config_f32 = torch.tensor(
        [config.near, config.far, config.invalid_epsilon, config.transmittance_threshold],
        device=coeff_f16.device,
        dtype=torch.float32,
    )
    ops = torch.ops.world_foam_lane2_fused_slab_v0
    op_name = "endpoint_record_delta_replace_coeff16_i16x3_mse_vjp_direct_atomic_rgb_only"
    if not hasattr(ops, op_name):
        raise RuntimeError(f"world_foam_lane2_fused_slab_v0 {op_name} op not found. Build this variant first.")
    return getattr(ops, op_name)(
        coeff_f16,
        frame_t_f32,
        base_offsets_i32,
        base_record_i16,
        track_change_offsets_i32,
        change_frame_i32,
        change_offsets_i32,
        change_record_i16,
        site_rgba_f32,
        target_rgb_f32,
        config_i32,
        config_f32,
    )


def endpoint_record_delta_replace_coeff16_i16x3_framegroup16_mse_vjp_direct_atomic_rgb_only(
    coeff_f16: Tensor,
    frame_t_f32: Tensor,
    base_offsets_i32: Tensor,
    base_record_i16: Tensor,
    track_change_offsets_i32: Tensor,
    track_chunk_change_offsets_i16: Tensor,
    change_frame_i32: Tensor,
    change_offsets_i32: Tensor,
    change_record_i16: Tensor,
    site_rgba_f32: Tensor,
    target_rgb_f32: Tensor,
    config: RealRayReplayConfig,
    *,
    track_count: int,
    frame_count: int,
    boundary_count: int,
) -> tuple[Tensor, Tensor]:
    """Compute RGB MSE/VJP for delta-replace records with 16-frame threadgroup chunks."""
    coeff_f16 = coeff_f16.contiguous()
    frame_t_f32 = frame_t_f32.contiguous()
    base_offsets_i32 = base_offsets_i32.contiguous()
    base_record_i16 = base_record_i16.contiguous()
    track_change_offsets_i32 = track_change_offsets_i32.contiguous()
    track_chunk_change_offsets_i16 = track_chunk_change_offsets_i16.contiguous()
    change_frame_i32 = change_frame_i32.contiguous()
    change_offsets_i32 = change_offsets_i32.contiguous()
    change_record_i16 = change_record_i16.contiguous()
    site_rgba_f32 = site_rgba_f32.contiguous()
    target_rgb_f32 = target_rgb_f32.contiguous()
    _require_mps_tensor(coeff_f16, name="coeff_f16", dtype=torch.float16, cols=4)
    _require_mps_tensor(frame_t_f32, name="frame_t_f32", dtype=torch.float32, cols=None)
    _require_mps_tensor(base_offsets_i32, name="base_offsets_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(base_record_i16, name="base_record_i16", dtype=torch.int16, cols=None)
    _require_mps_tensor(track_change_offsets_i32, name="track_change_offsets_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(
        track_chunk_change_offsets_i16,
        name="track_chunk_change_offsets_i16",
        dtype=torch.int16,
        cols=None,
    )
    _require_mps_tensor(change_frame_i32, name="change_frame_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(change_offsets_i32, name="change_offsets_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(change_record_i16, name="change_record_i16", dtype=torch.int16, cols=None)
    _require_mps_tensor(site_rgba_f32, name="site_rgba_f32", dtype=torch.float32, cols=4)
    if track_count <= 0 or frame_count <= 0 or boundary_count <= 0:
        raise ValueError("track_count, frame_count, and boundary_count must be positive")
    if boundary_count > 32765:
        raise ValueError(
            "endpoint record delta replace coeff16 i16x3 framegroup16 replay supports boundary count <= 32765"
        )
    if coeff_f16.shape[0] != track_count * boundary_count:
        raise ValueError("coeff_f16 row count must be track_count * boundary_count")
    if frame_t_f32.shape[0] != frame_count:
        raise ValueError("frame_t_f32 must have shape [frame_count]")
    if base_record_i16.numel() % 3 != 0 or change_record_i16.numel() % 3 != 0:
        raise ValueError("i16x3 record tensors must have length divisible by 3")
    if target_rgb_f32.device.type != "mps" or target_rgb_f32.dtype != torch.float32:
        raise ValueError("target_rgb_f32 must be float32 on MPS")
    if target_rgb_f32.shape != (track_count, frame_count, 3):
        raise ValueError("target_rgb_f32 must have shape [track_count, frame_count, 3]")
    if site_rgba_f32.shape[0] <= 0 or site_rgba_f32.shape[0] > 32767:
        raise ValueError(
            "endpoint record delta replace coeff16 i16x3 framegroup16 replay supports site count in [1, 32767]"
        )
    if config.far <= config.near:
        raise ValueError("config.far must be greater than config.near")
    if config.invalid_epsilon <= 0.0:
        raise ValueError("config.invalid_epsilon must be positive")
    if config.transmittance_threshold < 0.0:
        raise ValueError("config.transmittance_threshold must be nonnegative")
    base_record_count = base_record_i16.numel() // 3
    change_record_count = change_record_i16.numel() // 3
    chunk_count = (frame_count + 31) // 32
    if change_frame_i32.shape[0] > 32767:
        raise ValueError("framegroup16 chunk-start offsets use int16 and require change count <= 32767")
    _validate_segment_tape_offsets_cpu(
        base_offsets_i32.detach().cpu(),
        sample_count=track_count,
        segment_count=base_record_count,
        max_segments_per_sample=129,
    )
    _validate_segment_tape_offsets_cpu(
        track_change_offsets_i32.detach().cpu(),
        sample_count=track_count,
        segment_count=change_frame_i32.shape[0],
        max_segments_per_sample=None,
    )
    _validate_segment_tape_offsets_cpu(
        track_chunk_change_offsets_i16.detach().cpu(),
        sample_count=track_count * (chunk_count + 1) - 1,
        segment_count=change_frame_i32.shape[0],
        max_segments_per_sample=None,
    )
    _validate_segment_tape_offsets_cpu(
        change_offsets_i32.detach().cpu(),
        sample_count=change_frame_i32.shape[0],
        segment_count=change_record_count,
        max_segments_per_sample=129,
    )
    config_i32 = torch.tensor(
        [
            boundary_count,
            track_count,
            frame_count,
            site_rgba_f32.shape[0],
            base_record_count,
            change_frame_i32.shape[0],
            change_record_count,
        ],
        device=coeff_f16.device,
        dtype=torch.int32,
    )
    config_f32 = torch.tensor(
        [config.near, config.far, config.invalid_epsilon, config.transmittance_threshold],
        device=coeff_f16.device,
        dtype=torch.float32,
    )
    ops = torch.ops.world_foam_lane2_fused_slab_v0
    op_name = "endpoint_record_delta_replace_coeff16_i16x3_framegroup16_mse_vjp_direct_atomic_rgb_only"
    if not hasattr(ops, op_name):
        raise RuntimeError(f"world_foam_lane2_fused_slab_v0 {op_name} op not found. Build this variant first.")
    return getattr(ops, op_name)(
        coeff_f16,
        frame_t_f32,
        base_offsets_i32,
        base_record_i16,
        track_change_offsets_i32,
        track_chunk_change_offsets_i16,
        change_frame_i32,
        change_offsets_i32,
        change_record_i16,
        site_rgba_f32,
        target_rgb_f32,
        config_i32,
        config_f32,
    )


def endpoint_record_delta_replace_coeff16_i16x3_framegroup16_ownerreduce_mse_vjp_direct_atomic_rgb_only(
    coeff_f16: Tensor,
    frame_t_f32: Tensor,
    base_offsets_i32: Tensor,
    base_record_i16: Tensor,
    track_change_offsets_i32: Tensor,
    track_chunk_change_offsets_i16: Tensor,
    track_chunk_owner_offsets_i32: Tensor,
    track_chunk_owner_i16: Tensor,
    change_frame_i32: Tensor,
    change_offsets_i32: Tensor,
    change_record_i16: Tensor,
    site_rgba_f32: Tensor,
    target_rgb_f32: Tensor,
    config: RealRayReplayConfig,
    *,
    track_count: int,
    frame_count: int,
    boundary_count: int,
) -> tuple[Tensor, Tensor]:
    """Compute RGB MSE/VJP with a sidecar owner-list framegroup reduction."""
    coeff_f16 = coeff_f16.contiguous()
    frame_t_f32 = frame_t_f32.contiguous()
    base_offsets_i32 = base_offsets_i32.contiguous()
    base_record_i16 = base_record_i16.contiguous()
    track_change_offsets_i32 = track_change_offsets_i32.contiguous()
    track_chunk_change_offsets_i16 = track_chunk_change_offsets_i16.contiguous()
    track_chunk_owner_offsets_i32 = track_chunk_owner_offsets_i32.contiguous()
    track_chunk_owner_i16 = track_chunk_owner_i16.contiguous()
    change_frame_i32 = change_frame_i32.contiguous()
    change_offsets_i32 = change_offsets_i32.contiguous()
    change_record_i16 = change_record_i16.contiguous()
    site_rgba_f32 = site_rgba_f32.contiguous()
    target_rgb_f32 = target_rgb_f32.contiguous()
    _require_mps_tensor(coeff_f16, name="coeff_f16", dtype=torch.float16, cols=4)
    _require_mps_tensor(frame_t_f32, name="frame_t_f32", dtype=torch.float32, cols=None)
    _require_mps_tensor(base_offsets_i32, name="base_offsets_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(base_record_i16, name="base_record_i16", dtype=torch.int16, cols=None)
    _require_mps_tensor(track_change_offsets_i32, name="track_change_offsets_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(
        track_chunk_change_offsets_i16,
        name="track_chunk_change_offsets_i16",
        dtype=torch.int16,
        cols=None,
    )
    _require_mps_tensor(
        track_chunk_owner_offsets_i32,
        name="track_chunk_owner_offsets_i32",
        dtype=torch.int32,
        cols=None,
    )
    _require_mps_tensor(track_chunk_owner_i16, name="track_chunk_owner_i16", dtype=torch.int16, cols=None)
    _require_mps_tensor(change_frame_i32, name="change_frame_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(change_offsets_i32, name="change_offsets_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(change_record_i16, name="change_record_i16", dtype=torch.int16, cols=None)
    _require_mps_tensor(site_rgba_f32, name="site_rgba_f32", dtype=torch.float32, cols=4)
    if track_count <= 0 or frame_count <= 0 or boundary_count <= 0:
        raise ValueError("track_count, frame_count, and boundary_count must be positive")
    if boundary_count > 32765:
        raise ValueError(
            "endpoint record delta replace coeff16 i16x3 ownerreduce framegroup16 replay supports boundary count <= 32765"
        )
    if coeff_f16.shape[0] != track_count * boundary_count:
        raise ValueError("coeff_f16 row count must be track_count * boundary_count")
    if frame_t_f32.shape[0] != frame_count:
        raise ValueError("frame_t_f32 must have shape [frame_count]")
    if base_record_i16.numel() % 3 != 0 or change_record_i16.numel() % 3 != 0:
        raise ValueError("i16x3 record tensors must have length divisible by 3")
    if target_rgb_f32.device.type != "mps" or target_rgb_f32.dtype != torch.float32:
        raise ValueError("target_rgb_f32 must be float32 on MPS")
    if target_rgb_f32.shape != (track_count, frame_count, 3):
        raise ValueError("target_rgb_f32 must have shape [track_count, frame_count, 3]")
    if site_rgba_f32.shape[0] <= 0 or site_rgba_f32.shape[0] > 32767:
        raise ValueError("ownerreduce framegroup16 replay supports site count in [1, 32767]")
    if config.far <= config.near:
        raise ValueError("config.far must be greater than config.near")
    if config.invalid_epsilon <= 0.0:
        raise ValueError("config.invalid_epsilon must be positive")
    if config.transmittance_threshold < 0.0:
        raise ValueError("config.transmittance_threshold must be nonnegative")
    base_record_count = base_record_i16.numel() // 3
    change_record_count = change_record_i16.numel() // 3
    owner_list_count = int(track_chunk_owner_i16.shape[0])
    chunk_count = (frame_count + 31) // 32
    if change_frame_i32.shape[0] > 32767:
        raise ValueError("framegroup16 chunk-start offsets use int16 and require change count <= 32767")
    _validate_segment_tape_offsets_cpu(
        base_offsets_i32.detach().cpu(),
        sample_count=track_count,
        segment_count=base_record_count,
        max_segments_per_sample=129,
    )
    _validate_segment_tape_offsets_cpu(
        track_change_offsets_i32.detach().cpu(),
        sample_count=track_count,
        segment_count=change_frame_i32.shape[0],
        max_segments_per_sample=None,
    )
    _validate_segment_tape_offsets_cpu(
        track_chunk_change_offsets_i16.detach().cpu(),
        sample_count=track_count * (chunk_count + 1) - 1,
        segment_count=change_frame_i32.shape[0],
        max_segments_per_sample=None,
    )
    _validate_segment_tape_offsets_cpu(
        track_chunk_owner_offsets_i32.detach().cpu(),
        sample_count=track_count * chunk_count,
        segment_count=owner_list_count,
        max_segments_per_sample=None,
    )
    owner_ids_cpu = track_chunk_owner_i16.detach().cpu()
    if owner_ids_cpu.numel() > 0 and (
        int(owner_ids_cpu.min().item()) < 0 or int(owner_ids_cpu.max().item()) >= int(site_rgba_f32.shape[0])
    ):
        raise ValueError("track_chunk_owner_i16 ids must be in [0, site_count)")
    _validate_segment_tape_offsets_cpu(
        change_offsets_i32.detach().cpu(),
        sample_count=change_frame_i32.shape[0],
        segment_count=change_record_count,
        max_segments_per_sample=129,
    )
    config_i32 = torch.tensor(
        [
            boundary_count,
            track_count,
            frame_count,
            site_rgba_f32.shape[0],
            base_record_count,
            change_frame_i32.shape[0],
            change_record_count,
            owner_list_count,
        ],
        device=coeff_f16.device,
        dtype=torch.int32,
    )
    config_f32 = torch.tensor(
        [config.near, config.far, config.invalid_epsilon, config.transmittance_threshold],
        device=coeff_f16.device,
        dtype=torch.float32,
    )
    ops = torch.ops.world_foam_lane2_fused_slab_v0
    op_name = "endpoint_record_delta_replace_coeff16_i16x3_framegroup16_ownerreduce_mse_vjp_direct_atomic_rgb_only"
    if not hasattr(ops, op_name):
        raise RuntimeError(f"world_foam_lane2_fused_slab_v0 {op_name} op not found. Build this variant first.")
    return getattr(ops, op_name)(
        coeff_f16,
        frame_t_f32,
        base_offsets_i32,
        base_record_i16,
        track_change_offsets_i32,
        track_chunk_change_offsets_i16,
        track_chunk_owner_offsets_i32,
        track_chunk_owner_i16,
        change_frame_i32,
        change_offsets_i32,
        change_record_i16,
        site_rgba_f32,
        target_rgb_f32,
        config_i32,
        config_f32,
    )


def endpoint_record_delta_replace_coeff16_packed_framegroup16_mse_vjp_direct_atomic_rgb_only(
    coeff_f16: Tensor,
    frame_t_f32: Tensor,
    base_offsets_i32: Tensor,
    base_record_i32: Tensor,
    track_change_offsets_i32: Tensor,
    track_chunk_change_offsets_i16: Tensor,
    change_frame_i32: Tensor,
    change_offsets_i32: Tensor,
    change_record_i32: Tensor,
    site_rgba_f32: Tensor,
    target_rgb_f32: Tensor,
    config: RealRayReplayConfig,
    *,
    track_count: int,
    frame_count: int,
    boundary_count: int,
) -> tuple[Tensor, Tensor]:
    """Compute RGB MSE/VJP for delta-replace records with packed int32 rows and 32-frame chunks."""
    coeff_f16 = coeff_f16.contiguous()
    frame_t_f32 = frame_t_f32.contiguous()
    base_offsets_i32 = base_offsets_i32.contiguous()
    base_record_i32 = base_record_i32.contiguous()
    track_change_offsets_i32 = track_change_offsets_i32.contiguous()
    track_chunk_change_offsets_i16 = track_chunk_change_offsets_i16.contiguous()
    change_frame_i32 = change_frame_i32.contiguous()
    change_offsets_i32 = change_offsets_i32.contiguous()
    change_record_i32 = change_record_i32.contiguous()
    site_rgba_f32 = site_rgba_f32.contiguous()
    target_rgb_f32 = target_rgb_f32.contiguous()
    _require_mps_tensor(coeff_f16, name="coeff_f16", dtype=torch.float16, cols=4)
    _require_mps_tensor(frame_t_f32, name="frame_t_f32", dtype=torch.float32, cols=None)
    _require_mps_tensor(base_offsets_i32, name="base_offsets_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(base_record_i32, name="base_record_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(track_change_offsets_i32, name="track_change_offsets_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(
        track_chunk_change_offsets_i16,
        name="track_chunk_change_offsets_i16",
        dtype=torch.int16,
        cols=None,
    )
    _require_mps_tensor(change_frame_i32, name="change_frame_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(change_offsets_i32, name="change_offsets_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(change_record_i32, name="change_record_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(site_rgba_f32, name="site_rgba_f32", dtype=torch.float32, cols=4)
    if track_count <= 0 or frame_count <= 0 or boundary_count <= 0:
        raise ValueError("track_count, frame_count, and boundary_count must be positive")
    if boundary_count > 4093:
        raise ValueError("packed framegroup16 delta replace coeff16 replay supports boundary count <= 4093")
    if coeff_f16.shape[0] != track_count * boundary_count:
        raise ValueError("coeff_f16 row count must be track_count * boundary_count")
    if frame_t_f32.shape[0] != frame_count:
        raise ValueError("frame_t_f32 must have shape [frame_count]")
    if target_rgb_f32.device.type != "mps" or target_rgb_f32.dtype != torch.float32:
        raise ValueError("target_rgb_f32 must be float32 on MPS")
    if target_rgb_f32.shape != (track_count, frame_count, 3):
        raise ValueError("target_rgb_f32 must have shape [track_count, frame_count, 3]")
    if site_rgba_f32.shape[0] <= 0 or site_rgba_f32.shape[0] > 256:
        raise ValueError("packed framegroup16 delta replace coeff16 replay supports site count in [1, 256]")
    if config.far <= config.near:
        raise ValueError("config.far must be greater than config.near")
    if config.invalid_epsilon <= 0.0:
        raise ValueError("config.invalid_epsilon must be positive")
    if config.transmittance_threshold < 0.0:
        raise ValueError("config.transmittance_threshold must be nonnegative")
    base_record_count = base_record_i32.numel()
    change_record_count = change_record_i32.numel()
    chunk_count = (frame_count + 31) // 32
    if change_frame_i32.shape[0] > 32767:
        raise ValueError("framegroup16 chunk-start offsets use int16 and require change count <= 32767")
    _validate_segment_tape_offsets_cpu(
        base_offsets_i32.detach().cpu(),
        sample_count=track_count,
        segment_count=base_record_count,
        max_segments_per_sample=129,
    )
    _validate_segment_tape_offsets_cpu(
        track_change_offsets_i32.detach().cpu(),
        sample_count=track_count,
        segment_count=change_frame_i32.shape[0],
        max_segments_per_sample=None,
    )
    _validate_segment_tape_offsets_cpu(
        track_chunk_change_offsets_i16.detach().cpu(),
        sample_count=track_count * (chunk_count + 1) - 1,
        segment_count=change_frame_i32.shape[0],
        max_segments_per_sample=None,
    )
    _validate_segment_tape_offsets_cpu(
        change_offsets_i32.detach().cpu(),
        sample_count=change_frame_i32.shape[0],
        segment_count=change_record_count,
        max_segments_per_sample=129,
    )
    _validate_packed_endpoint_delta_records_cpu(
        base_record_i32=base_record_i32,
        change_record_i32=change_record_i32,
        site_count=site_rgba_f32.shape[0],
        boundary_count=boundary_count,
    )
    config_i32 = torch.tensor(
        [
            boundary_count,
            track_count,
            frame_count,
            site_rgba_f32.shape[0],
            base_record_count,
            change_frame_i32.shape[0],
            change_record_count,
        ],
        device=coeff_f16.device,
        dtype=torch.int32,
    )
    config_f32 = torch.tensor(
        [config.near, config.far, config.invalid_epsilon, config.transmittance_threshold],
        device=coeff_f16.device,
        dtype=torch.float32,
    )
    ops = torch.ops.world_foam_lane2_fused_slab_v0
    op_name = "endpoint_record_delta_replace_coeff16_packed_framegroup16_mse_vjp_direct_atomic_rgb_only"
    if not hasattr(ops, op_name):
        raise RuntimeError(f"world_foam_lane2_fused_slab_v0 {op_name} op not found. Build this variant first.")
    return getattr(ops, op_name)(
        coeff_f16,
        frame_t_f32,
        base_offsets_i32,
        base_record_i32,
        track_change_offsets_i32,
        track_chunk_change_offsets_i16,
        change_frame_i32,
        change_offsets_i32,
        change_record_i32,
        site_rgba_f32,
        target_rgb_f32,
        config_i32,
        config_f32,
    )


def endpoint_record_delta_replace_coeff16_packed_mse_vjp_direct_atomic_rgb_only(
    coeff_f16: Tensor,
    frame_t_f32: Tensor,
    base_offsets_i32: Tensor,
    base_record_i32: Tensor,
    track_change_offsets_i32: Tensor,
    change_frame_i32: Tensor,
    change_offsets_i32: Tensor,
    change_record_i32: Tensor,
    site_rgba_f32: Tensor,
    target_rgb_f32: Tensor,
    config: RealRayReplayConfig,
    *,
    track_count: int,
    frame_count: int,
    boundary_count: int,
) -> tuple[Tensor, Tensor]:
    """Compute RGB MSE/VJP for delta-replace records with packed int32 rows and scalar sample launches."""
    coeff_f16 = coeff_f16.contiguous()
    frame_t_f32 = frame_t_f32.contiguous()
    base_offsets_i32 = base_offsets_i32.contiguous()
    base_record_i32 = base_record_i32.contiguous()
    track_change_offsets_i32 = track_change_offsets_i32.contiguous()
    change_frame_i32 = change_frame_i32.contiguous()
    change_offsets_i32 = change_offsets_i32.contiguous()
    change_record_i32 = change_record_i32.contiguous()
    site_rgba_f32 = site_rgba_f32.contiguous()
    target_rgb_f32 = target_rgb_f32.contiguous()
    _require_mps_tensor(coeff_f16, name="coeff_f16", dtype=torch.float16, cols=4)
    _require_mps_tensor(frame_t_f32, name="frame_t_f32", dtype=torch.float32, cols=None)
    _require_mps_tensor(base_offsets_i32, name="base_offsets_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(base_record_i32, name="base_record_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(track_change_offsets_i32, name="track_change_offsets_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(change_frame_i32, name="change_frame_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(change_offsets_i32, name="change_offsets_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(change_record_i32, name="change_record_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(site_rgba_f32, name="site_rgba_f32", dtype=torch.float32, cols=4)
    if track_count <= 0 or frame_count <= 0 or boundary_count <= 0:
        raise ValueError("track_count, frame_count, and boundary_count must be positive")
    if boundary_count > 4093:
        raise ValueError("packed scalar delta replace coeff16 replay supports boundary count <= 4093")
    if coeff_f16.shape[0] != track_count * boundary_count:
        raise ValueError("coeff_f16 row count must be track_count * boundary_count")
    if frame_t_f32.shape[0] != frame_count:
        raise ValueError("frame_t_f32 must have shape [frame_count]")
    if target_rgb_f32.device.type != "mps" or target_rgb_f32.dtype != torch.float32:
        raise ValueError("target_rgb_f32 must be float32 on MPS")
    if target_rgb_f32.shape != (track_count, frame_count, 3):
        raise ValueError("target_rgb_f32 must have shape [track_count, frame_count, 3]")
    if site_rgba_f32.shape[0] <= 0 or site_rgba_f32.shape[0] > 256:
        raise ValueError("packed scalar delta replace coeff16 replay supports site count in [1, 256]")
    if config.far <= config.near:
        raise ValueError("config.far must be greater than config.near")
    if config.invalid_epsilon <= 0.0:
        raise ValueError("config.invalid_epsilon must be positive")
    if config.transmittance_threshold < 0.0:
        raise ValueError("config.transmittance_threshold must be nonnegative")
    base_record_count = base_record_i32.numel()
    change_record_count = change_record_i32.numel()
    _validate_segment_tape_offsets_cpu(
        base_offsets_i32.detach().cpu(),
        sample_count=track_count,
        segment_count=base_record_count,
        max_segments_per_sample=129,
    )
    _validate_segment_tape_offsets_cpu(
        track_change_offsets_i32.detach().cpu(),
        sample_count=track_count,
        segment_count=change_frame_i32.shape[0],
        max_segments_per_sample=None,
    )
    _validate_segment_tape_offsets_cpu(
        change_offsets_i32.detach().cpu(),
        sample_count=change_frame_i32.shape[0],
        segment_count=change_record_count,
        max_segments_per_sample=129,
    )
    _validate_packed_endpoint_delta_records_cpu(
        base_record_i32=base_record_i32,
        change_record_i32=change_record_i32,
        site_count=site_rgba_f32.shape[0],
        boundary_count=boundary_count,
    )
    config_i32 = torch.tensor(
        [
            boundary_count,
            track_count,
            frame_count,
            site_rgba_f32.shape[0],
            base_record_count,
            change_frame_i32.shape[0],
            change_record_count,
        ],
        device=coeff_f16.device,
        dtype=torch.int32,
    )
    config_f32 = torch.tensor(
        [config.near, config.far, config.invalid_epsilon, config.transmittance_threshold],
        device=coeff_f16.device,
        dtype=torch.float32,
    )
    ops = torch.ops.world_foam_lane2_fused_slab_v0
    op_name = "endpoint_record_delta_replace_coeff16_packed_mse_vjp_direct_atomic_rgb_only"
    if not hasattr(ops, op_name):
        raise RuntimeError(f"world_foam_lane2_fused_slab_v0 {op_name} op not found. Build this variant first.")
    return getattr(ops, op_name)(
        coeff_f16,
        frame_t_f32,
        base_offsets_i32,
        base_record_i32,
        track_change_offsets_i32,
        change_frame_i32,
        change_offsets_i32,
        change_record_i32,
        site_rgba_f32,
        target_rgb_f32,
        config_i32,
        config_f32,
    )


def endpoint_record_delta_replace_coeff16_packed_framegroup16_recompute_mse_vjp_direct_atomic_rgb_only(
    coeff_f16: Tensor,
    frame_t_f32: Tensor,
    base_offsets_i32: Tensor,
    base_record_i32: Tensor,
    track_change_offsets_i32: Tensor,
    track_chunk_change_offsets_i16: Tensor,
    change_frame_i32: Tensor,
    change_offsets_i32: Tensor,
    change_record_i32: Tensor,
    site_rgba_f32: Tensor,
    target_rgb_f32: Tensor,
    config: RealRayReplayConfig,
    *,
    track_count: int,
    frame_count: int,
    boundary_count: int,
) -> tuple[Tensor, Tensor]:
    """Compute packed framegroup16 RGB MSE/VJP while recomputing reverse replay terms."""
    coeff_f16 = coeff_f16.contiguous()
    frame_t_f32 = frame_t_f32.contiguous()
    base_offsets_i32 = base_offsets_i32.contiguous()
    base_record_i32 = base_record_i32.contiguous()
    track_change_offsets_i32 = track_change_offsets_i32.contiguous()
    track_chunk_change_offsets_i16 = track_chunk_change_offsets_i16.contiguous()
    change_frame_i32 = change_frame_i32.contiguous()
    change_offsets_i32 = change_offsets_i32.contiguous()
    change_record_i32 = change_record_i32.contiguous()
    site_rgba_f32 = site_rgba_f32.contiguous()
    target_rgb_f32 = target_rgb_f32.contiguous()
    _require_mps_tensor(coeff_f16, name="coeff_f16", dtype=torch.float16, cols=4)
    _require_mps_tensor(frame_t_f32, name="frame_t_f32", dtype=torch.float32, cols=None)
    _require_mps_tensor(base_offsets_i32, name="base_offsets_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(base_record_i32, name="base_record_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(track_change_offsets_i32, name="track_change_offsets_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(
        track_chunk_change_offsets_i16,
        name="track_chunk_change_offsets_i16",
        dtype=torch.int16,
        cols=None,
    )
    _require_mps_tensor(change_frame_i32, name="change_frame_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(change_offsets_i32, name="change_offsets_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(change_record_i32, name="change_record_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(site_rgba_f32, name="site_rgba_f32", dtype=torch.float32, cols=4)
    if track_count <= 0 or frame_count <= 0 or boundary_count <= 0:
        raise ValueError("track_count, frame_count, and boundary_count must be positive")
    if boundary_count > 4093:
        raise ValueError("packed framegroup16 recompute delta replace coeff16 replay supports boundary count <= 4093")
    if coeff_f16.shape[0] != track_count * boundary_count:
        raise ValueError("coeff_f16 row count must be track_count * boundary_count")
    if frame_t_f32.shape[0] != frame_count:
        raise ValueError("frame_t_f32 must have shape [frame_count]")
    if target_rgb_f32.device.type != "mps" or target_rgb_f32.dtype != torch.float32:
        raise ValueError("target_rgb_f32 must be float32 on MPS")
    if target_rgb_f32.shape != (track_count, frame_count, 3):
        raise ValueError("target_rgb_f32 must have shape [track_count, frame_count, 3]")
    if site_rgba_f32.shape[0] <= 0 or site_rgba_f32.shape[0] > 256:
        raise ValueError("packed framegroup16 recompute delta replace coeff16 replay supports site count in [1, 256]")
    if config.far <= config.near:
        raise ValueError("config.far must be greater than config.near")
    if config.invalid_epsilon <= 0.0:
        raise ValueError("config.invalid_epsilon must be positive")
    if config.transmittance_threshold < 0.0:
        raise ValueError("config.transmittance_threshold must be nonnegative")
    base_record_count = base_record_i32.numel()
    change_record_count = change_record_i32.numel()
    chunk_count = (frame_count + 31) // 32
    if change_frame_i32.shape[0] > 32767:
        raise ValueError("framegroup16 chunk-start offsets use int16 and require change count <= 32767")
    _validate_segment_tape_offsets_cpu(
        base_offsets_i32.detach().cpu(),
        sample_count=track_count,
        segment_count=base_record_count,
        max_segments_per_sample=129,
    )
    _validate_segment_tape_offsets_cpu(
        track_change_offsets_i32.detach().cpu(),
        sample_count=track_count,
        segment_count=change_frame_i32.shape[0],
        max_segments_per_sample=None,
    )
    _validate_segment_tape_offsets_cpu(
        track_chunk_change_offsets_i16.detach().cpu(),
        sample_count=track_count * (chunk_count + 1) - 1,
        segment_count=change_frame_i32.shape[0],
        max_segments_per_sample=None,
    )
    _validate_segment_tape_offsets_cpu(
        change_offsets_i32.detach().cpu(),
        sample_count=change_frame_i32.shape[0],
        segment_count=change_record_count,
        max_segments_per_sample=129,
    )
    _validate_packed_endpoint_delta_records_cpu(
        base_record_i32=base_record_i32,
        change_record_i32=change_record_i32,
        site_count=site_rgba_f32.shape[0],
        boundary_count=boundary_count,
    )
    config_i32 = torch.tensor(
        [
            boundary_count,
            track_count,
            frame_count,
            site_rgba_f32.shape[0],
            base_record_count,
            change_frame_i32.shape[0],
            change_record_count,
        ],
        device=coeff_f16.device,
        dtype=torch.int32,
    )
    config_f32 = torch.tensor(
        [config.near, config.far, config.invalid_epsilon, config.transmittance_threshold],
        device=coeff_f16.device,
        dtype=torch.float32,
    )
    ops = torch.ops.world_foam_lane2_fused_slab_v0
    op_name = "endpoint_record_delta_replace_coeff16_packed_framegroup16_recompute_mse_vjp_direct_atomic_rgb_only"
    if not hasattr(ops, op_name):
        raise RuntimeError(f"world_foam_lane2_fused_slab_v0 {op_name} op not found. Build this variant first.")
    return getattr(ops, op_name)(
        coeff_f16,
        frame_t_f32,
        base_offsets_i32,
        base_record_i32,
        track_change_offsets_i32,
        track_chunk_change_offsets_i16,
        change_frame_i32,
        change_offsets_i32,
        change_record_i32,
        site_rgba_f32,
        target_rgb_f32,
        config_i32,
        config_f32,
    )


def endpoint_record_delta_replace_factorized_packed_framegroup16_recompute_mse_vjp_direct_atomic_rgb_only(
    boundary_f32: Tensor,
    track_ray_coeff_f32: Tensor,
    frame_t_f32: Tensor,
    base_offsets_i16: Tensor,
    base_record_i32: Tensor,
    track_change_offsets_i16: Tensor,
    track_chunk_change_offsets_i16: Tensor,
    change_frame_i16: Tensor,
    change_offsets_i16: Tensor,
    change_record_i32: Tensor,
    site_rgba_f32: Tensor,
    target_rgb_f32: Tensor,
    config: RealRayReplayConfig,
    *,
    track_count: int,
    frame_count: int,
    boundary_count: int,
) -> tuple[Tensor, Tensor]:
    """Compute packed framegroup16 RGB MSE/VJP from factorized boundary/ray coefficients."""
    boundary_f32 = boundary_f32.contiguous()
    track_ray_coeff_f32 = track_ray_coeff_f32.contiguous()
    frame_t_f32 = frame_t_f32.contiguous()
    base_offsets_i16 = base_offsets_i16.contiguous()
    base_record_i32 = base_record_i32.contiguous()
    track_change_offsets_i16 = track_change_offsets_i16.contiguous()
    track_chunk_change_offsets_i16 = track_chunk_change_offsets_i16.contiguous()
    change_frame_i16 = change_frame_i16.contiguous()
    change_offsets_i16 = change_offsets_i16.contiguous()
    change_record_i32 = change_record_i32.contiguous()
    site_rgba_f32 = site_rgba_f32.contiguous()
    target_rgb_f32 = target_rgb_f32.contiguous()
    _require_mps_tensor(boundary_f32, name="boundary_f32", dtype=torch.float32, cols=5)
    _require_mps_tensor(track_ray_coeff_f32, name="track_ray_coeff_f32", dtype=torch.float32, cols=12)
    _require_mps_tensor(frame_t_f32, name="frame_t_f32", dtype=torch.float32, cols=None)
    _require_mps_tensor(base_offsets_i16, name="base_offsets_i16", dtype=torch.int16, cols=None)
    _require_mps_tensor(base_record_i32, name="base_record_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(track_change_offsets_i16, name="track_change_offsets_i16", dtype=torch.int16, cols=None)
    _require_mps_tensor(
        track_chunk_change_offsets_i16,
        name="track_chunk_change_offsets_i16",
        dtype=torch.int16,
        cols=None,
    )
    _require_mps_tensor(change_frame_i16, name="change_frame_i16", dtype=torch.int16, cols=None)
    _require_mps_tensor(change_offsets_i16, name="change_offsets_i16", dtype=torch.int16, cols=None)
    _require_mps_tensor(change_record_i32, name="change_record_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(site_rgba_f32, name="site_rgba_f32", dtype=torch.float32, cols=4)
    if track_count <= 0 or frame_count <= 0 or boundary_count <= 0:
        raise ValueError("track_count, frame_count, and boundary_count must be positive")
    if boundary_count > 4093:
        raise ValueError("packed framegroup16 factorized delta replace replay supports boundary count <= 4093")
    if boundary_f32.shape[0] != boundary_count:
        raise ValueError("boundary_f32 row count must match boundary_count")
    if track_ray_coeff_f32.shape[0] != track_count:
        raise ValueError("track_ray_coeff_f32 row count must match track_count")
    if frame_t_f32.shape[0] != frame_count:
        raise ValueError("frame_t_f32 must have shape [frame_count]")
    if target_rgb_f32.device.type != "mps" or target_rgb_f32.dtype != torch.float32:
        raise ValueError("target_rgb_f32 must be float32 on MPS")
    if target_rgb_f32.shape != (track_count, frame_count, 3):
        raise ValueError("target_rgb_f32 must have shape [track_count, frame_count, 3]")
    if site_rgba_f32.shape[0] <= 0 or site_rgba_f32.shape[0] > 256:
        raise ValueError("packed framegroup16 factorized delta replace replay supports site count in [1, 256]")
    if config.far <= config.near:
        raise ValueError("config.far must be greater than config.near")
    if config.invalid_epsilon <= 0.0:
        raise ValueError("config.invalid_epsilon must be positive")
    if config.transmittance_threshold < 0.0:
        raise ValueError("config.transmittance_threshold must be nonnegative")
    base_record_count = base_record_i32.numel()
    change_record_count = change_record_i32.numel()
    chunk_count = (frame_count + 31) // 32
    if change_frame_i16.shape[0] > 32767:
        raise ValueError("framegroup16 chunk-start offsets use int16 and require change count <= 32767")
    _validate_segment_tape_offsets_cpu(
        base_offsets_i16.detach().cpu(),
        sample_count=track_count,
        segment_count=base_record_count,
        max_segments_per_sample=129,
    )
    _validate_segment_tape_offsets_cpu(
        track_change_offsets_i16.detach().cpu(),
        sample_count=track_count,
        segment_count=change_frame_i16.shape[0],
        max_segments_per_sample=None,
    )
    _validate_segment_tape_offsets_cpu(
        track_chunk_change_offsets_i16.detach().cpu(),
        sample_count=track_count * (chunk_count + 1) - 1,
        segment_count=change_frame_i16.shape[0],
        max_segments_per_sample=None,
    )
    _validate_segment_tape_offsets_cpu(
        change_offsets_i16.detach().cpu(),
        sample_count=change_frame_i16.shape[0],
        segment_count=change_record_count,
        max_segments_per_sample=129,
    )
    _validate_packed_endpoint_delta_records_cpu(
        base_record_i32=base_record_i32,
        change_record_i32=change_record_i32,
        site_count=site_rgba_f32.shape[0],
        boundary_count=boundary_count,
    )
    config_i32 = torch.tensor(
        [
            boundary_count,
            track_count,
            frame_count,
            site_rgba_f32.shape[0],
            base_record_count,
            change_frame_i16.shape[0],
            change_record_count,
        ],
        device=boundary_f32.device,
        dtype=torch.int32,
    )
    config_f32 = torch.tensor(
        [config.near, config.far, config.invalid_epsilon, config.transmittance_threshold],
        device=boundary_f32.device,
        dtype=torch.float32,
    )
    ops = torch.ops.world_foam_lane2_fused_slab_v0
    op_name = "endpoint_record_delta_replace_factorized_packed_framegroup16_recompute_mse_vjp_direct_atomic_rgb_only"
    if not hasattr(ops, op_name):
        raise RuntimeError(f"world_foam_lane2_fused_slab_v0 {op_name} op not found. Build this variant first.")
    return getattr(ops, op_name)(
        boundary_f32,
        track_ray_coeff_f32,
        frame_t_f32,
        base_offsets_i16,
        base_record_i32,
        track_change_offsets_i16,
        track_chunk_change_offsets_i16,
        change_frame_i16,
        change_offsets_i16,
        change_record_i32,
        site_rgba_f32,
        target_rgb_f32,
        config_i32,
        config_f32,
    )


def endpoint_record_delta_replace_factorized_packed_framegroup16_constant_state_mse_vjp_direct_atomic_rgb_boundary(
    boundary_f32: Tensor,
    track_ray_coeff_f32: Tensor,
    frame_t_f32: Tensor,
    base_offsets_i16: Tensor,
    base_record_i32: Tensor,
    track_change_offsets_i16: Tensor,
    track_chunk_change_offsets_i16: Tensor,
    change_frame_i16: Tensor,
    change_offsets_i16: Tensor,
    change_record_i32: Tensor,
    site_rgba_f32: Tensor,
    target_rgb_f32: Tensor,
    config: RealRayReplayConfig,
    *,
    track_count: int,
    frame_count: int,
    boundary_count: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Run constant-state P0 replay and return loss, site RGBA VJP, and boundary VJP."""
    boundary_f32 = boundary_f32.contiguous()
    track_ray_coeff_f32 = track_ray_coeff_f32.contiguous()
    frame_t_f32 = frame_t_f32.contiguous()
    base_offsets_i16 = base_offsets_i16.contiguous()
    base_record_i32 = base_record_i32.contiguous()
    track_change_offsets_i16 = track_change_offsets_i16.contiguous()
    track_chunk_change_offsets_i16 = track_chunk_change_offsets_i16.contiguous()
    change_frame_i16 = change_frame_i16.contiguous()
    change_offsets_i16 = change_offsets_i16.contiguous()
    change_record_i32 = change_record_i32.contiguous()
    site_rgba_f32 = site_rgba_f32.contiguous()
    target_rgb_f32 = target_rgb_f32.contiguous()
    _require_mps_tensor(boundary_f32, name="boundary_f32", dtype=torch.float32, cols=5)
    _require_mps_tensor(track_ray_coeff_f32, name="track_ray_coeff_f32", dtype=torch.float32, cols=12)
    _require_mps_tensor(frame_t_f32, name="frame_t_f32", dtype=torch.float32, cols=None)
    _require_mps_tensor(base_offsets_i16, name="base_offsets_i16", dtype=torch.int16, cols=None)
    _require_mps_tensor(base_record_i32, name="base_record_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(track_change_offsets_i16, name="track_change_offsets_i16", dtype=torch.int16, cols=None)
    _require_mps_tensor(
        track_chunk_change_offsets_i16,
        name="track_chunk_change_offsets_i16",
        dtype=torch.int16,
        cols=None,
    )
    _require_mps_tensor(change_frame_i16, name="change_frame_i16", dtype=torch.int16, cols=None)
    _require_mps_tensor(change_offsets_i16, name="change_offsets_i16", dtype=torch.int16, cols=None)
    _require_mps_tensor(change_record_i32, name="change_record_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(site_rgba_f32, name="site_rgba_f32", dtype=torch.float32, cols=4)
    if track_count <= 0 or frame_count <= 0 or boundary_count <= 0:
        raise ValueError("track_count, frame_count, and boundary_count must be positive")
    if boundary_count > 4093:
        raise ValueError("constant-state packed factorized replay supports boundary count <= 4093")
    if boundary_f32.shape[0] != boundary_count:
        raise ValueError("boundary_f32 row count must match boundary_count")
    if track_ray_coeff_f32.shape[0] != track_count:
        raise ValueError("track_ray_coeff_f32 row count must match track_count")
    if frame_t_f32.shape[0] != frame_count:
        raise ValueError("frame_t_f32 must have shape [frame_count]")
    if target_rgb_f32.device.type != "mps" or target_rgb_f32.dtype != torch.float32:
        raise ValueError("target_rgb_f32 must be float32 on MPS")
    if target_rgb_f32.shape != (track_count, frame_count, 3):
        raise ValueError("target_rgb_f32 must have shape [track_count, frame_count, 3]")
    if site_rgba_f32.shape[0] <= 0 or site_rgba_f32.shape[0] > 256:
        raise ValueError("constant-state packed factorized replay supports site count in [1, 256]")
    if config.far <= config.near:
        raise ValueError("config.far must be greater than config.near")
    if config.invalid_epsilon <= 0.0:
        raise ValueError("config.invalid_epsilon must be positive")
    if config.transmittance_threshold < 0.0:
        raise ValueError("config.transmittance_threshold must be nonnegative")
    base_record_count = base_record_i32.numel()
    change_record_count = change_record_i32.numel()
    chunk_count = (frame_count + 31) // 32
    if change_frame_i16.shape[0] > 32767:
        raise ValueError("framegroup16 chunk-start offsets use int16 and require change count <= 32767")
    _validate_segment_tape_offsets_cpu(
        base_offsets_i16.detach().cpu(),
        sample_count=track_count,
        segment_count=base_record_count,
        max_segments_per_sample=129,
    )
    _validate_segment_tape_offsets_cpu(
        track_change_offsets_i16.detach().cpu(),
        sample_count=track_count,
        segment_count=change_frame_i16.shape[0],
        max_segments_per_sample=None,
    )
    _validate_segment_tape_offsets_cpu(
        track_chunk_change_offsets_i16.detach().cpu(),
        sample_count=track_count * (chunk_count + 1) - 1,
        segment_count=change_frame_i16.shape[0],
        max_segments_per_sample=None,
    )
    _validate_segment_tape_offsets_cpu(
        change_offsets_i16.detach().cpu(),
        sample_count=change_frame_i16.shape[0],
        segment_count=change_record_count,
        max_segments_per_sample=129,
    )
    _validate_packed_endpoint_delta_records_cpu(
        base_record_i32=base_record_i32,
        change_record_i32=change_record_i32,
        site_count=site_rgba_f32.shape[0],
        boundary_count=boundary_count,
    )
    config_i32 = torch.tensor(
        [
            boundary_count,
            track_count,
            frame_count,
            site_rgba_f32.shape[0],
            base_record_count,
            change_frame_i16.shape[0],
            change_record_count,
        ],
        device=boundary_f32.device,
        dtype=torch.int32,
    )
    config_f32 = torch.tensor(
        [config.near, config.far, config.invalid_epsilon, config.transmittance_threshold],
        device=boundary_f32.device,
        dtype=torch.float32,
    )
    ops = torch.ops.world_foam_lane2_fused_slab_v0
    op_name = (
        "endpoint_record_delta_replace_factorized_packed_framegroup16_constant_state_mse_vjp_direct_atomic_rgb_boundary"
    )
    if not hasattr(ops, op_name):
        raise RuntimeError(f"world_foam_lane2_fused_slab_v0 {op_name} op not found. Build this variant first.")
    return getattr(ops, op_name)(
        boundary_f32,
        track_ray_coeff_f32,
        frame_t_f32,
        base_offsets_i16,
        base_record_i32,
        track_change_offsets_i16,
        track_chunk_change_offsets_i16,
        change_frame_i16,
        change_offsets_i16,
        change_record_i32,
        site_rgba_f32,
        target_rgb_f32,
        config_i32,
        config_f32,
    )


def endpoint_record_delta_replace_factorized_packed_framegroup16_constant_state_p0_mse_vjp_sparse_mobius_rgb_boundary(
    boundary_f32: Tensor,
    track_ray_coeff_f32: Tensor,
    frame_t_f32: Tensor,
    base_offsets_i16: Tensor,
    base_record_incidence_i32: Tensor,
    track_change_offsets_i16: Tensor,
    track_chunk_change_offsets_i16: Tensor,
    change_frame_i16: Tensor,
    change_offsets_i16: Tensor,
    change_record_incidence_i32: Tensor,
    track_incidence_offsets_i32: Tensor,
    incidence_boundary_i32: Tensor,
    site_rgba_f32: Tensor,
    target_rgb_f32: Tensor,
    config: RealRayReplayConfig,
    *,
    track_count: int,
    frame_count: int,
    boundary_count: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Run staged constant-state P0 replay through sparse Mobius adjoints.

    Packed internal cuts are row-local incidence ids produced by
    :func:`recode_packed_endpoint_delta_records_to_track_incidence`. Returns
    ``(loss, grad_site_rgba, grad_mobius_coeff, grad_boundary)`` where the
    coefficient adjoint has shape ``[I,4]`` and the boundary VJP is applied
    once per incidence by a second Metal kernel.
    """
    boundary_f32 = boundary_f32.contiguous()
    track_ray_coeff_f32 = track_ray_coeff_f32.contiguous()
    frame_t_f32 = frame_t_f32.contiguous()
    base_offsets_i16 = base_offsets_i16.contiguous()
    base_record_incidence_i32 = base_record_incidence_i32.contiguous()
    track_change_offsets_i16 = track_change_offsets_i16.contiguous()
    track_chunk_change_offsets_i16 = track_chunk_change_offsets_i16.contiguous()
    change_frame_i16 = change_frame_i16.contiguous()
    change_offsets_i16 = change_offsets_i16.contiguous()
    change_record_incidence_i32 = change_record_incidence_i32.contiguous()
    track_incidence_offsets_i32 = track_incidence_offsets_i32.contiguous()
    incidence_boundary_i32 = incidence_boundary_i32.contiguous()
    site_rgba_f32 = site_rgba_f32.contiguous()
    target_rgb_f32 = target_rgb_f32.contiguous()
    _require_mps_tensor(boundary_f32, name="boundary_f32", dtype=torch.float32, cols=5)
    _require_mps_tensor(track_ray_coeff_f32, name="track_ray_coeff_f32", dtype=torch.float32, cols=12)
    _require_mps_tensor(frame_t_f32, name="frame_t_f32", dtype=torch.float32, cols=None)
    _require_mps_tensor(base_offsets_i16, name="base_offsets_i16", dtype=torch.int16, cols=None)
    _require_mps_tensor(
        base_record_incidence_i32,
        name="base_record_incidence_i32",
        dtype=torch.int32,
        cols=None,
    )
    _require_mps_tensor(track_change_offsets_i16, name="track_change_offsets_i16", dtype=torch.int16, cols=None)
    _require_mps_tensor(
        track_chunk_change_offsets_i16,
        name="track_chunk_change_offsets_i16",
        dtype=torch.int16,
        cols=None,
    )
    _require_mps_tensor(change_frame_i16, name="change_frame_i16", dtype=torch.int16, cols=None)
    _require_mps_tensor(change_offsets_i16, name="change_offsets_i16", dtype=torch.int16, cols=None)
    _require_mps_tensor(
        change_record_incidence_i32,
        name="change_record_incidence_i32",
        dtype=torch.int32,
        cols=None,
    )
    _require_mps_tensor(
        track_incidence_offsets_i32,
        name="track_incidence_offsets_i32",
        dtype=torch.int32,
        cols=None,
    )
    _require_mps_tensor(incidence_boundary_i32, name="incidence_boundary_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(site_rgba_f32, name="site_rgba_f32", dtype=torch.float32, cols=4)
    if track_count <= 0 or frame_count <= 0 or boundary_count <= 0:
        raise ValueError("track_count, frame_count, and boundary_count must be positive")
    if boundary_f32.shape[0] != boundary_count:
        raise ValueError("boundary_f32 row count must match boundary_count")
    if track_ray_coeff_f32.shape[0] != track_count:
        raise ValueError("track_ray_coeff_f32 row count must match track_count")
    if frame_t_f32.shape[0] != frame_count:
        raise ValueError("frame_t_f32 must have shape [frame_count]")
    if target_rgb_f32.device.type != "mps" or target_rgb_f32.dtype != torch.float32:
        raise ValueError("target_rgb_f32 must be float32 on MPS")
    if target_rgb_f32.shape != (track_count, frame_count, 3):
        raise ValueError("target_rgb_f32 must have shape [track_count, frame_count, 3]")
    if site_rgba_f32.shape[0] <= 0 or site_rgba_f32.shape[0] > 256:
        raise ValueError("sparse-Mobius constant-state P0 replay supports site count in [1, 256]")
    if config.far <= config.near:
        raise ValueError("config.far must be greater than config.near")
    if config.invalid_epsilon <= 0.0:
        raise ValueError("config.invalid_epsilon must be positive")
    if config.transmittance_threshold < 0.0:
        raise ValueError("config.transmittance_threshold must be nonnegative")
    base_record_count = base_record_incidence_i32.numel()
    change_record_count = change_record_incidence_i32.numel()
    incidence_count = incidence_boundary_i32.numel()
    change_count = change_frame_i16.shape[0]
    chunk_count = (frame_count + 31) // 32
    if change_count > 32767:
        raise ValueError("framegroup16 chunk-start offsets use int16 and require change count <= 32767")
    _validate_segment_tape_offsets_cpu(
        base_offsets_i16.detach().cpu(),
        sample_count=track_count,
        segment_count=base_record_count,
        max_segments_per_sample=129,
    )
    _validate_segment_tape_offsets_cpu(
        track_change_offsets_i16.detach().cpu(),
        sample_count=track_count,
        segment_count=change_count,
        max_segments_per_sample=None,
    )
    _validate_segment_tape_offsets_cpu(
        track_chunk_change_offsets_i16.detach().cpu(),
        sample_count=track_count * (chunk_count + 1) - 1,
        segment_count=change_count,
        max_segments_per_sample=None,
    )
    _validate_segment_tape_offsets_cpu(
        change_offsets_i16.detach().cpu(),
        sample_count=change_count,
        segment_count=change_record_count,
        max_segments_per_sample=129,
    )
    _validate_track_boundary_incidence_csr_cpu(
        track_incidence_offsets_i32,
        incidence_boundary_i32,
        track_count=track_count,
        boundary_count=boundary_count,
    )
    _validate_packed_endpoint_incidence_delta_records_cpu(
        base_offsets_i16=base_offsets_i16,
        base_record_incidence_i32=base_record_incidence_i32,
        track_change_offsets_i16=track_change_offsets_i16,
        change_offsets_i16=change_offsets_i16,
        change_record_incidence_i32=change_record_incidence_i32,
        track_incidence_offsets_i32=track_incidence_offsets_i32,
        track_count=track_count,
        site_count=site_rgba_f32.shape[0],
    )
    config_i32 = torch.tensor(
        [
            boundary_count,
            track_count,
            frame_count,
            site_rgba_f32.shape[0],
            base_record_count,
            change_count,
            change_record_count,
            incidence_count,
        ],
        device=boundary_f32.device,
        dtype=torch.int32,
    )
    config_f32 = torch.tensor(
        [config.near, config.far, config.invalid_epsilon, config.transmittance_threshold],
        device=boundary_f32.device,
        dtype=torch.float32,
    )
    ops = torch.ops.world_foam_lane2_fused_slab_v0
    op_name = (
        "endpoint_record_delta_replace_factorized_packed_framegroup16_"
        "constant_state_p0_mse_vjp_sparse_mobius_rgb_boundary"
    )
    if not hasattr(ops, op_name):
        raise RuntimeError(f"world_foam_lane2_fused_slab_v0 {op_name} op not found. Build this variant first.")
    return getattr(ops, op_name)(
        boundary_f32,
        track_ray_coeff_f32,
        frame_t_f32,
        base_offsets_i16,
        base_record_incidence_i32,
        track_change_offsets_i16,
        track_chunk_change_offsets_i16,
        change_frame_i16,
        change_offsets_i16,
        change_record_incidence_i32,
        track_incidence_offsets_i32,
        incidence_boundary_i32,
        site_rgba_f32,
        target_rgb_f32,
        config_i32,
        config_f32,
    )


def prepare_fixed_word_p0_compiled_lie_transfer_mse_vjp_sparse_mobius_boundary(
    boundary_f32: Tensor,
    track_ray_coeff_f32: Tensor,
    compiler_node_t_f32: Tensor,
    word_offsets_i32: Tensor,
    word_owner_i32: Tensor,
    word_left_incidence_i32: Tensor,
    word_right_incidence_i32: Tensor,
    track_incidence_offsets_i32: Tensor,
    incidence_boundary_i32: Tensor,
    site_rgba_f32: Tensor,
    sample_to_node_f32: Tensor,
    target_rgb_f32: Tensor,
    background_rgb_f32: Tensor,
    config: RealRayReplayConfig,
    *,
    track_count: int,
    node_count: int,
    sample_count: int,
    boundary_count: int,
    loss_scale: float,
    physical_length_epsilon: float = 1.0e-8,
    cone_tolerance: float = 1.0e-6,
) -> _CheckedFixedWordP0FusedCall:
    """Validate and retain launch-ready fixed-word P0 compiled-Lie storage.

    ``loss_scale`` is global, normally ``1 / (global_tracks * global_samples *
    3)``. It must not be recomputed from this local track/sample/chart block;
    retaining it across blocks makes the scalar loss and all adjoints exactly
    partition invariant.
    """
    boundary_f32 = boundary_f32.contiguous()
    track_ray_coeff_f32 = track_ray_coeff_f32.contiguous()
    compiler_node_t_f32 = compiler_node_t_f32.contiguous()
    word_offsets_i32 = word_offsets_i32.contiguous()
    word_owner_i32 = word_owner_i32.contiguous()
    word_left_incidence_i32 = word_left_incidence_i32.contiguous()
    word_right_incidence_i32 = word_right_incidence_i32.contiguous()
    track_incidence_offsets_i32 = track_incidence_offsets_i32.contiguous()
    incidence_boundary_i32 = incidence_boundary_i32.contiguous()
    site_rgba_f32 = site_rgba_f32.contiguous()
    sample_to_node_f32 = sample_to_node_f32.contiguous()
    target_rgb_f32 = target_rgb_f32.contiguous()
    background_rgb_f32 = background_rgb_f32.contiguous()
    _require_mps_tensor(boundary_f32, name="boundary_f32", dtype=torch.float32, cols=5)
    _require_mps_tensor(track_ray_coeff_f32, name="track_ray_coeff_f32", dtype=torch.float32, cols=12)
    _require_mps_tensor(compiler_node_t_f32, name="compiler_node_t_f32", dtype=torch.float32, cols=None)
    _require_mps_tensor(word_offsets_i32, name="word_offsets_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(word_owner_i32, name="word_owner_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(
        word_left_incidence_i32,
        name="word_left_incidence_i32",
        dtype=torch.int32,
        cols=None,
    )
    _require_mps_tensor(
        word_right_incidence_i32,
        name="word_right_incidence_i32",
        dtype=torch.int32,
        cols=None,
    )
    _require_mps_tensor(
        track_incidence_offsets_i32,
        name="track_incidence_offsets_i32",
        dtype=torch.int32,
        cols=None,
    )
    _require_mps_tensor(incidence_boundary_i32, name="incidence_boundary_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(site_rgba_f32, name="site_rgba_f32", dtype=torch.float32, cols=4)
    _require_mps_tensor(sample_to_node_f32, name="sample_to_node_f32", dtype=torch.float32, cols=node_count)
    _require_mps_tensor(background_rgb_f32, name="background_rgb_f32", dtype=torch.float32, cols=None)
    if target_rgb_f32.device.type != "mps" or target_rgb_f32.dtype != torch.float32:
        raise ValueError("target_rgb_f32 must be float32 on MPS")
    if not target_rgb_f32.is_contiguous():
        raise ValueError("target_rgb_f32 must be contiguous")
    if track_count <= 0 or node_count <= 0 or sample_count <= 0:
        raise ValueError("track_count, node_count, and sample_count must be positive")
    if boundary_count < 0:
        raise ValueError("boundary_count must be nonnegative")
    if boundary_f32.shape[0] != boundary_count:
        raise ValueError("boundary_f32 row count must match boundary_count")
    if track_ray_coeff_f32.shape[0] != track_count:
        raise ValueError("track_ray_coeff_f32 row count must match track_count")
    if compiler_node_t_f32.shape[0] != node_count:
        raise ValueError("compiler_node_t_f32 must have shape [node_count]")
    if word_offsets_i32.shape[0] != track_count + 1:
        raise ValueError("word_offsets_i32 must have shape [track_count + 1]")
    if track_incidence_offsets_i32.shape[0] != track_count + 1:
        raise ValueError("track_incidence_offsets_i32 must have shape [track_count + 1]")
    if site_rgba_f32.shape[0] <= 0:
        raise ValueError("site_rgba_f32 must contain at least one site")
    if sample_to_node_f32.shape != (sample_count, node_count):
        raise ValueError("sample_to_node_f32 must have shape [sample_count, node_count]")
    if target_rgb_f32.shape != (track_count, sample_count, 3):
        raise ValueError("target_rgb_f32 must have shape [track_count, sample_count, 3]")
    if background_rgb_f32.shape != (3,):
        raise ValueError("background_rgb_f32 must have shape [3]")
    scalar_config = (
        config.near,
        config.far,
        config.invalid_epsilon,
        physical_length_epsilon,
        cone_tolerance,
        loss_scale,
    )
    if not all(math.isfinite(value) for value in scalar_config):
        raise ValueError("compiled Lie transfer scalar configuration must be finite")
    if config.far <= config.near:
        raise ValueError("config.far must be greater than config.near")
    if config.invalid_epsilon <= 0.0:
        raise ValueError("config.invalid_epsilon must be positive")
    if physical_length_epsilon <= 0.0:
        raise ValueError("physical_length_epsilon must be positive")
    if cone_tolerance < 0.0:
        raise ValueError("cone_tolerance must be nonnegative")
    if loss_scale <= 0.0:
        raise ValueError("loss_scale must be positive")
    word_count = word_owner_i32.numel()
    incidence_count = incidence_boundary_i32.numel()
    if word_count < track_count:
        raise ValueError("fixed-word storage must contain at least one segment per track")
    _validate_track_boundary_incidence_csr_cpu(
        track_incidence_offsets_i32,
        incidence_boundary_i32,
        track_count=track_count,
        boundary_count=boundary_count,
    )
    referenced_boundary_ids = torch.unique(
        incidence_boundary_i32.detach().cpu().to(dtype=torch.int64),
        sorted=True,
    )
    compact_boundary_ids = torch.arange(boundary_count, dtype=torch.int64)
    if not torch.equal(referenced_boundary_ids, compact_boundary_ids):
        raise ValueError("boundary_site_pairs_i32 must be the exact compact referenced boundary table")
    _validate_fixed_word_incidence_csr_cpu(
        word_offsets_i32=word_offsets_i32,
        word_owner_i32=word_owner_i32,
        word_left_incidence_i32=word_left_incidence_i32,
        word_right_incidence_i32=word_right_incidence_i32,
        track_incidence_offsets_i32=track_incidence_offsets_i32,
        track_count=track_count,
        site_count=site_rgba_f32.shape[0],
    )
    config_i32 = torch.tensor(
        [
            boundary_count,
            track_count,
            node_count,
            sample_count,
            site_rgba_f32.shape[0],
            word_count,
            incidence_count,
            incidence_count,
        ],
        device=boundary_f32.device,
        dtype=torch.int32,
    )
    config_f32 = torch.tensor(
        scalar_config,
        device=boundary_f32.device,
        dtype=torch.float32,
    )
    return _CheckedFixedWordP0FusedCall(
        boundary_f32=boundary_f32,
        track_ray_coeff_f32=track_ray_coeff_f32,
        compiler_node_t_f32=compiler_node_t_f32,
        word_offsets_i32=word_offsets_i32,
        word_owner_i32=word_owner_i32,
        word_left_incidence_i32=word_left_incidence_i32,
        word_right_incidence_i32=word_right_incidence_i32,
        track_incidence_offsets_i32=track_incidence_offsets_i32,
        incidence_boundary_i32=incidence_boundary_i32,
        site_rgba_f32=site_rgba_f32,
        sample_to_node_f32=sample_to_node_f32,
        target_rgb_f32=target_rgb_f32,
        background_rgb_f32=background_rgb_f32,
        config_i32=config_i32,
        config_f32=config_f32,
        boundary_count=boundary_count,
        track_count=track_count,
        node_count=node_count,
        sample_count=sample_count,
        site_count=site_rgba_f32.shape[0],
        word_count=word_count,
        incidence_count=incidence_count,
    )


def launch_prepared_fixed_word_p0_compiled_lie_transfer_mse_vjp_sparse_mobius_boundary(
    prepared: _CheckedFixedWordP0FusedCall,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Launch a checked resident token without host reads or validation."""
    return torch.ops.world_foam_lane2_fused_slab_v0.fixed_word_p0_compiled_lie_transfer_mse_vjp_sparse_mobius_boundary_launch_only(
        prepared.boundary_f32,
        prepared.track_ray_coeff_f32,
        prepared.compiler_node_t_f32,
        prepared.word_offsets_i32,
        prepared.word_owner_i32,
        prepared.word_left_incidence_i32,
        prepared.word_right_incidence_i32,
        prepared.track_incidence_offsets_i32,
        prepared.incidence_boundary_i32,
        prepared.site_rgba_f32,
        prepared.sample_to_node_f32,
        prepared.target_rgb_f32,
        prepared.background_rgb_f32,
        prepared.config_i32,
        prepared.config_f32,
        prepared.boundary_count,
        prepared.track_count,
        prepared.node_count,
        prepared.sample_count,
        prepared.site_count,
        prepared.word_count,
        prepared.incidence_count,
    )


def prepare_fixed_word_p0_topology_token(
    word_offsets_i32: Tensor,
    word_owner_i32: Tensor,
    word_left_incidence_i32: Tensor,
    word_right_incidence_i32: Tensor,
    track_incidence_offsets_i32: Tensor,
    incidence_boundary_i32: Tensor,
    boundary_site_pairs_i32: Tensor,
    *,
    track_count: int,
    site_count: int,
    certificate_binding: NativeFixedWordP0RuntimeBinding,
) -> FixedWordP0TopologyToken:
    """Bind native CSR to a strict-eval or material-training capability."""
    certificate_binding = assert_native_fixed_word_p0_runtime_binding(certificate_binding)
    tensors = tuple(
        tensor.contiguous()
        for tensor in (
            word_offsets_i32,
            word_owner_i32,
            word_left_incidence_i32,
            word_right_incidence_i32,
            track_incidence_offsets_i32,
            incidence_boundary_i32,
            boundary_site_pairs_i32,
        )
    )
    (
        word_offsets_i32,
        word_owner_i32,
        word_left_incidence_i32,
        word_right_incidence_i32,
        track_incidence_offsets_i32,
        incidence_boundary_i32,
        boundary_site_pairs_i32,
    ) = tensors
    for name, tensor in (
        ("word_offsets_i32", word_offsets_i32),
        ("word_owner_i32", word_owner_i32),
        ("word_left_incidence_i32", word_left_incidence_i32),
        ("word_right_incidence_i32", word_right_incidence_i32),
        ("track_incidence_offsets_i32", track_incidence_offsets_i32),
        ("incidence_boundary_i32", incidence_boundary_i32),
    ):
        _require_mps_tensor(tensor, name=name, dtype=torch.int32, cols=None)
    _require_mps_tensor(
        boundary_site_pairs_i32,
        name="boundary_site_pairs_i32",
        dtype=torch.int32,
        cols=2,
    )
    if track_count <= 0 or site_count <= 0:
        raise ValueError("track_count and site_count must be positive")
    boundary_count = boundary_site_pairs_i32.shape[0]
    word_count = word_owner_i32.numel()
    incidence_count = incidence_boundary_i32.numel()
    _validate_track_boundary_incidence_csr_cpu(
        track_incidence_offsets_i32,
        incidence_boundary_i32,
        track_count=track_count,
        boundary_count=boundary_count,
    )
    _validate_compact_referenced_boundaries_cpu(
        incidence_boundary_i32,
        boundary_count=boundary_count,
    )
    _validate_fixed_word_incidence_csr_cpu(
        word_offsets_i32=word_offsets_i32,
        word_owner_i32=word_owner_i32,
        word_left_incidence_i32=word_left_incidence_i32,
        word_right_incidence_i32=word_right_incidence_i32,
        track_incidence_offsets_i32=track_incidence_offsets_i32,
        track_count=track_count,
        site_count=site_count,
    )
    pairs_cpu = boundary_site_pairs_i32.detach().cpu().to(dtype=torch.int64)
    if pairs_cpu.numel() and (int(pairs_cpu.min().item()) < 0 or int(pairs_cpu.max().item()) >= site_count):
        raise ValueError("boundary_site_pairs_i32 values must be in [0, site_count)")
    if pairs_cpu.numel() and bool((pairs_cpu[:, 0] == pairs_cpu[:, 1]).any().item()):
        raise ValueError("power boundaries must connect two distinct sites")
    certificate_binding.assert_native_topology(
        word_offsets_i32=word_offsets_i32,
        word_owner_i32=word_owner_i32,
        word_left_incidence_i32=word_left_incidence_i32,
        word_right_incidence_i32=word_right_incidence_i32,
        track_incidence_offsets_i32=track_incidence_offsets_i32,
        incidence_boundary_i32=incidence_boundary_i32,
        boundary_site_pairs_i32=boundary_site_pairs_i32,
    )
    boundary_ids = torch.arange(
        boundary_count,
        device=boundary_site_pairs_i32.device,
        dtype=torch.int32,
    )[:, None]
    active_pairs = torch.cat((boundary_ids, boundary_site_pairs_i32), dim=1).contiguous()
    topology_tensors = (
        word_offsets_i32,
        word_owner_i32,
        word_left_incidence_i32,
        word_right_incidence_i32,
        track_incidence_offsets_i32,
        incidence_boundary_i32,
        boundary_site_pairs_i32,
        active_pairs,
    )
    training_mode = type(certificate_binding) is NativeFixedWordP0TrainingTopologyBinding
    return FixedWordP0TopologyToken(
        word_offsets_i32=word_offsets_i32,
        word_owner_i32=word_owner_i32,
        word_left_incidence_i32=word_left_incidence_i32,
        word_right_incidence_i32=word_right_incidence_i32,
        track_incidence_offsets_i32=track_incidence_offsets_i32,
        incidence_boundary_i32=incidence_boundary_i32,
        boundary_site_pairs_i32=boundary_site_pairs_i32,
        active_boundary_site_pairs_i32=active_pairs,
        certificate_binding=certificate_binding,
        continuous_certificate_digest="" if training_mode else certificate_binding.canonical_digest,
        topology_generation_id=certificate_binding.topology_snapshot_generation,
        boundary_count=boundary_count,
        track_count=track_count,
        site_count=site_count,
        word_count=word_count,
        incidence_count=incidence_count,
        tensor_signatures=_capture_tensor_signatures(topology_tensors),
        training_binding_digest=certificate_binding.canonical_digest if training_mode else "",
        binding_mode=certificate_binding.binding_mode,
        paper_evidence_eligible=certificate_binding.paper_evidence_eligible,
        transfer_jacobian_certified=certificate_binding.transfer_jacobian_certified,
    )


def _fixed_word_p0_config_i32(
    topology: FixedWordP0TopologyToken,
    *,
    node_count: int,
    sample_count: int,
) -> Tensor:
    return torch.tensor(
        [
            topology.boundary_count,
            topology.track_count,
            node_count,
            sample_count,
            topology.site_count,
            topology.word_count,
            topology.incidence_count,
            topology.incidence_count,
        ],
        device=topology.word_offsets_i32.device,
        dtype=torch.int32,
    )


def _fixed_word_p0_config_f32(
    world: FixedWordP0WorldRefreshToken,
    *,
    loss_scale: float,
) -> Tensor:
    return torch.tensor(
        [
            world.config.near,
            world.config.far,
            world.config.invalid_epsilon,
            world.physical_length_epsilon,
            world.cone_tolerance,
            loss_scale,
        ],
        device=world.sites_f32.device,
        dtype=torch.float32,
    )


def refresh_fixed_word_p0_world_token(
    topology: FixedWordP0TopologyToken,
    sites_f32: Tensor,
    site_rgba_f32: Tensor,
    track_ray_coeff_f32: Tensor,
    config: RealRayReplayConfig,
    *,
    physical_length_epsilon: float = 1.0e-8,
    cone_tolerance: float = 1.0e-6,
) -> FixedWordP0WorldRefreshToken:
    """Refresh dynamic world values and derive boundaries from resident sites.

    Strict evaluation compares quantized sites/material/rays with the frozen
    CPU snapshot.  The explicit training capability instead compares only
    immutable sites/rays and permits a new live RGBA tensor per optimizer step;
    it remains transfer/Jacobian uncertified and non-paper. K-block construction
    and accumulation perform no world-value CPU copies.
    """
    _assert_topology_token_current(topology)
    sites_f32 = sites_f32.contiguous()
    site_rgba_f32 = site_rgba_f32.contiguous()
    track_ray_coeff_f32 = track_ray_coeff_f32.contiguous()
    _require_mps_tensor(sites_f32, name="sites_f32", dtype=torch.float32, cols=5)
    _require_mps_tensor(site_rgba_f32, name="site_rgba_f32", dtype=torch.float32, cols=4)
    _require_mps_tensor(
        track_ray_coeff_f32,
        name="track_ray_coeff_f32",
        dtype=torch.float32,
        cols=12,
    )
    if sites_f32.shape[0] != topology.site_count or site_rgba_f32.shape[0] != topology.site_count:
        raise ValueError("world site tensors must match topology.site_count")
    if track_ray_coeff_f32.shape[0] != topology.track_count:
        raise ValueError("track_ray_coeff_f32 rows must match topology.track_count")
    scalars = (
        config.near,
        config.far,
        config.invalid_epsilon,
        physical_length_epsilon,
        cone_tolerance,
    )
    if not all(math.isfinite(value) for value in scalars):
        raise ValueError("world refresh scalar configuration must be finite")
    if config.far <= config.near or config.invalid_epsilon <= 0.0:
        raise ValueError("world refresh requires far > near and positive invalid_epsilon")
    if physical_length_epsilon <= 0.0 or cone_tolerance < 0.0:
        raise ValueError("physical_length_epsilon must be positive and cone_tolerance nonnegative")
    certificate_binding = topology.certificate_binding
    certificate_binding.assert_native_world(
        sites_f32=sites_f32,
        site_rgba_f32=site_rgba_f32,
        track_ray_coeff_f32=track_ray_coeff_f32,
    )
    certificate_binding.assert_replay_interval(near=config.near, far=config.far)
    config_i32 = _fixed_word_p0_config_i32(topology, node_count=1, sample_count=1)
    boundary_f32 = torch.ops.world_foam_lane2_fused_slab_v0.sparse_power_boundary_from_sites_launch_only(
        topology.boundary_site_pairs_i32,
        sites_f32,
        topology.boundary_count,
    )
    mobius_coeff_f32 = torch.ops.world_foam_lane2_fused_slab_v0.fixed_word_p0_sparse_mobius_lower_launch_only(
        boundary_f32,
        track_ray_coeff_f32,
        topology.track_incidence_offsets_i32,
        topology.incidence_boundary_i32,
        config_i32,
        topology.track_count,
        topology.incidence_count,
    )
    world_tensors = (
        sites_f32,
        site_rgba_f32,
        track_ray_coeff_f32,
        boundary_f32,
        mobius_coeff_f32,
        config_i32,
    )
    return FixedWordP0WorldRefreshToken(
        topology=topology,
        sites_f32=sites_f32,
        site_rgba_f32=site_rgba_f32,
        track_ray_coeff_f32=track_ray_coeff_f32,
        boundary_f32=boundary_f32,
        mobius_coeff_f32=mobius_coeff_f32,
        config_i32=config_i32,
        config=config,
        physical_length_epsilon=physical_length_epsilon,
        cone_tolerance=cone_tolerance,
        world_generation_id=certificate_binding.world_snapshot_generation,
        tensor_signatures=_capture_tensor_signatures(world_tensors),
        binding_mode=topology.binding_mode,
        paper_evidence_eligible=topology.paper_evidence_eligible,
        transfer_jacobian_certified=topology.transfer_jacobian_certified,
    )


def prepare_fixed_word_p0_chart_token(
    world: FixedWordP0WorldRefreshToken,
    compiler_node_t_f32: Tensor,
    *,
    chart_index: int,
) -> FixedWordP0ChartToken:
    """Compile one chart selected by the sealed continuous acceptance.

    The runtime ``max(1, |C|+|D|)`` denominator guard remains active in both
    modes. Strict evaluation additionally has continuous transfer/Jacobian
    validity. Material training reuses only the immutable node schedule and
    all-site owner/topology result, so it must not claim approximation-error
    certification or paper eligibility.
    """
    _assert_world_token_current(world)
    compiler_node_t_f32 = compiler_node_t_f32.contiguous()
    _require_mps_tensor(
        compiler_node_t_f32,
        name="compiler_node_t_f32",
        dtype=torch.float32,
        cols=None,
    )
    certificate_binding = world.topology.certificate_binding
    certified_chart = certificate_binding.assert_native_chart(chart_index, compiler_node_t_f32)
    node_count = compiler_node_t_f32.numel()
    if node_count <= 0:
        raise ValueError("compiler_node_t_f32 must be nonempty")
    config_i32 = _fixed_word_p0_config_i32(world.topology, node_count=node_count, sample_count=1)
    config_f32 = _fixed_word_p0_config_f32(world, loss_scale=1.0)
    topology = world.topology
    node_chart_f32 = torch.ops.world_foam_lane2_fused_slab_v0.fixed_word_p0_lie_node_forward_launch_only(
        world.mobius_coeff_f32,
        world.track_ray_coeff_f32,
        compiler_node_t_f32,
        topology.word_offsets_i32,
        topology.word_owner_i32,
        topology.word_left_incidence_i32,
        topology.word_right_incidence_i32,
        topology.track_incidence_offsets_i32,
        world.site_rgba_f32,
        config_i32,
        config_f32,
        topology.track_count,
        node_count,
    )
    chart_tensors = (
        compiler_node_t_f32,
        node_chart_f32,
        config_i32,
        config_f32,
    )
    training_mode = type(certificate_binding) is NativeFixedWordP0TrainingTopologyBinding
    return FixedWordP0ChartToken(
        world=world,
        compiler_node_t_f32=compiler_node_t_f32,
        node_chart_f32=node_chart_f32,
        config_i32=config_i32,
        config_f32=config_f32,
        continuous_certificate_digest="" if training_mode else certificate_binding.canonical_digest,
        chart_index=chart_index,
        chart_generation_id=certified_chart.chart_digest,
        node_count=node_count,
        tensor_signatures=_capture_tensor_signatures(chart_tensors),
        training_binding_digest=certificate_binding.canonical_digest if training_mode else "",
        binding_mode=certificate_binding.binding_mode,
        paper_evidence_eligible=certificate_binding.paper_evidence_eligible,
        transfer_jacobian_certified=certificate_binding.transfer_jacobian_certified,
    )


def prepare_fixed_word_p0_sample_state_token(
    chart: FixedWordP0ChartToken,
    *,
    global_track_count: int,
    global_sample_count: int,
    global_sample_start: int,
    global_sample_end: int,
    global_loss_element_count: int,
    loss_normalization_id: str,
    sample_partition_generation_id: str,
    sample_block_size: int,
) -> FixedWordP0SampleStateToken:
    """Initialize one chart with one global normalization and no sample tape.

    Sample times are intentionally supplied only to each bounded ``K`` block.
    Retaining a chart-local clone here would make replay residency linear in the
    requested frame count even though the times are needed only while building
    that block's interpolation weights.
    """
    _assert_chart_token_current(chart)
    local_track_count = chart.world.topology.track_count
    if global_track_count < local_track_count:
        raise ValueError("global_track_count cannot be smaller than the resident track block")
    if global_sample_count <= 0 or not 0 <= global_sample_start < global_sample_end <= global_sample_count:
        raise ValueError("chart sample range must lie inside the nonempty global sample range")
    if global_loss_element_count != global_track_count * global_sample_count * 3:
        raise ValueError("global loss element count must equal global tracks * global samples * RGB")
    expected_local_element_count = local_track_count * (global_sample_end - global_sample_start) * 3
    if not loss_normalization_id.strip():
        raise ValueError("loss_normalization_id must be nonempty")
    if not sample_partition_generation_id.strip():
        raise ValueError("sample_partition_generation_id must be nonempty")
    if sample_block_size < 1:
        raise ValueError("sample_block_size must be positive")
    local_sample_count = global_sample_end - global_sample_start
    expected_block_count = (local_sample_count + sample_block_size - 1) // sample_block_size
    loss_f32, grad_node_chart_f32, cone_diagnostic_i32 = (
        torch.ops.world_foam_lane2_fused_slab_v0.fixed_word_p0_lie_sample_state_init_launch_only(
            chart.world.site_rgba_f32,
            chart.world.topology.track_count,
            chart.node_count,
        )
    )
    ledger = _FixedWordP0SampleLedger(
        sample_block_size=sample_block_size,
        expected_block_count=expected_block_count,
        next_global_sample_start=global_sample_start,
    )
    ledger.state_tensor_signatures = _capture_tensor_signatures((loss_f32, grad_node_chart_f32, cone_diagnostic_i32))
    return FixedWordP0SampleStateToken(
        chart=chart,
        loss_f32=loss_f32,
        grad_node_chart_f32=grad_node_chart_f32,
        cone_diagnostic_i32=cone_diagnostic_i32,
        loss_normalization_id=loss_normalization_id,
        sample_partition_generation_id=sample_partition_generation_id,
        global_track_count=global_track_count,
        global_sample_count=global_sample_count,
        global_sample_start=global_sample_start,
        global_sample_end=global_sample_end,
        global_loss_element_count=global_loss_element_count,
        expected_local_element_count=expected_local_element_count,
        global_loss_scale=1.0 / float(global_loss_element_count),
        ledger=ledger,
    )


def prepare_fixed_word_p0_sample_block_token(
    sample_state: FixedWordP0SampleStateToken,
    target_rgb_f32: Tensor,
    background_rgb_f32: Tensor,
    *,
    sample_t_f64: Tensor,
    sample_block_id: str,
    global_sample_start: int,
    global_sample_end: int,
) -> FixedWordP0SampleBlockToken:
    """Prepare only K-local sample data; topology and density remain shared."""
    _assert_chart_token_current(sample_state.chart)
    if not sample_block_id.strip():
        raise ValueError("sample_block_id must be nonempty")
    _assert_next_sample_block_range(
        sample_state,
        global_sample_start=global_sample_start,
        global_sample_end=global_sample_end,
    )
    target_rgb_f32 = target_rgb_f32.contiguous()
    background_rgb_f32 = background_rgb_f32.contiguous()
    _require_mps_tensor(background_rgb_f32, name="background_rgb_f32", dtype=torch.float32, cols=None)
    sample_count = global_sample_end - global_sample_start
    chart = sample_state.chart
    track_count = chart.world.topology.track_count
    if target_rgb_f32.device.type != "mps" or target_rgb_f32.dtype != torch.float32:
        raise ValueError("target_rgb_f32 must be float32 on MPS")
    if target_rgb_f32.shape != (track_count, sample_count, 3):
        raise ValueError("target_rgb_f32 must have shape [track_count,sample_count,3]")
    if background_rgb_f32.shape != (3,):
        raise ValueError("background_rgb_f32 must have shape [3]")
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    local_times = torch.as_tensor(sample_t_f64, dtype=torch.float64, device="cpu").reshape(-1)
    if local_times.numel() != sample_count or not bool(torch.isfinite(local_times).all().item()):
        raise ValueError("sample_t_f64 must contain one finite CPU time per local sample")
    sample_weight_result = chart.world.topology.certificate_binding.sample_to_node_weight_result(
        chart.chart_index,
        local_times,
    )
    sample_to_node_f32 = sample_weight_result.weights.to(
        device=target_rgb_f32.device,
        dtype=torch.float32,
    )
    _require_mps_tensor(
        sample_to_node_f32,
        name="sample_to_node_f32",
        dtype=torch.float32,
        cols=sample_state.chart.node_count,
    )
    config_i32 = _fixed_word_p0_config_i32(
        chart.world.topology,
        node_count=chart.node_count,
        sample_count=sample_count,
    )
    config_f32 = _fixed_word_p0_config_f32(
        chart.world,
        loss_scale=sample_state.global_loss_scale,
    )
    sample_block_tensors = (
        sample_to_node_f32,
        target_rgb_f32,
        background_rgb_f32,
        config_i32,
        config_f32,
    )
    return FixedWordP0SampleBlockToken(
        sample_state=sample_state,
        sample_to_node_f32=sample_to_node_f32,
        target_rgb_f32=target_rgb_f32,
        background_rgb_f32=background_rgb_f32,
        config_i32=config_i32,
        config_f32=config_f32,
        sample_block_id=sample_block_id,
        global_sample_start=global_sample_start,
        global_sample_end=global_sample_end,
        sample_count=sample_count,
        element_count=track_count * sample_count * 3,
        sample_weight_evaluation=sample_weight_result.evaluation,
        sample_weight_linear_interactions=sample_weight_result.linear_weight_interactions,
        sample_weight_dense_fallback_interactions=(sample_weight_result.dense_fallback_interactions),
        sample_weight_exact_node_rows=sample_weight_result.exact_node_row_count,
        sample_weight_dense_fallback_rows=sample_weight_result.dense_fallback_row_count,
        tensor_signatures=_capture_tensor_signatures(sample_block_tensors),
    )


def fixed_word_p0_lie_sample_state_init_launch_only(
    sample_state: FixedWordP0SampleStateToken,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return the already initialized resident state without reallocating it."""
    _assert_chart_token_current(sample_state.chart)
    _assert_sample_state_current(sample_state)
    return (
        sample_state.loss_f32,
        sample_state.grad_node_chart_f32,
        sample_state.cone_diagnostic_i32,
    )


def fixed_word_p0_lie_sample_accumulate_launch_only(
    sample_block: FixedWordP0SampleBlockToken,
    sample_state: FixedWordP0SampleStateToken,
) -> Tensor:
    """Accumulate one K block without owning or rebuilding world topology."""
    if sample_block.sample_state is not sample_state:
        raise ValueError("sample block belongs to a different chart/normalization state")
    _assert_chart_token_current(sample_state.chart)
    _assert_sample_state_current(sample_state)
    _assert_sample_block_current(sample_block)
    ledger = sample_state.ledger
    _assert_next_sample_block_range(
        sample_state,
        global_sample_start=sample_block.global_sample_start,
        global_sample_end=sample_block.global_sample_end,
    )
    next_element_count = ledger.accumulated_element_count + sample_block.element_count
    if next_element_count > sample_state.expected_local_element_count:
        raise ValueError("sample blocks exceed the chart-local expected element count")
    chart = sample_state.chart
    prediction_rgb = torch.ops.world_foam_lane2_fused_slab_v0.fixed_word_p0_lie_sample_accumulate_launch_only(
        chart.node_chart_f32,
        sample_block.sample_to_node_f32,
        sample_block.target_rgb_f32,
        sample_block.background_rgb_f32,
        sample_state.loss_f32,
        sample_state.grad_node_chart_f32,
        sample_state.cone_diagnostic_i32,
        sample_block.config_i32,
        sample_block.config_f32,
        chart.world.topology.track_count,
        sample_block.sample_count,
    )
    ledger.consumed_block_count += 1
    ledger.next_global_sample_start = sample_block.global_sample_end
    ledger.accumulated_element_count = next_element_count
    ledger.state_tensor_signatures = _capture_tensor_signatures(_sample_state_tensors(sample_state))
    return prediction_rgb


def fixed_word_p0_lie_sample_accumulate_loss_only_launch_only(
    sample_block: FixedWordP0SampleBlockToken,
    sample_state: FixedWordP0SampleStateToken,
) -> None:
    """Accumulate one K block without allocating a ``[Bp,K,3]`` prediction.

    This is the training hot path.  It deliberately has the same lifecycle,
    arithmetic kernel helper, loss, diagnostics, and node cotangent as
    :func:`fixed_word_p0_lie_sample_accumulate_launch_only`; only the optional
    diagnostic RGB materialization is removed.
    """
    if sample_block.sample_state is not sample_state:
        raise ValueError("sample block belongs to a different chart/normalization state")
    _assert_chart_token_current(sample_state.chart)
    _assert_sample_state_current(sample_state)
    _assert_sample_block_current(sample_block)
    ledger = sample_state.ledger
    _assert_next_sample_block_range(
        sample_state,
        global_sample_start=sample_block.global_sample_start,
        global_sample_end=sample_block.global_sample_end,
    )
    next_element_count = ledger.accumulated_element_count + sample_block.element_count
    if next_element_count > sample_state.expected_local_element_count:
        raise ValueError("sample blocks exceed the chart-local expected element count")
    chart = sample_state.chart
    torch.ops.world_foam_lane2_fused_slab_v0.fixed_word_p0_lie_sample_accumulate_loss_only_launch_only(
        chart.node_chart_f32,
        sample_block.sample_to_node_f32,
        sample_block.target_rgb_f32,
        sample_block.background_rgb_f32,
        sample_state.loss_f32,
        sample_state.grad_node_chart_f32,
        sample_state.cone_diagnostic_i32,
        sample_block.config_i32,
        sample_block.config_f32,
        chart.world.topology.track_count,
        sample_block.sample_count,
    )
    ledger.consumed_block_count += 1
    ledger.next_global_sample_start = sample_block.global_sample_end
    ledger.accumulated_element_count = next_element_count
    ledger.state_tensor_signatures = _capture_tensor_signatures(_sample_state_tensors(sample_state))


def fixed_word_p0_lie_world_grad_init_launch_only(
    world: FixedWordP0WorldRefreshToken,
    *,
    expected_chart_partitions: tuple[tuple[str, int, int], ...],
    global_track_count: int,
    global_sample_count: int,
    global_loss_element_count: int,
    loss_normalization_id: str,
    sample_partition_generation_id: str,
    resident_sample_start: int = 0,
    resident_sample_end: int | None = None,
) -> FixedWordP0WorldGradToken:
    """Allocate shared world adjoints once outside every resident chart loop.

    By default the resident fixed-topology world covers the complete logical
    sample range, preserving the original single-topology contract.  A
    piecewise-topology orchestrator may instead give one nonempty resident
    subrange while retaining the same global ``P * F * 3`` denominator.  The
    topology worlds must still tile that global range outside this token.

    A trainer spanning several spatial blocks must additionally tile their
    global track ranges and reduce their world adjoints under the same
    normalization/partition generation; that orchestration is intentionally
    outside this native token.
    """
    _assert_world_token_current(world)
    topology = world.topology
    if global_track_count < topology.track_count or global_sample_count <= 0:
        raise ValueError("global counts must cover the resident track block and samples")
    if global_loss_element_count != global_track_count * global_sample_count * 3:
        raise ValueError("global loss element count must equal global tracks * global samples * RGB")
    if not loss_normalization_id.strip() or not sample_partition_generation_id.strip():
        raise ValueError("loss normalization and sample partition generation ids must be nonempty")
    normalized_resident_end = global_sample_count if resident_sample_end is None else int(resident_sample_end)
    if not 0 <= resident_sample_start < normalized_resident_end <= global_sample_count:
        raise ValueError("resident sample range must be a nonempty subrange of the global samples")
    expected_chart_ranges = _validate_half_open_partition(
        expected_chart_partitions,
        expected_start=resident_sample_start,
        expected_end=normalized_resident_end,
        name="chart partition",
    )
    expected_chart_ids = frozenset(chart_id for chart_id, _, _ in expected_chart_ranges)
    grad_site_rgba, grad_mobius_coeff, grad_boundary = (
        torch.ops.world_foam_lane2_fused_slab_v0.fixed_word_p0_lie_world_grad_init_launch_only(
            world.site_rgba_f32,
            topology.site_count,
            topology.incidence_count,
            topology.boundary_count,
        )
    )
    ledger = _FixedWordP0WorldGradLedger(
        expected_chart_generation_ids=expected_chart_ids,
        expected_chart_ranges=expected_chart_ranges,
    )
    ledger.grad_tensor_signatures = _capture_tensor_signatures((grad_site_rgba, grad_mobius_coeff, grad_boundary))
    return FixedWordP0WorldGradToken(
        world=world,
        grad_site_rgba_f32=grad_site_rgba,
        grad_mobius_coeff_f32=grad_mobius_coeff,
        grad_boundary_f32=grad_boundary,
        loss_normalization_id=loss_normalization_id,
        sample_partition_generation_id=sample_partition_generation_id,
        global_track_count=global_track_count,
        global_sample_count=global_sample_count,
        global_loss_element_count=global_loss_element_count,
        ledger=ledger,
    )


def fixed_word_p0_lie_material_world_grad_init_launch_only(
    world: FixedWordP0WorldRefreshToken,
    *,
    expected_chart_partitions: tuple[tuple[str, int, int], ...],
    global_track_count: int,
    global_sample_count: int,
    global_loss_element_count: int,
    loss_normalization_id: str,
    sample_partition_generation_id: str,
    resident_sample_start: int = 0,
    resident_sample_end: int | None = None,
) -> FixedWordP0MaterialWorldGradToken:
    """Allocate only the shared material bar for owner-topology training."""

    _assert_world_token_current(world)
    if (
        world.binding_mode != "training_owner_topology_only"
        or world.paper_evidence_eligible
        or world.transfer_jacobian_certified
    ):
        raise ValueError("material-only reverse requires the non-paper owner-topology training capability")
    topology = world.topology
    if global_track_count < topology.track_count or global_sample_count <= 0:
        raise ValueError("global counts must cover the resident track block and samples")
    if global_loss_element_count != global_track_count * global_sample_count * 3:
        raise ValueError("global loss element count must equal global tracks * global samples * RGB")
    if not loss_normalization_id.strip() or not sample_partition_generation_id.strip():
        raise ValueError("loss normalization and sample partition generation ids must be nonempty")
    normalized_resident_end = global_sample_count if resident_sample_end is None else int(resident_sample_end)
    if not 0 <= resident_sample_start < normalized_resident_end <= global_sample_count:
        raise ValueError("resident sample range must be a nonempty subrange of the global samples")
    expected_chart_ranges = _validate_half_open_partition(
        expected_chart_partitions,
        expected_start=resident_sample_start,
        expected_end=normalized_resident_end,
        name="material chart partition",
    )
    grad_site_rgba = torch.ops.world_foam_lane2_fused_slab_v0.fixed_word_p0_lie_material_world_grad_init_launch_only(
        world.site_rgba_f32,
        topology.site_count,
    )
    ledger = _FixedWordP0MaterialWorldGradLedger(
        expected_chart_generation_ids=frozenset(chart_id for chart_id, _, _ in expected_chart_ranges),
        expected_chart_ranges=expected_chart_ranges,
        grad_tensor_signatures=_capture_tensor_signatures((grad_site_rgba,)),
    )
    return FixedWordP0MaterialWorldGradToken(
        world=world,
        grad_site_rgba_f32=grad_site_rgba,
        loss_normalization_id=loss_normalization_id,
        sample_partition_generation_id=sample_partition_generation_id,
        global_track_count=global_track_count,
        global_sample_count=global_sample_count,
        global_loss_element_count=global_loss_element_count,
        ledger=ledger,
    )


def fixed_word_p0_lie_node_vjp_accumulate_launch_only(
    chart: FixedWordP0ChartToken,
    sample_state: FixedWordP0SampleStateToken,
    world_grad: FixedWordP0WorldGradToken,
) -> None:
    """Reverse one chart into shared world adjoints after all its K blocks."""
    if sample_state.chart is not chart:
        raise ValueError("sample state belongs to a different chart generation")
    if world_grad.world is not chart.world:
        raise ValueError("world gradient token belongs to a different world refresh")
    _assert_chart_token_current(chart)
    _assert_sample_state_current(sample_state)
    _assert_world_grad_current(world_grad)
    sample_ledger = sample_state.ledger
    _require_sample_state_ready_for_reverse(sample_state)
    world_ledger = world_grad.ledger
    if sample_state.loss_normalization_id != world_grad.loss_normalization_id:
        raise ValueError("chart sample state uses a different loss normalization id")
    if sample_state.sample_partition_generation_id != world_grad.sample_partition_generation_id:
        raise ValueError("chart sample state uses a different sample partition generation")
    if sample_state.global_track_count != world_grad.global_track_count:
        raise ValueError("chart sample state uses a different global track count")
    if sample_state.global_sample_count != world_grad.global_sample_count:
        raise ValueError("chart sample state uses a different global sample count")
    if sample_state.global_loss_element_count != world_grad.global_loss_element_count:
        raise ValueError("chart sample state uses a different global loss denominator")
    if chart.chart_generation_id not in world_ledger.expected_chart_generation_ids:
        raise ValueError("chart generation is not registered with this world gradient token")
    if (
        chart.chart_generation_id,
        sample_state.global_sample_start,
        sample_state.global_sample_end,
    ) not in world_ledger.expected_chart_ranges:
        raise ValueError("chart sample range does not match the registered global partition")
    if chart.chart_generation_id in world_ledger.reversed_chart_generation_ids:
        raise ValueError("chart generation was already reversed")
    if world_ledger.boundary_finalized:
        raise ValueError("world boundary adjoint is already finalized")
    world = chart.world
    topology = world.topology
    torch.ops.world_foam_lane2_fused_slab_v0.fixed_word_p0_lie_node_vjp_accumulate_launch_only(
        world.mobius_coeff_f32,
        world.track_ray_coeff_f32,
        chart.compiler_node_t_f32,
        topology.word_offsets_i32,
        topology.word_owner_i32,
        topology.word_left_incidence_i32,
        topology.word_right_incidence_i32,
        topology.track_incidence_offsets_i32,
        world.site_rgba_f32,
        chart.node_chart_f32,
        sample_state.grad_node_chart_f32,
        world_grad.grad_site_rgba_f32,
        world_grad.grad_mobius_coeff_f32,
        chart.config_i32,
        chart.config_f32,
        topology.track_count,
        chart.node_count,
    )
    sample_ledger.finalized = True
    world_ledger.reversed_chart_generation_ids.add(chart.chart_generation_id)
    world_ledger.grad_tensor_signatures = _capture_tensor_signatures(_world_grad_tensors(world_grad))


def prepare_kinetic_ragged_p0_lie_sample_block(
    node_chart_f32: Tensor,
    sample_row_i32: Tensor,
    sample_to_node_f32: Tensor,
    target_rgb_f32: Tensor,
    background_rgb_f32: Tensor,
    *,
    loss_scale: float,
    cone_tolerance: float = 1.0e-5,
) -> PreparedKineticRaggedP0LieSampleBlock:
    """Validate row identities on CPU, then seal one bounded MPS sample block.

    ``sample_row_i32`` is required on CPU because row bounds are a cold
    compiler/sampler contract, never a hidden warm-path device sync.  The
    weights are row-local: row ``sample_row_i32[n]`` owns exactly
    ``sample_to_node_f32[n]`` and no rectangular row-by-sample expansion is
    constructed.
    """

    if (
        node_chart_f32.device.type != "mps"
        or node_chart_f32.dtype != torch.float32
        or node_chart_f32.ndim != 3
        or node_chart_f32.shape[2] != 4
        or not node_chart_f32.is_contiguous()
    ):
        raise ValueError("node_chart_f32 must be contiguous MPS float32 [row_count,node_count,4]")
    row_count, node_count = (int(value) for value in node_chart_f32.shape[:2])
    if row_count < 1 or node_count < 1:
        raise ValueError("kinetic ragged sample rows and nodes must be nonempty")
    if not torch.is_tensor(sample_row_i32) or sample_row_i32.device.type != "cpu":
        raise ValueError("sample_row_i32 must be a CPU tensor at the cold validation boundary")
    if (
        sample_row_i32.dtype not in {torch.int32, torch.int64}
        or sample_row_i32.ndim != 1
        or not sample_row_i32.is_contiguous()
    ):
        raise ValueError(
            "sample_row_i32 must be contiguous CPU int32 or int64 [sample_count]"
        )
    sample_count = int(sample_row_i32.numel())
    if sample_count < 1:
        raise ValueError("kinetic ragged sample blocks must be nonempty")
    if (
        int(sample_row_i32.min().item()) < 0
        or int(sample_row_i32.max().item()) >= row_count
    ):
        raise IndexError("sample_row_i32 contains a row outside node_chart_f32")
    for tensor, name, shape in (
        (sample_to_node_f32, "sample_to_node_f32", (sample_count, node_count)),
        (target_rgb_f32, "target_rgb_f32", (sample_count, 3)),
        (background_rgb_f32, "background_rgb_f32", (3,)),
    ):
        if (
            tensor.device != node_chart_f32.device
            or tensor.dtype != torch.float32
            or tuple(tensor.shape) != shape
            or not tensor.is_contiguous()
        ):
            raise ValueError(f"{name} must be contiguous MPS float32 {list(shape)}")
    normalized_loss_scale = float(loss_scale)
    normalized_cone_tolerance = float(cone_tolerance)
    if not math.isfinite(normalized_loss_scale) or normalized_loss_scale <= 0.0:
        raise ValueError("kinetic ragged loss_scale must be finite and positive")
    if not math.isfinite(normalized_cone_tolerance) or normalized_cone_tolerance < 0.0:
        raise ValueError("kinetic ragged cone_tolerance must be finite and nonnegative")
    # Keep the cold row identity in its supplied integer width and narrow only
    # while copying to MPS.  Production supplies CPU int32, so an intermediate
    # CPU int64 clone would add an avoidable 8*N bytes to every streamed block.
    sample_rows_mps = sample_row_i32.detach().to(
        device=node_chart_f32.device,
        dtype=torch.int32,
    )
    config_i32 = torch.tensor(
        [row_count, node_count, sample_count],
        dtype=torch.int32,
        device=node_chart_f32.device,
    )
    config_f32 = torch.tensor(
        [normalized_cone_tolerance, normalized_loss_scale],
        dtype=torch.float32,
        device=node_chart_f32.device,
    )
    tensors = (
        node_chart_f32,
        sample_rows_mps,
        sample_to_node_f32,
        target_rgb_f32,
        background_rgb_f32,
        config_i32,
        config_f32,
    )
    return PreparedKineticRaggedP0LieSampleBlock(
        node_chart_f32=node_chart_f32,
        sample_row_i32=sample_rows_mps,
        sample_to_node_f32=sample_to_node_f32,
        target_rgb_f32=target_rgb_f32,
        background_rgb_f32=background_rgb_f32,
        config_i32=config_i32,
        config_f32=config_f32,
        row_count=row_count,
        node_count=node_count,
        sample_count=sample_count,
        tensor_signatures=_capture_tensor_signatures(tensors),
    )


def _validate_kinetic_ragged_p0_lie_sample_launch(
    prepared: PreparedKineticRaggedP0LieSampleBlock,
    loss_f32: Tensor,
    grad_node_chart_f32: Tensor,
    cone_diagnostic_i32: Tensor,
) -> None:
    if not isinstance(prepared, PreparedKineticRaggedP0LieSampleBlock):
        raise TypeError("kinetic ragged sample launch requires a prepared block")
    tensors = (
        prepared.node_chart_f32,
        prepared.sample_row_i32,
        prepared.sample_to_node_f32,
        prepared.target_rgb_f32,
        prepared.background_rgb_f32,
        prepared.config_i32,
        prepared.config_f32,
    )
    _assert_tensor_signatures_current(
        tensors,
        prepared.tensor_signatures,
        token_name="kinetic ragged sample block",
    )
    for tensor, name, dtype, shape in (
        (loss_f32, "loss_f32", torch.float32, (1,)),
        (
            grad_node_chart_f32,
            "grad_node_chart_f32",
            torch.float32,
            (prepared.row_count, prepared.node_count, 4),
        ),
        (cone_diagnostic_i32, "cone_diagnostic_i32", torch.int32, (3,)),
    ):
        if (
            tensor.device != prepared.node_chart_f32.device
            or tensor.dtype != dtype
            or tuple(tensor.shape) != shape
            or not tensor.is_contiguous()
        ):
            raise ValueError(f"{name} has the wrong launch storage contract")


def kinetic_ragged_p0_lie_sample_accumulate_launch_only(
    prepared: PreparedKineticRaggedP0LieSampleBlock,
    loss_f32: Tensor,
    grad_node_chart_f32: Tensor,
    cone_diagnostic_i32: Tensor,
) -> Tensor:
    """Launch N row-selected samples and return only the optional [N,3] RGB."""

    _validate_kinetic_ragged_p0_lie_sample_launch(
        prepared,
        loss_f32,
        grad_node_chart_f32,
        cone_diagnostic_i32,
    )
    return torch.ops.world_foam_lane2_fused_slab_v0.kinetic_ragged_p0_lie_sample_accumulate_launch_only(
        prepared.node_chart_f32,
        prepared.sample_row_i32,
        prepared.sample_to_node_f32,
        prepared.target_rgb_f32,
        prepared.background_rgb_f32,
        loss_f32,
        grad_node_chart_f32,
        cone_diagnostic_i32,
        prepared.config_i32,
        prepared.config_f32,
        prepared.row_count,
        prepared.node_count,
        prepared.sample_count,
    )


def kinetic_ragged_p0_lie_sample_accumulate_loss_only_launch_only(
    prepared: PreparedKineticRaggedP0LieSampleBlock,
    loss_f32: Tensor,
    grad_node_chart_f32: Tensor,
    cone_diagnostic_i32: Tensor,
) -> None:
    """Launch the no-prediction hot path into caller-owned constant-size state."""

    _validate_kinetic_ragged_p0_lie_sample_launch(
        prepared,
        loss_f32,
        grad_node_chart_f32,
        cone_diagnostic_i32,
    )
    torch.ops.world_foam_lane2_fused_slab_v0.kinetic_ragged_p0_lie_sample_accumulate_loss_only_launch_only(
        prepared.node_chart_f32,
        prepared.sample_row_i32,
        prepared.sample_to_node_f32,
        prepared.target_rgb_f32,
        prepared.background_rgb_f32,
        loss_f32,
        grad_node_chart_f32,
        cone_diagnostic_i32,
        prepared.config_i32,
        prepared.config_f32,
        prepared.row_count,
        prepared.node_count,
        prepared.sample_count,
    )


def kinetic_precompiled_length_p0_lie_node_forward_launch_only(
    word_offsets_i32: Tensor,
    word_owner_i32: Tensor,
    node_physical_length_f32: Tensor,
    site_rgba_f32: Tensor,
    config_i32: Tensor,
    config_f32: Tensor,
    *,
    track_count: int,
    node_count: int,
) -> Tensor:
    """Evaluate a prevalidated kinetic chart without a frame-sized tape.

    This is deliberately a low-level launch boundary.  The caller must bind
    the CSR owner word and node lengths to the same certified compiler
    snapshot before entering this function; only shape/device invariants are
    checked here so the warm launch performs no CPU synchronization.
    """

    track_count = int(track_count)
    node_count = int(node_count)
    if track_count <= 0 or node_count <= 0:
        raise ValueError("track_count and node_count must be positive")
    _require_mps_tensor(word_offsets_i32, name="word_offsets_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(word_owner_i32, name="word_owner_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(site_rgba_f32, name="site_rgba_f32", dtype=torch.float32, cols=4)
    _require_mps_tensor(config_i32, name="config_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(config_f32, name="config_f32", dtype=torch.float32, cols=None)
    word_count = int(word_owner_i32.numel())
    if word_offsets_i32.numel() != track_count + 1:
        raise ValueError("word_offsets_i32 must have shape [track_count+1]")
    if word_count <= 0:
        raise ValueError("word_owner_i32 must be nonempty")
    if (
        node_physical_length_f32.device.type != "mps"
        or node_physical_length_f32.dtype != torch.float32
        or node_physical_length_f32.ndim != 2
        or tuple(node_physical_length_f32.shape) != (node_count, word_count)
        or not node_physical_length_f32.is_contiguous()
    ):
        raise ValueError("node_physical_length_f32 must be contiguous MPS float32 [node_count,word_count]")
    if config_i32.numel() != 4 or config_f32.numel() != 1:
        raise ValueError("kinetic node config must have shapes int32[4] and float32[1]")
    return torch.ops.world_foam_lane2_fused_slab_v0.kinetic_precompiled_length_p0_lie_node_forward_launch_only(
        word_offsets_i32,
        word_owner_i32,
        node_physical_length_f32,
        site_rgba_f32,
        config_i32,
        config_f32,
        track_count,
        node_count,
    )


def kinetic_precompiled_length_p0_lie_node_forward_into_launch_only_v1(
    word_offsets_i32: Tensor,
    word_owner_i32: Tensor,
    node_physical_length_f32: Tensor,
    site_rgba_f32: Tensor,
    config_i32: Tensor,
    config_f32: Tensor,
    node_chart_out_f32: Tensor,
    *,
    track_count: int,
    node_count: int,
) -> None:
    """Fill a caller-retained node chart using the selected forward kernel.

    This deliberately does not call the return-allocating oracle.  The output
    tensor must already be rooted by the caller before this launch boundary, so
    an exception after enqueue cannot hide an internally allocated result.
    """

    track_count = int(track_count)
    node_count = int(node_count)
    if track_count <= 0 or node_count <= 0:
        raise ValueError("track_count and node_count must be positive")
    _require_mps_tensor(word_offsets_i32, name="word_offsets_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(word_owner_i32, name="word_owner_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(site_rgba_f32, name="site_rgba_f32", dtype=torch.float32, cols=4)
    _require_mps_tensor(config_i32, name="config_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(config_f32, name="config_f32", dtype=torch.float32, cols=None)
    word_count = int(word_owner_i32.numel())
    if word_offsets_i32.numel() != track_count + 1:
        raise ValueError("word_offsets_i32 must have shape [track_count+1]")
    if word_count <= 0:
        raise ValueError("word_owner_i32 must be nonempty")
    if (
        node_physical_length_f32.device.type != "mps"
        or node_physical_length_f32.dtype != torch.float32
        or node_physical_length_f32.ndim != 2
        or tuple(node_physical_length_f32.shape) != (node_count, word_count)
        or not node_physical_length_f32.is_contiguous()
    ):
        raise ValueError(
            "node_physical_length_f32 must be contiguous MPS float32 "
            "[node_count,word_count]"
        )
    if (
        node_chart_out_f32.device.type != "mps"
        or node_chart_out_f32.dtype != torch.float32
        or tuple(node_chart_out_f32.shape) != (track_count, node_count, 4)
        or not node_chart_out_f32.is_contiguous()
    ):
        raise ValueError(
            "node_chart_out_f32 must be contiguous MPS float32 "
            "[track_count,node_count,4]"
        )
    if config_i32.numel() != 4 or config_f32.numel() != 1:
        raise ValueError("kinetic node config must have shapes int32[4] and float32[1]")
    if (
        node_chart_out_f32.untyped_storage().data_ptr()
        == site_rgba_f32.untyped_storage().data_ptr()
    ):
        raise ValueError("node_chart_out_f32 must not alias site_rgba_f32")
    returned = torch.ops.world_foam_lane2_fused_slab_v0.kinetic_precompiled_length_p0_lie_node_forward_into_launch_only_v1(
        word_offsets_i32,
        word_owner_i32,
        node_physical_length_f32,
        site_rgba_f32,
        config_i32,
        config_f32,
        node_chart_out_f32,
        track_count,
        node_count,
    )
    if not _same_exact_tensor_view(returned, node_chart_out_f32):
        raise RuntimeError("native forward-into returned a foreign output alias")


def _validate_kinetic_precompiled_length_p0_lie_node_vjp_launch(
    word_offsets_i32: Tensor,
    word_owner_i32: Tensor,
    node_physical_length_f32: Tensor,
    site_rgba_f32: Tensor,
    node_chart_f32: Tensor,
    grad_node_chart_f32: Tensor,
    grad_site_rgba_f32: Tensor,
    config_i32: Tensor,
    config_f32: Tensor,
    *,
    track_count: int,
    node_count: int,
) -> tuple[int, int]:
    track_count = int(track_count)
    node_count = int(node_count)
    if track_count <= 0 or node_count <= 0:
        raise ValueError("track_count and node_count must be positive")
    tensors = (
        word_offsets_i32,
        word_owner_i32,
        node_physical_length_f32,
        site_rgba_f32,
        node_chart_f32,
        grad_node_chart_f32,
        grad_site_rgba_f32,
        config_i32,
        config_f32,
    )
    if any(not tensor.is_contiguous() for tensor in tensors):
        raise ValueError("kinetic node VJP launch tensors must be contiguous")
    _require_mps_tensor(word_offsets_i32, name="word_offsets_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(word_owner_i32, name="word_owner_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(site_rgba_f32, name="site_rgba_f32", dtype=torch.float32, cols=4)
    _require_mps_tensor(grad_site_rgba_f32, name="grad_site_rgba_f32", dtype=torch.float32, cols=4)
    _require_mps_tensor(config_i32, name="config_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(config_f32, name="config_f32", dtype=torch.float32, cols=None)
    word_count = int(word_owner_i32.numel())
    expected_node_shape = (track_count, node_count, 4)
    if word_offsets_i32.numel() != track_count + 1:
        raise ValueError("word_offsets_i32 must have shape [track_count+1]")
    if word_count <= 0:
        raise ValueError("word_owner_i32 must be nonempty")
    if (
        node_physical_length_f32.device.type != "mps"
        or node_physical_length_f32.dtype != torch.float32
        or tuple(node_physical_length_f32.shape) != (node_count, word_count)
    ):
        raise ValueError("node_physical_length_f32 must be MPS float32 [node_count,word_count]")
    for tensor, name in (
        (node_chart_f32, "node_chart_f32"),
        (grad_node_chart_f32, "grad_node_chart_f32"),
    ):
        if tensor.device.type != "mps" or tensor.dtype != torch.float32 or tuple(tensor.shape) != expected_node_shape:
            raise ValueError(f"{name} must be MPS float32 [track_count,node_count,4]")
    if grad_site_rgba_f32.shape != site_rgba_f32.shape:
        raise ValueError("grad_site_rgba_f32 must match site_rgba_f32")
    if config_i32.numel() != 4 or config_f32.numel() != 1:
        raise ValueError("kinetic node config must have shapes int32[4] and float32[1]")
    return track_count, node_count


def kinetic_precompiled_length_p0_lie_node_vjp_accumulate_launch_only(
    word_offsets_i32: Tensor,
    word_owner_i32: Tensor,
    node_physical_length_f32: Tensor,
    site_rgba_f32: Tensor,
    node_chart_f32: Tensor,
    grad_node_chart_f32: Tensor,
    grad_site_rgba_f32: Tensor,
    config_i32: Tensor,
    config_f32: Tensor,
    *,
    track_count: int,
    node_count: int,
) -> tuple[Tensor, Tensor]:
    """Accumulate material bars and return bounded node-length bars.

    The returned geometry tape is ``[node_count, word_count]``.  Its size is
    controlled by the compiled chart, not by the requested temporal sample
    count; lowering those bars to kinetic site parameters is a separate
    stable-stratum compiler VJP.
    """

    track_count, node_count = _validate_kinetic_precompiled_length_p0_lie_node_vjp_launch(
        word_offsets_i32,
        word_owner_i32,
        node_physical_length_f32,
        site_rgba_f32,
        node_chart_f32,
        grad_node_chart_f32,
        grad_site_rgba_f32,
        config_i32,
        config_f32,
        track_count=track_count,
        node_count=node_count,
    )
    return torch.ops.world_foam_lane2_fused_slab_v0.kinetic_precompiled_length_p0_lie_node_vjp_accumulate_launch_only(
        word_offsets_i32,
        word_owner_i32,
        node_physical_length_f32,
        site_rgba_f32,
        node_chart_f32,
        grad_node_chart_f32,
        grad_site_rgba_f32,
        config_i32,
        config_f32,
        track_count,
        node_count,
    )


def kinetic_precompiled_length_p0_lie_material_node_vjp_accumulate_launch_only(
    word_offsets_i32: Tensor,
    word_owner_i32: Tensor,
    node_physical_length_f32: Tensor,
    site_rgba_f32: Tensor,
    node_chart_f32: Tensor,
    grad_node_chart_f32: Tensor,
    grad_site_rgba_f32: Tensor,
    config_i32: Tensor,
    config_f32: Tensor,
    *,
    track_count: int,
    node_count: int,
) -> Tensor:
    """Accumulate only material bars, allocating no ``[J,W]`` length bar."""

    track_count, node_count = _validate_kinetic_precompiled_length_p0_lie_node_vjp_launch(
        word_offsets_i32,
        word_owner_i32,
        node_physical_length_f32,
        site_rgba_f32,
        node_chart_f32,
        grad_node_chart_f32,
        grad_site_rgba_f32,
        config_i32,
        config_f32,
        track_count=track_count,
        node_count=node_count,
    )
    return torch.ops.world_foam_lane2_fused_slab_v0.kinetic_precompiled_length_p0_lie_material_node_vjp_accumulate_launch_only(
        word_offsets_i32,
        word_owner_i32,
        node_physical_length_f32,
        site_rgba_f32,
        node_chart_f32,
        grad_node_chart_f32,
        grad_site_rgba_f32,
        config_i32,
        config_f32,
        track_count,
        node_count,
    )


def prepare_kinetic_fused_direct_full_vjp_v1(
    word_offsets_i32: Tensor,
    word_owner_i32: Tensor,
    source_site_ids_i64: Tensor,
    node_physical_length_f32: Tensor,
    site_rgba_f32: Tensor,
    node_chart_f32: Tensor,
    row_node_time_f32: Tensor,
    row_near_far_f32: Tensor,
    row_ray_coeff_f32: Tensor,
    compact_positions0_f32: Tensor,
    compact_velocities_f32: Tensor,
    compact_weight_coefficients_f32: Tensor,
    *,
    global_site_count: int,
    physical_length_epsilon: float = 1.0e-8,
    minimum_absolute_cut_denominator: float = 1.0e-7,
    minimum_cut_cosine: float = 1.0e-8,
    minimum_coordinate_length: float = 1.0e-8,
    minimum_ray_speed: float = 1.0e-7,
    depth_closure_relative_tolerance: float = 2.0e-5,
    active_tie_relative_tolerance: float = 2.0e-5,
) -> PreparedKineticFusedDirectFullVjpV1:
    """Structurally validate one raw fixed-camera block for the fused VJP.

    CSR and compact-to-global identities may arrive as CPU cold inputs or as
    sealed resident MPS tensors.  Exact MPS int32/int64 layouts are aliased;
    only CPU or dtype-mismatched identities are copied.  All row-local
    floating-point payloads are already resident on MPS.  Equal rank means only
    equal ``J``: every row therefore supplies its own ``[J]`` node times rather
    than borrowing a block-global schedule.

    This low-level function does not accept certificate-shaped strings and does
    not claim provenance admission.  Callers must enter through the sealed
    equal-rank runtime adapter, which binds the live compiler payload, world,
    row-local schedules/domains, geometry, and continuous-owner certificates
    before launch.  Kernel guards are defensive per-thread checks, not a global
    transaction.  This fixed-camera v1 stays source-only and outside the
    selected trainer ABI until rebuild and staged sparse-oracle parity.
    """

    if not torch.is_tensor(site_rgba_f32) or site_rgba_f32.device.type != "mps":
        raise ValueError("fused kinetic floating-point payloads must be resident on MPS")
    device = site_rgba_f32.device
    for tensor, name in (
        (word_offsets_i32, "word_offsets_i32"),
        (word_owner_i32, "word_owner_i32"),
        (source_site_ids_i64, "source_site_ids_i64"),
    ):
        if (
            not torch.is_tensor(tensor)
            or tensor.device.type not in {"cpu", "mps"}
            or (tensor.device.type == "mps" and tensor.device != device)
            or tensor.ndim != 1
            or not tensor.is_contiguous()
        ):
            raise ValueError(f"{name} must be contiguous rank-1 CPU or same-device MPS")
    if word_offsets_i32.dtype not in {torch.int32, torch.int64}:
        raise ValueError("word_offsets_i32 must be int32 or int64")
    if word_owner_i32.dtype not in {torch.int32, torch.int64}:
        raise ValueError("word_owner_i32 must be int32 or int64")
    if source_site_ids_i64.dtype != torch.int64:
        raise ValueError("source_site_ids_i64 must be int64")

    offsets = tuple(int(value) for value in word_offsets_i32.tolist())
    owners = tuple(int(value) for value in word_owner_i32.tolist())
    source_site_ids = tuple(int(value) for value in source_site_ids_i64.tolist())
    row_count = len(offsets) - 1
    word_count = len(owners)
    compact_site_count = len(source_site_ids)
    global_site_count = int(global_site_count)
    if (
        row_count < 1
        or word_count < row_count
        or compact_site_count < 1
        or global_site_count < compact_site_count
    ):
        raise ValueError("fused kinetic CSR/site counts must be positive and bounded")
    if (
        offsets[0] != 0
        or offsets[-1] != word_count
        or any(right <= left for left, right in zip(offsets, offsets[1:], strict=True))
    ):
        raise ValueError("word_offsets_i32 must partition every word into nonempty rows")
    if any(owner < 0 or owner >= compact_site_count for owner in owners):
        raise IndexError("word_owner_i32 leaves the compact owner table")
    if (
        source_site_ids != tuple(sorted(set(source_site_ids)))
        or source_site_ids[0] < 0
        or source_site_ids[-1] >= global_site_count
    ):
        raise ValueError("source_site_ids_i64 must be sorted unique global site ids")

    floating_tensors = (
        node_physical_length_f32,
        site_rgba_f32,
        node_chart_f32,
        row_node_time_f32,
        row_near_far_f32,
        row_ray_coeff_f32,
        compact_positions0_f32,
        compact_velocities_f32,
        compact_weight_coefficients_f32,
    )
    if any(
        not torch.is_tensor(tensor)
        or tensor.device != device
        or tensor.dtype != torch.float32
        or not tensor.is_contiguous()
        for tensor in floating_tensors
    ):
        raise ValueError("fused kinetic floating-point payloads must be contiguous MPS float32")
    if node_physical_length_f32.ndim != 2:
        raise ValueError("node_physical_length_f32 must have shape [node_count,word_count]")
    node_count = int(node_physical_length_f32.shape[0])
    if node_count < 2 or tuple(node_physical_length_f32.shape) != (node_count, word_count):
        raise ValueError("node_physical_length_f32 has the wrong equal-rank shape")
    if tuple(site_rgba_f32.shape) != (compact_site_count, 4):
        raise ValueError("site_rgba_f32 must have shape [compact_site_count,4]")
    if tuple(node_chart_f32.shape) != (row_count, node_count, 4):
        raise ValueError("node_chart_f32 must have shape [row_count,node_count,4]")
    if tuple(row_node_time_f32.shape) != (row_count, node_count):
        raise ValueError("row_node_time_f32 must have shape [row_count,node_count]")
    if tuple(row_near_far_f32.shape) != (row_count, 2):
        raise ValueError("row_near_far_f32 must have shape [row_count,2]")
    if tuple(row_ray_coeff_f32.shape) != (row_count, 12):
        raise ValueError("row_ray_coeff_f32 must have shape [row_count,12]")
    if tuple(compact_positions0_f32.shape) != (compact_site_count, 3):
        raise ValueError("compact_positions0_f32 must have shape [compact_site_count,3]")
    if tuple(compact_velocities_f32.shape) != (compact_site_count, 3):
        raise ValueError("compact_velocities_f32 must have shape [compact_site_count,3]")
    if compact_weight_coefficients_f32.ndim != 2:
        raise ValueError("compact_weight_coefficients_f32 must be rank-2")
    weight_coefficient_count = int(compact_weight_coefficients_f32.shape[1])
    if (
        tuple(compact_weight_coefficients_f32.shape)
        != (compact_site_count, weight_coefficient_count)
        or weight_coefficient_count not in {1, 2, 3}
    ):
        raise ValueError("compact polynomial weights require one, two, or three coefficients")
    threshold_values = (
        float(physical_length_epsilon),
        float(minimum_absolute_cut_denominator),
        float(minimum_ray_speed),
        float(depth_closure_relative_tolerance),
        float(active_tie_relative_tolerance),
        float(minimum_cut_cosine),
        float(minimum_coordinate_length),
    )
    if any(not math.isfinite(value) or value < 0.0 for value in threshold_values):
        raise ValueError("fused kinetic thresholds must be finite and nonnegative")
    if (
        threshold_values[0] == 0.0
        or threshold_values[1] == 0.0
        or threshold_values[2] == 0.0
        or threshold_values[5] == 0.0
        or threshold_values[5] > 1.0
        or threshold_values[6] == 0.0
    ):
        raise ValueError(
            "physical-length, cut-denominator, cut-cosine, coordinate-length, and ray-speed thresholds must be valid and positive"
        )
    threshold_values_f32 = tuple(
        float(torch.tensor(value, dtype=torch.float32).item())
        for value in threshold_values
    )
    if any(
        not math.isfinite(value) or value < 0.0
        for value in threshold_values_f32
    ) or (
        threshold_values_f32[0] == 0.0
        or threshold_values_f32[1] == 0.0
        or threshold_values_f32[2] == 0.0
        or threshold_values_f32[5] == 0.0
        or threshold_values_f32[5] > 1.0
        or threshold_values_f32[6] == 0.0
    ):
        raise ValueError(
            "fused kinetic thresholds must remain valid after float32 conversion"
        )

    def resident_index_tensor(tensor: Tensor, *, dtype: torch.dtype) -> tuple[Tensor, bool]:
        if tensor.device == device and tensor.dtype == dtype:
            return tensor, False
        return tensor.detach().to(device=device, dtype=dtype).contiguous(), True

    resident_indices = (
        resident_index_tensor(word_offsets_i32, dtype=torch.int32),
        resident_index_tensor(word_owner_i32, dtype=torch.int32),
        resident_index_tensor(source_site_ids_i64, dtype=torch.int64),
    )
    prepared_tensors = tuple(value for value, _owned in resident_indices) + floating_tensors + (
        torch.tensor(
            [
                row_count,
                node_count,
                compact_site_count,
                word_count,
                weight_coefficient_count,
                global_site_count,
            ],
            dtype=torch.int32,
            device=device,
        ),
        torch.tensor(threshold_values_f32, dtype=torch.float32, device=device),
    )
    tensor_owned_by_preparer = (
        tuple(owned for _value, owned in resident_indices)
        + (False,) * len(floating_tensors)
        + (True, True)
    )
    retained_logical_tensor_bytes = sum(
        tensor.numel() * tensor.element_size() for tensor in prepared_tensors
    )
    preparer_owned_logical_tensor_bytes = sum(
        tensor.numel() * tensor.element_size()
        for tensor, owned in zip(
            prepared_tensors,
            tensor_owned_by_preparer,
            strict=True,
        )
        if owned
    )
    return PreparedKineticFusedDirectFullVjpV1(
        word_offsets_i32=prepared_tensors[0],
        word_owner_i32=prepared_tensors[1],
        source_site_ids_i64=prepared_tensors[2],
        node_physical_length_f32=prepared_tensors[3],
        site_rgba_f32=prepared_tensors[4],
        node_chart_f32=prepared_tensors[5],
        row_node_time_f32=prepared_tensors[6],
        row_near_far_f32=prepared_tensors[7],
        row_ray_coeff_f32=prepared_tensors[8],
        compact_positions0_f32=prepared_tensors[9],
        compact_velocities_f32=prepared_tensors[10],
        compact_weight_coefficients_f32=prepared_tensors[11],
        config_i32=prepared_tensors[12],
        config_f32=prepared_tensors[13],
        row_count=row_count,
        node_count=node_count,
        word_count=word_count,
        compact_site_count=compact_site_count,
        global_site_count=global_site_count,
        weight_coefficient_count=weight_coefficient_count,
        tensor_owned_by_preparer=tensor_owned_by_preparer,
        retained_logical_tensor_bytes=retained_logical_tensor_bytes,
        preparer_owned_logical_tensor_bytes=preparer_owned_logical_tensor_bytes,
        tensor_signatures=_capture_tensor_signatures(prepared_tensors),
    )


def _kinetic_fused_direct_full_vjp_v1_tensors(
    prepared: PreparedKineticFusedDirectFullVjpV1,
) -> tuple[Tensor, ...]:
    return (
        prepared.word_offsets_i32,
        prepared.word_owner_i32,
        prepared.source_site_ids_i64,
        prepared.node_physical_length_f32,
        prepared.site_rgba_f32,
        prepared.node_chart_f32,
        prepared.row_node_time_f32,
        prepared.row_near_far_f32,
        prepared.row_ray_coeff_f32,
        prepared.compact_positions0_f32,
        prepared.compact_velocities_f32,
        prepared.compact_weight_coefficients_f32,
        prepared.config_i32,
        prepared.config_f32,
    )


def kinetic_fused_direct_full_vjp_validation_status_init_v1(
    reference_f32: Tensor,
) -> Tensor:
    """Allocate the one clear-once status scalar for a split block transaction.

    The coordinator must allocate this once, enqueue every block validation with
    this same tensor, and only then enqueue every guarded accumulation with it.
    Re-clearing between blocks would destroy rejection reasons already ORed by
    an earlier validation dispatch.
    """

    if (
        not isinstance(reference_f32, Tensor)
        or reference_f32.device.type != "mps"
        or reference_f32.dtype != torch.float32
    ):
        raise ValueError("fused kinetic validation status requires an MPS float32 reference")
    return torch.zeros((1,), dtype=torch.int32, device=reference_f32.device)


def kinetic_fused_direct_full_vjp_accumulate_launch_only_v1(
    prepared: PreparedKineticFusedDirectFullVjpV1,
    grad_node_chart_f32: Tensor,
    grad_site_rgba_f32: Tensor,
    grad_global_positions0_f32: Tensor,
    grad_global_velocities_f32: Tensor,
    grad_global_weight_coefficients_f32: Tensor,
    *,
    validation_status_i32: Tensor | None = None,
    launch_phase: str = "combined",
    validate_shared_global_ledgers: bool | None = None,
    finalize_shared_global_ledgers: bool | None = None,
) -> KineticFusedDirectFullVjpResultV1:
    """Fuse ordered-word and kinetic geometry VJPs without a length-bar tape.

    Every gradient output is caller-owned.  The fifth native return is one
    bounded scalar validation receipt.  This v1 ABI is fixed-camera-only: it
    exposes no camera-ray cotangent argument, alias, or output at any public
    layer. ``combined`` is the convenient one-block transaction. For a request
    spanning several active blocks, create one scalar status, call ``validate``
    for every block, call ``accumulate`` for every block, then call ``finalize``
    for every block with the same status. Set each shared-global flag to ``True``
    on exactly the first corresponding phase and ``False`` on the rest: every
    block scans its distinct compact material bar, while the shared global bars
    are scanned once before and once after accumulation. Call
    :meth:`KineticFusedDirectFullVjpResultV1.accepted_bars` before consuming any
    combined result. That check closes the raw device receipt, but the combined
    convenience path does not provide the stronger prepared transaction's
    single-use or abort-settle lifetime contract and remains unpromoted. Split
    transactions are accepted by their coordinator only
    after all finalizers and one final status fence. Device prevalidation
    requires all four output ledgers to be finite and exactly zero before any
    write. The caller must still provide fresh single-use scratch with no
    hidden aliases, quarantine it after any exception or rejected receipt, and
    withhold persistent/optimizer commit until acceptance; the wrapper cannot
    prove those ownership obligations. Any rejected bars are disposable
    scratch and must never reach the optimizer.
    """

    if launch_phase not in {"combined", "validate", "accumulate", "finalize"}:
        raise ValueError(
            "fused kinetic launch_phase must be combined, validate, accumulate, or finalize"
        )
    if (launch_phase == "combined") != (validation_status_i32 is None):
        raise ValueError(
            "combined fused launch owns its status; split phases require one caller-owned shared status"
        )
    if launch_phase == "combined":
        if (
            validate_shared_global_ledgers is not None
            or finalize_shared_global_ledgers is not None
        ):
            raise ValueError(
                "combined fused launch always validates and finalizes all four ledgers"
            )
    elif launch_phase == "validate":
        if not isinstance(validate_shared_global_ledgers, bool):
            raise ValueError(
                "split validation must explicitly choose whether this is the one shared-global-ledger scan"
            )
        if finalize_shared_global_ledgers is not None:
            raise ValueError(
                "finalize_shared_global_ledgers only applies to the finalization phase"
            )
    elif launch_phase == "finalize":
        if validate_shared_global_ledgers is not None:
            raise ValueError(
                "validate_shared_global_ledgers only applies to the validation phase"
            )
        if not isinstance(finalize_shared_global_ledgers, bool):
            raise ValueError(
                "split finalization must explicitly choose whether this is the one shared-global-ledger scan"
            )
    elif (
        validate_shared_global_ledgers is not None
        or finalize_shared_global_ledgers is not None
    ):
        raise ValueError(
            "ledger scan flags apply only to validation or finalization"
        )
    if not isinstance(prepared, PreparedKineticFusedDirectFullVjpV1):
        raise TypeError("fused kinetic full VJP requires its suffixed prepared token")
    if prepared.runtime_status != (
        "raw_fixed_camera_source_only_until_native_rebuild_and_sparse_oracle_parity"
    ):
        raise ValueError("fused kinetic full VJP runtime status changed")
    tensors = _kinetic_fused_direct_full_vjp_v1_tensors(prepared)
    _assert_tensor_signatures_current(
        tensors,
        prepared.tensor_signatures,
        token_name="fused kinetic full VJP v1 token",
    )
    if len(prepared.tensor_owned_by_preparer) != len(tensors):
        raise ValueError("fused kinetic full VJP ownership accounting changed")
    observed_retained_bytes = sum(
        tensor.numel() * tensor.element_size() for tensor in tensors
    )
    observed_owned_bytes = sum(
        tensor.numel() * tensor.element_size()
        for tensor, owned in zip(
            tensors,
            prepared.tensor_owned_by_preparer,
            strict=True,
        )
        if owned
    )
    if (
        prepared.retained_logical_tensor_bytes != observed_retained_bytes
        or prepared.preparer_owned_logical_tensor_bytes != observed_owned_bytes
        or prepared.persistent_frame_tensor_bytes != 0
        or prepared.persistent_sample_tensor_bytes != 0
        or prepared.persistent_target_tensor_bytes != 0
        or prepared.persistent_prediction_tensor_bytes != 0
    ):
        raise ValueError("fused kinetic full VJP retained/owned byte accounting changed")
    device = prepared.site_rgba_f32.device
    outputs = (
        (
            grad_node_chart_f32,
            "grad_node_chart_f32",
            (prepared.row_count, prepared.node_count, 4),
        ),
        (
            grad_site_rgba_f32,
            "grad_site_rgba_f32",
            (prepared.compact_site_count, 4),
        ),
        (
            grad_global_positions0_f32,
            "grad_global_positions0_f32",
            (prepared.global_site_count, 3),
        ),
        (
            grad_global_velocities_f32,
            "grad_global_velocities_f32",
            (prepared.global_site_count, 3),
        ),
        (
            grad_global_weight_coefficients_f32,
            "grad_global_weight_coefficients_f32",
            (prepared.global_site_count, prepared.weight_coefficient_count),
        ),
    )
    for tensor, name, shape in outputs:
        if (
            tensor.device != device
            or tensor.dtype != torch.float32
            or tuple(tensor.shape) != shape
            or not tensor.is_contiguous()
        ):
            raise ValueError(f"{name} must be contiguous MPS float32 {list(shape)}")
    caller_launch_tensors = tuple(tensor for tensor, _, _ in outputs)
    caller_storage_ids = tuple(
        tensor.untyped_storage().data_ptr() for tensor in caller_launch_tensors
    )
    prepared_storage_ids = {
        tensor.untyped_storage().data_ptr() for tensor in tensors
    }
    if len(set(caller_storage_ids)) != len(caller_storage_ids):
        raise ValueError("fused kinetic cotangent inputs and output bars must be storage-distinct")
    if any(storage_id in prepared_storage_ids for storage_id in caller_storage_ids):
        raise ValueError("fused kinetic output bars must not alias prepared primal storage")

    if validation_status_i32 is not None:
        if (
            not isinstance(validation_status_i32, Tensor)
            or validation_status_i32.device != device
            or validation_status_i32.dtype != torch.int32
            or tuple(validation_status_i32.shape) != (1,)
            or not validation_status_i32.is_contiguous()
            or validation_status_i32.numel() * validation_status_i32.element_size() != 4
        ):
            raise ValueError(
                "split fused kinetic phases require one contiguous MPS int32 [1] status"
            )
        status_storage_id = validation_status_i32.untyped_storage().data_ptr()
        if status_storage_id in caller_storage_ids or status_storage_id in prepared_storage_ids:
            raise ValueError(
                "fused kinetic shared status must not alias inputs or output bars"
            )

    try:
        native_namespace = torch.ops.world_foam_lane2_fused_slab_v0
        if launch_phase == "combined":
            native_op = native_namespace.kinetic_fused_direct_full_vjp_accumulate_launch_only_v1
        elif launch_phase == "validate":
            native_op = native_namespace.kinetic_fused_direct_full_vjp_validate_shared_status_launch_only_v1
        elif launch_phase == "accumulate":
            native_op = native_namespace.kinetic_fused_direct_full_vjp_accumulate_shared_status_launch_only_v1
        else:
            native_op = native_namespace.kinetic_fused_direct_full_vjp_finalize_shared_status_launch_only_v1
    except AttributeError as exc:
        raise RuntimeError(
            "fused kinetic full VJP v1 is source-only until the native variant is rebuilt and checked against the staged sparse oracle"
        ) from exc

    native_args = (
        *tensors[:12],
        grad_node_chart_f32,
        grad_site_rgba_f32,
        grad_global_positions0_f32,
        grad_global_velocities_f32,
        grad_global_weight_coefficients_f32,
        prepared.config_i32,
        prepared.config_f32,
    )
    if launch_phase == "combined":
        native_result = native_op(
            *native_args,
            prepared.row_count,
            prepared.node_count,
        )
    elif launch_phase == "validate":
        returned_status = native_op(
            *native_args,
            validation_status_i32,
            validate_shared_global_ledgers,
            prepared.row_count,
            prepared.node_count,
        )
        if not isinstance(returned_status, Tensor) or not _same_exact_tensor_view(
            returned_status,
            validation_status_i32,
        ):
            raise RuntimeError(
                "fused kinetic validation phase must return the caller-owned shared status"
            )
        native_result = None
    elif launch_phase == "accumulate":
        native_result = native_op(
            *native_args,
            validation_status_i32,
            prepared.row_count,
            prepared.node_count,
        )
    else:
        native_result = native_op(
            grad_site_rgba_f32,
            grad_global_positions0_f32,
            grad_global_velocities_f32,
            grad_global_weight_coefficients_f32,
            validation_status_i32,
            finalize_shared_global_ledgers,
        )

    expected_aliases = (
        grad_site_rgba_f32,
        grad_global_positions0_f32,
        grad_global_velocities_f32,
        grad_global_weight_coefficients_f32,
    )
    if launch_phase != "validate":
        if (
            not isinstance(native_result, tuple)
            or len(native_result) != len(expected_aliases) + 1
            or any(
                not isinstance(returned, Tensor)
                or not _same_exact_tensor_view(returned, expected)
                for returned, expected in zip(
                    native_result[: len(expected_aliases)],
                    expected_aliases,
                    strict=True,
                )
            )
        ):
            raise RuntimeError(
                "fused kinetic full VJP native return must alias all four caller-owned bars"
            )
        returned_status = native_result[-1]
        if launch_phase in {"accumulate", "finalize"} and not _same_exact_tensor_view(
            returned_status,
            validation_status_i32,
        ):
            raise RuntimeError(
                "fused kinetic split phase must return the caller-owned shared status"
            )
        validation_status_i32 = returned_status

    if (
        not isinstance(validation_status_i32, Tensor)
        or validation_status_i32.device != device
        or validation_status_i32.dtype != torch.int32
        or tuple(validation_status_i32.shape) != (1,)
        or not validation_status_i32.is_contiguous()
        or validation_status_i32.numel() * validation_status_i32.element_size() != 4
        or validation_status_i32.untyped_storage().data_ptr() in caller_storage_ids
        or validation_status_i32.untyped_storage().data_ptr() in prepared_storage_ids
    ):
        raise RuntimeError(
            "fused kinetic full VJP native validation receipt must be one distinct MPS int32 scalar"
        )
    return KineticFusedDirectFullVjpResultV1(
        grad_site_rgba_f32=grad_site_rgba_f32,
        grad_global_positions0_f32=grad_global_positions0_f32,
        grad_global_velocities_f32=grad_global_velocities_f32,
        grad_global_weight_coefficients_f32=grad_global_weight_coefficients_f32,
        validation_status_i32=validation_status_i32,
        accumulation_enqueued=launch_phase in {"combined", "accumulate"},
        finalization_enqueued=launch_phase in {"combined", "finalize"},
        shared_status_reused=launch_phase != "combined",
    )


def prepare_kinetic_fused_union_full_vjp_v2(
    word_offsets_i32: Tensor,
    word_owner_i32: Tensor,
    source_site_ids_i64: Tensor,
    compact_to_geometry_output_i64: Tensor,
    geometry_output_source_site_ids_i64: Tensor,
    node_physical_length_f32: Tensor,
    site_rgba_f32: Tensor,
    node_chart_f32: Tensor,
    row_node_time_f32: Tensor,
    row_near_far_f32: Tensor,
    row_ray_coeff_f32: Tensor,
    compact_positions0_f32: Tensor,
    compact_velocities_f32: Tensor,
    compact_weight_coefficients_f32: Tensor,
    *,
    global_site_count: int,
    physical_length_epsilon: float = 1.0e-8,
    minimum_absolute_cut_denominator: float = 1.0e-7,
    minimum_cut_cosine: float = 1.0e-8,
    minimum_coordinate_length: float = 1.0e-8,
    minimum_ray_speed: float = 1.0e-7,
    depth_closure_relative_tolerance: float = 2.0e-5,
    active_tie_relative_tolerance: float = 2.0e-5,
) -> PreparedKineticFusedUnionFullVjpV2:
    """Cold-prepare one exact ``P_b=P_U Q_b`` union-local block.

    The unchanged v1 preparer remains the structural oracle for every primal.
    This wrapper adds two independent int64 identities and a seven-element
    config carrying both global ``S`` and request-union ``U``. Cross-block
    duplicate union destinations are legal; duplicate source ids inside one
    compact block remain rejected by v1.
    """

    direct_v1_oracle = prepare_kinetic_fused_direct_full_vjp_v1(
        word_offsets_i32,
        word_owner_i32,
        source_site_ids_i64,
        node_physical_length_f32,
        site_rgba_f32,
        node_chart_f32,
        row_node_time_f32,
        row_near_far_f32,
        row_ray_coeff_f32,
        compact_positions0_f32,
        compact_velocities_f32,
        compact_weight_coefficients_f32,
        global_site_count=global_site_count,
        physical_length_epsilon=physical_length_epsilon,
        minimum_absolute_cut_denominator=minimum_absolute_cut_denominator,
        minimum_cut_cosine=minimum_cut_cosine,
        minimum_coordinate_length=minimum_coordinate_length,
        minimum_ray_speed=minimum_ray_speed,
        depth_closure_relative_tolerance=depth_closure_relative_tolerance,
        active_tie_relative_tolerance=active_tie_relative_tolerance,
    )
    for tensor, name in (
        (compact_to_geometry_output_i64, "compact_to_geometry_output_i64"),
        (
            geometry_output_source_site_ids_i64,
            "geometry_output_source_site_ids_i64",
        ),
    ):
        if (
            not torch.is_tensor(tensor)
            or tensor.device.type not in {"cpu", "mps"}
            or (
                tensor.device.type == "mps"
                and tensor.device != direct_v1_oracle.site_rgba_f32.device
            )
            or tensor.dtype != torch.int64
            or tensor.ndim != 1
            or not tensor.is_contiguous()
        ):
            raise ValueError(
                f"{name} must be contiguous rank-1 int64 on CPU or the launch MPS device"
            )
    cold_identity_storage = tuple(
        (tensor.device, tensor.untyped_storage().data_ptr())
        for tensor in (
            source_site_ids_i64,
            compact_to_geometry_output_i64,
            geometry_output_source_site_ids_i64,
        )
    )
    if len(set(cold_identity_storage)) != len(cold_identity_storage):
        raise ValueError(
            "union-v2 cold compact/global, compact/union, and union/global identities must be storage-distinct"
        )
    compact_to_output = tuple(
        int(value) for value in compact_to_geometry_output_i64.tolist()
    )
    output_source_ids = tuple(
        int(value) for value in geometry_output_source_site_ids_i64.tolist()
    )
    source_ids = tuple(int(value) for value in source_site_ids_i64.tolist())
    union_site_count = len(output_source_ids)
    if (
        len(compact_to_output) != direct_v1_oracle.compact_site_count
        or union_site_count < 1
        or union_site_count > direct_v1_oracle.global_site_count
    ):
        raise ValueError("union-v2 mapping counts are inconsistent with compact/global S")
    if output_source_ids != tuple(sorted(set(output_source_ids))):
        raise ValueError(
            "geometry_output_source_site_ids_i64 must be sorted unique global ids"
        )
    if output_source_ids[0] < 0 or output_source_ids[-1] >= global_site_count:
        raise IndexError("union-v2 output source ids leave the global world")
    if any(index < 0 or index >= union_site_count for index in compact_to_output):
        raise IndexError("compact_to_geometry_output_i64 leaves the request union")
    if tuple(output_source_ids[index] for index in compact_to_output) != source_ids:
        raise ValueError(
            "union-v2 requires union_ids[compact_to_output[k]] == source_ids[k]"
        )

    device = direct_v1_oracle.site_rgba_f32.device

    def resident_map(tensor: Tensor) -> tuple[Tensor, bool]:
        if tensor.device == device:
            return tensor, False
        return tensor.detach().to(device=device, dtype=torch.int64).contiguous(), True

    resident_compact_to_output, compact_map_owned = resident_map(
        compact_to_geometry_output_i64
    )
    resident_output_source_ids, output_ids_owned = resident_map(
        geometry_output_source_site_ids_i64
    )
    direct_tensors = _kinetic_fused_direct_full_vjp_v1_tensors(direct_v1_oracle)
    identity_tensors = (
        direct_v1_oracle.source_site_ids_i64,
        resident_compact_to_output,
        resident_output_source_ids,
    )
    identity_storage = tuple(
        tensor.untyped_storage().data_ptr() for tensor in identity_tensors
    )
    if len(set(identity_storage)) != len(identity_storage):
        raise ValueError(
            "union-v2 compact/global, compact/union, and union/global identities must be storage-distinct"
        )
    config_i32 = torch.tensor(
        [
            direct_v1_oracle.row_count,
            direct_v1_oracle.node_count,
            direct_v1_oracle.compact_site_count,
            direct_v1_oracle.word_count,
            direct_v1_oracle.weight_coefficient_count,
            direct_v1_oracle.global_site_count,
            union_site_count,
        ],
        dtype=torch.int32,
        device=device,
    )
    transfer_predecessors = tuple(
        source
        for source, copied in (
            (compact_to_geometry_output_i64, compact_map_owned),
            (geometry_output_source_site_ids_i64, output_ids_owned),
        )
        if copied
    )
    v2_tensors = (
        *direct_tensors,
        resident_compact_to_output,
        resident_output_source_ids,
        config_i32,
        *transfer_predecessors,
    )
    retained_bytes = sum(
        tensor.numel() * tensor.element_size() for tensor in v2_tensors
    )
    owned_bytes = (
        direct_v1_oracle.preparer_owned_logical_tensor_bytes
        + (resident_compact_to_output.numel() * 8 if compact_map_owned else 0)
        + (resident_output_source_ids.numel() * 8 if output_ids_owned else 0)
        + config_i32.numel() * config_i32.element_size()
    )
    return PreparedKineticFusedUnionFullVjpV2(
        direct_v1_oracle=direct_v1_oracle,
        compact_to_geometry_output_i64=resident_compact_to_output,
        geometry_output_source_site_ids_i64=resident_output_source_ids,
        compact_to_geometry_output_transfer_source=compact_to_geometry_output_i64,
        geometry_output_source_site_ids_transfer_source=(
            geometry_output_source_site_ids_i64
        ),
        config_i32=config_i32,
        global_site_count=direct_v1_oracle.global_site_count,
        union_site_count=union_site_count,
        mapping_tensor_owned_by_preparer=(compact_map_owned, output_ids_owned),
        transfer_predecessor_logical_tensor_bytes=sum(
            tensor.numel() * tensor.element_size()
            for tensor in transfer_predecessors
        ),
        transfer_predecessor_release_requires_proven_fence=bool(
            transfer_predecessors
        ),
        retained_logical_tensor_bytes=retained_bytes,
        preparer_owned_logical_tensor_bytes=owned_bytes,
        tensor_signatures=_capture_tensor_signatures(v2_tensors),
    )


def _kinetic_fused_union_full_vjp_v2_tensors(
    prepared: PreparedKineticFusedUnionFullVjpV2,
) -> tuple[Tensor, ...]:
    if len(prepared.mapping_tensor_owned_by_preparer) != 2:
        raise ValueError("union-v2 map ownership arity changed")
    copied_predecessors = tuple(
        source
        for source, copied in (
            (
                prepared.compact_to_geometry_output_transfer_source,
                prepared.mapping_tensor_owned_by_preparer[0],
            ),
            (
                prepared.geometry_output_source_site_ids_transfer_source,
                prepared.mapping_tensor_owned_by_preparer[1],
            ),
        )
        if copied
    )
    return (
        *_kinetic_fused_direct_full_vjp_v1_tensors(prepared.direct_v1_oracle),
        prepared.compact_to_geometry_output_i64,
        prepared.geometry_output_source_site_ids_i64,
        prepared.config_i32,
        *copied_predecessors,
    )


def kinetic_fused_union_full_vjp_validation_status_init_v2(
    reference_f32: Tensor,
) -> Tensor:
    """Allocate one clear-once status for the suffixed union-v2 transaction."""

    return kinetic_fused_direct_full_vjp_validation_status_init_v1(reference_f32)


def kinetic_fused_union_full_vjp_accumulate_launch_only_v2(
    prepared: PreparedKineticFusedUnionFullVjpV2,
    grad_node_chart_f32: Tensor,
    grad_site_rgba_f32: Tensor,
    grad_union_positions0_f32: Tensor,
    grad_union_velocities_f32: Tensor,
    grad_union_weight_coefficients_f32: Tensor,
    *,
    validation_status_i32: Tensor,
    launch_phase: str,
    validate_shared_union_ledgers: bool | None = None,
    finalize_shared_union_ledgers: bool | None = None,
) -> KineticFusedUnionFullVjpResultV2:
    """Launch one union-v2 split phase without persistent commit.

    The raw route intentionally has no combined shortcut. A coordinator must
    validate every admitted block before any accumulation under the current
    all-block policy, or later use bounded transaction-local batches whose
    ledgers remain uncommitted until full-manifest acceptance.
    """

    if not isinstance(prepared, PreparedKineticFusedUnionFullVjpV2):
        raise TypeError("union-v2 requires its exact suffixed prepared token")
    if launch_phase not in {"validate", "accumulate", "finalize"}:
        raise ValueError("union-v2 launch_phase must be validate, accumulate, or finalize")
    if launch_phase == "validate":
        if not isinstance(validate_shared_union_ledgers, bool):
            raise ValueError("union-v2 validation requires an explicit shared-ledger flag")
        if finalize_shared_union_ledgers is not None:
            raise ValueError("finalize_shared_union_ledgers is finalizer-only")
    elif launch_phase == "finalize":
        if not isinstance(finalize_shared_union_ledgers, bool):
            raise ValueError("union-v2 finalization requires an explicit shared-ledger flag")
        if validate_shared_union_ledgers is not None:
            raise ValueError("validate_shared_union_ledgers is validation-only")
    elif (
        validate_shared_union_ledgers is not None
        or finalize_shared_union_ledgers is not None
    ):
        raise ValueError("union-v2 ledger flags do not apply to accumulation")
    if prepared.runtime_status != (
        "raw_union_v2_source_only_until_native_rebuild_v1_sparse_parity_and_allocator_evidence"
    ):
        raise ValueError("union-v2 runtime status changed")
    tensors = _kinetic_fused_union_full_vjp_v2_tensors(prepared)
    _assert_tensor_signatures_current(
        tensors,
        prepared.tensor_signatures,
        token_name="kinetic fused union full VJP v2 token",
    )
    observed_bytes = sum(
        tensor.numel() * tensor.element_size() for tensor in tensors
    )
    transfer_predecessors = tuple(
        source
        for source, copied in (
            (
                prepared.compact_to_geometry_output_transfer_source,
                prepared.mapping_tensor_owned_by_preparer[0],
            ),
            (
                prepared.geometry_output_source_site_ids_transfer_source,
                prepared.mapping_tensor_owned_by_preparer[1],
            ),
        )
        if copied
    )
    observed_predecessor_bytes = sum(
        tensor.numel() * tensor.element_size()
        for tensor in transfer_predecessors
    )
    if (
        observed_bytes != prepared.retained_logical_tensor_bytes
        or len(prepared.mapping_tensor_owned_by_preparer) != 2
        or observed_predecessor_bytes
        != prepared.transfer_predecessor_logical_tensor_bytes
        or prepared.transfer_predecessor_release_requires_proven_fence
        != bool(transfer_predecessors)
    ):
        raise ValueError("union-v2 retained logical byte accounting changed")

    direct = prepared.direct_v1_oracle
    device = direct.site_rgba_f32.device
    outputs = (
        (
            grad_node_chart_f32,
            "grad_node_chart_f32",
            (direct.row_count, direct.node_count, 4),
        ),
        (
            grad_site_rgba_f32,
            "grad_site_rgba_f32",
            (direct.compact_site_count, 4),
        ),
        (
            grad_union_positions0_f32,
            "grad_union_positions0_f32",
            (prepared.union_site_count, 3),
        ),
        (
            grad_union_velocities_f32,
            "grad_union_velocities_f32",
            (prepared.union_site_count, 3),
        ),
        (
            grad_union_weight_coefficients_f32,
            "grad_union_weight_coefficients_f32",
            (prepared.union_site_count, direct.weight_coefficient_count),
        ),
    )
    for tensor, name, shape in outputs:
        if (
            not isinstance(tensor, Tensor)
            or tensor.device != device
            or tensor.dtype != torch.float32
            or tuple(tensor.shape) != shape
            or not tensor.is_contiguous()
        ):
            raise ValueError(f"{name} must be contiguous MPS float32 {list(shape)}")
    if (
        not isinstance(validation_status_i32, Tensor)
        or validation_status_i32.device != device
        or validation_status_i32.dtype != torch.int32
        or tuple(validation_status_i32.shape) != (1,)
        or not validation_status_i32.is_contiguous()
    ):
        raise ValueError("union-v2 phases require one caller-owned MPS int32 [1] status")
    caller_tensors = tuple(tensor for tensor, _, _ in outputs)
    caller_storage = tuple(
        tensor.untyped_storage().data_ptr() for tensor in caller_tensors
    )
    prepared_storage = {
        tensor.untyped_storage().data_ptr() for tensor in tensors
    }
    if len(set(caller_storage)) != len(caller_storage):
        raise ValueError("union-v2 cotangent and output tensors must be storage-distinct")
    if set(caller_storage) & prepared_storage:
        raise ValueError("union-v2 output bars must not alias prepared inputs")
    status_storage = validation_status_i32.untyped_storage().data_ptr()
    if status_storage in set(caller_storage) | prepared_storage:
        raise ValueError("union-v2 status must not alias inputs or output bars")

    try:
        namespace = torch.ops.world_foam_lane2_fused_slab_v0
        native_op = getattr(
            namespace,
            f"kinetic_fused_union_full_vjp_{launch_phase}_shared_status_launch_only_v2",
        )
    except AttributeError as exc:
        raise RuntimeError(
            "union-v2 is source-only until the native variant is rebuilt and checked against v1/staged oracles"
        ) from exc
    if launch_phase == "finalize":
        native_result = native_op(
            grad_site_rgba_f32,
            grad_union_positions0_f32,
            grad_union_velocities_f32,
            grad_union_weight_coefficients_f32,
            validation_status_i32,
            finalize_shared_union_ledgers,
            prepared.union_site_count,
        )
    else:
        native_args = (
            direct.word_offsets_i32,
            direct.word_owner_i32,
            direct.source_site_ids_i64,
            prepared.compact_to_geometry_output_i64,
            prepared.geometry_output_source_site_ids_i64,
            direct.node_physical_length_f32,
            direct.site_rgba_f32,
            direct.node_chart_f32,
            direct.row_node_time_f32,
            direct.row_near_far_f32,
            direct.row_ray_coeff_f32,
            direct.compact_positions0_f32,
            direct.compact_velocities_f32,
            direct.compact_weight_coefficients_f32,
            grad_node_chart_f32,
            grad_site_rgba_f32,
            grad_union_positions0_f32,
            grad_union_velocities_f32,
            grad_union_weight_coefficients_f32,
            prepared.config_i32,
            direct.config_f32,
            validation_status_i32,
        )
        if launch_phase == "validate":
            native_result = native_op(
                *native_args,
                validate_shared_union_ledgers,
                prepared.global_site_count,
                prepared.union_site_count,
                direct.row_count,
                direct.node_count,
            )
        else:
            native_result = native_op(
                *native_args,
                prepared.global_site_count,
                prepared.union_site_count,
                direct.row_count,
                direct.node_count,
            )
    expected_aliases = (
        grad_site_rgba_f32,
        grad_union_positions0_f32,
        grad_union_velocities_f32,
        grad_union_weight_coefficients_f32,
    )
    if launch_phase == "validate":
        returned_status = native_result
        if not isinstance(returned_status, Tensor) or not _same_exact_tensor_view(
            returned_status, validation_status_i32
        ):
            raise RuntimeError("union-v2 validation must return the shared status alias")
    else:
        if (
            not isinstance(native_result, tuple)
            or len(native_result) != 5
            or any(
                not isinstance(returned, Tensor)
                or not _same_exact_tensor_view(returned, expected)
                for returned, expected in zip(
                    native_result[:4], expected_aliases, strict=True
                )
            )
        ):
            raise RuntimeError("union-v2 native phase must alias all four output bars")
        returned_status = native_result[4]
        if not _same_exact_tensor_view(returned_status, validation_status_i32):
            raise RuntimeError("union-v2 native phase must return the shared status alias")
    return KineticFusedUnionFullVjpResultV2(
        grad_site_rgba_f32=grad_site_rgba_f32,
        grad_union_positions0_f32=grad_union_positions0_f32,
        grad_union_velocities_f32=grad_union_velocities_f32,
        grad_union_weight_coefficients_f32=grad_union_weight_coefficients_f32,
        validation_status_i32=validation_status_i32,
        accumulation_enqueued=launch_phase == "accumulate",
        finalization_enqueued=launch_phase == "finalize",
    )


def fixed_word_p0_lie_material_node_vjp_accumulate_launch_only(
    chart: FixedWordP0ChartToken,
    sample_state: FixedWordP0SampleStateToken,
    world_grad: FixedWordP0MaterialWorldGradToken,
) -> None:
    """Reverse one chart into RGBA only, with no geometry-side writes."""

    if sample_state.chart is not chart:
        raise ValueError("sample state belongs to a different chart generation")
    if world_grad.world is not chart.world:
        raise ValueError("material world gradient token belongs to a different world refresh")
    _assert_chart_token_current(chart)
    _assert_sample_state_current(sample_state)
    _assert_material_world_grad_current(world_grad)
    _require_sample_state_ready_for_reverse(sample_state)
    world_ledger = world_grad.ledger
    if sample_state.loss_normalization_id != world_grad.loss_normalization_id:
        raise ValueError("chart sample state uses a different loss normalization id")
    if sample_state.sample_partition_generation_id != world_grad.sample_partition_generation_id:
        raise ValueError("chart sample state uses a different sample partition generation")
    if sample_state.global_track_count != world_grad.global_track_count:
        raise ValueError("chart sample state uses a different global track count")
    if sample_state.global_sample_count != world_grad.global_sample_count:
        raise ValueError("chart sample state uses a different global sample count")
    if sample_state.global_loss_element_count != world_grad.global_loss_element_count:
        raise ValueError("chart sample state uses a different global loss denominator")
    if chart.chart_generation_id not in world_ledger.expected_chart_generation_ids:
        raise ValueError("chart generation is not registered with this material world gradient token")
    if (
        chart.chart_generation_id,
        sample_state.global_sample_start,
        sample_state.global_sample_end,
    ) not in world_ledger.expected_chart_ranges:
        raise ValueError("chart sample range does not match the registered material partition")
    if chart.chart_generation_id in world_ledger.reversed_chart_generation_ids:
        raise ValueError("chart generation was already reversed")
    if world_ledger.finalized:
        raise ValueError("material world adjoint is already finalized")
    world = chart.world
    topology = world.topology
    torch.ops.world_foam_lane2_fused_slab_v0.fixed_word_p0_lie_material_node_vjp_accumulate_launch_only(
        world.mobius_coeff_f32,
        world.track_ray_coeff_f32,
        chart.compiler_node_t_f32,
        topology.word_offsets_i32,
        topology.word_owner_i32,
        topology.word_left_incidence_i32,
        topology.word_right_incidence_i32,
        topology.track_incidence_offsets_i32,
        world.site_rgba_f32,
        chart.node_chart_f32,
        sample_state.grad_node_chart_f32,
        world_grad.grad_site_rgba_f32,
        chart.config_i32,
        chart.config_f32,
        topology.track_count,
        chart.node_count,
    )
    sample_state.ledger.finalized = True
    world_ledger.reversed_chart_generation_ids.add(chart.chart_generation_id)
    world_ledger.grad_tensor_signatures = _capture_tensor_signatures((world_grad.grad_site_rgba_f32,))


def fixed_word_p0_lie_material_world_grad_finalize_launch_only(
    world_grad: FixedWordP0MaterialWorldGradToken,
) -> Tensor:
    """Seal and return the existing RGBA bar without another allocation."""

    _assert_material_world_grad_current(world_grad)
    ledger = world_grad.ledger
    if ledger.finalized:
        raise ValueError("material world adjoint was already finalized")
    if ledger.reversed_chart_generation_ids != ledger.expected_chart_generation_ids:
        raise ValueError("material world adjoint cannot finalize with missing chart reversals")
    ledger.finalized = True
    return world_grad.grad_site_rgba_f32


def fixed_word_p0_sparse_mobius_boundary_finalize_launch_only(
    world_grad: FixedWordP0WorldGradToken,
) -> Tensor:
    """Finalize shared incidence bars once after all certified chart tokens."""
    _assert_world_grad_current(world_grad)
    ledger = world_grad.ledger
    if ledger.boundary_finalized:
        raise ValueError("world boundary adjoint was already finalized")
    if ledger.reversed_chart_generation_ids != ledger.expected_chart_generation_ids:
        raise ValueError("world boundary adjoint cannot finalize with missing chart reversals")
    world = world_grad.world
    topology = world.topology
    result = torch.ops.world_foam_lane2_fused_slab_v0.fixed_word_p0_sparse_mobius_boundary_finalize_launch_only(
        world.track_ray_coeff_f32,
        topology.track_incidence_offsets_i32,
        topology.incidence_boundary_i32,
        world_grad.grad_mobius_coeff_f32,
        world_grad.grad_boundary_f32,
        world.config_i32,
        topology.track_count,
    )
    ledger.boundary_finalized = True
    ledger.grad_tensor_signatures = _capture_tensor_signatures(_world_grad_tensors(world_grad))
    return result


def fixed_word_p0_site_geometry_finalize_launch_only(
    world_grad: FixedWordP0WorldGradToken,
) -> Tensor:
    """Scatter through the exact sites/pairs that generated forward boundaries."""
    _assert_world_grad_current(world_grad)
    ledger = world_grad.ledger
    if not ledger.boundary_finalized:
        raise ValueError("site geometry cannot finalize before the boundary adjoint")
    if ledger.site_finalized:
        raise ValueError("site geometry was already finalized")
    world = world_grad.world
    grad_sites = torch.ops.world_foam_lane2_fused_slab_v0.sparse_power_boundary_vjp_to_sites_launch_only(
        world.topology.active_boundary_site_pairs_i32,
        world.sites_f32,
        world_grad.grad_boundary_f32,
    )
    ledger.site_finalized = True
    return grad_sites


def fixed_word_p0_compiled_lie_transfer_mse_vjp_sparse_mobius_boundary(
    boundary_f32: Tensor,
    track_ray_coeff_f32: Tensor,
    compiler_node_t_f32: Tensor,
    word_offsets_i32: Tensor,
    word_owner_i32: Tensor,
    word_left_incidence_i32: Tensor,
    word_right_incidence_i32: Tensor,
    track_incidence_offsets_i32: Tensor,
    incidence_boundary_i32: Tensor,
    site_rgba_f32: Tensor,
    sample_to_node_f32: Tensor,
    target_rgb_f32: Tensor,
    background_rgb_f32: Tensor,
    config: RealRayReplayConfig,
    *,
    track_count: int,
    node_count: int,
    sample_count: int,
    boundary_count: int,
    loss_scale: float,
    physical_length_epsilon: float = 1.0e-8,
    cone_tolerance: float = 1.0e-6,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Checked convenience call for the fixed-word P0 compiled-Lie VJP."""
    prepared = prepare_fixed_word_p0_compiled_lie_transfer_mse_vjp_sparse_mobius_boundary(
        boundary_f32,
        track_ray_coeff_f32,
        compiler_node_t_f32,
        word_offsets_i32,
        word_owner_i32,
        word_left_incidence_i32,
        word_right_incidence_i32,
        track_incidence_offsets_i32,
        incidence_boundary_i32,
        site_rgba_f32,
        sample_to_node_f32,
        target_rgb_f32,
        background_rgb_f32,
        config,
        track_count=track_count,
        node_count=node_count,
        sample_count=sample_count,
        boundary_count=boundary_count,
        loss_scale=loss_scale,
        physical_length_epsilon=physical_length_epsilon,
        cone_tolerance=cone_tolerance,
    )
    return launch_prepared_fixed_word_p0_compiled_lie_transfer_mse_vjp_sparse_mobius_boundary(prepared)


def sparse_power_boundary_vjp_to_sites_launch_only(
    active_boundary_site_pairs_i32: Tensor,
    sites_f32: Tensor,
    grad_boundary_f32: Tensor,
) -> Tensor:
    """Launch a prevalidated sparse 4D power-boundary-to-site pullback.

    The caller must supply the exact output of
    :func:`recode_sparse_power_boundary_site_pairs`; values are intentionally
    not copied to CPU or revalidated on this per-step path.
    """
    return torch.ops.world_foam_lane2_fused_slab_v0.sparse_power_boundary_vjp_to_sites_launch_only(
        active_boundary_site_pairs_i32,
        sites_f32,
        grad_boundary_f32,
    )


def prepare_sparse_power_boundary_vjp_to_sites(
    active_boundary_site_pairs_i32: Tensor,
    sites_f32: Tensor,
    *,
    boundary_count: int,
) -> PreparedSparsePowerBoundarySiteVjp:
    """Validate and retain the fixed topology used by the final site scatter."""
    active_boundary_site_pairs_i32 = active_boundary_site_pairs_i32.contiguous()
    sites_f32 = sites_f32.contiguous()
    _require_mps_tensor(
        active_boundary_site_pairs_i32,
        name="active_boundary_site_pairs_i32",
        dtype=torch.int32,
        cols=3,
    )
    _require_mps_tensor(sites_f32, name="sites_f32", dtype=torch.float32, cols=5)
    _validate_sparse_power_boundary_site_pairs_cpu(
        active_boundary_site_pairs_i32,
        boundary_count=boundary_count,
        site_count=sites_f32.shape[0],
    )
    return PreparedSparsePowerBoundarySiteVjp(
        active_boundary_site_pairs_i32=active_boundary_site_pairs_i32,
        sites_f32=sites_f32,
        boundary_count=boundary_count,
    )


def launch_prepared_sparse_power_boundary_vjp_to_sites(
    prepared: PreparedSparsePowerBoundarySiteVjp,
    grad_boundary_f32: Tensor,
) -> Tensor:
    """Run the resident final scatter; ``grad_boundary_f32`` is prevalidated."""
    return torch.ops.world_foam_lane2_fused_slab_v0.sparse_power_boundary_vjp_to_sites_launch_only(
        prepared.active_boundary_site_pairs_i32,
        prepared.sites_f32,
        grad_boundary_f32,
    )


def sparse_power_boundary_vjp_to_sites(
    active_boundary_site_pairs_i32: Tensor,
    sites_f32: Tensor,
    grad_boundary_f32: Tensor,
) -> Tensor:
    """Validate fixed topology, then scatter boundary bars into 4D site bars."""
    active_boundary_site_pairs_i32 = active_boundary_site_pairs_i32.contiguous()
    sites_f32 = sites_f32.contiguous()
    grad_boundary_f32 = grad_boundary_f32.contiguous()
    prepared = prepare_sparse_power_boundary_vjp_to_sites(
        active_boundary_site_pairs_i32,
        sites_f32,
        boundary_count=grad_boundary_f32.shape[0],
    )
    _require_mps_tensor(grad_boundary_f32, name="grad_boundary_f32", dtype=torch.float32, cols=5)
    return launch_prepared_sparse_power_boundary_vjp_to_sites(prepared, grad_boundary_f32)


def fixed_word_p0_compiled_lie_transfer_mse_vjp_sparse_mobius_site_geometry(
    track_ray_coeff_f32: Tensor,
    compiler_node_t_f32: Tensor,
    word_offsets_i32: Tensor,
    word_owner_i32: Tensor,
    word_left_incidence_i32: Tensor,
    word_right_incidence_i32: Tensor,
    track_incidence_offsets_i32: Tensor,
    incidence_boundary_i32: Tensor,
    boundary_site_pairs_i32: Tensor,
    sites_f32: Tensor,
    site_rgba_f32: Tensor,
    sample_t_f64: Tensor,
    target_rgb_f32: Tensor,
    background_rgb_f32: Tensor,
    config: RealRayReplayConfig,
    *,
    track_count: int,
    node_count: int,
    sample_count: int,
    chart_index: int,
    certificate_binding: NativeFixedWordP0ContinuousCertificateBinding,
    physical_length_epsilon: float = 1.0e-8,
    cone_tolerance: float = 1.0e-6,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """One-block checked site-geometry convenience over the staged tokens.

    Forward boundaries are derived from ``sites_f32`` and
    ``boundary_site_pairs_i32`` inside the world refresh. The ninth output is
    therefore the derivative of the exact geometry consumed by forward.
    """
    certificate_binding = assert_native_fixed_word_p0_certificate_binding(certificate_binding)
    topology = prepare_fixed_word_p0_topology_token(
        word_offsets_i32,
        word_owner_i32,
        word_left_incidence_i32,
        word_right_incidence_i32,
        track_incidence_offsets_i32,
        incidence_boundary_i32,
        boundary_site_pairs_i32,
        track_count=track_count,
        site_count=sites_f32.shape[0],
        certificate_binding=certificate_binding,
    )
    world = refresh_fixed_word_p0_world_token(
        topology,
        sites_f32,
        site_rgba_f32,
        track_ray_coeff_f32,
        config,
        physical_length_epsilon=physical_length_epsilon,
        cone_tolerance=cone_tolerance,
    )
    chart = prepare_fixed_word_p0_chart_token(
        world,
        compiler_node_t_f32,
        chart_index=chart_index,
    )
    if chart.node_count != node_count:
        raise ValueError("node_count must match compiler_node_t_f32")
    sample_state = prepare_fixed_word_p0_sample_state_token(
        chart,
        global_track_count=track_count,
        global_sample_count=sample_count,
        global_sample_start=0,
        global_sample_end=sample_count,
        global_loss_element_count=track_count * sample_count * 3,
        loss_normalization_id=f"{certificate_binding.canonical_digest}:full",
        sample_partition_generation_id=f"{certificate_binding.canonical_digest}:full-partition",
        sample_block_size=sample_count,
    )
    sample_block = prepare_fixed_word_p0_sample_block_token(
        sample_state,
        target_rgb_f32,
        background_rgb_f32,
        sample_t_f64=sample_t_f64,
        sample_block_id="full",
        global_sample_start=0,
        global_sample_end=sample_count,
    )
    if sample_block.sample_count != sample_count:
        raise ValueError("sample_count must match sample_t_f64")
    prediction_rgb = fixed_word_p0_lie_sample_accumulate_launch_only(
        sample_block,
        sample_state,
    )
    world_grad = fixed_word_p0_lie_world_grad_init_launch_only(
        world,
        expected_chart_partitions=((chart.chart_generation_id, 0, sample_count),),
        global_track_count=track_count,
        global_sample_count=sample_count,
        global_loss_element_count=track_count * sample_count * 3,
        loss_normalization_id=sample_state.loss_normalization_id,
        sample_partition_generation_id=sample_state.sample_partition_generation_id,
    )
    fixed_word_p0_lie_node_vjp_accumulate_launch_only(
        chart,
        sample_state,
        world_grad,
    )
    fixed_word_p0_sparse_mobius_boundary_finalize_launch_only(world_grad)
    grad_sites_f32 = fixed_word_p0_site_geometry_finalize_launch_only(world_grad)
    return (
        sample_state.loss_f32,
        prediction_rgb,
        chart.node_chart_f32,
        sample_state.grad_node_chart_f32,
        world_grad.grad_site_rgba_f32,
        world_grad.grad_mobius_coeff_f32,
        world_grad.grad_boundary_f32,
        sample_state.cone_diagnostic_i32,
        grad_sites_f32,
    )


def endpoint_record_delta_replace_factorized_frameselect_recompute_mse_vjp_direct_atomic_rgb_only(
    boundary_f32: Tensor,
    track_ray_coeff_f32: Tensor,
    frame_t_f32: Tensor,
    base_offsets_i16: Tensor,
    base_record_i32: Tensor,
    frame_change_index_i16: Tensor,
    change_offsets_i16: Tensor,
    change_record_i32: Tensor,
    site_rgba_f32: Tensor,
    target_rgb_f32: Tensor,
    config: RealRayReplayConfig,
    *,
    track_count: int,
    frame_count: int,
    boundary_count: int,
) -> tuple[Tensor, Tensor]:
    """Compute factorized RGB MSE/VJP using one selected sparse-change index per track/frame."""
    boundary_f32 = boundary_f32.contiguous()
    track_ray_coeff_f32 = track_ray_coeff_f32.contiguous()
    frame_t_f32 = frame_t_f32.contiguous()
    base_offsets_i16 = base_offsets_i16.contiguous()
    base_record_i32 = base_record_i32.contiguous()
    frame_change_index_i16 = frame_change_index_i16.contiguous()
    change_offsets_i16 = change_offsets_i16.contiguous()
    change_record_i32 = change_record_i32.contiguous()
    site_rgba_f32 = site_rgba_f32.contiguous()
    target_rgb_f32 = target_rgb_f32.contiguous()
    _require_mps_tensor(boundary_f32, name="boundary_f32", dtype=torch.float32, cols=5)
    _require_mps_tensor(track_ray_coeff_f32, name="track_ray_coeff_f32", dtype=torch.float32, cols=12)
    _require_mps_tensor(frame_t_f32, name="frame_t_f32", dtype=torch.float32, cols=None)
    _require_mps_tensor(base_offsets_i16, name="base_offsets_i16", dtype=torch.int16, cols=None)
    _require_mps_tensor(base_record_i32, name="base_record_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(frame_change_index_i16, name="frame_change_index_i16", dtype=torch.int16, cols=None)
    _require_mps_tensor(change_offsets_i16, name="change_offsets_i16", dtype=torch.int16, cols=None)
    _require_mps_tensor(change_record_i32, name="change_record_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(site_rgba_f32, name="site_rgba_f32", dtype=torch.float32, cols=4)
    if track_count <= 0 or frame_count <= 0 or boundary_count <= 0:
        raise ValueError("track_count, frame_count, and boundary_count must be positive")
    if boundary_count > 4093:
        raise ValueError("factorized frame-select replay supports boundary count <= 4093")
    if boundary_f32.shape[0] != boundary_count:
        raise ValueError("boundary_f32 row count must match boundary_count")
    if track_ray_coeff_f32.shape[0] != track_count:
        raise ValueError("track_ray_coeff_f32 row count must match track_count")
    if frame_t_f32.shape[0] != frame_count:
        raise ValueError("frame_t_f32 must have shape [frame_count]")
    if target_rgb_f32.device.type != "mps" or target_rgb_f32.dtype != torch.float32:
        raise ValueError("target_rgb_f32 must be float32 on MPS")
    if target_rgb_f32.shape != (track_count, frame_count, 3):
        raise ValueError("target_rgb_f32 must have shape [track_count, frame_count, 3]")
    if site_rgba_f32.shape[0] <= 0 or site_rgba_f32.shape[0] > 256:
        raise ValueError("factorized frame-select replay supports site count in [1, 256]")
    if config.far <= config.near:
        raise ValueError("config.far must be greater than config.near")
    if config.invalid_epsilon <= 0.0:
        raise ValueError("config.invalid_epsilon must be positive")
    if config.transmittance_threshold < 0.0:
        raise ValueError("config.transmittance_threshold must be nonnegative")
    base_record_count = base_record_i32.numel()
    change_count = change_offsets_i16.numel() - 1
    change_record_count = change_record_i32.numel()
    if change_count > 32767:
        raise ValueError("frame-select map stores sparse change indices as int16 and requires change count <= 32767")
    if frame_change_index_i16.shape[0] != track_count * max(frame_count - 1, 0):
        raise ValueError("frame_change_index_i16 must have shape [track_count * (frame_count - 1)]")
    frame_select_cpu = frame_change_index_i16.detach().cpu()
    if frame_select_cpu.numel():
        min_index = int(frame_select_cpu.min().item())
        max_index = int(frame_select_cpu.max().item())
        if min_index < -1 or max_index >= change_count:
            raise ValueError(
                f"frame_change_index_i16 values must be in [-1, change_count), got min={min_index} max={max_index}"
            )
    _validate_segment_tape_offsets_cpu(
        base_offsets_i16.detach().cpu(),
        sample_count=track_count,
        segment_count=base_record_count,
        max_segments_per_sample=129,
    )
    _validate_segment_tape_offsets_cpu(
        change_offsets_i16.detach().cpu(),
        sample_count=change_count,
        segment_count=change_record_count,
        max_segments_per_sample=129,
    )
    _validate_packed_endpoint_delta_records_cpu(
        base_record_i32=base_record_i32,
        change_record_i32=change_record_i32,
        site_count=site_rgba_f32.shape[0],
        boundary_count=boundary_count,
    )
    config_i32 = torch.tensor(
        [
            boundary_count,
            track_count,
            frame_count,
            site_rgba_f32.shape[0],
            base_record_count,
            change_count,
            change_record_count,
        ],
        device=boundary_f32.device,
        dtype=torch.int32,
    )
    config_f32 = torch.tensor(
        [config.near, config.far, config.invalid_epsilon, config.transmittance_threshold],
        device=boundary_f32.device,
        dtype=torch.float32,
    )
    ops = torch.ops.world_foam_lane2_fused_slab_v0
    op_name = "endpoint_record_delta_replace_factorized_frameselect_recompute_mse_vjp_direct_atomic_rgb_only"
    if not hasattr(ops, op_name):
        raise RuntimeError(f"world_foam_lane2_fused_slab_v0 {op_name} op not found. Build this variant first.")
    return getattr(ops, op_name)(
        boundary_f32,
        track_ray_coeff_f32,
        frame_t_f32,
        base_offsets_i16,
        base_record_i32,
        frame_change_index_i16,
        change_offsets_i16,
        change_record_i32,
        site_rgba_f32,
        target_rgb_f32,
        config_i32,
        config_f32,
    )


def endpoint_record_delta_replace_factorized_framebitmask_recompute_mse_vjp_direct_atomic_rgb_only(
    boundary_f32: Tensor,
    track_ray_coeff_f32: Tensor,
    frame_t_f32: Tensor,
    base_offsets_i32: Tensor,
    base_record_i32: Tensor,
    track_change_offsets_i32: Tensor,
    track_frame_mask_i32: Tensor,
    change_offsets_i32: Tensor,
    change_record_i32: Tensor,
    site_rgba_f32: Tensor,
    target_rgb_f32: Tensor,
    config: RealRayReplayConfig,
    *,
    track_count: int,
    frame_count: int,
    boundary_count: int,
) -> tuple[Tensor, Tensor]:
    """Compute factorized RGB MSE/VJP using per-track frame bitmasks and rank selection."""
    boundary_f32 = boundary_f32.contiguous()
    track_ray_coeff_f32 = track_ray_coeff_f32.contiguous()
    frame_t_f32 = frame_t_f32.contiguous()
    base_offsets_i32 = base_offsets_i32.contiguous()
    base_record_i32 = base_record_i32.contiguous()
    track_change_offsets_i32 = track_change_offsets_i32.contiguous()
    track_frame_mask_i32 = track_frame_mask_i32.contiguous()
    change_offsets_i32 = change_offsets_i32.contiguous()
    change_record_i32 = change_record_i32.contiguous()
    site_rgba_f32 = site_rgba_f32.contiguous()
    target_rgb_f32 = target_rgb_f32.contiguous()
    _require_mps_tensor(boundary_f32, name="boundary_f32", dtype=torch.float32, cols=5)
    _require_mps_tensor(track_ray_coeff_f32, name="track_ray_coeff_f32", dtype=torch.float32, cols=12)
    _require_mps_tensor(frame_t_f32, name="frame_t_f32", dtype=torch.float32, cols=None)
    _require_mps_tensor(base_offsets_i32, name="base_offsets_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(base_record_i32, name="base_record_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(track_change_offsets_i32, name="track_change_offsets_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(track_frame_mask_i32, name="track_frame_mask_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(change_offsets_i32, name="change_offsets_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(change_record_i32, name="change_record_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(site_rgba_f32, name="site_rgba_f32", dtype=torch.float32, cols=4)
    if track_count <= 0 or frame_count <= 0 or boundary_count <= 0:
        raise ValueError("track_count, frame_count, and boundary_count must be positive")
    if frame_count > 32:
        raise ValueError("factorized frame-bitmask replay requires frame_count <= 32")
    if boundary_count > 4093:
        raise ValueError("factorized frame-bitmask replay supports boundary count <= 4093")
    if boundary_f32.shape[0] != boundary_count:
        raise ValueError("boundary_f32 row count must match boundary_count")
    if track_ray_coeff_f32.shape[0] != track_count:
        raise ValueError("track_ray_coeff_f32 row count must match track_count")
    if frame_t_f32.shape[0] != frame_count:
        raise ValueError("frame_t_f32 must have shape [frame_count]")
    if track_change_offsets_i32.shape[0] != track_count + 1:
        raise ValueError("track_change_offsets_i32 must have shape [track_count + 1]")
    if track_frame_mask_i32.shape[0] != track_count:
        raise ValueError("track_frame_mask_i32 must have shape [track_count]")
    if target_rgb_f32.device.type != "mps" or target_rgb_f32.dtype != torch.float32:
        raise ValueError("target_rgb_f32 must be float32 on MPS")
    if target_rgb_f32.shape != (track_count, frame_count, 3):
        raise ValueError("target_rgb_f32 must have shape [track_count, frame_count, 3]")
    if site_rgba_f32.shape[0] <= 0 or site_rgba_f32.shape[0] > 256:
        raise ValueError("factorized frame-bitmask replay supports site count in [1, 256]")
    if config.far <= config.near:
        raise ValueError("config.far must be greater than config.near")
    if config.invalid_epsilon <= 0.0:
        raise ValueError("config.invalid_epsilon must be positive")
    if config.transmittance_threshold < 0.0:
        raise ValueError("config.transmittance_threshold must be nonnegative")
    base_record_count = base_record_i32.numel()
    if change_offsets_i32.numel() < 1:
        raise ValueError("change_offsets_i32 must contain at least one offset")
    change_count = change_offsets_i32.numel() - 1
    change_record_count = change_record_i32.numel()
    _validate_segment_tape_offsets_cpu(
        base_offsets_i32.detach().cpu(),
        sample_count=track_count,
        segment_count=base_record_count,
        max_segments_per_sample=129,
    )
    _validate_segment_tape_offsets_cpu(
        track_change_offsets_i32.detach().cpu(),
        sample_count=track_count,
        segment_count=change_count,
        max_segments_per_sample=None,
    )
    _validate_segment_tape_offsets_cpu(
        change_offsets_i32.detach().cpu(),
        sample_count=change_count,
        segment_count=change_record_count,
        max_segments_per_sample=129,
    )
    _validate_packed_endpoint_delta_records_cpu(
        base_record_i32=base_record_i32,
        change_record_i32=change_record_i32,
        site_count=site_rgba_f32.shape[0],
        boundary_count=boundary_count,
    )
    masks_cpu = track_frame_mask_i32.detach().cpu()
    if masks_cpu.numel():
        allowed_mask = 0xFFFFFFFE if frame_count == 32 else ((1 << frame_count) - 1) & ~1
        track_offsets = track_change_offsets_i32.detach().cpu().reshape(-1).tolist()
        for track_id, raw_mask in enumerate(masks_cpu.reshape(-1).tolist()):
            mask = int(raw_mask) & 0xFFFFFFFF
            if mask & ~allowed_mask:
                raise ValueError("track_frame_mask_i32 contains bits outside [1, frame_count)")
            expected_changes = int(track_offsets[track_id + 1]) - int(track_offsets[track_id])
            if mask.bit_count() != expected_changes:
                raise ValueError("track_frame_mask_i32 popcount must match per-track change count")
    config_i32 = torch.tensor(
        [
            boundary_count,
            track_count,
            frame_count,
            site_rgba_f32.shape[0],
            base_record_count,
            change_count,
            change_record_count,
        ],
        device=boundary_f32.device,
        dtype=torch.int32,
    )
    config_f32 = torch.tensor(
        [config.near, config.far, config.invalid_epsilon, config.transmittance_threshold],
        device=boundary_f32.device,
        dtype=torch.float32,
    )
    ops = torch.ops.world_foam_lane2_fused_slab_v0
    op_name = "endpoint_record_delta_replace_factorized_framebitmask_recompute_mse_vjp_direct_atomic_rgb_only"
    if not hasattr(ops, op_name):
        raise RuntimeError(f"world_foam_lane2_fused_slab_v0 {op_name} op not found. Build this variant first.")
    return getattr(ops, op_name)(
        boundary_f32,
        track_ray_coeff_f32,
        frame_t_f32,
        base_offsets_i32,
        base_record_i32,
        track_change_offsets_i32,
        track_frame_mask_i32,
        change_offsets_i32,
        change_record_i32,
        site_rgba_f32,
        target_rgb_f32,
        config_i32,
        config_f32,
    )


def endpoint_record_delta_replace_coeff16_packed_framegroup16_smallrun16_mse_vjp_direct_atomic_rgb_only(
    coeff_f16: Tensor,
    frame_t_f32: Tensor,
    base_offsets_i32: Tensor,
    base_record_i32: Tensor,
    track_change_offsets_i32: Tensor,
    track_chunk_change_offsets_i16: Tensor,
    change_frame_i32: Tensor,
    change_offsets_i32: Tensor,
    change_record_i32: Tensor,
    site_rgba_f32: Tensor,
    target_rgb_f32: Tensor,
    config: RealRayReplayConfig,
    *,
    track_count: int,
    frame_count: int,
    boundary_count: int,
) -> tuple[Tensor, Tensor]:
    """Compute RGB MSE/VJP for packed 32-frame chunks capped to 16 replay segments."""
    coeff_f16 = coeff_f16.contiguous()
    frame_t_f32 = frame_t_f32.contiguous()
    base_offsets_i32 = base_offsets_i32.contiguous()
    base_record_i32 = base_record_i32.contiguous()
    track_change_offsets_i32 = track_change_offsets_i32.contiguous()
    track_chunk_change_offsets_i16 = track_chunk_change_offsets_i16.contiguous()
    change_frame_i32 = change_frame_i32.contiguous()
    change_offsets_i32 = change_offsets_i32.contiguous()
    change_record_i32 = change_record_i32.contiguous()
    site_rgba_f32 = site_rgba_f32.contiguous()
    target_rgb_f32 = target_rgb_f32.contiguous()
    _require_mps_tensor(coeff_f16, name="coeff_f16", dtype=torch.float16, cols=4)
    _require_mps_tensor(frame_t_f32, name="frame_t_f32", dtype=torch.float32, cols=None)
    _require_mps_tensor(base_offsets_i32, name="base_offsets_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(base_record_i32, name="base_record_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(track_change_offsets_i32, name="track_change_offsets_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(
        track_chunk_change_offsets_i16,
        name="track_chunk_change_offsets_i16",
        dtype=torch.int16,
        cols=None,
    )
    _require_mps_tensor(change_frame_i32, name="change_frame_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(change_offsets_i32, name="change_offsets_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(change_record_i32, name="change_record_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(site_rgba_f32, name="site_rgba_f32", dtype=torch.float32, cols=4)
    if track_count <= 0 or frame_count <= 0 or boundary_count <= 0:
        raise ValueError("track_count, frame_count, and boundary_count must be positive")
    if boundary_count > 4093:
        raise ValueError("packed framegroup16 smallrun16 delta replace coeff16 replay supports boundary count <= 4093")
    if coeff_f16.shape[0] != track_count * boundary_count:
        raise ValueError("coeff_f16 row count must be track_count * boundary_count")
    if frame_t_f32.shape[0] != frame_count:
        raise ValueError("frame_t_f32 must have shape [frame_count]")
    if target_rgb_f32.device.type != "mps" or target_rgb_f32.dtype != torch.float32:
        raise ValueError("target_rgb_f32 must be float32 on MPS")
    if target_rgb_f32.shape != (track_count, frame_count, 3):
        raise ValueError("target_rgb_f32 must have shape [track_count, frame_count, 3]")
    if site_rgba_f32.shape[0] <= 0 or site_rgba_f32.shape[0] > 256:
        raise ValueError("packed framegroup16 smallrun16 delta replace coeff16 replay supports site count in [1, 256]")
    if config.far <= config.near:
        raise ValueError("config.far must be greater than config.near")
    if config.invalid_epsilon <= 0.0:
        raise ValueError("config.invalid_epsilon must be positive")
    if config.transmittance_threshold < 0.0:
        raise ValueError("config.transmittance_threshold must be nonnegative")
    base_record_count = base_record_i32.numel()
    change_record_count = change_record_i32.numel()
    chunk_count = (frame_count + 31) // 32
    if change_frame_i32.shape[0] > 32767:
        raise ValueError("framegroup16 chunk-start offsets use int16 and require change count <= 32767")
    _validate_segment_tape_offsets_cpu(
        base_offsets_i32.detach().cpu(),
        sample_count=track_count,
        segment_count=base_record_count,
        max_segments_per_sample=16,
    )
    _validate_segment_tape_offsets_cpu(
        track_change_offsets_i32.detach().cpu(),
        sample_count=track_count,
        segment_count=change_frame_i32.shape[0],
        max_segments_per_sample=None,
    )
    _validate_segment_tape_offsets_cpu(
        track_chunk_change_offsets_i16.detach().cpu(),
        sample_count=track_count * (chunk_count + 1) - 1,
        segment_count=change_frame_i32.shape[0],
        max_segments_per_sample=None,
    )
    _validate_segment_tape_offsets_cpu(
        change_offsets_i32.detach().cpu(),
        sample_count=change_frame_i32.shape[0],
        segment_count=change_record_count,
        max_segments_per_sample=16,
    )
    _validate_packed_endpoint_delta_records_cpu(
        base_record_i32=base_record_i32,
        change_record_i32=change_record_i32,
        site_count=site_rgba_f32.shape[0],
        boundary_count=boundary_count,
    )
    config_i32 = torch.tensor(
        [
            boundary_count,
            track_count,
            frame_count,
            site_rgba_f32.shape[0],
            base_record_count,
            change_frame_i32.shape[0],
            change_record_count,
        ],
        device=coeff_f16.device,
        dtype=torch.int32,
    )
    config_f32 = torch.tensor(
        [config.near, config.far, config.invalid_epsilon, config.transmittance_threshold],
        device=coeff_f16.device,
        dtype=torch.float32,
    )
    ops = torch.ops.world_foam_lane2_fused_slab_v0
    op_name = "endpoint_record_delta_replace_coeff16_packed_framegroup16_smallrun16_mse_vjp_direct_atomic_rgb_only"
    if not hasattr(ops, op_name):
        raise RuntimeError(f"world_foam_lane2_fused_slab_v0 {op_name} op not found. Build this variant first.")
    return getattr(ops, op_name)(
        coeff_f16,
        frame_t_f32,
        base_offsets_i32,
        base_record_i32,
        track_change_offsets_i32,
        track_chunk_change_offsets_i16,
        change_frame_i32,
        change_offsets_i32,
        change_record_i32,
        site_rgba_f32,
        target_rgb_f32,
        config_i32,
        config_f32,
    )


def endpoint_record_delta_replace_coeff16_packed_framegroup16_materialized_mse_vjp_direct_atomic_rgb_only(
    coeff_f16: Tensor,
    frame_t_f32: Tensor,
    base_offsets_i32: Tensor,
    base_record_i32: Tensor,
    track_change_offsets_i32: Tensor,
    track_chunk_change_offsets_i16: Tensor,
    change_frame_i32: Tensor,
    change_offsets_i32: Tensor,
    change_record_i32: Tensor,
    site_rgba_f32: Tensor,
    target_rgb_f32: Tensor,
    config: RealRayReplayConfig,
    *,
    track_count: int,
    frame_count: int,
    boundary_count: int,
) -> tuple[Tensor, Tensor]:
    """Compute RGB MSE/VJP for materialized packed int32 delta-replace rows with 16-frame chunks."""
    coeff_f16 = coeff_f16.contiguous()
    frame_t_f32 = frame_t_f32.contiguous()
    base_offsets_i32 = base_offsets_i32.contiguous()
    base_record_i32 = base_record_i32.contiguous()
    track_change_offsets_i32 = track_change_offsets_i32.contiguous()
    track_chunk_change_offsets_i16 = track_chunk_change_offsets_i16.contiguous()
    change_frame_i32 = change_frame_i32.contiguous()
    change_offsets_i32 = change_offsets_i32.contiguous()
    change_record_i32 = change_record_i32.contiguous()
    site_rgba_f32 = site_rgba_f32.contiguous()
    target_rgb_f32 = target_rgb_f32.contiguous()
    _require_mps_tensor(coeff_f16, name="coeff_f16", dtype=torch.float16, cols=4)
    _require_mps_tensor(frame_t_f32, name="frame_t_f32", dtype=torch.float32, cols=None)
    _require_mps_tensor(base_offsets_i32, name="base_offsets_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(base_record_i32, name="base_record_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(track_change_offsets_i32, name="track_change_offsets_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(
        track_chunk_change_offsets_i16,
        name="track_chunk_change_offsets_i16",
        dtype=torch.int16,
        cols=None,
    )
    _require_mps_tensor(change_frame_i32, name="change_frame_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(change_offsets_i32, name="change_offsets_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(change_record_i32, name="change_record_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(site_rgba_f32, name="site_rgba_f32", dtype=torch.float32, cols=4)
    if track_count <= 0 or frame_count <= 0 or boundary_count <= 0:
        raise ValueError("track_count, frame_count, and boundary_count must be positive")
    if boundary_count > 4093:
        raise ValueError(
            "packed materialized framegroup16 delta replace coeff16 replay supports boundary count <= 4093"
        )
    if coeff_f16.shape[0] != track_count * boundary_count:
        raise ValueError("coeff_f16 row count must be track_count * boundary_count")
    if frame_t_f32.shape[0] != frame_count:
        raise ValueError("frame_t_f32 must have shape [frame_count]")
    if target_rgb_f32.device.type != "mps" or target_rgb_f32.dtype != torch.float32:
        raise ValueError("target_rgb_f32 must be float32 on MPS")
    if target_rgb_f32.shape != (track_count, frame_count, 3):
        raise ValueError("target_rgb_f32 must have shape [track_count, frame_count, 3]")
    if site_rgba_f32.shape[0] <= 0 or site_rgba_f32.shape[0] > 256:
        raise ValueError(
            "packed materialized framegroup16 delta replace coeff16 replay supports site count in [1, 256]"
        )
    if config.far <= config.near:
        raise ValueError("config.far must be greater than config.near")
    if config.invalid_epsilon <= 0.0:
        raise ValueError("config.invalid_epsilon must be positive")
    if config.transmittance_threshold < 0.0:
        raise ValueError("config.transmittance_threshold must be nonnegative")
    base_record_count = base_record_i32.numel()
    change_record_count = change_record_i32.numel()
    chunk_count = (frame_count + 15) // 16
    if change_frame_i32.shape[0] > 32767:
        raise ValueError("framegroup16 chunk-start offsets use int16 and require change count <= 32767")
    _validate_segment_tape_offsets_cpu(
        base_offsets_i32.detach().cpu(),
        sample_count=track_count,
        segment_count=base_record_count,
        max_segments_per_sample=129,
    )
    _validate_segment_tape_offsets_cpu(
        track_change_offsets_i32.detach().cpu(),
        sample_count=track_count,
        segment_count=change_frame_i32.shape[0],
        max_segments_per_sample=None,
    )
    _validate_segment_tape_offsets_cpu(
        track_chunk_change_offsets_i16.detach().cpu(),
        sample_count=track_count * (chunk_count + 1) - 1,
        segment_count=change_frame_i32.shape[0],
        max_segments_per_sample=None,
    )
    _validate_segment_tape_offsets_cpu(
        change_offsets_i32.detach().cpu(),
        sample_count=change_frame_i32.shape[0],
        segment_count=change_record_count,
        max_segments_per_sample=129,
    )
    _validate_packed_endpoint_delta_records_cpu(
        base_record_i32=base_record_i32,
        change_record_i32=change_record_i32,
        site_count=site_rgba_f32.shape[0],
        boundary_count=boundary_count,
    )
    config_i32 = torch.tensor(
        [
            boundary_count,
            track_count,
            frame_count,
            site_rgba_f32.shape[0],
            base_record_count,
            change_frame_i32.shape[0],
            change_record_count,
        ],
        device=coeff_f16.device,
        dtype=torch.int32,
    )
    config_f32 = torch.tensor(
        [config.near, config.far, config.invalid_epsilon, config.transmittance_threshold],
        device=coeff_f16.device,
        dtype=torch.float32,
    )
    ops = torch.ops.world_foam_lane2_fused_slab_v0
    op_name = "endpoint_record_delta_replace_coeff16_packed_framegroup16_materialized_mse_vjp_direct_atomic_rgb_only"
    if not hasattr(ops, op_name):
        raise RuntimeError(f"world_foam_lane2_fused_slab_v0 {op_name} op not found. Build this variant first.")
    return getattr(ops, op_name)(
        coeff_f16,
        frame_t_f32,
        base_offsets_i32,
        base_record_i32,
        track_change_offsets_i32,
        track_chunk_change_offsets_i16,
        change_frame_i32,
        change_offsets_i32,
        change_record_i32,
        site_rgba_f32,
        target_rgb_f32,
        config_i32,
        config_f32,
    )


def endpoint_record_delta_replace_coeff16_i16x3_framegroup16_materialized_mse_vjp_direct_atomic_rgb_only(
    coeff_f16: Tensor,
    frame_t_f32: Tensor,
    base_offsets_i32: Tensor,
    base_record_i16: Tensor,
    track_change_offsets_i32: Tensor,
    track_chunk_change_offsets_i16: Tensor,
    change_frame_i32: Tensor,
    change_offsets_i32: Tensor,
    change_record_i16: Tensor,
    site_rgba_f32: Tensor,
    target_rgb_f32: Tensor,
    config: RealRayReplayConfig,
    *,
    track_count: int,
    frame_count: int,
    boundary_count: int,
) -> tuple[Tensor, Tensor]:
    """Compute RGB MSE/VJP for materialized i16x3 delta-replace rows with 16-frame chunks."""
    coeff_f16 = coeff_f16.contiguous()
    frame_t_f32 = frame_t_f32.contiguous()
    base_offsets_i32 = base_offsets_i32.contiguous()
    base_record_i16 = base_record_i16.contiguous()
    track_change_offsets_i32 = track_change_offsets_i32.contiguous()
    track_chunk_change_offsets_i16 = track_chunk_change_offsets_i16.contiguous()
    change_frame_i32 = change_frame_i32.contiguous()
    change_offsets_i32 = change_offsets_i32.contiguous()
    change_record_i16 = change_record_i16.contiguous()
    site_rgba_f32 = site_rgba_f32.contiguous()
    target_rgb_f32 = target_rgb_f32.contiguous()
    _require_mps_tensor(coeff_f16, name="coeff_f16", dtype=torch.float16, cols=4)
    _require_mps_tensor(frame_t_f32, name="frame_t_f32", dtype=torch.float32, cols=None)
    _require_mps_tensor(base_offsets_i32, name="base_offsets_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(base_record_i16, name="base_record_i16", dtype=torch.int16, cols=None)
    _require_mps_tensor(track_change_offsets_i32, name="track_change_offsets_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(
        track_chunk_change_offsets_i16,
        name="track_chunk_change_offsets_i16",
        dtype=torch.int16,
        cols=None,
    )
    _require_mps_tensor(change_frame_i32, name="change_frame_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(change_offsets_i32, name="change_offsets_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(change_record_i16, name="change_record_i16", dtype=torch.int16, cols=None)
    _require_mps_tensor(site_rgba_f32, name="site_rgba_f32", dtype=torch.float32, cols=4)
    if track_count <= 0 or frame_count <= 0 or boundary_count <= 0:
        raise ValueError("track_count, frame_count, and boundary_count must be positive")
    if boundary_count > 32765:
        raise ValueError(
            "endpoint record delta replace coeff16 i16x3 materialized framegroup16 replay supports boundary count <= 32765"
        )
    if coeff_f16.shape[0] != track_count * boundary_count:
        raise ValueError("coeff_f16 row count must be track_count * boundary_count")
    if frame_t_f32.shape[0] != frame_count:
        raise ValueError("frame_t_f32 must have shape [frame_count]")
    if base_record_i16.numel() % 3 != 0 or change_record_i16.numel() % 3 != 0:
        raise ValueError("i16x3 record tensors must have length divisible by 3")
    if target_rgb_f32.device.type != "mps" or target_rgb_f32.dtype != torch.float32:
        raise ValueError("target_rgb_f32 must be float32 on MPS")
    if target_rgb_f32.shape != (track_count, frame_count, 3):
        raise ValueError("target_rgb_f32 must have shape [track_count, frame_count, 3]")
    if site_rgba_f32.shape[0] <= 0 or site_rgba_f32.shape[0] > 32767:
        raise ValueError(
            "endpoint record delta replace coeff16 i16x3 materialized framegroup16 replay supports site count in [1, 32767]"
        )
    if config.far <= config.near:
        raise ValueError("config.far must be greater than config.near")
    if config.invalid_epsilon <= 0.0:
        raise ValueError("config.invalid_epsilon must be positive")
    if config.transmittance_threshold < 0.0:
        raise ValueError("config.transmittance_threshold must be nonnegative")
    base_record_count = base_record_i16.numel() // 3
    change_record_count = change_record_i16.numel() // 3
    chunk_count = (frame_count + 15) // 16
    if change_frame_i32.shape[0] > 32767:
        raise ValueError("framegroup16 chunk-start offsets use int16 and require change count <= 32767")
    _validate_segment_tape_offsets_cpu(
        base_offsets_i32.detach().cpu(),
        sample_count=track_count,
        segment_count=base_record_count,
        max_segments_per_sample=129,
    )
    _validate_segment_tape_offsets_cpu(
        track_change_offsets_i32.detach().cpu(),
        sample_count=track_count,
        segment_count=change_frame_i32.shape[0],
        max_segments_per_sample=None,
    )
    _validate_segment_tape_offsets_cpu(
        track_chunk_change_offsets_i16.detach().cpu(),
        sample_count=track_count * (chunk_count + 1) - 1,
        segment_count=change_frame_i32.shape[0],
        max_segments_per_sample=None,
    )
    _validate_segment_tape_offsets_cpu(
        change_offsets_i32.detach().cpu(),
        sample_count=change_frame_i32.shape[0],
        segment_count=change_record_count,
        max_segments_per_sample=129,
    )
    config_i32 = torch.tensor(
        [
            boundary_count,
            track_count,
            frame_count,
            site_rgba_f32.shape[0],
            base_record_count,
            change_frame_i32.shape[0],
            change_record_count,
        ],
        device=coeff_f16.device,
        dtype=torch.int32,
    )
    config_f32 = torch.tensor(
        [config.near, config.far, config.invalid_epsilon, config.transmittance_threshold],
        device=coeff_f16.device,
        dtype=torch.float32,
    )
    ops = torch.ops.world_foam_lane2_fused_slab_v0
    op_name = "endpoint_record_delta_replace_coeff16_i16x3_framegroup16_materialized_mse_vjp_direct_atomic_rgb_only"
    if not hasattr(ops, op_name):
        raise RuntimeError(f"world_foam_lane2_fused_slab_v0 {op_name} op not found. Build this variant first.")
    return getattr(ops, op_name)(
        coeff_f16,
        frame_t_f32,
        base_offsets_i32,
        base_record_i16,
        track_change_offsets_i32,
        track_chunk_change_offsets_i16,
        change_frame_i32,
        change_offsets_i32,
        change_record_i16,
        site_rgba_f32,
        target_rgb_f32,
        config_i32,
        config_f32,
    )


def endpoint_record_delta_replace_coeff16_i16x4_framegroup16_mse_vjp_direct_atomic_rgb_only(
    coeff_f16: Tensor,
    frame_t_f32: Tensor,
    base_offsets_i32: Tensor,
    base_record_i16: Tensor,
    track_change_offsets_i32: Tensor,
    track_chunk_change_offsets_i16: Tensor,
    change_frame_i32: Tensor,
    change_offsets_i32: Tensor,
    change_record_i16: Tensor,
    site_rgba_f32: Tensor,
    target_rgb_f32: Tensor,
    config: RealRayReplayConfig,
    *,
    track_count: int,
    frame_count: int,
    boundary_count: int,
) -> tuple[Tensor, Tensor]:
    """Compute RGB MSE/VJP for delta-replace records with 32-frame chunks and padded i16x4 rows."""
    coeff_f16 = coeff_f16.contiguous()
    frame_t_f32 = frame_t_f32.contiguous()
    base_offsets_i32 = base_offsets_i32.contiguous()
    base_record_i16 = base_record_i16.contiguous()
    track_change_offsets_i32 = track_change_offsets_i32.contiguous()
    track_chunk_change_offsets_i16 = track_chunk_change_offsets_i16.contiguous()
    change_frame_i32 = change_frame_i32.contiguous()
    change_offsets_i32 = change_offsets_i32.contiguous()
    change_record_i16 = change_record_i16.contiguous()
    site_rgba_f32 = site_rgba_f32.contiguous()
    target_rgb_f32 = target_rgb_f32.contiguous()
    _require_mps_tensor(coeff_f16, name="coeff_f16", dtype=torch.float16, cols=4)
    _require_mps_tensor(frame_t_f32, name="frame_t_f32", dtype=torch.float32, cols=None)
    _require_mps_tensor(base_offsets_i32, name="base_offsets_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(base_record_i16, name="base_record_i16", dtype=torch.int16, cols=None)
    _require_mps_tensor(track_change_offsets_i32, name="track_change_offsets_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(
        track_chunk_change_offsets_i16,
        name="track_chunk_change_offsets_i16",
        dtype=torch.int16,
        cols=None,
    )
    _require_mps_tensor(change_frame_i32, name="change_frame_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(change_offsets_i32, name="change_offsets_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(change_record_i16, name="change_record_i16", dtype=torch.int16, cols=None)
    _require_mps_tensor(site_rgba_f32, name="site_rgba_f32", dtype=torch.float32, cols=4)
    if track_count <= 0 or frame_count <= 0 or boundary_count <= 0:
        raise ValueError("track_count, frame_count, and boundary_count must be positive")
    if boundary_count > 32765:
        raise ValueError(
            "endpoint record delta replace coeff16 i16x4 framegroup16 replay supports boundary count <= 32765"
        )
    if coeff_f16.shape[0] != track_count * boundary_count:
        raise ValueError("coeff_f16 row count must be track_count * boundary_count")
    if frame_t_f32.shape[0] != frame_count:
        raise ValueError("frame_t_f32 must have shape [frame_count]")
    if base_record_i16.numel() % 4 != 0 or change_record_i16.numel() % 4 != 0:
        raise ValueError("i16x4 record tensors must have length divisible by 4")
    if target_rgb_f32.device.type != "mps" or target_rgb_f32.dtype != torch.float32:
        raise ValueError("target_rgb_f32 must be float32 on MPS")
    if target_rgb_f32.shape != (track_count, frame_count, 3):
        raise ValueError("target_rgb_f32 must have shape [track_count, frame_count, 3]")
    if site_rgba_f32.shape[0] <= 0 or site_rgba_f32.shape[0] > 32767:
        raise ValueError(
            "endpoint record delta replace coeff16 i16x4 framegroup16 replay supports site count in [1, 32767]"
        )
    if config.far <= config.near:
        raise ValueError("config.far must be greater than config.near")
    if config.invalid_epsilon <= 0.0:
        raise ValueError("config.invalid_epsilon must be positive")
    if config.transmittance_threshold < 0.0:
        raise ValueError("config.transmittance_threshold must be nonnegative")
    base_record_count = base_record_i16.numel() // 4
    change_record_count = change_record_i16.numel() // 4
    chunk_count = (frame_count + 31) // 32
    if change_frame_i32.shape[0] > 32767:
        raise ValueError("framegroup16 chunk-start offsets use int16 and require change count <= 32767")
    _validate_segment_tape_offsets_cpu(
        base_offsets_i32.detach().cpu(),
        sample_count=track_count,
        segment_count=base_record_count,
        max_segments_per_sample=129,
    )
    _validate_segment_tape_offsets_cpu(
        track_change_offsets_i32.detach().cpu(),
        sample_count=track_count,
        segment_count=change_frame_i32.shape[0],
        max_segments_per_sample=None,
    )
    _validate_segment_tape_offsets_cpu(
        track_chunk_change_offsets_i16.detach().cpu(),
        sample_count=track_count * (chunk_count + 1) - 1,
        segment_count=change_frame_i32.shape[0],
        max_segments_per_sample=None,
    )
    _validate_segment_tape_offsets_cpu(
        change_offsets_i32.detach().cpu(),
        sample_count=change_frame_i32.shape[0],
        segment_count=change_record_count,
        max_segments_per_sample=129,
    )
    config_i32 = torch.tensor(
        [
            boundary_count,
            track_count,
            frame_count,
            site_rgba_f32.shape[0],
            base_record_count,
            change_frame_i32.shape[0],
            change_record_count,
        ],
        device=coeff_f16.device,
        dtype=torch.int32,
    )
    config_f32 = torch.tensor(
        [config.near, config.far, config.invalid_epsilon, config.transmittance_threshold],
        device=coeff_f16.device,
        dtype=torch.float32,
    )
    ops = torch.ops.world_foam_lane2_fused_slab_v0
    op_name = "endpoint_record_delta_replace_coeff16_i16x4_framegroup16_mse_vjp_direct_atomic_rgb_only"
    if not hasattr(ops, op_name):
        raise RuntimeError(f"world_foam_lane2_fused_slab_v0 {op_name} op not found. Build this variant first.")
    return getattr(ops, op_name)(
        coeff_f16,
        frame_t_f32,
        base_offsets_i32,
        base_record_i16,
        track_change_offsets_i32,
        track_chunk_change_offsets_i16,
        change_frame_i32,
        change_offsets_i32,
        change_record_i16,
        site_rgba_f32,
        target_rgb_f32,
        config_i32,
        config_f32,
    )


class _EndpointRecordDeltaReplaceCoeff16I16x3Framegroup16MSEFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        coeff_f16: Tensor,
        frame_t_f32: Tensor,
        base_offsets_i32: Tensor,
        base_record_i16: Tensor,
        track_change_offsets_i32: Tensor,
        track_chunk_change_offsets_i16: Tensor,
        change_frame_i32: Tensor,
        change_offsets_i32: Tensor,
        change_record_i16: Tensor,
        site_rgba_f32: Tensor,
        target_rgb_f32: Tensor,
        config: RealRayReplayConfig,
        track_count: int,
        frame_count: int,
        boundary_count: int,
    ) -> Tensor:
        loss, grad_site_rgba = endpoint_record_delta_replace_coeff16_i16x3_framegroup16_mse_vjp_direct_atomic_rgb_only(
            coeff_f16,
            frame_t_f32,
            base_offsets_i32,
            base_record_i16,
            track_change_offsets_i32,
            track_chunk_change_offsets_i16,
            change_frame_i32,
            change_offsets_i32,
            change_record_i16,
            site_rgba_f32,
            target_rgb_f32,
            config,
            track_count=track_count,
            frame_count=frame_count,
            boundary_count=boundary_count,
        )
        ctx.save_for_backward(grad_site_rgba)
        return loss.reshape(())

    @staticmethod
    def backward(ctx: torch.autograd.function.FunctionCtx, grad_loss: Tensor) -> tuple[Any, ...]:
        (grad_site_rgba,) = ctx.saved_tensors
        grad_scale = grad_loss.to(device=grad_site_rgba.device, dtype=grad_site_rgba.dtype).reshape(())
        return (
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            grad_site_rgba * grad_scale,
            None,
            None,
            None,
            None,
            None,
        )


def endpoint_record_delta_replace_coeff16_i16x3_framegroup16_mse_autograd(
    coeff_f16: Tensor,
    frame_t_f32: Tensor,
    base_offsets_i32: Tensor,
    base_record_i16: Tensor,
    track_change_offsets_i32: Tensor,
    track_chunk_change_offsets_i16: Tensor,
    change_frame_i32: Tensor,
    change_offsets_i32: Tensor,
    change_record_i16: Tensor,
    site_rgba_f32: Tensor,
    target_rgb_f32: Tensor,
    config: RealRayReplayConfig,
    *,
    track_count: int,
    frame_count: int,
    boundary_count: int,
) -> Tensor:
    """Return fused RGB MSE with autograd for frozen-geometry site-RGBA training."""
    return _EndpointRecordDeltaReplaceCoeff16I16x3Framegroup16MSEFunction.apply(
        coeff_f16,
        frame_t_f32,
        base_offsets_i32,
        base_record_i16,
        track_change_offsets_i32,
        track_chunk_change_offsets_i16,
        change_frame_i32,
        change_offsets_i32,
        change_record_i16,
        site_rgba_f32,
        target_rgb_f32,
        config,
        int(track_count),
        int(frame_count),
        int(boundary_count),
    )


def endpoint_record_delta_replace_coeff16_i16x4_mse_vjp_direct_atomic_rgb_only(
    coeff_f16: Tensor,
    frame_t_f32: Tensor,
    base_offsets_i32: Tensor,
    base_record_i16: Tensor,
    track_change_offsets_i32: Tensor,
    change_frame_i32: Tensor,
    change_offsets_i32: Tensor,
    change_record_i16: Tensor,
    site_rgba_f32: Tensor,
    target_rgb_f32: Tensor,
    config: RealRayReplayConfig,
    *,
    track_count: int,
    frame_count: int,
    boundary_count: int,
) -> tuple[Tensor, Tensor]:
    """Compute RGB MSE/VJP for delta-replace endpoint records with coeff16 and padded i16x4 records."""
    coeff_f16 = coeff_f16.contiguous()
    frame_t_f32 = frame_t_f32.contiguous()
    base_offsets_i32 = base_offsets_i32.contiguous()
    base_record_i16 = base_record_i16.contiguous()
    track_change_offsets_i32 = track_change_offsets_i32.contiguous()
    change_frame_i32 = change_frame_i32.contiguous()
    change_offsets_i32 = change_offsets_i32.contiguous()
    change_record_i16 = change_record_i16.contiguous()
    site_rgba_f32 = site_rgba_f32.contiguous()
    target_rgb_f32 = target_rgb_f32.contiguous()
    _require_mps_tensor(coeff_f16, name="coeff_f16", dtype=torch.float16, cols=4)
    _require_mps_tensor(frame_t_f32, name="frame_t_f32", dtype=torch.float32, cols=None)
    _require_mps_tensor(base_offsets_i32, name="base_offsets_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(base_record_i16, name="base_record_i16", dtype=torch.int16, cols=None)
    _require_mps_tensor(track_change_offsets_i32, name="track_change_offsets_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(change_frame_i32, name="change_frame_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(change_offsets_i32, name="change_offsets_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(change_record_i16, name="change_record_i16", dtype=torch.int16, cols=None)
    _require_mps_tensor(site_rgba_f32, name="site_rgba_f32", dtype=torch.float32, cols=4)
    if track_count <= 0 or frame_count <= 0 or boundary_count <= 0:
        raise ValueError("track_count, frame_count, and boundary_count must be positive")
    if boundary_count > 32765:
        raise ValueError("endpoint record delta replace coeff16 i16x4 replay supports boundary count <= 32765")
    if coeff_f16.shape[0] != track_count * boundary_count:
        raise ValueError("coeff_f16 row count must be track_count * boundary_count")
    if frame_t_f32.shape[0] != frame_count:
        raise ValueError("frame_t_f32 must have shape [frame_count]")
    if base_record_i16.numel() % 4 != 0 or change_record_i16.numel() % 4 != 0:
        raise ValueError("i16x4 record tensors must have length divisible by 4")
    if target_rgb_f32.device.type != "mps" or target_rgb_f32.dtype != torch.float32:
        raise ValueError("target_rgb_f32 must be float32 on MPS")
    if target_rgb_f32.shape != (track_count, frame_count, 3):
        raise ValueError("target_rgb_f32 must have shape [track_count, frame_count, 3]")
    if site_rgba_f32.shape[0] <= 0 or site_rgba_f32.shape[0] > 32767:
        raise ValueError("endpoint record delta replace coeff16 i16x4 replay supports site count in [1, 32767]")
    if config.far <= config.near:
        raise ValueError("config.far must be greater than config.near")
    if config.invalid_epsilon <= 0.0:
        raise ValueError("config.invalid_epsilon must be positive")
    if config.transmittance_threshold < 0.0:
        raise ValueError("config.transmittance_threshold must be nonnegative")
    base_record_count = base_record_i16.numel() // 4
    change_record_count = change_record_i16.numel() // 4
    _validate_segment_tape_offsets_cpu(
        base_offsets_i32.detach().cpu(),
        sample_count=track_count,
        segment_count=base_record_count,
        max_segments_per_sample=129,
    )
    _validate_segment_tape_offsets_cpu(
        track_change_offsets_i32.detach().cpu(),
        sample_count=track_count,
        segment_count=change_frame_i32.shape[0],
        max_segments_per_sample=None,
    )
    _validate_segment_tape_offsets_cpu(
        change_offsets_i32.detach().cpu(),
        sample_count=change_frame_i32.shape[0],
        segment_count=change_record_count,
        max_segments_per_sample=129,
    )
    config_i32 = torch.tensor(
        [
            boundary_count,
            track_count,
            frame_count,
            site_rgba_f32.shape[0],
            base_record_count,
            change_frame_i32.shape[0],
            change_record_count,
        ],
        device=coeff_f16.device,
        dtype=torch.int32,
    )
    config_f32 = torch.tensor(
        [config.near, config.far, config.invalid_epsilon, config.transmittance_threshold],
        device=coeff_f16.device,
        dtype=torch.float32,
    )
    ops = torch.ops.world_foam_lane2_fused_slab_v0
    op_name = "endpoint_record_delta_replace_coeff16_i16x4_mse_vjp_direct_atomic_rgb_only"
    if not hasattr(ops, op_name):
        raise RuntimeError(f"world_foam_lane2_fused_slab_v0 {op_name} op not found. Build this variant first.")
    return getattr(ops, op_name)(
        coeff_f16,
        frame_t_f32,
        base_offsets_i32,
        base_record_i16,
        track_change_offsets_i32,
        change_frame_i32,
        change_offsets_i32,
        change_record_i16,
        site_rgba_f32,
        target_rgb_f32,
        config_i32,
        config_f32,
    )


def _endpoint_record_edit_config_tensors(
    *,
    boundary_f32: Tensor,
    rays_f32: Tensor,
    frame_t_f32: Tensor,
    base_offsets_i32: Tensor,
    base_owner_i32: Tensor,
    base_left_i32: Tensor,
    base_right_i32: Tensor,
    track_change_offsets_i32: Tensor,
    change_frame_i32: Tensor,
    op_offsets_i32: Tensor,
    op_type_i32: Tensor,
    op_pos_i32: Tensor,
    op_owner_i32: Tensor,
    op_left_i32: Tensor,
    op_right_i32: Tensor,
    site_rgba_f32: Tensor,
    config: RealRayReplayConfig,
    track_count: int,
    frame_count: int,
    max_runs_per_sample: int | None,
) -> tuple[Tensor, Tensor]:
    _require_mps_tensor(boundary_f32, name="boundary_f32", dtype=torch.float32, cols=5)
    _require_mps_tensor(frame_t_f32, name="frame_t_f32", dtype=torch.float32, cols=None)
    _require_mps_tensor(base_offsets_i32, name="base_offsets_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(base_owner_i32, name="base_owner_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(base_left_i32, name="base_left_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(base_right_i32, name="base_right_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(track_change_offsets_i32, name="track_change_offsets_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(change_frame_i32, name="change_frame_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(op_offsets_i32, name="op_offsets_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(op_type_i32, name="op_type_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(op_pos_i32, name="op_pos_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(op_owner_i32, name="op_owner_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(op_left_i32, name="op_left_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(op_right_i32, name="op_right_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(site_rgba_f32, name="site_rgba_f32", dtype=torch.float32, cols=4)
    if rays_f32.device.type != "mps" or rays_f32.dtype != torch.float32:
        raise ValueError("rays_f32 must be float32 on MPS")
    if rays_f32.shape != (track_count, frame_count, 6):
        raise ValueError("rays_f32 must have shape [track_count, frame_count, 6]")
    if not rays_f32.is_contiguous():
        raise ValueError("rays_f32 must be contiguous")
    if frame_t_f32.shape[0] != frame_count:
        raise ValueError("frame_t_f32 must have shape [frame_count]")
    if base_left_i32.shape != base_owner_i32.shape or base_right_i32.shape != base_owner_i32.shape:
        raise ValueError("base owner, left, and right arrays must have matching lengths")
    if op_pos_i32.shape != op_type_i32.shape:
        raise ValueError("op_pos_i32 length must match op_type_i32")
    if op_owner_i32.shape != op_type_i32.shape or op_left_i32.shape != op_type_i32.shape:
        raise ValueError("op owner/left length must match op_type_i32")
    if op_right_i32.shape != op_type_i32.shape:
        raise ValueError("op_right_i32 length must match op_type_i32")
    if track_count <= 0:
        raise ValueError("track_count must be positive")
    if frame_count <= 0:
        raise ValueError("frame_count must be positive")
    if boundary_f32.shape[0] <= 0 or boundary_f32.shape[0] > 32767:
        raise ValueError("endpoint record edit replay supports boundary count in [1, 32767]")
    if site_rgba_f32.shape[0] <= 0 or site_rgba_f32.shape[0] > 64:
        raise ValueError("endpoint record edit replay supports site count in [1, 64]")
    if config.far <= config.near:
        raise ValueError("config.far must be greater than config.near")
    if config.invalid_epsilon <= 0.0:
        raise ValueError("config.invalid_epsilon must be positive")
    if config.transmittance_threshold < 0.0:
        raise ValueError("config.transmittance_threshold must be nonnegative")
    _validate_segment_tape_offsets_cpu(
        base_offsets_i32.detach().cpu(),
        sample_count=track_count,
        segment_count=base_owner_i32.shape[0],
        max_segments_per_sample=max_runs_per_sample,
    )
    _validate_segment_tape_offsets_cpu(
        track_change_offsets_i32.detach().cpu(),
        sample_count=track_count,
        segment_count=change_frame_i32.shape[0],
        max_segments_per_sample=None,
    )
    _validate_segment_tape_offsets_cpu(
        op_offsets_i32.detach().cpu(),
        sample_count=change_frame_i32.shape[0],
        segment_count=op_type_i32.shape[0],
        max_segments_per_sample=None,
    )
    config_i32 = torch.tensor(
        [
            boundary_f32.shape[0],
            track_count,
            frame_count,
            site_rgba_f32.shape[0],
            base_owner_i32.shape[0],
            change_frame_i32.shape[0],
            op_type_i32.shape[0],
        ],
        device=boundary_f32.device,
        dtype=torch.int32,
    )
    config_f32 = torch.tensor(
        [config.near, config.far, config.invalid_epsilon, config.transmittance_threshold],
        device=boundary_f32.device,
        dtype=torch.float32,
    )
    return config_i32, config_f32


def endpoint_record_edit_rgba_depth_replay(
    boundary_f32: Tensor,
    rays_f32: Tensor,
    frame_t_f32: Tensor,
    base_offsets_i32: Tensor,
    base_owner_i32: Tensor,
    base_left_i32: Tensor,
    base_right_i32: Tensor,
    track_change_offsets_i32: Tensor,
    change_frame_i32: Tensor,
    op_offsets_i32: Tensor,
    op_type_i32: Tensor,
    op_pos_i32: Tensor,
    op_owner_i32: Tensor,
    op_left_i32: Tensor,
    op_right_i32: Tensor,
    site_rgba_f32: Tensor,
    config: RealRayReplayConfig,
    *,
    track_count: int,
    frame_count: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Replay endpoint owner+cut-id edit streams and recover depths from rays/boundaries."""
    boundary_f32 = boundary_f32.contiguous()
    rays_f32 = rays_f32.contiguous()
    frame_t_f32 = frame_t_f32.contiguous()
    base_offsets_i32 = base_offsets_i32.contiguous()
    base_owner_i32 = base_owner_i32.contiguous()
    base_left_i32 = base_left_i32.contiguous()
    base_right_i32 = base_right_i32.contiguous()
    track_change_offsets_i32 = track_change_offsets_i32.contiguous()
    change_frame_i32 = change_frame_i32.contiguous()
    op_offsets_i32 = op_offsets_i32.contiguous()
    op_type_i32 = op_type_i32.contiguous()
    op_pos_i32 = op_pos_i32.contiguous()
    op_owner_i32 = op_owner_i32.contiguous()
    op_left_i32 = op_left_i32.contiguous()
    op_right_i32 = op_right_i32.contiguous()
    site_rgba_f32 = site_rgba_f32.contiguous()
    config_i32, config_f32 = _endpoint_record_edit_config_tensors(
        boundary_f32=boundary_f32,
        rays_f32=rays_f32,
        frame_t_f32=frame_t_f32,
        base_offsets_i32=base_offsets_i32,
        base_owner_i32=base_owner_i32,
        base_left_i32=base_left_i32,
        base_right_i32=base_right_i32,
        track_change_offsets_i32=track_change_offsets_i32,
        change_frame_i32=change_frame_i32,
        op_offsets_i32=op_offsets_i32,
        op_type_i32=op_type_i32,
        op_pos_i32=op_pos_i32,
        op_owner_i32=op_owner_i32,
        op_left_i32=op_left_i32,
        op_right_i32=op_right_i32,
        site_rgba_f32=site_rgba_f32,
        config=config,
        track_count=track_count,
        frame_count=frame_count,
        max_runs_per_sample=None,
    )
    ops = torch.ops.world_foam_lane2_fused_slab_v0
    op_name = "endpoint_record_edit_rgba_depth_replay"
    if not hasattr(ops, op_name):
        raise RuntimeError(f"world_foam_lane2_fused_slab_v0 {op_name} op not found. Build this variant first.")
    return getattr(ops, op_name)(
        boundary_f32,
        rays_f32,
        frame_t_f32,
        base_offsets_i32,
        base_owner_i32,
        base_left_i32,
        base_right_i32,
        track_change_offsets_i32,
        change_frame_i32,
        op_offsets_i32,
        op_type_i32,
        op_pos_i32,
        op_owner_i32,
        op_left_i32,
        op_right_i32,
        site_rgba_f32,
        config_i32,
        config_f32,
    )


def endpoint_record_edit_block4_rgba_depth_replay(
    boundary_f32: Tensor,
    rays_f32: Tensor,
    frame_t_f32: Tensor,
    anchor_offsets_i32: Tensor,
    anchor_owner_i32: Tensor,
    anchor_left_i32: Tensor,
    anchor_right_i32: Tensor,
    track_block_change_offsets_i32: Tensor,
    change_frame_i32: Tensor,
    op_offsets_i32: Tensor,
    op_type_i32: Tensor,
    op_pos_i32: Tensor,
    op_owner_i32: Tensor,
    op_left_i32: Tensor,
    op_right_i32: Tensor,
    site_rgba_f32: Tensor,
    config: RealRayReplayConfig,
    *,
    track_count: int,
    frame_count: int,
    block_size: int = 4,
) -> tuple[Tensor, Tensor, Tensor]:
    """Replay endpoint edit streams from block anchor rows and recover depths from rays/boundaries."""
    if block_size <= 0:
        raise ValueError("endpoint_record_edit_block4_rgba_depth_replay requires block_size > 0")
    boundary_f32 = boundary_f32.contiguous()
    rays_f32 = rays_f32.contiguous()
    frame_t_f32 = frame_t_f32.contiguous()
    anchor_offsets_i32 = anchor_offsets_i32.contiguous()
    anchor_owner_i32 = anchor_owner_i32.contiguous()
    anchor_left_i32 = anchor_left_i32.contiguous()
    anchor_right_i32 = anchor_right_i32.contiguous()
    track_block_change_offsets_i32 = track_block_change_offsets_i32.contiguous()
    change_frame_i32 = change_frame_i32.contiguous()
    op_offsets_i32 = op_offsets_i32.contiguous()
    op_type_i32 = op_type_i32.contiguous()
    op_pos_i32 = op_pos_i32.contiguous()
    op_owner_i32 = op_owner_i32.contiguous()
    op_left_i32 = op_left_i32.contiguous()
    op_right_i32 = op_right_i32.contiguous()
    site_rgba_f32 = site_rgba_f32.contiguous()
    if rays_f32.ndim != 3 or rays_f32.shape[2] != 6:
        raise ValueError("rays_f32 must have shape [track_count, frame_count, 6]")
    if rays_f32.shape[0] != track_count or rays_f32.shape[1] != frame_count:
        raise ValueError("rays_f32 shape must match track_count/frame_count")
    if frame_t_f32.shape[0] != frame_count:
        raise ValueError("frame_t_f32 length must match frame_count")
    block_count = (frame_count + block_size - 1) // block_size
    config_i32 = torch.tensor(
        [
            boundary_f32.shape[0],
            track_count,
            frame_count,
            site_rgba_f32.shape[0],
            anchor_owner_i32.shape[0],
            change_frame_i32.shape[0],
            op_type_i32.shape[0],
            block_size,
            block_count,
        ],
        device=boundary_f32.device,
        dtype=torch.int32,
    )
    config_f32 = torch.tensor(
        [config.near, config.far, config.invalid_epsilon, config.transmittance_threshold],
        device=boundary_f32.device,
        dtype=torch.float32,
    )
    ops = torch.ops.world_foam_lane2_fused_slab_v0
    op_name = "endpoint_record_edit_block4_rgba_depth_replay"
    if not hasattr(ops, op_name):
        raise RuntimeError(f"world_foam_lane2_fused_slab_v0 {op_name} op not found. Build this variant first.")
    return getattr(ops, op_name)(
        boundary_f32,
        rays_f32,
        frame_t_f32,
        anchor_offsets_i32,
        anchor_owner_i32,
        anchor_left_i32,
        anchor_right_i32,
        track_block_change_offsets_i32,
        change_frame_i32,
        op_offsets_i32,
        op_type_i32,
        op_pos_i32,
        op_owner_i32,
        op_left_i32,
        op_right_i32,
        site_rgba_f32,
        config_i32,
        config_f32,
    )


def endpoint_record_edit_block_coeff_rgba_depth_replay(
    coeff_f32: Tensor,
    frame_t_f32: Tensor,
    anchor_offsets_i32: Tensor,
    anchor_owner_i32: Tensor,
    anchor_left_i32: Tensor,
    anchor_right_i32: Tensor,
    track_block_change_offsets_i32: Tensor,
    change_frame_i32: Tensor,
    op_offsets_i32: Tensor,
    op_type_i32: Tensor,
    op_pos_i32: Tensor,
    op_owner_i32: Tensor,
    op_left_i32: Tensor,
    op_right_i32: Tensor,
    site_rgba_f32: Tensor,
    config: RealRayReplayConfig,
    *,
    track_count: int,
    frame_count: int,
    boundary_count: int,
    block_size: int = 4,
) -> tuple[Tensor, Tensor, Tensor]:
    """Replay endpoint edit streams using precomputed per-track boundary depth coefficients."""
    if block_size <= 0:
        raise ValueError("endpoint_record_edit_block_coeff_rgba_depth_replay requires block_size > 0")
    coeff_f32 = coeff_f32.contiguous()
    frame_t_f32 = frame_t_f32.contiguous()
    anchor_offsets_i32 = anchor_offsets_i32.contiguous()
    anchor_owner_i32 = anchor_owner_i32.contiguous()
    anchor_left_i32 = anchor_left_i32.contiguous()
    anchor_right_i32 = anchor_right_i32.contiguous()
    track_block_change_offsets_i32 = track_block_change_offsets_i32.contiguous()
    change_frame_i32 = change_frame_i32.contiguous()
    op_offsets_i32 = op_offsets_i32.contiguous()
    op_type_i32 = op_type_i32.contiguous()
    op_pos_i32 = op_pos_i32.contiguous()
    op_owner_i32 = op_owner_i32.contiguous()
    op_left_i32 = op_left_i32.contiguous()
    op_right_i32 = op_right_i32.contiguous()
    site_rgba_f32 = site_rgba_f32.contiguous()
    if coeff_f32.ndim != 2 or coeff_f32.shape[1] != 4:
        raise ValueError("coeff_f32 must have shape [track_count * boundary_count, 4]")
    if coeff_f32.shape[0] != track_count * boundary_count:
        raise ValueError("coeff_f32 row count must match track_count * boundary_count")
    if frame_t_f32.shape[0] != frame_count:
        raise ValueError("frame_t_f32 length must match frame_count")
    block_count = (frame_count + block_size - 1) // block_size
    config_i32 = torch.tensor(
        [
            boundary_count,
            track_count,
            frame_count,
            site_rgba_f32.shape[0],
            anchor_owner_i32.shape[0],
            change_frame_i32.shape[0],
            op_type_i32.shape[0],
            block_size,
            block_count,
        ],
        device=coeff_f32.device,
        dtype=torch.int32,
    )
    config_f32 = torch.tensor(
        [config.near, config.far, config.invalid_epsilon, config.transmittance_threshold],
        device=coeff_f32.device,
        dtype=torch.float32,
    )
    ops = torch.ops.world_foam_lane2_fused_slab_v0
    op_name = "endpoint_record_edit_block_coeff_rgba_depth_replay"
    if not hasattr(ops, op_name):
        raise RuntimeError(f"world_foam_lane2_fused_slab_v0 {op_name} op not found. Build this variant first.")
    return getattr(ops, op_name)(
        coeff_f32,
        frame_t_f32,
        anchor_offsets_i32,
        anchor_owner_i32,
        anchor_left_i32,
        anchor_right_i32,
        track_block_change_offsets_i32,
        change_frame_i32,
        op_offsets_i32,
        op_type_i32,
        op_pos_i32,
        op_owner_i32,
        op_left_i32,
        op_right_i32,
        site_rgba_f32,
        config_i32,
        config_f32,
    )


def endpoint_record_edit_block_coeff_rgb_replay(
    coeff_f32: Tensor,
    frame_t_f32: Tensor,
    anchor_offsets_i32: Tensor,
    anchor_owner_i32: Tensor,
    anchor_left_i32: Tensor,
    anchor_right_i32: Tensor,
    track_block_change_offsets_i32: Tensor,
    change_frame_i32: Tensor,
    op_offsets_i32: Tensor,
    op_type_i32: Tensor,
    op_pos_i32: Tensor,
    op_owner_i32: Tensor,
    op_left_i32: Tensor,
    op_right_i32: Tensor,
    site_rgba_f32: Tensor,
    config: RealRayReplayConfig,
    *,
    track_count: int,
    frame_count: int,
    boundary_count: int,
    block_size: int = 4,
) -> Tensor:
    """Replay coefficient-cached block edits and return RGB only for train-path timing."""
    if block_size <= 0:
        raise ValueError("endpoint_record_edit_block_coeff_rgb_replay requires block_size > 0")
    coeff_f32 = coeff_f32.contiguous()
    frame_t_f32 = frame_t_f32.contiguous()
    anchor_offsets_i32 = anchor_offsets_i32.contiguous()
    anchor_owner_i32 = anchor_owner_i32.contiguous()
    anchor_left_i32 = anchor_left_i32.contiguous()
    anchor_right_i32 = anchor_right_i32.contiguous()
    track_block_change_offsets_i32 = track_block_change_offsets_i32.contiguous()
    change_frame_i32 = change_frame_i32.contiguous()
    op_offsets_i32 = op_offsets_i32.contiguous()
    op_type_i32 = op_type_i32.contiguous()
    op_pos_i32 = op_pos_i32.contiguous()
    op_owner_i32 = op_owner_i32.contiguous()
    op_left_i32 = op_left_i32.contiguous()
    op_right_i32 = op_right_i32.contiguous()
    site_rgba_f32 = site_rgba_f32.contiguous()
    if coeff_f32.ndim != 2 or coeff_f32.shape[1] != 4:
        raise ValueError("coeff_f32 must have shape [track_count * boundary_count, 4]")
    if coeff_f32.shape[0] != track_count * boundary_count:
        raise ValueError("coeff_f32 row count must match track_count * boundary_count")
    if frame_t_f32.shape[0] != frame_count:
        raise ValueError("frame_t_f32 length must match frame_count")
    block_count = (frame_count + block_size - 1) // block_size
    config_i32 = torch.tensor(
        [
            boundary_count,
            track_count,
            frame_count,
            site_rgba_f32.shape[0],
            anchor_owner_i32.shape[0],
            change_frame_i32.shape[0],
            op_type_i32.shape[0],
            block_size,
            block_count,
        ],
        device=coeff_f32.device,
        dtype=torch.int32,
    )
    config_f32 = torch.tensor(
        [config.near, config.far, config.invalid_epsilon, config.transmittance_threshold],
        device=coeff_f32.device,
        dtype=torch.float32,
    )
    ops = torch.ops.world_foam_lane2_fused_slab_v0
    op_name = "endpoint_record_edit_block_coeff_rgb_replay"
    if not hasattr(ops, op_name):
        raise RuntimeError(f"world_foam_lane2_fused_slab_v0 {op_name} op not found. Build this variant first.")
    return getattr(ops, op_name)(
        coeff_f32,
        frame_t_f32,
        anchor_offsets_i32,
        anchor_owner_i32,
        anchor_left_i32,
        anchor_right_i32,
        track_block_change_offsets_i32,
        change_frame_i32,
        op_offsets_i32,
        op_type_i32,
        op_pos_i32,
        op_owner_i32,
        op_left_i32,
        op_right_i32,
        site_rgba_f32,
        config_i32,
        config_f32,
    )


def endpoint_record_edit_block_coeff16_rgba_depth_replay(
    coeff_f16: Tensor,
    frame_t_f32: Tensor,
    anchor_offsets_i32: Tensor,
    anchor_owner_i32: Tensor,
    anchor_left_i32: Tensor,
    anchor_right_i32: Tensor,
    track_block_change_offsets_i32: Tensor,
    change_frame_i32: Tensor,
    op_offsets_i32: Tensor,
    op_type_i32: Tensor,
    op_pos_i32: Tensor,
    op_owner_i32: Tensor,
    op_left_i32: Tensor,
    op_right_i32: Tensor,
    site_rgba_f32: Tensor,
    config: RealRayReplayConfig,
    *,
    track_count: int,
    frame_count: int,
    boundary_count: int,
    block_size: int = 4,
) -> tuple[Tensor, Tensor, Tensor]:
    """Replay endpoint edit streams using float16 per-track boundary depth coefficients."""
    if block_size <= 0:
        raise ValueError("endpoint_record_edit_block_coeff16_rgba_depth_replay requires block_size > 0")
    coeff_f16 = coeff_f16.contiguous()
    frame_t_f32 = frame_t_f32.contiguous()
    anchor_offsets_i32 = anchor_offsets_i32.contiguous()
    anchor_owner_i32 = anchor_owner_i32.contiguous()
    anchor_left_i32 = anchor_left_i32.contiguous()
    anchor_right_i32 = anchor_right_i32.contiguous()
    track_block_change_offsets_i32 = track_block_change_offsets_i32.contiguous()
    change_frame_i32 = change_frame_i32.contiguous()
    op_offsets_i32 = op_offsets_i32.contiguous()
    op_type_i32 = op_type_i32.contiguous()
    op_pos_i32 = op_pos_i32.contiguous()
    op_owner_i32 = op_owner_i32.contiguous()
    op_left_i32 = op_left_i32.contiguous()
    op_right_i32 = op_right_i32.contiguous()
    site_rgba_f32 = site_rgba_f32.contiguous()
    if coeff_f16.dtype != torch.float16:
        raise ValueError("coeff_f16 must be float16")
    if coeff_f16.ndim != 2 or coeff_f16.shape[1] != 4:
        raise ValueError("coeff_f16 must have shape [track_count * boundary_count, 4]")
    if coeff_f16.shape[0] != track_count * boundary_count:
        raise ValueError("coeff_f16 row count must match track_count * boundary_count")
    if frame_t_f32.shape[0] != frame_count:
        raise ValueError("frame_t_f32 length must match frame_count")
    block_count = (frame_count + block_size - 1) // block_size
    config_i32 = torch.tensor(
        [
            boundary_count,
            track_count,
            frame_count,
            site_rgba_f32.shape[0],
            anchor_owner_i32.shape[0],
            change_frame_i32.shape[0],
            op_type_i32.shape[0],
            block_size,
            block_count,
        ],
        device=coeff_f16.device,
        dtype=torch.int32,
    )
    config_f32 = torch.tensor(
        [config.near, config.far, config.invalid_epsilon, config.transmittance_threshold],
        device=coeff_f16.device,
        dtype=torch.float32,
    )
    ops = torch.ops.world_foam_lane2_fused_slab_v0
    op_name = "endpoint_record_edit_block_coeff16_rgba_depth_replay"
    if not hasattr(ops, op_name):
        raise RuntimeError(f"world_foam_lane2_fused_slab_v0 {op_name} op not found. Build this variant first.")
    return getattr(ops, op_name)(
        coeff_f16,
        frame_t_f32,
        anchor_offsets_i32,
        anchor_owner_i32,
        anchor_left_i32,
        anchor_right_i32,
        track_block_change_offsets_i32,
        change_frame_i32,
        op_offsets_i32,
        op_type_i32,
        op_pos_i32,
        op_owner_i32,
        op_left_i32,
        op_right_i32,
        site_rgba_f32,
        config_i32,
        config_f32,
    )


def endpoint_record_edit_rgba_depth_replay_trackloop(
    boundary_f32: Tensor,
    rays_f32: Tensor,
    frame_t_f32: Tensor,
    base_offsets_i32: Tensor,
    base_owner_i32: Tensor,
    base_left_i32: Tensor,
    base_right_i32: Tensor,
    track_change_offsets_i32: Tensor,
    change_frame_i32: Tensor,
    op_offsets_i32: Tensor,
    op_type_i32: Tensor,
    op_pos_i32: Tensor,
    op_owner_i32: Tensor,
    op_left_i32: Tensor,
    op_right_i32: Tensor,
    site_rgba_f32: Tensor,
    config: RealRayReplayConfig,
    *,
    track_count: int,
    frame_count: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Replay endpoint edit streams by walking each track through frames in one Metal thread."""
    boundary_f32 = boundary_f32.contiguous()
    rays_f32 = rays_f32.contiguous()
    frame_t_f32 = frame_t_f32.contiguous()
    base_offsets_i32 = base_offsets_i32.contiguous()
    base_owner_i32 = base_owner_i32.contiguous()
    base_left_i32 = base_left_i32.contiguous()
    base_right_i32 = base_right_i32.contiguous()
    track_change_offsets_i32 = track_change_offsets_i32.contiguous()
    change_frame_i32 = change_frame_i32.contiguous()
    op_offsets_i32 = op_offsets_i32.contiguous()
    op_type_i32 = op_type_i32.contiguous()
    op_pos_i32 = op_pos_i32.contiguous()
    op_owner_i32 = op_owner_i32.contiguous()
    op_left_i32 = op_left_i32.contiguous()
    op_right_i32 = op_right_i32.contiguous()
    site_rgba_f32 = site_rgba_f32.contiguous()
    config_i32, config_f32 = _endpoint_record_edit_config_tensors(
        boundary_f32=boundary_f32,
        rays_f32=rays_f32,
        frame_t_f32=frame_t_f32,
        base_offsets_i32=base_offsets_i32,
        base_owner_i32=base_owner_i32,
        base_left_i32=base_left_i32,
        base_right_i32=base_right_i32,
        track_change_offsets_i32=track_change_offsets_i32,
        change_frame_i32=change_frame_i32,
        op_offsets_i32=op_offsets_i32,
        op_type_i32=op_type_i32,
        op_pos_i32=op_pos_i32,
        op_owner_i32=op_owner_i32,
        op_left_i32=op_left_i32,
        op_right_i32=op_right_i32,
        site_rgba_f32=site_rgba_f32,
        config=config,
        track_count=track_count,
        frame_count=frame_count,
        max_runs_per_sample=None,
    )
    ops = torch.ops.world_foam_lane2_fused_slab_v0
    op_name = "endpoint_record_edit_rgba_depth_replay_trackloop"
    if not hasattr(ops, op_name):
        raise RuntimeError(f"world_foam_lane2_fused_slab_v0 {op_name} op not found. Build this variant first.")
    return getattr(ops, op_name)(
        boundary_f32,
        rays_f32,
        frame_t_f32,
        base_offsets_i32,
        base_owner_i32,
        base_left_i32,
        base_right_i32,
        track_change_offsets_i32,
        change_frame_i32,
        op_offsets_i32,
        op_type_i32,
        op_pos_i32,
        op_owner_i32,
        op_left_i32,
        op_right_i32,
        site_rgba_f32,
        config_i32,
        config_f32,
    )


def endpoint_record_edit_rgba_depth_replay_framegroup16(
    boundary_f32: Tensor,
    rays_f32: Tensor,
    frame_t_f32: Tensor,
    base_offsets_i32: Tensor,
    base_owner_i32: Tensor,
    base_left_i32: Tensor,
    base_right_i32: Tensor,
    track_change_offsets_i32: Tensor,
    change_frame_i32: Tensor,
    op_offsets_i32: Tensor,
    op_type_i32: Tensor,
    op_pos_i32: Tensor,
    op_owner_i32: Tensor,
    op_left_i32: Tensor,
    op_right_i32: Tensor,
    site_rgba_f32: Tensor,
    config: RealRayReplayConfig,
    *,
    track_count: int,
    frame_count: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Replay endpoint edit streams with one threadgroup per track and up to 16 frame lanes."""
    if frame_count > 16:
        raise ValueError("endpoint_record_edit_rgba_depth_replay_framegroup16 supports at most 16 frames")
    boundary_f32 = boundary_f32.contiguous()
    rays_f32 = rays_f32.contiguous()
    frame_t_f32 = frame_t_f32.contiguous()
    base_offsets_i32 = base_offsets_i32.contiguous()
    base_owner_i32 = base_owner_i32.contiguous()
    base_left_i32 = base_left_i32.contiguous()
    base_right_i32 = base_right_i32.contiguous()
    track_change_offsets_i32 = track_change_offsets_i32.contiguous()
    change_frame_i32 = change_frame_i32.contiguous()
    op_offsets_i32 = op_offsets_i32.contiguous()
    op_type_i32 = op_type_i32.contiguous()
    op_pos_i32 = op_pos_i32.contiguous()
    op_owner_i32 = op_owner_i32.contiguous()
    op_left_i32 = op_left_i32.contiguous()
    op_right_i32 = op_right_i32.contiguous()
    site_rgba_f32 = site_rgba_f32.contiguous()
    config_i32, config_f32 = _endpoint_record_edit_config_tensors(
        boundary_f32=boundary_f32,
        rays_f32=rays_f32,
        frame_t_f32=frame_t_f32,
        base_offsets_i32=base_offsets_i32,
        base_owner_i32=base_owner_i32,
        base_left_i32=base_left_i32,
        base_right_i32=base_right_i32,
        track_change_offsets_i32=track_change_offsets_i32,
        change_frame_i32=change_frame_i32,
        op_offsets_i32=op_offsets_i32,
        op_type_i32=op_type_i32,
        op_pos_i32=op_pos_i32,
        op_owner_i32=op_owner_i32,
        op_left_i32=op_left_i32,
        op_right_i32=op_right_i32,
        site_rgba_f32=site_rgba_f32,
        config=config,
        track_count=track_count,
        frame_count=frame_count,
        max_runs_per_sample=None,
    )
    ops = torch.ops.world_foam_lane2_fused_slab_v0
    op_name = "endpoint_record_edit_rgba_depth_replay_framegroup16"
    if not hasattr(ops, op_name):
        raise RuntimeError(f"world_foam_lane2_fused_slab_v0 {op_name} op not found. Build this variant first.")
    return getattr(ops, op_name)(
        boundary_f32,
        rays_f32,
        frame_t_f32,
        base_offsets_i32,
        base_owner_i32,
        base_left_i32,
        base_right_i32,
        track_change_offsets_i32,
        change_frame_i32,
        op_offsets_i32,
        op_type_i32,
        op_pos_i32,
        op_owner_i32,
        op_left_i32,
        op_right_i32,
        site_rgba_f32,
        config_i32,
        config_f32,
    )


def endpoint_record_edit_vjp_direct_atomic_grad_only(
    boundary_f32: Tensor,
    rays_f32: Tensor,
    frame_t_f32: Tensor,
    base_offsets_i32: Tensor,
    base_owner_i32: Tensor,
    base_left_i32: Tensor,
    base_right_i32: Tensor,
    track_change_offsets_i32: Tensor,
    change_frame_i32: Tensor,
    op_offsets_i32: Tensor,
    op_type_i32: Tensor,
    op_pos_i32: Tensor,
    op_owner_i32: Tensor,
    op_left_i32: Tensor,
    op_right_i32: Tensor,
    site_rgba_f32: Tensor,
    grad_rgb_f32: Tensor,
    grad_alpha_f32: Tensor,
    grad_depth_f32: Tensor,
    config: RealRayReplayConfig,
    *,
    track_count: int,
    frame_count: int,
) -> Tensor:
    """Accumulate site RGBA gradients for endpoint owner+cut-id edit streams."""
    boundary_f32 = boundary_f32.contiguous()
    rays_f32 = rays_f32.contiguous()
    frame_t_f32 = frame_t_f32.contiguous()
    base_offsets_i32 = base_offsets_i32.contiguous()
    base_owner_i32 = base_owner_i32.contiguous()
    base_left_i32 = base_left_i32.contiguous()
    base_right_i32 = base_right_i32.contiguous()
    track_change_offsets_i32 = track_change_offsets_i32.contiguous()
    change_frame_i32 = change_frame_i32.contiguous()
    op_offsets_i32 = op_offsets_i32.contiguous()
    op_type_i32 = op_type_i32.contiguous()
    op_pos_i32 = op_pos_i32.contiguous()
    op_owner_i32 = op_owner_i32.contiguous()
    op_left_i32 = op_left_i32.contiguous()
    op_right_i32 = op_right_i32.contiguous()
    site_rgba_f32 = site_rgba_f32.contiguous()
    grad_rgb_f32 = grad_rgb_f32.contiguous()
    grad_alpha_f32 = grad_alpha_f32.contiguous()
    grad_depth_f32 = grad_depth_f32.contiguous()
    config_i32, config_f32 = _endpoint_record_edit_config_tensors(
        boundary_f32=boundary_f32,
        rays_f32=rays_f32,
        frame_t_f32=frame_t_f32,
        base_offsets_i32=base_offsets_i32,
        base_owner_i32=base_owner_i32,
        base_left_i32=base_left_i32,
        base_right_i32=base_right_i32,
        track_change_offsets_i32=track_change_offsets_i32,
        change_frame_i32=change_frame_i32,
        op_offsets_i32=op_offsets_i32,
        op_type_i32=op_type_i32,
        op_pos_i32=op_pos_i32,
        op_owner_i32=op_owner_i32,
        op_left_i32=op_left_i32,
        op_right_i32=op_right_i32,
        site_rgba_f32=site_rgba_f32,
        config=config,
        track_count=track_count,
        frame_count=frame_count,
        max_runs_per_sample=129,
    )
    if grad_rgb_f32.device.type != "mps" or grad_rgb_f32.dtype != torch.float32:
        raise ValueError("grad_rgb_f32 must be float32 on MPS")
    if grad_rgb_f32.shape != (track_count, frame_count, 3):
        raise ValueError("grad_rgb_f32 must have shape [track_count, frame_count, 3]")
    if grad_alpha_f32.device.type != "mps" or grad_alpha_f32.dtype != torch.float32:
        raise ValueError("grad_alpha_f32 must be float32 on MPS")
    if grad_alpha_f32.shape != (track_count, frame_count):
        raise ValueError("grad_alpha_f32 must have shape [track_count, frame_count]")
    if grad_depth_f32.device.type != "mps" or grad_depth_f32.dtype != torch.float32:
        raise ValueError("grad_depth_f32 must be float32 on MPS")
    if grad_depth_f32.shape != (track_count, frame_count):
        raise ValueError("grad_depth_f32 must have shape [track_count, frame_count]")
    ops = torch.ops.world_foam_lane2_fused_slab_v0
    op_name = "endpoint_record_edit_vjp_direct_atomic_grad_only"
    if not hasattr(ops, op_name):
        raise RuntimeError(f"world_foam_lane2_fused_slab_v0 {op_name} op not found. Build this variant first.")
    return getattr(ops, op_name)(
        boundary_f32,
        rays_f32,
        frame_t_f32,
        base_offsets_i32,
        base_owner_i32,
        base_left_i32,
        base_right_i32,
        track_change_offsets_i32,
        change_frame_i32,
        op_offsets_i32,
        op_type_i32,
        op_pos_i32,
        op_owner_i32,
        op_left_i32,
        op_right_i32,
        site_rgba_f32,
        grad_rgb_f32,
        grad_alpha_f32,
        grad_depth_f32,
        config_i32,
        config_f32,
    )


def endpoint_record_edit_vjp_direct_atomic_rgb_only(
    boundary_f32: Tensor,
    rays_f32: Tensor,
    frame_t_f32: Tensor,
    base_offsets_i32: Tensor,
    base_owner_i32: Tensor,
    base_left_i32: Tensor,
    base_right_i32: Tensor,
    track_change_offsets_i32: Tensor,
    change_frame_i32: Tensor,
    op_offsets_i32: Tensor,
    op_type_i32: Tensor,
    op_pos_i32: Tensor,
    op_owner_i32: Tensor,
    op_left_i32: Tensor,
    op_right_i32: Tensor,
    site_rgba_f32: Tensor,
    grad_rgb_f32: Tensor,
    config: RealRayReplayConfig,
    *,
    track_count: int,
    frame_count: int,
) -> Tensor:
    """Accumulate endpoint edit gradients for RGB-only losses."""
    boundary_f32 = boundary_f32.contiguous()
    rays_f32 = rays_f32.contiguous()
    frame_t_f32 = frame_t_f32.contiguous()
    base_offsets_i32 = base_offsets_i32.contiguous()
    base_owner_i32 = base_owner_i32.contiguous()
    base_left_i32 = base_left_i32.contiguous()
    base_right_i32 = base_right_i32.contiguous()
    track_change_offsets_i32 = track_change_offsets_i32.contiguous()
    change_frame_i32 = change_frame_i32.contiguous()
    op_offsets_i32 = op_offsets_i32.contiguous()
    op_type_i32 = op_type_i32.contiguous()
    op_pos_i32 = op_pos_i32.contiguous()
    op_owner_i32 = op_owner_i32.contiguous()
    op_left_i32 = op_left_i32.contiguous()
    op_right_i32 = op_right_i32.contiguous()
    site_rgba_f32 = site_rgba_f32.contiguous()
    grad_rgb_f32 = grad_rgb_f32.contiguous()
    config_i32, config_f32 = _endpoint_record_edit_config_tensors(
        boundary_f32=boundary_f32,
        rays_f32=rays_f32,
        frame_t_f32=frame_t_f32,
        base_offsets_i32=base_offsets_i32,
        base_owner_i32=base_owner_i32,
        base_left_i32=base_left_i32,
        base_right_i32=base_right_i32,
        track_change_offsets_i32=track_change_offsets_i32,
        change_frame_i32=change_frame_i32,
        op_offsets_i32=op_offsets_i32,
        op_type_i32=op_type_i32,
        op_pos_i32=op_pos_i32,
        op_owner_i32=op_owner_i32,
        op_left_i32=op_left_i32,
        op_right_i32=op_right_i32,
        site_rgba_f32=site_rgba_f32,
        config=config,
        track_count=track_count,
        frame_count=frame_count,
        max_runs_per_sample=129,
    )
    if grad_rgb_f32.device.type != "mps" or grad_rgb_f32.dtype != torch.float32:
        raise ValueError("grad_rgb_f32 must be float32 on MPS")
    if grad_rgb_f32.shape != (track_count, frame_count, 3):
        raise ValueError("grad_rgb_f32 must have shape [track_count, frame_count, 3]")
    ops = torch.ops.world_foam_lane2_fused_slab_v0
    op_name = "endpoint_record_edit_vjp_direct_atomic_rgb_only"
    if not hasattr(ops, op_name):
        raise RuntimeError(f"world_foam_lane2_fused_slab_v0 {op_name} op not found. Build this variant first.")
    return getattr(ops, op_name)(
        boundary_f32,
        rays_f32,
        frame_t_f32,
        base_offsets_i32,
        base_owner_i32,
        base_left_i32,
        base_right_i32,
        track_change_offsets_i32,
        change_frame_i32,
        op_offsets_i32,
        op_type_i32,
        op_pos_i32,
        op_owner_i32,
        op_left_i32,
        op_right_i32,
        site_rgba_f32,
        grad_rgb_f32,
        config_i32,
        config_f32,
    )


def endpoint_record_edit_mse_vjp_direct_atomic_rgb_only(
    boundary_f32: Tensor,
    rays_f32: Tensor,
    frame_t_f32: Tensor,
    base_offsets_i32: Tensor,
    base_owner_i32: Tensor,
    base_left_i32: Tensor,
    base_right_i32: Tensor,
    track_change_offsets_i32: Tensor,
    change_frame_i32: Tensor,
    op_offsets_i32: Tensor,
    op_type_i32: Tensor,
    op_pos_i32: Tensor,
    op_owner_i32: Tensor,
    op_left_i32: Tensor,
    op_right_i32: Tensor,
    site_rgba_f32: Tensor,
    target_rgb_f32: Tensor,
    config: RealRayReplayConfig,
    *,
    track_count: int,
    frame_count: int,
) -> tuple[Tensor, Tensor]:
    """Compute RGB MSE and site gradients for compact endpoint edit rows."""
    boundary_f32 = boundary_f32.contiguous()
    rays_f32 = rays_f32.contiguous()
    frame_t_f32 = frame_t_f32.contiguous()
    base_offsets_i32 = base_offsets_i32.contiguous()
    base_owner_i32 = base_owner_i32.contiguous()
    base_left_i32 = base_left_i32.contiguous()
    base_right_i32 = base_right_i32.contiguous()
    track_change_offsets_i32 = track_change_offsets_i32.contiguous()
    change_frame_i32 = change_frame_i32.contiguous()
    op_offsets_i32 = op_offsets_i32.contiguous()
    op_type_i32 = op_type_i32.contiguous()
    op_pos_i32 = op_pos_i32.contiguous()
    op_owner_i32 = op_owner_i32.contiguous()
    op_left_i32 = op_left_i32.contiguous()
    op_right_i32 = op_right_i32.contiguous()
    site_rgba_f32 = site_rgba_f32.contiguous()
    target_rgb_f32 = target_rgb_f32.contiguous()
    config_i32, config_f32 = _endpoint_record_edit_config_tensors(
        boundary_f32=boundary_f32,
        rays_f32=rays_f32,
        frame_t_f32=frame_t_f32,
        base_offsets_i32=base_offsets_i32,
        base_owner_i32=base_owner_i32,
        base_left_i32=base_left_i32,
        base_right_i32=base_right_i32,
        track_change_offsets_i32=track_change_offsets_i32,
        change_frame_i32=change_frame_i32,
        op_offsets_i32=op_offsets_i32,
        op_type_i32=op_type_i32,
        op_pos_i32=op_pos_i32,
        op_owner_i32=op_owner_i32,
        op_left_i32=op_left_i32,
        op_right_i32=op_right_i32,
        site_rgba_f32=site_rgba_f32,
        config=config,
        track_count=track_count,
        frame_count=frame_count,
        max_runs_per_sample=129,
    )
    if target_rgb_f32.device.type != "mps" or target_rgb_f32.dtype != torch.float32:
        raise ValueError("target_rgb_f32 must be float32 on MPS")
    if target_rgb_f32.shape != (track_count, frame_count, 3):
        raise ValueError("target_rgb_f32 must have shape [track_count, frame_count, 3]")
    ops = torch.ops.world_foam_lane2_fused_slab_v0
    op_name = "endpoint_record_edit_mse_vjp_direct_atomic_rgb_only"
    if not hasattr(ops, op_name):
        raise RuntimeError(f"world_foam_lane2_fused_slab_v0 {op_name} op not found. Build this variant first.")
    return getattr(ops, op_name)(
        boundary_f32,
        rays_f32,
        frame_t_f32,
        base_offsets_i32,
        base_owner_i32,
        base_left_i32,
        base_right_i32,
        track_change_offsets_i32,
        change_frame_i32,
        op_offsets_i32,
        op_type_i32,
        op_pos_i32,
        op_owner_i32,
        op_left_i32,
        op_right_i32,
        site_rgba_f32,
        target_rgb_f32,
        config_i32,
        config_f32,
    )


def endpoint_record_edit_coeff16_mse_vjp_direct_atomic_rgb_only(
    coeff_f16: Tensor,
    frame_t_f32: Tensor,
    base_offsets_i32: Tensor,
    base_owner_i32: Tensor,
    base_left_i32: Tensor,
    base_right_i32: Tensor,
    track_change_offsets_i32: Tensor,
    change_frame_i32: Tensor,
    op_offsets_i32: Tensor,
    op_type_i32: Tensor,
    op_pos_i32: Tensor,
    op_owner_i32: Tensor,
    op_left_i32: Tensor,
    op_right_i32: Tensor,
    site_rgba_f32: Tensor,
    target_rgb_f32: Tensor,
    config: RealRayReplayConfig,
    *,
    track_count: int,
    frame_count: int,
    boundary_count: int,
) -> tuple[Tensor, Tensor]:
    """Compute RGB MSE/VJP for raw endpoint edits with coeff16 cut depths."""
    coeff_f16 = coeff_f16.contiguous()
    frame_t_f32 = frame_t_f32.contiguous()
    base_offsets_i32 = base_offsets_i32.contiguous()
    base_owner_i32 = base_owner_i32.contiguous()
    base_left_i32 = base_left_i32.contiguous()
    base_right_i32 = base_right_i32.contiguous()
    track_change_offsets_i32 = track_change_offsets_i32.contiguous()
    change_frame_i32 = change_frame_i32.contiguous()
    op_offsets_i32 = op_offsets_i32.contiguous()
    op_type_i32 = op_type_i32.contiguous()
    op_pos_i32 = op_pos_i32.contiguous()
    op_owner_i32 = op_owner_i32.contiguous()
    op_left_i32 = op_left_i32.contiguous()
    op_right_i32 = op_right_i32.contiguous()
    site_rgba_f32 = site_rgba_f32.contiguous()
    target_rgb_f32 = target_rgb_f32.contiguous()
    _require_mps_tensor(coeff_f16, name="coeff_f16", dtype=torch.float16, cols=4)
    _require_mps_tensor(frame_t_f32, name="frame_t_f32", dtype=torch.float32, cols=None)
    _require_mps_tensor(base_offsets_i32, name="base_offsets_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(base_owner_i32, name="base_owner_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(base_left_i32, name="base_left_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(base_right_i32, name="base_right_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(track_change_offsets_i32, name="track_change_offsets_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(change_frame_i32, name="change_frame_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(op_offsets_i32, name="op_offsets_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(op_type_i32, name="op_type_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(op_pos_i32, name="op_pos_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(op_owner_i32, name="op_owner_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(op_left_i32, name="op_left_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(op_right_i32, name="op_right_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(site_rgba_f32, name="site_rgba_f32", dtype=torch.float32, cols=4)
    if track_count <= 0 or frame_count <= 0 or boundary_count <= 0:
        raise ValueError("track_count, frame_count, and boundary_count must be positive")
    if boundary_count > 32767:
        raise ValueError("endpoint record edit coeff16 replay supports boundary count <= 32767")
    if coeff_f16.shape[0] != track_count * boundary_count:
        raise ValueError("coeff_f16 row count must be track_count * boundary_count")
    if frame_t_f32.shape[0] != frame_count:
        raise ValueError("frame_t_f32 must have shape [frame_count]")
    if base_left_i32.shape != base_owner_i32.shape or base_right_i32.shape != base_owner_i32.shape:
        raise ValueError("base owner, left, and right arrays must have matching lengths")
    if op_pos_i32.shape != op_type_i32.shape:
        raise ValueError("op_pos_i32 length must match op_type_i32")
    if op_owner_i32.shape != op_type_i32.shape or op_left_i32.shape != op_type_i32.shape:
        raise ValueError("op owner/left length must match op_type_i32")
    if op_right_i32.shape != op_type_i32.shape:
        raise ValueError("op_right_i32 length must match op_type_i32")
    if target_rgb_f32.device.type != "mps" or target_rgb_f32.dtype != torch.float32:
        raise ValueError("target_rgb_f32 must be float32 on MPS")
    if target_rgb_f32.shape != (track_count, frame_count, 3):
        raise ValueError("target_rgb_f32 must have shape [track_count, frame_count, 3]")
    if site_rgba_f32.shape[0] <= 0 or site_rgba_f32.shape[0] > 64:
        raise ValueError("endpoint record edit coeff16 replay supports site count in [1, 64]")
    if config.far <= config.near:
        raise ValueError("config.far must be greater than config.near")
    if config.invalid_epsilon <= 0.0:
        raise ValueError("config.invalid_epsilon must be positive")
    if config.transmittance_threshold < 0.0:
        raise ValueError("config.transmittance_threshold must be nonnegative")
    _validate_segment_tape_offsets_cpu(
        base_offsets_i32.detach().cpu(),
        sample_count=track_count,
        segment_count=base_owner_i32.shape[0],
        max_segments_per_sample=129,
    )
    _validate_segment_tape_offsets_cpu(
        track_change_offsets_i32.detach().cpu(),
        sample_count=track_count,
        segment_count=change_frame_i32.shape[0],
        max_segments_per_sample=None,
    )
    _validate_segment_tape_offsets_cpu(
        op_offsets_i32.detach().cpu(),
        sample_count=change_frame_i32.shape[0],
        segment_count=op_type_i32.shape[0],
        max_segments_per_sample=None,
    )
    config_i32 = torch.tensor(
        [
            boundary_count,
            track_count,
            frame_count,
            site_rgba_f32.shape[0],
            base_owner_i32.shape[0],
            change_frame_i32.shape[0],
            op_type_i32.shape[0],
        ],
        device=coeff_f16.device,
        dtype=torch.int32,
    )
    config_f32 = torch.tensor(
        [config.near, config.far, config.invalid_epsilon, config.transmittance_threshold],
        device=coeff_f16.device,
        dtype=torch.float32,
    )
    ops = torch.ops.world_foam_lane2_fused_slab_v0
    op_name = "endpoint_record_edit_coeff16_mse_vjp_direct_atomic_rgb_only"
    if not hasattr(ops, op_name):
        raise RuntimeError(f"world_foam_lane2_fused_slab_v0 {op_name} op not found. Build this variant first.")
    return getattr(ops, op_name)(
        coeff_f16,
        frame_t_f32,
        base_offsets_i32,
        base_owner_i32,
        base_left_i32,
        base_right_i32,
        track_change_offsets_i32,
        change_frame_i32,
        op_offsets_i32,
        op_type_i32,
        op_pos_i32,
        op_owner_i32,
        op_left_i32,
        op_right_i32,
        site_rgba_f32,
        target_rgb_f32,
        config_i32,
        config_f32,
    )


def endpoint_record_edit_block4_vjp_direct_atomic_rgb_only(
    boundary_f32: Tensor,
    rays_f32: Tensor,
    frame_t_f32: Tensor,
    anchor_offsets_i32: Tensor,
    anchor_owner_i32: Tensor,
    anchor_left_i32: Tensor,
    anchor_right_i32: Tensor,
    track_block_change_offsets_i32: Tensor,
    change_frame_i32: Tensor,
    op_offsets_i32: Tensor,
    op_type_i32: Tensor,
    op_pos_i32: Tensor,
    op_owner_i32: Tensor,
    op_left_i32: Tensor,
    op_right_i32: Tensor,
    site_rgba_f32: Tensor,
    grad_rgb_f32: Tensor,
    config: RealRayReplayConfig,
    *,
    track_count: int,
    frame_count: int,
    block_size: int = 4,
) -> Tensor:
    """Accumulate RGB-only site gradients from block-anchored endpoint edit rows."""
    if block_size <= 0:
        raise ValueError("endpoint_record_edit_block4_vjp_direct_atomic_rgb_only requires block_size > 0")
    boundary_f32 = boundary_f32.contiguous()
    rays_f32 = rays_f32.contiguous()
    frame_t_f32 = frame_t_f32.contiguous()
    anchor_offsets_i32 = anchor_offsets_i32.contiguous()
    anchor_owner_i32 = anchor_owner_i32.contiguous()
    anchor_left_i32 = anchor_left_i32.contiguous()
    anchor_right_i32 = anchor_right_i32.contiguous()
    track_block_change_offsets_i32 = track_block_change_offsets_i32.contiguous()
    change_frame_i32 = change_frame_i32.contiguous()
    op_offsets_i32 = op_offsets_i32.contiguous()
    op_type_i32 = op_type_i32.contiguous()
    op_pos_i32 = op_pos_i32.contiguous()
    op_owner_i32 = op_owner_i32.contiguous()
    op_left_i32 = op_left_i32.contiguous()
    op_right_i32 = op_right_i32.contiguous()
    site_rgba_f32 = site_rgba_f32.contiguous()
    grad_rgb_f32 = grad_rgb_f32.contiguous()
    if rays_f32.ndim != 3 or rays_f32.shape[2] != 6:
        raise ValueError("rays_f32 must have shape [track_count, frame_count, 6]")
    if rays_f32.shape[0] != track_count or rays_f32.shape[1] != frame_count:
        raise ValueError("rays_f32 shape must match track_count/frame_count")
    if frame_t_f32.shape[0] != frame_count:
        raise ValueError("frame_t_f32 length must match frame_count")
    if grad_rgb_f32.device.type != "mps" or grad_rgb_f32.dtype != torch.float32:
        raise ValueError("grad_rgb_f32 must be float32 on MPS")
    if grad_rgb_f32.shape != (track_count, frame_count, 3):
        raise ValueError("grad_rgb_f32 must have shape [track_count, frame_count, 3]")
    block_count = (frame_count + block_size - 1) // block_size
    config_i32 = torch.tensor(
        [
            boundary_f32.shape[0],
            track_count,
            frame_count,
            site_rgba_f32.shape[0],
            anchor_owner_i32.shape[0],
            change_frame_i32.shape[0],
            op_type_i32.shape[0],
            block_size,
            block_count,
        ],
        device=boundary_f32.device,
        dtype=torch.int32,
    )
    config_f32 = torch.tensor(
        [config.near, config.far, config.invalid_epsilon, config.transmittance_threshold],
        device=boundary_f32.device,
        dtype=torch.float32,
    )
    ops = torch.ops.world_foam_lane2_fused_slab_v0
    op_name = "endpoint_record_edit_block4_vjp_direct_atomic_rgb_only"
    if not hasattr(ops, op_name):
        raise RuntimeError(f"world_foam_lane2_fused_slab_v0 {op_name} op not found. Build this variant first.")
    return getattr(ops, op_name)(
        boundary_f32,
        rays_f32,
        frame_t_f32,
        anchor_offsets_i32,
        anchor_owner_i32,
        anchor_left_i32,
        anchor_right_i32,
        track_block_change_offsets_i32,
        change_frame_i32,
        op_offsets_i32,
        op_type_i32,
        op_pos_i32,
        op_owner_i32,
        op_left_i32,
        op_right_i32,
        site_rgba_f32,
        grad_rgb_f32,
        config_i32,
        config_f32,
    )


def endpoint_record_edit_block_coeff_vjp_direct_atomic_rgb_only(
    coeff_f32: Tensor,
    frame_t_f32: Tensor,
    anchor_offsets_i32: Tensor,
    anchor_owner_i32: Tensor,
    anchor_left_i32: Tensor,
    anchor_right_i32: Tensor,
    track_block_change_offsets_i32: Tensor,
    change_frame_i32: Tensor,
    op_offsets_i32: Tensor,
    op_type_i32: Tensor,
    op_pos_i32: Tensor,
    op_owner_i32: Tensor,
    op_left_i32: Tensor,
    op_right_i32: Tensor,
    site_rgba_f32: Tensor,
    grad_rgb_f32: Tensor,
    config: RealRayReplayConfig,
    *,
    track_count: int,
    frame_count: int,
    boundary_count: int,
    block_size: int = 4,
) -> Tensor:
    """Accumulate RGB-only site gradients from coefficient-cached block edit rows."""
    if block_size <= 0:
        raise ValueError("endpoint_record_edit_block_coeff_vjp_direct_atomic_rgb_only requires block_size > 0")
    coeff_f32 = coeff_f32.contiguous()
    frame_t_f32 = frame_t_f32.contiguous()
    anchor_offsets_i32 = anchor_offsets_i32.contiguous()
    anchor_owner_i32 = anchor_owner_i32.contiguous()
    anchor_left_i32 = anchor_left_i32.contiguous()
    anchor_right_i32 = anchor_right_i32.contiguous()
    track_block_change_offsets_i32 = track_block_change_offsets_i32.contiguous()
    change_frame_i32 = change_frame_i32.contiguous()
    op_offsets_i32 = op_offsets_i32.contiguous()
    op_type_i32 = op_type_i32.contiguous()
    op_pos_i32 = op_pos_i32.contiguous()
    op_owner_i32 = op_owner_i32.contiguous()
    op_left_i32 = op_left_i32.contiguous()
    op_right_i32 = op_right_i32.contiguous()
    site_rgba_f32 = site_rgba_f32.contiguous()
    grad_rgb_f32 = grad_rgb_f32.contiguous()
    if coeff_f32.ndim != 2 or coeff_f32.shape[1] != 4:
        raise ValueError("coeff_f32 must have shape [track_count * boundary_count, 4]")
    if coeff_f32.shape[0] != track_count * boundary_count:
        raise ValueError("coeff_f32 row count must match track_count * boundary_count")
    if frame_t_f32.shape[0] != frame_count:
        raise ValueError("frame_t_f32 length must match frame_count")
    if grad_rgb_f32.device.type != "mps" or grad_rgb_f32.dtype != torch.float32:
        raise ValueError("grad_rgb_f32 must be float32 on MPS")
    if grad_rgb_f32.shape != (track_count, frame_count, 3):
        raise ValueError("grad_rgb_f32 must have shape [track_count, frame_count, 3]")
    block_count = (frame_count + block_size - 1) // block_size
    config_i32 = torch.tensor(
        [
            boundary_count,
            track_count,
            frame_count,
            site_rgba_f32.shape[0],
            anchor_owner_i32.shape[0],
            change_frame_i32.shape[0],
            op_type_i32.shape[0],
            block_size,
            block_count,
        ],
        device=coeff_f32.device,
        dtype=torch.int32,
    )
    config_f32 = torch.tensor(
        [config.near, config.far, config.invalid_epsilon, config.transmittance_threshold],
        device=coeff_f32.device,
        dtype=torch.float32,
    )
    ops = torch.ops.world_foam_lane2_fused_slab_v0
    op_name = "endpoint_record_edit_block_coeff_vjp_direct_atomic_rgb_only"
    if not hasattr(ops, op_name):
        raise RuntimeError(f"world_foam_lane2_fused_slab_v0 {op_name} op not found. Build this variant first.")
    return getattr(ops, op_name)(
        coeff_f32,
        frame_t_f32,
        anchor_offsets_i32,
        anchor_owner_i32,
        anchor_left_i32,
        anchor_right_i32,
        track_block_change_offsets_i32,
        change_frame_i32,
        op_offsets_i32,
        op_type_i32,
        op_pos_i32,
        op_owner_i32,
        op_left_i32,
        op_right_i32,
        site_rgba_f32,
        grad_rgb_f32,
        config_i32,
        config_f32,
    )


def endpoint_record_edit_block_coeff_mse_vjp_direct_atomic_rgb_only(
    coeff_f32: Tensor,
    frame_t_f32: Tensor,
    anchor_offsets_i32: Tensor,
    anchor_owner_i32: Tensor,
    anchor_left_i32: Tensor,
    anchor_right_i32: Tensor,
    track_block_change_offsets_i32: Tensor,
    change_frame_i32: Tensor,
    op_offsets_i32: Tensor,
    op_type_i32: Tensor,
    op_pos_i32: Tensor,
    op_owner_i32: Tensor,
    op_left_i32: Tensor,
    op_right_i32: Tensor,
    site_rgba_f32: Tensor,
    target_rgb_f32: Tensor,
    config: RealRayReplayConfig,
    *,
    track_count: int,
    frame_count: int,
    boundary_count: int,
    block_size: int = 4,
) -> tuple[Tensor, Tensor]:
    """Compute RGB MSE and site gradients for coefficient-cached block edit rows."""
    if block_size <= 0:
        raise ValueError("endpoint_record_edit_block_coeff_mse_vjp_direct_atomic_rgb_only requires block_size > 0")
    coeff_f32 = coeff_f32.contiguous()
    frame_t_f32 = frame_t_f32.contiguous()
    anchor_offsets_i32 = anchor_offsets_i32.contiguous()
    anchor_owner_i32 = anchor_owner_i32.contiguous()
    anchor_left_i32 = anchor_left_i32.contiguous()
    anchor_right_i32 = anchor_right_i32.contiguous()
    track_block_change_offsets_i32 = track_block_change_offsets_i32.contiguous()
    change_frame_i32 = change_frame_i32.contiguous()
    op_offsets_i32 = op_offsets_i32.contiguous()
    op_type_i32 = op_type_i32.contiguous()
    op_pos_i32 = op_pos_i32.contiguous()
    op_owner_i32 = op_owner_i32.contiguous()
    op_left_i32 = op_left_i32.contiguous()
    op_right_i32 = op_right_i32.contiguous()
    site_rgba_f32 = site_rgba_f32.contiguous()
    target_rgb_f32 = target_rgb_f32.contiguous()
    if coeff_f32.ndim != 2 or coeff_f32.shape[1] != 4:
        raise ValueError("coeff_f32 must have shape [track_count * boundary_count, 4]")
    if coeff_f32.shape[0] != track_count * boundary_count:
        raise ValueError("coeff_f32 row count must match track_count * boundary_count")
    if frame_t_f32.shape[0] != frame_count:
        raise ValueError("frame_t_f32 length must match frame_count")
    if target_rgb_f32.device.type != "mps" or target_rgb_f32.dtype != torch.float32:
        raise ValueError("target_rgb_f32 must be float32 on MPS")
    if target_rgb_f32.shape != (track_count, frame_count, 3):
        raise ValueError("target_rgb_f32 must have shape [track_count, frame_count, 3]")
    block_count = (frame_count + block_size - 1) // block_size
    config_i32 = torch.tensor(
        [
            boundary_count,
            track_count,
            frame_count,
            site_rgba_f32.shape[0],
            anchor_owner_i32.shape[0],
            change_frame_i32.shape[0],
            op_type_i32.shape[0],
            block_size,
            block_count,
        ],
        device=coeff_f32.device,
        dtype=torch.int32,
    )
    config_f32 = torch.tensor(
        [config.near, config.far, config.invalid_epsilon, config.transmittance_threshold],
        device=coeff_f32.device,
        dtype=torch.float32,
    )
    ops = torch.ops.world_foam_lane2_fused_slab_v0
    op_name = "endpoint_record_edit_block_coeff_mse_vjp_direct_atomic_rgb_only"
    if not hasattr(ops, op_name):
        raise RuntimeError(f"world_foam_lane2_fused_slab_v0 {op_name} op not found. Build this variant first.")
    return getattr(ops, op_name)(
        coeff_f32,
        frame_t_f32,
        anchor_offsets_i32,
        anchor_owner_i32,
        anchor_left_i32,
        anchor_right_i32,
        track_block_change_offsets_i32,
        change_frame_i32,
        op_offsets_i32,
        op_type_i32,
        op_pos_i32,
        op_owner_i32,
        op_left_i32,
        op_right_i32,
        site_rgba_f32,
        target_rgb_f32,
        config_i32,
        config_f32,
    )


def endpoint_record_edit_block_coeff16_mse_vjp_direct_atomic_rgb_only(
    coeff_f16: Tensor,
    frame_t_f32: Tensor,
    anchor_offsets_i32: Tensor,
    anchor_owner_i32: Tensor,
    anchor_left_i32: Tensor,
    anchor_right_i32: Tensor,
    track_block_change_offsets_i32: Tensor,
    change_frame_i32: Tensor,
    op_offsets_i32: Tensor,
    op_type_i32: Tensor,
    op_pos_i32: Tensor,
    op_owner_i32: Tensor,
    op_left_i32: Tensor,
    op_right_i32: Tensor,
    site_rgba_f32: Tensor,
    target_rgb_f32: Tensor,
    config: RealRayReplayConfig,
    *,
    track_count: int,
    frame_count: int,
    boundary_count: int,
    block_size: int = 4,
) -> tuple[Tensor, Tensor]:
    """Compute RGB MSE and site gradients for float16 coefficient-cached block edit rows."""
    if block_size <= 0:
        raise ValueError("endpoint_record_edit_block_coeff16_mse_vjp_direct_atomic_rgb_only requires block_size > 0")
    coeff_f16 = coeff_f16.contiguous()
    frame_t_f32 = frame_t_f32.contiguous()
    anchor_offsets_i32 = anchor_offsets_i32.contiguous()
    anchor_owner_i32 = anchor_owner_i32.contiguous()
    anchor_left_i32 = anchor_left_i32.contiguous()
    anchor_right_i32 = anchor_right_i32.contiguous()
    track_block_change_offsets_i32 = track_block_change_offsets_i32.contiguous()
    change_frame_i32 = change_frame_i32.contiguous()
    op_offsets_i32 = op_offsets_i32.contiguous()
    op_type_i32 = op_type_i32.contiguous()
    op_pos_i32 = op_pos_i32.contiguous()
    op_owner_i32 = op_owner_i32.contiguous()
    op_left_i32 = op_left_i32.contiguous()
    op_right_i32 = op_right_i32.contiguous()
    site_rgba_f32 = site_rgba_f32.contiguous()
    target_rgb_f32 = target_rgb_f32.contiguous()
    if coeff_f16.dtype != torch.float16:
        raise ValueError("coeff_f16 must be float16")
    if coeff_f16.ndim != 2 or coeff_f16.shape[1] != 4:
        raise ValueError("coeff_f16 must have shape [track_count * boundary_count, 4]")
    if coeff_f16.shape[0] != track_count * boundary_count:
        raise ValueError("coeff_f16 row count must match track_count * boundary_count")
    if frame_t_f32.shape[0] != frame_count:
        raise ValueError("frame_t_f32 length must match frame_count")
    if target_rgb_f32.device.type != "mps" or target_rgb_f32.dtype != torch.float32:
        raise ValueError("target_rgb_f32 must be float32 on MPS")
    if target_rgb_f32.shape != (track_count, frame_count, 3):
        raise ValueError("target_rgb_f32 must have shape [track_count, frame_count, 3]")
    block_count = (frame_count + block_size - 1) // block_size
    config_i32 = torch.tensor(
        [
            boundary_count,
            track_count,
            frame_count,
            site_rgba_f32.shape[0],
            anchor_owner_i32.shape[0],
            change_frame_i32.shape[0],
            op_type_i32.shape[0],
            block_size,
            block_count,
        ],
        device=coeff_f16.device,
        dtype=torch.int32,
    )
    config_f32 = torch.tensor(
        [config.near, config.far, config.invalid_epsilon, config.transmittance_threshold],
        device=coeff_f16.device,
        dtype=torch.float32,
    )
    ops = torch.ops.world_foam_lane2_fused_slab_v0
    op_name = "endpoint_record_edit_block_coeff16_mse_vjp_direct_atomic_rgb_only"
    if not hasattr(ops, op_name):
        raise RuntimeError(f"world_foam_lane2_fused_slab_v0 {op_name} op not found. Build this variant first.")
    return getattr(ops, op_name)(
        coeff_f16,
        frame_t_f32,
        anchor_offsets_i32,
        anchor_owner_i32,
        anchor_left_i32,
        anchor_right_i32,
        track_block_change_offsets_i32,
        change_frame_i32,
        op_offsets_i32,
        op_type_i32,
        op_pos_i32,
        op_owner_i32,
        op_left_i32,
        op_right_i32,
        site_rgba_f32,
        target_rgb_f32,
        config_i32,
        config_f32,
    )


def endpoint_record_edit_block_coeff16_packed_mse_vjp_direct_atomic_rgb_only(
    coeff_f16: Tensor,
    frame_t_f32: Tensor,
    anchor_offsets_i32: Tensor,
    anchor_record_i32: Tensor,
    track_block_change_offsets_i32: Tensor,
    change_frame_i32: Tensor,
    op_offsets_i32: Tensor,
    op_type_i32: Tensor,
    op_pos_i32: Tensor,
    op_record_i32: Tensor,
    site_rgba_f32: Tensor,
    target_rgb_f32: Tensor,
    config: RealRayReplayConfig,
    *,
    track_count: int,
    frame_count: int,
    boundary_count: int,
    block_size: int = 4,
) -> tuple[Tensor, Tensor]:
    """Compute RGB MSE/VJP from packed float16 coefficient-cached block edit rows."""
    if block_size <= 0:
        raise ValueError(
            "endpoint_record_edit_block_coeff16_packed_mse_vjp_direct_atomic_rgb_only requires block_size > 0"
        )
    coeff_f16 = coeff_f16.contiguous()
    frame_t_f32 = frame_t_f32.contiguous()
    anchor_offsets_i32 = anchor_offsets_i32.contiguous()
    anchor_record_i32 = anchor_record_i32.contiguous()
    track_block_change_offsets_i32 = track_block_change_offsets_i32.contiguous()
    change_frame_i32 = change_frame_i32.contiguous()
    op_offsets_i32 = op_offsets_i32.contiguous()
    op_type_i32 = op_type_i32.contiguous()
    op_pos_i32 = op_pos_i32.contiguous()
    op_record_i32 = op_record_i32.contiguous()
    site_rgba_f32 = site_rgba_f32.contiguous()
    target_rgb_f32 = target_rgb_f32.contiguous()
    if coeff_f16.dtype != torch.float16:
        raise ValueError("coeff_f16 must be float16")
    if coeff_f16.ndim != 2 or coeff_f16.shape[1] != 4:
        raise ValueError("coeff_f16 must have shape [track_count * boundary_count, 4]")
    if coeff_f16.shape[0] != track_count * boundary_count:
        raise ValueError("coeff_f16 row count must match track_count * boundary_count")
    if frame_t_f32.shape[0] != frame_count:
        raise ValueError("frame_t_f32 length must match frame_count")
    if anchor_record_i32.ndim != 1 or anchor_record_i32.dtype != torch.int32:
        raise ValueError("anchor_record_i32 must be int32 with shape [anchor_record_count]")
    if op_record_i32.ndim != 1 or op_record_i32.dtype != torch.int32:
        raise ValueError("op_record_i32 must be int32 with shape [op_count]")
    if target_rgb_f32.device.type != "mps" or target_rgb_f32.dtype != torch.float32:
        raise ValueError("target_rgb_f32 must be float32 on MPS")
    if target_rgb_f32.shape != (track_count, frame_count, 3):
        raise ValueError("target_rgb_f32 must have shape [track_count, frame_count, 3]")
    block_count = (frame_count + block_size - 1) // block_size
    config_i32 = torch.tensor(
        [
            boundary_count,
            track_count,
            frame_count,
            site_rgba_f32.shape[0],
            anchor_record_i32.shape[0],
            change_frame_i32.shape[0],
            op_type_i32.shape[0],
            block_size,
            block_count,
        ],
        device=coeff_f16.device,
        dtype=torch.int32,
    )
    config_f32 = torch.tensor(
        [config.near, config.far, config.invalid_epsilon, config.transmittance_threshold],
        device=coeff_f16.device,
        dtype=torch.float32,
    )
    ops = torch.ops.world_foam_lane2_fused_slab_v0
    op_name = "endpoint_record_edit_block_coeff16_packed_mse_vjp_direct_atomic_rgb_only"
    if not hasattr(ops, op_name):
        raise RuntimeError(f"world_foam_lane2_fused_slab_v0 {op_name} op not found. Build this variant first.")
    return getattr(ops, op_name)(
        coeff_f16,
        frame_t_f32,
        anchor_offsets_i32,
        anchor_record_i32,
        track_block_change_offsets_i32,
        change_frame_i32,
        op_offsets_i32,
        op_type_i32,
        op_pos_i32,
        op_record_i32,
        site_rgba_f32,
        target_rgb_f32,
        config_i32,
        config_f32,
    )


def endpoint_record_edit_block_coeff16_i16_mse_vjp_direct_atomic_rgb_only(
    coeff_f16: Tensor,
    frame_t_f32: Tensor,
    anchor_offsets_i32: Tensor,
    anchor_owner_i16: Tensor,
    anchor_left_i16: Tensor,
    anchor_right_i16: Tensor,
    track_block_change_offsets_i32: Tensor,
    change_frame_i32: Tensor,
    op_offsets_i32: Tensor,
    op_type_i32: Tensor,
    op_pos_i32: Tensor,
    op_owner_i16: Tensor,
    op_left_i16: Tensor,
    op_right_i16: Tensor,
    site_rgba_f32: Tensor,
    target_rgb_f32: Tensor,
    config: RealRayReplayConfig,
    *,
    track_count: int,
    frame_count: int,
    boundary_count: int,
    block_size: int = 4,
) -> tuple[Tensor, Tensor]:
    """Compute RGB MSE/VJP from int16-record float16 coefficient-cached block edit rows."""
    if block_size <= 0:
        raise ValueError(
            "endpoint_record_edit_block_coeff16_i16_mse_vjp_direct_atomic_rgb_only requires block_size > 0"
        )
    coeff_f16 = coeff_f16.contiguous()
    frame_t_f32 = frame_t_f32.contiguous()
    anchor_offsets_i32 = anchor_offsets_i32.contiguous()
    anchor_owner_i16 = anchor_owner_i16.contiguous()
    anchor_left_i16 = anchor_left_i16.contiguous()
    anchor_right_i16 = anchor_right_i16.contiguous()
    track_block_change_offsets_i32 = track_block_change_offsets_i32.contiguous()
    change_frame_i32 = change_frame_i32.contiguous()
    op_offsets_i32 = op_offsets_i32.contiguous()
    op_type_i32 = op_type_i32.contiguous()
    op_pos_i32 = op_pos_i32.contiguous()
    op_owner_i16 = op_owner_i16.contiguous()
    op_left_i16 = op_left_i16.contiguous()
    op_right_i16 = op_right_i16.contiguous()
    site_rgba_f32 = site_rgba_f32.contiguous()
    target_rgb_f32 = target_rgb_f32.contiguous()
    if coeff_f16.dtype != torch.float16:
        raise ValueError("coeff_f16 must be float16")
    if coeff_f16.ndim != 2 or coeff_f16.shape[1] != 4:
        raise ValueError("coeff_f16 must have shape [track_count * boundary_count, 4]")
    if coeff_f16.shape[0] != track_count * boundary_count:
        raise ValueError("coeff_f16 row count must match track_count * boundary_count")
    if frame_t_f32.shape[0] != frame_count:
        raise ValueError("frame_t_f32 length must match frame_count")
    for name, tensor in (
        ("anchor_owner_i16", anchor_owner_i16),
        ("anchor_left_i16", anchor_left_i16),
        ("anchor_right_i16", anchor_right_i16),
        ("op_owner_i16", op_owner_i16),
        ("op_left_i16", op_left_i16),
        ("op_right_i16", op_right_i16),
    ):
        if tensor.ndim != 1 or tensor.dtype != torch.int16:
            raise ValueError(f"{name} must be int16 with shape [N]")
    if target_rgb_f32.device.type != "mps" or target_rgb_f32.dtype != torch.float32:
        raise ValueError("target_rgb_f32 must be float32 on MPS")
    if target_rgb_f32.shape != (track_count, frame_count, 3):
        raise ValueError("target_rgb_f32 must have shape [track_count, frame_count, 3]")
    block_count = (frame_count + block_size - 1) // block_size
    config_i32 = torch.tensor(
        [
            boundary_count,
            track_count,
            frame_count,
            site_rgba_f32.shape[0],
            anchor_owner_i16.shape[0],
            change_frame_i32.shape[0],
            op_type_i32.shape[0],
            block_size,
            block_count,
        ],
        device=coeff_f16.device,
        dtype=torch.int32,
    )
    config_f32 = torch.tensor(
        [config.near, config.far, config.invalid_epsilon, config.transmittance_threshold],
        device=coeff_f16.device,
        dtype=torch.float32,
    )
    ops = torch.ops.world_foam_lane2_fused_slab_v0
    op_name = "endpoint_record_edit_block_coeff16_i16_mse_vjp_direct_atomic_rgb_only"
    if not hasattr(ops, op_name):
        raise RuntimeError(f"world_foam_lane2_fused_slab_v0 {op_name} op not found. Build this variant first.")
    return getattr(ops, op_name)(
        coeff_f16,
        frame_t_f32,
        anchor_offsets_i32,
        anchor_owner_i16,
        anchor_left_i16,
        anchor_right_i16,
        track_block_change_offsets_i32,
        change_frame_i32,
        op_offsets_i32,
        op_type_i32,
        op_pos_i32,
        op_owner_i16,
        op_left_i16,
        op_right_i16,
        site_rgba_f32,
        target_rgb_f32,
        config_i32,
        config_f32,
    )


def endpoint_record_edit_block_coeff16_i16x3_mse_vjp_direct_atomic_rgb_only(
    coeff_f16: Tensor,
    frame_t_f32: Tensor,
    anchor_offsets_i32: Tensor,
    anchor_record_i16: Tensor,
    track_block_change_offsets_i32: Tensor,
    change_frame_i32: Tensor,
    op_offsets_i32: Tensor,
    op_type_i32: Tensor,
    op_pos_i32: Tensor,
    op_record_i16: Tensor,
    site_rgba_f32: Tensor,
    target_rgb_f32: Tensor,
    config: RealRayReplayConfig,
    *,
    track_count: int,
    frame_count: int,
    boundary_count: int,
    block_size: int = 4,
) -> tuple[Tensor, Tensor]:
    """Compute RGB MSE/VJP from interleaved int16 coefficient-cached block edit rows."""
    if block_size <= 0:
        raise ValueError(
            "endpoint_record_edit_block_coeff16_i16x3_mse_vjp_direct_atomic_rgb_only requires block_size > 0"
        )
    coeff_f16 = coeff_f16.contiguous()
    frame_t_f32 = frame_t_f32.contiguous()
    anchor_offsets_i32 = anchor_offsets_i32.contiguous()
    anchor_record_i16 = anchor_record_i16.contiguous()
    track_block_change_offsets_i32 = track_block_change_offsets_i32.contiguous()
    change_frame_i32 = change_frame_i32.contiguous()
    op_offsets_i32 = op_offsets_i32.contiguous()
    op_type_i32 = op_type_i32.contiguous()
    op_pos_i32 = op_pos_i32.contiguous()
    op_record_i16 = op_record_i16.contiguous()
    site_rgba_f32 = site_rgba_f32.contiguous()
    target_rgb_f32 = target_rgb_f32.contiguous()
    if coeff_f16.dtype != torch.float16:
        raise ValueError("coeff_f16 must be float16")
    if coeff_f16.ndim != 2 or coeff_f16.shape[1] != 4:
        raise ValueError("coeff_f16 must have shape [track_count * boundary_count, 4]")
    if coeff_f16.shape[0] != track_count * boundary_count:
        raise ValueError("coeff_f16 row count must match track_count * boundary_count")
    if frame_t_f32.shape[0] != frame_count:
        raise ValueError("frame_t_f32 length must match frame_count")
    if anchor_record_i16.ndim != 1 or anchor_record_i16.dtype != torch.int16:
        raise ValueError("anchor_record_i16 must be int16 with shape [anchor_record_count * 3]")
    if op_record_i16.ndim != 1 or op_record_i16.dtype != torch.int16:
        raise ValueError("op_record_i16 must be int16 with shape [op_count * 3]")
    if anchor_record_i16.numel() % 3 != 0:
        raise ValueError("anchor_record_i16 length must be divisible by 3")
    if op_record_i16.numel() % 3 != 0:
        raise ValueError("op_record_i16 length must be divisible by 3")
    if target_rgb_f32.device.type != "mps" or target_rgb_f32.dtype != torch.float32:
        raise ValueError("target_rgb_f32 must be float32 on MPS")
    if target_rgb_f32.shape != (track_count, frame_count, 3):
        raise ValueError("target_rgb_f32 must have shape [track_count, frame_count, 3]")
    block_count = (frame_count + block_size - 1) // block_size
    config_i32 = torch.tensor(
        [
            boundary_count,
            track_count,
            frame_count,
            site_rgba_f32.shape[0],
            anchor_record_i16.numel() // 3,
            change_frame_i32.shape[0],
            op_type_i32.shape[0],
            block_size,
            block_count,
        ],
        device=coeff_f16.device,
        dtype=torch.int32,
    )
    config_f32 = torch.tensor(
        [config.near, config.far, config.invalid_epsilon, config.transmittance_threshold],
        device=coeff_f16.device,
        dtype=torch.float32,
    )
    ops = torch.ops.world_foam_lane2_fused_slab_v0
    op_name = "endpoint_record_edit_block_coeff16_i16x3_mse_vjp_direct_atomic_rgb_only"
    if not hasattr(ops, op_name):
        raise RuntimeError(f"world_foam_lane2_fused_slab_v0 {op_name} op not found. Build this variant first.")
    return getattr(ops, op_name)(
        coeff_f16,
        frame_t_f32,
        anchor_offsets_i32,
        anchor_record_i16,
        track_block_change_offsets_i32,
        change_frame_i32,
        op_offsets_i32,
        op_type_i32,
        op_pos_i32,
        op_record_i16,
        site_rgba_f32,
        target_rgb_f32,
        config_i32,
        config_f32,
    )


def endpoint_record_edit_block_coeff16_vjp_direct_atomic_rgb_only(
    coeff_f16: Tensor,
    frame_t_f32: Tensor,
    anchor_offsets_i32: Tensor,
    anchor_owner_i32: Tensor,
    anchor_left_i32: Tensor,
    anchor_right_i32: Tensor,
    track_block_change_offsets_i32: Tensor,
    change_frame_i32: Tensor,
    op_offsets_i32: Tensor,
    op_type_i32: Tensor,
    op_pos_i32: Tensor,
    op_owner_i32: Tensor,
    op_left_i32: Tensor,
    op_right_i32: Tensor,
    site_rgba_f32: Tensor,
    grad_rgb_f32: Tensor,
    config: RealRayReplayConfig,
    *,
    track_count: int,
    frame_count: int,
    boundary_count: int,
    block_size: int = 4,
) -> Tensor:
    """Accumulate RGB-only site gradients from float16 coefficient-cached block edit rows."""
    if block_size <= 0:
        raise ValueError("endpoint_record_edit_block_coeff16_vjp_direct_atomic_rgb_only requires block_size > 0")
    coeff_f16 = coeff_f16.contiguous()
    frame_t_f32 = frame_t_f32.contiguous()
    anchor_offsets_i32 = anchor_offsets_i32.contiguous()
    anchor_owner_i32 = anchor_owner_i32.contiguous()
    anchor_left_i32 = anchor_left_i32.contiguous()
    anchor_right_i32 = anchor_right_i32.contiguous()
    track_block_change_offsets_i32 = track_block_change_offsets_i32.contiguous()
    change_frame_i32 = change_frame_i32.contiguous()
    op_offsets_i32 = op_offsets_i32.contiguous()
    op_type_i32 = op_type_i32.contiguous()
    op_pos_i32 = op_pos_i32.contiguous()
    op_owner_i32 = op_owner_i32.contiguous()
    op_left_i32 = op_left_i32.contiguous()
    op_right_i32 = op_right_i32.contiguous()
    site_rgba_f32 = site_rgba_f32.contiguous()
    grad_rgb_f32 = grad_rgb_f32.contiguous()
    if coeff_f16.dtype != torch.float16:
        raise ValueError("coeff_f16 must be float16")
    if coeff_f16.ndim != 2 or coeff_f16.shape[1] != 4:
        raise ValueError("coeff_f16 must have shape [track_count * boundary_count, 4]")
    if coeff_f16.shape[0] != track_count * boundary_count:
        raise ValueError("coeff_f16 row count must match track_count * boundary_count")
    if frame_t_f32.shape[0] != frame_count:
        raise ValueError("frame_t_f32 length must match frame_count")
    if grad_rgb_f32.device.type != "mps" or grad_rgb_f32.dtype != torch.float32:
        raise ValueError("grad_rgb_f32 must be float32 on MPS")
    if grad_rgb_f32.shape != (track_count, frame_count, 3):
        raise ValueError("grad_rgb_f32 must have shape [track_count, frame_count, 3]")
    block_count = (frame_count + block_size - 1) // block_size
    config_i32 = torch.tensor(
        [
            boundary_count,
            track_count,
            frame_count,
            site_rgba_f32.shape[0],
            anchor_owner_i32.shape[0],
            change_frame_i32.shape[0],
            op_type_i32.shape[0],
            block_size,
            block_count,
        ],
        device=coeff_f16.device,
        dtype=torch.int32,
    )
    config_f32 = torch.tensor(
        [config.near, config.far, config.invalid_epsilon, config.transmittance_threshold],
        device=coeff_f16.device,
        dtype=torch.float32,
    )
    ops = torch.ops.world_foam_lane2_fused_slab_v0
    op_name = "endpoint_record_edit_block_coeff16_vjp_direct_atomic_rgb_only"
    if not hasattr(ops, op_name):
        raise RuntimeError(f"world_foam_lane2_fused_slab_v0 {op_name} op not found. Build this variant first.")
    return getattr(ops, op_name)(
        coeff_f16,
        frame_t_f32,
        anchor_offsets_i32,
        anchor_owner_i32,
        anchor_left_i32,
        anchor_right_i32,
        track_block_change_offsets_i32,
        change_frame_i32,
        op_offsets_i32,
        op_type_i32,
        op_pos_i32,
        op_owner_i32,
        op_left_i32,
        op_right_i32,
        site_rgba_f32,
        grad_rgb_f32,
        config_i32,
        config_f32,
    )


class _EndpointRecordEditRGBADepthReplayFunction(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx: torch.autograd.function.FunctionCtx,
        boundary_f32: Tensor,
        rays_f32: Tensor,
        frame_t_f32: Tensor,
        base_offsets_i32: Tensor,
        base_owner_i32: Tensor,
        base_left_i32: Tensor,
        base_right_i32: Tensor,
        track_change_offsets_i32: Tensor,
        change_frame_i32: Tensor,
        op_offsets_i32: Tensor,
        op_type_i32: Tensor,
        op_pos_i32: Tensor,
        op_owner_i32: Tensor,
        op_left_i32: Tensor,
        op_right_i32: Tensor,
        site_rgba_f32: Tensor,
        track_count: int,
        frame_count: int,
        near: float,
        far: float,
        invalid_epsilon: float,
        transmittance_threshold: float,
    ) -> tuple[Tensor, Tensor, Tensor]:
        config = RealRayReplayConfig(
            near=float(near),
            far=float(far),
            invalid_epsilon=float(invalid_epsilon),
            transmittance_threshold=float(transmittance_threshold),
        )
        output_rgb, output_alpha, output_depth = endpoint_record_edit_rgba_depth_replay(
            boundary_f32,
            rays_f32,
            frame_t_f32,
            base_offsets_i32,
            base_owner_i32,
            base_left_i32,
            base_right_i32,
            track_change_offsets_i32,
            change_frame_i32,
            op_offsets_i32,
            op_type_i32,
            op_pos_i32,
            op_owner_i32,
            op_left_i32,
            op_right_i32,
            site_rgba_f32,
            config,
            track_count=int(track_count),
            frame_count=int(frame_count),
        )
        ctx.save_for_backward(
            boundary_f32,
            rays_f32,
            frame_t_f32,
            base_offsets_i32,
            base_owner_i32,
            base_left_i32,
            base_right_i32,
            track_change_offsets_i32,
            change_frame_i32,
            op_offsets_i32,
            op_type_i32,
            op_pos_i32,
            op_owner_i32,
            op_left_i32,
            op_right_i32,
            site_rgba_f32,
        )
        ctx.config = config
        ctx.track_count = int(track_count)
        ctx.frame_count = int(frame_count)
        return output_rgb, output_alpha, output_depth

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: torch.autograd.function.FunctionCtx,
        grad_rgb: Tensor | None,
        grad_alpha: Tensor | None,
        grad_depth: Tensor | None,
    ) -> tuple[
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        Tensor,
        None,
        None,
        None,
        None,
        None,
        None,
    ]:
        (
            boundary_f32,
            rays_f32,
            frame_t_f32,
            base_offsets_i32,
            base_owner_i32,
            base_left_i32,
            base_right_i32,
            track_change_offsets_i32,
            change_frame_i32,
            op_offsets_i32,
            op_type_i32,
            op_pos_i32,
            op_owner_i32,
            op_left_i32,
            op_right_i32,
            site_rgba_f32,
        ) = ctx.saved_tensors
        track_count = int(ctx.track_count)
        frame_count = int(ctx.frame_count)
        grad_alpha_is_none = grad_alpha is None
        grad_depth_is_none = grad_depth is None
        if grad_rgb is None:
            grad_rgb = torch.zeros((track_count, frame_count, 3), device=site_rgba_f32.device, dtype=torch.float32)
        if grad_alpha is None:
            grad_alpha = torch.zeros((track_count, frame_count), device=site_rgba_f32.device, dtype=torch.float32)
        if grad_depth is None:
            grad_depth = torch.zeros((track_count, frame_count), device=site_rgba_f32.device, dtype=torch.float32)
        if grad_alpha_is_none and grad_depth_is_none:
            grad_site_rgba = endpoint_record_edit_vjp_direct_atomic_rgb_only(
                boundary_f32,
                rays_f32,
                frame_t_f32,
                base_offsets_i32,
                base_owner_i32,
                base_left_i32,
                base_right_i32,
                track_change_offsets_i32,
                change_frame_i32,
                op_offsets_i32,
                op_type_i32,
                op_pos_i32,
                op_owner_i32,
                op_left_i32,
                op_right_i32,
                site_rgba_f32,
                grad_rgb,
                ctx.config,
                track_count=track_count,
                frame_count=frame_count,
            )
        else:
            grad_site_rgba = endpoint_record_edit_vjp_direct_atomic_grad_only(
                boundary_f32,
                rays_f32,
                frame_t_f32,
                base_offsets_i32,
                base_owner_i32,
                base_left_i32,
                base_right_i32,
                track_change_offsets_i32,
                change_frame_i32,
                op_offsets_i32,
                op_type_i32,
                op_pos_i32,
                op_owner_i32,
                op_left_i32,
                op_right_i32,
                site_rgba_f32,
                grad_rgb,
                grad_alpha,
                grad_depth,
                ctx.config,
                track_count=track_count,
                frame_count=frame_count,
            )
        return (
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            grad_site_rgba,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def endpoint_record_edit_rgba_depth_autograd(
    boundary_f32: Tensor,
    rays_f32: Tensor,
    frame_t_f32: Tensor,
    base_offsets_i32: Tensor,
    base_owner_i32: Tensor,
    base_left_i32: Tensor,
    base_right_i32: Tensor,
    track_change_offsets_i32: Tensor,
    change_frame_i32: Tensor,
    op_offsets_i32: Tensor,
    op_type_i32: Tensor,
    op_pos_i32: Tensor,
    op_owner_i32: Tensor,
    op_left_i32: Tensor,
    op_right_i32: Tensor,
    site_rgba_f32: Tensor,
    config: RealRayReplayConfig,
    *,
    track_count: int,
    frame_count: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Replay endpoint owner+cut-id edit streams with frozen-geometry site-RGBA autograd."""
    return _EndpointRecordEditRGBADepthReplayFunction.apply(
        boundary_f32,
        rays_f32,
        frame_t_f32,
        base_offsets_i32,
        base_owner_i32,
        base_left_i32,
        base_right_i32,
        track_change_offsets_i32,
        change_frame_i32,
        op_offsets_i32,
        op_type_i32,
        op_pos_i32,
        op_owner_i32,
        op_left_i32,
        op_right_i32,
        site_rgba_f32,
        int(track_count),
        int(frame_count),
        config.near,
        config.far,
        config.invalid_epsilon,
        config.transmittance_threshold,
    )


class _EndpointRecordEditBlock4RGBADepthReplayFunction(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx: torch.autograd.function.FunctionCtx,
        boundary_f32: Tensor,
        rays_f32: Tensor,
        frame_t_f32: Tensor,
        anchor_offsets_i32: Tensor,
        anchor_owner_i32: Tensor,
        anchor_left_i32: Tensor,
        anchor_right_i32: Tensor,
        track_block_change_offsets_i32: Tensor,
        block_change_frame_i32: Tensor,
        block_op_offsets_i32: Tensor,
        block_op_type_i32: Tensor,
        block_op_pos_i32: Tensor,
        block_op_owner_i32: Tensor,
        block_op_left_i32: Tensor,
        block_op_right_i32: Tensor,
        base_offsets_i32: Tensor,
        base_owner_i32: Tensor,
        base_left_i32: Tensor,
        base_right_i32: Tensor,
        track_change_offsets_i32: Tensor,
        change_frame_i32: Tensor,
        op_offsets_i32: Tensor,
        op_type_i32: Tensor,
        op_pos_i32: Tensor,
        op_owner_i32: Tensor,
        op_left_i32: Tensor,
        op_right_i32: Tensor,
        site_rgba_f32: Tensor,
        track_count: int,
        frame_count: int,
        block_size: int,
        near: float,
        far: float,
        invalid_epsilon: float,
        transmittance_threshold: float,
    ) -> tuple[Tensor, Tensor, Tensor]:
        config = RealRayReplayConfig(
            near=float(near),
            far=float(far),
            invalid_epsilon=float(invalid_epsilon),
            transmittance_threshold=float(transmittance_threshold),
        )
        output_rgb, output_alpha, output_depth = endpoint_record_edit_block4_rgba_depth_replay(
            boundary_f32,
            rays_f32,
            frame_t_f32,
            anchor_offsets_i32,
            anchor_owner_i32,
            anchor_left_i32,
            anchor_right_i32,
            track_block_change_offsets_i32,
            block_change_frame_i32,
            block_op_offsets_i32,
            block_op_type_i32,
            block_op_pos_i32,
            block_op_owner_i32,
            block_op_left_i32,
            block_op_right_i32,
            site_rgba_f32,
            config,
            track_count=int(track_count),
            frame_count=int(frame_count),
            block_size=int(block_size),
        )
        ctx.save_for_backward(
            boundary_f32,
            rays_f32,
            frame_t_f32,
            anchor_offsets_i32,
            anchor_owner_i32,
            anchor_left_i32,
            anchor_right_i32,
            track_block_change_offsets_i32,
            block_change_frame_i32,
            block_op_offsets_i32,
            block_op_type_i32,
            block_op_pos_i32,
            block_op_owner_i32,
            block_op_left_i32,
            block_op_right_i32,
            base_offsets_i32,
            base_owner_i32,
            base_left_i32,
            base_right_i32,
            track_change_offsets_i32,
            change_frame_i32,
            op_offsets_i32,
            op_type_i32,
            op_pos_i32,
            op_owner_i32,
            op_left_i32,
            op_right_i32,
            site_rgba_f32,
        )
        ctx.config = config
        ctx.track_count = int(track_count)
        ctx.frame_count = int(frame_count)
        ctx.block_size = int(block_size)
        return output_rgb, output_alpha, output_depth

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: torch.autograd.function.FunctionCtx,
        grad_rgb: Tensor | None,
        grad_alpha: Tensor | None,
        grad_depth: Tensor | None,
    ) -> tuple[Tensor | None, ...]:
        (
            boundary_f32,
            rays_f32,
            frame_t_f32,
            anchor_offsets_i32,
            anchor_owner_i32,
            anchor_left_i32,
            anchor_right_i32,
            track_block_change_offsets_i32,
            block_change_frame_i32,
            block_op_offsets_i32,
            block_op_type_i32,
            block_op_pos_i32,
            block_op_owner_i32,
            block_op_left_i32,
            block_op_right_i32,
            base_offsets_i32,
            base_owner_i32,
            base_left_i32,
            base_right_i32,
            track_change_offsets_i32,
            change_frame_i32,
            op_offsets_i32,
            op_type_i32,
            op_pos_i32,
            op_owner_i32,
            op_left_i32,
            op_right_i32,
            site_rgba_f32,
        ) = ctx.saved_tensors
        track_count = int(ctx.track_count)
        frame_count = int(ctx.frame_count)
        block_size = int(ctx.block_size)
        grad_alpha_is_none = grad_alpha is None
        grad_depth_is_none = grad_depth is None
        if grad_rgb is None:
            grad_rgb = torch.zeros((track_count, frame_count, 3), device=site_rgba_f32.device, dtype=torch.float32)
        if grad_alpha is None:
            grad_alpha = torch.zeros((track_count, frame_count), device=site_rgba_f32.device, dtype=torch.float32)
        if grad_depth is None:
            grad_depth = torch.zeros((track_count, frame_count), device=site_rgba_f32.device, dtype=torch.float32)
        if grad_alpha_is_none and grad_depth_is_none:
            grad_site_rgba = endpoint_record_edit_block4_vjp_direct_atomic_rgb_only(
                boundary_f32,
                rays_f32,
                frame_t_f32,
                anchor_offsets_i32,
                anchor_owner_i32,
                anchor_left_i32,
                anchor_right_i32,
                track_block_change_offsets_i32,
                block_change_frame_i32,
                block_op_offsets_i32,
                block_op_type_i32,
                block_op_pos_i32,
                block_op_owner_i32,
                block_op_left_i32,
                block_op_right_i32,
                site_rgba_f32,
                grad_rgb,
                ctx.config,
                track_count=track_count,
                frame_count=frame_count,
                block_size=block_size,
            )
        else:
            grad_site_rgba = endpoint_record_edit_vjp_direct_atomic_grad_only(
                boundary_f32,
                rays_f32,
                frame_t_f32,
                base_offsets_i32,
                base_owner_i32,
                base_left_i32,
                base_right_i32,
                track_change_offsets_i32,
                change_frame_i32,
                op_offsets_i32,
                op_type_i32,
                op_pos_i32,
                op_owner_i32,
                op_left_i32,
                op_right_i32,
                site_rgba_f32,
                grad_rgb,
                grad_alpha,
                grad_depth,
                ctx.config,
                track_count=track_count,
                frame_count=frame_count,
            )
        return (*([None] * 27), grad_site_rgba, *([None] * 7))


def endpoint_record_edit_block4_rgba_depth_autograd(
    boundary_f32: Tensor,
    rays_f32: Tensor,
    frame_t_f32: Tensor,
    anchor_offsets_i32: Tensor,
    anchor_owner_i32: Tensor,
    anchor_left_i32: Tensor,
    anchor_right_i32: Tensor,
    track_block_change_offsets_i32: Tensor,
    block_change_frame_i32: Tensor,
    block_op_offsets_i32: Tensor,
    block_op_type_i32: Tensor,
    block_op_pos_i32: Tensor,
    block_op_owner_i32: Tensor,
    block_op_left_i32: Tensor,
    block_op_right_i32: Tensor,
    base_offsets_i32: Tensor,
    base_owner_i32: Tensor,
    base_left_i32: Tensor,
    base_right_i32: Tensor,
    track_change_offsets_i32: Tensor,
    change_frame_i32: Tensor,
    op_offsets_i32: Tensor,
    op_type_i32: Tensor,
    op_pos_i32: Tensor,
    op_owner_i32: Tensor,
    op_left_i32: Tensor,
    op_right_i32: Tensor,
    site_rgba_f32: Tensor,
    config: RealRayReplayConfig,
    *,
    track_count: int,
    frame_count: int,
    block_size: int = 4,
) -> tuple[Tensor, Tensor, Tensor]:
    """Replay block4 endpoint edits with frozen-geometry site-RGBA autograd.

    Forward and RGB-only backward use block-anchored replay shaders. Full
    alpha/depth backward still falls back to the exact endpoint-record edit VJP.
    """
    return _EndpointRecordEditBlock4RGBADepthReplayFunction.apply(
        boundary_f32,
        rays_f32,
        frame_t_f32,
        anchor_offsets_i32,
        anchor_owner_i32,
        anchor_left_i32,
        anchor_right_i32,
        track_block_change_offsets_i32,
        block_change_frame_i32,
        block_op_offsets_i32,
        block_op_type_i32,
        block_op_pos_i32,
        block_op_owner_i32,
        block_op_left_i32,
        block_op_right_i32,
        base_offsets_i32,
        base_owner_i32,
        base_left_i32,
        base_right_i32,
        track_change_offsets_i32,
        change_frame_i32,
        op_offsets_i32,
        op_type_i32,
        op_pos_i32,
        op_owner_i32,
        op_left_i32,
        op_right_i32,
        site_rgba_f32,
        int(track_count),
        int(frame_count),
        int(block_size),
        config.near,
        config.far,
        config.invalid_epsilon,
        config.transmittance_threshold,
    )


class _EndpointRecordEditBlockCoeffRGBADepthReplayFunction(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx: torch.autograd.function.FunctionCtx,
        coeff_f32: Tensor,
        boundary_f32: Tensor,
        rays_f32: Tensor,
        frame_t_f32: Tensor,
        anchor_offsets_i32: Tensor,
        anchor_owner_i32: Tensor,
        anchor_left_i32: Tensor,
        anchor_right_i32: Tensor,
        track_block_change_offsets_i32: Tensor,
        block_change_frame_i32: Tensor,
        block_op_offsets_i32: Tensor,
        block_op_type_i32: Tensor,
        block_op_pos_i32: Tensor,
        block_op_owner_i32: Tensor,
        block_op_left_i32: Tensor,
        block_op_right_i32: Tensor,
        base_offsets_i32: Tensor,
        base_owner_i32: Tensor,
        base_left_i32: Tensor,
        base_right_i32: Tensor,
        track_change_offsets_i32: Tensor,
        change_frame_i32: Tensor,
        op_offsets_i32: Tensor,
        op_type_i32: Tensor,
        op_pos_i32: Tensor,
        op_owner_i32: Tensor,
        op_left_i32: Tensor,
        op_right_i32: Tensor,
        site_rgba_f32: Tensor,
        track_count: int,
        frame_count: int,
        boundary_count: int,
        block_size: int,
        near: float,
        far: float,
        invalid_epsilon: float,
        transmittance_threshold: float,
    ) -> tuple[Tensor, Tensor, Tensor]:
        config = RealRayReplayConfig(
            near=float(near),
            far=float(far),
            invalid_epsilon=float(invalid_epsilon),
            transmittance_threshold=float(transmittance_threshold),
        )
        output_rgb, output_alpha, output_depth = endpoint_record_edit_block_coeff_rgba_depth_replay(
            coeff_f32,
            frame_t_f32,
            anchor_offsets_i32,
            anchor_owner_i32,
            anchor_left_i32,
            anchor_right_i32,
            track_block_change_offsets_i32,
            block_change_frame_i32,
            block_op_offsets_i32,
            block_op_type_i32,
            block_op_pos_i32,
            block_op_owner_i32,
            block_op_left_i32,
            block_op_right_i32,
            site_rgba_f32,
            config,
            track_count=int(track_count),
            frame_count=int(frame_count),
            boundary_count=int(boundary_count),
            block_size=int(block_size),
        )
        ctx.save_for_backward(
            coeff_f32,
            boundary_f32,
            rays_f32,
            frame_t_f32,
            anchor_offsets_i32,
            anchor_owner_i32,
            anchor_left_i32,
            anchor_right_i32,
            track_block_change_offsets_i32,
            block_change_frame_i32,
            block_op_offsets_i32,
            block_op_type_i32,
            block_op_pos_i32,
            block_op_owner_i32,
            block_op_left_i32,
            block_op_right_i32,
            base_offsets_i32,
            base_owner_i32,
            base_left_i32,
            base_right_i32,
            track_change_offsets_i32,
            change_frame_i32,
            op_offsets_i32,
            op_type_i32,
            op_pos_i32,
            op_owner_i32,
            op_left_i32,
            op_right_i32,
            site_rgba_f32,
        )
        ctx.config = config
        ctx.track_count = int(track_count)
        ctx.frame_count = int(frame_count)
        ctx.boundary_count = int(boundary_count)
        ctx.block_size = int(block_size)
        return output_rgb, output_alpha, output_depth

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: torch.autograd.function.FunctionCtx,
        grad_rgb: Tensor | None,
        grad_alpha: Tensor | None,
        grad_depth: Tensor | None,
    ) -> tuple[Tensor | None, ...]:
        (
            coeff_f32,
            boundary_f32,
            rays_f32,
            frame_t_f32,
            anchor_offsets_i32,
            anchor_owner_i32,
            anchor_left_i32,
            anchor_right_i32,
            track_block_change_offsets_i32,
            block_change_frame_i32,
            block_op_offsets_i32,
            block_op_type_i32,
            block_op_pos_i32,
            block_op_owner_i32,
            block_op_left_i32,
            block_op_right_i32,
            base_offsets_i32,
            base_owner_i32,
            base_left_i32,
            base_right_i32,
            track_change_offsets_i32,
            change_frame_i32,
            op_offsets_i32,
            op_type_i32,
            op_pos_i32,
            op_owner_i32,
            op_left_i32,
            op_right_i32,
            site_rgba_f32,
        ) = ctx.saved_tensors
        track_count = int(ctx.track_count)
        frame_count = int(ctx.frame_count)
        boundary_count = int(ctx.boundary_count)
        block_size = int(ctx.block_size)
        grad_alpha_is_none = grad_alpha is None
        grad_depth_is_none = grad_depth is None
        if grad_rgb is None:
            grad_rgb = torch.zeros((track_count, frame_count, 3), device=site_rgba_f32.device, dtype=torch.float32)
        if grad_alpha is None:
            grad_alpha = torch.zeros((track_count, frame_count), device=site_rgba_f32.device, dtype=torch.float32)
        if grad_depth is None:
            grad_depth = torch.zeros((track_count, frame_count), device=site_rgba_f32.device, dtype=torch.float32)
        if grad_alpha_is_none and grad_depth_is_none:
            grad_site_rgba = endpoint_record_edit_block_coeff_vjp_direct_atomic_rgb_only(
                coeff_f32,
                frame_t_f32,
                anchor_offsets_i32,
                anchor_owner_i32,
                anchor_left_i32,
                anchor_right_i32,
                track_block_change_offsets_i32,
                block_change_frame_i32,
                block_op_offsets_i32,
                block_op_type_i32,
                block_op_pos_i32,
                block_op_owner_i32,
                block_op_left_i32,
                block_op_right_i32,
                site_rgba_f32,
                grad_rgb,
                ctx.config,
                track_count=track_count,
                frame_count=frame_count,
                boundary_count=boundary_count,
                block_size=block_size,
            )
        else:
            grad_site_rgba = endpoint_record_edit_vjp_direct_atomic_grad_only(
                boundary_f32,
                rays_f32,
                frame_t_f32,
                base_offsets_i32,
                base_owner_i32,
                base_left_i32,
                base_right_i32,
                track_change_offsets_i32,
                change_frame_i32,
                op_offsets_i32,
                op_type_i32,
                op_pos_i32,
                op_owner_i32,
                op_left_i32,
                op_right_i32,
                site_rgba_f32,
                grad_rgb,
                grad_alpha,
                grad_depth,
                ctx.config,
                track_count=track_count,
                frame_count=frame_count,
            )
        return (*([None] * 28), grad_site_rgba, *([None] * 8))


def endpoint_record_edit_block_coeff_rgba_depth_autograd(
    coeff_f32: Tensor,
    boundary_f32: Tensor,
    rays_f32: Tensor,
    frame_t_f32: Tensor,
    anchor_offsets_i32: Tensor,
    anchor_owner_i32: Tensor,
    anchor_left_i32: Tensor,
    anchor_right_i32: Tensor,
    track_block_change_offsets_i32: Tensor,
    block_change_frame_i32: Tensor,
    block_op_offsets_i32: Tensor,
    block_op_type_i32: Tensor,
    block_op_pos_i32: Tensor,
    block_op_owner_i32: Tensor,
    block_op_left_i32: Tensor,
    block_op_right_i32: Tensor,
    base_offsets_i32: Tensor,
    base_owner_i32: Tensor,
    base_left_i32: Tensor,
    base_right_i32: Tensor,
    track_change_offsets_i32: Tensor,
    change_frame_i32: Tensor,
    op_offsets_i32: Tensor,
    op_type_i32: Tensor,
    op_pos_i32: Tensor,
    op_owner_i32: Tensor,
    op_left_i32: Tensor,
    op_right_i32: Tensor,
    site_rgba_f32: Tensor,
    config: RealRayReplayConfig,
    *,
    track_count: int,
    frame_count: int,
    boundary_count: int,
    block_size: int = 4,
) -> tuple[Tensor, Tensor, Tensor]:
    """Replay coefficient-cached block edits with frozen-geometry site-RGBA autograd.

    Forward and RGB-only backward use coefficient-cached block replay. Full
    alpha/depth backward falls back to the exact endpoint-record edit VJP.
    """
    return _EndpointRecordEditBlockCoeffRGBADepthReplayFunction.apply(
        coeff_f32,
        boundary_f32,
        rays_f32,
        frame_t_f32,
        anchor_offsets_i32,
        anchor_owner_i32,
        anchor_left_i32,
        anchor_right_i32,
        track_block_change_offsets_i32,
        block_change_frame_i32,
        block_op_offsets_i32,
        block_op_type_i32,
        block_op_pos_i32,
        block_op_owner_i32,
        block_op_left_i32,
        block_op_right_i32,
        base_offsets_i32,
        base_owner_i32,
        base_left_i32,
        base_right_i32,
        track_change_offsets_i32,
        change_frame_i32,
        op_offsets_i32,
        op_type_i32,
        op_pos_i32,
        op_owner_i32,
        op_left_i32,
        op_right_i32,
        site_rgba_f32,
        int(track_count),
        int(frame_count),
        int(boundary_count),
        int(block_size),
        config.near,
        config.far,
        config.invalid_epsilon,
        config.transmittance_threshold,
    )


class _EndpointRecordEditBlockCoeffRGBReplayFunction(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx: torch.autograd.function.FunctionCtx,
        coeff_f32: Tensor,
        boundary_f32: Tensor,
        rays_f32: Tensor,
        frame_t_f32: Tensor,
        anchor_offsets_i32: Tensor,
        anchor_owner_i32: Tensor,
        anchor_left_i32: Tensor,
        anchor_right_i32: Tensor,
        track_block_change_offsets_i32: Tensor,
        block_change_frame_i32: Tensor,
        block_op_offsets_i32: Tensor,
        block_op_type_i32: Tensor,
        block_op_pos_i32: Tensor,
        block_op_owner_i32: Tensor,
        block_op_left_i32: Tensor,
        block_op_right_i32: Tensor,
        base_offsets_i32: Tensor,
        base_owner_i32: Tensor,
        base_left_i32: Tensor,
        base_right_i32: Tensor,
        track_change_offsets_i32: Tensor,
        change_frame_i32: Tensor,
        op_offsets_i32: Tensor,
        op_type_i32: Tensor,
        op_pos_i32: Tensor,
        op_owner_i32: Tensor,
        op_left_i32: Tensor,
        op_right_i32: Tensor,
        site_rgba_f32: Tensor,
        track_count: int,
        frame_count: int,
        boundary_count: int,
        block_size: int,
        near: float,
        far: float,
        invalid_epsilon: float,
        transmittance_threshold: float,
    ) -> Tensor:
        del boundary_f32, rays_f32, base_offsets_i32, base_owner_i32, base_left_i32, base_right_i32
        del track_change_offsets_i32, change_frame_i32, op_offsets_i32, op_type_i32, op_pos_i32, op_owner_i32
        del op_left_i32, op_right_i32
        config = RealRayReplayConfig(
            near=float(near),
            far=float(far),
            invalid_epsilon=float(invalid_epsilon),
            transmittance_threshold=float(transmittance_threshold),
        )
        output_rgb = endpoint_record_edit_block_coeff_rgb_replay(
            coeff_f32,
            frame_t_f32,
            anchor_offsets_i32,
            anchor_owner_i32,
            anchor_left_i32,
            anchor_right_i32,
            track_block_change_offsets_i32,
            block_change_frame_i32,
            block_op_offsets_i32,
            block_op_type_i32,
            block_op_pos_i32,
            block_op_owner_i32,
            block_op_left_i32,
            block_op_right_i32,
            site_rgba_f32,
            config,
            track_count=int(track_count),
            frame_count=int(frame_count),
            boundary_count=int(boundary_count),
            block_size=int(block_size),
        )
        ctx.save_for_backward(
            coeff_f32,
            frame_t_f32,
            anchor_offsets_i32,
            anchor_owner_i32,
            anchor_left_i32,
            anchor_right_i32,
            track_block_change_offsets_i32,
            block_change_frame_i32,
            block_op_offsets_i32,
            block_op_type_i32,
            block_op_pos_i32,
            block_op_owner_i32,
            block_op_left_i32,
            block_op_right_i32,
            site_rgba_f32,
        )
        ctx.config = config
        ctx.track_count = int(track_count)
        ctx.frame_count = int(frame_count)
        ctx.boundary_count = int(boundary_count)
        ctx.block_size = int(block_size)
        return output_rgb

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: torch.autograd.function.FunctionCtx,
        grad_rgb: Tensor | None,
    ) -> tuple[Tensor | None, ...]:
        (
            coeff_f32,
            frame_t_f32,
            anchor_offsets_i32,
            anchor_owner_i32,
            anchor_left_i32,
            anchor_right_i32,
            track_block_change_offsets_i32,
            block_change_frame_i32,
            block_op_offsets_i32,
            block_op_type_i32,
            block_op_pos_i32,
            block_op_owner_i32,
            block_op_left_i32,
            block_op_right_i32,
            site_rgba_f32,
        ) = ctx.saved_tensors
        track_count = int(ctx.track_count)
        frame_count = int(ctx.frame_count)
        if grad_rgb is None:
            grad_rgb = torch.zeros((track_count, frame_count, 3), device=site_rgba_f32.device, dtype=torch.float32)
        grad_site_rgba = endpoint_record_edit_block_coeff_vjp_direct_atomic_rgb_only(
            coeff_f32,
            frame_t_f32,
            anchor_offsets_i32,
            anchor_owner_i32,
            anchor_left_i32,
            anchor_right_i32,
            track_block_change_offsets_i32,
            block_change_frame_i32,
            block_op_offsets_i32,
            block_op_type_i32,
            block_op_pos_i32,
            block_op_owner_i32,
            block_op_left_i32,
            block_op_right_i32,
            site_rgba_f32,
            grad_rgb,
            ctx.config,
            track_count=track_count,
            frame_count=frame_count,
            boundary_count=int(ctx.boundary_count),
            block_size=int(ctx.block_size),
        )
        return (*([None] * 28), grad_site_rgba, *([None] * 8))


def endpoint_record_edit_block_coeff_rgb_autograd(
    coeff_f32: Tensor,
    boundary_f32: Tensor,
    rays_f32: Tensor,
    frame_t_f32: Tensor,
    anchor_offsets_i32: Tensor,
    anchor_owner_i32: Tensor,
    anchor_left_i32: Tensor,
    anchor_right_i32: Tensor,
    track_block_change_offsets_i32: Tensor,
    block_change_frame_i32: Tensor,
    block_op_offsets_i32: Tensor,
    block_op_type_i32: Tensor,
    block_op_pos_i32: Tensor,
    block_op_owner_i32: Tensor,
    block_op_left_i32: Tensor,
    block_op_right_i32: Tensor,
    base_offsets_i32: Tensor,
    base_owner_i32: Tensor,
    base_left_i32: Tensor,
    base_right_i32: Tensor,
    track_change_offsets_i32: Tensor,
    change_frame_i32: Tensor,
    op_offsets_i32: Tensor,
    op_type_i32: Tensor,
    op_pos_i32: Tensor,
    op_owner_i32: Tensor,
    op_left_i32: Tensor,
    op_right_i32: Tensor,
    site_rgba_f32: Tensor,
    config: RealRayReplayConfig,
    *,
    track_count: int,
    frame_count: int,
    boundary_count: int,
    block_size: int = 4,
) -> Tensor:
    """RGB-only coefficient-cached replay with the existing RGB VJP."""
    return _EndpointRecordEditBlockCoeffRGBReplayFunction.apply(
        coeff_f32,
        boundary_f32,
        rays_f32,
        frame_t_f32,
        anchor_offsets_i32,
        anchor_owner_i32,
        anchor_left_i32,
        anchor_right_i32,
        track_block_change_offsets_i32,
        block_change_frame_i32,
        block_op_offsets_i32,
        block_op_type_i32,
        block_op_pos_i32,
        block_op_owner_i32,
        block_op_left_i32,
        block_op_right_i32,
        base_offsets_i32,
        base_owner_i32,
        base_left_i32,
        base_right_i32,
        track_change_offsets_i32,
        change_frame_i32,
        op_offsets_i32,
        op_type_i32,
        op_pos_i32,
        op_owner_i32,
        op_left_i32,
        op_right_i32,
        site_rgba_f32,
        int(track_count),
        int(frame_count),
        int(boundary_count),
        int(block_size),
        config.near,
        config.far,
        config.invalid_epsilon,
        config.transmittance_threshold,
    )


class _EndpointRunRGBADepthReplayFunction(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx: torch.autograd.function.FunctionCtx,
        run_offsets_i32: Tensor,
        run_owner_i32: Tensor,
        run_start_f32: Tensor,
        run_end_f32: Tensor,
        site_rgba_f32: Tensor,
        track_count: int,
        frame_count: int,
        near: float,
        far: float,
        invalid_epsilon: float,
        transmittance_threshold: float,
    ) -> tuple[Tensor, Tensor, Tensor]:
        config = RealRayReplayConfig(
            near=float(near),
            far=float(far),
            invalid_epsilon=float(invalid_epsilon),
            transmittance_threshold=float(transmittance_threshold),
        )
        output_rgb, output_alpha, output_depth = endpoint_run_rgba_depth_replay(
            run_offsets_i32,
            run_owner_i32,
            run_start_f32,
            run_end_f32,
            site_rgba_f32,
            config,
            track_count=int(track_count),
            frame_count=int(frame_count),
        )
        ctx.save_for_backward(
            run_offsets_i32,
            run_owner_i32,
            run_start_f32,
            run_end_f32,
            site_rgba_f32,
        )
        ctx.config = config
        ctx.track_count = int(track_count)
        ctx.frame_count = int(frame_count)
        return output_rgb, output_alpha, output_depth

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: torch.autograd.function.FunctionCtx,
        grad_rgb: Tensor | None,
        grad_alpha: Tensor | None,
        grad_depth: Tensor | None,
    ) -> tuple[None, None, None, None, Tensor, None, None, None, None, None, None]:
        (
            run_offsets_i32,
            run_owner_i32,
            run_start_f32,
            run_end_f32,
            site_rgba_f32,
        ) = ctx.saved_tensors
        track_count = int(ctx.track_count)
        frame_count = int(ctx.frame_count)
        if grad_rgb is None:
            grad_rgb = torch.zeros((track_count, frame_count, 3), device=site_rgba_f32.device, dtype=torch.float32)
        if grad_alpha is None:
            grad_alpha = torch.zeros((track_count, frame_count), device=site_rgba_f32.device, dtype=torch.float32)
        if grad_depth is None:
            grad_depth = torch.zeros((track_count, frame_count), device=site_rgba_f32.device, dtype=torch.float32)
        grad_site_rgba = endpoint_run_vjp_direct_atomic_grad_only(
            run_offsets_i32,
            run_owner_i32,
            run_start_f32,
            run_end_f32,
            site_rgba_f32,
            grad_rgb,
            grad_alpha,
            grad_depth,
            ctx.config,
            track_count=track_count,
            frame_count=frame_count,
        )
        return None, None, None, None, grad_site_rgba, None, None, None, None, None, None


def endpoint_run_rgba_depth_autograd(
    run_offsets_i32: Tensor,
    run_owner_i32: Tensor,
    run_start_f32: Tensor,
    run_end_f32: Tensor,
    site_rgba_f32: Tensor,
    config: RealRayReplayConfig,
    *,
    track_count: int,
    frame_count: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Replay endpoint runs with frozen-geometry site-RGBA autograd."""
    return _EndpointRunRGBADepthReplayFunction.apply(
        run_offsets_i32,
        run_owner_i32,
        run_start_f32,
        run_end_f32,
        site_rgba_f32,
        int(track_count),
        int(frame_count),
        config.near,
        config.far,
        config.invalid_epsilon,
        config.transmittance_threshold,
    )


def segment_tape_vjp_direct_atomic_grad_only(
    segment_offsets_i32: Tensor,
    segment_owner_i32: Tensor,
    segment_length_f32: Tensor,
    segment_mid_f32: Tensor,
    site_rgba_f32: Tensor,
    grad_rgb_f32: Tensor,
    grad_alpha_f32: Tensor,
    grad_depth_f32: Tensor,
    config: RealRayReplayConfig,
    *,
    track_count: int,
    frame_count: int,
) -> Tensor:
    """Accumulate fixed-geometry compact segment-tape site RGBA gradients."""
    segment_offsets_i32 = segment_offsets_i32.contiguous()
    segment_owner_i32 = segment_owner_i32.contiguous()
    segment_length_f32 = segment_length_f32.contiguous()
    segment_mid_f32 = segment_mid_f32.contiguous()
    site_rgba_f32 = site_rgba_f32.contiguous()
    grad_rgb_f32 = grad_rgb_f32.contiguous()
    grad_alpha_f32 = grad_alpha_f32.contiguous()
    grad_depth_f32 = grad_depth_f32.contiguous()
    config_i32, config_f32 = _segment_tape_config_tensors(
        segment_offsets_i32=segment_offsets_i32,
        segment_owner_i32=segment_owner_i32,
        segment_length_f32=segment_length_f32,
        segment_mid_f32=segment_mid_f32,
        site_rgba_f32=site_rgba_f32,
        config=config,
        track_count=track_count,
        frame_count=frame_count,
        max_segments_per_sample=129,
    )
    if grad_rgb_f32.device.type != "mps" or grad_rgb_f32.dtype != torch.float32:
        raise ValueError("grad_rgb_f32 must be float32 on MPS")
    if grad_rgb_f32.shape != (track_count, frame_count, 3):
        raise ValueError("grad_rgb_f32 must have shape [track_count, frame_count, 3]")
    if grad_alpha_f32.device.type != "mps" or grad_alpha_f32.dtype != torch.float32:
        raise ValueError("grad_alpha_f32 must be float32 on MPS")
    if grad_alpha_f32.shape != (track_count, frame_count):
        raise ValueError("grad_alpha_f32 must have shape [track_count, frame_count]")
    if grad_depth_f32.device.type != "mps" or grad_depth_f32.dtype != torch.float32:
        raise ValueError("grad_depth_f32 must be float32 on MPS")
    if grad_depth_f32.shape != (track_count, frame_count):
        raise ValueError("grad_depth_f32 must have shape [track_count, frame_count]")
    ops = torch.ops.world_foam_lane2_fused_slab_v0
    op_name = "segment_tape_vjp_direct_atomic_grad_only"
    if not hasattr(ops, op_name):
        raise RuntimeError(f"world_foam_lane2_fused_slab_v0 {op_name} op not found. Build this variant first.")
    return getattr(ops, op_name)(
        segment_offsets_i32,
        segment_owner_i32,
        segment_length_f32,
        segment_mid_f32,
        site_rgba_f32,
        grad_rgb_f32,
        grad_alpha_f32,
        grad_depth_f32,
        config_i32,
        config_f32,
    )


def segment_tape_vjp_direct_atomic_track(
    segment_offsets_i32: Tensor,
    segment_owner_i32: Tensor,
    segment_length_f32: Tensor,
    segment_mid_f32: Tensor,
    site_rgba_f32: Tensor,
    grad_rgb_f32: Tensor,
    grad_alpha_f32: Tensor,
    grad_depth_f32: Tensor,
    config: RealRayReplayConfig,
    *,
    track_count: int,
    frame_count: int,
) -> Tensor:
    """Accumulate compact segment-tape gradients once per track before atomics."""
    segment_offsets_i32 = segment_offsets_i32.contiguous()
    segment_owner_i32 = segment_owner_i32.contiguous()
    segment_length_f32 = segment_length_f32.contiguous()
    segment_mid_f32 = segment_mid_f32.contiguous()
    site_rgba_f32 = site_rgba_f32.contiguous()
    grad_rgb_f32 = grad_rgb_f32.contiguous()
    grad_alpha_f32 = grad_alpha_f32.contiguous()
    grad_depth_f32 = grad_depth_f32.contiguous()
    config_i32, config_f32 = _segment_tape_config_tensors(
        segment_offsets_i32=segment_offsets_i32,
        segment_owner_i32=segment_owner_i32,
        segment_length_f32=segment_length_f32,
        segment_mid_f32=segment_mid_f32,
        site_rgba_f32=site_rgba_f32,
        config=config,
        track_count=track_count,
        frame_count=frame_count,
        max_segments_per_sample=129,
    )
    if grad_rgb_f32.device.type != "mps" or grad_rgb_f32.dtype != torch.float32:
        raise ValueError("grad_rgb_f32 must be float32 on MPS")
    if grad_rgb_f32.shape != (track_count, frame_count, 3):
        raise ValueError("grad_rgb_f32 must have shape [track_count, frame_count, 3]")
    if grad_alpha_f32.device.type != "mps" or grad_alpha_f32.dtype != torch.float32:
        raise ValueError("grad_alpha_f32 must be float32 on MPS")
    if grad_alpha_f32.shape != (track_count, frame_count):
        raise ValueError("grad_alpha_f32 must have shape [track_count, frame_count]")
    if grad_depth_f32.device.type != "mps" or grad_depth_f32.dtype != torch.float32:
        raise ValueError("grad_depth_f32 must be float32 on MPS")
    if grad_depth_f32.shape != (track_count, frame_count):
        raise ValueError("grad_depth_f32 must have shape [track_count, frame_count]")
    ops = torch.ops.world_foam_lane2_fused_slab_v0
    op_name = "segment_tape_vjp_direct_atomic_track"
    if not hasattr(ops, op_name):
        raise RuntimeError(f"world_foam_lane2_fused_slab_v0 {op_name} op not found. Build this variant first.")
    return getattr(ops, op_name)(
        segment_offsets_i32,
        segment_owner_i32,
        segment_length_f32,
        segment_mid_f32,
        site_rgba_f32,
        grad_rgb_f32,
        grad_alpha_f32,
        grad_depth_f32,
        config_i32,
        config_f32,
    )


class _SegmentTapeRGBADepthReplayFunction(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx: torch.autograd.function.FunctionCtx,
        segment_offsets_i32: Tensor,
        segment_owner_i32: Tensor,
        segment_length_f32: Tensor,
        segment_mid_f32: Tensor,
        site_rgba_f32: Tensor,
        track_count: int,
        frame_count: int,
        near: float,
        far: float,
        invalid_epsilon: float,
        transmittance_threshold: float,
        use_track_vjp: bool,
    ) -> tuple[Tensor, Tensor, Tensor]:
        config = RealRayReplayConfig(
            near=float(near),
            far=float(far),
            invalid_epsilon=float(invalid_epsilon),
            transmittance_threshold=float(transmittance_threshold),
        )
        output_rgb, output_alpha, output_depth = segment_tape_rgba_depth_replay(
            segment_offsets_i32,
            segment_owner_i32,
            segment_length_f32,
            segment_mid_f32,
            site_rgba_f32,
            config,
            track_count=int(track_count),
            frame_count=int(frame_count),
        )
        ctx.save_for_backward(
            segment_offsets_i32,
            segment_owner_i32,
            segment_length_f32,
            segment_mid_f32,
            site_rgba_f32,
        )
        ctx.config = config
        ctx.track_count = int(track_count)
        ctx.frame_count = int(frame_count)
        ctx.use_track_vjp = bool(use_track_vjp)
        return output_rgb, output_alpha, output_depth

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: torch.autograd.function.FunctionCtx,
        grad_rgb: Tensor | None,
        grad_alpha: Tensor | None,
        grad_depth: Tensor | None,
    ) -> tuple[None, None, None, None, Tensor, None, None, None, None, None, None, None]:
        (
            segment_offsets_i32,
            segment_owner_i32,
            segment_length_f32,
            segment_mid_f32,
            site_rgba_f32,
        ) = ctx.saved_tensors
        track_count = int(ctx.track_count)
        frame_count = int(ctx.frame_count)
        if grad_rgb is None:
            grad_rgb = torch.zeros((track_count, frame_count, 3), device=site_rgba_f32.device, dtype=torch.float32)
        if grad_alpha is None:
            grad_alpha = torch.zeros((track_count, frame_count), device=site_rgba_f32.device, dtype=torch.float32)
        if grad_depth is None:
            grad_depth = torch.zeros((track_count, frame_count), device=site_rgba_f32.device, dtype=torch.float32)
        if bool(ctx.use_track_vjp):
            grad_site_rgba = segment_tape_vjp_direct_atomic_track(
                segment_offsets_i32,
                segment_owner_i32,
                segment_length_f32,
                segment_mid_f32,
                site_rgba_f32,
                grad_rgb,
                grad_alpha,
                grad_depth,
                ctx.config,
                track_count=track_count,
                frame_count=frame_count,
            )
        else:
            grad_site_rgba = segment_tape_vjp_direct_atomic_grad_only(
                segment_offsets_i32,
                segment_owner_i32,
                segment_length_f32,
                segment_mid_f32,
                site_rgba_f32,
                grad_rgb,
                grad_alpha,
                grad_depth,
                ctx.config,
                track_count=track_count,
                frame_count=frame_count,
            )
        return None, None, None, None, grad_site_rgba, None, None, None, None, None, None, None


def segment_tape_rgba_depth_autograd(
    segment_offsets_i32: Tensor,
    segment_owner_i32: Tensor,
    segment_length_f32: Tensor,
    segment_mid_f32: Tensor,
    site_rgba_f32: Tensor,
    config: RealRayReplayConfig,
    *,
    track_count: int,
    frame_count: int,
    vjp_mode: str = "direct_atomic_grad_only",
) -> tuple[Tensor, Tensor, Tensor]:
    """Replay a compact segment tape with frozen-geometry site-RGBA autograd."""
    if vjp_mode not in {"direct_atomic_grad_only", "direct_atomic_track"}:
        raise ValueError("vjp_mode must be 'direct_atomic_grad_only' or 'direct_atomic_track'")
    return _SegmentTapeRGBADepthReplayFunction.apply(
        segment_offsets_i32,
        segment_owner_i32,
        segment_length_f32,
        segment_mid_f32,
        site_rgba_f32,
        int(track_count),
        int(frame_count),
        config.near,
        config.far,
        config.invalid_epsilon,
        config.transmittance_threshold,
        vjp_mode == "direct_atomic_track",
    )


def fused_slab_affine_num32_den16_vjp_direct_atomic_grad_only_ownerupdate(
    row_index_i32: Tensor,
    candidate_row_offsets_i32: Tensor,
    candidate_boundary_ids_i32: Tensor,
    candidate_depth_num_f32: Tensor,
    candidate_depth_den_f16: Tensor,
    boundary_site_pairs_i32: Tensor,
    sites_f32: Tensor,
    site_rgba_f32: Tensor,
    ray_coeff_f32: Tensor,
    frame_t_f32: Tensor,
    grad_rgb_f32: Tensor,
    grad_alpha_f32: Tensor,
    grad_depth_f32: Tensor,
    config: RealRayReplayConfig,
    *,
    time_slab_count: int,
    row_count: int,
) -> Tensor:
    """Diagnostic VJP using midpoint owner selection for every replay segment."""
    row_index_i32 = row_index_i32.contiguous()
    candidate_row_offsets_i32 = candidate_row_offsets_i32.contiguous()
    candidate_boundary_ids_i32 = candidate_boundary_ids_i32.contiguous()
    candidate_depth_num_f32 = candidate_depth_num_f32.contiguous()
    candidate_depth_den_f16 = candidate_depth_den_f16.contiguous()
    boundary_site_pairs_i32 = boundary_site_pairs_i32.contiguous()
    sites_f32 = sites_f32.contiguous()
    site_rgba_f32 = site_rgba_f32.contiguous()
    ray_coeff_f32 = ray_coeff_f32.contiguous()
    frame_t_f32 = frame_t_f32.contiguous()
    grad_rgb_f32 = grad_rgb_f32.contiguous()
    grad_alpha_f32 = grad_alpha_f32.contiguous()
    grad_depth_f32 = grad_depth_f32.contiguous()
    _require_mps_tensor(row_index_i32, name="row_index_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(candidate_row_offsets_i32, name="candidate_row_offsets_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(candidate_boundary_ids_i32, name="candidate_boundary_ids_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(candidate_depth_num_f32, name="candidate_depth_num_f32", dtype=torch.float32, cols=2)
    _require_mps_tensor(candidate_depth_den_f16, name="candidate_depth_den_f16", dtype=torch.float16, cols=2)
    _require_mps_tensor(boundary_site_pairs_i32, name="boundary_site_pairs_i32", dtype=torch.int32, cols=2)
    _require_mps_tensor(sites_f32, name="sites_f32", dtype=torch.float32, cols=5)
    _require_mps_tensor(site_rgba_f32, name="site_rgba_f32", dtype=torch.float32, cols=4)
    _require_mps_tensor(ray_coeff_f32, name="ray_coeff_f32", dtype=torch.float32, cols=12)
    _require_mps_tensor(frame_t_f32, name="frame_t_f32", dtype=torch.float32, cols=None)
    if grad_rgb_f32.device.type != "mps" or grad_rgb_f32.dtype != torch.float32:
        raise ValueError("grad_rgb_f32 must be float32 on MPS")
    if grad_rgb_f32.ndim != 3 or grad_rgb_f32.shape[2] != 3:
        raise ValueError("grad_rgb_f32 must have shape [track_count, frame_count, 3]")
    if grad_alpha_f32.device.type != "mps" or grad_alpha_f32.dtype != torch.float32:
        raise ValueError("grad_alpha_f32 must be float32 on MPS")
    if grad_alpha_f32.ndim != 2:
        raise ValueError("grad_alpha_f32 must have shape [track_count, frame_count]")
    if grad_depth_f32.device.type != "mps" or grad_depth_f32.dtype != torch.float32:
        raise ValueError("grad_depth_f32 must be float32 on MPS")
    if grad_depth_f32.ndim != 2:
        raise ValueError("grad_depth_f32 must have shape [track_count, frame_count]")
    if ray_coeff_f32.shape[0] <= 0 or frame_t_f32.shape[0] <= 0:
        raise ValueError("ray_coeff_f32 and frame_t_f32 must be nonempty")
    if sites_f32.shape[0] <= 0 or sites_f32.shape[0] > 64:
        raise ValueError("sites_f32 row count must be in [1, 64]")
    if site_rgba_f32.shape[0] != sites_f32.shape[0]:
        raise ValueError("site_rgba_f32 row count must match sites_f32")
    if row_index_i32.shape[0] != ray_coeff_f32.shape[0]:
        raise ValueError("row_index_i32 length must match track count")
    if time_slab_count <= 0:
        raise ValueError("time_slab_count must be positive")
    if row_count <= 0:
        raise ValueError("row_count must be positive")
    if candidate_depth_den_f16.shape[0] != candidate_depth_num_f32.shape[0]:
        raise ValueError("candidate numerator and denominator row counts must match")
    if candidate_boundary_ids_i32.shape[0] != candidate_depth_num_f32.shape[0]:
        raise ValueError("candidate_boundary_ids_i32 length must match candidate rows")
    if candidate_row_offsets_i32.shape[0] != row_count * time_slab_count + 1:
        raise ValueError("candidate_row_offsets_i32 length must be row_count * time_slab_count + 1")
    row_index_cpu = row_index_i32.detach().cpu()
    row_offsets_cpu = candidate_row_offsets_i32.detach().cpu()
    candidate_ids_cpu = candidate_boundary_ids_i32.detach().cpu()
    boundary_pairs_cpu = boundary_site_pairs_i32.detach().cpu()
    if int(row_index_cpu.min().item()) < 0 or int(row_index_cpu.max().item()) >= row_count:
        raise ValueError("row_index_i32 values must be in [0, row_count)")
    _validate_csr_offsets_cpu(row_offsets_cpu, candidate_count=candidate_depth_num_f32.shape[0])
    if candidate_ids_cpu.numel() and (
        int(candidate_ids_cpu.min().item()) < 0
        or int(candidate_ids_cpu.max().item()) >= boundary_site_pairs_i32.shape[0]
    ):
        raise ValueError("candidate_boundary_ids_i32 values must be in [0, boundary_count)")
    if boundary_pairs_cpu.numel() and (
        int(boundary_pairs_cpu.min().item()) < 0 or int(boundary_pairs_cpu.max().item()) >= sites_f32.shape[0]
    ):
        raise ValueError("boundary_site_pairs_i32 values must be in [0, site_count)")
    expected_shape = (ray_coeff_f32.shape[0], frame_t_f32.shape[0])
    if grad_rgb_f32.shape[:2] != expected_shape:
        raise ValueError("grad_rgb_f32 shape must be [track_count, frame_count, 3]")
    if grad_alpha_f32.shape != expected_shape or grad_depth_f32.shape != expected_shape:
        raise ValueError("grad_alpha_f32 and grad_depth_f32 must have shape [track_count, frame_count]")
    if config.far <= config.near:
        raise ValueError("config.far must be greater than config.near")
    if config.invalid_epsilon <= 0.0:
        raise ValueError("config.invalid_epsilon must be positive")
    if config.transmittance_threshold < 0.0:
        raise ValueError("config.transmittance_threshold must be nonnegative")

    ops = torch.ops.world_foam_lane2_fused_slab_v0
    op_name = "fused_slab_affine_num32_den16_vjp_direct_atomic_grad_only_ownerupdate"
    if not hasattr(ops, op_name):
        raise RuntimeError(f"world_foam_lane2_fused_slab_v0 {op_name} op not found. Build this variant first.")
    config_i32 = torch.tensor(
        [
            0,
            ray_coeff_f32.shape[0],
            sites_f32.shape[0],
            frame_t_f32.shape[0],
            time_slab_count,
            row_count,
            candidate_depth_num_f32.shape[0],
            boundary_site_pairs_i32.shape[0],
        ],
        device=row_index_i32.device,
        dtype=torch.int32,
    )
    config_f32 = torch.tensor(
        [config.near, config.far, config.invalid_epsilon, config.transmittance_threshold],
        device=row_index_i32.device,
        dtype=torch.float32,
    )
    return getattr(ops, op_name)(
        row_index_i32,
        candidate_row_offsets_i32,
        candidate_boundary_ids_i32,
        candidate_depth_num_f32,
        candidate_depth_den_f16,
        boundary_site_pairs_i32,
        sites_f32,
        site_rgba_f32,
        ray_coeff_f32,
        frame_t_f32,
        grad_rgb_f32,
        grad_alpha_f32,
        grad_depth_f32,
        config_i32,
        config_f32,
    )


def fused_slab_affine_num32_den16_vjp_direct_atomic_rgb_only(
    row_index_i32: Tensor,
    candidate_row_offsets_i32: Tensor,
    candidate_depth_num_f32: Tensor,
    candidate_depth_den_f16: Tensor,
    sites_f32: Tensor,
    site_rgba_f32: Tensor,
    ray_coeff_f32: Tensor,
    frame_t_f32: Tensor,
    grad_rgb_f32: Tensor,
    config: RealRayReplayConfig,
    *,
    time_slab_count: int,
    row_count: int,
) -> Tensor:
    """Accumulate mixed affine site RGBA gradients for RGB-only losses."""
    row_index_i32 = row_index_i32.contiguous()
    candidate_row_offsets_i32 = candidate_row_offsets_i32.contiguous()
    candidate_depth_num_f32 = candidate_depth_num_f32.contiguous()
    candidate_depth_den_f16 = candidate_depth_den_f16.contiguous()
    sites_f32 = sites_f32.contiguous()
    site_rgba_f32 = site_rgba_f32.contiguous()
    ray_coeff_f32 = ray_coeff_f32.contiguous()
    frame_t_f32 = frame_t_f32.contiguous()
    grad_rgb_f32 = grad_rgb_f32.contiguous()
    _require_mps_tensor(row_index_i32, name="row_index_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(candidate_row_offsets_i32, name="candidate_row_offsets_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(candidate_depth_num_f32, name="candidate_depth_num_f32", dtype=torch.float32, cols=2)
    _require_mps_tensor(candidate_depth_den_f16, name="candidate_depth_den_f16", dtype=torch.float16, cols=2)
    _require_mps_tensor(sites_f32, name="sites_f32", dtype=torch.float32, cols=5)
    _require_mps_tensor(site_rgba_f32, name="site_rgba_f32", dtype=torch.float32, cols=4)
    _require_mps_tensor(ray_coeff_f32, name="ray_coeff_f32", dtype=torch.float32, cols=12)
    _require_mps_tensor(frame_t_f32, name="frame_t_f32", dtype=torch.float32, cols=None)
    if grad_rgb_f32.device.type != "mps" or grad_rgb_f32.dtype != torch.float32:
        raise ValueError("grad_rgb_f32 must be float32 on MPS")
    if grad_rgb_f32.ndim != 3 or grad_rgb_f32.shape[2] != 3:
        raise ValueError("grad_rgb_f32 must have shape [track_count, frame_count, 3]")
    if ray_coeff_f32.shape[0] <= 0 or frame_t_f32.shape[0] <= 0:
        raise ValueError("ray_coeff_f32 and frame_t_f32 must be nonempty")
    if sites_f32.shape[0] <= 0 or sites_f32.shape[0] > 64:
        raise ValueError("sites_f32 row count must be in [1, 64]")
    if site_rgba_f32.shape[0] != sites_f32.shape[0]:
        raise ValueError("site_rgba_f32 row count must match sites_f32")
    if row_index_i32.shape[0] != ray_coeff_f32.shape[0]:
        raise ValueError("row_index_i32 length must match track count")
    if time_slab_count <= 0:
        raise ValueError("time_slab_count must be positive")
    if row_count <= 0:
        raise ValueError("row_count must be positive")
    if candidate_depth_den_f16.shape[0] != candidate_depth_num_f32.shape[0]:
        raise ValueError("candidate numerator and denominator row counts must match")
    if candidate_row_offsets_i32.shape[0] != row_count * time_slab_count + 1:
        raise ValueError("candidate_row_offsets_i32 length must be row_count * time_slab_count + 1")
    row_index_cpu = row_index_i32.detach().cpu()
    row_offsets_cpu = candidate_row_offsets_i32.detach().cpu()
    if int(row_index_cpu.min().item()) < 0 or int(row_index_cpu.max().item()) >= row_count:
        raise ValueError("row_index_i32 values must be in [0, row_count)")
    _validate_csr_offsets_cpu(row_offsets_cpu, candidate_count=candidate_depth_num_f32.shape[0])
    expected_shape = (ray_coeff_f32.shape[0], frame_t_f32.shape[0])
    if grad_rgb_f32.shape[:2] != expected_shape:
        raise ValueError("grad_rgb_f32 shape must be [track_count, frame_count, 3]")
    if config.far <= config.near:
        raise ValueError("config.far must be greater than config.near")
    if config.invalid_epsilon <= 0.0:
        raise ValueError("config.invalid_epsilon must be positive")
    if config.transmittance_threshold < 0.0:
        raise ValueError("config.transmittance_threshold must be nonnegative")

    ops = torch.ops.world_foam_lane2_fused_slab_v0
    op_name = "fused_slab_affine_num32_den16_vjp_direct_atomic_rgb_only"
    if not hasattr(ops, op_name):
        raise RuntimeError(f"world_foam_lane2_fused_slab_v0 {op_name} op not found. Build this variant first.")
    config_i32 = torch.tensor(
        [
            0,
            ray_coeff_f32.shape[0],
            sites_f32.shape[0],
            frame_t_f32.shape[0],
            time_slab_count,
            row_count,
            candidate_depth_num_f32.shape[0],
            1,
        ],
        device=row_index_i32.device,
        dtype=torch.int32,
    )
    config_f32 = torch.tensor(
        [config.near, config.far, config.invalid_epsilon, config.transmittance_threshold],
        device=row_index_i32.device,
        dtype=torch.float32,
    )
    return getattr(ops, op_name)(
        row_index_i32,
        candidate_row_offsets_i32,
        candidate_depth_num_f32,
        candidate_depth_den_f16,
        sites_f32,
        site_rgba_f32,
        ray_coeff_f32,
        frame_t_f32,
        grad_rgb_f32,
        config_i32,
        config_f32,
    )


def fused_slab_affine_num32_den16_mse_vjp_direct_atomic_rgb_only(
    row_index_i32: Tensor,
    candidate_row_offsets_i32: Tensor,
    candidate_depth_num_f32: Tensor,
    candidate_depth_den_f16: Tensor,
    sites_f32: Tensor,
    site_rgba_f32: Tensor,
    ray_coeff_f32: Tensor,
    frame_t_f32: Tensor,
    target_rgb_f32: Tensor,
    config: RealRayReplayConfig,
    *,
    time_slab_count: int,
    row_count: int,
    _op_name: str = "fused_slab_affine_num32_den16_mse_vjp_direct_atomic_rgb_only",
) -> tuple[Tensor, Tensor]:
    """Compute RGB MSE and frozen-geometry site-RGBA gradients in one affine replay kernel."""
    row_index_i32 = row_index_i32.contiguous()
    candidate_row_offsets_i32 = candidate_row_offsets_i32.contiguous()
    candidate_depth_num_f32 = candidate_depth_num_f32.contiguous()
    candidate_depth_den_f16 = candidate_depth_den_f16.contiguous()
    sites_f32 = sites_f32.contiguous()
    site_rgba_f32 = site_rgba_f32.contiguous()
    ray_coeff_f32 = ray_coeff_f32.contiguous()
    frame_t_f32 = frame_t_f32.contiguous()
    target_rgb_f32 = target_rgb_f32.contiguous()
    _require_mps_tensor(row_index_i32, name="row_index_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(candidate_row_offsets_i32, name="candidate_row_offsets_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(candidate_depth_num_f32, name="candidate_depth_num_f32", dtype=torch.float32, cols=2)
    _require_mps_tensor(candidate_depth_den_f16, name="candidate_depth_den_f16", dtype=torch.float16, cols=2)
    _require_mps_tensor(sites_f32, name="sites_f32", dtype=torch.float32, cols=5)
    _require_mps_tensor(site_rgba_f32, name="site_rgba_f32", dtype=torch.float32, cols=4)
    _require_mps_tensor(ray_coeff_f32, name="ray_coeff_f32", dtype=torch.float32, cols=12)
    _require_mps_tensor(frame_t_f32, name="frame_t_f32", dtype=torch.float32, cols=None)
    if target_rgb_f32.device.type != "mps" or target_rgb_f32.dtype != torch.float32:
        raise ValueError("target_rgb_f32 must be float32 on MPS")
    if target_rgb_f32.ndim != 3 or target_rgb_f32.shape[2] != 3:
        raise ValueError("target_rgb_f32 must have shape [track_count, frame_count, 3]")
    if not target_rgb_f32.is_contiguous():
        raise ValueError("target_rgb_f32 must be contiguous")
    if ray_coeff_f32.shape[0] <= 0 or frame_t_f32.shape[0] <= 0:
        raise ValueError("ray_coeff_f32 and frame_t_f32 must be nonempty")
    if sites_f32.shape[0] <= 0 or sites_f32.shape[0] > 64:
        raise ValueError("sites_f32 row count must be in [1, 64]")
    if site_rgba_f32.shape[0] != sites_f32.shape[0]:
        raise ValueError("site_rgba_f32 row count must match sites_f32")
    if row_index_i32.shape[0] != ray_coeff_f32.shape[0]:
        raise ValueError("row_index_i32 length must match track count")
    if time_slab_count <= 0:
        raise ValueError("time_slab_count must be positive")
    if row_count <= 0:
        raise ValueError("row_count must be positive")
    if candidate_depth_den_f16.shape[0] != candidate_depth_num_f32.shape[0]:
        raise ValueError("candidate numerator and denominator row counts must match")
    if candidate_row_offsets_i32.shape[0] != row_count * time_slab_count + 1:
        raise ValueError("candidate_row_offsets_i32 length must be row_count * time_slab_count + 1")
    row_index_cpu = row_index_i32.detach().cpu()
    row_offsets_cpu = candidate_row_offsets_i32.detach().cpu()
    if int(row_index_cpu.min().item()) < 0 or int(row_index_cpu.max().item()) >= row_count:
        raise ValueError("row_index_i32 values must be in [0, row_count)")
    _validate_csr_offsets_cpu(
        row_offsets_cpu,
        candidate_count=candidate_depth_num_f32.shape[0],
        max_boundaries=MAX_REALRAY_FUSED_MSE_BOUNDARIES,
    )
    expected_shape = (ray_coeff_f32.shape[0], frame_t_f32.shape[0])
    if target_rgb_f32.shape[:2] != expected_shape:
        raise ValueError("target_rgb_f32 shape must be [track_count, frame_count, 3]")
    if config.far <= config.near:
        raise ValueError("config.far must be greater than config.near")
    if config.invalid_epsilon <= 0.0:
        raise ValueError("config.invalid_epsilon must be positive")
    if config.transmittance_threshold < 0.0:
        raise ValueError("config.transmittance_threshold must be nonnegative")

    ops = torch.ops.world_foam_lane2_fused_slab_v0
    if not hasattr(ops, _op_name):
        raise RuntimeError(f"world_foam_lane2_fused_slab_v0 {_op_name} op not found. Build this variant first.")
    config_i32 = torch.tensor(
        [
            0,
            ray_coeff_f32.shape[0],
            sites_f32.shape[0],
            frame_t_f32.shape[0],
            time_slab_count,
            row_count,
            candidate_depth_num_f32.shape[0],
            1,
        ],
        device=row_index_i32.device,
        dtype=torch.int32,
    )
    config_f32 = torch.tensor(
        [config.near, config.far, config.invalid_epsilon, config.transmittance_threshold],
        device=row_index_i32.device,
        dtype=torch.float32,
    )
    return getattr(ops, _op_name)(
        row_index_i32,
        candidate_row_offsets_i32,
        candidate_depth_num_f32,
        candidate_depth_den_f16,
        sites_f32,
        site_rgba_f32,
        ray_coeff_f32,
        frame_t_f32,
        target_rgb_f32,
        config_i32,
        config_f32,
    )


def fused_slab_affine_num32_den16_mse_vjp_direct_atomic_track_rgb_only(
    row_index_i32: Tensor,
    candidate_row_offsets_i32: Tensor,
    candidate_depth_num_f32: Tensor,
    candidate_depth_den_f16: Tensor,
    sites_f32: Tensor,
    site_rgba_f32: Tensor,
    ray_coeff_f32: Tensor,
    frame_t_f32: Tensor,
    target_rgb_f32: Tensor,
    config: RealRayReplayConfig,
    *,
    time_slab_count: int,
    row_count: int,
) -> tuple[Tensor, Tensor]:
    """Track-thread RGB MSE and frozen-geometry site-RGBA gradients for affine replay."""
    return fused_slab_affine_num32_den16_mse_vjp_direct_atomic_rgb_only(
        row_index_i32,
        candidate_row_offsets_i32,
        candidate_depth_num_f32,
        candidate_depth_den_f16,
        sites_f32,
        site_rgba_f32,
        ray_coeff_f32,
        frame_t_f32,
        target_rgb_f32,
        config,
        time_slab_count=time_slab_count,
        row_count=row_count,
        _op_name="fused_slab_affine_num32_den16_mse_vjp_direct_atomic_track_rgb_only",
    )


def fused_slab_affine_coeff16_mse_vjp_direct_atomic_rgb_only(
    row_index_i32: Tensor,
    candidate_row_offsets_i32: Tensor,
    candidate_depth_coeff_f16: Tensor,
    sites_f32: Tensor,
    site_rgba_f32: Tensor,
    ray_coeff_f32: Tensor,
    frame_t_f32: Tensor,
    target_rgb_f32: Tensor,
    config: RealRayReplayConfig,
    *,
    time_slab_count: int,
    row_count: int,
    _op_name: str = "fused_slab_affine_coeff16_mse_vjp_direct_atomic_rgb_only",
    _max_boundaries: int = MAX_REALRAY_FUSED_MSE_BOUNDARIES,
) -> tuple[Tensor, Tensor]:
    """Compute RGB MSE and site-RGBA gradients using fully half affine depth coefficients."""
    row_index_i32 = row_index_i32.contiguous()
    candidate_row_offsets_i32 = candidate_row_offsets_i32.contiguous()
    candidate_depth_coeff_f16 = candidate_depth_coeff_f16.contiguous()
    sites_f32 = sites_f32.contiguous()
    site_rgba_f32 = site_rgba_f32.contiguous()
    ray_coeff_f32 = ray_coeff_f32.contiguous()
    frame_t_f32 = frame_t_f32.contiguous()
    target_rgb_f32 = target_rgb_f32.contiguous()
    _require_mps_tensor(row_index_i32, name="row_index_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(candidate_row_offsets_i32, name="candidate_row_offsets_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(candidate_depth_coeff_f16, name="candidate_depth_coeff_f16", dtype=torch.float16, cols=4)
    _require_mps_tensor(sites_f32, name="sites_f32", dtype=torch.float32, cols=5)
    _require_mps_tensor(site_rgba_f32, name="site_rgba_f32", dtype=torch.float32, cols=4)
    _require_mps_tensor(ray_coeff_f32, name="ray_coeff_f32", dtype=torch.float32, cols=12)
    _require_mps_tensor(frame_t_f32, name="frame_t_f32", dtype=torch.float32, cols=None)
    if target_rgb_f32.device.type != "mps" or target_rgb_f32.dtype != torch.float32:
        raise ValueError("target_rgb_f32 must be float32 on MPS")
    if target_rgb_f32.ndim != 3 or target_rgb_f32.shape[2] != 3:
        raise ValueError("target_rgb_f32 must have shape [track_count, frame_count, 3]")
    if not target_rgb_f32.is_contiguous():
        raise ValueError("target_rgb_f32 must be contiguous")
    if ray_coeff_f32.shape[0] <= 0 or frame_t_f32.shape[0] <= 0:
        raise ValueError("ray_coeff_f32 and frame_t_f32 must be nonempty")
    if sites_f32.shape[0] <= 0 or sites_f32.shape[0] > 64:
        raise ValueError("sites_f32 row count must be in [1, 64]")
    if site_rgba_f32.shape[0] != sites_f32.shape[0]:
        raise ValueError("site_rgba_f32 row count must match sites_f32")
    if row_index_i32.shape[0] != ray_coeff_f32.shape[0]:
        raise ValueError("row_index_i32 length must match track count")
    if time_slab_count <= 0:
        raise ValueError("time_slab_count must be positive")
    if row_count <= 0:
        raise ValueError("row_count must be positive")
    if candidate_row_offsets_i32.shape[0] != row_count * time_slab_count + 1:
        raise ValueError("candidate_row_offsets_i32 length must be row_count * time_slab_count + 1")
    row_index_cpu = row_index_i32.detach().cpu()
    row_offsets_cpu = candidate_row_offsets_i32.detach().cpu()
    if int(row_index_cpu.min().item()) < 0 or int(row_index_cpu.max().item()) >= row_count:
        raise ValueError("row_index_i32 values must be in [0, row_count)")
    _validate_csr_offsets_cpu(
        row_offsets_cpu,
        candidate_count=candidate_depth_coeff_f16.shape[0],
        max_boundaries=_max_boundaries,
    )
    expected_shape = (ray_coeff_f32.shape[0], frame_t_f32.shape[0])
    if target_rgb_f32.shape[:2] != expected_shape:
        raise ValueError("target_rgb_f32 shape must be [track_count, frame_count, 3]")
    if config.far <= config.near:
        raise ValueError("config.far must be greater than config.near")
    if config.invalid_epsilon <= 0.0:
        raise ValueError("config.invalid_epsilon must be positive")
    if config.transmittance_threshold < 0.0:
        raise ValueError("config.transmittance_threshold must be nonnegative")

    ops = torch.ops.world_foam_lane2_fused_slab_v0
    if not hasattr(ops, _op_name):
        raise RuntimeError(f"world_foam_lane2_fused_slab_v0 {_op_name} op not found. Build this variant first.")
    config_i32 = torch.tensor(
        [
            0,
            ray_coeff_f32.shape[0],
            sites_f32.shape[0],
            frame_t_f32.shape[0],
            time_slab_count,
            row_count,
            candidate_depth_coeff_f16.shape[0],
            1,
        ],
        device=row_index_i32.device,
        dtype=torch.int32,
    )
    config_f32 = torch.tensor(
        [config.near, config.far, config.invalid_epsilon, config.transmittance_threshold],
        device=row_index_i32.device,
        dtype=torch.float32,
    )
    return getattr(ops, _op_name)(
        row_index_i32,
        candidate_row_offsets_i32,
        candidate_depth_coeff_f16,
        sites_f32,
        site_rgba_f32,
        ray_coeff_f32,
        frame_t_f32,
        target_rgb_f32,
        config_i32,
        config_f32,
    )


def fused_slab_affine_coeff16_cap224_mse_vjp_direct_atomic_rgb_only(
    row_index_i32: Tensor,
    candidate_row_offsets_i32: Tensor,
    candidate_depth_coeff_f16: Tensor,
    sites_f32: Tensor,
    site_rgba_f32: Tensor,
    ray_coeff_f32: Tensor,
    frame_t_f32: Tensor,
    target_rgb_f32: Tensor,
    config: RealRayReplayConfig,
    *,
    time_slab_count: int,
    row_count: int,
) -> tuple[Tensor, Tensor]:
    """Sample-parallel coeff16 RGB MSE VJP specialized for rows with at most 224 candidates."""
    return fused_slab_affine_coeff16_mse_vjp_direct_atomic_rgb_only(
        row_index_i32,
        candidate_row_offsets_i32,
        candidate_depth_coeff_f16,
        sites_f32,
        site_rgba_f32,
        ray_coeff_f32,
        frame_t_f32,
        target_rgb_f32,
        config,
        time_slab_count=time_slab_count,
        row_count=row_count,
        _op_name="fused_slab_affine_coeff16_cap224_mse_vjp_direct_atomic_rgb_only",
        _max_boundaries=MAX_REALRAY_FUSED_MSE_CAP224_BOUNDARIES,
    )


def fused_slab_affine_coeff16_densitymask_mse_vjp_direct_atomic_rgb_only(
    row_index_i32: Tensor,
    candidate_row_offsets_i32: Tensor,
    candidate_depth_coeff_f16: Tensor,
    sites_f32: Tensor,
    site_rgba_f32: Tensor,
    ray_coeff_f32: Tensor,
    frame_t_f32: Tensor,
    target_rgb_f32: Tensor,
    config: RealRayReplayConfig,
    *,
    time_slab_count: int,
    row_count: int,
) -> tuple[Tensor, Tensor]:
    """Sample-parallel coeff16 RGB MSE VJP caching the forward density activity bit."""
    return fused_slab_affine_coeff16_mse_vjp_direct_atomic_rgb_only(
        row_index_i32,
        candidate_row_offsets_i32,
        candidate_depth_coeff_f16,
        sites_f32,
        site_rgba_f32,
        ray_coeff_f32,
        frame_t_f32,
        target_rgb_f32,
        config,
        time_slab_count=time_slab_count,
        row_count=row_count,
        _op_name="fused_slab_affine_coeff16_densitymask_mse_vjp_direct_atomic_rgb_only",
    )


def fused_slab_affine_coeff16_mse_vjp_direct_atomic_framegroup16_cached_rgb_only(
    row_index_i32: Tensor,
    candidate_row_offsets_i32: Tensor,
    candidate_depth_coeff_f16: Tensor,
    sites_f32: Tensor,
    site_rgba_f32: Tensor,
    ray_coeff_f32: Tensor,
    frame_t_f32: Tensor,
    target_rgb_f32: Tensor,
    config: RealRayReplayConfig,
    *,
    time_slab_count: int,
    row_count: int,
    _op_name: str = "fused_slab_affine_coeff16_ownerupdate_mse_vjp_direct_atomic_rgb_only",
) -> tuple[Tensor, Tensor]:
    """Framegroup-16 coeff16 RGB MSE VJP that caches row coefficients in threadgroup memory."""
    if time_slab_count != 1:
        raise ValueError("framegroup16 cached coeff16 fused MSE currently requires time_slab_count == 1")
    return fused_slab_affine_coeff16_mse_vjp_direct_atomic_rgb_only(
        row_index_i32,
        candidate_row_offsets_i32,
        candidate_depth_coeff_f16,
        sites_f32,
        site_rgba_f32,
        ray_coeff_f32,
        frame_t_f32,
        target_rgb_f32,
        config,
        time_slab_count=time_slab_count,
        row_count=row_count,
        _op_name="fused_slab_affine_coeff16_mse_vjp_direct_atomic_framegroup16_cached_rgb_only",
    )


def fused_slab_affine_coeff16_mse_vjp_direct_atomic_sitecache_rgb_only(
    row_index_i32: Tensor,
    candidate_row_offsets_i32: Tensor,
    candidate_depth_coeff_f16: Tensor,
    sites_f32: Tensor,
    site_rgba_f32: Tensor,
    ray_coeff_f32: Tensor,
    frame_t_f32: Tensor,
    target_rgb_f32: Tensor,
    config: RealRayReplayConfig,
    *,
    time_slab_count: int,
    row_count: int,
) -> tuple[Tensor, Tensor]:
    """Sample-parallel coeff16 RGB MSE VJP with a per-threadgroup cached site table."""
    return fused_slab_affine_coeff16_mse_vjp_direct_atomic_rgb_only(
        row_index_i32,
        candidate_row_offsets_i32,
        candidate_depth_coeff_f16,
        sites_f32,
        site_rgba_f32,
        ray_coeff_f32,
        frame_t_f32,
        target_rgb_f32,
        config,
        time_slab_count=time_slab_count,
        row_count=row_count,
        _op_name="fused_slab_affine_coeff16_mse_vjp_direct_atomic_sitecache_rgb_only",
    )


def fused_slab_affine_coeff16_ownerupdate_mse_vjp_direct_atomic_rgb_only(
    row_index_i32: Tensor,
    candidate_row_offsets_i32: Tensor,
    candidate_boundary_ids_i32: Tensor,
    candidate_depth_coeff_f16: Tensor,
    boundary_site_pairs_i32: Tensor,
    sites_f32: Tensor,
    site_rgba_f32: Tensor,
    ray_coeff_f32: Tensor,
    frame_t_f32: Tensor,
    target_rgb_f32: Tensor,
    config: RealRayReplayConfig,
    *,
    time_slab_count: int,
    row_count: int,
    _op_name: str = "fused_slab_affine_coeff16_ownerupdate_mse_vjp_direct_atomic_rgb_only",
    _boundary_id_dtype: torch.dtype = torch.int32,
    _boundary_pair_dtype: torch.dtype = torch.int32,
) -> tuple[Tensor, Tensor]:
    """Compute RGB MSE gradients using coeff16 depths and boundary-id owner updates."""
    boundary_id_name = (
        "candidate_boundary_ids_i16" if _boundary_id_dtype == torch.int16 else "candidate_boundary_ids_i32"
    )
    boundary_pair_name = "boundary_site_pairs_i16" if _boundary_pair_dtype == torch.int16 else "boundary_site_pairs_i32"
    row_index_i32 = row_index_i32.contiguous()
    candidate_row_offsets_i32 = candidate_row_offsets_i32.contiguous()
    candidate_boundary_ids_i32 = candidate_boundary_ids_i32.contiguous()
    candidate_depth_coeff_f16 = candidate_depth_coeff_f16.contiguous()
    boundary_site_pairs_i32 = boundary_site_pairs_i32.contiguous()
    sites_f32 = sites_f32.contiguous()
    site_rgba_f32 = site_rgba_f32.contiguous()
    ray_coeff_f32 = ray_coeff_f32.contiguous()
    frame_t_f32 = frame_t_f32.contiguous()
    target_rgb_f32 = target_rgb_f32.contiguous()
    _require_mps_tensor(row_index_i32, name="row_index_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(candidate_row_offsets_i32, name="candidate_row_offsets_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(candidate_boundary_ids_i32, name=boundary_id_name, dtype=_boundary_id_dtype, cols=None)
    _require_mps_tensor(candidate_depth_coeff_f16, name="candidate_depth_coeff_f16", dtype=torch.float16, cols=4)
    _require_mps_tensor(boundary_site_pairs_i32, name=boundary_pair_name, dtype=_boundary_pair_dtype, cols=2)
    _require_mps_tensor(sites_f32, name="sites_f32", dtype=torch.float32, cols=5)
    _require_mps_tensor(site_rgba_f32, name="site_rgba_f32", dtype=torch.float32, cols=4)
    _require_mps_tensor(ray_coeff_f32, name="ray_coeff_f32", dtype=torch.float32, cols=12)
    _require_mps_tensor(frame_t_f32, name="frame_t_f32", dtype=torch.float32, cols=None)
    if target_rgb_f32.device.type != "mps" or target_rgb_f32.dtype != torch.float32:
        raise ValueError("target_rgb_f32 must be float32 on MPS")
    if target_rgb_f32.ndim != 3 or target_rgb_f32.shape[2] != 3:
        raise ValueError("target_rgb_f32 must have shape [track_count, frame_count, 3]")
    if not target_rgb_f32.is_contiguous():
        raise ValueError("target_rgb_f32 must be contiguous")
    if ray_coeff_f32.shape[0] <= 0 or frame_t_f32.shape[0] <= 0:
        raise ValueError("ray_coeff_f32 and frame_t_f32 must be nonempty")
    if sites_f32.shape[0] <= 0 or sites_f32.shape[0] > 64:
        raise ValueError("sites_f32 row count must be in [1, 64]")
    if site_rgba_f32.shape[0] != sites_f32.shape[0]:
        raise ValueError("site_rgba_f32 row count must match sites_f32")
    if row_index_i32.shape[0] != ray_coeff_f32.shape[0]:
        raise ValueError("row_index_i32 length must match track count")
    if candidate_boundary_ids_i32.shape[0] != candidate_depth_coeff_f16.shape[0]:
        raise ValueError(f"{boundary_id_name} length must match candidate_depth_coeff_f16 rows")
    if time_slab_count <= 0:
        raise ValueError("time_slab_count must be positive")
    if row_count <= 0:
        raise ValueError("row_count must be positive")
    if candidate_row_offsets_i32.shape[0] != row_count * time_slab_count + 1:
        raise ValueError("candidate_row_offsets_i32 length must be row_count * time_slab_count + 1")
    row_index_cpu = row_index_i32.detach().cpu()
    row_offsets_cpu = candidate_row_offsets_i32.detach().cpu()
    candidate_ids_cpu = candidate_boundary_ids_i32.detach().cpu()
    boundary_pairs_cpu = boundary_site_pairs_i32.detach().cpu()
    if int(row_index_cpu.min().item()) < 0 or int(row_index_cpu.max().item()) >= row_count:
        raise ValueError("row_index_i32 values must be in [0, row_count)")
    _validate_csr_offsets_cpu(
        row_offsets_cpu,
        candidate_count=candidate_depth_coeff_f16.shape[0],
        max_boundaries=MAX_REALRAY_FUSED_MSE_BOUNDARIES,
    )
    if candidate_ids_cpu.numel() and (
        int(candidate_ids_cpu.min().item()) < 0
        or int(candidate_ids_cpu.max().item()) >= boundary_site_pairs_i32.shape[0]
    ):
        raise ValueError(f"{boundary_id_name} values must be in [0, boundary_count)")
    if boundary_pairs_cpu.numel() and (
        int(boundary_pairs_cpu.min().item()) < 0 or int(boundary_pairs_cpu.max().item()) >= sites_f32.shape[0]
    ):
        raise ValueError(f"{boundary_pair_name} values must be in [0, site_count)")
    expected_shape = (ray_coeff_f32.shape[0], frame_t_f32.shape[0])
    if target_rgb_f32.shape[:2] != expected_shape:
        raise ValueError("target_rgb_f32 shape must be [track_count, frame_count, 3]")
    if config.far <= config.near:
        raise ValueError("config.far must be greater than config.near")
    if config.invalid_epsilon <= 0.0:
        raise ValueError("config.invalid_epsilon must be positive")
    if config.transmittance_threshold < 0.0:
        raise ValueError("config.transmittance_threshold must be nonnegative")

    ops = torch.ops.world_foam_lane2_fused_slab_v0
    if not hasattr(ops, _op_name):
        raise RuntimeError(f"world_foam_lane2_fused_slab_v0 {_op_name} op not found. Build this variant first.")
    config_i32 = torch.tensor(
        [
            0,
            ray_coeff_f32.shape[0],
            sites_f32.shape[0],
            frame_t_f32.shape[0],
            time_slab_count,
            row_count,
            candidate_depth_coeff_f16.shape[0],
            boundary_site_pairs_i32.shape[0],
        ],
        device=row_index_i32.device,
        dtype=torch.int32,
    )
    config_f32 = torch.tensor(
        [config.near, config.far, config.invalid_epsilon, config.transmittance_threshold],
        device=row_index_i32.device,
        dtype=torch.float32,
    )
    return getattr(ops, _op_name)(
        row_index_i32,
        candidate_row_offsets_i32,
        candidate_boundary_ids_i32,
        candidate_depth_coeff_f16,
        boundary_site_pairs_i32,
        sites_f32,
        site_rgba_f32,
        ray_coeff_f32,
        frame_t_f32,
        target_rgb_f32,
        config_i32,
        config_f32,
    )


def fused_slab_affine_coeff16_ownerkeep_mse_vjp_direct_atomic_rgb_only(
    row_index_i32: Tensor,
    candidate_row_offsets_i32: Tensor,
    candidate_boundary_ids_i32: Tensor,
    candidate_depth_coeff_f16: Tensor,
    boundary_site_pairs_i32: Tensor,
    sites_f32: Tensor,
    site_rgba_f32: Tensor,
    ray_coeff_f32: Tensor,
    frame_t_f32: Tensor,
    target_rgb_f32: Tensor,
    config: RealRayReplayConfig,
    *,
    time_slab_count: int,
    row_count: int,
) -> tuple[Tensor, Tensor]:
    """Owner-update coeff16 MSE/VJP that keeps the current owner across unrelated candidate boundaries."""
    return fused_slab_affine_coeff16_ownerupdate_mse_vjp_direct_atomic_rgb_only(
        row_index_i32,
        candidate_row_offsets_i32,
        candidate_boundary_ids_i32,
        candidate_depth_coeff_f16,
        boundary_site_pairs_i32,
        sites_f32,
        site_rgba_f32,
        ray_coeff_f32,
        frame_t_f32,
        target_rgb_f32,
        config,
        time_slab_count=time_slab_count,
        row_count=row_count,
        _op_name="fused_slab_affine_coeff16_ownerkeep_mse_vjp_direct_atomic_rgb_only",
    )


def fused_slab_affine_coeff16_ownerupdate_i16_mse_vjp_direct_atomic_rgb_only(
    row_index_i32: Tensor,
    candidate_row_offsets_i32: Tensor,
    candidate_boundary_ids_i16: Tensor,
    candidate_depth_coeff_f16: Tensor,
    boundary_site_pairs_i16: Tensor,
    sites_f32: Tensor,
    site_rgba_f32: Tensor,
    ray_coeff_f32: Tensor,
    frame_t_f32: Tensor,
    target_rgb_f32: Tensor,
    config: RealRayReplayConfig,
    *,
    time_slab_count: int,
    row_count: int,
) -> tuple[Tensor, Tensor]:
    """Owner-update coeff16 MSE/VJP using packed int16 boundary ids and site pairs."""
    return fused_slab_affine_coeff16_ownerupdate_mse_vjp_direct_atomic_rgb_only(
        row_index_i32,
        candidate_row_offsets_i32,
        candidate_boundary_ids_i16,
        candidate_depth_coeff_f16,
        boundary_site_pairs_i16,
        sites_f32,
        site_rgba_f32,
        ray_coeff_f32,
        frame_t_f32,
        target_rgb_f32,
        config,
        time_slab_count=time_slab_count,
        row_count=row_count,
        _op_name="fused_slab_affine_coeff16_ownerupdate_i16_mse_vjp_direct_atomic_rgb_only",
        _boundary_id_dtype=torch.int16,
        _boundary_pair_dtype=torch.int16,
    )


def fused_slab_affine_coeff16_ownerkeep_i16_mse_vjp_direct_atomic_rgb_only(
    row_index_i32: Tensor,
    candidate_row_offsets_i32: Tensor,
    candidate_boundary_ids_i16: Tensor,
    candidate_depth_coeff_f16: Tensor,
    boundary_site_pairs_i16: Tensor,
    sites_f32: Tensor,
    site_rgba_f32: Tensor,
    ray_coeff_f32: Tensor,
    frame_t_f32: Tensor,
    target_rgb_f32: Tensor,
    config: RealRayReplayConfig,
    *,
    time_slab_count: int,
    row_count: int,
) -> tuple[Tensor, Tensor]:
    """Owner-keep coeff16 MSE/VJP using packed int16 boundary ids and site pairs."""
    return fused_slab_affine_coeff16_ownerupdate_mse_vjp_direct_atomic_rgb_only(
        row_index_i32,
        candidate_row_offsets_i32,
        candidate_boundary_ids_i16,
        candidate_depth_coeff_f16,
        boundary_site_pairs_i16,
        sites_f32,
        site_rgba_f32,
        ray_coeff_f32,
        frame_t_f32,
        target_rgb_f32,
        config,
        time_slab_count=time_slab_count,
        row_count=row_count,
        _op_name="fused_slab_affine_coeff16_ownerkeep_i16_mse_vjp_direct_atomic_rgb_only",
        _boundary_id_dtype=torch.int16,
        _boundary_pair_dtype=torch.int16,
    )


def fused_slab_affine_coeff16_mse_vjp_direct_atomic_sample_reduce_rgb_only(
    row_index_i32: Tensor,
    candidate_row_offsets_i32: Tensor,
    candidate_depth_coeff_f16: Tensor,
    sites_f32: Tensor,
    site_rgba_f32: Tensor,
    ray_coeff_f32: Tensor,
    frame_t_f32: Tensor,
    target_rgb_f32: Tensor,
    config: RealRayReplayConfig,
    *,
    time_slab_count: int,
    row_count: int,
) -> tuple[Tensor, Tensor]:
    """Sample-parallel coeff16 MSE/VJP with local per-sample site-gradient reduction."""
    return fused_slab_affine_coeff16_mse_vjp_direct_atomic_rgb_only(
        row_index_i32,
        candidate_row_offsets_i32,
        candidate_depth_coeff_f16,
        sites_f32,
        site_rgba_f32,
        ray_coeff_f32,
        frame_t_f32,
        target_rgb_f32,
        config,
        time_slab_count=time_slab_count,
        row_count=row_count,
        _op_name="fused_slab_affine_coeff16_mse_vjp_direct_atomic_sample_reduce_rgb_only",
    )


def fused_slab_affine_coeff16_sortnet_mse_vjp_direct_atomic_rgb_only(
    row_index_i32: Tensor,
    candidate_row_offsets_i32: Tensor,
    candidate_depth_coeff_f16: Tensor,
    sites_f32: Tensor,
    site_rgba_f32: Tensor,
    ray_coeff_f32: Tensor,
    frame_t_f32: Tensor,
    target_rgb_f32: Tensor,
    config: RealRayReplayConfig,
    *,
    time_slab_count: int,
    row_count: int,
) -> tuple[Tensor, Tensor]:
    """Sample-parallel coeff16 MSE/VJP that collect-sorts depths with an in-thread bitonic network."""
    return fused_slab_affine_coeff16_mse_vjp_direct_atomic_rgb_only(
        row_index_i32,
        candidate_row_offsets_i32,
        candidate_depth_coeff_f16,
        sites_f32,
        site_rgba_f32,
        ray_coeff_f32,
        frame_t_f32,
        target_rgb_f32,
        config,
        time_slab_count=time_slab_count,
        row_count=row_count,
        _op_name="fused_slab_affine_coeff16_sortnet_mse_vjp_direct_atomic_rgb_only",
    )


def fused_slab_affine_coeff16_mse_vjp_direct_atomic_track_rgb_only(
    row_index_i32: Tensor,
    candidate_row_offsets_i32: Tensor,
    candidate_depth_coeff_f16: Tensor,
    sites_f32: Tensor,
    site_rgba_f32: Tensor,
    ray_coeff_f32: Tensor,
    frame_t_f32: Tensor,
    target_rgb_f32: Tensor,
    config: RealRayReplayConfig,
    *,
    time_slab_count: int,
    row_count: int,
) -> tuple[Tensor, Tensor]:
    """Track-thread RGB MSE and site-RGBA gradients using fully half affine depth coefficients."""
    return fused_slab_affine_coeff16_mse_vjp_direct_atomic_rgb_only(
        row_index_i32,
        candidate_row_offsets_i32,
        candidate_depth_coeff_f16,
        sites_f32,
        site_rgba_f32,
        ray_coeff_f32,
        frame_t_f32,
        target_rgb_f32,
        config,
        time_slab_count=time_slab_count,
        row_count=row_count,
        _op_name="fused_slab_affine_coeff16_mse_vjp_direct_atomic_track_rgb_only",
    )


def fused_slab_affine_num32_den16_vjp_direct_atomic_track(
    row_index_i32: Tensor,
    candidate_row_offsets_i32: Tensor,
    candidate_depth_num_f32: Tensor,
    candidate_depth_den_f16: Tensor,
    sites_f32: Tensor,
    site_rgba_f32: Tensor,
    ray_coeff_f32: Tensor,
    frame_t_f32: Tensor,
    grad_rgb_f32: Tensor,
    grad_alpha_f32: Tensor,
    grad_depth_f32: Tensor,
    config: RealRayReplayConfig,
    *,
    time_slab_count: int,
    row_count: int,
) -> Tensor:
    """Accumulate mixed affine site RGBA gradients locally per track, then atomically write sites."""
    row_index_i32 = row_index_i32.contiguous()
    candidate_row_offsets_i32 = candidate_row_offsets_i32.contiguous()
    candidate_depth_num_f32 = candidate_depth_num_f32.contiguous()
    candidate_depth_den_f16 = candidate_depth_den_f16.contiguous()
    sites_f32 = sites_f32.contiguous()
    site_rgba_f32 = site_rgba_f32.contiguous()
    ray_coeff_f32 = ray_coeff_f32.contiguous()
    frame_t_f32 = frame_t_f32.contiguous()
    grad_rgb_f32 = grad_rgb_f32.contiguous()
    grad_alpha_f32 = grad_alpha_f32.contiguous()
    grad_depth_f32 = grad_depth_f32.contiguous()
    _require_mps_tensor(row_index_i32, name="row_index_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(candidate_row_offsets_i32, name="candidate_row_offsets_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(candidate_depth_num_f32, name="candidate_depth_num_f32", dtype=torch.float32, cols=2)
    _require_mps_tensor(candidate_depth_den_f16, name="candidate_depth_den_f16", dtype=torch.float16, cols=2)
    _require_mps_tensor(sites_f32, name="sites_f32", dtype=torch.float32, cols=5)
    _require_mps_tensor(site_rgba_f32, name="site_rgba_f32", dtype=torch.float32, cols=4)
    _require_mps_tensor(ray_coeff_f32, name="ray_coeff_f32", dtype=torch.float32, cols=12)
    _require_mps_tensor(frame_t_f32, name="frame_t_f32", dtype=torch.float32, cols=None)
    if grad_rgb_f32.device.type != "mps" or grad_rgb_f32.dtype != torch.float32:
        raise ValueError("grad_rgb_f32 must be float32 on MPS")
    if grad_rgb_f32.ndim != 3 or grad_rgb_f32.shape[2] != 3:
        raise ValueError("grad_rgb_f32 must have shape [track_count, frame_count, 3]")
    if grad_alpha_f32.device.type != "mps" or grad_alpha_f32.dtype != torch.float32:
        raise ValueError("grad_alpha_f32 must be float32 on MPS")
    if grad_alpha_f32.ndim != 2:
        raise ValueError("grad_alpha_f32 must have shape [track_count, frame_count]")
    if grad_depth_f32.device.type != "mps" or grad_depth_f32.dtype != torch.float32:
        raise ValueError("grad_depth_f32 must be float32 on MPS")
    if grad_depth_f32.ndim != 2:
        raise ValueError("grad_depth_f32 must have shape [track_count, frame_count]")
    if ray_coeff_f32.shape[0] <= 0 or frame_t_f32.shape[0] <= 0:
        raise ValueError("ray_coeff_f32 and frame_t_f32 must be nonempty")
    if sites_f32.shape[0] <= 0 or sites_f32.shape[0] > 64:
        raise ValueError("sites_f32 row count must be in [1, 64]")
    if site_rgba_f32.shape[0] != sites_f32.shape[0]:
        raise ValueError("site_rgba_f32 row count must match sites_f32")
    if row_index_i32.shape[0] != ray_coeff_f32.shape[0]:
        raise ValueError("row_index_i32 length must match track count")
    if candidate_depth_den_f16.shape[0] != candidate_depth_num_f32.shape[0]:
        raise ValueError("candidate numerator and denominator row counts must match")
    if time_slab_count <= 0:
        raise ValueError("time_slab_count must be positive")
    if row_count <= 0:
        raise ValueError("row_count must be positive")
    if candidate_row_offsets_i32.shape[0] != row_count * time_slab_count + 1:
        raise ValueError("candidate_row_offsets_i32 length must be row_count * time_slab_count + 1")
    row_index_cpu = row_index_i32.detach().cpu()
    row_offsets_cpu = candidate_row_offsets_i32.detach().cpu()
    if int(row_index_cpu.min().item()) < 0 or int(row_index_cpu.max().item()) >= row_count:
        raise ValueError("row_index_i32 values must be in [0, row_count)")
    _validate_csr_offsets_cpu(row_offsets_cpu, candidate_count=candidate_depth_num_f32.shape[0])
    expected_shape = (ray_coeff_f32.shape[0], frame_t_f32.shape[0])
    if grad_rgb_f32.shape[:2] != expected_shape:
        raise ValueError("grad_rgb_f32 shape must be [track_count, frame_count, 3]")
    if grad_alpha_f32.shape != expected_shape or grad_depth_f32.shape != expected_shape:
        raise ValueError("grad_alpha_f32 and grad_depth_f32 must have shape [track_count, frame_count]")
    if config.far <= config.near:
        raise ValueError("config.far must be greater than config.near")
    if config.invalid_epsilon <= 0.0:
        raise ValueError("config.invalid_epsilon must be positive")
    if config.transmittance_threshold < 0.0:
        raise ValueError("config.transmittance_threshold must be nonnegative")
    ops = torch.ops.world_foam_lane2_fused_slab_v0
    op_name = "fused_slab_affine_num32_den16_vjp_direct_atomic_track"
    if not hasattr(ops, op_name):
        raise RuntimeError(f"world_foam_lane2_fused_slab_v0 {op_name} op not found. Build this variant first.")
    config_i32 = torch.tensor(
        [
            0,
            ray_coeff_f32.shape[0],
            sites_f32.shape[0],
            frame_t_f32.shape[0],
            time_slab_count,
            row_count,
            candidate_depth_num_f32.shape[0],
            1,
        ],
        device=row_index_i32.device,
        dtype=torch.int32,
    )
    config_f32 = torch.tensor(
        [config.near, config.far, config.invalid_epsilon, config.transmittance_threshold],
        device=row_index_i32.device,
        dtype=torch.float32,
    )
    return getattr(ops, op_name)(
        row_index_i32,
        candidate_row_offsets_i32,
        candidate_depth_num_f32,
        candidate_depth_den_f16,
        sites_f32,
        site_rgba_f32,
        ray_coeff_f32,
        frame_t_f32,
        grad_rgb_f32,
        grad_alpha_f32,
        grad_depth_f32,
        config_i32,
        config_f32,
    )


class _FusedSlabAffineNum32Den16ReplayFunction(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx: torch.autograd.function.FunctionCtx,
        row_index_i32: Tensor,
        candidate_row_offsets_i32: Tensor,
        candidate_depth_num_f32: Tensor,
        candidate_depth_den_f16: Tensor,
        sites_f32: Tensor,
        site_rgba_f32: Tensor,
        ray_coeff_f32: Tensor,
        frame_t_f32: Tensor,
        time_slab_count: int,
        row_count: int,
        near: float,
        far: float,
        invalid_epsilon: float,
        transmittance_threshold: float,
        reduce_chunk_size: int,
        use_direct_atomic: bool,
        direct_atomic_grad_only: bool,
        direct_atomic_rgb_only: bool,
        direct_atomic_track: bool,
    ) -> tuple[Tensor, Tensor, Tensor]:
        config = RealRayReplayConfig(
            near=float(near),
            far=float(far),
            invalid_epsilon=float(invalid_epsilon),
            transmittance_threshold=float(transmittance_threshold),
        )
        output_rgb, output_alpha, output_depth = fused_slab_affine_num32_den16_realray_rgba_depth_replay(
            row_index_i32,
            candidate_row_offsets_i32,
            candidate_depth_num_f32,
            candidate_depth_den_f16,
            sites_f32,
            site_rgba_f32,
            ray_coeff_f32,
            frame_t_f32,
            config,
            time_slab_count=int(time_slab_count),
            row_count=int(row_count),
        )
        ctx.save_for_backward(
            row_index_i32,
            candidate_row_offsets_i32,
            candidate_depth_num_f32,
            candidate_depth_den_f16,
            sites_f32,
            site_rgba_f32,
            ray_coeff_f32,
            frame_t_f32,
        )
        ctx.config = config
        ctx.time_slab_count = int(time_slab_count)
        ctx.row_count = int(row_count)
        ctx.reduce_chunk_size = int(reduce_chunk_size)
        ctx.use_direct_atomic = bool(use_direct_atomic)
        ctx.direct_atomic_grad_only = bool(direct_atomic_grad_only)
        ctx.direct_atomic_rgb_only = bool(direct_atomic_rgb_only)
        ctx.direct_atomic_track = bool(direct_atomic_track)
        return output_rgb, output_alpha, output_depth

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: torch.autograd.function.FunctionCtx,
        grad_rgb: Tensor | None,
        grad_alpha: Tensor | None,
        grad_depth: Tensor | None,
    ) -> tuple[
        None,
        None,
        None,
        None,
        None,
        Tensor,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    ]:
        (
            row_index_i32,
            candidate_row_offsets_i32,
            candidate_depth_num_f32,
            candidate_depth_den_f16,
            sites_f32,
            site_rgba_f32,
            ray_coeff_f32,
            frame_t_f32,
        ) = ctx.saved_tensors
        track_count = ray_coeff_f32.shape[0]
        frame_count = frame_t_f32.shape[0]
        grad_alpha_is_none = grad_alpha is None
        grad_depth_is_none = grad_depth is None
        if grad_rgb is None:
            grad_rgb = torch.zeros((track_count, frame_count, 3), device=site_rgba_f32.device, dtype=torch.float32)
        if bool(ctx.direct_atomic_rgb_only) and grad_alpha_is_none and grad_depth_is_none:
            grad_site_rgba = fused_slab_affine_num32_den16_vjp_direct_atomic_rgb_only(
                row_index_i32,
                candidate_row_offsets_i32,
                candidate_depth_num_f32,
                candidate_depth_den_f16,
                sites_f32,
                site_rgba_f32,
                ray_coeff_f32,
                frame_t_f32,
                grad_rgb,
                ctx.config,
                time_slab_count=ctx.time_slab_count,
                row_count=ctx.row_count,
            )
            return (
                None,
                None,
                None,
                None,
                None,
                grad_site_rgba,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            )
        if grad_alpha is None:
            grad_alpha = torch.zeros((track_count, frame_count), device=site_rgba_f32.device, dtype=torch.float32)
        if grad_depth is None:
            grad_depth = torch.zeros((track_count, frame_count), device=site_rgba_f32.device, dtype=torch.float32)
        if bool(ctx.direct_atomic_track):
            grad_site_rgba = fused_slab_affine_num32_den16_vjp_direct_atomic_track(
                row_index_i32,
                candidate_row_offsets_i32,
                candidate_depth_num_f32,
                candidate_depth_den_f16,
                sites_f32,
                site_rgba_f32,
                ray_coeff_f32,
                frame_t_f32,
                grad_rgb,
                grad_alpha,
                grad_depth,
                ctx.config,
                time_slab_count=ctx.time_slab_count,
                row_count=ctx.row_count,
            )
        elif bool(ctx.direct_atomic_grad_only):
            grad_site_rgba = fused_slab_affine_num32_den16_vjp_direct_atomic_grad_only(
                row_index_i32,
                candidate_row_offsets_i32,
                candidate_depth_num_f32,
                candidate_depth_den_f16,
                sites_f32,
                site_rgba_f32,
                ray_coeff_f32,
                frame_t_f32,
                grad_rgb,
                grad_alpha,
                grad_depth,
                ctx.config,
                time_slab_count=ctx.time_slab_count,
                row_count=ctx.row_count,
            )
        elif bool(ctx.use_direct_atomic):
            _, _, _, grad_site_rgba = fused_slab_affine_num32_den16_vjp_direct_atomic(
                row_index_i32,
                candidate_row_offsets_i32,
                candidate_depth_num_f32,
                candidate_depth_den_f16,
                sites_f32,
                site_rgba_f32,
                ray_coeff_f32,
                frame_t_f32,
                grad_rgb,
                grad_alpha,
                grad_depth,
                ctx.config,
                time_slab_count=ctx.time_slab_count,
                row_count=ctx.row_count,
            )
        else:
            _, _, _, grad_site_rgba = fused_slab_affine_num32_den16_vjp_reduce(
                row_index_i32,
                candidate_row_offsets_i32,
                candidate_depth_num_f32,
                candidate_depth_den_f16,
                sites_f32,
                site_rgba_f32,
                ray_coeff_f32,
                frame_t_f32,
                grad_rgb,
                grad_alpha,
                grad_depth,
                ctx.config,
                time_slab_count=ctx.time_slab_count,
                row_count=ctx.row_count,
                reduce_chunk_size=ctx.reduce_chunk_size,
            )
        return (
            None,
            None,
            None,
            None,
            None,
            grad_site_rgba,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class _FusedSlabAffineNum32Den16OwnerUpdateReplayFunction(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx: torch.autograd.function.FunctionCtx,
        row_index_i32: Tensor,
        candidate_row_offsets_i32: Tensor,
        candidate_boundary_ids_i32: Tensor,
        candidate_depth_num_f32: Tensor,
        candidate_depth_den_f16: Tensor,
        boundary_site_pairs_i32: Tensor,
        sites_f32: Tensor,
        site_rgba_f32: Tensor,
        ray_coeff_f32: Tensor,
        frame_t_f32: Tensor,
        time_slab_count: int,
        row_count: int,
        near: float,
        far: float,
        invalid_epsilon: float,
        transmittance_threshold: float,
    ) -> tuple[Tensor, Tensor, Tensor]:
        config = RealRayReplayConfig(
            near=float(near),
            far=float(far),
            invalid_epsilon=float(invalid_epsilon),
            transmittance_threshold=float(transmittance_threshold),
        )
        output_rgb, output_alpha, output_depth = fused_slab_affine_num32_den16_ownerupdate_realray_rgba_depth_replay(
            row_index_i32,
            candidate_row_offsets_i32,
            candidate_boundary_ids_i32,
            candidate_depth_num_f32,
            candidate_depth_den_f16,
            boundary_site_pairs_i32,
            sites_f32,
            site_rgba_f32,
            ray_coeff_f32,
            frame_t_f32,
            config,
            time_slab_count=int(time_slab_count),
            row_count=int(row_count),
        )
        ctx.save_for_backward(
            row_index_i32,
            candidate_row_offsets_i32,
            candidate_boundary_ids_i32,
            candidate_depth_num_f32,
            candidate_depth_den_f16,
            boundary_site_pairs_i32,
            sites_f32,
            site_rgba_f32,
            ray_coeff_f32,
            frame_t_f32,
        )
        ctx.config = config
        ctx.time_slab_count = int(time_slab_count)
        ctx.row_count = int(row_count)
        return output_rgb, output_alpha, output_depth

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: torch.autograd.function.FunctionCtx,
        grad_rgb: Tensor | None,
        grad_alpha: Tensor | None,
        grad_depth: Tensor | None,
    ) -> tuple[None, None, None, None, None, None, None, Tensor, None, None, None, None, None, None, None, None]:
        (
            row_index_i32,
            candidate_row_offsets_i32,
            candidate_boundary_ids_i32,
            candidate_depth_num_f32,
            candidate_depth_den_f16,
            boundary_site_pairs_i32,
            sites_f32,
            site_rgba_f32,
            ray_coeff_f32,
            frame_t_f32,
        ) = ctx.saved_tensors
        track_count = ray_coeff_f32.shape[0]
        frame_count = frame_t_f32.shape[0]
        if grad_rgb is None:
            grad_rgb = torch.zeros((track_count, frame_count, 3), device=site_rgba_f32.device, dtype=torch.float32)
        if grad_alpha is None:
            grad_alpha = torch.zeros((track_count, frame_count), device=site_rgba_f32.device, dtype=torch.float32)
        if grad_depth is None:
            grad_depth = torch.zeros((track_count, frame_count), device=site_rgba_f32.device, dtype=torch.float32)
        grad_site_rgba = fused_slab_affine_num32_den16_vjp_direct_atomic_grad_only_ownerupdate(
            row_index_i32,
            candidate_row_offsets_i32,
            candidate_boundary_ids_i32,
            candidate_depth_num_f32,
            candidate_depth_den_f16,
            boundary_site_pairs_i32,
            sites_f32,
            site_rgba_f32,
            ray_coeff_f32,
            frame_t_f32,
            grad_rgb,
            grad_alpha,
            grad_depth,
            ctx.config,
            time_slab_count=ctx.time_slab_count,
            row_count=ctx.row_count,
        )
        return None, None, None, None, None, None, None, grad_site_rgba, None, None, None, None, None, None, None, None


def fused_slab_affine_num32_den16_autograd(
    row_index_i32: Tensor,
    candidate_row_offsets_i32: Tensor,
    candidate_depth_num_f32: Tensor,
    candidate_depth_den_f16: Tensor,
    sites_f32: Tensor,
    site_rgba_f32: Tensor,
    ray_coeff_f32: Tensor,
    frame_t_f32: Tensor,
    config: RealRayReplayConfig,
    *,
    time_slab_count: int,
    row_count: int,
    reduce_chunk_size: int = 4,
    vjp_mode: str = "reduce",
) -> tuple[Tensor, Tensor, Tensor]:
    """Render mixed affine CSR candidates with frozen-geometry site-RGBA autograd."""
    if vjp_mode not in {
        "reduce",
        "direct_atomic",
        "direct_atomic_grad_only",
        "direct_atomic_rgb_only",
        "direct_atomic_track",
    }:
        raise ValueError(
            "vjp_mode must be 'reduce', 'direct_atomic', 'direct_atomic_grad_only', "
            "'direct_atomic_rgb_only', or 'direct_atomic_track'"
        )
    return _FusedSlabAffineNum32Den16ReplayFunction.apply(
        row_index_i32,
        candidate_row_offsets_i32,
        candidate_depth_num_f32,
        candidate_depth_den_f16,
        sites_f32,
        site_rgba_f32,
        ray_coeff_f32,
        frame_t_f32,
        int(time_slab_count),
        int(row_count),
        config.near,
        config.far,
        config.invalid_epsilon,
        config.transmittance_threshold,
        int(reduce_chunk_size),
        vjp_mode in {"direct_atomic", "direct_atomic_grad_only", "direct_atomic_rgb_only", "direct_atomic_track"},
        vjp_mode in {"direct_atomic_grad_only", "direct_atomic_rgb_only", "direct_atomic_track"},
        vjp_mode == "direct_atomic_rgb_only",
        vjp_mode == "direct_atomic_track",
    )


def fused_slab_affine_num32_den16_ownerupdate_autograd(
    row_index_i32: Tensor,
    candidate_row_offsets_i32: Tensor,
    candidate_boundary_ids_i32: Tensor,
    candidate_depth_num_f32: Tensor,
    candidate_depth_den_f16: Tensor,
    boundary_site_pairs_i32: Tensor,
    sites_f32: Tensor,
    site_rgba_f32: Tensor,
    ray_coeff_f32: Tensor,
    frame_t_f32: Tensor,
    config: RealRayReplayConfig,
    *,
    time_slab_count: int,
    row_count: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Render affine CSR candidates with owner-update forward and frozen-geometry site-RGBA autograd."""
    return _FusedSlabAffineNum32Den16OwnerUpdateReplayFunction.apply(
        row_index_i32,
        candidate_row_offsets_i32,
        candidate_boundary_ids_i32,
        candidate_depth_num_f32,
        candidate_depth_den_f16,
        boundary_site_pairs_i32,
        sites_f32,
        site_rgba_f32,
        ray_coeff_f32,
        frame_t_f32,
        int(time_slab_count),
        int(row_count),
        config.near,
        config.far,
        config.invalid_epsilon,
        config.transmittance_threshold,
    )


def fused_slab_affine_realray_rgba_depth_replay(
    boundary_f32: Tensor,
    row_index_i32: Tensor,
    candidate_row_offsets_i32: Tensor,
    candidate_boundary_ids_i32: Tensor,
    sites_f32: Tensor,
    site_rgba_f32: Tensor,
    ray_coeff_f32: Tensor,
    frame_t_f32: Tensor,
    config: RealRayReplayConfig,
    *,
    time_slab_count: int,
    row_count: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Render affine moving-ray tracks from tiled/slab CSR candidate rows.

    `ray_coeff_f32 [K,12]` stores `(origin_base xyz, origin_slope xyz,
    direction_base xyz, direction_slope xyz)`. The Metal shader evaluates the
    ray at each `frame_t_f32[T]`, maps the track through `row_index_i32[K]`, and
    uses CSR row `row_index * time_slab_count + slab_id` for replay.
    """
    boundary_f32 = boundary_f32.contiguous()
    row_index_i32 = row_index_i32.contiguous()
    candidate_row_offsets_i32 = candidate_row_offsets_i32.contiguous()
    candidate_boundary_ids_i32 = candidate_boundary_ids_i32.contiguous()
    sites_f32 = sites_f32.contiguous()
    site_rgba_f32 = site_rgba_f32.contiguous()
    ray_coeff_f32 = ray_coeff_f32.contiguous()
    frame_t_f32 = frame_t_f32.contiguous()
    _require_mps_tensor(boundary_f32, name="boundary_f32", dtype=torch.float32, cols=5)
    _require_mps_tensor(row_index_i32, name="row_index_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(candidate_row_offsets_i32, name="candidate_row_offsets_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(candidate_boundary_ids_i32, name="candidate_boundary_ids_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(sites_f32, name="sites_f32", dtype=torch.float32, cols=5)
    _require_mps_tensor(site_rgba_f32, name="site_rgba_f32", dtype=torch.float32, cols=4)
    _require_mps_tensor(ray_coeff_f32, name="ray_coeff_f32", dtype=torch.float32, cols=12)
    _require_mps_tensor(frame_t_f32, name="frame_t_f32", dtype=torch.float32, cols=None)
    if ray_coeff_f32.shape[0] <= 0:
        raise ValueError("ray_coeff_f32 must contain at least one track")
    if frame_t_f32.shape[0] <= 0:
        raise ValueError("frame_t_f32 must contain at least one frame")
    if sites_f32.shape[0] <= 0:
        raise ValueError("sites_f32 must contain at least one site")
    if boundary_f32.shape[0] > 128:
        raise ValueError("fused_slab_affine_realray_rgba_depth_replay currently supports at most 128 boundaries")
    if sites_f32.shape[0] > 64:
        raise ValueError("fused_slab_affine_realray_rgba_depth_replay currently supports at most 64 sites")
    if site_rgba_f32.shape[0] != sites_f32.shape[0]:
        raise ValueError("site_rgba_f32 row count must match sites_f32 rows")
    if row_index_i32.shape[0] != ray_coeff_f32.shape[0]:
        raise ValueError("row_index_i32 length must match track count")
    if time_slab_count <= 0:
        raise ValueError("time_slab_count must be positive")
    if row_count <= 0:
        raise ValueError("row_count must be positive")
    expected_offsets = row_count * time_slab_count + 1
    if candidate_row_offsets_i32.shape[0] != expected_offsets:
        raise ValueError("candidate_row_offsets_i32 length must be row_count * time_slab_count + 1")
    _validate_csr_values(
        row_index_i32=row_index_i32,
        candidate_row_offsets_i32=candidate_row_offsets_i32,
        candidate_boundary_ids_i32=candidate_boundary_ids_i32,
        row_count=row_count,
        candidate_count=candidate_boundary_ids_i32.shape[0],
        boundary_count=boundary_f32.shape[0],
    )
    if config.far <= config.near:
        raise ValueError("config.far must be greater than config.near")
    if config.invalid_epsilon <= 0.0:
        raise ValueError("config.invalid_epsilon must be positive")
    if config.transmittance_threshold < 0.0:
        raise ValueError("config.transmittance_threshold must be nonnegative")

    ops = torch.ops.world_foam_lane2_fused_slab_v0
    if not hasattr(ops, "fused_slab_affine_realray_rgba_depth_replay"):
        raise RuntimeError(
            "world_foam_lane2_fused_slab_v0 fused_slab_affine_realray_rgba_depth_replay op not found. "
            "Build this variant first."
        )
    config_i32 = torch.tensor(
        [
            boundary_f32.shape[0],
            ray_coeff_f32.shape[0],
            sites_f32.shape[0],
            frame_t_f32.shape[0],
            time_slab_count,
            row_count,
            candidate_boundary_ids_i32.shape[0],
        ],
        device=boundary_f32.device,
        dtype=torch.int32,
    )
    config_f32 = torch.tensor(
        [config.near, config.far, config.invalid_epsilon, config.transmittance_threshold],
        device=boundary_f32.device,
        dtype=torch.float32,
    )
    return ops.fused_slab_affine_realray_rgba_depth_replay(
        boundary_f32,
        row_index_i32,
        candidate_row_offsets_i32,
        candidate_boundary_ids_i32,
        sites_f32,
        site_rgba_f32,
        ray_coeff_f32,
        frame_t_f32,
        config_i32,
        config_f32,
    )


def fused_slab_affine_coeff_realray_rgba_depth_replay(
    row_index_i32: Tensor,
    candidate_row_offsets_i32: Tensor,
    candidate_depth_coeff_f32: Tensor,
    sites_f32: Tensor,
    site_rgba_f32: Tensor,
    ray_coeff_f32: Tensor,
    frame_t_f32: Tensor,
    config: RealRayReplayConfig,
    *,
    time_slab_count: int,
    row_count: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Render per-track affine CSR candidates using precomputed depth coefficients.

    `candidate_depth_coeff_f32 [C,4]` stores `(numer_base, numer_slope,
    denom_base, denom_slope)` for the same CSR cursor order as
    `candidate_row_offsets_i32`. These coefficients are track-specific, so this
    path is intended for per-track rows where `row_index_i32[track] == track`.
    """
    row_index_i32 = row_index_i32.contiguous()
    candidate_row_offsets_i32 = candidate_row_offsets_i32.contiguous()
    candidate_depth_coeff_f32 = candidate_depth_coeff_f32.contiguous()
    sites_f32 = sites_f32.contiguous()
    site_rgba_f32 = site_rgba_f32.contiguous()
    ray_coeff_f32 = ray_coeff_f32.contiguous()
    frame_t_f32 = frame_t_f32.contiguous()
    _require_mps_tensor(row_index_i32, name="row_index_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(candidate_row_offsets_i32, name="candidate_row_offsets_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(candidate_depth_coeff_f32, name="candidate_depth_coeff_f32", dtype=torch.float32, cols=4)
    _require_mps_tensor(sites_f32, name="sites_f32", dtype=torch.float32, cols=5)
    _require_mps_tensor(site_rgba_f32, name="site_rgba_f32", dtype=torch.float32, cols=4)
    _require_mps_tensor(ray_coeff_f32, name="ray_coeff_f32", dtype=torch.float32, cols=12)
    _require_mps_tensor(frame_t_f32, name="frame_t_f32", dtype=torch.float32, cols=None)
    if ray_coeff_f32.shape[0] <= 0:
        raise ValueError("ray_coeff_f32 must contain at least one track")
    if frame_t_f32.shape[0] <= 0:
        raise ValueError("frame_t_f32 must contain at least one frame")
    if sites_f32.shape[0] <= 0:
        raise ValueError("sites_f32 must contain at least one site")
    if sites_f32.shape[0] > 64:
        raise ValueError("fused_slab_affine_coeff_realray_rgba_depth_replay currently supports at most 64 sites")
    if site_rgba_f32.shape[0] != sites_f32.shape[0]:
        raise ValueError("site_rgba_f32 row count must match sites_f32 rows")
    if row_index_i32.shape[0] != ray_coeff_f32.shape[0]:
        raise ValueError("row_index_i32 length must match track count")
    if time_slab_count <= 0:
        raise ValueError("time_slab_count must be positive")
    if row_count <= 0:
        raise ValueError("row_count must be positive")
    expected_offsets = row_count * time_slab_count + 1
    if candidate_row_offsets_i32.shape[0] != expected_offsets:
        raise ValueError("candidate_row_offsets_i32 length must be row_count * time_slab_count + 1")
    row_index_cpu = row_index_i32.detach().cpu()
    row_offsets_cpu = candidate_row_offsets_i32.detach().cpu()
    if int(row_index_cpu.min().item()) < 0 or int(row_index_cpu.max().item()) >= row_count:
        raise ValueError("row_index_i32 values must be in [0, row_count)")
    _validate_csr_offsets_cpu(row_offsets_cpu, candidate_count=candidate_depth_coeff_f32.shape[0])
    if config.far <= config.near:
        raise ValueError("config.far must be greater than config.near")
    if config.invalid_epsilon <= 0.0:
        raise ValueError("config.invalid_epsilon must be positive")
    if config.transmittance_threshold < 0.0:
        raise ValueError("config.transmittance_threshold must be nonnegative")

    ops = torch.ops.world_foam_lane2_fused_slab_v0
    if not hasattr(ops, "fused_slab_affine_coeff_realray_rgba_depth_replay"):
        raise RuntimeError(
            "world_foam_lane2_fused_slab_v0 fused_slab_affine_coeff_realray_rgba_depth_replay op not found. "
            "Build this variant first."
        )
    config_i32 = torch.tensor(
        [
            ray_coeff_f32.shape[0],
            sites_f32.shape[0],
            frame_t_f32.shape[0],
            time_slab_count,
            row_count,
            candidate_depth_coeff_f32.shape[0],
        ],
        device=row_index_i32.device,
        dtype=torch.int32,
    )
    config_f32 = torch.tensor(
        [config.near, config.far, config.invalid_epsilon, config.transmittance_threshold],
        device=row_index_i32.device,
        dtype=torch.float32,
    )
    return ops.fused_slab_affine_coeff_realray_rgba_depth_replay(
        row_index_i32,
        candidate_row_offsets_i32,
        candidate_depth_coeff_f32,
        sites_f32,
        site_rgba_f32,
        ray_coeff_f32,
        frame_t_f32,
        config_i32,
        config_f32,
    )


def fused_slab_affine_coeff16_realray_rgba_depth_replay(
    row_index_i32: Tensor,
    candidate_row_offsets_i32: Tensor,
    candidate_depth_coeff_f16: Tensor,
    sites_f32: Tensor,
    site_rgba_f32: Tensor,
    ray_coeff_f32: Tensor,
    frame_t_f32: Tensor,
    config: RealRayReplayConfig,
    *,
    time_slab_count: int,
    row_count: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Render per-track affine CSR candidates using half-precision depth coefficients."""
    row_index_i32 = row_index_i32.contiguous()
    candidate_row_offsets_i32 = candidate_row_offsets_i32.contiguous()
    candidate_depth_coeff_f16 = candidate_depth_coeff_f16.contiguous()
    sites_f32 = sites_f32.contiguous()
    site_rgba_f32 = site_rgba_f32.contiguous()
    ray_coeff_f32 = ray_coeff_f32.contiguous()
    frame_t_f32 = frame_t_f32.contiguous()
    _require_mps_tensor(row_index_i32, name="row_index_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(candidate_row_offsets_i32, name="candidate_row_offsets_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(candidate_depth_coeff_f16, name="candidate_depth_coeff_f16", dtype=torch.float16, cols=4)
    _require_mps_tensor(sites_f32, name="sites_f32", dtype=torch.float32, cols=5)
    _require_mps_tensor(site_rgba_f32, name="site_rgba_f32", dtype=torch.float32, cols=4)
    _require_mps_tensor(ray_coeff_f32, name="ray_coeff_f32", dtype=torch.float32, cols=12)
    _require_mps_tensor(frame_t_f32, name="frame_t_f32", dtype=torch.float32, cols=None)
    if ray_coeff_f32.shape[0] <= 0:
        raise ValueError("ray_coeff_f32 must contain at least one track")
    if frame_t_f32.shape[0] <= 0:
        raise ValueError("frame_t_f32 must contain at least one frame")
    if sites_f32.shape[0] <= 0:
        raise ValueError("sites_f32 must contain at least one site")
    if sites_f32.shape[0] > 64:
        raise ValueError("fused_slab_affine_coeff16_realray_rgba_depth_replay currently supports at most 64 sites")
    if site_rgba_f32.shape[0] != sites_f32.shape[0]:
        raise ValueError("site_rgba_f32 row count must match sites_f32 rows")
    if row_index_i32.shape[0] != ray_coeff_f32.shape[0]:
        raise ValueError("row_index_i32 length must match track count")
    if time_slab_count <= 0:
        raise ValueError("time_slab_count must be positive")
    if row_count <= 0:
        raise ValueError("row_count must be positive")
    expected_offsets = row_count * time_slab_count + 1
    if candidate_row_offsets_i32.shape[0] != expected_offsets:
        raise ValueError("candidate_row_offsets_i32 length must be row_count * time_slab_count + 1")
    row_index_cpu = row_index_i32.detach().cpu()
    row_offsets_cpu = candidate_row_offsets_i32.detach().cpu()
    if int(row_index_cpu.min().item()) < 0 or int(row_index_cpu.max().item()) >= row_count:
        raise ValueError("row_index_i32 values must be in [0, row_count)")
    _validate_csr_offsets_cpu(
        row_offsets_cpu,
        candidate_count=candidate_depth_coeff_f16.shape[0],
        max_boundaries=MAX_REALRAY_FUSED_MSE_BOUNDARIES,
    )
    if config.far <= config.near:
        raise ValueError("config.far must be greater than config.near")
    if config.invalid_epsilon <= 0.0:
        raise ValueError("config.invalid_epsilon must be positive")
    if config.transmittance_threshold < 0.0:
        raise ValueError("config.transmittance_threshold must be nonnegative")

    ops = torch.ops.world_foam_lane2_fused_slab_v0
    if not hasattr(ops, "fused_slab_affine_coeff16_realray_rgba_depth_replay"):
        raise RuntimeError(
            "world_foam_lane2_fused_slab_v0 fused_slab_affine_coeff16_realray_rgba_depth_replay op not found. "
            "Build this variant first."
        )
    config_i32 = torch.tensor(
        [
            ray_coeff_f32.shape[0],
            sites_f32.shape[0],
            frame_t_f32.shape[0],
            time_slab_count,
            row_count,
            candidate_depth_coeff_f16.shape[0],
        ],
        device=row_index_i32.device,
        dtype=torch.int32,
    )
    config_f32 = torch.tensor(
        [config.near, config.far, config.invalid_epsilon, config.transmittance_threshold],
        device=row_index_i32.device,
        dtype=torch.float32,
    )
    return ops.fused_slab_affine_coeff16_realray_rgba_depth_replay(
        row_index_i32,
        candidate_row_offsets_i32,
        candidate_depth_coeff_f16,
        sites_f32,
        site_rgba_f32,
        ray_coeff_f32,
        frame_t_f32,
        config_i32,
        config_f32,
    )


def fused_slab_affine_num32_den16_realray_rgba_depth_replay(
    row_index_i32: Tensor,
    candidate_row_offsets_i32: Tensor,
    candidate_depth_num_f32: Tensor,
    candidate_depth_den_f16: Tensor,
    sites_f32: Tensor,
    site_rgba_f32: Tensor,
    ray_coeff_f32: Tensor,
    frame_t_f32: Tensor,
    config: RealRayReplayConfig,
    *,
    time_slab_count: int,
    row_count: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Render per-track affine CSR candidates with fp32 numerator and fp16 denominator coefficients."""
    row_index_i32 = row_index_i32.contiguous()
    candidate_row_offsets_i32 = candidate_row_offsets_i32.contiguous()
    candidate_depth_num_f32 = candidate_depth_num_f32.contiguous()
    candidate_depth_den_f16 = candidate_depth_den_f16.contiguous()
    sites_f32 = sites_f32.contiguous()
    site_rgba_f32 = site_rgba_f32.contiguous()
    ray_coeff_f32 = ray_coeff_f32.contiguous()
    frame_t_f32 = frame_t_f32.contiguous()
    _require_mps_tensor(row_index_i32, name="row_index_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(candidate_row_offsets_i32, name="candidate_row_offsets_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(candidate_depth_num_f32, name="candidate_depth_num_f32", dtype=torch.float32, cols=2)
    _require_mps_tensor(candidate_depth_den_f16, name="candidate_depth_den_f16", dtype=torch.float16, cols=2)
    _require_mps_tensor(sites_f32, name="sites_f32", dtype=torch.float32, cols=5)
    _require_mps_tensor(site_rgba_f32, name="site_rgba_f32", dtype=torch.float32, cols=4)
    _require_mps_tensor(ray_coeff_f32, name="ray_coeff_f32", dtype=torch.float32, cols=12)
    _require_mps_tensor(frame_t_f32, name="frame_t_f32", dtype=torch.float32, cols=None)
    if ray_coeff_f32.shape[0] <= 0:
        raise ValueError("ray_coeff_f32 must contain at least one track")
    if frame_t_f32.shape[0] <= 0:
        raise ValueError("frame_t_f32 must contain at least one frame")
    if sites_f32.shape[0] <= 0:
        raise ValueError("sites_f32 must contain at least one site")
    if sites_f32.shape[0] > 64:
        raise ValueError("fused_slab_affine_num32_den16_realray_rgba_depth_replay supports at most 64 sites")
    if site_rgba_f32.shape[0] != sites_f32.shape[0]:
        raise ValueError("site_rgba_f32 row count must match sites_f32 rows")
    if row_index_i32.shape[0] != ray_coeff_f32.shape[0]:
        raise ValueError("row_index_i32 length must match track count")
    if candidate_depth_den_f16.shape[0] != candidate_depth_num_f32.shape[0]:
        raise ValueError("candidate_depth_den_f16 row count must match candidate_depth_num_f32 rows")
    if time_slab_count <= 0:
        raise ValueError("time_slab_count must be positive")
    if row_count <= 0:
        raise ValueError("row_count must be positive")
    expected_offsets = row_count * time_slab_count + 1
    if candidate_row_offsets_i32.shape[0] != expected_offsets:
        raise ValueError("candidate_row_offsets_i32 length must be row_count * time_slab_count + 1")
    row_index_cpu = row_index_i32.detach().cpu()
    row_offsets_cpu = candidate_row_offsets_i32.detach().cpu()
    if int(row_index_cpu.min().item()) < 0 or int(row_index_cpu.max().item()) >= row_count:
        raise ValueError("row_index_i32 values must be in [0, row_count)")
    _validate_csr_offsets_cpu(
        row_offsets_cpu,
        candidate_count=candidate_depth_num_f32.shape[0],
        max_boundaries=MAX_REALRAY_FUSED_MSE_BOUNDARIES,
    )
    if config.far <= config.near:
        raise ValueError("config.far must be greater than config.near")
    if config.invalid_epsilon <= 0.0:
        raise ValueError("config.invalid_epsilon must be positive")
    if config.transmittance_threshold < 0.0:
        raise ValueError("config.transmittance_threshold must be nonnegative")

    ops = torch.ops.world_foam_lane2_fused_slab_v0
    if not hasattr(ops, "fused_slab_affine_num32_den16_realray_rgba_depth_replay"):
        raise RuntimeError(
            "world_foam_lane2_fused_slab_v0 fused_slab_affine_num32_den16_realray_rgba_depth_replay op not found. "
            "Build this variant first."
        )
    config_i32 = torch.tensor(
        [
            ray_coeff_f32.shape[0],
            sites_f32.shape[0],
            frame_t_f32.shape[0],
            time_slab_count,
            row_count,
            candidate_depth_num_f32.shape[0],
        ],
        device=row_index_i32.device,
        dtype=torch.int32,
    )
    config_f32 = torch.tensor(
        [config.near, config.far, config.invalid_epsilon, config.transmittance_threshold],
        device=row_index_i32.device,
        dtype=torch.float32,
    )
    return ops.fused_slab_affine_num32_den16_realray_rgba_depth_replay(
        row_index_i32,
        candidate_row_offsets_i32,
        candidate_depth_num_f32,
        candidate_depth_den_f16,
        sites_f32,
        site_rgba_f32,
        ray_coeff_f32,
        frame_t_f32,
        config_i32,
        config_f32,
    )


def fused_slab_affine_num32_den16_ownerupdate_realray_rgba_depth_replay(
    row_index_i32: Tensor,
    candidate_row_offsets_i32: Tensor,
    candidate_boundary_ids_i32: Tensor,
    candidate_depth_num_f32: Tensor,
    candidate_depth_den_f16: Tensor,
    boundary_site_pairs_i32: Tensor,
    sites_f32: Tensor,
    site_rgba_f32: Tensor,
    ray_coeff_f32: Tensor,
    frame_t_f32: Tensor,
    config: RealRayReplayConfig,
    *,
    time_slab_count: int,
    row_count: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Render mixed-precision affine CSR candidates with midpoint owner selection.

    This is a diagnostic topology path for candidate-id CSR tapes. It keeps
    parity with the mixed replay path by recomputing the winning owner at each
    segment midpoint instead of assuming every candidate is a true owner
    transition.
    """
    row_index_i32 = row_index_i32.contiguous()
    candidate_row_offsets_i32 = candidate_row_offsets_i32.contiguous()
    candidate_boundary_ids_i32 = candidate_boundary_ids_i32.contiguous()
    candidate_depth_num_f32 = candidate_depth_num_f32.contiguous()
    candidate_depth_den_f16 = candidate_depth_den_f16.contiguous()
    boundary_site_pairs_i32 = boundary_site_pairs_i32.contiguous()
    sites_f32 = sites_f32.contiguous()
    site_rgba_f32 = site_rgba_f32.contiguous()
    ray_coeff_f32 = ray_coeff_f32.contiguous()
    frame_t_f32 = frame_t_f32.contiguous()
    _require_mps_tensor(row_index_i32, name="row_index_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(candidate_row_offsets_i32, name="candidate_row_offsets_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(candidate_boundary_ids_i32, name="candidate_boundary_ids_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(candidate_depth_num_f32, name="candidate_depth_num_f32", dtype=torch.float32, cols=2)
    _require_mps_tensor(candidate_depth_den_f16, name="candidate_depth_den_f16", dtype=torch.float16, cols=2)
    _require_mps_tensor(boundary_site_pairs_i32, name="boundary_site_pairs_i32", dtype=torch.int32, cols=2)
    _require_mps_tensor(sites_f32, name="sites_f32", dtype=torch.float32, cols=5)
    _require_mps_tensor(site_rgba_f32, name="site_rgba_f32", dtype=torch.float32, cols=4)
    _require_mps_tensor(ray_coeff_f32, name="ray_coeff_f32", dtype=torch.float32, cols=12)
    _require_mps_tensor(frame_t_f32, name="frame_t_f32", dtype=torch.float32, cols=None)
    if ray_coeff_f32.shape[0] <= 0:
        raise ValueError("ray_coeff_f32 must contain at least one track")
    if frame_t_f32.shape[0] <= 0:
        raise ValueError("frame_t_f32 must contain at least one frame")
    if sites_f32.shape[0] <= 0:
        raise ValueError("sites_f32 must contain at least one site")
    if sites_f32.shape[0] > 64:
        raise ValueError(
            "fused_slab_affine_num32_den16_ownerupdate_realray_rgba_depth_replay supports at most 64 sites"
        )
    if site_rgba_f32.shape[0] != sites_f32.shape[0]:
        raise ValueError("site_rgba_f32 row count must match sites_f32 rows")
    if row_index_i32.shape[0] != ray_coeff_f32.shape[0]:
        raise ValueError("row_index_i32 length must match track count")
    if candidate_depth_den_f16.shape[0] != candidate_depth_num_f32.shape[0]:
        raise ValueError("candidate_depth_den_f16 row count must match candidate_depth_num_f32 rows")
    if candidate_boundary_ids_i32.shape[0] != candidate_depth_num_f32.shape[0]:
        raise ValueError("candidate_boundary_ids_i32 length must match candidate_depth_num_f32 rows")
    if time_slab_count <= 0:
        raise ValueError("time_slab_count must be positive")
    if row_count <= 0:
        raise ValueError("row_count must be positive")
    expected_offsets = row_count * time_slab_count + 1
    if candidate_row_offsets_i32.shape[0] != expected_offsets:
        raise ValueError("candidate_row_offsets_i32 length must be row_count * time_slab_count + 1")
    row_index_cpu = row_index_i32.detach().cpu()
    row_offsets_cpu = candidate_row_offsets_i32.detach().cpu()
    candidate_ids_cpu = candidate_boundary_ids_i32.detach().cpu()
    boundary_pairs_cpu = boundary_site_pairs_i32.detach().cpu()
    if int(row_index_cpu.min().item()) < 0 or int(row_index_cpu.max().item()) >= row_count:
        raise ValueError("row_index_i32 values must be in [0, row_count)")
    _validate_csr_offsets_cpu(row_offsets_cpu, candidate_count=candidate_depth_num_f32.shape[0])
    if candidate_ids_cpu.numel() and (
        int(candidate_ids_cpu.min().item()) < 0
        or int(candidate_ids_cpu.max().item()) >= boundary_site_pairs_i32.shape[0]
    ):
        raise ValueError("candidate_boundary_ids_i32 values must be in [0, boundary_count)")
    if boundary_pairs_cpu.numel() and (
        int(boundary_pairs_cpu.min().item()) < 0 or int(boundary_pairs_cpu.max().item()) >= sites_f32.shape[0]
    ):
        raise ValueError("boundary_site_pairs_i32 values must be in [0, site_count)")
    if config.far <= config.near:
        raise ValueError("config.far must be greater than config.near")
    if config.invalid_epsilon <= 0.0:
        raise ValueError("config.invalid_epsilon must be positive")
    if config.transmittance_threshold < 0.0:
        raise ValueError("config.transmittance_threshold must be nonnegative")

    ops = torch.ops.world_foam_lane2_fused_slab_v0
    if not hasattr(ops, "fused_slab_affine_num32_den16_ownerupdate_realray_rgba_depth_replay"):
        raise RuntimeError(
            "world_foam_lane2_fused_slab_v0 fused_slab_affine_num32_den16_ownerupdate_realray_rgba_depth_replay "
            "op not found. Build this variant first."
        )
    config_i32 = torch.tensor(
        [
            ray_coeff_f32.shape[0],
            sites_f32.shape[0],
            frame_t_f32.shape[0],
            time_slab_count,
            row_count,
            candidate_depth_num_f32.shape[0],
            boundary_site_pairs_i32.shape[0],
        ],
        device=row_index_i32.device,
        dtype=torch.int32,
    )
    config_f32 = torch.tensor(
        [config.near, config.far, config.invalid_epsilon, config.transmittance_threshold],
        device=row_index_i32.device,
        dtype=torch.float32,
    )
    return ops.fused_slab_affine_num32_den16_ownerupdate_realray_rgba_depth_replay(
        row_index_i32,
        candidate_row_offsets_i32,
        candidate_boundary_ids_i32,
        candidate_depth_num_f32,
        candidate_depth_den_f16,
        boundary_site_pairs_i32,
        sites_f32,
        site_rgba_f32,
        ray_coeff_f32,
        frame_t_f32,
        config_i32,
        config_f32,
    )


class _SharedRealRayRGBADepthReplayFunction(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx: torch.autograd.function.FunctionCtx,
        boundary_f32: Tensor,
        candidate_mask_i32: Tensor,
        sites_f32: Tensor,
        site_rgba_f32: Tensor,
        track_rays_f32: Tensor,
        frame_t_f32: Tensor,
        near: float,
        far: float,
        invalid_epsilon: float,
        transmittance_threshold: float,
    ) -> tuple[Tensor, Tensor, Tensor]:
        config = RealRayReplayConfig(
            near=float(near),
            far=float(far),
            invalid_epsilon=float(invalid_epsilon),
            transmittance_threshold=float(transmittance_threshold),
        )
        output_rgb, output_alpha, output_depth = shared_realray_rgba_depth_replay(
            boundary_f32,
            candidate_mask_i32,
            sites_f32,
            site_rgba_f32,
            track_rays_f32,
            frame_t_f32,
            config,
        )
        ctx.save_for_backward(
            boundary_f32,
            candidate_mask_i32,
            sites_f32,
            site_rgba_f32,
            track_rays_f32,
            frame_t_f32,
        )
        ctx.config = config
        return output_rgb, output_alpha, output_depth

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: torch.autograd.function.FunctionCtx,
        grad_rgb: Tensor | None,
        grad_alpha: Tensor | None,
        grad_depth: Tensor | None,
    ) -> tuple[None, None, None, Tensor, None, None, None, None, None, None]:
        (
            boundary_f32,
            candidate_mask_i32,
            sites_f32,
            site_rgba_f32,
            track_rays_f32,
            frame_t_f32,
        ) = ctx.saved_tensors
        track_count = track_rays_f32.shape[0]
        frame_count = frame_t_f32.shape[0]
        if grad_rgb is None:
            grad_rgb = torch.zeros(
                (track_count, frame_count, 3),
                device=site_rgba_f32.device,
                dtype=torch.float32,
            )
        if grad_alpha is None:
            grad_alpha = torch.zeros(
                (track_count, frame_count),
                device=site_rgba_f32.device,
                dtype=torch.float32,
            )
        if grad_depth is None:
            grad_depth = torch.zeros(
                (track_count, frame_count),
                device=site_rgba_f32.device,
                dtype=torch.float32,
            )
        _, _, _, grad_site_rgba = shared_realray_rgba_depth_vjp_reduce(
            boundary_f32,
            candidate_mask_i32,
            sites_f32,
            site_rgba_f32,
            track_rays_f32,
            frame_t_f32,
            grad_rgb,
            grad_alpha,
            grad_depth,
            ctx.config,
        )
        return None, None, None, grad_site_rgba, None, None, None, None, None, None


def shared_realray_rgba_depth_autograd(
    boundary_f32: Tensor,
    candidate_mask_i32: Tensor,
    sites_f32: Tensor,
    site_rgba_f32: Tensor,
    track_rays_f32: Tensor,
    frame_t_f32: Tensor,
    config: RealRayReplayConfig,
) -> tuple[Tensor, Tensor, Tensor]:
    """Render shared real-ray RGB/alpha/depth with frozen-geometry autograd.

    Backward returns gradients only for `site_rgba_f32 [S,4]`. Boundary cuts,
    candidate masks, site positions/weights, rays, camera geometry, sorting,
    ownership, and topology are fixed and receive no gradients.
    """
    return _SharedRealRayRGBADepthReplayFunction.apply(
        boundary_f32,
        candidate_mask_i32,
        sites_f32,
        site_rgba_f32,
        track_rays_f32,
        frame_t_f32,
        config.near,
        config.far,
        config.invalid_epsilon,
        config.transmittance_threshold,
    )


class _SharedRealRayRGBADepthCSRReplayFunction(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx: torch.autograd.function.FunctionCtx,
        boundary_f32: Tensor,
        row_index_i32: Tensor,
        candidate_row_offsets_i32: Tensor,
        candidate_boundary_ids_i32: Tensor,
        sites_f32: Tensor,
        site_rgba_f32: Tensor,
        track_rays_f32: Tensor,
        frame_t_f32: Tensor,
        time_slab_count: int,
        row_count: int,
        near: float,
        far: float,
        invalid_epsilon: float,
        transmittance_threshold: float,
    ) -> tuple[Tensor, Tensor, Tensor]:
        config = RealRayReplayConfig(
            near=float(near),
            far=float(far),
            invalid_epsilon=float(invalid_epsilon),
            transmittance_threshold=float(transmittance_threshold),
        )
        track_count = track_rays_f32.shape[0]
        frame_count = frame_t_f32.shape[0]
        grad_rgb = torch.zeros((track_count, frame_count, 3), device=site_rgba_f32.device, dtype=torch.float32)
        grad_alpha = torch.zeros((track_count, frame_count), device=site_rgba_f32.device, dtype=torch.float32)
        grad_depth = torch.zeros((track_count, frame_count), device=site_rgba_f32.device, dtype=torch.float32)
        output_rgb, output_alpha, output_depth, _ = shared_realray_rgba_depth_vjp_reduce_csr(
            boundary_f32,
            row_index_i32,
            candidate_row_offsets_i32,
            candidate_boundary_ids_i32,
            sites_f32,
            site_rgba_f32,
            track_rays_f32,
            frame_t_f32,
            grad_rgb,
            grad_alpha,
            grad_depth,
            config,
            time_slab_count=int(time_slab_count),
            row_count=int(row_count),
        )
        ctx.save_for_backward(
            boundary_f32,
            row_index_i32,
            candidate_row_offsets_i32,
            candidate_boundary_ids_i32,
            sites_f32,
            site_rgba_f32,
            track_rays_f32,
            frame_t_f32,
        )
        ctx.config = config
        ctx.time_slab_count = int(time_slab_count)
        ctx.row_count = int(row_count)
        return output_rgb, output_alpha, output_depth

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: torch.autograd.function.FunctionCtx,
        grad_rgb: Tensor | None,
        grad_alpha: Tensor | None,
        grad_depth: Tensor | None,
    ) -> tuple[None, None, None, None, None, Tensor, None, None, None, None, None, None, None, None]:
        (
            boundary_f32,
            row_index_i32,
            candidate_row_offsets_i32,
            candidate_boundary_ids_i32,
            sites_f32,
            site_rgba_f32,
            track_rays_f32,
            frame_t_f32,
        ) = ctx.saved_tensors
        track_count = track_rays_f32.shape[0]
        frame_count = frame_t_f32.shape[0]
        if grad_rgb is None:
            grad_rgb = torch.zeros((track_count, frame_count, 3), device=site_rgba_f32.device, dtype=torch.float32)
        if grad_alpha is None:
            grad_alpha = torch.zeros((track_count, frame_count), device=site_rgba_f32.device, dtype=torch.float32)
        if grad_depth is None:
            grad_depth = torch.zeros((track_count, frame_count), device=site_rgba_f32.device, dtype=torch.float32)
        _, _, _, grad_site_rgba = shared_realray_rgba_depth_vjp_reduce_csr(
            boundary_f32,
            row_index_i32,
            candidate_row_offsets_i32,
            candidate_boundary_ids_i32,
            sites_f32,
            site_rgba_f32,
            track_rays_f32,
            frame_t_f32,
            grad_rgb,
            grad_alpha,
            grad_depth,
            ctx.config,
            time_slab_count=ctx.time_slab_count,
            row_count=ctx.row_count,
        )
        return None, None, None, None, None, grad_site_rgba, None, None, None, None, None, None, None, None


def shared_realray_rgba_depth_csr_autograd(
    boundary_f32: Tensor,
    row_index_i32: Tensor,
    candidate_row_offsets_i32: Tensor,
    candidate_boundary_ids_i32: Tensor,
    sites_f32: Tensor,
    site_rgba_f32: Tensor,
    track_rays_f32: Tensor,
    frame_t_f32: Tensor,
    config: RealRayReplayConfig,
    *,
    time_slab_count: int,
    row_count: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Render shared real-ray RGB/alpha/depth with CSR frozen-geometry autograd.

    Backward returns gradients only for `site_rgba_f32 [S,4]`. CSR candidate
    storage, boundary cuts, site positions/weights, rays, camera geometry,
    sorting, ownership, and topology are fixed and receive no gradients.
    """
    return _SharedRealRayRGBADepthCSRReplayFunction.apply(
        boundary_f32,
        row_index_i32,
        candidate_row_offsets_i32,
        candidate_boundary_ids_i32,
        sites_f32,
        site_rgba_f32,
        track_rays_f32,
        frame_t_f32,
        int(time_slab_count),
        int(row_count),
        config.near,
        config.far,
        config.invalid_epsilon,
        config.transmittance_threshold,
    )

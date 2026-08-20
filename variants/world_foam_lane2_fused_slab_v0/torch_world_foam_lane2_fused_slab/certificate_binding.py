"""Content-bound CPU certificate handoff for native fixed-word WorldFoam.

The binding is intentionally an in-process capability, not a signed artifact.
It can only be created by rerunning ``certify_prepared_adaptive_lie_world`` on
the exact compact prepared snapshot.  Native launch tokens retain the sealed
binding and fail closed if any bound CPU or native tensor changes.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any

import torch
from torch import Tensor

_BINDING_SEAL = object()
_TRAINING_BINDING_SEAL = object()


@dataclass(frozen=True)
class NativeFixedWordP0ChartCertificate:
    chart_index: int
    t_min: float
    t_max: float
    node_count: int
    chart_digest: str
    estimated_interval_jet_work_units: int
    owner_identity_certificate_digest: str
    owner_identity_tolerance: float
    owner_certificate_leaf_count: int
    owner_certificate_deepest_split: int
    owner_checked_endpoint_inequality_count: int
    maximum_owner_difference_upper_bound: float
    minimum_certified_owner_margin: float


@dataclass(frozen=True)
class NativeFixedWordP0ContinuousCertificateBinding:
    """Sealed acceptance facts and exact compact snapshot provenance."""

    passed: bool
    canonical_digest: str
    topology_snapshot_generation: str
    world_snapshot_generation: str
    charts: tuple[NativeFixedWordP0ChartCertificate, ...]
    transfer_tolerance: float
    world_jacobian_tolerance: float
    site_geometry_jacobian_tolerance: float
    max_split_depth: int
    max_leaves_per_chart: int
    max_interval_jet_work_units_per_chart: int
    arithmetic_fraction_bits: int
    owner_identity_certified: bool
    owner_identity_scope: str
    owner_identity_tolerance: float
    owner_max_split_depth: int
    owner_max_leaves_per_chart: int
    owner_max_work_units_per_chart: int
    maximum_owner_difference_upper_bound: float
    total_owner_certificate_leaves: int
    runtime_floating_point_roundoff_certified: bool
    _canonical_payload_json: str = field(repr=False)
    _prepared: Any = field(repr=False)
    _bound_tensors: tuple[Tensor, ...] = field(repr=False)
    _bound_tensor_signatures: tuple[tuple[Any, ...], ...] = field(repr=False)
    _sample_barycentric_weights: tuple[Tensor, ...] = field(repr=False)
    _sample_barycentric_signatures: tuple[tuple[Any, ...], ...] = field(repr=False)
    _seal: object = field(repr=False)

    @property
    def binding_mode(self) -> str:
        return "strict_frozen_evaluation"

    @property
    def paper_evidence_eligible(self) -> bool:
        return True

    @property
    def transfer_jacobian_certified(self) -> bool:
        return True

    @property
    def approximation_error_certified(self) -> bool:
        return True

    @property
    def geometry_rays_immutable(self) -> bool:
        return True

    @property
    def sample_weight_evaluation(self) -> str:
        return "verified_fit_derived_second_form_barycentric"

    def assert_current(self) -> None:
        """Fail closed without rehashing tensor payloads on the K-block path."""
        if self._seal is not _BINDING_SEAL:
            raise ValueError("continuous certificate binding was fabricated outside the certifier")
        if not self.passed:
            raise ValueError("continuous certificate binding did not pass")
        if not self.owner_identity_certified:
            raise ValueError("continuous certificate binding does not certify owner identity")
        if self.owner_identity_scope != "all_competitor_sites_continuously_certified":
            raise ValueError("continuous certificate owner-identity scope is invalid")
        if self.owner_identity_tolerance < 0.0:
            raise ValueError("continuous certificate owner-identity tolerance is invalid")
        if self.total_owner_certificate_leaves != sum(
            chart.owner_certificate_leaf_count for chart in self.charts
        ):
            raise ValueError("continuous certificate owner-identity leaf accounting is invalid")
        if self.maximum_owner_difference_upper_bound != max(
            (chart.maximum_owner_difference_upper_bound for chart in self.charts),
            default=float("-inf"),
        ):
            raise ValueError("continuous certificate owner-identity bound aggregation is invalid")
        for chart in self.charts:
            if (
                chart.owner_certificate_leaf_count < 1
                or chart.owner_checked_endpoint_inequality_count < 1
                or chart.maximum_owner_difference_upper_bound > chart.owner_identity_tolerance
                or chart.owner_identity_tolerance != self.owner_identity_tolerance
            ):
                raise ValueError("continuous certificate chart owner-identity facts are invalid")
        if self.runtime_floating_point_roundoff_certified:
            raise ValueError("native runtime roundoff must remain explicitly uncertified")
        if hashlib.sha256(self._canonical_payload_json.encode("utf-8")).hexdigest() != self.canonical_digest:
            raise ValueError("continuous certificate canonical digest is invalid")
        payload = json.loads(self._canonical_payload_json)
        if payload.get("binding") != self._canonical_binding_facts():
            raise ValueError("continuous certificate exposed facts do not match its canonical payload")
        self._prepared.assert_current()
        if tuple(_tensor_signature(tensor) for tensor in self._bound_tensors) != self._bound_tensor_signatures:
            raise ValueError("continuous certificate snapshot tensors changed after certification")
        if len(self._sample_barycentric_weights) != len(self.charts) or tuple(
            _tensor_signature(tensor) for tensor in self._sample_barycentric_weights
        ) != self._sample_barycentric_signatures:
            raise ValueError("continuous certificate sample-weight schedule changed")

    def _canonical_binding_facts(self) -> dict[str, object]:
        return {
            "schema": "worldfoam-native-fixed-word-p0-continuous-binding-v3",
            "passed": self.passed,
            "topology_snapshot_generation": self.topology_snapshot_generation,
            "world_snapshot_generation": self.world_snapshot_generation,
            "charts": [asdict(chart) for chart in self.charts],
            "transfer_tolerance": self.transfer_tolerance,
            "world_jacobian_tolerance": self.world_jacobian_tolerance,
            "site_geometry_jacobian_tolerance": self.site_geometry_jacobian_tolerance,
            "max_split_depth": self.max_split_depth,
            "max_leaves_per_chart": self.max_leaves_per_chart,
            "max_interval_jet_work_units_per_chart": self.max_interval_jet_work_units_per_chart,
            "arithmetic_fraction_bits": self.arithmetic_fraction_bits,
            "owner_identity_certified": self.owner_identity_certified,
            "owner_identity_scope": self.owner_identity_scope,
            "owner_identity_tolerance": self.owner_identity_tolerance,
            "owner_max_split_depth": self.owner_max_split_depth,
            "owner_max_leaves_per_chart": self.owner_max_leaves_per_chart,
            "owner_max_work_units_per_chart": self.owner_max_work_units_per_chart,
            "maximum_owner_difference_upper_bound": self.maximum_owner_difference_upper_bound,
            "total_owner_certificate_leaves": self.total_owner_certificate_leaves,
            "runtime_floating_point_roundoff_certified": (
                self.runtime_floating_point_roundoff_certified
            ),
            "sample_weight_evaluation": self.sample_weight_evaluation,
        }

    def assert_native_topology(
        self,
        *,
        word_offsets_i32: Tensor,
        word_owner_i32: Tensor,
        word_left_incidence_i32: Tensor,
        word_right_incidence_i32: Tensor,
        track_incidence_offsets_i32: Tensor,
        incidence_boundary_i32: Tensor,
        boundary_site_pairs_i32: Tensor,
    ) -> None:
        self.assert_current()
        expected = self._prepared.topology
        for name, actual, reference in (
            ("word_offsets_i32", word_offsets_i32, expected.word_offsets_i32),
            ("word_owner_i32", word_owner_i32, expected.word_owner_i32),
            ("word_left_incidence_i32", word_left_incidence_i32, expected.word_left_incidence_i32),
            ("word_right_incidence_i32", word_right_incidence_i32, expected.word_right_incidence_i32),
            (
                "track_incidence_offsets_i32",
                track_incidence_offsets_i32,
                expected.track_incidence_offsets_i32,
            ),
            ("incidence_boundary_i32", incidence_boundary_i32, expected.incidence_boundary_i32),
            ("boundary_site_pairs_i32", boundary_site_pairs_i32, expected.boundary_site_pairs_i32),
        ):
            _assert_exact_quantized_tensor(name, actual, reference, dtype=torch.int32)

    def assert_native_world(
        self,
        *,
        sites_f32: Tensor,
        site_rgba_f32: Tensor,
        track_ray_coeff_f32: Tensor,
    ) -> None:
        self.assert_current()
        snapshot = self._prepared.world_snapshot
        expected_rgba = torch.cat(
            (snapshot.site_color, snapshot.site_density[:, None]),
            dim=1,
        )
        _assert_exact_quantized_tensor("sites_f32", sites_f32, self._prepared.site_geometry, dtype=torch.float32)
        _assert_exact_quantized_tensor("site_rgba_f32", site_rgba_f32, expected_rgba, dtype=torch.float32)
        _assert_exact_quantized_tensor(
            "track_ray_coeff_f32",
            track_ray_coeff_f32,
            snapshot.ray_coefficients,
            dtype=torch.float32,
        )

    def assert_replay_interval(self, *, near: float, far: float) -> None:
        self.assert_current()
        for chart in self._prepared.world_snapshot.atlas.charts:
            if float(near) != float(chart.near) or float(far) != float(chart.far):
                raise ValueError("native replay near/far interval does not match the certified snapshot")

    def assert_native_chart(self, chart_index: int, compiler_node_t_f32: Tensor) -> NativeFixedWordP0ChartCertificate:
        self.assert_current()
        if chart_index < 0 or chart_index >= len(self.charts):
            raise ValueError("chart_index is outside the continuous certificate")
        chart = self.charts[chart_index]
        expected_chart = self._prepared.world_snapshot.atlas.charts[chart_index]
        _assert_exact_quantized_tensor(
            "compiler_node_t_f32",
            compiler_node_t_f32,
            expected_chart.transfer_atlas.node_times,
            dtype=torch.float32,
        )
        if (
            chart.node_count != expected_chart.node_count
            or chart.t_min != expected_chart.transfer_atlas.t_min
            or chart.t_max != expected_chart.transfer_atlas.t_max
        ):
            raise ValueError("continuous certificate chart selection changed")
        return chart

    def validate_sample_times(self, chart_index: int, sample_t_f64: Tensor) -> Tensor:
        """Return a finite CPU time slice contained in one certified chart.

        This check is deliberately linear only in the number of supplied
        times.  Chart-state setup may validate an entire logical partition,
        but it must not materialize the corresponding ``F_c x J`` basis or
        interpolation weights; those belong to bounded ``K`` blocks.
        """
        self.assert_current()
        if chart_index < 0 or chart_index >= len(self.charts):
            raise ValueError("chart_index is outside the continuous certificate")
        times = torch.as_tensor(sample_t_f64, dtype=torch.float64, device="cpu").reshape(-1)
        if times.numel() == 0 or not bool(torch.isfinite(times).all().item()):
            raise ValueError("sample times must be nonempty finite float64 values")
        chart_binding = self.charts[chart_index]
        final_chart = chart_index + 1 == len(self.charts)
        below = bool(torch.any(times < chart_binding.t_min).item())
        above = bool(
            torch.any(times > chart_binding.t_max if final_chart else times >= chart_binding.t_max).item()
        )
        if below or above:
            bracket = "[t_min,t_max]" if final_chart else "[t_min,t_max)"
            raise ValueError(f"sample times leave certified chart interval {bracket}")
        return times

    def sample_to_node_weights(self, chart_index: int, sample_t_f64: Tensor) -> Tensor:
        """Build verified ``O(K J)`` cardinal weights on CPU."""
        return self.sample_to_node_weight_result(chart_index, sample_t_f64).weights.to(
            dtype=torch.float32
        ).contiguous()

    def sample_to_node_weight_result(self, chart_index: int, sample_t_f64: Tensor) -> Any:
        """Return weights plus explicit linear/fallback cost provenance."""
        from compact_lie_schedule import fit_derived_sample_to_node_weights

        times = self.validate_sample_times(chart_index, sample_t_f64)
        chart_binding = self.charts[chart_index]
        transfer = self._prepared.world_snapshot.atlas.charts[chart_index].transfer_atlas
        return fit_derived_sample_to_node_weights(
            times,
            t_min=chart_binding.t_min,
            t_max=chart_binding.t_max,
            node_times=transfer.node_times,
            fit_matrix=transfer.fit_matrix,
            barycentric_weights=self._sample_barycentric_weights[chart_index],
        )


@dataclass(frozen=True)
class NativeFixedWordP0TrainingChartSchedule:
    """Immutable chart schedule plus the retained all-site owner certificate.

    This is deliberately not a transfer or Jacobian certificate.  The node
    schedule is reused as a training approximation after material refreshes.
    """

    chart_index: int
    t_min: float
    t_max: float
    near: float
    far: float
    node_count: int
    chart_digest: str
    owner_identity_certificate_digest: str
    owner_identity_tolerance: float
    owner_certificate_leaf_count: int
    owner_checked_endpoint_inequality_count: int
    maximum_owner_difference_upper_bound: float
    transfer_jacobian_certified: bool = False
    approximation_error_certified: bool = False
    paper_evidence_eligible: bool = False


@dataclass(frozen=True)
class _TrainingOwnerEvidence:
    chart_index: int
    certificate_digest: str
    ownership_tolerance: float
    leaf_count: int
    checked_endpoint_inequality_count: int
    maximum_owner_difference_upper_bound: float


@dataclass(frozen=True)
class NativeFixedWordP0TrainingTopologyBinding:
    """Sealed material-training capability with immutable geometry and rays.

    It retains a passed all-competitor owner/topology result and compact copies
    of the exact topology, sites, rays, chart partition, and node schedule that
    result covered. It does not retain the prepared object or its full/material
    atlases. Live site RGBA is intentionally outside the binding. Consequently
    transfer/Jacobian approximation error is explicitly uncertified and results
    produced through this capability are not paper evidence.
    """

    canonical_digest: str
    topology_snapshot_generation: str
    training_snapshot_generation: str
    site_count: int
    charts: tuple[NativeFixedWordP0TrainingChartSchedule, ...]
    owner_identity_certified: bool
    owner_identity_scope: str
    owner_identity_tolerance: float
    maximum_owner_difference_upper_bound: float
    total_owner_certificate_leaves: int
    binding_mode: str
    transfer_jacobian_certified: bool
    approximation_error_certified: bool
    paper_evidence_eligible: bool
    geometry_rays_immutable: bool
    live_site_rgba_refresh_allowed: bool
    runtime_floating_point_roundoff_certified: bool
    _canonical_payload_json: str = field(repr=False)
    _bound_tensor_names: tuple[str, ...] = field(repr=False)
    _bound_tensors: tuple[Tensor, ...] = field(repr=False)
    _bound_tensor_signatures: tuple[tuple[Any, ...], ...] = field(repr=False)
    _sample_barycentric_weights: tuple[Tensor, ...] = field(repr=False)
    _sample_barycentric_signatures: tuple[tuple[Any, ...], ...] = field(repr=False)
    _seal: object = field(repr=False)

    @property
    def world_snapshot_generation(self) -> str:
        """Compatibility name for the immutable geometry/ray/schedule generation."""

        return self.training_snapshot_generation

    @property
    def resident_immutable_bytes(self) -> int:
        return sum(
            tensor.numel() * tensor.element_size()
            for tensor in (*self._bound_tensors, *self._sample_barycentric_weights)
        )

    @property
    def sample_weight_evaluation(self) -> str:
        return "verified_fit_derived_second_form_barycentric"

    def assert_current(self) -> None:
        if self._seal is not _TRAINING_BINDING_SEAL:
            raise ValueError("training topology binding was fabricated outside an owner certifier")
        if self.binding_mode != "training_owner_topology_only":
            raise ValueError("training topology binding cannot present itself as strict evaluation")
        if (
            self.transfer_jacobian_certified
            or self.approximation_error_certified
            or self.paper_evidence_eligible
        ):
            raise ValueError("training topology binding must remain transfer/Jacobian uncertified and non-paper")
        if not self.geometry_rays_immutable or not self.live_site_rgba_refresh_allowed:
            raise ValueError("training topology binding capability flags are invalid")
        if not self.owner_identity_certified:
            raise ValueError("training topology binding has no passed owner-identity certificate")
        if self.owner_identity_scope != "all_competitor_sites_continuously_certified":
            raise ValueError("training topology binding owner-identity scope is invalid")
        if self.owner_identity_tolerance < 0.0:
            raise ValueError("training topology binding owner-identity tolerance is invalid")
        if self.total_owner_certificate_leaves != sum(
            chart.owner_certificate_leaf_count for chart in self.charts
        ):
            raise ValueError("training topology binding owner leaf accounting is invalid")
        if self.maximum_owner_difference_upper_bound != max(
            (chart.maximum_owner_difference_upper_bound for chart in self.charts),
            default=float("-inf"),
        ):
            raise ValueError("training topology binding owner bound aggregation is invalid")
        for chart in self.charts:
            if (
                chart.transfer_jacobian_certified
                or chart.approximation_error_certified
                or chart.paper_evidence_eligible
                or chart.owner_certificate_leaf_count < 1
                or chart.owner_checked_endpoint_inequality_count < 1
                or chart.maximum_owner_difference_upper_bound > chart.owner_identity_tolerance
                or chart.owner_identity_tolerance != self.owner_identity_tolerance
            ):
                raise ValueError("training chart schedule certification facts are invalid")
        if self.runtime_floating_point_roundoff_certified:
            raise ValueError("native runtime roundoff must remain explicitly uncertified")
        if hashlib.sha256(self._canonical_payload_json.encode("utf-8")).hexdigest() != self.canonical_digest:
            raise ValueError("training topology binding canonical digest is invalid")
        payload = json.loads(self._canonical_payload_json)
        if payload.get("binding") != self._canonical_binding_facts():
            raise ValueError("training topology binding exposed facts do not match its canonical payload")
        if tuple(_tensor_signature(tensor) for tensor in self._bound_tensors) != self._bound_tensor_signatures:
            raise ValueError("training topology binding geometry, rays, topology, or schedule changed")
        if len(self._sample_barycentric_weights) != len(self.charts) or tuple(
            _tensor_signature(tensor) for tensor in self._sample_barycentric_weights
        ) != self._sample_barycentric_signatures:
            raise ValueError("training topology binding sample-weight schedule changed")

    def _canonical_binding_facts(self) -> dict[str, object]:
        return {
            "schema": "worldfoam-native-fixed-word-p0-training-topology-binding-v2",
            "canonical_scope": "owner_topology_geometry_rays_chart_schedule_only",
            "topology_snapshot_generation": self.topology_snapshot_generation,
            "training_snapshot_generation": self.training_snapshot_generation,
            "site_count": self.site_count,
            "charts": [asdict(chart) for chart in self.charts],
            "owner_identity_certified": self.owner_identity_certified,
            "owner_identity_scope": self.owner_identity_scope,
            "owner_identity_tolerance": self.owner_identity_tolerance,
            "maximum_owner_difference_upper_bound": self.maximum_owner_difference_upper_bound,
            "total_owner_certificate_leaves": self.total_owner_certificate_leaves,
            "binding_mode": self.binding_mode,
            "transfer_jacobian_certified": self.transfer_jacobian_certified,
            "approximation_error_certified": self.approximation_error_certified,
            "paper_evidence_eligible": self.paper_evidence_eligible,
            "geometry_rays_immutable": self.geometry_rays_immutable,
            "live_site_rgba_refresh_allowed": self.live_site_rgba_refresh_allowed,
            "runtime_floating_point_roundoff_certified": (
                self.runtime_floating_point_roundoff_certified
            ),
            "sample_weight_evaluation": self.sample_weight_evaluation,
        }

    def assert_native_topology(
        self,
        *,
        word_offsets_i32: Tensor,
        word_owner_i32: Tensor,
        word_left_incidence_i32: Tensor,
        word_right_incidence_i32: Tensor,
        track_incidence_offsets_i32: Tensor,
        incidence_boundary_i32: Tensor,
        boundary_site_pairs_i32: Tensor,
    ) -> None:
        self.assert_current()
        for name, actual in (
            ("word_offsets_i32", word_offsets_i32),
            ("word_owner_i32", word_owner_i32),
            ("word_left_incidence_i32", word_left_incidence_i32),
            ("word_right_incidence_i32", word_right_incidence_i32),
            ("track_incidence_offsets_i32", track_incidence_offsets_i32),
            ("incidence_boundary_i32", incidence_boundary_i32),
            ("boundary_site_pairs_i32", boundary_site_pairs_i32),
        ):
            _assert_exact_quantized_tensor(
                name,
                actual,
                _training_bound_tensor(self, name),
                dtype=torch.int32,
            )

    def assert_prepared_immutable(self, prepared: Any) -> None:
        """Accept a refreshed material snapshot only if its immutable core matches.

        A trainer may either reuse the original prepared object after its source
        material tensors advance, or build a fresh compact material snapshot.
        Neither path may change topology, geometry, rays, chart intervals, or
        node/interpolation schedules.
        """

        self.assert_current()
        candidate = _named_training_immutable_tensors(prepared)
        if tuple(name for name, _ in candidate) != self._bound_tensor_names:
            raise ValueError("refreshed training snapshot immutable tensor layout changed")
        for (name, actual), expected in zip(candidate, self._bound_tensors, strict=True):
            _assert_same_tensor_content(
                f"refreshed training snapshot {name}",
                actual,
                expected,
            )
        if len(prepared.world_snapshot.atlas.charts) != len(self.charts):
            raise ValueError("refreshed training snapshot chart count changed")
        for index, actual in enumerate(prepared.world_snapshot.atlas.charts):
            schedule = self.charts[index]
            if (
                actual.transfer_atlas.t_min != schedule.t_min
                or actual.transfer_atlas.t_max != schedule.t_max
                or actual.node_count != schedule.node_count
                or actual.near != schedule.near
                or actual.far != schedule.far
            ):
                raise ValueError("refreshed training snapshot chart partition changed")

    def assert_native_world(
        self,
        *,
        sites_f32: Tensor,
        site_rgba_f32: Tensor,
        track_ray_coeff_f32: Tensor,
    ) -> None:
        """Check immutable geometry/rays while intentionally accepting live RGBA."""

        self.assert_current()
        _assert_exact_quantized_tensor(
            "sites_f32",
            sites_f32,
            _training_bound_tensor(self, "site_geometry"),
            dtype=torch.float32,
        )
        _assert_exact_quantized_tensor(
            "track_ray_coeff_f32",
            track_ray_coeff_f32,
            _training_bound_tensor(self, "ray_coefficients"),
            dtype=torch.float32,
        )
        if site_rgba_f32.ndim != 2 or tuple(site_rgba_f32.shape) != (
            self.site_count,
            4,
        ):
            raise ValueError("live site_rgba_f32 must have shape [site_count,4]")

    def assert_replay_interval(self, *, near: float, far: float) -> None:
        self.assert_current()
        for chart in self.charts:
            if float(near) != chart.near or float(far) != chart.far:
                raise ValueError("native replay near/far interval does not match the training schedule")

    def assert_native_chart(
        self,
        chart_index: int,
        compiler_node_t_f32: Tensor,
    ) -> NativeFixedWordP0TrainingChartSchedule:
        self.assert_current()
        if chart_index < 0 or chart_index >= len(self.charts):
            raise ValueError("chart_index is outside the training topology binding")
        chart = self.charts[chart_index]
        _assert_exact_quantized_tensor(
            "compiler_node_t_f32",
            compiler_node_t_f32,
            _training_bound_tensor(self, f"chart[{chart_index}].node_times"),
            dtype=torch.float32,
        )
        return chart

    def validate_sample_times(self, chart_index: int, sample_t_f64: Tensor) -> Tensor:
        self.assert_current()
        if chart_index < 0 or chart_index >= len(self.charts):
            raise ValueError("chart_index is outside the training topology binding")
        times = torch.as_tensor(sample_t_f64, dtype=torch.float64, device="cpu").reshape(-1)
        if times.numel() == 0 or not bool(torch.isfinite(times).all().item()):
            raise ValueError("sample times must be nonempty finite float64 values")
        chart = self.charts[chart_index]
        final_chart = chart_index + 1 == len(self.charts)
        if bool(torch.any(times < chart.t_min).item()) or bool(
            torch.any(times > chart.t_max if final_chart else times >= chart.t_max).item()
        ):
            bracket = "[t_min,t_max]" if final_chart else "[t_min,t_max)"
            raise ValueError(f"sample times leave training chart interval {bracket}")
        return times

    def sample_to_node_weights(self, chart_index: int, sample_t_f64: Tensor) -> Tensor:
        return self.sample_to_node_weight_result(chart_index, sample_t_f64).weights.to(
            dtype=torch.float32
        ).contiguous()

    def sample_to_node_weight_result(self, chart_index: int, sample_t_f64: Tensor) -> Any:
        """Return weights plus explicit linear/fallback cost provenance."""
        from compact_lie_schedule import fit_derived_sample_to_node_weights

        times = self.validate_sample_times(chart_index, sample_t_f64)
        chart = self.charts[chart_index]
        return fit_derived_sample_to_node_weights(
            times,
            t_min=chart.t_min,
            t_max=chart.t_max,
            node_times=_training_bound_tensor(self, f"chart[{chart_index}].node_times"),
            fit_matrix=_training_bound_tensor(self, f"chart[{chart_index}].fit_matrix"),
            barycentric_weights=self._sample_barycentric_weights[chart_index],
        )


def certify_and_bind_native_fixed_word_p0(
    prepared: Any,
    *,
    policy: Any,
) -> NativeFixedWordP0ContinuousCertificateBinding:
    """Run real CPU continuous acceptance, then seal its canonical facts."""
    from continuous_adaptive_lie_acceptance import certify_prepared_adaptive_lie_world

    prepared.assert_current()
    acceptance = certify_prepared_adaptive_lie_world(
        prepared.world_snapshot,
        policy=policy,
        sites=prepared.site_geometry,
        boundary_pairs=prepared.topology.boundary_site_pairs_i32,
    )
    return _bind_passed_acceptance(prepared, acceptance)


def assert_native_fixed_word_p0_certificate_binding(
    binding: object,
) -> NativeFixedWordP0ContinuousCertificateBinding:
    if type(binding) is not NativeFixedWordP0ContinuousCertificateBinding:
        raise ValueError("expected a sealed native continuous certificate binding")
    binding.assert_current()
    return binding


def derive_native_fixed_word_p0_training_topology_binding(
    strict_binding: NativeFixedWordP0ContinuousCertificateBinding,
) -> NativeFixedWordP0TrainingTopologyBinding:
    """Narrow a passed frozen-evaluation binding to material-only training.

    This convenience path validates the strict binding once and discards its
    transfer/Jacobian facts. Production-size training should use
    :func:`certify_and_bind_native_fixed_word_p0_training_topology`, whose
    owner-only proof has no dense world-Jacobian dimension.
    """

    strict = assert_native_fixed_word_p0_certificate_binding(strict_binding)
    prepared = strict._prepared
    topology_generation = _training_topology_generation(prepared)
    if topology_generation != strict.topology_snapshot_generation:
        raise ValueError("strict binding topology generation changed before training derivation")
    evidence = tuple(
        _TrainingOwnerEvidence(
            chart_index=chart.chart_index,
            certificate_digest=chart.owner_identity_certificate_digest,
            ownership_tolerance=chart.owner_identity_tolerance,
            leaf_count=chart.owner_certificate_leaf_count,
            checked_endpoint_inequality_count=chart.owner_checked_endpoint_inequality_count,
            maximum_owner_difference_upper_bound=chart.maximum_owner_difference_upper_bound,
        )
        for chart in strict.charts
    )
    return _bind_native_training_topology(
        prepared,
        owner_evidence=evidence,
        topology_generation=topology_generation,
        evidence_source="narrowed_passed_strict_continuous_binding",
    )


def certify_and_bind_native_fixed_word_p0_training_topology(
    prepared: Any,
    *,
    ownership_tolerance: float = 1.0e-9,
    denominator_epsilon: float = 1.0e-9,
    segment_length_epsilon: float = 1.0e-8,
    max_split_depth: int = 14,
    max_leaf_count: int = 4096,
    max_work_units: int = 2_000_000,
    arithmetic_fraction_bits: int = 112,
) -> NativeFixedWordP0TrainingTopologyBinding:
    """Run only the continuous all-site owner proof and seal training topology.

    This path never constructs the generic transfer/world-Jacobian interval AD
    state. Its memory depends on owner-certificate interval work plus compact
    immutable topology/geometry/ray/schedule copies, not on a dense derivative
    dimension over all tracks. Mutable density, color, transfer coefficients,
    node charts, and the full prepared/template objects are not retained.
    """

    from continuous_owner_identity_certificate import certify_fixed_word_owner_identity

    prepared.assert_current()
    evidence = []
    for chart_index, chart in enumerate(prepared.world_snapshot.atlas.charts):
        certificate = certify_fixed_word_owner_identity(
            sites=prepared.site_geometry,
            boundary=prepared.world_snapshot.boundary,
            ray_coefficients=prepared.world_snapshot.ray_coefficients,
            words=chart.words,
            t_min=chart.transfer_atlas.t_min,
            t_max=chart.transfer_atlas.t_max,
            near=chart.near,
            far=chart.far,
            ownership_tolerance=ownership_tolerance,
            denominator_epsilon=denominator_epsilon,
            segment_length_epsilon=segment_length_epsilon,
            max_split_depth=max_split_depth,
            max_leaf_count=max_leaf_count,
            max_work_units=max_work_units,
            arithmetic_fraction_bits=arithmetic_fraction_bits,
        )
        if (
            not certificate.passed
            or not certificate.continuous_time_coverage
            or not certificate.owner_identity_certified
            or not certificate.all_competitor_sites_checked
            or certificate.runtime_floating_point_roundoff_certified
        ):
            raise ValueError("owner-only training topology certificate failed closed")
        evidence.append(
            _TrainingOwnerEvidence(
                chart_index=chart_index,
                certificate_digest=_canonical_digest(asdict(certificate)),
                ownership_tolerance=certificate.ownership_tolerance,
                leaf_count=certificate.leaf_count,
                checked_endpoint_inequality_count=(
                    certificate.checked_endpoint_inequality_count
                ),
                maximum_owner_difference_upper_bound=(
                    certificate.maximum_owner_difference_upper_bound
                ),
            )
        )
    prepared.assert_current()
    return _bind_native_training_topology(
        prepared,
        owner_evidence=tuple(evidence),
        topology_generation=_training_topology_generation(prepared),
        evidence_source="direct_continuous_all_site_owner_certificate",
    )


def _bind_native_training_topology(
    prepared: Any,
    *,
    owner_evidence: tuple[_TrainingOwnerEvidence, ...],
    topology_generation: str,
    evidence_source: str,
) -> NativeFixedWordP0TrainingTopologyBinding:
    from compact_lie_schedule import certify_fit_derived_barycentric_weights

    prepared.assert_current()
    charts = prepared.world_snapshot.atlas.charts
    if len(owner_evidence) != len(charts) or not owner_evidence:
        raise ValueError("owner evidence must cover every nonempty training chart")
    if tuple(record.chart_index for record in owner_evidence) != tuple(range(len(charts))):
        raise ValueError("owner evidence chart indices must be canonical and complete")
    tolerances = {record.ownership_tolerance for record in owner_evidence}
    if len(tolerances) != 1:
        raise ValueError("owner evidence must use one global ownership tolerance")
    owner_tolerance = next(iter(tolerances))
    for record in owner_evidence:
        if (
            len(record.certificate_digest) != 64
            or record.leaf_count < 1
            or record.checked_endpoint_inequality_count < 1
            or record.maximum_owner_difference_upper_bound > owner_tolerance
        ):
            raise ValueError("owner evidence facts are incomplete or failed")

    source_named_tensors = _named_training_immutable_tensors(prepared)
    named_tensors = tuple(
        (name, tensor.detach().cpu().clone().contiguous())
        for name, tensor in source_named_tensors
    )
    sample_barycentric_weights = tuple(
        certify_fit_derived_barycentric_weights(
            chart.transfer_atlas.node_times,
            chart.transfer_atlas.fit_matrix,
            t_min=float(chart.transfer_atlas.t_min),
            t_max=float(chart.transfer_atlas.t_max),
        )
        for chart in charts
    )
    tensor_payload = [(name, _tensor_content_digest(tensor)) for name, tensor in named_tensors]
    training_snapshot_generation = _canonical_digest(
        {
            "schema": "worldfoam-native-fixed-word-p0-training-snapshot-v1",
            "topology_snapshot_generation": topology_generation,
            "immutable_tensors": tensor_payload,
        }
    )
    chart_schedules = tuple(
        NativeFixedWordP0TrainingChartSchedule(
            chart_index=index,
            t_min=float(chart.transfer_atlas.t_min),
            t_max=float(chart.transfer_atlas.t_max),
            near=float(chart.near),
            far=float(chart.far),
            node_count=int(chart.node_count),
            chart_digest=_canonical_digest(
                {
                    "training_snapshot_generation": training_snapshot_generation,
                    "chart_index": index,
                    "t_min": chart.transfer_atlas.t_min,
                    "t_max": chart.transfer_atlas.t_max,
                    "near": chart.near,
                    "far": chart.far,
                    "node_count": chart.node_count,
                    "owner_identity_certificate_digest": evidence.certificate_digest,
                }
            ),
            owner_identity_certificate_digest=evidence.certificate_digest,
            owner_identity_tolerance=evidence.ownership_tolerance,
            owner_certificate_leaf_count=evidence.leaf_count,
            owner_checked_endpoint_inequality_count=(
                evidence.checked_endpoint_inequality_count
            ),
            maximum_owner_difference_upper_bound=(
                evidence.maximum_owner_difference_upper_bound
            ),
        )
        for index, (chart, evidence) in enumerate(zip(charts, owner_evidence, strict=True))
    )
    maximum_owner = max(record.maximum_owner_difference_upper_bound for record in owner_evidence)
    total_leaves = sum(record.leaf_count for record in owner_evidence)
    binding_facts = {
        "schema": "worldfoam-native-fixed-word-p0-training-topology-binding-v2",
        "canonical_scope": "owner_topology_geometry_rays_chart_schedule_only",
        "topology_snapshot_generation": topology_generation,
        "training_snapshot_generation": training_snapshot_generation,
        "site_count": int(prepared.topology.site_count),
        "charts": [asdict(chart) for chart in chart_schedules],
        "owner_identity_certified": True,
        "owner_identity_scope": "all_competitor_sites_continuously_certified",
        "owner_identity_tolerance": owner_tolerance,
        "maximum_owner_difference_upper_bound": maximum_owner,
        "total_owner_certificate_leaves": total_leaves,
        "binding_mode": "training_owner_topology_only",
        "transfer_jacobian_certified": False,
        "approximation_error_certified": False,
        "paper_evidence_eligible": False,
        "geometry_rays_immutable": True,
        "live_site_rgba_refresh_allowed": True,
        "runtime_floating_point_roundoff_certified": False,
        "sample_weight_evaluation": "verified_fit_derived_second_form_barycentric",
    }
    payload_json = _canonical_json(
        {
            "binding": binding_facts,
            "immutable_tensor_content": tensor_payload,
            "owner_certificate_source": evidence_source,
            "owner_certificate_digests": [
                record.certificate_digest for record in owner_evidence
            ],
            "sample_barycentric_weights": [
                _tensor_content_digest(tensor) for tensor in sample_barycentric_weights
            ],
        }
    )
    bound_tensors = tuple(tensor for _, tensor in named_tensors)
    return NativeFixedWordP0TrainingTopologyBinding(
        canonical_digest=hashlib.sha256(payload_json.encode("utf-8")).hexdigest(),
        topology_snapshot_generation=topology_generation,
        training_snapshot_generation=training_snapshot_generation,
        site_count=int(prepared.topology.site_count),
        charts=chart_schedules,
        owner_identity_certified=True,
        owner_identity_scope="all_competitor_sites_continuously_certified",
        owner_identity_tolerance=owner_tolerance,
        maximum_owner_difference_upper_bound=maximum_owner,
        total_owner_certificate_leaves=total_leaves,
        binding_mode="training_owner_topology_only",
        transfer_jacobian_certified=False,
        approximation_error_certified=False,
        paper_evidence_eligible=False,
        geometry_rays_immutable=True,
        live_site_rgba_refresh_allowed=True,
        runtime_floating_point_roundoff_certified=False,
        _canonical_payload_json=payload_json,
        _bound_tensor_names=tuple(name for name, _ in named_tensors),
        _bound_tensors=bound_tensors,
        _bound_tensor_signatures=tuple(_tensor_signature(tensor) for tensor in bound_tensors),
        _sample_barycentric_weights=sample_barycentric_weights,
        _sample_barycentric_signatures=tuple(
            _tensor_signature(tensor) for tensor in sample_barycentric_weights
        ),
        _seal=_TRAINING_BINDING_SEAL,
    )


def assert_native_fixed_word_p0_training_topology_binding(
    binding: object,
) -> NativeFixedWordP0TrainingTopologyBinding:
    if type(binding) is not NativeFixedWordP0TrainingTopologyBinding:
        raise ValueError("expected a sealed native material-training topology binding")
    binding.assert_current()
    return binding


NativeFixedWordP0RuntimeBinding = (
    NativeFixedWordP0ContinuousCertificateBinding | NativeFixedWordP0TrainingTopologyBinding
)


def assert_native_fixed_word_p0_runtime_binding(
    binding: object,
) -> NativeFixedWordP0RuntimeBinding:
    """Validate either capability without allowing one to impersonate the other."""

    if type(binding) is NativeFixedWordP0ContinuousCertificateBinding:
        return assert_native_fixed_word_p0_certificate_binding(binding)
    if type(binding) is NativeFixedWordP0TrainingTopologyBinding:
        return assert_native_fixed_word_p0_training_topology_binding(binding)
    raise ValueError("expected a sealed native fixed-word P0 runtime binding")


def _bind_passed_acceptance(
    prepared: Any,
    acceptance: Any,
) -> NativeFixedWordP0ContinuousCertificateBinding:
    from compact_lie_schedule import certify_fit_derived_barycentric_weights

    prepared.assert_current()
    reasons = []
    for field_name in (
        "passed",
        "continuous_time_coverage",
        "atlas_world_provenance_certified",
        "optimizer_site_geometry_covered",
        "optimizer_site_geometry_accepted",
        "atlas_selection_was_not_reperformed",
        "owner_identity_certified",
    ):
        if not bool(getattr(acceptance, field_name)):
            reasons.append(field_name)
    if bool(acceptance.continuous_acceptance_used_sampling):
        reasons.append("continuous_acceptance_used_sampling")
    if bool(acceptance.runtime_floating_point_roundoff_certified):
        reasons.append("runtime_floating_point_roundoff_certified_must_remain_false")
    if reasons:
        raise ValueError("native continuous certificate acceptance failed closed: " + ", ".join(reasons))
    if acceptance.policy.site_geometry_jacobian_tolerance is None:
        raise ValueError("native site geometry requires an explicit certified site-Jacobian tolerance")
    if acceptance.policy.owner_identity_tolerance is None:
        raise ValueError("native fixed-word topology requires an explicit owner-identity tolerance")
    if len(acceptance.charts) != len(prepared.world_snapshot.atlas.charts):
        raise ValueError("continuous certificate chart count does not match the prepared snapshot")
    if any(not chart.passed or chart.certificate is None for chart in acceptance.charts):
        raise ValueError("every selected chart must have a passed continuous certificate")
    owner_certificates = tuple(chart.owner_identity_certificate for chart in acceptance.charts)
    if any(
        certificate is None
        or not certificate.passed
        or not certificate.continuous_time_coverage
        or not certificate.owner_identity_certified
        or not certificate.all_competitor_sites_checked
        or certificate.runtime_floating_point_roundoff_certified
        for certificate in owner_certificates
    ):
        raise ValueError("every selected chart must have a passed all-site owner-identity certificate")
    owner_leaf_count = sum(certificate.leaf_count for certificate in owner_certificates)
    owner_maximum = max(
        certificate.maximum_owner_difference_upper_bound for certificate in owner_certificates
    )
    if (
        acceptance.total_owner_certificate_leaves != owner_leaf_count
        or acceptance.maximum_owner_difference_upper_bound != owner_maximum
        or owner_maximum > acceptance.policy.owner_identity_tolerance
    ):
        raise ValueError("aggregate owner-identity certificate facts are inconsistent")

    topology_tensors = _named_topology_tensors(prepared)
    world_tensors = _named_world_tensors(prepared)
    topology_payload = {
        "tensors": [(name, _tensor_content_digest(tensor)) for name, tensor in topology_tensors],
        "track_count": prepared.topology.track_count,
        "boundary_count": prepared.topology.boundary_count,
        "site_count": prepared.topology.site_count,
        "word_count": prepared.topology.word_count,
        "incidence_count": prepared.topology.incidence_count,
    }
    topology_generation = _canonical_digest(topology_payload)
    world_payload = {
        "topology_snapshot_generation": topology_generation,
        "tensors": [(name, _tensor_content_digest(tensor)) for name, tensor in world_tensors],
    }
    world_generation = _canonical_digest(world_payload)
    chart_bindings = []
    for index, chart_acceptance in enumerate(acceptance.charts):
        owner_certificate = chart_acceptance.owner_identity_certificate
        assert owner_certificate is not None
        owner_payload = asdict(owner_certificate)
        chart_bindings.append(
            NativeFixedWordP0ChartCertificate(
                chart_index=index,
                t_min=float(chart_acceptance.t_min),
                t_max=float(chart_acceptance.t_max),
                node_count=int(chart_acceptance.node_count),
                chart_digest=_canonical_digest(
                    {
                        "world_snapshot_generation": world_generation,
                        "chart_index": index,
                        "t_min": chart_acceptance.t_min,
                        "t_max": chart_acceptance.t_max,
                        "node_count": chart_acceptance.node_count,
                        "acceptance": asdict(chart_acceptance),
                    }
                ),
                estimated_interval_jet_work_units=int(
                    chart_acceptance.estimated_interval_jet_work_units
                ),
                owner_identity_certificate_digest=_canonical_digest(owner_payload),
                owner_identity_tolerance=float(owner_certificate.ownership_tolerance),
                owner_certificate_leaf_count=int(owner_certificate.leaf_count),
                owner_certificate_deepest_split=int(owner_certificate.deepest_split),
                owner_checked_endpoint_inequality_count=int(
                    owner_certificate.checked_endpoint_inequality_count
                ),
                maximum_owner_difference_upper_bound=float(
                    owner_certificate.maximum_owner_difference_upper_bound
                ),
                minimum_certified_owner_margin=float(
                    owner_certificate.minimum_certified_owner_margin
                ),
            )
        )
    chart_bindings = tuple(chart_bindings)
    sample_barycentric_weights = tuple(
        certify_fit_derived_barycentric_weights(
            chart.transfer_atlas.node_times,
            chart.transfer_atlas.fit_matrix,
            t_min=float(chart.transfer_atlas.t_min),
            t_max=float(chart.transfer_atlas.t_max),
        )
        for chart in prepared.world_snapshot.atlas.charts
    )
    binding_facts = {
        "schema": "worldfoam-native-fixed-word-p0-continuous-binding-v3",
        "passed": True,
        "topology_snapshot_generation": topology_generation,
        "world_snapshot_generation": world_generation,
        "charts": [asdict(chart) for chart in chart_bindings],
        "transfer_tolerance": float(acceptance.policy.transfer_tolerance),
        "world_jacobian_tolerance": float(acceptance.policy.world_jacobian_tolerance),
        "site_geometry_jacobian_tolerance": float(
            acceptance.policy.site_geometry_jacobian_tolerance
        ),
        "max_split_depth": int(acceptance.policy.max_split_depth),
        "max_leaves_per_chart": int(acceptance.policy.max_leaves_per_chart),
        "max_interval_jet_work_units_per_chart": int(
            acceptance.policy.max_interval_jet_work_units_per_chart
        ),
        "arithmetic_fraction_bits": int(acceptance.policy.arithmetic_fraction_bits),
        "owner_identity_certified": True,
        "owner_identity_scope": "all_competitor_sites_continuously_certified",
        "owner_identity_tolerance": float(acceptance.policy.owner_identity_tolerance),
        "owner_max_split_depth": int(acceptance.policy.owner_max_split_depth),
        "owner_max_leaves_per_chart": int(acceptance.policy.owner_max_leaves_per_chart),
        "owner_max_work_units_per_chart": int(acceptance.policy.owner_max_work_units_per_chart),
        "maximum_owner_difference_upper_bound": float(
            acceptance.maximum_owner_difference_upper_bound
        ),
        "total_owner_certificate_leaves": int(acceptance.total_owner_certificate_leaves),
        "runtime_floating_point_roundoff_certified": False,
        "sample_weight_evaluation": "verified_fit_derived_second_form_barycentric",
    }
    payload = {
        "binding": binding_facts,
        "policy": asdict(acceptance.policy),
        "acceptance": asdict(acceptance),
        "sample_barycentric_weights": [
            _tensor_content_digest(tensor) for tensor in sample_barycentric_weights
        ],
    }
    payload_json = _canonical_json(payload)
    bound_tensors = tuple(tensor for _, tensor in (*topology_tensors, *world_tensors))
    return NativeFixedWordP0ContinuousCertificateBinding(
        passed=True,
        canonical_digest=hashlib.sha256(payload_json.encode("utf-8")).hexdigest(),
        topology_snapshot_generation=topology_generation,
        world_snapshot_generation=world_generation,
        charts=chart_bindings,
        transfer_tolerance=float(acceptance.policy.transfer_tolerance),
        world_jacobian_tolerance=float(acceptance.policy.world_jacobian_tolerance),
        site_geometry_jacobian_tolerance=float(acceptance.policy.site_geometry_jacobian_tolerance),
        max_split_depth=int(acceptance.policy.max_split_depth),
        max_leaves_per_chart=int(acceptance.policy.max_leaves_per_chart),
        max_interval_jet_work_units_per_chart=int(
            acceptance.policy.max_interval_jet_work_units_per_chart
        ),
        arithmetic_fraction_bits=int(acceptance.policy.arithmetic_fraction_bits),
        owner_identity_certified=True,
        owner_identity_scope="all_competitor_sites_continuously_certified",
        owner_identity_tolerance=float(acceptance.policy.owner_identity_tolerance),
        owner_max_split_depth=int(acceptance.policy.owner_max_split_depth),
        owner_max_leaves_per_chart=int(acceptance.policy.owner_max_leaves_per_chart),
        owner_max_work_units_per_chart=int(acceptance.policy.owner_max_work_units_per_chart),
        maximum_owner_difference_upper_bound=float(
            acceptance.maximum_owner_difference_upper_bound
        ),
        total_owner_certificate_leaves=int(acceptance.total_owner_certificate_leaves),
        runtime_floating_point_roundoff_certified=False,
        _canonical_payload_json=payload_json,
        _prepared=prepared,
        _bound_tensors=bound_tensors,
        _bound_tensor_signatures=tuple(_tensor_signature(tensor) for tensor in bound_tensors),
        _sample_barycentric_weights=sample_barycentric_weights,
        _sample_barycentric_signatures=tuple(
            _tensor_signature(tensor) for tensor in sample_barycentric_weights
        ),
        _seal=_BINDING_SEAL,
    )


def _named_topology_tensors(prepared: Any) -> tuple[tuple[str, Tensor], ...]:
    topology = prepared.topology
    return tuple(
        (name, getattr(topology, name))
        for name in (
            "source_track_ids",
            "source_boundary_ids",
            "source_site_ids",
            "word_offsets_i32",
            "word_owner_i32",
            "word_left_incidence_i32",
            "word_right_incidence_i32",
            "track_incidence_offsets_i32",
            "incidence_boundary_i32",
            "boundary_site_pairs_i32",
        )
    )


def _named_world_tensors(prepared: Any) -> tuple[tuple[str, Tensor], ...]:
    snapshot = prepared.world_snapshot
    tensors: list[tuple[str, Tensor]] = [
        ("site_geometry", prepared.site_geometry),
        ("boundary", snapshot.boundary),
        ("ray_coefficients", snapshot.ray_coefficients),
        ("site_density", snapshot.site_density),
        ("site_color", snapshot.site_color),
    ]
    for chart_index, chart in enumerate(snapshot.atlas.charts):
        for name, tensor in (
            ("node_times", chart.transfer_atlas.node_times),
            ("fit_matrix", chart.transfer_atlas.fit_matrix),
            ("coefficients", chart.transfer_atlas.coefficients),
            ("node_chart", chart.node_chart),
            ("depth_coefficient_incidence", chart.depth_coefficient_incidence),
            ("sparse_depth_coefficients", chart.sparse_depth_coefficients),
        ):
            tensors.append((f"chart[{chart_index}].{name}", tensor))
        for word_index, word in enumerate(chart.words):
            tensors.extend(
                (
                    (f"chart[{chart_index}].word[{word_index}].owners", word.owners),
                    (f"chart[{chart_index}].word[{word_index}].left", word.left_cut_ids),
                    (f"chart[{chart_index}].word[{word_index}].right", word.right_cut_ids),
                )
            )
    return tuple(tensors)


def _named_training_immutable_tensors(prepared: Any) -> tuple[tuple[str, Tensor], ...]:
    """Return exactly the tensors retained by material-only training.

    Site density/color, compiled transfer coefficients, and node chart values
    are intentionally absent: all may change or be recomputed after an RGBA
    optimizer step.  The fit matrix is retained because it is part of the
    immutable interpolation schedule, not a material-dependent fit result.
    """

    tensors = list(_named_topology_tensors(prepared))
    tensors.extend(
        (
            ("site_geometry", prepared.site_geometry),
            ("ray_coefficients", prepared.world_snapshot.ray_coefficients),
        )
    )
    for chart_index, chart in enumerate(prepared.world_snapshot.atlas.charts):
        tensors.extend(
            (
                (f"chart[{chart_index}].node_times", chart.transfer_atlas.node_times),
                (f"chart[{chart_index}].fit_matrix", chart.transfer_atlas.fit_matrix),
            )
        )
    return tuple(tensors)


def _training_topology_generation(prepared: Any) -> str:
    topology_tensors = _named_topology_tensors(prepared)
    return _canonical_digest(
        {
            "tensors": [
                (name, _tensor_content_digest(tensor))
                for name, tensor in topology_tensors
            ],
            "track_count": prepared.topology.track_count,
            "boundary_count": prepared.topology.boundary_count,
            "site_count": prepared.topology.site_count,
            "word_count": prepared.topology.word_count,
            "incidence_count": prepared.topology.incidence_count,
        }
    )


def _training_bound_tensor(
    binding: NativeFixedWordP0TrainingTopologyBinding,
    name: str,
) -> Tensor:
    try:
        index = binding._bound_tensor_names.index(name)
    except ValueError as error:
        raise ValueError(f"training topology binding has no immutable tensor {name}") from error
    return binding._bound_tensors[index]


def _tensor_signature(tensor: Tensor) -> tuple[Any, ...]:
    return (
        tensor.untyped_storage().data_ptr(),
        tensor._version,
        tuple(tensor.shape),
        tuple(tensor.stride()),
        tensor.storage_offset(),
        tensor.dtype,
        tensor.device,
    )


def _tensor_content_digest(tensor: Tensor) -> str:
    cpu = tensor.detach().cpu().contiguous()
    header = _canonical_json(
        {
            "dtype": str(cpu.dtype),
            "shape": list(cpu.shape),
        }
    ).encode("utf-8")
    payload = cpu.numpy().tobytes(order="C")
    return hashlib.sha256(header + b"\0" + payload).hexdigest()


def _assert_exact_quantized_tensor(name: str, actual: Tensor, reference: Tensor, *, dtype: torch.dtype) -> None:
    actual_cpu = actual.detach().cpu().to(dtype=dtype).contiguous()
    expected_cpu = reference.detach().cpu().to(dtype=dtype).contiguous()
    if actual_cpu.shape != expected_cpu.shape or not torch.equal(actual_cpu, expected_cpu):
        raise ValueError(f"{name} does not match the certified compact snapshot")


def _assert_same_tensor_content(name: str, actual: Tensor, reference: Tensor) -> None:
    actual_cpu = actual.detach().cpu().contiguous()
    expected_cpu = reference.detach().cpu().contiguous()
    if (
        actual_cpu.dtype != expected_cpu.dtype
        or actual_cpu.shape != expected_cpu.shape
        or not torch.equal(actual_cpu, expected_cpu)
    ):
        raise ValueError(f"{name} does not match the immutable training snapshot")


def _canonical_digest(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


__all__ = [
    "NativeFixedWordP0ChartCertificate",
    "NativeFixedWordP0ContinuousCertificateBinding",
    "NativeFixedWordP0RuntimeBinding",
    "NativeFixedWordP0TrainingChartSchedule",
    "NativeFixedWordP0TrainingTopologyBinding",
    "assert_native_fixed_word_p0_certificate_binding",
    "assert_native_fixed_word_p0_runtime_binding",
    "assert_native_fixed_word_p0_training_topology_binding",
    "certify_and_bind_native_fixed_word_p0",
    "certify_and_bind_native_fixed_word_p0_training_topology",
    "derive_native_fixed_word_p0_training_topology_binding",
]

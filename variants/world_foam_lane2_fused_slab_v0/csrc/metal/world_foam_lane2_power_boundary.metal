#include <metal_stdlib>

#include "../shared/world_foam_lane2_types.h"

using namespace metal;

static inline bool wf2_power_valid_slab(const WF2PowerBeamSlab beam) {
  return isfinite(beam.u_center) && isfinite(beam.t0) && isfinite(beam.t1)
      && isfinite(beam.near_depth) && isfinite(beam.far_depth)
      && beam.far_depth >= beam.near_depth;
}

static inline bool wf2_power_depth_interval_overlaps(
    const WF2PowerBoundary3D boundary,
    const WF2PowerBeamSlab beam,
    constant WF2PowerBoundaryConfig& config,
    thread bool& invalid_denominator) {
  invalid_denominator = false;
  if (fabs(boundary.nz) < config.invalid_epsilon) {
    invalid_denominator = true;
    return false;
  }

  const float x0 = beam.u_center + config.camera_velocity_x * beam.t0;
  const float x1 = beam.u_center + config.camera_velocity_x * beam.t1;
  const float s0 = -(boundary.nx * x0 + boundary.nt * beam.t0 + boundary.b) / boundary.nz;
  const float s1 = -(boundary.nx * x1 + boundary.nt * beam.t1 + boundary.b) / boundary.nz;
  if (!isfinite(s0) || !isfinite(s1)) {
    invalid_denominator = true;
    return false;
  }

  const float s_min = min(s0, s1);
  const float s_max = max(s0, s1);
  return max(s_min, beam.near_depth) <= min(s_max, beam.far_depth);
}

kernel void wf2_count_power_boundary_events(
    device const WF2PowerBoundary3D* boundaries [[buffer(WF2_POWER_BUFFER_BOUNDARIES)]],
    device const WF2PowerBeamSlab* beams [[buffer(WF2_POWER_BUFFER_BEAMS)]],
    constant WF2PowerBoundaryConfig& config [[buffer(WF2_POWER_BUFFER_CONFIG)]],
    device WF2PowerBoundaryCount* counts [[buffer(WF2_POWER_BUFFER_COUNTS)]],
    uint beam_id [[thread_position_in_grid]]) {
  if (beam_id >= config.beam_count) {
    return;
  }

  const WF2PowerBeamSlab beam = beams[beam_id];
  WF2PowerBoundaryCount out;
  out.beam_id = beam_id;
  out.payload_id = beam.payload_id;
  out.boundary_event_count = 0u;
  out.invalid_denominator_count = 0u;
  out.flags = 0u;
  out.reserved0 = 0u;
  out.reserved1 = 0u;
  out.reserved2 = 0u;

  if (!wf2_power_valid_slab(beam)) {
    out.flags |= WF2_COUNT_FLAG_INVALID_BEAM;
    counts[beam_id] = out;
    return;
  }

  for (uint boundary_id = 0u; boundary_id < config.boundary_count; ++boundary_id) {
    bool invalid_denominator = false;
    const bool overlaps = wf2_power_depth_interval_overlaps(
        boundaries[boundary_id],
        beam,
        config,
        invalid_denominator);
    if (invalid_denominator) {
      out.invalid_denominator_count += 1u;
    } else if (overlaps) {
      out.boundary_event_count += 1u;
    }
  }

  if (out.invalid_denominator_count > 0u) {
    out.flags |= WF2_COUNT_FLAG_INVALID_DENOMINATOR;
  }
  counts[beam_id] = out;
}

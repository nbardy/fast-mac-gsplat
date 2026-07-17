#include <metal_stdlib>

using namespace metal;

#define WF2_COUNT_FLAG_INVALID_BEAM 1u
#define WF2_COUNT_FLAG_INVALID_DENOMINATOR 4u

static inline bool wf2_tensor_valid_slab(
    device const float* beam_f32,
    const uint beam_id) {
  const uint base = beam_id * 5u;
  const float u_center = beam_f32[base + 0u];
  const float t0 = beam_f32[base + 1u];
  const float t1 = beam_f32[base + 2u];
  const float near_depth = beam_f32[base + 3u];
  const float far_depth = beam_f32[base + 4u];
  return isfinite(u_center) && isfinite(t0) && isfinite(t1)
      && isfinite(near_depth) && isfinite(far_depth)
      && far_depth >= near_depth;
}

static inline bool wf2_tensor_depth_interval_overlaps(
    device const float* boundary_f32,
    device const float* beam_f32,
    const uint boundary_id,
    const uint beam_id,
    const float camera_velocity_x,
    const float invalid_epsilon,
    thread bool& invalid_denominator) {
  invalid_denominator = false;

  const uint boundary_base = boundary_id * 4u;
  const float nx = boundary_f32[boundary_base + 0u];
  const float nz = boundary_f32[boundary_base + 1u];
  const float nt = boundary_f32[boundary_base + 2u];
  const float b = boundary_f32[boundary_base + 3u];
  if (fabs(nz) < invalid_epsilon) {
    invalid_denominator = true;
    return false;
  }

  const uint beam_base = beam_id * 5u;
  const float u_center = beam_f32[beam_base + 0u];
  const float t0 = beam_f32[beam_base + 1u];
  const float t1 = beam_f32[beam_base + 2u];
  const float near_depth = beam_f32[beam_base + 3u];
  const float far_depth = beam_f32[beam_base + 4u];

  const float x0 = u_center + camera_velocity_x * t0;
  const float x1 = u_center + camera_velocity_x * t1;
  const float s0 = -(nx * x0 + nt * t0 + b) / nz;
  const float s1 = -(nx * x1 + nt * t1 + b) / nz;
  if (!isfinite(s0) || !isfinite(s1)) {
    invalid_denominator = true;
    return false;
  }

  const float s_min = min(s0, s1);
  const float s_max = max(s0, s1);
  return max(s_min, near_depth) <= min(s_max, far_depth);
}

kernel void wf2_count_power_boundary_events_tensor(
    device const float* boundary_f32 [[buffer(0)]],
    device const uint* boundary_u32 [[buffer(1)]],
    device const float* beam_f32 [[buffer(2)]],
    device const uint* beam_u32 [[buffer(3)]],
    device const int* config_i32 [[buffer(4)]],
    device const float* config_f32 [[buffer(5)]],
    device uint* counts_u32 [[buffer(6)]],
    uint beam_id [[thread_position_in_grid]]) {
  (void)boundary_u32;

  const uint boundary_count = uint(config_i32[0]);
  const uint beam_count = uint(config_i32[1]);
  const float camera_velocity_x = config_f32[0];
  const float invalid_epsilon = config_f32[1];
  if (beam_id >= beam_count) {
    return;
  }

  const uint beam_u32_base = beam_id * 4u;
  const uint count_base = beam_id * 8u;
  counts_u32[count_base + 0u] = beam_id;
  counts_u32[count_base + 1u] = beam_u32[beam_u32_base + 0u];
  counts_u32[count_base + 2u] = 0u;
  counts_u32[count_base + 3u] = 0u;
  counts_u32[count_base + 4u] = 0u;
  counts_u32[count_base + 5u] = 0u;
  counts_u32[count_base + 6u] = 0u;
  counts_u32[count_base + 7u] = 0u;

  if (!wf2_tensor_valid_slab(beam_f32, beam_id)) {
    counts_u32[count_base + 4u] |= WF2_COUNT_FLAG_INVALID_BEAM;
    return;
  }

  for (uint boundary_id = 0u; boundary_id < boundary_count; ++boundary_id) {
    bool invalid_denominator = false;
    const bool overlaps = wf2_tensor_depth_interval_overlaps(
        boundary_f32,
        beam_f32,
        boundary_id,
        beam_id,
        camera_velocity_x,
        invalid_epsilon,
        invalid_denominator);
    if (invalid_denominator) {
      counts_u32[count_base + 3u] += 1u;
    } else if (overlaps) {
      counts_u32[count_base + 2u] += 1u;
    }
  }

  if (counts_u32[count_base + 3u] > 0u) {
    counts_u32[count_base + 4u] |= WF2_COUNT_FLAG_INVALID_DENOMINATOR;
  }
}

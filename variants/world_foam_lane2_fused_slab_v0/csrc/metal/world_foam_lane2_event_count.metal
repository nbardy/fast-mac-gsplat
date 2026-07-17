#include <metal_stdlib>

#include "../shared/world_foam_lane2_types.h"

using namespace metal;

struct WF2AxisBoundaryRange {
  int first;
  int last;
};

static inline float3 wf2_load3(WF2Float3 value) {
  return float3(value.x, value.y, value.z);
}

static inline WF2Float3 wf2_store3(float3 value) {
  WF2Float3 out;
  out.x = value.x;
  out.y = value.y;
  out.z = value.z;
  return out;
}

static inline float wf2_axis_value(float3 value, uint axis) {
  if (axis == WF2_EVENT_AXIS_U) {
    return value.x;
  }
  if (axis == WF2_EVENT_AXIS_V) {
    return value.y;
  }
  return value.z;
}

static inline bool wf2_valid_vec3(float3 value) {
  return isfinite(value.x) && isfinite(value.y) && isfinite(value.z);
}

static inline WF2AxisBoundaryRange wf2_axis_boundary_range(
    float a,
    float b,
    float tile_size,
    uint tile_count) {
  WF2AxisBoundaryRange range;
  range.first = 1;
  range.last = 0;

  if (!(tile_size > 0.0f) || tile_count <= 1u || !(a != b)) {
    return range;
  }
  if (!isfinite(a) || !isfinite(b)) {
    return range;
  }

  const float lo = min(a, b);
  const float hi = max(a, b);
  const float eps = max(1.0e-5f, 1.0e-6f * max(max(fabs(lo), fabs(hi)), tile_size));
  const float hi_strict = hi - eps;

  int first = int(floor(lo / tile_size)) + 1;
  int last = int(floor(hi_strict / tile_size));
  first = max(first, 1);
  last = min(last, int(tile_count) - 1);

  range.first = first;
  range.last = last;
  return range;
}

static inline uint wf2_axis_crossing_count(
    float a,
    float b,
    float tile_size,
    uint tile_count) {
  const WF2AxisBoundaryRange range = wf2_axis_boundary_range(a, b, tile_size, tile_count);
  if (range.last < range.first) {
    return 0u;
  }
  return uint(range.last - range.first + 1);
}

static inline void wf2_emit_axis_events(
    uint axis,
    float a,
    float b,
    float3 start_uvt,
    float3 delta_uvt,
    float tile_size,
    uint tile_count,
    uint beam_id,
    uint payload_id,
    uint event_base,
    constant WF2GridConfig& grid,
    device WF2BoundaryEvent* events,
    thread uint& local_event_index) {
  const WF2AxisBoundaryRange range = wf2_axis_boundary_range(a, b, tile_size, tile_count);
  if (range.last < range.first) {
    return;
  }

  const float denom = b - a;
  for (int boundary = range.first; boundary <= range.last; ++boundary) {
    const float boundary_value = float(boundary) * tile_size;
    const float s = clamp((boundary_value - a) / denom, 0.0f, 1.0f);
    const uint global_index = event_base + local_event_index;

    if (global_index < grid.event_capacity) {
      WF2BoundaryEvent event;
      event.beam_id = beam_id;
      event.payload_id = payload_id;
      event.axis = axis;
      event.boundary_index = uint(boundary);
      event.s = s;
      event.uvt = wf2_store3(start_uvt + s * delta_uvt);
      event.flags = (b > a) ? WF2_EVENT_FLAG_POSITIVE_DIRECTION : 0u;
      event.reserved0 = 0u;
      events[global_index] = event;
    }

    local_event_index += 1u;
  }
}

kernel void wf2_count_screen_time_beam_events(
    device const WF2ScreenTimeBeam* beams [[buffer(WF2_BUFFER_BEAMS)]],
    constant WF2GridConfig& grid [[buffer(WF2_BUFFER_GRID)]],
    device WF2BeamEventCount* counts [[buffer(WF2_BUFFER_COUNTS)]],
    device atomic_uint* global_event_count [[buffer(WF2_BUFFER_GLOBAL_EVENT_COUNT)]],
    device WF2BoundaryEvent* events [[buffer(WF2_BUFFER_EVENTS)]],
    uint beam_id [[thread_position_in_grid]]) {
  if (beam_id >= grid.beam_count) {
    return;
  }

  const WF2ScreenTimeBeam beam = beams[beam_id];
  const float3 start_uvt = wf2_load3(beam.start_uvt);
  const float3 end_uvt = wf2_load3(beam.end_uvt);
  const float3 delta_uvt = end_uvt - start_uvt;

  WF2BeamEventCount out;
  out.beam_id = beam_id;
  out.payload_id = beam.payload_id;
  out.u_crossings = 0u;
  out.v_crossings = 0u;
  out.t_crossings = 0u;
  out.total_crossings = 0u;
  out.first_event_index = WF2_NO_EVENT_INDEX;
  out.flags = 0u;

  const bool valid_beam = wf2_valid_vec3(start_uvt) && wf2_valid_vec3(end_uvt);
  if (!valid_beam) {
    out.flags |= WF2_COUNT_FLAG_INVALID_BEAM;
    counts[beam_id] = out;
    return;
  }

  out.u_crossings = wf2_axis_crossing_count(
      start_uvt.x, end_uvt.x, grid.tile_size_u, grid.tile_count_u);
  out.v_crossings = wf2_axis_crossing_count(
      start_uvt.y, end_uvt.y, grid.tile_size_v, grid.tile_count_v);
  out.t_crossings = wf2_axis_crossing_count(
      start_uvt.z, end_uvt.z, grid.tile_size_t, grid.tile_count_t);
  out.total_crossings = out.u_crossings + out.v_crossings + out.t_crossings;

  const bool write_events = (grid.flags & WF2_GRID_FLAG_WRITE_EVENTS) != 0u;
  uint event_base = 0u;
  if (out.total_crossings > 0u) {
    event_base = atomic_fetch_add_explicit(
        global_event_count, out.total_crossings, memory_order_relaxed);
    if (write_events) {
      out.first_event_index = event_base;
      if (event_base + out.total_crossings > grid.event_capacity) {
        out.flags |= WF2_COUNT_FLAG_EVENT_OVERFLOW;
      }
    }
  }

  counts[beam_id] = out;

  if (!write_events || out.total_crossings == 0u) {
    return;
  }

  uint local_event_index = 0u;
  wf2_emit_axis_events(
      WF2_EVENT_AXIS_U,
      start_uvt.x,
      end_uvt.x,
      start_uvt,
      delta_uvt,
      grid.tile_size_u,
      grid.tile_count_u,
      beam_id,
      beam.payload_id,
      event_base,
      grid,
      events,
      local_event_index);
  wf2_emit_axis_events(
      WF2_EVENT_AXIS_V,
      start_uvt.y,
      end_uvt.y,
      start_uvt,
      delta_uvt,
      grid.tile_size_v,
      grid.tile_count_v,
      beam_id,
      beam.payload_id,
      event_base,
      grid,
      events,
      local_event_index);
  wf2_emit_axis_events(
      WF2_EVENT_AXIS_T,
      start_uvt.z,
      end_uvt.z,
      start_uvt,
      delta_uvt,
      grid.tile_size_t,
      grid.tile_count_t,
      beam_id,
      beam.payload_id,
      event_base,
      grid,
      events,
      local_event_index);
}

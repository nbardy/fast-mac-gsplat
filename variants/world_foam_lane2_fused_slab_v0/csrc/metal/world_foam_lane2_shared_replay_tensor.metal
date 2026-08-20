#include <metal_stdlib>

using namespace metal;

#define WF2_MAX_SHARED_BOUNDARIES 31u
#define WF2_MAX_SHARED_SITES 32u
#define WF2_MAX_SHARED_SEGMENTS 32u
#define WF2_MAX_REALRAY_BOUNDARIES 128u
#define WF2_MAX_REALRAY_SEGMENTS 129u
#define WF2_MAX_REALRAY_FUSED_MSE_BOUNDARIES 256u
#define WF2_MAX_REALRAY_FUSED_MSE_SEGMENTS 257u
#define WF2_MAX_REALRAY_FUSED_MSE_CAP224_BOUNDARIES 224u
#define WF2_MAX_REALRAY_FUSED_MSE_CAP224_SEGMENTS 225u
#define WF2_MAX_REALRAY_SITES 64u
#define WF2_ENDPOINT_EDIT_FRAMEGROUP_MAX_FRAMES 16u
#define WF2_ENDPOINT_DELTA_FRAMEGROUP_MAX_FRAMES 32u
#define WF2_ENDPOINT_DELTA_FRAMEGROUP64_MAX_FRAMES 64u
#define WF2_ENDPOINT_DELTA_SMALLRUN_MAX_SEGMENTS 16u
#define WF2_ENDPOINT_EDIT_FRAMEGROUP_REDUCE_SITES 16u
#define WF2_ENDPOINT_ROWDESC32_REDUCE_SITES 32u
#define WF2_ENDPOINT_COMPACT_REDUCE32_SITES 32u

static inline void wf2_atomic_add4(
    device atomic_float* ptr,
    const uint base,
    const float4 value) {
  atomic_fetch_add_explicit(&ptr[base + 0u], value.x, memory_order_relaxed);
  atomic_fetch_add_explicit(&ptr[base + 1u], value.y, memory_order_relaxed);
  atomic_fetch_add_explicit(&ptr[base + 2u], value.z, memory_order_relaxed);
  atomic_fetch_add_explicit(&ptr[base + 3u], value.w, memory_order_relaxed);
}

static inline void wf2_atomic_add3(
    device atomic_float* ptr,
    const uint base,
    const float3 value) {
  atomic_fetch_add_explicit(&ptr[base + 0u], value.x, memory_order_relaxed);
  atomic_fetch_add_explicit(&ptr[base + 1u], value.y, memory_order_relaxed);
  atomic_fetch_add_explicit(&ptr[base + 2u], value.z, memory_order_relaxed);
}

static inline void wf2_atomic_add5(
    device atomic_float* ptr,
    const uint base,
    const float3 xyz,
    const float t,
    const float bias) {
  atomic_fetch_add_explicit(&ptr[base + 0u], xyz.x, memory_order_relaxed);
  atomic_fetch_add_explicit(&ptr[base + 1u], xyz.y, memory_order_relaxed);
  atomic_fetch_add_explicit(&ptr[base + 2u], xyz.z, memory_order_relaxed);
  atomic_fetch_add_explicit(&ptr[base + 3u], t, memory_order_relaxed);
  atomic_fetch_add_explicit(&ptr[base + 4u], bias, memory_order_relaxed);
}

static inline uint wf2_replay_slab_id(const float t, const uint time_slab_count) {
  const uint clamped_count = max(time_slab_count, 1u);
  const uint raw = uint(floor(t * float(clamped_count)));
  return min(raw, clamped_count - 1u);
}

static inline float wf2_replay_power_depth(
    device const float* boundary_f32,
    const uint boundary_id,
    const float u_center,
    const float t,
    const float camera_velocity_x) {
  const uint base = boundary_id * 4u;
  const float nx = boundary_f32[base + 0u];
  const float nz = boundary_f32[base + 1u];
  const float nt = boundary_f32[base + 2u];
  const float b = boundary_f32[base + 3u];
  const float x = u_center + camera_velocity_x * t;
  return -(nx * x + nt * t + b) / nz;
}

static inline void wf2_replay_insert_depth(
    thread float* depths,
    thread uint& depth_count,
    const float depth) {
  uint out = depth_count;
  while (out > 0u && depths[out - 1u] > depth) {
    depths[out] = depths[out - 1u];
    out -= 1u;
  }
  depths[out] = depth;
  depth_count += 1u;
}

static inline uint wf2_replay_owner_at(
    device const float* sites_f32,
    const uint site_count,
    const float x,
    const float z,
    const float t) {
  uint owner = 0u;
  float best = INFINITY;
  for (uint site_id = 0u; site_id < site_count; ++site_id) {
    const uint base = site_id * 4u;
    const float dx = x - sites_f32[base + 0u];
    const float dz = z - sites_f32[base + 1u];
    const float dt = t - sites_f32[base + 2u];
    const float weight = sites_f32[base + 3u];
    const float distance = dx * dx + dz * dz + dt * dt - weight;
    if (distance < best) {
      best = distance;
      owner = site_id;
    }
  }
  return owner;
}

static inline void wf2_realray_insert_depth_capped(
    thread float* depths,
    thread uint& depth_count,
    const float depth,
    const uint max_boundaries) {
  if (depth_count >= max_boundaries) {
    return;
  }
  uint out = depth_count;
  while (out > 0u && depths[out - 1u] > depth) {
    depths[out] = depths[out - 1u];
    out -= 1u;
  }
  depths[out] = depth;
  depth_count += 1u;
}

static inline void wf2_realray_insert_depth(
    thread float* depths,
    thread uint& depth_count,
    const float depth) {
  wf2_realray_insert_depth_capped(depths, depth_count, depth, WF2_MAX_REALRAY_BOUNDARIES);
}

static inline void wf2_realray_insert_depth_fused_mse(
    thread float* depths,
    thread uint& depth_count,
    const float depth) {
  wf2_realray_insert_depth_capped(depths, depth_count, depth, WF2_MAX_REALRAY_FUSED_MSE_BOUNDARIES);
}

static inline void wf2_realray_sort_depths_bitonic_fused_mse(
    thread float* depths,
    const uint depth_count) {
  if (depth_count <= 1u) {
    return;
  }
  uint sort_count = 1u;
  while (sort_count < depth_count && sort_count < WF2_MAX_REALRAY_FUSED_MSE_BOUNDARIES) {
    sort_count <<= 1u;
  }
  sort_count = min(sort_count, WF2_MAX_REALRAY_FUSED_MSE_BOUNDARIES);
  for (uint index = depth_count; index < sort_count; ++index) {
    depths[index] = INFINITY;
  }
  for (uint k = 2u; k <= sort_count; k <<= 1u) {
    for (uint j = k >> 1u; j > 0u; j >>= 1u) {
      for (uint index = 0u; index < sort_count; ++index) {
        const uint peer = index ^ j;
        if (peer <= index || peer >= sort_count) {
          continue;
        }
        const bool ascending = (index & k) == 0u;
        const float left = depths[index];
        const float right = depths[peer];
        if ((ascending && left > right) || (!ascending && left < right)) {
          depths[index] = right;
          depths[peer] = left;
        }
      }
    }
  }
}

static inline void wf2_realray_insert_depth_with_boundary_capped(
    thread float* depths,
    thread uint* boundary_ids,
    thread uint& depth_count,
    const float depth,
    const uint boundary_id,
    const uint max_boundaries) {
  if (depth_count >= max_boundaries) {
    return;
  }
  uint out = depth_count;
  while (out > 0u && depths[out - 1u] > depth) {
    depths[out] = depths[out - 1u];
    boundary_ids[out] = boundary_ids[out - 1u];
    out -= 1u;
  }
  depths[out] = depth;
  boundary_ids[out] = boundary_id;
  depth_count += 1u;
}

static inline void wf2_realray_insert_depth_with_boundary(
    thread float* depths,
    thread uint* boundary_ids,
    thread uint& depth_count,
    const float depth,
    const uint boundary_id) {
  wf2_realray_insert_depth_with_boundary_capped(
      depths, boundary_ids, depth_count, depth, boundary_id, WF2_MAX_REALRAY_BOUNDARIES);
}

static inline void wf2_realray_insert_depth_with_boundary_fused_mse(
    thread float* depths,
    thread uint* boundary_ids,
    thread uint& depth_count,
    const float depth,
    const uint boundary_id) {
  wf2_realray_insert_depth_with_boundary_capped(
      depths, boundary_ids, depth_count, depth, boundary_id, WF2_MAX_REALRAY_FUSED_MSE_BOUNDARIES);
}

static inline uint wf2_realray_owner_at(
    device const float* sites_f32,
    const uint site_count,
    const float x,
    const float y,
    const float z,
    const float t) {
  uint owner = 0u;
  float best = INFINITY;
  for (uint site_id = 0u; site_id < site_count; ++site_id) {
    const uint base = site_id * 5u;
    const float dx = x - sites_f32[base + 0u];
    const float dy = y - sites_f32[base + 1u];
    const float dz = z - sites_f32[base + 2u];
    const float dt = t - sites_f32[base + 3u];
    const float weight = sites_f32[base + 4u];
    const float distance = dx * dx + dy * dy + dz * dz + dt * dt - weight;
    if (distance < best) {
      best = distance;
      owner = site_id;
    }
  }
  return owner;
}

static inline uint wf2_realray_owner_at_cached(
    threadgroup const float* sites_f32,
    const uint site_count,
    const float x,
    const float y,
    const float z,
    const float t) {
  uint owner = 0u;
  float best = INFINITY;
  for (uint site_id = 0u; site_id < site_count; ++site_id) {
    const uint base = site_id * 5u;
    const float dx = x - sites_f32[base + 0u];
    const float dy = y - sites_f32[base + 1u];
    const float dz = z - sites_f32[base + 2u];
    const float dt = t - sites_f32[base + 3u];
    const float weight = sites_f32[base + 4u];
    const float distance = dx * dx + dy * dy + dz * dz + dt * dt - weight;
    if (distance < best) {
      best = distance;
      owner = site_id;
    }
  }
  return owner;
}

static inline bool wf2_endpoint_record_cut_depth(
    device const float* boundary_f32,
    const uint boundary_count,
    const int cut_id,
    const float3 origin,
    const float3 direction,
    const float t,
    const float near_depth,
    const float far_depth,
    const float invalid_epsilon,
    thread float& out_depth) {
  if (cut_id == -1) {
    out_depth = near_depth;
    return true;
  }
  if (cut_id == -2) {
    out_depth = far_depth;
    return true;
  }
  if (cut_id < 0 || uint(cut_id) >= boundary_count) {
    return false;
  }
  const uint base = uint(cut_id) * 5u;
  const float nx = boundary_f32[base + 0u];
  const float ny = boundary_f32[base + 1u];
  const float nz = boundary_f32[base + 2u];
  const float nt = boundary_f32[base + 3u];
  const float b = boundary_f32[base + 4u];
  const float denom = nx * direction.x + ny * direction.y + nz * direction.z;
  if (fabs(denom) < invalid_epsilon) {
    return false;
  }
  const float depth = -(nx * origin.x + ny * origin.y + nz * origin.z + nt * t + b) / denom;
  if (!isfinite(depth) || depth < near_depth || depth > far_depth) {
    return false;
  }
  out_depth = depth;
  return true;
}

static inline bool wf2_endpoint_record_load_edit_row(
    device const int* base_offsets_i32,
    device const int* base_owner_i32,
    device const int* base_left_i32,
    device const int* base_right_i32,
    device const int* track_change_offsets_i32,
    device const int* change_frame_i32,
    device const int* op_offsets_i32,
    device const int* op_type_i32,
    device const int* op_pos_i32,
    device const int* op_owner_i32,
    device const int* op_left_i32,
    device const int* op_right_i32,
    const uint track_id,
    const uint frame_id,
    const uint base_record_count,
    const uint change_count,
    const uint op_count,
    thread int* row_owner,
    thread int* row_left,
    thread int* row_right,
    thread uint& row_count) {
  row_count = 0u;
  const int base_begin_raw = base_offsets_i32[track_id];
  const int base_end_raw = base_offsets_i32[track_id + 1u];
  if (base_begin_raw < 0 || base_end_raw < base_begin_raw || uint(base_end_raw) > base_record_count) {
    return false;
  }
  for (uint cursor = uint(base_begin_raw); cursor < uint(base_end_raw); ++cursor) {
    if (row_count >= WF2_MAX_REALRAY_SEGMENTS) {
      return false;
    }
    row_owner[row_count] = base_owner_i32[cursor];
    row_left[row_count] = base_left_i32[cursor];
    row_right[row_count] = base_right_i32[cursor];
    row_count += 1u;
  }

  const int change_begin_raw = track_change_offsets_i32[track_id];
  const int change_end_raw = track_change_offsets_i32[track_id + 1u];
  if (change_begin_raw < 0 || change_end_raw < change_begin_raw || uint(change_end_raw) > change_count) {
    return false;
  }
  for (uint change_cursor = uint(change_begin_raw); change_cursor < uint(change_end_raw); ++change_cursor) {
    const int changed_frame = change_frame_i32[change_cursor];
    if (changed_frame < 0) {
      continue;
    }
    if (uint(changed_frame) > frame_id) {
      break;
    }
    const int op_begin_raw = op_offsets_i32[change_cursor];
    const int op_end_raw = op_offsets_i32[change_cursor + 1u];
    if (op_begin_raw < 0 || op_end_raw < op_begin_raw || uint(op_end_raw) > op_count) {
      return false;
    }
    for (uint op_cursor = uint(op_begin_raw); op_cursor < uint(op_end_raw); ++op_cursor) {
      const int op_type = op_type_i32[op_cursor];
      const int pos_raw = op_pos_i32[op_cursor];
      if (pos_raw < 0) {
        return false;
      }
      const uint pos = uint(pos_raw);
      if (op_type == 0) {
        if (pos > row_count || row_count >= WF2_MAX_REALRAY_SEGMENTS) {
          return false;
        }
        for (uint shift = row_count; shift > pos; --shift) {
          row_owner[shift] = row_owner[shift - 1u];
          row_left[shift] = row_left[shift - 1u];
          row_right[shift] = row_right[shift - 1u];
        }
        row_owner[pos] = op_owner_i32[op_cursor];
        row_left[pos] = op_left_i32[op_cursor];
        row_right[pos] = op_right_i32[op_cursor];
        row_count += 1u;
      } else if (op_type == 1) {
        if (pos >= row_count) {
          return false;
        }
        for (uint shift = pos; shift + 1u < row_count; ++shift) {
          row_owner[shift] = row_owner[shift + 1u];
          row_left[shift] = row_left[shift + 1u];
          row_right[shift] = row_right[shift + 1u];
        }
        row_count -= 1u;
      } else if (op_type == 2) {
        if (pos >= row_count) {
          return false;
        }
        row_owner[pos] = op_owner_i32[op_cursor];
        row_left[pos] = op_left_i32[op_cursor];
        row_right[pos] = op_right_i32[op_cursor];
      } else {
        return false;
      }
    }
  }
  return true;
}

static inline bool wf2_endpoint_record_load_block_edit_row(
    device const int* anchor_offsets_i32,
    device const int* anchor_owner_i32,
    device const int* anchor_left_i32,
    device const int* anchor_right_i32,
    device const int* track_block_change_offsets_i32,
    device const int* change_frame_i32,
    device const int* op_offsets_i32,
    device const int* op_type_i32,
    device const int* op_pos_i32,
    device const int* op_owner_i32,
    device const int* op_left_i32,
    device const int* op_right_i32,
    const uint track_id,
    const uint frame_id,
    const uint block_size,
    const uint block_count,
    const uint anchor_record_count,
    const uint change_count,
    const uint op_count,
    thread int* row_owner,
    thread int* row_left,
    thread int* row_right,
    thread uint& row_count) {
  row_count = 0u;
  if (block_size == 0u || block_count == 0u) {
    return false;
  }
  const uint block_id = min(frame_id / block_size, block_count - 1u);
  const uint row_id = track_id * block_count + block_id;
  const int anchor_begin_raw = anchor_offsets_i32[row_id];
  const int anchor_end_raw = anchor_offsets_i32[row_id + 1u];
  if (anchor_begin_raw < 0 || anchor_end_raw < anchor_begin_raw || uint(anchor_end_raw) > anchor_record_count) {
    return false;
  }
  for (uint cursor = uint(anchor_begin_raw); cursor < uint(anchor_end_raw); ++cursor) {
    if (row_count >= WF2_MAX_REALRAY_SEGMENTS) {
      return false;
    }
    row_owner[row_count] = anchor_owner_i32[cursor];
    row_left[row_count] = anchor_left_i32[cursor];
    row_right[row_count] = anchor_right_i32[cursor];
    row_count += 1u;
  }

  const uint block_offset_base = track_id * (block_count + 1u) + block_id;
  const int change_begin_raw = track_block_change_offsets_i32[block_offset_base];
  const int change_end_raw = track_block_change_offsets_i32[block_offset_base + 1u];
  if (change_begin_raw < 0 || change_end_raw < change_begin_raw || uint(change_end_raw) > change_count) {
    return false;
  }
  for (uint change_cursor = uint(change_begin_raw); change_cursor < uint(change_end_raw); ++change_cursor) {
    const int changed_frame = change_frame_i32[change_cursor];
    if (changed_frame < 0) {
      continue;
    }
    if (uint(changed_frame) > frame_id) {
      break;
    }
    const int op_begin_raw = op_offsets_i32[change_cursor];
    const int op_end_raw = op_offsets_i32[change_cursor + 1u];
    if (op_begin_raw < 0 || op_end_raw < op_begin_raw || uint(op_end_raw) > op_count) {
      return false;
    }
    for (uint op_cursor = uint(op_begin_raw); op_cursor < uint(op_end_raw); ++op_cursor) {
      const int op_type = op_type_i32[op_cursor];
      const int pos_raw = op_pos_i32[op_cursor];
      if (pos_raw < 0) {
        return false;
      }
      const uint pos = uint(pos_raw);
      if (op_type == 0) {
        if (pos > row_count || row_count >= WF2_MAX_REALRAY_SEGMENTS) {
          return false;
        }
        for (uint shift = row_count; shift > pos; --shift) {
          row_owner[shift] = row_owner[shift - 1u];
          row_left[shift] = row_left[shift - 1u];
          row_right[shift] = row_right[shift - 1u];
        }
        row_owner[pos] = op_owner_i32[op_cursor];
        row_left[pos] = op_left_i32[op_cursor];
        row_right[pos] = op_right_i32[op_cursor];
        row_count += 1u;
      } else if (op_type == 1) {
        if (pos >= row_count) {
          return false;
        }
        for (uint shift = pos; shift + 1u < row_count; ++shift) {
          row_owner[shift] = row_owner[shift + 1u];
          row_left[shift] = row_left[shift + 1u];
          row_right[shift] = row_right[shift + 1u];
        }
        row_count -= 1u;
      } else if (op_type == 2) {
        if (pos >= row_count) {
          return false;
        }
        row_owner[pos] = op_owner_i32[op_cursor];
        row_left[pos] = op_left_i32[op_cursor];
        row_right[pos] = op_right_i32[op_cursor];
      } else {
        return false;
      }
    }
  }
  return true;
}

static inline int wf2_endpoint_record_unpack_cut_code(const uint code) {
  if (code == 0u) {
    return -1;
  }
  if (code == 1u) {
    return -2;
  }
  return int(code) - 2;
}

static inline void wf2_endpoint_record_unpack_record(
    const int packed_raw,
    thread int& owner,
    thread int& left,
    thread int& right) {
  const uint packed = uint(packed_raw);
  owner = int(packed & 255u);
  left = wf2_endpoint_record_unpack_cut_code((packed >> 8u) & 4095u);
  right = wf2_endpoint_record_unpack_cut_code((packed >> 20u) & 4095u);
}

static inline bool wf2_endpoint_record_load_block_edit_row_packed(
    device const int* anchor_offsets_i32,
    device const int* anchor_record_i32,
    device const int* track_block_change_offsets_i32,
    device const int* change_frame_i32,
    device const int* op_offsets_i32,
    device const int* op_type_i32,
    device const int* op_pos_i32,
    device const int* op_record_i32,
    const uint track_id,
    const uint frame_id,
    const uint block_size,
    const uint block_count,
    const uint anchor_record_count,
    const uint change_count,
    const uint op_count,
    thread int* row_owner,
    thread int* row_left,
    thread int* row_right,
    thread uint& row_count) {
  row_count = 0u;
  if (block_size == 0u || block_count == 0u) {
    return false;
  }
  const uint block_id = min(frame_id / block_size, block_count - 1u);
  const uint row_id = track_id * block_count + block_id;
  const int anchor_begin_raw = anchor_offsets_i32[row_id];
  const int anchor_end_raw = anchor_offsets_i32[row_id + 1u];
  if (anchor_begin_raw < 0 || anchor_end_raw < anchor_begin_raw || uint(anchor_end_raw) > anchor_record_count) {
    return false;
  }
  for (uint cursor = uint(anchor_begin_raw); cursor < uint(anchor_end_raw); ++cursor) {
    if (row_count >= WF2_MAX_REALRAY_SEGMENTS) {
      return false;
    }
    wf2_endpoint_record_unpack_record(anchor_record_i32[cursor], row_owner[row_count], row_left[row_count], row_right[row_count]);
    row_count += 1u;
  }

  const uint block_offset_base = track_id * (block_count + 1u) + block_id;
  const int change_begin_raw = track_block_change_offsets_i32[block_offset_base];
  const int change_end_raw = track_block_change_offsets_i32[block_offset_base + 1u];
  if (change_begin_raw < 0 || change_end_raw < change_begin_raw || uint(change_end_raw) > change_count) {
    return false;
  }
  for (uint change_cursor = uint(change_begin_raw); change_cursor < uint(change_end_raw); ++change_cursor) {
    const int changed_frame = change_frame_i32[change_cursor];
    if (changed_frame < 0) {
      continue;
    }
    if (uint(changed_frame) > frame_id) {
      break;
    }
    const int op_begin_raw = op_offsets_i32[change_cursor];
    const int op_end_raw = op_offsets_i32[change_cursor + 1u];
    if (op_begin_raw < 0 || op_end_raw < op_begin_raw || uint(op_end_raw) > op_count) {
      return false;
    }
    for (uint op_cursor = uint(op_begin_raw); op_cursor < uint(op_end_raw); ++op_cursor) {
      const int op_type = op_type_i32[op_cursor];
      const int pos_raw = op_pos_i32[op_cursor];
      if (pos_raw < 0) {
        return false;
      }
      const uint pos = uint(pos_raw);
      if (op_type == 0) {
        if (pos > row_count || row_count >= WF2_MAX_REALRAY_SEGMENTS) {
          return false;
        }
        for (uint shift = row_count; shift > pos; --shift) {
          row_owner[shift] = row_owner[shift - 1u];
          row_left[shift] = row_left[shift - 1u];
          row_right[shift] = row_right[shift - 1u];
        }
        wf2_endpoint_record_unpack_record(op_record_i32[op_cursor], row_owner[pos], row_left[pos], row_right[pos]);
        row_count += 1u;
      } else if (op_type == 1) {
        if (pos >= row_count) {
          return false;
        }
        for (uint shift = pos; shift + 1u < row_count; ++shift) {
          row_owner[shift] = row_owner[shift + 1u];
          row_left[shift] = row_left[shift + 1u];
          row_right[shift] = row_right[shift + 1u];
        }
        row_count -= 1u;
      } else if (op_type == 2) {
        if (pos >= row_count) {
          return false;
        }
        wf2_endpoint_record_unpack_record(op_record_i32[op_cursor], row_owner[pos], row_left[pos], row_right[pos]);
      } else {
        return false;
      }
    }
  }
  return true;
}

static inline bool wf2_endpoint_record_load_block_edit_row_i16(
    device const int* anchor_offsets_i32,
    device const short* anchor_owner_i16,
    device const short* anchor_left_i16,
    device const short* anchor_right_i16,
    device const int* track_block_change_offsets_i32,
    device const int* change_frame_i32,
    device const int* op_offsets_i32,
    device const int* op_type_i32,
    device const int* op_pos_i32,
    device const short* op_owner_i16,
    device const short* op_left_i16,
    device const short* op_right_i16,
    const uint track_id,
    const uint frame_id,
    const uint block_size,
    const uint block_count,
    const uint anchor_record_count,
    const uint change_count,
    const uint op_count,
    thread int* row_owner,
    thread int* row_left,
    thread int* row_right,
    thread uint& row_count) {
  row_count = 0u;
  if (block_size == 0u || block_count == 0u) {
    return false;
  }
  const uint block_id = min(frame_id / block_size, block_count - 1u);
  const uint row_id = track_id * block_count + block_id;
  const int anchor_begin_raw = anchor_offsets_i32[row_id];
  const int anchor_end_raw = anchor_offsets_i32[row_id + 1u];
  if (anchor_begin_raw < 0 || anchor_end_raw < anchor_begin_raw || uint(anchor_end_raw) > anchor_record_count) {
    return false;
  }
  for (uint cursor = uint(anchor_begin_raw); cursor < uint(anchor_end_raw); ++cursor) {
    if (row_count >= WF2_MAX_REALRAY_SEGMENTS) {
      return false;
    }
    row_owner[row_count] = int(anchor_owner_i16[cursor]);
    row_left[row_count] = int(anchor_left_i16[cursor]);
    row_right[row_count] = int(anchor_right_i16[cursor]);
    row_count += 1u;
  }

  const uint block_offset_base = track_id * (block_count + 1u) + block_id;
  const int change_begin_raw = track_block_change_offsets_i32[block_offset_base];
  const int change_end_raw = track_block_change_offsets_i32[block_offset_base + 1u];
  if (change_begin_raw < 0 || change_end_raw < change_begin_raw || uint(change_end_raw) > change_count) {
    return false;
  }
  for (uint change_cursor = uint(change_begin_raw); change_cursor < uint(change_end_raw); ++change_cursor) {
    const int changed_frame = change_frame_i32[change_cursor];
    if (changed_frame < 0) {
      continue;
    }
    if (uint(changed_frame) > frame_id) {
      break;
    }
    const int op_begin_raw = op_offsets_i32[change_cursor];
    const int op_end_raw = op_offsets_i32[change_cursor + 1u];
    if (op_begin_raw < 0 || op_end_raw < op_begin_raw || uint(op_end_raw) > op_count) {
      return false;
    }
    for (uint op_cursor = uint(op_begin_raw); op_cursor < uint(op_end_raw); ++op_cursor) {
      const int op_type = op_type_i32[op_cursor];
      const int pos_raw = op_pos_i32[op_cursor];
      if (pos_raw < 0) {
        return false;
      }
      const uint pos = uint(pos_raw);
      if (op_type == 0) {
        if (pos > row_count || row_count >= WF2_MAX_REALRAY_SEGMENTS) {
          return false;
        }
        for (uint shift = row_count; shift > pos; --shift) {
          row_owner[shift] = row_owner[shift - 1u];
          row_left[shift] = row_left[shift - 1u];
          row_right[shift] = row_right[shift - 1u];
        }
        row_owner[pos] = int(op_owner_i16[op_cursor]);
        row_left[pos] = int(op_left_i16[op_cursor]);
        row_right[pos] = int(op_right_i16[op_cursor]);
        row_count += 1u;
      } else if (op_type == 1) {
        if (pos >= row_count) {
          return false;
        }
        for (uint shift = pos; shift + 1u < row_count; ++shift) {
          row_owner[shift] = row_owner[shift + 1u];
          row_left[shift] = row_left[shift + 1u];
          row_right[shift] = row_right[shift + 1u];
        }
        row_count -= 1u;
      } else if (op_type == 2) {
        if (pos >= row_count) {
          return false;
        }
        row_owner[pos] = int(op_owner_i16[op_cursor]);
        row_left[pos] = int(op_left_i16[op_cursor]);
        row_right[pos] = int(op_right_i16[op_cursor]);
      } else {
        return false;
      }
    }
  }
  return true;
}

static inline bool wf2_endpoint_record_load_block_edit_row_i16x3(
    device const int* anchor_offsets_i32,
    device const short* anchor_record_i16,
    device const int* track_block_change_offsets_i32,
    device const int* change_frame_i32,
    device const int* op_offsets_i32,
    device const int* op_type_i32,
    device const int* op_pos_i32,
    device const short* op_record_i16,
    const uint track_id,
    const uint frame_id,
    const uint block_size,
    const uint block_count,
    const uint anchor_record_count,
    const uint change_count,
    const uint op_count,
    thread int* row_owner,
    thread int* row_left,
    thread int* row_right,
    thread uint& row_count) {
  row_count = 0u;
  if (block_size == 0u || block_count == 0u) {
    return false;
  }
  const uint block_id = min(frame_id / block_size, block_count - 1u);
  const uint row_id = track_id * block_count + block_id;
  const int anchor_begin_raw = anchor_offsets_i32[row_id];
  const int anchor_end_raw = anchor_offsets_i32[row_id + 1u];
  if (anchor_begin_raw < 0 || anchor_end_raw < anchor_begin_raw || uint(anchor_end_raw) > anchor_record_count) {
    return false;
  }
  for (uint cursor = uint(anchor_begin_raw); cursor < uint(anchor_end_raw); ++cursor) {
    if (row_count >= WF2_MAX_REALRAY_SEGMENTS) {
      return false;
    }
    const uint record_base = cursor * 3u;
    row_owner[row_count] = int(anchor_record_i16[record_base + 0u]);
    row_left[row_count] = int(anchor_record_i16[record_base + 1u]);
    row_right[row_count] = int(anchor_record_i16[record_base + 2u]);
    row_count += 1u;
  }

  const uint block_offset_base = track_id * (block_count + 1u) + block_id;
  const int change_begin_raw = track_block_change_offsets_i32[block_offset_base];
  const int change_end_raw = track_block_change_offsets_i32[block_offset_base + 1u];
  if (change_begin_raw < 0 || change_end_raw < change_begin_raw || uint(change_end_raw) > change_count) {
    return false;
  }
  for (uint change_cursor = uint(change_begin_raw); change_cursor < uint(change_end_raw); ++change_cursor) {
    const int changed_frame = change_frame_i32[change_cursor];
    if (changed_frame < 0) {
      continue;
    }
    if (uint(changed_frame) > frame_id) {
      break;
    }
    const int op_begin_raw = op_offsets_i32[change_cursor];
    const int op_end_raw = op_offsets_i32[change_cursor + 1u];
    if (op_begin_raw < 0 || op_end_raw < op_begin_raw || uint(op_end_raw) > op_count) {
      return false;
    }
    for (uint op_cursor = uint(op_begin_raw); op_cursor < uint(op_end_raw); ++op_cursor) {
      const int op_type = op_type_i32[op_cursor];
      const int pos_raw = op_pos_i32[op_cursor];
      if (pos_raw < 0) {
        return false;
      }
      const uint pos = uint(pos_raw);
      if (op_type == 0) {
        if (pos > row_count || row_count >= WF2_MAX_REALRAY_SEGMENTS) {
          return false;
        }
        for (uint shift = row_count; shift > pos; --shift) {
          row_owner[shift] = row_owner[shift - 1u];
          row_left[shift] = row_left[shift - 1u];
          row_right[shift] = row_right[shift - 1u];
        }
        const uint record_base = op_cursor * 3u;
        row_owner[pos] = int(op_record_i16[record_base + 0u]);
        row_left[pos] = int(op_record_i16[record_base + 1u]);
        row_right[pos] = int(op_record_i16[record_base + 2u]);
        row_count += 1u;
      } else if (op_type == 1) {
        if (pos >= row_count) {
          return false;
        }
        for (uint shift = pos; shift + 1u < row_count; ++shift) {
          row_owner[shift] = row_owner[shift + 1u];
          row_left[shift] = row_left[shift + 1u];
          row_right[shift] = row_right[shift + 1u];
        }
        row_count -= 1u;
      } else if (op_type == 2) {
        if (pos >= row_count) {
          return false;
        }
        const uint record_base = op_cursor * 3u;
        row_owner[pos] = int(op_record_i16[record_base + 0u]);
        row_left[pos] = int(op_record_i16[record_base + 1u]);
        row_right[pos] = int(op_record_i16[record_base + 2u]);
      } else {
        return false;
      }
    }
  }
  return true;
}

static inline bool wf2_endpoint_record_coeff_cut_depth(
    device const float* coeff_f32,
    const uint boundary_count,
    const uint track_id,
    const int cut_id,
    const float t,
    const float near_depth,
    const float far_depth,
    const float invalid_epsilon,
    thread float& out_depth) {
  if (cut_id == -1) {
    out_depth = near_depth;
    return true;
  }
  if (cut_id == -2) {
    out_depth = far_depth;
    return true;
  }
  if (cut_id < 0 || uint(cut_id) >= boundary_count) {
    return false;
  }
  const uint coeff_base = (track_id * boundary_count + uint(cut_id)) * 4u;
  const float numerator = coeff_f32[coeff_base + 0u] + coeff_f32[coeff_base + 1u] * t;
  const float denominator = coeff_f32[coeff_base + 2u] + coeff_f32[coeff_base + 3u] * t;
  if (fabs(denominator) < invalid_epsilon) {
    return false;
  }
  out_depth = numerator / denominator;
  return isfinite(out_depth) && out_depth >= near_depth && out_depth <= far_depth;
}

static inline bool wf2_endpoint_record_coeff16_cut_depth(
    device const half* coeff_f16,
    const uint boundary_count,
    const uint track_id,
    const int cut_id,
    const float t,
    const float near_depth,
    const float far_depth,
    const float invalid_epsilon,
    thread float& out_depth) {
  if (cut_id == -1) {
    out_depth = near_depth;
    return true;
  }
  if (cut_id == -2) {
    out_depth = far_depth;
    return true;
  }
  if (cut_id < 0 || uint(cut_id) >= boundary_count) {
    return false;
  }
  const uint coeff_base = (track_id * boundary_count + uint(cut_id)) * 4u;
  const float numerator = float(coeff_f16[coeff_base + 0u]) + float(coeff_f16[coeff_base + 1u]) * t;
  const float denominator = float(coeff_f16[coeff_base + 2u]) + float(coeff_f16[coeff_base + 3u]) * t;
  if (fabs(denominator) < invalid_epsilon) {
    return false;
  }
  out_depth = numerator / denominator;
  return isfinite(out_depth) && out_depth >= near_depth && out_depth <= far_depth;
}

static inline bool wf2_endpoint_record_factorized_cut_depth(
    device const float* boundary_f32,
    device const float* track_ray_coeff_f32,
    const uint boundary_count,
    const uint track_id,
    const int cut_id,
    const float t,
    const float near_depth,
    const float far_depth,
    const float invalid_epsilon,
    thread float& out_depth) {
  if (cut_id == -1) {
    out_depth = near_depth;
    return true;
  }
  if (cut_id == -2) {
    out_depth = far_depth;
    return true;
  }
  if (cut_id < 0 || uint(cut_id) >= boundary_count) {
    return false;
  }
  const uint boundary_base = uint(cut_id) * 5u;
  const float nx = boundary_f32[boundary_base + 0u];
  const float ny = boundary_f32[boundary_base + 1u];
  const float nz = boundary_f32[boundary_base + 2u];
  const float nt = boundary_f32[boundary_base + 3u];
  const float b = boundary_f32[boundary_base + 4u];
  const uint track_base = track_id * 12u;
  const float numer_base = -(
      nx * track_ray_coeff_f32[track_base + 0u] +
      ny * track_ray_coeff_f32[track_base + 1u] +
      nz * track_ray_coeff_f32[track_base + 2u] +
      b);
  const float numer_slope = -(
      nx * track_ray_coeff_f32[track_base + 3u] +
      ny * track_ray_coeff_f32[track_base + 4u] +
      nz * track_ray_coeff_f32[track_base + 5u] +
      nt);
  const float denom_base =
      nx * track_ray_coeff_f32[track_base + 6u] +
      ny * track_ray_coeff_f32[track_base + 7u] +
      nz * track_ray_coeff_f32[track_base + 8u];
  const float denom_slope =
      nx * track_ray_coeff_f32[track_base + 9u] +
      ny * track_ray_coeff_f32[track_base + 10u] +
      nz * track_ray_coeff_f32[track_base + 11u];
  const float denominator = denom_base + denom_slope * t;
  if (fabs(denominator) < invalid_epsilon) {
    return false;
  }
  const float numerator = numer_base + numer_slope * t;
  out_depth = numerator / denominator;
  return isfinite(out_depth) && out_depth >= near_depth && out_depth <= far_depth;
}

static inline bool wf2_endpoint_record_factorized_cut_depth_boundary_jacobian(
    device const float* boundary_f32,
    device const float* track_ray_coeff_f32,
    const uint boundary_count,
    const uint track_id,
    const int cut_id,
    const float t,
    const float near_depth,
    const float far_depth,
    const float invalid_epsilon,
    thread float& out_depth,
    thread float3& out_grad_normal,
    thread float& out_grad_time_normal,
    thread float& out_grad_bias) {
  out_grad_normal = float3(0.0f, 0.0f, 0.0f);
  out_grad_time_normal = 0.0f;
  out_grad_bias = 0.0f;
  if (cut_id == -1) {
    out_depth = near_depth;
    return true;
  }
  if (cut_id == -2) {
    out_depth = far_depth;
    return true;
  }
  if (cut_id < 0 || uint(cut_id) >= boundary_count) {
    return false;
  }

  const uint boundary_base = uint(cut_id) * 5u;
  const float3 normal = float3(
      boundary_f32[boundary_base + 0u],
      boundary_f32[boundary_base + 1u],
      boundary_f32[boundary_base + 2u]);
  const float time_normal = boundary_f32[boundary_base + 3u];
  const float bias = boundary_f32[boundary_base + 4u];
  const uint track_base = track_id * 12u;
  const float3 origin = float3(
      track_ray_coeff_f32[track_base + 0u] + t * track_ray_coeff_f32[track_base + 3u],
      track_ray_coeff_f32[track_base + 1u] + t * track_ray_coeff_f32[track_base + 4u],
      track_ray_coeff_f32[track_base + 2u] + t * track_ray_coeff_f32[track_base + 5u]);
  const float3 direction = float3(
      track_ray_coeff_f32[track_base + 6u] + t * track_ray_coeff_f32[track_base + 9u],
      track_ray_coeff_f32[track_base + 7u] + t * track_ray_coeff_f32[track_base + 10u],
      track_ray_coeff_f32[track_base + 8u] + t * track_ray_coeff_f32[track_base + 11u]);
  // Keep one lowering/evaluation order for both constant-state passes so their
  // fixed-topology branch decisions stay identical.
  const float numerator_base = -(
      normal.x * track_ray_coeff_f32[track_base + 0u] +
      normal.y * track_ray_coeff_f32[track_base + 1u] +
      normal.z * track_ray_coeff_f32[track_base + 2u] +
      bias);
  const float numerator_slope = -(
      normal.x * track_ray_coeff_f32[track_base + 3u] +
      normal.y * track_ray_coeff_f32[track_base + 4u] +
      normal.z * track_ray_coeff_f32[track_base + 5u] +
      time_normal);
  const float denominator_base =
      normal.x * track_ray_coeff_f32[track_base + 6u] +
      normal.y * track_ray_coeff_f32[track_base + 7u] +
      normal.z * track_ray_coeff_f32[track_base + 8u];
  const float denominator_slope =
      normal.x * track_ray_coeff_f32[track_base + 9u] +
      normal.y * track_ray_coeff_f32[track_base + 10u] +
      normal.z * track_ray_coeff_f32[track_base + 11u];
  const float numerator = numerator_base + numerator_slope * t;
  const float denominator = denominator_base + denominator_slope * t;
  const float denominator_scale = length(normal) * length(direction);
  if (!(denominator_scale > 0.0f) ||
      !isfinite(denominator_scale) ||
      fabs(denominator) < invalid_epsilon * denominator_scale) {
    return false;
  }
  out_depth = numerator / denominator;
  if (!isfinite(out_depth) || out_depth < near_depth || out_depth > far_depth) {
    return false;
  }
  const float inv_denominator = 1.0f / denominator;
  out_grad_normal = -(origin + out_depth * direction) * inv_denominator;
  out_grad_time_normal = -t * inv_denominator;
  out_grad_bias = -inv_denominator;
  return true;
}

static inline bool wf2_endpoint_record_sparse_mobius_cut_depth_jacobian(
    device const float* mobius_coeff_f32,
    device const int* track_incidence_offsets_i32,
    const uint incidence_count,
    const uint track_id,
    const int local_cut_id,
    const float t,
    const float near_depth,
    const float far_depth,
    const float invalid_epsilon,
    thread uint& out_incidence_id,
    thread float& out_depth,
    thread float4& out_grad_mobius) {
  out_incidence_id = 0xFFFFFFFFu;
  out_grad_mobius = float4(0.0f, 0.0f, 0.0f, 0.0f);
  if (local_cut_id == -1) {
    out_depth = near_depth;
    return true;
  }
  if (local_cut_id == -2) {
    out_depth = far_depth;
    return true;
  }
  if (local_cut_id < 0) {
    return false;
  }
  const int incidence_begin_raw = track_incidence_offsets_i32[track_id];
  const int incidence_end_raw = track_incidence_offsets_i32[track_id + 1u];
  if (incidence_begin_raw < 0 || incidence_end_raw < incidence_begin_raw) {
    return false;
  }
  const uint incidence_begin = uint(incidence_begin_raw);
  const uint incidence_end = uint(incidence_end_raw);
  const uint local_incidence_id = uint(local_cut_id);
  if (local_incidence_id >= incidence_end - incidence_begin) {
    return false;
  }
  const uint incidence_id = incidence_begin + local_incidence_id;
  if (incidence_id >= incidence_count) {
    return false;
  }
  const uint coefficient_base = incidence_id * 4u;
  const float coefficient_a = mobius_coeff_f32[coefficient_base + 0u];
  const float coefficient_b = mobius_coeff_f32[coefficient_base + 1u];
  const float coefficient_c = mobius_coeff_f32[coefficient_base + 2u];
  const float coefficient_d = mobius_coeff_f32[coefficient_base + 3u];
  const float numerator = coefficient_a + coefficient_b * t;
  const float denominator = coefficient_c + coefficient_d * t;
  // Use the time-uniform coefficient scale certified by the CPU compiler.
  // Scaling D by the queried time makes the predicate itself chart/sample
  // dependent and misses a near-parallel nonzero denominator at t == 0.
  const float denominator_scale = max(1.0f, fabs(coefficient_c) + fabs(coefficient_d));
  if (!isfinite(denominator_scale) ||
      fabs(denominator) <= invalid_epsilon * denominator_scale) {
    return false;
  }
  const float inv_denominator = 1.0f / denominator;
  out_depth = numerator * inv_denominator;
  if (!isfinite(out_depth) || out_depth < near_depth || out_depth > far_depth) {
    return false;
  }
  out_incidence_id = incidence_id;
  out_grad_mobius = float4(
      inv_denominator,
      t * inv_denominator,
      -numerator * inv_denominator * inv_denominator,
      -t * numerator * inv_denominator * inv_denominator);
  return true;
}

static inline void wf2_lie_phi_and_derivative(
    const float kappa,
    thread float& phi,
    thread float& phi_prime) {
  if (fabs(kappa) < 1.0e-4f) {
    const float k2 = kappa * kappa;
    const float k3 = k2 * kappa;
    const float k4 = k3 * kappa;
    const float k5 = k4 * kappa;
    const float k6 = k5 * kappa;
    phi = 1.0f - 0.5f * kappa + k2 / 6.0f - k3 / 24.0f +
        k4 / 120.0f - k5 / 720.0f + k6 / 5040.0f;
    phi_prime = -0.5f + kappa / 3.0f - k2 / 8.0f +
        k3 / 30.0f - k4 / 144.0f + k5 / 840.0f;
    return;
  }
  const float numerator = -expm1(-kappa);
  phi = numerator / kappa;
  phi_prime = (kappa * exp(-kappa) - numerator) / (kappa * kappa);
}

static inline void wf2_lie_inverse_phi_and_derivative(
    const float kappa,
    thread float& inverse_phi,
    thread float& inverse_phi_prime) {
  if (fabs(kappa) < 1.0e-4f) {
    const float k2 = kappa * kappa;
    const float k3 = k2 * kappa;
    const float k4 = k3 * kappa;
    const float k5 = k4 * kappa;
    const float k6 = k5 * kappa;
    inverse_phi = 1.0f + 0.5f * kappa + k2 / 12.0f -
        k4 / 720.0f + k6 / 30240.0f;
    inverse_phi_prime = 0.5f + kappa / 6.0f -
        k3 / 180.0f + k5 / 5040.0f;
    return;
  }
  const float denominator = -expm1(-kappa);
  inverse_phi = kappa / denominator;
  inverse_phi_prime = (denominator - kappa * exp(-kappa)) /
      (denominator * denominator);
}

static inline float wf2_quiet_nan() {
  return as_type<float>(0x7FC00000u);
}

static inline bool wf2_bitset_has_boundary(
    device const uint* mask_words,
    const uint mask_base,
    const uint boundary_id) {
  const uint word = mask_words[mask_base + boundary_id / 32u];
  return (word & (1u << (boundary_id % 32u))) != 0u;
}

static inline float3 wf2_affine_origin_at(
    device const float* ray_coeff_f32,
    const uint track_id,
    const float t) {
  const uint base = track_id * 12u;
  return float3(
      ray_coeff_f32[base + 0u] + ray_coeff_f32[base + 3u] * t,
      ray_coeff_f32[base + 1u] + ray_coeff_f32[base + 4u] * t,
      ray_coeff_f32[base + 2u] + ray_coeff_f32[base + 5u] * t);
}

static inline float3 wf2_affine_direction_at(
    device const float* ray_coeff_f32,
    const uint track_id,
    const float t) {
  const uint base = track_id * 12u;
  return float3(
      ray_coeff_f32[base + 6u] + ray_coeff_f32[base + 9u] * t,
      ray_coeff_f32[base + 7u] + ray_coeff_f32[base + 10u] * t,
      ray_coeff_f32[base + 8u] + ray_coeff_f32[base + 11u] * t);
}

kernel void wf2_realray_rgba_depth_replay_tensor(
    device const float* boundary_f32 [[buffer(0)]],
    device const float* sites_f32 [[buffer(1)]],
    device const float* site_rgba_f32 [[buffer(2)]],
    device const float* rays_f32 [[buffer(3)]],
    device const float* frame_t_f32 [[buffer(4)]],
    device const int* config_i32 [[buffer(5)]],
    device const float* config_f32 [[buffer(6)]],
    device float* output_rgb_f32 [[buffer(7)]],
    device float* output_alpha_f32 [[buffer(8)]],
    device float* output_depth_f32 [[buffer(9)]],
    uint ray_id [[thread_position_in_grid]]) {
  const uint boundary_count = uint(config_i32[0]);
  const uint ray_count = uint(config_i32[1]);
  const uint site_count = uint(config_i32[2]);
  if (ray_id >= ray_count) {
    return;
  }

  const float near_depth = config_f32[0];
  const float far_depth = config_f32[1];
  const float invalid_epsilon = config_f32[2];
  const float transmittance_threshold = config_f32[3];
  const uint ray_base = ray_id * 6u;
  const float ox = rays_f32[ray_base + 0u];
  const float oy = rays_f32[ray_base + 1u];
  const float oz = rays_f32[ray_base + 2u];
  const float dx = rays_f32[ray_base + 3u];
  const float dy = rays_f32[ray_base + 4u];
  const float dz = rays_f32[ray_base + 5u];
  const float t = frame_t_f32[ray_id];

  float depths[WF2_MAX_REALRAY_BOUNDARIES];
  uint depth_count = 0u;
  const uint clamped_boundary_count = min(boundary_count, WF2_MAX_REALRAY_BOUNDARIES);
  for (uint boundary_id = 0u; boundary_id < clamped_boundary_count; ++boundary_id) {
    const uint base = boundary_id * 5u;
    const float nx = boundary_f32[base + 0u];
    const float ny = boundary_f32[base + 1u];
    const float nz = boundary_f32[base + 2u];
    const float nt = boundary_f32[base + 3u];
    const float b = boundary_f32[base + 4u];
    const float denom = nx * dx + ny * dy + nz * dz;
    if (fabs(denom) < invalid_epsilon) {
      continue;
    }
    const float depth = -(nx * ox + ny * oy + nz * oz + nt * t + b) / denom;
    if (isfinite(depth) && depth >= near_depth && depth <= far_depth) {
      wf2_realray_insert_depth(depths, depth_count, depth);
    }
  }

  const uint clamped_site_count = min(site_count, WF2_MAX_REALRAY_SITES);
  float3 rgb_accum = float3(0.0f, 0.0f, 0.0f);
  float alpha_accum = 0.0f;
  float depth_weighted = 0.0f;
  float transmittance = 1.0f;
  float previous_depth = near_depth;
  for (uint cut_id = 0u; cut_id <= depth_count; ++cut_id) {
    const float next_depth = cut_id < depth_count ? depths[cut_id] : far_depth;
    const float length = next_depth - previous_depth;
    if (length > 1.0e-8f && transmittance > transmittance_threshold) {
      const float mid_depth = 0.5f * (previous_depth + next_depth);
      const float x = ox + dx * mid_depth;
      const float y = oy + dy * mid_depth;
      const float z = oz + dz * mid_depth;
      const uint owner = wf2_realray_owner_at(sites_f32, clamped_site_count, x, y, z, t);
      const uint rgba_base = owner * 4u;
      const float density = max(site_rgba_f32[rgba_base + 3u], 0.0f);
      const float segment_transmittance = exp(-density * length);
      const float segment_alpha = 1.0f - segment_transmittance;
      const float weight = transmittance * segment_alpha;
      const float3 rgb = float3(
          site_rgba_f32[rgba_base + 0u],
          site_rgba_f32[rgba_base + 1u],
          site_rgba_f32[rgba_base + 2u]);
      rgb_accum += weight * rgb;
      alpha_accum += weight;
      depth_weighted += weight * mid_depth;
      transmittance *= segment_transmittance;
    }
    previous_depth = next_depth;
  }

  const uint out_base = ray_id * 3u;
  output_rgb_f32[out_base + 0u] = rgb_accum.x;
  output_rgb_f32[out_base + 1u] = rgb_accum.y;
  output_rgb_f32[out_base + 2u] = rgb_accum.z;
  output_alpha_f32[ray_id] = alpha_accum;
  output_depth_f32[ray_id] = alpha_accum > 1.0e-8f ? depth_weighted / alpha_accum : far_depth;
}

kernel void wf2_shared_realray_rgba_depth_replay_tensor(
    device const float* boundary_f32 [[buffer(0)]],
    device const uint* candidate_mask_u32 [[buffer(1)]],
    device const float* sites_f32 [[buffer(2)]],
    device const float* site_rgba_f32 [[buffer(3)]],
    device const float* track_rays_f32 [[buffer(4)]],
    device const float* frame_t_f32 [[buffer(5)]],
    device const int* config_i32 [[buffer(6)]],
    device const float* config_f32 [[buffer(7)]],
    device float* output_rgb_f32 [[buffer(8)]],
    device float* output_alpha_f32 [[buffer(9)]],
    device float* output_depth_f32 [[buffer(10)]],
    uint sample_id [[thread_position_in_grid]]) {
  const uint boundary_count = uint(config_i32[0]);
  const uint track_count = uint(config_i32[1]);
  const uint site_count = uint(config_i32[2]);
  const uint frame_count = uint(config_i32[3]);
  const uint time_slab_count = uint(config_i32[4]);
  const uint mask_word_count = uint(config_i32[5]);
  const uint total_samples = track_count * frame_count;
  if (sample_id >= total_samples) {
    return;
  }

  const float near_depth = config_f32[0];
  const float far_depth = config_f32[1];
  const float invalid_epsilon = config_f32[2];
  const float transmittance_threshold = config_f32[3];
  const uint track_id = sample_id / frame_count;
  const uint frame_id = sample_id - track_id * frame_count;
  const uint ray_base = track_id * 6u;
  const float ox = track_rays_f32[ray_base + 0u];
  const float oy = track_rays_f32[ray_base + 1u];
  const float oz = track_rays_f32[ray_base + 2u];
  const float dx = track_rays_f32[ray_base + 3u];
  const float dy = track_rays_f32[ray_base + 4u];
  const float dz = track_rays_f32[ray_base + 5u];
  const float t = frame_t_f32[frame_id];
  const uint slab_id = wf2_replay_slab_id(t, time_slab_count);
  const uint mask_base = (track_id * time_slab_count + slab_id) * mask_word_count;

  float depths[WF2_MAX_REALRAY_BOUNDARIES];
  uint depth_count = 0u;
  const uint clamped_boundary_count = min(boundary_count, WF2_MAX_REALRAY_BOUNDARIES);
  for (uint boundary_id = 0u; boundary_id < clamped_boundary_count; ++boundary_id) {
    if (!wf2_bitset_has_boundary(candidate_mask_u32, mask_base, boundary_id)) {
      continue;
    }
    const uint base = boundary_id * 5u;
    const float nx = boundary_f32[base + 0u];
    const float ny = boundary_f32[base + 1u];
    const float nz = boundary_f32[base + 2u];
    const float nt = boundary_f32[base + 3u];
    const float b = boundary_f32[base + 4u];
    const float denom = nx * dx + ny * dy + nz * dz;
    if (fabs(denom) < invalid_epsilon) {
      continue;
    }
    const float depth = -(nx * ox + ny * oy + nz * oz + nt * t + b) / denom;
    if (isfinite(depth) && depth >= near_depth && depth <= far_depth) {
      wf2_realray_insert_depth(depths, depth_count, depth);
    }
  }

  const uint clamped_site_count = min(site_count, WF2_MAX_REALRAY_SITES);
  float3 rgb_accum = float3(0.0f, 0.0f, 0.0f);
  float alpha_accum = 0.0f;
  float depth_weighted = 0.0f;
  float transmittance = 1.0f;
  float previous_depth = near_depth;
  for (uint cut_id = 0u; cut_id <= depth_count; ++cut_id) {
    const float next_depth = cut_id < depth_count ? depths[cut_id] : far_depth;
    const float length = next_depth - previous_depth;
    if (length > 1.0e-8f && transmittance > transmittance_threshold) {
      const float mid_depth = 0.5f * (previous_depth + next_depth);
      const float x = ox + dx * mid_depth;
      const float y = oy + dy * mid_depth;
      const float z = oz + dz * mid_depth;
      const uint owner = wf2_realray_owner_at(sites_f32, clamped_site_count, x, y, z, t);
      const uint rgba_base = owner * 4u;
      const float density = max(site_rgba_f32[rgba_base + 3u], 0.0f);
      const float segment_transmittance = exp(-density * length);
      const float segment_alpha = 1.0f - segment_transmittance;
      const float weight = transmittance * segment_alpha;
      const float3 rgb = float3(
          site_rgba_f32[rgba_base + 0u],
          site_rgba_f32[rgba_base + 1u],
          site_rgba_f32[rgba_base + 2u]);
      rgb_accum += weight * rgb;
      alpha_accum += weight;
      depth_weighted += weight * mid_depth;
      transmittance *= segment_transmittance;
    }
    previous_depth = next_depth;
  }

  const uint out_base = sample_id * 3u;
  output_rgb_f32[out_base + 0u] = rgb_accum.x;
  output_rgb_f32[out_base + 1u] = rgb_accum.y;
  output_rgb_f32[out_base + 2u] = rgb_accum.z;
  output_alpha_f32[sample_id] = alpha_accum;
  output_depth_f32[sample_id] = alpha_accum > 1.0e-8f ? depth_weighted / alpha_accum : far_depth;
}

kernel void wf2_fused_slab_affine_realray_rgba_depth_replay_tensor(
    device const float* boundary_f32 [[buffer(0)]],
    device const int* row_index_i32 [[buffer(1)]],
    device const int* candidate_row_offsets_i32 [[buffer(2)]],
    device const int* candidate_boundary_ids_i32 [[buffer(3)]],
    device const float* sites_f32 [[buffer(4)]],
    device const float* site_rgba_f32 [[buffer(5)]],
    device const float* ray_coeff_f32 [[buffer(6)]],
    device const float* frame_t_f32 [[buffer(7)]],
    device const int* config_i32 [[buffer(8)]],
    device const float* config_f32 [[buffer(9)]],
    device float* output_rgb_f32 [[buffer(10)]],
    device float* output_alpha_f32 [[buffer(11)]],
    device float* output_depth_f32 [[buffer(12)]],
    uint sample_id [[thread_position_in_grid]]) {
  const uint boundary_count = uint(config_i32[0]);
  const uint track_count = uint(config_i32[1]);
  const uint site_count = uint(config_i32[2]);
  const uint frame_count = uint(config_i32[3]);
  const uint time_slab_count = uint(config_i32[4]);
  const uint row_count = uint(config_i32[5]);
  const uint candidate_count = uint(config_i32[6]);
  const uint total_samples = track_count * frame_count;
  if (sample_id >= total_samples) {
    return;
  }

  const float near_depth = config_f32[0];
  const float far_depth = config_f32[1];
  const float invalid_epsilon = config_f32[2];
  const float transmittance_threshold = config_f32[3];
  const uint track_id = sample_id / frame_count;
  const uint frame_id = sample_id - track_id * frame_count;
  const float t = frame_t_f32[frame_id];
  const float3 origin = wf2_affine_origin_at(ray_coeff_f32, track_id, t);
  const float3 direction = wf2_affine_direction_at(ray_coeff_f32, track_id, t);
  const uint slab_id = wf2_replay_slab_id(t, time_slab_count);
  const uint out_base = sample_id * 3u;

  const int row_index_raw = row_index_i32[track_id];
  if (row_index_raw < 0 || uint(row_index_raw) >= row_count) {
    output_rgb_f32[out_base + 0u] = 0.0f;
    output_rgb_f32[out_base + 1u] = 0.0f;
    output_rgb_f32[out_base + 2u] = 0.0f;
    output_alpha_f32[sample_id] = 0.0f;
    output_depth_f32[sample_id] = far_depth;
    return;
  }

  const uint row = uint(row_index_raw) * time_slab_count + slab_id;
  const int begin_raw = candidate_row_offsets_i32[row];
  const int end_raw = candidate_row_offsets_i32[row + 1u];
  if (begin_raw < 0 || end_raw < begin_raw || uint(end_raw) > candidate_count) {
    output_rgb_f32[out_base + 0u] = 0.0f;
    output_rgb_f32[out_base + 1u] = 0.0f;
    output_rgb_f32[out_base + 2u] = 0.0f;
    output_alpha_f32[sample_id] = 0.0f;
    output_depth_f32[sample_id] = far_depth;
    return;
  }

  float depths[WF2_MAX_REALRAY_BOUNDARIES];
  uint depth_count = 0u;
  for (uint cursor = uint(begin_raw); cursor < uint(end_raw); ++cursor) {
    const int boundary_raw = candidate_boundary_ids_i32[cursor];
    if (boundary_raw < 0 || uint(boundary_raw) >= boundary_count) {
      continue;
    }
    const uint boundary_id = uint(boundary_raw);
    const uint base = boundary_id * 5u;
    const float nx = boundary_f32[base + 0u];
    const float ny = boundary_f32[base + 1u];
    const float nz = boundary_f32[base + 2u];
    const float nt = boundary_f32[base + 3u];
    const float b = boundary_f32[base + 4u];
    const float denom = nx * direction.x + ny * direction.y + nz * direction.z;
    if (fabs(denom) < invalid_epsilon) {
      continue;
    }
    const float depth = -(nx * origin.x + ny * origin.y + nz * origin.z + nt * t + b) / denom;
    if (isfinite(depth) && depth >= near_depth && depth <= far_depth) {
      wf2_realray_insert_depth(depths, depth_count, depth);
    }
  }

  const uint clamped_site_count = min(site_count, WF2_MAX_REALRAY_SITES);
  float3 rgb_accum = float3(0.0f, 0.0f, 0.0f);
  float alpha_accum = 0.0f;
  float depth_weighted = 0.0f;
  float transmittance = 1.0f;
  float previous_depth = near_depth;
  for (uint cut_id = 0u; cut_id <= depth_count; ++cut_id) {
    const float next_depth = cut_id < depth_count ? depths[cut_id] : far_depth;
    const float length = next_depth - previous_depth;
    if (length > 1.0e-8f && transmittance > transmittance_threshold) {
      const float mid_depth = 0.5f * (previous_depth + next_depth);
      const float x = origin.x + direction.x * mid_depth;
      const float y = origin.y + direction.y * mid_depth;
      const float z = origin.z + direction.z * mid_depth;
      const uint owner = wf2_realray_owner_at(sites_f32, clamped_site_count, x, y, z, t);
      const uint rgba_base = owner * 4u;
      const float density = max(site_rgba_f32[rgba_base + 3u], 0.0f);
      const float segment_transmittance = exp(-density * length);
      const float segment_alpha = 1.0f - segment_transmittance;
      const float weight = transmittance * segment_alpha;
      const float3 rgb = float3(
          site_rgba_f32[rgba_base + 0u],
          site_rgba_f32[rgba_base + 1u],
          site_rgba_f32[rgba_base + 2u]);
      rgb_accum += weight * rgb;
      alpha_accum += weight;
      depth_weighted += weight * mid_depth;
      transmittance *= segment_transmittance;
    }
    previous_depth = next_depth;
  }

  output_rgb_f32[out_base + 0u] = rgb_accum.x;
  output_rgb_f32[out_base + 1u] = rgb_accum.y;
  output_rgb_f32[out_base + 2u] = rgb_accum.z;
  output_alpha_f32[sample_id] = alpha_accum;
  output_depth_f32[sample_id] = alpha_accum > 1.0e-8f ? depth_weighted / alpha_accum : far_depth;
}

kernel void wf2_fused_slab_affine_coeff_realray_rgba_depth_replay_tensor(
    device const int* row_index_i32 [[buffer(0)]],
    device const int* candidate_row_offsets_i32 [[buffer(1)]],
    device const float* candidate_depth_coeff_f32 [[buffer(2)]],
    device const float* sites_f32 [[buffer(3)]],
    device const float* site_rgba_f32 [[buffer(4)]],
    device const float* ray_coeff_f32 [[buffer(5)]],
    device const float* frame_t_f32 [[buffer(6)]],
    device const int* config_i32 [[buffer(7)]],
    device const float* config_f32 [[buffer(8)]],
    device float* output_rgb_f32 [[buffer(9)]],
    device float* output_alpha_f32 [[buffer(10)]],
    device float* output_depth_f32 [[buffer(11)]],
    uint sample_id [[thread_position_in_grid]]) {
  const uint track_count = uint(config_i32[0]);
  const uint site_count = uint(config_i32[1]);
  const uint frame_count = uint(config_i32[2]);
  const uint time_slab_count = uint(config_i32[3]);
  const uint row_count = uint(config_i32[4]);
  const uint candidate_count = uint(config_i32[5]);
  const uint total_samples = track_count * frame_count;
  if (sample_id >= total_samples) {
    return;
  }

  const float near_depth = config_f32[0];
  const float far_depth = config_f32[1];
  const float invalid_epsilon = config_f32[2];
  const float transmittance_threshold = config_f32[3];
  const uint track_id = sample_id / frame_count;
  const uint frame_id = sample_id - track_id * frame_count;
  const float t = frame_t_f32[frame_id];
  const float3 origin = wf2_affine_origin_at(ray_coeff_f32, track_id, t);
  const float3 direction = wf2_affine_direction_at(ray_coeff_f32, track_id, t);
  const uint slab_id = wf2_replay_slab_id(t, time_slab_count);
  const uint out_base = sample_id * 3u;

  const int row_index_raw = row_index_i32[track_id];
  if (row_index_raw < 0 || uint(row_index_raw) >= row_count) {
    output_rgb_f32[out_base + 0u] = 0.0f;
    output_rgb_f32[out_base + 1u] = 0.0f;
    output_rgb_f32[out_base + 2u] = 0.0f;
    output_alpha_f32[sample_id] = 0.0f;
    output_depth_f32[sample_id] = far_depth;
    return;
  }

  const uint row = uint(row_index_raw) * time_slab_count + slab_id;
  const int begin_raw = candidate_row_offsets_i32[row];
  const int end_raw = candidate_row_offsets_i32[row + 1u];
  if (begin_raw < 0 || end_raw < begin_raw || uint(end_raw) > candidate_count) {
    output_rgb_f32[out_base + 0u] = 0.0f;
    output_rgb_f32[out_base + 1u] = 0.0f;
    output_rgb_f32[out_base + 2u] = 0.0f;
    output_alpha_f32[sample_id] = 0.0f;
    output_depth_f32[sample_id] = far_depth;
    return;
  }

  float depths[WF2_MAX_REALRAY_FUSED_MSE_BOUNDARIES];
  uint depth_count = 0u;
  for (uint cursor = uint(begin_raw); cursor < uint(end_raw); ++cursor) {
    const uint coeff_base = cursor * 4u;
    const float numer = candidate_depth_coeff_f32[coeff_base + 0u] + candidate_depth_coeff_f32[coeff_base + 1u] * t;
    const float denom = candidate_depth_coeff_f32[coeff_base + 2u] + candidate_depth_coeff_f32[coeff_base + 3u] * t;
    if (fabs(denom) < invalid_epsilon) {
      continue;
    }
    const float depth = numer / denom;
    if (isfinite(depth) && depth >= near_depth && depth <= far_depth) {
      wf2_realray_insert_depth_fused_mse(depths, depth_count, depth);
    }
  }

  const uint clamped_site_count = min(site_count, WF2_MAX_REALRAY_SITES);
  float3 rgb_accum = float3(0.0f, 0.0f, 0.0f);
  float alpha_accum = 0.0f;
  float depth_weighted = 0.0f;
  float transmittance = 1.0f;
  float previous_depth = near_depth;
  for (uint cut_id = 0u; cut_id <= depth_count; ++cut_id) {
    const float next_depth = cut_id < depth_count ? depths[cut_id] : far_depth;
    const float length = next_depth - previous_depth;
    if (length > 1.0e-8f && transmittance > transmittance_threshold) {
      const float mid_depth = 0.5f * (previous_depth + next_depth);
      const float x = origin.x + direction.x * mid_depth;
      const float y = origin.y + direction.y * mid_depth;
      const float z = origin.z + direction.z * mid_depth;
      const uint owner = wf2_realray_owner_at(sites_f32, clamped_site_count, x, y, z, t);
      const uint rgba_base = owner * 4u;
      const float density = max(site_rgba_f32[rgba_base + 3u], 0.0f);
      const float segment_transmittance = exp(-density * length);
      const float segment_alpha = 1.0f - segment_transmittance;
      const float weight = transmittance * segment_alpha;
      const float3 rgb = float3(
          site_rgba_f32[rgba_base + 0u],
          site_rgba_f32[rgba_base + 1u],
          site_rgba_f32[rgba_base + 2u]);
      rgb_accum += weight * rgb;
      alpha_accum += weight;
      depth_weighted += weight * mid_depth;
      transmittance *= segment_transmittance;
    }
    previous_depth = next_depth;
  }

  output_rgb_f32[out_base + 0u] = rgb_accum.x;
  output_rgb_f32[out_base + 1u] = rgb_accum.y;
  output_rgb_f32[out_base + 2u] = rgb_accum.z;
  output_alpha_f32[sample_id] = alpha_accum;
  output_depth_f32[sample_id] = alpha_accum > 1.0e-8f ? depth_weighted / alpha_accum : far_depth;
}

kernel void wf2_fused_slab_affine_coeff16_realray_rgba_depth_replay_tensor(
    device const int* row_index_i32 [[buffer(0)]],
    device const int* candidate_row_offsets_i32 [[buffer(1)]],
    device const half* candidate_depth_coeff_f16 [[buffer(2)]],
    device const float* sites_f32 [[buffer(3)]],
    device const float* site_rgba_f32 [[buffer(4)]],
    device const float* ray_coeff_f32 [[buffer(5)]],
    device const float* frame_t_f32 [[buffer(6)]],
    device const int* config_i32 [[buffer(7)]],
    device const float* config_f32 [[buffer(8)]],
    device float* output_rgb_f32 [[buffer(9)]],
    device float* output_alpha_f32 [[buffer(10)]],
    device float* output_depth_f32 [[buffer(11)]],
    uint sample_id [[thread_position_in_grid]]) {
  const uint track_count = uint(config_i32[0]);
  const uint site_count = uint(config_i32[1]);
  const uint frame_count = uint(config_i32[2]);
  const uint time_slab_count = uint(config_i32[3]);
  const uint row_count = uint(config_i32[4]);
  const uint candidate_count = uint(config_i32[5]);
  const uint total_samples = track_count * frame_count;
  if (sample_id >= total_samples) {
    return;
  }

  const float near_depth = config_f32[0];
  const float far_depth = config_f32[1];
  const float invalid_epsilon = config_f32[2];
  const float transmittance_threshold = config_f32[3];
  const uint track_id = sample_id / frame_count;
  const uint frame_id = sample_id - track_id * frame_count;
  const float t = frame_t_f32[frame_id];
  const float3 origin = wf2_affine_origin_at(ray_coeff_f32, track_id, t);
  const float3 direction = wf2_affine_direction_at(ray_coeff_f32, track_id, t);
  const uint slab_id = wf2_replay_slab_id(t, time_slab_count);
  const uint out_base = sample_id * 3u;

  const int row_index_raw = row_index_i32[track_id];
  if (row_index_raw < 0 || uint(row_index_raw) >= row_count) {
    output_rgb_f32[out_base + 0u] = 0.0f;
    output_rgb_f32[out_base + 1u] = 0.0f;
    output_rgb_f32[out_base + 2u] = 0.0f;
    output_alpha_f32[sample_id] = 0.0f;
    output_depth_f32[sample_id] = far_depth;
    return;
  }

  const uint row = uint(row_index_raw) * time_slab_count + slab_id;
  const int begin_raw = candidate_row_offsets_i32[row];
  const int end_raw = candidate_row_offsets_i32[row + 1u];
  if (begin_raw < 0 || end_raw < begin_raw || uint(end_raw) > candidate_count) {
    output_rgb_f32[out_base + 0u] = 0.0f;
    output_rgb_f32[out_base + 1u] = 0.0f;
    output_rgb_f32[out_base + 2u] = 0.0f;
    output_alpha_f32[sample_id] = 0.0f;
    output_depth_f32[sample_id] = far_depth;
    return;
  }

  float depths[WF2_MAX_REALRAY_FUSED_MSE_BOUNDARIES];
  uint depth_count = 0u;
  for (uint cursor = uint(begin_raw); cursor < uint(end_raw); ++cursor) {
    const uint coeff_base = cursor * 4u;
    const float numer =
        float(candidate_depth_coeff_f16[coeff_base + 0u]) + float(candidate_depth_coeff_f16[coeff_base + 1u]) * t;
    const float denom =
        float(candidate_depth_coeff_f16[coeff_base + 2u]) + float(candidate_depth_coeff_f16[coeff_base + 3u]) * t;
    if (fabs(denom) < invalid_epsilon) {
      continue;
    }
    const float depth = numer / denom;
    if (isfinite(depth) && depth >= near_depth && depth <= far_depth) {
      wf2_realray_insert_depth_fused_mse(depths, depth_count, depth);
    }
  }

  const uint clamped_site_count = min(site_count, WF2_MAX_REALRAY_SITES);
  float3 rgb_accum = float3(0.0f, 0.0f, 0.0f);
  float alpha_accum = 0.0f;
  float depth_weighted = 0.0f;
  float transmittance = 1.0f;
  float previous_depth = near_depth;
  for (uint cut_id = 0u; cut_id <= depth_count; ++cut_id) {
    const float next_depth = cut_id < depth_count ? depths[cut_id] : far_depth;
    const float length = next_depth - previous_depth;
    if (length > 1.0e-8f && transmittance > transmittance_threshold) {
      const float mid_depth = 0.5f * (previous_depth + next_depth);
      const float x = origin.x + direction.x * mid_depth;
      const float y = origin.y + direction.y * mid_depth;
      const float z = origin.z + direction.z * mid_depth;
      const uint owner = wf2_realray_owner_at(sites_f32, clamped_site_count, x, y, z, t);
      const uint rgba_base = owner * 4u;
      const float density = max(site_rgba_f32[rgba_base + 3u], 0.0f);
      const float segment_transmittance = exp(-density * length);
      const float segment_alpha = 1.0f - segment_transmittance;
      const float weight = transmittance * segment_alpha;
      const float3 rgb = float3(
          site_rgba_f32[rgba_base + 0u],
          site_rgba_f32[rgba_base + 1u],
          site_rgba_f32[rgba_base + 2u]);
      rgb_accum += weight * rgb;
      alpha_accum += weight;
      depth_weighted += weight * mid_depth;
      transmittance *= segment_transmittance;
    }
    previous_depth = next_depth;
  }

  output_rgb_f32[out_base + 0u] = rgb_accum.x;
  output_rgb_f32[out_base + 1u] = rgb_accum.y;
  output_rgb_f32[out_base + 2u] = rgb_accum.z;
  output_alpha_f32[sample_id] = alpha_accum;
  output_depth_f32[sample_id] = alpha_accum > 1.0e-8f ? depth_weighted / alpha_accum : far_depth;
}

kernel void wf2_fused_slab_affine_num32_den16_realray_rgba_depth_replay_tensor(
    device const int* row_index_i32 [[buffer(0)]],
    device const int* candidate_row_offsets_i32 [[buffer(1)]],
    device const float* candidate_depth_num_f32 [[buffer(2)]],
    device const half* candidate_depth_den_f16 [[buffer(3)]],
    device const float* sites_f32 [[buffer(4)]],
    device const float* site_rgba_f32 [[buffer(5)]],
    device const float* ray_coeff_f32 [[buffer(6)]],
    device const float* frame_t_f32 [[buffer(7)]],
    device const int* config_i32 [[buffer(8)]],
    device const float* config_f32 [[buffer(9)]],
    device float* output_rgb_f32 [[buffer(10)]],
    device float* output_alpha_f32 [[buffer(11)]],
    device float* output_depth_f32 [[buffer(12)]],
    uint sample_id [[thread_position_in_grid]]) {
  const uint track_count = uint(config_i32[0]);
  const uint site_count = uint(config_i32[1]);
  const uint frame_count = uint(config_i32[2]);
  const uint time_slab_count = uint(config_i32[3]);
  const uint row_count = uint(config_i32[4]);
  const uint candidate_count = uint(config_i32[5]);
  const uint total_samples = track_count * frame_count;
  if (sample_id >= total_samples) {
    return;
  }

  const float near_depth = config_f32[0];
  const float far_depth = config_f32[1];
  const float invalid_epsilon = config_f32[2];
  const float transmittance_threshold = config_f32[3];
  const uint track_id = sample_id / frame_count;
  const uint frame_id = sample_id - track_id * frame_count;
  const float t = frame_t_f32[frame_id];
  const float3 origin = wf2_affine_origin_at(ray_coeff_f32, track_id, t);
  const float3 direction = wf2_affine_direction_at(ray_coeff_f32, track_id, t);
  const uint slab_id = wf2_replay_slab_id(t, time_slab_count);
  const uint out_base = sample_id * 3u;

  const int row_index_raw = row_index_i32[track_id];
  if (row_index_raw < 0 || uint(row_index_raw) >= row_count) {
    output_rgb_f32[out_base + 0u] = 0.0f;
    output_rgb_f32[out_base + 1u] = 0.0f;
    output_rgb_f32[out_base + 2u] = 0.0f;
    output_alpha_f32[sample_id] = 0.0f;
    output_depth_f32[sample_id] = far_depth;
    return;
  }

  const uint row = uint(row_index_raw) * time_slab_count + slab_id;
  const int begin_raw = candidate_row_offsets_i32[row];
  const int end_raw = candidate_row_offsets_i32[row + 1u];
  if (begin_raw < 0 || end_raw < begin_raw || uint(end_raw) > candidate_count) {
    output_rgb_f32[out_base + 0u] = 0.0f;
    output_rgb_f32[out_base + 1u] = 0.0f;
    output_rgb_f32[out_base + 2u] = 0.0f;
    output_alpha_f32[sample_id] = 0.0f;
    output_depth_f32[sample_id] = far_depth;
    return;
  }

  float depths[WF2_MAX_REALRAY_FUSED_MSE_BOUNDARIES];
  uint depth_count = 0u;
  for (uint cursor = uint(begin_raw); cursor < uint(end_raw); ++cursor) {
    const uint coeff_base = cursor * 2u;
    const float numer = candidate_depth_num_f32[coeff_base + 0u] + candidate_depth_num_f32[coeff_base + 1u] * t;
    const float denom = float(candidate_depth_den_f16[coeff_base + 0u]) +
        float(candidate_depth_den_f16[coeff_base + 1u]) * t;
    if (fabs(denom) < invalid_epsilon) {
      continue;
    }
    const float depth = numer / denom;
    if (isfinite(depth) && depth >= near_depth && depth <= far_depth) {
      wf2_realray_insert_depth_fused_mse(depths, depth_count, depth);
    }
  }

  const uint clamped_site_count = min(site_count, WF2_MAX_REALRAY_SITES);
  float3 rgb_accum = float3(0.0f, 0.0f, 0.0f);
  float alpha_accum = 0.0f;
  float depth_weighted = 0.0f;
  float transmittance = 1.0f;
  float previous_depth = near_depth;
  for (uint cut_id = 0u; cut_id <= depth_count; ++cut_id) {
    const float next_depth = cut_id < depth_count ? depths[cut_id] : far_depth;
    const float length = next_depth - previous_depth;
    if (length > 1.0e-8f && transmittance > transmittance_threshold) {
      const float mid_depth = 0.5f * (previous_depth + next_depth);
      const float x = origin.x + direction.x * mid_depth;
      const float y = origin.y + direction.y * mid_depth;
      const float z = origin.z + direction.z * mid_depth;
      const uint owner = wf2_realray_owner_at(sites_f32, clamped_site_count, x, y, z, t);
      const uint rgba_base = owner * 4u;
      const float density = max(site_rgba_f32[rgba_base + 3u], 0.0f);
      const float segment_transmittance = exp(-density * length);
      const float segment_alpha = 1.0f - segment_transmittance;
      const float weight = transmittance * segment_alpha;
      const float3 rgb = float3(
          site_rgba_f32[rgba_base + 0u],
          site_rgba_f32[rgba_base + 1u],
          site_rgba_f32[rgba_base + 2u]);
      rgb_accum += weight * rgb;
      alpha_accum += weight;
      depth_weighted += weight * mid_depth;
      transmittance *= segment_transmittance;
    }
    previous_depth = next_depth;
  }

  output_rgb_f32[out_base + 0u] = rgb_accum.x;
  output_rgb_f32[out_base + 1u] = rgb_accum.y;
  output_rgb_f32[out_base + 2u] = rgb_accum.z;
  output_alpha_f32[sample_id] = alpha_accum;
  output_depth_f32[sample_id] = alpha_accum > 1.0e-8f ? depth_weighted / alpha_accum : far_depth;
}

kernel void wf2_fused_slab_affine_num32_den16_ownerupdate_realray_rgba_depth_replay_tensor(
    device const int* row_index_i32 [[buffer(0)]],
    device const int* candidate_row_offsets_i32 [[buffer(1)]],
    device const int* candidate_boundary_ids_i32 [[buffer(2)]],
    device const float* candidate_depth_num_f32 [[buffer(3)]],
    device const half* candidate_depth_den_f16 [[buffer(4)]],
    device const int* boundary_site_pairs_i32 [[buffer(5)]],
    device const float* sites_f32 [[buffer(6)]],
    device const float* site_rgba_f32 [[buffer(7)]],
    device const float* ray_coeff_f32 [[buffer(8)]],
    device const float* frame_t_f32 [[buffer(9)]],
    device const int* config_i32 [[buffer(10)]],
    device const float* config_f32 [[buffer(11)]],
    device float* output_rgb_f32 [[buffer(12)]],
    device float* output_alpha_f32 [[buffer(13)]],
    device float* output_depth_f32 [[buffer(14)]],
    uint sample_id [[thread_position_in_grid]]) {
  const uint track_count = uint(config_i32[0]);
  const uint site_count = uint(config_i32[1]);
  const uint frame_count = uint(config_i32[2]);
  const uint time_slab_count = uint(config_i32[3]);
  const uint row_count = uint(config_i32[4]);
  const uint candidate_count = uint(config_i32[5]);
  const uint boundary_count = uint(config_i32[6]);
  const uint total_samples = track_count * frame_count;
  if (sample_id >= total_samples) {
    return;
  }

  const float near_depth = config_f32[0];
  const float far_depth = config_f32[1];
  const float invalid_epsilon = config_f32[2];
  const float transmittance_threshold = config_f32[3];
  const uint track_id = sample_id / frame_count;
  const uint frame_id = sample_id - track_id * frame_count;
  const float t = frame_t_f32[frame_id];
  const float3 origin = wf2_affine_origin_at(ray_coeff_f32, track_id, t);
  const float3 direction = wf2_affine_direction_at(ray_coeff_f32, track_id, t);
  const uint slab_id = wf2_replay_slab_id(t, time_slab_count);
  const uint out_base = sample_id * 3u;

  const int row_index_raw = row_index_i32[track_id];
  if (row_index_raw < 0 || uint(row_index_raw) >= row_count) {
    output_rgb_f32[out_base + 0u] = 0.0f;
    output_rgb_f32[out_base + 1u] = 0.0f;
    output_rgb_f32[out_base + 2u] = 0.0f;
    output_alpha_f32[sample_id] = 0.0f;
    output_depth_f32[sample_id] = far_depth;
    return;
  }

  const uint row = uint(row_index_raw) * time_slab_count + slab_id;
  const int begin_raw = candidate_row_offsets_i32[row];
  const int end_raw = candidate_row_offsets_i32[row + 1u];
  if (begin_raw < 0 || end_raw < begin_raw || uint(end_raw) > candidate_count) {
    output_rgb_f32[out_base + 0u] = 0.0f;
    output_rgb_f32[out_base + 1u] = 0.0f;
    output_rgb_f32[out_base + 2u] = 0.0f;
    output_alpha_f32[sample_id] = 0.0f;
    output_depth_f32[sample_id] = far_depth;
    return;
  }

  float depths[WF2_MAX_REALRAY_BOUNDARIES];
  uint boundary_ids[WF2_MAX_REALRAY_BOUNDARIES];
  uint depth_count = 0u;
  for (uint cursor = uint(begin_raw); cursor < uint(end_raw); ++cursor) {
    const int boundary_id_raw = candidate_boundary_ids_i32[cursor];
    if (boundary_id_raw < 0 || uint(boundary_id_raw) >= boundary_count) {
      continue;
    }
    const uint coeff_base = cursor * 2u;
    const float numer = candidate_depth_num_f32[coeff_base + 0u] + candidate_depth_num_f32[coeff_base + 1u] * t;
    const float denom = float(candidate_depth_den_f16[coeff_base + 0u]) +
        float(candidate_depth_den_f16[coeff_base + 1u]) * t;
    if (fabs(denom) < invalid_epsilon) {
      continue;
    }
    const float depth = numer / denom;
    if (isfinite(depth) && depth >= near_depth && depth <= far_depth) {
      wf2_realray_insert_depth_with_boundary(depths, boundary_ids, depth_count, depth, uint(boundary_id_raw));
    }
  }

  const uint clamped_site_count = min(site_count, WF2_MAX_REALRAY_SITES);
  float3 rgb_accum = float3(0.0f, 0.0f, 0.0f);
  float alpha_accum = 0.0f;
  float depth_weighted = 0.0f;
  float transmittance = 1.0f;
  float previous_depth = near_depth;

  for (uint cut_id = 0u; cut_id <= depth_count; ++cut_id) {
    const float next_depth = cut_id < depth_count ? depths[cut_id] : far_depth;
    const float length = next_depth - previous_depth;
    if (length > 1.0e-8f && transmittance > transmittance_threshold) {
      const float mid_depth = 0.5f * (previous_depth + next_depth);
      const uint owner = wf2_realray_owner_at(
          sites_f32,
          clamped_site_count,
          origin.x + direction.x * mid_depth,
          origin.y + direction.y * mid_depth,
          origin.z + direction.z * mid_depth,
          t);
      const uint rgba_base = owner * 4u;
      const float density = max(site_rgba_f32[rgba_base + 3u], 0.0f);
      const float segment_transmittance = exp(-density * length);
      const float segment_alpha = 1.0f - segment_transmittance;
      const float weight = transmittance * segment_alpha;
      const float3 rgb = float3(
          site_rgba_f32[rgba_base + 0u],
          site_rgba_f32[rgba_base + 1u],
          site_rgba_f32[rgba_base + 2u]);
      rgb_accum += weight * rgb;
      alpha_accum += weight;
      depth_weighted += weight * mid_depth;
      transmittance *= segment_transmittance;
    }
    previous_depth = next_depth;
  }

  output_rgb_f32[out_base + 0u] = rgb_accum.x;
  output_rgb_f32[out_base + 1u] = rgb_accum.y;
  output_rgb_f32[out_base + 2u] = rgb_accum.z;
  output_alpha_f32[sample_id] = alpha_accum;
  output_depth_f32[sample_id] = alpha_accum > 1.0e-8f ? depth_weighted / alpha_accum : far_depth;
}

kernel void wf2_shared_realray_rgba_depth_vjp_tensor(
    device const float* boundary_f32 [[buffer(0)]],
    device const uint* candidate_mask_u32 [[buffer(1)]],
    device const float* sites_f32 [[buffer(2)]],
    device const float* site_rgba_f32 [[buffer(3)]],
    device const float* track_rays_f32 [[buffer(4)]],
    device const float* frame_t_f32 [[buffer(5)]],
    device const float* grad_rgb_f32 [[buffer(6)]],
    device const float* grad_alpha_f32 [[buffer(7)]],
    device const float* grad_depth_f32 [[buffer(8)]],
    device const int* config_i32 [[buffer(9)]],
    device const float* config_f32 [[buffer(10)]],
    device float* output_rgb_f32 [[buffer(11)]],
    device float* output_alpha_f32 [[buffer(12)]],
    device float* output_depth_f32 [[buffer(13)]],
    device float* grad_sample_rgba_f32 [[buffer(14)]],
    uint sample_id [[thread_position_in_grid]]) {
  const uint boundary_count = uint(config_i32[0]);
  const uint track_count = uint(config_i32[1]);
  const uint site_count = uint(config_i32[2]);
  const uint frame_count = uint(config_i32[3]);
  const uint time_slab_count = uint(config_i32[4]);
  const uint mask_word_count = uint(config_i32[5]);
  const uint total_samples = track_count * frame_count;
  if (sample_id >= total_samples) {
    return;
  }

  const float near_depth = config_f32[0];
  const float far_depth = config_f32[1];
  const float invalid_epsilon = config_f32[2];
  const float transmittance_threshold = config_f32[3];
  const uint track_id = sample_id / frame_count;
  const uint frame_id = sample_id - track_id * frame_count;
  const uint ray_base = track_id * 6u;
  const float ox = track_rays_f32[ray_base + 0u];
  const float oy = track_rays_f32[ray_base + 1u];
  const float oz = track_rays_f32[ray_base + 2u];
  const float dx = track_rays_f32[ray_base + 3u];
  const float dy = track_rays_f32[ray_base + 4u];
  const float dz = track_rays_f32[ray_base + 5u];
  const float t = frame_t_f32[frame_id];
  const uint slab_id = wf2_replay_slab_id(t, time_slab_count);
  const uint mask_base = (track_id * time_slab_count + slab_id) * mask_word_count;

  float depths[WF2_MAX_REALRAY_BOUNDARIES];
  uint depth_count = 0u;
  const uint clamped_boundary_count = min(boundary_count, WF2_MAX_REALRAY_BOUNDARIES);
  for (uint boundary_id = 0u; boundary_id < clamped_boundary_count; ++boundary_id) {
    if (!wf2_bitset_has_boundary(candidate_mask_u32, mask_base, boundary_id)) {
      continue;
    }
    const uint base = boundary_id * 5u;
    const float nx = boundary_f32[base + 0u];
    const float ny = boundary_f32[base + 1u];
    const float nz = boundary_f32[base + 2u];
    const float nt = boundary_f32[base + 3u];
    const float b = boundary_f32[base + 4u];
    const float denom = nx * dx + ny * dy + nz * dz;
    if (fabs(denom) < invalid_epsilon) {
      continue;
    }
    const float depth = -(nx * ox + ny * oy + nz * oz + nt * t + b) / denom;
    if (isfinite(depth) && depth >= near_depth && depth <= far_depth) {
      wf2_realray_insert_depth(depths, depth_count, depth);
    }
  }

  const uint clamped_site_count = min(site_count, WF2_MAX_REALRAY_SITES);
  const uint grad_base = sample_id * site_count * 4u;
  for (uint site_id = 0u; site_id < clamped_site_count; ++site_id) {
    const uint site_base = grad_base + site_id * 4u;
    grad_sample_rgba_f32[site_base + 0u] = 0.0f;
    grad_sample_rgba_f32[site_base + 1u] = 0.0f;
    grad_sample_rgba_f32[site_base + 2u] = 0.0f;
    grad_sample_rgba_f32[site_base + 3u] = 0.0f;
  }

  uint owners[WF2_MAX_REALRAY_SEGMENTS];
  float lengths[WF2_MAX_REALRAY_SEGMENTS];
  float mids[WF2_MAX_REALRAY_SEGMENTS];
  float trans_before[WF2_MAX_REALRAY_SEGMENTS];
  float segment_trans[WF2_MAX_REALRAY_SEGMENTS];
  float segment_alpha[WF2_MAX_REALRAY_SEGMENTS];
  float weights[WF2_MAX_REALRAY_SEGMENTS];
  float3 segment_rgb[WF2_MAX_REALRAY_SEGMENTS];

  float3 rgb_accum = float3(0.0f, 0.0f, 0.0f);
  float alpha_accum = 0.0f;
  float depth_weighted = 0.0f;
  float transmittance = 1.0f;
  float previous_depth = near_depth;
  uint segment_count = 0u;
  for (uint cut_id = 0u; cut_id <= depth_count; ++cut_id) {
    const float next_depth = cut_id < depth_count ? depths[cut_id] : far_depth;
    const float length = next_depth - previous_depth;
    if (length > 1.0e-8f && transmittance > transmittance_threshold && segment_count < WF2_MAX_REALRAY_SEGMENTS) {
      const float mid_depth = 0.5f * (previous_depth + next_depth);
      const float x = ox + dx * mid_depth;
      const float y = oy + dy * mid_depth;
      const float z = oz + dz * mid_depth;
      const uint owner = wf2_realray_owner_at(sites_f32, clamped_site_count, x, y, z, t);
      const uint rgba_base = owner * 4u;
      const float density = max(site_rgba_f32[rgba_base + 3u], 0.0f);
      const float seg_trans = exp(-density * length);
      const float seg_alpha = 1.0f - seg_trans;
      const float weight = transmittance * seg_alpha;
      const float3 rgb = float3(
          site_rgba_f32[rgba_base + 0u],
          site_rgba_f32[rgba_base + 1u],
          site_rgba_f32[rgba_base + 2u]);

      owners[segment_count] = owner;
      lengths[segment_count] = length;
      mids[segment_count] = mid_depth;
      trans_before[segment_count] = transmittance;
      segment_trans[segment_count] = seg_trans;
      segment_alpha[segment_count] = seg_alpha;
      weights[segment_count] = weight;
      segment_rgb[segment_count] = rgb;
      segment_count += 1u;

      rgb_accum += weight * rgb;
      alpha_accum += weight;
      depth_weighted += weight * mid_depth;
      transmittance *= seg_trans;
    }
    previous_depth = next_depth;
  }

  const uint out_base = sample_id * 3u;
  output_rgb_f32[out_base + 0u] = rgb_accum.x;
  output_rgb_f32[out_base + 1u] = rgb_accum.y;
  output_rgb_f32[out_base + 2u] = rgb_accum.z;
  output_alpha_f32[sample_id] = alpha_accum;
  output_depth_f32[sample_id] = alpha_accum > 1.0e-8f ? depth_weighted / alpha_accum : far_depth;

  const float3 grad_rgb = float3(
      grad_rgb_f32[out_base + 0u],
      grad_rgb_f32[out_base + 1u],
      grad_rgb_f32[out_base + 2u]);
  const float grad_alpha = grad_alpha_f32[sample_id];
  const float grad_depth = grad_depth_f32[sample_id];
  float adj_next_transmittance = 0.0f;
  for (int segment_id = int(segment_count) - 1; segment_id >= 0; --segment_id) {
    const uint owner = owners[segment_id];
    float d_loss_d_weight = dot(grad_rgb, segment_rgb[segment_id]) + grad_alpha;
    if (alpha_accum > 1.0e-8f) {
      d_loss_d_weight += grad_depth *
          (mids[segment_id] * alpha_accum - depth_weighted) /
          (alpha_accum * alpha_accum);
    }

    const uint site_grad_base = grad_base + owner * 4u;
    grad_sample_rgba_f32[site_grad_base + 0u] += weights[segment_id] * grad_rgb.x;
    grad_sample_rgba_f32[site_grad_base + 1u] += weights[segment_id] * grad_rgb.y;
    grad_sample_rgba_f32[site_grad_base + 2u] += weights[segment_id] * grad_rgb.z;

    const float adj_trans_before =
        d_loss_d_weight * segment_alpha[segment_id] +
        adj_next_transmittance * segment_trans[segment_id];
    const float adj_segment_alpha = d_loss_d_weight * trans_before[segment_id];
    const float adj_segment_trans =
        adj_next_transmittance * trans_before[segment_id] - adj_segment_alpha;
    const uint rgba_base = owner * 4u;
    const float raw_density = site_rgba_f32[rgba_base + 3u];
    const float grad_density = raw_density > 0.0f
        ? adj_segment_trans * (-lengths[segment_id] * segment_trans[segment_id])
        : 0.0f;
    grad_sample_rgba_f32[site_grad_base + 3u] += grad_density;
    adj_next_transmittance = adj_trans_before;
  }
}

kernel void wf2_shared_realray_rgba_depth_vjp_reduce_tensor(
    device const float* boundary_f32 [[buffer(0)]],
    device const uint* candidate_mask_u32 [[buffer(1)]],
    device const float* sites_f32 [[buffer(2)]],
    device const float* site_rgba_f32 [[buffer(3)]],
    device const float* track_rays_f32 [[buffer(4)]],
    device const float* frame_t_f32 [[buffer(5)]],
    device const float* grad_rgb_f32 [[buffer(6)]],
    device const float* grad_alpha_f32 [[buffer(7)]],
    device const float* grad_depth_f32 [[buffer(8)]],
    device const int* config_i32 [[buffer(9)]],
    device const float* config_f32 [[buffer(10)]],
    device float* grad_site_rgba_f32 [[buffer(11)]],
    uint site_id [[thread_position_in_grid]]) {
  const uint boundary_count = uint(config_i32[0]);
  const uint track_count = uint(config_i32[1]);
  const uint site_count = uint(config_i32[2]);
  const uint frame_count = uint(config_i32[3]);
  const uint time_slab_count = uint(config_i32[4]);
  const uint mask_word_count = uint(config_i32[5]);
  const uint clamped_site_count = min(site_count, WF2_MAX_REALRAY_SITES);
  if (site_id >= clamped_site_count) {
    return;
  }

  const float near_depth = config_f32[0];
  const float far_depth = config_f32[1];
  const float invalid_epsilon = config_f32[2];
  const float transmittance_threshold = config_f32[3];
  const uint total_samples = track_count * frame_count;
  float4 grad_accum = float4(0.0f, 0.0f, 0.0f, 0.0f);

  for (uint sample_id = 0u; sample_id < total_samples; ++sample_id) {
    const uint track_id = sample_id / frame_count;
    const uint frame_id = sample_id - track_id * frame_count;
    const uint ray_base = track_id * 6u;
    const float ox = track_rays_f32[ray_base + 0u];
    const float oy = track_rays_f32[ray_base + 1u];
    const float oz = track_rays_f32[ray_base + 2u];
    const float dx = track_rays_f32[ray_base + 3u];
    const float dy = track_rays_f32[ray_base + 4u];
    const float dz = track_rays_f32[ray_base + 5u];
    const float t = frame_t_f32[frame_id];
    const uint slab_id = wf2_replay_slab_id(t, time_slab_count);
    const uint mask_base = (track_id * time_slab_count + slab_id) * mask_word_count;

    float depths[WF2_MAX_REALRAY_BOUNDARIES];
    uint depth_count = 0u;
    const uint clamped_boundary_count = min(boundary_count, WF2_MAX_REALRAY_BOUNDARIES);
    for (uint boundary_id = 0u; boundary_id < clamped_boundary_count; ++boundary_id) {
      if (!wf2_bitset_has_boundary(candidate_mask_u32, mask_base, boundary_id)) {
        continue;
      }
      const uint base = boundary_id * 5u;
      const float nx = boundary_f32[base + 0u];
      const float ny = boundary_f32[base + 1u];
      const float nz = boundary_f32[base + 2u];
      const float nt = boundary_f32[base + 3u];
      const float b = boundary_f32[base + 4u];
      const float denom = nx * dx + ny * dy + nz * dz;
      if (fabs(denom) < invalid_epsilon) {
        continue;
      }
      const float depth = -(nx * ox + ny * oy + nz * oz + nt * t + b) / denom;
      if (isfinite(depth) && depth >= near_depth && depth <= far_depth) {
        wf2_realray_insert_depth(depths, depth_count, depth);
      }
    }

    uint owners[WF2_MAX_REALRAY_SEGMENTS];
    float lengths[WF2_MAX_REALRAY_SEGMENTS];
    float mids[WF2_MAX_REALRAY_SEGMENTS];
    float trans_before[WF2_MAX_REALRAY_SEGMENTS];
    float segment_trans[WF2_MAX_REALRAY_SEGMENTS];
    float segment_alpha[WF2_MAX_REALRAY_SEGMENTS];
    float weights[WF2_MAX_REALRAY_SEGMENTS];
    float3 segment_rgb[WF2_MAX_REALRAY_SEGMENTS];

    float alpha_accum = 0.0f;
    float depth_weighted = 0.0f;
    float transmittance = 1.0f;
    float previous_depth = near_depth;
    uint segment_count = 0u;
    for (uint cut_id = 0u; cut_id <= depth_count; ++cut_id) {
      const float next_depth = cut_id < depth_count ? depths[cut_id] : far_depth;
      const float length = next_depth - previous_depth;
      if (length > 1.0e-8f && transmittance > transmittance_threshold && segment_count < WF2_MAX_REALRAY_SEGMENTS) {
        const float mid_depth = 0.5f * (previous_depth + next_depth);
        const float x = ox + dx * mid_depth;
        const float y = oy + dy * mid_depth;
        const float z = oz + dz * mid_depth;
        const uint owner = wf2_realray_owner_at(sites_f32, clamped_site_count, x, y, z, t);
        const uint rgba_base = owner * 4u;
        const float density = max(site_rgba_f32[rgba_base + 3u], 0.0f);
        const float seg_trans = exp(-density * length);
        const float seg_alpha = 1.0f - seg_trans;
        const float weight = transmittance * seg_alpha;
        const float3 rgb = float3(
            site_rgba_f32[rgba_base + 0u],
            site_rgba_f32[rgba_base + 1u],
            site_rgba_f32[rgba_base + 2u]);

        owners[segment_count] = owner;
        lengths[segment_count] = length;
        mids[segment_count] = mid_depth;
        trans_before[segment_count] = transmittance;
        segment_trans[segment_count] = seg_trans;
        segment_alpha[segment_count] = seg_alpha;
        weights[segment_count] = weight;
        segment_rgb[segment_count] = rgb;
        segment_count += 1u;

        alpha_accum += weight;
        depth_weighted += weight * mid_depth;
        transmittance *= seg_trans;
      }
      previous_depth = next_depth;
    }

    const uint out_base = sample_id * 3u;
    const float3 grad_rgb = float3(
        grad_rgb_f32[out_base + 0u],
        grad_rgb_f32[out_base + 1u],
        grad_rgb_f32[out_base + 2u]);
    const float grad_alpha = grad_alpha_f32[sample_id];
    const float grad_depth = grad_depth_f32[sample_id];
    float adj_next_transmittance = 0.0f;
    for (int segment_id = int(segment_count) - 1; segment_id >= 0; --segment_id) {
      const uint owner = owners[segment_id];
      float d_loss_d_weight = dot(grad_rgb, segment_rgb[segment_id]) + grad_alpha;
      if (alpha_accum > 1.0e-8f) {
        d_loss_d_weight += grad_depth *
            (mids[segment_id] * alpha_accum - depth_weighted) /
            (alpha_accum * alpha_accum);
      }

      const float adj_trans_before =
          d_loss_d_weight * segment_alpha[segment_id] +
          adj_next_transmittance * segment_trans[segment_id];
      const float adj_segment_alpha = d_loss_d_weight * trans_before[segment_id];
      const float adj_segment_trans =
          adj_next_transmittance * trans_before[segment_id] - adj_segment_alpha;
      if (owner == site_id) {
        grad_accum.x += weights[segment_id] * grad_rgb.x;
        grad_accum.y += weights[segment_id] * grad_rgb.y;
        grad_accum.z += weights[segment_id] * grad_rgb.z;
        const uint rgba_base = owner * 4u;
        const float raw_density = site_rgba_f32[rgba_base + 3u];
        if (raw_density > 0.0f) {
          grad_accum.w += adj_segment_trans * (-lengths[segment_id] * segment_trans[segment_id]);
        }
      }
      adj_next_transmittance = adj_trans_before;
    }
  }

  const uint site_grad_base = site_id * 4u;
  grad_site_rgba_f32[site_grad_base + 0u] = grad_accum.x;
  grad_site_rgba_f32[site_grad_base + 1u] = grad_accum.y;
  grad_site_rgba_f32[site_grad_base + 2u] = grad_accum.z;
  grad_site_rgba_f32[site_grad_base + 3u] = grad_accum.w;
}

kernel void wf2_shared_realray_rgba_depth_vjp_partial_reduce_tensor(
    device const float* boundary_f32 [[buffer(0)]],
    device const uint* candidate_mask_u32 [[buffer(1)]],
    device const float* sites_f32 [[buffer(2)]],
    device const float* site_rgba_f32 [[buffer(3)]],
    device const float* track_rays_f32 [[buffer(4)]],
    device const float* frame_t_f32 [[buffer(5)]],
    device const float* grad_rgb_f32 [[buffer(6)]],
    device const float* grad_alpha_f32 [[buffer(7)]],
    device const float* grad_depth_f32 [[buffer(8)]],
    device const int* config_i32 [[buffer(9)]],
    device const float* config_f32 [[buffer(10)]],
    device float* partial_grad_site_rgba_f32 [[buffer(11)]],
    uint chunk_id [[thread_position_in_grid]]) {
  const uint boundary_count = uint(config_i32[0]);
  const uint track_count = uint(config_i32[1]);
  const uint site_count = uint(config_i32[2]);
  const uint frame_count = uint(config_i32[3]);
  const uint time_slab_count = uint(config_i32[4]);
  const uint mask_word_count = uint(config_i32[5]);
  const uint clamped_site_count = min(site_count, WF2_MAX_REALRAY_SITES);
  constexpr uint reduce_chunk_size = 4u;
  const uint total_samples = track_count * frame_count;
  const uint sample_begin = chunk_id * reduce_chunk_size;
  if (sample_begin >= total_samples) {
    return;
  }
  const uint sample_end = min(sample_begin + reduce_chunk_size, total_samples);

  const float near_depth = config_f32[0];
  const float far_depth = config_f32[1];
  const float invalid_epsilon = config_f32[2];
  const float transmittance_threshold = config_f32[3];
  float4 grad_accum[WF2_MAX_REALRAY_SITES];
  for (uint site_id = 0u; site_id < clamped_site_count; ++site_id) {
    grad_accum[site_id] = float4(0.0f, 0.0f, 0.0f, 0.0f);
  }

  for (uint sample_id = sample_begin; sample_id < sample_end; ++sample_id) {
    const uint track_id = sample_id / frame_count;
    const uint frame_id = sample_id - track_id * frame_count;
    const uint ray_base = track_id * 6u;
    const float ox = track_rays_f32[ray_base + 0u];
    const float oy = track_rays_f32[ray_base + 1u];
    const float oz = track_rays_f32[ray_base + 2u];
    const float dx = track_rays_f32[ray_base + 3u];
    const float dy = track_rays_f32[ray_base + 4u];
    const float dz = track_rays_f32[ray_base + 5u];
    const float t = frame_t_f32[frame_id];
    const uint slab_id = wf2_replay_slab_id(t, time_slab_count);
    const uint mask_base = (track_id * time_slab_count + slab_id) * mask_word_count;

    float depths[WF2_MAX_REALRAY_BOUNDARIES];
    uint depth_count = 0u;
    const uint clamped_boundary_count = min(boundary_count, WF2_MAX_REALRAY_BOUNDARIES);
    for (uint boundary_id = 0u; boundary_id < clamped_boundary_count; ++boundary_id) {
      if (!wf2_bitset_has_boundary(candidate_mask_u32, mask_base, boundary_id)) {
        continue;
      }
      const uint base = boundary_id * 5u;
      const float nx = boundary_f32[base + 0u];
      const float ny = boundary_f32[base + 1u];
      const float nz = boundary_f32[base + 2u];
      const float nt = boundary_f32[base + 3u];
      const float b = boundary_f32[base + 4u];
      const float denom = nx * dx + ny * dy + nz * dz;
      if (fabs(denom) < invalid_epsilon) {
        continue;
      }
      const float depth = -(nx * ox + ny * oy + nz * oz + nt * t + b) / denom;
      if (isfinite(depth) && depth >= near_depth && depth <= far_depth) {
        wf2_realray_insert_depth(depths, depth_count, depth);
      }
    }

    uint owners[WF2_MAX_REALRAY_SEGMENTS];
    float lengths[WF2_MAX_REALRAY_SEGMENTS];
    float mids[WF2_MAX_REALRAY_SEGMENTS];
    float trans_before[WF2_MAX_REALRAY_SEGMENTS];
    float segment_trans[WF2_MAX_REALRAY_SEGMENTS];
    float segment_alpha[WF2_MAX_REALRAY_SEGMENTS];
    float weights[WF2_MAX_REALRAY_SEGMENTS];
    float3 segment_rgb[WF2_MAX_REALRAY_SEGMENTS];

    float alpha_accum = 0.0f;
    float depth_weighted = 0.0f;
    float transmittance = 1.0f;
    float previous_depth = near_depth;
    uint segment_count = 0u;
    for (uint cut_id = 0u; cut_id <= depth_count; ++cut_id) {
      const float next_depth = cut_id < depth_count ? depths[cut_id] : far_depth;
      const float length = next_depth - previous_depth;
      if (length > 1.0e-8f && transmittance > transmittance_threshold && segment_count < WF2_MAX_REALRAY_SEGMENTS) {
        const float mid_depth = 0.5f * (previous_depth + next_depth);
        const float x = ox + dx * mid_depth;
        const float y = oy + dy * mid_depth;
        const float z = oz + dz * mid_depth;
        const uint owner = wf2_realray_owner_at(sites_f32, clamped_site_count, x, y, z, t);
        const uint rgba_base = owner * 4u;
        const float density = max(site_rgba_f32[rgba_base + 3u], 0.0f);
        const float seg_trans = exp(-density * length);
        const float seg_alpha = 1.0f - seg_trans;
        const float weight = transmittance * seg_alpha;
        const float3 rgb = float3(
            site_rgba_f32[rgba_base + 0u],
            site_rgba_f32[rgba_base + 1u],
            site_rgba_f32[rgba_base + 2u]);

        owners[segment_count] = owner;
        lengths[segment_count] = length;
        mids[segment_count] = mid_depth;
        trans_before[segment_count] = transmittance;
        segment_trans[segment_count] = seg_trans;
        segment_alpha[segment_count] = seg_alpha;
        weights[segment_count] = weight;
        segment_rgb[segment_count] = rgb;
        segment_count += 1u;

        alpha_accum += weight;
        depth_weighted += weight * mid_depth;
        transmittance *= seg_trans;
      }
      previous_depth = next_depth;
    }

    const uint out_base = sample_id * 3u;
    const float3 grad_rgb = float3(
        grad_rgb_f32[out_base + 0u],
        grad_rgb_f32[out_base + 1u],
        grad_rgb_f32[out_base + 2u]);
    const float grad_alpha = grad_alpha_f32[sample_id];
    const float grad_depth = grad_depth_f32[sample_id];
    float adj_next_transmittance = 0.0f;
    for (int segment_id = int(segment_count) - 1; segment_id >= 0; --segment_id) {
      const uint owner = owners[segment_id];
      float d_loss_d_weight = dot(grad_rgb, segment_rgb[segment_id]) + grad_alpha;
      if (alpha_accum > 1.0e-8f) {
        d_loss_d_weight += grad_depth *
            (mids[segment_id] * alpha_accum - depth_weighted) /
            (alpha_accum * alpha_accum);
      }

      const float adj_trans_before =
          d_loss_d_weight * segment_alpha[segment_id] +
          adj_next_transmittance * segment_trans[segment_id];
      const float adj_segment_alpha = d_loss_d_weight * trans_before[segment_id];
      const float adj_segment_trans =
          adj_next_transmittance * trans_before[segment_id] - adj_segment_alpha;
      grad_accum[owner].x += weights[segment_id] * grad_rgb.x;
      grad_accum[owner].y += weights[segment_id] * grad_rgb.y;
      grad_accum[owner].z += weights[segment_id] * grad_rgb.z;
      const uint rgba_base = owner * 4u;
      const float raw_density = site_rgba_f32[rgba_base + 3u];
      if (raw_density > 0.0f) {
        grad_accum[owner].w += adj_segment_trans * (-lengths[segment_id] * segment_trans[segment_id]);
      }
      adj_next_transmittance = adj_trans_before;
    }
  }

  for (uint site_id = 0u; site_id < clamped_site_count; ++site_id) {
    const uint site_grad_base = (chunk_id * site_count + site_id) * 4u;
    partial_grad_site_rgba_f32[site_grad_base + 0u] = grad_accum[site_id].x;
    partial_grad_site_rgba_f32[site_grad_base + 1u] = grad_accum[site_id].y;
    partial_grad_site_rgba_f32[site_grad_base + 2u] = grad_accum[site_id].z;
    partial_grad_site_rgba_f32[site_grad_base + 3u] = grad_accum[site_id].w;
  }
}

kernel void wf2_shared_realray_rgba_depth_vjp_partial_reduce_csr_tensor(
    device const float* boundary_f32 [[buffer(0)]],
    device const int* row_index_i32 [[buffer(1)]],
    device const int* candidate_row_offsets_i32 [[buffer(2)]],
    device const int* candidate_boundary_ids_i32 [[buffer(3)]],
    device const float* sites_f32 [[buffer(4)]],
    device const float* site_rgba_f32 [[buffer(5)]],
    device const float* track_rays_f32 [[buffer(6)]],
    device const float* frame_t_f32 [[buffer(7)]],
    device const float* grad_rgb_f32 [[buffer(8)]],
    device const float* grad_alpha_f32 [[buffer(9)]],
    device const float* grad_depth_f32 [[buffer(10)]],
    device const int* config_i32 [[buffer(11)]],
    device const float* config_f32 [[buffer(12)]],
    device float* output_rgb_f32 [[buffer(13)]],
    device float* output_alpha_f32 [[buffer(14)]],
    device float* output_depth_f32 [[buffer(15)]],
    device float* partial_grad_site_rgba_f32 [[buffer(16)]],
    uint chunk_id [[thread_position_in_grid]]) {
  const uint boundary_count = uint(config_i32[0]);
  const uint track_count = uint(config_i32[1]);
  const uint site_count = uint(config_i32[2]);
  const uint frame_count = uint(config_i32[3]);
  const uint time_slab_count = uint(config_i32[4]);
  const uint row_count = uint(config_i32[5]);
  const uint candidate_count = uint(config_i32[6]);
  const uint clamped_site_count = min(site_count, WF2_MAX_REALRAY_SITES);
  constexpr uint reduce_chunk_size = 4u;
  const uint total_samples = track_count * frame_count;
  const uint sample_begin = chunk_id * reduce_chunk_size;
  if (sample_begin >= total_samples) {
    return;
  }
  const uint sample_end = min(sample_begin + reduce_chunk_size, total_samples);

  const float near_depth = config_f32[0];
  const float far_depth = config_f32[1];
  const float invalid_epsilon = config_f32[2];
  const float transmittance_threshold = config_f32[3];
  float4 grad_accum[WF2_MAX_REALRAY_SITES];
  for (uint site_id = 0u; site_id < clamped_site_count; ++site_id) {
    grad_accum[site_id] = float4(0.0f, 0.0f, 0.0f, 0.0f);
  }

  for (uint sample_id = sample_begin; sample_id < sample_end; ++sample_id) {
    const uint track_id = sample_id / frame_count;
    const uint frame_id = sample_id - track_id * frame_count;
    const uint ray_base = track_id * 6u;
    const float ox = track_rays_f32[ray_base + 0u];
    const float oy = track_rays_f32[ray_base + 1u];
    const float oz = track_rays_f32[ray_base + 2u];
    const float dx = track_rays_f32[ray_base + 3u];
    const float dy = track_rays_f32[ray_base + 4u];
    const float dz = track_rays_f32[ray_base + 5u];
    const float t = frame_t_f32[frame_id];
    const uint slab_id = wf2_replay_slab_id(t, time_slab_count);
    const int row_index_raw = row_index_i32[track_id];
    if (row_index_raw < 0 || uint(row_index_raw) >= row_count) {
      continue;
    }
    const uint row = uint(row_index_raw) * time_slab_count + slab_id;
    const int begin_raw = candidate_row_offsets_i32[row];
    const int end_raw = candidate_row_offsets_i32[row + 1u];
    if (begin_raw < 0 || end_raw < begin_raw || uint(end_raw) > candidate_count) {
      continue;
    }

    float depths[WF2_MAX_REALRAY_BOUNDARIES];
    uint depth_count = 0u;
    for (uint cursor = uint(begin_raw); cursor < uint(end_raw); ++cursor) {
      const int boundary_raw = candidate_boundary_ids_i32[cursor];
      if (boundary_raw < 0 || uint(boundary_raw) >= boundary_count) {
        continue;
      }
      const uint boundary_id = uint(boundary_raw);
      const uint base = boundary_id * 5u;
      const float nx = boundary_f32[base + 0u];
      const float ny = boundary_f32[base + 1u];
      const float nz = boundary_f32[base + 2u];
      const float nt = boundary_f32[base + 3u];
      const float b = boundary_f32[base + 4u];
      const float denom = nx * dx + ny * dy + nz * dz;
      if (fabs(denom) < invalid_epsilon) {
        continue;
      }
      const float depth = -(nx * ox + ny * oy + nz * oz + nt * t + b) / denom;
      if (isfinite(depth) && depth >= near_depth && depth <= far_depth) {
        wf2_realray_insert_depth(depths, depth_count, depth);
      }
    }

    uint owners[WF2_MAX_REALRAY_SEGMENTS];
    float lengths[WF2_MAX_REALRAY_SEGMENTS];
    float mids[WF2_MAX_REALRAY_SEGMENTS];
    float trans_before[WF2_MAX_REALRAY_SEGMENTS];
    float segment_trans[WF2_MAX_REALRAY_SEGMENTS];
    float segment_alpha[WF2_MAX_REALRAY_SEGMENTS];
    float weights[WF2_MAX_REALRAY_SEGMENTS];
    float3 segment_rgb[WF2_MAX_REALRAY_SEGMENTS];

    float3 rgb_accum = float3(0.0f, 0.0f, 0.0f);
    float alpha_accum = 0.0f;
    float depth_weighted = 0.0f;
    float transmittance = 1.0f;
    float previous_depth = near_depth;
    uint segment_count = 0u;
    for (uint cut_id = 0u; cut_id <= depth_count; ++cut_id) {
      const float next_depth = cut_id < depth_count ? depths[cut_id] : far_depth;
      const float length = next_depth - previous_depth;
      if (length > 1.0e-8f && transmittance > transmittance_threshold && segment_count < WF2_MAX_REALRAY_SEGMENTS) {
        const float mid_depth = 0.5f * (previous_depth + next_depth);
        const float x = ox + dx * mid_depth;
        const float y = oy + dy * mid_depth;
        const float z = oz + dz * mid_depth;
        const uint owner = wf2_realray_owner_at(sites_f32, clamped_site_count, x, y, z, t);
        const uint rgba_base = owner * 4u;
        const float density = max(site_rgba_f32[rgba_base + 3u], 0.0f);
        const float seg_trans = exp(-density * length);
        const float seg_alpha = 1.0f - seg_trans;
        const float weight = transmittance * seg_alpha;
        const float3 rgb = float3(
            site_rgba_f32[rgba_base + 0u],
            site_rgba_f32[rgba_base + 1u],
            site_rgba_f32[rgba_base + 2u]);

        owners[segment_count] = owner;
        lengths[segment_count] = length;
        mids[segment_count] = mid_depth;
        trans_before[segment_count] = transmittance;
        segment_trans[segment_count] = seg_trans;
        segment_alpha[segment_count] = seg_alpha;
        weights[segment_count] = weight;
        segment_rgb[segment_count] = rgb;
        segment_count += 1u;

        rgb_accum += weight * rgb;
        alpha_accum += weight;
        depth_weighted += weight * mid_depth;
        transmittance *= seg_trans;
      }
      previous_depth = next_depth;
    }

    const uint out_base = sample_id * 3u;
    output_rgb_f32[out_base + 0u] = rgb_accum.x;
    output_rgb_f32[out_base + 1u] = rgb_accum.y;
    output_rgb_f32[out_base + 2u] = rgb_accum.z;
    output_alpha_f32[sample_id] = alpha_accum;
    output_depth_f32[sample_id] = alpha_accum > 1.0e-8f ? depth_weighted / alpha_accum : far_depth;

    const float3 grad_rgb = float3(
        grad_rgb_f32[out_base + 0u],
        grad_rgb_f32[out_base + 1u],
        grad_rgb_f32[out_base + 2u]);
    const float grad_alpha = grad_alpha_f32[sample_id];
    const float grad_depth = grad_depth_f32[sample_id];
    float adj_next_transmittance = 0.0f;
    for (int segment_id = int(segment_count) - 1; segment_id >= 0; --segment_id) {
      const uint owner = owners[segment_id];
      float d_loss_d_weight = dot(grad_rgb, segment_rgb[segment_id]) + grad_alpha;
      if (alpha_accum > 1.0e-8f) {
        d_loss_d_weight += grad_depth *
            (mids[segment_id] * alpha_accum - depth_weighted) /
            (alpha_accum * alpha_accum);
      }

      const float adj_trans_before =
          d_loss_d_weight * segment_alpha[segment_id] +
          adj_next_transmittance * segment_trans[segment_id];
      const float adj_segment_alpha = d_loss_d_weight * trans_before[segment_id];
      const float adj_segment_trans =
          adj_next_transmittance * trans_before[segment_id] - adj_segment_alpha;
      grad_accum[owner].x += weights[segment_id] * grad_rgb.x;
      grad_accum[owner].y += weights[segment_id] * grad_rgb.y;
      grad_accum[owner].z += weights[segment_id] * grad_rgb.z;
      const uint rgba_base = owner * 4u;
      const float raw_density = site_rgba_f32[rgba_base + 3u];
      if (raw_density > 0.0f) {
        grad_accum[owner].w += adj_segment_trans * (-lengths[segment_id] * segment_trans[segment_id]);
      }
      adj_next_transmittance = adj_trans_before;
    }
  }

  for (uint site_id = 0u; site_id < clamped_site_count; ++site_id) {
    const uint site_grad_base = (chunk_id * site_count + site_id) * 4u;
    partial_grad_site_rgba_f32[site_grad_base + 0u] = grad_accum[site_id].x;
    partial_grad_site_rgba_f32[site_grad_base + 1u] = grad_accum[site_id].y;
    partial_grad_site_rgba_f32[site_grad_base + 2u] = grad_accum[site_id].z;
    partial_grad_site_rgba_f32[site_grad_base + 3u] = grad_accum[site_id].w;
  }
}

kernel void wf2_fused_slab_affine_num32_den16_vjp_partial_reduce_tensor(
    device const int* row_index_i32 [[buffer(0)]],
    device const int* candidate_row_offsets_i32 [[buffer(1)]],
    device const float* candidate_depth_num_f32 [[buffer(2)]],
    device const half* candidate_depth_den_f16 [[buffer(3)]],
    device const float* sites_f32 [[buffer(4)]],
    device const float* site_rgba_f32 [[buffer(5)]],
    device const float* ray_coeff_f32 [[buffer(6)]],
    device const float* frame_t_f32 [[buffer(7)]],
    device const float* grad_rgb_f32 [[buffer(8)]],
    device const float* grad_alpha_f32 [[buffer(9)]],
    device const float* grad_depth_f32 [[buffer(10)]],
    device const int* config_i32 [[buffer(11)]],
    device const float* config_f32 [[buffer(12)]],
    device float* output_rgb_f32 [[buffer(13)]],
    device float* output_alpha_f32 [[buffer(14)]],
    device float* output_depth_f32 [[buffer(15)]],
    device float* partial_grad_site_rgba_f32 [[buffer(16)]],
    uint chunk_id [[thread_position_in_grid]]) {
  const uint track_count = uint(config_i32[1]);
  const uint site_count = uint(config_i32[2]);
  const uint frame_count = uint(config_i32[3]);
  const uint time_slab_count = uint(config_i32[4]);
  const uint row_count = uint(config_i32[5]);
  const uint candidate_count = uint(config_i32[6]);
  const uint reduce_chunk_size = max(uint(config_i32[7]), 1u);
  const uint clamped_site_count = min(site_count, WF2_MAX_REALRAY_SITES);
  const uint total_samples = track_count * frame_count;
  const uint sample_begin = chunk_id * reduce_chunk_size;
  if (sample_begin >= total_samples) {
    return;
  }
  const uint sample_end = min(sample_begin + reduce_chunk_size, total_samples);

  const float near_depth = config_f32[0];
  const float far_depth = config_f32[1];
  const float invalid_epsilon = config_f32[2];
  const float transmittance_threshold = config_f32[3];
  float4 grad_accum[WF2_MAX_REALRAY_SITES];
  for (uint site_id = 0u; site_id < clamped_site_count; ++site_id) {
    grad_accum[site_id] = float4(0.0f, 0.0f, 0.0f, 0.0f);
  }

  for (uint sample_id = sample_begin; sample_id < sample_end; ++sample_id) {
    const uint track_id = sample_id / frame_count;
    const uint frame_id = sample_id - track_id * frame_count;
    const float t = frame_t_f32[frame_id];
    const float3 origin = wf2_affine_origin_at(ray_coeff_f32, track_id, t);
    const float3 direction = wf2_affine_direction_at(ray_coeff_f32, track_id, t);
    const uint slab_id = wf2_replay_slab_id(t, time_slab_count);
    const int row_index_raw = row_index_i32[track_id];
    if (row_index_raw < 0 || uint(row_index_raw) >= row_count) {
      continue;
    }
    const uint row = uint(row_index_raw) * time_slab_count + slab_id;
    const int begin_raw = candidate_row_offsets_i32[row];
    const int end_raw = candidate_row_offsets_i32[row + 1u];
    if (begin_raw < 0 || end_raw < begin_raw || uint(end_raw) > candidate_count) {
      continue;
    }

    float depths[WF2_MAX_REALRAY_BOUNDARIES];
    uint depth_count = 0u;
    for (uint cursor = uint(begin_raw); cursor < uint(end_raw); ++cursor) {
      const uint coeff_base = cursor * 2u;
      const float numer = candidate_depth_num_f32[coeff_base + 0u] + candidate_depth_num_f32[coeff_base + 1u] * t;
      const float denom = float(candidate_depth_den_f16[coeff_base + 0u]) +
          float(candidate_depth_den_f16[coeff_base + 1u]) * t;
      if (fabs(denom) < invalid_epsilon) {
        continue;
      }
      const float depth = numer / denom;
      if (isfinite(depth) && depth >= near_depth && depth <= far_depth) {
        wf2_realray_insert_depth(depths, depth_count, depth);
      }
    }

    uint owners[WF2_MAX_REALRAY_SEGMENTS];
    float lengths[WF2_MAX_REALRAY_SEGMENTS];
    float mids[WF2_MAX_REALRAY_SEGMENTS];
    float trans_before[WF2_MAX_REALRAY_SEGMENTS];
    float segment_trans[WF2_MAX_REALRAY_SEGMENTS];
    float segment_alpha[WF2_MAX_REALRAY_SEGMENTS];
    float weights[WF2_MAX_REALRAY_SEGMENTS];
    float3 segment_rgb[WF2_MAX_REALRAY_SEGMENTS];

    float3 rgb_accum = float3(0.0f, 0.0f, 0.0f);
    float alpha_accum = 0.0f;
    float depth_weighted = 0.0f;
    float transmittance = 1.0f;
    float previous_depth = near_depth;
    uint segment_count = 0u;
    for (uint cut_id = 0u; cut_id <= depth_count; ++cut_id) {
      const float next_depth = cut_id < depth_count ? depths[cut_id] : far_depth;
      const float length = next_depth - previous_depth;
      if (length > 1.0e-8f && transmittance > transmittance_threshold && segment_count < WF2_MAX_REALRAY_SEGMENTS) {
        const float mid_depth = 0.5f * (previous_depth + next_depth);
        const float x = origin.x + direction.x * mid_depth;
        const float y = origin.y + direction.y * mid_depth;
        const float z = origin.z + direction.z * mid_depth;
        const uint owner = wf2_realray_owner_at(sites_f32, clamped_site_count, x, y, z, t);
        const uint rgba_base = owner * 4u;
        const float density = max(site_rgba_f32[rgba_base + 3u], 0.0f);
        const float seg_trans = exp(-density * length);
        const float seg_alpha = 1.0f - seg_trans;
        const float weight = transmittance * seg_alpha;
        const float3 rgb = float3(
            site_rgba_f32[rgba_base + 0u],
            site_rgba_f32[rgba_base + 1u],
            site_rgba_f32[rgba_base + 2u]);

        owners[segment_count] = owner;
        lengths[segment_count] = length;
        mids[segment_count] = mid_depth;
        trans_before[segment_count] = transmittance;
        segment_trans[segment_count] = seg_trans;
        segment_alpha[segment_count] = seg_alpha;
        weights[segment_count] = weight;
        segment_rgb[segment_count] = rgb;
        segment_count += 1u;

        rgb_accum += weight * rgb;
        alpha_accum += weight;
        depth_weighted += weight * mid_depth;
        transmittance *= seg_trans;
      }
      previous_depth = next_depth;
    }

    const uint out_base = sample_id * 3u;
    output_rgb_f32[out_base + 0u] = rgb_accum.x;
    output_rgb_f32[out_base + 1u] = rgb_accum.y;
    output_rgb_f32[out_base + 2u] = rgb_accum.z;
    output_alpha_f32[sample_id] = alpha_accum;
    output_depth_f32[sample_id] = alpha_accum > 1.0e-8f ? depth_weighted / alpha_accum : far_depth;

    const float3 grad_rgb = float3(
        grad_rgb_f32[out_base + 0u],
        grad_rgb_f32[out_base + 1u],
        grad_rgb_f32[out_base + 2u]);
    const float grad_alpha = grad_alpha_f32[sample_id];
    const float grad_depth = grad_depth_f32[sample_id];
    float adj_next_transmittance = 0.0f;
    for (int segment_id = int(segment_count) - 1; segment_id >= 0; --segment_id) {
      const uint owner = owners[segment_id];
      float d_loss_d_weight = dot(grad_rgb, segment_rgb[segment_id]) + grad_alpha;
      if (alpha_accum > 1.0e-8f) {
        d_loss_d_weight += grad_depth *
            (mids[segment_id] * alpha_accum - depth_weighted) /
            (alpha_accum * alpha_accum);
      }

      const float adj_trans_before =
          d_loss_d_weight * segment_alpha[segment_id] +
          adj_next_transmittance * segment_trans[segment_id];
      const float adj_segment_alpha = d_loss_d_weight * trans_before[segment_id];
      const float adj_segment_trans =
          adj_next_transmittance * trans_before[segment_id] - adj_segment_alpha;
      grad_accum[owner].x += weights[segment_id] * grad_rgb.x;
      grad_accum[owner].y += weights[segment_id] * grad_rgb.y;
      grad_accum[owner].z += weights[segment_id] * grad_rgb.z;
      const uint rgba_base = owner * 4u;
      const float raw_density = site_rgba_f32[rgba_base + 3u];
      if (raw_density > 0.0f) {
        grad_accum[owner].w += adj_segment_trans * (-lengths[segment_id] * segment_trans[segment_id]);
      }
      adj_next_transmittance = adj_trans_before;
    }
  }

  for (uint site_id = 0u; site_id < clamped_site_count; ++site_id) {
    const uint site_grad_base = (chunk_id * site_count + site_id) * 4u;
    partial_grad_site_rgba_f32[site_grad_base + 0u] = grad_accum[site_id].x;
    partial_grad_site_rgba_f32[site_grad_base + 1u] = grad_accum[site_id].y;
    partial_grad_site_rgba_f32[site_grad_base + 2u] = grad_accum[site_id].z;
    partial_grad_site_rgba_f32[site_grad_base + 3u] = grad_accum[site_id].w;
  }
}

kernel void wf2_clear_site_rgba_grad_tensor(
    device float* grad_site_rgba_f32 [[buffer(0)]],
    device const int* config_i32 [[buffer(1)]],
    uint site_id [[thread_position_in_grid]]) {
  const uint site_count = uint(config_i32[2]);
  if (site_id >= site_count) {
    return;
  }
  const uint base = site_id * 4u;
  grad_site_rgba_f32[base + 0u] = 0.0f;
  grad_site_rgba_f32[base + 1u] = 0.0f;
  grad_site_rgba_f32[base + 2u] = 0.0f;
  grad_site_rgba_f32[base + 3u] = 0.0f;
}

kernel void wf2_clear_endpoint_loss_site_rgba_grad_tensor(
    device float* loss_f32 [[buffer(0)]],
    device float* grad_site_rgba_f32 [[buffer(1)]],
    device const int* config_i32 [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
  const uint site_count = uint(config_i32[3]);
  if (gid == 0u) {
    loss_f32[0] = 0.0f;
  }
  if (gid >= site_count) {
    return;
  }
  const uint base = gid * 4u;
  grad_site_rgba_f32[base + 0u] = 0.0f;
  grad_site_rgba_f32[base + 1u] = 0.0f;
  grad_site_rgba_f32[base + 2u] = 0.0f;
  grad_site_rgba_f32[base + 3u] = 0.0f;
}

kernel void wf2_clear_endpoint_loss_site_rgba_boundary_grad_tensor(
    device float* loss_f32 [[buffer(0)]],
    device float* grad_site_rgba_f32 [[buffer(1)]],
    device float* grad_boundary_f32 [[buffer(2)]],
    device const int* config_i32 [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
  const uint boundary_count = uint(config_i32[0]);
  const uint site_count = uint(config_i32[3]);
  if (gid == 0u) {
    loss_f32[0] = 0.0f;
  }
  if (gid < site_count) {
    const uint site_base = gid * 4u;
    grad_site_rgba_f32[site_base + 0u] = 0.0f;
    grad_site_rgba_f32[site_base + 1u] = 0.0f;
    grad_site_rgba_f32[site_base + 2u] = 0.0f;
    grad_site_rgba_f32[site_base + 3u] = 0.0f;
  }
  if (gid < boundary_count) {
    const uint boundary_base = gid * 5u;
    grad_boundary_f32[boundary_base + 0u] = 0.0f;
    grad_boundary_f32[boundary_base + 1u] = 0.0f;
    grad_boundary_f32[boundary_base + 2u] = 0.0f;
    grad_boundary_f32[boundary_base + 3u] = 0.0f;
    grad_boundary_f32[boundary_base + 4u] = 0.0f;
  }
}

kernel void wf2_clear_endpoint_loss_site_rgba_mobius_boundary_grad_tensor(
    device float* loss_f32 [[buffer(0)]],
    device float* grad_site_rgba_f32 [[buffer(1)]],
    device float* grad_mobius_coeff_f32 [[buffer(2)]],
    device float* grad_boundary_f32 [[buffer(3)]],
    device const int* config_i32 [[buffer(4)]],
    uint gid [[thread_position_in_grid]]) {
  const uint boundary_count = uint(config_i32[0]);
  const uint site_count = uint(config_i32[3]);
  const uint incidence_count = uint(config_i32[7]);
  if (gid == 0u) {
    loss_f32[0] = 0.0f;
  }
  if (gid < site_count) {
    const uint site_base = gid * 4u;
    grad_site_rgba_f32[site_base + 0u] = 0.0f;
    grad_site_rgba_f32[site_base + 1u] = 0.0f;
    grad_site_rgba_f32[site_base + 2u] = 0.0f;
    grad_site_rgba_f32[site_base + 3u] = 0.0f;
  }
  if (gid < incidence_count) {
    const uint incidence_base = gid * 4u;
    grad_mobius_coeff_f32[incidence_base + 0u] = 0.0f;
    grad_mobius_coeff_f32[incidence_base + 1u] = 0.0f;
    grad_mobius_coeff_f32[incidence_base + 2u] = 0.0f;
    grad_mobius_coeff_f32[incidence_base + 3u] = 0.0f;
  }
  if (gid < boundary_count) {
    const uint boundary_base = gid * 5u;
    grad_boundary_f32[boundary_base + 0u] = 0.0f;
    grad_boundary_f32[boundary_base + 1u] = 0.0f;
    grad_boundary_f32[boundary_base + 2u] = 0.0f;
    grad_boundary_f32[boundary_base + 3u] = 0.0f;
    grad_boundary_f32[boundary_base + 4u] = 0.0f;
  }
}

kernel void wf2_clear_fixed_word_p0_compiled_lie_grad_tensor(
    device float* loss_f32 [[buffer(0)]],
    device float* grad_node_chart_f32 [[buffer(1)]],
    device float* grad_site_rgba_f32 [[buffer(2)]],
    device float* grad_mobius_coeff_f32 [[buffer(3)]],
    device float* grad_boundary_f32 [[buffer(4)]],
    device int* cone_diagnostic_i32 [[buffer(5)]],
    device const int* config_i32 [[buffer(6)]],
    uint gid [[thread_position_in_grid]]) {
  const uint boundary_count = uint(config_i32[0]);
  const uint track_count = uint(config_i32[1]);
  const uint node_count = uint(config_i32[2]);
  const uint site_count = uint(config_i32[4]);
  const uint incidence_count = uint(config_i32[6]);
  const uint node_element_count = track_count * node_count * 4u;
  if (gid == 0u) {
    loss_f32[0] = 0.0f;
    cone_diagnostic_i32[0] = 0;
    cone_diagnostic_i32[1] = 0;
    cone_diagnostic_i32[2] = 0;
  }
  if (gid < node_element_count) {
    grad_node_chart_f32[gid] = 0.0f;
  }
  if (gid < site_count) {
    const uint site_base = gid * 4u;
    grad_site_rgba_f32[site_base + 0u] = 0.0f;
    grad_site_rgba_f32[site_base + 1u] = 0.0f;
    grad_site_rgba_f32[site_base + 2u] = 0.0f;
    grad_site_rgba_f32[site_base + 3u] = 0.0f;
  }
  if (gid < incidence_count) {
    const uint incidence_base = gid * 4u;
    grad_mobius_coeff_f32[incidence_base + 0u] = 0.0f;
    grad_mobius_coeff_f32[incidence_base + 1u] = 0.0f;
    grad_mobius_coeff_f32[incidence_base + 2u] = 0.0f;
    grad_mobius_coeff_f32[incidence_base + 3u] = 0.0f;
  }
  if (gid < boundary_count) {
    const uint boundary_base = gid * 5u;
    grad_boundary_f32[boundary_base + 0u] = 0.0f;
    grad_boundary_f32[boundary_base + 1u] = 0.0f;
    grad_boundary_f32[boundary_base + 2u] = 0.0f;
    grad_boundary_f32[boundary_base + 3u] = 0.0f;
    grad_boundary_f32[boundary_base + 4u] = 0.0f;
  }
}

kernel void wf2_sparse_mobius_incidence_lower_tensor(
    device const float* boundary_f32 [[buffer(0)]],
    device const float* track_ray_coeff_f32 [[buffer(1)]],
    device const int* track_incidence_offsets_i32 [[buffer(2)]],
    device const int* incidence_boundary_i32 [[buffer(3)]],
    device float* mobius_coeff_f32 [[buffer(4)]],
    device const int* config_i32 [[buffer(5)]],
    uint track_id [[thread_position_in_grid]]) {
  const uint boundary_count = uint(config_i32[0]);
  const uint track_count = uint(config_i32[1]);
  const uint incidence_count = uint(config_i32[7]);
  if (track_id >= track_count) {
    return;
  }
  const int incidence_begin_raw = track_incidence_offsets_i32[track_id];
  const int incidence_end_raw = track_incidence_offsets_i32[track_id + 1u];
  if (incidence_begin_raw < 0 || incidence_end_raw < incidence_begin_raw) {
    return;
  }
  const uint incidence_begin = uint(incidence_begin_raw);
  const uint incidence_end = min(uint(incidence_end_raw), incidence_count);
  const uint track_base = track_id * 12u;
  const float3 origin_base = float3(
      track_ray_coeff_f32[track_base + 0u],
      track_ray_coeff_f32[track_base + 1u],
      track_ray_coeff_f32[track_base + 2u]);
  const float3 origin_slope = float3(
      track_ray_coeff_f32[track_base + 3u],
      track_ray_coeff_f32[track_base + 4u],
      track_ray_coeff_f32[track_base + 5u]);
  const float3 direction_base = float3(
      track_ray_coeff_f32[track_base + 6u],
      track_ray_coeff_f32[track_base + 7u],
      track_ray_coeff_f32[track_base + 8u]);
  const float3 direction_slope = float3(
      track_ray_coeff_f32[track_base + 9u],
      track_ray_coeff_f32[track_base + 10u],
      track_ray_coeff_f32[track_base + 11u]);
  for (uint incidence_id = incidence_begin; incidence_id < incidence_end; ++incidence_id) {
    const int boundary_id_raw = incidence_boundary_i32[incidence_id];
    if (boundary_id_raw < 0 || uint(boundary_id_raw) >= boundary_count) {
      continue;
    }
    const uint boundary_base = uint(boundary_id_raw) * 5u;
    const float3 normal = float3(
        boundary_f32[boundary_base + 0u],
        boundary_f32[boundary_base + 1u],
        boundary_f32[boundary_base + 2u]);
    const uint coefficient_base = incidence_id * 4u;
    mobius_coeff_f32[coefficient_base + 0u] =
        -dot(origin_base, normal) - boundary_f32[boundary_base + 4u];
    mobius_coeff_f32[coefficient_base + 1u] =
        -dot(origin_slope, normal) - boundary_f32[boundary_base + 3u];
    mobius_coeff_f32[coefficient_base + 2u] = dot(direction_base, normal);
    mobius_coeff_f32[coefficient_base + 3u] = dot(direction_slope, normal);
  }
}

kernel void wf2_sparse_mobius_incidence_boundary_vjp_tensor(
    device const float* track_ray_coeff_f32 [[buffer(0)]],
    device const int* track_incidence_offsets_i32 [[buffer(1)]],
    device const int* incidence_boundary_i32 [[buffer(2)]],
    device const float* grad_mobius_coeff_f32 [[buffer(3)]],
    device atomic_float* grad_boundary_f32 [[buffer(4)]],
    device const int* config_i32 [[buffer(5)]],
    uint track_id [[thread_position_in_grid]]) {
  const uint boundary_count = uint(config_i32[0]);
  const uint track_count = uint(config_i32[1]);
  const uint incidence_count = uint(config_i32[7]);
  if (track_id >= track_count) {
    return;
  }
  const int incidence_begin_raw = track_incidence_offsets_i32[track_id];
  const int incidence_end_raw = track_incidence_offsets_i32[track_id + 1u];
  if (incidence_begin_raw < 0 || incidence_end_raw < incidence_begin_raw) {
    return;
  }
  const uint incidence_begin = uint(incidence_begin_raw);
  const uint incidence_end = min(uint(incidence_end_raw), incidence_count);
  const uint track_base = track_id * 12u;
  const float3 origin_base = float3(
      track_ray_coeff_f32[track_base + 0u],
      track_ray_coeff_f32[track_base + 1u],
      track_ray_coeff_f32[track_base + 2u]);
  const float3 origin_slope = float3(
      track_ray_coeff_f32[track_base + 3u],
      track_ray_coeff_f32[track_base + 4u],
      track_ray_coeff_f32[track_base + 5u]);
  const float3 direction_base = float3(
      track_ray_coeff_f32[track_base + 6u],
      track_ray_coeff_f32[track_base + 7u],
      track_ray_coeff_f32[track_base + 8u]);
  const float3 direction_slope = float3(
      track_ray_coeff_f32[track_base + 9u],
      track_ray_coeff_f32[track_base + 10u],
      track_ray_coeff_f32[track_base + 11u]);
  for (uint incidence_id = incidence_begin; incidence_id < incidence_end; ++incidence_id) {
    const int boundary_id_raw = incidence_boundary_i32[incidence_id];
    if (boundary_id_raw < 0 || uint(boundary_id_raw) >= boundary_count) {
      continue;
    }
    const uint grad_base = incidence_id * 4u;
    const float grad_a = grad_mobius_coeff_f32[grad_base + 0u];
    const float grad_b = grad_mobius_coeff_f32[grad_base + 1u];
    const float grad_c = grad_mobius_coeff_f32[grad_base + 2u];
    const float grad_d = grad_mobius_coeff_f32[grad_base + 3u];
    const float3 grad_normal =
        -grad_a * origin_base - grad_b * origin_slope +
        grad_c * direction_base + grad_d * direction_slope;
    wf2_atomic_add5(
        grad_boundary_f32,
        uint(boundary_id_raw) * 5u,
        grad_normal,
        -grad_b,
        -grad_a);
  }
}

// Derive the compact active 4D power boundaries from the exact resident sites
// and pair table later used by the site VJP. This closes the provenance chain:
// forward cannot consume an independently supplied boundary tensor.
kernel void wf2_sparse_power_boundary_from_sites_launch_only_tensor(
    device const int* boundary_site_pairs_i32 [[buffer(0)]],
    device const float* sites_f32 [[buffer(1)]],
    device float* boundary_f32 [[buffer(2)]],
    uint boundary_id [[thread_position_in_grid]]) {
  const uint pair_base = boundary_id * 2u;
  const uint left = uint(boundary_site_pairs_i32[pair_base + 0u]);
  const uint right = uint(boundary_site_pairs_i32[pair_base + 1u]);
  const uint left_base = left * 5u;
  const uint right_base = right * 5u;
  const float4 left_q = float4(
      sites_f32[left_base + 0u],
      sites_f32[left_base + 1u],
      sites_f32[left_base + 2u],
      sites_f32[left_base + 3u]);
  const float4 right_q = float4(
      sites_f32[right_base + 0u],
      sites_f32[right_base + 1u],
      sites_f32[right_base + 2u],
      sites_f32[right_base + 3u]);
  const float left_weight = sites_f32[left_base + 4u];
  const float right_weight = sites_f32[right_base + 4u];
  const float4 normal = 2.0f * (right_q - left_q);
  const float bias = dot(left_q, left_q) - dot(right_q, right_q) -
      left_weight + right_weight;
  const uint boundary_base = boundary_id * 5u;
  boundary_f32[boundary_base + 0u] = normal.x;
  boundary_f32[boundary_base + 1u] = normal.y;
  boundary_f32[boundary_base + 2u] = normal.z;
  boundary_f32[boundary_base + 3u] = normal.w;
  boundary_f32[boundary_base + 4u] = bias;
}

// Pull a sparse set of active 4D power-boundary cotangents back to the
// generating sites. Each active row is [boundary_id, left_site, right_site]
// and must occur exactly once; repeated track incidences have already been
// reduced into grad_boundary_f32. Shared sites are accumulated atomically.
//
// For q=(x,y,z,t) and weight w, the left/right power bisector is
//   n = 2(q_right - q_left)
//   b = ||q_left||^2 - ||q_right||^2 - w_left + w_right.
// This is the same symmetric power-face geometry used by the retained-depth
// PowerFoam endpoint VJP, expressed after the sparse boundary reduction.
kernel void wf2_sparse_power_boundary_site_vjp_launch_only_tensor(
    device const int* active_boundary_site_pairs_i32 [[buffer(0)]],
    device const float* sites_f32 [[buffer(1)]],
    device const float* grad_boundary_f32 [[buffer(2)]],
    device atomic_float* grad_sites_f32 [[buffer(3)]],
    uint active_id [[thread_position_in_grid]]) {
  // Launch-only contract: dispatch width equals U and the compile-time recoder
  // has already range-checked and deduplicated every topology row.
  const uint pair_base = active_id * 3u;
  const uint boundary_id = uint(active_boundary_site_pairs_i32[pair_base + 0u]);
  const uint left = uint(active_boundary_site_pairs_i32[pair_base + 1u]);
  const uint right = uint(active_boundary_site_pairs_i32[pair_base + 2u]);
  const uint boundary_base = boundary_id * 5u;
  const float3 grad_normal = float3(
      grad_boundary_f32[boundary_base + 0u],
      grad_boundary_f32[boundary_base + 1u],
      grad_boundary_f32[boundary_base + 2u]);
  const float grad_normal_t = grad_boundary_f32[boundary_base + 3u];
  const float grad_bias = grad_boundary_f32[boundary_base + 4u];
  const uint left_base = left * 5u;
  const uint right_base = right * 5u;
  const float3 left_xyz = float3(
      sites_f32[left_base + 0u],
      sites_f32[left_base + 1u],
      sites_f32[left_base + 2u]);
  const float3 right_xyz = float3(
      sites_f32[right_base + 0u],
      sites_f32[right_base + 1u],
      sites_f32[right_base + 2u]);
  const float left_t = sites_f32[left_base + 3u];
  const float right_t = sites_f32[right_base + 3u];

  wf2_atomic_add5(
      grad_sites_f32,
      left_base,
      -2.0f * grad_normal + 2.0f * grad_bias * left_xyz,
      -2.0f * grad_normal_t + 2.0f * grad_bias * left_t,
      -grad_bias);
  wf2_atomic_add5(
      grad_sites_f32,
      right_base,
      2.0f * grad_normal - 2.0f * grad_bias * right_xyz,
      2.0f * grad_normal_t - 2.0f * grad_bias * right_t,
      grad_bias);
}

kernel void wf2_fixed_word_p0_lie_node_forward_tensor(
    device const float* mobius_coeff_f32 [[buffer(0)]],
    device const float* track_ray_coeff_f32 [[buffer(1)]],
    device const float* compiler_node_t_f32 [[buffer(2)]],
    device const int* word_offsets_i32 [[buffer(3)]],
    device const int* word_owner_i32 [[buffer(4)]],
    device const int* word_left_incidence_i32 [[buffer(5)]],
    device const int* word_right_incidence_i32 [[buffer(6)]],
    device const int* track_incidence_offsets_i32 [[buffer(7)]],
    device const float* site_rgba_f32 [[buffer(8)]],
    device float* node_chart_f32 [[buffer(9)]],
    device const int* config_i32 [[buffer(10)]],
    device const float* config_f32 [[buffer(11)]],
    uint gid [[thread_position_in_grid]]) {
  const uint track_count = uint(config_i32[1]);
  const uint node_count = uint(config_i32[2]);
  const uint site_count = uint(config_i32[4]);
  const uint incidence_count = uint(config_i32[6]);
  if (gid >= track_count * node_count) {
    return;
  }
  const uint track_id = gid / node_count;
  const uint node_id = gid - track_id * node_count;
  const float t = compiler_node_t_f32[node_id];
  const float near_depth = config_f32[0];
  const float far_depth = config_f32[1];
  const float invalid_epsilon = config_f32[2];
  const float physical_length_epsilon = config_f32[3];
  const uint track_base = track_id * 12u;
  const float3 ray_direction = float3(
      track_ray_coeff_f32[track_base + 6u] + t * track_ray_coeff_f32[track_base + 9u],
      track_ray_coeff_f32[track_base + 7u] + t * track_ray_coeff_f32[track_base + 10u],
      track_ray_coeff_f32[track_base + 8u] + t * track_ray_coeff_f32[track_base + 11u]);
  const float fiber_speed = length(ray_direction);
  const int word_begin_raw = word_offsets_i32[track_id];
  const int word_end_raw = word_offsets_i32[track_id + 1u];
  bool valid = isfinite(t) && isfinite(fiber_speed) && fiber_speed > 0.0f &&
      word_begin_raw >= 0 && word_end_raw > word_begin_raw;
  float total_kappa = 0.0f;
  float total_beta = 1.0f;
  float3 total_m = float3(0.0f, 0.0f, 0.0f);
  int cached_right_cut = -2147483648;
  float cached_right_depth = 0.0f;
  bool cached_right_valid = false;
  for (int cursor = word_begin_raw; valid && cursor < word_end_raw; ++cursor) {
    const int owner_raw = word_owner_i32[uint(cursor)];
    const int left_cut = word_left_incidence_i32[uint(cursor)];
    const int right_cut = word_right_incidence_i32[uint(cursor)];
    if (owner_raw < 0 || uint(owner_raw) >= site_count) {
      valid = false;
      break;
    }
    float start_depth = 0.0f;
    uint start_incidence_id = 0xFFFFFFFFu;
    float4 start_grad_mobius = float4(0.0f);
    bool start_valid = false;
    if (cached_right_valid && left_cut == cached_right_cut) {
      start_depth = cached_right_depth;
      start_valid = true;
    } else {
      start_valid = wf2_endpoint_record_sparse_mobius_cut_depth_jacobian(
          mobius_coeff_f32,
          track_incidence_offsets_i32,
          incidence_count,
          track_id,
          left_cut,
          t,
          near_depth,
          far_depth,
          invalid_epsilon,
          start_incidence_id,
          start_depth,
          start_grad_mobius);
    }
    float end_depth = 0.0f;
    uint end_incidence_id = 0xFFFFFFFFu;
    float4 end_grad_mobius = float4(0.0f);
    if (!start_valid ||
        !wf2_endpoint_record_sparse_mobius_cut_depth_jacobian(
            mobius_coeff_f32,
            track_incidence_offsets_i32,
            incidence_count,
            track_id,
            right_cut,
            t,
            near_depth,
            far_depth,
            invalid_epsilon,
            end_incidence_id,
            end_depth,
            end_grad_mobius)) {
      valid = false;
      break;
    }
    cached_right_cut = right_cut;
    cached_right_depth = end_depth;
    cached_right_valid = true;
    const float physical_length = fiber_speed * (end_depth - start_depth);
    const uint owner = uint(owner_raw);
    const uint rgba_base = owner * 4u;
    const float raw_density = site_rgba_f32[rgba_base + 3u];
    if (!isfinite(raw_density) || raw_density < 0.0f) {
      valid = false;
      break;
    }
    // Compiled WorldFoam consumes physical nonnegative density. At the
    // boundary density==0 the reverse uses the one-sided derivative from the
    // feasible (positive-density) side, matching the CPU adaptive-rank gate.
    const float density = raw_density;
    const float optical_depth = density * physical_length;
    if (!(physical_length > physical_length_epsilon) ||
        !isfinite(physical_length) || !isfinite(optical_depth)) {
      valid = false;
      break;
    }
    const float segment_beta = exp(-optical_depth);
    const float segment_alpha = -expm1(-optical_depth);
    const float3 rgb = float3(
        site_rgba_f32[rgba_base + 0u],
        site_rgba_f32[rgba_base + 1u],
        site_rgba_f32[rgba_base + 2u]);
    total_kappa += optical_depth;
    total_m += total_beta * segment_alpha * rgb;
    total_beta *= segment_beta;
  }

  const uint chart_base = gid * 4u;
  if (!valid || !isfinite(total_kappa) || !all(isfinite(total_m))) {
    const float invalid = wf2_quiet_nan();
    node_chart_f32[chart_base + 0u] = invalid;
    node_chart_f32[chart_base + 1u] = invalid;
    node_chart_f32[chart_base + 2u] = invalid;
    node_chart_f32[chart_base + 3u] = invalid;
    return;
  }
  float inverse_phi = 1.0f;
  float inverse_phi_prime = 0.0f;
  wf2_lie_inverse_phi_and_derivative(total_kappa, inverse_phi, inverse_phi_prime);
  const float3 velocity = inverse_phi * total_m;
  node_chart_f32[chart_base + 0u] = total_kappa;
  node_chart_f32[chart_base + 1u] = velocity.x;
  node_chart_f32[chart_base + 2u] = velocity.y;
  node_chart_f32[chart_base + 3u] = velocity.z;
}

// Kinetic geometry does not in general reduce to the static-site Mobius cut
// ABI above: affine site motion produces quadratic-over-quadratic ray cuts.
// The kinetic compiler therefore seals the owner word and physical segment
// lengths at the bounded chart nodes.  This kernel consumes that frame-free
// node program directly.  The layout is [node_count, word_count], so storage
// depends on compiled chart density rather than requested sample/frame count.
kernel void wf2_kinetic_precompiled_length_p0_lie_node_forward_tensor(
    device const int* word_offsets_i32 [[buffer(0)]],
    device const int* word_owner_i32 [[buffer(1)]],
    device const float* node_physical_length_f32 [[buffer(2)]],
    device const float* site_rgba_f32 [[buffer(3)]],
    device float* node_chart_f32 [[buffer(4)]],
    device const int* config_i32 [[buffer(5)]],
    device const float* config_f32 [[buffer(6)]],
    uint gid [[thread_position_in_grid]]) {
  const uint track_count = uint(config_i32[0]);
  const uint node_count = uint(config_i32[1]);
  const uint site_count = uint(config_i32[2]);
  const uint word_count = uint(config_i32[3]);
  if (gid >= track_count * node_count) {
    return;
  }
  const uint track_id = gid / node_count;
  const uint node_id = gid - track_id * node_count;
  const int word_begin_raw = word_offsets_i32[track_id];
  const int word_end_raw = word_offsets_i32[track_id + 1u];
  const float physical_length_epsilon = config_f32[0];
  bool valid = word_begin_raw >= 0 && word_end_raw > word_begin_raw &&
      uint(word_end_raw) <= word_count;
  float total_kappa = 0.0f;
  float total_beta = 1.0f;
  float3 total_m = float3(0.0f);
  const uint node_length_base = node_id * word_count;
  for (int cursor = word_begin_raw; valid && cursor < word_end_raw; ++cursor) {
    const int owner_raw = word_owner_i32[uint(cursor)];
    if (owner_raw < 0 || uint(owner_raw) >= site_count) {
      valid = false;
      break;
    }
    const float physical_length =
        node_physical_length_f32[node_length_base + uint(cursor)];
    const uint owner = uint(owner_raw);
    const uint rgba_base = owner * 4u;
    const float raw_density = site_rgba_f32[rgba_base + 3u];
    if (!(physical_length > physical_length_epsilon) ||
        !isfinite(physical_length) || !isfinite(raw_density) || raw_density < 0.0f) {
      valid = false;
      break;
    }
    const float density = raw_density;
    const float optical_depth = density * physical_length;
    if (!isfinite(optical_depth)) {
      valid = false;
      break;
    }
    const float segment_beta = exp(-optical_depth);
    const float segment_alpha = -expm1(-optical_depth);
    const float3 rgb = float3(
        site_rgba_f32[rgba_base + 0u],
        site_rgba_f32[rgba_base + 1u],
        site_rgba_f32[rgba_base + 2u]);
    total_kappa += optical_depth;
    total_m += total_beta * segment_alpha * rgb;
    total_beta *= segment_beta;
  }

  const uint chart_base = gid * 4u;
  if (!valid || !isfinite(total_kappa) || !all(isfinite(total_m))) {
    const float invalid = wf2_quiet_nan();
    node_chart_f32[chart_base + 0u] = invalid;
    node_chart_f32[chart_base + 1u] = invalid;
    node_chart_f32[chart_base + 2u] = invalid;
    node_chart_f32[chart_base + 3u] = invalid;
    return;
  }
  float inverse_phi = 1.0f;
  float inverse_phi_prime = 0.0f;
  wf2_lie_inverse_phi_and_derivative(total_kappa, inverse_phi, inverse_phi_prime);
  const float3 velocity = inverse_phi * total_m;
  node_chart_f32[chart_base + 0u] = total_kappa;
  node_chart_f32[chart_base + 1u] = velocity.x;
  node_chart_f32[chart_base + 2u] = velocity.y;
  node_chart_f32[chart_base + 3u] = velocity.z;
}

// Shared sample reduction for both the diagnostic RGB-producing ABI and the
// training-only loss/VJP ABI.  Keeping the arithmetic in one helper prevents
// the memory-light path from drifting numerically while allowing it to avoid
// a Bp x K x 3 prediction allocation and write.
inline bool wf2_fixed_word_p0_lie_sample_mse_vjp(
    device const float* node_chart_f32,
    device const float* sample_to_node_f32,
    device const float* target_rgb_f32,
    device const float* background_rgb_f32,
    device atomic_float* loss_f32,
    device atomic_float* grad_node_chart_f32,
    device atomic_int* cone_diagnostic_i32,
    device const int* config_i32,
    device const float* config_f32,
    uint gid,
    thread float3& prediction) {
  const uint track_count = uint(config_i32[1]);
  const uint node_count = uint(config_i32[2]);
  const uint sample_count = uint(config_i32[3]);
  if (gid >= track_count * sample_count) {
    return false;
  }
  const uint track_id = gid / sample_count;
  const uint sample_id = gid - track_id * sample_count;
  float4 chart = float4(0.0f);
  const uint weight_base = sample_id * node_count;
  const uint node_base = track_id * node_count;
  for (uint node_id = 0u; node_id < node_count; ++node_id) {
    const float weight = sample_to_node_f32[weight_base + node_id];
    const uint chart_base = (node_base + node_id) * 4u;
    chart += weight * float4(
        node_chart_f32[chart_base + 0u],
        node_chart_f32[chart_base + 1u],
        node_chart_f32[chart_base + 2u],
        node_chart_f32[chart_base + 3u]);
  }
  const float kappa = chart.x;
  const float3 velocity = chart.yzw;
  const float cone_tolerance = config_f32[4];
  const float loss_scale = config_f32[5];
  const float cone_violation = max(
      max(-kappa, -min(velocity.x, min(velocity.y, velocity.z))),
      max(velocity.x, max(velocity.y, velocity.z)) - kappa);
  const bool chart_finite = all(isfinite(chart));
  bool valid = chart_finite && cone_violation <= cone_tolerance;
  float phi = 1.0f;
  float phi_prime = 0.0f;
  if (valid) {
    wf2_lie_phi_and_derivative(kappa, phi, phi_prime);
  }
  const float beta = valid ? exp(-kappa) : 0.0f;
  const float3 moment = phi * velocity;
  const float3 background = float3(
      background_rgb_f32[0],
      background_rgb_f32[1],
      background_rgb_f32[2]);
  const uint prediction_base = gid * 3u;
  const float3 target = float3(
      target_rgb_f32[prediction_base + 0u],
      target_rgb_f32[prediction_base + 1u],
      target_rgb_f32[prediction_base + 2u]);
  prediction = moment + beta * background;
  const bool prediction_finite = all(isfinite(prediction));
  valid = valid && isfinite(beta) && prediction_finite &&
      isfinite(loss_scale) && loss_scale > 0.0f &&
      all(isfinite(background)) && all(isfinite(target));
  if (!valid) {
    const float invalid = wf2_quiet_nan();
    prediction = float3(invalid);
    atomic_fetch_add_explicit(&cone_diagnostic_i32[0], 1, memory_order_relaxed);
    if (!chart_finite || !prediction_finite || !all(isfinite(target))) {
      atomic_fetch_add_explicit(&cone_diagnostic_i32[1], 1, memory_order_relaxed);
    }
    atomic_store_explicit(&cone_diagnostic_i32[2], int(gid + 1u), memory_order_relaxed);
    atomic_fetch_add_explicit(&loss_f32[0], invalid, memory_order_relaxed);
    return true;
  }
  const float3 diff = prediction - target;
  atomic_fetch_add_explicit(
      &loss_f32[0],
      dot(diff, diff) * loss_scale,
      memory_order_relaxed);
  const float3 grad_prediction = (2.0f * loss_scale) * diff;
  const float grad_beta = dot(grad_prediction, background);
  const float4 grad_chart = float4(
      -beta * grad_beta + phi_prime * dot(velocity, grad_prediction),
      phi * grad_prediction);
  for (uint node_id = 0u; node_id < node_count; ++node_id) {
    wf2_atomic_add4(
        grad_node_chart_f32,
        (node_base + node_id) * 4u,
        sample_to_node_f32[weight_base + node_id] * grad_chart);
  }
  return true;
}

kernel void wf2_fixed_word_p0_lie_sample_mse_vjp_tensor(
    device const float* node_chart_f32 [[buffer(0)]],
    device const float* sample_to_node_f32 [[buffer(1)]],
    device const float* target_rgb_f32 [[buffer(2)]],
    device const float* background_rgb_f32 [[buffer(3)]],
    device float* prediction_rgb_f32 [[buffer(4)]],
    device atomic_float* loss_f32 [[buffer(5)]],
    device atomic_float* grad_node_chart_f32 [[buffer(6)]],
    device atomic_int* cone_diagnostic_i32 [[buffer(7)]],
    device const int* config_i32 [[buffer(8)]],
    device const float* config_f32 [[buffer(9)]],
    uint gid [[thread_position_in_grid]]) {
  float3 prediction = float3(0.0f);
  if (!wf2_fixed_word_p0_lie_sample_mse_vjp(
          node_chart_f32,
          sample_to_node_f32,
          target_rgb_f32,
          background_rgb_f32,
          loss_f32,
          grad_node_chart_f32,
          cone_diagnostic_i32,
          config_i32,
          config_f32,
          gid,
          prediction)) {
    return;
  }
  const uint prediction_base = gid * 3u;
  prediction_rgb_f32[prediction_base + 0u] = prediction.x;
  prediction_rgb_f32[prediction_base + 1u] = prediction.y;
  prediction_rgb_f32[prediction_base + 2u] = prediction.z;
}

kernel void wf2_fixed_word_p0_lie_sample_mse_vjp_accumulate_only_tensor(
    device const float* node_chart_f32 [[buffer(0)]],
    device const float* sample_to_node_f32 [[buffer(1)]],
    device const float* target_rgb_f32 [[buffer(2)]],
    device const float* background_rgb_f32 [[buffer(3)]],
    device atomic_float* loss_f32 [[buffer(4)]],
    device atomic_float* grad_node_chart_f32 [[buffer(5)]],
    device atomic_int* cone_diagnostic_i32 [[buffer(6)]],
    device const int* config_i32 [[buffer(7)]],
    device const float* config_f32 [[buffer(8)]],
    uint gid [[thread_position_in_grid]]) {
  float3 discarded_prediction = float3(0.0f);
  wf2_fixed_word_p0_lie_sample_mse_vjp(
      node_chart_f32,
      sample_to_node_f32,
      target_rgb_f32,
      background_rgb_f32,
      loss_f32,
      grad_node_chart_f32,
      cone_diagnostic_i32,
      config_i32,
      config_f32,
      gid,
      discarded_prediction);
}

// Ragged kinetic charts have one compact Lie row per (track, chart), while
// the live sampler contributes only N selected observations.  Each sample
// names its row explicitly and carries only that row's J interpolation
// weights.  There is no row x sample Cartesian product and no common temporal
// refinement across rows.
inline bool wf2_kinetic_ragged_p0_lie_sample_mse_vjp(
    device const float* node_chart_f32,
    device const int* sample_row_i32,
    device const float* sample_to_node_f32,
    device const float* target_rgb_f32,
    device const float* background_rgb_f32,
    device atomic_float* loss_f32,
    device atomic_float* grad_node_chart_f32,
    device atomic_int* cone_diagnostic_i32,
    device const int* config_i32,
    device const float* config_f32,
    uint gid,
    thread float3& prediction) {
  const uint row_count = uint(config_i32[0]);
  const uint node_count = uint(config_i32[1]);
  const uint sample_count = uint(config_i32[2]);
  if (gid >= sample_count) {
    return false;
  }
  const int row_raw = sample_row_i32[gid];
  if (row_raw < 0 || uint(row_raw) >= row_count) {
    const float invalid = wf2_quiet_nan();
    prediction = float3(invalid);
    atomic_fetch_add_explicit(&cone_diagnostic_i32[0], 1, memory_order_relaxed);
    atomic_fetch_add_explicit(&cone_diagnostic_i32[1], 1, memory_order_relaxed);
    atomic_store_explicit(&cone_diagnostic_i32[2], int(gid + 1u), memory_order_relaxed);
    atomic_fetch_add_explicit(&loss_f32[0], invalid, memory_order_relaxed);
    return true;
  }
  const uint row_id = uint(row_raw);
  const uint weight_base = gid * node_count;
  const uint node_base = row_id * node_count;
  float4 chart = float4(0.0f);
  for (uint node_id = 0u; node_id < node_count; ++node_id) {
    const float weight = sample_to_node_f32[weight_base + node_id];
    const uint chart_base = (node_base + node_id) * 4u;
    chart += weight * float4(
        node_chart_f32[chart_base + 0u],
        node_chart_f32[chart_base + 1u],
        node_chart_f32[chart_base + 2u],
        node_chart_f32[chart_base + 3u]);
  }
  const float kappa = chart.x;
  const float3 velocity = chart.yzw;
  const float cone_tolerance = config_f32[0];
  const float loss_scale = config_f32[1];
  const float cone_violation = max(
      max(-kappa, -min(velocity.x, min(velocity.y, velocity.z))),
      max(velocity.x, max(velocity.y, velocity.z)) - kappa);
  const bool chart_finite = all(isfinite(chart));
  bool valid = chart_finite && cone_violation <= cone_tolerance;
  float phi = 1.0f;
  float phi_prime = 0.0f;
  if (valid) {
    wf2_lie_phi_and_derivative(kappa, phi, phi_prime);
  }
  const float beta = valid ? exp(-kappa) : 0.0f;
  const float3 moment = phi * velocity;
  const float3 background = float3(
      background_rgb_f32[0],
      background_rgb_f32[1],
      background_rgb_f32[2]);
  const uint prediction_base = gid * 3u;
  const float3 target = float3(
      target_rgb_f32[prediction_base + 0u],
      target_rgb_f32[prediction_base + 1u],
      target_rgb_f32[prediction_base + 2u]);
  prediction = moment + beta * background;
  const bool prediction_finite = all(isfinite(prediction));
  valid = valid && isfinite(beta) && prediction_finite &&
      isfinite(loss_scale) && loss_scale > 0.0f &&
      all(isfinite(background)) && all(isfinite(target));
  if (!valid) {
    const float invalid = wf2_quiet_nan();
    prediction = float3(invalid);
    atomic_fetch_add_explicit(&cone_diagnostic_i32[0], 1, memory_order_relaxed);
    if (!chart_finite || !prediction_finite || !all(isfinite(target))) {
      atomic_fetch_add_explicit(&cone_diagnostic_i32[1], 1, memory_order_relaxed);
    }
    atomic_store_explicit(&cone_diagnostic_i32[2], int(gid + 1u), memory_order_relaxed);
    atomic_fetch_add_explicit(&loss_f32[0], invalid, memory_order_relaxed);
    return true;
  }
  const float3 diff = prediction - target;
  atomic_fetch_add_explicit(
      &loss_f32[0],
      dot(diff, diff) * loss_scale,
      memory_order_relaxed);
  const float3 grad_prediction = (2.0f * loss_scale) * diff;
  const float grad_beta = dot(grad_prediction, background);
  const float4 grad_chart = float4(
      -beta * grad_beta + phi_prime * dot(velocity, grad_prediction),
      phi * grad_prediction);
  for (uint node_id = 0u; node_id < node_count; ++node_id) {
    wf2_atomic_add4(
        grad_node_chart_f32,
        (node_base + node_id) * 4u,
        sample_to_node_f32[weight_base + node_id] * grad_chart);
  }
  return true;
}

kernel void wf2_kinetic_ragged_p0_lie_sample_mse_vjp_tensor(
    device const float* node_chart_f32 [[buffer(0)]],
    device const int* sample_row_i32 [[buffer(1)]],
    device const float* sample_to_node_f32 [[buffer(2)]],
    device const float* target_rgb_f32 [[buffer(3)]],
    device const float* background_rgb_f32 [[buffer(4)]],
    device float* prediction_rgb_f32 [[buffer(5)]],
    device atomic_float* loss_f32 [[buffer(6)]],
    device atomic_float* grad_node_chart_f32 [[buffer(7)]],
    device atomic_int* cone_diagnostic_i32 [[buffer(8)]],
    device const int* config_i32 [[buffer(9)]],
    device const float* config_f32 [[buffer(10)]],
    uint gid [[thread_position_in_grid]]) {
  float3 prediction = float3(0.0f);
  if (!wf2_kinetic_ragged_p0_lie_sample_mse_vjp(
          node_chart_f32,
          sample_row_i32,
          sample_to_node_f32,
          target_rgb_f32,
          background_rgb_f32,
          loss_f32,
          grad_node_chart_f32,
          cone_diagnostic_i32,
          config_i32,
          config_f32,
          gid,
          prediction)) {
    return;
  }
  const uint prediction_base = gid * 3u;
  prediction_rgb_f32[prediction_base + 0u] = prediction.x;
  prediction_rgb_f32[prediction_base + 1u] = prediction.y;
  prediction_rgb_f32[prediction_base + 2u] = prediction.z;
}

kernel void wf2_kinetic_ragged_p0_lie_sample_mse_vjp_accumulate_only_tensor(
    device const float* node_chart_f32 [[buffer(0)]],
    device const int* sample_row_i32 [[buffer(1)]],
    device const float* sample_to_node_f32 [[buffer(2)]],
    device const float* target_rgb_f32 [[buffer(3)]],
    device const float* background_rgb_f32 [[buffer(4)]],
    device atomic_float* loss_f32 [[buffer(5)]],
    device atomic_float* grad_node_chart_f32 [[buffer(6)]],
    device atomic_int* cone_diagnostic_i32 [[buffer(7)]],
    device const int* config_i32 [[buffer(8)]],
    device const float* config_f32 [[buffer(9)]],
    uint gid [[thread_position_in_grid]]) {
  float3 discarded_prediction = float3(0.0f);
  wf2_kinetic_ragged_p0_lie_sample_mse_vjp(
      node_chart_f32,
      sample_row_i32,
      sample_to_node_f32,
      target_rgb_f32,
      background_rgb_f32,
      loss_f32,
      grad_node_chart_f32,
      cone_diagnostic_i32,
      config_i32,
      config_f32,
      gid,
      discarded_prediction);
}

static inline void wf2_fixed_word_p0_lie_node_vjp_impl(
    device const float* mobius_coeff_f32,
    device const float* track_ray_coeff_f32,
    device const float* compiler_node_t_f32,
    device const int* word_offsets_i32,
    device const int* word_owner_i32,
    device const int* word_left_incidence_i32,
    device const int* word_right_incidence_i32,
    device const int* track_incidence_offsets_i32,
    device const float* site_rgba_f32,
    device const float* node_chart_f32,
    device const float* grad_node_chart_f32,
    device atomic_float* grad_site_rgba_f32,
    device atomic_float* grad_mobius_coeff_f32,
    device const int* config_i32,
    device const float* config_f32,
    const bool accumulate_geometry,
    const uint gid) {
  const uint track_count = uint(config_i32[1]);
  const uint node_count = uint(config_i32[2]);
  const uint site_count = uint(config_i32[4]);
  const uint incidence_count = uint(config_i32[6]);
  if (gid >= track_count * node_count) {
    return;
  }
  const uint track_id = gid / node_count;
  const uint node_id = gid - track_id * node_count;
  const uint chart_base = gid * 4u;
  const float kappa_total = node_chart_f32[chart_base + 0u];
  const float3 velocity_total = float3(
      node_chart_f32[chart_base + 1u],
      node_chart_f32[chart_base + 2u],
      node_chart_f32[chart_base + 3u]);
  const float4 grad_chart = float4(
      grad_node_chart_f32[chart_base + 0u],
      grad_node_chart_f32[chart_base + 1u],
      grad_node_chart_f32[chart_base + 2u],
      grad_node_chart_f32[chart_base + 3u]);
  if (!isfinite(kappa_total) || !all(isfinite(velocity_total)) || !all(isfinite(grad_chart))) {
    return;
  }
  float phi = 1.0f;
  float phi_prime = 0.0f;
  float inverse_phi = 1.0f;
  float inverse_phi_prime = 0.0f;
  wf2_lie_phi_and_derivative(kappa_total, phi, phi_prime);
  wf2_lie_inverse_phi_and_derivative(kappa_total, inverse_phi, inverse_phi_prime);
  const float3 total_m = phi * velocity_total;
  const float3 bar_m = inverse_phi * grad_chart.yzw;
  const float bar_kappa_word =
      grad_chart.x + inverse_phi_prime * dot(total_m, grad_chart.yzw);
  const float t = compiler_node_t_f32[node_id];
  const float near_depth = config_f32[0];
  const float far_depth = config_f32[1];
  const float invalid_epsilon = config_f32[2];
  const float physical_length_epsilon = config_f32[3];
  const uint track_base = track_id * 12u;
  const float3 ray_direction = float3(
      track_ray_coeff_f32[track_base + 6u] + t * track_ray_coeff_f32[track_base + 9u],
      track_ray_coeff_f32[track_base + 7u] + t * track_ray_coeff_f32[track_base + 10u],
      track_ray_coeff_f32[track_base + 8u] + t * track_ray_coeff_f32[track_base + 11u]);
  const float fiber_speed = length(ray_direction);
  if (!(fiber_speed > 0.0f) || !isfinite(fiber_speed)) {
    return;
  }
  float prefix_beta = 1.0f;
  float3 prefix_m = float3(0.0f);
  int cached_right_cut = -2147483648;
  float cached_right_depth = 0.0f;
  uint cached_right_incidence_id = 0xFFFFFFFFu;
  float4 cached_right_grad_mobius = float4(0.0f);
  bool cached_right_valid = false;
  const int word_begin_raw = word_offsets_i32[track_id];
  const int word_end_raw = word_offsets_i32[track_id + 1u];
  for (int cursor = word_begin_raw; cursor < word_end_raw; ++cursor) {
    const int owner_raw = word_owner_i32[uint(cursor)];
    const int left_cut = word_left_incidence_i32[uint(cursor)];
    const int right_cut = word_right_incidence_i32[uint(cursor)];
    if (owner_raw < 0 || uint(owner_raw) >= site_count) {
      return;
    }
    float start_depth = 0.0f;
    uint start_incidence_id = 0xFFFFFFFFu;
    float4 start_grad_mobius = float4(0.0f);
    bool start_valid = false;
    if (cached_right_valid && left_cut == cached_right_cut) {
      start_depth = cached_right_depth;
      start_incidence_id = cached_right_incidence_id;
      start_grad_mobius = cached_right_grad_mobius;
      start_valid = true;
    } else {
      start_valid = wf2_endpoint_record_sparse_mobius_cut_depth_jacobian(
          mobius_coeff_f32,
          track_incidence_offsets_i32,
          incidence_count,
          track_id,
          left_cut,
          t,
          near_depth,
          far_depth,
          invalid_epsilon,
          start_incidence_id,
          start_depth,
          start_grad_mobius);
    }
    float end_depth = 0.0f;
    uint end_incidence_id = 0xFFFFFFFFu;
    float4 end_grad_mobius = float4(0.0f);
    if (!start_valid ||
        !wf2_endpoint_record_sparse_mobius_cut_depth_jacobian(
            mobius_coeff_f32,
            track_incidence_offsets_i32,
            incidence_count,
            track_id,
            right_cut,
            t,
            near_depth,
            far_depth,
            invalid_epsilon,
            end_incidence_id,
            end_depth,
            end_grad_mobius)) {
      return;
    }
    cached_right_cut = right_cut;
    cached_right_depth = end_depth;
    cached_right_incidence_id = end_incidence_id;
    cached_right_grad_mobius = end_grad_mobius;
    cached_right_valid = true;
    const float physical_length = fiber_speed * (end_depth - start_depth);
    if (!(physical_length > physical_length_epsilon) || !isfinite(physical_length)) {
      return;
    }
    const uint owner = uint(owner_raw);
    const uint rgba_base = owner * 4u;
    const float raw_density = site_rgba_f32[rgba_base + 3u];
    if (!isfinite(raw_density) || raw_density < 0.0f) {
      return;
    }
    const float density = raw_density;
    const float optical_depth = density * physical_length;
    const float segment_beta = exp(-optical_depth);
    const float segment_alpha = -expm1(-optical_depth);
    const float3 rgb = float3(
        site_rgba_f32[rgba_base + 0u],
        site_rgba_f32[rgba_base + 1u],
        site_rgba_f32[rgba_base + 2u]);
    const float tau_bar =
        dot(bar_m, prefix_m + prefix_beta * rgb - total_m) + bar_kappa_word;
    float4 grad_rgba = float4(
        prefix_beta * segment_alpha * bar_m,
        0.0f);
    // One-sided physical-density derivative at zero; do not erase dormant
    // material tangents used by the adaptive rank certificate.
    grad_rgba.w = physical_length * tau_bar;
    wf2_atomic_add4(grad_site_rgba_f32, rgba_base, grad_rgba);
    if (accumulate_geometry) {
      const float endpoint_depth_bar = fiber_speed * density * tau_bar;
      if (start_incidence_id != 0xFFFFFFFFu) {
        wf2_atomic_add4(
            grad_mobius_coeff_f32,
            start_incidence_id * 4u,
            -endpoint_depth_bar * start_grad_mobius);
      }
      if (end_incidence_id != 0xFFFFFFFFu) {
        wf2_atomic_add4(
            grad_mobius_coeff_f32,
            end_incidence_id * 4u,
            endpoint_depth_bar * end_grad_mobius);
      }
    }
    prefix_m += prefix_beta * segment_alpha * rgb;
    prefix_beta *= segment_beta;
  }
}

kernel void wf2_fixed_word_p0_lie_node_vjp_tensor(
    device const float* mobius_coeff_f32 [[buffer(0)]],
    device const float* track_ray_coeff_f32 [[buffer(1)]],
    device const float* compiler_node_t_f32 [[buffer(2)]],
    device const int* word_offsets_i32 [[buffer(3)]],
    device const int* word_owner_i32 [[buffer(4)]],
    device const int* word_left_incidence_i32 [[buffer(5)]],
    device const int* word_right_incidence_i32 [[buffer(6)]],
    device const int* track_incidence_offsets_i32 [[buffer(7)]],
    device const float* site_rgba_f32 [[buffer(8)]],
    device const float* node_chart_f32 [[buffer(9)]],
    device const float* grad_node_chart_f32 [[buffer(10)]],
    device atomic_float* grad_site_rgba_f32 [[buffer(11)]],
    device atomic_float* grad_mobius_coeff_f32 [[buffer(12)]],
    device const int* config_i32 [[buffer(13)]],
    device const float* config_f32 [[buffer(14)]],
    uint gid [[thread_position_in_grid]]) {
  wf2_fixed_word_p0_lie_node_vjp_impl(
      mobius_coeff_f32,
      track_ray_coeff_f32,
      compiler_node_t_f32,
      word_offsets_i32,
      word_owner_i32,
      word_left_incidence_i32,
      word_right_incidence_i32,
      track_incidence_offsets_i32,
      site_rgba_f32,
      node_chart_f32,
      grad_node_chart_f32,
      grad_site_rgba_f32,
      grad_mobius_coeff_f32,
      config_i32,
      config_f32,
      true,
      gid);
}

kernel void wf2_fixed_word_p0_lie_material_node_vjp_tensor(
    device const float* mobius_coeff_f32 [[buffer(0)]],
    device const float* track_ray_coeff_f32 [[buffer(1)]],
    device const float* compiler_node_t_f32 [[buffer(2)]],
    device const int* word_offsets_i32 [[buffer(3)]],
    device const int* word_owner_i32 [[buffer(4)]],
    device const int* word_left_incidence_i32 [[buffer(5)]],
    device const int* word_right_incidence_i32 [[buffer(6)]],
    device const int* track_incidence_offsets_i32 [[buffer(7)]],
    device const float* site_rgba_f32 [[buffer(8)]],
    device const float* node_chart_f32 [[buffer(9)]],
    device const float* grad_node_chart_f32 [[buffer(10)]],
    device atomic_float* grad_site_rgba_f32 [[buffer(11)]],
    device const int* config_i32 [[buffer(12)]],
    device const float* config_f32 [[buffer(13)]],
    uint gid [[thread_position_in_grid]]) {
  // The helper requires a typed pointer for its strict branch. Passing the
  // material bar as that unused pointer introduces no storage or write: the
  // false capability flag makes every geometry accumulation unreachable.
  wf2_fixed_word_p0_lie_node_vjp_impl(
      mobius_coeff_f32,
      track_ray_coeff_f32,
      compiler_node_t_f32,
      word_offsets_i32,
      word_owner_i32,
      word_left_incidence_i32,
      word_right_incidence_i32,
      track_incidence_offsets_i32,
      site_rgba_f32,
      node_chart_f32,
      grad_node_chart_f32,
      grad_site_rgba_f32,
      grad_site_rgba_f32,
      config_i32,
      config_f32,
      false,
      gid);
}

// Reverse of the bounded kinetic node program.  Each thread owns one
// (track,node) pair.  CSR word ranges are disjoint across tracks, so every
// [node,cursor] physical-length adjoint has a unique writer; material rows may
// repeat and therefore use atomics.  The material-only entry point passes the
// material bar as an unused buffer(7) alias and sets write_length_bar=false,
// avoiding the otherwise wasted [J,W] geometry-bar allocation.
static inline void wf2_kinetic_precompiled_length_p0_lie_node_vjp_impl(
    device const int* word_offsets_i32,
    device const int* word_owner_i32,
    device const float* node_physical_length_f32,
    device const float* site_rgba_f32,
    device const float* node_chart_f32,
    device const float* grad_node_chart_f32,
    device atomic_float* grad_site_rgba_f32,
    device float* grad_node_physical_length_f32,
    device const int* config_i32,
    device const float* config_f32,
    const bool write_length_bar,
    const uint gid) {
  const uint track_count = uint(config_i32[0]);
  const uint node_count = uint(config_i32[1]);
  const uint site_count = uint(config_i32[2]);
  const uint word_count = uint(config_i32[3]);
  if (gid >= track_count * node_count) {
    return;
  }
  const uint track_id = gid / node_count;
  const uint node_id = gid - track_id * node_count;
  const uint chart_base = gid * 4u;
  const float kappa_total = node_chart_f32[chart_base + 0u];
  const float3 velocity_total = float3(
      node_chart_f32[chart_base + 1u],
      node_chart_f32[chart_base + 2u],
      node_chart_f32[chart_base + 3u]);
  const float4 grad_chart = float4(
      grad_node_chart_f32[chart_base + 0u],
      grad_node_chart_f32[chart_base + 1u],
      grad_node_chart_f32[chart_base + 2u],
      grad_node_chart_f32[chart_base + 3u]);
  if (!isfinite(kappa_total) || !all(isfinite(velocity_total)) ||
      !all(isfinite(grad_chart))) {
    return;
  }
  float phi = 1.0f;
  float phi_prime = 0.0f;
  float inverse_phi = 1.0f;
  float inverse_phi_prime = 0.0f;
  wf2_lie_phi_and_derivative(kappa_total, phi, phi_prime);
  wf2_lie_inverse_phi_and_derivative(
      kappa_total, inverse_phi, inverse_phi_prime);
  const float3 total_m = phi * velocity_total;
  const float3 bar_m = inverse_phi * grad_chart.yzw;
  const float bar_kappa_word =
      grad_chart.x + inverse_phi_prime * dot(total_m, grad_chart.yzw);
  const int word_begin_raw = word_offsets_i32[track_id];
  const int word_end_raw = word_offsets_i32[track_id + 1u];
  const float physical_length_epsilon = config_f32[0];
  if (word_begin_raw < 0 || word_end_raw <= word_begin_raw ||
      uint(word_end_raw) > word_count) {
    return;
  }
  const uint node_length_base = node_id * word_count;
  float prefix_beta = 1.0f;
  float3 prefix_m = float3(0.0f);
  for (int cursor = word_begin_raw; cursor < word_end_raw; ++cursor) {
    const int owner_raw = word_owner_i32[uint(cursor)];
    if (owner_raw < 0 || uint(owner_raw) >= site_count) {
      return;
    }
    const uint length_index = node_length_base + uint(cursor);
    const float physical_length = node_physical_length_f32[length_index];
    const uint owner = uint(owner_raw);
    const uint rgba_base = owner * 4u;
    const float raw_density = site_rgba_f32[rgba_base + 3u];
    if (!(physical_length > physical_length_epsilon) ||
        !isfinite(physical_length) || !isfinite(raw_density) || raw_density < 0.0f) {
      return;
    }
    const float density = raw_density;
    const float optical_depth = density * physical_length;
    if (!isfinite(optical_depth)) {
      return;
    }
    const float segment_beta = exp(-optical_depth);
    const float segment_alpha = -expm1(-optical_depth);
    const float3 rgb = float3(
        site_rgba_f32[rgba_base + 0u],
        site_rgba_f32[rgba_base + 1u],
        site_rgba_f32[rgba_base + 2u]);
    const float tau_bar =
        dot(bar_m, prefix_m + prefix_beta * rgb - total_m) + bar_kappa_word;
    const float4 grad_rgba = float4(
        prefix_beta * segment_alpha * bar_m,
        physical_length * tau_bar);
    wf2_atomic_add4(grad_site_rgba_f32, rgba_base, grad_rgba);
    if (write_length_bar) {
      grad_node_physical_length_f32[length_index] = density * tau_bar;
    }
    prefix_m += prefix_beta * segment_alpha * rgb;
    prefix_beta *= segment_beta;
  }
}

kernel void wf2_kinetic_precompiled_length_p0_lie_node_vjp_tensor(
    device const int* word_offsets_i32 [[buffer(0)]],
    device const int* word_owner_i32 [[buffer(1)]],
    device const float* node_physical_length_f32 [[buffer(2)]],
    device const float* site_rgba_f32 [[buffer(3)]],
    device const float* node_chart_f32 [[buffer(4)]],
    device const float* grad_node_chart_f32 [[buffer(5)]],
    device atomic_float* grad_site_rgba_f32 [[buffer(6)]],
    device float* grad_node_physical_length_f32 [[buffer(7)]],
    device const int* config_i32 [[buffer(8)]],
    device const float* config_f32 [[buffer(9)]],
    uint gid [[thread_position_in_grid]]) {
  wf2_kinetic_precompiled_length_p0_lie_node_vjp_impl(
      word_offsets_i32,
      word_owner_i32,
      node_physical_length_f32,
      site_rgba_f32,
      node_chart_f32,
      grad_node_chart_f32,
      grad_site_rgba_f32,
      grad_node_physical_length_f32,
      config_i32,
      config_f32,
      true,
      gid);
}

kernel void wf2_kinetic_precompiled_length_p0_lie_material_node_vjp_tensor(
    device const int* word_offsets_i32 [[buffer(0)]],
    device const int* word_owner_i32 [[buffer(1)]],
    device const float* node_physical_length_f32 [[buffer(2)]],
    device const float* site_rgba_f32 [[buffer(3)]],
    device const float* node_chart_f32 [[buffer(4)]],
    device const float* grad_node_chart_f32 [[buffer(5)]],
    device atomic_float* grad_site_rgba_f32 [[buffer(6)]],
    device float* unused_length_bar_f32 [[buffer(7)]],
    device const int* config_i32 [[buffer(8)]],
    device const float* config_f32 [[buffer(9)]],
    uint gid [[thread_position_in_grid]]) {
  // buffer(7) aliases the caller-owned material bar.  The false capability
  // flag makes geometry writes unreachable, so this introduces no allocation
  // and no write outside the atomic material accumulation at buffer(6).
  wf2_kinetic_precompiled_length_p0_lie_node_vjp_impl(
      word_offsets_i32,
      word_owner_i32,
      node_physical_length_f32,
      site_rgba_f32,
      node_chart_f32,
      grad_node_chart_f32,
      grad_site_rgba_f32,
      unused_length_bar_f32,
      config_i32,
      config_f32,
      false,
      gid);
}

#define WF2_KINETIC_FUSED_V1_REASON_CONFIG 0x01u
#define WF2_KINETIC_FUSED_V1_REASON_TOPOLOGY 0x02u
#define WF2_KINETIC_FUSED_V1_REASON_THRESHOLDS 0x04u
#define WF2_KINETIC_FUSED_V1_REASON_RAY_DOMAIN 0x08u
#define WF2_KINETIC_FUSED_V1_REASON_CHART 0x10u
#define WF2_KINETIC_FUSED_V1_REASON_MATERIAL_LENGTH 0x20u
#define WF2_KINETIC_FUSED_V1_REASON_GEOMETRY 0x40u
#define WF2_KINETIC_FUSED_V1_REASON_REVERSE 0x80u
#define WF2_KINETIC_FUSED_V1_REASON_OUTPUT_LEDGER 0x100u
#define WF2_KINETIC_FUSED_V2_REASON_INDEX_SPACE 0x200u

// Dry-run the exact fixed-camera fused-v1 arithmetic before any output bar is
// touched.  A per-thread guard in the write kernel is not transactional: one
// row/node can already have atomically accumulated while another discovers an
// invalid value.  This helper therefore validates topology, float32 margins,
// the dynamic node cotangent, and every reverse contribution in a separate
// grid-wide pass.  That pass also admits only finite, exactly-zero output
// scratch, so the additive result cannot include a prior transaction prefix.
// The write kernel is enqueued later on the same serialized stream and reads
// the completed global reason mask before its first atomic; no host observation
// is required between the two device phases.
static inline uint wf2_kinetic_fused_direct_full_vjp_validation_reason_v1(
    device const int* word_offsets_i32,
    device const int* word_owner_i32,
    device const long* source_site_ids_i64,
    device const float* node_physical_length_f32,
    device const float* site_rgba_f32,
    device const float* node_chart_f32,
    device const float* row_node_time_f32,
    device const float* row_near_far_f32,
    device const float* row_ray_coeff_f32,
    device const float* compact_positions0_f32,
    device const float* compact_velocities_f32,
    device const float* compact_weight_coefficients_f32,
    device const float* grad_node_chart_f32,
    device const int* config_i32,
    device const float* config_f32,
    const uint row_count,
    const uint node_count,
    const uint compact_site_count,
    const uint word_count,
    const uint weight_coefficient_count,
    const uint global_site_count,
    const uint gid) {
  if (config_i32[0] != int(row_count) || config_i32[1] != int(node_count) ||
      config_i32[2] != int(compact_site_count) ||
      config_i32[3] != int(word_count) ||
      config_i32[4] != int(weight_coefficient_count) ||
      config_i32[5] != int(global_site_count) ||
      weight_coefficient_count < 1u || weight_coefficient_count > 3u) {
    return WF2_KINETIC_FUSED_V1_REASON_CONFIG;
  }
  if (gid >= row_count * node_count) {
    return 0u;
  }

  const uint row_id = gid / node_count;
  const uint node_id = gid - row_id * node_count;
  const int word_begin_raw = word_offsets_i32[row_id];
  const int word_end_raw = word_offsets_i32[row_id + 1u];
  if (word_begin_raw < 0 || word_end_raw <= word_begin_raw ||
      uint(word_end_raw) > word_count) {
    return WF2_KINETIC_FUSED_V1_REASON_TOPOLOGY;
  }

  const float physical_length_epsilon = config_f32[0];
  const float minimum_absolute_cut_denominator = config_f32[1];
  const float minimum_ray_speed = config_f32[2];
  const float depth_closure_relative_tolerance = config_f32[3];
  const float active_tie_relative_tolerance = config_f32[4];
  const float minimum_cut_cosine = config_f32[5];
  const float minimum_coordinate_length = config_f32[6];
  if (!isfinite(physical_length_epsilon) ||
      !(physical_length_epsilon > 0.0f) ||
      !isfinite(minimum_absolute_cut_denominator) ||
      !(minimum_absolute_cut_denominator > 0.0f) ||
      !isfinite(minimum_ray_speed) || !(minimum_ray_speed > 0.0f) ||
      !isfinite(depth_closure_relative_tolerance) ||
      depth_closure_relative_tolerance < 0.0f ||
      !isfinite(active_tie_relative_tolerance) ||
      active_tie_relative_tolerance < 0.0f ||
      !isfinite(minimum_cut_cosine) || !(minimum_cut_cosine > 0.0f) ||
      minimum_cut_cosine > 1.0f ||
      !isfinite(minimum_coordinate_length) ||
      !(minimum_coordinate_length > 0.0f)) {
    return WF2_KINETIC_FUSED_V1_REASON_THRESHOLDS;
  }

  const float time = row_node_time_f32[row_id * node_count + node_id];
  const uint near_far_base = row_id * 2u;
  const float near_depth = row_near_far_f32[near_far_base + 0u];
  const float far_depth = row_near_far_f32[near_far_base + 1u];
  const uint ray_base = row_id * 12u;
  const float3 origin0 = float3(
      row_ray_coeff_f32[ray_base + 0u],
      row_ray_coeff_f32[ray_base + 1u],
      row_ray_coeff_f32[ray_base + 2u]);
  const float3 origin1 = float3(
      row_ray_coeff_f32[ray_base + 3u],
      row_ray_coeff_f32[ray_base + 4u],
      row_ray_coeff_f32[ray_base + 5u]);
  const float3 direction0 = float3(
      row_ray_coeff_f32[ray_base + 6u],
      row_ray_coeff_f32[ray_base + 7u],
      row_ray_coeff_f32[ray_base + 8u]);
  const float3 direction1 = float3(
      row_ray_coeff_f32[ray_base + 9u],
      row_ray_coeff_f32[ray_base + 10u],
      row_ray_coeff_f32[ray_base + 11u]);
  const float3 origin = origin0 + time * origin1;
  const float3 direction = direction0 + time * direction1;
  const float ray_speed = length(direction);
  if (!isfinite(time) || !isfinite(near_depth) || !isfinite(far_depth) ||
      !(far_depth > near_depth) || !all(isfinite(origin)) ||
      !all(isfinite(direction)) || !(ray_speed > minimum_ray_speed) ||
      !isfinite(ray_speed)) {
    return WF2_KINETIC_FUSED_V1_REASON_RAY_DOMAIN;
  }

  const uint chart_base = gid * 4u;
  const float kappa_total = node_chart_f32[chart_base + 0u];
  const float3 velocity_total = float3(
      node_chart_f32[chart_base + 1u],
      node_chart_f32[chart_base + 2u],
      node_chart_f32[chart_base + 3u]);
  const float4 grad_chart = float4(
      grad_node_chart_f32[chart_base + 0u],
      grad_node_chart_f32[chart_base + 1u],
      grad_node_chart_f32[chart_base + 2u],
      grad_node_chart_f32[chart_base + 3u]);
  if (!isfinite(kappa_total) || !all(isfinite(velocity_total)) ||
      !all(isfinite(grad_chart))) {
    return WF2_KINETIC_FUSED_V1_REASON_CHART;
  }

  const uint node_length_base = node_id * word_count;
  float preflight_depth = near_depth;
  for (int cursor = word_begin_raw; cursor < word_end_raw; ++cursor) {
    const uint word_index = uint(cursor);
    const int owner_raw = word_owner_i32[word_index];
    if (owner_raw < 0 || uint(owner_raw) >= compact_site_count) {
      return WF2_KINETIC_FUSED_V1_REASON_TOPOLOGY;
    }
    const long global_owner_raw = source_site_ids_i64[uint(owner_raw)];
    if (global_owner_raw < 0l || global_owner_raw >= long(global_site_count)) {
      return WF2_KINETIC_FUSED_V1_REASON_TOPOLOGY;
    }
    const float physical_length =
        node_physical_length_f32[node_length_base + word_index];
    const uint rgba_base = uint(owner_raw) * 4u;
    const float4 rgba = float4(
        site_rgba_f32[rgba_base + 0u],
        site_rgba_f32[rgba_base + 1u],
        site_rgba_f32[rgba_base + 2u],
        site_rgba_f32[rgba_base + 3u]);
    const float optical_depth = rgba.w * physical_length;
    if (!(physical_length > physical_length_epsilon) ||
        !isfinite(physical_length) || !all(isfinite(rgba)) || rgba.w < 0.0f ||
        !isfinite(optical_depth)) {
      return WF2_KINETIC_FUSED_V1_REASON_MATERIAL_LENGTH;
    }
    const float coordinate_length = physical_length / ray_speed;
    if (!isfinite(coordinate_length) ||
        !(coordinate_length > minimum_coordinate_length)) {
      return WF2_KINETIC_FUSED_V1_REASON_MATERIAL_LENGTH;
    }
    preflight_depth += coordinate_length;
    if (cursor + 1 < word_end_raw) {
      const int right_owner_raw = word_owner_i32[word_index + 1u];
      if (right_owner_raw < 0 || uint(right_owner_raw) >= compact_site_count) {
        return WF2_KINETIC_FUSED_V1_REASON_TOPOLOGY;
      }
      const uint left_owner = uint(owner_raw);
      const uint right_owner = uint(right_owner_raw);
      const uint left_position_base = left_owner * 3u;
      const uint right_position_base = right_owner * 3u;
      const float3 left_position = float3(
          compact_positions0_f32[left_position_base + 0u],
          compact_positions0_f32[left_position_base + 1u],
          compact_positions0_f32[left_position_base + 2u]) +
          time * float3(
              compact_velocities_f32[left_position_base + 0u],
              compact_velocities_f32[left_position_base + 1u],
              compact_velocities_f32[left_position_base + 2u]);
      const float3 right_position = float3(
          compact_positions0_f32[right_position_base + 0u],
          compact_positions0_f32[right_position_base + 1u],
          compact_positions0_f32[right_position_base + 2u]) +
          time * float3(
              compact_velocities_f32[right_position_base + 0u],
              compact_velocities_f32[right_position_base + 1u],
              compact_velocities_f32[right_position_base + 2u]);
      float left_weight = 0.0f;
      float right_weight = 0.0f;
      float time_power = 1.0f;
      for (uint coefficient = 0u; coefficient < weight_coefficient_count;
           ++coefficient) {
        left_weight += compact_weight_coefficients_f32[
            left_owner * weight_coefficient_count + coefficient] * time_power;
        right_weight += compact_weight_coefficients_f32[
            right_owner * weight_coefficient_count + coefficient] * time_power;
        time_power *= time;
      }
      const float3 normal = 2.0f * (right_position - left_position);
      const float denominator = dot(normal, direction);
      const float denominator_scale = length(normal) * ray_speed;
      const float cut_cosine =
          denominator_scale > 0.0f ? fabs(denominator) / denominator_scale : 0.0f;
      const float intercept =
          dot(normal, origin) + dot(left_position, left_position) -
          dot(right_position, right_position) - left_weight + right_weight;
      const float tie_residual = fabs(intercept + preflight_depth * denominator);
      const float tie_scale = max(1.0f, fabs(intercept));
      const float tie_limit = active_tie_relative_tolerance * tie_scale;
      if (!all(isfinite(left_position)) || !all(isfinite(right_position)) ||
          !isfinite(left_weight) || !isfinite(right_weight) ||
          !all(isfinite(normal)) || !isfinite(denominator) ||
          !(fabs(denominator) > minimum_absolute_cut_denominator) ||
          !isfinite(denominator_scale) || !isfinite(cut_cosine) ||
          !(cut_cosine > minimum_cut_cosine) || !isfinite(intercept) ||
          !isfinite(tie_residual) || !isfinite(tie_limit) ||
          tie_residual > tie_limit) {
        return WF2_KINETIC_FUSED_V1_REASON_GEOMETRY;
      }
    }
  }
  const float closure_scale = max(1.0f, max(fabs(near_depth), fabs(far_depth)));
  const float closure_limit = depth_closure_relative_tolerance * closure_scale;
  if (!isfinite(preflight_depth) || !isfinite(closure_scale) ||
      !isfinite(closure_limit) ||
      fabs(preflight_depth - far_depth) > closure_limit) {
    return WF2_KINETIC_FUSED_V1_REASON_GEOMETRY;
  }

  float phi = 1.0f;
  float phi_prime = 0.0f;
  float inverse_phi = 1.0f;
  float inverse_phi_prime = 0.0f;
  wf2_lie_phi_and_derivative(kappa_total, phi, phi_prime);
  wf2_lie_inverse_phi_and_derivative(
      kappa_total, inverse_phi, inverse_phi_prime);
  const float3 total_m = phi * velocity_total;
  const float3 bar_m = inverse_phi * grad_chart.yzw;
  const float bar_kappa_word =
      grad_chart.x + inverse_phi_prime * dot(total_m, grad_chart.yzw);
  if (!isfinite(phi) || !isfinite(phi_prime) || !isfinite(inverse_phi) ||
      !isfinite(inverse_phi_prime) || !all(isfinite(total_m)) ||
      !all(isfinite(bar_m)) || !isfinite(bar_kappa_word)) {
    return WF2_KINETIC_FUSED_V1_REASON_REVERSE;
  }

  float prefix_beta = 1.0f;
  float3 prefix_m = float3(0.0f);
  float cut_depth = near_depth;
  float previous_bar_ell = 0.0f;
  int previous_owner_raw = -1;
  for (int cursor = word_begin_raw; cursor < word_end_raw; ++cursor) {
    const uint word_index = uint(cursor);
    const uint current_owner = uint(word_owner_i32[word_index]);
    const uint rgba_base = current_owner * 4u;
    const float physical_length =
        node_physical_length_f32[node_length_base + word_index];
    const float density = site_rgba_f32[rgba_base + 3u];
    const float optical_depth = density * physical_length;
    const float segment_beta = exp(-optical_depth);
    const float segment_alpha = -expm1(-optical_depth);
    const float3 rgb = float3(
        site_rgba_f32[rgba_base + 0u],
        site_rgba_f32[rgba_base + 1u],
        site_rgba_f32[rgba_base + 2u]);
    const float tau_bar =
        dot(bar_m, prefix_m + prefix_beta * rgb - total_m) + bar_kappa_word;
    const float current_bar_ell = density * tau_bar;
    const float4 material_bar = float4(
        prefix_beta * segment_alpha * bar_m,
        physical_length * tau_bar);
    if (!isfinite(segment_beta) || !isfinite(segment_alpha) ||
        !isfinite(tau_bar) || !isfinite(current_bar_ell) ||
        !all(isfinite(material_bar))) {
      return WF2_KINETIC_FUSED_V1_REASON_REVERSE;
    }

    if (previous_owner_raw >= 0) {
      const uint left_owner = uint(previous_owner_raw);
      const uint right_owner = current_owner;
      const float cut_bar = ray_speed * (previous_bar_ell - current_bar_ell);
      const uint left_position_base = left_owner * 3u;
      const uint right_position_base = right_owner * 3u;
      const float3 left_position = float3(
          compact_positions0_f32[left_position_base + 0u],
          compact_positions0_f32[left_position_base + 1u],
          compact_positions0_f32[left_position_base + 2u]) +
          time * float3(
              compact_velocities_f32[left_position_base + 0u],
              compact_velocities_f32[left_position_base + 1u],
              compact_velocities_f32[left_position_base + 2u]);
      const float3 right_position = float3(
          compact_positions0_f32[right_position_base + 0u],
          compact_positions0_f32[right_position_base + 1u],
          compact_positions0_f32[right_position_base + 2u]) +
          time * float3(
              compact_velocities_f32[right_position_base + 0u],
              compact_velocities_f32[right_position_base + 1u],
              compact_velocities_f32[right_position_base + 2u]);
      const float3 normal = 2.0f * (right_position - left_position);
      const float denominator = dot(normal, direction);
      const float implicit_bar = -cut_bar / denominator;
      const float3 point = origin + cut_depth * direction;
      const float3 left_position_bar =
          implicit_bar * 2.0f * (left_position - point);
      const float3 right_position_bar =
          implicit_bar * 2.0f * (point - right_position);
      const float3 left_velocity_bar = time * left_position_bar;
      const float3 right_velocity_bar = time * right_position_bar;
      if (!isfinite(cut_bar) || !isfinite(implicit_bar) ||
          !all(isfinite(point)) || !all(isfinite(left_position_bar)) ||
          !all(isfinite(right_position_bar)) ||
          !all(isfinite(left_velocity_bar)) ||
          !all(isfinite(right_velocity_bar))) {
        return WF2_KINETIC_FUSED_V1_REASON_REVERSE;
      }
      float time_power = 1.0f;
      for (uint coefficient = 0u; coefficient < weight_coefficient_count;
           ++coefficient) {
        const float left_weight_bar = -implicit_bar * time_power;
        const float right_weight_bar = implicit_bar * time_power;
        if (!isfinite(time_power) || !isfinite(left_weight_bar) ||
            !isfinite(right_weight_bar)) {
          return WF2_KINETIC_FUSED_V1_REASON_REVERSE;
        }
        time_power *= time;
      }
    }

    const float coordinate_length = physical_length / ray_speed;
    cut_depth += coordinate_length;
    previous_bar_ell = current_bar_ell;
    previous_owner_raw = int(current_owner);
    prefix_m += prefix_beta * segment_alpha * rgb;
    prefix_beta *= segment_beta;
    if (!isfinite(cut_depth) || !all(isfinite(prefix_m)) ||
        !isfinite(prefix_beta)) {
      return WF2_KINETIC_FUSED_V1_REASON_REVERSE;
    }
  }
  return 0u;
}

kernel void wf2_kinetic_fused_direct_full_vjp_validate_v1_tensor(
    device const int* word_offsets_i32 [[buffer(0)]],
    device const int* word_owner_i32 [[buffer(1)]],
    device const long* source_site_ids_i64 [[buffer(2)]],
    device const float* node_physical_length_f32 [[buffer(3)]],
    device const float* site_rgba_f32 [[buffer(4)]],
    device const float* node_chart_f32 [[buffer(5)]],
    device const float* row_node_time_f32 [[buffer(6)]],
    device const float* row_near_far_f32 [[buffer(7)]],
    device const float* row_ray_coeff_f32 [[buffer(8)]],
    device const float* compact_positions0_f32 [[buffer(9)]],
    device const float* compact_velocities_f32 [[buffer(10)]],
    device const float* compact_weight_coefficients_f32 [[buffer(11)]],
    device const float* grad_node_chart_f32 [[buffer(12)]],
    device const float* grad_site_rgba_f32 [[buffer(13)]],
    device const float* grad_global_positions0_f32 [[buffer(14)]],
    device const float* grad_global_velocities_f32 [[buffer(15)]],
    device const float* grad_global_weight_coefficients_f32 [[buffer(16)]],
    device const int* config_i32 [[buffer(17)]],
    device const float* config_f32 [[buffer(18)]],
    constant int& row_count_i32 [[buffer(19)]],
    constant int& node_count_i32 [[buffer(20)]],
    constant int& compact_site_count_i32 [[buffer(21)]],
    constant int& word_count_i32 [[buffer(22)]],
    constant int& weight_coefficient_count_i32 [[buffer(23)]],
    constant int& global_site_count_i32 [[buffer(24)]],
    device atomic_uint* validation_status_u32 [[buffer(25)]],
    constant int& validate_shared_global_ledgers_i32 [[buffer(26)]],
    uint gid [[thread_position_in_grid]]) {
  uint reason = wf2_kinetic_fused_direct_full_vjp_validation_reason_v1(
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
      grad_node_chart_f32,
      config_i32,
      config_f32,
      uint(row_count_i32),
      uint(node_count_i32),
      uint(compact_site_count_i32),
      uint(word_count_i32),
      uint(weight_coefficient_count_i32),
      uint(global_site_count_i32),
      gid);
  const uint validation_thread_count =
      uint(row_count_i32) * uint(node_count_i32);
  // Output bars are transaction-local additive scratch, not persistent
  // accumulators.  Requiring an exact zero value here (with either sign of
  // floating-point zero accepted) prevents this transaction from silently
  // extending a previously authorized prefix.  Storage ownership and
  // quarantine after rejection remain caller obligations because a kernel
  // cannot prove the absence of hidden aliases.
  for (uint index = gid; index < uint(compact_site_count_i32) * 4u;
       index += validation_thread_count) {
    const float value = grad_site_rgba_f32[index];
    if (!isfinite(value) || value != 0.0f) {
      reason |= WF2_KINETIC_FUSED_V1_REASON_OUTPUT_LEDGER;
    }
  }
  // Every block has a distinct compact material ledger, so scan it above on
  // every block.  Position/velocity/weight ledgers are shared by all blocks in
  // one transaction and must be scanned exactly once (the coordinator sets
  // this flag only on its first validation dispatch).  This preserves
  // O(sum_b compact_sites_b + global_sites) ledger work instead of
  // O(sum_b compact_sites_b + blocks * global_sites).
  if (validate_shared_global_ledgers_i32 != 0) {
    for (uint index = gid; index < uint(global_site_count_i32) * 3u;
         index += validation_thread_count) {
      const float position_value = grad_global_positions0_f32[index];
      const float velocity_value = grad_global_velocities_f32[index];
      if (!isfinite(position_value) || position_value != 0.0f ||
          !isfinite(velocity_value) || velocity_value != 0.0f) {
        reason |= WF2_KINETIC_FUSED_V1_REASON_OUTPUT_LEDGER;
      }
    }
    for (uint index = gid;
         index < uint(global_site_count_i32) * uint(weight_coefficient_count_i32);
         index += validation_thread_count) {
      const float value = grad_global_weight_coefficients_f32[index];
      if (!isfinite(value) || value != 0.0f) {
        reason |= WF2_KINETIC_FUSED_V1_REASON_OUTPUT_LEDGER;
      }
    }
  }
  if (reason != 0u) {
    atomic_fetch_or_explicit(
        validation_status_u32, reason, memory_order_relaxed);
  }
}

// Postwrite closure for finite atomic-sum overflow.  Prevalidation proves each
// proposed contribution finite, but several finite contributions can still
// overflow one shared destination.  The four ledgers are disposable
// transaction scratch: after every guarded accumulation, this kernel scans each
// compact material ledger once and the shared global ledgers exactly once.  A
// nonfinite result ORs into the same four-byte receipt, so the host can reject
// and quarantine the scratch before optimizer authorization.  This finalizer
// does not promise byte-for-byte rollback of a postwrite failure.
kernel void wf2_kinetic_fused_direct_full_vjp_finalize_v1_tensor(
    device const float* grad_site_rgba_f32 [[buffer(0)]],
    device const float* grad_global_positions0_f32 [[buffer(1)]],
    device const float* grad_global_velocities_f32 [[buffer(2)]],
    device const float* grad_global_weight_coefficients_f32 [[buffer(3)]],
    device atomic_uint* validation_status_u32 [[buffer(4)]],
    constant int& compact_site_count_i32 [[buffer(5)]],
    constant int& global_site_count_i32 [[buffer(6)]],
    constant int& weight_coefficient_count_i32 [[buffer(7)]],
    constant int& finalize_shared_global_ledgers_i32 [[buffer(8)]],
    uint gid [[thread_position_in_grid]]) {
  uint reason = 0u;
  if (gid < uint(compact_site_count_i32) * 4u &&
      !isfinite(grad_site_rgba_f32[gid])) {
    reason |= WF2_KINETIC_FUSED_V1_REASON_OUTPUT_LEDGER;
  }
  if (finalize_shared_global_ledgers_i32 != 0) {
    if (gid < uint(global_site_count_i32) * 3u &&
        (!isfinite(grad_global_positions0_f32[gid]) ||
         !isfinite(grad_global_velocities_f32[gid]))) {
      reason |= WF2_KINETIC_FUSED_V1_REASON_OUTPUT_LEDGER;
    }
    if (gid < uint(global_site_count_i32) *
                  uint(weight_coefficient_count_i32) &&
        !isfinite(grad_global_weight_coefficients_f32[gid])) {
      reason |= WF2_KINETIC_FUSED_V1_REASON_OUTPUT_LEDGER;
    }
  }
  if (reason != 0u) {
    atomic_fetch_or_explicit(
        validation_status_u32, reason, memory_order_relaxed);
  }
}

// Source-only v1 fused kinetic reverse.  This suffixed entry point stays out of
// the selected-kernel attestation until a rebuilt extension matches the staged
// certified sparse oracle.  Every thread owns one (row,node).  It produces the
// ordered-word bar_ell and immediately lowers adjacent bar pairs into global
// affine-position, velocity, and polynomial-weight bars.  This v1 is explicitly
// fixed-camera-only: there is no camera-ray cotangent buffer or alias.  There is
// also no grad_node_physical_length argument or output, so a [J,W] length tape
// cannot be allocated, copied, or returned by this ABI.
kernel void wf2_kinetic_fused_direct_full_vjp_v1_tensor(
    device const int* word_offsets_i32 [[buffer(0)]],
    device const int* word_owner_i32 [[buffer(1)]],
    device const long* source_site_ids_i64 [[buffer(2)]],
    device const float* node_physical_length_f32 [[buffer(3)]],
    device const float* site_rgba_f32 [[buffer(4)]],
    device const float* node_chart_f32 [[buffer(5)]],
    device const float* row_node_time_f32 [[buffer(6)]],
    device const float* row_near_far_f32 [[buffer(7)]],
    device const float* row_ray_coeff_f32 [[buffer(8)]],
    device const float* compact_positions0_f32 [[buffer(9)]],
    device const float* compact_velocities_f32 [[buffer(10)]],
    device const float* compact_weight_coefficients_f32 [[buffer(11)]],
    device const float* grad_node_chart_f32 [[buffer(12)]],
    device atomic_float* grad_site_rgba_f32 [[buffer(13)]],
    device atomic_float* grad_global_positions0_f32 [[buffer(14)]],
    device atomic_float* grad_global_velocities_f32 [[buffer(15)]],
    device atomic_float* grad_global_weight_coefficients_f32 [[buffer(16)]],
    device const int* config_i32 [[buffer(17)]],
    device const float* config_f32 [[buffer(18)]],
    constant int& row_count_i32 [[buffer(19)]],
    constant int& node_count_i32 [[buffer(20)]],
    constant int& compact_site_count_i32 [[buffer(21)]],
    constant int& word_count_i32 [[buffer(22)]],
    constant int& weight_coefficient_count_i32 [[buffer(23)]],
    constant int& global_site_count_i32 [[buffer(24)]],
    device atomic_uint* validation_status_u32 [[buffer(25)]],
    uint gid [[thread_position_in_grid]]) {
  // ObjC++ enqueues the complete validation grid before this grid on the same
  // serialized MPS stream.  The shared scalar is therefore complete and visible
  // before any thread reaches this device-side write gate.
  if (atomic_load_explicit(validation_status_u32, memory_order_relaxed) != 0u) {
    return;
  }
  // Counts come from host-validated tensor shapes rather than device config
  // contents, so a malformed direct torch.ops call cannot enlarge an index
  // domain before the kernel's per-element guards run. config_i32 remains the
  // sealed prepared-token mirror and is intentionally not trusted for bounds.
  const uint row_count = uint(row_count_i32);
  const uint node_count = uint(node_count_i32);
  const uint compact_site_count = uint(compact_site_count_i32);
  const uint word_count = uint(word_count_i32);
  const uint weight_coefficient_count = uint(weight_coefficient_count_i32);
  const uint global_site_count = uint(global_site_count_i32);
  if (config_i32[0] != row_count_i32 || config_i32[1] != node_count_i32 ||
      config_i32[2] != compact_site_count_i32 ||
      config_i32[3] != word_count_i32 ||
      config_i32[4] != weight_coefficient_count_i32 ||
      config_i32[5] != global_site_count_i32) {
    return;
  }
  if (gid >= row_count * node_count || weight_coefficient_count < 1u ||
      weight_coefficient_count > 3u) {
    return;
  }

  const uint row_id = gid / node_count;
  const uint node_id = gid - row_id * node_count;
  const int word_begin_raw = word_offsets_i32[row_id];
  const int word_end_raw = word_offsets_i32[row_id + 1u];
  if (word_begin_raw < 0 || word_end_raw <= word_begin_raw ||
      uint(word_end_raw) > word_count) {
    return;
  }

  const float physical_length_epsilon = config_f32[0];
  const float minimum_absolute_cut_denominator = config_f32[1];
  const float minimum_ray_speed = config_f32[2];
  const float depth_closure_relative_tolerance = config_f32[3];
  const float active_tie_relative_tolerance = config_f32[4];
  const float minimum_cut_cosine = config_f32[5];
  const float minimum_coordinate_length = config_f32[6];
  if (!isfinite(physical_length_epsilon) ||
      !(physical_length_epsilon > 0.0f) ||
      !isfinite(minimum_absolute_cut_denominator) ||
      !(minimum_absolute_cut_denominator > 0.0f) ||
      !isfinite(minimum_ray_speed) || !(minimum_ray_speed > 0.0f) ||
      !isfinite(depth_closure_relative_tolerance) ||
      depth_closure_relative_tolerance < 0.0f ||
      !isfinite(active_tie_relative_tolerance) ||
      active_tie_relative_tolerance < 0.0f ||
      !isfinite(minimum_cut_cosine) || !(minimum_cut_cosine > 0.0f) ||
      minimum_cut_cosine > 1.0f ||
      !isfinite(minimum_coordinate_length) ||
      !(minimum_coordinate_length > 0.0f)) {
    return;
  }
  const float time = row_node_time_f32[row_id * node_count + node_id];
  const uint near_far_base = row_id * 2u;
  const float near_depth = row_near_far_f32[near_far_base + 0u];
  const float far_depth = row_near_far_f32[near_far_base + 1u];
  const uint ray_base = row_id * 12u;
  const float3 origin0 = float3(
      row_ray_coeff_f32[ray_base + 0u],
      row_ray_coeff_f32[ray_base + 1u],
      row_ray_coeff_f32[ray_base + 2u]);
  const float3 origin1 = float3(
      row_ray_coeff_f32[ray_base + 3u],
      row_ray_coeff_f32[ray_base + 4u],
      row_ray_coeff_f32[ray_base + 5u]);
  const float3 direction0 = float3(
      row_ray_coeff_f32[ray_base + 6u],
      row_ray_coeff_f32[ray_base + 7u],
      row_ray_coeff_f32[ray_base + 8u]);
  const float3 direction1 = float3(
      row_ray_coeff_f32[ray_base + 9u],
      row_ray_coeff_f32[ray_base + 10u],
      row_ray_coeff_f32[ray_base + 11u]);
  const float3 origin = origin0 + time * origin1;
  const float3 direction = direction0 + time * direction1;
  const float ray_speed = length(direction);
  if (!isfinite(time) || !isfinite(near_depth) || !isfinite(far_depth) ||
      !(far_depth > near_depth) || !all(isfinite(origin)) ||
      !all(isfinite(direction)) || !(ray_speed > minimum_ray_speed) ||
      !isfinite(ray_speed)) {
    return;
  }

  const uint chart_base = gid * 4u;
  const float kappa_total = node_chart_f32[chart_base + 0u];
  const float3 velocity_total = float3(
      node_chart_f32[chart_base + 1u],
      node_chart_f32[chart_base + 2u],
      node_chart_f32[chart_base + 3u]);
  const float4 grad_chart = float4(
      grad_node_chart_f32[chart_base + 0u],
      grad_node_chart_f32[chart_base + 1u],
      grad_node_chart_f32[chart_base + 2u],
      grad_node_chart_f32[chart_base + 3u]);
  if (!isfinite(kappa_total) || !all(isfinite(velocity_total)) ||
      !all(isfinite(grad_chart))) {
    return;
  }

  // Defensive per-(row,node) checks run before this thread's first atomic.
  // They verify near/far closure and adjacent power ties, but they are not a
  // global transaction: another thread may already have accumulated.  Launch
  // correctness therefore requires the host's sealed continuous-owner
  // certificate to bind owner words, geometry, times, and lengths to one
  // immutable compiler snapshot.  This local return is only a last guard.
  const uint node_length_base = node_id * word_count;
  float preflight_depth = near_depth;
  for (int cursor = word_begin_raw; cursor < word_end_raw; ++cursor) {
    const uint word_index = uint(cursor);
    const int owner_raw = word_owner_i32[word_index];
    if (owner_raw < 0 || uint(owner_raw) >= compact_site_count) {
      return;
    }
    const long global_owner_raw = source_site_ids_i64[uint(owner_raw)];
    if (global_owner_raw < 0l || global_owner_raw >= long(global_site_count)) {
      return;
    }
    const float physical_length =
        node_physical_length_f32[node_length_base + word_index];
    const uint rgba_base = uint(owner_raw) * 4u;
    const float4 rgba = float4(
        site_rgba_f32[rgba_base + 0u],
        site_rgba_f32[rgba_base + 1u],
        site_rgba_f32[rgba_base + 2u],
        site_rgba_f32[rgba_base + 3u]);
    const float optical_depth = rgba.w * physical_length;
    if (!(physical_length > physical_length_epsilon) ||
        !isfinite(physical_length) || !all(isfinite(rgba)) || rgba.w < 0.0f ||
        !isfinite(optical_depth)) {
      return;
    }
    const float coordinate_length = physical_length / ray_speed;
    if (!isfinite(coordinate_length) ||
        !(coordinate_length > minimum_coordinate_length)) {
      return;
    }
    preflight_depth += coordinate_length;
    if (cursor + 1 < word_end_raw) {
      const int right_owner_raw = word_owner_i32[word_index + 1u];
      if (right_owner_raw < 0 || uint(right_owner_raw) >= compact_site_count) {
        return;
      }
      const uint left_owner = uint(owner_raw);
      const uint right_owner = uint(right_owner_raw);
      const uint left_position_base = left_owner * 3u;
      const uint right_position_base = right_owner * 3u;
      const float3 left_position = float3(
          compact_positions0_f32[left_position_base + 0u],
          compact_positions0_f32[left_position_base + 1u],
          compact_positions0_f32[left_position_base + 2u]) +
          time * float3(
              compact_velocities_f32[left_position_base + 0u],
              compact_velocities_f32[left_position_base + 1u],
              compact_velocities_f32[left_position_base + 2u]);
      const float3 right_position = float3(
          compact_positions0_f32[right_position_base + 0u],
          compact_positions0_f32[right_position_base + 1u],
          compact_positions0_f32[right_position_base + 2u]) +
          time * float3(
              compact_velocities_f32[right_position_base + 0u],
              compact_velocities_f32[right_position_base + 1u],
              compact_velocities_f32[right_position_base + 2u]);
      float left_weight = 0.0f;
      float right_weight = 0.0f;
      float time_power = 1.0f;
      for (uint coefficient = 0u; coefficient < weight_coefficient_count;
           ++coefficient) {
        left_weight += compact_weight_coefficients_f32[
            left_owner * weight_coefficient_count + coefficient] * time_power;
        right_weight += compact_weight_coefficients_f32[
            right_owner * weight_coefficient_count + coefficient] * time_power;
        time_power *= time;
      }
      const float3 normal = 2.0f * (right_position - left_position);
      const float denominator = dot(normal, direction);
      const float denominator_scale = length(normal) * ray_speed;
      const float cut_cosine =
          denominator_scale > 0.0f ? fabs(denominator) / denominator_scale : 0.0f;
      const float intercept =
          dot(normal, origin) + dot(left_position, left_position) -
          dot(right_position, right_position) - left_weight + right_weight;
      const float tie_residual = fabs(intercept + preflight_depth * denominator);
      const float tie_scale = max(1.0f, fabs(intercept));
      if (!all(isfinite(left_position)) || !all(isfinite(right_position)) ||
          !isfinite(left_weight) || !isfinite(right_weight) ||
          !all(isfinite(normal)) || !isfinite(denominator) ||
          !(fabs(denominator) > minimum_absolute_cut_denominator) ||
          !isfinite(cut_cosine) || !(cut_cosine > minimum_cut_cosine) ||
          !isfinite(intercept) || !isfinite(tie_residual) ||
          tie_residual > active_tie_relative_tolerance * tie_scale) {
        return;
      }
    }
  }
  const float closure_scale = max(1.0f, max(fabs(near_depth), fabs(far_depth)));
  if (!isfinite(preflight_depth) ||
      fabs(preflight_depth - far_depth) >
          depth_closure_relative_tolerance * closure_scale) {
    return;
  }

  float phi = 1.0f;
  float phi_prime = 0.0f;
  float inverse_phi = 1.0f;
  float inverse_phi_prime = 0.0f;
  wf2_lie_phi_and_derivative(kappa_total, phi, phi_prime);
  wf2_lie_inverse_phi_and_derivative(
      kappa_total, inverse_phi, inverse_phi_prime);
  const float3 total_m = phi * velocity_total;
  const float3 bar_m = inverse_phi * grad_chart.yzw;
  const float bar_kappa_word =
      grad_chart.x + inverse_phi_prime * dot(total_m, grad_chart.yzw);

  float prefix_beta = 1.0f;
  float3 prefix_m = float3(0.0f);
  float cut_depth = near_depth;
  float previous_bar_ell = 0.0f;
  int previous_owner_raw = -1;
  for (int cursor = word_begin_raw; cursor < word_end_raw; ++cursor) {
    const uint word_index = uint(cursor);
    const int current_owner_raw = word_owner_i32[word_index];
    const uint current_owner = uint(current_owner_raw);
    const uint rgba_base = current_owner * 4u;
    const float physical_length =
        node_physical_length_f32[node_length_base + word_index];
    const float density = site_rgba_f32[rgba_base + 3u];
    const float optical_depth = density * physical_length;
    const float segment_beta = exp(-optical_depth);
    const float segment_alpha = -expm1(-optical_depth);
    const float3 rgb = float3(
        site_rgba_f32[rgba_base + 0u],
        site_rgba_f32[rgba_base + 1u],
        site_rgba_f32[rgba_base + 2u]);
    const float tau_bar =
        dot(bar_m, prefix_m + prefix_beta * rgb - total_m) + bar_kappa_word;
    const float current_bar_ell = density * tau_bar;
    wf2_atomic_add4(
        grad_site_rgba_f32,
        rgba_base,
        float4(
            prefix_beta * segment_alpha * bar_m,
            physical_length * tau_bar));

    if (previous_owner_raw >= 0) {
      const uint left_owner = uint(previous_owner_raw);
      const uint right_owner = current_owner;
      const float cut_bar = ray_speed * (previous_bar_ell - current_bar_ell);
      const uint left_position_base = left_owner * 3u;
      const uint right_position_base = right_owner * 3u;
      const float3 left_position = float3(
          compact_positions0_f32[left_position_base + 0u],
          compact_positions0_f32[left_position_base + 1u],
          compact_positions0_f32[left_position_base + 2u]) +
          time * float3(
              compact_velocities_f32[left_position_base + 0u],
              compact_velocities_f32[left_position_base + 1u],
              compact_velocities_f32[left_position_base + 2u]);
      const float3 right_position = float3(
          compact_positions0_f32[right_position_base + 0u],
          compact_positions0_f32[right_position_base + 1u],
          compact_positions0_f32[right_position_base + 2u]) +
          time * float3(
              compact_velocities_f32[right_position_base + 0u],
              compact_velocities_f32[right_position_base + 1u],
              compact_velocities_f32[right_position_base + 2u]);
      const float3 normal = 2.0f * (right_position - left_position);
      const float denominator = dot(normal, direction);
      const float implicit_bar = -cut_bar / denominator;
      const float3 point = origin + cut_depth * direction;
      const float3 left_position_bar =
          implicit_bar * 2.0f * (left_position - point);
      const float3 right_position_bar =
          implicit_bar * 2.0f * (point - right_position);
      const long left_global_raw = source_site_ids_i64[left_owner];
      const long right_global_raw = source_site_ids_i64[right_owner];
      const uint left_global = uint(left_global_raw);
      const uint right_global = uint(right_global_raw);
      wf2_atomic_add3(
          grad_global_positions0_f32,
          left_global * 3u,
          left_position_bar);
      wf2_atomic_add3(
          grad_global_positions0_f32,
          right_global * 3u,
          right_position_bar);
      wf2_atomic_add3(
          grad_global_velocities_f32,
          left_global * 3u,
          time * left_position_bar);
      wf2_atomic_add3(
          grad_global_velocities_f32,
          right_global * 3u,
          time * right_position_bar);
      float time_power = 1.0f;
      for (uint coefficient = 0u; coefficient < weight_coefficient_count;
           ++coefficient) {
        atomic_fetch_add_explicit(
            &grad_global_weight_coefficients_f32[
                left_global * weight_coefficient_count + coefficient],
            -implicit_bar * time_power,
            memory_order_relaxed);
        atomic_fetch_add_explicit(
            &grad_global_weight_coefficients_f32[
                right_global * weight_coefficient_count + coefficient],
            implicit_bar * time_power,
            memory_order_relaxed);
        time_power *= time;
      }
    }

    const float coordinate_length = physical_length / ray_speed;
    cut_depth += coordinate_length;
    previous_bar_ell = current_bar_ell;
    previous_owner_raw = current_owner_raw;
    prefix_m += prefix_beta * segment_alpha * rgb;
    prefix_beta *= segment_beta;
  }
}

// Union-v2 validates the exact factorization P_b = P_U Q_b before its first
// write. The three identities remain physically distinct: source ids prove
// compact-to-global provenance, compact_to_geometry_output selects a U-row,
// and geometry_output_source_site_ids proves that row names the same global
// site. The shared U ledger is scanned once per request; each compact material
// ledger is still scanned once per block.
kernel void wf2_kinetic_fused_union_full_vjp_validate_v2_tensor(
    device const int* word_offsets_i32 [[buffer(0)]],
    device const int* word_owner_i32 [[buffer(1)]],
    device const long* source_site_ids_i64 [[buffer(2)]],
    device const float* node_physical_length_f32 [[buffer(3)]],
    device const float* site_rgba_f32 [[buffer(4)]],
    device const float* node_chart_f32 [[buffer(5)]],
    device const float* row_node_time_f32 [[buffer(6)]],
    device const float* row_near_far_f32 [[buffer(7)]],
    device const float* row_ray_coeff_f32 [[buffer(8)]],
    device const float* compact_positions0_f32 [[buffer(9)]],
    device const float* compact_velocities_f32 [[buffer(10)]],
    device const float* compact_weight_coefficients_f32 [[buffer(11)]],
    device const float* grad_node_chart_f32 [[buffer(12)]],
    device const float* grad_site_rgba_f32 [[buffer(13)]],
    device const float* grad_union_positions0_f32 [[buffer(14)]],
    device const float* grad_union_velocities_f32 [[buffer(15)]],
    device const float* grad_union_weight_coefficients_f32 [[buffer(16)]],
    device const int* config_i32 [[buffer(17)]],
    device const float* config_f32 [[buffer(18)]],
    constant int& row_count_i32 [[buffer(19)]],
    constant int& node_count_i32 [[buffer(20)]],
    constant int& compact_site_count_i32 [[buffer(21)]],
    constant int& word_count_i32 [[buffer(22)]],
    constant int& weight_coefficient_count_i32 [[buffer(23)]],
    constant int& global_site_count_i32 [[buffer(24)]],
    device atomic_uint* validation_status_u32 [[buffer(25)]],
    constant int& validate_shared_union_ledgers_i32 [[buffer(26)]],
    device const long* compact_to_geometry_output_i64 [[buffer(27)]],
    device const long* geometry_output_source_site_ids_i64 [[buffer(28)]],
    constant int& union_site_count_i32 [[buffer(29)]],
    uint gid [[thread_position_in_grid]]) {
  uint reason = wf2_kinetic_fused_direct_full_vjp_validation_reason_v1(
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
      grad_node_chart_f32,
      config_i32,
      config_f32,
      uint(row_count_i32),
      uint(node_count_i32),
      uint(compact_site_count_i32),
      uint(word_count_i32),
      uint(weight_coefficient_count_i32),
      uint(global_site_count_i32),
      gid);
  const uint validation_thread_count =
      uint(row_count_i32) * uint(node_count_i32);
  if (config_i32[6] != union_site_count_i32 || union_site_count_i32 <= 0 ||
      union_site_count_i32 > global_site_count_i32) {
    reason |= WF2_KINETIC_FUSED_V2_REASON_INDEX_SPACE;
  }
  for (uint compact_index = gid;
       compact_index < uint(compact_site_count_i32);
       compact_index += validation_thread_count) {
    const long global_raw = source_site_ids_i64[compact_index];
    const long union_raw = compact_to_geometry_output_i64[compact_index];
    if (global_raw < 0l || global_raw >= long(global_site_count_i32) ||
        union_raw < 0l || union_raw >= long(union_site_count_i32) ||
        geometry_output_source_site_ids_i64[uint(union_raw)] != global_raw) {
      reason |= WF2_KINETIC_FUSED_V2_REASON_INDEX_SPACE;
    }
  }
  for (uint index = gid; index < uint(compact_site_count_i32) * 4u;
       index += validation_thread_count) {
    const float value = grad_site_rgba_f32[index];
    if (!isfinite(value) || value != 0.0f) {
      reason |= WF2_KINETIC_FUSED_V1_REASON_OUTPUT_LEDGER;
    }
  }
  if (validate_shared_union_ledgers_i32 != 0) {
    for (uint union_index = gid; union_index < uint(union_site_count_i32);
         union_index += validation_thread_count) {
      const long global_raw = geometry_output_source_site_ids_i64[union_index];
      if (global_raw < 0l || global_raw >= long(global_site_count_i32) ||
          (union_index > 0u &&
           geometry_output_source_site_ids_i64[union_index - 1u] >= global_raw)) {
        reason |= WF2_KINETIC_FUSED_V2_REASON_INDEX_SPACE;
      }
    }
    for (uint index = gid; index < uint(union_site_count_i32) * 3u;
         index += validation_thread_count) {
      const float position_value = grad_union_positions0_f32[index];
      const float velocity_value = grad_union_velocities_f32[index];
      if (!isfinite(position_value) || position_value != 0.0f ||
          !isfinite(velocity_value) || velocity_value != 0.0f) {
        reason |= WF2_KINETIC_FUSED_V1_REASON_OUTPUT_LEDGER;
      }
    }
    for (uint index = gid;
         index < uint(union_site_count_i32) *
                     uint(weight_coefficient_count_i32);
         index += validation_thread_count) {
      const float value = grad_union_weight_coefficients_f32[index];
      if (!isfinite(value) || value != 0.0f) {
        reason |= WF2_KINETIC_FUSED_V1_REASON_OUTPUT_LEDGER;
      }
    }
  }
  if (reason != 0u) {
    atomic_fetch_or_explicit(
        validation_status_u32, reason, memory_order_relaxed);
  }
}

kernel void wf2_kinetic_fused_union_full_vjp_finalize_v2_tensor(
    device const float* grad_site_rgba_f32 [[buffer(0)]],
    device const float* grad_union_positions0_f32 [[buffer(1)]],
    device const float* grad_union_velocities_f32 [[buffer(2)]],
    device const float* grad_union_weight_coefficients_f32 [[buffer(3)]],
    device atomic_uint* validation_status_u32 [[buffer(4)]],
    constant int& compact_site_count_i32 [[buffer(5)]],
    constant int& union_site_count_i32 [[buffer(6)]],
    constant int& weight_coefficient_count_i32 [[buffer(7)]],
    constant int& finalize_shared_union_ledgers_i32 [[buffer(8)]],
    uint gid [[thread_position_in_grid]]) {
  uint reason = 0u;
  if (gid < uint(compact_site_count_i32) * 4u &&
      !isfinite(grad_site_rgba_f32[gid])) {
    reason |= WF2_KINETIC_FUSED_V1_REASON_OUTPUT_LEDGER;
  }
  if (finalize_shared_union_ledgers_i32 != 0) {
    if (gid < uint(union_site_count_i32) * 3u &&
        (!isfinite(grad_union_positions0_f32[gid]) ||
         !isfinite(grad_union_velocities_f32[gid]))) {
      reason |= WF2_KINETIC_FUSED_V1_REASON_OUTPUT_LEDGER;
    }
    if (gid < uint(union_site_count_i32) *
                  uint(weight_coefficient_count_i32) &&
        !isfinite(grad_union_weight_coefficients_f32[gid])) {
      reason |= WF2_KINETIC_FUSED_V1_REASON_OUTPUT_LEDGER;
    }
  }
  if (reason != 0u) {
    atomic_fetch_or_explicit(
        validation_status_u32, reason, memory_order_relaxed);
  }
}

kernel void wf2_kinetic_fused_union_full_vjp_v2_tensor(
    device const int* word_offsets_i32 [[buffer(0)]],
    device const int* word_owner_i32 [[buffer(1)]],
    device const long* source_site_ids_i64 [[buffer(2)]],
    device const float* node_physical_length_f32 [[buffer(3)]],
    device const float* site_rgba_f32 [[buffer(4)]],
    device const float* node_chart_f32 [[buffer(5)]],
    device const float* row_node_time_f32 [[buffer(6)]],
    device const float* row_near_far_f32 [[buffer(7)]],
    device const float* row_ray_coeff_f32 [[buffer(8)]],
    device const float* compact_positions0_f32 [[buffer(9)]],
    device const float* compact_velocities_f32 [[buffer(10)]],
    device const float* compact_weight_coefficients_f32 [[buffer(11)]],
    device const float* grad_node_chart_f32 [[buffer(12)]],
    device atomic_float* grad_site_rgba_f32 [[buffer(13)]],
    device atomic_float* grad_union_positions0_f32 [[buffer(14)]],
    device atomic_float* grad_union_velocities_f32 [[buffer(15)]],
    device atomic_float* grad_union_weight_coefficients_f32 [[buffer(16)]],
    device const int* config_i32 [[buffer(17)]],
    device const float* config_f32 [[buffer(18)]],
    constant int& row_count_i32 [[buffer(19)]],
    constant int& node_count_i32 [[buffer(20)]],
    constant int& compact_site_count_i32 [[buffer(21)]],
    constant int& word_count_i32 [[buffer(22)]],
    constant int& weight_coefficient_count_i32 [[buffer(23)]],
    constant int& global_site_count_i32 [[buffer(24)]],
    device atomic_uint* validation_status_u32 [[buffer(25)]],
    device const long* compact_to_geometry_output_i64 [[buffer(26)]],
    device const long* geometry_output_source_site_ids_i64 [[buffer(27)]],
    constant int& union_site_count_i32 [[buffer(28)]],
    uint gid [[thread_position_in_grid]]) {
  // ObjC++ enqueues the complete validation grid before this grid on the same
  // serialized MPS stream.  The shared scalar is therefore complete and visible
  // before any thread reaches this device-side write gate.
  if (atomic_load_explicit(validation_status_u32, memory_order_relaxed) != 0u) {
    return;
  }
  // Counts come from host-validated tensor shapes rather than device config
  // contents, so a malformed direct torch.ops call cannot enlarge an index
  // domain before the kernel's per-element guards run. config_i32 remains the
  // sealed prepared-token mirror and is intentionally not trusted for bounds.
  const uint row_count = uint(row_count_i32);
  const uint node_count = uint(node_count_i32);
  const uint compact_site_count = uint(compact_site_count_i32);
  const uint word_count = uint(word_count_i32);
  const uint weight_coefficient_count = uint(weight_coefficient_count_i32);
  const uint global_site_count = uint(global_site_count_i32);
  if (config_i32[0] != row_count_i32 || config_i32[1] != node_count_i32 ||
      config_i32[2] != compact_site_count_i32 ||
      config_i32[3] != word_count_i32 ||
      config_i32[4] != weight_coefficient_count_i32 ||
      config_i32[5] != global_site_count_i32 ||
      config_i32[6] != union_site_count_i32) {
    return;
  }
  if (gid >= row_count * node_count || weight_coefficient_count < 1u ||
      weight_coefficient_count > 3u) {
    return;
  }

  const uint row_id = gid / node_count;
  const uint node_id = gid - row_id * node_count;
  const int word_begin_raw = word_offsets_i32[row_id];
  const int word_end_raw = word_offsets_i32[row_id + 1u];
  if (word_begin_raw < 0 || word_end_raw <= word_begin_raw ||
      uint(word_end_raw) > word_count) {
    return;
  }

  const float physical_length_epsilon = config_f32[0];
  const float minimum_absolute_cut_denominator = config_f32[1];
  const float minimum_ray_speed = config_f32[2];
  const float depth_closure_relative_tolerance = config_f32[3];
  const float active_tie_relative_tolerance = config_f32[4];
  const float minimum_cut_cosine = config_f32[5];
  const float minimum_coordinate_length = config_f32[6];
  if (!isfinite(physical_length_epsilon) ||
      !(physical_length_epsilon > 0.0f) ||
      !isfinite(minimum_absolute_cut_denominator) ||
      !(minimum_absolute_cut_denominator > 0.0f) ||
      !isfinite(minimum_ray_speed) || !(minimum_ray_speed > 0.0f) ||
      !isfinite(depth_closure_relative_tolerance) ||
      depth_closure_relative_tolerance < 0.0f ||
      !isfinite(active_tie_relative_tolerance) ||
      active_tie_relative_tolerance < 0.0f ||
      !isfinite(minimum_cut_cosine) || !(minimum_cut_cosine > 0.0f) ||
      minimum_cut_cosine > 1.0f ||
      !isfinite(minimum_coordinate_length) ||
      !(minimum_coordinate_length > 0.0f)) {
    return;
  }
  const float time = row_node_time_f32[row_id * node_count + node_id];
  const uint near_far_base = row_id * 2u;
  const float near_depth = row_near_far_f32[near_far_base + 0u];
  const float far_depth = row_near_far_f32[near_far_base + 1u];
  const uint ray_base = row_id * 12u;
  const float3 origin0 = float3(
      row_ray_coeff_f32[ray_base + 0u],
      row_ray_coeff_f32[ray_base + 1u],
      row_ray_coeff_f32[ray_base + 2u]);
  const float3 origin1 = float3(
      row_ray_coeff_f32[ray_base + 3u],
      row_ray_coeff_f32[ray_base + 4u],
      row_ray_coeff_f32[ray_base + 5u]);
  const float3 direction0 = float3(
      row_ray_coeff_f32[ray_base + 6u],
      row_ray_coeff_f32[ray_base + 7u],
      row_ray_coeff_f32[ray_base + 8u]);
  const float3 direction1 = float3(
      row_ray_coeff_f32[ray_base + 9u],
      row_ray_coeff_f32[ray_base + 10u],
      row_ray_coeff_f32[ray_base + 11u]);
  const float3 origin = origin0 + time * origin1;
  const float3 direction = direction0 + time * direction1;
  const float ray_speed = length(direction);
  if (!isfinite(time) || !isfinite(near_depth) || !isfinite(far_depth) ||
      !(far_depth > near_depth) || !all(isfinite(origin)) ||
      !all(isfinite(direction)) || !(ray_speed > minimum_ray_speed) ||
      !isfinite(ray_speed)) {
    return;
  }

  const uint chart_base = gid * 4u;
  const float kappa_total = node_chart_f32[chart_base + 0u];
  const float3 velocity_total = float3(
      node_chart_f32[chart_base + 1u],
      node_chart_f32[chart_base + 2u],
      node_chart_f32[chart_base + 3u]);
  const float4 grad_chart = float4(
      grad_node_chart_f32[chart_base + 0u],
      grad_node_chart_f32[chart_base + 1u],
      grad_node_chart_f32[chart_base + 2u],
      grad_node_chart_f32[chart_base + 3u]);
  if (!isfinite(kappa_total) || !all(isfinite(velocity_total)) ||
      !all(isfinite(grad_chart))) {
    return;
  }

  // Defensive per-(row,node) checks run before this thread's first atomic.
  // They verify near/far closure and adjacent power ties, but they are not a
  // global transaction: another thread may already have accumulated.  Launch
  // correctness therefore requires the host's sealed continuous-owner
  // certificate to bind owner words, geometry, times, and lengths to one
  // immutable compiler snapshot.  This local return is only a last guard.
  const uint node_length_base = node_id * word_count;
  float preflight_depth = near_depth;
  for (int cursor = word_begin_raw; cursor < word_end_raw; ++cursor) {
    const uint word_index = uint(cursor);
    const int owner_raw = word_owner_i32[word_index];
    if (owner_raw < 0 || uint(owner_raw) >= compact_site_count) {
      return;
    }
    const long global_owner_raw = source_site_ids_i64[uint(owner_raw)];
    if (global_owner_raw < 0l || global_owner_raw >= long(global_site_count)) {
      return;
    }
    const float physical_length =
        node_physical_length_f32[node_length_base + word_index];
    const uint rgba_base = uint(owner_raw) * 4u;
    const float4 rgba = float4(
        site_rgba_f32[rgba_base + 0u],
        site_rgba_f32[rgba_base + 1u],
        site_rgba_f32[rgba_base + 2u],
        site_rgba_f32[rgba_base + 3u]);
    const float optical_depth = rgba.w * physical_length;
    if (!(physical_length > physical_length_epsilon) ||
        !isfinite(physical_length) || !all(isfinite(rgba)) || rgba.w < 0.0f ||
        !isfinite(optical_depth)) {
      return;
    }
    const float coordinate_length = physical_length / ray_speed;
    if (!isfinite(coordinate_length) ||
        !(coordinate_length > minimum_coordinate_length)) {
      return;
    }
    preflight_depth += coordinate_length;
    if (cursor + 1 < word_end_raw) {
      const int right_owner_raw = word_owner_i32[word_index + 1u];
      if (right_owner_raw < 0 || uint(right_owner_raw) >= compact_site_count) {
        return;
      }
      const uint left_owner = uint(owner_raw);
      const uint right_owner = uint(right_owner_raw);
      const uint left_position_base = left_owner * 3u;
      const uint right_position_base = right_owner * 3u;
      const float3 left_position = float3(
          compact_positions0_f32[left_position_base + 0u],
          compact_positions0_f32[left_position_base + 1u],
          compact_positions0_f32[left_position_base + 2u]) +
          time * float3(
              compact_velocities_f32[left_position_base + 0u],
              compact_velocities_f32[left_position_base + 1u],
              compact_velocities_f32[left_position_base + 2u]);
      const float3 right_position = float3(
          compact_positions0_f32[right_position_base + 0u],
          compact_positions0_f32[right_position_base + 1u],
          compact_positions0_f32[right_position_base + 2u]) +
          time * float3(
              compact_velocities_f32[right_position_base + 0u],
              compact_velocities_f32[right_position_base + 1u],
              compact_velocities_f32[right_position_base + 2u]);
      float left_weight = 0.0f;
      float right_weight = 0.0f;
      float time_power = 1.0f;
      for (uint coefficient = 0u; coefficient < weight_coefficient_count;
           ++coefficient) {
        left_weight += compact_weight_coefficients_f32[
            left_owner * weight_coefficient_count + coefficient] * time_power;
        right_weight += compact_weight_coefficients_f32[
            right_owner * weight_coefficient_count + coefficient] * time_power;
        time_power *= time;
      }
      const float3 normal = 2.0f * (right_position - left_position);
      const float denominator = dot(normal, direction);
      const float denominator_scale = length(normal) * ray_speed;
      const float cut_cosine =
          denominator_scale > 0.0f ? fabs(denominator) / denominator_scale : 0.0f;
      const float intercept =
          dot(normal, origin) + dot(left_position, left_position) -
          dot(right_position, right_position) - left_weight + right_weight;
      const float tie_residual = fabs(intercept + preflight_depth * denominator);
      const float tie_scale = max(1.0f, fabs(intercept));
      if (!all(isfinite(left_position)) || !all(isfinite(right_position)) ||
          !isfinite(left_weight) || !isfinite(right_weight) ||
          !all(isfinite(normal)) || !isfinite(denominator) ||
          !(fabs(denominator) > minimum_absolute_cut_denominator) ||
          !isfinite(cut_cosine) || !(cut_cosine > minimum_cut_cosine) ||
          !isfinite(intercept) || !isfinite(tie_residual) ||
          tie_residual > active_tie_relative_tolerance * tie_scale) {
        return;
      }
    }
  }
  const float closure_scale = max(1.0f, max(fabs(near_depth), fabs(far_depth)));
  if (!isfinite(preflight_depth) ||
      fabs(preflight_depth - far_depth) >
          depth_closure_relative_tolerance * closure_scale) {
    return;
  }

  float phi = 1.0f;
  float phi_prime = 0.0f;
  float inverse_phi = 1.0f;
  float inverse_phi_prime = 0.0f;
  wf2_lie_phi_and_derivative(kappa_total, phi, phi_prime);
  wf2_lie_inverse_phi_and_derivative(
      kappa_total, inverse_phi, inverse_phi_prime);
  const float3 total_m = phi * velocity_total;
  const float3 bar_m = inverse_phi * grad_chart.yzw;
  const float bar_kappa_word =
      grad_chart.x + inverse_phi_prime * dot(total_m, grad_chart.yzw);

  float prefix_beta = 1.0f;
  float3 prefix_m = float3(0.0f);
  float cut_depth = near_depth;
  float previous_bar_ell = 0.0f;
  int previous_owner_raw = -1;
  for (int cursor = word_begin_raw; cursor < word_end_raw; ++cursor) {
    const uint word_index = uint(cursor);
    const int current_owner_raw = word_owner_i32[word_index];
    const uint current_owner = uint(current_owner_raw);
    const uint rgba_base = current_owner * 4u;
    const float physical_length =
        node_physical_length_f32[node_length_base + word_index];
    const float density = site_rgba_f32[rgba_base + 3u];
    const float optical_depth = density * physical_length;
    const float segment_beta = exp(-optical_depth);
    const float segment_alpha = -expm1(-optical_depth);
    const float3 rgb = float3(
        site_rgba_f32[rgba_base + 0u],
        site_rgba_f32[rgba_base + 1u],
        site_rgba_f32[rgba_base + 2u]);
    const float tau_bar =
        dot(bar_m, prefix_m + prefix_beta * rgb - total_m) + bar_kappa_word;
    const float current_bar_ell = density * tau_bar;
    wf2_atomic_add4(
        grad_site_rgba_f32,
        rgba_base,
        float4(
            prefix_beta * segment_alpha * bar_m,
            physical_length * tau_bar));

    if (previous_owner_raw >= 0) {
      const uint left_owner = uint(previous_owner_raw);
      const uint right_owner = current_owner;
      const float cut_bar = ray_speed * (previous_bar_ell - current_bar_ell);
      const uint left_position_base = left_owner * 3u;
      const uint right_position_base = right_owner * 3u;
      const float3 left_position = float3(
          compact_positions0_f32[left_position_base + 0u],
          compact_positions0_f32[left_position_base + 1u],
          compact_positions0_f32[left_position_base + 2u]) +
          time * float3(
              compact_velocities_f32[left_position_base + 0u],
              compact_velocities_f32[left_position_base + 1u],
              compact_velocities_f32[left_position_base + 2u]);
      const float3 right_position = float3(
          compact_positions0_f32[right_position_base + 0u],
          compact_positions0_f32[right_position_base + 1u],
          compact_positions0_f32[right_position_base + 2u]) +
          time * float3(
              compact_velocities_f32[right_position_base + 0u],
              compact_velocities_f32[right_position_base + 1u],
              compact_velocities_f32[right_position_base + 2u]);
      const float3 normal = 2.0f * (right_position - left_position);
      const float denominator = dot(normal, direction);
      const float implicit_bar = -cut_bar / denominator;
      const float3 point = origin + cut_depth * direction;
      const float3 left_position_bar =
          implicit_bar * 2.0f * (left_position - point);
      const float3 right_position_bar =
          implicit_bar * 2.0f * (point - right_position);
      const long left_global_raw = source_site_ids_i64[left_owner];
      const long right_global_raw = source_site_ids_i64[right_owner];
      const long left_union_raw = compact_to_geometry_output_i64[left_owner];
      const long right_union_raw = compact_to_geometry_output_i64[right_owner];
      if (left_union_raw < 0l || left_union_raw >= long(union_site_count_i32) ||
          right_union_raw < 0l || right_union_raw >= long(union_site_count_i32) ||
          geometry_output_source_site_ids_i64[uint(left_union_raw)] != left_global_raw ||
          geometry_output_source_site_ids_i64[uint(right_union_raw)] != right_global_raw) {
        return;
      }
      const uint left_union = uint(left_union_raw);
      const uint right_union = uint(right_union_raw);
      wf2_atomic_add3(
          grad_union_positions0_f32,
          left_union * 3u,
          left_position_bar);
      wf2_atomic_add3(
          grad_union_positions0_f32,
          right_union * 3u,
          right_position_bar);
      wf2_atomic_add3(
          grad_union_velocities_f32,
          left_union * 3u,
          time * left_position_bar);
      wf2_atomic_add3(
          grad_union_velocities_f32,
          right_union * 3u,
          time * right_position_bar);
      float time_power = 1.0f;
      for (uint coefficient = 0u; coefficient < weight_coefficient_count;
           ++coefficient) {
        atomic_fetch_add_explicit(
            &grad_union_weight_coefficients_f32[
                left_union * weight_coefficient_count + coefficient],
            -implicit_bar * time_power,
            memory_order_relaxed);
        atomic_fetch_add_explicit(
            &grad_union_weight_coefficients_f32[
                right_union * weight_coefficient_count + coefficient],
            implicit_bar * time_power,
            memory_order_relaxed);
        time_power *= time;
      }
    }

    const float coordinate_length = physical_length / ray_speed;
    cut_depth += coordinate_length;
    previous_bar_ell = current_bar_ell;
    previous_owner_raw = current_owner_raw;
    prefix_m += prefix_beta * segment_alpha * rgb;
    prefix_beta *= segment_beta;
  }
}



kernel void wf2_clear_affine_loss_site_rgba_grad_tensor(
    device float* loss_f32 [[buffer(0)]],
    device float* grad_site_rgba_f32 [[buffer(1)]],
    device const int* config_i32 [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
  const uint site_count = uint(config_i32[2]);
  if (gid == 0u) {
    loss_f32[0] = 0.0f;
  }
  if (gid >= site_count) {
    return;
  }
  const uint base = gid * 4u;
  grad_site_rgba_f32[base + 0u] = 0.0f;
  grad_site_rgba_f32[base + 1u] = 0.0f;
  grad_site_rgba_f32[base + 2u] = 0.0f;
  grad_site_rgba_f32[base + 3u] = 0.0f;
}

inline float wf2_continuous_depth_mass(
    const float start_depth,
    const float length,
    const float density,
    const float segment_transmittance,
    const float segment_alpha) {
  if (!(density > 1.0e-6f)) {
    return density * (start_depth * length + 0.5f * length * length);
  }
  return start_depth * segment_alpha + segment_alpha / density - length * segment_transmittance;
}

inline float wf2_continuous_depth_mass_grad_density(
    const float start_depth,
    const float length,
    const float density,
    const float segment_transmittance,
    const float segment_alpha) {
  if (!(density > 1.0e-6f)) {
    return start_depth * length + 0.5f * length * length;
  }
  const float density_sq = density * density;
  return start_depth * length * segment_transmittance +
      (density * length * segment_transmittance - segment_alpha) / density_sq +
      length * length * segment_transmittance;
}

kernel void wf2_segment_tape_rgba_depth_replay_tensor(
    device const int* segment_offsets_i32 [[buffer(0)]],
    device const int* segment_owner_i32 [[buffer(1)]],
    device const float* segment_length_f32 [[buffer(2)]],
    device const float* segment_mid_f32 [[buffer(3)]],
    device const float* site_rgba_f32 [[buffer(4)]],
    device const int* config_i32 [[buffer(5)]],
    device const float* config_f32 [[buffer(6)]],
    device float* output_rgb_f32 [[buffer(7)]],
    device float* output_alpha_f32 [[buffer(8)]],
    device float* output_depth_f32 [[buffer(9)]],
    uint sample_id [[thread_position_in_grid]]) {
  const uint track_count = uint(config_i32[0]);
  const uint frame_count = uint(config_i32[1]);
  const uint site_count = uint(config_i32[2]);
  const uint segment_count = uint(config_i32[3]);
  const uint total_samples = track_count * frame_count;
  if (sample_id >= total_samples) {
    return;
  }

  const float far_depth = config_f32[0];
  const float transmittance_threshold = config_f32[1];
  const int begin_raw = segment_offsets_i32[sample_id];
  const int end_raw = segment_offsets_i32[sample_id + 1u];
  if (begin_raw < 0 || end_raw < begin_raw || uint(end_raw) > segment_count) {
    return;
  }

  float3 rgb_accum = float3(0.0f, 0.0f, 0.0f);
  float alpha_accum = 0.0f;
  float depth_weighted = 0.0f;
  float transmittance = 1.0f;
  for (uint cursor = uint(begin_raw); cursor < uint(end_raw); ++cursor) {
    if (transmittance <= transmittance_threshold) {
      break;
    }
    const int owner_raw = segment_owner_i32[cursor];
    if (owner_raw < 0 || uint(owner_raw) >= site_count) {
      continue;
    }
    const uint owner = uint(owner_raw);
    const float length = segment_length_f32[cursor];
    if (!(length > 1.0e-8f)) {
      continue;
    }
    const float mid_depth = segment_mid_f32[cursor];
    const uint rgba_base = owner * 4u;
    const float density = max(site_rgba_f32[rgba_base + 3u], 0.0f);
    const float segment_transmittance = exp(-density * length);
    const float segment_alpha = 1.0f - segment_transmittance;
    const float weight = transmittance * segment_alpha;
    const float3 rgb = float3(
        site_rgba_f32[rgba_base + 0u],
        site_rgba_f32[rgba_base + 1u],
        site_rgba_f32[rgba_base + 2u]);
    rgb_accum += weight * rgb;
    alpha_accum += weight;
    depth_weighted += weight * mid_depth;
    transmittance *= segment_transmittance;
  }

  const uint out_base = sample_id * 3u;
  output_rgb_f32[out_base + 0u] = rgb_accum.x;
  output_rgb_f32[out_base + 1u] = rgb_accum.y;
  output_rgb_f32[out_base + 2u] = rgb_accum.z;
  output_alpha_f32[sample_id] = alpha_accum;
  output_depth_f32[sample_id] = alpha_accum > 1.0e-8f ? depth_weighted / alpha_accum : far_depth;
}

kernel void wf2_segment_tape_vjp_direct_atomic_grad_only_tensor(
    device const int* segment_offsets_i32 [[buffer(0)]],
    device const int* segment_owner_i32 [[buffer(1)]],
    device const float* segment_length_f32 [[buffer(2)]],
    device const float* segment_mid_f32 [[buffer(3)]],
    device const float* site_rgba_f32 [[buffer(4)]],
    device const float* grad_rgb_f32 [[buffer(5)]],
    device const float* grad_alpha_f32 [[buffer(6)]],
    device const float* grad_depth_f32 [[buffer(7)]],
    device const int* config_i32 [[buffer(8)]],
    device const float* config_f32 [[buffer(9)]],
    device atomic_float* grad_site_rgba_f32 [[buffer(10)]],
    uint sample_id [[thread_position_in_grid]]) {
  const uint track_count = uint(config_i32[0]);
  const uint frame_count = uint(config_i32[1]);
  const uint site_count = uint(config_i32[2]);
  const uint segment_count = uint(config_i32[3]);
  const uint total_samples = track_count * frame_count;
  if (sample_id >= total_samples) {
    return;
  }

  const float transmittance_threshold = config_f32[1];
  const int begin_raw = segment_offsets_i32[sample_id];
  const int end_raw = segment_offsets_i32[sample_id + 1u];
  if (begin_raw < 0 || end_raw < begin_raw || uint(end_raw) > segment_count) {
    return;
  }

  uint owners[WF2_MAX_REALRAY_SEGMENTS];
  float lengths[WF2_MAX_REALRAY_SEGMENTS];
  float mids[WF2_MAX_REALRAY_SEGMENTS];
  float trans_before[WF2_MAX_REALRAY_SEGMENTS];
  float segment_trans[WF2_MAX_REALRAY_SEGMENTS];
  float segment_alpha[WF2_MAX_REALRAY_SEGMENTS];
  float weights[WF2_MAX_REALRAY_SEGMENTS];
  float3 segment_rgb[WF2_MAX_REALRAY_SEGMENTS];

  float alpha_accum = 0.0f;
  float depth_weighted = 0.0f;
  float transmittance = 1.0f;
  uint local_segment_count = 0u;
  for (uint cursor = uint(begin_raw); cursor < uint(end_raw); ++cursor) {
    if (transmittance <= transmittance_threshold || local_segment_count >= WF2_MAX_REALRAY_SEGMENTS) {
      break;
    }
    const int owner_raw = segment_owner_i32[cursor];
    if (owner_raw < 0 || uint(owner_raw) >= site_count) {
      continue;
    }
    const float length = segment_length_f32[cursor];
    if (!(length > 1.0e-8f)) {
      continue;
    }
    const uint owner = uint(owner_raw);
    const float mid_depth = segment_mid_f32[cursor];
    const uint rgba_base = owner * 4u;
    const float density = max(site_rgba_f32[rgba_base + 3u], 0.0f);
    const float seg_trans = exp(-density * length);
    const float seg_alpha = 1.0f - seg_trans;
    const float weight = transmittance * seg_alpha;
    const float3 rgb = float3(
        site_rgba_f32[rgba_base + 0u],
        site_rgba_f32[rgba_base + 1u],
        site_rgba_f32[rgba_base + 2u]);

    owners[local_segment_count] = owner;
    lengths[local_segment_count] = length;
    mids[local_segment_count] = mid_depth;
    trans_before[local_segment_count] = transmittance;
    segment_trans[local_segment_count] = seg_trans;
    segment_alpha[local_segment_count] = seg_alpha;
    weights[local_segment_count] = weight;
    segment_rgb[local_segment_count] = rgb;
    local_segment_count += 1u;

    alpha_accum += weight;
    depth_weighted += weight * mid_depth;
    transmittance *= seg_trans;
  }

  const uint out_base = sample_id * 3u;
  const float3 grad_rgb = float3(
      grad_rgb_f32[out_base + 0u],
      grad_rgb_f32[out_base + 1u],
      grad_rgb_f32[out_base + 2u]);
  const float grad_alpha = grad_alpha_f32[sample_id];
  const float grad_depth = grad_depth_f32[sample_id];
  float adj_next_transmittance = 0.0f;
  for (int segment_id = int(local_segment_count) - 1; segment_id >= 0; --segment_id) {
    const uint owner = owners[segment_id];
    float d_loss_d_weight = dot(grad_rgb, segment_rgb[segment_id]) + grad_alpha;
    if (alpha_accum > 1.0e-8f) {
      d_loss_d_weight += grad_depth *
          (mids[segment_id] * alpha_accum - depth_weighted) /
          (alpha_accum * alpha_accum);
    }

    const float adj_trans_before =
        d_loss_d_weight * segment_alpha[segment_id] +
        adj_next_transmittance * segment_trans[segment_id];
    const float adj_segment_alpha = d_loss_d_weight * trans_before[segment_id];
    const float adj_segment_trans =
        adj_next_transmittance * trans_before[segment_id] - adj_segment_alpha;
    float4 grad_rgba = float4(
        weights[segment_id] * grad_rgb.x,
        weights[segment_id] * grad_rgb.y,
        weights[segment_id] * grad_rgb.z,
        0.0f);
    const uint rgba_base = owner * 4u;
    const float raw_density = site_rgba_f32[rgba_base + 3u];
    if (raw_density > 0.0f) {
      grad_rgba.w = adj_segment_trans * (-lengths[segment_id] * segment_trans[segment_id]);
    }
    wf2_atomic_add4(grad_site_rgba_f32, rgba_base, grad_rgba);
    adj_next_transmittance = adj_trans_before;
  }
}

kernel void wf2_segment_tape_vjp_direct_atomic_track_tensor(
    device const int* segment_offsets_i32 [[buffer(0)]],
    device const int* segment_owner_i32 [[buffer(1)]],
    device const float* segment_length_f32 [[buffer(2)]],
    device const float* segment_mid_f32 [[buffer(3)]],
    device const float* site_rgba_f32 [[buffer(4)]],
    device const float* grad_rgb_f32 [[buffer(5)]],
    device const float* grad_alpha_f32 [[buffer(6)]],
    device const float* grad_depth_f32 [[buffer(7)]],
    device const int* config_i32 [[buffer(8)]],
    device const float* config_f32 [[buffer(9)]],
    device atomic_float* grad_site_rgba_f32 [[buffer(10)]],
    uint track_id [[thread_position_in_grid]]) {
  const uint track_count = uint(config_i32[0]);
  const uint frame_count = uint(config_i32[1]);
  const uint site_count = uint(config_i32[2]);
  const uint segment_count = uint(config_i32[3]);
  if (track_id >= track_count) {
    return;
  }

  const uint clamped_site_count = min(site_count, WF2_MAX_REALRAY_SITES);
  const float transmittance_threshold = config_f32[1];
  float4 grad_accum[WF2_MAX_REALRAY_SITES];
  for (uint site_id = 0u; site_id < WF2_MAX_REALRAY_SITES; ++site_id) {
    grad_accum[site_id] = float4(0.0f, 0.0f, 0.0f, 0.0f);
  }

  for (uint frame_id = 0u; frame_id < frame_count; ++frame_id) {
    const uint sample_id = track_id * frame_count + frame_id;
    const int begin_raw = segment_offsets_i32[sample_id];
    const int end_raw = segment_offsets_i32[sample_id + 1u];
    if (begin_raw < 0 || end_raw < begin_raw || uint(end_raw) > segment_count) {
      continue;
    }

    uint owners[WF2_MAX_REALRAY_SEGMENTS];
    float lengths[WF2_MAX_REALRAY_SEGMENTS];
    float mids[WF2_MAX_REALRAY_SEGMENTS];
    float trans_before[WF2_MAX_REALRAY_SEGMENTS];
    float segment_trans[WF2_MAX_REALRAY_SEGMENTS];
    float segment_alpha[WF2_MAX_REALRAY_SEGMENTS];
    float weights[WF2_MAX_REALRAY_SEGMENTS];
    float3 segment_rgb[WF2_MAX_REALRAY_SEGMENTS];

    float alpha_accum = 0.0f;
    float depth_weighted = 0.0f;
    float transmittance = 1.0f;
    uint local_segment_count = 0u;
    for (uint cursor = uint(begin_raw); cursor < uint(end_raw); ++cursor) {
      if (transmittance <= transmittance_threshold || local_segment_count >= WF2_MAX_REALRAY_SEGMENTS) {
        break;
      }
      const int owner_raw = segment_owner_i32[cursor];
      if (owner_raw < 0 || uint(owner_raw) >= clamped_site_count) {
        continue;
      }
      const float length = segment_length_f32[cursor];
      if (!(length > 1.0e-8f)) {
        continue;
      }
      const uint owner = uint(owner_raw);
      const float mid_depth = segment_mid_f32[cursor];
      const uint rgba_base = owner * 4u;
      const float density = max(site_rgba_f32[rgba_base + 3u], 0.0f);
      const float seg_trans = exp(-density * length);
      const float seg_alpha = 1.0f - seg_trans;
      const float weight = transmittance * seg_alpha;
      const float3 rgb = float3(
          site_rgba_f32[rgba_base + 0u],
          site_rgba_f32[rgba_base + 1u],
          site_rgba_f32[rgba_base + 2u]);

      owners[local_segment_count] = owner;
      lengths[local_segment_count] = length;
      mids[local_segment_count] = mid_depth;
      trans_before[local_segment_count] = transmittance;
      segment_trans[local_segment_count] = seg_trans;
      segment_alpha[local_segment_count] = seg_alpha;
      weights[local_segment_count] = weight;
      segment_rgb[local_segment_count] = rgb;
      local_segment_count += 1u;

      alpha_accum += weight;
      depth_weighted += weight * mid_depth;
      transmittance *= seg_trans;
    }

    const uint out_base = sample_id * 3u;
    const float3 grad_rgb = float3(
        grad_rgb_f32[out_base + 0u],
        grad_rgb_f32[out_base + 1u],
        grad_rgb_f32[out_base + 2u]);
    const float grad_alpha = grad_alpha_f32[sample_id];
    const float grad_depth = grad_depth_f32[sample_id];
    float adj_next_transmittance = 0.0f;
    for (int segment_id = int(local_segment_count) - 1; segment_id >= 0; --segment_id) {
      const uint owner = owners[segment_id];
      float d_loss_d_weight = dot(grad_rgb, segment_rgb[segment_id]) + grad_alpha;
      if (alpha_accum > 1.0e-8f) {
        d_loss_d_weight += grad_depth *
            (mids[segment_id] * alpha_accum - depth_weighted) /
            (alpha_accum * alpha_accum);
      }

      const float adj_trans_before =
          d_loss_d_weight * segment_alpha[segment_id] +
          adj_next_transmittance * segment_trans[segment_id];
      const float adj_segment_alpha = d_loss_d_weight * trans_before[segment_id];
      const float adj_segment_trans =
          adj_next_transmittance * trans_before[segment_id] - adj_segment_alpha;
      float4 grad_rgba = float4(
          weights[segment_id] * grad_rgb.x,
          weights[segment_id] * grad_rgb.y,
          weights[segment_id] * grad_rgb.z,
          0.0f);
      const uint rgba_base = owner * 4u;
      const float raw_density = site_rgba_f32[rgba_base + 3u];
      if (raw_density > 0.0f) {
        grad_rgba.w = adj_segment_trans * (-lengths[segment_id] * segment_trans[segment_id]);
      }
      grad_accum[owner] += grad_rgba;
      adj_next_transmittance = adj_trans_before;
    }
  }

  for (uint site_id = 0u; site_id < clamped_site_count; ++site_id) {
    wf2_atomic_add4(grad_site_rgba_f32, site_id * 4u, grad_accum[site_id]);
  }
}

kernel void wf2_endpoint_run_rgba_depth_replay_tensor(
    device const int* run_offsets_i32 [[buffer(0)]],
    device const int* run_owner_i32 [[buffer(1)]],
    device const float* run_start_f32 [[buffer(2)]],
    device const float* run_end_f32 [[buffer(3)]],
    device const float* site_rgba_f32 [[buffer(4)]],
    device const int* config_i32 [[buffer(5)]],
    device const float* config_f32 [[buffer(6)]],
    device float* output_rgb_f32 [[buffer(7)]],
    device float* output_alpha_f32 [[buffer(8)]],
    device float* output_depth_f32 [[buffer(9)]],
    uint sample_id [[thread_position_in_grid]]) {
  const uint track_count = uint(config_i32[0]);
  const uint frame_count = uint(config_i32[1]);
  const uint site_count = uint(config_i32[2]);
  const uint run_count = uint(config_i32[3]);
  const uint total_samples = track_count * frame_count;
  if (sample_id >= total_samples) {
    return;
  }

  const float far_depth = config_f32[0];
  const float transmittance_threshold = config_f32[1];
  const int begin_raw = run_offsets_i32[sample_id];
  const int end_raw = run_offsets_i32[sample_id + 1u];
  if (begin_raw < 0 || end_raw < begin_raw || uint(end_raw) > run_count) {
    return;
  }

  float3 rgb_accum = float3(0.0f, 0.0f, 0.0f);
  float alpha_accum = 0.0f;
  float depth_weighted = 0.0f;
  float transmittance = 1.0f;
  for (uint cursor = uint(begin_raw); cursor < uint(end_raw); ++cursor) {
    if (transmittance <= transmittance_threshold) {
      break;
    }
    const int owner_raw = run_owner_i32[cursor];
    if (owner_raw < 0 || uint(owner_raw) >= site_count) {
      continue;
    }
    const float start_depth = run_start_f32[cursor];
    const float end_depth = run_end_f32[cursor];
    const float length = end_depth - start_depth;
    if (!(length > 1.0e-8f)) {
      continue;
    }
    const uint owner = uint(owner_raw);
    const uint rgba_base = owner * 4u;
    const float density = max(site_rgba_f32[rgba_base + 3u], 0.0f);
    const float segment_transmittance = exp(-density * length);
    const float segment_alpha = 1.0f - segment_transmittance;
    const float weight = transmittance * segment_alpha;
    const float3 rgb = float3(
        site_rgba_f32[rgba_base + 0u],
        site_rgba_f32[rgba_base + 1u],
        site_rgba_f32[rgba_base + 2u]);
    const float depth_mass = wf2_continuous_depth_mass(
        start_depth,
        length,
        density,
        segment_transmittance,
        segment_alpha);
    rgb_accum += weight * rgb;
    alpha_accum += weight;
    depth_weighted += transmittance * depth_mass;
    transmittance *= segment_transmittance;
  }

  const uint out_base = sample_id * 3u;
  output_rgb_f32[out_base + 0u] = rgb_accum.x;
  output_rgb_f32[out_base + 1u] = rgb_accum.y;
  output_rgb_f32[out_base + 2u] = rgb_accum.z;
  output_alpha_f32[sample_id] = alpha_accum;
  output_depth_f32[sample_id] = alpha_accum > 1.0e-8f ? depth_weighted / alpha_accum : far_depth;
}

kernel void wf2_endpoint_run_vjp_direct_atomic_grad_only_tensor(
    device const int* run_offsets_i32 [[buffer(0)]],
    device const int* run_owner_i32 [[buffer(1)]],
    device const float* run_start_f32 [[buffer(2)]],
    device const float* run_end_f32 [[buffer(3)]],
    device const float* site_rgba_f32 [[buffer(4)]],
    device const float* grad_rgb_f32 [[buffer(5)]],
    device const float* grad_alpha_f32 [[buffer(6)]],
    device const float* grad_depth_f32 [[buffer(7)]],
    device const int* config_i32 [[buffer(8)]],
    device const float* config_f32 [[buffer(9)]],
    device atomic_float* grad_site_rgba_f32 [[buffer(10)]],
    uint sample_id [[thread_position_in_grid]]) {
  const uint track_count = uint(config_i32[0]);
  const uint frame_count = uint(config_i32[1]);
  const uint site_count = uint(config_i32[2]);
  const uint run_count = uint(config_i32[3]);
  const uint total_samples = track_count * frame_count;
  if (sample_id >= total_samples) {
    return;
  }

  const float transmittance_threshold = config_f32[1];
  const int begin_raw = run_offsets_i32[sample_id];
  const int end_raw = run_offsets_i32[sample_id + 1u];
  if (begin_raw < 0 || end_raw < begin_raw || uint(end_raw) > run_count) {
    return;
  }

  uint owners[WF2_MAX_REALRAY_SEGMENTS];
  float starts[WF2_MAX_REALRAY_SEGMENTS];
  float lengths[WF2_MAX_REALRAY_SEGMENTS];
  float depth_masses[WF2_MAX_REALRAY_SEGMENTS];
  float trans_before[WF2_MAX_REALRAY_SEGMENTS];
  float segment_trans[WF2_MAX_REALRAY_SEGMENTS];
  float segment_alpha[WF2_MAX_REALRAY_SEGMENTS];
  float weights[WF2_MAX_REALRAY_SEGMENTS];
  float3 segment_rgb[WF2_MAX_REALRAY_SEGMENTS];

  float alpha_accum = 0.0f;
  float depth_weighted = 0.0f;
  float transmittance = 1.0f;
  uint local_run_count = 0u;
  for (uint cursor = uint(begin_raw); cursor < uint(end_raw); ++cursor) {
    if (transmittance <= transmittance_threshold || local_run_count >= WF2_MAX_REALRAY_SEGMENTS) {
      break;
    }
    const int owner_raw = run_owner_i32[cursor];
    if (owner_raw < 0 || uint(owner_raw) >= site_count) {
      continue;
    }
    const float start_depth = run_start_f32[cursor];
    const float end_depth = run_end_f32[cursor];
    const float length = end_depth - start_depth;
    if (!(length > 1.0e-8f)) {
      continue;
    }
    const uint owner = uint(owner_raw);
    const uint rgba_base = owner * 4u;
    const float density = max(site_rgba_f32[rgba_base + 3u], 0.0f);
    const float seg_trans = exp(-density * length);
    const float seg_alpha = 1.0f - seg_trans;
    const float weight = transmittance * seg_alpha;
    const float3 rgb = float3(
        site_rgba_f32[rgba_base + 0u],
        site_rgba_f32[rgba_base + 1u],
        site_rgba_f32[rgba_base + 2u]);
    const float depth_mass = wf2_continuous_depth_mass(
        start_depth,
        length,
        density,
        seg_trans,
        seg_alpha);

    owners[local_run_count] = owner;
    starts[local_run_count] = start_depth;
    lengths[local_run_count] = length;
    depth_masses[local_run_count] = depth_mass;
    trans_before[local_run_count] = transmittance;
    segment_trans[local_run_count] = seg_trans;
    segment_alpha[local_run_count] = seg_alpha;
    weights[local_run_count] = weight;
    segment_rgb[local_run_count] = rgb;
    local_run_count += 1u;

    alpha_accum += weight;
    depth_weighted += transmittance * depth_mass;
    transmittance *= seg_trans;
  }

  const uint out_base = sample_id * 3u;
  const float3 grad_rgb = float3(
      grad_rgb_f32[out_base + 0u],
      grad_rgb_f32[out_base + 1u],
      grad_rgb_f32[out_base + 2u]);
  const float grad_alpha = grad_alpha_f32[sample_id];
  const float grad_depth = grad_depth_f32[sample_id];
  const bool has_depth = alpha_accum > 1.0e-8f;
  const float adj_alpha_out = has_depth
      ? grad_alpha - grad_depth * depth_weighted / (alpha_accum * alpha_accum)
      : grad_alpha;
  const float adj_depth_weighted_out = has_depth ? grad_depth / alpha_accum : 0.0f;
  float adj_next_transmittance = 0.0f;
  for (int run_id = int(local_run_count) - 1; run_id >= 0; --run_id) {
    const uint owner = owners[run_id];
    const uint rgba_base = owner * 4u;
    const float raw_density = site_rgba_f32[rgba_base + 3u];
    const float density = max(raw_density, 0.0f);
    const float adj_weight = dot(grad_rgb, segment_rgb[run_id]) + adj_alpha_out;
    const float adj_depth_mass = adj_depth_weighted_out * trans_before[run_id];
    const float adj_trans_before =
        adj_weight * segment_alpha[run_id] +
        adj_depth_weighted_out * depth_masses[run_id] +
        adj_next_transmittance * segment_trans[run_id];
    const float adj_segment_alpha = adj_weight * trans_before[run_id];
    float adj_segment_trans = adj_next_transmittance * trans_before[run_id] - adj_segment_alpha;

    float4 grad_rgba = float4(
        weights[run_id] * grad_rgb.x,
        weights[run_id] * grad_rgb.y,
        weights[run_id] * grad_rgb.z,
        0.0f);
    if (raw_density > 0.0f) {
      const float length = lengths[run_id];
      const float dtrans_dd = -length * segment_trans[run_id];
      const float dmass_dd = wf2_continuous_depth_mass_grad_density(
          starts[run_id],
          length,
          density,
          segment_trans[run_id],
          segment_alpha[run_id]);
      grad_rgba.w = adj_segment_trans * dtrans_dd + adj_depth_mass * dmass_dd;
    }
    wf2_atomic_add4(grad_site_rgba_f32, rgba_base, grad_rgba);
    adj_next_transmittance = adj_trans_before;
  }
}

kernel void wf2_endpoint_run_mse_vjp_direct_atomic_rgb_only_tensor(
    device const int* run_offsets_i32 [[buffer(0)]],
    device const int* run_owner_i32 [[buffer(1)]],
    device const float* run_start_f32 [[buffer(2)]],
    device const float* run_end_f32 [[buffer(3)]],
    device const float* site_rgba_f32 [[buffer(4)]],
    device const float* target_rgb_f32 [[buffer(5)]],
    device const int* config_i32 [[buffer(6)]],
    device const float* config_f32 [[buffer(7)]],
    device atomic_float* loss_f32 [[buffer(8)]],
    device atomic_float* grad_site_rgba_f32 [[buffer(9)]],
    uint sample_id [[thread_position_in_grid]]) {
  const uint track_count = uint(config_i32[0]);
  const uint frame_count = uint(config_i32[1]);
  const uint site_count = uint(config_i32[2]);
  const uint run_count = uint(config_i32[3]);
  const uint total_samples = track_count * frame_count;
  if (sample_id >= total_samples) {
    return;
  }

  const float transmittance_threshold = config_f32[1];
  const int begin_raw = run_offsets_i32[sample_id];
  const int end_raw = run_offsets_i32[sample_id + 1u];
  if (begin_raw < 0 || end_raw < begin_raw || uint(end_raw) > run_count) {
    return;
  }

  uint owners[WF2_MAX_REALRAY_SEGMENTS];
  float lengths[WF2_MAX_REALRAY_SEGMENTS];
  float trans_before[WF2_MAX_REALRAY_SEGMENTS];
  float segment_trans[WF2_MAX_REALRAY_SEGMENTS];

  float3 rgb_accum = float3(0.0f, 0.0f, 0.0f);
  float transmittance = 1.0f;
  uint local_run_count = 0u;
  for (uint cursor = uint(begin_raw); cursor < uint(end_raw); ++cursor) {
    if (transmittance <= transmittance_threshold || local_run_count >= WF2_MAX_REALRAY_SEGMENTS) {
      break;
    }
    const int owner_raw = run_owner_i32[cursor];
    if (owner_raw < 0 || uint(owner_raw) >= site_count) {
      continue;
    }
    const float start_depth = run_start_f32[cursor];
    const float end_depth = run_end_f32[cursor];
    const float length = end_depth - start_depth;
    if (!(length > 1.0e-8f)) {
      continue;
    }
    const uint owner = uint(owner_raw);
    const uint rgba_base = owner * 4u;
    const float density = max(site_rgba_f32[rgba_base + 3u], 0.0f);
    const float seg_trans = exp(-density * length);
    const float seg_alpha = 1.0f - seg_trans;
    const float weight = transmittance * seg_alpha;
    const float3 rgb = float3(
        site_rgba_f32[rgba_base + 0u],
        site_rgba_f32[rgba_base + 1u],
        site_rgba_f32[rgba_base + 2u]);

    owners[local_run_count] = owner;
    lengths[local_run_count] = length;
    trans_before[local_run_count] = transmittance;
    segment_trans[local_run_count] = seg_trans;
    local_run_count += 1u;

    rgb_accum += weight * rgb;
    transmittance *= seg_trans;
  }

  const uint out_base = sample_id * 3u;
  const float3 target_rgb = float3(
      target_rgb_f32[out_base + 0u],
      target_rgb_f32[out_base + 1u],
      target_rgb_f32[out_base + 2u]);
  const float3 diff = rgb_accum - target_rgb;
  const float inv_element_count = 1.0f / float(total_samples * 3u);
  const float sample_loss = dot(diff, diff) * inv_element_count;
  atomic_fetch_add_explicit(&loss_f32[0], sample_loss, memory_order_relaxed);
  const float3 grad_rgb = (2.0f * inv_element_count) * diff;

  float adj_next_transmittance = 0.0f;
  for (int run_id = int(local_run_count) - 1; run_id >= 0; --run_id) {
    const uint owner = owners[run_id];
    const uint rgba_base = owner * 4u;
    const float raw_density = site_rgba_f32[rgba_base + 3u];
    const float3 rgb = float3(
        site_rgba_f32[rgba_base + 0u],
        site_rgba_f32[rgba_base + 1u],
        site_rgba_f32[rgba_base + 2u]);
    const float seg_trans = segment_trans[run_id];
    const float seg_alpha = 1.0f - seg_trans;
    const float weight = trans_before[run_id] * seg_alpha;
    const float adj_weight = dot(grad_rgb, rgb);
    const float adj_trans_before =
        adj_weight * seg_alpha +
        adj_next_transmittance * seg_trans;
    const float adj_segment_alpha = adj_weight * trans_before[run_id];
    const float adj_segment_trans = adj_next_transmittance * trans_before[run_id] - adj_segment_alpha;
    float4 grad_rgba = float4(
        weight * grad_rgb.x,
        weight * grad_rgb.y,
        weight * grad_rgb.z,
        0.0f);
    if (raw_density > 0.0f) {
      grad_rgba.w = adj_segment_trans * (-lengths[run_id] * seg_trans);
    }
    wf2_atomic_add4(grad_site_rgba_f32, rgba_base, grad_rgba);
    adj_next_transmittance = adj_trans_before;
  }
}

kernel void wf2_segment_tape_mse_vjp_direct_atomic_rgb_only_tensor(
    device const int* segment_offsets_i32 [[buffer(0)]],
    device const int* segment_owner_i32 [[buffer(1)]],
    device const float* segment_length_f32 [[buffer(2)]],
    device const float* site_rgba_f32 [[buffer(3)]],
    device const float* target_rgb_f32 [[buffer(4)]],
    device const int* config_i32 [[buffer(5)]],
    device const float* config_f32 [[buffer(6)]],
    device atomic_float* loss_f32 [[buffer(7)]],
    device atomic_float* grad_site_rgba_f32 [[buffer(8)]],
    uint sample_id [[thread_position_in_grid]]) {
  const uint track_count = uint(config_i32[0]);
  const uint frame_count = uint(config_i32[1]);
  const uint site_count = uint(config_i32[2]);
  const uint segment_count = uint(config_i32[3]);
  const uint total_samples = track_count * frame_count;
  if (sample_id >= total_samples) {
    return;
  }

  const float transmittance_threshold = config_f32[1];
  const int begin_raw = segment_offsets_i32[sample_id];
  const int end_raw = segment_offsets_i32[sample_id + 1u];
  if (begin_raw < 0 || end_raw < begin_raw || uint(end_raw) > segment_count) {
    return;
  }

  uint owners[WF2_MAX_REALRAY_SEGMENTS];
  float lengths[WF2_MAX_REALRAY_SEGMENTS];
  float trans_before[WF2_MAX_REALRAY_SEGMENTS];
  float segment_trans[WF2_MAX_REALRAY_SEGMENTS];

  float3 rgb_accum = float3(0.0f, 0.0f, 0.0f);
  float transmittance = 1.0f;
  uint local_segment_count = 0u;
  for (uint cursor = uint(begin_raw); cursor < uint(end_raw); ++cursor) {
    if (transmittance <= transmittance_threshold || local_segment_count >= WF2_MAX_REALRAY_SEGMENTS) {
      break;
    }
    const int owner_raw = segment_owner_i32[cursor];
    if (owner_raw < 0 || uint(owner_raw) >= site_count) {
      continue;
    }
    const float length = segment_length_f32[cursor];
    if (!(length > 1.0e-8f)) {
      continue;
    }
    const uint owner = uint(owner_raw);
    const uint rgba_base = owner * 4u;
    const float density = max(site_rgba_f32[rgba_base + 3u], 0.0f);
    const float seg_trans = exp(-density * length);
    const float seg_alpha = 1.0f - seg_trans;
    const float weight = transmittance * seg_alpha;
    const float3 rgb = float3(
        site_rgba_f32[rgba_base + 0u],
        site_rgba_f32[rgba_base + 1u],
        site_rgba_f32[rgba_base + 2u]);

    owners[local_segment_count] = owner;
    lengths[local_segment_count] = length;
    trans_before[local_segment_count] = transmittance;
    segment_trans[local_segment_count] = seg_trans;
    local_segment_count += 1u;

    rgb_accum += weight * rgb;
    transmittance *= seg_trans;
  }

  const uint out_base = sample_id * 3u;
  const float3 target_rgb = float3(
      target_rgb_f32[out_base + 0u],
      target_rgb_f32[out_base + 1u],
      target_rgb_f32[out_base + 2u]);
  const float3 diff = rgb_accum - target_rgb;
  const float inv_element_count = 1.0f / float(total_samples * 3u);
  atomic_fetch_add_explicit(&loss_f32[0], dot(diff, diff) * inv_element_count, memory_order_relaxed);
  const float3 grad_rgb = (2.0f * inv_element_count) * diff;

  float adj_next_transmittance = 0.0f;
  for (int segment_id = int(local_segment_count) - 1; segment_id >= 0; --segment_id) {
    const uint owner = owners[segment_id];
    const uint rgba_base = owner * 4u;
    const float raw_density = site_rgba_f32[rgba_base + 3u];
    const float3 rgb = float3(
        site_rgba_f32[rgba_base + 0u],
        site_rgba_f32[rgba_base + 1u],
        site_rgba_f32[rgba_base + 2u]);
    const float seg_trans = segment_trans[segment_id];
    const float seg_alpha = 1.0f - seg_trans;
    const float weight = trans_before[segment_id] * seg_alpha;
    const float adj_weight = dot(grad_rgb, rgb);
    const float adj_trans_before =
        adj_weight * seg_alpha +
        adj_next_transmittance * seg_trans;
    const float adj_segment_alpha = adj_weight * trans_before[segment_id];
    const float adj_segment_trans = adj_next_transmittance * trans_before[segment_id] - adj_segment_alpha;
    float4 grad_rgba = float4(
        weight * grad_rgb.x,
        weight * grad_rgb.y,
        weight * grad_rgb.z,
        0.0f);
    if (raw_density > 0.0f) {
      grad_rgba.w = adj_segment_trans * (-lengths[segment_id] * seg_trans);
    }
    wf2_atomic_add4(grad_site_rgba_f32, rgba_base, grad_rgba);
    adj_next_transmittance = adj_trans_before;
  }
}

kernel void wf2_endpoint_delta_replace_rgba_depth_replay_tensor(
    device const int* base_offsets_i32 [[buffer(0)]],
    device const int* base_owner_i32 [[buffer(1)]],
    device const float* base_start_f32 [[buffer(2)]],
    device const float* base_end_f32 [[buffer(3)]],
    device const int* track_change_offsets_i32 [[buffer(4)]],
    device const int* change_frame_i32 [[buffer(5)]],
    device const int* change_offsets_i32 [[buffer(6)]],
    device const int* change_owner_i32 [[buffer(7)]],
    device const float* change_start_f32 [[buffer(8)]],
    device const float* change_end_f32 [[buffer(9)]],
    device const float* site_rgba_f32 [[buffer(10)]],
    device const int* config_i32 [[buffer(11)]],
    device const float* config_f32 [[buffer(12)]],
    device float* output_rgb_f32 [[buffer(13)]],
    device float* output_alpha_f32 [[buffer(14)]],
    device float* output_depth_f32 [[buffer(15)]],
    uint sample_id [[thread_position_in_grid]]) {
  const uint track_count = uint(config_i32[0]);
  const uint frame_count = uint(config_i32[1]);
  const uint site_count = uint(config_i32[2]);
  const uint base_record_count = uint(config_i32[3]);
  const uint change_count = uint(config_i32[4]);
  const uint change_record_count = uint(config_i32[5]);
  const uint total_samples = track_count * frame_count;
  if (sample_id >= total_samples) {
    return;
  }

  const uint track_id = sample_id / frame_count;
  const uint frame_id = sample_id - track_id * frame_count;
  const float far_depth = config_f32[0];
  const float transmittance_threshold = config_f32[1];

  int selected_change = -1;
  const int change_begin_raw = track_change_offsets_i32[track_id];
  const int change_end_raw = track_change_offsets_i32[track_id + 1u];
  if (change_begin_raw < 0 || change_end_raw < change_begin_raw || uint(change_end_raw) > change_count) {
    return;
  }
  for (uint change_cursor = uint(change_begin_raw); change_cursor < uint(change_end_raw); ++change_cursor) {
    const int changed_frame = change_frame_i32[change_cursor];
    if (changed_frame < 0) {
      continue;
    }
    if (uint(changed_frame) <= frame_id) {
      selected_change = int(change_cursor);
      continue;
    }
    break;
  }

  const bool use_change = selected_change >= 0;
  const int begin_raw = use_change ? change_offsets_i32[uint(selected_change)] : base_offsets_i32[track_id];
  const int end_raw = use_change ? change_offsets_i32[uint(selected_change) + 1u] : base_offsets_i32[track_id + 1u];
  const uint record_count = use_change ? change_record_count : base_record_count;
  if (begin_raw < 0 || end_raw < begin_raw || uint(end_raw) > record_count) {
    return;
  }

  float3 rgb_accum = float3(0.0f, 0.0f, 0.0f);
  float alpha_accum = 0.0f;
  float depth_weighted = 0.0f;
  float transmittance = 1.0f;
  for (uint cursor = uint(begin_raw); cursor < uint(end_raw); ++cursor) {
    if (transmittance <= transmittance_threshold) {
      break;
    }
    const int owner_raw = use_change ? change_owner_i32[cursor] : base_owner_i32[cursor];
    if (owner_raw < 0 || uint(owner_raw) >= site_count) {
      continue;
    }
    const float start_depth = use_change ? change_start_f32[cursor] : base_start_f32[cursor];
    const float end_depth = use_change ? change_end_f32[cursor] : base_end_f32[cursor];
    const float length = end_depth - start_depth;
    if (!(length > 1.0e-8f)) {
      continue;
    }
    const uint owner = uint(owner_raw);
    const uint rgba_base = owner * 4u;
    const float density = max(site_rgba_f32[rgba_base + 3u], 0.0f);
    const float segment_transmittance = exp(-density * length);
    const float segment_alpha = 1.0f - segment_transmittance;
    const float weight = transmittance * segment_alpha;
    const float3 rgb = float3(
        site_rgba_f32[rgba_base + 0u],
        site_rgba_f32[rgba_base + 1u],
        site_rgba_f32[rgba_base + 2u]);
    const float depth_mass = wf2_continuous_depth_mass(
        start_depth,
        length,
        density,
        segment_transmittance,
        segment_alpha);
    rgb_accum += weight * rgb;
    alpha_accum += weight;
    depth_weighted += transmittance * depth_mass;
    transmittance *= segment_transmittance;
  }

  const uint out_base = sample_id * 3u;
  output_rgb_f32[out_base + 0u] = rgb_accum.x;
  output_rgb_f32[out_base + 1u] = rgb_accum.y;
  output_rgb_f32[out_base + 2u] = rgb_accum.z;
  output_alpha_f32[sample_id] = alpha_accum;
  output_depth_f32[sample_id] = alpha_accum > 1.0e-8f ? depth_weighted / alpha_accum : far_depth;
}

kernel void wf2_endpoint_delta_replace_vjp_direct_atomic_grad_only_tensor(
    device const int* base_offsets_i32 [[buffer(0)]],
    device const int* base_owner_i32 [[buffer(1)]],
    device const float* base_start_f32 [[buffer(2)]],
    device const float* base_end_f32 [[buffer(3)]],
    device const int* track_change_offsets_i32 [[buffer(4)]],
    device const int* change_frame_i32 [[buffer(5)]],
    device const int* change_offsets_i32 [[buffer(6)]],
    device const int* change_owner_i32 [[buffer(7)]],
    device const float* change_start_f32 [[buffer(8)]],
    device const float* change_end_f32 [[buffer(9)]],
    device const float* site_rgba_f32 [[buffer(10)]],
    device const float* grad_rgb_f32 [[buffer(11)]],
    device const float* grad_alpha_f32 [[buffer(12)]],
    device const float* grad_depth_f32 [[buffer(13)]],
    device const int* config_i32 [[buffer(14)]],
    device const float* config_f32 [[buffer(15)]],
    device atomic_float* grad_site_rgba_f32 [[buffer(16)]],
    uint sample_id [[thread_position_in_grid]]) {
  const uint track_count = uint(config_i32[0]);
  const uint frame_count = uint(config_i32[1]);
  const uint site_count = uint(config_i32[2]);
  const uint base_record_count = uint(config_i32[3]);
  const uint change_count = uint(config_i32[4]);
  const uint change_record_count = uint(config_i32[5]);
  const uint total_samples = track_count * frame_count;
  if (sample_id >= total_samples) {
    return;
  }

  const uint track_id = sample_id / frame_count;
  const uint frame_id = sample_id - track_id * frame_count;
  const float transmittance_threshold = config_f32[1];

  int selected_change = -1;
  const int change_begin_raw = track_change_offsets_i32[track_id];
  const int change_end_raw = track_change_offsets_i32[track_id + 1u];
  if (change_begin_raw < 0 || change_end_raw < change_begin_raw || uint(change_end_raw) > change_count) {
    return;
  }
  for (uint change_cursor = uint(change_begin_raw); change_cursor < uint(change_end_raw); ++change_cursor) {
    const int changed_frame = change_frame_i32[change_cursor];
    if (changed_frame < 0) {
      continue;
    }
    if (uint(changed_frame) <= frame_id) {
      selected_change = int(change_cursor);
      continue;
    }
    break;
  }

  const bool use_change = selected_change >= 0;
  const int begin_raw = use_change ? change_offsets_i32[uint(selected_change)] : base_offsets_i32[track_id];
  const int end_raw = use_change ? change_offsets_i32[uint(selected_change) + 1u] : base_offsets_i32[track_id + 1u];
  const uint record_count = use_change ? change_record_count : base_record_count;
  if (begin_raw < 0 || end_raw < begin_raw || uint(end_raw) > record_count) {
    return;
  }

  uint owners[WF2_MAX_REALRAY_SEGMENTS];
  float starts[WF2_MAX_REALRAY_SEGMENTS];
  float lengths[WF2_MAX_REALRAY_SEGMENTS];
  float depth_masses[WF2_MAX_REALRAY_SEGMENTS];
  float trans_before[WF2_MAX_REALRAY_SEGMENTS];
  float segment_trans[WF2_MAX_REALRAY_SEGMENTS];
  float segment_alpha[WF2_MAX_REALRAY_SEGMENTS];
  float weights[WF2_MAX_REALRAY_SEGMENTS];
  float3 segment_rgb[WF2_MAX_REALRAY_SEGMENTS];

  float alpha_accum = 0.0f;
  float depth_weighted = 0.0f;
  float transmittance = 1.0f;
  uint local_run_count = 0u;
  for (uint cursor = uint(begin_raw); cursor < uint(end_raw); ++cursor) {
    if (transmittance <= transmittance_threshold || local_run_count >= WF2_MAX_REALRAY_SEGMENTS) {
      break;
    }
    const int owner_raw = use_change ? change_owner_i32[cursor] : base_owner_i32[cursor];
    if (owner_raw < 0 || uint(owner_raw) >= site_count) {
      continue;
    }
    const float start_depth = use_change ? change_start_f32[cursor] : base_start_f32[cursor];
    const float end_depth = use_change ? change_end_f32[cursor] : base_end_f32[cursor];
    const float length = end_depth - start_depth;
    if (!(length > 1.0e-8f)) {
      continue;
    }
    const uint owner = uint(owner_raw);
    const uint rgba_base = owner * 4u;
    const float density = max(site_rgba_f32[rgba_base + 3u], 0.0f);
    const float seg_trans = exp(-density * length);
    const float seg_alpha = 1.0f - seg_trans;
    const float weight = transmittance * seg_alpha;
    const float3 rgb = float3(
        site_rgba_f32[rgba_base + 0u],
        site_rgba_f32[rgba_base + 1u],
        site_rgba_f32[rgba_base + 2u]);
    const float depth_mass = wf2_continuous_depth_mass(
        start_depth,
        length,
        density,
        seg_trans,
        seg_alpha);

    owners[local_run_count] = owner;
    starts[local_run_count] = start_depth;
    lengths[local_run_count] = length;
    depth_masses[local_run_count] = depth_mass;
    trans_before[local_run_count] = transmittance;
    segment_trans[local_run_count] = seg_trans;
    segment_alpha[local_run_count] = seg_alpha;
    weights[local_run_count] = weight;
    segment_rgb[local_run_count] = rgb;
    local_run_count += 1u;

    alpha_accum += weight;
    depth_weighted += transmittance * depth_mass;
    transmittance *= seg_trans;
  }

  const uint out_base = sample_id * 3u;
  const float3 grad_rgb = float3(
      grad_rgb_f32[out_base + 0u],
      grad_rgb_f32[out_base + 1u],
      grad_rgb_f32[out_base + 2u]);
  const float grad_alpha = grad_alpha_f32[sample_id];
  const float grad_depth = grad_depth_f32[sample_id];
  const bool has_depth = alpha_accum > 1.0e-8f;
  const float adj_alpha_out = has_depth
      ? grad_alpha - grad_depth * depth_weighted / (alpha_accum * alpha_accum)
      : grad_alpha;
  const float adj_depth_weighted_out = has_depth ? grad_depth / alpha_accum : 0.0f;
  float adj_next_transmittance = 0.0f;
  for (int run_id = int(local_run_count) - 1; run_id >= 0; --run_id) {
    const uint owner = owners[run_id];
    const uint rgba_base = owner * 4u;
    const float raw_density = site_rgba_f32[rgba_base + 3u];
    const float density = max(raw_density, 0.0f);
    const float adj_weight = dot(grad_rgb, segment_rgb[run_id]) + adj_alpha_out;
    const float adj_depth_mass = adj_depth_weighted_out * trans_before[run_id];
    const float adj_trans_before =
        adj_weight * segment_alpha[run_id] +
        adj_depth_weighted_out * depth_masses[run_id] +
        adj_next_transmittance * segment_trans[run_id];
    const float adj_segment_alpha = adj_weight * trans_before[run_id];
    float adj_segment_trans = adj_next_transmittance * trans_before[run_id] - adj_segment_alpha;

    float4 grad_rgba = float4(
        weights[run_id] * grad_rgb.x,
        weights[run_id] * grad_rgb.y,
        weights[run_id] * grad_rgb.z,
        0.0f);
    if (raw_density > 0.0f) {
      const float length = lengths[run_id];
      const float dtrans_dd = -length * segment_trans[run_id];
      const float dmass_dd = wf2_continuous_depth_mass_grad_density(
          starts[run_id],
          length,
          density,
          segment_trans[run_id],
          segment_alpha[run_id]);
      grad_rgba.w = adj_segment_trans * dtrans_dd + adj_depth_mass * dmass_dd;
    }
    wf2_atomic_add4(grad_site_rgba_f32, rgba_base, grad_rgba);
    adj_next_transmittance = adj_trans_before;
  }
}

kernel void wf2_endpoint_record_delta_replace_rgba_depth_replay_tensor(
    device const float* boundary_f32 [[buffer(0)]],
    device const float* rays_f32 [[buffer(1)]],
    device const float* frame_t_f32 [[buffer(2)]],
    device const int* base_offsets_i32 [[buffer(3)]],
    device const int* base_owner_i32 [[buffer(4)]],
    device const int* base_left_i32 [[buffer(5)]],
    device const int* base_right_i32 [[buffer(6)]],
    device const int* track_change_offsets_i32 [[buffer(7)]],
    device const int* change_frame_i32 [[buffer(8)]],
    device const int* change_offsets_i32 [[buffer(9)]],
    device const int* change_owner_i32 [[buffer(10)]],
    device const int* change_left_i32 [[buffer(11)]],
    device const int* change_right_i32 [[buffer(12)]],
    device const float* site_rgba_f32 [[buffer(13)]],
    device const int* config_i32 [[buffer(14)]],
    device const float* config_f32 [[buffer(15)]],
    device float* output_rgb_f32 [[buffer(16)]],
    device float* output_alpha_f32 [[buffer(17)]],
    device float* output_depth_f32 [[buffer(18)]],
    uint sample_id [[thread_position_in_grid]]) {
  const uint boundary_count = uint(config_i32[0]);
  const uint track_count = uint(config_i32[1]);
  const uint frame_count = uint(config_i32[2]);
  const uint site_count = uint(config_i32[3]);
  const uint base_record_count = uint(config_i32[4]);
  const uint change_count = uint(config_i32[5]);
  const uint change_record_count = uint(config_i32[6]);
  const uint total_samples = track_count * frame_count;
  if (sample_id >= total_samples) {
    return;
  }

  const uint track_id = sample_id / frame_count;
  const uint frame_id = sample_id - track_id * frame_count;
  const float near_depth = config_f32[0];
  const float far_depth = config_f32[1];
  const float invalid_epsilon = config_f32[2];
  const float transmittance_threshold = config_f32[3];
  const uint ray_base = sample_id * 6u;
  const float3 origin = float3(rays_f32[ray_base + 0u], rays_f32[ray_base + 1u], rays_f32[ray_base + 2u]);
  const float3 direction = float3(rays_f32[ray_base + 3u], rays_f32[ray_base + 4u], rays_f32[ray_base + 5u]);
  const float t = frame_t_f32[frame_id];

  int selected_change = -1;
  const int change_begin_raw = track_change_offsets_i32[track_id];
  const int change_end_raw = track_change_offsets_i32[track_id + 1u];
  if (change_begin_raw < 0 || change_end_raw < change_begin_raw || uint(change_end_raw) > change_count) {
    return;
  }
  for (uint change_cursor = uint(change_begin_raw); change_cursor < uint(change_end_raw); ++change_cursor) {
    const int changed_frame = change_frame_i32[change_cursor];
    if (changed_frame < 0) {
      continue;
    }
    if (uint(changed_frame) <= frame_id) {
      selected_change = int(change_cursor);
      continue;
    }
    break;
  }

  const bool use_change = selected_change >= 0;
  const int begin_raw = use_change ? change_offsets_i32[uint(selected_change)] : base_offsets_i32[track_id];
  const int end_raw = use_change ? change_offsets_i32[uint(selected_change) + 1u] : base_offsets_i32[track_id + 1u];
  const uint record_count = use_change ? change_record_count : base_record_count;
  if (begin_raw < 0 || end_raw < begin_raw || uint(end_raw) > record_count) {
    return;
  }

  float3 rgb_accum = float3(0.0f, 0.0f, 0.0f);
  float alpha_accum = 0.0f;
  float depth_weighted = 0.0f;
  float transmittance = 1.0f;
  for (uint cursor = uint(begin_raw); cursor < uint(end_raw); ++cursor) {
    if (transmittance <= transmittance_threshold) {
      break;
    }
    const int owner_raw = use_change ? change_owner_i32[cursor] : base_owner_i32[cursor];
    if (owner_raw < 0 || uint(owner_raw) >= site_count) {
      continue;
    }
    const int left_cut = use_change ? change_left_i32[cursor] : base_left_i32[cursor];
    const int right_cut = use_change ? change_right_i32[cursor] : base_right_i32[cursor];
    float start_depth = 0.0f;
    float end_depth = 0.0f;
    if (!wf2_endpoint_record_cut_depth(
            boundary_f32,
            boundary_count,
            left_cut,
            origin,
            direction,
            t,
            near_depth,
            far_depth,
            invalid_epsilon,
            start_depth) ||
        !wf2_endpoint_record_cut_depth(
            boundary_f32,
            boundary_count,
            right_cut,
            origin,
            direction,
            t,
            near_depth,
            far_depth,
            invalid_epsilon,
            end_depth)) {
      continue;
    }
    const float length = end_depth - start_depth;
    if (!(length > 1.0e-8f)) {
      continue;
    }
    const uint owner = uint(owner_raw);
    const uint rgba_base = owner * 4u;
    const float density = max(site_rgba_f32[rgba_base + 3u], 0.0f);
    const float segment_transmittance = exp(-density * length);
    const float segment_alpha = 1.0f - segment_transmittance;
    const float weight = transmittance * segment_alpha;
    const float3 rgb = float3(
        site_rgba_f32[rgba_base + 0u],
        site_rgba_f32[rgba_base + 1u],
        site_rgba_f32[rgba_base + 2u]);
    const float depth_mass = wf2_continuous_depth_mass(
        start_depth,
        length,
        density,
        segment_transmittance,
        segment_alpha);
    rgb_accum += weight * rgb;
    alpha_accum += weight;
    depth_weighted += transmittance * depth_mass;
    transmittance *= segment_transmittance;
  }

  const uint out_base = sample_id * 3u;
  output_rgb_f32[out_base + 0u] = rgb_accum.x;
  output_rgb_f32[out_base + 1u] = rgb_accum.y;
  output_rgb_f32[out_base + 2u] = rgb_accum.z;
  output_alpha_f32[sample_id] = alpha_accum;
  output_depth_f32[sample_id] = alpha_accum > 1.0e-8f ? depth_weighted / alpha_accum : far_depth;
}

kernel void wf2_endpoint_record_delta_replace_vjp_direct_atomic_grad_only_tensor(
    device const float* boundary_f32 [[buffer(0)]],
    device const float* rays_f32 [[buffer(1)]],
    device const float* frame_t_f32 [[buffer(2)]],
    device const int* base_offsets_i32 [[buffer(3)]],
    device const int* base_owner_i32 [[buffer(4)]],
    device const int* base_left_i32 [[buffer(5)]],
    device const int* base_right_i32 [[buffer(6)]],
    device const int* track_change_offsets_i32 [[buffer(7)]],
    device const int* change_frame_i32 [[buffer(8)]],
    device const int* change_offsets_i32 [[buffer(9)]],
    device const int* change_owner_i32 [[buffer(10)]],
    device const int* change_left_i32 [[buffer(11)]],
    device const int* change_right_i32 [[buffer(12)]],
    device const float* site_rgba_f32 [[buffer(13)]],
    device const float* grad_rgb_f32 [[buffer(14)]],
    device const float* grad_alpha_f32 [[buffer(15)]],
    device const float* grad_depth_f32 [[buffer(16)]],
    device const int* config_i32 [[buffer(17)]],
    device const float* config_f32 [[buffer(18)]],
    device atomic_float* grad_site_rgba_f32 [[buffer(19)]],
    uint sample_id [[thread_position_in_grid]]) {
  const uint boundary_count = uint(config_i32[0]);
  const uint track_count = uint(config_i32[1]);
  const uint frame_count = uint(config_i32[2]);
  const uint site_count = uint(config_i32[3]);
  const uint base_record_count = uint(config_i32[4]);
  const uint change_count = uint(config_i32[5]);
  const uint change_record_count = uint(config_i32[6]);
  const uint total_samples = track_count * frame_count;
  if (sample_id >= total_samples) {
    return;
  }

  const uint track_id = sample_id / frame_count;
  const uint frame_id = sample_id - track_id * frame_count;
  const float near_depth = config_f32[0];
  const float far_depth = config_f32[1];
  const float invalid_epsilon = config_f32[2];
  const float transmittance_threshold = config_f32[3];
  const uint ray_base = sample_id * 6u;
  const float3 origin = float3(rays_f32[ray_base + 0u], rays_f32[ray_base + 1u], rays_f32[ray_base + 2u]);
  const float3 direction = float3(rays_f32[ray_base + 3u], rays_f32[ray_base + 4u], rays_f32[ray_base + 5u]);
  const float t = frame_t_f32[frame_id];

  int selected_change = -1;
  const int change_begin_raw = track_change_offsets_i32[track_id];
  const int change_end_raw = track_change_offsets_i32[track_id + 1u];
  if (change_begin_raw < 0 || change_end_raw < change_begin_raw || uint(change_end_raw) > change_count) {
    return;
  }
  for (uint change_cursor = uint(change_begin_raw); change_cursor < uint(change_end_raw); ++change_cursor) {
    const int changed_frame = change_frame_i32[change_cursor];
    if (changed_frame < 0) {
      continue;
    }
    if (uint(changed_frame) <= frame_id) {
      selected_change = int(change_cursor);
      continue;
    }
    break;
  }

  const bool use_change = selected_change >= 0;
  const int begin_raw = use_change ? change_offsets_i32[uint(selected_change)] : base_offsets_i32[track_id];
  const int end_raw = use_change ? change_offsets_i32[uint(selected_change) + 1u] : base_offsets_i32[track_id + 1u];
  const uint record_count = use_change ? change_record_count : base_record_count;
  if (begin_raw < 0 || end_raw < begin_raw || uint(end_raw) > record_count) {
    return;
  }

  uint owners[WF2_MAX_REALRAY_SEGMENTS];
  float starts[WF2_MAX_REALRAY_SEGMENTS];
  float lengths[WF2_MAX_REALRAY_SEGMENTS];
  float depth_masses[WF2_MAX_REALRAY_SEGMENTS];
  float trans_before[WF2_MAX_REALRAY_SEGMENTS];
  float segment_trans[WF2_MAX_REALRAY_SEGMENTS];
  float segment_alpha[WF2_MAX_REALRAY_SEGMENTS];
  float weights[WF2_MAX_REALRAY_SEGMENTS];
  float3 segment_rgb[WF2_MAX_REALRAY_SEGMENTS];

  float alpha_accum = 0.0f;
  float depth_weighted = 0.0f;
  float transmittance = 1.0f;
  uint local_run_count = 0u;
  for (uint cursor = uint(begin_raw); cursor < uint(end_raw); ++cursor) {
    if (transmittance <= transmittance_threshold || local_run_count >= WF2_MAX_REALRAY_SEGMENTS) {
      break;
    }
    const int owner_raw = use_change ? change_owner_i32[cursor] : base_owner_i32[cursor];
    if (owner_raw < 0 || uint(owner_raw) >= site_count) {
      continue;
    }
    const int left_cut = use_change ? change_left_i32[cursor] : base_left_i32[cursor];
    const int right_cut = use_change ? change_right_i32[cursor] : base_right_i32[cursor];
    float start_depth = 0.0f;
    float end_depth = 0.0f;
    if (!wf2_endpoint_record_cut_depth(
            boundary_f32,
            boundary_count,
            left_cut,
            origin,
            direction,
            t,
            near_depth,
            far_depth,
            invalid_epsilon,
            start_depth) ||
        !wf2_endpoint_record_cut_depth(
            boundary_f32,
            boundary_count,
            right_cut,
            origin,
            direction,
            t,
            near_depth,
            far_depth,
            invalid_epsilon,
            end_depth)) {
      continue;
    }
    const float length = end_depth - start_depth;
    if (!(length > 1.0e-8f)) {
      continue;
    }
    const uint owner = uint(owner_raw);
    const uint rgba_base = owner * 4u;
    const float density = max(site_rgba_f32[rgba_base + 3u], 0.0f);
    const float seg_trans = exp(-density * length);
    const float seg_alpha = 1.0f - seg_trans;
    const float weight = transmittance * seg_alpha;
    const float3 rgb = float3(
        site_rgba_f32[rgba_base + 0u],
        site_rgba_f32[rgba_base + 1u],
        site_rgba_f32[rgba_base + 2u]);
    const float depth_mass = wf2_continuous_depth_mass(
        start_depth,
        length,
        density,
        seg_trans,
        seg_alpha);

    owners[local_run_count] = owner;
    starts[local_run_count] = start_depth;
    lengths[local_run_count] = length;
    depth_masses[local_run_count] = depth_mass;
    trans_before[local_run_count] = transmittance;
    segment_trans[local_run_count] = seg_trans;
    segment_alpha[local_run_count] = seg_alpha;
    weights[local_run_count] = weight;
    segment_rgb[local_run_count] = rgb;
    local_run_count += 1u;

    alpha_accum += weight;
    depth_weighted += transmittance * depth_mass;
    transmittance *= seg_trans;
  }

  const uint out_base = sample_id * 3u;
  const float3 grad_rgb = float3(
      grad_rgb_f32[out_base + 0u],
      grad_rgb_f32[out_base + 1u],
      grad_rgb_f32[out_base + 2u]);
  const float grad_alpha = grad_alpha_f32[sample_id];
  const float grad_depth = grad_depth_f32[sample_id];
  const bool has_depth = alpha_accum > 1.0e-8f;
  const float adj_alpha_out = has_depth
      ? grad_alpha - grad_depth * depth_weighted / (alpha_accum * alpha_accum)
      : grad_alpha;
  const float adj_depth_weighted_out = has_depth ? grad_depth / alpha_accum : 0.0f;
  float adj_next_transmittance = 0.0f;
  for (int run_id = int(local_run_count) - 1; run_id >= 0; --run_id) {
    const uint owner = owners[run_id];
    const uint rgba_base = owner * 4u;
    const float raw_density = site_rgba_f32[rgba_base + 3u];
    const float density = max(raw_density, 0.0f);
    const float adj_weight = dot(grad_rgb, segment_rgb[run_id]) + adj_alpha_out;
    const float adj_depth_mass = adj_depth_weighted_out * trans_before[run_id];
    const float adj_trans_before =
        adj_weight * segment_alpha[run_id] +
        adj_depth_weighted_out * depth_masses[run_id] +
        adj_next_transmittance * segment_trans[run_id];
    const float adj_segment_alpha = adj_weight * trans_before[run_id];
    float adj_segment_trans = adj_next_transmittance * trans_before[run_id] - adj_segment_alpha;

    float4 grad_rgba = float4(
        weights[run_id] * grad_rgb.x,
        weights[run_id] * grad_rgb.y,
        weights[run_id] * grad_rgb.z,
        0.0f);
    if (raw_density > 0.0f) {
      const float length = lengths[run_id];
      const float dtrans_dd = -length * segment_trans[run_id];
      const float dmass_dd = wf2_continuous_depth_mass_grad_density(
          starts[run_id],
          length,
          density,
          segment_trans[run_id],
          segment_alpha[run_id]);
      grad_rgba.w = adj_segment_trans * dtrans_dd + adj_depth_mass * dmass_dd;
    }
    wf2_atomic_add4(grad_site_rgba_f32, rgba_base, grad_rgba);
    adj_next_transmittance = adj_trans_before;
  }
}

kernel void wf2_endpoint_record_edit_rgba_depth_replay_tensor(
    device const float* boundary_f32 [[buffer(0)]],
    device const float* rays_f32 [[buffer(1)]],
    device const float* frame_t_f32 [[buffer(2)]],
    device const int* base_offsets_i32 [[buffer(3)]],
    device const int* base_owner_i32 [[buffer(4)]],
    device const int* base_left_i32 [[buffer(5)]],
    device const int* base_right_i32 [[buffer(6)]],
    device const int* track_change_offsets_i32 [[buffer(7)]],
    device const int* change_frame_i32 [[buffer(8)]],
    device const int* op_offsets_i32 [[buffer(9)]],
    device const int* op_type_i32 [[buffer(10)]],
    device const int* op_pos_i32 [[buffer(11)]],
    device const int* op_owner_i32 [[buffer(12)]],
    device const int* op_left_i32 [[buffer(13)]],
    device const int* op_right_i32 [[buffer(14)]],
    device const float* site_rgba_f32 [[buffer(15)]],
    device const int* config_i32 [[buffer(16)]],
    device const float* config_f32 [[buffer(17)]],
    device float* output_rgb_f32 [[buffer(18)]],
    device float* output_alpha_f32 [[buffer(19)]],
    device float* output_depth_f32 [[buffer(20)]],
    uint sample_id [[thread_position_in_grid]]) {
  const uint boundary_count = uint(config_i32[0]);
  const uint track_count = uint(config_i32[1]);
  const uint frame_count = uint(config_i32[2]);
  const uint site_count = uint(config_i32[3]);
  const uint base_record_count = uint(config_i32[4]);
  const uint change_count = uint(config_i32[5]);
  const uint op_count = uint(config_i32[6]);
  const uint total_samples = track_count * frame_count;
  if (sample_id >= total_samples) {
    return;
  }

  const uint track_id = sample_id / frame_count;
  const uint frame_id = sample_id - track_id * frame_count;
  const float near_depth = config_f32[0];
  const float far_depth = config_f32[1];
  const float invalid_epsilon = config_f32[2];
  const float transmittance_threshold = config_f32[3];
  const uint ray_base = sample_id * 6u;
  const float3 origin = float3(rays_f32[ray_base + 0u], rays_f32[ray_base + 1u], rays_f32[ray_base + 2u]);
  const float3 direction = float3(rays_f32[ray_base + 3u], rays_f32[ray_base + 4u], rays_f32[ray_base + 5u]);
  const float t = frame_t_f32[frame_id];

  int row_owner[WF2_MAX_REALRAY_SEGMENTS];
  int row_left[WF2_MAX_REALRAY_SEGMENTS];
  int row_right[WF2_MAX_REALRAY_SEGMENTS];
  uint row_count = 0u;
  if (!wf2_endpoint_record_load_edit_row(
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
          track_id,
          frame_id,
          base_record_count,
          change_count,
          op_count,
          row_owner,
          row_left,
          row_right,
          row_count)) {
    return;
  }

  float3 rgb_accum = float3(0.0f, 0.0f, 0.0f);
  float alpha_accum = 0.0f;
  float depth_weighted = 0.0f;
  float transmittance = 1.0f;
  int cached_right_cut = -2147483648;
  float cached_right_depth = 0.0f;
  bool cached_right_valid = false;
  for (uint cursor = 0u; cursor < row_count; ++cursor) {
    if (transmittance <= transmittance_threshold) {
      break;
    }
    const int owner_raw = row_owner[cursor];
    if (owner_raw < 0 || uint(owner_raw) >= site_count) {
      continue;
    }
    float start_depth = 0.0f;
    float end_depth = 0.0f;
    const int left_cut = row_left[cursor];
    const int right_cut = row_right[cursor];
    bool start_valid = false;
    if (cached_right_valid && left_cut == cached_right_cut) {
      start_depth = cached_right_depth;
      start_valid = true;
    } else {
      start_valid = wf2_endpoint_record_cut_depth(
          boundary_f32,
          boundary_count,
          left_cut,
          origin,
          direction,
          t,
          near_depth,
          far_depth,
          invalid_epsilon,
          start_depth);
    }
    if (!start_valid ||
        !wf2_endpoint_record_cut_depth(
            boundary_f32,
            boundary_count,
            right_cut,
            origin,
            direction,
            t,
            near_depth,
            far_depth,
            invalid_epsilon,
            end_depth)) {
      cached_right_valid = false;
      continue;
    }
    cached_right_cut = right_cut;
    cached_right_depth = end_depth;
    cached_right_valid = true;
    const float length = end_depth - start_depth;
    if (!(length > 1.0e-8f)) {
      continue;
    }
    const uint owner = uint(owner_raw);
    const uint rgba_base = owner * 4u;
    const float density = max(site_rgba_f32[rgba_base + 3u], 0.0f);
    const float segment_transmittance = exp(-density * length);
    const float segment_alpha = 1.0f - segment_transmittance;
    const float weight = transmittance * segment_alpha;
    const float3 rgb = float3(
        site_rgba_f32[rgba_base + 0u],
        site_rgba_f32[rgba_base + 1u],
        site_rgba_f32[rgba_base + 2u]);
    const float depth_mass = wf2_continuous_depth_mass(
        start_depth,
        length,
        density,
        segment_transmittance,
        segment_alpha);
    rgb_accum += weight * rgb;
    alpha_accum += weight;
    depth_weighted += transmittance * depth_mass;
    transmittance *= segment_transmittance;
  }

  const uint out_base = sample_id * 3u;
  output_rgb_f32[out_base + 0u] = rgb_accum.x;
  output_rgb_f32[out_base + 1u] = rgb_accum.y;
  output_rgb_f32[out_base + 2u] = rgb_accum.z;
  output_alpha_f32[sample_id] = alpha_accum;
  output_depth_f32[sample_id] = alpha_accum > 1.0e-8f ? depth_weighted / alpha_accum : far_depth;
}

kernel void wf2_endpoint_record_edit_block4_rgba_depth_replay_tensor(
    device const float* boundary_f32 [[buffer(0)]],
    device const float* rays_f32 [[buffer(1)]],
    device const float* frame_t_f32 [[buffer(2)]],
    device const int* anchor_offsets_i32 [[buffer(3)]],
    device const int* anchor_owner_i32 [[buffer(4)]],
    device const int* anchor_left_i32 [[buffer(5)]],
    device const int* anchor_right_i32 [[buffer(6)]],
    device const int* track_block_change_offsets_i32 [[buffer(7)]],
    device const int* change_frame_i32 [[buffer(8)]],
    device const int* op_offsets_i32 [[buffer(9)]],
    device const int* op_type_i32 [[buffer(10)]],
    device const int* op_pos_i32 [[buffer(11)]],
    device const int* op_owner_i32 [[buffer(12)]],
    device const int* op_left_i32 [[buffer(13)]],
    device const int* op_right_i32 [[buffer(14)]],
    device const float* site_rgba_f32 [[buffer(15)]],
    device const int* config_i32 [[buffer(16)]],
    device const float* config_f32 [[buffer(17)]],
    device float* output_rgb_f32 [[buffer(18)]],
    device float* output_alpha_f32 [[buffer(19)]],
    device float* output_depth_f32 [[buffer(20)]],
    uint sample_id [[thread_position_in_grid]]) {
  const uint boundary_count = uint(config_i32[0]);
  const uint track_count = uint(config_i32[1]);
  const uint frame_count = uint(config_i32[2]);
  const uint site_count = uint(config_i32[3]);
  const uint anchor_record_count = uint(config_i32[4]);
  const uint change_count = uint(config_i32[5]);
  const uint op_count = uint(config_i32[6]);
  const uint block_size = uint(config_i32[7]);
  const uint block_count = uint(config_i32[8]);
  const uint total_samples = track_count * frame_count;
  if (sample_id >= total_samples) {
    return;
  }

  const uint track_id = sample_id / frame_count;
  const uint frame_id = sample_id - track_id * frame_count;
  const float near_depth = config_f32[0];
  const float far_depth = config_f32[1];
  const float invalid_epsilon = config_f32[2];
  const float transmittance_threshold = config_f32[3];
  const uint ray_base = sample_id * 6u;
  const float3 origin = float3(rays_f32[ray_base + 0u], rays_f32[ray_base + 1u], rays_f32[ray_base + 2u]);
  const float3 direction = float3(rays_f32[ray_base + 3u], rays_f32[ray_base + 4u], rays_f32[ray_base + 5u]);
  const float t = frame_t_f32[frame_id];

  int row_owner[WF2_MAX_REALRAY_SEGMENTS];
  int row_left[WF2_MAX_REALRAY_SEGMENTS];
  int row_right[WF2_MAX_REALRAY_SEGMENTS];
  uint row_count = 0u;
  if (!wf2_endpoint_record_load_block_edit_row(
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
          track_id,
          frame_id,
          block_size,
          block_count,
          anchor_record_count,
          change_count,
          op_count,
          row_owner,
          row_left,
          row_right,
          row_count)) {
    return;
  }

  float3 rgb_accum = float3(0.0f, 0.0f, 0.0f);
  float alpha_accum = 0.0f;
  float depth_weighted = 0.0f;
  float transmittance = 1.0f;
  int cached_right_cut = -2147483648;
  float cached_right_depth = 0.0f;
  bool cached_right_valid = false;
  for (uint cursor = 0u; cursor < row_count; ++cursor) {
    if (transmittance <= transmittance_threshold) {
      break;
    }
    const int owner_raw = row_owner[cursor];
    if (owner_raw < 0 || uint(owner_raw) >= site_count) {
      continue;
    }
    float start_depth = 0.0f;
    float end_depth = 0.0f;
    const int left_cut = row_left[cursor];
    const int right_cut = row_right[cursor];
    bool start_valid = false;
    if (cached_right_valid && left_cut == cached_right_cut) {
      start_depth = cached_right_depth;
      start_valid = true;
    } else {
      start_valid = wf2_endpoint_record_cut_depth(
          boundary_f32,
          boundary_count,
          left_cut,
          origin,
          direction,
          t,
          near_depth,
          far_depth,
          invalid_epsilon,
          start_depth);
    }
    if (!start_valid ||
        !wf2_endpoint_record_cut_depth(
            boundary_f32,
            boundary_count,
            right_cut,
            origin,
            direction,
            t,
            near_depth,
            far_depth,
            invalid_epsilon,
            end_depth)) {
      cached_right_valid = false;
      continue;
    }
    cached_right_cut = right_cut;
    cached_right_depth = end_depth;
    cached_right_valid = true;
    const float length = end_depth - start_depth;
    if (!(length > 1.0e-8f)) {
      continue;
    }
    const uint owner = uint(owner_raw);
    const uint rgba_base = owner * 4u;
    const float density = max(site_rgba_f32[rgba_base + 3u], 0.0f);
    const float segment_transmittance = exp(-density * length);
    const float segment_alpha = 1.0f - segment_transmittance;
    const float weight = transmittance * segment_alpha;
    const float3 rgb = float3(
        site_rgba_f32[rgba_base + 0u],
        site_rgba_f32[rgba_base + 1u],
        site_rgba_f32[rgba_base + 2u]);
    const float depth_mass = wf2_continuous_depth_mass(
        start_depth,
        length,
        density,
        segment_transmittance,
        segment_alpha);
    rgb_accum += weight * rgb;
    alpha_accum += weight;
    depth_weighted += transmittance * depth_mass;
    transmittance *= segment_transmittance;
  }

  const uint out_base = sample_id * 3u;
  output_rgb_f32[out_base + 0u] = rgb_accum.x;
  output_rgb_f32[out_base + 1u] = rgb_accum.y;
  output_rgb_f32[out_base + 2u] = rgb_accum.z;
  output_alpha_f32[sample_id] = alpha_accum;
  output_depth_f32[sample_id] = alpha_accum > 1.0e-8f ? depth_weighted / alpha_accum : far_depth;
}

kernel void wf2_endpoint_record_edit_block_coeff_rgba_depth_replay_tensor(
    device const float* coeff_f32 [[buffer(0)]],
    device const float* frame_t_f32 [[buffer(1)]],
    device const int* anchor_offsets_i32 [[buffer(2)]],
    device const int* anchor_owner_i32 [[buffer(3)]],
    device const int* anchor_left_i32 [[buffer(4)]],
    device const int* anchor_right_i32 [[buffer(5)]],
    device const int* track_block_change_offsets_i32 [[buffer(6)]],
    device const int* change_frame_i32 [[buffer(7)]],
    device const int* op_offsets_i32 [[buffer(8)]],
    device const int* op_type_i32 [[buffer(9)]],
    device const int* op_pos_i32 [[buffer(10)]],
    device const int* op_owner_i32 [[buffer(11)]],
    device const int* op_left_i32 [[buffer(12)]],
    device const int* op_right_i32 [[buffer(13)]],
    device const float* site_rgba_f32 [[buffer(14)]],
    device const int* config_i32 [[buffer(15)]],
    device const float* config_f32 [[buffer(16)]],
    device float* output_rgb_f32 [[buffer(17)]],
    device float* output_alpha_f32 [[buffer(18)]],
    device float* output_depth_f32 [[buffer(19)]],
    uint sample_id [[thread_position_in_grid]]) {
  const uint boundary_count = uint(config_i32[0]);
  const uint track_count = uint(config_i32[1]);
  const uint frame_count = uint(config_i32[2]);
  const uint site_count = uint(config_i32[3]);
  const uint anchor_record_count = uint(config_i32[4]);
  const uint change_count = uint(config_i32[5]);
  const uint op_count = uint(config_i32[6]);
  const uint block_size = uint(config_i32[7]);
  const uint block_count = uint(config_i32[8]);
  const uint total_samples = track_count * frame_count;
  if (sample_id >= total_samples) {
    return;
  }

  const uint track_id = sample_id / frame_count;
  const uint frame_id = sample_id - track_id * frame_count;
  const float near_depth = config_f32[0];
  const float far_depth = config_f32[1];
  const float invalid_epsilon = config_f32[2];
  const float transmittance_threshold = config_f32[3];
  const float t = frame_t_f32[frame_id];

  int row_owner[WF2_MAX_REALRAY_SEGMENTS];
  int row_left[WF2_MAX_REALRAY_SEGMENTS];
  int row_right[WF2_MAX_REALRAY_SEGMENTS];
  uint row_count = 0u;
  if (!wf2_endpoint_record_load_block_edit_row(
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
          track_id,
          frame_id,
          block_size,
          block_count,
          anchor_record_count,
          change_count,
          op_count,
          row_owner,
          row_left,
          row_right,
          row_count)) {
    return;
  }

  float3 rgb_accum = float3(0.0f, 0.0f, 0.0f);
  float alpha_accum = 0.0f;
  float depth_weighted = 0.0f;
  float transmittance = 1.0f;
  int cached_right_cut = -2147483648;
  float cached_right_depth = 0.0f;
  bool cached_right_valid = false;
  for (uint cursor = 0u; cursor < row_count; ++cursor) {
    if (transmittance <= transmittance_threshold) {
      break;
    }
    const int owner_raw = row_owner[cursor];
    if (owner_raw < 0 || uint(owner_raw) >= site_count) {
      continue;
    }
    float start_depth = 0.0f;
    float end_depth = 0.0f;
    const int left_cut = row_left[cursor];
    const int right_cut = row_right[cursor];
    bool start_valid = false;
    if (cached_right_valid && left_cut == cached_right_cut) {
      start_depth = cached_right_depth;
      start_valid = true;
    } else {
      start_valid = wf2_endpoint_record_coeff_cut_depth(
          coeff_f32,
          boundary_count,
          track_id,
          left_cut,
          t,
          near_depth,
          far_depth,
          invalid_epsilon,
          start_depth);
    }
    if (!start_valid ||
        !wf2_endpoint_record_coeff_cut_depth(
            coeff_f32,
            boundary_count,
            track_id,
            right_cut,
            t,
            near_depth,
            far_depth,
            invalid_epsilon,
            end_depth)) {
      cached_right_valid = false;
      continue;
    }
    cached_right_cut = right_cut;
    cached_right_depth = end_depth;
    cached_right_valid = true;
    const float length = end_depth - start_depth;
    if (!(length > 1.0e-8f)) {
      continue;
    }
    const uint owner = uint(owner_raw);
    const uint rgba_base = owner * 4u;
    const float density = max(site_rgba_f32[rgba_base + 3u], 0.0f);
    const float segment_transmittance = exp(-density * length);
    const float segment_alpha = 1.0f - segment_transmittance;
    const float weight = transmittance * segment_alpha;
    const float3 rgb = float3(
        site_rgba_f32[rgba_base + 0u],
        site_rgba_f32[rgba_base + 1u],
        site_rgba_f32[rgba_base + 2u]);
    const float depth_mass = wf2_continuous_depth_mass(
        start_depth,
        length,
        density,
        segment_transmittance,
        segment_alpha);
    rgb_accum += weight * rgb;
    alpha_accum += weight;
    depth_weighted += transmittance * depth_mass;
    transmittance *= segment_transmittance;
  }

  const uint out_base = sample_id * 3u;
  output_rgb_f32[out_base + 0u] = rgb_accum.x;
  output_rgb_f32[out_base + 1u] = rgb_accum.y;
  output_rgb_f32[out_base + 2u] = rgb_accum.z;
  output_alpha_f32[sample_id] = alpha_accum;
  output_depth_f32[sample_id] = alpha_accum > 1.0e-8f ? depth_weighted / alpha_accum : far_depth;
}

kernel void wf2_endpoint_record_edit_block_coeff_rgb_replay_tensor(
    device const float* coeff_f32 [[buffer(0)]],
    device const float* frame_t_f32 [[buffer(1)]],
    device const int* anchor_offsets_i32 [[buffer(2)]],
    device const int* anchor_owner_i32 [[buffer(3)]],
    device const int* anchor_left_i32 [[buffer(4)]],
    device const int* anchor_right_i32 [[buffer(5)]],
    device const int* track_block_change_offsets_i32 [[buffer(6)]],
    device const int* change_frame_i32 [[buffer(7)]],
    device const int* op_offsets_i32 [[buffer(8)]],
    device const int* op_type_i32 [[buffer(9)]],
    device const int* op_pos_i32 [[buffer(10)]],
    device const int* op_owner_i32 [[buffer(11)]],
    device const int* op_left_i32 [[buffer(12)]],
    device const int* op_right_i32 [[buffer(13)]],
    device const float* site_rgba_f32 [[buffer(14)]],
    device const int* config_i32 [[buffer(15)]],
    device const float* config_f32 [[buffer(16)]],
    device float* output_rgb_f32 [[buffer(17)]],
    uint sample_id [[thread_position_in_grid]]) {
  const uint boundary_count = uint(config_i32[0]);
  const uint track_count = uint(config_i32[1]);
  const uint frame_count = uint(config_i32[2]);
  const uint site_count = uint(config_i32[3]);
  const uint anchor_record_count = uint(config_i32[4]);
  const uint change_count = uint(config_i32[5]);
  const uint op_count = uint(config_i32[6]);
  const uint block_size = uint(config_i32[7]);
  const uint block_count = uint(config_i32[8]);
  const uint total_samples = track_count * frame_count;
  if (sample_id >= total_samples) {
    return;
  }

  const uint track_id = sample_id / frame_count;
  const uint frame_id = sample_id - track_id * frame_count;
  const float near_depth = config_f32[0];
  const float far_depth = config_f32[1];
  const float invalid_epsilon = config_f32[2];
  const float transmittance_threshold = config_f32[3];
  const float t = frame_t_f32[frame_id];

  int row_owner[WF2_MAX_REALRAY_SEGMENTS];
  int row_left[WF2_MAX_REALRAY_SEGMENTS];
  int row_right[WF2_MAX_REALRAY_SEGMENTS];
  uint row_count = 0u;
  if (!wf2_endpoint_record_load_block_edit_row(
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
          track_id,
          frame_id,
          block_size,
          block_count,
          anchor_record_count,
          change_count,
          op_count,
          row_owner,
          row_left,
          row_right,
          row_count)) {
    return;
  }

  float3 rgb_accum = float3(0.0f, 0.0f, 0.0f);
  float transmittance = 1.0f;
  int cached_right_cut = -2147483648;
  float cached_right_depth = 0.0f;
  bool cached_right_valid = false;
  for (uint cursor = 0u; cursor < row_count; ++cursor) {
    if (transmittance <= transmittance_threshold) {
      break;
    }
    const int owner_raw = row_owner[cursor];
    if (owner_raw < 0 || uint(owner_raw) >= site_count) {
      continue;
    }
    float start_depth = 0.0f;
    float end_depth = 0.0f;
    const int left_cut = row_left[cursor];
    const int right_cut = row_right[cursor];
    bool start_valid = false;
    if (cached_right_valid && left_cut == cached_right_cut) {
      start_depth = cached_right_depth;
      start_valid = true;
    } else {
      start_valid = wf2_endpoint_record_coeff_cut_depth(
          coeff_f32,
          boundary_count,
          track_id,
          left_cut,
          t,
          near_depth,
          far_depth,
          invalid_epsilon,
          start_depth);
    }
    if (!start_valid ||
        !wf2_endpoint_record_coeff_cut_depth(
            coeff_f32,
            boundary_count,
            track_id,
            right_cut,
            t,
            near_depth,
            far_depth,
            invalid_epsilon,
            end_depth)) {
      cached_right_valid = false;
      continue;
    }
    cached_right_cut = right_cut;
    cached_right_depth = end_depth;
    cached_right_valid = true;
    const float length = end_depth - start_depth;
    if (!(length > 1.0e-8f)) {
      continue;
    }
    const uint owner = uint(owner_raw);
    const uint rgba_base = owner * 4u;
    const float density = max(site_rgba_f32[rgba_base + 3u], 0.0f);
    const float segment_transmittance = exp(-density * length);
    const float segment_alpha = 1.0f - segment_transmittance;
    const float weight = transmittance * segment_alpha;
    const float3 rgb = float3(
        site_rgba_f32[rgba_base + 0u],
        site_rgba_f32[rgba_base + 1u],
        site_rgba_f32[rgba_base + 2u]);
    rgb_accum += weight * rgb;
    transmittance *= segment_transmittance;
  }

  const uint out_base = sample_id * 3u;
  output_rgb_f32[out_base + 0u] = rgb_accum.x;
  output_rgb_f32[out_base + 1u] = rgb_accum.y;
  output_rgb_f32[out_base + 2u] = rgb_accum.z;
}

kernel void wf2_endpoint_record_edit_block_coeff16_rgba_depth_replay_tensor(
    device const half* coeff_f16 [[buffer(0)]],
    device const float* frame_t_f32 [[buffer(1)]],
    device const int* anchor_offsets_i32 [[buffer(2)]],
    device const int* anchor_owner_i32 [[buffer(3)]],
    device const int* anchor_left_i32 [[buffer(4)]],
    device const int* anchor_right_i32 [[buffer(5)]],
    device const int* track_block_change_offsets_i32 [[buffer(6)]],
    device const int* change_frame_i32 [[buffer(7)]],
    device const int* op_offsets_i32 [[buffer(8)]],
    device const int* op_type_i32 [[buffer(9)]],
    device const int* op_pos_i32 [[buffer(10)]],
    device const int* op_owner_i32 [[buffer(11)]],
    device const int* op_left_i32 [[buffer(12)]],
    device const int* op_right_i32 [[buffer(13)]],
    device const float* site_rgba_f32 [[buffer(14)]],
    device const int* config_i32 [[buffer(15)]],
    device const float* config_f32 [[buffer(16)]],
    device float* output_rgb_f32 [[buffer(17)]],
    device float* output_alpha_f32 [[buffer(18)]],
    device float* output_depth_f32 [[buffer(19)]],
    uint sample_id [[thread_position_in_grid]]) {
  const uint boundary_count = uint(config_i32[0]);
  const uint track_count = uint(config_i32[1]);
  const uint frame_count = uint(config_i32[2]);
  const uint site_count = uint(config_i32[3]);
  const uint anchor_record_count = uint(config_i32[4]);
  const uint change_count = uint(config_i32[5]);
  const uint op_count = uint(config_i32[6]);
  const uint block_size = uint(config_i32[7]);
  const uint block_count = uint(config_i32[8]);
  const uint total_samples = track_count * frame_count;
  if (sample_id >= total_samples) {
    return;
  }

  const uint track_id = sample_id / frame_count;
  const uint frame_id = sample_id - track_id * frame_count;
  const float near_depth = config_f32[0];
  const float far_depth = config_f32[1];
  const float invalid_epsilon = config_f32[2];
  const float transmittance_threshold = config_f32[3];
  const float t = frame_t_f32[frame_id];

  int row_owner[WF2_MAX_REALRAY_SEGMENTS];
  int row_left[WF2_MAX_REALRAY_SEGMENTS];
  int row_right[WF2_MAX_REALRAY_SEGMENTS];
  uint row_count = 0u;
  if (!wf2_endpoint_record_load_block_edit_row(
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
          track_id,
          frame_id,
          block_size,
          block_count,
          anchor_record_count,
          change_count,
          op_count,
          row_owner,
          row_left,
          row_right,
          row_count)) {
    return;
  }

  float3 rgb_accum = float3(0.0f, 0.0f, 0.0f);
  float alpha_accum = 0.0f;
  float depth_weighted = 0.0f;
  float transmittance = 1.0f;
  int cached_right_cut = -2147483648;
  float cached_right_depth = 0.0f;
  bool cached_right_valid = false;
  for (uint cursor = 0u; cursor < row_count; ++cursor) {
    if (transmittance <= transmittance_threshold) {
      break;
    }
    const int owner_raw = row_owner[cursor];
    if (owner_raw < 0 || uint(owner_raw) >= site_count) {
      continue;
    }
    float start_depth = 0.0f;
    float end_depth = 0.0f;
    const int left_cut = row_left[cursor];
    const int right_cut = row_right[cursor];
    bool start_valid = false;
    if (cached_right_valid && left_cut == cached_right_cut) {
      start_depth = cached_right_depth;
      start_valid = true;
    } else {
      start_valid = wf2_endpoint_record_coeff16_cut_depth(
          coeff_f16,
          boundary_count,
          track_id,
          left_cut,
          t,
          near_depth,
          far_depth,
          invalid_epsilon,
          start_depth);
    }
    if (!start_valid ||
        !wf2_endpoint_record_coeff16_cut_depth(
            coeff_f16,
            boundary_count,
            track_id,
            right_cut,
            t,
            near_depth,
            far_depth,
            invalid_epsilon,
            end_depth)) {
      cached_right_valid = false;
      continue;
    }
    cached_right_cut = right_cut;
    cached_right_depth = end_depth;
    cached_right_valid = true;
    const float length = end_depth - start_depth;
    if (!(length > 1.0e-8f)) {
      continue;
    }
    const uint owner = uint(owner_raw);
    const uint rgba_base = owner * 4u;
    const float density = max(site_rgba_f32[rgba_base + 3u], 0.0f);
    const float segment_transmittance = exp(-density * length);
    const float segment_alpha = 1.0f - segment_transmittance;
    const float weight = transmittance * segment_alpha;
    const float3 rgb = float3(
        site_rgba_f32[rgba_base + 0u],
        site_rgba_f32[rgba_base + 1u],
        site_rgba_f32[rgba_base + 2u]);
    const float depth_mass = wf2_continuous_depth_mass(
        start_depth,
        length,
        density,
        segment_transmittance,
        segment_alpha);
    rgb_accum += weight * rgb;
    alpha_accum += weight;
    depth_weighted += transmittance * depth_mass;
    transmittance *= segment_transmittance;
  }

  const uint out_base = sample_id * 3u;
  output_rgb_f32[out_base + 0u] = rgb_accum.x;
  output_rgb_f32[out_base + 1u] = rgb_accum.y;
  output_rgb_f32[out_base + 2u] = rgb_accum.z;
  output_alpha_f32[sample_id] = alpha_accum;
  output_depth_f32[sample_id] = alpha_accum > 1.0e-8f ? depth_weighted / alpha_accum : far_depth;
}

kernel void wf2_endpoint_record_edit_rgba_depth_replay_trackloop_tensor(
    device const float* boundary_f32 [[buffer(0)]],
    device const float* rays_f32 [[buffer(1)]],
    device const float* frame_t_f32 [[buffer(2)]],
    device const int* base_offsets_i32 [[buffer(3)]],
    device const int* base_owner_i32 [[buffer(4)]],
    device const int* base_left_i32 [[buffer(5)]],
    device const int* base_right_i32 [[buffer(6)]],
    device const int* track_change_offsets_i32 [[buffer(7)]],
    device const int* change_frame_i32 [[buffer(8)]],
    device const int* op_offsets_i32 [[buffer(9)]],
    device const int* op_type_i32 [[buffer(10)]],
    device const int* op_pos_i32 [[buffer(11)]],
    device const int* op_owner_i32 [[buffer(12)]],
    device const int* op_left_i32 [[buffer(13)]],
    device const int* op_right_i32 [[buffer(14)]],
    device const float* site_rgba_f32 [[buffer(15)]],
    device const int* config_i32 [[buffer(16)]],
    device const float* config_f32 [[buffer(17)]],
    device float* output_rgb_f32 [[buffer(18)]],
    device float* output_alpha_f32 [[buffer(19)]],
    device float* output_depth_f32 [[buffer(20)]],
    uint track_id [[thread_position_in_grid]]) {
  const uint boundary_count = uint(config_i32[0]);
  const uint track_count = uint(config_i32[1]);
  const uint frame_count = uint(config_i32[2]);
  const uint site_count = uint(config_i32[3]);
  const uint base_record_count = uint(config_i32[4]);
  const uint change_count = uint(config_i32[5]);
  const uint op_count = uint(config_i32[6]);
  if (track_id >= track_count) {
    return;
  }

  const float near_depth = config_f32[0];
  const float far_depth = config_f32[1];
  const float invalid_epsilon = config_f32[2];
  const float transmittance_threshold = config_f32[3];

  int row_owner[WF2_MAX_REALRAY_SEGMENTS];
  int row_left[WF2_MAX_REALRAY_SEGMENTS];
  int row_right[WF2_MAX_REALRAY_SEGMENTS];
  uint row_count = 0u;
  const int base_begin_raw = base_offsets_i32[track_id];
  const int base_end_raw = base_offsets_i32[track_id + 1u];
  if (base_begin_raw < 0 || base_end_raw < base_begin_raw || uint(base_end_raw) > base_record_count) {
    return;
  }
  for (uint cursor = uint(base_begin_raw); cursor < uint(base_end_raw); ++cursor) {
    if (row_count >= WF2_MAX_REALRAY_SEGMENTS) {
      return;
    }
    row_owner[row_count] = base_owner_i32[cursor];
    row_left[row_count] = base_left_i32[cursor];
    row_right[row_count] = base_right_i32[cursor];
    row_count += 1u;
  }

  const int change_begin_raw = track_change_offsets_i32[track_id];
  const int change_end_raw = track_change_offsets_i32[track_id + 1u];
  if (change_begin_raw < 0 || change_end_raw < change_begin_raw || uint(change_end_raw) > change_count) {
    return;
  }
  uint change_cursor = uint(change_begin_raw);
  const uint change_end = uint(change_end_raw);

  for (uint frame_id = 0u; frame_id < frame_count; ++frame_id) {
    while (change_cursor < change_end) {
      const int changed_frame = change_frame_i32[change_cursor];
      if (changed_frame < 0) {
        change_cursor += 1u;
        continue;
      }
      if (uint(changed_frame) > frame_id) {
        break;
      }
      const int op_begin_raw = op_offsets_i32[change_cursor];
      const int op_end_raw = op_offsets_i32[change_cursor + 1u];
      if (op_begin_raw < 0 || op_end_raw < op_begin_raw || uint(op_end_raw) > op_count) {
        return;
      }
      for (uint op_cursor = uint(op_begin_raw); op_cursor < uint(op_end_raw); ++op_cursor) {
        const int op_type = op_type_i32[op_cursor];
        const int pos_raw = op_pos_i32[op_cursor];
        if (pos_raw < 0) {
          return;
        }
        const uint pos = uint(pos_raw);
        if (op_type == 0) {
          if (pos > row_count || row_count >= WF2_MAX_REALRAY_SEGMENTS) {
            return;
          }
          for (uint shift = row_count; shift > pos; --shift) {
            row_owner[shift] = row_owner[shift - 1u];
            row_left[shift] = row_left[shift - 1u];
            row_right[shift] = row_right[shift - 1u];
          }
          row_owner[pos] = op_owner_i32[op_cursor];
          row_left[pos] = op_left_i32[op_cursor];
          row_right[pos] = op_right_i32[op_cursor];
          row_count += 1u;
        } else if (op_type == 1) {
          if (pos >= row_count) {
            return;
          }
          for (uint shift = pos; shift + 1u < row_count; ++shift) {
            row_owner[shift] = row_owner[shift + 1u];
            row_left[shift] = row_left[shift + 1u];
            row_right[shift] = row_right[shift + 1u];
          }
          row_count -= 1u;
        } else if (op_type == 2) {
          if (pos >= row_count) {
            return;
          }
          row_owner[pos] = op_owner_i32[op_cursor];
          row_left[pos] = op_left_i32[op_cursor];
          row_right[pos] = op_right_i32[op_cursor];
        } else {
          return;
        }
      }
      change_cursor += 1u;
    }

    const uint sample_id = track_id * frame_count + frame_id;
    const uint ray_base = sample_id * 6u;
    const float3 origin = float3(rays_f32[ray_base + 0u], rays_f32[ray_base + 1u], rays_f32[ray_base + 2u]);
    const float3 direction = float3(rays_f32[ray_base + 3u], rays_f32[ray_base + 4u], rays_f32[ray_base + 5u]);
    const float t = frame_t_f32[frame_id];
    float3 rgb_accum = float3(0.0f, 0.0f, 0.0f);
    float alpha_accum = 0.0f;
    float depth_weighted = 0.0f;
    float transmittance = 1.0f;
    int cached_right_cut = -2147483648;
    float cached_right_depth = 0.0f;
    bool cached_right_valid = false;
    for (uint cursor = 0u; cursor < row_count; ++cursor) {
      if (transmittance <= transmittance_threshold) {
        break;
      }
      const int owner_raw = row_owner[cursor];
      if (owner_raw < 0 || uint(owner_raw) >= site_count) {
        continue;
      }
      float start_depth = 0.0f;
      float end_depth = 0.0f;
      const int left_cut = row_left[cursor];
      const int right_cut = row_right[cursor];
      bool start_valid = false;
      if (cached_right_valid && left_cut == cached_right_cut) {
        start_depth = cached_right_depth;
        start_valid = true;
      } else {
        start_valid = wf2_endpoint_record_cut_depth(
            boundary_f32,
            boundary_count,
            left_cut,
            origin,
            direction,
            t,
            near_depth,
            far_depth,
            invalid_epsilon,
            start_depth);
      }
      if (!start_valid ||
          !wf2_endpoint_record_cut_depth(
              boundary_f32,
              boundary_count,
              right_cut,
              origin,
              direction,
              t,
              near_depth,
              far_depth,
              invalid_epsilon,
              end_depth)) {
        cached_right_valid = false;
        continue;
      }
      cached_right_cut = right_cut;
      cached_right_depth = end_depth;
      cached_right_valid = true;
      const float length = end_depth - start_depth;
      if (!(length > 1.0e-8f)) {
        continue;
      }
      const uint owner = uint(owner_raw);
      const uint rgba_base = owner * 4u;
      const float density = max(site_rgba_f32[rgba_base + 3u], 0.0f);
      const float segment_transmittance = exp(-density * length);
      const float segment_alpha = 1.0f - segment_transmittance;
      const float weight = transmittance * segment_alpha;
      const float3 rgb = float3(
          site_rgba_f32[rgba_base + 0u],
          site_rgba_f32[rgba_base + 1u],
          site_rgba_f32[rgba_base + 2u]);
      const float depth_mass = wf2_continuous_depth_mass(
          start_depth,
          length,
          density,
          segment_transmittance,
          segment_alpha);
      rgb_accum += weight * rgb;
      alpha_accum += weight;
      depth_weighted += transmittance * depth_mass;
      transmittance *= segment_transmittance;
    }

    const uint out_base = sample_id * 3u;
    output_rgb_f32[out_base + 0u] = rgb_accum.x;
    output_rgb_f32[out_base + 1u] = rgb_accum.y;
    output_rgb_f32[out_base + 2u] = rgb_accum.z;
    output_alpha_f32[sample_id] = alpha_accum;
    output_depth_f32[sample_id] = alpha_accum > 1.0e-8f ? depth_weighted / alpha_accum : far_depth;
  }
}

kernel void wf2_endpoint_record_edit_rgba_depth_replay_framegroup16_tensor(
    device const float* boundary_f32 [[buffer(0)]],
    device const float* rays_f32 [[buffer(1)]],
    device const float* frame_t_f32 [[buffer(2)]],
    device const int* base_offsets_i32 [[buffer(3)]],
    device const int* base_owner_i32 [[buffer(4)]],
    device const int* base_left_i32 [[buffer(5)]],
    device const int* base_right_i32 [[buffer(6)]],
    device const int* track_change_offsets_i32 [[buffer(7)]],
    device const int* change_frame_i32 [[buffer(8)]],
    device const int* op_offsets_i32 [[buffer(9)]],
    device const int* op_type_i32 [[buffer(10)]],
    device const int* op_pos_i32 [[buffer(11)]],
    device const int* op_owner_i32 [[buffer(12)]],
    device const int* op_left_i32 [[buffer(13)]],
    device const int* op_right_i32 [[buffer(14)]],
    device const float* site_rgba_f32 [[buffer(15)]],
    device const int* config_i32 [[buffer(16)]],
    device const float* config_f32 [[buffer(17)]],
    device float* output_rgb_f32 [[buffer(18)]],
    device float* output_alpha_f32 [[buffer(19)]],
    device float* output_depth_f32 [[buffer(20)]],
    uint3 local_pos [[thread_position_in_threadgroup]],
    uint3 group_id [[threadgroup_position_in_grid]]) {
  const uint local_frame = local_pos.x;
  const uint boundary_count = uint(config_i32[0]);
  const uint track_count = uint(config_i32[1]);
  const uint frame_count = uint(config_i32[2]);
  const uint site_count = uint(config_i32[3]);
  const uint base_record_count = uint(config_i32[4]);
  const uint change_count = uint(config_i32[5]);
  const uint op_count = uint(config_i32[6]);
  const uint track_id = group_id.x;

  threadgroup int tg_valid[WF2_ENDPOINT_EDIT_FRAMEGROUP_MAX_FRAMES];
  threadgroup uint tg_count[WF2_ENDPOINT_EDIT_FRAMEGROUP_MAX_FRAMES];
  threadgroup int tg_owner[WF2_ENDPOINT_EDIT_FRAMEGROUP_MAX_FRAMES * WF2_MAX_REALRAY_SEGMENTS];
  threadgroup int tg_left[WF2_ENDPOINT_EDIT_FRAMEGROUP_MAX_FRAMES * WF2_MAX_REALRAY_SEGMENTS];
  threadgroup int tg_right[WF2_ENDPOINT_EDIT_FRAMEGROUP_MAX_FRAMES * WF2_MAX_REALRAY_SEGMENTS];

  if (local_frame == 0u) {
    for (uint frame_id = 0u; frame_id < WF2_ENDPOINT_EDIT_FRAMEGROUP_MAX_FRAMES; ++frame_id) {
      tg_valid[frame_id] = 0;
      tg_count[frame_id] = 0u;
    }

    bool ok = track_id < track_count && frame_count <= WF2_ENDPOINT_EDIT_FRAMEGROUP_MAX_FRAMES;
    int row_owner[WF2_MAX_REALRAY_SEGMENTS];
    int row_left[WF2_MAX_REALRAY_SEGMENTS];
    int row_right[WF2_MAX_REALRAY_SEGMENTS];
    uint row_count = 0u;

    if (ok) {
      const int base_begin_raw = base_offsets_i32[track_id];
      const int base_end_raw = base_offsets_i32[track_id + 1u];
      ok = base_begin_raw >= 0 && base_end_raw >= base_begin_raw && uint(base_end_raw) <= base_record_count;
      if (ok) {
        for (uint cursor = uint(base_begin_raw); cursor < uint(base_end_raw); ++cursor) {
          if (row_count >= WF2_MAX_REALRAY_SEGMENTS) {
            ok = false;
            break;
          }
          row_owner[row_count] = base_owner_i32[cursor];
          row_left[row_count] = base_left_i32[cursor];
          row_right[row_count] = base_right_i32[cursor];
          row_count += 1u;
        }
      }
    }

    int change_begin_raw = 0;
    int change_end_raw = 0;
    if (ok) {
      change_begin_raw = track_change_offsets_i32[track_id];
      change_end_raw = track_change_offsets_i32[track_id + 1u];
      ok = change_begin_raw >= 0 && change_end_raw >= change_begin_raw && uint(change_end_raw) <= change_count;
    }
    uint change_cursor = ok ? uint(change_begin_raw) : 0u;
    const uint change_end = ok ? uint(change_end_raw) : 0u;

    for (uint frame_id = 0u; frame_id < frame_count; ++frame_id) {
      while (ok && change_cursor < change_end) {
        const int changed_frame = change_frame_i32[change_cursor];
        if (changed_frame < 0) {
          change_cursor += 1u;
          continue;
        }
        if (uint(changed_frame) > frame_id) {
          break;
        }
        const int op_begin_raw = op_offsets_i32[change_cursor];
        const int op_end_raw = op_offsets_i32[change_cursor + 1u];
        ok = op_begin_raw >= 0 && op_end_raw >= op_begin_raw && uint(op_end_raw) <= op_count;
        for (uint op_cursor = ok ? uint(op_begin_raw) : 0u; ok && op_cursor < uint(op_end_raw); ++op_cursor) {
          const int op_type = op_type_i32[op_cursor];
          const int pos_raw = op_pos_i32[op_cursor];
          if (pos_raw < 0) {
            ok = false;
            break;
          }
          const uint pos = uint(pos_raw);
          if (op_type == 0) {
            if (pos > row_count || row_count >= WF2_MAX_REALRAY_SEGMENTS) {
              ok = false;
              break;
            }
            for (uint shift = row_count; shift > pos; --shift) {
              row_owner[shift] = row_owner[shift - 1u];
              row_left[shift] = row_left[shift - 1u];
              row_right[shift] = row_right[shift - 1u];
            }
            row_owner[pos] = op_owner_i32[op_cursor];
            row_left[pos] = op_left_i32[op_cursor];
            row_right[pos] = op_right_i32[op_cursor];
            row_count += 1u;
          } else if (op_type == 1) {
            if (pos >= row_count) {
              ok = false;
              break;
            }
            for (uint shift = pos; shift + 1u < row_count; ++shift) {
              row_owner[shift] = row_owner[shift + 1u];
              row_left[shift] = row_left[shift + 1u];
              row_right[shift] = row_right[shift + 1u];
            }
            row_count -= 1u;
          } else if (op_type == 2) {
            if (pos >= row_count) {
              ok = false;
              break;
            }
            row_owner[pos] = op_owner_i32[op_cursor];
            row_left[pos] = op_left_i32[op_cursor];
            row_right[pos] = op_right_i32[op_cursor];
          } else {
            ok = false;
            break;
          }
        }
        if (ok) {
          change_cursor += 1u;
        }
      }

      if (ok) {
        const uint frame_base = frame_id * WF2_MAX_REALRAY_SEGMENTS;
        tg_count[frame_id] = row_count;
        tg_valid[frame_id] = 1;
        for (uint cursor = 0u; cursor < row_count; ++cursor) {
          const uint out = frame_base + cursor;
          tg_owner[out] = row_owner[cursor];
          tg_left[out] = row_left[cursor];
          tg_right[out] = row_right[cursor];
        }
      }
    }
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (track_id >= track_count || local_frame >= frame_count ||
      frame_count > WF2_ENDPOINT_EDIT_FRAMEGROUP_MAX_FRAMES) {
    return;
  }

  const uint sample_id = track_id * frame_count + local_frame;
  const uint out_base = sample_id * 3u;
  if (tg_valid[local_frame] == 0) {
    output_rgb_f32[out_base + 0u] = 0.0f;
    output_rgb_f32[out_base + 1u] = 0.0f;
    output_rgb_f32[out_base + 2u] = 0.0f;
    output_alpha_f32[sample_id] = 0.0f;
    output_depth_f32[sample_id] = config_f32[1];
    return;
  }

  const float near_depth = config_f32[0];
  const float far_depth = config_f32[1];
  const float invalid_epsilon = config_f32[2];
  const float transmittance_threshold = config_f32[3];
  const uint ray_base = sample_id * 6u;
  const float3 origin = float3(rays_f32[ray_base + 0u], rays_f32[ray_base + 1u], rays_f32[ray_base + 2u]);
  const float3 direction = float3(rays_f32[ray_base + 3u], rays_f32[ray_base + 4u], rays_f32[ray_base + 5u]);
  const float t = frame_t_f32[local_frame];
  const uint row_count = tg_count[local_frame];
  const uint frame_base = local_frame * WF2_MAX_REALRAY_SEGMENTS;

  float3 rgb_accum = float3(0.0f, 0.0f, 0.0f);
  float alpha_accum = 0.0f;
  float depth_weighted = 0.0f;
  float transmittance = 1.0f;
  int cached_right_cut = -2147483648;
  float cached_right_depth = 0.0f;
  bool cached_right_valid = false;
  for (uint cursor = 0u; cursor < row_count; ++cursor) {
    if (transmittance <= transmittance_threshold) {
      break;
    }
    const uint row_index = frame_base + cursor;
    const int owner_raw = tg_owner[row_index];
    if (owner_raw < 0 || uint(owner_raw) >= site_count) {
      continue;
    }
    float start_depth = 0.0f;
    float end_depth = 0.0f;
    const int left_cut = tg_left[row_index];
    const int right_cut = tg_right[row_index];
    bool start_valid = false;
    if (cached_right_valid && left_cut == cached_right_cut) {
      start_depth = cached_right_depth;
      start_valid = true;
    } else {
      start_valid = wf2_endpoint_record_cut_depth(
          boundary_f32,
          boundary_count,
          left_cut,
          origin,
          direction,
          t,
          near_depth,
          far_depth,
          invalid_epsilon,
          start_depth);
    }
    if (!start_valid ||
        !wf2_endpoint_record_cut_depth(
            boundary_f32,
            boundary_count,
            right_cut,
            origin,
            direction,
            t,
            near_depth,
            far_depth,
            invalid_epsilon,
            end_depth)) {
      cached_right_valid = false;
      continue;
    }
    cached_right_cut = right_cut;
    cached_right_depth = end_depth;
    cached_right_valid = true;
    const float length = end_depth - start_depth;
    if (!(length > 1.0e-8f)) {
      continue;
    }
    const uint owner = uint(owner_raw);
    const uint rgba_base = owner * 4u;
    const float density = max(site_rgba_f32[rgba_base + 3u], 0.0f);
    const float segment_transmittance = exp(-density * length);
    const float segment_alpha = 1.0f - segment_transmittance;
    const float weight = transmittance * segment_alpha;
    const float3 rgb = float3(
        site_rgba_f32[rgba_base + 0u],
        site_rgba_f32[rgba_base + 1u],
        site_rgba_f32[rgba_base + 2u]);
    const float depth_mass = wf2_continuous_depth_mass(
        start_depth,
        length,
        density,
        segment_transmittance,
        segment_alpha);
    rgb_accum += weight * rgb;
    alpha_accum += weight;
    depth_weighted += transmittance * depth_mass;
    transmittance *= segment_transmittance;
  }

  output_rgb_f32[out_base + 0u] = rgb_accum.x;
  output_rgb_f32[out_base + 1u] = rgb_accum.y;
  output_rgb_f32[out_base + 2u] = rgb_accum.z;
  output_alpha_f32[sample_id] = alpha_accum;
  output_depth_f32[sample_id] = alpha_accum > 1.0e-8f ? depth_weighted / alpha_accum : far_depth;
}

kernel void wf2_endpoint_record_edit_vjp_direct_atomic_grad_only_tensor(
    device const float* boundary_f32 [[buffer(0)]],
    device const float* rays_f32 [[buffer(1)]],
    device const float* frame_t_f32 [[buffer(2)]],
    device const int* base_offsets_i32 [[buffer(3)]],
    device const int* base_owner_i32 [[buffer(4)]],
    device const int* base_left_i32 [[buffer(5)]],
    device const int* base_right_i32 [[buffer(6)]],
    device const int* track_change_offsets_i32 [[buffer(7)]],
    device const int* change_frame_i32 [[buffer(8)]],
    device const int* op_offsets_i32 [[buffer(9)]],
    device const int* op_type_i32 [[buffer(10)]],
    device const int* op_pos_i32 [[buffer(11)]],
    device const int* op_owner_i32 [[buffer(12)]],
    device const int* op_left_i32 [[buffer(13)]],
    device const int* op_right_i32 [[buffer(14)]],
    device const float* site_rgba_f32 [[buffer(15)]],
    device const float* grad_rgb_f32 [[buffer(16)]],
    device const float* grad_alpha_f32 [[buffer(17)]],
    device const float* grad_depth_f32 [[buffer(18)]],
    device const int* config_i32 [[buffer(19)]],
    device const float* config_f32 [[buffer(20)]],
    device atomic_float* grad_site_rgba_f32 [[buffer(21)]],
    uint sample_id [[thread_position_in_grid]]) {
  const uint boundary_count = uint(config_i32[0]);
  const uint track_count = uint(config_i32[1]);
  const uint frame_count = uint(config_i32[2]);
  const uint site_count = uint(config_i32[3]);
  const uint base_record_count = uint(config_i32[4]);
  const uint change_count = uint(config_i32[5]);
  const uint op_count = uint(config_i32[6]);
  const uint total_samples = track_count * frame_count;
  if (sample_id >= total_samples) {
    return;
  }

  const uint track_id = sample_id / frame_count;
  const uint frame_id = sample_id - track_id * frame_count;
  const float near_depth = config_f32[0];
  const float far_depth = config_f32[1];
  const float invalid_epsilon = config_f32[2];
  const float transmittance_threshold = config_f32[3];
  const uint ray_base = sample_id * 6u;
  const float3 origin = float3(rays_f32[ray_base + 0u], rays_f32[ray_base + 1u], rays_f32[ray_base + 2u]);
  const float3 direction = float3(rays_f32[ray_base + 3u], rays_f32[ray_base + 4u], rays_f32[ray_base + 5u]);
  const float t = frame_t_f32[frame_id];

  int row_owner[WF2_MAX_REALRAY_SEGMENTS];
  int row_left[WF2_MAX_REALRAY_SEGMENTS];
  int row_right[WF2_MAX_REALRAY_SEGMENTS];
  uint row_count = 0u;
  if (!wf2_endpoint_record_load_edit_row(
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
          track_id,
          frame_id,
          base_record_count,
          change_count,
          op_count,
          row_owner,
          row_left,
          row_right,
          row_count)) {
    return;
  }

  uint owners[WF2_MAX_REALRAY_SEGMENTS];
  float starts[WF2_MAX_REALRAY_SEGMENTS];
  float lengths[WF2_MAX_REALRAY_SEGMENTS];
  float depth_masses[WF2_MAX_REALRAY_SEGMENTS];
  float trans_before[WF2_MAX_REALRAY_SEGMENTS];
  float segment_trans[WF2_MAX_REALRAY_SEGMENTS];
  float segment_alpha[WF2_MAX_REALRAY_SEGMENTS];
  float weights[WF2_MAX_REALRAY_SEGMENTS];
  float3 segment_rgb[WF2_MAX_REALRAY_SEGMENTS];

  float alpha_accum = 0.0f;
  float depth_weighted = 0.0f;
  float transmittance = 1.0f;
  uint local_run_count = 0u;
  int cached_right_cut = -2147483648;
  float cached_right_depth = 0.0f;
  bool cached_right_valid = false;
  for (uint cursor = 0u; cursor < row_count; ++cursor) {
    if (transmittance <= transmittance_threshold || local_run_count >= WF2_MAX_REALRAY_SEGMENTS) {
      break;
    }
    const int owner_raw = row_owner[cursor];
    if (owner_raw < 0 || uint(owner_raw) >= site_count) {
      continue;
    }
    float start_depth = 0.0f;
    float end_depth = 0.0f;
    const int left_cut = row_left[cursor];
    const int right_cut = row_right[cursor];
    bool start_valid = false;
    if (cached_right_valid && left_cut == cached_right_cut) {
      start_depth = cached_right_depth;
      start_valid = true;
    } else {
      start_valid = wf2_endpoint_record_cut_depth(
          boundary_f32,
          boundary_count,
          left_cut,
          origin,
          direction,
          t,
          near_depth,
          far_depth,
          invalid_epsilon,
          start_depth);
    }
    if (!start_valid ||
        !wf2_endpoint_record_cut_depth(
            boundary_f32,
            boundary_count,
            right_cut,
            origin,
            direction,
            t,
            near_depth,
            far_depth,
            invalid_epsilon,
            end_depth)) {
      cached_right_valid = false;
      continue;
    }
    cached_right_cut = right_cut;
    cached_right_depth = end_depth;
    cached_right_valid = true;
    const float length = end_depth - start_depth;
    if (!(length > 1.0e-8f)) {
      continue;
    }
    const uint owner = uint(owner_raw);
    const uint rgba_base = owner * 4u;
    const float density = max(site_rgba_f32[rgba_base + 3u], 0.0f);
    const float seg_trans = exp(-density * length);
    const float seg_alpha = 1.0f - seg_trans;
    const float weight = transmittance * seg_alpha;
    const float3 rgb = float3(
        site_rgba_f32[rgba_base + 0u],
        site_rgba_f32[rgba_base + 1u],
        site_rgba_f32[rgba_base + 2u]);
    const float depth_mass = wf2_continuous_depth_mass(
        start_depth,
        length,
        density,
        seg_trans,
        seg_alpha);

    owners[local_run_count] = owner;
    starts[local_run_count] = start_depth;
    lengths[local_run_count] = length;
    depth_masses[local_run_count] = depth_mass;
    trans_before[local_run_count] = transmittance;
    segment_trans[local_run_count] = seg_trans;
    segment_alpha[local_run_count] = seg_alpha;
    weights[local_run_count] = weight;
    segment_rgb[local_run_count] = rgb;
    local_run_count += 1u;

    alpha_accum += weight;
    depth_weighted += transmittance * depth_mass;
    transmittance *= seg_trans;
  }

  const uint out_base = sample_id * 3u;
  const float3 grad_rgb = float3(
      grad_rgb_f32[out_base + 0u],
      grad_rgb_f32[out_base + 1u],
      grad_rgb_f32[out_base + 2u]);
  const float grad_alpha = grad_alpha_f32[sample_id];
  const float grad_depth = grad_depth_f32[sample_id];
  const bool has_depth = alpha_accum > 1.0e-8f;
  const float adj_alpha_out = has_depth
      ? grad_alpha - grad_depth * depth_weighted / (alpha_accum * alpha_accum)
      : grad_alpha;
  const float adj_depth_weighted_out = has_depth ? grad_depth / alpha_accum : 0.0f;
  float adj_next_transmittance = 0.0f;
  for (int run_id = int(local_run_count) - 1; run_id >= 0; --run_id) {
    const uint owner = owners[run_id];
    const uint rgba_base = owner * 4u;
    const float raw_density = site_rgba_f32[rgba_base + 3u];
    const float density = max(raw_density, 0.0f);
    const float adj_weight = dot(grad_rgb, segment_rgb[run_id]) + adj_alpha_out;
    const float adj_depth_mass = adj_depth_weighted_out * trans_before[run_id];
    const float adj_trans_before =
        adj_weight * segment_alpha[run_id] +
        adj_depth_weighted_out * depth_masses[run_id] +
        adj_next_transmittance * segment_trans[run_id];
    const float adj_segment_alpha = adj_weight * trans_before[run_id];
    float adj_segment_trans = adj_next_transmittance * trans_before[run_id] - adj_segment_alpha;

    float4 grad_rgba = float4(
        weights[run_id] * grad_rgb.x,
        weights[run_id] * grad_rgb.y,
        weights[run_id] * grad_rgb.z,
        0.0f);
    if (raw_density > 0.0f) {
      const float length = lengths[run_id];
      const float dtrans_dd = -length * segment_trans[run_id];
      const float dmass_dd = wf2_continuous_depth_mass_grad_density(
          starts[run_id],
          length,
          density,
          segment_trans[run_id],
          segment_alpha[run_id]);
      grad_rgba.w = adj_segment_trans * dtrans_dd + adj_depth_mass * dmass_dd;
    }
    wf2_atomic_add4(grad_site_rgba_f32, rgba_base, grad_rgba);
    adj_next_transmittance = adj_trans_before;
  }
}

kernel void wf2_endpoint_record_edit_vjp_direct_atomic_rgb_only_tensor(
    device const float* boundary_f32 [[buffer(0)]],
    device const float* rays_f32 [[buffer(1)]],
    device const float* frame_t_f32 [[buffer(2)]],
    device const int* base_offsets_i32 [[buffer(3)]],
    device const int* base_owner_i32 [[buffer(4)]],
    device const int* base_left_i32 [[buffer(5)]],
    device const int* base_right_i32 [[buffer(6)]],
    device const int* track_change_offsets_i32 [[buffer(7)]],
    device const int* change_frame_i32 [[buffer(8)]],
    device const int* op_offsets_i32 [[buffer(9)]],
    device const int* op_type_i32 [[buffer(10)]],
    device const int* op_pos_i32 [[buffer(11)]],
    device const int* op_owner_i32 [[buffer(12)]],
    device const int* op_left_i32 [[buffer(13)]],
    device const int* op_right_i32 [[buffer(14)]],
    device const float* site_rgba_f32 [[buffer(15)]],
    device const float* grad_rgb_f32 [[buffer(16)]],
    device const int* config_i32 [[buffer(17)]],
    device const float* config_f32 [[buffer(18)]],
    device atomic_float* grad_site_rgba_f32 [[buffer(19)]],
    uint sample_id [[thread_position_in_grid]]) {
  const uint boundary_count = uint(config_i32[0]);
  const uint track_count = uint(config_i32[1]);
  const uint frame_count = uint(config_i32[2]);
  const uint site_count = uint(config_i32[3]);
  const uint base_record_count = uint(config_i32[4]);
  const uint change_count = uint(config_i32[5]);
  const uint op_count = uint(config_i32[6]);
  const uint total_samples = track_count * frame_count;
  if (sample_id >= total_samples) {
    return;
  }

  const uint track_id = sample_id / frame_count;
  const uint frame_id = sample_id - track_id * frame_count;
  const float near_depth = config_f32[0];
  const float far_depth = config_f32[1];
  const float invalid_epsilon = config_f32[2];
  const float transmittance_threshold = config_f32[3];
  const uint ray_base = sample_id * 6u;
  const float3 origin = float3(rays_f32[ray_base + 0u], rays_f32[ray_base + 1u], rays_f32[ray_base + 2u]);
  const float3 direction = float3(rays_f32[ray_base + 3u], rays_f32[ray_base + 4u], rays_f32[ray_base + 5u]);
  const float t = frame_t_f32[frame_id];

  int row_owner[WF2_MAX_REALRAY_SEGMENTS];
  int row_left[WF2_MAX_REALRAY_SEGMENTS];
  int row_right[WF2_MAX_REALRAY_SEGMENTS];
  uint row_count = 0u;
  if (!wf2_endpoint_record_load_edit_row(
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
          track_id,
          frame_id,
          base_record_count,
          change_count,
          op_count,
          row_owner,
          row_left,
          row_right,
          row_count)) {
    return;
  }

  uint owners[WF2_MAX_REALRAY_SEGMENTS];
  float lengths[WF2_MAX_REALRAY_SEGMENTS];
  float trans_before[WF2_MAX_REALRAY_SEGMENTS];
  float segment_trans[WF2_MAX_REALRAY_SEGMENTS];
  float segment_alpha[WF2_MAX_REALRAY_SEGMENTS];
  float weights[WF2_MAX_REALRAY_SEGMENTS];
  float3 segment_rgb[WF2_MAX_REALRAY_SEGMENTS];

  float transmittance = 1.0f;
  uint local_run_count = 0u;
  int cached_right_cut = -2147483648;
  float cached_right_depth = 0.0f;
  bool cached_right_valid = false;
  for (uint cursor = 0u; cursor < row_count; ++cursor) {
    if (transmittance <= transmittance_threshold || local_run_count >= WF2_MAX_REALRAY_SEGMENTS) {
      break;
    }
    const int owner_raw = row_owner[cursor];
    if (owner_raw < 0 || uint(owner_raw) >= site_count) {
      continue;
    }
    float start_depth = 0.0f;
    float end_depth = 0.0f;
    const int left_cut = row_left[cursor];
    const int right_cut = row_right[cursor];
    bool start_valid = false;
    if (cached_right_valid && left_cut == cached_right_cut) {
      start_depth = cached_right_depth;
      start_valid = true;
    } else {
      start_valid = wf2_endpoint_record_cut_depth(
          boundary_f32,
          boundary_count,
          left_cut,
          origin,
          direction,
          t,
          near_depth,
          far_depth,
          invalid_epsilon,
          start_depth);
    }
    if (!start_valid ||
        !wf2_endpoint_record_cut_depth(
            boundary_f32,
            boundary_count,
            right_cut,
            origin,
            direction,
            t,
            near_depth,
            far_depth,
            invalid_epsilon,
            end_depth)) {
      cached_right_valid = false;
      continue;
    }
    cached_right_cut = right_cut;
    cached_right_depth = end_depth;
    cached_right_valid = true;
    const float length = end_depth - start_depth;
    if (!(length > 1.0e-8f)) {
      continue;
    }
    const uint owner = uint(owner_raw);
    const uint rgba_base = owner * 4u;
    const float density = max(site_rgba_f32[rgba_base + 3u], 0.0f);
    const float seg_trans = exp(-density * length);
    const float seg_alpha = 1.0f - seg_trans;
    const float weight = transmittance * seg_alpha;
    const float3 rgb = float3(
        site_rgba_f32[rgba_base + 0u],
        site_rgba_f32[rgba_base + 1u],
        site_rgba_f32[rgba_base + 2u]);

    owners[local_run_count] = owner;
    lengths[local_run_count] = length;
    trans_before[local_run_count] = transmittance;
    segment_trans[local_run_count] = seg_trans;
    segment_alpha[local_run_count] = seg_alpha;
    weights[local_run_count] = weight;
    segment_rgb[local_run_count] = rgb;
    local_run_count += 1u;

    transmittance *= seg_trans;
  }

  const uint out_base = sample_id * 3u;
  const float3 grad_rgb = float3(
      grad_rgb_f32[out_base + 0u],
      grad_rgb_f32[out_base + 1u],
      grad_rgb_f32[out_base + 2u]);
  float adj_next_transmittance = 0.0f;
  for (int run_id = int(local_run_count) - 1; run_id >= 0; --run_id) {
    const uint owner = owners[run_id];
    const uint rgba_base = owner * 4u;
    const float raw_density = site_rgba_f32[rgba_base + 3u];
    const float adj_weight = dot(grad_rgb, segment_rgb[run_id]);
    const float adj_trans_before =
        adj_weight * segment_alpha[run_id] +
        adj_next_transmittance * segment_trans[run_id];
    const float adj_segment_alpha = adj_weight * trans_before[run_id];
    const float adj_segment_trans = adj_next_transmittance * trans_before[run_id] - adj_segment_alpha;
    float4 grad_rgba = float4(
        weights[run_id] * grad_rgb.x,
        weights[run_id] * grad_rgb.y,
        weights[run_id] * grad_rgb.z,
        0.0f);
    if (raw_density > 0.0f) {
      grad_rgba.w = adj_segment_trans * (-lengths[run_id] * segment_trans[run_id]);
    }
    wf2_atomic_add4(grad_site_rgba_f32, rgba_base, grad_rgba);
    adj_next_transmittance = adj_trans_before;
  }
}

kernel void wf2_endpoint_record_edit_mse_vjp_direct_atomic_rgb_only_tensor(
    device const float* boundary_f32 [[buffer(0)]],
    device const float* rays_f32 [[buffer(1)]],
    device const float* frame_t_f32 [[buffer(2)]],
    device const int* base_offsets_i32 [[buffer(3)]],
    device const int* base_owner_i32 [[buffer(4)]],
    device const int* base_left_i32 [[buffer(5)]],
    device const int* base_right_i32 [[buffer(6)]],
    device const int* track_change_offsets_i32 [[buffer(7)]],
    device const int* change_frame_i32 [[buffer(8)]],
    device const int* op_offsets_i32 [[buffer(9)]],
    device const int* op_type_i32 [[buffer(10)]],
    device const int* op_pos_i32 [[buffer(11)]],
    device const int* op_owner_i32 [[buffer(12)]],
    device const int* op_left_i32 [[buffer(13)]],
    device const int* op_right_i32 [[buffer(14)]],
    device const float* site_rgba_f32 [[buffer(15)]],
    device const float* target_rgb_f32 [[buffer(16)]],
    device const int* config_i32 [[buffer(17)]],
    device const float* config_f32 [[buffer(18)]],
    device atomic_float* loss_f32 [[buffer(19)]],
    device atomic_float* grad_site_rgba_f32 [[buffer(20)]],
    uint sample_id [[thread_position_in_grid]]) {
  const uint boundary_count = uint(config_i32[0]);
  const uint track_count = uint(config_i32[1]);
  const uint frame_count = uint(config_i32[2]);
  const uint site_count = uint(config_i32[3]);
  const uint base_record_count = uint(config_i32[4]);
  const uint change_count = uint(config_i32[5]);
  const uint op_count = uint(config_i32[6]);
  const uint total_samples = track_count * frame_count;
  if (sample_id >= total_samples) {
    return;
  }

  const uint track_id = sample_id / frame_count;
  const uint frame_id = sample_id - track_id * frame_count;
  const float near_depth = config_f32[0];
  const float far_depth = config_f32[1];
  const float invalid_epsilon = config_f32[2];
  const float transmittance_threshold = config_f32[3];
  const uint ray_base = sample_id * 6u;
  const float3 origin = float3(rays_f32[ray_base + 0u], rays_f32[ray_base + 1u], rays_f32[ray_base + 2u]);
  const float3 direction = float3(rays_f32[ray_base + 3u], rays_f32[ray_base + 4u], rays_f32[ray_base + 5u]);
  const float t = frame_t_f32[frame_id];

  int row_owner[WF2_MAX_REALRAY_SEGMENTS];
  int row_left[WF2_MAX_REALRAY_SEGMENTS];
  int row_right[WF2_MAX_REALRAY_SEGMENTS];
  uint row_count = 0u;
  if (!wf2_endpoint_record_load_edit_row(
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
          track_id,
          frame_id,
          base_record_count,
          change_count,
          op_count,
          row_owner,
          row_left,
          row_right,
          row_count)) {
    return;
  }

  uint owners[WF2_MAX_REALRAY_SEGMENTS];
  float lengths[WF2_MAX_REALRAY_SEGMENTS];
  float trans_before[WF2_MAX_REALRAY_SEGMENTS];
  float segment_trans[WF2_MAX_REALRAY_SEGMENTS];
  float segment_alpha[WF2_MAX_REALRAY_SEGMENTS];
  float weights[WF2_MAX_REALRAY_SEGMENTS];
  float3 segment_rgb[WF2_MAX_REALRAY_SEGMENTS];

  float3 rgb_accum = float3(0.0f, 0.0f, 0.0f);
  float transmittance = 1.0f;
  uint local_run_count = 0u;
  int cached_right_cut = -2147483648;
  float cached_right_depth = 0.0f;
  bool cached_right_valid = false;
  for (uint cursor = 0u; cursor < row_count; ++cursor) {
    if (transmittance <= transmittance_threshold || local_run_count >= WF2_MAX_REALRAY_SEGMENTS) {
      break;
    }
    const int owner_raw = row_owner[cursor];
    if (owner_raw < 0 || uint(owner_raw) >= site_count) {
      continue;
    }
    float start_depth = 0.0f;
    float end_depth = 0.0f;
    const int left_cut = row_left[cursor];
    const int right_cut = row_right[cursor];
    bool start_valid = false;
    if (cached_right_valid && left_cut == cached_right_cut) {
      start_depth = cached_right_depth;
      start_valid = true;
    } else {
      start_valid = wf2_endpoint_record_cut_depth(
          boundary_f32,
          boundary_count,
          left_cut,
          origin,
          direction,
          t,
          near_depth,
          far_depth,
          invalid_epsilon,
          start_depth);
    }
    if (!start_valid ||
        !wf2_endpoint_record_cut_depth(
            boundary_f32,
            boundary_count,
            right_cut,
            origin,
            direction,
            t,
            near_depth,
            far_depth,
            invalid_epsilon,
            end_depth)) {
      cached_right_valid = false;
      continue;
    }
    cached_right_cut = right_cut;
    cached_right_depth = end_depth;
    cached_right_valid = true;
    const float length = end_depth - start_depth;
    if (!(length > 1.0e-8f)) {
      continue;
    }
    const uint owner = uint(owner_raw);
    const uint rgba_base = owner * 4u;
    const float density = max(site_rgba_f32[rgba_base + 3u], 0.0f);
    const float seg_trans = exp(-density * length);
    const float seg_alpha = 1.0f - seg_trans;
    const float weight = transmittance * seg_alpha;
    const float3 rgb = float3(
        site_rgba_f32[rgba_base + 0u],
        site_rgba_f32[rgba_base + 1u],
        site_rgba_f32[rgba_base + 2u]);

    owners[local_run_count] = owner;
    lengths[local_run_count] = length;
    trans_before[local_run_count] = transmittance;
    segment_trans[local_run_count] = seg_trans;
    segment_alpha[local_run_count] = seg_alpha;
    weights[local_run_count] = weight;
    segment_rgb[local_run_count] = rgb;
    local_run_count += 1u;

    rgb_accum += weight * rgb;
    transmittance *= seg_trans;
  }

  const uint out_base = sample_id * 3u;
  const float3 target_rgb = float3(
      target_rgb_f32[out_base + 0u],
      target_rgb_f32[out_base + 1u],
      target_rgb_f32[out_base + 2u]);
  const float3 diff = rgb_accum - target_rgb;
  const float inv_element_count = 1.0f / float(total_samples * 3u);
  const float sample_loss = dot(diff, diff) * inv_element_count;
  atomic_fetch_add_explicit(&loss_f32[0], sample_loss, memory_order_relaxed);
  const float3 grad_rgb = (2.0f * inv_element_count) * diff;

  float adj_next_transmittance = 0.0f;
  for (int run_id = int(local_run_count) - 1; run_id >= 0; --run_id) {
    const uint owner = owners[run_id];
    const uint rgba_base = owner * 4u;
    const float raw_density = site_rgba_f32[rgba_base + 3u];
    const float adj_weight = dot(grad_rgb, segment_rgb[run_id]);
    const float adj_trans_before =
        adj_weight * segment_alpha[run_id] +
        adj_next_transmittance * segment_trans[run_id];
    const float adj_segment_alpha = adj_weight * trans_before[run_id];
    const float adj_segment_trans = adj_next_transmittance * trans_before[run_id] - adj_segment_alpha;
    float4 grad_rgba = float4(
        weights[run_id] * grad_rgb.x,
        weights[run_id] * grad_rgb.y,
        weights[run_id] * grad_rgb.z,
        0.0f);
    if (raw_density > 0.0f) {
      grad_rgba.w = adj_segment_trans * (-lengths[run_id] * segment_trans[run_id]);
    }
    wf2_atomic_add4(grad_site_rgba_f32, rgba_base, grad_rgba);
    adj_next_transmittance = adj_trans_before;
  }
}

kernel void wf2_endpoint_record_edit_coeff16_mse_vjp_direct_atomic_rgb_only_tensor(
    device const half* coeff_f16 [[buffer(0)]],
    device const float* frame_t_f32 [[buffer(1)]],
    device const int* base_offsets_i32 [[buffer(2)]],
    device const int* base_owner_i32 [[buffer(3)]],
    device const int* base_left_i32 [[buffer(4)]],
    device const int* base_right_i32 [[buffer(5)]],
    device const int* track_change_offsets_i32 [[buffer(6)]],
    device const int* change_frame_i32 [[buffer(7)]],
    device const int* op_offsets_i32 [[buffer(8)]],
    device const int* op_type_i32 [[buffer(9)]],
    device const int* op_pos_i32 [[buffer(10)]],
    device const int* op_owner_i32 [[buffer(11)]],
    device const int* op_left_i32 [[buffer(12)]],
    device const int* op_right_i32 [[buffer(13)]],
    device const float* site_rgba_f32 [[buffer(14)]],
    device const float* target_rgb_f32 [[buffer(15)]],
    device const int* config_i32 [[buffer(16)]],
    device const float* config_f32 [[buffer(17)]],
    device atomic_float* loss_f32 [[buffer(18)]],
    device atomic_float* grad_site_rgba_f32 [[buffer(19)]],
    uint sample_id [[thread_position_in_grid]]) {
  const uint boundary_count = uint(config_i32[0]);
  const uint track_count = uint(config_i32[1]);
  const uint frame_count = uint(config_i32[2]);
  const uint site_count = uint(config_i32[3]);
  const uint base_record_count = uint(config_i32[4]);
  const uint change_count = uint(config_i32[5]);
  const uint op_count = uint(config_i32[6]);
  const uint total_samples = track_count * frame_count;
  if (sample_id >= total_samples) {
    return;
  }

  const uint track_id = sample_id / frame_count;
  const uint frame_id = sample_id - track_id * frame_count;
  const float near_depth = config_f32[0];
  const float far_depth = config_f32[1];
  const float invalid_epsilon = config_f32[2];
  const float transmittance_threshold = config_f32[3];
  const float t = frame_t_f32[frame_id];

  int row_owner[WF2_MAX_REALRAY_SEGMENTS];
  int row_left[WF2_MAX_REALRAY_SEGMENTS];
  int row_right[WF2_MAX_REALRAY_SEGMENTS];
  uint row_count = 0u;
  if (!wf2_endpoint_record_load_edit_row(
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
          track_id,
          frame_id,
          base_record_count,
          change_count,
          op_count,
          row_owner,
          row_left,
          row_right,
          row_count)) {
    return;
  }

  uint owners[WF2_MAX_REALRAY_SEGMENTS];
  float lengths[WF2_MAX_REALRAY_SEGMENTS];
  float trans_before[WF2_MAX_REALRAY_SEGMENTS];
  float segment_trans[WF2_MAX_REALRAY_SEGMENTS];
  float segment_alpha[WF2_MAX_REALRAY_SEGMENTS];
  float weights[WF2_MAX_REALRAY_SEGMENTS];
  float3 segment_rgb[WF2_MAX_REALRAY_SEGMENTS];

  float3 rgb_accum = float3(0.0f, 0.0f, 0.0f);
  float transmittance = 1.0f;
  uint local_run_count = 0u;
  int cached_right_cut = -2147483648;
  float cached_right_depth = 0.0f;
  bool cached_right_valid = false;
  for (uint cursor = 0u; cursor < row_count; ++cursor) {
    if (transmittance <= transmittance_threshold || local_run_count >= WF2_MAX_REALRAY_SEGMENTS) {
      break;
    }
    const int owner_raw = row_owner[cursor];
    if (owner_raw < 0 || uint(owner_raw) >= site_count) {
      continue;
    }
    float start_depth = 0.0f;
    float end_depth = 0.0f;
    const int left_cut = row_left[cursor];
    const int right_cut = row_right[cursor];
    bool start_valid = false;
    if (cached_right_valid && left_cut == cached_right_cut) {
      start_depth = cached_right_depth;
      start_valid = true;
    } else {
      start_valid = wf2_endpoint_record_coeff16_cut_depth(
          coeff_f16,
          boundary_count,
          track_id,
          left_cut,
          t,
          near_depth,
          far_depth,
          invalid_epsilon,
          start_depth);
    }
    if (!start_valid ||
        !wf2_endpoint_record_coeff16_cut_depth(
            coeff_f16,
            boundary_count,
            track_id,
            right_cut,
            t,
            near_depth,
            far_depth,
            invalid_epsilon,
            end_depth)) {
      cached_right_valid = false;
      continue;
    }
    cached_right_cut = right_cut;
    cached_right_depth = end_depth;
    cached_right_valid = true;
    const float length = end_depth - start_depth;
    if (!(length > 1.0e-8f)) {
      continue;
    }
    const uint owner = uint(owner_raw);
    const uint rgba_base = owner * 4u;
    const float density = max(site_rgba_f32[rgba_base + 3u], 0.0f);
    const float seg_trans = exp(-density * length);
    const float seg_alpha = 1.0f - seg_trans;
    const float weight = transmittance * seg_alpha;
    const float3 rgb = float3(
        site_rgba_f32[rgba_base + 0u],
        site_rgba_f32[rgba_base + 1u],
        site_rgba_f32[rgba_base + 2u]);

    owners[local_run_count] = owner;
    lengths[local_run_count] = length;
    trans_before[local_run_count] = transmittance;
    segment_trans[local_run_count] = seg_trans;
    segment_alpha[local_run_count] = seg_alpha;
    weights[local_run_count] = weight;
    segment_rgb[local_run_count] = rgb;
    local_run_count += 1u;

    rgb_accum += weight * rgb;
    transmittance *= seg_trans;
  }

  const uint out_base = sample_id * 3u;
  const float3 target_rgb = float3(
      target_rgb_f32[out_base + 0u],
      target_rgb_f32[out_base + 1u],
      target_rgb_f32[out_base + 2u]);
  const float3 diff = rgb_accum - target_rgb;
  const float inv_element_count = 1.0f / float(total_samples * 3u);
  const float sample_loss = dot(diff, diff) * inv_element_count;
  atomic_fetch_add_explicit(&loss_f32[0], sample_loss, memory_order_relaxed);
  const float3 grad_rgb = (2.0f * inv_element_count) * diff;

  float adj_next_transmittance = 0.0f;
  for (int run_id = int(local_run_count) - 1; run_id >= 0; --run_id) {
    const uint owner = owners[run_id];
    const uint rgba_base = owner * 4u;
    const float raw_density = site_rgba_f32[rgba_base + 3u];
    const float adj_weight = dot(grad_rgb, segment_rgb[run_id]);
    const float adj_trans_before =
        adj_weight * segment_alpha[run_id] +
        adj_next_transmittance * segment_trans[run_id];
    const float adj_segment_alpha = adj_weight * trans_before[run_id];
    const float adj_segment_trans = adj_next_transmittance * trans_before[run_id] - adj_segment_alpha;
    float4 grad_rgba = float4(
        weights[run_id] * grad_rgb.x,
        weights[run_id] * grad_rgb.y,
        weights[run_id] * grad_rgb.z,
        0.0f);
    if (raw_density > 0.0f) {
      grad_rgba.w = adj_segment_trans * (-lengths[run_id] * segment_trans[run_id]);
    }
    wf2_atomic_add4(grad_site_rgba_f32, rgba_base, grad_rgba);
    adj_next_transmittance = adj_trans_before;
  }
}

kernel void wf2_endpoint_record_delta_replace_coeff16_mse_vjp_direct_atomic_rgb_only_tensor(
    device const half* coeff_f16 [[buffer(0)]],
    device const float* frame_t_f32 [[buffer(1)]],
    device const int* base_offsets_i32 [[buffer(2)]],
    device const int* base_owner_i32 [[buffer(3)]],
    device const int* base_left_i32 [[buffer(4)]],
    device const int* base_right_i32 [[buffer(5)]],
    device const int* track_change_offsets_i32 [[buffer(6)]],
    device const int* change_frame_i32 [[buffer(7)]],
    device const int* change_offsets_i32 [[buffer(8)]],
    device const int* change_owner_i32 [[buffer(9)]],
    device const int* change_left_i32 [[buffer(10)]],
    device const int* change_right_i32 [[buffer(11)]],
    device const float* site_rgba_f32 [[buffer(12)]],
    device const float* target_rgb_f32 [[buffer(13)]],
    device const int* config_i32 [[buffer(14)]],
    device const float* config_f32 [[buffer(15)]],
    device atomic_float* loss_f32 [[buffer(16)]],
    device atomic_float* grad_site_rgba_f32 [[buffer(17)]],
    uint sample_id [[thread_position_in_grid]]) {
  const uint boundary_count = uint(config_i32[0]);
  const uint track_count = uint(config_i32[1]);
  const uint frame_count = uint(config_i32[2]);
  const uint site_count = uint(config_i32[3]);
  const uint base_record_count = uint(config_i32[4]);
  const uint change_count = uint(config_i32[5]);
  const uint change_record_count = uint(config_i32[6]);
  const uint total_samples = track_count * frame_count;
  if (sample_id >= total_samples) {
    return;
  }

  const uint track_id = sample_id / frame_count;
  const uint frame_id = sample_id - track_id * frame_count;
  const float near_depth = config_f32[0];
  const float far_depth = config_f32[1];
  const float invalid_epsilon = config_f32[2];
  const float transmittance_threshold = config_f32[3];
  const float t = frame_t_f32[frame_id];

  int selected_change = -1;
  const int change_begin_raw = track_change_offsets_i32[track_id];
  const int change_end_raw = track_change_offsets_i32[track_id + 1u];
  if (change_begin_raw < 0 || change_end_raw < change_begin_raw || uint(change_end_raw) > change_count) {
    return;
  }
  for (uint change_cursor = uint(change_begin_raw); change_cursor < uint(change_end_raw); ++change_cursor) {
    const int changed_frame = change_frame_i32[change_cursor];
    if (changed_frame < 0) {
      continue;
    }
    if (uint(changed_frame) <= frame_id) {
      selected_change = int(change_cursor);
      continue;
    }
    break;
  }

  const bool use_change = selected_change >= 0;
  const int begin_raw = use_change ? change_offsets_i32[uint(selected_change)] : base_offsets_i32[track_id];
  const int end_raw = use_change ? change_offsets_i32[uint(selected_change) + 1u] : base_offsets_i32[track_id + 1u];
  const uint record_count = use_change ? change_record_count : base_record_count;
  if (begin_raw < 0 || end_raw < begin_raw || uint(end_raw) > record_count) {
    return;
  }
  if (begin_raw == end_raw) {
    const uint out_base = sample_id * 3u;
    const float3 target_rgb = float3(
        target_rgb_f32[out_base + 0u],
        target_rgb_f32[out_base + 1u],
        target_rgb_f32[out_base + 2u]);
    const float inv_element_count = 1.0f / float(total_samples * 3u);
    const float sample_loss = dot(target_rgb, target_rgb) * inv_element_count;
    atomic_fetch_add_explicit(&loss_f32[0], sample_loss, memory_order_relaxed);
    return;
  }

  uint owners[WF2_MAX_REALRAY_SEGMENTS];
  float lengths[WF2_MAX_REALRAY_SEGMENTS];
  float trans_before[WF2_MAX_REALRAY_SEGMENTS];
  float segment_trans[WF2_MAX_REALRAY_SEGMENTS];
  float segment_alpha[WF2_MAX_REALRAY_SEGMENTS];
  float weights[WF2_MAX_REALRAY_SEGMENTS];
  float3 segment_rgb[WF2_MAX_REALRAY_SEGMENTS];

  float3 rgb_accum = float3(0.0f, 0.0f, 0.0f);
  float transmittance = 1.0f;
  uint local_run_count = 0u;
  int cached_right_cut = -2147483648;
  float cached_right_depth = 0.0f;
  bool cached_right_valid = false;
  for (uint cursor = uint(begin_raw); cursor < uint(end_raw); ++cursor) {
    if (transmittance <= transmittance_threshold || local_run_count >= WF2_MAX_REALRAY_SEGMENTS) {
      break;
    }
    const int owner_raw = use_change ? change_owner_i32[cursor] : base_owner_i32[cursor];
    if (owner_raw < 0 || uint(owner_raw) >= site_count) {
      continue;
    }
    float start_depth = 0.0f;
    float end_depth = 0.0f;
    const int left_cut = use_change ? change_left_i32[cursor] : base_left_i32[cursor];
    const int right_cut = use_change ? change_right_i32[cursor] : base_right_i32[cursor];
    bool start_valid = false;
    if (cached_right_valid && left_cut == cached_right_cut) {
      start_depth = cached_right_depth;
      start_valid = true;
    } else {
      start_valid = wf2_endpoint_record_coeff16_cut_depth(
          coeff_f16,
          boundary_count,
          track_id,
          left_cut,
          t,
          near_depth,
          far_depth,
          invalid_epsilon,
          start_depth);
    }
    if (!start_valid ||
        !wf2_endpoint_record_coeff16_cut_depth(
            coeff_f16,
            boundary_count,
            track_id,
            right_cut,
            t,
            near_depth,
            far_depth,
            invalid_epsilon,
            end_depth)) {
      cached_right_valid = false;
      continue;
    }
    cached_right_cut = right_cut;
    cached_right_depth = end_depth;
    cached_right_valid = true;
    const float length = end_depth - start_depth;
    if (!(length > 1.0e-8f)) {
      continue;
    }
    const uint owner = uint(owner_raw);
    const uint rgba_base = owner * 4u;
    const float density = max(site_rgba_f32[rgba_base + 3u], 0.0f);
    const float seg_trans = exp(-density * length);
    const float seg_alpha = 1.0f - seg_trans;
    const float weight = transmittance * seg_alpha;
    const float3 rgb = float3(
        site_rgba_f32[rgba_base + 0u],
        site_rgba_f32[rgba_base + 1u],
        site_rgba_f32[rgba_base + 2u]);

    owners[local_run_count] = owner;
    lengths[local_run_count] = length;
    trans_before[local_run_count] = transmittance;
    segment_trans[local_run_count] = seg_trans;
    segment_alpha[local_run_count] = seg_alpha;
    weights[local_run_count] = weight;
    segment_rgb[local_run_count] = rgb;
    local_run_count += 1u;

    rgb_accum += weight * rgb;
    transmittance *= seg_trans;
  }

  const uint out_base = sample_id * 3u;
  const float3 target_rgb = float3(
      target_rgb_f32[out_base + 0u],
      target_rgb_f32[out_base + 1u],
      target_rgb_f32[out_base + 2u]);
  const float3 diff = rgb_accum - target_rgb;
  const float inv_element_count = 1.0f / float(total_samples * 3u);
  const float sample_loss = dot(diff, diff) * inv_element_count;
  atomic_fetch_add_explicit(&loss_f32[0], sample_loss, memory_order_relaxed);
  const float3 grad_rgb = (2.0f * inv_element_count) * diff;

  float adj_next_transmittance = 0.0f;
  for (int run_id = int(local_run_count) - 1; run_id >= 0; --run_id) {
    const uint owner = owners[run_id];
    const uint rgba_base = owner * 4u;
    const float raw_density = site_rgba_f32[rgba_base + 3u];
    const float adj_weight = dot(grad_rgb, segment_rgb[run_id]);
    const float adj_trans_before =
        adj_weight * segment_alpha[run_id] +
        adj_next_transmittance * segment_trans[run_id];
    const float adj_segment_alpha = adj_weight * trans_before[run_id];
    const float adj_segment_trans = adj_next_transmittance * trans_before[run_id] - adj_segment_alpha;
    float4 grad_rgba = float4(
        weights[run_id] * grad_rgb.x,
        weights[run_id] * grad_rgb.y,
        weights[run_id] * grad_rgb.z,
        0.0f);
    if (raw_density > 0.0f) {
      grad_rgba.w = adj_segment_trans * (-lengths[run_id] * segment_trans[run_id]);
    }
    wf2_atomic_add4(grad_site_rgba_f32, rgba_base, grad_rgba);
    adj_next_transmittance = adj_trans_before;
  }
}

kernel void wf2_endpoint_record_delta_replace_coeff16_i16x3_mse_vjp_direct_atomic_rgb_only_tensor(
    device const half* coeff_f16 [[buffer(0)]],
    device const float* frame_t_f32 [[buffer(1)]],
    device const int* base_offsets_i32 [[buffer(2)]],
    device const short* base_record_i16 [[buffer(3)]],
    device const int* track_change_offsets_i32 [[buffer(4)]],
    device const int* change_frame_i32 [[buffer(5)]],
    device const int* change_offsets_i32 [[buffer(6)]],
    device const short* change_record_i16 [[buffer(7)]],
    device const float* site_rgba_f32 [[buffer(8)]],
    device const float* target_rgb_f32 [[buffer(9)]],
    device const int* config_i32 [[buffer(10)]],
    device const float* config_f32 [[buffer(11)]],
    device atomic_float* loss_f32 [[buffer(12)]],
    device atomic_float* grad_site_rgba_f32 [[buffer(13)]],
    uint sample_id [[thread_position_in_grid]]) {
  const uint boundary_count = uint(config_i32[0]);
  const uint track_count = uint(config_i32[1]);
  const uint frame_count = uint(config_i32[2]);
  const uint site_count = uint(config_i32[3]);
  const uint base_record_count = uint(config_i32[4]);
  const uint change_count = uint(config_i32[5]);
  const uint change_record_count = uint(config_i32[6]);
  const uint total_samples = track_count * frame_count;
  if (sample_id >= total_samples) {
    return;
  }

  const uint track_id = sample_id / frame_count;
  const uint frame_id = sample_id - track_id * frame_count;
  const float near_depth = config_f32[0];
  const float far_depth = config_f32[1];
  const float invalid_epsilon = config_f32[2];
  const float transmittance_threshold = config_f32[3];
  const float t = frame_t_f32[frame_id];

  int selected_change = -1;
  const int change_begin_raw = track_change_offsets_i32[track_id];
  const int change_end_raw = track_change_offsets_i32[track_id + 1u];
  if (change_begin_raw < 0 || change_end_raw < change_begin_raw || uint(change_end_raw) > change_count) {
    return;
  }
  for (uint change_cursor = uint(change_begin_raw); change_cursor < uint(change_end_raw); ++change_cursor) {
    const int changed_frame = change_frame_i32[change_cursor];
    if (changed_frame < 0) {
      continue;
    }
    if (uint(changed_frame) <= frame_id) {
      selected_change = int(change_cursor);
      continue;
    }
    break;
  }

  const bool use_change = selected_change >= 0;
  const int begin_raw = use_change ? change_offsets_i32[uint(selected_change)] : base_offsets_i32[track_id];
  const int end_raw = use_change ? change_offsets_i32[uint(selected_change) + 1u] : base_offsets_i32[track_id + 1u];
  const uint record_count = use_change ? change_record_count : base_record_count;
  if (begin_raw < 0 || end_raw < begin_raw || uint(end_raw) > record_count) {
    return;
  }
  if (begin_raw == end_raw) {
    const uint out_base = sample_id * 3u;
    const float3 target_rgb = float3(
        target_rgb_f32[out_base + 0u],
        target_rgb_f32[out_base + 1u],
        target_rgb_f32[out_base + 2u]);
    const float inv_element_count = 1.0f / float(total_samples * 3u);
    const float sample_loss = dot(target_rgb, target_rgb) * inv_element_count;
    atomic_fetch_add_explicit(&loss_f32[0], sample_loss, memory_order_relaxed);
    return;
  }

  uint owners[WF2_MAX_REALRAY_SEGMENTS];
  float lengths[WF2_MAX_REALRAY_SEGMENTS];
  float trans_before[WF2_MAX_REALRAY_SEGMENTS];
  float segment_trans[WF2_MAX_REALRAY_SEGMENTS];
  float segment_alpha[WF2_MAX_REALRAY_SEGMENTS];
  float weights[WF2_MAX_REALRAY_SEGMENTS];
  float3 segment_rgb[WF2_MAX_REALRAY_SEGMENTS];

  float3 rgb_accum = float3(0.0f, 0.0f, 0.0f);
  float transmittance = 1.0f;
  uint local_run_count = 0u;
  int cached_right_cut = -2147483648;
  float cached_right_depth = 0.0f;
  bool cached_right_valid = false;
  for (uint cursor = uint(begin_raw); cursor < uint(end_raw); ++cursor) {
    if (transmittance <= transmittance_threshold || local_run_count >= WF2_MAX_REALRAY_SEGMENTS) {
      break;
    }
    const uint record_base = cursor * 3u;
    const int owner_raw = use_change ? int(change_record_i16[record_base + 0u]) : int(base_record_i16[record_base + 0u]);
    if (owner_raw < 0 || uint(owner_raw) >= site_count) {
      continue;
    }
    float start_depth = 0.0f;
    float end_depth = 0.0f;
    const int left_cut = use_change ? int(change_record_i16[record_base + 1u]) : int(base_record_i16[record_base + 1u]);
    const int right_cut = use_change ? int(change_record_i16[record_base + 2u]) : int(base_record_i16[record_base + 2u]);
    bool start_valid = false;
    if (cached_right_valid && left_cut == cached_right_cut) {
      start_depth = cached_right_depth;
      start_valid = true;
    } else {
      start_valid = wf2_endpoint_record_coeff16_cut_depth(
          coeff_f16,
          boundary_count,
          track_id,
          left_cut,
          t,
          near_depth,
          far_depth,
          invalid_epsilon,
          start_depth);
    }
    if (!start_valid ||
        !wf2_endpoint_record_coeff16_cut_depth(
            coeff_f16,
            boundary_count,
            track_id,
            right_cut,
            t,
            near_depth,
            far_depth,
            invalid_epsilon,
            end_depth)) {
      cached_right_valid = false;
      continue;
    }
    cached_right_cut = right_cut;
    cached_right_depth = end_depth;
    cached_right_valid = true;
    const float length = end_depth - start_depth;
    if (!(length > 1.0e-8f)) {
      continue;
    }
    const uint owner = uint(owner_raw);
    const uint rgba_base = owner * 4u;
    const float density = max(site_rgba_f32[rgba_base + 3u], 0.0f);
    const float seg_trans = exp(-density * length);
    const float seg_alpha = 1.0f - seg_trans;
    const float weight = transmittance * seg_alpha;
    const float3 rgb = float3(
        site_rgba_f32[rgba_base + 0u],
        site_rgba_f32[rgba_base + 1u],
        site_rgba_f32[rgba_base + 2u]);

    owners[local_run_count] = owner;
    lengths[local_run_count] = length;
    trans_before[local_run_count] = transmittance;
    segment_trans[local_run_count] = seg_trans;
    segment_alpha[local_run_count] = seg_alpha;
    weights[local_run_count] = weight;
    segment_rgb[local_run_count] = rgb;
    local_run_count += 1u;

    rgb_accum += weight * rgb;
    transmittance *= seg_trans;
  }

  const uint out_base = sample_id * 3u;
  const float3 target_rgb = float3(
      target_rgb_f32[out_base + 0u],
      target_rgb_f32[out_base + 1u],
      target_rgb_f32[out_base + 2u]);
  const float3 diff = rgb_accum - target_rgb;
  const float inv_element_count = 1.0f / float(total_samples * 3u);
  const float sample_loss = dot(diff, diff) * inv_element_count;
  atomic_fetch_add_explicit(&loss_f32[0], sample_loss, memory_order_relaxed);
  const float3 grad_rgb = (2.0f * inv_element_count) * diff;

  float adj_next_transmittance = 0.0f;
  for (int run_id = int(local_run_count) - 1; run_id >= 0; --run_id) {
    const uint owner = owners[run_id];
    const uint rgba_base = owner * 4u;
    const float raw_density = site_rgba_f32[rgba_base + 3u];
    const float adj_weight = dot(grad_rgb, segment_rgb[run_id]);
    const float adj_trans_before =
        adj_weight * segment_alpha[run_id] +
        adj_next_transmittance * segment_trans[run_id];
    const float adj_segment_alpha = adj_weight * trans_before[run_id];
    const float adj_segment_trans = adj_next_transmittance * trans_before[run_id] - adj_segment_alpha;
    float4 grad_rgba = float4(
        weights[run_id] * grad_rgb.x,
        weights[run_id] * grad_rgb.y,
        weights[run_id] * grad_rgb.z,
        0.0f);
    if (raw_density > 0.0f) {
      grad_rgba.w = adj_segment_trans * (-lengths[run_id] * segment_trans[run_id]);
    }
    wf2_atomic_add4(grad_site_rgba_f32, rgba_base, grad_rgba);
    adj_next_transmittance = adj_trans_before;
  }
}

kernel void wf2_endpoint_record_delta_replace_coeff16_packed_mse_vjp_direct_atomic_rgb_only_tensor(
    device const half* coeff_f16 [[buffer(0)]],
    device const float* frame_t_f32 [[buffer(1)]],
    device const int* base_offsets_i32 [[buffer(2)]],
    device const int* base_record_i32 [[buffer(3)]],
    device const int* track_change_offsets_i32 [[buffer(4)]],
    device const int* change_frame_i32 [[buffer(5)]],
    device const int* change_offsets_i32 [[buffer(6)]],
    device const int* change_record_i32 [[buffer(7)]],
    device const float* site_rgba_f32 [[buffer(8)]],
    device const float* target_rgb_f32 [[buffer(9)]],
    device const int* config_i32 [[buffer(10)]],
    device const float* config_f32 [[buffer(11)]],
    device atomic_float* loss_f32 [[buffer(12)]],
    device atomic_float* grad_site_rgba_f32 [[buffer(13)]],
    uint sample_id [[thread_position_in_grid]]) {
  const uint boundary_count = uint(config_i32[0]);
  const uint track_count = uint(config_i32[1]);
  const uint frame_count = uint(config_i32[2]);
  const uint site_count = uint(config_i32[3]);
  const uint base_record_count = uint(config_i32[4]);
  const uint change_count = uint(config_i32[5]);
  const uint change_record_count = uint(config_i32[6]);
  const uint total_samples = track_count * frame_count;
  if (sample_id >= total_samples) {
    return;
  }

  const uint track_id = sample_id / frame_count;
  const uint frame_id = sample_id - track_id * frame_count;
  const float near_depth = config_f32[0];
  const float far_depth = config_f32[1];
  const float invalid_epsilon = config_f32[2];
  const float transmittance_threshold = config_f32[3];
  const float t = frame_t_f32[frame_id];

  int selected_change = -1;
  const int change_begin_raw = track_change_offsets_i32[track_id];
  const int change_end_raw = track_change_offsets_i32[track_id + 1u];
  if (change_begin_raw < 0 || change_end_raw < change_begin_raw || uint(change_end_raw) > change_count) {
    return;
  }
  for (uint change_cursor = uint(change_begin_raw); change_cursor < uint(change_end_raw); ++change_cursor) {
    const int changed_frame = change_frame_i32[change_cursor];
    if (changed_frame < 0) {
      continue;
    }
    if (uint(changed_frame) <= frame_id) {
      selected_change = int(change_cursor);
      continue;
    }
    break;
  }

  const bool use_change = selected_change >= 0;
  const int begin_raw = use_change ? change_offsets_i32[uint(selected_change)] : base_offsets_i32[track_id];
  const int end_raw = use_change ? change_offsets_i32[uint(selected_change) + 1u] : base_offsets_i32[track_id + 1u];
  const uint record_count = use_change ? change_record_count : base_record_count;
  if (begin_raw < 0 || end_raw < begin_raw || uint(end_raw) > record_count) {
    return;
  }
  if (begin_raw == end_raw) {
    const uint out_base = sample_id * 3u;
    const float3 target_rgb = float3(
        target_rgb_f32[out_base + 0u],
        target_rgb_f32[out_base + 1u],
        target_rgb_f32[out_base + 2u]);
    const float inv_element_count = 1.0f / float(total_samples * 3u);
    const float sample_loss = dot(target_rgb, target_rgb) * inv_element_count;
    atomic_fetch_add_explicit(&loss_f32[0], sample_loss, memory_order_relaxed);
    return;
  }

  uint owners[WF2_MAX_REALRAY_SEGMENTS];
  float lengths[WF2_MAX_REALRAY_SEGMENTS];
  float trans_before[WF2_MAX_REALRAY_SEGMENTS];
  float segment_trans[WF2_MAX_REALRAY_SEGMENTS];
  float segment_alpha[WF2_MAX_REALRAY_SEGMENTS];
  float weights[WF2_MAX_REALRAY_SEGMENTS];
  float3 segment_rgb[WF2_MAX_REALRAY_SEGMENTS];

  float3 rgb_accum = float3(0.0f, 0.0f, 0.0f);
  float transmittance = 1.0f;
  uint local_run_count = 0u;
  int cached_right_cut = -2147483648;
  float cached_right_depth = 0.0f;
  bool cached_right_valid = false;
  for (uint cursor = uint(begin_raw); cursor < uint(end_raw); ++cursor) {
    if (transmittance <= transmittance_threshold || local_run_count >= WF2_MAX_REALRAY_SEGMENTS) {
      break;
    }
    int owner_raw = 0;
    int left_cut = 0;
    int right_cut = 0;
    wf2_endpoint_record_unpack_record(
        use_change ? change_record_i32[cursor] : base_record_i32[cursor],
        owner_raw,
        left_cut,
        right_cut);
    if (owner_raw < 0 || uint(owner_raw) >= site_count) {
      continue;
    }
    float start_depth = 0.0f;
    float end_depth = 0.0f;
    bool start_valid = false;
    if (cached_right_valid && left_cut == cached_right_cut) {
      start_depth = cached_right_depth;
      start_valid = true;
    } else {
      start_valid = wf2_endpoint_record_coeff16_cut_depth(
          coeff_f16,
          boundary_count,
          track_id,
          left_cut,
          t,
          near_depth,
          far_depth,
          invalid_epsilon,
          start_depth);
    }
    if (!start_valid ||
        !wf2_endpoint_record_coeff16_cut_depth(
            coeff_f16,
            boundary_count,
            track_id,
            right_cut,
            t,
            near_depth,
            far_depth,
            invalid_epsilon,
            end_depth)) {
      cached_right_valid = false;
      continue;
    }
    cached_right_cut = right_cut;
    cached_right_depth = end_depth;
    cached_right_valid = true;
    const float length = end_depth - start_depth;
    if (!(length > 1.0e-8f)) {
      continue;
    }
    const uint owner = uint(owner_raw);
    const uint rgba_base = owner * 4u;
    const float density = max(site_rgba_f32[rgba_base + 3u], 0.0f);
    const float seg_trans = exp(-density * length);
    const float seg_alpha = 1.0f - seg_trans;
    const float weight = transmittance * seg_alpha;
    const float3 rgb = float3(
        site_rgba_f32[rgba_base + 0u],
        site_rgba_f32[rgba_base + 1u],
        site_rgba_f32[rgba_base + 2u]);

    owners[local_run_count] = owner;
    lengths[local_run_count] = length;
    trans_before[local_run_count] = transmittance;
    segment_trans[local_run_count] = seg_trans;
    segment_alpha[local_run_count] = seg_alpha;
    weights[local_run_count] = weight;
    segment_rgb[local_run_count] = rgb;
    local_run_count += 1u;

    rgb_accum += weight * rgb;
    transmittance *= seg_trans;
  }

  const uint out_base = sample_id * 3u;
  const float3 target_rgb = float3(
      target_rgb_f32[out_base + 0u],
      target_rgb_f32[out_base + 1u],
      target_rgb_f32[out_base + 2u]);
  const float3 diff = rgb_accum - target_rgb;
  const float inv_element_count = 1.0f / float(total_samples * 3u);
  const float sample_loss = dot(diff, diff) * inv_element_count;
  atomic_fetch_add_explicit(&loss_f32[0], sample_loss, memory_order_relaxed);
  const float3 grad_rgb = (2.0f * inv_element_count) * diff;

  float adj_next_transmittance = 0.0f;
  for (int run_id = int(local_run_count) - 1; run_id >= 0; --run_id) {
    const uint owner = owners[run_id];
    const uint rgba_base = owner * 4u;
    const float raw_density = site_rgba_f32[rgba_base + 3u];
    const float adj_weight = dot(grad_rgb, segment_rgb[run_id]);
    const float adj_trans_before =
        adj_weight * segment_alpha[run_id] +
        adj_next_transmittance * segment_trans[run_id];
    const float adj_segment_alpha = adj_weight * trans_before[run_id];
    const float adj_segment_trans = adj_next_transmittance * trans_before[run_id] - adj_segment_alpha;
    float4 grad_rgba = float4(
        weights[run_id] * grad_rgb.x,
        weights[run_id] * grad_rgb.y,
        weights[run_id] * grad_rgb.z,
        0.0f);
    if (raw_density > 0.0f) {
      grad_rgba.w = adj_segment_trans * (-lengths[run_id] * segment_trans[run_id]);
    }
    wf2_atomic_add4(grad_site_rgba_f32, rgba_base, grad_rgba);
    adj_next_transmittance = adj_trans_before;
  }
}

kernel void wf2_endpoint_record_delta_replace_coeff16_i16x3_framegroup16_mse_vjp_direct_atomic_rgb_only_tensor(
    device const half* coeff_f16 [[buffer(0)]],
    device const float* frame_t_f32 [[buffer(1)]],
    device const int* base_offsets_i32 [[buffer(2)]],
    device const short* base_record_i16 [[buffer(3)]],
    device const int* track_change_offsets_i32 [[buffer(4)]],
    device const short* track_chunk_change_offsets_i16 [[buffer(5)]],
    device const int* change_frame_i32 [[buffer(6)]],
    device const int* change_offsets_i32 [[buffer(7)]],
    device const short* change_record_i16 [[buffer(8)]],
    device const float* site_rgba_f32 [[buffer(9)]],
    device const float* target_rgb_f32 [[buffer(10)]],
    device const int* config_i32 [[buffer(11)]],
    device const float* config_f32 [[buffer(12)]],
    device atomic_float* loss_f32 [[buffer(13)]],
    device atomic_float* grad_site_rgba_f32 [[buffer(14)]],
    uint3 local_pos [[thread_position_in_threadgroup]],
    uint3 group_id [[threadgroup_position_in_grid]]) {
  const uint local_frame = local_pos.x;
  const uint boundary_count = uint(config_i32[0]);
  const uint track_count = uint(config_i32[1]);
  const uint frame_count = uint(config_i32[2]);
  const uint site_count = uint(config_i32[3]);
  const uint base_record_count = uint(config_i32[4]);
  const uint change_count = uint(config_i32[5]);
  const uint change_record_count = uint(config_i32[6]);
  const uint chunk_count = (frame_count + WF2_ENDPOINT_DELTA_FRAMEGROUP_MAX_FRAMES - 1u) /
      WF2_ENDPOINT_DELTA_FRAMEGROUP_MAX_FRAMES;
  const uint group_index = group_id.x;
  const uint track_id = group_index / max(chunk_count, 1u);
  const uint chunk_id = group_index - track_id * max(chunk_count, 1u);
  const uint frame_start = chunk_id * WF2_ENDPOINT_DELTA_FRAMEGROUP_MAX_FRAMES;
  const uint frames_in_chunk = frame_start < frame_count
      ? min(WF2_ENDPOINT_DELTA_FRAMEGROUP_MAX_FRAMES, frame_count - frame_start)
      : 0u;

  threadgroup int tg_valid[WF2_ENDPOINT_DELTA_FRAMEGROUP_MAX_FRAMES];
  threadgroup int tg_source[WF2_ENDPOINT_DELTA_FRAMEGROUP_MAX_FRAMES];
  threadgroup int tg_begin[WF2_ENDPOINT_DELTA_FRAMEGROUP_MAX_FRAMES];
  threadgroup int tg_end[WF2_ENDPOINT_DELTA_FRAMEGROUP_MAX_FRAMES];
  threadgroup float tg_loss[WF2_ENDPOINT_DELTA_FRAMEGROUP_MAX_FRAMES];
  threadgroup float4 tg_site_grad[
      WF2_ENDPOINT_DELTA_FRAMEGROUP_MAX_FRAMES * WF2_ENDPOINT_EDIT_FRAMEGROUP_REDUCE_SITES];

  if (local_frame == 0u) {
    for (uint frame_id = 0u; frame_id < WF2_ENDPOINT_DELTA_FRAMEGROUP_MAX_FRAMES; ++frame_id) {
      tg_valid[frame_id] = 0;
      tg_source[frame_id] = 0;
      tg_begin[frame_id] = 0;
      tg_end[frame_id] = 0;
    }

    bool ok = track_id < track_count && frames_in_chunk > 0u;
    int change_begin_raw = 0;
	    int change_end_raw = 0;
	    int chunk_change_begin_raw = 0;
	    int chunk_change_end_raw = 0;
    if (ok) {
	      change_begin_raw = track_change_offsets_i32[track_id];
	      change_end_raw = track_change_offsets_i32[track_id + 1u];
	      const uint chunk_offset_base = track_id * (chunk_count + 1u) + chunk_id;
	      chunk_change_begin_raw = int(track_chunk_change_offsets_i16[chunk_offset_base]);
	      chunk_change_end_raw = int(track_chunk_change_offsets_i16[chunk_offset_base + 1u]);
	      ok = change_begin_raw >= 0 && change_end_raw >= change_begin_raw && uint(change_end_raw) <= change_count &&
	          chunk_change_begin_raw >= change_begin_raw && chunk_change_end_raw >= chunk_change_begin_raw &&
	          chunk_change_end_raw <= change_end_raw;
	    }
    int selected_change = -1;
    if (ok && chunk_change_begin_raw > change_begin_raw) {
      selected_change = chunk_change_begin_raw - 1;
      while (selected_change >= change_begin_raw && change_frame_i32[uint(selected_change)] < 0) {
        selected_change -= 1;
      }
      if (selected_change < change_begin_raw) {
        selected_change = -1;
      }
    }

    uint change_cursor = ok ? uint(chunk_change_begin_raw) : 0u;
    const uint change_end = ok ? uint(chunk_change_end_raw) : 0u;

    for (uint local_frame_id = 0u; local_frame_id < frames_in_chunk; ++local_frame_id) {
      const uint global_frame_id = frame_start + local_frame_id;
      while (ok && change_cursor < change_end) {
        const int changed_frame = change_frame_i32[change_cursor];
        if (changed_frame < 0) {
          change_cursor += 1u;
          continue;
        }
        if (uint(changed_frame) > global_frame_id) {
          break;
        }

        selected_change = int(change_cursor);
        change_cursor += 1u;
      }

      if (ok) {
        const bool use_change = selected_change >= 0;
        const int begin_raw =
            use_change ? change_offsets_i32[uint(selected_change)] : base_offsets_i32[track_id];
        const int end_raw =
            use_change ? change_offsets_i32[uint(selected_change) + 1u] : base_offsets_i32[track_id + 1u];
        const uint record_count = use_change ? change_record_count : base_record_count;
        ok = begin_raw >= 0 && end_raw >= begin_raw && uint(end_raw) <= record_count &&
            uint(end_raw - begin_raw) <= WF2_MAX_REALRAY_SEGMENTS;
        if (!ok) {
          break;
        }
        tg_source[local_frame_id] = use_change ? 1 : 0;
        tg_begin[local_frame_id] = begin_raw;
	        tg_end[local_frame_id] = end_raw;
		        tg_valid[local_frame_id] = 1;
		      }
		    }

	  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

	  const bool reduce_small_sites = site_count <= WF2_ENDPOINT_EDIT_FRAMEGROUP_REDUCE_SITES;
	  if (reduce_small_sites) {
	    const uint site_grad_base = local_frame * WF2_ENDPOINT_EDIT_FRAMEGROUP_REDUCE_SITES;
	    for (uint site_slot = 0u; site_slot < WF2_ENDPOINT_EDIT_FRAMEGROUP_REDUCE_SITES; ++site_slot) {
	      tg_site_grad[site_grad_base + site_slot] = float4(0.0f, 0.0f, 0.0f, 0.0f);
	    }
  }
  tg_loss[local_frame] = 0.0f;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const bool active = track_id < track_count && local_frame < frames_in_chunk && tg_valid[local_frame] != 0;
  if (active) {

  const uint global_frame_id = frame_start + local_frame;
  const uint sample_id = track_id * frame_count + global_frame_id;
  const uint total_samples = track_count * frame_count;
  const float near_depth = config_f32[0];
  const float far_depth = config_f32[1];
  const float invalid_epsilon = config_f32[2];
  const float transmittance_threshold = config_f32[3];
  const float t = frame_t_f32[global_frame_id];
  const bool use_change = tg_source[local_frame] != 0;
  const int begin_raw = tg_begin[local_frame];
  const int end_raw = tg_end[local_frame];
  const bool valid_row_bounds = end_raw >= begin_raw;
  const uint row_count = valid_row_bounds ? uint(end_raw - begin_raw) : 0u;
  const uint out_base = sample_id * 3u;
  const float3 target_rgb = float3(
      target_rgb_f32[out_base + 0u],
      target_rgb_f32[out_base + 1u],
      target_rgb_f32[out_base + 2u]);
  const float inv_element_count = 1.0f / float(total_samples * 3u);
  if (valid_row_bounds && row_count == 0u) {
    tg_loss[local_frame] = dot(target_rgb, target_rgb) * inv_element_count;
  } else if (valid_row_bounds) {

  uint owners[WF2_MAX_REALRAY_SEGMENTS];
  float lengths[WF2_MAX_REALRAY_SEGMENTS];
  float trans_before[WF2_MAX_REALRAY_SEGMENTS];
  float segment_trans[WF2_MAX_REALRAY_SEGMENTS];
  float segment_alpha[WF2_MAX_REALRAY_SEGMENTS];
  float weights[WF2_MAX_REALRAY_SEGMENTS];
  float3 segment_rgb[WF2_MAX_REALRAY_SEGMENTS];

  float3 rgb_accum = float3(0.0f, 0.0f, 0.0f);
  float transmittance = 1.0f;
  uint local_run_count = 0u;
  int cached_right_cut = -2147483648;
  float cached_right_depth = 0.0f;
  bool cached_right_valid = false;
  for (uint cursor = 0u; cursor < row_count; ++cursor) {
    if (transmittance <= transmittance_threshold || local_run_count >= WF2_MAX_REALRAY_SEGMENTS) {
      break;
    }
    const uint record_base = uint(begin_raw) + cursor;
    const uint record_offset = record_base * 3u;
    const int owner_raw =
        use_change ? int(change_record_i16[record_offset + 0u]) : int(base_record_i16[record_offset + 0u]);
    if (owner_raw < 0 || uint(owner_raw) >= site_count) {
      continue;
    }
    float start_depth = 0.0f;
    float end_depth = 0.0f;
    const int left_cut =
        use_change ? int(change_record_i16[record_offset + 1u]) : int(base_record_i16[record_offset + 1u]);
    const int right_cut =
        use_change ? int(change_record_i16[record_offset + 2u]) : int(base_record_i16[record_offset + 2u]);
    bool start_valid = false;
    if (cached_right_valid && left_cut == cached_right_cut) {
      start_depth = cached_right_depth;
      start_valid = true;
    } else {
      start_valid = wf2_endpoint_record_coeff16_cut_depth(
          coeff_f16,
          boundary_count,
          track_id,
          left_cut,
          t,
          near_depth,
          far_depth,
          invalid_epsilon,
          start_depth);
    }
    if (!start_valid ||
        !wf2_endpoint_record_coeff16_cut_depth(
            coeff_f16,
            boundary_count,
            track_id,
            right_cut,
            t,
            near_depth,
            far_depth,
            invalid_epsilon,
            end_depth)) {
      cached_right_valid = false;
      continue;
    }
    cached_right_cut = right_cut;
    cached_right_depth = end_depth;
    cached_right_valid = true;
    const float length = end_depth - start_depth;
    if (!(length > 1.0e-8f)) {
      continue;
    }
    const uint owner = uint(owner_raw);
    const uint rgba_base = owner * 4u;
    const float density = max(site_rgba_f32[rgba_base + 3u], 0.0f);
    const float seg_trans = exp(-density * length);
    const float seg_alpha = 1.0f - seg_trans;
    const float weight = transmittance * seg_alpha;
    const float3 rgb = float3(
        site_rgba_f32[rgba_base + 0u],
        site_rgba_f32[rgba_base + 1u],
        site_rgba_f32[rgba_base + 2u]);

    owners[local_run_count] = owner;
    lengths[local_run_count] = length;
    trans_before[local_run_count] = transmittance;
    segment_trans[local_run_count] = seg_trans;
    segment_alpha[local_run_count] = seg_alpha;
    weights[local_run_count] = weight;
    segment_rgb[local_run_count] = rgb;
    local_run_count += 1u;

    rgb_accum += weight * rgb;
    transmittance *= seg_trans;
  }

  const float3 diff = rgb_accum - target_rgb;
  const float sample_loss = dot(diff, diff) * inv_element_count;
  tg_loss[local_frame] = sample_loss;
  const float3 grad_rgb = (2.0f * inv_element_count) * diff;

  float adj_next_transmittance = 0.0f;
  for (int run_id = int(local_run_count) - 1; run_id >= 0; --run_id) {
    const uint owner = owners[run_id];
    const uint rgba_base = owner * 4u;
    const float raw_density = site_rgba_f32[rgba_base + 3u];
    const float adj_weight = dot(grad_rgb, segment_rgb[run_id]);
    const float adj_trans_before =
        adj_weight * segment_alpha[run_id] +
        adj_next_transmittance * segment_trans[run_id];
    const float adj_segment_alpha = adj_weight * trans_before[run_id];
    const float adj_segment_trans = adj_next_transmittance * trans_before[run_id] - adj_segment_alpha;
    float4 grad_rgba = float4(
        weights[run_id] * grad_rgb.x,
        weights[run_id] * grad_rgb.y,
        weights[run_id] * grad_rgb.z,
        0.0f);
    if (raw_density > 0.0f) {
      grad_rgba.w = adj_segment_trans * (-lengths[run_id] * segment_trans[run_id]);
    }
		    if (reduce_small_sites) {
		      tg_site_grad[local_frame * WF2_ENDPOINT_EDIT_FRAMEGROUP_REDUCE_SITES + owner] += grad_rgba;
		    } else {
		      wf2_atomic_add4(grad_site_rgba_f32, rgba_base, grad_rgba);
		    }
    adj_next_transmittance = adj_trans_before;
  }
  }
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (local_frame == 0u) {
    float loss_sum = 0.0f;
    for (uint frame_id = 0u; frame_id < frames_in_chunk; ++frame_id) {
      loss_sum += tg_loss[frame_id];
    }
    atomic_fetch_add_explicit(&loss_f32[0], loss_sum, memory_order_relaxed);
  }

		  if (reduce_small_sites) {
		    if (local_frame < site_count) {
		      float4 grad_sum = float4(0.0f, 0.0f, 0.0f, 0.0f);
		      for (uint frame_id = 0u; frame_id < frames_in_chunk; ++frame_id) {
		        grad_sum += tg_site_grad[
		            frame_id * WF2_ENDPOINT_EDIT_FRAMEGROUP_REDUCE_SITES + local_frame];
		      }
		      wf2_atomic_add4(grad_site_rgba_f32, local_frame * 4u, grad_sum);
		    }
		  }
			}

kernel void wf2_endpoint_record_delta_replace_coeff16_i16x3_framegroup16_ownerreduce_mse_vjp_direct_atomic_rgb_only_tensor(
    device const half* coeff_f16 [[buffer(0)]],
    device const float* frame_t_f32 [[buffer(1)]],
    device const int* base_offsets_i32 [[buffer(2)]],
    device const short* base_record_i16 [[buffer(3)]],
    device const int* track_change_offsets_i32 [[buffer(4)]],
    device const short* track_chunk_change_offsets_i16 [[buffer(5)]],
    device const int* track_chunk_owner_offsets_i32 [[buffer(6)]],
    device const short* track_chunk_owner_i16 [[buffer(7)]],
    device const int* change_frame_i32 [[buffer(8)]],
    device const int* change_offsets_i32 [[buffer(9)]],
    device const short* change_record_i16 [[buffer(10)]],
    device const float* site_rgba_f32 [[buffer(11)]],
    device const float* target_rgb_f32 [[buffer(12)]],
    device const int* config_i32 [[buffer(13)]],
    device const float* config_f32 [[buffer(14)]],
    device atomic_float* loss_f32 [[buffer(15)]],
    device atomic_float* grad_site_rgba_f32 [[buffer(16)]],
    uint3 local_pos [[thread_position_in_threadgroup]],
    uint3 group_id [[threadgroup_position_in_grid]]) {
  const uint local_frame = local_pos.x;
  const uint boundary_count = uint(config_i32[0]);
  const uint track_count = uint(config_i32[1]);
  const uint frame_count = uint(config_i32[2]);
  const uint site_count = uint(config_i32[3]);
  const uint base_record_count = uint(config_i32[4]);
  const uint change_count = uint(config_i32[5]);
  const uint change_record_count = uint(config_i32[6]);
  const uint owner_list_count = uint(config_i32[7]);
  const uint chunk_count = (frame_count + WF2_ENDPOINT_DELTA_FRAMEGROUP_MAX_FRAMES - 1u) /
      WF2_ENDPOINT_DELTA_FRAMEGROUP_MAX_FRAMES;
  const uint group_index = group_id.x;
  const uint track_id = group_index / max(chunk_count, 1u);
  const uint chunk_id = group_index - track_id * max(chunk_count, 1u);
  const uint frame_start = chunk_id * WF2_ENDPOINT_DELTA_FRAMEGROUP_MAX_FRAMES;
  const uint frames_in_chunk = frame_start < frame_count
      ? min(WF2_ENDPOINT_DELTA_FRAMEGROUP_MAX_FRAMES, frame_count - frame_start)
      : 0u;

  threadgroup int tg_valid[WF2_ENDPOINT_DELTA_FRAMEGROUP_MAX_FRAMES];
  threadgroup int tg_source[WF2_ENDPOINT_DELTA_FRAMEGROUP_MAX_FRAMES];
  threadgroup int tg_begin[WF2_ENDPOINT_DELTA_FRAMEGROUP_MAX_FRAMES];
  threadgroup int tg_end[WF2_ENDPOINT_DELTA_FRAMEGROUP_MAX_FRAMES];
  threadgroup int tg_owner_count;
  threadgroup int tg_owner_ids[WF2_ENDPOINT_EDIT_FRAMEGROUP_REDUCE_SITES];
  threadgroup float tg_loss[WF2_ENDPOINT_DELTA_FRAMEGROUP_MAX_FRAMES];
  threadgroup float4 tg_site_grad[
      WF2_ENDPOINT_DELTA_FRAMEGROUP_MAX_FRAMES * WF2_ENDPOINT_EDIT_FRAMEGROUP_REDUCE_SITES];

  if (local_frame == 0u) {
    tg_owner_count = -1;
    for (uint site_slot = 0u; site_slot < WF2_ENDPOINT_EDIT_FRAMEGROUP_REDUCE_SITES; ++site_slot) {
      tg_owner_ids[site_slot] = -1;
    }
    for (uint frame_id = 0u; frame_id < WF2_ENDPOINT_DELTA_FRAMEGROUP_MAX_FRAMES; ++frame_id) {
      tg_valid[frame_id] = 0;
      tg_source[frame_id] = 0;
      tg_begin[frame_id] = 0;
      tg_end[frame_id] = 0;
    }

    bool ok = track_id < track_count && frames_in_chunk > 0u;
    int change_begin_raw = 0;
    int change_end_raw = 0;
    int chunk_change_begin_raw = 0;
    int chunk_change_end_raw = 0;
    if (ok) {
      change_begin_raw = track_change_offsets_i32[track_id];
      change_end_raw = track_change_offsets_i32[track_id + 1u];
      const uint chunk_offset_base = track_id * (chunk_count + 1u) + chunk_id;
      chunk_change_begin_raw = int(track_chunk_change_offsets_i16[chunk_offset_base]);
      chunk_change_end_raw = int(track_chunk_change_offsets_i16[chunk_offset_base + 1u]);
      ok = change_begin_raw >= 0 && change_end_raw >= change_begin_raw && uint(change_end_raw) <= change_count &&
          chunk_change_begin_raw >= change_begin_raw && chunk_change_end_raw >= chunk_change_begin_raw &&
          chunk_change_end_raw <= change_end_raw;
    }
    if (ok) {
      const uint owner_offset_index = track_id * chunk_count + chunk_id;
      const int owner_begin_raw = track_chunk_owner_offsets_i32[owner_offset_index];
      const int owner_end_raw = track_chunk_owner_offsets_i32[owner_offset_index + 1u];
      const int owner_count_raw = owner_end_raw - owner_begin_raw;
      bool owner_ok = owner_begin_raw >= 0 && owner_end_raw >= owner_begin_raw &&
          uint(owner_end_raw) <= owner_list_count &&
          owner_count_raw <= int(WF2_ENDPOINT_EDIT_FRAMEGROUP_REDUCE_SITES);
      if (owner_ok) {
        for (uint owner_slot = 0u; owner_slot < uint(owner_count_raw); ++owner_slot) {
          const int owner_raw = int(track_chunk_owner_i16[uint(owner_begin_raw) + owner_slot]);
          if (owner_raw < 0 || uint(owner_raw) >= site_count) {
            owner_ok = false;
            break;
          }
          tg_owner_ids[owner_slot] = owner_raw;
        }
        tg_owner_count = owner_ok ? owner_count_raw : -1;
      }
    }
    int selected_change = -1;
    if (ok && chunk_change_begin_raw > change_begin_raw) {
      selected_change = chunk_change_begin_raw - 1;
      while (selected_change >= change_begin_raw && change_frame_i32[uint(selected_change)] < 0) {
        selected_change -= 1;
      }
      if (selected_change < change_begin_raw) {
        selected_change = -1;
      }
    }

    uint change_cursor = ok ? uint(chunk_change_begin_raw) : 0u;
    const uint change_end = ok ? uint(chunk_change_end_raw) : 0u;

    for (uint local_frame_id = 0u; local_frame_id < frames_in_chunk; ++local_frame_id) {
      const uint global_frame_id = frame_start + local_frame_id;
      while (ok && change_cursor < change_end) {
        const int changed_frame = change_frame_i32[change_cursor];
        if (changed_frame < 0) {
          change_cursor += 1u;
          continue;
        }
        if (uint(changed_frame) > global_frame_id) {
          break;
        }

        selected_change = int(change_cursor);
        change_cursor += 1u;
      }

      if (ok) {
        const bool use_change = selected_change >= 0;
        const int begin_raw =
            use_change ? change_offsets_i32[uint(selected_change)] : base_offsets_i32[track_id];
        const int end_raw =
            use_change ? change_offsets_i32[uint(selected_change) + 1u] : base_offsets_i32[track_id + 1u];
        const uint record_count = use_change ? change_record_count : base_record_count;
        ok = begin_raw >= 0 && end_raw >= begin_raw && uint(end_raw) <= record_count &&
            uint(end_raw - begin_raw) <= WF2_MAX_REALRAY_SEGMENTS;
        if (!ok) {
          break;
        }
        tg_source[local_frame_id] = use_change ? 1 : 0;
        tg_begin[local_frame_id] = begin_raw;
        tg_end[local_frame_id] = end_raw;
        tg_valid[local_frame_id] = 1;
      }
    }
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  const bool reduce_small_sites = site_count <= WF2_ENDPOINT_EDIT_FRAMEGROUP_REDUCE_SITES;
  const bool reduce_owner_list = (!reduce_small_sites) && tg_owner_count >= 0;
  if (reduce_small_sites || reduce_owner_list) {
    const uint site_grad_base = local_frame * WF2_ENDPOINT_EDIT_FRAMEGROUP_REDUCE_SITES;
    for (uint site_slot = 0u; site_slot < WF2_ENDPOINT_EDIT_FRAMEGROUP_REDUCE_SITES; ++site_slot) {
      tg_site_grad[site_grad_base + site_slot] = float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
  }
  tg_loss[local_frame] = 0.0f;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const bool active = track_id < track_count && local_frame < frames_in_chunk && tg_valid[local_frame] != 0;
  if (active) {
    const uint global_frame_id = frame_start + local_frame;
    const uint sample_id = track_id * frame_count + global_frame_id;
    const uint total_samples = track_count * frame_count;
    const float near_depth = config_f32[0];
    const float far_depth = config_f32[1];
    const float invalid_epsilon = config_f32[2];
    const float transmittance_threshold = config_f32[3];
    const float t = frame_t_f32[global_frame_id];
    const bool use_change = tg_source[local_frame] != 0;
    const int begin_raw = tg_begin[local_frame];
    const int end_raw = tg_end[local_frame];
    const bool valid_row_bounds = end_raw >= begin_raw;
    const uint row_count = valid_row_bounds ? uint(end_raw - begin_raw) : 0u;
    const uint out_base = sample_id * 3u;
    const float3 target_rgb = float3(
        target_rgb_f32[out_base + 0u],
        target_rgb_f32[out_base + 1u],
        target_rgb_f32[out_base + 2u]);
    const float inv_element_count = 1.0f / float(total_samples * 3u);
    if (valid_row_bounds && row_count == 0u) {
      tg_loss[local_frame] = dot(target_rgb, target_rgb) * inv_element_count;
    } else if (valid_row_bounds) {
      uint owners[WF2_MAX_REALRAY_SEGMENTS];
      float lengths[WF2_MAX_REALRAY_SEGMENTS];
      float trans_before[WF2_MAX_REALRAY_SEGMENTS];
      float segment_trans[WF2_MAX_REALRAY_SEGMENTS];
      float segment_alpha[WF2_MAX_REALRAY_SEGMENTS];
      float weights[WF2_MAX_REALRAY_SEGMENTS];
      float3 segment_rgb[WF2_MAX_REALRAY_SEGMENTS];

      float3 rgb_accum = float3(0.0f, 0.0f, 0.0f);
      float transmittance = 1.0f;
      uint local_run_count = 0u;
      int cached_right_cut = -2147483648;
      float cached_right_depth = 0.0f;
      bool cached_right_valid = false;
      for (uint cursor = 0u; cursor < row_count; ++cursor) {
        if (transmittance <= transmittance_threshold || local_run_count >= WF2_MAX_REALRAY_SEGMENTS) {
          break;
        }
        const uint record_base = uint(begin_raw) + cursor;
        const uint record_offset = record_base * 3u;
        const int owner_raw =
            use_change ? int(change_record_i16[record_offset + 0u]) : int(base_record_i16[record_offset + 0u]);
        if (owner_raw < 0 || uint(owner_raw) >= site_count) {
          continue;
        }
        float start_depth = 0.0f;
        float end_depth = 0.0f;
        const int left_cut =
            use_change ? int(change_record_i16[record_offset + 1u]) : int(base_record_i16[record_offset + 1u]);
        const int right_cut =
            use_change ? int(change_record_i16[record_offset + 2u]) : int(base_record_i16[record_offset + 2u]);
        bool start_valid = false;
        if (cached_right_valid && left_cut == cached_right_cut) {
          start_depth = cached_right_depth;
          start_valid = true;
        } else {
          start_valid = wf2_endpoint_record_coeff16_cut_depth(
              coeff_f16,
              boundary_count,
              track_id,
              left_cut,
              t,
              near_depth,
              far_depth,
              invalid_epsilon,
              start_depth);
        }
        if (!start_valid ||
            !wf2_endpoint_record_coeff16_cut_depth(
                coeff_f16,
                boundary_count,
                track_id,
                right_cut,
                t,
                near_depth,
                far_depth,
                invalid_epsilon,
                end_depth)) {
          cached_right_valid = false;
          continue;
        }
        cached_right_cut = right_cut;
        cached_right_depth = end_depth;
        cached_right_valid = true;
        const float length = end_depth - start_depth;
        if (!(length > 1.0e-8f)) {
          continue;
        }
        const uint owner = uint(owner_raw);
        const uint rgba_base = owner * 4u;
        const float density = max(site_rgba_f32[rgba_base + 3u], 0.0f);
        const float seg_trans = exp(-density * length);
        const float seg_alpha = 1.0f - seg_trans;
        const float weight = transmittance * seg_alpha;
        const float3 rgb = float3(
            site_rgba_f32[rgba_base + 0u],
            site_rgba_f32[rgba_base + 1u],
            site_rgba_f32[rgba_base + 2u]);

        owners[local_run_count] = owner;
        lengths[local_run_count] = length;
        trans_before[local_run_count] = transmittance;
        segment_trans[local_run_count] = seg_trans;
        segment_alpha[local_run_count] = seg_alpha;
        weights[local_run_count] = weight;
        segment_rgb[local_run_count] = rgb;
        local_run_count += 1u;

        rgb_accum += weight * rgb;
        transmittance *= seg_trans;
      }

      const float3 diff = rgb_accum - target_rgb;
      const float sample_loss = dot(diff, diff) * inv_element_count;
      tg_loss[local_frame] = sample_loss;
      const float3 grad_rgb = (2.0f * inv_element_count) * diff;

      float adj_next_transmittance = 0.0f;
      for (int run_id = int(local_run_count) - 1; run_id >= 0; --run_id) {
        const uint owner = owners[run_id];
        const uint rgba_base = owner * 4u;
        const float raw_density = site_rgba_f32[rgba_base + 3u];
        const float adj_weight = dot(grad_rgb, segment_rgb[run_id]);
        const float adj_trans_before =
            adj_weight * segment_alpha[run_id] +
            adj_next_transmittance * segment_trans[run_id];
        const float adj_segment_alpha = adj_weight * trans_before[run_id];
        const float adj_segment_trans = adj_next_transmittance * trans_before[run_id] - adj_segment_alpha;
        float4 grad_rgba = float4(
            weights[run_id] * grad_rgb.x,
            weights[run_id] * grad_rgb.y,
            weights[run_id] * grad_rgb.z,
            0.0f);
        if (raw_density > 0.0f) {
          grad_rgba.w = adj_segment_trans * (-lengths[run_id] * segment_trans[run_id]);
        }
        if (reduce_small_sites) {
          tg_site_grad[local_frame * WF2_ENDPOINT_EDIT_FRAMEGROUP_REDUCE_SITES + owner] += grad_rgba;
        } else if (reduce_owner_list) {
          int owner_slot = -1;
          for (int slot = 0; slot < tg_owner_count; ++slot) {
            if (tg_owner_ids[slot] == int(owner)) {
              owner_slot = slot;
              break;
            }
          }
          if (owner_slot >= 0) {
            tg_site_grad[
                local_frame * WF2_ENDPOINT_EDIT_FRAMEGROUP_REDUCE_SITES + uint(owner_slot)] += grad_rgba;
          } else {
            wf2_atomic_add4(grad_site_rgba_f32, rgba_base, grad_rgba);
          }
        } else {
          wf2_atomic_add4(grad_site_rgba_f32, rgba_base, grad_rgba);
        }
        adj_next_transmittance = adj_trans_before;
      }
    }
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (local_frame == 0u) {
    float loss_sum = 0.0f;
    for (uint frame_id = 0u; frame_id < frames_in_chunk; ++frame_id) {
      loss_sum += tg_loss[frame_id];
    }
    atomic_fetch_add_explicit(&loss_f32[0], loss_sum, memory_order_relaxed);
  }

  if (reduce_small_sites) {
    if (local_frame < site_count) {
      float4 grad_sum = float4(0.0f, 0.0f, 0.0f, 0.0f);
      for (uint frame_id = 0u; frame_id < frames_in_chunk; ++frame_id) {
        grad_sum += tg_site_grad[
            frame_id * WF2_ENDPOINT_EDIT_FRAMEGROUP_REDUCE_SITES + local_frame];
      }
      wf2_atomic_add4(grad_site_rgba_f32, local_frame * 4u, grad_sum);
    }
  } else if (reduce_owner_list && local_frame < uint(tg_owner_count)) {
    const uint owner = uint(tg_owner_ids[local_frame]);
    float4 grad_sum = float4(0.0f, 0.0f, 0.0f, 0.0f);
    for (uint frame_id = 0u; frame_id < frames_in_chunk; ++frame_id) {
      grad_sum += tg_site_grad[
          frame_id * WF2_ENDPOINT_EDIT_FRAMEGROUP_REDUCE_SITES + local_frame];
    }
    wf2_atomic_add4(grad_site_rgba_f32, owner * 4u, grad_sum);
  }
}

kernel void wf2_endpoint_record_delta_replace_coeff16_i16cols_framegroup16_mse_vjp_direct_atomic_rgb_only_tensor(
    device const half* coeff_f16 [[buffer(0)]],
    device const float* frame_t_f32 [[buffer(1)]],
    device const int* base_offsets_i32 [[buffer(2)]],
    device const short* base_record_i16 [[buffer(3)]],
    device const int* track_change_offsets_i32 [[buffer(4)]],
    device const short* track_chunk_change_offsets_i16 [[buffer(5)]],
    device const int* change_frame_i32 [[buffer(6)]],
    device const int* change_offsets_i32 [[buffer(7)]],
    device const short* change_record_i16 [[buffer(8)]],
    device const float* site_rgba_f32 [[buffer(9)]],
    device const float* target_rgb_f32 [[buffer(10)]],
    device const int* config_i32 [[buffer(11)]],
    device const float* config_f32 [[buffer(12)]],
    device atomic_float* loss_f32 [[buffer(13)]],
    device atomic_float* grad_site_rgba_f32 [[buffer(14)]],
    uint3 local_pos [[thread_position_in_threadgroup]],
    uint3 group_id [[threadgroup_position_in_grid]]) {
  const uint local_frame = local_pos.x;
  const uint boundary_count = uint(config_i32[0]);
  const uint track_count = uint(config_i32[1]);
  const uint frame_count = uint(config_i32[2]);
  const uint site_count = uint(config_i32[3]);
  const uint base_record_count = uint(config_i32[4]);
  const uint change_count = uint(config_i32[5]);
  const uint change_record_count = uint(config_i32[6]);
  const uint chunk_count = (frame_count + WF2_ENDPOINT_DELTA_FRAMEGROUP_MAX_FRAMES - 1u) /
      WF2_ENDPOINT_DELTA_FRAMEGROUP_MAX_FRAMES;
  const uint group_index = group_id.x;
  const uint track_id = group_index / max(chunk_count, 1u);
  const uint chunk_id = group_index - track_id * max(chunk_count, 1u);
  const uint frame_start = chunk_id * WF2_ENDPOINT_DELTA_FRAMEGROUP_MAX_FRAMES;
  const uint frames_in_chunk = frame_start < frame_count
      ? min(WF2_ENDPOINT_DELTA_FRAMEGROUP_MAX_FRAMES, frame_count - frame_start)
      : 0u;

  threadgroup int tg_valid[WF2_ENDPOINT_DELTA_FRAMEGROUP_MAX_FRAMES];
  threadgroup int tg_source[WF2_ENDPOINT_DELTA_FRAMEGROUP_MAX_FRAMES];
  threadgroup int tg_begin[WF2_ENDPOINT_DELTA_FRAMEGROUP_MAX_FRAMES];
  threadgroup int tg_end[WF2_ENDPOINT_DELTA_FRAMEGROUP_MAX_FRAMES];
  threadgroup float tg_loss[WF2_ENDPOINT_DELTA_FRAMEGROUP_MAX_FRAMES];
  threadgroup float4 tg_site_grad[
      WF2_ENDPOINT_DELTA_FRAMEGROUP_MAX_FRAMES * WF2_ENDPOINT_EDIT_FRAMEGROUP_REDUCE_SITES];

  if (local_frame == 0u) {
    for (uint frame_id = 0u; frame_id < WF2_ENDPOINT_DELTA_FRAMEGROUP_MAX_FRAMES; ++frame_id) {
      tg_valid[frame_id] = 0;
      tg_source[frame_id] = 0;
      tg_begin[frame_id] = 0;
      tg_end[frame_id] = 0;
    }

    bool ok = track_id < track_count && frames_in_chunk > 0u;
    int change_begin_raw = 0;
    int change_end_raw = 0;
    int chunk_change_begin_raw = 0;
    int chunk_change_end_raw = 0;
    if (ok) {
      change_begin_raw = track_change_offsets_i32[track_id];
      change_end_raw = track_change_offsets_i32[track_id + 1u];
      const uint chunk_offset_base = track_id * (chunk_count + 1u) + chunk_id;
      chunk_change_begin_raw = int(track_chunk_change_offsets_i16[chunk_offset_base]);
      chunk_change_end_raw = int(track_chunk_change_offsets_i16[chunk_offset_base + 1u]);
      ok = change_begin_raw >= 0 && change_end_raw >= change_begin_raw && uint(change_end_raw) <= change_count &&
          chunk_change_begin_raw >= change_begin_raw && chunk_change_end_raw >= chunk_change_begin_raw &&
          chunk_change_end_raw <= change_end_raw;
    }
    int selected_change = -1;
    if (ok && chunk_change_begin_raw > change_begin_raw) {
      selected_change = chunk_change_begin_raw - 1;
      while (selected_change >= change_begin_raw && change_frame_i32[uint(selected_change)] < 0) {
        selected_change -= 1;
      }
      if (selected_change < change_begin_raw) {
        selected_change = -1;
      }
    }

    uint change_cursor = ok ? uint(chunk_change_begin_raw) : 0u;
    const uint change_end = ok ? uint(chunk_change_end_raw) : 0u;

    for (uint local_frame_id = 0u; local_frame_id < frames_in_chunk; ++local_frame_id) {
      const uint global_frame_id = frame_start + local_frame_id;
      while (ok && change_cursor < change_end) {
        const int changed_frame = change_frame_i32[change_cursor];
        if (changed_frame < 0) {
          change_cursor += 1u;
          continue;
        }
        if (uint(changed_frame) > global_frame_id) {
          break;
        }

        selected_change = int(change_cursor);
        change_cursor += 1u;
      }

      if (ok) {
        const bool use_change = selected_change >= 0;
        const int begin_raw =
            use_change ? change_offsets_i32[uint(selected_change)] : base_offsets_i32[track_id];
        const int end_raw =
            use_change ? change_offsets_i32[uint(selected_change) + 1u] : base_offsets_i32[track_id + 1u];
        const uint record_count = use_change ? change_record_count : base_record_count;
        ok = begin_raw >= 0 && end_raw >= begin_raw && uint(end_raw) <= record_count &&
            uint(end_raw - begin_raw) <= WF2_MAX_REALRAY_SEGMENTS;
        if (!ok) {
          break;
        }
        tg_source[local_frame_id] = use_change ? 1 : 0;
        tg_begin[local_frame_id] = begin_raw;
        tg_end[local_frame_id] = end_raw;
	        tg_valid[local_frame_id] = 1;
	      }
	    }

	  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

	  const bool reduce_small_sites = site_count <= WF2_ENDPOINT_EDIT_FRAMEGROUP_REDUCE_SITES;
	  if (reduce_small_sites) {
    const uint site_grad_base = local_frame * WF2_ENDPOINT_EDIT_FRAMEGROUP_REDUCE_SITES;
    for (uint site_slot = 0u; site_slot < WF2_ENDPOINT_EDIT_FRAMEGROUP_REDUCE_SITES; ++site_slot) {
      tg_site_grad[site_grad_base + site_slot] = float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
  }
  tg_loss[local_frame] = 0.0f;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const bool active = track_id < track_count && local_frame < frames_in_chunk && tg_valid[local_frame] != 0;
  if (active) {

  const uint global_frame_id = frame_start + local_frame;
  const uint sample_id = track_id * frame_count + global_frame_id;
  const uint total_samples = track_count * frame_count;
  const float near_depth = config_f32[0];
  const float far_depth = config_f32[1];
  const float invalid_epsilon = config_f32[2];
  const float transmittance_threshold = config_f32[3];
  const float t = frame_t_f32[global_frame_id];
  const bool use_change = tg_source[local_frame] != 0;
  const int begin_raw = tg_begin[local_frame];
  const int end_raw = tg_end[local_frame];
  const bool valid_row_bounds = end_raw >= begin_raw;
  const uint row_count = valid_row_bounds ? uint(end_raw - begin_raw) : 0u;
  const uint out_base = sample_id * 3u;
  const float3 target_rgb = float3(
      target_rgb_f32[out_base + 0u],
      target_rgb_f32[out_base + 1u],
      target_rgb_f32[out_base + 2u]);
  const float inv_element_count = 1.0f / float(total_samples * 3u);
  if (valid_row_bounds && row_count == 0u) {
    tg_loss[local_frame] = dot(target_rgb, target_rgb) * inv_element_count;
  } else if (valid_row_bounds) {

  uint owners[WF2_MAX_REALRAY_SEGMENTS];
  float lengths[WF2_MAX_REALRAY_SEGMENTS];
  float trans_before[WF2_MAX_REALRAY_SEGMENTS];
  float segment_trans[WF2_MAX_REALRAY_SEGMENTS];
  float segment_alpha[WF2_MAX_REALRAY_SEGMENTS];
  float weights[WF2_MAX_REALRAY_SEGMENTS];
  float3 segment_rgb[WF2_MAX_REALRAY_SEGMENTS];

  float3 rgb_accum = float3(0.0f, 0.0f, 0.0f);
  float transmittance = 1.0f;
  uint local_run_count = 0u;
  int cached_right_cut = -2147483648;
  float cached_right_depth = 0.0f;
  bool cached_right_valid = false;
  for (uint cursor = 0u; cursor < row_count; ++cursor) {
    if (transmittance <= transmittance_threshold || local_run_count >= WF2_MAX_REALRAY_SEGMENTS) {
      break;
    }
    const uint record_base = uint(begin_raw) + cursor;
    const uint record_count = use_change ? change_record_count : base_record_count;
    const int owner_raw =
        use_change ? int(change_record_i16[record_base]) : int(base_record_i16[record_base]);
    if (owner_raw < 0 || uint(owner_raw) >= site_count) {
      continue;
    }
    float start_depth = 0.0f;
    float end_depth = 0.0f;
    const int left_cut = use_change
        ? int(change_record_i16[record_count + record_base])
        : int(base_record_i16[record_count + record_base]);
    const int right_cut = use_change
        ? int(change_record_i16[2u * record_count + record_base])
        : int(base_record_i16[2u * record_count + record_base]);
    bool start_valid = false;
    if (cached_right_valid && left_cut == cached_right_cut) {
      start_depth = cached_right_depth;
      start_valid = true;
    } else {
      start_valid = wf2_endpoint_record_coeff16_cut_depth(
          coeff_f16,
          boundary_count,
          track_id,
          left_cut,
          t,
          near_depth,
          far_depth,
          invalid_epsilon,
          start_depth);
    }
    if (!start_valid ||
        !wf2_endpoint_record_coeff16_cut_depth(
            coeff_f16,
            boundary_count,
            track_id,
            right_cut,
            t,
            near_depth,
            far_depth,
            invalid_epsilon,
            end_depth)) {
      cached_right_valid = false;
      continue;
    }
    cached_right_cut = right_cut;
    cached_right_depth = end_depth;
    cached_right_valid = true;
    const float length = end_depth - start_depth;
    if (!(length > 1.0e-8f)) {
      continue;
    }
    const uint owner = uint(owner_raw);
    const uint rgba_base = owner * 4u;
    const float density = max(site_rgba_f32[rgba_base + 3u], 0.0f);
    const float seg_trans = exp(-density * length);
    const float seg_alpha = 1.0f - seg_trans;
    const float weight = transmittance * seg_alpha;
    const float3 rgb = float3(
        site_rgba_f32[rgba_base + 0u],
        site_rgba_f32[rgba_base + 1u],
        site_rgba_f32[rgba_base + 2u]);

    owners[local_run_count] = owner;
    lengths[local_run_count] = length;
    trans_before[local_run_count] = transmittance;
    segment_trans[local_run_count] = seg_trans;
    segment_alpha[local_run_count] = seg_alpha;
    weights[local_run_count] = weight;
    segment_rgb[local_run_count] = rgb;
    local_run_count += 1u;

    rgb_accum += weight * rgb;
    transmittance *= seg_trans;
  }

  const float3 diff = rgb_accum - target_rgb;
  const float sample_loss = dot(diff, diff) * inv_element_count;
  tg_loss[local_frame] = sample_loss;
  const float3 grad_rgb = (2.0f * inv_element_count) * diff;

  float adj_next_transmittance = 0.0f;
  for (int run_id = int(local_run_count) - 1; run_id >= 0; --run_id) {
    const uint owner = owners[run_id];
    const uint rgba_base = owner * 4u;
    const float raw_density = site_rgba_f32[rgba_base + 3u];
    const float adj_weight = dot(grad_rgb, segment_rgb[run_id]);
    const float adj_trans_before =
        adj_weight * segment_alpha[run_id] +
        adj_next_transmittance * segment_trans[run_id];
    const float adj_segment_alpha = adj_weight * trans_before[run_id];
    const float adj_segment_trans = adj_next_transmittance * trans_before[run_id] - adj_segment_alpha;
    float4 grad_rgba = float4(
        weights[run_id] * grad_rgb.x,
        weights[run_id] * grad_rgb.y,
        weights[run_id] * grad_rgb.z,
        0.0f);
    if (raw_density > 0.0f) {
      grad_rgba.w = adj_segment_trans * (-lengths[run_id] * segment_trans[run_id]);
    }
		    if (reduce_small_sites) {
		      tg_site_grad[local_frame * WF2_ENDPOINT_EDIT_FRAMEGROUP_REDUCE_SITES + owner] += grad_rgba;
		    } else {
		      wf2_atomic_add4(grad_site_rgba_f32, rgba_base, grad_rgba);
		    }
    adj_next_transmittance = adj_trans_before;
  }
  }
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (local_frame == 0u) {
    float loss_sum = 0.0f;
    for (uint frame_id = 0u; frame_id < frames_in_chunk; ++frame_id) {
      loss_sum += tg_loss[frame_id];
    }
    atomic_fetch_add_explicit(&loss_f32[0], loss_sum, memory_order_relaxed);
  }

		  if (reduce_small_sites) {
		    if (local_frame < site_count) {
		      float4 grad_sum = float4(0.0f, 0.0f, 0.0f, 0.0f);
		      for (uint frame_id = 0u; frame_id < frames_in_chunk; ++frame_id) {
		        grad_sum += tg_site_grad[
		            frame_id * WF2_ENDPOINT_EDIT_FRAMEGROUP_REDUCE_SITES + local_frame];
		      }
		      wf2_atomic_add4(grad_site_rgba_f32, local_frame * 4u, grad_sum);
		    }
		  }
		}

kernel void wf2_endpoint_record_delta_replace_coeff16_i16x3_framegroup64_mse_vjp_direct_atomic_rgb_only_tensor(
    device const half* coeff_f16 [[buffer(0)]],
    device const float* frame_t_f32 [[buffer(1)]],
    device const int* base_offsets_i32 [[buffer(2)]],
    device const short* base_record_i16 [[buffer(3)]],
    device const int* track_change_offsets_i32 [[buffer(4)]],
    device const short* track_chunk_change_offsets_i16 [[buffer(5)]],
    device const int* change_frame_i32 [[buffer(6)]],
    device const int* change_offsets_i32 [[buffer(7)]],
    device const short* change_record_i16 [[buffer(8)]],
    device const float* site_rgba_f32 [[buffer(9)]],
    device const float* target_rgb_f32 [[buffer(10)]],
    device const int* config_i32 [[buffer(11)]],
    device const float* config_f32 [[buffer(12)]],
    device atomic_float* loss_f32 [[buffer(13)]],
    device atomic_float* grad_site_rgba_f32 [[buffer(14)]],
    uint3 local_pos [[thread_position_in_threadgroup]],
    uint3 group_id [[threadgroup_position_in_grid]]) {
  const uint local_frame = local_pos.x;
  const uint boundary_count = uint(config_i32[0]);
  const uint track_count = uint(config_i32[1]);
  const uint frame_count = uint(config_i32[2]);
  const uint site_count = uint(config_i32[3]);
  const uint base_record_count = uint(config_i32[4]);
  const uint change_count = uint(config_i32[5]);
  const uint change_record_count = uint(config_i32[6]);
  const uint chunk_count = (frame_count + WF2_ENDPOINT_DELTA_FRAMEGROUP64_MAX_FRAMES - 1u) /
      WF2_ENDPOINT_DELTA_FRAMEGROUP64_MAX_FRAMES;
  const uint group_index = group_id.x;
  const uint track_id = group_index / max(chunk_count, 1u);
  const uint chunk_id = group_index - track_id * max(chunk_count, 1u);
  const uint frame_start = chunk_id * WF2_ENDPOINT_DELTA_FRAMEGROUP64_MAX_FRAMES;
  const uint frames_in_chunk = frame_start < frame_count
      ? min(WF2_ENDPOINT_DELTA_FRAMEGROUP64_MAX_FRAMES, frame_count - frame_start)
      : 0u;

  threadgroup int tg_valid[WF2_ENDPOINT_DELTA_FRAMEGROUP64_MAX_FRAMES];
  threadgroup int tg_source[WF2_ENDPOINT_DELTA_FRAMEGROUP64_MAX_FRAMES];
  threadgroup int tg_begin[WF2_ENDPOINT_DELTA_FRAMEGROUP64_MAX_FRAMES];
  threadgroup int tg_end[WF2_ENDPOINT_DELTA_FRAMEGROUP64_MAX_FRAMES];
  threadgroup float tg_loss[WF2_ENDPOINT_DELTA_FRAMEGROUP64_MAX_FRAMES];
  threadgroup float4 tg_site_grad[
      WF2_ENDPOINT_DELTA_FRAMEGROUP64_MAX_FRAMES * WF2_ENDPOINT_EDIT_FRAMEGROUP_REDUCE_SITES];

  if (local_frame == 0u) {
    for (uint frame_id = 0u; frame_id < WF2_ENDPOINT_DELTA_FRAMEGROUP64_MAX_FRAMES; ++frame_id) {
      tg_valid[frame_id] = 0;
      tg_source[frame_id] = 0;
      tg_begin[frame_id] = 0;
      tg_end[frame_id] = 0;
    }

    bool ok = track_id < track_count && frames_in_chunk > 0u;
    int change_begin_raw = 0;
    int change_end_raw = 0;
    int chunk_change_begin_raw = 0;
    int chunk_change_end_raw = 0;
    if (ok) {
      change_begin_raw = track_change_offsets_i32[track_id];
      change_end_raw = track_change_offsets_i32[track_id + 1u];
      const uint chunk_offset_base = track_id * (chunk_count + 1u) + chunk_id;
      chunk_change_begin_raw = int(track_chunk_change_offsets_i16[chunk_offset_base]);
      chunk_change_end_raw = int(track_chunk_change_offsets_i16[chunk_offset_base + 1u]);
      ok = change_begin_raw >= 0 && change_end_raw >= change_begin_raw && uint(change_end_raw) <= change_count &&
          chunk_change_begin_raw >= change_begin_raw && chunk_change_end_raw >= chunk_change_begin_raw &&
          chunk_change_end_raw <= change_end_raw;
    }
    int selected_change = -1;
    if (ok && chunk_change_begin_raw > change_begin_raw) {
      selected_change = chunk_change_begin_raw - 1;
      while (selected_change >= change_begin_raw && change_frame_i32[uint(selected_change)] < 0) {
        selected_change -= 1;
      }
      if (selected_change < change_begin_raw) {
        selected_change = -1;
      }
    }

    uint change_cursor = ok ? uint(chunk_change_begin_raw) : 0u;
    const uint change_end = ok ? uint(chunk_change_end_raw) : 0u;

    for (uint local_frame_id = 0u; local_frame_id < frames_in_chunk; ++local_frame_id) {
      const uint global_frame_id = frame_start + local_frame_id;
      while (ok && change_cursor < change_end) {
        const int changed_frame = change_frame_i32[change_cursor];
        if (changed_frame < 0) {
          change_cursor += 1u;
          continue;
        }
        if (uint(changed_frame) > global_frame_id) {
          break;
        }

        selected_change = int(change_cursor);
        change_cursor += 1u;
      }

      if (ok) {
        const bool use_change = selected_change >= 0;
        const int begin_raw =
            use_change ? change_offsets_i32[uint(selected_change)] : base_offsets_i32[track_id];
        const int end_raw =
            use_change ? change_offsets_i32[uint(selected_change) + 1u] : base_offsets_i32[track_id + 1u];
        const uint record_count = use_change ? change_record_count : base_record_count;
        ok = begin_raw >= 0 && end_raw >= begin_raw && uint(end_raw) <= record_count &&
            uint(end_raw - begin_raw) <= WF2_MAX_REALRAY_SEGMENTS;
        if (!ok) {
          break;
        }
        tg_source[local_frame_id] = use_change ? 1 : 0;
        tg_begin[local_frame_id] = begin_raw;
        tg_end[local_frame_id] = end_raw;
        tg_valid[local_frame_id] = 1;
      }
    }
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  const bool reduce_small_sites = site_count <= WF2_ENDPOINT_EDIT_FRAMEGROUP_REDUCE_SITES;
  if (reduce_small_sites) {
    const uint site_grad_base = local_frame * WF2_ENDPOINT_EDIT_FRAMEGROUP_REDUCE_SITES;
    for (uint site_slot = 0u; site_slot < WF2_ENDPOINT_EDIT_FRAMEGROUP_REDUCE_SITES; ++site_slot) {
      tg_site_grad[site_grad_base + site_slot] = float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
  }
  tg_loss[local_frame] = 0.0f;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const bool active = track_id < track_count && local_frame < frames_in_chunk && tg_valid[local_frame] != 0;
  if (active) {

  const uint global_frame_id = frame_start + local_frame;
  const uint sample_id = track_id * frame_count + global_frame_id;
  const uint total_samples = track_count * frame_count;
  const float near_depth = config_f32[0];
  const float far_depth = config_f32[1];
  const float invalid_epsilon = config_f32[2];
  const float transmittance_threshold = config_f32[3];
  const float t = frame_t_f32[global_frame_id];
  const bool use_change = tg_source[local_frame] != 0;
  const int begin_raw = tg_begin[local_frame];
  const int end_raw = tg_end[local_frame];
  const bool valid_row_bounds = end_raw >= begin_raw;
  const uint row_count = valid_row_bounds ? uint(end_raw - begin_raw) : 0u;
  const uint out_base = sample_id * 3u;
  const float3 target_rgb = float3(
      target_rgb_f32[out_base + 0u],
      target_rgb_f32[out_base + 1u],
      target_rgb_f32[out_base + 2u]);
  const float inv_element_count = 1.0f / float(total_samples * 3u);
  if (valid_row_bounds && row_count == 0u) {
    tg_loss[local_frame] = dot(target_rgb, target_rgb) * inv_element_count;
  } else if (valid_row_bounds) {

  uint owners[WF2_MAX_REALRAY_SEGMENTS];
  float lengths[WF2_MAX_REALRAY_SEGMENTS];
  float trans_before[WF2_MAX_REALRAY_SEGMENTS];
  float segment_trans[WF2_MAX_REALRAY_SEGMENTS];
  float segment_alpha[WF2_MAX_REALRAY_SEGMENTS];
  float weights[WF2_MAX_REALRAY_SEGMENTS];
  float3 segment_rgb[WF2_MAX_REALRAY_SEGMENTS];

  float3 rgb_accum = float3(0.0f, 0.0f, 0.0f);
  float transmittance = 1.0f;
  uint local_run_count = 0u;
  int cached_right_cut = -2147483648;
  float cached_right_depth = 0.0f;
  bool cached_right_valid = false;
  for (uint cursor = 0u; cursor < row_count; ++cursor) {
    if (transmittance <= transmittance_threshold || local_run_count >= WF2_MAX_REALRAY_SEGMENTS) {
      break;
    }
    const uint record_base = uint(begin_raw) + cursor;
    const uint record_offset = record_base * 3u;
    const int owner_raw =
        use_change ? int(change_record_i16[record_offset + 0u]) : int(base_record_i16[record_offset + 0u]);
    if (owner_raw < 0 || uint(owner_raw) >= site_count) {
      continue;
    }
    float start_depth = 0.0f;
    float end_depth = 0.0f;
    const int left_cut =
        use_change ? int(change_record_i16[record_offset + 1u]) : int(base_record_i16[record_offset + 1u]);
    const int right_cut =
        use_change ? int(change_record_i16[record_offset + 2u]) : int(base_record_i16[record_offset + 2u]);
    bool start_valid = false;
    if (cached_right_valid && left_cut == cached_right_cut) {
      start_depth = cached_right_depth;
      start_valid = true;
    } else {
      start_valid = wf2_endpoint_record_coeff16_cut_depth(
          coeff_f16,
          boundary_count,
          track_id,
          left_cut,
          t,
          near_depth,
          far_depth,
          invalid_epsilon,
          start_depth);
    }
    if (!start_valid ||
        !wf2_endpoint_record_coeff16_cut_depth(
            coeff_f16,
            boundary_count,
            track_id,
            right_cut,
            t,
            near_depth,
            far_depth,
            invalid_epsilon,
            end_depth)) {
      cached_right_valid = false;
      continue;
    }
    cached_right_cut = right_cut;
    cached_right_depth = end_depth;
    cached_right_valid = true;
    const float length = end_depth - start_depth;
    if (!(length > 1.0e-8f)) {
      continue;
    }
    const uint owner = uint(owner_raw);
    const uint rgba_base = owner * 4u;
    const float density = max(site_rgba_f32[rgba_base + 3u], 0.0f);
    const float seg_trans = exp(-density * length);
    const float seg_alpha = 1.0f - seg_trans;
    const float weight = transmittance * seg_alpha;
    const float3 rgb = float3(
        site_rgba_f32[rgba_base + 0u],
        site_rgba_f32[rgba_base + 1u],
        site_rgba_f32[rgba_base + 2u]);

    owners[local_run_count] = owner;
    lengths[local_run_count] = length;
    trans_before[local_run_count] = transmittance;
    segment_trans[local_run_count] = seg_trans;
    segment_alpha[local_run_count] = seg_alpha;
    weights[local_run_count] = weight;
    segment_rgb[local_run_count] = rgb;
    local_run_count += 1u;

    rgb_accum += weight * rgb;
    transmittance *= seg_trans;
  }

  const float3 diff = rgb_accum - target_rgb;
  const float sample_loss = dot(diff, diff) * inv_element_count;
  tg_loss[local_frame] = sample_loss;
  const float3 grad_rgb = (2.0f * inv_element_count) * diff;

  float adj_next_transmittance = 0.0f;
  for (int run_id = int(local_run_count) - 1; run_id >= 0; --run_id) {
    const uint owner = owners[run_id];
    const uint rgba_base = owner * 4u;
    const float raw_density = site_rgba_f32[rgba_base + 3u];
    const float adj_weight = dot(grad_rgb, segment_rgb[run_id]);
    const float adj_trans_before =
        adj_weight * segment_alpha[run_id] +
        adj_next_transmittance * segment_trans[run_id];
    const float adj_segment_alpha = adj_weight * trans_before[run_id];
    const float adj_segment_trans = adj_next_transmittance * trans_before[run_id] - adj_segment_alpha;
    float4 grad_rgba = float4(
        weights[run_id] * grad_rgb.x,
        weights[run_id] * grad_rgb.y,
        weights[run_id] * grad_rgb.z,
        0.0f);
    if (raw_density > 0.0f) {
      grad_rgba.w = adj_segment_trans * (-lengths[run_id] * segment_trans[run_id]);
    }
    if (reduce_small_sites) {
      tg_site_grad[local_frame * WF2_ENDPOINT_EDIT_FRAMEGROUP_REDUCE_SITES + owner] += grad_rgba;
    } else {
      wf2_atomic_add4(grad_site_rgba_f32, rgba_base, grad_rgba);
    }
    adj_next_transmittance = adj_trans_before;
  }
  }
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (local_frame == 0u) {
    float loss_sum = 0.0f;
    for (uint frame_id = 0u; frame_id < frames_in_chunk; ++frame_id) {
      loss_sum += tg_loss[frame_id];
    }
    atomic_fetch_add_explicit(&loss_f32[0], loss_sum, memory_order_relaxed);
  }

  if (reduce_small_sites) {
    if (local_frame < site_count) {
      float4 grad_sum = float4(0.0f, 0.0f, 0.0f, 0.0f);
      for (uint frame_id = 0u; frame_id < frames_in_chunk; ++frame_id) {
        grad_sum += tg_site_grad[
            frame_id * WF2_ENDPOINT_EDIT_FRAMEGROUP_REDUCE_SITES + local_frame];
      }
      wf2_atomic_add4(grad_site_rgba_f32, local_frame * 4u, grad_sum);
    }
  }
}

kernel void wf2_endpoint_record_delta_replace_coeff16_i16x4_framegroup16_mse_vjp_direct_atomic_rgb_only_tensor(
    device const half* coeff_f16 [[buffer(0)]],
    device const float* frame_t_f32 [[buffer(1)]],
    device const int* base_offsets_i32 [[buffer(2)]],
    device const short4* base_record_i16x4 [[buffer(3)]],
    device const int* track_change_offsets_i32 [[buffer(4)]],
    device const short* track_chunk_change_offsets_i16 [[buffer(5)]],
    device const int* change_frame_i32 [[buffer(6)]],
    device const int* change_offsets_i32 [[buffer(7)]],
    device const short4* change_record_i16x4 [[buffer(8)]],
    device const float* site_rgba_f32 [[buffer(9)]],
    device const float* target_rgb_f32 [[buffer(10)]],
    device const int* config_i32 [[buffer(11)]],
    device const float* config_f32 [[buffer(12)]],
    device atomic_float* loss_f32 [[buffer(13)]],
    device atomic_float* grad_site_rgba_f32 [[buffer(14)]],
    uint3 local_pos [[thread_position_in_threadgroup]],
    uint3 group_id [[threadgroup_position_in_grid]]) {
  const uint local_frame = local_pos.x;
  const uint boundary_count = uint(config_i32[0]);
  const uint track_count = uint(config_i32[1]);
  const uint frame_count = uint(config_i32[2]);
  const uint site_count = uint(config_i32[3]);
  const uint base_record_count = uint(config_i32[4]);
  const uint change_count = uint(config_i32[5]);
  const uint change_record_count = uint(config_i32[6]);
  const uint chunk_count = (frame_count + WF2_ENDPOINT_DELTA_FRAMEGROUP_MAX_FRAMES - 1u) /
      WF2_ENDPOINT_DELTA_FRAMEGROUP_MAX_FRAMES;
  const uint group_index = group_id.x;
  const uint track_id = group_index / max(chunk_count, 1u);
  const uint chunk_id = group_index - track_id * max(chunk_count, 1u);
  const uint frame_start = chunk_id * WF2_ENDPOINT_DELTA_FRAMEGROUP_MAX_FRAMES;
  const uint frames_in_chunk = frame_start < frame_count
      ? min(WF2_ENDPOINT_DELTA_FRAMEGROUP_MAX_FRAMES, frame_count - frame_start)
      : 0u;

  threadgroup int tg_valid[WF2_ENDPOINT_DELTA_FRAMEGROUP_MAX_FRAMES];
  threadgroup int tg_source[WF2_ENDPOINT_DELTA_FRAMEGROUP_MAX_FRAMES];
  threadgroup int tg_begin[WF2_ENDPOINT_DELTA_FRAMEGROUP_MAX_FRAMES];
  threadgroup int tg_end[WF2_ENDPOINT_DELTA_FRAMEGROUP_MAX_FRAMES];
  threadgroup float tg_loss[WF2_ENDPOINT_DELTA_FRAMEGROUP_MAX_FRAMES];
  threadgroup float4 tg_site_grad[
      WF2_ENDPOINT_DELTA_FRAMEGROUP_MAX_FRAMES * WF2_ENDPOINT_EDIT_FRAMEGROUP_REDUCE_SITES];

  if (local_frame == 0u) {
    for (uint frame_id = 0u; frame_id < WF2_ENDPOINT_DELTA_FRAMEGROUP_MAX_FRAMES; ++frame_id) {
      tg_valid[frame_id] = 0;
      tg_source[frame_id] = 0;
      tg_begin[frame_id] = 0;
      tg_end[frame_id] = 0;
    }

    bool ok = track_id < track_count && frames_in_chunk > 0u;
    int change_begin_raw = 0;
    int change_end_raw = 0;
    int chunk_change_begin_raw = 0;
    int chunk_change_end_raw = 0;
    if (ok) {
      change_begin_raw = track_change_offsets_i32[track_id];
      change_end_raw = track_change_offsets_i32[track_id + 1u];
      const uint chunk_offset_base = track_id * (chunk_count + 1u) + chunk_id;
      chunk_change_begin_raw = int(track_chunk_change_offsets_i16[chunk_offset_base]);
      chunk_change_end_raw = int(track_chunk_change_offsets_i16[chunk_offset_base + 1u]);
      ok = change_begin_raw >= 0 && change_end_raw >= change_begin_raw && uint(change_end_raw) <= change_count &&
          chunk_change_begin_raw >= change_begin_raw && chunk_change_end_raw >= chunk_change_begin_raw &&
          chunk_change_end_raw <= change_end_raw;
    }
    int selected_change = -1;
    if (ok && chunk_change_begin_raw > change_begin_raw) {
      selected_change = chunk_change_begin_raw - 1;
      while (selected_change >= change_begin_raw && change_frame_i32[uint(selected_change)] < 0) {
        selected_change -= 1;
      }
      if (selected_change < change_begin_raw) {
        selected_change = -1;
      }
    }

    uint change_cursor = ok ? uint(chunk_change_begin_raw) : 0u;
    const uint change_end = ok ? uint(chunk_change_end_raw) : 0u;

    for (uint local_frame_id = 0u; local_frame_id < frames_in_chunk; ++local_frame_id) {
      const uint global_frame_id = frame_start + local_frame_id;
      while (ok && change_cursor < change_end) {
        const int changed_frame = change_frame_i32[change_cursor];
        if (changed_frame < 0) {
          change_cursor += 1u;
          continue;
        }
        if (uint(changed_frame) > global_frame_id) {
          break;
        }

        selected_change = int(change_cursor);
        change_cursor += 1u;
      }

      if (ok) {
        const bool use_change = selected_change >= 0;
        const int begin_raw =
            use_change ? change_offsets_i32[uint(selected_change)] : base_offsets_i32[track_id];
        const int end_raw =
            use_change ? change_offsets_i32[uint(selected_change) + 1u] : base_offsets_i32[track_id + 1u];
        const uint record_count = use_change ? change_record_count : base_record_count;
        ok = begin_raw >= 0 && end_raw >= begin_raw && uint(end_raw) <= record_count &&
            uint(end_raw - begin_raw) <= WF2_MAX_REALRAY_SEGMENTS;
        if (!ok) {
          break;
        }
        tg_source[local_frame_id] = use_change ? 1 : 0;
        tg_begin[local_frame_id] = begin_raw;
        tg_end[local_frame_id] = end_raw;
        tg_valid[local_frame_id] = 1;
      }
    }
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  const bool reduce_small_sites = site_count <= WF2_ENDPOINT_EDIT_FRAMEGROUP_REDUCE_SITES;
  if (reduce_small_sites) {
    const uint site_grad_base = local_frame * WF2_ENDPOINT_EDIT_FRAMEGROUP_REDUCE_SITES;
    for (uint site_slot = 0u; site_slot < WF2_ENDPOINT_EDIT_FRAMEGROUP_REDUCE_SITES; ++site_slot) {
      tg_site_grad[site_grad_base + site_slot] = float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
  }
  tg_loss[local_frame] = 0.0f;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const bool active = track_id < track_count && local_frame < frames_in_chunk && tg_valid[local_frame] != 0;
  if (active) {

  const uint global_frame_id = frame_start + local_frame;
  const uint sample_id = track_id * frame_count + global_frame_id;
  const uint total_samples = track_count * frame_count;
  const float near_depth = config_f32[0];
  const float far_depth = config_f32[1];
  const float invalid_epsilon = config_f32[2];
  const float transmittance_threshold = config_f32[3];
  const float t = frame_t_f32[global_frame_id];
  const bool use_change = tg_source[local_frame] != 0;
  const int begin_raw = tg_begin[local_frame];
  const int end_raw = tg_end[local_frame];
  const bool valid_row_bounds = end_raw >= begin_raw;
  const uint row_count = valid_row_bounds ? uint(end_raw - begin_raw) : 0u;
  const uint out_base = sample_id * 3u;
  const float3 target_rgb = float3(
      target_rgb_f32[out_base + 0u],
      target_rgb_f32[out_base + 1u],
      target_rgb_f32[out_base + 2u]);
  const float inv_element_count = 1.0f / float(total_samples * 3u);
  if (valid_row_bounds && row_count == 0u) {
    tg_loss[local_frame] = dot(target_rgb, target_rgb) * inv_element_count;
  } else if (valid_row_bounds) {

  uint owners[WF2_MAX_REALRAY_SEGMENTS];
  float lengths[WF2_MAX_REALRAY_SEGMENTS];
  float trans_before[WF2_MAX_REALRAY_SEGMENTS];
  float segment_trans[WF2_MAX_REALRAY_SEGMENTS];
  float segment_alpha[WF2_MAX_REALRAY_SEGMENTS];
  float weights[WF2_MAX_REALRAY_SEGMENTS];
  float3 segment_rgb[WF2_MAX_REALRAY_SEGMENTS];

  float3 rgb_accum = float3(0.0f, 0.0f, 0.0f);
  float transmittance = 1.0f;
  uint local_run_count = 0u;
  int cached_right_cut = -2147483648;
  float cached_right_depth = 0.0f;
  bool cached_right_valid = false;
  for (uint cursor = 0u; cursor < row_count; ++cursor) {
    if (transmittance <= transmittance_threshold || local_run_count >= WF2_MAX_REALRAY_SEGMENTS) {
      break;
    }
    const uint record_base = uint(begin_raw) + cursor;
    const short4 record = use_change ? change_record_i16x4[record_base] : base_record_i16x4[record_base];
    const int owner_raw = int(record.x);
    if (owner_raw < 0 || uint(owner_raw) >= site_count) {
      continue;
    }
    float start_depth = 0.0f;
    float end_depth = 0.0f;
    const int left_cut = int(record.y);
    const int right_cut = int(record.z);
    bool start_valid = false;
    if (cached_right_valid && left_cut == cached_right_cut) {
      start_depth = cached_right_depth;
      start_valid = true;
    } else {
      start_valid = wf2_endpoint_record_coeff16_cut_depth(
          coeff_f16,
          boundary_count,
          track_id,
          left_cut,
          t,
          near_depth,
          far_depth,
          invalid_epsilon,
          start_depth);
    }
    if (!start_valid ||
        !wf2_endpoint_record_coeff16_cut_depth(
            coeff_f16,
            boundary_count,
            track_id,
            right_cut,
            t,
            near_depth,
            far_depth,
            invalid_epsilon,
            end_depth)) {
      cached_right_valid = false;
      continue;
    }
    cached_right_cut = right_cut;
    cached_right_depth = end_depth;
    cached_right_valid = true;
    const float length = end_depth - start_depth;
    if (!(length > 1.0e-8f)) {
      continue;
    }
    const uint owner = uint(owner_raw);
    const uint rgba_base = owner * 4u;
    const float density = max(site_rgba_f32[rgba_base + 3u], 0.0f);
    const float seg_trans = exp(-density * length);
    const float seg_alpha = 1.0f - seg_trans;
    const float weight = transmittance * seg_alpha;
    const float3 rgb = float3(
        site_rgba_f32[rgba_base + 0u],
        site_rgba_f32[rgba_base + 1u],
        site_rgba_f32[rgba_base + 2u]);

    owners[local_run_count] = owner;
    lengths[local_run_count] = length;
    trans_before[local_run_count] = transmittance;
    segment_trans[local_run_count] = seg_trans;
    segment_alpha[local_run_count] = seg_alpha;
    weights[local_run_count] = weight;
    segment_rgb[local_run_count] = rgb;
    local_run_count += 1u;

    rgb_accum += weight * rgb;
    transmittance *= seg_trans;
  }

  const float3 diff = rgb_accum - target_rgb;
  const float sample_loss = dot(diff, diff) * inv_element_count;
  tg_loss[local_frame] = sample_loss;
  const float3 grad_rgb = (2.0f * inv_element_count) * diff;

  float adj_next_transmittance = 0.0f;
  for (int run_id = int(local_run_count) - 1; run_id >= 0; --run_id) {
    const uint owner = owners[run_id];
    const uint rgba_base = owner * 4u;
    const float raw_density = site_rgba_f32[rgba_base + 3u];
    const float adj_weight = dot(grad_rgb, segment_rgb[run_id]);
    const float adj_trans_before =
        adj_weight * segment_alpha[run_id] +
        adj_next_transmittance * segment_trans[run_id];
    const float adj_segment_alpha = adj_weight * trans_before[run_id];
    const float adj_segment_trans = adj_next_transmittance * trans_before[run_id] - adj_segment_alpha;
    float4 grad_rgba = float4(
        weights[run_id] * grad_rgb.x,
        weights[run_id] * grad_rgb.y,
        weights[run_id] * grad_rgb.z,
        0.0f);
    if (raw_density > 0.0f) {
      grad_rgba.w = adj_segment_trans * (-lengths[run_id] * segment_trans[run_id]);
    }
    if (reduce_small_sites) {
      tg_site_grad[local_frame * WF2_ENDPOINT_EDIT_FRAMEGROUP_REDUCE_SITES + owner] += grad_rgba;
    } else {
      wf2_atomic_add4(grad_site_rgba_f32, rgba_base, grad_rgba);
    }
    adj_next_transmittance = adj_trans_before;
  }
  }
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (local_frame == 0u) {
    float loss_sum = 0.0f;
    for (uint frame_id = 0u; frame_id < frames_in_chunk; ++frame_id) {
      loss_sum += tg_loss[frame_id];
    }
    atomic_fetch_add_explicit(&loss_f32[0], loss_sum, memory_order_relaxed);
  }

  if (reduce_small_sites) {
    if (local_frame < site_count) {
      float4 grad_sum = float4(0.0f, 0.0f, 0.0f, 0.0f);
      for (uint frame_id = 0u; frame_id < frames_in_chunk; ++frame_id) {
        grad_sum += tg_site_grad[
            frame_id * WF2_ENDPOINT_EDIT_FRAMEGROUP_REDUCE_SITES + local_frame];
      }
      wf2_atomic_add4(grad_site_rgba_f32, local_frame * 4u, grad_sum);
    }
  }
}

kernel void wf2_endpoint_record_delta_replace_coeff16_packed_framegroup16_mse_vjp_direct_atomic_rgb_only_tensor(
    device const half* coeff_f16 [[buffer(0)]],
    device const float* frame_t_f32 [[buffer(1)]],
    device const int* base_offsets_i32 [[buffer(2)]],
    device const int* base_record_i32 [[buffer(3)]],
    device const int* track_change_offsets_i32 [[buffer(4)]],
    device const short* track_chunk_change_offsets_i16 [[buffer(5)]],
    device const int* change_frame_i32 [[buffer(6)]],
    device const int* change_offsets_i32 [[buffer(7)]],
    device const int* change_record_i32 [[buffer(8)]],
    device const float* site_rgba_f32 [[buffer(9)]],
    device const float* target_rgb_f32 [[buffer(10)]],
    device const int* config_i32 [[buffer(11)]],
    device const float* config_f32 [[buffer(12)]],
    device atomic_float* loss_f32 [[buffer(13)]],
    device atomic_float* grad_site_rgba_f32 [[buffer(14)]],
    uint3 local_pos [[thread_position_in_threadgroup]],
    uint3 group_id [[threadgroup_position_in_grid]]) {
  const uint local_frame = local_pos.x;
  const uint boundary_count = uint(config_i32[0]);
  const uint track_count = uint(config_i32[1]);
  const uint frame_count = uint(config_i32[2]);
  const uint site_count = uint(config_i32[3]);
  const uint base_record_count = uint(config_i32[4]);
  const uint change_count = uint(config_i32[5]);
  const uint change_record_count = uint(config_i32[6]);
  const uint chunk_count = (frame_count + WF2_ENDPOINT_DELTA_FRAMEGROUP_MAX_FRAMES - 1u) /
      WF2_ENDPOINT_DELTA_FRAMEGROUP_MAX_FRAMES;
  const uint group_index = group_id.x;
  const uint track_id = group_index / max(chunk_count, 1u);
  const uint chunk_id = group_index - track_id * max(chunk_count, 1u);
  const uint frame_start = chunk_id * WF2_ENDPOINT_DELTA_FRAMEGROUP_MAX_FRAMES;
  const uint frames_in_chunk = frame_start < frame_count
      ? min(WF2_ENDPOINT_DELTA_FRAMEGROUP_MAX_FRAMES, frame_count - frame_start)
      : 0u;

  threadgroup int tg_valid[WF2_ENDPOINT_DELTA_FRAMEGROUP_MAX_FRAMES];
  threadgroup int tg_source[WF2_ENDPOINT_DELTA_FRAMEGROUP_MAX_FRAMES];
  threadgroup int tg_begin[WF2_ENDPOINT_DELTA_FRAMEGROUP_MAX_FRAMES];
  threadgroup int tg_end[WF2_ENDPOINT_DELTA_FRAMEGROUP_MAX_FRAMES];
  threadgroup float tg_loss[WF2_ENDPOINT_DELTA_FRAMEGROUP_MAX_FRAMES];
  threadgroup float4 tg_site_grad[
      WF2_ENDPOINT_DELTA_FRAMEGROUP_MAX_FRAMES * WF2_ENDPOINT_EDIT_FRAMEGROUP_REDUCE_SITES];

  if (local_frame == 0u) {
    for (uint frame_id = 0u; frame_id < WF2_ENDPOINT_DELTA_FRAMEGROUP_MAX_FRAMES; ++frame_id) {
      tg_valid[frame_id] = 0;
      tg_source[frame_id] = 0;
      tg_begin[frame_id] = 0;
      tg_end[frame_id] = 0;
    }

    bool ok = track_id < track_count && frames_in_chunk > 0u;
    int change_begin_raw = 0;
    int change_end_raw = 0;
    int chunk_change_begin_raw = 0;
    int chunk_change_end_raw = 0;
    if (ok) {
      change_begin_raw = track_change_offsets_i32[track_id];
      change_end_raw = track_change_offsets_i32[track_id + 1u];
      const uint chunk_offset_base = track_id * (chunk_count + 1u) + chunk_id;
      chunk_change_begin_raw = int(track_chunk_change_offsets_i16[chunk_offset_base]);
      chunk_change_end_raw = int(track_chunk_change_offsets_i16[chunk_offset_base + 1u]);
      ok = change_begin_raw >= 0 && change_end_raw >= change_begin_raw && uint(change_end_raw) <= change_count &&
          chunk_change_begin_raw >= change_begin_raw && chunk_change_end_raw >= chunk_change_begin_raw &&
          chunk_change_end_raw <= change_end_raw;
    }
    int selected_change = -1;
    if (ok && chunk_change_begin_raw > change_begin_raw) {
      selected_change = chunk_change_begin_raw - 1;
      while (selected_change >= change_begin_raw && change_frame_i32[uint(selected_change)] < 0) {
        selected_change -= 1;
      }
      if (selected_change < change_begin_raw) {
        selected_change = -1;
      }
    }

    uint change_cursor = ok ? uint(chunk_change_begin_raw) : 0u;
    const uint change_end = ok ? uint(chunk_change_end_raw) : 0u;

    for (uint local_frame_id = 0u; local_frame_id < frames_in_chunk; ++local_frame_id) {
      const uint global_frame_id = frame_start + local_frame_id;
      while (ok && change_cursor < change_end) {
        const int changed_frame = change_frame_i32[change_cursor];
        if (changed_frame < 0) {
          change_cursor += 1u;
          continue;
        }
        if (uint(changed_frame) > global_frame_id) {
          break;
        }

        selected_change = int(change_cursor);
        change_cursor += 1u;
      }

      if (ok) {
        const bool use_change = selected_change >= 0;
        const int begin_raw =
            use_change ? change_offsets_i32[uint(selected_change)] : base_offsets_i32[track_id];
        const int end_raw =
            use_change ? change_offsets_i32[uint(selected_change) + 1u] : base_offsets_i32[track_id + 1u];
        const uint record_count = use_change ? change_record_count : base_record_count;
        ok = begin_raw >= 0 && end_raw >= begin_raw && uint(end_raw) <= record_count &&
            uint(end_raw - begin_raw) <= WF2_MAX_REALRAY_SEGMENTS;
        if (!ok) {
          break;
        }
        tg_source[local_frame_id] = use_change ? 1 : 0;
        tg_begin[local_frame_id] = begin_raw;
        tg_end[local_frame_id] = end_raw;
        tg_valid[local_frame_id] = 1;
      }
    }
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  const bool reduce_small_sites = site_count <= WF2_ENDPOINT_EDIT_FRAMEGROUP_REDUCE_SITES;
  if (reduce_small_sites) {
    const uint site_grad_base = local_frame * WF2_ENDPOINT_EDIT_FRAMEGROUP_REDUCE_SITES;
    for (uint site_slot = 0u; site_slot < WF2_ENDPOINT_EDIT_FRAMEGROUP_REDUCE_SITES; ++site_slot) {
      tg_site_grad[site_grad_base + site_slot] = float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
  }
  tg_loss[local_frame] = 0.0f;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const bool active = track_id < track_count && local_frame < frames_in_chunk && tg_valid[local_frame] != 0;
  if (active) {

  const uint global_frame_id = frame_start + local_frame;
  const uint sample_id = track_id * frame_count + global_frame_id;
  const uint total_samples = track_count * frame_count;
  const float near_depth = config_f32[0];
  const float far_depth = config_f32[1];
  const float invalid_epsilon = config_f32[2];
  const float transmittance_threshold = config_f32[3];
  const float t = frame_t_f32[global_frame_id];
  const bool use_change = tg_source[local_frame] != 0;
  const int begin_raw = tg_begin[local_frame];
  const int end_raw = tg_end[local_frame];
  const bool valid_row_bounds = end_raw >= begin_raw;
  const uint row_count = valid_row_bounds ? uint(end_raw - begin_raw) : 0u;
  const uint out_base = sample_id * 3u;
  const float3 target_rgb = float3(
      target_rgb_f32[out_base + 0u],
      target_rgb_f32[out_base + 1u],
      target_rgb_f32[out_base + 2u]);
  const float inv_element_count = 1.0f / float(total_samples * 3u);
  if (valid_row_bounds && row_count == 0u) {
    tg_loss[local_frame] = dot(target_rgb, target_rgb) * inv_element_count;
  } else if (valid_row_bounds) {

  uint owners[WF2_MAX_REALRAY_SEGMENTS];
  float lengths[WF2_MAX_REALRAY_SEGMENTS];
  float trans_before[WF2_MAX_REALRAY_SEGMENTS];
  float segment_trans[WF2_MAX_REALRAY_SEGMENTS];
  float segment_alpha[WF2_MAX_REALRAY_SEGMENTS];
  float weights[WF2_MAX_REALRAY_SEGMENTS];
  float3 segment_rgb[WF2_MAX_REALRAY_SEGMENTS];

  float3 rgb_accum = float3(0.0f, 0.0f, 0.0f);
  float transmittance = 1.0f;
  uint local_run_count = 0u;
  int cached_right_cut = -2147483648;
  float cached_right_depth = 0.0f;
  bool cached_right_valid = false;
  for (uint cursor = 0u; cursor < row_count; ++cursor) {
    if (transmittance <= transmittance_threshold || local_run_count >= WF2_MAX_REALRAY_SEGMENTS) {
      break;
    }
    const uint record_base = uint(begin_raw) + cursor;
    int owner_raw = 0;
    int left_cut = 0;
    int right_cut = 0;
    wf2_endpoint_record_unpack_record(
        use_change ? change_record_i32[record_base] : base_record_i32[record_base],
        owner_raw,
        left_cut,
        right_cut);
    if (owner_raw < 0 || uint(owner_raw) >= site_count) {
      continue;
    }
    float start_depth = 0.0f;
    float end_depth = 0.0f;
    bool start_valid = false;
    if (cached_right_valid && left_cut == cached_right_cut) {
      start_depth = cached_right_depth;
      start_valid = true;
    } else {
      start_valid = wf2_endpoint_record_coeff16_cut_depth(
          coeff_f16,
          boundary_count,
          track_id,
          left_cut,
          t,
          near_depth,
          far_depth,
          invalid_epsilon,
          start_depth);
    }
    if (!start_valid ||
        !wf2_endpoint_record_coeff16_cut_depth(
            coeff_f16,
            boundary_count,
            track_id,
            right_cut,
            t,
            near_depth,
            far_depth,
            invalid_epsilon,
            end_depth)) {
      cached_right_valid = false;
      continue;
    }
    cached_right_cut = right_cut;
    cached_right_depth = end_depth;
    cached_right_valid = true;
    const float length = end_depth - start_depth;
    if (!(length > 1.0e-8f)) {
      continue;
    }
    const uint owner = uint(owner_raw);
    const uint rgba_base = owner * 4u;
    const float density = max(site_rgba_f32[rgba_base + 3u], 0.0f);
    const float seg_trans = exp(-density * length);
    const float seg_alpha = 1.0f - seg_trans;
    const float weight = transmittance * seg_alpha;
    const float3 rgb = float3(
        site_rgba_f32[rgba_base + 0u],
        site_rgba_f32[rgba_base + 1u],
        site_rgba_f32[rgba_base + 2u]);

    owners[local_run_count] = owner;
    lengths[local_run_count] = length;
    trans_before[local_run_count] = transmittance;
    segment_trans[local_run_count] = seg_trans;
    segment_alpha[local_run_count] = seg_alpha;
    weights[local_run_count] = weight;
    segment_rgb[local_run_count] = rgb;
    local_run_count += 1u;

    rgb_accum += weight * rgb;
    transmittance *= seg_trans;
  }

  const float3 diff = rgb_accum - target_rgb;
  const float sample_loss = dot(diff, diff) * inv_element_count;
  tg_loss[local_frame] = sample_loss;
  const float3 grad_rgb = (2.0f * inv_element_count) * diff;

  float adj_next_transmittance = 0.0f;
  for (int run_id = int(local_run_count) - 1; run_id >= 0; --run_id) {
    const uint owner = owners[run_id];
    const uint rgba_base = owner * 4u;
    const float raw_density = site_rgba_f32[rgba_base + 3u];
    const float adj_weight = dot(grad_rgb, segment_rgb[run_id]);
    const float adj_trans_before =
        adj_weight * segment_alpha[run_id] +
        adj_next_transmittance * segment_trans[run_id];
    const float adj_segment_alpha = adj_weight * trans_before[run_id];
    const float adj_segment_trans = adj_next_transmittance * trans_before[run_id] - adj_segment_alpha;
    float4 grad_rgba = float4(
        weights[run_id] * grad_rgb.x,
        weights[run_id] * grad_rgb.y,
        weights[run_id] * grad_rgb.z,
        0.0f);
    if (raw_density > 0.0f) {
      grad_rgba.w = adj_segment_trans * (-lengths[run_id] * segment_trans[run_id]);
    }
    if (reduce_small_sites) {
      tg_site_grad[local_frame * WF2_ENDPOINT_EDIT_FRAMEGROUP_REDUCE_SITES + owner] += grad_rgba;
    } else {
      wf2_atomic_add4(grad_site_rgba_f32, rgba_base, grad_rgba);
    }
    adj_next_transmittance = adj_trans_before;
  }
  }
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (local_frame == 0u) {
    float loss_sum = 0.0f;
    for (uint frame_id = 0u; frame_id < frames_in_chunk; ++frame_id) {
      loss_sum += tg_loss[frame_id];
    }
    atomic_fetch_add_explicit(&loss_f32[0], loss_sum, memory_order_relaxed);
  }

  if (reduce_small_sites) {
    if (local_frame < site_count) {
      float4 grad_sum = float4(0.0f, 0.0f, 0.0f, 0.0f);
      for (uint frame_id = 0u; frame_id < frames_in_chunk; ++frame_id) {
        grad_sum += tg_site_grad[
            frame_id * WF2_ENDPOINT_EDIT_FRAMEGROUP_REDUCE_SITES + local_frame];
      }
      wf2_atomic_add4(grad_site_rgba_f32, local_frame * 4u, grad_sum);
    }
  }
}

kernel void wf2_endpoint_record_delta_replace_coeff16_packed_framegroup16_reduce32_mse_vjp_direct_atomic_rgb_only_tensor(
    device const half* coeff_f16 [[buffer(0)]],
    device const float* frame_t_f32 [[buffer(1)]],
    device const int* base_offsets_i32 [[buffer(2)]],
    device const int* base_record_i32 [[buffer(3)]],
    device const int* track_change_offsets_i32 [[buffer(4)]],
    device const short* track_chunk_change_offsets_i16 [[buffer(5)]],
    device const int* change_frame_i32 [[buffer(6)]],
    device const int* change_offsets_i32 [[buffer(7)]],
    device const int* change_record_i32 [[buffer(8)]],
    device const float* site_rgba_f32 [[buffer(9)]],
    device const float* target_rgb_f32 [[buffer(10)]],
    device const int* config_i32 [[buffer(11)]],
    device const float* config_f32 [[buffer(12)]],
    device atomic_float* loss_f32 [[buffer(13)]],
    device atomic_float* grad_site_rgba_f32 [[buffer(14)]],
    uint3 local_pos [[thread_position_in_threadgroup]],
    uint3 group_id [[threadgroup_position_in_grid]]) {
  const uint local_frame = local_pos.x;
  const uint boundary_count = uint(config_i32[0]);
  const uint track_count = uint(config_i32[1]);
  const uint frame_count = uint(config_i32[2]);
  const uint site_count = uint(config_i32[3]);
  const uint base_record_count = uint(config_i32[4]);
  const uint change_count = uint(config_i32[5]);
  const uint change_record_count = uint(config_i32[6]);
  const uint chunk_count = (frame_count + WF2_ENDPOINT_DELTA_FRAMEGROUP_MAX_FRAMES - 1u) /
      WF2_ENDPOINT_DELTA_FRAMEGROUP_MAX_FRAMES;
  const uint group_index = group_id.x;
  const uint track_id = group_index / max(chunk_count, 1u);
  const uint chunk_id = group_index - track_id * max(chunk_count, 1u);
  const uint frame_start = chunk_id * WF2_ENDPOINT_DELTA_FRAMEGROUP_MAX_FRAMES;
  const uint frames_in_chunk = frame_start < frame_count
      ? min(WF2_ENDPOINT_DELTA_FRAMEGROUP_MAX_FRAMES, frame_count - frame_start)
      : 0u;

  threadgroup int tg_valid[WF2_ENDPOINT_DELTA_FRAMEGROUP_MAX_FRAMES];
  threadgroup int tg_source[WF2_ENDPOINT_DELTA_FRAMEGROUP_MAX_FRAMES];
  threadgroup int tg_begin[WF2_ENDPOINT_DELTA_FRAMEGROUP_MAX_FRAMES];
  threadgroup int tg_end[WF2_ENDPOINT_DELTA_FRAMEGROUP_MAX_FRAMES];
  threadgroup float tg_loss[WF2_ENDPOINT_DELTA_FRAMEGROUP_MAX_FRAMES];
  threadgroup float4 tg_site_grad[
      WF2_ENDPOINT_DELTA_FRAMEGROUP_MAX_FRAMES * WF2_ENDPOINT_COMPACT_REDUCE32_SITES];

  if (local_frame == 0u) {
    for (uint frame_id = 0u; frame_id < WF2_ENDPOINT_DELTA_FRAMEGROUP_MAX_FRAMES; ++frame_id) {
      tg_valid[frame_id] = 0;
      tg_source[frame_id] = 0;
      tg_begin[frame_id] = 0;
      tg_end[frame_id] = 0;
    }

    bool ok = track_id < track_count && frames_in_chunk > 0u;
    int change_begin_raw = 0;
    int change_end_raw = 0;
    int chunk_change_begin_raw = 0;
    int chunk_change_end_raw = 0;
    if (ok) {
      change_begin_raw = track_change_offsets_i32[track_id];
      change_end_raw = track_change_offsets_i32[track_id + 1u];
      const uint chunk_offset_base = track_id * (chunk_count + 1u) + chunk_id;
      chunk_change_begin_raw = int(track_chunk_change_offsets_i16[chunk_offset_base]);
      chunk_change_end_raw = int(track_chunk_change_offsets_i16[chunk_offset_base + 1u]);
      ok = change_begin_raw >= 0 && change_end_raw >= change_begin_raw && uint(change_end_raw) <= change_count &&
          chunk_change_begin_raw >= change_begin_raw && chunk_change_end_raw >= chunk_change_begin_raw &&
          chunk_change_end_raw <= change_end_raw;
    }
    int selected_change = -1;
    if (ok && chunk_change_begin_raw > change_begin_raw) {
      selected_change = chunk_change_begin_raw - 1;
      while (selected_change >= change_begin_raw && change_frame_i32[uint(selected_change)] < 0) {
        selected_change -= 1;
      }
      if (selected_change < change_begin_raw) {
        selected_change = -1;
      }
    }

    uint change_cursor = ok ? uint(chunk_change_begin_raw) : 0u;
    const uint change_end = ok ? uint(chunk_change_end_raw) : 0u;

    for (uint local_frame_id = 0u; local_frame_id < frames_in_chunk; ++local_frame_id) {
      const uint global_frame_id = frame_start + local_frame_id;
      while (ok && change_cursor < change_end) {
        const int changed_frame = change_frame_i32[change_cursor];
        if (changed_frame < 0) {
          change_cursor += 1u;
          continue;
        }
        if (uint(changed_frame) > global_frame_id) {
          break;
        }

        selected_change = int(change_cursor);
        change_cursor += 1u;
      }

      if (ok) {
        const bool use_change = selected_change >= 0;
        const int begin_raw =
            use_change ? change_offsets_i32[uint(selected_change)] : base_offsets_i32[track_id];
        const int end_raw =
            use_change ? change_offsets_i32[uint(selected_change) + 1u] : base_offsets_i32[track_id + 1u];
        const uint record_count = use_change ? change_record_count : base_record_count;
        ok = begin_raw >= 0 && end_raw >= begin_raw && uint(end_raw) <= record_count &&
            uint(end_raw - begin_raw) <= WF2_MAX_REALRAY_SEGMENTS;
        if (!ok) {
          break;
        }
        tg_source[local_frame_id] = use_change ? 1 : 0;
        tg_begin[local_frame_id] = begin_raw;
        tg_end[local_frame_id] = end_raw;
        tg_valid[local_frame_id] = 1;
      }
    }
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  const bool reduce_small_sites = site_count <= WF2_ENDPOINT_COMPACT_REDUCE32_SITES;
  if (reduce_small_sites) {
    const uint site_grad_base = local_frame * WF2_ENDPOINT_COMPACT_REDUCE32_SITES;
    for (uint site_slot = 0u; site_slot < WF2_ENDPOINT_COMPACT_REDUCE32_SITES; ++site_slot) {
      tg_site_grad[site_grad_base + site_slot] = float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
  }
  tg_loss[local_frame] = 0.0f;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const bool active = track_id < track_count && local_frame < frames_in_chunk && tg_valid[local_frame] != 0;
  if (active) {

  const uint global_frame_id = frame_start + local_frame;
  const uint sample_id = track_id * frame_count + global_frame_id;
  const uint total_samples = track_count * frame_count;
  const float near_depth = config_f32[0];
  const float far_depth = config_f32[1];
  const float invalid_epsilon = config_f32[2];
  const float transmittance_threshold = config_f32[3];
  const float t = frame_t_f32[global_frame_id];
  const bool use_change = tg_source[local_frame] != 0;
  const int begin_raw = tg_begin[local_frame];
  const int end_raw = tg_end[local_frame];
  const bool valid_row_bounds = end_raw >= begin_raw;
  const uint row_count = valid_row_bounds ? uint(end_raw - begin_raw) : 0u;
  const uint out_base = sample_id * 3u;
  const float3 target_rgb = float3(
      target_rgb_f32[out_base + 0u],
      target_rgb_f32[out_base + 1u],
      target_rgb_f32[out_base + 2u]);
  const float inv_element_count = 1.0f / float(total_samples * 3u);
  if (valid_row_bounds && row_count == 0u) {
    tg_loss[local_frame] = dot(target_rgb, target_rgb) * inv_element_count;
  } else if (valid_row_bounds) {

  uint owners[WF2_MAX_REALRAY_SEGMENTS];
  float lengths[WF2_MAX_REALRAY_SEGMENTS];
  float trans_before[WF2_MAX_REALRAY_SEGMENTS];
  float segment_trans[WF2_MAX_REALRAY_SEGMENTS];
  float segment_alpha[WF2_MAX_REALRAY_SEGMENTS];
  float weights[WF2_MAX_REALRAY_SEGMENTS];
  float3 segment_rgb[WF2_MAX_REALRAY_SEGMENTS];

  float3 rgb_accum = float3(0.0f, 0.0f, 0.0f);
  float transmittance = 1.0f;
  uint local_run_count = 0u;
  int cached_right_cut = -2147483648;
  float cached_right_depth = 0.0f;
  bool cached_right_valid = false;
  for (uint cursor = 0u; cursor < row_count; ++cursor) {
    if (transmittance <= transmittance_threshold || local_run_count >= WF2_MAX_REALRAY_SEGMENTS) {
      break;
    }
    const uint record_base = uint(begin_raw) + cursor;
    int owner_raw = 0;
    int left_cut = 0;
    int right_cut = 0;
    wf2_endpoint_record_unpack_record(
        use_change ? change_record_i32[record_base] : base_record_i32[record_base],
        owner_raw,
        left_cut,
        right_cut);
    if (owner_raw < 0 || uint(owner_raw) >= site_count) {
      continue;
    }
    float start_depth = 0.0f;
    float end_depth = 0.0f;
    bool start_valid = false;
    if (cached_right_valid && left_cut == cached_right_cut) {
      start_depth = cached_right_depth;
      start_valid = true;
    } else {
      start_valid = wf2_endpoint_record_coeff16_cut_depth(
          coeff_f16,
          boundary_count,
          track_id,
          left_cut,
          t,
          near_depth,
          far_depth,
          invalid_epsilon,
          start_depth);
    }
    if (!start_valid ||
        !wf2_endpoint_record_coeff16_cut_depth(
            coeff_f16,
            boundary_count,
            track_id,
            right_cut,
            t,
            near_depth,
            far_depth,
            invalid_epsilon,
            end_depth)) {
      cached_right_valid = false;
      continue;
    }
    cached_right_cut = right_cut;
    cached_right_depth = end_depth;
    cached_right_valid = true;
    const float length = end_depth - start_depth;
    if (!(length > 1.0e-8f)) {
      continue;
    }
    const uint owner = uint(owner_raw);
    const uint rgba_base = owner * 4u;
    const float density = max(site_rgba_f32[rgba_base + 3u], 0.0f);
    const float seg_trans = exp(-density * length);
    const float seg_alpha = 1.0f - seg_trans;
    const float weight = transmittance * seg_alpha;
    const float3 rgb = float3(
        site_rgba_f32[rgba_base + 0u],
        site_rgba_f32[rgba_base + 1u],
        site_rgba_f32[rgba_base + 2u]);

    owners[local_run_count] = owner;
    lengths[local_run_count] = length;
    trans_before[local_run_count] = transmittance;
    segment_trans[local_run_count] = seg_trans;
    segment_alpha[local_run_count] = seg_alpha;
    weights[local_run_count] = weight;
    segment_rgb[local_run_count] = rgb;
    local_run_count += 1u;

    rgb_accum += weight * rgb;
    transmittance *= seg_trans;
  }

  const float3 diff = rgb_accum - target_rgb;
  const float sample_loss = dot(diff, diff) * inv_element_count;
  tg_loss[local_frame] = sample_loss;
  const float3 grad_rgb = (2.0f * inv_element_count) * diff;

  float adj_next_transmittance = 0.0f;
  for (int run_id = int(local_run_count) - 1; run_id >= 0; --run_id) {
    const uint owner = owners[run_id];
    const uint rgba_base = owner * 4u;
    const float raw_density = site_rgba_f32[rgba_base + 3u];
    const float adj_weight = dot(grad_rgb, segment_rgb[run_id]);
    const float adj_trans_before =
        adj_weight * segment_alpha[run_id] +
        adj_next_transmittance * segment_trans[run_id];
    const float adj_segment_alpha = adj_weight * trans_before[run_id];
    const float adj_segment_trans = adj_next_transmittance * trans_before[run_id] - adj_segment_alpha;
    float4 grad_rgba = float4(
        weights[run_id] * grad_rgb.x,
        weights[run_id] * grad_rgb.y,
        weights[run_id] * grad_rgb.z,
        0.0f);
    if (raw_density > 0.0f) {
      grad_rgba.w = adj_segment_trans * (-lengths[run_id] * segment_trans[run_id]);
    }
    if (reduce_small_sites) {
      tg_site_grad[local_frame * WF2_ENDPOINT_COMPACT_REDUCE32_SITES + owner] += grad_rgba;
    } else {
      wf2_atomic_add4(grad_site_rgba_f32, rgba_base, grad_rgba);
    }
    adj_next_transmittance = adj_trans_before;
  }
  }
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (local_frame == 0u) {
    float loss_sum = 0.0f;
    for (uint frame_id = 0u; frame_id < frames_in_chunk; ++frame_id) {
      loss_sum += tg_loss[frame_id];
    }
    atomic_fetch_add_explicit(&loss_f32[0], loss_sum, memory_order_relaxed);
  }

  if (reduce_small_sites) {
    if (local_frame < site_count) {
      float4 grad_sum = float4(0.0f, 0.0f, 0.0f, 0.0f);
      for (uint frame_id = 0u; frame_id < frames_in_chunk; ++frame_id) {
        grad_sum += tg_site_grad[
            frame_id * WF2_ENDPOINT_COMPACT_REDUCE32_SITES + local_frame];
      }
      wf2_atomic_add4(grad_site_rgba_f32, local_frame * 4u, grad_sum);
    }
  }
}
kernel void wf2_endpoint_record_delta_replace_coeff16_packed_framegroup16_rowselect32_mse_vjp_direct_atomic_rgb_only_tensor(
    device const half* coeff_f16 [[buffer(0)]],
    device const float* frame_t_f32 [[buffer(1)]],
    device const int* base_offsets_i32 [[buffer(2)]],
    device const int* base_record_i32 [[buffer(3)]],
    device const int* track_change_offsets_i32 [[buffer(4)]],
    device const short* track_chunk_change_offsets_i16 [[buffer(5)]],
    device const int* change_frame_i32 [[buffer(6)]],
    device const int* change_offsets_i32 [[buffer(7)]],
    device const int* change_record_i32 [[buffer(8)]],
    device const float* site_rgba_f32 [[buffer(9)]],
    device const float* target_rgb_f32 [[buffer(10)]],
    device const int* config_i32 [[buffer(11)]],
    device const float* config_f32 [[buffer(12)]],
    device atomic_float* loss_f32 [[buffer(13)]],
    device atomic_float* grad_site_rgba_f32 [[buffer(14)]],
    uint3 local_pos [[thread_position_in_threadgroup]],
    uint3 group_id [[threadgroup_position_in_grid]]) {
  const uint local_frame = local_pos.x;
  const uint boundary_count = uint(config_i32[0]);
  const uint track_count = uint(config_i32[1]);
  const uint frame_count = uint(config_i32[2]);
  const uint site_count = uint(config_i32[3]);
  const uint base_record_count = uint(config_i32[4]);
  const uint change_count = uint(config_i32[5]);
  const uint change_record_count = uint(config_i32[6]);
  const uint chunk_count = (frame_count + WF2_ENDPOINT_DELTA_FRAMEGROUP_MAX_FRAMES - 1u) /
      WF2_ENDPOINT_DELTA_FRAMEGROUP_MAX_FRAMES;
  const uint group_index = group_id.x;
  const uint track_id = group_index / max(chunk_count, 1u);
  const uint chunk_id = group_index - track_id * max(chunk_count, 1u);
  const uint frame_start = chunk_id * WF2_ENDPOINT_DELTA_FRAMEGROUP_MAX_FRAMES;
  const uint frames_in_chunk = frame_start < frame_count
      ? min(WF2_ENDPOINT_DELTA_FRAMEGROUP_MAX_FRAMES, frame_count - frame_start)
      : 0u;

  threadgroup float tg_loss[WF2_ENDPOINT_DELTA_FRAMEGROUP_MAX_FRAMES];
  threadgroup float4 tg_site_grad[
      WF2_ENDPOINT_DELTA_FRAMEGROUP_MAX_FRAMES * WF2_ENDPOINT_COMPACT_REDUCE32_SITES];

  const bool reduce_small_sites = site_count <= WF2_ENDPOINT_COMPACT_REDUCE32_SITES;
  if (reduce_small_sites) {
    const uint site_grad_base = local_frame * WF2_ENDPOINT_COMPACT_REDUCE32_SITES;
    for (uint site_slot = 0u; site_slot < WF2_ENDPOINT_COMPACT_REDUCE32_SITES; ++site_slot) {
      tg_site_grad[site_grad_base + site_slot] = float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
  }
  tg_loss[local_frame] = 0.0f;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const bool frame_active = track_id < track_count && local_frame < frames_in_chunk;
  if (frame_active) {

  const uint global_frame_id = frame_start + local_frame;
  const uint sample_id = track_id * frame_count + global_frame_id;
  const uint total_samples = track_count * frame_count;
  const float near_depth = config_f32[0];
  const float far_depth = config_f32[1];
  const float invalid_epsilon = config_f32[2];
  const float transmittance_threshold = config_f32[3];
  const float t = frame_t_f32[global_frame_id];

  const int change_begin_raw = track_change_offsets_i32[track_id];
  const int change_end_raw = track_change_offsets_i32[track_id + 1u];
  const uint chunk_offset_base = track_id * (chunk_count + 1u) + chunk_id;
  const int chunk_change_begin_raw = int(track_chunk_change_offsets_i16[chunk_offset_base]);
  const int chunk_change_end_raw = int(track_chunk_change_offsets_i16[chunk_offset_base + 1u]);
  bool row_ok = change_begin_raw >= 0 && change_end_raw >= change_begin_raw &&
      uint(change_end_raw) <= change_count && chunk_change_begin_raw >= change_begin_raw &&
      chunk_change_end_raw >= chunk_change_begin_raw && chunk_change_end_raw <= change_end_raw;

  int selected_change = -1;
  if (row_ok && chunk_change_begin_raw > change_begin_raw) {
    selected_change = chunk_change_begin_raw - 1;
    while (selected_change >= change_begin_raw && change_frame_i32[uint(selected_change)] < 0) {
      selected_change -= 1;
    }
    if (selected_change < change_begin_raw) {
      selected_change = -1;
    }
  }

  uint change_cursor = row_ok ? uint(chunk_change_begin_raw) : 0u;
  const uint change_end = row_ok ? uint(chunk_change_end_raw) : 0u;
  while (row_ok && change_cursor < change_end) {
    const int changed_frame = change_frame_i32[change_cursor];
    if (changed_frame < 0) {
      change_cursor += 1u;
      continue;
    }
    if (uint(changed_frame) > global_frame_id) {
      break;
    }
    selected_change = int(change_cursor);
    change_cursor += 1u;
  }

  const bool use_change = selected_change >= 0;
  const int begin_raw = row_ok
      ? (use_change ? change_offsets_i32[uint(selected_change)] : base_offsets_i32[track_id])
      : 0;
  const int end_raw = row_ok
      ? (use_change ? change_offsets_i32[uint(selected_change) + 1u] : base_offsets_i32[track_id + 1u])
      : 0;
  const uint record_count = use_change ? change_record_count : base_record_count;
  const bool valid_row_bounds = row_ok && begin_raw >= 0 && end_raw >= begin_raw &&
      uint(end_raw) <= record_count && uint(end_raw - begin_raw) <= WF2_MAX_REALRAY_SEGMENTS;
  const uint row_count = valid_row_bounds ? uint(end_raw - begin_raw) : 0u;
  const uint out_base = sample_id * 3u;
  const float3 target_rgb = float3(
      target_rgb_f32[out_base + 0u],
      target_rgb_f32[out_base + 1u],
      target_rgb_f32[out_base + 2u]);
  const float inv_element_count = 1.0f / float(total_samples * 3u);
  if (valid_row_bounds && row_count == 0u) {
    tg_loss[local_frame] = dot(target_rgb, target_rgb) * inv_element_count;
  } else if (valid_row_bounds) {

  uint owners[WF2_MAX_REALRAY_SEGMENTS];
  float lengths[WF2_MAX_REALRAY_SEGMENTS];
  float trans_before[WF2_MAX_REALRAY_SEGMENTS];
  float segment_trans[WF2_MAX_REALRAY_SEGMENTS];
  float segment_alpha[WF2_MAX_REALRAY_SEGMENTS];
  float weights[WF2_MAX_REALRAY_SEGMENTS];
  float3 segment_rgb[WF2_MAX_REALRAY_SEGMENTS];

  float3 rgb_accum = float3(0.0f, 0.0f, 0.0f);
  float transmittance = 1.0f;
  uint local_run_count = 0u;
  int cached_right_cut = -2147483648;
  float cached_right_depth = 0.0f;
  bool cached_right_valid = false;
  for (uint cursor = 0u; cursor < row_count; ++cursor) {
    if (transmittance <= transmittance_threshold || local_run_count >= WF2_MAX_REALRAY_SEGMENTS) {
      break;
    }
    const uint record_base = uint(begin_raw) + cursor;
    int owner_raw = 0;
    int left_cut = 0;
    int right_cut = 0;
    wf2_endpoint_record_unpack_record(
        use_change ? change_record_i32[record_base] : base_record_i32[record_base],
        owner_raw,
        left_cut,
        right_cut);
    if (owner_raw < 0 || uint(owner_raw) >= site_count) {
      continue;
    }
    float start_depth = 0.0f;
    float end_depth = 0.0f;
    bool start_valid = false;
    if (cached_right_valid && left_cut == cached_right_cut) {
      start_depth = cached_right_depth;
      start_valid = true;
    } else {
      start_valid = wf2_endpoint_record_coeff16_cut_depth(
          coeff_f16,
          boundary_count,
          track_id,
          left_cut,
          t,
          near_depth,
          far_depth,
          invalid_epsilon,
          start_depth);
    }
    if (!start_valid ||
        !wf2_endpoint_record_coeff16_cut_depth(
            coeff_f16,
            boundary_count,
            track_id,
            right_cut,
            t,
            near_depth,
            far_depth,
            invalid_epsilon,
            end_depth)) {
      cached_right_valid = false;
      continue;
    }
    cached_right_cut = right_cut;
    cached_right_depth = end_depth;
    cached_right_valid = true;
    const float length = end_depth - start_depth;
    if (!(length > 1.0e-8f)) {
      continue;
    }
    const uint owner = uint(owner_raw);
    const uint rgba_base = owner * 4u;
    const float density = max(site_rgba_f32[rgba_base + 3u], 0.0f);
    const float seg_trans = exp(-density * length);
    const float seg_alpha = 1.0f - seg_trans;
    const float weight = transmittance * seg_alpha;
    const float3 rgb = float3(
        site_rgba_f32[rgba_base + 0u],
        site_rgba_f32[rgba_base + 1u],
        site_rgba_f32[rgba_base + 2u]);

    owners[local_run_count] = owner;
    lengths[local_run_count] = length;
    trans_before[local_run_count] = transmittance;
    segment_trans[local_run_count] = seg_trans;
    segment_alpha[local_run_count] = seg_alpha;
    weights[local_run_count] = weight;
    segment_rgb[local_run_count] = rgb;
    local_run_count += 1u;

    rgb_accum += weight * rgb;
    transmittance *= seg_trans;
  }

  const float3 diff = rgb_accum - target_rgb;
  const float sample_loss = dot(diff, diff) * inv_element_count;
  tg_loss[local_frame] = sample_loss;
  const float3 grad_rgb = (2.0f * inv_element_count) * diff;

  float adj_next_transmittance = 0.0f;
  for (int run_id = int(local_run_count) - 1; run_id >= 0; --run_id) {
    const uint owner = owners[run_id];
    const uint rgba_base = owner * 4u;
    const float raw_density = site_rgba_f32[rgba_base + 3u];
    const float adj_weight = dot(grad_rgb, segment_rgb[run_id]);
    const float adj_trans_before =
        adj_weight * segment_alpha[run_id] +
        adj_next_transmittance * segment_trans[run_id];
    const float adj_segment_alpha = adj_weight * trans_before[run_id];
    const float adj_segment_trans = adj_next_transmittance * trans_before[run_id] - adj_segment_alpha;
    float4 grad_rgba = float4(
        weights[run_id] * grad_rgb.x,
        weights[run_id] * grad_rgb.y,
        weights[run_id] * grad_rgb.z,
        0.0f);
    if (raw_density > 0.0f) {
      grad_rgba.w = adj_segment_trans * (-lengths[run_id] * segment_trans[run_id]);
    }
    if (reduce_small_sites) {
      tg_site_grad[local_frame * WF2_ENDPOINT_COMPACT_REDUCE32_SITES + owner] += grad_rgba;
    } else {
      wf2_atomic_add4(grad_site_rgba_f32, rgba_base, grad_rgba);
    }
    adj_next_transmittance = adj_trans_before;
  }
  }
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (local_frame == 0u) {
    float loss_sum = 0.0f;
    for (uint frame_id = 0u; frame_id < frames_in_chunk; ++frame_id) {
      loss_sum += tg_loss[frame_id];
    }
    atomic_fetch_add_explicit(&loss_f32[0], loss_sum, memory_order_relaxed);
  }

  if (reduce_small_sites) {
    if (local_frame < site_count) {
      float4 grad_sum = float4(0.0f, 0.0f, 0.0f, 0.0f);
      for (uint frame_id = 0u; frame_id < frames_in_chunk; ++frame_id) {
        grad_sum += tg_site_grad[
            frame_id * WF2_ENDPOINT_COMPACT_REDUCE32_SITES + local_frame];
      }
      wf2_atomic_add4(grad_site_rgba_f32, local_frame * 4u, grad_sum);
    }
  }
}


kernel void wf2_endpoint_record_delta_replace_coeff16_packed_framegroup16_rowdesc_mse_vjp_direct_atomic_rgb_only_tensor(
    device const half* coeff_f16 [[buffer(0)]],
    device const float* frame_t_f32 [[buffer(1)]],
    device const int* row_begin_i32 [[buffer(2)]],
    device const short* row_len_source_i16 [[buffer(3)]],
    device const int* base_record_i32 [[buffer(4)]],
    device const int* change_record_i32 [[buffer(5)]],
    device const float* site_rgba_f32 [[buffer(6)]],
    device const float* target_rgb_f32 [[buffer(7)]],
    device const int* config_i32 [[buffer(8)]],
    device const float* config_f32 [[buffer(9)]],
    device atomic_float* loss_f32 [[buffer(10)]],
    device atomic_float* grad_site_rgba_f32 [[buffer(11)]],
    uint3 local_pos [[thread_position_in_threadgroup]],
    uint3 group_id [[threadgroup_position_in_grid]]) {
  const uint local_frame = local_pos.x;
  const uint boundary_count = uint(config_i32[0]);
  const uint track_count = uint(config_i32[1]);
  const uint frame_count = uint(config_i32[2]);
  const uint site_count = uint(config_i32[3]);
  const uint base_record_count = uint(config_i32[4]);
  const uint change_record_count = uint(config_i32[6]);
  const uint chunk_count = (frame_count + WF2_ENDPOINT_DELTA_FRAMEGROUP_MAX_FRAMES - 1u) /
      WF2_ENDPOINT_DELTA_FRAMEGROUP_MAX_FRAMES;
  const uint group_index = group_id.x;
  const uint track_id = group_index / max(chunk_count, 1u);
  const uint chunk_id = group_index - track_id * max(chunk_count, 1u);
  const uint frame_start = chunk_id * WF2_ENDPOINT_DELTA_FRAMEGROUP_MAX_FRAMES;
  const uint frames_in_chunk = frame_start < frame_count
      ? min(WF2_ENDPOINT_DELTA_FRAMEGROUP_MAX_FRAMES, frame_count - frame_start)
      : 0u;

  threadgroup float tg_loss[WF2_ENDPOINT_DELTA_FRAMEGROUP_MAX_FRAMES];
  threadgroup float4 tg_site_grad[
      WF2_ENDPOINT_DELTA_FRAMEGROUP_MAX_FRAMES * WF2_ENDPOINT_EDIT_FRAMEGROUP_REDUCE_SITES];

  const bool reduce_small_sites = site_count <= WF2_ENDPOINT_EDIT_FRAMEGROUP_REDUCE_SITES;
  if (reduce_small_sites) {
    const uint site_grad_base = local_frame * WF2_ENDPOINT_EDIT_FRAMEGROUP_REDUCE_SITES;
    for (uint site_slot = 0u; site_slot < WF2_ENDPOINT_EDIT_FRAMEGROUP_REDUCE_SITES; ++site_slot) {
      tg_site_grad[site_grad_base + site_slot] = float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
  }
  tg_loss[local_frame] = 0.0f;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const bool active = track_id < track_count && local_frame < frames_in_chunk;
  if (active) {

  const uint global_frame_id = frame_start + local_frame;
  const uint sample_id = track_id * frame_count + global_frame_id;
  const uint total_samples = track_count * frame_count;
  const float near_depth = config_f32[0];
  const float far_depth = config_f32[1];
  const float invalid_epsilon = config_f32[2];
  const float transmittance_threshold = config_f32[3];
  const float t = frame_t_f32[global_frame_id];
  const uint desc_index = track_id * frame_count + global_frame_id;
  const int len_source_raw = active ? int(row_len_source_i16[desc_index]) : 0;
  const bool use_change = (len_source_raw & 0x4000) != 0;
  const int row_count_raw = len_source_raw & 0x3fff;
  const int begin_raw = active ? row_begin_i32[desc_index] : 0;
  const int end_raw = begin_raw + row_count_raw;
  const uint record_count = use_change ? change_record_count : base_record_count;
  const bool valid_row_bounds = begin_raw >= 0 && end_raw >= begin_raw &&
      uint(end_raw) <= record_count && uint(row_count_raw) <= WF2_MAX_REALRAY_SEGMENTS;
  const uint row_count = valid_row_bounds ? uint(row_count_raw) : 0u;
  const uint out_base = sample_id * 3u;
  const float3 target_rgb = float3(
      target_rgb_f32[out_base + 0u],
      target_rgb_f32[out_base + 1u],
      target_rgb_f32[out_base + 2u]);
  const float inv_element_count = 1.0f / float(total_samples * 3u);
  if (valid_row_bounds && row_count == 0u) {
    tg_loss[local_frame] = dot(target_rgb, target_rgb) * inv_element_count;
  } else if (valid_row_bounds) {

  uint owners[WF2_MAX_REALRAY_SEGMENTS];
  float lengths[WF2_MAX_REALRAY_SEGMENTS];
  float trans_before[WF2_MAX_REALRAY_SEGMENTS];
  float segment_trans[WF2_MAX_REALRAY_SEGMENTS];
  float segment_alpha[WF2_MAX_REALRAY_SEGMENTS];
  float weights[WF2_MAX_REALRAY_SEGMENTS];
  float3 segment_rgb[WF2_MAX_REALRAY_SEGMENTS];

  float3 rgb_accum = float3(0.0f, 0.0f, 0.0f);
  float transmittance = 1.0f;
  uint local_run_count = 0u;
  int cached_right_cut = -2147483648;
  float cached_right_depth = 0.0f;
  bool cached_right_valid = false;
  for (uint cursor = 0u; cursor < row_count; ++cursor) {
    if (transmittance <= transmittance_threshold || local_run_count >= WF2_MAX_REALRAY_SEGMENTS) {
      break;
    }
    const uint record_base = uint(begin_raw) + cursor;
    int owner_raw = 0;
    int left_cut = 0;
    int right_cut = 0;
    wf2_endpoint_record_unpack_record(
        use_change ? change_record_i32[record_base] : base_record_i32[record_base],
        owner_raw,
        left_cut,
        right_cut);
    if (owner_raw < 0 || uint(owner_raw) >= site_count) {
      continue;
    }
    float start_depth = 0.0f;
    float end_depth = 0.0f;
    bool start_valid = false;
    if (cached_right_valid && left_cut == cached_right_cut) {
      start_depth = cached_right_depth;
      start_valid = true;
    } else {
      start_valid = wf2_endpoint_record_coeff16_cut_depth(
          coeff_f16,
          boundary_count,
          track_id,
          left_cut,
          t,
          near_depth,
          far_depth,
          invalid_epsilon,
          start_depth);
    }
    if (!start_valid ||
        !wf2_endpoint_record_coeff16_cut_depth(
            coeff_f16,
            boundary_count,
            track_id,
            right_cut,
            t,
            near_depth,
            far_depth,
            invalid_epsilon,
            end_depth)) {
      cached_right_valid = false;
      continue;
    }
    cached_right_cut = right_cut;
    cached_right_depth = end_depth;
    cached_right_valid = true;
    const float length = end_depth - start_depth;
    if (!(length > 1.0e-8f)) {
      continue;
    }
    const uint owner = uint(owner_raw);
    const uint rgba_base = owner * 4u;
    const float density = max(site_rgba_f32[rgba_base + 3u], 0.0f);
    const float seg_trans = exp(-density * length);
    const float seg_alpha = 1.0f - seg_trans;
    const float weight = transmittance * seg_alpha;
    const float3 rgb = float3(
        site_rgba_f32[rgba_base + 0u],
        site_rgba_f32[rgba_base + 1u],
        site_rgba_f32[rgba_base + 2u]);

    owners[local_run_count] = owner;
    lengths[local_run_count] = length;
    trans_before[local_run_count] = transmittance;
    segment_trans[local_run_count] = seg_trans;
    segment_alpha[local_run_count] = seg_alpha;
    weights[local_run_count] = weight;
    segment_rgb[local_run_count] = rgb;
    local_run_count += 1u;

    rgb_accum += weight * rgb;
    transmittance *= seg_trans;
  }

  const float3 diff = rgb_accum - target_rgb;
  const float sample_loss = dot(diff, diff) * inv_element_count;
  tg_loss[local_frame] = sample_loss;
  const float3 grad_rgb = (2.0f * inv_element_count) * diff;

  float adj_next_transmittance = 0.0f;
  for (int run_id = int(local_run_count) - 1; run_id >= 0; --run_id) {
    const uint owner = owners[run_id];
    const uint rgba_base = owner * 4u;
    const float raw_density = site_rgba_f32[rgba_base + 3u];
    const float adj_weight = dot(grad_rgb, segment_rgb[run_id]);
    const float adj_trans_before =
        adj_weight * segment_alpha[run_id] +
        adj_next_transmittance * segment_trans[run_id];
    const float adj_segment_alpha = adj_weight * trans_before[run_id];
    const float adj_segment_trans = adj_next_transmittance * trans_before[run_id] - adj_segment_alpha;
    float4 grad_rgba = float4(
        weights[run_id] * grad_rgb.x,
        weights[run_id] * grad_rgb.y,
        weights[run_id] * grad_rgb.z,
        0.0f);
    if (raw_density > 0.0f) {
      grad_rgba.w = adj_segment_trans * (-lengths[run_id] * segment_trans[run_id]);
    }
    if (reduce_small_sites) {
      tg_site_grad[local_frame * WF2_ENDPOINT_EDIT_FRAMEGROUP_REDUCE_SITES + owner] += grad_rgba;
    } else {
      wf2_atomic_add4(grad_site_rgba_f32, rgba_base, grad_rgba);
    }
    adj_next_transmittance = adj_trans_before;
  }
  }
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (local_frame == 0u) {
    float loss_sum = 0.0f;
    for (uint frame_id = 0u; frame_id < frames_in_chunk; ++frame_id) {
      loss_sum += tg_loss[frame_id];
    }
    atomic_fetch_add_explicit(&loss_f32[0], loss_sum, memory_order_relaxed);
  }

  if (reduce_small_sites) {
    if (local_frame < site_count) {
      float4 grad_sum = float4(0.0f, 0.0f, 0.0f, 0.0f);
      for (uint frame_id = 0u; frame_id < frames_in_chunk; ++frame_id) {
        grad_sum += tg_site_grad[
            frame_id * WF2_ENDPOINT_EDIT_FRAMEGROUP_REDUCE_SITES + local_frame];
      }
      wf2_atomic_add4(grad_site_rgba_f32, local_frame * 4u, grad_sum);
    }
  }
}

kernel void wf2_endpoint_record_delta_replace_coeff16_packed_framegroup16_rowdesc32_mse_vjp_direct_atomic_rgb_only_tensor(
    device const half* coeff_f16 [[buffer(0)]],
    device const float* frame_t_f32 [[buffer(1)]],
    device const int* row_begin_i32 [[buffer(2)]],
    device const short* row_len_source_i16 [[buffer(3)]],
    device const int* base_record_i32 [[buffer(4)]],
    device const int* change_record_i32 [[buffer(5)]],
    device const float* site_rgba_f32 [[buffer(6)]],
    device const float* target_rgb_f32 [[buffer(7)]],
    device const int* config_i32 [[buffer(8)]],
    device const float* config_f32 [[buffer(9)]],
    device atomic_float* loss_f32 [[buffer(10)]],
    device atomic_float* grad_site_rgba_f32 [[buffer(11)]],
    uint3 local_pos [[thread_position_in_threadgroup]],
    uint3 group_id [[threadgroup_position_in_grid]]) {
  const uint local_frame = local_pos.x;
  const uint boundary_count = uint(config_i32[0]);
  const uint track_count = uint(config_i32[1]);
  const uint frame_count = uint(config_i32[2]);
  const uint site_count = uint(config_i32[3]);
  const uint base_record_count = uint(config_i32[4]);
  const uint change_record_count = uint(config_i32[6]);
  const uint chunk_count = (frame_count + WF2_ENDPOINT_DELTA_FRAMEGROUP_MAX_FRAMES - 1u) /
      WF2_ENDPOINT_DELTA_FRAMEGROUP_MAX_FRAMES;
  const uint group_index = group_id.x;
  const uint track_id = group_index / max(chunk_count, 1u);
  const uint chunk_id = group_index - track_id * max(chunk_count, 1u);
  const uint frame_start = chunk_id * WF2_ENDPOINT_DELTA_FRAMEGROUP_MAX_FRAMES;
  const uint frames_in_chunk = frame_start < frame_count
      ? min(WF2_ENDPOINT_DELTA_FRAMEGROUP_MAX_FRAMES, frame_count - frame_start)
      : 0u;

  threadgroup float tg_loss[WF2_ENDPOINT_DELTA_FRAMEGROUP_MAX_FRAMES];
  threadgroup float4 tg_site_grad[
      WF2_ENDPOINT_DELTA_FRAMEGROUP_MAX_FRAMES * WF2_ENDPOINT_ROWDESC32_REDUCE_SITES];

  const bool reduce_small_sites = site_count <= WF2_ENDPOINT_ROWDESC32_REDUCE_SITES;
  if (reduce_small_sites) {
    const uint site_grad_base = local_frame * WF2_ENDPOINT_ROWDESC32_REDUCE_SITES;
    for (uint site_slot = 0u; site_slot < WF2_ENDPOINT_ROWDESC32_REDUCE_SITES; ++site_slot) {
      tg_site_grad[site_grad_base + site_slot] = float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
  }
  tg_loss[local_frame] = 0.0f;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const bool active = track_id < track_count && local_frame < frames_in_chunk;
  if (active) {

  const uint global_frame_id = frame_start + local_frame;
  const uint sample_id = track_id * frame_count + global_frame_id;
  const uint total_samples = track_count * frame_count;
  const float near_depth = config_f32[0];
  const float far_depth = config_f32[1];
  const float invalid_epsilon = config_f32[2];
  const float transmittance_threshold = config_f32[3];
  const float t = frame_t_f32[global_frame_id];
  const uint desc_index = track_id * frame_count + global_frame_id;
  const int len_source_raw = active ? int(row_len_source_i16[desc_index]) : 0;
  const bool use_change = (len_source_raw & 0x4000) != 0;
  const int row_count_raw = len_source_raw & 0x3fff;
  const int begin_raw = active ? row_begin_i32[desc_index] : 0;
  const int end_raw = begin_raw + row_count_raw;
  const uint record_count = use_change ? change_record_count : base_record_count;
  const bool valid_row_bounds = begin_raw >= 0 && end_raw >= begin_raw &&
      uint(end_raw) <= record_count && uint(row_count_raw) <= WF2_MAX_REALRAY_SEGMENTS;
  const uint row_count = valid_row_bounds ? uint(row_count_raw) : 0u;
  const uint out_base = sample_id * 3u;
  const float3 target_rgb = float3(
      target_rgb_f32[out_base + 0u],
      target_rgb_f32[out_base + 1u],
      target_rgb_f32[out_base + 2u]);
  const float inv_element_count = 1.0f / float(total_samples * 3u);
  if (valid_row_bounds && row_count == 0u) {
    tg_loss[local_frame] = dot(target_rgb, target_rgb) * inv_element_count;
  } else if (valid_row_bounds) {

  uint owners[WF2_MAX_REALRAY_SEGMENTS];
  float lengths[WF2_MAX_REALRAY_SEGMENTS];
  float trans_before[WF2_MAX_REALRAY_SEGMENTS];
  float segment_trans[WF2_MAX_REALRAY_SEGMENTS];
  float segment_alpha[WF2_MAX_REALRAY_SEGMENTS];
  float weights[WF2_MAX_REALRAY_SEGMENTS];
  float3 segment_rgb[WF2_MAX_REALRAY_SEGMENTS];

  float3 rgb_accum = float3(0.0f, 0.0f, 0.0f);
  float transmittance = 1.0f;
  uint local_run_count = 0u;
  int cached_right_cut = -2147483648;
  float cached_right_depth = 0.0f;
  bool cached_right_valid = false;
  for (uint cursor = 0u; cursor < row_count; ++cursor) {
    if (transmittance <= transmittance_threshold || local_run_count >= WF2_MAX_REALRAY_SEGMENTS) {
      break;
    }
    const uint record_base = uint(begin_raw) + cursor;
    int owner_raw = 0;
    int left_cut = 0;
    int right_cut = 0;
    wf2_endpoint_record_unpack_record(
        use_change ? change_record_i32[record_base] : base_record_i32[record_base],
        owner_raw,
        left_cut,
        right_cut);
    if (owner_raw < 0 || uint(owner_raw) >= site_count) {
      continue;
    }
    float start_depth = 0.0f;
    float end_depth = 0.0f;
    bool start_valid = false;
    if (cached_right_valid && left_cut == cached_right_cut) {
      start_depth = cached_right_depth;
      start_valid = true;
    } else {
      start_valid = wf2_endpoint_record_coeff16_cut_depth(
          coeff_f16,
          boundary_count,
          track_id,
          left_cut,
          t,
          near_depth,
          far_depth,
          invalid_epsilon,
          start_depth);
    }
    if (!start_valid ||
        !wf2_endpoint_record_coeff16_cut_depth(
            coeff_f16,
            boundary_count,
            track_id,
            right_cut,
            t,
            near_depth,
            far_depth,
            invalid_epsilon,
            end_depth)) {
      cached_right_valid = false;
      continue;
    }
    cached_right_cut = right_cut;
    cached_right_depth = end_depth;
    cached_right_valid = true;
    const float length = end_depth - start_depth;
    if (!(length > 1.0e-8f)) {
      continue;
    }
    const uint owner = uint(owner_raw);
    const uint rgba_base = owner * 4u;
    const float density = max(site_rgba_f32[rgba_base + 3u], 0.0f);
    const float seg_trans = exp(-density * length);
    const float seg_alpha = 1.0f - seg_trans;
    const float weight = transmittance * seg_alpha;
    const float3 rgb = float3(
        site_rgba_f32[rgba_base + 0u],
        site_rgba_f32[rgba_base + 1u],
        site_rgba_f32[rgba_base + 2u]);

    owners[local_run_count] = owner;
    lengths[local_run_count] = length;
    trans_before[local_run_count] = transmittance;
    segment_trans[local_run_count] = seg_trans;
    segment_alpha[local_run_count] = seg_alpha;
    weights[local_run_count] = weight;
    segment_rgb[local_run_count] = rgb;
    local_run_count += 1u;

    rgb_accum += weight * rgb;
    transmittance *= seg_trans;
  }

  const float3 diff = rgb_accum - target_rgb;
  const float sample_loss = dot(diff, diff) * inv_element_count;
  tg_loss[local_frame] = sample_loss;
  const float3 grad_rgb = (2.0f * inv_element_count) * diff;

  float adj_next_transmittance = 0.0f;
  for (int run_id = int(local_run_count) - 1; run_id >= 0; --run_id) {
    const uint owner = owners[run_id];
    const uint rgba_base = owner * 4u;
    const float raw_density = site_rgba_f32[rgba_base + 3u];
    const float adj_weight = dot(grad_rgb, segment_rgb[run_id]);
    const float adj_trans_before =
        adj_weight * segment_alpha[run_id] +
        adj_next_transmittance * segment_trans[run_id];
    const float adj_segment_alpha = adj_weight * trans_before[run_id];
    const float adj_segment_trans = adj_next_transmittance * trans_before[run_id] - adj_segment_alpha;
    float4 grad_rgba = float4(
        weights[run_id] * grad_rgb.x,
        weights[run_id] * grad_rgb.y,
        weights[run_id] * grad_rgb.z,
        0.0f);
    if (raw_density > 0.0f) {
      grad_rgba.w = adj_segment_trans * (-lengths[run_id] * segment_trans[run_id]);
    }
    if (reduce_small_sites) {
      tg_site_grad[local_frame * WF2_ENDPOINT_ROWDESC32_REDUCE_SITES + owner] += grad_rgba;
    } else {
      wf2_atomic_add4(grad_site_rgba_f32, rgba_base, grad_rgba);
    }
    adj_next_transmittance = adj_trans_before;
  }
  }
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (local_frame == 0u) {
    float loss_sum = 0.0f;
    for (uint frame_id = 0u; frame_id < frames_in_chunk; ++frame_id) {
      loss_sum += tg_loss[frame_id];
    }
    atomic_fetch_add_explicit(&loss_f32[0], loss_sum, memory_order_relaxed);
  }

  if (reduce_small_sites) {
    if (local_frame < site_count) {
      float4 grad_sum = float4(0.0f, 0.0f, 0.0f, 0.0f);
      for (uint frame_id = 0u; frame_id < frames_in_chunk; ++frame_id) {
        grad_sum += tg_site_grad[
            frame_id * WF2_ENDPOINT_ROWDESC32_REDUCE_SITES + local_frame];
      }
      wf2_atomic_add4(grad_site_rgba_f32, local_frame * 4u, grad_sum);
    }
  }
}


kernel void wf2_endpoint_record_delta_replace_coeff16_packed_framegroup16_recompute_mse_vjp_direct_atomic_rgb_only_tensor(
    device const half* coeff_f16 [[buffer(0)]],
    device const float* frame_t_f32 [[buffer(1)]],
    device const int* base_offsets_i32 [[buffer(2)]],
    device const int* base_record_i32 [[buffer(3)]],
    device const int* track_change_offsets_i32 [[buffer(4)]],
    device const short* track_chunk_change_offsets_i16 [[buffer(5)]],
    device const int* change_frame_i32 [[buffer(6)]],
    device const int* change_offsets_i32 [[buffer(7)]],
    device const int* change_record_i32 [[buffer(8)]],
    device const float* site_rgba_f32 [[buffer(9)]],
    device const float* target_rgb_f32 [[buffer(10)]],
    device const int* config_i32 [[buffer(11)]],
    device const float* config_f32 [[buffer(12)]],
    device atomic_float* loss_f32 [[buffer(13)]],
    device atomic_float* grad_site_rgba_f32 [[buffer(14)]],
    uint3 local_pos [[thread_position_in_threadgroup]],
    uint3 group_id [[threadgroup_position_in_grid]]) {
  const uint local_frame = local_pos.x;
  const uint boundary_count = uint(config_i32[0]);
  const uint track_count = uint(config_i32[1]);
  const uint frame_count = uint(config_i32[2]);
  const uint site_count = uint(config_i32[3]);
  const uint base_record_count = uint(config_i32[4]);
  const uint change_count = uint(config_i32[5]);
  const uint change_record_count = uint(config_i32[6]);
  const uint chunk_count = (frame_count + WF2_ENDPOINT_DELTA_FRAMEGROUP_MAX_FRAMES - 1u) /
      WF2_ENDPOINT_DELTA_FRAMEGROUP_MAX_FRAMES;
  const uint group_index = group_id.x;
  const uint track_id = group_index / max(chunk_count, 1u);
  const uint chunk_id = group_index - track_id * max(chunk_count, 1u);
  const uint frame_start = chunk_id * WF2_ENDPOINT_DELTA_FRAMEGROUP_MAX_FRAMES;
  const uint frames_in_chunk = frame_start < frame_count
      ? min(WF2_ENDPOINT_DELTA_FRAMEGROUP_MAX_FRAMES, frame_count - frame_start)
      : 0u;

  threadgroup int tg_valid[WF2_ENDPOINT_DELTA_FRAMEGROUP_MAX_FRAMES];
  threadgroup int tg_source[WF2_ENDPOINT_DELTA_FRAMEGROUP_MAX_FRAMES];
  threadgroup int tg_begin[WF2_ENDPOINT_DELTA_FRAMEGROUP_MAX_FRAMES];
  threadgroup int tg_end[WF2_ENDPOINT_DELTA_FRAMEGROUP_MAX_FRAMES];
  threadgroup float tg_loss[WF2_ENDPOINT_DELTA_FRAMEGROUP_MAX_FRAMES];
  threadgroup float4 tg_site_grad[
      WF2_ENDPOINT_DELTA_FRAMEGROUP_MAX_FRAMES * WF2_ENDPOINT_EDIT_FRAMEGROUP_REDUCE_SITES];

  if (local_frame == 0u) {
    for (uint frame_id = 0u; frame_id < WF2_ENDPOINT_DELTA_FRAMEGROUP_MAX_FRAMES; ++frame_id) {
      tg_valid[frame_id] = 0;
      tg_source[frame_id] = 0;
      tg_begin[frame_id] = 0;
      tg_end[frame_id] = 0;
    }

    bool ok = track_id < track_count && frames_in_chunk > 0u;
    int change_begin_raw = 0;
    int change_end_raw = 0;
    int chunk_change_begin_raw = 0;
    int chunk_change_end_raw = 0;
    if (ok) {
      change_begin_raw = track_change_offsets_i32[track_id];
      change_end_raw = track_change_offsets_i32[track_id + 1u];
      const uint chunk_offset_base = track_id * (chunk_count + 1u) + chunk_id;
      chunk_change_begin_raw = int(track_chunk_change_offsets_i16[chunk_offset_base]);
      chunk_change_end_raw = int(track_chunk_change_offsets_i16[chunk_offset_base + 1u]);
      ok = change_begin_raw >= 0 && change_end_raw >= change_begin_raw && uint(change_end_raw) <= change_count &&
          chunk_change_begin_raw >= change_begin_raw && chunk_change_end_raw >= chunk_change_begin_raw &&
          chunk_change_end_raw <= change_end_raw;
    }
    int selected_change = -1;
    if (ok && chunk_change_begin_raw > change_begin_raw) {
      selected_change = chunk_change_begin_raw - 1;
      while (selected_change >= change_begin_raw && change_frame_i32[uint(selected_change)] < 0) {
        selected_change -= 1;
      }
      if (selected_change < change_begin_raw) {
        selected_change = -1;
      }
    }

    uint change_cursor = ok ? uint(chunk_change_begin_raw) : 0u;
    const uint change_end = ok ? uint(chunk_change_end_raw) : 0u;

    for (uint local_frame_id = 0u; local_frame_id < frames_in_chunk; ++local_frame_id) {
      const uint global_frame_id = frame_start + local_frame_id;
      while (ok && change_cursor < change_end) {
        const int changed_frame = change_frame_i32[change_cursor];
        if (changed_frame < 0) {
          change_cursor += 1u;
          continue;
        }
        if (uint(changed_frame) > global_frame_id) {
          break;
        }

        selected_change = int(change_cursor);
        change_cursor += 1u;
      }

      if (ok) {
        const bool use_change = selected_change >= 0;
        const int begin_raw =
            use_change ? change_offsets_i32[uint(selected_change)] : base_offsets_i32[track_id];
        const int end_raw =
            use_change ? change_offsets_i32[uint(selected_change) + 1u] : base_offsets_i32[track_id + 1u];
        const uint record_count = use_change ? change_record_count : base_record_count;
        ok = begin_raw >= 0 && end_raw >= begin_raw && uint(end_raw) <= record_count &&
            uint(end_raw - begin_raw) <= WF2_MAX_REALRAY_SEGMENTS;
        if (!ok) {
          break;
        }
        tg_source[local_frame_id] = use_change ? 1 : 0;
        tg_begin[local_frame_id] = begin_raw;
        tg_end[local_frame_id] = end_raw;
        tg_valid[local_frame_id] = 1;
      }
    }
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  const bool reduce_small_sites = site_count <= WF2_ENDPOINT_EDIT_FRAMEGROUP_REDUCE_SITES;
  if (reduce_small_sites) {
    const uint site_grad_base = local_frame * WF2_ENDPOINT_EDIT_FRAMEGROUP_REDUCE_SITES;
    for (uint site_slot = 0u; site_slot < WF2_ENDPOINT_EDIT_FRAMEGROUP_REDUCE_SITES; ++site_slot) {
      tg_site_grad[site_grad_base + site_slot] = float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
  }
  tg_loss[local_frame] = 0.0f;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const bool active = track_id < track_count && local_frame < frames_in_chunk && tg_valid[local_frame] != 0;
  if (active) {

  const uint global_frame_id = frame_start + local_frame;
  const uint sample_id = track_id * frame_count + global_frame_id;
  const uint total_samples = track_count * frame_count;
  const float near_depth = config_f32[0];
  const float far_depth = config_f32[1];
  const float invalid_epsilon = config_f32[2];
  const float transmittance_threshold = config_f32[3];
  const float t = frame_t_f32[global_frame_id];
  const bool use_change = tg_source[local_frame] != 0;
  const int begin_raw = tg_begin[local_frame];
  const int end_raw = tg_end[local_frame];
  const bool valid_row_bounds = end_raw >= begin_raw;
  const uint row_count = valid_row_bounds ? uint(end_raw - begin_raw) : 0u;
  const uint out_base = sample_id * 3u;
  const float3 target_rgb = float3(
      target_rgb_f32[out_base + 0u],
      target_rgb_f32[out_base + 1u],
      target_rgb_f32[out_base + 2u]);
  const float inv_element_count = 1.0f / float(total_samples * 3u);
  if (valid_row_bounds && row_count == 0u) {
    tg_loss[local_frame] = dot(target_rgb, target_rgb) * inv_element_count;
  } else if (valid_row_bounds) {

  uint owners[WF2_MAX_REALRAY_SEGMENTS];
  float lengths[WF2_MAX_REALRAY_SEGMENTS];
  float trans_before[WF2_MAX_REALRAY_SEGMENTS];

  float3 rgb_accum = float3(0.0f, 0.0f, 0.0f);
  float transmittance = 1.0f;
  uint local_run_count = 0u;
  int cached_right_cut = -2147483648;
  float cached_right_depth = 0.0f;
  bool cached_right_valid = false;
  for (uint cursor = 0u; cursor < row_count; ++cursor) {
    if (transmittance <= transmittance_threshold || local_run_count >= WF2_MAX_REALRAY_SEGMENTS) {
      break;
    }
    const uint record_base = uint(begin_raw) + cursor;
    int owner_raw = 0;
    int left_cut = 0;
    int right_cut = 0;
    wf2_endpoint_record_unpack_record(
        use_change ? change_record_i32[record_base] : base_record_i32[record_base],
        owner_raw,
        left_cut,
        right_cut);
    if (owner_raw < 0 || uint(owner_raw) >= site_count) {
      continue;
    }
    float start_depth = 0.0f;
    float end_depth = 0.0f;
    bool start_valid = false;
    if (cached_right_valid && left_cut == cached_right_cut) {
      start_depth = cached_right_depth;
      start_valid = true;
    } else {
      start_valid = wf2_endpoint_record_coeff16_cut_depth(
          coeff_f16,
          boundary_count,
          track_id,
          left_cut,
          t,
          near_depth,
          far_depth,
          invalid_epsilon,
          start_depth);
    }
    if (!start_valid ||
        !wf2_endpoint_record_coeff16_cut_depth(
            coeff_f16,
            boundary_count,
            track_id,
            right_cut,
            t,
            near_depth,
            far_depth,
            invalid_epsilon,
            end_depth)) {
      cached_right_valid = false;
      continue;
    }
    cached_right_cut = right_cut;
    cached_right_depth = end_depth;
    cached_right_valid = true;
    const float length = end_depth - start_depth;
    if (!(length > 1.0e-8f)) {
      continue;
    }
    const uint owner = uint(owner_raw);
    const uint rgba_base = owner * 4u;
    const float density = max(site_rgba_f32[rgba_base + 3u], 0.0f);
    const float seg_trans = exp(-density * length);
    const float seg_alpha = 1.0f - seg_trans;
    const float weight = transmittance * seg_alpha;
    const float3 rgb = float3(
        site_rgba_f32[rgba_base + 0u],
        site_rgba_f32[rgba_base + 1u],
        site_rgba_f32[rgba_base + 2u]);

    owners[local_run_count] = owner;
    lengths[local_run_count] = length;
    trans_before[local_run_count] = transmittance;
    local_run_count += 1u;

    rgb_accum += weight * rgb;
    transmittance *= seg_trans;
  }

  const float3 diff = rgb_accum - target_rgb;
  const float sample_loss = dot(diff, diff) * inv_element_count;
  tg_loss[local_frame] = sample_loss;
  const float3 grad_rgb = (2.0f * inv_element_count) * diff;

  float adj_next_transmittance = 0.0f;
  for (int run_id = int(local_run_count) - 1; run_id >= 0; --run_id) {
    const uint owner = owners[run_id];
    const uint rgba_base = owner * 4u;
    const float raw_density = site_rgba_f32[rgba_base + 3u];
    const float density = max(raw_density, 0.0f);
    const float length = lengths[run_id];
    const float seg_trans = exp(-density * length);
    const float seg_alpha = 1.0f - seg_trans;
    const float weight = trans_before[run_id] * seg_alpha;
    const float3 rgb = float3(
        site_rgba_f32[rgba_base + 0u],
        site_rgba_f32[rgba_base + 1u],
        site_rgba_f32[rgba_base + 2u]);
    const float adj_weight = dot(grad_rgb, rgb);
    const float adj_trans_before = adj_weight * seg_alpha + adj_next_transmittance * seg_trans;
    const float adj_segment_alpha = adj_weight * trans_before[run_id];
    const float adj_segment_trans = adj_next_transmittance * trans_before[run_id] - adj_segment_alpha;
    float4 grad_rgba = float4(
        weight * grad_rgb.x,
        weight * grad_rgb.y,
        weight * grad_rgb.z,
        0.0f);
    if (raw_density > 0.0f) {
      grad_rgba.w = adj_segment_trans * (-length * seg_trans);
    }
    if (reduce_small_sites) {
      tg_site_grad[local_frame * WF2_ENDPOINT_EDIT_FRAMEGROUP_REDUCE_SITES + owner] += grad_rgba;
    } else {
      wf2_atomic_add4(grad_site_rgba_f32, rgba_base, grad_rgba);
    }
    adj_next_transmittance = adj_trans_before;
  }
  }
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (local_frame == 0u) {
    float loss_sum = 0.0f;
    for (uint frame_id = 0u; frame_id < frames_in_chunk; ++frame_id) {
      loss_sum += tg_loss[frame_id];
    }
    atomic_fetch_add_explicit(&loss_f32[0], loss_sum, memory_order_relaxed);
  }

  if (reduce_small_sites) {
    if (local_frame < site_count) {
      float4 grad_sum = float4(0.0f, 0.0f, 0.0f, 0.0f);
      for (uint frame_id = 0u; frame_id < frames_in_chunk; ++frame_id) {
        grad_sum += tg_site_grad[
            frame_id * WF2_ENDPOINT_EDIT_FRAMEGROUP_REDUCE_SITES + local_frame];
      }
      wf2_atomic_add4(grad_site_rgba_f32, local_frame * 4u, grad_sum);
    }
  }
}

kernel void wf2_endpoint_record_delta_replace_factorized_packed_framegroup16_recompute_mse_vjp_direct_atomic_rgb_only_tensor(
    device const float* boundary_f32 [[buffer(0)]],
    device const float* track_ray_coeff_f32 [[buffer(1)]],
    device const float* frame_t_f32 [[buffer(2)]],
    device const short* base_offsets_i32 [[buffer(3)]],
    device const int* base_record_i32 [[buffer(4)]],
    device const short* track_change_offsets_i32 [[buffer(5)]],
    device const short* track_chunk_change_offsets_i16 [[buffer(6)]],
    device const short* change_frame_i32 [[buffer(7)]],
    device const short* change_offsets_i32 [[buffer(8)]],
    device const int* change_record_i32 [[buffer(9)]],
    device const float* site_rgba_f32 [[buffer(10)]],
    device const float* target_rgb_f32 [[buffer(11)]],
    device const int* config_i32 [[buffer(12)]],
    device const float* config_f32 [[buffer(13)]],
    device atomic_float* loss_f32 [[buffer(14)]],
    device atomic_float* grad_site_rgba_f32 [[buffer(15)]],
    uint3 local_pos [[thread_position_in_threadgroup]],
    uint3 group_id [[threadgroup_position_in_grid]]) {
  const uint local_frame = local_pos.x;
  const uint boundary_count = uint(config_i32[0]);
  const uint track_count = uint(config_i32[1]);
  const uint frame_count = uint(config_i32[2]);
  const uint site_count = uint(config_i32[3]);
  const uint base_record_count = uint(config_i32[4]);
  const uint change_count = uint(config_i32[5]);
  const uint change_record_count = uint(config_i32[6]);
  const uint chunk_count = (frame_count + WF2_ENDPOINT_DELTA_FRAMEGROUP_MAX_FRAMES - 1u) /
      WF2_ENDPOINT_DELTA_FRAMEGROUP_MAX_FRAMES;
  const uint group_index = group_id.x;
  const uint track_id = group_index / max(chunk_count, 1u);
  const uint chunk_id = group_index - track_id * max(chunk_count, 1u);
  const uint frame_start = chunk_id * WF2_ENDPOINT_DELTA_FRAMEGROUP_MAX_FRAMES;
  const uint frames_in_chunk = frame_start < frame_count
      ? min(WF2_ENDPOINT_DELTA_FRAMEGROUP_MAX_FRAMES, frame_count - frame_start)
      : 0u;

  threadgroup int tg_valid[WF2_ENDPOINT_DELTA_FRAMEGROUP_MAX_FRAMES];
  threadgroup int tg_source[WF2_ENDPOINT_DELTA_FRAMEGROUP_MAX_FRAMES];
  threadgroup int tg_begin[WF2_ENDPOINT_DELTA_FRAMEGROUP_MAX_FRAMES];
  threadgroup int tg_end[WF2_ENDPOINT_DELTA_FRAMEGROUP_MAX_FRAMES];
  threadgroup float tg_loss[WF2_ENDPOINT_DELTA_FRAMEGROUP_MAX_FRAMES];
  threadgroup float4 tg_site_grad[
      WF2_ENDPOINT_DELTA_FRAMEGROUP_MAX_FRAMES * WF2_ENDPOINT_EDIT_FRAMEGROUP_REDUCE_SITES];

  if (local_frame == 0u) {
    for (uint frame_id = 0u; frame_id < WF2_ENDPOINT_DELTA_FRAMEGROUP_MAX_FRAMES; ++frame_id) {
      tg_valid[frame_id] = 0;
      tg_source[frame_id] = 0;
      tg_begin[frame_id] = 0;
      tg_end[frame_id] = 0;
    }

    bool ok = track_id < track_count && frames_in_chunk > 0u;
    int change_begin_raw = 0;
    int change_end_raw = 0;
    int chunk_change_begin_raw = 0;
    int chunk_change_end_raw = 0;
    if (ok) {
      change_begin_raw = track_change_offsets_i32[track_id];
      change_end_raw = track_change_offsets_i32[track_id + 1u];
      const uint chunk_offset_base = track_id * (chunk_count + 1u) + chunk_id;
      chunk_change_begin_raw = int(track_chunk_change_offsets_i16[chunk_offset_base]);
      chunk_change_end_raw = int(track_chunk_change_offsets_i16[chunk_offset_base + 1u]);
      ok = change_begin_raw >= 0 && change_end_raw >= change_begin_raw && uint(change_end_raw) <= change_count &&
          chunk_change_begin_raw >= change_begin_raw && chunk_change_end_raw >= chunk_change_begin_raw &&
          chunk_change_end_raw <= change_end_raw;
    }
    int selected_change = -1;
    if (ok && chunk_change_begin_raw > change_begin_raw) {
      selected_change = chunk_change_begin_raw - 1;
      while (selected_change >= change_begin_raw && change_frame_i32[uint(selected_change)] < 0) {
        selected_change -= 1;
      }
      if (selected_change < change_begin_raw) {
        selected_change = -1;
      }
    }

    uint change_cursor = ok ? uint(chunk_change_begin_raw) : 0u;
    const uint change_end = ok ? uint(chunk_change_end_raw) : 0u;

    for (uint local_frame_id = 0u; local_frame_id < frames_in_chunk; ++local_frame_id) {
      const uint global_frame_id = frame_start + local_frame_id;
      while (ok && change_cursor < change_end) {
        const int changed_frame = change_frame_i32[change_cursor];
        if (changed_frame < 0) {
          change_cursor += 1u;
          continue;
        }
        if (uint(changed_frame) > global_frame_id) {
          break;
        }

        selected_change = int(change_cursor);
        change_cursor += 1u;
      }

      if (ok) {
        const bool use_change = selected_change >= 0;
        const int begin_raw =
            use_change ? change_offsets_i32[uint(selected_change)] : base_offsets_i32[track_id];
        const int end_raw =
            use_change ? change_offsets_i32[uint(selected_change) + 1u] : base_offsets_i32[track_id + 1u];
        const uint record_count = use_change ? change_record_count : base_record_count;
        ok = begin_raw >= 0 && end_raw >= begin_raw && uint(end_raw) <= record_count &&
            uint(end_raw - begin_raw) <= WF2_MAX_REALRAY_SEGMENTS;
        if (!ok) {
          break;
        }
        tg_source[local_frame_id] = use_change ? 1 : 0;
        tg_begin[local_frame_id] = begin_raw;
        tg_end[local_frame_id] = end_raw;
        tg_valid[local_frame_id] = 1;
      }
    }
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  const bool reduce_small_sites = site_count <= WF2_ENDPOINT_EDIT_FRAMEGROUP_REDUCE_SITES;
  if (reduce_small_sites) {
    const uint site_grad_base = local_frame * WF2_ENDPOINT_EDIT_FRAMEGROUP_REDUCE_SITES;
    for (uint site_slot = 0u; site_slot < WF2_ENDPOINT_EDIT_FRAMEGROUP_REDUCE_SITES; ++site_slot) {
      tg_site_grad[site_grad_base + site_slot] = float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
  }
  tg_loss[local_frame] = 0.0f;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const bool active = track_id < track_count && local_frame < frames_in_chunk && tg_valid[local_frame] != 0;
  if (active) {

  const uint global_frame_id = frame_start + local_frame;
  const uint sample_id = track_id * frame_count + global_frame_id;
  const uint total_samples = track_count * frame_count;
  const float near_depth = config_f32[0];
  const float far_depth = config_f32[1];
  const float invalid_epsilon = config_f32[2];
  const float transmittance_threshold = config_f32[3];
  const float t = frame_t_f32[global_frame_id];
  const bool use_change = tg_source[local_frame] != 0;
  const int begin_raw = tg_begin[local_frame];
  const int end_raw = tg_end[local_frame];
  const bool valid_row_bounds = end_raw >= begin_raw;
  const uint row_count = valid_row_bounds ? uint(end_raw - begin_raw) : 0u;
  const uint out_base = sample_id * 3u;
  const float3 target_rgb = float3(
      target_rgb_f32[out_base + 0u],
      target_rgb_f32[out_base + 1u],
      target_rgb_f32[out_base + 2u]);
  const float inv_element_count = 1.0f / float(total_samples * 3u);
  if (valid_row_bounds && row_count == 0u) {
    tg_loss[local_frame] = dot(target_rgb, target_rgb) * inv_element_count;
  } else if (valid_row_bounds) {

  uint owners[WF2_MAX_REALRAY_SEGMENTS];
  float lengths[WF2_MAX_REALRAY_SEGMENTS];
  float trans_before[WF2_MAX_REALRAY_SEGMENTS];

  float3 rgb_accum = float3(0.0f, 0.0f, 0.0f);
  float transmittance = 1.0f;
  uint local_run_count = 0u;
  int cached_right_cut = -2147483648;
  float cached_right_depth = 0.0f;
  bool cached_right_valid = false;
  for (uint cursor = 0u; cursor < row_count; ++cursor) {
    if (transmittance <= transmittance_threshold || local_run_count >= WF2_MAX_REALRAY_SEGMENTS) {
      break;
    }
    const uint record_base = uint(begin_raw) + cursor;
    int owner_raw = 0;
    int left_cut = 0;
    int right_cut = 0;
    wf2_endpoint_record_unpack_record(
        use_change ? change_record_i32[record_base] : base_record_i32[record_base],
        owner_raw,
        left_cut,
        right_cut);
    if (owner_raw < 0 || uint(owner_raw) >= site_count) {
      continue;
    }
    float start_depth = 0.0f;
    float end_depth = 0.0f;
    bool start_valid = false;
    if (cached_right_valid && left_cut == cached_right_cut) {
      start_depth = cached_right_depth;
      start_valid = true;
    } else {
      start_valid = wf2_endpoint_record_factorized_cut_depth(
          boundary_f32,
          track_ray_coeff_f32,
          boundary_count,
          track_id,
          left_cut,
          t,
          near_depth,
          far_depth,
          invalid_epsilon,
          start_depth);
    }
    if (!start_valid ||
        !wf2_endpoint_record_factorized_cut_depth(
            boundary_f32,
            track_ray_coeff_f32,
            boundary_count,
            track_id,
            right_cut,
            t,
            near_depth,
            far_depth,
            invalid_epsilon,
            end_depth)) {
      cached_right_valid = false;
      continue;
    }
    cached_right_cut = right_cut;
    cached_right_depth = end_depth;
    cached_right_valid = true;
    const float length = end_depth - start_depth;
    if (!(length > 1.0e-8f)) {
      continue;
    }
    const uint owner = uint(owner_raw);
    const uint rgba_base = owner * 4u;
    const float density = max(site_rgba_f32[rgba_base + 3u], 0.0f);
    const float seg_trans = exp(-density * length);
    const float seg_alpha = 1.0f - seg_trans;
    const float weight = transmittance * seg_alpha;
    const float3 rgb = float3(
        site_rgba_f32[rgba_base + 0u],
        site_rgba_f32[rgba_base + 1u],
        site_rgba_f32[rgba_base + 2u]);

    owners[local_run_count] = owner;
    lengths[local_run_count] = length;
    trans_before[local_run_count] = transmittance;
    local_run_count += 1u;

    rgb_accum += weight * rgb;
    transmittance *= seg_trans;
  }

  const float3 diff = rgb_accum - target_rgb;
  const float sample_loss = dot(diff, diff) * inv_element_count;
  tg_loss[local_frame] = sample_loss;
  const float3 grad_rgb = (2.0f * inv_element_count) * diff;

  float adj_next_transmittance = 0.0f;
  for (int run_id = int(local_run_count) - 1; run_id >= 0; --run_id) {
    const uint owner = owners[run_id];
    const uint rgba_base = owner * 4u;
    const float raw_density = site_rgba_f32[rgba_base + 3u];
    const float density = max(raw_density, 0.0f);
    const float length = lengths[run_id];
    const float seg_trans = exp(-density * length);
    const float seg_alpha = 1.0f - seg_trans;
    const float weight = trans_before[run_id] * seg_alpha;
    const float3 rgb = float3(
        site_rgba_f32[rgba_base + 0u],
        site_rgba_f32[rgba_base + 1u],
        site_rgba_f32[rgba_base + 2u]);
    const float adj_weight = dot(grad_rgb, rgb);
    const float adj_trans_before = adj_weight * seg_alpha + adj_next_transmittance * seg_trans;
    const float adj_segment_alpha = adj_weight * trans_before[run_id];
    const float adj_segment_trans = adj_next_transmittance * trans_before[run_id] - adj_segment_alpha;
    float4 grad_rgba = float4(
        weight * grad_rgb.x,
        weight * grad_rgb.y,
        weight * grad_rgb.z,
        0.0f);
    if (raw_density > 0.0f) {
      grad_rgba.w = adj_segment_trans * (-length * seg_trans);
    }
    if (reduce_small_sites) {
      tg_site_grad[local_frame * WF2_ENDPOINT_EDIT_FRAMEGROUP_REDUCE_SITES + owner] += grad_rgba;
    } else {
      wf2_atomic_add4(grad_site_rgba_f32, rgba_base, grad_rgba);
    }
    adj_next_transmittance = adj_trans_before;
  }
  }
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (local_frame == 0u) {
    float loss_sum = 0.0f;
    for (uint frame_id = 0u; frame_id < frames_in_chunk; ++frame_id) {
      loss_sum += tg_loss[frame_id];
    }
    atomic_fetch_add_explicit(&loss_f32[0], loss_sum, memory_order_relaxed);
  }

  if (reduce_small_sites) {
    if (local_frame < site_count) {
      float4 grad_sum = float4(0.0f, 0.0f, 0.0f, 0.0f);
      for (uint frame_id = 0u; frame_id < frames_in_chunk; ++frame_id) {
        grad_sum += tg_site_grad[
            frame_id * WF2_ENDPOINT_EDIT_FRAMEGROUP_REDUCE_SITES + local_frame];
      }
      wf2_atomic_add4(grad_site_rgba_f32, local_frame * 4u, grad_sum);
    }
  }
}

kernel void wf2_endpoint_record_delta_replace_factorized_packed_framegroup16_constant_state_mse_vjp_direct_atomic_rgb_boundary_tensor(
    device const float* boundary_f32 [[buffer(0)]],
    device const float* track_ray_coeff_f32 [[buffer(1)]],
    device const float* frame_t_f32 [[buffer(2)]],
    device const short* base_offsets_i32 [[buffer(3)]],
    device const int* base_record_i32 [[buffer(4)]],
    device const short* track_change_offsets_i32 [[buffer(5)]],
    device const short* track_chunk_change_offsets_i16 [[buffer(6)]],
    device const short* change_frame_i32 [[buffer(7)]],
    device const short* change_offsets_i32 [[buffer(8)]],
    device const int* change_record_i32 [[buffer(9)]],
    device const float* site_rgba_f32 [[buffer(10)]],
    device const float* target_rgb_f32 [[buffer(11)]],
    device const int* config_i32 [[buffer(12)]],
    device const float* config_f32 [[buffer(13)]],
    device atomic_float* loss_f32 [[buffer(14)]],
    device atomic_float* grad_site_rgba_f32 [[buffer(15)]],
    device atomic_float* grad_boundary_f32 [[buffer(16)]],
    uint3 local_pos [[thread_position_in_threadgroup]],
    uint3 group_id [[threadgroup_position_in_grid]]) {
  const uint local_frame = local_pos.x;
  const uint boundary_count = uint(config_i32[0]);
  const uint track_count = uint(config_i32[1]);
  const uint frame_count = uint(config_i32[2]);
  const uint site_count = uint(config_i32[3]);
  const uint base_record_count = uint(config_i32[4]);
  const uint change_count = uint(config_i32[5]);
  const uint change_record_count = uint(config_i32[6]);
  const uint chunk_count = (frame_count + WF2_ENDPOINT_DELTA_FRAMEGROUP_MAX_FRAMES - 1u) /
      WF2_ENDPOINT_DELTA_FRAMEGROUP_MAX_FRAMES;
  const uint group_index = group_id.x;
  const uint track_id = group_index / max(chunk_count, 1u);
  const uint chunk_id = group_index - track_id * max(chunk_count, 1u);
  const uint frame_start = chunk_id * WF2_ENDPOINT_DELTA_FRAMEGROUP_MAX_FRAMES;
  const uint frames_in_chunk = frame_start < frame_count
      ? min(WF2_ENDPOINT_DELTA_FRAMEGROUP_MAX_FRAMES, frame_count - frame_start)
      : 0u;

  threadgroup int tg_valid[WF2_ENDPOINT_DELTA_FRAMEGROUP_MAX_FRAMES];
  threadgroup int tg_source[WF2_ENDPOINT_DELTA_FRAMEGROUP_MAX_FRAMES];
  threadgroup int tg_begin[WF2_ENDPOINT_DELTA_FRAMEGROUP_MAX_FRAMES];
  threadgroup int tg_end[WF2_ENDPOINT_DELTA_FRAMEGROUP_MAX_FRAMES];
  threadgroup float tg_loss[WF2_ENDPOINT_DELTA_FRAMEGROUP_MAX_FRAMES];
  threadgroup float4 tg_site_grad[
      WF2_ENDPOINT_DELTA_FRAMEGROUP_MAX_FRAMES * WF2_ENDPOINT_EDIT_FRAMEGROUP_REDUCE_SITES];

  if (local_frame == 0u) {
    for (uint frame_id = 0u; frame_id < WF2_ENDPOINT_DELTA_FRAMEGROUP_MAX_FRAMES; ++frame_id) {
      tg_valid[frame_id] = 0;
      tg_source[frame_id] = 0;
      tg_begin[frame_id] = 0;
      tg_end[frame_id] = 0;
    }

    bool ok = track_id < track_count && frames_in_chunk > 0u;
    int change_begin_raw = 0;
    int change_end_raw = 0;
    int chunk_change_begin_raw = 0;
    int chunk_change_end_raw = 0;
    if (ok) {
      change_begin_raw = track_change_offsets_i32[track_id];
      change_end_raw = track_change_offsets_i32[track_id + 1u];
      const uint chunk_offset_base = track_id * (chunk_count + 1u) + chunk_id;
      chunk_change_begin_raw = int(track_chunk_change_offsets_i16[chunk_offset_base]);
      chunk_change_end_raw = int(track_chunk_change_offsets_i16[chunk_offset_base + 1u]);
      ok = change_begin_raw >= 0 && change_end_raw >= change_begin_raw && uint(change_end_raw) <= change_count &&
          chunk_change_begin_raw >= change_begin_raw && chunk_change_end_raw >= chunk_change_begin_raw &&
          chunk_change_end_raw <= change_end_raw;
    }
    int selected_change = -1;
    if (ok && chunk_change_begin_raw > change_begin_raw) {
      selected_change = chunk_change_begin_raw - 1;
      while (selected_change >= change_begin_raw && change_frame_i32[uint(selected_change)] < 0) {
        selected_change -= 1;
      }
      if (selected_change < change_begin_raw) {
        selected_change = -1;
      }
    }

    uint change_cursor = ok ? uint(chunk_change_begin_raw) : 0u;
    const uint change_end = ok ? uint(chunk_change_end_raw) : 0u;
    for (uint local_frame_id = 0u; local_frame_id < frames_in_chunk; ++local_frame_id) {
      const uint global_frame_id = frame_start + local_frame_id;
      while (ok && change_cursor < change_end) {
        const int changed_frame = change_frame_i32[change_cursor];
        if (changed_frame < 0) {
          change_cursor += 1u;
          continue;
        }
        if (uint(changed_frame) > global_frame_id) {
          break;
        }
        selected_change = int(change_cursor);
        change_cursor += 1u;
      }

      if (ok) {
        const bool use_change = selected_change >= 0;
        const int begin_raw =
            use_change ? change_offsets_i32[uint(selected_change)] : base_offsets_i32[track_id];
        const int end_raw =
            use_change ? change_offsets_i32[uint(selected_change) + 1u] : base_offsets_i32[track_id + 1u];
        const uint record_count = use_change ? change_record_count : base_record_count;
        ok = begin_raw >= 0 && end_raw >= begin_raw && uint(end_raw) <= record_count &&
            uint(end_raw - begin_raw) <= WF2_MAX_REALRAY_SEGMENTS;
        if (!ok) {
          break;
        }
        tg_source[local_frame_id] = use_change ? 1 : 0;
        tg_begin[local_frame_id] = begin_raw;
        tg_end[local_frame_id] = end_raw;
        tg_valid[local_frame_id] = 1;
      }
    }
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  const bool reduce_small_sites = site_count <= WF2_ENDPOINT_EDIT_FRAMEGROUP_REDUCE_SITES;
  if (reduce_small_sites) {
    const uint site_grad_base = local_frame * WF2_ENDPOINT_EDIT_FRAMEGROUP_REDUCE_SITES;
    for (uint site_slot = 0u; site_slot < WF2_ENDPOINT_EDIT_FRAMEGROUP_REDUCE_SITES; ++site_slot) {
      tg_site_grad[site_grad_base + site_slot] = float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
  }
  tg_loss[local_frame] = 0.0f;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const bool active = track_id < track_count && local_frame < frames_in_chunk && tg_valid[local_frame] != 0;
  if (active) {
    const uint global_frame_id = frame_start + local_frame;
    const uint sample_id = track_id * frame_count + global_frame_id;
    const uint total_samples = track_count * frame_count;
    const float near_depth = config_f32[0];
    const float far_depth = config_f32[1];
    const float invalid_epsilon = config_f32[2];
    const float transmittance_threshold = config_f32[3];
    const float t = frame_t_f32[global_frame_id];
    const bool use_change = tg_source[local_frame] != 0;
    const int begin_raw = tg_begin[local_frame];
    const int end_raw = tg_end[local_frame];
    const bool valid_row_bounds = end_raw >= begin_raw;
    const uint row_count = valid_row_bounds ? uint(end_raw - begin_raw) : 0u;
    const uint out_base = sample_id * 3u;
    const float3 target_rgb = float3(
        target_rgb_f32[out_base + 0u],
        target_rgb_f32[out_base + 1u],
        target_rgb_f32[out_base + 2u]);
    const float inv_element_count = 1.0f / float(total_samples * 3u);
    if (valid_row_bounds && row_count == 0u) {
      tg_loss[local_frame] = dot(target_rgb, target_rgb) * inv_element_count;
    } else if (valid_row_bounds) {
      const uint track_base = track_id * 12u;
      const float3 ray_direction = float3(
          track_ray_coeff_f32[track_base + 6u] + t * track_ray_coeff_f32[track_base + 9u],
          track_ray_coeff_f32[track_base + 7u] + t * track_ray_coeff_f32[track_base + 10u],
          track_ray_coeff_f32[track_base + 8u] + t * track_ray_coeff_f32[track_base + 11u]);
      const float fiber_speed = length(ray_direction);
      float3 total_rgb = float3(0.0f, 0.0f, 0.0f);
      float total_transmittance = 1.0f;
      uint processed_run_count = 0u;
      int cached_right_cut = -2147483648;
      float cached_right_depth = 0.0f;
      bool cached_right_valid = false;
      for (uint cursor = 0u; cursor < row_count; ++cursor) {
        if (total_transmittance <= transmittance_threshold ||
            processed_run_count >= WF2_MAX_REALRAY_SEGMENTS) {
          break;
        }
        const uint record_base = uint(begin_raw) + cursor;
        int owner_raw = 0;
        int left_cut = 0;
        int right_cut = 0;
        wf2_endpoint_record_unpack_record(
            use_change ? change_record_i32[record_base] : base_record_i32[record_base],
            owner_raw,
            left_cut,
            right_cut);
        if (owner_raw < 0 || uint(owner_raw) >= site_count) {
          continue;
        }
        float start_depth = 0.0f;
        float end_depth = 0.0f;
        float3 start_grad_normal = float3(0.0f, 0.0f, 0.0f);
        float start_grad_time_normal = 0.0f;
        float start_grad_bias = 0.0f;
        bool start_valid = false;
        if (cached_right_valid && left_cut == cached_right_cut) {
          start_depth = cached_right_depth;
          start_valid = true;
        } else {
          start_valid = wf2_endpoint_record_factorized_cut_depth_boundary_jacobian(
              boundary_f32,
              track_ray_coeff_f32,
              boundary_count,
              track_id,
              left_cut,
              t,
              near_depth,
              far_depth,
              invalid_epsilon,
              start_depth,
              start_grad_normal,
              start_grad_time_normal,
              start_grad_bias);
        }
        float3 end_grad_normal = float3(0.0f, 0.0f, 0.0f);
        float end_grad_time_normal = 0.0f;
        float end_grad_bias = 0.0f;
        if (!start_valid ||
            !wf2_endpoint_record_factorized_cut_depth_boundary_jacobian(
                boundary_f32,
                track_ray_coeff_f32,
                boundary_count,
                track_id,
                right_cut,
                t,
                near_depth,
                far_depth,
                invalid_epsilon,
                end_depth,
                end_grad_normal,
                end_grad_time_normal,
                end_grad_bias)) {
          cached_right_valid = false;
          continue;
        }
        cached_right_cut = right_cut;
        cached_right_depth = end_depth;
        cached_right_valid = true;
        const float depth_length = end_depth - start_depth;
        const float physical_length = fiber_speed * depth_length;
        if (!(physical_length > 1.0e-8f) ||
            !(fiber_speed > 0.0f) ||
            !isfinite(fiber_speed) ||
            !isfinite(physical_length)) {
          continue;
        }
        const uint owner = uint(owner_raw);
        const uint rgba_base = owner * 4u;
        const float density = max(site_rgba_f32[rgba_base + 3u], 0.0f);
        const float segment_transmittance = exp(-density * physical_length);
        const float segment_alpha = 1.0f - segment_transmittance;
        const float weight = total_transmittance * segment_alpha;
        const float3 rgb = float3(
            site_rgba_f32[rgba_base + 0u],
            site_rgba_f32[rgba_base + 1u],
            site_rgba_f32[rgba_base + 2u]);
        total_rgb += weight * rgb;
        total_transmittance *= segment_transmittance;
        processed_run_count += 1u;
      }

      const float3 diff = total_rgb - target_rgb;
      tg_loss[local_frame] = dot(diff, diff) * inv_element_count;
      const float3 grad_rgb = (2.0f * inv_element_count) * diff;

      float prefix_transmittance = 1.0f;
      float3 prefix_rgb = float3(0.0f, 0.0f, 0.0f);
      uint replayed_run_count = 0u;
      cached_right_cut = -2147483648;
      cached_right_depth = 0.0f;
      float3 cached_right_grad_normal = float3(0.0f, 0.0f, 0.0f);
      float cached_right_grad_time_normal = 0.0f;
      float cached_right_grad_bias = 0.0f;
      cached_right_valid = false;
      for (uint cursor = 0u; cursor < row_count && replayed_run_count < processed_run_count; ++cursor) {
        const uint record_base = uint(begin_raw) + cursor;
        int owner_raw = 0;
        int left_cut = 0;
        int right_cut = 0;
        wf2_endpoint_record_unpack_record(
            use_change ? change_record_i32[record_base] : base_record_i32[record_base],
            owner_raw,
            left_cut,
            right_cut);
        if (owner_raw < 0 || uint(owner_raw) >= site_count) {
          continue;
        }

        float start_depth = 0.0f;
        float3 start_grad_normal = float3(0.0f, 0.0f, 0.0f);
        float start_grad_time_normal = 0.0f;
        float start_grad_bias = 0.0f;
        bool start_valid = false;
        if (cached_right_valid && left_cut == cached_right_cut) {
          start_depth = cached_right_depth;
          start_grad_normal = cached_right_grad_normal;
          start_grad_time_normal = cached_right_grad_time_normal;
          start_grad_bias = cached_right_grad_bias;
          start_valid = true;
        } else {
          start_valid = wf2_endpoint_record_factorized_cut_depth_boundary_jacobian(
              boundary_f32,
              track_ray_coeff_f32,
              boundary_count,
              track_id,
              left_cut,
              t,
              near_depth,
              far_depth,
              invalid_epsilon,
              start_depth,
              start_grad_normal,
              start_grad_time_normal,
              start_grad_bias);
        }

        float end_depth = 0.0f;
        float3 end_grad_normal = float3(0.0f, 0.0f, 0.0f);
        float end_grad_time_normal = 0.0f;
        float end_grad_bias = 0.0f;
        if (!start_valid ||
            !wf2_endpoint_record_factorized_cut_depth_boundary_jacobian(
                boundary_f32,
                track_ray_coeff_f32,
                boundary_count,
                track_id,
                right_cut,
                t,
                near_depth,
                far_depth,
                invalid_epsilon,
                end_depth,
                end_grad_normal,
                end_grad_time_normal,
                end_grad_bias)) {
          cached_right_valid = false;
          continue;
        }
        cached_right_cut = right_cut;
        cached_right_depth = end_depth;
        cached_right_grad_normal = end_grad_normal;
        cached_right_grad_time_normal = end_grad_time_normal;
        cached_right_grad_bias = end_grad_bias;
        cached_right_valid = true;

        const float depth_length = end_depth - start_depth;
        const float physical_length = fiber_speed * depth_length;
        if (!(physical_length > 1.0e-8f) ||
            !(fiber_speed > 0.0f) ||
            !isfinite(fiber_speed) ||
            !isfinite(physical_length)) {
          continue;
        }
        const uint owner = uint(owner_raw);
        const uint rgba_base = owner * 4u;
        const float raw_density = site_rgba_f32[rgba_base + 3u];
        const float density = max(raw_density, 0.0f);
        const float segment_transmittance = exp(-density * physical_length);
        const float segment_alpha = 1.0f - segment_transmittance;
        const float3 rgb = float3(
            site_rgba_f32[rgba_base + 0u],
            site_rgba_f32[rgba_base + 1u],
            site_rgba_f32[rgba_base + 2u]);
        const float weight = prefix_transmittance * segment_alpha;
        const float tau_bar = dot(
            grad_rgb,
            prefix_rgb + prefix_transmittance * rgb - total_rgb);
        const float endpoint_depth_bar = fiber_speed * density * tau_bar;
        float4 grad_rgba = float4(
            weight * grad_rgb.x,
            weight * grad_rgb.y,
            weight * grad_rgb.z,
            0.0f);
        if (raw_density > 0.0f) {
          grad_rgba.w = physical_length * tau_bar;
        }
        if (reduce_small_sites) {
          tg_site_grad[local_frame * WF2_ENDPOINT_EDIT_FRAMEGROUP_REDUCE_SITES + owner] += grad_rgba;
        } else {
          wf2_atomic_add4(grad_site_rgba_f32, rgba_base, grad_rgba);
        }

        if (left_cut >= 0) {
          wf2_atomic_add5(
              grad_boundary_f32,
              uint(left_cut) * 5u,
              -endpoint_depth_bar * start_grad_normal,
              -endpoint_depth_bar * start_grad_time_normal,
              -endpoint_depth_bar * start_grad_bias);
        }
        if (right_cut >= 0) {
          wf2_atomic_add5(
              grad_boundary_f32,
              uint(right_cut) * 5u,
              endpoint_depth_bar * end_grad_normal,
              endpoint_depth_bar * end_grad_time_normal,
              endpoint_depth_bar * end_grad_bias);
        }

        prefix_rgb += weight * rgb;
        prefix_transmittance *= segment_transmittance;
        replayed_run_count += 1u;
      }
    }
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (local_frame == 0u) {
    float loss_sum = 0.0f;
    for (uint frame_id = 0u; frame_id < frames_in_chunk; ++frame_id) {
      loss_sum += tg_loss[frame_id];
    }
    atomic_fetch_add_explicit(&loss_f32[0], loss_sum, memory_order_relaxed);
  }

  if (reduce_small_sites && local_frame < site_count) {
    float4 grad_sum = float4(0.0f, 0.0f, 0.0f, 0.0f);
    for (uint frame_id = 0u; frame_id < frames_in_chunk; ++frame_id) {
      grad_sum += tg_site_grad[
          frame_id * WF2_ENDPOINT_EDIT_FRAMEGROUP_REDUCE_SITES + local_frame];
    }
    wf2_atomic_add4(grad_site_rgba_f32, local_frame * 4u, grad_sum);
  }
}

kernel void wf2_endpoint_record_delta_replace_factorized_packed_framegroup16_constant_state_p0_mse_vjp_sparse_mobius_rgb_tensor(
    device const float* mobius_coeff_f32 [[buffer(0)]],
    device const float* track_ray_coeff_f32 [[buffer(1)]],
    device const float* frame_t_f32 [[buffer(2)]],
    device const short* base_offsets_i16 [[buffer(3)]],
    device const int* base_record_incidence_i32 [[buffer(4)]],
    device const short* track_change_offsets_i16 [[buffer(5)]],
    device const short* track_chunk_change_offsets_i16 [[buffer(6)]],
    device const short* change_frame_i16 [[buffer(7)]],
    device const short* change_offsets_i16 [[buffer(8)]],
    device const int* change_record_incidence_i32 [[buffer(9)]],
    device const int* track_incidence_offsets_i32 [[buffer(10)]],
    device const float* site_rgba_f32 [[buffer(11)]],
    device const float* target_rgb_f32 [[buffer(12)]],
    device const int* config_i32 [[buffer(13)]],
    device const float* config_f32 [[buffer(14)]],
    device atomic_float* loss_f32 [[buffer(15)]],
    device atomic_float* grad_site_rgba_f32 [[buffer(16)]],
    device atomic_float* grad_mobius_coeff_f32 [[buffer(17)]],
    uint3 local_pos [[thread_position_in_threadgroup]],
    uint3 group_id [[threadgroup_position_in_grid]]) {
  const uint local_frame = local_pos.x;
  const uint track_count = uint(config_i32[1]);
  const uint frame_count = uint(config_i32[2]);
  const uint site_count = uint(config_i32[3]);
  const uint base_record_count = uint(config_i32[4]);
  const uint change_count = uint(config_i32[5]);
  const uint change_record_count = uint(config_i32[6]);
  const uint incidence_count = uint(config_i32[7]);
  const uint chunk_count = (frame_count + WF2_ENDPOINT_DELTA_FRAMEGROUP_MAX_FRAMES - 1u) /
      WF2_ENDPOINT_DELTA_FRAMEGROUP_MAX_FRAMES;
  const uint group_index = group_id.x;
  const uint track_id = group_index / max(chunk_count, 1u);
  const uint chunk_id = group_index - track_id * max(chunk_count, 1u);
  const uint frame_start = chunk_id * WF2_ENDPOINT_DELTA_FRAMEGROUP_MAX_FRAMES;
  const uint frames_in_chunk = frame_start < frame_count
      ? min(WF2_ENDPOINT_DELTA_FRAMEGROUP_MAX_FRAMES, frame_count - frame_start)
      : 0u;

  threadgroup int tg_valid[WF2_ENDPOINT_DELTA_FRAMEGROUP_MAX_FRAMES];
  threadgroup int tg_source[WF2_ENDPOINT_DELTA_FRAMEGROUP_MAX_FRAMES];
  threadgroup int tg_begin[WF2_ENDPOINT_DELTA_FRAMEGROUP_MAX_FRAMES];
  threadgroup int tg_end[WF2_ENDPOINT_DELTA_FRAMEGROUP_MAX_FRAMES];
  threadgroup float tg_loss[WF2_ENDPOINT_DELTA_FRAMEGROUP_MAX_FRAMES];
  threadgroup float4 tg_site_grad[
      WF2_ENDPOINT_DELTA_FRAMEGROUP_MAX_FRAMES * WF2_ENDPOINT_EDIT_FRAMEGROUP_REDUCE_SITES];

  if (local_frame == 0u) {
    for (uint frame_id = 0u; frame_id < WF2_ENDPOINT_DELTA_FRAMEGROUP_MAX_FRAMES; ++frame_id) {
      tg_valid[frame_id] = 0;
      tg_source[frame_id] = 0;
      tg_begin[frame_id] = 0;
      tg_end[frame_id] = 0;
    }

    bool ok = track_id < track_count && frames_in_chunk > 0u;
    int change_begin_raw = 0;
    int change_end_raw = 0;
    int chunk_change_begin_raw = 0;
    int chunk_change_end_raw = 0;
    if (ok) {
      change_begin_raw = track_change_offsets_i16[track_id];
      change_end_raw = track_change_offsets_i16[track_id + 1u];
      const uint chunk_offset_base = track_id * (chunk_count + 1u) + chunk_id;
      chunk_change_begin_raw = int(track_chunk_change_offsets_i16[chunk_offset_base]);
      chunk_change_end_raw = int(track_chunk_change_offsets_i16[chunk_offset_base + 1u]);
      ok = change_begin_raw >= 0 && change_end_raw >= change_begin_raw && uint(change_end_raw) <= change_count &&
          chunk_change_begin_raw >= change_begin_raw && chunk_change_end_raw >= chunk_change_begin_raw &&
          chunk_change_end_raw <= change_end_raw;
    }
    int selected_change = -1;
    if (ok && chunk_change_begin_raw > change_begin_raw) {
      selected_change = chunk_change_begin_raw - 1;
      while (selected_change >= change_begin_raw && change_frame_i16[uint(selected_change)] < 0) {
        selected_change -= 1;
      }
      if (selected_change < change_begin_raw) {
        selected_change = -1;
      }
    }

    uint change_cursor = ok ? uint(chunk_change_begin_raw) : 0u;
    const uint change_end = ok ? uint(chunk_change_end_raw) : 0u;
    for (uint local_frame_id = 0u; local_frame_id < frames_in_chunk; ++local_frame_id) {
      const uint global_frame_id = frame_start + local_frame_id;
      while (ok && change_cursor < change_end) {
        const int changed_frame = change_frame_i16[change_cursor];
        if (changed_frame < 0) {
          change_cursor += 1u;
          continue;
        }
        if (uint(changed_frame) > global_frame_id) {
          break;
        }
        selected_change = int(change_cursor);
        change_cursor += 1u;
      }

      if (ok) {
        const bool use_change = selected_change >= 0;
        const int begin_raw =
            use_change ? change_offsets_i16[uint(selected_change)] : base_offsets_i16[track_id];
        const int end_raw =
            use_change ? change_offsets_i16[uint(selected_change) + 1u] : base_offsets_i16[track_id + 1u];
        const uint record_count = use_change ? change_record_count : base_record_count;
        ok = begin_raw >= 0 && end_raw >= begin_raw && uint(end_raw) <= record_count &&
            uint(end_raw - begin_raw) <= WF2_MAX_REALRAY_SEGMENTS;
        if (!ok) {
          break;
        }
        tg_source[local_frame_id] = use_change ? 1 : 0;
        tg_begin[local_frame_id] = begin_raw;
        tg_end[local_frame_id] = end_raw;
        tg_valid[local_frame_id] = 1;
      }
    }
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  const bool reduce_small_sites = site_count <= WF2_ENDPOINT_EDIT_FRAMEGROUP_REDUCE_SITES;
  if (reduce_small_sites) {
    const uint site_grad_base = local_frame * WF2_ENDPOINT_EDIT_FRAMEGROUP_REDUCE_SITES;
    for (uint site_slot = 0u; site_slot < WF2_ENDPOINT_EDIT_FRAMEGROUP_REDUCE_SITES; ++site_slot) {
      tg_site_grad[site_grad_base + site_slot] = float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
  }
  tg_loss[local_frame] = 0.0f;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const bool active = track_id < track_count && local_frame < frames_in_chunk && tg_valid[local_frame] != 0;
  if (active) {
    const uint global_frame_id = frame_start + local_frame;
    const uint sample_id = track_id * frame_count + global_frame_id;
    const uint total_samples = track_count * frame_count;
    const float near_depth = config_f32[0];
    const float far_depth = config_f32[1];
    const float invalid_epsilon = config_f32[2];
    const float transmittance_threshold = config_f32[3];
    const float t = frame_t_f32[global_frame_id];
    const bool use_change = tg_source[local_frame] != 0;
    const int begin_raw = tg_begin[local_frame];
    const int end_raw = tg_end[local_frame];
    const bool valid_row_bounds = end_raw >= begin_raw;
    const uint row_count = valid_row_bounds ? uint(end_raw - begin_raw) : 0u;
    const uint out_base = sample_id * 3u;
    const float3 target_rgb = float3(
        target_rgb_f32[out_base + 0u],
        target_rgb_f32[out_base + 1u],
        target_rgb_f32[out_base + 2u]);
    const float inv_element_count = 1.0f / float(total_samples * 3u);
    if (valid_row_bounds && row_count == 0u) {
      tg_loss[local_frame] = dot(target_rgb, target_rgb) * inv_element_count;
    } else if (valid_row_bounds) {
      const uint track_base = track_id * 12u;
      const float3 ray_direction = float3(
          track_ray_coeff_f32[track_base + 6u] + t * track_ray_coeff_f32[track_base + 9u],
          track_ray_coeff_f32[track_base + 7u] + t * track_ray_coeff_f32[track_base + 10u],
          track_ray_coeff_f32[track_base + 8u] + t * track_ray_coeff_f32[track_base + 11u]);
      const float fiber_speed = length(ray_direction);
      float3 total_rgb = float3(0.0f, 0.0f, 0.0f);
      float total_transmittance = 1.0f;
      uint processed_run_count = 0u;
      int cached_right_cut = -2147483648;
      float cached_right_depth = 0.0f;
      bool cached_right_valid = false;
      for (uint cursor = 0u; cursor < row_count; ++cursor) {
        if (total_transmittance <= transmittance_threshold ||
            processed_run_count >= WF2_MAX_REALRAY_SEGMENTS) {
          break;
        }
        const uint record_base = uint(begin_raw) + cursor;
        int owner_raw = 0;
        int left_cut = 0;
        int right_cut = 0;
        wf2_endpoint_record_unpack_record(
            use_change ? change_record_incidence_i32[record_base] : base_record_incidence_i32[record_base],
            owner_raw,
            left_cut,
            right_cut);
        if (owner_raw < 0 || uint(owner_raw) >= site_count) {
          continue;
        }
        float start_depth = 0.0f;
        uint start_incidence_id = 0xFFFFFFFFu;
        float4 start_grad_mobius = float4(0.0f, 0.0f, 0.0f, 0.0f);
        bool start_valid = false;
        if (cached_right_valid && left_cut == cached_right_cut) {
          start_depth = cached_right_depth;
          start_valid = true;
        } else {
          start_valid = wf2_endpoint_record_sparse_mobius_cut_depth_jacobian(
              mobius_coeff_f32,
              track_incidence_offsets_i32,
              incidence_count,
              track_id,
              left_cut,
              t,
              near_depth,
              far_depth,
              invalid_epsilon,
              start_incidence_id,
              start_depth,
              start_grad_mobius);
        }
        float end_depth = 0.0f;
        uint end_incidence_id = 0xFFFFFFFFu;
        float4 end_grad_mobius = float4(0.0f, 0.0f, 0.0f, 0.0f);
        if (!start_valid ||
            !wf2_endpoint_record_sparse_mobius_cut_depth_jacobian(
                mobius_coeff_f32,
                track_incidence_offsets_i32,
                incidence_count,
                track_id,
                right_cut,
                t,
                near_depth,
                far_depth,
                invalid_epsilon,
                end_incidence_id,
                end_depth,
                end_grad_mobius)) {
          cached_right_valid = false;
          continue;
        }
        cached_right_cut = right_cut;
        cached_right_depth = end_depth;
        cached_right_valid = true;
        const float depth_length = end_depth - start_depth;
        const float physical_length = fiber_speed * depth_length;
        if (!(physical_length > 1.0e-8f) ||
            !(fiber_speed > 0.0f) ||
            !isfinite(fiber_speed) ||
            !isfinite(physical_length)) {
          continue;
        }
        const uint owner = uint(owner_raw);
        const uint rgba_base = owner * 4u;
        const float density = max(site_rgba_f32[rgba_base + 3u], 0.0f);
        const float segment_transmittance = exp(-density * physical_length);
        const float segment_alpha = 1.0f - segment_transmittance;
        const float weight = total_transmittance * segment_alpha;
        const float3 rgb = float3(
            site_rgba_f32[rgba_base + 0u],
            site_rgba_f32[rgba_base + 1u],
            site_rgba_f32[rgba_base + 2u]);
        total_rgb += weight * rgb;
        total_transmittance *= segment_transmittance;
        processed_run_count += 1u;
      }

      const float3 diff = total_rgb - target_rgb;
      tg_loss[local_frame] = dot(diff, diff) * inv_element_count;
      const float3 grad_rgb = (2.0f * inv_element_count) * diff;

      float prefix_transmittance = 1.0f;
      float3 prefix_rgb = float3(0.0f, 0.0f, 0.0f);
      uint replayed_run_count = 0u;
      cached_right_cut = -2147483648;
      cached_right_depth = 0.0f;
      uint cached_right_incidence_id = 0xFFFFFFFFu;
      float4 cached_right_grad_mobius = float4(0.0f, 0.0f, 0.0f, 0.0f);
      cached_right_valid = false;
      for (uint cursor = 0u; cursor < row_count && replayed_run_count < processed_run_count; ++cursor) {
        const uint record_base = uint(begin_raw) + cursor;
        int owner_raw = 0;
        int left_cut = 0;
        int right_cut = 0;
        wf2_endpoint_record_unpack_record(
            use_change ? change_record_incidence_i32[record_base] : base_record_incidence_i32[record_base],
            owner_raw,
            left_cut,
            right_cut);
        if (owner_raw < 0 || uint(owner_raw) >= site_count) {
          continue;
        }

        float start_depth = 0.0f;
        uint start_incidence_id = 0xFFFFFFFFu;
        float4 start_grad_mobius = float4(0.0f, 0.0f, 0.0f, 0.0f);
        bool start_valid = false;
        if (cached_right_valid && left_cut == cached_right_cut) {
          start_depth = cached_right_depth;
          start_incidence_id = cached_right_incidence_id;
          start_grad_mobius = cached_right_grad_mobius;
          start_valid = true;
        } else {
          start_valid = wf2_endpoint_record_sparse_mobius_cut_depth_jacobian(
              mobius_coeff_f32,
              track_incidence_offsets_i32,
              incidence_count,
              track_id,
              left_cut,
              t,
              near_depth,
              far_depth,
              invalid_epsilon,
              start_incidence_id,
              start_depth,
              start_grad_mobius);
        }

        float end_depth = 0.0f;
        uint end_incidence_id = 0xFFFFFFFFu;
        float4 end_grad_mobius = float4(0.0f, 0.0f, 0.0f, 0.0f);
        if (!start_valid ||
            !wf2_endpoint_record_sparse_mobius_cut_depth_jacobian(
                mobius_coeff_f32,
                track_incidence_offsets_i32,
                incidence_count,
                track_id,
                right_cut,
                t,
                near_depth,
                far_depth,
                invalid_epsilon,
                end_incidence_id,
                end_depth,
                end_grad_mobius)) {
          cached_right_valid = false;
          continue;
        }
        cached_right_cut = right_cut;
        cached_right_depth = end_depth;
        cached_right_incidence_id = end_incidence_id;
        cached_right_grad_mobius = end_grad_mobius;
        cached_right_valid = true;

        const float depth_length = end_depth - start_depth;
        const float physical_length = fiber_speed * depth_length;
        if (!(physical_length > 1.0e-8f) ||
            !(fiber_speed > 0.0f) ||
            !isfinite(fiber_speed) ||
            !isfinite(physical_length)) {
          continue;
        }
        const uint owner = uint(owner_raw);
        const uint rgba_base = owner * 4u;
        const float raw_density = site_rgba_f32[rgba_base + 3u];
        const float density = max(raw_density, 0.0f);
        const float segment_transmittance = exp(-density * physical_length);
        const float segment_alpha = 1.0f - segment_transmittance;
        const float3 rgb = float3(
            site_rgba_f32[rgba_base + 0u],
            site_rgba_f32[rgba_base + 1u],
            site_rgba_f32[rgba_base + 2u]);
        const float weight = prefix_transmittance * segment_alpha;
        const float tau_bar = dot(
            grad_rgb,
            prefix_rgb + prefix_transmittance * rgb - total_rgb);
        const float endpoint_depth_bar = fiber_speed * density * tau_bar;
        float4 grad_rgba = float4(
            weight * grad_rgb.x,
            weight * grad_rgb.y,
            weight * grad_rgb.z,
            0.0f);
        if (raw_density > 0.0f) {
          grad_rgba.w = physical_length * tau_bar;
        }
        if (reduce_small_sites) {
          tg_site_grad[local_frame * WF2_ENDPOINT_EDIT_FRAMEGROUP_REDUCE_SITES + owner] += grad_rgba;
        } else {
          wf2_atomic_add4(grad_site_rgba_f32, rgba_base, grad_rgba);
        }

        if (start_incidence_id != 0xFFFFFFFFu) {
          wf2_atomic_add4(
              grad_mobius_coeff_f32,
              start_incidence_id * 4u,
              -endpoint_depth_bar * start_grad_mobius);
        }
        if (end_incidence_id != 0xFFFFFFFFu) {
          wf2_atomic_add4(
              grad_mobius_coeff_f32,
              end_incidence_id * 4u,
              endpoint_depth_bar * end_grad_mobius);
        }

        prefix_rgb += weight * rgb;
        prefix_transmittance *= segment_transmittance;
        replayed_run_count += 1u;
      }
    }
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (local_frame == 0u) {
    float loss_sum = 0.0f;
    for (uint frame_id = 0u; frame_id < frames_in_chunk; ++frame_id) {
      loss_sum += tg_loss[frame_id];
    }
    atomic_fetch_add_explicit(&loss_f32[0], loss_sum, memory_order_relaxed);
  }

  if (reduce_small_sites && local_frame < site_count) {
    float4 grad_sum = float4(0.0f, 0.0f, 0.0f, 0.0f);
    for (uint frame_id = 0u; frame_id < frames_in_chunk; ++frame_id) {
      grad_sum += tg_site_grad[
          frame_id * WF2_ENDPOINT_EDIT_FRAMEGROUP_REDUCE_SITES + local_frame];
    }
    wf2_atomic_add4(grad_site_rgba_f32, local_frame * 4u, grad_sum);
  }
}

kernel void wf2_endpoint_record_delta_replace_factorized_frameselect_recompute_mse_vjp_direct_atomic_rgb_only_tensor(
    device const float* boundary_f32 [[buffer(0)]],
    device const float* track_ray_coeff_f32 [[buffer(1)]],
    device const float* frame_t_f32 [[buffer(2)]],
    device const short* base_offsets_i32 [[buffer(3)]],
    device const int* base_record_i32 [[buffer(4)]],
    device const short* frame_change_index_i16 [[buffer(5)]],
    device const short* change_offsets_i32 [[buffer(6)]],
    device const int* change_record_i32 [[buffer(7)]],
    device const float* site_rgba_f32 [[buffer(8)]],
    device const float* target_rgb_f32 [[buffer(9)]],
    device const int* config_i32 [[buffer(10)]],
    device const float* config_f32 [[buffer(11)]],
    device atomic_float* loss_f32 [[buffer(12)]],
    device atomic_float* grad_site_rgba_f32 [[buffer(13)]],
    uint3 local_pos [[thread_position_in_threadgroup]],
    uint3 group_id [[threadgroup_position_in_grid]]) {
  const uint local_frame = local_pos.x;
  const uint boundary_count = uint(config_i32[0]);
  const uint track_count = uint(config_i32[1]);
  const uint frame_count = uint(config_i32[2]);
  const uint site_count = uint(config_i32[3]);
  const uint base_record_count = uint(config_i32[4]);
  const uint change_count = uint(config_i32[5]);
  const uint change_record_count = uint(config_i32[6]);
  const uint chunk_count = (frame_count + WF2_ENDPOINT_DELTA_FRAMEGROUP_MAX_FRAMES - 1u) /
      WF2_ENDPOINT_DELTA_FRAMEGROUP_MAX_FRAMES;
  const uint group_index = group_id.x;
  const uint track_id = group_index / max(chunk_count, 1u);
  const uint chunk_id = group_index - track_id * max(chunk_count, 1u);
  const uint frame_start = chunk_id * WF2_ENDPOINT_DELTA_FRAMEGROUP_MAX_FRAMES;
  const uint frames_in_chunk = frame_start < frame_count
      ? min(WF2_ENDPOINT_DELTA_FRAMEGROUP_MAX_FRAMES, frame_count - frame_start)
      : 0u;

  threadgroup int tg_valid[WF2_ENDPOINT_DELTA_FRAMEGROUP_MAX_FRAMES];
  threadgroup int tg_source[WF2_ENDPOINT_DELTA_FRAMEGROUP_MAX_FRAMES];
  threadgroup int tg_begin[WF2_ENDPOINT_DELTA_FRAMEGROUP_MAX_FRAMES];
  threadgroup int tg_end[WF2_ENDPOINT_DELTA_FRAMEGROUP_MAX_FRAMES];
  threadgroup float tg_loss[WF2_ENDPOINT_DELTA_FRAMEGROUP_MAX_FRAMES];
  threadgroup float4 tg_site_grad[
      WF2_ENDPOINT_DELTA_FRAMEGROUP_MAX_FRAMES * WF2_ENDPOINT_EDIT_FRAMEGROUP_REDUCE_SITES];

  if (local_frame == 0u) {
    for (uint frame_id = 0u; frame_id < WF2_ENDPOINT_DELTA_FRAMEGROUP_MAX_FRAMES; ++frame_id) {
      tg_valid[frame_id] = 0;
      tg_source[frame_id] = 0;
      tg_begin[frame_id] = 0;
      tg_end[frame_id] = 0;
    }

    bool ok = track_id < track_count && frames_in_chunk > 0u;
    for (uint local_frame_id = 0u; local_frame_id < frames_in_chunk; ++local_frame_id) {
      const uint global_frame_id = frame_start + local_frame_id;
      bool use_change = false;
      int selected_change = -1;
      if (ok && global_frame_id > 0u) {
        const uint select_index = track_id * (frame_count - 1u) + (global_frame_id - 1u);
        selected_change = int(frame_change_index_i16[select_index]);
        ok = selected_change >= -1 && (selected_change < 0 || uint(selected_change) < change_count);
        use_change = selected_change >= 0;
      }

      if (ok) {
        const int begin_raw =
            use_change ? change_offsets_i32[uint(selected_change)] : base_offsets_i32[track_id];
        const int end_raw =
            use_change ? change_offsets_i32[uint(selected_change) + 1u] : base_offsets_i32[track_id + 1u];
        const uint record_count = use_change ? change_record_count : base_record_count;
        ok = begin_raw >= 0 && end_raw >= begin_raw && uint(end_raw) <= record_count &&
            uint(end_raw - begin_raw) <= WF2_MAX_REALRAY_SEGMENTS;
        if (!ok) {
          break;
        }
        tg_source[local_frame_id] = use_change ? 1 : 0;
        tg_begin[local_frame_id] = begin_raw;
        tg_end[local_frame_id] = end_raw;
        tg_valid[local_frame_id] = 1;
      }
    }
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  const bool reduce_small_sites = site_count <= WF2_ENDPOINT_EDIT_FRAMEGROUP_REDUCE_SITES;
  if (reduce_small_sites) {
    const uint site_grad_base = local_frame * WF2_ENDPOINT_EDIT_FRAMEGROUP_REDUCE_SITES;
    for (uint site_slot = 0u; site_slot < WF2_ENDPOINT_EDIT_FRAMEGROUP_REDUCE_SITES; ++site_slot) {
      tg_site_grad[site_grad_base + site_slot] = float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
  }
  tg_loss[local_frame] = 0.0f;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const bool active = track_id < track_count && local_frame < frames_in_chunk && tg_valid[local_frame] != 0;
  if (active) {

  const uint global_frame_id = frame_start + local_frame;
  const uint sample_id = track_id * frame_count + global_frame_id;
  const uint total_samples = track_count * frame_count;
  const float near_depth = config_f32[0];
  const float far_depth = config_f32[1];
  const float invalid_epsilon = config_f32[2];
  const float transmittance_threshold = config_f32[3];
  const float t = frame_t_f32[global_frame_id];
  const bool use_change = tg_source[local_frame] != 0;
  const int begin_raw = tg_begin[local_frame];
  const int end_raw = tg_end[local_frame];
  const bool valid_row_bounds = end_raw >= begin_raw;
  const uint row_count = valid_row_bounds ? uint(end_raw - begin_raw) : 0u;
  const uint out_base = sample_id * 3u;
  const float3 target_rgb = float3(
      target_rgb_f32[out_base + 0u],
      target_rgb_f32[out_base + 1u],
      target_rgb_f32[out_base + 2u]);
  const float inv_element_count = 1.0f / float(total_samples * 3u);
  if (valid_row_bounds && row_count == 0u) {
    tg_loss[local_frame] = dot(target_rgb, target_rgb) * inv_element_count;
  } else if (valid_row_bounds) {

  uint owners[WF2_MAX_REALRAY_SEGMENTS];
  float lengths[WF2_MAX_REALRAY_SEGMENTS];
  float trans_before[WF2_MAX_REALRAY_SEGMENTS];

  float3 rgb_accum = float3(0.0f, 0.0f, 0.0f);
  float transmittance = 1.0f;
  uint local_run_count = 0u;
  int cached_right_cut = -2147483648;
  float cached_right_depth = 0.0f;
  bool cached_right_valid = false;
  for (uint cursor = 0u; cursor < row_count; ++cursor) {
    if (transmittance <= transmittance_threshold || local_run_count >= WF2_MAX_REALRAY_SEGMENTS) {
      break;
    }
    const uint record_base = uint(begin_raw) + cursor;
    int owner_raw = 0;
    int left_cut = 0;
    int right_cut = 0;
    wf2_endpoint_record_unpack_record(
        use_change ? change_record_i32[record_base] : base_record_i32[record_base],
        owner_raw,
        left_cut,
        right_cut);
    if (owner_raw < 0 || uint(owner_raw) >= site_count) {
      continue;
    }
    float start_depth = 0.0f;
    float end_depth = 0.0f;
    bool start_valid = false;
    if (cached_right_valid && left_cut == cached_right_cut) {
      start_depth = cached_right_depth;
      start_valid = true;
    } else {
      start_valid = wf2_endpoint_record_factorized_cut_depth(
          boundary_f32,
          track_ray_coeff_f32,
          boundary_count,
          track_id,
          left_cut,
          t,
          near_depth,
          far_depth,
          invalid_epsilon,
          start_depth);
    }
    if (!start_valid ||
        !wf2_endpoint_record_factorized_cut_depth(
            boundary_f32,
            track_ray_coeff_f32,
            boundary_count,
            track_id,
            right_cut,
            t,
            near_depth,
            far_depth,
            invalid_epsilon,
            end_depth)) {
      cached_right_valid = false;
      continue;
    }
    cached_right_cut = right_cut;
    cached_right_depth = end_depth;
    cached_right_valid = true;
    const float length = end_depth - start_depth;
    if (!(length > 1.0e-8f)) {
      continue;
    }
    const uint owner = uint(owner_raw);
    const uint rgba_base = owner * 4u;
    const float density = max(site_rgba_f32[rgba_base + 3u], 0.0f);
    const float seg_trans = exp(-density * length);
    const float seg_alpha = 1.0f - seg_trans;
    const float weight = transmittance * seg_alpha;
    const float3 rgb = float3(
        site_rgba_f32[rgba_base + 0u],
        site_rgba_f32[rgba_base + 1u],
        site_rgba_f32[rgba_base + 2u]);

    owners[local_run_count] = owner;
    lengths[local_run_count] = length;
    trans_before[local_run_count] = transmittance;
    local_run_count += 1u;

    rgb_accum += weight * rgb;
    transmittance *= seg_trans;
  }

  const float3 diff = rgb_accum - target_rgb;
  const float sample_loss = dot(diff, diff) * inv_element_count;
  tg_loss[local_frame] = sample_loss;
  const float3 grad_rgb = (2.0f * inv_element_count) * diff;

  float adj_next_transmittance = 0.0f;
  for (int run_id = int(local_run_count) - 1; run_id >= 0; --run_id) {
    const uint owner = owners[run_id];
    const uint rgba_base = owner * 4u;
    const float raw_density = site_rgba_f32[rgba_base + 3u];
    const float density = max(raw_density, 0.0f);
    const float length = lengths[run_id];
    const float seg_trans = exp(-density * length);
    const float seg_alpha = 1.0f - seg_trans;
    const float weight = trans_before[run_id] * seg_alpha;
    const float3 rgb = float3(
        site_rgba_f32[rgba_base + 0u],
        site_rgba_f32[rgba_base + 1u],
        site_rgba_f32[rgba_base + 2u]);
    const float adj_weight = dot(grad_rgb, rgb);
    const float adj_trans_before = adj_weight * seg_alpha + adj_next_transmittance * seg_trans;
    const float adj_segment_alpha = adj_weight * trans_before[run_id];
    const float adj_segment_trans = adj_next_transmittance * trans_before[run_id] - adj_segment_alpha;
    float4 grad_rgba = float4(
        weight * grad_rgb.x,
        weight * grad_rgb.y,
        weight * grad_rgb.z,
        0.0f);
    if (raw_density > 0.0f) {
      grad_rgba.w = adj_segment_trans * (-length * seg_trans);
    }
    if (reduce_small_sites) {
      tg_site_grad[local_frame * WF2_ENDPOINT_EDIT_FRAMEGROUP_REDUCE_SITES + owner] += grad_rgba;
    } else {
      wf2_atomic_add4(grad_site_rgba_f32, rgba_base, grad_rgba);
    }
    adj_next_transmittance = adj_trans_before;
  }
  }
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (local_frame == 0u) {
    float loss_sum = 0.0f;
    for (uint frame_id = 0u; frame_id < frames_in_chunk; ++frame_id) {
      loss_sum += tg_loss[frame_id];
    }
    atomic_fetch_add_explicit(&loss_f32[0], loss_sum, memory_order_relaxed);
  }

  if (reduce_small_sites) {
    if (local_frame < site_count) {
      float4 grad_sum = float4(0.0f, 0.0f, 0.0f, 0.0f);
      for (uint frame_id = 0u; frame_id < frames_in_chunk; ++frame_id) {
        grad_sum += tg_site_grad[
            frame_id * WF2_ENDPOINT_EDIT_FRAMEGROUP_REDUCE_SITES + local_frame];
      }
      wf2_atomic_add4(grad_site_rgba_f32, local_frame * 4u, grad_sum);
    }
  }
}

kernel void wf2_endpoint_record_delta_replace_factorized_framebitmask_recompute_mse_vjp_direct_atomic_rgb_only_tensor(
    device const float* boundary_f32 [[buffer(0)]],
    device const float* track_ray_coeff_f32 [[buffer(1)]],
    device const float* frame_t_f32 [[buffer(2)]],
    device const int* base_offsets_i32 [[buffer(3)]],
    device const int* base_record_i32 [[buffer(4)]],
    device const int* track_change_offsets_i32 [[buffer(5)]],
    device const int* track_frame_mask_i32 [[buffer(6)]],
    device const int* change_offsets_i32 [[buffer(7)]],
    device const int* change_record_i32 [[buffer(8)]],
    device const float* site_rgba_f32 [[buffer(9)]],
    device const float* target_rgb_f32 [[buffer(10)]],
    device const int* config_i32 [[buffer(11)]],
    device const float* config_f32 [[buffer(12)]],
    device atomic_float* loss_f32 [[buffer(13)]],
    device atomic_float* grad_site_rgba_f32 [[buffer(14)]],
    uint3 local_pos [[thread_position_in_threadgroup]],
    uint3 group_id [[threadgroup_position_in_grid]]) {
  const uint local_frame = local_pos.x;
  const uint boundary_count = uint(config_i32[0]);
  const uint track_count = uint(config_i32[1]);
  const uint frame_count = uint(config_i32[2]);
  const uint site_count = uint(config_i32[3]);
  const uint base_record_count = uint(config_i32[4]);
  const uint change_count = uint(config_i32[5]);
  const uint change_record_count = uint(config_i32[6]);
  const uint chunk_count = (frame_count + WF2_ENDPOINT_DELTA_FRAMEGROUP_MAX_FRAMES - 1u) /
      WF2_ENDPOINT_DELTA_FRAMEGROUP_MAX_FRAMES;
  const uint group_index = group_id.x;
  const uint track_id = group_index / max(chunk_count, 1u);
  const uint chunk_id = group_index - track_id * max(chunk_count, 1u);
  const uint frame_start = chunk_id * WF2_ENDPOINT_DELTA_FRAMEGROUP_MAX_FRAMES;
  const uint frames_in_chunk = frame_start < frame_count
      ? min(WF2_ENDPOINT_DELTA_FRAMEGROUP_MAX_FRAMES, frame_count - frame_start)
      : 0u;

  threadgroup int tg_valid[WF2_ENDPOINT_DELTA_FRAMEGROUP_MAX_FRAMES];
  threadgroup int tg_source[WF2_ENDPOINT_DELTA_FRAMEGROUP_MAX_FRAMES];
  threadgroup int tg_begin[WF2_ENDPOINT_DELTA_FRAMEGROUP_MAX_FRAMES];
  threadgroup int tg_end[WF2_ENDPOINT_DELTA_FRAMEGROUP_MAX_FRAMES];
  threadgroup uint tg_track_mask;
  threadgroup int tg_track_begin;
  threadgroup int tg_track_end;
  threadgroup int tg_track_ok;
  threadgroup float tg_loss[WF2_ENDPOINT_DELTA_FRAMEGROUP_MAX_FRAMES];
  threadgroup float4 tg_site_grad[
      WF2_ENDPOINT_DELTA_FRAMEGROUP_MAX_FRAMES * WF2_ENDPOINT_EDIT_FRAMEGROUP_REDUCE_SITES];

  if (local_frame == 0u) {
    bool ok = track_id < track_count && frames_in_chunk > 0u;
    const uint mask = ok ? uint(track_frame_mask_i32[track_id]) : 0u;
    const int track_begin = ok ? track_change_offsets_i32[track_id] : 0;
    const int track_end = ok ? track_change_offsets_i32[track_id + 1u] : 0;
    ok = ok && track_begin >= 0 && track_end >= track_begin && uint(track_end) <= change_count;
    tg_track_mask = mask;
    tg_track_begin = track_begin;
    tg_track_end = track_end;
    tg_track_ok = ok ? 1 : 0;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (local_frame < WF2_ENDPOINT_DELTA_FRAMEGROUP_MAX_FRAMES) {
    tg_valid[local_frame] = 0;
    tg_source[local_frame] = 0;
    tg_begin[local_frame] = 0;
    tg_end[local_frame] = 0;

    bool ok = tg_track_ok != 0 && local_frame < frames_in_chunk;
    const uint global_frame_id = frame_start + local_frame;
    bool use_change = false;
    int selected_change = -1;
    if (ok && global_frame_id > 0u) {
      const uint frame_bit = 1u << global_frame_id;
      use_change = (tg_track_mask & frame_bit) != 0u;
      if (use_change) {
        const uint lower_mask = tg_track_mask & (frame_bit - 1u);
        const int local_change_index = int(popcount(lower_mask));
        selected_change = tg_track_begin + local_change_index;
        ok = selected_change < tg_track_end && uint(selected_change) < change_count;
      }
    }

	    if (ok) {
      const int begin_raw =
          use_change ? change_offsets_i32[uint(selected_change)] : base_offsets_i32[track_id];
      const int end_raw =
          use_change ? change_offsets_i32[uint(selected_change) + 1u] : base_offsets_i32[track_id + 1u];
      const uint record_count = use_change ? change_record_count : base_record_count;
      ok = begin_raw >= 0 && end_raw >= begin_raw && uint(end_raw) <= record_count &&
          uint(end_raw - begin_raw) <= WF2_MAX_REALRAY_SEGMENTS;
      if (ok) {
        tg_source[local_frame] = use_change ? 1 : 0;
        tg_begin[local_frame] = begin_raw;
        tg_end[local_frame] = end_raw;
        tg_valid[local_frame] = 1;
      }
    }
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  const bool reduce_small_sites = site_count <= WF2_ENDPOINT_EDIT_FRAMEGROUP_REDUCE_SITES;
  if (reduce_small_sites) {
    const uint site_grad_base = local_frame * WF2_ENDPOINT_EDIT_FRAMEGROUP_REDUCE_SITES;
    for (uint site_slot = 0u; site_slot < WF2_ENDPOINT_EDIT_FRAMEGROUP_REDUCE_SITES; ++site_slot) {
      tg_site_grad[site_grad_base + site_slot] = float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
  }
  tg_loss[local_frame] = 0.0f;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const bool active = track_id < track_count && local_frame < frames_in_chunk && tg_valid[local_frame] != 0;
  if (active) {

  const uint global_frame_id = frame_start + local_frame;
  const uint sample_id = track_id * frame_count + global_frame_id;
  const uint total_samples = track_count * frame_count;
  const float near_depth = config_f32[0];
  const float far_depth = config_f32[1];
  const float invalid_epsilon = config_f32[2];
  const float transmittance_threshold = config_f32[3];
  const float t = frame_t_f32[global_frame_id];
  const bool use_change = tg_source[local_frame] != 0;
  const int begin_raw = tg_begin[local_frame];
  const int end_raw = tg_end[local_frame];
  const bool valid_row_bounds = end_raw >= begin_raw;
  const uint row_count = valid_row_bounds ? uint(end_raw - begin_raw) : 0u;
  const uint out_base = sample_id * 3u;
  const float3 target_rgb = float3(
      target_rgb_f32[out_base + 0u],
      target_rgb_f32[out_base + 1u],
      target_rgb_f32[out_base + 2u]);
  const float inv_element_count = 1.0f / float(total_samples * 3u);
  if (valid_row_bounds && row_count == 0u) {
    tg_loss[local_frame] = dot(target_rgb, target_rgb) * inv_element_count;
  } else if (valid_row_bounds) {

  uint owners[WF2_MAX_REALRAY_SEGMENTS];
  float lengths[WF2_MAX_REALRAY_SEGMENTS];
  float trans_before[WF2_MAX_REALRAY_SEGMENTS];

  float3 rgb_accum = float3(0.0f, 0.0f, 0.0f);
  float transmittance = 1.0f;
  uint local_run_count = 0u;
  int cached_right_cut = -2147483648;
  float cached_right_depth = 0.0f;
  bool cached_right_valid = false;
  for (uint cursor = 0u; cursor < row_count; ++cursor) {
    if (transmittance <= transmittance_threshold || local_run_count >= WF2_MAX_REALRAY_SEGMENTS) {
      break;
    }
    const uint record_base = uint(begin_raw) + cursor;
    int owner_raw = 0;
    int left_cut = 0;
    int right_cut = 0;
    wf2_endpoint_record_unpack_record(
        use_change ? change_record_i32[record_base] : base_record_i32[record_base],
        owner_raw,
        left_cut,
        right_cut);
    if (owner_raw < 0 || uint(owner_raw) >= site_count) {
      continue;
    }
    float start_depth = 0.0f;
    float end_depth = 0.0f;
    bool start_valid = false;
    if (cached_right_valid && left_cut == cached_right_cut) {
      start_depth = cached_right_depth;
      start_valid = true;
    } else {
      start_valid = wf2_endpoint_record_factorized_cut_depth(
          boundary_f32,
          track_ray_coeff_f32,
          boundary_count,
          track_id,
          left_cut,
          t,
          near_depth,
          far_depth,
          invalid_epsilon,
          start_depth);
    }
    if (!start_valid ||
        !wf2_endpoint_record_factorized_cut_depth(
            boundary_f32,
            track_ray_coeff_f32,
            boundary_count,
            track_id,
            right_cut,
            t,
            near_depth,
            far_depth,
            invalid_epsilon,
            end_depth)) {
      cached_right_valid = false;
      continue;
    }
    cached_right_cut = right_cut;
    cached_right_depth = end_depth;
    cached_right_valid = true;
    const float length = end_depth - start_depth;
    if (!(length > 1.0e-8f)) {
      continue;
    }
    const uint owner = uint(owner_raw);
    const uint rgba_base = owner * 4u;
    const float density = max(site_rgba_f32[rgba_base + 3u], 0.0f);
    const float seg_trans = exp(-density * length);
    const float seg_alpha = 1.0f - seg_trans;
    const float weight = transmittance * seg_alpha;
    const float3 rgb = float3(
        site_rgba_f32[rgba_base + 0u],
        site_rgba_f32[rgba_base + 1u],
        site_rgba_f32[rgba_base + 2u]);

    owners[local_run_count] = owner;
    lengths[local_run_count] = length;
    trans_before[local_run_count] = transmittance;
    local_run_count += 1u;

    rgb_accum += weight * rgb;
    transmittance *= seg_trans;
  }

  const float3 diff = rgb_accum - target_rgb;
  const float sample_loss = dot(diff, diff) * inv_element_count;
  tg_loss[local_frame] = sample_loss;
  const float3 grad_rgb = (2.0f * inv_element_count) * diff;

  float adj_next_transmittance = 0.0f;
  for (int run_id = int(local_run_count) - 1; run_id >= 0; --run_id) {
    const uint owner = owners[run_id];
    const uint rgba_base = owner * 4u;
    const float raw_density = site_rgba_f32[rgba_base + 3u];
    const float density = max(raw_density, 0.0f);
    const float length = lengths[run_id];
    const float seg_trans = exp(-density * length);
    const float seg_alpha = 1.0f - seg_trans;
    const float weight = trans_before[run_id] * seg_alpha;
    const float3 rgb = float3(
        site_rgba_f32[rgba_base + 0u],
        site_rgba_f32[rgba_base + 1u],
        site_rgba_f32[rgba_base + 2u]);
    const float adj_weight = dot(grad_rgb, rgb);
    const float adj_trans_before = adj_weight * seg_alpha + adj_next_transmittance * seg_trans;
    const float adj_segment_alpha = adj_weight * trans_before[run_id];
    const float adj_segment_trans = adj_next_transmittance * trans_before[run_id] - adj_segment_alpha;
    float4 grad_rgba = float4(
        weight * grad_rgb.x,
        weight * grad_rgb.y,
        weight * grad_rgb.z,
        0.0f);
    if (raw_density > 0.0f) {
      grad_rgba.w = adj_segment_trans * (-length * seg_trans);
    }
    if (reduce_small_sites) {
      tg_site_grad[local_frame * WF2_ENDPOINT_EDIT_FRAMEGROUP_REDUCE_SITES + owner] += grad_rgba;
    } else {
      wf2_atomic_add4(grad_site_rgba_f32, rgba_base, grad_rgba);
    }
    adj_next_transmittance = adj_trans_before;
  }
  }
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (local_frame == 0u) {
    float loss_sum = 0.0f;
    for (uint frame_id = 0u; frame_id < frames_in_chunk; ++frame_id) {
      loss_sum += tg_loss[frame_id];
    }
    atomic_fetch_add_explicit(&loss_f32[0], loss_sum, memory_order_relaxed);
  }

  if (reduce_small_sites) {
    if (local_frame < site_count) {
      float4 grad_sum = float4(0.0f, 0.0f, 0.0f, 0.0f);
      for (uint frame_id = 0u; frame_id < frames_in_chunk; ++frame_id) {
        grad_sum += tg_site_grad[
            frame_id * WF2_ENDPOINT_EDIT_FRAMEGROUP_REDUCE_SITES + local_frame];
      }
      wf2_atomic_add4(grad_site_rgba_f32, local_frame * 4u, grad_sum);
    }
  }
}

kernel void wf2_endpoint_record_delta_replace_coeff16_packed_framegroup16_smallrun16_mse_vjp_direct_atomic_rgb_only_tensor(
    device const half* coeff_f16 [[buffer(0)]],
    device const float* frame_t_f32 [[buffer(1)]],
    device const int* base_offsets_i32 [[buffer(2)]],
    device const int* base_record_i32 [[buffer(3)]],
    device const int* track_change_offsets_i32 [[buffer(4)]],
    device const short* track_chunk_change_offsets_i16 [[buffer(5)]],
    device const int* change_frame_i32 [[buffer(6)]],
    device const int* change_offsets_i32 [[buffer(7)]],
    device const int* change_record_i32 [[buffer(8)]],
    device const float* site_rgba_f32 [[buffer(9)]],
    device const float* target_rgb_f32 [[buffer(10)]],
    device const int* config_i32 [[buffer(11)]],
    device const float* config_f32 [[buffer(12)]],
    device atomic_float* loss_f32 [[buffer(13)]],
    device atomic_float* grad_site_rgba_f32 [[buffer(14)]],
    uint3 local_pos [[thread_position_in_threadgroup]],
    uint3 group_id [[threadgroup_position_in_grid]]) {
  const uint local_frame = local_pos.x;
  const uint boundary_count = uint(config_i32[0]);
  const uint track_count = uint(config_i32[1]);
  const uint frame_count = uint(config_i32[2]);
  const uint site_count = uint(config_i32[3]);
  const uint base_record_count = uint(config_i32[4]);
  const uint change_count = uint(config_i32[5]);
  const uint change_record_count = uint(config_i32[6]);
  const uint chunk_count = (frame_count + WF2_ENDPOINT_DELTA_FRAMEGROUP_MAX_FRAMES - 1u) /
      WF2_ENDPOINT_DELTA_FRAMEGROUP_MAX_FRAMES;
  const uint group_index = group_id.x;
  const uint track_id = group_index / max(chunk_count, 1u);
  const uint chunk_id = group_index - track_id * max(chunk_count, 1u);
  const uint frame_start = chunk_id * WF2_ENDPOINT_DELTA_FRAMEGROUP_MAX_FRAMES;
  const uint frames_in_chunk = frame_start < frame_count
      ? min(WF2_ENDPOINT_DELTA_FRAMEGROUP_MAX_FRAMES, frame_count - frame_start)
      : 0u;

  threadgroup int tg_valid[WF2_ENDPOINT_DELTA_FRAMEGROUP_MAX_FRAMES];
  threadgroup int tg_source[WF2_ENDPOINT_DELTA_FRAMEGROUP_MAX_FRAMES];
  threadgroup int tg_begin[WF2_ENDPOINT_DELTA_FRAMEGROUP_MAX_FRAMES];
  threadgroup int tg_end[WF2_ENDPOINT_DELTA_FRAMEGROUP_MAX_FRAMES];
  threadgroup float tg_loss[WF2_ENDPOINT_DELTA_FRAMEGROUP_MAX_FRAMES];
  threadgroup float4 tg_site_grad[
      WF2_ENDPOINT_DELTA_FRAMEGROUP_MAX_FRAMES * WF2_ENDPOINT_EDIT_FRAMEGROUP_REDUCE_SITES];

  if (local_frame == 0u) {
    for (uint frame_id = 0u; frame_id < WF2_ENDPOINT_DELTA_FRAMEGROUP_MAX_FRAMES; ++frame_id) {
      tg_valid[frame_id] = 0;
      tg_source[frame_id] = 0;
      tg_begin[frame_id] = 0;
      tg_end[frame_id] = 0;
    }

    bool ok = track_id < track_count && frames_in_chunk > 0u;
    int change_begin_raw = 0;
    int change_end_raw = 0;
    int chunk_change_begin_raw = 0;
    int chunk_change_end_raw = 0;
    if (ok) {
      change_begin_raw = track_change_offsets_i32[track_id];
      change_end_raw = track_change_offsets_i32[track_id + 1u];
      const uint chunk_offset_base = track_id * (chunk_count + 1u) + chunk_id;
      chunk_change_begin_raw = int(track_chunk_change_offsets_i16[chunk_offset_base]);
      chunk_change_end_raw = int(track_chunk_change_offsets_i16[chunk_offset_base + 1u]);
      ok = change_begin_raw >= 0 && change_end_raw >= change_begin_raw && uint(change_end_raw) <= change_count &&
          chunk_change_begin_raw >= change_begin_raw && chunk_change_end_raw >= chunk_change_begin_raw &&
          chunk_change_end_raw <= change_end_raw;
    }
    int selected_change = -1;
    if (ok && chunk_change_begin_raw > change_begin_raw) {
      selected_change = chunk_change_begin_raw - 1;
      while (selected_change >= change_begin_raw && change_frame_i32[uint(selected_change)] < 0) {
        selected_change -= 1;
      }
      if (selected_change < change_begin_raw) {
        selected_change = -1;
      }
    }

    uint change_cursor = ok ? uint(chunk_change_begin_raw) : 0u;
    const uint change_end = ok ? uint(chunk_change_end_raw) : 0u;

    for (uint local_frame_id = 0u; local_frame_id < frames_in_chunk; ++local_frame_id) {
      const uint global_frame_id = frame_start + local_frame_id;
      while (ok && change_cursor < change_end) {
        const int changed_frame = change_frame_i32[change_cursor];
        if (changed_frame < 0) {
          change_cursor += 1u;
          continue;
        }
        if (uint(changed_frame) > global_frame_id) {
          break;
        }

        selected_change = int(change_cursor);
        change_cursor += 1u;
      }

      if (ok) {
        const bool use_change = selected_change >= 0;
        const int begin_raw =
            use_change ? change_offsets_i32[uint(selected_change)] : base_offsets_i32[track_id];
        const int end_raw =
            use_change ? change_offsets_i32[uint(selected_change) + 1u] : base_offsets_i32[track_id + 1u];
        const uint record_count = use_change ? change_record_count : base_record_count;
        ok = begin_raw >= 0 && end_raw >= begin_raw && uint(end_raw) <= record_count &&
            uint(end_raw - begin_raw) <= WF2_ENDPOINT_DELTA_SMALLRUN_MAX_SEGMENTS;
        if (!ok) {
          break;
        }
        tg_source[local_frame_id] = use_change ? 1 : 0;
        tg_begin[local_frame_id] = begin_raw;
        tg_end[local_frame_id] = end_raw;
        tg_valid[local_frame_id] = 1;
      }
    }
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  const bool reduce_small_sites = site_count <= WF2_ENDPOINT_EDIT_FRAMEGROUP_REDUCE_SITES;
  if (reduce_small_sites) {
    const uint site_grad_base = local_frame * WF2_ENDPOINT_EDIT_FRAMEGROUP_REDUCE_SITES;
    for (uint site_slot = 0u; site_slot < WF2_ENDPOINT_EDIT_FRAMEGROUP_REDUCE_SITES; ++site_slot) {
      tg_site_grad[site_grad_base + site_slot] = float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
  }
  tg_loss[local_frame] = 0.0f;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const bool active = track_id < track_count && local_frame < frames_in_chunk && tg_valid[local_frame] != 0;
  if (active) {

  const uint global_frame_id = frame_start + local_frame;
  const uint sample_id = track_id * frame_count + global_frame_id;
  const uint total_samples = track_count * frame_count;
  const float near_depth = config_f32[0];
  const float far_depth = config_f32[1];
  const float invalid_epsilon = config_f32[2];
  const float transmittance_threshold = config_f32[3];
  const float t = frame_t_f32[global_frame_id];
  const bool use_change = tg_source[local_frame] != 0;
  const int begin_raw = tg_begin[local_frame];
  const int end_raw = tg_end[local_frame];
  const bool valid_row_bounds = end_raw >= begin_raw;
  const uint row_count = valid_row_bounds ? uint(end_raw - begin_raw) : 0u;
  const uint out_base = sample_id * 3u;
  const float3 target_rgb = float3(
      target_rgb_f32[out_base + 0u],
      target_rgb_f32[out_base + 1u],
      target_rgb_f32[out_base + 2u]);
  const float inv_element_count = 1.0f / float(total_samples * 3u);
  if (valid_row_bounds && row_count == 0u) {
    tg_loss[local_frame] = dot(target_rgb, target_rgb) * inv_element_count;
  } else if (valid_row_bounds) {

  uint owners[WF2_ENDPOINT_DELTA_SMALLRUN_MAX_SEGMENTS];
  float lengths[WF2_ENDPOINT_DELTA_SMALLRUN_MAX_SEGMENTS];
  float trans_before[WF2_ENDPOINT_DELTA_SMALLRUN_MAX_SEGMENTS];
  float segment_trans[WF2_ENDPOINT_DELTA_SMALLRUN_MAX_SEGMENTS];
  float segment_alpha[WF2_ENDPOINT_DELTA_SMALLRUN_MAX_SEGMENTS];
  float weights[WF2_ENDPOINT_DELTA_SMALLRUN_MAX_SEGMENTS];
  float3 segment_rgb[WF2_ENDPOINT_DELTA_SMALLRUN_MAX_SEGMENTS];

  float3 rgb_accum = float3(0.0f, 0.0f, 0.0f);
  float transmittance = 1.0f;
  uint local_run_count = 0u;
  int cached_right_cut = -2147483648;
  float cached_right_depth = 0.0f;
  bool cached_right_valid = false;
  for (uint cursor = 0u; cursor < row_count; ++cursor) {
    if (transmittance <= transmittance_threshold || local_run_count >= WF2_ENDPOINT_DELTA_SMALLRUN_MAX_SEGMENTS) {
      break;
    }
    const uint record_base = uint(begin_raw) + cursor;
    int owner_raw = 0;
    int left_cut = 0;
    int right_cut = 0;
    wf2_endpoint_record_unpack_record(
        use_change ? change_record_i32[record_base] : base_record_i32[record_base],
        owner_raw,
        left_cut,
        right_cut);
    if (owner_raw < 0 || uint(owner_raw) >= site_count) {
      continue;
    }
    float start_depth = 0.0f;
    float end_depth = 0.0f;
    bool start_valid = false;
    if (cached_right_valid && left_cut == cached_right_cut) {
      start_depth = cached_right_depth;
      start_valid = true;
    } else {
      start_valid = wf2_endpoint_record_coeff16_cut_depth(
          coeff_f16,
          boundary_count,
          track_id,
          left_cut,
          t,
          near_depth,
          far_depth,
          invalid_epsilon,
          start_depth);
    }
    if (!start_valid ||
        !wf2_endpoint_record_coeff16_cut_depth(
            coeff_f16,
            boundary_count,
            track_id,
            right_cut,
            t,
            near_depth,
            far_depth,
            invalid_epsilon,
            end_depth)) {
      cached_right_valid = false;
      continue;
    }
    cached_right_cut = right_cut;
    cached_right_depth = end_depth;
    cached_right_valid = true;
    const float length = end_depth - start_depth;
    if (!(length > 1.0e-8f)) {
      continue;
    }
    const uint owner = uint(owner_raw);
    const uint rgba_base = owner * 4u;
    const float density = max(site_rgba_f32[rgba_base + 3u], 0.0f);
    const float seg_trans = exp(-density * length);
    const float seg_alpha = 1.0f - seg_trans;
    const float weight = transmittance * seg_alpha;
    const float3 rgb = float3(
        site_rgba_f32[rgba_base + 0u],
        site_rgba_f32[rgba_base + 1u],
        site_rgba_f32[rgba_base + 2u]);

    owners[local_run_count] = owner;
    lengths[local_run_count] = length;
    trans_before[local_run_count] = transmittance;
    segment_trans[local_run_count] = seg_trans;
    segment_alpha[local_run_count] = seg_alpha;
    weights[local_run_count] = weight;
    segment_rgb[local_run_count] = rgb;
    local_run_count += 1u;

    rgb_accum += weight * rgb;
    transmittance *= seg_trans;
  }

  const float3 diff = rgb_accum - target_rgb;
  const float sample_loss = dot(diff, diff) * inv_element_count;
  tg_loss[local_frame] = sample_loss;
  const float3 grad_rgb = (2.0f * inv_element_count) * diff;

  float adj_next_transmittance = 0.0f;
  for (int run_id = int(local_run_count) - 1; run_id >= 0; --run_id) {
    const uint owner = owners[run_id];
    const uint rgba_base = owner * 4u;
    const float raw_density = site_rgba_f32[rgba_base + 3u];
    const float adj_weight = dot(grad_rgb, segment_rgb[run_id]);
    const float adj_trans_before =
        adj_weight * segment_alpha[run_id] +
        adj_next_transmittance * segment_trans[run_id];
    const float adj_segment_alpha = adj_weight * trans_before[run_id];
    const float adj_segment_trans = adj_next_transmittance * trans_before[run_id] - adj_segment_alpha;
    float4 grad_rgba = float4(
        weights[run_id] * grad_rgb.x,
        weights[run_id] * grad_rgb.y,
        weights[run_id] * grad_rgb.z,
        0.0f);
    if (raw_density > 0.0f) {
      grad_rgba.w = adj_segment_trans * (-lengths[run_id] * segment_trans[run_id]);
    }
    if (reduce_small_sites) {
      tg_site_grad[local_frame * WF2_ENDPOINT_EDIT_FRAMEGROUP_REDUCE_SITES + owner] += grad_rgba;
    } else {
      wf2_atomic_add4(grad_site_rgba_f32, rgba_base, grad_rgba);
    }
    adj_next_transmittance = adj_trans_before;
  }
  }
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (local_frame == 0u) {
    float loss_sum = 0.0f;
    for (uint frame_id = 0u; frame_id < frames_in_chunk; ++frame_id) {
      loss_sum += tg_loss[frame_id];
    }
    atomic_fetch_add_explicit(&loss_f32[0], loss_sum, memory_order_relaxed);
  }

  if (reduce_small_sites) {
    if (local_frame < site_count) {
      float4 grad_sum = float4(0.0f, 0.0f, 0.0f, 0.0f);
      for (uint frame_id = 0u; frame_id < frames_in_chunk; ++frame_id) {
        grad_sum += tg_site_grad[
            frame_id * WF2_ENDPOINT_EDIT_FRAMEGROUP_REDUCE_SITES + local_frame];
      }
      wf2_atomic_add4(grad_site_rgba_f32, local_frame * 4u, grad_sum);
    }
  }
}

kernel void wf2_endpoint_record_delta_replace_coeff16_packed_framegroup16_materialized_mse_vjp_direct_atomic_rgb_only_tensor(
    device const half* coeff_f16 [[buffer(0)]],
    device const float* frame_t_f32 [[buffer(1)]],
    device const int* base_offsets_i32 [[buffer(2)]],
    device const int* base_record_i32 [[buffer(3)]],
    device const int* track_change_offsets_i32 [[buffer(4)]],
    device const short* track_chunk_change_offsets_i16 [[buffer(5)]],
    device const int* change_frame_i32 [[buffer(6)]],
    device const int* change_offsets_i32 [[buffer(7)]],
    device const int* change_record_i32 [[buffer(8)]],
    device const float* site_rgba_f32 [[buffer(9)]],
    device const float* target_rgb_f32 [[buffer(10)]],
    device const int* config_i32 [[buffer(11)]],
    device const float* config_f32 [[buffer(12)]],
    device atomic_float* loss_f32 [[buffer(13)]],
    device atomic_float* grad_site_rgba_f32 [[buffer(14)]],
    uint3 local_pos [[thread_position_in_threadgroup]],
    uint3 group_id [[threadgroup_position_in_grid]]) {
  const uint local_frame = local_pos.x;
  const uint boundary_count = uint(config_i32[0]);
  const uint track_count = uint(config_i32[1]);
  const uint frame_count = uint(config_i32[2]);
  const uint site_count = uint(config_i32[3]);
  const uint base_record_count = uint(config_i32[4]);
  const uint change_count = uint(config_i32[5]);
  const uint change_record_count = uint(config_i32[6]);
  const uint chunk_count = (frame_count + WF2_ENDPOINT_EDIT_FRAMEGROUP_MAX_FRAMES - 1u) /
      WF2_ENDPOINT_EDIT_FRAMEGROUP_MAX_FRAMES;
  const uint group_index = group_id.x;
  const uint track_id = group_index / max(chunk_count, 1u);
  const uint chunk_id = group_index - track_id * max(chunk_count, 1u);
  const uint frame_start = chunk_id * WF2_ENDPOINT_EDIT_FRAMEGROUP_MAX_FRAMES;
  const uint frames_in_chunk = frame_start < frame_count
      ? min(WF2_ENDPOINT_EDIT_FRAMEGROUP_MAX_FRAMES, frame_count - frame_start)
      : 0u;

  threadgroup int tg_valid[WF2_ENDPOINT_EDIT_FRAMEGROUP_MAX_FRAMES];
  threadgroup uint tg_count[WF2_ENDPOINT_EDIT_FRAMEGROUP_MAX_FRAMES];
  threadgroup int tg_record[WF2_ENDPOINT_EDIT_FRAMEGROUP_MAX_FRAMES * WF2_MAX_REALRAY_SEGMENTS];
  threadgroup float4 tg_site_grad[
      WF2_ENDPOINT_EDIT_FRAMEGROUP_MAX_FRAMES * WF2_ENDPOINT_EDIT_FRAMEGROUP_REDUCE_SITES];

  if (local_frame == 0u) {
    for (uint frame_id = 0u; frame_id < WF2_ENDPOINT_EDIT_FRAMEGROUP_MAX_FRAMES; ++frame_id) {
      tg_valid[frame_id] = 0;
      tg_count[frame_id] = 0u;
    }

    bool ok = track_id < track_count && frames_in_chunk > 0u;
    int row_record[WF2_MAX_REALRAY_SEGMENTS];
    uint row_count = 0u;
    int change_begin_raw = 0;
    int change_end_raw = 0;
    int chunk_change_begin_raw = 0;
    int chunk_change_end_raw = 0;
    if (ok) {
      change_begin_raw = track_change_offsets_i32[track_id];
      change_end_raw = track_change_offsets_i32[track_id + 1u];
      const uint chunk_offset_base = track_id * (chunk_count + 1u) + chunk_id;
      chunk_change_begin_raw = int(track_chunk_change_offsets_i16[chunk_offset_base]);
      chunk_change_end_raw = int(track_chunk_change_offsets_i16[chunk_offset_base + 1u]);
      ok = change_begin_raw >= 0 && change_end_raw >= change_begin_raw && uint(change_end_raw) <= change_count &&
          chunk_change_begin_raw >= change_begin_raw && chunk_change_end_raw >= chunk_change_begin_raw &&
          chunk_change_end_raw <= change_end_raw;
    }
    int selected_change = -1;
    if (ok && chunk_change_begin_raw > change_begin_raw) {
      selected_change = chunk_change_begin_raw - 1;
      while (selected_change >= change_begin_raw && change_frame_i32[uint(selected_change)] < 0) {
        selected_change -= 1;
      }
      if (selected_change < change_begin_raw) {
        selected_change = -1;
      }
    }
    if (ok) {
      const bool use_change = selected_change >= 0;
      const int begin_raw =
          use_change ? change_offsets_i32[uint(selected_change)] : base_offsets_i32[track_id];
      const int end_raw =
          use_change ? change_offsets_i32[uint(selected_change) + 1u] : base_offsets_i32[track_id + 1u];
      const uint record_count = use_change ? change_record_count : base_record_count;
      ok = begin_raw >= 0 && end_raw >= begin_raw && uint(end_raw) <= record_count;
      for (uint cursor = ok ? uint(begin_raw) : 0u; ok && cursor < uint(end_raw); ++cursor) {
        if (row_count >= WF2_MAX_REALRAY_SEGMENTS) {
          ok = false;
          break;
        }
        row_record[row_count] = use_change ? change_record_i32[cursor] : base_record_i32[cursor];
        row_count += 1u;
      }
    }

    uint change_cursor = ok ? uint(chunk_change_begin_raw) : 0u;
    const uint change_end = ok ? uint(chunk_change_end_raw) : 0u;

    for (uint local_frame_id = 0u; local_frame_id < frames_in_chunk; ++local_frame_id) {
      const uint global_frame_id = frame_start + local_frame_id;
      while (ok && change_cursor < change_end) {
        const int changed_frame = change_frame_i32[change_cursor];
        if (changed_frame < 0) {
          change_cursor += 1u;
          continue;
        }
        if (uint(changed_frame) > global_frame_id) {
          break;
        }

        const int change_begin_row_raw = change_offsets_i32[change_cursor];
        const int change_end_row_raw = change_offsets_i32[change_cursor + 1u];
        ok = change_begin_row_raw >= 0 && change_end_row_raw >= change_begin_row_raw &&
            uint(change_end_row_raw) <= change_record_count;
        row_count = 0u;
        for (uint cursor = ok ? uint(change_begin_row_raw) : 0u; ok && cursor < uint(change_end_row_raw); ++cursor) {
          if (row_count >= WF2_MAX_REALRAY_SEGMENTS) {
            ok = false;
            break;
          }
          row_record[row_count] = change_record_i32[cursor];
          row_count += 1u;
        }
        if (ok) {
          change_cursor += 1u;
        }
      }

      if (ok) {
        const uint frame_base = local_frame_id * WF2_MAX_REALRAY_SEGMENTS;
        tg_count[local_frame_id] = row_count;
        tg_valid[local_frame_id] = 1;
        for (uint cursor = 0u; cursor < row_count; ++cursor) {
          tg_record[frame_base + cursor] = row_record[cursor];
        }
      }
    }
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  const bool reduce_small_sites = site_count <= WF2_ENDPOINT_EDIT_FRAMEGROUP_REDUCE_SITES;
  if (reduce_small_sites) {
    const uint site_grad_base = local_frame * WF2_ENDPOINT_EDIT_FRAMEGROUP_REDUCE_SITES;
    for (uint site_slot = 0u; site_slot < WF2_ENDPOINT_EDIT_FRAMEGROUP_REDUCE_SITES; ++site_slot) {
      tg_site_grad[site_grad_base + site_slot] = float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const bool active = track_id < track_count && local_frame < frames_in_chunk && tg_valid[local_frame] != 0;
  if (active) {

  const uint global_frame_id = frame_start + local_frame;
  const uint sample_id = track_id * frame_count + global_frame_id;
  const uint total_samples = track_count * frame_count;
  const float near_depth = config_f32[0];
  const float far_depth = config_f32[1];
  const float invalid_epsilon = config_f32[2];
  const float transmittance_threshold = config_f32[3];
  const float t = frame_t_f32[global_frame_id];
  const uint row_count = tg_count[local_frame];
  const uint frame_base = local_frame * WF2_MAX_REALRAY_SEGMENTS;

  uint owners[WF2_MAX_REALRAY_SEGMENTS];
  float lengths[WF2_MAX_REALRAY_SEGMENTS];
  float trans_before[WF2_MAX_REALRAY_SEGMENTS];
  float segment_trans[WF2_MAX_REALRAY_SEGMENTS];
  float segment_alpha[WF2_MAX_REALRAY_SEGMENTS];
  float weights[WF2_MAX_REALRAY_SEGMENTS];
  float3 segment_rgb[WF2_MAX_REALRAY_SEGMENTS];

  float3 rgb_accum = float3(0.0f, 0.0f, 0.0f);
  float transmittance = 1.0f;
  uint local_run_count = 0u;
  int cached_right_cut = -2147483648;
  float cached_right_depth = 0.0f;
  bool cached_right_valid = false;
  for (uint cursor = 0u; cursor < row_count; ++cursor) {
    if (transmittance <= transmittance_threshold || local_run_count >= WF2_MAX_REALRAY_SEGMENTS) {
      break;
    }
    const uint row_offset = frame_base + cursor;
    int owner_raw = 0;
    int left_cut = 0;
    int right_cut = 0;
    wf2_endpoint_record_unpack_record(tg_record[row_offset], owner_raw, left_cut, right_cut);
    if (owner_raw < 0 || uint(owner_raw) >= site_count) {
      continue;
    }
    float start_depth = 0.0f;
    float end_depth = 0.0f;
    bool start_valid = false;
    if (cached_right_valid && left_cut == cached_right_cut) {
      start_depth = cached_right_depth;
      start_valid = true;
    } else {
      start_valid = wf2_endpoint_record_coeff16_cut_depth(
          coeff_f16,
          boundary_count,
          track_id,
          left_cut,
          t,
          near_depth,
          far_depth,
          invalid_epsilon,
          start_depth);
    }
    if (!start_valid ||
        !wf2_endpoint_record_coeff16_cut_depth(
            coeff_f16,
            boundary_count,
            track_id,
            right_cut,
            t,
            near_depth,
            far_depth,
            invalid_epsilon,
            end_depth)) {
      cached_right_valid = false;
      continue;
    }
    cached_right_cut = right_cut;
    cached_right_depth = end_depth;
    cached_right_valid = true;
    const float length = end_depth - start_depth;
    if (!(length > 1.0e-8f)) {
      continue;
    }
    const uint owner = uint(owner_raw);
    const uint rgba_base = owner * 4u;
    const float density = max(site_rgba_f32[rgba_base + 3u], 0.0f);
    const float seg_trans = exp(-density * length);
    const float seg_alpha = 1.0f - seg_trans;
    const float weight = transmittance * seg_alpha;
    const float3 rgb = float3(
        site_rgba_f32[rgba_base + 0u],
        site_rgba_f32[rgba_base + 1u],
        site_rgba_f32[rgba_base + 2u]);

    owners[local_run_count] = owner;
    lengths[local_run_count] = length;
    trans_before[local_run_count] = transmittance;
    segment_trans[local_run_count] = seg_trans;
    segment_alpha[local_run_count] = seg_alpha;
    weights[local_run_count] = weight;
    segment_rgb[local_run_count] = rgb;
    local_run_count += 1u;

    rgb_accum += weight * rgb;
    transmittance *= seg_trans;
  }

  const uint out_base = sample_id * 3u;
  const float3 target_rgb = float3(
      target_rgb_f32[out_base + 0u],
      target_rgb_f32[out_base + 1u],
      target_rgb_f32[out_base + 2u]);
  const float3 diff = rgb_accum - target_rgb;
  const float inv_element_count = 1.0f / float(total_samples * 3u);
  const float sample_loss = dot(diff, diff) * inv_element_count;
  atomic_fetch_add_explicit(&loss_f32[0], sample_loss, memory_order_relaxed);
  const float3 grad_rgb = (2.0f * inv_element_count) * diff;

  float adj_next_transmittance = 0.0f;
  for (int run_id = int(local_run_count) - 1; run_id >= 0; --run_id) {
    const uint owner = owners[run_id];
    const uint rgba_base = owner * 4u;
    const float raw_density = site_rgba_f32[rgba_base + 3u];
    const float adj_weight = dot(grad_rgb, segment_rgb[run_id]);
    const float adj_trans_before =
        adj_weight * segment_alpha[run_id] +
        adj_next_transmittance * segment_trans[run_id];
    const float adj_segment_alpha = adj_weight * trans_before[run_id];
    const float adj_segment_trans = adj_next_transmittance * trans_before[run_id] - adj_segment_alpha;
    float4 grad_rgba = float4(
        weights[run_id] * grad_rgb.x,
        weights[run_id] * grad_rgb.y,
        weights[run_id] * grad_rgb.z,
        0.0f);
    if (raw_density > 0.0f) {
      grad_rgba.w = adj_segment_trans * (-lengths[run_id] * segment_trans[run_id]);
    }
    if (reduce_small_sites) {
      tg_site_grad[local_frame * WF2_ENDPOINT_EDIT_FRAMEGROUP_REDUCE_SITES + owner] += grad_rgba;
    } else {
      wf2_atomic_add4(grad_site_rgba_f32, rgba_base, grad_rgba);
    }
    adj_next_transmittance = adj_trans_before;
  }
  }

  if (reduce_small_sites) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (local_frame < site_count) {
      float4 grad_sum = float4(0.0f, 0.0f, 0.0f, 0.0f);
      for (uint frame_id = 0u; frame_id < frames_in_chunk; ++frame_id) {
        grad_sum += tg_site_grad[
            frame_id * WF2_ENDPOINT_EDIT_FRAMEGROUP_REDUCE_SITES + local_frame];
      }
      wf2_atomic_add4(grad_site_rgba_f32, local_frame * 4u, grad_sum);
    }
  }
}

kernel void wf2_endpoint_record_delta_replace_coeff16_i16x3_framegroup16_materialized_mse_vjp_direct_atomic_rgb_only_tensor(
    device const half* coeff_f16 [[buffer(0)]],
    device const float* frame_t_f32 [[buffer(1)]],
    device const int* base_offsets_i32 [[buffer(2)]],
    device const short* base_record_i16 [[buffer(3)]],
    device const int* track_change_offsets_i32 [[buffer(4)]],
    device const short* track_chunk_change_offsets_i16 [[buffer(5)]],
    device const int* change_frame_i32 [[buffer(6)]],
    device const int* change_offsets_i32 [[buffer(7)]],
    device const short* change_record_i16 [[buffer(8)]],
    device const float* site_rgba_f32 [[buffer(9)]],
    device const float* target_rgb_f32 [[buffer(10)]],
    device const int* config_i32 [[buffer(11)]],
    device const float* config_f32 [[buffer(12)]],
    device atomic_float* loss_f32 [[buffer(13)]],
    device atomic_float* grad_site_rgba_f32 [[buffer(14)]],
    uint3 local_pos [[thread_position_in_threadgroup]],
    uint3 group_id [[threadgroup_position_in_grid]]) {
  const uint local_frame = local_pos.x;
  const uint boundary_count = uint(config_i32[0]);
  const uint track_count = uint(config_i32[1]);
  const uint frame_count = uint(config_i32[2]);
  const uint site_count = uint(config_i32[3]);
  const uint base_record_count = uint(config_i32[4]);
  const uint change_count = uint(config_i32[5]);
  const uint change_record_count = uint(config_i32[6]);
  const uint chunk_count = (frame_count + WF2_ENDPOINT_EDIT_FRAMEGROUP_MAX_FRAMES - 1u) /
      WF2_ENDPOINT_EDIT_FRAMEGROUP_MAX_FRAMES;
  const uint group_index = group_id.x;
  const uint track_id = group_index / max(chunk_count, 1u);
  const uint chunk_id = group_index - track_id * max(chunk_count, 1u);
  const uint frame_start = chunk_id * WF2_ENDPOINT_EDIT_FRAMEGROUP_MAX_FRAMES;
  const uint frames_in_chunk = frame_start < frame_count
      ? min(WF2_ENDPOINT_EDIT_FRAMEGROUP_MAX_FRAMES, frame_count - frame_start)
      : 0u;

  threadgroup int tg_valid[WF2_ENDPOINT_EDIT_FRAMEGROUP_MAX_FRAMES];
  threadgroup uint tg_count[WF2_ENDPOINT_EDIT_FRAMEGROUP_MAX_FRAMES];
  threadgroup int tg_owner[WF2_ENDPOINT_EDIT_FRAMEGROUP_MAX_FRAMES * WF2_MAX_REALRAY_SEGMENTS];
  threadgroup int tg_left[WF2_ENDPOINT_EDIT_FRAMEGROUP_MAX_FRAMES * WF2_MAX_REALRAY_SEGMENTS];
  threadgroup int tg_right[WF2_ENDPOINT_EDIT_FRAMEGROUP_MAX_FRAMES * WF2_MAX_REALRAY_SEGMENTS];
  threadgroup float4 tg_site_grad[
      WF2_ENDPOINT_EDIT_FRAMEGROUP_MAX_FRAMES * WF2_ENDPOINT_EDIT_FRAMEGROUP_REDUCE_SITES];

  if (local_frame == 0u) {
    for (uint frame_id = 0u; frame_id < WF2_ENDPOINT_EDIT_FRAMEGROUP_MAX_FRAMES; ++frame_id) {
      tg_valid[frame_id] = 0;
      tg_count[frame_id] = 0u;
    }

    bool ok = track_id < track_count && frames_in_chunk > 0u;
    int row_owner[WF2_MAX_REALRAY_SEGMENTS];
    int row_left[WF2_MAX_REALRAY_SEGMENTS];
    int row_right[WF2_MAX_REALRAY_SEGMENTS];
    uint row_count = 0u;
    int change_begin_raw = 0;
    int change_end_raw = 0;
    int chunk_change_begin_raw = 0;
    int chunk_change_end_raw = 0;
    if (ok) {
      change_begin_raw = track_change_offsets_i32[track_id];
      change_end_raw = track_change_offsets_i32[track_id + 1u];
      const uint chunk_offset_base = track_id * (chunk_count + 1u) + chunk_id;
      chunk_change_begin_raw = int(track_chunk_change_offsets_i16[chunk_offset_base]);
      chunk_change_end_raw = int(track_chunk_change_offsets_i16[chunk_offset_base + 1u]);
      ok = change_begin_raw >= 0 && change_end_raw >= change_begin_raw && uint(change_end_raw) <= change_count &&
          chunk_change_begin_raw >= change_begin_raw && chunk_change_end_raw >= chunk_change_begin_raw &&
          chunk_change_end_raw <= change_end_raw;
    }
    int selected_change = -1;
    if (ok && chunk_change_begin_raw > change_begin_raw) {
      selected_change = chunk_change_begin_raw - 1;
      while (selected_change >= change_begin_raw && change_frame_i32[uint(selected_change)] < 0) {
        selected_change -= 1;
      }
      if (selected_change < change_begin_raw) {
        selected_change = -1;
      }
    }
    if (ok) {
      const bool use_change = selected_change >= 0;
      const int begin_raw =
          use_change ? change_offsets_i32[uint(selected_change)] : base_offsets_i32[track_id];
      const int end_raw =
          use_change ? change_offsets_i32[uint(selected_change) + 1u] : base_offsets_i32[track_id + 1u];
      const uint record_count = use_change ? change_record_count : base_record_count;
      ok = begin_raw >= 0 && end_raw >= begin_raw && uint(end_raw) <= record_count;
      for (uint cursor = ok ? uint(begin_raw) : 0u; ok && cursor < uint(end_raw); ++cursor) {
        if (row_count >= WF2_MAX_REALRAY_SEGMENTS) {
          ok = false;
          break;
        }
        const uint record_base = cursor * 3u;
        row_owner[row_count] =
            use_change ? int(change_record_i16[record_base + 0u]) : int(base_record_i16[record_base + 0u]);
        row_left[row_count] =
            use_change ? int(change_record_i16[record_base + 1u]) : int(base_record_i16[record_base + 1u]);
        row_right[row_count] =
            use_change ? int(change_record_i16[record_base + 2u]) : int(base_record_i16[record_base + 2u]);
        row_count += 1u;
      }
    }

    uint change_cursor = ok ? uint(chunk_change_begin_raw) : 0u;
    const uint change_end = ok ? uint(chunk_change_end_raw) : 0u;

    for (uint local_frame_id = 0u; local_frame_id < frames_in_chunk; ++local_frame_id) {
      const uint global_frame_id = frame_start + local_frame_id;
      while (ok && change_cursor < change_end) {
        const int changed_frame = change_frame_i32[change_cursor];
        if (changed_frame < 0) {
          change_cursor += 1u;
          continue;
        }
        if (uint(changed_frame) > global_frame_id) {
          break;
        }

        const int change_begin_row_raw = change_offsets_i32[change_cursor];
        const int change_end_row_raw = change_offsets_i32[change_cursor + 1u];
        ok = change_begin_row_raw >= 0 && change_end_row_raw >= change_begin_row_raw &&
            uint(change_end_row_raw) <= change_record_count;
        row_count = 0u;
        for (uint cursor = ok ? uint(change_begin_row_raw) : 0u; ok && cursor < uint(change_end_row_raw); ++cursor) {
          if (row_count >= WF2_MAX_REALRAY_SEGMENTS) {
            ok = false;
            break;
          }
          const uint record_base = cursor * 3u;
          row_owner[row_count] = int(change_record_i16[record_base + 0u]);
          row_left[row_count] = int(change_record_i16[record_base + 1u]);
          row_right[row_count] = int(change_record_i16[record_base + 2u]);
          row_count += 1u;
        }
        if (ok) {
          change_cursor += 1u;
        }
      }

      if (ok) {
        const uint frame_base = local_frame_id * WF2_MAX_REALRAY_SEGMENTS;
        tg_count[local_frame_id] = row_count;
        tg_valid[local_frame_id] = 1;
        for (uint cursor = 0u; cursor < row_count; ++cursor) {
          const uint out = frame_base + cursor;
          tg_owner[out] = row_owner[cursor];
          tg_left[out] = row_left[cursor];
          tg_right[out] = row_right[cursor];
        }
      }
    }
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  const bool reduce_small_sites = site_count <= WF2_ENDPOINT_EDIT_FRAMEGROUP_REDUCE_SITES;
  if (reduce_small_sites) {
    const uint site_grad_base = local_frame * WF2_ENDPOINT_EDIT_FRAMEGROUP_REDUCE_SITES;
    for (uint site_slot = 0u; site_slot < WF2_ENDPOINT_EDIT_FRAMEGROUP_REDUCE_SITES; ++site_slot) {
      tg_site_grad[site_grad_base + site_slot] = float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const bool active = track_id < track_count && local_frame < frames_in_chunk && tg_valid[local_frame] != 0;
  if (active) {

  const uint global_frame_id = frame_start + local_frame;
  const uint sample_id = track_id * frame_count + global_frame_id;
  const uint total_samples = track_count * frame_count;
  const float near_depth = config_f32[0];
  const float far_depth = config_f32[1];
  const float invalid_epsilon = config_f32[2];
  const float transmittance_threshold = config_f32[3];
  const float t = frame_t_f32[global_frame_id];
  const uint row_count = tg_count[local_frame];
  const uint frame_base = local_frame * WF2_MAX_REALRAY_SEGMENTS;

  uint owners[WF2_MAX_REALRAY_SEGMENTS];
  float lengths[WF2_MAX_REALRAY_SEGMENTS];
  float trans_before[WF2_MAX_REALRAY_SEGMENTS];
  float segment_trans[WF2_MAX_REALRAY_SEGMENTS];
  float segment_alpha[WF2_MAX_REALRAY_SEGMENTS];
  float weights[WF2_MAX_REALRAY_SEGMENTS];
  float3 segment_rgb[WF2_MAX_REALRAY_SEGMENTS];

  float3 rgb_accum = float3(0.0f, 0.0f, 0.0f);
  float transmittance = 1.0f;
  uint local_run_count = 0u;
  int cached_right_cut = -2147483648;
  float cached_right_depth = 0.0f;
  bool cached_right_valid = false;
  for (uint cursor = 0u; cursor < row_count; ++cursor) {
    if (transmittance <= transmittance_threshold || local_run_count >= WF2_MAX_REALRAY_SEGMENTS) {
      break;
    }
    const uint row_offset = frame_base + cursor;
    const int owner_raw = tg_owner[row_offset];
    if (owner_raw < 0 || uint(owner_raw) >= site_count) {
      continue;
    }
    float start_depth = 0.0f;
    float end_depth = 0.0f;
    const int left_cut = tg_left[row_offset];
    const int right_cut = tg_right[row_offset];
    bool start_valid = false;
    if (cached_right_valid && left_cut == cached_right_cut) {
      start_depth = cached_right_depth;
      start_valid = true;
    } else {
      start_valid = wf2_endpoint_record_coeff16_cut_depth(
          coeff_f16,
          boundary_count,
          track_id,
          left_cut,
          t,
          near_depth,
          far_depth,
          invalid_epsilon,
          start_depth);
    }
    if (!start_valid ||
        !wf2_endpoint_record_coeff16_cut_depth(
            coeff_f16,
            boundary_count,
            track_id,
            right_cut,
            t,
            near_depth,
            far_depth,
            invalid_epsilon,
            end_depth)) {
      cached_right_valid = false;
      continue;
    }
    cached_right_cut = right_cut;
    cached_right_depth = end_depth;
    cached_right_valid = true;
    const float length = end_depth - start_depth;
    if (!(length > 1.0e-8f)) {
      continue;
    }
    const uint owner = uint(owner_raw);
    const uint rgba_base = owner * 4u;
    const float density = max(site_rgba_f32[rgba_base + 3u], 0.0f);
    const float seg_trans = exp(-density * length);
    const float seg_alpha = 1.0f - seg_trans;
    const float weight = transmittance * seg_alpha;
    const float3 rgb = float3(
        site_rgba_f32[rgba_base + 0u],
        site_rgba_f32[rgba_base + 1u],
        site_rgba_f32[rgba_base + 2u]);

    owners[local_run_count] = owner;
    lengths[local_run_count] = length;
    trans_before[local_run_count] = transmittance;
    segment_trans[local_run_count] = seg_trans;
    segment_alpha[local_run_count] = seg_alpha;
    weights[local_run_count] = weight;
    segment_rgb[local_run_count] = rgb;
    local_run_count += 1u;

    rgb_accum += weight * rgb;
    transmittance *= seg_trans;
  }

  const uint out_base = sample_id * 3u;
  const float3 target_rgb = float3(
      target_rgb_f32[out_base + 0u],
      target_rgb_f32[out_base + 1u],
      target_rgb_f32[out_base + 2u]);
  const float3 diff = rgb_accum - target_rgb;
  const float inv_element_count = 1.0f / float(total_samples * 3u);
  const float sample_loss = dot(diff, diff) * inv_element_count;
  atomic_fetch_add_explicit(&loss_f32[0], sample_loss, memory_order_relaxed);
  const float3 grad_rgb = (2.0f * inv_element_count) * diff;

  float adj_next_transmittance = 0.0f;
  for (int run_id = int(local_run_count) - 1; run_id >= 0; --run_id) {
    const uint owner = owners[run_id];
    const uint rgba_base = owner * 4u;
    const float raw_density = site_rgba_f32[rgba_base + 3u];
    const float adj_weight = dot(grad_rgb, segment_rgb[run_id]);
    const float adj_trans_before =
        adj_weight * segment_alpha[run_id] +
        adj_next_transmittance * segment_trans[run_id];
    const float adj_segment_alpha = adj_weight * trans_before[run_id];
    const float adj_segment_trans = adj_next_transmittance * trans_before[run_id] - adj_segment_alpha;
    float4 grad_rgba = float4(
        weights[run_id] * grad_rgb.x,
        weights[run_id] * grad_rgb.y,
        weights[run_id] * grad_rgb.z,
        0.0f);
    if (raw_density > 0.0f) {
      grad_rgba.w = adj_segment_trans * (-lengths[run_id] * segment_trans[run_id]);
    }
    if (reduce_small_sites) {
      tg_site_grad[local_frame * WF2_ENDPOINT_EDIT_FRAMEGROUP_REDUCE_SITES + owner] += grad_rgba;
    } else {
      wf2_atomic_add4(grad_site_rgba_f32, rgba_base, grad_rgba);
    }
    adj_next_transmittance = adj_trans_before;
  }
  }

  if (reduce_small_sites) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (local_frame < site_count) {
      float4 grad_sum = float4(0.0f, 0.0f, 0.0f, 0.0f);
      for (uint frame_id = 0u; frame_id < frames_in_chunk; ++frame_id) {
        grad_sum += tg_site_grad[
            frame_id * WF2_ENDPOINT_EDIT_FRAMEGROUP_REDUCE_SITES + local_frame];
      }
      wf2_atomic_add4(grad_site_rgba_f32, local_frame * 4u, grad_sum);
    }
  }
}

kernel void wf2_endpoint_record_delta_replace_coeff16_i16x4_mse_vjp_direct_atomic_rgb_only_tensor(
    device const half* coeff_f16 [[buffer(0)]],
    device const float* frame_t_f32 [[buffer(1)]],
    device const int* base_offsets_i32 [[buffer(2)]],
    device const short* base_record_i16x4 [[buffer(3)]],
    device const int* track_change_offsets_i32 [[buffer(4)]],
    device const int* change_frame_i32 [[buffer(5)]],
    device const int* change_offsets_i32 [[buffer(6)]],
    device const short* change_record_i16x4 [[buffer(7)]],
    device const float* site_rgba_f32 [[buffer(8)]],
    device const float* target_rgb_f32 [[buffer(9)]],
    device const int* config_i32 [[buffer(10)]],
    device const float* config_f32 [[buffer(11)]],
    device atomic_float* loss_f32 [[buffer(12)]],
    device atomic_float* grad_site_rgba_f32 [[buffer(13)]],
    uint sample_id [[thread_position_in_grid]]) {
  const uint boundary_count = uint(config_i32[0]);
  const uint track_count = uint(config_i32[1]);
  const uint frame_count = uint(config_i32[2]);
  const uint site_count = uint(config_i32[3]);
  const uint base_record_count = uint(config_i32[4]);
  const uint change_count = uint(config_i32[5]);
  const uint change_record_count = uint(config_i32[6]);
  const uint total_samples = track_count * frame_count;
  if (sample_id >= total_samples) {
    return;
  }

  const uint track_id = sample_id / frame_count;
  const uint frame_id = sample_id - track_id * frame_count;
  const float near_depth = config_f32[0];
  const float far_depth = config_f32[1];
  const float invalid_epsilon = config_f32[2];
  const float transmittance_threshold = config_f32[3];
  const float t = frame_t_f32[frame_id];

  int selected_change = -1;
  const int change_begin_raw = track_change_offsets_i32[track_id];
  const int change_end_raw = track_change_offsets_i32[track_id + 1u];
  if (change_begin_raw < 0 || change_end_raw < change_begin_raw || uint(change_end_raw) > change_count) {
    return;
  }
  for (uint change_cursor = uint(change_begin_raw); change_cursor < uint(change_end_raw); ++change_cursor) {
    const int changed_frame = change_frame_i32[change_cursor];
    if (changed_frame < 0) {
      continue;
    }
    if (uint(changed_frame) <= frame_id) {
      selected_change = int(change_cursor);
      continue;
    }
    break;
  }

  const bool use_change = selected_change >= 0;
  const int begin_raw = use_change ? change_offsets_i32[uint(selected_change)] : base_offsets_i32[track_id];
  const int end_raw = use_change ? change_offsets_i32[uint(selected_change) + 1u] : base_offsets_i32[track_id + 1u];
  const uint record_count = use_change ? change_record_count : base_record_count;
  if (begin_raw < 0 || end_raw < begin_raw || uint(end_raw) > record_count) {
    return;
  }

  uint owners[WF2_MAX_REALRAY_SEGMENTS];
  float lengths[WF2_MAX_REALRAY_SEGMENTS];
  float trans_before[WF2_MAX_REALRAY_SEGMENTS];
  float segment_trans[WF2_MAX_REALRAY_SEGMENTS];
  float segment_alpha[WF2_MAX_REALRAY_SEGMENTS];
  float weights[WF2_MAX_REALRAY_SEGMENTS];
  float3 segment_rgb[WF2_MAX_REALRAY_SEGMENTS];

  float3 rgb_accum = float3(0.0f, 0.0f, 0.0f);
  float transmittance = 1.0f;
  uint local_run_count = 0u;
  int cached_right_cut = -2147483648;
  float cached_right_depth = 0.0f;
  bool cached_right_valid = false;
  for (uint cursor = uint(begin_raw); cursor < uint(end_raw); ++cursor) {
    if (transmittance <= transmittance_threshold || local_run_count >= WF2_MAX_REALRAY_SEGMENTS) {
      break;
    }
    const uint record_base = cursor * 4u;
    const int owner_raw = use_change ? int(change_record_i16x4[record_base + 0u]) : int(base_record_i16x4[record_base + 0u]);
    if (owner_raw < 0 || uint(owner_raw) >= site_count) {
      continue;
    }
    float start_depth = 0.0f;
    float end_depth = 0.0f;
    const int left_cut = use_change ? int(change_record_i16x4[record_base + 1u]) : int(base_record_i16x4[record_base + 1u]);
    const int right_cut = use_change ? int(change_record_i16x4[record_base + 2u]) : int(base_record_i16x4[record_base + 2u]);
    bool start_valid = false;
    if (cached_right_valid && left_cut == cached_right_cut) {
      start_depth = cached_right_depth;
      start_valid = true;
    } else {
      start_valid = wf2_endpoint_record_coeff16_cut_depth(
          coeff_f16,
          boundary_count,
          track_id,
          left_cut,
          t,
          near_depth,
          far_depth,
          invalid_epsilon,
          start_depth);
    }
    if (!start_valid ||
        !wf2_endpoint_record_coeff16_cut_depth(
            coeff_f16,
            boundary_count,
            track_id,
            right_cut,
            t,
            near_depth,
            far_depth,
            invalid_epsilon,
            end_depth)) {
      cached_right_valid = false;
      continue;
    }
    cached_right_cut = right_cut;
    cached_right_depth = end_depth;
    cached_right_valid = true;
    const float length = end_depth - start_depth;
    if (!(length > 1.0e-8f)) {
      continue;
    }
    const uint owner = uint(owner_raw);
    const uint rgba_base = owner * 4u;
    const float density = max(site_rgba_f32[rgba_base + 3u], 0.0f);
    const float seg_trans = exp(-density * length);
    const float seg_alpha = 1.0f - seg_trans;
    const float weight = transmittance * seg_alpha;
    const float3 rgb = float3(
        site_rgba_f32[rgba_base + 0u],
        site_rgba_f32[rgba_base + 1u],
        site_rgba_f32[rgba_base + 2u]);

    owners[local_run_count] = owner;
    lengths[local_run_count] = length;
    trans_before[local_run_count] = transmittance;
    segment_trans[local_run_count] = seg_trans;
    segment_alpha[local_run_count] = seg_alpha;
    weights[local_run_count] = weight;
    segment_rgb[local_run_count] = rgb;
    local_run_count += 1u;

    rgb_accum += weight * rgb;
    transmittance *= seg_trans;
  }

  const uint out_base = sample_id * 3u;
  const float3 target_rgb = float3(
      target_rgb_f32[out_base + 0u],
      target_rgb_f32[out_base + 1u],
      target_rgb_f32[out_base + 2u]);
  const float3 diff = rgb_accum - target_rgb;
  const float inv_element_count = 1.0f / float(total_samples * 3u);
  const float sample_loss = dot(diff, diff) * inv_element_count;
  atomic_fetch_add_explicit(&loss_f32[0], sample_loss, memory_order_relaxed);
  const float3 grad_rgb = (2.0f * inv_element_count) * diff;

  float adj_next_transmittance = 0.0f;
  for (int run_id = int(local_run_count) - 1; run_id >= 0; --run_id) {
    const uint owner = owners[run_id];
    const uint rgba_base = owner * 4u;
    const float raw_density = site_rgba_f32[rgba_base + 3u];
    const float adj_weight = dot(grad_rgb, segment_rgb[run_id]);
    const float adj_trans_before =
        adj_weight * segment_alpha[run_id] +
        adj_next_transmittance * segment_trans[run_id];
    const float adj_segment_alpha = adj_weight * trans_before[run_id];
    const float adj_segment_trans = adj_next_transmittance * trans_before[run_id] - adj_segment_alpha;
    float4 grad_rgba = float4(
        weights[run_id] * grad_rgb.x,
        weights[run_id] * grad_rgb.y,
        weights[run_id] * grad_rgb.z,
        0.0f);
    if (raw_density > 0.0f) {
      grad_rgba.w = adj_segment_trans * (-lengths[run_id] * segment_trans[run_id]);
    }
    wf2_atomic_add4(grad_site_rgba_f32, rgba_base, grad_rgba);
    adj_next_transmittance = adj_trans_before;
  }
}

kernel void wf2_endpoint_record_edit_block4_vjp_direct_atomic_rgb_only_tensor(
    device const float* boundary_f32 [[buffer(0)]],
    device const float* rays_f32 [[buffer(1)]],
    device const float* frame_t_f32 [[buffer(2)]],
    device const int* anchor_offsets_i32 [[buffer(3)]],
    device const int* anchor_owner_i32 [[buffer(4)]],
    device const int* anchor_left_i32 [[buffer(5)]],
    device const int* anchor_right_i32 [[buffer(6)]],
    device const int* track_block_change_offsets_i32 [[buffer(7)]],
    device const int* change_frame_i32 [[buffer(8)]],
    device const int* op_offsets_i32 [[buffer(9)]],
    device const int* op_type_i32 [[buffer(10)]],
    device const int* op_pos_i32 [[buffer(11)]],
    device const int* op_owner_i32 [[buffer(12)]],
    device const int* op_left_i32 [[buffer(13)]],
    device const int* op_right_i32 [[buffer(14)]],
    device const float* site_rgba_f32 [[buffer(15)]],
    device const float* grad_rgb_f32 [[buffer(16)]],
    device const int* config_i32 [[buffer(17)]],
    device const float* config_f32 [[buffer(18)]],
    device atomic_float* grad_site_rgba_f32 [[buffer(19)]],
    uint sample_id [[thread_position_in_grid]]) {
  const uint boundary_count = uint(config_i32[0]);
  const uint track_count = uint(config_i32[1]);
  const uint frame_count = uint(config_i32[2]);
  const uint site_count = uint(config_i32[3]);
  const uint anchor_record_count = uint(config_i32[4]);
  const uint change_count = uint(config_i32[5]);
  const uint op_count = uint(config_i32[6]);
  const uint block_size = uint(config_i32[7]);
  const uint block_count = uint(config_i32[8]);
  const uint total_samples = track_count * frame_count;
  if (sample_id >= total_samples) {
    return;
  }

  const uint track_id = sample_id / frame_count;
  const uint frame_id = sample_id - track_id * frame_count;
  const float near_depth = config_f32[0];
  const float far_depth = config_f32[1];
  const float invalid_epsilon = config_f32[2];
  const float transmittance_threshold = config_f32[3];
  const uint ray_base = sample_id * 6u;
  const float3 origin = float3(rays_f32[ray_base + 0u], rays_f32[ray_base + 1u], rays_f32[ray_base + 2u]);
  const float3 direction = float3(rays_f32[ray_base + 3u], rays_f32[ray_base + 4u], rays_f32[ray_base + 5u]);
  const float t = frame_t_f32[frame_id];

  int row_owner[WF2_MAX_REALRAY_SEGMENTS];
  int row_left[WF2_MAX_REALRAY_SEGMENTS];
  int row_right[WF2_MAX_REALRAY_SEGMENTS];
  uint row_count = 0u;
  if (!wf2_endpoint_record_load_block_edit_row(
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
          track_id,
          frame_id,
          block_size,
          block_count,
          anchor_record_count,
          change_count,
          op_count,
          row_owner,
          row_left,
          row_right,
          row_count)) {
    return;
  }

  uint owners[WF2_MAX_REALRAY_SEGMENTS];
  float lengths[WF2_MAX_REALRAY_SEGMENTS];
  float trans_before[WF2_MAX_REALRAY_SEGMENTS];
  float segment_trans[WF2_MAX_REALRAY_SEGMENTS];
  float segment_alpha[WF2_MAX_REALRAY_SEGMENTS];
  float weights[WF2_MAX_REALRAY_SEGMENTS];
  float3 segment_rgb[WF2_MAX_REALRAY_SEGMENTS];

  float transmittance = 1.0f;
  uint local_run_count = 0u;
  int cached_right_cut = -2147483648;
  float cached_right_depth = 0.0f;
  bool cached_right_valid = false;
  for (uint cursor = 0u; cursor < row_count; ++cursor) {
    if (transmittance <= transmittance_threshold || local_run_count >= WF2_MAX_REALRAY_SEGMENTS) {
      break;
    }
    const int owner_raw = row_owner[cursor];
    if (owner_raw < 0 || uint(owner_raw) >= site_count) {
      continue;
    }
    float start_depth = 0.0f;
    float end_depth = 0.0f;
    const int left_cut = row_left[cursor];
    const int right_cut = row_right[cursor];
    bool start_valid = false;
    if (cached_right_valid && left_cut == cached_right_cut) {
      start_depth = cached_right_depth;
      start_valid = true;
    } else {
      start_valid = wf2_endpoint_record_cut_depth(
          boundary_f32,
          boundary_count,
          left_cut,
          origin,
          direction,
          t,
          near_depth,
          far_depth,
          invalid_epsilon,
          start_depth);
    }
    if (!start_valid ||
        !wf2_endpoint_record_cut_depth(
            boundary_f32,
            boundary_count,
            right_cut,
            origin,
            direction,
            t,
            near_depth,
            far_depth,
            invalid_epsilon,
            end_depth)) {
      cached_right_valid = false;
      continue;
    }
    cached_right_cut = right_cut;
    cached_right_depth = end_depth;
    cached_right_valid = true;
    const float length = end_depth - start_depth;
    if (!(length > 1.0e-8f)) {
      continue;
    }
    const uint owner = uint(owner_raw);
    const uint rgba_base = owner * 4u;
    const float density = max(site_rgba_f32[rgba_base + 3u], 0.0f);
    const float seg_trans = exp(-density * length);
    const float seg_alpha = 1.0f - seg_trans;
    const float weight = transmittance * seg_alpha;
    const float3 rgb = float3(
        site_rgba_f32[rgba_base + 0u],
        site_rgba_f32[rgba_base + 1u],
        site_rgba_f32[rgba_base + 2u]);

    owners[local_run_count] = owner;
    lengths[local_run_count] = length;
    trans_before[local_run_count] = transmittance;
    segment_trans[local_run_count] = seg_trans;
    segment_alpha[local_run_count] = seg_alpha;
    weights[local_run_count] = weight;
    segment_rgb[local_run_count] = rgb;
    local_run_count += 1u;

    transmittance *= seg_trans;
  }

  const uint out_base = sample_id * 3u;
  const float3 grad_rgb = float3(
      grad_rgb_f32[out_base + 0u],
      grad_rgb_f32[out_base + 1u],
      grad_rgb_f32[out_base + 2u]);
  float adj_next_transmittance = 0.0f;
  for (int run_id = int(local_run_count) - 1; run_id >= 0; --run_id) {
    const uint owner = owners[run_id];
    const uint rgba_base = owner * 4u;
    const float raw_density = site_rgba_f32[rgba_base + 3u];
    const float adj_weight = dot(grad_rgb, segment_rgb[run_id]);
    const float adj_trans_before =
        adj_weight * segment_alpha[run_id] +
        adj_next_transmittance * segment_trans[run_id];
    const float adj_segment_alpha = adj_weight * trans_before[run_id];
    const float adj_segment_trans = adj_next_transmittance * trans_before[run_id] - adj_segment_alpha;
    float4 grad_rgba = float4(
        weights[run_id] * grad_rgb.x,
        weights[run_id] * grad_rgb.y,
        weights[run_id] * grad_rgb.z,
        0.0f);
    if (raw_density > 0.0f) {
      grad_rgba.w = adj_segment_trans * (-lengths[run_id] * segment_trans[run_id]);
    }
    wf2_atomic_add4(grad_site_rgba_f32, rgba_base, grad_rgba);
    adj_next_transmittance = adj_trans_before;
  }
}

kernel void wf2_endpoint_record_edit_block_coeff_vjp_direct_atomic_rgb_only_tensor(
    device const float* coeff_f32 [[buffer(0)]],
    device const float* frame_t_f32 [[buffer(1)]],
    device const int* anchor_offsets_i32 [[buffer(2)]],
    device const int* anchor_owner_i32 [[buffer(3)]],
    device const int* anchor_left_i32 [[buffer(4)]],
    device const int* anchor_right_i32 [[buffer(5)]],
    device const int* track_block_change_offsets_i32 [[buffer(6)]],
    device const int* change_frame_i32 [[buffer(7)]],
    device const int* op_offsets_i32 [[buffer(8)]],
    device const int* op_type_i32 [[buffer(9)]],
    device const int* op_pos_i32 [[buffer(10)]],
    device const int* op_owner_i32 [[buffer(11)]],
    device const int* op_left_i32 [[buffer(12)]],
    device const int* op_right_i32 [[buffer(13)]],
    device const float* site_rgba_f32 [[buffer(14)]],
    device const float* grad_rgb_f32 [[buffer(15)]],
    device const int* config_i32 [[buffer(16)]],
    device const float* config_f32 [[buffer(17)]],
    device atomic_float* grad_site_rgba_f32 [[buffer(18)]],
    uint sample_id [[thread_position_in_grid]]) {
  const uint boundary_count = uint(config_i32[0]);
  const uint track_count = uint(config_i32[1]);
  const uint frame_count = uint(config_i32[2]);
  const uint site_count = uint(config_i32[3]);
  const uint anchor_record_count = uint(config_i32[4]);
  const uint change_count = uint(config_i32[5]);
  const uint op_count = uint(config_i32[6]);
  const uint block_size = uint(config_i32[7]);
  const uint block_count = uint(config_i32[8]);
  const uint total_samples = track_count * frame_count;
  if (sample_id >= total_samples) {
    return;
  }

  const uint track_id = sample_id / frame_count;
  const uint frame_id = sample_id - track_id * frame_count;
  const float near_depth = config_f32[0];
  const float far_depth = config_f32[1];
  const float invalid_epsilon = config_f32[2];
  const float transmittance_threshold = config_f32[3];
  const float t = frame_t_f32[frame_id];

  int row_owner[WF2_MAX_REALRAY_SEGMENTS];
  int row_left[WF2_MAX_REALRAY_SEGMENTS];
  int row_right[WF2_MAX_REALRAY_SEGMENTS];
  uint row_count = 0u;
  if (!wf2_endpoint_record_load_block_edit_row(
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
          track_id,
          frame_id,
          block_size,
          block_count,
          anchor_record_count,
          change_count,
          op_count,
          row_owner,
          row_left,
          row_right,
          row_count)) {
    return;
  }

  uint owners[WF2_MAX_REALRAY_SEGMENTS];
  float lengths[WF2_MAX_REALRAY_SEGMENTS];
  float trans_before[WF2_MAX_REALRAY_SEGMENTS];
  float segment_trans[WF2_MAX_REALRAY_SEGMENTS];
  float segment_alpha[WF2_MAX_REALRAY_SEGMENTS];
  float weights[WF2_MAX_REALRAY_SEGMENTS];
  float3 segment_rgb[WF2_MAX_REALRAY_SEGMENTS];

  float transmittance = 1.0f;
  uint local_run_count = 0u;
  int cached_right_cut = -2147483648;
  float cached_right_depth = 0.0f;
  bool cached_right_valid = false;
  for (uint cursor = 0u; cursor < row_count; ++cursor) {
    if (transmittance <= transmittance_threshold || local_run_count >= WF2_MAX_REALRAY_SEGMENTS) {
      break;
    }
    const int owner_raw = row_owner[cursor];
    if (owner_raw < 0 || uint(owner_raw) >= site_count) {
      continue;
    }
    float start_depth = 0.0f;
    float end_depth = 0.0f;
    const int left_cut = row_left[cursor];
    const int right_cut = row_right[cursor];
    bool start_valid = false;
    if (cached_right_valid && left_cut == cached_right_cut) {
      start_depth = cached_right_depth;
      start_valid = true;
    } else {
      start_valid = wf2_endpoint_record_coeff_cut_depth(
          coeff_f32,
          boundary_count,
          track_id,
          left_cut,
          t,
          near_depth,
          far_depth,
          invalid_epsilon,
          start_depth);
    }
    if (!start_valid ||
        !wf2_endpoint_record_coeff_cut_depth(
            coeff_f32,
            boundary_count,
            track_id,
            right_cut,
            t,
            near_depth,
            far_depth,
            invalid_epsilon,
            end_depth)) {
      cached_right_valid = false;
      continue;
    }
    cached_right_cut = right_cut;
    cached_right_depth = end_depth;
    cached_right_valid = true;
    const float length = end_depth - start_depth;
    if (!(length > 1.0e-8f)) {
      continue;
    }
    const uint owner = uint(owner_raw);
    const uint rgba_base = owner * 4u;
    const float density = max(site_rgba_f32[rgba_base + 3u], 0.0f);
    const float seg_trans = exp(-density * length);
    const float seg_alpha = 1.0f - seg_trans;
    const float weight = transmittance * seg_alpha;
    const float3 rgb = float3(
        site_rgba_f32[rgba_base + 0u],
        site_rgba_f32[rgba_base + 1u],
        site_rgba_f32[rgba_base + 2u]);

    owners[local_run_count] = owner;
    lengths[local_run_count] = length;
    trans_before[local_run_count] = transmittance;
    segment_trans[local_run_count] = seg_trans;
    segment_alpha[local_run_count] = seg_alpha;
    weights[local_run_count] = weight;
    segment_rgb[local_run_count] = rgb;
    local_run_count += 1u;

    transmittance *= seg_trans;
  }

  const uint out_base = sample_id * 3u;
  const float3 grad_rgb = float3(
      grad_rgb_f32[out_base + 0u],
      grad_rgb_f32[out_base + 1u],
      grad_rgb_f32[out_base + 2u]);
  float adj_next_transmittance = 0.0f;
  for (int run_id = int(local_run_count) - 1; run_id >= 0; --run_id) {
    const uint owner = owners[run_id];
    const uint rgba_base = owner * 4u;
    const float raw_density = site_rgba_f32[rgba_base + 3u];
    const float adj_weight = dot(grad_rgb, segment_rgb[run_id]);
    const float adj_trans_before =
        adj_weight * segment_alpha[run_id] +
        adj_next_transmittance * segment_trans[run_id];
    const float adj_segment_alpha = adj_weight * trans_before[run_id];
    const float adj_segment_trans = adj_next_transmittance * trans_before[run_id] - adj_segment_alpha;
    float4 grad_rgba = float4(
        weights[run_id] * grad_rgb.x,
        weights[run_id] * grad_rgb.y,
        weights[run_id] * grad_rgb.z,
        0.0f);
    if (raw_density > 0.0f) {
      grad_rgba.w = adj_segment_trans * (-lengths[run_id] * segment_trans[run_id]);
    }
    wf2_atomic_add4(grad_site_rgba_f32, rgba_base, grad_rgba);
    adj_next_transmittance = adj_trans_before;
  }
}

kernel void wf2_endpoint_record_edit_block_coeff_mse_vjp_direct_atomic_rgb_only_tensor(
    device const float* coeff_f32 [[buffer(0)]],
    device const float* frame_t_f32 [[buffer(1)]],
    device const int* anchor_offsets_i32 [[buffer(2)]],
    device const int* anchor_owner_i32 [[buffer(3)]],
    device const int* anchor_left_i32 [[buffer(4)]],
    device const int* anchor_right_i32 [[buffer(5)]],
    device const int* track_block_change_offsets_i32 [[buffer(6)]],
    device const int* change_frame_i32 [[buffer(7)]],
    device const int* op_offsets_i32 [[buffer(8)]],
    device const int* op_type_i32 [[buffer(9)]],
    device const int* op_pos_i32 [[buffer(10)]],
    device const int* op_owner_i32 [[buffer(11)]],
    device const int* op_left_i32 [[buffer(12)]],
    device const int* op_right_i32 [[buffer(13)]],
    device const float* site_rgba_f32 [[buffer(14)]],
    device const float* target_rgb_f32 [[buffer(15)]],
    device const int* config_i32 [[buffer(16)]],
    device const float* config_f32 [[buffer(17)]],
    device atomic_float* loss_f32 [[buffer(18)]],
    device atomic_float* grad_site_rgba_f32 [[buffer(19)]],
    uint sample_id [[thread_position_in_grid]]) {
  const uint boundary_count = uint(config_i32[0]);
  const uint track_count = uint(config_i32[1]);
  const uint frame_count = uint(config_i32[2]);
  const uint site_count = uint(config_i32[3]);
  const uint anchor_record_count = uint(config_i32[4]);
  const uint change_count = uint(config_i32[5]);
  const uint op_count = uint(config_i32[6]);
  const uint block_size = uint(config_i32[7]);
  const uint block_count = uint(config_i32[8]);
  const uint total_samples = track_count * frame_count;
  if (sample_id >= total_samples) {
    return;
  }

  const uint track_id = sample_id / frame_count;
  const uint frame_id = sample_id - track_id * frame_count;
  const float near_depth = config_f32[0];
  const float far_depth = config_f32[1];
  const float invalid_epsilon = config_f32[2];
  const float transmittance_threshold = config_f32[3];
  const float t = frame_t_f32[frame_id];

  int row_owner[WF2_MAX_REALRAY_SEGMENTS];
  int row_left[WF2_MAX_REALRAY_SEGMENTS];
  int row_right[WF2_MAX_REALRAY_SEGMENTS];
  uint row_count = 0u;
  if (!wf2_endpoint_record_load_block_edit_row(
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
          track_id,
          frame_id,
          block_size,
          block_count,
          anchor_record_count,
          change_count,
          op_count,
          row_owner,
          row_left,
          row_right,
          row_count)) {
    return;
  }

  uint owners[WF2_MAX_REALRAY_SEGMENTS];
  float lengths[WF2_MAX_REALRAY_SEGMENTS];
  float trans_before[WF2_MAX_REALRAY_SEGMENTS];
  float segment_trans[WF2_MAX_REALRAY_SEGMENTS];
  float segment_alpha[WF2_MAX_REALRAY_SEGMENTS];
  float weights[WF2_MAX_REALRAY_SEGMENTS];
  float3 segment_rgb[WF2_MAX_REALRAY_SEGMENTS];

  float3 rgb_accum = float3(0.0f, 0.0f, 0.0f);
  float transmittance = 1.0f;
  uint local_run_count = 0u;
  int cached_right_cut = -2147483648;
  float cached_right_depth = 0.0f;
  bool cached_right_valid = false;
  for (uint cursor = 0u; cursor < row_count; ++cursor) {
    if (transmittance <= transmittance_threshold || local_run_count >= WF2_MAX_REALRAY_SEGMENTS) {
      break;
    }
    const int owner_raw = row_owner[cursor];
    if (owner_raw < 0 || uint(owner_raw) >= site_count) {
      continue;
    }
    float start_depth = 0.0f;
    float end_depth = 0.0f;
    const int left_cut = row_left[cursor];
    const int right_cut = row_right[cursor];
    bool start_valid = false;
    if (cached_right_valid && left_cut == cached_right_cut) {
      start_depth = cached_right_depth;
      start_valid = true;
    } else {
      start_valid = wf2_endpoint_record_coeff_cut_depth(
          coeff_f32,
          boundary_count,
          track_id,
          left_cut,
          t,
          near_depth,
          far_depth,
          invalid_epsilon,
          start_depth);
    }
    if (!start_valid ||
        !wf2_endpoint_record_coeff_cut_depth(
            coeff_f32,
            boundary_count,
            track_id,
            right_cut,
            t,
            near_depth,
            far_depth,
            invalid_epsilon,
            end_depth)) {
      cached_right_valid = false;
      continue;
    }
    cached_right_cut = right_cut;
    cached_right_depth = end_depth;
    cached_right_valid = true;
    const float length = end_depth - start_depth;
    if (!(length > 1.0e-8f)) {
      continue;
    }
    const uint owner = uint(owner_raw);
    const uint rgba_base = owner * 4u;
    const float density = max(site_rgba_f32[rgba_base + 3u], 0.0f);
    const float seg_trans = exp(-density * length);
    const float seg_alpha = 1.0f - seg_trans;
    const float weight = transmittance * seg_alpha;
    const float3 rgb = float3(
        site_rgba_f32[rgba_base + 0u],
        site_rgba_f32[rgba_base + 1u],
        site_rgba_f32[rgba_base + 2u]);

    owners[local_run_count] = owner;
    lengths[local_run_count] = length;
    trans_before[local_run_count] = transmittance;
    segment_trans[local_run_count] = seg_trans;
    segment_alpha[local_run_count] = seg_alpha;
    weights[local_run_count] = weight;
    segment_rgb[local_run_count] = rgb;
    local_run_count += 1u;

    rgb_accum += weight * rgb;
    transmittance *= seg_trans;
  }

  const uint out_base = sample_id * 3u;
  const float3 target_rgb = float3(
      target_rgb_f32[out_base + 0u],
      target_rgb_f32[out_base + 1u],
      target_rgb_f32[out_base + 2u]);
  const float3 diff = rgb_accum - target_rgb;
  const float inv_element_count = 1.0f / float(total_samples * 3u);
  const float sample_loss = dot(diff, diff) * inv_element_count;
  atomic_fetch_add_explicit(&loss_f32[0], sample_loss, memory_order_relaxed);
  const float3 grad_rgb = (2.0f * inv_element_count) * diff;

  float adj_next_transmittance = 0.0f;
  for (int run_id = int(local_run_count) - 1; run_id >= 0; --run_id) {
    const uint owner = owners[run_id];
    const uint rgba_base = owner * 4u;
    const float raw_density = site_rgba_f32[rgba_base + 3u];
    const float adj_weight = dot(grad_rgb, segment_rgb[run_id]);
    const float adj_trans_before =
        adj_weight * segment_alpha[run_id] +
        adj_next_transmittance * segment_trans[run_id];
    const float adj_segment_alpha = adj_weight * trans_before[run_id];
    const float adj_segment_trans = adj_next_transmittance * trans_before[run_id] - adj_segment_alpha;
    float4 grad_rgba = float4(
        weights[run_id] * grad_rgb.x,
        weights[run_id] * grad_rgb.y,
        weights[run_id] * grad_rgb.z,
        0.0f);
    if (raw_density > 0.0f) {
      grad_rgba.w = adj_segment_trans * (-lengths[run_id] * segment_trans[run_id]);
    }
    wf2_atomic_add4(grad_site_rgba_f32, rgba_base, grad_rgba);
    adj_next_transmittance = adj_trans_before;
  }
}

kernel void wf2_endpoint_record_edit_block_coeff16_mse_vjp_direct_atomic_rgb_only_tensor(
    device const half* coeff_f16 [[buffer(0)]],
    device const float* frame_t_f32 [[buffer(1)]],
    device const int* anchor_offsets_i32 [[buffer(2)]],
    device const int* anchor_owner_i32 [[buffer(3)]],
    device const int* anchor_left_i32 [[buffer(4)]],
    device const int* anchor_right_i32 [[buffer(5)]],
    device const int* track_block_change_offsets_i32 [[buffer(6)]],
    device const int* change_frame_i32 [[buffer(7)]],
    device const int* op_offsets_i32 [[buffer(8)]],
    device const int* op_type_i32 [[buffer(9)]],
    device const int* op_pos_i32 [[buffer(10)]],
    device const int* op_owner_i32 [[buffer(11)]],
    device const int* op_left_i32 [[buffer(12)]],
    device const int* op_right_i32 [[buffer(13)]],
    device const float* site_rgba_f32 [[buffer(14)]],
    device const float* target_rgb_f32 [[buffer(15)]],
    device const int* config_i32 [[buffer(16)]],
    device const float* config_f32 [[buffer(17)]],
    device atomic_float* loss_f32 [[buffer(18)]],
    device atomic_float* grad_site_rgba_f32 [[buffer(19)]],
    uint sample_id [[thread_position_in_grid]]) {
  const uint boundary_count = uint(config_i32[0]);
  const uint track_count = uint(config_i32[1]);
  const uint frame_count = uint(config_i32[2]);
  const uint site_count = uint(config_i32[3]);
  const uint anchor_record_count = uint(config_i32[4]);
  const uint change_count = uint(config_i32[5]);
  const uint op_count = uint(config_i32[6]);
  const uint block_size = uint(config_i32[7]);
  const uint block_count = uint(config_i32[8]);
  const uint total_samples = track_count * frame_count;
  if (sample_id >= total_samples) {
    return;
  }

  const uint track_id = sample_id / frame_count;
  const uint frame_id = sample_id - track_id * frame_count;
  const float near_depth = config_f32[0];
  const float far_depth = config_f32[1];
  const float invalid_epsilon = config_f32[2];
  const float transmittance_threshold = config_f32[3];
  const float t = frame_t_f32[frame_id];

  int row_owner[WF2_MAX_REALRAY_SEGMENTS];
  int row_left[WF2_MAX_REALRAY_SEGMENTS];
  int row_right[WF2_MAX_REALRAY_SEGMENTS];
  uint row_count = 0u;
  if (!wf2_endpoint_record_load_block_edit_row(
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
          track_id,
          frame_id,
          block_size,
          block_count,
          anchor_record_count,
          change_count,
          op_count,
          row_owner,
          row_left,
          row_right,
          row_count)) {
    return;
  }

  uint owners[WF2_MAX_REALRAY_SEGMENTS];
  float lengths[WF2_MAX_REALRAY_SEGMENTS];
  float trans_before[WF2_MAX_REALRAY_SEGMENTS];
  float segment_trans[WF2_MAX_REALRAY_SEGMENTS];
  float segment_alpha[WF2_MAX_REALRAY_SEGMENTS];
  float weights[WF2_MAX_REALRAY_SEGMENTS];
  float3 segment_rgb[WF2_MAX_REALRAY_SEGMENTS];

  float3 rgb_accum = float3(0.0f, 0.0f, 0.0f);
  float transmittance = 1.0f;
  uint local_run_count = 0u;
  int cached_right_cut = -2147483648;
  float cached_right_depth = 0.0f;
  bool cached_right_valid = false;
  for (uint cursor = 0u; cursor < row_count; ++cursor) {
    if (transmittance <= transmittance_threshold || local_run_count >= WF2_MAX_REALRAY_SEGMENTS) {
      break;
    }
    const int owner_raw = row_owner[cursor];
    if (owner_raw < 0 || uint(owner_raw) >= site_count) {
      continue;
    }
    float start_depth = 0.0f;
    float end_depth = 0.0f;
    const int left_cut = row_left[cursor];
    const int right_cut = row_right[cursor];
    bool start_valid = false;
    if (cached_right_valid && left_cut == cached_right_cut) {
      start_depth = cached_right_depth;
      start_valid = true;
    } else {
      start_valid = wf2_endpoint_record_coeff16_cut_depth(
          coeff_f16,
          boundary_count,
          track_id,
          left_cut,
          t,
          near_depth,
          far_depth,
          invalid_epsilon,
          start_depth);
    }
    if (!start_valid ||
        !wf2_endpoint_record_coeff16_cut_depth(
            coeff_f16,
            boundary_count,
            track_id,
            right_cut,
            t,
            near_depth,
            far_depth,
            invalid_epsilon,
            end_depth)) {
      cached_right_valid = false;
      continue;
    }
    cached_right_cut = right_cut;
    cached_right_depth = end_depth;
    cached_right_valid = true;
    const float length = end_depth - start_depth;
    if (!(length > 1.0e-8f)) {
      continue;
    }
    const uint owner = uint(owner_raw);
    const uint rgba_base = owner * 4u;
    const float density = max(site_rgba_f32[rgba_base + 3u], 0.0f);
    const float seg_trans = exp(-density * length);
    const float seg_alpha = 1.0f - seg_trans;
    const float weight = transmittance * seg_alpha;
    const float3 rgb = float3(
        site_rgba_f32[rgba_base + 0u],
        site_rgba_f32[rgba_base + 1u],
        site_rgba_f32[rgba_base + 2u]);

    owners[local_run_count] = owner;
    lengths[local_run_count] = length;
    trans_before[local_run_count] = transmittance;
    segment_trans[local_run_count] = seg_trans;
    segment_alpha[local_run_count] = seg_alpha;
    weights[local_run_count] = weight;
    segment_rgb[local_run_count] = rgb;
    local_run_count += 1u;

    rgb_accum += weight * rgb;
    transmittance *= seg_trans;
  }

  const uint out_base = sample_id * 3u;
  const float3 target_rgb = float3(
      target_rgb_f32[out_base + 0u],
      target_rgb_f32[out_base + 1u],
      target_rgb_f32[out_base + 2u]);
  const float3 diff = rgb_accum - target_rgb;
  const float inv_element_count = 1.0f / float(total_samples * 3u);
  const float sample_loss = dot(diff, diff) * inv_element_count;
  atomic_fetch_add_explicit(&loss_f32[0], sample_loss, memory_order_relaxed);
  const float3 grad_rgb = (2.0f * inv_element_count) * diff;

  float adj_next_transmittance = 0.0f;
  for (int run_id = int(local_run_count) - 1; run_id >= 0; --run_id) {
    const uint owner = owners[run_id];
    const uint rgba_base = owner * 4u;
    const float raw_density = site_rgba_f32[rgba_base + 3u];
    const float adj_weight = dot(grad_rgb, segment_rgb[run_id]);
    const float adj_trans_before =
        adj_weight * segment_alpha[run_id] +
        adj_next_transmittance * segment_trans[run_id];
    const float adj_segment_alpha = adj_weight * trans_before[run_id];
    const float adj_segment_trans = adj_next_transmittance * trans_before[run_id] - adj_segment_alpha;
    float4 grad_rgba = float4(
        weights[run_id] * grad_rgb.x,
        weights[run_id] * grad_rgb.y,
        weights[run_id] * grad_rgb.z,
        0.0f);
    if (raw_density > 0.0f) {
      grad_rgba.w = adj_segment_trans * (-lengths[run_id] * segment_trans[run_id]);
    }
    wf2_atomic_add4(grad_site_rgba_f32, rgba_base, grad_rgba);
    adj_next_transmittance = adj_trans_before;
  }
}

kernel void wf2_endpoint_record_edit_block_coeff16_packed_mse_vjp_direct_atomic_rgb_only_tensor(
    device const half* coeff_f16 [[buffer(0)]],
    device const float* frame_t_f32 [[buffer(1)]],
    device const int* anchor_offsets_i32 [[buffer(2)]],
    device const int* anchor_record_i32 [[buffer(3)]],
    device const int* track_block_change_offsets_i32 [[buffer(4)]],
    device const int* change_frame_i32 [[buffer(5)]],
    device const int* op_offsets_i32 [[buffer(6)]],
    device const int* op_type_i32 [[buffer(7)]],
    device const int* op_pos_i32 [[buffer(8)]],
    device const int* op_record_i32 [[buffer(9)]],
    device const float* site_rgba_f32 [[buffer(10)]],
    device const float* target_rgb_f32 [[buffer(11)]],
    device const int* config_i32 [[buffer(12)]],
    device const float* config_f32 [[buffer(13)]],
    device atomic_float* loss_f32 [[buffer(14)]],
    device atomic_float* grad_site_rgba_f32 [[buffer(15)]],
    uint sample_id [[thread_position_in_grid]]) {
  const uint boundary_count = uint(config_i32[0]);
  const uint track_count = uint(config_i32[1]);
  const uint frame_count = uint(config_i32[2]);
  const uint site_count = uint(config_i32[3]);
  const uint anchor_record_count = uint(config_i32[4]);
  const uint change_count = uint(config_i32[5]);
  const uint op_count = uint(config_i32[6]);
  const uint block_size = uint(config_i32[7]);
  const uint block_count = uint(config_i32[8]);
  const uint total_samples = track_count * frame_count;
  if (sample_id >= total_samples) {
    return;
  }

  const uint track_id = sample_id / frame_count;
  const uint frame_id = sample_id - track_id * frame_count;
  const float near_depth = config_f32[0];
  const float far_depth = config_f32[1];
  const float invalid_epsilon = config_f32[2];
  const float transmittance_threshold = config_f32[3];
  const float t = frame_t_f32[frame_id];

  int row_owner[WF2_MAX_REALRAY_SEGMENTS];
  int row_left[WF2_MAX_REALRAY_SEGMENTS];
  int row_right[WF2_MAX_REALRAY_SEGMENTS];
  uint row_count = 0u;
  if (!wf2_endpoint_record_load_block_edit_row_packed(
          anchor_offsets_i32,
          anchor_record_i32,
          track_block_change_offsets_i32,
          change_frame_i32,
          op_offsets_i32,
          op_type_i32,
          op_pos_i32,
          op_record_i32,
          track_id,
          frame_id,
          block_size,
          block_count,
          anchor_record_count,
          change_count,
          op_count,
          row_owner,
          row_left,
          row_right,
          row_count)) {
    return;
  }

  uint owners[WF2_MAX_REALRAY_SEGMENTS];
  float lengths[WF2_MAX_REALRAY_SEGMENTS];
  float trans_before[WF2_MAX_REALRAY_SEGMENTS];
  float segment_trans[WF2_MAX_REALRAY_SEGMENTS];
  float segment_alpha[WF2_MAX_REALRAY_SEGMENTS];
  float weights[WF2_MAX_REALRAY_SEGMENTS];
  float3 segment_rgb[WF2_MAX_REALRAY_SEGMENTS];

  float3 rgb_accum = float3(0.0f, 0.0f, 0.0f);
  float transmittance = 1.0f;
  uint local_run_count = 0u;
  int cached_right_cut = -2147483648;
  float cached_right_depth = 0.0f;
  bool cached_right_valid = false;
  for (uint cursor = 0u; cursor < row_count; ++cursor) {
    if (transmittance <= transmittance_threshold || local_run_count >= WF2_MAX_REALRAY_SEGMENTS) {
      break;
    }
    const int owner_raw = row_owner[cursor];
    if (owner_raw < 0 || uint(owner_raw) >= site_count) {
      continue;
    }
    float start_depth = 0.0f;
    float end_depth = 0.0f;
    const int left_cut = row_left[cursor];
    const int right_cut = row_right[cursor];
    bool start_valid = false;
    if (cached_right_valid && left_cut == cached_right_cut) {
      start_depth = cached_right_depth;
      start_valid = true;
    } else {
      start_valid = wf2_endpoint_record_coeff16_cut_depth(
          coeff_f16,
          boundary_count,
          track_id,
          left_cut,
          t,
          near_depth,
          far_depth,
          invalid_epsilon,
          start_depth);
    }
    if (!start_valid ||
        !wf2_endpoint_record_coeff16_cut_depth(
            coeff_f16,
            boundary_count,
            track_id,
            right_cut,
            t,
            near_depth,
            far_depth,
            invalid_epsilon,
            end_depth)) {
      cached_right_valid = false;
      continue;
    }
    cached_right_cut = right_cut;
    cached_right_depth = end_depth;
    cached_right_valid = true;
    const float length = end_depth - start_depth;
    if (!(length > 1.0e-8f)) {
      continue;
    }
    const uint owner = uint(owner_raw);
    const uint rgba_base = owner * 4u;
    const float density = max(site_rgba_f32[rgba_base + 3u], 0.0f);
    const float seg_trans = exp(-density * length);
    const float seg_alpha = 1.0f - seg_trans;
    const float weight = transmittance * seg_alpha;
    const float3 rgb = float3(
        site_rgba_f32[rgba_base + 0u],
        site_rgba_f32[rgba_base + 1u],
        site_rgba_f32[rgba_base + 2u]);

    owners[local_run_count] = owner;
    lengths[local_run_count] = length;
    trans_before[local_run_count] = transmittance;
    segment_trans[local_run_count] = seg_trans;
    segment_alpha[local_run_count] = seg_alpha;
    weights[local_run_count] = weight;
    segment_rgb[local_run_count] = rgb;
    local_run_count += 1u;

    rgb_accum += weight * rgb;
    transmittance *= seg_trans;
  }

  const uint out_base = sample_id * 3u;
  const float3 target_rgb = float3(
      target_rgb_f32[out_base + 0u],
      target_rgb_f32[out_base + 1u],
      target_rgb_f32[out_base + 2u]);
  const float3 diff = rgb_accum - target_rgb;
  const float inv_element_count = 1.0f / float(total_samples * 3u);
  const float sample_loss = dot(diff, diff) * inv_element_count;
  atomic_fetch_add_explicit(&loss_f32[0], sample_loss, memory_order_relaxed);
  const float3 grad_rgb = (2.0f * inv_element_count) * diff;

  float adj_next_transmittance = 0.0f;
  for (int run_id = int(local_run_count) - 1; run_id >= 0; --run_id) {
    const uint owner = owners[run_id];
    const uint rgba_base = owner * 4u;
    const float raw_density = site_rgba_f32[rgba_base + 3u];
    const float adj_weight = dot(grad_rgb, segment_rgb[run_id]);
    const float adj_trans_before =
        adj_weight * segment_alpha[run_id] +
        adj_next_transmittance * segment_trans[run_id];
    const float adj_segment_alpha = adj_weight * trans_before[run_id];
    const float adj_segment_trans = adj_next_transmittance * trans_before[run_id] - adj_segment_alpha;
    float4 grad_rgba = float4(
        weights[run_id] * grad_rgb.x,
        weights[run_id] * grad_rgb.y,
        weights[run_id] * grad_rgb.z,
        0.0f);
    if (raw_density > 0.0f) {
      grad_rgba.w = adj_segment_trans * (-lengths[run_id] * segment_trans[run_id]);
    }
    wf2_atomic_add4(grad_site_rgba_f32, rgba_base, grad_rgba);
    adj_next_transmittance = adj_trans_before;
  }
}

kernel void wf2_endpoint_record_edit_block_coeff16_i16_mse_vjp_direct_atomic_rgb_only_tensor(
    device const half* coeff_f16 [[buffer(0)]],
    device const float* frame_t_f32 [[buffer(1)]],
    device const int* anchor_offsets_i32 [[buffer(2)]],
    device const short* anchor_owner_i16 [[buffer(3)]],
    device const short* anchor_left_i16 [[buffer(4)]],
    device const short* anchor_right_i16 [[buffer(5)]],
    device const int* track_block_change_offsets_i32 [[buffer(6)]],
    device const int* change_frame_i32 [[buffer(7)]],
    device const int* op_offsets_i32 [[buffer(8)]],
    device const int* op_type_i32 [[buffer(9)]],
    device const int* op_pos_i32 [[buffer(10)]],
    device const short* op_owner_i16 [[buffer(11)]],
    device const short* op_left_i16 [[buffer(12)]],
    device const short* op_right_i16 [[buffer(13)]],
    device const float* site_rgba_f32 [[buffer(14)]],
    device const float* target_rgb_f32 [[buffer(15)]],
    device const int* config_i32 [[buffer(16)]],
    device const float* config_f32 [[buffer(17)]],
    device atomic_float* loss_f32 [[buffer(18)]],
    device atomic_float* grad_site_rgba_f32 [[buffer(19)]],
    uint sample_id [[thread_position_in_grid]]) {
  const uint boundary_count = uint(config_i32[0]);
  const uint track_count = uint(config_i32[1]);
  const uint frame_count = uint(config_i32[2]);
  const uint site_count = uint(config_i32[3]);
  const uint anchor_record_count = uint(config_i32[4]);
  const uint change_count = uint(config_i32[5]);
  const uint op_count = uint(config_i32[6]);
  const uint block_size = uint(config_i32[7]);
  const uint block_count = uint(config_i32[8]);
  const uint total_samples = track_count * frame_count;
  if (sample_id >= total_samples) {
    return;
  }

  const uint track_id = sample_id / frame_count;
  const uint frame_id = sample_id - track_id * frame_count;
  const float near_depth = config_f32[0];
  const float far_depth = config_f32[1];
  const float invalid_epsilon = config_f32[2];
  const float transmittance_threshold = config_f32[3];
  const float t = frame_t_f32[frame_id];

  int row_owner[WF2_MAX_REALRAY_SEGMENTS];
  int row_left[WF2_MAX_REALRAY_SEGMENTS];
  int row_right[WF2_MAX_REALRAY_SEGMENTS];
  uint row_count = 0u;
  if (!wf2_endpoint_record_load_block_edit_row_i16(
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
          track_id,
          frame_id,
          block_size,
          block_count,
          anchor_record_count,
          change_count,
          op_count,
          row_owner,
          row_left,
          row_right,
          row_count)) {
    return;
  }

  uint owners[WF2_MAX_REALRAY_SEGMENTS];
  float lengths[WF2_MAX_REALRAY_SEGMENTS];
  float trans_before[WF2_MAX_REALRAY_SEGMENTS];
  float segment_trans[WF2_MAX_REALRAY_SEGMENTS];
  float segment_alpha[WF2_MAX_REALRAY_SEGMENTS];
  float weights[WF2_MAX_REALRAY_SEGMENTS];
  float3 segment_rgb[WF2_MAX_REALRAY_SEGMENTS];

  float3 rgb_accum = float3(0.0f, 0.0f, 0.0f);
  float transmittance = 1.0f;
  uint local_run_count = 0u;
  int cached_right_cut = -2147483648;
  float cached_right_depth = 0.0f;
  bool cached_right_valid = false;
  for (uint cursor = 0u; cursor < row_count; ++cursor) {
    if (transmittance <= transmittance_threshold || local_run_count >= WF2_MAX_REALRAY_SEGMENTS) {
      break;
    }
    const int owner_raw = row_owner[cursor];
    if (owner_raw < 0 || uint(owner_raw) >= site_count) {
      continue;
    }
    float start_depth = 0.0f;
    float end_depth = 0.0f;
    const int left_cut = row_left[cursor];
    const int right_cut = row_right[cursor];
    bool start_valid = false;
    if (cached_right_valid && left_cut == cached_right_cut) {
      start_depth = cached_right_depth;
      start_valid = true;
    } else {
      start_valid = wf2_endpoint_record_coeff16_cut_depth(
          coeff_f16,
          boundary_count,
          track_id,
          left_cut,
          t,
          near_depth,
          far_depth,
          invalid_epsilon,
          start_depth);
    }
    if (!start_valid ||
        !wf2_endpoint_record_coeff16_cut_depth(
            coeff_f16,
            boundary_count,
            track_id,
            right_cut,
            t,
            near_depth,
            far_depth,
            invalid_epsilon,
            end_depth)) {
      cached_right_valid = false;
      continue;
    }
    cached_right_cut = right_cut;
    cached_right_depth = end_depth;
    cached_right_valid = true;
    const float length = end_depth - start_depth;
    if (!(length > 1.0e-8f)) {
      continue;
    }
    const uint owner = uint(owner_raw);
    const uint rgba_base = owner * 4u;
    const float density = max(site_rgba_f32[rgba_base + 3u], 0.0f);
    const float seg_trans = exp(-density * length);
    const float seg_alpha = 1.0f - seg_trans;
    const float weight = transmittance * seg_alpha;
    const float3 rgb = float3(
        site_rgba_f32[rgba_base + 0u],
        site_rgba_f32[rgba_base + 1u],
        site_rgba_f32[rgba_base + 2u]);

    owners[local_run_count] = owner;
    lengths[local_run_count] = length;
    trans_before[local_run_count] = transmittance;
    segment_trans[local_run_count] = seg_trans;
    segment_alpha[local_run_count] = seg_alpha;
    weights[local_run_count] = weight;
    segment_rgb[local_run_count] = rgb;
    local_run_count += 1u;

    rgb_accum += weight * rgb;
    transmittance *= seg_trans;
  }

  const uint out_base = sample_id * 3u;
  const float3 target_rgb = float3(
      target_rgb_f32[out_base + 0u],
      target_rgb_f32[out_base + 1u],
      target_rgb_f32[out_base + 2u]);
  const float3 diff = rgb_accum - target_rgb;
  const float inv_element_count = 1.0f / float(total_samples * 3u);
  const float sample_loss = dot(diff, diff) * inv_element_count;
  atomic_fetch_add_explicit(&loss_f32[0], sample_loss, memory_order_relaxed);
  const float3 grad_rgb = (2.0f * inv_element_count) * diff;

  float adj_next_transmittance = 0.0f;
  for (int run_id = int(local_run_count) - 1; run_id >= 0; --run_id) {
    const uint owner = owners[run_id];
    const uint rgba_base = owner * 4u;
    const float raw_density = site_rgba_f32[rgba_base + 3u];
    const float adj_weight = dot(grad_rgb, segment_rgb[run_id]);
    const float adj_trans_before =
        adj_weight * segment_alpha[run_id] +
        adj_next_transmittance * segment_trans[run_id];
    const float adj_segment_alpha = adj_weight * trans_before[run_id];
    const float adj_segment_trans = adj_next_transmittance * trans_before[run_id] - adj_segment_alpha;
    float4 grad_rgba = float4(
        weights[run_id] * grad_rgb.x,
        weights[run_id] * grad_rgb.y,
        weights[run_id] * grad_rgb.z,
        0.0f);
    if (raw_density > 0.0f) {
      grad_rgba.w = adj_segment_trans * (-lengths[run_id] * segment_trans[run_id]);
    }
    wf2_atomic_add4(grad_site_rgba_f32, rgba_base, grad_rgba);
    adj_next_transmittance = adj_trans_before;
  }
}

kernel void wf2_endpoint_record_edit_block_coeff16_i16x3_mse_vjp_direct_atomic_rgb_only_tensor(
    device const half* coeff_f16 [[buffer(0)]],
    device const float* frame_t_f32 [[buffer(1)]],
    device const int* anchor_offsets_i32 [[buffer(2)]],
    device const short* anchor_record_i16 [[buffer(3)]],
    device const int* track_block_change_offsets_i32 [[buffer(4)]],
    device const int* change_frame_i32 [[buffer(5)]],
    device const int* op_offsets_i32 [[buffer(6)]],
    device const int* op_type_i32 [[buffer(7)]],
    device const int* op_pos_i32 [[buffer(8)]],
    device const short* op_record_i16 [[buffer(9)]],
    device const float* site_rgba_f32 [[buffer(10)]],
    device const float* target_rgb_f32 [[buffer(11)]],
    device const int* config_i32 [[buffer(12)]],
    device const float* config_f32 [[buffer(13)]],
    device atomic_float* loss_f32 [[buffer(14)]],
    device atomic_float* grad_site_rgba_f32 [[buffer(15)]],
    uint sample_id [[thread_position_in_grid]]) {
  const uint boundary_count = uint(config_i32[0]);
  const uint track_count = uint(config_i32[1]);
  const uint frame_count = uint(config_i32[2]);
  const uint site_count = uint(config_i32[3]);
  const uint anchor_record_count = uint(config_i32[4]);
  const uint change_count = uint(config_i32[5]);
  const uint op_count = uint(config_i32[6]);
  const uint block_size = uint(config_i32[7]);
  const uint block_count = uint(config_i32[8]);
  const uint total_samples = track_count * frame_count;
  if (sample_id >= total_samples) {
    return;
  }

  const uint track_id = sample_id / frame_count;
  const uint frame_id = sample_id - track_id * frame_count;
  const float near_depth = config_f32[0];
  const float far_depth = config_f32[1];
  const float invalid_epsilon = config_f32[2];
  const float transmittance_threshold = config_f32[3];
  const float t = frame_t_f32[frame_id];

  int row_owner[WF2_MAX_REALRAY_SEGMENTS];
  int row_left[WF2_MAX_REALRAY_SEGMENTS];
  int row_right[WF2_MAX_REALRAY_SEGMENTS];
  uint row_count = 0u;
  if (!wf2_endpoint_record_load_block_edit_row_i16x3(
          anchor_offsets_i32,
          anchor_record_i16,
          track_block_change_offsets_i32,
          change_frame_i32,
          op_offsets_i32,
          op_type_i32,
          op_pos_i32,
          op_record_i16,
          track_id,
          frame_id,
          block_size,
          block_count,
          anchor_record_count,
          change_count,
          op_count,
          row_owner,
          row_left,
          row_right,
          row_count)) {
    return;
  }

  uint owners[WF2_MAX_REALRAY_SEGMENTS];
  float lengths[WF2_MAX_REALRAY_SEGMENTS];
  float trans_before[WF2_MAX_REALRAY_SEGMENTS];
  float segment_trans[WF2_MAX_REALRAY_SEGMENTS];
  float segment_alpha[WF2_MAX_REALRAY_SEGMENTS];
  float weights[WF2_MAX_REALRAY_SEGMENTS];
  float3 segment_rgb[WF2_MAX_REALRAY_SEGMENTS];

  float3 rgb_accum = float3(0.0f, 0.0f, 0.0f);
  float transmittance = 1.0f;
  uint local_run_count = 0u;
  int cached_right_cut = -2147483648;
  float cached_right_depth = 0.0f;
  bool cached_right_valid = false;
  for (uint cursor = 0u; cursor < row_count; ++cursor) {
    if (transmittance <= transmittance_threshold || local_run_count >= WF2_MAX_REALRAY_SEGMENTS) {
      break;
    }
    const int owner_raw = row_owner[cursor];
    if (owner_raw < 0 || uint(owner_raw) >= site_count) {
      continue;
    }
    float start_depth = 0.0f;
    float end_depth = 0.0f;
    const int left_cut = row_left[cursor];
    const int right_cut = row_right[cursor];
    bool start_valid = false;
    if (cached_right_valid && left_cut == cached_right_cut) {
      start_depth = cached_right_depth;
      start_valid = true;
    } else {
      start_valid = wf2_endpoint_record_coeff16_cut_depth(
          coeff_f16,
          boundary_count,
          track_id,
          left_cut,
          t,
          near_depth,
          far_depth,
          invalid_epsilon,
          start_depth);
    }
    if (!start_valid ||
        !wf2_endpoint_record_coeff16_cut_depth(
            coeff_f16,
            boundary_count,
            track_id,
            right_cut,
            t,
            near_depth,
            far_depth,
            invalid_epsilon,
            end_depth)) {
      cached_right_valid = false;
      continue;
    }
    cached_right_cut = right_cut;
    cached_right_depth = end_depth;
    cached_right_valid = true;
    const float length = end_depth - start_depth;
    if (!(length > 1.0e-8f)) {
      continue;
    }
    const uint owner = uint(owner_raw);
    const uint rgba_base = owner * 4u;
    const float density = max(site_rgba_f32[rgba_base + 3u], 0.0f);
    const float seg_trans = exp(-density * length);
    const float seg_alpha = 1.0f - seg_trans;
    const float weight = transmittance * seg_alpha;
    const float3 rgb = float3(
        site_rgba_f32[rgba_base + 0u],
        site_rgba_f32[rgba_base + 1u],
        site_rgba_f32[rgba_base + 2u]);

    owners[local_run_count] = owner;
    lengths[local_run_count] = length;
    trans_before[local_run_count] = transmittance;
    segment_trans[local_run_count] = seg_trans;
    segment_alpha[local_run_count] = seg_alpha;
    weights[local_run_count] = weight;
    segment_rgb[local_run_count] = rgb;
    local_run_count += 1u;

    rgb_accum += weight * rgb;
    transmittance *= seg_trans;
  }

  const uint out_base = sample_id * 3u;
  const float3 target_rgb = float3(
      target_rgb_f32[out_base + 0u],
      target_rgb_f32[out_base + 1u],
      target_rgb_f32[out_base + 2u]);
  const float3 diff = rgb_accum - target_rgb;
  const float inv_element_count = 1.0f / float(total_samples * 3u);
  const float sample_loss = dot(diff, diff) * inv_element_count;
  atomic_fetch_add_explicit(&loss_f32[0], sample_loss, memory_order_relaxed);
  const float3 grad_rgb = (2.0f * inv_element_count) * diff;

  float adj_next_transmittance = 0.0f;
  for (int run_id = int(local_run_count) - 1; run_id >= 0; --run_id) {
    const uint owner = owners[run_id];
    const uint rgba_base = owner * 4u;
    const float raw_density = site_rgba_f32[rgba_base + 3u];
    const float adj_weight = dot(grad_rgb, segment_rgb[run_id]);
    const float adj_trans_before =
        adj_weight * segment_alpha[run_id] +
        adj_next_transmittance * segment_trans[run_id];
    const float adj_segment_alpha = adj_weight * trans_before[run_id];
    const float adj_segment_trans = adj_next_transmittance * trans_before[run_id] - adj_segment_alpha;
    float4 grad_rgba = float4(
        weights[run_id] * grad_rgb.x,
        weights[run_id] * grad_rgb.y,
        weights[run_id] * grad_rgb.z,
        0.0f);
    if (raw_density > 0.0f) {
      grad_rgba.w = adj_segment_trans * (-lengths[run_id] * segment_trans[run_id]);
    }
    wf2_atomic_add4(grad_site_rgba_f32, rgba_base, grad_rgba);
    adj_next_transmittance = adj_trans_before;
  }
}

kernel void wf2_endpoint_record_edit_block_coeff16_vjp_direct_atomic_rgb_only_tensor(
    device const half* coeff_f16 [[buffer(0)]],
    device const float* frame_t_f32 [[buffer(1)]],
    device const int* anchor_offsets_i32 [[buffer(2)]],
    device const int* anchor_owner_i32 [[buffer(3)]],
    device const int* anchor_left_i32 [[buffer(4)]],
    device const int* anchor_right_i32 [[buffer(5)]],
    device const int* track_block_change_offsets_i32 [[buffer(6)]],
    device const int* change_frame_i32 [[buffer(7)]],
    device const int* op_offsets_i32 [[buffer(8)]],
    device const int* op_type_i32 [[buffer(9)]],
    device const int* op_pos_i32 [[buffer(10)]],
    device const int* op_owner_i32 [[buffer(11)]],
    device const int* op_left_i32 [[buffer(12)]],
    device const int* op_right_i32 [[buffer(13)]],
    device const float* site_rgba_f32 [[buffer(14)]],
    device const float* grad_rgb_f32 [[buffer(15)]],
    device const int* config_i32 [[buffer(16)]],
    device const float* config_f32 [[buffer(17)]],
    device atomic_float* grad_site_rgba_f32 [[buffer(18)]],
    uint sample_id [[thread_position_in_grid]]) {
  const uint boundary_count = uint(config_i32[0]);
  const uint track_count = uint(config_i32[1]);
  const uint frame_count = uint(config_i32[2]);
  const uint site_count = uint(config_i32[3]);
  const uint anchor_record_count = uint(config_i32[4]);
  const uint change_count = uint(config_i32[5]);
  const uint op_count = uint(config_i32[6]);
  const uint block_size = uint(config_i32[7]);
  const uint block_count = uint(config_i32[8]);
  const uint total_samples = track_count * frame_count;
  if (sample_id >= total_samples) {
    return;
  }

  const uint track_id = sample_id / frame_count;
  const uint frame_id = sample_id - track_id * frame_count;
  const float near_depth = config_f32[0];
  const float far_depth = config_f32[1];
  const float invalid_epsilon = config_f32[2];
  const float transmittance_threshold = config_f32[3];
  const float t = frame_t_f32[frame_id];

  int row_owner[WF2_MAX_REALRAY_SEGMENTS];
  int row_left[WF2_MAX_REALRAY_SEGMENTS];
  int row_right[WF2_MAX_REALRAY_SEGMENTS];
  uint row_count = 0u;
  if (!wf2_endpoint_record_load_block_edit_row(
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
          track_id,
          frame_id,
          block_size,
          block_count,
          anchor_record_count,
          change_count,
          op_count,
          row_owner,
          row_left,
          row_right,
          row_count)) {
    return;
  }

  uint owners[WF2_MAX_REALRAY_SEGMENTS];
  float lengths[WF2_MAX_REALRAY_SEGMENTS];
  float trans_before[WF2_MAX_REALRAY_SEGMENTS];
  float segment_trans[WF2_MAX_REALRAY_SEGMENTS];
  float segment_alpha[WF2_MAX_REALRAY_SEGMENTS];
  float weights[WF2_MAX_REALRAY_SEGMENTS];
  float3 segment_rgb[WF2_MAX_REALRAY_SEGMENTS];

  float transmittance = 1.0f;
  uint local_run_count = 0u;
  int cached_right_cut = -2147483648;
  float cached_right_depth = 0.0f;
  bool cached_right_valid = false;
  for (uint cursor = 0u; cursor < row_count; ++cursor) {
    if (transmittance <= transmittance_threshold || local_run_count >= WF2_MAX_REALRAY_SEGMENTS) {
      break;
    }
    const int owner_raw = row_owner[cursor];
    if (owner_raw < 0 || uint(owner_raw) >= site_count) {
      continue;
    }
    float start_depth = 0.0f;
    float end_depth = 0.0f;
    const int left_cut = row_left[cursor];
    const int right_cut = row_right[cursor];
    bool start_valid = false;
    if (cached_right_valid && left_cut == cached_right_cut) {
      start_depth = cached_right_depth;
      start_valid = true;
    } else {
      start_valid = wf2_endpoint_record_coeff16_cut_depth(
          coeff_f16,
          boundary_count,
          track_id,
          left_cut,
          t,
          near_depth,
          far_depth,
          invalid_epsilon,
          start_depth);
    }
    if (!start_valid ||
        !wf2_endpoint_record_coeff16_cut_depth(
            coeff_f16,
            boundary_count,
            track_id,
            right_cut,
            t,
            near_depth,
            far_depth,
            invalid_epsilon,
            end_depth)) {
      cached_right_valid = false;
      continue;
    }
    cached_right_cut = right_cut;
    cached_right_depth = end_depth;
    cached_right_valid = true;
    const float length = end_depth - start_depth;
    if (!(length > 1.0e-8f)) {
      continue;
    }
    const uint owner = uint(owner_raw);
    const uint rgba_base = owner * 4u;
    const float density = max(site_rgba_f32[rgba_base + 3u], 0.0f);
    const float seg_trans = exp(-density * length);
    const float seg_alpha = 1.0f - seg_trans;
    const float weight = transmittance * seg_alpha;
    const float3 rgb = float3(
        site_rgba_f32[rgba_base + 0u],
        site_rgba_f32[rgba_base + 1u],
        site_rgba_f32[rgba_base + 2u]);

    owners[local_run_count] = owner;
    lengths[local_run_count] = length;
    trans_before[local_run_count] = transmittance;
    segment_trans[local_run_count] = seg_trans;
    segment_alpha[local_run_count] = seg_alpha;
    weights[local_run_count] = weight;
    segment_rgb[local_run_count] = rgb;
    local_run_count += 1u;

    transmittance *= seg_trans;
  }

  const uint out_base = sample_id * 3u;
  const float3 grad_rgb = float3(
      grad_rgb_f32[out_base + 0u],
      grad_rgb_f32[out_base + 1u],
      grad_rgb_f32[out_base + 2u]);
  float adj_next_transmittance = 0.0f;
  for (int run_id = int(local_run_count) - 1; run_id >= 0; --run_id) {
    const uint owner = owners[run_id];
    const uint rgba_base = owner * 4u;
    const float raw_density = site_rgba_f32[rgba_base + 3u];
    const float adj_weight = dot(grad_rgb, segment_rgb[run_id]);
    const float adj_trans_before =
        adj_weight * segment_alpha[run_id] +
        adj_next_transmittance * segment_trans[run_id];
    const float adj_segment_alpha = adj_weight * trans_before[run_id];
    const float adj_segment_trans = adj_next_transmittance * trans_before[run_id] - adj_segment_alpha;
    float4 grad_rgba = float4(
        weights[run_id] * grad_rgb.x,
        weights[run_id] * grad_rgb.y,
        weights[run_id] * grad_rgb.z,
        0.0f);
    if (raw_density > 0.0f) {
      grad_rgba.w = adj_segment_trans * (-lengths[run_id] * segment_trans[run_id]);
    }
    wf2_atomic_add4(grad_site_rgba_f32, rgba_base, grad_rgba);
    adj_next_transmittance = adj_trans_before;
  }
}

kernel void wf2_fused_slab_affine_num32_den16_vjp_direct_atomic_tensor(
    device const int* row_index_i32 [[buffer(0)]],
    device const int* candidate_row_offsets_i32 [[buffer(1)]],
    device const float* candidate_depth_num_f32 [[buffer(2)]],
    device const half* candidate_depth_den_f16 [[buffer(3)]],
    device const float* sites_f32 [[buffer(4)]],
    device const float* site_rgba_f32 [[buffer(5)]],
    device const float* ray_coeff_f32 [[buffer(6)]],
    device const float* frame_t_f32 [[buffer(7)]],
    device const float* grad_rgb_f32 [[buffer(8)]],
    device const float* grad_alpha_f32 [[buffer(9)]],
    device const float* grad_depth_f32 [[buffer(10)]],
    device const int* config_i32 [[buffer(11)]],
    device const float* config_f32 [[buffer(12)]],
    device float* output_rgb_f32 [[buffer(13)]],
    device float* output_alpha_f32 [[buffer(14)]],
    device float* output_depth_f32 [[buffer(15)]],
    device atomic_float* grad_site_rgba_f32 [[buffer(16)]],
    uint sample_id [[thread_position_in_grid]]) {
  const uint track_count = uint(config_i32[1]);
  const uint site_count = uint(config_i32[2]);
  const uint frame_count = uint(config_i32[3]);
  const uint time_slab_count = uint(config_i32[4]);
  const uint row_count = uint(config_i32[5]);
  const uint candidate_count = uint(config_i32[6]);
  const uint total_samples = track_count * frame_count;
  if (sample_id >= total_samples) {
    return;
  }

  const uint clamped_site_count = min(site_count, WF2_MAX_REALRAY_SITES);
  const float near_depth = config_f32[0];
  const float far_depth = config_f32[1];
  const float invalid_epsilon = config_f32[2];
  const float transmittance_threshold = config_f32[3];

  const uint track_id = sample_id / frame_count;
  const uint frame_id = sample_id - track_id * frame_count;
  const float t = frame_t_f32[frame_id];
  const float3 origin = wf2_affine_origin_at(ray_coeff_f32, track_id, t);
  const float3 direction = wf2_affine_direction_at(ray_coeff_f32, track_id, t);
  const uint slab_id = wf2_replay_slab_id(t, time_slab_count);
  const int row_index_raw = row_index_i32[track_id];
  if (row_index_raw < 0 || uint(row_index_raw) >= row_count) {
    return;
  }
  const uint row = uint(row_index_raw) * time_slab_count + slab_id;
  const int begin_raw = candidate_row_offsets_i32[row];
  const int end_raw = candidate_row_offsets_i32[row + 1u];
  if (begin_raw < 0 || end_raw < begin_raw || uint(end_raw) > candidate_count) {
    return;
  }

  float depths[WF2_MAX_REALRAY_BOUNDARIES];
  uint depth_count = 0u;
  for (uint cursor = uint(begin_raw); cursor < uint(end_raw); ++cursor) {
    const uint coeff_base = cursor * 2u;
    const float numer = candidate_depth_num_f32[coeff_base + 0u] + candidate_depth_num_f32[coeff_base + 1u] * t;
    const float denom = float(candidate_depth_den_f16[coeff_base + 0u]) +
        float(candidate_depth_den_f16[coeff_base + 1u]) * t;
    if (fabs(denom) < invalid_epsilon) {
      continue;
    }
    const float depth = numer / denom;
    if (isfinite(depth) && depth >= near_depth && depth <= far_depth) {
      wf2_realray_insert_depth(depths, depth_count, depth);
    }
  }

  uint owners[WF2_MAX_REALRAY_SEGMENTS];
  float lengths[WF2_MAX_REALRAY_SEGMENTS];
  float mids[WF2_MAX_REALRAY_SEGMENTS];
  float trans_before[WF2_MAX_REALRAY_SEGMENTS];
  float segment_trans[WF2_MAX_REALRAY_SEGMENTS];
  float segment_alpha[WF2_MAX_REALRAY_SEGMENTS];
  float weights[WF2_MAX_REALRAY_SEGMENTS];
  float3 segment_rgb[WF2_MAX_REALRAY_SEGMENTS];

  float3 rgb_accum = float3(0.0f, 0.0f, 0.0f);
  float alpha_accum = 0.0f;
  float depth_weighted = 0.0f;
  float transmittance = 1.0f;
  float previous_depth = near_depth;
  uint segment_count = 0u;
  for (uint cut_id = 0u; cut_id <= depth_count; ++cut_id) {
    const float next_depth = cut_id < depth_count ? depths[cut_id] : far_depth;
    const float length = next_depth - previous_depth;
    if (length > 1.0e-8f && transmittance > transmittance_threshold && segment_count < WF2_MAX_REALRAY_SEGMENTS) {
      const float mid_depth = 0.5f * (previous_depth + next_depth);
      const float x = origin.x + direction.x * mid_depth;
      const float y = origin.y + direction.y * mid_depth;
      const float z = origin.z + direction.z * mid_depth;
      const uint owner = wf2_realray_owner_at(sites_f32, clamped_site_count, x, y, z, t);
      const uint rgba_base = owner * 4u;
      const float density = max(site_rgba_f32[rgba_base + 3u], 0.0f);
      const float seg_trans = exp(-density * length);
      const float seg_alpha = 1.0f - seg_trans;
      const float weight = transmittance * seg_alpha;
      const float3 rgb = float3(
          site_rgba_f32[rgba_base + 0u],
          site_rgba_f32[rgba_base + 1u],
          site_rgba_f32[rgba_base + 2u]);

      owners[segment_count] = owner;
      lengths[segment_count] = length;
      mids[segment_count] = mid_depth;
      trans_before[segment_count] = transmittance;
      segment_trans[segment_count] = seg_trans;
      segment_alpha[segment_count] = seg_alpha;
      weights[segment_count] = weight;
      segment_rgb[segment_count] = rgb;
      segment_count += 1u;

      rgb_accum += weight * rgb;
      alpha_accum += weight;
      depth_weighted += weight * mid_depth;
      transmittance *= seg_trans;
    }
    previous_depth = next_depth;
  }

  const uint out_base = sample_id * 3u;
  output_rgb_f32[out_base + 0u] = rgb_accum.x;
  output_rgb_f32[out_base + 1u] = rgb_accum.y;
  output_rgb_f32[out_base + 2u] = rgb_accum.z;
  output_alpha_f32[sample_id] = alpha_accum;
  output_depth_f32[sample_id] = alpha_accum > 1.0e-8f ? depth_weighted / alpha_accum : far_depth;

  const float3 grad_rgb = float3(
      grad_rgb_f32[out_base + 0u],
      grad_rgb_f32[out_base + 1u],
      grad_rgb_f32[out_base + 2u]);
  const float grad_alpha = grad_alpha_f32[sample_id];
  const float grad_depth = grad_depth_f32[sample_id];
  float adj_next_transmittance = 0.0f;
  for (int segment_id = int(segment_count) - 1; segment_id >= 0; --segment_id) {
    const uint owner = owners[segment_id];
    float d_loss_d_weight = dot(grad_rgb, segment_rgb[segment_id]) + grad_alpha;
    if (alpha_accum > 1.0e-8f) {
      d_loss_d_weight += grad_depth *
          (mids[segment_id] * alpha_accum - depth_weighted) /
          (alpha_accum * alpha_accum);
    }

    const float adj_trans_before =
        d_loss_d_weight * segment_alpha[segment_id] +
        adj_next_transmittance * segment_trans[segment_id];
    const float adj_segment_alpha = d_loss_d_weight * trans_before[segment_id];
    const float adj_segment_trans =
        adj_next_transmittance * trans_before[segment_id] - adj_segment_alpha;
    float4 grad_rgba = float4(
        weights[segment_id] * grad_rgb.x,
        weights[segment_id] * grad_rgb.y,
        weights[segment_id] * grad_rgb.z,
        0.0f);
    const uint rgba_base = owner * 4u;
    const float raw_density = site_rgba_f32[rgba_base + 3u];
    if (raw_density > 0.0f) {
      grad_rgba.w = adj_segment_trans * (-lengths[segment_id] * segment_trans[segment_id]);
    }
    wf2_atomic_add4(grad_site_rgba_f32, rgba_base, grad_rgba);
    adj_next_transmittance = adj_trans_before;
  }
}

kernel void wf2_fused_slab_affine_num32_den16_vjp_direct_atomic_grad_only_tensor(
    device const int* row_index_i32 [[buffer(0)]],
    device const int* candidate_row_offsets_i32 [[buffer(1)]],
    device const float* candidate_depth_num_f32 [[buffer(2)]],
    device const half* candidate_depth_den_f16 [[buffer(3)]],
    device const float* sites_f32 [[buffer(4)]],
    device const float* site_rgba_f32 [[buffer(5)]],
    device const float* ray_coeff_f32 [[buffer(6)]],
    device const float* frame_t_f32 [[buffer(7)]],
    device const float* grad_rgb_f32 [[buffer(8)]],
    device const float* grad_alpha_f32 [[buffer(9)]],
    device const float* grad_depth_f32 [[buffer(10)]],
    device const int* config_i32 [[buffer(11)]],
    device const float* config_f32 [[buffer(12)]],
    device atomic_float* grad_site_rgba_f32 [[buffer(13)]],
    uint sample_id [[thread_position_in_grid]]) {
  const uint track_count = uint(config_i32[1]);
  const uint site_count = uint(config_i32[2]);
  const uint frame_count = uint(config_i32[3]);
  const uint time_slab_count = uint(config_i32[4]);
  const uint row_count = uint(config_i32[5]);
  const uint candidate_count = uint(config_i32[6]);
  const uint total_samples = track_count * frame_count;
  if (sample_id >= total_samples) {
    return;
  }

  const uint clamped_site_count = min(site_count, WF2_MAX_REALRAY_SITES);
  const float near_depth = config_f32[0];
  const float far_depth = config_f32[1];
  const float invalid_epsilon = config_f32[2];
  const float transmittance_threshold = config_f32[3];

  const uint track_id = sample_id / frame_count;
  const uint frame_id = sample_id - track_id * frame_count;
  const float t = frame_t_f32[frame_id];
  const float3 origin = wf2_affine_origin_at(ray_coeff_f32, track_id, t);
  const float3 direction = wf2_affine_direction_at(ray_coeff_f32, track_id, t);
  const uint slab_id = wf2_replay_slab_id(t, time_slab_count);
  const int row_index_raw = row_index_i32[track_id];
  if (row_index_raw < 0 || uint(row_index_raw) >= row_count) {
    return;
  }
  const uint row = uint(row_index_raw) * time_slab_count + slab_id;
  const int begin_raw = candidate_row_offsets_i32[row];
  const int end_raw = candidate_row_offsets_i32[row + 1u];
  if (begin_raw < 0 || end_raw < begin_raw || uint(end_raw) > candidate_count) {
    return;
  }

  float depths[WF2_MAX_REALRAY_BOUNDARIES];
  uint depth_count = 0u;
  for (uint cursor = uint(begin_raw); cursor < uint(end_raw); ++cursor) {
    const uint coeff_base = cursor * 2u;
    const float numer = candidate_depth_num_f32[coeff_base + 0u] + candidate_depth_num_f32[coeff_base + 1u] * t;
    const float denom = float(candidate_depth_den_f16[coeff_base + 0u]) +
        float(candidate_depth_den_f16[coeff_base + 1u]) * t;
    if (fabs(denom) < invalid_epsilon) {
      continue;
    }
    const float depth = numer / denom;
    if (isfinite(depth) && depth >= near_depth && depth <= far_depth) {
      wf2_realray_insert_depth(depths, depth_count, depth);
    }
  }

  uint owners[WF2_MAX_REALRAY_SEGMENTS];
  float lengths[WF2_MAX_REALRAY_SEGMENTS];
  float mids[WF2_MAX_REALRAY_SEGMENTS];
  float trans_before[WF2_MAX_REALRAY_SEGMENTS];
  float segment_trans[WF2_MAX_REALRAY_SEGMENTS];
  float segment_alpha[WF2_MAX_REALRAY_SEGMENTS];
  float weights[WF2_MAX_REALRAY_SEGMENTS];
  float3 segment_rgb[WF2_MAX_REALRAY_SEGMENTS];

  float alpha_accum = 0.0f;
  float depth_weighted = 0.0f;
  float transmittance = 1.0f;
  float previous_depth = near_depth;
  uint segment_count = 0u;
  for (uint cut_id = 0u; cut_id <= depth_count; ++cut_id) {
    const float next_depth = cut_id < depth_count ? depths[cut_id] : far_depth;
    const float length = next_depth - previous_depth;
    if (length > 1.0e-8f && transmittance > transmittance_threshold && segment_count < WF2_MAX_REALRAY_SEGMENTS) {
      const float mid_depth = 0.5f * (previous_depth + next_depth);
      const float x = origin.x + direction.x * mid_depth;
      const float y = origin.y + direction.y * mid_depth;
      const float z = origin.z + direction.z * mid_depth;
      const uint owner = wf2_realray_owner_at(sites_f32, clamped_site_count, x, y, z, t);
      const uint rgba_base = owner * 4u;
      const float density = max(site_rgba_f32[rgba_base + 3u], 0.0f);
      const float seg_trans = exp(-density * length);
      const float seg_alpha = 1.0f - seg_trans;
      const float weight = transmittance * seg_alpha;
      const float3 rgb = float3(
          site_rgba_f32[rgba_base + 0u],
          site_rgba_f32[rgba_base + 1u],
          site_rgba_f32[rgba_base + 2u]);

      owners[segment_count] = owner;
      lengths[segment_count] = length;
      mids[segment_count] = mid_depth;
      trans_before[segment_count] = transmittance;
      segment_trans[segment_count] = seg_trans;
      segment_alpha[segment_count] = seg_alpha;
      weights[segment_count] = weight;
      segment_rgb[segment_count] = rgb;
      segment_count += 1u;

      alpha_accum += weight;
      depth_weighted += weight * mid_depth;
      transmittance *= seg_trans;
    }
    previous_depth = next_depth;
  }

  const uint out_base = sample_id * 3u;
  const float3 grad_rgb = float3(
      grad_rgb_f32[out_base + 0u],
      grad_rgb_f32[out_base + 1u],
      grad_rgb_f32[out_base + 2u]);
  const float grad_alpha = grad_alpha_f32[sample_id];
  const float grad_depth = grad_depth_f32[sample_id];
  float adj_next_transmittance = 0.0f;
  for (int segment_id = int(segment_count) - 1; segment_id >= 0; --segment_id) {
    const uint owner = owners[segment_id];
    float d_loss_d_weight = dot(grad_rgb, segment_rgb[segment_id]) + grad_alpha;
    if (alpha_accum > 1.0e-8f) {
      d_loss_d_weight += grad_depth *
          (mids[segment_id] * alpha_accum - depth_weighted) /
          (alpha_accum * alpha_accum);
    }

    const float adj_trans_before =
        d_loss_d_weight * segment_alpha[segment_id] +
        adj_next_transmittance * segment_trans[segment_id];
    const float adj_segment_alpha = d_loss_d_weight * trans_before[segment_id];
    const float adj_segment_trans =
        adj_next_transmittance * trans_before[segment_id] - adj_segment_alpha;
    float4 grad_rgba = float4(
        weights[segment_id] * grad_rgb.x,
        weights[segment_id] * grad_rgb.y,
        weights[segment_id] * grad_rgb.z,
        0.0f);
    const uint rgba_base = owner * 4u;
    const float raw_density = site_rgba_f32[rgba_base + 3u];
    if (raw_density > 0.0f) {
      grad_rgba.w = adj_segment_trans * (-lengths[segment_id] * segment_trans[segment_id]);
    }
    wf2_atomic_add4(grad_site_rgba_f32, rgba_base, grad_rgba);
    adj_next_transmittance = adj_trans_before;
  }
}

kernel void wf2_fused_slab_affine_num32_den16_vjp_direct_atomic_grad_only_ownerupdate_tensor(
    device const int* row_index_i32 [[buffer(0)]],
    device const int* candidate_row_offsets_i32 [[buffer(1)]],
    device const int* candidate_boundary_ids_i32 [[buffer(2)]],
    device const float* candidate_depth_num_f32 [[buffer(3)]],
    device const half* candidate_depth_den_f16 [[buffer(4)]],
    device const int* boundary_site_pairs_i32 [[buffer(5)]],
    device const float* sites_f32 [[buffer(6)]],
    device const float* site_rgba_f32 [[buffer(7)]],
    device const float* ray_coeff_f32 [[buffer(8)]],
    device const float* frame_t_f32 [[buffer(9)]],
    device const float* grad_rgb_f32 [[buffer(10)]],
    device const float* grad_alpha_f32 [[buffer(11)]],
    device const float* grad_depth_f32 [[buffer(12)]],
    device const int* config_i32 [[buffer(13)]],
    device const float* config_f32 [[buffer(14)]],
    device atomic_float* grad_site_rgba_f32 [[buffer(15)]],
    uint sample_id [[thread_position_in_grid]]) {
  const uint track_count = uint(config_i32[1]);
  const uint site_count = uint(config_i32[2]);
  const uint frame_count = uint(config_i32[3]);
  const uint time_slab_count = uint(config_i32[4]);
  const uint row_count = uint(config_i32[5]);
  const uint candidate_count = uint(config_i32[6]);
  const uint boundary_count = uint(config_i32[7]);
  const uint total_samples = track_count * frame_count;
  if (sample_id >= total_samples) {
    return;
  }

  const uint clamped_site_count = min(site_count, WF2_MAX_REALRAY_SITES);
  const float near_depth = config_f32[0];
  const float far_depth = config_f32[1];
  const float invalid_epsilon = config_f32[2];
  const float transmittance_threshold = config_f32[3];

  const uint track_id = sample_id / frame_count;
  const uint frame_id = sample_id - track_id * frame_count;
  const float t = frame_t_f32[frame_id];
  const float3 origin = wf2_affine_origin_at(ray_coeff_f32, track_id, t);
  const float3 direction = wf2_affine_direction_at(ray_coeff_f32, track_id, t);
  const uint slab_id = wf2_replay_slab_id(t, time_slab_count);
  const int row_index_raw = row_index_i32[track_id];
  if (row_index_raw < 0 || uint(row_index_raw) >= row_count) {
    return;
  }
  const uint row = uint(row_index_raw) * time_slab_count + slab_id;
  const int begin_raw = candidate_row_offsets_i32[row];
  const int end_raw = candidate_row_offsets_i32[row + 1u];
  if (begin_raw < 0 || end_raw < begin_raw || uint(end_raw) > candidate_count) {
    return;
  }

  float depths[WF2_MAX_REALRAY_BOUNDARIES];
  uint boundary_ids[WF2_MAX_REALRAY_BOUNDARIES];
  uint depth_count = 0u;
  for (uint cursor = uint(begin_raw); cursor < uint(end_raw); ++cursor) {
    const int boundary_id_raw = candidate_boundary_ids_i32[cursor];
    if (boundary_id_raw < 0 || uint(boundary_id_raw) >= boundary_count) {
      continue;
    }
    const uint coeff_base = cursor * 2u;
    const float numer = candidate_depth_num_f32[coeff_base + 0u] + candidate_depth_num_f32[coeff_base + 1u] * t;
    const float denom = float(candidate_depth_den_f16[coeff_base + 0u]) +
        float(candidate_depth_den_f16[coeff_base + 1u]) * t;
    if (fabs(denom) < invalid_epsilon) {
      continue;
    }
    const float depth = numer / denom;
    if (isfinite(depth) && depth >= near_depth && depth <= far_depth) {
      wf2_realray_insert_depth_with_boundary(depths, boundary_ids, depth_count, depth, uint(boundary_id_raw));
    }
  }

  uint owners[WF2_MAX_REALRAY_SEGMENTS];
  float lengths[WF2_MAX_REALRAY_SEGMENTS];
  float mids[WF2_MAX_REALRAY_SEGMENTS];
  float trans_before[WF2_MAX_REALRAY_SEGMENTS];
  float segment_trans[WF2_MAX_REALRAY_SEGMENTS];
  float segment_alpha[WF2_MAX_REALRAY_SEGMENTS];
  float weights[WF2_MAX_REALRAY_SEGMENTS];
  float3 segment_rgb[WF2_MAX_REALRAY_SEGMENTS];

  float alpha_accum = 0.0f;
  float depth_weighted = 0.0f;
  float transmittance = 1.0f;
  float previous_depth = near_depth;
  uint segment_count = 0u;

  for (uint cut_id = 0u; cut_id <= depth_count; ++cut_id) {
    const float next_depth = cut_id < depth_count ? depths[cut_id] : far_depth;
    const float length = next_depth - previous_depth;
    if (length > 1.0e-8f && transmittance > transmittance_threshold && segment_count < WF2_MAX_REALRAY_SEGMENTS) {
      const float mid_depth = 0.5f * (previous_depth + next_depth);
      const uint owner = wf2_realray_owner_at(
          sites_f32,
          clamped_site_count,
          origin.x + direction.x * mid_depth,
          origin.y + direction.y * mid_depth,
          origin.z + direction.z * mid_depth,
          t);
      const uint rgba_base = owner * 4u;
      const float density = max(site_rgba_f32[rgba_base + 3u], 0.0f);
      const float seg_trans = exp(-density * length);
      const float seg_alpha = 1.0f - seg_trans;
      const float weight = transmittance * seg_alpha;
      const float3 rgb = float3(
          site_rgba_f32[rgba_base + 0u],
          site_rgba_f32[rgba_base + 1u],
          site_rgba_f32[rgba_base + 2u]);

      owners[segment_count] = owner;
      lengths[segment_count] = length;
      mids[segment_count] = mid_depth;
      trans_before[segment_count] = transmittance;
      segment_trans[segment_count] = seg_trans;
      segment_alpha[segment_count] = seg_alpha;
      weights[segment_count] = weight;
      segment_rgb[segment_count] = rgb;
      segment_count += 1u;

      alpha_accum += weight;
      depth_weighted += weight * mid_depth;
      transmittance *= seg_trans;
    }
    previous_depth = next_depth;
  }

  const uint out_base = sample_id * 3u;
  const float3 grad_rgb = float3(
      grad_rgb_f32[out_base + 0u],
      grad_rgb_f32[out_base + 1u],
      grad_rgb_f32[out_base + 2u]);
  const float grad_alpha = grad_alpha_f32[sample_id];
  const float grad_depth = grad_depth_f32[sample_id];
  float adj_next_transmittance = 0.0f;
  for (int segment_id = int(segment_count) - 1; segment_id >= 0; --segment_id) {
    const uint segment_owner = owners[segment_id];
    float d_loss_d_weight = dot(grad_rgb, segment_rgb[segment_id]) + grad_alpha;
    if (alpha_accum > 1.0e-8f) {
      d_loss_d_weight += grad_depth *
          (mids[segment_id] * alpha_accum - depth_weighted) /
          (alpha_accum * alpha_accum);
    }

    const float adj_trans_before =
        d_loss_d_weight * segment_alpha[segment_id] +
        adj_next_transmittance * segment_trans[segment_id];
    const float adj_segment_alpha = d_loss_d_weight * trans_before[segment_id];
    const float adj_segment_trans =
        adj_next_transmittance * trans_before[segment_id] - adj_segment_alpha;
    float4 grad_rgba = float4(
        weights[segment_id] * grad_rgb.x,
        weights[segment_id] * grad_rgb.y,
        weights[segment_id] * grad_rgb.z,
        0.0f);
    const uint rgba_base = segment_owner * 4u;
    const float raw_density = site_rgba_f32[rgba_base + 3u];
    if (raw_density > 0.0f) {
      grad_rgba.w = adj_segment_trans * (-lengths[segment_id] * segment_trans[segment_id]);
    }
    wf2_atomic_add4(grad_site_rgba_f32, rgba_base, grad_rgba);
    adj_next_transmittance = adj_trans_before;
  }
}

kernel void wf2_fused_slab_affine_num32_den16_vjp_direct_atomic_rgb_only_tensor(
    device const int* row_index_i32 [[buffer(0)]],
    device const int* candidate_row_offsets_i32 [[buffer(1)]],
    device const float* candidate_depth_num_f32 [[buffer(2)]],
    device const half* candidate_depth_den_f16 [[buffer(3)]],
    device const float* sites_f32 [[buffer(4)]],
    device const float* site_rgba_f32 [[buffer(5)]],
    device const float* ray_coeff_f32 [[buffer(6)]],
    device const float* frame_t_f32 [[buffer(7)]],
    device const float* grad_rgb_f32 [[buffer(8)]],
    device const int* config_i32 [[buffer(9)]],
    device const float* config_f32 [[buffer(10)]],
    device atomic_float* grad_site_rgba_f32 [[buffer(11)]],
    uint sample_id [[thread_position_in_grid]]) {
  const uint track_count = uint(config_i32[1]);
  const uint site_count = uint(config_i32[2]);
  const uint frame_count = uint(config_i32[3]);
  const uint time_slab_count = uint(config_i32[4]);
  const uint row_count = uint(config_i32[5]);
  const uint candidate_count = uint(config_i32[6]);
  const uint total_samples = track_count * frame_count;
  if (sample_id >= total_samples) {
    return;
  }

  const uint clamped_site_count = min(site_count, WF2_MAX_REALRAY_SITES);
  const float near_depth = config_f32[0];
  const float far_depth = config_f32[1];
  const float invalid_epsilon = config_f32[2];
  const float transmittance_threshold = config_f32[3];

  const uint track_id = sample_id / frame_count;
  const uint frame_id = sample_id - track_id * frame_count;
  const float t = frame_t_f32[frame_id];
  const float3 origin = wf2_affine_origin_at(ray_coeff_f32, track_id, t);
  const float3 direction = wf2_affine_direction_at(ray_coeff_f32, track_id, t);
  const uint slab_id = wf2_replay_slab_id(t, time_slab_count);
  const int row_index_raw = row_index_i32[track_id];
  if (row_index_raw < 0 || uint(row_index_raw) >= row_count) {
    return;
  }
  const uint row = uint(row_index_raw) * time_slab_count + slab_id;
  const int begin_raw = candidate_row_offsets_i32[row];
  const int end_raw = candidate_row_offsets_i32[row + 1u];
  if (begin_raw < 0 || end_raw < begin_raw || uint(end_raw) > candidate_count) {
    return;
  }

  float depths[WF2_MAX_REALRAY_BOUNDARIES];
  uint depth_count = 0u;
  for (uint cursor = uint(begin_raw); cursor < uint(end_raw); ++cursor) {
    const uint coeff_base = cursor * 2u;
    const float numer = candidate_depth_num_f32[coeff_base + 0u] + candidate_depth_num_f32[coeff_base + 1u] * t;
    const float denom = float(candidate_depth_den_f16[coeff_base + 0u]) +
        float(candidate_depth_den_f16[coeff_base + 1u]) * t;
    if (fabs(denom) < invalid_epsilon) {
      continue;
    }
    const float depth = numer / denom;
    if (isfinite(depth) && depth >= near_depth && depth <= far_depth) {
      wf2_realray_insert_depth(depths, depth_count, depth);
    }
  }

  uint owners[WF2_MAX_REALRAY_SEGMENTS];
  float lengths[WF2_MAX_REALRAY_SEGMENTS];
  float trans_before[WF2_MAX_REALRAY_SEGMENTS];
  float segment_trans[WF2_MAX_REALRAY_SEGMENTS];
  float segment_alpha[WF2_MAX_REALRAY_SEGMENTS];
  float weights[WF2_MAX_REALRAY_SEGMENTS];
  float3 segment_rgb[WF2_MAX_REALRAY_SEGMENTS];

  float transmittance = 1.0f;
  float previous_depth = near_depth;
  uint segment_count = 0u;
  for (uint cut_id = 0u; cut_id <= depth_count; ++cut_id) {
    const float next_depth = cut_id < depth_count ? depths[cut_id] : far_depth;
    const float length = next_depth - previous_depth;
    if (length > 1.0e-8f && transmittance > transmittance_threshold && segment_count < WF2_MAX_REALRAY_SEGMENTS) {
      const float mid_depth = 0.5f * (previous_depth + next_depth);
      const float x = origin.x + direction.x * mid_depth;
      const float y = origin.y + direction.y * mid_depth;
      const float z = origin.z + direction.z * mid_depth;
      const uint owner = wf2_realray_owner_at(sites_f32, clamped_site_count, x, y, z, t);
      const uint rgba_base = owner * 4u;
      const float density = max(site_rgba_f32[rgba_base + 3u], 0.0f);
      const float seg_trans = exp(-density * length);
      const float seg_alpha = 1.0f - seg_trans;
      const float weight = transmittance * seg_alpha;
      const float3 rgb = float3(
          site_rgba_f32[rgba_base + 0u],
          site_rgba_f32[rgba_base + 1u],
          site_rgba_f32[rgba_base + 2u]);

      owners[segment_count] = owner;
      lengths[segment_count] = length;
      trans_before[segment_count] = transmittance;
      segment_trans[segment_count] = seg_trans;
      segment_alpha[segment_count] = seg_alpha;
      weights[segment_count] = weight;
      segment_rgb[segment_count] = rgb;
      segment_count += 1u;

      transmittance *= seg_trans;
    }
    previous_depth = next_depth;
  }

  const uint out_base = sample_id * 3u;
  const float3 grad_rgb = float3(
      grad_rgb_f32[out_base + 0u],
      grad_rgb_f32[out_base + 1u],
      grad_rgb_f32[out_base + 2u]);
  float adj_next_transmittance = 0.0f;
  for (int segment_id = int(segment_count) - 1; segment_id >= 0; --segment_id) {
    const uint owner = owners[segment_id];
    const float d_loss_d_weight = dot(grad_rgb, segment_rgb[segment_id]);
    const float adj_trans_before =
        d_loss_d_weight * segment_alpha[segment_id] +
        adj_next_transmittance * segment_trans[segment_id];
    const float adj_segment_alpha = d_loss_d_weight * trans_before[segment_id];
    const float adj_segment_trans =
        adj_next_transmittance * trans_before[segment_id] - adj_segment_alpha;
    float4 grad_rgba = float4(
        weights[segment_id] * grad_rgb.x,
        weights[segment_id] * grad_rgb.y,
        weights[segment_id] * grad_rgb.z,
        0.0f);
    const uint rgba_base = owner * 4u;
    const float raw_density = site_rgba_f32[rgba_base + 3u];
    if (raw_density > 0.0f) {
      grad_rgba.w = adj_segment_trans * (-lengths[segment_id] * segment_trans[segment_id]);
    }
    wf2_atomic_add4(grad_site_rgba_f32, rgba_base, grad_rgba);
    adj_next_transmittance = adj_trans_before;
  }
}

kernel void wf2_fused_slab_affine_num32_den16_mse_vjp_direct_atomic_rgb_only_tensor(
    device const int* row_index_i32 [[buffer(0)]],
    device const int* candidate_row_offsets_i32 [[buffer(1)]],
    device const float* candidate_depth_num_f32 [[buffer(2)]],
    device const half* candidate_depth_den_f16 [[buffer(3)]],
    device const float* sites_f32 [[buffer(4)]],
    device const float* site_rgba_f32 [[buffer(5)]],
    device const float* ray_coeff_f32 [[buffer(6)]],
    device const float* frame_t_f32 [[buffer(7)]],
    device const float* target_rgb_f32 [[buffer(8)]],
    device const int* config_i32 [[buffer(9)]],
    device const float* config_f32 [[buffer(10)]],
    device atomic_float* loss_f32 [[buffer(11)]],
    device atomic_float* grad_site_rgba_f32 [[buffer(12)]],
    uint sample_id [[thread_position_in_grid]]) {
  const uint track_count = uint(config_i32[1]);
  const uint site_count = uint(config_i32[2]);
  const uint frame_count = uint(config_i32[3]);
  const uint time_slab_count = uint(config_i32[4]);
  const uint row_count = uint(config_i32[5]);
  const uint candidate_count = uint(config_i32[6]);
  const uint total_samples = track_count * frame_count;
  if (sample_id >= total_samples) {
    return;
  }

  const uint out_base = sample_id * 3u;
  const float3 target_rgb = float3(
      target_rgb_f32[out_base + 0u],
      target_rgb_f32[out_base + 1u],
      target_rgb_f32[out_base + 2u]);
  const float inv_element_count = 1.0f / float(total_samples * 3u);

  const uint track_id = sample_id / frame_count;
  const uint frame_id = sample_id - track_id * frame_count;
  const float t = frame_t_f32[frame_id];
  const uint slab_id = wf2_replay_slab_id(t, time_slab_count);
  const int row_index_raw = row_index_i32[track_id];
  if (row_index_raw < 0 || uint(row_index_raw) >= row_count) {
    const float sample_loss = dot(target_rgb, target_rgb) * inv_element_count;
    atomic_fetch_add_explicit(&loss_f32[0], sample_loss, memory_order_relaxed);
    return;
  }

  const uint row = uint(row_index_raw) * time_slab_count + slab_id;
  const int begin_raw = candidate_row_offsets_i32[row];
  const int end_raw = candidate_row_offsets_i32[row + 1u];
  if (begin_raw < 0 || end_raw < begin_raw || uint(end_raw) > candidate_count) {
    const float sample_loss = dot(target_rgb, target_rgb) * inv_element_count;
    atomic_fetch_add_explicit(&loss_f32[0], sample_loss, memory_order_relaxed);
    return;
  }

  const uint clamped_site_count = min(site_count, WF2_MAX_REALRAY_SITES);
  const float near_depth = config_f32[0];
  const float far_depth = config_f32[1];
  const float invalid_epsilon = config_f32[2];
  const float transmittance_threshold = config_f32[3];
  const float3 origin = wf2_affine_origin_at(ray_coeff_f32, track_id, t);
  const float3 direction = wf2_affine_direction_at(ray_coeff_f32, track_id, t);

  float depths[WF2_MAX_REALRAY_FUSED_MSE_BOUNDARIES];
  uint depth_count = 0u;
  for (uint cursor = uint(begin_raw); cursor < uint(end_raw); ++cursor) {
    const uint coeff_base = cursor * 2u;
    const float numer = candidate_depth_num_f32[coeff_base + 0u] + candidate_depth_num_f32[coeff_base + 1u] * t;
    const float denom = float(candidate_depth_den_f16[coeff_base + 0u]) +
        float(candidate_depth_den_f16[coeff_base + 1u]) * t;
    if (fabs(denom) < invalid_epsilon) {
      continue;
    }
    const float depth = numer / denom;
    if (isfinite(depth) && depth >= near_depth && depth <= far_depth) {
      wf2_realray_insert_depth_fused_mse(depths, depth_count, depth);
    }
  }

  uint owners[WF2_MAX_REALRAY_FUSED_MSE_SEGMENTS];
  float lengths[WF2_MAX_REALRAY_FUSED_MSE_SEGMENTS];
  float trans_before[WF2_MAX_REALRAY_FUSED_MSE_SEGMENTS];
  float segment_trans[WF2_MAX_REALRAY_FUSED_MSE_SEGMENTS];

  float3 rgb_accum = float3(0.0f, 0.0f, 0.0f);
  float transmittance = 1.0f;
  float previous_depth = near_depth;
  uint segment_count = 0u;
  for (uint cut_id = 0u; cut_id <= depth_count; ++cut_id) {
    const float next_depth = cut_id < depth_count ? depths[cut_id] : far_depth;
    const float length = next_depth - previous_depth;
    if (length > 1.0e-8f && transmittance > transmittance_threshold && segment_count < WF2_MAX_REALRAY_FUSED_MSE_SEGMENTS) {
      const float mid_depth = 0.5f * (previous_depth + next_depth);
      const float x = origin.x + direction.x * mid_depth;
      const float y = origin.y + direction.y * mid_depth;
      const float z = origin.z + direction.z * mid_depth;
      const uint owner = wf2_realray_owner_at(sites_f32, clamped_site_count, x, y, z, t);
      const uint rgba_base = owner * 4u;
      const float density = max(site_rgba_f32[rgba_base + 3u], 0.0f);
      const float seg_trans = exp(-density * length);
      const float seg_alpha = 1.0f - seg_trans;
      const float weight = transmittance * seg_alpha;
      const float3 rgb = float3(
          site_rgba_f32[rgba_base + 0u],
          site_rgba_f32[rgba_base + 1u],
          site_rgba_f32[rgba_base + 2u]);

      rgb_accum += weight * rgb;
      if (segment_count > 0u && owners[segment_count - 1u] == owner) {
        const uint previous_segment = segment_count - 1u;
        lengths[previous_segment] += length;
        segment_trans[previous_segment] *= seg_trans;
      } else {
        owners[segment_count] = owner;
        lengths[segment_count] = length;
        trans_before[segment_count] = transmittance;
        segment_trans[segment_count] = seg_trans;
        segment_count += 1u;
      }
      transmittance *= seg_trans;
    }
    previous_depth = next_depth;
  }

  const float3 diff = rgb_accum - target_rgb;
  const float sample_loss = dot(diff, diff) * inv_element_count;
  atomic_fetch_add_explicit(&loss_f32[0], sample_loss, memory_order_relaxed);
  const float3 grad_rgb = (2.0f * inv_element_count) * diff;

  float adj_next_transmittance = 0.0f;
  for (int segment_id = int(segment_count) - 1; segment_id >= 0; --segment_id) {
    const uint owner = owners[segment_id];
    const uint rgba_base = owner * 4u;
    const float3 rgb = float3(
        site_rgba_f32[rgba_base + 0u],
        site_rgba_f32[rgba_base + 1u],
        site_rgba_f32[rgba_base + 2u]);
    const float seg_trans = segment_trans[segment_id];
    const float seg_alpha = 1.0f - seg_trans;
    const float weight = trans_before[segment_id] * seg_alpha;
    const float d_loss_d_weight = dot(grad_rgb, rgb);
    const float adj_trans_before =
        d_loss_d_weight * seg_alpha +
        adj_next_transmittance * seg_trans;
    const float adj_segment_alpha = d_loss_d_weight * trans_before[segment_id];
    const float adj_segment_trans =
        adj_next_transmittance * trans_before[segment_id] - adj_segment_alpha;
    float4 grad_rgba = float4(
        weight * grad_rgb.x,
        weight * grad_rgb.y,
        weight * grad_rgb.z,
        0.0f);
    const float raw_density = site_rgba_f32[rgba_base + 3u];
    if (raw_density > 0.0f) {
      grad_rgba.w = adj_segment_trans * (-lengths[segment_id] * seg_trans);
    }
    wf2_atomic_add4(grad_site_rgba_f32, rgba_base, grad_rgba);
    adj_next_transmittance = adj_trans_before;
  }
}

kernel void wf2_fused_slab_affine_num32_den16_mse_vjp_direct_atomic_track_rgb_only_tensor(
    device const int* row_index_i32 [[buffer(0)]],
    device const int* candidate_row_offsets_i32 [[buffer(1)]],
    device const float* candidate_depth_num_f32 [[buffer(2)]],
    device const half* candidate_depth_den_f16 [[buffer(3)]],
    device const float* sites_f32 [[buffer(4)]],
    device const float* site_rgba_f32 [[buffer(5)]],
    device const float* ray_coeff_f32 [[buffer(6)]],
    device const float* frame_t_f32 [[buffer(7)]],
    device const float* target_rgb_f32 [[buffer(8)]],
    device const int* config_i32 [[buffer(9)]],
    device const float* config_f32 [[buffer(10)]],
    device atomic_float* loss_f32 [[buffer(11)]],
    device atomic_float* grad_site_rgba_f32 [[buffer(12)]],
    uint track_id [[thread_position_in_grid]]) {
  const uint track_count = uint(config_i32[1]);
  const uint site_count = uint(config_i32[2]);
  const uint frame_count = uint(config_i32[3]);
  const uint time_slab_count = uint(config_i32[4]);
  const uint row_count = uint(config_i32[5]);
  const uint candidate_count = uint(config_i32[6]);
  if (track_id >= track_count) {
    return;
  }

  const uint total_samples = track_count * frame_count;
  const float inv_element_count = 1.0f / float(total_samples * 3u);
  const uint clamped_site_count = min(site_count, WF2_MAX_REALRAY_SITES);
  const float near_depth = config_f32[0];
  const float far_depth = config_f32[1];
  const float invalid_epsilon = config_f32[2];
  const float transmittance_threshold = config_f32[3];
  float4 grad_accum[WF2_MAX_REALRAY_SITES];
  for (uint site_id = 0u; site_id < clamped_site_count; ++site_id) {
    grad_accum[site_id] = float4(0.0f, 0.0f, 0.0f, 0.0f);
  }

  float track_loss = 0.0f;
  const int row_index_raw = row_index_i32[track_id];
  if (row_index_raw < 0 || uint(row_index_raw) >= row_count) {
    for (uint frame_id = 0u; frame_id < frame_count; ++frame_id) {
      const uint out_base = (track_id * frame_count + frame_id) * 3u;
      const float3 target_rgb = float3(
          target_rgb_f32[out_base + 0u],
          target_rgb_f32[out_base + 1u],
          target_rgb_f32[out_base + 2u]);
      track_loss += dot(target_rgb, target_rgb) * inv_element_count;
    }
    atomic_fetch_add_explicit(&loss_f32[0], track_loss, memory_order_relaxed);
    return;
  }

  const uint row_base = uint(row_index_raw) * time_slab_count;
  for (uint frame_id = 0u; frame_id < frame_count; ++frame_id) {
    const uint sample_id = track_id * frame_count + frame_id;
    const uint out_base = sample_id * 3u;
    const float3 target_rgb = float3(
        target_rgb_f32[out_base + 0u],
        target_rgb_f32[out_base + 1u],
        target_rgb_f32[out_base + 2u]);
    const float t = frame_t_f32[frame_id];
    const uint slab_id = wf2_replay_slab_id(t, time_slab_count);
    const uint row = row_base + slab_id;
    const int begin_raw = candidate_row_offsets_i32[row];
    const int end_raw = candidate_row_offsets_i32[row + 1u];
    if (begin_raw < 0 || end_raw < begin_raw || uint(end_raw) > candidate_count) {
      track_loss += dot(target_rgb, target_rgb) * inv_element_count;
      continue;
    }

    const float3 origin = wf2_affine_origin_at(ray_coeff_f32, track_id, t);
    const float3 direction = wf2_affine_direction_at(ray_coeff_f32, track_id, t);

    float depths[WF2_MAX_REALRAY_FUSED_MSE_BOUNDARIES];
    uint depth_count = 0u;
    for (uint cursor = uint(begin_raw); cursor < uint(end_raw); ++cursor) {
      const uint coeff_base = cursor * 2u;
      const float numer = candidate_depth_num_f32[coeff_base + 0u] + candidate_depth_num_f32[coeff_base + 1u] * t;
      const float denom = float(candidate_depth_den_f16[coeff_base + 0u]) +
          float(candidate_depth_den_f16[coeff_base + 1u]) * t;
      if (fabs(denom) < invalid_epsilon) {
        continue;
      }
      const float depth = numer / denom;
      if (isfinite(depth) && depth >= near_depth && depth <= far_depth) {
        wf2_realray_insert_depth_fused_mse(depths, depth_count, depth);
      }
    }

    uint owners[WF2_MAX_REALRAY_FUSED_MSE_SEGMENTS];
    float lengths[WF2_MAX_REALRAY_FUSED_MSE_SEGMENTS];
    float trans_before[WF2_MAX_REALRAY_FUSED_MSE_SEGMENTS];
    float segment_trans[WF2_MAX_REALRAY_FUSED_MSE_SEGMENTS];

    float3 rgb_accum = float3(0.0f, 0.0f, 0.0f);
    float transmittance = 1.0f;
    float previous_depth = near_depth;
    uint segment_count = 0u;
    for (uint cut_id = 0u; cut_id <= depth_count; ++cut_id) {
      const float next_depth = cut_id < depth_count ? depths[cut_id] : far_depth;
      const float length = next_depth - previous_depth;
      if (length > 1.0e-8f && transmittance > transmittance_threshold && segment_count < WF2_MAX_REALRAY_FUSED_MSE_SEGMENTS) {
        const float mid_depth = 0.5f * (previous_depth + next_depth);
        const float x = origin.x + direction.x * mid_depth;
        const float y = origin.y + direction.y * mid_depth;
        const float z = origin.z + direction.z * mid_depth;
        const uint owner = wf2_realray_owner_at(sites_f32, clamped_site_count, x, y, z, t);
        const uint rgba_base = owner * 4u;
        const float density = max(site_rgba_f32[rgba_base + 3u], 0.0f);
        const float seg_trans = exp(-density * length);
        const float seg_alpha = 1.0f - seg_trans;
        const float weight = transmittance * seg_alpha;
        const float3 rgb = float3(
            site_rgba_f32[rgba_base + 0u],
            site_rgba_f32[rgba_base + 1u],
            site_rgba_f32[rgba_base + 2u]);

        rgb_accum += weight * rgb;
        if (segment_count > 0u && owners[segment_count - 1u] == owner) {
          const uint previous_segment = segment_count - 1u;
          lengths[previous_segment] += length;
          segment_trans[previous_segment] *= seg_trans;
        } else {
          owners[segment_count] = owner;
          lengths[segment_count] = length;
          trans_before[segment_count] = transmittance;
          segment_trans[segment_count] = seg_trans;
          segment_count += 1u;
        }
        transmittance *= seg_trans;
      }
      previous_depth = next_depth;
    }

    const float3 diff = rgb_accum - target_rgb;
    track_loss += dot(diff, diff) * inv_element_count;
    const float3 grad_rgb = (2.0f * inv_element_count) * diff;

    float adj_next_transmittance = 0.0f;
    for (int segment_id = int(segment_count) - 1; segment_id >= 0; --segment_id) {
      const uint owner = owners[segment_id];
      const uint rgba_base = owner * 4u;
      const float3 rgb = float3(
          site_rgba_f32[rgba_base + 0u],
          site_rgba_f32[rgba_base + 1u],
          site_rgba_f32[rgba_base + 2u]);
      const float seg_trans = segment_trans[segment_id];
      const float seg_alpha = 1.0f - seg_trans;
      const float weight = trans_before[segment_id] * seg_alpha;
      const float d_loss_d_weight = dot(grad_rgb, rgb);
      const float adj_trans_before =
          d_loss_d_weight * seg_alpha +
          adj_next_transmittance * seg_trans;
      const float adj_segment_alpha = d_loss_d_weight * trans_before[segment_id];
      const float adj_segment_trans =
          adj_next_transmittance * trans_before[segment_id] - adj_segment_alpha;
      grad_accum[owner].x += weight * grad_rgb.x;
      grad_accum[owner].y += weight * grad_rgb.y;
      grad_accum[owner].z += weight * grad_rgb.z;
      const float raw_density = site_rgba_f32[rgba_base + 3u];
      if (raw_density > 0.0f) {
        grad_accum[owner].w += adj_segment_trans * (-lengths[segment_id] * seg_trans);
      }
      adj_next_transmittance = adj_trans_before;
    }
  }

  atomic_fetch_add_explicit(&loss_f32[0], track_loss, memory_order_relaxed);
  for (uint site_id = 0u; site_id < clamped_site_count; ++site_id) {
    const float4 grad = grad_accum[site_id];
    if (grad.x != 0.0f || grad.y != 0.0f || grad.z != 0.0f || grad.w != 0.0f) {
      wf2_atomic_add4(grad_site_rgba_f32, site_id * 4u, grad);
    }
  }
}

kernel void wf2_fused_slab_affine_coeff16_mse_vjp_direct_atomic_rgb_only_tensor(
    device const int* row_index_i32 [[buffer(0)]],
    device const int* candidate_row_offsets_i32 [[buffer(1)]],
    device const half* candidate_depth_coeff_f16 [[buffer(2)]],
    device const float* sites_f32 [[buffer(3)]],
    device const float* site_rgba_f32 [[buffer(4)]],
    device const float* ray_coeff_f32 [[buffer(5)]],
    device const float* frame_t_f32 [[buffer(6)]],
    device const float* target_rgb_f32 [[buffer(7)]],
    device const int* config_i32 [[buffer(8)]],
    device const float* config_f32 [[buffer(9)]],
    device atomic_float* loss_f32 [[buffer(10)]],
    device atomic_float* grad_site_rgba_f32 [[buffer(11)]],
    uint sample_id [[thread_position_in_grid]]) {
  const uint track_count = uint(config_i32[1]);
  const uint site_count = uint(config_i32[2]);
  const uint frame_count = uint(config_i32[3]);
  const uint time_slab_count = uint(config_i32[4]);
  const uint row_count = uint(config_i32[5]);
  const uint candidate_count = uint(config_i32[6]);
  const uint total_samples = track_count * frame_count;
  if (sample_id >= total_samples) {
    return;
  }

  const uint out_base = sample_id * 3u;
  const float3 target_rgb = float3(
      target_rgb_f32[out_base + 0u],
      target_rgb_f32[out_base + 1u],
      target_rgb_f32[out_base + 2u]);
  const float inv_element_count = 1.0f / float(total_samples * 3u);

  const uint track_id = sample_id / frame_count;
  const uint frame_id = sample_id - track_id * frame_count;
  const float t = frame_t_f32[frame_id];
  const uint slab_id = wf2_replay_slab_id(t, time_slab_count);
  const int row_index_raw = row_index_i32[track_id];
  if (row_index_raw < 0 || uint(row_index_raw) >= row_count) {
    const float sample_loss = dot(target_rgb, target_rgb) * inv_element_count;
    atomic_fetch_add_explicit(&loss_f32[0], sample_loss, memory_order_relaxed);
    return;
  }

  const uint row = uint(row_index_raw) * time_slab_count + slab_id;
  const int begin_raw = candidate_row_offsets_i32[row];
  const int end_raw = candidate_row_offsets_i32[row + 1u];
  if (begin_raw < 0 || end_raw < begin_raw || uint(end_raw) > candidate_count) {
    const float sample_loss = dot(target_rgb, target_rgb) * inv_element_count;
    atomic_fetch_add_explicit(&loss_f32[0], sample_loss, memory_order_relaxed);
    return;
  }

  const uint clamped_site_count = min(site_count, WF2_MAX_REALRAY_SITES);
  const float near_depth = config_f32[0];
  const float far_depth = config_f32[1];
  const float invalid_epsilon = config_f32[2];
  const float transmittance_threshold = config_f32[3];
  const float3 origin = wf2_affine_origin_at(ray_coeff_f32, track_id, t);
  const float3 direction = wf2_affine_direction_at(ray_coeff_f32, track_id, t);

  float depths[WF2_MAX_REALRAY_FUSED_MSE_BOUNDARIES];
  uint depth_count = 0u;
  for (uint cursor = uint(begin_raw); cursor < uint(end_raw); ++cursor) {
    const uint coeff_base = cursor * 4u;
    const float numer =
        float(candidate_depth_coeff_f16[coeff_base + 0u]) + float(candidate_depth_coeff_f16[coeff_base + 1u]) * t;
    const float denom =
        float(candidate_depth_coeff_f16[coeff_base + 2u]) + float(candidate_depth_coeff_f16[coeff_base + 3u]) * t;
    if (fabs(denom) < invalid_epsilon) {
      continue;
    }
    const float depth = numer / denom;
    if (isfinite(depth) && depth >= near_depth && depth <= far_depth) {
      wf2_realray_insert_depth_fused_mse(depths, depth_count, depth);
    }
  }

  uint owners[WF2_MAX_REALRAY_FUSED_MSE_SEGMENTS];
  float lengths[WF2_MAX_REALRAY_FUSED_MSE_SEGMENTS];
  float trans_before[WF2_MAX_REALRAY_FUSED_MSE_SEGMENTS];
  float segment_trans[WF2_MAX_REALRAY_FUSED_MSE_SEGMENTS];

  float3 rgb_accum = float3(0.0f, 0.0f, 0.0f);
  float transmittance = 1.0f;
  float previous_depth = near_depth;
  uint segment_count = 0u;
  for (uint cut_id = 0u; cut_id <= depth_count; ++cut_id) {
    const float next_depth = cut_id < depth_count ? depths[cut_id] : far_depth;
    const float length = next_depth - previous_depth;
    if (length > 1.0e-8f && transmittance > transmittance_threshold && segment_count < WF2_MAX_REALRAY_FUSED_MSE_SEGMENTS) {
      const float mid_depth = 0.5f * (previous_depth + next_depth);
      const float x = origin.x + direction.x * mid_depth;
      const float y = origin.y + direction.y * mid_depth;
      const float z = origin.z + direction.z * mid_depth;
      const uint owner = wf2_realray_owner_at(sites_f32, clamped_site_count, x, y, z, t);
      const uint rgba_base = owner * 4u;
      const float density = max(site_rgba_f32[rgba_base + 3u], 0.0f);
      const float seg_trans = exp(-density * length);
      const float seg_alpha = 1.0f - seg_trans;
      const float weight = transmittance * seg_alpha;
      const float3 rgb = float3(
          site_rgba_f32[rgba_base + 0u],
          site_rgba_f32[rgba_base + 1u],
          site_rgba_f32[rgba_base + 2u]);

      rgb_accum += weight * rgb;
      if (segment_count > 0u && owners[segment_count - 1u] == owner) {
        const uint previous_segment = segment_count - 1u;
        lengths[previous_segment] += length;
        segment_trans[previous_segment] *= seg_trans;
      } else {
        owners[segment_count] = owner;
        lengths[segment_count] = length;
        trans_before[segment_count] = transmittance;
        segment_trans[segment_count] = seg_trans;
        segment_count += 1u;
      }
      transmittance *= seg_trans;
    }
    previous_depth = next_depth;
  }

  const float3 diff = rgb_accum - target_rgb;
  const float sample_loss = dot(diff, diff) * inv_element_count;
  atomic_fetch_add_explicit(&loss_f32[0], sample_loss, memory_order_relaxed);
  const float3 grad_rgb = (2.0f * inv_element_count) * diff;

  float adj_next_transmittance = 0.0f;
  for (int segment_id = int(segment_count) - 1; segment_id >= 0; --segment_id) {
    const uint owner = owners[segment_id];
    const uint rgba_base = owner * 4u;
    const float3 rgb = float3(
        site_rgba_f32[rgba_base + 0u],
        site_rgba_f32[rgba_base + 1u],
        site_rgba_f32[rgba_base + 2u]);
    const float seg_trans = segment_trans[segment_id];
    const float seg_alpha = 1.0f - seg_trans;
    const float weight = trans_before[segment_id] * seg_alpha;
    const float d_loss_d_weight = dot(grad_rgb, rgb);
    const float adj_trans_before =
        d_loss_d_weight * seg_alpha +
        adj_next_transmittance * seg_trans;
    const float adj_segment_alpha = d_loss_d_weight * trans_before[segment_id];
    const float adj_segment_trans =
        adj_next_transmittance * trans_before[segment_id] - adj_segment_alpha;
    float4 grad_rgba = float4(
        weight * grad_rgb.x,
        weight * grad_rgb.y,
        weight * grad_rgb.z,
        0.0f);
    const float raw_density = site_rgba_f32[rgba_base + 3u];
    if (raw_density > 0.0f) {
      grad_rgba.w = adj_segment_trans * (-lengths[segment_id] * seg_trans);
    }
    wf2_atomic_add4(grad_site_rgba_f32, rgba_base, grad_rgba);
    adj_next_transmittance = adj_trans_before;
  }
}

kernel void wf2_fused_slab_affine_coeff16_cap224_mse_vjp_direct_atomic_rgb_only_tensor(
    device const int* row_index_i32 [[buffer(0)]],
    device const int* candidate_row_offsets_i32 [[buffer(1)]],
    device const half* candidate_depth_coeff_f16 [[buffer(2)]],
    device const float* sites_f32 [[buffer(3)]],
    device const float* site_rgba_f32 [[buffer(4)]],
    device const float* ray_coeff_f32 [[buffer(5)]],
    device const float* frame_t_f32 [[buffer(6)]],
    device const float* target_rgb_f32 [[buffer(7)]],
    device const int* config_i32 [[buffer(8)]],
    device const float* config_f32 [[buffer(9)]],
    device atomic_float* loss_f32 [[buffer(10)]],
    device atomic_float* grad_site_rgba_f32 [[buffer(11)]],
    uint sample_id [[thread_position_in_grid]]) {
  const uint track_count = uint(config_i32[1]);
  const uint site_count = uint(config_i32[2]);
  const uint frame_count = uint(config_i32[3]);
  const uint time_slab_count = uint(config_i32[4]);
  const uint row_count = uint(config_i32[5]);
  const uint candidate_count = uint(config_i32[6]);
  const uint total_samples = track_count * frame_count;
  if (sample_id >= total_samples) {
    return;
  }

  const uint out_base = sample_id * 3u;
  const float3 target_rgb = float3(
      target_rgb_f32[out_base + 0u],
      target_rgb_f32[out_base + 1u],
      target_rgb_f32[out_base + 2u]);
  const float inv_element_count = 1.0f / float(total_samples * 3u);

  const uint track_id = sample_id / frame_count;
  const uint frame_id = sample_id - track_id * frame_count;
  const float t = frame_t_f32[frame_id];
  const uint slab_id = wf2_replay_slab_id(t, time_slab_count);
  const int row_index_raw = row_index_i32[track_id];
  if (row_index_raw < 0 || uint(row_index_raw) >= row_count) {
    const float sample_loss = dot(target_rgb, target_rgb) * inv_element_count;
    atomic_fetch_add_explicit(&loss_f32[0], sample_loss, memory_order_relaxed);
    return;
  }

  const uint row = uint(row_index_raw) * time_slab_count + slab_id;
  const int begin_raw = candidate_row_offsets_i32[row];
  const int end_raw = candidate_row_offsets_i32[row + 1u];
  if (begin_raw < 0 || end_raw < begin_raw || uint(end_raw) > candidate_count ||
      uint(end_raw - begin_raw) > WF2_MAX_REALRAY_FUSED_MSE_CAP224_BOUNDARIES) {
    const float sample_loss = dot(target_rgb, target_rgb) * inv_element_count;
    atomic_fetch_add_explicit(&loss_f32[0], sample_loss, memory_order_relaxed);
    return;
  }

  const uint clamped_site_count = min(site_count, WF2_MAX_REALRAY_SITES);
  const float near_depth = config_f32[0];
  const float far_depth = config_f32[1];
  const float invalid_epsilon = config_f32[2];
  const float transmittance_threshold = config_f32[3];
  const float3 origin = wf2_affine_origin_at(ray_coeff_f32, track_id, t);
  const float3 direction = wf2_affine_direction_at(ray_coeff_f32, track_id, t);

  float depths[WF2_MAX_REALRAY_FUSED_MSE_CAP224_BOUNDARIES];
  uint depth_count = 0u;
  for (uint cursor = uint(begin_raw); cursor < uint(end_raw); ++cursor) {
    const uint coeff_base = cursor * 4u;
    const float numer =
        float(candidate_depth_coeff_f16[coeff_base + 0u]) + float(candidate_depth_coeff_f16[coeff_base + 1u]) * t;
    const float denom =
        float(candidate_depth_coeff_f16[coeff_base + 2u]) + float(candidate_depth_coeff_f16[coeff_base + 3u]) * t;
    if (fabs(denom) < invalid_epsilon) {
      continue;
    }
    const float depth = numer / denom;
    if (isfinite(depth) && depth >= near_depth && depth <= far_depth) {
      wf2_realray_insert_depth_capped(depths, depth_count, depth, WF2_MAX_REALRAY_FUSED_MSE_CAP224_BOUNDARIES);
    }
  }

  uint owners[WF2_MAX_REALRAY_FUSED_MSE_CAP224_SEGMENTS];
  float lengths[WF2_MAX_REALRAY_FUSED_MSE_CAP224_SEGMENTS];
  float trans_before[WF2_MAX_REALRAY_FUSED_MSE_CAP224_SEGMENTS];
  float segment_trans[WF2_MAX_REALRAY_FUSED_MSE_CAP224_SEGMENTS];

  float3 rgb_accum = float3(0.0f, 0.0f, 0.0f);
  float transmittance = 1.0f;
  float previous_depth = near_depth;
  uint segment_count = 0u;
  for (uint cut_id = 0u; cut_id <= depth_count; ++cut_id) {
    const float next_depth = cut_id < depth_count ? depths[cut_id] : far_depth;
    const float length = next_depth - previous_depth;
    if (length > 1.0e-8f && transmittance > transmittance_threshold &&
        segment_count < WF2_MAX_REALRAY_FUSED_MSE_CAP224_SEGMENTS) {
      const float mid_depth = 0.5f * (previous_depth + next_depth);
      const float x = origin.x + direction.x * mid_depth;
      const float y = origin.y + direction.y * mid_depth;
      const float z = origin.z + direction.z * mid_depth;
      const uint owner = wf2_realray_owner_at(sites_f32, clamped_site_count, x, y, z, t);
      const uint rgba_base = owner * 4u;
      const float density = max(site_rgba_f32[rgba_base + 3u], 0.0f);
      const float seg_trans = exp(-density * length);
      const float seg_alpha = 1.0f - seg_trans;
      const float weight = transmittance * seg_alpha;
      const float3 rgb = float3(
          site_rgba_f32[rgba_base + 0u],
          site_rgba_f32[rgba_base + 1u],
          site_rgba_f32[rgba_base + 2u]);

      rgb_accum += weight * rgb;
      if (segment_count > 0u && owners[segment_count - 1u] == owner) {
        const uint previous_segment = segment_count - 1u;
        lengths[previous_segment] += length;
        segment_trans[previous_segment] *= seg_trans;
      } else {
        owners[segment_count] = owner;
        lengths[segment_count] = length;
        trans_before[segment_count] = transmittance;
        segment_trans[segment_count] = seg_trans;
        segment_count += 1u;
      }
      transmittance *= seg_trans;
    }
    previous_depth = next_depth;
  }

  const float3 diff = rgb_accum - target_rgb;
  const float sample_loss = dot(diff, diff) * inv_element_count;
  atomic_fetch_add_explicit(&loss_f32[0], sample_loss, memory_order_relaxed);
  const float3 grad_rgb = (2.0f * inv_element_count) * diff;

  float adj_next_transmittance = 0.0f;
  for (int segment_id = int(segment_count) - 1; segment_id >= 0; --segment_id) {
    const uint owner = owners[segment_id];
    const uint rgba_base = owner * 4u;
    const float3 rgb = float3(
        site_rgba_f32[rgba_base + 0u],
        site_rgba_f32[rgba_base + 1u],
        site_rgba_f32[rgba_base + 2u]);
    const float seg_trans = segment_trans[segment_id];
    const float seg_alpha = 1.0f - seg_trans;
    const float weight = trans_before[segment_id] * seg_alpha;
    const float d_loss_d_weight = dot(grad_rgb, rgb);
    const float adj_trans_before =
        d_loss_d_weight * seg_alpha +
        adj_next_transmittance * seg_trans;
    const float adj_segment_alpha = d_loss_d_weight * trans_before[segment_id];
    const float adj_segment_trans =
        adj_next_transmittance * trans_before[segment_id] - adj_segment_alpha;
    float4 grad_rgba = float4(
        weight * grad_rgb.x,
        weight * grad_rgb.y,
        weight * grad_rgb.z,
        0.0f);
    const float raw_density = site_rgba_f32[rgba_base + 3u];
    if (raw_density > 0.0f) {
      grad_rgba.w = adj_segment_trans * (-lengths[segment_id] * seg_trans);
    }
    wf2_atomic_add4(grad_site_rgba_f32, rgba_base, grad_rgba);
    adj_next_transmittance = adj_trans_before;
  }
}

kernel void wf2_fused_slab_affine_coeff16_densitymask_mse_vjp_direct_atomic_rgb_only_tensor(
    device const int* row_index_i32 [[buffer(0)]],
    device const int* candidate_row_offsets_i32 [[buffer(1)]],
    device const half* candidate_depth_coeff_f16 [[buffer(2)]],
    device const float* sites_f32 [[buffer(3)]],
    device const float* site_rgba_f32 [[buffer(4)]],
    device const float* ray_coeff_f32 [[buffer(5)]],
    device const float* frame_t_f32 [[buffer(6)]],
    device const float* target_rgb_f32 [[buffer(7)]],
    device const int* config_i32 [[buffer(8)]],
    device const float* config_f32 [[buffer(9)]],
    device atomic_float* loss_f32 [[buffer(10)]],
    device atomic_float* grad_site_rgba_f32 [[buffer(11)]],
    uint sample_id [[thread_position_in_grid]]) {
  const uint track_count = uint(config_i32[1]);
  const uint site_count = uint(config_i32[2]);
  const uint frame_count = uint(config_i32[3]);
  const uint time_slab_count = uint(config_i32[4]);
  const uint row_count = uint(config_i32[5]);
  const uint candidate_count = uint(config_i32[6]);
  const uint total_samples = track_count * frame_count;
  if (sample_id >= total_samples) {
    return;
  }

  const uint out_base = sample_id * 3u;
  const float3 target_rgb = float3(
      target_rgb_f32[out_base + 0u],
      target_rgb_f32[out_base + 1u],
      target_rgb_f32[out_base + 2u]);
  const float inv_element_count = 1.0f / float(total_samples * 3u);

  const uint track_id = sample_id / frame_count;
  const uint frame_id = sample_id - track_id * frame_count;
  const float t = frame_t_f32[frame_id];
  const uint slab_id = wf2_replay_slab_id(t, time_slab_count);
  const int row_index_raw = row_index_i32[track_id];
  if (row_index_raw < 0 || uint(row_index_raw) >= row_count) {
    const float sample_loss = dot(target_rgb, target_rgb) * inv_element_count;
    atomic_fetch_add_explicit(&loss_f32[0], sample_loss, memory_order_relaxed);
    return;
  }

  const uint row = uint(row_index_raw) * time_slab_count + slab_id;
  const int begin_raw = candidate_row_offsets_i32[row];
  const int end_raw = candidate_row_offsets_i32[row + 1u];
  if (begin_raw < 0 || end_raw < begin_raw || uint(end_raw) > candidate_count) {
    const float sample_loss = dot(target_rgb, target_rgb) * inv_element_count;
    atomic_fetch_add_explicit(&loss_f32[0], sample_loss, memory_order_relaxed);
    return;
  }

  const uint clamped_site_count = min(site_count, WF2_MAX_REALRAY_SITES);
  const float near_depth = config_f32[0];
  const float far_depth = config_f32[1];
  const float invalid_epsilon = config_f32[2];
  const float transmittance_threshold = config_f32[3];
  const float3 origin = wf2_affine_origin_at(ray_coeff_f32, track_id, t);
  const float3 direction = wf2_affine_direction_at(ray_coeff_f32, track_id, t);

  float depths[WF2_MAX_REALRAY_FUSED_MSE_BOUNDARIES];
  uint depth_count = 0u;
  for (uint cursor = uint(begin_raw); cursor < uint(end_raw); ++cursor) {
    const uint coeff_base = cursor * 4u;
    const float numer =
        float(candidate_depth_coeff_f16[coeff_base + 0u]) + float(candidate_depth_coeff_f16[coeff_base + 1u]) * t;
    const float denom =
        float(candidate_depth_coeff_f16[coeff_base + 2u]) + float(candidate_depth_coeff_f16[coeff_base + 3u]) * t;
    if (fabs(denom) < invalid_epsilon) {
      continue;
    }
    const float depth = numer / denom;
    if (isfinite(depth) && depth >= near_depth && depth <= far_depth) {
      wf2_realray_insert_depth_fused_mse(depths, depth_count, depth);
    }
  }

  uint owners[WF2_MAX_REALRAY_FUSED_MSE_SEGMENTS];
  uchar density_active[WF2_MAX_REALRAY_FUSED_MSE_SEGMENTS];
  float lengths[WF2_MAX_REALRAY_FUSED_MSE_SEGMENTS];
  float trans_before[WF2_MAX_REALRAY_FUSED_MSE_SEGMENTS];
  float segment_trans[WF2_MAX_REALRAY_FUSED_MSE_SEGMENTS];

  float3 rgb_accum = float3(0.0f, 0.0f, 0.0f);
  float transmittance = 1.0f;
  float previous_depth = near_depth;
  uint segment_count = 0u;
  for (uint cut_id = 0u; cut_id <= depth_count; ++cut_id) {
    const float next_depth = cut_id < depth_count ? depths[cut_id] : far_depth;
    const float length = next_depth - previous_depth;
    if (length > 1.0e-8f && transmittance > transmittance_threshold && segment_count < WF2_MAX_REALRAY_FUSED_MSE_SEGMENTS) {
      const float mid_depth = 0.5f * (previous_depth + next_depth);
      const float x = origin.x + direction.x * mid_depth;
      const float y = origin.y + direction.y * mid_depth;
      const float z = origin.z + direction.z * mid_depth;
      const uint owner = wf2_realray_owner_at(sites_f32, clamped_site_count, x, y, z, t);
      const uint rgba_base = owner * 4u;
      const float raw_density = site_rgba_f32[rgba_base + 3u];
      const float density = max(raw_density, 0.0f);
      const float seg_trans = exp(-density * length);
      const float seg_alpha = 1.0f - seg_trans;
      const float weight = transmittance * seg_alpha;
      const float3 rgb = float3(
          site_rgba_f32[rgba_base + 0u],
          site_rgba_f32[rgba_base + 1u],
          site_rgba_f32[rgba_base + 2u]);

      rgb_accum += weight * rgb;
      if (segment_count > 0u && owners[segment_count - 1u] == owner) {
        const uint previous_segment = segment_count - 1u;
        lengths[previous_segment] += length;
        segment_trans[previous_segment] *= seg_trans;
      } else {
        owners[segment_count] = owner;
        density_active[segment_count] = raw_density > 0.0f ? uchar(1) : uchar(0);
        lengths[segment_count] = length;
        trans_before[segment_count] = transmittance;
        segment_trans[segment_count] = seg_trans;
        segment_count += 1u;
      }
      transmittance *= seg_trans;
    }
    previous_depth = next_depth;
  }

  const float3 diff = rgb_accum - target_rgb;
  const float sample_loss = dot(diff, diff) * inv_element_count;
  atomic_fetch_add_explicit(&loss_f32[0], sample_loss, memory_order_relaxed);
  const float3 grad_rgb = (2.0f * inv_element_count) * diff;

  float adj_next_transmittance = 0.0f;
  for (int segment_id = int(segment_count) - 1; segment_id >= 0; --segment_id) {
    const uint owner = owners[segment_id];
    const uint rgba_base = owner * 4u;
    const float3 rgb = float3(
        site_rgba_f32[rgba_base + 0u],
        site_rgba_f32[rgba_base + 1u],
        site_rgba_f32[rgba_base + 2u]);
    const float seg_trans = segment_trans[segment_id];
    const float seg_alpha = 1.0f - seg_trans;
    const float weight = trans_before[segment_id] * seg_alpha;
    const float d_loss_d_weight = dot(grad_rgb, rgb);
    const float adj_trans_before =
        d_loss_d_weight * seg_alpha +
        adj_next_transmittance * seg_trans;
    const float adj_segment_alpha = d_loss_d_weight * trans_before[segment_id];
    const float adj_segment_trans =
        adj_next_transmittance * trans_before[segment_id] - adj_segment_alpha;
    float4 grad_rgba = float4(
        weight * grad_rgb.x,
        weight * grad_rgb.y,
        weight * grad_rgb.z,
        0.0f);
    if (density_active[segment_id] != 0) {
      grad_rgba.w = adj_segment_trans * (-lengths[segment_id] * seg_trans);
    }
    wf2_atomic_add4(grad_site_rgba_f32, rgba_base, grad_rgba);
    adj_next_transmittance = adj_trans_before;
  }
}

kernel void wf2_fused_slab_affine_coeff16_sortnet_mse_vjp_direct_atomic_rgb_only_tensor(
    device const int* row_index_i32 [[buffer(0)]],
    device const int* candidate_row_offsets_i32 [[buffer(1)]],
    device const half* candidate_depth_coeff_f16 [[buffer(2)]],
    device const float* sites_f32 [[buffer(3)]],
    device const float* site_rgba_f32 [[buffer(4)]],
    device const float* ray_coeff_f32 [[buffer(5)]],
    device const float* frame_t_f32 [[buffer(6)]],
    device const float* target_rgb_f32 [[buffer(7)]],
    device const int* config_i32 [[buffer(8)]],
    device const float* config_f32 [[buffer(9)]],
    device atomic_float* loss_f32 [[buffer(10)]],
    device atomic_float* grad_site_rgba_f32 [[buffer(11)]],
    uint sample_id [[thread_position_in_grid]]) {
  const uint track_count = uint(config_i32[1]);
  const uint site_count = uint(config_i32[2]);
  const uint frame_count = uint(config_i32[3]);
  const uint time_slab_count = uint(config_i32[4]);
  const uint row_count = uint(config_i32[5]);
  const uint candidate_count = uint(config_i32[6]);
  const uint total_samples = track_count * frame_count;
  if (sample_id >= total_samples) {
    return;
  }

  const uint out_base = sample_id * 3u;
  const float3 target_rgb = float3(
      target_rgb_f32[out_base + 0u],
      target_rgb_f32[out_base + 1u],
      target_rgb_f32[out_base + 2u]);
  const float inv_element_count = 1.0f / float(total_samples * 3u);

  const uint track_id = sample_id / frame_count;
  const uint frame_id = sample_id - track_id * frame_count;
  const float t = frame_t_f32[frame_id];
  const uint slab_id = wf2_replay_slab_id(t, time_slab_count);
  const int row_index_raw = row_index_i32[track_id];
  if (row_index_raw < 0 || uint(row_index_raw) >= row_count) {
    const float sample_loss = dot(target_rgb, target_rgb) * inv_element_count;
    atomic_fetch_add_explicit(&loss_f32[0], sample_loss, memory_order_relaxed);
    return;
  }

  const uint row = uint(row_index_raw) * time_slab_count + slab_id;
  const int begin_raw = candidate_row_offsets_i32[row];
  const int end_raw = candidate_row_offsets_i32[row + 1u];
  if (begin_raw < 0 || end_raw < begin_raw || uint(end_raw) > candidate_count) {
    const float sample_loss = dot(target_rgb, target_rgb) * inv_element_count;
    atomic_fetch_add_explicit(&loss_f32[0], sample_loss, memory_order_relaxed);
    return;
  }

  const uint clamped_site_count = min(site_count, WF2_MAX_REALRAY_SITES);
  const float near_depth = config_f32[0];
  const float far_depth = config_f32[1];
  const float invalid_epsilon = config_f32[2];
  const float transmittance_threshold = config_f32[3];
  const float3 origin = wf2_affine_origin_at(ray_coeff_f32, track_id, t);
  const float3 direction = wf2_affine_direction_at(ray_coeff_f32, track_id, t);

  float depths[WF2_MAX_REALRAY_FUSED_MSE_BOUNDARIES];
  uint depth_count = 0u;
  for (uint cursor = uint(begin_raw); cursor < uint(end_raw); ++cursor) {
    const uint coeff_base = cursor * 4u;
    const float numer =
        float(candidate_depth_coeff_f16[coeff_base + 0u]) + float(candidate_depth_coeff_f16[coeff_base + 1u]) * t;
    const float denom =
        float(candidate_depth_coeff_f16[coeff_base + 2u]) + float(candidate_depth_coeff_f16[coeff_base + 3u]) * t;
    if (fabs(denom) < invalid_epsilon) {
      continue;
    }
    const float depth = numer / denom;
    if (isfinite(depth) && depth >= near_depth && depth <= far_depth &&
        depth_count < WF2_MAX_REALRAY_FUSED_MSE_BOUNDARIES) {
      depths[depth_count] = depth;
      depth_count += 1u;
    }
  }
  wf2_realray_sort_depths_bitonic_fused_mse(depths, depth_count);

  uint owners[WF2_MAX_REALRAY_FUSED_MSE_SEGMENTS];
  float lengths[WF2_MAX_REALRAY_FUSED_MSE_SEGMENTS];
  float trans_before[WF2_MAX_REALRAY_FUSED_MSE_SEGMENTS];
  float segment_trans[WF2_MAX_REALRAY_FUSED_MSE_SEGMENTS];

  float3 rgb_accum = float3(0.0f, 0.0f, 0.0f);
  float transmittance = 1.0f;
  float previous_depth = near_depth;
  uint segment_count = 0u;
  for (uint cut_id = 0u; cut_id <= depth_count; ++cut_id) {
    const float next_depth = cut_id < depth_count ? depths[cut_id] : far_depth;
    const float length = next_depth - previous_depth;
    if (length > 1.0e-8f && transmittance > transmittance_threshold && segment_count < WF2_MAX_REALRAY_FUSED_MSE_SEGMENTS) {
      const float mid_depth = 0.5f * (previous_depth + next_depth);
      const float x = origin.x + direction.x * mid_depth;
      const float y = origin.y + direction.y * mid_depth;
      const float z = origin.z + direction.z * mid_depth;
      const uint owner = wf2_realray_owner_at(sites_f32, clamped_site_count, x, y, z, t);
      const uint rgba_base = owner * 4u;
      const float density = max(site_rgba_f32[rgba_base + 3u], 0.0f);
      const float seg_trans = exp(-density * length);
      const float seg_alpha = 1.0f - seg_trans;
      const float weight = transmittance * seg_alpha;
      const float3 rgb = float3(
          site_rgba_f32[rgba_base + 0u],
          site_rgba_f32[rgba_base + 1u],
          site_rgba_f32[rgba_base + 2u]);

      rgb_accum += weight * rgb;
      if (segment_count > 0u && owners[segment_count - 1u] == owner) {
        const uint previous_segment = segment_count - 1u;
        lengths[previous_segment] += length;
        segment_trans[previous_segment] *= seg_trans;
      } else {
        owners[segment_count] = owner;
        lengths[segment_count] = length;
        trans_before[segment_count] = transmittance;
        segment_trans[segment_count] = seg_trans;
        segment_count += 1u;
      }
      transmittance *= seg_trans;
    }
    previous_depth = next_depth;
  }

  const float3 diff = rgb_accum - target_rgb;
  const float sample_loss = dot(diff, diff) * inv_element_count;
  atomic_fetch_add_explicit(&loss_f32[0], sample_loss, memory_order_relaxed);
  const float3 grad_rgb = (2.0f * inv_element_count) * diff;

  float adj_next_transmittance = 0.0f;
  for (int segment_id = int(segment_count) - 1; segment_id >= 0; --segment_id) {
    const uint owner = owners[segment_id];
    const uint rgba_base = owner * 4u;
    const float3 rgb = float3(
        site_rgba_f32[rgba_base + 0u],
        site_rgba_f32[rgba_base + 1u],
        site_rgba_f32[rgba_base + 2u]);
    const float seg_trans = segment_trans[segment_id];
    const float seg_alpha = 1.0f - seg_trans;
    const float weight = trans_before[segment_id] * seg_alpha;
    const float d_loss_d_weight = dot(grad_rgb, rgb);
    const float adj_trans_before =
        d_loss_d_weight * seg_alpha +
        adj_next_transmittance * seg_trans;
    const float adj_segment_alpha = d_loss_d_weight * trans_before[segment_id];
    const float adj_segment_trans =
        adj_next_transmittance * trans_before[segment_id] - adj_segment_alpha;
    float4 grad_rgba = float4(
        weight * grad_rgb.x,
        weight * grad_rgb.y,
        weight * grad_rgb.z,
        0.0f);
    const float raw_density = site_rgba_f32[rgba_base + 3u];
    if (raw_density > 0.0f) {
      grad_rgba.w = adj_segment_trans * (-lengths[segment_id] * seg_trans);
    }
    wf2_atomic_add4(grad_site_rgba_f32, rgba_base, grad_rgba);
    adj_next_transmittance = adj_trans_before;
  }
}

kernel void wf2_fused_slab_affine_coeff16_mse_vjp_direct_atomic_sitecache_rgb_only_tensor(
    device const int* row_index_i32 [[buffer(0)]],
    device const int* candidate_row_offsets_i32 [[buffer(1)]],
    device const half* candidate_depth_coeff_f16 [[buffer(2)]],
    device const float* sites_f32 [[buffer(3)]],
    device const float* site_rgba_f32 [[buffer(4)]],
    device const float* ray_coeff_f32 [[buffer(5)]],
    device const float* frame_t_f32 [[buffer(6)]],
    device const float* target_rgb_f32 [[buffer(7)]],
    device const int* config_i32 [[buffer(8)]],
    device const float* config_f32 [[buffer(9)]],
    device atomic_float* loss_f32 [[buffer(10)]],
    device atomic_float* grad_site_rgba_f32 [[buffer(11)]],
    uint sample_id [[thread_position_in_grid]],
    uint local_id [[thread_position_in_threadgroup]]) {
  constexpr uint sitecache_threads = 256u;
  threadgroup float tg_sites[WF2_MAX_REALRAY_SITES * 5u];
  threadgroup float tg_rgba[WF2_MAX_REALRAY_SITES * 4u];

  const uint track_count = uint(config_i32[1]);
  const uint site_count = uint(config_i32[2]);
  const uint frame_count = uint(config_i32[3]);
  const uint time_slab_count = uint(config_i32[4]);
  const uint row_count = uint(config_i32[5]);
  const uint candidate_count = uint(config_i32[6]);
  const uint total_samples = track_count * frame_count;
  const uint clamped_site_count = min(site_count, WF2_MAX_REALRAY_SITES);
  for (uint slot = local_id; slot < clamped_site_count * 5u; slot += sitecache_threads) {
    tg_sites[slot] = sites_f32[slot];
  }
  for (uint slot = local_id; slot < clamped_site_count * 4u; slot += sitecache_threads) {
    tg_rgba[slot] = site_rgba_f32[slot];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sample_id >= total_samples) {
    return;
  }

  const uint out_base = sample_id * 3u;
  const float3 target_rgb = float3(
      target_rgb_f32[out_base + 0u],
      target_rgb_f32[out_base + 1u],
      target_rgb_f32[out_base + 2u]);
  const float inv_element_count = 1.0f / float(total_samples * 3u);

  const uint track_id = sample_id / frame_count;
  const uint frame_id = sample_id - track_id * frame_count;
  const float t = frame_t_f32[frame_id];
  const uint slab_id = wf2_replay_slab_id(t, time_slab_count);
  const int row_index_raw = row_index_i32[track_id];
  if (row_index_raw < 0 || uint(row_index_raw) >= row_count) {
    const float sample_loss = dot(target_rgb, target_rgb) * inv_element_count;
    atomic_fetch_add_explicit(&loss_f32[0], sample_loss, memory_order_relaxed);
    return;
  }

  const uint row = uint(row_index_raw) * time_slab_count + slab_id;
  const int begin_raw = candidate_row_offsets_i32[row];
  const int end_raw = candidate_row_offsets_i32[row + 1u];
  if (begin_raw < 0 || end_raw < begin_raw || uint(end_raw) > candidate_count) {
    const float sample_loss = dot(target_rgb, target_rgb) * inv_element_count;
    atomic_fetch_add_explicit(&loss_f32[0], sample_loss, memory_order_relaxed);
    return;
  }

  const float near_depth = config_f32[0];
  const float far_depth = config_f32[1];
  const float invalid_epsilon = config_f32[2];
  const float transmittance_threshold = config_f32[3];
  const float3 origin = wf2_affine_origin_at(ray_coeff_f32, track_id, t);
  const float3 direction = wf2_affine_direction_at(ray_coeff_f32, track_id, t);

  float depths[WF2_MAX_REALRAY_FUSED_MSE_BOUNDARIES];
  uint depth_count = 0u;
  for (uint cursor = uint(begin_raw); cursor < uint(end_raw); ++cursor) {
    const uint coeff_base = cursor * 4u;
    const float numer =
        float(candidate_depth_coeff_f16[coeff_base + 0u]) + float(candidate_depth_coeff_f16[coeff_base + 1u]) * t;
    const float denom =
        float(candidate_depth_coeff_f16[coeff_base + 2u]) + float(candidate_depth_coeff_f16[coeff_base + 3u]) * t;
    if (fabs(denom) < invalid_epsilon) {
      continue;
    }
    const float depth = numer / denom;
    if (isfinite(depth) && depth >= near_depth && depth <= far_depth) {
      wf2_realray_insert_depth_fused_mse(depths, depth_count, depth);
    }
  }

  uint owners[WF2_MAX_REALRAY_FUSED_MSE_SEGMENTS];
  float lengths[WF2_MAX_REALRAY_FUSED_MSE_SEGMENTS];
  float trans_before[WF2_MAX_REALRAY_FUSED_MSE_SEGMENTS];
  float segment_trans[WF2_MAX_REALRAY_FUSED_MSE_SEGMENTS];

  float3 rgb_accum = float3(0.0f, 0.0f, 0.0f);
  float transmittance = 1.0f;
  float previous_depth = near_depth;
  uint segment_count = 0u;
  for (uint cut_id = 0u; cut_id <= depth_count; ++cut_id) {
    const float next_depth = cut_id < depth_count ? depths[cut_id] : far_depth;
    const float length = next_depth - previous_depth;
    if (length > 1.0e-8f && transmittance > transmittance_threshold && segment_count < WF2_MAX_REALRAY_FUSED_MSE_SEGMENTS) {
      const float mid_depth = 0.5f * (previous_depth + next_depth);
      const float x = origin.x + direction.x * mid_depth;
      const float y = origin.y + direction.y * mid_depth;
      const float z = origin.z + direction.z * mid_depth;
      const uint owner = wf2_realray_owner_at_cached(tg_sites, clamped_site_count, x, y, z, t);
      const uint rgba_base = owner * 4u;
      const float density = max(tg_rgba[rgba_base + 3u], 0.0f);
      const float seg_trans = exp(-density * length);
      const float seg_alpha = 1.0f - seg_trans;
      const float weight = transmittance * seg_alpha;
      const float3 rgb = float3(
          tg_rgba[rgba_base + 0u],
          tg_rgba[rgba_base + 1u],
          tg_rgba[rgba_base + 2u]);

      rgb_accum += weight * rgb;
      if (segment_count > 0u && owners[segment_count - 1u] == owner) {
        const uint previous_segment = segment_count - 1u;
        lengths[previous_segment] += length;
        segment_trans[previous_segment] *= seg_trans;
      } else {
        owners[segment_count] = owner;
        lengths[segment_count] = length;
        trans_before[segment_count] = transmittance;
        segment_trans[segment_count] = seg_trans;
        segment_count += 1u;
      }
      transmittance *= seg_trans;
    }
    previous_depth = next_depth;
  }

  const float3 diff = rgb_accum - target_rgb;
  const float sample_loss = dot(diff, diff) * inv_element_count;
  atomic_fetch_add_explicit(&loss_f32[0], sample_loss, memory_order_relaxed);
  const float3 grad_rgb = (2.0f * inv_element_count) * diff;

  float adj_next_transmittance = 0.0f;
  for (int segment_id = int(segment_count) - 1; segment_id >= 0; --segment_id) {
    const uint owner = owners[segment_id];
    const uint rgba_base = owner * 4u;
    const float3 rgb = float3(
        tg_rgba[rgba_base + 0u],
        tg_rgba[rgba_base + 1u],
        tg_rgba[rgba_base + 2u]);
    const float seg_trans = segment_trans[segment_id];
    const float seg_alpha = 1.0f - seg_trans;
    const float weight = trans_before[segment_id] * seg_alpha;
    const float d_loss_d_weight = dot(grad_rgb, rgb);
    const float adj_trans_before =
        d_loss_d_weight * seg_alpha +
        adj_next_transmittance * seg_trans;
    const float adj_segment_alpha = d_loss_d_weight * trans_before[segment_id];
    const float adj_segment_trans =
        adj_next_transmittance * trans_before[segment_id] - adj_segment_alpha;
    float4 grad_rgba = float4(
        weight * grad_rgb.x,
        weight * grad_rgb.y,
        weight * grad_rgb.z,
        0.0f);
    const float raw_density = tg_rgba[rgba_base + 3u];
    if (raw_density > 0.0f) {
      grad_rgba.w = adj_segment_trans * (-lengths[segment_id] * seg_trans);
    }
    wf2_atomic_add4(grad_site_rgba_f32, rgba_base, grad_rgba);
    adj_next_transmittance = adj_trans_before;
  }
}

kernel void wf2_fused_slab_affine_coeff16_mse_vjp_direct_atomic_sample_reduce_rgb_only_tensor(
    device const int* row_index_i32 [[buffer(0)]],
    device const int* candidate_row_offsets_i32 [[buffer(1)]],
    device const half* candidate_depth_coeff_f16 [[buffer(2)]],
    device const float* sites_f32 [[buffer(3)]],
    device const float* site_rgba_f32 [[buffer(4)]],
    device const float* ray_coeff_f32 [[buffer(5)]],
    device const float* frame_t_f32 [[buffer(6)]],
    device const float* target_rgb_f32 [[buffer(7)]],
    device const int* config_i32 [[buffer(8)]],
    device const float* config_f32 [[buffer(9)]],
    device atomic_float* loss_f32 [[buffer(10)]],
    device atomic_float* grad_site_rgba_f32 [[buffer(11)]],
    uint sample_id [[thread_position_in_grid]]) {
  const uint track_count = uint(config_i32[1]);
  const uint site_count = uint(config_i32[2]);
  const uint frame_count = uint(config_i32[3]);
  const uint time_slab_count = uint(config_i32[4]);
  const uint row_count = uint(config_i32[5]);
  const uint candidate_count = uint(config_i32[6]);
  const uint total_samples = track_count * frame_count;
  if (sample_id >= total_samples) {
    return;
  }

  const uint out_base = sample_id * 3u;
  const float3 target_rgb = float3(
      target_rgb_f32[out_base + 0u],
      target_rgb_f32[out_base + 1u],
      target_rgb_f32[out_base + 2u]);
  const float inv_element_count = 1.0f / float(total_samples * 3u);

  const uint track_id = sample_id / frame_count;
  const uint frame_id = sample_id - track_id * frame_count;
  const float t = frame_t_f32[frame_id];
  const uint slab_id = wf2_replay_slab_id(t, time_slab_count);
  const int row_index_raw = row_index_i32[track_id];
  if (row_index_raw < 0 || uint(row_index_raw) >= row_count) {
    const float sample_loss = dot(target_rgb, target_rgb) * inv_element_count;
    atomic_fetch_add_explicit(&loss_f32[0], sample_loss, memory_order_relaxed);
    return;
  }

  const uint row = uint(row_index_raw) * time_slab_count + slab_id;
  const int begin_raw = candidate_row_offsets_i32[row];
  const int end_raw = candidate_row_offsets_i32[row + 1u];
  if (begin_raw < 0 || end_raw < begin_raw || uint(end_raw) > candidate_count) {
    const float sample_loss = dot(target_rgb, target_rgb) * inv_element_count;
    atomic_fetch_add_explicit(&loss_f32[0], sample_loss, memory_order_relaxed);
    return;
  }

  const uint clamped_site_count = min(site_count, WF2_MAX_REALRAY_SITES);
  const float near_depth = config_f32[0];
  const float far_depth = config_f32[1];
  const float invalid_epsilon = config_f32[2];
  const float transmittance_threshold = config_f32[3];
  const float3 origin = wf2_affine_origin_at(ray_coeff_f32, track_id, t);
  const float3 direction = wf2_affine_direction_at(ray_coeff_f32, track_id, t);

  float depths[WF2_MAX_REALRAY_FUSED_MSE_BOUNDARIES];
  uint depth_count = 0u;
  for (uint cursor = uint(begin_raw); cursor < uint(end_raw); ++cursor) {
    const uint coeff_base = cursor * 4u;
    const float numer =
        float(candidate_depth_coeff_f16[coeff_base + 0u]) + float(candidate_depth_coeff_f16[coeff_base + 1u]) * t;
    const float denom =
        float(candidate_depth_coeff_f16[coeff_base + 2u]) + float(candidate_depth_coeff_f16[coeff_base + 3u]) * t;
    if (fabs(denom) < invalid_epsilon) {
      continue;
    }
    const float depth = numer / denom;
    if (isfinite(depth) && depth >= near_depth && depth <= far_depth) {
      wf2_realray_insert_depth_fused_mse(depths, depth_count, depth);
    }
  }

  uint owners[WF2_MAX_REALRAY_FUSED_MSE_SEGMENTS];
  float lengths[WF2_MAX_REALRAY_FUSED_MSE_SEGMENTS];
  float trans_before[WF2_MAX_REALRAY_FUSED_MSE_SEGMENTS];
  float segment_trans[WF2_MAX_REALRAY_FUSED_MSE_SEGMENTS];
  float4 grad_accum[WF2_MAX_REALRAY_SITES];
  for (uint site_id = 0u; site_id < clamped_site_count; ++site_id) {
    grad_accum[site_id] = float4(0.0f, 0.0f, 0.0f, 0.0f);
  }

  float3 rgb_accum = float3(0.0f, 0.0f, 0.0f);
  float transmittance = 1.0f;
  float previous_depth = near_depth;
  uint segment_count = 0u;
  for (uint cut_id = 0u; cut_id <= depth_count; ++cut_id) {
    const float next_depth = cut_id < depth_count ? depths[cut_id] : far_depth;
    const float length = next_depth - previous_depth;
    if (length > 1.0e-8f && transmittance > transmittance_threshold && segment_count < WF2_MAX_REALRAY_FUSED_MSE_SEGMENTS) {
      const float mid_depth = 0.5f * (previous_depth + next_depth);
      const float x = origin.x + direction.x * mid_depth;
      const float y = origin.y + direction.y * mid_depth;
      const float z = origin.z + direction.z * mid_depth;
      const uint owner = wf2_realray_owner_at(sites_f32, clamped_site_count, x, y, z, t);
      const uint rgba_base = owner * 4u;
      const float density = max(site_rgba_f32[rgba_base + 3u], 0.0f);
      const float seg_trans = exp(-density * length);
      const float seg_alpha = 1.0f - seg_trans;
      const float weight = transmittance * seg_alpha;
      const float3 rgb = float3(
          site_rgba_f32[rgba_base + 0u],
          site_rgba_f32[rgba_base + 1u],
          site_rgba_f32[rgba_base + 2u]);

      rgb_accum += weight * rgb;
      if (segment_count > 0u && owners[segment_count - 1u] == owner) {
        const uint previous_segment = segment_count - 1u;
        lengths[previous_segment] += length;
        segment_trans[previous_segment] *= seg_trans;
      } else {
        owners[segment_count] = owner;
        lengths[segment_count] = length;
        trans_before[segment_count] = transmittance;
        segment_trans[segment_count] = seg_trans;
        segment_count += 1u;
      }
      transmittance *= seg_trans;
    }
    previous_depth = next_depth;
  }

  const float3 diff = rgb_accum - target_rgb;
  const float sample_loss = dot(diff, diff) * inv_element_count;
  atomic_fetch_add_explicit(&loss_f32[0], sample_loss, memory_order_relaxed);
  const float3 grad_rgb = (2.0f * inv_element_count) * diff;

  float adj_next_transmittance = 0.0f;
  for (int segment_id = int(segment_count) - 1; segment_id >= 0; --segment_id) {
    const uint owner = owners[segment_id];
    const uint rgba_base = owner * 4u;
    const float3 rgb = float3(
        site_rgba_f32[rgba_base + 0u],
        site_rgba_f32[rgba_base + 1u],
        site_rgba_f32[rgba_base + 2u]);
    const float seg_trans = segment_trans[segment_id];
    const float seg_alpha = 1.0f - seg_trans;
    const float weight = trans_before[segment_id] * seg_alpha;
    const float d_loss_d_weight = dot(grad_rgb, rgb);
    const float adj_trans_before =
        d_loss_d_weight * seg_alpha +
        adj_next_transmittance * seg_trans;
    const float adj_segment_alpha = d_loss_d_weight * trans_before[segment_id];
    const float adj_segment_trans =
        adj_next_transmittance * trans_before[segment_id] - adj_segment_alpha;
    float4 grad_rgba = float4(
        weight * grad_rgb.x,
        weight * grad_rgb.y,
        weight * grad_rgb.z,
        0.0f);
    const float raw_density = site_rgba_f32[rgba_base + 3u];
    if (raw_density > 0.0f) {
      grad_rgba.w = adj_segment_trans * (-lengths[segment_id] * seg_trans);
    }
    grad_accum[owner] += grad_rgba;
    adj_next_transmittance = adj_trans_before;
  }

  for (uint site_id = 0u; site_id < clamped_site_count; ++site_id) {
    const float4 grad = grad_accum[site_id];
    if (grad.x != 0.0f || grad.y != 0.0f || grad.z != 0.0f || grad.w != 0.0f) {
      wf2_atomic_add4(grad_site_rgba_f32, site_id * 4u, grad);
    }
  }
}

kernel void wf2_fused_slab_affine_coeff16_mse_vjp_direct_atomic_framegroup16_cached_rgb_only_tensor(
    device const int* row_index_i32 [[buffer(0)]],
    device const int* candidate_row_offsets_i32 [[buffer(1)]],
    device const half* candidate_depth_coeff_f16 [[buffer(2)]],
    device const float* sites_f32 [[buffer(3)]],
    device const float* site_rgba_f32 [[buffer(4)]],
    device const float* ray_coeff_f32 [[buffer(5)]],
    device const float* frame_t_f32 [[buffer(6)]],
    device const float* target_rgb_f32 [[buffer(7)]],
    device const int* config_i32 [[buffer(8)]],
    device const float* config_f32 [[buffer(9)]],
    device atomic_float* loss_f32 [[buffer(10)]],
    device atomic_float* grad_site_rgba_f32 [[buffer(11)]],
    uint local_frame [[thread_position_in_threadgroup]],
    uint group_id [[threadgroup_position_in_grid]]) {
  constexpr uint framegroup_threads = WF2_ENDPOINT_EDIT_FRAMEGROUP_MAX_FRAMES;
  threadgroup half tg_coeff[WF2_MAX_REALRAY_FUSED_MSE_BOUNDARIES * 4u];
  threadgroup float tg_loss[WF2_ENDPOINT_EDIT_FRAMEGROUP_MAX_FRAMES];

  const uint track_count = uint(config_i32[1]);
  const uint site_count = uint(config_i32[2]);
  const uint frame_count = uint(config_i32[3]);
  const uint time_slab_count = uint(config_i32[4]);
  const uint row_count = uint(config_i32[5]);
  const uint candidate_count = uint(config_i32[6]);
  const uint chunk_count = (frame_count + framegroup_threads - 1u) / framegroup_threads;
  const uint track_id = group_id / chunk_count;
  const uint chunk_id = group_id - track_id * chunk_count;
  const uint frame_id = chunk_id * framegroup_threads + local_frame;
  const bool active_sample = track_id < track_count && frame_id < frame_count;
  const uint total_samples = track_count * frame_count;
  const float inv_element_count = 1.0f / float(total_samples * 3u);

  int begin_raw = 0;
  int end_raw = 0;
  bool row_valid = false;
  const int row_index_raw = track_id < track_count ? row_index_i32[track_id] : -1;
  if (row_index_raw >= 0 && uint(row_index_raw) < row_count) {
    const uint row = uint(row_index_raw) * time_slab_count;
    begin_raw = candidate_row_offsets_i32[row];
    end_raw = candidate_row_offsets_i32[row + 1u];
    row_valid = begin_raw >= 0 && end_raw >= begin_raw && uint(end_raw) <= candidate_count;
  }
  const uint row_candidate_count = row_valid ? uint(end_raw - begin_raw) : 0u;
  for (uint slot = local_frame; slot < row_candidate_count; slot += framegroup_threads) {
    const uint src = (uint(begin_raw) + slot) * 4u;
    const uint dst = slot * 4u;
    tg_coeff[dst + 0u] = candidate_depth_coeff_f16[src + 0u];
    tg_coeff[dst + 1u] = candidate_depth_coeff_f16[src + 1u];
    tg_coeff[dst + 2u] = candidate_depth_coeff_f16[src + 2u];
    tg_coeff[dst + 3u] = candidate_depth_coeff_f16[src + 3u];
  }
  tg_loss[local_frame] = 0.0f;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (active_sample) {
    const uint out_base = (track_id * frame_count + frame_id) * 3u;
    const float3 target_rgb = float3(
        target_rgb_f32[out_base + 0u],
        target_rgb_f32[out_base + 1u],
        target_rgb_f32[out_base + 2u]);
    if (!row_valid) {
      tg_loss[local_frame] = dot(target_rgb, target_rgb) * inv_element_count;
    } else {
      const float t = frame_t_f32[frame_id];
      const uint clamped_site_count = min(site_count, WF2_MAX_REALRAY_SITES);
      const float near_depth = config_f32[0];
      const float far_depth = config_f32[1];
      const float invalid_epsilon = config_f32[2];
      const float transmittance_threshold = config_f32[3];
      const float3 origin = wf2_affine_origin_at(ray_coeff_f32, track_id, t);
      const float3 direction = wf2_affine_direction_at(ray_coeff_f32, track_id, t);

      float depths[WF2_MAX_REALRAY_FUSED_MSE_BOUNDARIES];
      uint depth_count = 0u;
      for (uint slot = 0u; slot < row_candidate_count; ++slot) {
        const uint coeff_base = slot * 4u;
        const float numer = float(tg_coeff[coeff_base + 0u]) + float(tg_coeff[coeff_base + 1u]) * t;
        const float denom = float(tg_coeff[coeff_base + 2u]) + float(tg_coeff[coeff_base + 3u]) * t;
        if (fabs(denom) < invalid_epsilon) {
          continue;
        }
        const float depth = numer / denom;
        if (isfinite(depth) && depth >= near_depth && depth <= far_depth) {
          wf2_realray_insert_depth_fused_mse(depths, depth_count, depth);
        }
      }

      uint owners[WF2_MAX_REALRAY_FUSED_MSE_SEGMENTS];
      float lengths[WF2_MAX_REALRAY_FUSED_MSE_SEGMENTS];
      float trans_before[WF2_MAX_REALRAY_FUSED_MSE_SEGMENTS];
      float segment_trans[WF2_MAX_REALRAY_FUSED_MSE_SEGMENTS];

      float3 rgb_accum = float3(0.0f, 0.0f, 0.0f);
      float transmittance = 1.0f;
      float previous_depth = near_depth;
      uint segment_count = 0u;
      for (uint cut_id = 0u; cut_id <= depth_count; ++cut_id) {
        const float next_depth = cut_id < depth_count ? depths[cut_id] : far_depth;
        const float length = next_depth - previous_depth;
        if (length > 1.0e-8f && transmittance > transmittance_threshold &&
            segment_count < WF2_MAX_REALRAY_FUSED_MSE_SEGMENTS) {
          const float mid_depth = 0.5f * (previous_depth + next_depth);
          const float x = origin.x + direction.x * mid_depth;
          const float y = origin.y + direction.y * mid_depth;
          const float z = origin.z + direction.z * mid_depth;
          const uint owner = wf2_realray_owner_at(sites_f32, clamped_site_count, x, y, z, t);
          const uint rgba_base = owner * 4u;
          const float density = max(site_rgba_f32[rgba_base + 3u], 0.0f);
          const float seg_trans = exp(-density * length);
          const float seg_alpha = 1.0f - seg_trans;
          const float weight = transmittance * seg_alpha;
          const float3 rgb = float3(
              site_rgba_f32[rgba_base + 0u],
              site_rgba_f32[rgba_base + 1u],
              site_rgba_f32[rgba_base + 2u]);

          rgb_accum += weight * rgb;
          if (segment_count > 0u && owners[segment_count - 1u] == owner) {
            const uint previous_segment = segment_count - 1u;
            lengths[previous_segment] += length;
            segment_trans[previous_segment] *= seg_trans;
          } else {
            owners[segment_count] = owner;
            lengths[segment_count] = length;
            trans_before[segment_count] = transmittance;
            segment_trans[segment_count] = seg_trans;
            segment_count += 1u;
          }
          transmittance *= seg_trans;
        }
        previous_depth = next_depth;
      }

      const float3 diff = rgb_accum - target_rgb;
      tg_loss[local_frame] = dot(diff, diff) * inv_element_count;
      const float3 grad_rgb = (2.0f * inv_element_count) * diff;

      float adj_next_transmittance = 0.0f;
      for (int segment_id = int(segment_count) - 1; segment_id >= 0; --segment_id) {
        const uint owner = owners[segment_id];
        const uint rgba_base = owner * 4u;
        const float3 rgb = float3(
            site_rgba_f32[rgba_base + 0u],
            site_rgba_f32[rgba_base + 1u],
            site_rgba_f32[rgba_base + 2u]);
        const float seg_trans = segment_trans[segment_id];
        const float seg_alpha = 1.0f - seg_trans;
        const float weight = trans_before[segment_id] * seg_alpha;
        const float d_loss_d_weight = dot(grad_rgb, rgb);
        const float adj_trans_before =
            d_loss_d_weight * seg_alpha +
            adj_next_transmittance * seg_trans;
        const float adj_segment_alpha = d_loss_d_weight * trans_before[segment_id];
        const float adj_segment_trans =
            adj_next_transmittance * trans_before[segment_id] - adj_segment_alpha;
        float4 grad_rgba = float4(
            weight * grad_rgb.x,
            weight * grad_rgb.y,
            weight * grad_rgb.z,
            0.0f);
        const float raw_density = site_rgba_f32[rgba_base + 3u];
        if (raw_density > 0.0f) {
          grad_rgba.w = adj_segment_trans * (-lengths[segment_id] * seg_trans);
        }
        wf2_atomic_add4(grad_site_rgba_f32, rgba_base, grad_rgba);
        adj_next_transmittance = adj_trans_before;
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (local_frame == 0u) {
    float group_loss = 0.0f;
    for (uint i = 0u; i < framegroup_threads; ++i) {
      group_loss += tg_loss[i];
    }
    atomic_fetch_add_explicit(&loss_f32[0], group_loss, memory_order_relaxed);
  }
}

kernel void wf2_fused_slab_affine_coeff16_ownerupdate_mse_vjp_direct_atomic_rgb_only_tensor(
    device const int* row_index_i32 [[buffer(0)]],
    device const int* candidate_row_offsets_i32 [[buffer(1)]],
    device const int* candidate_boundary_ids_i32 [[buffer(2)]],
    device const half* candidate_depth_coeff_f16 [[buffer(3)]],
    device const int* boundary_site_pairs_i32 [[buffer(4)]],
    device const float* sites_f32 [[buffer(5)]],
    device const float* site_rgba_f32 [[buffer(6)]],
    device const float* ray_coeff_f32 [[buffer(7)]],
    device const float* frame_t_f32 [[buffer(8)]],
    device const float* target_rgb_f32 [[buffer(9)]],
    device const int* config_i32 [[buffer(10)]],
    device const float* config_f32 [[buffer(11)]],
    device atomic_float* loss_f32 [[buffer(12)]],
    device atomic_float* grad_site_rgba_f32 [[buffer(13)]],
    uint sample_id [[thread_position_in_grid]]) {
  const uint track_count = uint(config_i32[1]);
  const uint site_count = uint(config_i32[2]);
  const uint frame_count = uint(config_i32[3]);
  const uint time_slab_count = uint(config_i32[4]);
  const uint row_count = uint(config_i32[5]);
  const uint candidate_count = uint(config_i32[6]);
  const uint boundary_count = uint(config_i32[7]);
  const uint total_samples = track_count * frame_count;
  if (sample_id >= total_samples) {
    return;
  }

  const uint out_base = sample_id * 3u;
  const float3 target_rgb = float3(
      target_rgb_f32[out_base + 0u],
      target_rgb_f32[out_base + 1u],
      target_rgb_f32[out_base + 2u]);
  const float inv_element_count = 1.0f / float(total_samples * 3u);

  const uint track_id = sample_id / frame_count;
  const uint frame_id = sample_id - track_id * frame_count;
  const float t = frame_t_f32[frame_id];
  const uint slab_id = wf2_replay_slab_id(t, time_slab_count);
  const int row_index_raw = row_index_i32[track_id];
  if (row_index_raw < 0 || uint(row_index_raw) >= row_count) {
    const float sample_loss = dot(target_rgb, target_rgb) * inv_element_count;
    atomic_fetch_add_explicit(&loss_f32[0], sample_loss, memory_order_relaxed);
    return;
  }

  const uint row = uint(row_index_raw) * time_slab_count + slab_id;
  const int begin_raw = candidate_row_offsets_i32[row];
  const int end_raw = candidate_row_offsets_i32[row + 1u];
  if (begin_raw < 0 || end_raw < begin_raw || uint(end_raw) > candidate_count) {
    const float sample_loss = dot(target_rgb, target_rgb) * inv_element_count;
    atomic_fetch_add_explicit(&loss_f32[0], sample_loss, memory_order_relaxed);
    return;
  }

  const uint clamped_site_count = min(site_count, WF2_MAX_REALRAY_SITES);
  const float near_depth = config_f32[0];
  const float far_depth = config_f32[1];
  const float invalid_epsilon = config_f32[2];
  const float transmittance_threshold = config_f32[3];
  const float3 origin = wf2_affine_origin_at(ray_coeff_f32, track_id, t);
  const float3 direction = wf2_affine_direction_at(ray_coeff_f32, track_id, t);

  float depths[WF2_MAX_REALRAY_FUSED_MSE_BOUNDARIES];
  uint boundary_ids[WF2_MAX_REALRAY_FUSED_MSE_BOUNDARIES];
  uint depth_count = 0u;
  for (uint cursor = uint(begin_raw); cursor < uint(end_raw); ++cursor) {
    const int boundary_id_raw = candidate_boundary_ids_i32[cursor];
    if (boundary_id_raw < 0 || uint(boundary_id_raw) >= boundary_count) {
      continue;
    }
    const uint coeff_base = cursor * 4u;
    const float numer =
        float(candidate_depth_coeff_f16[coeff_base + 0u]) + float(candidate_depth_coeff_f16[coeff_base + 1u]) * t;
    const float denom =
        float(candidate_depth_coeff_f16[coeff_base + 2u]) + float(candidate_depth_coeff_f16[coeff_base + 3u]) * t;
    if (fabs(denom) < invalid_epsilon) {
      continue;
    }
    const float depth = numer / denom;
    if (isfinite(depth) && depth >= near_depth && depth <= far_depth) {
      wf2_realray_insert_depth_with_boundary_fused_mse(
          depths, boundary_ids, depth_count, depth, uint(boundary_id_raw));
    }
  }

  uint owners[WF2_MAX_REALRAY_FUSED_MSE_SEGMENTS];
  float lengths[WF2_MAX_REALRAY_FUSED_MSE_SEGMENTS];
  float trans_before[WF2_MAX_REALRAY_FUSED_MSE_SEGMENTS];
  float segment_trans[WF2_MAX_REALRAY_FUSED_MSE_SEGMENTS];

  float3 rgb_accum = float3(0.0f, 0.0f, 0.0f);
  float transmittance = 1.0f;
  float previous_depth = near_depth;
  uint segment_count = 0u;
  int current_owner = -1;
  for (uint cut_id = 0u; cut_id <= depth_count; ++cut_id) {
    const float next_depth = cut_id < depth_count ? depths[cut_id] : far_depth;
    const float length = next_depth - previous_depth;
    if (length > 1.0e-8f && transmittance > transmittance_threshold && segment_count < WF2_MAX_REALRAY_FUSED_MSE_SEGMENTS) {
      const float mid_depth = 0.5f * (previous_depth + next_depth);
      uint owner;
      if (current_owner >= 0 && uint(current_owner) < clamped_site_count) {
        owner = uint(current_owner);
      } else {
        const float x = origin.x + direction.x * mid_depth;
        const float y = origin.y + direction.y * mid_depth;
        const float z = origin.z + direction.z * mid_depth;
        owner = wf2_realray_owner_at(sites_f32, clamped_site_count, x, y, z, t);
      }
      const uint rgba_base = owner * 4u;
      const float density = max(site_rgba_f32[rgba_base + 3u], 0.0f);
      const float seg_trans = exp(-density * length);
      const float seg_alpha = 1.0f - seg_trans;
      const float weight = transmittance * seg_alpha;
      const float3 rgb = float3(
          site_rgba_f32[rgba_base + 0u],
          site_rgba_f32[rgba_base + 1u],
          site_rgba_f32[rgba_base + 2u]);

      rgb_accum += weight * rgb;
      if (segment_count > 0u && owners[segment_count - 1u] == owner) {
        const uint previous_segment = segment_count - 1u;
        lengths[previous_segment] += length;
        segment_trans[previous_segment] *= seg_trans;
      } else {
        owners[segment_count] = owner;
        lengths[segment_count] = length;
        trans_before[segment_count] = transmittance;
        segment_trans[segment_count] = seg_trans;
        segment_count += 1u;
      }
      transmittance *= seg_trans;
      current_owner = int(owner);
    } else if (length <= 1.0e-8f || transmittance <= transmittance_threshold ||
               segment_count >= WF2_MAX_REALRAY_FUSED_MSE_SEGMENTS) {
      current_owner = -1;
    }
    previous_depth = next_depth;

    if (cut_id < depth_count) {
      const uint boundary_id = boundary_ids[cut_id];
      if (current_owner >= 0 && boundary_id < boundary_count) {
        const int left = boundary_site_pairs_i32[boundary_id * 2u + 0u];
        const int right = boundary_site_pairs_i32[boundary_id * 2u + 1u];
        if (left >= 0 && right >= 0 && uint(left) < clamped_site_count && uint(right) < clamped_site_count) {
          if (current_owner == left) {
            current_owner = right;
          } else if (current_owner == right) {
            current_owner = left;
          } else {
            current_owner = -1;
          }
        } else {
          current_owner = -1;
        }
      } else {
        current_owner = -1;
      }
    }
  }

  const float3 diff = rgb_accum - target_rgb;
  const float sample_loss = dot(diff, diff) * inv_element_count;
  atomic_fetch_add_explicit(&loss_f32[0], sample_loss, memory_order_relaxed);
  const float3 grad_rgb = (2.0f * inv_element_count) * diff;

  float adj_next_transmittance = 0.0f;
  for (int segment_id = int(segment_count) - 1; segment_id >= 0; --segment_id) {
    const uint owner = owners[segment_id];
    const uint rgba_base = owner * 4u;
    const float3 rgb = float3(
        site_rgba_f32[rgba_base + 0u],
        site_rgba_f32[rgba_base + 1u],
        site_rgba_f32[rgba_base + 2u]);
    const float seg_trans = segment_trans[segment_id];
    const float seg_alpha = 1.0f - seg_trans;
    const float weight = trans_before[segment_id] * seg_alpha;
    const float d_loss_d_weight = dot(grad_rgb, rgb);
    const float adj_trans_before =
        d_loss_d_weight * seg_alpha +
        adj_next_transmittance * seg_trans;
    const float adj_segment_alpha = d_loss_d_weight * trans_before[segment_id];
    const float adj_segment_trans =
        adj_next_transmittance * trans_before[segment_id] - adj_segment_alpha;
    float4 grad_rgba = float4(
        weight * grad_rgb.x,
        weight * grad_rgb.y,
        weight * grad_rgb.z,
        0.0f);
    const float raw_density = site_rgba_f32[rgba_base + 3u];
    if (raw_density > 0.0f) {
      grad_rgba.w = adj_segment_trans * (-lengths[segment_id] * seg_trans);
    }
    wf2_atomic_add4(grad_site_rgba_f32, rgba_base, grad_rgba);
    adj_next_transmittance = adj_trans_before;
  }
}

kernel void wf2_fused_slab_affine_coeff16_ownerupdate_i16_mse_vjp_direct_atomic_rgb_only_tensor(
    device const int* row_index_i32 [[buffer(0)]],
    device const int* candidate_row_offsets_i32 [[buffer(1)]],
    device const short* candidate_boundary_ids_i16 [[buffer(2)]],
    device const half* candidate_depth_coeff_f16 [[buffer(3)]],
    device const short* boundary_site_pairs_i16 [[buffer(4)]],
    device const float* sites_f32 [[buffer(5)]],
    device const float* site_rgba_f32 [[buffer(6)]],
    device const float* ray_coeff_f32 [[buffer(7)]],
    device const float* frame_t_f32 [[buffer(8)]],
    device const float* target_rgb_f32 [[buffer(9)]],
    device const int* config_i32 [[buffer(10)]],
    device const float* config_f32 [[buffer(11)]],
    device atomic_float* loss_f32 [[buffer(12)]],
    device atomic_float* grad_site_rgba_f32 [[buffer(13)]],
    uint sample_id [[thread_position_in_grid]]) {
  const uint track_count = uint(config_i32[1]);
  const uint site_count = uint(config_i32[2]);
  const uint frame_count = uint(config_i32[3]);
  const uint time_slab_count = uint(config_i32[4]);
  const uint row_count = uint(config_i32[5]);
  const uint candidate_count = uint(config_i32[6]);
  const uint boundary_count = uint(config_i32[7]);
  const uint total_samples = track_count * frame_count;
  if (sample_id >= total_samples) {
    return;
  }

  const uint out_base = sample_id * 3u;
  const float3 target_rgb = float3(
      target_rgb_f32[out_base + 0u],
      target_rgb_f32[out_base + 1u],
      target_rgb_f32[out_base + 2u]);
  const float inv_element_count = 1.0f / float(total_samples * 3u);

  const uint track_id = sample_id / frame_count;
  const uint frame_id = sample_id - track_id * frame_count;
  const float t = frame_t_f32[frame_id];
  const uint slab_id = wf2_replay_slab_id(t, time_slab_count);
  const int row_index_raw = row_index_i32[track_id];
  if (row_index_raw < 0 || uint(row_index_raw) >= row_count) {
    const float sample_loss = dot(target_rgb, target_rgb) * inv_element_count;
    atomic_fetch_add_explicit(&loss_f32[0], sample_loss, memory_order_relaxed);
    return;
  }

  const uint row = uint(row_index_raw) * time_slab_count + slab_id;
  const int begin_raw = candidate_row_offsets_i32[row];
  const int end_raw = candidate_row_offsets_i32[row + 1u];
  if (begin_raw < 0 || end_raw < begin_raw || uint(end_raw) > candidate_count) {
    const float sample_loss = dot(target_rgb, target_rgb) * inv_element_count;
    atomic_fetch_add_explicit(&loss_f32[0], sample_loss, memory_order_relaxed);
    return;
  }

  const uint clamped_site_count = min(site_count, WF2_MAX_REALRAY_SITES);
  const float near_depth = config_f32[0];
  const float far_depth = config_f32[1];
  const float invalid_epsilon = config_f32[2];
  const float transmittance_threshold = config_f32[3];
  const float3 origin = wf2_affine_origin_at(ray_coeff_f32, track_id, t);
  const float3 direction = wf2_affine_direction_at(ray_coeff_f32, track_id, t);

  float depths[WF2_MAX_REALRAY_FUSED_MSE_BOUNDARIES];
  uint boundary_ids[WF2_MAX_REALRAY_FUSED_MSE_BOUNDARIES];
  uint depth_count = 0u;
  for (uint cursor = uint(begin_raw); cursor < uint(end_raw); ++cursor) {
    const int boundary_id_raw = int(candidate_boundary_ids_i16[cursor]);
    if (boundary_id_raw < 0 || uint(boundary_id_raw) >= boundary_count) {
      continue;
    }
    const uint coeff_base = cursor * 4u;
    const float numer =
        float(candidate_depth_coeff_f16[coeff_base + 0u]) + float(candidate_depth_coeff_f16[coeff_base + 1u]) * t;
    const float denom =
        float(candidate_depth_coeff_f16[coeff_base + 2u]) + float(candidate_depth_coeff_f16[coeff_base + 3u]) * t;
    if (fabs(denom) < invalid_epsilon) {
      continue;
    }
    const float depth = numer / denom;
    if (isfinite(depth) && depth >= near_depth && depth <= far_depth) {
      wf2_realray_insert_depth_with_boundary_fused_mse(
          depths, boundary_ids, depth_count, depth, uint(boundary_id_raw));
    }
  }

  uint owners[WF2_MAX_REALRAY_FUSED_MSE_SEGMENTS];
  float lengths[WF2_MAX_REALRAY_FUSED_MSE_SEGMENTS];
  float trans_before[WF2_MAX_REALRAY_FUSED_MSE_SEGMENTS];
  float segment_trans[WF2_MAX_REALRAY_FUSED_MSE_SEGMENTS];

  float3 rgb_accum = float3(0.0f, 0.0f, 0.0f);
  float transmittance = 1.0f;
  float previous_depth = near_depth;
  uint segment_count = 0u;
  int current_owner = -1;
  for (uint cut_id = 0u; cut_id <= depth_count; ++cut_id) {
    const float next_depth = cut_id < depth_count ? depths[cut_id] : far_depth;
    const float length = next_depth - previous_depth;
    if (length > 1.0e-8f && transmittance > transmittance_threshold && segment_count < WF2_MAX_REALRAY_FUSED_MSE_SEGMENTS) {
      const float mid_depth = 0.5f * (previous_depth + next_depth);
      uint owner;
      if (current_owner >= 0 && uint(current_owner) < clamped_site_count) {
        owner = uint(current_owner);
      } else {
        const float x = origin.x + direction.x * mid_depth;
        const float y = origin.y + direction.y * mid_depth;
        const float z = origin.z + direction.z * mid_depth;
        owner = wf2_realray_owner_at(sites_f32, clamped_site_count, x, y, z, t);
      }
      const uint rgba_base = owner * 4u;
      const float density = max(site_rgba_f32[rgba_base + 3u], 0.0f);
      const float seg_trans = exp(-density * length);
      const float seg_alpha = 1.0f - seg_trans;
      const float weight = transmittance * seg_alpha;
      const float3 rgb = float3(
          site_rgba_f32[rgba_base + 0u],
          site_rgba_f32[rgba_base + 1u],
          site_rgba_f32[rgba_base + 2u]);

      rgb_accum += weight * rgb;
      if (segment_count > 0u && owners[segment_count - 1u] == owner) {
        const uint previous_segment = segment_count - 1u;
        lengths[previous_segment] += length;
        segment_trans[previous_segment] *= seg_trans;
      } else {
        owners[segment_count] = owner;
        lengths[segment_count] = length;
        trans_before[segment_count] = transmittance;
        segment_trans[segment_count] = seg_trans;
        segment_count += 1u;
      }
      transmittance *= seg_trans;
      current_owner = int(owner);
    } else if (length <= 1.0e-8f || transmittance <= transmittance_threshold ||
               segment_count >= WF2_MAX_REALRAY_FUSED_MSE_SEGMENTS) {
      current_owner = -1;
    }
    previous_depth = next_depth;

    if (cut_id < depth_count) {
      const uint boundary_id = boundary_ids[cut_id];
      if (current_owner >= 0 && boundary_id < boundary_count) {
        const int left = int(boundary_site_pairs_i16[boundary_id * 2u + 0u]);
        const int right = int(boundary_site_pairs_i16[boundary_id * 2u + 1u]);
        if (left >= 0 && right >= 0 && uint(left) < clamped_site_count && uint(right) < clamped_site_count) {
          if (current_owner == left) {
            current_owner = right;
          } else if (current_owner == right) {
            current_owner = left;
          } else {
            current_owner = -1;
          }
        } else {
          current_owner = -1;
        }
      } else {
        current_owner = -1;
      }
    }
  }

  const float3 diff = rgb_accum - target_rgb;
  const float sample_loss = dot(diff, diff) * inv_element_count;
  atomic_fetch_add_explicit(&loss_f32[0], sample_loss, memory_order_relaxed);
  const float3 grad_rgb = (2.0f * inv_element_count) * diff;

  float adj_next_transmittance = 0.0f;
  for (int segment_id = int(segment_count) - 1; segment_id >= 0; --segment_id) {
    const uint owner = owners[segment_id];
    const uint rgba_base = owner * 4u;
    const float3 rgb = float3(
        site_rgba_f32[rgba_base + 0u],
        site_rgba_f32[rgba_base + 1u],
        site_rgba_f32[rgba_base + 2u]);
    const float seg_trans = segment_trans[segment_id];
    const float seg_alpha = 1.0f - seg_trans;
    const float weight = trans_before[segment_id] * seg_alpha;
    const float d_loss_d_weight = dot(grad_rgb, rgb);
    const float adj_trans_before =
        d_loss_d_weight * seg_alpha +
        adj_next_transmittance * seg_trans;
    const float adj_segment_alpha = d_loss_d_weight * trans_before[segment_id];
    const float adj_segment_trans =
        adj_next_transmittance * trans_before[segment_id] - adj_segment_alpha;
    float4 grad_rgba = float4(
        weight * grad_rgb.x,
        weight * grad_rgb.y,
        weight * grad_rgb.z,
        0.0f);
    const float raw_density = site_rgba_f32[rgba_base + 3u];
    if (raw_density > 0.0f) {
      grad_rgba.w = adj_segment_trans * (-lengths[segment_id] * seg_trans);
    }
    wf2_atomic_add4(grad_site_rgba_f32, rgba_base, grad_rgba);
    adj_next_transmittance = adj_trans_before;
  }
}

kernel void wf2_fused_slab_affine_coeff16_ownerkeep_i16_mse_vjp_direct_atomic_rgb_only_tensor(
    device const int* row_index_i32 [[buffer(0)]],
    device const int* candidate_row_offsets_i32 [[buffer(1)]],
    device const short* candidate_boundary_ids_i16 [[buffer(2)]],
    device const half* candidate_depth_coeff_f16 [[buffer(3)]],
    device const short* boundary_site_pairs_i16 [[buffer(4)]],
    device const float* sites_f32 [[buffer(5)]],
    device const float* site_rgba_f32 [[buffer(6)]],
    device const float* ray_coeff_f32 [[buffer(7)]],
    device const float* frame_t_f32 [[buffer(8)]],
    device const float* target_rgb_f32 [[buffer(9)]],
    device const int* config_i32 [[buffer(10)]],
    device const float* config_f32 [[buffer(11)]],
    device atomic_float* loss_f32 [[buffer(12)]],
    device atomic_float* grad_site_rgba_f32 [[buffer(13)]],
    uint sample_id [[thread_position_in_grid]]) {
  const uint track_count = uint(config_i32[1]);
  const uint site_count = uint(config_i32[2]);
  const uint frame_count = uint(config_i32[3]);
  const uint time_slab_count = uint(config_i32[4]);
  const uint row_count = uint(config_i32[5]);
  const uint candidate_count = uint(config_i32[6]);
  const uint boundary_count = uint(config_i32[7]);
  const uint total_samples = track_count * frame_count;
  if (sample_id >= total_samples) {
    return;
  }

  const uint out_base = sample_id * 3u;
  const float3 target_rgb = float3(
      target_rgb_f32[out_base + 0u],
      target_rgb_f32[out_base + 1u],
      target_rgb_f32[out_base + 2u]);
  const float inv_element_count = 1.0f / float(total_samples * 3u);

  const uint track_id = sample_id / frame_count;
  const uint frame_id = sample_id - track_id * frame_count;
  const float t = frame_t_f32[frame_id];
  const uint slab_id = wf2_replay_slab_id(t, time_slab_count);
  const int row_index_raw = row_index_i32[track_id];
  if (row_index_raw < 0 || uint(row_index_raw) >= row_count) {
    const float sample_loss = dot(target_rgb, target_rgb) * inv_element_count;
    atomic_fetch_add_explicit(&loss_f32[0], sample_loss, memory_order_relaxed);
    return;
  }

  const uint row = uint(row_index_raw) * time_slab_count + slab_id;
  const int begin_raw = candidate_row_offsets_i32[row];
  const int end_raw = candidate_row_offsets_i32[row + 1u];
  if (begin_raw < 0 || end_raw < begin_raw || uint(end_raw) > candidate_count) {
    const float sample_loss = dot(target_rgb, target_rgb) * inv_element_count;
    atomic_fetch_add_explicit(&loss_f32[0], sample_loss, memory_order_relaxed);
    return;
  }

  const uint clamped_site_count = min(site_count, WF2_MAX_REALRAY_SITES);
  const float near_depth = config_f32[0];
  const float far_depth = config_f32[1];
  const float invalid_epsilon = config_f32[2];
  const float transmittance_threshold = config_f32[3];
  const float3 origin = wf2_affine_origin_at(ray_coeff_f32, track_id, t);
  const float3 direction = wf2_affine_direction_at(ray_coeff_f32, track_id, t);

  float depths[WF2_MAX_REALRAY_FUSED_MSE_BOUNDARIES];
  uint boundary_ids[WF2_MAX_REALRAY_FUSED_MSE_BOUNDARIES];
  uint depth_count = 0u;
  for (uint cursor = uint(begin_raw); cursor < uint(end_raw); ++cursor) {
    const int boundary_id_raw = int(candidate_boundary_ids_i16[cursor]);
    if (boundary_id_raw < 0 || uint(boundary_id_raw) >= boundary_count) {
      continue;
    }
    const uint coeff_base = cursor * 4u;
    const float numer =
        float(candidate_depth_coeff_f16[coeff_base + 0u]) + float(candidate_depth_coeff_f16[coeff_base + 1u]) * t;
    const float denom =
        float(candidate_depth_coeff_f16[coeff_base + 2u]) + float(candidate_depth_coeff_f16[coeff_base + 3u]) * t;
    if (fabs(denom) < invalid_epsilon) {
      continue;
    }
    const float depth = numer / denom;
    if (isfinite(depth) && depth >= near_depth && depth <= far_depth) {
      wf2_realray_insert_depth_with_boundary_fused_mse(
          depths, boundary_ids, depth_count, depth, uint(boundary_id_raw));
    }
  }

  uint owners[WF2_MAX_REALRAY_FUSED_MSE_SEGMENTS];
  float lengths[WF2_MAX_REALRAY_FUSED_MSE_SEGMENTS];
  float trans_before[WF2_MAX_REALRAY_FUSED_MSE_SEGMENTS];
  float segment_trans[WF2_MAX_REALRAY_FUSED_MSE_SEGMENTS];

  float3 rgb_accum = float3(0.0f, 0.0f, 0.0f);
  float transmittance = 1.0f;
  float previous_depth = near_depth;
  uint segment_count = 0u;
  int current_owner = -1;
  for (uint cut_id = 0u; cut_id <= depth_count; ++cut_id) {
    const float next_depth = cut_id < depth_count ? depths[cut_id] : far_depth;
    const float length = next_depth - previous_depth;
    if (length > 1.0e-8f && transmittance > transmittance_threshold && segment_count < WF2_MAX_REALRAY_FUSED_MSE_SEGMENTS) {
      const float mid_depth = 0.5f * (previous_depth + next_depth);
      uint owner;
      if (current_owner >= 0 && uint(current_owner) < clamped_site_count) {
        owner = uint(current_owner);
      } else {
        const float x = origin.x + direction.x * mid_depth;
        const float y = origin.y + direction.y * mid_depth;
        const float z = origin.z + direction.z * mid_depth;
        owner = wf2_realray_owner_at(sites_f32, clamped_site_count, x, y, z, t);
      }
      const uint rgba_base = owner * 4u;
      const float density = max(site_rgba_f32[rgba_base + 3u], 0.0f);
      const float seg_trans = exp(-density * length);
      const float seg_alpha = 1.0f - seg_trans;
      const float weight = transmittance * seg_alpha;
      const float3 rgb = float3(
          site_rgba_f32[rgba_base + 0u],
          site_rgba_f32[rgba_base + 1u],
          site_rgba_f32[rgba_base + 2u]);

      rgb_accum += weight * rgb;
      if (segment_count > 0u && owners[segment_count - 1u] == owner) {
        const uint previous_segment = segment_count - 1u;
        lengths[previous_segment] += length;
        segment_trans[previous_segment] *= seg_trans;
      } else {
        owners[segment_count] = owner;
        lengths[segment_count] = length;
        trans_before[segment_count] = transmittance;
        segment_trans[segment_count] = seg_trans;
        segment_count += 1u;
      }
      transmittance *= seg_trans;
      current_owner = int(owner);
    // Duplicate-depth boundary cuts have zero length but still carry owner transitions.
    // Preserve current_owner through them so the owner-cache path is not defeated by ties.
    } else if (transmittance <= transmittance_threshold ||
               segment_count >= WF2_MAX_REALRAY_FUSED_MSE_SEGMENTS) {
      current_owner = -1;
    }
    previous_depth = next_depth;

    if (cut_id < depth_count) {
      const uint boundary_id = boundary_ids[cut_id];
      if (current_owner >= 0 && boundary_id < boundary_count) {
        const int left = int(boundary_site_pairs_i16[boundary_id * 2u + 0u]);
        const int right = int(boundary_site_pairs_i16[boundary_id * 2u + 1u]);
        if (left >= 0 && right >= 0 && uint(left) < clamped_site_count && uint(right) < clamped_site_count) {
          if (current_owner == left) {
            current_owner = right;
          } else if (current_owner == right) {
            current_owner = left;
          }
        } else {
          current_owner = -1;
        }
      } else if (current_owner >= 0) {
        current_owner = -1;
      }
    }
  }

  const float3 diff = rgb_accum - target_rgb;
  const float sample_loss = dot(diff, diff) * inv_element_count;
  atomic_fetch_add_explicit(&loss_f32[0], sample_loss, memory_order_relaxed);
  const float3 grad_rgb = (2.0f * inv_element_count) * diff;

  float adj_next_transmittance = 0.0f;
  for (int segment_id = int(segment_count) - 1; segment_id >= 0; --segment_id) {
    const uint owner = owners[segment_id];
    const uint rgba_base = owner * 4u;
    const float3 rgb = float3(
        site_rgba_f32[rgba_base + 0u],
        site_rgba_f32[rgba_base + 1u],
        site_rgba_f32[rgba_base + 2u]);
    const float seg_trans = segment_trans[segment_id];
    const float seg_alpha = 1.0f - seg_trans;
    const float weight = trans_before[segment_id] * seg_alpha;
    const float d_loss_d_weight = dot(grad_rgb, rgb);
    const float adj_trans_before =
        d_loss_d_weight * seg_alpha +
        adj_next_transmittance * seg_trans;
    const float adj_segment_alpha = d_loss_d_weight * trans_before[segment_id];
    const float adj_segment_trans =
        adj_next_transmittance * trans_before[segment_id] - adj_segment_alpha;
    float4 grad_rgba = float4(
        weight * grad_rgb.x,
        weight * grad_rgb.y,
        weight * grad_rgb.z,
        0.0f);
    const float raw_density = site_rgba_f32[rgba_base + 3u];
    if (raw_density > 0.0f) {
      grad_rgba.w = adj_segment_trans * (-lengths[segment_id] * seg_trans);
    }
    wf2_atomic_add4(grad_site_rgba_f32, rgba_base, grad_rgba);
    adj_next_transmittance = adj_trans_before;
  }
}

kernel void wf2_fused_slab_affine_coeff16_ownerkeep_mse_vjp_direct_atomic_rgb_only_tensor(
    device const int* row_index_i32 [[buffer(0)]],
    device const int* candidate_row_offsets_i32 [[buffer(1)]],
    device const int* candidate_boundary_ids_i32 [[buffer(2)]],
    device const half* candidate_depth_coeff_f16 [[buffer(3)]],
    device const int* boundary_site_pairs_i32 [[buffer(4)]],
    device const float* sites_f32 [[buffer(5)]],
    device const float* site_rgba_f32 [[buffer(6)]],
    device const float* ray_coeff_f32 [[buffer(7)]],
    device const float* frame_t_f32 [[buffer(8)]],
    device const float* target_rgb_f32 [[buffer(9)]],
    device const int* config_i32 [[buffer(10)]],
    device const float* config_f32 [[buffer(11)]],
    device atomic_float* loss_f32 [[buffer(12)]],
    device atomic_float* grad_site_rgba_f32 [[buffer(13)]],
    uint sample_id [[thread_position_in_grid]]) {
  const uint track_count = uint(config_i32[1]);
  const uint site_count = uint(config_i32[2]);
  const uint frame_count = uint(config_i32[3]);
  const uint time_slab_count = uint(config_i32[4]);
  const uint row_count = uint(config_i32[5]);
  const uint candidate_count = uint(config_i32[6]);
  const uint boundary_count = uint(config_i32[7]);
  const uint total_samples = track_count * frame_count;
  if (sample_id >= total_samples) {
    return;
  }

  const uint out_base = sample_id * 3u;
  const float3 target_rgb = float3(
      target_rgb_f32[out_base + 0u],
      target_rgb_f32[out_base + 1u],
      target_rgb_f32[out_base + 2u]);
  const float inv_element_count = 1.0f / float(total_samples * 3u);

  const uint track_id = sample_id / frame_count;
  const uint frame_id = sample_id - track_id * frame_count;
  const float t = frame_t_f32[frame_id];
  const uint slab_id = wf2_replay_slab_id(t, time_slab_count);
  const int row_index_raw = row_index_i32[track_id];
  if (row_index_raw < 0 || uint(row_index_raw) >= row_count) {
    const float sample_loss = dot(target_rgb, target_rgb) * inv_element_count;
    atomic_fetch_add_explicit(&loss_f32[0], sample_loss, memory_order_relaxed);
    return;
  }

  const uint row = uint(row_index_raw) * time_slab_count + slab_id;
  const int begin_raw = candidate_row_offsets_i32[row];
  const int end_raw = candidate_row_offsets_i32[row + 1u];
  if (begin_raw < 0 || end_raw < begin_raw || uint(end_raw) > candidate_count) {
    const float sample_loss = dot(target_rgb, target_rgb) * inv_element_count;
    atomic_fetch_add_explicit(&loss_f32[0], sample_loss, memory_order_relaxed);
    return;
  }

  const uint clamped_site_count = min(site_count, WF2_MAX_REALRAY_SITES);
  const float near_depth = config_f32[0];
  const float far_depth = config_f32[1];
  const float invalid_epsilon = config_f32[2];
  const float transmittance_threshold = config_f32[3];
  const float3 origin = wf2_affine_origin_at(ray_coeff_f32, track_id, t);
  const float3 direction = wf2_affine_direction_at(ray_coeff_f32, track_id, t);

  float depths[WF2_MAX_REALRAY_FUSED_MSE_BOUNDARIES];
  uint boundary_ids[WF2_MAX_REALRAY_FUSED_MSE_BOUNDARIES];
  uint depth_count = 0u;
  for (uint cursor = uint(begin_raw); cursor < uint(end_raw); ++cursor) {
    const int boundary_id_raw = candidate_boundary_ids_i32[cursor];
    if (boundary_id_raw < 0 || uint(boundary_id_raw) >= boundary_count) {
      continue;
    }
    const uint coeff_base = cursor * 4u;
    const float numer =
        float(candidate_depth_coeff_f16[coeff_base + 0u]) + float(candidate_depth_coeff_f16[coeff_base + 1u]) * t;
    const float denom =
        float(candidate_depth_coeff_f16[coeff_base + 2u]) + float(candidate_depth_coeff_f16[coeff_base + 3u]) * t;
    if (fabs(denom) < invalid_epsilon) {
      continue;
    }
    const float depth = numer / denom;
    if (isfinite(depth) && depth >= near_depth && depth <= far_depth) {
      wf2_realray_insert_depth_with_boundary_fused_mse(
          depths, boundary_ids, depth_count, depth, uint(boundary_id_raw));
    }
  }

  uint owners[WF2_MAX_REALRAY_FUSED_MSE_SEGMENTS];
  float lengths[WF2_MAX_REALRAY_FUSED_MSE_SEGMENTS];
  float trans_before[WF2_MAX_REALRAY_FUSED_MSE_SEGMENTS];
  float segment_trans[WF2_MAX_REALRAY_FUSED_MSE_SEGMENTS];

  float3 rgb_accum = float3(0.0f, 0.0f, 0.0f);
  float transmittance = 1.0f;
  float previous_depth = near_depth;
  uint segment_count = 0u;
  int current_owner = -1;
  for (uint cut_id = 0u; cut_id <= depth_count; ++cut_id) {
    const float next_depth = cut_id < depth_count ? depths[cut_id] : far_depth;
    const float length = next_depth - previous_depth;
    if (length > 1.0e-8f && transmittance > transmittance_threshold && segment_count < WF2_MAX_REALRAY_FUSED_MSE_SEGMENTS) {
      const float mid_depth = 0.5f * (previous_depth + next_depth);
      uint owner;
      if (current_owner >= 0 && uint(current_owner) < clamped_site_count) {
        owner = uint(current_owner);
      } else {
        const float x = origin.x + direction.x * mid_depth;
        const float y = origin.y + direction.y * mid_depth;
        const float z = origin.z + direction.z * mid_depth;
        owner = wf2_realray_owner_at(sites_f32, clamped_site_count, x, y, z, t);
      }
      const uint rgba_base = owner * 4u;
      const float density = max(site_rgba_f32[rgba_base + 3u], 0.0f);
      const float seg_trans = exp(-density * length);
      const float seg_alpha = 1.0f - seg_trans;
      const float weight = transmittance * seg_alpha;
      const float3 rgb = float3(
          site_rgba_f32[rgba_base + 0u],
          site_rgba_f32[rgba_base + 1u],
          site_rgba_f32[rgba_base + 2u]);

      rgb_accum += weight * rgb;
      if (segment_count > 0u && owners[segment_count - 1u] == owner) {
        const uint previous_segment = segment_count - 1u;
        lengths[previous_segment] += length;
        segment_trans[previous_segment] *= seg_trans;
      } else {
        owners[segment_count] = owner;
        lengths[segment_count] = length;
        trans_before[segment_count] = transmittance;
        segment_trans[segment_count] = seg_trans;
        segment_count += 1u;
      }
      transmittance *= seg_trans;
      current_owner = int(owner);
    } else if (length <= 1.0e-8f || transmittance <= transmittance_threshold ||
               segment_count >= WF2_MAX_REALRAY_FUSED_MSE_SEGMENTS) {
      current_owner = -1;
    }
    previous_depth = next_depth;

    if (cut_id < depth_count) {
      const uint boundary_id = boundary_ids[cut_id];
      if (current_owner >= 0 && boundary_id < boundary_count) {
        const int left = boundary_site_pairs_i32[boundary_id * 2u + 0u];
        const int right = boundary_site_pairs_i32[boundary_id * 2u + 1u];
        if (left >= 0 && right >= 0 && uint(left) < clamped_site_count && uint(right) < clamped_site_count) {
          if (current_owner == left) {
            current_owner = right;
          } else if (current_owner == right) {
            current_owner = left;
          }
        } else {
          current_owner = -1;
        }
      } else if (current_owner >= 0) {
        current_owner = -1;
      }
    }
  }

  const float3 diff = rgb_accum - target_rgb;
  const float sample_loss = dot(diff, diff) * inv_element_count;
  atomic_fetch_add_explicit(&loss_f32[0], sample_loss, memory_order_relaxed);
  const float3 grad_rgb = (2.0f * inv_element_count) * diff;

  float adj_next_transmittance = 0.0f;
  for (int segment_id = int(segment_count) - 1; segment_id >= 0; --segment_id) {
    const uint owner = owners[segment_id];
    const uint rgba_base = owner * 4u;
    const float3 rgb = float3(
        site_rgba_f32[rgba_base + 0u],
        site_rgba_f32[rgba_base + 1u],
        site_rgba_f32[rgba_base + 2u]);
    const float seg_trans = segment_trans[segment_id];
    const float seg_alpha = 1.0f - seg_trans;
    const float weight = trans_before[segment_id] * seg_alpha;
    const float d_loss_d_weight = dot(grad_rgb, rgb);
    const float adj_trans_before =
        d_loss_d_weight * seg_alpha +
        adj_next_transmittance * seg_trans;
    const float adj_segment_alpha = d_loss_d_weight * trans_before[segment_id];
    const float adj_segment_trans =
        adj_next_transmittance * trans_before[segment_id] - adj_segment_alpha;
    float4 grad_rgba = float4(
        weight * grad_rgb.x,
        weight * grad_rgb.y,
        weight * grad_rgb.z,
        0.0f);
    const float raw_density = site_rgba_f32[rgba_base + 3u];
    if (raw_density > 0.0f) {
      grad_rgba.w = adj_segment_trans * (-lengths[segment_id] * seg_trans);
    }
    wf2_atomic_add4(grad_site_rgba_f32, rgba_base, grad_rgba);
    adj_next_transmittance = adj_trans_before;
  }
}

kernel void wf2_fused_slab_affine_coeff16_mse_vjp_direct_atomic_track_rgb_only_tensor(
    device const int* row_index_i32 [[buffer(0)]],
    device const int* candidate_row_offsets_i32 [[buffer(1)]],
    device const half* candidate_depth_coeff_f16 [[buffer(2)]],
    device const float* sites_f32 [[buffer(3)]],
    device const float* site_rgba_f32 [[buffer(4)]],
    device const float* ray_coeff_f32 [[buffer(5)]],
    device const float* frame_t_f32 [[buffer(6)]],
    device const float* target_rgb_f32 [[buffer(7)]],
    device const int* config_i32 [[buffer(8)]],
    device const float* config_f32 [[buffer(9)]],
    device atomic_float* loss_f32 [[buffer(10)]],
    device atomic_float* grad_site_rgba_f32 [[buffer(11)]],
    uint track_id [[thread_position_in_grid]]) {
  const uint track_count = uint(config_i32[1]);
  const uint site_count = uint(config_i32[2]);
  const uint frame_count = uint(config_i32[3]);
  const uint time_slab_count = uint(config_i32[4]);
  const uint row_count = uint(config_i32[5]);
  const uint candidate_count = uint(config_i32[6]);
  if (track_id >= track_count) {
    return;
  }

  const uint total_samples = track_count * frame_count;
  const float inv_element_count = 1.0f / float(total_samples * 3u);
  const uint clamped_site_count = min(site_count, WF2_MAX_REALRAY_SITES);
  const float near_depth = config_f32[0];
  const float far_depth = config_f32[1];
  const float invalid_epsilon = config_f32[2];
  const float transmittance_threshold = config_f32[3];
  float4 grad_accum[WF2_MAX_REALRAY_SITES];
  for (uint site_id = 0u; site_id < clamped_site_count; ++site_id) {
    grad_accum[site_id] = float4(0.0f, 0.0f, 0.0f, 0.0f);
  }

  float track_loss = 0.0f;
  const int row_index_raw = row_index_i32[track_id];
  if (row_index_raw < 0 || uint(row_index_raw) >= row_count) {
    for (uint frame_id = 0u; frame_id < frame_count; ++frame_id) {
      const uint out_base = (track_id * frame_count + frame_id) * 3u;
      const float3 target_rgb = float3(
          target_rgb_f32[out_base + 0u],
          target_rgb_f32[out_base + 1u],
          target_rgb_f32[out_base + 2u]);
      track_loss += dot(target_rgb, target_rgb) * inv_element_count;
    }
    atomic_fetch_add_explicit(&loss_f32[0], track_loss, memory_order_relaxed);
    return;
  }

  const uint row_base = uint(row_index_raw) * time_slab_count;
  for (uint frame_id = 0u; frame_id < frame_count; ++frame_id) {
    const uint sample_id = track_id * frame_count + frame_id;
    const uint out_base = sample_id * 3u;
    const float3 target_rgb = float3(
        target_rgb_f32[out_base + 0u],
        target_rgb_f32[out_base + 1u],
        target_rgb_f32[out_base + 2u]);
    const float t = frame_t_f32[frame_id];
    const uint slab_id = wf2_replay_slab_id(t, time_slab_count);
    const uint row = row_base + slab_id;
    const int begin_raw = candidate_row_offsets_i32[row];
    const int end_raw = candidate_row_offsets_i32[row + 1u];
    if (begin_raw < 0 || end_raw < begin_raw || uint(end_raw) > candidate_count) {
      track_loss += dot(target_rgb, target_rgb) * inv_element_count;
      continue;
    }

    const float3 origin = wf2_affine_origin_at(ray_coeff_f32, track_id, t);
    const float3 direction = wf2_affine_direction_at(ray_coeff_f32, track_id, t);

    float depths[WF2_MAX_REALRAY_FUSED_MSE_BOUNDARIES];
    uint depth_count = 0u;
    for (uint cursor = uint(begin_raw); cursor < uint(end_raw); ++cursor) {
      const uint coeff_base = cursor * 4u;
      const float numer =
          float(candidate_depth_coeff_f16[coeff_base + 0u]) + float(candidate_depth_coeff_f16[coeff_base + 1u]) * t;
      const float denom =
          float(candidate_depth_coeff_f16[coeff_base + 2u]) + float(candidate_depth_coeff_f16[coeff_base + 3u]) * t;
      if (fabs(denom) < invalid_epsilon) {
        continue;
      }
      const float depth = numer / denom;
      if (isfinite(depth) && depth >= near_depth && depth <= far_depth) {
        wf2_realray_insert_depth_fused_mse(depths, depth_count, depth);
      }
    }

    uint owners[WF2_MAX_REALRAY_FUSED_MSE_SEGMENTS];
    float lengths[WF2_MAX_REALRAY_FUSED_MSE_SEGMENTS];
    float trans_before[WF2_MAX_REALRAY_FUSED_MSE_SEGMENTS];
    float segment_trans[WF2_MAX_REALRAY_FUSED_MSE_SEGMENTS];

    float3 rgb_accum = float3(0.0f, 0.0f, 0.0f);
    float transmittance = 1.0f;
    float previous_depth = near_depth;
    uint segment_count = 0u;
    for (uint cut_id = 0u; cut_id <= depth_count; ++cut_id) {
      const float next_depth = cut_id < depth_count ? depths[cut_id] : far_depth;
      const float length = next_depth - previous_depth;
      if (length > 1.0e-8f && transmittance > transmittance_threshold && segment_count < WF2_MAX_REALRAY_FUSED_MSE_SEGMENTS) {
        const float mid_depth = 0.5f * (previous_depth + next_depth);
        const float x = origin.x + direction.x * mid_depth;
        const float y = origin.y + direction.y * mid_depth;
        const float z = origin.z + direction.z * mid_depth;
        const uint owner = wf2_realray_owner_at(sites_f32, clamped_site_count, x, y, z, t);
        const uint rgba_base = owner * 4u;
        const float density = max(site_rgba_f32[rgba_base + 3u], 0.0f);
        const float seg_trans = exp(-density * length);
        const float seg_alpha = 1.0f - seg_trans;
        const float weight = transmittance * seg_alpha;
        const float3 rgb = float3(
            site_rgba_f32[rgba_base + 0u],
            site_rgba_f32[rgba_base + 1u],
            site_rgba_f32[rgba_base + 2u]);

        rgb_accum += weight * rgb;
        if (segment_count > 0u && owners[segment_count - 1u] == owner) {
          const uint previous_segment = segment_count - 1u;
          lengths[previous_segment] += length;
          segment_trans[previous_segment] *= seg_trans;
        } else {
          owners[segment_count] = owner;
          lengths[segment_count] = length;
          trans_before[segment_count] = transmittance;
          segment_trans[segment_count] = seg_trans;
          segment_count += 1u;
        }
        transmittance *= seg_trans;
      }
      previous_depth = next_depth;
    }

    const float3 diff = rgb_accum - target_rgb;
    track_loss += dot(diff, diff) * inv_element_count;
    const float3 grad_rgb = (2.0f * inv_element_count) * diff;

    float adj_next_transmittance = 0.0f;
    for (int segment_id = int(segment_count) - 1; segment_id >= 0; --segment_id) {
      const uint owner = owners[segment_id];
      const uint rgba_base = owner * 4u;
      const float3 rgb = float3(
          site_rgba_f32[rgba_base + 0u],
          site_rgba_f32[rgba_base + 1u],
          site_rgba_f32[rgba_base + 2u]);
      const float seg_trans = segment_trans[segment_id];
      const float seg_alpha = 1.0f - seg_trans;
      const float weight = trans_before[segment_id] * seg_alpha;
      const float d_loss_d_weight = dot(grad_rgb, rgb);
      const float adj_trans_before =
          d_loss_d_weight * seg_alpha +
          adj_next_transmittance * seg_trans;
      const float adj_segment_alpha = d_loss_d_weight * trans_before[segment_id];
      const float adj_segment_trans =
          adj_next_transmittance * trans_before[segment_id] - adj_segment_alpha;
      grad_accum[owner].x += weight * grad_rgb.x;
      grad_accum[owner].y += weight * grad_rgb.y;
      grad_accum[owner].z += weight * grad_rgb.z;
      const float raw_density = site_rgba_f32[rgba_base + 3u];
      if (raw_density > 0.0f) {
        grad_accum[owner].w += adj_segment_trans * (-lengths[segment_id] * seg_trans);
      }
      adj_next_transmittance = adj_trans_before;
    }
  }

  atomic_fetch_add_explicit(&loss_f32[0], track_loss, memory_order_relaxed);
  for (uint site_id = 0u; site_id < clamped_site_count; ++site_id) {
    const float4 grad = grad_accum[site_id];
    if (grad.x != 0.0f || grad.y != 0.0f || grad.z != 0.0f || grad.w != 0.0f) {
      wf2_atomic_add4(grad_site_rgba_f32, site_id * 4u, grad);
    }
  }
}

kernel void wf2_fused_slab_affine_num32_den16_vjp_direct_atomic_track_tensor(
    device const int* row_index_i32 [[buffer(0)]],
    device const int* candidate_row_offsets_i32 [[buffer(1)]],
    device const float* candidate_depth_num_f32 [[buffer(2)]],
    device const half* candidate_depth_den_f16 [[buffer(3)]],
    device const float* sites_f32 [[buffer(4)]],
    device const float* site_rgba_f32 [[buffer(5)]],
    device const float* ray_coeff_f32 [[buffer(6)]],
    device const float* frame_t_f32 [[buffer(7)]],
    device const float* grad_rgb_f32 [[buffer(8)]],
    device const float* grad_alpha_f32 [[buffer(9)]],
    device const float* grad_depth_f32 [[buffer(10)]],
    device const int* config_i32 [[buffer(11)]],
    device const float* config_f32 [[buffer(12)]],
    device atomic_float* grad_site_rgba_f32 [[buffer(13)]],
    uint track_id [[thread_position_in_grid]]) {
  const uint track_count = uint(config_i32[1]);
  const uint site_count = uint(config_i32[2]);
  const uint frame_count = uint(config_i32[3]);
  const uint time_slab_count = uint(config_i32[4]);
  const uint row_count = uint(config_i32[5]);
  const uint candidate_count = uint(config_i32[6]);
  if (track_id >= track_count) {
    return;
  }

  const uint clamped_site_count = min(site_count, WF2_MAX_REALRAY_SITES);
  const float near_depth = config_f32[0];
  const float far_depth = config_f32[1];
  const float invalid_epsilon = config_f32[2];
  const float transmittance_threshold = config_f32[3];
  float4 grad_accum[WF2_MAX_REALRAY_SITES];
  for (uint site_id = 0u; site_id < clamped_site_count; ++site_id) {
    grad_accum[site_id] = float4(0.0f, 0.0f, 0.0f, 0.0f);
  }

  const int row_index_raw = row_index_i32[track_id];
  if (row_index_raw < 0 || uint(row_index_raw) >= row_count) {
    return;
  }
  const uint row_base = uint(row_index_raw) * time_slab_count;

  for (uint frame_id = 0u; frame_id < frame_count; ++frame_id) {
    const float t = frame_t_f32[frame_id];
    const float3 origin = wf2_affine_origin_at(ray_coeff_f32, track_id, t);
    const float3 direction = wf2_affine_direction_at(ray_coeff_f32, track_id, t);
    const uint slab_id = wf2_replay_slab_id(t, time_slab_count);
    const uint row = row_base + slab_id;
    const int begin_raw = candidate_row_offsets_i32[row];
    const int end_raw = candidate_row_offsets_i32[row + 1u];
    if (begin_raw < 0 || end_raw < begin_raw || uint(end_raw) > candidate_count) {
      continue;
    }

    float depths[WF2_MAX_REALRAY_BOUNDARIES];
    uint depth_count = 0u;
    for (uint cursor = uint(begin_raw); cursor < uint(end_raw); ++cursor) {
      const uint coeff_base = cursor * 2u;
      const float numer = candidate_depth_num_f32[coeff_base + 0u] + candidate_depth_num_f32[coeff_base + 1u] * t;
      const float denom = float(candidate_depth_den_f16[coeff_base + 0u]) +
          float(candidate_depth_den_f16[coeff_base + 1u]) * t;
      if (fabs(denom) < invalid_epsilon) {
        continue;
      }
      const float depth = numer / denom;
      if (isfinite(depth) && depth >= near_depth && depth <= far_depth) {
        wf2_realray_insert_depth(depths, depth_count, depth);
      }
    }

    uint owners[WF2_MAX_REALRAY_SEGMENTS];
    float lengths[WF2_MAX_REALRAY_SEGMENTS];
    float mids[WF2_MAX_REALRAY_SEGMENTS];
    float trans_before[WF2_MAX_REALRAY_SEGMENTS];
    float segment_trans[WF2_MAX_REALRAY_SEGMENTS];
    float segment_alpha[WF2_MAX_REALRAY_SEGMENTS];
    float weights[WF2_MAX_REALRAY_SEGMENTS];
    float3 segment_rgb[WF2_MAX_REALRAY_SEGMENTS];

    float alpha_accum = 0.0f;
    float depth_weighted = 0.0f;
    float transmittance = 1.0f;
    float previous_depth = near_depth;
    uint segment_count = 0u;
    for (uint cut_id = 0u; cut_id <= depth_count; ++cut_id) {
      const float next_depth = cut_id < depth_count ? depths[cut_id] : far_depth;
      const float length = next_depth - previous_depth;
      if (length > 1.0e-8f && transmittance > transmittance_threshold && segment_count < WF2_MAX_REALRAY_SEGMENTS) {
        const float mid_depth = 0.5f * (previous_depth + next_depth);
        const float x = origin.x + direction.x * mid_depth;
        const float y = origin.y + direction.y * mid_depth;
        const float z = origin.z + direction.z * mid_depth;
        const uint owner = wf2_realray_owner_at(sites_f32, clamped_site_count, x, y, z, t);
        const uint rgba_base = owner * 4u;
        const float density = max(site_rgba_f32[rgba_base + 3u], 0.0f);
        const float seg_trans = exp(-density * length);
        const float seg_alpha = 1.0f - seg_trans;
        const float weight = transmittance * seg_alpha;
        const float3 rgb = float3(
            site_rgba_f32[rgba_base + 0u],
            site_rgba_f32[rgba_base + 1u],
            site_rgba_f32[rgba_base + 2u]);

        owners[segment_count] = owner;
        lengths[segment_count] = length;
        mids[segment_count] = mid_depth;
        trans_before[segment_count] = transmittance;
        segment_trans[segment_count] = seg_trans;
        segment_alpha[segment_count] = seg_alpha;
        weights[segment_count] = weight;
        segment_rgb[segment_count] = rgb;
        segment_count += 1u;

        alpha_accum += weight;
        depth_weighted += weight * mid_depth;
        transmittance *= seg_trans;
      }
      previous_depth = next_depth;
    }

    const uint sample_id = track_id * frame_count + frame_id;
    const uint out_base = sample_id * 3u;
    const float3 grad_rgb = float3(
        grad_rgb_f32[out_base + 0u],
        grad_rgb_f32[out_base + 1u],
        grad_rgb_f32[out_base + 2u]);
    const float grad_alpha = grad_alpha_f32[sample_id];
    const float grad_depth = grad_depth_f32[sample_id];
    float adj_next_transmittance = 0.0f;
    for (int segment_id = int(segment_count) - 1; segment_id >= 0; --segment_id) {
      const uint owner = owners[segment_id];
      float d_loss_d_weight = dot(grad_rgb, segment_rgb[segment_id]) + grad_alpha;
      if (alpha_accum > 1.0e-8f) {
        d_loss_d_weight += grad_depth *
            (mids[segment_id] * alpha_accum - depth_weighted) /
            (alpha_accum * alpha_accum);
      }

      const float adj_trans_before =
          d_loss_d_weight * segment_alpha[segment_id] +
          adj_next_transmittance * segment_trans[segment_id];
      const float adj_segment_alpha = d_loss_d_weight * trans_before[segment_id];
      const float adj_segment_trans =
          adj_next_transmittance * trans_before[segment_id] - adj_segment_alpha;
      grad_accum[owner].x += weights[segment_id] * grad_rgb.x;
      grad_accum[owner].y += weights[segment_id] * grad_rgb.y;
      grad_accum[owner].z += weights[segment_id] * grad_rgb.z;
      const uint rgba_base = owner * 4u;
      const float raw_density = site_rgba_f32[rgba_base + 3u];
      if (raw_density > 0.0f) {
        grad_accum[owner].w += adj_segment_trans * (-lengths[segment_id] * segment_trans[segment_id]);
      }
      adj_next_transmittance = adj_trans_before;
    }
  }

  for (uint site_id = 0u; site_id < clamped_site_count; ++site_id) {
    wf2_atomic_add4(grad_site_rgba_f32, site_id * 4u, grad_accum[site_id]);
  }
}

kernel void wf2_fused_slab_affine_num32_den16_vjp_finalize_reduce_tensor(
    device const float* partial_grad_site_rgba_f32 [[buffer(0)]],
    device const int* config_i32 [[buffer(1)]],
    device float* grad_site_rgba_f32 [[buffer(2)]],
    uint site_id [[thread_position_in_grid]]) {
  const uint site_count = uint(config_i32[2]);
  if (site_id >= site_count) {
    return;
  }
  const uint track_count = uint(config_i32[1]);
  const uint frame_count = uint(config_i32[3]);
  const uint reduce_chunk_size = max(uint(config_i32[7]), 1u);
  const uint total_samples = track_count * frame_count;
  const uint chunk_count = (total_samples + reduce_chunk_size - 1u) / reduce_chunk_size;
  float4 grad_accum = float4(0.0f, 0.0f, 0.0f, 0.0f);
  for (uint chunk_id = 0u; chunk_id < chunk_count; ++chunk_id) {
    const uint site_grad_base = (chunk_id * site_count + site_id) * 4u;
    grad_accum.x += partial_grad_site_rgba_f32[site_grad_base + 0u];
    grad_accum.y += partial_grad_site_rgba_f32[site_grad_base + 1u];
    grad_accum.z += partial_grad_site_rgba_f32[site_grad_base + 2u];
    grad_accum.w += partial_grad_site_rgba_f32[site_grad_base + 3u];
  }
  const uint out_base = site_id * 4u;
  grad_site_rgba_f32[out_base + 0u] = grad_accum.x;
  grad_site_rgba_f32[out_base + 1u] = grad_accum.y;
  grad_site_rgba_f32[out_base + 2u] = grad_accum.z;
  grad_site_rgba_f32[out_base + 3u] = grad_accum.w;
}

kernel void wf2_shared_realray_rgba_depth_vjp_finalize_reduce_tensor(
    device const float* partial_grad_site_rgba_f32 [[buffer(0)]],
    device const int* config_i32 [[buffer(1)]],
    device float* grad_site_rgba_f32 [[buffer(2)]],
    uint site_id [[thread_position_in_grid]]) {
  const uint site_count = uint(config_i32[2]);
  if (site_id >= site_count) {
    return;
  }
  const uint track_count = uint(config_i32[1]);
  const uint frame_count = uint(config_i32[3]);
  constexpr uint reduce_chunk_size = 4u;
  const uint total_samples = track_count * frame_count;
  const uint chunk_count = (total_samples + reduce_chunk_size - 1u) / reduce_chunk_size;
  float4 grad_accum = float4(0.0f, 0.0f, 0.0f, 0.0f);
  for (uint chunk_id = 0u; chunk_id < chunk_count; ++chunk_id) {
    const uint site_grad_base = (chunk_id * site_count + site_id) * 4u;
    grad_accum.x += partial_grad_site_rgba_f32[site_grad_base + 0u];
    grad_accum.y += partial_grad_site_rgba_f32[site_grad_base + 1u];
    grad_accum.z += partial_grad_site_rgba_f32[site_grad_base + 2u];
    grad_accum.w += partial_grad_site_rgba_f32[site_grad_base + 3u];
  }
  const uint out_base = site_id * 4u;
  grad_site_rgba_f32[out_base + 0u] = grad_accum.x;
  grad_site_rgba_f32[out_base + 1u] = grad_accum.y;
  grad_site_rgba_f32[out_base + 2u] = grad_accum.z;
  grad_site_rgba_f32[out_base + 3u] = grad_accum.w;
}

kernel void wf2_shared_signal_replay_tensor(
    device const float* boundary_f32 [[buffer(0)]],
    device const uint* candidate_mask_u32 [[buffer(1)]],
    device const float* sites_f32 [[buffer(2)]],
    device const float* site_signal_f32 [[buffer(3)]],
    device const float* beam_f32 [[buffer(4)]],
    device const float* frame_t_f32 [[buffer(5)]],
    device const float* grad_output_f32 [[buffer(6)]],
    device const int* config_i32 [[buffer(7)]],
    device const float* config_f32 [[buffer(8)]],
    device float* output_f32 [[buffer(9)]],
    device float* grad_sample_f32 [[buffer(10)]],
    uint ray_id [[thread_position_in_grid]]) {
  const uint boundary_count = uint(config_i32[0]);
  const uint beam_count = uint(config_i32[1]);
  const uint site_count = uint(config_i32[2]);
  const uint frame_count = uint(config_i32[3]);
  const uint time_slab_count = uint(config_i32[4]);
  const float camera_velocity_x = config_f32[0];
  const uint total_rays = beam_count * frame_count;
  if (ray_id >= total_rays) {
    return;
  }

  const uint beam_id = ray_id / frame_count;
  const uint frame_id = ray_id - beam_id * frame_count;
  const float u_center = beam_f32[beam_id * 5u + 0u];
  const float near_depth = beam_f32[beam_id * 5u + 3u];
  const float far_depth = beam_f32[beam_id * 5u + 4u];
  const float t = frame_t_f32[frame_id];
  const float x = u_center + camera_velocity_x * t;
  const float grad_output = grad_output_f32[ray_id];
  const uint slab_id = wf2_replay_slab_id(t, time_slab_count);
  const uint mask = candidate_mask_u32[beam_id * time_slab_count + slab_id];

  float depths[WF2_MAX_SHARED_BOUNDARIES];
  uint depth_count = 0u;
  const uint clamped_boundary_count = min(boundary_count, WF2_MAX_SHARED_BOUNDARIES);
  for (uint boundary_id = 0u; boundary_id < clamped_boundary_count; ++boundary_id) {
    if ((mask & (1u << boundary_id)) == 0u) {
      continue;
    }
    const float depth = wf2_replay_power_depth(
        boundary_f32,
        boundary_id,
        u_center,
        t,
        camera_velocity_x);
    if (isfinite(depth) && depth >= near_depth && depth <= far_depth) {
      wf2_replay_insert_depth(depths, depth_count, depth);
    }
  }

  const uint clamped_site_count = min(site_count, WF2_MAX_SHARED_SITES);
  const uint grad_base = ray_id * site_count;
  for (uint site_id = 0u; site_id < clamped_site_count; ++site_id) {
    grad_sample_f32[grad_base + site_id] = 0.0f;
  }

  float previous_depth = near_depth;
  float output = 0.0f;
  for (uint cut_id = 0u; cut_id <= depth_count; ++cut_id) {
    const float next_depth = cut_id < depth_count ? depths[cut_id] : far_depth;
    const float length = next_depth - previous_depth;
    if (length > 1.0e-8f) {
      const float mid_depth = 0.5f * (previous_depth + next_depth);
      const uint owner = wf2_replay_owner_at(sites_f32, clamped_site_count, x, mid_depth, t);
      output += site_signal_f32[owner] * length;
      grad_sample_f32[grad_base + owner] += grad_output * length;
    }
    previous_depth = next_depth;
  }
  output_f32[ray_id] = output;
}

kernel void wf2_shared_rgb_replay_tensor(
    device const float* boundary_f32 [[buffer(0)]],
    device const uint* candidate_mask_u32 [[buffer(1)]],
    device const float* sites_f32 [[buffer(2)]],
    device const float* site_rgb_f32 [[buffer(3)]],
    device const float* beam_f32 [[buffer(4)]],
    device const float* frame_t_f32 [[buffer(5)]],
    device const float* grad_output_rgb_f32 [[buffer(6)]],
    device const int* config_i32 [[buffer(7)]],
    device const float* config_f32 [[buffer(8)]],
    device float* output_rgb_f32 [[buffer(9)]],
    device float* grad_sample_rgb_f32 [[buffer(10)]],
    uint ray_id [[thread_position_in_grid]]) {
  const uint boundary_count = uint(config_i32[0]);
  const uint beam_count = uint(config_i32[1]);
  const uint site_count = uint(config_i32[2]);
  const uint frame_count = uint(config_i32[3]);
  const uint time_slab_count = uint(config_i32[4]);
  const float camera_velocity_x = config_f32[0];
  const uint total_rays = beam_count * frame_count;
  if (ray_id >= total_rays) {
    return;
  }

  const uint beam_id = ray_id / frame_count;
  const uint frame_id = ray_id - beam_id * frame_count;
  const float u_center = beam_f32[beam_id * 5u + 0u];
  const float near_depth = beam_f32[beam_id * 5u + 3u];
  const float far_depth = beam_f32[beam_id * 5u + 4u];
  const float t = frame_t_f32[frame_id];
  const float x = u_center + camera_velocity_x * t;
  const uint slab_id = wf2_replay_slab_id(t, time_slab_count);
  const uint mask = candidate_mask_u32[beam_id * time_slab_count + slab_id];

  float depths[WF2_MAX_SHARED_BOUNDARIES];
  uint depth_count = 0u;
  const uint clamped_boundary_count = min(boundary_count, WF2_MAX_SHARED_BOUNDARIES);
  for (uint boundary_id = 0u; boundary_id < clamped_boundary_count; ++boundary_id) {
    if ((mask & (1u << boundary_id)) == 0u) {
      continue;
    }
    const float depth = wf2_replay_power_depth(
        boundary_f32,
        boundary_id,
        u_center,
        t,
        camera_velocity_x);
    if (isfinite(depth) && depth >= near_depth && depth <= far_depth) {
      wf2_replay_insert_depth(depths, depth_count, depth);
    }
  }

  const uint clamped_site_count = min(site_count, WF2_MAX_SHARED_SITES);
  const uint grad_base = ray_id * site_count * 3u;
  for (uint site_id = 0u; site_id < clamped_site_count; ++site_id) {
    const uint site_base = grad_base + site_id * 3u;
    grad_sample_rgb_f32[site_base + 0u] = 0.0f;
    grad_sample_rgb_f32[site_base + 1u] = 0.0f;
    grad_sample_rgb_f32[site_base + 2u] = 0.0f;
  }

  float3 output = float3(0.0f, 0.0f, 0.0f);
  float previous_depth = near_depth;
  for (uint cut_id = 0u; cut_id <= depth_count; ++cut_id) {
    const float next_depth = cut_id < depth_count ? depths[cut_id] : far_depth;
    const float length = next_depth - previous_depth;
    if (length > 1.0e-8f) {
      const float mid_depth = 0.5f * (previous_depth + next_depth);
      const uint owner = wf2_replay_owner_at(sites_f32, clamped_site_count, x, mid_depth, t);
      const uint rgb_base = owner * 3u;
      const float3 rgb = float3(
          site_rgb_f32[rgb_base + 0u],
          site_rgb_f32[rgb_base + 1u],
          site_rgb_f32[rgb_base + 2u]);
      output += rgb * length;
      const uint sample_base = grad_base + owner * 3u;
      const uint ray_rgb_base = ray_id * 3u;
      grad_sample_rgb_f32[sample_base + 0u] += grad_output_rgb_f32[ray_rgb_base + 0u] * length;
      grad_sample_rgb_f32[sample_base + 1u] += grad_output_rgb_f32[ray_rgb_base + 1u] * length;
      grad_sample_rgb_f32[sample_base + 2u] += grad_output_rgb_f32[ray_rgb_base + 2u] * length;
    }
    previous_depth = next_depth;
  }

  const uint out_base = ray_id * 3u;
  output_rgb_f32[out_base + 0u] = output.x;
  output_rgb_f32[out_base + 1u] = output.y;
  output_rgb_f32[out_base + 2u] = output.z;
}

kernel void wf2_shared_rgba_depth_replay_tensor(
    device const float* boundary_f32 [[buffer(0)]],
    device const uint* candidate_mask_u32 [[buffer(1)]],
    device const float* sites_f32 [[buffer(2)]],
    device const float* site_rgba_f32 [[buffer(3)]],
    device const float* beam_f32 [[buffer(4)]],
    device const float* frame_t_f32 [[buffer(5)]],
    device const int* config_i32 [[buffer(6)]],
    device const float* config_f32 [[buffer(7)]],
    device float* output_rgb_f32 [[buffer(8)]],
    device float* output_alpha_f32 [[buffer(9)]],
    device float* output_depth_f32 [[buffer(10)]],
    uint ray_id [[thread_position_in_grid]]) {
  const uint boundary_count = uint(config_i32[0]);
  const uint beam_count = uint(config_i32[1]);
  const uint site_count = uint(config_i32[2]);
  const uint frame_count = uint(config_i32[3]);
  const uint time_slab_count = uint(config_i32[4]);
  const float camera_velocity_x = config_f32[0];
  const uint total_rays = beam_count * frame_count;
  if (ray_id >= total_rays) {
    return;
  }

  const uint beam_id = ray_id / frame_count;
  const uint frame_id = ray_id - beam_id * frame_count;
  const float u_center = beam_f32[beam_id * 5u + 0u];
  const float near_depth = beam_f32[beam_id * 5u + 3u];
  const float far_depth = beam_f32[beam_id * 5u + 4u];
  const float t = frame_t_f32[frame_id];
  const float x = u_center + camera_velocity_x * t;
  const uint slab_id = wf2_replay_slab_id(t, time_slab_count);
  const uint mask = candidate_mask_u32[beam_id * time_slab_count + slab_id];

  float depths[WF2_MAX_SHARED_BOUNDARIES];
  uint depth_count = 0u;
  const uint clamped_boundary_count = min(boundary_count, WF2_MAX_SHARED_BOUNDARIES);
  for (uint boundary_id = 0u; boundary_id < clamped_boundary_count; ++boundary_id) {
    if ((mask & (1u << boundary_id)) == 0u) {
      continue;
    }
    const float depth = wf2_replay_power_depth(
        boundary_f32,
        boundary_id,
        u_center,
        t,
        camera_velocity_x);
    if (isfinite(depth) && depth >= near_depth && depth <= far_depth) {
      wf2_replay_insert_depth(depths, depth_count, depth);
    }
  }

  const uint clamped_site_count = min(site_count, WF2_MAX_SHARED_SITES);
  float3 rgb_accum = float3(0.0f, 0.0f, 0.0f);
  float alpha_accum = 0.0f;
  float depth_weighted = 0.0f;
  float transmittance = 1.0f;
  float previous_depth = near_depth;
  for (uint cut_id = 0u; cut_id <= depth_count; ++cut_id) {
    const float next_depth = cut_id < depth_count ? depths[cut_id] : far_depth;
    const float length = next_depth - previous_depth;
    if (length > 1.0e-8f && transmittance > 1.0e-5f) {
      const float mid_depth = 0.5f * (previous_depth + next_depth);
      const uint owner = wf2_replay_owner_at(sites_f32, clamped_site_count, x, mid_depth, t);
      const uint rgba_base = owner * 4u;
      const float density = max(site_rgba_f32[rgba_base + 3u], 0.0f);
      const float segment_transmittance = exp(-density * length);
      const float segment_alpha = 1.0f - segment_transmittance;
      const float weight = transmittance * segment_alpha;
      const float3 rgb = float3(
          site_rgba_f32[rgba_base + 0u],
          site_rgba_f32[rgba_base + 1u],
          site_rgba_f32[rgba_base + 2u]);
      rgb_accum += weight * rgb;
      alpha_accum += weight;
      depth_weighted += weight * mid_depth;
      transmittance *= segment_transmittance;
    }
    previous_depth = next_depth;
  }

  const uint out_base = ray_id * 3u;
  output_rgb_f32[out_base + 0u] = rgb_accum.x;
  output_rgb_f32[out_base + 1u] = rgb_accum.y;
  output_rgb_f32[out_base + 2u] = rgb_accum.z;
  output_alpha_f32[ray_id] = alpha_accum;
  output_depth_f32[ray_id] = alpha_accum > 1.0e-8f ? depth_weighted / alpha_accum : far_depth;
}

kernel void wf2_shared_rgba_depth_vjp_tensor(
    device const float* boundary_f32 [[buffer(0)]],
    device const uint* candidate_mask_u32 [[buffer(1)]],
    device const float* sites_f32 [[buffer(2)]],
    device const float* site_rgba_f32 [[buffer(3)]],
    device const float* beam_f32 [[buffer(4)]],
    device const float* frame_t_f32 [[buffer(5)]],
    device const float* grad_rgb_f32 [[buffer(6)]],
    device const float* grad_alpha_f32 [[buffer(7)]],
    device const float* grad_depth_f32 [[buffer(8)]],
    device const int* config_i32 [[buffer(9)]],
    device const float* config_f32 [[buffer(10)]],
    device float* output_rgb_f32 [[buffer(11)]],
    device float* output_alpha_f32 [[buffer(12)]],
    device float* output_depth_f32 [[buffer(13)]],
    device float* grad_sample_rgba_f32 [[buffer(14)]],
    uint ray_id [[thread_position_in_grid]]) {
  const uint boundary_count = uint(config_i32[0]);
  const uint beam_count = uint(config_i32[1]);
  const uint site_count = uint(config_i32[2]);
  const uint frame_count = uint(config_i32[3]);
  const uint time_slab_count = uint(config_i32[4]);
  const float camera_velocity_x = config_f32[0];
  const uint total_rays = beam_count * frame_count;
  if (ray_id >= total_rays) {
    return;
  }

  const uint beam_id = ray_id / frame_count;
  const uint frame_id = ray_id - beam_id * frame_count;
  const float u_center = beam_f32[beam_id * 5u + 0u];
  const float near_depth = beam_f32[beam_id * 5u + 3u];
  const float far_depth = beam_f32[beam_id * 5u + 4u];
  const float t = frame_t_f32[frame_id];
  const float x = u_center + camera_velocity_x * t;
  const uint slab_id = wf2_replay_slab_id(t, time_slab_count);
  const uint mask = candidate_mask_u32[beam_id * time_slab_count + slab_id];

  float depths[WF2_MAX_SHARED_BOUNDARIES];
  uint depth_count = 0u;
  const uint clamped_boundary_count = min(boundary_count, WF2_MAX_SHARED_BOUNDARIES);
  for (uint boundary_id = 0u; boundary_id < clamped_boundary_count; ++boundary_id) {
    if ((mask & (1u << boundary_id)) == 0u) {
      continue;
    }
    const float depth = wf2_replay_power_depth(
        boundary_f32,
        boundary_id,
        u_center,
        t,
        camera_velocity_x);
    if (isfinite(depth) && depth >= near_depth && depth <= far_depth) {
      wf2_replay_insert_depth(depths, depth_count, depth);
    }
  }

  const uint clamped_site_count = min(site_count, WF2_MAX_SHARED_SITES);
  const uint grad_base = ray_id * site_count * 4u;
  for (uint site_id = 0u; site_id < clamped_site_count; ++site_id) {
    const uint site_base = grad_base + site_id * 4u;
    grad_sample_rgba_f32[site_base + 0u] = 0.0f;
    grad_sample_rgba_f32[site_base + 1u] = 0.0f;
    grad_sample_rgba_f32[site_base + 2u] = 0.0f;
    grad_sample_rgba_f32[site_base + 3u] = 0.0f;
  }

  uint owners[WF2_MAX_SHARED_SEGMENTS];
  float lengths[WF2_MAX_SHARED_SEGMENTS];
  float mids[WF2_MAX_SHARED_SEGMENTS];
  float trans_before[WF2_MAX_SHARED_SEGMENTS];
  float segment_trans[WF2_MAX_SHARED_SEGMENTS];
  float segment_alpha[WF2_MAX_SHARED_SEGMENTS];
  float weights[WF2_MAX_SHARED_SEGMENTS];
  float3 segment_rgb[WF2_MAX_SHARED_SEGMENTS];

  float3 rgb_accum = float3(0.0f, 0.0f, 0.0f);
  float alpha_accum = 0.0f;
  float depth_weighted = 0.0f;
  float transmittance = 1.0f;
  float previous_depth = near_depth;
  uint segment_count = 0u;
  for (uint cut_id = 0u; cut_id <= depth_count; ++cut_id) {
    const float next_depth = cut_id < depth_count ? depths[cut_id] : far_depth;
    const float length = next_depth - previous_depth;
    if (length > 1.0e-8f && transmittance > 1.0e-5f && segment_count < WF2_MAX_SHARED_SEGMENTS) {
      const float mid_depth = 0.5f * (previous_depth + next_depth);
      const uint owner = wf2_replay_owner_at(sites_f32, clamped_site_count, x, mid_depth, t);
      const uint rgba_base = owner * 4u;
      const float density = max(site_rgba_f32[rgba_base + 3u], 0.0f);
      const float seg_trans = exp(-density * length);
      const float seg_alpha = 1.0f - seg_trans;
      const float weight = transmittance * seg_alpha;
      const float3 rgb = float3(
          site_rgba_f32[rgba_base + 0u],
          site_rgba_f32[rgba_base + 1u],
          site_rgba_f32[rgba_base + 2u]);

      owners[segment_count] = owner;
      lengths[segment_count] = length;
      mids[segment_count] = mid_depth;
      trans_before[segment_count] = transmittance;
      segment_trans[segment_count] = seg_trans;
      segment_alpha[segment_count] = seg_alpha;
      weights[segment_count] = weight;
      segment_rgb[segment_count] = rgb;
      segment_count += 1u;

      rgb_accum += weight * rgb;
      alpha_accum += weight;
      depth_weighted += weight * mid_depth;
      transmittance *= seg_trans;
    }
    previous_depth = next_depth;
  }

  const uint out_base = ray_id * 3u;
  output_rgb_f32[out_base + 0u] = rgb_accum.x;
  output_rgb_f32[out_base + 1u] = rgb_accum.y;
  output_rgb_f32[out_base + 2u] = rgb_accum.z;
  output_alpha_f32[ray_id] = alpha_accum;
  output_depth_f32[ray_id] = alpha_accum > 1.0e-8f ? depth_weighted / alpha_accum : far_depth;

  const float3 grad_rgb = float3(
      grad_rgb_f32[out_base + 0u],
      grad_rgb_f32[out_base + 1u],
      grad_rgb_f32[out_base + 2u]);
  const float grad_alpha = grad_alpha_f32[ray_id];
  const float grad_depth = grad_depth_f32[ray_id];
  float adj_next_transmittance = 0.0f;
  for (int segment_id = int(segment_count) - 1; segment_id >= 0; --segment_id) {
    const uint owner = owners[segment_id];
    float d_loss_d_weight = dot(grad_rgb, segment_rgb[segment_id]) + grad_alpha;
    if (alpha_accum > 1.0e-8f) {
      d_loss_d_weight += grad_depth *
          (mids[segment_id] * alpha_accum - depth_weighted) /
          (alpha_accum * alpha_accum);
    }

    const uint site_grad_base = grad_base + owner * 4u;
    grad_sample_rgba_f32[site_grad_base + 0u] += weights[segment_id] * grad_rgb.x;
    grad_sample_rgba_f32[site_grad_base + 1u] += weights[segment_id] * grad_rgb.y;
    grad_sample_rgba_f32[site_grad_base + 2u] += weights[segment_id] * grad_rgb.z;

    const float adj_trans_before =
        d_loss_d_weight * segment_alpha[segment_id] +
        adj_next_transmittance * segment_trans[segment_id];
    const float adj_segment_alpha = d_loss_d_weight * trans_before[segment_id];
    const float adj_segment_trans =
        adj_next_transmittance * trans_before[segment_id] - adj_segment_alpha;
    const uint rgba_base = owner * 4u;
    const float raw_density = site_rgba_f32[rgba_base + 3u];
    const float grad_density = raw_density > 0.0f
        ? adj_segment_trans * (-lengths[segment_id] * segment_trans[segment_id])
        : 0.0f;
    grad_sample_rgba_f32[site_grad_base + 3u] += grad_density;
    adj_next_transmittance = adj_trans_before;
  }
}

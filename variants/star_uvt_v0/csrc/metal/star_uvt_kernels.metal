#include <metal_stdlib>
using namespace metal;

#ifndef STAR_TILE_X
#define STAR_TILE_X 8u
#endif
#ifndef STAR_TILE_Y
#define STAR_TILE_Y 8u
#endif
#ifndef STAR_TILE_T
#define STAR_TILE_T 2u
#endif
#ifndef STAR_TILE_CAPACITY
#define STAR_TILE_CAPACITY 128u
#endif
#ifndef STAR_THREADS
#define STAR_THREADS 128u
#endif

struct MetaI32 {
  int height;
  int width;
  int frames;
  int tile_x;
  int tile_y;
  int tile_t;
  int tiles_x;
  int tiles_y;
  int tiles_t;
  int tile_count;
  int tube_count;
  int tile_capacity;
  int reserved0;
  int reserved1;
};

struct MetaF32 {
  float alpha_threshold;
  float transmittance_threshold;
  float bg_r;
  float bg_g;
  float bg_b;
  float eps;
  float max_alpha;
};

struct Bounds3i {
  int x0;
  int x1;
  int y0;
  int y1;
  int f0;
  int f1;
};

inline uint next_pow2_u32(uint x) {
  x = max(x, 1u);
  x -= 1u;
  x |= x >> 1u;
  x |= x >> 2u;
  x |= x >> 4u;
  x |= x >> 8u;
  x |= x >> 16u;
  return x + 1u;
}

inline float frame_time(uint frame, constant MetaI32& mi) {
  return float(frame) - 0.5f * float(mi.frames - 1);
}

inline float3 load3(const device float* ptr, uint i) {
  uint b = i * 3u;
  return float3(ptr[b + 0u], ptr[b + 1u], ptr[b + 2u]);
}

inline float3 load_q_row0(const device float* q, uint i) {
  uint b = i * 6u;
  return float3(q[b + 0u], q[b + 1u], q[b + 2u]);
}

inline float3 load_q_row1(const device float* q, uint i) {
  uint b = i * 6u;
  return float3(q[b + 1u], q[b + 3u], q[b + 4u]);
}

inline float3 load_q_row2(const device float* q, uint i) {
  uint b = i * 6u;
  return float3(q[b + 2u], q[b + 4u], q[b + 5u]);
}

inline float quadratic_q(const device float* q, uint i, float3 d) {
  uint b = i * 6u;
  float q00 = q[b + 0u];
  float q01 = q[b + 1u];
  float q02 = q[b + 2u];
  float q11 = q[b + 3u];
  float q12 = q[b + 4u];
  float q22 = q[b + 5u];
  return q00 * d.x * d.x + 2.0f * q01 * d.x * d.y + 2.0f * q02 * d.x * d.z +
         q11 * d.y * d.y + 2.0f * q12 * d.y * d.z + q22 * d.z * d.z;
}

inline float eval_depth(const device float* ma, const device float* depth0, const device float* depth_beta, uint tube_id, float3 a) {
  float3 m = load3(ma, tube_id);
  float3 beta = load3(depth_beta, tube_id);
  return depth0[tube_id] + dot(beta, a - m);
}

inline bool inverse_sym3_diag(const device float* q, uint i, float eps, thread float3& diag_out) {
  uint b = i * 6u;
  float a = q[b + 0u];
  float b01 = q[b + 1u];
  float c = q[b + 2u];
  float d = q[b + 3u];
  float e = q[b + 4u];
  float f = q[b + 5u];

  float co00 = d * f - e * e;
  float co11 = a * f - c * c;
  float co22 = a * d - b01 * b01;
  float det = a * co00 - b01 * (b01 * f - c * e) + c * (b01 * e - c * d);
  if (!isfinite(det) || fabs(det) <= eps) {
    return false;
  }
  float inv_det = 1.0f / det;
  diag_out = abs(float3(co00, co11, co22) * inv_det);
  return all(isfinite(diag_out));
}

inline Bounds3i tube_bounds(
    const device float* ma,
    const device float* q,
    const device float* opacity,
    uint tube_id,
    constant MetaI32& mi,
    constant MetaF32& mf) {
  Bounds3i out;
  out.x0 = 1; out.x1 = 0; out.y0 = 1; out.y1 = 0; out.f0 = 1; out.f1 = 0;

  float op = opacity[tube_id];
  if (!(op > mf.alpha_threshold)) return out;
  float tau = -2.0f * log(max(mf.alpha_threshold / max(op, mf.eps), mf.eps));
  if (!isfinite(tau) || tau <= 0.0f) return out;

  float3 inv_diag;
  bool ok = inverse_sym3_diag(q, tube_id, mf.eps, inv_diag);
  float3 m = load3(ma, tube_id);
  float3 half_extent;
  if (ok) {
    half_extent = sqrt(max(tau * inv_diag, float3(0.0f)));
  } else {
    half_extent = float3(float(mi.width), float(mi.height), float(mi.frames));
  }

  out.x0 = max(0, int(floor(m.x - half_extent.x - 0.5f)));
  out.x1 = min(mi.width - 1, int(ceil(m.x + half_extent.x - 0.5f)));
  out.y0 = max(0, int(floor(m.y - half_extent.y - 0.5f)));
  out.y1 = min(mi.height - 1, int(ceil(m.y + half_extent.y - 0.5f)));

  float center = 0.5f * float(mi.frames - 1);
  out.f0 = max(0, int(floor(m.z - half_extent.z + center)));
  out.f1 = min(mi.frames - 1, int(ceil(m.z + half_extent.z + center)));
  return out;
}

inline uint encode_tile(uint tx, uint ty, uint tz, constant MetaI32& mi) {
  return (tz * uint(mi.tiles_y) + ty) * uint(mi.tiles_x) + tx;
}

inline void decode_tile(uint tile_id, constant MetaI32& mi, thread uint& tx, thread uint& ty, thread uint& tz) {
  tx = tile_id % uint(mi.tiles_x);
  uint rem = tile_id / uint(mi.tiles_x);
  ty = rem % uint(mi.tiles_y);
  tz = rem / uint(mi.tiles_y);
}

inline float3 tile_center(uint tx, uint ty, uint tz, constant MetaI32& mi) {
  uint x0 = tx * uint(mi.tile_x);
  uint x1 = min(uint(mi.width - 1), x0 + uint(mi.tile_x - 1));
  uint y0 = ty * uint(mi.tile_y);
  uint y1 = min(uint(mi.height - 1), y0 + uint(mi.tile_y - 1));
  uint f0 = tz * uint(mi.tile_t);
  uint f1 = min(uint(mi.frames - 1), f0 + uint(mi.tile_t - 1));
  return float3(
      0.5f * (float(x0) + float(x1)) + 0.5f,
      0.5f * (float(y0) + float(y1)) + 0.5f,
      0.5f * (frame_time(f0, mi) + frame_time(f1, mi)));
}

inline float3 tile_half_extent(uint tx, uint ty, uint tz, constant MetaI32& mi) {
  uint x0 = tx * uint(mi.tile_x);
  uint x1 = min(uint(mi.width - 1), x0 + uint(mi.tile_x - 1));
  uint y0 = ty * uint(mi.tile_y);
  uint y1 = min(uint(mi.height - 1), y0 + uint(mi.tile_y - 1));
  uint f0 = tz * uint(mi.tile_t);
  uint f1 = min(uint(mi.frames - 1), f0 + uint(mi.tile_t - 1));
  return float3(
      max(0.5f, 0.5f * (float(x1) - float(x0) + 1.0f)),
      max(0.5f, 0.5f * (float(y1) - float(y0) + 1.0f)),
      max(0.0f, 0.5f * (frame_time(f1, mi) - frame_time(f0, mi))));
}

inline void sort_by_depth(threadgroup uint* ids, threadgroup float* depths, uint count, uint tid) {
  uint sort_n = next_pow2_u32(count);
  for (uint i = tid; i < sort_n; i += STAR_THREADS) {
    if (i >= count) {
      ids[i] = 0xFFFFFFFFu;
      depths[i] = INFINITY;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint k = 2u; k <= sort_n; k <<= 1u) {
    for (uint j = k >> 1u; j > 0u; j >>= 1u) {
      uint n_pairs = sort_n >> 1u;
      for (uint pair = tid; pair < n_pairs; pair += STAR_THREADS) {
        uint pos = 2u * j * (pair / j) + (pair % j);
        uint ixj = pos + j;
        bool ascending = ((pos & k) == 0u);
        float da = depths[pos];
        float db = depths[ixj];
        uint ia = ids[pos];
        uint ib = ids[ixj];
        bool greater = (da > db) || (da == db && ia > ib);
        if (greater == ascending) {
          depths[pos] = db;
          depths[ixj] = da;
          ids[pos] = ib;
          ids[ixj] = ia;
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  }
}

inline bool tile_order_unstable(
    threadgroup uint* ids,
    uint count,
    const device float* ma,
    const device float* depth0,
    const device float* depth_beta,
    uint tx,
    uint ty,
    uint tz,
    constant MetaI32& mi) {
  if (count < 2u) return false;
  float3 ac = tile_center(tx, ty, tz, mi);
  float3 h = tile_half_extent(tx, ty, tz, mi);
  for (uint i = 0u; i + 1u < count; ++i) {
    uint a_id = ids[i];
    uint b_id = ids[i + 1u];
    float3 ba = load3(depth_beta, a_id);
    float3 bb = load3(depth_beta, b_id);
    float3 g = ba - bb;
    float c = eval_depth(ma, depth0, depth_beta, a_id, ac) - eval_depth(ma, depth0, depth_beta, b_id, ac);
    float r = abs(g.x) * h.x + abs(g.y) * h.y + abs(g.z) * h.z;
    if (c - r <= 0.0f && c + r >= 0.0f) return true;
  }
  return false;
}

inline void composite_tube(
    uint tube_id,
    float3 sample_a,
    const device float* ma,
    const device float* q,
    const device float* opacity,
    const device float* color,
    constant MetaF32& mf,
    thread float3& accum,
    thread float& transmittance) {
  float3 d = sample_a - load3(ma, tube_id);
  float qv = quadratic_q(q, tube_id, d);
  if (!isfinite(qv)) return;
  float alpha = min(mf.max_alpha, opacity[tube_id] * exp(-0.5f * qv));
  if (!(alpha >= mf.alpha_threshold)) return;
  float w = transmittance * alpha;
  accum += w * load3(color, tube_id);
  transmittance *= (1.0f - alpha);
}

inline uint select_sample_order_id(
    threadgroup uint* ids,
    uint count,
    const device float* ma,
    const device float* depth0,
    const device float* depth_beta,
    float3 sample_a,
    float last_depth,
    uint last_id,
    thread float& out_depth) {
  uint best_id = 0xFFFFFFFFu;
  float best_depth = INFINITY;
  for (uint i = 0u; i < count; ++i) {
    uint tube_id = ids[i];
    float d = eval_depth(ma, depth0, depth_beta, tube_id, sample_a);
    bool after_last = (d > last_depth) || (d == last_depth && tube_id > last_id);
    bool better = (d < best_depth) || (d == best_depth && tube_id < best_id);
    if (after_last && better) {
      best_depth = d;
      best_id = tube_id;
    }
  }
  out_depth = best_depth;
  return best_id;
}

kernel void clear_tiles(
    device atomic_uint* tile_counts [[buffer(0)]],
    device atomic_uint* tile_overflow [[buffer(1)]],
    device atomic_uint* tile_unstable [[buffer(2)]],
    constant MetaI32& mi [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
  if (tid >= uint(mi.tile_count)) return;
  atomic_store_explicit(tile_counts + tid, 0u, memory_order_relaxed);
  atomic_store_explicit(tile_overflow + tid, 0u, memory_order_relaxed);
  atomic_store_explicit(tile_unstable + tid, 0u, memory_order_relaxed);
}

kernel void bin_screen_tubes_to_uvt_tiles(
    const device float* ma [[buffer(0)]],
    const device float* q_uvt [[buffer(1)]],
    const device float* depth0 [[buffer(2)]],
    const device float* depth_beta [[buffer(3)]],
    const device float* opacity [[buffer(4)]],
    constant MetaI32& mi [[buffer(5)]],
    constant MetaF32& mf [[buffer(6)]],
    device atomic_uint* tile_counts [[buffer(7)]],
    device uint* tile_tube_ids [[buffer(8)]],
    device float* tile_depths [[buffer(9)]],
    device atomic_uint* tile_overflow [[buffer(10)]],
    uint tube_id [[thread_position_in_grid]]) {
  if (tube_id >= uint(mi.tube_count)) return;
  Bounds3i b = tube_bounds(ma, q_uvt, opacity, tube_id, mi, mf);
  if (b.x0 > b.x1 || b.y0 > b.y1 || b.f0 > b.f1) return;

  uint tx0 = uint(b.x0 / mi.tile_x);
  uint tx1 = uint(b.x1 / mi.tile_x);
  uint ty0 = uint(b.y0 / mi.tile_y);
  uint ty1 = uint(b.y1 / mi.tile_y);
  uint tz0 = uint(b.f0 / mi.tile_t);
  uint tz1 = uint(b.f1 / mi.tile_t);

  for (uint tz = tz0; tz <= tz1; ++tz) {
    for (uint ty = ty0; ty <= ty1; ++ty) {
      for (uint tx = tx0; tx <= tx1; ++tx) {
        uint tile_id = encode_tile(tx, ty, tz, mi);
        uint slot = atomic_fetch_add_explicit(tile_counts + tile_id, 1u, memory_order_relaxed);
        if (slot < STAR_TILE_CAPACITY) {
          uint idx = tile_id * STAR_TILE_CAPACITY + slot;
          tile_tube_ids[idx] = tube_id;
          tile_depths[idx] = eval_depth(ma, depth0, depth_beta, tube_id, tile_center(tx, ty, tz, mi));
        } else {
          atomic_store_explicit(tile_overflow + tile_id, 1u, memory_order_relaxed);
        }
      }
    }
  }
}

kernel void render_uvt_tiles(
    const device float* ma [[buffer(0)]],
    const device float* q_uvt [[buffer(1)]],
    const device float* depth0 [[buffer(2)]],
    const device float* depth_beta [[buffer(3)]],
    const device float* opacity [[buffer(4)]],
    const device float* color [[buffer(5)]],
    constant MetaI32& mi [[buffer(6)]],
    constant MetaF32& mf [[buffer(7)]],
    const device atomic_uint* tile_counts [[buffer(8)]],
    const device uint* tile_tube_ids [[buffer(9)]],
    const device float* tile_depths [[buffer(10)]],
    device atomic_uint* tile_unstable [[buffer(11)]],
    device float* out_rgb [[buffer(12)]],
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]]) {
  uint tile_id = gid / STAR_THREADS;
  if (tile_id >= uint(mi.tile_count)) return;
  uint local_tid = tid;

  uint tx, ty, tz;
  decode_tile(tile_id, mi, tx, ty, tz);

  threadgroup uint local_ids[STAR_TILE_CAPACITY];
  threadgroup float local_depths[STAR_TILE_CAPACITY];
  uint raw_count = atomic_load_explicit(tile_counts + tile_id, memory_order_relaxed);
  uint count = min(raw_count, STAR_TILE_CAPACITY);
  for (uint i = local_tid; i < count; i += STAR_THREADS) {
    uint idx = tile_id * STAR_TILE_CAPACITY + i;
    local_ids[i] = tile_tube_ids[idx];
    local_depths[i] = tile_depths[idx];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  sort_by_depth(local_ids, local_depths, count, local_tid);
  threadgroup_barrier(mem_flags::mem_threadgroup);

  bool unstable = tile_order_unstable(local_ids, count, ma, depth0, depth_beta, tx, ty, tz, mi);
  if (local_tid == 0u && unstable) {
    atomic_store_explicit(tile_unstable + tile_id, 1u, memory_order_relaxed);
  }

  uint samples_per_frame = STAR_TILE_X * STAR_TILE_Y;
  uint lt = local_tid / samples_per_frame;
  uint rem = local_tid - lt * samples_per_frame;
  uint ly = rem / STAR_TILE_X;
  uint lx = rem - ly * STAR_TILE_X;
  uint x = tx * STAR_TILE_X + lx;
  uint y = ty * STAR_TILE_Y + ly;
  uint f = tz * STAR_TILE_T + lt;
  if (x >= uint(mi.width) || y >= uint(mi.height) || f >= uint(mi.frames)) return;

  float3 sample_a = float3(float(x) + 0.5f, float(y) + 0.5f, frame_time(f, mi));
  float3 accum = float3(0.0f);
  float T = 1.0f;

  if (!unstable) {
    for (uint i = 0u; i < count; ++i) {
      composite_tube(local_ids[i], sample_a, ma, q_uvt, opacity, color, mf, accum, T);
      if (T <= mf.transmittance_threshold) break;
    }
  } else {
    float last_depth = -INFINITY;
    uint last_id = 0u;
    for (uint rank = 0u; rank < count; ++rank) {
      float selected_depth;
      uint tube_id = select_sample_order_id(local_ids, count, ma, depth0, depth_beta, sample_a, last_depth, last_id, selected_depth);
      if (tube_id == 0xFFFFFFFFu) break;
      composite_tube(tube_id, sample_a, ma, q_uvt, opacity, color, mf, accum, T);
      last_depth = selected_depth;
      last_id = tube_id;
      if (T <= mf.transmittance_threshold) break;
    }
  }

  uint pix = (f * uint(mi.height) * uint(mi.width) + y * uint(mi.width) + x) * 3u;
  out_rgb[pix + 0u] = accum.x + T * mf.bg_r;
  out_rgb[pix + 1u] = accum.y + T * mf.bg_g;
  out_rgb[pix + 2u] = accum.z + T * mf.bg_b;
}

kernel void simple_backward_samples(
    const device float* ma [[buffer(0)]],
    const device float* q_uvt [[buffer(1)]],
    const device float* opacity [[buffer(2)]],
    const device float* color [[buffer(3)]],
    const device float* grad_image [[buffer(4)]],
    constant MetaI32& mi [[buffer(5)]],
    constant MetaF32& mf [[buffer(6)]],
    device float* grad_ma_samples [[buffer(7)]],
    device float* grad_q_samples [[buffer(8)]],
    device float* grad_opacity_samples [[buffer(9)]],
    device float* grad_color_samples [[buffer(10)]],
    uint idx [[thread_position_in_grid]]) {
  uint tube_count = uint(mi.tube_count);
  uint total = uint(mi.frames * mi.height * mi.width) * tube_count;
  if (idx >= total) return;

  uint tube_id = idx % tube_count;
  uint sample_id = idx / tube_count;
  uint x = sample_id % uint(mi.width);
  uint rem = sample_id / uint(mi.width);
  uint y = rem % uint(mi.height);
  uint frame = rem / uint(mi.height);
  float3 sample_a = float3(float(x) + 0.5f, float(y) + 0.5f, frame_time(frame, mi));
  float3 d = sample_a - load3(ma, tube_id);
  float qv = quadratic_q(q_uvt, tube_id, d);
  float exp_term = exp(-0.5f * qv);
  float alpha_unclamped = opacity[tube_id] * exp_term;
  bool active = isfinite(qv) && alpha_unclamped < mf.max_alpha;
  float alpha = active ? alpha_unclamped : mf.max_alpha;

  uint image_base = ((frame * uint(mi.height) + y) * uint(mi.width) + x) * 3u;
  float3 grad_rgb = float3(grad_image[image_base + 0u], grad_image[image_base + 1u], grad_image[image_base + 2u]);
  float3 c = load3(color, tube_id);
  float grad_alpha = active ? dot(grad_rgb, c) : 0.0f;
  float grad_qv = -0.5f * alpha * grad_alpha;
  float3 qd = load_q_row0(q_uvt, tube_id) * d.x + load_q_row1(q_uvt, tube_id) * d.y + load_q_row2(q_uvt, tube_id) * d.z;
  float3 grad_m = -2.0f * grad_qv * qd;

  uint ma_base = idx * 3u;
  grad_ma_samples[ma_base + 0u] = grad_m.x;
  grad_ma_samples[ma_base + 1u] = grad_m.y;
  grad_ma_samples[ma_base + 2u] = grad_m.z;

  uint q_base = idx * 6u;
  grad_q_samples[q_base + 0u] = grad_qv * d.x * d.x;
  grad_q_samples[q_base + 1u] = grad_qv * 2.0f * d.x * d.y;
  grad_q_samples[q_base + 2u] = grad_qv * 2.0f * d.x * d.z;
  grad_q_samples[q_base + 3u] = grad_qv * d.y * d.y;
  grad_q_samples[q_base + 4u] = grad_qv * 2.0f * d.y * d.z;
  grad_q_samples[q_base + 5u] = grad_qv * d.z * d.z;
  grad_opacity_samples[idx] = active ? grad_alpha * exp_term : 0.0f;

  uint color_base = idx * 3u;
  grad_color_samples[color_base + 0u] = grad_rgb.x * alpha;
  grad_color_samples[color_base + 1u] = grad_rgb.y * alpha;
  grad_color_samples[color_base + 2u] = grad_rgb.z * alpha;
}

kernel void stable_backward_samples(
    const device float* ma [[buffer(0)]],
    const device float* q_uvt [[buffer(1)]],
    const device float* depth0 [[buffer(2)]],
    const device float* depth_beta [[buffer(3)]],
    const device float* opacity [[buffer(4)]],
    const device float* color [[buffer(5)]],
    const device float* grad_image [[buffer(6)]],
    constant MetaI32& mi [[buffer(7)]],
    constant MetaF32& mf [[buffer(8)]],
    const device atomic_uint* tile_counts [[buffer(9)]],
    const device uint* tile_tube_ids [[buffer(10)]],
    const device float* tile_depths [[buffer(11)]],
    device atomic_uint* tile_unstable [[buffer(12)]],
    device atomic_uint* grad_count [[buffer(13)]],
    device int* grad_ids [[buffer(14)]],
    device float* grad_ma_samples [[buffer(15)]],
    device float* grad_q_samples [[buffer(16)]],
    device float* grad_opacity_samples [[buffer(17)]],
    device float* grad_color_samples [[buffer(18)]],
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]]) {
  uint tile_id = gid / STAR_THREADS;
  if (tile_id >= uint(mi.tile_count)) return;
  uint local_tid = tid;

  uint tx, ty, tz;
  decode_tile(tile_id, mi, tx, ty, tz);

  threadgroup uint local_ids[STAR_TILE_CAPACITY];
  threadgroup float local_depths[STAR_TILE_CAPACITY];
  uint raw_count = atomic_load_explicit(tile_counts + tile_id, memory_order_relaxed);
  uint count = min(raw_count, STAR_TILE_CAPACITY);
  for (uint i = local_tid; i < count; i += STAR_THREADS) {
    uint idx = tile_id * STAR_TILE_CAPACITY + i;
    local_ids[i] = tile_tube_ids[idx];
    local_depths[i] = tile_depths[idx];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  sort_by_depth(local_ids, local_depths, count, local_tid);
  threadgroup_barrier(mem_flags::mem_threadgroup);

  bool unstable = tile_order_unstable(local_ids, count, ma, depth0, depth_beta, tx, ty, tz, mi);
  if (local_tid == 0u && unstable) {
    atomic_store_explicit(tile_unstable + tile_id, 1u, memory_order_relaxed);
  }

  uint samples_per_frame = STAR_TILE_X * STAR_TILE_Y;
  uint lt = local_tid / samples_per_frame;
  uint rem = local_tid - lt * samples_per_frame;
  uint ly = rem / STAR_TILE_X;
  uint lx = rem - ly * STAR_TILE_X;
  uint x = tx * STAR_TILE_X + lx;
  uint y = ty * STAR_TILE_Y + ly;
  uint f = tz * STAR_TILE_T + lt;
  if (x >= uint(mi.width) || y >= uint(mi.height) || f >= uint(mi.frames)) return;

  float3 sample_a = float3(float(x) + 0.5f, float(y) + 0.5f, frame_time(f, mi));
  uint ordered_ids[STAR_TILE_CAPACITY];
  uint ordered_count = count;
  if (!unstable) {
    for (uint i = 0u; i < count; ++i) {
      ordered_ids[i] = local_ids[i];
    }
  } else {
    float last_depth = -INFINITY;
    uint last_id = 0u;
    ordered_count = 0u;
    for (uint rank = 0u; rank < count; ++rank) {
      float selected_depth;
      uint tube_id = select_sample_order_id(local_ids, count, ma, depth0, depth_beta, sample_a, last_depth, last_id, selected_depth);
      if (tube_id == 0xFFFFFFFFu) break;
      ordered_ids[ordered_count] = tube_id;
      ordered_count += 1u;
      last_depth = selected_depth;
      last_id = tube_id;
    }
  }

  float t_before[STAR_TILE_CAPACITY];
  float alpha_values[STAR_TILE_CAPACITY];
  bool processed[STAR_TILE_CAPACITY];
  bool differentiable_alpha[STAR_TILE_CAPACITY];
  float T = 1.0f;
  for (uint i = 0u; i < STAR_TILE_CAPACITY; ++i) {
    t_before[i] = 0.0f;
    alpha_values[i] = 0.0f;
    processed[i] = false;
    differentiable_alpha[i] = false;
  }
  for (uint i = 0u; i < ordered_count; ++i) {
    uint tube_id = ordered_ids[i];
    float3 d = sample_a - load3(ma, tube_id);
    float qv = quadratic_q(q_uvt, tube_id, d);
    if (!isfinite(qv)) continue;
    float alpha_raw = opacity[tube_id] * exp(-0.5f * qv);
    float alpha = min(mf.max_alpha, alpha_raw);
    if (!(alpha >= mf.alpha_threshold)) continue;
    t_before[i] = T;
    alpha_values[i] = alpha;
    processed[i] = true;
    differentiable_alpha[i] = alpha_raw < mf.max_alpha;
    T *= (1.0f - alpha);
    if (T <= mf.transmittance_threshold) break;
  }

  uint image_base = ((f * uint(mi.height) + y) * uint(mi.width) + x) * 3u;
  float3 grad_rgb = float3(grad_image[image_base + 0u], grad_image[image_base + 1u], grad_image[image_base + 2u]);
  float dT_next = dot(grad_rgb, float3(mf.bg_r, mf.bg_g, mf.bg_b));
  for (int si = int(ordered_count) - 1; si >= 0; --si) {
    uint i = uint(si);
    if (!processed[i]) continue;
    uint tube_id = ordered_ids[i];
    float alpha = alpha_values[i];
    float t_i = t_before[i];
    float3 c = load3(color, tube_id);
    float d_alpha = dot(grad_rgb, t_i * c) - dT_next * t_i;
    float3 d_color = grad_rgb * (t_i * alpha);
    float dT_i = dot(grad_rgb, alpha * c) + dT_next * (1.0f - alpha);
    dT_next = dT_i;

    uint entry = atomic_fetch_add_explicit(grad_count, 1u, memory_order_relaxed);
    grad_ids[entry] = int(tube_id);
    uint color_base = entry * 3u;
    grad_color_samples[color_base + 0u] = d_color.x;
    grad_color_samples[color_base + 1u] = d_color.y;
    grad_color_samples[color_base + 2u] = d_color.z;
    uint ma_base = entry * 3u;
    grad_ma_samples[ma_base + 0u] = 0.0f;
    grad_ma_samples[ma_base + 1u] = 0.0f;
    grad_ma_samples[ma_base + 2u] = 0.0f;
    uint q_base = entry * 6u;
    grad_q_samples[q_base + 0u] = 0.0f;
    grad_q_samples[q_base + 1u] = 0.0f;
    grad_q_samples[q_base + 2u] = 0.0f;
    grad_q_samples[q_base + 3u] = 0.0f;
    grad_q_samples[q_base + 4u] = 0.0f;
    grad_q_samples[q_base + 5u] = 0.0f;
    grad_opacity_samples[entry] = 0.0f;
    if (!differentiable_alpha[i]) continue;

    float3 d = sample_a - load3(ma, tube_id);
    float qv = quadratic_q(q_uvt, tube_id, d);
    float exp_term = exp(-0.5f * qv);
    float grad_qv = -0.5f * alpha * d_alpha;
    float3 qd = load_q_row0(q_uvt, tube_id) * d.x + load_q_row1(q_uvt, tube_id) * d.y + load_q_row2(q_uvt, tube_id) * d.z;
    float3 grad_m = -2.0f * grad_qv * qd;
    grad_ma_samples[ma_base + 0u] = grad_m.x;
    grad_ma_samples[ma_base + 1u] = grad_m.y;
    grad_ma_samples[ma_base + 2u] = grad_m.z;
    grad_q_samples[q_base + 0u] = grad_qv * d.x * d.x;
    grad_q_samples[q_base + 1u] = grad_qv * 2.0f * d.x * d.y;
    grad_q_samples[q_base + 2u] = grad_qv * 2.0f * d.x * d.z;
    grad_q_samples[q_base + 3u] = grad_qv * d.y * d.y;
    grad_q_samples[q_base + 4u] = grad_qv * 2.0f * d.y * d.z;
    grad_q_samples[q_base + 5u] = grad_qv * d.z * d.z;
    grad_opacity_samples[entry] = d_alpha * exp_term;
  }
}

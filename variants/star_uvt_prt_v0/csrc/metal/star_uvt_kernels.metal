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
#ifndef STAR_FIXEDPOINT_SCALE
#define STAR_FIXEDPOINT_SCALE 1000000.0f
#endif
#ifndef STAR_SPLIT_FIXEDPOINT_COARSE_SCALE
#define STAR_SPLIT_FIXEDPOINT_COARSE_SCALE 100.0f
#endif
#ifndef STAR_SPLIT_FIXEDPOINT_FINE_SCALE
#define STAR_SPLIT_FIXEDPOINT_FINE_SCALE 1000000.0f
#endif

#define KAHAN_ADD(sum, compensation, value) \
  do { \
    auto kahan_value = (value) - (compensation); \
    auto kahan_next = (sum) + kahan_value; \
    (compensation) = (kahan_next - (sum)) - kahan_value; \
    (sum) = kahan_next; \
  } while (false)

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
  float support_alpha_threshold;
};

struct ReduceMeta {
  int sample_count;
  int tube_count;
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

inline void atomic_add3(device atomic_float* ptr, uint base, float3 value) {
  atomic_fetch_add_explicit(&ptr[base + 0u], value.x, memory_order_relaxed);
  atomic_fetch_add_explicit(&ptr[base + 1u], value.y, memory_order_relaxed);
  atomic_fetch_add_explicit(&ptr[base + 2u], value.z, memory_order_relaxed);
}

inline int fixedpoint_value(float value) {
  float scaled = round(value * STAR_FIXEDPOINT_SCALE);
  scaled = clamp(scaled, -2147483000.0f, 2147483000.0f);
  return int(scaled);
}

inline void atomic_add_fixedpoint(device atomic_int* ptr, uint index, float value) {
  atomic_fetch_add_explicit(&ptr[index], fixedpoint_value(value), memory_order_relaxed);
}

inline void atomic_add3_fixedpoint(device atomic_int* ptr, uint base, float3 value) {
  atomic_add_fixedpoint(ptr, base + 0u, value.x);
  atomic_add_fixedpoint(ptr, base + 1u, value.y);
  atomic_add_fixedpoint(ptr, base + 2u, value.z);
}

inline int fixedpoint_i32(float scaled) {
  scaled = round(scaled);
  scaled = clamp(scaled, -2147483000.0f, 2147483000.0f);
  return int(scaled);
}

inline void atomic_add_split_fixedpoint(device atomic_int* coarse, device atomic_int* fine, uint index, float value) {
  int coarse_value = fixedpoint_i32(value * STAR_SPLIT_FIXEDPOINT_COARSE_SCALE);
  float residual = value - float(coarse_value) / STAR_SPLIT_FIXEDPOINT_COARSE_SCALE;
  int fine_value = fixedpoint_i32(residual * STAR_SPLIT_FIXEDPOINT_FINE_SCALE);
  atomic_fetch_add_explicit(&coarse[index], coarse_value, memory_order_relaxed);
  atomic_fetch_add_explicit(&fine[index], fine_value, memory_order_relaxed);
}

inline void atomic_add3_split_fixedpoint(
    device atomic_int* coarse,
    device atomic_int* fine,
    uint base,
    float3 value) {
  atomic_add_split_fixedpoint(coarse, fine, base + 0u, value.x);
  atomic_add_split_fixedpoint(coarse, fine, base + 1u, value.y);
  atomic_add_split_fixedpoint(coarse, fine, base + 2u, value.z);
}

inline float eval_depth(const device float* ma, const device float* depth0, const device float* depth_beta, uint tube_id, float3 a) {
  float3 m = load3(ma, tube_id);
  float3 beta = load3(depth_beta, tube_id);
  return depth0[tube_id] + dot(beta, a - m);
}

inline float3 eval_prt_h(const device float* h_coeff, uint tube_id, uint h_terms, float tau) {
  float3 h = float3(0.0f);
  for (int p = int(h_terms) - 1; p >= 0; --p) {
    uint b = (tube_id * h_terms + uint(p)) * 3u;
    h = h * tau + float3(h_coeff[b + 0u], h_coeff[b + 1u], h_coeff[b + 2u]);
  }
  return h;
}

inline float3 eval_prt_h_dtau(const device float* h_coeff, uint tube_id, uint h_terms, float tau) {
  float3 dh = float3(0.0f);
  float power = 1.0f;
  for (uint p = 1u; p < h_terms; ++p) {
    uint b = (tube_id * h_terms + p) * 3u;
    dh += float(p) * power * float3(h_coeff[b + 0u], h_coeff[b + 1u], h_coeff[b + 2u]);
    power *= tau;
  }
  return dh;
}

inline bool inverse_sym2_diag(const device float* lambda_uv, uint tube_id, float eps, thread float2& diag_out) {
  uint b = tube_id * 3u;
  float a = lambda_uv[b + 0u];
  float c = lambda_uv[b + 1u];
  float d = lambda_uv[b + 2u];
  float det = a * d - c * c;
  if (!isfinite(det) || fabs(det) <= eps) {
    return false;
  }
  float inv_det = 1.0f / det;
  diag_out = abs(float2(d * inv_det, a * inv_det));
  return all(isfinite(diag_out));
}

inline float2 prt_spatial_half_extent_for_budget(
    const device float* lambda_uv,
    uint tube_id,
    float spatial_budget,
    constant MetaF32& mf) {
  if (!isfinite(spatial_budget) || spatial_budget <= 0.0f) return float2(-1.0f);

  float2 inv_diag;
  bool ok = inverse_sym2_diag(lambda_uv, tube_id, mf.eps, inv_diag);
  if (!ok) {
    return float2(INFINITY);
  }
  return sqrt(max(spatial_budget * inv_diag, float2(0.0f)));
}

inline float prt_alpha_at(
    uint tube_id,
    float2 pixel,
    float t,
    const device float* h_coeff,
    const device float* lambda_uv,
    const device float* lambda_t,
    const device float* center_t,
    const device float* opacity,
    uint h_terms,
    constant MetaF32& mf) {
  float tau = t - center_t[tube_id];
  float3 h = eval_prt_h(h_coeff, tube_id, h_terms, tau);
  float depth = max(h.z, mf.eps);
  float2 center = h.xy / depth;
  float2 d = pixel - center;
  uint b = tube_id * 3u;
  float spatial = lambda_uv[b + 0u] * d.x * d.x +
                  2.0f * lambda_uv[b + 1u] * d.x * d.y +
                  lambda_uv[b + 2u] * d.y * d.y;
  float temporal = lambda_t[tube_id] * tau * tau;
  return clamp(opacity[tube_id] * exp(-0.5f * (spatial + temporal)), 0.0f, mf.max_alpha);
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

inline void sort_by_depth_thread(thread uint* ids, thread float* depths, uint count) {
  for (uint i = 0u; i < count; ++i) {
    uint best = i;
    for (uint j = i + 1u; j < count; ++j) {
      bool better = (depths[j] < depths[best]) || (depths[j] == depths[best] && ids[j] < ids[best]);
      if (better) best = j;
    }
    if (best != i) {
      float d = depths[i];
      depths[i] = depths[best];
      depths[best] = d;
      uint id = ids[i];
      ids[i] = ids[best];
      ids[best] = id;
    }
  }
}

inline float reduce_threadgroup_sum(threadgroup float* scratch, float value, uint tid) {
  scratch[tid] = value;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint stride = STAR_THREADS >> 1u; stride > 0u; stride >>= 1u) {
    if (tid < stride) {
      scratch[tid] += scratch[tid + stride];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  return scratch[0];
}

kernel void reduce_sample_bundle_scan(
    const device int* ids [[buffer(0)]],
    const device float* grad_ma_samples [[buffer(1)]],
    const device float* grad_q_samples [[buffer(2)]],
    const device float* grad_opacity_samples [[buffer(3)]],
    const device float* grad_color_samples [[buffer(4)]],
    constant ReduceMeta& rm [[buffer(5)]],
    device float* grad_ma [[buffer(6)]],
    device float* grad_q [[buffer(7)]],
    device float* grad_opacity [[buffer(8)]],
    device float* grad_color [[buffer(9)]],
    uint tube_id [[thread_position_in_grid]]) {
  if (tube_id >= uint(rm.tube_count)) return;
  float ma0 = 0.0f;
  float ma1 = 0.0f;
  float ma2 = 0.0f;
  float q0 = 0.0f;
  float q1 = 0.0f;
  float q2 = 0.0f;
  float q3 = 0.0f;
  float q4 = 0.0f;
  float q5 = 0.0f;
  float opacity_sum = 0.0f;
  float color0 = 0.0f;
  float color1 = 0.0f;
  float color2 = 0.0f;
  int wanted = int(tube_id);
  for (uint row = 0u; row < uint(rm.sample_count); ++row) {
    if (ids[row] != wanted) continue;
    uint ma_base = row * 3u;
    ma0 += grad_ma_samples[ma_base + 0u];
    ma1 += grad_ma_samples[ma_base + 1u];
    ma2 += grad_ma_samples[ma_base + 2u];
    uint q_base = row * 6u;
    q0 += grad_q_samples[q_base + 0u];
    q1 += grad_q_samples[q_base + 1u];
    q2 += grad_q_samples[q_base + 2u];
    q3 += grad_q_samples[q_base + 3u];
    q4 += grad_q_samples[q_base + 4u];
    q5 += grad_q_samples[q_base + 5u];
    opacity_sum += grad_opacity_samples[row];
    uint color_base = row * 3u;
    color0 += grad_color_samples[color_base + 0u];
    color1 += grad_color_samples[color_base + 1u];
    color2 += grad_color_samples[color_base + 2u];
  }
  uint ma_out = tube_id * 3u;
  grad_ma[ma_out + 0u] = ma0;
  grad_ma[ma_out + 1u] = ma1;
  grad_ma[ma_out + 2u] = ma2;
  uint q_out = tube_id * 6u;
  grad_q[q_out + 0u] = q0;
  grad_q[q_out + 1u] = q1;
  grad_q[q_out + 2u] = q2;
  grad_q[q_out + 3u] = q3;
  grad_q[q_out + 4u] = q4;
  grad_q[q_out + 5u] = q5;
  grad_opacity[tube_id] = opacity_sum;
  uint color_out = tube_id * 3u;
  grad_color[color_out + 0u] = color0;
  grad_color[color_out + 1u] = color1;
  grad_color[color_out + 2u] = color2;
}

kernel void reduce_sample_bundle_scan_compensated(
    const device int* ids [[buffer(0)]],
    const device float* grad_ma_samples [[buffer(1)]],
    const device float* grad_q_samples [[buffer(2)]],
    const device float* grad_opacity_samples [[buffer(3)]],
    const device float* grad_color_samples [[buffer(4)]],
    constant ReduceMeta& rm [[buffer(5)]],
    device float* grad_ma [[buffer(6)]],
    device float* grad_q [[buffer(7)]],
    device float* grad_opacity [[buffer(8)]],
    device float* grad_color [[buffer(9)]],
    uint tube_id [[thread_position_in_grid]]) {
  if (tube_id >= uint(rm.tube_count)) return;
  float ma0 = 0.0f;
  float ma1 = 0.0f;
  float ma2 = 0.0f;
  float q0 = 0.0f;
  float q1 = 0.0f;
  float q2 = 0.0f;
  float q3 = 0.0f;
  float q4 = 0.0f;
  float q5 = 0.0f;
  float opacity_sum = 0.0f;
  float color0 = 0.0f;
  float color1 = 0.0f;
  float color2 = 0.0f;
  float ma0_comp = 0.0f;
  float ma1_comp = 0.0f;
  float ma2_comp = 0.0f;
  float q0_comp = 0.0f;
  float q1_comp = 0.0f;
  float q2_comp = 0.0f;
  float q3_comp = 0.0f;
  float q4_comp = 0.0f;
  float q5_comp = 0.0f;
  float opacity_comp = 0.0f;
  float color0_comp = 0.0f;
  float color1_comp = 0.0f;
  float color2_comp = 0.0f;
  int wanted = int(tube_id);
  for (uint row = 0u; row < uint(rm.sample_count); ++row) {
    if (ids[row] != wanted) continue;
    uint ma_base = row * 3u;
    KAHAN_ADD(ma0, ma0_comp, grad_ma_samples[ma_base + 0u]);
    KAHAN_ADD(ma1, ma1_comp, grad_ma_samples[ma_base + 1u]);
    KAHAN_ADD(ma2, ma2_comp, grad_ma_samples[ma_base + 2u]);
    uint q_base = row * 6u;
    KAHAN_ADD(q0, q0_comp, grad_q_samples[q_base + 0u]);
    KAHAN_ADD(q1, q1_comp, grad_q_samples[q_base + 1u]);
    KAHAN_ADD(q2, q2_comp, grad_q_samples[q_base + 2u]);
    KAHAN_ADD(q3, q3_comp, grad_q_samples[q_base + 3u]);
    KAHAN_ADD(q4, q4_comp, grad_q_samples[q_base + 4u]);
    KAHAN_ADD(q5, q5_comp, grad_q_samples[q_base + 5u]);
    KAHAN_ADD(opacity_sum, opacity_comp, grad_opacity_samples[row]);
    uint color_base = row * 3u;
    KAHAN_ADD(color0, color0_comp, grad_color_samples[color_base + 0u]);
    KAHAN_ADD(color1, color1_comp, grad_color_samples[color_base + 1u]);
    KAHAN_ADD(color2, color2_comp, grad_color_samples[color_base + 2u]);
  }
  uint ma_out = tube_id * 3u;
  grad_ma[ma_out + 0u] = ma0;
  grad_ma[ma_out + 1u] = ma1;
  grad_ma[ma_out + 2u] = ma2;
  uint q_out = tube_id * 6u;
  grad_q[q_out + 0u] = q0;
  grad_q[q_out + 1u] = q1;
  grad_q[q_out + 2u] = q2;
  grad_q[q_out + 3u] = q3;
  grad_q[q_out + 4u] = q4;
  grad_q[q_out + 5u] = q5;
  grad_opacity[tube_id] = opacity_sum;
  uint color_out = tube_id * 3u;
  grad_color[color_out + 0u] = color0;
  grad_color[color_out + 1u] = color1;
  grad_color[color_out + 2u] = color2;
}

inline uint lower_bound_sample_id(const device int* ids, uint count, int wanted) {
  uint lo = 0u;
  uint hi = count;
  while (lo < hi) {
    uint mid = lo + ((hi - lo) >> 1u);
    if (ids[mid] < wanted) {
      lo = mid + 1u;
    } else {
      hi = mid;
    }
  }
  return lo;
}

kernel void reduce_sample_bundle_sorted_segments(
    const device int* ids [[buffer(0)]],
    const device float* grad_ma_samples [[buffer(1)]],
    const device float* grad_q_samples [[buffer(2)]],
    const device float* grad_opacity_samples [[buffer(3)]],
    const device float* grad_color_samples [[buffer(4)]],
    constant ReduceMeta& rm [[buffer(5)]],
    device float* grad_ma [[buffer(6)]],
    device float* grad_q [[buffer(7)]],
    device float* grad_opacity [[buffer(8)]],
    device float* grad_color [[buffer(9)]],
    uint tube_id [[thread_position_in_grid]]) {
  if (tube_id >= uint(rm.tube_count)) return;
  uint sample_count = uint(rm.sample_count);
  int wanted = int(tube_id);
  uint start = lower_bound_sample_id(ids, sample_count, wanted);
  uint end = lower_bound_sample_id(ids, sample_count, wanted + 1);

  float ma0 = 0.0f;
  float ma1 = 0.0f;
  float ma2 = 0.0f;
  float q0 = 0.0f;
  float q1 = 0.0f;
  float q2 = 0.0f;
  float q3 = 0.0f;
  float q4 = 0.0f;
  float q5 = 0.0f;
  float opacity_sum = 0.0f;
  float color0 = 0.0f;
  float color1 = 0.0f;
  float color2 = 0.0f;

  for (uint row = start; row < end; ++row) {
    uint ma_base = row * 3u;
    ma0 += grad_ma_samples[ma_base + 0u];
    ma1 += grad_ma_samples[ma_base + 1u];
    ma2 += grad_ma_samples[ma_base + 2u];
    uint q_base = row * 6u;
    q0 += grad_q_samples[q_base + 0u];
    q1 += grad_q_samples[q_base + 1u];
    q2 += grad_q_samples[q_base + 2u];
    q3 += grad_q_samples[q_base + 3u];
    q4 += grad_q_samples[q_base + 4u];
    q5 += grad_q_samples[q_base + 5u];
    opacity_sum += grad_opacity_samples[row];
    uint color_base = row * 3u;
    color0 += grad_color_samples[color_base + 0u];
    color1 += grad_color_samples[color_base + 1u];
    color2 += grad_color_samples[color_base + 2u];
  }

  uint ma_out = tube_id * 3u;
  grad_ma[ma_out + 0u] = ma0;
  grad_ma[ma_out + 1u] = ma1;
  grad_ma[ma_out + 2u] = ma2;
  uint q_out = tube_id * 6u;
  grad_q[q_out + 0u] = q0;
  grad_q[q_out + 1u] = q1;
  grad_q[q_out + 2u] = q2;
  grad_q[q_out + 3u] = q3;
  grad_q[q_out + 4u] = q4;
  grad_q[q_out + 5u] = q5;
  grad_opacity[tube_id] = opacity_sum;
  uint color_out = tube_id * 3u;
  grad_color[color_out + 0u] = color0;
  grad_color[color_out + 1u] = color1;
  grad_color[color_out + 2u] = color2;
}

kernel void reduce_tile_pair_bounds_scan(
    const device float* ma [[buffer(0)]],
    const device float* q_uvt [[buffer(1)]],
    const device float* opacity [[buffer(2)]],
    constant MetaI32& mi [[buffer(3)]],
    constant MetaF32& mf [[buffer(4)]],
    const device int* grad_ids [[buffer(5)]],
    const device float* grad_ma_samples [[buffer(6)]],
    const device float* grad_q_samples [[buffer(7)]],
    const device float* grad_opacity_samples [[buffer(8)]],
    const device float* grad_color_samples [[buffer(9)]],
    device float* grad_ma [[buffer(10)]],
    device float* grad_q [[buffer(11)]],
    device float* grad_opacity [[buffer(12)]],
    device float* grad_color [[buffer(13)]],
    uint tube_id [[thread_position_in_grid]]) {
  if (tube_id >= uint(mi.tube_count)) return;

  float ma0 = 0.0f;
  float ma1 = 0.0f;
  float ma2 = 0.0f;
  float q0 = 0.0f;
  float q1 = 0.0f;
  float q2 = 0.0f;
  float q3 = 0.0f;
  float q4 = 0.0f;
  float q5 = 0.0f;
  float opacity_sum = 0.0f;
  float color0 = 0.0f;
  float color1 = 0.0f;
  float color2 = 0.0f;

  Bounds3i b = tube_bounds(ma, q_uvt, opacity, tube_id, mi, mf);
  if (b.x0 <= b.x1 && b.y0 <= b.y1 && b.f0 <= b.f1) {
    uint tx0 = uint(b.x0 / mi.tile_x);
    uint tx1 = uint(b.x1 / mi.tile_x);
    uint ty0 = uint(b.y0 / mi.tile_y);
    uint ty1 = uint(b.y1 / mi.tile_y);
    uint tz0 = uint(b.f0 / mi.tile_t);
    uint tz1 = uint(b.f1 / mi.tile_t);
    int wanted = int(tube_id);

    for (uint tz = tz0; tz <= tz1; ++tz) {
      for (uint ty = ty0; ty <= ty1; ++ty) {
        for (uint tx = tx0; tx <= tx1; ++tx) {
          uint row_base = encode_tile(tx, ty, tz, mi) * STAR_TILE_CAPACITY;
          for (uint slot = 0u; slot < STAR_TILE_CAPACITY; ++slot) {
            uint row = row_base + slot;
            if (grad_ids[row] != wanted) continue;
            uint ma_base = row * 3u;
            ma0 += grad_ma_samples[ma_base + 0u];
            ma1 += grad_ma_samples[ma_base + 1u];
            ma2 += grad_ma_samples[ma_base + 2u];
            uint q_base = row * 6u;
            q0 += grad_q_samples[q_base + 0u];
            q1 += grad_q_samples[q_base + 1u];
            q2 += grad_q_samples[q_base + 2u];
            q3 += grad_q_samples[q_base + 3u];
            q4 += grad_q_samples[q_base + 4u];
            q5 += grad_q_samples[q_base + 5u];
            opacity_sum += grad_opacity_samples[row];
            uint color_base = row * 3u;
            color0 += grad_color_samples[color_base + 0u];
            color1 += grad_color_samples[color_base + 1u];
            color2 += grad_color_samples[color_base + 2u];
          }
        }
      }
    }
  }

  uint ma_out = tube_id * 3u;
  grad_ma[ma_out + 0u] = ma0;
  grad_ma[ma_out + 1u] = ma1;
  grad_ma[ma_out + 2u] = ma2;
  uint q_out = tube_id * 6u;
  grad_q[q_out + 0u] = q0;
  grad_q[q_out + 1u] = q1;
  grad_q[q_out + 2u] = q2;
  grad_q[q_out + 3u] = q3;
  grad_q[q_out + 4u] = q4;
  grad_q[q_out + 5u] = q5;
  grad_opacity[tube_id] = opacity_sum;
  uint color_out = tube_id * 3u;
  grad_color[color_out + 0u] = color0;
  grad_color[color_out + 1u] = color1;
  grad_color[color_out + 2u] = color2;
}

kernel void reduce_tile_pair_bounds_scan_parallel(
    const device float* ma [[buffer(0)]],
    const device float* q_uvt [[buffer(1)]],
    const device float* opacity [[buffer(2)]],
    constant MetaI32& mi [[buffer(3)]],
    constant MetaF32& mf [[buffer(4)]],
    const device int* grad_ids [[buffer(5)]],
    const device float* grad_ma_samples [[buffer(6)]],
    const device float* grad_q_samples [[buffer(7)]],
    const device float* grad_opacity_samples [[buffer(8)]],
    const device float* grad_color_samples [[buffer(9)]],
    device float* grad_ma [[buffer(10)]],
    device float* grad_q [[buffer(11)]],
    device float* grad_opacity [[buffer(12)]],
    device float* grad_color [[buffer(13)]],
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]]) {
  uint tube_id = gid / STAR_THREADS;
  uint local_tid = tid;

  float ma0 = 0.0f;
  float ma1 = 0.0f;
  float ma2 = 0.0f;
  float q0 = 0.0f;
  float q1 = 0.0f;
  float q2 = 0.0f;
  float q3 = 0.0f;
  float q4 = 0.0f;
  float q5 = 0.0f;
  float opacity_sum = 0.0f;
  float color0 = 0.0f;
  float color1 = 0.0f;
  float color2 = 0.0f;
  threadgroup float reduce_scratch[STAR_THREADS];

  if (tube_id < uint(mi.tube_count)) {
    Bounds3i b = tube_bounds(ma, q_uvt, opacity, tube_id, mi, mf);
    if (b.x0 <= b.x1 && b.y0 <= b.y1 && b.f0 <= b.f1) {
      uint tx0 = uint(b.x0 / mi.tile_x);
      uint tx1 = uint(b.x1 / mi.tile_x);
      uint ty0 = uint(b.y0 / mi.tile_y);
      uint ty1 = uint(b.y1 / mi.tile_y);
      uint tz0 = uint(b.f0 / mi.tile_t);
      uint tz1 = uint(b.f1 / mi.tile_t);
      int wanted = int(tube_id);
      uint work_index = 0u;

      for (uint tz = tz0; tz <= tz1; ++tz) {
        for (uint ty = ty0; ty <= ty1; ++ty) {
          for (uint tx = tx0; tx <= tx1; ++tx) {
            uint row_base = encode_tile(tx, ty, tz, mi) * STAR_TILE_CAPACITY;
            for (uint slot = 0u; slot < STAR_TILE_CAPACITY; ++slot) {
              uint row = row_base + slot;
              bool take = (work_index % STAR_THREADS) == local_tid;
              work_index += 1u;
              if (!take || grad_ids[row] != wanted) continue;
              uint ma_base = row * 3u;
              ma0 += grad_ma_samples[ma_base + 0u];
              ma1 += grad_ma_samples[ma_base + 1u];
              ma2 += grad_ma_samples[ma_base + 2u];
              uint q_base = row * 6u;
              q0 += grad_q_samples[q_base + 0u];
              q1 += grad_q_samples[q_base + 1u];
              q2 += grad_q_samples[q_base + 2u];
              q3 += grad_q_samples[q_base + 3u];
              q4 += grad_q_samples[q_base + 4u];
              q5 += grad_q_samples[q_base + 5u];
              opacity_sum += grad_opacity_samples[row];
              uint color_base = row * 3u;
              color0 += grad_color_samples[color_base + 0u];
              color1 += grad_color_samples[color_base + 1u];
              color2 += grad_color_samples[color_base + 2u];
            }
          }
        }
      }
    }
  }

  float ma0_sum = reduce_threadgroup_sum(reduce_scratch, ma0, local_tid);
  float ma1_sum = reduce_threadgroup_sum(reduce_scratch, ma1, local_tid);
  float ma2_sum = reduce_threadgroup_sum(reduce_scratch, ma2, local_tid);
  float q0_sum = reduce_threadgroup_sum(reduce_scratch, q0, local_tid);
  float q1_sum = reduce_threadgroup_sum(reduce_scratch, q1, local_tid);
  float q2_sum = reduce_threadgroup_sum(reduce_scratch, q2, local_tid);
  float q3_sum = reduce_threadgroup_sum(reduce_scratch, q3, local_tid);
  float q4_sum = reduce_threadgroup_sum(reduce_scratch, q4, local_tid);
  float q5_sum = reduce_threadgroup_sum(reduce_scratch, q5, local_tid);
  float opacity_value = reduce_threadgroup_sum(reduce_scratch, opacity_sum, local_tid);
  float color0_sum = reduce_threadgroup_sum(reduce_scratch, color0, local_tid);
  float color1_sum = reduce_threadgroup_sum(reduce_scratch, color1, local_tid);
  float color2_sum = reduce_threadgroup_sum(reduce_scratch, color2, local_tid);

  if (local_tid == 0u && tube_id < uint(mi.tube_count)) {
    uint ma_out = tube_id * 3u;
    grad_ma[ma_out + 0u] = ma0_sum;
    grad_ma[ma_out + 1u] = ma1_sum;
    grad_ma[ma_out + 2u] = ma2_sum;
    uint q_out = tube_id * 6u;
    grad_q[q_out + 0u] = q0_sum;
    grad_q[q_out + 1u] = q1_sum;
    grad_q[q_out + 2u] = q2_sum;
    grad_q[q_out + 3u] = q3_sum;
    grad_q[q_out + 4u] = q4_sum;
    grad_q[q_out + 5u] = q5_sum;
    grad_opacity[tube_id] = opacity_value;
    uint color_out = tube_id * 3u;
    grad_color[color_out + 0u] = color0_sum;
    grad_color[color_out + 1u] = color1_sum;
    grad_color[color_out + 2u] = color2_sum;
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

inline bool tile_order_unstable_thread(
    thread uint* ids,
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

inline void composite_prt_tube(
    uint tube_id,
    float2 pixel,
    float t,
    const device float* h_coeff,
    const device float* lambda_uv,
    const device float* lambda_t,
    const device float* center_t,
    const device float* opacity,
    const device float* color,
    uint h_terms,
    constant MetaF32& mf,
    thread float3& accum,
    thread float& transmittance) {
  float alpha = prt_alpha_at(tube_id, pixel, t, h_coeff, lambda_uv, lambda_t, center_t, opacity, h_terms, mf);
  if (!(alpha >= mf.alpha_threshold)) return;
  uint cbase = tube_id * 3u;
  float3 c = float3(color[cbase + 0u], color[cbase + 1u], color[cbase + 2u]);
  accum += transmittance * alpha * c;
  transmittance *= (1.0f - alpha);
}

inline uint select_prt_sample_order_id(
    threadgroup uint* ids,
    uint count,
    const device float* h_coeff,
    const device float* center_t,
    uint h_terms,
    float t,
    float last_depth,
    uint last_id,
    constant MetaF32& mf,
    thread float& out_depth) {
  uint best_id = 0xFFFFFFFFu;
  float best_depth = INFINITY;
  for (uint i = 0u; i < count; ++i) {
    uint tube_id = ids[i];
    float tau = t - center_t[tube_id];
    float3 h = eval_prt_h(h_coeff, tube_id, h_terms, tau);
    float d = max(h.z, mf.eps);
    bool after_last = (last_id == 0xFFFFFFFFu) || (d > last_depth) || (d == last_depth && tube_id > last_id);
    bool better = (d < best_depth) || (d == best_depth && tube_id < best_id);
    if (after_last && better) {
      best_depth = d;
      best_id = tube_id;
    }
  }
  out_depth = best_depth;
  return best_id;
}

inline uint select_prt_sample_order_id_thread(
    thread uint* ids,
    uint count,
    const device float* h_coeff,
    const device float* center_t,
    uint h_terms,
    float t,
    float last_depth,
    uint last_id,
    constant MetaF32& mf,
    thread float& out_depth) {
  uint best_id = 0xFFFFFFFFu;
  float best_depth = INFINITY;
  for (uint i = 0u; i < count; ++i) {
    uint tube_id = ids[i];
    float tau = t - center_t[tube_id];
    float3 h = eval_prt_h(h_coeff, tube_id, h_terms, tau);
    float d = max(h.z, mf.eps);
    bool after_last = (last_id == 0xFFFFFFFFu) || (d > last_depth) || (d == last_depth && tube_id > last_id);
    bool better = (d < best_depth) || (d == best_depth && tube_id < best_id);
    if (after_last && better) {
      best_depth = d;
      best_id = tube_id;
    }
  }
  out_depth = best_depth;
  return best_id;
}

inline uint select_prt_direct_order_id(
    const device float* h_coeff,
    const device float* lambda_uv,
    const device float* lambda_t,
    const device float* center_t,
    const device float* opacity,
    uint h_terms,
    uint tube_count,
    float2 pixel,
    float t,
    float last_depth,
    uint last_id,
    constant MetaF32& mf,
    thread float& out_depth,
    thread float& out_alpha,
    thread float& out_alpha_raw) {
  uint best_id = 0xFFFFFFFFu;
  float best_depth = INFINITY;
  float best_alpha = 0.0f;
  float best_alpha_raw = 0.0f;
  for (uint tube_id = 0u; tube_id < tube_count; ++tube_id) {
    float tau = t - center_t[tube_id];
    float3 h = eval_prt_h(h_coeff, tube_id, h_terms, tau);
    float depth = max(h.z, mf.eps);
    bool after_last = (last_id == 0xFFFFFFFFu) || (depth > last_depth) ||
                      (depth == last_depth && tube_id > last_id);
    if (!after_last) continue;

    float2 center = h.xy / depth;
    float2 d = pixel - center;
    uint qbase = tube_id * 3u;
    float spatial = lambda_uv[qbase + 0u] * d.x * d.x +
                    2.0f * lambda_uv[qbase + 1u] * d.x * d.y +
                    lambda_uv[qbase + 2u] * d.y * d.y;
    float temporal = lambda_t[tube_id] * tau * tau;
    float qv = spatial + temporal;
    if (!isfinite(qv)) continue;
    float alpha_raw = opacity[tube_id] * exp(-0.5f * qv);
    float alpha = min(mf.max_alpha, alpha_raw);
    if (!(alpha >= mf.alpha_threshold)) continue;

    bool better = (depth < best_depth) || (depth == best_depth && tube_id < best_id);
    if (better) {
      best_depth = depth;
      best_id = tube_id;
      best_alpha = alpha;
      best_alpha_raw = alpha_raw;
    }
  }
  out_depth = best_depth;
  out_alpha = best_alpha;
  out_alpha_raw = best_alpha_raw;
  return best_id;
}

inline uint select_sample_order_id_thread(
    thread uint* ids,
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

kernel void clear_direct_gradients(
    device float* grad_ma [[buffer(0)]],
    device float* grad_q [[buffer(1)]],
    device float* grad_opacity [[buffer(2)]],
    device float* grad_color [[buffer(3)]],
    constant MetaI32& mi [[buffer(4)]],
    uint tube_id [[thread_position_in_grid]]) {
  if (tube_id >= uint(mi.tube_count)) return;
  uint ma_base = tube_id * 3u;
  grad_ma[ma_base + 0u] = 0.0f;
  grad_ma[ma_base + 1u] = 0.0f;
  grad_ma[ma_base + 2u] = 0.0f;
  uint q_base = tube_id * 6u;
  grad_q[q_base + 0u] = 0.0f;
  grad_q[q_base + 1u] = 0.0f;
  grad_q[q_base + 2u] = 0.0f;
  grad_q[q_base + 3u] = 0.0f;
  grad_q[q_base + 4u] = 0.0f;
  grad_q[q_base + 5u] = 0.0f;
  grad_opacity[tube_id] = 0.0f;
  uint color_base = tube_id * 3u;
  grad_color[color_base + 0u] = 0.0f;
  grad_color[color_base + 1u] = 0.0f;
  grad_color[color_base + 2u] = 0.0f;
}

kernel void clear_prt_gradients(
    device float* grad_h_coeff [[buffer(0)]],
    device float* grad_lambda_uv [[buffer(1)]],
    device float* grad_lambda_t [[buffer(2)]],
    device float* grad_center_t [[buffer(3)]],
    device float* grad_opacity [[buffer(4)]],
    device float* grad_color [[buffer(5)]],
    constant MetaI32& mi [[buffer(6)]],
    uint tube_id [[thread_position_in_grid]]) {
  if (tube_id >= uint(mi.tube_count)) return;
  uint h_terms = uint(mi.reserved0);
  for (uint p = 0u; p < h_terms; ++p) {
    uint h_base = (tube_id * h_terms + p) * 3u;
    grad_h_coeff[h_base + 0u] = 0.0f;
    grad_h_coeff[h_base + 1u] = 0.0f;
    grad_h_coeff[h_base + 2u] = 0.0f;
  }
  uint q_base = tube_id * 3u;
  grad_lambda_uv[q_base + 0u] = 0.0f;
  grad_lambda_uv[q_base + 1u] = 0.0f;
  grad_lambda_uv[q_base + 2u] = 0.0f;
  grad_lambda_t[tube_id] = 0.0f;
  grad_center_t[tube_id] = 0.0f;
  grad_opacity[tube_id] = 0.0f;
  uint color_base = tube_id * 3u;
  grad_color[color_base + 0u] = 0.0f;
  grad_color[color_base + 1u] = 0.0f;
  grad_color[color_base + 2u] = 0.0f;
}

kernel void clear_direct_gradients_i32(
    device atomic_int* grad_ma [[buffer(0)]],
    device atomic_int* grad_q [[buffer(1)]],
    device atomic_int* grad_opacity [[buffer(2)]],
    device atomic_int* grad_color [[buffer(3)]],
    constant MetaI32& mi [[buffer(4)]],
    uint tube_id [[thread_position_in_grid]]) {
  if (tube_id >= uint(mi.tube_count)) return;
  uint ma_base = tube_id * 3u;
  atomic_store_explicit(&grad_ma[ma_base + 0u], 0, memory_order_relaxed);
  atomic_store_explicit(&grad_ma[ma_base + 1u], 0, memory_order_relaxed);
  atomic_store_explicit(&grad_ma[ma_base + 2u], 0, memory_order_relaxed);
  uint q_base = tube_id * 6u;
  atomic_store_explicit(&grad_q[q_base + 0u], 0, memory_order_relaxed);
  atomic_store_explicit(&grad_q[q_base + 1u], 0, memory_order_relaxed);
  atomic_store_explicit(&grad_q[q_base + 2u], 0, memory_order_relaxed);
  atomic_store_explicit(&grad_q[q_base + 3u], 0, memory_order_relaxed);
  atomic_store_explicit(&grad_q[q_base + 4u], 0, memory_order_relaxed);
  atomic_store_explicit(&grad_q[q_base + 5u], 0, memory_order_relaxed);
  atomic_store_explicit(&grad_opacity[tube_id], 0, memory_order_relaxed);
  uint color_base = tube_id * 3u;
  atomic_store_explicit(&grad_color[color_base + 0u], 0, memory_order_relaxed);
  atomic_store_explicit(&grad_color[color_base + 1u], 0, memory_order_relaxed);
  atomic_store_explicit(&grad_color[color_base + 2u], 0, memory_order_relaxed);
}

kernel void fixedpoint_gradients_to_float(
    const device atomic_int* grad_ma_i32 [[buffer(0)]],
    const device atomic_int* grad_q_i32 [[buffer(1)]],
    const device atomic_int* grad_opacity_i32 [[buffer(2)]],
    const device atomic_int* grad_color_i32 [[buffer(3)]],
    constant MetaI32& mi [[buffer(4)]],
    device float* grad_ma [[buffer(5)]],
    device float* grad_q [[buffer(6)]],
    device float* grad_opacity [[buffer(7)]],
    device float* grad_color [[buffer(8)]],
    uint tube_id [[thread_position_in_grid]]) {
  if (tube_id >= uint(mi.tube_count)) return;
  float inv_scale = 1.0f / STAR_FIXEDPOINT_SCALE;
  uint ma_base = tube_id * 3u;
  grad_ma[ma_base + 0u] = float(atomic_load_explicit(&grad_ma_i32[ma_base + 0u], memory_order_relaxed)) * inv_scale;
  grad_ma[ma_base + 1u] = float(atomic_load_explicit(&grad_ma_i32[ma_base + 1u], memory_order_relaxed)) * inv_scale;
  grad_ma[ma_base + 2u] = float(atomic_load_explicit(&grad_ma_i32[ma_base + 2u], memory_order_relaxed)) * inv_scale;
  uint q_base = tube_id * 6u;
  grad_q[q_base + 0u] = float(atomic_load_explicit(&grad_q_i32[q_base + 0u], memory_order_relaxed)) * inv_scale;
  grad_q[q_base + 1u] = float(atomic_load_explicit(&grad_q_i32[q_base + 1u], memory_order_relaxed)) * inv_scale;
  grad_q[q_base + 2u] = float(atomic_load_explicit(&grad_q_i32[q_base + 2u], memory_order_relaxed)) * inv_scale;
  grad_q[q_base + 3u] = float(atomic_load_explicit(&grad_q_i32[q_base + 3u], memory_order_relaxed)) * inv_scale;
  grad_q[q_base + 4u] = float(atomic_load_explicit(&grad_q_i32[q_base + 4u], memory_order_relaxed)) * inv_scale;
  grad_q[q_base + 5u] = float(atomic_load_explicit(&grad_q_i32[q_base + 5u], memory_order_relaxed)) * inv_scale;
  grad_opacity[tube_id] = float(atomic_load_explicit(&grad_opacity_i32[tube_id], memory_order_relaxed)) * inv_scale;
  uint color_base = tube_id * 3u;
  grad_color[color_base + 0u] = float(atomic_load_explicit(&grad_color_i32[color_base + 0u], memory_order_relaxed)) * inv_scale;
  grad_color[color_base + 1u] = float(atomic_load_explicit(&grad_color_i32[color_base + 1u], memory_order_relaxed)) * inv_scale;
  grad_color[color_base + 2u] = float(atomic_load_explicit(&grad_color_i32[color_base + 2u], memory_order_relaxed)) * inv_scale;
}

kernel void split_fixedpoint_gradients_to_float(
    const device atomic_int* grad_ma_coarse [[buffer(0)]],
    const device atomic_int* grad_q_coarse [[buffer(1)]],
    const device atomic_int* grad_opacity_coarse [[buffer(2)]],
    const device atomic_int* grad_color_coarse [[buffer(3)]],
    const device atomic_int* grad_ma_fine [[buffer(4)]],
    const device atomic_int* grad_q_fine [[buffer(5)]],
    const device atomic_int* grad_opacity_fine [[buffer(6)]],
    const device atomic_int* grad_color_fine [[buffer(7)]],
    constant MetaI32& mi [[buffer(8)]],
    device float* grad_ma [[buffer(9)]],
    device float* grad_q [[buffer(10)]],
    device float* grad_opacity [[buffer(11)]],
    device float* grad_color [[buffer(12)]],
    uint tube_id [[thread_position_in_grid]]) {
  if (tube_id >= uint(mi.tube_count)) return;
  float inv_coarse = 1.0f / STAR_SPLIT_FIXEDPOINT_COARSE_SCALE;
  float inv_fine = 1.0f / STAR_SPLIT_FIXEDPOINT_FINE_SCALE;
  uint ma_base = tube_id * 3u;
  grad_ma[ma_base + 0u] =
      float(atomic_load_explicit(&grad_ma_coarse[ma_base + 0u], memory_order_relaxed)) * inv_coarse +
      float(atomic_load_explicit(&grad_ma_fine[ma_base + 0u], memory_order_relaxed)) * inv_fine;
  grad_ma[ma_base + 1u] =
      float(atomic_load_explicit(&grad_ma_coarse[ma_base + 1u], memory_order_relaxed)) * inv_coarse +
      float(atomic_load_explicit(&grad_ma_fine[ma_base + 1u], memory_order_relaxed)) * inv_fine;
  grad_ma[ma_base + 2u] =
      float(atomic_load_explicit(&grad_ma_coarse[ma_base + 2u], memory_order_relaxed)) * inv_coarse +
      float(atomic_load_explicit(&grad_ma_fine[ma_base + 2u], memory_order_relaxed)) * inv_fine;
  uint q_base = tube_id * 6u;
  grad_q[q_base + 0u] =
      float(atomic_load_explicit(&grad_q_coarse[q_base + 0u], memory_order_relaxed)) * inv_coarse +
      float(atomic_load_explicit(&grad_q_fine[q_base + 0u], memory_order_relaxed)) * inv_fine;
  grad_q[q_base + 1u] =
      float(atomic_load_explicit(&grad_q_coarse[q_base + 1u], memory_order_relaxed)) * inv_coarse +
      float(atomic_load_explicit(&grad_q_fine[q_base + 1u], memory_order_relaxed)) * inv_fine;
  grad_q[q_base + 2u] =
      float(atomic_load_explicit(&grad_q_coarse[q_base + 2u], memory_order_relaxed)) * inv_coarse +
      float(atomic_load_explicit(&grad_q_fine[q_base + 2u], memory_order_relaxed)) * inv_fine;
  grad_q[q_base + 3u] =
      float(atomic_load_explicit(&grad_q_coarse[q_base + 3u], memory_order_relaxed)) * inv_coarse +
      float(atomic_load_explicit(&grad_q_fine[q_base + 3u], memory_order_relaxed)) * inv_fine;
  grad_q[q_base + 4u] =
      float(atomic_load_explicit(&grad_q_coarse[q_base + 4u], memory_order_relaxed)) * inv_coarse +
      float(atomic_load_explicit(&grad_q_fine[q_base + 4u], memory_order_relaxed)) * inv_fine;
  grad_q[q_base + 5u] =
      float(atomic_load_explicit(&grad_q_coarse[q_base + 5u], memory_order_relaxed)) * inv_coarse +
      float(atomic_load_explicit(&grad_q_fine[q_base + 5u], memory_order_relaxed)) * inv_fine;
  grad_opacity[tube_id] =
      float(atomic_load_explicit(&grad_opacity_coarse[tube_id], memory_order_relaxed)) * inv_coarse +
      float(atomic_load_explicit(&grad_opacity_fine[tube_id], memory_order_relaxed)) * inv_fine;
  uint color_base = tube_id * 3u;
  grad_color[color_base + 0u] =
      float(atomic_load_explicit(&grad_color_coarse[color_base + 0u], memory_order_relaxed)) * inv_coarse +
      float(atomic_load_explicit(&grad_color_fine[color_base + 0u], memory_order_relaxed)) * inv_fine;
  grad_color[color_base + 1u] =
      float(atomic_load_explicit(&grad_color_coarse[color_base + 1u], memory_order_relaxed)) * inv_coarse +
      float(atomic_load_explicit(&grad_color_fine[color_base + 1u], memory_order_relaxed)) * inv_fine;
  grad_color[color_base + 2u] =
      float(atomic_load_explicit(&grad_color_coarse[color_base + 2u], memory_order_relaxed)) * inv_coarse +
      float(atomic_load_explicit(&grad_color_fine[color_base + 2u], memory_order_relaxed)) * inv_fine;
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

kernel void bin_projective_rational_tubes_to_uvt_tiles(
    const device float* h_coeff [[buffer(0)]],
    const device float* lambda_uv [[buffer(1)]],
    const device float* lambda_t [[buffer(2)]],
    const device float* center_t [[buffer(3)]],
    const device float* opacity [[buffer(4)]],
    constant MetaI32& mi [[buffer(5)]],
    constant MetaF32& mf [[buffer(6)]],
    device atomic_uint* tile_counts [[buffer(7)]],
    device uint* tile_tube_ids [[buffer(8)]],
    device float* tile_depths [[buffer(9)]],
    device atomic_uint* tile_overflow [[buffer(10)]],
    uint tube_id [[thread_position_in_grid]]) {
  if (tube_id >= uint(mi.tube_count)) return;
  uint h_terms = uint(mi.reserved0);

  float op = opacity[tube_id];
  if (!(op > mf.support_alpha_threshold)) return;
  float support_tau = -2.0f * log(max(mf.support_alpha_threshold / max(op, mf.eps), mf.eps));
  if (!isfinite(support_tau) || support_tau <= 0.0f) return;

  float time_precision = lambda_t[tube_id];
  float half_t = time_precision > mf.eps ? sqrt(max(support_tau / time_precision, 0.0f)) : float(mi.frames);
  float frame_center = 0.5f * float(mi.frames - 1);
  int f0 = max(0, int(floor(center_t[tube_id] - half_t + frame_center)));
  int f1 = min(mi.frames - 1, int(ceil(center_t[tube_id] + half_t + frame_center)));
  if (f0 > f1) return;

  uint tz0 = uint(f0 / mi.tile_t);
  uint tz1 = uint(f1 / mi.tile_t);
  for (uint tz = tz0; tz <= tz1; ++tz) {
    uint tile_f0 = tz * uint(mi.tile_t);
    uint tile_f1 = min(uint(mi.frames - 1), tile_f0 + uint(mi.tile_t - 1));
    uint zf0 = max(uint(f0), tile_f0);
    uint zf1 = min(uint(f1), tile_f1);
    int x0 = mi.width;
    int x1 = -1;
    int y0 = mi.height;
    int y1 = -1;
    for (uint f = zf0; f <= zf1; ++f) {
      float t = frame_time(f, mi);
      float tau_t = t - center_t[tube_id];
      float spatial_budget = support_tau - time_precision * tau_t * tau_t;
      float2 half_xy = prt_spatial_half_extent_for_budget(lambda_uv, tube_id, spatial_budget, mf);
      if (half_xy.x < 0.0f || half_xy.y < 0.0f) continue;
      bool full_xy = !isfinite(half_xy.x) || !isfinite(half_xy.y);

      float3 h = eval_prt_h(h_coeff, tube_id, h_terms, tau_t);
      float depth = max(h.z, mf.eps);
      float2 center = h.xy / depth;
      float hx = full_xy ? float(mi.width) : half_xy.x;
      float hy = full_xy ? float(mi.height) : half_xy.y;
      x0 = min(x0, max(0, int(floor(center.x - hx - 0.5f))));
      x1 = max(x1, min(mi.width - 1, int(ceil(center.x + hx - 0.5f))));
      y0 = min(y0, max(0, int(floor(center.y - hy - 0.5f))));
      y1 = max(y1, min(mi.height - 1, int(ceil(center.y + hy - 0.5f))));
    }
    if (x0 > x1 || y0 > y1) continue;

    uint tx0 = uint(x0 / mi.tile_x);
    uint tx1 = uint(x1 / mi.tile_x);
    uint ty0 = uint(y0 / mi.tile_y);
    uint ty1 = uint(y1 / mi.tile_y);
    for (uint ty = ty0; ty <= ty1; ++ty) {
      for (uint tx = tx0; tx <= tx1; ++tx) {
        uint tile_id = encode_tile(tx, ty, tz, mi);
        uint slot = atomic_fetch_add_explicit(tile_counts + tile_id, 1u, memory_order_relaxed);
        if (slot < STAR_TILE_CAPACITY) {
          uint idx = tile_id * STAR_TILE_CAPACITY + slot;
          tile_tube_ids[idx] = tube_id;
          float t = tile_center(tx, ty, tz, mi).z;
          tile_depths[idx] = max(eval_prt_h(h_coeff, tube_id, h_terms, t - center_t[tube_id]).z, mf.eps);
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

kernel void render_projective_rational_tiles(
    const device float* h_coeff [[buffer(0)]],
    const device float* lambda_uv [[buffer(1)]],
    const device float* lambda_t [[buffer(2)]],
    const device float* center_t [[buffer(3)]],
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
  uint h_terms = uint(mi.reserved0);

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

  if (local_tid == 0u && count > 1u) {
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

  float t = frame_time(f, mi);
  float2 pixel = float2(float(x) + 0.5f, float(y) + 0.5f);
  float3 accum = float3(0.0f);
  float T = 1.0f;
  if (uint(STAR_TILE_T) == 1u) {
    for (uint i = 0u; i < count; ++i) {
      composite_prt_tube(
          local_ids[i],
          pixel,
          t,
          h_coeff,
          lambda_uv,
          lambda_t,
          center_t,
          opacity,
          color,
          h_terms,
          mf,
          accum,
          T);
      if (T <= mf.transmittance_threshold) break;
    }
  } else {
    float last_depth = -INFINITY;
    uint last_id = 0xFFFFFFFFu;
    for (uint rank = 0u; rank < count; ++rank) {
      float selected_depth;
      uint tube_id = select_prt_sample_order_id(
          local_ids,
          count,
          h_coeff,
          center_t,
          h_terms,
          t,
          last_depth,
          last_id,
          mf,
          selected_depth);
      if (tube_id == 0xFFFFFFFFu) break;
      composite_prt_tube(
          tube_id,
          pixel,
          t,
          h_coeff,
          lambda_uv,
          lambda_t,
          center_t,
          opacity,
          color,
          h_terms,
          mf,
          accum,
          T);
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

kernel void render_projective_rational_direct(
    const device float* h_coeff [[buffer(0)]],
    const device float* lambda_uv [[buffer(1)]],
    const device float* lambda_t [[buffer(2)]],
    const device float* center_t [[buffer(3)]],
    const device float* opacity [[buffer(4)]],
    const device float* color [[buffer(5)]],
    constant MetaI32& mi [[buffer(6)]],
    constant MetaF32& mf [[buffer(7)]],
    device float* out_rgb [[buffer(8)]],
    uint pixel_id [[thread_position_in_grid]]) {
  uint pixels_per_frame = uint(mi.height * mi.width);
  uint total = pixels_per_frame * uint(mi.frames);
  if (pixel_id >= total) return;

  uint frame = pixel_id / pixels_per_frame;
  uint rem = pixel_id - frame * pixels_per_frame;
  uint y = rem / uint(mi.width);
  uint x = rem - y * uint(mi.width);
  float t = frame_time(frame, mi);
  float px = float(x) + 0.5f;
  float py = float(y) + 0.5f;
  uint h_terms = uint(mi.reserved0);

  float3 accum = float3(0.0f);
  float T = 1.0f;
  float last_depth = -INFINITY;
  uint last_id = 0xFFFFFFFFu;

  for (uint rank = 0u; rank < uint(mi.tube_count); ++rank) {
    float best_depth = INFINITY;
    uint best_id = 0xFFFFFFFFu;
    float best_alpha = 0.0f;

    for (uint tube_id = 0u; tube_id < uint(mi.tube_count); ++tube_id) {
      float tau = t - center_t[tube_id];
      float3 h = eval_prt_h(h_coeff, tube_id, h_terms, tau);
      float depth = max(h.z, mf.eps);
      bool after_last = (last_id == 0xFFFFFFFFu) || (depth > last_depth) ||
                        (depth == last_depth && tube_id > last_id);
      if (!after_last) continue;

      float inv_depth = 1.0f / depth;
      float u = h.x * inv_depth;
      float v = h.y * inv_depth;
      uint qbase = tube_id * 3u;
      float du = px - u;
      float dv = py - v;
      float spatial = lambda_uv[qbase + 0u] * du * du +
                      2.0f * lambda_uv[qbase + 1u] * du * dv +
                      lambda_uv[qbase + 2u] * dv * dv;
      float temporal = lambda_t[tube_id] * tau * tau;
      float alpha = clamp(opacity[tube_id] * exp(-0.5f * (spatial + temporal)), 0.0f, mf.max_alpha);
      if (!(alpha >= mf.alpha_threshold)) continue;

      bool better = (depth < best_depth) || (depth == best_depth && tube_id < best_id);
      if (better) {
        best_depth = depth;
        best_id = tube_id;
        best_alpha = alpha;
      }
    }

    if (best_id == 0xFFFFFFFFu) break;
    uint cbase = best_id * 3u;
    float3 c = float3(color[cbase + 0u], color[cbase + 1u], color[cbase + 2u]);
    accum += T * best_alpha * c;
    T *= (1.0f - best_alpha);
    last_depth = best_depth;
    last_id = best_id;
    if (T <= mf.transmittance_threshold) break;
  }

  uint out_base = pixel_id * 3u;
  out_rgb[out_base + 0u] = accum.x + T * mf.bg_r;
  out_rgb[out_base + 1u] = accum.y + T * mf.bg_g;
  out_rgb[out_base + 2u] = accum.z + T * mf.bg_b;
}

kernel void projective_rational_direct_serial_backward(
    const device float* h_coeff [[buffer(0)]],
    const device float* lambda_uv [[buffer(1)]],
    const device float* lambda_t [[buffer(2)]],
    const device float* center_t [[buffer(3)]],
    const device float* opacity [[buffer(4)]],
    const device float* color [[buffer(5)]],
    const device float* grad_image [[buffer(6)]],
    constant MetaI32& mi [[buffer(7)]],
    constant MetaF32& mf [[buffer(8)]],
    device float* grad_h_coeff [[buffer(9)]],
    device float* grad_lambda_uv [[buffer(10)]],
    device float* grad_lambda_t [[buffer(11)]],
    device float* grad_center_t [[buffer(12)]],
    device float* grad_opacity [[buffer(13)]],
    device float* grad_color [[buffer(14)]],
    uint target_id [[thread_position_in_grid]]) {
  uint tube_count = uint(mi.tube_count);
  if (target_id >= tube_count) return;
  uint h_terms = uint(mi.reserved0);

  for (uint p = 0u; p < h_terms; ++p) {
    uint h_base = (target_id * h_terms + p) * 3u;
    grad_h_coeff[h_base + 0u] = 0.0f;
    grad_h_coeff[h_base + 1u] = 0.0f;
    grad_h_coeff[h_base + 2u] = 0.0f;
  }
  uint q_base = target_id * 3u;
  grad_lambda_uv[q_base + 0u] = 0.0f;
  grad_lambda_uv[q_base + 1u] = 0.0f;
  grad_lambda_uv[q_base + 2u] = 0.0f;
  grad_lambda_t[target_id] = 0.0f;
  grad_center_t[target_id] = 0.0f;
  grad_opacity[target_id] = 0.0f;
  uint c_base = target_id * 3u;
  grad_color[c_base + 0u] = 0.0f;
  grad_color[c_base + 1u] = 0.0f;
  grad_color[c_base + 2u] = 0.0f;

  float3 h_coeff_sum[8];
  for (uint p = 0u; p < 8u; ++p) {
    h_coeff_sum[p] = float3(0.0f);
  }
  float3 lambda_uv_sum = float3(0.0f);
  float lambda_t_sum = 0.0f;
  float center_t_sum = 0.0f;
  float opacity_sum = 0.0f;
  float3 color_sum = float3(0.0f);

  for (uint f = 0u; f < uint(mi.frames); ++f) {
    float t = frame_time(f, mi);
    for (uint y = 0u; y < uint(mi.height); ++y) {
      for (uint x = 0u; x < uint(mi.width); ++x) {
        float2 pixel = float2(float(x) + 0.5f, float(y) + 0.5f);
        float prefix_T = 1.0f;
        bool target_processed = false;
        bool target_differentiable = false;
        float target_alpha = 0.0f;
        float target_t = 0.0f;
        float target_depth = -INFINITY;
        float last_depth = -INFINITY;
        uint last_id = 0xFFFFFFFFu;

        for (uint rank = 0u; rank < tube_count; ++rank) {
          float selected_depth;
          float selected_alpha;
          float selected_alpha_raw;
          uint tube_id = select_prt_direct_order_id(
              h_coeff,
              lambda_uv,
              lambda_t,
              center_t,
              opacity,
              h_terms,
              tube_count,
              pixel,
              t,
              last_depth,
              last_id,
              mf,
              selected_depth,
              selected_alpha,
              selected_alpha_raw);
          if (tube_id == 0xFFFFFFFFu) break;
          if (tube_id == target_id) {
            target_t = prefix_T;
            target_alpha = selected_alpha;
            target_depth = selected_depth;
            target_differentiable = selected_alpha_raw < mf.max_alpha;
            prefix_T *= (1.0f - selected_alpha);
            target_processed = true;
            last_depth = selected_depth;
            last_id = tube_id;
            break;
          }
          prefix_T *= (1.0f - selected_alpha);
          last_depth = selected_depth;
          last_id = tube_id;
          if (prefix_T <= mf.transmittance_threshold) break;
        }
        if (!target_processed) continue;

        float suffix_T = 1.0f;
        float3 suffix_accum = float3(0.0f);
        if (prefix_T > mf.transmittance_threshold) {
          last_depth = target_depth;
          last_id = target_id;
          for (uint rank = 0u; rank < tube_count; ++rank) {
            float selected_depth;
            float selected_alpha;
            float selected_alpha_raw;
            uint tube_id = select_prt_direct_order_id(
                h_coeff,
                lambda_uv,
                lambda_t,
                center_t,
                opacity,
                h_terms,
                tube_count,
                pixel,
                t,
                last_depth,
                last_id,
                mf,
                selected_depth,
                selected_alpha,
                selected_alpha_raw);
            if (tube_id == 0xFFFFFFFFu) break;
            suffix_accum += suffix_T * selected_alpha * load3(color, tube_id);
            suffix_T *= (1.0f - selected_alpha);
            last_depth = selected_depth;
            last_id = tube_id;
            if (prefix_T * suffix_T <= mf.transmittance_threshold) break;
          }
        }

        uint image_base = ((f * uint(mi.height) + y) * uint(mi.width) + x) * 3u;
        float3 grad_rgb = float3(grad_image[image_base + 0u], grad_image[image_base + 1u], grad_image[image_base + 2u]);
        float3 suffix_color = suffix_accum + suffix_T * float3(mf.bg_r, mf.bg_g, mf.bg_b);
        float3 target_color = load3(color, target_id);
        float d_alpha = dot(grad_rgb, target_t * target_color) - dot(grad_rgb, suffix_color) * target_t;
        color_sum += grad_rgb * (target_t * target_alpha);
        if (!target_differentiable) continue;

        float tau = t - center_t[target_id];
        float3 h = eval_prt_h(h_coeff, target_id, h_terms, tau);
        float depth = max(h.z, mf.eps);
        float inv_depth = 1.0f / depth;
        float2 center = h.xy * inv_depth;
        float2 d = pixel - center;
        uint target_q_base = target_id * 3u;
        float luu = lambda_uv[target_q_base + 0u];
        float luv = lambda_uv[target_q_base + 1u];
        float lvv = lambda_uv[target_q_base + 2u];
        float spatial = luu * d.x * d.x + 2.0f * luv * d.x * d.y + lvv * d.y * d.y;
        float temporal = lambda_t[target_id] * tau * tau;
        float qv = spatial + temporal;
        float exp_term = exp(-0.5f * qv);
        float grad_qv = -0.5f * target_alpha * d_alpha;
        float2 qd = float2(luu * d.x + luv * d.y, luv * d.x + lvv * d.y);
        float2 grad_center = -2.0f * grad_qv * qd;
        float3 grad_h = float3(
            grad_center.x * inv_depth,
            grad_center.y * inv_depth,
            h.z > mf.eps ? -(grad_center.x * h.x + grad_center.y * h.y) * inv_depth * inv_depth : 0.0f);

        float power = 1.0f;
        for (uint p = 0u; p < h_terms && p < 8u; ++p) {
          h_coeff_sum[p] += grad_h * power;
          power *= tau;
        }
        float3 dh_dtau = eval_prt_h_dtau(h_coeff, target_id, h_terms, tau);
        float grad_tau = dot(grad_h, dh_dtau) + grad_qv * 2.0f * lambda_t[target_id] * tau;
        center_t_sum += -grad_tau;
        lambda_t_sum += grad_qv * tau * tau;
        lambda_uv_sum += float3(grad_qv * d.x * d.x, grad_qv * 2.0f * d.x * d.y, grad_qv * d.y * d.y);
        opacity_sum += d_alpha * exp_term;
      }
    }
  }

  for (uint p = 0u; p < h_terms && p < 8u; ++p) {
    uint h_base = (target_id * h_terms + p) * 3u;
    grad_h_coeff[h_base + 0u] = h_coeff_sum[p].x;
    grad_h_coeff[h_base + 1u] = h_coeff_sum[p].y;
    grad_h_coeff[h_base + 2u] = h_coeff_sum[p].z;
  }
  grad_lambda_uv[q_base + 0u] = lambda_uv_sum.x;
  grad_lambda_uv[q_base + 1u] = lambda_uv_sum.y;
  grad_lambda_uv[q_base + 2u] = lambda_uv_sum.z;
  grad_lambda_t[target_id] = lambda_t_sum;
  grad_center_t[target_id] = center_t_sum;
  grad_opacity[target_id] = opacity_sum;
  grad_color[c_base + 0u] = color_sum.x;
  grad_color[c_base + 1u] = color_sum.y;
  grad_color[c_base + 2u] = color_sum.z;
}

kernel void projective_rational_tile_pair_atomic_backward(
    const device float* h_coeff [[buffer(0)]],
    const device float* lambda_uv [[buffer(1)]],
    const device float* lambda_t [[buffer(2)]],
    const device float* center_t [[buffer(3)]],
    const device float* opacity [[buffer(4)]],
    const device float* color [[buffer(5)]],
    const device float* grad_image [[buffer(6)]],
    constant MetaI32& mi [[buffer(7)]],
    constant MetaF32& mf [[buffer(8)]],
    const device atomic_uint* tile_counts [[buffer(9)]],
    const device uint* tile_tube_ids [[buffer(10)]],
    const device float* tile_depths [[buffer(11)]],
    device atomic_uint* tile_unstable [[buffer(12)]],
    device atomic_float* grad_h_coeff [[buffer(13)]],
    device atomic_float* grad_lambda_uv [[buffer(14)]],
    device atomic_float* grad_lambda_t [[buffer(15)]],
    device atomic_float* grad_center_t [[buffer(16)]],
    device atomic_float* grad_opacity [[buffer(17)]],
    device atomic_float* grad_color [[buffer(18)]],
    uint gid [[thread_position_in_grid]]) {
  uint tile_id = gid / STAR_TILE_CAPACITY;
  uint slot = gid - tile_id * STAR_TILE_CAPACITY;
  if (tile_id >= uint(mi.tile_count)) return;
  uint h_terms = uint(mi.reserved0);

  uint raw_count = atomic_load_explicit(tile_counts + tile_id, memory_order_relaxed);
  uint count = min(raw_count, STAR_TILE_CAPACITY);
  if (slot >= count) return;

  uint tx, ty, tz;
  decode_tile(tile_id, mi, tx, ty, tz);

  uint local_ids[STAR_TILE_CAPACITY];
  float local_depths[STAR_TILE_CAPACITY];
  for (uint i = 0u; i < count; ++i) {
    uint idx = tile_id * STAR_TILE_CAPACITY + i;
    local_ids[i] = tile_tube_ids[idx];
    local_depths[i] = tile_depths[idx];
  }
  sort_by_depth_thread(local_ids, local_depths, count);
  uint target_id = local_ids[slot];
  if (slot == 0u && count > 1u) {
    atomic_store_explicit(tile_unstable + tile_id, 1u, memory_order_relaxed);
  }

  float3 h_coeff_sum[8];
  for (uint p = 0u; p < 8u; ++p) {
    h_coeff_sum[p] = float3(0.0f);
  }
  float3 lambda_uv_sum = float3(0.0f);
  float lambda_t_sum = 0.0f;
  float center_t_sum = 0.0f;
  float opacity_sum = 0.0f;
  float3 color_sum = float3(0.0f);

  uint ordered_ids[STAR_TILE_CAPACITY];
  float t_before[STAR_TILE_CAPACITY];
  float alpha_values[STAR_TILE_CAPACITY];
  bool processed[STAR_TILE_CAPACITY];
  bool differentiable_alpha[STAR_TILE_CAPACITY];

  for (uint lt = 0u; lt < STAR_TILE_T; ++lt) {
    uint f = tz * STAR_TILE_T + lt;
    if (f >= uint(mi.frames)) continue;
    float t = frame_time(f, mi);
    for (uint ly = 0u; ly < STAR_TILE_Y; ++ly) {
      uint y = ty * STAR_TILE_Y + ly;
      if (y >= uint(mi.height)) continue;
      for (uint lx = 0u; lx < STAR_TILE_X; ++lx) {
        uint x = tx * STAR_TILE_X + lx;
        if (x >= uint(mi.width)) continue;
        float2 pixel = float2(float(x) + 0.5f, float(y) + 0.5f);

        uint ordered_count = 0u;
        float last_depth = -INFINITY;
        uint last_id = 0xFFFFFFFFu;
        for (uint rank = 0u; rank < count; ++rank) {
          float selected_depth;
          uint tube_id = select_prt_sample_order_id_thread(
              local_ids,
              count,
              h_coeff,
              center_t,
              h_terms,
              t,
              last_depth,
              last_id,
              mf,
              selected_depth);
          if (tube_id == 0xFFFFFFFFu) break;
          ordered_ids[ordered_count] = tube_id;
          ordered_count += 1u;
          last_depth = selected_depth;
          last_id = tube_id;
        }

        for (uint i = 0u; i < ordered_count; ++i) {
          t_before[i] = 0.0f;
          alpha_values[i] = 0.0f;
          processed[i] = false;
          differentiable_alpha[i] = false;
        }

        float T = 1.0f;
        for (uint i = 0u; i < ordered_count; ++i) {
          uint tube_id = ordered_ids[i];
          float tau = t - center_t[tube_id];
          float3 h = eval_prt_h(h_coeff, tube_id, h_terms, tau);
          float depth = max(h.z, mf.eps);
          float2 center = h.xy / depth;
          float2 d = pixel - center;
          uint qbase = tube_id * 3u;
          float spatial = lambda_uv[qbase + 0u] * d.x * d.x +
                          2.0f * lambda_uv[qbase + 1u] * d.x * d.y +
                          lambda_uv[qbase + 2u] * d.y * d.y;
          float temporal = lambda_t[tube_id] * tau * tau;
          float qv = spatial + temporal;
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
          if (tube_id != target_id) continue;

          color_sum += d_color;
          if (differentiable_alpha[i]) {
            float tau = t - center_t[target_id];
            float3 h = eval_prt_h(h_coeff, target_id, h_terms, tau);
            float depth = max(h.z, mf.eps);
            float inv_depth = 1.0f / depth;
            float2 center = h.xy * inv_depth;
            float2 d = pixel - center;
            uint qbase = target_id * 3u;
            float luu = lambda_uv[qbase + 0u];
            float luv = lambda_uv[qbase + 1u];
            float lvv = lambda_uv[qbase + 2u];
            float spatial = luu * d.x * d.x + 2.0f * luv * d.x * d.y + lvv * d.y * d.y;
            float temporal = lambda_t[target_id] * tau * tau;
            float qv = spatial + temporal;
            float exp_term = exp(-0.5f * qv);
            float grad_qv = -0.5f * alpha * d_alpha;
            float2 qd = float2(luu * d.x + luv * d.y, luv * d.x + lvv * d.y);
            float2 grad_center = -2.0f * grad_qv * qd;
            float3 grad_h = float3(
                grad_center.x * inv_depth,
                grad_center.y * inv_depth,
                h.z > mf.eps ? -(grad_center.x * h.x + grad_center.y * h.y) * inv_depth * inv_depth : 0.0f);

            float power = 1.0f;
            for (uint p = 0u; p < h_terms && p < 8u; ++p) {
              h_coeff_sum[p] += grad_h * power;
              power *= tau;
            }
            float3 dh_dtau = eval_prt_h_dtau(h_coeff, target_id, h_terms, tau);
            float grad_tau = dot(grad_h, dh_dtau) + grad_qv * 2.0f * lambda_t[target_id] * tau;
            center_t_sum += -grad_tau;
            lambda_t_sum += grad_qv * tau * tau;
            lambda_uv_sum += float3(grad_qv * d.x * d.x, grad_qv * 2.0f * d.x * d.y, grad_qv * d.y * d.y);
            opacity_sum += d_alpha * exp_term;
          }
          break;
        }
      }
    }
  }

  bool has_gradient =
      lambda_uv_sum.x != 0.0f || lambda_uv_sum.y != 0.0f || lambda_uv_sum.z != 0.0f ||
      lambda_t_sum != 0.0f || center_t_sum != 0.0f || opacity_sum != 0.0f ||
      color_sum.x != 0.0f || color_sum.y != 0.0f || color_sum.z != 0.0f;
  for (uint p = 0u; p < h_terms && p < 8u; ++p) {
    has_gradient = has_gradient ||
        h_coeff_sum[p].x != 0.0f || h_coeff_sum[p].y != 0.0f || h_coeff_sum[p].z != 0.0f;
  }
  if (!has_gradient) return;

  for (uint p = 0u; p < h_terms && p < 8u; ++p) {
    uint h_base = (target_id * h_terms + p) * 3u;
    atomic_add3(grad_h_coeff, h_base, h_coeff_sum[p]);
  }
  uint q_base = target_id * 3u;
  atomic_fetch_add_explicit(&grad_lambda_uv[q_base + 0u], lambda_uv_sum.x, memory_order_relaxed);
  atomic_fetch_add_explicit(&grad_lambda_uv[q_base + 1u], lambda_uv_sum.y, memory_order_relaxed);
  atomic_fetch_add_explicit(&grad_lambda_uv[q_base + 2u], lambda_uv_sum.z, memory_order_relaxed);
  atomic_fetch_add_explicit(&grad_lambda_t[target_id], lambda_t_sum, memory_order_relaxed);
  atomic_fetch_add_explicit(&grad_center_t[target_id], center_t_sum, memory_order_relaxed);
  atomic_fetch_add_explicit(&grad_opacity[target_id], opacity_sum, memory_order_relaxed);
  uint color_base = target_id * 3u;
  atomic_add3(grad_color, color_base, color_sum);
}

kernel void projective_rational_tile_pixel_atomic_backward(
    const device float* h_coeff [[buffer(0)]],
    const device float* lambda_uv [[buffer(1)]],
    const device float* lambda_t [[buffer(2)]],
    const device float* center_t [[buffer(3)]],
    const device float* opacity [[buffer(4)]],
    const device float* color [[buffer(5)]],
    const device float* grad_image [[buffer(6)]],
    constant MetaI32& mi [[buffer(7)]],
    constant MetaF32& mf [[buffer(8)]],
    const device atomic_uint* tile_counts [[buffer(9)]],
    const device uint* tile_tube_ids [[buffer(10)]],
    const device float* tile_depths [[buffer(11)]],
    device atomic_uint* tile_unstable [[buffer(12)]],
    device atomic_float* grad_h_coeff [[buffer(13)]],
    device atomic_float* grad_lambda_uv [[buffer(14)]],
    device atomic_float* grad_lambda_t [[buffer(15)]],
    device atomic_float* grad_center_t [[buffer(16)]],
    device atomic_float* grad_opacity [[buffer(17)]],
    device atomic_float* grad_color [[buffer(18)]],
    uint gid [[thread_position_in_grid]]) {
  uint pixels_per_tile = uint(STAR_TILE_X * STAR_TILE_Y * STAR_TILE_T);
  uint tile_id = gid / pixels_per_tile;
  uint local_pixel = gid - tile_id * pixels_per_tile;
  if (tile_id >= uint(mi.tile_count)) return;
  uint h_terms = uint(mi.reserved0);

  uint raw_count = atomic_load_explicit(tile_counts + tile_id, memory_order_relaxed);
  uint count = min(raw_count, STAR_TILE_CAPACITY);
  if (count == 0u) return;

  uint tx, ty, tz;
  decode_tile(tile_id, mi, tx, ty, tz);
  uint local_xy = local_pixel % uint(STAR_TILE_X * STAR_TILE_Y);
  uint lt = local_pixel / uint(STAR_TILE_X * STAR_TILE_Y);
  uint lx = local_xy % uint(STAR_TILE_X);
  uint ly = local_xy / uint(STAR_TILE_X);
  uint f = tz * STAR_TILE_T + lt;
  uint x = tx * STAR_TILE_X + lx;
  uint y = ty * STAR_TILE_Y + ly;
  if (f >= uint(mi.frames) || x >= uint(mi.width) || y >= uint(mi.height)) return;

  uint local_ids[STAR_TILE_CAPACITY];
  float local_depths[STAR_TILE_CAPACITY];
  for (uint i = 0u; i < count; ++i) {
    uint idx = tile_id * STAR_TILE_CAPACITY + i;
    local_ids[i] = tile_tube_ids[idx];
    local_depths[i] = tile_depths[idx];
  }
  sort_by_depth_thread(local_ids, local_depths, count);
  if (count > 1u) {
    atomic_store_explicit(tile_unstable + tile_id, 1u, memory_order_relaxed);
  }

  float t = frame_time(f, mi);
  float2 pixel = float2(float(x) + 0.5f, float(y) + 0.5f);
  uint ordered_ids[STAR_TILE_CAPACITY];
  float t_before[STAR_TILE_CAPACITY];
  float alpha_values[STAR_TILE_CAPACITY];
  bool processed[STAR_TILE_CAPACITY];
  bool differentiable_alpha[STAR_TILE_CAPACITY];

  uint ordered_count = 0u;
  if (uint(STAR_TILE_T) == 1u) {
    ordered_count = count;
    for (uint i = 0u; i < count; ++i) {
      ordered_ids[i] = local_ids[i];
    }
  } else {
    float last_depth = -INFINITY;
    uint last_id = 0xFFFFFFFFu;
    for (uint rank = 0u; rank < count; ++rank) {
      float selected_depth;
      uint tube_id = select_prt_sample_order_id_thread(
          local_ids,
          count,
          h_coeff,
          center_t,
          h_terms,
          t,
          last_depth,
          last_id,
          mf,
          selected_depth);
      if (tube_id == 0xFFFFFFFFu) break;
      ordered_ids[ordered_count] = tube_id;
      ordered_count += 1u;
      last_depth = selected_depth;
      last_id = tube_id;
    }
  }

  for (uint i = 0u; i < ordered_count; ++i) {
    t_before[i] = 0.0f;
    alpha_values[i] = 0.0f;
    processed[i] = false;
    differentiable_alpha[i] = false;
  }

  float T = 1.0f;
  for (uint i = 0u; i < ordered_count; ++i) {
    uint tube_id = ordered_ids[i];
    float tau = t - center_t[tube_id];
    float3 h = eval_prt_h(h_coeff, tube_id, h_terms, tau);
    float depth = max(h.z, mf.eps);
    float2 center = h.xy / depth;
    float2 d = pixel - center;
    uint qbase = tube_id * 3u;
    float spatial = lambda_uv[qbase + 0u] * d.x * d.x +
                    2.0f * lambda_uv[qbase + 1u] * d.x * d.y +
                    lambda_uv[qbase + 2u] * d.y * d.y;
    float temporal = lambda_t[tube_id] * tau * tau;
    float qv = spatial + temporal;
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

    uint color_base = tube_id * 3u;
    atomic_add3(grad_color, color_base, d_color);
    if (!differentiable_alpha[i]) continue;

    float tau = t - center_t[tube_id];
    float3 h = eval_prt_h(h_coeff, tube_id, h_terms, tau);
    float depth = max(h.z, mf.eps);
    float inv_depth = 1.0f / depth;
    float2 center = h.xy * inv_depth;
    float2 d = pixel - center;
    uint qbase = tube_id * 3u;
    float luu = lambda_uv[qbase + 0u];
    float luv = lambda_uv[qbase + 1u];
    float lvv = lambda_uv[qbase + 2u];
    float spatial = luu * d.x * d.x + 2.0f * luv * d.x * d.y + lvv * d.y * d.y;
    float temporal = lambda_t[tube_id] * tau * tau;
    float qv = spatial + temporal;
    float exp_term = exp(-0.5f * qv);
    float grad_qv = -0.5f * alpha * d_alpha;
    float2 qd = float2(luu * d.x + luv * d.y, luv * d.x + lvv * d.y);
    float2 grad_center = -2.0f * grad_qv * qd;
    float3 grad_h = float3(
        grad_center.x * inv_depth,
        grad_center.y * inv_depth,
        h.z > mf.eps ? -(grad_center.x * h.x + grad_center.y * h.y) * inv_depth * inv_depth : 0.0f);

    float power = 1.0f;
    for (uint p = 0u; p < h_terms && p < 8u; ++p) {
      uint h_base = (tube_id * h_terms + p) * 3u;
      atomic_add3(grad_h_coeff, h_base, grad_h * power);
      power *= tau;
    }
    float3 dh_dtau = eval_prt_h_dtau(h_coeff, tube_id, h_terms, tau);
    float grad_tau = dot(grad_h, dh_dtau) + grad_qv * 2.0f * lambda_t[tube_id] * tau;
    atomic_fetch_add_explicit(&grad_center_t[tube_id], -grad_tau, memory_order_relaxed);
    atomic_fetch_add_explicit(&grad_lambda_t[tube_id], grad_qv * tau * tau, memory_order_relaxed);
    atomic_fetch_add_explicit(&grad_lambda_uv[qbase + 0u], grad_qv * d.x * d.x, memory_order_relaxed);
    atomic_fetch_add_explicit(&grad_lambda_uv[qbase + 1u], grad_qv * 2.0f * d.x * d.y, memory_order_relaxed);
    atomic_fetch_add_explicit(&grad_lambda_uv[qbase + 2u], grad_qv * d.y * d.y, memory_order_relaxed);
    atomic_fetch_add_explicit(&grad_opacity[tube_id], d_alpha * exp_term, memory_order_relaxed);
  }
}

kernel void projective_rational_tile_pixel_fused_mse_backward(
    const device float* h_coeff [[buffer(0)]],
    const device float* lambda_uv [[buffer(1)]],
    const device float* lambda_t [[buffer(2)]],
    const device float* center_t [[buffer(3)]],
    const device float* opacity [[buffer(4)]],
    const device float* color [[buffer(5)]],
    const device float* target_image [[buffer(6)]],
    constant MetaI32& mi [[buffer(7)]],
    constant MetaF32& mf [[buffer(8)]],
    const device atomic_uint* tile_counts [[buffer(9)]],
    const device uint* tile_tube_ids [[buffer(10)]],
    const device float* tile_depths [[buffer(11)]],
    device atomic_uint* tile_unstable [[buffer(12)]],
    device atomic_float* grad_h_coeff [[buffer(13)]],
    device atomic_float* grad_lambda_uv [[buffer(14)]],
    device atomic_float* grad_lambda_t [[buffer(15)]],
    device atomic_float* grad_center_t [[buffer(16)]],
    device atomic_float* grad_opacity [[buffer(17)]],
    device atomic_float* grad_color [[buffer(18)]],
    device atomic_float* loss_sum [[buffer(19)]],
    uint gid [[thread_position_in_grid]]) {
  uint pixels_per_tile = uint(STAR_TILE_X * STAR_TILE_Y * STAR_TILE_T);
  uint tile_id = gid / pixels_per_tile;
  uint local_pixel = gid - tile_id * pixels_per_tile;
  if (tile_id >= uint(mi.tile_count)) return;
  uint h_terms = uint(mi.reserved0);

  uint raw_count = atomic_load_explicit(tile_counts + tile_id, memory_order_relaxed);
  uint count = min(raw_count, STAR_TILE_CAPACITY);
  if (count == 0u) return;

  uint tx, ty, tz;
  decode_tile(tile_id, mi, tx, ty, tz);
  uint local_xy = local_pixel % uint(STAR_TILE_X * STAR_TILE_Y);
  uint lt = local_pixel / uint(STAR_TILE_X * STAR_TILE_Y);
  uint lx = local_xy % uint(STAR_TILE_X);
  uint ly = local_xy / uint(STAR_TILE_X);
  uint f = tz * STAR_TILE_T + lt;
  uint x = tx * STAR_TILE_X + lx;
  uint y = ty * STAR_TILE_Y + ly;
  if (f >= uint(mi.frames) || x >= uint(mi.width) || y >= uint(mi.height)) return;

  uint local_ids[STAR_TILE_CAPACITY];
  float local_depths[STAR_TILE_CAPACITY];
  for (uint i = 0u; i < count; ++i) {
    uint idx = tile_id * STAR_TILE_CAPACITY + i;
    local_ids[i] = tile_tube_ids[idx];
    local_depths[i] = tile_depths[idx];
  }
  sort_by_depth_thread(local_ids, local_depths, count);
  if (count > 1u) {
    atomic_store_explicit(tile_unstable + tile_id, 1u, memory_order_relaxed);
  }

  float t = frame_time(f, mi);
  float2 pixel = float2(float(x) + 0.5f, float(y) + 0.5f);
  uint ordered_ids[STAR_TILE_CAPACITY];
  float t_before[STAR_TILE_CAPACITY];
  float alpha_values[STAR_TILE_CAPACITY];
  bool processed[STAR_TILE_CAPACITY];
  bool differentiable_alpha[STAR_TILE_CAPACITY];

  uint ordered_count = 0u;
  if (uint(STAR_TILE_T) == 1u) {
    ordered_count = count;
    for (uint i = 0u; i < count; ++i) {
      ordered_ids[i] = local_ids[i];
    }
  } else {
    float last_depth = -INFINITY;
    uint last_id = 0xFFFFFFFFu;
    for (uint rank = 0u; rank < count; ++rank) {
      float selected_depth;
      uint tube_id = select_prt_sample_order_id_thread(
          local_ids,
          count,
          h_coeff,
          center_t,
          h_terms,
          t,
          last_depth,
          last_id,
          mf,
          selected_depth);
      if (tube_id == 0xFFFFFFFFu) break;
      ordered_ids[ordered_count] = tube_id;
      ordered_count += 1u;
      last_depth = selected_depth;
      last_id = tube_id;
    }
  }

  for (uint i = 0u; i < ordered_count; ++i) {
    t_before[i] = 0.0f;
    alpha_values[i] = 0.0f;
    processed[i] = false;
    differentiable_alpha[i] = false;
  }

  float3 accum = float3(0.0f);
  float T = 1.0f;
  for (uint i = 0u; i < ordered_count; ++i) {
    uint tube_id = ordered_ids[i];
    float tau = t - center_t[tube_id];
    float3 h = eval_prt_h(h_coeff, tube_id, h_terms, tau);
    float depth = max(h.z, mf.eps);
    float2 center = h.xy / depth;
    float2 d = pixel - center;
    uint qbase = tube_id * 3u;
    float spatial = lambda_uv[qbase + 0u] * d.x * d.x +
                    2.0f * lambda_uv[qbase + 1u] * d.x * d.y +
                    lambda_uv[qbase + 2u] * d.y * d.y;
    float temporal = lambda_t[tube_id] * tau * tau;
    float qv = spatial + temporal;
    if (!isfinite(qv)) continue;
    float alpha_raw = opacity[tube_id] * exp(-0.5f * qv);
    float alpha = min(mf.max_alpha, alpha_raw);
    if (!(alpha >= mf.alpha_threshold)) continue;
    t_before[i] = T;
    alpha_values[i] = alpha;
    processed[i] = true;
    differentiable_alpha[i] = alpha_raw < mf.max_alpha;
    accum += T * alpha * load3(color, tube_id);
    T *= (1.0f - alpha);
    if (T <= mf.transmittance_threshold) break;
  }

  uint image_base = ((f * uint(mi.height) + y) * uint(mi.width) + x) * 3u;
  float3 rgb = accum + T * float3(mf.bg_r, mf.bg_g, mf.bg_b);
  float3 target_rgb = float3(target_image[image_base + 0u], target_image[image_base + 1u], target_image[image_base + 2u]);
  float3 diff = rgb - target_rgb;
  atomic_fetch_add_explicit(loss_sum, dot(diff, diff), memory_order_relaxed);
  float inv_numel = 1.0f / float(uint(mi.frames) * uint(mi.height) * uint(mi.width) * 3u);
  float3 grad_rgb = 2.0f * diff * inv_numel;

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

    uint color_base = tube_id * 3u;
    atomic_add3(grad_color, color_base, d_color);
    if (!differentiable_alpha[i]) continue;

    float tau = t - center_t[tube_id];
    float3 h = eval_prt_h(h_coeff, tube_id, h_terms, tau);
    float depth = max(h.z, mf.eps);
    float inv_depth = 1.0f / depth;
    float2 center = h.xy * inv_depth;
    float2 d = pixel - center;
    uint qbase = tube_id * 3u;
    float luu = lambda_uv[qbase + 0u];
    float luv = lambda_uv[qbase + 1u];
    float lvv = lambda_uv[qbase + 2u];
    float spatial = luu * d.x * d.x + 2.0f * luv * d.x * d.y + lvv * d.y * d.y;
    float temporal = lambda_t[tube_id] * tau * tau;
    float qv = spatial + temporal;
    float exp_term = exp(-0.5f * qv);
    float grad_qv = -0.5f * alpha * d_alpha;
    float2 qd = float2(luu * d.x + luv * d.y, luv * d.x + lvv * d.y);
    float2 grad_center = -2.0f * grad_qv * qd;
    float3 grad_h = float3(
        grad_center.x * inv_depth,
        grad_center.y * inv_depth,
        h.z > mf.eps ? -(grad_center.x * h.x + grad_center.y * h.y) * inv_depth * inv_depth : 0.0f);

    float power = 1.0f;
    for (uint p = 0u; p < h_terms && p < 8u; ++p) {
      uint h_base = (tube_id * h_terms + p) * 3u;
      atomic_add3(grad_h_coeff, h_base, grad_h * power);
      power *= tau;
    }
    float3 dh_dtau = eval_prt_h_dtau(h_coeff, tube_id, h_terms, tau);
    float grad_tau = dot(grad_h, dh_dtau) + grad_qv * 2.0f * lambda_t[tube_id] * tau;
    atomic_fetch_add_explicit(&grad_center_t[tube_id], -grad_tau, memory_order_relaxed);
    atomic_fetch_add_explicit(&grad_lambda_t[tube_id], grad_qv * tau * tau, memory_order_relaxed);
    atomic_fetch_add_explicit(&grad_lambda_uv[qbase + 0u], grad_qv * d.x * d.x, memory_order_relaxed);
    atomic_fetch_add_explicit(&grad_lambda_uv[qbase + 1u], grad_qv * 2.0f * d.x * d.y, memory_order_relaxed);
    atomic_fetch_add_explicit(&grad_lambda_uv[qbase + 2u], grad_qv * d.y * d.y, memory_order_relaxed);
    atomic_fetch_add_explicit(&grad_opacity[tube_id], d_alpha * exp_term, memory_order_relaxed);
  }
}

kernel void projective_rational_tile_pixel_compute_only_backward(
    const device float* h_coeff [[buffer(0)]],
    const device float* lambda_uv [[buffer(1)]],
    const device float* lambda_t [[buffer(2)]],
    const device float* center_t [[buffer(3)]],
    const device float* opacity [[buffer(4)]],
    const device float* color [[buffer(5)]],
    const device float* grad_image [[buffer(6)]],
    constant MetaI32& mi [[buffer(7)]],
    constant MetaF32& mf [[buffer(8)]],
    const device atomic_uint* tile_counts [[buffer(9)]],
    const device uint* tile_tube_ids [[buffer(10)]],
    const device float* tile_depths [[buffer(11)]],
    device atomic_uint* tile_unstable [[buffer(12)]],
    device float* debug_sink [[buffer(13)]],
    uint gid [[thread_position_in_grid]]) {
  uint pixels_per_tile = uint(STAR_TILE_X * STAR_TILE_Y * STAR_TILE_T);
  uint tile_id = gid / pixels_per_tile;
  uint local_pixel = gid - tile_id * pixels_per_tile;
  if (tile_id >= uint(mi.tile_count)) return;
  uint h_terms = uint(mi.reserved0);

  uint raw_count = atomic_load_explicit(tile_counts + tile_id, memory_order_relaxed);
  uint count = min(raw_count, STAR_TILE_CAPACITY);
  if (count == 0u) {
    debug_sink[gid] = 0.0f;
    return;
  }

  uint tx, ty, tz;
  decode_tile(tile_id, mi, tx, ty, tz);
  uint local_xy = local_pixel % uint(STAR_TILE_X * STAR_TILE_Y);
  uint lt = local_pixel / uint(STAR_TILE_X * STAR_TILE_Y);
  uint lx = local_xy % uint(STAR_TILE_X);
  uint ly = local_xy / uint(STAR_TILE_X);
  uint f = tz * STAR_TILE_T + lt;
  uint x = tx * STAR_TILE_X + lx;
  uint y = ty * STAR_TILE_Y + ly;
  if (f >= uint(mi.frames) || x >= uint(mi.width) || y >= uint(mi.height)) {
    debug_sink[gid] = 0.0f;
    return;
  }

  uint local_ids[STAR_TILE_CAPACITY];
  float local_depths[STAR_TILE_CAPACITY];
  for (uint i = 0u; i < count; ++i) {
    uint idx = tile_id * STAR_TILE_CAPACITY + i;
    local_ids[i] = tile_tube_ids[idx];
    local_depths[i] = tile_depths[idx];
  }
  sort_by_depth_thread(local_ids, local_depths, count);
  if (count > 1u) {
    atomic_store_explicit(tile_unstable + tile_id, 1u, memory_order_relaxed);
  }

  float t = frame_time(f, mi);
  float2 pixel = float2(float(x) + 0.5f, float(y) + 0.5f);
  uint ordered_ids[STAR_TILE_CAPACITY];
  float t_before[STAR_TILE_CAPACITY];
  float alpha_values[STAR_TILE_CAPACITY];
  bool processed[STAR_TILE_CAPACITY];
  bool differentiable_alpha[STAR_TILE_CAPACITY];

  uint ordered_count = 0u;
  if (uint(STAR_TILE_T) == 1u) {
    ordered_count = count;
    for (uint i = 0u; i < count; ++i) {
      ordered_ids[i] = local_ids[i];
    }
  } else {
    float last_depth = -INFINITY;
    uint last_id = 0xFFFFFFFFu;
    for (uint rank = 0u; rank < count; ++rank) {
      float selected_depth;
      uint tube_id = select_prt_sample_order_id_thread(
          local_ids,
          count,
          h_coeff,
          center_t,
          h_terms,
          t,
          last_depth,
          last_id,
          mf,
          selected_depth);
      if (tube_id == 0xFFFFFFFFu) break;
      ordered_ids[ordered_count] = tube_id;
      ordered_count += 1u;
      last_depth = selected_depth;
      last_id = tube_id;
    }
  }

  for (uint i = 0u; i < ordered_count; ++i) {
    t_before[i] = 0.0f;
    alpha_values[i] = 0.0f;
    processed[i] = false;
    differentiable_alpha[i] = false;
  }

  float T = 1.0f;
  for (uint i = 0u; i < ordered_count; ++i) {
    uint tube_id = ordered_ids[i];
    float tau = t - center_t[tube_id];
    float3 h = eval_prt_h(h_coeff, tube_id, h_terms, tau);
    float depth = max(h.z, mf.eps);
    float2 center = h.xy / depth;
    float2 d = pixel - center;
    uint qbase = tube_id * 3u;
    float spatial = lambda_uv[qbase + 0u] * d.x * d.x +
                    2.0f * lambda_uv[qbase + 1u] * d.x * d.y +
                    lambda_uv[qbase + 2u] * d.y * d.y;
    float temporal = lambda_t[tube_id] * tau * tau;
    float qv = spatial + temporal;
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

  float debug_value = 0.0f;
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

    debug_value += d_color.x + d_color.y + d_color.z;
    if (!differentiable_alpha[i]) continue;

    float tau = t - center_t[tube_id];
    float3 h = eval_prt_h(h_coeff, tube_id, h_terms, tau);
    float depth = max(h.z, mf.eps);
    float inv_depth = 1.0f / depth;
    float2 center = h.xy * inv_depth;
    float2 d = pixel - center;
    uint qbase = tube_id * 3u;
    float luu = lambda_uv[qbase + 0u];
    float luv = lambda_uv[qbase + 1u];
    float lvv = lambda_uv[qbase + 2u];
    float spatial = luu * d.x * d.x + 2.0f * luv * d.x * d.y + lvv * d.y * d.y;
    float temporal = lambda_t[tube_id] * tau * tau;
    float qv = spatial + temporal;
    float exp_term = exp(-0.5f * qv);
    float grad_qv = -0.5f * alpha * d_alpha;
    float2 qd = float2(luu * d.x + luv * d.y, luv * d.x + lvv * d.y);
    float2 grad_center = -2.0f * grad_qv * qd;
    float3 grad_h = float3(
        grad_center.x * inv_depth,
        grad_center.y * inv_depth,
        h.z > mf.eps ? -(grad_center.x * h.x + grad_center.y * h.y) * inv_depth * inv_depth : 0.0f);

    float power = 1.0f;
    for (uint p = 0u; p < h_terms && p < 8u; ++p) {
      float3 grad_hp = grad_h * power;
      debug_value += grad_hp.x + grad_hp.y + grad_hp.z;
      power *= tau;
    }
    float3 dh_dtau = eval_prt_h_dtau(h_coeff, tube_id, h_terms, tau);
    float grad_tau = dot(grad_h, dh_dtau) + grad_qv * 2.0f * lambda_t[tube_id] * tau;
    debug_value += -grad_tau;
    debug_value += grad_qv * tau * tau;
    debug_value += grad_qv * d.x * d.x;
    debug_value += grad_qv * 2.0f * d.x * d.y;
    debug_value += grad_qv * d.y * d.y;
    debug_value += d_alpha * exp_term;
  }
  debug_sink[gid] = debug_value;
}

kernel void projective_rational_tile_pixel_replay_only_backward(
    const device float* h_coeff [[buffer(0)]],
    const device float* lambda_uv [[buffer(1)]],
    const device float* lambda_t [[buffer(2)]],
    const device float* center_t [[buffer(3)]],
    const device float* opacity [[buffer(4)]],
    const device float* color [[buffer(5)]],
    const device float* grad_image [[buffer(6)]],
    constant MetaI32& mi [[buffer(7)]],
    constant MetaF32& mf [[buffer(8)]],
    const device atomic_uint* tile_counts [[buffer(9)]],
    const device uint* tile_tube_ids [[buffer(10)]],
    const device float* tile_depths [[buffer(11)]],
    device atomic_uint* tile_unstable [[buffer(12)]],
    device float* debug_sink [[buffer(13)]],
    uint gid [[thread_position_in_grid]]) {
  uint pixels_per_tile = uint(STAR_TILE_X * STAR_TILE_Y * STAR_TILE_T);
  uint tile_id = gid / pixels_per_tile;
  uint local_pixel = gid - tile_id * pixels_per_tile;
  if (tile_id >= uint(mi.tile_count)) return;
  uint h_terms = uint(mi.reserved0);

  uint raw_count = atomic_load_explicit(tile_counts + tile_id, memory_order_relaxed);
  uint count = min(raw_count, STAR_TILE_CAPACITY);
  if (count == 0u) {
    debug_sink[gid] = 0.0f;
    return;
  }

  uint tx, ty, tz;
  decode_tile(tile_id, mi, tx, ty, tz);
  uint local_xy = local_pixel % uint(STAR_TILE_X * STAR_TILE_Y);
  uint lt = local_pixel / uint(STAR_TILE_X * STAR_TILE_Y);
  uint lx = local_xy % uint(STAR_TILE_X);
  uint ly = local_xy / uint(STAR_TILE_X);
  uint f = tz * STAR_TILE_T + lt;
  uint x = tx * STAR_TILE_X + lx;
  uint y = ty * STAR_TILE_Y + ly;
  if (f >= uint(mi.frames) || x >= uint(mi.width) || y >= uint(mi.height)) {
    debug_sink[gid] = 0.0f;
    return;
  }

  uint local_ids[STAR_TILE_CAPACITY];
  float local_depths[STAR_TILE_CAPACITY];
  for (uint i = 0u; i < count; ++i) {
    uint idx = tile_id * STAR_TILE_CAPACITY + i;
    local_ids[i] = tile_tube_ids[idx];
    local_depths[i] = tile_depths[idx];
  }
  sort_by_depth_thread(local_ids, local_depths, count);
  if (count > 1u) {
    atomic_store_explicit(tile_unstable + tile_id, 1u, memory_order_relaxed);
  }

  float t = frame_time(f, mi);
  float2 pixel = float2(float(x) + 0.5f, float(y) + 0.5f);
  uint ordered_ids[STAR_TILE_CAPACITY];

  uint ordered_count = 0u;
  if (uint(STAR_TILE_T) == 1u) {
    ordered_count = count;
    for (uint i = 0u; i < count; ++i) {
      ordered_ids[i] = local_ids[i];
    }
  } else {
    float last_depth = -INFINITY;
    uint last_id = 0xFFFFFFFFu;
    for (uint rank = 0u; rank < count; ++rank) {
      float selected_depth;
      uint tube_id = select_prt_sample_order_id_thread(
          local_ids,
          count,
          h_coeff,
          center_t,
          h_terms,
          t,
          last_depth,
          last_id,
          mf,
          selected_depth);
      if (tube_id == 0xFFFFFFFFu) break;
      ordered_ids[ordered_count] = tube_id;
      ordered_count += 1u;
      last_depth = selected_depth;
      last_id = tube_id;
    }
  }

  float debug_value = 0.0f;
  float T = 1.0f;
  for (uint i = 0u; i < ordered_count; ++i) {
    uint tube_id = ordered_ids[i];
    float tau = t - center_t[tube_id];
    float3 h = eval_prt_h(h_coeff, tube_id, h_terms, tau);
    float depth = max(h.z, mf.eps);
    float2 center = h.xy / depth;
    float2 d = pixel - center;
    uint qbase = tube_id * 3u;
    float spatial = lambda_uv[qbase + 0u] * d.x * d.x +
                    2.0f * lambda_uv[qbase + 1u] * d.x * d.y +
                    lambda_uv[qbase + 2u] * d.y * d.y;
    float temporal = lambda_t[tube_id] * tau * tau;
    float qv = spatial + temporal;
    if (!isfinite(qv)) continue;
    float alpha_raw = opacity[tube_id] * exp(-0.5f * qv);
    float alpha = min(mf.max_alpha, alpha_raw);
    if (!(alpha >= mf.alpha_threshold)) continue;
    debug_value += T * alpha + center.x * 1.0e-6f + center.y * 1.0e-6f;
    T *= (1.0f - alpha);
    if (T <= mf.transmittance_threshold) break;
  }
  debug_sink[gid] = debug_value + T * 1.0e-6f;
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
    device int* grad_keys [[buffer(19)]],
    constant int& write_keys [[buffer(20)]],
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
    if (write_keys != 0) {
      grad_keys[entry] = int(gid * STAR_TILE_CAPACITY + i);
    }
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

kernel void direct_atomic_backward(
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
    device atomic_float* grad_ma [[buffer(13)]],
    device atomic_float* grad_q [[buffer(14)]],
    device atomic_float* grad_opacity [[buffer(15)]],
    device atomic_float* grad_color [[buffer(16)]],
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

    uint color_base = tube_id * 3u;
    atomic_add3(grad_color, color_base, d_color);
    if (!differentiable_alpha[i]) continue;

    float3 d = sample_a - load3(ma, tube_id);
    float qv = quadratic_q(q_uvt, tube_id, d);
    float exp_term = exp(-0.5f * qv);
    float grad_qv = -0.5f * alpha * d_alpha;
    float3 qd = load_q_row0(q_uvt, tube_id) * d.x + load_q_row1(q_uvt, tube_id) * d.y + load_q_row2(q_uvt, tube_id) * d.z;
    float3 grad_m = -2.0f * grad_qv * qd;
    uint ma_base = tube_id * 3u;
    atomic_add3(grad_ma, ma_base, grad_m);
    uint q_base = tube_id * 6u;
    atomic_fetch_add_explicit(&grad_q[q_base + 0u], grad_qv * d.x * d.x, memory_order_relaxed);
    atomic_fetch_add_explicit(&grad_q[q_base + 1u], grad_qv * 2.0f * d.x * d.y, memory_order_relaxed);
    atomic_fetch_add_explicit(&grad_q[q_base + 2u], grad_qv * 2.0f * d.x * d.z, memory_order_relaxed);
    atomic_fetch_add_explicit(&grad_q[q_base + 3u], grad_qv * d.y * d.y, memory_order_relaxed);
    atomic_fetch_add_explicit(&grad_q[q_base + 4u], grad_qv * 2.0f * d.y * d.z, memory_order_relaxed);
    atomic_fetch_add_explicit(&grad_q[q_base + 5u], grad_qv * d.z * d.z, memory_order_relaxed);
    atomic_fetch_add_explicit(&grad_opacity[tube_id], d_alpha * exp_term, memory_order_relaxed);
  }
}

kernel void direct_fixedpoint_backward(
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
    device atomic_int* grad_ma [[buffer(13)]],
    device atomic_int* grad_q [[buffer(14)]],
    device atomic_int* grad_opacity [[buffer(15)]],
    device atomic_int* grad_color [[buffer(16)]],
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

    uint color_base = tube_id * 3u;
    atomic_add3_fixedpoint(grad_color, color_base, d_color);
    if (!differentiable_alpha[i]) continue;

    float3 d = sample_a - load3(ma, tube_id);
    float qv = quadratic_q(q_uvt, tube_id, d);
    float exp_term = exp(-0.5f * qv);
    float grad_qv = -0.5f * alpha * d_alpha;
    float3 qd = load_q_row0(q_uvt, tube_id) * d.x + load_q_row1(q_uvt, tube_id) * d.y + load_q_row2(q_uvt, tube_id) * d.z;
    float3 grad_m = -2.0f * grad_qv * qd;
    uint ma_base = tube_id * 3u;
    atomic_add3_fixedpoint(grad_ma, ma_base, grad_m);
    uint q_base = tube_id * 6u;
    atomic_add_fixedpoint(grad_q, q_base + 0u, grad_qv * d.x * d.x);
    atomic_add_fixedpoint(grad_q, q_base + 1u, grad_qv * 2.0f * d.x * d.y);
    atomic_add_fixedpoint(grad_q, q_base + 2u, grad_qv * 2.0f * d.x * d.z);
    atomic_add_fixedpoint(grad_q, q_base + 3u, grad_qv * d.y * d.y);
    atomic_add_fixedpoint(grad_q, q_base + 4u, grad_qv * 2.0f * d.y * d.z);
    atomic_add_fixedpoint(grad_q, q_base + 5u, grad_qv * d.z * d.z);
    atomic_add_fixedpoint(grad_opacity, tube_id, d_alpha * exp_term);
  }
}

kernel void direct_split_fixedpoint_backward(
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
    device atomic_int* grad_ma_coarse [[buffer(13)]],
    device atomic_int* grad_q_coarse [[buffer(14)]],
    device atomic_int* grad_opacity_coarse [[buffer(15)]],
    device atomic_int* grad_color_coarse [[buffer(16)]],
    device atomic_int* grad_ma_fine [[buffer(17)]],
    device atomic_int* grad_q_fine [[buffer(18)]],
    device atomic_int* grad_opacity_fine [[buffer(19)]],
    device atomic_int* grad_color_fine [[buffer(20)]],
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

    uint color_base = tube_id * 3u;
    atomic_add3_split_fixedpoint(grad_color_coarse, grad_color_fine, color_base, d_color);
    if (!differentiable_alpha[i]) continue;

    float3 d = sample_a - load3(ma, tube_id);
    float qv = quadratic_q(q_uvt, tube_id, d);
    float exp_term = exp(-0.5f * qv);
    float grad_qv = -0.5f * alpha * d_alpha;
    float3 qd = load_q_row0(q_uvt, tube_id) * d.x + load_q_row1(q_uvt, tube_id) * d.y + load_q_row2(q_uvt, tube_id) * d.z;
    float3 grad_m = -2.0f * grad_qv * qd;
    uint ma_base = tube_id * 3u;
    atomic_add3_split_fixedpoint(grad_ma_coarse, grad_ma_fine, ma_base, grad_m);
    uint q_base = tube_id * 6u;
    atomic_add_split_fixedpoint(grad_q_coarse, grad_q_fine, q_base + 0u, grad_qv * d.x * d.x);
    atomic_add_split_fixedpoint(grad_q_coarse, grad_q_fine, q_base + 1u, grad_qv * 2.0f * d.x * d.y);
    atomic_add_split_fixedpoint(grad_q_coarse, grad_q_fine, q_base + 2u, grad_qv * 2.0f * d.x * d.z);
    atomic_add_split_fixedpoint(grad_q_coarse, grad_q_fine, q_base + 3u, grad_qv * d.y * d.y);
    atomic_add_split_fixedpoint(grad_q_coarse, grad_q_fine, q_base + 4u, grad_qv * 2.0f * d.y * d.z);
    atomic_add_split_fixedpoint(grad_q_coarse, grad_q_fine, q_base + 5u, grad_qv * d.z * d.z);
    atomic_add_split_fixedpoint(grad_opacity_coarse, grad_opacity_fine, tube_id, d_alpha * exp_term);
  }
}

kernel void direct_serial_backward(
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
    device float* grad_ma [[buffer(13)]],
    device float* grad_q [[buffer(14)]],
    device float* grad_opacity [[buffer(15)]],
    device float* grad_color [[buffer(16)]],
    uint tube_id [[thread_position_in_grid]]) {
  if (tube_id >= uint(mi.tube_count)) return;

  float3 grad_m_sum = float3(0.0f);
  float q_sum0 = 0.0f;
  float q_sum1 = 0.0f;
  float q_sum2 = 0.0f;
  float q_sum3 = 0.0f;
  float q_sum4 = 0.0f;
  float q_sum5 = 0.0f;
  float opacity_sum = 0.0f;
  float3 color_sum = float3(0.0f);

  Bounds3i b = tube_bounds(ma, q_uvt, opacity, tube_id, mi, mf);
  if (!(b.x0 > b.x1 || b.y0 > b.y1 || b.f0 > b.f1)) {
    uint tx0 = uint(b.x0 / mi.tile_x);
    uint tx1 = uint(b.x1 / mi.tile_x);
    uint ty0 = uint(b.y0 / mi.tile_y);
    uint ty1 = uint(b.y1 / mi.tile_y);
    uint tz0 = uint(b.f0 / mi.tile_t);
    uint tz1 = uint(b.f1 / mi.tile_t);

    uint local_ids[STAR_TILE_CAPACITY];
    float local_depths[STAR_TILE_CAPACITY];
    uint ordered_ids[STAR_TILE_CAPACITY];
    float t_before[STAR_TILE_CAPACITY];
    float alpha_values[STAR_TILE_CAPACITY];
    bool processed[STAR_TILE_CAPACITY];
    bool differentiable_alpha[STAR_TILE_CAPACITY];

    for (uint tz = tz0; tz <= tz1; ++tz) {
      for (uint ty = ty0; ty <= ty1; ++ty) {
        for (uint tx = tx0; tx <= tx1; ++tx) {
          uint tile_id = encode_tile(tx, ty, tz, mi);
          uint raw_count = atomic_load_explicit(tile_counts + tile_id, memory_order_relaxed);
          uint count = min(raw_count, STAR_TILE_CAPACITY);
          bool target_present = false;
          for (uint i = 0u; i < count; ++i) {
            uint idx = tile_id * STAR_TILE_CAPACITY + i;
            local_ids[i] = tile_tube_ids[idx];
            local_depths[i] = tile_depths[idx];
            target_present = target_present || local_ids[i] == tube_id;
          }
          if (!target_present) continue;

          sort_by_depth_thread(local_ids, local_depths, count);
          bool unstable = tile_order_unstable_thread(local_ids, count, ma, depth0, depth_beta, tx, ty, tz, mi);
          if (unstable) {
            atomic_store_explicit(tile_unstable + tile_id, 1u, memory_order_relaxed);
          }

          for (uint lt = 0u; lt < STAR_TILE_T; ++lt) {
            uint f = tz * STAR_TILE_T + lt;
            if (f >= uint(mi.frames)) continue;
            for (uint ly = 0u; ly < STAR_TILE_Y; ++ly) {
              uint y = ty * STAR_TILE_Y + ly;
              if (y >= uint(mi.height)) continue;
              for (uint lx = 0u; lx < STAR_TILE_X; ++lx) {
                uint x = tx * STAR_TILE_X + lx;
                if (x >= uint(mi.width)) continue;
                float3 sample_a = float3(float(x) + 0.5f, float(y) + 0.5f, frame_time(f, mi));

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
                    uint selected_id = select_sample_order_id_thread(
                        local_ids,
                        count,
                        ma,
                        depth0,
                        depth_beta,
                        sample_a,
                        last_depth,
                        last_id,
                        selected_depth);
                    if (selected_id == 0xFFFFFFFFu) break;
                    ordered_ids[ordered_count] = selected_id;
                    ordered_count += 1u;
                    last_depth = selected_depth;
                    last_id = selected_id;
                  }
                }

                float T = 1.0f;
                for (uint i = 0u; i < ordered_count; ++i) {
                  t_before[i] = 0.0f;
                  alpha_values[i] = 0.0f;
                  processed[i] = false;
                  differentiable_alpha[i] = false;
                }
                for (uint i = 0u; i < ordered_count; ++i) {
                  uint ordered_tube = ordered_ids[i];
                  float3 d = sample_a - load3(ma, ordered_tube);
                  float qv = quadratic_q(q_uvt, ordered_tube, d);
                  if (!isfinite(qv)) continue;
                  float alpha_raw = opacity[ordered_tube] * exp(-0.5f * qv);
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
                  uint ordered_tube = ordered_ids[i];
                  float alpha = alpha_values[i];
                  float t_i = t_before[i];
                  float3 c = load3(color, ordered_tube);
                  float d_alpha = dot(grad_rgb, t_i * c) - dT_next * t_i;
                  float3 d_color = grad_rgb * (t_i * alpha);
                  float dT_i = dot(grad_rgb, alpha * c) + dT_next * (1.0f - alpha);
                  dT_next = dT_i;
                  if (ordered_tube != tube_id) continue;

                  color_sum += d_color;
                  if (differentiable_alpha[i]) {
                    float3 d = sample_a - load3(ma, tube_id);
                    float qv = quadratic_q(q_uvt, tube_id, d);
                    float exp_term = exp(-0.5f * qv);
                    float grad_qv = -0.5f * alpha * d_alpha;
                    float3 qd = load_q_row0(q_uvt, tube_id) * d.x + load_q_row1(q_uvt, tube_id) * d.y + load_q_row2(q_uvt, tube_id) * d.z;
                    grad_m_sum += -2.0f * grad_qv * qd;
                    q_sum0 += grad_qv * d.x * d.x;
                    q_sum1 += grad_qv * 2.0f * d.x * d.y;
                    q_sum2 += grad_qv * 2.0f * d.x * d.z;
                    q_sum3 += grad_qv * d.y * d.y;
                    q_sum4 += grad_qv * 2.0f * d.y * d.z;
                    q_sum5 += grad_qv * d.z * d.z;
                    opacity_sum += d_alpha * exp_term;
                  }
                  break;
                }
              }
            }
          }
        }
      }
    }
  }

  uint ma_base = tube_id * 3u;
  grad_ma[ma_base + 0u] = grad_m_sum.x;
  grad_ma[ma_base + 1u] = grad_m_sum.y;
  grad_ma[ma_base + 2u] = grad_m_sum.z;
  uint q_base = tube_id * 6u;
  grad_q[q_base + 0u] = q_sum0;
  grad_q[q_base + 1u] = q_sum1;
  grad_q[q_base + 2u] = q_sum2;
  grad_q[q_base + 3u] = q_sum3;
  grad_q[q_base + 4u] = q_sum4;
  grad_q[q_base + 5u] = q_sum5;
  grad_opacity[tube_id] = opacity_sum;
  uint color_base = tube_id * 3u;
  grad_color[color_base + 0u] = color_sum.x;
  grad_color[color_base + 1u] = color_sum.y;
  grad_color[color_base + 2u] = color_sum.z;
}

inline void tile_pair_backward_samples_impl(
    const device float* ma,
    const device float* q_uvt,
    const device float* depth0,
    const device float* depth_beta,
    const device float* opacity,
    const device float* color,
    const device float* grad_image,
    constant MetaI32& mi,
    constant MetaF32& mf,
    const device atomic_uint* tile_counts,
    const device uint* tile_tube_ids,
    const device float* tile_depths,
    device atomic_uint* tile_unstable,
    device int* grad_ids,
    device float* grad_ma,
    device float* grad_q,
    device float* grad_opacity,
    device float* grad_color,
    device int* grad_keys,
    uint gid,
    bool compensated_sums,
    bool target_bounds_only,
    bool suffix_composite) {
  uint tile_id = gid / STAR_TILE_CAPACITY;
  uint slot = gid - tile_id * STAR_TILE_CAPACITY;
  if (tile_id >= uint(mi.tile_count)) return;

  grad_ids[gid] = -1;
  grad_keys[gid] = int(gid);
  uint ma_base = gid * 3u;
  grad_ma[ma_base + 0u] = 0.0f;
  grad_ma[ma_base + 1u] = 0.0f;
  grad_ma[ma_base + 2u] = 0.0f;
  uint q_base = gid * 6u;
  grad_q[q_base + 0u] = 0.0f;
  grad_q[q_base + 1u] = 0.0f;
  grad_q[q_base + 2u] = 0.0f;
  grad_q[q_base + 3u] = 0.0f;
  grad_q[q_base + 4u] = 0.0f;
  grad_q[q_base + 5u] = 0.0f;
  grad_opacity[gid] = 0.0f;
  uint color_base = gid * 3u;
  grad_color[color_base + 0u] = 0.0f;
  grad_color[color_base + 1u] = 0.0f;
  grad_color[color_base + 2u] = 0.0f;

  uint raw_count = atomic_load_explicit(tile_counts + tile_id, memory_order_relaxed);
  uint count = min(raw_count, STAR_TILE_CAPACITY);
  if (slot >= count) return;

  uint tx, ty, tz;
  decode_tile(tile_id, mi, tx, ty, tz);

  uint local_ids[STAR_TILE_CAPACITY];
  float local_depths[STAR_TILE_CAPACITY];
  for (uint i = 0u; i < count; ++i) {
    uint idx = tile_id * STAR_TILE_CAPACITY + i;
    local_ids[i] = tile_tube_ids[idx];
    local_depths[i] = tile_depths[idx];
  }
  sort_by_depth_thread(local_ids, local_depths, count);
  uint target_id = local_ids[slot];
  Bounds3i target_bounds;
  if (target_bounds_only) {
    target_bounds = tube_bounds(ma, q_uvt, opacity, target_id, mi, mf);
    if (target_bounds.x0 > target_bounds.x1 || target_bounds.y0 > target_bounds.y1 || target_bounds.f0 > target_bounds.f1) {
      return;
    }
  }

  bool unstable = tile_order_unstable_thread(local_ids, count, ma, depth0, depth_beta, tx, ty, tz, mi);
  if (slot == 0u && unstable) {
    atomic_store_explicit(tile_unstable + tile_id, 1u, memory_order_relaxed);
  }

  float3 grad_m_sum = float3(0.0f);
  float q_sum0 = 0.0f;
  float q_sum1 = 0.0f;
  float q_sum2 = 0.0f;
  float q_sum3 = 0.0f;
  float q_sum4 = 0.0f;
  float q_sum5 = 0.0f;
  float opacity_sum = 0.0f;
  float3 color_sum = float3(0.0f);
  float3 grad_m_comp = float3(0.0f);
  float q_comp0 = 0.0f;
  float q_comp1 = 0.0f;
  float q_comp2 = 0.0f;
  float q_comp3 = 0.0f;
  float q_comp4 = 0.0f;
  float q_comp5 = 0.0f;
  float opacity_comp = 0.0f;
  float3 color_comp = float3(0.0f);

  uint ordered_ids[STAR_TILE_CAPACITY];
  float t_before[STAR_TILE_CAPACITY];
  float alpha_values[STAR_TILE_CAPACITY];
  bool processed[STAR_TILE_CAPACITY];
  bool differentiable_alpha[STAR_TILE_CAPACITY];

  for (uint lt = 0u; lt < STAR_TILE_T; ++lt) {
    uint f = tz * STAR_TILE_T + lt;
    if (f >= uint(mi.frames)) continue;
    if (target_bounds_only && (int(f) < target_bounds.f0 || int(f) > target_bounds.f1)) continue;
    for (uint ly = 0u; ly < STAR_TILE_Y; ++ly) {
      uint y = ty * STAR_TILE_Y + ly;
      if (y >= uint(mi.height)) continue;
      if (target_bounds_only && (int(y) < target_bounds.y0 || int(y) > target_bounds.y1)) continue;
      for (uint lx = 0u; lx < STAR_TILE_X; ++lx) {
        uint x = tx * STAR_TILE_X + lx;
        if (x >= uint(mi.width)) continue;
        if (target_bounds_only && (int(x) < target_bounds.x0 || int(x) > target_bounds.x1)) continue;
        float3 sample_a = float3(float(x) + 0.5f, float(y) + 0.5f, frame_time(f, mi));

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
            uint tube_id = select_sample_order_id_thread(local_ids, count, ma, depth0, depth_beta, sample_a, last_depth, last_id, selected_depth);
            if (tube_id == 0xFFFFFFFFu) break;
            ordered_ids[ordered_count] = tube_id;
            ordered_count += 1u;
            last_depth = selected_depth;
            last_id = tube_id;
          }
        }

        if (suffix_composite) {
          float prefix_T = 1.0f;
          float target_t = 0.0f;
          float target_alpha = 0.0f;
          bool target_processed = false;
          bool target_differentiable = false;
          uint target_rank = 0u;
          for (uint i = 0u; i < ordered_count; ++i) {
            uint tube_id = ordered_ids[i];
            float3 d = sample_a - load3(ma, tube_id);
            float qv = quadratic_q(q_uvt, tube_id, d);
            if (!isfinite(qv)) continue;
            float alpha_raw = opacity[tube_id] * exp(-0.5f * qv);
            float alpha = min(mf.max_alpha, alpha_raw);
            if (!(alpha >= mf.alpha_threshold)) continue;
            if (tube_id == target_id) {
              target_t = prefix_T;
              target_alpha = alpha;
              target_processed = true;
              target_differentiable = alpha_raw < mf.max_alpha;
              target_rank = i;
              prefix_T *= (1.0f - alpha);
              break;
            }
            prefix_T *= (1.0f - alpha);
            if (prefix_T <= mf.transmittance_threshold) break;
          }
          if (!target_processed) continue;

          float T_after_target = prefix_T;
          float suffix_T = 1.0f;
          float3 suffix_accum = float3(0.0f);
          if (T_after_target > mf.transmittance_threshold) {
            for (uint i = target_rank + 1u; i < ordered_count; ++i) {
              uint tube_id = ordered_ids[i];
              float3 d = sample_a - load3(ma, tube_id);
              float qv = quadratic_q(q_uvt, tube_id, d);
              if (!isfinite(qv)) continue;
              float alpha_raw = opacity[tube_id] * exp(-0.5f * qv);
              float alpha = min(mf.max_alpha, alpha_raw);
              if (!(alpha >= mf.alpha_threshold)) continue;
              float w = suffix_T * alpha;
              suffix_accum += w * load3(color, tube_id);
              suffix_T *= (1.0f - alpha);
              if (T_after_target * suffix_T <= mf.transmittance_threshold) break;
            }
          }

          uint image_base = ((f * uint(mi.height) + y) * uint(mi.width) + x) * 3u;
          float3 grad_rgb = float3(grad_image[image_base + 0u], grad_image[image_base + 1u], grad_image[image_base + 2u]);
          float3 suffix_color = suffix_accum + suffix_T * float3(mf.bg_r, mf.bg_g, mf.bg_b);
          float3 c = load3(color, target_id);
          float d_alpha = dot(grad_rgb, target_t * c) - dot(grad_rgb, suffix_color) * target_t;
          float3 d_color = grad_rgb * (target_t * target_alpha);
          if (compensated_sums) {
            KAHAN_ADD(color_sum, color_comp, d_color);
          } else {
            color_sum += d_color;
          }
          if (target_differentiable) {
            float3 d = sample_a - load3(ma, target_id);
            float qv = quadratic_q(q_uvt, target_id, d);
            float exp_term = exp(-0.5f * qv);
            float grad_qv = -0.5f * target_alpha * d_alpha;
            float3 qd = load_q_row0(q_uvt, target_id) * d.x + load_q_row1(q_uvt, target_id) * d.y + load_q_row2(q_uvt, target_id) * d.z;
            float3 grad_m_value = -2.0f * grad_qv * qd;
            float q_value0 = grad_qv * d.x * d.x;
            float q_value1 = grad_qv * 2.0f * d.x * d.y;
            float q_value2 = grad_qv * 2.0f * d.x * d.z;
            float q_value3 = grad_qv * d.y * d.y;
            float q_value4 = grad_qv * 2.0f * d.y * d.z;
            float q_value5 = grad_qv * d.z * d.z;
            float opacity_value = d_alpha * exp_term;
            if (compensated_sums) {
              KAHAN_ADD(grad_m_sum, grad_m_comp, grad_m_value);
              KAHAN_ADD(q_sum0, q_comp0, q_value0);
              KAHAN_ADD(q_sum1, q_comp1, q_value1);
              KAHAN_ADD(q_sum2, q_comp2, q_value2);
              KAHAN_ADD(q_sum3, q_comp3, q_value3);
              KAHAN_ADD(q_sum4, q_comp4, q_value4);
              KAHAN_ADD(q_sum5, q_comp5, q_value5);
              KAHAN_ADD(opacity_sum, opacity_comp, opacity_value);
            } else {
              grad_m_sum += grad_m_value;
              q_sum0 += q_value0;
              q_sum1 += q_value1;
              q_sum2 += q_value2;
              q_sum3 += q_value3;
              q_sum4 += q_value4;
              q_sum5 += q_value5;
              opacity_sum += opacity_value;
            }
          }
          continue;
        }

        float T = 1.0f;
        for (uint i = 0u; i < ordered_count; ++i) {
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
          if (tube_id != target_id) continue;

          if (compensated_sums) {
            KAHAN_ADD(color_sum, color_comp, d_color);
          } else {
            color_sum += d_color;
          }
          if (!differentiable_alpha[i]) continue;
          float3 d = sample_a - load3(ma, tube_id);
          float qv = quadratic_q(q_uvt, tube_id, d);
          float exp_term = exp(-0.5f * qv);
          float grad_qv = -0.5f * alpha * d_alpha;
          float3 qd = load_q_row0(q_uvt, tube_id) * d.x + load_q_row1(q_uvt, tube_id) * d.y + load_q_row2(q_uvt, tube_id) * d.z;
          float3 grad_m_value = -2.0f * grad_qv * qd;
          float q_value0 = grad_qv * d.x * d.x;
          float q_value1 = grad_qv * 2.0f * d.x * d.y;
          float q_value2 = grad_qv * 2.0f * d.x * d.z;
          float q_value3 = grad_qv * d.y * d.y;
          float q_value4 = grad_qv * 2.0f * d.y * d.z;
          float q_value5 = grad_qv * d.z * d.z;
          float opacity_value = d_alpha * exp_term;
          if (compensated_sums) {
            KAHAN_ADD(grad_m_sum, grad_m_comp, grad_m_value);
            KAHAN_ADD(q_sum0, q_comp0, q_value0);
            KAHAN_ADD(q_sum1, q_comp1, q_value1);
            KAHAN_ADD(q_sum2, q_comp2, q_value2);
            KAHAN_ADD(q_sum3, q_comp3, q_value3);
            KAHAN_ADD(q_sum4, q_comp4, q_value4);
            KAHAN_ADD(q_sum5, q_comp5, q_value5);
            KAHAN_ADD(opacity_sum, opacity_comp, opacity_value);
          } else {
            grad_m_sum += grad_m_value;
            q_sum0 += q_value0;
            q_sum1 += q_value1;
            q_sum2 += q_value2;
            q_sum3 += q_value3;
            q_sum4 += q_value4;
            q_sum5 += q_value5;
            opacity_sum += opacity_value;
          }
          break;
        }
      }
    }
  }

  grad_ma[ma_base + 0u] = grad_m_sum.x;
  grad_ma[ma_base + 1u] = grad_m_sum.y;
  grad_ma[ma_base + 2u] = grad_m_sum.z;
  grad_q[q_base + 0u] = q_sum0;
  grad_q[q_base + 1u] = q_sum1;
  grad_q[q_base + 2u] = q_sum2;
  grad_q[q_base + 3u] = q_sum3;
  grad_q[q_base + 4u] = q_sum4;
  grad_q[q_base + 5u] = q_sum5;
  grad_opacity[gid] = opacity_sum;
  grad_color[color_base + 0u] = color_sum.x;
  grad_color[color_base + 1u] = color_sum.y;
  grad_color[color_base + 2u] = color_sum.z;
  bool has_gradient =
      grad_m_sum.x != 0.0f || grad_m_sum.y != 0.0f || grad_m_sum.z != 0.0f ||
      q_sum0 != 0.0f || q_sum1 != 0.0f || q_sum2 != 0.0f ||
      q_sum3 != 0.0f || q_sum4 != 0.0f || q_sum5 != 0.0f ||
      opacity_sum != 0.0f ||
      color_sum.x != 0.0f || color_sum.y != 0.0f || color_sum.z != 0.0f;
  if (has_gradient) {
    grad_ids[gid] = int(target_id);
  }
}

kernel void tile_pair_parallel_backward_samples(
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
    device int* grad_ids [[buffer(13)]],
    device float* grad_ma [[buffer(14)]],
    device float* grad_q [[buffer(15)]],
    device float* grad_opacity [[buffer(16)]],
    device float* grad_color [[buffer(17)]],
    device int* grad_keys [[buffer(18)]],
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]]) {
  uint row = gid / STAR_THREADS;
  uint local_tid = tid;
  uint tile_id = row / STAR_TILE_CAPACITY;
  uint slot = row - tile_id * STAR_TILE_CAPACITY;
  if (row >= uint(mi.tile_count) * STAR_TILE_CAPACITY) return;

  if (local_tid == 0u) {
    grad_ids[row] = -1;
    grad_keys[row] = int(row);
    uint ma_base = row * 3u;
    grad_ma[ma_base + 0u] = 0.0f;
    grad_ma[ma_base + 1u] = 0.0f;
    grad_ma[ma_base + 2u] = 0.0f;
    uint q_base = row * 6u;
    grad_q[q_base + 0u] = 0.0f;
    grad_q[q_base + 1u] = 0.0f;
    grad_q[q_base + 2u] = 0.0f;
    grad_q[q_base + 3u] = 0.0f;
    grad_q[q_base + 4u] = 0.0f;
    grad_q[q_base + 5u] = 0.0f;
    grad_opacity[row] = 0.0f;
    uint color_base = row * 3u;
    grad_color[color_base + 0u] = 0.0f;
    grad_color[color_base + 1u] = 0.0f;
    grad_color[color_base + 2u] = 0.0f;
  }

  uint raw_count = atomic_load_explicit(tile_counts + tile_id, memory_order_relaxed);
  uint count = min(raw_count, STAR_TILE_CAPACITY);
  if (slot >= count) return;

  threadgroup uint local_ids[STAR_TILE_CAPACITY];
  threadgroup float local_depths[STAR_TILE_CAPACITY];
  threadgroup uint unstable_flag;
  threadgroup float reduce_scratch[STAR_THREADS];

  for (uint i = local_tid; i < count; i += STAR_THREADS) {
    uint idx = tile_id * STAR_TILE_CAPACITY + i;
    local_ids[i] = tile_tube_ids[idx];
    local_depths[i] = tile_depths[idx];
  }
  sort_by_depth(local_ids, local_depths, count, local_tid);
  uint target_id = local_ids[slot];

  uint tx, ty, tz;
  decode_tile(tile_id, mi, tx, ty, tz);
  if (local_tid == 0u) {
    unstable_flag = tile_order_unstable(local_ids, count, ma, depth0, depth_beta, tx, ty, tz, mi) ? 1u : 0u;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool unstable = unstable_flag != 0u;
  if (local_tid == 0u && slot == 0u && unstable) {
    atomic_store_explicit(tile_unstable + tile_id, 1u, memory_order_relaxed);
  }

  float3 grad_m_sum = float3(0.0f);
  float q_sum0 = 0.0f;
  float q_sum1 = 0.0f;
  float q_sum2 = 0.0f;
  float q_sum3 = 0.0f;
  float q_sum4 = 0.0f;
  float q_sum5 = 0.0f;
  float opacity_sum = 0.0f;
  float3 color_sum = float3(0.0f);

  uint lt = local_tid / (STAR_TILE_X * STAR_TILE_Y);
  uint rem = local_tid - lt * STAR_TILE_X * STAR_TILE_Y;
  uint ly = rem / STAR_TILE_X;
  uint lx = rem - ly * STAR_TILE_X;
  uint f = tz * STAR_TILE_T + lt;
  uint y = ty * STAR_TILE_Y + ly;
  uint x = tx * STAR_TILE_X + lx;

  if (f < uint(mi.frames) && y < uint(mi.height) && x < uint(mi.width)) {
    float3 sample_a = float3(float(x) + 0.5f, float(y) + 0.5f, frame_time(f, mi));

    uint ordered_ids[STAR_TILE_CAPACITY];
    float t_before[STAR_TILE_CAPACITY];
    float alpha_values[STAR_TILE_CAPACITY];
    bool processed[STAR_TILE_CAPACITY];
    bool differentiable_alpha[STAR_TILE_CAPACITY];

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

    float T = 1.0f;
    for (uint i = 0u; i < ordered_count; ++i) {
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
      if (tube_id != target_id) continue;

      color_sum += d_color;
      if (!differentiable_alpha[i]) break;
      float3 d = sample_a - load3(ma, tube_id);
      float qv = quadratic_q(q_uvt, tube_id, d);
      float exp_term = exp(-0.5f * qv);
      float grad_qv = -0.5f * alpha * d_alpha;
      float3 qd = load_q_row0(q_uvt, tube_id) * d.x + load_q_row1(q_uvt, tube_id) * d.y + load_q_row2(q_uvt, tube_id) * d.z;
      grad_m_sum += -2.0f * grad_qv * qd;
      q_sum0 += grad_qv * d.x * d.x;
      q_sum1 += grad_qv * 2.0f * d.x * d.y;
      q_sum2 += grad_qv * 2.0f * d.x * d.z;
      q_sum3 += grad_qv * d.y * d.y;
      q_sum4 += grad_qv * 2.0f * d.y * d.z;
      q_sum5 += grad_qv * d.z * d.z;
      opacity_sum += d_alpha * exp_term;
      break;
    }
  }

  float grad_m0 = reduce_threadgroup_sum(reduce_scratch, grad_m_sum.x, local_tid);
  float grad_m1 = reduce_threadgroup_sum(reduce_scratch, grad_m_sum.y, local_tid);
  float grad_m2 = reduce_threadgroup_sum(reduce_scratch, grad_m_sum.z, local_tid);
  float q0 = reduce_threadgroup_sum(reduce_scratch, q_sum0, local_tid);
  float q1 = reduce_threadgroup_sum(reduce_scratch, q_sum1, local_tid);
  float q2 = reduce_threadgroup_sum(reduce_scratch, q_sum2, local_tid);
  float q3 = reduce_threadgroup_sum(reduce_scratch, q_sum3, local_tid);
  float q4 = reduce_threadgroup_sum(reduce_scratch, q_sum4, local_tid);
  float q5 = reduce_threadgroup_sum(reduce_scratch, q_sum5, local_tid);
  float opacity_value = reduce_threadgroup_sum(reduce_scratch, opacity_sum, local_tid);
  float color0 = reduce_threadgroup_sum(reduce_scratch, color_sum.x, local_tid);
  float color1 = reduce_threadgroup_sum(reduce_scratch, color_sum.y, local_tid);
  float color2 = reduce_threadgroup_sum(reduce_scratch, color_sum.z, local_tid);

  if (local_tid == 0u) {
    uint ma_base = row * 3u;
    grad_ma[ma_base + 0u] = grad_m0;
    grad_ma[ma_base + 1u] = grad_m1;
    grad_ma[ma_base + 2u] = grad_m2;
    uint q_base = row * 6u;
    grad_q[q_base + 0u] = q0;
    grad_q[q_base + 1u] = q1;
    grad_q[q_base + 2u] = q2;
    grad_q[q_base + 3u] = q3;
    grad_q[q_base + 4u] = q4;
    grad_q[q_base + 5u] = q5;
    grad_opacity[row] = opacity_value;
    uint color_base = row * 3u;
    grad_color[color_base + 0u] = color0;
    grad_color[color_base + 1u] = color1;
    grad_color[color_base + 2u] = color2;
    bool has_gradient =
        grad_m0 != 0.0f || grad_m1 != 0.0f || grad_m2 != 0.0f ||
        q0 != 0.0f || q1 != 0.0f || q2 != 0.0f ||
        q3 != 0.0f || q4 != 0.0f || q5 != 0.0f ||
        opacity_value != 0.0f ||
        color0 != 0.0f || color1 != 0.0f || color2 != 0.0f;
    if (has_gradient) {
      grad_ids[row] = int(target_id);
    }
  }
}

kernel void tile_pair_atomic_backward(
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
    device atomic_float* grad_ma [[buffer(13)]],
    device atomic_float* grad_q [[buffer(14)]],
    device atomic_float* grad_opacity [[buffer(15)]],
    device atomic_float* grad_color [[buffer(16)]],
    uint gid [[thread_position_in_grid]]) {
  uint tile_id = gid / STAR_TILE_CAPACITY;
  uint slot = gid - tile_id * STAR_TILE_CAPACITY;
  if (tile_id >= uint(mi.tile_count)) return;

  uint raw_count = atomic_load_explicit(tile_counts + tile_id, memory_order_relaxed);
  uint count = min(raw_count, STAR_TILE_CAPACITY);
  if (slot >= count) return;

  uint tx, ty, tz;
  decode_tile(tile_id, mi, tx, ty, tz);

  uint local_ids[STAR_TILE_CAPACITY];
  float local_depths[STAR_TILE_CAPACITY];
  for (uint i = 0u; i < count; ++i) {
    uint idx = tile_id * STAR_TILE_CAPACITY + i;
    local_ids[i] = tile_tube_ids[idx];
    local_depths[i] = tile_depths[idx];
  }
  sort_by_depth_thread(local_ids, local_depths, count);
  uint target_id = local_ids[slot];

  bool unstable = tile_order_unstable_thread(local_ids, count, ma, depth0, depth_beta, tx, ty, tz, mi);
  if (slot == 0u && unstable) {
    atomic_store_explicit(tile_unstable + tile_id, 1u, memory_order_relaxed);
  }

  float3 grad_m_sum = float3(0.0f);
  float q_sum0 = 0.0f;
  float q_sum1 = 0.0f;
  float q_sum2 = 0.0f;
  float q_sum3 = 0.0f;
  float q_sum4 = 0.0f;
  float q_sum5 = 0.0f;
  float opacity_sum = 0.0f;
  float3 color_sum = float3(0.0f);

  uint ordered_ids[STAR_TILE_CAPACITY];
  float t_before[STAR_TILE_CAPACITY];
  float alpha_values[STAR_TILE_CAPACITY];
  bool processed[STAR_TILE_CAPACITY];
  bool differentiable_alpha[STAR_TILE_CAPACITY];

  for (uint lt = 0u; lt < STAR_TILE_T; ++lt) {
    uint f = tz * STAR_TILE_T + lt;
    if (f >= uint(mi.frames)) continue;
    for (uint ly = 0u; ly < STAR_TILE_Y; ++ly) {
      uint y = ty * STAR_TILE_Y + ly;
      if (y >= uint(mi.height)) continue;
      for (uint lx = 0u; lx < STAR_TILE_X; ++lx) {
        uint x = tx * STAR_TILE_X + lx;
        if (x >= uint(mi.width)) continue;
        float3 sample_a = float3(float(x) + 0.5f, float(y) + 0.5f, frame_time(f, mi));

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
            uint tube_id = select_sample_order_id_thread(local_ids, count, ma, depth0, depth_beta, sample_a, last_depth, last_id, selected_depth);
            if (tube_id == 0xFFFFFFFFu) break;
            ordered_ids[ordered_count] = tube_id;
            ordered_count += 1u;
            last_depth = selected_depth;
            last_id = tube_id;
          }
        }

        float T = 1.0f;
        for (uint i = 0u; i < ordered_count; ++i) {
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
          if (tube_id != target_id) continue;

          color_sum += d_color;
          if (!differentiable_alpha[i]) continue;
          float3 d = sample_a - load3(ma, tube_id);
          float qv = quadratic_q(q_uvt, tube_id, d);
          float exp_term = exp(-0.5f * qv);
          float grad_qv = -0.5f * alpha * d_alpha;
          float3 qd = load_q_row0(q_uvt, tube_id) * d.x + load_q_row1(q_uvt, tube_id) * d.y + load_q_row2(q_uvt, tube_id) * d.z;
          grad_m_sum += -2.0f * grad_qv * qd;
          q_sum0 += grad_qv * d.x * d.x;
          q_sum1 += grad_qv * 2.0f * d.x * d.y;
          q_sum2 += grad_qv * 2.0f * d.x * d.z;
          q_sum3 += grad_qv * d.y * d.y;
          q_sum4 += grad_qv * 2.0f * d.y * d.z;
          q_sum5 += grad_qv * d.z * d.z;
          opacity_sum += d_alpha * exp_term;
          break;
        }
      }
    }
  }

  bool has_gradient =
      grad_m_sum.x != 0.0f || grad_m_sum.y != 0.0f || grad_m_sum.z != 0.0f ||
      q_sum0 != 0.0f || q_sum1 != 0.0f || q_sum2 != 0.0f ||
      q_sum3 != 0.0f || q_sum4 != 0.0f || q_sum5 != 0.0f ||
      opacity_sum != 0.0f ||
      color_sum.x != 0.0f || color_sum.y != 0.0f || color_sum.z != 0.0f;
  if (!has_gradient) return;

  uint ma_base = target_id * 3u;
  atomic_add3(grad_ma, ma_base, grad_m_sum);
  uint q_base = target_id * 6u;
  atomic_fetch_add_explicit(&grad_q[q_base + 0u], q_sum0, memory_order_relaxed);
  atomic_fetch_add_explicit(&grad_q[q_base + 1u], q_sum1, memory_order_relaxed);
  atomic_fetch_add_explicit(&grad_q[q_base + 2u], q_sum2, memory_order_relaxed);
  atomic_fetch_add_explicit(&grad_q[q_base + 3u], q_sum3, memory_order_relaxed);
  atomic_fetch_add_explicit(&grad_q[q_base + 4u], q_sum4, memory_order_relaxed);
  atomic_fetch_add_explicit(&grad_q[q_base + 5u], q_sum5, memory_order_relaxed);
  atomic_fetch_add_explicit(&grad_opacity[target_id], opacity_sum, memory_order_relaxed);
  uint color_base = target_id * 3u;
  atomic_add3(grad_color, color_base, color_sum);
}

kernel void tile_pair_fixedpoint_backward(
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
    device atomic_int* grad_ma [[buffer(13)]],
    device atomic_int* grad_q [[buffer(14)]],
    device atomic_int* grad_opacity [[buffer(15)]],
    device atomic_int* grad_color [[buffer(16)]],
    uint gid [[thread_position_in_grid]]) {
  uint tile_id = gid / STAR_TILE_CAPACITY;
  uint slot = gid - tile_id * STAR_TILE_CAPACITY;
  if (tile_id >= uint(mi.tile_count)) return;

  uint raw_count = atomic_load_explicit(tile_counts + tile_id, memory_order_relaxed);
  uint count = min(raw_count, STAR_TILE_CAPACITY);
  if (slot >= count) return;

  uint tx, ty, tz;
  decode_tile(tile_id, mi, tx, ty, tz);

  uint local_ids[STAR_TILE_CAPACITY];
  float local_depths[STAR_TILE_CAPACITY];
  for (uint i = 0u; i < count; ++i) {
    uint idx = tile_id * STAR_TILE_CAPACITY + i;
    local_ids[i] = tile_tube_ids[idx];
    local_depths[i] = tile_depths[idx];
  }
  sort_by_depth_thread(local_ids, local_depths, count);
  uint target_id = local_ids[slot];

  bool unstable = tile_order_unstable_thread(local_ids, count, ma, depth0, depth_beta, tx, ty, tz, mi);
  if (slot == 0u && unstable) {
    atomic_store_explicit(tile_unstable + tile_id, 1u, memory_order_relaxed);
  }

  float3 grad_m_sum = float3(0.0f);
  float q_sum0 = 0.0f;
  float q_sum1 = 0.0f;
  float q_sum2 = 0.0f;
  float q_sum3 = 0.0f;
  float q_sum4 = 0.0f;
  float q_sum5 = 0.0f;
  float opacity_sum = 0.0f;
  float3 color_sum = float3(0.0f);

  uint ordered_ids[STAR_TILE_CAPACITY];
  float t_before[STAR_TILE_CAPACITY];
  float alpha_values[STAR_TILE_CAPACITY];
  bool processed[STAR_TILE_CAPACITY];
  bool differentiable_alpha[STAR_TILE_CAPACITY];

  for (uint lt = 0u; lt < STAR_TILE_T; ++lt) {
    uint f = tz * STAR_TILE_T + lt;
    if (f >= uint(mi.frames)) continue;
    for (uint ly = 0u; ly < STAR_TILE_Y; ++ly) {
      uint y = ty * STAR_TILE_Y + ly;
      if (y >= uint(mi.height)) continue;
      for (uint lx = 0u; lx < STAR_TILE_X; ++lx) {
        uint x = tx * STAR_TILE_X + lx;
        if (x >= uint(mi.width)) continue;
        float3 sample_a = float3(float(x) + 0.5f, float(y) + 0.5f, frame_time(f, mi));

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
            uint tube_id = select_sample_order_id_thread(local_ids, count, ma, depth0, depth_beta, sample_a, last_depth, last_id, selected_depth);
            if (tube_id == 0xFFFFFFFFu) break;
            ordered_ids[ordered_count] = tube_id;
            ordered_count += 1u;
            last_depth = selected_depth;
            last_id = tube_id;
          }
        }

        float T = 1.0f;
        for (uint i = 0u; i < ordered_count; ++i) {
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
          if (tube_id != target_id) continue;

          color_sum += d_color;
          if (!differentiable_alpha[i]) continue;
          float3 d = sample_a - load3(ma, tube_id);
          float qv = quadratic_q(q_uvt, tube_id, d);
          float exp_term = exp(-0.5f * qv);
          float grad_qv = -0.5f * alpha * d_alpha;
          float3 qd = load_q_row0(q_uvt, tube_id) * d.x + load_q_row1(q_uvt, tube_id) * d.y + load_q_row2(q_uvt, tube_id) * d.z;
          grad_m_sum += -2.0f * grad_qv * qd;
          q_sum0 += grad_qv * d.x * d.x;
          q_sum1 += grad_qv * 2.0f * d.x * d.y;
          q_sum2 += grad_qv * 2.0f * d.x * d.z;
          q_sum3 += grad_qv * d.y * d.y;
          q_sum4 += grad_qv * 2.0f * d.y * d.z;
          q_sum5 += grad_qv * d.z * d.z;
          opacity_sum += d_alpha * exp_term;
          break;
        }
      }
    }
  }

  bool has_gradient =
      grad_m_sum.x != 0.0f || grad_m_sum.y != 0.0f || grad_m_sum.z != 0.0f ||
      q_sum0 != 0.0f || q_sum1 != 0.0f || q_sum2 != 0.0f ||
      q_sum3 != 0.0f || q_sum4 != 0.0f || q_sum5 != 0.0f ||
      opacity_sum != 0.0f ||
      color_sum.x != 0.0f || color_sum.y != 0.0f || color_sum.z != 0.0f;
  if (!has_gradient) return;

  uint ma_base = target_id * 3u;
  atomic_add3_fixedpoint(grad_ma, ma_base, grad_m_sum);
  uint q_base = target_id * 6u;
  atomic_add_fixedpoint(grad_q, q_base + 0u, q_sum0);
  atomic_add_fixedpoint(grad_q, q_base + 1u, q_sum1);
  atomic_add_fixedpoint(grad_q, q_base + 2u, q_sum2);
  atomic_add_fixedpoint(grad_q, q_base + 3u, q_sum3);
  atomic_add_fixedpoint(grad_q, q_base + 4u, q_sum4);
  atomic_add_fixedpoint(grad_q, q_base + 5u, q_sum5);
  atomic_add_fixedpoint(grad_opacity, target_id, opacity_sum);
  uint color_base = target_id * 3u;
  atomic_add3_fixedpoint(grad_color, color_base, color_sum);
}

kernel void tile_pair_grouped_backward_samples(
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
    device int* grad_ids [[buffer(13)]],
    device float* grad_ma [[buffer(14)]],
    device float* grad_q [[buffer(15)]],
    device float* grad_opacity [[buffer(16)]],
    device float* grad_color [[buffer(17)]],
    device int* grad_keys [[buffer(18)]],
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]]) {
  uint tile_id = gid / STAR_THREADS;
  uint local_tid = tid;
  if (tile_id >= uint(mi.tile_count)) return;

  threadgroup uint local_ids[STAR_TILE_CAPACITY];
  threadgroup float local_depths[STAR_TILE_CAPACITY];
  threadgroup uint unstable_flag;
  threadgroup float reduce_scratch[STAR_THREADS];

  for (uint slot = local_tid; slot < STAR_TILE_CAPACITY; slot += STAR_THREADS) {
    uint row = tile_id * STAR_TILE_CAPACITY + slot;
    grad_ids[row] = -1;
    grad_keys[row] = int(row);
    uint ma_base = row * 3u;
    grad_ma[ma_base + 0u] = 0.0f;
    grad_ma[ma_base + 1u] = 0.0f;
    grad_ma[ma_base + 2u] = 0.0f;
    uint q_base = row * 6u;
    grad_q[q_base + 0u] = 0.0f;
    grad_q[q_base + 1u] = 0.0f;
    grad_q[q_base + 2u] = 0.0f;
    grad_q[q_base + 3u] = 0.0f;
    grad_q[q_base + 4u] = 0.0f;
    grad_q[q_base + 5u] = 0.0f;
    grad_opacity[row] = 0.0f;
    uint color_base = row * 3u;
    grad_color[color_base + 0u] = 0.0f;
    grad_color[color_base + 1u] = 0.0f;
    grad_color[color_base + 2u] = 0.0f;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  uint raw_count = atomic_load_explicit(tile_counts + tile_id, memory_order_relaxed);
  uint count = min(raw_count, STAR_TILE_CAPACITY);
  if (count == 0u) return;

  for (uint i = local_tid; i < count; i += STAR_THREADS) {
    uint idx = tile_id * STAR_TILE_CAPACITY + i;
    local_ids[i] = tile_tube_ids[idx];
    local_depths[i] = tile_depths[idx];
  }
  sort_by_depth(local_ids, local_depths, count, local_tid);

  uint tx, ty, tz;
  decode_tile(tile_id, mi, tx, ty, tz);
  if (local_tid == 0u) {
    unstable_flag = tile_order_unstable(local_ids, count, ma, depth0, depth_beta, tx, ty, tz, mi) ? 1u : 0u;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool unstable = unstable_flag != 0u;
  if (local_tid == 0u && unstable) {
    atomic_store_explicit(tile_unstable + tile_id, 1u, memory_order_relaxed);
  }

  uint lt = local_tid / (STAR_TILE_X * STAR_TILE_Y);
  uint rem = local_tid - lt * STAR_TILE_X * STAR_TILE_Y;
  uint ly = rem / STAR_TILE_X;
  uint lx = rem - ly * STAR_TILE_X;
  uint f = tz * STAR_TILE_T + lt;
  uint y = ty * STAR_TILE_Y + ly;
  uint x = tx * STAR_TILE_X + lx;
  bool valid_pixel = f < uint(mi.frames) && y < uint(mi.height) && x < uint(mi.width);
  float3 sample_a = float3(float(x) + 0.5f, float(y) + 0.5f, frame_time(min(f, uint(mi.frames - 1)), mi));

  for (uint slot = 0u; slot < count; ++slot) {
    uint target_id = local_ids[slot];
    float3 grad_m_sum = float3(0.0f);
    float q_sum0 = 0.0f;
    float q_sum1 = 0.0f;
    float q_sum2 = 0.0f;
    float q_sum3 = 0.0f;
    float q_sum4 = 0.0f;
    float q_sum5 = 0.0f;
    float opacity_sum = 0.0f;
    float3 color_sum = float3(0.0f);

    if (valid_pixel) {
      uint ordered_ids[STAR_TILE_CAPACITY];
      float t_before[STAR_TILE_CAPACITY];
      float alpha_values[STAR_TILE_CAPACITY];
      bool processed[STAR_TILE_CAPACITY];
      bool differentiable_alpha[STAR_TILE_CAPACITY];

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

      float T = 1.0f;
      for (uint i = 0u; i < ordered_count; ++i) {
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
        if (tube_id != target_id) continue;

        color_sum += d_color;
        if (!differentiable_alpha[i]) break;
        float3 d = sample_a - load3(ma, tube_id);
        float qv = quadratic_q(q_uvt, tube_id, d);
        float exp_term = exp(-0.5f * qv);
        float grad_qv = -0.5f * alpha * d_alpha;
        float3 qd = load_q_row0(q_uvt, tube_id) * d.x + load_q_row1(q_uvt, tube_id) * d.y + load_q_row2(q_uvt, tube_id) * d.z;
        grad_m_sum += -2.0f * grad_qv * qd;
        q_sum0 += grad_qv * d.x * d.x;
        q_sum1 += grad_qv * 2.0f * d.x * d.y;
        q_sum2 += grad_qv * 2.0f * d.x * d.z;
        q_sum3 += grad_qv * d.y * d.y;
        q_sum4 += grad_qv * 2.0f * d.y * d.z;
        q_sum5 += grad_qv * d.z * d.z;
        opacity_sum += d_alpha * exp_term;
        break;
      }
    }

    float grad_m0 = reduce_threadgroup_sum(reduce_scratch, grad_m_sum.x, local_tid);
    float grad_m1 = reduce_threadgroup_sum(reduce_scratch, grad_m_sum.y, local_tid);
    float grad_m2 = reduce_threadgroup_sum(reduce_scratch, grad_m_sum.z, local_tid);
    float q0 = reduce_threadgroup_sum(reduce_scratch, q_sum0, local_tid);
    float q1 = reduce_threadgroup_sum(reduce_scratch, q_sum1, local_tid);
    float q2 = reduce_threadgroup_sum(reduce_scratch, q_sum2, local_tid);
    float q3 = reduce_threadgroup_sum(reduce_scratch, q_sum3, local_tid);
    float q4 = reduce_threadgroup_sum(reduce_scratch, q_sum4, local_tid);
    float q5 = reduce_threadgroup_sum(reduce_scratch, q_sum5, local_tid);
    float opacity_value = reduce_threadgroup_sum(reduce_scratch, opacity_sum, local_tid);
    float color0 = reduce_threadgroup_sum(reduce_scratch, color_sum.x, local_tid);
    float color1 = reduce_threadgroup_sum(reduce_scratch, color_sum.y, local_tid);
    float color2 = reduce_threadgroup_sum(reduce_scratch, color_sum.z, local_tid);

    if (local_tid == 0u) {
      uint row = tile_id * STAR_TILE_CAPACITY + slot;
      uint ma_base = row * 3u;
      grad_ma[ma_base + 0u] = grad_m0;
      grad_ma[ma_base + 1u] = grad_m1;
      grad_ma[ma_base + 2u] = grad_m2;
      uint q_base = row * 6u;
      grad_q[q_base + 0u] = q0;
      grad_q[q_base + 1u] = q1;
      grad_q[q_base + 2u] = q2;
      grad_q[q_base + 3u] = q3;
      grad_q[q_base + 4u] = q4;
      grad_q[q_base + 5u] = q5;
      grad_opacity[row] = opacity_value;
      uint color_base = row * 3u;
      grad_color[color_base + 0u] = color0;
      grad_color[color_base + 1u] = color1;
      grad_color[color_base + 2u] = color2;
      bool has_gradient =
          grad_m0 != 0.0f || grad_m1 != 0.0f || grad_m2 != 0.0f ||
          q0 != 0.0f || q1 != 0.0f || q2 != 0.0f ||
          q3 != 0.0f || q4 != 0.0f || q5 != 0.0f ||
          opacity_value != 0.0f ||
          color0 != 0.0f || color1 != 0.0f || color2 != 0.0f;
      if (has_gradient) {
        grad_ids[row] = int(target_id);
      }
    }
  }
}

kernel void tile_pair_sharedsort_backward_samples(
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
    device int* grad_ids [[buffer(13)]],
    device float* grad_ma [[buffer(14)]],
    device float* grad_q [[buffer(15)]],
    device float* grad_opacity [[buffer(16)]],
    device float* grad_color [[buffer(17)]],
    device int* grad_keys [[buffer(18)]],
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]]) {
  uint tile_id = gid / STAR_THREADS;
  uint local_tid = tid;
  if (tile_id >= uint(mi.tile_count)) return;

  threadgroup uint local_ids[STAR_TILE_CAPACITY];
  threadgroup float local_depths[STAR_TILE_CAPACITY];
  threadgroup uint unstable_flag;

  for (uint slot = local_tid; slot < STAR_TILE_CAPACITY; slot += STAR_THREADS) {
    uint row = tile_id * STAR_TILE_CAPACITY + slot;
    grad_ids[row] = -1;
    grad_keys[row] = int(row);
    uint ma_base = row * 3u;
    grad_ma[ma_base + 0u] = 0.0f;
    grad_ma[ma_base + 1u] = 0.0f;
    grad_ma[ma_base + 2u] = 0.0f;
    uint q_base = row * 6u;
    grad_q[q_base + 0u] = 0.0f;
    grad_q[q_base + 1u] = 0.0f;
    grad_q[q_base + 2u] = 0.0f;
    grad_q[q_base + 3u] = 0.0f;
    grad_q[q_base + 4u] = 0.0f;
    grad_q[q_base + 5u] = 0.0f;
    grad_opacity[row] = 0.0f;
    uint color_base = row * 3u;
    grad_color[color_base + 0u] = 0.0f;
    grad_color[color_base + 1u] = 0.0f;
    grad_color[color_base + 2u] = 0.0f;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  uint raw_count = atomic_load_explicit(tile_counts + tile_id, memory_order_relaxed);
  uint count = min(raw_count, STAR_TILE_CAPACITY);
  if (count == 0u) return;

  for (uint i = local_tid; i < count; i += STAR_THREADS) {
    uint idx = tile_id * STAR_TILE_CAPACITY + i;
    local_ids[i] = tile_tube_ids[idx];
    local_depths[i] = tile_depths[idx];
  }
  sort_by_depth(local_ids, local_depths, count, local_tid);

  uint tx, ty, tz;
  decode_tile(tile_id, mi, tx, ty, tz);
  if (local_tid == 0u) {
    unstable_flag = tile_order_unstable(local_ids, count, ma, depth0, depth_beta, tx, ty, tz, mi) ? 1u : 0u;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool unstable = unstable_flag != 0u;
  if (local_tid == 0u && unstable) {
    atomic_store_explicit(tile_unstable + tile_id, 1u, memory_order_relaxed);
  }

  for (uint slot = local_tid; slot < count; slot += STAR_THREADS) {
    uint row = tile_id * STAR_TILE_CAPACITY + slot;
    uint target_id = local_ids[slot];
    uint ordered_ids[STAR_TILE_CAPACITY];
    float t_before[STAR_TILE_CAPACITY];
    float alpha_values[STAR_TILE_CAPACITY];
    bool processed[STAR_TILE_CAPACITY];
    bool differentiable_alpha[STAR_TILE_CAPACITY];

    float3 grad_m_sum = float3(0.0f);
    float q_sum0 = 0.0f;
    float q_sum1 = 0.0f;
    float q_sum2 = 0.0f;
    float q_sum3 = 0.0f;
    float q_sum4 = 0.0f;
    float q_sum5 = 0.0f;
    float opacity_sum = 0.0f;
    float3 color_sum = float3(0.0f);

    for (uint lt = 0u; lt < STAR_TILE_T; ++lt) {
      uint f = tz * STAR_TILE_T + lt;
      if (f >= uint(mi.frames)) continue;
      for (uint ly = 0u; ly < STAR_TILE_Y; ++ly) {
        uint y = ty * STAR_TILE_Y + ly;
        if (y >= uint(mi.height)) continue;
        for (uint lx = 0u; lx < STAR_TILE_X; ++lx) {
          uint x = tx * STAR_TILE_X + lx;
          if (x >= uint(mi.width)) continue;
          float3 sample_a = float3(float(x) + 0.5f, float(y) + 0.5f, frame_time(f, mi));

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

          float T = 1.0f;
          for (uint i = 0u; i < ordered_count; ++i) {
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
            if (tube_id != target_id) continue;

            color_sum += d_color;
            if (!differentiable_alpha[i]) break;
            float3 d = sample_a - load3(ma, tube_id);
            float qv = quadratic_q(q_uvt, tube_id, d);
            float exp_term = exp(-0.5f * qv);
            float grad_qv = -0.5f * alpha * d_alpha;
            float3 qd = load_q_row0(q_uvt, tube_id) * d.x + load_q_row1(q_uvt, tube_id) * d.y + load_q_row2(q_uvt, tube_id) * d.z;
            grad_m_sum += -2.0f * grad_qv * qd;
            q_sum0 += grad_qv * d.x * d.x;
            q_sum1 += grad_qv * 2.0f * d.x * d.y;
            q_sum2 += grad_qv * 2.0f * d.x * d.z;
            q_sum3 += grad_qv * d.y * d.y;
            q_sum4 += grad_qv * 2.0f * d.y * d.z;
            q_sum5 += grad_qv * d.z * d.z;
            opacity_sum += d_alpha * exp_term;
            break;
          }
        }
      }
    }

    uint ma_base = row * 3u;
    grad_ma[ma_base + 0u] = grad_m_sum.x;
    grad_ma[ma_base + 1u] = grad_m_sum.y;
    grad_ma[ma_base + 2u] = grad_m_sum.z;
    uint q_base = row * 6u;
    grad_q[q_base + 0u] = q_sum0;
    grad_q[q_base + 1u] = q_sum1;
    grad_q[q_base + 2u] = q_sum2;
    grad_q[q_base + 3u] = q_sum3;
    grad_q[q_base + 4u] = q_sum4;
    grad_q[q_base + 5u] = q_sum5;
    grad_opacity[row] = opacity_sum;
    uint color_base = row * 3u;
    grad_color[color_base + 0u] = color_sum.x;
    grad_color[color_base + 1u] = color_sum.y;
    grad_color[color_base + 2u] = color_sum.z;
    bool has_gradient =
        grad_m_sum.x != 0.0f || grad_m_sum.y != 0.0f || grad_m_sum.z != 0.0f ||
        q_sum0 != 0.0f || q_sum1 != 0.0f || q_sum2 != 0.0f ||
        q_sum3 != 0.0f || q_sum4 != 0.0f || q_sum5 != 0.0f ||
        opacity_sum != 0.0f ||
        color_sum.x != 0.0f || color_sum.y != 0.0f || color_sum.z != 0.0f;
    if (has_gradient) {
      grad_ids[row] = int(target_id);
    }
  }
}

kernel void tile_pair_scanline_backward_samples(
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
    device int* grad_ids [[buffer(13)]],
    device float* grad_ma [[buffer(14)]],
    device float* grad_q [[buffer(15)]],
    device float* grad_opacity [[buffer(16)]],
    device float* grad_color [[buffer(17)]],
    device int* grad_keys [[buffer(18)]],
    uint gid [[thread_position_in_grid]]) {
  constexpr uint chunks_per_tile_slot = STAR_TILE_T * STAR_TILE_Y;
  uint rows_per_tile = STAR_TILE_CAPACITY * chunks_per_tile_slot;
  uint tile_id = gid / rows_per_tile;
  uint rem = gid - tile_id * rows_per_tile;
  uint slot = rem / chunks_per_tile_slot;
  uint chunk = rem - slot * chunks_per_tile_slot;
  uint lt = chunk / STAR_TILE_Y;
  uint ly = chunk - lt * STAR_TILE_Y;
  if (tile_id >= uint(mi.tile_count)) return;

  grad_ids[gid] = -1;
  uint local_tid_base = lt * STAR_TILE_X * STAR_TILE_Y + ly * STAR_TILE_X;
  grad_keys[gid] = int((tile_id * STAR_THREADS + local_tid_base) * STAR_TILE_CAPACITY + slot);
  uint ma_base = gid * 3u;
  grad_ma[ma_base + 0u] = 0.0f;
  grad_ma[ma_base + 1u] = 0.0f;
  grad_ma[ma_base + 2u] = 0.0f;
  uint q_base = gid * 6u;
  grad_q[q_base + 0u] = 0.0f;
  grad_q[q_base + 1u] = 0.0f;
  grad_q[q_base + 2u] = 0.0f;
  grad_q[q_base + 3u] = 0.0f;
  grad_q[q_base + 4u] = 0.0f;
  grad_q[q_base + 5u] = 0.0f;
  grad_opacity[gid] = 0.0f;
  uint color_base = gid * 3u;
  grad_color[color_base + 0u] = 0.0f;
  grad_color[color_base + 1u] = 0.0f;
  grad_color[color_base + 2u] = 0.0f;

  uint raw_count = atomic_load_explicit(tile_counts + tile_id, memory_order_relaxed);
  uint count = min(raw_count, STAR_TILE_CAPACITY);
  if (slot >= count) return;

  uint tx, ty, tz;
  decode_tile(tile_id, mi, tx, ty, tz);
  uint f = tz * STAR_TILE_T + lt;
  uint y = ty * STAR_TILE_Y + ly;
  if (f >= uint(mi.frames) || y >= uint(mi.height)) return;

  uint local_ids[STAR_TILE_CAPACITY];
  float local_depths[STAR_TILE_CAPACITY];
  for (uint i = 0u; i < count; ++i) {
    uint idx = tile_id * STAR_TILE_CAPACITY + i;
    local_ids[i] = tile_tube_ids[idx];
    local_depths[i] = tile_depths[idx];
  }
  sort_by_depth_thread(local_ids, local_depths, count);
  uint target_id = local_ids[slot];

  bool unstable = tile_order_unstable_thread(local_ids, count, ma, depth0, depth_beta, tx, ty, tz, mi);
  if (slot == 0u && chunk == 0u && unstable) {
    atomic_store_explicit(tile_unstable + tile_id, 1u, memory_order_relaxed);
  }

  float3 grad_m_sum = float3(0.0f);
  float q_sum0 = 0.0f;
  float q_sum1 = 0.0f;
  float q_sum2 = 0.0f;
  float q_sum3 = 0.0f;
  float q_sum4 = 0.0f;
  float q_sum5 = 0.0f;
  float opacity_sum = 0.0f;
  float3 color_sum = float3(0.0f);

  uint ordered_ids[STAR_TILE_CAPACITY];
  float t_before[STAR_TILE_CAPACITY];
  float alpha_values[STAR_TILE_CAPACITY];
  bool processed[STAR_TILE_CAPACITY];
  bool differentiable_alpha[STAR_TILE_CAPACITY];

  for (uint lx = 0u; lx < STAR_TILE_X; ++lx) {
    uint x = tx * STAR_TILE_X + lx;
    if (x >= uint(mi.width)) continue;
    float3 sample_a = float3(float(x) + 0.5f, float(y) + 0.5f, frame_time(f, mi));

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
        uint tube_id = select_sample_order_id_thread(local_ids, count, ma, depth0, depth_beta, sample_a, last_depth, last_id, selected_depth);
        if (tube_id == 0xFFFFFFFFu) break;
        ordered_ids[ordered_count] = tube_id;
        ordered_count += 1u;
        last_depth = selected_depth;
        last_id = tube_id;
      }
    }

    float T = 1.0f;
    for (uint i = 0u; i < ordered_count; ++i) {
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
      if (tube_id != target_id) continue;

      color_sum += d_color;
      if (!differentiable_alpha[i]) continue;
      float3 d = sample_a - load3(ma, tube_id);
      float qv = quadratic_q(q_uvt, tube_id, d);
      float exp_term = exp(-0.5f * qv);
      float grad_qv = -0.5f * alpha * d_alpha;
      float3 qd = load_q_row0(q_uvt, tube_id) * d.x + load_q_row1(q_uvt, tube_id) * d.y + load_q_row2(q_uvt, tube_id) * d.z;
      float3 grad_m_value = -2.0f * grad_qv * qd;
      grad_m_sum += grad_m_value;
      q_sum0 += grad_qv * d.x * d.x;
      q_sum1 += grad_qv * 2.0f * d.x * d.y;
      q_sum2 += grad_qv * 2.0f * d.x * d.z;
      q_sum3 += grad_qv * d.y * d.y;
      q_sum4 += grad_qv * 2.0f * d.y * d.z;
      q_sum5 += grad_qv * d.z * d.z;
      opacity_sum += d_alpha * exp_term;
      break;
    }
  }

  grad_ma[ma_base + 0u] = grad_m_sum.x;
  grad_ma[ma_base + 1u] = grad_m_sum.y;
  grad_ma[ma_base + 2u] = grad_m_sum.z;
  grad_q[q_base + 0u] = q_sum0;
  grad_q[q_base + 1u] = q_sum1;
  grad_q[q_base + 2u] = q_sum2;
  grad_q[q_base + 3u] = q_sum3;
  grad_q[q_base + 4u] = q_sum4;
  grad_q[q_base + 5u] = q_sum5;
  grad_opacity[gid] = opacity_sum;
  grad_color[color_base + 0u] = color_sum.x;
  grad_color[color_base + 1u] = color_sum.y;
  grad_color[color_base + 2u] = color_sum.z;
  bool has_gradient =
      grad_m_sum.x != 0.0f || grad_m_sum.y != 0.0f || grad_m_sum.z != 0.0f ||
      q_sum0 != 0.0f || q_sum1 != 0.0f || q_sum2 != 0.0f ||
      q_sum3 != 0.0f || q_sum4 != 0.0f || q_sum5 != 0.0f ||
      opacity_sum != 0.0f ||
      color_sum.x != 0.0f || color_sum.y != 0.0f || color_sum.z != 0.0f;
  if (has_gradient) {
    grad_ids[gid] = int(target_id);
  }
}

kernel void tile_pair_backward_samples(
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
    device int* grad_ids [[buffer(13)]],
    device float* grad_ma [[buffer(14)]],
    device float* grad_q [[buffer(15)]],
    device float* grad_opacity [[buffer(16)]],
    device float* grad_color [[buffer(17)]],
    device int* grad_keys [[buffer(18)]],
    uint gid [[thread_position_in_grid]]) {
  tile_pair_backward_samples_impl(
      ma, q_uvt, depth0, depth_beta, opacity, color, grad_image, mi, mf,
      tile_counts, tile_tube_ids, tile_depths, tile_unstable, grad_ids,
      grad_ma, grad_q, grad_opacity, grad_color, grad_keys, gid, false, false, false);
}

kernel void tile_pair_backward_samples_compensated(
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
    device int* grad_ids [[buffer(13)]],
    device float* grad_ma [[buffer(14)]],
    device float* grad_q [[buffer(15)]],
    device float* grad_opacity [[buffer(16)]],
    device float* grad_color [[buffer(17)]],
    device int* grad_keys [[buffer(18)]],
    uint gid [[thread_position_in_grid]]) {
  tile_pair_backward_samples_impl(
      ma, q_uvt, depth0, depth_beta, opacity, color, grad_image, mi, mf,
      tile_counts, tile_tube_ids, tile_depths, tile_unstable, grad_ids,
      grad_ma, grad_q, grad_opacity, grad_color, grad_keys, gid, true, false, false);
}

kernel void tile_pair_target_bounds_backward_samples(
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
    device int* grad_ids [[buffer(13)]],
    device float* grad_ma [[buffer(14)]],
    device float* grad_q [[buffer(15)]],
    device float* grad_opacity [[buffer(16)]],
    device float* grad_color [[buffer(17)]],
    device int* grad_keys [[buffer(18)]],
    uint gid [[thread_position_in_grid]]) {
  tile_pair_backward_samples_impl(
      ma, q_uvt, depth0, depth_beta, opacity, color, grad_image, mi, mf,
      tile_counts, tile_tube_ids, tile_depths, tile_unstable, grad_ids,
      grad_ma, grad_q, grad_opacity, grad_color, grad_keys, gid, false, true, false);
}

kernel void tile_pair_suffix_backward_samples(
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
    device int* grad_ids [[buffer(13)]],
    device float* grad_ma [[buffer(14)]],
    device float* grad_q [[buffer(15)]],
    device float* grad_opacity [[buffer(16)]],
    device float* grad_color [[buffer(17)]],
    device int* grad_keys [[buffer(18)]],
    uint gid [[thread_position_in_grid]]) {
  tile_pair_backward_samples_impl(
      ma, q_uvt, depth0, depth_beta, opacity, color, grad_image, mi, mf,
      tile_counts, tile_tube_ids, tile_depths, tile_unstable, grad_ids,
      grad_ma, grad_q, grad_opacity, grad_color, grad_keys, gid, false, false, true);
}

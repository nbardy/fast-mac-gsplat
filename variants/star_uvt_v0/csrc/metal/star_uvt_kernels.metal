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
#ifndef STAR_SIMD_WIDTH
#define STAR_SIMD_WIDTH 32u
#endif
#ifndef STAR_SIMDGROUPS
#define STAR_SIMDGROUPS 4u
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
#ifndef STAR_FEATURE_GRAD_CACHE_CAP
#define STAR_FEATURE_GRAD_CACHE_CAP 64u
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
  // 0: historical peak-splat alpha; 1: Beer-Lambert optical thickness.
  float alpha_mode;
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

inline uint feature_dim(constant MetaI32& mi) {
  return uint(max(mi.reserved0, 0));
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

inline bool beer_lambert_alpha(constant MetaF32& mf) {
  return mf.alpha_mode > 0.5f;
}

inline float primitive_alpha_raw(
    float opacity_or_optical_thickness,
    float gaussian_density,
    constant MetaF32& mf) {
  float optical_depth = opacity_or_optical_thickness * gaussian_density;
  return beer_lambert_alpha(mf) ? 1.0f - exp(-optical_depth) : optical_depth;
}

inline float primitive_alpha_d_opacity(
    float opacity_or_optical_thickness,
    float gaussian_density,
    constant MetaF32& mf) {
  if (!beer_lambert_alpha(mf)) return gaussian_density;
  float optical_depth = opacity_or_optical_thickness * gaussian_density;
  return exp(-optical_depth) * gaussian_density;
}

inline float primitive_alpha_d_qv(
    float opacity_or_optical_thickness,
    float gaussian_density,
    constant MetaF32& mf) {
  float optical_depth = opacity_or_optical_thickness * gaussian_density;
  if (!beer_lambert_alpha(mf)) return -0.5f * optical_depth;
  return -0.5f * optical_depth * exp(-optical_depth);
}

inline float primitive_support_numerator(constant MetaF32& mf) {
  return beer_lambert_alpha(mf)
      ? -log(max(1.0f - mf.alpha_threshold, mf.eps))
      : mf.alpha_threshold;
}

inline void atomic_add3(device atomic_float* ptr, uint base, float3 value) {
  atomic_fetch_add_explicit(&ptr[base + 0u], value.x, memory_order_relaxed);
  atomic_fetch_add_explicit(&ptr[base + 1u], value.y, memory_order_relaxed);
  atomic_fetch_add_explicit(&ptr[base + 2u], value.z, memory_order_relaxed);
}

inline void simd_reduced_atomic_add(device atomic_float* ptr, float value, uint simd_lane) {
  float total = simd_sum(value);
  if (simd_lane == 0u) {
    atomic_fetch_add_explicit(ptr, total, memory_order_relaxed);
  }
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

inline bool tube_active_for_frame(const device int* active_start, const device int* active_stop, uint tube_id, uint frame) {
  int f = int(frame);
  return active_start[tube_id] <= f && f < active_stop[tube_id];
}

kernel void projective_trace_eval(
    const device float* coeffs [[buffer(0)]],
    const device float* times [[buffer(1)]],
    const device int* meta_i32 [[buffer(2)]],
    const device float* meta_f32 [[buffer(3)]],
    device float* out [[buffer(4)]],
    uint gid [[thread_position_in_grid]]) {
  uint tube_count = uint(meta_i32[0]);
  uint time_count = uint(meta_i32[1]);
  uint total = tube_count * time_count;
  if (gid >= total || time_count == 0u) {
    return;
  }

  uint tube_id = gid / time_count;
  uint time_id = gid - tube_id * time_count;
  float t = times[time_id];
  float t2 = t * t;
  uint c = tube_id * 9u;
  float hu = coeffs[c + 0u] + coeffs[c + 1u] * t + coeffs[c + 2u] * t2;
  float hv = coeffs[c + 3u] + coeffs[c + 4u] * t + coeffs[c + 5u] * t2;
  float hz = coeffs[c + 6u] + coeffs[c + 7u] * t + coeffs[c + 8u] * t2;
  float eps = meta_f32[0];
  float abs_hz = fabs(hz);
  bool finite = isfinite(hu) && isfinite(hv) && isfinite(hz);
  bool valid = finite && abs_hz > eps;
  float inv_hz = valid ? 1.0f / hz : 0.0f;
  float valid_sign = valid ? (hz > 0.0f ? 1.0f : -1.0f) : 0.0f;
  uint o = gid * 4u;
  out[o + 0u] = valid ? hu * inv_hz : 0.0f;
  out[o + 1u] = valid ? hv * inv_hz : 0.0f;
  out[o + 2u] = hz;
  out[o + 3u] = valid_sign;
}

inline float family_coeff_at(
    const device float* family_coeffs,
    const device float* q_basis,
    uint tube_id,
    uint coeff_id,
    uint q_id,
    uint basis_count) {
  uint coeff_base = (tube_id * 9u + coeff_id) * basis_count;
  uint q_base = q_id * basis_count;
  float value = 0.0f;
  for (uint b = 0u; b < basis_count; ++b) {
    value += family_coeffs[coeff_base + b] * q_basis[q_base + b];
  }
  return value;
}

kernel void projective_trace_family_eval(
    const device float* family_coeffs [[buffer(0)]],
    const device float* q_basis [[buffer(1)]],
    const device float* times [[buffer(2)]],
    const device int* meta_i32 [[buffer(3)]],
    const device float* meta_f32 [[buffer(4)]],
    device float* out [[buffer(5)]],
    uint gid [[thread_position_in_grid]]) {
  uint tube_count = uint(meta_i32[0]);
  uint q_count = uint(meta_i32[1]);
  uint time_count = uint(meta_i32[2]);
  uint basis_count = uint(meta_i32[3]);
  uint total = tube_count * q_count * time_count;
  if (gid >= total || time_count == 0u || basis_count == 0u) {
    return;
  }

  uint time_id = gid % time_count;
  uint trace_index = gid / time_count;
  uint tube_id = trace_index % tube_count;
  uint q_id = trace_index / tube_count;
  float t = times[time_id];
  float t2 = t * t;
  float hu = family_coeff_at(family_coeffs, q_basis, tube_id, 0u, q_id, basis_count)
      + family_coeff_at(family_coeffs, q_basis, tube_id, 1u, q_id, basis_count) * t
      + family_coeff_at(family_coeffs, q_basis, tube_id, 2u, q_id, basis_count) * t2;
  float hv = family_coeff_at(family_coeffs, q_basis, tube_id, 3u, q_id, basis_count)
      + family_coeff_at(family_coeffs, q_basis, tube_id, 4u, q_id, basis_count) * t
      + family_coeff_at(family_coeffs, q_basis, tube_id, 5u, q_id, basis_count) * t2;
  float hz = family_coeff_at(family_coeffs, q_basis, tube_id, 6u, q_id, basis_count)
      + family_coeff_at(family_coeffs, q_basis, tube_id, 7u, q_id, basis_count) * t
      + family_coeff_at(family_coeffs, q_basis, tube_id, 8u, q_id, basis_count) * t2;
  float eps = meta_f32[0];
  bool finite = isfinite(hu) && isfinite(hv) && isfinite(hz);
  bool valid = finite && fabs(hz) > eps;
  float inv_hz = valid ? 1.0f / hz : 0.0f;
  float valid_sign = valid ? (hz > 0.0f ? 1.0f : -1.0f) : 0.0f;
  uint o = gid * 4u;
  out[o + 0u] = valid ? hu * inv_hz : 0.0f;
  out[o + 1u] = valid ? hv * inv_hz : 0.0f;
  out[o + 2u] = hz;
  out[o + 3u] = valid_sign;
}

kernel void projective_trace_family_backward(
    const device float* family_coeffs [[buffer(0)]],
    const device float* q_basis [[buffer(1)]],
    const device float* times [[buffer(2)]],
    const device float* grad_out [[buffer(3)]],
    const device int* meta_i32 [[buffer(4)]],
    const device float* meta_f32 [[buffer(5)]],
    device atomic_float* grad_family_coeffs [[buffer(6)]],
    device atomic_float* grad_q_basis [[buffer(7)]],
    uint gid [[thread_position_in_grid]]) {
  uint tube_count = uint(meta_i32[0]);
  uint q_count = uint(meta_i32[1]);
  uint time_count = uint(meta_i32[2]);
  uint basis_count = uint(meta_i32[3]);
  uint total = tube_count * q_count * time_count;
  if (gid >= total || time_count == 0u || basis_count == 0u) {
    return;
  }

  uint time_id = gid % time_count;
  uint trace_index = gid / time_count;
  uint tube_id = trace_index % tube_count;
  uint q_id = trace_index / tube_count;
  float t = times[time_id];
  float t2 = t * t;
  float time_basis[3] = {1.0f, t, t2};
  float coeff_values[9];
  for (uint k = 0u; k < 9u; ++k) {
    coeff_values[k] = family_coeff_at(family_coeffs, q_basis, tube_id, k, q_id, basis_count);
  }
  float hu = coeff_values[0] + coeff_values[1] * t + coeff_values[2] * t2;
  float hv = coeff_values[3] + coeff_values[4] * t + coeff_values[5] * t2;
  float hz = coeff_values[6] + coeff_values[7] * t + coeff_values[8] * t2;
  if (!(isfinite(hu) && isfinite(hv) && isfinite(hz))) {
    return;
  }

  uint out_base = gid * 4u;
  float grad_u = grad_out[out_base + 0u];
  float grad_v = grad_out[out_base + 1u];
  float grad_depth = grad_out[out_base + 2u];
  float eps = meta_f32[0];
  bool valid = fabs(hz) > eps;
  float inv_hz = valid ? 1.0f / hz : 0.0f;
  float grad_hu = valid ? grad_u * inv_hz : 0.0f;
  float grad_hv = valid ? grad_v * inv_hz : 0.0f;
  float grad_hz = grad_depth;
  if (valid) {
    grad_hz += grad_u * (-hu * inv_hz * inv_hz) + grad_v * (-hv * inv_hz * inv_hz);
  }

  float grad_coeff[9];
  grad_coeff[0] = grad_hu * time_basis[0];
  grad_coeff[1] = grad_hu * time_basis[1];
  grad_coeff[2] = grad_hu * time_basis[2];
  grad_coeff[3] = grad_hv * time_basis[0];
  grad_coeff[4] = grad_hv * time_basis[1];
  grad_coeff[5] = grad_hv * time_basis[2];
  grad_coeff[6] = grad_hz * time_basis[0];
  grad_coeff[7] = grad_hz * time_basis[1];
  grad_coeff[8] = grad_hz * time_basis[2];

  uint q_base = q_id * basis_count;
  for (uint b = 0u; b < basis_count; ++b) {
    float qb = q_basis[q_base + b];
    float grad_qb = 0.0f;
    for (uint k = 0u; k < 9u; ++k) {
      uint family_index = (tube_id * 9u + k) * basis_count + b;
      float g = grad_coeff[k];
      atomic_fetch_add_explicit(&grad_family_coeffs[family_index], g * qb, memory_order_relaxed);
      grad_qb += g * family_coeffs[family_index];
    }
    atomic_fetch_add_explicit(&grad_q_basis[q_base + b], grad_qb, memory_order_relaxed);
  }
}

inline bool eval_projective_trace_point(
    const device float* coeffs,
    uint tube_id,
    float t,
    float eps,
    thread float& u,
    thread float& v,
    thread float& depth) {
  float t2 = t * t;
  uint c = tube_id * 9u;
  float hu = coeffs[c + 0u] + coeffs[c + 1u] * t + coeffs[c + 2u] * t2;
  float hv = coeffs[c + 3u] + coeffs[c + 4u] * t + coeffs[c + 5u] * t2;
  float hz = coeffs[c + 6u] + coeffs[c + 7u] * t + coeffs[c + 8u] * t2;
  bool valid = isfinite(hu) && isfinite(hv) && isfinite(hz) && fabs(hz) > eps;
  if (!valid) {
    u = 0.0f;
    v = 0.0f;
    depth = hz;
    return false;
  }
  float inv_hz = 1.0f / hz;
  u = hu * inv_hz;
  v = hv * inv_hz;
  depth = hz;
  return true;
}

inline void composite_projective_trace(
    uint tube_id,
    float2 pixel_center,
    float t,
    float sigma_px,
    const device float* coeffs,
    const device float* opacity,
    const device float* color,
    constant MetaF32& mf,
    thread float3& accum,
    thread float& transmittance) {
  float u;
  float v;
  float depth;
  if (!eval_projective_trace_point(coeffs, tube_id, t, mf.eps, u, v, depth)) return;
  float2 d = pixel_center - float2(u, v);
  float inv_sigma2 = 1.0f / max(sigma_px * sigma_px, mf.eps);
  float density = exp(-0.5f * dot(d, d) * inv_sigma2);
  float alpha = min(mf.max_alpha, primitive_alpha_raw(opacity[tube_id], density, mf));
  if (!(alpha >= mf.alpha_threshold)) return;
  float w = transmittance * alpha;
  accum += w * load3(color, tube_id);
  transmittance *= (1.0f - alpha);
}

inline uint select_projective_order_id(
    threadgroup uint* ids,
    threadgroup int* active_start,
    threadgroup int* active_stop,
    uint count,
    const device float* coeffs,
    uint frame,
    float t,
    float eps,
    float last_depth,
    uint last_id,
    thread float& out_depth) {
  uint best_id = 0xFFFFFFFFu;
  float best_depth = INFINITY;
  for (uint i = 0u; i < count; ++i) {
    uint tube_id = ids[i];
    if (tube_id == 0xFFFFFFFFu) continue;
    int f = int(frame);
    if (active_start[i] > f || f >= active_stop[i]) continue;
    float u;
    float v;
    float depth;
    if (!eval_projective_trace_point(coeffs, tube_id, t, eps, u, v, depth)) continue;
    bool after_last = (depth > last_depth) || (depth == last_depth && tube_id > last_id);
    bool better = (depth < best_depth) || (depth == best_depth && tube_id < best_id);
    if (after_last && better) {
      best_depth = depth;
      best_id = tube_id;
    }
  }
  out_depth = best_depth;
  return best_id;
}

inline bool eval_projective_cell_trace_point(
    const device float* coeffs,
    uint trace_id,
    float t,
    thread float& u,
    thread float& v,
    thread float& depth) {
  float t2 = t * t;
  uint c = trace_id * 9u;
  u = coeffs[c + 0u] + coeffs[c + 1u] * t + coeffs[c + 2u] * t2;
  v = coeffs[c + 3u] + coeffs[c + 4u] * t + coeffs[c + 5u] * t2;
  depth = coeffs[c + 6u] + coeffs[c + 7u] * t + coeffs[c + 8u] * t2;
  bool valid = isfinite(u) && isfinite(v) && isfinite(depth);
  if (!valid) {
    u = 0.0f;
    v = 0.0f;
    depth = 0.0f;
  }
  return valid;
}

inline float projective_cell_depth_at_pixel(
    const device float* depth_affine_uv,
    uint trace_id,
    float2 pixel_center,
    float t,
    float center_u,
    float center_v,
    float center_depth) {
  float t2 = t * t;
  uint b = trace_id * 6u;
  float slope_u = depth_affine_uv[b + 0u] + depth_affine_uv[b + 1u] * t + depth_affine_uv[b + 2u] * t2;
  float slope_v = depth_affine_uv[b + 3u] + depth_affine_uv[b + 4u] * t + depth_affine_uv[b + 5u] * t2;
  return center_depth + slope_u * (pixel_center.x - center_u) + slope_v * (pixel_center.y - center_v);
}

inline void composite_projective_cell_trace(
    uint trace_id,
    float2 pixel_center,
    float t,
    float sigma_px,
    const device float* coeffs,
    const device float* opacity,
    const device float* color,
    constant MetaF32& mf,
    thread float3& accum,
    thread float& transmittance) {
  float u;
  float v;
  float depth;
  if (!eval_projective_cell_trace_point(coeffs, trace_id, t, u, v, depth)) return;
  float2 d = pixel_center - float2(u, v);
  float inv_sigma2 = 1.0f / max(sigma_px * sigma_px, mf.eps);
  float density = exp(-0.5f * dot(d, d) * inv_sigma2);
  float alpha = min(mf.max_alpha, primitive_alpha_raw(opacity[trace_id], density, mf));
  if (!(alpha >= mf.alpha_threshold)) return;
  float w = transmittance * alpha;
  accum += w * load3(color, trace_id);
  transmittance *= (1.0f - alpha);
}

inline float projective_cell_opacity_time_scale(
    const device float* opacity_time_coeffs,
    uint trace_id,
    float t) {
  float t2 = t * t;
  uint c = trace_id * 3u;
  float qv = opacity_time_coeffs[c + 0u] + opacity_time_coeffs[c + 1u] * t + opacity_time_coeffs[c + 2u] * t2;
  return exp(-0.5f * qv);
}

inline float projective_cell_precision_radius2(
    const device float* spatial_precision_uv,
    uint trace_id,
    float2 d) {
  uint p = trace_id * 3u;
  float q_uu = spatial_precision_uv[p + 0u];
  float q_uv = spatial_precision_uv[p + 1u];
  float q_vv = spatial_precision_uv[p + 2u];
  return q_uu * d.x * d.x + 2.0f * q_uv * d.x * d.y + q_vv * d.y * d.y;
}

inline float2 projective_cell_precision_center_grad(
    const device float* spatial_precision_uv,
    uint trace_id,
    float2 d) {
  uint p = trace_id * 3u;
  float q_uu = spatial_precision_uv[p + 0u];
  float q_uv = spatial_precision_uv[p + 1u];
  float q_vv = spatial_precision_uv[p + 2u];
  return float2(q_uu * d.x + q_uv * d.y, q_uv * d.x + q_vv * d.y);
}

inline void composite_projective_cell_trace_with_time_opacity(
    uint trace_id,
    float2 pixel_center,
    float t,
    const device float* coeffs,
    const device float* opacity,
    const device float* opacity_time_coeffs,
    const device float* spatial_precision_uv,
    const device float* color,
    constant MetaF32& mf,
    thread float3& accum,
    thread float& transmittance) {
  float u;
  float v;
  float depth;
  if (!eval_projective_cell_trace_point(coeffs, trace_id, t, u, v, depth)) return;
  float2 d = pixel_center - float2(u, v);
  float time_scale = projective_cell_opacity_time_scale(opacity_time_coeffs, trace_id, t);
  float radius2 = projective_cell_precision_radius2(spatial_precision_uv, trace_id, d);
  float density = time_scale * exp(-0.5f * radius2);
  float alpha = min(mf.max_alpha, primitive_alpha_raw(opacity[trace_id], density, mf));
  if (!(alpha >= mf.alpha_threshold)) return;
  float w = transmittance * alpha;
  accum += w * load3(color, trace_id);
  transmittance *= (1.0f - alpha);
}

inline uint select_projective_cell_order_id(
    threadgroup uint* ids,
    threadgroup int* active_start,
    threadgroup int* active_stop,
    uint count,
    const device float* coeffs,
    uint frame,
    float t,
    float last_depth,
    uint last_id,
    thread float& out_depth) {
  uint best_id = 0xFFFFFFFFu;
  float best_depth = INFINITY;
  for (uint i = 0u; i < count; ++i) {
    uint trace_id = ids[i];
    if (trace_id == 0xFFFFFFFFu) continue;
    int f = int(frame);
    if (active_start[i] > f || f >= active_stop[i]) continue;
    float u;
    float v;
    float depth;
    if (!eval_projective_cell_trace_point(coeffs, trace_id, t, u, v, depth)) continue;
    bool after_last = (depth > last_depth) || (depth == last_depth && trace_id > last_id);
    bool better = (depth < best_depth) || (depth == best_depth && trace_id < best_id);
    if (after_last && better) {
      best_depth = depth;
      best_id = trace_id;
    }
  }
  out_depth = best_depth;
  return best_id;
}

inline uint select_projective_cell_order_id_interval(
    const device int* tile_trace_ids,
    const device int* tile_active_start,
    const device int* tile_active_stop,
    uint tile_id,
    uint count,
    const device float* coeffs,
    const device float* depth_affine_uv,
    float2 pixel_center,
    uint frame,
    float t,
    float last_depth,
    uint last_id,
    thread float& out_depth) {
  uint best_id = 0xFFFFFFFFu;
  float best_depth = INFINITY;
  uint row_base = tile_id * STAR_TILE_CAPACITY;
  for (uint i = 0u; i < count; ++i) {
    uint row = row_base + i;
    int raw_trace_id = tile_trace_ids[row];
    if (raw_trace_id < 0) continue;
    int f = int(frame);
    if (tile_active_start[row] > f || f >= tile_active_stop[row]) continue;
    uint trace_id = uint(raw_trace_id);
    float u;
    float v;
    float depth;
    if (!eval_projective_cell_trace_point(coeffs, trace_id, t, u, v, depth)) continue;
    depth = projective_cell_depth_at_pixel(depth_affine_uv, trace_id, pixel_center, t, u, v, depth);
    bool after_last = (depth > last_depth) || (depth == last_depth && trace_id > last_id);
    bool better = (depth < best_depth) || (depth == best_depth && trace_id < best_id);
    if (after_last && better) {
      best_depth = depth;
      best_id = trace_id;
    }
  }
  out_depth = best_depth;
  return best_id;
}

inline bool eval_projective_family_cell_trace_point(
    const device float* family_coeffs,
    const device float* q_basis,
    uint global_trace_id,
    uint base_trace_count,
    uint basis_count,
    float t,
    thread float& u,
    thread float& v,
    thread float& depth) {
  if (base_trace_count == 0u || basis_count == 0u) {
    u = 0.0f;
    v = 0.0f;
    depth = 0.0f;
    return false;
  }
  uint base_trace_id = global_trace_id % base_trace_count;
  uint q_id = global_trace_id / base_trace_count;
  float t2 = t * t;
  float c0 = family_coeff_at(family_coeffs, q_basis, base_trace_id, 0u, q_id, basis_count);
  float c1 = family_coeff_at(family_coeffs, q_basis, base_trace_id, 1u, q_id, basis_count);
  float c2 = family_coeff_at(family_coeffs, q_basis, base_trace_id, 2u, q_id, basis_count);
  float c3 = family_coeff_at(family_coeffs, q_basis, base_trace_id, 3u, q_id, basis_count);
  float c4 = family_coeff_at(family_coeffs, q_basis, base_trace_id, 4u, q_id, basis_count);
  float c5 = family_coeff_at(family_coeffs, q_basis, base_trace_id, 5u, q_id, basis_count);
  float c6 = family_coeff_at(family_coeffs, q_basis, base_trace_id, 6u, q_id, basis_count);
  float c7 = family_coeff_at(family_coeffs, q_basis, base_trace_id, 7u, q_id, basis_count);
  float c8 = family_coeff_at(family_coeffs, q_basis, base_trace_id, 8u, q_id, basis_count);
  u = c0 + c1 * t + c2 * t2;
  v = c3 + c4 * t + c5 * t2;
  depth = c6 + c7 * t + c8 * t2;
  bool valid = isfinite(u) && isfinite(v) && isfinite(depth);
  if (!valid) {
    u = 0.0f;
    v = 0.0f;
    depth = 0.0f;
  }
  return valid;
}

inline float projective_family_cell_depth_at_pixel(
    const device float* depth_affine_uv,
    uint global_trace_id,
    uint base_trace_count,
    float2 pixel_center,
    float t,
    float center_u,
    float center_v,
    float center_depth) {
  uint base_trace_id = global_trace_id % base_trace_count;
  return projective_cell_depth_at_pixel(depth_affine_uv, base_trace_id, pixel_center, t, center_u, center_v, center_depth);
}

inline void composite_projective_family_cell_trace_with_time_opacity(
    uint global_trace_id,
    uint base_trace_count,
    uint basis_count,
    float2 pixel_center,
    float t,
    const device float* family_coeffs,
    const device float* q_basis,
    const device float* opacity,
    const device float* opacity_time_coeffs,
    const device float* spatial_precision_uv,
    const device float* color,
    constant MetaF32& mf,
    thread float3& accum,
    thread float& transmittance) {
  float u;
  float v;
  float depth;
  if (!eval_projective_family_cell_trace_point(
          family_coeffs, q_basis, global_trace_id, base_trace_count, basis_count, t, u, v, depth)) return;
  uint base_trace_id = global_trace_id % base_trace_count;
  float2 d = pixel_center - float2(u, v);
  float time_scale = projective_cell_opacity_time_scale(opacity_time_coeffs, base_trace_id, t);
  float radius2 = projective_cell_precision_radius2(spatial_precision_uv, base_trace_id, d);
  float density = time_scale * exp(-0.5f * radius2);
  float alpha = min(mf.max_alpha, primitive_alpha_raw(opacity[base_trace_id], density, mf));
  if (!(alpha >= mf.alpha_threshold)) return;
  float w = transmittance * alpha;
  accum += w * load3(color, base_trace_id);
  transmittance *= (1.0f - alpha);
}

inline uint select_projective_family_cell_order_id_interval(
    const device int* tile_trace_ids,
    const device int* tile_active_start,
    const device int* tile_active_stop,
    uint tile_id,
    uint count,
    const device float* family_coeffs,
    const device float* q_basis,
    const device float* depth_affine_uv,
    uint base_trace_count,
    uint basis_count,
    float2 pixel_center,
    uint frame,
    float t,
    float last_depth,
    uint last_id,
    thread float& out_depth) {
  uint best_id = 0xFFFFFFFFu;
  float best_depth = INFINITY;
  uint row_base = tile_id * STAR_TILE_CAPACITY;
  for (uint i = 0u; i < count; ++i) {
    uint row = row_base + i;
    int raw_trace_id = tile_trace_ids[row];
    if (raw_trace_id < 0) continue;
    int f = int(frame);
    if (tile_active_start[row] > f || f >= tile_active_stop[row]) continue;
    uint trace_id = uint(raw_trace_id);
    float u;
    float v;
    float depth;
    if (!eval_projective_family_cell_trace_point(
            family_coeffs, q_basis, trace_id, base_trace_count, basis_count, t, u, v, depth)) continue;
    depth = projective_family_cell_depth_at_pixel(depth_affine_uv, trace_id, base_trace_count, pixel_center, t, u, v, depth);
    bool after_last = (depth > last_depth) || (depth == last_depth && trace_id > last_id);
    bool better = (depth < best_depth) || (depth == best_depth && trace_id < best_id);
    if (after_last && better) {
      best_depth = depth;
      best_id = trace_id;
    }
  }
  out_depth = best_depth;
  return best_id;
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
  float det_eps = max(eps * eps, 1.0e-20f);
  if (!isfinite(det) || fabs(det) <= det_eps) {
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
  float support_numerator = primitive_support_numerator(mf);
  if (!(op > support_numerator)) return out;
  float tau = -2.0f * log(max(support_numerator / max(op, mf.eps), mf.eps));
  if (!isfinite(tau) || tau <= 0.0f) return out;

  float3 inv_diag;
  bool ok = inverse_sym3_diag(q, tube_id, mf.eps, inv_diag);
  float3 m = load3(ma, tube_id);
  float3 half_extent;
  if (ok) {
    half_extent = sqrt(max(tau * inv_diag, float3(0.0f)));
  } else {
    half_extent = abs(m) + float3(float(mi.width), float(mi.height), float(mi.frames));
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

inline void reduce_atomic_add_feature_grads_cached(
    device atomic_float* grad_feature,
    const thread float* grad_feature_cache,
    uint tube_id,
    float feature_scale,
    bool pixel_active,
    constant MetaI32& mi,
    uint simd_lane,
    uint simd_group,
    threadgroup float* partial_features) {
  uint fdim = feature_dim(mi);
  uint tube_feature_base = tube_id * fdim;
  for (uint c = 0u; c < fdim; ++c) {
    float contribution = pixel_active ? grad_feature_cache[c] * feature_scale : 0.0f;
    float sg_sum = simd_sum(contribution);
    if (simd_lane == 0u) {
      partial_features[simd_group * STAR_FEATURE_GRAD_CACHE_CAP + c] = sg_sum;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_group == 0u) {
    for (uint c = simd_lane; c < fdim; c += STAR_SIMD_WIDTH) {
      float total = 0.0f;
      for (uint sg = 0u; sg < STAR_SIMDGROUPS; ++sg) {
        total += partial_features[sg * STAR_FEATURE_GRAD_CACHE_CAP + c];
      }
      atomic_fetch_add_explicit(&grad_feature[tube_feature_base + c], total, memory_order_relaxed);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
}

inline void reduce_atomic_add_feature_grads_cached_vec4(
    device atomic_float* grad_feature,
    const thread float* grad_feature_cache,
    uint tube_id,
    float feature_scale,
    bool pixel_active,
    constant MetaI32& mi,
    uint simd_lane,
    uint simd_group,
    threadgroup float* partial_features) {
  uint fdim = feature_dim(mi);
  uint tube_feature_base = tube_id * fdim;
  for (uint c = 0u; c < fdim; c += 4u) {
    float4 grad_v = float4(
        grad_feature_cache[c + 0u],
        (c + 1u < fdim) ? grad_feature_cache[c + 1u] : 0.0f,
        (c + 2u < fdim) ? grad_feature_cache[c + 2u] : 0.0f,
        (c + 3u < fdim) ? grad_feature_cache[c + 3u] : 0.0f);
    float4 contribution = pixel_active ? grad_v * feature_scale : float4(0.0f);
    float4 sg_sum = simd_sum(contribution);
    if (simd_lane == 0u) {
      partial_features[simd_group * STAR_FEATURE_GRAD_CACHE_CAP + c + 0u] = sg_sum.x;
      if (c + 1u < fdim) partial_features[simd_group * STAR_FEATURE_GRAD_CACHE_CAP + c + 1u] = sg_sum.y;
      if (c + 2u < fdim) partial_features[simd_group * STAR_FEATURE_GRAD_CACHE_CAP + c + 2u] = sg_sum.z;
      if (c + 3u < fdim) partial_features[simd_group * STAR_FEATURE_GRAD_CACHE_CAP + c + 3u] = sg_sum.w;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_group == 0u) {
    for (uint c = simd_lane * 4u; c < fdim; c += STAR_SIMD_WIDTH * 4u) {
      float4 total = float4(0.0f);
      for (uint sg = 0u; sg < STAR_SIMDGROUPS; ++sg) {
        uint base = sg * STAR_FEATURE_GRAD_CACHE_CAP + c;
        total.x += partial_features[base + 0u];
        if (c + 1u < fdim) total.y += partial_features[base + 1u];
        if (c + 2u < fdim) total.z += partial_features[base + 2u];
        if (c + 3u < fdim) total.w += partial_features[base + 3u];
      }
      atomic_fetch_add_explicit(&grad_feature[tube_feature_base + c + 0u], total.x, memory_order_relaxed);
      if (c + 1u < fdim) {
        atomic_fetch_add_explicit(&grad_feature[tube_feature_base + c + 1u], total.y, memory_order_relaxed);
      }
      if (c + 2u < fdim) {
        atomic_fetch_add_explicit(&grad_feature[tube_feature_base + c + 2u], total.z, memory_order_relaxed);
      }
      if (c + 3u < fdim) {
        atomic_fetch_add_explicit(&grad_feature[tube_feature_base + c + 3u], total.w, memory_order_relaxed);
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
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
  float density = exp(-0.5f * qv);
  float alpha = min(mf.max_alpha, primitive_alpha_raw(opacity[tube_id], density, mf));
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

inline uint select_sample_order_id_gated(
    threadgroup uint* ids,
    uint count,
    const device float* ma,
    const device float* depth0,
    const device float* depth_beta,
    const device int* active_start,
    const device int* active_stop,
    uint frame,
    float3 sample_a,
    float last_depth,
    uint last_id,
    thread float& out_depth) {
  uint best_id = 0xFFFFFFFFu;
  float best_depth = INFINITY;
  for (uint i = 0u; i < count; ++i) {
    uint tube_id = ids[i];
    if (!tube_active_for_frame(active_start, active_stop, tube_id, frame)) continue;
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

kernel void clear_feature_direct_gradients(
    device float* grad_ma [[buffer(0)]],
    device float* grad_q [[buffer(1)]],
    device float* grad_opacity [[buffer(2)]],
    device float* grad_feature [[buffer(3)]],
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
  uint fdim = feature_dim(mi);
  uint feature_base = tube_id * fdim;
  for (uint c = 0u; c < fdim; ++c) {
    grad_feature[feature_base + c] = 0.0f;
  }
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

kernel void bin_screen_tubes_to_uvt_tiles_gated(
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
    const device int* active_start [[buffer(11)]],
    const device int* active_stop [[buffer(12)]],
    uint tube_id [[thread_position_in_grid]]) {
  if (tube_id >= uint(mi.tube_count)) return;
  int start = clamp(active_start[tube_id], 0, mi.frames);
  int stop = clamp(active_stop[tube_id], 0, mi.frames);
  if (start >= stop) return;

  Bounds3i b = tube_bounds(ma, q_uvt, opacity, tube_id, mi, mf);
  b.f0 = max(b.f0, start);
  b.f1 = min(b.f1, stop - 1);
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

kernel void render_uvt_tiles_gated(
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
    const device int* active_start [[buffer(13)]],
    const device int* active_stop [[buffer(14)]],
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
      uint tube_id = local_ids[i];
      if (!tube_active_for_frame(active_start, active_stop, tube_id, f)) continue;
      composite_tube(tube_id, sample_a, ma, q_uvt, opacity, color, mf, accum, T);
      if (T <= mf.transmittance_threshold) break;
    }
  } else {
    float last_depth = -INFINITY;
    uint last_id = 0u;
    for (uint rank = 0u; rank < count; ++rank) {
      float selected_depth;
      uint tube_id = select_sample_order_id_gated(
          local_ids, count, ma, depth0, depth_beta, active_start, active_stop, f, sample_a, last_depth, last_id, selected_depth);
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

kernel void render_projective_trace_tiles(
    const device float* coeffs [[buffer(0)]],
    const device float* times [[buffer(1)]],
    const device float* opacity [[buffer(2)]],
    const device float* color [[buffer(3)]],
    const device int* tile_counts [[buffer(4)]],
    const device int* tile_primitive_ids [[buffer(5)]],
    const device int* tile_active_start [[buffer(6)]],
    const device int* tile_active_stop [[buffer(7)]],
    constant MetaI32& mi [[buffer(8)]],
    constant MetaF32& mf [[buffer(9)]],
    device float* out_rgb [[buffer(10)]],
    constant float* projective_f32 [[buffer(11)]],
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]]) {
  uint tile_id = gid / STAR_THREADS;
  if (tile_id >= uint(mi.tile_count)) return;
  uint local_tid = tid;

  uint tx, ty, tz;
  decode_tile(tile_id, mi, tx, ty, tz);

  threadgroup uint local_ids[STAR_TILE_CAPACITY];
  threadgroup int local_start[STAR_TILE_CAPACITY];
  threadgroup int local_stop[STAR_TILE_CAPACITY];
  uint raw_count = uint(max(tile_counts[tile_id], 0));
  uint count = min(raw_count, STAR_TILE_CAPACITY);
  for (uint i = local_tid; i < count; i += STAR_THREADS) {
    uint idx = tile_id * STAR_TILE_CAPACITY + i;
    int primitive_id = tile_primitive_ids[idx];
    local_ids[i] = (primitive_id >= 0 && primitive_id < mi.tube_count) ? uint(primitive_id) : 0xFFFFFFFFu;
    local_start[i] = tile_active_start[idx];
    local_stop[i] = tile_active_stop[idx];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  uint samples_per_frame = STAR_TILE_X * STAR_TILE_Y;
  uint lt = local_tid / samples_per_frame;
  uint rem = local_tid - lt * samples_per_frame;
  uint ly = rem / STAR_TILE_X;
  uint lx = rem - ly * STAR_TILE_X;
  uint x = tx * STAR_TILE_X + lx;
  uint y = ty * STAR_TILE_Y + ly;
  uint f = tz * STAR_TILE_T + lt;
  if (x >= uint(mi.width) || y >= uint(mi.height) || f >= uint(mi.frames)) return;

  float t = times[f];
  float2 pixel_center = float2(float(x) + 0.5f, float(y) + 0.5f);
  float sigma_px = max(projective_f32[0], mf.eps);
  float3 accum = float3(0.0f);
  float T = 1.0f;

  float last_depth = -INFINITY;
  uint last_id = 0u;
  for (uint rank = 0u; rank < count; ++rank) {
    float selected_depth;
    uint tube_id = select_projective_order_id(
        local_ids, local_start, local_stop, count, coeffs, f, t, mf.eps, last_depth, last_id, selected_depth);
    if (tube_id == 0xFFFFFFFFu || tube_id >= uint(mi.tube_count)) break;
    composite_projective_trace(tube_id, pixel_center, t, sigma_px, coeffs, opacity, color, mf, accum, T);
    last_depth = selected_depth;
    last_id = tube_id;
    if (T <= mf.transmittance_threshold) break;
  }

  uint pix = (f * uint(mi.height) * uint(mi.width) + y * uint(mi.width) + x) * 3u;
  out_rgb[pix + 0u] = accum.x + T * mf.bg_r;
  out_rgb[pix + 1u] = accum.y + T * mf.bg_g;
  out_rgb[pix + 2u] = accum.z + T * mf.bg_b;
}

kernel void render_projective_trace_cell_tiles(
    const device float* coeffs [[buffer(0)]],
    const device float* times [[buffer(1)]],
    const device float* opacity [[buffer(2)]],
    const device float* color [[buffer(3)]],
    const device int* tile_counts [[buffer(4)]],
    const device int* tile_trace_ids [[buffer(5)]],
    const device int* tile_active_start [[buffer(6)]],
    const device int* tile_active_stop [[buffer(7)]],
    constant MetaI32& mi [[buffer(8)]],
    constant MetaF32& mf [[buffer(9)]],
    device float* out_rgb [[buffer(10)]],
    constant float* projective_f32 [[buffer(11)]],
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]]) {
  uint tile_id = gid / STAR_THREADS;
  if (tile_id >= uint(mi.tile_count)) return;
  uint local_tid = tid;

  uint tx, ty, tz;
  decode_tile(tile_id, mi, tx, ty, tz);

  threadgroup uint local_ids[STAR_TILE_CAPACITY];
  threadgroup int local_start[STAR_TILE_CAPACITY];
  threadgroup int local_stop[STAR_TILE_CAPACITY];
  uint raw_count = uint(max(tile_counts[tile_id], 0));
  uint count = min(raw_count, STAR_TILE_CAPACITY);
  for (uint i = local_tid; i < count; i += STAR_THREADS) {
    uint idx = tile_id * STAR_TILE_CAPACITY + i;
    int trace_id = tile_trace_ids[idx];
    local_ids[i] = (trace_id >= 0 && trace_id < mi.tube_count) ? uint(trace_id) : 0xFFFFFFFFu;
    local_start[i] = tile_active_start[idx];
    local_stop[i] = tile_active_stop[idx];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  uint samples_per_frame = STAR_TILE_X * STAR_TILE_Y;
  uint lt = local_tid / samples_per_frame;
  uint rem = local_tid - lt * samples_per_frame;
  uint ly = rem / STAR_TILE_X;
  uint lx = rem - ly * STAR_TILE_X;
  uint x = tx * STAR_TILE_X + lx;
  uint y = ty * STAR_TILE_Y + ly;
  uint f = tz * STAR_TILE_T + lt;
  if (x >= uint(mi.width) || y >= uint(mi.height) || f >= uint(mi.frames)) return;

  float t = times[f];
  float2 pixel_center = float2(float(x) + 0.5f, float(y) + 0.5f);
  float sigma_px = max(projective_f32[0], mf.eps);
  float3 accum = float3(0.0f);
  float T = 1.0f;

  float last_depth = -INFINITY;
  uint last_id = 0u;
  for (uint rank = 0u; rank < count; ++rank) {
    float selected_depth;
    uint trace_id = select_projective_cell_order_id(
        local_ids, local_start, local_stop, count, coeffs, f, t, last_depth, last_id, selected_depth);
    if (trace_id == 0xFFFFFFFFu || trace_id >= uint(mi.tube_count)) break;
    composite_projective_cell_trace(trace_id, pixel_center, t, sigma_px, coeffs, opacity, color, mf, accum, T);
    last_depth = selected_depth;
    last_id = trace_id;
    if (T <= mf.transmittance_threshold) break;
  }

  uint pix = (f * uint(mi.height) * uint(mi.width) + y * uint(mi.width) + x) * 3u;
  out_rgb[pix + 0u] = accum.x + T * mf.bg_r;
  out_rgb[pix + 1u] = accum.y + T * mf.bg_g;
  out_rgb[pix + 2u] = accum.z + T * mf.bg_b;
}

kernel void render_projective_trace_cell_interval_tiles(
    const device float* coeffs [[buffer(0)]],
    const device float* times [[buffer(1)]],
    const device float* opacity [[buffer(2)]],
    const device float* opacity_time_coeffs [[buffer(3)]],
    const device float* color [[buffer(4)]],
    const device int* tile_counts [[buffer(5)]],
    const device int* tile_trace_ids [[buffer(6)]],
    const device int* tile_active_start [[buffer(7)]],
    const device int* tile_active_stop [[buffer(8)]],
    constant MetaI32& mi [[buffer(9)]],
    constant MetaF32& mf [[buffer(10)]],
    device float* out_rgb [[buffer(11)]],
    constant float* projective_f32 [[buffer(12)]],
    const device float* spatial_precision_uv [[buffer(13)]],
    const device float* depth_affine_uv [[buffer(14)]],
    uint gid [[thread_position_in_grid]]) {
  uint total_pixels = uint(mi.frames) * uint(mi.height) * uint(mi.width);
  if (gid >= total_pixels) return;

  uint frame_area = uint(mi.height) * uint(mi.width);
  uint f = gid / frame_area;
  uint rem = gid - f * frame_area;
  uint y = rem / uint(mi.width);
  uint x = rem - y * uint(mi.width);
  uint tx = x / STAR_TILE_X;
  uint ty = y / STAR_TILE_Y;
  uint tile_id = ty * uint(mi.tiles_x) + tx;
  if (tile_id >= uint(mi.tile_count)) return;

  uint raw_count = uint(max(tile_counts[tile_id], 0));
  uint count = min(raw_count, STAR_TILE_CAPACITY);
  float t = times[f];
  float2 pixel_center = float2(float(x) + 0.5f, float(y) + 0.5f);
  float3 accum = float3(0.0f);
  float T = 1.0f;

  float last_depth = -INFINITY;
  uint last_id = 0u;
  for (uint rank = 0u; rank < count; ++rank) {
    float selected_depth;
    uint trace_id = select_projective_cell_order_id_interval(
        tile_trace_ids, tile_active_start, tile_active_stop, tile_id, count, coeffs, depth_affine_uv, pixel_center, f, t, last_depth, last_id, selected_depth);
    if (trace_id == 0xFFFFFFFFu || trace_id >= uint(mi.tube_count)) break;
    composite_projective_cell_trace_with_time_opacity(trace_id, pixel_center, t, coeffs, opacity, opacity_time_coeffs, spatial_precision_uv, color, mf, accum, T);
    last_depth = selected_depth;
    last_id = trace_id;
    if (T <= mf.transmittance_threshold) break;
  }

  uint pix = gid * 3u;
  out_rgb[pix + 0u] = accum.x + T * mf.bg_r;
  out_rgb[pix + 1u] = accum.y + T * mf.bg_g;
  out_rgb[pix + 2u] = accum.z + T * mf.bg_b;
}

kernel void render_projective_trace_family_interval_tiles(
    const device float* family_coeffs [[buffer(0)]],
    const device float* q_basis [[buffer(1)]],
    const device float* times [[buffer(2)]],
    const device float* opacity [[buffer(3)]],
    const device float* opacity_time_coeffs [[buffer(4)]],
    const device float* color [[buffer(5)]],
    const device int* tile_counts [[buffer(6)]],
    const device int* tile_trace_ids [[buffer(7)]],
    const device int* tile_active_start [[buffer(8)]],
    const device int* tile_active_stop [[buffer(9)]],
    constant MetaI32& mi [[buffer(10)]],
    constant MetaF32& mf [[buffer(11)]],
    device float* out_rgb [[buffer(12)]],
    constant float* projective_f32 [[buffer(13)]],
    const device float* spatial_precision_uv [[buffer(14)]],
    const device float* depth_affine_uv [[buffer(15)]],
    uint gid [[thread_position_in_grid]]) {
  uint total_pixels = uint(mi.frames) * uint(mi.height) * uint(mi.width);
  if (gid >= total_pixels) return;

  uint base_trace_count = uint(max(mi.reserved0, 0));
  uint basis_count = uint(max(mi.reserved1, 0));
  if (base_trace_count == 0u || basis_count == 0u) return;

  uint frame_area = uint(mi.height) * uint(mi.width);
  uint f = gid / frame_area;
  uint rem = gid - f * frame_area;
  uint y = rem / uint(mi.width);
  uint x = rem - y * uint(mi.width);
  uint tx = x / STAR_TILE_X;
  uint ty = y / STAR_TILE_Y;
  uint tile_id = ty * uint(mi.tiles_x) + tx;
  if (tile_id >= uint(mi.tile_count)) return;

  uint raw_count = uint(max(tile_counts[tile_id], 0));
  uint count = min(raw_count, STAR_TILE_CAPACITY);
  float t = times[f];
  float2 pixel_center = float2(float(x) + 0.5f, float(y) + 0.5f);
  float3 accum = float3(0.0f);
  float T = 1.0f;

  float last_depth = -INFINITY;
  uint last_id = 0u;
  for (uint rank = 0u; rank < count; ++rank) {
    float selected_depth;
    uint trace_id = select_projective_family_cell_order_id_interval(
        tile_trace_ids,
        tile_active_start,
        tile_active_stop,
        tile_id,
        count,
        family_coeffs,
        q_basis,
        depth_affine_uv,
        base_trace_count,
        basis_count,
        pixel_center,
        f,
        t,
        last_depth,
        last_id,
        selected_depth);
    if (trace_id == 0xFFFFFFFFu || trace_id >= uint(mi.tube_count)) break;
    composite_projective_family_cell_trace_with_time_opacity(
        trace_id,
        base_trace_count,
        basis_count,
        pixel_center,
        t,
        family_coeffs,
        q_basis,
        opacity,
        opacity_time_coeffs,
        spatial_precision_uv,
        color,
        mf,
        accum,
        T);
    last_depth = selected_depth;
    last_id = trace_id;
    if (T <= mf.transmittance_threshold) break;
  }

  uint pix = gid * 3u;
  out_rgb[pix + 0u] = accum.x + T * mf.bg_r;
  out_rgb[pix + 1u] = accum.y + T * mf.bg_g;
  out_rgb[pix + 2u] = accum.z + T * mf.bg_b;
}

kernel void render_projective_trace_cell_interval_rows(
    const device float* coeffs [[buffer(0)]],
    const device float* times [[buffer(1)]],
    const device float* opacity [[buffer(2)]],
    const device float* opacity_time_coeffs [[buffer(3)]],
    const device float* color [[buffer(4)]],
    const device int* tile_counts [[buffer(5)]],
    const device int* tile_trace_ids [[buffer(6)]],
    const device int* tile_active_start [[buffer(7)]],
    const device int* tile_active_stop [[buffer(8)]],
    const device float* row_weights [[buffer(9)]],
    constant MetaI32& mi [[buffer(10)]],
    constant MetaF32& mf [[buffer(11)]],
    device float* out_rgb [[buffer(12)]],
    constant float* projective_f32 [[buffer(13)]],
    const device float* spatial_precision_uv [[buffer(14)]],
    const device float* depth_affine_uv [[buffer(15)]],
    uint gid [[thread_position_in_grid]]) {
  uint total_pixels = uint(mi.height) * uint(mi.width);
  if (gid >= total_pixels) return;

  uint y = gid / uint(mi.width);
  uint x = gid - y * uint(mi.width);
  uint tx = x / STAR_TILE_X;
  uint ty = y / STAR_TILE_Y;
  uint tile_id = ty * uint(mi.tiles_x) + tx;
  if (tile_id >= uint(mi.tile_count)) return;

  uint raw_count = uint(max(tile_counts[tile_id], 0));
  uint count = min(raw_count, STAR_TILE_CAPACITY);
  float2 pixel_center = float2(float(x) + 0.5f, float(y) + 0.5f);
  float3 final_rgb = float3(0.0f);

  for (uint f = 0u; f < uint(mi.frames); ++f) {
    float w = row_weights[f * uint(mi.height) + y];
    if (w == 0.0f) continue;

    float t = times[f];
    float3 accum = float3(0.0f);
    float T = 1.0f;
    float last_depth = -INFINITY;
    uint last_id = 0u;
    for (uint rank = 0u; rank < count; ++rank) {
      float selected_depth;
      uint trace_id = select_projective_cell_order_id_interval(
          tile_trace_ids, tile_active_start, tile_active_stop, tile_id, count, coeffs, depth_affine_uv, pixel_center, f, t, last_depth, last_id, selected_depth);
      if (trace_id == 0xFFFFFFFFu || trace_id >= uint(mi.tube_count)) break;
      composite_projective_cell_trace_with_time_opacity(trace_id, pixel_center, t, coeffs, opacity, opacity_time_coeffs, spatial_precision_uv, color, mf, accum, T);
      last_depth = selected_depth;
      last_id = trace_id;
      if (T <= mf.transmittance_threshold) break;
    }

    final_rgb += w * (accum + T * float3(mf.bg_r, mf.bg_g, mf.bg_b));
  }

  uint pix = gid * 3u;
  out_rgb[pix + 0u] = final_rgb.x;
  out_rgb[pix + 1u] = final_rgb.y;
  out_rgb[pix + 2u] = final_rgb.z;
}

kernel void direct_atomic_projective_cell_interval_backward(
    const device float* coeffs [[buffer(0)]],
    const device float* times [[buffer(1)]],
    const device float* opacity [[buffer(2)]],
    const device float* opacity_time_coeffs [[buffer(3)]],
    const device float* color [[buffer(4)]],
    const device float* grad_image [[buffer(5)]],
    const device int* tile_counts [[buffer(6)]],
    const device int* tile_trace_ids [[buffer(7)]],
    const device int* tile_active_start [[buffer(8)]],
    const device int* tile_active_stop [[buffer(9)]],
    constant MetaI32& mi [[buffer(10)]],
    constant MetaF32& mf [[buffer(11)]],
    device atomic_float* grad_coeffs [[buffer(12)]],
    device atomic_float* grad_opacity [[buffer(13)]],
    device atomic_float* grad_opacity_time_coeffs [[buffer(14)]],
    device atomic_float* grad_color [[buffer(15)]],
    device atomic_float* grad_spatial_precision_uv [[buffer(16)]],
    constant float* projective_f32 [[buffer(17)]],
    const device float* spatial_precision_uv [[buffer(18)]],
    const device float* depth_affine_uv [[buffer(19)]],
    uint gid [[thread_position_in_grid]]) {
  uint total_pixels = uint(mi.frames) * uint(mi.height) * uint(mi.width);
  if (gid >= total_pixels) return;

  uint frame_area = uint(mi.height) * uint(mi.width);
  uint f = gid / frame_area;
  uint rem = gid - f * frame_area;
  uint y = rem / uint(mi.width);
  uint x = rem - y * uint(mi.width);
  uint tx = x / STAR_TILE_X;
  uint ty = y / STAR_TILE_Y;
  uint tile_id = ty * uint(mi.tiles_x) + tx;
  if (tile_id >= uint(mi.tile_count)) return;

  uint raw_count = uint(max(tile_counts[tile_id], 0));
  uint count = min(raw_count, STAR_TILE_CAPACITY);
  float t = times[f];
  float t2 = t * t;
  float basis0 = 1.0f;
  float basis1 = t;
  float basis2 = t2;
  float2 pixel_center = float2(float(x) + 0.5f, float(y) + 0.5f);
  uint ordered_ids[STAR_TILE_CAPACITY];
  uint ordered_count = 0u;
  float last_depth = -INFINITY;
  uint last_id = 0u;
  for (uint rank = 0u; rank < count; ++rank) {
    float selected_depth;
    uint trace_id = select_projective_cell_order_id_interval(
        tile_trace_ids, tile_active_start, tile_active_stop, tile_id, count, coeffs, depth_affine_uv, pixel_center, f, t, last_depth, last_id, selected_depth);
    if (trace_id == 0xFFFFFFFFu || trace_id >= uint(mi.tube_count)) break;
    ordered_ids[ordered_count] = trace_id;
    ordered_count += 1u;
    last_depth = selected_depth;
    last_id = trace_id;
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
    uint trace_id = ordered_ids[i];
    float u;
    float v;
    float depth;
    if (!eval_projective_cell_trace_point(coeffs, trace_id, t, u, v, depth)) continue;
    float2 d = pixel_center - float2(u, v);
    float radius2 = projective_cell_precision_radius2(spatial_precision_uv, trace_id, d);
    float exp_term = exp(-0.5f * radius2);
    float time_scale = projective_cell_opacity_time_scale(opacity_time_coeffs, trace_id, t);
    float alpha_raw = primitive_alpha_raw(opacity[trace_id], time_scale * exp_term, mf);
    float alpha = min(mf.max_alpha, alpha_raw);
    if (!(alpha >= mf.alpha_threshold)) continue;
    t_before[i] = T;
    alpha_values[i] = alpha;
    processed[i] = true;
    differentiable_alpha[i] = alpha_raw < mf.max_alpha;
    T *= (1.0f - alpha);
    if (T <= mf.transmittance_threshold) break;
  }

  uint image_base = gid * 3u;
  float3 grad_rgb = float3(grad_image[image_base + 0u], grad_image[image_base + 1u], grad_image[image_base + 2u]);
  float dT_next = dot(grad_rgb, float3(mf.bg_r, mf.bg_g, mf.bg_b));

  for (int si = int(ordered_count) - 1; si >= 0; --si) {
    uint i = uint(si);
    if (!processed[i]) continue;
    uint trace_id = ordered_ids[i];
    float alpha = alpha_values[i];
    float t_i = t_before[i];
    float3 c = load3(color, trace_id);
    float d_alpha = dot(grad_rgb, t_i * c) - dT_next * t_i;
    float3 d_color = grad_rgb * (t_i * alpha);
    float dT_i = dot(grad_rgb, alpha * c) + dT_next * (1.0f - alpha);
    dT_next = dT_i;

    uint color_base = trace_id * 3u;
    atomic_add3(grad_color, color_base, d_color);
    if (!differentiable_alpha[i]) continue;

    uint coeff_base = trace_id * 9u;
    float u = coeffs[coeff_base + 0u] + coeffs[coeff_base + 1u] * t + coeffs[coeff_base + 2u] * t2;
    float v = coeffs[coeff_base + 3u] + coeffs[coeff_base + 4u] * t + coeffs[coeff_base + 5u] * t2;
    if (!(isfinite(u) && isfinite(v))) continue;
    float2 d = pixel_center - float2(u, v);
    float radius2 = projective_cell_precision_radius2(spatial_precision_uv, trace_id, d);
    float exp_term = exp(-0.5f * radius2);
    float time_scale = projective_cell_opacity_time_scale(opacity_time_coeffs, trace_id, t);
    float density = time_scale * exp_term;
    float alpha_shape_scale = -2.0f * d_alpha * primitive_alpha_d_qv(opacity[trace_id], density, mf);
    float2 center_grad = projective_cell_precision_center_grad(spatial_precision_uv, trace_id, d);
    float grad_u = alpha_shape_scale * center_grad.x;
    float grad_v = alpha_shape_scale * center_grad.y;
    float grad_time = -0.5f * alpha_shape_scale;
    float grad_precision_scale = alpha_shape_scale;

    atomic_fetch_add_explicit(&grad_coeffs[coeff_base + 0u], grad_u * basis0, memory_order_relaxed);
    atomic_fetch_add_explicit(&grad_coeffs[coeff_base + 1u], grad_u * basis1, memory_order_relaxed);
    atomic_fetch_add_explicit(&grad_coeffs[coeff_base + 2u], grad_u * basis2, memory_order_relaxed);
    atomic_fetch_add_explicit(&grad_coeffs[coeff_base + 3u], grad_v * basis0, memory_order_relaxed);
    atomic_fetch_add_explicit(&grad_coeffs[coeff_base + 4u], grad_v * basis1, memory_order_relaxed);
    atomic_fetch_add_explicit(&grad_coeffs[coeff_base + 5u], grad_v * basis2, memory_order_relaxed);
    atomic_fetch_add_explicit(
        &grad_opacity[trace_id],
        d_alpha * primitive_alpha_d_opacity(opacity[trace_id], density, mf),
        memory_order_relaxed);
    uint time_coeff_base = trace_id * 3u;
    atomic_fetch_add_explicit(&grad_opacity_time_coeffs[time_coeff_base + 0u], grad_time * basis0, memory_order_relaxed);
    atomic_fetch_add_explicit(&grad_opacity_time_coeffs[time_coeff_base + 1u], grad_time * basis1, memory_order_relaxed);
    atomic_fetch_add_explicit(&grad_opacity_time_coeffs[time_coeff_base + 2u], grad_time * basis2, memory_order_relaxed);
    uint precision_base = trace_id * 3u;
    atomic_fetch_add_explicit(&grad_spatial_precision_uv[precision_base + 0u], -0.5f * grad_precision_scale * d.x * d.x, memory_order_relaxed);
    atomic_fetch_add_explicit(&grad_spatial_precision_uv[precision_base + 1u], -grad_precision_scale * d.x * d.y, memory_order_relaxed);
    atomic_fetch_add_explicit(&grad_spatial_precision_uv[precision_base + 2u], -0.5f * grad_precision_scale * d.y * d.y, memory_order_relaxed);
  }
}

kernel void direct_atomic_projective_family_cell_interval_backward(
    const device float* family_coeffs [[buffer(0)]],
    const device float* q_basis [[buffer(1)]],
    const device float* times [[buffer(2)]],
    const device float* opacity [[buffer(3)]],
    const device float* opacity_time_coeffs [[buffer(4)]],
    const device float* color [[buffer(5)]],
    const device float* grad_image [[buffer(6)]],
    const device int* tile_counts [[buffer(7)]],
    const device int* tile_trace_ids [[buffer(8)]],
    const device int* tile_active_start [[buffer(9)]],
    const device int* tile_active_stop [[buffer(10)]],
    constant MetaI32& mi [[buffer(11)]],
    constant MetaF32& mf [[buffer(12)]],
    device atomic_float* grad_family_coeffs [[buffer(13)]],
    device atomic_float* grad_q_basis [[buffer(14)]],
    device atomic_float* grad_opacity [[buffer(15)]],
    device atomic_float* grad_opacity_time_coeffs [[buffer(16)]],
    device atomic_float* grad_color [[buffer(17)]],
    device atomic_float* grad_spatial_precision_uv [[buffer(18)]],
    constant float* projective_f32 [[buffer(19)]],
    const device float* spatial_precision_uv [[buffer(20)]],
    const device float* depth_affine_uv [[buffer(21)]],
    uint gid [[thread_position_in_grid]]) {
  uint total_pixels = uint(mi.frames) * uint(mi.height) * uint(mi.width);
  if (gid >= total_pixels) return;

  uint base_trace_count = uint(max(mi.reserved0, 0));
  uint basis_count = uint(max(mi.reserved1, 0));
  if (base_trace_count == 0u || basis_count == 0u) return;

  uint frame_area = uint(mi.height) * uint(mi.width);
  uint f = gid / frame_area;
  uint rem = gid - f * frame_area;
  uint y = rem / uint(mi.width);
  uint x = rem - y * uint(mi.width);
  uint tx = x / STAR_TILE_X;
  uint ty = y / STAR_TILE_Y;
  uint tile_id = ty * uint(mi.tiles_x) + tx;
  if (tile_id >= uint(mi.tile_count)) return;

  uint raw_count = uint(max(tile_counts[tile_id], 0));
  uint count = min(raw_count, STAR_TILE_CAPACITY);
  float t = times[f];
  float t2 = t * t;
  float basis0 = 1.0f;
  float basis1 = t;
  float basis2 = t2;
  float2 pixel_center = float2(float(x) + 0.5f, float(y) + 0.5f);
  uint ordered_ids[STAR_TILE_CAPACITY];
  uint ordered_count = 0u;
  float last_depth = -INFINITY;
  uint last_id = 0u;
  for (uint rank = 0u; rank < count; ++rank) {
    float selected_depth;
    uint trace_id = select_projective_family_cell_order_id_interval(
        tile_trace_ids,
        tile_active_start,
        tile_active_stop,
        tile_id,
        count,
        family_coeffs,
        q_basis,
        depth_affine_uv,
        base_trace_count,
        basis_count,
        pixel_center,
        f,
        t,
        last_depth,
        last_id,
        selected_depth);
    if (trace_id == 0xFFFFFFFFu || trace_id >= uint(mi.tube_count)) break;
    ordered_ids[ordered_count] = trace_id;
    ordered_count += 1u;
    last_depth = selected_depth;
    last_id = trace_id;
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
    uint global_trace_id = ordered_ids[i];
    float u;
    float v;
    float depth;
    if (!eval_projective_family_cell_trace_point(
            family_coeffs, q_basis, global_trace_id, base_trace_count, basis_count, t, u, v, depth)) continue;
    uint base_trace_id = global_trace_id % base_trace_count;
    float2 d = pixel_center - float2(u, v);
    float radius2 = projective_cell_precision_radius2(spatial_precision_uv, base_trace_id, d);
    float exp_term = exp(-0.5f * radius2);
    float time_scale = projective_cell_opacity_time_scale(opacity_time_coeffs, base_trace_id, t);
    float alpha_raw = primitive_alpha_raw(opacity[base_trace_id], time_scale * exp_term, mf);
    float alpha = min(mf.max_alpha, alpha_raw);
    if (!(alpha >= mf.alpha_threshold)) continue;
    t_before[i] = T;
    alpha_values[i] = alpha;
    processed[i] = true;
    differentiable_alpha[i] = alpha_raw < mf.max_alpha;
    T *= (1.0f - alpha);
    if (T <= mf.transmittance_threshold) break;
  }

  uint image_base = gid * 3u;
  float3 grad_rgb = float3(grad_image[image_base + 0u], grad_image[image_base + 1u], grad_image[image_base + 2u]);
  float dT_next = dot(grad_rgb, float3(mf.bg_r, mf.bg_g, mf.bg_b));

  for (int si = int(ordered_count) - 1; si >= 0; --si) {
    uint i = uint(si);
    if (!processed[i]) continue;
    uint global_trace_id = ordered_ids[i];
    uint base_trace_id = global_trace_id % base_trace_count;
    uint q_id = global_trace_id / base_trace_count;
    float alpha = alpha_values[i];
    float t_i = t_before[i];
    float3 c = load3(color, base_trace_id);
    float d_alpha = dot(grad_rgb, t_i * c) - dT_next * t_i;
    float3 d_color = grad_rgb * (t_i * alpha);
    float dT_i = dot(grad_rgb, alpha * c) + dT_next * (1.0f - alpha);
    dT_next = dT_i;

    uint color_base = base_trace_id * 3u;
    atomic_add3(grad_color, color_base, d_color);
    if (!differentiable_alpha[i]) continue;

    float c0 = family_coeff_at(family_coeffs, q_basis, base_trace_id, 0u, q_id, basis_count);
    float c1 = family_coeff_at(family_coeffs, q_basis, base_trace_id, 1u, q_id, basis_count);
    float c2 = family_coeff_at(family_coeffs, q_basis, base_trace_id, 2u, q_id, basis_count);
    float c3 = family_coeff_at(family_coeffs, q_basis, base_trace_id, 3u, q_id, basis_count);
    float c4 = family_coeff_at(family_coeffs, q_basis, base_trace_id, 4u, q_id, basis_count);
    float c5 = family_coeff_at(family_coeffs, q_basis, base_trace_id, 5u, q_id, basis_count);
    float u = c0 + c1 * t + c2 * t2;
    float v = c3 + c4 * t + c5 * t2;
    if (!(isfinite(u) && isfinite(v))) continue;
    float2 d = pixel_center - float2(u, v);
    float radius2 = projective_cell_precision_radius2(spatial_precision_uv, base_trace_id, d);
    float exp_term = exp(-0.5f * radius2);
    float time_scale = projective_cell_opacity_time_scale(opacity_time_coeffs, base_trace_id, t);
    float density = time_scale * exp_term;
    float alpha_shape_scale = -2.0f * d_alpha * primitive_alpha_d_qv(opacity[base_trace_id], density, mf);
    float2 center_grad = projective_cell_precision_center_grad(spatial_precision_uv, base_trace_id, d);
    float grad_u = alpha_shape_scale * center_grad.x;
    float grad_v = alpha_shape_scale * center_grad.y;
    float grad_time = -0.5f * alpha_shape_scale;
    float grad_precision_scale = alpha_shape_scale;

    float grad_coeff[6];
    grad_coeff[0] = grad_u * basis0;
    grad_coeff[1] = grad_u * basis1;
    grad_coeff[2] = grad_u * basis2;
    grad_coeff[3] = grad_v * basis0;
    grad_coeff[4] = grad_v * basis1;
    grad_coeff[5] = grad_v * basis2;

    uint q_base = q_id * basis_count;
    for (uint b = 0u; b < basis_count; ++b) {
      float qb = q_basis[q_base + b];
      float grad_qb = 0.0f;
      for (uint k = 0u; k < 6u; ++k) {
        uint family_index = (base_trace_id * 9u + k) * basis_count + b;
        float g = grad_coeff[k];
        atomic_fetch_add_explicit(&grad_family_coeffs[family_index], g * qb, memory_order_relaxed);
        grad_qb += g * family_coeffs[family_index];
      }
      atomic_fetch_add_explicit(&grad_q_basis[q_base + b], grad_qb, memory_order_relaxed);
    }

    atomic_fetch_add_explicit(
        &grad_opacity[base_trace_id],
        d_alpha * primitive_alpha_d_opacity(opacity[base_trace_id], density, mf),
        memory_order_relaxed);
    uint time_coeff_base = base_trace_id * 3u;
    atomic_fetch_add_explicit(&grad_opacity_time_coeffs[time_coeff_base + 0u], grad_time * basis0, memory_order_relaxed);
    atomic_fetch_add_explicit(&grad_opacity_time_coeffs[time_coeff_base + 1u], grad_time * basis1, memory_order_relaxed);
    atomic_fetch_add_explicit(&grad_opacity_time_coeffs[time_coeff_base + 2u], grad_time * basis2, memory_order_relaxed);
    uint precision_base = base_trace_id * 3u;
    atomic_fetch_add_explicit(&grad_spatial_precision_uv[precision_base + 0u], -0.5f * grad_precision_scale * d.x * d.x, memory_order_relaxed);
    atomic_fetch_add_explicit(&grad_spatial_precision_uv[precision_base + 1u], -grad_precision_scale * d.x * d.y, memory_order_relaxed);
    atomic_fetch_add_explicit(&grad_spatial_precision_uv[precision_base + 2u], -0.5f * grad_precision_scale * d.y * d.y, memory_order_relaxed);
  }
}

kernel void direct_atomic_projective_trace_backward(
    const device float* coeffs [[buffer(0)]],
    const device float* times [[buffer(1)]],
    const device float* opacity [[buffer(2)]],
    const device float* color [[buffer(3)]],
    const device float* grad_image [[buffer(4)]],
    const device int* tile_counts [[buffer(5)]],
    const device int* tile_primitive_ids [[buffer(6)]],
    const device int* tile_active_start [[buffer(7)]],
    const device int* tile_active_stop [[buffer(8)]],
    constant MetaI32& mi [[buffer(9)]],
    constant MetaF32& mf [[buffer(10)]],
    device atomic_float* grad_coeffs [[buffer(11)]],
    device atomic_float* grad_opacity [[buffer(12)]],
    device atomic_float* grad_color [[buffer(13)]],
    constant float* projective_f32 [[buffer(14)]],
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]]) {
  uint tile_id = gid / STAR_THREADS;
  if (tile_id >= uint(mi.tile_count)) return;
  uint local_tid = tid;

  uint tx, ty, tz;
  decode_tile(tile_id, mi, tx, ty, tz);

  threadgroup uint local_ids[STAR_TILE_CAPACITY];
  threadgroup int local_start[STAR_TILE_CAPACITY];
  threadgroup int local_stop[STAR_TILE_CAPACITY];
  uint raw_count = uint(max(tile_counts[tile_id], 0));
  uint count = min(raw_count, STAR_TILE_CAPACITY);
  for (uint i = local_tid; i < count; i += STAR_THREADS) {
    uint idx = tile_id * STAR_TILE_CAPACITY + i;
    int primitive_id = tile_primitive_ids[idx];
    local_ids[i] = (primitive_id >= 0 && primitive_id < mi.tube_count) ? uint(primitive_id) : 0xFFFFFFFFu;
    local_start[i] = tile_active_start[idx];
    local_stop[i] = tile_active_stop[idx];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  uint samples_per_frame = STAR_TILE_X * STAR_TILE_Y;
  uint lt = local_tid / samples_per_frame;
  uint rem = local_tid - lt * samples_per_frame;
  uint ly = rem / STAR_TILE_X;
  uint lx = rem - ly * STAR_TILE_X;
  uint x = tx * STAR_TILE_X + lx;
  uint y = ty * STAR_TILE_Y + ly;
  uint f = tz * STAR_TILE_T + lt;
  if (x >= uint(mi.width) || y >= uint(mi.height) || f >= uint(mi.frames)) return;

  float t = times[f];
  float t2 = t * t;
  float basis0 = 1.0f;
  float basis1 = t;
  float basis2 = t2;
  float2 pixel_center = float2(float(x) + 0.5f, float(y) + 0.5f);
  float sigma_px = max(projective_f32[0], mf.eps);
  float inv_sigma2 = 1.0f / max(sigma_px * sigma_px, mf.eps);

  uint ordered_ids[STAR_TILE_CAPACITY];
  uint ordered_count = 0u;
  float last_depth = -INFINITY;
  uint last_id = 0u;
  for (uint rank = 0u; rank < count; ++rank) {
    float selected_depth;
    uint tube_id = select_projective_order_id(
        local_ids, local_start, local_stop, count, coeffs, f, t, mf.eps, last_depth, last_id, selected_depth);
    if (tube_id == 0xFFFFFFFFu || tube_id >= uint(mi.tube_count)) break;
    ordered_ids[ordered_count] = tube_id;
    ordered_count += 1u;
    last_depth = selected_depth;
    last_id = tube_id;
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
    float u;
    float v;
    float depth;
    if (!eval_projective_trace_point(coeffs, tube_id, t, mf.eps, u, v, depth)) continue;
    float2 d = pixel_center - float2(u, v);
    float exp_term = exp(-0.5f * dot(d, d) * inv_sigma2);
    float alpha_raw = primitive_alpha_raw(opacity[tube_id], exp_term, mf);
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

    uint coeff_base = tube_id * 9u;
    float hu = coeffs[coeff_base + 0u] + coeffs[coeff_base + 1u] * t + coeffs[coeff_base + 2u] * t2;
    float hv = coeffs[coeff_base + 3u] + coeffs[coeff_base + 4u] * t + coeffs[coeff_base + 5u] * t2;
    float hz = coeffs[coeff_base + 6u] + coeffs[coeff_base + 7u] * t + coeffs[coeff_base + 8u] * t2;
    if (!(isfinite(hu) && isfinite(hv) && isfinite(hz) && fabs(hz) > mf.eps)) continue;
    float inv_hz = 1.0f / hz;
    float u = hu * inv_hz;
    float v = hv * inv_hz;
    float2 d = pixel_center - float2(u, v);
    float exp_term = exp(-0.5f * dot(d, d) * inv_sigma2);
    float alpha_shape_scale = -2.0f * d_alpha * primitive_alpha_d_qv(opacity[tube_id], exp_term, mf);
    float grad_u = alpha_shape_scale * d.x * inv_sigma2;
    float grad_v = alpha_shape_scale * d.y * inv_sigma2;
    float grad_hu = grad_u * inv_hz;
    float grad_hv = grad_v * inv_hz;
    float grad_hz = -(grad_u * hu + grad_v * hv) * inv_hz * inv_hz;

    atomic_fetch_add_explicit(&grad_coeffs[coeff_base + 0u], grad_hu * basis0, memory_order_relaxed);
    atomic_fetch_add_explicit(&grad_coeffs[coeff_base + 1u], grad_hu * basis1, memory_order_relaxed);
    atomic_fetch_add_explicit(&grad_coeffs[coeff_base + 2u], grad_hu * basis2, memory_order_relaxed);
    atomic_fetch_add_explicit(&grad_coeffs[coeff_base + 3u], grad_hv * basis0, memory_order_relaxed);
    atomic_fetch_add_explicit(&grad_coeffs[coeff_base + 4u], grad_hv * basis1, memory_order_relaxed);
    atomic_fetch_add_explicit(&grad_coeffs[coeff_base + 5u], grad_hv * basis2, memory_order_relaxed);
    atomic_fetch_add_explicit(&grad_coeffs[coeff_base + 6u], grad_hz * basis0, memory_order_relaxed);
    atomic_fetch_add_explicit(&grad_coeffs[coeff_base + 7u], grad_hz * basis1, memory_order_relaxed);
    atomic_fetch_add_explicit(&grad_coeffs[coeff_base + 8u], grad_hz * basis2, memory_order_relaxed);
    atomic_fetch_add_explicit(
        &grad_opacity[tube_id],
        d_alpha * primitive_alpha_d_opacity(opacity[tube_id], exp_term, mf),
        memory_order_relaxed);
  }
}

kernel void render_uvt_feature_tiles(
    const device float* ma [[buffer(0)]],
    const device float* q_uvt [[buffer(1)]],
    const device float* depth0 [[buffer(2)]],
    const device float* depth_beta [[buffer(3)]],
    const device float* opacity [[buffer(4)]],
    const device float* feature [[buffer(5)]],
    constant MetaI32& mi [[buffer(6)]],
    constant MetaF32& mf [[buffer(7)]],
    const device atomic_uint* tile_counts [[buffer(8)]],
    const device uint* tile_tube_ids [[buffer(9)]],
    const device float* tile_depths [[buffer(10)]],
    device atomic_uint* tile_unstable [[buffer(11)]],
    device float* out_feature [[buffer(12)]],
    device float* out_alpha [[buffer(13)]],
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

  uint fdim = feature_dim(mi);
  uint pixel_id = (f * uint(mi.height) + y) * uint(mi.width) + x;
  uint feature_base = pixel_id * fdim;
  for (uint c = 0u; c < fdim; ++c) {
    out_feature[feature_base + c] = 0.0f;
  }

  float3 sample_a = float3(float(x) + 0.5f, float(y) + 0.5f, frame_time(f, mi));
  float T = 1.0f;

  if (!unstable) {
    for (uint i = 0u; i < count; ++i) {
      uint tube_id = local_ids[i];
      float3 d = sample_a - load3(ma, tube_id);
      float qv = quadratic_q(q_uvt, tube_id, d);
      if (!isfinite(qv)) continue;
      float alpha_raw = primitive_alpha_raw(opacity[tube_id], exp(-0.5f * qv), mf);
      float alpha = min(mf.max_alpha, alpha_raw);
      if (!(alpha >= mf.alpha_threshold)) continue;
      float w = T * alpha;
      uint tube_feature_base = tube_id * fdim;
      for (uint c = 0u; c < fdim; ++c) {
        out_feature[feature_base + c] += w * feature[tube_feature_base + c];
      }
      T *= (1.0f - alpha);
      if (T <= mf.transmittance_threshold) break;
    }
  } else {
    float last_depth = -INFINITY;
    uint last_id = 0u;
    for (uint rank = 0u; rank < count; ++rank) {
      float selected_depth;
      uint tube_id = select_sample_order_id(local_ids, count, ma, depth0, depth_beta, sample_a, last_depth, last_id, selected_depth);
      if (tube_id == 0xFFFFFFFFu) break;
      float3 d = sample_a - load3(ma, tube_id);
      float qv = quadratic_q(q_uvt, tube_id, d);
      if (isfinite(qv)) {
        float alpha_raw = primitive_alpha_raw(opacity[tube_id], exp(-0.5f * qv), mf);
        float alpha = min(mf.max_alpha, alpha_raw);
        if (alpha >= mf.alpha_threshold) {
          float w = T * alpha;
          uint tube_feature_base = tube_id * fdim;
          for (uint c = 0u; c < fdim; ++c) {
            out_feature[feature_base + c] += w * feature[tube_feature_base + c];
          }
          T *= (1.0f - alpha);
          if (T <= mf.transmittance_threshold) break;
        }
      }
      last_depth = selected_depth;
      last_id = tube_id;
    }
  }

  out_alpha[pixel_id] = 1.0f - T;
}

kernel void render_uvt_feature_sparse_pixels(
    const device float* ma [[buffer(0)]],
    const device float* q_uvt [[buffer(1)]],
    const device float* depth0 [[buffer(2)]],
    const device float* depth_beta [[buffer(3)]],
    const device float* opacity [[buffer(4)]],
    const device float* feature [[buffer(5)]],
    const device uint* pixel_ids [[buffer(6)]],
    constant MetaI32& mi [[buffer(7)]],
    constant MetaF32& mf [[buffer(8)]],
    const device atomic_uint* tile_counts [[buffer(9)]],
    const device uint* tile_tube_ids [[buffer(10)]],
    const device float* tile_depths [[buffer(11)]],
    device atomic_uint* tile_unstable [[buffer(12)]],
    device float* out_feature_values [[buffer(13)]],
    device float* out_alpha_values [[buffer(14)]],
    uint gid [[thread_position_in_grid]]) {
  uint fdim = feature_dim(mi);
  uint feature_base = gid * fdim;
  for (uint c = 0u; c < fdim; ++c) {
    out_feature_values[feature_base + c] = 0.0f;
  }
  out_alpha_values[gid] = 0.0f;

  uint total_pixels = uint(mi.frames) * uint(mi.height) * uint(mi.width);
  uint pixel_id = pixel_ids[gid];
  if (pixel_id >= total_pixels) return;

  uint hw = uint(mi.height) * uint(mi.width);
  uint f = pixel_id / hw;
  uint rem = pixel_id - f * hw;
  uint y = rem / uint(mi.width);
  uint x = rem - y * uint(mi.width);
  uint tx = x / STAR_TILE_X;
  uint ty = y / STAR_TILE_Y;
  uint tz = f / STAR_TILE_T;
  uint tile_id = encode_tile(tx, ty, tz, mi);
  if (tile_id >= uint(mi.tile_count)) return;

  thread uint local_ids[STAR_TILE_CAPACITY];
  thread float local_depths[STAR_TILE_CAPACITY];
  uint raw_count = atomic_load_explicit(tile_counts + tile_id, memory_order_relaxed);
  uint count = min(raw_count, STAR_TILE_CAPACITY);
  for (uint i = 0u; i < count; ++i) {
    uint idx = tile_id * STAR_TILE_CAPACITY + i;
    local_ids[i] = tile_tube_ids[idx];
    local_depths[i] = tile_depths[idx];
  }
  sort_by_depth_thread(local_ids, local_depths, count);

  bool unstable = tile_order_unstable_thread(local_ids, count, ma, depth0, depth_beta, tx, ty, tz, mi);
  if (unstable) {
    atomic_store_explicit(tile_unstable + tile_id, 1u, memory_order_relaxed);
  }

  float3 sample_a = float3(float(x) + 0.5f, float(y) + 0.5f, frame_time(f, mi));
  float T = 1.0f;

  if (!unstable) {
    for (uint i = 0u; i < count; ++i) {
      uint tube_id = local_ids[i];
      float3 d = sample_a - load3(ma, tube_id);
      float qv = quadratic_q(q_uvt, tube_id, d);
      if (!isfinite(qv)) continue;
      float alpha_raw = primitive_alpha_raw(opacity[tube_id], exp(-0.5f * qv), mf);
      float alpha = min(mf.max_alpha, alpha_raw);
      if (!(alpha >= mf.alpha_threshold)) continue;
      float w = T * alpha;
      uint tube_feature_base = tube_id * fdim;
      for (uint c = 0u; c < fdim; ++c) {
        out_feature_values[feature_base + c] += w * feature[tube_feature_base + c];
      }
      T *= (1.0f - alpha);
      if (T <= mf.transmittance_threshold) break;
    }
  } else {
    float last_depth = -INFINITY;
    uint last_id = 0u;
    for (uint rank = 0u; rank < count; ++rank) {
      float selected_depth;
      uint tube_id = select_sample_order_id_thread(
          local_ids, count, ma, depth0, depth_beta, sample_a, last_depth, last_id, selected_depth);
      if (tube_id == 0xFFFFFFFFu) break;
      float3 d = sample_a - load3(ma, tube_id);
      float qv = quadratic_q(q_uvt, tube_id, d);
      if (isfinite(qv)) {
        float alpha_raw = primitive_alpha_raw(opacity[tube_id], exp(-0.5f * qv), mf);
        float alpha = min(mf.max_alpha, alpha_raw);
        if (alpha >= mf.alpha_threshold) {
          float w = T * alpha;
          uint tube_feature_base = tube_id * fdim;
          for (uint c = 0u; c < fdim; ++c) {
            out_feature_values[feature_base + c] += w * feature[tube_feature_base + c];
          }
          T *= (1.0f - alpha);
          if (T <= mf.transmittance_threshold) break;
        }
      }
      last_depth = selected_depth;
      last_id = tube_id;
    }
  }

  out_alpha_values[gid] = 1.0f - T;
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
  float alpha_unclamped = primitive_alpha_raw(opacity[tube_id], exp_term, mf);
  bool active = isfinite(qv) && alpha_unclamped < mf.max_alpha;
  float alpha = active ? alpha_unclamped : mf.max_alpha;

  uint image_base = ((frame * uint(mi.height) + y) * uint(mi.width) + x) * 3u;
  float3 grad_rgb = float3(grad_image[image_base + 0u], grad_image[image_base + 1u], grad_image[image_base + 2u]);
  float3 c = load3(color, tube_id);
  float grad_alpha = active ? dot(grad_rgb, c) : 0.0f;
  float grad_qv = active
      ? grad_alpha * primitive_alpha_d_qv(opacity[tube_id], exp_term, mf)
      : 0.0f;
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
  grad_opacity_samples[idx] = active
      ? grad_alpha * primitive_alpha_d_opacity(opacity[tube_id], exp_term, mf)
      : 0.0f;

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
    float alpha_raw = primitive_alpha_raw(opacity[tube_id], exp(-0.5f * qv), mf);
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
    float grad_qv = d_alpha * primitive_alpha_d_qv(opacity[tube_id], exp_term, mf);
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
    grad_opacity_samples[entry] =
        d_alpha * primitive_alpha_d_opacity(opacity[tube_id], exp_term, mf);
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
    float alpha_raw = primitive_alpha_raw(opacity[tube_id], exp(-0.5f * qv), mf);
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
    float grad_qv = d_alpha * primitive_alpha_d_qv(opacity[tube_id], exp_term, mf);
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
    atomic_fetch_add_explicit(
        &grad_opacity[tube_id],
        d_alpha * primitive_alpha_d_opacity(opacity[tube_id], exp_term, mf),
        memory_order_relaxed);
  }
}

kernel void direct_atomic_backward_gated(
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
    const device int* active_start [[buffer(17)]],
    const device int* active_stop [[buffer(18)]],
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
      uint tube_id = select_sample_order_id_gated(
          local_ids, count, ma, depth0, depth_beta, active_start, active_stop, f, sample_a, last_depth, last_id, selected_depth);
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
    if (!tube_active_for_frame(active_start, active_stop, tube_id, f)) continue;
    float3 d = sample_a - load3(ma, tube_id);
    float qv = quadratic_q(q_uvt, tube_id, d);
    if (!isfinite(qv)) continue;
    float alpha_raw = primitive_alpha_raw(opacity[tube_id], exp(-0.5f * qv), mf);
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
    float grad_qv = d_alpha * primitive_alpha_d_qv(opacity[tube_id], exp_term, mf);
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
    atomic_fetch_add_explicit(
        &grad_opacity[tube_id],
        d_alpha * primitive_alpha_d_opacity(opacity[tube_id], exp_term, mf),
        memory_order_relaxed);
  }
}

kernel void direct_atomic_feature_backward(
    const device float* ma [[buffer(0)]],
    const device float* q_uvt [[buffer(1)]],
    const device float* depth0 [[buffer(2)]],
    const device float* depth_beta [[buffer(3)]],
    const device float* opacity [[buffer(4)]],
    const device float* feature [[buffer(5)]],
    const device float* grad_feature_image [[buffer(6)]],
    const device float* grad_alpha_image [[buffer(7)]],
    constant MetaI32& mi [[buffer(8)]],
    constant MetaF32& mf [[buffer(9)]],
    const device atomic_uint* tile_counts [[buffer(10)]],
    const device uint* tile_tube_ids [[buffer(11)]],
    const device float* tile_depths [[buffer(12)]],
    device atomic_uint* tile_unstable [[buffer(13)]],
    device atomic_float* grad_ma [[buffer(14)]],
    device atomic_float* grad_q [[buffer(15)]],
    device atomic_float* grad_opacity [[buffer(16)]],
    device atomic_float* grad_feature [[buffer(17)]],
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  uint tile_id = gid / STAR_THREADS;
  if (tile_id >= uint(mi.tile_count)) return;
  uint local_tid = tid;

  uint tx, ty, tz;
  decode_tile(tile_id, mi, tx, ty, tz);

  threadgroup uint local_ids[STAR_TILE_CAPACITY];
  threadgroup float local_depths[STAR_TILE_CAPACITY];
  threadgroup float partial_features[STAR_SIMDGROUPS * STAR_FEATURE_GRAD_CACHE_CAP];
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
  uint fdim = feature_dim(mi);
  bool use_grad_feature_cache = ((mi.reserved1 & 1) != 0) && fdim <= STAR_FEATURE_GRAD_CACHE_CAP;
  bool skip_feature_grad_atomic = ((mi.reserved1 & 2) != 0);
  bool reduce_feature_grad_atomic = ((mi.reserved1 & 4) != 0) && use_grad_feature_cache && !unstable;
  bool reduce_feature_grad_atomic_vec4 = ((mi.reserved1 & 16) != 0) && use_grad_feature_cache && !unstable;
  bool fused_first3_sigmoid_mse = ((mi.reserved1 & 8) != 0) && fdim >= 3u && fdim <= STAR_FEATURE_GRAD_CACHE_CAP;
  bool feature_grad_only = ((mi.reserved1 & 32) != 0);
  bool pixel_valid = x < uint(mi.width) && y < uint(mi.height) && f < uint(mi.frames);
  if (!pixel_valid && !reduce_feature_grad_atomic && !reduce_feature_grad_atomic_vec4) return;

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
  float3 pixel_feature3 = float3(0.0f);
  for (uint i = 0u; i < STAR_TILE_CAPACITY; ++i) {
    t_before[i] = 0.0f;
    alpha_values[i] = 0.0f;
    processed[i] = false;
    differentiable_alpha[i] = false;
  }
  if (pixel_valid) {
    for (uint i = 0u; i < ordered_count; ++i) {
      uint tube_id = ordered_ids[i];
      float3 d = sample_a - load3(ma, tube_id);
      float qv = quadratic_q(q_uvt, tube_id, d);
      if (!isfinite(qv)) continue;
      float alpha_raw = primitive_alpha_raw(opacity[tube_id], exp(-0.5f * qv), mf);
      float alpha = min(mf.max_alpha, alpha_raw);
      if (!(alpha >= mf.alpha_threshold)) continue;
      t_before[i] = T;
      alpha_values[i] = alpha;
      processed[i] = true;
      differentiable_alpha[i] = alpha_raw < mf.max_alpha;
      if (fused_first3_sigmoid_mse) {
        uint tube_feature_base = tube_id * fdim;
        pixel_feature3 += T * alpha * float3(
            feature[tube_feature_base + 0u],
            feature[tube_feature_base + 1u],
            feature[tube_feature_base + 2u]);
      }
      T *= (1.0f - alpha);
      if (T <= mf.transmittance_threshold) break;
    }
  }

  uint pixel_id = (f * uint(mi.height) + y) * uint(mi.width) + x;
  uint grad_feature_base = pixel_id * fdim;
  thread float grad_feature_cache[STAR_FEATURE_GRAD_CACHE_CAP];
  float dT_next = pixel_valid ? -grad_alpha_image[pixel_id] : 0.0f;
  uint active_feature_grad_dim = fdim;
  if (fused_first3_sigmoid_mse) {
    float alpha_out = 1.0f - T;
    float3 target_rgb = pixel_valid ? float3(
        grad_feature_image[grad_feature_base + 0u],
        grad_feature_image[grad_feature_base + 1u],
        grad_feature_image[grad_feature_base + 2u]) : float3(0.0f);
    float3 splat_rgb = 1.0f / (1.0f + exp(-pixel_feature3));
    float3 rgb = alpha_out * splat_rgb;
    float inv_n = 1.0f / max(1.0f, float(mi.frames * mi.height * mi.width * 3));
    float3 grad_rgb = pixel_valid ? (2.0f * inv_n) * (rgb - target_rgb) : float3(0.0f);
    float3 grad_feature3 = grad_rgb * alpha_out * splat_rgb * (1.0f - splat_rgb);
    grad_feature_cache[0u] = grad_feature3.x;
    grad_feature_cache[1u] = grad_feature3.y;
    grad_feature_cache[2u] = grad_feature3.z;
    for (uint c = 3u; c < fdim; ++c) {
      grad_feature_cache[c] = 0.0f;
    }
    dT_next = pixel_valid ? -dot(grad_rgb, splat_rgb) : 0.0f;
    active_feature_grad_dim = 3u;
    use_grad_feature_cache = true;
  } else if (use_grad_feature_cache) {
    for (uint c = 0u; c < fdim; ++c) {
      grad_feature_cache[c] = pixel_valid ? grad_feature_image[grad_feature_base + c] : 0.0f;
    }
  }
  for (int si = int(ordered_count) - 1; si >= 0; --si) {
    uint i = uint(si);
    uint tube_id = ordered_ids[i];
    bool active = pixel_valid && processed[i];
    float alpha = active ? alpha_values[i] : 0.0f;
    float t_i = active ? t_before[i] : 0.0f;
    uint tube_feature_base = tube_id * fdim;

    float grad_dot_feature = 0.0f;
    if (active && !feature_grad_only) {
      for (uint c = 0u; c < fdim; ++c) {
        float grad_feature_c = use_grad_feature_cache ? grad_feature_cache[c] : grad_feature_image[grad_feature_base + c];
        grad_dot_feature += grad_feature_c * feature[tube_feature_base + c];
      }
    }
    float d_alpha = 0.0f;
    if (!feature_grad_only) {
      d_alpha = t_i * grad_dot_feature - dT_next * t_i;
      float dT_i = alpha * grad_dot_feature + dT_next * (1.0f - alpha);
      dT_next = dT_i;
    }

    float feature_scale = t_i * alpha;
    if (!skip_feature_grad_atomic) {
      if (reduce_feature_grad_atomic_vec4) {
        reduce_atomic_add_feature_grads_cached_vec4(
            grad_feature,
            grad_feature_cache,
            tube_id,
            feature_scale,
            active,
            mi,
            simd_lane,
            simd_group,
            partial_features);
      } else if (reduce_feature_grad_atomic) {
        reduce_atomic_add_feature_grads_cached(
            grad_feature,
            grad_feature_cache,
            tube_id,
            feature_scale,
            active,
            mi,
            simd_lane,
            simd_group,
            partial_features);
      } else if (active) {
        for (uint c = 0u; c < active_feature_grad_dim; ++c) {
          float grad_feature_c = use_grad_feature_cache ? grad_feature_cache[c] : grad_feature_image[grad_feature_base + c];
          atomic_fetch_add_explicit(
              &grad_feature[tube_feature_base + c],
              grad_feature_c * feature_scale,
              memory_order_relaxed);
        }
      }
    }
    if (feature_grad_only) continue;
    if (!active || !differentiable_alpha[i]) continue;

    float3 d = sample_a - load3(ma, tube_id);
    float qv = quadratic_q(q_uvt, tube_id, d);
    float exp_term = exp(-0.5f * qv);
    float grad_qv = d_alpha * primitive_alpha_d_qv(opacity[tube_id], exp_term, mf);
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
    atomic_fetch_add_explicit(
        &grad_opacity[tube_id],
        d_alpha * primitive_alpha_d_opacity(opacity[tube_id], exp_term, mf),
        memory_order_relaxed);
  }
}

kernel void direct_atomic_feature_sparse_pixels_backward(
    const device float* ma [[buffer(0)]],
    const device float* q_uvt [[buffer(1)]],
    const device float* depth0 [[buffer(2)]],
    const device float* depth_beta [[buffer(3)]],
    const device float* opacity [[buffer(4)]],
    const device float* feature [[buffer(5)]],
    const device uint* pixel_ids [[buffer(6)]],
    const device float* grad_feature_values [[buffer(7)]],
    const device float* grad_alpha_values [[buffer(8)]],
    constant MetaI32& mi [[buffer(9)]],
    constant MetaF32& mf [[buffer(10)]],
    const device atomic_uint* tile_counts [[buffer(11)]],
    const device uint* tile_tube_ids [[buffer(12)]],
    const device float* tile_depths [[buffer(13)]],
    device atomic_uint* tile_unstable [[buffer(14)]],
    device atomic_float* grad_ma [[buffer(15)]],
    device atomic_float* grad_q [[buffer(16)]],
    device atomic_float* grad_opacity [[buffer(17)]],
    device atomic_float* grad_feature [[buffer(18)]],
    uint gid [[thread_position_in_grid]]) {
  uint fdim = feature_dim(mi);
  uint total_pixels = uint(mi.frames) * uint(mi.height) * uint(mi.width);
  uint pixel_id = pixel_ids[gid];
  if (pixel_id >= total_pixels) return;

  uint hw = uint(mi.height) * uint(mi.width);
  uint f = pixel_id / hw;
  uint rem = pixel_id - f * hw;
  uint y = rem / uint(mi.width);
  uint x = rem - y * uint(mi.width);
  uint tx = x / STAR_TILE_X;
  uint ty = y / STAR_TILE_Y;
  uint tz = f / STAR_TILE_T;
  uint tile_id = encode_tile(tx, ty, tz, mi);
  if (tile_id >= uint(mi.tile_count)) return;

  thread uint local_ids[STAR_TILE_CAPACITY];
  thread float local_depths[STAR_TILE_CAPACITY];
  uint raw_count = atomic_load_explicit(tile_counts + tile_id, memory_order_relaxed);
  uint count = min(raw_count, STAR_TILE_CAPACITY);
  for (uint i = 0u; i < count; ++i) {
    uint idx = tile_id * STAR_TILE_CAPACITY + i;
    local_ids[i] = tile_tube_ids[idx];
    local_depths[i] = tile_depths[idx];
  }
  sort_by_depth_thread(local_ids, local_depths, count);

  bool unstable = tile_order_unstable_thread(local_ids, count, ma, depth0, depth_beta, tx, ty, tz, mi);
  if (unstable) {
    atomic_store_explicit(tile_unstable + tile_id, 1u, memory_order_relaxed);
  }

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
      uint tube_id = select_sample_order_id_thread(
          local_ids, count, ma, depth0, depth_beta, sample_a, last_depth, last_id, selected_depth);
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
  for (uint i = 0u; i < STAR_TILE_CAPACITY; ++i) {
    t_before[i] = 0.0f;
    alpha_values[i] = 0.0f;
    processed[i] = false;
    differentiable_alpha[i] = false;
  }

  float T = 1.0f;
  for (uint i = 0u; i < ordered_count; ++i) {
    uint tube_id = ordered_ids[i];
    float3 d = sample_a - load3(ma, tube_id);
    float qv = quadratic_q(q_uvt, tube_id, d);
    if (!isfinite(qv)) continue;
    float alpha_raw = primitive_alpha_raw(opacity[tube_id], exp(-0.5f * qv), mf);
    float alpha = min(mf.max_alpha, alpha_raw);
    if (!(alpha >= mf.alpha_threshold)) continue;
    t_before[i] = T;
    alpha_values[i] = alpha;
    processed[i] = true;
    differentiable_alpha[i] = alpha_raw < mf.max_alpha;
    T *= (1.0f - alpha);
    if (T <= mf.transmittance_threshold) break;
  }

  uint grad_feature_base = gid * fdim;
  float dT_next = -grad_alpha_values[gid];
  for (int si = int(ordered_count) - 1; si >= 0; --si) {
    uint i = uint(si);
    if (!processed[i]) continue;
    uint tube_id = ordered_ids[i];
    float alpha = alpha_values[i];
    float t_i = t_before[i];
    uint tube_feature_base = tube_id * fdim;

    float grad_dot_feature = 0.0f;
    for (uint c = 0u; c < fdim; ++c) {
      grad_dot_feature += grad_feature_values[grad_feature_base + c] * feature[tube_feature_base + c];
    }
    float d_alpha = t_i * grad_dot_feature - dT_next * t_i;
    float dT_i = alpha * grad_dot_feature + dT_next * (1.0f - alpha);
    dT_next = dT_i;

    float feature_scale = t_i * alpha;
    for (uint c = 0u; c < fdim; ++c) {
      atomic_fetch_add_explicit(
          &grad_feature[tube_feature_base + c],
          grad_feature_values[grad_feature_base + c] * feature_scale,
          memory_order_relaxed);
    }
    if (!differentiable_alpha[i]) continue;

    float3 d = sample_a - load3(ma, tube_id);
    float qv = quadratic_q(q_uvt, tube_id, d);
    float exp_term = exp(-0.5f * qv);
    float grad_qv = d_alpha * primitive_alpha_d_qv(opacity[tube_id], exp_term, mf);
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
    atomic_fetch_add_explicit(
        &grad_opacity[tube_id],
        d_alpha * primitive_alpha_d_opacity(opacity[tube_id], exp_term, mf),
        memory_order_relaxed);
  }
}

inline float erf_approx(float x) {
  // Abramowitz-Stegun 7.1.26; Metal on this target does not expose erf().
  float sign = x < 0.0f ? -1.0f : 1.0f;
  float ax = abs(x);
  float t = 1.0f / (1.0f + 0.3275911f * ax);
  float y = 1.0f -
      (((((1.061405429f * t - 1.453152027f) * t + 1.421413741f) * t - 0.284496736f) * t +
        0.254829592f) * t) *
          exp(-ax * ax);
  return sign * y;
}

inline float gelu_exact_forward(float x) {
  return 0.5f * x * (1.0f + erf_approx(x * 0.7071067811865475f));
}

inline float gelu_exact_grad(float x) {
  float cdf = 0.5f * (1.0f + erf_approx(x * 0.7071067811865475f));
  float pdf = exp(-0.5f * x * x) * 0.3989422804014327f;
  return cdf + x * pdf;
}

kernel void direct_atomic_feature_sparse_hidden_sigmoid_mse_backward(
    const device float* ma [[buffer(0)]],
    const device float* q_uvt [[buffer(1)]],
    const device float* depth0 [[buffer(2)]],
    const device float* depth_beta [[buffer(3)]],
    const device float* opacity [[buffer(4)]],
    const device float* feature [[buffer(5)]],
    const device uint* pixel_ids [[buffer(6)]],
    const device float* target_rgb_values [[buffer(7)]],
    const device float* hidden_weight [[buffer(8)]],
    const device float* hidden_bias [[buffer(9)]],
    const device float* output_weight [[buffer(10)]],
    const device float* output_bias [[buffer(11)]],
    constant int* hidden_meta [[buffer(12)]],
    constant MetaI32& mi [[buffer(13)]],
    constant MetaF32& mf [[buffer(14)]],
    const device atomic_uint* tile_counts [[buffer(15)]],
    const device uint* tile_tube_ids [[buffer(16)]],
    const device float* tile_depths [[buffer(17)]],
    device atomic_uint* tile_unstable [[buffer(18)]],
    device atomic_float* grad_ma [[buffer(19)]],
    device atomic_float* grad_q [[buffer(20)]],
    device atomic_float* grad_opacity [[buffer(21)]],
    device atomic_float* grad_feature [[buffer(22)]],
    device atomic_float* loss_sum [[buffer(23)]],
    uint gid [[thread_position_in_grid]]) {
  uint fdim = feature_dim(mi);
  uint hdim = uint(hidden_meta[0]);
  uint loss_norm_elems = uint(max(hidden_meta[2], 1));
  uint total_pixels = uint(mi.frames) * uint(mi.height) * uint(mi.width);
  uint pixel_id = pixel_ids[gid];
  if (pixel_id >= total_pixels) return;

  uint hw = uint(mi.height) * uint(mi.width);
  uint f = pixel_id / hw;
  uint rem = pixel_id - f * hw;
  uint y = rem / uint(mi.width);
  uint x = rem - y * uint(mi.width);
  uint tx = x / STAR_TILE_X;
  uint ty = y / STAR_TILE_Y;
  uint tz = f / STAR_TILE_T;
  uint tile_id = encode_tile(tx, ty, tz, mi);
  if (tile_id >= uint(mi.tile_count)) return;

  thread uint local_ids[STAR_TILE_CAPACITY];
  thread float local_depths[STAR_TILE_CAPACITY];
  uint raw_count = atomic_load_explicit(tile_counts + tile_id, memory_order_relaxed);
  uint count = min(raw_count, STAR_TILE_CAPACITY);
  for (uint i = 0u; i < count; ++i) {
    uint idx = tile_id * STAR_TILE_CAPACITY + i;
    local_ids[i] = tile_tube_ids[idx];
    local_depths[i] = tile_depths[idx];
  }
  sort_by_depth_thread(local_ids, local_depths, count);

  bool unstable = tile_order_unstable_thread(local_ids, count, ma, depth0, depth_beta, tx, ty, tz, mi);
  if (unstable) {
    atomic_store_explicit(tile_unstable + tile_id, 1u, memory_order_relaxed);
  }

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
      uint tube_id = select_sample_order_id_thread(
          local_ids, count, ma, depth0, depth_beta, sample_a, last_depth, last_id, selected_depth);
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
  thread float pixel_feature_cache[STAR_FEATURE_GRAD_CACHE_CAP];
  thread float grad_feature_cache[STAR_FEATURE_GRAD_CACHE_CAP];
  thread float hidden_pre_cache[STAR_FEATURE_GRAD_CACHE_CAP];
  thread float hidden_cache[STAR_FEATURE_GRAD_CACHE_CAP];
  thread float grad_hidden_pre_cache[STAR_FEATURE_GRAD_CACHE_CAP];
  for (uint c = 0u; c < fdim; ++c) {
    pixel_feature_cache[c] = 0.0f;
    grad_feature_cache[c] = 0.0f;
  }
  for (uint i = 0u; i < STAR_TILE_CAPACITY; ++i) {
    t_before[i] = 0.0f;
    alpha_values[i] = 0.0f;
    processed[i] = false;
    differentiable_alpha[i] = false;
  }

  float T = 1.0f;
  for (uint i = 0u; i < ordered_count; ++i) {
    uint tube_id = ordered_ids[i];
    float3 d = sample_a - load3(ma, tube_id);
    float qv = quadratic_q(q_uvt, tube_id, d);
    if (!isfinite(qv)) continue;
    float alpha_raw = primitive_alpha_raw(opacity[tube_id], exp(-0.5f * qv), mf);
    float alpha = min(mf.max_alpha, alpha_raw);
    if (!(alpha >= mf.alpha_threshold)) continue;
    t_before[i] = T;
    alpha_values[i] = alpha;
    processed[i] = true;
    differentiable_alpha[i] = alpha_raw < mf.max_alpha;
    uint tube_feature_base = tube_id * fdim;
    float feature_scale = T * alpha;
    for (uint c = 0u; c < fdim; ++c) {
      pixel_feature_cache[c] += feature_scale * feature[tube_feature_base + c];
    }
    T *= (1.0f - alpha);
    if (T <= mf.transmittance_threshold) break;
  }

  float alpha_out = 1.0f - T;
  for (uint h = 0u; h < hdim; ++h) {
    float v = hidden_bias[h];
    uint hidden_base = h * fdim;
    for (uint c = 0u; c < fdim; ++c) {
      v += hidden_weight[hidden_base + c] * pixel_feature_cache[c];
    }
    hidden_pre_cache[h] = v;
    hidden_cache[h] = gelu_exact_forward(v);
  }
  float3 logits = float3(output_bias[0u], output_bias[1u], output_bias[2u]);
  for (uint h = 0u; h < hdim; ++h) {
    float v = hidden_cache[h];
    logits.x += output_weight[h] * v;
    logits.y += output_weight[hdim + h] * v;
    logits.z += output_weight[2u * hdim + h] * v;
  }
  float3 splat_rgb = 1.0f / (1.0f + exp(-logits));
  uint target_base = gid * 3u;
  float3 target = float3(
      target_rgb_values[target_base + 0u],
      target_rgb_values[target_base + 1u],
      target_rgb_values[target_base + 2u]);
  float3 rgb = alpha_out * splat_rgb;
  float inv_n = 1.0f / max(1.0f, float(loss_norm_elems));
  float3 diff = rgb - target;
  atomic_fetch_add_explicit(loss_sum, inv_n * dot(diff, diff), memory_order_relaxed);
  float3 grad_rgb = (2.0f * inv_n) * diff;
  float3 grad_logits = grad_rgb * alpha_out * splat_rgb * (1.0f - splat_rgb);
  for (uint h = 0u; h < hdim; ++h) {
    float grad_hidden =
        grad_logits.x * output_weight[h] +
        grad_logits.y * output_weight[hdim + h] +
        grad_logits.z * output_weight[2u * hdim + h];
    grad_hidden_pre_cache[h] = grad_hidden * gelu_exact_grad(hidden_pre_cache[h]);
  }
  for (uint c = 0u; c < fdim; ++c) {
    float v = 0.0f;
    for (uint h = 0u; h < hdim; ++h) {
      v += grad_hidden_pre_cache[h] * hidden_weight[h * fdim + c];
    }
    grad_feature_cache[c] = v;
  }

  float dT_next = -dot(grad_rgb, splat_rgb);
  for (int si = int(ordered_count) - 1; si >= 0; --si) {
    uint i = uint(si);
    if (!processed[i]) continue;
    uint tube_id = ordered_ids[i];
    float alpha = alpha_values[i];
    float t_i = t_before[i];
    uint tube_feature_base = tube_id * fdim;

    float grad_dot_feature = 0.0f;
    for (uint c = 0u; c < fdim; ++c) {
      grad_dot_feature += grad_feature_cache[c] * feature[tube_feature_base + c];
    }
    float d_alpha = t_i * grad_dot_feature - dT_next * t_i;
    float dT_i = alpha * grad_dot_feature + dT_next * (1.0f - alpha);
    dT_next = dT_i;

    float feature_scale = t_i * alpha;
    for (uint c = 0u; c < fdim; ++c) {
      atomic_fetch_add_explicit(
          &grad_feature[tube_feature_base + c],
          grad_feature_cache[c] * feature_scale,
          memory_order_relaxed);
    }
    if (!differentiable_alpha[i]) continue;

    float3 d = sample_a - load3(ma, tube_id);
    float qv = quadratic_q(q_uvt, tube_id, d);
    float exp_term = exp(-0.5f * qv);
    float grad_qv = d_alpha * primitive_alpha_d_qv(opacity[tube_id], exp_term, mf);
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
    atomic_fetch_add_explicit(
        &grad_opacity[tube_id],
        d_alpha * primitive_alpha_d_opacity(opacity[tube_id], exp_term, mf),
        memory_order_relaxed);
  }
}

kernel void sparse_hidden_sigmoid_target_area_forward_sums(
    const device float* ma [[buffer(0)]],
    const device float* q_uvt [[buffer(1)]],
    const device float* depth0 [[buffer(2)]],
    const device float* depth_beta [[buffer(3)]],
    const device float* opacity [[buffer(4)]],
    const device float* feature [[buffer(5)]],
    const device uint* pixel_ids [[buffer(6)]],
    const device uint* cell_ids [[buffer(7)]],
    const device float* hidden_weight [[buffer(8)]],
    const device float* hidden_bias [[buffer(9)]],
    const device float* output_weight [[buffer(10)]],
    const device float* output_bias [[buffer(11)]],
    constant int* hidden_meta [[buffer(12)]],
    constant MetaI32& mi [[buffer(13)]],
    constant MetaF32& mf [[buffer(14)]],
    const device atomic_uint* tile_counts [[buffer(15)]],
    const device uint* tile_tube_ids [[buffer(16)]],
    const device float* tile_depths [[buffer(17)]],
    device atomic_uint* tile_unstable [[buffer(18)]],
    device atomic_float* pred_sums [[buffer(19)]],
    uint gid [[thread_position_in_grid]]) {
  uint fdim = feature_dim(mi);
  uint hdim = uint(hidden_meta[0]);
  uint cell_count = uint(max(hidden_meta[1], 0));
  uint cell_id = cell_ids[gid];
  if (cell_id >= cell_count) return;
  uint total_pixels = uint(mi.frames) * uint(mi.height) * uint(mi.width);
  uint pixel_id = pixel_ids[gid];
  if (pixel_id >= total_pixels) return;

  uint hw = uint(mi.height) * uint(mi.width);
  uint f = pixel_id / hw;
  uint rem = pixel_id - f * hw;
  uint y = rem / uint(mi.width);
  uint x = rem - y * uint(mi.width);
  uint tx = x / STAR_TILE_X;
  uint ty = y / STAR_TILE_Y;
  uint tz = f / STAR_TILE_T;
  uint tile_id = encode_tile(tx, ty, tz, mi);
  if (tile_id >= uint(mi.tile_count)) return;

  thread uint local_ids[STAR_TILE_CAPACITY];
  thread float local_depths[STAR_TILE_CAPACITY];
  uint raw_count = atomic_load_explicit(tile_counts + tile_id, memory_order_relaxed);
  uint count = min(raw_count, STAR_TILE_CAPACITY);
  for (uint i = 0u; i < count; ++i) {
    uint idx = tile_id * STAR_TILE_CAPACITY + i;
    local_ids[i] = tile_tube_ids[idx];
    local_depths[i] = tile_depths[idx];
  }
  sort_by_depth_thread(local_ids, local_depths, count);

  bool unstable = tile_order_unstable_thread(local_ids, count, ma, depth0, depth_beta, tx, ty, tz, mi);
  if (unstable) {
    atomic_store_explicit(tile_unstable + tile_id, 1u, memory_order_relaxed);
  }

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
      uint tube_id = select_sample_order_id_thread(
          local_ids, count, ma, depth0, depth_beta, sample_a, last_depth, last_id, selected_depth);
      if (tube_id == 0xFFFFFFFFu) break;
      ordered_ids[ordered_count] = tube_id;
      ordered_count += 1u;
      last_depth = selected_depth;
      last_id = tube_id;
    }
  }

  thread float pixel_feature_cache[STAR_FEATURE_GRAD_CACHE_CAP];
  thread float hidden_cache[STAR_FEATURE_GRAD_CACHE_CAP];
  for (uint c = 0u; c < fdim; ++c) {
    pixel_feature_cache[c] = 0.0f;
  }

  float T = 1.0f;
  for (uint i = 0u; i < ordered_count; ++i) {
    uint tube_id = ordered_ids[i];
    float3 d = sample_a - load3(ma, tube_id);
    float qv = quadratic_q(q_uvt, tube_id, d);
    if (!isfinite(qv)) continue;
    float alpha_raw = primitive_alpha_raw(opacity[tube_id], exp(-0.5f * qv), mf);
    float alpha = min(mf.max_alpha, alpha_raw);
    if (!(alpha >= mf.alpha_threshold)) continue;
    uint tube_feature_base = tube_id * fdim;
    float feature_scale = T * alpha;
    for (uint c = 0u; c < fdim; ++c) {
      pixel_feature_cache[c] += feature_scale * feature[tube_feature_base + c];
    }
    T *= (1.0f - alpha);
    if (T <= mf.transmittance_threshold) break;
  }

  float alpha_out = 1.0f - T;
  for (uint h = 0u; h < hdim; ++h) {
    float v = hidden_bias[h];
    uint hidden_base = h * fdim;
    for (uint c = 0u; c < fdim; ++c) {
      v += hidden_weight[hidden_base + c] * pixel_feature_cache[c];
    }
    hidden_cache[h] = gelu_exact_forward(v);
  }
  float3 logits = float3(output_bias[0u], output_bias[1u], output_bias[2u]);
  for (uint h = 0u; h < hdim; ++h) {
    float v = hidden_cache[h];
    logits.x += output_weight[h] * v;
    logits.y += output_weight[hdim + h] * v;
    logits.z += output_weight[2u * hdim + h] * v;
  }
  float3 splat_rgb = 1.0f / (1.0f + exp(-logits));
  float3 rgb = alpha_out * splat_rgb;
  uint out_base = cell_id * 3u;
  atomic_fetch_add_explicit(&pred_sums[out_base + 0u], rgb.x, memory_order_relaxed);
  atomic_fetch_add_explicit(&pred_sums[out_base + 1u], rgb.y, memory_order_relaxed);
  atomic_fetch_add_explicit(&pred_sums[out_base + 2u], rgb.z, memory_order_relaxed);
}

kernel void direct_atomic_feature_sparse_hidden_target_area_backward(
    const device float* ma [[buffer(0)]],
    const device float* q_uvt [[buffer(1)]],
    const device float* depth0 [[buffer(2)]],
    const device float* depth_beta [[buffer(3)]],
    const device float* opacity [[buffer(4)]],
    const device float* feature [[buffer(5)]],
    const device uint* pixel_ids [[buffer(6)]],
    const device uint* cell_ids [[buffer(7)]],
    const device float* cell_grad_rgb [[buffer(8)]],
    const device float* hidden_weight [[buffer(9)]],
    const device float* hidden_bias [[buffer(10)]],
    const device float* output_weight [[buffer(11)]],
    const device float* output_bias [[buffer(12)]],
    constant int* hidden_meta [[buffer(13)]],
    constant MetaI32& mi [[buffer(14)]],
    constant MetaF32& mf [[buffer(15)]],
    const device atomic_uint* tile_counts [[buffer(16)]],
    const device uint* tile_tube_ids [[buffer(17)]],
    const device float* tile_depths [[buffer(18)]],
    device atomic_uint* tile_unstable [[buffer(19)]],
    device atomic_float* grad_ma [[buffer(20)]],
    device atomic_float* grad_q [[buffer(21)]],
    device atomic_float* grad_opacity [[buffer(22)]],
    device atomic_float* grad_feature [[buffer(23)]],
    device atomic_float* grad_hidden_weight [[buffer(24)]],
    device atomic_float* grad_hidden_bias [[buffer(25)]],
    device atomic_float* grad_output_weight [[buffer(26)]],
    device atomic_float* grad_output_bias [[buffer(27)]],
    uint gid [[thread_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]]) {
  uint fdim = feature_dim(mi);
  uint hdim = uint(hidden_meta[0]);
  uint cell_count = uint(max(hidden_meta[1], 0));
  uint mode_bits = uint(max(hidden_meta[2], 0));
  bool skip_feature_grad = (mode_bits & 1u) != 0u;
  bool skip_geometry_grad = (mode_bits & 2u) != 0u;
  bool skip_hidden_forward = (mode_bits & 4u) != 0u;
  bool skip_hidden_backward = skip_hidden_forward || ((mode_bits & 8u) != 0u);
  bool skip_hidden_feature_vjp = (mode_bits & 16u) != 0u;
  bool rowmajor_hidden_feature_vjp = (mode_bits & 32u) != 0u;
  bool vec4_hidden_feature_vjp = (mode_bits & 64u) != 0u;
  bool compute_colorizer_grad = (mode_bits & 128u) != 0u;
  bool reduce_colorizer_grad_simd = compute_colorizer_grad && ((mode_bits & 256u) != 0u);
  if (skip_hidden_backward || skip_hidden_feature_vjp) {
    skip_feature_grad = true;
    skip_geometry_grad = true;
  }
  uint cell_id = cell_ids[gid];
  if (cell_id >= cell_count) return;
  uint total_pixels = uint(mi.frames) * uint(mi.height) * uint(mi.width);
  uint pixel_id = pixel_ids[gid];
  if (pixel_id >= total_pixels) return;

  uint hw = uint(mi.height) * uint(mi.width);
  uint f = pixel_id / hw;
  uint rem = pixel_id - f * hw;
  uint y = rem / uint(mi.width);
  uint x = rem - y * uint(mi.width);
  uint tx = x / STAR_TILE_X;
  uint ty = y / STAR_TILE_Y;
  uint tz = f / STAR_TILE_T;
  uint tile_id = encode_tile(tx, ty, tz, mi);
  if (tile_id >= uint(mi.tile_count)) return;

  thread uint local_ids[STAR_TILE_CAPACITY];
  thread float local_depths[STAR_TILE_CAPACITY];
  uint raw_count = atomic_load_explicit(tile_counts + tile_id, memory_order_relaxed);
  uint count = min(raw_count, STAR_TILE_CAPACITY);
  for (uint i = 0u; i < count; ++i) {
    uint idx = tile_id * STAR_TILE_CAPACITY + i;
    local_ids[i] = tile_tube_ids[idx];
    local_depths[i] = tile_depths[idx];
  }
  sort_by_depth_thread(local_ids, local_depths, count);

  bool unstable = tile_order_unstable_thread(local_ids, count, ma, depth0, depth_beta, tx, ty, tz, mi);
  if (unstable) {
    atomic_store_explicit(tile_unstable + tile_id, 1u, memory_order_relaxed);
  }

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
      uint tube_id = select_sample_order_id_thread(
          local_ids, count, ma, depth0, depth_beta, sample_a, last_depth, last_id, selected_depth);
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
  thread float pixel_feature_cache[STAR_FEATURE_GRAD_CACHE_CAP];
  thread float grad_feature_cache[STAR_FEATURE_GRAD_CACHE_CAP];
  thread float hidden_pre_cache[STAR_FEATURE_GRAD_CACHE_CAP];
  thread float hidden_cache[STAR_FEATURE_GRAD_CACHE_CAP];
  thread float grad_hidden_pre_cache[STAR_FEATURE_GRAD_CACHE_CAP];
  for (uint c = 0u; c < fdim; ++c) {
    pixel_feature_cache[c] = 0.0f;
    grad_feature_cache[c] = 0.0f;
  }
  for (uint i = 0u; i < STAR_TILE_CAPACITY; ++i) {
    t_before[i] = 0.0f;
    alpha_values[i] = 0.0f;
    processed[i] = false;
    differentiable_alpha[i] = false;
  }

  float T = 1.0f;
  for (uint i = 0u; i < ordered_count; ++i) {
    uint tube_id = ordered_ids[i];
    float3 d = sample_a - load3(ma, tube_id);
    float qv = quadratic_q(q_uvt, tube_id, d);
    if (!isfinite(qv)) continue;
    float alpha_raw = primitive_alpha_raw(opacity[tube_id], exp(-0.5f * qv), mf);
    float alpha = min(mf.max_alpha, alpha_raw);
    if (!(alpha >= mf.alpha_threshold)) continue;
    t_before[i] = T;
    alpha_values[i] = alpha;
    processed[i] = true;
    differentiable_alpha[i] = alpha_raw < mf.max_alpha;
    uint tube_feature_base = tube_id * fdim;
    float feature_scale = T * alpha;
    for (uint c = 0u; c < fdim; ++c) {
      pixel_feature_cache[c] += feature_scale * feature[tube_feature_base + c];
    }
    T *= (1.0f - alpha);
    if (T <= mf.transmittance_threshold) break;
  }

  float3 splat_rgb = float3(0.0f);
  float3 grad_rgb = float3(0.0f);
  if (!skip_hidden_forward) {
    float alpha_out = 1.0f - T;
    for (uint h = 0u; h < hdim; ++h) {
      float v = hidden_bias[h];
      uint hidden_base = h * fdim;
      for (uint c = 0u; c < fdim; ++c) {
        v += hidden_weight[hidden_base + c] * pixel_feature_cache[c];
      }
      hidden_pre_cache[h] = v;
      hidden_cache[h] = gelu_exact_forward(v);
    }
    float3 logits = float3(output_bias[0u], output_bias[1u], output_bias[2u]);
    for (uint h = 0u; h < hdim; ++h) {
      float v = hidden_cache[h];
      logits.x += output_weight[h] * v;
      logits.y += output_weight[hdim + h] * v;
      logits.z += output_weight[2u * hdim + h] * v;
    }
    splat_rgb = 1.0f / (1.0f + exp(-logits));
    uint grad_base = cell_id * 3u;
    grad_rgb = float3(
        cell_grad_rgb[grad_base + 0u],
        cell_grad_rgb[grad_base + 1u],
        cell_grad_rgb[grad_base + 2u]);
    if (!skip_hidden_backward) {
      float3 grad_logits = grad_rgb * alpha_out * splat_rgb * (1.0f - splat_rgb);
      if (compute_colorizer_grad) {
        for (uint h = 0u; h < hdim; ++h) {
          float hidden_v = hidden_cache[h];
          if (reduce_colorizer_grad_simd) {
            simd_reduced_atomic_add(&grad_output_weight[h], grad_logits.x * hidden_v, simd_lane);
            simd_reduced_atomic_add(&grad_output_weight[hdim + h], grad_logits.y * hidden_v, simd_lane);
            simd_reduced_atomic_add(&grad_output_weight[2u * hdim + h], grad_logits.z * hidden_v, simd_lane);
          } else {
            atomic_fetch_add_explicit(&grad_output_weight[h], grad_logits.x * hidden_v, memory_order_relaxed);
            atomic_fetch_add_explicit(&grad_output_weight[hdim + h], grad_logits.y * hidden_v, memory_order_relaxed);
            atomic_fetch_add_explicit(&grad_output_weight[2u * hdim + h], grad_logits.z * hidden_v, memory_order_relaxed);
          }
        }
        if (reduce_colorizer_grad_simd) {
          simd_reduced_atomic_add(&grad_output_bias[0u], grad_logits.x, simd_lane);
          simd_reduced_atomic_add(&grad_output_bias[1u], grad_logits.y, simd_lane);
          simd_reduced_atomic_add(&grad_output_bias[2u], grad_logits.z, simd_lane);
        } else {
          atomic_fetch_add_explicit(&grad_output_bias[0u], grad_logits.x, memory_order_relaxed);
          atomic_fetch_add_explicit(&grad_output_bias[1u], grad_logits.y, memory_order_relaxed);
          atomic_fetch_add_explicit(&grad_output_bias[2u], grad_logits.z, memory_order_relaxed);
        }
      }
      for (uint h = 0u; h < hdim; ++h) {
        float grad_hidden =
            grad_logits.x * output_weight[h] +
            grad_logits.y * output_weight[hdim + h] +
            grad_logits.z * output_weight[2u * hdim + h];
        grad_hidden_pre_cache[h] = grad_hidden * gelu_exact_grad(hidden_pre_cache[h]);
      }
      if (compute_colorizer_grad) {
        for (uint h = 0u; h < hdim; ++h) {
          float grad_hidden_pre = grad_hidden_pre_cache[h];
          uint hidden_base = h * fdim;
          for (uint c = 0u; c < fdim; ++c) {
            float grad_weight = grad_hidden_pre * pixel_feature_cache[c];
            if (reduce_colorizer_grad_simd) {
              simd_reduced_atomic_add(&grad_hidden_weight[hidden_base + c], grad_weight, simd_lane);
            } else {
              atomic_fetch_add_explicit(
                  &grad_hidden_weight[hidden_base + c],
                  grad_weight,
                  memory_order_relaxed);
            }
          }
          if (reduce_colorizer_grad_simd) {
            simd_reduced_atomic_add(&grad_hidden_bias[h], grad_hidden_pre, simd_lane);
          } else {
            atomic_fetch_add_explicit(&grad_hidden_bias[h], grad_hidden_pre, memory_order_relaxed);
          }
        }
      }
      if (!skip_hidden_feature_vjp) {
        if (vec4_hidden_feature_vjp) {
          for (uint c = 0u; c < fdim; c += 4u) {
            float4 v = float4(0.0f);
            for (uint h = 0u; h < hdim; ++h) {
              uint hidden_base = h * fdim + c;
              float g = grad_hidden_pre_cache[h];
              float4 w = float4(
                  hidden_weight[hidden_base + 0u],
                  (c + 1u < fdim) ? hidden_weight[hidden_base + 1u] : 0.0f,
                  (c + 2u < fdim) ? hidden_weight[hidden_base + 2u] : 0.0f,
                  (c + 3u < fdim) ? hidden_weight[hidden_base + 3u] : 0.0f);
              v += g * w;
            }
            grad_feature_cache[c + 0u] = v.x;
            if (c + 1u < fdim) grad_feature_cache[c + 1u] = v.y;
            if (c + 2u < fdim) grad_feature_cache[c + 2u] = v.z;
            if (c + 3u < fdim) grad_feature_cache[c + 3u] = v.w;
          }
        } else if (rowmajor_hidden_feature_vjp) {
          for (uint h = 0u; h < hdim; ++h) {
            float g = grad_hidden_pre_cache[h];
            uint hidden_base = h * fdim;
            for (uint c = 0u; c < fdim; ++c) {
              grad_feature_cache[c] += g * hidden_weight[hidden_base + c];
            }
          }
        } else {
          for (uint c = 0u; c < fdim; ++c) {
            float v = 0.0f;
            for (uint h = 0u; h < hdim; ++h) {
              v += grad_hidden_pre_cache[h] * hidden_weight[h * fdim + c];
            }
            grad_feature_cache[c] = v;
          }
        }
      }
    }
  }

  float dT_next = skip_geometry_grad ? 0.0f : -dot(grad_rgb, splat_rgb);
  for (int si = int(ordered_count) - 1; si >= 0; --si) {
    uint i = uint(si);
    if (!processed[i]) continue;
    uint tube_id = ordered_ids[i];
    float alpha = alpha_values[i];
    float t_i = t_before[i];
    uint tube_feature_base = tube_id * fdim;

    float feature_scale = t_i * alpha;
    if (!skip_feature_grad) {
      for (uint c = 0u; c < fdim; ++c) {
        atomic_fetch_add_explicit(
            &grad_feature[tube_feature_base + c],
            grad_feature_cache[c] * feature_scale,
          memory_order_relaxed);
      }
    }
    if (skip_geometry_grad) continue;

    float grad_dot_feature = 0.0f;
    for (uint c = 0u; c < fdim; ++c) {
      grad_dot_feature += grad_feature_cache[c] * feature[tube_feature_base + c];
    }
    float d_alpha = t_i * grad_dot_feature - dT_next * t_i;
    float dT_i = alpha * grad_dot_feature + dT_next * (1.0f - alpha);
    dT_next = dT_i;

    if (!differentiable_alpha[i]) continue;

    float3 d = sample_a - load3(ma, tube_id);
    float qv = quadratic_q(q_uvt, tube_id, d);
    float exp_term = exp(-0.5f * qv);
    float grad_qv = d_alpha * primitive_alpha_d_qv(opacity[tube_id], exp_term, mf);
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
    atomic_fetch_add_explicit(
        &grad_opacity[tube_id],
        d_alpha * primitive_alpha_d_opacity(opacity[tube_id], exp_term, mf),
        memory_order_relaxed);
  }
}

kernel void clear_linear_colorizer_gradients(
    device float* grad_color_weight [[buffer(0)]],
    device float* grad_color_bias [[buffer(1)]],
    constant MetaI32& mi [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
  uint fdim = feature_dim(mi);
  uint weight_count = 3u * fdim;
  if (gid < weight_count) {
    grad_color_weight[gid] = 0.0f;
  }
  if (gid < 3u) {
    grad_color_bias[gid] = 0.0f;
  }
}

kernel void direct_atomic_feature_linear_sigmoid_mse_backward(
    const device float* ma [[buffer(0)]],
    const device float* q_uvt [[buffer(1)]],
    const device float* depth0 [[buffer(2)]],
    const device float* depth_beta [[buffer(3)]],
    const device float* opacity [[buffer(4)]],
    const device float* feature [[buffer(5)]],
    const device float* target_rgb [[buffer(6)]],
    const device float* color_weight [[buffer(7)]],
    const device float* color_bias [[buffer(8)]],
    constant MetaI32& mi [[buffer(9)]],
    constant MetaF32& mf [[buffer(10)]],
    const device atomic_uint* tile_counts [[buffer(11)]],
    const device uint* tile_tube_ids [[buffer(12)]],
    const device float* tile_depths [[buffer(13)]],
    device atomic_uint* tile_unstable [[buffer(14)]],
    device atomic_float* grad_ma [[buffer(15)]],
    device atomic_float* grad_q [[buffer(16)]],
    device atomic_float* grad_opacity [[buffer(17)]],
    device atomic_float* grad_feature [[buffer(18)]],
    device atomic_float* grad_color_weight [[buffer(19)]],
    device atomic_float* grad_color_bias [[buffer(20)]],
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

  uint fdim = feature_dim(mi);
  bool skip_colorizer_grad = (mi.reserved1 & 16) != 0;
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
  thread float pixel_feature_cache[STAR_FEATURE_GRAD_CACHE_CAP];
  thread float grad_feature_cache[STAR_FEATURE_GRAD_CACHE_CAP];
  for (uint c = 0u; c < fdim; ++c) {
    pixel_feature_cache[c] = 0.0f;
    grad_feature_cache[c] = 0.0f;
  }

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
    float alpha_raw = primitive_alpha_raw(opacity[tube_id], exp(-0.5f * qv), mf);
    float alpha = min(mf.max_alpha, alpha_raw);
    if (!(alpha >= mf.alpha_threshold)) continue;
    t_before[i] = T;
    alpha_values[i] = alpha;
    processed[i] = true;
    differentiable_alpha[i] = alpha_raw < mf.max_alpha;
    uint tube_feature_base = tube_id * fdim;
    float feature_scale = T * alpha;
    for (uint c = 0u; c < fdim; ++c) {
      pixel_feature_cache[c] += feature_scale * feature[tube_feature_base + c];
    }
    T *= (1.0f - alpha);
    if (T <= mf.transmittance_threshold) break;
  }

  uint pixel_id = (f * uint(mi.height) + y) * uint(mi.width) + x;
  uint target_base = pixel_id * 3u;
  float alpha_out = 1.0f - T;
  float3 logits = float3(color_bias[0u], color_bias[1u], color_bias[2u]);
  for (uint c = 0u; c < fdim; ++c) {
    float v = pixel_feature_cache[c];
    logits.x += color_weight[c] * v;
    logits.y += color_weight[fdim + c] * v;
    logits.z += color_weight[2u * fdim + c] * v;
  }
  float3 splat_rgb = 1.0f / (1.0f + exp(-logits));
  float3 rgb = alpha_out * splat_rgb;
  float3 target = float3(target_rgb[target_base + 0u], target_rgb[target_base + 1u], target_rgb[target_base + 2u]);
  float inv_n = 1.0f / max(1.0f, float(mi.frames * mi.height * mi.width * 3));
  float3 grad_rgb = (2.0f * inv_n) * (rgb - target);
  float3 grad_logits = grad_rgb * alpha_out * splat_rgb * (1.0f - splat_rgb);
  for (uint c = 0u; c < fdim; ++c) {
    float v = pixel_feature_cache[c];
    if (!skip_colorizer_grad) {
      atomic_fetch_add_explicit(&grad_color_weight[c], grad_logits.x * v, memory_order_relaxed);
      atomic_fetch_add_explicit(&grad_color_weight[fdim + c], grad_logits.y * v, memory_order_relaxed);
      atomic_fetch_add_explicit(&grad_color_weight[2u * fdim + c], grad_logits.z * v, memory_order_relaxed);
    }
    grad_feature_cache[c] =
        grad_logits.x * color_weight[c] +
        grad_logits.y * color_weight[fdim + c] +
        grad_logits.z * color_weight[2u * fdim + c];
  }
  if (!skip_colorizer_grad) {
    atomic_fetch_add_explicit(&grad_color_bias[0u], grad_logits.x, memory_order_relaxed);
    atomic_fetch_add_explicit(&grad_color_bias[1u], grad_logits.y, memory_order_relaxed);
    atomic_fetch_add_explicit(&grad_color_bias[2u], grad_logits.z, memory_order_relaxed);
  }

  float dT_next = -dot(grad_rgb, splat_rgb);
  for (int si = int(ordered_count) - 1; si >= 0; --si) {
    uint i = uint(si);
    if (!processed[i]) continue;
    uint tube_id = ordered_ids[i];
    float alpha = alpha_values[i];
    float t_i = t_before[i];
    uint tube_feature_base = tube_id * fdim;

    float grad_dot_feature = 0.0f;
    for (uint c = 0u; c < fdim; ++c) {
      grad_dot_feature += grad_feature_cache[c] * feature[tube_feature_base + c];
    }
    float d_alpha = t_i * grad_dot_feature - dT_next * t_i;
    float dT_i = alpha * grad_dot_feature + dT_next * (1.0f - alpha);
    dT_next = dT_i;

    float feature_scale = t_i * alpha;
    for (uint c = 0u; c < fdim; ++c) {
      atomic_fetch_add_explicit(&grad_feature[tube_feature_base + c], grad_feature_cache[c] * feature_scale, memory_order_relaxed);
    }
    if (!differentiable_alpha[i]) continue;

    float3 d = sample_a - load3(ma, tube_id);
    float qv = quadratic_q(q_uvt, tube_id, d);
    float exp_term = exp(-0.5f * qv);
    float grad_qv = d_alpha * primitive_alpha_d_qv(opacity[tube_id], exp_term, mf);
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
    atomic_fetch_add_explicit(
        &grad_opacity[tube_id],
        d_alpha * primitive_alpha_d_opacity(opacity[tube_id], exp_term, mf),
        memory_order_relaxed);
  }
}

kernel void linear_sigmoid_mse_handoff_prep(
    const device float* feature_image [[buffer(0)]],
    const device float* alpha_image [[buffer(1)]],
    const device float* target_rgb [[buffer(2)]],
    const device float* color_weight [[buffer(3)]],
    const device float* color_bias [[buffer(4)]],
    constant int* prep_meta [[buffer(5)]],
    device float* grad_logits_image [[buffer(6)]],
    device float* grad_alpha_image [[buffer(7)]],
    uint gid [[thread_position_in_grid]]) {
  uint frames = uint(prep_meta[0]);
  uint fdim = uint(prep_meta[1]);
  uint height = uint(prep_meta[2]);
  uint width = uint(prep_meta[3]);
  uint pixel_count = frames * height * width;
  if (gid >= pixel_count) return;

  uint x = gid % width;
  uint y = (gid / width) % height;
  uint f = gid / (height * width);
  float3 logits = float3(color_bias[0u], color_bias[1u], color_bias[2u]);
  for (uint c = 0u; c < fdim; ++c) {
    uint feature_idx = ((f * fdim + c) * height + y) * width + x;
    float v = feature_image[feature_idx];
    logits.x += color_weight[c] * v;
    logits.y += color_weight[fdim + c] * v;
    logits.z += color_weight[2u * fdim + c] * v;
  }

  float3 splat_rgb = 1.0f / (1.0f + exp(-logits));
  float alpha = alpha_image[gid];
  uint target_base = ((f * 3u) * height + y) * width + x;
  float3 target = float3(
      target_rgb[target_base],
      target_rgb[((f * 3u + 1u) * height + y) * width + x],
      target_rgb[((f * 3u + 2u) * height + y) * width + x]);
  float3 rgb = alpha * splat_rgb;
  float inv_n = 1.0f / max(1.0f, float(frames * height * width * 3u));
  float3 grad_rgb = (2.0f * inv_n) * (rgb - target);
  float3 grad_logits = grad_rgb * alpha * splat_rgb * (1.0f - splat_rgb);
  uint logit_base = gid * 3u;
  grad_logits_image[logit_base + 0u] = grad_logits.x;
  grad_logits_image[logit_base + 1u] = grad_logits.y;
  grad_logits_image[logit_base + 2u] = grad_logits.z;
  grad_alpha_image[gid] = dot(grad_rgb, splat_rgb);
}

kernel void direct_atomic_feature_hidden_sigmoid_mse_backward(
    const device float* ma [[buffer(0)]],
    const device float* q_uvt [[buffer(1)]],
    const device float* depth0 [[buffer(2)]],
    const device float* depth_beta [[buffer(3)]],
    const device float* opacity [[buffer(4)]],
    const device float* feature [[buffer(5)]],
    const device float* target_rgb [[buffer(6)]],
    const device float* hidden_weight [[buffer(7)]],
    const device float* hidden_bias [[buffer(8)]],
    const device float* output_weight [[buffer(9)]],
    const device float* output_bias [[buffer(10)]],
    constant int* hidden_meta [[buffer(11)]],
    constant MetaI32& mi [[buffer(12)]],
    constant MetaF32& mf [[buffer(13)]],
    const device atomic_uint* tile_counts [[buffer(14)]],
    const device uint* tile_tube_ids [[buffer(15)]],
    const device float* tile_depths [[buffer(16)]],
    device atomic_uint* tile_unstable [[buffer(17)]],
    device atomic_float* grad_ma [[buffer(18)]],
    device atomic_float* grad_q [[buffer(19)]],
    device atomic_float* grad_opacity [[buffer(20)]],
    device atomic_float* grad_feature [[buffer(21)]],
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  uint tile_id = gid / STAR_THREADS;
  if (tile_id >= uint(mi.tile_count)) return;
  uint local_tid = tid;

  uint tx, ty, tz;
  decode_tile(tile_id, mi, tx, ty, tz);

  threadgroup uint local_ids[STAR_TILE_CAPACITY];
  threadgroup float local_depths[STAR_TILE_CAPACITY];
  threadgroup float partial_features[STAR_SIMDGROUPS * STAR_FEATURE_GRAD_CACHE_CAP];
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
  bool pixel_valid = x < uint(mi.width) && y < uint(mi.height) && f < uint(mi.frames);
  bool reduce_feature_grad_atomic = ((mi.reserved1 & 4) != 0) && !unstable;
  bool reduce_feature_grad_atomic_vec4 = ((mi.reserved1 & 16) != 0) && !unstable;
  if (!pixel_valid && !reduce_feature_grad_atomic && !reduce_feature_grad_atomic_vec4) return;

  uint fdim = feature_dim(mi);
  uint hdim = uint(hidden_meta[0]);
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
  thread float pixel_feature_cache[STAR_FEATURE_GRAD_CACHE_CAP];
  thread float grad_feature_cache[STAR_FEATURE_GRAD_CACHE_CAP];
  thread float hidden_pre_cache[STAR_FEATURE_GRAD_CACHE_CAP];
  thread float hidden_cache[STAR_FEATURE_GRAD_CACHE_CAP];
  thread float grad_hidden_pre_cache[STAR_FEATURE_GRAD_CACHE_CAP];
  for (uint c = 0u; c < fdim; ++c) {
    pixel_feature_cache[c] = 0.0f;
    grad_feature_cache[c] = 0.0f;
  }

  float T = 1.0f;
  for (uint i = 0u; i < STAR_TILE_CAPACITY; ++i) {
    t_before[i] = 0.0f;
    alpha_values[i] = 0.0f;
    processed[i] = false;
    differentiable_alpha[i] = false;
  }
  if (pixel_valid) {
    for (uint i = 0u; i < ordered_count; ++i) {
      uint tube_id = ordered_ids[i];
      float3 d = sample_a - load3(ma, tube_id);
      float qv = quadratic_q(q_uvt, tube_id, d);
      if (!isfinite(qv)) continue;
      float alpha_raw = primitive_alpha_raw(opacity[tube_id], exp(-0.5f * qv), mf);
      float alpha = min(mf.max_alpha, alpha_raw);
      if (!(alpha >= mf.alpha_threshold)) continue;
      t_before[i] = T;
      alpha_values[i] = alpha;
      processed[i] = true;
      differentiable_alpha[i] = alpha_raw < mf.max_alpha;
      uint tube_feature_base = tube_id * fdim;
      float feature_scale = T * alpha;
      for (uint c = 0u; c < fdim; ++c) {
        pixel_feature_cache[c] += feature_scale * feature[tube_feature_base + c];
      }
      T *= (1.0f - alpha);
      if (T <= mf.transmittance_threshold) break;
    }
  }

  float alpha_out = pixel_valid ? (1.0f - T) : 0.0f;
  for (uint h = 0u; h < hdim; ++h) {
    float v = hidden_bias[h];
    uint hidden_base = h * fdim;
    for (uint c = 0u; c < fdim; ++c) {
      v += hidden_weight[hidden_base + c] * pixel_feature_cache[c];
    }
    hidden_pre_cache[h] = v;
    hidden_cache[h] = gelu_exact_forward(v);
  }
  float3 logits = float3(output_bias[0u], output_bias[1u], output_bias[2u]);
  for (uint h = 0u; h < hdim; ++h) {
    float v = hidden_cache[h];
    logits.x += output_weight[h] * v;
    logits.y += output_weight[hdim + h] * v;
    logits.z += output_weight[2u * hdim + h] * v;
  }
  float3 splat_rgb = 1.0f / (1.0f + exp(-logits));
  uint pixel_id = (f * uint(mi.height) + y) * uint(mi.width) + x;
  uint target_base = pixel_id * 3u;
  float3 target = pixel_valid ? float3(
      target_rgb[target_base + 0u],
      target_rgb[target_base + 1u],
      target_rgb[target_base + 2u]) : float3(0.0f);
  float3 rgb = alpha_out * splat_rgb;
  float inv_n = 1.0f / max(1.0f, float(mi.frames * mi.height * mi.width * 3));
  float3 grad_rgb = pixel_valid ? (2.0f * inv_n) * (rgb - target) : float3(0.0f);
  float3 grad_logits = grad_rgb * alpha_out * splat_rgb * (1.0f - splat_rgb);
  for (uint h = 0u; h < hdim; ++h) {
    float grad_hidden =
        grad_logits.x * output_weight[h] +
        grad_logits.y * output_weight[hdim + h] +
        grad_logits.z * output_weight[2u * hdim + h];
    grad_hidden_pre_cache[h] = grad_hidden * gelu_exact_grad(hidden_pre_cache[h]);
  }
  for (uint c = 0u; c < fdim; ++c) {
    float v = 0.0f;
    for (uint h = 0u; h < hdim; ++h) {
      v += grad_hidden_pre_cache[h] * hidden_weight[h * fdim + c];
    }
    grad_feature_cache[c] = v;
  }

  float dT_next = pixel_valid ? -dot(grad_rgb, splat_rgb) : 0.0f;
  for (int si = int(ordered_count) - 1; si >= 0; --si) {
    uint i = uint(si);
    bool active = pixel_valid && processed[i];
    uint tube_id = ordered_ids[i];
    float alpha = active ? alpha_values[i] : 0.0f;
    float t_i = active ? t_before[i] : 0.0f;
    uint tube_feature_base = tube_id * fdim;

    float grad_dot_feature = 0.0f;
    if (active) {
      for (uint c = 0u; c < fdim; ++c) {
        grad_dot_feature += grad_feature_cache[c] * feature[tube_feature_base + c];
      }
    }
    float d_alpha = t_i * grad_dot_feature - dT_next * t_i;
    float dT_i = alpha * grad_dot_feature + dT_next * (1.0f - alpha);
    dT_next = dT_i;

    float feature_scale = t_i * alpha;
    if (reduce_feature_grad_atomic_vec4) {
      reduce_atomic_add_feature_grads_cached_vec4(
          grad_feature,
          grad_feature_cache,
          tube_id,
          feature_scale,
          active,
          mi,
          simd_lane,
          simd_group,
          partial_features);
    } else if (reduce_feature_grad_atomic) {
      reduce_atomic_add_feature_grads_cached(
          grad_feature,
          grad_feature_cache,
          tube_id,
          feature_scale,
          active,
          mi,
          simd_lane,
          simd_group,
          partial_features);
    } else if (active) {
      for (uint c = 0u; c < fdim; ++c) {
        atomic_fetch_add_explicit(&grad_feature[tube_feature_base + c], grad_feature_cache[c] * feature_scale, memory_order_relaxed);
      }
    }
    if (!active || !differentiable_alpha[i]) continue;

    float3 d = sample_a - load3(ma, tube_id);
    float qv = quadratic_q(q_uvt, tube_id, d);
    float exp_term = exp(-0.5f * qv);
    float grad_qv = d_alpha * primitive_alpha_d_qv(opacity[tube_id], exp_term, mf);
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
    atomic_fetch_add_explicit(
        &grad_opacity[tube_id],
        d_alpha * primitive_alpha_d_opacity(opacity[tube_id], exp_term, mf),
        memory_order_relaxed);
  }
}

kernel void direct_atomic_feature_logit_handoff_backward(
    const device float* ma [[buffer(0)]],
    const device float* q_uvt [[buffer(1)]],
    const device float* depth0 [[buffer(2)]],
    const device float* depth_beta [[buffer(3)]],
    const device float* opacity [[buffer(4)]],
    const device float* feature [[buffer(5)]],
    const device float* grad_logits_image [[buffer(6)]],
    const device float* grad_alpha_image [[buffer(7)]],
    const device float* color_weight [[buffer(8)]],
    constant MetaI32& mi [[buffer(9)]],
    constant MetaF32& mf [[buffer(10)]],
    const device atomic_uint* tile_counts [[buffer(11)]],
    const device uint* tile_tube_ids [[buffer(12)]],
    const device float* tile_depths [[buffer(13)]],
    device atomic_uint* tile_unstable [[buffer(14)]],
    device atomic_float* grad_ma [[buffer(15)]],
    device atomic_float* grad_q [[buffer(16)]],
    device atomic_float* grad_opacity [[buffer(17)]],
    device atomic_float* grad_feature [[buffer(18)]],
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  uint tile_id = gid / STAR_THREADS;
  if (tile_id >= uint(mi.tile_count)) return;
  uint local_tid = tid;

  uint tx, ty, tz;
  decode_tile(tile_id, mi, tx, ty, tz);

  threadgroup uint local_ids[STAR_TILE_CAPACITY];
  threadgroup float local_depths[STAR_TILE_CAPACITY];
  threadgroup float partial_features[STAR_SIMDGROUPS * STAR_FEATURE_GRAD_CACHE_CAP];
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
  bool pixel_valid = x < uint(mi.width) && y < uint(mi.height) && f < uint(mi.frames);
  bool reduce_feature_grad_atomic = ((mi.reserved1 & 4) != 0) && !unstable;
  bool reduce_feature_grad_atomic_vec4 = ((mi.reserved1 & 16) != 0) && !unstable;
  if (!pixel_valid && !reduce_feature_grad_atomic && !reduce_feature_grad_atomic_vec4) return;

  uint fdim = feature_dim(mi);
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
  thread float grad_feature_cache[STAR_FEATURE_GRAD_CACHE_CAP];
  uint pixel_id = (f * uint(mi.height) + y) * uint(mi.width) + x;
  uint grad_logits_base = pixel_id * 3u;
  float3 grad_logits = pixel_valid ? float3(
      grad_logits_image[grad_logits_base + 0u],
      grad_logits_image[grad_logits_base + 1u],
      grad_logits_image[grad_logits_base + 2u]) : float3(0.0f);
  for (uint c = 0u; c < fdim; ++c) {
    grad_feature_cache[c] =
        grad_logits.x * color_weight[c] +
        grad_logits.y * color_weight[fdim + c] +
        grad_logits.z * color_weight[2u * fdim + c];
  }

  float T = 1.0f;
  for (uint i = 0u; i < STAR_TILE_CAPACITY; ++i) {
    t_before[i] = 0.0f;
    alpha_values[i] = 0.0f;
    processed[i] = false;
    differentiable_alpha[i] = false;
  }
  if (pixel_valid) {
    for (uint i = 0u; i < ordered_count; ++i) {
      uint tube_id = ordered_ids[i];
      float3 d = sample_a - load3(ma, tube_id);
      float qv = quadratic_q(q_uvt, tube_id, d);
      if (!isfinite(qv)) continue;
      float alpha_raw = primitive_alpha_raw(opacity[tube_id], exp(-0.5f * qv), mf);
      float alpha = min(mf.max_alpha, alpha_raw);
      if (!(alpha >= mf.alpha_threshold)) continue;
      t_before[i] = T;
      alpha_values[i] = alpha;
      processed[i] = true;
      differentiable_alpha[i] = alpha_raw < mf.max_alpha;
      T *= (1.0f - alpha);
      if (T <= mf.transmittance_threshold) break;
    }
  }

  float dT_next = pixel_valid ? -grad_alpha_image[pixel_id] : 0.0f;
  for (int si = int(ordered_count) - 1; si >= 0; --si) {
    uint i = uint(si);
    bool active = pixel_valid && processed[i];
    uint tube_id = ordered_ids[i];
    float alpha = active ? alpha_values[i] : 0.0f;
    float t_i = active ? t_before[i] : 0.0f;
    uint tube_feature_base = tube_id * fdim;

    float grad_dot_feature = 0.0f;
    if (active) {
      for (uint c = 0u; c < fdim; ++c) {
        grad_dot_feature += grad_feature_cache[c] * feature[tube_feature_base + c];
      }
    }
    float d_alpha = t_i * grad_dot_feature - dT_next * t_i;
    float dT_i = alpha * grad_dot_feature + dT_next * (1.0f - alpha);
    dT_next = dT_i;

    float feature_scale = t_i * alpha;
    if (reduce_feature_grad_atomic_vec4) {
      reduce_atomic_add_feature_grads_cached_vec4(
          grad_feature,
          grad_feature_cache,
          tube_id,
          feature_scale,
          active,
          mi,
          simd_lane,
          simd_group,
          partial_features);
    } else if (reduce_feature_grad_atomic) {
      reduce_atomic_add_feature_grads_cached(
          grad_feature,
          grad_feature_cache,
          tube_id,
          feature_scale,
          active,
          mi,
          simd_lane,
          simd_group,
          partial_features);
    } else if (active) {
      for (uint c = 0u; c < fdim; ++c) {
        atomic_fetch_add_explicit(&grad_feature[tube_feature_base + c], grad_feature_cache[c] * feature_scale, memory_order_relaxed);
      }
    }
    if (!active || !differentiable_alpha[i]) continue;

    float3 d = sample_a - load3(ma, tube_id);
    float qv = quadratic_q(q_uvt, tube_id, d);
    float exp_term = exp(-0.5f * qv);
    float grad_qv = d_alpha * primitive_alpha_d_qv(opacity[tube_id], exp_term, mf);
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
    atomic_fetch_add_explicit(
        &grad_opacity[tube_id],
        d_alpha * primitive_alpha_d_opacity(opacity[tube_id], exp_term, mf),
        memory_order_relaxed);
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
    float alpha_raw = primitive_alpha_raw(opacity[tube_id], exp(-0.5f * qv), mf);
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
    float grad_qv = d_alpha * primitive_alpha_d_qv(opacity[tube_id], exp_term, mf);
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
    atomic_add_fixedpoint(
        grad_opacity,
        tube_id,
        d_alpha * primitive_alpha_d_opacity(opacity[tube_id], exp_term, mf));
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
    float alpha_raw = primitive_alpha_raw(opacity[tube_id], exp(-0.5f * qv), mf);
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
    float grad_qv = d_alpha * primitive_alpha_d_qv(opacity[tube_id], exp_term, mf);
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
    atomic_add_split_fixedpoint(
        grad_opacity_coarse,
        grad_opacity_fine,
        tube_id,
        d_alpha * primitive_alpha_d_opacity(opacity[tube_id], exp_term, mf));
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
                  float alpha_raw = primitive_alpha_raw(opacity[ordered_tube], exp(-0.5f * qv), mf);
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
                    float grad_qv = d_alpha * primitive_alpha_d_qv(opacity[tube_id], exp_term, mf);
                    float3 qd = load_q_row0(q_uvt, tube_id) * d.x + load_q_row1(q_uvt, tube_id) * d.y + load_q_row2(q_uvt, tube_id) * d.z;
                    grad_m_sum += -2.0f * grad_qv * qd;
                    q_sum0 += grad_qv * d.x * d.x;
                    q_sum1 += grad_qv * 2.0f * d.x * d.y;
                    q_sum2 += grad_qv * 2.0f * d.x * d.z;
                    q_sum3 += grad_qv * d.y * d.y;
                    q_sum4 += grad_qv * 2.0f * d.y * d.z;
                    q_sum5 += grad_qv * d.z * d.z;
                    opacity_sum +=
                        d_alpha * primitive_alpha_d_opacity(opacity[tube_id], exp_term, mf);
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
            float alpha_raw = primitive_alpha_raw(opacity[tube_id], exp(-0.5f * qv), mf);
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
              float alpha_raw = primitive_alpha_raw(opacity[tube_id], exp(-0.5f * qv), mf);
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
            float grad_qv =
                d_alpha * primitive_alpha_d_qv(opacity[target_id], exp_term, mf);
            float3 qd = load_q_row0(q_uvt, target_id) * d.x + load_q_row1(q_uvt, target_id) * d.y + load_q_row2(q_uvt, target_id) * d.z;
            float3 grad_m_value = -2.0f * grad_qv * qd;
            float q_value0 = grad_qv * d.x * d.x;
            float q_value1 = grad_qv * 2.0f * d.x * d.y;
            float q_value2 = grad_qv * 2.0f * d.x * d.z;
            float q_value3 = grad_qv * d.y * d.y;
            float q_value4 = grad_qv * 2.0f * d.y * d.z;
            float q_value5 = grad_qv * d.z * d.z;
            float opacity_value =
                d_alpha * primitive_alpha_d_opacity(opacity[target_id], exp_term, mf);
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
          float alpha_raw = primitive_alpha_raw(opacity[tube_id], exp(-0.5f * qv), mf);
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
          float grad_qv = d_alpha * primitive_alpha_d_qv(opacity[tube_id], exp_term, mf);
          float3 qd = load_q_row0(q_uvt, tube_id) * d.x + load_q_row1(q_uvt, tube_id) * d.y + load_q_row2(q_uvt, tube_id) * d.z;
          float3 grad_m_value = -2.0f * grad_qv * qd;
          float q_value0 = grad_qv * d.x * d.x;
          float q_value1 = grad_qv * 2.0f * d.x * d.y;
          float q_value2 = grad_qv * 2.0f * d.x * d.z;
          float q_value3 = grad_qv * d.y * d.y;
          float q_value4 = grad_qv * 2.0f * d.y * d.z;
          float q_value5 = grad_qv * d.z * d.z;
          float opacity_value =
              d_alpha * primitive_alpha_d_opacity(opacity[tube_id], exp_term, mf);
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
      float alpha_raw = primitive_alpha_raw(opacity[tube_id], exp(-0.5f * qv), mf);
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
      float grad_qv = d_alpha * primitive_alpha_d_qv(opacity[tube_id], exp_term, mf);
      float3 qd = load_q_row0(q_uvt, tube_id) * d.x + load_q_row1(q_uvt, tube_id) * d.y + load_q_row2(q_uvt, tube_id) * d.z;
      grad_m_sum += -2.0f * grad_qv * qd;
      q_sum0 += grad_qv * d.x * d.x;
      q_sum1 += grad_qv * 2.0f * d.x * d.y;
      q_sum2 += grad_qv * 2.0f * d.x * d.z;
      q_sum3 += grad_qv * d.y * d.y;
      q_sum4 += grad_qv * 2.0f * d.y * d.z;
      q_sum5 += grad_qv * d.z * d.z;
      opacity_sum += d_alpha * primitive_alpha_d_opacity(opacity[tube_id], exp_term, mf);
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
          float alpha_raw = primitive_alpha_raw(opacity[tube_id], exp(-0.5f * qv), mf);
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
          float grad_qv = d_alpha * primitive_alpha_d_qv(opacity[tube_id], exp_term, mf);
          float3 qd = load_q_row0(q_uvt, tube_id) * d.x + load_q_row1(q_uvt, tube_id) * d.y + load_q_row2(q_uvt, tube_id) * d.z;
          grad_m_sum += -2.0f * grad_qv * qd;
          q_sum0 += grad_qv * d.x * d.x;
          q_sum1 += grad_qv * 2.0f * d.x * d.y;
          q_sum2 += grad_qv * 2.0f * d.x * d.z;
          q_sum3 += grad_qv * d.y * d.y;
          q_sum4 += grad_qv * 2.0f * d.y * d.z;
          q_sum5 += grad_qv * d.z * d.z;
          opacity_sum += d_alpha * primitive_alpha_d_opacity(opacity[tube_id], exp_term, mf);
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
          float alpha_raw = primitive_alpha_raw(opacity[tube_id], exp(-0.5f * qv), mf);
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
          float grad_qv = d_alpha * primitive_alpha_d_qv(opacity[tube_id], exp_term, mf);
          float3 qd = load_q_row0(q_uvt, tube_id) * d.x + load_q_row1(q_uvt, tube_id) * d.y + load_q_row2(q_uvt, tube_id) * d.z;
          grad_m_sum += -2.0f * grad_qv * qd;
          q_sum0 += grad_qv * d.x * d.x;
          q_sum1 += grad_qv * 2.0f * d.x * d.y;
          q_sum2 += grad_qv * 2.0f * d.x * d.z;
          q_sum3 += grad_qv * d.y * d.y;
          q_sum4 += grad_qv * 2.0f * d.y * d.z;
          q_sum5 += grad_qv * d.z * d.z;
          opacity_sum += d_alpha * primitive_alpha_d_opacity(opacity[tube_id], exp_term, mf);
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
        float alpha_raw = primitive_alpha_raw(opacity[tube_id], exp(-0.5f * qv), mf);
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
        float grad_qv = d_alpha * primitive_alpha_d_qv(opacity[tube_id], exp_term, mf);
        float3 qd = load_q_row0(q_uvt, tube_id) * d.x + load_q_row1(q_uvt, tube_id) * d.y + load_q_row2(q_uvt, tube_id) * d.z;
        grad_m_sum += -2.0f * grad_qv * qd;
        q_sum0 += grad_qv * d.x * d.x;
        q_sum1 += grad_qv * 2.0f * d.x * d.y;
        q_sum2 += grad_qv * 2.0f * d.x * d.z;
        q_sum3 += grad_qv * d.y * d.y;
        q_sum4 += grad_qv * 2.0f * d.y * d.z;
        q_sum5 += grad_qv * d.z * d.z;
        opacity_sum += d_alpha * primitive_alpha_d_opacity(opacity[tube_id], exp_term, mf);
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
            float alpha_raw = primitive_alpha_raw(opacity[tube_id], exp(-0.5f * qv), mf);
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
            float grad_qv = d_alpha * primitive_alpha_d_qv(opacity[tube_id], exp_term, mf);
            float3 qd = load_q_row0(q_uvt, tube_id) * d.x + load_q_row1(q_uvt, tube_id) * d.y + load_q_row2(q_uvt, tube_id) * d.z;
            grad_m_sum += -2.0f * grad_qv * qd;
            q_sum0 += grad_qv * d.x * d.x;
            q_sum1 += grad_qv * 2.0f * d.x * d.y;
            q_sum2 += grad_qv * 2.0f * d.x * d.z;
            q_sum3 += grad_qv * d.y * d.y;
            q_sum4 += grad_qv * 2.0f * d.y * d.z;
            q_sum5 += grad_qv * d.z * d.z;
            opacity_sum += d_alpha * primitive_alpha_d_opacity(opacity[tube_id], exp_term, mf);
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
      float alpha_raw = primitive_alpha_raw(opacity[tube_id], exp(-0.5f * qv), mf);
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
      float grad_qv = d_alpha * primitive_alpha_d_qv(opacity[tube_id], exp_term, mf);
      float3 qd = load_q_row0(q_uvt, tube_id) * d.x + load_q_row1(q_uvt, tube_id) * d.y + load_q_row2(q_uvt, tube_id) * d.z;
      float3 grad_m_value = -2.0f * grad_qv * qd;
      grad_m_sum += grad_m_value;
      q_sum0 += grad_qv * d.x * d.x;
      q_sum1 += grad_qv * 2.0f * d.x * d.y;
      q_sum2 += grad_qv * 2.0f * d.x * d.z;
      q_sum3 += grad_qv * d.y * d.y;
      q_sum4 += grad_qv * 2.0f * d.y * d.z;
      q_sum5 += grad_qv * d.z * d.z;
      opacity_sum += d_alpha * primitive_alpha_d_opacity(opacity[tube_id], exp_term, mf);
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

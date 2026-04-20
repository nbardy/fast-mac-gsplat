#include <metal_stdlib>
using namespace metal;

#define GSP_TILE_SIZE 16u
#define GSP_TILE_PIXELS 256u
#define GSP_THREADS 256u
#define GSP_FAST_CAP 2048u
#define GSP_CHUNK 64u
#define GSP_SIMD_WIDTH 32u
#define GSP_SIMDGROUPS 8u

struct MetaI32 {
  int height;
  int width;
  int tiles_y;
  int tiles_x;
  int tile_size;
  int gaussians;
  int tile_count;
  int max_fast_pairs;
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

inline float3 load3(const device float* base, uint idx) {
  uint b = idx * 3u;
  return float3(base[b + 0u], base[b + 1u], base[b + 2u]);
}

inline float3 load3_sh(const threadgroup float* base, uint idx) {
  uint b = idx * 3u;
  return float3(base[b + 0u], base[b + 1u], base[b + 2u]);
}

inline void atomic_add3(device atomic_float* base, uint idx, float3 v) {
  uint b = idx * 3u;
  atomic_fetch_add_explicit(&base[b + 0u], v.x, memory_order_relaxed);
  atomic_fetch_add_explicit(&base[b + 1u], v.y, memory_order_relaxed);
  atomic_fetch_add_explicit(&base[b + 2u], v.z, memory_order_relaxed);
}

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

inline float safe_det(float a, float b, float c, float eps) {
  return max(a * c - b * b, eps);
}

inline bool alpha_support_params(float opacity, constant MetaF32& mf, thread float& tau) {
  if (opacity <= mf.alpha_threshold) return false;
  float ratio = max(mf.alpha_threshold / max(opacity, mf.eps), mf.eps);
  tau = -2.0f * log(ratio);
  return isfinite(tau) && (tau > 0.0f);
}

inline int4 snugbox(float2 m, float a, float b, float c, float tau, constant MetaI32& mi, constant MetaF32& mf) {
  float det = safe_det(a, b, c, mf.eps);
  float half_x = sqrt(max(tau * c / det, 0.0f));
  float half_y = sqrt(max(tau * a / det, 0.0f));

  int x0 = max(0, int(floor(m.x - half_x - 0.5f)));
  int x1 = min(mi.width - 1, int(ceil(m.x + half_x - 0.5f)));
  int y0 = max(0, int(floor(m.y - half_y - 0.5f)));
  int y1 = min(mi.height - 1, int(ceil(m.y + half_y - 0.5f)));
  return int4(x0, y0, x1, y1);
}

inline bool ellipse_intersects_rect(float2 m, float a, float b, float c, float tau, float rx0, float ry0, float rx1, float ry1) {
  float dx0 = rx0 - m.x;
  float dx1 = rx1 - m.x;
  float dy0 = ry0 - m.y;
  float dy1 = ry1 - m.y;

  if (m.x >= rx0 && m.x <= rx1 && m.y >= ry0 && m.y <= ry1) return true;

  float qmin = INFINITY;
  qmin = min(qmin, a * dx0 * dx0 + 2.0f * b * dx0 * dy0 + c * dy0 * dy0);
  qmin = min(qmin, a * dx0 * dx0 + 2.0f * b * dx0 * dy1 + c * dy1 * dy1);
  qmin = min(qmin, a * dx1 * dx1 + 2.0f * b * dx1 * dy0 + c * dy0 * dy0);
  qmin = min(qmin, a * dx1 * dx1 + 2.0f * b * dx1 * dy1 + c * dy1 * dy1);
  if (c > 1e-8f) {
    float dy = clamp(-(b / c) * dx0, dy0, dy1);
    qmin = min(qmin, a * dx0 * dx0 + 2.0f * b * dx0 * dy + c * dy * dy);
    dy = clamp(-(b / c) * dx1, dy0, dy1);
    qmin = min(qmin, a * dx1 * dx1 + 2.0f * b * dx1 * dy + c * dy * dy);
  }
  if (a > 1e-8f) {
    float dx = clamp(-(b / a) * dy0, dx0, dx1);
    qmin = min(qmin, a * dx * dx + 2.0f * b * dx * dy0 + c * dy0 * dy0);
    dx = clamp(-(b / a) * dy1, dx0, dx1);
    qmin = min(qmin, a * dx * dx + 2.0f * b * dx * dy1 + c * dy1 * dy1);
  }
  return qmin <= tau;
}

inline bool eval_alpha(
    float2 p,
    float2 m,
    float a,
    float b,
    float c,
    float opacity,
    constant MetaF32& mf,
    thread float& alpha,
    thread float& raw_alpha,
    thread float& power,
    thread float2& d) {
  d = p - m;
  power = -0.5f * (a * d.x * d.x + 2.0f * b * d.x * d.y + c * d.y * d.y);
  if (power > 0.0f) return false;
  raw_alpha = opacity * exp(power);
  alpha = min(mf.max_alpha, raw_alpha);
  return alpha >= mf.alpha_threshold;
}

inline void bitonic_sort_ids(threadgroup uint* shared_ids, uint valid_count, uint tid) {
  uint sort_n = next_pow2_u32(valid_count);
  for (uint i = tid; i < sort_n; i += GSP_THREADS) {
    if (i >= valid_count) shared_ids[i] = 0xFFFFFFFFu;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint k = 2u; k <= sort_n; k <<= 1u) {
    for (uint j = k >> 1u; j > 0u; j >>= 1u) {
      uint n_pairs = sort_n >> 1u;
      for (uint pair = tid; pair < n_pairs; pair += GSP_THREADS) {
        uint pos = 2u * j * (pair / j) + (pair % j);
        uint ixj = pos + j;
        bool ascending = ((pos & k) == 0u);
        uint va = shared_ids[pos];
        uint vb = shared_ids[ixj];
        if ((va > vb) == ascending) {
          shared_ids[pos] = vb;
          shared_ids[ixj] = va;
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  }
}

inline void load_chunk_params(
    const device float2* means2d,
    const device float* conics,
    const device float* colors,
    const device float* opacities,
    const threadgroup uint* shared_ids,
    uint chunk_start,
    uint chunk_n,
    uint tid,
    threadgroup float2* sh_means,
    threadgroup float* sh_conics,
    threadgroup float* sh_colors,
    threadgroup float* sh_opacities) {
  for (uint i = tid; i < chunk_n; i += GSP_THREADS) {
    uint g = shared_ids[chunk_start + i];
    sh_means[i] = means2d[g];
    uint b3 = i * 3u;
    uint g3 = g * 3u;
    sh_conics[b3 + 0u] = conics[g3 + 0u];
    sh_conics[b3 + 1u] = conics[g3 + 1u];
    sh_conics[b3 + 2u] = conics[g3 + 2u];
    sh_colors[b3 + 0u] = colors[g3 + 0u];
    sh_colors[b3 + 1u] = colors[g3 + 1u];
    sh_colors[b3 + 2u] = colors[g3 + 2u];
    sh_opacities[i] = opacities[g];
  }
}

inline uint reduce_alive(uint alive, uint simd_lane, uint simd_group, threadgroup uint* tg_alive) {
  uint sg_sum = simd_sum(alive);
  if (simd_lane == 0u) tg_alive[simd_group] = sg_sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint total = 0u;
  if (simd_group == 0u) {
    uint v = (simd_lane < GSP_SIMDGROUPS) ? tg_alive[simd_lane] : 0u;
    total = simd_sum(v);
    if (simd_lane == 0u) tg_alive[0] = total;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return tg_alive[0];
}

inline void tile_pixel_from_tid(uint tg_id, uint tid, constant MetaI32& mi, thread uint& x, thread uint& y) {
  uint tile_x = tg_id % uint(mi.tiles_x);
  uint tile_y = tg_id / uint(mi.tiles_x);
  uint px = tid % GSP_TILE_SIZE;
  uint py = tid / GSP_TILE_SIZE;
  x = tile_x * GSP_TILE_SIZE + px;
  y = tile_y * GSP_TILE_SIZE + py;
}

inline void overflow_tile_pixel_from_tid(uint tile_id, uint tid, constant MetaI32& mi, thread uint& x, thread uint& y) {
  uint tile_x = tile_id % uint(mi.tiles_x);
  uint tile_y = tile_id / uint(mi.tiles_x);
  uint px = tid % GSP_TILE_SIZE;
  uint py = tid / GSP_TILE_SIZE;
  x = tile_x * GSP_TILE_SIZE + px;
  y = tile_y * GSP_TILE_SIZE + py;
}

inline uint tile_img_base(uint local_tile_idx, uint px, uint py) {
  return (((local_tile_idx * GSP_TILE_SIZE + py) * GSP_TILE_SIZE + px) * 3u);
}

kernel void count_tiles(
    const device float2* means2d [[buffer(0)]],
    const device float* conics [[buffer(1)]],
    const device float* opacities [[buffer(2)]],
    constant MetaI32& mi [[buffer(3)]],
    constant MetaF32& mf [[buffer(4)]],
    device int4* bbox_out [[buffer(5)]],
    device float* tau_out [[buffer(6)]],
    device atomic_uint* tile_counts [[buffer(7)]],
    uint gid [[thread_position_in_grid]]) {
  if (gid >= uint(mi.gaussians)) return;
  float tau;
  if (!alpha_support_params(opacities[gid], mf, tau)) {
    bbox_out[gid] = int4(1, 1, 0, 0);
    tau_out[gid] = 0.0f;
    return;
  }

  float2 m = means2d[gid];
  uint g3 = gid * 3u;
  float a = conics[g3 + 0u];
  float b = conics[g3 + 1u];
  float c = conics[g3 + 2u];
  int4 bb = snugbox(m, a, b, c, tau, mi, mf);
  bbox_out[gid] = bb;
  tau_out[gid] = tau;
  if (bb.x > bb.z || bb.y > bb.w) return;

  int tx0 = bb.x / mi.tile_size;
  int tx1 = bb.z / mi.tile_size;
  int ty0 = bb.y / mi.tile_size;
  int ty1 = bb.w / mi.tile_size;

  for (int ty = ty0; ty <= ty1; ++ty) {
    float ry0 = float(ty * mi.tile_size) + 0.5f;
    float ry1 = min(float(mi.height - 1) + 0.5f, float((ty + 1) * mi.tile_size - 1) + 0.5f);
    for (int tx = tx0; tx <= tx1; ++tx) {
      float rx0 = float(tx * mi.tile_size) + 0.5f;
      float rx1 = min(float(mi.width - 1) + 0.5f, float((tx + 1) * mi.tile_size - 1) + 0.5f);
      if (ellipse_intersects_rect(m, a, b, c, tau, rx0, ry0, rx1, ry1)) {
        uint tile = uint(ty * mi.tiles_x + tx);
        atomic_fetch_add_explicit(tile_counts + tile, 1u, memory_order_relaxed);
      }
    }
  }
}

kernel void emit_binned_ids(
    const device float2* means2d [[buffer(0)]],
    const device float* conics [[buffer(1)]],
    const device int4* bbox_in [[buffer(2)]],
    const device float* tau_in [[buffer(3)]],
    constant MetaI32& mi [[buffer(4)]],
    device atomic_uint* tile_cursors [[buffer(5)]],
    device uint* binned_ids [[buffer(6)]],
    uint gid [[thread_position_in_grid]]) {
  if (gid >= uint(mi.gaussians)) return;
  int4 bb = bbox_in[gid];
  if (bb.x > bb.z || bb.y > bb.w) return;
  float tau = tau_in[gid];
  float2 m = means2d[gid];
  uint g3 = gid * 3u;
  float a = conics[g3 + 0u];
  float b = conics[g3 + 1u];
  float c = conics[g3 + 2u];

  int tx0 = bb.x / mi.tile_size;
  int tx1 = bb.z / mi.tile_size;
  int ty0 = bb.y / mi.tile_size;
  int ty1 = bb.w / mi.tile_size;

  for (int ty = ty0; ty <= ty1; ++ty) {
    float ry0 = float(ty * mi.tile_size) + 0.5f;
    float ry1 = min(float(mi.height - 1) + 0.5f, float((ty + 1) * mi.tile_size - 1) + 0.5f);
    for (int tx = tx0; tx <= tx1; ++tx) {
      float rx0 = float(tx * mi.tile_size) + 0.5f;
      float rx1 = min(float(mi.width - 1) + 0.5f, float((tx + 1) * mi.tile_size - 1) + 0.5f);
      if (ellipse_intersects_rect(m, a, b, c, tau, rx0, ry0, rx1, ry1)) {
        uint tile = uint(ty * mi.tiles_x + tx);
        uint idx = atomic_fetch_add_explicit(tile_cursors + tile, 1u, memory_order_relaxed);
        binned_ids[idx] = gid;
      }
    }
  }
}

kernel void tile_fast_forward(
    const device uint* tile_counts [[buffer(0)]],
    const device int* tile_offsets [[buffer(1)]],
    const device uint* binned_ids [[buffer(2)]],
    const device float2* means2d [[buffer(3)]],
    const device float* conics [[buffer(4)]],
    const device float* colors [[buffer(5)]],
    const device float* opacities [[buffer(6)]],
    constant MetaI32& mi [[buffer(7)]],
    constant MetaF32& mf [[buffer(8)]],
    device float* out_rgb [[buffer(9)]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  if (tg_id >= uint(mi.tile_count)) return;

  uint x, y;
  tile_pixel_from_tid(tg_id, tid, mi, x, y);
  bool valid = (x < uint(mi.width) && y < uint(mi.height));
  uint pix = valid ? (y * uint(mi.width) + x) : 0u;

  uint count = tile_counts[tg_id];
  if (count == 0u || count > uint(mi.max_fast_pairs)) {
    if (valid) {
      out_rgb[pix * 3u + 0u] = mf.bg_r;
      out_rgb[pix * 3u + 1u] = mf.bg_g;
      out_rgb[pix * 3u + 2u] = mf.bg_b;
    }
    return;
  }

  threadgroup uint shared_ids[GSP_FAST_CAP];
  threadgroup float2 sh_means[GSP_CHUNK];
  threadgroup float sh_conics[GSP_CHUNK * 3u];
  threadgroup float sh_colors[GSP_CHUNK * 3u];
  threadgroup float sh_opacities[GSP_CHUNK];
  threadgroup uint tg_alive[GSP_SIMDGROUPS];

  uint start = uint(tile_offsets[tg_id]);
  for (uint i = tid; i < count; i += GSP_THREADS) {
    shared_ids[i] = binned_ids[start + i];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bitonic_sort_ids(shared_ids, count, tid);

  float3 accum = float3(0.0f);
  float T = 1.0f;
  float2 p = float2(float(x) + 0.5f, float(y) + 0.5f);

  for (uint chunk_start = 0u; chunk_start < count; chunk_start += GSP_CHUNK) {
    uint chunk_n = min(GSP_CHUNK, count - chunk_start);
    load_chunk_params(means2d, conics, colors, opacities, shared_ids, chunk_start, chunk_n, tid, sh_means, sh_conics, sh_colors, sh_opacities);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint alive_total = reduce_alive((valid && T > mf.transmittance_threshold) ? 1u : 0u, simd_lane, simd_group, tg_alive);
    if (alive_total == 0u) break;

    if (valid && T > mf.transmittance_threshold) {
      for (uint j = 0u; j < chunk_n; ++j) {
        float alpha, raw_alpha, power;
        float2 d;
        float2 m = sh_means[j];
        float3 q = load3_sh(sh_conics, j);
        if (!eval_alpha(p, m, q.x, q.y, q.z, sh_opacities[j], mf, alpha, raw_alpha, power, d)) continue;
        float w = T * alpha;
        accum += w * load3_sh(sh_colors, j);
        T *= (1.0f - alpha);
        if (T <= mf.transmittance_threshold) break;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (valid) {
    out_rgb[pix * 3u + 0u] = accum.x + T * mf.bg_r;
    out_rgb[pix * 3u + 1u] = accum.y + T * mf.bg_g;
    out_rgb[pix * 3u + 2u] = accum.z + T * mf.bg_b;
  }
}

kernel void tile_fast_backward(
    const device float* grad_rgb [[buffer(0)]],
    const device uint* tile_counts [[buffer(1)]],
    const device int* tile_offsets [[buffer(2)]],
    const device uint* binned_ids [[buffer(3)]],
    const device float2* means2d [[buffer(4)]],
    const device float* conics [[buffer(5)]],
    const device float* colors [[buffer(6)]],
    const device float* opacities [[buffer(7)]],
    constant MetaI32& mi [[buffer(8)]],
    constant MetaF32& mf [[buffer(9)]],
    device atomic_float* g_means2d [[buffer(10)]],
    device atomic_float* g_conics [[buffer(11)]],
    device atomic_float* g_colors [[buffer(12)]],
    device atomic_float* g_opacities [[buffer(13)]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  if (tg_id >= uint(mi.tile_count)) return;
  uint count = tile_counts[tg_id];
  if (count == 0u || count > uint(mi.max_fast_pairs)) return;

  threadgroup uint shared_ids[GSP_FAST_CAP];
  threadgroup float2 sh_means[GSP_CHUNK];
  threadgroup float sh_conics[GSP_CHUNK * 3u];
  threadgroup float sh_colors[GSP_CHUNK * 3u];
  threadgroup float sh_opacities[GSP_CHUNK];
  threadgroup uint tg_alive[GSP_SIMDGROUPS];
  threadgroup float4 partial0[GSP_SIMDGROUPS];
  threadgroup float4 partial1[GSP_SIMDGROUPS];
  threadgroup float partial2[GSP_SIMDGROUPS];

  uint start = uint(tile_offsets[tg_id]);
  for (uint i = tid; i < count; i += GSP_THREADS) {
    shared_ids[i] = binned_ids[start + i];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bitonic_sort_ids(shared_ids, count, tid);

  uint x, y;
  tile_pixel_from_tid(tg_id, tid, mi, x, y);
  bool valid = (x < uint(mi.width) && y < uint(mi.height));
  uint pix = valid ? (y * uint(mi.width) + x) : 0u;
  float2 p = float2(float(x) + 0.5f, float(y) + 0.5f);
  float3 go = valid ? float3(grad_rgb[pix * 3u + 0u], grad_rgb[pix * 3u + 1u], grad_rgb[pix * 3u + 2u]) : float3(0.0f);

  // Recompute the forward chain per pixel to recover the reverse-scan start state.
  float T_final = 1.0f;
  uint end_i = count;
  for (uint chunk_start = 0u; chunk_start < count; chunk_start += GSP_CHUNK) {
    uint chunk_n = min(GSP_CHUNK, count - chunk_start);
    load_chunk_params(means2d, conics, colors, opacities, shared_ids, chunk_start, chunk_n, tid, sh_means, sh_conics, sh_colors, sh_opacities);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint alive_total = reduce_alive((valid && T_final > mf.transmittance_threshold) ? 1u : 0u, simd_lane, simd_group, tg_alive);
    if (alive_total == 0u) break;

    if (valid && T_final > mf.transmittance_threshold) {
      for (uint j = 0u; j < chunk_n; ++j) {
        float alpha, raw_alpha, power;
        float2 d;
        float2 m = sh_means[j];
        float3 q = load3_sh(sh_conics, j);
        if (!eval_alpha(p, m, q.x, q.y, q.z, sh_opacities[j], mf, alpha, raw_alpha, power, d)) continue;
        T_final *= (1.0f - alpha);
        if (T_final <= mf.transmittance_threshold) {
          end_i = chunk_start + j + 1u;
          break;
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  float T_cur = T_final;
  float gT = valid ? dot(go, float3(mf.bg_r, mf.bg_g, mf.bg_b)) : 0.0f;

  for (int chunk_end = int(end_i); chunk_end > 0; chunk_end -= int(GSP_CHUNK)) {
    int chunk_start_i = max(0, chunk_end - int(GSP_CHUNK));
    uint chunk_start = uint(chunk_start_i);
    uint chunk_n = uint(chunk_end - chunk_start_i);
    load_chunk_params(means2d, conics, colors, opacities, shared_ids, chunk_start, chunk_n, tid, sh_means, sh_conics, sh_colors, sh_opacities);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int local = int(chunk_n) - 1; local >= 0; --local) {
      uint global_i = chunk_start + uint(local);
      uint g = shared_ids[global_i];
      float2 m = sh_means[uint(local)];
      float3 q = load3_sh(sh_conics, uint(local));
      float3 c = load3_sh(sh_colors, uint(local));
      float opacity = sh_opacities[uint(local)];

      float l_gmx = 0.0f;
      float l_gmy = 0.0f;
      float l_ga = 0.0f;
      float l_gb = 0.0f;
      float l_gc = 0.0f;
      float l_gop = 0.0f;
      float3 l_gcol = float3(0.0f);

      if (valid && global_i < end_i) {
        float alpha, raw_alpha, power;
        float2 d;
        if (eval_alpha(p, m, q.x, q.y, q.z, opacity, mf, alpha, raw_alpha, power, d)) {
          float denom = max(1.0f - alpha, mf.eps);
          float T_prev = T_cur / denom;
          float dot_gc = dot(go, c);
          float g_alpha = T_prev * (dot_gc - gT);

          l_gcol = go * (T_prev * alpha);
          float gate = (raw_alpha < mf.max_alpha) ? 1.0f : 0.0f;
          float g_raw = g_alpha * gate;
          float g_power = g_raw * raw_alpha;

          l_ga = g_power * (-0.5f) * d.x * d.x;
          l_gb = g_power * (-1.0f) * d.x * d.y;
          l_gc = g_power * (-0.5f) * d.y * d.y;
          float g_dx = g_power * (-(q.x * d.x + q.y * d.y));
          float g_dy = g_power * (-(q.y * d.x + q.z * d.y));
          l_gmx = -g_dx;
          l_gmy = -g_dy;
          l_gop = g_raw * (raw_alpha / max(opacity, mf.eps));

          gT = alpha * dot_gc + (1.0f - alpha) * gT;
          T_cur = T_prev;
        }
      }

      float4 s0 = simd_sum(float4(l_gmx, l_gmy, l_ga, l_gb));
      float4 s1 = simd_sum(float4(l_gc, l_gcol.x, l_gcol.y, l_gcol.z));
      float s2 = simd_sum(l_gop);
      if (simd_lane == 0u) {
        partial0[simd_group] = s0;
        partial1[simd_group] = s1;
        partial2[simd_group] = s2;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      if (simd_group == 0u) {
        float4 v0 = (simd_lane < GSP_SIMDGROUPS) ? partial0[simd_lane] : float4(0.0f);
        float4 v1 = (simd_lane < GSP_SIMDGROUPS) ? partial1[simd_lane] : float4(0.0f);
        float v2 = (simd_lane < GSP_SIMDGROUPS) ? partial2[simd_lane] : 0.0f;
        float4 t0 = simd_sum(v0);
        float4 t1 = simd_sum(v1);
        float t2 = simd_sum(v2);
        if (simd_lane == 0u) {
          atomic_fetch_add_explicit(&g_means2d[g * 2u + 0u], t0.x, memory_order_relaxed);
          atomic_fetch_add_explicit(&g_means2d[g * 2u + 1u], t0.y, memory_order_relaxed);
          atomic_fetch_add_explicit(&g_conics[g * 3u + 0u], t0.z, memory_order_relaxed);
          atomic_fetch_add_explicit(&g_conics[g * 3u + 1u], t0.w, memory_order_relaxed);
          atomic_fetch_add_explicit(&g_conics[g * 3u + 2u], t1.x, memory_order_relaxed);
          atomic_add3(g_colors, g, float3(t1.y, t1.z, t1.w));
          atomic_fetch_add_explicit(&g_opacities[g], t2, memory_order_relaxed);
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

kernel void tile_overflow_forward(
    const device uint* overflow_tile_ids [[buffer(0)]],
    const device int* overflow_tile_offsets [[buffer(1)]],
    const device uint* overflow_sorted_ids [[buffer(2)]],
    const device float2* means2d [[buffer(3)]],
    const device float* conics [[buffer(4)]],
    const device float* colors [[buffer(5)]],
    const device float* opacities [[buffer(6)]],
    constant MetaI32& mi [[buffer(7)]],
    constant MetaF32& mf [[buffer(8)]],
    device float* out_tiles [[buffer(9)]],
    uint local_tile_idx [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  uint tile_id = overflow_tile_ids[local_tile_idx];
  uint start = uint(overflow_tile_offsets[local_tile_idx]);
  uint end = uint(overflow_tile_offsets[local_tile_idx + 1]);
  uint count = end - start;

  threadgroup float2 sh_means[GSP_CHUNK];
  threadgroup float sh_conics[GSP_CHUNK * 3u];
  threadgroup float sh_colors[GSP_CHUNK * 3u];
  threadgroup float sh_opacities[GSP_CHUNK];
  threadgroup uint tg_alive[GSP_SIMDGROUPS];

  uint x, y;
  overflow_tile_pixel_from_tid(tile_id, tid, mi, x, y);
  bool valid = (x < uint(mi.width) && y < uint(mi.height));
  uint px = tid % GSP_TILE_SIZE;
  uint py = tid / GSP_TILE_SIZE;
  float2 p = float2(float(x) + 0.5f, float(y) + 0.5f);
  float3 accum = float3(0.0f);
  float T = 1.0f;

  // Need a local id buffer for chunk loader.
  threadgroup uint sh_ids[GSP_CHUNK];

  for (uint chunk_start = 0u; chunk_start < count; chunk_start += GSP_CHUNK) {
    uint chunk_n = min(GSP_CHUNK, count - chunk_start);
    for (uint i = tid; i < chunk_n; i += GSP_THREADS) {
      sh_ids[i] = overflow_sorted_ids[start + i + chunk_start];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    load_chunk_params(means2d, conics, colors, opacities, sh_ids, 0u, chunk_n, tid, sh_means, sh_conics, sh_colors, sh_opacities);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint alive_total = reduce_alive((valid && T > mf.transmittance_threshold) ? 1u : 0u, simd_lane, simd_group, tg_alive);
    if (alive_total == 0u) break;

    if (valid && T > mf.transmittance_threshold) {
      for (uint j = 0u; j < chunk_n; ++j) {
        float alpha, raw_alpha, power;
        float2 d;
        float2 m = sh_means[j];
        float3 q = load3_sh(sh_conics, j);
        if (!eval_alpha(p, m, q.x, q.y, q.z, sh_opacities[j], mf, alpha, raw_alpha, power, d)) continue;
        float w = T * alpha;
        accum += w * load3_sh(sh_colors, j);
        T *= (1.0f - alpha);
        if (T <= mf.transmittance_threshold) break;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  uint base = tile_img_base(local_tile_idx, px, py);
  out_tiles[base + 0u] = accum.x + T * mf.bg_r;
  out_tiles[base + 1u] = accum.y + T * mf.bg_g;
  out_tiles[base + 2u] = accum.z + T * mf.bg_b;
}

kernel void tile_overflow_backward(
    const device float* grad_tiles [[buffer(0)]],
    const device uint* overflow_tile_ids [[buffer(1)]],
    const device int* overflow_tile_offsets [[buffer(2)]],
    const device uint* overflow_sorted_ids [[buffer(3)]],
    const device float2* means2d [[buffer(4)]],
    const device float* conics [[buffer(5)]],
    const device float* colors [[buffer(6)]],
    const device float* opacities [[buffer(7)]],
    constant MetaI32& mi [[buffer(8)]],
    constant MetaF32& mf [[buffer(9)]],
    device atomic_float* g_means2d [[buffer(10)]],
    device atomic_float* g_conics [[buffer(11)]],
    device atomic_float* g_colors [[buffer(12)]],
    device atomic_float* g_opacities [[buffer(13)]],
    uint local_tile_idx [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  uint tile_id = overflow_tile_ids[local_tile_idx];
  uint start = uint(overflow_tile_offsets[local_tile_idx]);
  uint end = uint(overflow_tile_offsets[local_tile_idx + 1]);
  uint count = end - start;
  if (count == 0u) return;

  threadgroup float2 sh_means[GSP_CHUNK];
  threadgroup float sh_conics[GSP_CHUNK * 3u];
  threadgroup float sh_colors[GSP_CHUNK * 3u];
  threadgroup float sh_opacities[GSP_CHUNK];
  threadgroup uint tg_alive[GSP_SIMDGROUPS];
  threadgroup float4 partial0[GSP_SIMDGROUPS];
  threadgroup float4 partial1[GSP_SIMDGROUPS];
  threadgroup float partial2[GSP_SIMDGROUPS];
  threadgroup uint sh_ids[GSP_CHUNK];

  uint x, y;
  overflow_tile_pixel_from_tid(tile_id, tid, mi, x, y);
  bool valid = (x < uint(mi.width) && y < uint(mi.height));
  uint px = tid % GSP_TILE_SIZE;
  uint py = tid / GSP_TILE_SIZE;
  float2 p = float2(float(x) + 0.5f, float(y) + 0.5f);
  uint base = tile_img_base(local_tile_idx, px, py);
  float3 go = float3(grad_tiles[base + 0u], grad_tiles[base + 1u], grad_tiles[base + 2u]);

  float T_final = 1.0f;
  uint end_i = count;
  for (uint chunk_start = 0u; chunk_start < count; chunk_start += GSP_CHUNK) {
    uint chunk_n = min(GSP_CHUNK, count - chunk_start);
    for (uint i = tid; i < chunk_n; i += GSP_THREADS) {
      sh_ids[i] = overflow_sorted_ids[start + chunk_start + i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    load_chunk_params(means2d, conics, colors, opacities, sh_ids, 0u, chunk_n, tid, sh_means, sh_conics, sh_colors, sh_opacities);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint alive_total = reduce_alive((valid && T_final > mf.transmittance_threshold) ? 1u : 0u, simd_lane, simd_group, tg_alive);
    if (alive_total == 0u) break;

    if (valid && T_final > mf.transmittance_threshold) {
      for (uint j = 0u; j < chunk_n; ++j) {
        float alpha, raw_alpha, power;
        float2 d;
        float2 m = sh_means[j];
        float3 q = load3_sh(sh_conics, j);
        if (!eval_alpha(p, m, q.x, q.y, q.z, sh_opacities[j], mf, alpha, raw_alpha, power, d)) continue;
        T_final *= (1.0f - alpha);
        if (T_final <= mf.transmittance_threshold) {
          end_i = chunk_start + j + 1u;
          break;
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  float T_cur = T_final;
  float gT = valid ? dot(go, float3(mf.bg_r, mf.bg_g, mf.bg_b)) : 0.0f;

  for (int chunk_end = int(end_i); chunk_end > 0; chunk_end -= int(GSP_CHUNK)) {
    int chunk_start_i = max(0, chunk_end - int(GSP_CHUNK));
    uint chunk_start = uint(chunk_start_i);
    uint chunk_n = uint(chunk_end - chunk_start_i);

    for (uint i = tid; i < chunk_n; i += GSP_THREADS) {
      sh_ids[i] = overflow_sorted_ids[start + chunk_start + i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    load_chunk_params(means2d, conics, colors, opacities, sh_ids, 0u, chunk_n, tid, sh_means, sh_conics, sh_colors, sh_opacities);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int local = int(chunk_n) - 1; local >= 0; --local) {
      uint global_i = chunk_start + uint(local);
      uint g = sh_ids[uint(local)];
      float2 m = sh_means[uint(local)];
      float3 q = load3_sh(sh_conics, uint(local));
      float3 c = load3_sh(sh_colors, uint(local));
      float opacity = sh_opacities[uint(local)];

      float l_gmx = 0.0f;
      float l_gmy = 0.0f;
      float l_ga = 0.0f;
      float l_gb = 0.0f;
      float l_gc = 0.0f;
      float l_gop = 0.0f;
      float3 l_gcol = float3(0.0f);

      if (valid && global_i < end_i) {
        float alpha, raw_alpha, power;
        float2 d;
        if (eval_alpha(p, m, q.x, q.y, q.z, opacity, mf, alpha, raw_alpha, power, d)) {
          float denom = max(1.0f - alpha, mf.eps);
          float T_prev = T_cur / denom;
          float dot_gc = dot(go, c);
          float g_alpha = T_prev * (dot_gc - gT);

          l_gcol = go * (T_prev * alpha);
          float gate = (raw_alpha < mf.max_alpha) ? 1.0f : 0.0f;
          float g_raw = g_alpha * gate;
          float g_power = g_raw * raw_alpha;

          l_ga = g_power * (-0.5f) * d.x * d.x;
          l_gb = g_power * (-1.0f) * d.x * d.y;
          l_gc = g_power * (-0.5f) * d.y * d.y;
          float g_dx = g_power * (-(q.x * d.x + q.y * d.y));
          float g_dy = g_power * (-(q.y * d.x + q.z * d.y));
          l_gmx = -g_dx;
          l_gmy = -g_dy;
          l_gop = g_raw * (raw_alpha / max(opacity, mf.eps));

          gT = alpha * dot_gc + (1.0f - alpha) * gT;
          T_cur = T_prev;
        }
      }

      float4 s0 = simd_sum(float4(l_gmx, l_gmy, l_ga, l_gb));
      float4 s1 = simd_sum(float4(l_gc, l_gcol.x, l_gcol.y, l_gcol.z));
      float s2 = simd_sum(l_gop);
      if (simd_lane == 0u) {
        partial0[simd_group] = s0;
        partial1[simd_group] = s1;
        partial2[simd_group] = s2;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      if (simd_group == 0u) {
        float4 v0 = (simd_lane < GSP_SIMDGROUPS) ? partial0[simd_lane] : float4(0.0f);
        float4 v1 = (simd_lane < GSP_SIMDGROUPS) ? partial1[simd_lane] : float4(0.0f);
        float v2 = (simd_lane < GSP_SIMDGROUPS) ? partial2[simd_lane] : 0.0f;
        float4 t0 = simd_sum(v0);
        float4 t1 = simd_sum(v1);
        float t2 = simd_sum(v2);
        if (simd_lane == 0u) {
          atomic_fetch_add_explicit(&g_means2d[g * 2u + 0u], t0.x, memory_order_relaxed);
          atomic_fetch_add_explicit(&g_means2d[g * 2u + 1u], t0.y, memory_order_relaxed);
          atomic_fetch_add_explicit(&g_conics[g * 3u + 0u], t0.z, memory_order_relaxed);
          atomic_fetch_add_explicit(&g_conics[g * 3u + 1u], t0.w, memory_order_relaxed);
          atomic_fetch_add_explicit(&g_conics[g * 3u + 2u], t1.x, memory_order_relaxed);
          atomic_add3(g_colors, g, float3(t1.y, t1.z, t1.w));
          atomic_fetch_add_explicit(&g_opacities[g], t2, memory_order_relaxed);
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

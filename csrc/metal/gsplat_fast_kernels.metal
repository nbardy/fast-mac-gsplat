#include <metal_stdlib>
using namespace metal;

#define GSP_TILE_SIZE 16u
#define GSP_TILE_PIXELS 256u
#define GSP_THREADS_PER_TG 1024u
#define GSP_MAX_TILE_PAIRS 4096u

struct MetaI32 {
  int height;
  int width;
  int tiles_y;
  int tiles_x;
  int tile_size;
  int gaussians;
  int tile_count;
  int max_tile_pairs;
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

inline float safe_det(float3 q, float eps) {
  return max(q.x * q.z - q.y * q.y, eps);
}

inline float3 load3(const device float* values, uint index) {
  return float3(values[index * 3u + 0u], values[index * 3u + 1u], values[index * 3u + 2u]);
}

inline bool alpha_support_params(float opacity, constant MetaF32& mf, thread float& tau) {
  if (opacity <= mf.alpha_threshold) return false;
  float ratio = max(mf.alpha_threshold / max(opacity, mf.eps), mf.eps);
  tau = -2.0f * log(ratio);
  return isfinite(tau) && (tau > 0.0f);
}

inline int4 snugbox(float2 m, float3 q, float tau, constant MetaI32& mi, constant MetaF32& mf) {
  float det = safe_det(q, mf.eps);
  float half_x = sqrt(max(tau * q.z / det, 0.0f));
  float half_y = sqrt(max(tau * q.x / det, 0.0f));

  int x0 = max(0, int(floor(m.x - half_x - 0.5f)));
  int x1 = min(mi.width - 1, int(ceil(m.x + half_x - 0.5f)));
  int y0 = max(0, int(floor(m.y - half_y - 0.5f)));
  int y1 = min(mi.height - 1, int(ceil(m.y + half_y - 0.5f)));
  return int4(x0, y0, x1, y1);
}

inline bool ellipse_intersects_rect(float2 m, float3 q, float tau, float rx0, float ry0, float rx1, float ry1) {
  float dx0 = rx0 - m.x;
  float dx1 = rx1 - m.x;
  float dy0 = ry0 - m.y;
  float dy1 = ry1 - m.y;

  if (m.x >= rx0 && m.x <= rx1 && m.y >= ry0 && m.y <= ry1) return true;

  float qmin = INFINITY;
  // corners
  qmin = min(qmin, q.x * dx0 * dx0 + 2.0f * q.y * dx0 * dy0 + q.z * dy0 * dy0);
  qmin = min(qmin, q.x * dx0 * dx0 + 2.0f * q.y * dx0 * dy1 + q.z * dy1 * dy1);
  qmin = min(qmin, q.x * dx1 * dx1 + 2.0f * q.y * dx1 * dy0 + q.z * dy0 * dy0);
  qmin = min(qmin, q.x * dx1 * dx1 + 2.0f * q.y * dx1 * dy1 + q.z * dy1 * dy1);
  // vertical edges
  if (q.z > 1e-8f) {
    float dy = clamp(-(q.y / q.z) * dx0, dy0, dy1);
    qmin = min(qmin, q.x * dx0 * dx0 + 2.0f * q.y * dx0 * dy + q.z * dy * dy);
    dy = clamp(-(q.y / q.z) * dx1, dy0, dy1);
    qmin = min(qmin, q.x * dx1 * dx1 + 2.0f * q.y * dx1 * dy + q.z * dy * dy);
  }
  // horizontal edges
  if (q.x > 1e-8f) {
    float dx = clamp(-(q.y / q.x) * dy0, dx0, dx1);
    qmin = min(qmin, q.x * dx * dx + 2.0f * q.y * dx * dy0 + q.z * dy0 * dy0);
    dx = clamp(-(q.y / q.x) * dy1, dx0, dx1);
    qmin = min(qmin, q.x * dx * dx + 2.0f * q.y * dx * dy1 + q.z * dy1 * dy1);
  }
  return qmin <= tau;
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
  float3 q = load3(conics, gid);
  int4 bb = snugbox(m, q, tau, mi, mf);
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
      if (ellipse_intersects_rect(m, q, tau, rx0, ry0, rx1, ry1)) {
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
  float3 q = load3(conics, gid);

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
      if (ellipse_intersects_rect(m, q, tau, rx0, ry0, rx1, ry1)) {
        uint tile = uint(ty * mi.tiles_x + tx);
        uint idx = atomic_fetch_add_explicit(tile_cursors + tile, 1u, memory_order_relaxed);
        binned_ids[idx] = gid;
      }
    }
  }
}

inline void bitonic_sort_ids(threadgroup uint* shared_ids, uint valid_count, uint tid) {
  uint sort_n = next_pow2_u32(valid_count);
  for (uint i = tid; i < sort_n; i += GSP_THREADS_PER_TG) {
    if (i >= valid_count) shared_ids[i] = 0xFFFFFFFFu;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint k = 2u; k <= sort_n; k <<= 1u) {
    for (uint j = k >> 1u; j > 0u; j >>= 1u) {
      uint n_pairs = sort_n >> 1u;
      for (uint pair = tid; pair < n_pairs; pair += GSP_THREADS_PER_TG) {
        uint pos = 2u * j * (pair / j) + (pair % j);
        uint ixj = pos + j;
        bool ascending = ((pos & k) == 0u);
        uint a = shared_ids[pos];
        uint b = shared_ids[ixj];
        if ((a > b) == ascending) {
          shared_ids[pos] = b;
          shared_ids[ixj] = a;
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  }
}

inline bool eval_alpha_for_pixel(
    float2 p,
    float2 m,
    float3 q,
    float opacity,
    constant MetaF32& mf,
    thread float& alpha,
    thread float& power,
    thread float2& d) {
  d = p - m;
  power = -0.5f * (q.x * d.x * d.x + 2.0f * q.y * d.x * d.y + q.z * d.y * d.y);
  if (power > 0.0f) return false;
  float raw_alpha = opacity * exp(power);
  alpha = min(mf.max_alpha, raw_alpha);
  return alpha >= mf.alpha_threshold;
}

kernel void tile_sort_render_forward(
    const device uint* tile_counts [[buffer(0)]],
    const device int* tile_offsets [[buffer(1)]],
    device uint* binned_ids [[buffer(2)]],
    const device float2* means2d [[buffer(3)]],
    const device float* conics [[buffer(4)]],
    const device float* colors [[buffer(5)]],
    const device float* opacities [[buffer(6)]],
    constant MetaI32& mi [[buffer(7)]],
    constant MetaF32& mf [[buffer(8)]],
    device float* out_rgb [[buffer(9)]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]]) {
  if (tg_id >= uint(mi.tile_count)) return;
  uint count = tile_counts[tg_id];
  if (count == 0u) {
    if (tid < GSP_TILE_PIXELS) {
      uint tile_x = tg_id % uint(mi.tiles_x);
      uint tile_y = tg_id / uint(mi.tiles_x);
      uint px = tid % GSP_TILE_SIZE;
      uint py = tid / GSP_TILE_SIZE;
      uint x = tile_x * GSP_TILE_SIZE + px;
      uint y = tile_y * GSP_TILE_SIZE + py;
      if (x < uint(mi.width) && y < uint(mi.height)) {
        uint pix = y * uint(mi.width) + x;
        out_rgb[pix * 3u + 0u] = mf.bg_r;
        out_rgb[pix * 3u + 1u] = mf.bg_g;
        out_rgb[pix * 3u + 2u] = mf.bg_b;
      }
    }
    return;
  }

  threadgroup uint shared_ids[GSP_MAX_TILE_PAIRS];
  uint start = uint(tile_offsets[tg_id]);
  for (uint i = tid; i < count; i += GSP_THREADS_PER_TG) {
    shared_ids[i] = binned_ids[start + i];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bitonic_sort_ids(shared_ids, count, tid);
  for (uint i = tid; i < count; i += GSP_THREADS_PER_TG) {
    binned_ids[start + i] = shared_ids[i];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (tid < GSP_TILE_PIXELS) {
    uint tile_x = tg_id % uint(mi.tiles_x);
    uint tile_y = tg_id / uint(mi.tiles_x);
    uint px = tid % GSP_TILE_SIZE;
    uint py = tid / GSP_TILE_SIZE;
    uint x = tile_x * GSP_TILE_SIZE + px;
    uint y = tile_y * GSP_TILE_SIZE + py;
    if (x >= uint(mi.width) || y >= uint(mi.height)) return;

    float2 p = float2(float(x) + 0.5f, float(y) + 0.5f);
    float3 accum = float3(0.0f, 0.0f, 0.0f);
    float T = 1.0f;
    for (uint i = 0u; i < count; ++i) {
      uint g = shared_ids[i];
      float alpha, power;
      float2 d;
      if (!eval_alpha_for_pixel(p, means2d[g], load3(conics, g), opacities[g], mf, alpha, power, d)) continue;
      float w = T * alpha;
      accum += w * load3(colors, g);
      T *= (1.0f - alpha);
      if (T < mf.transmittance_threshold) break;
    }

    uint pix = y * uint(mi.width) + x;
    out_rgb[pix * 3u + 0u] = accum.x + T * mf.bg_r;
    out_rgb[pix * 3u + 1u] = accum.y + T * mf.bg_g;
    out_rgb[pix * 3u + 2u] = accum.z + T * mf.bg_b;
  }
}

kernel void tile_sort_render_backward(
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
    uint tid [[thread_position_in_threadgroup]]) {
  if (tg_id >= uint(mi.tile_count)) return;
  uint count = tile_counts[tg_id];
  if (count == 0u) return;

  threadgroup uint shared_ids[GSP_MAX_TILE_PAIRS];
  uint start = uint(tile_offsets[tg_id]);
  for (uint i = tid; i < count; i += GSP_THREADS_PER_TG) {
    shared_ids[i] = binned_ids[start + i];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // Forward writes tile-local IDs back in sorted order, so backward can skip
  // the second bitonic sort and use the saved ordering directly.

  if (tid < GSP_TILE_PIXELS) {
    uint tile_x = tg_id % uint(mi.tiles_x);
    uint tile_y = tg_id / uint(mi.tiles_x);
    uint px = tid % GSP_TILE_SIZE;
    uint py = tid / GSP_TILE_SIZE;
    uint x = tile_x * GSP_TILE_SIZE + px;
    uint y = tile_y * GSP_TILE_SIZE + py;
    if (x >= uint(mi.width) || y >= uint(mi.height)) return;

    float2 p = float2(float(x) + 0.5f, float(y) + 0.5f);
    uint pix = y * uint(mi.width) + x;
    float3 go = float3(grad_rgb[pix * 3u + 0u], grad_rgb[pix * 3u + 1u], grad_rgb[pix * 3u + 2u]);

    float T_final = 1.0f;
    uint end_i = count;
    for (uint i = 0u; i < count; ++i) {
      uint g = shared_ids[i];
      float alpha, power;
      float2 d;
      if (!eval_alpha_for_pixel(p, means2d[g], load3(conics, g), opacities[g], mf, alpha, power, d)) continue;
      T_final *= (1.0f - alpha);
      if (T_final < mf.transmittance_threshold) {
        end_i = i + 1u;
        break;
      }
    }

    float T_cur = T_final;
    float gT = go.x * mf.bg_r + go.y * mf.bg_g + go.z * mf.bg_b;

    for (int ii = int(end_i) - 1; ii >= 0; --ii) {
      uint g = shared_ids[uint(ii)];
      float alpha, power;
      float2 d;
      float3 q = load3(conics, g);
      if (!eval_alpha_for_pixel(p, means2d[g], q, opacities[g], mf, alpha, power, d)) continue;

      float denom = max(1.0f - alpha, mf.eps);
      float T_prev = T_cur / denom;
      float3 c = load3(colors, g);
      float dot_gc = go.x * c.x + go.y * c.y + go.z * c.z;
      float g_alpha = T_prev * (dot_gc - gT);

      atomic_fetch_add_explicit(&g_colors[g * 3u + 0u], go.x * T_prev * alpha, memory_order_relaxed);
      atomic_fetch_add_explicit(&g_colors[g * 3u + 1u], go.y * T_prev * alpha, memory_order_relaxed);
      atomic_fetch_add_explicit(&g_colors[g * 3u + 2u], go.z * T_prev * alpha, memory_order_relaxed);

      float raw_alpha = opacities[g] * exp(power);
      float gate = (raw_alpha < mf.max_alpha) ? 1.0f : 0.0f;
      float dalpha_draw = gate;
      float g_raw_alpha = g_alpha * dalpha_draw;

      float g_power = g_raw_alpha * raw_alpha;
      float g_a = g_power * (-0.5f) * d.x * d.x;
      float g_b = g_power * (-1.0f) * d.x * d.y;
      float g_c = g_power * (-0.5f) * d.y * d.y;
      float g_dx = g_power * (-(q.x * d.x + q.y * d.y));
      float g_dy = g_power * (-(q.y * d.x + q.z * d.y));

      atomic_fetch_add_explicit(&g_conics[g * 3u + 0u], g_a, memory_order_relaxed);
      atomic_fetch_add_explicit(&g_conics[g * 3u + 1u], g_b, memory_order_relaxed);
      atomic_fetch_add_explicit(&g_conics[g * 3u + 2u], g_c, memory_order_relaxed);
      atomic_fetch_add_explicit(&g_means2d[g * 2u + 0u], -g_dx, memory_order_relaxed);
      atomic_fetch_add_explicit(&g_means2d[g * 2u + 1u], -g_dy, memory_order_relaxed);
      atomic_fetch_add_explicit(&g_opacities[g], g_raw_alpha * (raw_alpha / max(opacities[g], mf.eps)), memory_order_relaxed);

      gT = alpha * dot_gc + (1.0f - alpha) * gT;
      T_cur = T_prev;
    }
  }
}

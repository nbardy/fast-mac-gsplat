#include <metal_stdlib>
using namespace metal;

#ifndef GSP_TILE_SIZE
#define GSP_TILE_SIZE 16u
#endif
#ifndef GSP_THREADS
#define GSP_THREADS 256u
#endif
#ifndef GSP_FAST_CAP
#define GSP_FAST_CAP 2048u
#endif
#ifndef GSP_TAPE_CAP
#define GSP_TAPE_CAP 8u
#endif
#ifndef GSP_CHUNK
#define GSP_CHUNK 64u
#endif
#ifndef GSP_SIMD_WIDTH
#define GSP_SIMD_WIDTH 32u
#endif
#ifndef GSP_SIMDGROUPS
#define GSP_SIMDGROUPS 8u
#endif

struct MetaI32 {
  int height;
  int width;
  int tiles_y;
  int tiles_x;
  int tile_size;
  int gaussians;
  int tile_count;
  int max_fast_pairs;
  int batch_size;
  int gaussians_per_batch;
  int tiles_per_image;
  int reserved;
  int softmax_tape_k;
};

struct MetaF32 {
  float alpha_threshold;
  float transmittance_threshold;
  float bg_r;
  float bg_g;
  float bg_b;
  float eps;
  float max_alpha;
  float softmax_beta;
  float softmax_gamma;
};

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

inline bool softmax_gs_enabled(constant MetaI32& mi) {
  return mi.reserved != 0;
}

inline void apply_softmax_gs_forward(
    constant MetaF32& mf,
    float depth_cur,
    float power_cur,
    thread float& alpha_cur,
    thread float& T,
    thread float3& accum,
    thread float& past_depth,
    thread float& past_power) {
  if (T >= 1.0f - mf.eps) return;
  float T_orig = T * (1.0f - alpha_cur);
  float a_past = 1.0f - T;
  float w_cur = 1.0f / (1.0f + exp(mf.softmax_beta * (past_power - power_cur)));
  float soft_cur = w_cur * alpha_cur;
  float soft_past = (1.0f - w_cur) * a_past;
  float soft_denom = max(soft_past + soft_cur, mf.eps);
  float tilde_past = soft_past * (1.0f - T_orig) / soft_denom;
  float tilde_cur = soft_cur * (1.0f - T_orig) / max(soft_cur + soft_past * T_orig, mf.eps);
  float decay = exp(-max(mf.softmax_gamma, 0.0f) * abs(depth_cur - past_depth));
  float effective_past = decay * tilde_past + (1.0f - decay) * a_past;
  alpha_cur = decay * tilde_cur + (1.0f - decay) * alpha_cur;
  float pair_product = effective_past * alpha_cur;
  if (pair_product > mf.eps) {
    float pair_sum = effective_past + alpha_cur;
    float disc = max(pair_sum * pair_sum - 4.0f * (1.0f - T_orig) * pair_product, 0.0f);
    float scale = (2.0f * (1.0f - T_orig)) / max(pair_sum + sqrt(disc), mf.eps);
    effective_past *= scale;
    alpha_cur *= scale;
  }
  T = 1.0f - effective_past;
  accum *= effective_past / max(a_past, mf.eps);
}

inline float apply_softmax_gs_tape_scalar(
    constant MetaF32& mf,
    float depth_cur,
    float power_cur,
    thread float& alpha_cur,
    thread float& T,
    thread float& past_depth,
    thread float& past_power) {
  if (T >= 1.0f - mf.eps) return 1.0f;
  float T_orig = T * (1.0f - alpha_cur);
  float a_past = 1.0f - T;
  float w_cur = 1.0f / (1.0f + exp(mf.softmax_beta * (past_power - power_cur)));
  float soft_cur = w_cur * alpha_cur;
  float soft_past = (1.0f - w_cur) * a_past;
  float soft_denom = max(soft_past + soft_cur, mf.eps);
  float target_absorbance = 1.0f - T_orig;
  float tilde_past = soft_past * target_absorbance / soft_denom;
  float tilde_cur = soft_cur * target_absorbance / max(soft_cur + soft_past * T_orig, mf.eps);
  float decay = exp(-max(mf.softmax_gamma, 0.0f) * abs(depth_cur - past_depth));
  float effective_past = decay * tilde_past + (1.0f - decay) * a_past;
  alpha_cur = decay * tilde_cur + (1.0f - decay) * alpha_cur;
  float pair_product = effective_past * alpha_cur;
  if (pair_product > mf.eps) {
    float pair_sum = effective_past + alpha_cur;
    float disc = max(pair_sum * pair_sum - 4.0f * target_absorbance * pair_product, 0.0f);
    float scale = (2.0f * target_absorbance) / max(pair_sum + sqrt(disc), mf.eps);
    effective_past *= scale;
    alpha_cur *= scale;
  }
  T = 1.0f - effective_past;
  return effective_past / max(a_past, mf.eps);
}

inline void init_bounded_tape(thread uint* selected_ids, thread float* selected_weights) {
  for (uint i = 0u; i < GSP_TAPE_CAP; ++i) {
    selected_ids[i] = 0xFFFFFFFFu;
    selected_weights[i] = 0.0f;
  }
}

inline void scale_bounded_tape(thread uint* selected_ids, thread float* selected_weights, uint tape_k, float scale) {
  for (uint i = 0u; i < GSP_TAPE_CAP; ++i) {
    if (i < tape_k && selected_ids[i] != 0xFFFFFFFFu) {
      selected_weights[i] *= scale;
    }
  }
}

inline void insert_bounded_tape_weight(
    thread uint* selected_ids,
    thread float* selected_weights,
    uint tape_k,
    uint gaussian_id,
    float weight) {
  if (weight <= 0.0f || tape_k == 0u) return;
  uint slot = 0xFFFFFFFFu;
  uint min_slot = 0u;
  float min_weight = INFINITY;
  for (uint i = 0u; i < GSP_TAPE_CAP; ++i) {
    if (i >= tape_k) break;
    if (selected_ids[i] == 0xFFFFFFFFu) {
      slot = i;
      break;
    }
    if (selected_weights[i] < min_weight) {
      min_weight = selected_weights[i];
      min_slot = i;
    }
  }
  if (slot == 0xFFFFFFFFu) {
    if (weight <= min_weight) return;
    slot = min_slot;
  }
  selected_ids[slot] = gaussian_id;
  selected_weights[slot] = weight;
}

inline void sort_bounded_tape_by_id(thread uint* selected_ids, thread float* selected_weights, uint tape_k) {
  for (uint i = 0u; i < GSP_TAPE_CAP; ++i) {
    if (i >= tape_k) break;
    for (uint j = i + 1u; j < GSP_TAPE_CAP; ++j) {
      if (j >= tape_k) break;
      if (selected_ids[i] > selected_ids[j]) {
        uint id_tmp = selected_ids[i];
        float w_tmp = selected_weights[i];
        selected_ids[i] = selected_ids[j];
        selected_weights[i] = selected_weights[j];
        selected_ids[j] = id_tmp;
        selected_weights[j] = w_tmp;
      }
    }
  }
}

inline float selected_weight_sum(thread uint* selected_ids, thread float* selected_weights, uint tape_k) {
  float total = 0.0f;
  for (uint i = 0u; i < GSP_TAPE_CAP; ++i) {
    if (i >= tape_k) break;
    if (selected_ids[i] != 0xFFFFFFFFu) total += selected_weights[i];
  }
  return total;
}

struct SoftmaxReplayState {
  bool active;
  bool has_softmax;
  float2 delta;
  float raw_alpha;
  float alpha_input;
  float power;
  float depth;
  float T0;
  float3 accum0;
  float past_depth0;
  float past_power0;
  float T_orig;
  float past_absorbance;
  float w_cur;
  float soft_cur;
  float soft_past;
  float soft_denom;
  float target_absorbance;
  float tilde_past;
  float tilde_cur;
  float tilde_cur_denom;
  float depth_diff;
  float decay;
  float effective_past0;
  float alpha_soft0;
  float pair_product;
  float pair_sum;
  float sqrt_disc;
  float scale;
  float effective_past;
  float color_scale;
  float T_pre;
  float3 accum_pre;
  float alpha_eff;
  float contribution_weight;
  float denom;
};

inline SoftmaxReplayState make_empty_replay_state() {
  SoftmaxReplayState st;
  st.active = false;
  st.has_softmax = false;
  st.delta = float2(0.0f);
  st.raw_alpha = 0.0f;
  st.alpha_input = 0.0f;
  st.power = 0.0f;
  st.depth = 0.0f;
  st.T0 = 1.0f;
  st.accum0 = float3(0.0f);
  st.past_depth0 = 0.0f;
  st.past_power0 = 0.0f;
  st.T_orig = 1.0f;
  st.past_absorbance = 0.0f;
  st.w_cur = 1.0f;
  st.soft_cur = 0.0f;
  st.soft_past = 0.0f;
  st.soft_denom = 1.0f;
  st.target_absorbance = 0.0f;
  st.tilde_past = 0.0f;
  st.tilde_cur = 0.0f;
  st.tilde_cur_denom = 1.0f;
  st.depth_diff = 0.0f;
  st.decay = 0.0f;
  st.effective_past0 = 0.0f;
  st.alpha_soft0 = 0.0f;
  st.pair_product = 0.0f;
  st.pair_sum = 0.0f;
  st.sqrt_disc = 0.0f;
  st.scale = 1.0f;
  st.effective_past = 0.0f;
  st.color_scale = 1.0f;
  st.T_pre = 1.0f;
  st.accum_pre = float3(0.0f);
  st.alpha_eff = 0.0f;
  st.contribution_weight = 0.0f;
  st.denom = 1.0f;
  return st;
}

inline SoftmaxReplayState replay_softmax_to_index(
    const device float2* means2d,
    const device float* conics,
    const device float* colors,
    const device float* opacities,
    const device float* depths,
    const threadgroup uint* shared_ids,
    uint target_i,
    float2 pixel,
    constant MetaF32& mf) {
  SoftmaxReplayState target = make_empty_replay_state();
  float3 accum = float3(0.0f);
  float T = 1.0f;
  float past_depth = 0.0f;
  float past_power = 0.0f;
  for (uint i = 0u; i <= target_i; ++i) {
    uint g = shared_ids[i];
    float2 m = means2d[g];
    uint g3 = g * 3u;
    float qa = conics[g3 + 0u];
    float qb = conics[g3 + 1u];
    float qc = conics[g3 + 2u];
    float alpha, raw_alpha, power;
    float2 dxy;
    bool active = eval_alpha(pixel, m, qa, qb, qc, opacities[g], mf, alpha, raw_alpha, power, dxy);
    if (!active || T <= mf.transmittance_threshold) {
      if (i == target_i) {
        target.active = false;
        target.delta = dxy;
        target.raw_alpha = raw_alpha;
        target.alpha_input = alpha;
        target.power = power;
        target.depth = depths[g];
        target.T0 = T;
        target.accum0 = accum;
        target.past_depth0 = past_depth;
        target.past_power0 = past_power;
      }
      continue;
    }

    bool is_target = (i == target_i);
    if (is_target) {
      target.active = true;
      target.delta = dxy;
      target.raw_alpha = raw_alpha;
      target.alpha_input = alpha;
      target.power = power;
      target.depth = depths[g];
      target.T0 = T;
      target.accum0 = accum;
      target.past_depth0 = past_depth;
      target.past_power0 = past_power;
    }

    if (T < 1.0f - mf.eps) {
      float T_orig = T * (1.0f - alpha);
      float a_past = 1.0f - T;
      float w_cur = 1.0f / (1.0f + exp(mf.softmax_beta * (past_power - power)));
      float soft_cur = w_cur * alpha;
      float soft_past = (1.0f - w_cur) * a_past;
      float soft_denom = max(soft_past + soft_cur, mf.eps);
      float target_absorbance = 1.0f - T_orig;
      float tilde_past = soft_past * target_absorbance / soft_denom;
      float tilde_cur_denom = max(soft_cur + soft_past * T_orig, mf.eps);
      float tilde_cur = soft_cur * target_absorbance / tilde_cur_denom;
      float depth_diff = depths[g] - past_depth;
      float decay = exp(-max(mf.softmax_gamma, 0.0f) * abs(depth_diff));
      float effective_past0 = decay * tilde_past + (1.0f - decay) * a_past;
      float alpha_soft0 = decay * tilde_cur + (1.0f - decay) * alpha;
      float pair_product = effective_past0 * alpha_soft0;
      float pair_sum = effective_past0 + alpha_soft0;
      float sqrt_disc = 0.0f;
      float scale = 1.0f;
      if (pair_product > mf.eps) {
        float disc = max(pair_sum * pair_sum - 4.0f * target_absorbance * pair_product, mf.eps);
        sqrt_disc = sqrt(disc);
        scale = (2.0f * target_absorbance) / max(pair_sum + sqrt_disc, mf.eps);
      }
      float effective_past = scale * effective_past0;
      alpha = scale * alpha_soft0;
      T = 1.0f - effective_past;
      float color_scale = effective_past / max(a_past, mf.eps);
      accum *= color_scale;
      if (is_target) {
        target.has_softmax = true;
        target.T_orig = T_orig;
        target.past_absorbance = a_past;
        target.w_cur = w_cur;
        target.soft_cur = soft_cur;
        target.soft_past = soft_past;
        target.soft_denom = soft_denom;
        target.target_absorbance = target_absorbance;
        target.tilde_past = tilde_past;
        target.tilde_cur = tilde_cur;
        target.tilde_cur_denom = tilde_cur_denom;
        target.depth_diff = depth_diff;
        target.decay = decay;
        target.effective_past0 = effective_past0;
        target.alpha_soft0 = alpha_soft0;
        target.pair_product = pair_product;
        target.pair_sum = pair_sum;
        target.sqrt_disc = sqrt_disc;
        target.scale = scale;
        target.effective_past = effective_past;
        target.color_scale = color_scale;
      }
    }

    if (is_target) {
      target.T_pre = T;
      target.accum_pre = accum;
      target.alpha_eff = alpha;
    }

    float w = T * alpha;
    accum += w * float3(colors[g3 + 0u], colors[g3 + 1u], colors[g3 + 2u]);
    float denom = max(1.0f - T + w, mf.eps);
    past_depth = (past_depth * (1.0f - T) + depths[g] * w) / denom;
    past_power = (past_power * (1.0f - T) + power * w) / denom;
    T *= (1.0f - alpha);
    if (is_target) {
      target.contribution_weight = w;
      target.denom = denom;
    }
  }
  return target;
}

inline SoftmaxReplayState replay_softmax_to_device_index(
    const device float2* means2d,
    const device float* conics,
    const device float* colors,
    const device float* opacities,
    const device float* depths,
    const device uint* sorted_ids,
    uint sorted_start,
    uint target_i,
    float2 pixel,
    constant MetaF32& mf) {
  SoftmaxReplayState target = make_empty_replay_state();
  float3 accum = float3(0.0f);
  float T = 1.0f;
  float past_depth = 0.0f;
  float past_power = 0.0f;
  for (uint i = 0u; i <= target_i; ++i) {
    uint g = sorted_ids[sorted_start + i];
    float2 m = means2d[g];
    uint g3 = g * 3u;
    float qa = conics[g3 + 0u];
    float qb = conics[g3 + 1u];
    float qc = conics[g3 + 2u];
    float alpha, raw_alpha, power;
    float2 dxy;
    bool active = eval_alpha(pixel, m, qa, qb, qc, opacities[g], mf, alpha, raw_alpha, power, dxy);
    if (!active || T <= mf.transmittance_threshold) {
      if (i == target_i) {
        target.active = false;
        target.delta = dxy;
        target.raw_alpha = raw_alpha;
        target.alpha_input = alpha;
        target.power = power;
        target.depth = depths[g];
        target.T0 = T;
        target.accum0 = accum;
        target.past_depth0 = past_depth;
        target.past_power0 = past_power;
      }
      continue;
    }

    bool is_target = (i == target_i);
    if (is_target) {
      target.active = true;
      target.delta = dxy;
      target.raw_alpha = raw_alpha;
      target.alpha_input = alpha;
      target.power = power;
      target.depth = depths[g];
      target.T0 = T;
      target.accum0 = accum;
      target.past_depth0 = past_depth;
      target.past_power0 = past_power;
    }

    if (T < 1.0f - mf.eps) {
      float T_orig = T * (1.0f - alpha);
      float a_past = 1.0f - T;
      float w_cur = 1.0f / (1.0f + exp(mf.softmax_beta * (past_power - power)));
      float soft_cur = w_cur * alpha;
      float soft_past = (1.0f - w_cur) * a_past;
      float soft_denom = max(soft_past + soft_cur, mf.eps);
      float target_absorbance = 1.0f - T_orig;
      float tilde_past = soft_past * target_absorbance / soft_denom;
      float tilde_cur_denom = max(soft_cur + soft_past * T_orig, mf.eps);
      float tilde_cur = soft_cur * target_absorbance / tilde_cur_denom;
      float depth_diff = depths[g] - past_depth;
      float decay = exp(-max(mf.softmax_gamma, 0.0f) * abs(depth_diff));
      float effective_past0 = decay * tilde_past + (1.0f - decay) * a_past;
      float alpha_soft0 = decay * tilde_cur + (1.0f - decay) * alpha;
      float pair_product = effective_past0 * alpha_soft0;
      float pair_sum = effective_past0 + alpha_soft0;
      float sqrt_disc = 0.0f;
      float scale = 1.0f;
      if (pair_product > mf.eps) {
        float disc = max(pair_sum * pair_sum - 4.0f * target_absorbance * pair_product, mf.eps);
        sqrt_disc = sqrt(disc);
        scale = (2.0f * target_absorbance) / max(pair_sum + sqrt_disc, mf.eps);
      }
      float effective_past = scale * effective_past0;
      alpha = scale * alpha_soft0;
      T = 1.0f - effective_past;
      float color_scale = effective_past / max(a_past, mf.eps);
      accum *= color_scale;
      if (is_target) {
        target.has_softmax = true;
        target.T_orig = T_orig;
        target.past_absorbance = a_past;
        target.w_cur = w_cur;
        target.soft_cur = soft_cur;
        target.soft_past = soft_past;
        target.soft_denom = soft_denom;
        target.target_absorbance = target_absorbance;
        target.tilde_past = tilde_past;
        target.tilde_cur = tilde_cur;
        target.tilde_cur_denom = tilde_cur_denom;
        target.depth_diff = depth_diff;
        target.decay = decay;
        target.effective_past0 = effective_past0;
        target.alpha_soft0 = alpha_soft0;
        target.pair_product = pair_product;
        target.pair_sum = pair_sum;
        target.sqrt_disc = sqrt_disc;
        target.scale = scale;
        target.effective_past = effective_past;
        target.color_scale = color_scale;
      }
    }

    if (is_target) {
      target.T_pre = T;
      target.accum_pre = accum;
      target.alpha_eff = alpha;
    }

    float w = T * alpha;
    accum += w * float3(colors[g3 + 0u], colors[g3 + 1u], colors[g3 + 2u]);
    float denom = max(1.0f - T + w, mf.eps);
    past_depth = (past_depth * (1.0f - T) + depths[g] * w) / denom;
    past_power = (past_power * (1.0f - T) + power * w) / denom;
    T *= (1.0f - alpha);
    if (is_target) {
      target.contribution_weight = w;
      target.denom = denom;
    }
  }
  return target;
}

inline SoftmaxReplayState replay_softmax_to_selected_slot(
    const device float2* means2d,
    const device float* conics,
    const device float* colors,
    const device float* opacities,
    const device float* depths,
    const device int* selected_ids,
    uint selected_base,
    uint target_i,
    float2 pixel,
    constant MetaF32& mf) {
  SoftmaxReplayState target = make_empty_replay_state();
  float3 accum = float3(0.0f);
  float T = 1.0f;
  float past_depth = 0.0f;
  float past_power = 0.0f;
  for (uint i = 0u; i <= target_i; ++i) {
    int selected = selected_ids[selected_base + i];
    if (selected < 0) continue;
    uint g = uint(selected);
    float2 m = means2d[g];
    uint g3 = g * 3u;
    float qa = conics[g3 + 0u];
    float qb = conics[g3 + 1u];
    float qc = conics[g3 + 2u];
    float alpha, raw_alpha, power;
    float2 dxy;
    bool active = eval_alpha(pixel, m, qa, qb, qc, opacities[g], mf, alpha, raw_alpha, power, dxy);
    if (!active || T <= mf.transmittance_threshold) {
      if (i == target_i) {
        target.active = false;
        target.delta = dxy;
        target.raw_alpha = raw_alpha;
        target.alpha_input = alpha;
        target.power = power;
        target.depth = depths[g];
        target.T0 = T;
        target.accum0 = accum;
        target.past_depth0 = past_depth;
        target.past_power0 = past_power;
      }
      continue;
    }

    bool is_target = (i == target_i);
    if (is_target) {
      target.active = true;
      target.delta = dxy;
      target.raw_alpha = raw_alpha;
      target.alpha_input = alpha;
      target.power = power;
      target.depth = depths[g];
      target.T0 = T;
      target.accum0 = accum;
      target.past_depth0 = past_depth;
      target.past_power0 = past_power;
    }

    if (T < 1.0f - mf.eps) {
      float T_orig = T * (1.0f - alpha);
      float a_past = 1.0f - T;
      float w_cur = 1.0f / (1.0f + exp(mf.softmax_beta * (past_power - power)));
      float soft_cur = w_cur * alpha;
      float soft_past = (1.0f - w_cur) * a_past;
      float soft_denom = max(soft_past + soft_cur, mf.eps);
      float target_absorbance = 1.0f - T_orig;
      float tilde_past = soft_past * target_absorbance / soft_denom;
      float tilde_cur_denom = max(soft_cur + soft_past * T_orig, mf.eps);
      float tilde_cur = soft_cur * target_absorbance / tilde_cur_denom;
      float depth_diff = depths[g] - past_depth;
      float decay = exp(-max(mf.softmax_gamma, 0.0f) * abs(depth_diff));
      float effective_past0 = decay * tilde_past + (1.0f - decay) * a_past;
      float alpha_soft0 = decay * tilde_cur + (1.0f - decay) * alpha;
      float pair_product = effective_past0 * alpha_soft0;
      float pair_sum = effective_past0 + alpha_soft0;
      float sqrt_disc = 0.0f;
      float scale = 1.0f;
      if (pair_product > mf.eps) {
        float disc = max(pair_sum * pair_sum - 4.0f * target_absorbance * pair_product, mf.eps);
        sqrt_disc = sqrt(disc);
        scale = (2.0f * target_absorbance) / max(pair_sum + sqrt_disc, mf.eps);
      }
      float effective_past = scale * effective_past0;
      alpha = scale * alpha_soft0;
      T = 1.0f - effective_past;
      float color_scale = effective_past / max(a_past, mf.eps);
      accum *= color_scale;
      if (is_target) {
        target.has_softmax = true;
        target.T_orig = T_orig;
        target.past_absorbance = a_past;
        target.w_cur = w_cur;
        target.soft_cur = soft_cur;
        target.soft_past = soft_past;
        target.soft_denom = soft_denom;
        target.target_absorbance = target_absorbance;
        target.tilde_past = tilde_past;
        target.tilde_cur = tilde_cur;
        target.tilde_cur_denom = tilde_cur_denom;
        target.depth_diff = depth_diff;
        target.decay = decay;
        target.effective_past0 = effective_past0;
        target.alpha_soft0 = alpha_soft0;
        target.pair_product = pair_product;
        target.pair_sum = pair_sum;
        target.sqrt_disc = sqrt_disc;
        target.scale = scale;
        target.effective_past = effective_past;
        target.color_scale = color_scale;
      }
    }

    if (is_target) {
      target.T_pre = T;
      target.accum_pre = accum;
      target.alpha_eff = alpha;
    }

    float w = T * alpha;
    accum += w * float3(colors[g3 + 0u], colors[g3 + 1u], colors[g3 + 2u]);
    float denom = max(1.0f - T + w, mf.eps);
    past_depth = (past_depth * (1.0f - T) + depths[g] * w) / denom;
    past_power = (past_power * (1.0f - T) + power * w) / denom;
    T *= (1.0f - alpha);
    if (is_target) {
      target.contribution_weight = w;
      target.denom = denom;
    }
  }
  return target;
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

inline void load_chunk_params_with_depths(
    const device float2* means2d,
    const device float* conics,
    const device float* colors,
    const device float* opacities,
    const device float* depths,
    const threadgroup uint* shared_ids,
    uint chunk_start,
    uint chunk_n,
    uint tid,
    threadgroup float2* sh_means,
    threadgroup float* sh_conics,
    threadgroup float* sh_colors,
    threadgroup float* sh_opacities,
    threadgroup float* sh_depths) {
  load_chunk_params(
      means2d, conics, colors, opacities, shared_ids, chunk_start, chunk_n, tid,
      sh_means, sh_conics, sh_colors, sh_opacities);
  for (uint i = tid; i < chunk_n; i += GSP_THREADS) {
    uint g = shared_ids[chunk_start + i];
    sh_depths[i] = depths[g];
  }
}

inline uint reduce_alive(uint alive, uint simd_lane, uint simd_group, threadgroup uint* tg_alive) {
  uint sg_sum = simd_sum(alive);
  if (simd_lane == 0u) tg_alive[simd_group] = sg_sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_group == 0u) {
    uint v = (simd_lane < GSP_SIMDGROUPS) ? tg_alive[simd_lane] : 0u;
    uint total = simd_sum(v);
    if (simd_lane == 0u) tg_alive[0] = total;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return tg_alive[0];
}

inline uint reduce_max_u32(uint value, uint simd_lane, uint simd_group, threadgroup uint* tg_tmp) {
  uint sg_max = simd_max(value);
  if (simd_lane == 0u) tg_tmp[simd_group] = sg_max;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_group == 0u) {
    uint v = (simd_lane < GSP_SIMDGROUPS) ? tg_tmp[simd_lane] : 0u;
    uint total = simd_max(v);
    if (simd_lane == 0u) tg_tmp[0] = total;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return tg_tmp[0];
}

inline void tile_batch_local(uint global_tile, constant MetaI32& mi, thread uint& batch, thread uint& local_tile) {
  batch = global_tile / uint(mi.tiles_per_image);
  local_tile = global_tile - batch * uint(mi.tiles_per_image);
}

inline uint pixel_index(uint batch, uint x, uint y, constant MetaI32& mi) {
  return (batch * uint(mi.height) + y) * uint(mi.width) + x;
}

inline void tile_pixel_from_tid(uint global_tile, uint tid, constant MetaI32& mi, thread uint& batch, thread uint& x, thread uint& y) {
  uint local_tile;
  tile_batch_local(global_tile, mi, batch, local_tile);
  uint tile_x = local_tile % uint(mi.tiles_x);
  uint tile_y = local_tile / uint(mi.tiles_x);
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
  uint batch = gid / uint(mi.gaussians_per_batch);
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
        uint local_tile = uint(ty * mi.tiles_x + tx);
        uint tile = batch * uint(mi.tiles_per_image) + local_tile;
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
  uint batch = gid / uint(mi.gaussians_per_batch);
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
        uint local_tile = uint(ty * mi.tiles_x + tx);
        uint tile = batch * uint(mi.tiles_per_image) + local_tile;
        uint idx = atomic_fetch_add_explicit(tile_cursors + tile, 1u, memory_order_relaxed);
        binned_ids[idx] = gid;
      }
    }
  }
}

kernel void tile_fast_softmax_bounded_tape(
    const device uint* tile_counts [[buffer(0)]],
    const device int* tile_offsets [[buffer(1)]],
    const device uint* binned_ids [[buffer(2)]],
    const device float2* means2d [[buffer(3)]],
    const device float* conics [[buffer(4)]],
    const device float* opacities [[buffer(5)]],
    const device float* depths [[buffer(6)]],
    constant MetaI32& mi [[buffer(7)]],
    constant MetaF32& mf [[buffer(8)]],
    device int* out_ids [[buffer(9)]],
    device float* out_weights [[buffer(10)]],
    device float* out_residual [[buffer(11)]],
    device float* out_final_alpha [[buffer(12)]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]]) {
  if (tg_id >= uint(mi.tile_count)) return;
  uint batch, x, y;
  tile_pixel_from_tid(tg_id, tid, mi, batch, x, y);
  bool valid = (x < uint(mi.width) && y < uint(mi.height));
  uint count = tile_counts[tg_id];
  if (!valid || count == 0u || count > uint(mi.max_fast_pairs)) return;

  threadgroup uint shared_ids[GSP_FAST_CAP];
  uint start = uint(tile_offsets[tg_id]);
  for (uint i = tid; i < count; i += GSP_THREADS) shared_ids[i] = binned_ids[start + i];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bitonic_sort_ids(shared_ids, count, tid);

  uint tape_k = min(uint(mi.softmax_tape_k), GSP_TAPE_CAP);
  uint ids[GSP_TAPE_CAP];
  float weights[GSP_TAPE_CAP];
  init_bounded_tape(ids, weights);

  float T = 1.0f;
  float past_depth = 0.0f;
  float past_power = 0.0f;
  float2 p = float2(float(x) + 0.5f, float(y) + 0.5f);
  for (uint i = 0u; i < count; ++i) {
    if (T <= mf.transmittance_threshold) break;
    uint g = shared_ids[i];
    uint g3 = g * 3u;
    float alpha, raw_alpha, power;
    float2 dxy;
    if (!eval_alpha(
            p,
            means2d[g],
            conics[g3 + 0u],
            conics[g3 + 1u],
            conics[g3 + 2u],
            opacities[g],
            mf,
            alpha,
            raw_alpha,
            power,
            dxy)) {
      continue;
    }
    if (softmax_gs_enabled(mi)) {
      float prefix_scale = apply_softmax_gs_tape_scalar(mf, depths[g], power, alpha, T, past_depth, past_power);
      scale_bounded_tape(ids, weights, tape_k, prefix_scale);
    }
    float w = T * alpha;
    insert_bounded_tape_weight(ids, weights, tape_k, g, w);
    float denom = max(1.0f - T + w, mf.eps);
    past_depth = (past_depth * (1.0f - T) + depths[g] * w) / denom;
    past_power = (past_power * (1.0f - T) + power * w) / denom;
    T *= (1.0f - alpha);
  }

  sort_bounded_tape_by_id(ids, weights, tape_k);
  uint pix = pixel_index(batch, x, y, mi);
  float final_alpha = 1.0f - T;
  float residual = max(final_alpha - selected_weight_sum(ids, weights, tape_k), 0.0f);
  for (uint slot = 0u; slot < GSP_TAPE_CAP; ++slot) {
    if (slot >= tape_k) break;
    uint out_idx = pix * tape_k + slot;
    out_ids[out_idx] = (ids[slot] == 0xFFFFFFFFu) ? -1 : int(ids[slot]);
    out_weights[out_idx] = (ids[slot] == 0xFFFFFFFFu) ? 0.0f : weights[slot];
  }
  out_residual[pix] = residual;
  out_final_alpha[pix] = final_alpha;
}

kernel void softmax_tape_color_backward(
    const device float* grad_rgb [[buffer(0)]],
    const device int* selected_ids [[buffer(1)]],
    const device float* selected_weights [[buffer(2)]],
    constant MetaI32& mi [[buffer(3)]],
    device atomic_float* g_colors [[buffer(4)]],
    uint gid [[thread_position_in_grid]]) {
  uint tape_k = uint(mi.softmax_tape_k);
  if (tape_k == 0u) return;
  uint total = uint(mi.batch_size) * uint(mi.height) * uint(mi.width) * tape_k;
  if (gid >= total) return;
  int selected = selected_ids[gid];
  if (selected < 0) return;
  float w = selected_weights[gid];
  if (w == 0.0f) return;
  uint pix = gid / tape_k;
  uint rgb = pix * 3u;
  uint g3 = uint(selected) * 3u;
  atomic_fetch_add_explicit(&g_colors[g3 + 0u], w * grad_rgb[rgb + 0u], memory_order_relaxed);
  atomic_fetch_add_explicit(&g_colors[g3 + 1u], w * grad_rgb[rgb + 1u], memory_order_relaxed);
  atomic_fetch_add_explicit(&g_colors[g3 + 2u], w * grad_rgb[rgb + 2u], memory_order_relaxed);
}

kernel void softmax_tape_scalar_backward(
    const device float* grad_rgb [[buffer(0)]],
    const device int* selected_ids [[buffer(1)]],
    const device float2* means2d [[buffer(2)]],
    const device float* conics [[buffer(3)]],
    const device float* colors [[buffer(4)]],
    const device float* opacities [[buffer(5)]],
    const device float* depths [[buffer(6)]],
    constant MetaI32& mi [[buffer(7)]],
    constant MetaF32& mf [[buffer(8)]],
    device atomic_float* g_means2d [[buffer(9)]],
    device atomic_float* g_conics [[buffer(10)]],
    device atomic_float* g_opacities [[buffer(11)]],
    device atomic_float* g_depths [[buffer(12)]],
    uint gid [[thread_position_in_grid]]) {
  uint tape_k = min(uint(mi.softmax_tape_k), GSP_TAPE_CAP);
  if (tape_k == 0u) return;
  uint total_pixels = uint(mi.batch_size) * uint(mi.height) * uint(mi.width);
  if (gid >= total_pixels) return;

  uint x = gid % uint(mi.width);
  uint tmp = gid / uint(mi.width);
  uint y = tmp % uint(mi.height);
  float2 pxy = float2(float(x) + 0.5f, float(y) + 0.5f);
  uint selected_base = gid * tape_k;
  uint rgb = gid * 3u;
  float3 go = float3(grad_rgb[rgb + 0u], grad_rgb[rgb + 1u], grad_rgb[rgb + 2u]);

  float3 g_accum = go;
  float gT = dot(go, float3(mf.bg_r, mf.bg_g, mf.bg_b));
  float gD = 0.0f;
  float gP = 0.0f;

  for (int si = int(tape_k) - 1; si >= 0; --si) {
    int selected = selected_ids[selected_base + uint(si)];
    if (selected < 0) continue;
    uint g = uint(selected);
    uint g3 = g * 3u;
    SoftmaxReplayState st = replay_softmax_to_selected_slot(
        means2d, conics, colors, opacities, depths, selected_ids, selected_base, uint(si), pxy, mf);
    if (!st.active) continue;

    float3 c = float3(colors[g3 + 0u], colors[g3 + 1u], colors[g3 + 2u]);
    float a = st.alpha_eff;
    float Tpre = st.T_pre;
    float w = st.contribution_weight;
    float denom = st.denom;

    float g_Tpre = gT * (1.0f - a);
    float g_alpha = gT * (-Tpre);

    float nD = st.past_depth0 * (1.0f - Tpre) + st.depth * w;
    float nP = st.past_power0 * (1.0f - Tpre) + st.power * w;
    float g_nD = gD / denom;
    float g_den = gD * (-nD / max(denom * denom, mf.eps));
    float g_nP = gP / denom;
    g_den += gP * (-nP / max(denom * denom, mf.eps));

    float g_past_depth0 = g_nD * (1.0f - Tpre);
    g_Tpre += g_nD * (-st.past_depth0);
    float g_depth_cur = g_nD * w;
    float g_w = g_nD * st.depth;

    float g_past_power0 = g_nP * (1.0f - Tpre);
    g_Tpre += g_nP * (-st.past_power0);
    float g_power_cur = g_nP * w;
    g_w += g_nP * st.power;

    g_Tpre -= g_den;
    g_w += g_den;

    float3 g_accum_pre = g_accum;
    g_w += dot(g_accum, c);

    g_Tpre += g_w * a;
    g_alpha += g_w * Tpre;

    float g_a_input = 0.0f;
    if (st.has_softmax) {
      float g_T1 = g_Tpre;
      float3 g_accum1 = g_accum_pre;
      float g_alpha_eff = g_alpha;

      float3 g_accum0 = g_accum1 * st.color_scale;
      float g_color_scale = dot(g_accum1, st.accum0);
      float g_effective_past = -g_T1 + g_color_scale / max(st.past_absorbance, mf.eps);
      float g_past_absorbance = g_color_scale * (-st.effective_past / max(st.past_absorbance * st.past_absorbance, mf.eps));

      float g_scale = g_effective_past * st.effective_past0 + g_alpha_eff * st.alpha_soft0;
      float g_effective_past0 = g_effective_past * st.scale;
      float g_alpha_soft0 = g_alpha_eff * st.scale;
      float g_target_absorbance = 0.0f;
      if (st.pair_product > mf.eps) {
        float denom_scale = max(st.pair_sum + st.sqrt_disc, mf.eps);
        g_target_absorbance += g_scale * (2.0f / denom_scale);
        float g_denom_scale = g_scale * (-2.0f * st.target_absorbance / max(denom_scale * denom_scale, mf.eps));
        float g_pair_sum = g_denom_scale;
        float g_sqrt_disc = g_denom_scale;
        float g_disc = g_sqrt_disc * (0.5f / max(st.sqrt_disc, mf.eps));
        g_pair_sum += g_disc * (2.0f * st.pair_sum);
        g_target_absorbance += g_disc * (-4.0f * st.pair_product);
        float g_pair_product = g_disc * (-4.0f * st.target_absorbance);
        g_effective_past0 += g_pair_sum + g_pair_product * st.alpha_soft0;
        g_alpha_soft0 += g_pair_sum + g_pair_product * st.effective_past0;
      }

      float g_decay = g_effective_past0 * (st.tilde_past - st.past_absorbance)
          + g_alpha_soft0 * (st.tilde_cur - st.alpha_input);
      float g_tilde_past = g_effective_past0 * st.decay;
      g_past_absorbance += g_effective_past0 * (1.0f - st.decay);
      float g_tilde_cur = g_alpha_soft0 * st.decay;
      g_a_input += g_alpha_soft0 * (1.0f - st.decay);

      float depth_sign = sign(st.depth_diff);
      float g_depth_diff = g_decay * st.decay * (-max(mf.softmax_gamma, 0.0f) * depth_sign);
      g_depth_cur += g_depth_diff;
      g_past_depth0 -= g_depth_diff;

      float g_soft_cur = g_tilde_cur * st.target_absorbance / st.tilde_cur_denom;
      g_target_absorbance += g_tilde_cur * st.soft_cur / st.tilde_cur_denom;
      float g_tilde_cur_denom = g_tilde_cur * (-st.soft_cur * st.target_absorbance / max(st.tilde_cur_denom * st.tilde_cur_denom, mf.eps));
      g_soft_cur += g_tilde_cur_denom;
      float g_soft_past = g_tilde_cur_denom * st.T_orig;
      float g_T_orig = g_tilde_cur_denom * st.soft_past;

      g_soft_past += g_tilde_past * st.target_absorbance / st.soft_denom;
      g_target_absorbance += g_tilde_past * st.soft_past / st.soft_denom;
      float g_soft_denom = g_tilde_past * (-st.soft_past * st.target_absorbance / max(st.soft_denom * st.soft_denom, mf.eps));
      g_soft_past += g_soft_denom;
      g_soft_cur += g_soft_denom;

      g_T_orig -= g_target_absorbance;

      float g_w_cur = g_soft_cur * st.alpha_input;
      g_a_input += g_soft_cur * st.w_cur;
      g_w_cur -= g_soft_past * st.past_absorbance;
      g_past_absorbance += g_soft_past * (1.0f - st.w_cur);

      float g_z = g_w_cur * st.w_cur * (1.0f - st.w_cur);
      g_power_cur += g_z * mf.softmax_beta;
      g_past_power0 -= g_z * mf.softmax_beta;

      float g_T0 = g_T_orig * (1.0f - st.alpha_input);
      g_a_input += g_T_orig * (-st.T0);
      g_T0 -= g_past_absorbance;

      g_accum = g_accum0;
      gT = g_T0;
      gD = g_past_depth0;
      gP = g_past_power0;
    } else {
      g_accum = g_accum_pre;
      gT = g_Tpre;
      gD = g_past_depth0;
      gP = g_past_power0;
      g_a_input += g_alpha;
    }

    float gate = (st.raw_alpha < mf.max_alpha) ? 1.0f : 0.0f;
    float g_raw = g_a_input * gate;
    float g_power_from_alpha = g_raw * st.raw_alpha;
    float g_power_total = g_power_cur + g_power_from_alpha;
    float l_ga = g_power_total * (-0.5f) * st.delta.x * st.delta.x;
    float l_gb = g_power_total * (-1.0f) * st.delta.x * st.delta.y;
    float l_gc = g_power_total * (-0.5f) * st.delta.y * st.delta.y;
    float qa = conics[g3 + 0u];
    float qb = conics[g3 + 1u];
    float qc = conics[g3 + 2u];
    float g_dx = g_power_total * (-(qa * st.delta.x + qb * st.delta.y));
    float g_dy = g_power_total * (-(qb * st.delta.x + qc * st.delta.y));
    float l_gmx = -g_dx;
    float l_gmy = -g_dy;
    float l_gop = g_raw * (st.raw_alpha / max(opacities[g], mf.eps));

    atomic_fetch_add_explicit(&g_means2d[g * 2u + 0u], l_gmx, memory_order_relaxed);
    atomic_fetch_add_explicit(&g_means2d[g * 2u + 1u], l_gmy, memory_order_relaxed);
    atomic_fetch_add_explicit(&g_conics[g * 3u + 0u], l_ga, memory_order_relaxed);
    atomic_fetch_add_explicit(&g_conics[g * 3u + 1u], l_gb, memory_order_relaxed);
    atomic_fetch_add_explicit(&g_conics[g * 3u + 2u], l_gc, memory_order_relaxed);
    atomic_fetch_add_explicit(&g_opacities[g], l_gop, memory_order_relaxed);
    atomic_fetch_add_explicit(&g_depths[g], g_depth_cur, memory_order_relaxed);
  }
}

// eval fast path: no writeback, no stop-count save
kernel void tile_fast_forward_eval(
    const device uint* tile_counts [[buffer(0)]],
    const device int* tile_offsets [[buffer(1)]],
    const device uint* binned_ids [[buffer(2)]],
    const device float2* means2d [[buffer(3)]],
    const device float* conics [[buffer(4)]],
    const device float* colors [[buffer(5)]],
    const device float* opacities [[buffer(6)]],
    const device float* depths [[buffer(7)]],
    constant MetaI32& mi [[buffer(8)]],
    constant MetaF32& mf [[buffer(9)]],
    device float* out_rgb [[buffer(10)]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  if (tg_id >= uint(mi.tile_count)) return;
  uint batch, x, y;
  tile_pixel_from_tid(tg_id, tid, mi, batch, x, y);
  bool valid = (x < uint(mi.width) && y < uint(mi.height));
  uint pix = valid ? pixel_index(batch, x, y, mi) : 0u;
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
  threadgroup float sh_depths[GSP_CHUNK];
  threadgroup uint tg_alive[GSP_SIMDGROUPS];
  uint start = uint(tile_offsets[tg_id]);
  for (uint i = tid; i < count; i += GSP_THREADS) shared_ids[i] = binned_ids[start + i];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bitonic_sort_ids(shared_ids, count, tid);
  float3 accum = float3(0.0f);
  float T = 1.0f;
  float past_depth = 0.0f;
  float past_power = 0.0f;
  float2 p = float2(float(x) + 0.5f, float(y) + 0.5f);
  for (uint chunk_start = 0u; chunk_start < count; chunk_start += GSP_CHUNK) {
    uint chunk_n = min(GSP_CHUNK, count - chunk_start);
    load_chunk_params_with_depths(
        means2d, conics, colors, opacities, depths, shared_ids, chunk_start, chunk_n, tid,
        sh_means, sh_conics, sh_colors, sh_opacities, sh_depths);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint alive_total = reduce_alive((valid && T > mf.transmittance_threshold) ? 1u : 0u, simd_lane, simd_group, tg_alive);
    if (alive_total == 0u) break;
    if (valid && T > mf.transmittance_threshold) {
      for (uint j = 0u; j < chunk_n; ++j) {
        float alpha, raw_alpha, power; float2 d;
        float2 m = sh_means[j];
        float3 q = load3_sh(sh_conics, j);
        if (!eval_alpha(p, m, q.x, q.y, q.z, sh_opacities[j], mf, alpha, raw_alpha, power, d)) continue;
        if (softmax_gs_enabled(mi)) {
          apply_softmax_gs_forward(mf, sh_depths[j], power, alpha, T, accum, past_depth, past_power);
        }
        float w = T * alpha;
        accum += w * load3_sh(sh_colors, j);
        float denom = max(1.0f - T + w, mf.eps);
        past_depth = (past_depth * (1.0f - T) + sh_depths[j] * w) / denom;
        past_power = (past_power * (1.0f - T) + power * w) / denom;
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

// train fast path: writes sorted IDs back into binned_ids and saves per-tile stop count
kernel void tile_fast_forward_state(
    const device uint* tile_counts [[buffer(0)]],
    const device int* tile_offsets [[buffer(1)]],
    device uint* binned_ids [[buffer(2)]],
    const device float2* means2d [[buffer(3)]],
    const device float* conics [[buffer(4)]],
    const device float* colors [[buffer(5)]],
    const device float* opacities [[buffer(6)]],
    const device float* depths [[buffer(7)]],
    constant MetaI32& mi [[buffer(8)]],
    constant MetaF32& mf [[buffer(9)]],
    device float* out_rgb [[buffer(10)]],
    device int* out_stop_counts [[buffer(11)]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  if (tg_id >= uint(mi.tile_count)) return;
  uint batch, x, y;
  tile_pixel_from_tid(tg_id, tid, mi, batch, x, y);
  bool valid = (x < uint(mi.width) && y < uint(mi.height));
  uint pix = valid ? pixel_index(batch, x, y, mi) : 0u;
  uint count = tile_counts[tg_id];
  uint start = uint(tile_offsets[tg_id]);
  if (count == 0u || count > uint(mi.max_fast_pairs)) {
    if (tid == 0u) out_stop_counts[tg_id] = 0;
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
  threadgroup float sh_depths[GSP_CHUNK];
  threadgroup uint tg_alive[GSP_SIMDGROUPS];
  threadgroup uint tg_stop[GSP_SIMDGROUPS];
  for (uint i = tid; i < count; i += GSP_THREADS) shared_ids[i] = binned_ids[start + i];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bitonic_sort_ids(shared_ids, count, tid);
  for (uint i = tid; i < count; i += GSP_THREADS) binned_ids[start + i] = shared_ids[i];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float3 accum = float3(0.0f);
  float T = 1.0f;
  float past_depth = 0.0f;
  float past_power = 0.0f;
  uint local_stop = 0u;
  float2 p = float2(float(x) + 0.5f, float(y) + 0.5f);
  for (uint chunk_start = 0u; chunk_start < count; chunk_start += GSP_CHUNK) {
    uint chunk_n = min(GSP_CHUNK, count - chunk_start);
    load_chunk_params_with_depths(
        means2d, conics, colors, opacities, depths, shared_ids, chunk_start, chunk_n, tid,
        sh_means, sh_conics, sh_colors, sh_opacities, sh_depths);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint alive_total = reduce_alive((valid && T > mf.transmittance_threshold) ? 1u : 0u, simd_lane, simd_group, tg_alive);
    if (alive_total == 0u) break;
    if (valid && T > mf.transmittance_threshold) {
      for (uint j = 0u; j < chunk_n; ++j) {
        local_stop = chunk_start + j + 1u;
        float alpha, raw_alpha, power; float2 d;
        float2 m = sh_means[j];
        float3 q = load3_sh(sh_conics, j);
        if (!eval_alpha(p, m, q.x, q.y, q.z, sh_opacities[j], mf, alpha, raw_alpha, power, d)) continue;
        if (softmax_gs_enabled(mi)) {
          apply_softmax_gs_forward(mf, sh_depths[j], power, alpha, T, accum, past_depth, past_power);
        }
        float w = T * alpha;
        accum += w * load3_sh(sh_colors, j);
        float denom = max(1.0f - T + w, mf.eps);
        past_depth = (past_depth * (1.0f - T) + sh_depths[j] * w) / denom;
        past_power = (past_power * (1.0f - T) + power * w) / denom;
        T *= (1.0f - alpha);
        if (T <= mf.transmittance_threshold) break;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  uint tile_stop = reduce_max_u32(local_stop, simd_lane, simd_group, tg_stop);
  if (tid == 0u) out_stop_counts[tg_id] = int(tile_stop);
  if (valid) {
    out_rgb[pix * 3u + 0u] = accum.x + T * mf.bg_r;
    out_rgb[pix * 3u + 1u] = accum.y + T * mf.bg_g;
    out_rgb[pix * 3u + 2u] = accum.z + T * mf.bg_b;
  }
}

kernel void tile_fast_backward_saved(
    const device float* grad_rgb [[buffer(0)]],
    const device uint* tile_counts [[buffer(1)]],
    const device int* tile_offsets [[buffer(2)]],
    const device uint* binned_ids [[buffer(3)]],
    const device int* tile_stop_counts [[buffer(4)]],
    const device float2* means2d [[buffer(5)]],
    const device float* conics [[buffer(6)]],
    const device float* colors [[buffer(7)]],
    const device float* opacities [[buffer(8)]],
    constant MetaI32& mi [[buffer(9)]],
    constant MetaF32& mf [[buffer(10)]],
    device atomic_float* g_means2d [[buffer(11)]],
    device atomic_float* g_conics [[buffer(12)]],
    device atomic_float* g_colors [[buffer(13)]],
    device atomic_float* g_opacities [[buffer(14)]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  if (tg_id >= uint(mi.tile_count)) return;
  uint count = tile_counts[tg_id];
  uint stop_count = min(count, uint(max(tile_stop_counts[tg_id], 0)));
  if (count == 0u || count > uint(mi.max_fast_pairs) || stop_count == 0u) return;
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
  for (uint i = tid; i < stop_count; i += GSP_THREADS) shared_ids[i] = binned_ids[start + i];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint batch, x, y;
  tile_pixel_from_tid(tg_id, tid, mi, batch, x, y);
  bool valid = (x < uint(mi.width) && y < uint(mi.height));
  uint pix = valid ? pixel_index(batch, x, y, mi) : 0u;
  float2 p = float2(float(x) + 0.5f, float(y) + 0.5f);
  float3 go = valid ? float3(grad_rgb[pix * 3u + 0u], grad_rgb[pix * 3u + 1u], grad_rgb[pix * 3u + 2u]) : float3(0.0f);
  float T_final = 1.0f;
  uint end_i = stop_count;
  for (uint chunk_start = 0u; chunk_start < stop_count; chunk_start += GSP_CHUNK) {
    uint chunk_n = min(GSP_CHUNK, stop_count - chunk_start);
    load_chunk_params(means2d, conics, colors, opacities, shared_ids, chunk_start, chunk_n, tid, sh_means, sh_conics, sh_colors, sh_opacities);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint alive_total = reduce_alive((valid && T_final > mf.transmittance_threshold) ? 1u : 0u, simd_lane, simd_group, tg_alive);
    if (alive_total == 0u) break;
    if (valid && T_final > mf.transmittance_threshold) {
      for (uint j = 0u; j < chunk_n; ++j) {
        float alpha, raw_alpha, power; float2 d;
        float2 m = sh_means[j];
        float3 q = load3_sh(sh_conics, j);
        if (!eval_alpha(p, m, q.x, q.y, q.z, sh_opacities[j], mf, alpha, raw_alpha, power, d)) continue;
        T_final *= (1.0f - alpha);
        if (T_final <= mf.transmittance_threshold) { end_i = chunk_start + j + 1u; break; }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  float T_cur = T_final;
  float gT = valid ? dot(go, float3(mf.bg_r, mf.bg_g, mf.bg_b)) : 0.0f;
  // Keep barrier control uniform across the threadgroup. `end_i` is per pixel,
  // so using it as the loop bound makes saturated pixels take fewer barriers
  // than unsaturated pixels. Iterate over the tile-level stop count and mask
  // each pixel with `global_i < end_i` inside the loop.
  for (int chunk_end = int(stop_count); chunk_end > 0; chunk_end -= int(GSP_CHUNK)) {
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
      float l_gmx = 0.0f, l_gmy = 0.0f, l_ga = 0.0f, l_gb = 0.0f, l_gc = 0.0f, l_gop = 0.0f;
      float3 l_gcol = float3(0.0f);
      if (valid && global_i < end_i) {
        float alpha, raw_alpha, power; float2 d;
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
      if (simd_lane == 0u) { partial0[simd_group] = s0; partial1[simd_group] = s1; partial2[simd_group] = s2; }
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

kernel void tile_fast_backward_softmax_recompute(
    const device float* grad_rgb [[buffer(0)]],
    const device uint* tile_counts [[buffer(1)]],
    const device int* tile_offsets [[buffer(2)]],
    const device uint* binned_ids [[buffer(3)]],
    const device int* tile_stop_counts [[buffer(4)]],
    const device float2* means2d [[buffer(5)]],
    const device float* conics [[buffer(6)]],
    const device float* colors [[buffer(7)]],
    const device float* opacities [[buffer(8)]],
    const device float* depths [[buffer(9)]],
    constant MetaI32& mi [[buffer(10)]],
    constant MetaF32& mf [[buffer(11)]],
    device atomic_float* g_means2d [[buffer(12)]],
    device atomic_float* g_conics [[buffer(13)]],
    device atomic_float* g_colors [[buffer(14)]],
    device atomic_float* g_opacities [[buffer(15)]],
    device atomic_float* g_depths [[buffer(16)]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  if (tg_id >= uint(mi.tile_count)) return;
  uint count = tile_counts[tg_id];
  uint stop_count = min(count, uint(max(tile_stop_counts[tg_id], 0)));
  if (count == 0u || count > uint(mi.max_fast_pairs) || stop_count == 0u) return;

  threadgroup uint shared_ids[GSP_FAST_CAP];
  threadgroup float4 partial0[GSP_SIMDGROUPS];
  threadgroup float4 partial1[GSP_SIMDGROUPS];
  threadgroup float4 partial2[GSP_SIMDGROUPS];
  uint start = uint(tile_offsets[tg_id]);
  for (uint i = tid; i < stop_count; i += GSP_THREADS) shared_ids[i] = binned_ids[start + i];
  threadgroup_barrier(mem_flags::mem_threadgroup);

  uint batch, x, y;
  tile_pixel_from_tid(tg_id, tid, mi, batch, x, y);
  bool valid = (x < uint(mi.width) && y < uint(mi.height));
  uint pix = valid ? pixel_index(batch, x, y, mi) : 0u;
  float2 pxy = float2(float(x) + 0.5f, float(y) + 0.5f);
  float3 go = valid ? float3(grad_rgb[pix * 3u + 0u], grad_rgb[pix * 3u + 1u], grad_rgb[pix * 3u + 2u]) : float3(0.0f);

  float T_forward = 1.0f;
  float3 accum_forward = float3(0.0f);
  float past_depth = 0.0f;
  float past_power = 0.0f;
  uint end_i = stop_count;
  if (valid) {
    for (uint i = 0u; i < stop_count; ++i) {
      uint g = shared_ids[i];
      uint g3 = g * 3u;
      float2 m = means2d[g];
      float alpha, raw_alpha, power;
      float2 dxy;
      if (!eval_alpha(
              pxy,
              m,
              conics[g3 + 0u],
              conics[g3 + 1u],
              conics[g3 + 2u],
              opacities[g],
              mf,
              alpha,
              raw_alpha,
              power,
              dxy)) {
        continue;
      }
      if (T_forward <= mf.transmittance_threshold) {
        end_i = i;
        break;
      }
      apply_softmax_gs_forward(mf, depths[g], power, alpha, T_forward, accum_forward, past_depth, past_power);
      float w = T_forward * alpha;
      accum_forward += w * float3(colors[g3 + 0u], colors[g3 + 1u], colors[g3 + 2u]);
      float denom = max(1.0f - T_forward + w, mf.eps);
      past_depth = (past_depth * (1.0f - T_forward) + depths[g] * w) / denom;
      past_power = (past_power * (1.0f - T_forward) + power * w) / denom;
      T_forward *= (1.0f - alpha);
      if (T_forward <= mf.transmittance_threshold) {
        end_i = i + 1u;
        break;
      }
    }
  }

  float3 g_accum = go;
  float gT = valid ? dot(go, float3(mf.bg_r, mf.bg_g, mf.bg_b)) : 0.0f;
  float gD = 0.0f;
  float gP = 0.0f;

  for (int gi = int(stop_count) - 1; gi >= 0; --gi) {
    uint global_i = uint(gi);
    uint g = shared_ids[global_i];
    uint g3 = g * 3u;
    float l_gmx = 0.0f, l_gmy = 0.0f, l_ga = 0.0f, l_gb = 0.0f, l_gc = 0.0f;
    float l_gop = 0.0f, l_gdepth = 0.0f;
    float3 l_gcol = float3(0.0f);

    if (valid && global_i < end_i) {
      SoftmaxReplayState st = replay_softmax_to_index(
          means2d, conics, colors, opacities, depths, shared_ids, global_i, pxy, mf);
      if (st.active) {
        float3 c = float3(colors[g3 + 0u], colors[g3 + 1u], colors[g3 + 2u]);
        float a = st.alpha_eff;
        float Tpre = st.T_pre;
        float w = st.contribution_weight;
        float denom = st.denom;

        float g_Tpre = gT * (1.0f - a);
        float g_alpha = gT * (-Tpre);

        float nD = st.past_depth0 * (1.0f - Tpre) + st.depth * w;
        float nP = st.past_power0 * (1.0f - Tpre) + st.power * w;
        float g_nD = gD / denom;
        float g_den = gD * (-nD / max(denom * denom, mf.eps));
        float g_nP = gP / denom;
        g_den += gP * (-nP / max(denom * denom, mf.eps));

        float g_past_depth0 = g_nD * (1.0f - Tpre);
        g_Tpre += g_nD * (-st.past_depth0);
        float g_depth_cur = g_nD * w;
        float g_w = g_nD * st.depth;

        float g_past_power0 = g_nP * (1.0f - Tpre);
        g_Tpre += g_nP * (-st.past_power0);
        float g_power_cur = g_nP * w;
        g_w += g_nP * st.power;

        g_Tpre -= g_den;
        g_w += g_den;

        float3 g_accum_pre = g_accum;
        g_w += dot(g_accum, c);
        l_gcol = g_accum * w;

        g_Tpre += g_w * a;
        g_alpha += g_w * Tpre;

        float g_a_input = 0.0f;
        if (st.has_softmax) {
          float g_T1 = g_Tpre;
          float3 g_accum1 = g_accum_pre;
          float g_alpha_eff = g_alpha;

          float3 g_accum0 = g_accum1 * st.color_scale;
          float g_color_scale = dot(g_accum1, st.accum0);
          float g_effective_past = -g_T1 + g_color_scale / max(st.past_absorbance, mf.eps);
          float g_past_absorbance = g_color_scale * (-st.effective_past / max(st.past_absorbance * st.past_absorbance, mf.eps));

          float g_scale = g_effective_past * st.effective_past0 + g_alpha_eff * st.alpha_soft0;
          float g_effective_past0 = g_effective_past * st.scale;
          float g_alpha_soft0 = g_alpha_eff * st.scale;
          float g_target_absorbance = 0.0f;
          if (st.pair_product > mf.eps) {
            float denom_scale = max(st.pair_sum + st.sqrt_disc, mf.eps);
            g_target_absorbance += g_scale * (2.0f / denom_scale);
            float g_denom_scale = g_scale * (-2.0f * st.target_absorbance / max(denom_scale * denom_scale, mf.eps));
            float g_pair_sum = g_denom_scale;
            float g_sqrt_disc = g_denom_scale;
            float g_disc = g_sqrt_disc * (0.5f / max(st.sqrt_disc, mf.eps));
            g_pair_sum += g_disc * (2.0f * st.pair_sum);
            g_target_absorbance += g_disc * (-4.0f * st.pair_product);
            float g_pair_product = g_disc * (-4.0f * st.target_absorbance);
            g_effective_past0 += g_pair_sum + g_pair_product * st.alpha_soft0;
            g_alpha_soft0 += g_pair_sum + g_pair_product * st.effective_past0;
          }

          float g_decay = g_effective_past0 * (st.tilde_past - st.past_absorbance)
              + g_alpha_soft0 * (st.tilde_cur - st.alpha_input);
          float g_tilde_past = g_effective_past0 * st.decay;
          g_past_absorbance += g_effective_past0 * (1.0f - st.decay);
          float g_tilde_cur = g_alpha_soft0 * st.decay;
          g_a_input += g_alpha_soft0 * (1.0f - st.decay);

          float depth_sign = sign(st.depth_diff);
          float g_depth_diff = g_decay * st.decay * (-max(mf.softmax_gamma, 0.0f) * depth_sign);
          g_depth_cur += g_depth_diff;
          g_past_depth0 -= g_depth_diff;

          float g_soft_cur = g_tilde_cur * st.target_absorbance / st.tilde_cur_denom;
          g_target_absorbance += g_tilde_cur * st.soft_cur / st.tilde_cur_denom;
          float g_tilde_cur_denom = g_tilde_cur * (-st.soft_cur * st.target_absorbance / max(st.tilde_cur_denom * st.tilde_cur_denom, mf.eps));
          g_soft_cur += g_tilde_cur_denom;
          float g_soft_past = g_tilde_cur_denom * st.T_orig;
          float g_T_orig = g_tilde_cur_denom * st.soft_past;

          g_soft_past += g_tilde_past * st.target_absorbance / st.soft_denom;
          g_target_absorbance += g_tilde_past * st.soft_past / st.soft_denom;
          float g_soft_denom = g_tilde_past * (-st.soft_past * st.target_absorbance / max(st.soft_denom * st.soft_denom, mf.eps));
          g_soft_past += g_soft_denom;
          g_soft_cur += g_soft_denom;

          g_T_orig -= g_target_absorbance;

          float g_w_cur = g_soft_cur * st.alpha_input;
          g_a_input += g_soft_cur * st.w_cur;
          g_w_cur -= g_soft_past * st.past_absorbance;
          g_past_absorbance += g_soft_past * (1.0f - st.w_cur);

          float g_z = g_w_cur * st.w_cur * (1.0f - st.w_cur);
          g_power_cur += g_z * mf.softmax_beta;
          g_past_power0 -= g_z * mf.softmax_beta;

          float g_T0 = g_T_orig * (1.0f - st.alpha_input);
          g_a_input += g_T_orig * (-st.T0);
          g_T0 -= g_past_absorbance;

          g_accum = g_accum0;
          gT = g_T0;
          gD = g_past_depth0;
          gP = g_past_power0;
        } else {
          g_accum = g_accum_pre;
          gT = g_Tpre;
          gD = g_past_depth0;
          gP = g_past_power0;
          g_a_input += g_alpha;
        }

        float gate = (st.raw_alpha < mf.max_alpha) ? 1.0f : 0.0f;
        float g_raw = g_a_input * gate;
        float g_power_from_alpha = g_raw * st.raw_alpha;
        float g_power_total = g_power_cur + g_power_from_alpha;
        l_ga = g_power_total * (-0.5f) * st.delta.x * st.delta.x;
        l_gb = g_power_total * (-1.0f) * st.delta.x * st.delta.y;
        l_gc = g_power_total * (-0.5f) * st.delta.y * st.delta.y;
        float qa = conics[g3 + 0u];
        float qb = conics[g3 + 1u];
        float qc = conics[g3 + 2u];
        float g_dx = g_power_total * (-(qa * st.delta.x + qb * st.delta.y));
        float g_dy = g_power_total * (-(qb * st.delta.x + qc * st.delta.y));
        l_gmx = -g_dx;
        l_gmy = -g_dy;
        l_gop = g_raw * (st.raw_alpha / max(opacities[g], mf.eps));
        l_gdepth = g_depth_cur;
      }
    }

    float4 s0 = simd_sum(float4(l_gmx, l_gmy, l_ga, l_gb));
    float4 s1 = simd_sum(float4(l_gc, l_gcol.x, l_gcol.y, l_gcol.z));
    float4 s2 = simd_sum(float4(l_gop, l_gdepth, 0.0f, 0.0f));
    if (simd_lane == 0u) {
      partial0[simd_group] = s0;
      partial1[simd_group] = s1;
      partial2[simd_group] = s2;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group == 0u) {
      float4 v0 = (simd_lane < GSP_SIMDGROUPS) ? partial0[simd_lane] : float4(0.0f);
      float4 v1 = (simd_lane < GSP_SIMDGROUPS) ? partial1[simd_lane] : float4(0.0f);
      float4 v2 = (simd_lane < GSP_SIMDGROUPS) ? partial2[simd_lane] : float4(0.0f);
      float4 t0 = simd_sum(v0);
      float4 t1 = simd_sum(v1);
      float4 t2 = simd_sum(v2);
      if (simd_lane == 0u) {
        atomic_fetch_add_explicit(&g_means2d[g * 2u + 0u], t0.x, memory_order_relaxed);
        atomic_fetch_add_explicit(&g_means2d[g * 2u + 1u], t0.y, memory_order_relaxed);
        atomic_fetch_add_explicit(&g_conics[g * 3u + 0u], t0.z, memory_order_relaxed);
        atomic_fetch_add_explicit(&g_conics[g * 3u + 1u], t0.w, memory_order_relaxed);
        atomic_fetch_add_explicit(&g_conics[g * 3u + 2u], t1.x, memory_order_relaxed);
        atomic_add3(g_colors, g, float3(t1.y, t1.z, t1.w));
        atomic_fetch_add_explicit(&g_opacities[g], t2.x, memory_order_relaxed);
        atomic_fetch_add_explicit(&g_depths[g], t2.y, memory_order_relaxed);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

kernel void tile_overflow_backward_softmax_recompute(
    const device float* grad_tiles [[buffer(0)]],
    const device uint* overflow_tile_ids [[buffer(1)]],
    const device int* overflow_tile_offsets [[buffer(2)]],
    const device uint* overflow_sorted_ids [[buffer(3)]],
    const device float2* means2d [[buffer(4)]],
    const device float* conics [[buffer(5)]],
    const device float* colors [[buffer(6)]],
    const device float* opacities [[buffer(7)]],
    const device float* depths [[buffer(8)]],
    constant MetaI32& mi [[buffer(9)]],
    constant MetaF32& mf [[buffer(10)]],
    device atomic_float* g_means2d [[buffer(11)]],
    device atomic_float* g_conics [[buffer(12)]],
    device atomic_float* g_colors [[buffer(13)]],
    device atomic_float* g_opacities [[buffer(14)]],
    device atomic_float* g_depths [[buffer(15)]],
    uint local_tile_idx [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  uint tile_id = overflow_tile_ids[local_tile_idx];
  uint start = uint(overflow_tile_offsets[local_tile_idx]);
  uint end = uint(overflow_tile_offsets[local_tile_idx + 1]);
  uint count = end - start;
  if (count == 0u) return;

  threadgroup float4 partial0[GSP_SIMDGROUPS];
  threadgroup float4 partial1[GSP_SIMDGROUPS];
  threadgroup float4 partial2[GSP_SIMDGROUPS];

  uint batch, x, y;
  tile_pixel_from_tid(tile_id, tid, mi, batch, x, y);
  bool valid = (x < uint(mi.width) && y < uint(mi.height));
  uint px = tid % GSP_TILE_SIZE;
  uint py = tid / GSP_TILE_SIZE;
  uint base = tile_img_base(local_tile_idx, px, py);
  float2 pxy = float2(float(x) + 0.5f, float(y) + 0.5f);
  float3 go = valid ? float3(grad_tiles[base + 0u], grad_tiles[base + 1u], grad_tiles[base + 2u]) : float3(0.0f);

  float T_forward = 1.0f;
  float3 accum_forward = float3(0.0f);
  float past_depth = 0.0f;
  float past_power = 0.0f;
  uint end_i = count;
  if (valid) {
    for (uint i = 0u; i < count; ++i) {
      uint g = overflow_sorted_ids[start + i];
      uint g3 = g * 3u;
      float2 m = means2d[g];
      float alpha, raw_alpha, power;
      float2 dxy;
      if (!eval_alpha(
              pxy,
              m,
              conics[g3 + 0u],
              conics[g3 + 1u],
              conics[g3 + 2u],
              opacities[g],
              mf,
              alpha,
              raw_alpha,
              power,
              dxy)) {
        continue;
      }
      if (T_forward <= mf.transmittance_threshold) {
        end_i = i;
        break;
      }
      apply_softmax_gs_forward(mf, depths[g], power, alpha, T_forward, accum_forward, past_depth, past_power);
      float w = T_forward * alpha;
      accum_forward += w * float3(colors[g3 + 0u], colors[g3 + 1u], colors[g3 + 2u]);
      float denom = max(1.0f - T_forward + w, mf.eps);
      past_depth = (past_depth * (1.0f - T_forward) + depths[g] * w) / denom;
      past_power = (past_power * (1.0f - T_forward) + power * w) / denom;
      T_forward *= (1.0f - alpha);
      if (T_forward <= mf.transmittance_threshold) {
        end_i = i + 1u;
        break;
      }
    }
  }

  float3 g_accum = go;
  float gT = valid ? dot(go, float3(mf.bg_r, mf.bg_g, mf.bg_b)) : 0.0f;
  float gD = 0.0f;
  float gP = 0.0f;

  for (int gi = int(count) - 1; gi >= 0; --gi) {
    uint global_i = uint(gi);
    uint g = overflow_sorted_ids[start + global_i];
    uint g3 = g * 3u;
    float l_gmx = 0.0f, l_gmy = 0.0f, l_ga = 0.0f, l_gb = 0.0f, l_gc = 0.0f;
    float l_gop = 0.0f, l_gdepth = 0.0f;
    float3 l_gcol = float3(0.0f);

    if (valid && global_i < end_i) {
      SoftmaxReplayState st = replay_softmax_to_device_index(
          means2d, conics, colors, opacities, depths, overflow_sorted_ids, start, global_i, pxy, mf);
      if (st.active) {
        float3 c = float3(colors[g3 + 0u], colors[g3 + 1u], colors[g3 + 2u]);
        float a = st.alpha_eff;
        float Tpre = st.T_pre;
        float w = st.contribution_weight;
        float denom = st.denom;

        float g_Tpre = gT * (1.0f - a);
        float g_alpha = gT * (-Tpre);

        float nD = st.past_depth0 * (1.0f - Tpre) + st.depth * w;
        float nP = st.past_power0 * (1.0f - Tpre) + st.power * w;
        float g_nD = gD / denom;
        float g_den = gD * (-nD / max(denom * denom, mf.eps));
        float g_nP = gP / denom;
        g_den += gP * (-nP / max(denom * denom, mf.eps));

        float g_past_depth0 = g_nD * (1.0f - Tpre);
        g_Tpre += g_nD * (-st.past_depth0);
        float g_depth_cur = g_nD * w;
        float g_w = g_nD * st.depth;

        float g_past_power0 = g_nP * (1.0f - Tpre);
        g_Tpre += g_nP * (-st.past_power0);
        float g_power_cur = g_nP * w;
        g_w += g_nP * st.power;

        g_Tpre -= g_den;
        g_w += g_den;

        float3 g_accum_pre = g_accum;
        g_w += dot(g_accum, c);
        l_gcol = g_accum * w;

        g_Tpre += g_w * a;
        g_alpha += g_w * Tpre;

        float g_a_input = 0.0f;
        if (st.has_softmax) {
          float g_T1 = g_Tpre;
          float3 g_accum1 = g_accum_pre;
          float g_alpha_eff = g_alpha;

          float3 g_accum0 = g_accum1 * st.color_scale;
          float g_color_scale = dot(g_accum1, st.accum0);
          float g_effective_past = -g_T1 + g_color_scale / max(st.past_absorbance, mf.eps);
          float g_past_absorbance = g_color_scale * (-st.effective_past / max(st.past_absorbance * st.past_absorbance, mf.eps));

          float g_scale = g_effective_past * st.effective_past0 + g_alpha_eff * st.alpha_soft0;
          float g_effective_past0 = g_effective_past * st.scale;
          float g_alpha_soft0 = g_alpha_eff * st.scale;
          float g_target_absorbance = 0.0f;
          if (st.pair_product > mf.eps) {
            float denom_scale = max(st.pair_sum + st.sqrt_disc, mf.eps);
            g_target_absorbance += g_scale * (2.0f / denom_scale);
            float g_denom_scale = g_scale * (-2.0f * st.target_absorbance / max(denom_scale * denom_scale, mf.eps));
            float g_pair_sum = g_denom_scale;
            float g_sqrt_disc = g_denom_scale;
            float g_disc = g_sqrt_disc * (0.5f / max(st.sqrt_disc, mf.eps));
            g_pair_sum += g_disc * (2.0f * st.pair_sum);
            g_target_absorbance += g_disc * (-4.0f * st.pair_product);
            float g_pair_product = g_disc * (-4.0f * st.target_absorbance);
            g_effective_past0 += g_pair_sum + g_pair_product * st.alpha_soft0;
            g_alpha_soft0 += g_pair_sum + g_pair_product * st.effective_past0;
          }

          float g_decay = g_effective_past0 * (st.tilde_past - st.past_absorbance)
              + g_alpha_soft0 * (st.tilde_cur - st.alpha_input);
          float g_tilde_past = g_effective_past0 * st.decay;
          g_past_absorbance += g_effective_past0 * (1.0f - st.decay);
          float g_tilde_cur = g_alpha_soft0 * st.decay;
          g_a_input += g_alpha_soft0 * (1.0f - st.decay);

          float depth_sign = sign(st.depth_diff);
          float g_depth_diff = g_decay * st.decay * (-max(mf.softmax_gamma, 0.0f) * depth_sign);
          g_depth_cur += g_depth_diff;
          g_past_depth0 -= g_depth_diff;

          float g_soft_cur = g_tilde_cur * st.target_absorbance / st.tilde_cur_denom;
          g_target_absorbance += g_tilde_cur * st.soft_cur / st.tilde_cur_denom;
          float g_tilde_cur_denom = g_tilde_cur * (-st.soft_cur * st.target_absorbance / max(st.tilde_cur_denom * st.tilde_cur_denom, mf.eps));
          g_soft_cur += g_tilde_cur_denom;
          float g_soft_past = g_tilde_cur_denom * st.T_orig;
          float g_T_orig = g_tilde_cur_denom * st.soft_past;

          g_soft_past += g_tilde_past * st.target_absorbance / st.soft_denom;
          g_target_absorbance += g_tilde_past * st.soft_past / st.soft_denom;
          float g_soft_denom = g_tilde_past * (-st.soft_past * st.target_absorbance / max(st.soft_denom * st.soft_denom, mf.eps));
          g_soft_past += g_soft_denom;
          g_soft_cur += g_soft_denom;

          g_T_orig -= g_target_absorbance;

          float g_w_cur = g_soft_cur * st.alpha_input;
          g_a_input += g_soft_cur * st.w_cur;
          g_w_cur -= g_soft_past * st.past_absorbance;
          g_past_absorbance += g_soft_past * (1.0f - st.w_cur);

          float g_z = g_w_cur * st.w_cur * (1.0f - st.w_cur);
          g_power_cur += g_z * mf.softmax_beta;
          g_past_power0 -= g_z * mf.softmax_beta;

          float g_T0 = g_T_orig * (1.0f - st.alpha_input);
          g_a_input += g_T_orig * (-st.T0);
          g_T0 -= g_past_absorbance;

          g_accum = g_accum0;
          gT = g_T0;
          gD = g_past_depth0;
          gP = g_past_power0;
        } else {
          g_accum = g_accum_pre;
          gT = g_Tpre;
          gD = g_past_depth0;
          gP = g_past_power0;
          g_a_input += g_alpha;
        }

        float gate = (st.raw_alpha < mf.max_alpha) ? 1.0f : 0.0f;
        float g_raw = g_a_input * gate;
        float g_power_from_alpha = g_raw * st.raw_alpha;
        float g_power_total = g_power_cur + g_power_from_alpha;
        l_ga = g_power_total * (-0.5f) * st.delta.x * st.delta.x;
        l_gb = g_power_total * (-1.0f) * st.delta.x * st.delta.y;
        l_gc = g_power_total * (-0.5f) * st.delta.y * st.delta.y;
        float qa = conics[g3 + 0u];
        float qb = conics[g3 + 1u];
        float qc = conics[g3 + 2u];
        float g_dx = g_power_total * (-(qa * st.delta.x + qb * st.delta.y));
        float g_dy = g_power_total * (-(qb * st.delta.x + qc * st.delta.y));
        l_gmx = -g_dx;
        l_gmy = -g_dy;
        l_gop = g_raw * (st.raw_alpha / max(opacities[g], mf.eps));
        l_gdepth = g_depth_cur;
      }
    }

    float4 s0 = simd_sum(float4(l_gmx, l_gmy, l_ga, l_gb));
    float4 s1 = simd_sum(float4(l_gc, l_gcol.x, l_gcol.y, l_gcol.z));
    float4 s2 = simd_sum(float4(l_gop, l_gdepth, 0.0f, 0.0f));
    if (simd_lane == 0u) {
      partial0[simd_group] = s0;
      partial1[simd_group] = s1;
      partial2[simd_group] = s2;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group == 0u) {
      float4 v0 = (simd_lane < GSP_SIMDGROUPS) ? partial0[simd_lane] : float4(0.0f);
      float4 v1 = (simd_lane < GSP_SIMDGROUPS) ? partial1[simd_lane] : float4(0.0f);
      float4 v2 = (simd_lane < GSP_SIMDGROUPS) ? partial2[simd_lane] : float4(0.0f);
      float4 t0 = simd_sum(v0);
      float4 t1 = simd_sum(v1);
      float4 t2 = simd_sum(v2);
      if (simd_lane == 0u) {
        atomic_fetch_add_explicit(&g_means2d[g * 2u + 0u], t0.x, memory_order_relaxed);
        atomic_fetch_add_explicit(&g_means2d[g * 2u + 1u], t0.y, memory_order_relaxed);
        atomic_fetch_add_explicit(&g_conics[g * 3u + 0u], t0.z, memory_order_relaxed);
        atomic_fetch_add_explicit(&g_conics[g * 3u + 1u], t0.w, memory_order_relaxed);
        atomic_fetch_add_explicit(&g_conics[g * 3u + 2u], t1.x, memory_order_relaxed);
        atomic_add3(g_colors, g, float3(t1.y, t1.z, t1.w));
        atomic_fetch_add_explicit(&g_opacities[g], t2.x, memory_order_relaxed);
        atomic_fetch_add_explicit(&g_depths[g], t2.y, memory_order_relaxed);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

kernel void tile_overflow_softmax_bounded_tape(
    const device uint* overflow_tile_ids [[buffer(0)]],
    const device int* overflow_tile_offsets [[buffer(1)]],
    const device uint* overflow_sorted_ids [[buffer(2)]],
    const device float2* means2d [[buffer(3)]],
    const device float* conics [[buffer(4)]],
    const device float* opacities [[buffer(5)]],
    const device float* depths [[buffer(6)]],
    constant MetaI32& mi [[buffer(7)]],
    constant MetaF32& mf [[buffer(8)]],
    device int* out_ids [[buffer(9)]],
    device float* out_weights [[buffer(10)]],
    device float* out_residual [[buffer(11)]],
    device float* out_final_alpha [[buffer(12)]],
    uint local_tile_idx [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]]) {
  uint tile_id = overflow_tile_ids[local_tile_idx];
  uint start = uint(overflow_tile_offsets[local_tile_idx]);
  uint end = uint(overflow_tile_offsets[local_tile_idx + 1]);
  uint count = end - start;
  if (count == 0u) return;

  uint batch, x, y;
  tile_pixel_from_tid(tile_id, tid, mi, batch, x, y);
  bool valid = (x < uint(mi.width) && y < uint(mi.height));
  if (!valid) return;
  uint px = tid % GSP_TILE_SIZE;
  uint py = tid / GSP_TILE_SIZE;
  uint tape_k = min(uint(mi.softmax_tape_k), GSP_TAPE_CAP);
  uint ids[GSP_TAPE_CAP];
  float weights[GSP_TAPE_CAP];
  init_bounded_tape(ids, weights);

  float T = 1.0f;
  float past_depth = 0.0f;
  float past_power = 0.0f;
  float2 p = float2(float(x) + 0.5f, float(y) + 0.5f);
  for (uint i = 0u; i < count; ++i) {
    if (T <= mf.transmittance_threshold) break;
    uint g = overflow_sorted_ids[start + i];
    uint g3 = g * 3u;
    float alpha, raw_alpha, power;
    float2 dxy;
    if (!eval_alpha(
            p,
            means2d[g],
            conics[g3 + 0u],
            conics[g3 + 1u],
            conics[g3 + 2u],
            opacities[g],
            mf,
            alpha,
            raw_alpha,
            power,
            dxy)) {
      continue;
    }
    if (softmax_gs_enabled(mi)) {
      float prefix_scale = apply_softmax_gs_tape_scalar(mf, depths[g], power, alpha, T, past_depth, past_power);
      scale_bounded_tape(ids, weights, tape_k, prefix_scale);
    }
    float w = T * alpha;
    insert_bounded_tape_weight(ids, weights, tape_k, g, w);
    float denom = max(1.0f - T + w, mf.eps);
    past_depth = (past_depth * (1.0f - T) + depths[g] * w) / denom;
    past_power = (past_power * (1.0f - T) + power * w) / denom;
    T *= (1.0f - alpha);
  }

  sort_bounded_tape_by_id(ids, weights, tape_k);
  uint pixel_base = ((local_tile_idx * GSP_TILE_SIZE + py) * GSP_TILE_SIZE + px);
  float final_alpha = 1.0f - T;
  float residual = max(final_alpha - selected_weight_sum(ids, weights, tape_k), 0.0f);
  for (uint slot = 0u; slot < GSP_TAPE_CAP; ++slot) {
    if (slot >= tape_k) break;
    uint out_idx = pixel_base * tape_k + slot;
    out_ids[out_idx] = (ids[slot] == 0xFFFFFFFFu) ? -1 : int(ids[slot]);
    out_weights[out_idx] = (ids[slot] == 0xFFFFFFFFu) ? 0.0f : weights[slot];
  }
  out_residual[pixel_base] = residual;
  out_final_alpha[pixel_base] = final_alpha;
}

kernel void tile_overflow_forward(
    const device uint* overflow_tile_ids [[buffer(0)]],
    const device int* overflow_tile_offsets [[buffer(1)]],
    const device uint* overflow_sorted_ids [[buffer(2)]],
    const device float2* means2d [[buffer(3)]],
    const device float* conics [[buffer(4)]],
    const device float* colors [[buffer(5)]],
    const device float* opacities [[buffer(6)]],
    const device float* depths [[buffer(7)]],
    constant MetaI32& mi [[buffer(8)]],
    constant MetaF32& mf [[buffer(9)]],
    device float* out_tiles [[buffer(10)]],
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
  threadgroup float sh_depths[GSP_CHUNK];
  threadgroup uint tg_alive[GSP_SIMDGROUPS];
  threadgroup uint sh_ids[GSP_CHUNK];
  uint batch, x, y;
  tile_pixel_from_tid(tile_id, tid, mi, batch, x, y);
  bool valid = (x < uint(mi.width) && y < uint(mi.height));
  uint px = tid % GSP_TILE_SIZE;
  uint py = tid / GSP_TILE_SIZE;
  float2 p = float2(float(x) + 0.5f, float(y) + 0.5f);
  float3 accum = float3(0.0f);
  float T = 1.0f;
  float past_depth = 0.0f;
  float past_power = 0.0f;
  for (uint chunk_start = 0u; chunk_start < count; chunk_start += GSP_CHUNK) {
    uint chunk_n = min(GSP_CHUNK, count - chunk_start);
    for (uint i = tid; i < chunk_n; i += GSP_THREADS) sh_ids[i] = overflow_sorted_ids[start + i + chunk_start];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    load_chunk_params_with_depths(
        means2d, conics, colors, opacities, depths, sh_ids, 0u, chunk_n, tid,
        sh_means, sh_conics, sh_colors, sh_opacities, sh_depths);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint alive_total = reduce_alive((valid && T > mf.transmittance_threshold) ? 1u : 0u, simd_lane, simd_group, tg_alive);
    if (alive_total == 0u) break;
    if (valid && T > mf.transmittance_threshold) {
      for (uint j = 0u; j < chunk_n; ++j) {
        float alpha, raw_alpha, power; float2 d;
        float2 m = sh_means[j];
        float3 q = load3_sh(sh_conics, j);
        if (!eval_alpha(p, m, q.x, q.y, q.z, sh_opacities[j], mf, alpha, raw_alpha, power, d)) continue;
        if (softmax_gs_enabled(mi)) {
          apply_softmax_gs_forward(mf, sh_depths[j], power, alpha, T, accum, past_depth, past_power);
        }
        float w = T * alpha;
        accum += w * load3_sh(sh_colors, j);
        float denom = max(1.0f - T + w, mf.eps);
        past_depth = (past_depth * (1.0f - T) + sh_depths[j] * w) / denom;
        past_power = (past_power * (1.0f - T) + power * w) / denom;
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
  uint batch, x, y;
  tile_pixel_from_tid(tile_id, tid, mi, batch, x, y);
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
    for (uint i = tid; i < chunk_n; i += GSP_THREADS) sh_ids[i] = overflow_sorted_ids[start + chunk_start + i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    load_chunk_params(means2d, conics, colors, opacities, sh_ids, 0u, chunk_n, tid, sh_means, sh_conics, sh_colors, sh_opacities);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint alive_total = reduce_alive((valid && T_final > mf.transmittance_threshold) ? 1u : 0u, simd_lane, simd_group, tg_alive);
    if (alive_total == 0u) break;
    if (valid && T_final > mf.transmittance_threshold) {
      for (uint j = 0u; j < chunk_n; ++j) {
        float alpha, raw_alpha, power; float2 d;
        float2 m = sh_means[j];
        float3 q = load3_sh(sh_conics, j);
        if (!eval_alpha(p, m, q.x, q.y, q.z, sh_opacities[j], mf, alpha, raw_alpha, power, d)) continue;
        T_final *= (1.0f - alpha);
        if (T_final <= mf.transmittance_threshold) { end_i = chunk_start + j + 1u; break; }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  float T_cur = T_final;
  float gT = valid ? dot(go, float3(mf.bg_r, mf.bg_g, mf.bg_b)) : 0.0f;
  // Keep barrier control uniform across the threadgroup. `end_i` is per pixel,
  // so using it as the loop bound makes saturated pixels take fewer barriers
  // than unsaturated pixels. Iterate over the tile count and mask each pixel
  // with `global_i < end_i` inside the loop.
  for (int chunk_end = int(count); chunk_end > 0; chunk_end -= int(GSP_CHUNK)) {
    int chunk_start_i = max(0, chunk_end - int(GSP_CHUNK));
    uint chunk_start = uint(chunk_start_i);
    uint chunk_n = uint(chunk_end - chunk_start_i);
    for (uint i = tid; i < chunk_n; i += GSP_THREADS) sh_ids[i] = overflow_sorted_ids[start + chunk_start + i];
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
      float l_gmx = 0.0f, l_gmy = 0.0f, l_ga = 0.0f, l_gb = 0.0f, l_gc = 0.0f, l_gop = 0.0f;
      float3 l_gcol = float3(0.0f);
      if (valid && global_i < end_i) {
        float alpha, raw_alpha, power; float2 d;
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
      if (simd_lane == 0u) { partial0[simd_group] = s0; partial1[simd_group] = s1; partial2[simd_group] = s2; }
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

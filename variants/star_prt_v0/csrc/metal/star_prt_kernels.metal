#include <metal_stdlib>

using namespace metal;

struct StarPRTMeta {
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
  int h_terms;
  int reserved0;

  float alpha_threshold;
  float transmittance_threshold;
  float bg_r;
  float bg_g;
  float bg_b;
  float depth_epsilon;
  float max_alpha;
  float reserved_f0;
};

// Intended PRT tensor contract:
// h_coeff: [N,H,3] homogeneous curve coefficients for h(tau)=(x*w, y*w, w)
// lambda_uv: [N,3] symmetric screen precision (uu, uv, vv)
// lambda_t: [N] temporal precision
// center_t: [N] center time used by both tau and temporal falloff
// opacity: [N], color: [N,3]
// output image: [F,H,W,3] float32
//
// These kernels are names and buffer-contract placeholders only. The C++ Metal
// entrypoints intentionally raise until real binning, sorting, compositing, and
// backward kernels land.

kernel void star_prt_bin_projective_rational_tubes(
    device const float* h_coeff [[buffer(0)]],
    device const float* lambda_uv [[buffer(1)]],
    device const float* lambda_t [[buffer(2)]],
    device const float* center_t [[buffer(3)]],
    device const float* opacity [[buffer(4)]],
    constant StarPRTMeta& meta [[buffer(5)]],
    uint gid [[thread_position_in_grid]]) {
  (void)h_coeff;
  (void)lambda_uv;
  (void)lambda_t;
  (void)center_t;
  (void)opacity;
  (void)meta;
  (void)gid;
}

kernel void star_prt_render_projective_rational_tubes(
    device const float* h_coeff [[buffer(0)]],
    device const float* lambda_uv [[buffer(1)]],
    device const float* lambda_t [[buffer(2)]],
    device const float* center_t [[buffer(3)]],
    device const float* opacity [[buffer(4)]],
    device const float* color [[buffer(5)]],
    device float* image [[buffer(6)]],
    constant StarPRTMeta& meta [[buffer(7)]],
    uint gid [[thread_position_in_grid]]) {
  (void)h_coeff;
  (void)lambda_uv;
  (void)lambda_t;
  (void)center_t;
  (void)opacity;
  (void)color;
  (void)image;
  (void)meta;
  (void)gid;
}

kernel void star_prt_render_compiled_curve_tubes(
    device const float* curve_uv_depth [[buffer(0)]],
    device const float* lambda_uv [[buffer(1)]],
    device const float* lambda_t [[buffer(2)]],
    device const float* center_t [[buffer(3)]],
    device const float* opacity [[buffer(4)]],
    device const float* color [[buffer(5)]],
    device float* image [[buffer(6)]],
    constant StarPRTMeta& meta [[buffer(7)]],
    uint gid [[thread_position_in_grid]]) {
  (void)curve_uv_depth;
  (void)lambda_uv;
  (void)lambda_t;
  (void)center_t;
  (void)opacity;
  (void)color;
  (void)image;
  (void)meta;
  (void)gid;
}

kernel void star_prt_compact_backward_projective_rational_tubes(
    device const float* h_coeff [[buffer(0)]],
    device const float* lambda_uv [[buffer(1)]],
    device const float* lambda_t [[buffer(2)]],
    device const float* center_t [[buffer(3)]],
    device const float* opacity [[buffer(4)]],
    device const float* color [[buffer(5)]],
    device const float* grad_image [[buffer(6)]],
    constant StarPRTMeta& meta [[buffer(7)]],
    uint gid [[thread_position_in_grid]]) {
  (void)h_coeff;
  (void)lambda_uv;
  (void)lambda_t;
  (void)center_t;
  (void)opacity;
  (void)color;
  (void)grad_image;
  (void)meta;
  (void)gid;
}

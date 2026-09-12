#import <Foundation/Foundation.h>

#include <ATen/ATen.h>
#include <ATen/native/mps/MetalShaderLibrary.h>
#include <torch/extension.h>
#include <torch/mps.h>

#include <algorithm>
#include <climits>
#include <cstdlib>
#include <mutex>
#include <string>

#include "shared/common.h"

namespace star_uvt {
namespace {

using at::native::mps::DynamicMetalShaderLibrary;
using at::native::mps::MetalKernelFunction;

struct ShaderConfig {
  int tile_x;
  int tile_y;
  int tile_t;
  int tile_capacity;
  int threads;
  int fixedpoint_scale;
  int split_fixedpoint_coarse_scale;
  int split_fixedpoint_fine_scale;
};

int env_int(const char* name, int default_value) {
  const char* raw = std::getenv(name);
  if (raw == nullptr || raw[0] == '\0') return default_value;
  return std::atoi(raw);
}

ShaderConfig& shader_config() {
  static ShaderConfig cfg = []() {
    ShaderConfig c;
    c.tile_x = env_int("STAR_UVT_TILE_X", 8);
    c.tile_y = env_int("STAR_UVT_TILE_Y", 8);
    c.tile_t = env_int("STAR_UVT_TILE_T", 2);
    c.tile_capacity = env_int("STAR_UVT_TILE_CAPACITY", 128);
    c.fixedpoint_scale = env_int("STAR_UVT_FIXEDPOINT_SCALE", 1000000);
    c.split_fixedpoint_coarse_scale = env_int("STAR_UVT_SPLIT_FIXEDPOINT_COARSE_SCALE", 100);
    c.split_fixedpoint_fine_scale = env_int("STAR_UVT_SPLIT_FIXEDPOINT_FINE_SCALE", 1000000);
    TORCH_CHECK(c.tile_x == 8 || c.tile_x == 16, "STAR_UVT_TILE_X must be 8 or 16");
    TORCH_CHECK(c.tile_y == 8 || c.tile_y == 16, "STAR_UVT_TILE_Y must be 8 or 16");
    TORCH_CHECK(c.tile_t == 1 || c.tile_t == 2 || c.tile_t == 4, "STAR_UVT_TILE_T must be 1, 2, or 4");
    TORCH_CHECK(c.tile_capacity == 32 || c.tile_capacity == 64 || c.tile_capacity == 128 || c.tile_capacity == 256,
                "STAR_UVT_TILE_CAPACITY must be 32, 64, 128, or 256");
    TORCH_CHECK(c.fixedpoint_scale > 0, "STAR_UVT_FIXEDPOINT_SCALE must be positive");
    TORCH_CHECK(c.split_fixedpoint_coarse_scale > 0, "STAR_UVT_SPLIT_FIXEDPOINT_COARSE_SCALE must be positive");
    TORCH_CHECK(c.split_fixedpoint_fine_scale > 0, "STAR_UVT_SPLIT_FIXEDPOINT_FINE_SCALE must be positive");
    c.threads = c.tile_x * c.tile_y * c.tile_t;
    TORCH_CHECK(c.threads <= 1024, "STAR-UVT threadgroup exceeds 1024 threads");
    return c;
  }();
  return cfg;
}

std::string load_shader_source() {
  auto& cfg = shader_config();
  NSString* metalPath = [[NSString stringWithUTF8String:__FILE__] stringByDeletingLastPathComponent];
  metalPath = [metalPath stringByAppendingPathComponent:@"star_uvt_kernels.metal"];
  NSError* err = nil;
  NSString* src = [NSString stringWithContentsOfFile:metalPath encoding:NSUTF8StringEncoding error:&err];
  TORCH_CHECK(src != nil, "Failed to read star_uvt_kernels.metal: ", err.localizedDescription.UTF8String);

  std::string preamble;
  preamble += "#define STAR_TILE_X " + std::to_string(cfg.tile_x) + "u\n";
  preamble += "#define STAR_TILE_Y " + std::to_string(cfg.tile_y) + "u\n";
  preamble += "#define STAR_TILE_T " + std::to_string(cfg.tile_t) + "u\n";
  preamble += "#define STAR_TILE_CAPACITY " + std::to_string(cfg.tile_capacity) + "u\n";
  preamble += "#define STAR_THREADS " + std::to_string(cfg.threads) + "u\n\n";
  preamble += "#define STAR_FIXEDPOINT_SCALE " + std::to_string(cfg.fixedpoint_scale) + ".0f\n\n";
  preamble += "#define STAR_SPLIT_FIXEDPOINT_COARSE_SCALE " +
              std::to_string(cfg.split_fixedpoint_coarse_scale) + ".0f\n";
  preamble += "#define STAR_SPLIT_FIXEDPOINT_FINE_SCALE " +
              std::to_string(cfg.split_fixedpoint_fine_scale) + ".0f\n\n";
  return preamble + std::string([src UTF8String]);
}

struct MetalKernels {
  std::shared_ptr<MetalKernelFunction> clear_tiles;
  std::shared_ptr<MetalKernelFunction> clear_direct_gradients;
  std::shared_ptr<MetalKernelFunction> clear_feature_direct_gradients;
  std::shared_ptr<MetalKernelFunction> clear_linear_colorizer_gradients;
  std::shared_ptr<MetalKernelFunction> clear_direct_gradients_i32;
  std::shared_ptr<MetalKernelFunction> fixedpoint_gradients_to_float;
  std::shared_ptr<MetalKernelFunction> split_fixedpoint_gradients_to_float;
  std::shared_ptr<MetalKernelFunction> bin_tubes;
  std::shared_ptr<MetalKernelFunction> bin_tubes_gated;
  std::shared_ptr<MetalKernelFunction> render_tiles;
  std::shared_ptr<MetalKernelFunction> render_tiles_gated;
  std::shared_ptr<MetalKernelFunction> render_projective_trace_tiles;
  std::shared_ptr<MetalKernelFunction> render_projective_trace_cell_tiles;
  std::shared_ptr<MetalKernelFunction> render_projective_trace_cell_interval_tiles;
  std::shared_ptr<MetalKernelFunction> render_projective_trace_family_interval_tiles;
  std::shared_ptr<MetalKernelFunction> render_projective_trace_cell_interval_rows;
  std::shared_ptr<MetalKernelFunction> direct_atomic_projective_trace_backward;
  std::shared_ptr<MetalKernelFunction> direct_atomic_projective_cell_interval_backward;
  std::shared_ptr<MetalKernelFunction> direct_atomic_projective_family_cell_interval_backward;
  std::shared_ptr<MetalKernelFunction> render_feature_tiles;
  std::shared_ptr<MetalKernelFunction> render_feature_sparse_pixels;
  std::shared_ptr<MetalKernelFunction> simple_backward_samples;
  std::shared_ptr<MetalKernelFunction> stable_backward_samples;
  std::shared_ptr<MetalKernelFunction> direct_atomic_backward;
  std::shared_ptr<MetalKernelFunction> direct_atomic_backward_gated;
  std::shared_ptr<MetalKernelFunction> direct_atomic_feature_backward;
  std::shared_ptr<MetalKernelFunction> direct_atomic_feature_sparse_pixels_backward;
  std::shared_ptr<MetalKernelFunction> direct_atomic_feature_sparse_hidden_sigmoid_mse_backward;
  std::shared_ptr<MetalKernelFunction> sparse_hidden_sigmoid_target_area_forward_sums;
  std::shared_ptr<MetalKernelFunction> direct_atomic_feature_sparse_hidden_target_area_backward;
  std::shared_ptr<MetalKernelFunction> direct_atomic_feature_linear_sigmoid_mse_backward;
  std::shared_ptr<MetalKernelFunction> linear_sigmoid_mse_handoff_prep;
  std::shared_ptr<MetalKernelFunction> direct_atomic_feature_hidden_sigmoid_mse_backward;
  std::shared_ptr<MetalKernelFunction> direct_atomic_feature_logit_handoff_backward;
  std::shared_ptr<MetalKernelFunction> direct_fixedpoint_backward;
  std::shared_ptr<MetalKernelFunction> direct_split_fixedpoint_backward;
  std::shared_ptr<MetalKernelFunction> tile_pair_atomic_backward;
  std::shared_ptr<MetalKernelFunction> tile_pair_fixedpoint_backward;
  std::shared_ptr<MetalKernelFunction> direct_serial_backward;
  std::shared_ptr<MetalKernelFunction> tile_pair_backward_samples;
  std::shared_ptr<MetalKernelFunction> tile_pair_backward_samples_compensated;
  std::shared_ptr<MetalKernelFunction> tile_pair_target_bounds_backward_samples;
  std::shared_ptr<MetalKernelFunction> tile_pair_suffix_backward_samples;
  std::shared_ptr<MetalKernelFunction> tile_pair_parallel_backward_samples;
  std::shared_ptr<MetalKernelFunction> tile_pair_grouped_backward_samples;
  std::shared_ptr<MetalKernelFunction> tile_pair_sharedsort_backward_samples;
  std::shared_ptr<MetalKernelFunction> tile_pair_scanline_backward_samples;
  std::shared_ptr<MetalKernelFunction> reduce_sample_bundle_scan;
  std::shared_ptr<MetalKernelFunction> reduce_sample_bundle_scan_compensated;
  std::shared_ptr<MetalKernelFunction> reduce_sample_bundle_sorted_segments;
  std::shared_ptr<MetalKernelFunction> reduce_tile_pair_bounds_scan;
  std::shared_ptr<MetalKernelFunction> reduce_tile_pair_bounds_scan_parallel;
  std::shared_ptr<MetalKernelFunction> projective_trace_eval;
  std::shared_ptr<MetalKernelFunction> projective_trace_family_eval;
  std::shared_ptr<MetalKernelFunction> projective_trace_family_backward;
};

MetalKernels& kernels() {
  static std::once_flag once;
  static std::unique_ptr<DynamicMetalShaderLibrary> lib;
  static MetalKernels out;
  std::call_once(once, []() {
    lib = std::make_unique<DynamicMetalShaderLibrary>(load_shader_source());
    out.clear_tiles = lib->getKernelFunction("clear_tiles");
    out.clear_direct_gradients = lib->getKernelFunction("clear_direct_gradients");
    out.clear_feature_direct_gradients = lib->getKernelFunction("clear_feature_direct_gradients");
    out.clear_linear_colorizer_gradients = lib->getKernelFunction("clear_linear_colorizer_gradients");
    out.clear_direct_gradients_i32 = lib->getKernelFunction("clear_direct_gradients_i32");
    out.fixedpoint_gradients_to_float = lib->getKernelFunction("fixedpoint_gradients_to_float");
    out.split_fixedpoint_gradients_to_float = lib->getKernelFunction("split_fixedpoint_gradients_to_float");
    out.bin_tubes = lib->getKernelFunction("bin_screen_tubes_to_uvt_tiles");
    out.bin_tubes_gated = lib->getKernelFunction("bin_screen_tubes_to_uvt_tiles_gated");
    out.render_tiles = lib->getKernelFunction("render_uvt_tiles");
    out.render_tiles_gated = lib->getKernelFunction("render_uvt_tiles_gated");
    out.render_projective_trace_tiles = lib->getKernelFunction("render_projective_trace_tiles");
    out.render_projective_trace_cell_tiles = lib->getKernelFunction("render_projective_trace_cell_tiles");
    out.render_projective_trace_cell_interval_tiles = lib->getKernelFunction("render_projective_trace_cell_interval_tiles");
    out.render_projective_trace_family_interval_tiles = lib->getKernelFunction("render_projective_trace_family_interval_tiles");
    out.render_projective_trace_cell_interval_rows = lib->getKernelFunction("render_projective_trace_cell_interval_rows");
    out.direct_atomic_projective_trace_backward = lib->getKernelFunction("direct_atomic_projective_trace_backward");
    out.direct_atomic_projective_cell_interval_backward = lib->getKernelFunction("direct_atomic_projective_cell_interval_backward");
    out.direct_atomic_projective_family_cell_interval_backward =
        lib->getKernelFunction("direct_atomic_projective_family_cell_interval_backward");
    out.render_feature_tiles = lib->getKernelFunction("render_uvt_feature_tiles");
    out.render_feature_sparse_pixels = lib->getKernelFunction("render_uvt_feature_sparse_pixels");
    out.simple_backward_samples = lib->getKernelFunction("simple_backward_samples");
    out.stable_backward_samples = lib->getKernelFunction("stable_backward_samples");
    out.direct_atomic_backward = lib->getKernelFunction("direct_atomic_backward");
    out.direct_atomic_backward_gated = lib->getKernelFunction("direct_atomic_backward_gated");
    out.direct_atomic_feature_backward = lib->getKernelFunction("direct_atomic_feature_backward");
    out.direct_atomic_feature_sparse_pixels_backward = lib->getKernelFunction("direct_atomic_feature_sparse_pixels_backward");
    out.direct_atomic_feature_sparse_hidden_sigmoid_mse_backward =
        lib->getKernelFunction("direct_atomic_feature_sparse_hidden_sigmoid_mse_backward");
    out.sparse_hidden_sigmoid_target_area_forward_sums =
        lib->getKernelFunction("sparse_hidden_sigmoid_target_area_forward_sums");
    out.direct_atomic_feature_sparse_hidden_target_area_backward =
        lib->getKernelFunction("direct_atomic_feature_sparse_hidden_target_area_backward");
    out.direct_atomic_feature_linear_sigmoid_mse_backward = lib->getKernelFunction("direct_atomic_feature_linear_sigmoid_mse_backward");
    out.linear_sigmoid_mse_handoff_prep = lib->getKernelFunction("linear_sigmoid_mse_handoff_prep");
    out.direct_atomic_feature_hidden_sigmoid_mse_backward = lib->getKernelFunction("direct_atomic_feature_hidden_sigmoid_mse_backward");
    out.direct_atomic_feature_logit_handoff_backward = lib->getKernelFunction("direct_atomic_feature_logit_handoff_backward");
    out.direct_fixedpoint_backward = lib->getKernelFunction("direct_fixedpoint_backward");
    out.direct_split_fixedpoint_backward = lib->getKernelFunction("direct_split_fixedpoint_backward");
    out.tile_pair_atomic_backward = lib->getKernelFunction("tile_pair_atomic_backward");
    out.tile_pair_fixedpoint_backward = lib->getKernelFunction("tile_pair_fixedpoint_backward");
    out.direct_serial_backward = lib->getKernelFunction("direct_serial_backward");
    out.tile_pair_backward_samples = lib->getKernelFunction("tile_pair_backward_samples");
    out.tile_pair_backward_samples_compensated = lib->getKernelFunction("tile_pair_backward_samples_compensated");
    out.tile_pair_target_bounds_backward_samples = lib->getKernelFunction("tile_pair_target_bounds_backward_samples");
    out.tile_pair_suffix_backward_samples = lib->getKernelFunction("tile_pair_suffix_backward_samples");
    out.tile_pair_parallel_backward_samples = lib->getKernelFunction("tile_pair_parallel_backward_samples");
    out.tile_pair_grouped_backward_samples = lib->getKernelFunction("tile_pair_grouped_backward_samples");
    out.tile_pair_sharedsort_backward_samples = lib->getKernelFunction("tile_pair_sharedsort_backward_samples");
    out.tile_pair_scanline_backward_samples = lib->getKernelFunction("tile_pair_scanline_backward_samples");
    out.reduce_sample_bundle_scan = lib->getKernelFunction("reduce_sample_bundle_scan");
    out.reduce_sample_bundle_scan_compensated = lib->getKernelFunction("reduce_sample_bundle_scan_compensated");
    out.reduce_sample_bundle_sorted_segments = lib->getKernelFunction("reduce_sample_bundle_sorted_segments");
    out.reduce_tile_pair_bounds_scan = lib->getKernelFunction("reduce_tile_pair_bounds_scan");
    out.reduce_tile_pair_bounds_scan_parallel = lib->getKernelFunction("reduce_tile_pair_bounds_scan_parallel");
    out.projective_trace_eval = lib->getKernelFunction("projective_trace_eval");
    out.projective_trace_family_eval = lib->getKernelFunction("projective_trace_family_eval");
    out.projective_trace_family_backward = lib->getKernelFunction("projective_trace_family_backward");
  });
  return out;
}

template <typename Fn>
void launch(std::shared_ptr<MetalKernelFunction> fn, Fn&& body) {
  fn->runCommandBlock([&]() {
    fn->startEncoding();
    body(*fn);
  });
}

void check_float_mps_2d(const torch::Tensor& t, const char* name, int64_t cols) {
  TORCH_CHECK(t.device().is_mps(), name, " must be on MPS");
  TORCH_CHECK(t.scalar_type() == torch::kFloat32, name, " must be float32");
  TORCH_CHECK(t.dim() == 2 && t.size(1) == cols, name, " must have shape [N,", cols, "]");
  TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

void check_float_mps_3d(const torch::Tensor& t, const char* name, int64_t dim1) {
  TORCH_CHECK(t.device().is_mps(), name, " must be on MPS");
  TORCH_CHECK(t.scalar_type() == torch::kFloat32, name, " must be float32");
  TORCH_CHECK(t.dim() == 3 && t.size(1) == dim1, name, " must have shape [N,", dim1, ",B]");
  TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

void check_float_mps_1d(const torch::Tensor& t, const char* name) {
  TORCH_CHECK(t.device().is_mps(), name, " must be on MPS");
  TORCH_CHECK(t.scalar_type() == torch::kFloat32, name, " must be float32");
  TORCH_CHECK(t.dim() == 1, name, " must have shape [N]");
  TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

void check_int_mps_1d(const torch::Tensor& t, const char* name) {
  TORCH_CHECK(t.device().is_mps(), name, " must be on MPS");
  TORCH_CHECK(t.scalar_type() == torch::kInt32, name, " must be int32");
  TORCH_CHECK(t.dim() == 1, name, " must have shape [N]");
  TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

void check_meta(const ParsedMeta& meta, int64_t n, const ShaderConfig& sc) {
  TORCH_CHECK(meta.height > 0 && meta.width > 0 && meta.frames > 0, "height, width, and frames must be positive");
  TORCH_CHECK(meta.tile_x == sc.tile_x && meta.tile_y == sc.tile_y && meta.tile_t == sc.tile_t,
              "meta tile shape must match STAR_UVT_TILE_* shader constants");
  TORCH_CHECK(meta.tile_capacity == sc.tile_capacity, "meta tile_capacity must match STAR_UVT_TILE_CAPACITY");
  TORCH_CHECK(meta.tube_count == n, "meta tube_count mismatch");
  TORCH_CHECK(meta.tiles_x == (meta.width + meta.tile_x - 1) / meta.tile_x, "tiles_x mismatch");
  TORCH_CHECK(meta.tiles_y == (meta.height + meta.tile_y - 1) / meta.tile_y, "tiles_y mismatch");
  TORCH_CHECK(meta.tiles_t == (meta.frames + meta.tile_t - 1) / meta.tile_t, "tiles_t mismatch");
  TORCH_CHECK(meta.tile_count == meta.tiles_x * meta.tiles_y * meta.tiles_t, "tile_count mismatch");
  TORCH_CHECK(
      meta.alpha_mode == 0.0f || meta.alpha_mode == 1.0f,
      "alpha_mode metadata must be 0 (peak_splat) or 1 (beer_lambert)");
}

void check_projective_interval_meta(const ParsedMeta& meta, int64_t n, const ShaderConfig& sc) {
  TORCH_CHECK(meta.height > 0 && meta.width > 0 && meta.frames > 0, "height, width, and frames must be positive");
  TORCH_CHECK(meta.tile_x == sc.tile_x && meta.tile_y == sc.tile_y,
              "meta tile_x/tile_y must match STAR_UVT_TILE_X/Y shader constants");
  TORCH_CHECK(meta.tile_capacity == sc.tile_capacity, "meta tile_capacity must match STAR_UVT_TILE_CAPACITY");
  TORCH_CHECK(meta.tube_count == n, "meta tube_count mismatch");
  TORCH_CHECK(meta.tiles_x == (meta.width + meta.tile_x - 1) / meta.tile_x, "tiles_x mismatch");
  TORCH_CHECK(meta.tiles_y == (meta.height + meta.tile_y - 1) / meta.tile_y, "tiles_y mismatch");
  TORCH_CHECK(meta.tiles_t == 1, "projective interval renderer expects one spatial tile layer");
  TORCH_CHECK(meta.tile_t == meta.frames, "projective interval renderer expects tile_t == frames");
  TORCH_CHECK(meta.tile_count == meta.tiles_x * meta.tiles_y, "projective interval tile_count mismatch");
  TORCH_CHECK(
      meta.alpha_mode == 0.0f || meta.alpha_mode == 1.0f,
      "alpha_mode metadata must be 0 (peak_splat) or 1 (beer_lambert)");
}

void check_feature_meta(const ParsedMeta& meta, int64_t n, const ShaderConfig& sc, int64_t feature_dim) {
  check_meta(meta, n, sc);
  TORCH_CHECK(meta.reserved0 == feature_dim, "meta feature_dim mismatch");
  TORCH_CHECK(feature_dim > 0 && feature_dim <= 128, "feature_dim must be in 1..128 for experimental STAR UVT features");
}

}  // namespace

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> metal_render_uvt(
    const torch::Tensor& ma,
    const torch::Tensor& q_uvt,
    const torch::Tensor& depth0,
    const torch::Tensor& depth_beta,
    const torch::Tensor& opacity,
    const torch::Tensor& color,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32) {
  check_float_mps_2d(ma, "ma", 3);
  check_float_mps_2d(q_uvt, "q_uvt", 6);
  check_float_mps_1d(depth0, "depth0");
  check_float_mps_2d(depth_beta, "depth_beta", 3);
  check_float_mps_1d(opacity, "opacity");
  check_float_mps_2d(color, "color", 3);
  TORCH_CHECK(ma.size(0) == q_uvt.size(0) && ma.size(0) == depth0.size(0) && ma.size(0) == depth_beta.size(0) &&
                  ma.size(0) == opacity.size(0) && ma.size(0) == color.size(0),
              "all tube inputs must agree on N");

  auto meta = parse_meta(meta_i32, meta_f32);
  auto& sc = shader_config();
  check_meta(meta, ma.size(0), sc);
  auto& k = kernels();

  auto opts_f = ma.options().dtype(torch::kFloat32);
  auto opts_i32 = ma.options().dtype(torch::kInt32);

  auto out = torch::empty({meta.frames, meta.height, meta.width, 3}, opts_f);
  auto tile_counts = torch::empty({meta.tile_count}, opts_i32);
  auto tile_overflow = torch::empty({meta.tile_count}, opts_i32);
  auto tile_unstable = torch::empty({meta.tile_count}, opts_i32);
  auto tile_tube_ids = torch::empty({meta.tile_count * meta.tile_capacity}, opts_i32);
  auto tile_depths = torch::empty({meta.tile_count * meta.tile_capacity}, opts_f);

  launch(k.clear_tiles, [&](MetalKernelFunction& fn) {
    fn.setArg(0, tile_counts);
    fn.setArg(1, tile_overflow);
    fn.setArg(2, tile_unstable);
    fn.setArg(3, meta_i32);
    fn.dispatch((uint64_t)meta.tile_count, 256);
  });

  launch(k.bin_tubes, [&](MetalKernelFunction& fn) {
    fn.setArg(0, ma);
    fn.setArg(1, q_uvt);
    fn.setArg(2, depth0);
    fn.setArg(3, depth_beta);
    fn.setArg(4, opacity);
    fn.setArg(5, meta_i32);
    fn.setArg(6, meta_f32);
    fn.setArg(7, tile_counts);
    fn.setArg(8, tile_tube_ids);
    fn.setArg(9, tile_depths);
    fn.setArg(10, tile_overflow);
    fn.dispatch((uint64_t)meta.tube_count, 256);
  });

  launch(k.render_tiles, [&](MetalKernelFunction& fn) {
    fn.setArg(0, ma);
    fn.setArg(1, q_uvt);
    fn.setArg(2, depth0);
    fn.setArg(3, depth_beta);
    fn.setArg(4, opacity);
    fn.setArg(5, color);
    fn.setArg(6, meta_i32);
    fn.setArg(7, meta_f32);
    fn.setArg(8, tile_counts);
    fn.setArg(9, tile_tube_ids);
    fn.setArg(10, tile_depths);
    fn.setArg(11, tile_unstable);
    fn.setArg(12, out);
    fn.dispatch((uint64_t)meta.tile_count * (uint64_t)sc.threads, (uint64_t)sc.threads);
  });

  return std::make_tuple(out, tile_counts, tile_overflow, tile_unstable);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> metal_render_uvt_gated(
    const torch::Tensor& ma,
    const torch::Tensor& q_uvt,
    const torch::Tensor& depth0,
    const torch::Tensor& depth_beta,
    const torch::Tensor& opacity,
    const torch::Tensor& color,
    const torch::Tensor& active_start,
    const torch::Tensor& active_stop,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32) {
  check_float_mps_2d(ma, "ma", 3);
  check_float_mps_2d(q_uvt, "q_uvt", 6);
  check_float_mps_1d(depth0, "depth0");
  check_float_mps_2d(depth_beta, "depth_beta", 3);
  check_float_mps_1d(opacity, "opacity");
  check_float_mps_2d(color, "color", 3);
  check_int_mps_1d(active_start, "active_start");
  check_int_mps_1d(active_stop, "active_stop");
  TORCH_CHECK(ma.size(0) == q_uvt.size(0) && ma.size(0) == depth0.size(0) && ma.size(0) == depth_beta.size(0) &&
                  ma.size(0) == opacity.size(0) && ma.size(0) == color.size(0) &&
                  ma.size(0) == active_start.size(0) && ma.size(0) == active_stop.size(0),
              "all gated tube inputs must agree on N");

  auto meta = parse_meta(meta_i32, meta_f32);
  auto& sc = shader_config();
  check_meta(meta, ma.size(0), sc);
  auto& k = kernels();

  auto opts_f = ma.options().dtype(torch::kFloat32);
  auto opts_i32 = ma.options().dtype(torch::kInt32);

  auto out = torch::empty({meta.frames, meta.height, meta.width, 3}, opts_f);
  auto tile_counts = torch::empty({meta.tile_count}, opts_i32);
  auto tile_overflow = torch::empty({meta.tile_count}, opts_i32);
  auto tile_unstable = torch::empty({meta.tile_count}, opts_i32);
  auto tile_tube_ids = torch::empty({meta.tile_count * meta.tile_capacity}, opts_i32);
  auto tile_depths = torch::empty({meta.tile_count * meta.tile_capacity}, opts_f);

  launch(k.clear_tiles, [&](MetalKernelFunction& fn) {
    fn.setArg(0, tile_counts);
    fn.setArg(1, tile_overflow);
    fn.setArg(2, tile_unstable);
    fn.setArg(3, meta_i32);
    fn.dispatch((uint64_t)meta.tile_count, 256);
  });

  launch(k.bin_tubes_gated, [&](MetalKernelFunction& fn) {
    fn.setArg(0, ma);
    fn.setArg(1, q_uvt);
    fn.setArg(2, depth0);
    fn.setArg(3, depth_beta);
    fn.setArg(4, opacity);
    fn.setArg(5, meta_i32);
    fn.setArg(6, meta_f32);
    fn.setArg(7, tile_counts);
    fn.setArg(8, tile_tube_ids);
    fn.setArg(9, tile_depths);
    fn.setArg(10, tile_overflow);
    fn.setArg(11, active_start);
    fn.setArg(12, active_stop);
    fn.dispatch((uint64_t)meta.tube_count, 256);
  });

  launch(k.render_tiles_gated, [&](MetalKernelFunction& fn) {
    fn.setArg(0, ma);
    fn.setArg(1, q_uvt);
    fn.setArg(2, depth0);
    fn.setArg(3, depth_beta);
    fn.setArg(4, opacity);
    fn.setArg(5, color);
    fn.setArg(6, meta_i32);
    fn.setArg(7, meta_f32);
    fn.setArg(8, tile_counts);
    fn.setArg(9, tile_tube_ids);
    fn.setArg(10, tile_depths);
    fn.setArg(11, tile_unstable);
    fn.setArg(12, out);
    fn.setArg(13, active_start);
    fn.setArg(14, active_stop);
    fn.dispatch((uint64_t)meta.tile_count * (uint64_t)sc.threads, (uint64_t)sc.threads);
  });

  return std::make_tuple(out, tile_counts, tile_overflow, tile_unstable);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> metal_render_uvt_features(
    const torch::Tensor& ma,
    const torch::Tensor& q_uvt,
    const torch::Tensor& depth0,
    const torch::Tensor& depth_beta,
    const torch::Tensor& opacity,
    const torch::Tensor& feature,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32) {
  check_float_mps_2d(ma, "ma", 3);
  check_float_mps_2d(q_uvt, "q_uvt", 6);
  check_float_mps_1d(depth0, "depth0");
  check_float_mps_2d(depth_beta, "depth_beta", 3);
  check_float_mps_1d(opacity, "opacity");
  TORCH_CHECK(feature.device().is_mps(), "feature must be on MPS");
  TORCH_CHECK(feature.scalar_type() == torch::kFloat32, "feature must be float32");
  TORCH_CHECK(feature.dim() == 2, "feature must have shape [N,F]");
  TORCH_CHECK(feature.is_contiguous(), "feature must be contiguous");
  TORCH_CHECK(ma.size(0) == q_uvt.size(0) && ma.size(0) == depth0.size(0) && ma.size(0) == depth_beta.size(0) &&
                  ma.size(0) == opacity.size(0) && ma.size(0) == feature.size(0),
              "all tube inputs must agree on N");

  auto meta = parse_meta(meta_i32, meta_f32);
  auto& sc = shader_config();
  check_feature_meta(meta, ma.size(0), sc, feature.size(1));
  auto& k = kernels();

  auto opts_f = ma.options().dtype(torch::kFloat32);
  auto opts_i32 = ma.options().dtype(torch::kInt32);

  auto out_feature = torch::empty({meta.frames, meta.height, meta.width, meta.reserved0}, opts_f);
  auto out_alpha = torch::empty({meta.frames, meta.height, meta.width}, opts_f);
  auto tile_counts = torch::empty({meta.tile_count}, opts_i32);
  auto tile_overflow = torch::empty({meta.tile_count}, opts_i32);
  auto tile_unstable = torch::empty({meta.tile_count}, opts_i32);
  auto tile_tube_ids = torch::empty({meta.tile_count * meta.tile_capacity}, opts_i32);
  auto tile_depths = torch::empty({meta.tile_count * meta.tile_capacity}, opts_f);

  launch(k.clear_tiles, [&](MetalKernelFunction& fn) {
    fn.setArg(0, tile_counts);
    fn.setArg(1, tile_overflow);
    fn.setArg(2, tile_unstable);
    fn.setArg(3, meta_i32);
    fn.dispatch((uint64_t)meta.tile_count, 256);
  });

  launch(k.bin_tubes, [&](MetalKernelFunction& fn) {
    fn.setArg(0, ma);
    fn.setArg(1, q_uvt);
    fn.setArg(2, depth0);
    fn.setArg(3, depth_beta);
    fn.setArg(4, opacity);
    fn.setArg(5, meta_i32);
    fn.setArg(6, meta_f32);
    fn.setArg(7, tile_counts);
    fn.setArg(8, tile_tube_ids);
    fn.setArg(9, tile_depths);
    fn.setArg(10, tile_overflow);
    fn.dispatch((uint64_t)meta.tube_count, 256);
  });

  launch(k.render_feature_tiles, [&](MetalKernelFunction& fn) {
    fn.setArg(0, ma);
    fn.setArg(1, q_uvt);
    fn.setArg(2, depth0);
    fn.setArg(3, depth_beta);
    fn.setArg(4, opacity);
    fn.setArg(5, feature);
    fn.setArg(6, meta_i32);
    fn.setArg(7, meta_f32);
    fn.setArg(8, tile_counts);
    fn.setArg(9, tile_tube_ids);
    fn.setArg(10, tile_depths);
    fn.setArg(11, tile_unstable);
    fn.setArg(12, out_feature);
    fn.setArg(13, out_alpha);
    fn.dispatch((uint64_t)meta.tile_count * (uint64_t)sc.threads, (uint64_t)sc.threads);
  });

  return std::make_tuple(out_feature, out_alpha, tile_counts, tile_overflow, tile_unstable);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> metal_render_uvt_features_with_bins(
    const torch::Tensor& ma,
    const torch::Tensor& q_uvt,
    const torch::Tensor& depth0,
    const torch::Tensor& depth_beta,
    const torch::Tensor& opacity,
    const torch::Tensor& feature,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32) {
  check_float_mps_2d(ma, "ma", 3);
  check_float_mps_2d(q_uvt, "q_uvt", 6);
  check_float_mps_1d(depth0, "depth0");
  check_float_mps_2d(depth_beta, "depth_beta", 3);
  check_float_mps_1d(opacity, "opacity");
  TORCH_CHECK(feature.device().is_mps(), "feature must be on MPS");
  TORCH_CHECK(feature.scalar_type() == torch::kFloat32, "feature must be float32");
  TORCH_CHECK(feature.dim() == 2, "feature must have shape [N,F]");
  TORCH_CHECK(feature.is_contiguous(), "feature must be contiguous");
  TORCH_CHECK(ma.size(0) == q_uvt.size(0) && ma.size(0) == depth0.size(0) && ma.size(0) == depth_beta.size(0) &&
                  ma.size(0) == opacity.size(0) && ma.size(0) == feature.size(0),
              "all tube inputs must agree on N");

  auto meta = parse_meta(meta_i32, meta_f32);
  auto& sc = shader_config();
  check_feature_meta(meta, ma.size(0), sc, feature.size(1));
  auto& k = kernels();

  auto opts_f = ma.options().dtype(torch::kFloat32);
  auto opts_i32 = ma.options().dtype(torch::kInt32);

  auto out_feature = torch::empty({meta.frames, meta.height, meta.width, meta.reserved0}, opts_f);
  auto out_alpha = torch::empty({meta.frames, meta.height, meta.width}, opts_f);
  auto tile_counts = torch::empty({meta.tile_count}, opts_i32);
  auto tile_overflow = torch::empty({meta.tile_count}, opts_i32);
  auto tile_unstable = torch::empty({meta.tile_count}, opts_i32);
  auto tile_tube_ids = torch::empty({meta.tile_count * meta.tile_capacity}, opts_i32);
  auto tile_depths = torch::empty({meta.tile_count * meta.tile_capacity}, opts_f);

  launch(k.clear_tiles, [&](MetalKernelFunction& fn) {
    fn.setArg(0, tile_counts);
    fn.setArg(1, tile_overflow);
    fn.setArg(2, tile_unstable);
    fn.setArg(3, meta_i32);
    fn.dispatch((uint64_t)meta.tile_count, 256);
  });

  launch(k.bin_tubes, [&](MetalKernelFunction& fn) {
    fn.setArg(0, ma);
    fn.setArg(1, q_uvt);
    fn.setArg(2, depth0);
    fn.setArg(3, depth_beta);
    fn.setArg(4, opacity);
    fn.setArg(5, meta_i32);
    fn.setArg(6, meta_f32);
    fn.setArg(7, tile_counts);
    fn.setArg(8, tile_tube_ids);
    fn.setArg(9, tile_depths);
    fn.setArg(10, tile_overflow);
    fn.dispatch((uint64_t)meta.tube_count, 256);
  });

  launch(k.render_feature_tiles, [&](MetalKernelFunction& fn) {
    fn.setArg(0, ma);
    fn.setArg(1, q_uvt);
    fn.setArg(2, depth0);
    fn.setArg(3, depth_beta);
    fn.setArg(4, opacity);
    fn.setArg(5, feature);
    fn.setArg(6, meta_i32);
    fn.setArg(7, meta_f32);
    fn.setArg(8, tile_counts);
    fn.setArg(9, tile_tube_ids);
    fn.setArg(10, tile_depths);
    fn.setArg(11, tile_unstable);
    fn.setArg(12, out_feature);
    fn.setArg(13, out_alpha);
    fn.dispatch((uint64_t)meta.tile_count * (uint64_t)sc.threads, (uint64_t)sc.threads);
  });

  return std::make_tuple(out_feature, out_alpha, tile_counts, tile_overflow, tile_unstable, tile_tube_ids, tile_depths);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
metal_render_feature_sparse_pixels_with_bins(
    const torch::Tensor& ma,
    const torch::Tensor& q_uvt,
    const torch::Tensor& depth0,
    const torch::Tensor& depth_beta,
    const torch::Tensor& opacity,
    const torch::Tensor& feature,
    const torch::Tensor& pixel_ids,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32) {
  check_float_mps_2d(ma, "ma", 3);
  check_float_mps_2d(q_uvt, "q_uvt", 6);
  check_float_mps_1d(depth0, "depth0");
  check_float_mps_2d(depth_beta, "depth_beta", 3);
  check_float_mps_1d(opacity, "opacity");
  TORCH_CHECK(feature.device().is_mps(), "feature must be on MPS");
  TORCH_CHECK(feature.scalar_type() == torch::kFloat32, "feature must be float32");
  TORCH_CHECK(feature.dim() == 2, "feature must have shape [N,F]");
  TORCH_CHECK(feature.is_contiguous(), "feature must be contiguous");
  check_int_mps_1d(pixel_ids, "pixel_ids");
  TORCH_CHECK(ma.size(0) == q_uvt.size(0) && ma.size(0) == depth0.size(0) && ma.size(0) == depth_beta.size(0) &&
                  ma.size(0) == opacity.size(0) && ma.size(0) == feature.size(0),
              "all tube inputs must agree on N");

  auto meta = parse_meta(meta_i32, meta_f32);
  auto& sc = shader_config();
  check_feature_meta(meta, ma.size(0), sc, feature.size(1));
  auto& k = kernels();

  auto opts_f = ma.options().dtype(torch::kFloat32);
  auto opts_i32 = ma.options().dtype(torch::kInt32);
  auto feature_values = torch::empty({pixel_ids.numel(), meta.reserved0}, opts_f);
  auto alpha_values = torch::empty({pixel_ids.numel()}, opts_f);
  auto tile_counts = torch::empty({meta.tile_count}, opts_i32);
  auto tile_overflow = torch::empty({meta.tile_count}, opts_i32);
  auto tile_unstable = torch::empty({meta.tile_count}, opts_i32);
  auto tile_tube_ids = torch::empty({meta.tile_count * meta.tile_capacity}, opts_i32);
  auto tile_depths = torch::empty({meta.tile_count * meta.tile_capacity}, opts_f);

  launch(k.clear_tiles, [&](MetalKernelFunction& fn) {
    fn.setArg(0, tile_counts);
    fn.setArg(1, tile_overflow);
    fn.setArg(2, tile_unstable);
    fn.setArg(3, meta_i32);
    fn.dispatch((uint64_t)meta.tile_count, 256);
  });

  launch(k.bin_tubes, [&](MetalKernelFunction& fn) {
    fn.setArg(0, ma);
    fn.setArg(1, q_uvt);
    fn.setArg(2, depth0);
    fn.setArg(3, depth_beta);
    fn.setArg(4, opacity);
    fn.setArg(5, meta_i32);
    fn.setArg(6, meta_f32);
    fn.setArg(7, tile_counts);
    fn.setArg(8, tile_tube_ids);
    fn.setArg(9, tile_depths);
    fn.setArg(10, tile_overflow);
    fn.dispatch((uint64_t)meta.tube_count, 256);
  });

  if (pixel_ids.numel() > 0) {
    launch(k.render_feature_sparse_pixels, [&](MetalKernelFunction& fn) {
      fn.setArg(0, ma);
      fn.setArg(1, q_uvt);
      fn.setArg(2, depth0);
      fn.setArg(3, depth_beta);
      fn.setArg(4, opacity);
      fn.setArg(5, feature);
      fn.setArg(6, pixel_ids);
      fn.setArg(7, meta_i32);
      fn.setArg(8, meta_f32);
      fn.setArg(9, tile_counts);
      fn.setArg(10, tile_tube_ids);
      fn.setArg(11, tile_depths);
      fn.setArg(12, tile_unstable);
      fn.setArg(13, feature_values);
      fn.setArg(14, alpha_values);
      fn.dispatch((uint64_t)pixel_ids.numel(), 256);
    });
  }

  return std::make_tuple(feature_values, alpha_values, tile_counts, tile_overflow, tile_unstable, tile_tube_ids, tile_depths);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
metal_bin_feature_tubes(
    const torch::Tensor& ma,
    const torch::Tensor& q_uvt,
    const torch::Tensor& depth0,
    const torch::Tensor& depth_beta,
    const torch::Tensor& opacity,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32) {
  check_float_mps_2d(ma, "ma", 3);
  check_float_mps_2d(q_uvt, "q_uvt", 6);
  check_float_mps_1d(depth0, "depth0");
  check_float_mps_2d(depth_beta, "depth_beta", 3);
  check_float_mps_1d(opacity, "opacity");
  TORCH_CHECK(ma.size(0) == q_uvt.size(0) && ma.size(0) == depth0.size(0) && ma.size(0) == depth_beta.size(0) &&
                  ma.size(0) == opacity.size(0),
              "all tube inputs must agree on N");

  auto meta = parse_meta(meta_i32, meta_f32);
  auto& sc = shader_config();
  check_meta(meta, ma.size(0), sc);
  auto& k = kernels();

  auto opts_f = ma.options().dtype(torch::kFloat32);
  auto opts_i32 = ma.options().dtype(torch::kInt32);
  auto tile_counts = torch::empty({meta.tile_count}, opts_i32);
  auto tile_overflow = torch::empty({meta.tile_count}, opts_i32);
  auto tile_unstable = torch::empty({meta.tile_count}, opts_i32);
  auto tile_tube_ids = torch::empty({meta.tile_count * meta.tile_capacity}, opts_i32);
  auto tile_depths = torch::empty({meta.tile_count * meta.tile_capacity}, opts_f);

  launch(k.clear_tiles, [&](MetalKernelFunction& fn) {
    fn.setArg(0, tile_counts);
    fn.setArg(1, tile_overflow);
    fn.setArg(2, tile_unstable);
    fn.setArg(3, meta_i32);
    fn.dispatch((uint64_t)meta.tile_count, 256);
  });

  launch(k.bin_tubes, [&](MetalKernelFunction& fn) {
    fn.setArg(0, ma);
    fn.setArg(1, q_uvt);
    fn.setArg(2, depth0);
    fn.setArg(3, depth_beta);
    fn.setArg(4, opacity);
    fn.setArg(5, meta_i32);
    fn.setArg(6, meta_f32);
    fn.setArg(7, tile_counts);
    fn.setArg(8, tile_tube_ids);
    fn.setArg(9, tile_depths);
    fn.setArg(10, tile_overflow);
    fn.dispatch((uint64_t)meta.tube_count, 256);
  });

  return std::make_tuple(tile_counts, tile_overflow, tile_unstable, tile_tube_ids, tile_depths);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> metal_simple_backward_samples(
    const torch::Tensor& ma,
    const torch::Tensor& q_uvt,
    const torch::Tensor& opacity,
    const torch::Tensor& color,
    const torch::Tensor& grad_image,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32) {
  check_float_mps_2d(ma, "ma", 3);
  check_float_mps_2d(q_uvt, "q_uvt", 6);
  check_float_mps_1d(opacity, "opacity");
  check_float_mps_2d(color, "color", 3);
  TORCH_CHECK(grad_image.device().is_mps(), "grad_image must be on MPS");
  TORCH_CHECK(grad_image.scalar_type() == torch::kFloat32, "grad_image must be float32");
  TORCH_CHECK(grad_image.dim() == 4 && grad_image.size(3) == 3, "grad_image must have shape [F,H,W,3]");
  TORCH_CHECK(grad_image.is_contiguous(), "grad_image must be contiguous");
  TORCH_CHECK(ma.size(0) == q_uvt.size(0) && ma.size(0) == opacity.size(0) && ma.size(0) == color.size(0),
              "all tube inputs must agree on N");

  auto meta = parse_meta(meta_i32, meta_f32);
  auto& sc = shader_config();
  check_meta(meta, ma.size(0), sc);
  TORCH_CHECK(grad_image.size(0) == meta.frames && grad_image.size(1) == meta.height && grad_image.size(2) == meta.width,
              "grad_image shape must match meta");
  auto& k = kernels();

  auto opts_f = ma.options().dtype(torch::kFloat32);
  int64_t total = (int64_t)meta.frames * (int64_t)meta.height * (int64_t)meta.width * ma.size(0);
  auto grad_ma_samples = torch::empty({total, 3}, opts_f);
  auto grad_q_samples = torch::empty({total, 6}, opts_f);
  auto grad_opacity_samples = torch::empty({total}, opts_f);
  auto grad_color_samples = torch::empty({total, 3}, opts_f);

  launch(k.simple_backward_samples, [&](MetalKernelFunction& fn) {
    fn.setArg(0, ma);
    fn.setArg(1, q_uvt);
    fn.setArg(2, opacity);
    fn.setArg(3, color);
    fn.setArg(4, grad_image);
    fn.setArg(5, meta_i32);
    fn.setArg(6, meta_f32);
    fn.setArg(7, grad_ma_samples);
    fn.setArg(8, grad_q_samples);
    fn.setArg(9, grad_opacity_samples);
    fn.setArg(10, grad_color_samples);
    fn.dispatch((uint64_t)total, 256);
  });

  return std::make_tuple(grad_ma_samples, grad_q_samples, grad_opacity_samples, grad_color_samples);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> stable_backward_samples_impl(
    const torch::Tensor& ma,
    const torch::Tensor& q_uvt,
    const torch::Tensor& depth0,
    const torch::Tensor& depth_beta,
    const torch::Tensor& opacity,
    const torch::Tensor& color,
    const torch::Tensor& grad_image,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32,
    bool write_keys) {
  check_float_mps_2d(ma, "ma", 3);
  check_float_mps_2d(q_uvt, "q_uvt", 6);
  check_float_mps_1d(depth0, "depth0");
  check_float_mps_2d(depth_beta, "depth_beta", 3);
  check_float_mps_1d(opacity, "opacity");
  check_float_mps_2d(color, "color", 3);
  TORCH_CHECK(grad_image.device().is_mps(), "grad_image must be on MPS");
  TORCH_CHECK(grad_image.scalar_type() == torch::kFloat32, "grad_image must be float32");
  TORCH_CHECK(grad_image.dim() == 4 && grad_image.size(3) == 3, "grad_image must have shape [F,H,W,3]");
  TORCH_CHECK(grad_image.is_contiguous(), "grad_image must be contiguous");
  TORCH_CHECK(ma.size(0) == q_uvt.size(0) && ma.size(0) == depth0.size(0) && ma.size(0) == depth_beta.size(0) &&
                  ma.size(0) == opacity.size(0) && ma.size(0) == color.size(0),
              "all tube inputs must agree on N");

  auto meta = parse_meta(meta_i32, meta_f32);
  auto& sc = shader_config();
  check_meta(meta, ma.size(0), sc);
  TORCH_CHECK(grad_image.size(0) == meta.frames && grad_image.size(1) == meta.height && grad_image.size(2) == meta.width,
              "grad_image shape must match meta");
  auto& k = kernels();

  auto opts_f = ma.options().dtype(torch::kFloat32);
  auto opts_i32 = ma.options().dtype(torch::kInt32);
  auto tile_counts = torch::empty({meta.tile_count}, opts_i32);
  auto tile_overflow = torch::empty({meta.tile_count}, opts_i32);
  auto tile_unstable = torch::empty({meta.tile_count}, opts_i32);
  auto tile_tube_ids = torch::empty({meta.tile_count * meta.tile_capacity}, opts_i32);
  auto tile_depths = torch::empty({meta.tile_count * meta.tile_capacity}, opts_f);

  launch(k.clear_tiles, [&](MetalKernelFunction& fn) {
    fn.setArg(0, tile_counts);
    fn.setArg(1, tile_overflow);
    fn.setArg(2, tile_unstable);
    fn.setArg(3, meta_i32);
    fn.dispatch((uint64_t)meta.tile_count, 256);
  });

  launch(k.bin_tubes, [&](MetalKernelFunction& fn) {
    fn.setArg(0, ma);
    fn.setArg(1, q_uvt);
    fn.setArg(2, depth0);
    fn.setArg(3, depth_beta);
    fn.setArg(4, opacity);
    fn.setArg(5, meta_i32);
    fn.setArg(6, meta_f32);
    fn.setArg(7, tile_counts);
    fn.setArg(8, tile_tube_ids);
    fn.setArg(9, tile_depths);
    fn.setArg(10, tile_overflow);
    fn.dispatch((uint64_t)meta.tube_count, 256);
  });

  int64_t entry_count = (int64_t)meta.tile_count * (int64_t)sc.threads * (int64_t)meta.tile_capacity;
  auto grad_ids = torch::empty({entry_count}, opts_i32);
  auto grad_ma_samples = torch::empty({entry_count, 3}, opts_f);
  auto grad_q_samples = torch::empty({entry_count, 6}, opts_f);
  auto grad_opacity_samples = torch::empty({entry_count}, opts_f);
  auto grad_color_samples = torch::empty({entry_count, 3}, opts_f);
  auto grad_keys = write_keys ? torch::empty({entry_count}, opts_i32) : torch::empty({1}, opts_i32);
  auto grad_count = torch::zeros({1}, opts_i32);
  auto key_mode = torch::tensor({static_cast<int32_t>(write_keys ? 1 : 0)}, opts_i32);

  launch(k.stable_backward_samples, [&](MetalKernelFunction& fn) {
    fn.setArg(0, ma);
    fn.setArg(1, q_uvt);
    fn.setArg(2, depth0);
    fn.setArg(3, depth_beta);
    fn.setArg(4, opacity);
    fn.setArg(5, color);
    fn.setArg(6, grad_image);
    fn.setArg(7, meta_i32);
    fn.setArg(8, meta_f32);
    fn.setArg(9, tile_counts);
    fn.setArg(10, tile_tube_ids);
    fn.setArg(11, tile_depths);
    fn.setArg(12, tile_unstable);
    fn.setArg(13, grad_count);
    fn.setArg(14, grad_ids);
    fn.setArg(15, grad_ma_samples);
    fn.setArg(16, grad_q_samples);
    fn.setArg(17, grad_opacity_samples);
    fn.setArg(18, grad_color_samples);
    fn.setArg(19, grad_keys);
    fn.setArg(20, key_mode);
    fn.dispatch((uint64_t)meta.tile_count * (uint64_t)sc.threads, (uint64_t)sc.threads);
  });

  return std::make_tuple(grad_ids, grad_ma_samples, grad_q_samples, grad_opacity_samples, grad_color_samples, grad_keys, tile_unstable, grad_count);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> metal_stable_backward_samples(
    const torch::Tensor& ma,
    const torch::Tensor& q_uvt,
    const torch::Tensor& depth0,
    const torch::Tensor& depth_beta,
    const torch::Tensor& opacity,
    const torch::Tensor& color,
    const torch::Tensor& grad_image,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32) {
  auto out = stable_backward_samples_impl(ma, q_uvt, depth0, depth_beta, opacity, color, grad_image, meta_i32, meta_f32, false);
  return std::make_tuple(
      std::get<0>(out),
      std::get<1>(out),
      std::get<2>(out),
      std::get<3>(out),
      std::get<4>(out),
      std::get<6>(out),
      std::get<7>(out));
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> metal_stable_backward_samples_with_keys(
    const torch::Tensor& ma,
    const torch::Tensor& q_uvt,
    const torch::Tensor& depth0,
    const torch::Tensor& depth_beta,
    const torch::Tensor& opacity,
    const torch::Tensor& color,
    const torch::Tensor& grad_image,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32) {
  return stable_backward_samples_impl(ma, q_uvt, depth0, depth_beta, opacity, color, grad_image, meta_i32, meta_f32, true);
}

namespace {

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> metal_tile_pair_backward_samples_impl(
    const torch::Tensor& ma,
    const torch::Tensor& q_uvt,
    const torch::Tensor& depth0,
    const torch::Tensor& depth_beta,
    const torch::Tensor& opacity,
    const torch::Tensor& color,
    const torch::Tensor& grad_image,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32,
    bool compensated,
    bool target_bounds_only,
    bool suffix_composite,
    bool tile_parallel,
    bool tile_grouped,
    bool tile_sharedsort) {
  check_float_mps_2d(ma, "ma", 3);
  check_float_mps_2d(q_uvt, "q_uvt", 6);
  check_float_mps_1d(depth0, "depth0");
  check_float_mps_2d(depth_beta, "depth_beta", 3);
  check_float_mps_1d(opacity, "opacity");
  check_float_mps_2d(color, "color", 3);
  TORCH_CHECK(grad_image.device().is_mps(), "grad_image must be on MPS");
  TORCH_CHECK(grad_image.scalar_type() == torch::kFloat32, "grad_image must be float32");
  TORCH_CHECK(grad_image.dim() == 4 && grad_image.size(3) == 3, "grad_image must have shape [F,H,W,3]");
  TORCH_CHECK(grad_image.is_contiguous(), "grad_image must be contiguous");
  TORCH_CHECK(ma.size(0) == q_uvt.size(0) && ma.size(0) == depth0.size(0) && ma.size(0) == depth_beta.size(0) &&
                  ma.size(0) == opacity.size(0) && ma.size(0) == color.size(0),
              "all tube inputs must agree on N");

  auto meta = parse_meta(meta_i32, meta_f32);
  auto& sc = shader_config();
  check_meta(meta, ma.size(0), sc);
  TORCH_CHECK(grad_image.size(0) == meta.frames && grad_image.size(1) == meta.height && grad_image.size(2) == meta.width,
              "grad_image shape must match meta");
  auto& k = kernels();

  auto opts_f = ma.options().dtype(torch::kFloat32);
  auto opts_i32 = ma.options().dtype(torch::kInt32);
  auto tile_counts = torch::empty({meta.tile_count}, opts_i32);
  auto tile_overflow = torch::empty({meta.tile_count}, opts_i32);
  auto tile_unstable = torch::empty({meta.tile_count}, opts_i32);
  auto tile_tube_ids = torch::empty({meta.tile_count * meta.tile_capacity}, opts_i32);
  auto tile_depths = torch::empty({meta.tile_count * meta.tile_capacity}, opts_f);

  launch(k.clear_tiles, [&](MetalKernelFunction& fn) {
    fn.setArg(0, tile_counts);
    fn.setArg(1, tile_overflow);
    fn.setArg(2, tile_unstable);
    fn.setArg(3, meta_i32);
    fn.dispatch((uint64_t)meta.tile_count, 256);
  });

  launch(k.bin_tubes, [&](MetalKernelFunction& fn) {
    fn.setArg(0, ma);
    fn.setArg(1, q_uvt);
    fn.setArg(2, depth0);
    fn.setArg(3, depth_beta);
    fn.setArg(4, opacity);
    fn.setArg(5, meta_i32);
    fn.setArg(6, meta_f32);
    fn.setArg(7, tile_counts);
    fn.setArg(8, tile_tube_ids);
    fn.setArg(9, tile_depths);
    fn.setArg(10, tile_overflow);
    fn.dispatch((uint64_t)meta.tube_count, 256);
  });

  int64_t entry_count = (int64_t)meta.tile_count * (int64_t)meta.tile_capacity;
  auto grad_ids = torch::empty({entry_count}, opts_i32);
  auto grad_ma = torch::empty({entry_count, 3}, opts_f);
  auto grad_q = torch::empty({entry_count, 6}, opts_f);
  auto grad_opacity = torch::empty({entry_count}, opts_f);
  auto grad_color = torch::empty({entry_count, 3}, opts_f);
  auto grad_keys = torch::empty({entry_count}, opts_i32);

  auto kernel = suffix_composite
      ? k.tile_pair_suffix_backward_samples
      : target_bounds_only
      ? k.tile_pair_target_bounds_backward_samples
      : tile_parallel
      ? k.tile_pair_parallel_backward_samples
      : tile_grouped
      ? k.tile_pair_grouped_backward_samples
      : tile_sharedsort
      ? k.tile_pair_sharedsort_backward_samples
      : (compensated ? k.tile_pair_backward_samples_compensated : k.tile_pair_backward_samples);
  launch(kernel, [&](MetalKernelFunction& fn) {
    fn.setArg(0, ma);
    fn.setArg(1, q_uvt);
    fn.setArg(2, depth0);
    fn.setArg(3, depth_beta);
    fn.setArg(4, opacity);
    fn.setArg(5, color);
    fn.setArg(6, grad_image);
    fn.setArg(7, meta_i32);
    fn.setArg(8, meta_f32);
    fn.setArg(9, tile_counts);
    fn.setArg(10, tile_tube_ids);
    fn.setArg(11, tile_depths);
    fn.setArg(12, tile_unstable);
    fn.setArg(13, grad_ids);
    fn.setArg(14, grad_ma);
    fn.setArg(15, grad_q);
    fn.setArg(16, grad_opacity);
    fn.setArg(17, grad_color);
    fn.setArg(18, grad_keys);
    if (tile_parallel) {
      fn.dispatch((uint64_t)entry_count * (uint64_t)sc.threads, (uint64_t)sc.threads);
    } else if (tile_grouped || tile_sharedsort) {
      fn.dispatch((uint64_t)meta.tile_count * (uint64_t)sc.threads, (uint64_t)sc.threads);
    } else {
      fn.dispatch((uint64_t)entry_count, 256);
    }
  });

  return std::make_tuple(grad_ids, grad_ma, grad_q, grad_opacity, grad_color, grad_keys, tile_unstable);
}

}  // namespace

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> metal_tile_pair_backward_samples(
    const torch::Tensor& ma,
    const torch::Tensor& q_uvt,
    const torch::Tensor& depth0,
    const torch::Tensor& depth_beta,
    const torch::Tensor& opacity,
    const torch::Tensor& color,
    const torch::Tensor& grad_image,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32) {
  return metal_tile_pair_backward_samples_impl(ma, q_uvt, depth0, depth_beta, opacity, color, grad_image, meta_i32, meta_f32, false, false, false, false, false, false);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> metal_tile_pair_backward_samples_compensated(
    const torch::Tensor& ma,
    const torch::Tensor& q_uvt,
    const torch::Tensor& depth0,
    const torch::Tensor& depth_beta,
    const torch::Tensor& opacity,
    const torch::Tensor& color,
    const torch::Tensor& grad_image,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32) {
  return metal_tile_pair_backward_samples_impl(ma, q_uvt, depth0, depth_beta, opacity, color, grad_image, meta_i32, meta_f32, true, false, false, false, false, false);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> metal_tile_pair_target_bounds_backward_samples(
    const torch::Tensor& ma,
    const torch::Tensor& q_uvt,
    const torch::Tensor& depth0,
    const torch::Tensor& depth_beta,
    const torch::Tensor& opacity,
    const torch::Tensor& color,
    const torch::Tensor& grad_image,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32) {
  return metal_tile_pair_backward_samples_impl(ma, q_uvt, depth0, depth_beta, opacity, color, grad_image, meta_i32, meta_f32, false, true, false, false, false, false);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> metal_tile_pair_suffix_backward_samples(
    const torch::Tensor& ma,
    const torch::Tensor& q_uvt,
    const torch::Tensor& depth0,
    const torch::Tensor& depth_beta,
    const torch::Tensor& opacity,
    const torch::Tensor& color,
    const torch::Tensor& grad_image,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32) {
  return metal_tile_pair_backward_samples_impl(ma, q_uvt, depth0, depth_beta, opacity, color, grad_image, meta_i32, meta_f32, false, false, true, false, false, false);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> metal_tile_pair_parallel_backward_samples(
    const torch::Tensor& ma,
    const torch::Tensor& q_uvt,
    const torch::Tensor& depth0,
    const torch::Tensor& depth_beta,
    const torch::Tensor& opacity,
    const torch::Tensor& color,
    const torch::Tensor& grad_image,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32) {
  return metal_tile_pair_backward_samples_impl(ma, q_uvt, depth0, depth_beta, opacity, color, grad_image, meta_i32, meta_f32, false, false, false, true, false, false);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> metal_tile_pair_grouped_backward_samples(
    const torch::Tensor& ma,
    const torch::Tensor& q_uvt,
    const torch::Tensor& depth0,
    const torch::Tensor& depth_beta,
    const torch::Tensor& opacity,
    const torch::Tensor& color,
    const torch::Tensor& grad_image,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32) {
  return metal_tile_pair_backward_samples_impl(ma, q_uvt, depth0, depth_beta, opacity, color, grad_image, meta_i32, meta_f32, false, false, false, false, true, false);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> metal_tile_pair_sharedsort_backward_samples(
    const torch::Tensor& ma,
    const torch::Tensor& q_uvt,
    const torch::Tensor& depth0,
    const torch::Tensor& depth_beta,
    const torch::Tensor& opacity,
    const torch::Tensor& color,
    const torch::Tensor& grad_image,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32) {
  return metal_tile_pair_backward_samples_impl(ma, q_uvt, depth0, depth_beta, opacity, color, grad_image, meta_i32, meta_f32, false, false, false, false, false, true);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> metal_tile_pair_scanline_backward_samples(
    const torch::Tensor& ma,
    const torch::Tensor& q_uvt,
    const torch::Tensor& depth0,
    const torch::Tensor& depth_beta,
    const torch::Tensor& opacity,
    const torch::Tensor& color,
    const torch::Tensor& grad_image,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32) {
  check_float_mps_2d(ma, "ma", 3);
  check_float_mps_2d(q_uvt, "q_uvt", 6);
  check_float_mps_1d(depth0, "depth0");
  check_float_mps_2d(depth_beta, "depth_beta", 3);
  check_float_mps_1d(opacity, "opacity");
  check_float_mps_2d(color, "color", 3);
  TORCH_CHECK(grad_image.device().is_mps(), "grad_image must be on MPS");
  TORCH_CHECK(grad_image.scalar_type() == torch::kFloat32, "grad_image must be float32");
  TORCH_CHECK(grad_image.dim() == 4 && grad_image.size(3) == 3, "grad_image must have shape [F,H,W,3]");
  TORCH_CHECK(grad_image.is_contiguous(), "grad_image must be contiguous");
  TORCH_CHECK(ma.size(0) == q_uvt.size(0) && ma.size(0) == depth0.size(0) && ma.size(0) == depth_beta.size(0) &&
                  ma.size(0) == opacity.size(0) && ma.size(0) == color.size(0),
              "all tube inputs must agree on N");

  auto meta = parse_meta(meta_i32, meta_f32);
  auto& sc = shader_config();
  check_meta(meta, ma.size(0), sc);
  TORCH_CHECK(grad_image.size(0) == meta.frames && grad_image.size(1) == meta.height && grad_image.size(2) == meta.width,
              "grad_image shape must match meta");
  auto& k = kernels();

  auto opts_f = ma.options().dtype(torch::kFloat32);
  auto opts_i32 = ma.options().dtype(torch::kInt32);
  auto tile_counts = torch::empty({meta.tile_count}, opts_i32);
  auto tile_overflow = torch::empty({meta.tile_count}, opts_i32);
  auto tile_unstable = torch::empty({meta.tile_count}, opts_i32);
  auto tile_tube_ids = torch::empty({meta.tile_count * meta.tile_capacity}, opts_i32);
  auto tile_depths = torch::empty({meta.tile_count * meta.tile_capacity}, opts_f);

  launch(k.clear_tiles, [&](MetalKernelFunction& fn) {
    fn.setArg(0, tile_counts);
    fn.setArg(1, tile_overflow);
    fn.setArg(2, tile_unstable);
    fn.setArg(3, meta_i32);
    fn.dispatch((uint64_t)meta.tile_count, 256);
  });

  launch(k.bin_tubes, [&](MetalKernelFunction& fn) {
    fn.setArg(0, ma);
    fn.setArg(1, q_uvt);
    fn.setArg(2, depth0);
    fn.setArg(3, depth_beta);
    fn.setArg(4, opacity);
    fn.setArg(5, meta_i32);
    fn.setArg(6, meta_f32);
    fn.setArg(7, tile_counts);
    fn.setArg(8, tile_tube_ids);
    fn.setArg(9, tile_depths);
    fn.setArg(10, tile_overflow);
    fn.dispatch((uint64_t)meta.tube_count, 256);
  });

  int64_t chunks_per_tile_slot = (int64_t)meta.tile_t * (int64_t)meta.tile_y;
  int64_t entry_count = (int64_t)meta.tile_count * (int64_t)meta.tile_capacity * chunks_per_tile_slot;
  auto grad_ids = torch::empty({entry_count}, opts_i32);
  auto grad_ma = torch::empty({entry_count, 3}, opts_f);
  auto grad_q = torch::empty({entry_count, 6}, opts_f);
  auto grad_opacity = torch::empty({entry_count}, opts_f);
  auto grad_color = torch::empty({entry_count, 3}, opts_f);
  auto grad_keys = torch::empty({entry_count}, opts_i32);

  launch(k.tile_pair_scanline_backward_samples, [&](MetalKernelFunction& fn) {
    fn.setArg(0, ma);
    fn.setArg(1, q_uvt);
    fn.setArg(2, depth0);
    fn.setArg(3, depth_beta);
    fn.setArg(4, opacity);
    fn.setArg(5, color);
    fn.setArg(6, grad_image);
    fn.setArg(7, meta_i32);
    fn.setArg(8, meta_f32);
    fn.setArg(9, tile_counts);
    fn.setArg(10, tile_tube_ids);
    fn.setArg(11, tile_depths);
    fn.setArg(12, tile_unstable);
    fn.setArg(13, grad_ids);
    fn.setArg(14, grad_ma);
    fn.setArg(15, grad_q);
    fn.setArg(16, grad_opacity);
    fn.setArg(17, grad_color);
    fn.setArg(18, grad_keys);
    fn.dispatch((uint64_t)entry_count, 256);
  });

  return std::make_tuple(grad_ids, grad_ma, grad_q, grad_opacity, grad_color, grad_keys, tile_unstable);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> metal_direct_atomic_backward(
    const torch::Tensor& ma,
    const torch::Tensor& q_uvt,
    const torch::Tensor& depth0,
    const torch::Tensor& depth_beta,
    const torch::Tensor& opacity,
    const torch::Tensor& color,
    const torch::Tensor& grad_image,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32) {
  check_float_mps_2d(ma, "ma", 3);
  check_float_mps_2d(q_uvt, "q_uvt", 6);
  check_float_mps_1d(depth0, "depth0");
  check_float_mps_2d(depth_beta, "depth_beta", 3);
  check_float_mps_1d(opacity, "opacity");
  check_float_mps_2d(color, "color", 3);
  TORCH_CHECK(grad_image.device().is_mps(), "grad_image must be on MPS");
  TORCH_CHECK(grad_image.scalar_type() == torch::kFloat32, "grad_image must be float32");
  TORCH_CHECK(grad_image.dim() == 4 && grad_image.size(3) == 3, "grad_image must have shape [F,H,W,3]");
  TORCH_CHECK(grad_image.is_contiguous(), "grad_image must be contiguous");
  TORCH_CHECK(ma.size(0) == q_uvt.size(0) && ma.size(0) == depth0.size(0) && ma.size(0) == depth_beta.size(0) &&
                  ma.size(0) == opacity.size(0) && ma.size(0) == color.size(0),
              "all tube inputs must agree on N");

  auto meta = parse_meta(meta_i32, meta_f32);
  auto& sc = shader_config();
  check_meta(meta, ma.size(0), sc);
  TORCH_CHECK(grad_image.size(0) == meta.frames && grad_image.size(1) == meta.height && grad_image.size(2) == meta.width,
              "grad_image shape must match meta");
  auto& k = kernels();

  auto opts_f = ma.options().dtype(torch::kFloat32);
  auto opts_i32 = ma.options().dtype(torch::kInt32);
  auto tile_counts = torch::empty({meta.tile_count}, opts_i32);
  auto tile_overflow = torch::empty({meta.tile_count}, opts_i32);
  auto tile_unstable = torch::empty({meta.tile_count}, opts_i32);
  auto tile_tube_ids = torch::empty({meta.tile_count * meta.tile_capacity}, opts_i32);
  auto tile_depths = torch::empty({meta.tile_count * meta.tile_capacity}, opts_f);

  launch(k.clear_tiles, [&](MetalKernelFunction& fn) {
    fn.setArg(0, tile_counts);
    fn.setArg(1, tile_overflow);
    fn.setArg(2, tile_unstable);
    fn.setArg(3, meta_i32);
    fn.dispatch((uint64_t)meta.tile_count, 256);
  });

  launch(k.bin_tubes, [&](MetalKernelFunction& fn) {
    fn.setArg(0, ma);
    fn.setArg(1, q_uvt);
    fn.setArg(2, depth0);
    fn.setArg(3, depth_beta);
    fn.setArg(4, opacity);
    fn.setArg(5, meta_i32);
    fn.setArg(6, meta_f32);
    fn.setArg(7, tile_counts);
    fn.setArg(8, tile_tube_ids);
    fn.setArg(9, tile_depths);
    fn.setArg(10, tile_overflow);
    fn.dispatch((uint64_t)meta.tube_count, 256);
  });

  auto grad_ma = torch::empty({meta.tube_count, 3}, opts_f);
  auto grad_q = torch::empty({meta.tube_count, 6}, opts_f);
  auto grad_opacity = torch::empty({meta.tube_count}, opts_f);
  auto grad_color = torch::empty({meta.tube_count, 3}, opts_f);

  launch(k.clear_direct_gradients, [&](MetalKernelFunction& fn) {
    fn.setArg(0, grad_ma);
    fn.setArg(1, grad_q);
    fn.setArg(2, grad_opacity);
    fn.setArg(3, grad_color);
    fn.setArg(4, meta_i32);
    fn.dispatch((uint64_t)meta.tube_count, 256);
  });

  launch(k.direct_atomic_backward, [&](MetalKernelFunction& fn) {
    fn.setArg(0, ma);
    fn.setArg(1, q_uvt);
    fn.setArg(2, depth0);
    fn.setArg(3, depth_beta);
    fn.setArg(4, opacity);
    fn.setArg(5, color);
    fn.setArg(6, grad_image);
    fn.setArg(7, meta_i32);
    fn.setArg(8, meta_f32);
    fn.setArg(9, tile_counts);
    fn.setArg(10, tile_tube_ids);
    fn.setArg(11, tile_depths);
    fn.setArg(12, tile_unstable);
    fn.setArg(13, grad_ma);
    fn.setArg(14, grad_q);
    fn.setArg(15, grad_opacity);
    fn.setArg(16, grad_color);
    fn.dispatch((uint64_t)meta.tile_count * (uint64_t)sc.threads, (uint64_t)sc.threads);
  });

  return std::make_tuple(grad_ma, grad_q, grad_opacity, grad_color, tile_unstable);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> metal_direct_atomic_backward_gated(
    const torch::Tensor& ma,
    const torch::Tensor& q_uvt,
    const torch::Tensor& depth0,
    const torch::Tensor& depth_beta,
    const torch::Tensor& opacity,
    const torch::Tensor& color,
    const torch::Tensor& grad_image,
    const torch::Tensor& active_start,
    const torch::Tensor& active_stop,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32) {
  check_float_mps_2d(ma, "ma", 3);
  check_float_mps_2d(q_uvt, "q_uvt", 6);
  check_float_mps_1d(depth0, "depth0");
  check_float_mps_2d(depth_beta, "depth_beta", 3);
  check_float_mps_1d(opacity, "opacity");
  check_float_mps_2d(color, "color", 3);
  check_int_mps_1d(active_start, "active_start");
  check_int_mps_1d(active_stop, "active_stop");
  TORCH_CHECK(grad_image.device().is_mps(), "grad_image must be on MPS");
  TORCH_CHECK(grad_image.scalar_type() == torch::kFloat32, "grad_image must be float32");
  TORCH_CHECK(grad_image.dim() == 4 && grad_image.size(3) == 3, "grad_image must have shape [F,H,W,3]");
  TORCH_CHECK(grad_image.is_contiguous(), "grad_image must be contiguous");
  TORCH_CHECK(ma.size(0) == q_uvt.size(0) && ma.size(0) == depth0.size(0) && ma.size(0) == depth_beta.size(0) &&
                  ma.size(0) == opacity.size(0) && ma.size(0) == color.size(0) &&
                  ma.size(0) == active_start.size(0) && ma.size(0) == active_stop.size(0),
              "all gated tube inputs must agree on N");

  auto meta = parse_meta(meta_i32, meta_f32);
  auto& sc = shader_config();
  check_meta(meta, ma.size(0), sc);
  TORCH_CHECK(grad_image.size(0) == meta.frames && grad_image.size(1) == meta.height && grad_image.size(2) == meta.width,
              "grad_image shape must match meta");
  auto& k = kernels();

  auto opts_f = ma.options().dtype(torch::kFloat32);
  auto opts_i32 = ma.options().dtype(torch::kInt32);
  auto tile_counts = torch::empty({meta.tile_count}, opts_i32);
  auto tile_overflow = torch::empty({meta.tile_count}, opts_i32);
  auto tile_unstable = torch::empty({meta.tile_count}, opts_i32);
  auto tile_tube_ids = torch::empty({meta.tile_count * meta.tile_capacity}, opts_i32);
  auto tile_depths = torch::empty({meta.tile_count * meta.tile_capacity}, opts_f);

  launch(k.clear_tiles, [&](MetalKernelFunction& fn) {
    fn.setArg(0, tile_counts);
    fn.setArg(1, tile_overflow);
    fn.setArg(2, tile_unstable);
    fn.setArg(3, meta_i32);
    fn.dispatch((uint64_t)meta.tile_count, 256);
  });

  launch(k.bin_tubes_gated, [&](MetalKernelFunction& fn) {
    fn.setArg(0, ma);
    fn.setArg(1, q_uvt);
    fn.setArg(2, depth0);
    fn.setArg(3, depth_beta);
    fn.setArg(4, opacity);
    fn.setArg(5, meta_i32);
    fn.setArg(6, meta_f32);
    fn.setArg(7, tile_counts);
    fn.setArg(8, tile_tube_ids);
    fn.setArg(9, tile_depths);
    fn.setArg(10, tile_overflow);
    fn.setArg(11, active_start);
    fn.setArg(12, active_stop);
    fn.dispatch((uint64_t)meta.tube_count, 256);
  });

  auto grad_ma = torch::empty({meta.tube_count, 3}, opts_f);
  auto grad_q = torch::empty({meta.tube_count, 6}, opts_f);
  auto grad_opacity = torch::empty({meta.tube_count}, opts_f);
  auto grad_color = torch::empty({meta.tube_count, 3}, opts_f);

  launch(k.clear_direct_gradients, [&](MetalKernelFunction& fn) {
    fn.setArg(0, grad_ma);
    fn.setArg(1, grad_q);
    fn.setArg(2, grad_opacity);
    fn.setArg(3, grad_color);
    fn.setArg(4, meta_i32);
    fn.dispatch((uint64_t)meta.tube_count, 256);
  });

  launch(k.direct_atomic_backward_gated, [&](MetalKernelFunction& fn) {
    fn.setArg(0, ma);
    fn.setArg(1, q_uvt);
    fn.setArg(2, depth0);
    fn.setArg(3, depth_beta);
    fn.setArg(4, opacity);
    fn.setArg(5, color);
    fn.setArg(6, grad_image);
    fn.setArg(7, meta_i32);
    fn.setArg(8, meta_f32);
    fn.setArg(9, tile_counts);
    fn.setArg(10, tile_tube_ids);
    fn.setArg(11, tile_depths);
    fn.setArg(12, tile_unstable);
    fn.setArg(13, grad_ma);
    fn.setArg(14, grad_q);
    fn.setArg(15, grad_opacity);
    fn.setArg(16, grad_color);
    fn.setArg(17, active_start);
    fn.setArg(18, active_stop);
    fn.dispatch((uint64_t)meta.tile_count * (uint64_t)sc.threads, (uint64_t)sc.threads);
  });

  return std::make_tuple(grad_ma, grad_q, grad_opacity, grad_color, tile_unstable);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> metal_direct_atomic_feature_backward(
    const torch::Tensor& ma,
    const torch::Tensor& q_uvt,
    const torch::Tensor& depth0,
    const torch::Tensor& depth_beta,
    const torch::Tensor& opacity,
    const torch::Tensor& feature,
    const torch::Tensor& grad_feature_image,
    const torch::Tensor& grad_alpha_image,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32) {
  check_float_mps_2d(ma, "ma", 3);
  check_float_mps_2d(q_uvt, "q_uvt", 6);
  check_float_mps_1d(depth0, "depth0");
  check_float_mps_2d(depth_beta, "depth_beta", 3);
  check_float_mps_1d(opacity, "opacity");
  TORCH_CHECK(feature.device().is_mps(), "feature must be on MPS");
  TORCH_CHECK(feature.scalar_type() == torch::kFloat32, "feature must be float32");
  TORCH_CHECK(feature.dim() == 2, "feature must have shape [N,F]");
  TORCH_CHECK(feature.is_contiguous(), "feature must be contiguous");
  TORCH_CHECK(grad_feature_image.device().is_mps(), "grad_feature_image must be on MPS");
  TORCH_CHECK(grad_feature_image.scalar_type() == torch::kFloat32, "grad_feature_image must be float32");
  TORCH_CHECK(grad_feature_image.dim() == 4, "grad_feature_image must have shape [frames,height,width,feature_dim]");
  TORCH_CHECK(grad_feature_image.is_contiguous(), "grad_feature_image must be contiguous");
  TORCH_CHECK(grad_alpha_image.device().is_mps(), "grad_alpha_image must be on MPS");
  TORCH_CHECK(grad_alpha_image.scalar_type() == torch::kFloat32, "grad_alpha_image must be float32");
  TORCH_CHECK(grad_alpha_image.dim() == 3, "grad_alpha_image must have shape [F,H,W]");
  TORCH_CHECK(grad_alpha_image.is_contiguous(), "grad_alpha_image must be contiguous");
  TORCH_CHECK(ma.size(0) == q_uvt.size(0) && ma.size(0) == depth0.size(0) && ma.size(0) == depth_beta.size(0) &&
                  ma.size(0) == opacity.size(0) && ma.size(0) == feature.size(0),
              "all tube inputs must agree on N");

  auto meta = parse_meta(meta_i32, meta_f32);
  auto& sc = shader_config();
  check_feature_meta(meta, ma.size(0), sc, feature.size(1));
  TORCH_CHECK(grad_feature_image.size(0) == meta.frames && grad_feature_image.size(1) == meta.height &&
                  grad_feature_image.size(2) == meta.width && grad_feature_image.size(3) == meta.reserved0,
              "grad_feature_image shape must match meta");
  TORCH_CHECK(grad_alpha_image.size(0) == meta.frames && grad_alpha_image.size(1) == meta.height &&
                  grad_alpha_image.size(2) == meta.width,
              "grad_alpha_image shape must match meta");
  auto& k = kernels();

  auto opts_f = ma.options().dtype(torch::kFloat32);
  auto opts_i32 = ma.options().dtype(torch::kInt32);
  auto tile_counts = torch::empty({meta.tile_count}, opts_i32);
  auto tile_overflow = torch::empty({meta.tile_count}, opts_i32);
  auto tile_unstable = torch::empty({meta.tile_count}, opts_i32);
  auto tile_tube_ids = torch::empty({meta.tile_count * meta.tile_capacity}, opts_i32);
  auto tile_depths = torch::empty({meta.tile_count * meta.tile_capacity}, opts_f);

  launch(k.clear_tiles, [&](MetalKernelFunction& fn) {
    fn.setArg(0, tile_counts);
    fn.setArg(1, tile_overflow);
    fn.setArg(2, tile_unstable);
    fn.setArg(3, meta_i32);
    fn.dispatch((uint64_t)meta.tile_count, 256);
  });

  launch(k.bin_tubes, [&](MetalKernelFunction& fn) {
    fn.setArg(0, ma);
    fn.setArg(1, q_uvt);
    fn.setArg(2, depth0);
    fn.setArg(3, depth_beta);
    fn.setArg(4, opacity);
    fn.setArg(5, meta_i32);
    fn.setArg(6, meta_f32);
    fn.setArg(7, tile_counts);
    fn.setArg(8, tile_tube_ids);
    fn.setArg(9, tile_depths);
    fn.setArg(10, tile_overflow);
    fn.dispatch((uint64_t)meta.tube_count, 256);
  });

  auto grad_ma = torch::empty({meta.tube_count, 3}, opts_f);
  auto grad_q = torch::empty({meta.tube_count, 6}, opts_f);
  auto grad_opacity = torch::empty({meta.tube_count}, opts_f);
  auto grad_feature = torch::empty({meta.tube_count, meta.reserved0}, opts_f);

  launch(k.clear_feature_direct_gradients, [&](MetalKernelFunction& fn) {
    fn.setArg(0, grad_ma);
    fn.setArg(1, grad_q);
    fn.setArg(2, grad_opacity);
    fn.setArg(3, grad_feature);
    fn.setArg(4, meta_i32);
    fn.dispatch((uint64_t)meta.tube_count, 256);
  });

  launch(k.direct_atomic_feature_backward, [&](MetalKernelFunction& fn) {
    fn.setArg(0, ma);
    fn.setArg(1, q_uvt);
    fn.setArg(2, depth0);
    fn.setArg(3, depth_beta);
    fn.setArg(4, opacity);
    fn.setArg(5, feature);
    fn.setArg(6, grad_feature_image);
    fn.setArg(7, grad_alpha_image);
    fn.setArg(8, meta_i32);
    fn.setArg(9, meta_f32);
    fn.setArg(10, tile_counts);
    fn.setArg(11, tile_tube_ids);
    fn.setArg(12, tile_depths);
    fn.setArg(13, tile_unstable);
    fn.setArg(14, grad_ma);
    fn.setArg(15, grad_q);
    fn.setArg(16, grad_opacity);
    fn.setArg(17, grad_feature);
    fn.dispatch((uint64_t)meta.tile_count * (uint64_t)sc.threads, (uint64_t)sc.threads);
  });

  return std::make_tuple(grad_ma, grad_q, grad_opacity, grad_feature, tile_unstable);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> metal_direct_atomic_feature_backward_with_bins(
    const torch::Tensor& ma,
    const torch::Tensor& q_uvt,
    const torch::Tensor& depth0,
    const torch::Tensor& depth_beta,
    const torch::Tensor& opacity,
    const torch::Tensor& feature,
    const torch::Tensor& grad_feature_image,
    const torch::Tensor& grad_alpha_image,
    const torch::Tensor& tile_counts,
    const torch::Tensor& tile_tube_ids,
    const torch::Tensor& tile_depths,
    const torch::Tensor& tile_unstable,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32) {
  check_float_mps_2d(ma, "ma", 3);
  check_float_mps_2d(q_uvt, "q_uvt", 6);
  check_float_mps_1d(depth0, "depth0");
  check_float_mps_2d(depth_beta, "depth_beta", 3);
  check_float_mps_1d(opacity, "opacity");
  TORCH_CHECK(feature.device().is_mps(), "feature must be on MPS");
  TORCH_CHECK(feature.scalar_type() == torch::kFloat32, "feature must be float32");
  TORCH_CHECK(feature.dim() == 2, "feature must have shape [N,F]");
  TORCH_CHECK(feature.is_contiguous(), "feature must be contiguous");
  TORCH_CHECK(grad_feature_image.device().is_mps(), "grad_feature_image must be on MPS");
  TORCH_CHECK(grad_feature_image.scalar_type() == torch::kFloat32, "grad_feature_image must be float32");
  TORCH_CHECK(grad_feature_image.dim() == 4, "grad_feature_image must have shape [frames,height,width,feature_dim]");
  TORCH_CHECK(grad_feature_image.is_contiguous(), "grad_feature_image must be contiguous");
  TORCH_CHECK(grad_alpha_image.device().is_mps(), "grad_alpha_image must be on MPS");
  TORCH_CHECK(grad_alpha_image.scalar_type() == torch::kFloat32, "grad_alpha_image must be float32");
  TORCH_CHECK(grad_alpha_image.dim() == 3, "grad_alpha_image must have shape [F,H,W]");
  TORCH_CHECK(grad_alpha_image.is_contiguous(), "grad_alpha_image must be contiguous");
  check_int_mps_1d(tile_counts, "tile_counts");
  check_int_mps_1d(tile_tube_ids, "tile_tube_ids");
  check_float_mps_1d(tile_depths, "tile_depths");
  check_int_mps_1d(tile_unstable, "tile_unstable");
  TORCH_CHECK(ma.size(0) == q_uvt.size(0) && ma.size(0) == depth0.size(0) && ma.size(0) == depth_beta.size(0) &&
                  ma.size(0) == opacity.size(0) && ma.size(0) == feature.size(0),
              "all tube inputs must agree on N");

  auto meta = parse_meta(meta_i32, meta_f32);
  auto& sc = shader_config();
  check_feature_meta(meta, ma.size(0), sc, feature.size(1));
  TORCH_CHECK(grad_feature_image.size(0) == meta.frames && grad_feature_image.size(1) == meta.height &&
                  grad_feature_image.size(2) == meta.width && grad_feature_image.size(3) == meta.reserved0,
              "grad_feature_image shape must match meta");
  TORCH_CHECK(grad_alpha_image.size(0) == meta.frames && grad_alpha_image.size(1) == meta.height &&
                  grad_alpha_image.size(2) == meta.width,
              "grad_alpha_image shape must match meta");
  TORCH_CHECK(tile_counts.numel() == meta.tile_count, "tile_counts shape must match meta");
  TORCH_CHECK(tile_unstable.numel() == meta.tile_count, "tile_unstable shape must match meta");
  TORCH_CHECK(tile_tube_ids.numel() == meta.tile_count * meta.tile_capacity, "tile_tube_ids shape must match meta");
  TORCH_CHECK(tile_depths.numel() == meta.tile_count * meta.tile_capacity, "tile_depths shape must match meta");
  auto& k = kernels();

  auto opts_f = ma.options().dtype(torch::kFloat32);
  auto grad_ma = torch::empty({meta.tube_count, 3}, opts_f);
  auto grad_q = torch::empty({meta.tube_count, 6}, opts_f);
  auto grad_opacity = torch::empty({meta.tube_count}, opts_f);
  auto grad_feature = torch::empty({meta.tube_count, meta.reserved0}, opts_f);

  launch(k.clear_feature_direct_gradients, [&](MetalKernelFunction& fn) {
    fn.setArg(0, grad_ma);
    fn.setArg(1, grad_q);
    fn.setArg(2, grad_opacity);
    fn.setArg(3, grad_feature);
    fn.setArg(4, meta_i32);
    fn.dispatch((uint64_t)meta.tube_count, 256);
  });

  launch(k.direct_atomic_feature_backward, [&](MetalKernelFunction& fn) {
    fn.setArg(0, ma);
    fn.setArg(1, q_uvt);
    fn.setArg(2, depth0);
    fn.setArg(3, depth_beta);
    fn.setArg(4, opacity);
    fn.setArg(5, feature);
    fn.setArg(6, grad_feature_image);
    fn.setArg(7, grad_alpha_image);
    fn.setArg(8, meta_i32);
    fn.setArg(9, meta_f32);
    fn.setArg(10, tile_counts);
    fn.setArg(11, tile_tube_ids);
    fn.setArg(12, tile_depths);
    fn.setArg(13, tile_unstable);
    fn.setArg(14, grad_ma);
    fn.setArg(15, grad_q);
    fn.setArg(16, grad_opacity);
    fn.setArg(17, grad_feature);
    fn.dispatch((uint64_t)meta.tile_count * (uint64_t)sc.threads, (uint64_t)sc.threads);
  });

  return std::make_tuple(grad_ma, grad_q, grad_opacity, grad_feature, tile_unstable);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
metal_direct_atomic_feature_sparse_pixels_backward_with_bins(
    const torch::Tensor& ma,
    const torch::Tensor& q_uvt,
    const torch::Tensor& depth0,
    const torch::Tensor& depth_beta,
    const torch::Tensor& opacity,
    const torch::Tensor& feature,
    const torch::Tensor& pixel_ids,
    const torch::Tensor& grad_feature_values,
    const torch::Tensor& grad_alpha_values,
    const torch::Tensor& tile_counts,
    const torch::Tensor& tile_tube_ids,
    const torch::Tensor& tile_depths,
    const torch::Tensor& tile_unstable,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32) {
  check_float_mps_2d(ma, "ma", 3);
  check_float_mps_2d(q_uvt, "q_uvt", 6);
  check_float_mps_1d(depth0, "depth0");
  check_float_mps_2d(depth_beta, "depth_beta", 3);
  check_float_mps_1d(opacity, "opacity");
  TORCH_CHECK(feature.device().is_mps(), "feature must be on MPS");
  TORCH_CHECK(feature.scalar_type() == torch::kFloat32, "feature must be float32");
  TORCH_CHECK(feature.dim() == 2, "feature must have shape [N,F]");
  TORCH_CHECK(feature.is_contiguous(), "feature must be contiguous");
  check_int_mps_1d(pixel_ids, "pixel_ids");
  TORCH_CHECK(grad_feature_values.device().is_mps(), "grad_feature_values must be on MPS");
  TORCH_CHECK(grad_feature_values.scalar_type() == torch::kFloat32, "grad_feature_values must be float32");
  TORCH_CHECK(grad_feature_values.dim() == 2, "grad_feature_values must have shape [M,F]");
  TORCH_CHECK(grad_feature_values.is_contiguous(), "grad_feature_values must be contiguous");
  check_float_mps_1d(grad_alpha_values, "grad_alpha_values");
  check_int_mps_1d(tile_counts, "tile_counts");
  check_int_mps_1d(tile_tube_ids, "tile_tube_ids");
  check_float_mps_1d(tile_depths, "tile_depths");
  check_int_mps_1d(tile_unstable, "tile_unstable");
  TORCH_CHECK(ma.size(0) == q_uvt.size(0) && ma.size(0) == depth0.size(0) && ma.size(0) == depth_beta.size(0) &&
                  ma.size(0) == opacity.size(0) && ma.size(0) == feature.size(0),
              "all tube inputs must agree on N");
  TORCH_CHECK(pixel_ids.size(0) == grad_feature_values.size(0) && pixel_ids.size(0) == grad_alpha_values.size(0),
              "sparse pixel inputs must agree on M");

  auto meta = parse_meta(meta_i32, meta_f32);
  auto& sc = shader_config();
  check_feature_meta(meta, ma.size(0), sc, feature.size(1));
  TORCH_CHECK(grad_feature_values.size(1) == meta.reserved0, "grad_feature_values feature dim must match meta");
  TORCH_CHECK(tile_counts.numel() == meta.tile_count, "tile_counts shape must match meta");
  TORCH_CHECK(tile_unstable.numel() == meta.tile_count, "tile_unstable shape must match meta");
  TORCH_CHECK(tile_tube_ids.numel() == meta.tile_count * meta.tile_capacity, "tile_tube_ids shape must match meta");
  TORCH_CHECK(tile_depths.numel() == meta.tile_count * meta.tile_capacity, "tile_depths shape must match meta");
  auto& k = kernels();

  auto opts_f = ma.options().dtype(torch::kFloat32);
  auto grad_ma = torch::empty({meta.tube_count, 3}, opts_f);
  auto grad_q = torch::empty({meta.tube_count, 6}, opts_f);
  auto grad_opacity = torch::empty({meta.tube_count}, opts_f);
  auto grad_feature = torch::empty({meta.tube_count, meta.reserved0}, opts_f);

  launch(k.clear_feature_direct_gradients, [&](MetalKernelFunction& fn) {
    fn.setArg(0, grad_ma);
    fn.setArg(1, grad_q);
    fn.setArg(2, grad_opacity);
    fn.setArg(3, grad_feature);
    fn.setArg(4, meta_i32);
    fn.dispatch((uint64_t)meta.tube_count, 256);
  });

  int64_t sparse_count = pixel_ids.numel();
  if (sparse_count > 0) {
    launch(k.direct_atomic_feature_sparse_pixels_backward, [&](MetalKernelFunction& fn) {
      fn.setArg(0, ma);
      fn.setArg(1, q_uvt);
      fn.setArg(2, depth0);
      fn.setArg(3, depth_beta);
      fn.setArg(4, opacity);
      fn.setArg(5, feature);
      fn.setArg(6, pixel_ids);
      fn.setArg(7, grad_feature_values);
      fn.setArg(8, grad_alpha_values);
      fn.setArg(9, meta_i32);
      fn.setArg(10, meta_f32);
      fn.setArg(11, tile_counts);
      fn.setArg(12, tile_tube_ids);
      fn.setArg(13, tile_depths);
      fn.setArg(14, tile_unstable);
      fn.setArg(15, grad_ma);
      fn.setArg(16, grad_q);
      fn.setArg(17, grad_opacity);
      fn.setArg(18, grad_feature);
      fn.dispatch((uint64_t)sparse_count, 256);
    });
  }

  return std::make_tuple(grad_ma, grad_q, grad_opacity, grad_feature, tile_unstable);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
metal_direct_atomic_feature_sparse_hidden_sigmoid_mse_backward_with_bins(
    const torch::Tensor& ma,
    const torch::Tensor& q_uvt,
    const torch::Tensor& depth0,
    const torch::Tensor& depth_beta,
    const torch::Tensor& opacity,
    const torch::Tensor& feature,
    const torch::Tensor& pixel_ids,
    const torch::Tensor& target_rgb_values,
    const torch::Tensor& hidden_weight,
    const torch::Tensor& hidden_bias,
    const torch::Tensor& output_weight,
    const torch::Tensor& output_bias,
    const torch::Tensor& tile_counts,
    const torch::Tensor& tile_tube_ids,
    const torch::Tensor& tile_depths,
    const torch::Tensor& tile_unstable,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32) {
  check_float_mps_2d(ma, "ma", 3);
  check_float_mps_2d(q_uvt, "q_uvt", 6);
  check_float_mps_1d(depth0, "depth0");
  check_float_mps_2d(depth_beta, "depth_beta", 3);
  check_float_mps_1d(opacity, "opacity");
  TORCH_CHECK(feature.device().is_mps(), "feature must be on MPS");
  TORCH_CHECK(feature.scalar_type() == torch::kFloat32, "feature must be float32");
  TORCH_CHECK(feature.dim() == 2, "feature must have shape [N,F]");
  TORCH_CHECK(feature.is_contiguous(), "feature must be contiguous");
  check_int_mps_1d(pixel_ids, "pixel_ids");
  check_float_mps_2d(target_rgb_values, "target_rgb_values", 3);
  TORCH_CHECK(hidden_weight.device().is_mps(), "hidden_weight must be on MPS");
  TORCH_CHECK(hidden_weight.scalar_type() == torch::kFloat32, "hidden_weight must be float32");
  TORCH_CHECK(hidden_weight.dim() == 2 && hidden_weight.size(1) == feature.size(1),
              "hidden_weight must have shape [hidden_dim,F]");
  TORCH_CHECK(hidden_weight.is_contiguous(), "hidden_weight must be contiguous");
  check_float_mps_1d(hidden_bias, "hidden_bias");
  TORCH_CHECK(output_weight.device().is_mps(), "output_weight must be on MPS");
  TORCH_CHECK(output_weight.scalar_type() == torch::kFloat32, "output_weight must be float32");
  TORCH_CHECK(output_weight.dim() == 2 && output_weight.size(0) == 3 && output_weight.size(1) == hidden_weight.size(0),
              "output_weight must have shape [3,hidden_dim]");
  TORCH_CHECK(output_weight.is_contiguous(), "output_weight must be contiguous");
  TORCH_CHECK(output_bias.device().is_mps(), "output_bias must be on MPS");
  TORCH_CHECK(output_bias.scalar_type() == torch::kFloat32, "output_bias must be float32");
  TORCH_CHECK(output_bias.dim() == 1 && output_bias.size(0) == 3, "output_bias must have shape [3]");
  TORCH_CHECK(output_bias.is_contiguous(), "output_bias must be contiguous");
  TORCH_CHECK(hidden_bias.size(0) == hidden_weight.size(0), "hidden_bias shape must match hidden_weight");
  check_int_mps_1d(tile_counts, "tile_counts");
  check_int_mps_1d(tile_tube_ids, "tile_tube_ids");
  check_float_mps_1d(tile_depths, "tile_depths");
  check_int_mps_1d(tile_unstable, "tile_unstable");
  TORCH_CHECK(ma.size(0) == q_uvt.size(0) && ma.size(0) == depth0.size(0) && ma.size(0) == depth_beta.size(0) &&
                  ma.size(0) == opacity.size(0) && ma.size(0) == feature.size(0),
              "all tube inputs must agree on N");
  TORCH_CHECK(pixel_ids.size(0) == target_rgb_values.size(0), "sparse pixel target inputs must agree on M");

  auto meta = parse_meta(meta_i32, meta_f32);
  auto& sc = shader_config();
  check_feature_meta(meta, ma.size(0), sc, feature.size(1));
  TORCH_CHECK(meta.reserved0 <= 64, "sparse hidden sigmoid MSE requires feature_dim <= 64; got ", meta.reserved0);
  TORCH_CHECK(hidden_weight.size(0) > 0 && hidden_weight.size(0) <= 64,
              "sparse hidden sigmoid MSE requires hidden_dim in 1..64; got ", hidden_weight.size(0));
  TORCH_CHECK(tile_counts.numel() == meta.tile_count, "tile_counts shape must match meta");
  TORCH_CHECK(tile_unstable.numel() == meta.tile_count, "tile_unstable shape must match meta");
  TORCH_CHECK(tile_tube_ids.numel() == meta.tile_count * meta.tile_capacity, "tile_tube_ids shape must match meta");
  TORCH_CHECK(tile_depths.numel() == meta.tile_count * meta.tile_capacity, "tile_depths shape must match meta");
  auto& k = kernels();

  auto opts_f = ma.options().dtype(torch::kFloat32);
  auto opts_i32 = ma.options().dtype(torch::kInt32);
  auto grad_ma = torch::empty({meta.tube_count, 3}, opts_f);
  auto grad_q = torch::empty({meta.tube_count, 6}, opts_f);
  auto grad_opacity = torch::empty({meta.tube_count}, opts_f);
  auto grad_feature = torch::empty({meta.tube_count, meta.reserved0}, opts_f);
  auto loss = torch::zeros({1}, opts_f);

  launch(k.clear_feature_direct_gradients, [&](MetalKernelFunction& fn) {
    fn.setArg(0, grad_ma);
    fn.setArg(1, grad_q);
    fn.setArg(2, grad_opacity);
    fn.setArg(3, grad_feature);
    fn.setArg(4, meta_i32);
    fn.dispatch((uint64_t)meta.tube_count, 256);
  });

  int64_t sparse_count = pixel_ids.numel();
  if (sparse_count > 0) {
    int32_t loss_norm_elems =
        meta.reserved1 > 0 ? static_cast<int32_t>(meta.reserved1) : static_cast<int32_t>(sparse_count * 3);
    auto hidden_meta = torch::tensor(
        {static_cast<int32_t>(hidden_weight.size(0)), static_cast<int32_t>(sparse_count), loss_norm_elems},
        opts_i32);
    launch(k.direct_atomic_feature_sparse_hidden_sigmoid_mse_backward, [&](MetalKernelFunction& fn) {
      fn.setArg(0, ma);
      fn.setArg(1, q_uvt);
      fn.setArg(2, depth0);
      fn.setArg(3, depth_beta);
      fn.setArg(4, opacity);
      fn.setArg(5, feature);
      fn.setArg(6, pixel_ids);
      fn.setArg(7, target_rgb_values);
      fn.setArg(8, hidden_weight);
      fn.setArg(9, hidden_bias);
      fn.setArg(10, output_weight);
      fn.setArg(11, output_bias);
      fn.setArg(12, hidden_meta);
      fn.setArg(13, meta_i32);
      fn.setArg(14, meta_f32);
      fn.setArg(15, tile_counts);
      fn.setArg(16, tile_tube_ids);
      fn.setArg(17, tile_depths);
      fn.setArg(18, tile_unstable);
      fn.setArg(19, grad_ma);
      fn.setArg(20, grad_q);
      fn.setArg(21, grad_opacity);
      fn.setArg(22, grad_feature);
      fn.setArg(23, loss);
      fn.dispatch((uint64_t)sparse_count, 256);
    });
  }

  return std::make_tuple(grad_ma, grad_q, grad_opacity, grad_feature, loss, tile_unstable);
}

std::tuple<torch::Tensor, torch::Tensor>
metal_sparse_hidden_sigmoid_target_area_forward_sums_with_bins(
    const torch::Tensor& ma,
    const torch::Tensor& q_uvt,
    const torch::Tensor& depth0,
    const torch::Tensor& depth_beta,
    const torch::Tensor& opacity,
    const torch::Tensor& feature,
    const torch::Tensor& pixel_ids,
    const torch::Tensor& cell_ids,
    const torch::Tensor& hidden_weight,
    const torch::Tensor& hidden_bias,
    const torch::Tensor& output_weight,
    const torch::Tensor& output_bias,
    const torch::Tensor& tile_counts,
    const torch::Tensor& tile_tube_ids,
    const torch::Tensor& tile_depths,
    const torch::Tensor& tile_unstable,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32,
    int64_t cell_count) {
  check_float_mps_2d(ma, "ma", 3);
  check_float_mps_2d(q_uvt, "q_uvt", 6);
  check_float_mps_1d(depth0, "depth0");
  check_float_mps_2d(depth_beta, "depth_beta", 3);
  check_float_mps_1d(opacity, "opacity");
  TORCH_CHECK(feature.device().is_mps(), "feature must be on MPS");
  TORCH_CHECK(feature.scalar_type() == torch::kFloat32, "feature must be float32");
  TORCH_CHECK(feature.dim() == 2, "feature must have shape [N,F]");
  TORCH_CHECK(feature.is_contiguous(), "feature must be contiguous");
  check_int_mps_1d(pixel_ids, "pixel_ids");
  check_int_mps_1d(cell_ids, "cell_ids");
  TORCH_CHECK(pixel_ids.numel() == cell_ids.numel(), "pixel_ids and cell_ids must agree on M");
  TORCH_CHECK(cell_count > 0, "cell_count must be positive");
  TORCH_CHECK(hidden_weight.device().is_mps(), "hidden_weight must be on MPS");
  TORCH_CHECK(hidden_weight.scalar_type() == torch::kFloat32, "hidden_weight must be float32");
  TORCH_CHECK(hidden_weight.dim() == 2 && hidden_weight.size(1) == feature.size(1),
              "hidden_weight must have shape [hidden_dim,F]");
  TORCH_CHECK(hidden_weight.is_contiguous(), "hidden_weight must be contiguous");
  check_float_mps_1d(hidden_bias, "hidden_bias");
  TORCH_CHECK(output_weight.device().is_mps(), "output_weight must be on MPS");
  TORCH_CHECK(output_weight.scalar_type() == torch::kFloat32, "output_weight must be float32");
  TORCH_CHECK(output_weight.dim() == 2 && output_weight.size(0) == 3 && output_weight.size(1) == hidden_weight.size(0),
              "output_weight must have shape [3,hidden_dim]");
  TORCH_CHECK(output_weight.is_contiguous(), "output_weight must be contiguous");
  TORCH_CHECK(output_bias.device().is_mps(), "output_bias must be on MPS");
  TORCH_CHECK(output_bias.scalar_type() == torch::kFloat32, "output_bias must be float32");
  TORCH_CHECK(output_bias.dim() == 1 && output_bias.size(0) == 3, "output_bias must have shape [3]");
  TORCH_CHECK(output_bias.is_contiguous(), "output_bias must be contiguous");
  TORCH_CHECK(hidden_bias.size(0) == hidden_weight.size(0), "hidden_bias shape must match hidden_weight");
  check_int_mps_1d(tile_counts, "tile_counts");
  check_int_mps_1d(tile_tube_ids, "tile_tube_ids");
  check_float_mps_1d(tile_depths, "tile_depths");
  check_int_mps_1d(tile_unstable, "tile_unstable");
  TORCH_CHECK(ma.size(0) == q_uvt.size(0) && ma.size(0) == depth0.size(0) && ma.size(0) == depth_beta.size(0) &&
                  ma.size(0) == opacity.size(0) && ma.size(0) == feature.size(0),
              "all tube inputs must agree on N");

  auto meta = parse_meta(meta_i32, meta_f32);
  auto& sc = shader_config();
  check_feature_meta(meta, ma.size(0), sc, feature.size(1));
  TORCH_CHECK(meta.reserved0 <= 64, "target-area hidden VJP requires feature_dim <= 64; got ", meta.reserved0);
  TORCH_CHECK(hidden_weight.size(0) > 0 && hidden_weight.size(0) <= 64,
              "target-area hidden VJP requires hidden_dim in 1..64; got ", hidden_weight.size(0));
  TORCH_CHECK(tile_counts.numel() == meta.tile_count, "tile_counts shape must match meta");
  TORCH_CHECK(tile_unstable.numel() == meta.tile_count, "tile_unstable shape must match meta");
  TORCH_CHECK(tile_tube_ids.numel() == meta.tile_count * meta.tile_capacity, "tile_tube_ids shape must match meta");
  TORCH_CHECK(tile_depths.numel() == meta.tile_count * meta.tile_capacity, "tile_depths shape must match meta");

  auto opts_f = ma.options().dtype(torch::kFloat32);
  auto opts_i32 = ma.options().dtype(torch::kInt32);
  auto pred_sums = torch::zeros({cell_count, 3}, opts_f);
  int64_t sparse_count = pixel_ids.numel();
  if (sparse_count > 0) {
    auto hidden_meta = torch::tensor(
        {static_cast<int32_t>(hidden_weight.size(0)), static_cast<int32_t>(cell_count)},
        opts_i32);
    auto& k = kernels();
    launch(k.sparse_hidden_sigmoid_target_area_forward_sums, [&](MetalKernelFunction& fn) {
      fn.setArg(0, ma);
      fn.setArg(1, q_uvt);
      fn.setArg(2, depth0);
      fn.setArg(3, depth_beta);
      fn.setArg(4, opacity);
      fn.setArg(5, feature);
      fn.setArg(6, pixel_ids);
      fn.setArg(7, cell_ids);
      fn.setArg(8, hidden_weight);
      fn.setArg(9, hidden_bias);
      fn.setArg(10, output_weight);
      fn.setArg(11, output_bias);
      fn.setArg(12, hidden_meta);
      fn.setArg(13, meta_i32);
      fn.setArg(14, meta_f32);
      fn.setArg(15, tile_counts);
      fn.setArg(16, tile_tube_ids);
      fn.setArg(17, tile_depths);
      fn.setArg(18, tile_unstable);
      fn.setArg(19, pred_sums);
      fn.dispatch((uint64_t)sparse_count, 256);
    });
  }
  return std::make_tuple(pred_sums, tile_unstable);
}

std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor>
metal_direct_atomic_feature_sparse_hidden_target_area_backward_with_bins(
    const torch::Tensor& ma,
    const torch::Tensor& q_uvt,
    const torch::Tensor& depth0,
    const torch::Tensor& depth_beta,
    const torch::Tensor& opacity,
    const torch::Tensor& feature,
    const torch::Tensor& pixel_ids,
    const torch::Tensor& cell_ids,
    const torch::Tensor& cell_grad_rgb,
    const torch::Tensor& hidden_weight,
    const torch::Tensor& hidden_bias,
    const torch::Tensor& output_weight,
    const torch::Tensor& output_bias,
    const torch::Tensor& tile_counts,
    const torch::Tensor& tile_tube_ids,
    const torch::Tensor& tile_depths,
    const torch::Tensor& tile_unstable,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32,
    int64_t mode_bits) {
  check_float_mps_2d(ma, "ma", 3);
  check_float_mps_2d(q_uvt, "q_uvt", 6);
  check_float_mps_1d(depth0, "depth0");
  check_float_mps_2d(depth_beta, "depth_beta", 3);
  check_float_mps_1d(opacity, "opacity");
  TORCH_CHECK(feature.device().is_mps(), "feature must be on MPS");
  TORCH_CHECK(feature.scalar_type() == torch::kFloat32, "feature must be float32");
  TORCH_CHECK(feature.dim() == 2, "feature must have shape [N,F]");
  TORCH_CHECK(feature.is_contiguous(), "feature must be contiguous");
  check_int_mps_1d(pixel_ids, "pixel_ids");
  check_int_mps_1d(cell_ids, "cell_ids");
  check_float_mps_2d(cell_grad_rgb, "cell_grad_rgb", 3);
  TORCH_CHECK(pixel_ids.numel() == cell_ids.numel(), "pixel_ids and cell_ids must agree on M");
  TORCH_CHECK(hidden_weight.device().is_mps(), "hidden_weight must be on MPS");
  TORCH_CHECK(hidden_weight.scalar_type() == torch::kFloat32, "hidden_weight must be float32");
  TORCH_CHECK(hidden_weight.dim() == 2 && hidden_weight.size(1) == feature.size(1),
              "hidden_weight must have shape [hidden_dim,F]");
  TORCH_CHECK(hidden_weight.is_contiguous(), "hidden_weight must be contiguous");
  check_float_mps_1d(hidden_bias, "hidden_bias");
  TORCH_CHECK(output_weight.device().is_mps(), "output_weight must be on MPS");
  TORCH_CHECK(output_weight.scalar_type() == torch::kFloat32, "output_weight must be float32");
  TORCH_CHECK(output_weight.dim() == 2 && output_weight.size(0) == 3 && output_weight.size(1) == hidden_weight.size(0),
              "output_weight must have shape [3,hidden_dim]");
  TORCH_CHECK(output_weight.is_contiguous(), "output_weight must be contiguous");
  TORCH_CHECK(output_bias.device().is_mps(), "output_bias must be on MPS");
  TORCH_CHECK(output_bias.scalar_type() == torch::kFloat32, "output_bias must be float32");
  TORCH_CHECK(output_bias.dim() == 1 && output_bias.size(0) == 3, "output_bias must have shape [3]");
  TORCH_CHECK(output_bias.is_contiguous(), "output_bias must be contiguous");
  TORCH_CHECK(hidden_bias.size(0) == hidden_weight.size(0), "hidden_bias shape must match hidden_weight");
  check_int_mps_1d(tile_counts, "tile_counts");
  check_int_mps_1d(tile_tube_ids, "tile_tube_ids");
  check_float_mps_1d(tile_depths, "tile_depths");
  check_int_mps_1d(tile_unstable, "tile_unstable");
  TORCH_CHECK(mode_bits >= 0 && mode_bits <= 511, "target-area hidden VJP mode_bits must be in 0..511");
  TORCH_CHECK(ma.size(0) == q_uvt.size(0) && ma.size(0) == depth0.size(0) && ma.size(0) == depth_beta.size(0) &&
                  ma.size(0) == opacity.size(0) && ma.size(0) == feature.size(0),
              "all tube inputs must agree on N");

  auto meta = parse_meta(meta_i32, meta_f32);
  auto& sc = shader_config();
  check_feature_meta(meta, ma.size(0), sc, feature.size(1));
  TORCH_CHECK(meta.reserved0 <= 64, "target-area hidden VJP requires feature_dim <= 64; got ", meta.reserved0);
  TORCH_CHECK(hidden_weight.size(0) > 0 && hidden_weight.size(0) <= 64,
              "target-area hidden VJP requires hidden_dim in 1..64; got ", hidden_weight.size(0));
  TORCH_CHECK(tile_counts.numel() == meta.tile_count, "tile_counts shape must match meta");
  TORCH_CHECK(tile_unstable.numel() == meta.tile_count, "tile_unstable shape must match meta");
  TORCH_CHECK(tile_tube_ids.numel() == meta.tile_count * meta.tile_capacity, "tile_tube_ids shape must match meta");
  TORCH_CHECK(tile_depths.numel() == meta.tile_count * meta.tile_capacity, "tile_depths shape must match meta");

  auto opts_f = ma.options().dtype(torch::kFloat32);
  auto opts_i32 = ma.options().dtype(torch::kInt32);
  auto grad_ma = torch::empty({meta.tube_count, 3}, opts_f);
  auto grad_q = torch::empty({meta.tube_count, 6}, opts_f);
  auto grad_opacity = torch::empty({meta.tube_count}, opts_f);
  auto grad_feature = torch::empty({meta.tube_count, meta.reserved0}, opts_f);
  auto grad_hidden_weight = torch::zeros({hidden_weight.size(0), meta.reserved0}, opts_f);
  auto grad_hidden_bias = torch::zeros({hidden_weight.size(0)}, opts_f);
  auto grad_output_weight = torch::zeros({3, hidden_weight.size(0)}, opts_f);
  auto grad_output_bias = torch::zeros({3}, opts_f);
  auto& k = kernels();
  launch(k.clear_feature_direct_gradients, [&](MetalKernelFunction& fn) {
    fn.setArg(0, grad_ma);
    fn.setArg(1, grad_q);
    fn.setArg(2, grad_opacity);
    fn.setArg(3, grad_feature);
    fn.setArg(4, meta_i32);
    fn.dispatch((uint64_t)meta.tube_count, 256);
  });

  int64_t sparse_count = pixel_ids.numel();
  if (sparse_count > 0) {
    auto hidden_meta = torch::tensor(
        {
            static_cast<int32_t>(hidden_weight.size(0)),
            static_cast<int32_t>(cell_grad_rgb.size(0)),
            static_cast<int32_t>(mode_bits),
        },
        opts_i32);
    launch(k.direct_atomic_feature_sparse_hidden_target_area_backward, [&](MetalKernelFunction& fn) {
      fn.setArg(0, ma);
      fn.setArg(1, q_uvt);
      fn.setArg(2, depth0);
      fn.setArg(3, depth_beta);
      fn.setArg(4, opacity);
      fn.setArg(5, feature);
      fn.setArg(6, pixel_ids);
      fn.setArg(7, cell_ids);
      fn.setArg(8, cell_grad_rgb);
      fn.setArg(9, hidden_weight);
      fn.setArg(10, hidden_bias);
      fn.setArg(11, output_weight);
      fn.setArg(12, output_bias);
      fn.setArg(13, hidden_meta);
      fn.setArg(14, meta_i32);
      fn.setArg(15, meta_f32);
      fn.setArg(16, tile_counts);
      fn.setArg(17, tile_tube_ids);
      fn.setArg(18, tile_depths);
      fn.setArg(19, tile_unstable);
      fn.setArg(20, grad_ma);
      fn.setArg(21, grad_q);
      fn.setArg(22, grad_opacity);
      fn.setArg(23, grad_feature);
      fn.setArg(24, grad_hidden_weight);
      fn.setArg(25, grad_hidden_bias);
      fn.setArg(26, grad_output_weight);
      fn.setArg(27, grad_output_bias);
      fn.dispatch((uint64_t)sparse_count, 256);
    });
  }

  return std::make_tuple(
      grad_ma,
      grad_q,
      grad_opacity,
      grad_feature,
      tile_unstable,
      grad_hidden_weight,
      grad_hidden_bias,
      grad_output_weight,
      grad_output_bias);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
metal_direct_atomic_feature_linear_sigmoid_mse_backward(
    const torch::Tensor& ma,
    const torch::Tensor& q_uvt,
    const torch::Tensor& depth0,
    const torch::Tensor& depth_beta,
    const torch::Tensor& opacity,
    const torch::Tensor& feature,
    const torch::Tensor& target_rgb,
    const torch::Tensor& color_weight,
    const torch::Tensor& color_bias,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32) {
  check_float_mps_2d(ma, "ma", 3);
  check_float_mps_2d(q_uvt, "q_uvt", 6);
  check_float_mps_1d(depth0, "depth0");
  check_float_mps_2d(depth_beta, "depth_beta", 3);
  check_float_mps_1d(opacity, "opacity");
  TORCH_CHECK(feature.device().is_mps(), "feature must be on MPS");
  TORCH_CHECK(feature.scalar_type() == torch::kFloat32, "feature must be float32");
  TORCH_CHECK(feature.dim() == 2, "feature must have shape [N,F]");
  TORCH_CHECK(feature.is_contiguous(), "feature must be contiguous");
  TORCH_CHECK(target_rgb.device().is_mps(), "target_rgb must be on MPS");
  TORCH_CHECK(target_rgb.scalar_type() == torch::kFloat32, "target_rgb must be float32");
  TORCH_CHECK(target_rgb.dim() == 4 && target_rgb.size(3) == 3, "target_rgb must have shape [frames,height,width,3]");
  TORCH_CHECK(target_rgb.is_contiguous(), "target_rgb must be contiguous");
  TORCH_CHECK(color_weight.device().is_mps(), "color_weight must be on MPS");
  TORCH_CHECK(color_weight.scalar_type() == torch::kFloat32, "color_weight must be float32");
  TORCH_CHECK(color_weight.dim() == 2 && color_weight.size(0) == 3 && color_weight.size(1) == feature.size(1),
              "color_weight must have shape [3,F]");
  TORCH_CHECK(color_weight.is_contiguous(), "color_weight must be contiguous");
  TORCH_CHECK(color_bias.device().is_mps(), "color_bias must be on MPS");
  TORCH_CHECK(color_bias.scalar_type() == torch::kFloat32, "color_bias must be float32");
  TORCH_CHECK(color_bias.dim() == 1 && color_bias.size(0) == 3, "color_bias must have shape [3]");
  TORCH_CHECK(color_bias.is_contiguous(), "color_bias must be contiguous");
  TORCH_CHECK(ma.size(0) == q_uvt.size(0) && ma.size(0) == depth0.size(0) && ma.size(0) == depth_beta.size(0) &&
                  ma.size(0) == opacity.size(0) && ma.size(0) == feature.size(0),
              "all tube inputs must agree on N");

  auto meta = parse_meta(meta_i32, meta_f32);
  auto& sc = shader_config();
  check_feature_meta(meta, ma.size(0), sc, feature.size(1));
  TORCH_CHECK(meta.reserved0 <= 64, "linear sigmoid MSE handoff requires feature_dim <= 64; got ", meta.reserved0);
  TORCH_CHECK(target_rgb.size(0) == meta.frames && target_rgb.size(1) == meta.height && target_rgb.size(2) == meta.width,
              "target_rgb shape must match meta");
  auto& k = kernels();

  auto opts_f = ma.options().dtype(torch::kFloat32);
  auto opts_i32 = ma.options().dtype(torch::kInt32);
  auto tile_counts = torch::empty({meta.tile_count}, opts_i32);
  auto tile_overflow = torch::empty({meta.tile_count}, opts_i32);
  auto tile_unstable = torch::empty({meta.tile_count}, opts_i32);
  auto tile_tube_ids = torch::empty({meta.tile_count * meta.tile_capacity}, opts_i32);
  auto tile_depths = torch::empty({meta.tile_count * meta.tile_capacity}, opts_f);

  launch(k.clear_tiles, [&](MetalKernelFunction& fn) {
    fn.setArg(0, tile_counts);
    fn.setArg(1, tile_overflow);
    fn.setArg(2, tile_unstable);
    fn.setArg(3, meta_i32);
    fn.dispatch((uint64_t)meta.tile_count, 256);
  });

  launch(k.bin_tubes, [&](MetalKernelFunction& fn) {
    fn.setArg(0, ma);
    fn.setArg(1, q_uvt);
    fn.setArg(2, depth0);
    fn.setArg(3, depth_beta);
    fn.setArg(4, opacity);
    fn.setArg(5, meta_i32);
    fn.setArg(6, meta_f32);
    fn.setArg(7, tile_counts);
    fn.setArg(8, tile_tube_ids);
    fn.setArg(9, tile_depths);
    fn.setArg(10, tile_overflow);
    fn.dispatch((uint64_t)meta.tube_count, 256);
  });

  auto grad_ma = torch::empty({meta.tube_count, 3}, opts_f);
  auto grad_q = torch::empty({meta.tube_count, 6}, opts_f);
  auto grad_opacity = torch::empty({meta.tube_count}, opts_f);
  auto grad_feature = torch::empty({meta.tube_count, meta.reserved0}, opts_f);
  auto grad_color_weight = torch::empty({3, meta.reserved0}, opts_f);
  auto grad_color_bias = torch::empty({3}, opts_f);

  launch(k.clear_feature_direct_gradients, [&](MetalKernelFunction& fn) {
    fn.setArg(0, grad_ma);
    fn.setArg(1, grad_q);
    fn.setArg(2, grad_opacity);
    fn.setArg(3, grad_feature);
    fn.setArg(4, meta_i32);
    fn.dispatch((uint64_t)meta.tube_count, 256);
  });
  launch(k.clear_linear_colorizer_gradients, [&](MetalKernelFunction& fn) {
    fn.setArg(0, grad_color_weight);
    fn.setArg(1, grad_color_bias);
    fn.setArg(2, meta_i32);
    fn.dispatch((uint64_t)std::max<int64_t>(3 * meta.reserved0, 3), 256);
  });

  launch(k.direct_atomic_feature_linear_sigmoid_mse_backward, [&](MetalKernelFunction& fn) {
    fn.setArg(0, ma);
    fn.setArg(1, q_uvt);
    fn.setArg(2, depth0);
    fn.setArg(3, depth_beta);
    fn.setArg(4, opacity);
    fn.setArg(5, feature);
    fn.setArg(6, target_rgb);
    fn.setArg(7, color_weight);
    fn.setArg(8, color_bias);
    fn.setArg(9, meta_i32);
    fn.setArg(10, meta_f32);
    fn.setArg(11, tile_counts);
    fn.setArg(12, tile_tube_ids);
    fn.setArg(13, tile_depths);
    fn.setArg(14, tile_unstable);
    fn.setArg(15, grad_ma);
    fn.setArg(16, grad_q);
    fn.setArg(17, grad_opacity);
    fn.setArg(18, grad_feature);
    fn.setArg(19, grad_color_weight);
    fn.setArg(20, grad_color_bias);
    fn.dispatch((uint64_t)meta.tile_count * (uint64_t)sc.threads, (uint64_t)sc.threads);
  });

  return std::make_tuple(grad_ma, grad_q, grad_opacity, grad_feature, grad_color_weight, grad_color_bias, tile_unstable);
}

std::tuple<torch::Tensor, torch::Tensor> metal_linear_sigmoid_mse_handoff_prep(
    const torch::Tensor& feature_image,
    const torch::Tensor& alpha,
    const torch::Tensor& target_rgb,
    const torch::Tensor& color_weight,
    const torch::Tensor& color_bias) {
  TORCH_CHECK(feature_image.device().is_mps(), "feature_image must be on MPS");
  TORCH_CHECK(feature_image.scalar_type() == torch::kFloat32, "feature_image must be float32");
  TORCH_CHECK(feature_image.dim() == 4, "feature_image must have shape [frames,feature_dim,height,width]");
  TORCH_CHECK(feature_image.is_contiguous(), "feature_image must be contiguous");
  TORCH_CHECK(alpha.device().is_mps(), "alpha must be on MPS");
  TORCH_CHECK(alpha.scalar_type() == torch::kFloat32, "alpha must be float32");
  TORCH_CHECK(alpha.dim() == 3, "alpha must have shape [frames,height,width]");
  TORCH_CHECK(alpha.is_contiguous(), "alpha must be contiguous");
  TORCH_CHECK(target_rgb.device().is_mps(), "target_rgb must be on MPS");
  TORCH_CHECK(target_rgb.scalar_type() == torch::kFloat32, "target_rgb must be float32");
  TORCH_CHECK(target_rgb.dim() == 4 && target_rgb.size(1) == 3, "target_rgb must have shape [frames,3,height,width]");
  TORCH_CHECK(target_rgb.is_contiguous(), "target_rgb must be contiguous");
  TORCH_CHECK(color_weight.device().is_mps(), "color_weight must be on MPS");
  TORCH_CHECK(color_weight.scalar_type() == torch::kFloat32, "color_weight must be float32");
  TORCH_CHECK(color_weight.dim() == 2 && color_weight.size(0) == 3 && color_weight.size(1) == feature_image.size(1),
              "color_weight must have shape [3,feature_dim]");
  TORCH_CHECK(color_weight.is_contiguous(), "color_weight must be contiguous");
  TORCH_CHECK(color_bias.device().is_mps(), "color_bias must be on MPS");
  TORCH_CHECK(color_bias.scalar_type() == torch::kFloat32, "color_bias must be float32");
  TORCH_CHECK(color_bias.dim() == 1 && color_bias.size(0) == 3, "color_bias must have shape [3]");
  TORCH_CHECK(color_bias.is_contiguous(), "color_bias must be contiguous");

  const int64_t frames = feature_image.size(0);
  const int64_t feature_dim = feature_image.size(1);
  const int64_t height = feature_image.size(2);
  const int64_t width = feature_image.size(3);
  TORCH_CHECK(feature_dim > 0 && feature_dim <= 64, "linear sigmoid handoff prep requires feature_dim in 1..64");
  TORCH_CHECK(alpha.size(0) == frames && alpha.size(1) == height && alpha.size(2) == width, "alpha shape must match feature_image");
  TORCH_CHECK(target_rgb.size(0) == frames && target_rgb.size(2) == height && target_rgb.size(3) == width,
              "target_rgb shape must match feature_image");

  auto opts_f = feature_image.options().dtype(torch::kFloat32);
  auto opts_i32 = feature_image.options().dtype(torch::kInt32);
  auto grad_logits = torch::empty({frames, height, width, 3}, opts_f);
  auto grad_alpha = torch::empty({frames, height, width}, opts_f);
  auto prep_meta = torch::tensor(
      {static_cast<int32_t>(frames), static_cast<int32_t>(feature_dim), static_cast<int32_t>(height),
       static_cast<int32_t>(width)},
      opts_i32);
  auto& k = kernels();
  launch(k.linear_sigmoid_mse_handoff_prep, [&](MetalKernelFunction& fn) {
    fn.setArg(0, feature_image);
    fn.setArg(1, alpha);
    fn.setArg(2, target_rgb);
    fn.setArg(3, color_weight);
    fn.setArg(4, color_bias);
    fn.setArg(5, prep_meta);
    fn.setArg(6, grad_logits);
    fn.setArg(7, grad_alpha);
    fn.dispatch((uint64_t)frames * (uint64_t)height * (uint64_t)width, 256);
  });

  return std::make_tuple(grad_logits, grad_alpha);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
metal_direct_atomic_feature_hidden_sigmoid_mse_backward(
    const torch::Tensor& ma,
    const torch::Tensor& q_uvt,
    const torch::Tensor& depth0,
    const torch::Tensor& depth_beta,
    const torch::Tensor& opacity,
    const torch::Tensor& feature,
    const torch::Tensor& target_rgb,
    const torch::Tensor& hidden_weight,
    const torch::Tensor& hidden_bias,
    const torch::Tensor& output_weight,
    const torch::Tensor& output_bias,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32) {
  check_float_mps_2d(ma, "ma", 3);
  check_float_mps_2d(q_uvt, "q_uvt", 6);
  check_float_mps_1d(depth0, "depth0");
  check_float_mps_2d(depth_beta, "depth_beta", 3);
  check_float_mps_1d(opacity, "opacity");
  TORCH_CHECK(feature.device().is_mps(), "feature must be on MPS");
  TORCH_CHECK(feature.scalar_type() == torch::kFloat32, "feature must be float32");
  TORCH_CHECK(feature.dim() == 2, "feature must have shape [N,F]");
  TORCH_CHECK(feature.is_contiguous(), "feature must be contiguous");
  TORCH_CHECK(target_rgb.device().is_mps(), "target_rgb must be on MPS");
  TORCH_CHECK(target_rgb.scalar_type() == torch::kFloat32, "target_rgb must be float32");
  TORCH_CHECK(target_rgb.dim() == 4 && target_rgb.size(3) == 3, "target_rgb must have shape [frames,height,width,3]");
  TORCH_CHECK(target_rgb.is_contiguous(), "target_rgb must be contiguous");
  TORCH_CHECK(hidden_weight.device().is_mps(), "hidden_weight must be on MPS");
  TORCH_CHECK(hidden_weight.scalar_type() == torch::kFloat32, "hidden_weight must be float32");
  TORCH_CHECK(hidden_weight.dim() == 2 && hidden_weight.size(1) == feature.size(1),
              "hidden_weight must have shape [hidden_dim,F]");
  TORCH_CHECK(hidden_weight.is_contiguous(), "hidden_weight must be contiguous");
  check_float_mps_1d(hidden_bias, "hidden_bias");
  TORCH_CHECK(output_weight.device().is_mps(), "output_weight must be on MPS");
  TORCH_CHECK(output_weight.scalar_type() == torch::kFloat32, "output_weight must be float32");
  TORCH_CHECK(output_weight.dim() == 2 && output_weight.size(0) == 3 && output_weight.size(1) == hidden_weight.size(0),
              "output_weight must have shape [3,hidden_dim]");
  TORCH_CHECK(output_weight.is_contiguous(), "output_weight must be contiguous");
  TORCH_CHECK(output_bias.device().is_mps(), "output_bias must be on MPS");
  TORCH_CHECK(output_bias.scalar_type() == torch::kFloat32, "output_bias must be float32");
  TORCH_CHECK(output_bias.dim() == 1 && output_bias.size(0) == 3, "output_bias must have shape [3]");
  TORCH_CHECK(output_bias.is_contiguous(), "output_bias must be contiguous");
  TORCH_CHECK(hidden_bias.size(0) == hidden_weight.size(0), "hidden_bias shape must match hidden_weight");
  TORCH_CHECK(ma.size(0) == q_uvt.size(0) && ma.size(0) == depth0.size(0) && ma.size(0) == depth_beta.size(0) &&
                  ma.size(0) == opacity.size(0) && ma.size(0) == feature.size(0),
              "all tube inputs must agree on N");

  auto meta = parse_meta(meta_i32, meta_f32);
  auto& sc = shader_config();
  check_feature_meta(meta, ma.size(0), sc, feature.size(1));
  TORCH_CHECK(meta.reserved0 <= 64, "hidden sigmoid MSE handoff requires feature_dim <= 64; got ", meta.reserved0);
  TORCH_CHECK(hidden_weight.size(0) > 0 && hidden_weight.size(0) <= 64,
              "hidden sigmoid MSE handoff requires hidden_dim in 1..64; got ", hidden_weight.size(0));
  TORCH_CHECK(target_rgb.size(0) == meta.frames && target_rgb.size(1) == meta.height && target_rgb.size(2) == meta.width,
              "target_rgb shape must match meta");
  auto& k = kernels();

  auto opts_f = ma.options().dtype(torch::kFloat32);
  auto opts_i32 = ma.options().dtype(torch::kInt32);
  auto tile_counts = torch::empty({meta.tile_count}, opts_i32);
  auto tile_overflow = torch::empty({meta.tile_count}, opts_i32);
  auto tile_unstable = torch::empty({meta.tile_count}, opts_i32);
  auto tile_tube_ids = torch::empty({meta.tile_count * meta.tile_capacity}, opts_i32);
  auto tile_depths = torch::empty({meta.tile_count * meta.tile_capacity}, opts_f);
  auto hidden_meta = torch::tensor({static_cast<int32_t>(hidden_weight.size(0))}, opts_i32);

  launch(k.clear_tiles, [&](MetalKernelFunction& fn) {
    fn.setArg(0, tile_counts);
    fn.setArg(1, tile_overflow);
    fn.setArg(2, tile_unstable);
    fn.setArg(3, meta_i32);
    fn.dispatch((uint64_t)meta.tile_count, 256);
  });

  launch(k.bin_tubes, [&](MetalKernelFunction& fn) {
    fn.setArg(0, ma);
    fn.setArg(1, q_uvt);
    fn.setArg(2, depth0);
    fn.setArg(3, depth_beta);
    fn.setArg(4, opacity);
    fn.setArg(5, meta_i32);
    fn.setArg(6, meta_f32);
    fn.setArg(7, tile_counts);
    fn.setArg(8, tile_tube_ids);
    fn.setArg(9, tile_depths);
    fn.setArg(10, tile_overflow);
    fn.dispatch((uint64_t)meta.tube_count, 256);
  });

  auto grad_ma = torch::empty({meta.tube_count, 3}, opts_f);
  auto grad_q = torch::empty({meta.tube_count, 6}, opts_f);
  auto grad_opacity = torch::empty({meta.tube_count}, opts_f);
  auto grad_feature = torch::empty({meta.tube_count, meta.reserved0}, opts_f);

  launch(k.clear_feature_direct_gradients, [&](MetalKernelFunction& fn) {
    fn.setArg(0, grad_ma);
    fn.setArg(1, grad_q);
    fn.setArg(2, grad_opacity);
    fn.setArg(3, grad_feature);
    fn.setArg(4, meta_i32);
    fn.dispatch((uint64_t)meta.tube_count, 256);
  });

  launch(k.direct_atomic_feature_hidden_sigmoid_mse_backward, [&](MetalKernelFunction& fn) {
    fn.setArg(0, ma);
    fn.setArg(1, q_uvt);
    fn.setArg(2, depth0);
    fn.setArg(3, depth_beta);
    fn.setArg(4, opacity);
    fn.setArg(5, feature);
    fn.setArg(6, target_rgb);
    fn.setArg(7, hidden_weight);
    fn.setArg(8, hidden_bias);
    fn.setArg(9, output_weight);
    fn.setArg(10, output_bias);
    fn.setArg(11, hidden_meta);
    fn.setArg(12, meta_i32);
    fn.setArg(13, meta_f32);
    fn.setArg(14, tile_counts);
    fn.setArg(15, tile_tube_ids);
    fn.setArg(16, tile_depths);
    fn.setArg(17, tile_unstable);
    fn.setArg(18, grad_ma);
    fn.setArg(19, grad_q);
    fn.setArg(20, grad_opacity);
    fn.setArg(21, grad_feature);
    fn.dispatch((uint64_t)meta.tile_count * (uint64_t)sc.threads, (uint64_t)sc.threads);
  });

  return std::make_tuple(grad_ma, grad_q, grad_opacity, grad_feature, tile_unstable);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
metal_direct_atomic_feature_logit_handoff_backward(
    const torch::Tensor& ma,
    const torch::Tensor& q_uvt,
    const torch::Tensor& depth0,
    const torch::Tensor& depth_beta,
    const torch::Tensor& opacity,
    const torch::Tensor& feature,
    const torch::Tensor& grad_logits,
    const torch::Tensor& grad_alpha_image,
    const torch::Tensor& color_weight,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32) {
  check_float_mps_2d(ma, "ma", 3);
  check_float_mps_2d(q_uvt, "q_uvt", 6);
  check_float_mps_1d(depth0, "depth0");
  check_float_mps_2d(depth_beta, "depth_beta", 3);
  check_float_mps_1d(opacity, "opacity");
  TORCH_CHECK(feature.device().is_mps(), "feature must be on MPS");
  TORCH_CHECK(feature.scalar_type() == torch::kFloat32, "feature must be float32");
  TORCH_CHECK(feature.dim() == 2, "feature must have shape [N,F]");
  TORCH_CHECK(feature.is_contiguous(), "feature must be contiguous");
  TORCH_CHECK(grad_logits.device().is_mps(), "grad_logits must be on MPS");
  TORCH_CHECK(grad_logits.scalar_type() == torch::kFloat32, "grad_logits must be float32");
  TORCH_CHECK(grad_logits.dim() == 4 && grad_logits.size(3) == 3, "grad_logits must have shape [frames,height,width,3]");
  TORCH_CHECK(grad_logits.is_contiguous(), "grad_logits must be contiguous");
  TORCH_CHECK(grad_alpha_image.device().is_mps(), "grad_alpha_image must be on MPS");
  TORCH_CHECK(grad_alpha_image.scalar_type() == torch::kFloat32, "grad_alpha_image must be float32");
  TORCH_CHECK(grad_alpha_image.dim() == 3, "grad_alpha_image must have shape [frames,height,width]");
  TORCH_CHECK(grad_alpha_image.is_contiguous(), "grad_alpha_image must be contiguous");
  TORCH_CHECK(color_weight.device().is_mps(), "color_weight must be on MPS");
  TORCH_CHECK(color_weight.scalar_type() == torch::kFloat32, "color_weight must be float32");
  TORCH_CHECK(color_weight.dim() == 2 && color_weight.size(0) == 3 && color_weight.size(1) == feature.size(1),
              "color_weight must have shape [3,F]");
  TORCH_CHECK(color_weight.is_contiguous(), "color_weight must be contiguous");
  TORCH_CHECK(ma.size(0) == q_uvt.size(0) && ma.size(0) == depth0.size(0) && ma.size(0) == depth_beta.size(0) &&
                  ma.size(0) == opacity.size(0) && ma.size(0) == feature.size(0),
              "all tube inputs must agree on N");

  auto meta = parse_meta(meta_i32, meta_f32);
  auto& sc = shader_config();
  check_feature_meta(meta, ma.size(0), sc, feature.size(1));
  TORCH_CHECK(meta.reserved0 <= 64, "logit handoff requires feature_dim <= 64; got ", meta.reserved0);
  TORCH_CHECK(grad_logits.size(0) == meta.frames && grad_logits.size(1) == meta.height && grad_logits.size(2) == meta.width,
              "grad_logits shape must match meta");
  TORCH_CHECK(grad_alpha_image.size(0) == meta.frames && grad_alpha_image.size(1) == meta.height &&
                  grad_alpha_image.size(2) == meta.width,
              "grad_alpha_image shape must match meta");
  auto& k = kernels();

  auto opts_f = ma.options().dtype(torch::kFloat32);
  auto opts_i32 = ma.options().dtype(torch::kInt32);
  auto tile_counts = torch::empty({meta.tile_count}, opts_i32);
  auto tile_overflow = torch::empty({meta.tile_count}, opts_i32);
  auto tile_unstable = torch::empty({meta.tile_count}, opts_i32);
  auto tile_tube_ids = torch::empty({meta.tile_count * meta.tile_capacity}, opts_i32);
  auto tile_depths = torch::empty({meta.tile_count * meta.tile_capacity}, opts_f);

  launch(k.clear_tiles, [&](MetalKernelFunction& fn) {
    fn.setArg(0, tile_counts);
    fn.setArg(1, tile_overflow);
    fn.setArg(2, tile_unstable);
    fn.setArg(3, meta_i32);
    fn.dispatch((uint64_t)meta.tile_count, 256);
  });

  launch(k.bin_tubes, [&](MetalKernelFunction& fn) {
    fn.setArg(0, ma);
    fn.setArg(1, q_uvt);
    fn.setArg(2, depth0);
    fn.setArg(3, depth_beta);
    fn.setArg(4, opacity);
    fn.setArg(5, meta_i32);
    fn.setArg(6, meta_f32);
    fn.setArg(7, tile_counts);
    fn.setArg(8, tile_tube_ids);
    fn.setArg(9, tile_depths);
    fn.setArg(10, tile_overflow);
    fn.dispatch((uint64_t)meta.tube_count, 256);
  });

  auto grad_ma = torch::empty({meta.tube_count, 3}, opts_f);
  auto grad_q = torch::empty({meta.tube_count, 6}, opts_f);
  auto grad_opacity = torch::empty({meta.tube_count}, opts_f);
  auto grad_feature = torch::empty({meta.tube_count, meta.reserved0}, opts_f);

  launch(k.clear_feature_direct_gradients, [&](MetalKernelFunction& fn) {
    fn.setArg(0, grad_ma);
    fn.setArg(1, grad_q);
    fn.setArg(2, grad_opacity);
    fn.setArg(3, grad_feature);
    fn.setArg(4, meta_i32);
    fn.dispatch((uint64_t)meta.tube_count, 256);
  });

  launch(k.direct_atomic_feature_logit_handoff_backward, [&](MetalKernelFunction& fn) {
    fn.setArg(0, ma);
    fn.setArg(1, q_uvt);
    fn.setArg(2, depth0);
    fn.setArg(3, depth_beta);
    fn.setArg(4, opacity);
    fn.setArg(5, feature);
    fn.setArg(6, grad_logits);
    fn.setArg(7, grad_alpha_image);
    fn.setArg(8, color_weight);
    fn.setArg(9, meta_i32);
    fn.setArg(10, meta_f32);
    fn.setArg(11, tile_counts);
    fn.setArg(12, tile_tube_ids);
    fn.setArg(13, tile_depths);
    fn.setArg(14, tile_unstable);
    fn.setArg(15, grad_ma);
    fn.setArg(16, grad_q);
    fn.setArg(17, grad_opacity);
    fn.setArg(18, grad_feature);
    fn.dispatch((uint64_t)meta.tile_count * (uint64_t)sc.threads, (uint64_t)sc.threads);
  });

  return std::make_tuple(grad_ma, grad_q, grad_opacity, grad_feature, tile_unstable);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> metal_direct_fixedpoint_backward(
    const torch::Tensor& ma,
    const torch::Tensor& q_uvt,
    const torch::Tensor& depth0,
    const torch::Tensor& depth_beta,
    const torch::Tensor& opacity,
    const torch::Tensor& color,
    const torch::Tensor& grad_image,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32) {
  check_float_mps_2d(ma, "ma", 3);
  check_float_mps_2d(q_uvt, "q_uvt", 6);
  check_float_mps_1d(depth0, "depth0");
  check_float_mps_2d(depth_beta, "depth_beta", 3);
  check_float_mps_1d(opacity, "opacity");
  check_float_mps_2d(color, "color", 3);
  TORCH_CHECK(grad_image.device().is_mps(), "grad_image must be on MPS");
  TORCH_CHECK(grad_image.scalar_type() == torch::kFloat32, "grad_image must be float32");
  TORCH_CHECK(grad_image.dim() == 4 && grad_image.size(3) == 3, "grad_image must have shape [F,H,W,3]");
  TORCH_CHECK(grad_image.is_contiguous(), "grad_image must be contiguous");
  TORCH_CHECK(ma.size(0) == q_uvt.size(0) && ma.size(0) == depth0.size(0) && ma.size(0) == depth_beta.size(0) &&
                  ma.size(0) == opacity.size(0) && ma.size(0) == color.size(0),
              "all tube inputs must agree on N");

  auto meta = parse_meta(meta_i32, meta_f32);
  auto& sc = shader_config();
  check_meta(meta, ma.size(0), sc);
  TORCH_CHECK(grad_image.size(0) == meta.frames && grad_image.size(1) == meta.height && grad_image.size(2) == meta.width,
              "grad_image shape must match meta");
  auto& k = kernels();

  auto opts_f = ma.options().dtype(torch::kFloat32);
  auto opts_i32 = ma.options().dtype(torch::kInt32);
  auto tile_counts = torch::empty({meta.tile_count}, opts_i32);
  auto tile_overflow = torch::empty({meta.tile_count}, opts_i32);
  auto tile_unstable = torch::empty({meta.tile_count}, opts_i32);
  auto tile_tube_ids = torch::empty({meta.tile_count * meta.tile_capacity}, opts_i32);
  auto tile_depths = torch::empty({meta.tile_count * meta.tile_capacity}, opts_f);

  launch(k.clear_tiles, [&](MetalKernelFunction& fn) {
    fn.setArg(0, tile_counts);
    fn.setArg(1, tile_overflow);
    fn.setArg(2, tile_unstable);
    fn.setArg(3, meta_i32);
    fn.dispatch((uint64_t)meta.tile_count, 256);
  });

  launch(k.bin_tubes, [&](MetalKernelFunction& fn) {
    fn.setArg(0, ma);
    fn.setArg(1, q_uvt);
    fn.setArg(2, depth0);
    fn.setArg(3, depth_beta);
    fn.setArg(4, opacity);
    fn.setArg(5, meta_i32);
    fn.setArg(6, meta_f32);
    fn.setArg(7, tile_counts);
    fn.setArg(8, tile_tube_ids);
    fn.setArg(9, tile_depths);
    fn.setArg(10, tile_overflow);
    fn.dispatch((uint64_t)meta.tube_count, 256);
  });

  auto grad_ma_i32 = torch::empty({meta.tube_count, 3}, opts_i32);
  auto grad_q_i32 = torch::empty({meta.tube_count, 6}, opts_i32);
  auto grad_opacity_i32 = torch::empty({meta.tube_count}, opts_i32);
  auto grad_color_i32 = torch::empty({meta.tube_count, 3}, opts_i32);
  auto grad_ma = torch::empty({meta.tube_count, 3}, opts_f);
  auto grad_q = torch::empty({meta.tube_count, 6}, opts_f);
  auto grad_opacity = torch::empty({meta.tube_count}, opts_f);
  auto grad_color = torch::empty({meta.tube_count, 3}, opts_f);

  launch(k.clear_direct_gradients_i32, [&](MetalKernelFunction& fn) {
    fn.setArg(0, grad_ma_i32);
    fn.setArg(1, grad_q_i32);
    fn.setArg(2, grad_opacity_i32);
    fn.setArg(3, grad_color_i32);
    fn.setArg(4, meta_i32);
    fn.dispatch((uint64_t)meta.tube_count, 256);
  });

  launch(k.direct_fixedpoint_backward, [&](MetalKernelFunction& fn) {
    fn.setArg(0, ma);
    fn.setArg(1, q_uvt);
    fn.setArg(2, depth0);
    fn.setArg(3, depth_beta);
    fn.setArg(4, opacity);
    fn.setArg(5, color);
    fn.setArg(6, grad_image);
    fn.setArg(7, meta_i32);
    fn.setArg(8, meta_f32);
    fn.setArg(9, tile_counts);
    fn.setArg(10, tile_tube_ids);
    fn.setArg(11, tile_depths);
    fn.setArg(12, tile_unstable);
    fn.setArg(13, grad_ma_i32);
    fn.setArg(14, grad_q_i32);
    fn.setArg(15, grad_opacity_i32);
    fn.setArg(16, grad_color_i32);
    fn.dispatch((uint64_t)meta.tile_count * (uint64_t)sc.threads, (uint64_t)sc.threads);
  });

  launch(k.fixedpoint_gradients_to_float, [&](MetalKernelFunction& fn) {
    fn.setArg(0, grad_ma_i32);
    fn.setArg(1, grad_q_i32);
    fn.setArg(2, grad_opacity_i32);
    fn.setArg(3, grad_color_i32);
    fn.setArg(4, meta_i32);
    fn.setArg(5, grad_ma);
    fn.setArg(6, grad_q);
    fn.setArg(7, grad_opacity);
    fn.setArg(8, grad_color);
    fn.dispatch((uint64_t)meta.tube_count, 256);
  });

  return std::make_tuple(grad_ma, grad_q, grad_opacity, grad_color, tile_unstable);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> metal_tile_pair_atomic_backward(
    const torch::Tensor& ma,
    const torch::Tensor& q_uvt,
    const torch::Tensor& depth0,
    const torch::Tensor& depth_beta,
    const torch::Tensor& opacity,
    const torch::Tensor& color,
    const torch::Tensor& grad_image,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32) {
  check_float_mps_2d(ma, "ma", 3);
  check_float_mps_2d(q_uvt, "q_uvt", 6);
  check_float_mps_1d(depth0, "depth0");
  check_float_mps_2d(depth_beta, "depth_beta", 3);
  check_float_mps_1d(opacity, "opacity");
  check_float_mps_2d(color, "color", 3);
  TORCH_CHECK(grad_image.device().is_mps(), "grad_image must be on MPS");
  TORCH_CHECK(grad_image.scalar_type() == torch::kFloat32, "grad_image must be float32");
  TORCH_CHECK(grad_image.dim() == 4 && grad_image.size(3) == 3, "grad_image must have shape [F,H,W,3]");
  TORCH_CHECK(grad_image.is_contiguous(), "grad_image must be contiguous");
  TORCH_CHECK(ma.size(0) == q_uvt.size(0) && ma.size(0) == depth0.size(0) && ma.size(0) == depth_beta.size(0) &&
                  ma.size(0) == opacity.size(0) && ma.size(0) == color.size(0),
              "all tube inputs must agree on N");

  auto meta = parse_meta(meta_i32, meta_f32);
  auto& sc = shader_config();
  check_meta(meta, ma.size(0), sc);
  TORCH_CHECK(grad_image.size(0) == meta.frames && grad_image.size(1) == meta.height && grad_image.size(2) == meta.width,
              "grad_image shape must match meta");
  auto& k = kernels();

  auto opts_f = ma.options().dtype(torch::kFloat32);
  auto opts_i32 = ma.options().dtype(torch::kInt32);
  auto tile_counts = torch::empty({meta.tile_count}, opts_i32);
  auto tile_overflow = torch::empty({meta.tile_count}, opts_i32);
  auto tile_unstable = torch::empty({meta.tile_count}, opts_i32);
  auto tile_tube_ids = torch::empty({meta.tile_count * meta.tile_capacity}, opts_i32);
  auto tile_depths = torch::empty({meta.tile_count * meta.tile_capacity}, opts_f);

  launch(k.clear_tiles, [&](MetalKernelFunction& fn) {
    fn.setArg(0, tile_counts);
    fn.setArg(1, tile_overflow);
    fn.setArg(2, tile_unstable);
    fn.setArg(3, meta_i32);
    fn.dispatch((uint64_t)meta.tile_count, 256);
  });

  launch(k.bin_tubes, [&](MetalKernelFunction& fn) {
    fn.setArg(0, ma);
    fn.setArg(1, q_uvt);
    fn.setArg(2, depth0);
    fn.setArg(3, depth_beta);
    fn.setArg(4, opacity);
    fn.setArg(5, meta_i32);
    fn.setArg(6, meta_f32);
    fn.setArg(7, tile_counts);
    fn.setArg(8, tile_tube_ids);
    fn.setArg(9, tile_depths);
    fn.setArg(10, tile_overflow);
    fn.dispatch((uint64_t)meta.tube_count, 256);
  });

  auto grad_ma = torch::empty({meta.tube_count, 3}, opts_f);
  auto grad_q = torch::empty({meta.tube_count, 6}, opts_f);
  auto grad_opacity = torch::empty({meta.tube_count}, opts_f);
  auto grad_color = torch::empty({meta.tube_count, 3}, opts_f);

  launch(k.clear_direct_gradients, [&](MetalKernelFunction& fn) {
    fn.setArg(0, grad_ma);
    fn.setArg(1, grad_q);
    fn.setArg(2, grad_opacity);
    fn.setArg(3, grad_color);
    fn.setArg(4, meta_i32);
    fn.dispatch((uint64_t)meta.tube_count, 256);
  });

  int64_t entry_count = (int64_t)meta.tile_count * (int64_t)meta.tile_capacity;
  launch(k.tile_pair_atomic_backward, [&](MetalKernelFunction& fn) {
    fn.setArg(0, ma);
    fn.setArg(1, q_uvt);
    fn.setArg(2, depth0);
    fn.setArg(3, depth_beta);
    fn.setArg(4, opacity);
    fn.setArg(5, color);
    fn.setArg(6, grad_image);
    fn.setArg(7, meta_i32);
    fn.setArg(8, meta_f32);
    fn.setArg(9, tile_counts);
    fn.setArg(10, tile_tube_ids);
    fn.setArg(11, tile_depths);
    fn.setArg(12, tile_unstable);
    fn.setArg(13, grad_ma);
    fn.setArg(14, grad_q);
    fn.setArg(15, grad_opacity);
    fn.setArg(16, grad_color);
    fn.dispatch((uint64_t)entry_count, 256);
  });

  return std::make_tuple(grad_ma, grad_q, grad_opacity, grad_color, tile_unstable);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> metal_tile_pair_fixedpoint_backward(
    const torch::Tensor& ma,
    const torch::Tensor& q_uvt,
    const torch::Tensor& depth0,
    const torch::Tensor& depth_beta,
    const torch::Tensor& opacity,
    const torch::Tensor& color,
    const torch::Tensor& grad_image,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32) {
  check_float_mps_2d(ma, "ma", 3);
  check_float_mps_2d(q_uvt, "q_uvt", 6);
  check_float_mps_1d(depth0, "depth0");
  check_float_mps_2d(depth_beta, "depth_beta", 3);
  check_float_mps_1d(opacity, "opacity");
  check_float_mps_2d(color, "color", 3);
  TORCH_CHECK(grad_image.device().is_mps(), "grad_image must be on MPS");
  TORCH_CHECK(grad_image.scalar_type() == torch::kFloat32, "grad_image must be float32");
  TORCH_CHECK(grad_image.dim() == 4 && grad_image.size(3) == 3, "grad_image must have shape [F,H,W,3]");
  TORCH_CHECK(grad_image.is_contiguous(), "grad_image must be contiguous");
  TORCH_CHECK(ma.size(0) == q_uvt.size(0) && ma.size(0) == depth0.size(0) && ma.size(0) == depth_beta.size(0) &&
                  ma.size(0) == opacity.size(0) && ma.size(0) == color.size(0),
              "all tube inputs must agree on N");

  auto meta = parse_meta(meta_i32, meta_f32);
  auto& sc = shader_config();
  check_meta(meta, ma.size(0), sc);
  TORCH_CHECK(grad_image.size(0) == meta.frames && grad_image.size(1) == meta.height && grad_image.size(2) == meta.width,
              "grad_image shape must match meta");
  auto& k = kernels();

  auto opts_f = ma.options().dtype(torch::kFloat32);
  auto opts_i32 = ma.options().dtype(torch::kInt32);
  auto tile_counts = torch::empty({meta.tile_count}, opts_i32);
  auto tile_overflow = torch::empty({meta.tile_count}, opts_i32);
  auto tile_unstable = torch::empty({meta.tile_count}, opts_i32);
  auto tile_tube_ids = torch::empty({meta.tile_count * meta.tile_capacity}, opts_i32);
  auto tile_depths = torch::empty({meta.tile_count * meta.tile_capacity}, opts_f);

  launch(k.clear_tiles, [&](MetalKernelFunction& fn) {
    fn.setArg(0, tile_counts);
    fn.setArg(1, tile_overflow);
    fn.setArg(2, tile_unstable);
    fn.setArg(3, meta_i32);
    fn.dispatch((uint64_t)meta.tile_count, 256);
  });

  launch(k.bin_tubes, [&](MetalKernelFunction& fn) {
    fn.setArg(0, ma);
    fn.setArg(1, q_uvt);
    fn.setArg(2, depth0);
    fn.setArg(3, depth_beta);
    fn.setArg(4, opacity);
    fn.setArg(5, meta_i32);
    fn.setArg(6, meta_f32);
    fn.setArg(7, tile_counts);
    fn.setArg(8, tile_tube_ids);
    fn.setArg(9, tile_depths);
    fn.setArg(10, tile_overflow);
    fn.dispatch((uint64_t)meta.tube_count, 256);
  });

  auto grad_ma_i32 = torch::empty({meta.tube_count, 3}, opts_i32);
  auto grad_q_i32 = torch::empty({meta.tube_count, 6}, opts_i32);
  auto grad_opacity_i32 = torch::empty({meta.tube_count}, opts_i32);
  auto grad_color_i32 = torch::empty({meta.tube_count, 3}, opts_i32);
  auto grad_ma = torch::empty({meta.tube_count, 3}, opts_f);
  auto grad_q = torch::empty({meta.tube_count, 6}, opts_f);
  auto grad_opacity = torch::empty({meta.tube_count}, opts_f);
  auto grad_color = torch::empty({meta.tube_count, 3}, opts_f);

  launch(k.clear_direct_gradients_i32, [&](MetalKernelFunction& fn) {
    fn.setArg(0, grad_ma_i32);
    fn.setArg(1, grad_q_i32);
    fn.setArg(2, grad_opacity_i32);
    fn.setArg(3, grad_color_i32);
    fn.setArg(4, meta_i32);
    fn.dispatch((uint64_t)meta.tube_count, 256);
  });

  int64_t entry_count = (int64_t)meta.tile_count * (int64_t)meta.tile_capacity;
  launch(k.tile_pair_fixedpoint_backward, [&](MetalKernelFunction& fn) {
    fn.setArg(0, ma);
    fn.setArg(1, q_uvt);
    fn.setArg(2, depth0);
    fn.setArg(3, depth_beta);
    fn.setArg(4, opacity);
    fn.setArg(5, color);
    fn.setArg(6, grad_image);
    fn.setArg(7, meta_i32);
    fn.setArg(8, meta_f32);
    fn.setArg(9, tile_counts);
    fn.setArg(10, tile_tube_ids);
    fn.setArg(11, tile_depths);
    fn.setArg(12, tile_unstable);
    fn.setArg(13, grad_ma_i32);
    fn.setArg(14, grad_q_i32);
    fn.setArg(15, grad_opacity_i32);
    fn.setArg(16, grad_color_i32);
    fn.dispatch((uint64_t)entry_count, 256);
  });

  launch(k.fixedpoint_gradients_to_float, [&](MetalKernelFunction& fn) {
    fn.setArg(0, grad_ma_i32);
    fn.setArg(1, grad_q_i32);
    fn.setArg(2, grad_opacity_i32);
    fn.setArg(3, grad_color_i32);
    fn.setArg(4, meta_i32);
    fn.setArg(5, grad_ma);
    fn.setArg(6, grad_q);
    fn.setArg(7, grad_opacity);
    fn.setArg(8, grad_color);
    fn.dispatch((uint64_t)meta.tube_count, 256);
  });

  return std::make_tuple(grad_ma, grad_q, grad_opacity, grad_color, tile_unstable);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> metal_direct_split_fixedpoint_backward(
    const torch::Tensor& ma,
    const torch::Tensor& q_uvt,
    const torch::Tensor& depth0,
    const torch::Tensor& depth_beta,
    const torch::Tensor& opacity,
    const torch::Tensor& color,
    const torch::Tensor& grad_image,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32) {
  check_float_mps_2d(ma, "ma", 3);
  check_float_mps_2d(q_uvt, "q_uvt", 6);
  check_float_mps_1d(depth0, "depth0");
  check_float_mps_2d(depth_beta, "depth_beta", 3);
  check_float_mps_1d(opacity, "opacity");
  check_float_mps_2d(color, "color", 3);
  TORCH_CHECK(grad_image.device().is_mps(), "grad_image must be on MPS");
  TORCH_CHECK(grad_image.scalar_type() == torch::kFloat32, "grad_image must be float32");
  TORCH_CHECK(grad_image.dim() == 4 && grad_image.size(3) == 3, "grad_image must have shape [F,H,W,3]");
  TORCH_CHECK(grad_image.is_contiguous(), "grad_image must be contiguous");
  TORCH_CHECK(ma.size(0) == q_uvt.size(0) && ma.size(0) == depth0.size(0) && ma.size(0) == depth_beta.size(0) &&
                  ma.size(0) == opacity.size(0) && ma.size(0) == color.size(0),
              "all tube inputs must agree on N");

  auto meta = parse_meta(meta_i32, meta_f32);
  auto& sc = shader_config();
  check_meta(meta, ma.size(0), sc);
  TORCH_CHECK(grad_image.size(0) == meta.frames && grad_image.size(1) == meta.height && grad_image.size(2) == meta.width,
              "grad_image shape must match meta");
  auto& k = kernels();

  auto opts_f = ma.options().dtype(torch::kFloat32);
  auto opts_i32 = ma.options().dtype(torch::kInt32);
  auto tile_counts = torch::empty({meta.tile_count}, opts_i32);
  auto tile_overflow = torch::empty({meta.tile_count}, opts_i32);
  auto tile_unstable = torch::empty({meta.tile_count}, opts_i32);
  auto tile_tube_ids = torch::empty({meta.tile_count * meta.tile_capacity}, opts_i32);
  auto tile_depths = torch::empty({meta.tile_count * meta.tile_capacity}, opts_f);

  launch(k.clear_tiles, [&](MetalKernelFunction& fn) {
    fn.setArg(0, tile_counts);
    fn.setArg(1, tile_overflow);
    fn.setArg(2, tile_unstable);
    fn.setArg(3, meta_i32);
    fn.dispatch((uint64_t)meta.tile_count, 256);
  });

  launch(k.bin_tubes, [&](MetalKernelFunction& fn) {
    fn.setArg(0, ma);
    fn.setArg(1, q_uvt);
    fn.setArg(2, depth0);
    fn.setArg(3, depth_beta);
    fn.setArg(4, opacity);
    fn.setArg(5, meta_i32);
    fn.setArg(6, meta_f32);
    fn.setArg(7, tile_counts);
    fn.setArg(8, tile_tube_ids);
    fn.setArg(9, tile_depths);
    fn.setArg(10, tile_overflow);
    fn.dispatch((uint64_t)meta.tube_count, 256);
  });

  auto grad_ma_coarse = torch::empty({meta.tube_count, 3}, opts_i32);
  auto grad_q_coarse = torch::empty({meta.tube_count, 6}, opts_i32);
  auto grad_opacity_coarse = torch::empty({meta.tube_count}, opts_i32);
  auto grad_color_coarse = torch::empty({meta.tube_count, 3}, opts_i32);
  auto grad_ma_fine = torch::empty({meta.tube_count, 3}, opts_i32);
  auto grad_q_fine = torch::empty({meta.tube_count, 6}, opts_i32);
  auto grad_opacity_fine = torch::empty({meta.tube_count}, opts_i32);
  auto grad_color_fine = torch::empty({meta.tube_count, 3}, opts_i32);
  auto grad_ma = torch::empty({meta.tube_count, 3}, opts_f);
  auto grad_q = torch::empty({meta.tube_count, 6}, opts_f);
  auto grad_opacity = torch::empty({meta.tube_count}, opts_f);
  auto grad_color = torch::empty({meta.tube_count, 3}, opts_f);

  launch(k.clear_direct_gradients_i32, [&](MetalKernelFunction& fn) {
    fn.setArg(0, grad_ma_coarse);
    fn.setArg(1, grad_q_coarse);
    fn.setArg(2, grad_opacity_coarse);
    fn.setArg(3, grad_color_coarse);
    fn.setArg(4, meta_i32);
    fn.dispatch((uint64_t)meta.tube_count, 256);
  });

  launch(k.clear_direct_gradients_i32, [&](MetalKernelFunction& fn) {
    fn.setArg(0, grad_ma_fine);
    fn.setArg(1, grad_q_fine);
    fn.setArg(2, grad_opacity_fine);
    fn.setArg(3, grad_color_fine);
    fn.setArg(4, meta_i32);
    fn.dispatch((uint64_t)meta.tube_count, 256);
  });

  launch(k.direct_split_fixedpoint_backward, [&](MetalKernelFunction& fn) {
    fn.setArg(0, ma);
    fn.setArg(1, q_uvt);
    fn.setArg(2, depth0);
    fn.setArg(3, depth_beta);
    fn.setArg(4, opacity);
    fn.setArg(5, color);
    fn.setArg(6, grad_image);
    fn.setArg(7, meta_i32);
    fn.setArg(8, meta_f32);
    fn.setArg(9, tile_counts);
    fn.setArg(10, tile_tube_ids);
    fn.setArg(11, tile_depths);
    fn.setArg(12, tile_unstable);
    fn.setArg(13, grad_ma_coarse);
    fn.setArg(14, grad_q_coarse);
    fn.setArg(15, grad_opacity_coarse);
    fn.setArg(16, grad_color_coarse);
    fn.setArg(17, grad_ma_fine);
    fn.setArg(18, grad_q_fine);
    fn.setArg(19, grad_opacity_fine);
    fn.setArg(20, grad_color_fine);
    fn.dispatch((uint64_t)meta.tile_count * (uint64_t)sc.threads, (uint64_t)sc.threads);
  });

  launch(k.split_fixedpoint_gradients_to_float, [&](MetalKernelFunction& fn) {
    fn.setArg(0, grad_ma_coarse);
    fn.setArg(1, grad_q_coarse);
    fn.setArg(2, grad_opacity_coarse);
    fn.setArg(3, grad_color_coarse);
    fn.setArg(4, grad_ma_fine);
    fn.setArg(5, grad_q_fine);
    fn.setArg(6, grad_opacity_fine);
    fn.setArg(7, grad_color_fine);
    fn.setArg(8, meta_i32);
    fn.setArg(9, grad_ma);
    fn.setArg(10, grad_q);
    fn.setArg(11, grad_opacity);
    fn.setArg(12, grad_color);
    fn.dispatch((uint64_t)meta.tube_count, 256);
  });

  return std::make_tuple(grad_ma, grad_q, grad_opacity, grad_color, tile_unstable);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> metal_direct_serial_backward(
    const torch::Tensor& ma,
    const torch::Tensor& q_uvt,
    const torch::Tensor& depth0,
    const torch::Tensor& depth_beta,
    const torch::Tensor& opacity,
    const torch::Tensor& color,
    const torch::Tensor& grad_image,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32) {
  check_float_mps_2d(ma, "ma", 3);
  check_float_mps_2d(q_uvt, "q_uvt", 6);
  check_float_mps_1d(depth0, "depth0");
  check_float_mps_2d(depth_beta, "depth_beta", 3);
  check_float_mps_1d(opacity, "opacity");
  check_float_mps_2d(color, "color", 3);
  TORCH_CHECK(grad_image.device().is_mps(), "grad_image must be on MPS");
  TORCH_CHECK(grad_image.scalar_type() == torch::kFloat32, "grad_image must be float32");
  TORCH_CHECK(grad_image.dim() == 4 && grad_image.size(3) == 3, "grad_image must have shape [F,H,W,3]");
  TORCH_CHECK(grad_image.is_contiguous(), "grad_image must be contiguous");
  TORCH_CHECK(ma.size(0) == q_uvt.size(0) && ma.size(0) == depth0.size(0) && ma.size(0) == depth_beta.size(0) &&
                  ma.size(0) == opacity.size(0) && ma.size(0) == color.size(0),
              "all tube inputs must agree on N");

  auto meta = parse_meta(meta_i32, meta_f32);
  auto& sc = shader_config();
  check_meta(meta, ma.size(0), sc);
  TORCH_CHECK(grad_image.size(0) == meta.frames && grad_image.size(1) == meta.height && grad_image.size(2) == meta.width,
              "grad_image shape must match meta");
  auto& k = kernels();

  auto opts_f = ma.options().dtype(torch::kFloat32);
  auto opts_i32 = ma.options().dtype(torch::kInt32);
  auto tile_counts = torch::empty({meta.tile_count}, opts_i32);
  auto tile_overflow = torch::empty({meta.tile_count}, opts_i32);
  auto tile_unstable = torch::empty({meta.tile_count}, opts_i32);
  auto tile_tube_ids = torch::empty({meta.tile_count * meta.tile_capacity}, opts_i32);
  auto tile_depths = torch::empty({meta.tile_count * meta.tile_capacity}, opts_f);

  launch(k.clear_tiles, [&](MetalKernelFunction& fn) {
    fn.setArg(0, tile_counts);
    fn.setArg(1, tile_overflow);
    fn.setArg(2, tile_unstable);
    fn.setArg(3, meta_i32);
    fn.dispatch((uint64_t)meta.tile_count, 256);
  });

  launch(k.bin_tubes, [&](MetalKernelFunction& fn) {
    fn.setArg(0, ma);
    fn.setArg(1, q_uvt);
    fn.setArg(2, depth0);
    fn.setArg(3, depth_beta);
    fn.setArg(4, opacity);
    fn.setArg(5, meta_i32);
    fn.setArg(6, meta_f32);
    fn.setArg(7, tile_counts);
    fn.setArg(8, tile_tube_ids);
    fn.setArg(9, tile_depths);
    fn.setArg(10, tile_overflow);
    fn.dispatch((uint64_t)meta.tube_count, 256);
  });

  auto grad_ma = torch::empty({meta.tube_count, 3}, opts_f);
  auto grad_q = torch::empty({meta.tube_count, 6}, opts_f);
  auto grad_opacity = torch::empty({meta.tube_count}, opts_f);
  auto grad_color = torch::empty({meta.tube_count, 3}, opts_f);

  launch(k.direct_serial_backward, [&](MetalKernelFunction& fn) {
    fn.setArg(0, ma);
    fn.setArg(1, q_uvt);
    fn.setArg(2, depth0);
    fn.setArg(3, depth_beta);
    fn.setArg(4, opacity);
    fn.setArg(5, color);
    fn.setArg(6, grad_image);
    fn.setArg(7, meta_i32);
    fn.setArg(8, meta_f32);
    fn.setArg(9, tile_counts);
    fn.setArg(10, tile_tube_ids);
    fn.setArg(11, tile_depths);
    fn.setArg(12, tile_unstable);
    fn.setArg(13, grad_ma);
    fn.setArg(14, grad_q);
    fn.setArg(15, grad_opacity);
    fn.setArg(16, grad_color);
    fn.dispatch((uint64_t)meta.tube_count, 256);
  });

  return std::make_tuple(grad_ma, grad_q, grad_opacity, grad_color, tile_unstable);
}

namespace {

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> metal_reduce_sample_bundle_scan_impl(
    const torch::Tensor& ids,
    const torch::Tensor& grad_ma_samples,
    const torch::Tensor& grad_q_samples,
    const torch::Tensor& grad_opacity_samples,
    const torch::Tensor& grad_color_samples,
    int64_t tube_count,
    bool compensated) {
  check_int_mps_1d(ids, "ids");
  check_float_mps_2d(grad_ma_samples, "grad_ma_samples", 3);
  check_float_mps_2d(grad_q_samples, "grad_q_samples", 6);
  check_float_mps_1d(grad_opacity_samples, "grad_opacity_samples");
  check_float_mps_2d(grad_color_samples, "grad_color_samples", 3);
  TORCH_CHECK(tube_count > 0, "tube_count must be positive");
  TORCH_CHECK(ids.size(0) == grad_ma_samples.size(0) && ids.size(0) == grad_q_samples.size(0) &&
                  ids.size(0) == grad_opacity_samples.size(0) && ids.size(0) == grad_color_samples.size(0),
              "compact sample ids and gradient rows must have the same length");
  TORCH_CHECK(tube_count <= INT32_MAX, "tube_count exceeds int32 range");
  TORCH_CHECK(ids.size(0) <= INT32_MAX, "sample count exceeds int32 range");

  auto opts_f = grad_ma_samples.options().dtype(torch::kFloat32);
  auto opts_i32 = ids.options().dtype(torch::kInt32);
  auto grad_ma = torch::empty({tube_count, 3}, opts_f);
  auto grad_q = torch::empty({tube_count, 6}, opts_f);
  auto grad_opacity = torch::empty({tube_count}, opts_f);
  auto grad_color = torch::empty({tube_count, 3}, opts_f);
  auto reduce_meta = torch::tensor({static_cast<int32_t>(ids.size(0)), static_cast<int32_t>(tube_count)}, opts_i32);
  auto& k = kernels();

  launch(compensated ? k.reduce_sample_bundle_scan_compensated : k.reduce_sample_bundle_scan, [&](MetalKernelFunction& fn) {
    fn.setArg(0, ids);
    fn.setArg(1, grad_ma_samples);
    fn.setArg(2, grad_q_samples);
    fn.setArg(3, grad_opacity_samples);
    fn.setArg(4, grad_color_samples);
    fn.setArg(5, reduce_meta);
    fn.setArg(6, grad_ma);
    fn.setArg(7, grad_q);
    fn.setArg(8, grad_opacity);
    fn.setArg(9, grad_color);
    fn.dispatch((uint64_t)tube_count, 256);
  });

  return std::make_tuple(grad_ma, grad_q, grad_opacity, grad_color);
}

}  // namespace

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> metal_reduce_sample_bundle_scan(
    const torch::Tensor& ids,
    const torch::Tensor& grad_ma_samples,
    const torch::Tensor& grad_q_samples,
    const torch::Tensor& grad_opacity_samples,
    const torch::Tensor& grad_color_samples,
    int64_t tube_count) {
  return metal_reduce_sample_bundle_scan_impl(
      ids,
      grad_ma_samples,
      grad_q_samples,
      grad_opacity_samples,
      grad_color_samples,
      tube_count,
      false);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> metal_reduce_sample_bundle_scan_compensated(
    const torch::Tensor& ids,
    const torch::Tensor& grad_ma_samples,
    const torch::Tensor& grad_q_samples,
    const torch::Tensor& grad_opacity_samples,
    const torch::Tensor& grad_color_samples,
    int64_t tube_count) {
  return metal_reduce_sample_bundle_scan_impl(
      ids,
      grad_ma_samples,
      grad_q_samples,
      grad_opacity_samples,
      grad_color_samples,
      tube_count,
      true);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> metal_reduce_sample_bundle_sorted_segments(
    const torch::Tensor& ids,
    const torch::Tensor& grad_ma_samples,
    const torch::Tensor& grad_q_samples,
    const torch::Tensor& grad_opacity_samples,
    const torch::Tensor& grad_color_samples,
    int64_t tube_count) {
  check_int_mps_1d(ids, "ids");
  check_float_mps_2d(grad_ma_samples, "grad_ma_samples", 3);
  check_float_mps_2d(grad_q_samples, "grad_q_samples", 6);
  check_float_mps_1d(grad_opacity_samples, "grad_opacity_samples");
  check_float_mps_2d(grad_color_samples, "grad_color_samples", 3);
  TORCH_CHECK(tube_count > 0, "tube_count must be positive");
  TORCH_CHECK(ids.size(0) == grad_ma_samples.size(0) && ids.size(0) == grad_q_samples.size(0) &&
                  ids.size(0) == grad_opacity_samples.size(0) && ids.size(0) == grad_color_samples.size(0),
              "compact sample ids and gradient rows must have the same length");
  TORCH_CHECK(tube_count <= INT32_MAX, "tube_count exceeds int32 range");
  TORCH_CHECK(ids.size(0) <= INT32_MAX, "sample count exceeds int32 range");

  auto opts_f = grad_ma_samples.options().dtype(torch::kFloat32);
  auto opts_i32 = ids.options().dtype(torch::kInt32);
  auto grad_ma = torch::empty({tube_count, 3}, opts_f);
  auto grad_q = torch::empty({tube_count, 6}, opts_f);
  auto grad_opacity = torch::empty({tube_count}, opts_f);
  auto grad_color = torch::empty({tube_count, 3}, opts_f);
  auto reduce_meta = torch::tensor({static_cast<int32_t>(ids.size(0)), static_cast<int32_t>(tube_count)}, opts_i32);
  auto& k = kernels();

  launch(k.reduce_sample_bundle_sorted_segments, [&](MetalKernelFunction& fn) {
    fn.setArg(0, ids);
    fn.setArg(1, grad_ma_samples);
    fn.setArg(2, grad_q_samples);
    fn.setArg(3, grad_opacity_samples);
    fn.setArg(4, grad_color_samples);
    fn.setArg(5, reduce_meta);
    fn.setArg(6, grad_ma);
    fn.setArg(7, grad_q);
    fn.setArg(8, grad_opacity);
    fn.setArg(9, grad_color);
    fn.dispatch((uint64_t)tube_count, 256);
  });

  return std::make_tuple(grad_ma, grad_q, grad_opacity, grad_color);
}

namespace {

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> metal_tile_pair_reduced_backward_impl(
    const torch::Tensor& ma,
    const torch::Tensor& q_uvt,
    const torch::Tensor& depth0,
    const torch::Tensor& depth_beta,
    const torch::Tensor& opacity,
    const torch::Tensor& color,
    const torch::Tensor& grad_image,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32,
    bool suffix_composite,
    bool parallel_reduce) {
  check_float_mps_2d(ma, "ma", 3);
  check_float_mps_2d(q_uvt, "q_uvt", 6);
  check_float_mps_1d(depth0, "depth0");
  check_float_mps_2d(depth_beta, "depth_beta", 3);
  check_float_mps_1d(opacity, "opacity");
  check_float_mps_2d(color, "color", 3);
  TORCH_CHECK(grad_image.device().is_mps(), "grad_image must be on MPS");
  TORCH_CHECK(grad_image.scalar_type() == torch::kFloat32, "grad_image must be float32");
  TORCH_CHECK(grad_image.dim() == 4 && grad_image.size(3) == 3, "grad_image must have shape [F,H,W,3]");
  TORCH_CHECK(grad_image.is_contiguous(), "grad_image must be contiguous");
  TORCH_CHECK(ma.size(0) == q_uvt.size(0) && ma.size(0) == depth0.size(0) && ma.size(0) == depth_beta.size(0) &&
                  ma.size(0) == opacity.size(0) && ma.size(0) == color.size(0),
              "all tube inputs must agree on N");

  auto meta = parse_meta(meta_i32, meta_f32);
  auto& sc = shader_config();
  check_meta(meta, ma.size(0), sc);
  TORCH_CHECK(grad_image.size(0) == meta.frames && grad_image.size(1) == meta.height && grad_image.size(2) == meta.width,
              "grad_image shape must match meta");
  auto& k = kernels();

  auto opts_f = ma.options().dtype(torch::kFloat32);
  auto opts_i32 = ma.options().dtype(torch::kInt32);
  auto tile_counts = torch::empty({meta.tile_count}, opts_i32);
  auto tile_overflow = torch::empty({meta.tile_count}, opts_i32);
  auto tile_unstable = torch::empty({meta.tile_count}, opts_i32);
  auto tile_tube_ids = torch::empty({meta.tile_count * meta.tile_capacity}, opts_i32);
  auto tile_depths = torch::empty({meta.tile_count * meta.tile_capacity}, opts_f);

  launch(k.clear_tiles, [&](MetalKernelFunction& fn) {
    fn.setArg(0, tile_counts);
    fn.setArg(1, tile_overflow);
    fn.setArg(2, tile_unstable);
    fn.setArg(3, meta_i32);
    fn.dispatch((uint64_t)meta.tile_count, 256);
  });

  launch(k.bin_tubes, [&](MetalKernelFunction& fn) {
    fn.setArg(0, ma);
    fn.setArg(1, q_uvt);
    fn.setArg(2, depth0);
    fn.setArg(3, depth_beta);
    fn.setArg(4, opacity);
    fn.setArg(5, meta_i32);
    fn.setArg(6, meta_f32);
    fn.setArg(7, tile_counts);
    fn.setArg(8, tile_tube_ids);
    fn.setArg(9, tile_depths);
    fn.setArg(10, tile_overflow);
    fn.dispatch((uint64_t)meta.tube_count, 256);
  });

  int64_t entry_count = (int64_t)meta.tile_count * (int64_t)meta.tile_capacity;
  auto grad_ids = torch::empty({entry_count}, opts_i32);
  auto grad_ma_samples = torch::empty({entry_count, 3}, opts_f);
  auto grad_q_samples = torch::empty({entry_count, 6}, opts_f);
  auto grad_opacity_samples = torch::empty({entry_count}, opts_f);
  auto grad_color_samples = torch::empty({entry_count, 3}, opts_f);
  auto grad_keys = torch::empty({entry_count}, opts_i32);

  auto sample_kernel = suffix_composite ? k.tile_pair_suffix_backward_samples : k.tile_pair_backward_samples;
  launch(sample_kernel, [&](MetalKernelFunction& fn) {
    fn.setArg(0, ma);
    fn.setArg(1, q_uvt);
    fn.setArg(2, depth0);
    fn.setArg(3, depth_beta);
    fn.setArg(4, opacity);
    fn.setArg(5, color);
    fn.setArg(6, grad_image);
    fn.setArg(7, meta_i32);
    fn.setArg(8, meta_f32);
    fn.setArg(9, tile_counts);
    fn.setArg(10, tile_tube_ids);
    fn.setArg(11, tile_depths);
    fn.setArg(12, tile_unstable);
    fn.setArg(13, grad_ids);
    fn.setArg(14, grad_ma_samples);
    fn.setArg(15, grad_q_samples);
    fn.setArg(16, grad_opacity_samples);
    fn.setArg(17, grad_color_samples);
    fn.setArg(18, grad_keys);
    fn.dispatch((uint64_t)entry_count, 256);
  });

  auto grad_ma = torch::empty({meta.tube_count, 3}, opts_f);
  auto grad_q = torch::empty({meta.tube_count, 6}, opts_f);
  auto grad_opacity = torch::empty({meta.tube_count}, opts_f);
  auto grad_color = torch::empty({meta.tube_count, 3}, opts_f);

  auto reduce_kernel = parallel_reduce ? k.reduce_tile_pair_bounds_scan_parallel : k.reduce_tile_pair_bounds_scan;
  launch(reduce_kernel, [&](MetalKernelFunction& fn) {
    fn.setArg(0, ma);
    fn.setArg(1, q_uvt);
    fn.setArg(2, opacity);
    fn.setArg(3, meta_i32);
    fn.setArg(4, meta_f32);
    fn.setArg(5, grad_ids);
    fn.setArg(6, grad_ma_samples);
    fn.setArg(7, grad_q_samples);
    fn.setArg(8, grad_opacity_samples);
    fn.setArg(9, grad_color_samples);
    fn.setArg(10, grad_ma);
    fn.setArg(11, grad_q);
    fn.setArg(12, grad_opacity);
    fn.setArg(13, grad_color);
    if (parallel_reduce) {
      fn.dispatch((uint64_t)meta.tube_count * (uint64_t)sc.threads, (uint64_t)sc.threads);
    } else {
      fn.dispatch((uint64_t)meta.tube_count, 256);
    }
  });

  return std::make_tuple(grad_ma, grad_q, grad_opacity, grad_color, tile_unstable);
}

}  // namespace

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> metal_tile_pair_reduced_backward(
    const torch::Tensor& ma,
    const torch::Tensor& q_uvt,
    const torch::Tensor& depth0,
    const torch::Tensor& depth_beta,
    const torch::Tensor& opacity,
    const torch::Tensor& color,
    const torch::Tensor& grad_image,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32) {
  return metal_tile_pair_reduced_backward_impl(ma, q_uvt, depth0, depth_beta, opacity, color, grad_image, meta_i32, meta_f32, false, false);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> metal_tile_pair_reduced_parallel_backward(
    const torch::Tensor& ma,
    const torch::Tensor& q_uvt,
    const torch::Tensor& depth0,
    const torch::Tensor& depth_beta,
    const torch::Tensor& opacity,
    const torch::Tensor& color,
    const torch::Tensor& grad_image,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32) {
  return metal_tile_pair_reduced_backward_impl(ma, q_uvt, depth0, depth_beta, opacity, color, grad_image, meta_i32, meta_f32, false, true);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> metal_tile_pair_suffix_reduced_backward(
    const torch::Tensor& ma,
    const torch::Tensor& q_uvt,
    const torch::Tensor& depth0,
    const torch::Tensor& depth_beta,
    const torch::Tensor& opacity,
    const torch::Tensor& color,
    const torch::Tensor& grad_image,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32) {
  return metal_tile_pair_reduced_backward_impl(ma, q_uvt, depth0, depth_beta, opacity, color, grad_image, meta_i32, meta_f32, true, false);
}

torch::Tensor metal_render_projective_trace_tiles(
    const torch::Tensor& coeffs,
    const torch::Tensor& times,
    const torch::Tensor& opacity,
    const torch::Tensor& color,
    const torch::Tensor& tile_counts,
    const torch::Tensor& tile_primitive_ids,
    const torch::Tensor& tile_active_start,
    const torch::Tensor& tile_active_stop,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32,
    double sigma_px) {
  check_float_mps_2d(coeffs, "coeffs", 9);
  check_float_mps_1d(times, "times");
  check_float_mps_1d(opacity, "opacity");
  check_float_mps_2d(color, "color", 3);
  check_int_mps_1d(tile_counts, "tile_counts");
  check_int_mps_1d(tile_primitive_ids, "tile_primitive_ids");
  check_int_mps_1d(tile_active_start, "tile_active_start");
  check_int_mps_1d(tile_active_stop, "tile_active_stop");
  TORCH_CHECK(coeffs.size(0) == opacity.size(0) && coeffs.size(0) == color.size(0),
              "coeffs, opacity, and color must agree on N");
  TORCH_CHECK(sigma_px > 0.0, "sigma_px must be positive");

  auto meta = parse_meta(meta_i32, meta_f32);
  auto& sc = shader_config();
  check_meta(meta, coeffs.size(0), sc);
  TORCH_CHECK(times.size(0) == meta.frames, "times must have shape [frames]");
  TORCH_CHECK(tile_counts.size(0) == meta.tile_count, "tile_counts must have shape [tile_count]");
  TORCH_CHECK(tile_primitive_ids.size(0) == meta.tile_count * meta.tile_capacity,
              "tile_primitive_ids must have shape [tile_count * tile_capacity]");
  TORCH_CHECK(tile_active_start.size(0) == meta.tile_count * meta.tile_capacity,
              "tile_active_start must have shape [tile_count * tile_capacity]");
  TORCH_CHECK(tile_active_stop.size(0) == meta.tile_count * meta.tile_capacity,
              "tile_active_stop must have shape [tile_count * tile_capacity]");

  auto opts_f = coeffs.options().dtype(torch::kFloat32);
  auto out = torch::empty({meta.frames, meta.height, meta.width, 3}, opts_f);
  auto projective_f32 = torch::tensor({static_cast<float>(sigma_px)}, opts_f);
  auto& k = kernels();

  launch(k.render_projective_trace_tiles, [&](MetalKernelFunction& fn) {
    fn.setArg(0, coeffs);
    fn.setArg(1, times);
    fn.setArg(2, opacity);
    fn.setArg(3, color);
    fn.setArg(4, tile_counts);
    fn.setArg(5, tile_primitive_ids);
    fn.setArg(6, tile_active_start);
    fn.setArg(7, tile_active_stop);
    fn.setArg(8, meta_i32);
    fn.setArg(9, meta_f32);
    fn.setArg(10, out);
    fn.setArg(11, projective_f32);
    fn.dispatch((uint64_t)meta.tile_count * (uint64_t)sc.threads, (uint64_t)sc.threads);
  });
  return out;
}

torch::Tensor metal_render_projective_trace_cell_tiles(
    const torch::Tensor& coeffs,
    const torch::Tensor& times,
    const torch::Tensor& opacity,
    const torch::Tensor& color,
    const torch::Tensor& tile_counts,
    const torch::Tensor& tile_trace_ids,
    const torch::Tensor& tile_active_start,
    const torch::Tensor& tile_active_stop,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32,
    double sigma_px) {
  check_float_mps_2d(coeffs, "coeffs", 9);
  check_float_mps_1d(times, "times");
  check_float_mps_1d(opacity, "opacity");
  check_float_mps_2d(color, "color", 3);
  check_int_mps_1d(tile_counts, "tile_counts");
  check_int_mps_1d(tile_trace_ids, "tile_trace_ids");
  check_int_mps_1d(tile_active_start, "tile_active_start");
  check_int_mps_1d(tile_active_stop, "tile_active_stop");
  TORCH_CHECK(coeffs.size(0) == opacity.size(0) && coeffs.size(0) == color.size(0),
              "coeffs, opacity, and color must agree on M");
  TORCH_CHECK(sigma_px > 0.0, "sigma_px must be positive");

  auto meta = parse_meta(meta_i32, meta_f32);
  auto& sc = shader_config();
  check_meta(meta, coeffs.size(0), sc);
  TORCH_CHECK(times.size(0) == meta.frames, "times must have shape [frames]");
  TORCH_CHECK(tile_counts.size(0) == meta.tile_count, "tile_counts must have shape [tile_count]");
  TORCH_CHECK(tile_trace_ids.size(0) == meta.tile_count * meta.tile_capacity,
              "tile_trace_ids must have shape [tile_count * tile_capacity]");
  TORCH_CHECK(tile_active_start.size(0) == meta.tile_count * meta.tile_capacity,
              "tile_active_start must have shape [tile_count * tile_capacity]");
  TORCH_CHECK(tile_active_stop.size(0) == meta.tile_count * meta.tile_capacity,
              "tile_active_stop must have shape [tile_count * tile_capacity]");

  auto opts_f = coeffs.options().dtype(torch::kFloat32);
  auto out = torch::empty({meta.frames, meta.height, meta.width, 3}, opts_f);
  auto projective_f32 = torch::tensor({static_cast<float>(sigma_px)}, opts_f);
  auto& k = kernels();

  launch(k.render_projective_trace_cell_tiles, [&](MetalKernelFunction& fn) {
    fn.setArg(0, coeffs);
    fn.setArg(1, times);
    fn.setArg(2, opacity);
    fn.setArg(3, color);
    fn.setArg(4, tile_counts);
    fn.setArg(5, tile_trace_ids);
    fn.setArg(6, tile_active_start);
    fn.setArg(7, tile_active_stop);
    fn.setArg(8, meta_i32);
    fn.setArg(9, meta_f32);
    fn.setArg(10, out);
    fn.setArg(11, projective_f32);
    fn.dispatch((uint64_t)meta.tile_count * (uint64_t)sc.threads, (uint64_t)sc.threads);
  });
  return out;
}

torch::Tensor metal_render_projective_trace_cell_interval_tiles(
    const torch::Tensor& coeffs,
    const torch::Tensor& times,
    const torch::Tensor& opacity,
    const torch::Tensor& opacity_time_coeffs,
    const torch::Tensor& spatial_precision_uv,
    const torch::Tensor& depth_affine_uv,
    const torch::Tensor& alpha_cutoff_reference_uvt,
    const torch::Tensor& color,
    const torch::Tensor& tile_counts,
    const torch::Tensor& tile_trace_ids,
    const torch::Tensor& tile_active_start,
    const torch::Tensor& tile_active_stop,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32,
    double sigma_px) {
  check_float_mps_2d(coeffs, "coeffs", 9);
  check_float_mps_1d(times, "times");
  check_float_mps_1d(opacity, "opacity");
  check_float_mps_2d(opacity_time_coeffs, "opacity_time_coeffs", 3);
  check_float_mps_2d(spatial_precision_uv, "spatial_precision_uv", 3);
  check_float_mps_2d(depth_affine_uv, "depth_affine_uv", 6);
  check_float_mps_2d(color, "color", 3);
  check_int_mps_1d(tile_counts, "tile_counts");
  check_int_mps_1d(tile_trace_ids, "tile_trace_ids");
  check_int_mps_1d(tile_active_start, "tile_active_start");
  check_int_mps_1d(tile_active_stop, "tile_active_stop");
  TORCH_CHECK(coeffs.size(0) == opacity.size(0) && coeffs.size(0) == opacity_time_coeffs.size(0) && coeffs.size(0) == spatial_precision_uv.size(0) && coeffs.size(0) == depth_affine_uv.size(0) && coeffs.size(0) == color.size(0),
              "coeffs, opacity, opacity_time_coeffs, spatial_precision_uv, depth_affine_uv, and color must agree on M");
  TORCH_CHECK(sigma_px > 0.0, "sigma_px must be positive");

  auto meta = parse_meta(meta_i32, meta_f32);
  check_float_mps_2d(alpha_cutoff_reference_uvt, "alpha_cutoff_reference_uvt", 9);
  TORCH_CHECK(alpha_cutoff_reference_uvt.size(0) == (meta.reserved1 ? coeffs.size(0) : 1),
              "alpha cutoff reference rows must match the metadata flag");
  auto& sc = shader_config();
  check_projective_interval_meta(meta, coeffs.size(0), sc);
  TORCH_CHECK(times.size(0) == meta.frames, "times must have shape [frames]");
  TORCH_CHECK(tile_counts.size(0) == meta.tile_count, "tile_counts must have shape [tile_count]");
  TORCH_CHECK(tile_trace_ids.size(0) == meta.tile_count * meta.tile_capacity,
              "tile_trace_ids must have shape [tile_count * tile_capacity]");
  TORCH_CHECK(tile_active_start.size(0) == meta.tile_count * meta.tile_capacity,
              "tile_active_start must have shape [tile_count * tile_capacity]");
  TORCH_CHECK(tile_active_stop.size(0) == meta.tile_count * meta.tile_capacity,
              "tile_active_stop must have shape [tile_count * tile_capacity]");

  auto opts_f = coeffs.options().dtype(torch::kFloat32);
  auto out = torch::empty({meta.frames, meta.height, meta.width, 3}, opts_f);
  auto projective_f32 = torch::tensor({static_cast<float>(sigma_px)}, opts_f);
  auto& k = kernels();
  uint64_t total_pixels = (uint64_t)meta.frames * (uint64_t)meta.height * (uint64_t)meta.width;

  launch(k.render_projective_trace_cell_interval_tiles, [&](MetalKernelFunction& fn) {
    fn.setArg(0, coeffs);
    fn.setArg(1, times);
    fn.setArg(2, opacity);
    fn.setArg(3, opacity_time_coeffs);
    fn.setArg(4, color);
    fn.setArg(5, tile_counts);
    fn.setArg(6, tile_trace_ids);
    fn.setArg(7, tile_active_start);
    fn.setArg(8, tile_active_stop);
    fn.setArg(9, meta_i32);
    fn.setArg(10, meta_f32);
    fn.setArg(11, out);
    fn.setArg(12, projective_f32);
    fn.setArg(13, spatial_precision_uv);
    fn.setArg(14, depth_affine_uv);
    fn.setArg(15, alpha_cutoff_reference_uvt);
    fn.dispatch(total_pixels, 256);
  });
  return out;
}

torch::Tensor metal_render_projective_trace_family_interval_tiles(
    const torch::Tensor& family_coeffs,
    const torch::Tensor& q_basis,
    const torch::Tensor& times,
    const torch::Tensor& opacity,
    const torch::Tensor& opacity_time_coeffs,
    const torch::Tensor& spatial_precision_uv,
    const torch::Tensor& depth_affine_uv,
    const torch::Tensor& color,
    const torch::Tensor& tile_counts,
    const torch::Tensor& tile_trace_ids,
    const torch::Tensor& tile_active_start,
    const torch::Tensor& tile_active_stop,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32,
    double sigma_px) {
  check_float_mps_3d(family_coeffs, "family_coeffs", 9);
  check_float_mps_2d(q_basis, "q_basis", family_coeffs.size(2));
  check_float_mps_1d(times, "times");
  check_float_mps_1d(opacity, "opacity");
  check_float_mps_2d(opacity_time_coeffs, "opacity_time_coeffs", 3);
  check_float_mps_2d(spatial_precision_uv, "spatial_precision_uv", 3);
  check_float_mps_2d(depth_affine_uv, "depth_affine_uv", 6);
  check_float_mps_2d(color, "color", 3);
  check_int_mps_1d(tile_counts, "tile_counts");
  check_int_mps_1d(tile_trace_ids, "tile_trace_ids");
  check_int_mps_1d(tile_active_start, "tile_active_start");
  check_int_mps_1d(tile_active_stop, "tile_active_stop");

  int64_t base_count = family_coeffs.size(0);
  int64_t q_count = q_basis.size(0);
  int64_t basis_count = family_coeffs.size(2);
  TORCH_CHECK(base_count > 0 && q_count > 0 && basis_count > 0,
              "family_coeffs and q_basis must have positive base/q/basis dimensions");
  TORCH_CHECK(opacity.size(0) == base_count && opacity_time_coeffs.size(0) == base_count &&
                  spatial_precision_uv.size(0) == base_count && depth_affine_uv.size(0) == base_count &&
                  color.size(0) == base_count,
              "opacity, opacity_time_coeffs, spatial_precision_uv, depth_affine_uv, and color must match base family trace count");
  TORCH_CHECK(sigma_px > 0.0, "sigma_px must be positive");

  auto meta = parse_meta(meta_i32, meta_f32);
  auto& sc = shader_config();
  check_projective_interval_meta(meta, base_count * q_count, sc);
  TORCH_CHECK(meta.reserved0 == base_count, "meta reserved0 must equal family base trace count");
  TORCH_CHECK(meta.reserved1 == basis_count, "meta reserved1 must equal family basis count");
  TORCH_CHECK(times.size(0) == meta.frames, "times must have shape [frames]");
  TORCH_CHECK(tile_counts.size(0) == meta.tile_count, "tile_counts must have shape [tile_count]");
  TORCH_CHECK(tile_trace_ids.size(0) == meta.tile_count * meta.tile_capacity,
              "tile_trace_ids must have shape [tile_count * tile_capacity]");
  TORCH_CHECK(tile_active_start.size(0) == meta.tile_count * meta.tile_capacity,
              "tile_active_start must have shape [tile_count * tile_capacity]");
  TORCH_CHECK(tile_active_stop.size(0) == meta.tile_count * meta.tile_capacity,
              "tile_active_stop must have shape [tile_count * tile_capacity]");

  auto opts_f = family_coeffs.options().dtype(torch::kFloat32);
  auto out = torch::empty({meta.frames, meta.height, meta.width, 3}, opts_f);
  auto projective_f32 = torch::tensor({static_cast<float>(sigma_px)}, opts_f);
  auto& k = kernels();
  uint64_t total_pixels = (uint64_t)meta.frames * (uint64_t)meta.height * (uint64_t)meta.width;

  launch(k.render_projective_trace_family_interval_tiles, [&](MetalKernelFunction& fn) {
    fn.setArg(0, family_coeffs);
    fn.setArg(1, q_basis);
    fn.setArg(2, times);
    fn.setArg(3, opacity);
    fn.setArg(4, opacity_time_coeffs);
    fn.setArg(5, color);
    fn.setArg(6, tile_counts);
    fn.setArg(7, tile_trace_ids);
    fn.setArg(8, tile_active_start);
    fn.setArg(9, tile_active_stop);
    fn.setArg(10, meta_i32);
    fn.setArg(11, meta_f32);
    fn.setArg(12, out);
    fn.setArg(13, projective_f32);
    fn.setArg(14, spatial_precision_uv);
    fn.setArg(15, depth_affine_uv);
    fn.dispatch(total_pixels, 256);
  });
  return out;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> metal_direct_projective_trace_family_interval_backward(
    const torch::Tensor& family_coeffs,
    const torch::Tensor& q_basis,
    const torch::Tensor& times,
    const torch::Tensor& opacity,
    const torch::Tensor& opacity_time_coeffs,
    const torch::Tensor& spatial_precision_uv,
    const torch::Tensor& depth_affine_uv,
    const torch::Tensor& color,
    const torch::Tensor& grad_image,
    const torch::Tensor& tile_counts,
    const torch::Tensor& tile_trace_ids,
    const torch::Tensor& tile_active_start,
    const torch::Tensor& tile_active_stop,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32,
    double sigma_px) {
  check_float_mps_3d(family_coeffs, "family_coeffs", 9);
  check_float_mps_2d(q_basis, "q_basis", family_coeffs.size(2));
  check_float_mps_1d(times, "times");
  check_float_mps_1d(opacity, "opacity");
  check_float_mps_2d(opacity_time_coeffs, "opacity_time_coeffs", 3);
  check_float_mps_2d(spatial_precision_uv, "spatial_precision_uv", 3);
  check_float_mps_2d(depth_affine_uv, "depth_affine_uv", 6);
  check_float_mps_2d(color, "color", 3);
  check_int_mps_1d(tile_counts, "tile_counts");
  check_int_mps_1d(tile_trace_ids, "tile_trace_ids");
  check_int_mps_1d(tile_active_start, "tile_active_start");
  check_int_mps_1d(tile_active_stop, "tile_active_stop");
  TORCH_CHECK(grad_image.device().is_mps(), "grad_image must be on MPS");
  TORCH_CHECK(grad_image.scalar_type() == torch::kFloat32, "grad_image must be float32");
  TORCH_CHECK(grad_image.dim() == 4 && grad_image.size(3) == 3, "grad_image must have shape [F,H,W,3]");
  TORCH_CHECK(grad_image.is_contiguous(), "grad_image must be contiguous");

  int64_t base_count = family_coeffs.size(0);
  int64_t q_count = q_basis.size(0);
  int64_t basis_count = family_coeffs.size(2);
  TORCH_CHECK(base_count > 0 && q_count > 0 && basis_count > 0,
              "family_coeffs and q_basis must have positive base/q/basis dimensions");
  TORCH_CHECK(opacity.size(0) == base_count && opacity_time_coeffs.size(0) == base_count &&
                  spatial_precision_uv.size(0) == base_count && depth_affine_uv.size(0) == base_count &&
                  color.size(0) == base_count,
              "opacity, opacity_time_coeffs, spatial_precision_uv, depth_affine_uv, and color must match base family trace count");
  TORCH_CHECK(sigma_px > 0.0, "sigma_px must be positive");

  auto meta = parse_meta(meta_i32, meta_f32);
  auto& sc = shader_config();
  check_projective_interval_meta(meta, base_count * q_count, sc);
  TORCH_CHECK(meta.reserved0 == base_count, "meta reserved0 must equal family base trace count");
  TORCH_CHECK(meta.reserved1 == basis_count, "meta reserved1 must equal family basis count");
  TORCH_CHECK(times.size(0) == meta.frames, "times must have shape [frames]");
  TORCH_CHECK(grad_image.size(0) == meta.frames && grad_image.size(1) == meta.height && grad_image.size(2) == meta.width,
              "grad_image shape must match meta");
  TORCH_CHECK(tile_counts.size(0) == meta.tile_count, "tile_counts must have shape [tile_count]");
  TORCH_CHECK(tile_trace_ids.size(0) == meta.tile_count * meta.tile_capacity,
              "tile_trace_ids must have shape [tile_count * tile_capacity]");
  TORCH_CHECK(tile_active_start.size(0) == meta.tile_count * meta.tile_capacity,
              "tile_active_start must have shape [tile_count * tile_capacity]");
  TORCH_CHECK(tile_active_stop.size(0) == meta.tile_count * meta.tile_capacity,
              "tile_active_stop must have shape [tile_count * tile_capacity]");

  auto opts_f = family_coeffs.options().dtype(torch::kFloat32);
  auto grad_family_coeffs = torch::zeros_like(family_coeffs);
  auto grad_q_basis = torch::zeros_like(q_basis);
  auto grad_opacity = torch::zeros({base_count}, opts_f);
  auto grad_opacity_time_coeffs = torch::zeros({base_count, 3}, opts_f);
  auto grad_spatial_precision_uv = torch::zeros({base_count, 3}, opts_f);
  auto grad_color = torch::zeros({base_count, 3}, opts_f);
  auto projective_f32 = torch::tensor({static_cast<float>(sigma_px)}, opts_f);
  auto& k = kernels();
  uint64_t total_pixels = (uint64_t)meta.frames * (uint64_t)meta.height * (uint64_t)meta.width;

  launch(k.direct_atomic_projective_family_cell_interval_backward, [&](MetalKernelFunction& fn) {
    fn.setArg(0, family_coeffs);
    fn.setArg(1, q_basis);
    fn.setArg(2, times);
    fn.setArg(3, opacity);
    fn.setArg(4, opacity_time_coeffs);
    fn.setArg(5, color);
    fn.setArg(6, grad_image);
    fn.setArg(7, tile_counts);
    fn.setArg(8, tile_trace_ids);
    fn.setArg(9, tile_active_start);
    fn.setArg(10, tile_active_stop);
    fn.setArg(11, meta_i32);
    fn.setArg(12, meta_f32);
    fn.setArg(13, grad_family_coeffs);
    fn.setArg(14, grad_q_basis);
    fn.setArg(15, grad_opacity);
    fn.setArg(16, grad_opacity_time_coeffs);
    fn.setArg(17, grad_color);
    fn.setArg(18, grad_spatial_precision_uv);
    fn.setArg(19, projective_f32);
    fn.setArg(20, spatial_precision_uv);
    fn.setArg(21, depth_affine_uv);
    fn.dispatch(total_pixels, 256);
  });
  return std::make_tuple(
      grad_family_coeffs,
      grad_q_basis,
      grad_opacity,
      grad_opacity_time_coeffs,
      grad_spatial_precision_uv,
      grad_color);
}

torch::Tensor metal_render_projective_trace_cell_interval_rows(
    const torch::Tensor& coeffs,
    const torch::Tensor& times,
    const torch::Tensor& opacity,
    const torch::Tensor& opacity_time_coeffs,
    const torch::Tensor& spatial_precision_uv,
    const torch::Tensor& depth_affine_uv,
    const torch::Tensor& alpha_cutoff_reference_uvt,
    const torch::Tensor& color,
    const torch::Tensor& tile_counts,
    const torch::Tensor& tile_trace_ids,
    const torch::Tensor& tile_active_start,
    const torch::Tensor& tile_active_stop,
    const torch::Tensor& row_weights,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32,
    double sigma_px) {
  check_float_mps_2d(coeffs, "coeffs", 9);
  check_float_mps_1d(times, "times");
  check_float_mps_1d(opacity, "opacity");
  check_float_mps_2d(opacity_time_coeffs, "opacity_time_coeffs", 3);
  check_float_mps_2d(spatial_precision_uv, "spatial_precision_uv", 3);
  check_float_mps_2d(depth_affine_uv, "depth_affine_uv", 6);
  check_float_mps_2d(color, "color", 3);
  check_int_mps_1d(tile_counts, "tile_counts");
  check_int_mps_1d(tile_trace_ids, "tile_trace_ids");
  check_int_mps_1d(tile_active_start, "tile_active_start");
  check_int_mps_1d(tile_active_stop, "tile_active_stop");
  TORCH_CHECK(coeffs.size(0) == opacity.size(0) && coeffs.size(0) == opacity_time_coeffs.size(0) && coeffs.size(0) == spatial_precision_uv.size(0) && coeffs.size(0) == depth_affine_uv.size(0) && coeffs.size(0) == color.size(0),
              "coeffs, opacity, opacity_time_coeffs, spatial_precision_uv, depth_affine_uv, and color must agree on M");
  TORCH_CHECK(sigma_px > 0.0, "sigma_px must be positive");

  auto meta = parse_meta(meta_i32, meta_f32);
  check_float_mps_2d(alpha_cutoff_reference_uvt, "alpha_cutoff_reference_uvt", 9);
  TORCH_CHECK(alpha_cutoff_reference_uvt.size(0) == (meta.reserved1 ? coeffs.size(0) : 1),
              "alpha cutoff reference rows must match the metadata flag");
  auto& sc = shader_config();
  check_projective_interval_meta(meta, coeffs.size(0), sc);
  TORCH_CHECK(times.size(0) == meta.frames, "times must have shape [frames]");
  check_float_mps_2d(row_weights, "row_weights", meta.height);
  TORCH_CHECK(row_weights.size(0) == meta.frames, "row_weights must have shape [frames,height]");
  TORCH_CHECK(tile_counts.size(0) == meta.tile_count, "tile_counts must have shape [tile_count]");
  TORCH_CHECK(tile_trace_ids.size(0) == meta.tile_count * meta.tile_capacity,
              "tile_trace_ids must have shape [tile_count * tile_capacity]");
  TORCH_CHECK(tile_active_start.size(0) == meta.tile_count * meta.tile_capacity,
              "tile_active_start must have shape [tile_count * tile_capacity]");
  TORCH_CHECK(tile_active_stop.size(0) == meta.tile_count * meta.tile_capacity,
              "tile_active_stop must have shape [tile_count * tile_capacity]");

  auto opts_f = coeffs.options().dtype(torch::kFloat32);
  auto out = torch::empty({meta.height, meta.width, 3}, opts_f);
  auto projective_f32 = torch::tensor({static_cast<float>(sigma_px)}, opts_f);
  auto& k = kernels();
  uint64_t total_pixels = (uint64_t)meta.height * (uint64_t)meta.width;

  launch(k.render_projective_trace_cell_interval_rows, [&](MetalKernelFunction& fn) {
    fn.setArg(0, coeffs);
    fn.setArg(1, times);
    fn.setArg(2, opacity);
    fn.setArg(3, opacity_time_coeffs);
    fn.setArg(4, color);
    fn.setArg(5, tile_counts);
    fn.setArg(6, tile_trace_ids);
    fn.setArg(7, tile_active_start);
    fn.setArg(8, tile_active_stop);
    fn.setArg(9, row_weights);
    fn.setArg(10, meta_i32);
    fn.setArg(11, meta_f32);
    fn.setArg(12, out);
    fn.setArg(13, projective_f32);
    fn.setArg(14, spatial_precision_uv);
    fn.setArg(15, depth_affine_uv);
    fn.setArg(16, alpha_cutoff_reference_uvt);
    fn.dispatch(total_pixels, 256);
  });
  return out;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> metal_direct_projective_trace_cell_interval_backward(
    const torch::Tensor& coeffs,
    const torch::Tensor& times,
    const torch::Tensor& opacity,
    const torch::Tensor& opacity_time_coeffs,
    const torch::Tensor& spatial_precision_uv,
    const torch::Tensor& depth_affine_uv,
    const torch::Tensor& alpha_cutoff_reference_uvt,
    const torch::Tensor& color,
    const torch::Tensor& grad_image,
    const torch::Tensor& tile_counts,
    const torch::Tensor& tile_trace_ids,
    const torch::Tensor& tile_active_start,
    const torch::Tensor& tile_active_stop,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32,
    double sigma_px) {
  check_float_mps_2d(coeffs, "coeffs", 9);
  check_float_mps_1d(times, "times");
  check_float_mps_1d(opacity, "opacity");
  check_float_mps_2d(opacity_time_coeffs, "opacity_time_coeffs", 3);
  check_float_mps_2d(spatial_precision_uv, "spatial_precision_uv", 3);
  check_float_mps_2d(depth_affine_uv, "depth_affine_uv", 6);
  check_float_mps_2d(color, "color", 3);
  check_int_mps_1d(tile_counts, "tile_counts");
  check_int_mps_1d(tile_trace_ids, "tile_trace_ids");
  check_int_mps_1d(tile_active_start, "tile_active_start");
  check_int_mps_1d(tile_active_stop, "tile_active_stop");
  TORCH_CHECK(grad_image.device().is_mps(), "grad_image must be on MPS");
  TORCH_CHECK(grad_image.scalar_type() == torch::kFloat32, "grad_image must be float32");
  TORCH_CHECK(grad_image.dim() == 4 && grad_image.size(3) == 3, "grad_image must have shape [F,H,W,3]");
  TORCH_CHECK(grad_image.is_contiguous(), "grad_image must be contiguous");
  TORCH_CHECK(coeffs.size(0) == opacity.size(0) && coeffs.size(0) == opacity_time_coeffs.size(0) && coeffs.size(0) == spatial_precision_uv.size(0) && coeffs.size(0) == depth_affine_uv.size(0) && coeffs.size(0) == color.size(0),
              "coeffs, opacity, opacity_time_coeffs, spatial_precision_uv, depth_affine_uv, and color must agree on M");
  TORCH_CHECK(sigma_px > 0.0, "sigma_px must be positive");

  auto meta = parse_meta(meta_i32, meta_f32);
  check_float_mps_2d(alpha_cutoff_reference_uvt, "alpha_cutoff_reference_uvt", 9);
  TORCH_CHECK(alpha_cutoff_reference_uvt.size(0) == (meta.reserved1 ? coeffs.size(0) : 1),
              "alpha cutoff reference rows must match the metadata flag");
  auto& sc = shader_config();
  check_projective_interval_meta(meta, coeffs.size(0), sc);
  TORCH_CHECK(times.size(0) == meta.frames, "times must have shape [frames]");
  TORCH_CHECK(grad_image.size(0) == meta.frames && grad_image.size(1) == meta.height && grad_image.size(2) == meta.width,
              "grad_image shape must match meta");
  TORCH_CHECK(tile_counts.size(0) == meta.tile_count, "tile_counts must have shape [tile_count]");
  TORCH_CHECK(tile_trace_ids.size(0) == meta.tile_count * meta.tile_capacity,
              "tile_trace_ids must have shape [tile_count * tile_capacity]");
  TORCH_CHECK(tile_active_start.size(0) == meta.tile_count * meta.tile_capacity,
              "tile_active_start must have shape [tile_count * tile_capacity]");
  TORCH_CHECK(tile_active_stop.size(0) == meta.tile_count * meta.tile_capacity,
              "tile_active_stop must have shape [tile_count * tile_capacity]");

  auto opts_f = coeffs.options().dtype(torch::kFloat32);
  auto grad_coeffs = torch::zeros({meta.tube_count, 9}, opts_f);
  auto grad_opacity = torch::zeros({meta.tube_count}, opts_f);
  auto grad_opacity_time_coeffs = torch::zeros({meta.tube_count, 3}, opts_f);
  auto grad_spatial_precision_uv = torch::zeros({meta.tube_count, 3}, opts_f);
  auto grad_color = torch::zeros({meta.tube_count, 3}, opts_f);
  auto projective_f32 = torch::tensor({static_cast<float>(sigma_px)}, opts_f);
  auto& k = kernels();
  uint64_t total_pixels = (uint64_t)meta.frames * (uint64_t)meta.height * (uint64_t)meta.width;

  launch(k.direct_atomic_projective_cell_interval_backward, [&](MetalKernelFunction& fn) {
    fn.setArg(0, coeffs);
    fn.setArg(1, times);
    fn.setArg(2, opacity);
    fn.setArg(3, opacity_time_coeffs);
    fn.setArg(4, color);
    fn.setArg(5, grad_image);
    fn.setArg(6, tile_counts);
    fn.setArg(7, tile_trace_ids);
    fn.setArg(8, tile_active_start);
    fn.setArg(9, tile_active_stop);
    fn.setArg(10, meta_i32);
    fn.setArg(11, meta_f32);
    fn.setArg(12, grad_coeffs);
    fn.setArg(13, grad_opacity);
    fn.setArg(14, grad_opacity_time_coeffs);
    fn.setArg(15, grad_color);
    fn.setArg(16, grad_spatial_precision_uv);
    fn.setArg(17, projective_f32);
    fn.setArg(18, spatial_precision_uv);
    fn.setArg(19, depth_affine_uv);
    fn.setArg(20, alpha_cutoff_reference_uvt);
    fn.dispatch(total_pixels, 256);
  });
  return std::make_tuple(grad_coeffs, grad_opacity, grad_opacity_time_coeffs, grad_spatial_precision_uv, grad_color);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> metal_direct_projective_trace_backward(
    const torch::Tensor& coeffs,
    const torch::Tensor& times,
    const torch::Tensor& opacity,
    const torch::Tensor& color,
    const torch::Tensor& grad_image,
    const torch::Tensor& tile_counts,
    const torch::Tensor& tile_primitive_ids,
    const torch::Tensor& tile_active_start,
    const torch::Tensor& tile_active_stop,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32,
    double sigma_px) {
  check_float_mps_2d(coeffs, "coeffs", 9);
  check_float_mps_1d(times, "times");
  check_float_mps_1d(opacity, "opacity");
  check_float_mps_2d(color, "color", 3);
  check_int_mps_1d(tile_counts, "tile_counts");
  check_int_mps_1d(tile_primitive_ids, "tile_primitive_ids");
  check_int_mps_1d(tile_active_start, "tile_active_start");
  check_int_mps_1d(tile_active_stop, "tile_active_stop");
  TORCH_CHECK(grad_image.device().is_mps(), "grad_image must be on MPS");
  TORCH_CHECK(grad_image.scalar_type() == torch::kFloat32, "grad_image must be float32");
  TORCH_CHECK(grad_image.dim() == 4 && grad_image.size(3) == 3, "grad_image must have shape [F,H,W,3]");
  TORCH_CHECK(grad_image.is_contiguous(), "grad_image must be contiguous");
  TORCH_CHECK(coeffs.size(0) == opacity.size(0) && coeffs.size(0) == color.size(0),
              "coeffs, opacity, and color must agree on N");
  TORCH_CHECK(sigma_px > 0.0, "sigma_px must be positive");

  auto meta = parse_meta(meta_i32, meta_f32);
  auto& sc = shader_config();
  check_meta(meta, coeffs.size(0), sc);
  TORCH_CHECK(times.size(0) == meta.frames, "times must have shape [frames]");
  TORCH_CHECK(grad_image.size(0) == meta.frames && grad_image.size(1) == meta.height && grad_image.size(2) == meta.width,
              "grad_image shape must match meta");
  TORCH_CHECK(tile_counts.size(0) == meta.tile_count, "tile_counts must have shape [tile_count]");
  TORCH_CHECK(tile_primitive_ids.size(0) == meta.tile_count * meta.tile_capacity,
              "tile_primitive_ids must have shape [tile_count * tile_capacity]");
  TORCH_CHECK(tile_active_start.size(0) == meta.tile_count * meta.tile_capacity,
              "tile_active_start must have shape [tile_count * tile_capacity]");
  TORCH_CHECK(tile_active_stop.size(0) == meta.tile_count * meta.tile_capacity,
              "tile_active_stop must have shape [tile_count * tile_capacity]");

  auto opts_f = coeffs.options().dtype(torch::kFloat32);
  auto grad_coeffs = torch::zeros({meta.tube_count, 9}, opts_f);
  auto grad_opacity = torch::zeros({meta.tube_count}, opts_f);
  auto grad_color = torch::zeros({meta.tube_count, 3}, opts_f);
  auto projective_f32 = torch::tensor({static_cast<float>(sigma_px)}, opts_f);
  auto& k = kernels();

  launch(k.direct_atomic_projective_trace_backward, [&](MetalKernelFunction& fn) {
    fn.setArg(0, coeffs);
    fn.setArg(1, times);
    fn.setArg(2, opacity);
    fn.setArg(3, color);
    fn.setArg(4, grad_image);
    fn.setArg(5, tile_counts);
    fn.setArg(6, tile_primitive_ids);
    fn.setArg(7, tile_active_start);
    fn.setArg(8, tile_active_stop);
    fn.setArg(9, meta_i32);
    fn.setArg(10, meta_f32);
    fn.setArg(11, grad_coeffs);
    fn.setArg(12, grad_opacity);
    fn.setArg(13, grad_color);
    fn.setArg(14, projective_f32);
    fn.dispatch((uint64_t)meta.tile_count * (uint64_t)sc.threads, (uint64_t)sc.threads);
  });
  return std::make_tuple(grad_coeffs, grad_opacity, grad_color);
}

torch::Tensor metal_projective_trace_eval(
    const torch::Tensor& coeffs,
    const torch::Tensor& times,
    double eps) {
  check_float_mps_2d(coeffs, "coeffs", 9);
  check_float_mps_1d(times, "times");
  TORCH_CHECK(coeffs.size(0) <= INT32_MAX, "coeff row count exceeds int32 range");
  TORCH_CHECK(times.size(0) <= INT32_MAX, "time count exceeds int32 range");
  TORCH_CHECK(eps > 0.0, "eps must be positive");

  auto opts_f = coeffs.options().dtype(torch::kFloat32);
  auto opts_i32 = coeffs.options().dtype(torch::kInt32);
  auto out = torch::empty({coeffs.size(0), times.size(0), 4}, opts_f);
  auto meta_i32 = torch::tensor({static_cast<int32_t>(coeffs.size(0)), static_cast<int32_t>(times.size(0))}, opts_i32);
  auto meta_f32 = torch::tensor({static_cast<float>(eps)}, opts_f);
  auto& k = kernels();

  uint64_t sample_count = static_cast<uint64_t>(coeffs.size(0)) * static_cast<uint64_t>(times.size(0));
  if (sample_count == 0) {
    return out;
  }
  launch(k.projective_trace_eval, [&](MetalKernelFunction& fn) {
    fn.setArg(0, coeffs);
    fn.setArg(1, times);
    fn.setArg(2, meta_i32);
    fn.setArg(3, meta_f32);
    fn.setArg(4, out);
    fn.dispatch(sample_count, 256);
  });
  return out;
}

torch::Tensor metal_projective_trace_family_eval(
    const torch::Tensor& family_coeffs,
    const torch::Tensor& q_basis,
    const torch::Tensor& times,
    double eps) {
  check_float_mps_3d(family_coeffs, "family_coeffs", 9);
  check_float_mps_2d(q_basis, "q_basis", family_coeffs.size(2));
  check_float_mps_1d(times, "times");
  TORCH_CHECK(q_basis.device() == family_coeffs.device(), "q_basis must be on the same device as family_coeffs");
  TORCH_CHECK(times.device() == family_coeffs.device(), "times must be on the same device as family_coeffs");
  TORCH_CHECK(family_coeffs.size(0) <= INT32_MAX, "family trace count exceeds int32 range");
  TORCH_CHECK(q_basis.size(0) <= INT32_MAX, "q count exceeds int32 range");
  TORCH_CHECK(times.size(0) <= INT32_MAX, "time count exceeds int32 range");
  TORCH_CHECK(family_coeffs.size(2) > 0 && family_coeffs.size(2) <= INT32_MAX, "basis count must be positive and fit int32");
  TORCH_CHECK(eps > 0.0, "eps must be positive");

  auto opts_f = family_coeffs.options().dtype(torch::kFloat32);
  auto opts_i32 = family_coeffs.options().dtype(torch::kInt32);
  auto out = torch::empty({q_basis.size(0), family_coeffs.size(0), times.size(0), 4}, opts_f);
  auto meta_i32 = torch::tensor(
      {
          static_cast<int32_t>(family_coeffs.size(0)),
          static_cast<int32_t>(q_basis.size(0)),
          static_cast<int32_t>(times.size(0)),
          static_cast<int32_t>(family_coeffs.size(2)),
      },
      opts_i32);
  auto meta_f32 = torch::tensor({static_cast<float>(eps)}, opts_f);
  auto& k = kernels();

  uint64_t sample_count = static_cast<uint64_t>(family_coeffs.size(0)) *
                          static_cast<uint64_t>(q_basis.size(0)) *
                          static_cast<uint64_t>(times.size(0));
  if (sample_count == 0) {
    return out;
  }
  launch(k.projective_trace_family_eval, [&](MetalKernelFunction& fn) {
    fn.setArg(0, family_coeffs);
    fn.setArg(1, q_basis);
    fn.setArg(2, times);
    fn.setArg(3, meta_i32);
    fn.setArg(4, meta_f32);
    fn.setArg(5, out);
    fn.dispatch(sample_count, 256);
  });
  return out;
}

std::tuple<torch::Tensor, torch::Tensor> metal_projective_trace_family_backward(
    const torch::Tensor& family_coeffs,
    const torch::Tensor& q_basis,
    const torch::Tensor& times,
    const torch::Tensor& grad_out,
    double eps) {
  check_float_mps_3d(family_coeffs, "family_coeffs", 9);
  check_float_mps_2d(q_basis, "q_basis", family_coeffs.size(2));
  check_float_mps_1d(times, "times");
  TORCH_CHECK(grad_out.device().is_mps(), "grad_out must be on MPS");
  TORCH_CHECK(grad_out.scalar_type() == torch::kFloat32, "grad_out must be float32");
  TORCH_CHECK(grad_out.dim() == 4 && grad_out.size(3) == 4, "grad_out must have shape [Q,N,S,4]");
  TORCH_CHECK(grad_out.is_contiguous(), "grad_out must be contiguous");
  TORCH_CHECK(grad_out.size(0) == q_basis.size(0), "grad_out Q dimension must match q_basis");
  TORCH_CHECK(grad_out.size(1) == family_coeffs.size(0), "grad_out N dimension must match family_coeffs");
  TORCH_CHECK(grad_out.size(2) == times.size(0), "grad_out S dimension must match times");
  TORCH_CHECK(q_basis.device() == family_coeffs.device(), "q_basis must be on the same device as family_coeffs");
  TORCH_CHECK(times.device() == family_coeffs.device(), "times must be on the same device as family_coeffs");
  TORCH_CHECK(grad_out.device() == family_coeffs.device(), "grad_out must be on the same device as family_coeffs");
  TORCH_CHECK(family_coeffs.size(0) <= INT32_MAX, "family trace count exceeds int32 range");
  TORCH_CHECK(q_basis.size(0) <= INT32_MAX, "q count exceeds int32 range");
  TORCH_CHECK(times.size(0) <= INT32_MAX, "time count exceeds int32 range");
  TORCH_CHECK(family_coeffs.size(2) > 0 && family_coeffs.size(2) <= INT32_MAX, "basis count must be positive and fit int32");
  TORCH_CHECK(eps > 0.0, "eps must be positive");

  auto opts_f = family_coeffs.options().dtype(torch::kFloat32);
  auto opts_i32 = family_coeffs.options().dtype(torch::kInt32);
  auto grad_family_coeffs = torch::zeros_like(family_coeffs);
  auto grad_q_basis = torch::zeros_like(q_basis);
  auto meta_i32 = torch::tensor(
      {
          static_cast<int32_t>(family_coeffs.size(0)),
          static_cast<int32_t>(q_basis.size(0)),
          static_cast<int32_t>(times.size(0)),
          static_cast<int32_t>(family_coeffs.size(2)),
      },
      opts_i32);
  auto meta_f32 = torch::tensor({static_cast<float>(eps)}, opts_f);
  auto& k = kernels();

  uint64_t sample_count = static_cast<uint64_t>(family_coeffs.size(0)) *
                          static_cast<uint64_t>(q_basis.size(0)) *
                          static_cast<uint64_t>(times.size(0));
  if (sample_count == 0) {
    return std::make_tuple(grad_family_coeffs, grad_q_basis);
  }
  launch(k.projective_trace_family_backward, [&](MetalKernelFunction& fn) {
    fn.setArg(0, family_coeffs);
    fn.setArg(1, q_basis);
    fn.setArg(2, times);
    fn.setArg(3, grad_out);
    fn.setArg(4, meta_i32);
    fn.setArg(5, meta_f32);
    fn.setArg(6, grad_family_coeffs);
    fn.setArg(7, grad_q_basis);
    fn.dispatch(sample_count, 256);
  });
  return std::make_tuple(grad_family_coeffs, grad_q_basis);
}

}  // namespace star_uvt

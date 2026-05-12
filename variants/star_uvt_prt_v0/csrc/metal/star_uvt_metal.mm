#import <Foundation/Foundation.h>

#include <ATen/ATen.h>
#include <ATen/native/mps/MetalShaderLibrary.h>
#include <torch/extension.h>
#include <torch/mps.h>

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
    TORCH_CHECK(c.tile_x == 4 || c.tile_x == 8 || c.tile_x == 16, "STAR_UVT_TILE_X must be 4, 8, or 16");
    TORCH_CHECK(c.tile_y == 4 || c.tile_y == 8 || c.tile_y == 16, "STAR_UVT_TILE_Y must be 4, 8, or 16");
    TORCH_CHECK(c.tile_t == 1 || c.tile_t == 2 || c.tile_t == 4, "STAR_UVT_TILE_T must be 1, 2, or 4");
    TORCH_CHECK(c.tile_capacity == 32 || c.tile_capacity == 64 || c.tile_capacity == 128 ||
                    c.tile_capacity == 256 || c.tile_capacity == 512,
                "STAR_UVT_TILE_CAPACITY must be 32, 64, 128, 256, or 512");
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
  std::shared_ptr<MetalKernelFunction> clear_direct_gradients_i32;
  std::shared_ptr<MetalKernelFunction> fixedpoint_gradients_to_float;
  std::shared_ptr<MetalKernelFunction> split_fixedpoint_gradients_to_float;
  std::shared_ptr<MetalKernelFunction> bin_tubes;
  std::shared_ptr<MetalKernelFunction> render_tiles;
  std::shared_ptr<MetalKernelFunction> render_projective_rational_direct;
  std::shared_ptr<MetalKernelFunction> bin_projective_rational_tubes;
  std::shared_ptr<MetalKernelFunction> render_projective_rational_tiles;
  std::shared_ptr<MetalKernelFunction> simple_backward_samples;
  std::shared_ptr<MetalKernelFunction> stable_backward_samples;
  std::shared_ptr<MetalKernelFunction> direct_atomic_backward;
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
};

MetalKernels& kernels() {
  static std::once_flag once;
  static std::unique_ptr<DynamicMetalShaderLibrary> lib;
  static MetalKernels out;
  std::call_once(once, []() {
    lib = std::make_unique<DynamicMetalShaderLibrary>(load_shader_source());
    out.clear_tiles = lib->getKernelFunction("clear_tiles");
    out.clear_direct_gradients = lib->getKernelFunction("clear_direct_gradients");
    out.clear_direct_gradients_i32 = lib->getKernelFunction("clear_direct_gradients_i32");
    out.fixedpoint_gradients_to_float = lib->getKernelFunction("fixedpoint_gradients_to_float");
    out.split_fixedpoint_gradients_to_float = lib->getKernelFunction("split_fixedpoint_gradients_to_float");
    out.bin_tubes = lib->getKernelFunction("bin_screen_tubes_to_uvt_tiles");
    out.render_tiles = lib->getKernelFunction("render_uvt_tiles");
    out.render_projective_rational_direct = lib->getKernelFunction("render_projective_rational_direct");
    out.bin_projective_rational_tubes = lib->getKernelFunction("bin_projective_rational_tubes_to_uvt_tiles");
    out.render_projective_rational_tiles = lib->getKernelFunction("render_projective_rational_tiles");
    out.simple_backward_samples = lib->getKernelFunction("simple_backward_samples");
    out.stable_backward_samples = lib->getKernelFunction("stable_backward_samples");
    out.direct_atomic_backward = lib->getKernelFunction("direct_atomic_backward");
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

void check_float_mps_3d(const torch::Tensor& t, const char* name, int64_t cols) {
  TORCH_CHECK(t.device().is_mps(), name, " must be on MPS");
  TORCH_CHECK(t.scalar_type() == torch::kFloat32, name, " must be float32");
  TORCH_CHECK(t.dim() == 3 && t.size(2) == cols, name, " must have shape [N,H,", cols, "]");
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

torch::Tensor metal_render_projective_rational_direct(
    const torch::Tensor& h_coeff,
    const torch::Tensor& lambda_uv,
    const torch::Tensor& lambda_t,
    const torch::Tensor& center_t,
    const torch::Tensor& opacity,
    const torch::Tensor& color,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32) {
  check_float_mps_3d(h_coeff, "h_coeff", 3);
  check_float_mps_2d(lambda_uv, "lambda_uv", 3);
  check_float_mps_1d(lambda_t, "lambda_t");
  check_float_mps_1d(center_t, "center_t");
  check_float_mps_1d(opacity, "opacity");
  check_float_mps_2d(color, "color", 3);
  TORCH_CHECK(h_coeff.size(0) == lambda_uv.size(0) && h_coeff.size(0) == lambda_t.size(0) &&
                  h_coeff.size(0) == center_t.size(0) && h_coeff.size(0) == opacity.size(0) &&
                  h_coeff.size(0) == color.size(0),
              "all PRT inputs must agree on N");

  auto meta = parse_meta(meta_i32, meta_f32);
  auto& sc = shader_config();
  check_meta(meta, h_coeff.size(0), sc);
  TORCH_CHECK(meta.reserved0 == h_coeff.size(1), "meta reserved0 must equal h_coeff term count");
  TORCH_CHECK(meta.reserved0 > 0, "h_coeff term count must be positive");
  auto& k = kernels();

  auto opts_f = h_coeff.options().dtype(torch::kFloat32);
  auto out = torch::empty({meta.frames, meta.height, meta.width, 3}, opts_f);
  int64_t total_pixels = (int64_t)meta.frames * (int64_t)meta.height * (int64_t)meta.width;

  launch(k.render_projective_rational_direct, [&](MetalKernelFunction& fn) {
    fn.setArg(0, h_coeff);
    fn.setArg(1, lambda_uv);
    fn.setArg(2, lambda_t);
    fn.setArg(3, center_t);
    fn.setArg(4, opacity);
    fn.setArg(5, color);
    fn.setArg(6, meta_i32);
    fn.setArg(7, meta_f32);
    fn.setArg(8, out);
    fn.dispatch((uint64_t)total_pixels, 256);
  });

  return out;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> metal_render_projective_rational_tiled(
    const torch::Tensor& h_coeff,
    const torch::Tensor& lambda_uv,
    const torch::Tensor& lambda_t,
    const torch::Tensor& center_t,
    const torch::Tensor& opacity,
    const torch::Tensor& color,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32) {
  check_float_mps_3d(h_coeff, "h_coeff", 3);
  check_float_mps_2d(lambda_uv, "lambda_uv", 3);
  check_float_mps_1d(lambda_t, "lambda_t");
  check_float_mps_1d(center_t, "center_t");
  check_float_mps_1d(opacity, "opacity");
  check_float_mps_2d(color, "color", 3);
  TORCH_CHECK(h_coeff.size(0) == lambda_uv.size(0) && h_coeff.size(0) == lambda_t.size(0) &&
                  h_coeff.size(0) == center_t.size(0) && h_coeff.size(0) == opacity.size(0) &&
                  h_coeff.size(0) == color.size(0),
              "all PRT inputs must agree on N");

  auto meta = parse_meta(meta_i32, meta_f32);
  auto& sc = shader_config();
  check_meta(meta, h_coeff.size(0), sc);
  TORCH_CHECK(meta.reserved0 == h_coeff.size(1), "meta reserved0 must equal h_coeff term count");
  TORCH_CHECK(meta.reserved0 > 0, "h_coeff term count must be positive");
  auto& k = kernels();

  auto opts_f = h_coeff.options().dtype(torch::kFloat32);
  auto opts_i32 = h_coeff.options().dtype(torch::kInt32);

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

  launch(k.bin_projective_rational_tubes, [&](MetalKernelFunction& fn) {
    fn.setArg(0, h_coeff);
    fn.setArg(1, lambda_uv);
    fn.setArg(2, lambda_t);
    fn.setArg(3, center_t);
    fn.setArg(4, opacity);
    fn.setArg(5, meta_i32);
    fn.setArg(6, meta_f32);
    fn.setArg(7, tile_counts);
    fn.setArg(8, tile_tube_ids);
    fn.setArg(9, tile_depths);
    fn.setArg(10, tile_overflow);
    fn.dispatch((uint64_t)meta.tube_count, 256);
  });

  launch(k.render_projective_rational_tiles, [&](MetalKernelFunction& fn) {
    fn.setArg(0, h_coeff);
    fn.setArg(1, lambda_uv);
    fn.setArg(2, lambda_t);
    fn.setArg(3, center_t);
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

}  // namespace star_uvt

#import <Foundation/Foundation.h>

#include <ATen/ATen.h>
#include <ATen/native/mps/MetalShaderLibrary.h>
#include <torch/extension.h>
#include <torch/mps.h>

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
    TORCH_CHECK(c.tile_x == 8 || c.tile_x == 16, "STAR_UVT_TILE_X must be 8 or 16");
    TORCH_CHECK(c.tile_y == 8 || c.tile_y == 16, "STAR_UVT_TILE_Y must be 8 or 16");
    TORCH_CHECK(c.tile_t == 1 || c.tile_t == 2 || c.tile_t == 4, "STAR_UVT_TILE_T must be 1, 2, or 4");
    TORCH_CHECK(c.tile_capacity == 32 || c.tile_capacity == 64 || c.tile_capacity == 128 || c.tile_capacity == 256,
                "STAR_UVT_TILE_CAPACITY must be 32, 64, 128, or 256");
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
  return preamble + std::string([src UTF8String]);
}

struct MetalKernels {
  std::shared_ptr<MetalKernelFunction> clear_tiles;
  std::shared_ptr<MetalKernelFunction> bin_tubes;
  std::shared_ptr<MetalKernelFunction> render_tiles;
  std::shared_ptr<MetalKernelFunction> simple_backward_samples;
  std::shared_ptr<MetalKernelFunction> stable_backward_samples;
};

MetalKernels& kernels() {
  static std::once_flag once;
  static std::unique_ptr<DynamicMetalShaderLibrary> lib;
  static MetalKernels out;
  std::call_once(once, []() {
    lib = std::make_unique<DynamicMetalShaderLibrary>(load_shader_source());
    out.clear_tiles = lib->getKernelFunction("clear_tiles");
    out.bin_tubes = lib->getKernelFunction("bin_screen_tubes_to_uvt_tiles");
    out.render_tiles = lib->getKernelFunction("render_uvt_tiles");
    out.simple_backward_samples = lib->getKernelFunction("simple_backward_samples");
    out.stable_backward_samples = lib->getKernelFunction("stable_backward_samples");
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

void check_float_mps_1d(const torch::Tensor& t, const char* name) {
  TORCH_CHECK(t.device().is_mps(), name, " must be on MPS");
  TORCH_CHECK(t.scalar_type() == torch::kFloat32, name, " must be float32");
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
  auto grad_count = torch::zeros({1}, opts_i32);

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
    fn.dispatch((uint64_t)meta.tile_count * (uint64_t)sc.threads, (uint64_t)sc.threads);
  });

  return std::make_tuple(grad_ids, grad_ma_samples, grad_q_samples, grad_opacity_samples, grad_color_samples, tile_unstable, grad_count);
}

}  // namespace star_uvt

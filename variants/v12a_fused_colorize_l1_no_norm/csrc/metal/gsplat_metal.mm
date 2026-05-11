#import <Foundation/Foundation.h>

#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/native/mps/MetalShaderLibrary.h>
#include <torch/mps.h>

#include <cstdlib>
#include <limits>
#include <mutex>
#include <string>
#include <vector>

#include "shared/common.h"

namespace gsplat {
namespace {
using at::native::mps::DynamicMetalShaderLibrary;
using at::native::mps::MetalKernelFunction;

struct ShaderConfig {
  int tile_size;
  int threads;
  int chunk;
  int fast_cap;
  int simdgroups;
  int feature_cap;
};

int env_int(const char* name, int default_value) {
  const char* raw = std::getenv(name);
  if (raw == nullptr || raw[0] == '\0') return default_value;
  return std::atoi(raw);
}

ShaderConfig& shader_config() {
  static ShaderConfig cfg = []() {
    ShaderConfig c;
    c.tile_size = env_int("GSP_TILE_SIZE", 16);
    TORCH_CHECK(c.tile_size == 8 || c.tile_size == 16 || c.tile_size == 32,
                "GSP_TILE_SIZE must be one of 8, 16, 32; got ", c.tile_size);
    c.threads = c.tile_size * c.tile_size;
    TORCH_CHECK(c.threads <= 1024, "GSP threads exceed 1024: ", c.threads);
    c.chunk = env_int("GSP_CHUNK", 64);
    TORCH_CHECK(c.chunk > 0, "GSP_CHUNK must be positive");
    c.fast_cap = env_int("GSP_FAST_CAP", 2048);
    TORCH_CHECK(c.fast_cap > 0, "GSP_FAST_CAP must be positive");
    c.simdgroups = (c.threads + 31) / 32;
    c.feature_cap = env_int("GSP_FEATURE_CAP", 64);
    TORCH_CHECK(c.feature_cap > 0, "GSP_FEATURE_CAP must be positive");
    return c;
  }();
  return cfg;
}

std::string load_shader_source() {
  auto& cfg = shader_config();
  NSString* metalPath = [[NSString stringWithUTF8String:__FILE__] stringByDeletingLastPathComponent];
  metalPath = [metalPath stringByAppendingPathComponent:@"gsplat_v12a_fused_colorize_l1_no_norm_kernels.metal"];
  NSError* err = nil;
  NSString* src = [NSString stringWithContentsOfFile:metalPath encoding:NSUTF8StringEncoding error:&err];
  TORCH_CHECK(src != nil, "Failed to read gsplat_v12a_fused_colorize_l1_no_norm_kernels.metal: ", err.localizedDescription.UTF8String);

  std::string preamble;
  preamble += "#define GSP_TILE_SIZE " + std::to_string(cfg.tile_size) + "u\n";
  preamble += "#define GSP_THREADS " + std::to_string(cfg.threads) + "u\n";
  preamble += "#define GSP_FAST_CAP " + std::to_string(cfg.fast_cap) + "u\n";
  preamble += "#define GSP_CHUNK " + std::to_string(cfg.chunk) + "u\n";
  preamble += "#define GSP_SIMD_WIDTH 32u\n";
  preamble += "#define GSP_SIMDGROUPS " + std::to_string(cfg.simdgroups) + "u\n";
  preamble += "#define GSP_FEATURE_CAP " + std::to_string(cfg.feature_cap) + "u\n";
  preamble += "\n";
  return preamble + std::string([src UTF8String]);
}

struct MetalV6RefinedFeaturesKernels {
  std::shared_ptr<MetalKernelFunction> count_tiles;
  std::shared_ptr<MetalKernelFunction> init_fixed_bin_offsets;
  std::shared_ptr<MetalKernelFunction> emit_binned_ids;
  std::shared_ptr<MetalKernelFunction> tile_fast_forward_eval;
  std::shared_ptr<MetalKernelFunction> tile_fast_forward_state;
  std::shared_ptr<MetalKernelFunction> tile_fast_backward_saved;
  std::shared_ptr<MetalKernelFunction> tile_active_forward_eval;
  std::shared_ptr<MetalKernelFunction> tile_active_forward_state;
  std::shared_ptr<MetalKernelFunction> tile_active_backward_saved;
  std::shared_ptr<MetalKernelFunction> tile_overflow_forward;
  std::shared_ptr<MetalKernelFunction> tile_overflow_backward;
  std::shared_ptr<MetalKernelFunction> fused_no_norm_l1_grad;
  std::shared_ptr<MetalKernelFunction> dssim_forward_grad;
};

MetalV6RefinedFeaturesKernels& kernels() {
  static std::once_flag once;
  static std::unique_ptr<DynamicMetalShaderLibrary> lib;
  static MetalV6RefinedFeaturesKernels out;
  std::call_once(once, []() {
    lib = std::make_unique<DynamicMetalShaderLibrary>(load_shader_source());
    out.count_tiles = lib->getKernelFunction("v12a_fused_colorize_l1_no_norm_count_tiles");
    out.init_fixed_bin_offsets = lib->getKernelFunction("v12a_fused_colorize_l1_no_norm_init_fixed_bin_offsets");
    out.emit_binned_ids = lib->getKernelFunction("v12a_fused_colorize_l1_no_norm_emit_binned_ids");
    out.tile_fast_forward_eval = lib->getKernelFunction("v12a_fused_colorize_l1_no_norm_tile_fast_forward_eval");
    out.tile_fast_forward_state = lib->getKernelFunction("v12a_fused_colorize_l1_no_norm_tile_fast_forward_state");
    out.tile_fast_backward_saved = lib->getKernelFunction("v12a_fused_colorize_l1_no_norm_tile_fast_backward_saved");
    out.tile_active_forward_eval = lib->getKernelFunction("v12a_fused_colorize_l1_no_norm_tile_active_forward_eval");
    out.tile_active_forward_state = lib->getKernelFunction("v12a_fused_colorize_l1_no_norm_tile_active_forward_state");
    out.tile_active_backward_saved = lib->getKernelFunction("v12a_fused_colorize_l1_no_norm_tile_active_backward_saved");
    out.tile_overflow_forward = lib->getKernelFunction("v12a_fused_colorize_l1_no_norm_tile_overflow_forward");
    out.tile_overflow_backward = lib->getKernelFunction("v12a_fused_colorize_l1_no_norm_tile_overflow_backward");
    out.fused_no_norm_l1_grad = lib->getKernelFunction("v12a_fused_colorize_l1_no_norm_fused_no_norm_l1_grad");
    out.dssim_forward_grad = lib->getKernelFunction("v12a_fused_colorize_l1_no_norm_dssim_forward_grad");
  });
  return out;
}

void check_inputs(
    const torch::Tensor& means2d,
    const torch::Tensor& conics,
    const torch::Tensor& colors,
    const torch::Tensor& opacities) {
  TORCH_CHECK(means2d.device().is_mps(), "means2d must be on MPS");
  TORCH_CHECK(conics.device().is_mps(), "conics must be on MPS");
  TORCH_CHECK(colors.device().is_mps(), "colors must be on MPS");
  TORCH_CHECK(opacities.device().is_mps(), "opacities must be on MPS");
  TORCH_CHECK(means2d.scalar_type() == torch::kFloat32, "means2d must be float32");
  TORCH_CHECK(conics.scalar_type() == torch::kFloat32, "conics must be float32");
  TORCH_CHECK(colors.scalar_type() == torch::kFloat32, "colors must be float32");
  TORCH_CHECK(opacities.scalar_type() == torch::kFloat32, "opacities must be float32");
  TORCH_CHECK(means2d.dim() == 2 && means2d.size(1) == 2, "means2d must be [BG,2]");
  TORCH_CHECK(conics.dim() == 2 && conics.size(1) == 3, "conics must be [BG,3]");
  TORCH_CHECK(colors.dim() == 2 && colors.size(1) > 0, "colors/features must be [BG,F] with F > 0");
  TORCH_CHECK(opacities.dim() == 1, "opacities must be [BG]");
  TORCH_CHECK(means2d.size(0) == conics.size(0) && means2d.size(0) == colors.size(0) && means2d.size(0) == opacities.size(0),
              "All inputs must agree on flattened BG");
}

void check_meta_inputs(const ParsedMeta& meta, const ShaderConfig& sc, const torch::Tensor& means2d, const torch::Tensor& colors) {
  TORCH_CHECK(meta.tile_size == sc.tile_size, "meta.tile_size must match shader tile size ", sc.tile_size);
  TORCH_CHECK(meta.max_fast_pairs <= sc.fast_cap, "meta.max_fast_pairs exceeds shader compile-time cap ", sc.fast_cap);
  TORCH_CHECK(meta.feature_dim > 0, "meta.feature_dim must be positive");
  TORCH_CHECK(meta.feature_dim <= sc.feature_cap,
              "meta.feature_dim=", meta.feature_dim, " exceeds shader feature cap ", sc.feature_cap);
  TORCH_CHECK(colors.size(1) == meta.feature_dim,
              "colors/features trailing dimension ", colors.size(1), " does not match meta.feature_dim ", meta.feature_dim);
  TORCH_CHECK(meta.batch_size > 0, "meta.batch_size must be positive");
  TORCH_CHECK(meta.gaussians_per_batch > 0, "meta.gaussians_per_batch must be positive");
  TORCH_CHECK(meta.tiles_per_image == meta.tiles_y * meta.tiles_x, "meta.tiles_per_image mismatch");
  TORCH_CHECK(meta.gaussians == means2d.size(0), "flattened Gaussian count mismatch");
}

void check_image_grad(const torch::Tensor& grad, const ParsedMeta& meta, const char* name) {
  TORCH_CHECK(grad.dim() == 4, name, " must be rank-4");
  TORCH_CHECK(grad.size(0) == meta.batch_size && grad.size(1) == meta.height &&
              grad.size(2) == meta.width && grad.size(3) == meta.feature_dim,
              name, " shape must be [B,H,W,F] matching metadata");
}

void check_alpha_grad(const torch::Tensor& grad, const ParsedMeta& meta, const char* name) {
  TORCH_CHECK(grad.dim() == 3, name, " must be rank-3");
  TORCH_CHECK(grad.size(0) == meta.batch_size && grad.size(1) == meta.height &&
              grad.size(2) == meta.width,
              name, " shape must be [B,H,W] matching metadata");
}

void check_tile_grad(const torch::Tensor& grad, int64_t tile_count, const ParsedMeta& meta, const char* name) {
  TORCH_CHECK(grad.dim() == 4, name, " must be rank-4");
  TORCH_CHECK(grad.size(0) == tile_count && grad.size(1) == meta.tile_size &&
              grad.size(2) == meta.tile_size && grad.size(3) == meta.feature_dim,
              name, " shape must be [To,tile,tile,F] matching metadata");
}

void check_tile_alpha_grad(const torch::Tensor& grad, int64_t tile_count, const ParsedMeta& meta, const char* name) {
  TORCH_CHECK(grad.dim() == 3, name, " must be rank-3");
  TORCH_CHECK(grad.size(0) == tile_count && grad.size(1) == meta.tile_size &&
              grad.size(2) == meta.tile_size,
              name, " shape must be [To,tile,tile] matching metadata");
}

void check_fused_no_norm_l1_inputs(
    const torch::Tensor& features,
    const torch::Tensor& alpha,
    const torch::Tensor& target_rgb,
    const torch::Tensor& background_rgb,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    const ShaderConfig& sc) {
  TORCH_CHECK(features.device().is_mps(), "features must be on MPS");
  TORCH_CHECK(alpha.device().is_mps(), "alpha must be on MPS");
  TORCH_CHECK(target_rgb.device().is_mps(), "target_rgb must be on MPS");
  TORCH_CHECK(background_rgb.device().is_mps(), "background_rgb must be on MPS");
  TORCH_CHECK(weight.device().is_mps(), "weight must be on MPS");
  TORCH_CHECK(bias.device().is_mps(), "bias must be on MPS");
  TORCH_CHECK(features.scalar_type() == torch::kFloat32, "features must be float32");
  TORCH_CHECK(alpha.scalar_type() == torch::kFloat32, "alpha must be float32");
  TORCH_CHECK(target_rgb.scalar_type() == torch::kFloat32, "target_rgb must be float32");
  TORCH_CHECK(background_rgb.scalar_type() == torch::kFloat32, "background_rgb must be float32");
  TORCH_CHECK(weight.scalar_type() == torch::kFloat32, "weight must be float32");
  TORCH_CHECK(bias.scalar_type() == torch::kFloat32, "bias must be float32");
  TORCH_CHECK(features.is_contiguous(), "features must be contiguous NHWF");
  TORCH_CHECK(alpha.is_contiguous(), "alpha must be contiguous NHW");
  TORCH_CHECK(target_rgb.is_contiguous(), "target_rgb must be contiguous NCHW");
  TORCH_CHECK(background_rgb.is_contiguous(), "background_rgb must be contiguous NCHW");
  TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous [3,F]");
  TORCH_CHECK(bias.is_contiguous(), "bias must be contiguous [3]");
  TORCH_CHECK(features.dim() == 4, "features must have shape [N,H,W,F]");
  TORCH_CHECK(alpha.dim() == 3, "alpha must have shape [N,H,W]");
  TORCH_CHECK(target_rgb.dim() == 4, "target_rgb must have shape [N,3,H,W]");
  TORCH_CHECK(background_rgb.dim() == 4, "background_rgb must have shape [N,3,H,W]");
  TORCH_CHECK(weight.dim() == 2, "weight must have shape [3,F]");
  TORCH_CHECK(bias.dim() == 1, "bias must have shape [3]");

  const int64_t N = features.size(0);
  const int64_t H = features.size(1);
  const int64_t W = features.size(2);
  const int64_t F = features.size(3);
  TORCH_CHECK(N > 0 && H > 0 && W > 0 && F > 0, "features dimensions must be positive");
  TORCH_CHECK(F <= sc.feature_cap, "feature dim ", F, " exceeds GSP_FEATURE_CAP ", sc.feature_cap);
  TORCH_CHECK(alpha.size(0) == N && alpha.size(1) == H && alpha.size(2) == W,
              "alpha must match features [N,H,W]");
  TORCH_CHECK(target_rgb.size(0) == N && target_rgb.size(1) == 3 && target_rgb.size(2) == H && target_rgb.size(3) == W,
              "target_rgb must match [N,3,H,W]");
  TORCH_CHECK(background_rgb.size(0) == N && background_rgb.size(1) == 3 &&
              background_rgb.size(2) == H && background_rgb.size(3) == W,
              "background_rgb must match [N,3,H,W]");
  TORCH_CHECK(weight.size(0) == 3 && weight.size(1) == F, "weight must match [3,F]");
  TORCH_CHECK(bias.size(0) == 3, "bias must match [3]");
}

void check_dssim_inputs(
    const torch::Tensor& prediction,
    const torch::Tensor& target,
    int64_t window_size) {
  TORCH_CHECK(prediction.device().is_mps(), "prediction must be on MPS");
  TORCH_CHECK(target.device().is_mps(), "target must be on MPS");
  TORCH_CHECK(prediction.scalar_type() == torch::kFloat32, "prediction must be float32");
  TORCH_CHECK(target.scalar_type() == torch::kFloat32, "target must be float32");
  TORCH_CHECK(prediction.is_contiguous(), "prediction must be contiguous NCHW");
  TORCH_CHECK(target.is_contiguous(), "target must be contiguous NCHW");
  TORCH_CHECK(prediction.dim() == 4, "prediction must have shape [N,C,H,W]");
  TORCH_CHECK(target.sizes() == prediction.sizes(), "target must match prediction shape");
  TORCH_CHECK(window_size > 0 && (window_size % 2) == 1,
              "window_size must be a positive odd integer; got ", window_size);
  TORCH_CHECK(prediction.size(0) > 0 && prediction.size(1) > 0 &&
              prediction.size(2) > 0 && prediction.size(3) > 0,
              "prediction dimensions must be positive");
  const int64_t pad = window_size / 2;
  TORCH_CHECK(pad < prediction.size(2) && pad < prediction.size(3),
              "reflect-pad DSSIM requires window_size/2 < height and width; got window_size=",
              window_size, " for image ", prediction.size(2), "x", prediction.size(3));
}

void check_aux_i32(const torch::Tensor& t, const char* name) {
  TORCH_CHECK(t.device().is_mps(), name, " must be on MPS");
  TORCH_CHECK(t.scalar_type() == torch::kInt32, name, " must be int32");
}

template <typename Fn>
void launch(std::shared_ptr<MetalKernelFunction> fn, Fn&& body) {
  fn->runCommandBlock([&]() {
    fn->startEncoding();
    body(*fn);
  });
}

std::tuple<torch::Tensor, torch::Tensor> make_background_outputs(
    const ParsedMeta& meta,
    const torch::Tensor& meta_host_f32,
    const torch::TensorOptions& opts_f) {
  auto out = torch::empty({meta.batch_size, meta.height, meta.width, meta.feature_dim}, opts_f);
  auto out_alpha = torch::zeros({meta.batch_size, meta.height, meta.width}, opts_f);
  auto* fp = meta_host_f32.data_ptr<float>();
  for (int f = 0; f < meta.feature_dim; ++f) {
    out.select(-1, f).fill_(fp[4 + f]);
  }
  return std::make_tuple(out, out_alpha);
}

}  // namespace

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> metal_bin(
    const torch::Tensor& means2d,
    const torch::Tensor& conics,
    const torch::Tensor& colors,
    const torch::Tensor& opacities,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32,
    const torch::Tensor& meta_host_i32,
    const torch::Tensor& meta_host_f32) {
  check_inputs(means2d, conics, colors, opacities);
  auto meta = parse_meta(meta_host_i32, meta_host_f32);
  auto& sc = shader_config();
  check_meta_inputs(meta, sc, means2d, colors);
  TORCH_CHECK(meta_f32.numel() >= 4 + sc.feature_cap, "meta_f32 must include alpha/transmittance scalars and feature-cap background");

  auto& k = kernels();
  auto opts_f = means2d.options().dtype(torch::kFloat32);
  auto opts_i32 = means2d.options().dtype(torch::kInt32);
  constexpr uint64_t kBinThreads = 256ull;

  const int64_t G = means2d.size(0);
  const int64_t T = meta.tile_count;
  const int64_t fixed_slots = T * (int64_t)meta.max_fast_pairs;
  TORCH_CHECK(
      fixed_slots <= (int64_t)std::numeric_limits<int32_t>::max(),
      "fixedbin requires tile_count * max_fast_pairs to fit int32 offsets; got ",
      fixed_slots);

  auto bbox = torch::empty({G, 4}, opts_i32);
  auto tau = torch::empty({G}, opts_f);
  auto tile_counts = torch::zeros({T}, opts_i32);

  launch(k.count_tiles, [&](MetalKernelFunction& fn) {
    fn.setArg(0, means2d);
    fn.setArg(1, conics);
    fn.setArg(2, opacities);
    fn.setArg(3, meta_i32);
    fn.setArg(4, meta_f32);
    fn.setArg(5, bbox);
    fn.setArg(6, tau);
    fn.setArg(7, tile_counts);
    fn.dispatch((uint64_t)G, kBinThreads);
  });

  auto tile_offsets = torch::empty({T + 1}, opts_i32);
  auto tile_cursors = torch::empty({T}, opts_i32);
  auto binned_ids = torch::empty({fixed_slots}, opts_i32);

  if (T > 0) {
    launch(k.init_fixed_bin_offsets, [&](MetalKernelFunction& fn) {
      fn.setArg(0, meta_i32);
      fn.setArg(1, tile_offsets);
      fn.setArg(2, tile_cursors);
      fn.dispatch((uint64_t)(T + 1), kBinThreads);
    });
    launch(k.emit_binned_ids, [&](MetalKernelFunction& fn) {
      fn.setArg(0, means2d);
      fn.setArg(1, conics);
      fn.setArg(2, bbox);
      fn.setArg(3, tau);
      fn.setArg(4, meta_i32);
      fn.setArg(5, tile_cursors);
      fn.setArg(6, binned_ids);
      fn.dispatch((uint64_t)G, kBinThreads);
    });
  }

  return std::make_tuple(tile_counts, tile_offsets, binned_ids);
}

std::tuple<torch::Tensor, torch::Tensor> metal_render_fast_forward_eval(
    const torch::Tensor& means2d,
    const torch::Tensor& conics,
    const torch::Tensor& colors,
    const torch::Tensor& opacities,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32,
    const torch::Tensor& meta_host_i32,
    const torch::Tensor& meta_host_f32,
    const torch::Tensor& tile_counts,
    const torch::Tensor& tile_offsets,
    const torch::Tensor& binned_ids) {
  check_inputs(means2d, conics, colors, opacities);
  check_aux_i32(tile_counts, "tile_counts");
  check_aux_i32(tile_offsets, "tile_offsets");
  check_aux_i32(binned_ids, "binned_ids");
  auto meta = parse_meta(meta_host_i32, meta_host_f32);
  auto& sc = shader_config();
  check_meta_inputs(meta, sc, means2d, colors);
  auto& k = kernels();
  auto opts_f = means2d.options().dtype(torch::kFloat32);

  auto out = torch::empty({meta.batch_size, meta.height, meta.width, meta.feature_dim}, opts_f);
  auto out_alpha = torch::empty({meta.batch_size, meta.height, meta.width}, opts_f);
  if (meta.tile_count > 0) {
    launch(k.tile_fast_forward_eval, [&](MetalKernelFunction& fn) {
      fn.setArg(0, tile_counts);
      fn.setArg(1, tile_offsets);
      fn.setArg(2, binned_ids);
      fn.setArg(3, means2d);
      fn.setArg(4, conics);
      fn.setArg(5, colors);
      fn.setArg(6, opacities);
      fn.setArg(7, meta_i32);
      fn.setArg(8, meta_f32);
      fn.setArg(9, out);
      fn.setArg(10, out_alpha);
      fn.dispatch((uint64_t)meta.tile_count * (uint64_t)sc.threads, (uint64_t)sc.threads);
    });
  }
  return std::make_tuple(out, out_alpha);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> metal_render_fast_forward_state(
    const torch::Tensor& means2d,
    const torch::Tensor& conics,
    const torch::Tensor& colors,
    const torch::Tensor& opacities,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32,
    const torch::Tensor& meta_host_i32,
    const torch::Tensor& meta_host_f32,
    torch::Tensor& binned_ids,
    const torch::Tensor& tile_counts,
    const torch::Tensor& tile_offsets) {
  check_inputs(means2d, conics, colors, opacities);
  check_aux_i32(tile_counts, "tile_counts");
  check_aux_i32(tile_offsets, "tile_offsets");
  check_aux_i32(binned_ids, "binned_ids");
  auto meta = parse_meta(meta_host_i32, meta_host_f32);
  auto& sc = shader_config();
  check_meta_inputs(meta, sc, means2d, colors);
  auto& k = kernels();
  auto opts_f = means2d.options().dtype(torch::kFloat32);
  auto opts_i32 = means2d.options().dtype(torch::kInt32);

  auto out = torch::empty({meta.batch_size, meta.height, meta.width, meta.feature_dim}, opts_f);
  auto out_alpha = torch::empty({meta.batch_size, meta.height, meta.width}, opts_f);
  auto tile_stop_counts = torch::zeros({meta.tile_count}, opts_i32);

  if (meta.tile_count > 0) {
    launch(k.tile_fast_forward_state, [&](MetalKernelFunction& fn) {
      fn.setArg(0, tile_counts);
      fn.setArg(1, tile_offsets);
      fn.setArg(2, binned_ids);
      fn.setArg(3, means2d);
      fn.setArg(4, conics);
      fn.setArg(5, colors);
      fn.setArg(6, opacities);
      fn.setArg(7, meta_i32);
      fn.setArg(8, meta_f32);
      fn.setArg(9, out);
      fn.setArg(10, out_alpha);
      fn.setArg(11, tile_stop_counts);
      fn.dispatch((uint64_t)meta.tile_count * (uint64_t)sc.threads, (uint64_t)sc.threads);
    });
  }
  return std::make_tuple(out, out_alpha, tile_stop_counts);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> metal_render_fast_backward_saved(
    const torch::Tensor& grad_features,
    const torch::Tensor& grad_alpha,
    const torch::Tensor& means2d,
    const torch::Tensor& conics,
    const torch::Tensor& colors,
    const torch::Tensor& opacities,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32,
    const torch::Tensor& meta_host_i32,
    const torch::Tensor& meta_host_f32,
    const torch::Tensor& tile_counts,
    const torch::Tensor& tile_offsets,
    const torch::Tensor& binned_ids,
    const torch::Tensor& tile_stop_counts) {
  check_inputs(means2d, conics, colors, opacities);
  TORCH_CHECK(grad_features.device().is_mps(), "grad_features must be on MPS");
  TORCH_CHECK(grad_alpha.device().is_mps(), "grad_alpha must be on MPS");
  check_aux_i32(tile_counts, "tile_counts");
  check_aux_i32(tile_offsets, "tile_offsets");
  check_aux_i32(binned_ids, "binned_ids");
  check_aux_i32(tile_stop_counts, "tile_stop_counts");

  auto meta = parse_meta(meta_host_i32, meta_host_f32);
  auto& sc = shader_config();
  check_meta_inputs(meta, sc, means2d, colors);
  check_image_grad(grad_features, meta, "grad_features");
  check_alpha_grad(grad_alpha, meta, "grad_alpha");
  auto& k = kernels();
  auto opts_f = means2d.options().dtype(torch::kFloat32);

  auto g_means2d = torch::zeros_like(means2d, opts_f);
  auto g_conics = torch::zeros_like(conics, opts_f);
  const bool skip_color_grad = (meta.reserved0 & 1) != 0;
  auto g_colors = skip_color_grad ? torch::empty({0}, opts_f) : torch::zeros_like(colors, opts_f);
  auto g_colors_arg = skip_color_grad ? torch::empty({1}, opts_f) : g_colors;
  auto g_opacities = torch::zeros_like(opacities, opts_f);

  if (meta.tile_count > 0) {
    launch(k.tile_fast_backward_saved, [&](MetalKernelFunction& fn) {
      fn.setArg(0, grad_features.contiguous());
      fn.setArg(1, grad_alpha.contiguous());
      fn.setArg(2, tile_counts);
      fn.setArg(3, tile_offsets);
      fn.setArg(4, binned_ids);
      fn.setArg(5, tile_stop_counts);
      fn.setArg(6, means2d);
      fn.setArg(7, conics);
      fn.setArg(8, colors);
      fn.setArg(9, opacities);
      fn.setArg(10, meta_i32);
      fn.setArg(11, meta_f32);
      fn.setArg(12, g_means2d);
      fn.setArg(13, g_conics);
      fn.setArg(14, g_colors_arg);
      fn.setArg(15, g_opacities);
      fn.dispatch((uint64_t)meta.tile_count * (uint64_t)sc.threads, (uint64_t)sc.threads);
    });
  }

  return std::make_tuple(g_means2d, g_conics, g_colors, g_opacities);
}

std::tuple<torch::Tensor, torch::Tensor> metal_render_active_forward_eval(
    const torch::Tensor& means2d,
    const torch::Tensor& conics,
    const torch::Tensor& colors,
    const torch::Tensor& opacities,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32,
    const torch::Tensor& meta_host_i32,
    const torch::Tensor& meta_host_f32,
    const torch::Tensor& active_tile_ids,
    const torch::Tensor& tile_counts,
    const torch::Tensor& tile_offsets,
    const torch::Tensor& binned_ids) {
  check_inputs(means2d, conics, colors, opacities);
  check_aux_i32(active_tile_ids, "active_tile_ids");
  check_aux_i32(tile_counts, "tile_counts");
  check_aux_i32(tile_offsets, "tile_offsets");
  check_aux_i32(binned_ids, "binned_ids");
  auto meta = parse_meta(meta_host_i32, meta_host_f32);
  auto& sc = shader_config();
  check_meta_inputs(meta, sc, means2d, colors);
  auto& k = kernels();
  auto opts_f = means2d.options().dtype(torch::kFloat32);

  auto outputs = make_background_outputs(meta, meta_host_f32, opts_f);
  auto out = std::get<0>(outputs);
  auto out_alpha = std::get<1>(outputs);
  const int64_t Ta = active_tile_ids.size(0);
  if (Ta > 0) {
    launch(k.tile_active_forward_eval, [&](MetalKernelFunction& fn) {
      fn.setArg(0, active_tile_ids);
      fn.setArg(1, tile_counts);
      fn.setArg(2, tile_offsets);
      fn.setArg(3, binned_ids);
      fn.setArg(4, means2d);
      fn.setArg(5, conics);
      fn.setArg(6, colors);
      fn.setArg(7, opacities);
      fn.setArg(8, meta_i32);
      fn.setArg(9, meta_f32);
      fn.setArg(10, out);
      fn.setArg(11, out_alpha);
      fn.dispatch((uint64_t)Ta * (uint64_t)sc.threads, (uint64_t)sc.threads);
    });
  }
  return std::make_tuple(out, out_alpha);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> metal_render_active_forward_state(
    const torch::Tensor& means2d,
    const torch::Tensor& conics,
    const torch::Tensor& colors,
    const torch::Tensor& opacities,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32,
    const torch::Tensor& meta_host_i32,
    const torch::Tensor& meta_host_f32,
    torch::Tensor& binned_ids,
    const torch::Tensor& active_tile_ids,
    const torch::Tensor& tile_counts,
    const torch::Tensor& tile_offsets) {
  check_inputs(means2d, conics, colors, opacities);
  check_aux_i32(active_tile_ids, "active_tile_ids");
  check_aux_i32(tile_counts, "tile_counts");
  check_aux_i32(tile_offsets, "tile_offsets");
  check_aux_i32(binned_ids, "binned_ids");
  auto meta = parse_meta(meta_host_i32, meta_host_f32);
  auto& sc = shader_config();
  check_meta_inputs(meta, sc, means2d, colors);
  auto& k = kernels();
  auto opts_f = means2d.options().dtype(torch::kFloat32);
  auto opts_i32 = means2d.options().dtype(torch::kInt32);

  auto outputs = make_background_outputs(meta, meta_host_f32, opts_f);
  auto out = std::get<0>(outputs);
  auto out_alpha = std::get<1>(outputs);
  auto tile_stop_counts = torch::zeros({meta.tile_count}, opts_i32);
  const int64_t Ta = active_tile_ids.size(0);
  if (Ta > 0) {
    launch(k.tile_active_forward_state, [&](MetalKernelFunction& fn) {
      fn.setArg(0, active_tile_ids);
      fn.setArg(1, tile_counts);
      fn.setArg(2, tile_offsets);
      fn.setArg(3, binned_ids);
      fn.setArg(4, means2d);
      fn.setArg(5, conics);
      fn.setArg(6, colors);
      fn.setArg(7, opacities);
      fn.setArg(8, meta_i32);
      fn.setArg(9, meta_f32);
      fn.setArg(10, out);
      fn.setArg(11, out_alpha);
      fn.setArg(12, tile_stop_counts);
      fn.dispatch((uint64_t)Ta * (uint64_t)sc.threads, (uint64_t)sc.threads);
    });
  }
  return std::make_tuple(out, out_alpha, tile_stop_counts);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> metal_render_active_backward_saved(
    const torch::Tensor& grad_features,
    const torch::Tensor& grad_alpha,
    const torch::Tensor& means2d,
    const torch::Tensor& conics,
    const torch::Tensor& colors,
    const torch::Tensor& opacities,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32,
    const torch::Tensor& meta_host_i32,
    const torch::Tensor& meta_host_f32,
    const torch::Tensor& active_tile_ids,
    const torch::Tensor& tile_counts,
    const torch::Tensor& tile_offsets,
    const torch::Tensor& binned_ids,
    const torch::Tensor& tile_stop_counts) {
  check_inputs(means2d, conics, colors, opacities);
  TORCH_CHECK(grad_features.device().is_mps(), "grad_features must be on MPS");
  TORCH_CHECK(grad_alpha.device().is_mps(), "grad_alpha must be on MPS");
  check_aux_i32(active_tile_ids, "active_tile_ids");
  check_aux_i32(tile_counts, "tile_counts");
  check_aux_i32(tile_offsets, "tile_offsets");
  check_aux_i32(binned_ids, "binned_ids");
  check_aux_i32(tile_stop_counts, "tile_stop_counts");

  auto meta = parse_meta(meta_host_i32, meta_host_f32);
  auto& sc = shader_config();
  check_meta_inputs(meta, sc, means2d, colors);
  check_image_grad(grad_features, meta, "grad_features");
  check_alpha_grad(grad_alpha, meta, "grad_alpha");
  auto& k = kernels();
  auto opts_f = means2d.options().dtype(torch::kFloat32);

  auto g_means2d = torch::zeros_like(means2d, opts_f);
  auto g_conics = torch::zeros_like(conics, opts_f);
  const bool skip_color_grad = (meta.reserved0 & 1) != 0;
  auto g_colors = skip_color_grad ? torch::empty({0}, opts_f) : torch::zeros_like(colors, opts_f);
  auto g_colors_arg = skip_color_grad ? torch::empty({1}, opts_f) : g_colors;
  auto g_opacities = torch::zeros_like(opacities, opts_f);

  const int64_t Ta = active_tile_ids.size(0);
  if (Ta > 0) {
    launch(k.tile_active_backward_saved, [&](MetalKernelFunction& fn) {
      fn.setArg(0, grad_features.contiguous());
      fn.setArg(1, grad_alpha.contiguous());
      fn.setArg(2, active_tile_ids);
      fn.setArg(3, tile_counts);
      fn.setArg(4, tile_offsets);
      fn.setArg(5, binned_ids);
      fn.setArg(6, tile_stop_counts);
      fn.setArg(7, means2d);
      fn.setArg(8, conics);
      fn.setArg(9, colors);
      fn.setArg(10, opacities);
      fn.setArg(11, meta_i32);
      fn.setArg(12, meta_f32);
      fn.setArg(13, g_means2d);
      fn.setArg(14, g_conics);
      fn.setArg(15, g_colors_arg);
      fn.setArg(16, g_opacities);
      fn.dispatch((uint64_t)Ta * (uint64_t)sc.threads, (uint64_t)sc.threads);
    });
  }
  return std::make_tuple(g_means2d, g_conics, g_colors, g_opacities);
}

std::tuple<torch::Tensor, torch::Tensor> metal_render_overflow_forward(
    const torch::Tensor& means2d,
    const torch::Tensor& conics,
    const torch::Tensor& colors,
    const torch::Tensor& opacities,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32,
    const torch::Tensor& meta_host_i32,
    const torch::Tensor& meta_host_f32,
    const torch::Tensor& overflow_tile_ids,
    const torch::Tensor& overflow_tile_offsets,
    const torch::Tensor& overflow_sorted_ids) {
  check_inputs(means2d, conics, colors, opacities);
  check_aux_i32(overflow_tile_ids, "overflow_tile_ids");
  check_aux_i32(overflow_tile_offsets, "overflow_tile_offsets");
  check_aux_i32(overflow_sorted_ids, "overflow_sorted_ids");

  auto meta = parse_meta(meta_host_i32, meta_host_f32);
  auto& sc = shader_config();
  check_meta_inputs(meta, sc, means2d, colors);
  auto& k = kernels();
  auto opts_f = means2d.options().dtype(torch::kFloat32);

  const int64_t To = overflow_tile_ids.size(0);
  auto out = torch::zeros({To, meta.tile_size, meta.tile_size, meta.feature_dim}, opts_f);
  auto out_alpha = torch::zeros({To, meta.tile_size, meta.tile_size}, opts_f);
  if (To > 0) {
    launch(k.tile_overflow_forward, [&](MetalKernelFunction& fn) {
      fn.setArg(0, overflow_tile_ids);
      fn.setArg(1, overflow_tile_offsets);
      fn.setArg(2, overflow_sorted_ids);
      fn.setArg(3, means2d);
      fn.setArg(4, conics);
      fn.setArg(5, colors);
      fn.setArg(6, opacities);
      fn.setArg(7, meta_i32);
      fn.setArg(8, meta_f32);
      fn.setArg(9, out);
      fn.setArg(10, out_alpha);
      fn.dispatch((uint64_t)To * (uint64_t)sc.threads, (uint64_t)sc.threads);
    });
  }
  return std::make_tuple(out, out_alpha);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> metal_render_overflow_backward(
    const torch::Tensor& grad_features_tiles,
    const torch::Tensor& grad_alpha_tiles,
    const torch::Tensor& means2d,
    const torch::Tensor& conics,
    const torch::Tensor& colors,
    const torch::Tensor& opacities,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32,
    const torch::Tensor& meta_host_i32,
    const torch::Tensor& meta_host_f32,
    const torch::Tensor& overflow_tile_ids,
    const torch::Tensor& overflow_tile_offsets,
    const torch::Tensor& overflow_sorted_ids) {
  check_inputs(means2d, conics, colors, opacities);
  TORCH_CHECK(grad_features_tiles.device().is_mps(), "grad_features_tiles must be on MPS");
  TORCH_CHECK(grad_alpha_tiles.device().is_mps(), "grad_alpha_tiles must be on MPS");
  auto meta = parse_meta(meta_host_i32, meta_host_f32);
  auto& sc = shader_config();
  check_meta_inputs(meta, sc, means2d, colors);
  check_tile_grad(grad_features_tiles, overflow_tile_ids.size(0), meta, "grad_features_tiles");
  check_tile_alpha_grad(grad_alpha_tiles, overflow_tile_ids.size(0), meta, "grad_alpha_tiles");
  auto& k = kernels();
  auto opts_f = means2d.options().dtype(torch::kFloat32);

  auto g_means2d = torch::zeros_like(means2d, opts_f);
  auto g_conics = torch::zeros_like(conics, opts_f);
  const bool skip_color_grad = (meta.reserved0 & 1) != 0;
  auto g_colors = skip_color_grad ? torch::empty({0}, opts_f) : torch::zeros_like(colors, opts_f);
  auto g_colors_arg = skip_color_grad ? torch::empty({1}, opts_f) : g_colors;
  auto g_opacities = torch::zeros_like(opacities, opts_f);

  const int64_t To = overflow_tile_ids.size(0);
  if (To > 0) {
    launch(k.tile_overflow_backward, [&](MetalKernelFunction& fn) {
      fn.setArg(0, grad_features_tiles.contiguous());
      fn.setArg(1, grad_alpha_tiles.contiguous());
      fn.setArg(2, overflow_tile_ids);
      fn.setArg(3, overflow_tile_offsets);
      fn.setArg(4, overflow_sorted_ids);
      fn.setArg(5, means2d);
      fn.setArg(6, conics);
      fn.setArg(7, colors);
      fn.setArg(8, opacities);
      fn.setArg(9, meta_i32);
      fn.setArg(10, meta_f32);
      fn.setArg(11, g_means2d);
      fn.setArg(12, g_conics);
      fn.setArg(13, g_colors_arg);
      fn.setArg(14, g_opacities);
      fn.dispatch((uint64_t)To * (uint64_t)sc.threads, (uint64_t)sc.threads);
    });
  }
  return std::make_tuple(g_means2d, g_conics, g_colors, g_opacities);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> metal_fused_no_norm_l1_grad(
    const torch::Tensor& features,
    const torch::Tensor& alpha,
    const torch::Tensor& target_rgb,
    const torch::Tensor& background_rgb,
    const torch::Tensor& weight,
    const torch::Tensor& bias) {
  auto& sc = shader_config();
  check_fused_no_norm_l1_inputs(features, alpha, target_rgb, background_rgb, weight, bias, sc);
  auto& k = kernels();
  auto opts_f = features.options().dtype(torch::kFloat32);

  const int64_t N = features.size(0);
  const int64_t H = features.size(1);
  const int64_t W = features.size(2);
  const int64_t F = features.size(3);
  const int64_t total_pixels = N * H * W;

  auto loss_per_image = torch::zeros({N}, opts_f);
  auto grad_features = torch::empty_like(features, opts_f);
  auto grad_alpha = torch::empty_like(alpha, opts_f);
  auto grad_weight = torch::zeros_like(weight, opts_f);
  auto grad_bias = torch::zeros_like(bias, opts_f);

  constexpr uint64_t kThreads = 256ull;
  launch(k.fused_no_norm_l1_grad, [&](MetalKernelFunction& fn) {
    fn.setArg(0, features);
    fn.setArg(1, alpha);
    fn.setArg(2, target_rgb);
    fn.setArg(3, background_rgb);
    fn.setArg(4, weight);
    fn.setArg(5, bias);
    fn.setArg(6, loss_per_image);
    fn.setArg(7, grad_features);
    fn.setArg(8, grad_alpha);
    fn.setArg(9, grad_weight);
    fn.setArg(10, grad_bias);
    fn.setArg(11, (int)N);
    fn.setArg(12, (int)H);
    fn.setArg(13, (int)W);
    fn.setArg(14, (int)F);
    fn.dispatch((uint64_t)total_pixels, kThreads);
  });

  return std::make_tuple(loss_per_image, grad_features, grad_alpha, grad_weight, grad_bias);
}

std::tuple<torch::Tensor, torch::Tensor> metal_dssim_forward_grad(
    const torch::Tensor& prediction,
    const torch::Tensor& target,
    int64_t window_size,
    double c1,
    double c2) {
  check_dssim_inputs(prediction, target, window_size);
  auto& k = kernels();
  auto opts_f = prediction.options().dtype(torch::kFloat32);

  const int64_t N = prediction.size(0);
  const int64_t C = prediction.size(1);
  const int64_t H = prediction.size(2);
  const int64_t W = prediction.size(3);
  const int64_t total_centers = N * C * H * W;

  auto loss_per_image = torch::zeros({N}, opts_f);
  auto grad_prediction = torch::zeros_like(prediction, opts_f);

  constexpr uint64_t kThreads = 256ull;
  const int window_size_i32 = static_cast<int>(window_size);
  const float c1_f32 = static_cast<float>(c1);
  const float c2_f32 = static_cast<float>(c2);
  launch(k.dssim_forward_grad, [&](MetalKernelFunction& fn) {
    fn.setArg(0, prediction);
    fn.setArg(1, target);
    fn.setArg(2, loss_per_image);
    fn.setArg(3, grad_prediction);
    fn.setArg(4, (int)N);
    fn.setArg(5, (int)C);
    fn.setArg(6, (int)H);
    fn.setArg(7, (int)W);
    fn.setArg(8, window_size_i32);
    fn.setArg(9, c1_f32);
    fn.setArg(10, c2_f32);
    fn.dispatch((uint64_t)total_centers, kThreads);
  });

  return std::make_tuple(loss_per_image, grad_prediction);
}

}  // namespace gsplat

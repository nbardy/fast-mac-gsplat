#import <Foundation/Foundation.h>

#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/native/mps/MetalShaderLibrary.h>
#include <torch/mps.h>

#include <mutex>
#include <string>
#include <vector>

#include "shared/common.h"

namespace gsplat {
namespace {
using at::native::mps::DynamicMetalShaderLibrary;
using at::native::mps::MetalKernelFunction;

std::string load_shader_source() {
  NSString* metalPath = [[NSString stringWithUTF8String:__FILE__] stringByDeletingLastPathComponent];
  metalPath = [metalPath stringByAppendingPathComponent:@"gsplat_v3_kernels.metal"];
  NSError* err = nil;
  NSString* src = [NSString stringWithContentsOfFile:metalPath encoding:NSUTF8StringEncoding error:&err];
  TORCH_CHECK(src != nil, "Failed to read gsplat_v3_kernels.metal: ", err.localizedDescription.UTF8String);
  return std::string([src UTF8String]);
}

struct MetalV3Kernels {
  std::shared_ptr<MetalKernelFunction> count_tiles;
  std::shared_ptr<MetalKernelFunction> emit_binned_ids;
  std::shared_ptr<MetalKernelFunction> tile_fast_forward;
  std::shared_ptr<MetalKernelFunction> tile_fast_backward;
  std::shared_ptr<MetalKernelFunction> tile_overflow_forward;
  std::shared_ptr<MetalKernelFunction> tile_overflow_backward;
};

MetalV3Kernels& kernels() {
  static std::once_flag once;
  static std::unique_ptr<DynamicMetalShaderLibrary> lib;
  static MetalV3Kernels out;
  std::call_once(once, []() {
    lib = std::make_unique<DynamicMetalShaderLibrary>(load_shader_source());
    out.count_tiles = lib->getKernelFunction("count_tiles");
    out.emit_binned_ids = lib->getKernelFunction("emit_binned_ids");
    out.tile_fast_forward = lib->getKernelFunction("tile_fast_forward");
    out.tile_fast_backward = lib->getKernelFunction("tile_fast_backward");
    out.tile_overflow_forward = lib->getKernelFunction("tile_overflow_forward");
    out.tile_overflow_backward = lib->getKernelFunction("tile_overflow_backward");
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
  TORCH_CHECK(means2d.dim() == 2 && means2d.size(1) == 2, "means2d must be [G,2]");
  TORCH_CHECK(conics.dim() == 2 && conics.size(1) == 3, "conics must be [G,3]");
  TORCH_CHECK(colors.dim() == 2 && colors.size(1) == 3, "colors must be [G,3]");
  TORCH_CHECK(opacities.dim() == 1, "opacities must be [G]");
  TORCH_CHECK(means2d.size(0) == conics.size(0) && means2d.size(0) == colors.size(0) && means2d.size(0) == opacities.size(0),
              "All inputs must agree on G");
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

}  // namespace

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> metal_bin(
    const torch::Tensor& means2d,
    const torch::Tensor& conics,
    const torch::Tensor& colors,
    const torch::Tensor& opacities,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32) {
  check_inputs(means2d, conics, colors, opacities);
  auto meta = parse_meta(meta_i32, meta_f32);
  TORCH_CHECK(meta.tile_size == 16, "v3 Metal path currently requires tile_size=16");
  TORCH_CHECK(meta.max_fast_pairs <= 2048, "meta.max_fast_pairs exceeds shader compile-time cap 2048");
  auto& k = kernels();
  auto opts_f = means2d.options().dtype(torch::kFloat32);
  auto opts_i32 = means2d.options().dtype(torch::kInt32);

  const int64_t G = means2d.size(0);
  const int64_t T = meta.tile_count;

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
    fn.dispatch((uint64_t)G, (uint64_t)256);
  });

  auto offsets_body = torch::cumsum(tile_counts, 0, torch::kInt32);
  auto zero = torch::zeros({1}, opts_i32);
  auto tile_offsets = torch::cat({zero, offsets_body}, 0).contiguous();
  auto tile_cursors = tile_offsets.narrow(0, 0, T).clone();

  int64_t N = tile_offsets[tile_offsets.size(0) - 1].item<int64_t>();
  auto binned_ids = torch::empty({N}, opts_i32);
  if (N > 0) {
    launch(k.emit_binned_ids, [&](MetalKernelFunction& fn) {
      fn.setArg(0, means2d);
      fn.setArg(1, conics);
      fn.setArg(2, bbox);
      fn.setArg(3, tau);
      fn.setArg(4, meta_i32);
      fn.setArg(5, tile_cursors);
      fn.setArg(6, binned_ids);
      fn.dispatch((uint64_t)G, (uint64_t)256);
    });
  }

  return std::make_tuple(tile_counts, tile_offsets, binned_ids);
}

torch::Tensor metal_render_fast_forward(
    const torch::Tensor& means2d,
    const torch::Tensor& conics,
    const torch::Tensor& colors,
    const torch::Tensor& opacities,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32,
    const torch::Tensor& tile_counts,
    const torch::Tensor& tile_offsets,
    const torch::Tensor& binned_ids) {
  check_inputs(means2d, conics, colors, opacities);
  check_aux_i32(tile_counts, "tile_counts");
  check_aux_i32(tile_offsets, "tile_offsets");
  check_aux_i32(binned_ids, "binned_ids");
  auto meta = parse_meta(meta_i32, meta_f32);
  auto& k = kernels();
  auto opts_f = means2d.options().dtype(torch::kFloat32);

  auto out = torch::empty({meta.height, meta.width, 3}, opts_f);
  if (meta.tile_count > 0) {
    launch(k.tile_fast_forward, [&](MetalKernelFunction& fn) {
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
      fn.dispatch((uint64_t)meta.tile_count * 256ull, (uint64_t)256ull);
    });
  }
  return out;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> metal_render_fast_backward(
    const torch::Tensor& grad_out,
    const torch::Tensor& means2d,
    const torch::Tensor& conics,
    const torch::Tensor& colors,
    const torch::Tensor& opacities,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32,
    const torch::Tensor& tile_counts,
    const torch::Tensor& tile_offsets,
    const torch::Tensor& binned_ids) {
  check_inputs(means2d, conics, colors, opacities);
  TORCH_CHECK(grad_out.device().is_mps(), "grad_out must be on MPS");
  auto meta = parse_meta(meta_i32, meta_f32);
  auto& k = kernels();
  auto opts_f = means2d.options().dtype(torch::kFloat32);

  auto g_means2d = torch::zeros_like(means2d, opts_f);
  auto g_conics = torch::zeros_like(conics, opts_f);
  auto g_colors = torch::zeros_like(colors, opts_f);
  auto g_opacities = torch::zeros_like(opacities, opts_f);

  if (meta.tile_count > 0) {
    launch(k.tile_fast_backward, [&](MetalKernelFunction& fn) {
      fn.setArg(0, grad_out.contiguous());
      fn.setArg(1, tile_counts);
      fn.setArg(2, tile_offsets);
      fn.setArg(3, binned_ids);
      fn.setArg(4, means2d);
      fn.setArg(5, conics);
      fn.setArg(6, colors);
      fn.setArg(7, opacities);
      fn.setArg(8, meta_i32);
      fn.setArg(9, meta_f32);
      fn.setArg(10, g_means2d);
      fn.setArg(11, g_conics);
      fn.setArg(12, g_colors);
      fn.setArg(13, g_opacities);
      fn.dispatch((uint64_t)meta.tile_count * 256ull, (uint64_t)256ull);
    });
  }

  return std::make_tuple(g_means2d, g_conics, g_colors, g_opacities);
}

torch::Tensor metal_render_overflow_forward(
    const torch::Tensor& means2d,
    const torch::Tensor& conics,
    const torch::Tensor& colors,
    const torch::Tensor& opacities,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32,
    const torch::Tensor& overflow_tile_ids,
    const torch::Tensor& overflow_tile_offsets,
    const torch::Tensor& overflow_sorted_ids) {
  check_inputs(means2d, conics, colors, opacities);
  check_aux_i32(overflow_tile_ids, "overflow_tile_ids");
  check_aux_i32(overflow_tile_offsets, "overflow_tile_offsets");
  check_aux_i32(overflow_sorted_ids, "overflow_sorted_ids");
  auto meta = parse_meta(meta_i32, meta_f32);
  auto& k = kernels();
  auto opts_f = means2d.options().dtype(torch::kFloat32);

  const int64_t To = overflow_tile_ids.size(0);
  auto out = torch::zeros({To, meta.tile_size, meta.tile_size, 3}, opts_f);
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
      fn.dispatch((uint64_t)To * 256ull, (uint64_t)256ull);
    });
  }
  return out;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> metal_render_overflow_backward(
    const torch::Tensor& grad_tiles,
    const torch::Tensor& means2d,
    const torch::Tensor& conics,
    const torch::Tensor& colors,
    const torch::Tensor& opacities,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32,
    const torch::Tensor& overflow_tile_ids,
    const torch::Tensor& overflow_tile_offsets,
    const torch::Tensor& overflow_sorted_ids) {
  check_inputs(means2d, conics, colors, opacities);
  TORCH_CHECK(grad_tiles.device().is_mps(), "grad_tiles must be on MPS");
  auto meta = parse_meta(meta_i32, meta_f32);
  auto& k = kernels();
  auto opts_f = means2d.options().dtype(torch::kFloat32);

  auto g_means2d = torch::zeros_like(means2d, opts_f);
  auto g_conics = torch::zeros_like(conics, opts_f);
  auto g_colors = torch::zeros_like(colors, opts_f);
  auto g_opacities = torch::zeros_like(opacities, opts_f);

  const int64_t To = overflow_tile_ids.size(0);
  if (To > 0) {
    launch(k.tile_overflow_backward, [&](MetalKernelFunction& fn) {
      fn.setArg(0, grad_tiles.contiguous());
      fn.setArg(1, overflow_tile_ids);
      fn.setArg(2, overflow_tile_offsets);
      fn.setArg(3, overflow_sorted_ids);
      fn.setArg(4, means2d);
      fn.setArg(5, conics);
      fn.setArg(6, colors);
      fn.setArg(7, opacities);
      fn.setArg(8, meta_i32);
      fn.setArg(9, meta_f32);
      fn.setArg(10, g_means2d);
      fn.setArg(11, g_conics);
      fn.setArg(12, g_colors);
      fn.setArg(13, g_opacities);
      fn.dispatch((uint64_t)To * 256ull, (uint64_t)256ull);
    });
  }

  return std::make_tuple(g_means2d, g_conics, g_colors, g_opacities);
}

}  // namespace gsplat

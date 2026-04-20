#import <Foundation/Foundation.h>

#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/native/mps/MetalShaderLibrary.h>
#include <torch/mps.h>

#include <mutex>
#include <string>
#include <tuple>
#include <vector>

#include "shared/common.h"

namespace gsplat {
namespace {
using at::native::mps::DynamicMetalShaderLibrary;
using at::native::mps::MetalKernelFunction;

std::string load_shader_source() {
  NSString* metalPath = [[NSString stringWithUTF8String:__FILE__] stringByDeletingLastPathComponent];
  metalPath = [metalPath stringByAppendingPathComponent:@"gsplat_fast_kernels.metal"];
  NSError* err = nil;
  NSString* src = [NSString stringWithContentsOfFile:metalPath encoding:NSUTF8StringEncoding error:&err];
  TORCH_CHECK(src != nil, "Failed to read gsplat_fast_kernels.metal: ", err.localizedDescription.UTF8String);
  return std::string([src UTF8String]);
}

struct MetalFastKernels {
  std::shared_ptr<MetalKernelFunction> count_tiles;
  std::shared_ptr<MetalKernelFunction> emit_binned_ids;
  std::shared_ptr<MetalKernelFunction> tile_sort_render_forward;
  std::shared_ptr<MetalKernelFunction> tile_sort_render_backward;
};

MetalFastKernels& kernels() {
  static std::once_flag once;
  static std::unique_ptr<DynamicMetalShaderLibrary> lib;
  static MetalFastKernels out;
  std::call_once(once, []() {
    lib = std::make_unique<DynamicMetalShaderLibrary>(load_shader_source());
    out.count_tiles = lib->getKernelFunction("count_tiles");
    out.emit_binned_ids = lib->getKernelFunction("emit_binned_ids");
    out.tile_sort_render_forward = lib->getKernelFunction("tile_sort_render_forward");
    out.tile_sort_render_backward = lib->getKernelFunction("tile_sort_render_backward");
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
  TORCH_CHECK(means2d.size(0) == conics.size(0) && means2d.size(0) == colors.size(0) && means2d.size(0) == opacities.size(0),
              "All inputs must agree on G");
}

template <typename Fn>
void launch(std::shared_ptr<MetalKernelFunction> fn, Fn&& body) {
  fn->runCommandBlock([&]() {
    fn->startEncoding();
    body(*fn);
  });
}

}  // namespace

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> metal_forward(
    const torch::Tensor& means2d,
    const torch::Tensor& conics,
    const torch::Tensor& colors,
    const torch::Tensor& opacities,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32) {
  check_inputs(means2d, conics, colors, opacities);
  auto meta = parse_meta(meta_i32, meta_f32);
  TORCH_CHECK(meta.tile_size == 16, "fast Metal path currently requires tile_size=16");
  TORCH_CHECK(meta.max_tile_pairs <= 4096, "meta.max_tile_pairs exceeds shader compile-time limit 4096");

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

  int64_t max_count = tile_counts.max().item<int64_t>();
  TORCH_CHECK(max_count <= meta.max_tile_pairs,
              "Fast local-sort tile overflow: max tile count ", max_count,
              " exceeds max_tile_pairs=", meta.max_tile_pairs,
              ". Increase max_tile_pairs, reduce splats, or add the slower overflow fallback path.");

  auto out = torch::empty({meta.height, meta.width, 3}, opts_f);
  if (T > 0) {
    launch(k.tile_sort_render_forward, [&](MetalKernelFunction& fn) {
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
      fn.dispatch((uint64_t)T * 1024ull, (uint64_t)1024ull);
    });
  }

  return std::make_tuple(out, tile_counts, tile_offsets, binned_ids);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> metal_backward(
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

  const int64_t T = meta.tile_count;
  if (T > 0) {
    launch(k.tile_sort_render_backward, [&](MetalKernelFunction& fn) {
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
      fn.dispatch((uint64_t)T * 1024ull, (uint64_t)1024ull);
    });
  }

  return std::make_tuple(g_means2d, g_conics, g_colors, g_opacities);
}

}  // namespace gsplat

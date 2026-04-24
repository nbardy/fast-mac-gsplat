#import <Foundation/Foundation.h>

#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/native/mps/MetalShaderLibrary.h>
#include <torch/mps.h>

#include <cstdlib>
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
    return c;
  }();
  return cfg;
}

std::string load_shader_source() {
  auto& cfg = shader_config();
  NSString* metalPath = [[NSString stringWithUTF8String:__FILE__] stringByDeletingLastPathComponent];
  metalPath = [metalPath stringByAppendingPathComponent:@"gsplat_v9_project3d_train_kernels.metal"];
  NSError* err = nil;
  NSString* src = [NSString stringWithContentsOfFile:metalPath encoding:NSUTF8StringEncoding error:&err];
  TORCH_CHECK(src != nil, "Failed to read gsplat_v9_project3d_train_kernels.metal: ", err.localizedDescription.UTF8String);

  std::string preamble;
  preamble += "#define GSP_TILE_SIZE " + std::to_string(cfg.tile_size) + "u\n";
  preamble += "#define GSP_THREADS " + std::to_string(cfg.threads) + "u\n";
  preamble += "#define GSP_FAST_CAP " + std::to_string(cfg.fast_cap) + "u\n";
  preamble += "#define GSP_CHUNK " + std::to_string(cfg.chunk) + "u\n";
  preamble += "#define GSP_SIMD_WIDTH 32u\n";
  preamble += "#define GSP_SIMDGROUPS " + std::to_string(cfg.simdgroups) + "u\n";
  preamble += "\n";
  return preamble + std::string([src UTF8String]);
}

struct MetalV9Project3DTrainKernels {
  std::shared_ptr<MetalKernelFunction> project_pinhole_forward;
  std::shared_ptr<MetalKernelFunction> project_pinhole_backward;
  std::shared_ptr<MetalKernelFunction> count_tiles;
  std::shared_ptr<MetalKernelFunction> emit_binned_ids;
  std::shared_ptr<MetalKernelFunction> tile_fast_forward_eval;
  std::shared_ptr<MetalKernelFunction> tile_fast_forward_state;
  std::shared_ptr<MetalKernelFunction> tile_fast_backward_saved;
  std::shared_ptr<MetalKernelFunction> tile_overflow_forward;
  std::shared_ptr<MetalKernelFunction> tile_overflow_backward;
};

MetalV9Project3DTrainKernels& kernels() {
  static std::once_flag once;
  static std::unique_ptr<DynamicMetalShaderLibrary> lib;
  static MetalV9Project3DTrainKernels out;
  std::call_once(once, []() {
    lib = std::make_unique<DynamicMetalShaderLibrary>(load_shader_source());
    out.project_pinhole_forward = lib->getKernelFunction("project_pinhole_forward");
    out.project_pinhole_backward = lib->getKernelFunction("project_pinhole_backward");
    out.count_tiles = lib->getKernelFunction("count_tiles");
    out.emit_binned_ids = lib->getKernelFunction("emit_binned_ids");
    out.tile_fast_forward_eval = lib->getKernelFunction("tile_fast_forward_eval");
    out.tile_fast_forward_state = lib->getKernelFunction("tile_fast_forward_state");
    out.tile_fast_backward_saved = lib->getKernelFunction("tile_fast_backward_saved");
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
  TORCH_CHECK(means2d.dim() == 2 && means2d.size(1) == 2, "means2d must be [BG,2]");
  TORCH_CHECK(conics.dim() == 2 && conics.size(1) == 3, "conics must be [BG,3]");
  TORCH_CHECK(colors.dim() == 2 && colors.size(1) == 3, "colors must be [BG,3]");
  TORCH_CHECK(opacities.dim() == 1, "opacities must be [BG]");
  TORCH_CHECK(means2d.size(0) == conics.size(0) && means2d.size(0) == colors.size(0) && means2d.size(0) == opacities.size(0),
              "All inputs must agree on flattened BG");
}

void check_aux_i32(const torch::Tensor& t, const char* name) {
  TORCH_CHECK(t.device().is_mps(), name, " must be on MPS");
  TORCH_CHECK(t.scalar_type() == torch::kInt32, name, " must be int32");
}

void check_project_inputs(
    const torch::Tensor& means3d,
    const torch::Tensor& scales,
    const torch::Tensor& quats,
    const torch::Tensor& opacities,
    const torch::Tensor& camera_to_world,
    const torch::Tensor& camera_params,
    const torch::Tensor& project_f32) {
  TORCH_CHECK(means3d.device().is_mps(), "means3d must be on MPS");
  TORCH_CHECK(scales.device().is_mps(), "scales must be on MPS");
  TORCH_CHECK(quats.device().is_mps(), "quats must be on MPS");
  TORCH_CHECK(opacities.device().is_mps(), "opacities must be on MPS");
  TORCH_CHECK(camera_to_world.device().is_mps(), "camera_to_world must be on MPS");
  TORCH_CHECK(camera_params.device().is_mps(), "camera_params must be on MPS");
  TORCH_CHECK(project_f32.device().is_mps(), "project_f32 must be on MPS");
  TORCH_CHECK(means3d.scalar_type() == torch::kFloat32, "means3d must be float32");
  TORCH_CHECK(scales.scalar_type() == torch::kFloat32, "scales must be float32");
  TORCH_CHECK(quats.scalar_type() == torch::kFloat32, "quats must be float32");
  TORCH_CHECK(opacities.scalar_type() == torch::kFloat32, "opacities must be float32");
  TORCH_CHECK(camera_to_world.scalar_type() == torch::kFloat32, "camera_to_world must be float32");
  TORCH_CHECK(camera_params.scalar_type() == torch::kFloat32, "camera_params must be float32");
  TORCH_CHECK(project_f32.scalar_type() == torch::kFloat32, "project_f32 must be float32");
  TORCH_CHECK(means3d.dim() == 2 && means3d.size(1) == 3, "means3d must be [BG,3]");
  TORCH_CHECK(scales.dim() == 2 && scales.size(1) == 3, "scales must be [BG,3]");
  TORCH_CHECK(quats.dim() == 2 && quats.size(1) == 4, "quats must be [BG,4]");
  TORCH_CHECK(opacities.dim() == 1, "opacities must be [BG]");
  TORCH_CHECK(camera_to_world.dim() == 3 && camera_to_world.size(1) == 4 && camera_to_world.size(2) == 4,
              "camera_to_world must be [B,4,4]");
  TORCH_CHECK(camera_params.dim() == 2 && camera_params.size(1) == 4, "camera_params must be [B,4]");
  TORCH_CHECK(project_f32.dim() == 1 && project_f32.size(0) >= 1, "project_f32 must be [>=1]");
  TORCH_CHECK(means3d.size(0) == scales.size(0) && means3d.size(0) == quats.size(0) && means3d.size(0) == opacities.size(0),
              "3D Gaussian inputs must agree on flattened BG");
}

void check_project_grad_inputs(
    const torch::Tensor& grad_means2d,
    const torch::Tensor& grad_conics,
    const torch::Tensor& grad_projected_opacities,
    const torch::Tensor& grad_depths,
    const torch::Tensor& means3d) {
  TORCH_CHECK(grad_means2d.device().is_mps(), "grad_means2d must be on MPS");
  TORCH_CHECK(grad_conics.device().is_mps(), "grad_conics must be on MPS");
  TORCH_CHECK(grad_projected_opacities.device().is_mps(), "grad_projected_opacities must be on MPS");
  TORCH_CHECK(grad_depths.device().is_mps(), "grad_depths must be on MPS");
  TORCH_CHECK(grad_means2d.scalar_type() == torch::kFloat32, "grad_means2d must be float32");
  TORCH_CHECK(grad_conics.scalar_type() == torch::kFloat32, "grad_conics must be float32");
  TORCH_CHECK(grad_projected_opacities.scalar_type() == torch::kFloat32, "grad_projected_opacities must be float32");
  TORCH_CHECK(grad_depths.scalar_type() == torch::kFloat32, "grad_depths must be float32");
  TORCH_CHECK(grad_means2d.sizes() == torch::IntArrayRef({means3d.size(0), 2}), "grad_means2d must be [BG,2]");
  TORCH_CHECK(grad_conics.sizes() == torch::IntArrayRef({means3d.size(0), 3}), "grad_conics must be [BG,3]");
  TORCH_CHECK(grad_projected_opacities.sizes() == torch::IntArrayRef({means3d.size(0)}), "grad_projected_opacities must be [BG]");
  TORCH_CHECK(grad_depths.sizes() == torch::IntArrayRef({means3d.size(0)}), "grad_depths must be [BG]");
}

template <typename Fn>
void launch(std::shared_ptr<MetalKernelFunction> fn, Fn&& body) {
  fn->runCommandBlock([&]() {
    fn->startEncoding();
    body(*fn);
  });
}

}  // namespace

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> metal_project_pinhole_forward(
    const torch::Tensor& means3d,
    const torch::Tensor& scales,
    const torch::Tensor& quats,
    const torch::Tensor& opacities,
    const torch::Tensor& camera_to_world,
    const torch::Tensor& camera_params,
    const torch::Tensor& meta_i32,
    const torch::Tensor& project_f32) {
  check_project_inputs(means3d, scales, quats, opacities, camera_to_world, camera_params, project_f32);
  check_aux_i32(meta_i32, "meta_i32");
  auto meta_cpu = meta_i32.cpu();
  auto* ip = meta_cpu.data_ptr<int32_t>();
  int gaussians = ip[5];
  int batch_size = ip[8];
  int gaussians_per_batch = ip[9];
  TORCH_CHECK(batch_size == camera_to_world.size(0), "camera_to_world batch size mismatch");
  TORCH_CHECK(batch_size == camera_params.size(0), "camera_params batch size mismatch");
  TORCH_CHECK(gaussians == means3d.size(0), "flattened Gaussian count mismatch");
  TORCH_CHECK(gaussians_per_batch > 0, "meta.gaussians_per_batch must be positive");
  TORCH_CHECK(gaussians == batch_size * gaussians_per_batch, "meta Gaussian shape mismatch");

  auto& k = kernels();
  auto opts_f = means3d.options().dtype(torch::kFloat32);
  constexpr uint64_t kProjectThreads = 256ull;
  const int64_t G = means3d.size(0);

  auto means2d = torch::empty({G, 2}, opts_f);
  auto conics = torch::empty({G, 3}, opts_f);
  auto projected_opacities = torch::empty({G}, opts_f);
  auto depths = torch::empty({G}, opts_f);

  if (G > 0) {
    launch(k.project_pinhole_forward, [&](MetalKernelFunction& fn) {
      fn.setArg(0, means3d);
      fn.setArg(1, scales);
      fn.setArg(2, quats);
      fn.setArg(3, opacities);
      fn.setArg(4, camera_to_world);
      fn.setArg(5, camera_params);
      fn.setArg(6, meta_i32);
      fn.setArg(7, project_f32);
      fn.setArg(8, means2d);
      fn.setArg(9, conics);
      fn.setArg(10, projected_opacities);
      fn.setArg(11, depths);
      fn.dispatch((uint64_t)G, kProjectThreads);
    });
  }

  return std::make_tuple(means2d, conics, projected_opacities, depths);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
metal_project_pinhole_backward(
    const torch::Tensor& grad_means2d,
    const torch::Tensor& grad_conics,
    const torch::Tensor& grad_projected_opacities,
    const torch::Tensor& grad_depths,
    const torch::Tensor& means3d,
    const torch::Tensor& scales,
    const torch::Tensor& quats,
    const torch::Tensor& opacities,
    const torch::Tensor& camera_to_world,
    const torch::Tensor& camera_params,
    const torch::Tensor& meta_i32,
    const torch::Tensor& project_f32) {
  check_project_inputs(means3d, scales, quats, opacities, camera_to_world, camera_params, project_f32);
  check_project_grad_inputs(grad_means2d, grad_conics, grad_projected_opacities, grad_depths, means3d);
  check_aux_i32(meta_i32, "meta_i32");
  auto meta_cpu = meta_i32.cpu();
  auto* ip = meta_cpu.data_ptr<int32_t>();
  int gaussians = ip[5];
  int batch_size = ip[8];
  int gaussians_per_batch = ip[9];
  TORCH_CHECK(batch_size == camera_to_world.size(0), "camera_to_world batch size mismatch");
  TORCH_CHECK(batch_size == camera_params.size(0), "camera_params batch size mismatch");
  TORCH_CHECK(gaussians == means3d.size(0), "flattened Gaussian count mismatch");
  TORCH_CHECK(gaussians_per_batch > 0, "meta.gaussians_per_batch must be positive");
  TORCH_CHECK(gaussians == batch_size * gaussians_per_batch, "meta Gaussian shape mismatch");

  auto& k = kernels();
  constexpr uint64_t kProjectThreads = 256ull;
  const int64_t G = means3d.size(0);

  auto grad_means3d = torch::empty_like(means3d);
  auto grad_scales = torch::empty_like(scales);
  auto grad_quats = torch::empty_like(quats);
  auto grad_opacities = torch::empty_like(opacities);
  auto grad_camera_to_world = torch::zeros_like(camera_to_world);
  auto grad_camera_params = torch::zeros_like(camera_params);

  if (G > 0) {
    launch(k.project_pinhole_backward, [&](MetalKernelFunction& fn) {
      fn.setArg(0, grad_means2d);
      fn.setArg(1, grad_conics);
      fn.setArg(2, grad_projected_opacities);
      fn.setArg(3, grad_depths);
      fn.setArg(4, means3d);
      fn.setArg(5, scales);
      fn.setArg(6, quats);
      fn.setArg(7, opacities);
      fn.setArg(8, camera_to_world);
      fn.setArg(9, camera_params);
      fn.setArg(10, meta_i32);
      fn.setArg(11, project_f32);
      fn.setArg(12, grad_means3d);
      fn.setArg(13, grad_scales);
      fn.setArg(14, grad_quats);
      fn.setArg(15, grad_opacities);
      fn.setArg(16, grad_camera_to_world);
      fn.setArg(17, grad_camera_params);
      fn.dispatch((uint64_t)G, kProjectThreads);
    });
  }

  return std::make_tuple(
      grad_means3d,
      grad_scales,
      grad_quats,
      grad_opacities,
      grad_camera_to_world,
      grad_camera_params);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> metal_bin(
    const torch::Tensor& means2d,
    const torch::Tensor& conics,
    const torch::Tensor& colors,
    const torch::Tensor& opacities,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32) {
  check_inputs(means2d, conics, colors, opacities);
  auto meta = parse_meta(meta_i32, meta_f32);
  auto& sc = shader_config();
  TORCH_CHECK(meta.tile_size == sc.tile_size, "meta.tile_size must match shader tile size ", sc.tile_size);
  TORCH_CHECK(meta.max_fast_pairs <= sc.fast_cap, "meta.max_fast_pairs exceeds shader compile-time cap ", sc.fast_cap);
  TORCH_CHECK(meta.batch_size > 0, "meta.batch_size must be positive");
  TORCH_CHECK(meta.gaussians_per_batch > 0, "meta.gaussians_per_batch must be positive");
  TORCH_CHECK(meta.tiles_per_image == meta.tiles_y * meta.tiles_x, "meta.tiles_per_image mismatch");
  TORCH_CHECK(meta.gaussians == means2d.size(0), "flattened Gaussian count mismatch");

  auto& k = kernels();
  auto opts_f = means2d.options().dtype(torch::kFloat32);
  auto opts_i32 = means2d.options().dtype(torch::kInt32);
  constexpr uint64_t kBinThreads = 256ull;

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
    fn.dispatch((uint64_t)G, kBinThreads);
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
      fn.dispatch((uint64_t)G, kBinThreads);
    });
  }

  return std::make_tuple(tile_counts, tile_offsets, binned_ids);
}

torch::Tensor metal_render_fast_forward_eval(
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
  auto& sc = shader_config();
  auto& k = kernels();
  auto opts_f = means2d.options().dtype(torch::kFloat32);

  auto out = torch::empty({meta.batch_size, meta.height, meta.width, 3}, opts_f);
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
      fn.dispatch((uint64_t)meta.tile_count * (uint64_t)sc.threads, (uint64_t)sc.threads);
    });
  }
  return out;
}

std::tuple<torch::Tensor, torch::Tensor> metal_render_fast_forward_state(
    const torch::Tensor& means2d,
    const torch::Tensor& conics,
    const torch::Tensor& colors,
    const torch::Tensor& opacities,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32,
    torch::Tensor& binned_ids,
    const torch::Tensor& tile_counts,
    const torch::Tensor& tile_offsets) {
  check_inputs(means2d, conics, colors, opacities);
  check_aux_i32(tile_counts, "tile_counts");
  check_aux_i32(tile_offsets, "tile_offsets");
  check_aux_i32(binned_ids, "binned_ids");
  auto meta = parse_meta(meta_i32, meta_f32);
  auto& sc = shader_config();
  auto& k = kernels();
  auto opts_f = means2d.options().dtype(torch::kFloat32);
  auto opts_i32 = means2d.options().dtype(torch::kInt32);

  auto out = torch::empty({meta.batch_size, meta.height, meta.width, 3}, opts_f);
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
      fn.setArg(10, tile_stop_counts);
      fn.dispatch((uint64_t)meta.tile_count * (uint64_t)sc.threads, (uint64_t)sc.threads);
    });
  }
  return std::make_tuple(out, tile_stop_counts);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> metal_render_fast_backward_saved(
    const torch::Tensor& grad_out,
    const torch::Tensor& means2d,
    const torch::Tensor& conics,
    const torch::Tensor& colors,
    const torch::Tensor& opacities,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32,
    const torch::Tensor& tile_counts,
    const torch::Tensor& tile_offsets,
    const torch::Tensor& binned_ids,
    const torch::Tensor& tile_stop_counts) {
  check_inputs(means2d, conics, colors, opacities);
  TORCH_CHECK(grad_out.device().is_mps(), "grad_out must be on MPS");
  check_aux_i32(tile_counts, "tile_counts");
  check_aux_i32(tile_offsets, "tile_offsets");
  check_aux_i32(binned_ids, "binned_ids");
  check_aux_i32(tile_stop_counts, "tile_stop_counts");

  auto meta = parse_meta(meta_i32, meta_f32);
  auto& sc = shader_config();
  auto& k = kernels();
  auto opts_f = means2d.options().dtype(torch::kFloat32);

  auto g_means2d = torch::zeros_like(means2d, opts_f);
  auto g_conics = torch::zeros_like(conics, opts_f);
  auto g_colors = torch::zeros_like(colors, opts_f);
  auto g_opacities = torch::zeros_like(opacities, opts_f);

  if (meta.tile_count > 0) {
    launch(k.tile_fast_backward_saved, [&](MetalKernelFunction& fn) {
      fn.setArg(0, grad_out.contiguous());
      fn.setArg(1, tile_counts);
      fn.setArg(2, tile_offsets);
      fn.setArg(3, binned_ids);
      fn.setArg(4, tile_stop_counts);
      fn.setArg(5, means2d);
      fn.setArg(6, conics);
      fn.setArg(7, colors);
      fn.setArg(8, opacities);
      fn.setArg(9, meta_i32);
      fn.setArg(10, meta_f32);
      fn.setArg(11, g_means2d);
      fn.setArg(12, g_conics);
      fn.setArg(13, g_colors);
      fn.setArg(14, g_opacities);
      fn.dispatch((uint64_t)meta.tile_count * (uint64_t)sc.threads, (uint64_t)sc.threads);
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
  auto& sc = shader_config();
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
      fn.dispatch((uint64_t)To * (uint64_t)sc.threads, (uint64_t)sc.threads);
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
  auto& sc = shader_config();
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
      fn.dispatch((uint64_t)To * (uint64_t)sc.threads, (uint64_t)sc.threads);
    });
  }
  return std::make_tuple(g_means2d, g_conics, g_colors, g_opacities);
}

}  // namespace gsplat

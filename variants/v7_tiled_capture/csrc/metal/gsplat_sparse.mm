#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <QuartzCore/QuartzCore.h>

#include <torch/extension.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <mutex>
#include <numeric>
#include <string>
#include <tuple>
#include <vector>

#include "shared/common.h"

namespace gsplat {
namespace {

static constexpr uint32_t kMaxFrontK = 8u;

struct CPUUniforms {
  uint32_t width;
  uint32_t height;
  uint32_t gaussians;
  uint32_t front_k;
  uint32_t tile_size;
  uint32_t tiles_x;
  uint32_t tiles_y;
  float alpha_threshold;
  float bg_r;
  float bg_g;
  float bg_b;
  float eps;
};

struct MetalContext {
  id<MTLDevice> device = nil;
  id<MTLCommandQueue> queue = nil;
  id<MTLLibrary> library = nil;
  id<MTLRenderPipelineState> render_pso = nil;
  id<MTLComputePipelineState> capture_pso = nil;
  id<MTLComputePipelineState> backward_frontk_pso = nil;
  id<MTLComputePipelineState> backward_overflow_pso = nil;
};

static std::string read_shader_source() {
  NSString* metalPath = [[NSString stringWithUTF8String:__FILE__] stringByDeletingLastPathComponent];
  metalPath = [metalPath stringByAppendingPathComponent:@"gsplat_v72_sparse.metal"];
  NSError* err = nil;
  NSString* src = [NSString stringWithContentsOfFile:metalPath encoding:NSUTF8StringEncoding error:&err];
  TORCH_CHECK(src != nil, "Failed to read gsplat_v72_sparse.metal: ", err.localizedDescription.UTF8String);
  return std::string([src UTF8String]);
}

MetalContext& ctx() {
  static MetalContext c;
  static std::once_flag once;
  std::call_once(once, []() {
    c.device = MTLCreateSystemDefaultDevice();
    TORCH_CHECK(c.device != nil, "No Metal device found");
    c.queue = [c.device newCommandQueue];
    TORCH_CHECK(c.queue != nil, "Failed to create command queue");
    NSError* err = nil;
    NSString* src = [NSString stringWithUTF8String:read_shader_source().c_str()];
    c.library = [c.device newLibraryWithSource:src options:nil error:&err];
    TORCH_CHECK(c.library != nil, "Failed to compile Metal library: ", err.localizedDescription.UTF8String);

    id<MTLFunction> vs = [c.library newFunctionWithName:@"ellipse_vs"];
    id<MTLFunction> fs = [c.library newFunctionWithName:@"gaussian_fs"];
    MTLRenderPipelineDescriptor* desc = [[MTLRenderPipelineDescriptor alloc] init];
    desc.vertexFunction = vs;
    desc.fragmentFunction = fs;
    desc.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA32Float;
    desc.colorAttachments[0].blendingEnabled = YES;
    desc.colorAttachments[0].rgbBlendOperation = MTLBlendOperationAdd;
    desc.colorAttachments[0].alphaBlendOperation = MTLBlendOperationAdd;
    desc.colorAttachments[0].sourceRGBBlendFactor = MTLBlendFactorOne;
    desc.colorAttachments[0].sourceAlphaBlendFactor = MTLBlendFactorOne;
    desc.colorAttachments[0].destinationRGBBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
    desc.colorAttachments[0].destinationAlphaBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
    c.render_pso = [c.device newRenderPipelineStateWithDescriptor:desc error:&err];
    TORCH_CHECK(c.render_pso != nil, "Failed to create render pipeline: ", err.localizedDescription.UTF8String);

    id<MTLFunction> capture = [c.library newFunctionWithName:@"capture_front_k_binned"];
    c.capture_pso = [c.device newComputePipelineStateWithFunction:capture error:&err];
    TORCH_CHECK(c.capture_pso != nil, "Failed to create tiled capture pipeline: ", err.localizedDescription.UTF8String);

    id<MTLFunction> bw_frontk = [c.library newFunctionWithName:@"backward_front_k"];
    c.backward_frontk_pso = [c.device newComputePipelineStateWithFunction:bw_frontk error:&err];
    TORCH_CHECK(c.backward_frontk_pso != nil, "Failed to create front-k backward pipeline: ", err.localizedDescription.UTF8String);

    id<MTLFunction> bw_overflow = [c.library newFunctionWithName:@"backward_overflow_replay_binned"];
    c.backward_overflow_pso = [c.device newComputePipelineStateWithFunction:bw_overflow error:&err];
    TORCH_CHECK(c.backward_overflow_pso != nil, "Failed to create overflow replay pipeline: ", err.localizedDescription.UTF8String);
  });
  return c;
}

void check_inputs(const torch::Tensor& means2d,
                  const torch::Tensor& conics,
                  const torch::Tensor& colors,
                  const torch::Tensor& opacities,
                  const torch::Tensor& depths,
                  const MetaView& meta) {
  TORCH_CHECK(means2d.scalar_type() == torch::kFloat32, "means2d must be float32");
  TORCH_CHECK(conics.scalar_type() == torch::kFloat32, "conics must be float32");
  TORCH_CHECK(colors.scalar_type() == torch::kFloat32, "colors must be float32");
  TORCH_CHECK(opacities.scalar_type() == torch::kFloat32, "opacities must be float32");
  TORCH_CHECK(depths.scalar_type() == torch::kFloat32, "depths must be float32");
  TORCH_CHECK(meta.front_k > 0 && meta.front_k <= static_cast<int32_t>(kMaxFrontK),
              "front_k must be in [1, ", kMaxFrontK, "]");
  TORCH_CHECK(meta.tile_size > 0, "tile_size must be positive");
  TORCH_CHECK(meta.gaussians_per_batch <= 65536,
              "gaussians_per_batch must be <= 65536 because ids are stored as uint16");
}

torch::Tensor pack_gaussians_cpu(const torch::Tensor& means2d,
                                 const torch::Tensor& conics,
                                 const torch::Tensor& colors,
                                 const torch::Tensor& opacities) {
  auto means_cpu = means2d.contiguous().to(torch::kCPU);
  auto conics_cpu = conics.contiguous().to(torch::kCPU);
  auto colors_cpu = colors.contiguous().to(torch::kCPU);
  auto opacities_cpu = opacities.contiguous().to(torch::kCPU);
  int64_t B = means_cpu.size(0);
  int64_t G = means_cpu.size(1);
  auto packed = torch::empty({B, G, 12}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
  auto* out = packed.data_ptr<float>();
  auto* m = means_cpu.data_ptr<float>();
  auto* q = conics_cpu.data_ptr<float>();
  auto* c = colors_cpu.data_ptr<float>();
  auto* o = opacities_cpu.data_ptr<float>();
  for (int64_t i = 0; i < B * G; ++i) {
    out[i * 12 + 0] = m[i * 2 + 0];
    out[i * 12 + 1] = m[i * 2 + 1];
    out[i * 12 + 2] = o[i];
    out[i * 12 + 3] = 0.0f;
    out[i * 12 + 4] = q[i * 3 + 0];
    out[i * 12 + 5] = q[i * 3 + 1];
    out[i * 12 + 6] = q[i * 3 + 2];
    out[i * 12 + 7] = 0.0f;
    out[i * 12 + 8] = c[i * 3 + 0];
    out[i * 12 + 9] = c[i * 3 + 1];
    out[i * 12 + 10] = c[i * 3 + 2];
    out[i * 12 + 11] = 0.0f;
  }
  return packed;
}

std::tuple<id<MTLBuffer>, NSUInteger> make_buffer_from_tensor(id<MTLDevice> device, const torch::Tensor& t) {
  auto cpu = t.contiguous().to(torch::kCPU);
  NSUInteger bytes = cpu.nbytes();
  id<MTLBuffer> buf = [device newBufferWithBytes:cpu.data_ptr() length:bytes options:MTLResourceStorageModeShared];
  TORCH_CHECK(buf != nil, "Failed to create Metal buffer");
  return {buf, bytes};
}

id<MTLBuffer> make_zero_buffer(id<MTLDevice> device, NSUInteger bytes) {
  id<MTLBuffer> buf = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
  TORCH_CHECK(buf != nil, "Failed to allocate Metal buffer");
  std::memset([buf contents], 0, bytes);
  return buf;
}

torch::Tensor texture_to_torch_cpu(id<MTLTexture> tex, int B, int H, int W) {
  auto out_cpu = torch::empty({B, H, W, 3}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
  auto* dst = out_cpu.data_ptr<float>();
  MTLRegion region = MTLRegionMake2D(0, 0, W, H);
  size_t bytesPerRow = sizeof(float) * 4 * W;
  std::vector<float> tmp(static_cast<size_t>(H) * static_cast<size_t>(W) * 4u);
  for (int b = 0; b < B; ++b) {
    [tex getBytes:tmp.data()
       bytesPerRow:bytesPerRow
     bytesPerImage:bytesPerRow * H
        fromRegion:region
       mipmapLevel:0
             slice:b];
    float* outb = dst + static_cast<size_t>(b) * H * W * 3u;
    for (int i = 0; i < H * W; ++i) {
      outb[i * 3 + 0] = tmp[i * 4 + 0];
      outb[i * 3 + 1] = tmp[i * 4 + 1];
      outb[i * 3 + 2] = tmp[i * 4 + 2];
    }
  }
  return out_cpu;
}

torch::Tensor tensor_from_buffer_u16_cpu(id<MTLBuffer> buf, std::vector<int64_t> shape) {
  return torch::from_blob([buf contents], shape, torch::TensorOptions().dtype(torch::kUInt16).device(torch::kCPU)).clone();
}

torch::Tensor tensor_from_buffer_u8_cpu(id<MTLBuffer> buf, std::vector<int64_t> shape) {
  return torch::from_blob([buf contents], shape, torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU)).clone();
}

torch::Tensor tensor_from_buffer_i32_cpu(id<MTLBuffer> buf, std::vector<int64_t> shape) {
  return torch::from_blob([buf contents], shape, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU)).clone();
}

torch::Tensor tensor_from_buffer_f32_cpu(id<MTLBuffer> buf, std::vector<int64_t> shape) {
  return torch::from_blob([buf contents], shape, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU)).clone();
}

inline float safe_det(float a, float b, float c, float eps) {
  return std::max(a * c - b * b, eps);
}

bool compute_support_bbox(const float* rec,
                          const MetaView& meta,
                          float& x0,
                          float& x1,
                          float& y0,
                          float& y1) {
  float mx = rec[0];
  float my = rec[1];
  float opacity = rec[2];
  if (opacity <= meta.alpha_threshold) return false;
  float a = rec[4];
  float b = rec[5];
  float c = rec[6];
  float ratio = std::max(meta.alpha_threshold / std::max(opacity, meta.eps), meta.eps);
  float tau = -2.0f * std::log(ratio);
  if (!(std::isfinite(tau) && tau > 0.0f)) return false;
  float det = safe_det(a, b, c, meta.eps);
  float hx = std::sqrt(std::max(tau * c / det, 0.0f));
  float hy = std::sqrt(std::max(tau * a / det, 0.0f));
  x0 = std::clamp(mx - hx, 0.0f, static_cast<float>(meta.width));
  x1 = std::clamp(mx + hx, 0.0f, static_cast<float>(meta.width));
  y0 = std::clamp(my - hy, 0.0f, static_cast<float>(meta.height));
  y1 = std::clamp(my + hy, 0.0f, static_cast<float>(meta.height));
  return x0 <= x1 && y0 <= y1;
}

std::tuple<torch::Tensor, torch::Tensor> build_tile_bins_cpu(const torch::Tensor& packed_gaussians, const MetaView& meta) {
  auto packed = packed_gaussians.contiguous().to(torch::kCPU);
  const float* recs = packed.data_ptr<float>();
  const int32_t B = meta.batch_size;
  const int32_t G = meta.gaussians_per_batch;
  const int32_t tiles_x = (meta.width + meta.tile_size - 1) / meta.tile_size;
  const int32_t tiles_y = (meta.height + meta.tile_size - 1) / meta.tile_size;
  const int32_t tiles_per_batch = tiles_x * tiles_y;

  std::vector<int32_t> counts(static_cast<size_t>(B) * tiles_per_batch, 0);

  for (int32_t b = 0; b < B; ++b) {
    for (int32_t i = 0; i < G; ++i) {
      const float* rec = recs + (static_cast<int64_t>(b) * G + i) * 12;
      float x0 = 0.0f, x1 = 0.0f, y0 = 0.0f, y1 = 0.0f;
      if (!compute_support_bbox(rec, meta, x0, x1, y0, y1)) continue;
      int32_t tx0 = static_cast<int32_t>(std::floor(x0 / meta.tile_size));
      int32_t tx1 = static_cast<int32_t>(std::floor(x1 / meta.tile_size));
      int32_t ty0 = static_cast<int32_t>(std::floor(y0 / meta.tile_size));
      int32_t ty1 = static_cast<int32_t>(std::floor(y1 / meta.tile_size));
      tx0 = std::clamp(tx0, 0, tiles_x - 1);
      tx1 = std::clamp(tx1, 0, tiles_x - 1);
      ty0 = std::clamp(ty0, 0, tiles_y - 1);
      ty1 = std::clamp(ty1, 0, tiles_y - 1);
      for (int32_t ty = ty0; ty <= ty1; ++ty) {
        for (int32_t tx = tx0; tx <= tx1; ++tx) {
          counts[static_cast<size_t>(b) * tiles_per_batch + ty * tiles_x + tx] += 1;
        }
      }
    }
  }

  auto tile_offsets = torch::empty({B, tiles_per_batch + 1}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU));
  int32_t* off = tile_offsets.data_ptr<int32_t>();
  int64_t total_refs = 0;
  for (int32_t b = 0; b < B; ++b) {
    off[static_cast<int64_t>(b) * (tiles_per_batch + 1)] = static_cast<int32_t>(total_refs);
    for (int32_t t = 0; t < tiles_per_batch; ++t) {
      total_refs += counts[static_cast<size_t>(b) * tiles_per_batch + t];
      TORCH_CHECK(total_refs <= std::numeric_limits<int32_t>::max(),
                  "tile reference list exceeded int32 capacity; choose a larger tile_size or add multi-stage binning");
      off[static_cast<int64_t>(b) * (tiles_per_batch + 1) + t + 1] = static_cast<int32_t>(total_refs);
    }
  }

  auto tile_ids = torch::empty({total_refs}, torch::TensorOptions().dtype(torch::kUInt16).device(torch::kCPU));
  uint16_t* ids = tile_ids.data_ptr<uint16_t>();
  std::vector<int32_t> cursor(static_cast<size_t>(B) * tiles_per_batch);
  for (int32_t b = 0; b < B; ++b) {
    for (int32_t t = 0; t < tiles_per_batch; ++t) {
      cursor[static_cast<size_t>(b) * tiles_per_batch + t] = off[static_cast<int64_t>(b) * (tiles_per_batch + 1) + t];
    }
  }

  for (int32_t b = 0; b < B; ++b) {
    for (int32_t i = 0; i < G; ++i) {
      const float* rec = recs + (static_cast<int64_t>(b) * G + i) * 12;
      float x0 = 0.0f, x1 = 0.0f, y0 = 0.0f, y1 = 0.0f;
      if (!compute_support_bbox(rec, meta, x0, x1, y0, y1)) continue;
      int32_t tx0 = static_cast<int32_t>(std::floor(x0 / meta.tile_size));
      int32_t tx1 = static_cast<int32_t>(std::floor(x1 / meta.tile_size));
      int32_t ty0 = static_cast<int32_t>(std::floor(y0 / meta.tile_size));
      int32_t ty1 = static_cast<int32_t>(std::floor(y1 / meta.tile_size));
      tx0 = std::clamp(tx0, 0, tiles_x - 1);
      tx1 = std::clamp(tx1, 0, tiles_x - 1);
      ty0 = std::clamp(ty0, 0, tiles_y - 1);
      ty1 = std::clamp(ty1, 0, tiles_y - 1);
      for (int32_t ty = ty0; ty <= ty1; ++ty) {
        for (int32_t tx = tx0; tx <= tx1; ++tx) {
          int32_t tile = ty * tiles_x + tx;
          int32_t& dst = cursor[static_cast<size_t>(b) * tiles_per_batch + tile];
          ids[dst++] = static_cast<uint16_t>(i);
        }
      }
    }
  }

  return {tile_offsets, tile_ids};
}

}  // namespace

ForwardOut metal_forward(
    const torch::Tensor& means2d,
    const torch::Tensor& conics,
    const torch::Tensor& colors,
    const torch::Tensor& opacities,
    const torch::Tensor& depths,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32) {
  auto meta = parse_meta(meta_i32, meta_f32);
  check_inputs(means2d, conics, colors, opacities, depths, meta);
  auto& c = ctx();

  auto packed_cpu = pack_gaussians_cpu(means2d, conics, colors, opacities);
  auto [tile_offsets_cpu, tile_ids_cpu] = build_tile_bins_cpu(packed_cpu, meta);

  auto [gauss_buf, gauss_bytes] = make_buffer_from_tensor(c.device, packed_cpu);
  auto [tile_offsets_buf, tile_offsets_bytes] = make_buffer_from_tensor(c.device, tile_offsets_cpu);
  auto [tile_ids_buf, tile_ids_bytes] = make_buffer_from_tensor(c.device, tile_ids_cpu);

  CPUUniforms u;
  u.width = static_cast<uint32_t>(meta.width);
  u.height = static_cast<uint32_t>(meta.height);
  u.gaussians = static_cast<uint32_t>(meta.gaussians_per_batch);
  u.front_k = static_cast<uint32_t>(meta.front_k);
  u.tile_size = static_cast<uint32_t>(meta.tile_size);
  u.tiles_x = static_cast<uint32_t>((meta.width + meta.tile_size - 1) / meta.tile_size);
  u.tiles_y = static_cast<uint32_t>((meta.height + meta.tile_size - 1) / meta.tile_size);
  u.alpha_threshold = meta.alpha_threshold;
  u.bg_r = meta.bg_r;
  u.bg_g = meta.bg_g;
  u.bg_b = meta.bg_b;
  u.eps = meta.eps;
  id<MTLBuffer> u_buf = [c.device newBufferWithBytes:&u length:sizeof(u) options:MTLResourceStorageModeShared];

  MTLTextureDescriptor* td =
      [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA32Float
                                                         width:meta.width
                                                        height:meta.height
                                                     mipmapped:NO];
  td.textureType = MTLTextureType2DArray;
  td.arrayLength = meta.batch_size;
  td.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
  td.storageMode = MTLStorageModeShared;
  id<MTLTexture> tex = [c.device newTextureWithDescriptor:td];
  TORCH_CHECK(tex != nil, "Failed to allocate output texture");

  NSUInteger pixels_per_batch = static_cast<NSUInteger>(meta.width) * static_cast<NSUInteger>(meta.height);
  NSUInteger ids_bytes_batch = sizeof(uint16_t) * pixels_per_batch * static_cast<NSUInteger>(meta.front_k);
  NSUInteger raw_bytes_batch = sizeof(float) * pixels_per_batch * static_cast<NSUInteger>(meta.front_k);
  NSUInteger meta_bytes_batch = sizeof(uint8_t) * pixels_per_batch;

  id<MTLBuffer> front_ids_buf = make_zero_buffer(c.device, ids_bytes_batch * meta.batch_size);
  id<MTLBuffer> front_raw_buf = make_zero_buffer(c.device, raw_bytes_batch * meta.batch_size);
  id<MTLBuffer> front_meta_buf = make_zero_buffer(c.device, meta_bytes_batch * meta.batch_size);

  id<MTLCommandBuffer> cb = [c.queue commandBuffer];
  TORCH_CHECK(cb != nil, "Failed to create Metal command buffer");

  NSUInteger gauss_batch_bytes = gauss_bytes / meta.batch_size;
  NSUInteger tile_row_bytes = sizeof(int32_t) * (static_cast<NSUInteger>(u.tiles_x) * static_cast<NSUInteger>(u.tiles_y) + 1u);

  for (int b = 0; b < meta.batch_size; ++b) {
    MTLRenderPassDescriptor* rp = [MTLRenderPassDescriptor renderPassDescriptor];
    rp.colorAttachments[0].texture = tex;
    rp.colorAttachments[0].slice = b;
    rp.colorAttachments[0].loadAction = MTLLoadActionClear;
    rp.colorAttachments[0].storeAction = MTLStoreActionStore;
    rp.colorAttachments[0].clearColor = MTLClearColorMake(meta.bg_r, meta.bg_g, meta.bg_b, 1.0);

    id<MTLRenderCommandEncoder> renc = [cb renderCommandEncoderWithDescriptor:rp];
    [renc setRenderPipelineState:c.render_pso];
    [renc setVertexBuffer:gauss_buf offset:gauss_batch_bytes * b atIndex:0];
    [renc setVertexBuffer:u_buf offset:0 atIndex:1];
    [renc setFragmentBuffer:u_buf offset:0 atIndex:0];
    [renc drawPrimitives:MTLPrimitiveTypeTriangleStrip
             vertexStart:0
             vertexCount:4
           instanceCount:meta.gaussians_per_batch];
    [renc endEncoding];

    id<MTLComputeCommandEncoder> cenc = [cb computeCommandEncoder];
    [cenc setComputePipelineState:c.capture_pso];
    [cenc setBuffer:gauss_buf offset:gauss_batch_bytes * b atIndex:0];
    [cenc setBuffer:tile_offsets_buf offset:tile_row_bytes * b atIndex:1];
    [cenc setBuffer:tile_ids_buf offset:0 atIndex:2];
    [cenc setBuffer:u_buf offset:0 atIndex:3];
    [cenc setBuffer:front_ids_buf offset:ids_bytes_batch * b atIndex:4];
    [cenc setBuffer:front_raw_buf offset:raw_bytes_batch * b atIndex:5];
    [cenc setBuffer:front_meta_buf offset:meta_bytes_batch * b atIndex:6];
    NSUInteger tg = 256;
    [cenc dispatchThreads:MTLSizeMake(pixels_per_batch, 1, 1)
   threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    [cenc endEncoding];
  }

  [cb commit];
  [cb waitUntilCompleted];

  auto out_cpu = texture_to_torch_cpu(tex, meta.batch_size, meta.height, meta.width);
  auto out = out_cpu.to(means2d.device());

  auto front_ids = tensor_from_buffer_u16_cpu(front_ids_buf, {meta.batch_size, meta.height, meta.width, meta.front_k});
  auto front_raw_alpha = tensor_from_buffer_f32_cpu(front_raw_buf, {meta.batch_size, meta.height, meta.width, meta.front_k});
  auto front_meta = tensor_from_buffer_u8_cpu(front_meta_buf, {meta.batch_size, meta.height, meta.width});

  return {out, packed_cpu, out_cpu, front_ids, front_raw_alpha, front_meta, tile_offsets_cpu, tile_ids_cpu};
}

BackwardOut metal_backward(
    const torch::Tensor& grad_out,
    const torch::Tensor& packed_gaussians,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32,
    const torch::Tensor& out_cpu,
    const torch::Tensor& front_ids,
    const torch::Tensor& front_raw_alpha,
    const torch::Tensor& front_meta,
    const torch::Tensor& tile_offsets,
    const torch::Tensor& tile_ids) {
  auto meta = parse_meta(meta_i32, meta_f32);
  auto& c = ctx();

  auto [gauss_buf, gauss_bytes] = make_buffer_from_tensor(c.device, packed_gaussians);
  auto [grad_buf, grad_bytes] = make_buffer_from_tensor(c.device, grad_out);
  auto [out_buf, out_bytes] = make_buffer_from_tensor(c.device, out_cpu);
  auto [front_ids_buf, front_ids_bytes] = make_buffer_from_tensor(c.device, front_ids);
  auto [front_raw_buf, front_raw_bytes] = make_buffer_from_tensor(c.device, front_raw_alpha);
  auto [front_meta_buf, front_meta_bytes] = make_buffer_from_tensor(c.device, front_meta);
  auto [tile_offsets_buf, tile_offsets_bytes] = make_buffer_from_tensor(c.device, tile_offsets);
  auto [tile_ids_buf, tile_ids_bytes] = make_buffer_from_tensor(c.device, tile_ids);

  CPUUniforms u;
  u.width = static_cast<uint32_t>(meta.width);
  u.height = static_cast<uint32_t>(meta.height);
  u.gaussians = static_cast<uint32_t>(meta.gaussians_per_batch);
  u.front_k = static_cast<uint32_t>(meta.front_k);
  u.tile_size = static_cast<uint32_t>(meta.tile_size);
  u.tiles_x = static_cast<uint32_t>((meta.width + meta.tile_size - 1) / meta.tile_size);
  u.tiles_y = static_cast<uint32_t>((meta.height + meta.tile_size - 1) / meta.tile_size);
  u.alpha_threshold = meta.alpha_threshold;
  u.bg_r = meta.bg_r;
  u.bg_g = meta.bg_g;
  u.bg_b = meta.bg_b;
  u.eps = meta.eps;
  id<MTLBuffer> u_buf = [c.device newBufferWithBytes:&u length:sizeof(u) options:MTLResourceStorageModeShared];

  NSUInteger grad_g_bytes = sizeof(float) * static_cast<NSUInteger>(meta.batch_size) *
                            static_cast<NSUInteger>(meta.gaussians_per_batch) * 4u;
  id<MTLBuffer> g_geom = make_zero_buffer(c.device, grad_g_bytes);
  id<MTLBuffer> g_conic = make_zero_buffer(c.device, grad_g_bytes);
  id<MTLBuffer> g_color = make_zero_buffer(c.device, grad_g_bytes);

  id<MTLCommandBuffer> cb = [c.queue commandBuffer];
  TORCH_CHECK(cb != nil, "Failed to create Metal command buffer");

  NSUInteger pixels_per_batch = static_cast<NSUInteger>(meta.width) * static_cast<NSUInteger>(meta.height);
  NSUInteger gauss_batch_bytes = gauss_bytes / meta.batch_size;
  NSUInteger grad_batch_bytes = grad_bytes / meta.batch_size;
  NSUInteger out_batch_bytes = out_bytes / meta.batch_size;
  NSUInteger front_ids_batch_bytes = front_ids_bytes / meta.batch_size;
  NSUInteger front_raw_batch_bytes = front_raw_bytes / meta.batch_size;
  NSUInteger front_meta_batch_bytes = front_meta_bytes / meta.batch_size;
  NSUInteger grad_gauss_batch_bytes = grad_g_bytes / meta.batch_size;
  NSUInteger tile_row_bytes = sizeof(int32_t) * (static_cast<NSUInteger>(u.tiles_x) * static_cast<NSUInteger>(u.tiles_y) + 1u);

  for (int b = 0; b < meta.batch_size; ++b) {
    NSUInteger tg = 256;

    id<MTLComputeCommandEncoder> front_enc = [cb computeCommandEncoder];
    [front_enc setComputePipelineState:c.backward_frontk_pso];
    [front_enc setBuffer:gauss_buf offset:gauss_batch_bytes * b atIndex:0];
    [front_enc setBuffer:grad_buf offset:grad_batch_bytes * b atIndex:1];
    [front_enc setBuffer:front_ids_buf offset:front_ids_batch_bytes * b atIndex:2];
    [front_enc setBuffer:front_raw_buf offset:front_raw_batch_bytes * b atIndex:3];
    [front_enc setBuffer:front_meta_buf offset:front_meta_batch_bytes * b atIndex:4];
    [front_enc setBuffer:u_buf offset:0 atIndex:5];
    [front_enc setBuffer:g_geom offset:grad_gauss_batch_bytes * b atIndex:6];
    [front_enc setBuffer:g_conic offset:grad_gauss_batch_bytes * b atIndex:7];
    [front_enc setBuffer:g_color offset:grad_gauss_batch_bytes * b atIndex:8];
    [front_enc dispatchThreads:MTLSizeMake(pixels_per_batch, 1, 1)
     threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    [front_enc endEncoding];

    id<MTLComputeCommandEncoder> overflow_enc = [cb computeCommandEncoder];
    [overflow_enc setComputePipelineState:c.backward_overflow_pso];
    [overflow_enc setBuffer:gauss_buf offset:gauss_batch_bytes * b atIndex:0];
    [overflow_enc setBuffer:grad_buf offset:grad_batch_bytes * b atIndex:1];
    [overflow_enc setBuffer:out_buf offset:out_batch_bytes * b atIndex:2];
    [overflow_enc setBuffer:tile_offsets_buf offset:tile_row_bytes * b atIndex:3];
    [overflow_enc setBuffer:tile_ids_buf offset:0 atIndex:4];
    [overflow_enc setBuffer:front_meta_buf offset:front_meta_batch_bytes * b atIndex:5];
    [overflow_enc setBuffer:u_buf offset:0 atIndex:6];
    [overflow_enc setBuffer:g_geom offset:grad_gauss_batch_bytes * b atIndex:7];
    [overflow_enc setBuffer:g_conic offset:grad_gauss_batch_bytes * b atIndex:8];
    [overflow_enc setBuffer:g_color offset:grad_gauss_batch_bytes * b atIndex:9];
    [overflow_enc dispatchThreads:MTLSizeMake(pixels_per_batch, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    [overflow_enc endEncoding];
  }

  [cb commit];
  [cb waitUntilCompleted];

  auto g_geom_cpu = tensor_from_buffer_f32_cpu(g_geom, {meta.batch_size, meta.gaussians_per_batch, 4});
  auto g_conic_cpu = tensor_from_buffer_f32_cpu(g_conic, {meta.batch_size, meta.gaussians_per_batch, 4});
  auto g_color_cpu = tensor_from_buffer_f32_cpu(g_color, {meta.batch_size, meta.gaussians_per_batch, 4});

  auto out_dev = grad_out.device();
  auto g_geom_t = g_geom_cpu.to(out_dev);
  auto g_conic_t = g_conic_cpu.to(out_dev);
  auto g_color_t = g_color_cpu.to(out_dev);

  auto g_means = g_geom_t.slice(-1, 0, 2).contiguous();
  auto g_opacity = g_geom_t.select(-1, 2).contiguous();
  auto g_conics = g_conic_t.slice(-1, 0, 3).contiguous();
  auto g_colors = g_color_t.slice(-1, 0, 3).contiguous();
  auto g_depths = torch::zeros({meta.batch_size, meta.gaussians_per_batch},
                               torch::TensorOptions().dtype(torch::kFloat32).device(out_dev));
  return {g_means, g_conics, g_colors, g_opacity, g_depths};
}

}  // namespace gsplat

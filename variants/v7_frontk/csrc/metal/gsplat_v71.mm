#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <QuartzCore/QuartzCore.h>

#include <torch/extension.h>
#include <vector>
#include <mutex>
#include <tuple>
#include <cstring>

#include "shared/common.h"

namespace gsplat {
namespace {

static std::string read_shader_source() {
  NSString* metalPath = [[NSString stringWithUTF8String:__FILE__] stringByDeletingLastPathComponent];
  metalPath = [metalPath stringByAppendingPathComponent:@"gsplat_v71.metal"];
  NSError* err = nil;
  NSString* src = [NSString stringWithContentsOfFile:metalPath encoding:NSUTF8StringEncoding error:&err];
  TORCH_CHECK(src != nil, "Failed to read gsplat_v71.metal: ", err.localizedDescription.UTF8String);
  return std::string([src UTF8String]);
}

struct MetalContext {
  id<MTLDevice> device = nil;
  id<MTLCommandQueue> queue = nil;
  id<MTLLibrary> library = nil;
  id<MTLRenderPipelineState> render_pso = nil;
  id<MTLComputePipelineState> capture_pso = nil;
  id<MTLComputePipelineState> backward_frontk_pso = nil;
  id<MTLComputePipelineState> backward_overflow_pso = nil;
};

struct CPUUniforms {
  uint32_t width;
  uint32_t height;
  uint32_t gaussians;
  uint32_t front_k;
  float alpha_threshold;
  float bg_r;
  float bg_g;
  float bg_b;
  float eps;
};

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

    id<MTLFunction> capture = [c.library newFunctionWithName:@"capture_front_k"];
    c.capture_pso = [c.device newComputePipelineStateWithFunction:capture error:&err];
    TORCH_CHECK(c.capture_pso != nil, "Failed to create capture pipeline: ", err.localizedDescription.UTF8String);

    id<MTLFunction> bw_frontk = [c.library newFunctionWithName:@"backward_front_k"];
    c.backward_frontk_pso = [c.device newComputePipelineStateWithFunction:bw_frontk error:&err];
    TORCH_CHECK(c.backward_frontk_pso != nil, "Failed to create front-k backward pipeline: ", err.localizedDescription.UTF8String);

    id<MTLFunction> bw_overflow = [c.library newFunctionWithName:@"backward_overflow_replay"];
    c.backward_overflow_pso = [c.device newComputePipelineStateWithFunction:bw_overflow error:&err];
    TORCH_CHECK(c.backward_overflow_pso != nil, "Failed to create overflow backward pipeline: ", err.localizedDescription.UTF8String);
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
  TORCH_CHECK(meta.front_k > 0 && meta.front_k <= kMaxFrontK, "front_k must be in [1, ", kMaxFrontK, "]");
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
  auto packed = torch::empty({B * G, 12}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
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
  return {buf, bytes};
}

id<MTLBuffer> make_zero_buffer(id<MTLDevice> device, NSUInteger bytes) {
  id<MTLBuffer> buf = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
  TORCH_CHECK(buf != nil, "Failed to allocate Metal buffer");
  std::memset([buf contents], 0, bytes);
  return buf;
}

torch::Tensor texture_to_torch(id<MTLTexture> tex, int B, int H, int W, torch::Device out_dev) {
  auto out_cpu = torch::empty({B, H, W, 3}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
  auto* dst = out_cpu.data_ptr<float>();
  MTLRegion region = MTLRegionMake2D(0, 0, W, H);
  size_t bytesPerRow = sizeof(float) * 4 * W;
  std::vector<float> tmp(H * W * 4);
  for (int b = 0; b < B; ++b) {
    [tex getBytes:tmp.data() bytesPerRow:bytesPerRow bytesPerImage:bytesPerRow * H fromRegion:region mipmapLevel:0 slice:b];
    float* outb = dst + (size_t)b * H * W * 3;
    for (int i = 0; i < H * W; ++i) {
      outb[i * 3 + 0] = tmp[i * 4 + 0];
      outb[i * 3 + 1] = tmp[i * 4 + 1];
      outb[i * 3 + 2] = tmp[i * 4 + 2];
    }
  }
  return out_cpu.to(out_dev);
}

torch::Tensor tensor_from_buffer_i32(id<MTLBuffer> buf, std::vector<int64_t> shape, torch::Device out_dev) {
  auto cpu = torch::from_blob([buf contents], shape, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU)).clone();
  return cpu.to(out_dev);
}

torch::Tensor tensor_from_buffer_u8(id<MTLBuffer> buf, std::vector<int64_t> shape, torch::Device out_dev) {
  auto cpu = torch::from_blob([buf contents], shape, torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU)).clone();
  return cpu.to(out_dev);
}

torch::Tensor tensor_from_buffer_f32(id<MTLBuffer> buf, std::vector<int64_t> shape, torch::Device out_dev) {
  auto cpu = torch::from_blob([buf contents], shape, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU)).clone();
  return cpu.to(out_dev);
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
  auto [gauss_buf, gauss_bytes] = make_buffer_from_tensor(c.device, packed_cpu);

  CPUUniforms u;
  u.width = static_cast<uint32_t>(meta.width);
  u.height = static_cast<uint32_t>(meta.height);
  u.gaussians = static_cast<uint32_t>(meta.gaussians_per_batch);
  u.front_k = static_cast<uint32_t>(meta.front_k);
  u.alpha_threshold = meta.alpha_threshold;
  u.bg_r = meta.bg_r;
  u.bg_g = meta.bg_g;
  u.bg_b = meta.bg_b;
  u.eps = meta.eps;
  id<MTLBuffer> u_buf = [c.device newBufferWithBytes:&u length:sizeof(u) options:MTLResourceStorageModeShared];

  MTLTextureDescriptor* td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA32Float width:meta.width height:meta.height mipmapped:NO];
  td.textureType = MTLTextureType2DArray;
  td.arrayLength = meta.batch_size;
  td.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
  td.storageMode = MTLStorageModeShared;
  id<MTLTexture> tex = [c.device newTextureWithDescriptor:td];
  TORCH_CHECK(tex != nil, "Failed to allocate output texture");

  NSUInteger pixels_per_batch = static_cast<NSUInteger>(meta.width) * static_cast<NSUInteger>(meta.height);
  NSUInteger ids_bytes_batch = sizeof(int32_t) * pixels_per_batch * static_cast<NSUInteger>(meta.front_k);
  NSUInteger raw_bytes_batch = sizeof(float) * pixels_per_batch * static_cast<NSUInteger>(meta.front_k);
  NSUInteger count_bytes_batch = sizeof(int32_t) * pixels_per_batch;
  NSUInteger overflow_bytes_batch = sizeof(uint8_t) * pixels_per_batch;

  id<MTLBuffer> front_ids_buf = make_zero_buffer(c.device, ids_bytes_batch * meta.batch_size);
  id<MTLBuffer> front_raw_buf = make_zero_buffer(c.device, raw_bytes_batch * meta.batch_size);
  id<MTLBuffer> front_count_buf = make_zero_buffer(c.device, count_bytes_batch * meta.batch_size);
  id<MTLBuffer> overflow_buf = make_zero_buffer(c.device, overflow_bytes_batch * meta.batch_size);

  id<MTLCommandBuffer> cb = [c.queue commandBuffer];
  TORCH_CHECK(cb != nil, "Failed to create Metal command buffer");

  NSUInteger gauss_batch_bytes = gauss_bytes / meta.batch_size;
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
    [renc drawPrimitives:MTLPrimitiveTypeTriangleStrip vertexStart:0 vertexCount:4 instanceCount:meta.gaussians_per_batch];
    [renc endEncoding];

    id<MTLComputeCommandEncoder> cenc = [cb computeCommandEncoder];
    [cenc setComputePipelineState:c.capture_pso];
    [cenc setBuffer:gauss_buf offset:gauss_batch_bytes * b atIndex:0];
    [cenc setBuffer:u_buf offset:0 atIndex:1];
    [cenc setBuffer:front_ids_buf offset:ids_bytes_batch * b atIndex:2];
    [cenc setBuffer:front_raw_buf offset:raw_bytes_batch * b atIndex:3];
    [cenc setBuffer:front_count_buf offset:count_bytes_batch * b atIndex:4];
    [cenc setBuffer:overflow_buf offset:overflow_bytes_batch * b atIndex:5];
    NSUInteger tg = 256;
    [cenc dispatchThreads:MTLSizeMake(pixels_per_batch, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    [cenc endEncoding];
  }

  [cb commit];
  [cb waitUntilCompleted];

  auto out = texture_to_torch(tex, meta.batch_size, meta.height, meta.width, means2d.device());
  auto front_ids = tensor_from_buffer_i32(front_ids_buf, {meta.batch_size, meta.height, meta.width, meta.front_k}, means2d.device());
  auto front_raw_alpha = tensor_from_buffer_f32(front_raw_buf, {meta.batch_size, meta.height, meta.width, meta.front_k}, means2d.device());
  auto front_count = tensor_from_buffer_i32(front_count_buf, {meta.batch_size, meta.height, meta.width}, means2d.device());
  auto overflow_mask = tensor_from_buffer_u8(overflow_buf, {meta.batch_size, meta.height, meta.width}, means2d.device());
  return {out, front_ids, front_raw_alpha, front_count, overflow_mask};
}

BackwardOut metal_backward(
    const torch::Tensor& grad_out,
    const torch::Tensor& means2d,
    const torch::Tensor& conics,
    const torch::Tensor& colors,
    const torch::Tensor& opacities,
    const torch::Tensor& depths,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32,
    const torch::Tensor& out_image,
    const torch::Tensor& front_ids,
    const torch::Tensor& front_raw_alpha,
    const torch::Tensor& front_count,
    const torch::Tensor& overflow_mask) {
  auto meta = parse_meta(meta_i32, meta_f32);
  check_inputs(means2d, conics, colors, opacities, depths, meta);
  auto& c = ctx();

  auto packed_cpu = pack_gaussians_cpu(means2d, conics, colors, opacities);
  auto [gauss_buf, gauss_bytes] = make_buffer_from_tensor(c.device, packed_cpu);
  auto [grad_buf, grad_bytes] = make_buffer_from_tensor(c.device, grad_out);
  auto [out_buf, out_bytes] = make_buffer_from_tensor(c.device, out_image);
  auto [front_ids_buf, front_ids_bytes] = make_buffer_from_tensor(c.device, front_ids);
  auto [front_raw_buf, front_raw_bytes] = make_buffer_from_tensor(c.device, front_raw_alpha);
  auto [front_count_buf, front_count_bytes] = make_buffer_from_tensor(c.device, front_count);
  auto [overflow_buf, overflow_bytes] = make_buffer_from_tensor(c.device, overflow_mask);

  CPUUniforms u;
  u.width = static_cast<uint32_t>(meta.width);
  u.height = static_cast<uint32_t>(meta.height);
  u.gaussians = static_cast<uint32_t>(meta.gaussians_per_batch);
  u.front_k = static_cast<uint32_t>(meta.front_k);
  u.alpha_threshold = meta.alpha_threshold;
  u.bg_r = meta.bg_r;
  u.bg_g = meta.bg_g;
  u.bg_b = meta.bg_b;
  u.eps = meta.eps;
  id<MTLBuffer> u_buf = [c.device newBufferWithBytes:&u length:sizeof(u) options:MTLResourceStorageModeShared];

  NSUInteger grad_g_bytes = sizeof(float) * static_cast<NSUInteger>(meta.batch_size) * static_cast<NSUInteger>(meta.gaussians_per_batch) * 4u;
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
  NSUInteger front_count_batch_bytes = front_count_bytes / meta.batch_size;
  NSUInteger overflow_batch_bytes = overflow_bytes / meta.batch_size;
  NSUInteger grad_gauss_batch_bytes = grad_g_bytes / meta.batch_size;

  for (int b = 0; b < meta.batch_size; ++b) {
    NSUInteger tg = 256;

    id<MTLComputeCommandEncoder> front_enc = [cb computeCommandEncoder];
    [front_enc setComputePipelineState:c.backward_frontk_pso];
    [front_enc setBuffer:gauss_buf offset:gauss_batch_bytes * b atIndex:0];
    [front_enc setBuffer:grad_buf offset:grad_batch_bytes * b atIndex:1];
    [front_enc setBuffer:front_ids_buf offset:front_ids_batch_bytes * b atIndex:2];
    [front_enc setBuffer:front_raw_buf offset:front_raw_batch_bytes * b atIndex:3];
    [front_enc setBuffer:front_count_buf offset:front_count_batch_bytes * b atIndex:4];
    [front_enc setBuffer:overflow_buf offset:overflow_batch_bytes * b atIndex:5];
    [front_enc setBuffer:u_buf offset:0 atIndex:6];
    [front_enc setBuffer:g_geom offset:grad_gauss_batch_bytes * b atIndex:7];
    [front_enc setBuffer:g_conic offset:grad_gauss_batch_bytes * b atIndex:8];
    [front_enc setBuffer:g_color offset:grad_gauss_batch_bytes * b atIndex:9];
    [front_enc dispatchThreads:MTLSizeMake(pixels_per_batch, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    [front_enc endEncoding];

    id<MTLComputeCommandEncoder> overflow_enc = [cb computeCommandEncoder];
    [overflow_enc setComputePipelineState:c.backward_overflow_pso];
    [overflow_enc setBuffer:gauss_buf offset:gauss_batch_bytes * b atIndex:0];
    [overflow_enc setBuffer:grad_buf offset:grad_batch_bytes * b atIndex:1];
    [overflow_enc setBuffer:out_buf offset:out_batch_bytes * b atIndex:2];
    [overflow_enc setBuffer:overflow_buf offset:overflow_batch_bytes * b atIndex:3];
    [overflow_enc setBuffer:u_buf offset:0 atIndex:4];
    [overflow_enc setBuffer:g_geom offset:grad_gauss_batch_bytes * b atIndex:5];
    [overflow_enc setBuffer:g_conic offset:grad_gauss_batch_bytes * b atIndex:6];
    [overflow_enc setBuffer:g_color offset:grad_gauss_batch_bytes * b atIndex:7];
    [overflow_enc dispatchThreads:MTLSizeMake(pixels_per_batch, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    [overflow_enc endEncoding];
  }

  [cb commit];
  [cb waitUntilCompleted];

  auto g_geom_t = tensor_from_buffer_f32(g_geom, {meta.batch_size, meta.gaussians_per_batch, 4}, means2d.device());
  auto g_conic_t = tensor_from_buffer_f32(g_conic, {meta.batch_size, meta.gaussians_per_batch, 4}, means2d.device());
  auto g_color_t = tensor_from_buffer_f32(g_color, {meta.batch_size, meta.gaussians_per_batch, 4}, means2d.device());

  auto g_means = g_geom_t.slice(-1, 0, 2).contiguous();
  auto g_opacity = g_geom_t.select(-1, 2).contiguous();
  auto g_conics = g_conic_t.slice(-1, 0, 3).contiguous();
  auto g_colors = g_color_t.slice(-1, 0, 3).contiguous();
  auto g_depths = torch::zeros_like(depths);
  return {g_means, g_conics, g_colors, g_opacity, g_depths};
}

}  // namespace gsplat

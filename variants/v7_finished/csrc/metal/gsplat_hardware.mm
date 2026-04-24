#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <QuartzCore/QuartzCore.h>

#include <torch/extension.h>
#include <vector>
#include <mutex>
#include <tuple>

#include "shared/common.h"

namespace gsplat {
namespace {

struct ProjectedGaussianCPU {
  float geom[4];
  float conic[4];
  float color[4];
};

static std::string read_shader_source() {
  NSString* metalPath = [[NSString stringWithUTF8String:__FILE__] stringByDeletingLastPathComponent];
  metalPath = [metalPath stringByAppendingPathComponent:@"gsplat_v7_hardware.metal"];
  NSError* err = nil;
  NSString* src = [NSString stringWithContentsOfFile:metalPath encoding:NSUTF8StringEncoding error:&err];
  TORCH_CHECK(src != nil, "Failed to read gsplat_v7_hardware.metal: ", err.localizedDescription.UTF8String);
  return std::string([src UTF8String]);
}

struct MetalContext {
  id<MTLDevice> device = nil;
  id<MTLCommandQueue> queue = nil;
  id<MTLLibrary> library = nil;
  id<MTLRenderPipelineState> render_pso = nil;
  id<MTLComputePipelineState> backward_pso = nil;
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

    id<MTLFunction> bw = [c.library newFunctionWithName:@"backward_replay"];
    c.backward_pso = [c.device newComputePipelineStateWithFunction:bw error:&err];
    TORCH_CHECK(c.backward_pso != nil, "Failed to create backward compute pipeline: ", err.localizedDescription.UTF8String);
  });
  return c;
}

void check_inputs(const torch::Tensor& means2d,
                  const torch::Tensor& conics,
                  const torch::Tensor& colors,
                  const torch::Tensor& opacities,
                  const torch::Tensor& depths) {
  TORCH_CHECK(means2d.scalar_type() == torch::kFloat32, "means2d must be float32");
  TORCH_CHECK(conics.scalar_type() == torch::kFloat32, "conics must be float32");
  TORCH_CHECK(colors.scalar_type() == torch::kFloat32, "colors must be float32");
  TORCH_CHECK(opacities.scalar_type() == torch::kFloat32, "opacities must be float32");
  TORCH_CHECK(depths.scalar_type() == torch::kFloat32, "depths must be float32");
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
    out[i*12 + 0] = m[i*2 + 0];
    out[i*12 + 1] = m[i*2 + 1];
    out[i*12 + 2] = o[i];
    out[i*12 + 3] = 0.0f;
    out[i*12 + 4] = q[i*3 + 0];
    out[i*12 + 5] = q[i*3 + 1];
    out[i*12 + 6] = q[i*3 + 2];
    out[i*12 + 7] = 0.0f;
    out[i*12 + 8] = c[i*3 + 0];
    out[i*12 + 9] = c[i*3 + 1];
    out[i*12 + 10] = c[i*3 + 2];
    out[i*12 + 11] = 0.0f;
  }
  return packed;
}

struct CPUUniforms {
  uint32_t width;
  uint32_t height;
  uint32_t gaussians;
  float alpha_threshold;
  float bg_r;
  float bg_g;
  float bg_b;
  float eps;
};

std::tuple<id<MTLBuffer>, NSUInteger> make_buffer_from_tensor(id<MTLDevice> device, const torch::Tensor& t) {
  auto cpu = t.contiguous().to(torch::kCPU);
  NSUInteger bytes = cpu.nbytes();
  id<MTLBuffer> buf = [device newBufferWithBytes:cpu.data_ptr() length:bytes options:MTLResourceStorageModeShared];
  return {buf, bytes};
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
      outb[i*3 + 0] = tmp[i*4 + 0];
      outb[i*3 + 1] = tmp[i*4 + 1];
      outb[i*3 + 2] = tmp[i*4 + 2];
    }
  }
  return out_cpu.to(out_dev);
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
  check_inputs(means2d, conics, colors, opacities, depths);
  auto meta = parse_meta(meta_i32, meta_f32);
  auto& c = ctx();

  auto packed_cpu = pack_gaussians_cpu(means2d, conics, colors, opacities);
  auto [gauss_buf, gauss_bytes] = make_buffer_from_tensor(c.device, packed_cpu);

  CPUUniforms u;
  u.width = (uint32_t)meta.width;
  u.height = (uint32_t)meta.height;
  u.gaussians = (uint32_t)meta.gaussians_per_batch;
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

  id<MTLCommandBuffer> cb = [c.queue commandBuffer];
  TORCH_CHECK(cb != nil, "Failed to create Metal command buffer");

  for (int b = 0; b < meta.batch_size; ++b) {
    MTLRenderPassDescriptor* rp = [MTLRenderPassDescriptor renderPassDescriptor];
    rp.colorAttachments[0].texture = tex;
    rp.colorAttachments[0].slice = b;
    rp.colorAttachments[0].loadAction = MTLLoadActionClear;
    rp.colorAttachments[0].storeAction = MTLStoreActionStore;
    rp.colorAttachments[0].clearColor = MTLClearColorMake(meta.bg_r, meta.bg_g, meta.bg_b, 1.0);

    id<MTLRenderCommandEncoder> enc = [cb renderCommandEncoderWithDescriptor:rp];
    [enc setRenderPipelineState:c.render_pso];
    [enc setVertexBuffer:gauss_buf offset:gauss_bytes / meta.batch_size * b atIndex:0];
    [enc setVertexBuffer:u_buf offset:0 atIndex:1];
    [enc setFragmentBuffer:u_buf offset:0 atIndex:0];
    [enc drawPrimitives:MTLPrimitiveTypeTriangleStrip vertexStart:0 vertexCount:4 instanceCount:meta.gaussians_per_batch];
    [enc endEncoding];
  }

  [cb commit];
  [cb waitUntilCompleted];

  auto out = texture_to_torch(tex, meta.batch_size, meta.height, meta.width, means2d.device());
  // save output image as aux so backward can use the exact forward result
  auto aux = out.clone();
  return {out, aux};
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
    const torch::Tensor& aux) {
  check_inputs(means2d, conics, colors, opacities, depths);
  auto meta = parse_meta(meta_i32, meta_f32);
  auto& c = ctx();

  auto packed_cpu = pack_gaussians_cpu(means2d, conics, colors, opacities);
  auto [gauss_buf, gauss_bytes] = make_buffer_from_tensor(c.device, packed_cpu);
  auto [grad_buf, grad_bytes] = make_buffer_from_tensor(c.device, grad_out);
  auto [out_buf, out_bytes] = make_buffer_from_tensor(c.device, aux);

  CPUUniforms u;
  u.width = (uint32_t)meta.width;
  u.height = (uint32_t)meta.height;
  u.gaussians = (uint32_t)meta.gaussians_per_batch;
  u.alpha_threshold = meta.alpha_threshold;
  u.bg_r = meta.bg_r;
  u.bg_g = meta.bg_g;
  u.bg_b = meta.bg_b;
  u.eps = meta.eps;
  id<MTLBuffer> u_buf = [c.device newBufferWithBytes:&u length:sizeof(u) options:MTLResourceStorageModeShared];

  NSUInteger grad_g_bytes = sizeof(float) * meta.batch_size * meta.gaussians_per_batch * 4;
  id<MTLBuffer> g_geom = [c.device newBufferWithLength:grad_g_bytes options:MTLResourceStorageModeShared];
  id<MTLBuffer> g_conic = [c.device newBufferWithLength:grad_g_bytes options:MTLResourceStorageModeShared];
  id<MTLBuffer> g_color = [c.device newBufferWithLength:grad_g_bytes options:MTLResourceStorageModeShared];
  memset([g_geom contents], 0, grad_g_bytes);
  memset([g_conic contents], 0, grad_g_bytes);
  memset([g_color contents], 0, grad_g_bytes);

  id<MTLCommandBuffer> cb = [c.queue commandBuffer];
  TORCH_CHECK(cb != nil, "Failed to create Metal command buffer");

  for (int b = 0; b < meta.batch_size; ++b) {
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:c.backward_pso];
    [enc setBuffer:gauss_buf offset:gauss_bytes / meta.batch_size * b atIndex:0];
    [enc setBuffer:grad_buf offset:grad_bytes / meta.batch_size * b atIndex:1];
    [enc setBuffer:out_buf offset:out_bytes / meta.batch_size * b atIndex:2];
    [enc setBuffer:u_buf offset:0 atIndex:3];
    [enc setBuffer:g_geom offset:grad_g_bytes / meta.batch_size * b atIndex:4];
    [enc setBuffer:g_conic offset:grad_g_bytes / meta.batch_size * b atIndex:5];
    [enc setBuffer:g_color offset:grad_g_bytes / meta.batch_size * b atIndex:6];
    NSUInteger P = meta.width * meta.height;
    NSUInteger tg = 256;
    [enc dispatchThreads:MTLSizeMake(P,1,1) threadsPerThreadgroup:MTLSizeMake(tg,1,1)];
    [enc endEncoding];
  }

  [cb commit];
  [cb waitUntilCompleted];

  auto make_out = [&](id<MTLBuffer> buf) {
    auto cpu = torch::from_blob([buf contents], {meta.batch_size, meta.gaussians_per_batch, 4}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU)).clone();
    return cpu.to(means2d.device());
  };
  auto g_geom_t = make_out(g_geom);
  auto g_conic_t = make_out(g_conic);
  auto g_color_t = make_out(g_color);
  auto g_means = g_geom_t.slice(-1, 0, 2).contiguous();
  auto g_opacity = g_geom_t.select(-1, 2).contiguous();
  auto g_conics = g_conic_t.slice(-1, 0, 3).contiguous();
  auto g_colors = g_color_t.slice(-1, 0, 3).contiguous();
  auto g_depths = torch::zeros_like(depths);
  return {g_means, g_conics, g_colors, g_opacity, g_depths};
}

}  // namespace gsplat

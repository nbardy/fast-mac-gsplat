#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <torch/extension.h>
#include <torch/mps.h>
#include <ATen/mps/MPSStream.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <string>

namespace py = pybind11;

namespace {

const char* kRenderProbeSource = R"METAL(
#include <metal_stdlib>
using namespace metal;

struct VSOut {
  float4 position [[position]];
};

vertex VSOut v9_fullscreen_vs(uint vid [[vertex_id]]) {
  constexpr float2 pos[3] = {
      float2(-1.0f, -1.0f),
      float2( 3.0f, -1.0f),
      float2(-1.0f,  3.0f),
  };
  VSOut out;
  out.position = float4(pos[vid], 0.0f, 1.0f);
  return out;
}

fragment float4 v9_constant_fs(VSOut in [[stage_in]], constant float4& rgba [[buffer(0)]]) {
  return rgba;
}

struct V9GaussianEvalParams {
  uint width;
  uint height;
  uint count;
  float alpha_threshold;
  float eps;
};

struct V9GaussianVSOut {
  float4 position [[position]];
  float2 mean_px;
  float3 conic;
  float3 color;
  float opacity;
};

inline bool v9_alpha_support_params(float opacity, constant V9GaussianEvalParams& params, thread float& tau) {
  if (opacity <= params.alpha_threshold) {
    return false;
  }
  float ratio = max(params.alpha_threshold / max(opacity, params.eps), params.eps);
  tau = -2.0f * log(ratio);
  return isfinite(tau) && tau > 0.0f;
}

inline float2 v9_quad_corner_px(uint vid, float x0, float y0, float x1, float y1) {
  switch (vid & 3u) {
    case 0u:
      return float2(x0, y0);
    case 1u:
      return float2(x1, y0);
    case 2u:
      return float2(x0, y1);
    default:
      return float2(x1, y1);
  }
}

inline float v9_px_to_ndc_x(float x, uint width) {
  return 2.0f * (x / float(width)) - 1.0f;
}

inline float v9_px_to_ndc_y(float y, uint height) {
  return 1.0f - 2.0f * (y / float(height));
}

vertex V9GaussianVSOut v9_gaussian_quad_vs(
    uint vid [[vertex_id]],
    uint iid [[instance_id]],
    const device float* means2d [[buffer(0)]],
    const device float* conics [[buffer(1)]],
    const device float* colors [[buffer(2)]],
    const device float* opacities [[buffer(3)]],
    constant V9GaussianEvalParams& params [[buffer(4)]]) {
  V9GaussianVSOut out;
  const uint g = iid;
  const float2 mean = float2(means2d[g * 2u + 0u], means2d[g * 2u + 1u]);
  const float3 conic = float3(conics[g * 3u + 0u], conics[g * 3u + 1u], conics[g * 3u + 2u]);
  const float3 color = float3(colors[g * 3u + 0u], colors[g * 3u + 1u], colors[g * 3u + 2u]);
  const float opacity = opacities[g];

  float tau = 0.0f;
  float4 pos = float4(-2.0f, -2.0f, 0.0f, 1.0f);
  if (g < params.count && v9_alpha_support_params(opacity, params, tau)) {
    const float a = conic.x;
    const float b = conic.y;
    const float c = conic.z;
    const float det = max(a * c - b * b, params.eps);
    const float hx = sqrt(max(tau * c / det, 0.0f));
    const float hy = sqrt(max(tau * a / det, 0.0f));
    const float x0 = clamp(mean.x - hx, 0.0f, float(params.width));
    const float x1 = clamp(mean.x + hx, 0.0f, float(params.width));
    const float y0 = clamp(mean.y - hy, 0.0f, float(params.height));
    const float y1 = clamp(mean.y + hy, 0.0f, float(params.height));
    const float2 corner = v9_quad_corner_px(vid, x0, y0, x1, y1);
    pos = float4(v9_px_to_ndc_x(corner.x, params.width),
                 v9_px_to_ndc_y(corner.y, params.height),
                 0.0f,
                 1.0f);
  }

  out.position = pos;
  out.mean_px = mean;
  out.conic = conic;
  out.color = color;
  out.opacity = opacity;
  return out;
}

fragment float4 v9_gaussian_quad_fs(
    V9GaussianVSOut in [[stage_in]],
    constant V9GaussianEvalParams& params [[buffer(0)]]) {
  const float2 d = in.position.xy - in.mean_px;
  const float power = -0.5f * (in.conic.x * d.x * d.x +
                               2.0f * in.conic.y * d.x * d.y +
                               in.conic.z * d.y * d.y);
  if (power > 0.0f) {
    discard_fragment();
  }
  const float alpha = min(0.99f, in.opacity * exp(power));
  if (alpha < params.alpha_threshold) {
    discard_fragment();
  }
  return float4(in.color * alpha, alpha);
}
)METAL";

const char* kTileProbeSource = R"METAL(
#include <metal_stdlib>
using namespace metal;

struct V9TilePixel {
  half4 color;
};

kernel void v9_tile_imageblock_probe(
    imageblock<V9TilePixel> imageblock_data,
    ushort2 tid [[thread_position_in_threadgroup]]) {
  imageblock_data.write(V9TilePixel{half4(0.0h, 0.0h, 0.0h, 1.0h)}, tid, 0xF);
}
)METAL";

std::string nsstring_to_string(NSString* s) {
  if (s == nil) return "";
  const char* raw = [s UTF8String];
  return raw == nullptr ? "" : std::string(raw);
}

std::string nserror_to_string(NSError* err) {
  if (err == nil) return "";
  return nsstring_to_string([err localizedDescription]);
}

id<MTLDevice> system_device() {
  id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  TORCH_CHECK(device != nil, "No Metal device found");
  return device;
}

id<MTLLibrary> compile_library(id<MTLDevice> device, const char* source, std::string& error) {
  NSError* err = nil;
  NSString* src = [NSString stringWithUTF8String:source];
  id<MTLLibrary> library = [device newLibraryWithSource:src options:nil error:&err];
  if (library == nil) {
    error = nserror_to_string(err);
  }
  return library;
}

struct GaussianEvalParams {
  uint32_t width;
  uint32_t height;
  uint32_t count;
  float alpha_threshold;
  float eps;
};

id<MTLRenderPipelineState> build_render_pipeline(id<MTLDevice> device, std::string& error) {
  id<MTLLibrary> library = compile_library(device, kRenderProbeSource, error);
  if (library == nil) return nil;

  id<MTLFunction> vs = [library newFunctionWithName:@"v9_fullscreen_vs"];
  id<MTLFunction> fs = [library newFunctionWithName:@"v9_constant_fs"];
  if (vs == nil || fs == nil) {
    error = "compiled render library is missing v9_fullscreen_vs or v9_constant_fs";
    return nil;
  }

  MTLRenderPipelineDescriptor* desc = [[MTLRenderPipelineDescriptor alloc] init];
  desc.label = @"v9_hw_fixed_eval_render_probe";
  desc.vertexFunction = vs;
  desc.fragmentFunction = fs;
  desc.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA32Float;

  NSError* err = nil;
  id<MTLRenderPipelineState> pso = [device newRenderPipelineStateWithDescriptor:desc error:&err];
  if (pso == nil) {
    error = nserror_to_string(err);
  }
  return pso;
}

id<MTLRenderPipelineState> build_gaussian_eval_pipeline(id<MTLDevice> device, std::string& error) {
  id<MTLLibrary> library = compile_library(device, kRenderProbeSource, error);
  if (library == nil) return nil;

  id<MTLFunction> vs = [library newFunctionWithName:@"v9_gaussian_quad_vs"];
  id<MTLFunction> fs = [library newFunctionWithName:@"v9_gaussian_quad_fs"];
  if (vs == nil || fs == nil) {
    error = "compiled render library is missing v9_gaussian_quad_vs or v9_gaussian_quad_fs";
    return nil;
  }

  MTLRenderPipelineDescriptor* desc = [[MTLRenderPipelineDescriptor alloc] init];
  desc.label = @"v9_hw_fixed_eval_gaussian_eval_probe";
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

  NSError* err = nil;
  id<MTLRenderPipelineState> pso = [device newRenderPipelineStateWithDescriptor:desc error:&err];
  if (pso == nil) {
    error = nserror_to_string(err);
  }
  return pso;
}

id<MTLRenderPipelineState> cached_render_pipeline(id<MTLDevice> device) {
  static std::mutex mu;
  static id<MTLRenderPipelineState> pso = nil;
  static std::string error;
  std::lock_guard<std::mutex> lock(mu);
  if (pso == nil && error.empty()) {
    pso = build_render_pipeline(device, error);
  }
  TORCH_CHECK(pso != nil, "Failed to build v9 render pipeline: ", error);
  return pso;
}

id<MTLRenderPipelineState> cached_gaussian_eval_pipeline(id<MTLDevice> device) {
  static std::mutex mu;
  static id<MTLRenderPipelineState> pso = nil;
  static std::string error;
  std::lock_guard<std::mutex> lock(mu);
  if (pso == nil && error.empty()) {
    pso = build_gaussian_eval_pipeline(device, error);
  }
  TORCH_CHECK(pso != nil, "Failed to build v9 Gaussian eval pipeline: ", error);
  return pso;
}

id<MTLRenderPipelineState> build_tile_pipeline(
    id<MTLDevice> device,
    std::string& error,
    NSUInteger& imageblock_sample_length,
    NSUInteger& imageblock_memory_16x16) {
  imageblock_sample_length = 0;
  imageblock_memory_16x16 = 0;
  id<MTLLibrary> library = compile_library(device, kTileProbeSource, error);
  if (library == nil) return nil;

  id<MTLFunction> tile_fn = [library newFunctionWithName:@"v9_tile_imageblock_probe"];
  if (tile_fn == nil) {
    error = "compiled tile library is missing v9_tile_imageblock_probe";
    return nil;
  }

  if (@available(macOS 11.0, *)) {
    MTLTileRenderPipelineDescriptor* desc = [[MTLTileRenderPipelineDescriptor alloc] init];
    desc.label = @"v9_hw_fixed_eval_tile_imageblock_probe";
    desc.tileFunction = tile_fn;
    desc.threadgroupSizeMatchesTileSize = YES;
    desc.maxTotalThreadsPerThreadgroup = 256;
    desc.rasterSampleCount = 1;
    desc.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA32Float;

    NSError* err = nil;
    id<MTLRenderPipelineState> pso =
        [device newRenderPipelineStateWithTileDescriptor:desc options:MTLPipelineOptionNone reflection:nil error:&err];
    if (pso == nil) {
      error = nserror_to_string(err);
      return nil;
    }
    imageblock_sample_length = [pso imageblockSampleLength];
    imageblock_memory_16x16 = [pso imageblockMemoryLengthForDimensions:MTLSizeMake(16, 16, 1)];
    return pso;
  }

  error = "MTLTileRenderPipelineDescriptor requires macOS 11.0 or newer";
  return nil;
}

id<MTLBuffer> tensor_mtl_buffer(const torch::Tensor& tensor) {
  return __builtin_bit_cast(id<MTLBuffer>, tensor.storage().data());
}

torch::Tensor make_output_tensor(int64_t height, int64_t width) {
  auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kMPS);
  torch::Tensor out = torch::empty({height, width, 4}, opts);
  TORCH_CHECK(out.is_contiguous(), "new MPS output tensor is unexpectedly non-contiguous");
  return out;
}

void check_gaussian_eval_inputs(
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
  TORCH_CHECK(means2d.size(0) == conics.size(0) &&
                  means2d.size(0) == colors.size(0) &&
                  means2d.size(0) == opacities.size(0),
              "all Gaussian inputs must have the same G dimension");
  TORCH_CHECK(means2d.is_contiguous(), "means2d must be contiguous");
  TORCH_CHECK(conics.is_contiguous(), "conics must be contiguous");
  TORCH_CHECK(colors.is_contiguous(), "colors must be contiguous");
  TORCH_CHECK(opacities.is_contiguous(), "opacities must be contiguous");
  TORCH_CHECK(means2d.size(0) <= UINT32_MAX, "G must fit uint32");
}

void encode_constant_render(
    id<MTLCommandBuffer> command_buffer,
    id<MTLRenderPipelineState> pso,
    id<MTLTexture> texture,
    const std::array<float, 4>& rgba) {
  MTLRenderPassDescriptor* pass = [MTLRenderPassDescriptor renderPassDescriptor];
  pass.colorAttachments[0].texture = texture;
  pass.colorAttachments[0].loadAction = MTLLoadActionClear;
  pass.colorAttachments[0].storeAction = MTLStoreActionStore;
  pass.colorAttachments[0].clearColor = MTLClearColorMake(0.0, 0.0, 0.0, 0.0);

  id<MTLRenderCommandEncoder> render_encoder = [command_buffer renderCommandEncoderWithDescriptor:pass];
  if (render_encoder == nil) {
    @throw [NSException exceptionWithName:@"V9HWFixedEval"
                                   reason:@"failed to create render command encoder"
                                 userInfo:nil];
  }
  [render_encoder setRenderPipelineState:pso];
  [render_encoder setFragmentBytes:rgba.data() length:sizeof(float) * rgba.size() atIndex:0];
  [render_encoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
  [render_encoder endEncoding];
}

void encode_gaussian_eval_render(
    id<MTLCommandBuffer> command_buffer,
    id<MTLRenderPipelineState> pso,
    id<MTLTexture> texture,
    id<MTLBuffer> means_buffer,
    NSUInteger means_offset,
    id<MTLBuffer> conics_buffer,
    NSUInteger conics_offset,
    id<MTLBuffer> colors_buffer,
    NSUInteger colors_offset,
    id<MTLBuffer> opacities_buffer,
    NSUInteger opacities_offset,
    const GaussianEvalParams& params) {
  MTLRenderPassDescriptor* pass = [MTLRenderPassDescriptor renderPassDescriptor];
  pass.colorAttachments[0].texture = texture;
  pass.colorAttachments[0].loadAction = MTLLoadActionClear;
  pass.colorAttachments[0].storeAction = MTLStoreActionStore;
  pass.colorAttachments[0].clearColor = MTLClearColorMake(0.0, 0.0, 0.0, 0.0);

  id<MTLRenderCommandEncoder> render_encoder = [command_buffer renderCommandEncoderWithDescriptor:pass];
  if (render_encoder == nil) {
    @throw [NSException exceptionWithName:@"V9HWFixedEval"
                                   reason:@"failed to create Gaussian eval render command encoder"
                                 userInfo:nil];
  }
  [render_encoder setRenderPipelineState:pso];
  [render_encoder setVertexBuffer:means_buffer offset:means_offset atIndex:0];
  [render_encoder setVertexBuffer:conics_buffer offset:conics_offset atIndex:1];
  [render_encoder setVertexBuffer:colors_buffer offset:colors_offset atIndex:2];
  [render_encoder setVertexBuffer:opacities_buffer offset:opacities_offset atIndex:3];
  [render_encoder setVertexBytes:&params length:sizeof(GaussianEvalParams) atIndex:4];
  [render_encoder setFragmentBytes:&params length:sizeof(GaussianEvalParams) atIndex:0];
  if (params.count > 0) {
    [render_encoder drawPrimitives:MTLPrimitiveTypeTriangleStrip
                       vertexStart:0
                       vertexCount:4
                     instanceCount:(NSUInteger)params.count];
  }
  [render_encoder endEncoding];
}

torch::Tensor render_constant_rgba_native(
    int64_t height,
    int64_t width,
    std::array<float, 4> rgba) {
  TORCH_CHECK(height > 0 && width > 0, "height and width must be positive");
  TORCH_CHECK(height <= INT32_MAX && width <= INT32_MAX, "height and width must fit int32");
  TORCH_CHECK(torch::mps::is_available(), "MPS is not available");

  id<MTLDevice> device = system_device();
  id<MTLRenderPipelineState> pso = cached_render_pipeline(device);

  torch::Tensor out = make_output_tensor(height, width);

  MTLTextureDescriptor* tex_desc = [MTLTextureDescriptor
      texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA32Float
                                   width:(NSUInteger)width
                                  height:(NSUInteger)height
                               mipmapped:NO];
  tex_desc.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
  tex_desc.storageMode = MTLStorageModePrivate;
  id<MTLTexture> texture = [device newTextureWithDescriptor:tex_desc];
  TORCH_CHECK(texture != nil, "Failed to allocate private RGBA32F render texture");

  id<MTLBuffer> out_buffer = tensor_mtl_buffer(out);
  TORCH_CHECK(out_buffer != nil, "Failed to access output tensor MTLBuffer storage");

  const NSUInteger dst_offset = (NSUInteger)(out.storage_offset() * out.element_size());
  const NSUInteger bytes_per_pixel = 4 * sizeof(float);
  const NSUInteger bytes_per_row = (NSUInteger)width * bytes_per_pixel;
  const NSUInteger bytes_per_image = bytes_per_row * (NSUInteger)height;
  const MTLSize source_size = MTLSizeMake((NSUInteger)width, (NSUInteger)height, 1);
  const MTLOrigin source_origin = MTLOriginMake(0, 0, 0);
  const std::array<float, 4> rgba_copy = rgba;

  dispatch_queue_t queue = torch::mps::get_dispatch_queue();
  TORCH_CHECK(queue != nullptr, "torch::mps::get_dispatch_queue returned null");

  __block NSString* block_error = nil;
  dispatch_sync(queue, ^{
    @autoreleasepool {
      at::mps::MPSStream* stream = at::mps::getCurrentMPSStream();
      if (stream != nullptr) {
        stream->endKernelCoalescing();
      }
      id<MTLCommandBuffer> command_buffer = torch::mps::get_command_buffer();
      if (command_buffer == nil) {
        block_error = @"torch::mps::get_command_buffer returned nil";
        return;
      }

      @try {
        encode_constant_render(command_buffer, pso, texture, rgba_copy);
      } @catch (NSException* exception) {
        block_error = [exception reason];
        return;
      }

      id<MTLBlitCommandEncoder> blit_encoder = [command_buffer blitCommandEncoder];
      if (blit_encoder == nil) {
        block_error = @"failed to create blit command encoder";
        return;
      }
      [blit_encoder copyFromTexture:texture
                         sourceSlice:0
                         sourceLevel:0
                        sourceOrigin:source_origin
                          sourceSize:source_size
                            toBuffer:out_buffer
                   destinationOffset:dst_offset
              destinationBytesPerRow:bytes_per_row
            destinationBytesPerImage:bytes_per_image];
      [blit_encoder endEncoding];
    }
  });

  TORCH_CHECK(block_error == nil, nsstring_to_string(block_error));
  return out;
}

torch::Tensor render_constant_rgba_direct_native(
    int64_t height,
    int64_t width,
    std::array<float, 4> rgba) {
  TORCH_CHECK(height > 0 && width > 0, "height and width must be positive");
  TORCH_CHECK(height <= INT32_MAX && width <= INT32_MAX, "height and width must fit int32");
  TORCH_CHECK(torch::mps::is_available(), "MPS is not available");

  id<MTLDevice> device = system_device();
  id<MTLRenderPipelineState> pso = cached_render_pipeline(device);
  torch::Tensor out = make_output_tensor(height, width);

  id<MTLBuffer> out_buffer = tensor_mtl_buffer(out);
  TORCH_CHECK(out_buffer != nil, "Failed to access output tensor MTLBuffer storage");

  const NSUInteger dst_offset = (NSUInteger)(out.storage_offset() * out.element_size());
  const NSUInteger bytes_per_pixel = 4 * sizeof(float);
  const NSUInteger bytes_per_row = (NSUInteger)width * bytes_per_pixel;
  TORCH_CHECK(bytes_per_row % 256 == 0,
              "direct render-to-buffer-backed-texture requires width*16 to be 256-byte aligned; got bytes_per_row=",
              bytes_per_row);

  MTLTextureDescriptor* tex_desc = [MTLTextureDescriptor
      texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA32Float
                                   width:(NSUInteger)width
                                  height:(NSUInteger)height
                               mipmapped:NO];
  tex_desc.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
  tex_desc.storageMode = [out_buffer storageMode];

  id<MTLTexture> texture = [out_buffer newTextureWithDescriptor:tex_desc offset:dst_offset bytesPerRow:bytes_per_row];
  TORCH_CHECK(texture != nil,
              "Failed to create a render-target texture view over Torch MPS tensor storage; "
              "this device/buffer may not support direct render-to-buffer texture interop");

  const std::array<float, 4> rgba_copy = rgba;
  dispatch_queue_t queue = torch::mps::get_dispatch_queue();
  TORCH_CHECK(queue != nullptr, "torch::mps::get_dispatch_queue returned null");

  __block NSString* block_error = nil;
  dispatch_sync(queue, ^{
    @autoreleasepool {
      at::mps::MPSStream* stream = at::mps::getCurrentMPSStream();
      if (stream != nullptr) {
        stream->endKernelCoalescing();
      }
      id<MTLCommandBuffer> command_buffer = torch::mps::get_command_buffer();
      if (command_buffer == nil) {
        block_error = @"torch::mps::get_command_buffer returned nil";
        return;
      }
      @try {
        encode_constant_render(command_buffer, pso, texture, rgba_copy);
      } @catch (NSException* exception) {
        block_error = [exception reason];
        return;
      }
    }
  });

  TORCH_CHECK(block_error == nil, nsstring_to_string(block_error));
  return out;
}

torch::Tensor render_gaussian_eval_rgba_native(
    const torch::Tensor& means2d,
    const torch::Tensor& conics,
    const torch::Tensor& colors,
    const torch::Tensor& opacities,
    int64_t height,
    int64_t width,
    bool direct) {
  TORCH_CHECK(height > 0 && width > 0, "height and width must be positive");
  TORCH_CHECK(height <= INT32_MAX && width <= INT32_MAX, "height and width must fit int32");
  TORCH_CHECK(torch::mps::is_available(), "MPS is not available");
  check_gaussian_eval_inputs(means2d, conics, colors, opacities);

  id<MTLDevice> device = system_device();
  id<MTLRenderPipelineState> pso = cached_gaussian_eval_pipeline(device);
  torch::Tensor out = make_output_tensor(height, width);

  id<MTLBuffer> means_buffer = tensor_mtl_buffer(means2d);
  id<MTLBuffer> conics_buffer = tensor_mtl_buffer(conics);
  id<MTLBuffer> colors_buffer = tensor_mtl_buffer(colors);
  id<MTLBuffer> opacities_buffer = tensor_mtl_buffer(opacities);
  id<MTLBuffer> out_buffer = tensor_mtl_buffer(out);
  TORCH_CHECK(means_buffer != nil, "Failed to access means2d MTLBuffer storage");
  TORCH_CHECK(conics_buffer != nil, "Failed to access conics MTLBuffer storage");
  TORCH_CHECK(colors_buffer != nil, "Failed to access colors MTLBuffer storage");
  TORCH_CHECK(opacities_buffer != nil, "Failed to access opacities MTLBuffer storage");
  TORCH_CHECK(out_buffer != nil, "Failed to access output tensor MTLBuffer storage");

  const NSUInteger means_offset = (NSUInteger)(means2d.storage_offset() * means2d.element_size());
  const NSUInteger conics_offset = (NSUInteger)(conics.storage_offset() * conics.element_size());
  const NSUInteger colors_offset = (NSUInteger)(colors.storage_offset() * colors.element_size());
  const NSUInteger opacities_offset = (NSUInteger)(opacities.storage_offset() * opacities.element_size());
  const NSUInteger dst_offset = (NSUInteger)(out.storage_offset() * out.element_size());
  const NSUInteger bytes_per_pixel = 4 * sizeof(float);
  const NSUInteger bytes_per_row = (NSUInteger)width * bytes_per_pixel;
  const NSUInteger bytes_per_image = bytes_per_row * (NSUInteger)height;

  MTLTextureDescriptor* tex_desc = [MTLTextureDescriptor
      texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA32Float
                                   width:(NSUInteger)width
                                  height:(NSUInteger)height
                               mipmapped:NO];
  tex_desc.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;

  id<MTLTexture> texture = nil;
  if (direct) {
    TORCH_CHECK(bytes_per_row % 256 == 0,
                "direct Gaussian eval render requires width*16 to be 256-byte aligned; got bytes_per_row=",
                bytes_per_row);
    tex_desc.storageMode = [out_buffer storageMode];
    texture = [out_buffer newTextureWithDescriptor:tex_desc offset:dst_offset bytesPerRow:bytes_per_row];
    TORCH_CHECK(texture != nil,
                "Failed to create a Gaussian eval render-target texture view over Torch MPS tensor storage");
  } else {
    tex_desc.storageMode = MTLStorageModePrivate;
    texture = [device newTextureWithDescriptor:tex_desc];
    TORCH_CHECK(texture != nil, "Failed to allocate private RGBA32F Gaussian eval render texture");
  }

  const GaussianEvalParams params{
      (uint32_t)width,
      (uint32_t)height,
      (uint32_t)means2d.size(0),
      1.0f / 255.0f,
      1.0e-8f,
  };
  const MTLSize source_size = MTLSizeMake((NSUInteger)width, (NSUInteger)height, 1);
  const MTLOrigin source_origin = MTLOriginMake(0, 0, 0);

  dispatch_queue_t queue = torch::mps::get_dispatch_queue();
  TORCH_CHECK(queue != nullptr, "torch::mps::get_dispatch_queue returned null");

  __block NSString* block_error = nil;
  dispatch_sync(queue, ^{
    @autoreleasepool {
      at::mps::MPSStream* stream = at::mps::getCurrentMPSStream();
      if (stream != nullptr) {
        stream->endKernelCoalescing();
      }
      id<MTLCommandBuffer> command_buffer = torch::mps::get_command_buffer();
      if (command_buffer == nil) {
        block_error = @"torch::mps::get_command_buffer returned nil";
        return;
      }

      @try {
        encode_gaussian_eval_render(
            command_buffer,
            pso,
            texture,
            means_buffer,
            means_offset,
            conics_buffer,
            conics_offset,
            colors_buffer,
            colors_offset,
            opacities_buffer,
            opacities_offset,
            params);
      } @catch (NSException* exception) {
        block_error = [exception reason];
        return;
      }

      if (!direct) {
        id<MTLBlitCommandEncoder> blit_encoder = [command_buffer blitCommandEncoder];
        if (blit_encoder == nil) {
          block_error = @"failed to create Gaussian eval blit command encoder";
          return;
        }
        [blit_encoder copyFromTexture:texture
                           sourceSlice:0
                           sourceLevel:0
                          sourceOrigin:source_origin
                            sourceSize:source_size
                              toBuffer:out_buffer
                     destinationOffset:dst_offset
                destinationBytesPerRow:bytes_per_row
              destinationBytesPerImage:bytes_per_image];
        [blit_encoder endEncoding];
      }
    }
  });

  TORCH_CHECK(block_error == nil, nsstring_to_string(block_error));
  return out;
}

py::dict probe_native(bool compile_pipelines, bool compile_advanced) {
  py::dict out;
  out["native_probe_available"] = true;
  out["native_probe_error"] = "";
  out["compile_pipelines_requested"] = compile_pipelines;
  out["compile_advanced_requested"] = compile_advanced;
  out["native_op_uses_cpu_readback"] = false;
  out["torch_mps_command_buffer_api"] = true;
  out["torch_mps_dispatch_queue_api"] = true;

  id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  out["metal_available"] = device != nil;
  out["metal_device_name"] = device == nil ? "" : nsstring_to_string([device name]);
  out["has_unified_memory"] = py::none();
  out["recommended_max_working_set_size"] = py::none();
  out["supports_family_apple4"] = py::none();
  out["supports_family_mac2"] = py::none();
  out["raster_order_groups_supported"] = py::none();
  out["render_pipeline_ready"] = py::none();
  out["render_pipeline_error"] = "";
  out["gaussian_eval_pipeline_ready"] = py::none();
  out["gaussian_eval_pipeline_error"] = "";
  out["tile_pipeline_ready"] = py::none();
  out["tile_pipeline_error"] = "";
  out["tile_imageblock_sample_length"] = py::none();
  out["tile_imageblock_memory_16x16"] = py::none();
  out["icb_created"] = py::none();
  out["icb_error"] = "";

  if (device == nil) {
    out["render_pipeline_error"] = "No Metal device found";
    return out;
  }

  if (@available(macOS 10.15, *)) {
    out["has_unified_memory"] = (bool)[device hasUnifiedMemory];
    out["recommended_max_working_set_size"] = (unsigned long long)[device recommendedMaxWorkingSetSize];
    out["supports_family_apple4"] = (bool)[device supportsFamily:MTLGPUFamilyApple4];
    out["supports_family_mac2"] = (bool)[device supportsFamily:MTLGPUFamilyMac2];
  }
  if (@available(macOS 10.13, *)) {
    out["raster_order_groups_supported"] = (bool)[device areRasterOrderGroupsSupported];
  }

  if (compile_pipelines) {
    std::string render_error;
    id<MTLRenderPipelineState> render_pso = build_render_pipeline(device, render_error);
    out["render_pipeline_ready"] = render_pso != nil;
    out["render_pipeline_error"] = render_error;

    std::string gaussian_error;
    id<MTLRenderPipelineState> gaussian_pso = build_gaussian_eval_pipeline(device, gaussian_error);
    out["gaussian_eval_pipeline_ready"] = gaussian_pso != nil;
    out["gaussian_eval_pipeline_error"] = gaussian_error;
  }

  if (compile_advanced) {
    std::string tile_error;
    NSUInteger sample_length = 0;
    NSUInteger memory_16x16 = 0;
    id<MTLRenderPipelineState> tile_pso =
        build_tile_pipeline(device, tile_error, sample_length, memory_16x16);
    out["tile_pipeline_ready"] = tile_pso != nil;
    out["tile_pipeline_error"] = tile_error;
    if (tile_pso != nil) {
      out["tile_imageblock_sample_length"] = (unsigned long long)sample_length;
      out["tile_imageblock_memory_16x16"] = (unsigned long long)memory_16x16;
    }

    if (@available(macOS 10.14, *)) {
      MTLIndirectCommandBufferDescriptor* icb_desc = [[MTLIndirectCommandBufferDescriptor alloc] init];
      icb_desc.commandTypes = MTLIndirectCommandTypeDraw;
      icb_desc.inheritPipelineState = YES;
      icb_desc.inheritBuffers = YES;
      icb_desc.maxVertexBufferBindCount = 0;
      icb_desc.maxFragmentBufferBindCount = 0;
      id<MTLIndirectCommandBuffer> icb =
          [device newIndirectCommandBufferWithDescriptor:icb_desc maxCommandCount:1 options:MTLResourceStorageModePrivate];
      out["icb_created"] = icb != nil;
      out["icb_error"] = icb == nil ? "newIndirectCommandBufferWithDescriptor returned nil" : "";
    } else {
      out["icb_created"] = false;
      out["icb_error"] = "MTLIndirectCommandBuffer requires macOS 10.14 or newer";
    }
  }

  return out;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "probe_native",
      &probe_native,
      py::arg("compile_pipelines") = true,
      py::arg("compile_advanced") = true,
      "Probe v9 hardware-raster interop prerequisites.");
  m.def(
      "render_constant_rgba",
      &render_constant_rgba_native,
      py::arg("height"),
      py::arg("width"),
      py::arg("rgba") = std::array<float, 4>{0.125f, 0.5f, 0.875f, 1.0f},
      "Render a constant RGBA32F image into a Torch MPS tensor through a Metal render pass and GPU blit.");
  m.def(
      "render_constant_rgba_direct",
      &render_constant_rgba_direct_native,
      py::arg("height"),
      py::arg("width"),
      py::arg("rgba") = std::array<float, 4>{0.125f, 0.5f, 0.875f, 1.0f},
      "Render a constant RGBA32F image directly into a buffer-backed texture over Torch MPS tensor storage.");
  m.def(
      "render_gaussian_eval_rgba",
      &render_gaussian_eval_rgba_native,
      py::arg("means2d"),
      py::arg("conics"),
      py::arg("colors"),
      py::arg("opacities"),
      py::arg("height"),
      py::arg("width"),
      py::arg("direct") = true,
      "Render screen-space Gaussian splats from MPS tensors into an RGBA32F MPS tensor.");
}

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
  desc.label = @"v9_hw_interop_render_probe";
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
    desc.label = @"v9_hw_interop_tile_imageblock_probe";
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
    @throw [NSException exceptionWithName:@"V9HWInterop"
                                   reason:@"failed to create render command encoder"
                                 userInfo:nil];
  }
  [render_encoder setRenderPipelineState:pso];
  [render_encoder setFragmentBytes:rgba.data() length:sizeof(float) * rgba.size() atIndex:0];
  [render_encoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
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
}

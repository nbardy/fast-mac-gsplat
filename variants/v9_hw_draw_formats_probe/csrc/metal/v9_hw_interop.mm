#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <torch/extension.h>
#include <torch/mps.h>
#include <ATen/mps/MPSStream.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <map>
#include <mutex>
#include <numeric>
#include <string>
#include <vector>

#include <pybind11/stl.h>

namespace py = pybind11;

namespace {

constexpr NSUInteger kTextureBufferRowAlignment = 256;

struct OutputFormatSpec {
  const char* name;
  MTLPixelFormat pixel_format;
  c10::ScalarType scalar_type;
  int64_t channels;
  NSUInteger bytes_per_pixel;
  const char* fragment_function;
  const char* torch_dtype_name;
};

const std::array<OutputFormatSpec, 4> kOutputFormats = {{
    {"rgba32f", MTLPixelFormatRGBA32Float, torch::kFloat32, 4, 16, "v9_constant_rgba_fs", "float32"},
    {"rgba16f", MTLPixelFormatRGBA16Float, torch::kFloat16, 4, 8, "v9_constant_rgba_fs", "float16"},
    {"r32f", MTLPixelFormatR32Float, torch::kFloat32, 1, 4, "v9_constant_r_fs", "float32"},
    {"rg32f", MTLPixelFormatRG32Float, torch::kFloat32, 2, 8, "v9_constant_rg_fs", "float32"},
}};

const OutputFormatSpec& output_format(std::string name) {
  for (const OutputFormatSpec& spec : kOutputFormats) {
    if (name == spec.name) {
      return spec;
    }
  }
  TORCH_CHECK(false, "unknown output format '", name, "'; expected one of rgba32f, rgba16f, r32f, rg32f");
}

const OutputFormatSpec& rgba32f_format() {
  return output_format("rgba32f");
}

NSUInteger width_multiple_for_row_alignment(const OutputFormatSpec& spec) {
  return kTextureBufferRowAlignment / std::gcd(kTextureBufferRowAlignment, spec.bytes_per_pixel);
}

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

fragment float4 v9_constant_rgba_fs(VSOut in [[stage_in]], constant float4& rgba [[buffer(0)]]) {
  return rgba;
}

fragment float2 v9_constant_rg_fs(VSOut in [[stage_in]], constant float4& rgba [[buffer(0)]]) {
  return rgba.xy;
}

fragment float v9_constant_r_fs(VSOut in [[stage_in]], constant float4& rgba [[buffer(0)]]) {
  return rgba.x;
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

id<MTLRenderPipelineState> build_render_pipeline(
    id<MTLDevice> device,
    const OutputFormatSpec& format,
    std::string& error) {
  id<MTLLibrary> library = compile_library(device, kRenderProbeSource, error);
  if (library == nil) return nil;

  id<MTLFunction> vs = [library newFunctionWithName:@"v9_fullscreen_vs"];
  id<MTLFunction> fs = [library newFunctionWithName:[NSString stringWithUTF8String:format.fragment_function]];
  if (vs == nil || fs == nil) {
    error = std::string("compiled render library is missing v9_fullscreen_vs or ") + format.fragment_function;
    return nil;
  }

  MTLRenderPipelineDescriptor* desc = [[MTLRenderPipelineDescriptor alloc] init];
  desc.label = @"v9_hw_draw_formats_render_probe";
  desc.vertexFunction = vs;
  desc.fragmentFunction = fs;
  desc.colorAttachments[0].pixelFormat = format.pixel_format;

  NSError* err = nil;
  id<MTLRenderPipelineState> pso = [device newRenderPipelineStateWithDescriptor:desc error:&err];
  if (pso == nil) {
    error = nserror_to_string(err);
  }
  return pso;
}

id<MTLRenderPipelineState> cached_render_pipeline(id<MTLDevice> device, const OutputFormatSpec& format) {
  static std::mutex mu;
  static std::map<std::string, id<MTLRenderPipelineState>> psos;
  static std::map<std::string, std::string> errors;

  const std::string key(format.name);
  std::lock_guard<std::mutex> lock(mu);
  if (psos.find(key) == psos.end() && errors.find(key) == errors.end()) {
    std::string error;
    id<MTLRenderPipelineState> pso = build_render_pipeline(device, format, error);
    if (pso == nil) {
      errors[key] = error;
    } else {
      psos[key] = pso;
    }
  }
  auto pso_it = psos.find(key);
  TORCH_CHECK(pso_it != psos.end() && pso_it->second != nil,
              "Failed to build v9 render pipeline for ",
              format.name,
              ": ",
              errors[key]);
  return pso_it->second;
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
    desc.label = @"v9_hw_draw_formats_tile_imageblock_probe";
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

torch::Tensor make_output_tensor(const OutputFormatSpec& format, int64_t height, int64_t width) {
  auto opts = torch::TensorOptions().dtype(format.scalar_type).device(torch::kMPS);
  torch::Tensor out =
      format.channels == 1 ? torch::empty({height, width}, opts) : torch::empty({height, width, format.channels}, opts);
  TORCH_CHECK(out.is_contiguous(), "new MPS output tensor is unexpectedly non-contiguous");
  return out;
}

NSUInteger tensor_bytes_per_row(const torch::Tensor& tensor) {
  TORCH_CHECK(tensor.dim() >= 2, "output tensor must be at least 2D");
  return (NSUInteger)(tensor.stride(0) * tensor.element_size());
}

MTLTextureDescriptor* render_texture_descriptor(
    const OutputFormatSpec& format,
    NSUInteger width,
    NSUInteger height,
    MTLStorageMode storage_mode) {
  MTLTextureDescriptor* tex_desc = [MTLTextureDescriptor
      texture2DDescriptorWithPixelFormat:format.pixel_format
                                   width:width
                                  height:height
                               mipmapped:NO];
  tex_desc.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
  tex_desc.storageMode = storage_mode;
  return tex_desc;
}

id<MTLTexture> new_buffer_backed_render_texture(
    const torch::Tensor& out,
    id<MTLBuffer> out_buffer,
    const OutputFormatSpec& format,
    int64_t height,
    int64_t width,
    NSUInteger& dst_offset,
    NSUInteger& bytes_per_row) {
  dst_offset = (NSUInteger)(out.storage_offset() * out.element_size());
  bytes_per_row = tensor_bytes_per_row(out);
  const NSUInteger expected_row_bytes = (NSUInteger)width * format.bytes_per_pixel;
  TORCH_CHECK(bytes_per_row == expected_row_bytes,
              "unexpected contiguous row pitch for ",
              format.name,
              ": got ",
              bytes_per_row,
              " bytes, expected ",
              expected_row_bytes);
  TORCH_CHECK(bytes_per_row % kTextureBufferRowAlignment == 0,
              "direct render-to-buffer-backed-texture for ",
              format.name,
              " requires row bytes to be 256-byte aligned; got bytes_per_row=",
              bytes_per_row,
              ", width must be a multiple of ",
              width_multiple_for_row_alignment(format));

  MTLTextureDescriptor* tex_desc =
      render_texture_descriptor(format, (NSUInteger)width, (NSUInteger)height, [out_buffer storageMode]);
  id<MTLTexture> texture = [out_buffer newTextureWithDescriptor:tex_desc offset:dst_offset bytesPerRow:bytes_per_row];
  TORCH_CHECK(texture != nil,
              "Failed to create a ",
              format.name,
              " render-target texture view over Torch MPS tensor storage; "
              "this device/buffer may not support direct render-to-buffer texture interop");
  return texture;
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
    @throw [NSException exceptionWithName:@"V9HWDrawFormats"
                                   reason:@"failed to create render command encoder"
                                 userInfo:nil];
  }
  [render_encoder setRenderPipelineState:pso];
  [render_encoder setFragmentBytes:rgba.data() length:sizeof(float) * rgba.size() atIndex:0];
  [render_encoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
  [render_encoder endEncoding];
}

void encode_constant_render_icb(
    id<MTLDevice> device,
    id<MTLCommandBuffer> command_buffer,
    id<MTLRenderPipelineState> pso,
    id<MTLTexture> texture,
    const std::array<float, 4>& rgba) {
  if (@available(macOS 10.14, *)) {
    MTLIndirectCommandBufferDescriptor* icb_desc = [[MTLIndirectCommandBufferDescriptor alloc] init];
    icb_desc.commandTypes = MTLIndirectCommandTypeDraw;
    icb_desc.inheritPipelineState = NO;
    icb_desc.inheritBuffers = NO;
    icb_desc.maxVertexBufferBindCount = 1;
    icb_desc.maxFragmentBufferBindCount = 1;

    id<MTLIndirectCommandBuffer> icb =
        [device newIndirectCommandBufferWithDescriptor:icb_desc maxCommandCount:1 options:0];
    if (icb == nil) {
      @throw [NSException exceptionWithName:@"V9HWDrawFormats"
                                     reason:@"failed to allocate one-command ICB"
                                   userInfo:nil];
    }

    [icb resetWithRange:NSMakeRange(0, 1)];
    id<MTLIndirectRenderCommand> cmd = [icb indirectRenderCommandAtIndex:0];
    if (cmd == nil) {
      @throw [NSException exceptionWithName:@"V9HWDrawFormats"
                                     reason:@"failed to access ICB render command 0"
                                   userInfo:nil];
    }
    id<MTLBuffer> rgba_buffer =
        [device newBufferWithBytes:rgba.data() length:sizeof(float) * rgba.size() options:MTLResourceStorageModeShared];
    if (rgba_buffer == nil) {
      @throw [NSException exceptionWithName:@"V9HWDrawFormats"
                                     reason:@"failed to allocate ICB fragment constant buffer"
                                   userInfo:nil];
    }
    std::array<float, 4> dummy_vertex = {0.0f, 0.0f, 0.0f, 0.0f};
    id<MTLBuffer> dummy_vertex_buffer =
        [device newBufferWithBytes:dummy_vertex.data() length:sizeof(float) * dummy_vertex.size() options:MTLResourceStorageModeShared];
    if (dummy_vertex_buffer == nil) {
      @throw [NSException exceptionWithName:@"V9HWDrawFormats"
                                     reason:@"failed to allocate ICB dummy vertex buffer"
                                   userInfo:nil];
    }
    [cmd setRenderPipelineState:pso];
    [cmd setVertexBuffer:dummy_vertex_buffer offset:0 atIndex:0];
    [cmd setFragmentBuffer:rgba_buffer offset:0 atIndex:0];
    [cmd drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3 instanceCount:1 baseInstance:0];

    id<MTLBlitCommandEncoder> blit_encoder = [command_buffer blitCommandEncoder];
    if (blit_encoder == nil) {
      @throw [NSException exceptionWithName:@"V9HWDrawFormats"
                                     reason:@"failed to create ICB optimize blit encoder"
                                   userInfo:nil];
    }
    [blit_encoder optimizeIndirectCommandBuffer:icb withRange:NSMakeRange(0, 1)];
    [blit_encoder endEncoding];

    MTLRenderPassDescriptor* pass = [MTLRenderPassDescriptor renderPassDescriptor];
    pass.colorAttachments[0].texture = texture;
    pass.colorAttachments[0].loadAction = MTLLoadActionClear;
    pass.colorAttachments[0].storeAction = MTLStoreActionStore;
    pass.colorAttachments[0].clearColor = MTLClearColorMake(0.0, 0.0, 0.0, 0.0);

    id<MTLRenderCommandEncoder> render_encoder = [command_buffer renderCommandEncoderWithDescriptor:pass];
    if (render_encoder == nil) {
      @throw [NSException exceptionWithName:@"V9HWDrawFormats"
                                     reason:@"failed to create render command encoder for ICB"
                                   userInfo:nil];
    }
    [render_encoder setRenderPipelineState:pso];
    [render_encoder setVertexBuffer:dummy_vertex_buffer offset:0 atIndex:0];
    [render_encoder setFragmentBuffer:rgba_buffer offset:0 atIndex:0];
    [render_encoder useResource:dummy_vertex_buffer usage:MTLResourceUsageRead stages:MTLRenderStageVertex];
    [render_encoder useResource:rgba_buffer usage:MTLResourceUsageRead stages:MTLRenderStageFragment];
    [render_encoder executeCommandsInBuffer:icb withRange:NSMakeRange(0, 1)];
    [render_encoder endEncoding];
    return;
  }

  @throw [NSException exceptionWithName:@"V9HWDrawFormats"
                                 reason:@"ICB execution requires macOS 10.14 or newer"
                               userInfo:nil];
}

torch::Tensor render_constant_rgba_native(
    int64_t height,
    int64_t width,
    std::array<float, 4> rgba) {
  TORCH_CHECK(height > 0 && width > 0, "height and width must be positive");
  TORCH_CHECK(height <= INT32_MAX && width <= INT32_MAX, "height and width must fit int32");
  TORCH_CHECK(torch::mps::is_available(), "MPS is not available");

  const OutputFormatSpec& format = rgba32f_format();
  id<MTLDevice> device = system_device();
  id<MTLRenderPipelineState> pso = cached_render_pipeline(device, format);

  torch::Tensor out = make_output_tensor(format, height, width);

  MTLTextureDescriptor* tex_desc =
      render_texture_descriptor(format, (NSUInteger)width, (NSUInteger)height, MTLStorageModePrivate);
  id<MTLTexture> texture = [device newTextureWithDescriptor:tex_desc];
  TORCH_CHECK(texture != nil, "Failed to allocate private RGBA32F render texture");

  id<MTLBuffer> out_buffer = tensor_mtl_buffer(out);
  TORCH_CHECK(out_buffer != nil, "Failed to access output tensor MTLBuffer storage");

  const NSUInteger dst_offset = (NSUInteger)(out.storage_offset() * out.element_size());
  const NSUInteger bytes_per_row = tensor_bytes_per_row(out);
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

torch::Tensor render_constant_format_direct_native(
    const std::string& format_name,
    int64_t height,
    int64_t width,
    std::array<float, 4> rgba,
    bool use_icb) {
  TORCH_CHECK(!use_icb,
              "ICB execution is disabled: minimal ICB render execution crashed in "
              "AGX executeCommandsInBufferCommon on macOS/Apple M4 during this probe.");
  TORCH_CHECK(height > 0 && width > 0, "height and width must be positive");
  TORCH_CHECK(height <= INT32_MAX && width <= INT32_MAX, "height and width must fit int32");
  TORCH_CHECK(torch::mps::is_available(), "MPS is not available");

  const OutputFormatSpec& format = output_format(format_name);
  id<MTLDevice> device = system_device();
  id<MTLRenderPipelineState> pso = cached_render_pipeline(device, format);
  torch::Tensor out = make_output_tensor(format, height, width);

  id<MTLBuffer> out_buffer = tensor_mtl_buffer(out);
  TORCH_CHECK(out_buffer != nil, "Failed to access output tensor MTLBuffer storage");

  NSUInteger dst_offset = 0;
  NSUInteger bytes_per_row = 0;
  id<MTLTexture> texture = new_buffer_backed_render_texture(out, out_buffer, format, height, width, dst_offset, bytes_per_row);

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
        if (use_icb) {
          encode_constant_render_icb(device, command_buffer, pso, texture, rgba_copy);
        } else {
          encode_constant_render(command_buffer, pso, texture, rgba_copy);
        }
      } @catch (NSException* exception) {
        block_error = [exception reason];
        return;
      }
    }
  });

  TORCH_CHECK(block_error == nil, nsstring_to_string(block_error));
  return out;
}

torch::Tensor render_constant_rgba_direct_native(
    int64_t height,
    int64_t width,
    std::array<float, 4> rgba) {
  return render_constant_format_direct_native("rgba32f", height, width, rgba, false);
}

torch::Tensor render_constant_rgba_direct_icb_native(
    int64_t height,
    int64_t width,
    std::array<float, 4> rgba) {
  TORCH_CHECK(false,
              "render_constant_rgba_direct_icb is disabled: minimal ICB execution crashed in "
              "AGX executeCommandsInBufferCommon on macOS/Apple M4 during this probe. "
              "Treat ICB execution as unsafe until reworked in a separate isolated harness.");
}

py::list probe_direct_output_formats_native(int64_t height, std::vector<int64_t> widths) {
  TORCH_CHECK(height > 0, "height must be positive");
  TORCH_CHECK(torch::mps::is_available(), "MPS is not available");

  id<MTLDevice> device = system_device();
  py::list rows;
  for (const OutputFormatSpec& format : kOutputFormats) {
    std::string pipeline_error;
    id<MTLRenderPipelineState> pso = build_render_pipeline(device, format, pipeline_error);
    const bool pipeline_ready = pso != nil;

    for (int64_t width : widths) {
      TORCH_CHECK(width > 0, "all widths must be positive");
      py::dict row;
      row["format"] = format.name;
      row["metal_pixel_format"] = (unsigned long long)format.pixel_format;
      row["torch_dtype"] = format.torch_dtype_name;
      row["channels"] = format.channels;
      row["bytes_per_pixel"] = (unsigned long long)format.bytes_per_pixel;
      row["height"] = height;
      row["width"] = width;
      const NSUInteger bytes_per_row = (NSUInteger)width * format.bytes_per_pixel;
      const bool row_aligned = (bytes_per_row % kTextureBufferRowAlignment) == 0;
      row["bytes_per_row"] = (unsigned long long)bytes_per_row;
      row["row_alignment_bytes"] = (unsigned long long)kTextureBufferRowAlignment;
      row["width_multiple_for_direct"] = (unsigned long long)width_multiple_for_row_alignment(format);
      row["row_256_aligned"] = row_aligned;
      row["render_pipeline_ready"] = pipeline_ready;
      row["render_pipeline_error"] = pipeline_error;
      row["buffer_backed_texture_created"] = false;
      row["buffer_backed_texture_error"] = "";

      if (!pipeline_ready) {
        row["buffer_backed_texture_error"] = "render pipeline did not compile";
        rows.append(row);
        continue;
      }
      if (!row_aligned) {
        row["buffer_backed_texture_error"] = "skipped because row bytes are not 256-byte aligned";
        rows.append(row);
        continue;
      }

      try {
        torch::Tensor out = make_output_tensor(format, height, width);
        id<MTLBuffer> out_buffer = tensor_mtl_buffer(out);
        TORCH_CHECK(out_buffer != nil, "Failed to access output tensor MTLBuffer storage");
        NSUInteger dst_offset = 0;
        NSUInteger bytes_per_row = 0;
        id<MTLTexture> texture =
            new_buffer_backed_render_texture(out, out_buffer, format, height, width, dst_offset, bytes_per_row);
        row["buffer_backed_texture_created"] = texture != nil;
        row["buffer_storage_mode"] = (unsigned long long)[out_buffer storageMode];
        if (texture == nil) {
          row["texture_storage_mode"] = py::none();
        } else {
          row["texture_storage_mode"] = (unsigned long long)[texture storageMode];
        }
      } catch (const c10::Error& err) {
        row["buffer_backed_texture_error"] = err.what_without_backtrace();
      } catch (const std::exception& err) {
        row["buffer_backed_texture_error"] = err.what();
      }
      rows.append(row);
    }
  }
  return rows;
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
  out["direct_output_formats"] = py::list();

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
    id<MTLRenderPipelineState> render_pso = build_render_pipeline(device, rgba32f_format(), render_error);
    out["render_pipeline_ready"] = render_pso != nil;
    out["render_pipeline_error"] = render_error;

    py::list format_rows;
    for (const OutputFormatSpec& format : kOutputFormats) {
      py::dict row;
      std::string format_error;
      id<MTLRenderPipelineState> format_pso = build_render_pipeline(device, format, format_error);
      row["format"] = format.name;
      row["torch_dtype"] = format.torch_dtype_name;
      row["channels"] = format.channels;
      row["bytes_per_pixel"] = (unsigned long long)format.bytes_per_pixel;
      row["width_multiple_for_direct"] = (unsigned long long)width_multiple_for_row_alignment(format);
      row["render_pipeline_ready"] = format_pso != nil;
      row["render_pipeline_error"] = format_error;
      format_rows.append(row);
    }
    out["direct_output_formats"] = format_rows;
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
      icb_desc.inheritPipelineState = NO;
      icb_desc.inheritBuffers = NO;
      icb_desc.maxVertexBufferBindCount = 1;
      icb_desc.maxFragmentBufferBindCount = 1;
      id<MTLIndirectCommandBuffer> icb =
          [device newIndirectCommandBufferWithDescriptor:icb_desc maxCommandCount:1 options:0];
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
      "render_constant_rgba_direct_icb",
      &render_constant_rgba_direct_icb_native,
      py::arg("height"),
      py::arg("width"),
      py::arg("rgba") = std::array<float, 4>{0.125f, 0.5f, 0.875f, 1.0f},
      "Disabled: the minimal ICB execute probe crashed in AGX executeCommandsInBufferCommon.");
  m.def(
      "render_constant_format_direct",
      [](const std::string& format_name, int64_t height, int64_t width, std::array<float, 4> rgba) {
        return render_constant_format_direct_native(format_name, height, width, rgba, false);
      },
      py::arg("format_name"),
      py::arg("height"),
      py::arg("width"),
      py::arg("rgba") = std::array<float, 4>{0.125f, 0.5f, 0.875f, 1.0f},
      "Render a constant image directly into a buffer-backed Torch MPS tensor using rgba32f, rgba16f, r32f, or rg32f.");
  m.def(
      "probe_direct_output_formats",
      &probe_direct_output_formats_native,
      py::arg("height") = 1,
      py::arg("widths") = std::vector<int64_t>{16, 32, 64, 1920, 3840},
      "Probe buffer-backed render target creation and row alignment for direct Torch MPS output formats.");
}

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
#include <vector>

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

struct V9TileHalf4 {
  half4 color;
};

struct V9TileCT {
  float4 c_t;
};

struct V9TileCTStop {
  float4 c_t;
  uint stop_count;
};

struct V9TileCTStopFlags {
  float4 c_t;
  uint stop_count;
  uint flags;
};

struct V9TileExecParams {
  uint width;
  uint height;
  uint tiles_x;
  uint tile_size;
};

kernel void v9_tile_half4_probe(
    imageblock<V9TileHalf4> imageblock_data,
    ushort2 tid [[thread_position_in_threadgroup]]) {
  threadgroup_imageblock V9TileHalf4* state = imageblock_data.data(tid, 0, imageblock_data_rate::color);
  state->color = half4(0.0h, 0.0h, 0.0h, 1.0h);
}

kernel void v9_tile_ct_probe(
    imageblock<V9TileCT> imageblock_data,
    ushort2 tid [[thread_position_in_threadgroup]]) {
  threadgroup_imageblock V9TileCT* state = imageblock_data.data(tid, 0, imageblock_data_rate::color);
  state->c_t = float4(0.0f, 0.0f, 0.0f, 1.0f);
}

kernel void v9_tile_ct_stop_probe(
    imageblock<V9TileCTStop> imageblock_data,
    ushort2 tid [[thread_position_in_threadgroup]]) {
  threadgroup_imageblock V9TileCTStop* state = imageblock_data.data(tid, 0, imageblock_data_rate::color);
  state->c_t = float4(0.0f, 0.0f, 0.0f, 1.0f);
  state->stop_count = 0u;
}

kernel void v9_tile_ct_stop_flags_probe(
    imageblock<V9TileCTStopFlags> imageblock_data,
    ushort2 tid [[thread_position_in_threadgroup]]) {
  threadgroup_imageblock V9TileCTStopFlags* state = imageblock_data.data(tid, 0, imageblock_data_rate::color);
  state->c_t = float4(0.0f, 0.0f, 0.0f, 1.0f);
  state->stop_count = 0u;
  state->flags = 0u;
}

kernel void v9_tile_exec_probe(
    imageblock<V9TileCTStopFlags> imageblock_data,
    device float4* reports [[buffer(0)]],
    constant V9TileExecParams& params [[buffer(1)]],
    ushort2 tid [[thread_position_in_threadgroup]],
    uint2 tile_id [[threadgroup_position_in_grid]]) {
  threadgroup_imageblock V9TileCTStopFlags* state = imageblock_data.data(tid, 0, imageblock_data_rate::color);
  state->c_t = float4(float(tile_id.x), float(tile_id.y), 0.25f, 0.75f);
  state->stop_count = uint(tid.x) + uint(tid.y) + 1u;
  state->flags = 0x5A5Au;
  const uint lane_written_state = state->stop_count + state->flags;

  threadgroup_barrier(mem_flags::mem_threadgroup_imageblock);

  if (tid.x == 0 && tid.y == 0) {
    const uint tile_index = tile_id.y * params.tiles_x + tile_id.x;
    reports[tile_index] = float4(
        9013.0f + float(tile_index),
        float(imageblock_data.get_width()),
        float(imageblock_data.get_height()),
        float(lane_written_state));
  }
}
)METAL";

struct TileLayoutSpec {
  const char* name;
  const char* function_name;
  NSUInteger logical_bytes_per_pixel;
  const char* purpose;
};

struct TilePipelineResult {
  id<MTLRenderPipelineState> pso;
  std::string error;
  NSUInteger imageblock_sample_length;
  NSUInteger imageblock_memory_8x8;
  NSUInteger imageblock_memory_16x16;
  NSUInteger imageblock_memory_32x32;
};

struct V9TileExecParamsHost {
  uint32_t width;
  uint32_t height;
  uint32_t tiles_x;
  uint32_t tile_size;
};

constexpr TileLayoutSpec kTileLayoutSpecs[] = {
    {"half4_baseline", "v9_tile_half4_probe", 8, "prior minimal imageblock sample probe"},
    {"ct_fp32", "v9_tile_ct_probe", 16, "C.rgb plus T packed as float4"},
    {"ct_stop_fp32_u32", "v9_tile_ct_stop_probe", 20, "C/T plus one uint stop_count or stopped flag"},
    {"ct_stop_flags_fp32_u32x2", "v9_tile_ct_stop_flags_probe", 24,
     "C/T plus stop_count and flags, matching the execution probe"},
};

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
  desc.label = @"v9_hw_tile_state_render_probe";
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

TilePipelineResult build_tile_pipeline_for_function(
    id<MTLDevice> device,
    const char* function_name,
    NSString* label,
    NSUInteger max_threads_per_threadgroup) {
  TilePipelineResult result{nil, "", 0, 0, 0, 0};
  id<MTLLibrary> library = compile_library(device, kTileProbeSource, result.error);
  if (library == nil) return result;

  id<MTLFunction> tile_fn = [library newFunctionWithName:[NSString stringWithUTF8String:function_name]];
  if (tile_fn == nil) {
    result.error = std::string("compiled tile library is missing ") + function_name;
    return result;
  }

  if (@available(macOS 11.0, *)) {
    MTLTileRenderPipelineDescriptor* desc = [[MTLTileRenderPipelineDescriptor alloc] init];
    desc.label = label;
    desc.tileFunction = tile_fn;
    desc.threadgroupSizeMatchesTileSize = YES;
    desc.maxTotalThreadsPerThreadgroup = max_threads_per_threadgroup;
    desc.rasterSampleCount = 1;
    desc.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA32Float;

    NSError* err = nil;
    id<MTLRenderPipelineState> pso =
        [device newRenderPipelineStateWithTileDescriptor:desc options:MTLPipelineOptionNone reflection:nil error:&err];
    if (pso == nil) {
      result.error = nserror_to_string(err);
      return result;
    }
    result.pso = pso;
    result.imageblock_sample_length = [pso imageblockSampleLength];
    result.imageblock_memory_8x8 = [pso imageblockMemoryLengthForDimensions:MTLSizeMake(8, 8, 1)];
    result.imageblock_memory_16x16 = [pso imageblockMemoryLengthForDimensions:MTLSizeMake(16, 16, 1)];
    result.imageblock_memory_32x32 = [pso imageblockMemoryLengthForDimensions:MTLSizeMake(32, 32, 1)];
    return result;
  }

  result.error = "MTLTileRenderPipelineDescriptor requires macOS 11.0 or newer";
  return result;
}

id<MTLRenderPipelineState> cached_tile_execution_pipeline(id<MTLDevice> device) {
  static std::mutex mu;
  static id<MTLRenderPipelineState> pso = nil;
  static std::string error;
  std::lock_guard<std::mutex> lock(mu);
  if (pso == nil && error.empty()) {
    TilePipelineResult result = build_tile_pipeline_for_function(
        device, "v9_tile_exec_probe", @"v9_hw_tile_state_tile_execution_probe", 1024);
    pso = result.pso;
    error = result.error;
  }
  TORCH_CHECK(pso != nil, "Failed to build v9 tile execution pipeline: ", error);
  return pso;
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

torch::Tensor make_report_tensor(int64_t tile_count) {
  auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kMPS);
  torch::Tensor out = torch::empty({tile_count, 4}, opts);
  TORCH_CHECK(out.is_contiguous(), "new MPS report tensor is unexpectedly non-contiguous");
  return out;
}

NSUInteger ceil_div_u64(NSUInteger value, NSUInteger divisor) {
  return (value + divisor - 1) / divisor;
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
    @throw [NSException exceptionWithName:@"V9HWTileState"
                                   reason:@"failed to create render command encoder"
                                 userInfo:nil];
  }
  [render_encoder setRenderPipelineState:pso];
  [render_encoder setFragmentBytes:rgba.data() length:sizeof(float) * rgba.size() atIndex:0];
  [render_encoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
  [render_encoder endEncoding];
}

void encode_tile_execution_probe(
    id<MTLCommandBuffer> command_buffer,
    id<MTLRenderPipelineState> pso,
    id<MTLTexture> texture,
    id<MTLBuffer> report_buffer,
    NSUInteger report_offset,
    const V9TileExecParamsHost& params) {
  if (@available(macOS 11.0, *)) {
    MTLRenderPassDescriptor* pass = [MTLRenderPassDescriptor renderPassDescriptor];
    pass.colorAttachments[0].texture = texture;
    pass.colorAttachments[0].loadAction = MTLLoadActionClear;
    pass.colorAttachments[0].storeAction = MTLStoreActionStore;
    pass.colorAttachments[0].clearColor = MTLClearColorMake(0.03125, 0.0625, 0.125, 1.0);

    id<MTLRenderCommandEncoder> render_encoder = [command_buffer renderCommandEncoderWithDescriptor:pass];
    if (render_encoder == nil) {
      @throw [NSException exceptionWithName:@"V9HWTileState"
                                     reason:@"failed to create tile render command encoder"
                                   userInfo:nil];
    }
    [render_encoder setRenderPipelineState:pso];
    [render_encoder setTileBuffer:report_buffer offset:report_offset atIndex:0];
    [render_encoder setTileBytes:&params length:sizeof(params) atIndex:1];
    [render_encoder dispatchThreadsPerTile:MTLSizeMake(params.tile_size, params.tile_size, 1)];
    [render_encoder endEncoding];
    return;
  }

  @throw [NSException exceptionWithName:@"V9HWTileState"
                                 reason:@"dispatchThreadsPerTile requires macOS 11.0 or newer"
                               userInfo:nil];
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

py::dict run_tile_state_execution_probe_native(
    int64_t height,
    int64_t width,
    int64_t tile_size) {
  TORCH_CHECK(height > 0 && width > 0, "height and width must be positive");
  TORCH_CHECK(height <= INT32_MAX && width <= INT32_MAX, "height and width must fit int32");
  TORCH_CHECK(tile_size == 8 || tile_size == 16 || tile_size == 32,
              "tile_size must be 8, 16, or 32 for this probe; got ", tile_size);
  TORCH_CHECK(torch::mps::is_available(), "MPS is not available");

  id<MTLDevice> device = system_device();
  id<MTLRenderPipelineState> pso = cached_tile_execution_pipeline(device);

  torch::Tensor target = make_output_tensor(height, width);
  const NSUInteger tiles_x = ceil_div_u64((NSUInteger)width, (NSUInteger)tile_size);
  const NSUInteger tiles_y = ceil_div_u64((NSUInteger)height, (NSUInteger)tile_size);
  const NSUInteger tile_count = tiles_x * tiles_y;
  torch::Tensor reports = make_report_tensor((int64_t)tile_count);

  id<MTLBuffer> target_buffer = tensor_mtl_buffer(target);
  TORCH_CHECK(target_buffer != nil, "Failed to access target tensor MTLBuffer storage");
  id<MTLBuffer> report_buffer = tensor_mtl_buffer(reports);
  TORCH_CHECK(report_buffer != nil, "Failed to access report tensor MTLBuffer storage");

  const NSUInteger target_offset = (NSUInteger)(target.storage_offset() * target.element_size());
  const NSUInteger report_offset = (NSUInteger)(reports.storage_offset() * reports.element_size());
  const NSUInteger bytes_per_pixel = 4 * sizeof(float);
  const NSUInteger bytes_per_row = (NSUInteger)width * bytes_per_pixel;
  TORCH_CHECK(bytes_per_row % 256 == 0,
              "tile execution probe uses a direct buffer-backed RGBA32F render target and requires "
              "width*16 to be 256-byte aligned; got bytes_per_row=",
              bytes_per_row);

  MTLTextureDescriptor* tex_desc = [MTLTextureDescriptor
      texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA32Float
                                   width:(NSUInteger)width
                                  height:(NSUInteger)height
                               mipmapped:NO];
  tex_desc.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
  tex_desc.storageMode = [target_buffer storageMode];

  id<MTLTexture> texture =
      [target_buffer newTextureWithDescriptor:tex_desc offset:target_offset bytesPerRow:bytes_per_row];
  TORCH_CHECK(texture != nil,
              "Failed to create a render-target texture view over Torch MPS tensor storage for tile probe");

  V9TileExecParamsHost params{
      (uint32_t)width,
      (uint32_t)height,
      (uint32_t)tiles_x,
      (uint32_t)tile_size,
  };

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
        encode_tile_execution_probe(command_buffer, pso, texture, report_buffer, report_offset, params);
      } @catch (NSException* exception) {
        block_error = [exception reason];
        return;
      }
    }
  });

  TORCH_CHECK(block_error == nil, nsstring_to_string(block_error));

  py::dict out;
  out["target"] = target;
  out["tile_reports"] = reports;
  out["height"] = height;
  out["width"] = width;
  out["tile_size"] = tile_size;
  out["tiles_x"] = (unsigned long long)tiles_x;
  out["tiles_y"] = (unsigned long long)tiles_y;
  out["tile_count"] = (unsigned long long)tile_count;
  out["direct_target_bytes_per_row"] = (unsigned long long)bytes_per_row;
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
  out["tile_imageblock_memory_8x8"] = py::none();
  out["tile_imageblock_memory_16x16"] = py::none();
  out["tile_imageblock_memory_32x32"] = py::none();
  out["tile_layouts"] = py::list();
  out["tile_execution_probe_available"] = py::none();
  out["tile_execution_probe_error"] = "";
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
    py::list layout_results;
    TilePipelineResult primary_tile_result{nil, "", 0, 0, 0, 0};
    for (const TileLayoutSpec& spec : kTileLayoutSpecs) {
      TilePipelineResult result = build_tile_pipeline_for_function(
          device,
          spec.function_name,
          [NSString stringWithFormat:@"v9_hw_tile_state_%s", spec.function_name],
          256);
      const bool is_primary = std::strcmp(spec.name, "ct_stop_flags_fp32_u32x2") == 0;
      if (is_primary) {
        primary_tile_result = result;
      }

      py::dict row;
      row["name"] = spec.name;
      row["function_name"] = spec.function_name;
      row["purpose"] = spec.purpose;
      row["logical_bytes_per_pixel"] = (unsigned long long)spec.logical_bytes_per_pixel;
      row["pipeline_ready"] = result.pso != nil;
      row["pipeline_error"] = result.error;
      if (result.pso != nil) {
        row["imageblock_sample_length"] = (unsigned long long)result.imageblock_sample_length;
        row["imageblock_memory_8x8"] = (unsigned long long)result.imageblock_memory_8x8;
        row["imageblock_memory_16x16"] = (unsigned long long)result.imageblock_memory_16x16;
        row["imageblock_memory_32x32"] = (unsigned long long)result.imageblock_memory_32x32;
        row["max_total_threads_per_threadgroup"] = (unsigned long long)[result.pso maxTotalThreadsPerThreadgroup];
        row["threadgroup_size_matches_tile_size"] = (bool)[result.pso threadgroupSizeMatchesTileSize];
      } else {
        row["imageblock_sample_length"] = py::none();
        row["imageblock_memory_8x8"] = py::none();
        row["imageblock_memory_16x16"] = py::none();
        row["imageblock_memory_32x32"] = py::none();
        row["max_total_threads_per_threadgroup"] = py::none();
        row["threadgroup_size_matches_tile_size"] = py::none();
      }
      layout_results.append(row);
    }
    out["tile_layouts"] = layout_results;
    out["tile_pipeline_ready"] = primary_tile_result.pso != nil;
    out["tile_pipeline_error"] = primary_tile_result.error;
    if (primary_tile_result.pso != nil) {
      out["tile_imageblock_sample_length"] = (unsigned long long)primary_tile_result.imageblock_sample_length;
      out["tile_imageblock_memory_8x8"] = (unsigned long long)primary_tile_result.imageblock_memory_8x8;
      out["tile_imageblock_memory_16x16"] = (unsigned long long)primary_tile_result.imageblock_memory_16x16;
      out["tile_imageblock_memory_32x32"] = (unsigned long long)primary_tile_result.imageblock_memory_32x32;
    }

    TilePipelineResult exec_result = build_tile_pipeline_for_function(
        device, "v9_tile_exec_probe", @"v9_hw_tile_state_tile_execution_probe", 256);
    out["tile_execution_probe_available"] = exec_result.pso != nil;
    out["tile_execution_probe_error"] = exec_result.error;

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
      "run_tile_state_execution_probe",
      &run_tile_state_execution_probe_native,
      py::arg("height") = 32,
      py::arg("width") = 32,
      py::arg("tile_size") = 32,
      "Dispatch a tile shader with imageblock C/T/stop state inside a render pass on a direct Torch MPS target.");
}

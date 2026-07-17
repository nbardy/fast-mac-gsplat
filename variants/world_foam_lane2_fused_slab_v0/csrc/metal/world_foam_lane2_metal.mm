#import <Foundation/Foundation.h>

#include <ATen/ATen.h>
#include <ATen/native/mps/MetalShaderLibrary.h>
#include <torch/extension.h>
#include <torch/mps.h>

#include <algorithm>
#include <memory>
#include <mutex>
#include <string>

namespace world_foam_lane2_fused_slab {
namespace {

using at::native::mps::DynamicMetalShaderLibrary;
using at::native::mps::MetalKernelFunction;

std::string load_shader_source() {
  NSString* metalPath = [[NSString stringWithUTF8String:__FILE__] stringByDeletingLastPathComponent];
  NSString* tensorPath = [metalPath stringByAppendingPathComponent:@"world_foam_lane2_power_boundary_tensor.metal"];
  NSString* replayPath = [metalPath stringByAppendingPathComponent:@"world_foam_lane2_shared_replay_tensor.metal"];
  NSError* err = nil;
  NSString* tensorSrc = [NSString stringWithContentsOfFile:tensorPath encoding:NSUTF8StringEncoding error:&err];
  TORCH_CHECK(
      tensorSrc != nil,
      "Failed to read world_foam_lane2_power_boundary_tensor.metal: ",
      err.localizedDescription.UTF8String);
  err = nil;
  NSString* replaySrc = [NSString stringWithContentsOfFile:replayPath encoding:NSUTF8StringEncoding error:&err];
  TORCH_CHECK(
      replaySrc != nil,
      "Failed to read world_foam_lane2_shared_replay_tensor.metal: ",
      err.localizedDescription.UTF8String);
  return std::string([tensorSrc UTF8String]) + "\n" + std::string([replaySrc UTF8String]);
}

struct MetalKernels {
  std::shared_ptr<MetalKernelFunction> count_power_boundary_events;
  std::shared_ptr<MetalKernelFunction> shared_signal_replay;
  std::shared_ptr<MetalKernelFunction> shared_rgb_replay;
  std::shared_ptr<MetalKernelFunction> shared_rgba_depth_replay;
  std::shared_ptr<MetalKernelFunction> shared_rgba_depth_vjp;
  std::shared_ptr<MetalKernelFunction> realray_rgba_depth_replay;
  std::shared_ptr<MetalKernelFunction> shared_realray_rgba_depth_replay;
  std::shared_ptr<MetalKernelFunction> shared_realray_rgba_depth_vjp;
  std::shared_ptr<MetalKernelFunction> shared_realray_rgba_depth_vjp_reduce;
  std::shared_ptr<MetalKernelFunction> shared_realray_rgba_depth_vjp_partial_reduce;
  std::shared_ptr<MetalKernelFunction> shared_realray_rgba_depth_vjp_partial_reduce_csr;
  std::shared_ptr<MetalKernelFunction> fused_slab_affine_realray_rgba_depth_replay;
  std::shared_ptr<MetalKernelFunction> fused_slab_affine_coeff_realray_rgba_depth_replay;
  std::shared_ptr<MetalKernelFunction> fused_slab_affine_coeff16_realray_rgba_depth_replay;
  std::shared_ptr<MetalKernelFunction> fused_slab_affine_num32_den16_realray_rgba_depth_replay;
  std::shared_ptr<MetalKernelFunction> fused_slab_affine_num32_den16_ownerupdate_realray_rgba_depth_replay;
  std::shared_ptr<MetalKernelFunction> fused_slab_affine_num32_den16_vjp_partial_reduce;
  std::shared_ptr<MetalKernelFunction> fused_slab_affine_num32_den16_vjp_finalize_reduce;
  std::shared_ptr<MetalKernelFunction> clear_site_rgba_grad;
  std::shared_ptr<MetalKernelFunction> clear_endpoint_loss_site_rgba_grad;
  std::shared_ptr<MetalKernelFunction> clear_affine_loss_site_rgba_grad;
  std::shared_ptr<MetalKernelFunction> fused_slab_affine_num32_den16_vjp_direct_atomic;
  std::shared_ptr<MetalKernelFunction> fused_slab_affine_num32_den16_vjp_direct_atomic_grad_only;
  std::shared_ptr<MetalKernelFunction> fused_slab_affine_num32_den16_vjp_direct_atomic_grad_only_ownerupdate;
  std::shared_ptr<MetalKernelFunction> fused_slab_affine_num32_den16_vjp_direct_atomic_rgb_only;
  std::shared_ptr<MetalKernelFunction> fused_slab_affine_num32_den16_mse_vjp_direct_atomic_rgb_only;
  std::shared_ptr<MetalKernelFunction> fused_slab_affine_num32_den16_mse_vjp_direct_atomic_track_rgb_only;
  std::shared_ptr<MetalKernelFunction> fused_slab_affine_coeff16_mse_vjp_direct_atomic_rgb_only;
  std::shared_ptr<MetalKernelFunction> fused_slab_affine_coeff16_cap224_mse_vjp_direct_atomic_rgb_only;
  std::shared_ptr<MetalKernelFunction> fused_slab_affine_coeff16_densitymask_mse_vjp_direct_atomic_rgb_only;
  std::shared_ptr<MetalKernelFunction> fused_slab_affine_coeff16_sortnet_mse_vjp_direct_atomic_rgb_only;
  std::shared_ptr<MetalKernelFunction> fused_slab_affine_coeff16_mse_vjp_direct_atomic_sitecache_rgb_only;
  std::shared_ptr<MetalKernelFunction> fused_slab_affine_coeff16_mse_vjp_direct_atomic_sample_reduce_rgb_only;
  std::shared_ptr<MetalKernelFunction> fused_slab_affine_coeff16_mse_vjp_direct_atomic_framegroup16_cached_rgb_only;
  std::shared_ptr<MetalKernelFunction> fused_slab_affine_coeff16_ownerupdate_mse_vjp_direct_atomic_rgb_only;
  std::shared_ptr<MetalKernelFunction> fused_slab_affine_coeff16_ownerupdate_i16_mse_vjp_direct_atomic_rgb_only;
  std::shared_ptr<MetalKernelFunction> fused_slab_affine_coeff16_ownerkeep_i16_mse_vjp_direct_atomic_rgb_only;
  std::shared_ptr<MetalKernelFunction> fused_slab_affine_coeff16_ownerkeep_mse_vjp_direct_atomic_rgb_only;
  std::shared_ptr<MetalKernelFunction> fused_slab_affine_coeff16_mse_vjp_direct_atomic_track_rgb_only;
  std::shared_ptr<MetalKernelFunction> fused_slab_affine_num32_den16_vjp_direct_atomic_track;
  std::shared_ptr<MetalKernelFunction> segment_tape_rgba_depth_replay;
  std::shared_ptr<MetalKernelFunction> segment_tape_vjp_direct_atomic_grad_only;
  std::shared_ptr<MetalKernelFunction> segment_tape_vjp_direct_atomic_track;
  std::shared_ptr<MetalKernelFunction> segment_tape_mse_vjp_direct_atomic_rgb_only;
  std::shared_ptr<MetalKernelFunction> endpoint_run_rgba_depth_replay;
  std::shared_ptr<MetalKernelFunction> endpoint_run_vjp_direct_atomic_grad_only;
  std::shared_ptr<MetalKernelFunction> endpoint_run_mse_vjp_direct_atomic_rgb_only;
  std::shared_ptr<MetalKernelFunction> endpoint_delta_replace_rgba_depth_replay;
  std::shared_ptr<MetalKernelFunction> endpoint_delta_replace_vjp_direct_atomic_grad_only;
  std::shared_ptr<MetalKernelFunction> endpoint_record_delta_replace_rgba_depth_replay;
  std::shared_ptr<MetalKernelFunction> endpoint_record_delta_replace_vjp_direct_atomic_grad_only;
  std::shared_ptr<MetalKernelFunction> endpoint_record_delta_replace_coeff16_mse_vjp_direct_atomic_rgb_only;
  std::shared_ptr<MetalKernelFunction> endpoint_record_delta_replace_coeff16_i16x3_mse_vjp_direct_atomic_rgb_only;
  std::shared_ptr<MetalKernelFunction> endpoint_record_delta_replace_coeff16_i16x3_framegroup16_mse_vjp_direct_atomic_rgb_only;
  std::shared_ptr<MetalKernelFunction> endpoint_record_delta_replace_coeff16_i16x3_framegroup16_ownerreduce_mse_vjp_direct_atomic_rgb_only;
  std::shared_ptr<MetalKernelFunction> endpoint_record_delta_replace_coeff16_i16cols_framegroup16_mse_vjp_direct_atomic_rgb_only;
  std::shared_ptr<MetalKernelFunction> endpoint_record_delta_replace_coeff16_i16x3_framegroup64_mse_vjp_direct_atomic_rgb_only;
  std::shared_ptr<MetalKernelFunction> endpoint_record_delta_replace_coeff16_packed_mse_vjp_direct_atomic_rgb_only;
  std::shared_ptr<MetalKernelFunction> endpoint_record_delta_replace_coeff16_packed_framegroup16_mse_vjp_direct_atomic_rgb_only;
  std::shared_ptr<MetalKernelFunction> endpoint_record_delta_replace_coeff16_packed_framegroup16_reduce32_mse_vjp_direct_atomic_rgb_only;
  std::shared_ptr<MetalKernelFunction> endpoint_record_delta_replace_coeff16_packed_framegroup16_rowselect32_mse_vjp_direct_atomic_rgb_only;
  std::shared_ptr<MetalKernelFunction> endpoint_record_delta_replace_coeff16_packed_framegroup16_rowdesc_mse_vjp_direct_atomic_rgb_only;
  std::shared_ptr<MetalKernelFunction> endpoint_record_delta_replace_coeff16_packed_framegroup16_rowdesc32_mse_vjp_direct_atomic_rgb_only;
  std::shared_ptr<MetalKernelFunction> endpoint_record_delta_replace_coeff16_packed_framegroup16_recompute_mse_vjp_direct_atomic_rgb_only;
  std::shared_ptr<MetalKernelFunction> endpoint_record_delta_replace_factorized_packed_framegroup16_recompute_mse_vjp_direct_atomic_rgb_only;
  std::shared_ptr<MetalKernelFunction> endpoint_record_delta_replace_factorized_frameselect_recompute_mse_vjp_direct_atomic_rgb_only;
  std::shared_ptr<MetalKernelFunction> endpoint_record_delta_replace_factorized_framebitmask_recompute_mse_vjp_direct_atomic_rgb_only;
  std::shared_ptr<MetalKernelFunction> endpoint_record_delta_replace_coeff16_packed_framegroup16_smallrun16_mse_vjp_direct_atomic_rgb_only;
  std::shared_ptr<MetalKernelFunction> endpoint_record_delta_replace_coeff16_packed_framegroup16_materialized_mse_vjp_direct_atomic_rgb_only;
  std::shared_ptr<MetalKernelFunction> endpoint_record_delta_replace_coeff16_i16x3_framegroup16_materialized_mse_vjp_direct_atomic_rgb_only;
  std::shared_ptr<MetalKernelFunction> endpoint_record_delta_replace_coeff16_i16x4_framegroup16_mse_vjp_direct_atomic_rgb_only;
  std::shared_ptr<MetalKernelFunction> endpoint_record_delta_replace_coeff16_i16x4_mse_vjp_direct_atomic_rgb_only;
  std::shared_ptr<MetalKernelFunction> endpoint_record_edit_rgba_depth_replay;
  std::shared_ptr<MetalKernelFunction> endpoint_record_edit_block4_rgba_depth_replay;
  std::shared_ptr<MetalKernelFunction> endpoint_record_edit_block_coeff_rgba_depth_replay;
  std::shared_ptr<MetalKernelFunction> endpoint_record_edit_block_coeff_rgb_replay;
  std::shared_ptr<MetalKernelFunction> endpoint_record_edit_block_coeff16_rgba_depth_replay;
  std::shared_ptr<MetalKernelFunction> endpoint_record_edit_rgba_depth_replay_trackloop;
  std::shared_ptr<MetalKernelFunction> endpoint_record_edit_rgba_depth_replay_framegroup16;
  std::shared_ptr<MetalKernelFunction> endpoint_record_edit_vjp_direct_atomic_grad_only;
  std::shared_ptr<MetalKernelFunction> endpoint_record_edit_vjp_direct_atomic_rgb_only;
  std::shared_ptr<MetalKernelFunction> endpoint_record_edit_mse_vjp_direct_atomic_rgb_only;
  std::shared_ptr<MetalKernelFunction> endpoint_record_edit_coeff16_mse_vjp_direct_atomic_rgb_only;
  std::shared_ptr<MetalKernelFunction> endpoint_record_edit_block4_vjp_direct_atomic_rgb_only;
  std::shared_ptr<MetalKernelFunction> endpoint_record_edit_block_coeff_vjp_direct_atomic_rgb_only;
  std::shared_ptr<MetalKernelFunction> endpoint_record_edit_block_coeff_mse_vjp_direct_atomic_rgb_only;
  std::shared_ptr<MetalKernelFunction> endpoint_record_edit_block_coeff16_mse_vjp_direct_atomic_rgb_only;
  std::shared_ptr<MetalKernelFunction> endpoint_record_edit_block_coeff16_packed_mse_vjp_direct_atomic_rgb_only;
  std::shared_ptr<MetalKernelFunction> endpoint_record_edit_block_coeff16_i16_mse_vjp_direct_atomic_rgb_only;
  std::shared_ptr<MetalKernelFunction> endpoint_record_edit_block_coeff16_i16x3_mse_vjp_direct_atomic_rgb_only;
  std::shared_ptr<MetalKernelFunction> endpoint_record_edit_block_coeff16_vjp_direct_atomic_rgb_only;
  std::shared_ptr<MetalKernelFunction> shared_realray_rgba_depth_vjp_finalize_reduce;
};

MetalKernels& kernels() {
  static std::once_flag once;
  static std::unique_ptr<DynamicMetalShaderLibrary> lib;
  static MetalKernels out;
  std::call_once(once, []() {
    lib = std::make_unique<DynamicMetalShaderLibrary>(load_shader_source());
    out.count_power_boundary_events = lib->getKernelFunction("wf2_count_power_boundary_events_tensor");
    out.shared_signal_replay = lib->getKernelFunction("wf2_shared_signal_replay_tensor");
    out.shared_rgb_replay = lib->getKernelFunction("wf2_shared_rgb_replay_tensor");
    out.shared_rgba_depth_replay = lib->getKernelFunction("wf2_shared_rgba_depth_replay_tensor");
    out.shared_rgba_depth_vjp = lib->getKernelFunction("wf2_shared_rgba_depth_vjp_tensor");
    out.realray_rgba_depth_replay = lib->getKernelFunction("wf2_realray_rgba_depth_replay_tensor");
    out.shared_realray_rgba_depth_replay = lib->getKernelFunction("wf2_shared_realray_rgba_depth_replay_tensor");
    out.shared_realray_rgba_depth_vjp = lib->getKernelFunction("wf2_shared_realray_rgba_depth_vjp_tensor");
    out.shared_realray_rgba_depth_vjp_reduce = lib->getKernelFunction("wf2_shared_realray_rgba_depth_vjp_reduce_tensor");
    out.shared_realray_rgba_depth_vjp_partial_reduce =
        lib->getKernelFunction("wf2_shared_realray_rgba_depth_vjp_partial_reduce_tensor");
    out.shared_realray_rgba_depth_vjp_partial_reduce_csr =
        lib->getKernelFunction("wf2_shared_realray_rgba_depth_vjp_partial_reduce_csr_tensor");
    out.fused_slab_affine_realray_rgba_depth_replay =
        lib->getKernelFunction("wf2_fused_slab_affine_realray_rgba_depth_replay_tensor");
    out.fused_slab_affine_coeff_realray_rgba_depth_replay =
        lib->getKernelFunction("wf2_fused_slab_affine_coeff_realray_rgba_depth_replay_tensor");
    out.fused_slab_affine_coeff16_realray_rgba_depth_replay =
        lib->getKernelFunction("wf2_fused_slab_affine_coeff16_realray_rgba_depth_replay_tensor");
    out.fused_slab_affine_num32_den16_realray_rgba_depth_replay =
        lib->getKernelFunction("wf2_fused_slab_affine_num32_den16_realray_rgba_depth_replay_tensor");
    out.fused_slab_affine_num32_den16_ownerupdate_realray_rgba_depth_replay =
        lib->getKernelFunction("wf2_fused_slab_affine_num32_den16_ownerupdate_realray_rgba_depth_replay_tensor");
    out.fused_slab_affine_num32_den16_vjp_partial_reduce =
        lib->getKernelFunction("wf2_fused_slab_affine_num32_den16_vjp_partial_reduce_tensor");
    out.fused_slab_affine_num32_den16_vjp_finalize_reduce =
        lib->getKernelFunction("wf2_fused_slab_affine_num32_den16_vjp_finalize_reduce_tensor");
    out.clear_site_rgba_grad = lib->getKernelFunction("wf2_clear_site_rgba_grad_tensor");
    out.clear_endpoint_loss_site_rgba_grad =
        lib->getKernelFunction("wf2_clear_endpoint_loss_site_rgba_grad_tensor");
    out.clear_affine_loss_site_rgba_grad =
        lib->getKernelFunction("wf2_clear_affine_loss_site_rgba_grad_tensor");
    out.fused_slab_affine_num32_den16_vjp_direct_atomic =
        lib->getKernelFunction("wf2_fused_slab_affine_num32_den16_vjp_direct_atomic_tensor");
    out.fused_slab_affine_num32_den16_vjp_direct_atomic_grad_only =
        lib->getKernelFunction("wf2_fused_slab_affine_num32_den16_vjp_direct_atomic_grad_only_tensor");
    out.fused_slab_affine_num32_den16_vjp_direct_atomic_grad_only_ownerupdate =
        lib->getKernelFunction(
            "wf2_fused_slab_affine_num32_den16_vjp_direct_atomic_grad_only_ownerupdate_tensor");
    out.fused_slab_affine_num32_den16_vjp_direct_atomic_rgb_only =
        lib->getKernelFunction("wf2_fused_slab_affine_num32_den16_vjp_direct_atomic_rgb_only_tensor");
    out.fused_slab_affine_num32_den16_mse_vjp_direct_atomic_rgb_only =
        lib->getKernelFunction("wf2_fused_slab_affine_num32_den16_mse_vjp_direct_atomic_rgb_only_tensor");
    out.fused_slab_affine_num32_den16_mse_vjp_direct_atomic_track_rgb_only =
        lib->getKernelFunction("wf2_fused_slab_affine_num32_den16_mse_vjp_direct_atomic_track_rgb_only_tensor");
    out.fused_slab_affine_coeff16_mse_vjp_direct_atomic_rgb_only =
        lib->getKernelFunction("wf2_fused_slab_affine_coeff16_mse_vjp_direct_atomic_rgb_only_tensor");
    out.fused_slab_affine_coeff16_cap224_mse_vjp_direct_atomic_rgb_only =
        lib->getKernelFunction("wf2_fused_slab_affine_coeff16_cap224_mse_vjp_direct_atomic_rgb_only_tensor");
    out.fused_slab_affine_coeff16_densitymask_mse_vjp_direct_atomic_rgb_only =
        lib->getKernelFunction("wf2_fused_slab_affine_coeff16_densitymask_mse_vjp_direct_atomic_rgb_only_tensor");
    out.fused_slab_affine_coeff16_sortnet_mse_vjp_direct_atomic_rgb_only =
        lib->getKernelFunction("wf2_fused_slab_affine_coeff16_sortnet_mse_vjp_direct_atomic_rgb_only_tensor");
    out.fused_slab_affine_coeff16_mse_vjp_direct_atomic_sitecache_rgb_only =
        lib->getKernelFunction("wf2_fused_slab_affine_coeff16_mse_vjp_direct_atomic_sitecache_rgb_only_tensor");
    out.fused_slab_affine_coeff16_mse_vjp_direct_atomic_sample_reduce_rgb_only = lib->getKernelFunction(
        "wf2_fused_slab_affine_coeff16_mse_vjp_direct_atomic_sample_reduce_rgb_only_tensor");
    out.fused_slab_affine_coeff16_mse_vjp_direct_atomic_framegroup16_cached_rgb_only =
        lib->getKernelFunction(
            "wf2_fused_slab_affine_coeff16_mse_vjp_direct_atomic_framegroup16_cached_rgb_only_tensor");
    out.fused_slab_affine_coeff16_ownerupdate_mse_vjp_direct_atomic_rgb_only =
        lib->getKernelFunction("wf2_fused_slab_affine_coeff16_ownerupdate_mse_vjp_direct_atomic_rgb_only_tensor");
    out.fused_slab_affine_coeff16_ownerupdate_i16_mse_vjp_direct_atomic_rgb_only =
        lib->getKernelFunction(
            "wf2_fused_slab_affine_coeff16_ownerupdate_i16_mse_vjp_direct_atomic_rgb_only_tensor");
    out.fused_slab_affine_coeff16_ownerkeep_i16_mse_vjp_direct_atomic_rgb_only =
        lib->getKernelFunction("wf2_fused_slab_affine_coeff16_ownerkeep_i16_mse_vjp_direct_atomic_rgb_only_tensor");
    out.fused_slab_affine_coeff16_ownerkeep_mse_vjp_direct_atomic_rgb_only =
        lib->getKernelFunction("wf2_fused_slab_affine_coeff16_ownerkeep_mse_vjp_direct_atomic_rgb_only_tensor");
    out.fused_slab_affine_coeff16_mse_vjp_direct_atomic_track_rgb_only =
        lib->getKernelFunction("wf2_fused_slab_affine_coeff16_mse_vjp_direct_atomic_track_rgb_only_tensor");
    out.fused_slab_affine_num32_den16_vjp_direct_atomic_track =
        lib->getKernelFunction("wf2_fused_slab_affine_num32_den16_vjp_direct_atomic_track_tensor");
    out.segment_tape_rgba_depth_replay =
        lib->getKernelFunction("wf2_segment_tape_rgba_depth_replay_tensor");
    out.segment_tape_vjp_direct_atomic_grad_only =
        lib->getKernelFunction("wf2_segment_tape_vjp_direct_atomic_grad_only_tensor");
    out.segment_tape_vjp_direct_atomic_track =
        lib->getKernelFunction("wf2_segment_tape_vjp_direct_atomic_track_tensor");
    out.segment_tape_mse_vjp_direct_atomic_rgb_only =
        lib->getKernelFunction("wf2_segment_tape_mse_vjp_direct_atomic_rgb_only_tensor");
    out.endpoint_run_rgba_depth_replay =
        lib->getKernelFunction("wf2_endpoint_run_rgba_depth_replay_tensor");
    out.endpoint_run_vjp_direct_atomic_grad_only =
        lib->getKernelFunction("wf2_endpoint_run_vjp_direct_atomic_grad_only_tensor");
    out.endpoint_run_mse_vjp_direct_atomic_rgb_only =
        lib->getKernelFunction("wf2_endpoint_run_mse_vjp_direct_atomic_rgb_only_tensor");
    out.endpoint_delta_replace_rgba_depth_replay =
        lib->getKernelFunction("wf2_endpoint_delta_replace_rgba_depth_replay_tensor");
    out.endpoint_delta_replace_vjp_direct_atomic_grad_only =
        lib->getKernelFunction("wf2_endpoint_delta_replace_vjp_direct_atomic_grad_only_tensor");
    out.endpoint_record_delta_replace_rgba_depth_replay =
        lib->getKernelFunction("wf2_endpoint_record_delta_replace_rgba_depth_replay_tensor");
    out.endpoint_record_delta_replace_vjp_direct_atomic_grad_only =
        lib->getKernelFunction("wf2_endpoint_record_delta_replace_vjp_direct_atomic_grad_only_tensor");
    out.endpoint_record_delta_replace_coeff16_mse_vjp_direct_atomic_rgb_only =
        lib->getKernelFunction("wf2_endpoint_record_delta_replace_coeff16_mse_vjp_direct_atomic_rgb_only_tensor");
    out.endpoint_record_delta_replace_coeff16_i16x3_mse_vjp_direct_atomic_rgb_only =
        lib->getKernelFunction("wf2_endpoint_record_delta_replace_coeff16_i16x3_mse_vjp_direct_atomic_rgb_only_tensor");
    out.endpoint_record_delta_replace_coeff16_i16x3_framegroup16_mse_vjp_direct_atomic_rgb_only = lib->getKernelFunction(
        "wf2_endpoint_record_delta_replace_coeff16_i16x3_framegroup16_mse_vjp_direct_atomic_rgb_only_tensor");
    out.endpoint_record_delta_replace_coeff16_i16x3_framegroup16_ownerreduce_mse_vjp_direct_atomic_rgb_only =
        lib->getKernelFunction(
            "wf2_endpoint_record_delta_replace_coeff16_i16x3_framegroup16_ownerreduce_mse_vjp_direct_atomic_rgb_only_tensor");
    out.endpoint_record_delta_replace_coeff16_i16cols_framegroup16_mse_vjp_direct_atomic_rgb_only =
        lib->getKernelFunction(
            "wf2_endpoint_record_delta_replace_coeff16_i16cols_framegroup16_mse_vjp_direct_atomic_rgb_only_tensor");
    out.endpoint_record_delta_replace_coeff16_i16x3_framegroup64_mse_vjp_direct_atomic_rgb_only =
        lib->getKernelFunction(
            "wf2_endpoint_record_delta_replace_coeff16_i16x3_framegroup64_mse_vjp_direct_atomic_rgb_only_tensor");
    out.endpoint_record_delta_replace_coeff16_packed_mse_vjp_direct_atomic_rgb_only =
        lib->getKernelFunction("wf2_endpoint_record_delta_replace_coeff16_packed_mse_vjp_direct_atomic_rgb_only_tensor");
    out.endpoint_record_delta_replace_coeff16_packed_framegroup16_mse_vjp_direct_atomic_rgb_only = lib->getKernelFunction(
        "wf2_endpoint_record_delta_replace_coeff16_packed_framegroup16_mse_vjp_direct_atomic_rgb_only_tensor");
    out.endpoint_record_delta_replace_coeff16_packed_framegroup16_reduce32_mse_vjp_direct_atomic_rgb_only =
        lib->getKernelFunction(
            "wf2_endpoint_record_delta_replace_coeff16_packed_framegroup16_reduce32_mse_vjp_direct_atomic_rgb_only_tensor");
    out.endpoint_record_delta_replace_coeff16_packed_framegroup16_rowselect32_mse_vjp_direct_atomic_rgb_only =
        lib->getKernelFunction(
            "wf2_endpoint_record_delta_replace_coeff16_packed_framegroup16_rowselect32_mse_vjp_direct_atomic_rgb_only_tensor");
    out.endpoint_record_delta_replace_coeff16_packed_framegroup16_rowdesc_mse_vjp_direct_atomic_rgb_only =
        lib->getKernelFunction(
            "wf2_endpoint_record_delta_replace_coeff16_packed_framegroup16_rowdesc_mse_vjp_direct_atomic_rgb_only_tensor");
    out.endpoint_record_delta_replace_coeff16_packed_framegroup16_rowdesc32_mse_vjp_direct_atomic_rgb_only =
        lib->getKernelFunction(
            "wf2_endpoint_record_delta_replace_coeff16_packed_framegroup16_rowdesc32_mse_vjp_direct_atomic_rgb_only_tensor");
    out.endpoint_record_delta_replace_coeff16_packed_framegroup16_recompute_mse_vjp_direct_atomic_rgb_only =
        lib->getKernelFunction(
            "wf2_endpoint_record_delta_replace_coeff16_packed_framegroup16_recompute_mse_vjp_direct_atomic_rgb_only_tensor");
    out.endpoint_record_delta_replace_factorized_packed_framegroup16_recompute_mse_vjp_direct_atomic_rgb_only =
        lib->getKernelFunction(
            "wf2_endpoint_record_delta_replace_factorized_packed_framegroup16_recompute_mse_vjp_direct_atomic_rgb_only_tensor");
    out.endpoint_record_delta_replace_factorized_frameselect_recompute_mse_vjp_direct_atomic_rgb_only =
        lib->getKernelFunction(
            "wf2_endpoint_record_delta_replace_factorized_frameselect_recompute_mse_vjp_direct_atomic_rgb_only_tensor");
    out.endpoint_record_delta_replace_factorized_framebitmask_recompute_mse_vjp_direct_atomic_rgb_only =
        lib->getKernelFunction(
            "wf2_endpoint_record_delta_replace_factorized_framebitmask_recompute_mse_vjp_direct_atomic_rgb_only_tensor");
    out.endpoint_record_delta_replace_coeff16_packed_framegroup16_smallrun16_mse_vjp_direct_atomic_rgb_only =
        lib->getKernelFunction(
            "wf2_endpoint_record_delta_replace_coeff16_packed_framegroup16_smallrun16_mse_vjp_direct_atomic_rgb_only_tensor");
    out.endpoint_record_delta_replace_coeff16_packed_framegroup16_materialized_mse_vjp_direct_atomic_rgb_only =
        lib->getKernelFunction(
            "wf2_endpoint_record_delta_replace_coeff16_packed_framegroup16_materialized_mse_vjp_direct_atomic_rgb_only_tensor");
    out.endpoint_record_delta_replace_coeff16_i16x3_framegroup16_materialized_mse_vjp_direct_atomic_rgb_only =
        lib->getKernelFunction(
            "wf2_endpoint_record_delta_replace_coeff16_i16x3_framegroup16_materialized_mse_vjp_direct_atomic_rgb_only_tensor");
    out.endpoint_record_delta_replace_coeff16_i16x4_framegroup16_mse_vjp_direct_atomic_rgb_only = lib->getKernelFunction(
        "wf2_endpoint_record_delta_replace_coeff16_i16x4_framegroup16_mse_vjp_direct_atomic_rgb_only_tensor");
    out.endpoint_record_delta_replace_coeff16_i16x4_mse_vjp_direct_atomic_rgb_only =
        lib->getKernelFunction("wf2_endpoint_record_delta_replace_coeff16_i16x4_mse_vjp_direct_atomic_rgb_only_tensor");
    out.endpoint_record_edit_rgba_depth_replay =
        lib->getKernelFunction("wf2_endpoint_record_edit_rgba_depth_replay_tensor");
    out.endpoint_record_edit_block4_rgba_depth_replay =
        lib->getKernelFunction("wf2_endpoint_record_edit_block4_rgba_depth_replay_tensor");
    out.endpoint_record_edit_block_coeff_rgba_depth_replay =
        lib->getKernelFunction("wf2_endpoint_record_edit_block_coeff_rgba_depth_replay_tensor");
    out.endpoint_record_edit_block_coeff_rgb_replay =
        lib->getKernelFunction("wf2_endpoint_record_edit_block_coeff_rgb_replay_tensor");
    out.endpoint_record_edit_block_coeff16_rgba_depth_replay =
        lib->getKernelFunction("wf2_endpoint_record_edit_block_coeff16_rgba_depth_replay_tensor");
    out.endpoint_record_edit_rgba_depth_replay_trackloop =
        lib->getKernelFunction("wf2_endpoint_record_edit_rgba_depth_replay_trackloop_tensor");
    out.endpoint_record_edit_rgba_depth_replay_framegroup16 =
        lib->getKernelFunction("wf2_endpoint_record_edit_rgba_depth_replay_framegroup16_tensor");
    out.endpoint_record_edit_vjp_direct_atomic_grad_only =
        lib->getKernelFunction("wf2_endpoint_record_edit_vjp_direct_atomic_grad_only_tensor");
    out.endpoint_record_edit_vjp_direct_atomic_rgb_only =
        lib->getKernelFunction("wf2_endpoint_record_edit_vjp_direct_atomic_rgb_only_tensor");
    out.endpoint_record_edit_mse_vjp_direct_atomic_rgb_only =
        lib->getKernelFunction("wf2_endpoint_record_edit_mse_vjp_direct_atomic_rgb_only_tensor");
    out.endpoint_record_edit_coeff16_mse_vjp_direct_atomic_rgb_only =
        lib->getKernelFunction("wf2_endpoint_record_edit_coeff16_mse_vjp_direct_atomic_rgb_only_tensor");
    out.endpoint_record_edit_block4_vjp_direct_atomic_rgb_only =
        lib->getKernelFunction("wf2_endpoint_record_edit_block4_vjp_direct_atomic_rgb_only_tensor");
    out.endpoint_record_edit_block_coeff_vjp_direct_atomic_rgb_only =
        lib->getKernelFunction("wf2_endpoint_record_edit_block_coeff_vjp_direct_atomic_rgb_only_tensor");
    out.endpoint_record_edit_block_coeff_mse_vjp_direct_atomic_rgb_only =
        lib->getKernelFunction("wf2_endpoint_record_edit_block_coeff_mse_vjp_direct_atomic_rgb_only_tensor");
    out.endpoint_record_edit_block_coeff16_mse_vjp_direct_atomic_rgb_only =
        lib->getKernelFunction("wf2_endpoint_record_edit_block_coeff16_mse_vjp_direct_atomic_rgb_only_tensor");
    out.endpoint_record_edit_block_coeff16_packed_mse_vjp_direct_atomic_rgb_only =
        lib->getKernelFunction("wf2_endpoint_record_edit_block_coeff16_packed_mse_vjp_direct_atomic_rgb_only_tensor");
    out.endpoint_record_edit_block_coeff16_i16_mse_vjp_direct_atomic_rgb_only =
        lib->getKernelFunction("wf2_endpoint_record_edit_block_coeff16_i16_mse_vjp_direct_atomic_rgb_only_tensor");
    out.endpoint_record_edit_block_coeff16_i16x3_mse_vjp_direct_atomic_rgb_only =
        lib->getKernelFunction("wf2_endpoint_record_edit_block_coeff16_i16x3_mse_vjp_direct_atomic_rgb_only_tensor");
    out.endpoint_record_edit_block_coeff16_vjp_direct_atomic_rgb_only =
        lib->getKernelFunction("wf2_endpoint_record_edit_block_coeff16_vjp_direct_atomic_rgb_only_tensor");
    out.shared_realray_rgba_depth_vjp_finalize_reduce =
        lib->getKernelFunction("wf2_shared_realray_rgba_depth_vjp_finalize_reduce_tensor");
  });
  return out;
}

template <typename Fn>
void launch(std::shared_ptr<MetalKernelFunction> fn, Fn&& body) {
  fn->runCommandBlock([&]() {
    fn->startEncoding();
    body(*fn);
  });
}

void check_float_mps_2d(const torch::Tensor& t, const char* name, int64_t cols) {
  TORCH_CHECK(t.device().is_mps(), name, " must be on MPS");
  TORCH_CHECK(t.scalar_type() == torch::kFloat32, name, " must be float32");
  TORCH_CHECK(t.dim() == 2 && t.size(1) == cols, name, " must have shape [N,", cols, "]");
  TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

void check_half_mps_2d(const torch::Tensor& t, const char* name, int64_t cols) {
  TORCH_CHECK(t.device().is_mps(), name, " must be on MPS");
  TORCH_CHECK(t.scalar_type() == torch::kFloat16, name, " must be float16");
  TORCH_CHECK(t.dim() == 2 && t.size(1) == cols, name, " must have shape [N,", cols, "]");
  TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

void check_i32_mps_1d(const torch::Tensor& t, const char* name, int64_t count) {
  TORCH_CHECK(t.device().is_mps(), name, " must be on MPS");
  TORCH_CHECK(t.scalar_type() == torch::kInt32, name, " must be int32");
  TORCH_CHECK(t.dim() == 1 && t.size(0) == count, name, " must have shape [", count, "]");
  TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

void check_i32_mps_1d_any(const torch::Tensor& t, const char* name) {
  TORCH_CHECK(t.device().is_mps(), name, " must be on MPS");
  TORCH_CHECK(t.scalar_type() == torch::kInt32, name, " must be int32");
  TORCH_CHECK(t.dim() == 1, name, " must be rank-1");
  TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

void check_i16_mps_1d_any(const torch::Tensor& t, const char* name) {
  TORCH_CHECK(t.device().is_mps(), name, " must be on MPS");
  TORCH_CHECK(t.scalar_type() == torch::kInt16, name, " must be int16");
  TORCH_CHECK(t.dim() == 1, name, " must be rank-1");
  TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

void check_i16_mps_2d(const torch::Tensor& t, const char* name, int64_t cols) {
  TORCH_CHECK(t.device().is_mps(), name, " must be on MPS");
  TORCH_CHECK(t.scalar_type() == torch::kInt16, name, " must be int16");
  TORCH_CHECK(t.dim() == 2 && t.size(1) == cols, name, " must have shape [N,", cols, "]");
  TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

void check_i32_mps_2d(const torch::Tensor& t, const char* name, int64_t cols) {
  TORCH_CHECK(t.device().is_mps(), name, " must be on MPS");
  TORCH_CHECK(t.scalar_type() == torch::kInt32, name, " must be int32");
  TORCH_CHECK(t.dim() == 2 && t.size(1) == cols, name, " must have shape [N,", cols, "]");
  TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

void check_float_mps_1d_any(const torch::Tensor& t, const char* name) {
  TORCH_CHECK(t.device().is_mps(), name, " must be on MPS");
  TORCH_CHECK(t.scalar_type() == torch::kFloat32, name, " must be float32");
  TORCH_CHECK(t.dim() == 1, name, " must be rank-1");
  TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

void check_segment_tape_offsets_cpu(
    const torch::Tensor& segment_offsets_i32,
    const int64_t sample_count,
    const int64_t segment_count,
    const int64_t max_segments_per_sample) {
  auto offsets_cpu = segment_offsets_i32.cpu();
  const int32_t* offsets = offsets_cpu.data_ptr<int32_t>();
  TORCH_CHECK(offsets[0] == 0, "segment_offsets_i32[0] must be 0");
  for (int64_t sample = 0; sample < sample_count; ++sample) {
    const int32_t begin = offsets[sample];
    const int32_t end = offsets[sample + 1];
    TORCH_CHECK(end >= begin, "segment_offsets_i32 must be monotonic nondecreasing");
    TORCH_CHECK(begin >= 0, "segment_offsets_i32 values must be nonnegative");
    TORCH_CHECK(
        (int64_t)(end - begin) <= max_segments_per_sample,
        "segment tape row exceeds per-sample Metal replay cap");
  }
  TORCH_CHECK(offsets[sample_count] == segment_count, "segment_offsets_i32[-1] must match segment count");
}

void check_segment_tape_offsets_i16_cpu(
    const torch::Tensor& segment_offsets_i16,
    const int64_t sample_count,
    const int64_t segment_count,
    const int64_t max_segments_per_sample) {
  auto offsets_cpu = segment_offsets_i16.cpu();
  const int16_t* offsets = offsets_cpu.data_ptr<int16_t>();
  TORCH_CHECK(offsets[0] == 0, "segment_offsets_i16[0] must be 0");
  for (int64_t sample = 0; sample < sample_count; ++sample) {
    const int32_t begin = static_cast<int32_t>(offsets[sample]);
    const int32_t end = static_cast<int32_t>(offsets[sample + 1]);
    TORCH_CHECK(end >= begin, "segment_offsets_i16 must be monotonic nondecreasing");
    TORCH_CHECK(begin >= 0, "segment_offsets_i16 values must be nonnegative");
    TORCH_CHECK(
        (int64_t)(end - begin) <= max_segments_per_sample,
        "segment tape row exceeds per-sample Metal replay cap");
  }
  TORCH_CHECK(
      static_cast<int64_t>(offsets[sample_count]) == segment_count,
      "segment_offsets_i16[-1] must match segment count");
}

void check_replay_candidate_mask_count(
    const torch::Tensor& candidate_mask_u32,
    const torch::Tensor& beam_f32,
    const int32_t time_slab_count) {
  TORCH_CHECK(time_slab_count > 0, "config_i32[4] time_slab_count must be positive");
  TORCH_CHECK(
      candidate_mask_u32.size(0) == beam_f32.size(0) * (int64_t)time_slab_count,
      "candidate_mask_u32 length must be beam_count * time_slab_count");
}

}  // namespace

torch::Tensor metal_count_power_boundary_events(
    const torch::Tensor& boundary_f32,
    const torch::Tensor& boundary_u32,
    const torch::Tensor& beam_f32,
    const torch::Tensor& beam_u32,
    const torch::Tensor& config_i32,
    const torch::Tensor& config_f32) {
  check_float_mps_2d(boundary_f32, "boundary_f32", 4);
  check_i32_mps_2d(boundary_u32, "boundary_u32", 4);
  check_float_mps_2d(beam_f32, "beam_f32", 5);
  check_i32_mps_2d(beam_u32, "beam_u32", 4);
  check_i32_mps_1d(config_i32, "config_i32", 2);
  TORCH_CHECK(config_f32.device().is_mps(), "config_f32 must be on MPS");
  TORCH_CHECK(config_f32.scalar_type() == torch::kFloat32, "config_f32 must be float32");
  TORCH_CHECK(config_f32.dim() == 1 && config_f32.size(0) == 2, "config_f32 must have shape [2]");
  TORCH_CHECK(config_f32.is_contiguous(), "config_f32 must be contiguous");
  TORCH_CHECK(boundary_u32.size(0) == boundary_f32.size(0), "boundary tensor row count mismatch");
  TORCH_CHECK(beam_u32.size(0) == beam_f32.size(0), "beam tensor row count mismatch");

  auto config_i32_cpu = config_i32.cpu();
  const int32_t* config = config_i32_cpu.data_ptr<int32_t>();
  TORCH_CHECK(config[0] == boundary_f32.size(0), "config_i32[0] must match boundary count");
  TORCH_CHECK(config[1] == beam_f32.size(0), "config_i32[1] must match beam count");

  auto counts = torch::empty({beam_f32.size(0), 8}, beam_u32.options().dtype(torch::kInt32));
  auto& k = kernels();
  constexpr uint64_t threads = 256ull;
  launch(k.count_power_boundary_events, [&](MetalKernelFunction& fn) {
    fn.setArg(0, boundary_f32);
    fn.setArg(1, boundary_u32);
    fn.setArg(2, beam_f32);
    fn.setArg(3, beam_u32);
    fn.setArg(4, config_i32);
    fn.setArg(5, config_f32);
    fn.setArg(6, counts);
    fn.dispatch((uint64_t)beam_f32.size(0), threads);
  });
  return counts;
}

std::tuple<torch::Tensor, torch::Tensor> metal_shared_signal_replay(
    const torch::Tensor& boundary_f32,
    const torch::Tensor& candidate_mask_u32,
    const torch::Tensor& sites_f32,
    const torch::Tensor& site_signal_f32,
    const torch::Tensor& beam_f32,
    const torch::Tensor& frame_t_f32,
    const torch::Tensor& grad_output_f32,
    const torch::Tensor& config_i32,
    const torch::Tensor& config_f32) {
  check_float_mps_2d(boundary_f32, "boundary_f32", 4);
  check_i32_mps_1d_any(candidate_mask_u32, "candidate_mask_u32");
  check_float_mps_2d(sites_f32, "sites_f32", 4);
  TORCH_CHECK(site_signal_f32.device().is_mps(), "site_signal_f32 must be on MPS");
  TORCH_CHECK(site_signal_f32.scalar_type() == torch::kFloat32, "site_signal_f32 must be float32");
  TORCH_CHECK(site_signal_f32.dim() == 1, "site_signal_f32 must have shape [N]");
  TORCH_CHECK(site_signal_f32.is_contiguous(), "site_signal_f32 must be contiguous");
  check_float_mps_2d(beam_f32, "beam_f32", 5);
  TORCH_CHECK(frame_t_f32.device().is_mps(), "frame_t_f32 must be on MPS");
  TORCH_CHECK(frame_t_f32.scalar_type() == torch::kFloat32, "frame_t_f32 must be float32");
  TORCH_CHECK(frame_t_f32.dim() == 1, "frame_t_f32 must have shape [T]");
  TORCH_CHECK(frame_t_f32.is_contiguous(), "frame_t_f32 must be contiguous");
  TORCH_CHECK(grad_output_f32.device().is_mps(), "grad_output_f32 must be on MPS");
  TORCH_CHECK(grad_output_f32.scalar_type() == torch::kFloat32, "grad_output_f32 must be float32");
  TORCH_CHECK(grad_output_f32.dim() == 2, "grad_output_f32 must have shape [M,T]");
  TORCH_CHECK(grad_output_f32.is_contiguous(), "grad_output_f32 must be contiguous");
  check_i32_mps_1d(config_i32, "config_i32", 5);
  TORCH_CHECK(config_f32.device().is_mps(), "config_f32 must be on MPS");
  TORCH_CHECK(config_f32.scalar_type() == torch::kFloat32, "config_f32 must be float32");
  TORCH_CHECK(config_f32.dim() == 1 && config_f32.size(0) == 1, "config_f32 must have shape [1]");
  TORCH_CHECK(config_f32.is_contiguous(), "config_f32 must be contiguous");

  auto config_i32_cpu = config_i32.cpu();
  const int32_t* config = config_i32_cpu.data_ptr<int32_t>();
  TORCH_CHECK(config[0] == boundary_f32.size(0), "config_i32[0] must match boundary count");
  TORCH_CHECK(config[1] == beam_f32.size(0), "config_i32[1] must match beam count");
  TORCH_CHECK(config[2] == sites_f32.size(0), "config_i32[2] must match site count");
  TORCH_CHECK(config[3] == frame_t_f32.size(0), "config_i32[3] must match frame count");
  check_replay_candidate_mask_count(candidate_mask_u32, beam_f32, config[4]);
  TORCH_CHECK(boundary_f32.size(0) <= 31, "shared replay currently supports at most 31 boundaries");
  TORCH_CHECK(sites_f32.size(0) <= 32, "shared replay currently supports at most 32 sites");
  TORCH_CHECK(site_signal_f32.size(0) == sites_f32.size(0), "site signal count mismatch");
  TORCH_CHECK(
      grad_output_f32.size(0) == beam_f32.size(0) && grad_output_f32.size(1) == frame_t_f32.size(0),
      "grad_output_f32 shape mismatch");

  auto output = torch::empty({beam_f32.size(0), frame_t_f32.size(0)}, beam_f32.options().dtype(torch::kFloat32));
  auto grad_samples = torch::empty(
      {beam_f32.size(0), frame_t_f32.size(0), sites_f32.size(0)},
      beam_f32.options().dtype(torch::kFloat32));
  auto& k = kernels();
  constexpr uint64_t threads = 256ull;
  const uint64_t total = (uint64_t)beam_f32.size(0) * (uint64_t)frame_t_f32.size(0);
  launch(k.shared_signal_replay, [&](MetalKernelFunction& fn) {
    fn.setArg(0, boundary_f32);
    fn.setArg(1, candidate_mask_u32);
    fn.setArg(2, sites_f32);
    fn.setArg(3, site_signal_f32);
    fn.setArg(4, beam_f32);
    fn.setArg(5, frame_t_f32);
    fn.setArg(6, grad_output_f32);
    fn.setArg(7, config_i32);
    fn.setArg(8, config_f32);
    fn.setArg(9, output);
    fn.setArg(10, grad_samples);
    fn.dispatch(total, threads);
  });
  return std::make_tuple(output, grad_samples);
}

std::tuple<torch::Tensor, torch::Tensor> metal_shared_rgb_replay(
    const torch::Tensor& boundary_f32,
    const torch::Tensor& candidate_mask_u32,
    const torch::Tensor& sites_f32,
    const torch::Tensor& site_rgb_f32,
    const torch::Tensor& beam_f32,
    const torch::Tensor& frame_t_f32,
    const torch::Tensor& grad_output_rgb_f32,
    const torch::Tensor& config_i32,
    const torch::Tensor& config_f32) {
  check_float_mps_2d(boundary_f32, "boundary_f32", 4);
  check_i32_mps_1d_any(candidate_mask_u32, "candidate_mask_u32");
  check_float_mps_2d(sites_f32, "sites_f32", 4);
  check_float_mps_2d(site_rgb_f32, "site_rgb_f32", 3);
  check_float_mps_2d(beam_f32, "beam_f32", 5);
  TORCH_CHECK(frame_t_f32.device().is_mps(), "frame_t_f32 must be on MPS");
  TORCH_CHECK(frame_t_f32.scalar_type() == torch::kFloat32, "frame_t_f32 must be float32");
  TORCH_CHECK(frame_t_f32.dim() == 1, "frame_t_f32 must have shape [T]");
  TORCH_CHECK(frame_t_f32.is_contiguous(), "frame_t_f32 must be contiguous");
  TORCH_CHECK(grad_output_rgb_f32.device().is_mps(), "grad_output_rgb_f32 must be on MPS");
  TORCH_CHECK(grad_output_rgb_f32.scalar_type() == torch::kFloat32, "grad_output_rgb_f32 must be float32");
  TORCH_CHECK(grad_output_rgb_f32.dim() == 3, "grad_output_rgb_f32 must have shape [M,T,3]");
  TORCH_CHECK(grad_output_rgb_f32.size(2) == 3, "grad_output_rgb_f32 must have shape [M,T,3]");
  TORCH_CHECK(grad_output_rgb_f32.is_contiguous(), "grad_output_rgb_f32 must be contiguous");
  check_i32_mps_1d(config_i32, "config_i32", 5);
  TORCH_CHECK(config_f32.device().is_mps(), "config_f32 must be on MPS");
  TORCH_CHECK(config_f32.scalar_type() == torch::kFloat32, "config_f32 must be float32");
  TORCH_CHECK(config_f32.dim() == 1 && config_f32.size(0) == 1, "config_f32 must have shape [1]");
  TORCH_CHECK(config_f32.is_contiguous(), "config_f32 must be contiguous");

  auto config_i32_cpu = config_i32.cpu();
  const int32_t* config = config_i32_cpu.data_ptr<int32_t>();
  TORCH_CHECK(config[0] == boundary_f32.size(0), "config_i32[0] must match boundary count");
  TORCH_CHECK(config[1] == beam_f32.size(0), "config_i32[1] must match beam count");
  TORCH_CHECK(config[2] == sites_f32.size(0), "config_i32[2] must match site count");
  TORCH_CHECK(config[3] == frame_t_f32.size(0), "config_i32[3] must match frame count");
  check_replay_candidate_mask_count(candidate_mask_u32, beam_f32, config[4]);
  TORCH_CHECK(boundary_f32.size(0) <= 31, "shared RGB replay currently supports at most 31 boundaries");
  TORCH_CHECK(sites_f32.size(0) <= 32, "shared RGB replay currently supports at most 32 sites");
  TORCH_CHECK(site_rgb_f32.size(0) == sites_f32.size(0), "site RGB count mismatch");
  TORCH_CHECK(
      grad_output_rgb_f32.size(0) == beam_f32.size(0) &&
          grad_output_rgb_f32.size(1) == frame_t_f32.size(0),
      "grad_output_rgb_f32 shape mismatch");

  auto output_rgb = torch::empty(
      {beam_f32.size(0), frame_t_f32.size(0), 3},
      beam_f32.options().dtype(torch::kFloat32));
  auto grad_samples_rgb = torch::empty(
      {beam_f32.size(0), frame_t_f32.size(0), sites_f32.size(0), 3},
      beam_f32.options().dtype(torch::kFloat32));
  auto& k = kernels();
  constexpr uint64_t threads = 256ull;
  const uint64_t total = (uint64_t)beam_f32.size(0) * (uint64_t)frame_t_f32.size(0);
  launch(k.shared_rgb_replay, [&](MetalKernelFunction& fn) {
    fn.setArg(0, boundary_f32);
    fn.setArg(1, candidate_mask_u32);
    fn.setArg(2, sites_f32);
    fn.setArg(3, site_rgb_f32);
    fn.setArg(4, beam_f32);
    fn.setArg(5, frame_t_f32);
    fn.setArg(6, grad_output_rgb_f32);
    fn.setArg(7, config_i32);
    fn.setArg(8, config_f32);
    fn.setArg(9, output_rgb);
    fn.setArg(10, grad_samples_rgb);
    fn.dispatch(total, threads);
  });
  return std::make_tuple(output_rgb, grad_samples_rgb);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> metal_shared_rgba_depth_replay(
    const torch::Tensor& boundary_f32,
    const torch::Tensor& candidate_mask_u32,
    const torch::Tensor& sites_f32,
    const torch::Tensor& site_rgba_f32,
    const torch::Tensor& beam_f32,
    const torch::Tensor& frame_t_f32,
    const torch::Tensor& config_i32,
    const torch::Tensor& config_f32) {
  check_float_mps_2d(boundary_f32, "boundary_f32", 4);
  check_i32_mps_1d_any(candidate_mask_u32, "candidate_mask_u32");
  check_float_mps_2d(sites_f32, "sites_f32", 4);
  check_float_mps_2d(site_rgba_f32, "site_rgba_f32", 4);
  check_float_mps_2d(beam_f32, "beam_f32", 5);
  TORCH_CHECK(frame_t_f32.device().is_mps(), "frame_t_f32 must be on MPS");
  TORCH_CHECK(frame_t_f32.scalar_type() == torch::kFloat32, "frame_t_f32 must be float32");
  TORCH_CHECK(frame_t_f32.dim() == 1, "frame_t_f32 must have shape [T]");
  TORCH_CHECK(frame_t_f32.is_contiguous(), "frame_t_f32 must be contiguous");
  check_i32_mps_1d(config_i32, "config_i32", 5);
  TORCH_CHECK(config_f32.device().is_mps(), "config_f32 must be on MPS");
  TORCH_CHECK(config_f32.scalar_type() == torch::kFloat32, "config_f32 must be float32");
  TORCH_CHECK(config_f32.dim() == 1 && config_f32.size(0) == 1, "config_f32 must have shape [1]");
  TORCH_CHECK(config_f32.is_contiguous(), "config_f32 must be contiguous");

  auto config_i32_cpu = config_i32.cpu();
  const int32_t* config = config_i32_cpu.data_ptr<int32_t>();
  TORCH_CHECK(config[0] == boundary_f32.size(0), "config_i32[0] must match boundary count");
  TORCH_CHECK(config[1] == beam_f32.size(0), "config_i32[1] must match beam count");
  TORCH_CHECK(config[2] == sites_f32.size(0), "config_i32[2] must match site count");
  TORCH_CHECK(config[3] == frame_t_f32.size(0), "config_i32[3] must match frame count");
  check_replay_candidate_mask_count(candidate_mask_u32, beam_f32, config[4]);
  TORCH_CHECK(boundary_f32.size(0) <= 31, "shared RGBA/depth replay currently supports at most 31 boundaries");
  TORCH_CHECK(sites_f32.size(0) <= 32, "shared RGBA/depth replay currently supports at most 32 sites");
  TORCH_CHECK(site_rgba_f32.size(0) == sites_f32.size(0), "site RGBA count mismatch");

  auto output_rgb = torch::empty(
      {beam_f32.size(0), frame_t_f32.size(0), 3},
      beam_f32.options().dtype(torch::kFloat32));
  auto output_alpha = torch::empty(
      {beam_f32.size(0), frame_t_f32.size(0)},
      beam_f32.options().dtype(torch::kFloat32));
  auto output_depth = torch::empty(
      {beam_f32.size(0), frame_t_f32.size(0)},
      beam_f32.options().dtype(torch::kFloat32));
  auto& k = kernels();
  constexpr uint64_t threads = 256ull;
  const uint64_t total = (uint64_t)beam_f32.size(0) * (uint64_t)frame_t_f32.size(0);
  launch(k.shared_rgba_depth_replay, [&](MetalKernelFunction& fn) {
    fn.setArg(0, boundary_f32);
    fn.setArg(1, candidate_mask_u32);
    fn.setArg(2, sites_f32);
    fn.setArg(3, site_rgba_f32);
    fn.setArg(4, beam_f32);
    fn.setArg(5, frame_t_f32);
    fn.setArg(6, config_i32);
    fn.setArg(7, config_f32);
    fn.setArg(8, output_rgb);
    fn.setArg(9, output_alpha);
    fn.setArg(10, output_depth);
    fn.dispatch(total, threads);
  });
  return std::make_tuple(output_rgb, output_alpha, output_depth);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> metal_shared_rgba_depth_vjp(
    const torch::Tensor& boundary_f32,
    const torch::Tensor& candidate_mask_u32,
    const torch::Tensor& sites_f32,
    const torch::Tensor& site_rgba_f32,
    const torch::Tensor& beam_f32,
    const torch::Tensor& frame_t_f32,
    const torch::Tensor& grad_rgb_f32,
    const torch::Tensor& grad_alpha_f32,
    const torch::Tensor& grad_depth_f32,
    const torch::Tensor& config_i32,
    const torch::Tensor& config_f32) {
  check_float_mps_2d(boundary_f32, "boundary_f32", 4);
  check_i32_mps_1d_any(candidate_mask_u32, "candidate_mask_u32");
  check_float_mps_2d(sites_f32, "sites_f32", 4);
  check_float_mps_2d(site_rgba_f32, "site_rgba_f32", 4);
  check_float_mps_2d(beam_f32, "beam_f32", 5);
  TORCH_CHECK(frame_t_f32.device().is_mps(), "frame_t_f32 must be on MPS");
  TORCH_CHECK(frame_t_f32.scalar_type() == torch::kFloat32, "frame_t_f32 must be float32");
  TORCH_CHECK(frame_t_f32.dim() == 1, "frame_t_f32 must have shape [T]");
  TORCH_CHECK(frame_t_f32.is_contiguous(), "frame_t_f32 must be contiguous");
  TORCH_CHECK(grad_rgb_f32.device().is_mps(), "grad_rgb_f32 must be on MPS");
  TORCH_CHECK(grad_rgb_f32.scalar_type() == torch::kFloat32, "grad_rgb_f32 must be float32");
  TORCH_CHECK(grad_rgb_f32.dim() == 3 && grad_rgb_f32.size(2) == 3, "grad_rgb_f32 must have shape [M,T,3]");
  TORCH_CHECK(grad_rgb_f32.is_contiguous(), "grad_rgb_f32 must be contiguous");
  TORCH_CHECK(grad_alpha_f32.device().is_mps(), "grad_alpha_f32 must be on MPS");
  TORCH_CHECK(grad_alpha_f32.scalar_type() == torch::kFloat32, "grad_alpha_f32 must be float32");
  TORCH_CHECK(grad_alpha_f32.dim() == 2, "grad_alpha_f32 must have shape [M,T]");
  TORCH_CHECK(grad_alpha_f32.is_contiguous(), "grad_alpha_f32 must be contiguous");
  TORCH_CHECK(grad_depth_f32.device().is_mps(), "grad_depth_f32 must be on MPS");
  TORCH_CHECK(grad_depth_f32.scalar_type() == torch::kFloat32, "grad_depth_f32 must be float32");
  TORCH_CHECK(grad_depth_f32.dim() == 2, "grad_depth_f32 must have shape [M,T]");
  TORCH_CHECK(grad_depth_f32.is_contiguous(), "grad_depth_f32 must be contiguous");
  check_i32_mps_1d(config_i32, "config_i32", 5);
  TORCH_CHECK(config_f32.device().is_mps(), "config_f32 must be on MPS");
  TORCH_CHECK(config_f32.scalar_type() == torch::kFloat32, "config_f32 must be float32");
  TORCH_CHECK(config_f32.dim() == 1 && config_f32.size(0) == 1, "config_f32 must have shape [1]");
  TORCH_CHECK(config_f32.is_contiguous(), "config_f32 must be contiguous");

  auto config_i32_cpu = config_i32.cpu();
  const int32_t* config = config_i32_cpu.data_ptr<int32_t>();
  TORCH_CHECK(config[0] == boundary_f32.size(0), "config_i32[0] must match boundary count");
  TORCH_CHECK(config[1] == beam_f32.size(0), "config_i32[1] must match beam count");
  TORCH_CHECK(config[2] == sites_f32.size(0), "config_i32[2] must match site count");
  TORCH_CHECK(config[3] == frame_t_f32.size(0), "config_i32[3] must match frame count");
  check_replay_candidate_mask_count(candidate_mask_u32, beam_f32, config[4]);
  TORCH_CHECK(boundary_f32.size(0) <= 31, "shared RGBA/depth VJP currently supports at most 31 boundaries");
  TORCH_CHECK(sites_f32.size(0) <= 32, "shared RGBA/depth VJP currently supports at most 32 sites");
  TORCH_CHECK(site_rgba_f32.size(0) == sites_f32.size(0), "site RGBA count mismatch");
  TORCH_CHECK(
      grad_rgb_f32.size(0) == beam_f32.size(0) &&
          grad_rgb_f32.size(1) == frame_t_f32.size(0),
      "grad_rgb_f32 shape mismatch");
  TORCH_CHECK(
      grad_alpha_f32.size(0) == beam_f32.size(0) &&
          grad_alpha_f32.size(1) == frame_t_f32.size(0),
      "grad_alpha_f32 shape mismatch");
  TORCH_CHECK(
      grad_depth_f32.size(0) == beam_f32.size(0) &&
          grad_depth_f32.size(1) == frame_t_f32.size(0),
      "grad_depth_f32 shape mismatch");

  auto output_rgb = torch::empty(
      {beam_f32.size(0), frame_t_f32.size(0), 3},
      beam_f32.options().dtype(torch::kFloat32));
  auto output_alpha = torch::empty(
      {beam_f32.size(0), frame_t_f32.size(0)},
      beam_f32.options().dtype(torch::kFloat32));
  auto output_depth = torch::empty(
      {beam_f32.size(0), frame_t_f32.size(0)},
      beam_f32.options().dtype(torch::kFloat32));
  auto grad_samples_rgba = torch::empty(
      {beam_f32.size(0), frame_t_f32.size(0), sites_f32.size(0), 4},
      beam_f32.options().dtype(torch::kFloat32));
  auto& k = kernels();
  constexpr uint64_t threads = 256ull;
  const uint64_t total = (uint64_t)beam_f32.size(0) * (uint64_t)frame_t_f32.size(0);
  launch(k.shared_rgba_depth_vjp, [&](MetalKernelFunction& fn) {
    fn.setArg(0, boundary_f32);
    fn.setArg(1, candidate_mask_u32);
    fn.setArg(2, sites_f32);
    fn.setArg(3, site_rgba_f32);
    fn.setArg(4, beam_f32);
    fn.setArg(5, frame_t_f32);
    fn.setArg(6, grad_rgb_f32);
    fn.setArg(7, grad_alpha_f32);
    fn.setArg(8, grad_depth_f32);
    fn.setArg(9, config_i32);
    fn.setArg(10, config_f32);
    fn.setArg(11, output_rgb);
    fn.setArg(12, output_alpha);
    fn.setArg(13, output_depth);
    fn.setArg(14, grad_samples_rgba);
    fn.dispatch(total, threads);
  });
  return std::make_tuple(output_rgb, output_alpha, output_depth, grad_samples_rgba);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> metal_realray_rgba_depth_replay(
    const torch::Tensor& boundary_f32,
    const torch::Tensor& sites_f32,
    const torch::Tensor& site_rgba_f32,
    const torch::Tensor& rays_f32,
    const torch::Tensor& frame_t_f32,
    const torch::Tensor& config_i32,
    const torch::Tensor& config_f32) {
  check_float_mps_2d(boundary_f32, "boundary_f32", 5);
  check_float_mps_2d(sites_f32, "sites_f32", 5);
  check_float_mps_2d(site_rgba_f32, "site_rgba_f32", 4);
  check_float_mps_2d(rays_f32, "rays_f32", 6);
  TORCH_CHECK(frame_t_f32.device().is_mps(), "frame_t_f32 must be on MPS");
  TORCH_CHECK(frame_t_f32.scalar_type() == torch::kFloat32, "frame_t_f32 must be float32");
  TORCH_CHECK(frame_t_f32.dim() == 1, "frame_t_f32 must have shape [R]");
  TORCH_CHECK(frame_t_f32.is_contiguous(), "frame_t_f32 must be contiguous");
  check_i32_mps_1d(config_i32, "config_i32", 3);
  TORCH_CHECK(config_f32.device().is_mps(), "config_f32 must be on MPS");
  TORCH_CHECK(config_f32.scalar_type() == torch::kFloat32, "config_f32 must be float32");
  TORCH_CHECK(config_f32.dim() == 1 && config_f32.size(0) == 4, "config_f32 must have shape [4]");
  TORCH_CHECK(config_f32.is_contiguous(), "config_f32 must be contiguous");

  auto config_i32_cpu = config_i32.cpu();
  const int32_t* config = config_i32_cpu.data_ptr<int32_t>();
  TORCH_CHECK(config[0] == boundary_f32.size(0), "config_i32[0] must match boundary count");
  TORCH_CHECK(config[1] == rays_f32.size(0), "config_i32[1] must match ray count");
  TORCH_CHECK(config[2] == sites_f32.size(0), "config_i32[2] must match site count");
  TORCH_CHECK(boundary_f32.size(0) <= 128, "real-ray replay currently supports at most 128 boundaries");
  TORCH_CHECK(sites_f32.size(0) <= 64, "real-ray replay currently supports at most 64 sites");
  TORCH_CHECK(site_rgba_f32.size(0) == sites_f32.size(0), "site RGBA count mismatch");
  TORCH_CHECK(frame_t_f32.size(0) == rays_f32.size(0), "frame_t_f32 length must match rays_f32 rows");

  auto output_rgb = torch::empty({rays_f32.size(0), 3}, rays_f32.options().dtype(torch::kFloat32));
  auto output_alpha = torch::empty({rays_f32.size(0)}, rays_f32.options().dtype(torch::kFloat32));
  auto output_depth = torch::empty({rays_f32.size(0)}, rays_f32.options().dtype(torch::kFloat32));
  auto& k = kernels();
  constexpr uint64_t threads = 256ull;
  launch(k.realray_rgba_depth_replay, [&](MetalKernelFunction& fn) {
    fn.setArg(0, boundary_f32);
    fn.setArg(1, sites_f32);
    fn.setArg(2, site_rgba_f32);
    fn.setArg(3, rays_f32);
    fn.setArg(4, frame_t_f32);
    fn.setArg(5, config_i32);
    fn.setArg(6, config_f32);
    fn.setArg(7, output_rgb);
    fn.setArg(8, output_alpha);
    fn.setArg(9, output_depth);
    fn.dispatch((uint64_t)rays_f32.size(0), threads);
  });
  return std::make_tuple(output_rgb, output_alpha, output_depth);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> metal_shared_realray_rgba_depth_replay(
    const torch::Tensor& boundary_f32,
    const torch::Tensor& candidate_mask_i32,
    const torch::Tensor& sites_f32,
    const torch::Tensor& site_rgba_f32,
    const torch::Tensor& track_rays_f32,
    const torch::Tensor& frame_t_f32,
    const torch::Tensor& config_i32,
    const torch::Tensor& config_f32) {
  check_float_mps_2d(boundary_f32, "boundary_f32", 5);
  TORCH_CHECK(candidate_mask_i32.device().is_mps(), "candidate_mask_i32 must be on MPS");
  TORCH_CHECK(candidate_mask_i32.scalar_type() == torch::kInt32, "candidate_mask_i32 must be int32");
  TORCH_CHECK(candidate_mask_i32.dim() == 2, "candidate_mask_i32 must have shape [track_count * time_slab_count, mask_word_count]");
  TORCH_CHECK(candidate_mask_i32.is_contiguous(), "candidate_mask_i32 must be contiguous");
  check_float_mps_2d(sites_f32, "sites_f32", 5);
  check_float_mps_2d(site_rgba_f32, "site_rgba_f32", 4);
  check_float_mps_2d(track_rays_f32, "track_rays_f32", 6);
  TORCH_CHECK(frame_t_f32.device().is_mps(), "frame_t_f32 must be on MPS");
  TORCH_CHECK(frame_t_f32.scalar_type() == torch::kFloat32, "frame_t_f32 must be float32");
  TORCH_CHECK(frame_t_f32.dim() == 1, "frame_t_f32 must have shape [T]");
  TORCH_CHECK(frame_t_f32.is_contiguous(), "frame_t_f32 must be contiguous");
  check_i32_mps_1d(config_i32, "config_i32", 6);
  TORCH_CHECK(config_f32.device().is_mps(), "config_f32 must be on MPS");
  TORCH_CHECK(config_f32.scalar_type() == torch::kFloat32, "config_f32 must be float32");
  TORCH_CHECK(config_f32.dim() == 1 && config_f32.size(0) == 4, "config_f32 must have shape [4]");
  TORCH_CHECK(config_f32.is_contiguous(), "config_f32 must be contiguous");

  auto config_i32_cpu = config_i32.cpu();
  const int32_t* config = config_i32_cpu.data_ptr<int32_t>();
  TORCH_CHECK(config[0] == boundary_f32.size(0), "config_i32[0] must match boundary count");
  TORCH_CHECK(config[1] == track_rays_f32.size(0), "config_i32[1] must match track count");
  TORCH_CHECK(config[2] == sites_f32.size(0), "config_i32[2] must match site count");
  TORCH_CHECK(config[3] == frame_t_f32.size(0), "config_i32[3] must match frame count");
  TORCH_CHECK(config[4] > 0, "config_i32[4] time_slab_count must be positive");
  TORCH_CHECK(config[5] > 0, "config_i32[5] mask_word_count must be positive");
  TORCH_CHECK(boundary_f32.size(0) <= 128, "shared real-ray replay currently supports at most 128 boundaries");
  TORCH_CHECK(sites_f32.size(0) <= 64, "shared real-ray replay currently supports at most 64 sites");
  TORCH_CHECK(site_rgba_f32.size(0) == sites_f32.size(0), "site RGBA count mismatch");
  TORCH_CHECK(
      candidate_mask_i32.size(0) == track_rays_f32.size(0) * (int64_t)config[4],
      "candidate_mask_i32 row count must be track_count * time_slab_count");
  TORCH_CHECK(candidate_mask_i32.size(1) == config[5], "candidate_mask_i32 column count must match mask_word_count");
  TORCH_CHECK(
      config[5] == (int32_t)((boundary_f32.size(0) + 31) / 32),
      "mask_word_count must equal ceil(boundary_count / 32)");

  auto output_rgb = torch::empty(
      {track_rays_f32.size(0), frame_t_f32.size(0), 3},
      track_rays_f32.options().dtype(torch::kFloat32));
  auto output_alpha = torch::empty(
      {track_rays_f32.size(0), frame_t_f32.size(0)},
      track_rays_f32.options().dtype(torch::kFloat32));
  auto output_depth = torch::empty(
      {track_rays_f32.size(0), frame_t_f32.size(0)},
      track_rays_f32.options().dtype(torch::kFloat32));
  auto& k = kernels();
  constexpr uint64_t threads = 256ull;
  const uint64_t total = (uint64_t)track_rays_f32.size(0) * (uint64_t)frame_t_f32.size(0);
  launch(k.shared_realray_rgba_depth_replay, [&](MetalKernelFunction& fn) {
    fn.setArg(0, boundary_f32);
    fn.setArg(1, candidate_mask_i32);
    fn.setArg(2, sites_f32);
    fn.setArg(3, site_rgba_f32);
    fn.setArg(4, track_rays_f32);
    fn.setArg(5, frame_t_f32);
    fn.setArg(6, config_i32);
    fn.setArg(7, config_f32);
    fn.setArg(8, output_rgb);
    fn.setArg(9, output_alpha);
    fn.setArg(10, output_depth);
    fn.dispatch(total, threads);
  });
  return std::make_tuple(output_rgb, output_alpha, output_depth);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> metal_fused_slab_affine_realray_rgba_depth_replay(
    const torch::Tensor& boundary_f32,
    const torch::Tensor& row_index_i32,
    const torch::Tensor& candidate_row_offsets_i32,
    const torch::Tensor& candidate_boundary_ids_i32,
    const torch::Tensor& sites_f32,
    const torch::Tensor& site_rgba_f32,
    const torch::Tensor& ray_coeff_f32,
    const torch::Tensor& frame_t_f32,
    const torch::Tensor& config_i32,
    const torch::Tensor& config_f32) {
  check_float_mps_2d(boundary_f32, "boundary_f32", 5);
  check_i32_mps_1d_any(row_index_i32, "row_index_i32");
  check_i32_mps_1d_any(candidate_row_offsets_i32, "candidate_row_offsets_i32");
  check_i32_mps_1d_any(candidate_boundary_ids_i32, "candidate_boundary_ids_i32");
  check_float_mps_2d(sites_f32, "sites_f32", 5);
  check_float_mps_2d(site_rgba_f32, "site_rgba_f32", 4);
  check_float_mps_2d(ray_coeff_f32, "ray_coeff_f32", 12);
  TORCH_CHECK(frame_t_f32.device().is_mps(), "frame_t_f32 must be on MPS");
  TORCH_CHECK(frame_t_f32.scalar_type() == torch::kFloat32, "frame_t_f32 must be float32");
  TORCH_CHECK(frame_t_f32.dim() == 1, "frame_t_f32 must have shape [T]");
  TORCH_CHECK(frame_t_f32.is_contiguous(), "frame_t_f32 must be contiguous");
  check_i32_mps_1d(config_i32, "config_i32", 7);
  TORCH_CHECK(config_f32.device().is_mps(), "config_f32 must be on MPS");
  TORCH_CHECK(config_f32.scalar_type() == torch::kFloat32, "config_f32 must be float32");
  TORCH_CHECK(config_f32.dim() == 1 && config_f32.size(0) == 4, "config_f32 must have shape [4]");
  TORCH_CHECK(config_f32.is_contiguous(), "config_f32 must be contiguous");

  auto config_i32_cpu = config_i32.cpu();
  const int32_t* config = config_i32_cpu.data_ptr<int32_t>();
  TORCH_CHECK(config[0] == boundary_f32.size(0), "config_i32[0] must match boundary count");
  TORCH_CHECK(config[1] == ray_coeff_f32.size(0), "config_i32[1] must match track count");
  TORCH_CHECK(config[2] == sites_f32.size(0), "config_i32[2] must match site count");
  TORCH_CHECK(config[3] == frame_t_f32.size(0), "config_i32[3] must match frame count");
  TORCH_CHECK(config[4] > 0, "config_i32[4] time_slab_count must be positive");
  TORCH_CHECK(config[5] > 0, "config_i32[5] row_count must be positive");
  TORCH_CHECK(config[6] >= 0, "config_i32[6] candidate_count must be nonnegative");
  TORCH_CHECK(ray_coeff_f32.size(0) > 0, "fused slab affine replay requires at least one track");
  TORCH_CHECK(frame_t_f32.size(0) > 0, "fused slab affine replay requires at least one frame");
  TORCH_CHECK(sites_f32.size(0) > 0, "fused slab affine replay requires at least one site");
  TORCH_CHECK(boundary_f32.size(0) <= 128, "fused slab affine replay currently supports at most 128 boundaries");
  TORCH_CHECK(sites_f32.size(0) <= 64, "fused slab affine replay currently supports at most 64 sites");
  TORCH_CHECK(site_rgba_f32.size(0) == sites_f32.size(0), "site RGBA count mismatch");
  TORCH_CHECK(row_index_i32.size(0) == ray_coeff_f32.size(0), "row_index_i32 length must match track count");
  TORCH_CHECK(
      candidate_row_offsets_i32.size(0) == (int64_t)config[5] * (int64_t)config[4] + 1,
      "candidate_row_offsets_i32 length must be row_count * time_slab_count + 1");
  TORCH_CHECK(
      candidate_boundary_ids_i32.size(0) == config[6],
      "candidate_boundary_ids_i32 length must match candidate_count");

  auto output_rgb = torch::empty(
      {ray_coeff_f32.size(0), frame_t_f32.size(0), 3},
      ray_coeff_f32.options().dtype(torch::kFloat32));
  auto output_alpha = torch::empty(
      {ray_coeff_f32.size(0), frame_t_f32.size(0)},
      ray_coeff_f32.options().dtype(torch::kFloat32));
  auto output_depth = torch::empty(
      {ray_coeff_f32.size(0), frame_t_f32.size(0)},
      ray_coeff_f32.options().dtype(torch::kFloat32));
  auto& k = kernels();
  constexpr uint64_t threads = 256ull;
  const uint64_t total = (uint64_t)ray_coeff_f32.size(0) * (uint64_t)frame_t_f32.size(0);
  launch(k.fused_slab_affine_realray_rgba_depth_replay, [&](MetalKernelFunction& fn) {
    fn.setArg(0, boundary_f32);
    fn.setArg(1, row_index_i32);
    fn.setArg(2, candidate_row_offsets_i32);
    fn.setArg(3, candidate_boundary_ids_i32);
    fn.setArg(4, sites_f32);
    fn.setArg(5, site_rgba_f32);
    fn.setArg(6, ray_coeff_f32);
    fn.setArg(7, frame_t_f32);
    fn.setArg(8, config_i32);
    fn.setArg(9, config_f32);
    fn.setArg(10, output_rgb);
    fn.setArg(11, output_alpha);
    fn.setArg(12, output_depth);
    fn.dispatch(total, threads);
  });
  return std::make_tuple(output_rgb, output_alpha, output_depth);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> metal_fused_slab_affine_coeff_realray_rgba_depth_replay(
    const torch::Tensor& row_index_i32,
    const torch::Tensor& candidate_row_offsets_i32,
    const torch::Tensor& candidate_depth_coeff_f32,
    const torch::Tensor& sites_f32,
    const torch::Tensor& site_rgba_f32,
    const torch::Tensor& ray_coeff_f32,
    const torch::Tensor& frame_t_f32,
    const torch::Tensor& config_i32,
    const torch::Tensor& config_f32) {
  check_i32_mps_1d_any(row_index_i32, "row_index_i32");
  check_i32_mps_1d_any(candidate_row_offsets_i32, "candidate_row_offsets_i32");
  check_float_mps_2d(candidate_depth_coeff_f32, "candidate_depth_coeff_f32", 4);
  check_float_mps_2d(sites_f32, "sites_f32", 5);
  check_float_mps_2d(site_rgba_f32, "site_rgba_f32", 4);
  check_float_mps_2d(ray_coeff_f32, "ray_coeff_f32", 12);
  TORCH_CHECK(frame_t_f32.device().is_mps(), "frame_t_f32 must be on MPS");
  TORCH_CHECK(frame_t_f32.scalar_type() == torch::kFloat32, "frame_t_f32 must be float32");
  TORCH_CHECK(frame_t_f32.dim() == 1, "frame_t_f32 must have shape [T]");
  TORCH_CHECK(frame_t_f32.is_contiguous(), "frame_t_f32 must be contiguous");
  check_i32_mps_1d(config_i32, "config_i32", 6);
  TORCH_CHECK(config_f32.device().is_mps(), "config_f32 must be on MPS");
  TORCH_CHECK(config_f32.scalar_type() == torch::kFloat32, "config_f32 must be float32");
  TORCH_CHECK(config_f32.dim() == 1 && config_f32.size(0) == 4, "config_f32 must have shape [4]");
  TORCH_CHECK(config_f32.is_contiguous(), "config_f32 must be contiguous");

  auto config_i32_cpu = config_i32.cpu();
  const int32_t* config = config_i32_cpu.data_ptr<int32_t>();
  TORCH_CHECK(config[0] == ray_coeff_f32.size(0), "config_i32[0] must match track count");
  TORCH_CHECK(config[1] == sites_f32.size(0), "config_i32[1] must match site count");
  TORCH_CHECK(config[2] == frame_t_f32.size(0), "config_i32[2] must match frame count");
  TORCH_CHECK(config[3] > 0, "config_i32[3] time_slab_count must be positive");
  TORCH_CHECK(config[4] > 0, "config_i32[4] row_count must be positive");
  TORCH_CHECK(config[5] >= 0, "config_i32[5] candidate_count must be nonnegative");
  TORCH_CHECK(ray_coeff_f32.size(0) > 0, "fused slab affine coeff replay requires at least one track");
  TORCH_CHECK(frame_t_f32.size(0) > 0, "fused slab affine coeff replay requires at least one frame");
  TORCH_CHECK(sites_f32.size(0) > 0, "fused slab affine coeff replay requires at least one site");
  TORCH_CHECK(sites_f32.size(0) <= 64, "fused slab affine coeff replay currently supports at most 64 sites");
  TORCH_CHECK(site_rgba_f32.size(0) == sites_f32.size(0), "site RGBA count mismatch");
  TORCH_CHECK(row_index_i32.size(0) == ray_coeff_f32.size(0), "row_index_i32 length must match track count");
  TORCH_CHECK(
      candidate_row_offsets_i32.size(0) == (int64_t)config[4] * (int64_t)config[3] + 1,
      "candidate_row_offsets_i32 length must be row_count * time_slab_count + 1");
  TORCH_CHECK(
      candidate_depth_coeff_f32.size(0) == config[5],
      "candidate_depth_coeff_f32 row count must match candidate_count");

  auto output_rgb = torch::empty(
      {ray_coeff_f32.size(0), frame_t_f32.size(0), 3},
      ray_coeff_f32.options().dtype(torch::kFloat32));
  auto output_alpha = torch::empty(
      {ray_coeff_f32.size(0), frame_t_f32.size(0)},
      ray_coeff_f32.options().dtype(torch::kFloat32));
  auto output_depth = torch::empty(
      {ray_coeff_f32.size(0), frame_t_f32.size(0)},
      ray_coeff_f32.options().dtype(torch::kFloat32));
  auto& k = kernels();
  constexpr uint64_t threads = 256ull;
  const uint64_t total = (uint64_t)ray_coeff_f32.size(0) * (uint64_t)frame_t_f32.size(0);
  launch(k.fused_slab_affine_coeff_realray_rgba_depth_replay, [&](MetalKernelFunction& fn) {
    fn.setArg(0, row_index_i32);
    fn.setArg(1, candidate_row_offsets_i32);
    fn.setArg(2, candidate_depth_coeff_f32);
    fn.setArg(3, sites_f32);
    fn.setArg(4, site_rgba_f32);
    fn.setArg(5, ray_coeff_f32);
    fn.setArg(6, frame_t_f32);
    fn.setArg(7, config_i32);
    fn.setArg(8, config_f32);
    fn.setArg(9, output_rgb);
    fn.setArg(10, output_alpha);
    fn.setArg(11, output_depth);
    fn.dispatch(total, threads);
  });
  return std::make_tuple(output_rgb, output_alpha, output_depth);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> metal_fused_slab_affine_coeff16_realray_rgba_depth_replay(
    const torch::Tensor& row_index_i32,
    const torch::Tensor& candidate_row_offsets_i32,
    const torch::Tensor& candidate_depth_coeff_f16,
    const torch::Tensor& sites_f32,
    const torch::Tensor& site_rgba_f32,
    const torch::Tensor& ray_coeff_f32,
    const torch::Tensor& frame_t_f32,
    const torch::Tensor& config_i32,
    const torch::Tensor& config_f32) {
  check_i32_mps_1d_any(row_index_i32, "row_index_i32");
  check_i32_mps_1d_any(candidate_row_offsets_i32, "candidate_row_offsets_i32");
  check_half_mps_2d(candidate_depth_coeff_f16, "candidate_depth_coeff_f16", 4);
  check_float_mps_2d(sites_f32, "sites_f32", 5);
  check_float_mps_2d(site_rgba_f32, "site_rgba_f32", 4);
  check_float_mps_2d(ray_coeff_f32, "ray_coeff_f32", 12);
  TORCH_CHECK(frame_t_f32.device().is_mps(), "frame_t_f32 must be on MPS");
  TORCH_CHECK(frame_t_f32.scalar_type() == torch::kFloat32, "frame_t_f32 must be float32");
  TORCH_CHECK(frame_t_f32.dim() == 1, "frame_t_f32 must have shape [T]");
  TORCH_CHECK(frame_t_f32.is_contiguous(), "frame_t_f32 must be contiguous");
  check_i32_mps_1d(config_i32, "config_i32", 6);
  TORCH_CHECK(config_f32.device().is_mps(), "config_f32 must be on MPS");
  TORCH_CHECK(config_f32.scalar_type() == torch::kFloat32, "config_f32 must be float32");
  TORCH_CHECK(config_f32.dim() == 1 && config_f32.size(0) == 4, "config_f32 must have shape [4]");
  TORCH_CHECK(config_f32.is_contiguous(), "config_f32 must be contiguous");

  auto config_i32_cpu = config_i32.cpu();
  const int32_t* config = config_i32_cpu.data_ptr<int32_t>();
  TORCH_CHECK(config[0] == ray_coeff_f32.size(0), "config_i32[0] must match track count");
  TORCH_CHECK(config[1] == sites_f32.size(0), "config_i32[1] must match site count");
  TORCH_CHECK(config[2] == frame_t_f32.size(0), "config_i32[2] must match frame count");
  TORCH_CHECK(config[3] > 0, "config_i32[3] time_slab_count must be positive");
  TORCH_CHECK(config[4] > 0, "config_i32[4] row_count must be positive");
  TORCH_CHECK(config[5] >= 0, "config_i32[5] candidate_count must be nonnegative");
  TORCH_CHECK(ray_coeff_f32.size(0) > 0, "fused slab affine coeff16 replay requires at least one track");
  TORCH_CHECK(frame_t_f32.size(0) > 0, "fused slab affine coeff16 replay requires at least one frame");
  TORCH_CHECK(sites_f32.size(0) > 0, "fused slab affine coeff16 replay requires at least one site");
  TORCH_CHECK(sites_f32.size(0) <= 64, "fused slab affine coeff16 replay currently supports at most 64 sites");
  TORCH_CHECK(site_rgba_f32.size(0) == sites_f32.size(0), "site RGBA count mismatch");
  TORCH_CHECK(row_index_i32.size(0) == ray_coeff_f32.size(0), "row_index_i32 length must match track count");
  TORCH_CHECK(
      candidate_row_offsets_i32.size(0) == (int64_t)config[4] * (int64_t)config[3] + 1,
      "candidate_row_offsets_i32 length must be row_count * time_slab_count + 1");
  TORCH_CHECK(
      candidate_depth_coeff_f16.size(0) == config[5],
      "candidate_depth_coeff_f16 row count must match candidate_count");

  auto output_rgb = torch::empty(
      {ray_coeff_f32.size(0), frame_t_f32.size(0), 3},
      ray_coeff_f32.options().dtype(torch::kFloat32));
  auto output_alpha = torch::empty(
      {ray_coeff_f32.size(0), frame_t_f32.size(0)},
      ray_coeff_f32.options().dtype(torch::kFloat32));
  auto output_depth = torch::empty(
      {ray_coeff_f32.size(0), frame_t_f32.size(0)},
      ray_coeff_f32.options().dtype(torch::kFloat32));
  auto& k = kernels();
  constexpr uint64_t threads = 256ull;
  const uint64_t total = (uint64_t)ray_coeff_f32.size(0) * (uint64_t)frame_t_f32.size(0);
  launch(k.fused_slab_affine_coeff16_realray_rgba_depth_replay, [&](MetalKernelFunction& fn) {
    fn.setArg(0, row_index_i32);
    fn.setArg(1, candidate_row_offsets_i32);
    fn.setArg(2, candidate_depth_coeff_f16);
    fn.setArg(3, sites_f32);
    fn.setArg(4, site_rgba_f32);
    fn.setArg(5, ray_coeff_f32);
    fn.setArg(6, frame_t_f32);
    fn.setArg(7, config_i32);
    fn.setArg(8, config_f32);
    fn.setArg(9, output_rgb);
    fn.setArg(10, output_alpha);
    fn.setArg(11, output_depth);
    fn.dispatch(total, threads);
  });
  return std::make_tuple(output_rgb, output_alpha, output_depth);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> metal_fused_slab_affine_num32_den16_realray_rgba_depth_replay(
    const torch::Tensor& row_index_i32,
    const torch::Tensor& candidate_row_offsets_i32,
    const torch::Tensor& candidate_depth_num_f32,
    const torch::Tensor& candidate_depth_den_f16,
    const torch::Tensor& sites_f32,
    const torch::Tensor& site_rgba_f32,
    const torch::Tensor& ray_coeff_f32,
    const torch::Tensor& frame_t_f32,
    const torch::Tensor& config_i32,
    const torch::Tensor& config_f32) {
  check_i32_mps_1d_any(row_index_i32, "row_index_i32");
  check_i32_mps_1d_any(candidate_row_offsets_i32, "candidate_row_offsets_i32");
  check_float_mps_2d(candidate_depth_num_f32, "candidate_depth_num_f32", 2);
  check_half_mps_2d(candidate_depth_den_f16, "candidate_depth_den_f16", 2);
  check_float_mps_2d(sites_f32, "sites_f32", 5);
  check_float_mps_2d(site_rgba_f32, "site_rgba_f32", 4);
  check_float_mps_2d(ray_coeff_f32, "ray_coeff_f32", 12);
  TORCH_CHECK(frame_t_f32.device().is_mps(), "frame_t_f32 must be on MPS");
  TORCH_CHECK(frame_t_f32.scalar_type() == torch::kFloat32, "frame_t_f32 must be float32");
  TORCH_CHECK(frame_t_f32.dim() == 1, "frame_t_f32 must have shape [T]");
  TORCH_CHECK(frame_t_f32.is_contiguous(), "frame_t_f32 must be contiguous");
  check_i32_mps_1d(config_i32, "config_i32", 6);
  TORCH_CHECK(config_f32.device().is_mps(), "config_f32 must be on MPS");
  TORCH_CHECK(config_f32.scalar_type() == torch::kFloat32, "config_f32 must be float32");
  TORCH_CHECK(config_f32.dim() == 1 && config_f32.size(0) == 4, "config_f32 must have shape [4]");
  TORCH_CHECK(config_f32.is_contiguous(), "config_f32 must be contiguous");

  auto config_i32_cpu = config_i32.cpu();
  const int32_t* config = config_i32_cpu.data_ptr<int32_t>();
  TORCH_CHECK(config[0] == ray_coeff_f32.size(0), "config_i32[0] must match track count");
  TORCH_CHECK(config[1] == sites_f32.size(0), "config_i32[1] must match site count");
  TORCH_CHECK(config[2] == frame_t_f32.size(0), "config_i32[2] must match frame count");
  TORCH_CHECK(config[3] > 0, "config_i32[3] time_slab_count must be positive");
  TORCH_CHECK(config[4] > 0, "config_i32[4] row_count must be positive");
  TORCH_CHECK(config[5] >= 0, "config_i32[5] candidate_count must be nonnegative");
  TORCH_CHECK(ray_coeff_f32.size(0) > 0, "fused slab affine num32/den16 replay requires at least one track");
  TORCH_CHECK(frame_t_f32.size(0) > 0, "fused slab affine num32/den16 replay requires at least one frame");
  TORCH_CHECK(sites_f32.size(0) > 0, "fused slab affine num32/den16 replay requires at least one site");
  TORCH_CHECK(sites_f32.size(0) <= 64, "fused slab affine num32/den16 replay currently supports at most 64 sites");
  TORCH_CHECK(site_rgba_f32.size(0) == sites_f32.size(0), "site RGBA count mismatch");
  TORCH_CHECK(row_index_i32.size(0) == ray_coeff_f32.size(0), "row_index_i32 length must match track count");
  TORCH_CHECK(
      candidate_row_offsets_i32.size(0) == (int64_t)config[4] * (int64_t)config[3] + 1,
      "candidate_row_offsets_i32 length must be row_count * time_slab_count + 1");
  TORCH_CHECK(
      candidate_depth_num_f32.size(0) == config[5],
      "candidate_depth_num_f32 row count must match candidate_count");
  TORCH_CHECK(
      candidate_depth_den_f16.size(0) == config[5],
      "candidate_depth_den_f16 row count must match candidate_count");

  auto output_rgb = torch::empty(
      {ray_coeff_f32.size(0), frame_t_f32.size(0), 3},
      ray_coeff_f32.options().dtype(torch::kFloat32));
  auto output_alpha = torch::empty(
      {ray_coeff_f32.size(0), frame_t_f32.size(0)},
      ray_coeff_f32.options().dtype(torch::kFloat32));
  auto output_depth = torch::empty(
      {ray_coeff_f32.size(0), frame_t_f32.size(0)},
      ray_coeff_f32.options().dtype(torch::kFloat32));
  auto& k = kernels();
  constexpr uint64_t threads = 256ull;
  const uint64_t total = (uint64_t)ray_coeff_f32.size(0) * (uint64_t)frame_t_f32.size(0);
  launch(k.fused_slab_affine_num32_den16_realray_rgba_depth_replay, [&](MetalKernelFunction& fn) {
    fn.setArg(0, row_index_i32);
    fn.setArg(1, candidate_row_offsets_i32);
    fn.setArg(2, candidate_depth_num_f32);
    fn.setArg(3, candidate_depth_den_f16);
    fn.setArg(4, sites_f32);
    fn.setArg(5, site_rgba_f32);
    fn.setArg(6, ray_coeff_f32);
    fn.setArg(7, frame_t_f32);
    fn.setArg(8, config_i32);
    fn.setArg(9, config_f32);
    fn.setArg(10, output_rgb);
    fn.setArg(11, output_alpha);
    fn.setArg(12, output_depth);
    fn.dispatch(total, threads);
  });
  return std::make_tuple(output_rgb, output_alpha, output_depth);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
metal_fused_slab_affine_num32_den16_ownerupdate_realray_rgba_depth_replay(
    const torch::Tensor& row_index_i32,
    const torch::Tensor& candidate_row_offsets_i32,
    const torch::Tensor& candidate_boundary_ids_i32,
    const torch::Tensor& candidate_depth_num_f32,
    const torch::Tensor& candidate_depth_den_f16,
    const torch::Tensor& boundary_site_pairs_i32,
    const torch::Tensor& sites_f32,
    const torch::Tensor& site_rgba_f32,
    const torch::Tensor& ray_coeff_f32,
    const torch::Tensor& frame_t_f32,
    const torch::Tensor& config_i32,
    const torch::Tensor& config_f32) {
  check_i32_mps_1d_any(row_index_i32, "row_index_i32");
  check_i32_mps_1d_any(candidate_row_offsets_i32, "candidate_row_offsets_i32");
  check_i32_mps_1d_any(candidate_boundary_ids_i32, "candidate_boundary_ids_i32");
  check_float_mps_2d(candidate_depth_num_f32, "candidate_depth_num_f32", 2);
  check_half_mps_2d(candidate_depth_den_f16, "candidate_depth_den_f16", 2);
  check_float_mps_2d(sites_f32, "sites_f32", 5);
  check_float_mps_2d(site_rgba_f32, "site_rgba_f32", 4);
  check_float_mps_2d(ray_coeff_f32, "ray_coeff_f32", 12);
  TORCH_CHECK(boundary_site_pairs_i32.device().is_mps(), "boundary_site_pairs_i32 must be on MPS");
  TORCH_CHECK(boundary_site_pairs_i32.scalar_type() == torch::kInt32, "boundary_site_pairs_i32 must be int32");
  TORCH_CHECK(boundary_site_pairs_i32.dim() == 2, "boundary_site_pairs_i32 must have shape [B,2]");
  TORCH_CHECK(boundary_site_pairs_i32.size(1) == 2, "boundary_site_pairs_i32 must have shape [B,2]");
  TORCH_CHECK(boundary_site_pairs_i32.is_contiguous(), "boundary_site_pairs_i32 must be contiguous");
  TORCH_CHECK(frame_t_f32.device().is_mps(), "frame_t_f32 must be on MPS");
  TORCH_CHECK(frame_t_f32.scalar_type() == torch::kFloat32, "frame_t_f32 must be float32");
  TORCH_CHECK(frame_t_f32.dim() == 1, "frame_t_f32 must have shape [T]");
  TORCH_CHECK(frame_t_f32.is_contiguous(), "frame_t_f32 must be contiguous");
  check_i32_mps_1d(config_i32, "config_i32", 7);
  TORCH_CHECK(config_f32.device().is_mps(), "config_f32 must be on MPS");
  TORCH_CHECK(config_f32.scalar_type() == torch::kFloat32, "config_f32 must be float32");
  TORCH_CHECK(config_f32.dim() == 1 && config_f32.size(0) == 4, "config_f32 must have shape [4]");
  TORCH_CHECK(config_f32.is_contiguous(), "config_f32 must be contiguous");

  auto config_i32_cpu = config_i32.cpu();
  const int32_t* config = config_i32_cpu.data_ptr<int32_t>();
  TORCH_CHECK(config[0] == ray_coeff_f32.size(0), "config_i32[0] must match track count");
  TORCH_CHECK(config[1] == sites_f32.size(0), "config_i32[1] must match site count");
  TORCH_CHECK(config[2] == frame_t_f32.size(0), "config_i32[2] must match frame count");
  TORCH_CHECK(config[3] > 0, "config_i32[3] time_slab_count must be positive");
  TORCH_CHECK(config[4] > 0, "config_i32[4] row_count must be positive");
  TORCH_CHECK(config[5] >= 0, "config_i32[5] candidate_count must be nonnegative");
  TORCH_CHECK(config[6] == boundary_site_pairs_i32.size(0), "config_i32[6] must match boundary count");
  TORCH_CHECK(ray_coeff_f32.size(0) > 0, "owner-update replay requires at least one track");
  TORCH_CHECK(frame_t_f32.size(0) > 0, "owner-update replay requires at least one frame");
  TORCH_CHECK(sites_f32.size(0) > 0, "owner-update replay requires at least one site");
  TORCH_CHECK(sites_f32.size(0) <= 64, "owner-update replay currently supports at most 64 sites");
  TORCH_CHECK(site_rgba_f32.size(0) == sites_f32.size(0), "site RGBA count mismatch");
  TORCH_CHECK(row_index_i32.size(0) == ray_coeff_f32.size(0), "row_index_i32 length must match track count");
  TORCH_CHECK(
      candidate_row_offsets_i32.size(0) == (int64_t)config[4] * (int64_t)config[3] + 1,
      "candidate_row_offsets_i32 length must be row_count * time_slab_count + 1");
  TORCH_CHECK(
      candidate_boundary_ids_i32.size(0) == config[5],
      "candidate_boundary_ids_i32 length must match candidate_count");
  TORCH_CHECK(
      candidate_depth_num_f32.size(0) == config[5],
      "candidate_depth_num_f32 row count must match candidate_count");
  TORCH_CHECK(
      candidate_depth_den_f16.size(0) == config[5],
      "candidate_depth_den_f16 row count must match candidate_count");

  auto output_rgb = torch::empty(
      {ray_coeff_f32.size(0), frame_t_f32.size(0), 3},
      ray_coeff_f32.options().dtype(torch::kFloat32));
  auto output_alpha = torch::empty(
      {ray_coeff_f32.size(0), frame_t_f32.size(0)},
      ray_coeff_f32.options().dtype(torch::kFloat32));
  auto output_depth = torch::empty(
      {ray_coeff_f32.size(0), frame_t_f32.size(0)},
      ray_coeff_f32.options().dtype(torch::kFloat32));
  auto& k = kernels();
  constexpr uint64_t threads = 256ull;
  const uint64_t total = (uint64_t)ray_coeff_f32.size(0) * (uint64_t)frame_t_f32.size(0);
  launch(k.fused_slab_affine_num32_den16_ownerupdate_realray_rgba_depth_replay, [&](MetalKernelFunction& fn) {
    fn.setArg(0, row_index_i32);
    fn.setArg(1, candidate_row_offsets_i32);
    fn.setArg(2, candidate_boundary_ids_i32);
    fn.setArg(3, candidate_depth_num_f32);
    fn.setArg(4, candidate_depth_den_f16);
    fn.setArg(5, boundary_site_pairs_i32);
    fn.setArg(6, sites_f32);
    fn.setArg(7, site_rgba_f32);
    fn.setArg(8, ray_coeff_f32);
    fn.setArg(9, frame_t_f32);
    fn.setArg(10, config_i32);
    fn.setArg(11, config_f32);
    fn.setArg(12, output_rgb);
    fn.setArg(13, output_alpha);
    fn.setArg(14, output_depth);
    fn.dispatch(total, threads);
  });
  return std::make_tuple(output_rgb, output_alpha, output_depth);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> metal_segment_tape_rgba_depth_replay(
    const torch::Tensor& segment_offsets_i32,
    const torch::Tensor& segment_owner_i32,
    const torch::Tensor& segment_length_f32,
    const torch::Tensor& segment_mid_f32,
    const torch::Tensor& site_rgba_f32,
    const torch::Tensor& config_i32,
    const torch::Tensor& config_f32) {
  check_i32_mps_1d_any(segment_offsets_i32, "segment_offsets_i32");
  check_i32_mps_1d_any(segment_owner_i32, "segment_owner_i32");
  check_float_mps_1d_any(segment_length_f32, "segment_length_f32");
  check_float_mps_1d_any(segment_mid_f32, "segment_mid_f32");
  check_float_mps_2d(site_rgba_f32, "site_rgba_f32", 4);
  check_i32_mps_1d(config_i32, "config_i32", 4);
  TORCH_CHECK(config_f32.device().is_mps(), "config_f32 must be on MPS");
  TORCH_CHECK(config_f32.scalar_type() == torch::kFloat32, "config_f32 must be float32");
  TORCH_CHECK(config_f32.dim() == 1 && config_f32.size(0) == 2, "config_f32 must have shape [2]");
  TORCH_CHECK(config_f32.is_contiguous(), "config_f32 must be contiguous");

  auto config_i32_cpu = config_i32.cpu();
  const int32_t* config = config_i32_cpu.data_ptr<int32_t>();
  const int64_t track_count = config[0];
  const int64_t frame_count = config[1];
  const int64_t site_count = config[2];
  const int64_t segment_count = config[3];
  TORCH_CHECK(track_count > 0, "config_i32[0] track count must be positive");
  TORCH_CHECK(frame_count > 0, "config_i32[1] frame count must be positive");
  TORCH_CHECK(site_count == site_rgba_f32.size(0), "config_i32[2] must match site_rgba_f32 rows");
  TORCH_CHECK(site_count > 0 && site_count <= 64, "segment tape replay supports site count in [1, 64]");
  TORCH_CHECK(segment_count >= 0, "config_i32[3] segment count must be nonnegative");
  TORCH_CHECK(segment_owner_i32.size(0) == segment_count, "segment_owner_i32 length must match segment count");
  TORCH_CHECK(segment_length_f32.size(0) == segment_count, "segment_length_f32 length must match segment count");
  TORCH_CHECK(segment_mid_f32.size(0) == segment_count, "segment_mid_f32 length must match segment count");
  const int64_t sample_count = track_count * frame_count;
  TORCH_CHECK(segment_offsets_i32.size(0) == sample_count + 1, "segment_offsets_i32 length must be sample_count + 1");
  check_segment_tape_offsets_cpu(segment_offsets_i32, sample_count, segment_count, 2147483647);

  auto output_rgb = torch::empty({track_count, frame_count, 3}, site_rgba_f32.options().dtype(torch::kFloat32));
  auto output_alpha = torch::empty({track_count, frame_count}, site_rgba_f32.options().dtype(torch::kFloat32));
  auto output_depth = torch::empty({track_count, frame_count}, site_rgba_f32.options().dtype(torch::kFloat32));
  auto& k = kernels();
  constexpr uint64_t threads = 256ull;
  launch(k.segment_tape_rgba_depth_replay, [&](MetalKernelFunction& fn) {
    fn.setArg(0, segment_offsets_i32);
    fn.setArg(1, segment_owner_i32);
    fn.setArg(2, segment_length_f32);
    fn.setArg(3, segment_mid_f32);
    fn.setArg(4, site_rgba_f32);
    fn.setArg(5, config_i32);
    fn.setArg(6, config_f32);
    fn.setArg(7, output_rgb);
    fn.setArg(8, output_alpha);
    fn.setArg(9, output_depth);
    fn.dispatch((uint64_t)sample_count, threads);
  });
  return std::make_tuple(output_rgb, output_alpha, output_depth);
}

torch::Tensor metal_segment_tape_vjp_direct_atomic_grad_only(
    const torch::Tensor& segment_offsets_i32,
    const torch::Tensor& segment_owner_i32,
    const torch::Tensor& segment_length_f32,
    const torch::Tensor& segment_mid_f32,
    const torch::Tensor& site_rgba_f32,
    const torch::Tensor& grad_rgb_f32,
    const torch::Tensor& grad_alpha_f32,
    const torch::Tensor& grad_depth_f32,
    const torch::Tensor& config_i32,
    const torch::Tensor& config_f32) {
  check_i32_mps_1d_any(segment_offsets_i32, "segment_offsets_i32");
  check_i32_mps_1d_any(segment_owner_i32, "segment_owner_i32");
  check_float_mps_1d_any(segment_length_f32, "segment_length_f32");
  check_float_mps_1d_any(segment_mid_f32, "segment_mid_f32");
  check_float_mps_2d(site_rgba_f32, "site_rgba_f32", 4);
  TORCH_CHECK(grad_rgb_f32.device().is_mps(), "grad_rgb_f32 must be on MPS");
  TORCH_CHECK(grad_rgb_f32.scalar_type() == torch::kFloat32, "grad_rgb_f32 must be float32");
  TORCH_CHECK(grad_rgb_f32.dim() == 3 && grad_rgb_f32.size(2) == 3, "grad_rgb_f32 must have shape [K,T,3]");
  TORCH_CHECK(grad_rgb_f32.is_contiguous(), "grad_rgb_f32 must be contiguous");
  TORCH_CHECK(grad_alpha_f32.device().is_mps(), "grad_alpha_f32 must be on MPS");
  TORCH_CHECK(grad_alpha_f32.scalar_type() == torch::kFloat32, "grad_alpha_f32 must be float32");
  TORCH_CHECK(grad_alpha_f32.dim() == 2, "grad_alpha_f32 must have shape [K,T]");
  TORCH_CHECK(grad_alpha_f32.is_contiguous(), "grad_alpha_f32 must be contiguous");
  TORCH_CHECK(grad_depth_f32.device().is_mps(), "grad_depth_f32 must be on MPS");
  TORCH_CHECK(grad_depth_f32.scalar_type() == torch::kFloat32, "grad_depth_f32 must be float32");
  TORCH_CHECK(grad_depth_f32.dim() == 2, "grad_depth_f32 must have shape [K,T]");
  TORCH_CHECK(grad_depth_f32.is_contiguous(), "grad_depth_f32 must be contiguous");
  check_i32_mps_1d(config_i32, "config_i32", 4);
  TORCH_CHECK(config_f32.device().is_mps(), "config_f32 must be on MPS");
  TORCH_CHECK(config_f32.scalar_type() == torch::kFloat32, "config_f32 must be float32");
  TORCH_CHECK(config_f32.dim() == 1 && config_f32.size(0) == 2, "config_f32 must have shape [2]");
  TORCH_CHECK(config_f32.is_contiguous(), "config_f32 must be contiguous");

  auto config_i32_cpu = config_i32.cpu();
  const int32_t* config = config_i32_cpu.data_ptr<int32_t>();
  const int64_t track_count = config[0];
  const int64_t frame_count = config[1];
  const int64_t site_count = config[2];
  const int64_t segment_count = config[3];
  TORCH_CHECK(track_count > 0, "config_i32[0] track count must be positive");
  TORCH_CHECK(frame_count > 0, "config_i32[1] frame count must be positive");
  TORCH_CHECK(site_count == site_rgba_f32.size(0), "config_i32[2] must match site_rgba_f32 rows");
  TORCH_CHECK(site_count > 0 && site_count <= 64, "segment tape VJP supports site count in [1, 64]");
  TORCH_CHECK(segment_count >= 0, "config_i32[3] segment count must be nonnegative");
  TORCH_CHECK(segment_owner_i32.size(0) == segment_count, "segment_owner_i32 length must match segment count");
  TORCH_CHECK(segment_length_f32.size(0) == segment_count, "segment_length_f32 length must match segment count");
  TORCH_CHECK(segment_mid_f32.size(0) == segment_count, "segment_mid_f32 length must match segment count");
  const int64_t sample_count = track_count * frame_count;
  TORCH_CHECK(segment_offsets_i32.size(0) == sample_count + 1, "segment_offsets_i32 length must be sample_count + 1");
  TORCH_CHECK(
      grad_rgb_f32.size(0) == track_count && grad_rgb_f32.size(1) == frame_count,
      "grad_rgb_f32 shape mismatch");
  TORCH_CHECK(
      grad_alpha_f32.size(0) == track_count && grad_alpha_f32.size(1) == frame_count,
      "grad_alpha_f32 shape mismatch");
  TORCH_CHECK(
      grad_depth_f32.size(0) == track_count && grad_depth_f32.size(1) == frame_count,
      "grad_depth_f32 shape mismatch");
  check_segment_tape_offsets_cpu(segment_offsets_i32, sample_count, segment_count, 129);

  auto grad_site_rgba = torch::empty({site_count, 4}, site_rgba_f32.options().dtype(torch::kFloat32));
  auto& k = kernels();
  constexpr uint64_t threads = 256ull;
  launch(k.clear_site_rgba_grad, [&](MetalKernelFunction& fn) {
    fn.setArg(0, grad_site_rgba);
    fn.setArg(1, config_i32);
    fn.dispatch((uint64_t)site_count, threads);
  });
  launch(k.segment_tape_vjp_direct_atomic_grad_only, [&](MetalKernelFunction& fn) {
    fn.setArg(0, segment_offsets_i32);
    fn.setArg(1, segment_owner_i32);
    fn.setArg(2, segment_length_f32);
    fn.setArg(3, segment_mid_f32);
    fn.setArg(4, site_rgba_f32);
    fn.setArg(5, grad_rgb_f32);
    fn.setArg(6, grad_alpha_f32);
    fn.setArg(7, grad_depth_f32);
    fn.setArg(8, config_i32);
    fn.setArg(9, config_f32);
    fn.setArg(10, grad_site_rgba);
    fn.dispatch((uint64_t)sample_count, threads);
  });
  return grad_site_rgba;
}

std::tuple<torch::Tensor, torch::Tensor> metal_segment_tape_mse_vjp_direct_atomic_rgb_only(
    const torch::Tensor& segment_offsets_i32,
    const torch::Tensor& segment_owner_i32,
    const torch::Tensor& segment_length_f32,
    const torch::Tensor& site_rgba_f32,
    const torch::Tensor& target_rgb_f32,
    const torch::Tensor& config_i32,
    const torch::Tensor& config_f32) {
  check_i32_mps_1d_any(segment_offsets_i32, "segment_offsets_i32");
  check_i32_mps_1d_any(segment_owner_i32, "segment_owner_i32");
  check_float_mps_1d_any(segment_length_f32, "segment_length_f32");
  check_float_mps_2d(site_rgba_f32, "site_rgba_f32", 4);
  TORCH_CHECK(target_rgb_f32.device().is_mps(), "target_rgb_f32 must be on MPS");
  TORCH_CHECK(target_rgb_f32.scalar_type() == torch::kFloat32, "target_rgb_f32 must be float32");
  TORCH_CHECK(target_rgb_f32.dim() == 3 && target_rgb_f32.size(2) == 3, "target_rgb_f32 must have shape [K,T,3]");
  TORCH_CHECK(target_rgb_f32.is_contiguous(), "target_rgb_f32 must be contiguous");
  check_i32_mps_1d(config_i32, "config_i32", 4);
  TORCH_CHECK(config_f32.device().is_mps(), "config_f32 must be on MPS");
  TORCH_CHECK(config_f32.scalar_type() == torch::kFloat32, "config_f32 must be float32");
  TORCH_CHECK(config_f32.dim() == 1 && config_f32.size(0) == 2, "config_f32 must have shape [2]");
  TORCH_CHECK(config_f32.is_contiguous(), "config_f32 must be contiguous");

  auto config_i32_cpu = config_i32.cpu();
  const int32_t* config = config_i32_cpu.data_ptr<int32_t>();
  const int64_t track_count = config[0];
  const int64_t frame_count = config[1];
  const int64_t site_count = config[2];
  const int64_t segment_count = config[3];
  TORCH_CHECK(track_count > 0, "config_i32[0] track count must be positive");
  TORCH_CHECK(frame_count > 0, "config_i32[1] frame count must be positive");
  TORCH_CHECK(site_count == site_rgba_f32.size(0), "config_i32[2] must match site_rgba_f32 rows");
  TORCH_CHECK(site_count > 0 && site_count <= 64, "segment tape fused MSE supports site count in [1, 64]");
  TORCH_CHECK(segment_count >= 0, "config_i32[3] segment count must be nonnegative");
  TORCH_CHECK(segment_owner_i32.size(0) == segment_count, "segment_owner_i32 length must match segment count");
  TORCH_CHECK(segment_length_f32.size(0) == segment_count, "segment_length_f32 length must match segment count");
  const int64_t sample_count = track_count * frame_count;
  TORCH_CHECK(segment_offsets_i32.size(0) == sample_count + 1, "segment_offsets_i32 length must be sample_count + 1");
  TORCH_CHECK(
      target_rgb_f32.size(0) == track_count && target_rgb_f32.size(1) == frame_count,
      "target_rgb_f32 shape mismatch");
  check_segment_tape_offsets_cpu(segment_offsets_i32, sample_count, segment_count, 129);

  auto loss = torch::empty({1}, site_rgba_f32.options().dtype(torch::kFloat32));
  auto grad_site_rgba = torch::empty({site_count, 4}, site_rgba_f32.options().dtype(torch::kFloat32));
  auto& k = kernels();
  constexpr uint64_t threads = 256ull;
  launch(k.clear_affine_loss_site_rgba_grad, [&](MetalKernelFunction& fn) {
    fn.setArg(0, loss);
    fn.setArg(1, grad_site_rgba);
    fn.setArg(2, config_i32);
    fn.dispatch((uint64_t)std::max<int64_t>(site_count, 1), threads);
  });
  launch(k.segment_tape_mse_vjp_direct_atomic_rgb_only, [&](MetalKernelFunction& fn) {
    fn.setArg(0, segment_offsets_i32);
    fn.setArg(1, segment_owner_i32);
    fn.setArg(2, segment_length_f32);
    fn.setArg(3, site_rgba_f32);
    fn.setArg(4, target_rgb_f32);
    fn.setArg(5, config_i32);
    fn.setArg(6, config_f32);
    fn.setArg(7, loss);
    fn.setArg(8, grad_site_rgba);
    fn.dispatch((uint64_t)sample_count, threads);
  });
  return std::make_tuple(loss, grad_site_rgba);
}

torch::Tensor metal_segment_tape_vjp_direct_atomic_track(
    const torch::Tensor& segment_offsets_i32,
    const torch::Tensor& segment_owner_i32,
    const torch::Tensor& segment_length_f32,
    const torch::Tensor& segment_mid_f32,
    const torch::Tensor& site_rgba_f32,
    const torch::Tensor& grad_rgb_f32,
    const torch::Tensor& grad_alpha_f32,
    const torch::Tensor& grad_depth_f32,
    const torch::Tensor& config_i32,
    const torch::Tensor& config_f32) {
  check_i32_mps_1d_any(segment_offsets_i32, "segment_offsets_i32");
  check_i32_mps_1d_any(segment_owner_i32, "segment_owner_i32");
  check_float_mps_1d_any(segment_length_f32, "segment_length_f32");
  check_float_mps_1d_any(segment_mid_f32, "segment_mid_f32");
  check_float_mps_2d(site_rgba_f32, "site_rgba_f32", 4);
  TORCH_CHECK(grad_rgb_f32.device().is_mps(), "grad_rgb_f32 must be on MPS");
  TORCH_CHECK(grad_rgb_f32.scalar_type() == torch::kFloat32, "grad_rgb_f32 must be float32");
  TORCH_CHECK(grad_rgb_f32.dim() == 3 && grad_rgb_f32.size(2) == 3, "grad_rgb_f32 must have shape [K,T,3]");
  TORCH_CHECK(grad_rgb_f32.is_contiguous(), "grad_rgb_f32 must be contiguous");
  TORCH_CHECK(grad_alpha_f32.device().is_mps(), "grad_alpha_f32 must be on MPS");
  TORCH_CHECK(grad_alpha_f32.scalar_type() == torch::kFloat32, "grad_alpha_f32 must be float32");
  TORCH_CHECK(grad_alpha_f32.dim() == 2, "grad_alpha_f32 must have shape [K,T]");
  TORCH_CHECK(grad_alpha_f32.is_contiguous(), "grad_alpha_f32 must be contiguous");
  TORCH_CHECK(grad_depth_f32.device().is_mps(), "grad_depth_f32 must be on MPS");
  TORCH_CHECK(grad_depth_f32.scalar_type() == torch::kFloat32, "grad_depth_f32 must be float32");
  TORCH_CHECK(grad_depth_f32.dim() == 2, "grad_depth_f32 must have shape [K,T]");
  TORCH_CHECK(grad_depth_f32.is_contiguous(), "grad_depth_f32 must be contiguous");
  check_i32_mps_1d(config_i32, "config_i32", 4);
  TORCH_CHECK(config_f32.device().is_mps(), "config_f32 must be on MPS");
  TORCH_CHECK(config_f32.scalar_type() == torch::kFloat32, "config_f32 must be float32");
  TORCH_CHECK(config_f32.dim() == 1 && config_f32.size(0) == 2, "config_f32 must have shape [2]");
  TORCH_CHECK(config_f32.is_contiguous(), "config_f32 must be contiguous");

  auto config_i32_cpu = config_i32.cpu();
  const int32_t* config = config_i32_cpu.data_ptr<int32_t>();
  const int64_t track_count = config[0];
  const int64_t frame_count = config[1];
  const int64_t site_count = config[2];
  const int64_t segment_count = config[3];
  TORCH_CHECK(track_count > 0, "config_i32[0] track count must be positive");
  TORCH_CHECK(frame_count > 0, "config_i32[1] frame count must be positive");
  TORCH_CHECK(site_count == site_rgba_f32.size(0), "config_i32[2] must match site_rgba_f32 rows");
  TORCH_CHECK(site_count > 0 && site_count <= 64, "segment tape track VJP supports site count in [1, 64]");
  TORCH_CHECK(segment_count >= 0, "config_i32[3] segment count must be nonnegative");
  TORCH_CHECK(segment_owner_i32.size(0) == segment_count, "segment_owner_i32 length must match segment count");
  TORCH_CHECK(segment_length_f32.size(0) == segment_count, "segment_length_f32 length must match segment count");
  TORCH_CHECK(segment_mid_f32.size(0) == segment_count, "segment_mid_f32 length must match segment count");
  const int64_t sample_count = track_count * frame_count;
  TORCH_CHECK(segment_offsets_i32.size(0) == sample_count + 1, "segment_offsets_i32 length must be sample_count + 1");
  TORCH_CHECK(
      grad_rgb_f32.size(0) == track_count && grad_rgb_f32.size(1) == frame_count,
      "grad_rgb_f32 shape mismatch");
  TORCH_CHECK(
      grad_alpha_f32.size(0) == track_count && grad_alpha_f32.size(1) == frame_count,
      "grad_alpha_f32 shape mismatch");
  TORCH_CHECK(
      grad_depth_f32.size(0) == track_count && grad_depth_f32.size(1) == frame_count,
      "grad_depth_f32 shape mismatch");
  check_segment_tape_offsets_cpu(segment_offsets_i32, sample_count, segment_count, 129);

  auto grad_site_rgba = torch::empty({site_count, 4}, site_rgba_f32.options().dtype(torch::kFloat32));
  auto& k = kernels();
  constexpr uint64_t threads = 256ull;
  launch(k.clear_site_rgba_grad, [&](MetalKernelFunction& fn) {
    fn.setArg(0, grad_site_rgba);
    fn.setArg(1, config_i32);
    fn.dispatch((uint64_t)site_count, threads);
  });
  launch(k.segment_tape_vjp_direct_atomic_track, [&](MetalKernelFunction& fn) {
    fn.setArg(0, segment_offsets_i32);
    fn.setArg(1, segment_owner_i32);
    fn.setArg(2, segment_length_f32);
    fn.setArg(3, segment_mid_f32);
    fn.setArg(4, site_rgba_f32);
    fn.setArg(5, grad_rgb_f32);
    fn.setArg(6, grad_alpha_f32);
    fn.setArg(7, grad_depth_f32);
    fn.setArg(8, config_i32);
    fn.setArg(9, config_f32);
    fn.setArg(10, grad_site_rgba);
    fn.dispatch((uint64_t)track_count, threads);
  });
  return grad_site_rgba;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> metal_endpoint_run_rgba_depth_replay(
    const torch::Tensor& run_offsets_i32,
    const torch::Tensor& run_owner_i32,
    const torch::Tensor& run_start_f32,
    const torch::Tensor& run_end_f32,
    const torch::Tensor& site_rgba_f32,
    const torch::Tensor& config_i32,
    const torch::Tensor& config_f32) {
  check_i32_mps_1d_any(run_offsets_i32, "run_offsets_i32");
  check_i32_mps_1d_any(run_owner_i32, "run_owner_i32");
  check_float_mps_1d_any(run_start_f32, "run_start_f32");
  check_float_mps_1d_any(run_end_f32, "run_end_f32");
  check_float_mps_2d(site_rgba_f32, "site_rgba_f32", 4);
  check_i32_mps_1d(config_i32, "config_i32", 4);
  TORCH_CHECK(config_f32.device().is_mps(), "config_f32 must be on MPS");
  TORCH_CHECK(config_f32.scalar_type() == torch::kFloat32, "config_f32 must be float32");
  TORCH_CHECK(config_f32.dim() == 1 && config_f32.size(0) == 2, "config_f32 must have shape [2]");
  TORCH_CHECK(config_f32.is_contiguous(), "config_f32 must be contiguous");

  auto config_i32_cpu = config_i32.cpu();
  const int32_t* config = config_i32_cpu.data_ptr<int32_t>();
  const int64_t track_count = config[0];
  const int64_t frame_count = config[1];
  const int64_t site_count = config[2];
  const int64_t run_count = config[3];
  TORCH_CHECK(track_count > 0, "config_i32[0] track count must be positive");
  TORCH_CHECK(frame_count > 0, "config_i32[1] frame count must be positive");
  TORCH_CHECK(site_count == site_rgba_f32.size(0), "config_i32[2] must match site_rgba_f32 rows");
  TORCH_CHECK(site_count > 0 && site_count <= 64, "endpoint run replay supports site count in [1, 64]");
  TORCH_CHECK(run_count >= 0, "config_i32[3] run count must be nonnegative");
  TORCH_CHECK(run_owner_i32.size(0) == run_count, "run_owner_i32 length must match run count");
  TORCH_CHECK(run_start_f32.size(0) == run_count, "run_start_f32 length must match run count");
  TORCH_CHECK(run_end_f32.size(0) == run_count, "run_end_f32 length must match run count");
  const int64_t sample_count = track_count * frame_count;
  TORCH_CHECK(run_offsets_i32.size(0) == sample_count + 1, "run_offsets_i32 length must be sample_count + 1");
  check_segment_tape_offsets_cpu(run_offsets_i32, sample_count, run_count, 2147483647);

  auto output_rgb = torch::empty({track_count, frame_count, 3}, site_rgba_f32.options().dtype(torch::kFloat32));
  auto output_alpha = torch::empty({track_count, frame_count}, site_rgba_f32.options().dtype(torch::kFloat32));
  auto output_depth = torch::empty({track_count, frame_count}, site_rgba_f32.options().dtype(torch::kFloat32));
  auto& k = kernels();
  constexpr uint64_t threads = 256ull;
  launch(k.endpoint_run_rgba_depth_replay, [&](MetalKernelFunction& fn) {
    fn.setArg(0, run_offsets_i32);
    fn.setArg(1, run_owner_i32);
    fn.setArg(2, run_start_f32);
    fn.setArg(3, run_end_f32);
    fn.setArg(4, site_rgba_f32);
    fn.setArg(5, config_i32);
    fn.setArg(6, config_f32);
    fn.setArg(7, output_rgb);
    fn.setArg(8, output_alpha);
    fn.setArg(9, output_depth);
    fn.dispatch((uint64_t)sample_count, threads);
  });
  return std::make_tuple(output_rgb, output_alpha, output_depth);
}

torch::Tensor metal_endpoint_run_vjp_direct_atomic_grad_only(
    const torch::Tensor& run_offsets_i32,
    const torch::Tensor& run_owner_i32,
    const torch::Tensor& run_start_f32,
    const torch::Tensor& run_end_f32,
    const torch::Tensor& site_rgba_f32,
    const torch::Tensor& grad_rgb_f32,
    const torch::Tensor& grad_alpha_f32,
    const torch::Tensor& grad_depth_f32,
    const torch::Tensor& config_i32,
    const torch::Tensor& config_f32) {
  check_i32_mps_1d_any(run_offsets_i32, "run_offsets_i32");
  check_i32_mps_1d_any(run_owner_i32, "run_owner_i32");
  check_float_mps_1d_any(run_start_f32, "run_start_f32");
  check_float_mps_1d_any(run_end_f32, "run_end_f32");
  check_float_mps_2d(site_rgba_f32, "site_rgba_f32", 4);
  TORCH_CHECK(grad_rgb_f32.device().is_mps(), "grad_rgb_f32 must be on MPS");
  TORCH_CHECK(grad_rgb_f32.scalar_type() == torch::kFloat32, "grad_rgb_f32 must be float32");
  TORCH_CHECK(grad_rgb_f32.dim() == 3 && grad_rgb_f32.size(2) == 3, "grad_rgb_f32 must have shape [K,T,3]");
  TORCH_CHECK(grad_rgb_f32.is_contiguous(), "grad_rgb_f32 must be contiguous");
  TORCH_CHECK(grad_alpha_f32.device().is_mps(), "grad_alpha_f32 must be on MPS");
  TORCH_CHECK(grad_alpha_f32.scalar_type() == torch::kFloat32, "grad_alpha_f32 must be float32");
  TORCH_CHECK(grad_alpha_f32.dim() == 2, "grad_alpha_f32 must have shape [K,T]");
  TORCH_CHECK(grad_alpha_f32.is_contiguous(), "grad_alpha_f32 must be contiguous");
  TORCH_CHECK(grad_depth_f32.device().is_mps(), "grad_depth_f32 must be on MPS");
  TORCH_CHECK(grad_depth_f32.scalar_type() == torch::kFloat32, "grad_depth_f32 must be float32");
  TORCH_CHECK(grad_depth_f32.dim() == 2, "grad_depth_f32 must have shape [K,T]");
  TORCH_CHECK(grad_depth_f32.is_contiguous(), "grad_depth_f32 must be contiguous");
  check_i32_mps_1d(config_i32, "config_i32", 4);
  TORCH_CHECK(config_f32.device().is_mps(), "config_f32 must be on MPS");
  TORCH_CHECK(config_f32.scalar_type() == torch::kFloat32, "config_f32 must be float32");
  TORCH_CHECK(config_f32.dim() == 1 && config_f32.size(0) == 2, "config_f32 must have shape [2]");
  TORCH_CHECK(config_f32.is_contiguous(), "config_f32 must be contiguous");

  auto config_i32_cpu = config_i32.cpu();
  const int32_t* config = config_i32_cpu.data_ptr<int32_t>();
  const int64_t track_count = config[0];
  const int64_t frame_count = config[1];
  const int64_t site_count = config[2];
  const int64_t run_count = config[3];
  TORCH_CHECK(track_count > 0, "config_i32[0] track count must be positive");
  TORCH_CHECK(frame_count > 0, "config_i32[1] frame count must be positive");
  TORCH_CHECK(site_count == site_rgba_f32.size(0), "config_i32[2] must match site_rgba_f32 rows");
  TORCH_CHECK(site_count > 0 && site_count <= 64, "endpoint run VJP supports site count in [1, 64]");
  TORCH_CHECK(run_count >= 0, "config_i32[3] run count must be nonnegative");
  TORCH_CHECK(run_owner_i32.size(0) == run_count, "run_owner_i32 length must match run count");
  TORCH_CHECK(run_start_f32.size(0) == run_count, "run_start_f32 length must match run count");
  TORCH_CHECK(run_end_f32.size(0) == run_count, "run_end_f32 length must match run count");
  const int64_t sample_count = track_count * frame_count;
  TORCH_CHECK(run_offsets_i32.size(0) == sample_count + 1, "run_offsets_i32 length must be sample_count + 1");
  TORCH_CHECK(
      grad_rgb_f32.size(0) == track_count && grad_rgb_f32.size(1) == frame_count,
      "grad_rgb_f32 shape mismatch");
  TORCH_CHECK(
      grad_alpha_f32.size(0) == track_count && grad_alpha_f32.size(1) == frame_count,
      "grad_alpha_f32 shape mismatch");
  TORCH_CHECK(
      grad_depth_f32.size(0) == track_count && grad_depth_f32.size(1) == frame_count,
      "grad_depth_f32 shape mismatch");
  check_segment_tape_offsets_cpu(run_offsets_i32, sample_count, run_count, 129);

  auto grad_site_rgba = torch::empty({site_count, 4}, site_rgba_f32.options().dtype(torch::kFloat32));
  auto& k = kernels();
  constexpr uint64_t threads = 256ull;
  launch(k.clear_site_rgba_grad, [&](MetalKernelFunction& fn) {
    fn.setArg(0, grad_site_rgba);
    fn.setArg(1, config_i32);
    fn.dispatch((uint64_t)site_count, threads);
  });
  launch(k.endpoint_run_vjp_direct_atomic_grad_only, [&](MetalKernelFunction& fn) {
    fn.setArg(0, run_offsets_i32);
    fn.setArg(1, run_owner_i32);
    fn.setArg(2, run_start_f32);
    fn.setArg(3, run_end_f32);
    fn.setArg(4, site_rgba_f32);
    fn.setArg(5, grad_rgb_f32);
    fn.setArg(6, grad_alpha_f32);
    fn.setArg(7, grad_depth_f32);
    fn.setArg(8, config_i32);
    fn.setArg(9, config_f32);
    fn.setArg(10, grad_site_rgba);
    fn.dispatch((uint64_t)sample_count, threads);
  });
  return grad_site_rgba;
}

std::tuple<torch::Tensor, torch::Tensor> metal_endpoint_run_mse_vjp_direct_atomic_rgb_only(
    const torch::Tensor& run_offsets_i32,
    const torch::Tensor& run_owner_i32,
    const torch::Tensor& run_start_f32,
    const torch::Tensor& run_end_f32,
    const torch::Tensor& site_rgba_f32,
    const torch::Tensor& target_rgb_f32,
    const torch::Tensor& config_i32,
    const torch::Tensor& config_f32) {
  check_i32_mps_1d_any(run_offsets_i32, "run_offsets_i32");
  check_i32_mps_1d_any(run_owner_i32, "run_owner_i32");
  check_float_mps_1d_any(run_start_f32, "run_start_f32");
  check_float_mps_1d_any(run_end_f32, "run_end_f32");
  check_float_mps_2d(site_rgba_f32, "site_rgba_f32", 4);
  TORCH_CHECK(target_rgb_f32.device().is_mps(), "target_rgb_f32 must be on MPS");
  TORCH_CHECK(target_rgb_f32.scalar_type() == torch::kFloat32, "target_rgb_f32 must be float32");
  TORCH_CHECK(target_rgb_f32.dim() == 3 && target_rgb_f32.size(2) == 3, "target_rgb_f32 must have shape [K,T,3]");
  TORCH_CHECK(target_rgb_f32.is_contiguous(), "target_rgb_f32 must be contiguous");
  check_i32_mps_1d(config_i32, "config_i32", 4);
  TORCH_CHECK(config_f32.device().is_mps(), "config_f32 must be on MPS");
  TORCH_CHECK(config_f32.scalar_type() == torch::kFloat32, "config_f32 must be float32");
  TORCH_CHECK(config_f32.dim() == 1 && config_f32.size(0) == 2, "config_f32 must have shape [2]");
  TORCH_CHECK(config_f32.is_contiguous(), "config_f32 must be contiguous");

  auto config_i32_cpu = config_i32.cpu();
  const int32_t* config = config_i32_cpu.data_ptr<int32_t>();
  const int64_t track_count = config[0];
  const int64_t frame_count = config[1];
  const int64_t site_count = config[2];
  const int64_t run_count = config[3];
  TORCH_CHECK(track_count > 0, "config_i32[0] track count must be positive");
  TORCH_CHECK(frame_count > 0, "config_i32[1] frame count must be positive");
  TORCH_CHECK(site_count == site_rgba_f32.size(0), "config_i32[2] must match site_rgba_f32 rows");
  TORCH_CHECK(site_count > 0 && site_count <= 64, "endpoint run fused MSE supports site count in [1, 64]");
  TORCH_CHECK(run_count >= 0, "config_i32[3] run count must be nonnegative");
  TORCH_CHECK(run_owner_i32.size(0) == run_count, "run_owner_i32 length must match run count");
  TORCH_CHECK(run_start_f32.size(0) == run_count, "run_start_f32 length must match run count");
  TORCH_CHECK(run_end_f32.size(0) == run_count, "run_end_f32 length must match run count");
  const int64_t sample_count = track_count * frame_count;
  TORCH_CHECK(run_offsets_i32.size(0) == sample_count + 1, "run_offsets_i32 length must be sample_count + 1");
  TORCH_CHECK(
      target_rgb_f32.size(0) == track_count && target_rgb_f32.size(1) == frame_count,
      "target_rgb_f32 shape mismatch");
  check_segment_tape_offsets_cpu(run_offsets_i32, sample_count, run_count, 129);

  auto loss = torch::empty({1}, site_rgba_f32.options().dtype(torch::kFloat32));
  auto grad_site_rgba = torch::empty({site_count, 4}, site_rgba_f32.options().dtype(torch::kFloat32));
  auto& k = kernels();
  constexpr uint64_t threads = 256ull;
  launch(k.clear_affine_loss_site_rgba_grad, [&](MetalKernelFunction& fn) {
    fn.setArg(0, loss);
    fn.setArg(1, grad_site_rgba);
    fn.setArg(2, config_i32);
    fn.dispatch((uint64_t)std::max<int64_t>(site_count, 1), threads);
  });
  launch(k.endpoint_run_mse_vjp_direct_atomic_rgb_only, [&](MetalKernelFunction& fn) {
    fn.setArg(0, run_offsets_i32);
    fn.setArg(1, run_owner_i32);
    fn.setArg(2, run_start_f32);
    fn.setArg(3, run_end_f32);
    fn.setArg(4, site_rgba_f32);
    fn.setArg(5, target_rgb_f32);
    fn.setArg(6, config_i32);
    fn.setArg(7, config_f32);
    fn.setArg(8, loss);
    fn.setArg(9, grad_site_rgba);
    fn.dispatch((uint64_t)sample_count, threads);
  });
  return std::make_tuple(loss, grad_site_rgba);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> metal_endpoint_delta_replace_rgba_depth_replay(
    const torch::Tensor& base_offsets_i32,
    const torch::Tensor& base_owner_i32,
    const torch::Tensor& base_start_f32,
    const torch::Tensor& base_end_f32,
    const torch::Tensor& track_change_offsets_i32,
    const torch::Tensor& change_frame_i32,
    const torch::Tensor& change_offsets_i32,
    const torch::Tensor& change_owner_i32,
    const torch::Tensor& change_start_f32,
    const torch::Tensor& change_end_f32,
    const torch::Tensor& site_rgba_f32,
    const torch::Tensor& config_i32,
    const torch::Tensor& config_f32) {
  check_i32_mps_1d_any(base_offsets_i32, "base_offsets_i32");
  check_i32_mps_1d_any(base_owner_i32, "base_owner_i32");
  check_float_mps_1d_any(base_start_f32, "base_start_f32");
  check_float_mps_1d_any(base_end_f32, "base_end_f32");
  check_i32_mps_1d_any(track_change_offsets_i32, "track_change_offsets_i32");
  check_i32_mps_1d_any(change_frame_i32, "change_frame_i32");
  check_i32_mps_1d_any(change_offsets_i32, "change_offsets_i32");
  check_i32_mps_1d_any(change_owner_i32, "change_owner_i32");
  check_float_mps_1d_any(change_start_f32, "change_start_f32");
  check_float_mps_1d_any(change_end_f32, "change_end_f32");
  check_float_mps_2d(site_rgba_f32, "site_rgba_f32", 4);
  check_i32_mps_1d(config_i32, "config_i32", 6);
  TORCH_CHECK(config_f32.device().is_mps(), "config_f32 must be on MPS");
  TORCH_CHECK(config_f32.scalar_type() == torch::kFloat32, "config_f32 must be float32");
  TORCH_CHECK(config_f32.dim() == 1 && config_f32.size(0) == 2, "config_f32 must have shape [2]");
  TORCH_CHECK(config_f32.is_contiguous(), "config_f32 must be contiguous");

  auto config_i32_cpu = config_i32.cpu();
  const int32_t* config = config_i32_cpu.data_ptr<int32_t>();
  const int64_t track_count = config[0];
  const int64_t frame_count = config[1];
  const int64_t site_count = config[2];
  const int64_t base_record_count = config[3];
  const int64_t change_count = config[4];
  const int64_t change_record_count = config[5];
  TORCH_CHECK(track_count > 0, "config_i32[0] track count must be positive");
  TORCH_CHECK(frame_count > 0, "config_i32[1] frame count must be positive");
  TORCH_CHECK(site_count == site_rgba_f32.size(0), "config_i32[2] must match site_rgba_f32 rows");
  TORCH_CHECK(site_count > 0 && site_count <= 64, "endpoint delta replay supports site count in [1, 64]");
  TORCH_CHECK(base_record_count >= 0, "config_i32[3] base record count must be nonnegative");
  TORCH_CHECK(change_count >= 0, "config_i32[4] change count must be nonnegative");
  TORCH_CHECK(change_record_count >= 0, "config_i32[5] change record count must be nonnegative");
  TORCH_CHECK(base_owner_i32.size(0) == base_record_count, "base_owner_i32 length must match base record count");
  TORCH_CHECK(base_start_f32.size(0) == base_record_count, "base_start_f32 length must match base record count");
  TORCH_CHECK(base_end_f32.size(0) == base_record_count, "base_end_f32 length must match base record count");
  TORCH_CHECK(change_frame_i32.size(0) == change_count, "change_frame_i32 length must match change count");
  TORCH_CHECK(change_owner_i32.size(0) == change_record_count, "change_owner_i32 length must match change record count");
  TORCH_CHECK(change_start_f32.size(0) == change_record_count, "change_start_f32 length must match change record count");
  TORCH_CHECK(change_end_f32.size(0) == change_record_count, "change_end_f32 length must match change record count");
  TORCH_CHECK(base_offsets_i32.size(0) == track_count + 1, "base_offsets_i32 length must be track_count + 1");
  TORCH_CHECK(
      track_change_offsets_i32.size(0) == track_count + 1,
      "track_change_offsets_i32 length must be track_count + 1");
  TORCH_CHECK(change_offsets_i32.size(0) == change_count + 1, "change_offsets_i32 length must be change_count + 1");
  check_segment_tape_offsets_cpu(base_offsets_i32, track_count, base_record_count, 2147483647);
  check_segment_tape_offsets_cpu(track_change_offsets_i32, track_count, change_count, 2147483647);
  check_segment_tape_offsets_cpu(change_offsets_i32, change_count, change_record_count, 2147483647);

  auto output_rgb = torch::empty({track_count, frame_count, 3}, site_rgba_f32.options().dtype(torch::kFloat32));
  auto output_alpha = torch::empty({track_count, frame_count}, site_rgba_f32.options().dtype(torch::kFloat32));
  auto output_depth = torch::empty({track_count, frame_count}, site_rgba_f32.options().dtype(torch::kFloat32));
  auto& k = kernels();
  constexpr uint64_t threads = 256ull;
  const int64_t sample_count = track_count * frame_count;
  launch(k.endpoint_delta_replace_rgba_depth_replay, [&](MetalKernelFunction& fn) {
    fn.setArg(0, base_offsets_i32);
    fn.setArg(1, base_owner_i32);
    fn.setArg(2, base_start_f32);
    fn.setArg(3, base_end_f32);
    fn.setArg(4, track_change_offsets_i32);
    fn.setArg(5, change_frame_i32);
    fn.setArg(6, change_offsets_i32);
    fn.setArg(7, change_owner_i32);
    fn.setArg(8, change_start_f32);
    fn.setArg(9, change_end_f32);
    fn.setArg(10, site_rgba_f32);
    fn.setArg(11, config_i32);
    fn.setArg(12, config_f32);
    fn.setArg(13, output_rgb);
    fn.setArg(14, output_alpha);
    fn.setArg(15, output_depth);
    fn.dispatch((uint64_t)sample_count, threads);
  });
  return std::make_tuple(output_rgb, output_alpha, output_depth);
}

torch::Tensor metal_endpoint_delta_replace_vjp_direct_atomic_grad_only(
    const torch::Tensor& base_offsets_i32,
    const torch::Tensor& base_owner_i32,
    const torch::Tensor& base_start_f32,
    const torch::Tensor& base_end_f32,
    const torch::Tensor& track_change_offsets_i32,
    const torch::Tensor& change_frame_i32,
    const torch::Tensor& change_offsets_i32,
    const torch::Tensor& change_owner_i32,
    const torch::Tensor& change_start_f32,
    const torch::Tensor& change_end_f32,
    const torch::Tensor& site_rgba_f32,
    const torch::Tensor& grad_rgb_f32,
    const torch::Tensor& grad_alpha_f32,
    const torch::Tensor& grad_depth_f32,
    const torch::Tensor& config_i32,
    const torch::Tensor& config_f32) {
  check_i32_mps_1d_any(base_offsets_i32, "base_offsets_i32");
  check_i32_mps_1d_any(base_owner_i32, "base_owner_i32");
  check_float_mps_1d_any(base_start_f32, "base_start_f32");
  check_float_mps_1d_any(base_end_f32, "base_end_f32");
  check_i32_mps_1d_any(track_change_offsets_i32, "track_change_offsets_i32");
  check_i32_mps_1d_any(change_frame_i32, "change_frame_i32");
  check_i32_mps_1d_any(change_offsets_i32, "change_offsets_i32");
  check_i32_mps_1d_any(change_owner_i32, "change_owner_i32");
  check_float_mps_1d_any(change_start_f32, "change_start_f32");
  check_float_mps_1d_any(change_end_f32, "change_end_f32");
  check_float_mps_2d(site_rgba_f32, "site_rgba_f32", 4);
  TORCH_CHECK(grad_rgb_f32.device().is_mps(), "grad_rgb_f32 must be on MPS");
  TORCH_CHECK(grad_rgb_f32.scalar_type() == torch::kFloat32, "grad_rgb_f32 must be float32");
  TORCH_CHECK(grad_rgb_f32.dim() == 3 && grad_rgb_f32.size(2) == 3, "grad_rgb_f32 must have shape [K,T,3]");
  TORCH_CHECK(grad_rgb_f32.is_contiguous(), "grad_rgb_f32 must be contiguous");
  TORCH_CHECK(grad_alpha_f32.device().is_mps(), "grad_alpha_f32 must be on MPS");
  TORCH_CHECK(grad_alpha_f32.scalar_type() == torch::kFloat32, "grad_alpha_f32 must be float32");
  TORCH_CHECK(grad_alpha_f32.dim() == 2, "grad_alpha_f32 must have shape [K,T]");
  TORCH_CHECK(grad_alpha_f32.is_contiguous(), "grad_alpha_f32 must be contiguous");
  TORCH_CHECK(grad_depth_f32.device().is_mps(), "grad_depth_f32 must be on MPS");
  TORCH_CHECK(grad_depth_f32.scalar_type() == torch::kFloat32, "grad_depth_f32 must be float32");
  TORCH_CHECK(grad_depth_f32.dim() == 2, "grad_depth_f32 must have shape [K,T]");
  TORCH_CHECK(grad_depth_f32.is_contiguous(), "grad_depth_f32 must be contiguous");
  check_i32_mps_1d(config_i32, "config_i32", 6);
  TORCH_CHECK(config_f32.device().is_mps(), "config_f32 must be on MPS");
  TORCH_CHECK(config_f32.scalar_type() == torch::kFloat32, "config_f32 must be float32");
  TORCH_CHECK(config_f32.dim() == 1 && config_f32.size(0) == 2, "config_f32 must have shape [2]");
  TORCH_CHECK(config_f32.is_contiguous(), "config_f32 must be contiguous");

  auto config_i32_cpu = config_i32.cpu();
  const int32_t* config = config_i32_cpu.data_ptr<int32_t>();
  const int64_t track_count = config[0];
  const int64_t frame_count = config[1];
  const int64_t site_count = config[2];
  const int64_t base_record_count = config[3];
  const int64_t change_count = config[4];
  const int64_t change_record_count = config[5];
  TORCH_CHECK(track_count > 0, "config_i32[0] track count must be positive");
  TORCH_CHECK(frame_count > 0, "config_i32[1] frame count must be positive");
  TORCH_CHECK(site_count == site_rgba_f32.size(0), "config_i32[2] must match site_rgba_f32 rows");
  TORCH_CHECK(site_count > 0 && site_count <= 64, "endpoint delta VJP supports site count in [1, 64]");
  TORCH_CHECK(base_record_count >= 0, "config_i32[3] base record count must be nonnegative");
  TORCH_CHECK(change_count >= 0, "config_i32[4] change count must be nonnegative");
  TORCH_CHECK(change_record_count >= 0, "config_i32[5] change record count must be nonnegative");
  TORCH_CHECK(base_owner_i32.size(0) == base_record_count, "base_owner_i32 length must match base record count");
  TORCH_CHECK(base_start_f32.size(0) == base_record_count, "base_start_f32 length must match base record count");
  TORCH_CHECK(base_end_f32.size(0) == base_record_count, "base_end_f32 length must match base record count");
  TORCH_CHECK(change_frame_i32.size(0) == change_count, "change_frame_i32 length must match change count");
  TORCH_CHECK(change_owner_i32.size(0) == change_record_count, "change_owner_i32 length must match change record count");
  TORCH_CHECK(change_start_f32.size(0) == change_record_count, "change_start_f32 length must match change record count");
  TORCH_CHECK(change_end_f32.size(0) == change_record_count, "change_end_f32 length must match change record count");
  TORCH_CHECK(base_offsets_i32.size(0) == track_count + 1, "base_offsets_i32 length must be track_count + 1");
  TORCH_CHECK(
      track_change_offsets_i32.size(0) == track_count + 1,
      "track_change_offsets_i32 length must be track_count + 1");
  TORCH_CHECK(change_offsets_i32.size(0) == change_count + 1, "change_offsets_i32 length must be change_count + 1");
  TORCH_CHECK(
      grad_rgb_f32.size(0) == track_count && grad_rgb_f32.size(1) == frame_count,
      "grad_rgb_f32 shape mismatch");
  TORCH_CHECK(
      grad_alpha_f32.size(0) == track_count && grad_alpha_f32.size(1) == frame_count,
      "grad_alpha_f32 shape mismatch");
  TORCH_CHECK(
      grad_depth_f32.size(0) == track_count && grad_depth_f32.size(1) == frame_count,
      "grad_depth_f32 shape mismatch");
  check_segment_tape_offsets_cpu(base_offsets_i32, track_count, base_record_count, 129);
  check_segment_tape_offsets_cpu(track_change_offsets_i32, track_count, change_count, 2147483647);
  check_segment_tape_offsets_cpu(change_offsets_i32, change_count, change_record_count, 129);

  auto grad_site_rgba = torch::empty({site_count, 4}, site_rgba_f32.options().dtype(torch::kFloat32));
  auto& k = kernels();
  constexpr uint64_t threads = 256ull;
  const int64_t sample_count = track_count * frame_count;
  launch(k.clear_site_rgba_grad, [&](MetalKernelFunction& fn) {
    fn.setArg(0, grad_site_rgba);
    fn.setArg(1, config_i32);
    fn.dispatch((uint64_t)site_count, threads);
  });
  launch(k.endpoint_delta_replace_vjp_direct_atomic_grad_only, [&](MetalKernelFunction& fn) {
    fn.setArg(0, base_offsets_i32);
    fn.setArg(1, base_owner_i32);
    fn.setArg(2, base_start_f32);
    fn.setArg(3, base_end_f32);
    fn.setArg(4, track_change_offsets_i32);
    fn.setArg(5, change_frame_i32);
    fn.setArg(6, change_offsets_i32);
    fn.setArg(7, change_owner_i32);
    fn.setArg(8, change_start_f32);
    fn.setArg(9, change_end_f32);
    fn.setArg(10, site_rgba_f32);
    fn.setArg(11, grad_rgb_f32);
    fn.setArg(12, grad_alpha_f32);
    fn.setArg(13, grad_depth_f32);
    fn.setArg(14, config_i32);
    fn.setArg(15, config_f32);
    fn.setArg(16, grad_site_rgba);
    fn.dispatch((uint64_t)sample_count, threads);
  });
  return grad_site_rgba;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> metal_endpoint_record_delta_replace_rgba_depth_replay(
    const torch::Tensor& boundary_f32,
    const torch::Tensor& rays_f32,
    const torch::Tensor& frame_t_f32,
    const torch::Tensor& base_offsets_i32,
    const torch::Tensor& base_owner_i32,
    const torch::Tensor& base_left_i32,
    const torch::Tensor& base_right_i32,
    const torch::Tensor& track_change_offsets_i32,
    const torch::Tensor& change_frame_i32,
    const torch::Tensor& change_offsets_i32,
    const torch::Tensor& change_owner_i32,
    const torch::Tensor& change_left_i32,
    const torch::Tensor& change_right_i32,
    const torch::Tensor& site_rgba_f32,
    const torch::Tensor& config_i32,
    const torch::Tensor& config_f32) {
  check_float_mps_2d(boundary_f32, "boundary_f32", 5);
  TORCH_CHECK(rays_f32.device().is_mps(), "rays_f32 must be on MPS");
  TORCH_CHECK(rays_f32.scalar_type() == torch::kFloat32, "rays_f32 must be float32");
  TORCH_CHECK(rays_f32.dim() == 3 && rays_f32.size(2) == 6, "rays_f32 must have shape [K,T,6]");
  TORCH_CHECK(rays_f32.is_contiguous(), "rays_f32 must be contiguous");
  check_float_mps_1d_any(frame_t_f32, "frame_t_f32");
  check_i32_mps_1d_any(base_offsets_i32, "base_offsets_i32");
  check_i32_mps_1d_any(base_owner_i32, "base_owner_i32");
  check_i32_mps_1d_any(base_left_i32, "base_left_i32");
  check_i32_mps_1d_any(base_right_i32, "base_right_i32");
  check_i32_mps_1d_any(track_change_offsets_i32, "track_change_offsets_i32");
  check_i32_mps_1d_any(change_frame_i32, "change_frame_i32");
  check_i32_mps_1d_any(change_offsets_i32, "change_offsets_i32");
  check_i32_mps_1d_any(change_owner_i32, "change_owner_i32");
  check_i32_mps_1d_any(change_left_i32, "change_left_i32");
  check_i32_mps_1d_any(change_right_i32, "change_right_i32");
  check_float_mps_2d(site_rgba_f32, "site_rgba_f32", 4);
  check_i32_mps_1d(config_i32, "config_i32", 7);
  TORCH_CHECK(config_f32.device().is_mps(), "config_f32 must be on MPS");
  TORCH_CHECK(config_f32.scalar_type() == torch::kFloat32, "config_f32 must be float32");
  TORCH_CHECK(config_f32.dim() == 1 && config_f32.size(0) == 4, "config_f32 must have shape [4]");
  TORCH_CHECK(config_f32.is_contiguous(), "config_f32 must be contiguous");

  auto config_i32_cpu = config_i32.cpu();
  const int32_t* config = config_i32_cpu.data_ptr<int32_t>();
  const int64_t boundary_count = config[0];
  const int64_t track_count = config[1];
  const int64_t frame_count = config[2];
  const int64_t site_count = config[3];
  const int64_t base_record_count = config[4];
  const int64_t change_count = config[5];
  const int64_t change_record_count = config[6];
  TORCH_CHECK(boundary_count == boundary_f32.size(0), "config_i32[0] must match boundary_f32 rows");
  TORCH_CHECK(boundary_count > 0, "config_i32[0] boundary count must be positive");
  TORCH_CHECK(track_count > 0, "config_i32[1] track count must be positive");
  TORCH_CHECK(frame_count > 0, "config_i32[2] frame count must be positive");
  TORCH_CHECK(site_count == site_rgba_f32.size(0), "config_i32[3] must match site_rgba_f32 rows");
  TORCH_CHECK(site_count > 0 && site_count <= 64, "endpoint record delta replay supports site count in [1, 64]");
  TORCH_CHECK(base_record_count >= 0, "config_i32[4] base record count must be nonnegative");
  TORCH_CHECK(change_count >= 0, "config_i32[5] change count must be nonnegative");
  TORCH_CHECK(change_record_count >= 0, "config_i32[6] change record count must be nonnegative");
  TORCH_CHECK(rays_f32.size(0) == track_count && rays_f32.size(1) == frame_count, "rays_f32 shape mismatch");
  TORCH_CHECK(frame_t_f32.size(0) == frame_count, "frame_t_f32 length must match frame count");
  TORCH_CHECK(base_owner_i32.size(0) == base_record_count, "base_owner_i32 length must match base record count");
  TORCH_CHECK(base_left_i32.size(0) == base_record_count, "base_left_i32 length must match base record count");
  TORCH_CHECK(base_right_i32.size(0) == base_record_count, "base_right_i32 length must match base record count");
  TORCH_CHECK(change_frame_i32.size(0) == change_count, "change_frame_i32 length must match change count");
  TORCH_CHECK(change_owner_i32.size(0) == change_record_count, "change_owner_i32 length must match change record count");
  TORCH_CHECK(change_left_i32.size(0) == change_record_count, "change_left_i32 length must match change record count");
  TORCH_CHECK(change_right_i32.size(0) == change_record_count, "change_right_i32 length must match change record count");
  TORCH_CHECK(base_offsets_i32.size(0) == track_count + 1, "base_offsets_i32 length must be track_count + 1");
  TORCH_CHECK(
      track_change_offsets_i32.size(0) == track_count + 1,
      "track_change_offsets_i32 length must be track_count + 1");
  TORCH_CHECK(change_offsets_i32.size(0) == change_count + 1, "change_offsets_i32 length must be change_count + 1");
  check_segment_tape_offsets_cpu(base_offsets_i32, track_count, base_record_count, 2147483647);
  check_segment_tape_offsets_cpu(track_change_offsets_i32, track_count, change_count, 2147483647);
  check_segment_tape_offsets_cpu(change_offsets_i32, change_count, change_record_count, 2147483647);

  auto output_rgb = torch::empty({track_count, frame_count, 3}, site_rgba_f32.options().dtype(torch::kFloat32));
  auto output_alpha = torch::empty({track_count, frame_count}, site_rgba_f32.options().dtype(torch::kFloat32));
  auto output_depth = torch::empty({track_count, frame_count}, site_rgba_f32.options().dtype(torch::kFloat32));
  auto& k = kernels();
  constexpr uint64_t threads = 256ull;
  const int64_t sample_count = track_count * frame_count;
  launch(k.endpoint_record_delta_replace_rgba_depth_replay, [&](MetalKernelFunction& fn) {
    fn.setArg(0, boundary_f32);
    fn.setArg(1, rays_f32);
    fn.setArg(2, frame_t_f32);
    fn.setArg(3, base_offsets_i32);
    fn.setArg(4, base_owner_i32);
    fn.setArg(5, base_left_i32);
    fn.setArg(6, base_right_i32);
    fn.setArg(7, track_change_offsets_i32);
    fn.setArg(8, change_frame_i32);
    fn.setArg(9, change_offsets_i32);
    fn.setArg(10, change_owner_i32);
    fn.setArg(11, change_left_i32);
    fn.setArg(12, change_right_i32);
    fn.setArg(13, site_rgba_f32);
    fn.setArg(14, config_i32);
    fn.setArg(15, config_f32);
    fn.setArg(16, output_rgb);
    fn.setArg(17, output_alpha);
    fn.setArg(18, output_depth);
    fn.dispatch((uint64_t)sample_count, threads);
  });
  return std::make_tuple(output_rgb, output_alpha, output_depth);
}

torch::Tensor metal_endpoint_record_delta_replace_vjp_direct_atomic_grad_only(
    const torch::Tensor& boundary_f32,
    const torch::Tensor& rays_f32,
    const torch::Tensor& frame_t_f32,
    const torch::Tensor& base_offsets_i32,
    const torch::Tensor& base_owner_i32,
    const torch::Tensor& base_left_i32,
    const torch::Tensor& base_right_i32,
    const torch::Tensor& track_change_offsets_i32,
    const torch::Tensor& change_frame_i32,
    const torch::Tensor& change_offsets_i32,
    const torch::Tensor& change_owner_i32,
    const torch::Tensor& change_left_i32,
    const torch::Tensor& change_right_i32,
    const torch::Tensor& site_rgba_f32,
    const torch::Tensor& grad_rgb_f32,
    const torch::Tensor& grad_alpha_f32,
    const torch::Tensor& grad_depth_f32,
    const torch::Tensor& config_i32,
    const torch::Tensor& config_f32) {
  check_float_mps_2d(boundary_f32, "boundary_f32", 5);
  TORCH_CHECK(rays_f32.device().is_mps(), "rays_f32 must be on MPS");
  TORCH_CHECK(rays_f32.scalar_type() == torch::kFloat32, "rays_f32 must be float32");
  TORCH_CHECK(rays_f32.dim() == 3 && rays_f32.size(2) == 6, "rays_f32 must have shape [K,T,6]");
  TORCH_CHECK(rays_f32.is_contiguous(), "rays_f32 must be contiguous");
  check_float_mps_1d_any(frame_t_f32, "frame_t_f32");
  check_i32_mps_1d_any(base_offsets_i32, "base_offsets_i32");
  check_i32_mps_1d_any(base_owner_i32, "base_owner_i32");
  check_i32_mps_1d_any(base_left_i32, "base_left_i32");
  check_i32_mps_1d_any(base_right_i32, "base_right_i32");
  check_i32_mps_1d_any(track_change_offsets_i32, "track_change_offsets_i32");
  check_i32_mps_1d_any(change_frame_i32, "change_frame_i32");
  check_i32_mps_1d_any(change_offsets_i32, "change_offsets_i32");
  check_i32_mps_1d_any(change_owner_i32, "change_owner_i32");
  check_i32_mps_1d_any(change_left_i32, "change_left_i32");
  check_i32_mps_1d_any(change_right_i32, "change_right_i32");
  check_float_mps_2d(site_rgba_f32, "site_rgba_f32", 4);
  TORCH_CHECK(grad_rgb_f32.device().is_mps(), "grad_rgb_f32 must be on MPS");
  TORCH_CHECK(grad_rgb_f32.scalar_type() == torch::kFloat32, "grad_rgb_f32 must be float32");
  TORCH_CHECK(grad_rgb_f32.dim() == 3 && grad_rgb_f32.size(2) == 3, "grad_rgb_f32 must have shape [K,T,3]");
  TORCH_CHECK(grad_rgb_f32.is_contiguous(), "grad_rgb_f32 must be contiguous");
  TORCH_CHECK(grad_alpha_f32.device().is_mps(), "grad_alpha_f32 must be on MPS");
  TORCH_CHECK(grad_alpha_f32.scalar_type() == torch::kFloat32, "grad_alpha_f32 must be float32");
  TORCH_CHECK(grad_alpha_f32.dim() == 2, "grad_alpha_f32 must have shape [K,T]");
  TORCH_CHECK(grad_alpha_f32.is_contiguous(), "grad_alpha_f32 must be contiguous");
  TORCH_CHECK(grad_depth_f32.device().is_mps(), "grad_depth_f32 must be on MPS");
  TORCH_CHECK(grad_depth_f32.scalar_type() == torch::kFloat32, "grad_depth_f32 must be float32");
  TORCH_CHECK(grad_depth_f32.dim() == 2, "grad_depth_f32 must have shape [K,T]");
  TORCH_CHECK(grad_depth_f32.is_contiguous(), "grad_depth_f32 must be contiguous");
  check_i32_mps_1d(config_i32, "config_i32", 7);
  TORCH_CHECK(config_f32.device().is_mps(), "config_f32 must be on MPS");
  TORCH_CHECK(config_f32.scalar_type() == torch::kFloat32, "config_f32 must be float32");
  TORCH_CHECK(config_f32.dim() == 1 && config_f32.size(0) == 4, "config_f32 must have shape [4]");
  TORCH_CHECK(config_f32.is_contiguous(), "config_f32 must be contiguous");

  auto config_i32_cpu = config_i32.cpu();
  const int32_t* config = config_i32_cpu.data_ptr<int32_t>();
  const int64_t boundary_count = config[0];
  const int64_t track_count = config[1];
  const int64_t frame_count = config[2];
  const int64_t site_count = config[3];
  const int64_t base_record_count = config[4];
  const int64_t change_count = config[5];
  const int64_t change_record_count = config[6];
  TORCH_CHECK(boundary_count == boundary_f32.size(0), "config_i32[0] must match boundary_f32 rows");
  TORCH_CHECK(boundary_count > 0, "config_i32[0] boundary count must be positive");
  TORCH_CHECK(track_count > 0, "config_i32[1] track count must be positive");
  TORCH_CHECK(frame_count > 0, "config_i32[2] frame count must be positive");
  TORCH_CHECK(site_count == site_rgba_f32.size(0), "config_i32[3] must match site_rgba_f32 rows");
  TORCH_CHECK(site_count > 0 && site_count <= 64, "endpoint record delta VJP supports site count in [1, 64]");
  TORCH_CHECK(base_record_count >= 0, "config_i32[4] base record count must be nonnegative");
  TORCH_CHECK(change_count >= 0, "config_i32[5] change count must be nonnegative");
  TORCH_CHECK(change_record_count >= 0, "config_i32[6] change record count must be nonnegative");
  TORCH_CHECK(rays_f32.size(0) == track_count && rays_f32.size(1) == frame_count, "rays_f32 shape mismatch");
  TORCH_CHECK(frame_t_f32.size(0) == frame_count, "frame_t_f32 length must match frame count");
  TORCH_CHECK(base_owner_i32.size(0) == base_record_count, "base_owner_i32 length must match base record count");
  TORCH_CHECK(base_left_i32.size(0) == base_record_count, "base_left_i32 length must match base record count");
  TORCH_CHECK(base_right_i32.size(0) == base_record_count, "base_right_i32 length must match base record count");
  TORCH_CHECK(change_frame_i32.size(0) == change_count, "change_frame_i32 length must match change count");
  TORCH_CHECK(change_owner_i32.size(0) == change_record_count, "change_owner_i32 length must match change record count");
  TORCH_CHECK(change_left_i32.size(0) == change_record_count, "change_left_i32 length must match change record count");
  TORCH_CHECK(change_right_i32.size(0) == change_record_count, "change_right_i32 length must match change record count");
  TORCH_CHECK(base_offsets_i32.size(0) == track_count + 1, "base_offsets_i32 length must be track_count + 1");
  TORCH_CHECK(
      track_change_offsets_i32.size(0) == track_count + 1,
      "track_change_offsets_i32 length must be track_count + 1");
  TORCH_CHECK(change_offsets_i32.size(0) == change_count + 1, "change_offsets_i32 length must be change_count + 1");
  TORCH_CHECK(
      grad_rgb_f32.size(0) == track_count && grad_rgb_f32.size(1) == frame_count,
      "grad_rgb_f32 shape mismatch");
  TORCH_CHECK(
      grad_alpha_f32.size(0) == track_count && grad_alpha_f32.size(1) == frame_count,
      "grad_alpha_f32 shape mismatch");
  TORCH_CHECK(
      grad_depth_f32.size(0) == track_count && grad_depth_f32.size(1) == frame_count,
      "grad_depth_f32 shape mismatch");
  check_segment_tape_offsets_cpu(base_offsets_i32, track_count, base_record_count, 129);
  check_segment_tape_offsets_cpu(track_change_offsets_i32, track_count, change_count, 2147483647);
  check_segment_tape_offsets_cpu(change_offsets_i32, change_count, change_record_count, 129);

  auto grad_site_rgba = torch::zeros({site_count, 4}, site_rgba_f32.options().dtype(torch::kFloat32));
  auto& k = kernels();
  constexpr uint64_t threads = 256ull;
  const int64_t sample_count = track_count * frame_count;
  launch(k.endpoint_record_delta_replace_vjp_direct_atomic_grad_only, [&](MetalKernelFunction& fn) {
    fn.setArg(0, boundary_f32);
    fn.setArg(1, rays_f32);
    fn.setArg(2, frame_t_f32);
    fn.setArg(3, base_offsets_i32);
    fn.setArg(4, base_owner_i32);
    fn.setArg(5, base_left_i32);
    fn.setArg(6, base_right_i32);
    fn.setArg(7, track_change_offsets_i32);
    fn.setArg(8, change_frame_i32);
    fn.setArg(9, change_offsets_i32);
    fn.setArg(10, change_owner_i32);
    fn.setArg(11, change_left_i32);
    fn.setArg(12, change_right_i32);
    fn.setArg(13, site_rgba_f32);
    fn.setArg(14, grad_rgb_f32);
    fn.setArg(15, grad_alpha_f32);
    fn.setArg(16, grad_depth_f32);
    fn.setArg(17, config_i32);
    fn.setArg(18, config_f32);
    fn.setArg(19, grad_site_rgba);
    fn.dispatch((uint64_t)sample_count, threads);
  });
  return grad_site_rgba;
}

std::tuple<torch::Tensor, torch::Tensor> metal_endpoint_record_delta_replace_coeff16_mse_vjp_direct_atomic_rgb_only(
    const torch::Tensor& coeff_f16,
    const torch::Tensor& frame_t_f32,
    const torch::Tensor& base_offsets_i32,
    const torch::Tensor& base_owner_i32,
    const torch::Tensor& base_left_i32,
    const torch::Tensor& base_right_i32,
    const torch::Tensor& track_change_offsets_i32,
    const torch::Tensor& change_frame_i32,
    const torch::Tensor& change_offsets_i32,
    const torch::Tensor& change_owner_i32,
    const torch::Tensor& change_left_i32,
    const torch::Tensor& change_right_i32,
    const torch::Tensor& site_rgba_f32,
    const torch::Tensor& target_rgb_f32,
    const torch::Tensor& config_i32,
    const torch::Tensor& config_f32) {
  check_half_mps_2d(coeff_f16, "coeff_f16", 4);
  check_float_mps_1d_any(frame_t_f32, "frame_t_f32");
  check_i32_mps_1d_any(base_offsets_i32, "base_offsets_i32");
  check_i32_mps_1d_any(base_owner_i32, "base_owner_i32");
  check_i32_mps_1d_any(base_left_i32, "base_left_i32");
  check_i32_mps_1d_any(base_right_i32, "base_right_i32");
  check_i32_mps_1d_any(track_change_offsets_i32, "track_change_offsets_i32");
  check_i32_mps_1d_any(change_frame_i32, "change_frame_i32");
  check_i32_mps_1d_any(change_offsets_i32, "change_offsets_i32");
  check_i32_mps_1d_any(change_owner_i32, "change_owner_i32");
  check_i32_mps_1d_any(change_left_i32, "change_left_i32");
  check_i32_mps_1d_any(change_right_i32, "change_right_i32");
  check_float_mps_2d(site_rgba_f32, "site_rgba_f32", 4);
  TORCH_CHECK(target_rgb_f32.device().is_mps(), "target_rgb_f32 must be on MPS");
  TORCH_CHECK(target_rgb_f32.scalar_type() == torch::kFloat32, "target_rgb_f32 must be float32");
  TORCH_CHECK(target_rgb_f32.dim() == 3 && target_rgb_f32.size(2) == 3, "target_rgb_f32 must have shape [K,T,3]");
  TORCH_CHECK(target_rgb_f32.is_contiguous(), "target_rgb_f32 must be contiguous");
  check_i32_mps_1d(config_i32, "config_i32", 7);
  TORCH_CHECK(config_f32.device().is_mps(), "config_f32 must be on MPS");
  TORCH_CHECK(config_f32.scalar_type() == torch::kFloat32, "config_f32 must be float32");
  TORCH_CHECK(config_f32.dim() == 1 && config_f32.size(0) == 4, "config_f32 must have shape [4]");
  TORCH_CHECK(config_f32.is_contiguous(), "config_f32 must be contiguous");

  auto config_i32_cpu = config_i32.cpu();
  const int32_t* config = config_i32_cpu.data_ptr<int32_t>();
  const int64_t boundary_count = config[0];
  const int64_t track_count = config[1];
  const int64_t frame_count = config[2];
  const int64_t site_count = config[3];
  const int64_t base_record_count = config[4];
  const int64_t change_count = config[5];
  const int64_t change_record_count = config[6];
  TORCH_CHECK(boundary_count > 0, "config_i32[0] boundary count must be positive");
  TORCH_CHECK(track_count > 0, "config_i32[1] track count must be positive");
  TORCH_CHECK(frame_count > 0, "config_i32[2] frame count must be positive");
  TORCH_CHECK(site_count == site_rgba_f32.size(0), "config_i32[3] must match site_rgba_f32 rows");
  TORCH_CHECK(
      site_count > 0 && site_count <= 64,
      "endpoint record delta replace coeff16 fused MSE VJP supports site count in [1, 64]");
  TORCH_CHECK(base_record_count >= 0, "config_i32[4] base record count must be nonnegative");
  TORCH_CHECK(change_count >= 0, "config_i32[5] change count must be nonnegative");
  TORCH_CHECK(change_record_count >= 0, "config_i32[6] change record count must be nonnegative");
  TORCH_CHECK(coeff_f16.size(0) == track_count * boundary_count, "coeff_f16 row count mismatch");
  TORCH_CHECK(frame_t_f32.size(0) == frame_count, "frame_t_f32 length must match frame count");
  TORCH_CHECK(base_owner_i32.size(0) == base_record_count, "base_owner_i32 length must match base record count");
  TORCH_CHECK(base_left_i32.size(0) == base_record_count, "base_left_i32 length must match base record count");
  TORCH_CHECK(base_right_i32.size(0) == base_record_count, "base_right_i32 length must match base record count");
  TORCH_CHECK(change_frame_i32.size(0) == change_count, "change_frame_i32 length must match change count");
  TORCH_CHECK(change_owner_i32.size(0) == change_record_count, "change_owner_i32 length must match change record count");
  TORCH_CHECK(change_left_i32.size(0) == change_record_count, "change_left_i32 length must match change record count");
  TORCH_CHECK(change_right_i32.size(0) == change_record_count, "change_right_i32 length must match change record count");
  TORCH_CHECK(base_offsets_i32.size(0) == track_count + 1, "base_offsets_i32 length must be track_count + 1");
  TORCH_CHECK(
      track_change_offsets_i32.size(0) == track_count + 1,
      "track_change_offsets_i32 length must be track_count + 1");
  TORCH_CHECK(change_offsets_i32.size(0) == change_count + 1, "change_offsets_i32 length must be change_count + 1");
  TORCH_CHECK(
      target_rgb_f32.size(0) == track_count && target_rgb_f32.size(1) == frame_count,
      "target_rgb_f32 shape mismatch");
  check_segment_tape_offsets_cpu(base_offsets_i32, track_count, base_record_count, 129);
  check_segment_tape_offsets_cpu(track_change_offsets_i32, track_count, change_count, 2147483647);
  check_segment_tape_offsets_cpu(change_offsets_i32, change_count, change_record_count, 129);

  auto loss = torch::empty({1}, site_rgba_f32.options().dtype(torch::kFloat32));
  auto grad_site_rgba = torch::empty({site_count, 4}, site_rgba_f32.options().dtype(torch::kFloat32));
  auto& k = kernels();
  constexpr uint64_t threads = 256ull;
  const int64_t sample_count = track_count * frame_count;
  launch(k.clear_endpoint_loss_site_rgba_grad, [&](MetalKernelFunction& fn) {
    fn.setArg(0, loss);
    fn.setArg(1, grad_site_rgba);
    fn.setArg(2, config_i32);
    fn.dispatch((uint64_t)std::max<int64_t>(site_count, 1), threads);
  });
  launch(k.endpoint_record_delta_replace_coeff16_mse_vjp_direct_atomic_rgb_only, [&](MetalKernelFunction& fn) {
    fn.setArg(0, coeff_f16);
    fn.setArg(1, frame_t_f32);
    fn.setArg(2, base_offsets_i32);
    fn.setArg(3, base_owner_i32);
    fn.setArg(4, base_left_i32);
    fn.setArg(5, base_right_i32);
    fn.setArg(6, track_change_offsets_i32);
    fn.setArg(7, change_frame_i32);
    fn.setArg(8, change_offsets_i32);
    fn.setArg(9, change_owner_i32);
    fn.setArg(10, change_left_i32);
    fn.setArg(11, change_right_i32);
    fn.setArg(12, site_rgba_f32);
    fn.setArg(13, target_rgb_f32);
    fn.setArg(14, config_i32);
    fn.setArg(15, config_f32);
    fn.setArg(16, loss);
    fn.setArg(17, grad_site_rgba);
    fn.dispatch((uint64_t)sample_count, threads);
  });
  return std::make_tuple(loss, grad_site_rgba);
}

std::tuple<torch::Tensor, torch::Tensor> metal_endpoint_record_delta_replace_coeff16_i16x3_mse_vjp_direct_atomic_rgb_only(
    const torch::Tensor& coeff_f16,
    const torch::Tensor& frame_t_f32,
    const torch::Tensor& base_offsets_i32,
    const torch::Tensor& base_record_i16,
    const torch::Tensor& track_change_offsets_i32,
    const torch::Tensor& change_frame_i32,
    const torch::Tensor& change_offsets_i32,
    const torch::Tensor& change_record_i16,
    const torch::Tensor& site_rgba_f32,
    const torch::Tensor& target_rgb_f32,
    const torch::Tensor& config_i32,
    const torch::Tensor& config_f32) {
  check_half_mps_2d(coeff_f16, "coeff_f16", 4);
  check_float_mps_1d_any(frame_t_f32, "frame_t_f32");
  check_i32_mps_1d_any(base_offsets_i32, "base_offsets_i32");
  check_i16_mps_1d_any(base_record_i16, "base_record_i16");
  check_i32_mps_1d_any(track_change_offsets_i32, "track_change_offsets_i32");
  check_i32_mps_1d_any(change_frame_i32, "change_frame_i32");
  check_i32_mps_1d_any(change_offsets_i32, "change_offsets_i32");
  check_i16_mps_1d_any(change_record_i16, "change_record_i16");
  check_float_mps_2d(site_rgba_f32, "site_rgba_f32", 4);
  TORCH_CHECK(target_rgb_f32.device().is_mps(), "target_rgb_f32 must be on MPS");
  TORCH_CHECK(target_rgb_f32.scalar_type() == torch::kFloat32, "target_rgb_f32 must be float32");
  TORCH_CHECK(target_rgb_f32.dim() == 3 && target_rgb_f32.size(2) == 3, "target_rgb_f32 must have shape [K,T,3]");
  TORCH_CHECK(target_rgb_f32.is_contiguous(), "target_rgb_f32 must be contiguous");
  check_i32_mps_1d(config_i32, "config_i32", 7);
  TORCH_CHECK(config_f32.device().is_mps(), "config_f32 must be on MPS");
  TORCH_CHECK(config_f32.scalar_type() == torch::kFloat32, "config_f32 must be float32");
  TORCH_CHECK(config_f32.dim() == 1 && config_f32.size(0) == 4, "config_f32 must have shape [4]");
  TORCH_CHECK(config_f32.is_contiguous(), "config_f32 must be contiguous");

  auto config_i32_cpu = config_i32.cpu();
  const int32_t* config = config_i32_cpu.data_ptr<int32_t>();
  const int64_t boundary_count = config[0];
  const int64_t track_count = config[1];
  const int64_t frame_count = config[2];
  const int64_t site_count = config[3];
  const int64_t base_record_count = config[4];
  const int64_t change_count = config[5];
  const int64_t change_record_count = config[6];
  TORCH_CHECK(boundary_count > 0, "config_i32[0] boundary count must be positive");
  TORCH_CHECK(boundary_count <= 32765, "i16x3 delta replace coeff16 fused MSE supports boundary count <= 32765");
  TORCH_CHECK(track_count > 0, "config_i32[1] track count must be positive");
  TORCH_CHECK(frame_count > 0, "config_i32[2] frame count must be positive");
  TORCH_CHECK(site_count == site_rgba_f32.size(0), "config_i32[3] must match site_rgba_f32 rows");
  TORCH_CHECK(
      site_count > 0 && site_count <= 32767,
      "i16x3 endpoint record delta replace coeff16 fused MSE VJP supports site count in [1, 32767]");
  TORCH_CHECK(base_record_count >= 0, "config_i32[4] base record count must be nonnegative");
  TORCH_CHECK(change_count >= 0, "config_i32[5] change count must be nonnegative");
  TORCH_CHECK(change_record_count >= 0, "config_i32[6] change record count must be nonnegative");
  TORCH_CHECK(coeff_f16.size(0) == track_count * boundary_count, "coeff_f16 row count mismatch");
  TORCH_CHECK(frame_t_f32.size(0) == frame_count, "frame_t_f32 length must match frame count");
  TORCH_CHECK(base_record_i16.size(0) == base_record_count * 3, "base_record_i16 length mismatch");
  TORCH_CHECK(change_frame_i32.size(0) == change_count, "change_frame_i32 length must match change count");
  TORCH_CHECK(change_record_i16.size(0) == change_record_count * 3, "change_record_i16 length mismatch");
  TORCH_CHECK(base_offsets_i32.size(0) == track_count + 1, "base_offsets_i32 length must be track_count + 1");
  TORCH_CHECK(
      track_change_offsets_i32.size(0) == track_count + 1,
      "track_change_offsets_i32 length must be track_count + 1");
  TORCH_CHECK(change_offsets_i32.size(0) == change_count + 1, "change_offsets_i32 length must be change_count + 1");
  TORCH_CHECK(
      target_rgb_f32.size(0) == track_count && target_rgb_f32.size(1) == frame_count,
      "target_rgb_f32 shape mismatch");
  check_segment_tape_offsets_cpu(base_offsets_i32, track_count, base_record_count, 129);
  check_segment_tape_offsets_cpu(track_change_offsets_i32, track_count, change_count, 2147483647);
  check_segment_tape_offsets_cpu(change_offsets_i32, change_count, change_record_count, 129);

  auto loss = torch::empty({1}, site_rgba_f32.options().dtype(torch::kFloat32));
  auto grad_site_rgba = torch::empty({site_count, 4}, site_rgba_f32.options().dtype(torch::kFloat32));
  auto& k = kernels();
  constexpr uint64_t threads = 256ull;
  const int64_t sample_count = track_count * frame_count;
  launch(k.clear_endpoint_loss_site_rgba_grad, [&](MetalKernelFunction& fn) {
    fn.setArg(0, loss);
    fn.setArg(1, grad_site_rgba);
    fn.setArg(2, config_i32);
    fn.dispatch((uint64_t)std::max<int64_t>(site_count, 1), threads);
  });
  launch(k.endpoint_record_delta_replace_coeff16_i16x3_mse_vjp_direct_atomic_rgb_only, [&](MetalKernelFunction& fn) {
    fn.setArg(0, coeff_f16);
    fn.setArg(1, frame_t_f32);
    fn.setArg(2, base_offsets_i32);
    fn.setArg(3, base_record_i16);
    fn.setArg(4, track_change_offsets_i32);
    fn.setArg(5, change_frame_i32);
    fn.setArg(6, change_offsets_i32);
    fn.setArg(7, change_record_i16);
    fn.setArg(8, site_rgba_f32);
    fn.setArg(9, target_rgb_f32);
    fn.setArg(10, config_i32);
    fn.setArg(11, config_f32);
    fn.setArg(12, loss);
    fn.setArg(13, grad_site_rgba);
    fn.dispatch((uint64_t)sample_count, threads);
  });
  return std::make_tuple(loss, grad_site_rgba);
}

std::tuple<torch::Tensor, torch::Tensor>
metal_endpoint_record_delta_replace_coeff16_i16x3_framegroup16_mse_vjp_direct_atomic_rgb_only(
    const torch::Tensor& coeff_f16,
    const torch::Tensor& frame_t_f32,
    const torch::Tensor& base_offsets_i32,
    const torch::Tensor& base_record_i16,
    const torch::Tensor& track_change_offsets_i32,
    const torch::Tensor& track_chunk_change_offsets_i16,
    const torch::Tensor& change_frame_i32,
    const torch::Tensor& change_offsets_i32,
    const torch::Tensor& change_record_i16,
    const torch::Tensor& site_rgba_f32,
    const torch::Tensor& target_rgb_f32,
    const torch::Tensor& config_i32,
    const torch::Tensor& config_f32) {
  check_half_mps_2d(coeff_f16, "coeff_f16", 4);
  check_float_mps_1d_any(frame_t_f32, "frame_t_f32");
  check_i32_mps_1d_any(base_offsets_i32, "base_offsets_i32");
  check_i16_mps_1d_any(base_record_i16, "base_record_i16");
  check_i32_mps_1d_any(track_change_offsets_i32, "track_change_offsets_i32");
  check_i16_mps_1d_any(track_chunk_change_offsets_i16, "track_chunk_change_offsets_i16");
  check_i32_mps_1d_any(change_frame_i32, "change_frame_i32");
  check_i32_mps_1d_any(change_offsets_i32, "change_offsets_i32");
  check_i16_mps_1d_any(change_record_i16, "change_record_i16");
  check_float_mps_2d(site_rgba_f32, "site_rgba_f32", 4);
  TORCH_CHECK(target_rgb_f32.device().is_mps(), "target_rgb_f32 must be on MPS");
  TORCH_CHECK(target_rgb_f32.scalar_type() == torch::kFloat32, "target_rgb_f32 must be float32");
  TORCH_CHECK(target_rgb_f32.dim() == 3 && target_rgb_f32.size(2) == 3, "target_rgb_f32 must have shape [K,T,3]");
  TORCH_CHECK(target_rgb_f32.is_contiguous(), "target_rgb_f32 must be contiguous");
  check_i32_mps_1d(config_i32, "config_i32", 7);
  TORCH_CHECK(config_f32.device().is_mps(), "config_f32 must be on MPS");
  TORCH_CHECK(config_f32.scalar_type() == torch::kFloat32, "config_f32 must be float32");
  TORCH_CHECK(config_f32.dim() == 1 && config_f32.size(0) == 4, "config_f32 must have shape [4]");
  TORCH_CHECK(config_f32.is_contiguous(), "config_f32 must be contiguous");

  auto config_i32_cpu = config_i32.cpu();
  const int32_t* config = config_i32_cpu.data_ptr<int32_t>();
  const int64_t boundary_count = config[0];
  const int64_t track_count = config[1];
  const int64_t frame_count = config[2];
  const int64_t site_count = config[3];
  const int64_t base_record_count = config[4];
  const int64_t change_count = config[5];
  const int64_t change_record_count = config[6];
  TORCH_CHECK(boundary_count > 0, "config_i32[0] boundary count must be positive");
  TORCH_CHECK(boundary_count <= 32765, "i16x3 framegroup16 delta replace coeff16 fused MSE supports boundary count <= 32765");
  TORCH_CHECK(track_count > 0, "config_i32[1] track count must be positive");
  TORCH_CHECK(frame_count > 0, "config_i32[2] frame count must be positive");
  TORCH_CHECK(site_count == site_rgba_f32.size(0), "config_i32[3] must match site_rgba_f32 rows");
  TORCH_CHECK(
      site_count > 0 && site_count <= 32767,
      "i16x3 framegroup16 endpoint record delta replace coeff16 fused MSE VJP supports site count in [1, 32767]");
  TORCH_CHECK(base_record_count >= 0, "config_i32[4] base record count must be nonnegative");
  TORCH_CHECK(change_count >= 0, "config_i32[5] change count must be nonnegative");
  TORCH_CHECK(change_count <= 32767, "int16 chunk-start offsets require change count <= 32767");
  TORCH_CHECK(change_record_count >= 0, "config_i32[6] change record count must be nonnegative");
  TORCH_CHECK(coeff_f16.size(0) == track_count * boundary_count, "coeff_f16 row count mismatch");
  TORCH_CHECK(frame_t_f32.size(0) == frame_count, "frame_t_f32 length must match frame count");
  TORCH_CHECK(base_record_i16.size(0) == base_record_count * 3, "base_record_i16 length mismatch");
  TORCH_CHECK(change_frame_i32.size(0) == change_count, "change_frame_i32 length must match change count");
  TORCH_CHECK(change_record_i16.size(0) == change_record_count * 3, "change_record_i16 length mismatch");
  TORCH_CHECK(base_offsets_i32.size(0) == track_count + 1, "base_offsets_i32 length must be track_count + 1");
  TORCH_CHECK(
      track_change_offsets_i32.size(0) == track_count + 1,
      "track_change_offsets_i32 length must be track_count + 1");
  TORCH_CHECK(change_offsets_i32.size(0) == change_count + 1, "change_offsets_i32 length must be change_count + 1");
  TORCH_CHECK(
      target_rgb_f32.size(0) == track_count && target_rgb_f32.size(1) == frame_count,
      "target_rgb_f32 shape mismatch");
  check_segment_tape_offsets_cpu(base_offsets_i32, track_count, base_record_count, 129);
  check_segment_tape_offsets_cpu(track_change_offsets_i32, track_count, change_count, 2147483647);
  check_segment_tape_offsets_cpu(change_offsets_i32, change_count, change_record_count, 129);

  auto loss = torch::empty({1}, site_rgba_f32.options().dtype(torch::kFloat32));
  auto grad_site_rgba = torch::empty({site_count, 4}, site_rgba_f32.options().dtype(torch::kFloat32));
  auto& k = kernels();
  constexpr uint64_t clear_threads = 256ull;
  constexpr uint64_t framegroup_threads = 32ull;
  const int64_t chunk_count = (frame_count + int64_t(framegroup_threads) - 1) / int64_t(framegroup_threads);
  TORCH_CHECK(
      track_chunk_change_offsets_i16.size(0) == track_count * (chunk_count + 1),
      "track_chunk_change_offsets_i16 length mismatch");
  auto chunk_offsets_cpu = track_chunk_change_offsets_i16.cpu();
  auto track_offsets_cpu = track_change_offsets_i32.cpu();
  const int16_t* chunk_offsets = chunk_offsets_cpu.data_ptr<int16_t>();
  const int32_t* track_offsets = track_offsets_cpu.data_ptr<int32_t>();
  for (int64_t track_id = 0; track_id < track_count; ++track_id) {
    const int32_t track_begin = track_offsets[track_id];
    const int32_t track_end = track_offsets[track_id + 1];
    TORCH_CHECK(track_begin >= 0 && track_end >= track_begin && track_end <= change_count, "track change offset bounds");
    int32_t previous = track_begin;
    for (int64_t chunk_id = 0; chunk_id <= chunk_count; ++chunk_id) {
      const int64_t chunk_index = track_id * (chunk_count + 1) + chunk_id;
      const int32_t value = static_cast<int32_t>(chunk_offsets[chunk_index]);
      TORCH_CHECK(
          value >= previous && value >= track_begin && value <= track_end,
          "track_chunk_change_offsets_i16 must be monotonic within each track and bounded by track changes");
      previous = value;
    }
    TORCH_CHECK(previous == track_end, "final chunk change offset must match track change end");
  }
  launch(k.clear_endpoint_loss_site_rgba_grad, [&](MetalKernelFunction& fn) {
    fn.setArg(0, loss);
    fn.setArg(1, grad_site_rgba);
    fn.setArg(2, config_i32);
    fn.dispatch((uint64_t)std::max<int64_t>(site_count, 1), clear_threads);
  });
  auto framegroup_kernel = k.endpoint_record_delta_replace_coeff16_i16x3_framegroup16_mse_vjp_direct_atomic_rgb_only;
  launch(
      framegroup_kernel,
      [&](MetalKernelFunction& fn) {
        fn.setArg(0, coeff_f16);
        fn.setArg(1, frame_t_f32);
        fn.setArg(2, base_offsets_i32);
        fn.setArg(3, base_record_i16);
        fn.setArg(4, track_change_offsets_i32);
        fn.setArg(5, track_chunk_change_offsets_i16);
        fn.setArg(6, change_frame_i32);
        fn.setArg(7, change_offsets_i32);
        fn.setArg(8, change_record_i16);
        fn.setArg(9, site_rgba_f32);
        fn.setArg(10, target_rgb_f32);
        fn.setArg(11, config_i32);
        fn.setArg(12, config_f32);
        fn.setArg(13, loss);
        fn.setArg(14, grad_site_rgba);
        fn.dispatch((uint64_t)track_count * (uint64_t)chunk_count * framegroup_threads, framegroup_threads);
  });
  return std::make_tuple(loss, grad_site_rgba);
}

std::tuple<torch::Tensor, torch::Tensor>
metal_endpoint_record_delta_replace_coeff16_i16x3_framegroup16_ownerreduce_mse_vjp_direct_atomic_rgb_only(
    const torch::Tensor& coeff_f16,
    const torch::Tensor& frame_t_f32,
    const torch::Tensor& base_offsets_i32,
    const torch::Tensor& base_record_i16,
    const torch::Tensor& track_change_offsets_i32,
    const torch::Tensor& track_chunk_change_offsets_i16,
    const torch::Tensor& track_chunk_owner_offsets_i32,
    const torch::Tensor& track_chunk_owner_i16,
    const torch::Tensor& change_frame_i32,
    const torch::Tensor& change_offsets_i32,
    const torch::Tensor& change_record_i16,
    const torch::Tensor& site_rgba_f32,
    const torch::Tensor& target_rgb_f32,
    const torch::Tensor& config_i32,
    const torch::Tensor& config_f32) {
  check_half_mps_2d(coeff_f16, "coeff_f16", 4);
  check_float_mps_1d_any(frame_t_f32, "frame_t_f32");
  check_i32_mps_1d_any(base_offsets_i32, "base_offsets_i32");
  check_i16_mps_1d_any(base_record_i16, "base_record_i16");
  check_i32_mps_1d_any(track_change_offsets_i32, "track_change_offsets_i32");
  check_i16_mps_1d_any(track_chunk_change_offsets_i16, "track_chunk_change_offsets_i16");
  check_i32_mps_1d_any(track_chunk_owner_offsets_i32, "track_chunk_owner_offsets_i32");
  check_i16_mps_1d_any(track_chunk_owner_i16, "track_chunk_owner_i16");
  check_i32_mps_1d_any(change_frame_i32, "change_frame_i32");
  check_i32_mps_1d_any(change_offsets_i32, "change_offsets_i32");
  check_i16_mps_1d_any(change_record_i16, "change_record_i16");
  check_float_mps_2d(site_rgba_f32, "site_rgba_f32", 4);
  TORCH_CHECK(target_rgb_f32.device().is_mps(), "target_rgb_f32 must be on MPS");
  TORCH_CHECK(target_rgb_f32.scalar_type() == torch::kFloat32, "target_rgb_f32 must be float32");
  TORCH_CHECK(target_rgb_f32.dim() == 3 && target_rgb_f32.size(2) == 3, "target_rgb_f32 must have shape [K,T,3]");
  TORCH_CHECK(target_rgb_f32.is_contiguous(), "target_rgb_f32 must be contiguous");
  check_i32_mps_1d(config_i32, "config_i32", 8);
  TORCH_CHECK(config_f32.device().is_mps(), "config_f32 must be on MPS");
  TORCH_CHECK(config_f32.scalar_type() == torch::kFloat32, "config_f32 must be float32");
  TORCH_CHECK(config_f32.dim() == 1 && config_f32.size(0) == 4, "config_f32 must have shape [4]");
  TORCH_CHECK(config_f32.is_contiguous(), "config_f32 must be contiguous");

  auto config_i32_cpu = config_i32.cpu();
  const int32_t* config = config_i32_cpu.data_ptr<int32_t>();
  const int64_t boundary_count = config[0];
  const int64_t track_count = config[1];
  const int64_t frame_count = config[2];
  const int64_t site_count = config[3];
  const int64_t base_record_count = config[4];
  const int64_t change_count = config[5];
  const int64_t change_record_count = config[6];
  const int64_t owner_list_count = config[7];
  TORCH_CHECK(boundary_count > 0, "config_i32[0] boundary count must be positive");
  TORCH_CHECK(
      boundary_count <= 32765,
      "i16x3 ownerreduce framegroup16 delta replace coeff16 fused MSE supports boundary count <= 32765");
  TORCH_CHECK(track_count > 0, "config_i32[1] track count must be positive");
  TORCH_CHECK(frame_count > 0, "config_i32[2] frame count must be positive");
  TORCH_CHECK(site_count == site_rgba_f32.size(0), "config_i32[3] must match site_rgba_f32 rows");
  TORCH_CHECK(
      site_count > 0 && site_count <= 32767,
      "i16x3 ownerreduce framegroup16 endpoint record delta replace coeff16 fused MSE VJP supports site count in [1, 32767]");
  TORCH_CHECK(base_record_count >= 0, "config_i32[4] base record count must be nonnegative");
  TORCH_CHECK(change_count >= 0, "config_i32[5] change count must be nonnegative");
  TORCH_CHECK(change_count <= 32767, "int16 chunk-start offsets require change count <= 32767");
  TORCH_CHECK(change_record_count >= 0, "config_i32[6] change record count must be nonnegative");
  TORCH_CHECK(owner_list_count >= 0, "config_i32[7] owner list count must be nonnegative");
  TORCH_CHECK(coeff_f16.size(0) == track_count * boundary_count, "coeff_f16 row count mismatch");
  TORCH_CHECK(frame_t_f32.size(0) == frame_count, "frame_t_f32 length must match frame count");
  TORCH_CHECK(base_record_i16.size(0) == base_record_count * 3, "base_record_i16 length mismatch");
  TORCH_CHECK(change_frame_i32.size(0) == change_count, "change_frame_i32 length must match change count");
  TORCH_CHECK(change_record_i16.size(0) == change_record_count * 3, "change_record_i16 length mismatch");
  TORCH_CHECK(track_chunk_owner_i16.size(0) == owner_list_count, "track_chunk_owner_i16 length mismatch");
  TORCH_CHECK(base_offsets_i32.size(0) == track_count + 1, "base_offsets_i32 length must be track_count + 1");
  TORCH_CHECK(
      track_change_offsets_i32.size(0) == track_count + 1,
      "track_change_offsets_i32 length must be track_count + 1");
  TORCH_CHECK(change_offsets_i32.size(0) == change_count + 1, "change_offsets_i32 must be change_count + 1");
  TORCH_CHECK(
      target_rgb_f32.size(0) == track_count && target_rgb_f32.size(1) == frame_count,
      "target_rgb_f32 shape mismatch");
  check_segment_tape_offsets_cpu(base_offsets_i32, track_count, base_record_count, 129);
  check_segment_tape_offsets_cpu(track_change_offsets_i32, track_count, change_count, 2147483647);
  check_segment_tape_offsets_cpu(change_offsets_i32, change_count, change_record_count, 129);

  auto loss = torch::empty({1}, site_rgba_f32.options().dtype(torch::kFloat32));
  auto grad_site_rgba = torch::empty({site_count, 4}, site_rgba_f32.options().dtype(torch::kFloat32));
  auto& k = kernels();
  constexpr uint64_t clear_threads = 256ull;
  constexpr uint64_t framegroup_threads = 32ull;
  const int64_t chunk_count = (frame_count + int64_t(framegroup_threads) - 1) / int64_t(framegroup_threads);
  TORCH_CHECK(
      track_chunk_change_offsets_i16.size(0) == track_count * (chunk_count + 1),
      "track_chunk_change_offsets_i16 length mismatch");
  TORCH_CHECK(
      track_chunk_owner_offsets_i32.size(0) == track_count * chunk_count + 1,
      "track_chunk_owner_offsets_i32 length mismatch");
  auto chunk_offsets_cpu = track_chunk_change_offsets_i16.cpu();
  auto track_offsets_cpu = track_change_offsets_i32.cpu();
  const int16_t* chunk_offsets = chunk_offsets_cpu.data_ptr<int16_t>();
  const int32_t* track_offsets = track_offsets_cpu.data_ptr<int32_t>();
  for (int64_t track_id = 0; track_id < track_count; ++track_id) {
    const int32_t track_begin = track_offsets[track_id];
    const int32_t track_end = track_offsets[track_id + 1];
    TORCH_CHECK(track_begin >= 0 && track_end >= track_begin && track_end <= change_count, "track change offset bounds");
    int32_t previous = track_begin;
    for (int64_t chunk_id = 0; chunk_id <= chunk_count; ++chunk_id) {
      const int64_t chunk_index = track_id * (chunk_count + 1) + chunk_id;
      const int32_t value = static_cast<int32_t>(chunk_offsets[chunk_index]);
      TORCH_CHECK(
          value >= previous && value >= track_begin && value <= track_end,
          "track_chunk_change_offsets_i16 must be monotonic within each track and bounded by track changes");
      previous = value;
    }
    TORCH_CHECK(previous == track_end, "final chunk change offset must match track change end");
  }
  check_segment_tape_offsets_cpu(
      track_chunk_owner_offsets_i32,
      track_count * chunk_count,
      owner_list_count,
      2147483647);
  auto owner_ids_cpu = track_chunk_owner_i16.cpu();
  const int16_t* owner_ids = owner_ids_cpu.data_ptr<int16_t>();
  for (int64_t index = 0; index < owner_list_count; ++index) {
    const int32_t owner = static_cast<int32_t>(owner_ids[index]);
    TORCH_CHECK(owner >= 0 && owner < site_count, "track_chunk_owner_i16 ids must be in [0, site_count)");
  }

  launch(k.clear_endpoint_loss_site_rgba_grad, [&](MetalKernelFunction& fn) {
    fn.setArg(0, loss);
    fn.setArg(1, grad_site_rgba);
    fn.setArg(2, config_i32);
    fn.dispatch((uint64_t)std::max<int64_t>(site_count, 1), clear_threads);
  });
  auto framegroup_kernel =
      k.endpoint_record_delta_replace_coeff16_i16x3_framegroup16_ownerreduce_mse_vjp_direct_atomic_rgb_only;
  launch(
      framegroup_kernel,
      [&](MetalKernelFunction& fn) {
        fn.setArg(0, coeff_f16);
        fn.setArg(1, frame_t_f32);
        fn.setArg(2, base_offsets_i32);
        fn.setArg(3, base_record_i16);
        fn.setArg(4, track_change_offsets_i32);
        fn.setArg(5, track_chunk_change_offsets_i16);
        fn.setArg(6, track_chunk_owner_offsets_i32);
        fn.setArg(7, track_chunk_owner_i16);
        fn.setArg(8, change_frame_i32);
        fn.setArg(9, change_offsets_i32);
        fn.setArg(10, change_record_i16);
        fn.setArg(11, site_rgba_f32);
        fn.setArg(12, target_rgb_f32);
        fn.setArg(13, config_i32);
        fn.setArg(14, config_f32);
        fn.setArg(15, loss);
        fn.setArg(16, grad_site_rgba);
        fn.dispatch((uint64_t)track_count * (uint64_t)chunk_count * framegroup_threads, framegroup_threads);
  });
  return std::make_tuple(loss, grad_site_rgba);
}

std::tuple<torch::Tensor, torch::Tensor>
metal_endpoint_record_delta_replace_coeff16_i16cols_framegroup16_mse_vjp_direct_atomic_rgb_only(
    const torch::Tensor& coeff_f16,
    const torch::Tensor& frame_t_f32,
    const torch::Tensor& base_offsets_i32,
    const torch::Tensor& base_record_i16,
    const torch::Tensor& track_change_offsets_i32,
    const torch::Tensor& track_chunk_change_offsets_i16,
    const torch::Tensor& change_frame_i32,
    const torch::Tensor& change_offsets_i32,
    const torch::Tensor& change_record_i16,
    const torch::Tensor& site_rgba_f32,
    const torch::Tensor& target_rgb_f32,
    const torch::Tensor& config_i32,
    const torch::Tensor& config_f32) {
  check_half_mps_2d(coeff_f16, "coeff_f16", 4);
  check_float_mps_1d_any(frame_t_f32, "frame_t_f32");
  check_i32_mps_1d_any(base_offsets_i32, "base_offsets_i32");
  check_i16_mps_1d_any(base_record_i16, "base_record_i16");
  check_i32_mps_1d_any(track_change_offsets_i32, "track_change_offsets_i32");
  check_i16_mps_1d_any(track_chunk_change_offsets_i16, "track_chunk_change_offsets_i16");
  check_i32_mps_1d_any(change_frame_i32, "change_frame_i32");
  check_i32_mps_1d_any(change_offsets_i32, "change_offsets_i32");
  check_i16_mps_1d_any(change_record_i16, "change_record_i16");
  check_float_mps_2d(site_rgba_f32, "site_rgba_f32", 4);
  TORCH_CHECK(target_rgb_f32.device().is_mps(), "target_rgb_f32 must be on MPS");
  TORCH_CHECK(target_rgb_f32.scalar_type() == torch::kFloat32, "target_rgb_f32 must be float32");
  TORCH_CHECK(target_rgb_f32.dim() == 3 && target_rgb_f32.size(2) == 3, "target_rgb_f32 must have shape [K,T,3]");
  TORCH_CHECK(target_rgb_f32.is_contiguous(), "target_rgb_f32 must be contiguous");
  check_i32_mps_1d(config_i32, "config_i32", 7);
  TORCH_CHECK(config_f32.device().is_mps(), "config_f32 must be on MPS");
  TORCH_CHECK(config_f32.scalar_type() == torch::kFloat32, "config_f32 must be float32");
  TORCH_CHECK(config_f32.dim() == 1 && config_f32.size(0) == 4, "config_f32 must have shape [4]");
  TORCH_CHECK(config_f32.is_contiguous(), "config_f32 must be contiguous");

  auto config_i32_cpu = config_i32.cpu();
  const int32_t* config = config_i32_cpu.data_ptr<int32_t>();
  const int64_t boundary_count = config[0];
  const int64_t track_count = config[1];
  const int64_t frame_count = config[2];
  const int64_t site_count = config[3];
  const int64_t base_record_count = config[4];
  const int64_t change_count = config[5];
  const int64_t change_record_count = config[6];
  TORCH_CHECK(boundary_count > 0, "config_i32[0] boundary count must be positive");
  TORCH_CHECK(boundary_count <= 32765, "i16cols framegroup16 delta replace coeff16 fused MSE supports boundary count <= 32765");
  TORCH_CHECK(track_count > 0, "config_i32[1] track count must be positive");
  TORCH_CHECK(frame_count > 0, "config_i32[2] frame count must be positive");
  TORCH_CHECK(site_count == site_rgba_f32.size(0), "config_i32[3] must match site_rgba_f32 rows");
  TORCH_CHECK(
      site_count > 0 && site_count <= 32767,
      "i16cols framegroup16 endpoint record delta replace coeff16 fused MSE VJP supports site count in [1, 32767]");
  TORCH_CHECK(base_record_count >= 0, "config_i32[4] base record count must be nonnegative");
  TORCH_CHECK(change_count >= 0, "config_i32[5] change count must be nonnegative");
  TORCH_CHECK(change_count <= 32767, "int16 chunk-start offsets require change count <= 32767");
  TORCH_CHECK(change_record_count >= 0, "config_i32[6] change record count must be nonnegative");
  TORCH_CHECK(coeff_f16.size(0) == track_count * boundary_count, "coeff_f16 row count mismatch");
  TORCH_CHECK(frame_t_f32.size(0) == frame_count, "frame_t_f32 length must match frame count");
  TORCH_CHECK(base_record_i16.size(0) == base_record_count * 3, "base_record_i16 length mismatch");
  TORCH_CHECK(change_frame_i32.size(0) == change_count, "change_frame_i32 length must match change count");
  TORCH_CHECK(change_record_i16.size(0) == change_record_count * 3, "change_record_i16 length mismatch");
  TORCH_CHECK(base_offsets_i32.size(0) == track_count + 1, "base_offsets_i32 length must be track_count + 1");
  TORCH_CHECK(
      track_change_offsets_i32.size(0) == track_count + 1,
      "track_change_offsets_i32 length must be track_count + 1");
  TORCH_CHECK(change_offsets_i32.size(0) == change_count + 1, "change_offsets_i32 length must be change_count + 1");
  TORCH_CHECK(
      target_rgb_f32.size(0) == track_count && target_rgb_f32.size(1) == frame_count,
      "target_rgb_f32 shape mismatch");
  check_segment_tape_offsets_cpu(base_offsets_i32, track_count, base_record_count, 129);
  check_segment_tape_offsets_cpu(track_change_offsets_i32, track_count, change_count, 2147483647);
  check_segment_tape_offsets_cpu(change_offsets_i32, change_count, change_record_count, 129);

  auto loss = torch::empty({1}, site_rgba_f32.options().dtype(torch::kFloat32));
  auto grad_site_rgba = torch::empty({site_count, 4}, site_rgba_f32.options().dtype(torch::kFloat32));
  auto& k = kernels();
  constexpr uint64_t clear_threads = 256ull;
  constexpr uint64_t framegroup_threads = 32ull;
  const int64_t chunk_count = (frame_count + int64_t(framegroup_threads) - 1) / int64_t(framegroup_threads);
  TORCH_CHECK(
      track_chunk_change_offsets_i16.size(0) == track_count * (chunk_count + 1),
      "track_chunk_change_offsets_i16 length mismatch");
  auto chunk_offsets_cpu = track_chunk_change_offsets_i16.cpu();
  auto track_offsets_cpu = track_change_offsets_i32.cpu();
  const int16_t* chunk_offsets = chunk_offsets_cpu.data_ptr<int16_t>();
  const int32_t* track_offsets = track_offsets_cpu.data_ptr<int32_t>();
  for (int64_t track_id = 0; track_id < track_count; ++track_id) {
    const int32_t track_begin = track_offsets[track_id];
    const int32_t track_end = track_offsets[track_id + 1];
    TORCH_CHECK(track_begin >= 0 && track_end >= track_begin && track_end <= change_count, "track change offset bounds");
    int32_t previous = track_begin;
    for (int64_t chunk_id = 0; chunk_id <= chunk_count; ++chunk_id) {
      const int64_t chunk_index = track_id * (chunk_count + 1) + chunk_id;
      const int32_t value = static_cast<int32_t>(chunk_offsets[chunk_index]);
      TORCH_CHECK(
          value >= previous && value >= track_begin && value <= track_end,
          "track_chunk_change_offsets_i16 must be monotonic within each track and bounded by track changes");
      previous = value;
    }
    TORCH_CHECK(previous == track_end, "final chunk change offset must match track change end");
  }
  launch(k.clear_endpoint_loss_site_rgba_grad, [&](MetalKernelFunction& fn) {
    fn.setArg(0, loss);
    fn.setArg(1, grad_site_rgba);
    fn.setArg(2, config_i32);
    fn.dispatch((uint64_t)std::max<int64_t>(site_count, 1), clear_threads);
  });
  auto framegroup_kernel = k.endpoint_record_delta_replace_coeff16_i16cols_framegroup16_mse_vjp_direct_atomic_rgb_only;
  launch(
      framegroup_kernel,
      [&](MetalKernelFunction& fn) {
        fn.setArg(0, coeff_f16);
        fn.setArg(1, frame_t_f32);
        fn.setArg(2, base_offsets_i32);
        fn.setArg(3, base_record_i16);
        fn.setArg(4, track_change_offsets_i32);
        fn.setArg(5, track_chunk_change_offsets_i16);
        fn.setArg(6, change_frame_i32);
        fn.setArg(7, change_offsets_i32);
        fn.setArg(8, change_record_i16);
        fn.setArg(9, site_rgba_f32);
        fn.setArg(10, target_rgb_f32);
        fn.setArg(11, config_i32);
        fn.setArg(12, config_f32);
        fn.setArg(13, loss);
        fn.setArg(14, grad_site_rgba);
        fn.dispatch((uint64_t)track_count * (uint64_t)chunk_count * framegroup_threads, framegroup_threads);
  });
  return std::make_tuple(loss, grad_site_rgba);
}

std::tuple<torch::Tensor, torch::Tensor>
metal_endpoint_record_delta_replace_coeff16_i16x3_framegroup64_mse_vjp_direct_atomic_rgb_only(
    const torch::Tensor& coeff_f16,
    const torch::Tensor& frame_t_f32,
    const torch::Tensor& base_offsets_i32,
    const torch::Tensor& base_record_i16,
    const torch::Tensor& track_change_offsets_i32,
    const torch::Tensor& track_chunk_change_offsets_i16,
    const torch::Tensor& change_frame_i32,
    const torch::Tensor& change_offsets_i32,
    const torch::Tensor& change_record_i16,
    const torch::Tensor& site_rgba_f32,
    const torch::Tensor& target_rgb_f32,
    const torch::Tensor& config_i32,
    const torch::Tensor& config_f32) {
  check_half_mps_2d(coeff_f16, "coeff_f16", 4);
  check_float_mps_1d_any(frame_t_f32, "frame_t_f32");
  check_i32_mps_1d_any(base_offsets_i32, "base_offsets_i32");
  check_i16_mps_1d_any(base_record_i16, "base_record_i16");
  check_i32_mps_1d_any(track_change_offsets_i32, "track_change_offsets_i32");
  check_i16_mps_1d_any(track_chunk_change_offsets_i16, "track_chunk_change_offsets_i16");
  check_i32_mps_1d_any(change_frame_i32, "change_frame_i32");
  check_i32_mps_1d_any(change_offsets_i32, "change_offsets_i32");
  check_i16_mps_1d_any(change_record_i16, "change_record_i16");
  check_float_mps_2d(site_rgba_f32, "site_rgba_f32", 4);
  TORCH_CHECK(target_rgb_f32.device().is_mps(), "target_rgb_f32 must be on MPS");
  TORCH_CHECK(target_rgb_f32.scalar_type() == torch::kFloat32, "target_rgb_f32 must be float32");
  TORCH_CHECK(target_rgb_f32.dim() == 3 && target_rgb_f32.size(2) == 3, "target_rgb_f32 must have shape [K,T,3]");
  TORCH_CHECK(target_rgb_f32.is_contiguous(), "target_rgb_f32 must be contiguous");
  check_i32_mps_1d(config_i32, "config_i32", 7);
  TORCH_CHECK(config_f32.device().is_mps(), "config_f32 must be on MPS");
  TORCH_CHECK(config_f32.scalar_type() == torch::kFloat32, "config_f32 must be float32");
  TORCH_CHECK(config_f32.dim() == 1 && config_f32.size(0) == 4, "config_f32 must have shape [4]");
  TORCH_CHECK(config_f32.is_contiguous(), "config_f32 must be contiguous");

  auto config_i32_cpu = config_i32.cpu();
  const int32_t* config = config_i32_cpu.data_ptr<int32_t>();
  const int64_t boundary_count = config[0];
  const int64_t track_count = config[1];
  const int64_t frame_count = config[2];
  const int64_t site_count = config[3];
  const int64_t base_record_count = config[4];
  const int64_t change_count = config[5];
  const int64_t change_record_count = config[6];
  TORCH_CHECK(boundary_count > 0, "config_i32[0] boundary count must be positive");
  TORCH_CHECK(boundary_count <= 32765, "i16x3 framegroup64 delta replace coeff16 fused MSE supports boundary count <= 32765");
  TORCH_CHECK(track_count > 0, "config_i32[1] track count must be positive");
  TORCH_CHECK(frame_count > 0, "config_i32[2] frame count must be positive");
  TORCH_CHECK(site_count == site_rgba_f32.size(0), "config_i32[3] must match site_rgba_f32 rows");
  TORCH_CHECK(
      site_count > 0 && site_count <= 32767,
      "i16x3 framegroup64 endpoint record delta replace coeff16 fused MSE VJP supports site count in [1, 32767]");
  TORCH_CHECK(base_record_count >= 0, "config_i32[4] base record count must be nonnegative");
  TORCH_CHECK(change_count >= 0, "config_i32[5] change count must be nonnegative");
  TORCH_CHECK(change_count <= 32767, "int16 chunk-start offsets require change count <= 32767");
  TORCH_CHECK(change_record_count >= 0, "config_i32[6] change record count must be nonnegative");
  TORCH_CHECK(coeff_f16.size(0) == track_count * boundary_count, "coeff_f16 row count mismatch");
  TORCH_CHECK(frame_t_f32.size(0) == frame_count, "frame_t_f32 length must match frame count");
  TORCH_CHECK(base_record_i16.size(0) == base_record_count * 3, "base_record_i16 length mismatch");
  TORCH_CHECK(change_frame_i32.size(0) == change_count, "change_frame_i32 length must match change count");
  TORCH_CHECK(change_record_i16.size(0) == change_record_count * 3, "change_record_i16 length mismatch");
  TORCH_CHECK(base_offsets_i32.size(0) == track_count + 1, "base_offsets_i32 length must be track_count + 1");
  TORCH_CHECK(
      track_change_offsets_i32.size(0) == track_count + 1,
      "track_change_offsets_i32 length must be track_count + 1");
  TORCH_CHECK(change_offsets_i32.size(0) == change_count + 1, "change_offsets_i32 length must be change_count + 1");
  TORCH_CHECK(
      target_rgb_f32.size(0) == track_count && target_rgb_f32.size(1) == frame_count,
      "target_rgb_f32 shape mismatch");
  check_segment_tape_offsets_cpu(base_offsets_i32, track_count, base_record_count, 129);
  check_segment_tape_offsets_cpu(track_change_offsets_i32, track_count, change_count, 2147483647);
  check_segment_tape_offsets_cpu(change_offsets_i32, change_count, change_record_count, 129);

  auto loss = torch::empty({1}, site_rgba_f32.options().dtype(torch::kFloat32));
  auto grad_site_rgba = torch::empty({site_count, 4}, site_rgba_f32.options().dtype(torch::kFloat32));
  auto& k = kernels();
  constexpr uint64_t clear_threads = 256ull;
  constexpr uint64_t framegroup_threads = 64ull;
  const int64_t chunk_count = (frame_count + int64_t(framegroup_threads) - 1) / int64_t(framegroup_threads);
  TORCH_CHECK(
      track_chunk_change_offsets_i16.size(0) == track_count * (chunk_count + 1),
      "track_chunk_change_offsets_i16 length mismatch");
  auto chunk_offsets_cpu = track_chunk_change_offsets_i16.cpu();
  auto track_offsets_cpu = track_change_offsets_i32.cpu();
  const int16_t* chunk_offsets = chunk_offsets_cpu.data_ptr<int16_t>();
  const int32_t* track_offsets = track_offsets_cpu.data_ptr<int32_t>();
  for (int64_t track_id = 0; track_id < track_count; ++track_id) {
    const int32_t track_begin = track_offsets[track_id];
    const int32_t track_end = track_offsets[track_id + 1];
    TORCH_CHECK(track_begin >= 0 && track_end >= track_begin && track_end <= change_count, "track change offset bounds");
    int32_t previous = track_begin;
    for (int64_t chunk_id = 0; chunk_id <= chunk_count; ++chunk_id) {
      const int64_t chunk_index = track_id * (chunk_count + 1) + chunk_id;
      const int32_t value = static_cast<int32_t>(chunk_offsets[chunk_index]);
      TORCH_CHECK(
          value >= previous && value >= track_begin && value <= track_end,
          "track_chunk_change_offsets_i16 must be monotonic within each track and bounded by track changes");
      previous = value;
    }
    TORCH_CHECK(previous == track_end, "final chunk change offset must match track change end");
  }
  launch(k.clear_endpoint_loss_site_rgba_grad, [&](MetalKernelFunction& fn) {
    fn.setArg(0, loss);
    fn.setArg(1, grad_site_rgba);
    fn.setArg(2, config_i32);
    fn.dispatch((uint64_t)std::max<int64_t>(site_count, 1), clear_threads);
  });
  auto framegroup_kernel = k.endpoint_record_delta_replace_coeff16_i16x3_framegroup64_mse_vjp_direct_atomic_rgb_only;
  launch(
      framegroup_kernel,
      [&](MetalKernelFunction& fn) {
        fn.setArg(0, coeff_f16);
        fn.setArg(1, frame_t_f32);
        fn.setArg(2, base_offsets_i32);
        fn.setArg(3, base_record_i16);
        fn.setArg(4, track_change_offsets_i32);
        fn.setArg(5, track_chunk_change_offsets_i16);
        fn.setArg(6, change_frame_i32);
        fn.setArg(7, change_offsets_i32);
        fn.setArg(8, change_record_i16);
        fn.setArg(9, site_rgba_f32);
        fn.setArg(10, target_rgb_f32);
        fn.setArg(11, config_i32);
        fn.setArg(12, config_f32);
        fn.setArg(13, loss);
        fn.setArg(14, grad_site_rgba);
        fn.dispatch((uint64_t)track_count * (uint64_t)chunk_count * framegroup_threads, framegroup_threads);
  });
  return std::make_tuple(loss, grad_site_rgba);
}

std::tuple<torch::Tensor, torch::Tensor>
metal_endpoint_record_delta_replace_coeff16_packed_mse_vjp_direct_atomic_rgb_only(
    const torch::Tensor& coeff_f16,
    const torch::Tensor& frame_t_f32,
    const torch::Tensor& base_offsets_i32,
    const torch::Tensor& base_record_i32,
    const torch::Tensor& track_change_offsets_i32,
    const torch::Tensor& change_frame_i32,
    const torch::Tensor& change_offsets_i32,
    const torch::Tensor& change_record_i32,
    const torch::Tensor& site_rgba_f32,
    const torch::Tensor& target_rgb_f32,
    const torch::Tensor& config_i32,
    const torch::Tensor& config_f32) {
  check_half_mps_2d(coeff_f16, "coeff_f16", 4);
  check_float_mps_1d_any(frame_t_f32, "frame_t_f32");
  check_i32_mps_1d_any(base_offsets_i32, "base_offsets_i32");
  check_i32_mps_1d_any(base_record_i32, "base_record_i32");
  check_i32_mps_1d_any(track_change_offsets_i32, "track_change_offsets_i32");
  check_i32_mps_1d_any(change_frame_i32, "change_frame_i32");
  check_i32_mps_1d_any(change_offsets_i32, "change_offsets_i32");
  check_i32_mps_1d_any(change_record_i32, "change_record_i32");
  check_float_mps_2d(site_rgba_f32, "site_rgba_f32", 4);
  TORCH_CHECK(target_rgb_f32.device().is_mps(), "target_rgb_f32 must be on MPS");
  TORCH_CHECK(target_rgb_f32.scalar_type() == torch::kFloat32, "target_rgb_f32 must be float32");
  TORCH_CHECK(target_rgb_f32.dim() == 3 && target_rgb_f32.size(2) == 3, "target_rgb_f32 must have shape [K,T,3]");
  TORCH_CHECK(target_rgb_f32.is_contiguous(), "target_rgb_f32 must be contiguous");
  check_i32_mps_1d(config_i32, "config_i32", 7);
  TORCH_CHECK(config_f32.device().is_mps(), "config_f32 must be on MPS");
  TORCH_CHECK(config_f32.scalar_type() == torch::kFloat32, "config_f32 must be float32");
  TORCH_CHECK(config_f32.dim() == 1 && config_f32.size(0) == 4, "config_f32 must have shape [4]");
  TORCH_CHECK(config_f32.is_contiguous(), "config_f32 must be contiguous");

  auto config_i32_cpu = config_i32.cpu();
  const int32_t* config = config_i32_cpu.data_ptr<int32_t>();
  const int64_t boundary_count = config[0];
  const int64_t track_count = config[1];
  const int64_t frame_count = config[2];
  const int64_t site_count = config[3];
  const int64_t base_record_count = config[4];
  const int64_t change_count = config[5];
  const int64_t change_record_count = config[6];
  TORCH_CHECK(boundary_count > 0, "config_i32[0] boundary count must be positive");
  TORCH_CHECK(boundary_count <= 4093, "packed scalar delta replace coeff16 fused MSE supports boundary count <= 4093");
  TORCH_CHECK(track_count > 0, "config_i32[1] track count must be positive");
  TORCH_CHECK(frame_count > 0, "config_i32[2] frame count must be positive");
  TORCH_CHECK(site_count == site_rgba_f32.size(0), "config_i32[3] must match site_rgba_f32 rows");
  TORCH_CHECK(
      site_count > 0 && site_count <= 256,
      "packed scalar endpoint record delta replace coeff16 fused MSE VJP supports site count in [1, 256]");
  TORCH_CHECK(base_record_count >= 0, "config_i32[4] base record count must be nonnegative");
  TORCH_CHECK(change_count >= 0, "config_i32[5] change count must be nonnegative");
  TORCH_CHECK(change_record_count >= 0, "config_i32[6] change record count must be nonnegative");
  TORCH_CHECK(coeff_f16.size(0) == track_count * boundary_count, "coeff_f16 row count mismatch");
  TORCH_CHECK(frame_t_f32.size(0) == frame_count, "frame_t_f32 length must match frame count");
  TORCH_CHECK(base_record_i32.size(0) == base_record_count, "base_record_i32 length mismatch");
  TORCH_CHECK(change_frame_i32.size(0) == change_count, "change_frame_i32 length must match change count");
  TORCH_CHECK(change_record_i32.size(0) == change_record_count, "change_record_i32 length mismatch");
  TORCH_CHECK(base_offsets_i32.size(0) == track_count + 1, "base_offsets_i32 length must be track_count + 1");
  TORCH_CHECK(
      track_change_offsets_i32.size(0) == track_count + 1,
      "track_change_offsets_i32 length must be track_count + 1");
  TORCH_CHECK(change_offsets_i32.size(0) == change_count + 1, "change_offsets_i32 length must be change_count + 1");
  TORCH_CHECK(
      target_rgb_f32.size(0) == track_count && target_rgb_f32.size(1) == frame_count,
      "target_rgb_f32 shape mismatch");
  check_segment_tape_offsets_cpu(base_offsets_i32, track_count, base_record_count, 129);
  check_segment_tape_offsets_cpu(track_change_offsets_i32, track_count, change_count, 2147483647);
  check_segment_tape_offsets_cpu(change_offsets_i32, change_count, change_record_count, 129);

  auto loss = torch::empty({1}, site_rgba_f32.options().dtype(torch::kFloat32));
  auto grad_site_rgba = torch::empty({site_count, 4}, site_rgba_f32.options().dtype(torch::kFloat32));
  auto& k = kernels();
  constexpr uint64_t threads = 256ull;
  const int64_t sample_count = track_count * frame_count;
  launch(k.clear_endpoint_loss_site_rgba_grad, [&](MetalKernelFunction& fn) {
    fn.setArg(0, loss);
    fn.setArg(1, grad_site_rgba);
    fn.setArg(2, config_i32);
    fn.dispatch((uint64_t)std::max<int64_t>(site_count, 1), threads);
  });
  launch(
      k.endpoint_record_delta_replace_coeff16_packed_mse_vjp_direct_atomic_rgb_only,
      [&](MetalKernelFunction& fn) {
        fn.setArg(0, coeff_f16);
        fn.setArg(1, frame_t_f32);
        fn.setArg(2, base_offsets_i32);
        fn.setArg(3, base_record_i32);
        fn.setArg(4, track_change_offsets_i32);
        fn.setArg(5, change_frame_i32);
        fn.setArg(6, change_offsets_i32);
        fn.setArg(7, change_record_i32);
        fn.setArg(8, site_rgba_f32);
        fn.setArg(9, target_rgb_f32);
        fn.setArg(10, config_i32);
        fn.setArg(11, config_f32);
        fn.setArg(12, loss);
        fn.setArg(13, grad_site_rgba);
        fn.dispatch((uint64_t)sample_count, threads);
      });
  return std::make_tuple(loss, grad_site_rgba);
}

std::tuple<torch::Tensor, torch::Tensor>
metal_endpoint_record_delta_replace_coeff16_packed_framegroup16_mse_vjp_direct_atomic_rgb_only(
    const torch::Tensor& coeff_f16,
    const torch::Tensor& frame_t_f32,
    const torch::Tensor& base_offsets_i32,
    const torch::Tensor& base_record_i32,
    const torch::Tensor& track_change_offsets_i32,
    const torch::Tensor& track_chunk_change_offsets_i16,
    const torch::Tensor& change_frame_i32,
    const torch::Tensor& change_offsets_i32,
    const torch::Tensor& change_record_i32,
    const torch::Tensor& site_rgba_f32,
    const torch::Tensor& target_rgb_f32,
    const torch::Tensor& config_i32,
    const torch::Tensor& config_f32) {
  check_half_mps_2d(coeff_f16, "coeff_f16", 4);
  check_float_mps_1d_any(frame_t_f32, "frame_t_f32");
  check_i32_mps_1d_any(base_offsets_i32, "base_offsets_i32");
  check_i32_mps_1d_any(base_record_i32, "base_record_i32");
  check_i32_mps_1d_any(track_change_offsets_i32, "track_change_offsets_i32");
  check_i16_mps_1d_any(track_chunk_change_offsets_i16, "track_chunk_change_offsets_i16");
  check_i32_mps_1d_any(change_frame_i32, "change_frame_i32");
  check_i32_mps_1d_any(change_offsets_i32, "change_offsets_i32");
  check_i32_mps_1d_any(change_record_i32, "change_record_i32");
  check_float_mps_2d(site_rgba_f32, "site_rgba_f32", 4);
  TORCH_CHECK(target_rgb_f32.device().is_mps(), "target_rgb_f32 must be on MPS");
  TORCH_CHECK(target_rgb_f32.scalar_type() == torch::kFloat32, "target_rgb_f32 must be float32");
  TORCH_CHECK(target_rgb_f32.dim() == 3 && target_rgb_f32.size(2) == 3, "target_rgb_f32 must have shape [K,T,3]");
  TORCH_CHECK(target_rgb_f32.is_contiguous(), "target_rgb_f32 must be contiguous");
  check_i32_mps_1d(config_i32, "config_i32", 7);
  TORCH_CHECK(config_f32.device().is_mps(), "config_f32 must be on MPS");
  TORCH_CHECK(config_f32.scalar_type() == torch::kFloat32, "config_f32 must be float32");
  TORCH_CHECK(config_f32.dim() == 1 && config_f32.size(0) == 4, "config_f32 must have shape [4]");
  TORCH_CHECK(config_f32.is_contiguous(), "config_f32 must be contiguous");

  auto config_i32_cpu = config_i32.cpu();
  const int32_t* config = config_i32_cpu.data_ptr<int32_t>();
  const int64_t boundary_count = config[0];
  const int64_t track_count = config[1];
  const int64_t frame_count = config[2];
  const int64_t site_count = config[3];
  const int64_t base_record_count = config[4];
  const int64_t change_count = config[5];
  const int64_t change_record_count = config[6];
  TORCH_CHECK(boundary_count > 0, "config_i32[0] boundary count must be positive");
  TORCH_CHECK(boundary_count <= 4093, "packed framegroup16 delta replace coeff16 fused MSE supports boundary count <= 4093");
  TORCH_CHECK(track_count > 0, "config_i32[1] track count must be positive");
  TORCH_CHECK(frame_count > 0, "config_i32[2] frame count must be positive");
  TORCH_CHECK(site_count == site_rgba_f32.size(0), "config_i32[3] must match site_rgba_f32 rows");
  TORCH_CHECK(
      site_count > 0 && site_count <= 256,
      "packed framegroup16 endpoint record delta replace coeff16 fused MSE VJP supports site count in [1, 256]");
  TORCH_CHECK(base_record_count >= 0, "config_i32[4] base record count must be nonnegative");
  TORCH_CHECK(change_count >= 0, "config_i32[5] change count must be nonnegative");
  TORCH_CHECK(change_count <= 32767, "int16 chunk-start offsets require change count <= 32767");
  TORCH_CHECK(change_record_count >= 0, "config_i32[6] change record count must be nonnegative");
  TORCH_CHECK(coeff_f16.size(0) == track_count * boundary_count, "coeff_f16 row count mismatch");
  TORCH_CHECK(frame_t_f32.size(0) == frame_count, "frame_t_f32 length must match frame count");
  TORCH_CHECK(base_record_i32.size(0) == base_record_count, "base_record_i32 length mismatch");
  TORCH_CHECK(change_frame_i32.size(0) == change_count, "change_frame_i32 length must match change count");
  TORCH_CHECK(change_record_i32.size(0) == change_record_count, "change_record_i32 length mismatch");
  TORCH_CHECK(base_offsets_i32.size(0) == track_count + 1, "base_offsets_i32 length must be track_count + 1");
  TORCH_CHECK(
      track_change_offsets_i32.size(0) == track_count + 1,
      "track_change_offsets_i32 length must be track_count + 1");
  TORCH_CHECK(change_offsets_i32.size(0) == change_count + 1, "change_offsets_i32 length must be change_count + 1");
  TORCH_CHECK(
      target_rgb_f32.size(0) == track_count && target_rgb_f32.size(1) == frame_count,
      "target_rgb_f32 shape mismatch");
  check_segment_tape_offsets_cpu(base_offsets_i32, track_count, base_record_count, 129);
  check_segment_tape_offsets_cpu(track_change_offsets_i32, track_count, change_count, 2147483647);
  check_segment_tape_offsets_cpu(change_offsets_i32, change_count, change_record_count, 129);

  auto loss = torch::empty({1}, site_rgba_f32.options().dtype(torch::kFloat32));
  auto grad_site_rgba = torch::empty({site_count, 4}, site_rgba_f32.options().dtype(torch::kFloat32));
  auto& k = kernels();
  constexpr uint64_t clear_threads = 256ull;
  constexpr uint64_t framegroup_threads = 32ull;
  const int64_t chunk_count = (frame_count + int64_t(framegroup_threads) - 1) / int64_t(framegroup_threads);
  TORCH_CHECK(
      track_chunk_change_offsets_i16.size(0) == track_count * (chunk_count + 1),
      "track_chunk_change_offsets_i16 length mismatch");
  auto chunk_offsets_cpu = track_chunk_change_offsets_i16.cpu();
  auto track_offsets_cpu = track_change_offsets_i32.cpu();
  const int16_t* chunk_offsets = chunk_offsets_cpu.data_ptr<int16_t>();
  const int32_t* track_offsets = track_offsets_cpu.data_ptr<int32_t>();
  for (int64_t track_id = 0; track_id < track_count; ++track_id) {
    const int32_t track_begin = track_offsets[track_id];
    const int32_t track_end = track_offsets[track_id + 1];
    TORCH_CHECK(track_begin >= 0 && track_end >= track_begin && track_end <= change_count, "track change offset bounds");
    int32_t previous = track_begin;
    for (int64_t chunk_id = 0; chunk_id <= chunk_count; ++chunk_id) {
      const int64_t chunk_index = track_id * (chunk_count + 1) + chunk_id;
      const int32_t value = static_cast<int32_t>(chunk_offsets[chunk_index]);
      TORCH_CHECK(
          value >= previous && value >= track_begin && value <= track_end,
          "track_chunk_change_offsets_i16 must be monotonic within each track and bounded by track changes");
      previous = value;
    }
    TORCH_CHECK(previous == track_end, "final chunk change offset must match track change end");
  }
  launch(k.clear_endpoint_loss_site_rgba_grad, [&](MetalKernelFunction& fn) {
    fn.setArg(0, loss);
    fn.setArg(1, grad_site_rgba);
    fn.setArg(2, config_i32);
    fn.dispatch((uint64_t)std::max<int64_t>(site_count, 1), clear_threads);
  });
  auto framegroup_kernel = k.endpoint_record_delta_replace_coeff16_packed_framegroup16_mse_vjp_direct_atomic_rgb_only;
  launch(
      framegroup_kernel,
      [&](MetalKernelFunction& fn) {
        fn.setArg(0, coeff_f16);
        fn.setArg(1, frame_t_f32);
        fn.setArg(2, base_offsets_i32);
        fn.setArg(3, base_record_i32);
        fn.setArg(4, track_change_offsets_i32);
        fn.setArg(5, track_chunk_change_offsets_i16);
        fn.setArg(6, change_frame_i32);
        fn.setArg(7, change_offsets_i32);
        fn.setArg(8, change_record_i32);
        fn.setArg(9, site_rgba_f32);
        fn.setArg(10, target_rgb_f32);
        fn.setArg(11, config_i32);
        fn.setArg(12, config_f32);
        fn.setArg(13, loss);
        fn.setArg(14, grad_site_rgba);
        fn.dispatch((uint64_t)track_count * (uint64_t)chunk_count * framegroup_threads, framegroup_threads);
      });
  return std::make_tuple(loss, grad_site_rgba);
}

std::tuple<torch::Tensor, torch::Tensor>
metal_endpoint_record_delta_replace_coeff16_packed_framegroup16_launch_only_mse_vjp_direct_atomic_rgb_only(
    const torch::Tensor& coeff_f16,
    const torch::Tensor& frame_t_f32,
    const torch::Tensor& base_offsets_i32,
    const torch::Tensor& base_record_i32,
    const torch::Tensor& track_change_offsets_i32,
    const torch::Tensor& track_chunk_change_offsets_i16,
    const torch::Tensor& change_frame_i32,
    const torch::Tensor& change_offsets_i32,
    const torch::Tensor& change_record_i32,
    const torch::Tensor& site_rgba_f32,
    const torch::Tensor& target_rgb_f32,
    const torch::Tensor& config_i32,
    const torch::Tensor& config_f32,
    const int64_t boundary_count,
    const int64_t track_count,
    const int64_t frame_count,
    const int64_t site_count,
    const int64_t base_record_count,
    const int64_t change_count,
    const int64_t change_record_count) {
  check_half_mps_2d(coeff_f16, "coeff_f16", 4);
  check_float_mps_1d_any(frame_t_f32, "frame_t_f32");
  check_i32_mps_1d_any(base_offsets_i32, "base_offsets_i32");
  check_i32_mps_1d_any(base_record_i32, "base_record_i32");
  check_i32_mps_1d_any(track_change_offsets_i32, "track_change_offsets_i32");
  check_i16_mps_1d_any(track_chunk_change_offsets_i16, "track_chunk_change_offsets_i16");
  check_i32_mps_1d_any(change_frame_i32, "change_frame_i32");
  check_i32_mps_1d_any(change_offsets_i32, "change_offsets_i32");
  check_i32_mps_1d_any(change_record_i32, "change_record_i32");
  check_float_mps_2d(site_rgba_f32, "site_rgba_f32", 4);
  TORCH_CHECK(target_rgb_f32.device().is_mps(), "target_rgb_f32 must be on MPS");
  TORCH_CHECK(target_rgb_f32.scalar_type() == torch::kFloat32, "target_rgb_f32 must be float32");
  TORCH_CHECK(target_rgb_f32.dim() == 3 && target_rgb_f32.size(2) == 3, "target_rgb_f32 must have shape [K,T,3]");
  TORCH_CHECK(target_rgb_f32.is_contiguous(), "target_rgb_f32 must be contiguous");
  check_i32_mps_1d(config_i32, "config_i32", 7);
  TORCH_CHECK(config_f32.device().is_mps(), "config_f32 must be on MPS");
  TORCH_CHECK(config_f32.scalar_type() == torch::kFloat32, "config_f32 must be float32");
  TORCH_CHECK(config_f32.dim() == 1 && config_f32.size(0) == 4, "config_f32 must have shape [4]");
  TORCH_CHECK(config_f32.is_contiguous(), "config_f32 must be contiguous");

  TORCH_CHECK(boundary_count > 0, "boundary_count must be positive");
  TORCH_CHECK(boundary_count <= 4093, "packed framegroup16 launch-only delta replace coeff16 fused MSE supports boundary count <= 4093");
  TORCH_CHECK(track_count > 0, "track_count must be positive");
  TORCH_CHECK(frame_count > 0, "frame_count must be positive");
  TORCH_CHECK(site_count == site_rgba_f32.size(0), "site_count must match site_rgba_f32 rows");
  TORCH_CHECK(
      site_count > 0 && site_count <= 256,
      "packed framegroup16 launch-only endpoint record delta replace coeff16 fused MSE VJP supports site count in [1, 256]");
  TORCH_CHECK(base_record_count >= 0, "base_record_count must be nonnegative");
  TORCH_CHECK(change_count >= 0, "change_count must be nonnegative");
  TORCH_CHECK(change_count <= 32767, "int16 chunk-start offsets require change count <= 32767");
  TORCH_CHECK(change_record_count >= 0, "change_record_count must be nonnegative");
  TORCH_CHECK(coeff_f16.size(0) == track_count * boundary_count, "coeff_f16 row count mismatch");
  TORCH_CHECK(frame_t_f32.size(0) == frame_count, "frame_t_f32 length must match frame count");
  TORCH_CHECK(base_record_i32.size(0) == base_record_count, "base_record_i32 length mismatch");
  TORCH_CHECK(change_frame_i32.size(0) == change_count, "change_frame_i32 length must match change count");
  TORCH_CHECK(change_record_i32.size(0) == change_record_count, "change_record_i32 length mismatch");
  TORCH_CHECK(base_offsets_i32.size(0) == track_count + 1, "base_offsets_i32 length must be track_count + 1");
  TORCH_CHECK(
      track_change_offsets_i32.size(0) == track_count + 1,
      "track_change_offsets_i32 length must be track_count + 1");
  TORCH_CHECK(change_offsets_i32.size(0) == change_count + 1, "change_offsets_i32 length must be change_count + 1");
  TORCH_CHECK(
      target_rgb_f32.size(0) == track_count && target_rgb_f32.size(1) == frame_count,
      "target_rgb_f32 shape mismatch");

  auto loss = torch::empty({1}, site_rgba_f32.options().dtype(torch::kFloat32));
  auto grad_site_rgba = torch::empty({site_count, 4}, site_rgba_f32.options().dtype(torch::kFloat32));
  auto& k = kernels();
  constexpr uint64_t clear_threads = 256ull;
  constexpr uint64_t framegroup_threads = 32ull;
  const int64_t chunk_count = (frame_count + int64_t(framegroup_threads) - 1) / int64_t(framegroup_threads);
  TORCH_CHECK(
      track_chunk_change_offsets_i16.size(0) == track_count * (chunk_count + 1),
      "track_chunk_change_offsets_i16 length mismatch");
  launch(k.clear_endpoint_loss_site_rgba_grad, [&](MetalKernelFunction& fn) {
    fn.setArg(0, loss);
    fn.setArg(1, grad_site_rgba);
    fn.setArg(2, config_i32);
    fn.dispatch((uint64_t)std::max<int64_t>(site_count, 1), clear_threads);
  });
  launch(
      k.endpoint_record_delta_replace_coeff16_packed_framegroup16_mse_vjp_direct_atomic_rgb_only,
      [&](MetalKernelFunction& fn) {
        fn.setArg(0, coeff_f16);
        fn.setArg(1, frame_t_f32);
        fn.setArg(2, base_offsets_i32);
        fn.setArg(3, base_record_i32);
        fn.setArg(4, track_change_offsets_i32);
        fn.setArg(5, track_chunk_change_offsets_i16);
        fn.setArg(6, change_frame_i32);
        fn.setArg(7, change_offsets_i32);
        fn.setArg(8, change_record_i32);
        fn.setArg(9, site_rgba_f32);
        fn.setArg(10, target_rgb_f32);
        fn.setArg(11, config_i32);
        fn.setArg(12, config_f32);
        fn.setArg(13, loss);
        fn.setArg(14, grad_site_rgba);
        fn.dispatch((uint64_t)track_count * (uint64_t)chunk_count * framegroup_threads, framegroup_threads);
      });
  return std::make_tuple(loss, grad_site_rgba);
}

std::tuple<torch::Tensor, torch::Tensor>
metal_endpoint_record_delta_replace_coeff16_packed_framegroup16_unchecked_launch_only_mse_vjp_direct_atomic_rgb_only(
    const torch::Tensor& coeff_f16,
    const torch::Tensor& frame_t_f32,
    const torch::Tensor& base_offsets_i32,
    const torch::Tensor& base_record_i32,
    const torch::Tensor& track_change_offsets_i32,
    const torch::Tensor& track_chunk_change_offsets_i16,
    const torch::Tensor& change_frame_i32,
    const torch::Tensor& change_offsets_i32,
    const torch::Tensor& change_record_i32,
    const torch::Tensor& site_rgba_f32,
    const torch::Tensor& target_rgb_f32,
    const torch::Tensor& config_i32,
    const torch::Tensor& config_f32,
    const int64_t track_count,
    const int64_t frame_count,
    const int64_t site_count) {
  auto loss = torch::empty({1}, site_rgba_f32.options().dtype(torch::kFloat32));
  auto grad_site_rgba = torch::empty({site_count, 4}, site_rgba_f32.options().dtype(torch::kFloat32));
  auto& k = kernels();
  constexpr uint64_t clear_threads = 256ull;
  constexpr uint64_t framegroup_threads = 32ull;
  const int64_t chunk_count = (frame_count + int64_t(framegroup_threads) - 1) / int64_t(framegroup_threads);
  launch(k.clear_endpoint_loss_site_rgba_grad, [&](MetalKernelFunction& fn) {
    fn.setArg(0, loss);
    fn.setArg(1, grad_site_rgba);
    fn.setArg(2, config_i32);
    fn.dispatch((uint64_t)std::max<int64_t>(site_count, 1), clear_threads);
  });
  launch(
      k.endpoint_record_delta_replace_coeff16_packed_framegroup16_mse_vjp_direct_atomic_rgb_only,
      [&](MetalKernelFunction& fn) {
        fn.setArg(0, coeff_f16);
        fn.setArg(1, frame_t_f32);
        fn.setArg(2, base_offsets_i32);
        fn.setArg(3, base_record_i32);
        fn.setArg(4, track_change_offsets_i32);
        fn.setArg(5, track_chunk_change_offsets_i16);
        fn.setArg(6, change_frame_i32);
        fn.setArg(7, change_offsets_i32);
        fn.setArg(8, change_record_i32);
        fn.setArg(9, site_rgba_f32);
        fn.setArg(10, target_rgb_f32);
        fn.setArg(11, config_i32);
        fn.setArg(12, config_f32);
        fn.setArg(13, loss);
        fn.setArg(14, grad_site_rgba);
        fn.dispatch((uint64_t)track_count * (uint64_t)chunk_count * framegroup_threads, framegroup_threads);
      });
  return std::make_tuple(loss, grad_site_rgba);
}

std::tuple<torch::Tensor, torch::Tensor>
metal_endpoint_record_delta_replace_coeff16_packed_framegroup16_reduce32_launch_only_mse_vjp_direct_atomic_rgb_only(
    const torch::Tensor& coeff_f16,
    const torch::Tensor& frame_t_f32,
    const torch::Tensor& base_offsets_i32,
    const torch::Tensor& base_record_i32,
    const torch::Tensor& track_change_offsets_i32,
    const torch::Tensor& track_chunk_change_offsets_i16,
    const torch::Tensor& change_frame_i32,
    const torch::Tensor& change_offsets_i32,
    const torch::Tensor& change_record_i32,
    const torch::Tensor& site_rgba_f32,
    const torch::Tensor& target_rgb_f32,
    const torch::Tensor& config_i32,
    const torch::Tensor& config_f32,
    const int64_t boundary_count,
    const int64_t track_count,
    const int64_t frame_count,
    const int64_t site_count,
    const int64_t base_record_count,
    const int64_t change_count,
    const int64_t change_record_count) {
  check_half_mps_2d(coeff_f16, "coeff_f16", 4);
  check_float_mps_1d_any(frame_t_f32, "frame_t_f32");
  check_i32_mps_1d_any(base_offsets_i32, "base_offsets_i32");
  check_i32_mps_1d_any(base_record_i32, "base_record_i32");
  check_i32_mps_1d_any(track_change_offsets_i32, "track_change_offsets_i32");
  check_i16_mps_1d_any(track_chunk_change_offsets_i16, "track_chunk_change_offsets_i16");
  check_i32_mps_1d_any(change_frame_i32, "change_frame_i32");
  check_i32_mps_1d_any(change_offsets_i32, "change_offsets_i32");
  check_i32_mps_1d_any(change_record_i32, "change_record_i32");
  check_float_mps_2d(site_rgba_f32, "site_rgba_f32", 4);
  TORCH_CHECK(target_rgb_f32.device().is_mps(), "target_rgb_f32 must be on MPS");
  TORCH_CHECK(target_rgb_f32.scalar_type() == torch::kFloat32, "target_rgb_f32 must be float32");
  TORCH_CHECK(target_rgb_f32.dim() == 3 && target_rgb_f32.size(2) == 3, "target_rgb_f32 must have shape [K,T,3]");
  TORCH_CHECK(target_rgb_f32.is_contiguous(), "target_rgb_f32 must be contiguous");
  check_i32_mps_1d(config_i32, "config_i32", 7);
  TORCH_CHECK(config_f32.device().is_mps(), "config_f32 must be on MPS");
  TORCH_CHECK(config_f32.scalar_type() == torch::kFloat32, "config_f32 must be float32");
  TORCH_CHECK(config_f32.dim() == 1 && config_f32.size(0) == 4, "config_f32 must have shape [4]");
  TORCH_CHECK(config_f32.is_contiguous(), "config_f32 must be contiguous");

  TORCH_CHECK(boundary_count > 0, "boundary_count must be positive");
  TORCH_CHECK(boundary_count <= 4093, "packed framegroup16 launch-only delta replace coeff16 fused MSE supports boundary count <= 4093");
  TORCH_CHECK(track_count > 0, "track_count must be positive");
  TORCH_CHECK(frame_count > 0, "frame_count must be positive");
  TORCH_CHECK(site_count == site_rgba_f32.size(0), "site_count must match site_rgba_f32 rows");
  TORCH_CHECK(
      site_count > 0 && site_count <= 256,
      "packed framegroup16 launch-only endpoint record delta replace coeff16 fused MSE VJP supports site count in [1, 256]");
  TORCH_CHECK(base_record_count >= 0, "base_record_count must be nonnegative");
  TORCH_CHECK(change_count >= 0, "change_count must be nonnegative");
  TORCH_CHECK(change_count <= 32767, "int16 chunk-start offsets require change count <= 32767");
  TORCH_CHECK(change_record_count >= 0, "change_record_count must be nonnegative");
  TORCH_CHECK(coeff_f16.size(0) == track_count * boundary_count, "coeff_f16 row count mismatch");
  TORCH_CHECK(frame_t_f32.size(0) == frame_count, "frame_t_f32 length must match frame count");
  TORCH_CHECK(base_record_i32.size(0) == base_record_count, "base_record_i32 length mismatch");
  TORCH_CHECK(change_frame_i32.size(0) == change_count, "change_frame_i32 length must match change count");
  TORCH_CHECK(change_record_i32.size(0) == change_record_count, "change_record_i32 length mismatch");
  TORCH_CHECK(base_offsets_i32.size(0) == track_count + 1, "base_offsets_i32 length must be track_count + 1");
  TORCH_CHECK(
      track_change_offsets_i32.size(0) == track_count + 1,
      "track_change_offsets_i32 length must be track_count + 1");
  TORCH_CHECK(change_offsets_i32.size(0) == change_count + 1, "change_offsets_i32 length must be change_count + 1");
  TORCH_CHECK(
      target_rgb_f32.size(0) == track_count && target_rgb_f32.size(1) == frame_count,
      "target_rgb_f32 shape mismatch");

  auto loss = torch::empty({1}, site_rgba_f32.options().dtype(torch::kFloat32));
  auto grad_site_rgba = torch::empty({site_count, 4}, site_rgba_f32.options().dtype(torch::kFloat32));
  auto& k = kernels();
  constexpr uint64_t clear_threads = 256ull;
  constexpr uint64_t framegroup_threads = 32ull;
  const int64_t chunk_count = (frame_count + int64_t(framegroup_threads) - 1) / int64_t(framegroup_threads);
  TORCH_CHECK(
      track_chunk_change_offsets_i16.size(0) == track_count * (chunk_count + 1),
      "track_chunk_change_offsets_i16 length mismatch");
  launch(k.clear_endpoint_loss_site_rgba_grad, [&](MetalKernelFunction& fn) {
    fn.setArg(0, loss);
    fn.setArg(1, grad_site_rgba);
    fn.setArg(2, config_i32);
    fn.dispatch((uint64_t)std::max<int64_t>(site_count, 1), clear_threads);
  });
  launch(
      k.endpoint_record_delta_replace_coeff16_packed_framegroup16_reduce32_mse_vjp_direct_atomic_rgb_only,
      [&](MetalKernelFunction& fn) {
        fn.setArg(0, coeff_f16);
        fn.setArg(1, frame_t_f32);
        fn.setArg(2, base_offsets_i32);
        fn.setArg(3, base_record_i32);
        fn.setArg(4, track_change_offsets_i32);
        fn.setArg(5, track_chunk_change_offsets_i16);
        fn.setArg(6, change_frame_i32);
        fn.setArg(7, change_offsets_i32);
        fn.setArg(8, change_record_i32);
        fn.setArg(9, site_rgba_f32);
        fn.setArg(10, target_rgb_f32);
        fn.setArg(11, config_i32);
        fn.setArg(12, config_f32);
        fn.setArg(13, loss);
        fn.setArg(14, grad_site_rgba);
        fn.dispatch((uint64_t)track_count * (uint64_t)chunk_count * framegroup_threads, framegroup_threads);
      });
  return std::make_tuple(loss, grad_site_rgba);
}

std::tuple<torch::Tensor, torch::Tensor>
metal_endpoint_record_delta_replace_coeff16_packed_framegroup16_reduce32_unchecked_launch_only_mse_vjp_direct_atomic_rgb_only(
    const torch::Tensor& coeff_f16,
    const torch::Tensor& frame_t_f32,
    const torch::Tensor& base_offsets_i32,
    const torch::Tensor& base_record_i32,
    const torch::Tensor& track_change_offsets_i32,
    const torch::Tensor& track_chunk_change_offsets_i16,
    const torch::Tensor& change_frame_i32,
    const torch::Tensor& change_offsets_i32,
    const torch::Tensor& change_record_i32,
    const torch::Tensor& site_rgba_f32,
    const torch::Tensor& target_rgb_f32,
    const torch::Tensor& config_i32,
    const torch::Tensor& config_f32,
    const int64_t track_count,
    const int64_t frame_count,
    const int64_t site_count) {
  auto loss = torch::empty({1}, site_rgba_f32.options().dtype(torch::kFloat32));
  auto grad_site_rgba = torch::empty({site_count, 4}, site_rgba_f32.options().dtype(torch::kFloat32));
  auto& k = kernels();
  constexpr uint64_t clear_threads = 256ull;
  constexpr uint64_t framegroup_threads = 32ull;
  const int64_t chunk_count = (frame_count + int64_t(framegroup_threads) - 1) / int64_t(framegroup_threads);
  launch(k.clear_endpoint_loss_site_rgba_grad, [&](MetalKernelFunction& fn) {
    fn.setArg(0, loss);
    fn.setArg(1, grad_site_rgba);
    fn.setArg(2, config_i32);
    fn.dispatch((uint64_t)std::max<int64_t>(site_count, 1), clear_threads);
  });
  launch(
      k.endpoint_record_delta_replace_coeff16_packed_framegroup16_reduce32_mse_vjp_direct_atomic_rgb_only,
      [&](MetalKernelFunction& fn) {
        fn.setArg(0, coeff_f16);
        fn.setArg(1, frame_t_f32);
        fn.setArg(2, base_offsets_i32);
        fn.setArg(3, base_record_i32);
        fn.setArg(4, track_change_offsets_i32);
        fn.setArg(5, track_chunk_change_offsets_i16);
        fn.setArg(6, change_frame_i32);
        fn.setArg(7, change_offsets_i32);
        fn.setArg(8, change_record_i32);
        fn.setArg(9, site_rgba_f32);
        fn.setArg(10, target_rgb_f32);
        fn.setArg(11, config_i32);
        fn.setArg(12, config_f32);
        fn.setArg(13, loss);
        fn.setArg(14, grad_site_rgba);
        fn.dispatch((uint64_t)track_count * (uint64_t)chunk_count * framegroup_threads, framegroup_threads);
      });
  return std::make_tuple(loss, grad_site_rgba);
}

std::tuple<torch::Tensor, torch::Tensor>
metal_endpoint_record_delta_replace_coeff16_packed_framegroup16_rowselect32_launch_only_mse_vjp_direct_atomic_rgb_only(
    const torch::Tensor& coeff_f16,
    const torch::Tensor& frame_t_f32,
    const torch::Tensor& base_offsets_i32,
    const torch::Tensor& base_record_i32,
    const torch::Tensor& track_change_offsets_i32,
    const torch::Tensor& track_chunk_change_offsets_i16,
    const torch::Tensor& change_frame_i32,
    const torch::Tensor& change_offsets_i32,
    const torch::Tensor& change_record_i32,
    const torch::Tensor& site_rgba_f32,
    const torch::Tensor& target_rgb_f32,
    const torch::Tensor& config_i32,
    const torch::Tensor& config_f32,
    const int64_t boundary_count,
    const int64_t track_count,
    const int64_t frame_count,
    const int64_t site_count,
    const int64_t base_record_count,
    const int64_t change_count,
    const int64_t change_record_count) {
  check_half_mps_2d(coeff_f16, "coeff_f16", 4);
  check_float_mps_1d_any(frame_t_f32, "frame_t_f32");
  check_i32_mps_1d_any(base_offsets_i32, "base_offsets_i32");
  check_i32_mps_1d_any(base_record_i32, "base_record_i32");
  check_i32_mps_1d_any(track_change_offsets_i32, "track_change_offsets_i32");
  check_i16_mps_1d_any(track_chunk_change_offsets_i16, "track_chunk_change_offsets_i16");
  check_i32_mps_1d_any(change_frame_i32, "change_frame_i32");
  check_i32_mps_1d_any(change_offsets_i32, "change_offsets_i32");
  check_i32_mps_1d_any(change_record_i32, "change_record_i32");
  check_float_mps_2d(site_rgba_f32, "site_rgba_f32", 4);
  TORCH_CHECK(target_rgb_f32.device().is_mps(), "target_rgb_f32 must be on MPS");
  TORCH_CHECK(target_rgb_f32.scalar_type() == torch::kFloat32, "target_rgb_f32 must be float32");
  TORCH_CHECK(target_rgb_f32.dim() == 3 && target_rgb_f32.size(2) == 3, "target_rgb_f32 must have shape [K,T,3]");
  TORCH_CHECK(target_rgb_f32.is_contiguous(), "target_rgb_f32 must be contiguous");
  check_i32_mps_1d(config_i32, "config_i32", 7);
  TORCH_CHECK(config_f32.device().is_mps(), "config_f32 must be on MPS");
  TORCH_CHECK(config_f32.scalar_type() == torch::kFloat32, "config_f32 must be float32");
  TORCH_CHECK(config_f32.dim() == 1 && config_f32.size(0) == 4, "config_f32 must have shape [4]");
  TORCH_CHECK(config_f32.is_contiguous(), "config_f32 must be contiguous");

  TORCH_CHECK(boundary_count > 0, "boundary_count must be positive");
  TORCH_CHECK(boundary_count <= 4093, "packed framegroup16 launch-only delta replace coeff16 fused MSE supports boundary count <= 4093");
  TORCH_CHECK(track_count > 0, "track_count must be positive");
  TORCH_CHECK(frame_count > 0, "frame_count must be positive");
  TORCH_CHECK(site_count == site_rgba_f32.size(0), "site_count must match site_rgba_f32 rows");
  TORCH_CHECK(
      site_count > 0 && site_count <= 256,
      "packed framegroup16 launch-only endpoint record delta replace coeff16 fused MSE VJP supports site count in [1, 256]");
  TORCH_CHECK(base_record_count >= 0, "base_record_count must be nonnegative");
  TORCH_CHECK(change_count >= 0, "change_count must be nonnegative");
  TORCH_CHECK(change_count <= 32767, "int16 chunk-start offsets require change count <= 32767");
  TORCH_CHECK(change_record_count >= 0, "change_record_count must be nonnegative");
  TORCH_CHECK(coeff_f16.size(0) == track_count * boundary_count, "coeff_f16 row count mismatch");
  TORCH_CHECK(frame_t_f32.size(0) == frame_count, "frame_t_f32 length must match frame count");
  TORCH_CHECK(base_record_i32.size(0) == base_record_count, "base_record_i32 length mismatch");
  TORCH_CHECK(change_frame_i32.size(0) == change_count, "change_frame_i32 length must match change count");
  TORCH_CHECK(change_record_i32.size(0) == change_record_count, "change_record_i32 length mismatch");
  TORCH_CHECK(base_offsets_i32.size(0) == track_count + 1, "base_offsets_i32 length must be track_count + 1");
  TORCH_CHECK(
      track_change_offsets_i32.size(0) == track_count + 1,
      "track_change_offsets_i32 length must be track_count + 1");
  TORCH_CHECK(change_offsets_i32.size(0) == change_count + 1, "change_offsets_i32 length must be change_count + 1");
  TORCH_CHECK(
      target_rgb_f32.size(0) == track_count && target_rgb_f32.size(1) == frame_count,
      "target_rgb_f32 shape mismatch");

  auto loss = torch::empty({1}, site_rgba_f32.options().dtype(torch::kFloat32));
  auto grad_site_rgba = torch::empty({site_count, 4}, site_rgba_f32.options().dtype(torch::kFloat32));
  auto& k = kernels();
  constexpr uint64_t clear_threads = 256ull;
  constexpr uint64_t framegroup_threads = 32ull;
  const int64_t chunk_count = (frame_count + int64_t(framegroup_threads) - 1) / int64_t(framegroup_threads);
  TORCH_CHECK(
      track_chunk_change_offsets_i16.size(0) == track_count * (chunk_count + 1),
      "track_chunk_change_offsets_i16 length mismatch");
  launch(k.clear_endpoint_loss_site_rgba_grad, [&](MetalKernelFunction& fn) {
    fn.setArg(0, loss);
    fn.setArg(1, grad_site_rgba);
    fn.setArg(2, config_i32);
    fn.dispatch((uint64_t)std::max<int64_t>(site_count, 1), clear_threads);
  });
  launch(
      k.endpoint_record_delta_replace_coeff16_packed_framegroup16_rowselect32_mse_vjp_direct_atomic_rgb_only,
      [&](MetalKernelFunction& fn) {
        fn.setArg(0, coeff_f16);
        fn.setArg(1, frame_t_f32);
        fn.setArg(2, base_offsets_i32);
        fn.setArg(3, base_record_i32);
        fn.setArg(4, track_change_offsets_i32);
        fn.setArg(5, track_chunk_change_offsets_i16);
        fn.setArg(6, change_frame_i32);
        fn.setArg(7, change_offsets_i32);
        fn.setArg(8, change_record_i32);
        fn.setArg(9, site_rgba_f32);
        fn.setArg(10, target_rgb_f32);
        fn.setArg(11, config_i32);
        fn.setArg(12, config_f32);
        fn.setArg(13, loss);
        fn.setArg(14, grad_site_rgba);
        fn.dispatch((uint64_t)track_count * (uint64_t)chunk_count * framegroup_threads, framegroup_threads);
      });
  return std::make_tuple(loss, grad_site_rgba);
}

std::tuple<torch::Tensor, torch::Tensor>
metal_endpoint_record_delta_replace_coeff16_packed_framegroup16_rowselect32_unchecked_launch_only_mse_vjp_direct_atomic_rgb_only(
    const torch::Tensor& coeff_f16,
    const torch::Tensor& frame_t_f32,
    const torch::Tensor& base_offsets_i32,
    const torch::Tensor& base_record_i32,
    const torch::Tensor& track_change_offsets_i32,
    const torch::Tensor& track_chunk_change_offsets_i16,
    const torch::Tensor& change_frame_i32,
    const torch::Tensor& change_offsets_i32,
    const torch::Tensor& change_record_i32,
    const torch::Tensor& site_rgba_f32,
    const torch::Tensor& target_rgb_f32,
    const torch::Tensor& config_i32,
    const torch::Tensor& config_f32,
    const int64_t track_count,
    const int64_t frame_count,
    const int64_t site_count) {
  auto loss = torch::empty({1}, site_rgba_f32.options().dtype(torch::kFloat32));
  auto grad_site_rgba = torch::empty({site_count, 4}, site_rgba_f32.options().dtype(torch::kFloat32));
  auto& k = kernels();
  constexpr uint64_t clear_threads = 256ull;
  constexpr uint64_t framegroup_threads = 32ull;
  const int64_t chunk_count = (frame_count + int64_t(framegroup_threads) - 1) / int64_t(framegroup_threads);
  launch(k.clear_endpoint_loss_site_rgba_grad, [&](MetalKernelFunction& fn) {
    fn.setArg(0, loss);
    fn.setArg(1, grad_site_rgba);
    fn.setArg(2, config_i32);
    fn.dispatch((uint64_t)std::max<int64_t>(site_count, 1), clear_threads);
  });
  launch(
      k.endpoint_record_delta_replace_coeff16_packed_framegroup16_rowselect32_mse_vjp_direct_atomic_rgb_only,
      [&](MetalKernelFunction& fn) {
        fn.setArg(0, coeff_f16);
        fn.setArg(1, frame_t_f32);
        fn.setArg(2, base_offsets_i32);
        fn.setArg(3, base_record_i32);
        fn.setArg(4, track_change_offsets_i32);
        fn.setArg(5, track_chunk_change_offsets_i16);
        fn.setArg(6, change_frame_i32);
        fn.setArg(7, change_offsets_i32);
        fn.setArg(8, change_record_i32);
        fn.setArg(9, site_rgba_f32);
        fn.setArg(10, target_rgb_f32);
        fn.setArg(11, config_i32);
        fn.setArg(12, config_f32);
        fn.setArg(13, loss);
        fn.setArg(14, grad_site_rgba);
        fn.dispatch((uint64_t)track_count * (uint64_t)chunk_count * framegroup_threads, framegroup_threads);
      });
  return std::make_tuple(loss, grad_site_rgba);
}

std::tuple<torch::Tensor, torch::Tensor>
metal_endpoint_record_delta_replace_coeff16_packed_framegroup16_rowdesc_launch_only_mse_vjp_direct_atomic_rgb_only(
    const torch::Tensor& coeff_f16,
    const torch::Tensor& frame_t_f32,
    const torch::Tensor& row_begin_i32,
    const torch::Tensor& row_len_source_i16,
    const torch::Tensor& base_record_i32,
    const torch::Tensor& change_record_i32,
    const torch::Tensor& site_rgba_f32,
    const torch::Tensor& target_rgb_f32,
    const torch::Tensor& config_i32,
    const torch::Tensor& config_f32,
    const int64_t boundary_count,
    const int64_t track_count,
    const int64_t frame_count,
    const int64_t site_count,
    const int64_t base_record_count,
    const int64_t change_record_count) {
  check_half_mps_2d(coeff_f16, "coeff_f16", 4);
  check_float_mps_1d_any(frame_t_f32, "frame_t_f32");
  check_i32_mps_1d_any(row_begin_i32, "row_begin_i32");
  check_i16_mps_1d_any(row_len_source_i16, "row_len_source_i16");
  check_i32_mps_1d_any(base_record_i32, "base_record_i32");
  check_i32_mps_1d_any(change_record_i32, "change_record_i32");
  check_float_mps_2d(site_rgba_f32, "site_rgba_f32", 4);
  TORCH_CHECK(target_rgb_f32.device().is_mps(), "target_rgb_f32 must be on MPS");
  TORCH_CHECK(target_rgb_f32.scalar_type() == torch::kFloat32, "target_rgb_f32 must be float32");
  TORCH_CHECK(target_rgb_f32.dim() == 3 && target_rgb_f32.size(2) == 3, "target_rgb_f32 must have shape [K,T,3]");
  TORCH_CHECK(target_rgb_f32.is_contiguous(), "target_rgb_f32 must be contiguous");
  check_i32_mps_1d(config_i32, "config_i32", 7);
  TORCH_CHECK(config_f32.device().is_mps(), "config_f32 must be on MPS");
  TORCH_CHECK(config_f32.scalar_type() == torch::kFloat32, "config_f32 must be float32");
  TORCH_CHECK(config_f32.dim() == 1 && config_f32.size(0) == 4, "config_f32 must have shape [4]");
  TORCH_CHECK(config_f32.is_contiguous(), "config_f32 must be contiguous");

  TORCH_CHECK(boundary_count > 0, "rowdesc packed framegroup16 boundary_count must be positive");
  TORCH_CHECK(boundary_count <= 4093, "rowdesc packed framegroup16 supports boundary count <= 4093");
  TORCH_CHECK(track_count > 0, "rowdesc packed framegroup16 track_count must be positive");
  TORCH_CHECK(frame_count > 0, "rowdesc packed framegroup16 frame_count must be positive");
  TORCH_CHECK(site_count == site_rgba_f32.size(0), "rowdesc packed framegroup16 site_count must match site_rgba_f32");
  TORCH_CHECK(site_count > 0 && site_count <= 256, "rowdesc packed framegroup16 supports site count in [1, 256]");
  TORCH_CHECK(base_record_count >= 0, "rowdesc packed framegroup16 base_record_count must be nonnegative");
  TORCH_CHECK(change_record_count >= 0, "rowdesc packed framegroup16 change_record_count must be nonnegative");
  TORCH_CHECK(coeff_f16.size(0) == track_count * boundary_count, "coeff_f16 row count mismatch");
  TORCH_CHECK(frame_t_f32.size(0) == frame_count, "frame_t_f32 length must match frame count");
  TORCH_CHECK(base_record_i32.size(0) == base_record_count, "base_record_i32 length mismatch");
  TORCH_CHECK(change_record_i32.size(0) == change_record_count, "change_record_i32 length mismatch");
  TORCH_CHECK(
      target_rgb_f32.size(0) == track_count && target_rgb_f32.size(1) == frame_count,
      "target_rgb_f32 shape mismatch");

  auto loss = torch::empty({1}, site_rgba_f32.options().dtype(torch::kFloat32));
  auto grad_site_rgba = torch::empty({site_count, 4}, site_rgba_f32.options().dtype(torch::kFloat32));
  auto& k = kernels();
  constexpr uint64_t clear_threads = 256ull;
  constexpr uint64_t framegroup_threads = 32ull;
  const int64_t chunk_count = (frame_count + int64_t(framegroup_threads) - 1) / int64_t(framegroup_threads);
  TORCH_CHECK(row_begin_i32.size(0) == track_count * frame_count, "row_begin_i32 length mismatch");
  TORCH_CHECK(
      row_len_source_i16.size(0) == row_begin_i32.size(0),
      "row_len_source_i16 length must match row_begin_i32");
  launch(k.clear_endpoint_loss_site_rgba_grad, [&](MetalKernelFunction& fn) {
    fn.setArg(0, loss);
    fn.setArg(1, grad_site_rgba);
    fn.setArg(2, config_i32);
    fn.dispatch((uint64_t)std::max<int64_t>(site_count, 1), clear_threads);
  });
  launch(
      k.endpoint_record_delta_replace_coeff16_packed_framegroup16_rowdesc_mse_vjp_direct_atomic_rgb_only,
      [&](MetalKernelFunction& fn) {
        fn.setArg(0, coeff_f16);
        fn.setArg(1, frame_t_f32);
        fn.setArg(2, row_begin_i32);
        fn.setArg(3, row_len_source_i16);
        fn.setArg(4, base_record_i32);
        fn.setArg(5, change_record_i32);
        fn.setArg(6, site_rgba_f32);
        fn.setArg(7, target_rgb_f32);
        fn.setArg(8, config_i32);
        fn.setArg(9, config_f32);
        fn.setArg(10, loss);
        fn.setArg(11, grad_site_rgba);
        fn.dispatch((uint64_t)track_count * (uint64_t)chunk_count * framegroup_threads, framegroup_threads);
      });
  return std::make_tuple(loss, grad_site_rgba);
}

std::tuple<torch::Tensor, torch::Tensor>
metal_endpoint_record_delta_replace_coeff16_packed_framegroup16_rowdesc_unchecked_launch_only_mse_vjp_direct_atomic_rgb_only(
    const torch::Tensor& coeff_f16,
    const torch::Tensor& frame_t_f32,
    const torch::Tensor& row_begin_i32,
    const torch::Tensor& row_len_source_i16,
    const torch::Tensor& base_record_i32,
    const torch::Tensor& change_record_i32,
    const torch::Tensor& site_rgba_f32,
    const torch::Tensor& target_rgb_f32,
    const torch::Tensor& config_i32,
    const torch::Tensor& config_f32,
    const int64_t track_count,
    const int64_t frame_count,
    const int64_t site_count) {
  auto loss = torch::empty({1}, site_rgba_f32.options().dtype(torch::kFloat32));
  auto grad_site_rgba = torch::empty({site_count, 4}, site_rgba_f32.options().dtype(torch::kFloat32));
  auto& k = kernels();
  constexpr uint64_t clear_threads = 256ull;
  constexpr uint64_t framegroup_threads = 32ull;
  const int64_t chunk_count = (frame_count + int64_t(framegroup_threads) - 1) / int64_t(framegroup_threads);
  launch(k.clear_endpoint_loss_site_rgba_grad, [&](MetalKernelFunction& fn) {
    fn.setArg(0, loss);
    fn.setArg(1, grad_site_rgba);
    fn.setArg(2, config_i32);
    fn.dispatch((uint64_t)std::max<int64_t>(site_count, 1), clear_threads);
  });
  launch(
      k.endpoint_record_delta_replace_coeff16_packed_framegroup16_rowdesc_mse_vjp_direct_atomic_rgb_only,
      [&](MetalKernelFunction& fn) {
        fn.setArg(0, coeff_f16);
        fn.setArg(1, frame_t_f32);
        fn.setArg(2, row_begin_i32);
        fn.setArg(3, row_len_source_i16);
        fn.setArg(4, base_record_i32);
        fn.setArg(5, change_record_i32);
        fn.setArg(6, site_rgba_f32);
        fn.setArg(7, target_rgb_f32);
        fn.setArg(8, config_i32);
        fn.setArg(9, config_f32);
        fn.setArg(10, loss);
        fn.setArg(11, grad_site_rgba);
        fn.dispatch((uint64_t)track_count * (uint64_t)chunk_count * framegroup_threads, framegroup_threads);
      });
  return std::make_tuple(loss, grad_site_rgba);
}

std::tuple<torch::Tensor, torch::Tensor>
metal_endpoint_record_delta_replace_coeff16_packed_framegroup16_rowdesc32_launch_only_mse_vjp_direct_atomic_rgb_only(
    const torch::Tensor& coeff_f16,
    const torch::Tensor& frame_t_f32,
    const torch::Tensor& row_begin_i32,
    const torch::Tensor& row_len_source_i16,
    const torch::Tensor& base_record_i32,
    const torch::Tensor& change_record_i32,
    const torch::Tensor& site_rgba_f32,
    const torch::Tensor& target_rgb_f32,
    const torch::Tensor& config_i32,
    const torch::Tensor& config_f32,
    const int64_t boundary_count,
    const int64_t track_count,
    const int64_t frame_count,
    const int64_t site_count,
    const int64_t base_record_count,
    const int64_t change_record_count) {
  check_half_mps_2d(coeff_f16, "coeff_f16", 4);
  check_float_mps_1d_any(frame_t_f32, "frame_t_f32");
  check_i32_mps_1d_any(row_begin_i32, "row_begin_i32");
  check_i16_mps_1d_any(row_len_source_i16, "row_len_source_i16");
  check_i32_mps_1d_any(base_record_i32, "base_record_i32");
  check_i32_mps_1d_any(change_record_i32, "change_record_i32");
  check_float_mps_2d(site_rgba_f32, "site_rgba_f32", 4);
  TORCH_CHECK(target_rgb_f32.device().is_mps(), "target_rgb_f32 must be on MPS");
  TORCH_CHECK(target_rgb_f32.scalar_type() == torch::kFloat32, "target_rgb_f32 must be float32");
  TORCH_CHECK(target_rgb_f32.dim() == 3 && target_rgb_f32.size(2) == 3, "target_rgb_f32 must have shape [K,T,3]");
  TORCH_CHECK(target_rgb_f32.is_contiguous(), "target_rgb_f32 must be contiguous");
  check_i32_mps_1d(config_i32, "config_i32", 7);
  TORCH_CHECK(config_f32.device().is_mps(), "config_f32 must be on MPS");
  TORCH_CHECK(config_f32.scalar_type() == torch::kFloat32, "config_f32 must be float32");
  TORCH_CHECK(config_f32.dim() == 1 && config_f32.size(0) == 4, "config_f32 must have shape [4]");
  TORCH_CHECK(config_f32.is_contiguous(), "config_f32 must be contiguous");

  TORCH_CHECK(boundary_count > 0, "rowdesc32 packed framegroup16 boundary_count must be positive");
  TORCH_CHECK(boundary_count <= 4093, "rowdesc32 packed framegroup16 supports boundary count <= 4093");
  TORCH_CHECK(track_count > 0, "rowdesc32 packed framegroup16 track_count must be positive");
  TORCH_CHECK(frame_count > 0, "rowdesc32 packed framegroup16 frame_count must be positive");
  TORCH_CHECK(site_count == site_rgba_f32.size(0), "rowdesc32 packed framegroup16 site_count must match site_rgba_f32");
  TORCH_CHECK(site_count > 0 && site_count <= 256, "rowdesc32 packed framegroup16 supports site count in [1, 256]");
  TORCH_CHECK(base_record_count >= 0, "rowdesc32 packed framegroup16 base_record_count must be nonnegative");
  TORCH_CHECK(change_record_count >= 0, "rowdesc32 packed framegroup16 change_record_count must be nonnegative");
  TORCH_CHECK(coeff_f16.size(0) == track_count * boundary_count, "coeff_f16 row count mismatch");
  TORCH_CHECK(frame_t_f32.size(0) == frame_count, "frame_t_f32 length must match frame count");
  TORCH_CHECK(base_record_i32.size(0) == base_record_count, "base_record_i32 length mismatch");
  TORCH_CHECK(change_record_i32.size(0) == change_record_count, "change_record_i32 length mismatch");
  TORCH_CHECK(
      target_rgb_f32.size(0) == track_count && target_rgb_f32.size(1) == frame_count,
      "target_rgb_f32 shape mismatch");

  auto loss = torch::empty({1}, site_rgba_f32.options().dtype(torch::kFloat32));
  auto grad_site_rgba = torch::empty({site_count, 4}, site_rgba_f32.options().dtype(torch::kFloat32));
  auto& k = kernels();
  constexpr uint64_t clear_threads = 256ull;
  constexpr uint64_t framegroup_threads = 32ull;
  const int64_t chunk_count = (frame_count + int64_t(framegroup_threads) - 1) / int64_t(framegroup_threads);
  TORCH_CHECK(row_begin_i32.size(0) == track_count * frame_count, "row_begin_i32 length mismatch");
  TORCH_CHECK(
      row_len_source_i16.size(0) == row_begin_i32.size(0),
      "row_len_source_i16 length must match row_begin_i32");
  launch(k.clear_endpoint_loss_site_rgba_grad, [&](MetalKernelFunction& fn) {
    fn.setArg(0, loss);
    fn.setArg(1, grad_site_rgba);
    fn.setArg(2, config_i32);
    fn.dispatch((uint64_t)std::max<int64_t>(site_count, 1), clear_threads);
  });
  launch(
      k.endpoint_record_delta_replace_coeff16_packed_framegroup16_rowdesc32_mse_vjp_direct_atomic_rgb_only,
      [&](MetalKernelFunction& fn) {
        fn.setArg(0, coeff_f16);
        fn.setArg(1, frame_t_f32);
        fn.setArg(2, row_begin_i32);
        fn.setArg(3, row_len_source_i16);
        fn.setArg(4, base_record_i32);
        fn.setArg(5, change_record_i32);
        fn.setArg(6, site_rgba_f32);
        fn.setArg(7, target_rgb_f32);
        fn.setArg(8, config_i32);
        fn.setArg(9, config_f32);
        fn.setArg(10, loss);
        fn.setArg(11, grad_site_rgba);
        fn.dispatch((uint64_t)track_count * (uint64_t)chunk_count * framegroup_threads, framegroup_threads);
      });
  return std::make_tuple(loss, grad_site_rgba);
}

std::tuple<torch::Tensor, torch::Tensor>
metal_endpoint_record_delta_replace_coeff16_packed_framegroup16_rowdesc32_unchecked_launch_only_mse_vjp_direct_atomic_rgb_only(
    const torch::Tensor& coeff_f16,
    const torch::Tensor& frame_t_f32,
    const torch::Tensor& row_begin_i32,
    const torch::Tensor& row_len_source_i16,
    const torch::Tensor& base_record_i32,
    const torch::Tensor& change_record_i32,
    const torch::Tensor& site_rgba_f32,
    const torch::Tensor& target_rgb_f32,
    const torch::Tensor& config_i32,
    const torch::Tensor& config_f32,
    const int64_t track_count,
    const int64_t frame_count,
    const int64_t site_count) {
  auto loss = torch::empty({1}, site_rgba_f32.options().dtype(torch::kFloat32));
  auto grad_site_rgba = torch::empty({site_count, 4}, site_rgba_f32.options().dtype(torch::kFloat32));
  auto& k = kernels();
  constexpr uint64_t clear_threads = 256ull;
  constexpr uint64_t framegroup_threads = 32ull;
  const int64_t chunk_count = (frame_count + int64_t(framegroup_threads) - 1) / int64_t(framegroup_threads);
  launch(k.clear_endpoint_loss_site_rgba_grad, [&](MetalKernelFunction& fn) {
    fn.setArg(0, loss);
    fn.setArg(1, grad_site_rgba);
    fn.setArg(2, config_i32);
    fn.dispatch((uint64_t)std::max<int64_t>(site_count, 1), clear_threads);
  });
  launch(
      k.endpoint_record_delta_replace_coeff16_packed_framegroup16_rowdesc32_mse_vjp_direct_atomic_rgb_only,
      [&](MetalKernelFunction& fn) {
        fn.setArg(0, coeff_f16);
        fn.setArg(1, frame_t_f32);
        fn.setArg(2, row_begin_i32);
        fn.setArg(3, row_len_source_i16);
        fn.setArg(4, base_record_i32);
        fn.setArg(5, change_record_i32);
        fn.setArg(6, site_rgba_f32);
        fn.setArg(7, target_rgb_f32);
        fn.setArg(8, config_i32);
        fn.setArg(9, config_f32);
        fn.setArg(10, loss);
        fn.setArg(11, grad_site_rgba);
        fn.dispatch((uint64_t)track_count * (uint64_t)chunk_count * framegroup_threads, framegroup_threads);
      });
  return std::make_tuple(loss, grad_site_rgba);
}

std::tuple<torch::Tensor, torch::Tensor>
metal_endpoint_record_delta_replace_coeff16_packed_framegroup16_launch_only_variant_mse_vjp_direct_atomic_rgb_only(
    const torch::Tensor& coeff_f16,
    const torch::Tensor& frame_t_f32,
    const torch::Tensor& base_offsets_i32,
    const torch::Tensor& base_record_i32,
    const torch::Tensor& track_change_offsets_i32,
    const torch::Tensor& track_chunk_change_offsets_i16,
    const torch::Tensor& change_frame_i32,
    const torch::Tensor& change_offsets_i32,
    const torch::Tensor& change_record_i32,
    const torch::Tensor& site_rgba_f32,
    const torch::Tensor& target_rgb_f32,
    const torch::Tensor& config_i32,
    const torch::Tensor& config_f32,
    const int64_t boundary_count,
    const int64_t track_count,
    const int64_t frame_count,
    const int64_t site_count,
    const int64_t base_record_count,
    const int64_t change_count,
    const int64_t change_record_count,
    std::shared_ptr<MetalKernelFunction> framegroup_kernel,
    const uint64_t framegroup_threads,
    const char* label) {
  check_half_mps_2d(coeff_f16, "coeff_f16", 4);
  check_float_mps_1d_any(frame_t_f32, "frame_t_f32");
  check_i32_mps_1d_any(base_offsets_i32, "base_offsets_i32");
  check_i32_mps_1d_any(base_record_i32, "base_record_i32");
  check_i32_mps_1d_any(track_change_offsets_i32, "track_change_offsets_i32");
  check_i16_mps_1d_any(track_chunk_change_offsets_i16, "track_chunk_change_offsets_i16");
  check_i32_mps_1d_any(change_frame_i32, "change_frame_i32");
  check_i32_mps_1d_any(change_offsets_i32, "change_offsets_i32");
  check_i32_mps_1d_any(change_record_i32, "change_record_i32");
  check_float_mps_2d(site_rgba_f32, "site_rgba_f32", 4);
  TORCH_CHECK(target_rgb_f32.device().is_mps(), "target_rgb_f32 must be on MPS");
  TORCH_CHECK(target_rgb_f32.scalar_type() == torch::kFloat32, "target_rgb_f32 must be float32");
  TORCH_CHECK(target_rgb_f32.dim() == 3 && target_rgb_f32.size(2) == 3, "target_rgb_f32 must have shape [K,T,3]");
  TORCH_CHECK(target_rgb_f32.is_contiguous(), "target_rgb_f32 must be contiguous");
  check_i32_mps_1d(config_i32, "config_i32", 7);
  TORCH_CHECK(config_f32.device().is_mps(), "config_f32 must be on MPS");
  TORCH_CHECK(config_f32.scalar_type() == torch::kFloat32, "config_f32 must be float32");
  TORCH_CHECK(config_f32.dim() == 1 && config_f32.size(0) == 4, "config_f32 must have shape [4]");
  TORCH_CHECK(config_f32.is_contiguous(), "config_f32 must be contiguous");

  TORCH_CHECK(boundary_count > 0, label, " boundary_count must be positive");
  TORCH_CHECK(boundary_count <= 4093, label, " supports boundary count <= 4093");
  TORCH_CHECK(track_count > 0, label, " track_count must be positive");
  TORCH_CHECK(frame_count > 0, label, " frame_count must be positive");
  TORCH_CHECK(site_count == site_rgba_f32.size(0), label, " site_count must match site_rgba_f32 rows");
  TORCH_CHECK(site_count > 0 && site_count <= 256, label, " supports site count in [1, 256]");
  TORCH_CHECK(base_record_count >= 0, label, " base_record_count must be nonnegative");
  TORCH_CHECK(change_count >= 0, label, " change_count must be nonnegative");
  TORCH_CHECK(change_count <= 32767, "int16 chunk-start offsets require change count <= 32767");
  TORCH_CHECK(change_record_count >= 0, label, " change_record_count must be nonnegative");
  TORCH_CHECK(coeff_f16.size(0) == track_count * boundary_count, "coeff_f16 row count mismatch");
  TORCH_CHECK(frame_t_f32.size(0) == frame_count, "frame_t_f32 length must match frame count");
  TORCH_CHECK(base_record_i32.size(0) == base_record_count, "base_record_i32 length mismatch");
  TORCH_CHECK(change_frame_i32.size(0) == change_count, "change_frame_i32 length must match change count");
  TORCH_CHECK(change_record_i32.size(0) == change_record_count, "change_record_i32 length mismatch");
  TORCH_CHECK(base_offsets_i32.size(0) == track_count + 1, "base_offsets_i32 length must be track_count + 1");
  TORCH_CHECK(
      track_change_offsets_i32.size(0) == track_count + 1,
      "track_change_offsets_i32 length must be track_count + 1");
  TORCH_CHECK(change_offsets_i32.size(0) == change_count + 1, "change_offsets_i32 length must be change_count + 1");
  TORCH_CHECK(
      target_rgb_f32.size(0) == track_count && target_rgb_f32.size(1) == frame_count,
      "target_rgb_f32 shape mismatch");

  auto loss = torch::empty({1}, site_rgba_f32.options().dtype(torch::kFloat32));
  auto grad_site_rgba = torch::empty({site_count, 4}, site_rgba_f32.options().dtype(torch::kFloat32));
  auto& k = kernels();
  constexpr uint64_t clear_threads = 256ull;
  const int64_t chunk_count = (frame_count + int64_t(framegroup_threads) - 1) / int64_t(framegroup_threads);
  TORCH_CHECK(
      track_chunk_change_offsets_i16.size(0) == track_count * (chunk_count + 1),
      "track_chunk_change_offsets_i16 length mismatch");
  launch(k.clear_endpoint_loss_site_rgba_grad, [&](MetalKernelFunction& fn) {
    fn.setArg(0, loss);
    fn.setArg(1, grad_site_rgba);
    fn.setArg(2, config_i32);
    fn.dispatch((uint64_t)std::max<int64_t>(site_count, 1), clear_threads);
  });
  launch(framegroup_kernel, [&](MetalKernelFunction& fn) {
    fn.setArg(0, coeff_f16);
    fn.setArg(1, frame_t_f32);
    fn.setArg(2, base_offsets_i32);
    fn.setArg(3, base_record_i32);
    fn.setArg(4, track_change_offsets_i32);
    fn.setArg(5, track_chunk_change_offsets_i16);
    fn.setArg(6, change_frame_i32);
    fn.setArg(7, change_offsets_i32);
    fn.setArg(8, change_record_i32);
    fn.setArg(9, site_rgba_f32);
    fn.setArg(10, target_rgb_f32);
    fn.setArg(11, config_i32);
    fn.setArg(12, config_f32);
    fn.setArg(13, loss);
    fn.setArg(14, grad_site_rgba);
    fn.dispatch((uint64_t)track_count * (uint64_t)chunk_count * framegroup_threads, framegroup_threads);
  });
  return std::make_tuple(loss, grad_site_rgba);
}

std::tuple<torch::Tensor, torch::Tensor>
metal_endpoint_record_delta_replace_coeff16_packed_framegroup16_recompute_launch_only_mse_vjp_direct_atomic_rgb_only(
    const torch::Tensor& coeff_f16,
    const torch::Tensor& frame_t_f32,
    const torch::Tensor& base_offsets_i32,
    const torch::Tensor& base_record_i32,
    const torch::Tensor& track_change_offsets_i32,
    const torch::Tensor& track_chunk_change_offsets_i16,
    const torch::Tensor& change_frame_i32,
    const torch::Tensor& change_offsets_i32,
    const torch::Tensor& change_record_i32,
    const torch::Tensor& site_rgba_f32,
    const torch::Tensor& target_rgb_f32,
    const torch::Tensor& config_i32,
    const torch::Tensor& config_f32,
    const int64_t boundary_count,
    const int64_t track_count,
    const int64_t frame_count,
    const int64_t site_count,
    const int64_t base_record_count,
    const int64_t change_count,
    const int64_t change_record_count) {
  return metal_endpoint_record_delta_replace_coeff16_packed_framegroup16_launch_only_variant_mse_vjp_direct_atomic_rgb_only(
      coeff_f16,
      frame_t_f32,
      base_offsets_i32,
      base_record_i32,
      track_change_offsets_i32,
      track_chunk_change_offsets_i16,
      change_frame_i32,
      change_offsets_i32,
      change_record_i32,
      site_rgba_f32,
      target_rgb_f32,
      config_i32,
      config_f32,
      boundary_count,
      track_count,
      frame_count,
      site_count,
      base_record_count,
      change_count,
      change_record_count,
      kernels().endpoint_record_delta_replace_coeff16_packed_framegroup16_recompute_mse_vjp_direct_atomic_rgb_only,
      32ull,
      "packed framegroup16 recompute launch-only");
}

std::tuple<torch::Tensor, torch::Tensor>
metal_endpoint_record_delta_replace_coeff16_packed_framegroup16_smallrun16_launch_only_mse_vjp_direct_atomic_rgb_only(
    const torch::Tensor& coeff_f16,
    const torch::Tensor& frame_t_f32,
    const torch::Tensor& base_offsets_i32,
    const torch::Tensor& base_record_i32,
    const torch::Tensor& track_change_offsets_i32,
    const torch::Tensor& track_chunk_change_offsets_i16,
    const torch::Tensor& change_frame_i32,
    const torch::Tensor& change_offsets_i32,
    const torch::Tensor& change_record_i32,
    const torch::Tensor& site_rgba_f32,
    const torch::Tensor& target_rgb_f32,
    const torch::Tensor& config_i32,
    const torch::Tensor& config_f32,
    const int64_t boundary_count,
    const int64_t track_count,
    const int64_t frame_count,
    const int64_t site_count,
    const int64_t base_record_count,
    const int64_t change_count,
    const int64_t change_record_count) {
  return metal_endpoint_record_delta_replace_coeff16_packed_framegroup16_launch_only_variant_mse_vjp_direct_atomic_rgb_only(
      coeff_f16,
      frame_t_f32,
      base_offsets_i32,
      base_record_i32,
      track_change_offsets_i32,
      track_chunk_change_offsets_i16,
      change_frame_i32,
      change_offsets_i32,
      change_record_i32,
      site_rgba_f32,
      target_rgb_f32,
      config_i32,
      config_f32,
      boundary_count,
      track_count,
      frame_count,
      site_count,
      base_record_count,
      change_count,
      change_record_count,
      kernels().endpoint_record_delta_replace_coeff16_packed_framegroup16_smallrun16_mse_vjp_direct_atomic_rgb_only,
      32ull,
      "packed framegroup16 smallrun16 launch-only");
}

std::tuple<torch::Tensor, torch::Tensor>
metal_endpoint_record_delta_replace_coeff16_packed_framegroup16_materialized_launch_only_mse_vjp_direct_atomic_rgb_only(
    const torch::Tensor& coeff_f16,
    const torch::Tensor& frame_t_f32,
    const torch::Tensor& base_offsets_i32,
    const torch::Tensor& base_record_i32,
    const torch::Tensor& track_change_offsets_i32,
    const torch::Tensor& track_chunk_change_offsets_i16,
    const torch::Tensor& change_frame_i32,
    const torch::Tensor& change_offsets_i32,
    const torch::Tensor& change_record_i32,
    const torch::Tensor& site_rgba_f32,
    const torch::Tensor& target_rgb_f32,
    const torch::Tensor& config_i32,
    const torch::Tensor& config_f32,
    const int64_t boundary_count,
    const int64_t track_count,
    const int64_t frame_count,
    const int64_t site_count,
    const int64_t base_record_count,
    const int64_t change_count,
    const int64_t change_record_count) {
  return metal_endpoint_record_delta_replace_coeff16_packed_framegroup16_launch_only_variant_mse_vjp_direct_atomic_rgb_only(
      coeff_f16,
      frame_t_f32,
      base_offsets_i32,
      base_record_i32,
      track_change_offsets_i32,
      track_chunk_change_offsets_i16,
      change_frame_i32,
      change_offsets_i32,
      change_record_i32,
      site_rgba_f32,
      target_rgb_f32,
      config_i32,
      config_f32,
      boundary_count,
      track_count,
      frame_count,
      site_count,
      base_record_count,
      change_count,
      change_record_count,
      kernels().endpoint_record_delta_replace_coeff16_packed_framegroup16_materialized_mse_vjp_direct_atomic_rgb_only,
      16ull,
      "packed framegroup16 materialized launch-only");
}

std::tuple<torch::Tensor, torch::Tensor>
metal_endpoint_record_delta_replace_coeff16_packed_framegroup16_recompute_mse_vjp_direct_atomic_rgb_only(
    const torch::Tensor& coeff_f16,
    const torch::Tensor& frame_t_f32,
    const torch::Tensor& base_offsets_i32,
    const torch::Tensor& base_record_i32,
    const torch::Tensor& track_change_offsets_i32,
    const torch::Tensor& track_chunk_change_offsets_i16,
    const torch::Tensor& change_frame_i32,
    const torch::Tensor& change_offsets_i32,
    const torch::Tensor& change_record_i32,
    const torch::Tensor& site_rgba_f32,
    const torch::Tensor& target_rgb_f32,
    const torch::Tensor& config_i32,
    const torch::Tensor& config_f32) {
  check_half_mps_2d(coeff_f16, "coeff_f16", 4);
  check_float_mps_1d_any(frame_t_f32, "frame_t_f32");
  check_i32_mps_1d_any(base_offsets_i32, "base_offsets_i32");
  check_i32_mps_1d_any(base_record_i32, "base_record_i32");
  check_i32_mps_1d_any(track_change_offsets_i32, "track_change_offsets_i32");
  check_i16_mps_1d_any(track_chunk_change_offsets_i16, "track_chunk_change_offsets_i16");
  check_i32_mps_1d_any(change_frame_i32, "change_frame_i32");
  check_i32_mps_1d_any(change_offsets_i32, "change_offsets_i32");
  check_i32_mps_1d_any(change_record_i32, "change_record_i32");
  check_float_mps_2d(site_rgba_f32, "site_rgba_f32", 4);
  TORCH_CHECK(target_rgb_f32.device().is_mps(), "target_rgb_f32 must be on MPS");
  TORCH_CHECK(target_rgb_f32.scalar_type() == torch::kFloat32, "target_rgb_f32 must be float32");
  TORCH_CHECK(target_rgb_f32.dim() == 3 && target_rgb_f32.size(2) == 3, "target_rgb_f32 must have shape [K,T,3]");
  TORCH_CHECK(target_rgb_f32.is_contiguous(), "target_rgb_f32 must be contiguous");
  check_i32_mps_1d(config_i32, "config_i32", 7);
  TORCH_CHECK(config_f32.device().is_mps(), "config_f32 must be on MPS");
  TORCH_CHECK(config_f32.scalar_type() == torch::kFloat32, "config_f32 must be float32");
  TORCH_CHECK(config_f32.dim() == 1 && config_f32.size(0) == 4, "config_f32 must have shape [4]");
  TORCH_CHECK(config_f32.is_contiguous(), "config_f32 must be contiguous");

  auto config_i32_cpu = config_i32.cpu();
  const int32_t* config = config_i32_cpu.data_ptr<int32_t>();
  const int64_t boundary_count = config[0];
  const int64_t track_count = config[1];
  const int64_t frame_count = config[2];
  const int64_t site_count = config[3];
  const int64_t base_record_count = config[4];
  const int64_t change_count = config[5];
  const int64_t change_record_count = config[6];
  TORCH_CHECK(boundary_count > 0, "config_i32[0] boundary count must be positive");
  TORCH_CHECK(boundary_count <= 4093, "packed framegroup16 recompute delta replace coeff16 fused MSE supports boundary count <= 4093");
  TORCH_CHECK(track_count > 0, "config_i32[1] track count must be positive");
  TORCH_CHECK(frame_count > 0, "config_i32[2] frame count must be positive");
  TORCH_CHECK(site_count == site_rgba_f32.size(0), "config_i32[3] must match site_rgba_f32 rows");
  TORCH_CHECK(
      site_count > 0 && site_count <= 256,
      "packed framegroup16 recompute endpoint record delta replace coeff16 fused MSE VJP supports site count in [1, 256]");
  TORCH_CHECK(base_record_count >= 0, "config_i32[4] base record count must be nonnegative");
  TORCH_CHECK(change_count >= 0, "config_i32[5] change count must be nonnegative");
  TORCH_CHECK(change_count <= 32767, "int16 chunk-start offsets require change count <= 32767");
  TORCH_CHECK(change_record_count >= 0, "config_i32[6] change record count must be nonnegative");
  TORCH_CHECK(coeff_f16.size(0) == track_count * boundary_count, "coeff_f16 row count mismatch");
  TORCH_CHECK(frame_t_f32.size(0) == frame_count, "frame_t_f32 length must match frame count");
  TORCH_CHECK(base_record_i32.size(0) == base_record_count, "base_record_i32 length mismatch");
  TORCH_CHECK(change_frame_i32.size(0) == change_count, "change_frame_i32 length must match change count");
  TORCH_CHECK(change_record_i32.size(0) == change_record_count, "change_record_i32 length mismatch");
  TORCH_CHECK(base_offsets_i32.size(0) == track_count + 1, "base_offsets_i32 length must be track_count + 1");
  TORCH_CHECK(
      track_change_offsets_i32.size(0) == track_count + 1,
      "track_change_offsets_i32 length must be track_count + 1");
  TORCH_CHECK(change_offsets_i32.size(0) == change_count + 1, "change_offsets_i32 length must be change_count + 1");
  TORCH_CHECK(
      target_rgb_f32.size(0) == track_count && target_rgb_f32.size(1) == frame_count,
      "target_rgb_f32 shape mismatch");
  check_segment_tape_offsets_cpu(base_offsets_i32, track_count, base_record_count, 129);
  check_segment_tape_offsets_cpu(track_change_offsets_i32, track_count, change_count, 2147483647);
  check_segment_tape_offsets_cpu(change_offsets_i32, change_count, change_record_count, 129);

  auto loss = torch::empty({1}, site_rgba_f32.options().dtype(torch::kFloat32));
  auto grad_site_rgba = torch::empty({site_count, 4}, site_rgba_f32.options().dtype(torch::kFloat32));
  auto& k = kernels();
  constexpr uint64_t clear_threads = 256ull;
  constexpr uint64_t framegroup_threads = 32ull;
  const int64_t chunk_count = (frame_count + int64_t(framegroup_threads) - 1) / int64_t(framegroup_threads);
  TORCH_CHECK(
      track_chunk_change_offsets_i16.size(0) == track_count * (chunk_count + 1),
      "track_chunk_change_offsets_i16 length mismatch");
  auto chunk_offsets_cpu = track_chunk_change_offsets_i16.cpu();
  auto track_offsets_cpu = track_change_offsets_i32.cpu();
  const int16_t* chunk_offsets = chunk_offsets_cpu.data_ptr<int16_t>();
  const int32_t* track_offsets = track_offsets_cpu.data_ptr<int32_t>();
  for (int64_t track_id = 0; track_id < track_count; ++track_id) {
    const int32_t track_begin = track_offsets[track_id];
    const int32_t track_end = track_offsets[track_id + 1];
    TORCH_CHECK(track_begin >= 0 && track_end >= track_begin && track_end <= change_count, "track change offset bounds");
    int32_t previous = track_begin;
    for (int64_t chunk_id = 0; chunk_id <= chunk_count; ++chunk_id) {
      const int64_t chunk_index = track_id * (chunk_count + 1) + chunk_id;
      const int32_t value = static_cast<int32_t>(chunk_offsets[chunk_index]);
      TORCH_CHECK(
          value >= previous && value >= track_begin && value <= track_end,
          "track_chunk_change_offsets_i16 must be monotonic within each track and bounded by track changes");
      previous = value;
    }
    TORCH_CHECK(previous == track_end, "final chunk change offset must match track change end");
  }
  launch(k.clear_endpoint_loss_site_rgba_grad, [&](MetalKernelFunction& fn) {
    fn.setArg(0, loss);
    fn.setArg(1, grad_site_rgba);
    fn.setArg(2, config_i32);
    fn.dispatch((uint64_t)std::max<int64_t>(site_count, 1), clear_threads);
  });
  auto framegroup_kernel = k.endpoint_record_delta_replace_coeff16_packed_framegroup16_recompute_mse_vjp_direct_atomic_rgb_only;
  launch(
      framegroup_kernel,
      [&](MetalKernelFunction& fn) {
        fn.setArg(0, coeff_f16);
        fn.setArg(1, frame_t_f32);
        fn.setArg(2, base_offsets_i32);
        fn.setArg(3, base_record_i32);
        fn.setArg(4, track_change_offsets_i32);
        fn.setArg(5, track_chunk_change_offsets_i16);
        fn.setArg(6, change_frame_i32);
        fn.setArg(7, change_offsets_i32);
        fn.setArg(8, change_record_i32);
        fn.setArg(9, site_rgba_f32);
        fn.setArg(10, target_rgb_f32);
        fn.setArg(11, config_i32);
        fn.setArg(12, config_f32);
        fn.setArg(13, loss);
        fn.setArg(14, grad_site_rgba);
        fn.dispatch((uint64_t)track_count * (uint64_t)chunk_count * framegroup_threads, framegroup_threads);
      });
  return std::make_tuple(loss, grad_site_rgba);
}

std::tuple<torch::Tensor, torch::Tensor>
metal_endpoint_record_delta_replace_factorized_packed_framegroup16_recompute_mse_vjp_direct_atomic_rgb_only(
    const torch::Tensor& boundary_f32,
    const torch::Tensor& track_ray_coeff_f32,
    const torch::Tensor& frame_t_f32,
    const torch::Tensor& base_offsets_i16,
    const torch::Tensor& base_record_i32,
    const torch::Tensor& track_change_offsets_i16,
    const torch::Tensor& track_chunk_change_offsets_i16,
    const torch::Tensor& change_frame_i16,
    const torch::Tensor& change_offsets_i16,
    const torch::Tensor& change_record_i32,
    const torch::Tensor& site_rgba_f32,
    const torch::Tensor& target_rgb_f32,
    const torch::Tensor& config_i32,
    const torch::Tensor& config_f32) {
  check_float_mps_2d(boundary_f32, "boundary_f32", 5);
  check_float_mps_2d(track_ray_coeff_f32, "track_ray_coeff_f32", 12);
  check_float_mps_1d_any(frame_t_f32, "frame_t_f32");
  check_i16_mps_1d_any(base_offsets_i16, "base_offsets_i16");
  check_i32_mps_1d_any(base_record_i32, "base_record_i32");
  check_i16_mps_1d_any(track_change_offsets_i16, "track_change_offsets_i16");
  check_i16_mps_1d_any(track_chunk_change_offsets_i16, "track_chunk_change_offsets_i16");
  check_i16_mps_1d_any(change_frame_i16, "change_frame_i16");
  check_i16_mps_1d_any(change_offsets_i16, "change_offsets_i16");
  check_i32_mps_1d_any(change_record_i32, "change_record_i32");
  check_float_mps_2d(site_rgba_f32, "site_rgba_f32", 4);
  TORCH_CHECK(target_rgb_f32.device().is_mps(), "target_rgb_f32 must be on MPS");
  TORCH_CHECK(target_rgb_f32.scalar_type() == torch::kFloat32, "target_rgb_f32 must be float32");
  TORCH_CHECK(target_rgb_f32.dim() == 3 && target_rgb_f32.size(2) == 3, "target_rgb_f32 must have shape [K,T,3]");
  TORCH_CHECK(target_rgb_f32.is_contiguous(), "target_rgb_f32 must be contiguous");
  check_i32_mps_1d(config_i32, "config_i32", 7);
  TORCH_CHECK(config_f32.device().is_mps(), "config_f32 must be on MPS");
  TORCH_CHECK(config_f32.scalar_type() == torch::kFloat32, "config_f32 must be float32");
  TORCH_CHECK(config_f32.dim() == 1 && config_f32.size(0) == 4, "config_f32 must have shape [4]");
  TORCH_CHECK(config_f32.is_contiguous(), "config_f32 must be contiguous");

  auto config_i32_cpu = config_i32.cpu();
  const int32_t* config = config_i32_cpu.data_ptr<int32_t>();
  const int64_t boundary_count = config[0];
  const int64_t track_count = config[1];
  const int64_t frame_count = config[2];
  const int64_t site_count = config[3];
  const int64_t base_record_count = config[4];
  const int64_t change_count = config[5];
  const int64_t change_record_count = config[6];
  TORCH_CHECK(boundary_count > 0, "config_i32[0] boundary count must be positive");
  TORCH_CHECK(boundary_count <= 4093, "packed framegroup16 factorized delta replace fused MSE supports boundary count <= 4093");
  TORCH_CHECK(track_count > 0, "config_i32[1] track count must be positive");
  TORCH_CHECK(frame_count > 0, "config_i32[2] frame count must be positive");
  TORCH_CHECK(site_count == site_rgba_f32.size(0), "config_i32[3] must match site_rgba_f32 rows");
  TORCH_CHECK(
      site_count > 0 && site_count <= 256,
      "packed framegroup16 factorized endpoint record delta replace fused MSE VJP supports site count in [1, 256]");
  TORCH_CHECK(base_record_count >= 0, "config_i32[4] base record count must be nonnegative");
  TORCH_CHECK(change_count >= 0, "config_i32[5] change count must be nonnegative");
  TORCH_CHECK(change_count <= 32767, "int16 chunk-start offsets require change count <= 32767");
  TORCH_CHECK(change_record_count >= 0, "config_i32[6] change record count must be nonnegative");
  TORCH_CHECK(boundary_f32.size(0) == boundary_count, "boundary_f32 row count mismatch");
  TORCH_CHECK(track_ray_coeff_f32.size(0) == track_count, "track_ray_coeff_f32 row count mismatch");
  TORCH_CHECK(frame_t_f32.size(0) == frame_count, "frame_t_f32 length must match frame count");
  TORCH_CHECK(base_record_i32.size(0) == base_record_count, "base_record_i32 length mismatch");
  TORCH_CHECK(change_frame_i16.size(0) == change_count, "change_frame_i16 length must match change count");
  TORCH_CHECK(change_record_i32.size(0) == change_record_count, "change_record_i32 length mismatch");
  TORCH_CHECK(base_offsets_i16.size(0) == track_count + 1, "base_offsets_i16 length must be track_count + 1");
  TORCH_CHECK(
      track_change_offsets_i16.size(0) == track_count + 1,
      "track_change_offsets_i16 length must be track_count + 1");
  TORCH_CHECK(change_offsets_i16.size(0) == change_count + 1, "change_offsets_i16 length must be change_count + 1");
  TORCH_CHECK(
      target_rgb_f32.size(0) == track_count && target_rgb_f32.size(1) == frame_count,
      "target_rgb_f32 shape mismatch");
  check_segment_tape_offsets_i16_cpu(base_offsets_i16, track_count, base_record_count, 129);
  check_segment_tape_offsets_i16_cpu(track_change_offsets_i16, track_count, change_count, 2147483647);
  check_segment_tape_offsets_i16_cpu(change_offsets_i16, change_count, change_record_count, 129);

  auto loss = torch::empty({1}, site_rgba_f32.options().dtype(torch::kFloat32));
  auto grad_site_rgba = torch::empty({site_count, 4}, site_rgba_f32.options().dtype(torch::kFloat32));
  auto& k = kernels();
  constexpr uint64_t clear_threads = 256ull;
  constexpr uint64_t framegroup_threads = 32ull;
  const int64_t chunk_count = (frame_count + int64_t(framegroup_threads) - 1) / int64_t(framegroup_threads);
  TORCH_CHECK(
      track_chunk_change_offsets_i16.size(0) == track_count * (chunk_count + 1),
      "track_chunk_change_offsets_i16 length mismatch");
  auto chunk_offsets_cpu = track_chunk_change_offsets_i16.cpu();
  auto track_offsets_cpu = track_change_offsets_i16.cpu();
  const int16_t* chunk_offsets = chunk_offsets_cpu.data_ptr<int16_t>();
  const int16_t* track_offsets = track_offsets_cpu.data_ptr<int16_t>();
  for (int64_t track_id = 0; track_id < track_count; ++track_id) {
    const int32_t track_begin = static_cast<int32_t>(track_offsets[track_id]);
    const int32_t track_end = static_cast<int32_t>(track_offsets[track_id + 1]);
    TORCH_CHECK(track_begin >= 0 && track_end >= track_begin && track_end <= change_count, "track change offset bounds");
    int32_t previous = track_begin;
    for (int64_t chunk_id = 0; chunk_id <= chunk_count; ++chunk_id) {
      const int64_t chunk_index = track_id * (chunk_count + 1) + chunk_id;
      const int32_t value = static_cast<int32_t>(chunk_offsets[chunk_index]);
      TORCH_CHECK(
          value >= previous && value >= track_begin && value <= track_end,
          "track_chunk_change_offsets_i16 must be monotonic within each track and bounded by track changes");
      previous = value;
    }
    TORCH_CHECK(previous == track_end, "final chunk change offset must match track change end");
  }
  launch(k.clear_endpoint_loss_site_rgba_grad, [&](MetalKernelFunction& fn) {
    fn.setArg(0, loss);
    fn.setArg(1, grad_site_rgba);
    fn.setArg(2, config_i32);
    fn.dispatch((uint64_t)std::max<int64_t>(site_count, 1), clear_threads);
  });
  auto framegroup_kernel = k.endpoint_record_delta_replace_factorized_packed_framegroup16_recompute_mse_vjp_direct_atomic_rgb_only;
  launch(
      framegroup_kernel,
      [&](MetalKernelFunction& fn) {
        fn.setArg(0, boundary_f32);
        fn.setArg(1, track_ray_coeff_f32);
        fn.setArg(2, frame_t_f32);
        fn.setArg(3, base_offsets_i16);
        fn.setArg(4, base_record_i32);
        fn.setArg(5, track_change_offsets_i16);
        fn.setArg(6, track_chunk_change_offsets_i16);
        fn.setArg(7, change_frame_i16);
        fn.setArg(8, change_offsets_i16);
        fn.setArg(9, change_record_i32);
        fn.setArg(10, site_rgba_f32);
        fn.setArg(11, target_rgb_f32);
        fn.setArg(12, config_i32);
        fn.setArg(13, config_f32);
        fn.setArg(14, loss);
        fn.setArg(15, grad_site_rgba);
        fn.dispatch((uint64_t)track_count * (uint64_t)chunk_count * framegroup_threads, framegroup_threads);
      });
  return std::make_tuple(loss, grad_site_rgba);
}

std::tuple<torch::Tensor, torch::Tensor>
metal_endpoint_record_delta_replace_factorized_frameselect_recompute_mse_vjp_direct_atomic_rgb_only(
    const torch::Tensor& boundary_f32,
    const torch::Tensor& track_ray_coeff_f32,
    const torch::Tensor& frame_t_f32,
    const torch::Tensor& base_offsets_i16,
    const torch::Tensor& base_record_i32,
    const torch::Tensor& frame_change_index_i16,
    const torch::Tensor& change_offsets_i16,
    const torch::Tensor& change_record_i32,
    const torch::Tensor& site_rgba_f32,
    const torch::Tensor& target_rgb_f32,
    const torch::Tensor& config_i32,
    const torch::Tensor& config_f32) {
  check_float_mps_2d(boundary_f32, "boundary_f32", 5);
  check_float_mps_2d(track_ray_coeff_f32, "track_ray_coeff_f32", 12);
  check_float_mps_1d_any(frame_t_f32, "frame_t_f32");
  check_i16_mps_1d_any(base_offsets_i16, "base_offsets_i16");
  check_i32_mps_1d_any(base_record_i32, "base_record_i32");
  check_i16_mps_1d_any(frame_change_index_i16, "frame_change_index_i16");
  check_i16_mps_1d_any(change_offsets_i16, "change_offsets_i16");
  check_i32_mps_1d_any(change_record_i32, "change_record_i32");
  check_float_mps_2d(site_rgba_f32, "site_rgba_f32", 4);
  TORCH_CHECK(target_rgb_f32.device().is_mps(), "target_rgb_f32 must be on MPS");
  TORCH_CHECK(target_rgb_f32.scalar_type() == torch::kFloat32, "target_rgb_f32 must be float32");
  TORCH_CHECK(target_rgb_f32.dim() == 3 && target_rgb_f32.size(2) == 3, "target_rgb_f32 must have shape [K,T,3]");
  TORCH_CHECK(target_rgb_f32.is_contiguous(), "target_rgb_f32 must be contiguous");
  check_i32_mps_1d(config_i32, "config_i32", 7);
  TORCH_CHECK(config_f32.device().is_mps(), "config_f32 must be on MPS");
  TORCH_CHECK(config_f32.scalar_type() == torch::kFloat32, "config_f32 must be float32");
  TORCH_CHECK(config_f32.dim() == 1 && config_f32.size(0) == 4, "config_f32 must have shape [4]");
  TORCH_CHECK(config_f32.is_contiguous(), "config_f32 must be contiguous");

  auto config_i32_cpu = config_i32.cpu();
  const int32_t* config = config_i32_cpu.data_ptr<int32_t>();
  const int64_t boundary_count = config[0];
  const int64_t track_count = config[1];
  const int64_t frame_count = config[2];
  const int64_t site_count = config[3];
  const int64_t base_record_count = config[4];
  const int64_t change_count = config[5];
  const int64_t change_record_count = config[6];
  TORCH_CHECK(boundary_count > 0, "config_i32[0] boundary count must be positive");
  TORCH_CHECK(boundary_count <= 4093, "factorized frame-select delta replace fused MSE supports boundary count <= 4093");
  TORCH_CHECK(track_count > 0, "config_i32[1] track count must be positive");
  TORCH_CHECK(frame_count > 0, "config_i32[2] frame count must be positive");
  TORCH_CHECK(site_count == site_rgba_f32.size(0), "config_i32[3] must match site_rgba_f32 rows");
  TORCH_CHECK(site_count > 0 && site_count <= 256, "factorized frame-select fused MSE VJP supports site count in [1, 256]");
  TORCH_CHECK(base_record_count >= 0, "config_i32[4] base record count must be nonnegative");
  TORCH_CHECK(change_count >= 0, "config_i32[5] change count must be nonnegative");
  TORCH_CHECK(change_count <= 32767, "frame_change_index_i16 requires change count <= 32767");
  TORCH_CHECK(change_record_count >= 0, "config_i32[6] change record count must be nonnegative");
  TORCH_CHECK(boundary_f32.size(0) == boundary_count, "boundary_f32 row count mismatch");
  TORCH_CHECK(track_ray_coeff_f32.size(0) == track_count, "track_ray_coeff_f32 row count mismatch");
  TORCH_CHECK(frame_t_f32.size(0) == frame_count, "frame_t_f32 length must match frame count");
  TORCH_CHECK(base_record_i32.size(0) == base_record_count, "base_record_i32 length mismatch");
  TORCH_CHECK(change_record_i32.size(0) == change_record_count, "change_record_i32 length mismatch");
  TORCH_CHECK(base_offsets_i16.size(0) == track_count + 1, "base_offsets_i16 length must be track_count + 1");
  TORCH_CHECK(
      frame_change_index_i16.size(0) == track_count * std::max<int64_t>(frame_count - 1, 0),
      "frame_change_index_i16 length must be track_count * (frame_count - 1)");
  TORCH_CHECK(change_offsets_i16.size(0) == change_count + 1, "change_offsets_i16 length must be change_count + 1");
  TORCH_CHECK(target_rgb_f32.size(0) == track_count && target_rgb_f32.size(1) == frame_count, "target_rgb_f32 shape mismatch");
  check_segment_tape_offsets_i16_cpu(base_offsets_i16, track_count, base_record_count, 129);
  check_segment_tape_offsets_i16_cpu(change_offsets_i16, change_count, change_record_count, 129);
  auto frame_select_cpu = frame_change_index_i16.cpu();
  const int16_t* frame_select = frame_select_cpu.data_ptr<int16_t>();
  const int64_t frame_select_count = frame_select_cpu.numel();
  for (int64_t index = 0; index < frame_select_count; ++index) {
    const int32_t value = static_cast<int32_t>(frame_select[index]);
    TORCH_CHECK(value >= -1 && value < change_count, "frame_change_index_i16 values must be in [-1, change_count)");
  }

  auto loss = torch::empty({1}, site_rgba_f32.options().dtype(torch::kFloat32));
  auto grad_site_rgba = torch::empty({site_count, 4}, site_rgba_f32.options().dtype(torch::kFloat32));
  auto& k = kernels();
  constexpr uint64_t clear_threads = 256ull;
  constexpr uint64_t framegroup_threads = 32ull;
  const int64_t chunk_count = (frame_count + int64_t(framegroup_threads) - 1) / int64_t(framegroup_threads);
  launch(k.clear_endpoint_loss_site_rgba_grad, [&](MetalKernelFunction& fn) {
    fn.setArg(0, loss);
    fn.setArg(1, grad_site_rgba);
    fn.setArg(2, config_i32);
    fn.dispatch((uint64_t)std::max<int64_t>(site_count, 1), clear_threads);
  });
  auto framegroup_kernel = k.endpoint_record_delta_replace_factorized_frameselect_recompute_mse_vjp_direct_atomic_rgb_only;
  launch(
      framegroup_kernel,
      [&](MetalKernelFunction& fn) {
        fn.setArg(0, boundary_f32);
        fn.setArg(1, track_ray_coeff_f32);
        fn.setArg(2, frame_t_f32);
        fn.setArg(3, base_offsets_i16);
        fn.setArg(4, base_record_i32);
        fn.setArg(5, frame_change_index_i16);
        fn.setArg(6, change_offsets_i16);
        fn.setArg(7, change_record_i32);
        fn.setArg(8, site_rgba_f32);
        fn.setArg(9, target_rgb_f32);
        fn.setArg(10, config_i32);
        fn.setArg(11, config_f32);
        fn.setArg(12, loss);
        fn.setArg(13, grad_site_rgba);
        fn.dispatch((uint64_t)track_count * (uint64_t)chunk_count * framegroup_threads, framegroup_threads);
      });
  return std::make_tuple(loss, grad_site_rgba);
}

std::tuple<torch::Tensor, torch::Tensor>
metal_endpoint_record_delta_replace_factorized_framebitmask_recompute_mse_vjp_direct_atomic_rgb_only(
    const torch::Tensor& boundary_f32,
    const torch::Tensor& track_ray_coeff_f32,
    const torch::Tensor& frame_t_f32,
    const torch::Tensor& base_offsets_i32,
    const torch::Tensor& base_record_i32,
    const torch::Tensor& track_change_offsets_i32,
    const torch::Tensor& track_frame_mask_i32,
    const torch::Tensor& change_offsets_i32,
    const torch::Tensor& change_record_i32,
    const torch::Tensor& site_rgba_f32,
    const torch::Tensor& target_rgb_f32,
    const torch::Tensor& config_i32,
    const torch::Tensor& config_f32) {
  check_float_mps_2d(boundary_f32, "boundary_f32", 5);
  check_float_mps_2d(track_ray_coeff_f32, "track_ray_coeff_f32", 12);
  check_float_mps_1d_any(frame_t_f32, "frame_t_f32");
  check_i32_mps_1d_any(base_offsets_i32, "base_offsets_i32");
  check_i32_mps_1d_any(base_record_i32, "base_record_i32");
  check_i32_mps_1d_any(track_change_offsets_i32, "track_change_offsets_i32");
  check_i32_mps_1d_any(track_frame_mask_i32, "track_frame_mask_i32");
  check_i32_mps_1d_any(change_offsets_i32, "change_offsets_i32");
  check_i32_mps_1d_any(change_record_i32, "change_record_i32");
  check_float_mps_2d(site_rgba_f32, "site_rgba_f32", 4);
  TORCH_CHECK(target_rgb_f32.device().is_mps(), "target_rgb_f32 must be on MPS");
  TORCH_CHECK(target_rgb_f32.scalar_type() == torch::kFloat32, "target_rgb_f32 must be float32");
  TORCH_CHECK(target_rgb_f32.dim() == 3 && target_rgb_f32.size(2) == 3, "target_rgb_f32 must have shape [K,T,3]");
  TORCH_CHECK(target_rgb_f32.is_contiguous(), "target_rgb_f32 must be contiguous");
  check_i32_mps_1d(config_i32, "config_i32", 7);
  TORCH_CHECK(config_f32.device().is_mps(), "config_f32 must be on MPS");
  TORCH_CHECK(config_f32.scalar_type() == torch::kFloat32, "config_f32 must be float32");
  TORCH_CHECK(config_f32.dim() == 1 && config_f32.size(0) == 4, "config_f32 must have shape [4]");
  TORCH_CHECK(config_f32.is_contiguous(), "config_f32 must be contiguous");

  auto config_i32_cpu = config_i32.cpu();
  const int32_t* config = config_i32_cpu.data_ptr<int32_t>();
  const int64_t boundary_count = config[0];
  const int64_t track_count = config[1];
  const int64_t frame_count = config[2];
  const int64_t site_count = config[3];
  const int64_t base_record_count = config[4];
  const int64_t change_count = config[5];
  const int64_t change_record_count = config[6];
  TORCH_CHECK(boundary_count > 0, "config_i32[0] boundary count must be positive");
  TORCH_CHECK(boundary_count <= 4093, "factorized frame-bitmask delta replace fused MSE supports boundary count <= 4093");
  TORCH_CHECK(track_count > 0, "config_i32[1] track count must be positive");
  TORCH_CHECK(frame_count > 0, "config_i32[2] frame count must be positive");
  TORCH_CHECK(frame_count <= 32, "factorized frame-bitmask fused MSE supports frame count <= 32");
  TORCH_CHECK(site_count == site_rgba_f32.size(0), "config_i32[3] must match site_rgba_f32 rows");
  TORCH_CHECK(site_count > 0 && site_count <= 256, "factorized frame-bitmask fused MSE VJP supports site count in [1, 256]");
  TORCH_CHECK(base_record_count >= 0, "config_i32[4] base record count must be nonnegative");
  TORCH_CHECK(change_count >= 0, "config_i32[5] change count must be nonnegative");
  TORCH_CHECK(change_record_count >= 0, "config_i32[6] change record count must be nonnegative");
  TORCH_CHECK(boundary_f32.size(0) == boundary_count, "boundary_f32 row count mismatch");
  TORCH_CHECK(track_ray_coeff_f32.size(0) == track_count, "track_ray_coeff_f32 row count mismatch");
  TORCH_CHECK(frame_t_f32.size(0) == frame_count, "frame_t_f32 length must match frame count");
  TORCH_CHECK(base_record_i32.size(0) == base_record_count, "base_record_i32 length mismatch");
  TORCH_CHECK(change_record_i32.size(0) == change_record_count, "change_record_i32 length mismatch");
  TORCH_CHECK(base_offsets_i32.size(0) == track_count + 1, "base_offsets_i32 length must be track_count + 1");
  TORCH_CHECK(
      track_change_offsets_i32.size(0) == track_count + 1,
      "track_change_offsets_i32 length must be track_count + 1");
  TORCH_CHECK(track_frame_mask_i32.size(0) == track_count, "track_frame_mask_i32 length must be track_count");
  TORCH_CHECK(change_offsets_i32.size(0) == change_count + 1, "change_offsets_i32 length must be change_count + 1");
  TORCH_CHECK(target_rgb_f32.size(0) == track_count && target_rgb_f32.size(1) == frame_count, "target_rgb_f32 shape mismatch");
  check_segment_tape_offsets_cpu(base_offsets_i32, track_count, base_record_count, 129);
  check_segment_tape_offsets_cpu(track_change_offsets_i32, track_count, change_count, 2147483647);
  check_segment_tape_offsets_cpu(change_offsets_i32, change_count, change_record_count, 129);
  auto mask_cpu = track_frame_mask_i32.cpu();
  const int32_t* masks = mask_cpu.data_ptr<int32_t>();
  const int64_t mask_count = mask_cpu.numel();
  const uint32_t allowed_mask = frame_count == 32
      ? 0xFFFFFFFEu
      : ((uint32_t{1} << frame_count) - uint32_t{1}) & ~uint32_t{1};
  for (int64_t index = 0; index < mask_count; ++index) {
    const uint32_t mask = static_cast<uint32_t>(masks[index]);
    TORCH_CHECK((mask & ~allowed_mask) == 0u, "track_frame_mask_i32 contains bits outside [1, frame_count)");
  }

  auto loss = torch::empty({1}, site_rgba_f32.options().dtype(torch::kFloat32));
  auto grad_site_rgba = torch::empty({site_count, 4}, site_rgba_f32.options().dtype(torch::kFloat32));
  auto& k = kernels();
  constexpr uint64_t clear_threads = 256ull;
  constexpr uint64_t framegroup_threads = 32ull;
  const int64_t chunk_count = (frame_count + int64_t(framegroup_threads) - 1) / int64_t(framegroup_threads);
  launch(k.clear_endpoint_loss_site_rgba_grad, [&](MetalKernelFunction& fn) {
    fn.setArg(0, loss);
    fn.setArg(1, grad_site_rgba);
    fn.setArg(2, config_i32);
    fn.dispatch((uint64_t)std::max<int64_t>(site_count, 1), clear_threads);
  });
  auto framegroup_kernel = k.endpoint_record_delta_replace_factorized_framebitmask_recompute_mse_vjp_direct_atomic_rgb_only;
  launch(
      framegroup_kernel,
      [&](MetalKernelFunction& fn) {
        fn.setArg(0, boundary_f32);
        fn.setArg(1, track_ray_coeff_f32);
        fn.setArg(2, frame_t_f32);
        fn.setArg(3, base_offsets_i32);
        fn.setArg(4, base_record_i32);
        fn.setArg(5, track_change_offsets_i32);
        fn.setArg(6, track_frame_mask_i32);
        fn.setArg(7, change_offsets_i32);
        fn.setArg(8, change_record_i32);
        fn.setArg(9, site_rgba_f32);
        fn.setArg(10, target_rgb_f32);
        fn.setArg(11, config_i32);
        fn.setArg(12, config_f32);
        fn.setArg(13, loss);
        fn.setArg(14, grad_site_rgba);
        fn.dispatch((uint64_t)track_count * (uint64_t)chunk_count * framegroup_threads, framegroup_threads);
      });
  return std::make_tuple(loss, grad_site_rgba);
}

std::tuple<torch::Tensor, torch::Tensor>
metal_endpoint_record_delta_replace_coeff16_packed_framegroup16_smallrun16_mse_vjp_direct_atomic_rgb_only(
    const torch::Tensor& coeff_f16,
    const torch::Tensor& frame_t_f32,
    const torch::Tensor& base_offsets_i32,
    const torch::Tensor& base_record_i32,
    const torch::Tensor& track_change_offsets_i32,
    const torch::Tensor& track_chunk_change_offsets_i16,
    const torch::Tensor& change_frame_i32,
    const torch::Tensor& change_offsets_i32,
    const torch::Tensor& change_record_i32,
    const torch::Tensor& site_rgba_f32,
    const torch::Tensor& target_rgb_f32,
    const torch::Tensor& config_i32,
    const torch::Tensor& config_f32) {
  check_half_mps_2d(coeff_f16, "coeff_f16", 4);
  check_float_mps_1d_any(frame_t_f32, "frame_t_f32");
  check_i32_mps_1d_any(base_offsets_i32, "base_offsets_i32");
  check_i32_mps_1d_any(base_record_i32, "base_record_i32");
  check_i32_mps_1d_any(track_change_offsets_i32, "track_change_offsets_i32");
  check_i16_mps_1d_any(track_chunk_change_offsets_i16, "track_chunk_change_offsets_i16");
  check_i32_mps_1d_any(change_frame_i32, "change_frame_i32");
  check_i32_mps_1d_any(change_offsets_i32, "change_offsets_i32");
  check_i32_mps_1d_any(change_record_i32, "change_record_i32");
  check_float_mps_2d(site_rgba_f32, "site_rgba_f32", 4);
  TORCH_CHECK(target_rgb_f32.device().is_mps(), "target_rgb_f32 must be on MPS");
  TORCH_CHECK(target_rgb_f32.scalar_type() == torch::kFloat32, "target_rgb_f32 must be float32");
  TORCH_CHECK(target_rgb_f32.dim() == 3 && target_rgb_f32.size(2) == 3, "target_rgb_f32 must have shape [K,T,3]");
  TORCH_CHECK(target_rgb_f32.is_contiguous(), "target_rgb_f32 must be contiguous");
  check_i32_mps_1d(config_i32, "config_i32", 7);
  TORCH_CHECK(config_f32.device().is_mps(), "config_f32 must be on MPS");
  TORCH_CHECK(config_f32.scalar_type() == torch::kFloat32, "config_f32 must be float32");
  TORCH_CHECK(config_f32.dim() == 1 && config_f32.size(0) == 4, "config_f32 must have shape [4]");
  TORCH_CHECK(config_f32.is_contiguous(), "config_f32 must be contiguous");

  auto config_i32_cpu = config_i32.cpu();
  const int32_t* config = config_i32_cpu.data_ptr<int32_t>();
  const int64_t boundary_count = config[0];
  const int64_t track_count = config[1];
  const int64_t frame_count = config[2];
  const int64_t site_count = config[3];
  const int64_t base_record_count = config[4];
  const int64_t change_count = config[5];
  const int64_t change_record_count = config[6];
  TORCH_CHECK(boundary_count > 0, "config_i32[0] boundary count must be positive");
  TORCH_CHECK(boundary_count <= 4093, "packed framegroup16 smallrun16 delta replace coeff16 fused MSE supports boundary count <= 4093");
  TORCH_CHECK(track_count > 0, "config_i32[1] track count must be positive");
  TORCH_CHECK(frame_count > 0, "config_i32[2] frame count must be positive");
  TORCH_CHECK(site_count == site_rgba_f32.size(0), "config_i32[3] must match site_rgba_f32 rows");
  TORCH_CHECK(
      site_count > 0 && site_count <= 256,
      "packed framegroup16 smallrun16 endpoint record delta replace coeff16 fused MSE VJP supports site count in [1, 256]");
  TORCH_CHECK(base_record_count >= 0, "config_i32[4] base record count must be nonnegative");
  TORCH_CHECK(change_count >= 0, "config_i32[5] change count must be nonnegative");
  TORCH_CHECK(change_count <= 32767, "int16 chunk-start offsets require change count <= 32767");
  TORCH_CHECK(change_record_count >= 0, "config_i32[6] change record count must be nonnegative");
  TORCH_CHECK(coeff_f16.size(0) == track_count * boundary_count, "coeff_f16 row count mismatch");
  TORCH_CHECK(frame_t_f32.size(0) == frame_count, "frame_t_f32 length must match frame count");
  TORCH_CHECK(base_record_i32.size(0) == base_record_count, "base_record_i32 length mismatch");
  TORCH_CHECK(change_frame_i32.size(0) == change_count, "change_frame_i32 length must match change count");
  TORCH_CHECK(change_record_i32.size(0) == change_record_count, "change_record_i32 length mismatch");
  TORCH_CHECK(base_offsets_i32.size(0) == track_count + 1, "base_offsets_i32 length must be track_count + 1");
  TORCH_CHECK(
      track_change_offsets_i32.size(0) == track_count + 1,
      "track_change_offsets_i32 length must be track_count + 1");
  TORCH_CHECK(change_offsets_i32.size(0) == change_count + 1, "change_offsets_i32 length must be change_count + 1");
  TORCH_CHECK(
      target_rgb_f32.size(0) == track_count && target_rgb_f32.size(1) == frame_count,
      "target_rgb_f32 shape mismatch");
  check_segment_tape_offsets_cpu(base_offsets_i32, track_count, base_record_count, 16);
  check_segment_tape_offsets_cpu(track_change_offsets_i32, track_count, change_count, 2147483647);
  check_segment_tape_offsets_cpu(change_offsets_i32, change_count, change_record_count, 16);

  auto loss = torch::empty({1}, site_rgba_f32.options().dtype(torch::kFloat32));
  auto grad_site_rgba = torch::empty({site_count, 4}, site_rgba_f32.options().dtype(torch::kFloat32));
  auto& k = kernels();
  constexpr uint64_t clear_threads = 256ull;
  constexpr uint64_t framegroup_threads = 32ull;
  const int64_t chunk_count = (frame_count + int64_t(framegroup_threads) - 1) / int64_t(framegroup_threads);
  TORCH_CHECK(
      track_chunk_change_offsets_i16.size(0) == track_count * (chunk_count + 1),
      "track_chunk_change_offsets_i16 length mismatch");
  auto chunk_offsets_cpu = track_chunk_change_offsets_i16.cpu();
  auto track_offsets_cpu = track_change_offsets_i32.cpu();
  const int16_t* chunk_offsets = chunk_offsets_cpu.data_ptr<int16_t>();
  const int32_t* track_offsets = track_offsets_cpu.data_ptr<int32_t>();
  for (int64_t track_id = 0; track_id < track_count; ++track_id) {
    const int32_t track_begin = track_offsets[track_id];
    const int32_t track_end = track_offsets[track_id + 1];
    TORCH_CHECK(track_begin >= 0 && track_end >= track_begin && track_end <= change_count, "track change offset bounds");
    int32_t previous = track_begin;
    for (int64_t chunk_id = 0; chunk_id <= chunk_count; ++chunk_id) {
      const int64_t chunk_index = track_id * (chunk_count + 1) + chunk_id;
      const int32_t value = static_cast<int32_t>(chunk_offsets[chunk_index]);
      TORCH_CHECK(
          value >= previous && value >= track_begin && value <= track_end,
          "track_chunk_change_offsets_i16 must be monotonic within each track and bounded by track changes");
      previous = value;
    }
    TORCH_CHECK(previous == track_end, "final chunk change offset must match track change end");
  }
  launch(k.clear_endpoint_loss_site_rgba_grad, [&](MetalKernelFunction& fn) {
    fn.setArg(0, loss);
    fn.setArg(1, grad_site_rgba);
    fn.setArg(2, config_i32);
    fn.dispatch((uint64_t)std::max<int64_t>(site_count, 1), clear_threads);
  });
  auto framegroup_kernel = k.endpoint_record_delta_replace_coeff16_packed_framegroup16_smallrun16_mse_vjp_direct_atomic_rgb_only;
  launch(
      framegroup_kernel,
      [&](MetalKernelFunction& fn) {
        fn.setArg(0, coeff_f16);
        fn.setArg(1, frame_t_f32);
        fn.setArg(2, base_offsets_i32);
        fn.setArg(3, base_record_i32);
        fn.setArg(4, track_change_offsets_i32);
        fn.setArg(5, track_chunk_change_offsets_i16);
        fn.setArg(6, change_frame_i32);
        fn.setArg(7, change_offsets_i32);
        fn.setArg(8, change_record_i32);
        fn.setArg(9, site_rgba_f32);
        fn.setArg(10, target_rgb_f32);
        fn.setArg(11, config_i32);
        fn.setArg(12, config_f32);
        fn.setArg(13, loss);
        fn.setArg(14, grad_site_rgba);
        fn.dispatch((uint64_t)track_count * (uint64_t)chunk_count * framegroup_threads, framegroup_threads);
      });
  return std::make_tuple(loss, grad_site_rgba);
}

std::tuple<torch::Tensor, torch::Tensor>
metal_endpoint_record_delta_replace_coeff16_packed_framegroup16_materialized_mse_vjp_direct_atomic_rgb_only(
    const torch::Tensor& coeff_f16,
    const torch::Tensor& frame_t_f32,
    const torch::Tensor& base_offsets_i32,
    const torch::Tensor& base_record_i32,
    const torch::Tensor& track_change_offsets_i32,
    const torch::Tensor& track_chunk_change_offsets_i16,
    const torch::Tensor& change_frame_i32,
    const torch::Tensor& change_offsets_i32,
    const torch::Tensor& change_record_i32,
    const torch::Tensor& site_rgba_f32,
    const torch::Tensor& target_rgb_f32,
    const torch::Tensor& config_i32,
    const torch::Tensor& config_f32) {
  check_half_mps_2d(coeff_f16, "coeff_f16", 4);
  check_float_mps_1d_any(frame_t_f32, "frame_t_f32");
  check_i32_mps_1d_any(base_offsets_i32, "base_offsets_i32");
  check_i32_mps_1d_any(base_record_i32, "base_record_i32");
  check_i32_mps_1d_any(track_change_offsets_i32, "track_change_offsets_i32");
  check_i16_mps_1d_any(track_chunk_change_offsets_i16, "track_chunk_change_offsets_i16");
  check_i32_mps_1d_any(change_frame_i32, "change_frame_i32");
  check_i32_mps_1d_any(change_offsets_i32, "change_offsets_i32");
  check_i32_mps_1d_any(change_record_i32, "change_record_i32");
  check_float_mps_2d(site_rgba_f32, "site_rgba_f32", 4);
  TORCH_CHECK(target_rgb_f32.device().is_mps(), "target_rgb_f32 must be on MPS");
  TORCH_CHECK(target_rgb_f32.scalar_type() == torch::kFloat32, "target_rgb_f32 must be float32");
  TORCH_CHECK(target_rgb_f32.dim() == 3 && target_rgb_f32.size(2) == 3, "target_rgb_f32 must have shape [K,T,3]");
  TORCH_CHECK(target_rgb_f32.is_contiguous(), "target_rgb_f32 must be contiguous");
  check_i32_mps_1d(config_i32, "config_i32", 7);
  TORCH_CHECK(config_f32.device().is_mps(), "config_f32 must be on MPS");
  TORCH_CHECK(config_f32.scalar_type() == torch::kFloat32, "config_f32 must be float32");
  TORCH_CHECK(config_f32.dim() == 1 && config_f32.size(0) == 4, "config_f32 must have shape [4]");
  TORCH_CHECK(config_f32.is_contiguous(), "config_f32 must be contiguous");

  auto config_i32_cpu = config_i32.cpu();
  const int32_t* config = config_i32_cpu.data_ptr<int32_t>();
  const int64_t boundary_count = config[0];
  const int64_t track_count = config[1];
  const int64_t frame_count = config[2];
  const int64_t site_count = config[3];
  const int64_t base_record_count = config[4];
  const int64_t change_count = config[5];
  const int64_t change_record_count = config[6];
  TORCH_CHECK(boundary_count > 0, "config_i32[0] boundary count must be positive");
  TORCH_CHECK(
      boundary_count <= 4093,
      "packed materialized framegroup16 delta replace coeff16 fused MSE supports boundary count <= 4093");
  TORCH_CHECK(track_count > 0, "config_i32[1] track count must be positive");
  TORCH_CHECK(frame_count > 0, "config_i32[2] frame count must be positive");
  TORCH_CHECK(site_count == site_rgba_f32.size(0), "config_i32[3] must match site_rgba_f32 rows");
  TORCH_CHECK(
      site_count > 0 && site_count <= 256,
      "packed materialized framegroup16 endpoint record delta replace coeff16 fused MSE VJP supports site count in [1, 256]");
  TORCH_CHECK(base_record_count >= 0, "config_i32[4] base record count must be nonnegative");
  TORCH_CHECK(change_count >= 0, "config_i32[5] change count must be nonnegative");
  TORCH_CHECK(change_count <= 32767, "int16 chunk-start offsets require change count <= 32767");
  TORCH_CHECK(change_record_count >= 0, "config_i32[6] change record count must be nonnegative");
  TORCH_CHECK(coeff_f16.size(0) == track_count * boundary_count, "coeff_f16 row count mismatch");
  TORCH_CHECK(frame_t_f32.size(0) == frame_count, "frame_t_f32 length must match frame count");
  TORCH_CHECK(base_record_i32.size(0) == base_record_count, "base_record_i32 length mismatch");
  TORCH_CHECK(change_frame_i32.size(0) == change_count, "change_frame_i32 length must match change count");
  TORCH_CHECK(change_record_i32.size(0) == change_record_count, "change_record_i32 length mismatch");
  TORCH_CHECK(base_offsets_i32.size(0) == track_count + 1, "base_offsets_i32 length must be track_count + 1");
  TORCH_CHECK(
      track_change_offsets_i32.size(0) == track_count + 1,
      "track_change_offsets_i32 length must be track_count + 1");
  TORCH_CHECK(change_offsets_i32.size(0) == change_count + 1, "change_offsets_i32 length must be change_count + 1");
  TORCH_CHECK(
      target_rgb_f32.size(0) == track_count && target_rgb_f32.size(1) == frame_count,
      "target_rgb_f32 shape mismatch");
  check_segment_tape_offsets_cpu(base_offsets_i32, track_count, base_record_count, 129);
  check_segment_tape_offsets_cpu(track_change_offsets_i32, track_count, change_count, 2147483647);
  check_segment_tape_offsets_cpu(change_offsets_i32, change_count, change_record_count, 129);

  auto loss = torch::empty({1}, site_rgba_f32.options().dtype(torch::kFloat32));
  auto grad_site_rgba = torch::empty({site_count, 4}, site_rgba_f32.options().dtype(torch::kFloat32));
  auto& k = kernels();
  constexpr uint64_t clear_threads = 256ull;
  constexpr uint64_t framegroup_threads = 16ull;
  const int64_t chunk_count = (frame_count + int64_t(framegroup_threads) - 1) / int64_t(framegroup_threads);
  TORCH_CHECK(
      track_chunk_change_offsets_i16.size(0) == track_count * (chunk_count + 1),
      "track_chunk_change_offsets_i16 length mismatch");
  auto chunk_offsets_cpu = track_chunk_change_offsets_i16.cpu();
  auto track_offsets_cpu = track_change_offsets_i32.cpu();
  const int16_t* chunk_offsets = chunk_offsets_cpu.data_ptr<int16_t>();
  const int32_t* track_offsets = track_offsets_cpu.data_ptr<int32_t>();
  for (int64_t track_id = 0; track_id < track_count; ++track_id) {
    const int32_t track_begin = track_offsets[track_id];
    const int32_t track_end = track_offsets[track_id + 1];
    TORCH_CHECK(track_begin >= 0 && track_end >= track_begin && track_end <= change_count, "track change offset bounds");
    int32_t previous = track_begin;
    for (int64_t chunk_id = 0; chunk_id <= chunk_count; ++chunk_id) {
      const int64_t chunk_index = track_id * (chunk_count + 1) + chunk_id;
      const int32_t value = static_cast<int32_t>(chunk_offsets[chunk_index]);
      TORCH_CHECK(
          value >= previous && value >= track_begin && value <= track_end,
          "track_chunk_change_offsets_i16 must be monotonic within each track and bounded by track changes");
      previous = value;
    }
    TORCH_CHECK(previous == track_end, "final chunk change offset must match track change end");
  }
  launch(k.clear_endpoint_loss_site_rgba_grad, [&](MetalKernelFunction& fn) {
    fn.setArg(0, loss);
    fn.setArg(1, grad_site_rgba);
    fn.setArg(2, config_i32);
    fn.dispatch((uint64_t)std::max<int64_t>(site_count, 1), clear_threads);
  });
  launch(
      k.endpoint_record_delta_replace_coeff16_packed_framegroup16_materialized_mse_vjp_direct_atomic_rgb_only,
      [&](MetalKernelFunction& fn) {
        fn.setArg(0, coeff_f16);
        fn.setArg(1, frame_t_f32);
        fn.setArg(2, base_offsets_i32);
        fn.setArg(3, base_record_i32);
        fn.setArg(4, track_change_offsets_i32);
        fn.setArg(5, track_chunk_change_offsets_i16);
        fn.setArg(6, change_frame_i32);
        fn.setArg(7, change_offsets_i32);
        fn.setArg(8, change_record_i32);
        fn.setArg(9, site_rgba_f32);
        fn.setArg(10, target_rgb_f32);
        fn.setArg(11, config_i32);
        fn.setArg(12, config_f32);
        fn.setArg(13, loss);
        fn.setArg(14, grad_site_rgba);
        fn.dispatch((uint64_t)track_count * (uint64_t)chunk_count * framegroup_threads, framegroup_threads);
      });
  return std::make_tuple(loss, grad_site_rgba);
}

std::tuple<torch::Tensor, torch::Tensor>
metal_endpoint_record_delta_replace_coeff16_i16x3_framegroup16_materialized_mse_vjp_direct_atomic_rgb_only(
    const torch::Tensor& coeff_f16,
    const torch::Tensor& frame_t_f32,
    const torch::Tensor& base_offsets_i32,
    const torch::Tensor& base_record_i16,
    const torch::Tensor& track_change_offsets_i32,
    const torch::Tensor& track_chunk_change_offsets_i16,
    const torch::Tensor& change_frame_i32,
    const torch::Tensor& change_offsets_i32,
    const torch::Tensor& change_record_i16,
    const torch::Tensor& site_rgba_f32,
    const torch::Tensor& target_rgb_f32,
    const torch::Tensor& config_i32,
    const torch::Tensor& config_f32) {
  check_half_mps_2d(coeff_f16, "coeff_f16", 4);
  check_float_mps_1d_any(frame_t_f32, "frame_t_f32");
  check_i32_mps_1d_any(base_offsets_i32, "base_offsets_i32");
  check_i16_mps_1d_any(base_record_i16, "base_record_i16");
  check_i32_mps_1d_any(track_change_offsets_i32, "track_change_offsets_i32");
  check_i16_mps_1d_any(track_chunk_change_offsets_i16, "track_chunk_change_offsets_i16");
  check_i32_mps_1d_any(change_frame_i32, "change_frame_i32");
  check_i32_mps_1d_any(change_offsets_i32, "change_offsets_i32");
  check_i16_mps_1d_any(change_record_i16, "change_record_i16");
  check_float_mps_2d(site_rgba_f32, "site_rgba_f32", 4);
  TORCH_CHECK(target_rgb_f32.device().is_mps(), "target_rgb_f32 must be on MPS");
  TORCH_CHECK(target_rgb_f32.scalar_type() == torch::kFloat32, "target_rgb_f32 must be float32");
  TORCH_CHECK(target_rgb_f32.dim() == 3 && target_rgb_f32.size(2) == 3, "target_rgb_f32 must have shape [K,T,3]");
  TORCH_CHECK(target_rgb_f32.is_contiguous(), "target_rgb_f32 must be contiguous");
  check_i32_mps_1d(config_i32, "config_i32", 7);
  TORCH_CHECK(config_f32.device().is_mps(), "config_f32 must be on MPS");
  TORCH_CHECK(config_f32.scalar_type() == torch::kFloat32, "config_f32 must be float32");
  TORCH_CHECK(config_f32.dim() == 1 && config_f32.size(0) == 4, "config_f32 must have shape [4]");
  TORCH_CHECK(config_f32.is_contiguous(), "config_f32 must be contiguous");

  auto config_i32_cpu = config_i32.cpu();
  const int32_t* config = config_i32_cpu.data_ptr<int32_t>();
  const int64_t boundary_count = config[0];
  const int64_t track_count = config[1];
  const int64_t frame_count = config[2];
  const int64_t site_count = config[3];
  const int64_t base_record_count = config[4];
  const int64_t change_count = config[5];
  const int64_t change_record_count = config[6];
  TORCH_CHECK(boundary_count > 0, "config_i32[0] boundary count must be positive");
  TORCH_CHECK(
      boundary_count <= 32765,
      "i16x3 materialized framegroup16 delta replace coeff16 fused MSE supports boundary count <= 32765");
  TORCH_CHECK(track_count > 0, "config_i32[1] track count must be positive");
  TORCH_CHECK(frame_count > 0, "config_i32[2] frame count must be positive");
  TORCH_CHECK(site_count == site_rgba_f32.size(0), "config_i32[3] must match site_rgba_f32 rows");
  TORCH_CHECK(
      site_count > 0 && site_count <= 32767,
      "i16x3 materialized framegroup16 endpoint record delta replace coeff16 fused MSE VJP supports site count in [1, 32767]");
  TORCH_CHECK(base_record_count >= 0, "config_i32[4] base record count must be nonnegative");
  TORCH_CHECK(change_count >= 0, "config_i32[5] change count must be nonnegative");
  TORCH_CHECK(change_count <= 32767, "int16 chunk-start offsets require change count <= 32767");
  TORCH_CHECK(change_record_count >= 0, "config_i32[6] change record count must be nonnegative");
  TORCH_CHECK(coeff_f16.size(0) == track_count * boundary_count, "coeff_f16 row count mismatch");
  TORCH_CHECK(frame_t_f32.size(0) == frame_count, "frame_t_f32 length must match frame count");
  TORCH_CHECK(base_record_i16.size(0) == base_record_count * 3, "base_record_i16 length mismatch");
  TORCH_CHECK(change_frame_i32.size(0) == change_count, "change_frame_i32 length must match change count");
  TORCH_CHECK(change_record_i16.size(0) == change_record_count * 3, "change_record_i16 length mismatch");
  TORCH_CHECK(base_offsets_i32.size(0) == track_count + 1, "base_offsets_i32 length must be track_count + 1");
  TORCH_CHECK(
      track_change_offsets_i32.size(0) == track_count + 1,
      "track_change_offsets_i32 length must be track_count + 1");
  TORCH_CHECK(change_offsets_i32.size(0) == change_count + 1, "change_offsets_i32 length must be change_count + 1");
  TORCH_CHECK(
      target_rgb_f32.size(0) == track_count && target_rgb_f32.size(1) == frame_count,
      "target_rgb_f32 shape mismatch");
  check_segment_tape_offsets_cpu(base_offsets_i32, track_count, base_record_count, 129);
  check_segment_tape_offsets_cpu(track_change_offsets_i32, track_count, change_count, 2147483647);
  check_segment_tape_offsets_cpu(change_offsets_i32, change_count, change_record_count, 129);

  auto loss = torch::empty({1}, site_rgba_f32.options().dtype(torch::kFloat32));
  auto grad_site_rgba = torch::empty({site_count, 4}, site_rgba_f32.options().dtype(torch::kFloat32));
  auto& k = kernels();
  constexpr uint64_t clear_threads = 256ull;
  constexpr uint64_t framegroup_threads = 16ull;
  const int64_t chunk_count = (frame_count + int64_t(framegroup_threads) - 1) / int64_t(framegroup_threads);
  TORCH_CHECK(
      track_chunk_change_offsets_i16.size(0) == track_count * (chunk_count + 1),
      "track_chunk_change_offsets_i16 length mismatch");
  auto chunk_offsets_cpu = track_chunk_change_offsets_i16.cpu();
  auto track_offsets_cpu = track_change_offsets_i32.cpu();
  const int16_t* chunk_offsets = chunk_offsets_cpu.data_ptr<int16_t>();
  const int32_t* track_offsets = track_offsets_cpu.data_ptr<int32_t>();
  for (int64_t track_id = 0; track_id < track_count; ++track_id) {
    const int32_t track_begin = track_offsets[track_id];
    const int32_t track_end = track_offsets[track_id + 1];
    TORCH_CHECK(track_begin >= 0 && track_end >= track_begin && track_end <= change_count, "track change offset bounds");
    int32_t previous = track_begin;
    for (int64_t chunk_id = 0; chunk_id <= chunk_count; ++chunk_id) {
      const int64_t chunk_index = track_id * (chunk_count + 1) + chunk_id;
      const int32_t value = static_cast<int32_t>(chunk_offsets[chunk_index]);
      TORCH_CHECK(
          value >= previous && value >= track_begin && value <= track_end,
          "track_chunk_change_offsets_i16 must be monotonic within each track and bounded by track changes");
      previous = value;
    }
    TORCH_CHECK(previous == track_end, "final chunk change offset must match track change end");
  }
  launch(k.clear_endpoint_loss_site_rgba_grad, [&](MetalKernelFunction& fn) {
    fn.setArg(0, loss);
    fn.setArg(1, grad_site_rgba);
    fn.setArg(2, config_i32);
    fn.dispatch((uint64_t)std::max<int64_t>(site_count, 1), clear_threads);
  });
  launch(
      k.endpoint_record_delta_replace_coeff16_i16x3_framegroup16_materialized_mse_vjp_direct_atomic_rgb_only,
      [&](MetalKernelFunction& fn) {
        fn.setArg(0, coeff_f16);
        fn.setArg(1, frame_t_f32);
        fn.setArg(2, base_offsets_i32);
        fn.setArg(3, base_record_i16);
        fn.setArg(4, track_change_offsets_i32);
        fn.setArg(5, track_chunk_change_offsets_i16);
        fn.setArg(6, change_frame_i32);
        fn.setArg(7, change_offsets_i32);
        fn.setArg(8, change_record_i16);
        fn.setArg(9, site_rgba_f32);
        fn.setArg(10, target_rgb_f32);
        fn.setArg(11, config_i32);
        fn.setArg(12, config_f32);
        fn.setArg(13, loss);
        fn.setArg(14, grad_site_rgba);
        fn.dispatch((uint64_t)track_count * (uint64_t)chunk_count * framegroup_threads, framegroup_threads);
      });
  return std::make_tuple(loss, grad_site_rgba);
}

std::tuple<torch::Tensor, torch::Tensor>
metal_endpoint_record_delta_replace_coeff16_i16x4_framegroup16_mse_vjp_direct_atomic_rgb_only(
    const torch::Tensor& coeff_f16,
    const torch::Tensor& frame_t_f32,
    const torch::Tensor& base_offsets_i32,
    const torch::Tensor& base_record_i16,
    const torch::Tensor& track_change_offsets_i32,
    const torch::Tensor& track_chunk_change_offsets_i16,
    const torch::Tensor& change_frame_i32,
    const torch::Tensor& change_offsets_i32,
    const torch::Tensor& change_record_i16,
    const torch::Tensor& site_rgba_f32,
    const torch::Tensor& target_rgb_f32,
    const torch::Tensor& config_i32,
    const torch::Tensor& config_f32) {
  check_half_mps_2d(coeff_f16, "coeff_f16", 4);
  check_float_mps_1d_any(frame_t_f32, "frame_t_f32");
  check_i32_mps_1d_any(base_offsets_i32, "base_offsets_i32");
  check_i16_mps_1d_any(base_record_i16, "base_record_i16");
  check_i32_mps_1d_any(track_change_offsets_i32, "track_change_offsets_i32");
  check_i16_mps_1d_any(track_chunk_change_offsets_i16, "track_chunk_change_offsets_i16");
  check_i32_mps_1d_any(change_frame_i32, "change_frame_i32");
  check_i32_mps_1d_any(change_offsets_i32, "change_offsets_i32");
  check_i16_mps_1d_any(change_record_i16, "change_record_i16");
  check_float_mps_2d(site_rgba_f32, "site_rgba_f32", 4);
  TORCH_CHECK(target_rgb_f32.device().is_mps(), "target_rgb_f32 must be on MPS");
  TORCH_CHECK(target_rgb_f32.scalar_type() == torch::kFloat32, "target_rgb_f32 must be float32");
  TORCH_CHECK(target_rgb_f32.dim() == 3 && target_rgb_f32.size(2) == 3, "target_rgb_f32 must have shape [K,T,3]");
  TORCH_CHECK(target_rgb_f32.is_contiguous(), "target_rgb_f32 must be contiguous");
  check_i32_mps_1d(config_i32, "config_i32", 7);
  TORCH_CHECK(config_f32.device().is_mps(), "config_f32 must be on MPS");
  TORCH_CHECK(config_f32.scalar_type() == torch::kFloat32, "config_f32 must be float32");
  TORCH_CHECK(config_f32.dim() == 1 && config_f32.size(0) == 4, "config_f32 must have shape [4]");
  TORCH_CHECK(config_f32.is_contiguous(), "config_f32 must be contiguous");

  auto config_i32_cpu = config_i32.cpu();
  const int32_t* config = config_i32_cpu.data_ptr<int32_t>();
  const int64_t boundary_count = config[0];
  const int64_t track_count = config[1];
  const int64_t frame_count = config[2];
  const int64_t site_count = config[3];
  const int64_t base_record_count = config[4];
  const int64_t change_count = config[5];
  const int64_t change_record_count = config[6];
  TORCH_CHECK(boundary_count > 0, "config_i32[0] boundary count must be positive");
  TORCH_CHECK(boundary_count <= 32765, "i16x4 framegroup16 delta replace coeff16 fused MSE supports boundary count <= 32765");
  TORCH_CHECK(track_count > 0, "config_i32[1] track count must be positive");
  TORCH_CHECK(frame_count > 0, "config_i32[2] frame count must be positive");
  TORCH_CHECK(site_count == site_rgba_f32.size(0), "config_i32[3] must match site_rgba_f32 rows");
  TORCH_CHECK(
      site_count > 0 && site_count <= 32767,
      "i16x4 framegroup16 endpoint record delta replace coeff16 fused MSE VJP supports site count in [1, 32767]");
  TORCH_CHECK(base_record_count >= 0, "config_i32[4] base record count must be nonnegative");
  TORCH_CHECK(change_count >= 0, "config_i32[5] change count must be nonnegative");
  TORCH_CHECK(change_count <= 32767, "int16 chunk-start offsets require change count <= 32767");
  TORCH_CHECK(change_record_count >= 0, "config_i32[6] change record count must be nonnegative");
  TORCH_CHECK(coeff_f16.size(0) == track_count * boundary_count, "coeff_f16 row count mismatch");
  TORCH_CHECK(frame_t_f32.size(0) == frame_count, "frame_t_f32 length must match frame count");
  TORCH_CHECK(base_record_i16.size(0) == base_record_count * 4, "base_record_i16 length mismatch");
  TORCH_CHECK(change_frame_i32.size(0) == change_count, "change_frame_i32 length must match change count");
  TORCH_CHECK(change_record_i16.size(0) == change_record_count * 4, "change_record_i16 length mismatch");
  TORCH_CHECK(base_offsets_i32.size(0) == track_count + 1, "base_offsets_i32 length must be track_count + 1");
  TORCH_CHECK(
      track_change_offsets_i32.size(0) == track_count + 1,
      "track_change_offsets_i32 length must be track_count + 1");
  TORCH_CHECK(change_offsets_i32.size(0) == change_count + 1, "change_offsets_i32 length must be change_count + 1");
  TORCH_CHECK(
      target_rgb_f32.size(0) == track_count && target_rgb_f32.size(1) == frame_count,
      "target_rgb_f32 shape mismatch");
  check_segment_tape_offsets_cpu(base_offsets_i32, track_count, base_record_count, 129);
  check_segment_tape_offsets_cpu(track_change_offsets_i32, track_count, change_count, 2147483647);
  check_segment_tape_offsets_cpu(change_offsets_i32, change_count, change_record_count, 129);

  auto loss = torch::empty({1}, site_rgba_f32.options().dtype(torch::kFloat32));
  auto grad_site_rgba = torch::empty({site_count, 4}, site_rgba_f32.options().dtype(torch::kFloat32));
  auto& k = kernels();
  constexpr uint64_t clear_threads = 256ull;
  constexpr uint64_t framegroup_threads = 32ull;
  const int64_t chunk_count = (frame_count + int64_t(framegroup_threads) - 1) / int64_t(framegroup_threads);
  TORCH_CHECK(
      track_chunk_change_offsets_i16.size(0) == track_count * (chunk_count + 1),
      "track_chunk_change_offsets_i16 length mismatch");
  auto chunk_offsets_cpu = track_chunk_change_offsets_i16.cpu();
  auto track_offsets_cpu = track_change_offsets_i32.cpu();
  const int16_t* chunk_offsets = chunk_offsets_cpu.data_ptr<int16_t>();
  const int32_t* track_offsets = track_offsets_cpu.data_ptr<int32_t>();
  for (int64_t track_id = 0; track_id < track_count; ++track_id) {
    const int32_t track_begin = track_offsets[track_id];
    const int32_t track_end = track_offsets[track_id + 1];
    TORCH_CHECK(track_begin >= 0 && track_end >= track_begin && track_end <= change_count, "track change offset bounds");
    int32_t previous = track_begin;
    for (int64_t chunk_id = 0; chunk_id <= chunk_count; ++chunk_id) {
      const int64_t chunk_index = track_id * (chunk_count + 1) + chunk_id;
      const int32_t value = static_cast<int32_t>(chunk_offsets[chunk_index]);
      TORCH_CHECK(
          value >= previous && value >= track_begin && value <= track_end,
          "track_chunk_change_offsets_i16 must be monotonic within each track and bounded by track changes");
      previous = value;
    }
    TORCH_CHECK(previous == track_end, "final chunk change offset must match track change end");
  }
  launch(k.clear_endpoint_loss_site_rgba_grad, [&](MetalKernelFunction& fn) {
    fn.setArg(0, loss);
    fn.setArg(1, grad_site_rgba);
    fn.setArg(2, config_i32);
    fn.dispatch((uint64_t)std::max<int64_t>(site_count, 1), clear_threads);
  });
  auto framegroup_kernel = k.endpoint_record_delta_replace_coeff16_i16x4_framegroup16_mse_vjp_direct_atomic_rgb_only;
  launch(
      framegroup_kernel,
      [&](MetalKernelFunction& fn) {
        fn.setArg(0, coeff_f16);
        fn.setArg(1, frame_t_f32);
        fn.setArg(2, base_offsets_i32);
        fn.setArg(3, base_record_i16);
        fn.setArg(4, track_change_offsets_i32);
        fn.setArg(5, track_chunk_change_offsets_i16);
        fn.setArg(6, change_frame_i32);
        fn.setArg(7, change_offsets_i32);
        fn.setArg(8, change_record_i16);
        fn.setArg(9, site_rgba_f32);
        fn.setArg(10, target_rgb_f32);
        fn.setArg(11, config_i32);
        fn.setArg(12, config_f32);
        fn.setArg(13, loss);
        fn.setArg(14, grad_site_rgba);
        fn.dispatch((uint64_t)track_count * (uint64_t)chunk_count * framegroup_threads, framegroup_threads);
      });
  return std::make_tuple(loss, grad_site_rgba);
}

std::tuple<torch::Tensor, torch::Tensor> metal_endpoint_record_delta_replace_coeff16_i16x4_mse_vjp_direct_atomic_rgb_only(
    const torch::Tensor& coeff_f16,
    const torch::Tensor& frame_t_f32,
    const torch::Tensor& base_offsets_i32,
    const torch::Tensor& base_record_i16,
    const torch::Tensor& track_change_offsets_i32,
    const torch::Tensor& change_frame_i32,
    const torch::Tensor& change_offsets_i32,
    const torch::Tensor& change_record_i16,
    const torch::Tensor& site_rgba_f32,
    const torch::Tensor& target_rgb_f32,
    const torch::Tensor& config_i32,
    const torch::Tensor& config_f32) {
  check_half_mps_2d(coeff_f16, "coeff_f16", 4);
  check_float_mps_1d_any(frame_t_f32, "frame_t_f32");
  check_i32_mps_1d_any(base_offsets_i32, "base_offsets_i32");
  check_i16_mps_1d_any(base_record_i16, "base_record_i16");
  check_i32_mps_1d_any(track_change_offsets_i32, "track_change_offsets_i32");
  check_i32_mps_1d_any(change_frame_i32, "change_frame_i32");
  check_i32_mps_1d_any(change_offsets_i32, "change_offsets_i32");
  check_i16_mps_1d_any(change_record_i16, "change_record_i16");
  check_float_mps_2d(site_rgba_f32, "site_rgba_f32", 4);
  TORCH_CHECK(target_rgb_f32.device().is_mps(), "target_rgb_f32 must be on MPS");
  TORCH_CHECK(target_rgb_f32.scalar_type() == torch::kFloat32, "target_rgb_f32 must be float32");
  TORCH_CHECK(target_rgb_f32.dim() == 3 && target_rgb_f32.size(2) == 3, "target_rgb_f32 must have shape [K,T,3]");
  TORCH_CHECK(target_rgb_f32.is_contiguous(), "target_rgb_f32 must be contiguous");
  check_i32_mps_1d(config_i32, "config_i32", 7);
  TORCH_CHECK(config_f32.device().is_mps(), "config_f32 must be on MPS");
  TORCH_CHECK(config_f32.scalar_type() == torch::kFloat32, "config_f32 must be float32");
  TORCH_CHECK(config_f32.dim() == 1 && config_f32.size(0) == 4, "config_f32 must have shape [4]");
  TORCH_CHECK(config_f32.is_contiguous(), "config_f32 must be contiguous");

  auto config_i32_cpu = config_i32.cpu();
  const int32_t* config = config_i32_cpu.data_ptr<int32_t>();
  const int64_t boundary_count = config[0];
  const int64_t track_count = config[1];
  const int64_t frame_count = config[2];
  const int64_t site_count = config[3];
  const int64_t base_record_count = config[4];
  const int64_t change_count = config[5];
  const int64_t change_record_count = config[6];
  TORCH_CHECK(boundary_count > 0, "config_i32[0] boundary count must be positive");
  TORCH_CHECK(boundary_count <= 32765, "i16x4 delta replace coeff16 fused MSE supports boundary count <= 32765");
  TORCH_CHECK(track_count > 0, "config_i32[1] track count must be positive");
  TORCH_CHECK(frame_count > 0, "config_i32[2] frame count must be positive");
  TORCH_CHECK(site_count == site_rgba_f32.size(0), "config_i32[3] must match site_rgba_f32 rows");
  TORCH_CHECK(
      site_count > 0 && site_count <= 32767,
      "i16x4 endpoint record delta replace coeff16 fused MSE VJP supports site count in [1, 32767]");
  TORCH_CHECK(base_record_count >= 0, "config_i32[4] base record count must be nonnegative");
  TORCH_CHECK(change_count >= 0, "config_i32[5] change count must be nonnegative");
  TORCH_CHECK(change_record_count >= 0, "config_i32[6] change record count must be nonnegative");
  TORCH_CHECK(coeff_f16.size(0) == track_count * boundary_count, "coeff_f16 row count mismatch");
  TORCH_CHECK(frame_t_f32.size(0) == frame_count, "frame_t_f32 length must match frame count");
  TORCH_CHECK(base_record_i16.size(0) == base_record_count * 4, "base_record_i16 length mismatch");
  TORCH_CHECK(change_frame_i32.size(0) == change_count, "change_frame_i32 length must match change count");
  TORCH_CHECK(change_record_i16.size(0) == change_record_count * 4, "change_record_i16 length mismatch");
  TORCH_CHECK(base_offsets_i32.size(0) == track_count + 1, "base_offsets_i32 length must be track_count + 1");
  TORCH_CHECK(
      track_change_offsets_i32.size(0) == track_count + 1,
      "track_change_offsets_i32 length must be track_count + 1");
  TORCH_CHECK(change_offsets_i32.size(0) == change_count + 1, "change_offsets_i32 length must be change_count + 1");
  TORCH_CHECK(
      target_rgb_f32.size(0) == track_count && target_rgb_f32.size(1) == frame_count,
      "target_rgb_f32 shape mismatch");
  check_segment_tape_offsets_cpu(base_offsets_i32, track_count, base_record_count, 129);
  check_segment_tape_offsets_cpu(track_change_offsets_i32, track_count, change_count, 2147483647);
  check_segment_tape_offsets_cpu(change_offsets_i32, change_count, change_record_count, 129);

  auto loss = torch::empty({1}, site_rgba_f32.options().dtype(torch::kFloat32));
  auto grad_site_rgba = torch::empty({site_count, 4}, site_rgba_f32.options().dtype(torch::kFloat32));
  auto& k = kernels();
  constexpr uint64_t threads = 256ull;
  const int64_t sample_count = track_count * frame_count;
  launch(k.clear_endpoint_loss_site_rgba_grad, [&](MetalKernelFunction& fn) {
    fn.setArg(0, loss);
    fn.setArg(1, grad_site_rgba);
    fn.setArg(2, config_i32);
    fn.dispatch((uint64_t)std::max<int64_t>(site_count, 1), threads);
  });
  launch(k.endpoint_record_delta_replace_coeff16_i16x4_mse_vjp_direct_atomic_rgb_only, [&](MetalKernelFunction& fn) {
    fn.setArg(0, coeff_f16);
    fn.setArg(1, frame_t_f32);
    fn.setArg(2, base_offsets_i32);
    fn.setArg(3, base_record_i16);
    fn.setArg(4, track_change_offsets_i32);
    fn.setArg(5, change_frame_i32);
    fn.setArg(6, change_offsets_i32);
    fn.setArg(7, change_record_i16);
    fn.setArg(8, site_rgba_f32);
    fn.setArg(9, target_rgb_f32);
    fn.setArg(10, config_i32);
    fn.setArg(11, config_f32);
    fn.setArg(12, loss);
    fn.setArg(13, grad_site_rgba);
    fn.dispatch((uint64_t)sample_count, threads);
  });
  return std::make_tuple(loss, grad_site_rgba);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> metal_endpoint_record_edit_rgba_depth_replay(
    const torch::Tensor& boundary_f32,
    const torch::Tensor& rays_f32,
    const torch::Tensor& frame_t_f32,
    const torch::Tensor& base_offsets_i32,
    const torch::Tensor& base_owner_i32,
    const torch::Tensor& base_left_i32,
    const torch::Tensor& base_right_i32,
    const torch::Tensor& track_change_offsets_i32,
    const torch::Tensor& change_frame_i32,
    const torch::Tensor& op_offsets_i32,
    const torch::Tensor& op_type_i32,
    const torch::Tensor& op_pos_i32,
    const torch::Tensor& op_owner_i32,
    const torch::Tensor& op_left_i32,
    const torch::Tensor& op_right_i32,
    const torch::Tensor& site_rgba_f32,
    const torch::Tensor& config_i32,
    const torch::Tensor& config_f32) {
  check_float_mps_2d(boundary_f32, "boundary_f32", 5);
  TORCH_CHECK(rays_f32.device().is_mps(), "rays_f32 must be on MPS");
  TORCH_CHECK(rays_f32.scalar_type() == torch::kFloat32, "rays_f32 must be float32");
  TORCH_CHECK(rays_f32.dim() == 3 && rays_f32.size(2) == 6, "rays_f32 must have shape [K,T,6]");
  TORCH_CHECK(rays_f32.is_contiguous(), "rays_f32 must be contiguous");
  check_float_mps_1d_any(frame_t_f32, "frame_t_f32");
  check_i32_mps_1d_any(base_offsets_i32, "base_offsets_i32");
  check_i32_mps_1d_any(base_owner_i32, "base_owner_i32");
  check_i32_mps_1d_any(base_left_i32, "base_left_i32");
  check_i32_mps_1d_any(base_right_i32, "base_right_i32");
  check_i32_mps_1d_any(track_change_offsets_i32, "track_change_offsets_i32");
  check_i32_mps_1d_any(change_frame_i32, "change_frame_i32");
  check_i32_mps_1d_any(op_offsets_i32, "op_offsets_i32");
  check_i32_mps_1d_any(op_type_i32, "op_type_i32");
  check_i32_mps_1d_any(op_pos_i32, "op_pos_i32");
  check_i32_mps_1d_any(op_owner_i32, "op_owner_i32");
  check_i32_mps_1d_any(op_left_i32, "op_left_i32");
  check_i32_mps_1d_any(op_right_i32, "op_right_i32");
  check_float_mps_2d(site_rgba_f32, "site_rgba_f32", 4);
  check_i32_mps_1d(config_i32, "config_i32", 7);
  TORCH_CHECK(config_f32.device().is_mps(), "config_f32 must be on MPS");
  TORCH_CHECK(config_f32.scalar_type() == torch::kFloat32, "config_f32 must be float32");
  TORCH_CHECK(config_f32.dim() == 1 && config_f32.size(0) == 4, "config_f32 must have shape [4]");
  TORCH_CHECK(config_f32.is_contiguous(), "config_f32 must be contiguous");

  auto config_i32_cpu = config_i32.cpu();
  const int32_t* config = config_i32_cpu.data_ptr<int32_t>();
  const int64_t boundary_count = config[0];
  const int64_t track_count = config[1];
  const int64_t frame_count = config[2];
  const int64_t site_count = config[3];
  const int64_t base_record_count = config[4];
  const int64_t change_count = config[5];
  const int64_t op_count = config[6];
  TORCH_CHECK(boundary_count == boundary_f32.size(0), "config_i32[0] must match boundary_f32 rows");
  TORCH_CHECK(boundary_count > 0, "config_i32[0] boundary count must be positive");
  TORCH_CHECK(track_count > 0, "config_i32[1] track count must be positive");
  TORCH_CHECK(frame_count > 0, "config_i32[2] frame count must be positive");
  TORCH_CHECK(site_count == site_rgba_f32.size(0), "config_i32[3] must match site_rgba_f32 rows");
  TORCH_CHECK(site_count > 0 && site_count <= 64, "endpoint record edit replay supports site count in [1, 64]");
  TORCH_CHECK(base_record_count >= 0, "config_i32[4] base record count must be nonnegative");
  TORCH_CHECK(change_count >= 0, "config_i32[5] change count must be nonnegative");
  TORCH_CHECK(op_count >= 0, "config_i32[6] op count must be nonnegative");
  TORCH_CHECK(rays_f32.size(0) == track_count && rays_f32.size(1) == frame_count, "rays_f32 shape mismatch");
  TORCH_CHECK(frame_t_f32.size(0) == frame_count, "frame_t_f32 length must match frame count");
  TORCH_CHECK(base_owner_i32.size(0) == base_record_count, "base_owner_i32 length must match base record count");
  TORCH_CHECK(base_left_i32.size(0) == base_record_count, "base_left_i32 length must match base record count");
  TORCH_CHECK(base_right_i32.size(0) == base_record_count, "base_right_i32 length must match base record count");
  TORCH_CHECK(change_frame_i32.size(0) == change_count, "change_frame_i32 length must match change count");
  TORCH_CHECK(op_type_i32.size(0) == op_count, "op_type_i32 length must match op count");
  TORCH_CHECK(op_pos_i32.size(0) == op_count, "op_pos_i32 length must match op count");
  TORCH_CHECK(op_owner_i32.size(0) == op_count, "op_owner_i32 length must match op count");
  TORCH_CHECK(op_left_i32.size(0) == op_count, "op_left_i32 length must match op count");
  TORCH_CHECK(op_right_i32.size(0) == op_count, "op_right_i32 length must match op count");
  TORCH_CHECK(base_offsets_i32.size(0) == track_count + 1, "base_offsets_i32 length must be track_count + 1");
  TORCH_CHECK(
      track_change_offsets_i32.size(0) == track_count + 1,
      "track_change_offsets_i32 length must be track_count + 1");
  TORCH_CHECK(op_offsets_i32.size(0) == change_count + 1, "op_offsets_i32 length must be change_count + 1");
  check_segment_tape_offsets_cpu(base_offsets_i32, track_count, base_record_count, 2147483647);
  check_segment_tape_offsets_cpu(track_change_offsets_i32, track_count, change_count, 2147483647);
  check_segment_tape_offsets_cpu(op_offsets_i32, change_count, op_count, 2147483647);

  auto output_rgb = torch::empty({track_count, frame_count, 3}, site_rgba_f32.options().dtype(torch::kFloat32));
  auto output_alpha = torch::empty({track_count, frame_count}, site_rgba_f32.options().dtype(torch::kFloat32));
  auto output_depth = torch::empty({track_count, frame_count}, site_rgba_f32.options().dtype(torch::kFloat32));
  auto& k = kernels();
  constexpr uint64_t threads = 256ull;
  const int64_t sample_count = track_count * frame_count;
  launch(k.endpoint_record_edit_rgba_depth_replay, [&](MetalKernelFunction& fn) {
    fn.setArg(0, boundary_f32);
    fn.setArg(1, rays_f32);
    fn.setArg(2, frame_t_f32);
    fn.setArg(3, base_offsets_i32);
    fn.setArg(4, base_owner_i32);
    fn.setArg(5, base_left_i32);
    fn.setArg(6, base_right_i32);
    fn.setArg(7, track_change_offsets_i32);
    fn.setArg(8, change_frame_i32);
    fn.setArg(9, op_offsets_i32);
    fn.setArg(10, op_type_i32);
    fn.setArg(11, op_pos_i32);
    fn.setArg(12, op_owner_i32);
    fn.setArg(13, op_left_i32);
    fn.setArg(14, op_right_i32);
    fn.setArg(15, site_rgba_f32);
    fn.setArg(16, config_i32);
    fn.setArg(17, config_f32);
    fn.setArg(18, output_rgb);
    fn.setArg(19, output_alpha);
    fn.setArg(20, output_depth);
    fn.dispatch((uint64_t)sample_count, threads);
  });
  return std::make_tuple(output_rgb, output_alpha, output_depth);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> metal_endpoint_record_edit_block4_rgba_depth_replay(
    const torch::Tensor& boundary_f32,
    const torch::Tensor& rays_f32,
    const torch::Tensor& frame_t_f32,
    const torch::Tensor& anchor_offsets_i32,
    const torch::Tensor& anchor_owner_i32,
    const torch::Tensor& anchor_left_i32,
    const torch::Tensor& anchor_right_i32,
    const torch::Tensor& track_block_change_offsets_i32,
    const torch::Tensor& change_frame_i32,
    const torch::Tensor& op_offsets_i32,
    const torch::Tensor& op_type_i32,
    const torch::Tensor& op_pos_i32,
    const torch::Tensor& op_owner_i32,
    const torch::Tensor& op_left_i32,
    const torch::Tensor& op_right_i32,
    const torch::Tensor& site_rgba_f32,
    const torch::Tensor& config_i32,
    const torch::Tensor& config_f32) {
  check_float_mps_2d(boundary_f32, "boundary_f32", 5);
  TORCH_CHECK(rays_f32.device().is_mps(), "rays_f32 must be on MPS");
  TORCH_CHECK(rays_f32.scalar_type() == torch::kFloat32, "rays_f32 must be float32");
  TORCH_CHECK(rays_f32.dim() == 3 && rays_f32.size(2) == 6, "rays_f32 must have shape [K,T,6]");
  TORCH_CHECK(rays_f32.is_contiguous(), "rays_f32 must be contiguous");
  check_float_mps_1d_any(frame_t_f32, "frame_t_f32");
  check_i32_mps_1d_any(anchor_offsets_i32, "anchor_offsets_i32");
  check_i32_mps_1d_any(anchor_owner_i32, "anchor_owner_i32");
  check_i32_mps_1d_any(anchor_left_i32, "anchor_left_i32");
  check_i32_mps_1d_any(anchor_right_i32, "anchor_right_i32");
  check_i32_mps_1d_any(track_block_change_offsets_i32, "track_block_change_offsets_i32");
  check_i32_mps_1d_any(change_frame_i32, "change_frame_i32");
  check_i32_mps_1d_any(op_offsets_i32, "op_offsets_i32");
  check_i32_mps_1d_any(op_type_i32, "op_type_i32");
  check_i32_mps_1d_any(op_pos_i32, "op_pos_i32");
  check_i32_mps_1d_any(op_owner_i32, "op_owner_i32");
  check_i32_mps_1d_any(op_left_i32, "op_left_i32");
  check_i32_mps_1d_any(op_right_i32, "op_right_i32");
  check_float_mps_2d(site_rgba_f32, "site_rgba_f32", 4);
  check_i32_mps_1d(config_i32, "config_i32", 9);
  TORCH_CHECK(config_f32.device().is_mps(), "config_f32 must be on MPS");
  TORCH_CHECK(config_f32.scalar_type() == torch::kFloat32, "config_f32 must be float32");
  TORCH_CHECK(config_f32.dim() == 1 && config_f32.size(0) == 4, "config_f32 must have shape [4]");
  TORCH_CHECK(config_f32.is_contiguous(), "config_f32 must be contiguous");

  auto config_i32_cpu = config_i32.cpu();
  const int32_t* config = config_i32_cpu.data_ptr<int32_t>();
  const int64_t boundary_count = config[0];
  const int64_t track_count = config[1];
  const int64_t frame_count = config[2];
  const int64_t site_count = config[3];
  const int64_t anchor_record_count = config[4];
  const int64_t change_count = config[5];
  const int64_t op_count = config[6];
  const int64_t block_size = config[7];
  const int64_t block_count = config[8];
  TORCH_CHECK(boundary_count == boundary_f32.size(0), "config_i32[0] must match boundary_f32 rows");
  TORCH_CHECK(boundary_count > 0, "config_i32[0] boundary count must be positive");
  TORCH_CHECK(track_count > 0, "config_i32[1] track count must be positive");
  TORCH_CHECK(frame_count > 0, "config_i32[2] frame count must be positive");
  TORCH_CHECK(site_count == site_rgba_f32.size(0), "config_i32[3] must match site_rgba_f32 rows");
  TORCH_CHECK(site_count > 0 && site_count <= 64, "endpoint record block replay supports site count in [1, 64]");
  TORCH_CHECK(anchor_record_count >= 0, "config_i32[4] anchor record count must be nonnegative");
  TORCH_CHECK(change_count >= 0, "config_i32[5] change count must be nonnegative");
  TORCH_CHECK(op_count >= 0, "config_i32[6] op count must be nonnegative");
  TORCH_CHECK(block_size > 0, "endpoint_record_edit_block4_rgba_depth_replay requires positive block size");
  TORCH_CHECK(block_count == (frame_count + block_size - 1) / block_size, "config_i32[8] block count mismatch");
  TORCH_CHECK(rays_f32.size(0) == track_count && rays_f32.size(1) == frame_count, "rays_f32 shape mismatch");
  TORCH_CHECK(frame_t_f32.size(0) == frame_count, "frame_t_f32 length must match frame count");
  TORCH_CHECK(anchor_owner_i32.size(0) == anchor_record_count, "anchor_owner_i32 length must match anchor record count");
  TORCH_CHECK(anchor_left_i32.size(0) == anchor_record_count, "anchor_left_i32 length must match anchor record count");
  TORCH_CHECK(anchor_right_i32.size(0) == anchor_record_count, "anchor_right_i32 length must match anchor record count");
  TORCH_CHECK(change_frame_i32.size(0) == change_count, "change_frame_i32 length must match change count");
  TORCH_CHECK(op_type_i32.size(0) == op_count, "op_type_i32 length must match op count");
  TORCH_CHECK(op_pos_i32.size(0) == op_count, "op_pos_i32 length must match op count");
  TORCH_CHECK(op_owner_i32.size(0) == op_count, "op_owner_i32 length must match op count");
  TORCH_CHECK(op_left_i32.size(0) == op_count, "op_left_i32 length must match op count");
  TORCH_CHECK(op_right_i32.size(0) == op_count, "op_right_i32 length must match op count");
  TORCH_CHECK(anchor_offsets_i32.size(0) == track_count * block_count + 1, "anchor_offsets_i32 length mismatch");
  TORCH_CHECK(
      track_block_change_offsets_i32.size(0) == track_count * (block_count + 1),
      "track_block_change_offsets_i32 length mismatch");
  TORCH_CHECK(op_offsets_i32.size(0) == change_count + 1, "op_offsets_i32 length must be change_count + 1");
  check_segment_tape_offsets_cpu(anchor_offsets_i32, track_count * block_count, anchor_record_count, 2147483647);
  check_segment_tape_offsets_cpu(
      track_block_change_offsets_i32, track_count * (block_count + 1) - 1, change_count, 2147483647);
  check_segment_tape_offsets_cpu(op_offsets_i32, change_count, op_count, 2147483647);

  auto output_rgb = torch::empty({track_count, frame_count, 3}, site_rgba_f32.options().dtype(torch::kFloat32));
  auto output_alpha = torch::empty({track_count, frame_count}, site_rgba_f32.options().dtype(torch::kFloat32));
  auto output_depth = torch::empty({track_count, frame_count}, site_rgba_f32.options().dtype(torch::kFloat32));
  auto& k = kernels();
  constexpr uint64_t threads = 256ull;
  const int64_t sample_count = track_count * frame_count;
  launch(k.endpoint_record_edit_block4_rgba_depth_replay, [&](MetalKernelFunction& fn) {
    fn.setArg(0, boundary_f32);
    fn.setArg(1, rays_f32);
    fn.setArg(2, frame_t_f32);
    fn.setArg(3, anchor_offsets_i32);
    fn.setArg(4, anchor_owner_i32);
    fn.setArg(5, anchor_left_i32);
    fn.setArg(6, anchor_right_i32);
    fn.setArg(7, track_block_change_offsets_i32);
    fn.setArg(8, change_frame_i32);
    fn.setArg(9, op_offsets_i32);
    fn.setArg(10, op_type_i32);
    fn.setArg(11, op_pos_i32);
    fn.setArg(12, op_owner_i32);
    fn.setArg(13, op_left_i32);
    fn.setArg(14, op_right_i32);
    fn.setArg(15, site_rgba_f32);
    fn.setArg(16, config_i32);
    fn.setArg(17, config_f32);
    fn.setArg(18, output_rgb);
    fn.setArg(19, output_alpha);
    fn.setArg(20, output_depth);
    fn.dispatch((uint64_t)sample_count, threads);
  });
  return std::make_tuple(output_rgb, output_alpha, output_depth);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> metal_endpoint_record_edit_block_coeff_rgba_depth_replay(
    const torch::Tensor& coeff_f32,
    const torch::Tensor& frame_t_f32,
    const torch::Tensor& anchor_offsets_i32,
    const torch::Tensor& anchor_owner_i32,
    const torch::Tensor& anchor_left_i32,
    const torch::Tensor& anchor_right_i32,
    const torch::Tensor& track_block_change_offsets_i32,
    const torch::Tensor& change_frame_i32,
    const torch::Tensor& op_offsets_i32,
    const torch::Tensor& op_type_i32,
    const torch::Tensor& op_pos_i32,
    const torch::Tensor& op_owner_i32,
    const torch::Tensor& op_left_i32,
    const torch::Tensor& op_right_i32,
    const torch::Tensor& site_rgba_f32,
    const torch::Tensor& config_i32,
    const torch::Tensor& config_f32) {
  check_float_mps_2d(coeff_f32, "coeff_f32", 4);
  check_float_mps_1d_any(frame_t_f32, "frame_t_f32");
  check_i32_mps_1d_any(anchor_offsets_i32, "anchor_offsets_i32");
  check_i32_mps_1d_any(anchor_owner_i32, "anchor_owner_i32");
  check_i32_mps_1d_any(anchor_left_i32, "anchor_left_i32");
  check_i32_mps_1d_any(anchor_right_i32, "anchor_right_i32");
  check_i32_mps_1d_any(track_block_change_offsets_i32, "track_block_change_offsets_i32");
  check_i32_mps_1d_any(change_frame_i32, "change_frame_i32");
  check_i32_mps_1d_any(op_offsets_i32, "op_offsets_i32");
  check_i32_mps_1d_any(op_type_i32, "op_type_i32");
  check_i32_mps_1d_any(op_pos_i32, "op_pos_i32");
  check_i32_mps_1d_any(op_owner_i32, "op_owner_i32");
  check_i32_mps_1d_any(op_left_i32, "op_left_i32");
  check_i32_mps_1d_any(op_right_i32, "op_right_i32");
  check_float_mps_2d(site_rgba_f32, "site_rgba_f32", 4);
  check_i32_mps_1d(config_i32, "config_i32", 9);
  TORCH_CHECK(config_f32.device().is_mps(), "config_f32 must be on MPS");
  TORCH_CHECK(config_f32.scalar_type() == torch::kFloat32, "config_f32 must be float32");
  TORCH_CHECK(config_f32.dim() == 1 && config_f32.size(0) == 4, "config_f32 must have shape [4]");
  TORCH_CHECK(config_f32.is_contiguous(), "config_f32 must be contiguous");

  auto config_i32_cpu = config_i32.cpu();
  const int32_t* config = config_i32_cpu.data_ptr<int32_t>();
  const int64_t boundary_count = config[0];
  const int64_t track_count = config[1];
  const int64_t frame_count = config[2];
  const int64_t site_count = config[3];
  const int64_t anchor_record_count = config[4];
  const int64_t change_count = config[5];
  const int64_t op_count = config[6];
  const int64_t block_size = config[7];
  const int64_t block_count = config[8];
  TORCH_CHECK(boundary_count > 0, "config_i32[0] boundary count must be positive");
  TORCH_CHECK(track_count > 0, "config_i32[1] track count must be positive");
  TORCH_CHECK(frame_count > 0, "config_i32[2] frame count must be positive");
  TORCH_CHECK(site_count == site_rgba_f32.size(0), "config_i32[3] must match site_rgba_f32 rows");
  TORCH_CHECK(site_count > 0 && site_count <= 64, "endpoint record block coeff replay supports site count in [1, 64]");
  TORCH_CHECK(anchor_record_count >= 0, "config_i32[4] anchor record count must be nonnegative");
  TORCH_CHECK(change_count >= 0, "config_i32[5] change count must be nonnegative");
  TORCH_CHECK(op_count >= 0, "config_i32[6] op count must be nonnegative");
  TORCH_CHECK(block_size > 0, "endpoint_record_edit_block_coeff_rgba_depth_replay requires positive block size");
  TORCH_CHECK(block_count == (frame_count + block_size - 1) / block_size, "config_i32[8] block count mismatch");
  TORCH_CHECK(coeff_f32.size(0) == track_count * boundary_count, "coeff_f32 row count mismatch");
  TORCH_CHECK(frame_t_f32.size(0) == frame_count, "frame_t_f32 length must match frame count");
  TORCH_CHECK(anchor_owner_i32.size(0) == anchor_record_count, "anchor_owner_i32 length must match anchor record count");
  TORCH_CHECK(anchor_left_i32.size(0) == anchor_record_count, "anchor_left_i32 length must match anchor record count");
  TORCH_CHECK(anchor_right_i32.size(0) == anchor_record_count, "anchor_right_i32 length must match anchor record count");
  TORCH_CHECK(change_frame_i32.size(0) == change_count, "change_frame_i32 length must match change count");
  TORCH_CHECK(op_type_i32.size(0) == op_count, "op_type_i32 length must match op count");
  TORCH_CHECK(op_pos_i32.size(0) == op_count, "op_pos_i32 length must match op count");
  TORCH_CHECK(op_owner_i32.size(0) == op_count, "op_owner_i32 length must match op count");
  TORCH_CHECK(op_left_i32.size(0) == op_count, "op_left_i32 length must match op count");
  TORCH_CHECK(op_right_i32.size(0) == op_count, "op_right_i32 length must match op count");
  TORCH_CHECK(anchor_offsets_i32.size(0) == track_count * block_count + 1, "anchor_offsets_i32 length mismatch");
  TORCH_CHECK(
      track_block_change_offsets_i32.size(0) == track_count * (block_count + 1),
      "track_block_change_offsets_i32 length mismatch");
  TORCH_CHECK(op_offsets_i32.size(0) == change_count + 1, "op_offsets_i32 length must be change_count + 1");
  check_segment_tape_offsets_cpu(anchor_offsets_i32, track_count * block_count, anchor_record_count, 2147483647);
  check_segment_tape_offsets_cpu(
      track_block_change_offsets_i32, track_count * (block_count + 1) - 1, change_count, 2147483647);
  check_segment_tape_offsets_cpu(op_offsets_i32, change_count, op_count, 2147483647);

  auto output_rgb = torch::empty({track_count, frame_count, 3}, site_rgba_f32.options().dtype(torch::kFloat32));
  auto output_alpha = torch::empty({track_count, frame_count}, site_rgba_f32.options().dtype(torch::kFloat32));
  auto output_depth = torch::empty({track_count, frame_count}, site_rgba_f32.options().dtype(torch::kFloat32));
  auto& k = kernels();
  constexpr uint64_t threads = 256ull;
  const int64_t sample_count = track_count * frame_count;
  launch(k.endpoint_record_edit_block_coeff_rgba_depth_replay, [&](MetalKernelFunction& fn) {
    fn.setArg(0, coeff_f32);
    fn.setArg(1, frame_t_f32);
    fn.setArg(2, anchor_offsets_i32);
    fn.setArg(3, anchor_owner_i32);
    fn.setArg(4, anchor_left_i32);
    fn.setArg(5, anchor_right_i32);
    fn.setArg(6, track_block_change_offsets_i32);
    fn.setArg(7, change_frame_i32);
    fn.setArg(8, op_offsets_i32);
    fn.setArg(9, op_type_i32);
    fn.setArg(10, op_pos_i32);
    fn.setArg(11, op_owner_i32);
    fn.setArg(12, op_left_i32);
    fn.setArg(13, op_right_i32);
    fn.setArg(14, site_rgba_f32);
    fn.setArg(15, config_i32);
    fn.setArg(16, config_f32);
    fn.setArg(17, output_rgb);
    fn.setArg(18, output_alpha);
    fn.setArg(19, output_depth);
    fn.dispatch((uint64_t)sample_count, threads);
  });
  return std::make_tuple(output_rgb, output_alpha, output_depth);
}

torch::Tensor metal_endpoint_record_edit_block_coeff_rgb_replay(
    const torch::Tensor& coeff_f32,
    const torch::Tensor& frame_t_f32,
    const torch::Tensor& anchor_offsets_i32,
    const torch::Tensor& anchor_owner_i32,
    const torch::Tensor& anchor_left_i32,
    const torch::Tensor& anchor_right_i32,
    const torch::Tensor& track_block_change_offsets_i32,
    const torch::Tensor& change_frame_i32,
    const torch::Tensor& op_offsets_i32,
    const torch::Tensor& op_type_i32,
    const torch::Tensor& op_pos_i32,
    const torch::Tensor& op_owner_i32,
    const torch::Tensor& op_left_i32,
    const torch::Tensor& op_right_i32,
    const torch::Tensor& site_rgba_f32,
    const torch::Tensor& config_i32,
    const torch::Tensor& config_f32) {
  check_float_mps_2d(coeff_f32, "coeff_f32", 4);
  check_float_mps_1d_any(frame_t_f32, "frame_t_f32");
  check_i32_mps_1d_any(anchor_offsets_i32, "anchor_offsets_i32");
  check_i32_mps_1d_any(anchor_owner_i32, "anchor_owner_i32");
  check_i32_mps_1d_any(anchor_left_i32, "anchor_left_i32");
  check_i32_mps_1d_any(anchor_right_i32, "anchor_right_i32");
  check_i32_mps_1d_any(track_block_change_offsets_i32, "track_block_change_offsets_i32");
  check_i32_mps_1d_any(change_frame_i32, "change_frame_i32");
  check_i32_mps_1d_any(op_offsets_i32, "op_offsets_i32");
  check_i32_mps_1d_any(op_type_i32, "op_type_i32");
  check_i32_mps_1d_any(op_pos_i32, "op_pos_i32");
  check_i32_mps_1d_any(op_owner_i32, "op_owner_i32");
  check_i32_mps_1d_any(op_left_i32, "op_left_i32");
  check_i32_mps_1d_any(op_right_i32, "op_right_i32");
  check_float_mps_2d(site_rgba_f32, "site_rgba_f32", 4);
  check_i32_mps_1d(config_i32, "config_i32", 9);
  TORCH_CHECK(config_f32.device().is_mps(), "config_f32 must be on MPS");
  TORCH_CHECK(config_f32.scalar_type() == torch::kFloat32, "config_f32 must be float32");
  TORCH_CHECK(config_f32.dim() == 1 && config_f32.size(0) == 4, "config_f32 must have shape [4]");
  TORCH_CHECK(config_f32.is_contiguous(), "config_f32 must be contiguous");

  auto config_i32_cpu = config_i32.cpu();
  const int32_t* config = config_i32_cpu.data_ptr<int32_t>();
  const int64_t boundary_count = config[0];
  const int64_t track_count = config[1];
  const int64_t frame_count = config[2];
  const int64_t site_count = config[3];
  const int64_t anchor_record_count = config[4];
  const int64_t change_count = config[5];
  const int64_t op_count = config[6];
  const int64_t block_size = config[7];
  const int64_t block_count = config[8];
  TORCH_CHECK(boundary_count > 0, "config_i32[0] boundary count must be positive");
  TORCH_CHECK(track_count > 0, "config_i32[1] track count must be positive");
  TORCH_CHECK(frame_count > 0, "config_i32[2] frame count must be positive");
  TORCH_CHECK(site_count == site_rgba_f32.size(0), "config_i32[3] must match site_rgba_f32 rows");
  TORCH_CHECK(site_count > 0 && site_count <= 64, "endpoint record block coeff RGB replay supports site count in [1, 64]");
  TORCH_CHECK(anchor_record_count >= 0, "config_i32[4] anchor record count must be nonnegative");
  TORCH_CHECK(change_count >= 0, "config_i32[5] change count must be nonnegative");
  TORCH_CHECK(op_count >= 0, "config_i32[6] op count must be nonnegative");
  TORCH_CHECK(block_size > 0, "endpoint_record_edit_block_coeff_rgb_replay requires positive block size");
  TORCH_CHECK(block_count == (frame_count + block_size - 1) / block_size, "config_i32[8] block count mismatch");
  TORCH_CHECK(coeff_f32.size(0) == track_count * boundary_count, "coeff_f32 row count mismatch");
  TORCH_CHECK(frame_t_f32.size(0) == frame_count, "frame_t_f32 length must match frame count");
  TORCH_CHECK(anchor_owner_i32.size(0) == anchor_record_count, "anchor_owner_i32 length must match anchor record count");
  TORCH_CHECK(anchor_left_i32.size(0) == anchor_record_count, "anchor_left_i32 length must match anchor record count");
  TORCH_CHECK(anchor_right_i32.size(0) == anchor_record_count, "anchor_right_i32 length must match anchor record count");
  TORCH_CHECK(change_frame_i32.size(0) == change_count, "change_frame_i32 length must match change count");
  TORCH_CHECK(op_type_i32.size(0) == op_count, "op_type_i32 length must match op count");
  TORCH_CHECK(op_pos_i32.size(0) == op_count, "op_pos_i32 length must match op count");
  TORCH_CHECK(op_owner_i32.size(0) == op_count, "op_owner_i32 length must match op count");
  TORCH_CHECK(op_left_i32.size(0) == op_count, "op_left_i32 length must match op count");
  TORCH_CHECK(op_right_i32.size(0) == op_count, "op_right_i32 length must match op count");
  TORCH_CHECK(anchor_offsets_i32.size(0) == track_count * block_count + 1, "anchor_offsets_i32 length mismatch");
  TORCH_CHECK(
      track_block_change_offsets_i32.size(0) == track_count * (block_count + 1),
      "track_block_change_offsets_i32 length mismatch");
  TORCH_CHECK(op_offsets_i32.size(0) == change_count + 1, "op_offsets_i32 length must be change_count + 1");
  check_segment_tape_offsets_cpu(anchor_offsets_i32, track_count * block_count, anchor_record_count, 2147483647);
  check_segment_tape_offsets_cpu(
      track_block_change_offsets_i32, track_count * (block_count + 1) - 1, change_count, 2147483647);
  check_segment_tape_offsets_cpu(op_offsets_i32, change_count, op_count, 2147483647);

  auto output_rgb = torch::empty({track_count, frame_count, 3}, site_rgba_f32.options().dtype(torch::kFloat32));
  auto& k = kernels();
  constexpr uint64_t threads = 256ull;
  const int64_t sample_count = track_count * frame_count;
  launch(k.endpoint_record_edit_block_coeff_rgb_replay, [&](MetalKernelFunction& fn) {
    fn.setArg(0, coeff_f32);
    fn.setArg(1, frame_t_f32);
    fn.setArg(2, anchor_offsets_i32);
    fn.setArg(3, anchor_owner_i32);
    fn.setArg(4, anchor_left_i32);
    fn.setArg(5, anchor_right_i32);
    fn.setArg(6, track_block_change_offsets_i32);
    fn.setArg(7, change_frame_i32);
    fn.setArg(8, op_offsets_i32);
    fn.setArg(9, op_type_i32);
    fn.setArg(10, op_pos_i32);
    fn.setArg(11, op_owner_i32);
    fn.setArg(12, op_left_i32);
    fn.setArg(13, op_right_i32);
    fn.setArg(14, site_rgba_f32);
    fn.setArg(15, config_i32);
    fn.setArg(16, config_f32);
    fn.setArg(17, output_rgb);
    fn.dispatch((uint64_t)sample_count, threads);
  });
  return output_rgb;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> metal_endpoint_record_edit_block_coeff16_rgba_depth_replay(
    const torch::Tensor& coeff_f16,
    const torch::Tensor& frame_t_f32,
    const torch::Tensor& anchor_offsets_i32,
    const torch::Tensor& anchor_owner_i32,
    const torch::Tensor& anchor_left_i32,
    const torch::Tensor& anchor_right_i32,
    const torch::Tensor& track_block_change_offsets_i32,
    const torch::Tensor& change_frame_i32,
    const torch::Tensor& op_offsets_i32,
    const torch::Tensor& op_type_i32,
    const torch::Tensor& op_pos_i32,
    const torch::Tensor& op_owner_i32,
    const torch::Tensor& op_left_i32,
    const torch::Tensor& op_right_i32,
    const torch::Tensor& site_rgba_f32,
    const torch::Tensor& config_i32,
    const torch::Tensor& config_f32) {
  check_half_mps_2d(coeff_f16, "coeff_f16", 4);
  check_float_mps_1d_any(frame_t_f32, "frame_t_f32");
  check_i32_mps_1d_any(anchor_offsets_i32, "anchor_offsets_i32");
  check_i32_mps_1d_any(anchor_owner_i32, "anchor_owner_i32");
  check_i32_mps_1d_any(anchor_left_i32, "anchor_left_i32");
  check_i32_mps_1d_any(anchor_right_i32, "anchor_right_i32");
  check_i32_mps_1d_any(track_block_change_offsets_i32, "track_block_change_offsets_i32");
  check_i32_mps_1d_any(change_frame_i32, "change_frame_i32");
  check_i32_mps_1d_any(op_offsets_i32, "op_offsets_i32");
  check_i32_mps_1d_any(op_type_i32, "op_type_i32");
  check_i32_mps_1d_any(op_pos_i32, "op_pos_i32");
  check_i32_mps_1d_any(op_owner_i32, "op_owner_i32");
  check_i32_mps_1d_any(op_left_i32, "op_left_i32");
  check_i32_mps_1d_any(op_right_i32, "op_right_i32");
  check_float_mps_2d(site_rgba_f32, "site_rgba_f32", 4);
  check_i32_mps_1d(config_i32, "config_i32", 9);
  TORCH_CHECK(config_f32.device().is_mps(), "config_f32 must be on MPS");
  TORCH_CHECK(config_f32.scalar_type() == torch::kFloat32, "config_f32 must be float32");
  TORCH_CHECK(config_f32.dim() == 1 && config_f32.size(0) == 4, "config_f32 must have shape [4]");
  TORCH_CHECK(config_f32.is_contiguous(), "config_f32 must be contiguous");

  auto config_i32_cpu = config_i32.cpu();
  const int32_t* config = config_i32_cpu.data_ptr<int32_t>();
  const int64_t boundary_count = config[0];
  const int64_t track_count = config[1];
  const int64_t frame_count = config[2];
  const int64_t site_count = config[3];
  const int64_t anchor_record_count = config[4];
  const int64_t change_count = config[5];
  const int64_t op_count = config[6];
  const int64_t block_size = config[7];
  const int64_t block_count = config[8];
  TORCH_CHECK(boundary_count > 0, "config_i32[0] boundary count must be positive");
  TORCH_CHECK(track_count > 0, "config_i32[1] track count must be positive");
  TORCH_CHECK(frame_count > 0, "config_i32[2] frame count must be positive");
  TORCH_CHECK(site_count == site_rgba_f32.size(0), "config_i32[3] must match site_rgba_f32 rows");
  TORCH_CHECK(site_count > 0 && site_count <= 64, "endpoint record block coeff16 replay supports site count in [1, 64]");
  TORCH_CHECK(anchor_record_count >= 0, "config_i32[4] anchor record count must be nonnegative");
  TORCH_CHECK(change_count >= 0, "config_i32[5] change count must be nonnegative");
  TORCH_CHECK(op_count >= 0, "config_i32[6] op count must be nonnegative");
  TORCH_CHECK(block_size > 0, "endpoint_record_edit_block_coeff16_rgba_depth_replay requires positive block size");
  TORCH_CHECK(block_count == (frame_count + block_size - 1) / block_size, "config_i32[8] block count mismatch");
  TORCH_CHECK(coeff_f16.size(0) == track_count * boundary_count, "coeff_f16 row count mismatch");
  TORCH_CHECK(frame_t_f32.size(0) == frame_count, "frame_t_f32 length must match frame count");
  TORCH_CHECK(anchor_owner_i32.size(0) == anchor_record_count, "anchor_owner_i32 length must match anchor record count");
  TORCH_CHECK(anchor_left_i32.size(0) == anchor_record_count, "anchor_left_i32 length must match anchor record count");
  TORCH_CHECK(anchor_right_i32.size(0) == anchor_record_count, "anchor_right_i32 length must match anchor record count");
  TORCH_CHECK(change_frame_i32.size(0) == change_count, "change_frame_i32 length must match change count");
  TORCH_CHECK(op_type_i32.size(0) == op_count, "op_type_i32 length must match op count");
  TORCH_CHECK(op_pos_i32.size(0) == op_count, "op_pos_i32 length must match op count");
  TORCH_CHECK(op_owner_i32.size(0) == op_count, "op_owner_i32 length must match op count");
  TORCH_CHECK(op_left_i32.size(0) == op_count, "op_left_i32 length must match op count");
  TORCH_CHECK(op_right_i32.size(0) == op_count, "op_right_i32 length must match op count");
  TORCH_CHECK(anchor_offsets_i32.size(0) == track_count * block_count + 1, "anchor_offsets_i32 length mismatch");
  TORCH_CHECK(
      track_block_change_offsets_i32.size(0) == track_count * (block_count + 1),
      "track_block_change_offsets_i32 length mismatch");
  TORCH_CHECK(op_offsets_i32.size(0) == change_count + 1, "op_offsets_i32 length must be change_count + 1");
  check_segment_tape_offsets_cpu(anchor_offsets_i32, track_count * block_count, anchor_record_count, 2147483647);
  check_segment_tape_offsets_cpu(
      track_block_change_offsets_i32, track_count * (block_count + 1) - 1, change_count, 2147483647);
  check_segment_tape_offsets_cpu(op_offsets_i32, change_count, op_count, 2147483647);

  auto output_rgb = torch::empty({track_count, frame_count, 3}, site_rgba_f32.options().dtype(torch::kFloat32));
  auto output_alpha = torch::empty({track_count, frame_count}, site_rgba_f32.options().dtype(torch::kFloat32));
  auto output_depth = torch::empty({track_count, frame_count}, site_rgba_f32.options().dtype(torch::kFloat32));
  auto& k = kernels();
  constexpr uint64_t threads = 256ull;
  const int64_t sample_count = track_count * frame_count;
  launch(k.endpoint_record_edit_block_coeff16_rgba_depth_replay, [&](MetalKernelFunction& fn) {
    fn.setArg(0, coeff_f16);
    fn.setArg(1, frame_t_f32);
    fn.setArg(2, anchor_offsets_i32);
    fn.setArg(3, anchor_owner_i32);
    fn.setArg(4, anchor_left_i32);
    fn.setArg(5, anchor_right_i32);
    fn.setArg(6, track_block_change_offsets_i32);
    fn.setArg(7, change_frame_i32);
    fn.setArg(8, op_offsets_i32);
    fn.setArg(9, op_type_i32);
    fn.setArg(10, op_pos_i32);
    fn.setArg(11, op_owner_i32);
    fn.setArg(12, op_left_i32);
    fn.setArg(13, op_right_i32);
    fn.setArg(14, site_rgba_f32);
    fn.setArg(15, config_i32);
    fn.setArg(16, config_f32);
    fn.setArg(17, output_rgb);
    fn.setArg(18, output_alpha);
    fn.setArg(19, output_depth);
    fn.dispatch((uint64_t)sample_count, threads);
  });
  return std::make_tuple(output_rgb, output_alpha, output_depth);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> metal_endpoint_record_edit_rgba_depth_replay_trackloop(
    const torch::Tensor& boundary_f32,
    const torch::Tensor& rays_f32,
    const torch::Tensor& frame_t_f32,
    const torch::Tensor& base_offsets_i32,
    const torch::Tensor& base_owner_i32,
    const torch::Tensor& base_left_i32,
    const torch::Tensor& base_right_i32,
    const torch::Tensor& track_change_offsets_i32,
    const torch::Tensor& change_frame_i32,
    const torch::Tensor& op_offsets_i32,
    const torch::Tensor& op_type_i32,
    const torch::Tensor& op_pos_i32,
    const torch::Tensor& op_owner_i32,
    const torch::Tensor& op_left_i32,
    const torch::Tensor& op_right_i32,
    const torch::Tensor& site_rgba_f32,
    const torch::Tensor& config_i32,
    const torch::Tensor& config_f32) {
  check_float_mps_2d(boundary_f32, "boundary_f32", 5);
  TORCH_CHECK(rays_f32.device().is_mps(), "rays_f32 must be on MPS");
  TORCH_CHECK(rays_f32.scalar_type() == torch::kFloat32, "rays_f32 must be float32");
  TORCH_CHECK(rays_f32.dim() == 3 && rays_f32.size(2) == 6, "rays_f32 must have shape [K,T,6]");
  TORCH_CHECK(rays_f32.is_contiguous(), "rays_f32 must be contiguous");
  check_float_mps_1d_any(frame_t_f32, "frame_t_f32");
  check_i32_mps_1d_any(base_offsets_i32, "base_offsets_i32");
  check_i32_mps_1d_any(base_owner_i32, "base_owner_i32");
  check_i32_mps_1d_any(base_left_i32, "base_left_i32");
  check_i32_mps_1d_any(base_right_i32, "base_right_i32");
  check_i32_mps_1d_any(track_change_offsets_i32, "track_change_offsets_i32");
  check_i32_mps_1d_any(change_frame_i32, "change_frame_i32");
  check_i32_mps_1d_any(op_offsets_i32, "op_offsets_i32");
  check_i32_mps_1d_any(op_type_i32, "op_type_i32");
  check_i32_mps_1d_any(op_pos_i32, "op_pos_i32");
  check_i32_mps_1d_any(op_owner_i32, "op_owner_i32");
  check_i32_mps_1d_any(op_left_i32, "op_left_i32");
  check_i32_mps_1d_any(op_right_i32, "op_right_i32");
  check_float_mps_2d(site_rgba_f32, "site_rgba_f32", 4);
  check_i32_mps_1d(config_i32, "config_i32", 7);
  TORCH_CHECK(config_f32.device().is_mps(), "config_f32 must be on MPS");
  TORCH_CHECK(config_f32.scalar_type() == torch::kFloat32, "config_f32 must be float32");
  TORCH_CHECK(config_f32.dim() == 1 && config_f32.size(0) == 4, "config_f32 must have shape [4]");
  TORCH_CHECK(config_f32.is_contiguous(), "config_f32 must be contiguous");

  auto config_i32_cpu = config_i32.cpu();
  const int32_t* config = config_i32_cpu.data_ptr<int32_t>();
  const int64_t boundary_count = config[0];
  const int64_t track_count = config[1];
  const int64_t frame_count = config[2];
  const int64_t site_count = config[3];
  const int64_t base_record_count = config[4];
  const int64_t change_count = config[5];
  const int64_t op_count = config[6];
  TORCH_CHECK(boundary_count == boundary_f32.size(0), "config_i32[0] must match boundary_f32 rows");
  TORCH_CHECK(boundary_count > 0, "config_i32[0] boundary count must be positive");
  TORCH_CHECK(track_count > 0, "config_i32[1] track count must be positive");
  TORCH_CHECK(frame_count > 0, "config_i32[2] frame count must be positive");
  TORCH_CHECK(site_count == site_rgba_f32.size(0), "config_i32[3] must match site_rgba_f32 rows");
  TORCH_CHECK(site_count > 0 && site_count <= 64, "endpoint record edit replay supports site count in [1, 64]");
  TORCH_CHECK(base_record_count >= 0, "config_i32[4] base record count must be nonnegative");
  TORCH_CHECK(change_count >= 0, "config_i32[5] change count must be nonnegative");
  TORCH_CHECK(op_count >= 0, "config_i32[6] op count must be nonnegative");
  TORCH_CHECK(rays_f32.size(0) == track_count && rays_f32.size(1) == frame_count, "rays_f32 shape mismatch");
  TORCH_CHECK(frame_t_f32.size(0) == frame_count, "frame_t_f32 length must match frame count");
  TORCH_CHECK(base_owner_i32.size(0) == base_record_count, "base_owner_i32 length must match base record count");
  TORCH_CHECK(base_left_i32.size(0) == base_record_count, "base_left_i32 length must match base record count");
  TORCH_CHECK(base_right_i32.size(0) == base_record_count, "base_right_i32 length must match base record count");
  TORCH_CHECK(change_frame_i32.size(0) == change_count, "change_frame_i32 length must match change count");
  TORCH_CHECK(op_type_i32.size(0) == op_count, "op_type_i32 length must match op count");
  TORCH_CHECK(op_pos_i32.size(0) == op_count, "op_pos_i32 length must match op count");
  TORCH_CHECK(op_owner_i32.size(0) == op_count, "op_owner_i32 length must match op count");
  TORCH_CHECK(op_left_i32.size(0) == op_count, "op_left_i32 length must match op count");
  TORCH_CHECK(op_right_i32.size(0) == op_count, "op_right_i32 length must match op count");
  TORCH_CHECK(base_offsets_i32.size(0) == track_count + 1, "base_offsets_i32 length must be track_count + 1");
  TORCH_CHECK(
      track_change_offsets_i32.size(0) == track_count + 1,
      "track_change_offsets_i32 length must be track_count + 1");
  TORCH_CHECK(op_offsets_i32.size(0) == change_count + 1, "op_offsets_i32 length must be change_count + 1");
  check_segment_tape_offsets_cpu(base_offsets_i32, track_count, base_record_count, 2147483647);
  check_segment_tape_offsets_cpu(track_change_offsets_i32, track_count, change_count, 2147483647);
  check_segment_tape_offsets_cpu(op_offsets_i32, change_count, op_count, 2147483647);

  auto output_rgb = torch::empty({track_count, frame_count, 3}, site_rgba_f32.options().dtype(torch::kFloat32));
  auto output_alpha = torch::empty({track_count, frame_count}, site_rgba_f32.options().dtype(torch::kFloat32));
  auto output_depth = torch::empty({track_count, frame_count}, site_rgba_f32.options().dtype(torch::kFloat32));
  auto& k = kernels();
  constexpr uint64_t threads = 256ull;
  launch(k.endpoint_record_edit_rgba_depth_replay_trackloop, [&](MetalKernelFunction& fn) {
    fn.setArg(0, boundary_f32);
    fn.setArg(1, rays_f32);
    fn.setArg(2, frame_t_f32);
    fn.setArg(3, base_offsets_i32);
    fn.setArg(4, base_owner_i32);
    fn.setArg(5, base_left_i32);
    fn.setArg(6, base_right_i32);
    fn.setArg(7, track_change_offsets_i32);
    fn.setArg(8, change_frame_i32);
    fn.setArg(9, op_offsets_i32);
    fn.setArg(10, op_type_i32);
    fn.setArg(11, op_pos_i32);
    fn.setArg(12, op_owner_i32);
    fn.setArg(13, op_left_i32);
    fn.setArg(14, op_right_i32);
    fn.setArg(15, site_rgba_f32);
    fn.setArg(16, config_i32);
    fn.setArg(17, config_f32);
    fn.setArg(18, output_rgb);
    fn.setArg(19, output_alpha);
    fn.setArg(20, output_depth);
    fn.dispatch((uint64_t)track_count, threads);
  });
  return std::make_tuple(output_rgb, output_alpha, output_depth);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> metal_endpoint_record_edit_rgba_depth_replay_framegroup16(
    const torch::Tensor& boundary_f32,
    const torch::Tensor& rays_f32,
    const torch::Tensor& frame_t_f32,
    const torch::Tensor& base_offsets_i32,
    const torch::Tensor& base_owner_i32,
    const torch::Tensor& base_left_i32,
    const torch::Tensor& base_right_i32,
    const torch::Tensor& track_change_offsets_i32,
    const torch::Tensor& change_frame_i32,
    const torch::Tensor& op_offsets_i32,
    const torch::Tensor& op_type_i32,
    const torch::Tensor& op_pos_i32,
    const torch::Tensor& op_owner_i32,
    const torch::Tensor& op_left_i32,
    const torch::Tensor& op_right_i32,
    const torch::Tensor& site_rgba_f32,
    const torch::Tensor& config_i32,
    const torch::Tensor& config_f32) {
  check_float_mps_2d(boundary_f32, "boundary_f32", 5);
  TORCH_CHECK(rays_f32.device().is_mps(), "rays_f32 must be on MPS");
  TORCH_CHECK(rays_f32.scalar_type() == torch::kFloat32, "rays_f32 must be float32");
  TORCH_CHECK(rays_f32.dim() == 3 && rays_f32.size(2) == 6, "rays_f32 must have shape [K,T,6]");
  TORCH_CHECK(rays_f32.is_contiguous(), "rays_f32 must be contiguous");
  check_float_mps_1d_any(frame_t_f32, "frame_t_f32");
  check_i32_mps_1d_any(base_offsets_i32, "base_offsets_i32");
  check_i32_mps_1d_any(base_owner_i32, "base_owner_i32");
  check_i32_mps_1d_any(base_left_i32, "base_left_i32");
  check_i32_mps_1d_any(base_right_i32, "base_right_i32");
  check_i32_mps_1d_any(track_change_offsets_i32, "track_change_offsets_i32");
  check_i32_mps_1d_any(change_frame_i32, "change_frame_i32");
  check_i32_mps_1d_any(op_offsets_i32, "op_offsets_i32");
  check_i32_mps_1d_any(op_type_i32, "op_type_i32");
  check_i32_mps_1d_any(op_pos_i32, "op_pos_i32");
  check_i32_mps_1d_any(op_owner_i32, "op_owner_i32");
  check_i32_mps_1d_any(op_left_i32, "op_left_i32");
  check_i32_mps_1d_any(op_right_i32, "op_right_i32");
  check_float_mps_2d(site_rgba_f32, "site_rgba_f32", 4);
  check_i32_mps_1d(config_i32, "config_i32", 7);
  TORCH_CHECK(config_f32.device().is_mps(), "config_f32 must be on MPS");
  TORCH_CHECK(config_f32.scalar_type() == torch::kFloat32, "config_f32 must be float32");
  TORCH_CHECK(config_f32.dim() == 1 && config_f32.size(0) == 4, "config_f32 must have shape [4]");
  TORCH_CHECK(config_f32.is_contiguous(), "config_f32 must be contiguous");

  auto config_i32_cpu = config_i32.cpu();
  const int32_t* config = config_i32_cpu.data_ptr<int32_t>();
  const int64_t boundary_count = config[0];
  const int64_t track_count = config[1];
  const int64_t frame_count = config[2];
  const int64_t site_count = config[3];
  const int64_t base_record_count = config[4];
  const int64_t change_count = config[5];
  const int64_t op_count = config[6];
  TORCH_CHECK(boundary_count == boundary_f32.size(0), "config_i32[0] must match boundary_f32 rows");
  TORCH_CHECK(boundary_count > 0, "config_i32[0] boundary count must be positive");
  TORCH_CHECK(track_count > 0, "config_i32[1] track count must be positive");
  TORCH_CHECK(frame_count > 0 && frame_count <= 16, "framegroup16 replay supports frame count in [1, 16]");
  TORCH_CHECK(site_count == site_rgba_f32.size(0), "config_i32[3] must match site_rgba_f32 rows");
  TORCH_CHECK(site_count > 0 && site_count <= 64, "endpoint record edit replay supports site count in [1, 64]");
  TORCH_CHECK(base_record_count >= 0, "config_i32[4] base record count must be nonnegative");
  TORCH_CHECK(change_count >= 0, "config_i32[5] change count must be nonnegative");
  TORCH_CHECK(op_count >= 0, "config_i32[6] op count must be nonnegative");
  TORCH_CHECK(rays_f32.size(0) == track_count && rays_f32.size(1) == frame_count, "rays_f32 shape mismatch");
  TORCH_CHECK(frame_t_f32.size(0) == frame_count, "frame_t_f32 length must match frame count");
  TORCH_CHECK(base_owner_i32.size(0) == base_record_count, "base_owner_i32 length must match base record count");
  TORCH_CHECK(base_left_i32.size(0) == base_record_count, "base_left_i32 length must match base record count");
  TORCH_CHECK(base_right_i32.size(0) == base_record_count, "base_right_i32 length must match base record count");
  TORCH_CHECK(change_frame_i32.size(0) == change_count, "change_frame_i32 length must match change count");
  TORCH_CHECK(op_type_i32.size(0) == op_count, "op_type_i32 length must match op count");
  TORCH_CHECK(op_pos_i32.size(0) == op_count, "op_pos_i32 length must match op count");
  TORCH_CHECK(op_owner_i32.size(0) == op_count, "op_owner_i32 length must match op count");
  TORCH_CHECK(op_left_i32.size(0) == op_count, "op_left_i32 length must match op count");
  TORCH_CHECK(op_right_i32.size(0) == op_count, "op_right_i32 length must match op count");
  TORCH_CHECK(base_offsets_i32.size(0) == track_count + 1, "base_offsets_i32 length must be track_count + 1");
  TORCH_CHECK(
      track_change_offsets_i32.size(0) == track_count + 1,
      "track_change_offsets_i32 length must be track_count + 1");
  TORCH_CHECK(op_offsets_i32.size(0) == change_count + 1, "op_offsets_i32 length must be change_count + 1");
  check_segment_tape_offsets_cpu(base_offsets_i32, track_count, base_record_count, 2147483647);
  check_segment_tape_offsets_cpu(track_change_offsets_i32, track_count, change_count, 2147483647);
  check_segment_tape_offsets_cpu(op_offsets_i32, change_count, op_count, 2147483647);

  auto output_rgb = torch::empty({track_count, frame_count, 3}, site_rgba_f32.options().dtype(torch::kFloat32));
  auto output_alpha = torch::empty({track_count, frame_count}, site_rgba_f32.options().dtype(torch::kFloat32));
  auto output_depth = torch::empty({track_count, frame_count}, site_rgba_f32.options().dtype(torch::kFloat32));
  auto& k = kernels();
  constexpr uint64_t threads = 16ull;
  launch(k.endpoint_record_edit_rgba_depth_replay_framegroup16, [&](MetalKernelFunction& fn) {
    fn.setArg(0, boundary_f32);
    fn.setArg(1, rays_f32);
    fn.setArg(2, frame_t_f32);
    fn.setArg(3, base_offsets_i32);
    fn.setArg(4, base_owner_i32);
    fn.setArg(5, base_left_i32);
    fn.setArg(6, base_right_i32);
    fn.setArg(7, track_change_offsets_i32);
    fn.setArg(8, change_frame_i32);
    fn.setArg(9, op_offsets_i32);
    fn.setArg(10, op_type_i32);
    fn.setArg(11, op_pos_i32);
    fn.setArg(12, op_owner_i32);
    fn.setArg(13, op_left_i32);
    fn.setArg(14, op_right_i32);
    fn.setArg(15, site_rgba_f32);
    fn.setArg(16, config_i32);
    fn.setArg(17, config_f32);
    fn.setArg(18, output_rgb);
    fn.setArg(19, output_alpha);
    fn.setArg(20, output_depth);
    fn.dispatch((uint64_t)track_count * threads, threads);
  });
  return std::make_tuple(output_rgb, output_alpha, output_depth);
}

torch::Tensor metal_endpoint_record_edit_vjp_direct_atomic_grad_only(
    const torch::Tensor& boundary_f32,
    const torch::Tensor& rays_f32,
    const torch::Tensor& frame_t_f32,
    const torch::Tensor& base_offsets_i32,
    const torch::Tensor& base_owner_i32,
    const torch::Tensor& base_left_i32,
    const torch::Tensor& base_right_i32,
    const torch::Tensor& track_change_offsets_i32,
    const torch::Tensor& change_frame_i32,
    const torch::Tensor& op_offsets_i32,
    const torch::Tensor& op_type_i32,
    const torch::Tensor& op_pos_i32,
    const torch::Tensor& op_owner_i32,
    const torch::Tensor& op_left_i32,
    const torch::Tensor& op_right_i32,
    const torch::Tensor& site_rgba_f32,
    const torch::Tensor& grad_rgb_f32,
    const torch::Tensor& grad_alpha_f32,
    const torch::Tensor& grad_depth_f32,
    const torch::Tensor& config_i32,
    const torch::Tensor& config_f32) {
  check_float_mps_2d(boundary_f32, "boundary_f32", 5);
  TORCH_CHECK(rays_f32.device().is_mps(), "rays_f32 must be on MPS");
  TORCH_CHECK(rays_f32.scalar_type() == torch::kFloat32, "rays_f32 must be float32");
  TORCH_CHECK(rays_f32.dim() == 3 && rays_f32.size(2) == 6, "rays_f32 must have shape [K,T,6]");
  TORCH_CHECK(rays_f32.is_contiguous(), "rays_f32 must be contiguous");
  check_float_mps_1d_any(frame_t_f32, "frame_t_f32");
  check_i32_mps_1d_any(base_offsets_i32, "base_offsets_i32");
  check_i32_mps_1d_any(base_owner_i32, "base_owner_i32");
  check_i32_mps_1d_any(base_left_i32, "base_left_i32");
  check_i32_mps_1d_any(base_right_i32, "base_right_i32");
  check_i32_mps_1d_any(track_change_offsets_i32, "track_change_offsets_i32");
  check_i32_mps_1d_any(change_frame_i32, "change_frame_i32");
  check_i32_mps_1d_any(op_offsets_i32, "op_offsets_i32");
  check_i32_mps_1d_any(op_type_i32, "op_type_i32");
  check_i32_mps_1d_any(op_pos_i32, "op_pos_i32");
  check_i32_mps_1d_any(op_owner_i32, "op_owner_i32");
  check_i32_mps_1d_any(op_left_i32, "op_left_i32");
  check_i32_mps_1d_any(op_right_i32, "op_right_i32");
  check_float_mps_2d(site_rgba_f32, "site_rgba_f32", 4);
  TORCH_CHECK(grad_rgb_f32.device().is_mps(), "grad_rgb_f32 must be on MPS");
  TORCH_CHECK(grad_rgb_f32.scalar_type() == torch::kFloat32, "grad_rgb_f32 must be float32");
  TORCH_CHECK(grad_rgb_f32.dim() == 3 && grad_rgb_f32.size(2) == 3, "grad_rgb_f32 must have shape [K,T,3]");
  TORCH_CHECK(grad_rgb_f32.is_contiguous(), "grad_rgb_f32 must be contiguous");
  TORCH_CHECK(grad_alpha_f32.device().is_mps(), "grad_alpha_f32 must be on MPS");
  TORCH_CHECK(grad_alpha_f32.scalar_type() == torch::kFloat32, "grad_alpha_f32 must be float32");
  TORCH_CHECK(grad_alpha_f32.dim() == 2, "grad_alpha_f32 must have shape [K,T]");
  TORCH_CHECK(grad_alpha_f32.is_contiguous(), "grad_alpha_f32 must be contiguous");
  TORCH_CHECK(grad_depth_f32.device().is_mps(), "grad_depth_f32 must be on MPS");
  TORCH_CHECK(grad_depth_f32.scalar_type() == torch::kFloat32, "grad_depth_f32 must be float32");
  TORCH_CHECK(grad_depth_f32.dim() == 2, "grad_depth_f32 must have shape [K,T]");
  TORCH_CHECK(grad_depth_f32.is_contiguous(), "grad_depth_f32 must be contiguous");
  check_i32_mps_1d(config_i32, "config_i32", 7);
  TORCH_CHECK(config_f32.device().is_mps(), "config_f32 must be on MPS");
  TORCH_CHECK(config_f32.scalar_type() == torch::kFloat32, "config_f32 must be float32");
  TORCH_CHECK(config_f32.dim() == 1 && config_f32.size(0) == 4, "config_f32 must have shape [4]");
  TORCH_CHECK(config_f32.is_contiguous(), "config_f32 must be contiguous");

  auto config_i32_cpu = config_i32.cpu();
  const int32_t* config = config_i32_cpu.data_ptr<int32_t>();
  const int64_t boundary_count = config[0];
  const int64_t track_count = config[1];
  const int64_t frame_count = config[2];
  const int64_t site_count = config[3];
  const int64_t base_record_count = config[4];
  const int64_t change_count = config[5];
  const int64_t op_count = config[6];
  TORCH_CHECK(boundary_count == boundary_f32.size(0), "config_i32[0] must match boundary_f32 rows");
  TORCH_CHECK(boundary_count > 0, "config_i32[0] boundary count must be positive");
  TORCH_CHECK(track_count > 0, "config_i32[1] track count must be positive");
  TORCH_CHECK(frame_count > 0, "config_i32[2] frame count must be positive");
  TORCH_CHECK(site_count == site_rgba_f32.size(0), "config_i32[3] must match site_rgba_f32 rows");
  TORCH_CHECK(site_count > 0 && site_count <= 64, "endpoint record edit VJP supports site count in [1, 64]");
  TORCH_CHECK(base_record_count >= 0, "config_i32[4] base record count must be nonnegative");
  TORCH_CHECK(change_count >= 0, "config_i32[5] change count must be nonnegative");
  TORCH_CHECK(op_count >= 0, "config_i32[6] op count must be nonnegative");
  TORCH_CHECK(rays_f32.size(0) == track_count && rays_f32.size(1) == frame_count, "rays_f32 shape mismatch");
  TORCH_CHECK(frame_t_f32.size(0) == frame_count, "frame_t_f32 length must match frame count");
  TORCH_CHECK(base_owner_i32.size(0) == base_record_count, "base_owner_i32 length must match base record count");
  TORCH_CHECK(base_left_i32.size(0) == base_record_count, "base_left_i32 length must match base record count");
  TORCH_CHECK(base_right_i32.size(0) == base_record_count, "base_right_i32 length must match base record count");
  TORCH_CHECK(change_frame_i32.size(0) == change_count, "change_frame_i32 length must match change count");
  TORCH_CHECK(op_type_i32.size(0) == op_count, "op_type_i32 length must match op count");
  TORCH_CHECK(op_pos_i32.size(0) == op_count, "op_pos_i32 length must match op count");
  TORCH_CHECK(op_owner_i32.size(0) == op_count, "op_owner_i32 length must match op count");
  TORCH_CHECK(op_left_i32.size(0) == op_count, "op_left_i32 length must match op count");
  TORCH_CHECK(op_right_i32.size(0) == op_count, "op_right_i32 length must match op count");
  TORCH_CHECK(base_offsets_i32.size(0) == track_count + 1, "base_offsets_i32 length must be track_count + 1");
  TORCH_CHECK(
      track_change_offsets_i32.size(0) == track_count + 1,
      "track_change_offsets_i32 length must be track_count + 1");
  TORCH_CHECK(op_offsets_i32.size(0) == change_count + 1, "op_offsets_i32 length must be change_count + 1");
  TORCH_CHECK(
      grad_rgb_f32.size(0) == track_count && grad_rgb_f32.size(1) == frame_count,
      "grad_rgb_f32 shape mismatch");
  TORCH_CHECK(
      grad_alpha_f32.size(0) == track_count && grad_alpha_f32.size(1) == frame_count,
      "grad_alpha_f32 shape mismatch");
  TORCH_CHECK(
      grad_depth_f32.size(0) == track_count && grad_depth_f32.size(1) == frame_count,
      "grad_depth_f32 shape mismatch");
  check_segment_tape_offsets_cpu(base_offsets_i32, track_count, base_record_count, 129);
  check_segment_tape_offsets_cpu(track_change_offsets_i32, track_count, change_count, 2147483647);
  check_segment_tape_offsets_cpu(op_offsets_i32, change_count, op_count, 2147483647);

  auto grad_site_rgba = torch::zeros({site_count, 4}, site_rgba_f32.options().dtype(torch::kFloat32));
  auto& k = kernels();
  constexpr uint64_t threads = 256ull;
  const int64_t sample_count = track_count * frame_count;
  launch(k.endpoint_record_edit_vjp_direct_atomic_grad_only, [&](MetalKernelFunction& fn) {
    fn.setArg(0, boundary_f32);
    fn.setArg(1, rays_f32);
    fn.setArg(2, frame_t_f32);
    fn.setArg(3, base_offsets_i32);
    fn.setArg(4, base_owner_i32);
    fn.setArg(5, base_left_i32);
    fn.setArg(6, base_right_i32);
    fn.setArg(7, track_change_offsets_i32);
    fn.setArg(8, change_frame_i32);
    fn.setArg(9, op_offsets_i32);
    fn.setArg(10, op_type_i32);
    fn.setArg(11, op_pos_i32);
    fn.setArg(12, op_owner_i32);
    fn.setArg(13, op_left_i32);
    fn.setArg(14, op_right_i32);
    fn.setArg(15, site_rgba_f32);
    fn.setArg(16, grad_rgb_f32);
    fn.setArg(17, grad_alpha_f32);
    fn.setArg(18, grad_depth_f32);
    fn.setArg(19, config_i32);
    fn.setArg(20, config_f32);
    fn.setArg(21, grad_site_rgba);
    fn.dispatch((uint64_t)sample_count, threads);
  });
  return grad_site_rgba;
}

torch::Tensor metal_endpoint_record_edit_vjp_direct_atomic_rgb_only(
    const torch::Tensor& boundary_f32,
    const torch::Tensor& rays_f32,
    const torch::Tensor& frame_t_f32,
    const torch::Tensor& base_offsets_i32,
    const torch::Tensor& base_owner_i32,
    const torch::Tensor& base_left_i32,
    const torch::Tensor& base_right_i32,
    const torch::Tensor& track_change_offsets_i32,
    const torch::Tensor& change_frame_i32,
    const torch::Tensor& op_offsets_i32,
    const torch::Tensor& op_type_i32,
    const torch::Tensor& op_pos_i32,
    const torch::Tensor& op_owner_i32,
    const torch::Tensor& op_left_i32,
    const torch::Tensor& op_right_i32,
    const torch::Tensor& site_rgba_f32,
    const torch::Tensor& grad_rgb_f32,
    const torch::Tensor& config_i32,
    const torch::Tensor& config_f32) {
  check_float_mps_2d(boundary_f32, "boundary_f32", 5);
  TORCH_CHECK(rays_f32.device().is_mps(), "rays_f32 must be on MPS");
  TORCH_CHECK(rays_f32.scalar_type() == torch::kFloat32, "rays_f32 must be float32");
  TORCH_CHECK(rays_f32.dim() == 3 && rays_f32.size(2) == 6, "rays_f32 must have shape [K,T,6]");
  TORCH_CHECK(rays_f32.is_contiguous(), "rays_f32 must be contiguous");
  check_float_mps_1d_any(frame_t_f32, "frame_t_f32");
  check_i32_mps_1d_any(base_offsets_i32, "base_offsets_i32");
  check_i32_mps_1d_any(base_owner_i32, "base_owner_i32");
  check_i32_mps_1d_any(base_left_i32, "base_left_i32");
  check_i32_mps_1d_any(base_right_i32, "base_right_i32");
  check_i32_mps_1d_any(track_change_offsets_i32, "track_change_offsets_i32");
  check_i32_mps_1d_any(change_frame_i32, "change_frame_i32");
  check_i32_mps_1d_any(op_offsets_i32, "op_offsets_i32");
  check_i32_mps_1d_any(op_type_i32, "op_type_i32");
  check_i32_mps_1d_any(op_pos_i32, "op_pos_i32");
  check_i32_mps_1d_any(op_owner_i32, "op_owner_i32");
  check_i32_mps_1d_any(op_left_i32, "op_left_i32");
  check_i32_mps_1d_any(op_right_i32, "op_right_i32");
  check_float_mps_2d(site_rgba_f32, "site_rgba_f32", 4);
  TORCH_CHECK(grad_rgb_f32.device().is_mps(), "grad_rgb_f32 must be on MPS");
  TORCH_CHECK(grad_rgb_f32.scalar_type() == torch::kFloat32, "grad_rgb_f32 must be float32");
  TORCH_CHECK(grad_rgb_f32.dim() == 3 && grad_rgb_f32.size(2) == 3, "grad_rgb_f32 must have shape [K,T,3]");
  TORCH_CHECK(grad_rgb_f32.is_contiguous(), "grad_rgb_f32 must be contiguous");
  check_i32_mps_1d(config_i32, "config_i32", 7);
  TORCH_CHECK(config_f32.device().is_mps(), "config_f32 must be on MPS");
  TORCH_CHECK(config_f32.scalar_type() == torch::kFloat32, "config_f32 must be float32");
  TORCH_CHECK(config_f32.dim() == 1 && config_f32.size(0) == 4, "config_f32 must have shape [4]");
  TORCH_CHECK(config_f32.is_contiguous(), "config_f32 must be contiguous");

  auto config_i32_cpu = config_i32.cpu();
  const int32_t* config = config_i32_cpu.data_ptr<int32_t>();
  const int64_t boundary_count = config[0];
  const int64_t track_count = config[1];
  const int64_t frame_count = config[2];
  const int64_t site_count = config[3];
  const int64_t base_record_count = config[4];
  const int64_t change_count = config[5];
  const int64_t op_count = config[6];
  TORCH_CHECK(boundary_count == boundary_f32.size(0), "config_i32[0] must match boundary_f32 rows");
  TORCH_CHECK(boundary_count > 0, "config_i32[0] boundary count must be positive");
  TORCH_CHECK(track_count > 0, "config_i32[1] track count must be positive");
  TORCH_CHECK(frame_count > 0, "config_i32[2] frame count must be positive");
  TORCH_CHECK(site_count == site_rgba_f32.size(0), "config_i32[3] must match site_rgba_f32 rows");
  TORCH_CHECK(site_count > 0 && site_count <= 64, "endpoint record edit RGB-only VJP supports site count in [1, 64]");
  TORCH_CHECK(base_record_count >= 0, "config_i32[4] base record count must be nonnegative");
  TORCH_CHECK(change_count >= 0, "config_i32[5] change count must be nonnegative");
  TORCH_CHECK(op_count >= 0, "config_i32[6] op count must be nonnegative");
  TORCH_CHECK(rays_f32.size(0) == track_count && rays_f32.size(1) == frame_count, "rays_f32 shape mismatch");
  TORCH_CHECK(frame_t_f32.size(0) == frame_count, "frame_t_f32 length must match frame count");
  TORCH_CHECK(base_owner_i32.size(0) == base_record_count, "base_owner_i32 length must match base record count");
  TORCH_CHECK(base_left_i32.size(0) == base_record_count, "base_left_i32 length must match base record count");
  TORCH_CHECK(base_right_i32.size(0) == base_record_count, "base_right_i32 length must match base record count");
  TORCH_CHECK(change_frame_i32.size(0) == change_count, "change_frame_i32 length must match change count");
  TORCH_CHECK(op_type_i32.size(0) == op_count, "op_type_i32 length must match op count");
  TORCH_CHECK(op_pos_i32.size(0) == op_count, "op_pos_i32 length must match op count");
  TORCH_CHECK(op_owner_i32.size(0) == op_count, "op_owner_i32 length must match op count");
  TORCH_CHECK(op_left_i32.size(0) == op_count, "op_left_i32 length must match op count");
  TORCH_CHECK(op_right_i32.size(0) == op_count, "op_right_i32 length must match op count");
  TORCH_CHECK(base_offsets_i32.size(0) == track_count + 1, "base_offsets_i32 length must be track_count + 1");
  TORCH_CHECK(
      track_change_offsets_i32.size(0) == track_count + 1,
      "track_change_offsets_i32 length must be track_count + 1");
  TORCH_CHECK(op_offsets_i32.size(0) == change_count + 1, "op_offsets_i32 length must be change_count + 1");
  TORCH_CHECK(
      grad_rgb_f32.size(0) == track_count && grad_rgb_f32.size(1) == frame_count,
      "grad_rgb_f32 shape mismatch");
  check_segment_tape_offsets_cpu(base_offsets_i32, track_count, base_record_count, 129);
  check_segment_tape_offsets_cpu(track_change_offsets_i32, track_count, change_count, 2147483647);
  check_segment_tape_offsets_cpu(op_offsets_i32, change_count, op_count, 2147483647);

  auto grad_site_rgba = torch::zeros({site_count, 4}, site_rgba_f32.options().dtype(torch::kFloat32));
  auto& k = kernels();
  constexpr uint64_t threads = 256ull;
  const int64_t sample_count = track_count * frame_count;
  launch(k.endpoint_record_edit_vjp_direct_atomic_rgb_only, [&](MetalKernelFunction& fn) {
    fn.setArg(0, boundary_f32);
    fn.setArg(1, rays_f32);
    fn.setArg(2, frame_t_f32);
    fn.setArg(3, base_offsets_i32);
    fn.setArg(4, base_owner_i32);
    fn.setArg(5, base_left_i32);
    fn.setArg(6, base_right_i32);
    fn.setArg(7, track_change_offsets_i32);
    fn.setArg(8, change_frame_i32);
    fn.setArg(9, op_offsets_i32);
    fn.setArg(10, op_type_i32);
    fn.setArg(11, op_pos_i32);
    fn.setArg(12, op_owner_i32);
    fn.setArg(13, op_left_i32);
    fn.setArg(14, op_right_i32);
    fn.setArg(15, site_rgba_f32);
    fn.setArg(16, grad_rgb_f32);
    fn.setArg(17, config_i32);
    fn.setArg(18, config_f32);
    fn.setArg(19, grad_site_rgba);
    fn.dispatch((uint64_t)sample_count, threads);
  });
  return grad_site_rgba;
}

std::tuple<torch::Tensor, torch::Tensor> metal_endpoint_record_edit_mse_vjp_direct_atomic_rgb_only(
    const torch::Tensor& boundary_f32,
    const torch::Tensor& rays_f32,
    const torch::Tensor& frame_t_f32,
    const torch::Tensor& base_offsets_i32,
    const torch::Tensor& base_owner_i32,
    const torch::Tensor& base_left_i32,
    const torch::Tensor& base_right_i32,
    const torch::Tensor& track_change_offsets_i32,
    const torch::Tensor& change_frame_i32,
    const torch::Tensor& op_offsets_i32,
    const torch::Tensor& op_type_i32,
    const torch::Tensor& op_pos_i32,
    const torch::Tensor& op_owner_i32,
    const torch::Tensor& op_left_i32,
    const torch::Tensor& op_right_i32,
    const torch::Tensor& site_rgba_f32,
    const torch::Tensor& target_rgb_f32,
    const torch::Tensor& config_i32,
    const torch::Tensor& config_f32) {
  check_float_mps_2d(boundary_f32, "boundary_f32", 5);
  TORCH_CHECK(rays_f32.device().is_mps(), "rays_f32 must be on MPS");
  TORCH_CHECK(rays_f32.scalar_type() == torch::kFloat32, "rays_f32 must be float32");
  TORCH_CHECK(rays_f32.dim() == 3 && rays_f32.size(2) == 6, "rays_f32 must have shape [K,T,6]");
  TORCH_CHECK(rays_f32.is_contiguous(), "rays_f32 must be contiguous");
  check_float_mps_1d_any(frame_t_f32, "frame_t_f32");
  check_i32_mps_1d_any(base_offsets_i32, "base_offsets_i32");
  check_i32_mps_1d_any(base_owner_i32, "base_owner_i32");
  check_i32_mps_1d_any(base_left_i32, "base_left_i32");
  check_i32_mps_1d_any(base_right_i32, "base_right_i32");
  check_i32_mps_1d_any(track_change_offsets_i32, "track_change_offsets_i32");
  check_i32_mps_1d_any(change_frame_i32, "change_frame_i32");
  check_i32_mps_1d_any(op_offsets_i32, "op_offsets_i32");
  check_i32_mps_1d_any(op_type_i32, "op_type_i32");
  check_i32_mps_1d_any(op_pos_i32, "op_pos_i32");
  check_i32_mps_1d_any(op_owner_i32, "op_owner_i32");
  check_i32_mps_1d_any(op_left_i32, "op_left_i32");
  check_i32_mps_1d_any(op_right_i32, "op_right_i32");
  check_float_mps_2d(site_rgba_f32, "site_rgba_f32", 4);
  TORCH_CHECK(target_rgb_f32.device().is_mps(), "target_rgb_f32 must be on MPS");
  TORCH_CHECK(target_rgb_f32.scalar_type() == torch::kFloat32, "target_rgb_f32 must be float32");
  TORCH_CHECK(target_rgb_f32.dim() == 3 && target_rgb_f32.size(2) == 3, "target_rgb_f32 must have shape [K,T,3]");
  TORCH_CHECK(target_rgb_f32.is_contiguous(), "target_rgb_f32 must be contiguous");
  check_i32_mps_1d(config_i32, "config_i32", 7);
  TORCH_CHECK(config_f32.device().is_mps(), "config_f32 must be on MPS");
  TORCH_CHECK(config_f32.scalar_type() == torch::kFloat32, "config_f32 must be float32");
  TORCH_CHECK(config_f32.dim() == 1 && config_f32.size(0) == 4, "config_f32 must have shape [4]");
  TORCH_CHECK(config_f32.is_contiguous(), "config_f32 must be contiguous");

  auto config_i32_cpu = config_i32.cpu();
  const int32_t* config = config_i32_cpu.data_ptr<int32_t>();
  const int64_t boundary_count = config[0];
  const int64_t track_count = config[1];
  const int64_t frame_count = config[2];
  const int64_t site_count = config[3];
  const int64_t base_record_count = config[4];
  const int64_t change_count = config[5];
  const int64_t op_count = config[6];
  TORCH_CHECK(boundary_count == boundary_f32.size(0), "config_i32[0] must match boundary_f32 rows");
  TORCH_CHECK(boundary_count > 0, "config_i32[0] boundary count must be positive");
  TORCH_CHECK(track_count > 0, "config_i32[1] track count must be positive");
  TORCH_CHECK(frame_count > 0, "config_i32[2] frame count must be positive");
  TORCH_CHECK(site_count == site_rgba_f32.size(0), "config_i32[3] must match site_rgba_f32 rows");
  TORCH_CHECK(
      site_count > 0 && site_count <= 64,
      "endpoint record edit fused MSE VJP supports site count in [1, 64]");
  TORCH_CHECK(base_record_count >= 0, "config_i32[4] base record count must be nonnegative");
  TORCH_CHECK(change_count >= 0, "config_i32[5] change count must be nonnegative");
  TORCH_CHECK(op_count >= 0, "config_i32[6] op count must be nonnegative");
  TORCH_CHECK(rays_f32.size(0) == track_count && rays_f32.size(1) == frame_count, "rays_f32 shape mismatch");
  TORCH_CHECK(frame_t_f32.size(0) == frame_count, "frame_t_f32 length must match frame count");
  TORCH_CHECK(base_owner_i32.size(0) == base_record_count, "base_owner_i32 length must match base record count");
  TORCH_CHECK(base_left_i32.size(0) == base_record_count, "base_left_i32 length must match base record count");
  TORCH_CHECK(base_right_i32.size(0) == base_record_count, "base_right_i32 length must match base record count");
  TORCH_CHECK(change_frame_i32.size(0) == change_count, "change_frame_i32 length must match change count");
  TORCH_CHECK(op_type_i32.size(0) == op_count, "op_type_i32 length must match op count");
  TORCH_CHECK(op_pos_i32.size(0) == op_count, "op_pos_i32 length must match op count");
  TORCH_CHECK(op_owner_i32.size(0) == op_count, "op_owner_i32 length must match op count");
  TORCH_CHECK(op_left_i32.size(0) == op_count, "op_left_i32 length must match op count");
  TORCH_CHECK(op_right_i32.size(0) == op_count, "op_right_i32 length must match op count");
  TORCH_CHECK(base_offsets_i32.size(0) == track_count + 1, "base_offsets_i32 length must be track_count + 1");
  TORCH_CHECK(
      track_change_offsets_i32.size(0) == track_count + 1,
      "track_change_offsets_i32 length must be track_count + 1");
  TORCH_CHECK(op_offsets_i32.size(0) == change_count + 1, "op_offsets_i32 length must be change_count + 1");
  TORCH_CHECK(
      target_rgb_f32.size(0) == track_count && target_rgb_f32.size(1) == frame_count,
      "target_rgb_f32 shape mismatch");
  check_segment_tape_offsets_cpu(base_offsets_i32, track_count, base_record_count, 129);
  check_segment_tape_offsets_cpu(track_change_offsets_i32, track_count, change_count, 2147483647);
  check_segment_tape_offsets_cpu(op_offsets_i32, change_count, op_count, 2147483647);

  auto loss = torch::empty({1}, site_rgba_f32.options().dtype(torch::kFloat32));
  auto grad_site_rgba = torch::empty({site_count, 4}, site_rgba_f32.options().dtype(torch::kFloat32));
  auto& k = kernels();
  constexpr uint64_t threads = 256ull;
  const int64_t sample_count = track_count * frame_count;
  launch(k.clear_endpoint_loss_site_rgba_grad, [&](MetalKernelFunction& fn) {
    fn.setArg(0, loss);
    fn.setArg(1, grad_site_rgba);
    fn.setArg(2, config_i32);
    fn.dispatch((uint64_t)std::max<int64_t>(site_count, 1), threads);
  });
  launch(k.endpoint_record_edit_mse_vjp_direct_atomic_rgb_only, [&](MetalKernelFunction& fn) {
    fn.setArg(0, boundary_f32);
    fn.setArg(1, rays_f32);
    fn.setArg(2, frame_t_f32);
    fn.setArg(3, base_offsets_i32);
    fn.setArg(4, base_owner_i32);
    fn.setArg(5, base_left_i32);
    fn.setArg(6, base_right_i32);
    fn.setArg(7, track_change_offsets_i32);
    fn.setArg(8, change_frame_i32);
    fn.setArg(9, op_offsets_i32);
    fn.setArg(10, op_type_i32);
    fn.setArg(11, op_pos_i32);
    fn.setArg(12, op_owner_i32);
    fn.setArg(13, op_left_i32);
    fn.setArg(14, op_right_i32);
    fn.setArg(15, site_rgba_f32);
    fn.setArg(16, target_rgb_f32);
    fn.setArg(17, config_i32);
    fn.setArg(18, config_f32);
    fn.setArg(19, loss);
    fn.setArg(20, grad_site_rgba);
    fn.dispatch((uint64_t)sample_count, threads);
  });
  return std::make_tuple(loss, grad_site_rgba);
}

std::tuple<torch::Tensor, torch::Tensor> metal_endpoint_record_edit_coeff16_mse_vjp_direct_atomic_rgb_only(
    const torch::Tensor& coeff_f16,
    const torch::Tensor& frame_t_f32,
    const torch::Tensor& base_offsets_i32,
    const torch::Tensor& base_owner_i32,
    const torch::Tensor& base_left_i32,
    const torch::Tensor& base_right_i32,
    const torch::Tensor& track_change_offsets_i32,
    const torch::Tensor& change_frame_i32,
    const torch::Tensor& op_offsets_i32,
    const torch::Tensor& op_type_i32,
    const torch::Tensor& op_pos_i32,
    const torch::Tensor& op_owner_i32,
    const torch::Tensor& op_left_i32,
    const torch::Tensor& op_right_i32,
    const torch::Tensor& site_rgba_f32,
    const torch::Tensor& target_rgb_f32,
    const torch::Tensor& config_i32,
    const torch::Tensor& config_f32) {
  check_half_mps_2d(coeff_f16, "coeff_f16", 4);
  check_float_mps_1d_any(frame_t_f32, "frame_t_f32");
  check_i32_mps_1d_any(base_offsets_i32, "base_offsets_i32");
  check_i32_mps_1d_any(base_owner_i32, "base_owner_i32");
  check_i32_mps_1d_any(base_left_i32, "base_left_i32");
  check_i32_mps_1d_any(base_right_i32, "base_right_i32");
  check_i32_mps_1d_any(track_change_offsets_i32, "track_change_offsets_i32");
  check_i32_mps_1d_any(change_frame_i32, "change_frame_i32");
  check_i32_mps_1d_any(op_offsets_i32, "op_offsets_i32");
  check_i32_mps_1d_any(op_type_i32, "op_type_i32");
  check_i32_mps_1d_any(op_pos_i32, "op_pos_i32");
  check_i32_mps_1d_any(op_owner_i32, "op_owner_i32");
  check_i32_mps_1d_any(op_left_i32, "op_left_i32");
  check_i32_mps_1d_any(op_right_i32, "op_right_i32");
  check_float_mps_2d(site_rgba_f32, "site_rgba_f32", 4);
  TORCH_CHECK(target_rgb_f32.device().is_mps(), "target_rgb_f32 must be on MPS");
  TORCH_CHECK(target_rgb_f32.scalar_type() == torch::kFloat32, "target_rgb_f32 must be float32");
  TORCH_CHECK(target_rgb_f32.dim() == 3 && target_rgb_f32.size(2) == 3, "target_rgb_f32 must have shape [K,T,3]");
  TORCH_CHECK(target_rgb_f32.is_contiguous(), "target_rgb_f32 must be contiguous");
  check_i32_mps_1d(config_i32, "config_i32", 7);
  TORCH_CHECK(config_f32.device().is_mps(), "config_f32 must be on MPS");
  TORCH_CHECK(config_f32.scalar_type() == torch::kFloat32, "config_f32 must be float32");
  TORCH_CHECK(config_f32.dim() == 1 && config_f32.size(0) == 4, "config_f32 must have shape [4]");
  TORCH_CHECK(config_f32.is_contiguous(), "config_f32 must be contiguous");

  auto config_i32_cpu = config_i32.cpu();
  const int32_t* config = config_i32_cpu.data_ptr<int32_t>();
  const int64_t boundary_count = config[0];
  const int64_t track_count = config[1];
  const int64_t frame_count = config[2];
  const int64_t site_count = config[3];
  const int64_t base_record_count = config[4];
  const int64_t change_count = config[5];
  const int64_t op_count = config[6];
  TORCH_CHECK(boundary_count > 0, "config_i32[0] boundary count must be positive");
  TORCH_CHECK(track_count > 0, "config_i32[1] track count must be positive");
  TORCH_CHECK(frame_count > 0, "config_i32[2] frame count must be positive");
  TORCH_CHECK(site_count == site_rgba_f32.size(0), "config_i32[3] must match site_rgba_f32 rows");
  TORCH_CHECK(
      site_count > 0 && site_count <= 64,
      "endpoint record edit coeff16 fused MSE VJP supports site count in [1, 64]");
  TORCH_CHECK(base_record_count >= 0, "config_i32[4] base record count must be nonnegative");
  TORCH_CHECK(change_count >= 0, "config_i32[5] change count must be nonnegative");
  TORCH_CHECK(op_count >= 0, "config_i32[6] op count must be nonnegative");
  TORCH_CHECK(coeff_f16.size(0) == track_count * boundary_count, "coeff_f16 row count mismatch");
  TORCH_CHECK(frame_t_f32.size(0) == frame_count, "frame_t_f32 length must match frame count");
  TORCH_CHECK(base_owner_i32.size(0) == base_record_count, "base_owner_i32 length must match base record count");
  TORCH_CHECK(base_left_i32.size(0) == base_record_count, "base_left_i32 length must match base record count");
  TORCH_CHECK(base_right_i32.size(0) == base_record_count, "base_right_i32 length must match base record count");
  TORCH_CHECK(change_frame_i32.size(0) == change_count, "change_frame_i32 length must match change count");
  TORCH_CHECK(op_type_i32.size(0) == op_count, "op_type_i32 length must match op count");
  TORCH_CHECK(op_pos_i32.size(0) == op_count, "op_pos_i32 length must match op count");
  TORCH_CHECK(op_owner_i32.size(0) == op_count, "op_owner_i32 length must match op count");
  TORCH_CHECK(op_left_i32.size(0) == op_count, "op_left_i32 length must match op count");
  TORCH_CHECK(op_right_i32.size(0) == op_count, "op_right_i32 length must match op count");
  TORCH_CHECK(base_offsets_i32.size(0) == track_count + 1, "base_offsets_i32 length must be track_count + 1");
  TORCH_CHECK(
      track_change_offsets_i32.size(0) == track_count + 1,
      "track_change_offsets_i32 length must be track_count + 1");
  TORCH_CHECK(op_offsets_i32.size(0) == change_count + 1, "op_offsets_i32 length must be change_count + 1");
  TORCH_CHECK(
      target_rgb_f32.size(0) == track_count && target_rgb_f32.size(1) == frame_count,
      "target_rgb_f32 shape mismatch");
  check_segment_tape_offsets_cpu(base_offsets_i32, track_count, base_record_count, 129);
  check_segment_tape_offsets_cpu(track_change_offsets_i32, track_count, change_count, 2147483647);
  check_segment_tape_offsets_cpu(op_offsets_i32, change_count, op_count, 2147483647);

  auto loss = torch::empty({1}, site_rgba_f32.options().dtype(torch::kFloat32));
  auto grad_site_rgba = torch::empty({site_count, 4}, site_rgba_f32.options().dtype(torch::kFloat32));
  auto& k = kernels();
  constexpr uint64_t threads = 256ull;
  const int64_t sample_count = track_count * frame_count;
  launch(k.clear_endpoint_loss_site_rgba_grad, [&](MetalKernelFunction& fn) {
    fn.setArg(0, loss);
    fn.setArg(1, grad_site_rgba);
    fn.setArg(2, config_i32);
    fn.dispatch((uint64_t)std::max<int64_t>(site_count, 1), threads);
  });
  launch(k.endpoint_record_edit_coeff16_mse_vjp_direct_atomic_rgb_only, [&](MetalKernelFunction& fn) {
    fn.setArg(0, coeff_f16);
    fn.setArg(1, frame_t_f32);
    fn.setArg(2, base_offsets_i32);
    fn.setArg(3, base_owner_i32);
    fn.setArg(4, base_left_i32);
    fn.setArg(5, base_right_i32);
    fn.setArg(6, track_change_offsets_i32);
    fn.setArg(7, change_frame_i32);
    fn.setArg(8, op_offsets_i32);
    fn.setArg(9, op_type_i32);
    fn.setArg(10, op_pos_i32);
    fn.setArg(11, op_owner_i32);
    fn.setArg(12, op_left_i32);
    fn.setArg(13, op_right_i32);
    fn.setArg(14, site_rgba_f32);
    fn.setArg(15, target_rgb_f32);
    fn.setArg(16, config_i32);
    fn.setArg(17, config_f32);
    fn.setArg(18, loss);
    fn.setArg(19, grad_site_rgba);
    fn.dispatch((uint64_t)sample_count, threads);
  });
  return std::make_tuple(loss, grad_site_rgba);
}

torch::Tensor metal_endpoint_record_edit_block4_vjp_direct_atomic_rgb_only(
    const torch::Tensor& boundary_f32,
    const torch::Tensor& rays_f32,
    const torch::Tensor& frame_t_f32,
    const torch::Tensor& anchor_offsets_i32,
    const torch::Tensor& anchor_owner_i32,
    const torch::Tensor& anchor_left_i32,
    const torch::Tensor& anchor_right_i32,
    const torch::Tensor& track_block_change_offsets_i32,
    const torch::Tensor& change_frame_i32,
    const torch::Tensor& op_offsets_i32,
    const torch::Tensor& op_type_i32,
    const torch::Tensor& op_pos_i32,
    const torch::Tensor& op_owner_i32,
    const torch::Tensor& op_left_i32,
    const torch::Tensor& op_right_i32,
    const torch::Tensor& site_rgba_f32,
    const torch::Tensor& grad_rgb_f32,
    const torch::Tensor& config_i32,
    const torch::Tensor& config_f32) {
  check_float_mps_2d(boundary_f32, "boundary_f32", 5);
  TORCH_CHECK(rays_f32.device().is_mps(), "rays_f32 must be on MPS");
  TORCH_CHECK(rays_f32.scalar_type() == torch::kFloat32, "rays_f32 must be float32");
  TORCH_CHECK(rays_f32.dim() == 3 && rays_f32.size(2) == 6, "rays_f32 must have shape [K,T,6]");
  TORCH_CHECK(rays_f32.is_contiguous(), "rays_f32 must be contiguous");
  check_float_mps_1d_any(frame_t_f32, "frame_t_f32");
  check_i32_mps_1d_any(anchor_offsets_i32, "anchor_offsets_i32");
  check_i32_mps_1d_any(anchor_owner_i32, "anchor_owner_i32");
  check_i32_mps_1d_any(anchor_left_i32, "anchor_left_i32");
  check_i32_mps_1d_any(anchor_right_i32, "anchor_right_i32");
  check_i32_mps_1d_any(track_block_change_offsets_i32, "track_block_change_offsets_i32");
  check_i32_mps_1d_any(change_frame_i32, "change_frame_i32");
  check_i32_mps_1d_any(op_offsets_i32, "op_offsets_i32");
  check_i32_mps_1d_any(op_type_i32, "op_type_i32");
  check_i32_mps_1d_any(op_pos_i32, "op_pos_i32");
  check_i32_mps_1d_any(op_owner_i32, "op_owner_i32");
  check_i32_mps_1d_any(op_left_i32, "op_left_i32");
  check_i32_mps_1d_any(op_right_i32, "op_right_i32");
  check_float_mps_2d(site_rgba_f32, "site_rgba_f32", 4);
  TORCH_CHECK(grad_rgb_f32.device().is_mps(), "grad_rgb_f32 must be on MPS");
  TORCH_CHECK(grad_rgb_f32.scalar_type() == torch::kFloat32, "grad_rgb_f32 must be float32");
  TORCH_CHECK(grad_rgb_f32.dim() == 3 && grad_rgb_f32.size(2) == 3, "grad_rgb_f32 must have shape [K,T,3]");
  TORCH_CHECK(grad_rgb_f32.is_contiguous(), "grad_rgb_f32 must be contiguous");
  check_i32_mps_1d(config_i32, "config_i32", 9);
  TORCH_CHECK(config_f32.device().is_mps(), "config_f32 must be on MPS");
  TORCH_CHECK(config_f32.scalar_type() == torch::kFloat32, "config_f32 must be float32");
  TORCH_CHECK(config_f32.dim() == 1 && config_f32.size(0) == 4, "config_f32 must have shape [4]");
  TORCH_CHECK(config_f32.is_contiguous(), "config_f32 must be contiguous");

  auto config_i32_cpu = config_i32.cpu();
  const int32_t* config = config_i32_cpu.data_ptr<int32_t>();
  const int64_t boundary_count = config[0];
  const int64_t track_count = config[1];
  const int64_t frame_count = config[2];
  const int64_t site_count = config[3];
  const int64_t anchor_record_count = config[4];
  const int64_t change_count = config[5];
  const int64_t op_count = config[6];
  const int64_t block_size = config[7];
  const int64_t block_count = config[8];
  TORCH_CHECK(boundary_count == boundary_f32.size(0), "config_i32[0] must match boundary_f32 rows");
  TORCH_CHECK(boundary_count > 0, "config_i32[0] boundary count must be positive");
  TORCH_CHECK(track_count > 0, "config_i32[1] track count must be positive");
  TORCH_CHECK(frame_count > 0, "config_i32[2] frame count must be positive");
  TORCH_CHECK(site_count == site_rgba_f32.size(0), "config_i32[3] must match site_rgba_f32 rows");
  TORCH_CHECK(site_count > 0 && site_count <= 64, "endpoint record block4 RGB-only VJP supports site count in [1, 64]");
  TORCH_CHECK(anchor_record_count >= 0, "config_i32[4] anchor record count must be nonnegative");
  TORCH_CHECK(change_count >= 0, "config_i32[5] change count must be nonnegative");
  TORCH_CHECK(op_count >= 0, "config_i32[6] op count must be nonnegative");
  TORCH_CHECK(block_size > 0, "endpoint_record_edit_block4_vjp_direct_atomic_rgb_only requires positive block size");
  TORCH_CHECK(block_count == (frame_count + block_size - 1) / block_size, "config_i32[8] block count mismatch");
  TORCH_CHECK(rays_f32.size(0) == track_count && rays_f32.size(1) == frame_count, "rays_f32 shape mismatch");
  TORCH_CHECK(frame_t_f32.size(0) == frame_count, "frame_t_f32 length must match frame count");
  TORCH_CHECK(anchor_owner_i32.size(0) == anchor_record_count, "anchor_owner_i32 length must match anchor record count");
  TORCH_CHECK(anchor_left_i32.size(0) == anchor_record_count, "anchor_left_i32 length must match anchor record count");
  TORCH_CHECK(anchor_right_i32.size(0) == anchor_record_count, "anchor_right_i32 length must match anchor record count");
  TORCH_CHECK(change_frame_i32.size(0) == change_count, "change_frame_i32 length must match change count");
  TORCH_CHECK(op_type_i32.size(0) == op_count, "op_type_i32 length must match op count");
  TORCH_CHECK(op_pos_i32.size(0) == op_count, "op_pos_i32 length must match op count");
  TORCH_CHECK(op_owner_i32.size(0) == op_count, "op_owner_i32 length must match op count");
  TORCH_CHECK(op_left_i32.size(0) == op_count, "op_left_i32 length must match op count");
  TORCH_CHECK(op_right_i32.size(0) == op_count, "op_right_i32 length must match op count");
  TORCH_CHECK(anchor_offsets_i32.size(0) == track_count * block_count + 1, "anchor_offsets_i32 length mismatch");
  TORCH_CHECK(
      track_block_change_offsets_i32.size(0) == track_count * (block_count + 1),
      "track_block_change_offsets_i32 length mismatch");
  TORCH_CHECK(op_offsets_i32.size(0) == change_count + 1, "op_offsets_i32 length must be change_count + 1");
  TORCH_CHECK(
      grad_rgb_f32.size(0) == track_count && grad_rgb_f32.size(1) == frame_count,
      "grad_rgb_f32 shape mismatch");
  check_segment_tape_offsets_cpu(anchor_offsets_i32, track_count * block_count, anchor_record_count, 2147483647);
  check_segment_tape_offsets_cpu(
      track_block_change_offsets_i32, track_count * (block_count + 1) - 1, change_count, 2147483647);
  check_segment_tape_offsets_cpu(op_offsets_i32, change_count, op_count, 2147483647);

  auto grad_site_rgba = torch::zeros({site_count, 4}, site_rgba_f32.options().dtype(torch::kFloat32));
  auto& k = kernels();
  constexpr uint64_t threads = 256ull;
  const int64_t sample_count = track_count * frame_count;
  launch(k.endpoint_record_edit_block4_vjp_direct_atomic_rgb_only, [&](MetalKernelFunction& fn) {
    fn.setArg(0, boundary_f32);
    fn.setArg(1, rays_f32);
    fn.setArg(2, frame_t_f32);
    fn.setArg(3, anchor_offsets_i32);
    fn.setArg(4, anchor_owner_i32);
    fn.setArg(5, anchor_left_i32);
    fn.setArg(6, anchor_right_i32);
    fn.setArg(7, track_block_change_offsets_i32);
    fn.setArg(8, change_frame_i32);
    fn.setArg(9, op_offsets_i32);
    fn.setArg(10, op_type_i32);
    fn.setArg(11, op_pos_i32);
    fn.setArg(12, op_owner_i32);
    fn.setArg(13, op_left_i32);
    fn.setArg(14, op_right_i32);
    fn.setArg(15, site_rgba_f32);
    fn.setArg(16, grad_rgb_f32);
    fn.setArg(17, config_i32);
    fn.setArg(18, config_f32);
    fn.setArg(19, grad_site_rgba);
    fn.dispatch((uint64_t)sample_count, threads);
  });
  return grad_site_rgba;
}

torch::Tensor metal_endpoint_record_edit_block_coeff_vjp_direct_atomic_rgb_only(
    const torch::Tensor& coeff_f32,
    const torch::Tensor& frame_t_f32,
    const torch::Tensor& anchor_offsets_i32,
    const torch::Tensor& anchor_owner_i32,
    const torch::Tensor& anchor_left_i32,
    const torch::Tensor& anchor_right_i32,
    const torch::Tensor& track_block_change_offsets_i32,
    const torch::Tensor& change_frame_i32,
    const torch::Tensor& op_offsets_i32,
    const torch::Tensor& op_type_i32,
    const torch::Tensor& op_pos_i32,
    const torch::Tensor& op_owner_i32,
    const torch::Tensor& op_left_i32,
    const torch::Tensor& op_right_i32,
    const torch::Tensor& site_rgba_f32,
    const torch::Tensor& grad_rgb_f32,
    const torch::Tensor& config_i32,
    const torch::Tensor& config_f32) {
  check_float_mps_2d(coeff_f32, "coeff_f32", 4);
  check_float_mps_1d_any(frame_t_f32, "frame_t_f32");
  check_i32_mps_1d_any(anchor_offsets_i32, "anchor_offsets_i32");
  check_i32_mps_1d_any(anchor_owner_i32, "anchor_owner_i32");
  check_i32_mps_1d_any(anchor_left_i32, "anchor_left_i32");
  check_i32_mps_1d_any(anchor_right_i32, "anchor_right_i32");
  check_i32_mps_1d_any(track_block_change_offsets_i32, "track_block_change_offsets_i32");
  check_i32_mps_1d_any(change_frame_i32, "change_frame_i32");
  check_i32_mps_1d_any(op_offsets_i32, "op_offsets_i32");
  check_i32_mps_1d_any(op_type_i32, "op_type_i32");
  check_i32_mps_1d_any(op_pos_i32, "op_pos_i32");
  check_i32_mps_1d_any(op_owner_i32, "op_owner_i32");
  check_i32_mps_1d_any(op_left_i32, "op_left_i32");
  check_i32_mps_1d_any(op_right_i32, "op_right_i32");
  check_float_mps_2d(site_rgba_f32, "site_rgba_f32", 4);
  TORCH_CHECK(grad_rgb_f32.device().is_mps(), "grad_rgb_f32 must be on MPS");
  TORCH_CHECK(grad_rgb_f32.scalar_type() == torch::kFloat32, "grad_rgb_f32 must be float32");
  TORCH_CHECK(grad_rgb_f32.dim() == 3 && grad_rgb_f32.size(2) == 3, "grad_rgb_f32 must have shape [K,T,3]");
  TORCH_CHECK(grad_rgb_f32.is_contiguous(), "grad_rgb_f32 must be contiguous");
  check_i32_mps_1d(config_i32, "config_i32", 9);
  TORCH_CHECK(config_f32.device().is_mps(), "config_f32 must be on MPS");
  TORCH_CHECK(config_f32.scalar_type() == torch::kFloat32, "config_f32 must be float32");
  TORCH_CHECK(config_f32.dim() == 1 && config_f32.size(0) == 4, "config_f32 must have shape [4]");
  TORCH_CHECK(config_f32.is_contiguous(), "config_f32 must be contiguous");

  auto config_i32_cpu = config_i32.cpu();
  const int32_t* config = config_i32_cpu.data_ptr<int32_t>();
  const int64_t boundary_count = config[0];
  const int64_t track_count = config[1];
  const int64_t frame_count = config[2];
  const int64_t site_count = config[3];
  const int64_t anchor_record_count = config[4];
  const int64_t change_count = config[5];
  const int64_t op_count = config[6];
  const int64_t block_size = config[7];
  const int64_t block_count = config[8];
  TORCH_CHECK(boundary_count > 0, "config_i32[0] boundary count must be positive");
  TORCH_CHECK(track_count > 0, "config_i32[1] track count must be positive");
  TORCH_CHECK(frame_count > 0, "config_i32[2] frame count must be positive");
  TORCH_CHECK(site_count == site_rgba_f32.size(0), "config_i32[3] must match site_rgba_f32 rows");
  TORCH_CHECK(site_count > 0 && site_count <= 64, "endpoint record block coeff RGB-only VJP supports site count in [1, 64]");
  TORCH_CHECK(anchor_record_count >= 0, "config_i32[4] anchor record count must be nonnegative");
  TORCH_CHECK(change_count >= 0, "config_i32[5] change count must be nonnegative");
  TORCH_CHECK(op_count >= 0, "config_i32[6] op count must be nonnegative");
  TORCH_CHECK(block_size > 0, "endpoint_record_edit_block_coeff_vjp_direct_atomic_rgb_only requires positive block size");
  TORCH_CHECK(block_count == (frame_count + block_size - 1) / block_size, "config_i32[8] block count mismatch");
  TORCH_CHECK(coeff_f32.size(0) == track_count * boundary_count, "coeff_f32 row count mismatch");
  TORCH_CHECK(frame_t_f32.size(0) == frame_count, "frame_t_f32 length must match frame count");
  TORCH_CHECK(anchor_owner_i32.size(0) == anchor_record_count, "anchor_owner_i32 length must match anchor record count");
  TORCH_CHECK(anchor_left_i32.size(0) == anchor_record_count, "anchor_left_i32 length must match anchor record count");
  TORCH_CHECK(anchor_right_i32.size(0) == anchor_record_count, "anchor_right_i32 length must match anchor record count");
  TORCH_CHECK(change_frame_i32.size(0) == change_count, "change_frame_i32 length must match change count");
  TORCH_CHECK(op_type_i32.size(0) == op_count, "op_type_i32 length must match op count");
  TORCH_CHECK(op_pos_i32.size(0) == op_count, "op_pos_i32 length must match op count");
  TORCH_CHECK(op_owner_i32.size(0) == op_count, "op_owner_i32 length must match op count");
  TORCH_CHECK(op_left_i32.size(0) == op_count, "op_left_i32 length must match op count");
  TORCH_CHECK(op_right_i32.size(0) == op_count, "op_right_i32 length must match op count");
  TORCH_CHECK(anchor_offsets_i32.size(0) == track_count * block_count + 1, "anchor_offsets_i32 length mismatch");
  TORCH_CHECK(
      track_block_change_offsets_i32.size(0) == track_count * (block_count + 1),
      "track_block_change_offsets_i32 length mismatch");
  TORCH_CHECK(op_offsets_i32.size(0) == change_count + 1, "op_offsets_i32 length must be change_count + 1");
  TORCH_CHECK(
      grad_rgb_f32.size(0) == track_count && grad_rgb_f32.size(1) == frame_count,
      "grad_rgb_f32 shape mismatch");
  check_segment_tape_offsets_cpu(anchor_offsets_i32, track_count * block_count, anchor_record_count, 2147483647);
  check_segment_tape_offsets_cpu(
      track_block_change_offsets_i32, track_count * (block_count + 1) - 1, change_count, 2147483647);
  check_segment_tape_offsets_cpu(op_offsets_i32, change_count, op_count, 2147483647);

  auto grad_site_rgba = torch::zeros({site_count, 4}, site_rgba_f32.options().dtype(torch::kFloat32));
  auto& k = kernels();
  constexpr uint64_t threads = 256ull;
  const int64_t sample_count = track_count * frame_count;
  launch(k.endpoint_record_edit_block_coeff_vjp_direct_atomic_rgb_only, [&](MetalKernelFunction& fn) {
    fn.setArg(0, coeff_f32);
    fn.setArg(1, frame_t_f32);
    fn.setArg(2, anchor_offsets_i32);
    fn.setArg(3, anchor_owner_i32);
    fn.setArg(4, anchor_left_i32);
    fn.setArg(5, anchor_right_i32);
    fn.setArg(6, track_block_change_offsets_i32);
    fn.setArg(7, change_frame_i32);
    fn.setArg(8, op_offsets_i32);
    fn.setArg(9, op_type_i32);
    fn.setArg(10, op_pos_i32);
    fn.setArg(11, op_owner_i32);
    fn.setArg(12, op_left_i32);
    fn.setArg(13, op_right_i32);
    fn.setArg(14, site_rgba_f32);
    fn.setArg(15, grad_rgb_f32);
    fn.setArg(16, config_i32);
    fn.setArg(17, config_f32);
    fn.setArg(18, grad_site_rgba);
    fn.dispatch((uint64_t)sample_count, threads);
  });
  return grad_site_rgba;
}

std::tuple<torch::Tensor, torch::Tensor> metal_endpoint_record_edit_block_coeff_mse_vjp_direct_atomic_rgb_only(
    const torch::Tensor& coeff_f32,
    const torch::Tensor& frame_t_f32,
    const torch::Tensor& anchor_offsets_i32,
    const torch::Tensor& anchor_owner_i32,
    const torch::Tensor& anchor_left_i32,
    const torch::Tensor& anchor_right_i32,
    const torch::Tensor& track_block_change_offsets_i32,
    const torch::Tensor& change_frame_i32,
    const torch::Tensor& op_offsets_i32,
    const torch::Tensor& op_type_i32,
    const torch::Tensor& op_pos_i32,
    const torch::Tensor& op_owner_i32,
    const torch::Tensor& op_left_i32,
    const torch::Tensor& op_right_i32,
    const torch::Tensor& site_rgba_f32,
    const torch::Tensor& target_rgb_f32,
    const torch::Tensor& config_i32,
    const torch::Tensor& config_f32) {
  check_float_mps_2d(coeff_f32, "coeff_f32", 4);
  check_float_mps_1d_any(frame_t_f32, "frame_t_f32");
  check_i32_mps_1d_any(anchor_offsets_i32, "anchor_offsets_i32");
  check_i32_mps_1d_any(anchor_owner_i32, "anchor_owner_i32");
  check_i32_mps_1d_any(anchor_left_i32, "anchor_left_i32");
  check_i32_mps_1d_any(anchor_right_i32, "anchor_right_i32");
  check_i32_mps_1d_any(track_block_change_offsets_i32, "track_block_change_offsets_i32");
  check_i32_mps_1d_any(change_frame_i32, "change_frame_i32");
  check_i32_mps_1d_any(op_offsets_i32, "op_offsets_i32");
  check_i32_mps_1d_any(op_type_i32, "op_type_i32");
  check_i32_mps_1d_any(op_pos_i32, "op_pos_i32");
  check_i32_mps_1d_any(op_owner_i32, "op_owner_i32");
  check_i32_mps_1d_any(op_left_i32, "op_left_i32");
  check_i32_mps_1d_any(op_right_i32, "op_right_i32");
  check_float_mps_2d(site_rgba_f32, "site_rgba_f32", 4);
  TORCH_CHECK(target_rgb_f32.device().is_mps(), "target_rgb_f32 must be on MPS");
  TORCH_CHECK(target_rgb_f32.scalar_type() == torch::kFloat32, "target_rgb_f32 must be float32");
  TORCH_CHECK(target_rgb_f32.dim() == 3 && target_rgb_f32.size(2) == 3, "target_rgb_f32 must have shape [K,T,3]");
  TORCH_CHECK(target_rgb_f32.is_contiguous(), "target_rgb_f32 must be contiguous");
  check_i32_mps_1d(config_i32, "config_i32", 9);
  TORCH_CHECK(config_f32.device().is_mps(), "config_f32 must be on MPS");
  TORCH_CHECK(config_f32.scalar_type() == torch::kFloat32, "config_f32 must be float32");
  TORCH_CHECK(config_f32.dim() == 1 && config_f32.size(0) == 4, "config_f32 must have shape [4]");
  TORCH_CHECK(config_f32.is_contiguous(), "config_f32 must be contiguous");

  auto config_i32_cpu = config_i32.cpu();
  const int32_t* config = config_i32_cpu.data_ptr<int32_t>();
  const int64_t boundary_count = config[0];
  const int64_t track_count = config[1];
  const int64_t frame_count = config[2];
  const int64_t site_count = config[3];
  const int64_t anchor_record_count = config[4];
  const int64_t change_count = config[5];
  const int64_t op_count = config[6];
  const int64_t block_size = config[7];
  const int64_t block_count = config[8];
  TORCH_CHECK(boundary_count > 0, "config_i32[0] boundary count must be positive");
  TORCH_CHECK(track_count > 0, "config_i32[1] track count must be positive");
  TORCH_CHECK(frame_count > 0, "config_i32[2] frame count must be positive");
  TORCH_CHECK(site_count == site_rgba_f32.size(0), "config_i32[3] must match site_rgba_f32 rows");
  TORCH_CHECK(
      site_count > 0 && site_count <= 64,
      "endpoint record block coeff fused MSE VJP supports site count in [1, 64]");
  TORCH_CHECK(anchor_record_count >= 0, "config_i32[4] anchor record count must be nonnegative");
  TORCH_CHECK(change_count >= 0, "config_i32[5] change count must be nonnegative");
  TORCH_CHECK(op_count >= 0, "config_i32[6] op count must be nonnegative");
  TORCH_CHECK(
      block_size > 0,
      "endpoint_record_edit_block_coeff_mse_vjp_direct_atomic_rgb_only requires positive block size");
  TORCH_CHECK(block_count == (frame_count + block_size - 1) / block_size, "config_i32[8] block count mismatch");
  TORCH_CHECK(coeff_f32.size(0) == track_count * boundary_count, "coeff_f32 row count mismatch");
  TORCH_CHECK(frame_t_f32.size(0) == frame_count, "frame_t_f32 length must match frame count");
  TORCH_CHECK(anchor_owner_i32.size(0) == anchor_record_count, "anchor_owner_i32 length must match anchor record count");
  TORCH_CHECK(anchor_left_i32.size(0) == anchor_record_count, "anchor_left_i32 length must match anchor record count");
  TORCH_CHECK(anchor_right_i32.size(0) == anchor_record_count, "anchor_right_i32 length must match anchor record count");
  TORCH_CHECK(change_frame_i32.size(0) == change_count, "change_frame_i32 length must match change count");
  TORCH_CHECK(op_type_i32.size(0) == op_count, "op_type_i32 length must match op count");
  TORCH_CHECK(op_pos_i32.size(0) == op_count, "op_pos_i32 length must match op count");
  TORCH_CHECK(op_owner_i32.size(0) == op_count, "op_owner_i32 length must match op count");
  TORCH_CHECK(op_left_i32.size(0) == op_count, "op_left_i32 length must match op count");
  TORCH_CHECK(op_right_i32.size(0) == op_count, "op_right_i32 length must match op count");
  TORCH_CHECK(anchor_offsets_i32.size(0) == track_count * block_count + 1, "anchor_offsets_i32 length mismatch");
  TORCH_CHECK(
      track_block_change_offsets_i32.size(0) == track_count * (block_count + 1),
      "track_block_change_offsets_i32 length mismatch");
  TORCH_CHECK(op_offsets_i32.size(0) == change_count + 1, "op_offsets_i32 length must be change_count + 1");
  TORCH_CHECK(
      target_rgb_f32.size(0) == track_count && target_rgb_f32.size(1) == frame_count,
      "target_rgb_f32 shape mismatch");
  check_segment_tape_offsets_cpu(anchor_offsets_i32, track_count * block_count, anchor_record_count, 2147483647);
  check_segment_tape_offsets_cpu(
      track_block_change_offsets_i32, track_count * (block_count + 1) - 1, change_count, 2147483647);
  check_segment_tape_offsets_cpu(op_offsets_i32, change_count, op_count, 2147483647);

  auto loss = torch::empty({1}, site_rgba_f32.options().dtype(torch::kFloat32));
  auto grad_site_rgba = torch::empty({site_count, 4}, site_rgba_f32.options().dtype(torch::kFloat32));
  auto& k = kernels();
  constexpr uint64_t threads = 256ull;
  const int64_t sample_count = track_count * frame_count;
  launch(k.clear_endpoint_loss_site_rgba_grad, [&](MetalKernelFunction& fn) {
    fn.setArg(0, loss);
    fn.setArg(1, grad_site_rgba);
    fn.setArg(2, config_i32);
    fn.dispatch((uint64_t)std::max<int64_t>(site_count, 1), threads);
  });
  launch(k.endpoint_record_edit_block_coeff_mse_vjp_direct_atomic_rgb_only, [&](MetalKernelFunction& fn) {
    fn.setArg(0, coeff_f32);
    fn.setArg(1, frame_t_f32);
    fn.setArg(2, anchor_offsets_i32);
    fn.setArg(3, anchor_owner_i32);
    fn.setArg(4, anchor_left_i32);
    fn.setArg(5, anchor_right_i32);
    fn.setArg(6, track_block_change_offsets_i32);
    fn.setArg(7, change_frame_i32);
    fn.setArg(8, op_offsets_i32);
    fn.setArg(9, op_type_i32);
    fn.setArg(10, op_pos_i32);
    fn.setArg(11, op_owner_i32);
    fn.setArg(12, op_left_i32);
    fn.setArg(13, op_right_i32);
    fn.setArg(14, site_rgba_f32);
    fn.setArg(15, target_rgb_f32);
    fn.setArg(16, config_i32);
    fn.setArg(17, config_f32);
    fn.setArg(18, loss);
    fn.setArg(19, grad_site_rgba);
    fn.dispatch((uint64_t)sample_count, threads);
  });
  return std::make_tuple(loss, grad_site_rgba);
}

std::tuple<torch::Tensor, torch::Tensor> metal_endpoint_record_edit_block_coeff16_mse_vjp_direct_atomic_rgb_only(
    const torch::Tensor& coeff_f16,
    const torch::Tensor& frame_t_f32,
    const torch::Tensor& anchor_offsets_i32,
    const torch::Tensor& anchor_owner_i32,
    const torch::Tensor& anchor_left_i32,
    const torch::Tensor& anchor_right_i32,
    const torch::Tensor& track_block_change_offsets_i32,
    const torch::Tensor& change_frame_i32,
    const torch::Tensor& op_offsets_i32,
    const torch::Tensor& op_type_i32,
    const torch::Tensor& op_pos_i32,
    const torch::Tensor& op_owner_i32,
    const torch::Tensor& op_left_i32,
    const torch::Tensor& op_right_i32,
    const torch::Tensor& site_rgba_f32,
    const torch::Tensor& target_rgb_f32,
    const torch::Tensor& config_i32,
    const torch::Tensor& config_f32) {
  check_half_mps_2d(coeff_f16, "coeff_f16", 4);
  check_float_mps_1d_any(frame_t_f32, "frame_t_f32");
  check_i32_mps_1d_any(anchor_offsets_i32, "anchor_offsets_i32");
  check_i32_mps_1d_any(anchor_owner_i32, "anchor_owner_i32");
  check_i32_mps_1d_any(anchor_left_i32, "anchor_left_i32");
  check_i32_mps_1d_any(anchor_right_i32, "anchor_right_i32");
  check_i32_mps_1d_any(track_block_change_offsets_i32, "track_block_change_offsets_i32");
  check_i32_mps_1d_any(change_frame_i32, "change_frame_i32");
  check_i32_mps_1d_any(op_offsets_i32, "op_offsets_i32");
  check_i32_mps_1d_any(op_type_i32, "op_type_i32");
  check_i32_mps_1d_any(op_pos_i32, "op_pos_i32");
  check_i32_mps_1d_any(op_owner_i32, "op_owner_i32");
  check_i32_mps_1d_any(op_left_i32, "op_left_i32");
  check_i32_mps_1d_any(op_right_i32, "op_right_i32");
  check_float_mps_2d(site_rgba_f32, "site_rgba_f32", 4);
  TORCH_CHECK(target_rgb_f32.device().is_mps(), "target_rgb_f32 must be on MPS");
  TORCH_CHECK(target_rgb_f32.scalar_type() == torch::kFloat32, "target_rgb_f32 must be float32");
  TORCH_CHECK(target_rgb_f32.dim() == 3 && target_rgb_f32.size(2) == 3, "target_rgb_f32 must have shape [K,T,3]");
  TORCH_CHECK(target_rgb_f32.is_contiguous(), "target_rgb_f32 must be contiguous");
  check_i32_mps_1d(config_i32, "config_i32", 9);
  TORCH_CHECK(config_f32.device().is_mps(), "config_f32 must be on MPS");
  TORCH_CHECK(config_f32.scalar_type() == torch::kFloat32, "config_f32 must be float32");
  TORCH_CHECK(config_f32.dim() == 1 && config_f32.size(0) == 4, "config_f32 must have shape [4]");
  TORCH_CHECK(config_f32.is_contiguous(), "config_f32 must be contiguous");

  auto config_i32_cpu = config_i32.cpu();
  const int32_t* config = config_i32_cpu.data_ptr<int32_t>();
  const int64_t boundary_count = config[0];
  const int64_t track_count = config[1];
  const int64_t frame_count = config[2];
  const int64_t site_count = config[3];
  const int64_t anchor_record_count = config[4];
  const int64_t change_count = config[5];
  const int64_t op_count = config[6];
  const int64_t block_size = config[7];
  const int64_t block_count = config[8];
  TORCH_CHECK(boundary_count > 0, "config_i32[0] boundary count must be positive");
  TORCH_CHECK(track_count > 0, "config_i32[1] track count must be positive");
  TORCH_CHECK(frame_count > 0, "config_i32[2] frame count must be positive");
  TORCH_CHECK(site_count == site_rgba_f32.size(0), "config_i32[3] must match site_rgba_f32 rows");
  TORCH_CHECK(
      site_count > 0 && site_count <= 64,
      "endpoint record block coeff16 fused MSE VJP supports site count in [1, 64]");
  TORCH_CHECK(anchor_record_count >= 0, "config_i32[4] anchor record count must be nonnegative");
  TORCH_CHECK(change_count >= 0, "config_i32[5] change count must be nonnegative");
  TORCH_CHECK(op_count >= 0, "config_i32[6] op count must be nonnegative");
  TORCH_CHECK(
      block_size > 0,
      "endpoint_record_edit_block_coeff16_mse_vjp_direct_atomic_rgb_only requires positive block size");
  TORCH_CHECK(block_count == (frame_count + block_size - 1) / block_size, "config_i32[8] block count mismatch");
  TORCH_CHECK(coeff_f16.size(0) == track_count * boundary_count, "coeff_f16 row count mismatch");
  TORCH_CHECK(frame_t_f32.size(0) == frame_count, "frame_t_f32 length must match frame count");
  TORCH_CHECK(anchor_owner_i32.size(0) == anchor_record_count, "anchor_owner_i32 length must match anchor record count");
  TORCH_CHECK(anchor_left_i32.size(0) == anchor_record_count, "anchor_left_i32 length must match anchor record count");
  TORCH_CHECK(anchor_right_i32.size(0) == anchor_record_count, "anchor_right_i32 length must match anchor record count");
  TORCH_CHECK(change_frame_i32.size(0) == change_count, "change_frame_i32 length must match change count");
  TORCH_CHECK(op_type_i32.size(0) == op_count, "op_type_i32 length must match op count");
  TORCH_CHECK(op_pos_i32.size(0) == op_count, "op_pos_i32 length must match op count");
  TORCH_CHECK(op_owner_i32.size(0) == op_count, "op_owner_i32 length must match op count");
  TORCH_CHECK(op_left_i32.size(0) == op_count, "op_left_i32 length must match op count");
  TORCH_CHECK(op_right_i32.size(0) == op_count, "op_right_i32 length must match op count");
  TORCH_CHECK(anchor_offsets_i32.size(0) == track_count * block_count + 1, "anchor_offsets_i32 length mismatch");
  TORCH_CHECK(
      track_block_change_offsets_i32.size(0) == track_count * (block_count + 1),
      "track_block_change_offsets_i32 length mismatch");
  TORCH_CHECK(op_offsets_i32.size(0) == change_count + 1, "op_offsets_i32 length must be change_count + 1");
  TORCH_CHECK(
      target_rgb_f32.size(0) == track_count && target_rgb_f32.size(1) == frame_count,
      "target_rgb_f32 shape mismatch");
  check_segment_tape_offsets_cpu(anchor_offsets_i32, track_count * block_count, anchor_record_count, 2147483647);
  check_segment_tape_offsets_cpu(
      track_block_change_offsets_i32, track_count * (block_count + 1) - 1, change_count, 2147483647);
  check_segment_tape_offsets_cpu(op_offsets_i32, change_count, op_count, 2147483647);

  auto loss = torch::empty({1}, site_rgba_f32.options().dtype(torch::kFloat32));
  auto grad_site_rgba = torch::empty({site_count, 4}, site_rgba_f32.options().dtype(torch::kFloat32));
  auto& k = kernels();
  constexpr uint64_t threads = 256ull;
  const int64_t sample_count = track_count * frame_count;
  launch(k.clear_endpoint_loss_site_rgba_grad, [&](MetalKernelFunction& fn) {
    fn.setArg(0, loss);
    fn.setArg(1, grad_site_rgba);
    fn.setArg(2, config_i32);
    fn.dispatch((uint64_t)std::max<int64_t>(site_count, 1), threads);
  });
  launch(k.endpoint_record_edit_block_coeff16_mse_vjp_direct_atomic_rgb_only, [&](MetalKernelFunction& fn) {
    fn.setArg(0, coeff_f16);
    fn.setArg(1, frame_t_f32);
    fn.setArg(2, anchor_offsets_i32);
    fn.setArg(3, anchor_owner_i32);
    fn.setArg(4, anchor_left_i32);
    fn.setArg(5, anchor_right_i32);
    fn.setArg(6, track_block_change_offsets_i32);
    fn.setArg(7, change_frame_i32);
    fn.setArg(8, op_offsets_i32);
    fn.setArg(9, op_type_i32);
    fn.setArg(10, op_pos_i32);
    fn.setArg(11, op_owner_i32);
    fn.setArg(12, op_left_i32);
    fn.setArg(13, op_right_i32);
    fn.setArg(14, site_rgba_f32);
    fn.setArg(15, target_rgb_f32);
    fn.setArg(16, config_i32);
    fn.setArg(17, config_f32);
    fn.setArg(18, loss);
    fn.setArg(19, grad_site_rgba);
    fn.dispatch((uint64_t)sample_count, threads);
  });
  return std::make_tuple(loss, grad_site_rgba);
}

std::tuple<torch::Tensor, torch::Tensor> metal_endpoint_record_edit_block_coeff16_packed_mse_vjp_direct_atomic_rgb_only(
    const torch::Tensor& coeff_f16,
    const torch::Tensor& frame_t_f32,
    const torch::Tensor& anchor_offsets_i32,
    const torch::Tensor& anchor_record_i32,
    const torch::Tensor& track_block_change_offsets_i32,
    const torch::Tensor& change_frame_i32,
    const torch::Tensor& op_offsets_i32,
    const torch::Tensor& op_type_i32,
    const torch::Tensor& op_pos_i32,
    const torch::Tensor& op_record_i32,
    const torch::Tensor& site_rgba_f32,
    const torch::Tensor& target_rgb_f32,
    const torch::Tensor& config_i32,
    const torch::Tensor& config_f32) {
  check_half_mps_2d(coeff_f16, "coeff_f16", 4);
  check_float_mps_1d_any(frame_t_f32, "frame_t_f32");
  check_i32_mps_1d_any(anchor_offsets_i32, "anchor_offsets_i32");
  check_i32_mps_1d_any(anchor_record_i32, "anchor_record_i32");
  check_i32_mps_1d_any(track_block_change_offsets_i32, "track_block_change_offsets_i32");
  check_i32_mps_1d_any(change_frame_i32, "change_frame_i32");
  check_i32_mps_1d_any(op_offsets_i32, "op_offsets_i32");
  check_i32_mps_1d_any(op_type_i32, "op_type_i32");
  check_i32_mps_1d_any(op_pos_i32, "op_pos_i32");
  check_i32_mps_1d_any(op_record_i32, "op_record_i32");
  check_float_mps_2d(site_rgba_f32, "site_rgba_f32", 4);
  TORCH_CHECK(target_rgb_f32.device().is_mps(), "target_rgb_f32 must be on MPS");
  TORCH_CHECK(target_rgb_f32.scalar_type() == torch::kFloat32, "target_rgb_f32 must be float32");
  TORCH_CHECK(target_rgb_f32.dim() == 3 && target_rgb_f32.size(2) == 3, "target_rgb_f32 must have shape [K,T,3]");
  TORCH_CHECK(target_rgb_f32.is_contiguous(), "target_rgb_f32 must be contiguous");
  check_i32_mps_1d(config_i32, "config_i32", 9);
  TORCH_CHECK(config_f32.device().is_mps(), "config_f32 must be on MPS");
  TORCH_CHECK(config_f32.scalar_type() == torch::kFloat32, "config_f32 must be float32");
  TORCH_CHECK(config_f32.dim() == 1 && config_f32.size(0) == 4, "config_f32 must have shape [4]");
  TORCH_CHECK(config_f32.is_contiguous(), "config_f32 must be contiguous");

  auto config_i32_cpu = config_i32.cpu();
  const int32_t* config = config_i32_cpu.data_ptr<int32_t>();
  const int64_t boundary_count = config[0];
  const int64_t track_count = config[1];
  const int64_t frame_count = config[2];
  const int64_t site_count = config[3];
  const int64_t anchor_record_count = config[4];
  const int64_t change_count = config[5];
  const int64_t op_count = config[6];
  const int64_t block_size = config[7];
  const int64_t block_count = config[8];
  TORCH_CHECK(boundary_count > 0, "config_i32[0] boundary count must be positive");
  TORCH_CHECK(boundary_count <= 2046, "packed block coeff16 fused MSE supports boundary count <= 2046");
  TORCH_CHECK(track_count > 0, "config_i32[1] track count must be positive");
  TORCH_CHECK(frame_count > 0, "config_i32[2] frame count must be positive");
  TORCH_CHECK(site_count == site_rgba_f32.size(0), "config_i32[3] must match site_rgba_f32 rows");
  TORCH_CHECK(
      site_count > 0 && site_count <= 255,
      "packed endpoint record block coeff16 fused MSE VJP supports site count in [1, 255]");
  TORCH_CHECK(anchor_record_count >= 0, "config_i32[4] anchor record count must be nonnegative");
  TORCH_CHECK(change_count >= 0, "config_i32[5] change count must be nonnegative");
  TORCH_CHECK(op_count >= 0, "config_i32[6] op count must be nonnegative");
  TORCH_CHECK(
      block_size > 0,
      "endpoint_record_edit_block_coeff16_packed_mse_vjp_direct_atomic_rgb_only requires positive block size");
  TORCH_CHECK(block_count == (frame_count + block_size - 1) / block_size, "config_i32[8] block count mismatch");
  TORCH_CHECK(coeff_f16.size(0) == track_count * boundary_count, "coeff_f16 row count mismatch");
  TORCH_CHECK(frame_t_f32.size(0) == frame_count, "frame_t_f32 length must match frame count");
  TORCH_CHECK(anchor_record_i32.size(0) == anchor_record_count, "anchor_record_i32 length mismatch");
  TORCH_CHECK(change_frame_i32.size(0) == change_count, "change_frame_i32 length must match change count");
  TORCH_CHECK(op_type_i32.size(0) == op_count, "op_type_i32 length must match op count");
  TORCH_CHECK(op_pos_i32.size(0) == op_count, "op_pos_i32 length must match op count");
  TORCH_CHECK(op_record_i32.size(0) == op_count, "op_record_i32 length must match op count");
  TORCH_CHECK(anchor_offsets_i32.size(0) == track_count * block_count + 1, "anchor_offsets_i32 length mismatch");
  TORCH_CHECK(
      track_block_change_offsets_i32.size(0) == track_count * (block_count + 1),
      "track_block_change_offsets_i32 length mismatch");
  TORCH_CHECK(op_offsets_i32.size(0) == change_count + 1, "op_offsets_i32 length must be change_count + 1");
  TORCH_CHECK(
      target_rgb_f32.size(0) == track_count && target_rgb_f32.size(1) == frame_count,
      "target_rgb_f32 shape mismatch");
  check_segment_tape_offsets_cpu(anchor_offsets_i32, track_count * block_count, anchor_record_count, 2147483647);
  check_segment_tape_offsets_cpu(
      track_block_change_offsets_i32, track_count * (block_count + 1) - 1, change_count, 2147483647);
  check_segment_tape_offsets_cpu(op_offsets_i32, change_count, op_count, 2147483647);

  auto loss = torch::empty({1}, site_rgba_f32.options().dtype(torch::kFloat32));
  auto grad_site_rgba = torch::empty({site_count, 4}, site_rgba_f32.options().dtype(torch::kFloat32));
  auto& k = kernels();
  constexpr uint64_t threads = 256ull;
  const int64_t sample_count = track_count * frame_count;
  launch(k.clear_endpoint_loss_site_rgba_grad, [&](MetalKernelFunction& fn) {
    fn.setArg(0, loss);
    fn.setArg(1, grad_site_rgba);
    fn.setArg(2, config_i32);
    fn.dispatch((uint64_t)std::max<int64_t>(site_count, 1), threads);
  });
  launch(k.endpoint_record_edit_block_coeff16_packed_mse_vjp_direct_atomic_rgb_only, [&](MetalKernelFunction& fn) {
    fn.setArg(0, coeff_f16);
    fn.setArg(1, frame_t_f32);
    fn.setArg(2, anchor_offsets_i32);
    fn.setArg(3, anchor_record_i32);
    fn.setArg(4, track_block_change_offsets_i32);
    fn.setArg(5, change_frame_i32);
    fn.setArg(6, op_offsets_i32);
    fn.setArg(7, op_type_i32);
    fn.setArg(8, op_pos_i32);
    fn.setArg(9, op_record_i32);
    fn.setArg(10, site_rgba_f32);
    fn.setArg(11, target_rgb_f32);
    fn.setArg(12, config_i32);
    fn.setArg(13, config_f32);
    fn.setArg(14, loss);
    fn.setArg(15, grad_site_rgba);
    fn.dispatch((uint64_t)sample_count, threads);
  });
  return std::make_tuple(loss, grad_site_rgba);
}

std::tuple<torch::Tensor, torch::Tensor> metal_endpoint_record_edit_block_coeff16_i16_mse_vjp_direct_atomic_rgb_only(
    const torch::Tensor& coeff_f16,
    const torch::Tensor& frame_t_f32,
    const torch::Tensor& anchor_offsets_i32,
    const torch::Tensor& anchor_owner_i16,
    const torch::Tensor& anchor_left_i16,
    const torch::Tensor& anchor_right_i16,
    const torch::Tensor& track_block_change_offsets_i32,
    const torch::Tensor& change_frame_i32,
    const torch::Tensor& op_offsets_i32,
    const torch::Tensor& op_type_i32,
    const torch::Tensor& op_pos_i32,
    const torch::Tensor& op_owner_i16,
    const torch::Tensor& op_left_i16,
    const torch::Tensor& op_right_i16,
    const torch::Tensor& site_rgba_f32,
    const torch::Tensor& target_rgb_f32,
    const torch::Tensor& config_i32,
    const torch::Tensor& config_f32) {
  check_half_mps_2d(coeff_f16, "coeff_f16", 4);
  check_float_mps_1d_any(frame_t_f32, "frame_t_f32");
  check_i32_mps_1d_any(anchor_offsets_i32, "anchor_offsets_i32");
  check_i16_mps_1d_any(anchor_owner_i16, "anchor_owner_i16");
  check_i16_mps_1d_any(anchor_left_i16, "anchor_left_i16");
  check_i16_mps_1d_any(anchor_right_i16, "anchor_right_i16");
  check_i32_mps_1d_any(track_block_change_offsets_i32, "track_block_change_offsets_i32");
  check_i32_mps_1d_any(change_frame_i32, "change_frame_i32");
  check_i32_mps_1d_any(op_offsets_i32, "op_offsets_i32");
  check_i32_mps_1d_any(op_type_i32, "op_type_i32");
  check_i32_mps_1d_any(op_pos_i32, "op_pos_i32");
  check_i16_mps_1d_any(op_owner_i16, "op_owner_i16");
  check_i16_mps_1d_any(op_left_i16, "op_left_i16");
  check_i16_mps_1d_any(op_right_i16, "op_right_i16");
  check_float_mps_2d(site_rgba_f32, "site_rgba_f32", 4);
  TORCH_CHECK(target_rgb_f32.device().is_mps(), "target_rgb_f32 must be on MPS");
  TORCH_CHECK(target_rgb_f32.scalar_type() == torch::kFloat32, "target_rgb_f32 must be float32");
  TORCH_CHECK(target_rgb_f32.dim() == 3 && target_rgb_f32.size(2) == 3, "target_rgb_f32 must have shape [K,T,3]");
  TORCH_CHECK(target_rgb_f32.is_contiguous(), "target_rgb_f32 must be contiguous");
  check_i32_mps_1d(config_i32, "config_i32", 9);
  TORCH_CHECK(config_f32.device().is_mps(), "config_f32 must be on MPS");
  TORCH_CHECK(config_f32.scalar_type() == torch::kFloat32, "config_f32 must be float32");
  TORCH_CHECK(config_f32.dim() == 1 && config_f32.size(0) == 4, "config_f32 must have shape [4]");
  TORCH_CHECK(config_f32.is_contiguous(), "config_f32 must be contiguous");

  auto config_i32_cpu = config_i32.cpu();
  const int32_t* config = config_i32_cpu.data_ptr<int32_t>();
  const int64_t boundary_count = config[0];
  const int64_t track_count = config[1];
  const int64_t frame_count = config[2];
  const int64_t site_count = config[3];
  const int64_t anchor_record_count = config[4];
  const int64_t change_count = config[5];
  const int64_t op_count = config[6];
  const int64_t block_size = config[7];
  const int64_t block_count = config[8];
  TORCH_CHECK(boundary_count > 0, "config_i32[0] boundary count must be positive");
  TORCH_CHECK(boundary_count <= 32765, "i16 block coeff16 fused MSE supports boundary count <= 32765");
  TORCH_CHECK(track_count > 0, "config_i32[1] track count must be positive");
  TORCH_CHECK(frame_count > 0, "config_i32[2] frame count must be positive");
  TORCH_CHECK(site_count == site_rgba_f32.size(0), "config_i32[3] must match site_rgba_f32 rows");
  TORCH_CHECK(
      site_count > 0 && site_count <= 32767,
      "i16 endpoint record block coeff16 fused MSE VJP supports site count in [1, 32767]");
  TORCH_CHECK(anchor_record_count >= 0, "config_i32[4] anchor record count must be nonnegative");
  TORCH_CHECK(change_count >= 0, "config_i32[5] change count must be nonnegative");
  TORCH_CHECK(op_count >= 0, "config_i32[6] op count must be nonnegative");
  TORCH_CHECK(
      block_size > 0,
      "endpoint_record_edit_block_coeff16_i16_mse_vjp_direct_atomic_rgb_only requires positive block size");
  TORCH_CHECK(block_count == (frame_count + block_size - 1) / block_size, "config_i32[8] block count mismatch");
  TORCH_CHECK(coeff_f16.size(0) == track_count * boundary_count, "coeff_f16 row count mismatch");
  TORCH_CHECK(frame_t_f32.size(0) == frame_count, "frame_t_f32 length must match frame count");
  TORCH_CHECK(anchor_owner_i16.size(0) == anchor_record_count, "anchor_owner_i16 length mismatch");
  TORCH_CHECK(anchor_left_i16.size(0) == anchor_record_count, "anchor_left_i16 length mismatch");
  TORCH_CHECK(anchor_right_i16.size(0) == anchor_record_count, "anchor_right_i16 length mismatch");
  TORCH_CHECK(change_frame_i32.size(0) == change_count, "change_frame_i32 length must match change count");
  TORCH_CHECK(op_type_i32.size(0) == op_count, "op_type_i32 length must match op count");
  TORCH_CHECK(op_pos_i32.size(0) == op_count, "op_pos_i32 length must match op count");
  TORCH_CHECK(op_owner_i16.size(0) == op_count, "op_owner_i16 length must match op count");
  TORCH_CHECK(op_left_i16.size(0) == op_count, "op_left_i16 length must match op count");
  TORCH_CHECK(op_right_i16.size(0) == op_count, "op_right_i16 length must match op count");
  TORCH_CHECK(anchor_offsets_i32.size(0) == track_count * block_count + 1, "anchor_offsets_i32 length mismatch");
  TORCH_CHECK(
      track_block_change_offsets_i32.size(0) == track_count * (block_count + 1),
      "track_block_change_offsets_i32 length mismatch");
  TORCH_CHECK(op_offsets_i32.size(0) == change_count + 1, "op_offsets_i32 length must be change_count + 1");
  TORCH_CHECK(
      target_rgb_f32.size(0) == track_count && target_rgb_f32.size(1) == frame_count,
      "target_rgb_f32 shape mismatch");
  check_segment_tape_offsets_cpu(anchor_offsets_i32, track_count * block_count, anchor_record_count, 2147483647);
  check_segment_tape_offsets_cpu(
      track_block_change_offsets_i32, track_count * (block_count + 1) - 1, change_count, 2147483647);
  check_segment_tape_offsets_cpu(op_offsets_i32, change_count, op_count, 2147483647);

  auto loss = torch::empty({1}, site_rgba_f32.options().dtype(torch::kFloat32));
  auto grad_site_rgba = torch::empty({site_count, 4}, site_rgba_f32.options().dtype(torch::kFloat32));
  auto& k = kernels();
  constexpr uint64_t threads = 256ull;
  const int64_t sample_count = track_count * frame_count;
  launch(k.clear_endpoint_loss_site_rgba_grad, [&](MetalKernelFunction& fn) {
    fn.setArg(0, loss);
    fn.setArg(1, grad_site_rgba);
    fn.setArg(2, config_i32);
    fn.dispatch((uint64_t)std::max<int64_t>(site_count, 1), threads);
  });
  launch(k.endpoint_record_edit_block_coeff16_i16_mse_vjp_direct_atomic_rgb_only, [&](MetalKernelFunction& fn) {
    fn.setArg(0, coeff_f16);
    fn.setArg(1, frame_t_f32);
    fn.setArg(2, anchor_offsets_i32);
    fn.setArg(3, anchor_owner_i16);
    fn.setArg(4, anchor_left_i16);
    fn.setArg(5, anchor_right_i16);
    fn.setArg(6, track_block_change_offsets_i32);
    fn.setArg(7, change_frame_i32);
    fn.setArg(8, op_offsets_i32);
    fn.setArg(9, op_type_i32);
    fn.setArg(10, op_pos_i32);
    fn.setArg(11, op_owner_i16);
    fn.setArg(12, op_left_i16);
    fn.setArg(13, op_right_i16);
    fn.setArg(14, site_rgba_f32);
    fn.setArg(15, target_rgb_f32);
    fn.setArg(16, config_i32);
    fn.setArg(17, config_f32);
    fn.setArg(18, loss);
    fn.setArg(19, grad_site_rgba);
    fn.dispatch((uint64_t)sample_count, threads);
  });
  return std::make_tuple(loss, grad_site_rgba);
}

std::tuple<torch::Tensor, torch::Tensor> metal_endpoint_record_edit_block_coeff16_i16x3_mse_vjp_direct_atomic_rgb_only(
    const torch::Tensor& coeff_f16,
    const torch::Tensor& frame_t_f32,
    const torch::Tensor& anchor_offsets_i32,
    const torch::Tensor& anchor_record_i16,
    const torch::Tensor& track_block_change_offsets_i32,
    const torch::Tensor& change_frame_i32,
    const torch::Tensor& op_offsets_i32,
    const torch::Tensor& op_type_i32,
    const torch::Tensor& op_pos_i32,
    const torch::Tensor& op_record_i16,
    const torch::Tensor& site_rgba_f32,
    const torch::Tensor& target_rgb_f32,
    const torch::Tensor& config_i32,
    const torch::Tensor& config_f32) {
  check_half_mps_2d(coeff_f16, "coeff_f16", 4);
  check_float_mps_1d_any(frame_t_f32, "frame_t_f32");
  check_i32_mps_1d_any(anchor_offsets_i32, "anchor_offsets_i32");
  check_i16_mps_1d_any(anchor_record_i16, "anchor_record_i16");
  check_i32_mps_1d_any(track_block_change_offsets_i32, "track_block_change_offsets_i32");
  check_i32_mps_1d_any(change_frame_i32, "change_frame_i32");
  check_i32_mps_1d_any(op_offsets_i32, "op_offsets_i32");
  check_i32_mps_1d_any(op_type_i32, "op_type_i32");
  check_i32_mps_1d_any(op_pos_i32, "op_pos_i32");
  check_i16_mps_1d_any(op_record_i16, "op_record_i16");
  check_float_mps_2d(site_rgba_f32, "site_rgba_f32", 4);
  TORCH_CHECK(target_rgb_f32.device().is_mps(), "target_rgb_f32 must be on MPS");
  TORCH_CHECK(target_rgb_f32.scalar_type() == torch::kFloat32, "target_rgb_f32 must be float32");
  TORCH_CHECK(target_rgb_f32.dim() == 3 && target_rgb_f32.size(2) == 3, "target_rgb_f32 must have shape [K,T,3]");
  TORCH_CHECK(target_rgb_f32.is_contiguous(), "target_rgb_f32 must be contiguous");
  check_i32_mps_1d(config_i32, "config_i32", 9);
  TORCH_CHECK(config_f32.device().is_mps(), "config_f32 must be on MPS");
  TORCH_CHECK(config_f32.scalar_type() == torch::kFloat32, "config_f32 must be float32");
  TORCH_CHECK(config_f32.dim() == 1 && config_f32.size(0) == 4, "config_f32 must have shape [4]");
  TORCH_CHECK(config_f32.is_contiguous(), "config_f32 must be contiguous");

  auto config_i32_cpu = config_i32.cpu();
  const int32_t* config = config_i32_cpu.data_ptr<int32_t>();
  const int64_t boundary_count = config[0];
  const int64_t track_count = config[1];
  const int64_t frame_count = config[2];
  const int64_t site_count = config[3];
  const int64_t anchor_record_count = config[4];
  const int64_t change_count = config[5];
  const int64_t op_count = config[6];
  const int64_t block_size = config[7];
  const int64_t block_count = config[8];
  TORCH_CHECK(boundary_count > 0, "config_i32[0] boundary count must be positive");
  TORCH_CHECK(boundary_count <= 32765, "i16x3 block coeff16 fused MSE supports boundary count <= 32765");
  TORCH_CHECK(track_count > 0, "config_i32[1] track count must be positive");
  TORCH_CHECK(frame_count > 0, "config_i32[2] frame count must be positive");
  TORCH_CHECK(site_count == site_rgba_f32.size(0), "config_i32[3] must match site_rgba_f32 rows");
  TORCH_CHECK(
      site_count > 0 && site_count <= 32767,
      "i16x3 endpoint record block coeff16 fused MSE VJP supports site count in [1, 32767]");
  TORCH_CHECK(anchor_record_count >= 0, "config_i32[4] anchor record count must be nonnegative");
  TORCH_CHECK(change_count >= 0, "config_i32[5] change count must be nonnegative");
  TORCH_CHECK(op_count >= 0, "config_i32[6] op count must be nonnegative");
  TORCH_CHECK(
      block_size > 0,
      "endpoint_record_edit_block_coeff16_i16x3_mse_vjp_direct_atomic_rgb_only requires positive block size");
  TORCH_CHECK(block_count == (frame_count + block_size - 1) / block_size, "config_i32[8] block count mismatch");
  TORCH_CHECK(coeff_f16.size(0) == track_count * boundary_count, "coeff_f16 row count mismatch");
  TORCH_CHECK(frame_t_f32.size(0) == frame_count, "frame_t_f32 length must match frame count");
  TORCH_CHECK(anchor_record_i16.size(0) == anchor_record_count * 3, "anchor_record_i16 length mismatch");
  TORCH_CHECK(change_frame_i32.size(0) == change_count, "change_frame_i32 length must match change count");
  TORCH_CHECK(op_type_i32.size(0) == op_count, "op_type_i32 length must match op count");
  TORCH_CHECK(op_pos_i32.size(0) == op_count, "op_pos_i32 length must match op count");
  TORCH_CHECK(op_record_i16.size(0) == op_count * 3, "op_record_i16 length must match op count * 3");
  TORCH_CHECK(anchor_offsets_i32.size(0) == track_count * block_count + 1, "anchor_offsets_i32 length mismatch");
  TORCH_CHECK(
      track_block_change_offsets_i32.size(0) == track_count * (block_count + 1),
      "track_block_change_offsets_i32 length mismatch");
  TORCH_CHECK(op_offsets_i32.size(0) == change_count + 1, "op_offsets_i32 length must be change_count + 1");
  TORCH_CHECK(
      target_rgb_f32.size(0) == track_count && target_rgb_f32.size(1) == frame_count,
      "target_rgb_f32 shape mismatch");
  check_segment_tape_offsets_cpu(anchor_offsets_i32, track_count * block_count, anchor_record_count, 2147483647);
  check_segment_tape_offsets_cpu(
      track_block_change_offsets_i32, track_count * (block_count + 1) - 1, change_count, 2147483647);
  check_segment_tape_offsets_cpu(op_offsets_i32, change_count, op_count, 2147483647);

  auto loss = torch::empty({1}, site_rgba_f32.options().dtype(torch::kFloat32));
  auto grad_site_rgba = torch::empty({site_count, 4}, site_rgba_f32.options().dtype(torch::kFloat32));
  auto& k = kernels();
  constexpr uint64_t threads = 256ull;
  const int64_t sample_count = track_count * frame_count;
  launch(k.clear_endpoint_loss_site_rgba_grad, [&](MetalKernelFunction& fn) {
    fn.setArg(0, loss);
    fn.setArg(1, grad_site_rgba);
    fn.setArg(2, config_i32);
    fn.dispatch((uint64_t)std::max<int64_t>(site_count, 1), threads);
  });
  launch(k.endpoint_record_edit_block_coeff16_i16x3_mse_vjp_direct_atomic_rgb_only, [&](MetalKernelFunction& fn) {
    fn.setArg(0, coeff_f16);
    fn.setArg(1, frame_t_f32);
    fn.setArg(2, anchor_offsets_i32);
    fn.setArg(3, anchor_record_i16);
    fn.setArg(4, track_block_change_offsets_i32);
    fn.setArg(5, change_frame_i32);
    fn.setArg(6, op_offsets_i32);
    fn.setArg(7, op_type_i32);
    fn.setArg(8, op_pos_i32);
    fn.setArg(9, op_record_i16);
    fn.setArg(10, site_rgba_f32);
    fn.setArg(11, target_rgb_f32);
    fn.setArg(12, config_i32);
    fn.setArg(13, config_f32);
    fn.setArg(14, loss);
    fn.setArg(15, grad_site_rgba);
    fn.dispatch((uint64_t)sample_count, threads);
  });
  return std::make_tuple(loss, grad_site_rgba);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> metal_shared_realray_rgba_depth_vjp(
    const torch::Tensor& boundary_f32,
    const torch::Tensor& candidate_mask_i32,
    const torch::Tensor& sites_f32,
    const torch::Tensor& site_rgba_f32,
    const torch::Tensor& track_rays_f32,
    const torch::Tensor& frame_t_f32,
    const torch::Tensor& grad_rgb_f32,
    const torch::Tensor& grad_alpha_f32,
    const torch::Tensor& grad_depth_f32,
    const torch::Tensor& config_i32,
    const torch::Tensor& config_f32) {
  check_float_mps_2d(boundary_f32, "boundary_f32", 5);
  TORCH_CHECK(candidate_mask_i32.device().is_mps(), "candidate_mask_i32 must be on MPS");
  TORCH_CHECK(candidate_mask_i32.scalar_type() == torch::kInt32, "candidate_mask_i32 must be int32");
  TORCH_CHECK(candidate_mask_i32.dim() == 2, "candidate_mask_i32 must have shape [track_count * time_slab_count, mask_word_count]");
  TORCH_CHECK(candidate_mask_i32.is_contiguous(), "candidate_mask_i32 must be contiguous");
  check_float_mps_2d(sites_f32, "sites_f32", 5);
  check_float_mps_2d(site_rgba_f32, "site_rgba_f32", 4);
  check_float_mps_2d(track_rays_f32, "track_rays_f32", 6);
  TORCH_CHECK(frame_t_f32.device().is_mps(), "frame_t_f32 must be on MPS");
  TORCH_CHECK(frame_t_f32.scalar_type() == torch::kFloat32, "frame_t_f32 must be float32");
  TORCH_CHECK(frame_t_f32.dim() == 1, "frame_t_f32 must have shape [T]");
  TORCH_CHECK(frame_t_f32.is_contiguous(), "frame_t_f32 must be contiguous");
  TORCH_CHECK(grad_rgb_f32.device().is_mps(), "grad_rgb_f32 must be on MPS");
  TORCH_CHECK(grad_rgb_f32.scalar_type() == torch::kFloat32, "grad_rgb_f32 must be float32");
  TORCH_CHECK(grad_rgb_f32.dim() == 3 && grad_rgb_f32.size(2) == 3, "grad_rgb_f32 must have shape [K,T,3]");
  TORCH_CHECK(grad_rgb_f32.is_contiguous(), "grad_rgb_f32 must be contiguous");
  TORCH_CHECK(grad_alpha_f32.device().is_mps(), "grad_alpha_f32 must be on MPS");
  TORCH_CHECK(grad_alpha_f32.scalar_type() == torch::kFloat32, "grad_alpha_f32 must be float32");
  TORCH_CHECK(grad_alpha_f32.dim() == 2, "grad_alpha_f32 must have shape [K,T]");
  TORCH_CHECK(grad_alpha_f32.is_contiguous(), "grad_alpha_f32 must be contiguous");
  TORCH_CHECK(grad_depth_f32.device().is_mps(), "grad_depth_f32 must be on MPS");
  TORCH_CHECK(grad_depth_f32.scalar_type() == torch::kFloat32, "grad_depth_f32 must be float32");
  TORCH_CHECK(grad_depth_f32.dim() == 2, "grad_depth_f32 must have shape [K,T]");
  TORCH_CHECK(grad_depth_f32.is_contiguous(), "grad_depth_f32 must be contiguous");
  check_i32_mps_1d(config_i32, "config_i32", 6);
  TORCH_CHECK(config_f32.device().is_mps(), "config_f32 must be on MPS");
  TORCH_CHECK(config_f32.scalar_type() == torch::kFloat32, "config_f32 must be float32");
  TORCH_CHECK(config_f32.dim() == 1 && config_f32.size(0) == 4, "config_f32 must have shape [4]");
  TORCH_CHECK(config_f32.is_contiguous(), "config_f32 must be contiguous");

  auto config_i32_cpu = config_i32.cpu();
  const int32_t* config = config_i32_cpu.data_ptr<int32_t>();
  TORCH_CHECK(config[0] == boundary_f32.size(0), "config_i32[0] must match boundary count");
  TORCH_CHECK(config[1] == track_rays_f32.size(0), "config_i32[1] must match track count");
  TORCH_CHECK(config[2] == sites_f32.size(0), "config_i32[2] must match site count");
  TORCH_CHECK(config[3] == frame_t_f32.size(0), "config_i32[3] must match frame count");
  TORCH_CHECK(config[4] > 0, "config_i32[4] time_slab_count must be positive");
  TORCH_CHECK(config[5] > 0, "config_i32[5] mask_word_count must be positive");
  TORCH_CHECK(boundary_f32.size(0) <= 128, "shared real-ray VJP currently supports at most 128 boundaries");
  TORCH_CHECK(sites_f32.size(0) <= 64, "shared real-ray VJP currently supports at most 64 sites");
  TORCH_CHECK(site_rgba_f32.size(0) == sites_f32.size(0), "site RGBA count mismatch");
  TORCH_CHECK(
      candidate_mask_i32.size(0) == track_rays_f32.size(0) * (int64_t)config[4],
      "candidate_mask_i32 row count must be track_count * time_slab_count");
  TORCH_CHECK(candidate_mask_i32.size(1) == config[5], "candidate_mask_i32 column count must match mask_word_count");
  TORCH_CHECK(
      config[5] == (int32_t)((boundary_f32.size(0) + 31) / 32),
      "mask_word_count must equal ceil(boundary_count / 32)");
  TORCH_CHECK(
      grad_rgb_f32.size(0) == track_rays_f32.size(0) && grad_rgb_f32.size(1) == frame_t_f32.size(0),
      "grad_rgb_f32 shape mismatch");
  TORCH_CHECK(
      grad_alpha_f32.size(0) == track_rays_f32.size(0) && grad_alpha_f32.size(1) == frame_t_f32.size(0),
      "grad_alpha_f32 shape mismatch");
  TORCH_CHECK(
      grad_depth_f32.size(0) == track_rays_f32.size(0) && grad_depth_f32.size(1) == frame_t_f32.size(0),
      "grad_depth_f32 shape mismatch");

  auto output_rgb = torch::empty(
      {track_rays_f32.size(0), frame_t_f32.size(0), 3},
      track_rays_f32.options().dtype(torch::kFloat32));
  auto output_alpha = torch::empty(
      {track_rays_f32.size(0), frame_t_f32.size(0)},
      track_rays_f32.options().dtype(torch::kFloat32));
  auto output_depth = torch::empty(
      {track_rays_f32.size(0), frame_t_f32.size(0)},
      track_rays_f32.options().dtype(torch::kFloat32));
  auto grad_samples_rgba = torch::empty(
      {track_rays_f32.size(0), frame_t_f32.size(0), sites_f32.size(0), 4},
      track_rays_f32.options().dtype(torch::kFloat32));
  auto& k = kernels();
  constexpr uint64_t threads = 256ull;
  const uint64_t total = (uint64_t)track_rays_f32.size(0) * (uint64_t)frame_t_f32.size(0);
  launch(k.shared_realray_rgba_depth_vjp, [&](MetalKernelFunction& fn) {
    fn.setArg(0, boundary_f32);
    fn.setArg(1, candidate_mask_i32);
    fn.setArg(2, sites_f32);
    fn.setArg(3, site_rgba_f32);
    fn.setArg(4, track_rays_f32);
    fn.setArg(5, frame_t_f32);
    fn.setArg(6, grad_rgb_f32);
    fn.setArg(7, grad_alpha_f32);
    fn.setArg(8, grad_depth_f32);
    fn.setArg(9, config_i32);
    fn.setArg(10, config_f32);
    fn.setArg(11, output_rgb);
    fn.setArg(12, output_alpha);
    fn.setArg(13, output_depth);
    fn.setArg(14, grad_samples_rgba);
    fn.dispatch(total, threads);
  });
  return std::make_tuple(output_rgb, output_alpha, output_depth, grad_samples_rgba);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> metal_shared_realray_rgba_depth_vjp_reduce(
    const torch::Tensor& boundary_f32,
    const torch::Tensor& candidate_mask_i32,
    const torch::Tensor& sites_f32,
    const torch::Tensor& site_rgba_f32,
    const torch::Tensor& track_rays_f32,
    const torch::Tensor& frame_t_f32,
    const torch::Tensor& grad_rgb_f32,
    const torch::Tensor& grad_alpha_f32,
    const torch::Tensor& grad_depth_f32,
    const torch::Tensor& config_i32,
    const torch::Tensor& config_f32) {
  check_float_mps_2d(boundary_f32, "boundary_f32", 5);
  TORCH_CHECK(candidate_mask_i32.device().is_mps(), "candidate_mask_i32 must be on MPS");
  TORCH_CHECK(candidate_mask_i32.scalar_type() == torch::kInt32, "candidate_mask_i32 must be int32");
  TORCH_CHECK(candidate_mask_i32.dim() == 2, "candidate_mask_i32 must have shape [track_count * time_slab_count, mask_word_count]");
  TORCH_CHECK(candidate_mask_i32.is_contiguous(), "candidate_mask_i32 must be contiguous");
  check_float_mps_2d(sites_f32, "sites_f32", 5);
  check_float_mps_2d(site_rgba_f32, "site_rgba_f32", 4);
  check_float_mps_2d(track_rays_f32, "track_rays_f32", 6);
  TORCH_CHECK(frame_t_f32.device().is_mps(), "frame_t_f32 must be on MPS");
  TORCH_CHECK(frame_t_f32.scalar_type() == torch::kFloat32, "frame_t_f32 must be float32");
  TORCH_CHECK(frame_t_f32.dim() == 1, "frame_t_f32 must have shape [T]");
  TORCH_CHECK(frame_t_f32.is_contiguous(), "frame_t_f32 must be contiguous");
  TORCH_CHECK(grad_rgb_f32.device().is_mps(), "grad_rgb_f32 must be on MPS");
  TORCH_CHECK(grad_rgb_f32.scalar_type() == torch::kFloat32, "grad_rgb_f32 must be float32");
  TORCH_CHECK(grad_rgb_f32.dim() == 3 && grad_rgb_f32.size(2) == 3, "grad_rgb_f32 must have shape [K,T,3]");
  TORCH_CHECK(grad_rgb_f32.is_contiguous(), "grad_rgb_f32 must be contiguous");
  TORCH_CHECK(grad_alpha_f32.device().is_mps(), "grad_alpha_f32 must be on MPS");
  TORCH_CHECK(grad_alpha_f32.scalar_type() == torch::kFloat32, "grad_alpha_f32 must be float32");
  TORCH_CHECK(grad_alpha_f32.dim() == 2, "grad_alpha_f32 must have shape [K,T]");
  TORCH_CHECK(grad_alpha_f32.is_contiguous(), "grad_alpha_f32 must be contiguous");
  TORCH_CHECK(grad_depth_f32.device().is_mps(), "grad_depth_f32 must be on MPS");
  TORCH_CHECK(grad_depth_f32.scalar_type() == torch::kFloat32, "grad_depth_f32 must be float32");
  TORCH_CHECK(grad_depth_f32.dim() == 2, "grad_depth_f32 must have shape [K,T]");
  TORCH_CHECK(grad_depth_f32.is_contiguous(), "grad_depth_f32 must be contiguous");
  check_i32_mps_1d(config_i32, "config_i32", 6);
  TORCH_CHECK(config_f32.device().is_mps(), "config_f32 must be on MPS");
  TORCH_CHECK(config_f32.scalar_type() == torch::kFloat32, "config_f32 must be float32");
  TORCH_CHECK(config_f32.dim() == 1 && config_f32.size(0) == 4, "config_f32 must have shape [4]");
  TORCH_CHECK(config_f32.is_contiguous(), "config_f32 must be contiguous");

  auto config_i32_cpu = config_i32.cpu();
  const int32_t* config = config_i32_cpu.data_ptr<int32_t>();
  TORCH_CHECK(config[0] == boundary_f32.size(0), "config_i32[0] must match boundary count");
  TORCH_CHECK(config[1] == track_rays_f32.size(0), "config_i32[1] must match track count");
  TORCH_CHECK(config[2] == sites_f32.size(0), "config_i32[2] must match site count");
  TORCH_CHECK(config[3] == frame_t_f32.size(0), "config_i32[3] must match frame count");
  TORCH_CHECK(config[4] > 0, "config_i32[4] time_slab_count must be positive");
  TORCH_CHECK(config[5] > 0, "config_i32[5] mask_word_count must be positive");
  TORCH_CHECK(track_rays_f32.size(0) > 0, "shared real-ray reduced VJP requires at least one track");
  TORCH_CHECK(frame_t_f32.size(0) > 0, "shared real-ray reduced VJP requires at least one frame");
  TORCH_CHECK(sites_f32.size(0) > 0, "shared real-ray reduced VJP requires at least one site");
  TORCH_CHECK(boundary_f32.size(0) <= 128, "shared real-ray reduced VJP currently supports at most 128 boundaries");
  TORCH_CHECK(sites_f32.size(0) <= 64, "shared real-ray reduced VJP currently supports at most 64 sites");
  TORCH_CHECK(site_rgba_f32.size(0) == sites_f32.size(0), "site RGBA count mismatch");
  TORCH_CHECK(
      candidate_mask_i32.size(0) == track_rays_f32.size(0) * (int64_t)config[4],
      "candidate_mask_i32 row count must be track_count * time_slab_count");
  TORCH_CHECK(candidate_mask_i32.size(1) == config[5], "candidate_mask_i32 column count must match mask_word_count");
  TORCH_CHECK(
      config[5] == (int32_t)((boundary_f32.size(0) + 31) / 32),
      "mask_word_count must equal ceil(boundary_count / 32)");
  TORCH_CHECK(
      grad_rgb_f32.size(0) == track_rays_f32.size(0) && grad_rgb_f32.size(1) == frame_t_f32.size(0),
      "grad_rgb_f32 shape mismatch");
  TORCH_CHECK(
      grad_alpha_f32.size(0) == track_rays_f32.size(0) && grad_alpha_f32.size(1) == frame_t_f32.size(0),
      "grad_alpha_f32 shape mismatch");
  TORCH_CHECK(
      grad_depth_f32.size(0) == track_rays_f32.size(0) && grad_depth_f32.size(1) == frame_t_f32.size(0),
      "grad_depth_f32 shape mismatch");

  auto output_rgb = torch::empty(
      {track_rays_f32.size(0), frame_t_f32.size(0), 3},
      track_rays_f32.options().dtype(torch::kFloat32));
  auto output_alpha = torch::empty(
      {track_rays_f32.size(0), frame_t_f32.size(0)},
      track_rays_f32.options().dtype(torch::kFloat32));
  auto output_depth = torch::empty(
      {track_rays_f32.size(0), frame_t_f32.size(0)},
      track_rays_f32.options().dtype(torch::kFloat32));
  auto grad_site_rgba = torch::empty({sites_f32.size(0), 4}, sites_f32.options().dtype(torch::kFloat32));
  constexpr uint64_t reduce_chunk_size = 4ull;
  const uint64_t total = (uint64_t)track_rays_f32.size(0) * (uint64_t)frame_t_f32.size(0);
  const uint64_t chunk_count = (total + reduce_chunk_size - 1ull) / reduce_chunk_size;
  auto partial_grad_site_rgba = torch::empty(
      {(int64_t)chunk_count, sites_f32.size(0), 4},
      sites_f32.options().dtype(torch::kFloat32));
  auto& k = kernels();
  constexpr uint64_t threads = 256ull;
  launch(k.shared_realray_rgba_depth_replay, [&](MetalKernelFunction& fn) {
    fn.setArg(0, boundary_f32);
    fn.setArg(1, candidate_mask_i32);
    fn.setArg(2, sites_f32);
    fn.setArg(3, site_rgba_f32);
    fn.setArg(4, track_rays_f32);
    fn.setArg(5, frame_t_f32);
    fn.setArg(6, config_i32);
    fn.setArg(7, config_f32);
    fn.setArg(8, output_rgb);
    fn.setArg(9, output_alpha);
    fn.setArg(10, output_depth);
    fn.dispatch(total, threads);
  });
  launch(k.shared_realray_rgba_depth_vjp_partial_reduce, [&](MetalKernelFunction& fn) {
    fn.setArg(0, boundary_f32);
    fn.setArg(1, candidate_mask_i32);
    fn.setArg(2, sites_f32);
    fn.setArg(3, site_rgba_f32);
    fn.setArg(4, track_rays_f32);
    fn.setArg(5, frame_t_f32);
    fn.setArg(6, grad_rgb_f32);
    fn.setArg(7, grad_alpha_f32);
    fn.setArg(8, grad_depth_f32);
    fn.setArg(9, config_i32);
    fn.setArg(10, config_f32);
    fn.setArg(11, partial_grad_site_rgba);
    fn.dispatch(chunk_count, threads);
  });
  launch(k.shared_realray_rgba_depth_vjp_finalize_reduce, [&](MetalKernelFunction& fn) {
    fn.setArg(0, partial_grad_site_rgba);
    fn.setArg(1, config_i32);
    fn.setArg(2, grad_site_rgba);
    fn.dispatch((uint64_t)sites_f32.size(0), threads);
  });
  return std::make_tuple(output_rgb, output_alpha, output_depth, grad_site_rgba);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> metal_shared_realray_rgba_depth_vjp_reduce_csr(
    const torch::Tensor& boundary_f32,
    const torch::Tensor& row_index_i32,
    const torch::Tensor& candidate_row_offsets_i32,
    const torch::Tensor& candidate_boundary_ids_i32,
    const torch::Tensor& sites_f32,
    const torch::Tensor& site_rgba_f32,
    const torch::Tensor& track_rays_f32,
    const torch::Tensor& frame_t_f32,
    const torch::Tensor& grad_rgb_f32,
    const torch::Tensor& grad_alpha_f32,
    const torch::Tensor& grad_depth_f32,
    const torch::Tensor& config_i32,
    const torch::Tensor& config_f32) {
  check_float_mps_2d(boundary_f32, "boundary_f32", 5);
  check_i32_mps_1d_any(row_index_i32, "row_index_i32");
  check_i32_mps_1d_any(candidate_row_offsets_i32, "candidate_row_offsets_i32");
  check_i32_mps_1d_any(candidate_boundary_ids_i32, "candidate_boundary_ids_i32");
  check_float_mps_2d(sites_f32, "sites_f32", 5);
  check_float_mps_2d(site_rgba_f32, "site_rgba_f32", 4);
  check_float_mps_2d(track_rays_f32, "track_rays_f32", 6);
  TORCH_CHECK(frame_t_f32.device().is_mps(), "frame_t_f32 must be on MPS");
  TORCH_CHECK(frame_t_f32.scalar_type() == torch::kFloat32, "frame_t_f32 must be float32");
  TORCH_CHECK(frame_t_f32.dim() == 1, "frame_t_f32 must have shape [T]");
  TORCH_CHECK(frame_t_f32.is_contiguous(), "frame_t_f32 must be contiguous");
  TORCH_CHECK(grad_rgb_f32.device().is_mps(), "grad_rgb_f32 must be on MPS");
  TORCH_CHECK(grad_rgb_f32.scalar_type() == torch::kFloat32, "grad_rgb_f32 must be float32");
  TORCH_CHECK(grad_rgb_f32.dim() == 3 && grad_rgb_f32.size(2) == 3, "grad_rgb_f32 must have shape [K,T,3]");
  TORCH_CHECK(grad_rgb_f32.is_contiguous(), "grad_rgb_f32 must be contiguous");
  TORCH_CHECK(grad_alpha_f32.device().is_mps(), "grad_alpha_f32 must be on MPS");
  TORCH_CHECK(grad_alpha_f32.scalar_type() == torch::kFloat32, "grad_alpha_f32 must be float32");
  TORCH_CHECK(grad_alpha_f32.dim() == 2, "grad_alpha_f32 must have shape [K,T]");
  TORCH_CHECK(grad_alpha_f32.is_contiguous(), "grad_alpha_f32 must be contiguous");
  TORCH_CHECK(grad_depth_f32.device().is_mps(), "grad_depth_f32 must be on MPS");
  TORCH_CHECK(grad_depth_f32.scalar_type() == torch::kFloat32, "grad_depth_f32 must be float32");
  TORCH_CHECK(grad_depth_f32.dim() == 2, "grad_depth_f32 must have shape [K,T]");
  TORCH_CHECK(grad_depth_f32.is_contiguous(), "grad_depth_f32 must be contiguous");
  check_i32_mps_1d(config_i32, "config_i32", 7);
  TORCH_CHECK(config_f32.device().is_mps(), "config_f32 must be on MPS");
  TORCH_CHECK(config_f32.scalar_type() == torch::kFloat32, "config_f32 must be float32");
  TORCH_CHECK(config_f32.dim() == 1 && config_f32.size(0) == 4, "config_f32 must have shape [4]");
  TORCH_CHECK(config_f32.is_contiguous(), "config_f32 must be contiguous");

  auto config_i32_cpu = config_i32.cpu();
  const int32_t* config = config_i32_cpu.data_ptr<int32_t>();
  TORCH_CHECK(config[0] == boundary_f32.size(0), "config_i32[0] must match boundary count");
  TORCH_CHECK(config[1] == track_rays_f32.size(0), "config_i32[1] must match track count");
  TORCH_CHECK(config[2] == sites_f32.size(0), "config_i32[2] must match site count");
  TORCH_CHECK(config[3] == frame_t_f32.size(0), "config_i32[3] must match frame count");
  TORCH_CHECK(config[4] > 0, "config_i32[4] time_slab_count must be positive");
  TORCH_CHECK(config[5] > 0, "config_i32[5] row_count must be positive");
  TORCH_CHECK(config[6] >= 0, "config_i32[6] candidate_count must be nonnegative");
  TORCH_CHECK(track_rays_f32.size(0) > 0, "shared real-ray CSR reduced VJP requires at least one track");
  TORCH_CHECK(frame_t_f32.size(0) > 0, "shared real-ray CSR reduced VJP requires at least one frame");
  TORCH_CHECK(sites_f32.size(0) > 0, "shared real-ray CSR reduced VJP requires at least one site");
  TORCH_CHECK(boundary_f32.size(0) <= 128, "shared real-ray CSR reduced VJP currently supports at most 128 boundaries");
  TORCH_CHECK(sites_f32.size(0) <= 64, "shared real-ray CSR reduced VJP currently supports at most 64 sites");
  TORCH_CHECK(site_rgba_f32.size(0) == sites_f32.size(0), "site RGBA count mismatch");
  TORCH_CHECK(row_index_i32.size(0) == track_rays_f32.size(0), "row_index_i32 length must match track count");
  TORCH_CHECK(
      candidate_row_offsets_i32.size(0) == (int64_t)config[5] * (int64_t)config[4] + 1,
      "candidate_row_offsets_i32 length must be row_count * time_slab_count + 1");
  TORCH_CHECK(
      candidate_boundary_ids_i32.size(0) == config[6],
      "candidate_boundary_ids_i32 length must match candidate_count");
  TORCH_CHECK(
      grad_rgb_f32.size(0) == track_rays_f32.size(0) && grad_rgb_f32.size(1) == frame_t_f32.size(0),
      "grad_rgb_f32 shape mismatch");
  TORCH_CHECK(
      grad_alpha_f32.size(0) == track_rays_f32.size(0) && grad_alpha_f32.size(1) == frame_t_f32.size(0),
      "grad_alpha_f32 shape mismatch");
  TORCH_CHECK(
      grad_depth_f32.size(0) == track_rays_f32.size(0) && grad_depth_f32.size(1) == frame_t_f32.size(0),
      "grad_depth_f32 shape mismatch");

  auto output_rgb = torch::empty(
      {track_rays_f32.size(0), frame_t_f32.size(0), 3},
      track_rays_f32.options().dtype(torch::kFloat32));
  auto output_alpha = torch::empty(
      {track_rays_f32.size(0), frame_t_f32.size(0)},
      track_rays_f32.options().dtype(torch::kFloat32));
  auto output_depth = torch::empty(
      {track_rays_f32.size(0), frame_t_f32.size(0)},
      track_rays_f32.options().dtype(torch::kFloat32));
  auto grad_site_rgba = torch::empty({sites_f32.size(0), 4}, sites_f32.options().dtype(torch::kFloat32));
  constexpr uint64_t reduce_chunk_size = 4ull;
  const uint64_t total = (uint64_t)track_rays_f32.size(0) * (uint64_t)frame_t_f32.size(0);
  const uint64_t chunk_count = (total + reduce_chunk_size - 1ull) / reduce_chunk_size;
  auto partial_grad_site_rgba = torch::empty(
      {(int64_t)chunk_count, sites_f32.size(0), 4},
      sites_f32.options().dtype(torch::kFloat32));
  auto& k = kernels();
  constexpr uint64_t threads = 256ull;
  launch(k.shared_realray_rgba_depth_vjp_partial_reduce_csr, [&](MetalKernelFunction& fn) {
    fn.setArg(0, boundary_f32);
    fn.setArg(1, row_index_i32);
    fn.setArg(2, candidate_row_offsets_i32);
    fn.setArg(3, candidate_boundary_ids_i32);
    fn.setArg(4, sites_f32);
    fn.setArg(5, site_rgba_f32);
    fn.setArg(6, track_rays_f32);
    fn.setArg(7, frame_t_f32);
    fn.setArg(8, grad_rgb_f32);
    fn.setArg(9, grad_alpha_f32);
    fn.setArg(10, grad_depth_f32);
    fn.setArg(11, config_i32);
    fn.setArg(12, config_f32);
    fn.setArg(13, output_rgb);
    fn.setArg(14, output_alpha);
    fn.setArg(15, output_depth);
    fn.setArg(16, partial_grad_site_rgba);
    fn.dispatch(chunk_count, threads);
  });
  launch(k.shared_realray_rgba_depth_vjp_finalize_reduce, [&](MetalKernelFunction& fn) {
    fn.setArg(0, partial_grad_site_rgba);
    fn.setArg(1, config_i32);
    fn.setArg(2, grad_site_rgba);
    fn.dispatch((uint64_t)sites_f32.size(0), threads);
  });
  return std::make_tuple(output_rgb, output_alpha, output_depth, grad_site_rgba);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
metal_fused_slab_affine_num32_den16_vjp_reduce(
    const torch::Tensor& row_index_i32,
    const torch::Tensor& candidate_row_offsets_i32,
    const torch::Tensor& candidate_depth_num_f32,
    const torch::Tensor& candidate_depth_den_f16,
    const torch::Tensor& sites_f32,
    const torch::Tensor& site_rgba_f32,
    const torch::Tensor& ray_coeff_f32,
    const torch::Tensor& frame_t_f32,
    const torch::Tensor& grad_rgb_f32,
    const torch::Tensor& grad_alpha_f32,
    const torch::Tensor& grad_depth_f32,
    const torch::Tensor& config_i32,
    const torch::Tensor& config_f32) {
  check_i32_mps_1d_any(row_index_i32, "row_index_i32");
  check_i32_mps_1d_any(candidate_row_offsets_i32, "candidate_row_offsets_i32");
  check_float_mps_2d(candidate_depth_num_f32, "candidate_depth_num_f32", 2);
  check_half_mps_2d(candidate_depth_den_f16, "candidate_depth_den_f16", 2);
  check_float_mps_2d(sites_f32, "sites_f32", 5);
  check_float_mps_2d(site_rgba_f32, "site_rgba_f32", 4);
  check_float_mps_2d(ray_coeff_f32, "ray_coeff_f32", 12);
  TORCH_CHECK(frame_t_f32.device().is_mps(), "frame_t_f32 must be on MPS");
  TORCH_CHECK(frame_t_f32.scalar_type() == torch::kFloat32, "frame_t_f32 must be float32");
  TORCH_CHECK(frame_t_f32.dim() == 1, "frame_t_f32 must have shape [T]");
  TORCH_CHECK(frame_t_f32.is_contiguous(), "frame_t_f32 must be contiguous");
  TORCH_CHECK(grad_rgb_f32.device().is_mps(), "grad_rgb_f32 must be on MPS");
  TORCH_CHECK(grad_rgb_f32.scalar_type() == torch::kFloat32, "grad_rgb_f32 must be float32");
  TORCH_CHECK(grad_rgb_f32.dim() == 3 && grad_rgb_f32.size(2) == 3, "grad_rgb_f32 must have shape [K,T,3]");
  TORCH_CHECK(grad_rgb_f32.is_contiguous(), "grad_rgb_f32 must be contiguous");
  TORCH_CHECK(grad_alpha_f32.device().is_mps(), "grad_alpha_f32 must be on MPS");
  TORCH_CHECK(grad_alpha_f32.scalar_type() == torch::kFloat32, "grad_alpha_f32 must be float32");
  TORCH_CHECK(grad_alpha_f32.dim() == 2, "grad_alpha_f32 must have shape [K,T]");
  TORCH_CHECK(grad_alpha_f32.is_contiguous(), "grad_alpha_f32 must be contiguous");
  TORCH_CHECK(grad_depth_f32.device().is_mps(), "grad_depth_f32 must be on MPS");
  TORCH_CHECK(grad_depth_f32.scalar_type() == torch::kFloat32, "grad_depth_f32 must be float32");
  TORCH_CHECK(grad_depth_f32.dim() == 2, "grad_depth_f32 must have shape [K,T]");
  TORCH_CHECK(grad_depth_f32.is_contiguous(), "grad_depth_f32 must be contiguous");
  check_i32_mps_1d(config_i32, "config_i32", 8);
  TORCH_CHECK(config_f32.device().is_mps(), "config_f32 must be on MPS");
  TORCH_CHECK(config_f32.scalar_type() == torch::kFloat32, "config_f32 must be float32");
  TORCH_CHECK(config_f32.dim() == 1 && config_f32.size(0) == 4, "config_f32 must have shape [4]");
  TORCH_CHECK(config_f32.is_contiguous(), "config_f32 must be contiguous");

  auto config_i32_cpu = config_i32.cpu();
  const int32_t* config = config_i32_cpu.data_ptr<int32_t>();
  TORCH_CHECK(config[1] == ray_coeff_f32.size(0), "config_i32[1] must match track count");
  TORCH_CHECK(config[2] == sites_f32.size(0), "config_i32[2] must match site count");
  TORCH_CHECK(config[3] == frame_t_f32.size(0), "config_i32[3] must match frame count");
  TORCH_CHECK(config[4] > 0, "config_i32[4] time_slab_count must be positive");
  TORCH_CHECK(config[5] > 0, "config_i32[5] row_count must be positive");
  TORCH_CHECK(config[6] >= 0, "config_i32[6] candidate_count must be nonnegative");
  TORCH_CHECK(config[7] > 0, "config_i32[7] reduce_chunk_size must be positive");
  TORCH_CHECK(ray_coeff_f32.size(0) > 0, "mixed affine reduced VJP requires at least one track");
  TORCH_CHECK(frame_t_f32.size(0) > 0, "mixed affine reduced VJP requires at least one frame");
  TORCH_CHECK(sites_f32.size(0) > 0, "mixed affine reduced VJP requires at least one site");
  TORCH_CHECK(sites_f32.size(0) <= 64, "mixed affine reduced VJP currently supports at most 64 sites");
  TORCH_CHECK(site_rgba_f32.size(0) == sites_f32.size(0), "site RGBA count mismatch");
  TORCH_CHECK(row_index_i32.size(0) == ray_coeff_f32.size(0), "row_index_i32 length must match track count");
  TORCH_CHECK(
      candidate_row_offsets_i32.size(0) == (int64_t)config[5] * (int64_t)config[4] + 1,
      "candidate_row_offsets_i32 length must be row_count * time_slab_count + 1");
  TORCH_CHECK(
      candidate_depth_num_f32.size(0) == config[6],
      "candidate_depth_num_f32 row count must match candidate_count");
  TORCH_CHECK(
      candidate_depth_den_f16.size(0) == config[6],
      "candidate_depth_den_f16 row count must match candidate_count");
  TORCH_CHECK(
      grad_rgb_f32.size(0) == ray_coeff_f32.size(0) && grad_rgb_f32.size(1) == frame_t_f32.size(0),
      "grad_rgb_f32 shape mismatch");
  TORCH_CHECK(
      grad_alpha_f32.size(0) == ray_coeff_f32.size(0) && grad_alpha_f32.size(1) == frame_t_f32.size(0),
      "grad_alpha_f32 shape mismatch");
  TORCH_CHECK(
      grad_depth_f32.size(0) == ray_coeff_f32.size(0) && grad_depth_f32.size(1) == frame_t_f32.size(0),
      "grad_depth_f32 shape mismatch");

  auto output_rgb = torch::empty(
      {ray_coeff_f32.size(0), frame_t_f32.size(0), 3},
      ray_coeff_f32.options().dtype(torch::kFloat32));
  auto output_alpha = torch::empty(
      {ray_coeff_f32.size(0), frame_t_f32.size(0)},
      ray_coeff_f32.options().dtype(torch::kFloat32));
  auto output_depth = torch::empty(
      {ray_coeff_f32.size(0), frame_t_f32.size(0)},
      ray_coeff_f32.options().dtype(torch::kFloat32));
  auto grad_site_rgba = torch::empty({sites_f32.size(0), 4}, sites_f32.options().dtype(torch::kFloat32));
  const uint64_t reduce_chunk_size = (uint64_t)config[7];
  const uint64_t total = (uint64_t)ray_coeff_f32.size(0) * (uint64_t)frame_t_f32.size(0);
  const uint64_t chunk_count = (total + reduce_chunk_size - 1ull) / reduce_chunk_size;
  auto partial_grad_site_rgba = torch::empty(
      {(int64_t)chunk_count, sites_f32.size(0), 4},
      sites_f32.options().dtype(torch::kFloat32));
  auto& k = kernels();
  constexpr uint64_t threads = 256ull;
  launch(k.fused_slab_affine_num32_den16_vjp_partial_reduce, [&](MetalKernelFunction& fn) {
    fn.setArg(0, row_index_i32);
    fn.setArg(1, candidate_row_offsets_i32);
    fn.setArg(2, candidate_depth_num_f32);
    fn.setArg(3, candidate_depth_den_f16);
    fn.setArg(4, sites_f32);
    fn.setArg(5, site_rgba_f32);
    fn.setArg(6, ray_coeff_f32);
    fn.setArg(7, frame_t_f32);
    fn.setArg(8, grad_rgb_f32);
    fn.setArg(9, grad_alpha_f32);
    fn.setArg(10, grad_depth_f32);
    fn.setArg(11, config_i32);
    fn.setArg(12, config_f32);
    fn.setArg(13, output_rgb);
    fn.setArg(14, output_alpha);
    fn.setArg(15, output_depth);
    fn.setArg(16, partial_grad_site_rgba);
    fn.dispatch(chunk_count, threads);
  });
  launch(k.fused_slab_affine_num32_den16_vjp_finalize_reduce, [&](MetalKernelFunction& fn) {
    fn.setArg(0, partial_grad_site_rgba);
    fn.setArg(1, config_i32);
    fn.setArg(2, grad_site_rgba);
    fn.dispatch((uint64_t)sites_f32.size(0), threads);
  });
  return std::make_tuple(output_rgb, output_alpha, output_depth, grad_site_rgba);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
metal_fused_slab_affine_num32_den16_vjp_direct_atomic(
    const torch::Tensor& row_index_i32,
    const torch::Tensor& candidate_row_offsets_i32,
    const torch::Tensor& candidate_depth_num_f32,
    const torch::Tensor& candidate_depth_den_f16,
    const torch::Tensor& sites_f32,
    const torch::Tensor& site_rgba_f32,
    const torch::Tensor& ray_coeff_f32,
    const torch::Tensor& frame_t_f32,
    const torch::Tensor& grad_rgb_f32,
    const torch::Tensor& grad_alpha_f32,
    const torch::Tensor& grad_depth_f32,
    const torch::Tensor& config_i32,
    const torch::Tensor& config_f32) {
  check_i32_mps_1d_any(row_index_i32, "row_index_i32");
  check_i32_mps_1d_any(candidate_row_offsets_i32, "candidate_row_offsets_i32");
  check_float_mps_2d(candidate_depth_num_f32, "candidate_depth_num_f32", 2);
  check_half_mps_2d(candidate_depth_den_f16, "candidate_depth_den_f16", 2);
  check_float_mps_2d(sites_f32, "sites_f32", 5);
  check_float_mps_2d(site_rgba_f32, "site_rgba_f32", 4);
  check_float_mps_2d(ray_coeff_f32, "ray_coeff_f32", 12);
  TORCH_CHECK(frame_t_f32.device().is_mps(), "frame_t_f32 must be on MPS");
  TORCH_CHECK(frame_t_f32.scalar_type() == torch::kFloat32, "frame_t_f32 must be float32");
  TORCH_CHECK(frame_t_f32.dim() == 1, "frame_t_f32 must have shape [T]");
  TORCH_CHECK(frame_t_f32.is_contiguous(), "frame_t_f32 must be contiguous");
  TORCH_CHECK(grad_rgb_f32.device().is_mps(), "grad_rgb_f32 must be on MPS");
  TORCH_CHECK(grad_rgb_f32.scalar_type() == torch::kFloat32, "grad_rgb_f32 must be float32");
  TORCH_CHECK(grad_rgb_f32.dim() == 3 && grad_rgb_f32.size(2) == 3, "grad_rgb_f32 must have shape [K,T,3]");
  TORCH_CHECK(grad_rgb_f32.is_contiguous(), "grad_rgb_f32 must be contiguous");
  TORCH_CHECK(grad_alpha_f32.device().is_mps(), "grad_alpha_f32 must be on MPS");
  TORCH_CHECK(grad_alpha_f32.scalar_type() == torch::kFloat32, "grad_alpha_f32 must be float32");
  TORCH_CHECK(grad_alpha_f32.dim() == 2, "grad_alpha_f32 must have shape [K,T]");
  TORCH_CHECK(grad_alpha_f32.is_contiguous(), "grad_alpha_f32 must be contiguous");
  TORCH_CHECK(grad_depth_f32.device().is_mps(), "grad_depth_f32 must be on MPS");
  TORCH_CHECK(grad_depth_f32.scalar_type() == torch::kFloat32, "grad_depth_f32 must be float32");
  TORCH_CHECK(grad_depth_f32.dim() == 2, "grad_depth_f32 must have shape [K,T]");
  TORCH_CHECK(grad_depth_f32.is_contiguous(), "grad_depth_f32 must be contiguous");
  check_i32_mps_1d(config_i32, "config_i32", 8);
  TORCH_CHECK(config_f32.device().is_mps(), "config_f32 must be on MPS");
  TORCH_CHECK(config_f32.scalar_type() == torch::kFloat32, "config_f32 must be float32");
  TORCH_CHECK(config_f32.dim() == 1 && config_f32.size(0) == 4, "config_f32 must have shape [4]");
  TORCH_CHECK(config_f32.is_contiguous(), "config_f32 must be contiguous");

  auto config_i32_cpu = config_i32.cpu();
  const int32_t* config = config_i32_cpu.data_ptr<int32_t>();
  TORCH_CHECK(config[1] == ray_coeff_f32.size(0), "config_i32[1] must match track count");
  TORCH_CHECK(config[2] == sites_f32.size(0), "config_i32[2] must match site count");
  TORCH_CHECK(config[3] == frame_t_f32.size(0), "config_i32[3] must match frame count");
  TORCH_CHECK(config[4] > 0, "config_i32[4] time_slab_count must be positive");
  TORCH_CHECK(config[5] > 0, "config_i32[5] row_count must be positive");
  TORCH_CHECK(config[6] >= 0, "config_i32[6] candidate_count must be nonnegative");
  TORCH_CHECK(ray_coeff_f32.size(0) > 0, "mixed affine direct VJP requires at least one track");
  TORCH_CHECK(frame_t_f32.size(0) > 0, "mixed affine direct VJP requires at least one frame");
  TORCH_CHECK(sites_f32.size(0) > 0, "mixed affine direct VJP requires at least one site");
  TORCH_CHECK(sites_f32.size(0) <= 64, "mixed affine direct VJP currently supports at most 64 sites");
  TORCH_CHECK(site_rgba_f32.size(0) == sites_f32.size(0), "site RGBA count mismatch");
  TORCH_CHECK(row_index_i32.size(0) == ray_coeff_f32.size(0), "row_index_i32 length must match track count");
  TORCH_CHECK(
      candidate_row_offsets_i32.size(0) == (int64_t)config[5] * (int64_t)config[4] + 1,
      "candidate_row_offsets_i32 length must be row_count * time_slab_count + 1");
  TORCH_CHECK(
      candidate_depth_num_f32.size(0) == config[6],
      "candidate_depth_num_f32 row count must match candidate_count");
  TORCH_CHECK(
      candidate_depth_den_f16.size(0) == config[6],
      "candidate_depth_den_f16 row count must match candidate_count");
  TORCH_CHECK(
      grad_rgb_f32.size(0) == ray_coeff_f32.size(0) && grad_rgb_f32.size(1) == frame_t_f32.size(0),
      "grad_rgb_f32 shape mismatch");
  TORCH_CHECK(
      grad_alpha_f32.size(0) == ray_coeff_f32.size(0) && grad_alpha_f32.size(1) == frame_t_f32.size(0),
      "grad_alpha_f32 shape mismatch");
  TORCH_CHECK(
      grad_depth_f32.size(0) == ray_coeff_f32.size(0) && grad_depth_f32.size(1) == frame_t_f32.size(0),
      "grad_depth_f32 shape mismatch");

  auto output_rgb = torch::empty(
      {ray_coeff_f32.size(0), frame_t_f32.size(0), 3},
      ray_coeff_f32.options().dtype(torch::kFloat32));
  auto output_alpha = torch::empty(
      {ray_coeff_f32.size(0), frame_t_f32.size(0)},
      ray_coeff_f32.options().dtype(torch::kFloat32));
  auto output_depth = torch::empty(
      {ray_coeff_f32.size(0), frame_t_f32.size(0)},
      ray_coeff_f32.options().dtype(torch::kFloat32));
  auto grad_site_rgba = torch::empty({sites_f32.size(0), 4}, sites_f32.options().dtype(torch::kFloat32));
  const uint64_t total = (uint64_t)ray_coeff_f32.size(0) * (uint64_t)frame_t_f32.size(0);
  auto& k = kernels();
  constexpr uint64_t threads = 256ull;
  launch(k.clear_site_rgba_grad, [&](MetalKernelFunction& fn) {
    fn.setArg(0, grad_site_rgba);
    fn.setArg(1, config_i32);
    fn.dispatch((uint64_t)sites_f32.size(0), threads);
  });
  launch(k.fused_slab_affine_num32_den16_vjp_direct_atomic, [&](MetalKernelFunction& fn) {
    fn.setArg(0, row_index_i32);
    fn.setArg(1, candidate_row_offsets_i32);
    fn.setArg(2, candidate_depth_num_f32);
    fn.setArg(3, candidate_depth_den_f16);
    fn.setArg(4, sites_f32);
    fn.setArg(5, site_rgba_f32);
    fn.setArg(6, ray_coeff_f32);
    fn.setArg(7, frame_t_f32);
    fn.setArg(8, grad_rgb_f32);
    fn.setArg(9, grad_alpha_f32);
    fn.setArg(10, grad_depth_f32);
    fn.setArg(11, config_i32);
    fn.setArg(12, config_f32);
    fn.setArg(13, output_rgb);
    fn.setArg(14, output_alpha);
    fn.setArg(15, output_depth);
    fn.setArg(16, grad_site_rgba);
    fn.dispatch(total, threads);
  });
  return std::make_tuple(output_rgb, output_alpha, output_depth, grad_site_rgba);
}

torch::Tensor metal_fused_slab_affine_num32_den16_vjp_direct_atomic_grad_only(
    const torch::Tensor& row_index_i32,
    const torch::Tensor& candidate_row_offsets_i32,
    const torch::Tensor& candidate_depth_num_f32,
    const torch::Tensor& candidate_depth_den_f16,
    const torch::Tensor& sites_f32,
    const torch::Tensor& site_rgba_f32,
    const torch::Tensor& ray_coeff_f32,
    const torch::Tensor& frame_t_f32,
    const torch::Tensor& grad_rgb_f32,
    const torch::Tensor& grad_alpha_f32,
    const torch::Tensor& grad_depth_f32,
    const torch::Tensor& config_i32,
    const torch::Tensor& config_f32) {
  check_i32_mps_1d_any(row_index_i32, "row_index_i32");
  check_i32_mps_1d_any(candidate_row_offsets_i32, "candidate_row_offsets_i32");
  check_float_mps_2d(candidate_depth_num_f32, "candidate_depth_num_f32", 2);
  check_half_mps_2d(candidate_depth_den_f16, "candidate_depth_den_f16", 2);
  check_float_mps_2d(sites_f32, "sites_f32", 5);
  check_float_mps_2d(site_rgba_f32, "site_rgba_f32", 4);
  check_float_mps_2d(ray_coeff_f32, "ray_coeff_f32", 12);
  TORCH_CHECK(frame_t_f32.device().is_mps(), "frame_t_f32 must be on MPS");
  TORCH_CHECK(frame_t_f32.scalar_type() == torch::kFloat32, "frame_t_f32 must be float32");
  TORCH_CHECK(frame_t_f32.dim() == 1, "frame_t_f32 must have shape [T]");
  TORCH_CHECK(frame_t_f32.is_contiguous(), "frame_t_f32 must be contiguous");
  TORCH_CHECK(grad_rgb_f32.device().is_mps(), "grad_rgb_f32 must be on MPS");
  TORCH_CHECK(grad_rgb_f32.scalar_type() == torch::kFloat32, "grad_rgb_f32 must be float32");
  TORCH_CHECK(grad_rgb_f32.dim() == 3 && grad_rgb_f32.size(2) == 3, "grad_rgb_f32 must have shape [K,T,3]");
  TORCH_CHECK(grad_rgb_f32.is_contiguous(), "grad_rgb_f32 must be contiguous");
  TORCH_CHECK(grad_alpha_f32.device().is_mps(), "grad_alpha_f32 must be on MPS");
  TORCH_CHECK(grad_alpha_f32.scalar_type() == torch::kFloat32, "grad_alpha_f32 must be float32");
  TORCH_CHECK(grad_alpha_f32.dim() == 2, "grad_alpha_f32 must have shape [K,T]");
  TORCH_CHECK(grad_alpha_f32.is_contiguous(), "grad_alpha_f32 must be contiguous");
  TORCH_CHECK(grad_depth_f32.device().is_mps(), "grad_depth_f32 must be on MPS");
  TORCH_CHECK(grad_depth_f32.scalar_type() == torch::kFloat32, "grad_depth_f32 must be float32");
  TORCH_CHECK(grad_depth_f32.dim() == 2, "grad_depth_f32 must have shape [K,T]");
  TORCH_CHECK(grad_depth_f32.is_contiguous(), "grad_depth_f32 must be contiguous");
  check_i32_mps_1d(config_i32, "config_i32", 8);
  TORCH_CHECK(config_f32.device().is_mps(), "config_f32 must be on MPS");
  TORCH_CHECK(config_f32.scalar_type() == torch::kFloat32, "config_f32 must be float32");
  TORCH_CHECK(config_f32.dim() == 1 && config_f32.size(0) == 4, "config_f32 must have shape [4]");
  TORCH_CHECK(config_f32.is_contiguous(), "config_f32 must be contiguous");

  auto config_i32_cpu = config_i32.cpu();
  const int32_t* config = config_i32_cpu.data_ptr<int32_t>();
  TORCH_CHECK(config[1] == ray_coeff_f32.size(0), "config_i32[1] must match track count");
  TORCH_CHECK(config[2] == sites_f32.size(0), "config_i32[2] must match site count");
  TORCH_CHECK(config[3] == frame_t_f32.size(0), "config_i32[3] must match frame count");
  TORCH_CHECK(config[4] > 0, "config_i32[4] time_slab_count must be positive");
  TORCH_CHECK(config[5] > 0, "config_i32[5] row_count must be positive");
  TORCH_CHECK(config[6] >= 0, "config_i32[6] candidate_count must be nonnegative");
  TORCH_CHECK(ray_coeff_f32.size(0) > 0, "mixed affine grad-only direct VJP requires at least one track");
  TORCH_CHECK(frame_t_f32.size(0) > 0, "mixed affine grad-only direct VJP requires at least one frame");
  TORCH_CHECK(sites_f32.size(0) > 0, "mixed affine grad-only direct VJP requires at least one site");
  TORCH_CHECK(sites_f32.size(0) <= 64, "mixed affine grad-only direct VJP currently supports at most 64 sites");
  TORCH_CHECK(site_rgba_f32.size(0) == sites_f32.size(0), "site RGBA count mismatch");
  TORCH_CHECK(row_index_i32.size(0) == ray_coeff_f32.size(0), "row_index_i32 length must match track count");
  TORCH_CHECK(
      candidate_row_offsets_i32.size(0) == (int64_t)config[5] * (int64_t)config[4] + 1,
      "candidate_row_offsets_i32 length must be row_count * time_slab_count + 1");
  TORCH_CHECK(
      candidate_depth_num_f32.size(0) == config[6],
      "candidate_depth_num_f32 row count must match candidate_count");
  TORCH_CHECK(
      candidate_depth_den_f16.size(0) == config[6],
      "candidate_depth_den_f16 row count must match candidate_count");
  TORCH_CHECK(
      grad_rgb_f32.size(0) == ray_coeff_f32.size(0) && grad_rgb_f32.size(1) == frame_t_f32.size(0),
      "grad_rgb_f32 shape mismatch");
  TORCH_CHECK(
      grad_alpha_f32.size(0) == ray_coeff_f32.size(0) && grad_alpha_f32.size(1) == frame_t_f32.size(0),
      "grad_alpha_f32 shape mismatch");
  TORCH_CHECK(
      grad_depth_f32.size(0) == ray_coeff_f32.size(0) && grad_depth_f32.size(1) == frame_t_f32.size(0),
      "grad_depth_f32 shape mismatch");

  auto grad_site_rgba = torch::empty({sites_f32.size(0), 4}, sites_f32.options().dtype(torch::kFloat32));
  const uint64_t total = (uint64_t)ray_coeff_f32.size(0) * (uint64_t)frame_t_f32.size(0);
  auto& k = kernels();
  constexpr uint64_t threads = 256ull;
  launch(k.clear_site_rgba_grad, [&](MetalKernelFunction& fn) {
    fn.setArg(0, grad_site_rgba);
    fn.setArg(1, config_i32);
    fn.dispatch((uint64_t)sites_f32.size(0), threads);
  });
  launch(k.fused_slab_affine_num32_den16_vjp_direct_atomic_grad_only, [&](MetalKernelFunction& fn) {
    fn.setArg(0, row_index_i32);
    fn.setArg(1, candidate_row_offsets_i32);
    fn.setArg(2, candidate_depth_num_f32);
    fn.setArg(3, candidate_depth_den_f16);
    fn.setArg(4, sites_f32);
    fn.setArg(5, site_rgba_f32);
    fn.setArg(6, ray_coeff_f32);
    fn.setArg(7, frame_t_f32);
    fn.setArg(8, grad_rgb_f32);
    fn.setArg(9, grad_alpha_f32);
    fn.setArg(10, grad_depth_f32);
    fn.setArg(11, config_i32);
    fn.setArg(12, config_f32);
    fn.setArg(13, grad_site_rgba);
    fn.dispatch(total, threads);
  });
  return grad_site_rgba;
}

torch::Tensor metal_fused_slab_affine_num32_den16_vjp_direct_atomic_grad_only_ownerupdate(
    const torch::Tensor& row_index_i32,
    const torch::Tensor& candidate_row_offsets_i32,
    const torch::Tensor& candidate_boundary_ids_i32,
    const torch::Tensor& candidate_depth_num_f32,
    const torch::Tensor& candidate_depth_den_f16,
    const torch::Tensor& boundary_site_pairs_i32,
    const torch::Tensor& sites_f32,
    const torch::Tensor& site_rgba_f32,
    const torch::Tensor& ray_coeff_f32,
    const torch::Tensor& frame_t_f32,
    const torch::Tensor& grad_rgb_f32,
    const torch::Tensor& grad_alpha_f32,
    const torch::Tensor& grad_depth_f32,
    const torch::Tensor& config_i32,
    const torch::Tensor& config_f32) {
  check_i32_mps_1d_any(row_index_i32, "row_index_i32");
  check_i32_mps_1d_any(candidate_row_offsets_i32, "candidate_row_offsets_i32");
  check_i32_mps_1d_any(candidate_boundary_ids_i32, "candidate_boundary_ids_i32");
  check_float_mps_2d(candidate_depth_num_f32, "candidate_depth_num_f32", 2);
  check_half_mps_2d(candidate_depth_den_f16, "candidate_depth_den_f16", 2);
  check_i32_mps_2d(boundary_site_pairs_i32, "boundary_site_pairs_i32", 2);
  check_float_mps_2d(sites_f32, "sites_f32", 5);
  check_float_mps_2d(site_rgba_f32, "site_rgba_f32", 4);
  check_float_mps_2d(ray_coeff_f32, "ray_coeff_f32", 12);
  TORCH_CHECK(frame_t_f32.device().is_mps(), "frame_t_f32 must be on MPS");
  TORCH_CHECK(frame_t_f32.scalar_type() == torch::kFloat32, "frame_t_f32 must be float32");
  TORCH_CHECK(frame_t_f32.dim() == 1, "frame_t_f32 must have shape [T]");
  TORCH_CHECK(frame_t_f32.is_contiguous(), "frame_t_f32 must be contiguous");
  TORCH_CHECK(grad_rgb_f32.device().is_mps(), "grad_rgb_f32 must be on MPS");
  TORCH_CHECK(grad_rgb_f32.scalar_type() == torch::kFloat32, "grad_rgb_f32 must be float32");
  TORCH_CHECK(grad_rgb_f32.dim() == 3 && grad_rgb_f32.size(2) == 3, "grad_rgb_f32 must have shape [K,T,3]");
  TORCH_CHECK(grad_rgb_f32.is_contiguous(), "grad_rgb_f32 must be contiguous");
  TORCH_CHECK(grad_alpha_f32.device().is_mps(), "grad_alpha_f32 must be on MPS");
  TORCH_CHECK(grad_alpha_f32.scalar_type() == torch::kFloat32, "grad_alpha_f32 must be float32");
  TORCH_CHECK(grad_alpha_f32.dim() == 2, "grad_alpha_f32 must have shape [K,T]");
  TORCH_CHECK(grad_alpha_f32.is_contiguous(), "grad_alpha_f32 must be contiguous");
  TORCH_CHECK(grad_depth_f32.device().is_mps(), "grad_depth_f32 must be on MPS");
  TORCH_CHECK(grad_depth_f32.scalar_type() == torch::kFloat32, "grad_depth_f32 must be float32");
  TORCH_CHECK(grad_depth_f32.dim() == 2, "grad_depth_f32 must have shape [K,T]");
  TORCH_CHECK(grad_depth_f32.is_contiguous(), "grad_depth_f32 must be contiguous");
  check_i32_mps_1d(config_i32, "config_i32", 8);
  TORCH_CHECK(config_f32.device().is_mps(), "config_f32 must be on MPS");
  TORCH_CHECK(config_f32.scalar_type() == torch::kFloat32, "config_f32 must be float32");
  TORCH_CHECK(config_f32.dim() == 1 && config_f32.size(0) == 4, "config_f32 must have shape [4]");
  TORCH_CHECK(config_f32.is_contiguous(), "config_f32 must be contiguous");

  auto config_i32_cpu = config_i32.cpu();
  const int32_t* config = config_i32_cpu.data_ptr<int32_t>();
  TORCH_CHECK(config[1] == ray_coeff_f32.size(0), "config_i32[1] must match track count");
  TORCH_CHECK(config[2] == sites_f32.size(0), "config_i32[2] must match site count");
  TORCH_CHECK(config[3] == frame_t_f32.size(0), "config_i32[3] must match frame count");
  TORCH_CHECK(config[4] > 0, "config_i32[4] time_slab_count must be positive");
  TORCH_CHECK(config[5] > 0, "config_i32[5] row_count must be positive");
  TORCH_CHECK(config[6] >= 0, "config_i32[6] candidate_count must be nonnegative");
  TORCH_CHECK(config[7] == boundary_site_pairs_i32.size(0), "config_i32[7] must match boundary count");
  TORCH_CHECK(ray_coeff_f32.size(0) > 0, "mixed affine ownerupdate VJP requires at least one track");
  TORCH_CHECK(frame_t_f32.size(0) > 0, "mixed affine ownerupdate VJP requires at least one frame");
  TORCH_CHECK(sites_f32.size(0) > 0, "mixed affine ownerupdate VJP requires at least one site");
  TORCH_CHECK(sites_f32.size(0) <= 64, "mixed affine ownerupdate VJP currently supports at most 64 sites");
  TORCH_CHECK(site_rgba_f32.size(0) == sites_f32.size(0), "site RGBA count mismatch");
  TORCH_CHECK(row_index_i32.size(0) == ray_coeff_f32.size(0), "row_index_i32 length must match track count");
  TORCH_CHECK(
      candidate_row_offsets_i32.size(0) == (int64_t)config[5] * (int64_t)config[4] + 1,
      "candidate_row_offsets_i32 length must be row_count * time_slab_count + 1");
  TORCH_CHECK(candidate_boundary_ids_i32.size(0) == config[6], "candidate_boundary_ids_i32 length mismatch");
  TORCH_CHECK(candidate_depth_num_f32.size(0) == config[6], "candidate_depth_num_f32 row count mismatch");
  TORCH_CHECK(candidate_depth_den_f16.size(0) == config[6], "candidate_depth_den_f16 row count mismatch");
  TORCH_CHECK(
      grad_rgb_f32.size(0) == ray_coeff_f32.size(0) && grad_rgb_f32.size(1) == frame_t_f32.size(0),
      "grad_rgb_f32 shape mismatch");
  TORCH_CHECK(
      grad_alpha_f32.size(0) == ray_coeff_f32.size(0) && grad_alpha_f32.size(1) == frame_t_f32.size(0),
      "grad_alpha_f32 shape mismatch");
  TORCH_CHECK(
      grad_depth_f32.size(0) == ray_coeff_f32.size(0) && grad_depth_f32.size(1) == frame_t_f32.size(0),
      "grad_depth_f32 shape mismatch");

  auto grad_site_rgba = torch::empty({sites_f32.size(0), 4}, sites_f32.options().dtype(torch::kFloat32));
  const uint64_t total = (uint64_t)ray_coeff_f32.size(0) * (uint64_t)frame_t_f32.size(0);
  auto& k = kernels();
  constexpr uint64_t threads = 256ull;
  launch(k.clear_site_rgba_grad, [&](MetalKernelFunction& fn) {
    fn.setArg(0, grad_site_rgba);
    fn.setArg(1, config_i32);
    fn.dispatch((uint64_t)sites_f32.size(0), threads);
  });
  launch(k.fused_slab_affine_num32_den16_vjp_direct_atomic_grad_only_ownerupdate, [&](MetalKernelFunction& fn) {
    fn.setArg(0, row_index_i32);
    fn.setArg(1, candidate_row_offsets_i32);
    fn.setArg(2, candidate_boundary_ids_i32);
    fn.setArg(3, candidate_depth_num_f32);
    fn.setArg(4, candidate_depth_den_f16);
    fn.setArg(5, boundary_site_pairs_i32);
    fn.setArg(6, sites_f32);
    fn.setArg(7, site_rgba_f32);
    fn.setArg(8, ray_coeff_f32);
    fn.setArg(9, frame_t_f32);
    fn.setArg(10, grad_rgb_f32);
    fn.setArg(11, grad_alpha_f32);
    fn.setArg(12, grad_depth_f32);
    fn.setArg(13, config_i32);
    fn.setArg(14, config_f32);
    fn.setArg(15, grad_site_rgba);
    fn.dispatch(total, threads);
  });
  return grad_site_rgba;
}

torch::Tensor metal_endpoint_record_edit_block_coeff16_vjp_direct_atomic_rgb_only(
    const torch::Tensor& coeff_f16,
    const torch::Tensor& frame_t_f32,
    const torch::Tensor& anchor_offsets_i32,
    const torch::Tensor& anchor_owner_i32,
    const torch::Tensor& anchor_left_i32,
    const torch::Tensor& anchor_right_i32,
    const torch::Tensor& track_block_change_offsets_i32,
    const torch::Tensor& change_frame_i32,
    const torch::Tensor& op_offsets_i32,
    const torch::Tensor& op_type_i32,
    const torch::Tensor& op_pos_i32,
    const torch::Tensor& op_owner_i32,
    const torch::Tensor& op_left_i32,
    const torch::Tensor& op_right_i32,
    const torch::Tensor& site_rgba_f32,
    const torch::Tensor& grad_rgb_f32,
    const torch::Tensor& config_i32,
    const torch::Tensor& config_f32) {
  check_half_mps_2d(coeff_f16, "coeff_f16", 4);
  check_float_mps_1d_any(frame_t_f32, "frame_t_f32");
  check_i32_mps_1d_any(anchor_offsets_i32, "anchor_offsets_i32");
  check_i32_mps_1d_any(anchor_owner_i32, "anchor_owner_i32");
  check_i32_mps_1d_any(anchor_left_i32, "anchor_left_i32");
  check_i32_mps_1d_any(anchor_right_i32, "anchor_right_i32");
  check_i32_mps_1d_any(track_block_change_offsets_i32, "track_block_change_offsets_i32");
  check_i32_mps_1d_any(change_frame_i32, "change_frame_i32");
  check_i32_mps_1d_any(op_offsets_i32, "op_offsets_i32");
  check_i32_mps_1d_any(op_type_i32, "op_type_i32");
  check_i32_mps_1d_any(op_pos_i32, "op_pos_i32");
  check_i32_mps_1d_any(op_owner_i32, "op_owner_i32");
  check_i32_mps_1d_any(op_left_i32, "op_left_i32");
  check_i32_mps_1d_any(op_right_i32, "op_right_i32");
  check_float_mps_2d(site_rgba_f32, "site_rgba_f32", 4);
  TORCH_CHECK(grad_rgb_f32.device().is_mps(), "grad_rgb_f32 must be on MPS");
  TORCH_CHECK(grad_rgb_f32.scalar_type() == torch::kFloat32, "grad_rgb_f32 must be float32");
  TORCH_CHECK(grad_rgb_f32.dim() == 3 && grad_rgb_f32.size(2) == 3, "grad_rgb_f32 must have shape [K,T,3]");
  TORCH_CHECK(grad_rgb_f32.is_contiguous(), "grad_rgb_f32 must be contiguous");
  check_i32_mps_1d(config_i32, "config_i32", 9);
  TORCH_CHECK(config_f32.device().is_mps(), "config_f32 must be on MPS");
  TORCH_CHECK(config_f32.scalar_type() == torch::kFloat32, "config_f32 must be float32");
  TORCH_CHECK(config_f32.dim() == 1 && config_f32.size(0) == 4, "config_f32 must have shape [4]");
  TORCH_CHECK(config_f32.is_contiguous(), "config_f32 must be contiguous");

  auto config_i32_cpu = config_i32.cpu();
  const int32_t* config = config_i32_cpu.data_ptr<int32_t>();
  const int64_t boundary_count = config[0];
  const int64_t track_count = config[1];
  const int64_t frame_count = config[2];
  const int64_t site_count = config[3];
  const int64_t anchor_record_count = config[4];
  const int64_t change_count = config[5];
  const int64_t op_count = config[6];
  const int64_t block_size = config[7];
  const int64_t block_count = config[8];
  TORCH_CHECK(boundary_count > 0, "config_i32[0] boundary count must be positive");
  TORCH_CHECK(track_count > 0, "config_i32[1] track count must be positive");
  TORCH_CHECK(frame_count > 0, "config_i32[2] frame count must be positive");
  TORCH_CHECK(site_count == site_rgba_f32.size(0), "config_i32[3] must match site_rgba_f32 rows");
  TORCH_CHECK(site_count > 0 && site_count <= 64, "endpoint record block coeff16 RGB-only VJP supports site count in [1, 64]");
  TORCH_CHECK(anchor_record_count >= 0, "config_i32[4] anchor record count must be nonnegative");
  TORCH_CHECK(change_count >= 0, "config_i32[5] change count must be nonnegative");
  TORCH_CHECK(op_count >= 0, "config_i32[6] op count must be nonnegative");
  TORCH_CHECK(block_size > 0, "endpoint_record_edit_block_coeff16_vjp_direct_atomic_rgb_only requires positive block size");
  TORCH_CHECK(block_count == (frame_count + block_size - 1) / block_size, "config_i32[8] block count mismatch");
  TORCH_CHECK(coeff_f16.size(0) == track_count * boundary_count, "coeff_f16 row count mismatch");
  TORCH_CHECK(frame_t_f32.size(0) == frame_count, "frame_t_f32 length must match frame count");
  TORCH_CHECK(anchor_owner_i32.size(0) == anchor_record_count, "anchor_owner_i32 length must match anchor record count");
  TORCH_CHECK(anchor_left_i32.size(0) == anchor_record_count, "anchor_left_i32 length must match anchor record count");
  TORCH_CHECK(anchor_right_i32.size(0) == anchor_record_count, "anchor_right_i32 length must match anchor record count");
  TORCH_CHECK(change_frame_i32.size(0) == change_count, "change_frame_i32 length must match change count");
  TORCH_CHECK(op_type_i32.size(0) == op_count, "op_type_i32 length must match op count");
  TORCH_CHECK(op_pos_i32.size(0) == op_count, "op_pos_i32 length must match op count");
  TORCH_CHECK(op_owner_i32.size(0) == op_count, "op_owner_i32 length must match op count");
  TORCH_CHECK(op_left_i32.size(0) == op_count, "op_left_i32 length must match op count");
  TORCH_CHECK(op_right_i32.size(0) == op_count, "op_right_i32 length must match op count");
  TORCH_CHECK(anchor_offsets_i32.size(0) == track_count * block_count + 1, "anchor_offsets_i32 length mismatch");
  TORCH_CHECK(
      track_block_change_offsets_i32.size(0) == track_count * (block_count + 1),
      "track_block_change_offsets_i32 length mismatch");
  TORCH_CHECK(op_offsets_i32.size(0) == change_count + 1, "op_offsets_i32 length must be change_count + 1");
  TORCH_CHECK(
      grad_rgb_f32.size(0) == track_count && grad_rgb_f32.size(1) == frame_count,
      "grad_rgb_f32 shape mismatch");
  check_segment_tape_offsets_cpu(anchor_offsets_i32, track_count * block_count, anchor_record_count, 2147483647);
  check_segment_tape_offsets_cpu(
      track_block_change_offsets_i32, track_count * (block_count + 1) - 1, change_count, 2147483647);
  check_segment_tape_offsets_cpu(op_offsets_i32, change_count, op_count, 2147483647);

  auto grad_site_rgba = torch::zeros({site_count, 4}, site_rgba_f32.options().dtype(torch::kFloat32));
  auto& k = kernels();
  constexpr uint64_t threads = 256ull;
  const int64_t sample_count = track_count * frame_count;
  launch(k.endpoint_record_edit_block_coeff16_vjp_direct_atomic_rgb_only, [&](MetalKernelFunction& fn) {
    fn.setArg(0, coeff_f16);
    fn.setArg(1, frame_t_f32);
    fn.setArg(2, anchor_offsets_i32);
    fn.setArg(3, anchor_owner_i32);
    fn.setArg(4, anchor_left_i32);
    fn.setArg(5, anchor_right_i32);
    fn.setArg(6, track_block_change_offsets_i32);
    fn.setArg(7, change_frame_i32);
    fn.setArg(8, op_offsets_i32);
    fn.setArg(9, op_type_i32);
    fn.setArg(10, op_pos_i32);
    fn.setArg(11, op_owner_i32);
    fn.setArg(12, op_left_i32);
    fn.setArg(13, op_right_i32);
    fn.setArg(14, site_rgba_f32);
    fn.setArg(15, grad_rgb_f32);
    fn.setArg(16, config_i32);
    fn.setArg(17, config_f32);
    fn.setArg(18, grad_site_rgba);
    fn.dispatch((uint64_t)sample_count, threads);
  });
  return grad_site_rgba;
}

torch::Tensor metal_fused_slab_affine_num32_den16_vjp_direct_atomic_rgb_only(
    const torch::Tensor& row_index_i32,
    const torch::Tensor& candidate_row_offsets_i32,
    const torch::Tensor& candidate_depth_num_f32,
    const torch::Tensor& candidate_depth_den_f16,
    const torch::Tensor& sites_f32,
    const torch::Tensor& site_rgba_f32,
    const torch::Tensor& ray_coeff_f32,
    const torch::Tensor& frame_t_f32,
    const torch::Tensor& grad_rgb_f32,
    const torch::Tensor& config_i32,
    const torch::Tensor& config_f32) {
  check_i32_mps_1d_any(row_index_i32, "row_index_i32");
  check_i32_mps_1d_any(candidate_row_offsets_i32, "candidate_row_offsets_i32");
  check_float_mps_2d(candidate_depth_num_f32, "candidate_depth_num_f32", 2);
  check_half_mps_2d(candidate_depth_den_f16, "candidate_depth_den_f16", 2);
  check_float_mps_2d(sites_f32, "sites_f32", 5);
  check_float_mps_2d(site_rgba_f32, "site_rgba_f32", 4);
  check_float_mps_2d(ray_coeff_f32, "ray_coeff_f32", 12);
  TORCH_CHECK(frame_t_f32.device().is_mps(), "frame_t_f32 must be on MPS");
  TORCH_CHECK(frame_t_f32.scalar_type() == torch::kFloat32, "frame_t_f32 must be float32");
  TORCH_CHECK(frame_t_f32.dim() == 1, "frame_t_f32 must have shape [T]");
  TORCH_CHECK(frame_t_f32.is_contiguous(), "frame_t_f32 must be contiguous");
  TORCH_CHECK(grad_rgb_f32.device().is_mps(), "grad_rgb_f32 must be on MPS");
  TORCH_CHECK(grad_rgb_f32.scalar_type() == torch::kFloat32, "grad_rgb_f32 must be float32");
  TORCH_CHECK(grad_rgb_f32.dim() == 3 && grad_rgb_f32.size(2) == 3, "grad_rgb_f32 must have shape [K,T,3]");
  TORCH_CHECK(grad_rgb_f32.is_contiguous(), "grad_rgb_f32 must be contiguous");
  check_i32_mps_1d(config_i32, "config_i32", 8);
  TORCH_CHECK(config_f32.device().is_mps(), "config_f32 must be on MPS");
  TORCH_CHECK(config_f32.scalar_type() == torch::kFloat32, "config_f32 must be float32");
  TORCH_CHECK(config_f32.dim() == 1 && config_f32.size(0) == 4, "config_f32 must have shape [4]");
  TORCH_CHECK(config_f32.is_contiguous(), "config_f32 must be contiguous");

  auto config_i32_cpu = config_i32.cpu();
  const int32_t* config = config_i32_cpu.data_ptr<int32_t>();
  TORCH_CHECK(config[1] == ray_coeff_f32.size(0), "config_i32[1] must match track count");
  TORCH_CHECK(config[2] == sites_f32.size(0), "config_i32[2] must match site count");
  TORCH_CHECK(config[3] == frame_t_f32.size(0), "config_i32[3] must match frame count");
  TORCH_CHECK(config[4] > 0, "config_i32[4] time_slab_count must be positive");
  TORCH_CHECK(config[5] > 0, "config_i32[5] row_count must be positive");
  TORCH_CHECK(config[6] >= 0, "config_i32[6] candidate_count must be nonnegative");
  TORCH_CHECK(ray_coeff_f32.size(0) > 0, "mixed affine rgb-only direct VJP requires at least one track");
  TORCH_CHECK(frame_t_f32.size(0) > 0, "mixed affine rgb-only direct VJP requires at least one frame");
  TORCH_CHECK(sites_f32.size(0) > 0, "mixed affine rgb-only direct VJP requires at least one site");
  TORCH_CHECK(sites_f32.size(0) <= 64, "mixed affine rgb-only direct VJP currently supports at most 64 sites");
  TORCH_CHECK(site_rgba_f32.size(0) == sites_f32.size(0), "site RGBA count mismatch");
  TORCH_CHECK(row_index_i32.size(0) == ray_coeff_f32.size(0), "row_index_i32 length must match track count");
  TORCH_CHECK(
      candidate_row_offsets_i32.size(0) == (int64_t)config[5] * (int64_t)config[4] + 1,
      "candidate_row_offsets_i32 length must be row_count * time_slab_count + 1");
  TORCH_CHECK(
      candidate_depth_num_f32.size(0) == config[6],
      "candidate_depth_num_f32 row count must match candidate_count");
  TORCH_CHECK(
      candidate_depth_den_f16.size(0) == config[6],
      "candidate_depth_den_f16 row count must match candidate_count");
  TORCH_CHECK(
      grad_rgb_f32.size(0) == ray_coeff_f32.size(0) && grad_rgb_f32.size(1) == frame_t_f32.size(0),
      "grad_rgb_f32 shape mismatch");

  auto grad_site_rgba = torch::empty({sites_f32.size(0), 4}, sites_f32.options().dtype(torch::kFloat32));
  const uint64_t total = (uint64_t)ray_coeff_f32.size(0) * (uint64_t)frame_t_f32.size(0);
  auto& k = kernels();
  constexpr uint64_t threads = 256ull;
  launch(k.clear_site_rgba_grad, [&](MetalKernelFunction& fn) {
    fn.setArg(0, grad_site_rgba);
    fn.setArg(1, config_i32);
    fn.dispatch((uint64_t)sites_f32.size(0), threads);
  });
  launch(k.fused_slab_affine_num32_den16_vjp_direct_atomic_rgb_only, [&](MetalKernelFunction& fn) {
    fn.setArg(0, row_index_i32);
    fn.setArg(1, candidate_row_offsets_i32);
    fn.setArg(2, candidate_depth_num_f32);
    fn.setArg(3, candidate_depth_den_f16);
    fn.setArg(4, sites_f32);
    fn.setArg(5, site_rgba_f32);
    fn.setArg(6, ray_coeff_f32);
    fn.setArg(7, frame_t_f32);
    fn.setArg(8, grad_rgb_f32);
    fn.setArg(9, config_i32);
    fn.setArg(10, config_f32);
    fn.setArg(11, grad_site_rgba);
    fn.dispatch(total, threads);
  });
  return grad_site_rgba;
}

std::tuple<torch::Tensor, torch::Tensor> metal_fused_slab_affine_num32_den16_mse_vjp_direct_atomic_rgb_only(
    const torch::Tensor& row_index_i32,
    const torch::Tensor& candidate_row_offsets_i32,
    const torch::Tensor& candidate_depth_num_f32,
    const torch::Tensor& candidate_depth_den_f16,
    const torch::Tensor& sites_f32,
    const torch::Tensor& site_rgba_f32,
    const torch::Tensor& ray_coeff_f32,
    const torch::Tensor& frame_t_f32,
    const torch::Tensor& target_rgb_f32,
    const torch::Tensor& config_i32,
    const torch::Tensor& config_f32) {
  check_i32_mps_1d_any(row_index_i32, "row_index_i32");
  check_i32_mps_1d_any(candidate_row_offsets_i32, "candidate_row_offsets_i32");
  check_float_mps_2d(candidate_depth_num_f32, "candidate_depth_num_f32", 2);
  check_half_mps_2d(candidate_depth_den_f16, "candidate_depth_den_f16", 2);
  check_float_mps_2d(sites_f32, "sites_f32", 5);
  check_float_mps_2d(site_rgba_f32, "site_rgba_f32", 4);
  check_float_mps_2d(ray_coeff_f32, "ray_coeff_f32", 12);
  TORCH_CHECK(frame_t_f32.device().is_mps(), "frame_t_f32 must be on MPS");
  TORCH_CHECK(frame_t_f32.scalar_type() == torch::kFloat32, "frame_t_f32 must be float32");
  TORCH_CHECK(frame_t_f32.dim() == 1, "frame_t_f32 must have shape [T]");
  TORCH_CHECK(frame_t_f32.is_contiguous(), "frame_t_f32 must be contiguous");
  TORCH_CHECK(target_rgb_f32.device().is_mps(), "target_rgb_f32 must be on MPS");
  TORCH_CHECK(target_rgb_f32.scalar_type() == torch::kFloat32, "target_rgb_f32 must be float32");
  TORCH_CHECK(target_rgb_f32.dim() == 3 && target_rgb_f32.size(2) == 3, "target_rgb_f32 must have shape [K,T,3]");
  TORCH_CHECK(target_rgb_f32.is_contiguous(), "target_rgb_f32 must be contiguous");
  check_i32_mps_1d(config_i32, "config_i32", 8);
  TORCH_CHECK(config_f32.device().is_mps(), "config_f32 must be on MPS");
  TORCH_CHECK(config_f32.scalar_type() == torch::kFloat32, "config_f32 must be float32");
  TORCH_CHECK(config_f32.dim() == 1 && config_f32.size(0) == 4, "config_f32 must have shape [4]");
  TORCH_CHECK(config_f32.is_contiguous(), "config_f32 must be contiguous");

  auto config_i32_cpu = config_i32.cpu();
  const int32_t* config = config_i32_cpu.data_ptr<int32_t>();
  TORCH_CHECK(config[1] == ray_coeff_f32.size(0), "config_i32[1] must match track count");
  TORCH_CHECK(config[2] == sites_f32.size(0), "config_i32[2] must match site count");
  TORCH_CHECK(config[3] == frame_t_f32.size(0), "config_i32[3] must match frame count");
  TORCH_CHECK(config[4] > 0, "config_i32[4] time_slab_count must be positive");
  TORCH_CHECK(config[5] > 0, "config_i32[5] row_count must be positive");
  TORCH_CHECK(config[6] >= 0, "config_i32[6] candidate_count must be nonnegative");
  TORCH_CHECK(ray_coeff_f32.size(0) > 0, "mixed affine fused MSE VJP requires at least one track");
  TORCH_CHECK(frame_t_f32.size(0) > 0, "mixed affine fused MSE VJP requires at least one frame");
  TORCH_CHECK(sites_f32.size(0) > 0, "mixed affine fused MSE VJP requires at least one site");
  TORCH_CHECK(sites_f32.size(0) <= 64, "mixed affine fused MSE VJP currently supports at most 64 sites");
  TORCH_CHECK(site_rgba_f32.size(0) == sites_f32.size(0), "site RGBA count mismatch");
  TORCH_CHECK(row_index_i32.size(0) == ray_coeff_f32.size(0), "row_index_i32 length must match track count");
  TORCH_CHECK(
      candidate_row_offsets_i32.size(0) == (int64_t)config[5] * (int64_t)config[4] + 1,
      "candidate_row_offsets_i32 length must be row_count * time_slab_count + 1");
  TORCH_CHECK(
      candidate_depth_num_f32.size(0) == config[6],
      "candidate_depth_num_f32 row count must match candidate_count");
  TORCH_CHECK(
      candidate_depth_den_f16.size(0) == config[6],
      "candidate_depth_den_f16 row count must match candidate_count");
  TORCH_CHECK(
      target_rgb_f32.size(0) == ray_coeff_f32.size(0) && target_rgb_f32.size(1) == frame_t_f32.size(0),
      "target_rgb_f32 shape mismatch");

  auto loss = torch::empty({1}, site_rgba_f32.options().dtype(torch::kFloat32));
  auto grad_site_rgba = torch::empty({sites_f32.size(0), 4}, sites_f32.options().dtype(torch::kFloat32));
  const uint64_t total = (uint64_t)ray_coeff_f32.size(0) * (uint64_t)frame_t_f32.size(0);
  auto& k = kernels();
  constexpr uint64_t threads = 256ull;
  launch(k.clear_affine_loss_site_rgba_grad, [&](MetalKernelFunction& fn) {
    fn.setArg(0, loss);
    fn.setArg(1, grad_site_rgba);
    fn.setArg(2, config_i32);
    fn.dispatch((uint64_t)std::max<int64_t>(sites_f32.size(0), 1), threads);
  });
  launch(k.fused_slab_affine_num32_den16_mse_vjp_direct_atomic_rgb_only, [&](MetalKernelFunction& fn) {
    fn.setArg(0, row_index_i32);
    fn.setArg(1, candidate_row_offsets_i32);
    fn.setArg(2, candidate_depth_num_f32);
    fn.setArg(3, candidate_depth_den_f16);
    fn.setArg(4, sites_f32);
    fn.setArg(5, site_rgba_f32);
    fn.setArg(6, ray_coeff_f32);
    fn.setArg(7, frame_t_f32);
    fn.setArg(8, target_rgb_f32);
    fn.setArg(9, config_i32);
    fn.setArg(10, config_f32);
    fn.setArg(11, loss);
    fn.setArg(12, grad_site_rgba);
    fn.dispatch(total, threads);
  });
  return std::make_tuple(loss, grad_site_rgba);
}

std::tuple<torch::Tensor, torch::Tensor> metal_fused_slab_affine_num32_den16_mse_vjp_direct_atomic_track_rgb_only(
    const torch::Tensor& row_index_i32,
    const torch::Tensor& candidate_row_offsets_i32,
    const torch::Tensor& candidate_depth_num_f32,
    const torch::Tensor& candidate_depth_den_f16,
    const torch::Tensor& sites_f32,
    const torch::Tensor& site_rgba_f32,
    const torch::Tensor& ray_coeff_f32,
    const torch::Tensor& frame_t_f32,
    const torch::Tensor& target_rgb_f32,
    const torch::Tensor& config_i32,
    const torch::Tensor& config_f32) {
  check_i32_mps_1d_any(row_index_i32, "row_index_i32");
  check_i32_mps_1d_any(candidate_row_offsets_i32, "candidate_row_offsets_i32");
  check_float_mps_2d(candidate_depth_num_f32, "candidate_depth_num_f32", 2);
  check_half_mps_2d(candidate_depth_den_f16, "candidate_depth_den_f16", 2);
  check_float_mps_2d(sites_f32, "sites_f32", 5);
  check_float_mps_2d(site_rgba_f32, "site_rgba_f32", 4);
  check_float_mps_2d(ray_coeff_f32, "ray_coeff_f32", 12);
  TORCH_CHECK(frame_t_f32.device().is_mps(), "frame_t_f32 must be on MPS");
  TORCH_CHECK(frame_t_f32.scalar_type() == torch::kFloat32, "frame_t_f32 must be float32");
  TORCH_CHECK(frame_t_f32.dim() == 1, "frame_t_f32 must have shape [T]");
  TORCH_CHECK(frame_t_f32.is_contiguous(), "frame_t_f32 must be contiguous");
  TORCH_CHECK(target_rgb_f32.device().is_mps(), "target_rgb_f32 must be on MPS");
  TORCH_CHECK(target_rgb_f32.scalar_type() == torch::kFloat32, "target_rgb_f32 must be float32");
  TORCH_CHECK(target_rgb_f32.dim() == 3 && target_rgb_f32.size(2) == 3, "target_rgb_f32 must have shape [K,T,3]");
  TORCH_CHECK(target_rgb_f32.is_contiguous(), "target_rgb_f32 must be contiguous");
  check_i32_mps_1d(config_i32, "config_i32", 8);
  TORCH_CHECK(config_f32.device().is_mps(), "config_f32 must be on MPS");
  TORCH_CHECK(config_f32.scalar_type() == torch::kFloat32, "config_f32 must be float32");
  TORCH_CHECK(config_f32.dim() == 1 && config_f32.size(0) == 4, "config_f32 must have shape [4]");
  TORCH_CHECK(config_f32.is_contiguous(), "config_f32 must be contiguous");

  auto config_i32_cpu = config_i32.cpu();
  const int32_t* config = config_i32_cpu.data_ptr<int32_t>();
  TORCH_CHECK(config[1] == ray_coeff_f32.size(0), "config_i32[1] must match track count");
  TORCH_CHECK(config[2] == sites_f32.size(0), "config_i32[2] must match site count");
  TORCH_CHECK(config[3] == frame_t_f32.size(0), "config_i32[3] must match frame count");
  TORCH_CHECK(config[4] > 0, "config_i32[4] time_slab_count must be positive");
  TORCH_CHECK(config[5] > 0, "config_i32[5] row_count must be positive");
  TORCH_CHECK(config[6] >= 0, "config_i32[6] candidate_count must be nonnegative");
  TORCH_CHECK(ray_coeff_f32.size(0) > 0, "mixed affine track fused MSE VJP requires at least one track");
  TORCH_CHECK(frame_t_f32.size(0) > 0, "mixed affine track fused MSE VJP requires at least one frame");
  TORCH_CHECK(sites_f32.size(0) > 0, "mixed affine track fused MSE VJP requires at least one site");
  TORCH_CHECK(sites_f32.size(0) <= 64, "mixed affine track fused MSE VJP currently supports at most 64 sites");
  TORCH_CHECK(site_rgba_f32.size(0) == sites_f32.size(0), "site RGBA count mismatch");
  TORCH_CHECK(row_index_i32.size(0) == ray_coeff_f32.size(0), "row_index_i32 length must match track count");
  TORCH_CHECK(
      candidate_row_offsets_i32.size(0) == (int64_t)config[5] * (int64_t)config[4] + 1,
      "candidate_row_offsets_i32 length must be row_count * time_slab_count + 1");
  TORCH_CHECK(
      candidate_depth_num_f32.size(0) == config[6],
      "candidate_depth_num_f32 row count must match candidate_count");
  TORCH_CHECK(
      candidate_depth_den_f16.size(0) == config[6],
      "candidate_depth_den_f16 row count must match candidate_count");
  TORCH_CHECK(
      target_rgb_f32.size(0) == ray_coeff_f32.size(0) && target_rgb_f32.size(1) == frame_t_f32.size(0),
      "target_rgb_f32 shape mismatch");

  auto loss = torch::empty({1}, site_rgba_f32.options().dtype(torch::kFloat32));
  auto grad_site_rgba = torch::empty({sites_f32.size(0), 4}, sites_f32.options().dtype(torch::kFloat32));
  auto& k = kernels();
  constexpr uint64_t threads = 256ull;
  launch(k.clear_affine_loss_site_rgba_grad, [&](MetalKernelFunction& fn) {
    fn.setArg(0, loss);
    fn.setArg(1, grad_site_rgba);
    fn.setArg(2, config_i32);
    fn.dispatch((uint64_t)std::max<int64_t>(sites_f32.size(0), 1), threads);
  });
  launch(k.fused_slab_affine_num32_den16_mse_vjp_direct_atomic_track_rgb_only, [&](MetalKernelFunction& fn) {
    fn.setArg(0, row_index_i32);
    fn.setArg(1, candidate_row_offsets_i32);
    fn.setArg(2, candidate_depth_num_f32);
    fn.setArg(3, candidate_depth_den_f16);
    fn.setArg(4, sites_f32);
    fn.setArg(5, site_rgba_f32);
    fn.setArg(6, ray_coeff_f32);
    fn.setArg(7, frame_t_f32);
    fn.setArg(8, target_rgb_f32);
    fn.setArg(9, config_i32);
    fn.setArg(10, config_f32);
    fn.setArg(11, loss);
    fn.setArg(12, grad_site_rgba);
    fn.dispatch((uint64_t)ray_coeff_f32.size(0), threads);
  });
  return std::make_tuple(loss, grad_site_rgba);
}

std::tuple<torch::Tensor, torch::Tensor> metal_fused_slab_affine_coeff16_mse_vjp_direct_atomic_rgb_only(
    const torch::Tensor& row_index_i32,
    const torch::Tensor& candidate_row_offsets_i32,
    const torch::Tensor& candidate_depth_coeff_f16,
    const torch::Tensor& sites_f32,
    const torch::Tensor& site_rgba_f32,
    const torch::Tensor& ray_coeff_f32,
    const torch::Tensor& frame_t_f32,
    const torch::Tensor& target_rgb_f32,
    const torch::Tensor& config_i32,
    const torch::Tensor& config_f32) {
  check_i32_mps_1d_any(row_index_i32, "row_index_i32");
  check_i32_mps_1d_any(candidate_row_offsets_i32, "candidate_row_offsets_i32");
  check_half_mps_2d(candidate_depth_coeff_f16, "candidate_depth_coeff_f16", 4);
  check_float_mps_2d(sites_f32, "sites_f32", 5);
  check_float_mps_2d(site_rgba_f32, "site_rgba_f32", 4);
  check_float_mps_2d(ray_coeff_f32, "ray_coeff_f32", 12);
  TORCH_CHECK(frame_t_f32.device().is_mps(), "frame_t_f32 must be on MPS");
  TORCH_CHECK(frame_t_f32.scalar_type() == torch::kFloat32, "frame_t_f32 must be float32");
  TORCH_CHECK(frame_t_f32.dim() == 1, "frame_t_f32 must have shape [T]");
  TORCH_CHECK(frame_t_f32.is_contiguous(), "frame_t_f32 must be contiguous");
  TORCH_CHECK(target_rgb_f32.device().is_mps(), "target_rgb_f32 must be on MPS");
  TORCH_CHECK(target_rgb_f32.scalar_type() == torch::kFloat32, "target_rgb_f32 must be float32");
  TORCH_CHECK(target_rgb_f32.dim() == 3 && target_rgb_f32.size(2) == 3, "target_rgb_f32 must have shape [K,T,3]");
  TORCH_CHECK(target_rgb_f32.is_contiguous(), "target_rgb_f32 must be contiguous");
  check_i32_mps_1d(config_i32, "config_i32", 8);
  TORCH_CHECK(config_f32.device().is_mps(), "config_f32 must be on MPS");
  TORCH_CHECK(config_f32.scalar_type() == torch::kFloat32, "config_f32 must be float32");
  TORCH_CHECK(config_f32.dim() == 1 && config_f32.size(0) == 4, "config_f32 must have shape [4]");
  TORCH_CHECK(config_f32.is_contiguous(), "config_f32 must be contiguous");

  auto config_i32_cpu = config_i32.cpu();
  const int32_t* config = config_i32_cpu.data_ptr<int32_t>();
  TORCH_CHECK(config[1] == ray_coeff_f32.size(0), "config_i32[1] must match track count");
  TORCH_CHECK(config[2] == sites_f32.size(0), "config_i32[2] must match site count");
  TORCH_CHECK(config[3] == frame_t_f32.size(0), "config_i32[3] must match frame count");
  TORCH_CHECK(config[4] > 0, "config_i32[4] time_slab_count must be positive");
  TORCH_CHECK(config[5] > 0, "config_i32[5] row_count must be positive");
  TORCH_CHECK(config[6] >= 0, "config_i32[6] candidate_count must be nonnegative");
  TORCH_CHECK(ray_coeff_f32.size(0) > 0, "mixed affine coeff16 fused MSE VJP requires at least one track");
  TORCH_CHECK(frame_t_f32.size(0) > 0, "mixed affine coeff16 fused MSE VJP requires at least one frame");
  TORCH_CHECK(sites_f32.size(0) > 0, "mixed affine coeff16 fused MSE VJP requires at least one site");
  TORCH_CHECK(sites_f32.size(0) <= 64, "mixed affine coeff16 fused MSE VJP currently supports at most 64 sites");
  TORCH_CHECK(site_rgba_f32.size(0) == sites_f32.size(0), "site RGBA count mismatch");
  TORCH_CHECK(row_index_i32.size(0) == ray_coeff_f32.size(0), "row_index_i32 length must match track count");
  TORCH_CHECK(
      candidate_row_offsets_i32.size(0) == (int64_t)config[5] * (int64_t)config[4] + 1,
      "candidate_row_offsets_i32 length must be row_count * time_slab_count + 1");
  TORCH_CHECK(
      candidate_depth_coeff_f16.size(0) == config[6],
      "candidate_depth_coeff_f16 row count must match candidate_count");
  TORCH_CHECK(
      target_rgb_f32.size(0) == ray_coeff_f32.size(0) && target_rgb_f32.size(1) == frame_t_f32.size(0),
      "target_rgb_f32 shape mismatch");

  auto loss = torch::empty({1}, site_rgba_f32.options().dtype(torch::kFloat32));
  auto grad_site_rgba = torch::empty({sites_f32.size(0), 4}, sites_f32.options().dtype(torch::kFloat32));
  const uint64_t total = (uint64_t)ray_coeff_f32.size(0) * (uint64_t)frame_t_f32.size(0);
  auto& k = kernels();
  constexpr uint64_t threads = 256ull;
  launch(k.clear_affine_loss_site_rgba_grad, [&](MetalKernelFunction& fn) {
    fn.setArg(0, loss);
    fn.setArg(1, grad_site_rgba);
    fn.setArg(2, config_i32);
    fn.dispatch((uint64_t)std::max<int64_t>(sites_f32.size(0), 1), threads);
  });
  launch(k.fused_slab_affine_coeff16_mse_vjp_direct_atomic_rgb_only, [&](MetalKernelFunction& fn) {
    fn.setArg(0, row_index_i32);
    fn.setArg(1, candidate_row_offsets_i32);
    fn.setArg(2, candidate_depth_coeff_f16);
    fn.setArg(3, sites_f32);
    fn.setArg(4, site_rgba_f32);
    fn.setArg(5, ray_coeff_f32);
    fn.setArg(6, frame_t_f32);
    fn.setArg(7, target_rgb_f32);
    fn.setArg(8, config_i32);
    fn.setArg(9, config_f32);
    fn.setArg(10, loss);
    fn.setArg(11, grad_site_rgba);
    fn.dispatch(total, threads);
  });
  return std::make_tuple(loss, grad_site_rgba);
}

std::tuple<torch::Tensor, torch::Tensor> metal_fused_slab_affine_coeff16_cap224_mse_vjp_direct_atomic_rgb_only(
    const torch::Tensor& row_index_i32,
    const torch::Tensor& candidate_row_offsets_i32,
    const torch::Tensor& candidate_depth_coeff_f16,
    const torch::Tensor& sites_f32,
    const torch::Tensor& site_rgba_f32,
    const torch::Tensor& ray_coeff_f32,
    const torch::Tensor& frame_t_f32,
    const torch::Tensor& target_rgb_f32,
    const torch::Tensor& config_i32,
    const torch::Tensor& config_f32) {
  check_i32_mps_1d_any(row_index_i32, "row_index_i32");
  check_i32_mps_1d_any(candidate_row_offsets_i32, "candidate_row_offsets_i32");
  check_half_mps_2d(candidate_depth_coeff_f16, "candidate_depth_coeff_f16", 4);
  check_float_mps_2d(sites_f32, "sites_f32", 5);
  check_float_mps_2d(site_rgba_f32, "site_rgba_f32", 4);
  check_float_mps_2d(ray_coeff_f32, "ray_coeff_f32", 12);
  TORCH_CHECK(frame_t_f32.device().is_mps(), "frame_t_f32 must be on MPS");
  TORCH_CHECK(frame_t_f32.scalar_type() == torch::kFloat32, "frame_t_f32 must be float32");
  TORCH_CHECK(frame_t_f32.dim() == 1, "frame_t_f32 must have shape [T]");
  TORCH_CHECK(frame_t_f32.is_contiguous(), "frame_t_f32 must be contiguous");
  TORCH_CHECK(target_rgb_f32.device().is_mps(), "target_rgb_f32 must be on MPS");
  TORCH_CHECK(target_rgb_f32.scalar_type() == torch::kFloat32, "target_rgb_f32 must be float32");
  TORCH_CHECK(target_rgb_f32.dim() == 3 && target_rgb_f32.size(2) == 3, "target_rgb_f32 must have shape [K,T,3]");
  TORCH_CHECK(target_rgb_f32.is_contiguous(), "target_rgb_f32 must be contiguous");
  check_i32_mps_1d(config_i32, "config_i32", 8);
  TORCH_CHECK(config_f32.device().is_mps(), "config_f32 must be on MPS");
  TORCH_CHECK(config_f32.scalar_type() == torch::kFloat32, "config_f32 must be float32");
  TORCH_CHECK(config_f32.dim() == 1 && config_f32.size(0) == 4, "config_f32 must have shape [4]");
  TORCH_CHECK(config_f32.is_contiguous(), "config_f32 must be contiguous");

  auto config_i32_cpu = config_i32.cpu();
  const int32_t* config = config_i32_cpu.data_ptr<int32_t>();
  TORCH_CHECK(config[1] == ray_coeff_f32.size(0), "config_i32[1] must match track count");
  TORCH_CHECK(config[2] == sites_f32.size(0), "config_i32[2] must match site count");
  TORCH_CHECK(config[3] == frame_t_f32.size(0), "config_i32[3] must match frame count");
  TORCH_CHECK(config[4] > 0, "config_i32[4] time_slab_count must be positive");
  TORCH_CHECK(config[5] > 0, "config_i32[5] row_count must be positive");
  TORCH_CHECK(config[6] >= 0, "config_i32[6] candidate_count must be nonnegative");
  TORCH_CHECK(ray_coeff_f32.size(0) > 0, "mixed affine coeff16 cap224 fused MSE VJP requires at least one track");
  TORCH_CHECK(frame_t_f32.size(0) > 0, "mixed affine coeff16 cap224 fused MSE VJP requires at least one frame");
  TORCH_CHECK(sites_f32.size(0) > 0, "mixed affine coeff16 cap224 fused MSE VJP requires at least one site");
  TORCH_CHECK(sites_f32.size(0) <= 64, "mixed affine coeff16 cap224 fused MSE VJP currently supports at most 64 sites");
  TORCH_CHECK(site_rgba_f32.size(0) == sites_f32.size(0), "site RGBA count mismatch");
  TORCH_CHECK(row_index_i32.size(0) == ray_coeff_f32.size(0), "row_index_i32 length must match track count");
  TORCH_CHECK(
      candidate_row_offsets_i32.size(0) == (int64_t)config[5] * (int64_t)config[4] + 1,
      "candidate_row_offsets_i32 length must be row_count * time_slab_count + 1");
  TORCH_CHECK(
      candidate_depth_coeff_f16.size(0) == config[6],
      "candidate_depth_coeff_f16 row count must match candidate_count");
  TORCH_CHECK(
      target_rgb_f32.size(0) == ray_coeff_f32.size(0) && target_rgb_f32.size(1) == frame_t_f32.size(0),
      "target_rgb_f32 shape mismatch");

  auto loss = torch::empty({1}, site_rgba_f32.options().dtype(torch::kFloat32));
  auto grad_site_rgba = torch::empty({sites_f32.size(0), 4}, sites_f32.options().dtype(torch::kFloat32));
  const uint64_t total = (uint64_t)ray_coeff_f32.size(0) * (uint64_t)frame_t_f32.size(0);
  auto& k = kernels();
  constexpr uint64_t threads = 256ull;
  launch(k.clear_affine_loss_site_rgba_grad, [&](MetalKernelFunction& fn) {
    fn.setArg(0, loss);
    fn.setArg(1, grad_site_rgba);
    fn.setArg(2, config_i32);
    fn.dispatch((uint64_t)std::max<int64_t>(sites_f32.size(0), 1), threads);
  });
  launch(k.fused_slab_affine_coeff16_cap224_mse_vjp_direct_atomic_rgb_only, [&](MetalKernelFunction& fn) {
    fn.setArg(0, row_index_i32);
    fn.setArg(1, candidate_row_offsets_i32);
    fn.setArg(2, candidate_depth_coeff_f16);
    fn.setArg(3, sites_f32);
    fn.setArg(4, site_rgba_f32);
    fn.setArg(5, ray_coeff_f32);
    fn.setArg(6, frame_t_f32);
    fn.setArg(7, target_rgb_f32);
    fn.setArg(8, config_i32);
    fn.setArg(9, config_f32);
    fn.setArg(10, loss);
    fn.setArg(11, grad_site_rgba);
    fn.dispatch(total, threads);
  });
  return std::make_tuple(loss, grad_site_rgba);
}

std::tuple<torch::Tensor, torch::Tensor> metal_fused_slab_affine_coeff16_densitymask_mse_vjp_direct_atomic_rgb_only(
    const torch::Tensor& row_index_i32,
    const torch::Tensor& candidate_row_offsets_i32,
    const torch::Tensor& candidate_depth_coeff_f16,
    const torch::Tensor& sites_f32,
    const torch::Tensor& site_rgba_f32,
    const torch::Tensor& ray_coeff_f32,
    const torch::Tensor& frame_t_f32,
    const torch::Tensor& target_rgb_f32,
    const torch::Tensor& config_i32,
    const torch::Tensor& config_f32) {
  check_i32_mps_1d_any(row_index_i32, "row_index_i32");
  check_i32_mps_1d_any(candidate_row_offsets_i32, "candidate_row_offsets_i32");
  check_half_mps_2d(candidate_depth_coeff_f16, "candidate_depth_coeff_f16", 4);
  check_float_mps_2d(sites_f32, "sites_f32", 5);
  check_float_mps_2d(site_rgba_f32, "site_rgba_f32", 4);
  check_float_mps_2d(ray_coeff_f32, "ray_coeff_f32", 12);
  TORCH_CHECK(frame_t_f32.device().is_mps(), "frame_t_f32 must be on MPS");
  TORCH_CHECK(frame_t_f32.scalar_type() == torch::kFloat32, "frame_t_f32 must be float32");
  TORCH_CHECK(frame_t_f32.dim() == 1, "frame_t_f32 must have shape [T]");
  TORCH_CHECK(frame_t_f32.is_contiguous(), "frame_t_f32 must be contiguous");
  TORCH_CHECK(target_rgb_f32.device().is_mps(), "target_rgb_f32 must be on MPS");
  TORCH_CHECK(target_rgb_f32.scalar_type() == torch::kFloat32, "target_rgb_f32 must be float32");
  TORCH_CHECK(target_rgb_f32.dim() == 3 && target_rgb_f32.size(2) == 3, "target_rgb_f32 must have shape [K,T,3]");
  TORCH_CHECK(target_rgb_f32.is_contiguous(), "target_rgb_f32 must be contiguous");
  check_i32_mps_1d(config_i32, "config_i32", 8);
  TORCH_CHECK(config_f32.device().is_mps(), "config_f32 must be on MPS");
  TORCH_CHECK(config_f32.scalar_type() == torch::kFloat32, "config_f32 must be float32");
  TORCH_CHECK(config_f32.dim() == 1 && config_f32.size(0) == 4, "config_f32 must have shape [4]");
  TORCH_CHECK(config_f32.is_contiguous(), "config_f32 must be contiguous");

  auto config_i32_cpu = config_i32.cpu();
  const int32_t* config = config_i32_cpu.data_ptr<int32_t>();
  TORCH_CHECK(config[1] == ray_coeff_f32.size(0), "config_i32[1] must match track count");
  TORCH_CHECK(config[2] == sites_f32.size(0), "config_i32[2] must match site count");
  TORCH_CHECK(config[3] == frame_t_f32.size(0), "config_i32[3] must match frame count");
  TORCH_CHECK(config[4] > 0, "config_i32[4] time_slab_count must be positive");
  TORCH_CHECK(config[5] > 0, "config_i32[5] row_count must be positive");
  TORCH_CHECK(config[6] >= 0, "config_i32[6] candidate_count must be nonnegative");
  TORCH_CHECK(ray_coeff_f32.size(0) > 0, "mixed affine coeff16 densitymask fused MSE VJP requires at least one track");
  TORCH_CHECK(frame_t_f32.size(0) > 0, "mixed affine coeff16 densitymask fused MSE VJP requires at least one frame");
  TORCH_CHECK(sites_f32.size(0) > 0, "mixed affine coeff16 densitymask fused MSE VJP requires at least one site");
  TORCH_CHECK(sites_f32.size(0) <= 64, "mixed affine coeff16 densitymask fused MSE VJP currently supports at most 64 sites");
  TORCH_CHECK(site_rgba_f32.size(0) == sites_f32.size(0), "site RGBA count mismatch");
  TORCH_CHECK(row_index_i32.size(0) == ray_coeff_f32.size(0), "row_index_i32 length must match track count");
  TORCH_CHECK(
      candidate_row_offsets_i32.size(0) == (int64_t)config[5] * (int64_t)config[4] + 1,
      "candidate_row_offsets_i32 length must be row_count * time_slab_count + 1");
  TORCH_CHECK(
      candidate_depth_coeff_f16.size(0) == config[6],
      "candidate_depth_coeff_f16 row count must match candidate_count");
  TORCH_CHECK(
      target_rgb_f32.size(0) == ray_coeff_f32.size(0) && target_rgb_f32.size(1) == frame_t_f32.size(0),
      "target_rgb_f32 shape mismatch");

  auto loss = torch::empty({1}, site_rgba_f32.options().dtype(torch::kFloat32));
  auto grad_site_rgba = torch::empty({sites_f32.size(0), 4}, sites_f32.options().dtype(torch::kFloat32));
  const uint64_t total = (uint64_t)ray_coeff_f32.size(0) * (uint64_t)frame_t_f32.size(0);
  auto& k = kernels();
  constexpr uint64_t threads = 256ull;
  launch(k.clear_affine_loss_site_rgba_grad, [&](MetalKernelFunction& fn) {
    fn.setArg(0, loss);
    fn.setArg(1, grad_site_rgba);
    fn.setArg(2, config_i32);
    fn.dispatch((uint64_t)std::max<int64_t>(sites_f32.size(0), 1), threads);
  });
  launch(k.fused_slab_affine_coeff16_densitymask_mse_vjp_direct_atomic_rgb_only, [&](MetalKernelFunction& fn) {
    fn.setArg(0, row_index_i32);
    fn.setArg(1, candidate_row_offsets_i32);
    fn.setArg(2, candidate_depth_coeff_f16);
    fn.setArg(3, sites_f32);
    fn.setArg(4, site_rgba_f32);
    fn.setArg(5, ray_coeff_f32);
    fn.setArg(6, frame_t_f32);
    fn.setArg(7, target_rgb_f32);
    fn.setArg(8, config_i32);
    fn.setArg(9, config_f32);
    fn.setArg(10, loss);
    fn.setArg(11, grad_site_rgba);
    fn.dispatch(total, threads);
  });
  return std::make_tuple(loss, grad_site_rgba);
}

std::tuple<torch::Tensor, torch::Tensor>
metal_fused_slab_affine_coeff16_mse_vjp_direct_atomic_sample_reduce_rgb_only(
    const torch::Tensor& row_index_i32,
    const torch::Tensor& candidate_row_offsets_i32,
    const torch::Tensor& candidate_depth_coeff_f16,
    const torch::Tensor& sites_f32,
    const torch::Tensor& site_rgba_f32,
    const torch::Tensor& ray_coeff_f32,
    const torch::Tensor& frame_t_f32,
    const torch::Tensor& target_rgb_f32,
    const torch::Tensor& config_i32,
    const torch::Tensor& config_f32) {
  check_i32_mps_1d_any(row_index_i32, "row_index_i32");
  check_i32_mps_1d_any(candidate_row_offsets_i32, "candidate_row_offsets_i32");
  check_half_mps_2d(candidate_depth_coeff_f16, "candidate_depth_coeff_f16", 4);
  check_float_mps_2d(sites_f32, "sites_f32", 5);
  check_float_mps_2d(site_rgba_f32, "site_rgba_f32", 4);
  check_float_mps_2d(ray_coeff_f32, "ray_coeff_f32", 12);
  TORCH_CHECK(frame_t_f32.device().is_mps(), "frame_t_f32 must be on MPS");
  TORCH_CHECK(frame_t_f32.scalar_type() == torch::kFloat32, "frame_t_f32 must be float32");
  TORCH_CHECK(frame_t_f32.dim() == 1, "frame_t_f32 must have shape [T]");
  TORCH_CHECK(frame_t_f32.is_contiguous(), "frame_t_f32 must be contiguous");
  TORCH_CHECK(target_rgb_f32.device().is_mps(), "target_rgb_f32 must be on MPS");
  TORCH_CHECK(target_rgb_f32.scalar_type() == torch::kFloat32, "target_rgb_f32 must be float32");
  TORCH_CHECK(target_rgb_f32.dim() == 3 && target_rgb_f32.size(2) == 3, "target_rgb_f32 must have shape [K,T,3]");
  TORCH_CHECK(target_rgb_f32.is_contiguous(), "target_rgb_f32 must be contiguous");
  check_i32_mps_1d(config_i32, "config_i32", 8);
  TORCH_CHECK(config_f32.device().is_mps(), "config_f32 must be on MPS");
  TORCH_CHECK(config_f32.scalar_type() == torch::kFloat32, "config_f32 must be float32");
  TORCH_CHECK(config_f32.dim() == 1 && config_f32.size(0) == 4, "config_f32 must have shape [4]");
  TORCH_CHECK(config_f32.is_contiguous(), "config_f32 must be contiguous");

  auto config_i32_cpu = config_i32.cpu();
  const int32_t* config = config_i32_cpu.data_ptr<int32_t>();
  TORCH_CHECK(config[1] == ray_coeff_f32.size(0), "config_i32[1] must match track count");
  TORCH_CHECK(config[2] == sites_f32.size(0), "config_i32[2] must match site count");
  TORCH_CHECK(config[3] == frame_t_f32.size(0), "config_i32[3] must match frame count");
  TORCH_CHECK(config[4] > 0, "config_i32[4] time_slab_count must be positive");
  TORCH_CHECK(config[5] > 0, "config_i32[5] row_count must be positive");
  TORCH_CHECK(config[6] >= 0, "config_i32[6] candidate_count must be nonnegative");
  TORCH_CHECK(ray_coeff_f32.size(0) > 0, "mixed affine coeff16 sample-reduce fused MSE VJP requires at least one track");
  TORCH_CHECK(frame_t_f32.size(0) > 0, "mixed affine coeff16 sample-reduce fused MSE VJP requires at least one frame");
  TORCH_CHECK(sites_f32.size(0) > 0, "mixed affine coeff16 sample-reduce fused MSE VJP requires at least one site");
  TORCH_CHECK(
      sites_f32.size(0) <= 64, "mixed affine coeff16 sample-reduce fused MSE VJP currently supports at most 64 sites");
  TORCH_CHECK(site_rgba_f32.size(0) == sites_f32.size(0), "site RGBA count mismatch");
  TORCH_CHECK(row_index_i32.size(0) == ray_coeff_f32.size(0), "row_index_i32 length must match track count");
  TORCH_CHECK(
      candidate_row_offsets_i32.size(0) == (int64_t)config[5] * (int64_t)config[4] + 1,
      "candidate_row_offsets_i32 length must be row_count * time_slab_count + 1");
  TORCH_CHECK(
      candidate_depth_coeff_f16.size(0) == config[6],
      "candidate_depth_coeff_f16 row count must match candidate_count");
  TORCH_CHECK(
      target_rgb_f32.size(0) == ray_coeff_f32.size(0) && target_rgb_f32.size(1) == frame_t_f32.size(0),
      "target_rgb_f32 shape mismatch");

  auto loss = torch::empty({1}, site_rgba_f32.options().dtype(torch::kFloat32));
  auto grad_site_rgba = torch::empty({sites_f32.size(0), 4}, sites_f32.options().dtype(torch::kFloat32));
  const uint64_t total = (uint64_t)ray_coeff_f32.size(0) * (uint64_t)frame_t_f32.size(0);
  auto& k = kernels();
  constexpr uint64_t threads = 256ull;
  launch(k.clear_affine_loss_site_rgba_grad, [&](MetalKernelFunction& fn) {
    fn.setArg(0, loss);
    fn.setArg(1, grad_site_rgba);
    fn.setArg(2, config_i32);
    fn.dispatch((uint64_t)std::max<int64_t>(sites_f32.size(0), 1), threads);
  });
  launch(k.fused_slab_affine_coeff16_mse_vjp_direct_atomic_sample_reduce_rgb_only, [&](MetalKernelFunction& fn) {
    fn.setArg(0, row_index_i32);
    fn.setArg(1, candidate_row_offsets_i32);
    fn.setArg(2, candidate_depth_coeff_f16);
    fn.setArg(3, sites_f32);
    fn.setArg(4, site_rgba_f32);
    fn.setArg(5, ray_coeff_f32);
    fn.setArg(6, frame_t_f32);
    fn.setArg(7, target_rgb_f32);
    fn.setArg(8, config_i32);
    fn.setArg(9, config_f32);
    fn.setArg(10, loss);
    fn.setArg(11, grad_site_rgba);
    fn.dispatch(total, threads);
  });
  return std::make_tuple(loss, grad_site_rgba);
}

std::tuple<torch::Tensor, torch::Tensor>
metal_fused_slab_affine_coeff16_sortnet_mse_vjp_direct_atomic_rgb_only(
    const torch::Tensor& row_index_i32,
    const torch::Tensor& candidate_row_offsets_i32,
    const torch::Tensor& candidate_depth_coeff_f16,
    const torch::Tensor& sites_f32,
    const torch::Tensor& site_rgba_f32,
    const torch::Tensor& ray_coeff_f32,
    const torch::Tensor& frame_t_f32,
    const torch::Tensor& target_rgb_f32,
    const torch::Tensor& config_i32,
    const torch::Tensor& config_f32) {
  check_i32_mps_1d_any(row_index_i32, "row_index_i32");
  check_i32_mps_1d_any(candidate_row_offsets_i32, "candidate_row_offsets_i32");
  check_half_mps_2d(candidate_depth_coeff_f16, "candidate_depth_coeff_f16", 4);
  check_float_mps_2d(sites_f32, "sites_f32", 5);
  check_float_mps_2d(site_rgba_f32, "site_rgba_f32", 4);
  check_float_mps_2d(ray_coeff_f32, "ray_coeff_f32", 12);
  TORCH_CHECK(frame_t_f32.device().is_mps(), "frame_t_f32 must be on MPS");
  TORCH_CHECK(frame_t_f32.scalar_type() == torch::kFloat32, "frame_t_f32 must be float32");
  TORCH_CHECK(frame_t_f32.dim() == 1, "frame_t_f32 must have shape [T]");
  TORCH_CHECK(frame_t_f32.is_contiguous(), "frame_t_f32 must be contiguous");
  TORCH_CHECK(target_rgb_f32.device().is_mps(), "target_rgb_f32 must be on MPS");
  TORCH_CHECK(target_rgb_f32.scalar_type() == torch::kFloat32, "target_rgb_f32 must be float32");
  TORCH_CHECK(target_rgb_f32.dim() == 3 && target_rgb_f32.size(2) == 3, "target_rgb_f32 must have shape [K,T,3]");
  TORCH_CHECK(target_rgb_f32.is_contiguous(), "target_rgb_f32 must be contiguous");
  check_i32_mps_1d(config_i32, "config_i32", 8);
  TORCH_CHECK(config_f32.device().is_mps(), "config_f32 must be on MPS");
  TORCH_CHECK(config_f32.scalar_type() == torch::kFloat32, "config_f32 must be float32");
  TORCH_CHECK(config_f32.dim() == 1 && config_f32.size(0) == 4, "config_f32 must have shape [4]");
  TORCH_CHECK(config_f32.is_contiguous(), "config_f32 must be contiguous");

  auto config_i32_cpu = config_i32.cpu();
  const int32_t* config = config_i32_cpu.data_ptr<int32_t>();
  TORCH_CHECK(config[1] == ray_coeff_f32.size(0), "config_i32[1] must match track count");
  TORCH_CHECK(config[2] == sites_f32.size(0), "config_i32[2] must match site count");
  TORCH_CHECK(config[3] == frame_t_f32.size(0), "config_i32[3] must match frame count");
  TORCH_CHECK(config[4] > 0, "config_i32[4] time_slab_count must be positive");
  TORCH_CHECK(config[5] > 0, "config_i32[5] row_count must be positive");
  TORCH_CHECK(config[6] >= 0, "config_i32[6] candidate_count must be nonnegative");
  TORCH_CHECK(ray_coeff_f32.size(0) > 0, "mixed affine coeff16 sortnet fused MSE VJP requires at least one track");
  TORCH_CHECK(frame_t_f32.size(0) > 0, "mixed affine coeff16 sortnet fused MSE VJP requires at least one frame");
  TORCH_CHECK(sites_f32.size(0) > 0, "mixed affine coeff16 sortnet fused MSE VJP requires at least one site");
  TORCH_CHECK(sites_f32.size(0) <= 64, "mixed affine coeff16 sortnet fused MSE VJP currently supports at most 64 sites");
  TORCH_CHECK(site_rgba_f32.size(0) == sites_f32.size(0), "site RGBA count mismatch");
  TORCH_CHECK(row_index_i32.size(0) == ray_coeff_f32.size(0), "row_index_i32 length must match track count");
  TORCH_CHECK(
      candidate_row_offsets_i32.size(0) == (int64_t)config[5] * (int64_t)config[4] + 1,
      "candidate_row_offsets_i32 length must be row_count * time_slab_count + 1");
  TORCH_CHECK(
      candidate_depth_coeff_f16.size(0) == config[6],
      "candidate_depth_coeff_f16 row count must match candidate_count");
  TORCH_CHECK(
      target_rgb_f32.size(0) == ray_coeff_f32.size(0) && target_rgb_f32.size(1) == frame_t_f32.size(0),
      "target_rgb_f32 shape mismatch");

  auto loss = torch::empty({1}, site_rgba_f32.options().dtype(torch::kFloat32));
  auto grad_site_rgba = torch::empty({sites_f32.size(0), 4}, sites_f32.options().dtype(torch::kFloat32));
  const uint64_t total = (uint64_t)ray_coeff_f32.size(0) * (uint64_t)frame_t_f32.size(0);
  auto& k = kernels();
  constexpr uint64_t threads = 256ull;
  launch(k.clear_affine_loss_site_rgba_grad, [&](MetalKernelFunction& fn) {
    fn.setArg(0, loss);
    fn.setArg(1, grad_site_rgba);
    fn.setArg(2, config_i32);
    fn.dispatch((uint64_t)std::max<int64_t>(sites_f32.size(0), 1), threads);
  });
  launch(k.fused_slab_affine_coeff16_sortnet_mse_vjp_direct_atomic_rgb_only, [&](MetalKernelFunction& fn) {
    fn.setArg(0, row_index_i32);
    fn.setArg(1, candidate_row_offsets_i32);
    fn.setArg(2, candidate_depth_coeff_f16);
    fn.setArg(3, sites_f32);
    fn.setArg(4, site_rgba_f32);
    fn.setArg(5, ray_coeff_f32);
    fn.setArg(6, frame_t_f32);
    fn.setArg(7, target_rgb_f32);
    fn.setArg(8, config_i32);
    fn.setArg(9, config_f32);
    fn.setArg(10, loss);
    fn.setArg(11, grad_site_rgba);
    fn.dispatch(total, threads);
  });
  return std::make_tuple(loss, grad_site_rgba);
}

std::tuple<torch::Tensor, torch::Tensor>
metal_fused_slab_affine_coeff16_mse_vjp_direct_atomic_sitecache_rgb_only(
    const torch::Tensor& row_index_i32,
    const torch::Tensor& candidate_row_offsets_i32,
    const torch::Tensor& candidate_depth_coeff_f16,
    const torch::Tensor& sites_f32,
    const torch::Tensor& site_rgba_f32,
    const torch::Tensor& ray_coeff_f32,
    const torch::Tensor& frame_t_f32,
    const torch::Tensor& target_rgb_f32,
    const torch::Tensor& config_i32,
    const torch::Tensor& config_f32) {
  check_i32_mps_1d_any(row_index_i32, "row_index_i32");
  check_i32_mps_1d_any(candidate_row_offsets_i32, "candidate_row_offsets_i32");
  check_half_mps_2d(candidate_depth_coeff_f16, "candidate_depth_coeff_f16", 4);
  check_float_mps_2d(sites_f32, "sites_f32", 5);
  check_float_mps_2d(site_rgba_f32, "site_rgba_f32", 4);
  check_float_mps_2d(ray_coeff_f32, "ray_coeff_f32", 12);
  TORCH_CHECK(frame_t_f32.device().is_mps(), "frame_t_f32 must be on MPS");
  TORCH_CHECK(frame_t_f32.scalar_type() == torch::kFloat32, "frame_t_f32 must be float32");
  TORCH_CHECK(frame_t_f32.dim() == 1, "frame_t_f32 must have shape [T]");
  TORCH_CHECK(frame_t_f32.is_contiguous(), "frame_t_f32 must be contiguous");
  TORCH_CHECK(target_rgb_f32.device().is_mps(), "target_rgb_f32 must be on MPS");
  TORCH_CHECK(target_rgb_f32.scalar_type() == torch::kFloat32, "target_rgb_f32 must be float32");
  TORCH_CHECK(target_rgb_f32.dim() == 3 && target_rgb_f32.size(2) == 3, "target_rgb_f32 must have shape [K,T,3]");
  TORCH_CHECK(target_rgb_f32.is_contiguous(), "target_rgb_f32 must be contiguous");
  check_i32_mps_1d(config_i32, "config_i32", 8);
  TORCH_CHECK(config_f32.device().is_mps(), "config_f32 must be on MPS");
  TORCH_CHECK(config_f32.scalar_type() == torch::kFloat32, "config_f32 must be float32");
  TORCH_CHECK(config_f32.dim() == 1 && config_f32.size(0) == 4, "config_f32 must have shape [4]");
  TORCH_CHECK(config_f32.is_contiguous(), "config_f32 must be contiguous");

  auto config_i32_cpu = config_i32.cpu();
  const int32_t* config = config_i32_cpu.data_ptr<int32_t>();
  TORCH_CHECK(config[1] == ray_coeff_f32.size(0), "config_i32[1] must match track count");
  TORCH_CHECK(config[2] == sites_f32.size(0), "config_i32[2] must match site count");
  TORCH_CHECK(config[3] == frame_t_f32.size(0), "config_i32[3] must match frame count");
  TORCH_CHECK(config[4] > 0, "config_i32[4] time_slab_count must be positive");
  TORCH_CHECK(config[5] > 0, "config_i32[5] row_count must be positive");
  TORCH_CHECK(config[6] >= 0, "config_i32[6] candidate_count must be nonnegative");
  TORCH_CHECK(ray_coeff_f32.size(0) > 0, "mixed affine coeff16 sitecache fused MSE VJP requires at least one track");
  TORCH_CHECK(frame_t_f32.size(0) > 0, "mixed affine coeff16 sitecache fused MSE VJP requires at least one frame");
  TORCH_CHECK(sites_f32.size(0) > 0, "mixed affine coeff16 sitecache fused MSE VJP requires at least one site");
  TORCH_CHECK(sites_f32.size(0) <= 64, "mixed affine coeff16 sitecache fused MSE VJP currently supports at most 64 sites");
  TORCH_CHECK(site_rgba_f32.size(0) == sites_f32.size(0), "site RGBA count mismatch");
  TORCH_CHECK(row_index_i32.size(0) == ray_coeff_f32.size(0), "row_index_i32 length must match track count");
  TORCH_CHECK(
      candidate_row_offsets_i32.size(0) == (int64_t)config[5] * (int64_t)config[4] + 1,
      "candidate_row_offsets_i32 length must be row_count * time_slab_count + 1");
  TORCH_CHECK(
      candidate_depth_coeff_f16.size(0) == config[6],
      "candidate_depth_coeff_f16 row count must match candidate_count");
  TORCH_CHECK(
      target_rgb_f32.size(0) == ray_coeff_f32.size(0) && target_rgb_f32.size(1) == frame_t_f32.size(0),
      "target_rgb_f32 shape mismatch");

  auto loss = torch::empty({1}, site_rgba_f32.options().dtype(torch::kFloat32));
  auto grad_site_rgba = torch::empty({sites_f32.size(0), 4}, sites_f32.options().dtype(torch::kFloat32));
  const uint64_t sample_total = (uint64_t)ray_coeff_f32.size(0) * (uint64_t)frame_t_f32.size(0);
  auto& k = kernels();
  constexpr uint64_t threads = 256ull;
  const uint64_t total = ((sample_total + threads - 1ull) / threads) * threads;
  launch(k.clear_affine_loss_site_rgba_grad, [&](MetalKernelFunction& fn) {
    fn.setArg(0, loss);
    fn.setArg(1, grad_site_rgba);
    fn.setArg(2, config_i32);
    fn.dispatch((uint64_t)std::max<int64_t>(sites_f32.size(0), 1), threads);
  });
  launch(k.fused_slab_affine_coeff16_mse_vjp_direct_atomic_sitecache_rgb_only, [&](MetalKernelFunction& fn) {
    fn.setArg(0, row_index_i32);
    fn.setArg(1, candidate_row_offsets_i32);
    fn.setArg(2, candidate_depth_coeff_f16);
    fn.setArg(3, sites_f32);
    fn.setArg(4, site_rgba_f32);
    fn.setArg(5, ray_coeff_f32);
    fn.setArg(6, frame_t_f32);
    fn.setArg(7, target_rgb_f32);
    fn.setArg(8, config_i32);
    fn.setArg(9, config_f32);
    fn.setArg(10, loss);
    fn.setArg(11, grad_site_rgba);
    fn.dispatch(total, threads);
  });
  return std::make_tuple(loss, grad_site_rgba);
}

std::tuple<torch::Tensor, torch::Tensor>
metal_fused_slab_affine_coeff16_mse_vjp_direct_atomic_framegroup16_cached_rgb_only(
    const torch::Tensor& row_index_i32,
    const torch::Tensor& candidate_row_offsets_i32,
    const torch::Tensor& candidate_depth_coeff_f16,
    const torch::Tensor& sites_f32,
    const torch::Tensor& site_rgba_f32,
    const torch::Tensor& ray_coeff_f32,
    const torch::Tensor& frame_t_f32,
    const torch::Tensor& target_rgb_f32,
    const torch::Tensor& config_i32,
    const torch::Tensor& config_f32) {
  check_i32_mps_1d_any(row_index_i32, "row_index_i32");
  check_i32_mps_1d_any(candidate_row_offsets_i32, "candidate_row_offsets_i32");
  check_half_mps_2d(candidate_depth_coeff_f16, "candidate_depth_coeff_f16", 4);
  check_float_mps_2d(sites_f32, "sites_f32", 5);
  check_float_mps_2d(site_rgba_f32, "site_rgba_f32", 4);
  check_float_mps_2d(ray_coeff_f32, "ray_coeff_f32", 12);
  TORCH_CHECK(frame_t_f32.device().is_mps(), "frame_t_f32 must be on MPS");
  TORCH_CHECK(frame_t_f32.scalar_type() == torch::kFloat32, "frame_t_f32 must be float32");
  TORCH_CHECK(frame_t_f32.dim() == 1, "frame_t_f32 must have shape [T]");
  TORCH_CHECK(frame_t_f32.is_contiguous(), "frame_t_f32 must be contiguous");
  TORCH_CHECK(target_rgb_f32.device().is_mps(), "target_rgb_f32 must be on MPS");
  TORCH_CHECK(target_rgb_f32.scalar_type() == torch::kFloat32, "target_rgb_f32 must be float32");
  TORCH_CHECK(target_rgb_f32.dim() == 3 && target_rgb_f32.size(2) == 3, "target_rgb_f32 must have shape [K,T,3]");
  TORCH_CHECK(target_rgb_f32.is_contiguous(), "target_rgb_f32 must be contiguous");
  check_i32_mps_1d(config_i32, "config_i32", 8);
  TORCH_CHECK(config_f32.device().is_mps(), "config_f32 must be on MPS");
  TORCH_CHECK(config_f32.scalar_type() == torch::kFloat32, "config_f32 must be float32");
  TORCH_CHECK(config_f32.dim() == 1 && config_f32.size(0) == 4, "config_f32 must have shape [4]");
  TORCH_CHECK(config_f32.is_contiguous(), "config_f32 must be contiguous");

  auto config_i32_cpu = config_i32.cpu();
  const int32_t* config = config_i32_cpu.data_ptr<int32_t>();
  TORCH_CHECK(config[1] == ray_coeff_f32.size(0), "config_i32[1] must match track count");
  TORCH_CHECK(config[2] == sites_f32.size(0), "config_i32[2] must match site count");
  TORCH_CHECK(config[3] == frame_t_f32.size(0), "config_i32[3] must match frame count");
  TORCH_CHECK(config[4] == 1, "framegroup16 cached coeff16 fused MSE currently requires time_slab_count == 1");
  TORCH_CHECK(config[5] > 0, "config_i32[5] row_count must be positive");
  TORCH_CHECK(config[6] >= 0, "config_i32[6] candidate_count must be nonnegative");
  TORCH_CHECK(ray_coeff_f32.size(0) > 0, "mixed affine coeff16 framegroup16 cached fused MSE VJP requires at least one track");
  TORCH_CHECK(frame_t_f32.size(0) > 0, "mixed affine coeff16 framegroup16 cached fused MSE VJP requires at least one frame");
  TORCH_CHECK(sites_f32.size(0) > 0, "mixed affine coeff16 framegroup16 cached fused MSE VJP requires at least one site");
  TORCH_CHECK(
      sites_f32.size(0) <= 64,
      "mixed affine coeff16 framegroup16 cached fused MSE VJP currently supports at most 64 sites");
  TORCH_CHECK(site_rgba_f32.size(0) == sites_f32.size(0), "site RGBA count mismatch");
  TORCH_CHECK(row_index_i32.size(0) == ray_coeff_f32.size(0), "row_index_i32 length must match track count");
  TORCH_CHECK(
      candidate_row_offsets_i32.size(0) == (int64_t)config[5] * (int64_t)config[4] + 1,
      "candidate_row_offsets_i32 length must be row_count * time_slab_count + 1");
  TORCH_CHECK(
      candidate_depth_coeff_f16.size(0) == config[6],
      "candidate_depth_coeff_f16 row count must match candidate_count");
  TORCH_CHECK(
      target_rgb_f32.size(0) == ray_coeff_f32.size(0) && target_rgb_f32.size(1) == frame_t_f32.size(0),
      "target_rgb_f32 shape mismatch");

  auto loss = torch::empty({1}, site_rgba_f32.options().dtype(torch::kFloat32));
  auto grad_site_rgba = torch::empty({sites_f32.size(0), 4}, sites_f32.options().dtype(torch::kFloat32));
  const uint64_t framegroup_threads = 16ull;
  const uint64_t chunk_count =
      ((uint64_t)frame_t_f32.size(0) + framegroup_threads - 1ull) / framegroup_threads;
  const uint64_t total = (uint64_t)ray_coeff_f32.size(0) * chunk_count * framegroup_threads;
  auto& k = kernels();
  constexpr uint64_t threads = 256ull;
  launch(k.clear_affine_loss_site_rgba_grad, [&](MetalKernelFunction& fn) {
    fn.setArg(0, loss);
    fn.setArg(1, grad_site_rgba);
    fn.setArg(2, config_i32);
    fn.dispatch((uint64_t)std::max<int64_t>(sites_f32.size(0), 1), threads);
  });
  launch(k.fused_slab_affine_coeff16_mse_vjp_direct_atomic_framegroup16_cached_rgb_only, [&](MetalKernelFunction& fn) {
    fn.setArg(0, row_index_i32);
    fn.setArg(1, candidate_row_offsets_i32);
    fn.setArg(2, candidate_depth_coeff_f16);
    fn.setArg(3, sites_f32);
    fn.setArg(4, site_rgba_f32);
    fn.setArg(5, ray_coeff_f32);
    fn.setArg(6, frame_t_f32);
    fn.setArg(7, target_rgb_f32);
    fn.setArg(8, config_i32);
    fn.setArg(9, config_f32);
    fn.setArg(10, loss);
    fn.setArg(11, grad_site_rgba);
    fn.dispatch(total, framegroup_threads);
  });
  return std::make_tuple(loss, grad_site_rgba);
}

std::tuple<torch::Tensor, torch::Tensor>
metal_fused_slab_affine_coeff16_ownerupdate_mse_vjp_direct_atomic_rgb_only(
    const torch::Tensor& row_index_i32,
    const torch::Tensor& candidate_row_offsets_i32,
    const torch::Tensor& candidate_boundary_ids_i32,
    const torch::Tensor& candidate_depth_coeff_f16,
    const torch::Tensor& boundary_site_pairs_i32,
    const torch::Tensor& sites_f32,
    const torch::Tensor& site_rgba_f32,
    const torch::Tensor& ray_coeff_f32,
    const torch::Tensor& frame_t_f32,
    const torch::Tensor& target_rgb_f32,
    const torch::Tensor& config_i32,
    const torch::Tensor& config_f32) {
  check_i32_mps_1d_any(row_index_i32, "row_index_i32");
  check_i32_mps_1d_any(candidate_row_offsets_i32, "candidate_row_offsets_i32");
  check_i32_mps_1d_any(candidate_boundary_ids_i32, "candidate_boundary_ids_i32");
  check_half_mps_2d(candidate_depth_coeff_f16, "candidate_depth_coeff_f16", 4);
  check_i32_mps_2d(boundary_site_pairs_i32, "boundary_site_pairs_i32", 2);
  check_float_mps_2d(sites_f32, "sites_f32", 5);
  check_float_mps_2d(site_rgba_f32, "site_rgba_f32", 4);
  check_float_mps_2d(ray_coeff_f32, "ray_coeff_f32", 12);
  TORCH_CHECK(frame_t_f32.device().is_mps(), "frame_t_f32 must be on MPS");
  TORCH_CHECK(frame_t_f32.scalar_type() == torch::kFloat32, "frame_t_f32 must be float32");
  TORCH_CHECK(frame_t_f32.dim() == 1, "frame_t_f32 must have shape [T]");
  TORCH_CHECK(frame_t_f32.is_contiguous(), "frame_t_f32 must be contiguous");
  TORCH_CHECK(target_rgb_f32.device().is_mps(), "target_rgb_f32 must be on MPS");
  TORCH_CHECK(target_rgb_f32.scalar_type() == torch::kFloat32, "target_rgb_f32 must be float32");
  TORCH_CHECK(target_rgb_f32.dim() == 3 && target_rgb_f32.size(2) == 3, "target_rgb_f32 must have shape [K,T,3]");
  TORCH_CHECK(target_rgb_f32.is_contiguous(), "target_rgb_f32 must be contiguous");
  check_i32_mps_1d(config_i32, "config_i32", 8);
  TORCH_CHECK(config_f32.device().is_mps(), "config_f32 must be on MPS");
  TORCH_CHECK(config_f32.scalar_type() == torch::kFloat32, "config_f32 must be float32");
  TORCH_CHECK(config_f32.dim() == 1 && config_f32.size(0) == 4, "config_f32 must have shape [4]");
  TORCH_CHECK(config_f32.is_contiguous(), "config_f32 must be contiguous");

  auto config_i32_cpu = config_i32.cpu();
  const int32_t* config = config_i32_cpu.data_ptr<int32_t>();
  TORCH_CHECK(config[1] == ray_coeff_f32.size(0), "config_i32[1] must match track count");
  TORCH_CHECK(config[2] == sites_f32.size(0), "config_i32[2] must match site count");
  TORCH_CHECK(config[3] == frame_t_f32.size(0), "config_i32[3] must match frame count");
  TORCH_CHECK(config[4] > 0, "config_i32[4] time_slab_count must be positive");
  TORCH_CHECK(config[5] > 0, "config_i32[5] row_count must be positive");
  TORCH_CHECK(config[6] >= 0, "config_i32[6] candidate_count must be nonnegative");
  TORCH_CHECK(config[7] == boundary_site_pairs_i32.size(0), "config_i32[7] must match boundary count");
  TORCH_CHECK(ray_coeff_f32.size(0) > 0, "mixed affine coeff16 ownerupdate fused MSE VJP requires at least one track");
  TORCH_CHECK(frame_t_f32.size(0) > 0, "mixed affine coeff16 ownerupdate fused MSE VJP requires at least one frame");
  TORCH_CHECK(sites_f32.size(0) > 0, "mixed affine coeff16 ownerupdate fused MSE VJP requires at least one site");
  TORCH_CHECK(
      sites_f32.size(0) <= 64, "mixed affine coeff16 ownerupdate fused MSE VJP currently supports at most 64 sites");
  TORCH_CHECK(site_rgba_f32.size(0) == sites_f32.size(0), "site RGBA count mismatch");
  TORCH_CHECK(row_index_i32.size(0) == ray_coeff_f32.size(0), "row_index_i32 length must match track count");
  TORCH_CHECK(
      candidate_row_offsets_i32.size(0) == (int64_t)config[5] * (int64_t)config[4] + 1,
      "candidate_row_offsets_i32 length must be row_count * time_slab_count + 1");
  TORCH_CHECK(candidate_boundary_ids_i32.size(0) == config[6], "candidate_boundary_ids_i32 length mismatch");
  TORCH_CHECK(
      candidate_depth_coeff_f16.size(0) == config[6],
      "candidate_depth_coeff_f16 row count must match candidate_count");
  TORCH_CHECK(
      target_rgb_f32.size(0) == ray_coeff_f32.size(0) && target_rgb_f32.size(1) == frame_t_f32.size(0),
      "target_rgb_f32 shape mismatch");

  auto loss = torch::empty({1}, site_rgba_f32.options().dtype(torch::kFloat32));
  auto grad_site_rgba = torch::empty({sites_f32.size(0), 4}, sites_f32.options().dtype(torch::kFloat32));
  const uint64_t total = (uint64_t)ray_coeff_f32.size(0) * (uint64_t)frame_t_f32.size(0);
  auto& k = kernels();
  constexpr uint64_t threads = 256ull;
  launch(k.clear_affine_loss_site_rgba_grad, [&](MetalKernelFunction& fn) {
    fn.setArg(0, loss);
    fn.setArg(1, grad_site_rgba);
    fn.setArg(2, config_i32);
    fn.dispatch((uint64_t)std::max<int64_t>(sites_f32.size(0), 1), threads);
  });
  launch(k.fused_slab_affine_coeff16_ownerupdate_mse_vjp_direct_atomic_rgb_only, [&](MetalKernelFunction& fn) {
    fn.setArg(0, row_index_i32);
    fn.setArg(1, candidate_row_offsets_i32);
    fn.setArg(2, candidate_boundary_ids_i32);
    fn.setArg(3, candidate_depth_coeff_f16);
    fn.setArg(4, boundary_site_pairs_i32);
    fn.setArg(5, sites_f32);
    fn.setArg(6, site_rgba_f32);
    fn.setArg(7, ray_coeff_f32);
    fn.setArg(8, frame_t_f32);
    fn.setArg(9, target_rgb_f32);
    fn.setArg(10, config_i32);
    fn.setArg(11, config_f32);
    fn.setArg(12, loss);
    fn.setArg(13, grad_site_rgba);
    fn.dispatch(total, threads);
  });
  return std::make_tuple(loss, grad_site_rgba);
}

std::tuple<torch::Tensor, torch::Tensor>
metal_fused_slab_affine_coeff16_ownerupdate_i16_mse_vjp_direct_atomic_rgb_only(
    const torch::Tensor& row_index_i32,
    const torch::Tensor& candidate_row_offsets_i32,
    const torch::Tensor& candidate_boundary_ids_i16,
    const torch::Tensor& candidate_depth_coeff_f16,
    const torch::Tensor& boundary_site_pairs_i16,
    const torch::Tensor& sites_f32,
    const torch::Tensor& site_rgba_f32,
    const torch::Tensor& ray_coeff_f32,
    const torch::Tensor& frame_t_f32,
    const torch::Tensor& target_rgb_f32,
    const torch::Tensor& config_i32,
    const torch::Tensor& config_f32) {
  check_i32_mps_1d_any(row_index_i32, "row_index_i32");
  check_i32_mps_1d_any(candidate_row_offsets_i32, "candidate_row_offsets_i32");
  check_i16_mps_1d_any(candidate_boundary_ids_i16, "candidate_boundary_ids_i16");
  check_half_mps_2d(candidate_depth_coeff_f16, "candidate_depth_coeff_f16", 4);
  check_i16_mps_2d(boundary_site_pairs_i16, "boundary_site_pairs_i16", 2);
  check_float_mps_2d(sites_f32, "sites_f32", 5);
  check_float_mps_2d(site_rgba_f32, "site_rgba_f32", 4);
  check_float_mps_2d(ray_coeff_f32, "ray_coeff_f32", 12);
  TORCH_CHECK(frame_t_f32.device().is_mps(), "frame_t_f32 must be on MPS");
  TORCH_CHECK(frame_t_f32.scalar_type() == torch::kFloat32, "frame_t_f32 must be float32");
  TORCH_CHECK(frame_t_f32.dim() == 1, "frame_t_f32 must have shape [T]");
  TORCH_CHECK(frame_t_f32.is_contiguous(), "frame_t_f32 must be contiguous");
  TORCH_CHECK(target_rgb_f32.device().is_mps(), "target_rgb_f32 must be on MPS");
  TORCH_CHECK(target_rgb_f32.scalar_type() == torch::kFloat32, "target_rgb_f32 must be float32");
  TORCH_CHECK(target_rgb_f32.dim() == 3 && target_rgb_f32.size(2) == 3, "target_rgb_f32 must have shape [K,T,3]");
  TORCH_CHECK(target_rgb_f32.is_contiguous(), "target_rgb_f32 must be contiguous");
  check_i32_mps_1d(config_i32, "config_i32", 8);
  TORCH_CHECK(config_f32.device().is_mps(), "config_f32 must be on MPS");
  TORCH_CHECK(config_f32.scalar_type() == torch::kFloat32, "config_f32 must be float32");
  TORCH_CHECK(config_f32.dim() == 1 && config_f32.size(0) == 4, "config_f32 must have shape [4]");
  TORCH_CHECK(config_f32.is_contiguous(), "config_f32 must be contiguous");

  auto config_i32_cpu = config_i32.cpu();
  const int32_t* config = config_i32_cpu.data_ptr<int32_t>();
  TORCH_CHECK(config[1] == ray_coeff_f32.size(0), "config_i32[1] must match track count");
  TORCH_CHECK(config[2] == sites_f32.size(0), "config_i32[2] must match site count");
  TORCH_CHECK(config[3] == frame_t_f32.size(0), "config_i32[3] must match frame count");
  TORCH_CHECK(config[4] > 0, "config_i32[4] time_slab_count must be positive");
  TORCH_CHECK(config[5] > 0, "config_i32[5] row_count must be positive");
  TORCH_CHECK(config[6] >= 0, "config_i32[6] candidate_count must be nonnegative");
  TORCH_CHECK(config[7] == boundary_site_pairs_i16.size(0), "config_i32[7] must match boundary count");
  TORCH_CHECK(config[7] <= 32767, "packed ownerupdate boundary count must fit int16");
  TORCH_CHECK(ray_coeff_f32.size(0) > 0, "mixed affine coeff16 ownerupdate-i16 fused MSE VJP requires at least one track");
  TORCH_CHECK(frame_t_f32.size(0) > 0, "mixed affine coeff16 ownerupdate-i16 fused MSE VJP requires at least one frame");
  TORCH_CHECK(sites_f32.size(0) > 0, "mixed affine coeff16 ownerupdate-i16 fused MSE VJP requires at least one site");
  TORCH_CHECK(
      sites_f32.size(0) <= 64, "mixed affine coeff16 ownerupdate-i16 fused MSE VJP currently supports at most 64 sites");
  TORCH_CHECK(site_rgba_f32.size(0) == sites_f32.size(0), "site RGBA count mismatch");
  TORCH_CHECK(row_index_i32.size(0) == ray_coeff_f32.size(0), "row_index_i32 length must match track count");
  TORCH_CHECK(
      candidate_row_offsets_i32.size(0) == (int64_t)config[5] * (int64_t)config[4] + 1,
      "candidate_row_offsets_i32 length must be row_count * time_slab_count + 1");
  TORCH_CHECK(candidate_boundary_ids_i16.size(0) == config[6], "candidate_boundary_ids_i16 length mismatch");
  TORCH_CHECK(
      candidate_depth_coeff_f16.size(0) == config[6],
      "candidate_depth_coeff_f16 row count must match candidate_count");
  TORCH_CHECK(
      target_rgb_f32.size(0) == ray_coeff_f32.size(0) && target_rgb_f32.size(1) == frame_t_f32.size(0),
      "target_rgb_f32 shape mismatch");

  auto loss = torch::empty({1}, site_rgba_f32.options().dtype(torch::kFloat32));
  auto grad_site_rgba = torch::empty({sites_f32.size(0), 4}, sites_f32.options().dtype(torch::kFloat32));
  const uint64_t total = (uint64_t)ray_coeff_f32.size(0) * (uint64_t)frame_t_f32.size(0);
  auto& k = kernels();
  constexpr uint64_t threads = 256ull;
  launch(k.clear_affine_loss_site_rgba_grad, [&](MetalKernelFunction& fn) {
    fn.setArg(0, loss);
    fn.setArg(1, grad_site_rgba);
    fn.setArg(2, config_i32);
    fn.dispatch((uint64_t)std::max<int64_t>(sites_f32.size(0), 1), threads);
  });
  launch(k.fused_slab_affine_coeff16_ownerupdate_i16_mse_vjp_direct_atomic_rgb_only, [&](MetalKernelFunction& fn) {
    fn.setArg(0, row_index_i32);
    fn.setArg(1, candidate_row_offsets_i32);
    fn.setArg(2, candidate_boundary_ids_i16);
    fn.setArg(3, candidate_depth_coeff_f16);
    fn.setArg(4, boundary_site_pairs_i16);
    fn.setArg(5, sites_f32);
    fn.setArg(6, site_rgba_f32);
    fn.setArg(7, ray_coeff_f32);
    fn.setArg(8, frame_t_f32);
    fn.setArg(9, target_rgb_f32);
    fn.setArg(10, config_i32);
    fn.setArg(11, config_f32);
    fn.setArg(12, loss);
    fn.setArg(13, grad_site_rgba);
    fn.dispatch(total, threads);
  });
  return std::make_tuple(loss, grad_site_rgba);
}

std::tuple<torch::Tensor, torch::Tensor>
metal_fused_slab_affine_coeff16_ownerkeep_i16_mse_vjp_direct_atomic_rgb_only(
    const torch::Tensor& row_index_i32,
    const torch::Tensor& candidate_row_offsets_i32,
    const torch::Tensor& candidate_boundary_ids_i16,
    const torch::Tensor& candidate_depth_coeff_f16,
    const torch::Tensor& boundary_site_pairs_i16,
    const torch::Tensor& sites_f32,
    const torch::Tensor& site_rgba_f32,
    const torch::Tensor& ray_coeff_f32,
    const torch::Tensor& frame_t_f32,
    const torch::Tensor& target_rgb_f32,
    const torch::Tensor& config_i32,
    const torch::Tensor& config_f32) {
  check_i32_mps_1d_any(row_index_i32, "row_index_i32");
  check_i32_mps_1d_any(candidate_row_offsets_i32, "candidate_row_offsets_i32");
  check_i16_mps_1d_any(candidate_boundary_ids_i16, "candidate_boundary_ids_i16");
  check_half_mps_2d(candidate_depth_coeff_f16, "candidate_depth_coeff_f16", 4);
  check_i16_mps_2d(boundary_site_pairs_i16, "boundary_site_pairs_i16", 2);
  check_float_mps_2d(sites_f32, "sites_f32", 5);
  check_float_mps_2d(site_rgba_f32, "site_rgba_f32", 4);
  check_float_mps_2d(ray_coeff_f32, "ray_coeff_f32", 12);
  TORCH_CHECK(frame_t_f32.device().is_mps(), "frame_t_f32 must be on MPS");
  TORCH_CHECK(frame_t_f32.scalar_type() == torch::kFloat32, "frame_t_f32 must be float32");
  TORCH_CHECK(frame_t_f32.dim() == 1, "frame_t_f32 must have shape [T]");
  TORCH_CHECK(frame_t_f32.is_contiguous(), "frame_t_f32 must be contiguous");
  TORCH_CHECK(target_rgb_f32.device().is_mps(), "target_rgb_f32 must be on MPS");
  TORCH_CHECK(target_rgb_f32.scalar_type() == torch::kFloat32, "target_rgb_f32 must be float32");
  TORCH_CHECK(target_rgb_f32.dim() == 3 && target_rgb_f32.size(2) == 3, "target_rgb_f32 must have shape [K,T,3]");
  TORCH_CHECK(target_rgb_f32.is_contiguous(), "target_rgb_f32 must be contiguous");
  check_i32_mps_1d(config_i32, "config_i32", 8);
  TORCH_CHECK(config_f32.device().is_mps(), "config_f32 must be on MPS");
  TORCH_CHECK(config_f32.scalar_type() == torch::kFloat32, "config_f32 must be float32");
  TORCH_CHECK(config_f32.dim() == 1 && config_f32.size(0) == 4, "config_f32 must have shape [4]");
  TORCH_CHECK(config_f32.is_contiguous(), "config_f32 must be contiguous");

  auto config_i32_cpu = config_i32.cpu();
  const int32_t* config = config_i32_cpu.data_ptr<int32_t>();
  TORCH_CHECK(config[1] == ray_coeff_f32.size(0), "config_i32[1] must match track count");
  TORCH_CHECK(config[2] == sites_f32.size(0), "config_i32[2] must match site count");
  TORCH_CHECK(config[3] == frame_t_f32.size(0), "config_i32[3] must match frame count");
  TORCH_CHECK(config[4] > 0, "config_i32[4] time_slab_count must be positive");
  TORCH_CHECK(config[5] > 0, "config_i32[5] row_count must be positive");
  TORCH_CHECK(config[6] >= 0, "config_i32[6] candidate_count must be nonnegative");
  TORCH_CHECK(config[7] == boundary_site_pairs_i16.size(0), "config_i32[7] must match boundary count");
  TORCH_CHECK(config[7] <= 32767, "packed ownerkeep boundary count must fit int16");
  TORCH_CHECK(ray_coeff_f32.size(0) > 0, "mixed affine coeff16 ownerkeep-i16 fused MSE VJP requires at least one track");
  TORCH_CHECK(frame_t_f32.size(0) > 0, "mixed affine coeff16 ownerkeep-i16 fused MSE VJP requires at least one frame");
  TORCH_CHECK(sites_f32.size(0) > 0, "mixed affine coeff16 ownerkeep-i16 fused MSE VJP requires at least one site");
  TORCH_CHECK(
      sites_f32.size(0) <= 64, "mixed affine coeff16 ownerkeep-i16 fused MSE VJP currently supports at most 64 sites");
  TORCH_CHECK(site_rgba_f32.size(0) == sites_f32.size(0), "site RGBA count mismatch");
  TORCH_CHECK(row_index_i32.size(0) == ray_coeff_f32.size(0), "row_index_i32 length must match track count");
  TORCH_CHECK(
      candidate_row_offsets_i32.size(0) == (int64_t)config[5] * (int64_t)config[4] + 1,
      "candidate_row_offsets_i32 length must be row_count * time_slab_count + 1");
  TORCH_CHECK(candidate_boundary_ids_i16.size(0) == config[6], "candidate_boundary_ids_i16 length mismatch");
  TORCH_CHECK(
      candidate_depth_coeff_f16.size(0) == config[6],
      "candidate_depth_coeff_f16 row count must match candidate_count");
  TORCH_CHECK(
      target_rgb_f32.size(0) == ray_coeff_f32.size(0) && target_rgb_f32.size(1) == frame_t_f32.size(0),
      "target_rgb_f32 shape mismatch");

  auto loss = torch::empty({1}, site_rgba_f32.options().dtype(torch::kFloat32));
  auto grad_site_rgba = torch::empty({sites_f32.size(0), 4}, sites_f32.options().dtype(torch::kFloat32));
  const uint64_t total = (uint64_t)ray_coeff_f32.size(0) * (uint64_t)frame_t_f32.size(0);
  auto& k = kernels();
  constexpr uint64_t threads = 256ull;
  launch(k.clear_affine_loss_site_rgba_grad, [&](MetalKernelFunction& fn) {
    fn.setArg(0, loss);
    fn.setArg(1, grad_site_rgba);
    fn.setArg(2, config_i32);
    fn.dispatch((uint64_t)std::max<int64_t>(sites_f32.size(0), 1), threads);
  });
  launch(k.fused_slab_affine_coeff16_ownerkeep_i16_mse_vjp_direct_atomic_rgb_only, [&](MetalKernelFunction& fn) {
    fn.setArg(0, row_index_i32);
    fn.setArg(1, candidate_row_offsets_i32);
    fn.setArg(2, candidate_boundary_ids_i16);
    fn.setArg(3, candidate_depth_coeff_f16);
    fn.setArg(4, boundary_site_pairs_i16);
    fn.setArg(5, sites_f32);
    fn.setArg(6, site_rgba_f32);
    fn.setArg(7, ray_coeff_f32);
    fn.setArg(8, frame_t_f32);
    fn.setArg(9, target_rgb_f32);
    fn.setArg(10, config_i32);
    fn.setArg(11, config_f32);
    fn.setArg(12, loss);
    fn.setArg(13, grad_site_rgba);
    fn.dispatch(total, threads);
  });
  return std::make_tuple(loss, grad_site_rgba);
}

std::tuple<torch::Tensor, torch::Tensor>
metal_fused_slab_affine_coeff16_ownerkeep_mse_vjp_direct_atomic_rgb_only(
    const torch::Tensor& row_index_i32,
    const torch::Tensor& candidate_row_offsets_i32,
    const torch::Tensor& candidate_boundary_ids_i32,
    const torch::Tensor& candidate_depth_coeff_f16,
    const torch::Tensor& boundary_site_pairs_i32,
    const torch::Tensor& sites_f32,
    const torch::Tensor& site_rgba_f32,
    const torch::Tensor& ray_coeff_f32,
    const torch::Tensor& frame_t_f32,
    const torch::Tensor& target_rgb_f32,
    const torch::Tensor& config_i32,
    const torch::Tensor& config_f32) {
  check_i32_mps_1d_any(row_index_i32, "row_index_i32");
  check_i32_mps_1d_any(candidate_row_offsets_i32, "candidate_row_offsets_i32");
  check_i32_mps_1d_any(candidate_boundary_ids_i32, "candidate_boundary_ids_i32");
  check_half_mps_2d(candidate_depth_coeff_f16, "candidate_depth_coeff_f16", 4);
  check_i32_mps_2d(boundary_site_pairs_i32, "boundary_site_pairs_i32", 2);
  check_float_mps_2d(sites_f32, "sites_f32", 5);
  check_float_mps_2d(site_rgba_f32, "site_rgba_f32", 4);
  check_float_mps_2d(ray_coeff_f32, "ray_coeff_f32", 12);
  TORCH_CHECK(frame_t_f32.device().is_mps(), "frame_t_f32 must be on MPS");
  TORCH_CHECK(frame_t_f32.scalar_type() == torch::kFloat32, "frame_t_f32 must be float32");
  TORCH_CHECK(frame_t_f32.dim() == 1, "frame_t_f32 must have shape [T]");
  TORCH_CHECK(frame_t_f32.is_contiguous(), "frame_t_f32 must be contiguous");
  TORCH_CHECK(target_rgb_f32.device().is_mps(), "target_rgb_f32 must be on MPS");
  TORCH_CHECK(target_rgb_f32.scalar_type() == torch::kFloat32, "target_rgb_f32 must be float32");
  TORCH_CHECK(target_rgb_f32.dim() == 3 && target_rgb_f32.size(2) == 3, "target_rgb_f32 must have shape [K,T,3]");
  TORCH_CHECK(target_rgb_f32.is_contiguous(), "target_rgb_f32 must be contiguous");
  check_i32_mps_1d(config_i32, "config_i32", 8);
  TORCH_CHECK(config_f32.device().is_mps(), "config_f32 must be on MPS");
  TORCH_CHECK(config_f32.scalar_type() == torch::kFloat32, "config_f32 must be float32");
  TORCH_CHECK(config_f32.dim() == 1 && config_f32.size(0) == 4, "config_f32 must have shape [4]");
  TORCH_CHECK(config_f32.is_contiguous(), "config_f32 must be contiguous");

  auto config_i32_cpu = config_i32.cpu();
  const int32_t* config = config_i32_cpu.data_ptr<int32_t>();
  TORCH_CHECK(config[1] == ray_coeff_f32.size(0), "config_i32[1] must match track count");
  TORCH_CHECK(config[2] == sites_f32.size(0), "config_i32[2] must match site count");
  TORCH_CHECK(config[3] == frame_t_f32.size(0), "config_i32[3] must match frame count");
  TORCH_CHECK(config[4] > 0, "config_i32[4] time_slab_count must be positive");
  TORCH_CHECK(config[5] > 0, "config_i32[5] row_count must be positive");
  TORCH_CHECK(config[6] >= 0, "config_i32[6] candidate_count must be nonnegative");
  TORCH_CHECK(config[7] == boundary_site_pairs_i32.size(0), "config_i32[7] must match boundary count");
  TORCH_CHECK(ray_coeff_f32.size(0) > 0, "mixed affine coeff16 ownerkeep fused MSE VJP requires at least one track");
  TORCH_CHECK(frame_t_f32.size(0) > 0, "mixed affine coeff16 ownerkeep fused MSE VJP requires at least one frame");
  TORCH_CHECK(sites_f32.size(0) > 0, "mixed affine coeff16 ownerkeep fused MSE VJP requires at least one site");
  TORCH_CHECK(
      sites_f32.size(0) <= 64, "mixed affine coeff16 ownerkeep fused MSE VJP currently supports at most 64 sites");
  TORCH_CHECK(site_rgba_f32.size(0) == sites_f32.size(0), "site RGBA count mismatch");
  TORCH_CHECK(row_index_i32.size(0) == ray_coeff_f32.size(0), "row_index_i32 length must match track count");
  TORCH_CHECK(
      candidate_row_offsets_i32.size(0) == (int64_t)config[5] * (int64_t)config[4] + 1,
      "candidate_row_offsets_i32 length must be row_count * time_slab_count + 1");
  TORCH_CHECK(candidate_boundary_ids_i32.size(0) == config[6], "candidate_boundary_ids_i32 length mismatch");
  TORCH_CHECK(
      candidate_depth_coeff_f16.size(0) == config[6],
      "candidate_depth_coeff_f16 row count must match candidate_count");
  TORCH_CHECK(
      target_rgb_f32.size(0) == ray_coeff_f32.size(0) && target_rgb_f32.size(1) == frame_t_f32.size(0),
      "target_rgb_f32 shape mismatch");

  auto loss = torch::empty({1}, site_rgba_f32.options().dtype(torch::kFloat32));
  auto grad_site_rgba = torch::empty({sites_f32.size(0), 4}, sites_f32.options().dtype(torch::kFloat32));
  const uint64_t total = (uint64_t)ray_coeff_f32.size(0) * (uint64_t)frame_t_f32.size(0);
  auto& k = kernels();
  constexpr uint64_t threads = 256ull;
  launch(k.clear_affine_loss_site_rgba_grad, [&](MetalKernelFunction& fn) {
    fn.setArg(0, loss);
    fn.setArg(1, grad_site_rgba);
    fn.setArg(2, config_i32);
    fn.dispatch((uint64_t)std::max<int64_t>(sites_f32.size(0), 1), threads);
  });
  launch(k.fused_slab_affine_coeff16_ownerkeep_mse_vjp_direct_atomic_rgb_only, [&](MetalKernelFunction& fn) {
    fn.setArg(0, row_index_i32);
    fn.setArg(1, candidate_row_offsets_i32);
    fn.setArg(2, candidate_boundary_ids_i32);
    fn.setArg(3, candidate_depth_coeff_f16);
    fn.setArg(4, boundary_site_pairs_i32);
    fn.setArg(5, sites_f32);
    fn.setArg(6, site_rgba_f32);
    fn.setArg(7, ray_coeff_f32);
    fn.setArg(8, frame_t_f32);
    fn.setArg(9, target_rgb_f32);
    fn.setArg(10, config_i32);
    fn.setArg(11, config_f32);
    fn.setArg(12, loss);
    fn.setArg(13, grad_site_rgba);
    fn.dispatch(total, threads);
  });
  return std::make_tuple(loss, grad_site_rgba);
}

std::tuple<torch::Tensor, torch::Tensor> metal_fused_slab_affine_coeff16_mse_vjp_direct_atomic_track_rgb_only(
    const torch::Tensor& row_index_i32,
    const torch::Tensor& candidate_row_offsets_i32,
    const torch::Tensor& candidate_depth_coeff_f16,
    const torch::Tensor& sites_f32,
    const torch::Tensor& site_rgba_f32,
    const torch::Tensor& ray_coeff_f32,
    const torch::Tensor& frame_t_f32,
    const torch::Tensor& target_rgb_f32,
    const torch::Tensor& config_i32,
    const torch::Tensor& config_f32) {
  check_i32_mps_1d_any(row_index_i32, "row_index_i32");
  check_i32_mps_1d_any(candidate_row_offsets_i32, "candidate_row_offsets_i32");
  check_half_mps_2d(candidate_depth_coeff_f16, "candidate_depth_coeff_f16", 4);
  check_float_mps_2d(sites_f32, "sites_f32", 5);
  check_float_mps_2d(site_rgba_f32, "site_rgba_f32", 4);
  check_float_mps_2d(ray_coeff_f32, "ray_coeff_f32", 12);
  TORCH_CHECK(frame_t_f32.device().is_mps(), "frame_t_f32 must be on MPS");
  TORCH_CHECK(frame_t_f32.scalar_type() == torch::kFloat32, "frame_t_f32 must be float32");
  TORCH_CHECK(frame_t_f32.dim() == 1, "frame_t_f32 must have shape [T]");
  TORCH_CHECK(frame_t_f32.is_contiguous(), "frame_t_f32 must be contiguous");
  TORCH_CHECK(target_rgb_f32.device().is_mps(), "target_rgb_f32 must be on MPS");
  TORCH_CHECK(target_rgb_f32.scalar_type() == torch::kFloat32, "target_rgb_f32 must be float32");
  TORCH_CHECK(target_rgb_f32.dim() == 3 && target_rgb_f32.size(2) == 3, "target_rgb_f32 must have shape [K,T,3]");
  TORCH_CHECK(target_rgb_f32.is_contiguous(), "target_rgb_f32 must be contiguous");
  check_i32_mps_1d(config_i32, "config_i32", 8);
  TORCH_CHECK(config_f32.device().is_mps(), "config_f32 must be on MPS");
  TORCH_CHECK(config_f32.scalar_type() == torch::kFloat32, "config_f32 must be float32");
  TORCH_CHECK(config_f32.dim() == 1 && config_f32.size(0) == 4, "config_f32 must have shape [4]");
  TORCH_CHECK(config_f32.is_contiguous(), "config_f32 must be contiguous");

  auto config_i32_cpu = config_i32.cpu();
  const int32_t* config = config_i32_cpu.data_ptr<int32_t>();
  TORCH_CHECK(config[1] == ray_coeff_f32.size(0), "config_i32[1] must match track count");
  TORCH_CHECK(config[2] == sites_f32.size(0), "config_i32[2] must match site count");
  TORCH_CHECK(config[3] == frame_t_f32.size(0), "config_i32[3] must match frame count");
  TORCH_CHECK(config[4] > 0, "config_i32[4] time_slab_count must be positive");
  TORCH_CHECK(config[5] > 0, "config_i32[5] row_count must be positive");
  TORCH_CHECK(config[6] >= 0, "config_i32[6] candidate_count must be nonnegative");
  TORCH_CHECK(ray_coeff_f32.size(0) > 0, "mixed affine coeff16 track fused MSE VJP requires at least one track");
  TORCH_CHECK(frame_t_f32.size(0) > 0, "mixed affine coeff16 track fused MSE VJP requires at least one frame");
  TORCH_CHECK(sites_f32.size(0) > 0, "mixed affine coeff16 track fused MSE VJP requires at least one site");
  TORCH_CHECK(sites_f32.size(0) <= 64, "mixed affine coeff16 track fused MSE VJP currently supports at most 64 sites");
  TORCH_CHECK(site_rgba_f32.size(0) == sites_f32.size(0), "site RGBA count mismatch");
  TORCH_CHECK(row_index_i32.size(0) == ray_coeff_f32.size(0), "row_index_i32 length must match track count");
  TORCH_CHECK(
      candidate_row_offsets_i32.size(0) == (int64_t)config[5] * (int64_t)config[4] + 1,
      "candidate_row_offsets_i32 length must be row_count * time_slab_count + 1");
  TORCH_CHECK(
      candidate_depth_coeff_f16.size(0) == config[6],
      "candidate_depth_coeff_f16 row count must match candidate_count");
  TORCH_CHECK(
      target_rgb_f32.size(0) == ray_coeff_f32.size(0) && target_rgb_f32.size(1) == frame_t_f32.size(0),
      "target_rgb_f32 shape mismatch");

  auto loss = torch::empty({1}, site_rgba_f32.options().dtype(torch::kFloat32));
  auto grad_site_rgba = torch::empty({sites_f32.size(0), 4}, sites_f32.options().dtype(torch::kFloat32));
  auto& k = kernels();
  constexpr uint64_t threads = 256ull;
  launch(k.clear_affine_loss_site_rgba_grad, [&](MetalKernelFunction& fn) {
    fn.setArg(0, loss);
    fn.setArg(1, grad_site_rgba);
    fn.setArg(2, config_i32);
    fn.dispatch((uint64_t)std::max<int64_t>(sites_f32.size(0), 1), threads);
  });
  launch(k.fused_slab_affine_coeff16_mse_vjp_direct_atomic_track_rgb_only, [&](MetalKernelFunction& fn) {
    fn.setArg(0, row_index_i32);
    fn.setArg(1, candidate_row_offsets_i32);
    fn.setArg(2, candidate_depth_coeff_f16);
    fn.setArg(3, sites_f32);
    fn.setArg(4, site_rgba_f32);
    fn.setArg(5, ray_coeff_f32);
    fn.setArg(6, frame_t_f32);
    fn.setArg(7, target_rgb_f32);
    fn.setArg(8, config_i32);
    fn.setArg(9, config_f32);
    fn.setArg(10, loss);
    fn.setArg(11, grad_site_rgba);
    fn.dispatch((uint64_t)ray_coeff_f32.size(0), threads);
  });
  return std::make_tuple(loss, grad_site_rgba);
}

torch::Tensor metal_fused_slab_affine_num32_den16_vjp_direct_atomic_track(
    const torch::Tensor& row_index_i32,
    const torch::Tensor& candidate_row_offsets_i32,
    const torch::Tensor& candidate_depth_num_f32,
    const torch::Tensor& candidate_depth_den_f16,
    const torch::Tensor& sites_f32,
    const torch::Tensor& site_rgba_f32,
    const torch::Tensor& ray_coeff_f32,
    const torch::Tensor& frame_t_f32,
    const torch::Tensor& grad_rgb_f32,
    const torch::Tensor& grad_alpha_f32,
    const torch::Tensor& grad_depth_f32,
    const torch::Tensor& config_i32,
    const torch::Tensor& config_f32) {
  check_i32_mps_1d_any(row_index_i32, "row_index_i32");
  check_i32_mps_1d_any(candidate_row_offsets_i32, "candidate_row_offsets_i32");
  check_float_mps_2d(candidate_depth_num_f32, "candidate_depth_num_f32", 2);
  check_half_mps_2d(candidate_depth_den_f16, "candidate_depth_den_f16", 2);
  check_float_mps_2d(sites_f32, "sites_f32", 5);
  check_float_mps_2d(site_rgba_f32, "site_rgba_f32", 4);
  check_float_mps_2d(ray_coeff_f32, "ray_coeff_f32", 12);
  TORCH_CHECK(frame_t_f32.device().is_mps(), "frame_t_f32 must be on MPS");
  TORCH_CHECK(frame_t_f32.scalar_type() == torch::kFloat32, "frame_t_f32 must be float32");
  TORCH_CHECK(frame_t_f32.dim() == 1, "frame_t_f32 must have shape [T]");
  TORCH_CHECK(frame_t_f32.is_contiguous(), "frame_t_f32 must be contiguous");
  TORCH_CHECK(grad_rgb_f32.device().is_mps(), "grad_rgb_f32 must be on MPS");
  TORCH_CHECK(grad_rgb_f32.scalar_type() == torch::kFloat32, "grad_rgb_f32 must be float32");
  TORCH_CHECK(grad_rgb_f32.dim() == 3 && grad_rgb_f32.size(2) == 3, "grad_rgb_f32 must have shape [K,T,3]");
  TORCH_CHECK(grad_rgb_f32.is_contiguous(), "grad_rgb_f32 must be contiguous");
  TORCH_CHECK(grad_alpha_f32.device().is_mps(), "grad_alpha_f32 must be on MPS");
  TORCH_CHECK(grad_alpha_f32.scalar_type() == torch::kFloat32, "grad_alpha_f32 must be float32");
  TORCH_CHECK(grad_alpha_f32.dim() == 2, "grad_alpha_f32 must have shape [K,T]");
  TORCH_CHECK(grad_alpha_f32.is_contiguous(), "grad_alpha_f32 must be contiguous");
  TORCH_CHECK(grad_depth_f32.device().is_mps(), "grad_depth_f32 must be on MPS");
  TORCH_CHECK(grad_depth_f32.scalar_type() == torch::kFloat32, "grad_depth_f32 must be float32");
  TORCH_CHECK(grad_depth_f32.dim() == 2, "grad_depth_f32 must have shape [K,T]");
  TORCH_CHECK(grad_depth_f32.is_contiguous(), "grad_depth_f32 must be contiguous");
  check_i32_mps_1d(config_i32, "config_i32", 8);
  TORCH_CHECK(config_f32.device().is_mps(), "config_f32 must be on MPS");
  TORCH_CHECK(config_f32.scalar_type() == torch::kFloat32, "config_f32 must be float32");
  TORCH_CHECK(config_f32.dim() == 1 && config_f32.size(0) == 4, "config_f32 must have shape [4]");
  TORCH_CHECK(config_f32.is_contiguous(), "config_f32 must be contiguous");

  auto config_i32_cpu = config_i32.cpu();
  const int32_t* config = config_i32_cpu.data_ptr<int32_t>();
  TORCH_CHECK(config[1] == ray_coeff_f32.size(0), "config_i32[1] must match track count");
  TORCH_CHECK(config[2] == sites_f32.size(0), "config_i32[2] must match site count");
  TORCH_CHECK(config[3] == frame_t_f32.size(0), "config_i32[3] must match frame count");
  TORCH_CHECK(config[4] > 0, "config_i32[4] time_slab_count must be positive");
  TORCH_CHECK(config[5] > 0, "config_i32[5] row_count must be positive");
  TORCH_CHECK(config[6] >= 0, "config_i32[6] candidate_count must be nonnegative");
  TORCH_CHECK(ray_coeff_f32.size(0) > 0, "mixed affine track direct VJP requires at least one track");
  TORCH_CHECK(frame_t_f32.size(0) > 0, "mixed affine track direct VJP requires at least one frame");
  TORCH_CHECK(sites_f32.size(0) > 0, "mixed affine track direct VJP requires at least one site");
  TORCH_CHECK(sites_f32.size(0) <= 64, "mixed affine track direct VJP currently supports at most 64 sites");
  TORCH_CHECK(site_rgba_f32.size(0) == sites_f32.size(0), "site RGBA count mismatch");
  TORCH_CHECK(row_index_i32.size(0) == ray_coeff_f32.size(0), "row_index_i32 length must match track count");
  TORCH_CHECK(
      candidate_row_offsets_i32.size(0) == (int64_t)config[5] * (int64_t)config[4] + 1,
      "candidate_row_offsets_i32 length must be row_count * time_slab_count + 1");
  TORCH_CHECK(
      candidate_depth_num_f32.size(0) == config[6],
      "candidate_depth_num_f32 row count must match candidate_count");
  TORCH_CHECK(
      candidate_depth_den_f16.size(0) == config[6],
      "candidate_depth_den_f16 row count must match candidate_count");
  TORCH_CHECK(
      grad_rgb_f32.size(0) == ray_coeff_f32.size(0) && grad_rgb_f32.size(1) == frame_t_f32.size(0),
      "grad_rgb_f32 shape mismatch");
  TORCH_CHECK(
      grad_alpha_f32.size(0) == ray_coeff_f32.size(0) && grad_alpha_f32.size(1) == frame_t_f32.size(0),
      "grad_alpha_f32 shape mismatch");
  TORCH_CHECK(
      grad_depth_f32.size(0) == ray_coeff_f32.size(0) && grad_depth_f32.size(1) == frame_t_f32.size(0),
      "grad_depth_f32 shape mismatch");

  auto grad_site_rgba = torch::empty({sites_f32.size(0), 4}, sites_f32.options().dtype(torch::kFloat32));
  auto& k = kernels();
  constexpr uint64_t threads = 256ull;
  launch(k.clear_site_rgba_grad, [&](MetalKernelFunction& fn) {
    fn.setArg(0, grad_site_rgba);
    fn.setArg(1, config_i32);
    fn.dispatch((uint64_t)sites_f32.size(0), threads);
  });
  launch(k.fused_slab_affine_num32_den16_vjp_direct_atomic_track, [&](MetalKernelFunction& fn) {
    fn.setArg(0, row_index_i32);
    fn.setArg(1, candidate_row_offsets_i32);
    fn.setArg(2, candidate_depth_num_f32);
    fn.setArg(3, candidate_depth_den_f16);
    fn.setArg(4, sites_f32);
    fn.setArg(5, site_rgba_f32);
    fn.setArg(6, ray_coeff_f32);
    fn.setArg(7, frame_t_f32);
    fn.setArg(8, grad_rgb_f32);
    fn.setArg(9, grad_alpha_f32);
    fn.setArg(10, grad_depth_f32);
    fn.setArg(11, config_i32);
    fn.setArg(12, config_f32);
    fn.setArg(13, grad_site_rgba);
    fn.dispatch((uint64_t)ray_coeff_f32.size(0), threads);
  });
  return grad_site_rgba;
}

}  // namespace world_foam_lane2_fused_slab

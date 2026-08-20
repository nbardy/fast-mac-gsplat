#import <Foundation/Foundation.h>

#include <ATen/ATen.h>
#include <ATen/native/mps/MetalShaderLibrary.h>
#include <torch/extension.h>
#include <torch/mps.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace world_foam_lane2_fused_slab {
namespace {

using at::native::mps::DynamicMetalShaderLibrary;
using at::native::mps::MetalKernelFunction;

constexpr const char* kKineticNodeForwardOperator =
    "kinetic_precompiled_length_p0_lie_node_forward_into_launch_only_v1";
constexpr const char* kKineticNodeForwardMetalFunction =
    "wf2_kinetic_precompiled_length_p0_lie_node_forward_tensor";
constexpr const char* kKineticSampleAccumulateOperator =
    "kinetic_ragged_p0_lie_sample_accumulate_loss_only_launch_only";
constexpr const char* kKineticSampleAccumulateMetalFunction =
    "wf2_kinetic_ragged_p0_lie_sample_mse_vjp_accumulate_only_tensor";
constexpr const char* kKineticMaterialVjpOperator =
    "kinetic_precompiled_length_p0_lie_material_node_vjp_accumulate_launch_only";
constexpr const char* kKineticMaterialVjpMetalFunction =
    "wf2_kinetic_precompiled_length_p0_lie_material_node_vjp_tensor";

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
  std::shared_ptr<MetalKernelFunction> clear_endpoint_loss_site_rgba_boundary_grad;
  std::shared_ptr<MetalKernelFunction> clear_endpoint_loss_site_rgba_mobius_boundary_grad;
  std::shared_ptr<MetalKernelFunction> clear_fixed_word_p0_compiled_lie_grad;
  std::shared_ptr<MetalKernelFunction> sparse_mobius_incidence_lower;
  std::shared_ptr<MetalKernelFunction> sparse_mobius_incidence_boundary_vjp;
  std::shared_ptr<MetalKernelFunction> sparse_power_boundary_from_sites_launch_only;
  std::shared_ptr<MetalKernelFunction> sparse_power_boundary_site_vjp_launch_only;
  std::shared_ptr<MetalKernelFunction> fixed_word_p0_lie_node_forward;
  std::shared_ptr<MetalKernelFunction> kinetic_precompiled_length_p0_lie_node_forward;
  std::shared_ptr<MetalKernelFunction> fixed_word_p0_lie_sample_mse_vjp;
  std::shared_ptr<MetalKernelFunction> fixed_word_p0_lie_sample_mse_vjp_accumulate_only;
  std::shared_ptr<MetalKernelFunction> kinetic_ragged_p0_lie_sample_mse_vjp;
  std::shared_ptr<MetalKernelFunction> kinetic_ragged_p0_lie_sample_mse_vjp_accumulate_only;
  std::shared_ptr<MetalKernelFunction> fixed_word_p0_lie_node_vjp;
  std::shared_ptr<MetalKernelFunction> fixed_word_p0_lie_material_node_vjp;
  std::shared_ptr<MetalKernelFunction> kinetic_precompiled_length_p0_lie_node_vjp;
  std::shared_ptr<MetalKernelFunction> kinetic_precompiled_length_p0_lie_material_node_vjp;
  std::shared_ptr<MetalKernelFunction> kinetic_fused_direct_full_vjp_validate_v1;
  std::shared_ptr<MetalKernelFunction> kinetic_fused_direct_full_vjp_v1;
  std::shared_ptr<MetalKernelFunction> kinetic_fused_direct_full_vjp_finalize_v1;
  std::shared_ptr<MetalKernelFunction> kinetic_fused_union_full_vjp_validate_v2;
  std::shared_ptr<MetalKernelFunction> kinetic_fused_union_full_vjp_v2;
  std::shared_ptr<MetalKernelFunction> kinetic_fused_union_full_vjp_finalize_v2;
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
  std::shared_ptr<MetalKernelFunction> endpoint_record_delta_replace_factorized_packed_framegroup16_constant_state_mse_vjp_direct_atomic_rgb_boundary;
  std::shared_ptr<MetalKernelFunction> endpoint_record_delta_replace_factorized_packed_framegroup16_constant_state_p0_mse_vjp_sparse_mobius_rgb;
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
    out.clear_endpoint_loss_site_rgba_boundary_grad =
        lib->getKernelFunction("wf2_clear_endpoint_loss_site_rgba_boundary_grad_tensor");
    out.clear_endpoint_loss_site_rgba_mobius_boundary_grad =
        lib->getKernelFunction("wf2_clear_endpoint_loss_site_rgba_mobius_boundary_grad_tensor");
    out.clear_fixed_word_p0_compiled_lie_grad =
        lib->getKernelFunction("wf2_clear_fixed_word_p0_compiled_lie_grad_tensor");
    out.sparse_mobius_incidence_lower =
        lib->getKernelFunction("wf2_sparse_mobius_incidence_lower_tensor");
    out.sparse_mobius_incidence_boundary_vjp =
        lib->getKernelFunction("wf2_sparse_mobius_incidence_boundary_vjp_tensor");
    out.sparse_power_boundary_from_sites_launch_only =
        lib->getKernelFunction("wf2_sparse_power_boundary_from_sites_launch_only_tensor");
    out.sparse_power_boundary_site_vjp_launch_only =
        lib->getKernelFunction("wf2_sparse_power_boundary_site_vjp_launch_only_tensor");
    out.fixed_word_p0_lie_node_forward =
        lib->getKernelFunction("wf2_fixed_word_p0_lie_node_forward_tensor");
    out.kinetic_precompiled_length_p0_lie_node_forward =
        lib->getKernelFunction(kKineticNodeForwardMetalFunction);
    out.fixed_word_p0_lie_sample_mse_vjp =
        lib->getKernelFunction("wf2_fixed_word_p0_lie_sample_mse_vjp_tensor");
    out.fixed_word_p0_lie_sample_mse_vjp_accumulate_only =
        lib->getKernelFunction("wf2_fixed_word_p0_lie_sample_mse_vjp_accumulate_only_tensor");
    out.kinetic_ragged_p0_lie_sample_mse_vjp =
        lib->getKernelFunction("wf2_kinetic_ragged_p0_lie_sample_mse_vjp_tensor");
    out.kinetic_ragged_p0_lie_sample_mse_vjp_accumulate_only =
        lib->getKernelFunction(kKineticSampleAccumulateMetalFunction);
    out.fixed_word_p0_lie_node_vjp =
        lib->getKernelFunction("wf2_fixed_word_p0_lie_node_vjp_tensor");
    out.fixed_word_p0_lie_material_node_vjp =
        lib->getKernelFunction("wf2_fixed_word_p0_lie_material_node_vjp_tensor");
    out.kinetic_precompiled_length_p0_lie_node_vjp =
        lib->getKernelFunction("wf2_kinetic_precompiled_length_p0_lie_node_vjp_tensor");
    out.kinetic_precompiled_length_p0_lie_material_node_vjp =
        lib->getKernelFunction(kKineticMaterialVjpMetalFunction);
    out.kinetic_fused_direct_full_vjp_validate_v1 =
        lib->getKernelFunction("wf2_kinetic_fused_direct_full_vjp_validate_v1_tensor");
    out.kinetic_fused_direct_full_vjp_v1 =
        lib->getKernelFunction("wf2_kinetic_fused_direct_full_vjp_v1_tensor");
    out.kinetic_fused_direct_full_vjp_finalize_v1 =
        lib->getKernelFunction("wf2_kinetic_fused_direct_full_vjp_finalize_v1_tensor");
    out.kinetic_fused_union_full_vjp_validate_v2 =
        lib->getKernelFunction("wf2_kinetic_fused_union_full_vjp_validate_v2_tensor");
    out.kinetic_fused_union_full_vjp_v2 =
        lib->getKernelFunction("wf2_kinetic_fused_union_full_vjp_v2_tensor");
    out.kinetic_fused_union_full_vjp_finalize_v2 =
        lib->getKernelFunction("wf2_kinetic_fused_union_full_vjp_finalize_v2_tensor");
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
    out.endpoint_record_delta_replace_factorized_packed_framegroup16_constant_state_mse_vjp_direct_atomic_rgb_boundary =
        lib->getKernelFunction(
            "wf2_endpoint_record_delta_replace_factorized_packed_framegroup16_constant_state_mse_vjp_direct_atomic_rgb_boundary_tensor");
    out.endpoint_record_delta_replace_factorized_packed_framegroup16_constant_state_p0_mse_vjp_sparse_mobius_rgb =
        lib->getKernelFunction(
            "wf2_endpoint_record_delta_replace_factorized_packed_framegroup16_constant_state_p0_mse_vjp_sparse_mobius_rgb_tensor");
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

void check_track_boundary_incidence_csr_cpu(
    const torch::Tensor& track_incidence_offsets_i32,
    const torch::Tensor& incidence_boundary_i32,
    const int64_t track_count,
    const int64_t boundary_count,
    const int64_t incidence_count) {
  auto offsets_cpu = track_incidence_offsets_i32.cpu();
  auto boundary_ids_cpu = incidence_boundary_i32.cpu();
  const int32_t* offsets = offsets_cpu.data_ptr<int32_t>();
  const int32_t* boundary_ids = boundary_ids_cpu.data_ptr<int32_t>();
  TORCH_CHECK(offsets[0] == 0, "track_incidence_offsets_i32[0] must be 0");
  for (int64_t track_id = 0; track_id < track_count; ++track_id) {
    const int32_t begin = offsets[track_id];
    const int32_t end = offsets[track_id + 1];
    TORCH_CHECK(begin >= 0 && end >= begin, "track incidence CSR offsets must be monotonic nonnegative");
    TORCH_CHECK(end <= incidence_count, "track incidence CSR offset exceeds incidence count");
    TORCH_CHECK(end - begin <= 4093, "packed row-local incidence codes support at most 4093 incidences per track");
    int32_t previous_boundary = -1;
    for (int32_t incidence_id = begin; incidence_id < end; ++incidence_id) {
      const int32_t boundary_id = boundary_ids[incidence_id];
      TORCH_CHECK(
          boundary_id >= 0 && boundary_id < boundary_count,
          "incidence_boundary_i32 values must be in [0, boundary_count)");
      TORCH_CHECK(
          boundary_id > previous_boundary,
          "each incidence CSR row must contain strictly increasing unique boundary ids");
      previous_boundary = boundary_id;
    }
  }
  TORCH_CHECK(offsets[track_count] == incidence_count, "track incidence CSR final offset mismatch");
}

void check_fixed_word_incidence_csr_cpu(
    const torch::Tensor& word_offsets_i32,
    const torch::Tensor& word_owner_i32,
    const torch::Tensor& word_left_incidence_i32,
    const torch::Tensor& word_right_incidence_i32,
    const torch::Tensor& track_incidence_offsets_i32,
    const int64_t track_count,
    const int64_t site_count,
    const int64_t word_count) {
  auto word_offsets_cpu = word_offsets_i32.cpu();
  auto word_owner_cpu = word_owner_i32.cpu();
  auto word_left_cpu = word_left_incidence_i32.cpu();
  auto word_right_cpu = word_right_incidence_i32.cpu();
  auto incidence_offsets_cpu = track_incidence_offsets_i32.cpu();
  const int32_t* word_offsets = word_offsets_cpu.data_ptr<int32_t>();
  const int32_t* word_owner = word_owner_cpu.data_ptr<int32_t>();
  const int32_t* word_left = word_left_cpu.data_ptr<int32_t>();
  const int32_t* word_right = word_right_cpu.data_ptr<int32_t>();
  const int32_t* incidence_offsets = incidence_offsets_cpu.data_ptr<int32_t>();
  TORCH_CHECK(word_offsets[0] == 0, "word_offsets_i32[0] must be 0");
  for (int64_t track_id = 0; track_id < track_count; ++track_id) {
    const int32_t begin = word_offsets[track_id];
    const int32_t end = word_offsets[track_id + 1];
    TORCH_CHECK(
        begin >= 0 && end > begin && end <= word_count,
        "each fixed-word CSR track row must be nonempty, monotonic, and in bounds");
    TORCH_CHECK(word_left[begin] == -1, "each fixed-word CSR row must start at the near cut (-1)");
    TORCH_CHECK(word_right[end - 1] == -2, "each fixed-word CSR row must end at the far cut (-2)");
    const int32_t incidence_row_size = incidence_offsets[track_id + 1] - incidence_offsets[track_id];
    int32_t previous_right = -1;
    for (int32_t cursor = begin; cursor < end; ++cursor) {
      const int32_t owner = word_owner[cursor];
      const int32_t left = word_left[cursor];
      const int32_t right = word_right[cursor];
      TORCH_CHECK(owner >= 0 && owner < site_count, "fixed-word owner id is outside site range");
      TORCH_CHECK(
          cursor == begin ? left == -1 : left == previous_right,
          "fixed-word cuts must form one adjacent stable ordered word");
      TORCH_CHECK(
          left == -1 || (left >= 0 && left < incidence_row_size),
          "fixed-word left cut must be near or a row-local incidence id");
      TORCH_CHECK(
          right == -2 || (right >= 0 && right < incidence_row_size),
          "fixed-word right cut must be far or a row-local incidence id");
      TORCH_CHECK(left != right, "fixed-word segments must have distinct endpoint cuts");
      TORCH_CHECK(
          cursor + 1 == end ? right == -2 : right >= 0,
          "only the final fixed-word segment may use the far cut");
      previous_right = right;
    }
  }
  TORCH_CHECK(word_offsets[track_count] == word_count, "fixed-word CSR final offset mismatch");
}

void check_packed_endpoint_incidence_delta_records_cpu(
    const torch::Tensor& base_offsets_i16,
    const torch::Tensor& base_record_incidence_i32,
    const torch::Tensor& track_change_offsets_i16,
    const torch::Tensor& change_offsets_i16,
    const torch::Tensor& change_record_incidence_i32,
    const torch::Tensor& track_incidence_offsets_i32,
    const int64_t track_count,
    const int64_t site_count) {
  auto base_offsets_cpu = base_offsets_i16.cpu();
  auto base_records_cpu = base_record_incidence_i32.cpu();
  auto track_change_offsets_cpu = track_change_offsets_i16.cpu();
  auto change_offsets_cpu = change_offsets_i16.cpu();
  auto change_records_cpu = change_record_incidence_i32.cpu();
  auto incidence_offsets_cpu = track_incidence_offsets_i32.cpu();
  const int16_t* base_offsets = base_offsets_cpu.data_ptr<int16_t>();
  const int32_t* base_records = base_records_cpu.data_ptr<int32_t>();
  const int16_t* track_change_offsets = track_change_offsets_cpu.data_ptr<int16_t>();
  const int16_t* change_offsets = change_offsets_cpu.data_ptr<int16_t>();
  const int32_t* change_records = change_records_cpu.data_ptr<int32_t>();
  const int32_t* incidence_offsets = incidence_offsets_cpu.data_ptr<int32_t>();

  auto check_row = [&](const int32_t* records, const int32_t begin, const int32_t end, const int32_t row_size) {
    for (int32_t record_id = begin; record_id < end; ++record_id) {
      const uint32_t packed = static_cast<uint32_t>(records[record_id]);
      TORCH_CHECK((packed & 255u) < static_cast<uint32_t>(site_count), "packed incidence owner is out of range");
      const uint32_t left_code = (packed >> 8u) & 4095u;
      const uint32_t right_code = (packed >> 20u) & 4095u;
      TORCH_CHECK(
          left_code < 2u || left_code - 2u < static_cast<uint32_t>(row_size),
          "packed left row-local incidence id is outside its track CSR row");
      TORCH_CHECK(
          right_code < 2u || right_code - 2u < static_cast<uint32_t>(row_size),
          "packed right row-local incidence id is outside its track CSR row");
    }
  };

  for (int64_t track_id = 0; track_id < track_count; ++track_id) {
    const int32_t row_size = incidence_offsets[track_id + 1] - incidence_offsets[track_id];
    check_row(
        base_records,
        static_cast<int32_t>(base_offsets[track_id]),
        static_cast<int32_t>(base_offsets[track_id + 1]),
        row_size);
    const int32_t change_begin = static_cast<int32_t>(track_change_offsets[track_id]);
    const int32_t change_end = static_cast<int32_t>(track_change_offsets[track_id + 1]);
    for (int32_t change_id = change_begin; change_id < change_end; ++change_id) {
      check_row(
          change_records,
          static_cast<int32_t>(change_offsets[change_id]),
          static_cast<int32_t>(change_offsets[change_id + 1]),
          row_size);
    }
  }
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

std::tuple<
    std::vector<std::string>,
    std::vector<std::string>,
    std::vector<int64_t>,
    std::vector<int64_t>,
    std::vector<int64_t>>
metal_kinetic_memory_light_selected_kernel_resource_attestation() {
  // These are exactly the three custom Metal kernels reached by the
  // material-only executor.  The optional full-geometry VJP is intentionally
  // absent.  Pair the public op and Metal symbol with the same compiled handle
  // used by the launch path so the report cannot silently describe a sibling
  // kernel.
  auto& k = kernels();
  const std::vector<std::tuple<
      const char*,
      const char*,
      std::shared_ptr<MetalKernelFunction>>>
      selected = {
          {
              kKineticNodeForwardOperator,
              kKineticNodeForwardMetalFunction,
              k.kinetic_precompiled_length_p0_lie_node_forward,
          },
          {
              kKineticSampleAccumulateOperator,
              kKineticSampleAccumulateMetalFunction,
              k.kinetic_ragged_p0_lie_sample_mse_vjp_accumulate_only,
          },
          {
              kKineticMaterialVjpOperator,
              kKineticMaterialVjpMetalFunction,
              k.kinetic_precompiled_length_p0_lie_material_node_vjp,
          },
      };

  std::vector<std::string> operator_names;
  std::vector<std::string> metal_function_names;
  std::vector<int64_t> max_threads_per_threadgroup;
  std::vector<int64_t> thread_execution_width;
  std::vector<int64_t> static_threadgroup_memory_length_bytes;
  operator_names.reserve(selected.size());
  metal_function_names.reserve(selected.size());
  max_threads_per_threadgroup.reserve(selected.size());
  thread_execution_width.reserve(selected.size());
  static_threadgroup_memory_length_bytes.reserve(selected.size());

  const auto checked_i64 = [](const uint64_t value, const char* property) {
    TORCH_CHECK(
        value <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
        property,
        " exceeds the signed int64 ABI range");
    return static_cast<int64_t>(value);
  };
  for (const auto& [operator_name, metal_function_name, function] : selected) {
    TORCH_CHECK(function != nullptr, "selected Metal kernel handle is null: ", metal_function_name);
    operator_names.emplace_back(operator_name);
    metal_function_names.emplace_back(metal_function_name);
    max_threads_per_threadgroup.emplace_back(checked_i64(
        function->getMaxThreadsPerThreadgroup(),
        "getMaxThreadsPerThreadgroup()"));
    thread_execution_width.emplace_back(checked_i64(
        function->getThreadExecutionWidth(),
        "getThreadExecutionWidth()"));
    static_threadgroup_memory_length_bytes.emplace_back(checked_i64(
        function->getStaticThreadGroupMemoryLength(),
        "getStaticThreadGroupMemoryLength()"));
  }

  // MetalKernelFunction does not expose register allocation, per-thread
  // private storage, or compiler spill bytes.  Those values are deliberately
  // not estimated or returned by this ABI.
  return std::make_tuple(
      std::move(operator_names),
      std::move(metal_function_names),
      std::move(max_threads_per_threadgroup),
      std::move(thread_execution_width),
      std::move(static_threadgroup_memory_length_bytes));
}

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

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
metal_endpoint_record_delta_replace_factorized_packed_framegroup16_constant_state_mse_vjp_direct_atomic_rgb_boundary(
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
  TORCH_CHECK(boundary_count <= 4093, "constant-state packed factorized fused MSE supports boundary count <= 4093");
  TORCH_CHECK(track_count > 0, "config_i32[1] track count must be positive");
  TORCH_CHECK(frame_count > 0, "config_i32[2] frame count must be positive");
  TORCH_CHECK(site_count == site_rgba_f32.size(0), "config_i32[3] must match site_rgba_f32 rows");
  TORCH_CHECK(
      site_count > 0 && site_count <= 256,
      "constant-state packed factorized fused MSE VJP supports site count in [1, 256]");
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
  auto grad_boundary = torch::empty({boundary_count, 5}, boundary_f32.options().dtype(torch::kFloat32));
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
  launch(k.clear_endpoint_loss_site_rgba_boundary_grad, [&](MetalKernelFunction& fn) {
    fn.setArg(0, loss);
    fn.setArg(1, grad_site_rgba);
    fn.setArg(2, grad_boundary);
    fn.setArg(3, config_i32);
    fn.dispatch((uint64_t)std::max<int64_t>(std::max<int64_t>(site_count, boundary_count), 1), clear_threads);
  });
  auto framegroup_kernel =
      k.endpoint_record_delta_replace_factorized_packed_framegroup16_constant_state_mse_vjp_direct_atomic_rgb_boundary;
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
        fn.setArg(16, grad_boundary);
        fn.dispatch((uint64_t)track_count * (uint64_t)chunk_count * framegroup_threads, framegroup_threads);
      });
  return std::make_tuple(loss, grad_site_rgba, grad_boundary);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
metal_endpoint_record_delta_replace_factorized_packed_framegroup16_constant_state_p0_mse_vjp_sparse_mobius_rgb_boundary(
    const torch::Tensor& boundary_f32,
    const torch::Tensor& track_ray_coeff_f32,
    const torch::Tensor& frame_t_f32,
    const torch::Tensor& base_offsets_i16,
    const torch::Tensor& base_record_incidence_i32,
    const torch::Tensor& track_change_offsets_i16,
    const torch::Tensor& track_chunk_change_offsets_i16,
    const torch::Tensor& change_frame_i16,
    const torch::Tensor& change_offsets_i16,
    const torch::Tensor& change_record_incidence_i32,
    const torch::Tensor& track_incidence_offsets_i32,
    const torch::Tensor& incidence_boundary_i32,
    const torch::Tensor& site_rgba_f32,
    const torch::Tensor& target_rgb_f32,
    const torch::Tensor& config_i32,
    const torch::Tensor& config_f32) {
  check_float_mps_2d(boundary_f32, "boundary_f32", 5);
  check_float_mps_2d(track_ray_coeff_f32, "track_ray_coeff_f32", 12);
  check_float_mps_1d_any(frame_t_f32, "frame_t_f32");
  check_i16_mps_1d_any(base_offsets_i16, "base_offsets_i16");
  check_i32_mps_1d_any(base_record_incidence_i32, "base_record_incidence_i32");
  check_i16_mps_1d_any(track_change_offsets_i16, "track_change_offsets_i16");
  check_i16_mps_1d_any(track_chunk_change_offsets_i16, "track_chunk_change_offsets_i16");
  check_i16_mps_1d_any(change_frame_i16, "change_frame_i16");
  check_i16_mps_1d_any(change_offsets_i16, "change_offsets_i16");
  check_i32_mps_1d_any(change_record_incidence_i32, "change_record_incidence_i32");
  check_i32_mps_1d_any(track_incidence_offsets_i32, "track_incidence_offsets_i32");
  check_i32_mps_1d_any(incidence_boundary_i32, "incidence_boundary_i32");
  check_float_mps_2d(site_rgba_f32, "site_rgba_f32", 4);
  TORCH_CHECK(target_rgb_f32.device().is_mps(), "target_rgb_f32 must be on MPS");
  TORCH_CHECK(target_rgb_f32.scalar_type() == torch::kFloat32, "target_rgb_f32 must be float32");
  TORCH_CHECK(
      target_rgb_f32.dim() == 3 && target_rgb_f32.size(2) == 3,
      "target_rgb_f32 must have shape [K,T,3]");
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
  const int64_t incidence_count = config[7];
  TORCH_CHECK(boundary_count > 0, "config_i32[0] boundary count must be positive");
  TORCH_CHECK(track_count > 0, "config_i32[1] track count must be positive");
  TORCH_CHECK(frame_count > 0, "config_i32[2] frame count must be positive");
  TORCH_CHECK(site_count == site_rgba_f32.size(0), "config_i32[3] must match site_rgba_f32 rows");
  TORCH_CHECK(
      site_count > 0 && site_count <= 256,
      "sparse-Mobius constant-state P0 VJP supports site count in [1, 256]");
  TORCH_CHECK(base_record_count >= 0, "config_i32[4] base record count must be nonnegative");
  TORCH_CHECK(change_count >= 0 && change_count <= 32767, "change count must fit int16 chunk offsets");
  TORCH_CHECK(change_record_count >= 0, "config_i32[6] change record count must be nonnegative");
  TORCH_CHECK(incidence_count >= 0, "config_i32[7] incidence count must be nonnegative");
  TORCH_CHECK(boundary_f32.size(0) == boundary_count, "boundary_f32 row count mismatch");
  TORCH_CHECK(track_ray_coeff_f32.size(0) == track_count, "track_ray_coeff_f32 row count mismatch");
  TORCH_CHECK(frame_t_f32.size(0) == frame_count, "frame_t_f32 length must match frame count");
  TORCH_CHECK(base_record_incidence_i32.size(0) == base_record_count, "base incidence record length mismatch");
  TORCH_CHECK(change_frame_i16.size(0) == change_count, "change_frame_i16 length must match change count");
  TORCH_CHECK(
      change_record_incidence_i32.size(0) == change_record_count,
      "change incidence record length mismatch");
  TORCH_CHECK(incidence_boundary_i32.size(0) == incidence_count, "incidence boundary length mismatch");
  TORCH_CHECK(base_offsets_i16.size(0) == track_count + 1, "base_offsets_i16 length mismatch");
  TORCH_CHECK(track_change_offsets_i16.size(0) == track_count + 1, "track change offset length mismatch");
  TORCH_CHECK(change_offsets_i16.size(0) == change_count + 1, "change_offsets_i16 length mismatch");
  TORCH_CHECK(
      track_incidence_offsets_i32.size(0) == track_count + 1,
      "track_incidence_offsets_i32 length mismatch");
  TORCH_CHECK(
      target_rgb_f32.size(0) == track_count && target_rgb_f32.size(1) == frame_count,
      "target_rgb_f32 shape mismatch");
  check_segment_tape_offsets_i16_cpu(base_offsets_i16, track_count, base_record_count, 129);
  check_segment_tape_offsets_i16_cpu(track_change_offsets_i16, track_count, change_count, 2147483647);
  check_segment_tape_offsets_i16_cpu(change_offsets_i16, change_count, change_record_count, 129);
  check_track_boundary_incidence_csr_cpu(
      track_incidence_offsets_i32,
      incidence_boundary_i32,
      track_count,
      boundary_count,
      incidence_count);
  check_packed_endpoint_incidence_delta_records_cpu(
      base_offsets_i16,
      base_record_incidence_i32,
      track_change_offsets_i16,
      change_offsets_i16,
      change_record_incidence_i32,
      track_incidence_offsets_i32,
      track_count,
      site_count);

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
    TORCH_CHECK(track_begin >= 0 && track_end >= track_begin && track_end <= change_count, "track change bounds");
    int32_t previous = track_begin;
    for (int64_t chunk_id = 0; chunk_id <= chunk_count; ++chunk_id) {
      const int32_t value = static_cast<int32_t>(
          chunk_offsets[track_id * (chunk_count + 1) + chunk_id]);
      TORCH_CHECK(
          value >= previous && value >= track_begin && value <= track_end,
          "track chunk change offsets must be monotonic and bounded by their track row");
      previous = value;
    }
    TORCH_CHECK(previous == track_end, "final chunk change offset must match track change end");
  }

  auto loss = torch::empty({1}, site_rgba_f32.options().dtype(torch::kFloat32));
  auto grad_site_rgba = torch::empty({site_count, 4}, site_rgba_f32.options().dtype(torch::kFloat32));
  auto mobius_coeff = torch::empty({incidence_count, 4}, boundary_f32.options().dtype(torch::kFloat32));
  auto grad_mobius_coeff = torch::empty({incidence_count, 4}, boundary_f32.options().dtype(torch::kFloat32));
  auto grad_boundary = torch::empty({boundary_count, 5}, boundary_f32.options().dtype(torch::kFloat32));
  auto& k = kernels();
  launch(k.clear_endpoint_loss_site_rgba_mobius_boundary_grad, [&](MetalKernelFunction& fn) {
    fn.setArg(0, loss);
    fn.setArg(1, grad_site_rgba);
    fn.setArg(2, grad_mobius_coeff);
    fn.setArg(3, grad_boundary);
    fn.setArg(4, config_i32);
    fn.dispatch(
        (uint64_t)std::max<int64_t>(
            std::max<int64_t>(std::max<int64_t>(site_count, boundary_count), incidence_count),
            1),
        clear_threads);
  });
  launch(k.sparse_mobius_incidence_lower, [&](MetalKernelFunction& fn) {
    fn.setArg(0, boundary_f32);
    fn.setArg(1, track_ray_coeff_f32);
    fn.setArg(2, track_incidence_offsets_i32);
    fn.setArg(3, incidence_boundary_i32);
    fn.setArg(4, mobius_coeff);
    fn.setArg(5, config_i32);
    fn.dispatch((uint64_t)track_count, clear_threads);
  });
  auto framegroup_kernel =
      k.endpoint_record_delta_replace_factorized_packed_framegroup16_constant_state_p0_mse_vjp_sparse_mobius_rgb;
  launch(framegroup_kernel, [&](MetalKernelFunction& fn) {
    fn.setArg(0, mobius_coeff);
    fn.setArg(1, track_ray_coeff_f32);
    fn.setArg(2, frame_t_f32);
    fn.setArg(3, base_offsets_i16);
    fn.setArg(4, base_record_incidence_i32);
    fn.setArg(5, track_change_offsets_i16);
    fn.setArg(6, track_chunk_change_offsets_i16);
    fn.setArg(7, change_frame_i16);
    fn.setArg(8, change_offsets_i16);
    fn.setArg(9, change_record_incidence_i32);
    fn.setArg(10, track_incidence_offsets_i32);
    fn.setArg(11, site_rgba_f32);
    fn.setArg(12, target_rgb_f32);
    fn.setArg(13, config_i32);
    fn.setArg(14, config_f32);
    fn.setArg(15, loss);
    fn.setArg(16, grad_site_rgba);
    fn.setArg(17, grad_mobius_coeff);
    fn.dispatch((uint64_t)track_count * (uint64_t)chunk_count * framegroup_threads, framegroup_threads);
  });
  launch(k.sparse_mobius_incidence_boundary_vjp, [&](MetalKernelFunction& fn) {
    fn.setArg(0, track_ray_coeff_f32);
    fn.setArg(1, track_incidence_offsets_i32);
    fn.setArg(2, incidence_boundary_i32);
    fn.setArg(3, grad_mobius_coeff);
    fn.setArg(4, grad_boundary);
    fn.setArg(5, config_i32);
    fn.dispatch((uint64_t)track_count, clear_threads);
  });
  return std::make_tuple(loss, grad_site_rgba, grad_mobius_coeff, grad_boundary);
}

using FixedWordP0CompiledLieResult = std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor>;

// Warm launch core. All shapes, values, topology, scalar configuration, and
// physical-density constraints are certified before this function is reached.
// Keep this path free of host reads, synchronization, and validation.
FixedWordP0CompiledLieResult launch_fixed_word_p0_compiled_lie_prevalidated(
    const torch::Tensor& boundary_f32,
    const torch::Tensor& track_ray_coeff_f32,
    const torch::Tensor& compiler_node_t_f32,
    const torch::Tensor& word_offsets_i32,
    const torch::Tensor& word_owner_i32,
    const torch::Tensor& word_left_incidence_i32,
    const torch::Tensor& word_right_incidence_i32,
    const torch::Tensor& track_incidence_offsets_i32,
    const torch::Tensor& incidence_boundary_i32,
    const torch::Tensor& site_rgba_f32,
    const torch::Tensor& sample_to_node_f32,
    const torch::Tensor& target_rgb_f32,
    const torch::Tensor& background_rgb_f32,
    const torch::Tensor& config_i32,
    const torch::Tensor& config_f32,
    const int64_t boundary_count,
    const int64_t track_count,
    const int64_t node_count,
    const int64_t sample_count,
    const int64_t site_count,
    const int64_t word_count,
    const int64_t incidence_count) {
  auto loss = torch::empty({1}, site_rgba_f32.options().dtype(torch::kFloat32));
  auto prediction_rgb = torch::empty(
      {track_count, sample_count, 3},
      site_rgba_f32.options().dtype(torch::kFloat32));
  auto node_chart = torch::empty(
      {track_count, node_count, 4},
      site_rgba_f32.options().dtype(torch::kFloat32));
  auto grad_node_chart = torch::empty(
      {track_count, node_count, 4},
      site_rgba_f32.options().dtype(torch::kFloat32));
  auto grad_site_rgba = torch::empty({site_count, 4}, site_rgba_f32.options().dtype(torch::kFloat32));
  auto mobius_coeff = torch::empty({incidence_count, 4}, boundary_f32.options().dtype(torch::kFloat32));
  auto grad_mobius_coeff = torch::empty({incidence_count, 4}, boundary_f32.options().dtype(torch::kFloat32));
  auto grad_boundary = torch::empty({boundary_count, 5}, boundary_f32.options().dtype(torch::kFloat32));
  auto cone_diagnostic = torch::empty({3}, boundary_f32.options().dtype(torch::kInt32));
  auto& k = kernels();
  constexpr uint64_t threads = 256ull;
  const int64_t node_element_count = track_count * node_count * 4;
  const int64_t clear_count = std::max<int64_t>(
      std::max<int64_t>(
          std::max<int64_t>(node_element_count, site_count),
          std::max<int64_t>(incidence_count, boundary_count)),
      3);
  launch(k.clear_fixed_word_p0_compiled_lie_grad, [&](MetalKernelFunction& fn) {
    fn.setArg(0, loss);
    fn.setArg(1, grad_node_chart);
    fn.setArg(2, grad_site_rgba);
    fn.setArg(3, grad_mobius_coeff);
    fn.setArg(4, grad_boundary);
    fn.setArg(5, cone_diagnostic);
    fn.setArg(6, config_i32);
    fn.dispatch((uint64_t)clear_count, threads);
  });
  launch(k.sparse_mobius_incidence_lower, [&](MetalKernelFunction& fn) {
    fn.setArg(0, boundary_f32);
    fn.setArg(1, track_ray_coeff_f32);
    fn.setArg(2, track_incidence_offsets_i32);
    fn.setArg(3, incidence_boundary_i32);
    fn.setArg(4, mobius_coeff);
    fn.setArg(5, config_i32);
    fn.dispatch((uint64_t)track_count, threads);
  });
  launch(k.fixed_word_p0_lie_node_forward, [&](MetalKernelFunction& fn) {
    fn.setArg(0, mobius_coeff);
    fn.setArg(1, track_ray_coeff_f32);
    fn.setArg(2, compiler_node_t_f32);
    fn.setArg(3, word_offsets_i32);
    fn.setArg(4, word_owner_i32);
    fn.setArg(5, word_left_incidence_i32);
    fn.setArg(6, word_right_incidence_i32);
    fn.setArg(7, track_incidence_offsets_i32);
    fn.setArg(8, site_rgba_f32);
    fn.setArg(9, node_chart);
    fn.setArg(10, config_i32);
    fn.setArg(11, config_f32);
    fn.dispatch((uint64_t)track_count * (uint64_t)node_count, threads);
  });
  launch(k.fixed_word_p0_lie_sample_mse_vjp, [&](MetalKernelFunction& fn) {
    fn.setArg(0, node_chart);
    fn.setArg(1, sample_to_node_f32);
    fn.setArg(2, target_rgb_f32);
    fn.setArg(3, background_rgb_f32);
    fn.setArg(4, prediction_rgb);
    fn.setArg(5, loss);
    fn.setArg(6, grad_node_chart);
    fn.setArg(7, cone_diagnostic);
    fn.setArg(8, config_i32);
    fn.setArg(9, config_f32);
    fn.dispatch((uint64_t)track_count * (uint64_t)sample_count, threads);
  });
  launch(k.fixed_word_p0_lie_node_vjp, [&](MetalKernelFunction& fn) {
    fn.setArg(0, mobius_coeff);
    fn.setArg(1, track_ray_coeff_f32);
    fn.setArg(2, compiler_node_t_f32);
    fn.setArg(3, word_offsets_i32);
    fn.setArg(4, word_owner_i32);
    fn.setArg(5, word_left_incidence_i32);
    fn.setArg(6, word_right_incidence_i32);
    fn.setArg(7, track_incidence_offsets_i32);
    fn.setArg(8, site_rgba_f32);
    fn.setArg(9, node_chart);
    fn.setArg(10, grad_node_chart);
    fn.setArg(11, grad_site_rgba);
    fn.setArg(12, grad_mobius_coeff);
    fn.setArg(13, config_i32);
    fn.setArg(14, config_f32);
    fn.dispatch((uint64_t)track_count * (uint64_t)node_count, threads);
  });
  launch(k.sparse_mobius_incidence_boundary_vjp, [&](MetalKernelFunction& fn) {
    fn.setArg(0, track_ray_coeff_f32);
    fn.setArg(1, track_incidence_offsets_i32);
    fn.setArg(2, incidence_boundary_i32);
    fn.setArg(3, grad_mobius_coeff);
    fn.setArg(4, grad_boundary);
    fn.setArg(5, config_i32);
    fn.dispatch((uint64_t)track_count, threads);
  });
  return std::make_tuple(
      loss,
      prediction_rgb,
      node_chart,
      grad_node_chart,
      grad_site_rgba,
      grad_mobius_coeff,
      grad_boundary,
      cone_diagnostic);
}

std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor>
metal_fixed_word_p0_compiled_lie_transfer_mse_vjp_sparse_mobius_boundary(
    const torch::Tensor& boundary_f32,
    const torch::Tensor& track_ray_coeff_f32,
    const torch::Tensor& compiler_node_t_f32,
    const torch::Tensor& word_offsets_i32,
    const torch::Tensor& word_owner_i32,
    const torch::Tensor& word_left_incidence_i32,
    const torch::Tensor& word_right_incidence_i32,
    const torch::Tensor& track_incidence_offsets_i32,
    const torch::Tensor& incidence_boundary_i32,
    const torch::Tensor& site_rgba_f32,
    const torch::Tensor& sample_to_node_f32,
    const torch::Tensor& target_rgb_f32,
    const torch::Tensor& background_rgb_f32,
    const torch::Tensor& config_i32,
    const torch::Tensor& config_f32) {
  check_float_mps_2d(boundary_f32, "boundary_f32", 5);
  check_float_mps_2d(track_ray_coeff_f32, "track_ray_coeff_f32", 12);
  check_float_mps_1d_any(compiler_node_t_f32, "compiler_node_t_f32");
  check_i32_mps_1d_any(word_offsets_i32, "word_offsets_i32");
  check_i32_mps_1d_any(word_owner_i32, "word_owner_i32");
  check_i32_mps_1d_any(word_left_incidence_i32, "word_left_incidence_i32");
  check_i32_mps_1d_any(word_right_incidence_i32, "word_right_incidence_i32");
  check_i32_mps_1d_any(track_incidence_offsets_i32, "track_incidence_offsets_i32");
  check_i32_mps_1d_any(incidence_boundary_i32, "incidence_boundary_i32");
  check_float_mps_2d(site_rgba_f32, "site_rgba_f32", 4);
  TORCH_CHECK(sample_to_node_f32.device().is_mps(), "sample_to_node_f32 must be on MPS");
  TORCH_CHECK(sample_to_node_f32.scalar_type() == torch::kFloat32, "sample_to_node_f32 must be float32");
  TORCH_CHECK(sample_to_node_f32.dim() == 2, "sample_to_node_f32 must have shape [K,J]");
  TORCH_CHECK(sample_to_node_f32.is_contiguous(), "sample_to_node_f32 must be contiguous");
  TORCH_CHECK(target_rgb_f32.device().is_mps(), "target_rgb_f32 must be on MPS");
  TORCH_CHECK(target_rgb_f32.scalar_type() == torch::kFloat32, "target_rgb_f32 must be float32");
  TORCH_CHECK(
      target_rgb_f32.dim() == 3 && target_rgb_f32.size(2) == 3,
      "target_rgb_f32 must have shape [T,K,3]");
  TORCH_CHECK(target_rgb_f32.is_contiguous(), "target_rgb_f32 must be contiguous");
  check_float_mps_1d_any(background_rgb_f32, "background_rgb_f32");
  TORCH_CHECK(background_rgb_f32.size(0) == 3, "background_rgb_f32 must have shape [3]");
  check_i32_mps_1d(config_i32, "config_i32", 8);
  TORCH_CHECK(config_f32.device().is_mps(), "config_f32 must be on MPS");
  TORCH_CHECK(config_f32.scalar_type() == torch::kFloat32, "config_f32 must be float32");
  TORCH_CHECK(config_f32.dim() == 1 && config_f32.size(0) == 6, "config_f32 must have shape [6]");
  TORCH_CHECK(config_f32.is_contiguous(), "config_f32 must be contiguous");

  auto config_i32_cpu = config_i32.cpu();
  const int32_t* config = config_i32_cpu.data_ptr<int32_t>();
  const int64_t boundary_count = config[0];
  const int64_t track_count = config[1];
  const int64_t node_count = config[2];
  const int64_t sample_count = config[3];
  const int64_t site_count = config[4];
  const int64_t word_count = config[5];
  const int64_t incidence_count = config[6];
  TORCH_CHECK(boundary_count >= 0, "config_i32[0] boundary count must be nonnegative");
  TORCH_CHECK(track_count > 0, "config_i32[1] track count must be positive");
  TORCH_CHECK(node_count > 0, "config_i32[2] compiler node count must be positive");
  TORCH_CHECK(sample_count > 0, "config_i32[3] selected sample count must be positive");
  TORCH_CHECK(site_count > 0, "config_i32[4] site count must be positive");
  TORCH_CHECK(word_count >= track_count, "config_i32[5] word count must cover every nonempty track row");
  TORCH_CHECK(incidence_count >= 0, "config_i32[6] incidence count must be nonnegative");
  TORCH_CHECK(config[7] == incidence_count, "config_i32[7] must repeat incidence count for sparse-Mobius lowering");
  TORCH_CHECK(boundary_f32.size(0) == boundary_count, "boundary_f32 row count mismatch");
  TORCH_CHECK(track_ray_coeff_f32.size(0) == track_count, "track_ray_coeff_f32 row count mismatch");
  TORCH_CHECK(compiler_node_t_f32.size(0) == node_count, "compiler_node_t_f32 length mismatch");
  TORCH_CHECK(word_offsets_i32.size(0) == track_count + 1, "word_offsets_i32 length mismatch");
  TORCH_CHECK(word_owner_i32.size(0) == word_count, "word_owner_i32 length mismatch");
  TORCH_CHECK(word_left_incidence_i32.size(0) == word_count, "word_left_incidence_i32 length mismatch");
  TORCH_CHECK(word_right_incidence_i32.size(0) == word_count, "word_right_incidence_i32 length mismatch");
  TORCH_CHECK(
      track_incidence_offsets_i32.size(0) == track_count + 1,
      "track_incidence_offsets_i32 length mismatch");
  TORCH_CHECK(incidence_boundary_i32.size(0) == incidence_count, "incidence_boundary_i32 length mismatch");
  TORCH_CHECK(site_rgba_f32.size(0) == site_count, "site_rgba_f32 row count mismatch");
  TORCH_CHECK(
      sample_to_node_f32.size(0) == sample_count && sample_to_node_f32.size(1) == node_count,
      "sample_to_node_f32 shape mismatch");
  TORCH_CHECK(
      target_rgb_f32.size(0) == track_count && target_rgb_f32.size(1) == sample_count,
      "target_rgb_f32 shape mismatch");
  check_track_boundary_incidence_csr_cpu(
      track_incidence_offsets_i32,
      incidence_boundary_i32,
      track_count,
      boundary_count,
      incidence_count);
  check_fixed_word_incidence_csr_cpu(
      word_offsets_i32,
      word_owner_i32,
      word_left_incidence_i32,
      word_right_incidence_i32,
      track_incidence_offsets_i32,
      track_count,
      site_count,
      word_count);
  auto config_f32_cpu = config_f32.cpu();
  const float* scalar_config = config_f32_cpu.data_ptr<float>();
  TORCH_CHECK(
      std::isfinite(scalar_config[0]) && std::isfinite(scalar_config[1]) &&
          std::isfinite(scalar_config[2]) && std::isfinite(scalar_config[3]) &&
          std::isfinite(scalar_config[4]) && std::isfinite(scalar_config[5]),
      "config_f32 values must be finite");
  TORCH_CHECK(scalar_config[1] > scalar_config[0], "config_f32 far must be greater than near");
  TORCH_CHECK(scalar_config[2] > 0.0f, "config_f32 invalid epsilon must be positive");
  TORCH_CHECK(scalar_config[3] > 0.0f, "config_f32 physical length epsilon must be positive");
  TORCH_CHECK(scalar_config[4] >= 0.0f, "config_f32 cone tolerance must be nonnegative");
  TORCH_CHECK(scalar_config[5] > 0.0f, "config_f32 global loss scale must be positive");
  return launch_fixed_word_p0_compiled_lie_prevalidated(
      boundary_f32,
      track_ray_coeff_f32,
      compiler_node_t_f32,
      word_offsets_i32,
      word_owner_i32,
      word_left_incidence_i32,
      word_right_incidence_i32,
      track_incidence_offsets_i32,
      incidence_boundary_i32,
      site_rgba_f32,
      sample_to_node_f32,
      target_rgb_f32,
      background_rgb_f32,
      config_i32,
      config_f32,
      boundary_count,
      track_count,
      node_count,
      sample_count,
      site_count,
      word_count,
      incidence_count);
}

FixedWordP0CompiledLieResult
metal_fixed_word_p0_compiled_lie_transfer_mse_vjp_sparse_mobius_boundary_launch_only(
    const torch::Tensor& boundary_f32,
    const torch::Tensor& track_ray_coeff_f32,
    const torch::Tensor& compiler_node_t_f32,
    const torch::Tensor& word_offsets_i32,
    const torch::Tensor& word_owner_i32,
    const torch::Tensor& word_left_incidence_i32,
    const torch::Tensor& word_right_incidence_i32,
    const torch::Tensor& track_incidence_offsets_i32,
    const torch::Tensor& incidence_boundary_i32,
    const torch::Tensor& site_rgba_f32,
    const torch::Tensor& sample_to_node_f32,
    const torch::Tensor& target_rgb_f32,
    const torch::Tensor& background_rgb_f32,
    const torch::Tensor& config_i32,
    const torch::Tensor& config_f32,
    const int64_t boundary_count,
    const int64_t track_count,
    const int64_t node_count,
    const int64_t sample_count,
    const int64_t site_count,
    const int64_t word_count,
    const int64_t incidence_count) {
  return launch_fixed_word_p0_compiled_lie_prevalidated(
      boundary_f32,
      track_ray_coeff_f32,
      compiler_node_t_f32,
      word_offsets_i32,
      word_owner_i32,
      word_left_incidence_i32,
      word_right_incidence_i32,
      track_incidence_offsets_i32,
      incidence_boundary_i32,
      site_rgba_f32,
      sample_to_node_f32,
      target_rgb_f32,
      background_rgb_f32,
      config_i32,
      config_f32,
      boundary_count,
      track_count,
      node_count,
      sample_count,
      site_count,
      word_count,
      incidence_count);
}

torch::Tensor metal_sparse_power_boundary_from_sites_launch_only(
    const torch::Tensor& boundary_site_pairs_i32,
    const torch::Tensor& sites_f32,
    const int64_t boundary_count) {
  auto boundary_f32 = torch::empty(
      {boundary_count, 5},
      sites_f32.options().dtype(torch::kFloat32));
  if (boundary_count == 0) {
    return boundary_f32;
  }
  auto& k = kernels();
  constexpr uint64_t threads = 256ull;
  launch(k.sparse_power_boundary_from_sites_launch_only, [&](MetalKernelFunction& fn) {
    fn.setArg(0, boundary_site_pairs_i32);
    fn.setArg(1, sites_f32);
    fn.setArg(2, boundary_f32);
    fn.dispatch((uint64_t)boundary_count, threads);
  });
  return boundary_f32;
}

torch::Tensor metal_fixed_word_p0_sparse_mobius_lower_launch_only(
    const torch::Tensor& boundary_f32,
    const torch::Tensor& track_ray_coeff_f32,
    const torch::Tensor& track_incidence_offsets_i32,
    const torch::Tensor& incidence_boundary_i32,
    const torch::Tensor& config_i32,
    const int64_t track_count,
    const int64_t incidence_count) {
  auto mobius_coeff = torch::empty(
      {incidence_count, 4},
      boundary_f32.options().dtype(torch::kFloat32));
  auto& k = kernels();
  constexpr uint64_t threads = 256ull;
  launch(k.sparse_mobius_incidence_lower, [&](MetalKernelFunction& fn) {
    fn.setArg(0, boundary_f32);
    fn.setArg(1, track_ray_coeff_f32);
    fn.setArg(2, track_incidence_offsets_i32);
    fn.setArg(3, incidence_boundary_i32);
    fn.setArg(4, mobius_coeff);
    fn.setArg(5, config_i32);
    fn.dispatch((uint64_t)track_count, threads);
  });
  return mobius_coeff;
}

torch::Tensor metal_fixed_word_p0_lie_node_forward_launch_only(
    const torch::Tensor& mobius_coeff_f32,
    const torch::Tensor& track_ray_coeff_f32,
    const torch::Tensor& compiler_node_t_f32,
    const torch::Tensor& word_offsets_i32,
    const torch::Tensor& word_owner_i32,
    const torch::Tensor& word_left_incidence_i32,
    const torch::Tensor& word_right_incidence_i32,
    const torch::Tensor& track_incidence_offsets_i32,
    const torch::Tensor& site_rgba_f32,
    const torch::Tensor& config_i32,
    const torch::Tensor& config_f32,
    const int64_t track_count,
    const int64_t node_count) {
  auto node_chart = torch::empty(
      {track_count, node_count, 4},
      site_rgba_f32.options().dtype(torch::kFloat32));
  auto& k = kernels();
  constexpr uint64_t threads = 256ull;
  launch(k.fixed_word_p0_lie_node_forward, [&](MetalKernelFunction& fn) {
    fn.setArg(0, mobius_coeff_f32);
    fn.setArg(1, track_ray_coeff_f32);
    fn.setArg(2, compiler_node_t_f32);
    fn.setArg(3, word_offsets_i32);
    fn.setArg(4, word_owner_i32);
    fn.setArg(5, word_left_incidence_i32);
    fn.setArg(6, word_right_incidence_i32);
    fn.setArg(7, track_incidence_offsets_i32);
    fn.setArg(8, site_rgba_f32);
    fn.setArg(9, node_chart);
    fn.setArg(10, config_i32);
    fn.setArg(11, config_f32);
    fn.dispatch((uint64_t)track_count * (uint64_t)node_count, threads);
  });
  return node_chart;
}

torch::Tensor metal_kinetic_precompiled_length_p0_lie_node_forward_launch_only(
    const torch::Tensor& word_offsets_i32,
    const torch::Tensor& word_owner_i32,
    const torch::Tensor& node_physical_length_f32,
    const torch::Tensor& site_rgba_f32,
    const torch::Tensor& config_i32,
    const torch::Tensor& config_f32,
    const int64_t track_count,
    const int64_t node_count) {
  auto node_chart = torch::empty(
      {track_count, node_count, 4},
      site_rgba_f32.options().dtype(torch::kFloat32));
  auto& k = kernels();
  constexpr uint64_t threads = 256ull;
  launch(k.kinetic_precompiled_length_p0_lie_node_forward, [&](MetalKernelFunction& fn) {
    fn.setArg(0, word_offsets_i32);
    fn.setArg(1, word_owner_i32);
    fn.setArg(2, node_physical_length_f32);
    fn.setArg(3, site_rgba_f32);
    fn.setArg(4, node_chart);
    fn.setArg(5, config_i32);
    fn.setArg(6, config_f32);
    fn.dispatch((uint64_t)track_count * (uint64_t)node_count, threads);
  });
  return node_chart;
}

torch::Tensor metal_kinetic_precompiled_length_p0_lie_node_forward_into_launch_only_v1(
    const torch::Tensor& word_offsets_i32,
    const torch::Tensor& word_owner_i32,
    const torch::Tensor& node_physical_length_f32,
    const torch::Tensor& site_rgba_f32,
    const torch::Tensor& config_i32,
    const torch::Tensor& config_f32,
    const torch::Tensor& node_chart_out_f32,
    const int64_t track_count,
    const int64_t node_count) {
  TORCH_CHECK(
      node_chart_out_f32.device() == site_rgba_f32.device(),
      "node_chart_out_f32 must share the site material device");
  TORCH_CHECK(
      node_chart_out_f32.scalar_type() == torch::kFloat32 &&
          node_chart_out_f32.dim() == 3 &&
          node_chart_out_f32.size(0) == track_count &&
          node_chart_out_f32.size(1) == node_count &&
          node_chart_out_f32.size(2) == 4 &&
          node_chart_out_f32.is_contiguous(),
      "node_chart_out_f32 must be contiguous float32 [track_count,node_count,4]");
  auto& k = kernels();
  constexpr uint64_t threads = 256ull;
  launch(k.kinetic_precompiled_length_p0_lie_node_forward, [&](MetalKernelFunction& fn) {
    fn.setArg(0, word_offsets_i32);
    fn.setArg(1, word_owner_i32);
    fn.setArg(2, node_physical_length_f32);
    fn.setArg(3, site_rgba_f32);
    fn.setArg(4, node_chart_out_f32);
    fn.setArg(5, config_i32);
    fn.setArg(6, config_f32);
    fn.dispatch((uint64_t)track_count * (uint64_t)node_count, threads);
  });
  return node_chart_out_f32;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
metal_fixed_word_p0_lie_sample_state_init_launch_only(
    const torch::Tensor& reference_f32,
    const int64_t track_count,
    const int64_t node_count) {
  auto loss = torch::zeros({1}, reference_f32.options().dtype(torch::kFloat32));
  auto grad_node_chart = torch::zeros(
      {track_count, node_count, 4},
      reference_f32.options().dtype(torch::kFloat32));
  auto cone_diagnostic = torch::zeros({3}, reference_f32.options().dtype(torch::kInt32));
  return std::make_tuple(loss, grad_node_chart, cone_diagnostic);
}

torch::Tensor metal_fixed_word_p0_lie_sample_accumulate_launch_only(
    const torch::Tensor& node_chart_f32,
    const torch::Tensor& sample_to_node_f32,
    const torch::Tensor& target_rgb_f32,
    const torch::Tensor& background_rgb_f32,
    const torch::Tensor& loss_f32,
    const torch::Tensor& grad_node_chart_f32,
    const torch::Tensor& cone_diagnostic_i32,
    const torch::Tensor& config_i32,
    const torch::Tensor& config_f32,
    const int64_t track_count,
    const int64_t sample_count) {
  auto prediction_rgb = torch::empty(
      {track_count, sample_count, 3},
      node_chart_f32.options().dtype(torch::kFloat32));
  auto& k = kernels();
  constexpr uint64_t threads = 256ull;
  launch(k.fixed_word_p0_lie_sample_mse_vjp, [&](MetalKernelFunction& fn) {
    fn.setArg(0, node_chart_f32);
    fn.setArg(1, sample_to_node_f32);
    fn.setArg(2, target_rgb_f32);
    fn.setArg(3, background_rgb_f32);
    fn.setArg(4, prediction_rgb);
    fn.setArg(5, loss_f32);
    fn.setArg(6, grad_node_chart_f32);
    fn.setArg(7, cone_diagnostic_i32);
    fn.setArg(8, config_i32);
    fn.setArg(9, config_f32);
    fn.dispatch((uint64_t)track_count * (uint64_t)sample_count, threads);
  });
  return prediction_rgb;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
metal_fixed_word_p0_lie_sample_accumulate_loss_only_launch_only(
    const torch::Tensor& node_chart_f32,
    const torch::Tensor& sample_to_node_f32,
    const torch::Tensor& target_rgb_f32,
    const torch::Tensor& background_rgb_f32,
    const torch::Tensor& loss_f32,
    const torch::Tensor& grad_node_chart_f32,
    const torch::Tensor& cone_diagnostic_i32,
    const torch::Tensor& config_i32,
    const torch::Tensor& config_f32,
    const int64_t track_count,
    const int64_t sample_count) {
  // Training consumes only the accumulated scalar loss and node cotangent.
  // Do not allocate or write the diagnostic [Bp,K,3] prediction tensor.
  auto& k = kernels();
  constexpr uint64_t threads = 256ull;
  launch(k.fixed_word_p0_lie_sample_mse_vjp_accumulate_only, [&](MetalKernelFunction& fn) {
    fn.setArg(0, node_chart_f32);
    fn.setArg(1, sample_to_node_f32);
    fn.setArg(2, target_rgb_f32);
    fn.setArg(3, background_rgb_f32);
    fn.setArg(4, loss_f32);
    fn.setArg(5, grad_node_chart_f32);
    fn.setArg(6, cone_diagnostic_i32);
    fn.setArg(7, config_i32);
    fn.setArg(8, config_f32);
    fn.dispatch((uint64_t)track_count * (uint64_t)sample_count, threads);
  });
  return std::make_tuple(loss_f32, grad_node_chart_f32, cone_diagnostic_i32);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
metal_fixed_word_p0_lie_world_grad_init_launch_only(
    const torch::Tensor& reference_f32,
    const int64_t site_count,
    const int64_t incidence_count,
    const int64_t boundary_count) {
  auto grad_site_rgba = torch::zeros(
      {site_count, 4},
      reference_f32.options().dtype(torch::kFloat32));
  auto grad_mobius_coeff = torch::zeros(
      {incidence_count, 4},
      reference_f32.options().dtype(torch::kFloat32));
  auto grad_boundary = torch::zeros(
      {boundary_count, 5},
      reference_f32.options().dtype(torch::kFloat32));
  return std::make_tuple(grad_site_rgba, grad_mobius_coeff, grad_boundary);
}

torch::Tensor metal_fixed_word_p0_lie_material_world_grad_init_launch_only(
    const torch::Tensor& reference_f32,
    const int64_t site_count) {
  return torch::zeros(
      {site_count, 4},
      reference_f32.options().dtype(torch::kFloat32));
}

std::tuple<torch::Tensor, torch::Tensor>
metal_fixed_word_p0_lie_node_vjp_accumulate_launch_only(
    const torch::Tensor& mobius_coeff_f32,
    const torch::Tensor& track_ray_coeff_f32,
    const torch::Tensor& compiler_node_t_f32,
    const torch::Tensor& word_offsets_i32,
    const torch::Tensor& word_owner_i32,
    const torch::Tensor& word_left_incidence_i32,
    const torch::Tensor& word_right_incidence_i32,
    const torch::Tensor& track_incidence_offsets_i32,
    const torch::Tensor& site_rgba_f32,
    const torch::Tensor& node_chart_f32,
    const torch::Tensor& grad_node_chart_f32,
    const torch::Tensor& grad_site_rgba_f32,
    const torch::Tensor& grad_mobius_coeff_f32,
    const torch::Tensor& config_i32,
    const torch::Tensor& config_f32,
    const int64_t track_count,
    const int64_t node_count) {
  auto& k = kernels();
  constexpr uint64_t threads = 256ull;
  launch(k.fixed_word_p0_lie_node_vjp, [&](MetalKernelFunction& fn) {
    fn.setArg(0, mobius_coeff_f32);
    fn.setArg(1, track_ray_coeff_f32);
    fn.setArg(2, compiler_node_t_f32);
    fn.setArg(3, word_offsets_i32);
    fn.setArg(4, word_owner_i32);
    fn.setArg(5, word_left_incidence_i32);
    fn.setArg(6, word_right_incidence_i32);
    fn.setArg(7, track_incidence_offsets_i32);
    fn.setArg(8, site_rgba_f32);
    fn.setArg(9, node_chart_f32);
    fn.setArg(10, grad_node_chart_f32);
    fn.setArg(11, grad_site_rgba_f32);
    fn.setArg(12, grad_mobius_coeff_f32);
    fn.setArg(13, config_i32);
    fn.setArg(14, config_f32);
    fn.dispatch((uint64_t)track_count * (uint64_t)node_count, threads);
  });
  return std::make_tuple(grad_site_rgba_f32, grad_mobius_coeff_f32);
}

std::tuple<torch::Tensor, torch::Tensor>
metal_kinetic_precompiled_length_p0_lie_node_vjp_accumulate_launch_only(
    const torch::Tensor& word_offsets_i32,
    const torch::Tensor& word_owner_i32,
    const torch::Tensor& node_physical_length_f32,
    const torch::Tensor& site_rgba_f32,
    const torch::Tensor& node_chart_f32,
    const torch::Tensor& grad_node_chart_f32,
    const torch::Tensor& grad_site_rgba_f32,
    const torch::Tensor& config_i32,
    const torch::Tensor& config_f32,
    const int64_t track_count,
    const int64_t node_count) {
  auto grad_node_physical_length = torch::zeros_like(node_physical_length_f32);
  auto& k = kernels();
  constexpr uint64_t threads = 256ull;
  launch(k.kinetic_precompiled_length_p0_lie_node_vjp, [&](MetalKernelFunction& fn) {
    fn.setArg(0, word_offsets_i32);
    fn.setArg(1, word_owner_i32);
    fn.setArg(2, node_physical_length_f32);
    fn.setArg(3, site_rgba_f32);
    fn.setArg(4, node_chart_f32);
    fn.setArg(5, grad_node_chart_f32);
    fn.setArg(6, grad_site_rgba_f32);
    fn.setArg(7, grad_node_physical_length);
    fn.setArg(8, config_i32);
    fn.setArg(9, config_f32);
    fn.dispatch((uint64_t)track_count * (uint64_t)node_count, threads);
  });
  return std::make_tuple(grad_site_rgba_f32, grad_node_physical_length);
}

torch::Tensor
metal_kinetic_precompiled_length_p0_lie_material_node_vjp_accumulate_launch_only(
    const torch::Tensor& word_offsets_i32,
    const torch::Tensor& word_owner_i32,
    const torch::Tensor& node_physical_length_f32,
    const torch::Tensor& site_rgba_f32,
    const torch::Tensor& node_chart_f32,
    const torch::Tensor& grad_node_chart_f32,
    const torch::Tensor& grad_site_rgba_f32,
    const torch::Tensor& config_i32,
    const torch::Tensor& config_f32,
    const int64_t track_count,
    const int64_t node_count) {
  // Material training freezes compiled geometry. Reuse the material bar as
  // the kernel's disabled buffer(7) argument instead of allocating [J,W].
  auto& k = kernels();
  constexpr uint64_t threads = 256ull;
  launch(k.kinetic_precompiled_length_p0_lie_material_node_vjp, [&](MetalKernelFunction& fn) {
    fn.setArg(0, word_offsets_i32);
    fn.setArg(1, word_owner_i32);
    fn.setArg(2, node_physical_length_f32);
    fn.setArg(3, site_rgba_f32);
    fn.setArg(4, node_chart_f32);
    fn.setArg(5, grad_node_chart_f32);
    fn.setArg(6, grad_site_rgba_f32);
    fn.setArg(7, grad_site_rgba_f32);
    fn.setArg(8, config_i32);
    fn.setArg(9, config_f32);
    fn.dispatch((uint64_t)track_count * (uint64_t)node_count, threads);
  });
  return grad_site_rgba_f32;
}

std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor>
metal_kinetic_fused_union_full_vjp_phase_core_v2(
    const torch::Tensor& word_offsets_i32,
    const torch::Tensor& word_owner_i32,
    const torch::Tensor& source_site_ids_i64,
    const torch::Tensor& compact_to_geometry_output_i64,
    const torch::Tensor& geometry_output_source_site_ids_i64,
    const torch::Tensor& node_physical_length_f32,
    const torch::Tensor& site_rgba_f32,
    const torch::Tensor& node_chart_f32,
    const torch::Tensor& row_node_time_f32,
    const torch::Tensor& row_near_far_f32,
    const torch::Tensor& row_ray_coeff_f32,
    const torch::Tensor& compact_positions0_f32,
    const torch::Tensor& compact_velocities_f32,
    const torch::Tensor& compact_weight_coefficients_f32,
    const torch::Tensor& grad_node_chart_f32,
    const torch::Tensor& grad_site_rgba_f32,
    const torch::Tensor& grad_union_positions0_f32,
    const torch::Tensor& grad_union_velocities_f32,
    const torch::Tensor& grad_union_weight_coefficients_f32,
    const torch::Tensor& config_i32,
    const torch::Tensor& config_f32,
    const torch::Tensor& validation_status_i32,
    const bool run_validation,
    const bool run_accumulation,
    const bool validate_shared_union_ledgers,
    const int64_t global_site_count,
    const int64_t union_site_count,
    const int64_t row_count,
    const int64_t node_count) {
  // This is the raw exact P_b=P_U Q_b boundary. It deliberately performs no
  // persistent commit and retains split phases so an all-block coordinator or
  // a later bounded-block transaction-local coordinator can own admission.
  TORCH_CHECK(row_count > 0, "union-v2 row_count must be positive");
  TORCH_CHECK(node_count > 1, "union-v2 node_count must be at least two");
  TORCH_CHECK(global_site_count > 0, "union-v2 global_site_count must be positive");
  TORCH_CHECK(
      union_site_count > 0 && union_site_count <= global_site_count,
      "union-v2 requires 0 < union_site_count <= global_site_count");
  constexpr int64_t max_i32 = std::numeric_limits<int32_t>::max();
  TORCH_CHECK(
      row_count <= max_i32 && node_count <= max_i32 &&
          global_site_count <= max_i32 && union_site_count <= max_i32,
      "union-v2 counts must fit int32 Metal constants");
  check_i32_mps_1d(word_offsets_i32, "word_offsets_i32", row_count + 1);
  check_i32_mps_1d_any(word_owner_i32, "word_owner_i32");
  TORCH_CHECK(
      word_owner_i32.numel() >= row_count,
      "union-v2 rows must own at least one word");
  const int64_t word_count = word_owner_i32.numel();

  const auto check_i64_index = [&](const torch::Tensor& tensor,
                                   const char* name,
                                   const int64_t expected_count) {
    TORCH_CHECK(tensor.device().is_mps(), name, " must be on MPS");
    TORCH_CHECK(tensor.scalar_type() == torch::kInt64, name, " must be int64");
    TORCH_CHECK(
        tensor.dim() == 1 && tensor.numel() == expected_count &&
            tensor.is_contiguous(),
        name,
        " must be contiguous rank-1 with the exact expected count");
  };
  TORCH_CHECK(
      source_site_ids_i64.device().is_mps() &&
          source_site_ids_i64.scalar_type() == torch::kInt64 &&
          source_site_ids_i64.dim() == 1 && source_site_ids_i64.numel() > 0 &&
          source_site_ids_i64.is_contiguous(),
      "source_site_ids_i64 must be nonempty contiguous MPS int64");
  const int64_t compact_site_count = source_site_ids_i64.numel();
  TORCH_CHECK(
      compact_site_count <= global_site_count,
      "union-v2 compact source table cannot exceed the global world");
  check_i64_index(
      compact_to_geometry_output_i64,
      "compact_to_geometry_output_i64",
      compact_site_count);
  check_i64_index(
      geometry_output_source_site_ids_i64,
      "geometry_output_source_site_ids_i64",
      union_site_count);
  TORCH_CHECK(
      compact_site_count <= max_i32 && word_count <= max_i32,
      "union-v2 compact/word counts must fit int32 Metal constants");

  TORCH_CHECK(
      node_physical_length_f32.device().is_mps() &&
          node_physical_length_f32.scalar_type() == torch::kFloat32 &&
          node_physical_length_f32.dim() == 2 &&
          node_physical_length_f32.size(0) == node_count &&
          node_physical_length_f32.size(1) == word_count &&
          node_physical_length_f32.is_contiguous(),
      "node_physical_length_f32 must be contiguous MPS float32 [node_count,word_count]");
  check_float_mps_2d(site_rgba_f32, "site_rgba_f32", 4);
  TORCH_CHECK(
      site_rgba_f32.size(0) == compact_site_count,
      "site_rgba_f32 must have compact_site_count rows");
  TORCH_CHECK(
      node_chart_f32.device().is_mps() &&
          node_chart_f32.scalar_type() == torch::kFloat32 &&
          node_chart_f32.dim() == 3 && node_chart_f32.size(0) == row_count &&
          node_chart_f32.size(1) == node_count && node_chart_f32.size(2) == 4 &&
          node_chart_f32.is_contiguous(),
      "node_chart_f32 must be contiguous MPS float32 [row_count,node_count,4]");
  TORCH_CHECK(
      grad_node_chart_f32.device().is_mps() &&
          grad_node_chart_f32.scalar_type() == torch::kFloat32 &&
          grad_node_chart_f32.sizes() == node_chart_f32.sizes() &&
          grad_node_chart_f32.is_contiguous(),
      "grad_node_chart_f32 must match node_chart_f32");
  TORCH_CHECK(
      row_node_time_f32.device().is_mps() &&
          row_node_time_f32.scalar_type() == torch::kFloat32 &&
          row_node_time_f32.dim() == 2 &&
          row_node_time_f32.size(0) == row_count &&
          row_node_time_f32.size(1) == node_count &&
          row_node_time_f32.is_contiguous(),
      "row_node_time_f32 must be contiguous MPS float32 [row_count,node_count]");
  check_float_mps_2d(row_near_far_f32, "row_near_far_f32", 2);
  check_float_mps_2d(row_ray_coeff_f32, "row_ray_coeff_f32", 12);
  TORCH_CHECK(
      row_near_far_f32.size(0) == row_count &&
          row_ray_coeff_f32.size(0) == row_count,
      "union-v2 row payloads must have row_count rows");
  check_float_mps_2d(compact_positions0_f32, "compact_positions0_f32", 3);
  check_float_mps_2d(compact_velocities_f32, "compact_velocities_f32", 3);
  TORCH_CHECK(
      compact_positions0_f32.size(0) == compact_site_count &&
          compact_velocities_f32.size(0) == compact_site_count,
      "compact geometry tables must have compact_site_count rows");
  TORCH_CHECK(
      compact_weight_coefficients_f32.device().is_mps() &&
          compact_weight_coefficients_f32.scalar_type() == torch::kFloat32 &&
          compact_weight_coefficients_f32.dim() == 2 &&
          compact_weight_coefficients_f32.size(0) == compact_site_count &&
          compact_weight_coefficients_f32.size(1) >= 1 &&
          compact_weight_coefficients_f32.size(1) <= 3 &&
          compact_weight_coefficients_f32.is_contiguous(),
      "compact_weight_coefficients_f32 must be contiguous MPS float32 [compact_site_count,C<=3]");
  const int64_t weight_coefficient_count =
      compact_weight_coefficients_f32.size(1);

  check_float_mps_2d(grad_site_rgba_f32, "grad_site_rgba_f32", 4);
  check_float_mps_2d(grad_union_positions0_f32, "grad_union_positions0_f32", 3);
  check_float_mps_2d(grad_union_velocities_f32, "grad_union_velocities_f32", 3);
  TORCH_CHECK(
      grad_site_rgba_f32.size(0) == compact_site_count,
      "grad_site_rgba_f32 must have compact_site_count rows");
  TORCH_CHECK(
      grad_union_positions0_f32.size(0) == union_site_count &&
          grad_union_velocities_f32.size(0) == union_site_count,
      "union geometry bars must have union_site_count rows");
  TORCH_CHECK(
      grad_union_weight_coefficients_f32.device().is_mps() &&
          grad_union_weight_coefficients_f32.scalar_type() == torch::kFloat32 &&
          grad_union_weight_coefficients_f32.dim() == 2 &&
          grad_union_weight_coefficients_f32.size(0) == union_site_count &&
          grad_union_weight_coefficients_f32.size(1) ==
              weight_coefficient_count &&
          grad_union_weight_coefficients_f32.is_contiguous(),
      "grad_union_weight_coefficients_f32 must be contiguous MPS float32 [union_site_count,C]");
  check_i32_mps_1d(config_i32, "config_i32", 7);
  TORCH_CHECK(
      config_f32.device().is_mps() &&
          config_f32.scalar_type() == torch::kFloat32 &&
          config_f32.dim() == 1 && config_f32.numel() == 7 &&
          config_f32.is_contiguous(),
      "config_f32 must be contiguous MPS float32 [7]");
  TORCH_CHECK(
      validation_status_i32.device().is_mps() &&
          validation_status_i32.scalar_type() == torch::kInt32 &&
          validation_status_i32.dim() == 1 &&
          validation_status_i32.numel() == 1 &&
          validation_status_i32.is_contiguous(),
      "union-v2 validation status must be contiguous MPS int32 [1]");
  TORCH_CHECK(
      run_validation || run_accumulation,
      "union-v2 phase must validate, accumulate, or do both");
  TORCH_CHECK(
      run_validation || !validate_shared_union_ledgers,
      "shared union ledgers can only be scanned during validation");

  const std::array<const torch::Tensor*, 21> all_tensors = {
      &word_offsets_i32,
      &word_owner_i32,
      &source_site_ids_i64,
      &compact_to_geometry_output_i64,
      &geometry_output_source_site_ids_i64,
      &node_physical_length_f32,
      &site_rgba_f32,
      &node_chart_f32,
      &row_node_time_f32,
      &row_near_far_f32,
      &row_ray_coeff_f32,
      &compact_positions0_f32,
      &compact_velocities_f32,
      &compact_weight_coefficients_f32,
      &grad_node_chart_f32,
      &grad_site_rgba_f32,
      &grad_union_positions0_f32,
      &grad_union_velocities_f32,
      &grad_union_weight_coefficients_f32,
      &config_i32,
      &config_f32};
  for (const torch::Tensor* tensor : all_tensors) {
    TORCH_CHECK(
        tensor->device() == site_rgba_f32.device(),
        "all union-v2 tensors must share one MPS device");
    TORCH_CHECK(
        !validation_status_i32.is_alias_of(*tensor),
        "union-v2 status must not alias launch tensors");
  }
  const std::array<const torch::Tensor*, 4> output_bars = {
      &grad_site_rgba_f32,
      &grad_union_positions0_f32,
      &grad_union_velocities_f32,
      &grad_union_weight_coefficients_f32};
  const std::array<const torch::Tensor*, 17> read_inputs = {
      &word_offsets_i32,
      &word_owner_i32,
      &source_site_ids_i64,
      &compact_to_geometry_output_i64,
      &geometry_output_source_site_ids_i64,
      &node_physical_length_f32,
      &site_rgba_f32,
      &node_chart_f32,
      &row_node_time_f32,
      &row_near_far_f32,
      &row_ray_coeff_f32,
      &compact_positions0_f32,
      &compact_velocities_f32,
      &compact_weight_coefficients_f32,
      &grad_node_chart_f32,
      &config_i32,
      &config_f32};
  for (size_t left = 0; left < output_bars.size(); ++left) {
    for (size_t right = left + 1; right < output_bars.size(); ++right) {
      TORCH_CHECK(
          !output_bars[left]->is_alias_of(*output_bars[right]),
          "union-v2 output bars must be storage-distinct");
    }
    for (const torch::Tensor* input : read_inputs) {
      TORCH_CHECK(
          !output_bars[left]->is_alias_of(*input),
          "union-v2 output bars must not alias read inputs");
    }
  }

  constexpr uint64_t max_u32 = std::numeric_limits<uint32_t>::max();
  const auto require_u32_product = [max_u32](
                                       const uint64_t left,
                                       const uint64_t right,
                                       const char* message) {
    TORCH_CHECK(left == 0 || right <= max_u32 / left, message);
  };
  require_u32_product(row_count, node_count, "union-v2 row-node indexing exceeds uint32");
  require_u32_product(node_count, word_count, "union-v2 node-word indexing exceeds uint32");
  require_u32_product(compact_site_count, 4u, "union-v2 compact material indexing exceeds uint32");
  require_u32_product(compact_site_count, 3u, "union-v2 compact geometry indexing exceeds uint32");
  require_u32_product(
      compact_site_count,
      weight_coefficient_count,
      "union-v2 compact weight indexing exceeds uint32");
  require_u32_product(union_site_count, 3u, "union-v2 geometry indexing exceeds uint32");
  require_u32_product(
      union_site_count,
      weight_coefficient_count,
      "union-v2 weight indexing exceeds uint32");

  const int32_t row_count_i32 = static_cast<int32_t>(row_count);
  const int32_t node_count_i32 = static_cast<int32_t>(node_count);
  const int32_t compact_site_count_i32 =
      static_cast<int32_t>(compact_site_count);
  const int32_t word_count_i32 = static_cast<int32_t>(word_count);
  const int32_t weight_coefficient_count_i32 =
      static_cast<int32_t>(weight_coefficient_count);
  const int32_t global_site_count_i32 =
      static_cast<int32_t>(global_site_count);
  const int32_t union_site_count_i32 =
      static_cast<int32_t>(union_site_count);
  const int32_t validate_shared_union_ledgers_i32 =
      validate_shared_union_ledgers ? 1 : 0;
  auto& k = kernels();
  constexpr uint64_t threads = 256ull;
  if (run_validation) {
    launch(k.kinetic_fused_union_full_vjp_validate_v2, [&](MetalKernelFunction& fn) {
      fn.setArg(0, word_offsets_i32);
      fn.setArg(1, word_owner_i32);
      fn.setArg(2, source_site_ids_i64);
      fn.setArg(3, node_physical_length_f32);
      fn.setArg(4, site_rgba_f32);
      fn.setArg(5, node_chart_f32);
      fn.setArg(6, row_node_time_f32);
      fn.setArg(7, row_near_far_f32);
      fn.setArg(8, row_ray_coeff_f32);
      fn.setArg(9, compact_positions0_f32);
      fn.setArg(10, compact_velocities_f32);
      fn.setArg(11, compact_weight_coefficients_f32);
      fn.setArg(12, grad_node_chart_f32);
      fn.setArg(13, grad_site_rgba_f32);
      fn.setArg(14, grad_union_positions0_f32);
      fn.setArg(15, grad_union_velocities_f32);
      fn.setArg(16, grad_union_weight_coefficients_f32);
      fn.setArg(17, config_i32);
      fn.setArg(18, config_f32);
      fn.setArg(19, row_count_i32);
      fn.setArg(20, node_count_i32);
      fn.setArg(21, compact_site_count_i32);
      fn.setArg(22, word_count_i32);
      fn.setArg(23, weight_coefficient_count_i32);
      fn.setArg(24, global_site_count_i32);
      fn.setArg(25, validation_status_i32);
      fn.setArg(26, validate_shared_union_ledgers_i32);
      fn.setArg(27, compact_to_geometry_output_i64);
      fn.setArg(28, geometry_output_source_site_ids_i64);
      fn.setArg(29, union_site_count_i32);
      fn.dispatch(
          static_cast<uint64_t>(row_count) * static_cast<uint64_t>(node_count),
          threads);
    });
  }
  if (run_accumulation) {
    launch(k.kinetic_fused_union_full_vjp_v2, [&](MetalKernelFunction& fn) {
      fn.setArg(0, word_offsets_i32);
      fn.setArg(1, word_owner_i32);
      fn.setArg(2, source_site_ids_i64);
      fn.setArg(3, node_physical_length_f32);
      fn.setArg(4, site_rgba_f32);
      fn.setArg(5, node_chart_f32);
      fn.setArg(6, row_node_time_f32);
      fn.setArg(7, row_near_far_f32);
      fn.setArg(8, row_ray_coeff_f32);
      fn.setArg(9, compact_positions0_f32);
      fn.setArg(10, compact_velocities_f32);
      fn.setArg(11, compact_weight_coefficients_f32);
      fn.setArg(12, grad_node_chart_f32);
      fn.setArg(13, grad_site_rgba_f32);
      fn.setArg(14, grad_union_positions0_f32);
      fn.setArg(15, grad_union_velocities_f32);
      fn.setArg(16, grad_union_weight_coefficients_f32);
      fn.setArg(17, config_i32);
      fn.setArg(18, config_f32);
      fn.setArg(19, row_count_i32);
      fn.setArg(20, node_count_i32);
      fn.setArg(21, compact_site_count_i32);
      fn.setArg(22, word_count_i32);
      fn.setArg(23, weight_coefficient_count_i32);
      fn.setArg(24, global_site_count_i32);
      fn.setArg(25, validation_status_i32);
      fn.setArg(26, compact_to_geometry_output_i64);
      fn.setArg(27, geometry_output_source_site_ids_i64);
      fn.setArg(28, union_site_count_i32);
      fn.dispatch(
          static_cast<uint64_t>(row_count) * static_cast<uint64_t>(node_count),
          threads);
    });
  }
  return std::make_tuple(
      grad_site_rgba_f32,
      grad_union_positions0_f32,
      grad_union_velocities_f32,
      grad_union_weight_coefficients_f32,
      validation_status_i32);
}

torch::Tensor
metal_kinetic_fused_union_full_vjp_validate_shared_status_launch_only_v2(
    const torch::Tensor& word_offsets_i32,
    const torch::Tensor& word_owner_i32,
    const torch::Tensor& source_site_ids_i64,
    const torch::Tensor& compact_to_geometry_output_i64,
    const torch::Tensor& geometry_output_source_site_ids_i64,
    const torch::Tensor& node_physical_length_f32,
    const torch::Tensor& site_rgba_f32,
    const torch::Tensor& node_chart_f32,
    const torch::Tensor& row_node_time_f32,
    const torch::Tensor& row_near_far_f32,
    const torch::Tensor& row_ray_coeff_f32,
    const torch::Tensor& compact_positions0_f32,
    const torch::Tensor& compact_velocities_f32,
    const torch::Tensor& compact_weight_coefficients_f32,
    const torch::Tensor& grad_node_chart_f32,
    const torch::Tensor& grad_site_rgba_f32,
    const torch::Tensor& grad_union_positions0_f32,
    const torch::Tensor& grad_union_velocities_f32,
    const torch::Tensor& grad_union_weight_coefficients_f32,
    const torch::Tensor& config_i32,
    const torch::Tensor& config_f32,
    const torch::Tensor& validation_status_i32,
    const bool validate_shared_union_ledgers,
    const int64_t global_site_count,
    const int64_t union_site_count,
    const int64_t row_count,
    const int64_t node_count) {
  auto result = metal_kinetic_fused_union_full_vjp_phase_core_v2(
      word_offsets_i32,
      word_owner_i32,
      source_site_ids_i64,
      compact_to_geometry_output_i64,
      geometry_output_source_site_ids_i64,
      node_physical_length_f32,
      site_rgba_f32,
      node_chart_f32,
      row_node_time_f32,
      row_near_far_f32,
      row_ray_coeff_f32,
      compact_positions0_f32,
      compact_velocities_f32,
      compact_weight_coefficients_f32,
      grad_node_chart_f32,
      grad_site_rgba_f32,
      grad_union_positions0_f32,
      grad_union_velocities_f32,
      grad_union_weight_coefficients_f32,
      config_i32,
      config_f32,
      validation_status_i32,
      true,
      false,
      validate_shared_union_ledgers,
      global_site_count,
      union_site_count,
      row_count,
      node_count);
  return std::get<4>(result);
}

std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor>
metal_kinetic_fused_union_full_vjp_accumulate_shared_status_launch_only_v2(
    const torch::Tensor& word_offsets_i32,
    const torch::Tensor& word_owner_i32,
    const torch::Tensor& source_site_ids_i64,
    const torch::Tensor& compact_to_geometry_output_i64,
    const torch::Tensor& geometry_output_source_site_ids_i64,
    const torch::Tensor& node_physical_length_f32,
    const torch::Tensor& site_rgba_f32,
    const torch::Tensor& node_chart_f32,
    const torch::Tensor& row_node_time_f32,
    const torch::Tensor& row_near_far_f32,
    const torch::Tensor& row_ray_coeff_f32,
    const torch::Tensor& compact_positions0_f32,
    const torch::Tensor& compact_velocities_f32,
    const torch::Tensor& compact_weight_coefficients_f32,
    const torch::Tensor& grad_node_chart_f32,
    const torch::Tensor& grad_site_rgba_f32,
    const torch::Tensor& grad_union_positions0_f32,
    const torch::Tensor& grad_union_velocities_f32,
    const torch::Tensor& grad_union_weight_coefficients_f32,
    const torch::Tensor& config_i32,
    const torch::Tensor& config_f32,
    const torch::Tensor& validation_status_i32,
    const int64_t global_site_count,
    const int64_t union_site_count,
    const int64_t row_count,
    const int64_t node_count) {
  return metal_kinetic_fused_union_full_vjp_phase_core_v2(
      word_offsets_i32,
      word_owner_i32,
      source_site_ids_i64,
      compact_to_geometry_output_i64,
      geometry_output_source_site_ids_i64,
      node_physical_length_f32,
      site_rgba_f32,
      node_chart_f32,
      row_node_time_f32,
      row_near_far_f32,
      row_ray_coeff_f32,
      compact_positions0_f32,
      compact_velocities_f32,
      compact_weight_coefficients_f32,
      grad_node_chart_f32,
      grad_site_rgba_f32,
      grad_union_positions0_f32,
      grad_union_velocities_f32,
      grad_union_weight_coefficients_f32,
      config_i32,
      config_f32,
      validation_status_i32,
      false,
      true,
      false,
      global_site_count,
      union_site_count,
      row_count,
      node_count);
}

std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor>
metal_kinetic_fused_union_full_vjp_finalize_shared_status_launch_only_v2(
    const torch::Tensor& grad_site_rgba_f32,
    const torch::Tensor& grad_union_positions0_f32,
    const torch::Tensor& grad_union_velocities_f32,
    const torch::Tensor& grad_union_weight_coefficients_f32,
    const torch::Tensor& validation_status_i32,
    const bool finalize_shared_union_ledgers,
    const int64_t union_site_count) {
  check_float_mps_2d(grad_site_rgba_f32, "grad_site_rgba_f32", 4);
  check_float_mps_2d(grad_union_positions0_f32, "grad_union_positions0_f32", 3);
  check_float_mps_2d(grad_union_velocities_f32, "grad_union_velocities_f32", 3);
  TORCH_CHECK(
      union_site_count > 0 &&
          grad_union_positions0_f32.size(0) == union_site_count &&
          grad_union_velocities_f32.size(0) == union_site_count,
      "union-v2 finalizer geometry bars must have union_site_count rows");
  TORCH_CHECK(
      grad_union_weight_coefficients_f32.device().is_mps() &&
          grad_union_weight_coefficients_f32.scalar_type() == torch::kFloat32 &&
          grad_union_weight_coefficients_f32.dim() == 2 &&
          grad_union_weight_coefficients_f32.size(0) == union_site_count &&
          grad_union_weight_coefficients_f32.size(1) >= 1 &&
          grad_union_weight_coefficients_f32.size(1) <= 3 &&
          grad_union_weight_coefficients_f32.is_contiguous(),
      "union-v2 finalizer weight bar must be MPS float32 [union_site_count,C<=3]");
  TORCH_CHECK(
      validation_status_i32.device().is_mps() &&
          validation_status_i32.scalar_type() == torch::kInt32 &&
          validation_status_i32.dim() == 1 &&
          validation_status_i32.numel() == 1 &&
          validation_status_i32.is_contiguous(),
      "union-v2 finalizer status must be contiguous MPS int32 [1]");
  const std::array<const torch::Tensor*, 4> bars = {
      &grad_site_rgba_f32,
      &grad_union_positions0_f32,
      &grad_union_velocities_f32,
      &grad_union_weight_coefficients_f32};
  for (size_t left = 0; left < bars.size(); ++left) {
    TORCH_CHECK(
        bars[left]->device() == grad_site_rgba_f32.device(),
        "union-v2 finalizer bars must share one device");
    TORCH_CHECK(
        !validation_status_i32.is_alias_of(*bars[left]),
        "union-v2 finalizer status must not alias bars");
    for (size_t right = left + 1; right < bars.size(); ++right) {
      TORCH_CHECK(
          !bars[left]->is_alias_of(*bars[right]),
          "union-v2 finalizer bars must be storage-distinct");
    }
  }
  constexpr uint64_t max_u32 = std::numeric_limits<uint32_t>::max();
  const int64_t compact_site_count = grad_site_rgba_f32.size(0);
  const int64_t weight_coefficient_count =
      grad_union_weight_coefficients_f32.size(1);
  TORCH_CHECK(
      compact_site_count <= std::numeric_limits<int32_t>::max() &&
          union_site_count <= std::numeric_limits<int32_t>::max(),
      "union-v2 finalizer counts must fit int32");
  const uint64_t compact_entries =
      static_cast<uint64_t>(compact_site_count) * 4u;
  const uint64_t union_geometry_entries = finalize_shared_union_ledgers
      ? static_cast<uint64_t>(union_site_count) * 3u
      : 0u;
  const uint64_t union_weight_entries = finalize_shared_union_ledgers
      ? static_cast<uint64_t>(union_site_count) *
            static_cast<uint64_t>(weight_coefficient_count)
      : 0u;
  TORCH_CHECK(
      compact_entries <= max_u32 && union_geometry_entries <= max_u32 &&
          union_weight_entries <= max_u32,
      "union-v2 finalizer indexing exceeds uint32");
  const uint64_t finalizer_entry_count = std::max(
      compact_entries,
      std::max(union_geometry_entries, union_weight_entries));
  const int32_t compact_site_count_i32 =
      static_cast<int32_t>(compact_site_count);
  const int32_t union_site_count_i32 =
      static_cast<int32_t>(union_site_count);
  const int32_t weight_coefficient_count_i32 =
      static_cast<int32_t>(weight_coefficient_count);
  const int32_t finalize_shared_union_ledgers_i32 =
      finalize_shared_union_ledgers ? 1 : 0;
  auto& k = kernels();
  launch(k.kinetic_fused_union_full_vjp_finalize_v2, [&](MetalKernelFunction& fn) {
    fn.setArg(0, grad_site_rgba_f32);
    fn.setArg(1, grad_union_positions0_f32);
    fn.setArg(2, grad_union_velocities_f32);
    fn.setArg(3, grad_union_weight_coefficients_f32);
    fn.setArg(4, validation_status_i32);
    fn.setArg(5, compact_site_count_i32);
    fn.setArg(6, union_site_count_i32);
    fn.setArg(7, weight_coefficient_count_i32);
    fn.setArg(8, finalize_shared_union_ledgers_i32);
    fn.dispatch(finalizer_entry_count, 256ull);
  });
  return std::make_tuple(
      grad_site_rgba_f32,
      grad_union_positions0_f32,
      grad_union_velocities_f32,
      grad_union_weight_coefficients_f32,
      validation_status_i32);
}

std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor>
metal_kinetic_fused_direct_full_vjp_phase_core_v1(
    const torch::Tensor& word_offsets_i32,
    const torch::Tensor& word_owner_i32,
    const torch::Tensor& source_site_ids_i64,
    const torch::Tensor& node_physical_length_f32,
    const torch::Tensor& site_rgba_f32,
    const torch::Tensor& node_chart_f32,
    const torch::Tensor& row_node_time_f32,
    const torch::Tensor& row_near_far_f32,
    const torch::Tensor& row_ray_coeff_f32,
    const torch::Tensor& compact_positions0_f32,
    const torch::Tensor& compact_velocities_f32,
    const torch::Tensor& compact_weight_coefficients_f32,
    const torch::Tensor& grad_node_chart_f32,
    const torch::Tensor& grad_site_rgba_f32,
    const torch::Tensor& grad_global_positions0_f32,
    const torch::Tensor& grad_global_velocities_f32,
    const torch::Tensor& grad_global_weight_coefficients_f32,
    const torch::Tensor& config_i32,
    const torch::Tensor& config_f32,
    const int64_t row_count,
    const int64_t node_count,
    const torch::Tensor& validation_status_i32,
    const bool run_validation,
    const bool run_accumulation,
    const bool validate_shared_global_ledgers) {
  // Source-only until this suffixed ABI is rebuilt and checked against the
  // staged certified sparse oracle.  Every gradient output is caller-owned.
  // The fifth return aliases one scalar int32 validation receipt.  The combined
  // wrapper owns that scalar; split-phase callers own and reuse it across every
  // block in one transaction.  This v1 is fixed-camera-only and has neither a
  // ray-bar output nor a placeholder alias; it also has no [J,W] length-bar
  // allocation or copy.
  TORCH_CHECK(row_count > 0, "fused kinetic row_count must be positive");
  TORCH_CHECK(node_count > 1, "fused kinetic node_count must be at least two");
  constexpr int64_t max_i32 = std::numeric_limits<int32_t>::max();
  TORCH_CHECK(
      row_count < max_i32 && node_count <= max_i32,
      "fused kinetic row/node counts must fit int32 Metal constants");
  check_i32_mps_1d(word_offsets_i32, "word_offsets_i32", row_count + 1);
  check_i32_mps_1d_any(word_owner_i32, "word_owner_i32");
  TORCH_CHECK(word_owner_i32.numel() >= row_count, "fused kinetic rows must own at least one word");
  TORCH_CHECK(
      source_site_ids_i64.device().is_mps(),
      "source_site_ids_i64 must be on MPS");
  TORCH_CHECK(
      source_site_ids_i64.scalar_type() == torch::kInt64,
      "source_site_ids_i64 must be int64");
  TORCH_CHECK(
      source_site_ids_i64.dim() == 1 && source_site_ids_i64.numel() > 0,
      "source_site_ids_i64 must be a nonempty rank-1 tensor");
  TORCH_CHECK(source_site_ids_i64.is_contiguous(), "source_site_ids_i64 must be contiguous");

  const int64_t word_count = word_owner_i32.numel();
  const int64_t compact_site_count = source_site_ids_i64.numel();
  const int64_t global_site_count = grad_global_positions0_f32.size(0);
  TORCH_CHECK(
      node_physical_length_f32.device().is_mps() &&
          node_physical_length_f32.scalar_type() == torch::kFloat32 &&
          node_physical_length_f32.dim() == 2 &&
          node_physical_length_f32.size(0) == node_count &&
          node_physical_length_f32.size(1) == word_count &&
          node_physical_length_f32.is_contiguous(),
      "node_physical_length_f32 must be contiguous MPS float32 [node_count,word_count]");
  check_float_mps_2d(site_rgba_f32, "site_rgba_f32", 4);
  TORCH_CHECK(
      site_rgba_f32.size(0) == compact_site_count,
      "site_rgba_f32 must have compact_site_count rows");
  TORCH_CHECK(
      node_chart_f32.device().is_mps() &&
          node_chart_f32.scalar_type() == torch::kFloat32 &&
          node_chart_f32.dim() == 3 && node_chart_f32.size(0) == row_count &&
          node_chart_f32.size(1) == node_count && node_chart_f32.size(2) == 4 &&
          node_chart_f32.is_contiguous(),
      "node_chart_f32 must be contiguous MPS float32 [row_count,node_count,4]");
  TORCH_CHECK(
      grad_node_chart_f32.device().is_mps() &&
          grad_node_chart_f32.scalar_type() == torch::kFloat32 &&
          grad_node_chart_f32.sizes() == node_chart_f32.sizes() &&
          grad_node_chart_f32.is_contiguous(),
      "grad_node_chart_f32 must match node_chart_f32");
  TORCH_CHECK(
      row_node_time_f32.device().is_mps() &&
          row_node_time_f32.scalar_type() == torch::kFloat32 &&
          row_node_time_f32.dim() == 2 && row_node_time_f32.size(0) == row_count &&
          row_node_time_f32.size(1) == node_count && row_node_time_f32.is_contiguous(),
      "row_node_time_f32 must be contiguous MPS float32 [row_count,node_count]");
  check_float_mps_2d(row_near_far_f32, "row_near_far_f32", 2);
  TORCH_CHECK(
      row_near_far_f32.size(0) == row_count,
      "row_near_far_f32 must have row_count rows");
  check_float_mps_2d(row_ray_coeff_f32, "row_ray_coeff_f32", 12);
  TORCH_CHECK(
      row_ray_coeff_f32.size(0) == row_count,
      "row_ray_coeff_f32 must have row_count rows");
  check_float_mps_2d(compact_positions0_f32, "compact_positions0_f32", 3);
  check_float_mps_2d(compact_velocities_f32, "compact_velocities_f32", 3);
  TORCH_CHECK(
      compact_positions0_f32.size(0) == compact_site_count &&
          compact_velocities_f32.size(0) == compact_site_count,
      "compact position and velocity tables must have compact_site_count rows");
  TORCH_CHECK(
      compact_weight_coefficients_f32.device().is_mps() &&
          compact_weight_coefficients_f32.scalar_type() == torch::kFloat32 &&
          compact_weight_coefficients_f32.dim() == 2 &&
          compact_weight_coefficients_f32.size(0) == compact_site_count &&
          compact_weight_coefficients_f32.size(1) >= 1 &&
          compact_weight_coefficients_f32.size(1) <= 3 &&
          compact_weight_coefficients_f32.is_contiguous(),
      "compact_weight_coefficients_f32 must be contiguous MPS float32 [compact_site_count,C<=3]");
  const int64_t weight_coefficient_count = compact_weight_coefficients_f32.size(1);

  check_float_mps_2d(grad_site_rgba_f32, "grad_site_rgba_f32", 4);
  check_float_mps_2d(grad_global_positions0_f32, "grad_global_positions0_f32", 3);
  check_float_mps_2d(grad_global_velocities_f32, "grad_global_velocities_f32", 3);
  TORCH_CHECK(
      grad_site_rgba_f32.size(0) == compact_site_count,
      "grad_site_rgba_f32 must have compact_site_count rows");
  TORCH_CHECK(global_site_count >= compact_site_count, "global site table cannot be smaller than compact sites");
  TORCH_CHECK(
      grad_global_velocities_f32.size(0) == global_site_count,
      "global position and velocity bars must have equal row counts");
  TORCH_CHECK(
      grad_global_weight_coefficients_f32.device().is_mps() &&
          grad_global_weight_coefficients_f32.scalar_type() == torch::kFloat32 &&
          grad_global_weight_coefficients_f32.dim() == 2 &&
          grad_global_weight_coefficients_f32.size(0) == global_site_count &&
          grad_global_weight_coefficients_f32.size(1) == weight_coefficient_count &&
          grad_global_weight_coefficients_f32.is_contiguous(),
      "grad_global_weight_coefficients_f32 must be contiguous MPS float32 [global_site_count,C]");
  check_i32_mps_1d(config_i32, "config_i32", 6);
  TORCH_CHECK(config_f32.device().is_mps(), "config_f32 must be on MPS");
  TORCH_CHECK(config_f32.scalar_type() == torch::kFloat32, "config_f32 must be float32");
  TORCH_CHECK(config_f32.dim() == 1 && config_f32.numel() == 7, "config_f32 must have shape [7]");
  TORCH_CHECK(config_f32.is_contiguous(), "config_f32 must be contiguous");
  TORCH_CHECK(
      validation_status_i32.device().is_mps() &&
          validation_status_i32.scalar_type() == torch::kInt32 &&
          validation_status_i32.dim() == 1 &&
          validation_status_i32.numel() == 1 &&
          validation_status_i32.is_contiguous(),
      "fused kinetic shared validation status must be contiguous MPS int32 [1]");
  TORCH_CHECK(
      validation_status_i32.device() == site_rgba_f32.device(),
      "fused kinetic shared validation status must use the launch MPS device");
  TORCH_CHECK(
      run_validation || run_accumulation,
      "fused kinetic phase core must validate, accumulate, or perform both phases");
  TORCH_CHECK(
      run_validation || !validate_shared_global_ledgers,
      "shared global ledgers can only be scanned by a validation phase");

  const std::array<const torch::Tensor*, 19> all_tensors = {
      &word_offsets_i32,
      &word_owner_i32,
      &source_site_ids_i64,
      &node_physical_length_f32,
      &site_rgba_f32,
      &node_chart_f32,
      &row_node_time_f32,
      &row_near_far_f32,
      &row_ray_coeff_f32,
      &compact_positions0_f32,
      &compact_velocities_f32,
      &compact_weight_coefficients_f32,
      &grad_node_chart_f32,
      &grad_site_rgba_f32,
      &grad_global_positions0_f32,
      &grad_global_velocities_f32,
      &grad_global_weight_coefficients_f32,
      &config_i32,
      &config_f32};
  for (const torch::Tensor* tensor : all_tensors) {
    TORCH_CHECK(
        tensor->device() == site_rgba_f32.device(),
        "all fused kinetic tensors must share one MPS device");
  }

  const std::array<const torch::Tensor*, 4> output_bars = {
      &grad_site_rgba_f32,
      &grad_global_positions0_f32,
      &grad_global_velocities_f32,
      &grad_global_weight_coefficients_f32};
  const std::array<const torch::Tensor*, 15> read_inputs = {
      &word_offsets_i32,
      &word_owner_i32,
      &source_site_ids_i64,
      &node_physical_length_f32,
      &site_rgba_f32,
      &node_chart_f32,
      &row_node_time_f32,
      &row_near_far_f32,
      &row_ray_coeff_f32,
      &compact_positions0_f32,
      &compact_velocities_f32,
      &compact_weight_coefficients_f32,
      &grad_node_chart_f32,
      &config_i32,
      &config_f32};
  for (size_t left = 0; left < output_bars.size(); ++left) {
    for (size_t right = left + 1; right < output_bars.size(); ++right) {
      TORCH_CHECK(
          !output_bars[left]->is_alias_of(*output_bars[right]),
          "fused kinetic output bars must be storage-distinct");
    }
    for (const torch::Tensor* input : read_inputs) {
      TORCH_CHECK(
          !output_bars[left]->is_alias_of(*input),
          "fused kinetic output bars must not alias primal/cotangent inputs");
    }
  }
  for (const torch::Tensor* tensor : all_tensors) {
    TORCH_CHECK(
        !validation_status_i32.is_alias_of(*tensor),
        "fused kinetic validation status must not alias launch inputs or bars");
  }
  TORCH_CHECK(
      row_count <= max_i32 && node_count <= max_i32 &&
          compact_site_count <= max_i32 && word_count <= max_i32 &&
          weight_coefficient_count <= max_i32 && global_site_count <= max_i32,
      "fused kinetic launch counts must fit int32 Metal constants");
  constexpr uint64_t max_u32 = std::numeric_limits<uint32_t>::max();
  const auto require_u32_product = [max_u32](
                                       const uint64_t left,
                                       const uint64_t right,
                                       const char* message) {
    TORCH_CHECK(left == 0 || right <= max_u32 / left, message);
  };
  const uint64_t row_count_u64 = static_cast<uint64_t>(row_count);
  const uint64_t node_count_u64 = static_cast<uint64_t>(node_count);
  const uint64_t word_count_u64 = static_cast<uint64_t>(word_count);
  const uint64_t compact_site_count_u64 =
      static_cast<uint64_t>(compact_site_count);
  const uint64_t global_site_count_u64 =
      static_cast<uint64_t>(global_site_count);
  const uint64_t weight_coefficient_count_u64 =
      static_cast<uint64_t>(weight_coefficient_count);
  require_u32_product(
      row_count_u64,
      node_count_u64,
      "fused kinetic row-node indexing exceeds uint32");
  const uint64_t row_node_count_u64 = row_count_u64 * node_count_u64;
  require_u32_product(
      row_node_count_u64,
      4u,
      "fused kinetic row-node chart indexing exceeds uint32");
  require_u32_product(
      node_count_u64,
      word_count_u64,
      "fused kinetic node-word indexing exceeds uint32");
  require_u32_product(
      row_count_u64,
      12u,
      "fused kinetic row-ray indexing exceeds uint32");
  require_u32_product(
      row_count_u64,
      2u,
      "fused kinetic row-domain indexing exceeds uint32");
  require_u32_product(
      compact_site_count_u64,
      4u,
      "fused kinetic compact material indexing exceeds uint32");
  require_u32_product(
      compact_site_count_u64,
      3u,
      "fused kinetic compact geometry indexing exceeds uint32");
  require_u32_product(
      compact_site_count_u64,
      weight_coefficient_count_u64,
      "fused kinetic compact weight indexing exceeds uint32");
  require_u32_product(
      global_site_count_u64,
      3u,
      "fused kinetic global geometry indexing exceeds uint32");
  require_u32_product(
      global_site_count_u64,
      weight_coefficient_count_u64,
      "fused kinetic global weight indexing exceeds uint32");
  const int32_t row_count_i32 = static_cast<int32_t>(row_count);
  const int32_t node_count_i32 = static_cast<int32_t>(node_count);
  const int32_t compact_site_count_i32 = static_cast<int32_t>(compact_site_count);
  const int32_t word_count_i32 = static_cast<int32_t>(word_count);
  const int32_t weight_coefficient_count_i32 =
      static_cast<int32_t>(weight_coefficient_count);
  const int32_t global_site_count_i32 = static_cast<int32_t>(global_site_count);
  const int32_t validate_shared_global_ledgers_i32 =
      validate_shared_global_ledgers ? 1 : 0;
  auto& k = kernels();
  constexpr uint64_t threads = 256ull;
  // A single bounded status scalar is the only extra device state. Validation
  // admits only finite, exactly-zero output scratch; validation and guarded
  // accumulation are enqueued in that order on the same MPS stream. The
  // accumulation grid reads the completed shared status before any atomic, so
  // prewrite rejection leaves all four caller-owned bars byte-for-byte
  // untouched without introducing a mid-block host fence. Hidden-alias
  // ownership and post-rejection quarantine remain caller obligations.
  if (run_validation) {
    launch(k.kinetic_fused_direct_full_vjp_validate_v1, [&](MetalKernelFunction& fn) {
      fn.setArg(0, word_offsets_i32);
      fn.setArg(1, word_owner_i32);
      fn.setArg(2, source_site_ids_i64);
      fn.setArg(3, node_physical_length_f32);
      fn.setArg(4, site_rgba_f32);
      fn.setArg(5, node_chart_f32);
      fn.setArg(6, row_node_time_f32);
      fn.setArg(7, row_near_far_f32);
      fn.setArg(8, row_ray_coeff_f32);
      fn.setArg(9, compact_positions0_f32);
      fn.setArg(10, compact_velocities_f32);
      fn.setArg(11, compact_weight_coefficients_f32);
      fn.setArg(12, grad_node_chart_f32);
      fn.setArg(13, grad_site_rgba_f32);
      fn.setArg(14, grad_global_positions0_f32);
      fn.setArg(15, grad_global_velocities_f32);
      fn.setArg(16, grad_global_weight_coefficients_f32);
      fn.setArg(17, config_i32);
      fn.setArg(18, config_f32);
      fn.setArg(19, row_count_i32);
      fn.setArg(20, node_count_i32);
      fn.setArg(21, compact_site_count_i32);
      fn.setArg(22, word_count_i32);
      fn.setArg(23, weight_coefficient_count_i32);
      fn.setArg(24, global_site_count_i32);
      fn.setArg(25, validation_status_i32);
      fn.setArg(26, validate_shared_global_ledgers_i32);
      fn.dispatch((uint64_t)row_count * (uint64_t)node_count, threads);
    });
  }
  if (run_accumulation) {
    launch(k.kinetic_fused_direct_full_vjp_v1, [&](MetalKernelFunction& fn) {
      fn.setArg(0, word_offsets_i32);
      fn.setArg(1, word_owner_i32);
      fn.setArg(2, source_site_ids_i64);
      fn.setArg(3, node_physical_length_f32);
      fn.setArg(4, site_rgba_f32);
      fn.setArg(5, node_chart_f32);
      fn.setArg(6, row_node_time_f32);
      fn.setArg(7, row_near_far_f32);
      fn.setArg(8, row_ray_coeff_f32);
      fn.setArg(9, compact_positions0_f32);
      fn.setArg(10, compact_velocities_f32);
      fn.setArg(11, compact_weight_coefficients_f32);
      fn.setArg(12, grad_node_chart_f32);
      fn.setArg(13, grad_site_rgba_f32);
      fn.setArg(14, grad_global_positions0_f32);
      fn.setArg(15, grad_global_velocities_f32);
      fn.setArg(16, grad_global_weight_coefficients_f32);
      fn.setArg(17, config_i32);
      fn.setArg(18, config_f32);
      fn.setArg(19, row_count_i32);
      fn.setArg(20, node_count_i32);
      fn.setArg(21, compact_site_count_i32);
      fn.setArg(22, word_count_i32);
      fn.setArg(23, weight_coefficient_count_i32);
      fn.setArg(24, global_site_count_i32);
      fn.setArg(25, validation_status_i32);
      fn.dispatch((uint64_t)row_count * (uint64_t)node_count, threads);
    });
  }
  return std::make_tuple(
      grad_site_rgba_f32,
      grad_global_positions0_f32,
      grad_global_velocities_f32,
      grad_global_weight_coefficients_f32,
      validation_status_i32);
}

std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor>
metal_kinetic_fused_direct_full_vjp_finalize_shared_status_launch_only_v1(
    const torch::Tensor& grad_site_rgba_f32,
    const torch::Tensor& grad_global_positions0_f32,
    const torch::Tensor& grad_global_velocities_f32,
    const torch::Tensor& grad_global_weight_coefficients_f32,
    const torch::Tensor& validation_status_i32,
    const bool finalize_shared_global_ledgers) {
  // This pass detects nonfinite destination sums after guarded atomics.  The
  // bars are transaction-local scratch: a nonzero receipt quarantines them
  // instead of promising rollback.  State remains one caller-owned int32.
  check_float_mps_2d(grad_site_rgba_f32, "grad_site_rgba_f32", 4);
  check_float_mps_2d(
      grad_global_positions0_f32, "grad_global_positions0_f32", 3);
  check_float_mps_2d(
      grad_global_velocities_f32, "grad_global_velocities_f32", 3);
  TORCH_CHECK(
      grad_global_weight_coefficients_f32.device().is_mps() &&
          grad_global_weight_coefficients_f32.scalar_type() == torch::kFloat32 &&
          grad_global_weight_coefficients_f32.dim() == 2 &&
          grad_global_weight_coefficients_f32.size(1) >= 1 &&
          grad_global_weight_coefficients_f32.size(1) <= 3 &&
          grad_global_weight_coefficients_f32.is_contiguous(),
      "grad_global_weight_coefficients_f32 must be contiguous MPS float32 [global_site_count,C<=3]");
  const int64_t compact_site_count = grad_site_rgba_f32.size(0);
  const int64_t global_site_count = grad_global_positions0_f32.size(0);
  const int64_t weight_coefficient_count =
      grad_global_weight_coefficients_f32.size(1);
  TORCH_CHECK(
      compact_site_count > 0 && global_site_count >= compact_site_count,
      "fused kinetic finalizer requires 0 < compact_site_count <= global_site_count");
  TORCH_CHECK(
      grad_global_velocities_f32.size(0) == global_site_count &&
          grad_global_weight_coefficients_f32.size(0) == global_site_count,
      "fused kinetic finalizer global ledgers must share one site count");
  TORCH_CHECK(
      validation_status_i32.device().is_mps() &&
          validation_status_i32.scalar_type() == torch::kInt32 &&
          validation_status_i32.dim() == 1 &&
          validation_status_i32.numel() == 1 &&
          validation_status_i32.is_contiguous(),
      "fused kinetic finalizer status must be contiguous MPS int32 [1]");
  TORCH_CHECK(
      validation_status_i32.device() == grad_site_rgba_f32.device(),
      "fused kinetic finalizer status must share the gradient-ledger MPS device");
  const std::array<const torch::Tensor*, 4> bars = {
      &grad_site_rgba_f32,
      &grad_global_positions0_f32,
      &grad_global_velocities_f32,
      &grad_global_weight_coefficients_f32};
  for (size_t left = 0; left < bars.size(); ++left) {
    TORCH_CHECK(
        bars[left]->device() == grad_site_rgba_f32.device(),
        "fused kinetic finalizer bars must share one MPS device");
    TORCH_CHECK(
        !validation_status_i32.is_alias_of(*bars[left]),
        "fused kinetic finalizer status must not alias a gradient ledger");
    for (size_t right = left + 1; right < bars.size(); ++right) {
      TORCH_CHECK(
          !bars[left]->is_alias_of(*bars[right]),
          "fused kinetic finalizer gradient ledgers must be storage-distinct");
    }
  }
  constexpr int64_t max_i32 = std::numeric_limits<int32_t>::max();
  TORCH_CHECK(
      compact_site_count <= max_i32 && global_site_count <= max_i32 &&
          weight_coefficient_count <= max_i32,
      "fused kinetic finalizer counts must fit int32 Metal constants");
  constexpr uint64_t max_u32 = std::numeric_limits<uint32_t>::max();
  const uint64_t compact_entries =
      static_cast<uint64_t>(compact_site_count) * 4u;
  const uint64_t global_geometry_entries =
      finalize_shared_global_ledgers
      ? static_cast<uint64_t>(global_site_count) * 3u
      : 0u;
  const uint64_t global_weight_entries =
      finalize_shared_global_ledgers
      ? static_cast<uint64_t>(global_site_count) *
          static_cast<uint64_t>(weight_coefficient_count)
      : 0u;
  const uint64_t finalizer_entry_count = std::max(
      compact_entries,
      std::max(global_geometry_entries, global_weight_entries));
  TORCH_CHECK(
      compact_entries <= max_u32 && global_geometry_entries <= max_u32 &&
          global_weight_entries <= max_u32,
      "fused kinetic finalizer indexing exceeds uint32");
  const int32_t compact_site_count_i32 =
      static_cast<int32_t>(compact_site_count);
  const int32_t global_site_count_i32 =
      static_cast<int32_t>(global_site_count);
  const int32_t weight_coefficient_count_i32 =
      static_cast<int32_t>(weight_coefficient_count);
  const int32_t finalize_shared_global_ledgers_i32 =
      finalize_shared_global_ledgers ? 1 : 0;
  auto& k = kernels();
  constexpr uint64_t threads = 256ull;
  launch(k.kinetic_fused_direct_full_vjp_finalize_v1, [&](MetalKernelFunction& fn) {
    fn.setArg(0, grad_site_rgba_f32);
    fn.setArg(1, grad_global_positions0_f32);
    fn.setArg(2, grad_global_velocities_f32);
    fn.setArg(3, grad_global_weight_coefficients_f32);
    fn.setArg(4, validation_status_i32);
    fn.setArg(5, compact_site_count_i32);
    fn.setArg(6, global_site_count_i32);
    fn.setArg(7, weight_coefficient_count_i32);
    fn.setArg(8, finalize_shared_global_ledgers_i32);
    fn.dispatch(finalizer_entry_count, threads);
  });
  return std::make_tuple(
      grad_site_rgba_f32,
      grad_global_positions0_f32,
      grad_global_velocities_f32,
      grad_global_weight_coefficients_f32,
      validation_status_i32);
}

std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor>
metal_kinetic_fused_direct_full_vjp_accumulate_launch_only_v1(
    const torch::Tensor& word_offsets_i32,
    const torch::Tensor& word_owner_i32,
    const torch::Tensor& source_site_ids_i64,
    const torch::Tensor& node_physical_length_f32,
    const torch::Tensor& site_rgba_f32,
    const torch::Tensor& node_chart_f32,
    const torch::Tensor& row_node_time_f32,
    const torch::Tensor& row_near_far_f32,
    const torch::Tensor& row_ray_coeff_f32,
    const torch::Tensor& compact_positions0_f32,
    const torch::Tensor& compact_velocities_f32,
    const torch::Tensor& compact_weight_coefficients_f32,
    const torch::Tensor& grad_node_chart_f32,
    const torch::Tensor& grad_site_rgba_f32,
    const torch::Tensor& grad_global_positions0_f32,
    const torch::Tensor& grad_global_velocities_f32,
    const torch::Tensor& grad_global_weight_coefficients_f32,
    const torch::Tensor& config_i32,
    const torch::Tensor& config_f32,
    const int64_t row_count,
    const int64_t node_count) {
  torch::Tensor validation_status_i32 = torch::zeros({1}, config_i32.options());
  auto guarded_result = metal_kinetic_fused_direct_full_vjp_phase_core_v1(
      word_offsets_i32,
      word_owner_i32,
      source_site_ids_i64,
      node_physical_length_f32,
      site_rgba_f32,
      node_chart_f32,
      row_node_time_f32,
      row_near_far_f32,
      row_ray_coeff_f32,
      compact_positions0_f32,
      compact_velocities_f32,
      compact_weight_coefficients_f32,
      grad_node_chart_f32,
      grad_site_rgba_f32,
      grad_global_positions0_f32,
      grad_global_velocities_f32,
      grad_global_weight_coefficients_f32,
      config_i32,
      config_f32,
      row_count,
      node_count,
      validation_status_i32,
      true,
      true,
      true);
  return metal_kinetic_fused_direct_full_vjp_finalize_shared_status_launch_only_v1(
      std::get<0>(guarded_result),
      std::get<1>(guarded_result),
      std::get<2>(guarded_result),
      std::get<3>(guarded_result),
      std::get<4>(guarded_result),
      true);
}

torch::Tensor
metal_kinetic_fused_direct_full_vjp_validate_shared_status_launch_only_v1(
    const torch::Tensor& word_offsets_i32,
    const torch::Tensor& word_owner_i32,
    const torch::Tensor& source_site_ids_i64,
    const torch::Tensor& node_physical_length_f32,
    const torch::Tensor& site_rgba_f32,
    const torch::Tensor& node_chart_f32,
    const torch::Tensor& row_node_time_f32,
    const torch::Tensor& row_near_far_f32,
    const torch::Tensor& row_ray_coeff_f32,
    const torch::Tensor& compact_positions0_f32,
    const torch::Tensor& compact_velocities_f32,
    const torch::Tensor& compact_weight_coefficients_f32,
    const torch::Tensor& grad_node_chart_f32,
    const torch::Tensor& grad_site_rgba_f32,
    const torch::Tensor& grad_global_positions0_f32,
    const torch::Tensor& grad_global_velocities_f32,
    const torch::Tensor& grad_global_weight_coefficients_f32,
    const torch::Tensor& config_i32,
    const torch::Tensor& config_f32,
    const torch::Tensor& validation_status_i32,
    const bool validate_shared_global_ledgers,
    const int64_t row_count,
    const int64_t node_count) {
  auto result = metal_kinetic_fused_direct_full_vjp_phase_core_v1(
      word_offsets_i32,
      word_owner_i32,
      source_site_ids_i64,
      node_physical_length_f32,
      site_rgba_f32,
      node_chart_f32,
      row_node_time_f32,
      row_near_far_f32,
      row_ray_coeff_f32,
      compact_positions0_f32,
      compact_velocities_f32,
      compact_weight_coefficients_f32,
      grad_node_chart_f32,
      grad_site_rgba_f32,
      grad_global_positions0_f32,
      grad_global_velocities_f32,
      grad_global_weight_coefficients_f32,
      config_i32,
      config_f32,
      row_count,
      node_count,
      validation_status_i32,
      true,
      false,
      validate_shared_global_ledgers);
  return std::get<4>(result);
}

std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor>
metal_kinetic_fused_direct_full_vjp_accumulate_shared_status_launch_only_v1(
    const torch::Tensor& word_offsets_i32,
    const torch::Tensor& word_owner_i32,
    const torch::Tensor& source_site_ids_i64,
    const torch::Tensor& node_physical_length_f32,
    const torch::Tensor& site_rgba_f32,
    const torch::Tensor& node_chart_f32,
    const torch::Tensor& row_node_time_f32,
    const torch::Tensor& row_near_far_f32,
    const torch::Tensor& row_ray_coeff_f32,
    const torch::Tensor& compact_positions0_f32,
    const torch::Tensor& compact_velocities_f32,
    const torch::Tensor& compact_weight_coefficients_f32,
    const torch::Tensor& grad_node_chart_f32,
    const torch::Tensor& grad_site_rgba_f32,
    const torch::Tensor& grad_global_positions0_f32,
    const torch::Tensor& grad_global_velocities_f32,
    const torch::Tensor& grad_global_weight_coefficients_f32,
    const torch::Tensor& config_i32,
    const torch::Tensor& config_f32,
    const torch::Tensor& validation_status_i32,
    const int64_t row_count,
    const int64_t node_count) {
  return metal_kinetic_fused_direct_full_vjp_phase_core_v1(
      word_offsets_i32,
      word_owner_i32,
      source_site_ids_i64,
      node_physical_length_f32,
      site_rgba_f32,
      node_chart_f32,
      row_node_time_f32,
      row_near_far_f32,
      row_ray_coeff_f32,
      compact_positions0_f32,
      compact_velocities_f32,
      compact_weight_coefficients_f32,
      grad_node_chart_f32,
      grad_site_rgba_f32,
      grad_global_positions0_f32,
      grad_global_velocities_f32,
      grad_global_weight_coefficients_f32,
      config_i32,
      config_f32,
      row_count,
      node_count,
      validation_status_i32,
      false,
      true,
      false);
}

torch::Tensor metal_fixed_word_p0_lie_material_node_vjp_accumulate_launch_only(
    const torch::Tensor& mobius_coeff_f32,
    const torch::Tensor& track_ray_coeff_f32,
    const torch::Tensor& compiler_node_t_f32,
    const torch::Tensor& word_offsets_i32,
    const torch::Tensor& word_owner_i32,
    const torch::Tensor& word_left_incidence_i32,
    const torch::Tensor& word_right_incidence_i32,
    const torch::Tensor& track_incidence_offsets_i32,
    const torch::Tensor& site_rgba_f32,
    const torch::Tensor& node_chart_f32,
    const torch::Tensor& grad_node_chart_f32,
    const torch::Tensor& grad_site_rgba_f32,
    const torch::Tensor& config_i32,
    const torch::Tensor& config_f32,
    const int64_t track_count,
    const int64_t node_count) {
  auto& k = kernels();
  constexpr uint64_t threads = 256ull;
  launch(k.fixed_word_p0_lie_material_node_vjp, [&](MetalKernelFunction& fn) {
    fn.setArg(0, mobius_coeff_f32);
    fn.setArg(1, track_ray_coeff_f32);
    fn.setArg(2, compiler_node_t_f32);
    fn.setArg(3, word_offsets_i32);
    fn.setArg(4, word_owner_i32);
    fn.setArg(5, word_left_incidence_i32);
    fn.setArg(6, word_right_incidence_i32);
    fn.setArg(7, track_incidence_offsets_i32);
    fn.setArg(8, site_rgba_f32);
    fn.setArg(9, node_chart_f32);
    fn.setArg(10, grad_node_chart_f32);
    fn.setArg(11, grad_site_rgba_f32);
    fn.setArg(12, config_i32);
    fn.setArg(13, config_f32);
    fn.dispatch((uint64_t)track_count * (uint64_t)node_count, threads);
  });
  return grad_site_rgba_f32;
}

torch::Tensor metal_kinetic_ragged_p0_lie_sample_accumulate_launch_only(
    const torch::Tensor& node_chart_f32,
    const torch::Tensor& sample_row_i32,
    const torch::Tensor& sample_to_node_f32,
    const torch::Tensor& target_rgb_f32,
    const torch::Tensor& background_rgb_f32,
    const torch::Tensor& loss_f32,
    const torch::Tensor& grad_node_chart_f32,
    const torch::Tensor& cone_diagnostic_i32,
    const torch::Tensor& config_i32,
    const torch::Tensor& config_f32,
    const int64_t row_count,
    const int64_t node_count,
    const int64_t sample_count) {
  (void)row_count;
  (void)node_count;
  auto prediction_rgb = torch::empty(
      {sample_count, 3},
      node_chart_f32.options().dtype(torch::kFloat32));
  auto& k = kernels();
  constexpr uint64_t threads = 256ull;
  launch(k.kinetic_ragged_p0_lie_sample_mse_vjp, [&](MetalKernelFunction& fn) {
    fn.setArg(0, node_chart_f32);
    fn.setArg(1, sample_row_i32);
    fn.setArg(2, sample_to_node_f32);
    fn.setArg(3, target_rgb_f32);
    fn.setArg(4, background_rgb_f32);
    fn.setArg(5, prediction_rgb);
    fn.setArg(6, loss_f32);
    fn.setArg(7, grad_node_chart_f32);
    fn.setArg(8, cone_diagnostic_i32);
    fn.setArg(9, config_i32);
    fn.setArg(10, config_f32);
    fn.dispatch((uint64_t)sample_count, threads);
  });
  return prediction_rgb;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
metal_kinetic_ragged_p0_lie_sample_accumulate_loss_only_launch_only(
    const torch::Tensor& node_chart_f32,
    const torch::Tensor& sample_row_i32,
    const torch::Tensor& sample_to_node_f32,
    const torch::Tensor& target_rgb_f32,
    const torch::Tensor& background_rgb_f32,
    const torch::Tensor& loss_f32,
    const torch::Tensor& grad_node_chart_f32,
    const torch::Tensor& cone_diagnostic_i32,
    const torch::Tensor& config_i32,
    const torch::Tensor& config_f32,
    const int64_t row_count,
    const int64_t node_count,
    const int64_t sample_count) {
  (void)row_count;
  (void)node_count;
  // Hot-path reduction mutates only caller-owned scalar/node/diagnostic state.
  auto& k = kernels();
  constexpr uint64_t threads = 256ull;
  launch(k.kinetic_ragged_p0_lie_sample_mse_vjp_accumulate_only, [&](MetalKernelFunction& fn) {
    fn.setArg(0, node_chart_f32);
    fn.setArg(1, sample_row_i32);
    fn.setArg(2, sample_to_node_f32);
    fn.setArg(3, target_rgb_f32);
    fn.setArg(4, background_rgb_f32);
    fn.setArg(5, loss_f32);
    fn.setArg(6, grad_node_chart_f32);
    fn.setArg(7, cone_diagnostic_i32);
    fn.setArg(8, config_i32);
    fn.setArg(9, config_f32);
    fn.dispatch((uint64_t)sample_count, threads);
  });
  return std::make_tuple(loss_f32, grad_node_chart_f32, cone_diagnostic_i32);
}

torch::Tensor metal_fixed_word_p0_sparse_mobius_boundary_finalize_launch_only(
    const torch::Tensor& track_ray_coeff_f32,
    const torch::Tensor& track_incidence_offsets_i32,
    const torch::Tensor& incidence_boundary_i32,
    const torch::Tensor& grad_mobius_coeff_f32,
    const torch::Tensor& grad_boundary_f32,
    const torch::Tensor& config_i32,
    const int64_t track_count) {
  auto& k = kernels();
  constexpr uint64_t threads = 256ull;
  launch(k.sparse_mobius_incidence_boundary_vjp, [&](MetalKernelFunction& fn) {
    fn.setArg(0, track_ray_coeff_f32);
    fn.setArg(1, track_incidence_offsets_i32);
    fn.setArg(2, incidence_boundary_i32);
    fn.setArg(3, grad_mobius_coeff_f32);
    fn.setArg(4, grad_boundary_f32);
    fn.setArg(5, config_i32);
    fn.dispatch((uint64_t)track_count, threads);
  });
  return grad_boundary_f32;
}

torch::Tensor metal_sparse_power_boundary_vjp_to_sites_launch_only(
    const torch::Tensor& active_boundary_site_pairs_i32,
    const torch::Tensor& sites_f32,
    const torch::Tensor& grad_boundary_f32) {
  const int64_t active_count = active_boundary_site_pairs_i32.size(0);

  // Shapes, devices, topology, and site count were checked while preparing the
  // resident token. The kernel still guards every sparse index.
  auto grad_sites = torch::zeros_like(sites_f32);
  if (active_count == 0) {
    return grad_sites;
  }
  auto& k = kernels();
  constexpr uint64_t threads = 256ull;
  launch(k.sparse_power_boundary_site_vjp_launch_only, [&](MetalKernelFunction& fn) {
    fn.setArg(0, active_boundary_site_pairs_i32);
    fn.setArg(1, sites_f32);
    fn.setArg(2, grad_boundary_f32);
    fn.setArg(3, grad_sites);
    fn.dispatch((uint64_t)active_count, threads);
  });
  return grad_sites;
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

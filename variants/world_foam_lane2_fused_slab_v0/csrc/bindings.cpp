#include <torch/extension.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace world_foam_lane2_fused_slab {

namespace {

struct Gate4EndpointRecord {
  int32_t owner;
  int32_t left;
  int32_t right;
};

struct Gate4DepthCandidate {
  double depth;
  int64_t boundary_id;
};

torch::Tensor i32_tensor_from_vector(const std::vector<int32_t>& values) {
  return torch::tensor(values, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU));
}

torch::Tensor i64_tensor_from_vector(const std::vector<int64_t>& values) {
  return torch::tensor(values, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
}

torch::Tensor f64_tensor_from_vector(const std::vector<double>& values) {
  return torch::tensor(values, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
}

bool same_records(const std::vector<Gate4EndpointRecord>& lhs, const std::vector<Gate4EndpointRecord>& rhs) {
  if (lhs.size() != rhs.size()) {
    return false;
  }
  for (size_t index = 0; index < lhs.size(); ++index) {
    if (lhs[index].owner != rhs[index].owner || lhs[index].left != rhs[index].left ||
        lhs[index].right != rhs[index].right) {
      return false;
    }
  }
  return true;
}

int32_t pack_cut_id_i32(const int32_t cut_id) {
  if (cut_id == -1) {
    return 0;
  }
  if (cut_id == -2) {
    return 1;
  }
  TORCH_CHECK(cut_id >= 0, "packed endpoint records only support cut ids -1, -2, or nonnegative boundary ids");
  const int64_t code = static_cast<int64_t>(cut_id) + 2;
  TORCH_CHECK(code <= 4095, "packed endpoint records support cut codes up to 4095");
  return static_cast<int32_t>(code);
}

int32_t pack_endpoint_record_i32(const int32_t owner, const int32_t left, const int32_t right) {
  TORCH_CHECK(owner >= -1 && owner <= 255, "packed endpoint records support owner ids in [-1, 255]");
  const int32_t owner_code = owner < 0 ? 0 : owner;
  const int32_t left_code = pack_cut_id_i32(left);
  const int32_t right_code = pack_cut_id_i32(right);
  const int64_t packed =
      static_cast<int64_t>(owner_code) | (static_cast<int64_t>(left_code) << 8) |
      (static_cast<int64_t>(right_code) << 20);
  TORCH_CHECK(packed <= std::numeric_limits<int32_t>::max(), "packed endpoint record exceeded signed int32 range");
  return static_cast<int32_t>(packed);
}

bool cut_row_is_active(
    const int64_t start_segment,
    const int64_t initial_owner,
    const int64_t segment_count) {
  TORCH_CHECK(start_segment >= -1, "start_segment must be >= -1");
  TORCH_CHECK(initial_owner >= -1, "initial_owner must be >= -1");
  const bool has_start = start_segment >= 0;
  const bool has_owner = initial_owner >= 0;
  TORCH_CHECK(
      has_start == has_owner,
      "start_segment and initial_owner must be both active or both inactive");
  if (!has_start) {
    return false;
  }
  TORCH_CHECK(segment_count > 0, "active cut row requires at least one segment");
  TORCH_CHECK(start_segment < segment_count, "start_segment out of bounds");
  return true;
}

bool sorted_row_is_active(const int64_t active) {
  TORCH_CHECK(active == 0 || active == 1, "row_active_i64 values must be 0 or 1");
  return active == 1;
}

int64_t checked_sorted_boundary_id(const int64_t boundary_id, const int64_t boundary_count) {
  TORCH_CHECK(boundary_id >= 0 && boundary_id < boundary_count, "sorted_ids_i64 boundary id out of bounds");
  return boundary_id;
}

int64_t checked_sorted_boundary_id_nonnegative(const int64_t boundary_id) {
  TORCH_CHECK(boundary_id >= 0, "sorted_ids_i64 values must be nonnegative boundary ids");
  return boundary_id;
}

double checked_sorted_depth(const double depth, const double near, const double far) {
  TORCH_CHECK(std::isfinite(depth), "sorted_depths_f64 values must be finite");
  TORCH_CHECK(depth >= near && depth <= far, "sorted_depths_f64 values must be within [near, far]");
  return depth;
}

void check_sorted_depth_order(const double depth, const double previous_depth) {
  TORCH_CHECK(depth >= previous_depth, "sorted_depths_f64 valid depths must be nondecreasing");
}

void validate_boundary_other_table(
    const int64_t* boundary_other,
    const int64_t site_count,
    const int64_t boundary_count) {
  for (int64_t site = 0; site < site_count; ++site) {
    for (int64_t boundary = 0; boundary < boundary_count; ++boundary) {
      const int64_t other = boundary_other[site * boundary_count + boundary];
      TORCH_CHECK(
          other >= -1 && other < site_count,
          "boundary_other_by_owner_i64 values must be -1 or valid site ids");
    }
  }
}

double checked_cut_depth(const double depth) {
  TORCH_CHECK(std::isfinite(depth), "cut_depths_f64 values must be finite");
  return depth;
}

void validate_cut_row_arrays(
    const double* cut_depths,
    const int64_t* cut_ids,
    const int64_t cut_begin,
    const int64_t cut_end,
    const int64_t boundary_count) {
  const int64_t cut_count = cut_end - cut_begin;
  if (cut_count == 0) {
    return;
  }
  TORCH_CHECK(cut_count >= 2, "cut row with cuts requires at least near/far sentinels");
  TORCH_CHECK(cut_ids[cut_begin] == -1, "cut row first id must be -1 near sentinel");
  TORCH_CHECK(cut_ids[cut_end - 1] == -2, "cut row last id must be -2 far sentinel");
  double previous_depth = checked_cut_depth(cut_depths[cut_begin]);
  for (int64_t index = cut_begin + 1; index < cut_end; ++index) {
    const double depth = checked_cut_depth(cut_depths[index]);
    TORCH_CHECK(depth >= previous_depth, "cut_depths_f64 values must be nondecreasing within each row");
    previous_depth = depth;
  }
  for (int64_t index = cut_begin + 1; index < cut_end - 1; ++index) {
    const int64_t cut_id = cut_ids[index];
    TORCH_CHECK(cut_id >= 0 && cut_id < boundary_count, "cut row internal ids must be valid boundary ids");
  }
}

}  // namespace

torch::Tensor pack_endpoint_records_i32_cpu(
    const torch::Tensor& owner_i32,
    const torch::Tensor& left_i32,
    const torch::Tensor& right_i32) {
  TORCH_CHECK(owner_i32.device().is_cpu(), "owner_i32 must be a CPU tensor");
  TORCH_CHECK(left_i32.device().is_cpu(), "left_i32 must be a CPU tensor");
  TORCH_CHECK(right_i32.device().is_cpu(), "right_i32 must be a CPU tensor");
  TORCH_CHECK(owner_i32.scalar_type() == torch::kInt32, "owner_i32 must be int32");
  TORCH_CHECK(left_i32.scalar_type() == torch::kInt32, "left_i32 must be int32");
  TORCH_CHECK(right_i32.scalar_type() == torch::kInt32, "right_i32 must be int32");
  TORCH_CHECK(owner_i32.dim() == 1, "owner_i32 must be rank-1");
  TORCH_CHECK(left_i32.dim() == 1, "left_i32 must be rank-1");
  TORCH_CHECK(right_i32.dim() == 1, "right_i32 must be rank-1");
  TORCH_CHECK(owner_i32.sizes() == left_i32.sizes(), "owner_i32 and left_i32 shapes must match");
  TORCH_CHECK(owner_i32.sizes() == right_i32.sizes(), "owner_i32 and right_i32 shapes must match");
  TORCH_CHECK(owner_i32.is_contiguous(), "owner_i32 must be contiguous");
  TORCH_CHECK(left_i32.is_contiguous(), "left_i32 must be contiguous");
  TORCH_CHECK(right_i32.is_contiguous(), "right_i32 must be contiguous");

  auto packed_i32 = torch::empty_like(owner_i32);
  const int32_t* owner = owner_i32.data_ptr<int32_t>();
  const int32_t* left = left_i32.data_ptr<int32_t>();
  const int32_t* right = right_i32.data_ptr<int32_t>();
  int32_t* packed = packed_i32.data_ptr<int32_t>();
  const int64_t count = owner_i32.numel();
  for (int64_t index = 0; index < count; ++index) {
    packed[index] = pack_endpoint_record_i32(owner[index], left[index], right[index]);
  }
  return packed_i32;
}

std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor>
gate4_delta_replace_from_cuts_cpu(
    const torch::Tensor& cut_depths_f64,
    const torch::Tensor& cut_ids_i64,
    const torch::Tensor& cut_offsets_i64,
    const torch::Tensor& start_segments_i64,
    const torch::Tensor& initial_owner_i64,
    const torch::Tensor& boundary_other_by_owner_i64,
    const int64_t frame_count,
    const double epsilon) {
  TORCH_CHECK(cut_depths_f64.device().is_cpu(), "cut_depths_f64 must be a CPU tensor");
  TORCH_CHECK(cut_ids_i64.device().is_cpu(), "cut_ids_i64 must be a CPU tensor");
  TORCH_CHECK(cut_offsets_i64.device().is_cpu(), "cut_offsets_i64 must be a CPU tensor");
  TORCH_CHECK(start_segments_i64.device().is_cpu(), "start_segments_i64 must be a CPU tensor");
  TORCH_CHECK(initial_owner_i64.device().is_cpu(), "initial_owner_i64 must be a CPU tensor");
  TORCH_CHECK(boundary_other_by_owner_i64.device().is_cpu(), "boundary_other_by_owner_i64 must be a CPU tensor");
  TORCH_CHECK(cut_depths_f64.scalar_type() == torch::kFloat64, "cut_depths_f64 must be float64");
  TORCH_CHECK(cut_ids_i64.scalar_type() == torch::kInt64, "cut_ids_i64 must be int64");
  TORCH_CHECK(cut_offsets_i64.scalar_type() == torch::kInt64, "cut_offsets_i64 must be int64");
  TORCH_CHECK(start_segments_i64.scalar_type() == torch::kInt64, "start_segments_i64 must be int64");
  TORCH_CHECK(initial_owner_i64.scalar_type() == torch::kInt64, "initial_owner_i64 must be int64");
  TORCH_CHECK(boundary_other_by_owner_i64.scalar_type() == torch::kInt64, "boundary_other_by_owner_i64 must be int64");
  TORCH_CHECK(cut_depths_f64.dim() == 1, "cut_depths_f64 must be rank-1");
  TORCH_CHECK(cut_ids_i64.dim() == 1, "cut_ids_i64 must be rank-1");
  TORCH_CHECK(cut_offsets_i64.dim() == 1, "cut_offsets_i64 must be rank-1");
  TORCH_CHECK(start_segments_i64.dim() == 1, "start_segments_i64 must be rank-1");
  TORCH_CHECK(initial_owner_i64.dim() == 1, "initial_owner_i64 must be rank-1");
  TORCH_CHECK(boundary_other_by_owner_i64.dim() == 2, "boundary_other_by_owner_i64 must be rank-2");
  TORCH_CHECK(cut_depths_f64.is_contiguous(), "cut_depths_f64 must be contiguous");
  TORCH_CHECK(cut_ids_i64.is_contiguous(), "cut_ids_i64 must be contiguous");
  TORCH_CHECK(cut_offsets_i64.is_contiguous(), "cut_offsets_i64 must be contiguous");
  TORCH_CHECK(start_segments_i64.is_contiguous(), "start_segments_i64 must be contiguous");
  TORCH_CHECK(initial_owner_i64.is_contiguous(), "initial_owner_i64 must be contiguous");
  TORCH_CHECK(boundary_other_by_owner_i64.is_contiguous(), "boundary_other_by_owner_i64 must be contiguous");
  TORCH_CHECK(frame_count > 0, "frame_count must be positive");

  const int64_t work_count = start_segments_i64.numel();
  TORCH_CHECK(initial_owner_i64.numel() == work_count, "initial_owner_i64 length must match start_segments_i64");
  TORCH_CHECK(cut_offsets_i64.numel() == work_count + 1, "cut_offsets_i64 length must be work_count + 1");
  TORCH_CHECK(work_count % frame_count == 0, "work_count must be track_count * frame_count");
  TORCH_CHECK(cut_depths_f64.numel() == cut_ids_i64.numel(), "cut depth/id arrays must have matching lengths");

  const int64_t track_count = work_count / frame_count;
  const int64_t site_count = boundary_other_by_owner_i64.size(0);
  const int64_t boundary_count = boundary_other_by_owner_i64.size(1);
  const double* cut_depths = cut_depths_f64.data_ptr<double>();
  const int64_t* cut_ids = cut_ids_i64.data_ptr<int64_t>();
  const int64_t* cut_offsets = cut_offsets_i64.data_ptr<int64_t>();
  const int64_t* start_segments = start_segments_i64.data_ptr<int64_t>();
  const int64_t* initial_owners = initial_owner_i64.data_ptr<int64_t>();
  const int64_t* boundary_other = boundary_other_by_owner_i64.data_ptr<int64_t>();
  validate_boundary_other_table(boundary_other, site_count, boundary_count);

  std::vector<int32_t> base_offsets;
  std::vector<int32_t> base_owner;
  std::vector<int32_t> base_left;
  std::vector<int32_t> base_right;
  std::vector<int32_t> track_change_offsets;
  std::vector<int32_t> change_frame;
  std::vector<int32_t> change_offsets;
  std::vector<int32_t> change_owner;
  std::vector<int32_t> change_left;
  std::vector<int32_t> change_right;
  base_offsets.reserve(static_cast<size_t>(track_count) + 1);
  track_change_offsets.reserve(static_cast<size_t>(track_count) + 1);
  change_offsets.push_back(0);
  base_offsets.push_back(0);
  track_change_offsets.push_back(0);

  std::vector<Gate4EndpointRecord> current;
  std::vector<Gate4EndpointRecord> previous;
  for (int64_t track = 0; track < track_count; ++track) {
    previous.clear();
    bool has_previous = false;
    for (int64_t frame = 0; frame < frame_count; ++frame) {
      const int64_t work = track * frame_count + frame;
      current.clear();
      const int64_t start_segment = start_segments[work];
      int64_t current_owner = initial_owners[work];
      const int64_t cut_begin = cut_offsets[work];
      const int64_t cut_end = cut_offsets[work + 1];
      TORCH_CHECK(cut_begin <= cut_end, "cut_offsets_i64 must be monotonic nondecreasing");
      TORCH_CHECK(cut_begin >= 0 && cut_end <= cut_depths_f64.numel(), "cut offsets out of bounds");
      validate_cut_row_arrays(cut_depths, cut_ids, cut_begin, cut_end, boundary_count);
      const int64_t segment_count = cut_end - cut_begin - 1;
      if (cut_row_is_active(start_segment, current_owner, segment_count)) {
        TORCH_CHECK(current_owner < site_count, "initial owner out of bounds");
        int64_t cursor = start_segment;
        while (cursor < segment_count) {
          int64_t next_cut_index = segment_count;
          int64_t boundary_id = -1;
          for (int64_t local_cut = cursor + 1; local_cut < segment_count; ++local_cut) {
            const int64_t candidate_boundary = cut_ids[cut_begin + local_cut];
            TORCH_CHECK(candidate_boundary >= 0 && candidate_boundary < boundary_count, "boundary id out of bounds");
            if (boundary_other[current_owner * boundary_count + candidate_boundary] >= 0) {
              next_cut_index = local_cut;
              boundary_id = candidate_boundary;
              break;
            }
          }
          if (cut_depths[cut_begin + next_cut_index] - cut_depths[cut_begin + cursor] > epsilon) {
            current.push_back(Gate4EndpointRecord{
                static_cast<int32_t>(current_owner),
                static_cast<int32_t>(cut_ids[cut_begin + cursor]),
                static_cast<int32_t>(cut_ids[cut_begin + next_cut_index])});
          }
          if (next_cut_index >= segment_count) {
            break;
          }
          const int64_t other_owner = boundary_other[current_owner * boundary_count + boundary_id];
          TORCH_CHECK(other_owner >= 0 && other_owner < site_count, "boundary owner transition out of bounds");
          current_owner = other_owner;
          cursor = next_cut_index;
        }
      }

      if (frame == 0) {
        for (const auto& record : current) {
          base_owner.push_back(record.owner);
          base_left.push_back(record.left);
          base_right.push_back(record.right);
        }
        base_offsets.push_back(static_cast<int32_t>(base_owner.size()));
        previous = current;
        has_previous = true;
        continue;
      }
      if (has_previous && same_records(current, previous)) {
        continue;
      }
      change_frame.push_back(static_cast<int32_t>(frame));
      for (const auto& record : current) {
        change_owner.push_back(record.owner);
        change_left.push_back(record.left);
        change_right.push_back(record.right);
      }
      change_offsets.push_back(static_cast<int32_t>(change_owner.size()));
      previous = current;
      has_previous = true;
    }
    track_change_offsets.push_back(static_cast<int32_t>(change_frame.size()));
  }

  return {
      i32_tensor_from_vector(base_offsets),
      i32_tensor_from_vector(base_owner),
      i32_tensor_from_vector(base_left),
      i32_tensor_from_vector(base_right),
      i32_tensor_from_vector(track_change_offsets),
      i32_tensor_from_vector(change_frame),
      i32_tensor_from_vector(change_offsets),
      i32_tensor_from_vector(change_owner),
      i32_tensor_from_vector(change_left),
      i32_tensor_from_vector(change_right),
  };
}

std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor>
gate4_owner_run_delta_replace_from_rays_cpu(
    const torch::Tensor& boundary_f64,
    const torch::Tensor& site_f64,
    const torch::Tensor& site_density_f32,
    const torch::Tensor& rays_f32,
    const torch::Tensor& frame_indices_i64,
    const int64_t frame_count,
    const double near,
    const double far,
    const double invalid_epsilon,
    const double transmittance_threshold,
    const double dedupe_epsilon,
    const double segment_epsilon) {
  TORCH_CHECK(boundary_f64.device().is_cpu(), "boundary_f64 must be a CPU tensor");
  TORCH_CHECK(site_f64.device().is_cpu(), "site_f64 must be a CPU tensor");
  TORCH_CHECK(site_density_f32.device().is_cpu(), "site_density_f32 must be a CPU tensor");
  TORCH_CHECK(rays_f32.device().is_cpu(), "rays_f32 must be a CPU tensor");
  TORCH_CHECK(frame_indices_i64.device().is_cpu(), "frame_indices_i64 must be a CPU tensor");
  TORCH_CHECK(boundary_f64.scalar_type() == torch::kFloat64, "boundary_f64 must be float64");
  TORCH_CHECK(site_f64.scalar_type() == torch::kFloat64, "site_f64 must be float64");
  TORCH_CHECK(site_density_f32.scalar_type() == torch::kFloat32, "site_density_f32 must be float32");
  TORCH_CHECK(rays_f32.scalar_type() == torch::kFloat32, "rays_f32 must be float32");
  TORCH_CHECK(frame_indices_i64.scalar_type() == torch::kInt64, "frame_indices_i64 must be int64");
  TORCH_CHECK(boundary_f64.dim() == 2 && boundary_f64.size(1) == 5, "boundary_f64 must be [boundary, 5]");
  TORCH_CHECK(site_f64.dim() == 2 && site_f64.size(1) == 5, "site_f64 must be [site, 5]");
  TORCH_CHECK(site_density_f32.dim() == 1, "site_density_f32 must be rank-1");
  TORCH_CHECK(rays_f32.dim() == 4 && rays_f32.size(3) == 6, "rays_f32 must be [sample, height, width, 6]");
  TORCH_CHECK(frame_indices_i64.dim() == 1, "frame_indices_i64 must be rank-1");
  TORCH_CHECK(boundary_f64.is_contiguous(), "boundary_f64 must be contiguous");
  TORCH_CHECK(site_f64.is_contiguous(), "site_f64 must be contiguous");
  TORCH_CHECK(site_density_f32.is_contiguous(), "site_density_f32 must be contiguous");
  TORCH_CHECK(rays_f32.is_contiguous(), "rays_f32 must be contiguous");
  TORCH_CHECK(frame_indices_i64.is_contiguous(), "frame_indices_i64 must be contiguous");
  TORCH_CHECK(frame_count > 0, "frame_count must be positive");
  TORCH_CHECK(near < far, "near must be less than far");
  TORCH_CHECK(invalid_epsilon >= 0.0, "invalid_epsilon must be nonnegative");
  TORCH_CHECK(transmittance_threshold >= 0.0, "transmittance_threshold must be nonnegative");
  TORCH_CHECK(dedupe_epsilon >= 0.0, "dedupe_epsilon must be nonnegative");
  TORCH_CHECK(segment_epsilon >= 0.0, "segment_epsilon must be nonnegative");

  const int64_t boundary_count = boundary_f64.size(0);
  const int64_t site_count = site_f64.size(0);
  const int64_t sample_count = rays_f32.size(0);
  const int64_t height = rays_f32.size(1);
  const int64_t width = rays_f32.size(2);
  TORCH_CHECK(site_count > 0, "site_f64 must contain at least one site");
  TORCH_CHECK(site_density_f32.numel() == site_count, "site_density_f32 length must match site count");
  TORCH_CHECK(frame_indices_i64.numel() == sample_count, "frame_indices_i64 length must match sample count");
  TORCH_CHECK(sample_count % frame_count == 0, "sample count must be view_count * frame_count");

  const int64_t view_count = sample_count / frame_count;
  const int64_t track_count = view_count * height * width;
  const double* boundary = boundary_f64.data_ptr<double>();
  const double* site = site_f64.data_ptr<double>();
  const float* site_density = site_density_f32.data_ptr<float>();
  const float* rays = rays_f32.data_ptr<float>();
  const int64_t* frame_indices = frame_indices_i64.data_ptr<int64_t>();

  std::vector<int32_t> base_offsets;
  std::vector<int32_t> base_owner;
  std::vector<int32_t> base_left;
  std::vector<int32_t> base_right;
  std::vector<int32_t> track_change_offsets;
  std::vector<int32_t> change_frame;
  std::vector<int32_t> change_offsets;
  std::vector<int32_t> change_owner;
  std::vector<int32_t> change_left;
  std::vector<int32_t> change_right;
  base_offsets.reserve(static_cast<size_t>(track_count) + 1);
  track_change_offsets.reserve(static_cast<size_t>(track_count) + 1);
  change_offsets.push_back(0);
  base_offsets.push_back(0);
  track_change_offsets.push_back(0);

  std::vector<Gate4DepthCandidate> candidates;
  std::vector<double> cut_depths;
  std::vector<int64_t> cut_ids;
  std::vector<Gate4EndpointRecord> current;
  std::vector<Gate4EndpointRecord> previous;
  candidates.reserve(static_cast<size_t>(boundary_count));
  cut_depths.reserve(static_cast<size_t>(boundary_count) + 2);
  cut_ids.reserve(static_cast<size_t>(boundary_count) + 2);

  auto owner_at = [&](const double px, const double py, const double pz, const double t) -> int64_t {
    int64_t best_owner = 0;
    double best_power = std::numeric_limits<double>::infinity();
    for (int64_t site_id = 0; site_id < site_count; ++site_id) {
      const double dx = px - site[site_id * 5 + 0];
      const double dy = py - site[site_id * 5 + 1];
      const double dz = pz - site[site_id * 5 + 2];
      const double dt = t - site[site_id * 5 + 3];
      const double power = dx * dx + dy * dy + dz * dz + dt * dt - site[site_id * 5 + 4];
      if (power < best_power) {
        best_power = power;
        best_owner = site_id;
      }
    }
    return best_owner;
  };

  int64_t track = 0;
  for (int64_t view = 0; view < view_count; ++view) {
    for (int64_t y = 0; y < height; ++y) {
      for (int64_t x = 0; x < width; ++x, ++track) {
        previous.clear();
        bool has_previous = false;
        for (int64_t frame = 0; frame < frame_count; ++frame) {
          current.clear();
          candidates.clear();
          cut_depths.clear();
          cut_ids.clear();

          const int64_t sample_index = view * frame_count + frame;
          const int64_t ray_base = ((sample_index * height + y) * width + x) * 6;
          const double ox = static_cast<double>(rays[ray_base + 0]);
          const double oy = static_cast<double>(rays[ray_base + 1]);
          const double oz = static_cast<double>(rays[ray_base + 2]);
          const double dx = static_cast<double>(rays[ray_base + 3]);
          const double dy = static_cast<double>(rays[ray_base + 4]);
          const double dz = static_cast<double>(rays[ray_base + 5]);
          const double t = frame_count <= 1
              ? 0.0
              : static_cast<double>(frame_indices[sample_index]) / static_cast<double>(frame_count - 1);

          for (int64_t boundary_id = 0; boundary_id < boundary_count; ++boundary_id) {
            const double nx = boundary[boundary_id * 5 + 0];
            const double ny = boundary[boundary_id * 5 + 1];
            const double nz = boundary[boundary_id * 5 + 2];
            const double nt = boundary[boundary_id * 5 + 3];
            const double b = boundary[boundary_id * 5 + 4];
            const double denom = nx * dx + ny * dy + nz * dz;
            if (std::abs(denom) < invalid_epsilon) {
              continue;
            }
            const double depth = -(nx * ox + ny * oy + nz * oz + nt * t + b) / denom;
            if (std::isfinite(depth) && depth >= near && depth <= far) {
              candidates.push_back(Gate4DepthCandidate{depth, boundary_id});
            }
          }
          std::sort(candidates.begin(), candidates.end(), [](const auto& lhs, const auto& rhs) {
            if (lhs.depth != rhs.depth) {
              return lhs.depth < rhs.depth;
            }
            return lhs.boundary_id < rhs.boundary_id;
          });

          cut_depths.push_back(near);
          cut_ids.push_back(-1);
          bool has_unique_internal = false;
          double previous_internal_depth = 0.0;
          for (const auto& candidate : candidates) {
            if (!has_unique_internal || std::abs(candidate.depth - previous_internal_depth) > dedupe_epsilon) {
              cut_depths.push_back(candidate.depth);
              cut_ids.push_back(candidate.boundary_id);
              previous_internal_depth = candidate.depth;
              has_unique_internal = true;
            }
          }
          cut_depths.push_back(far);
          cut_ids.push_back(-2);

          int32_t current_owner = -1;
          int32_t left_cut = 0;
          int32_t right_cut = 0;
          int64_t run_segment_count = 0;
          auto flush = [&]() {
            if (current_owner < 0 || run_segment_count == 0) {
              return;
            }
            current.push_back(Gate4EndpointRecord{current_owner, left_cut, right_cut});
            current_owner = -1;
            run_segment_count = 0;
          };

          double transmittance = 1.0;
          const int64_t segment_count = static_cast<int64_t>(cut_depths.size()) - 1;
          for (int64_t segment = 0; segment < segment_count; ++segment) {
            if (transmittance <= transmittance_threshold) {
              break;
            }
            const double depth0 = cut_depths[static_cast<size_t>(segment)];
            const double depth1 = cut_depths[static_cast<size_t>(segment + 1)];
            const double length = depth1 - depth0;
            if (length <= segment_epsilon) {
              continue;
            }
            const double mid = 0.5 * (depth0 + depth1);
            const int64_t owner = owner_at(ox + dx * mid, oy + dy * mid, oz + dz * mid, t);
            if (current_owner >= 0 && owner != current_owner) {
              flush();
            }
            if (current_owner < 0) {
              current_owner = static_cast<int32_t>(owner);
              left_cut = static_cast<int32_t>(cut_ids[static_cast<size_t>(segment)]);
            }
            right_cut = static_cast<int32_t>(cut_ids[static_cast<size_t>(segment + 1)]);
            ++run_segment_count;
            const double density = std::max(static_cast<double>(site_density[owner]), 0.0);
            transmittance *= std::exp(-density * length);
          }
          flush();

          if (frame == 0) {
            for (const auto& record : current) {
              base_owner.push_back(record.owner);
              base_left.push_back(record.left);
              base_right.push_back(record.right);
            }
            base_offsets.push_back(static_cast<int32_t>(base_owner.size()));
            previous = current;
            has_previous = true;
            continue;
          }
          if (has_previous && same_records(current, previous)) {
            continue;
          }
          change_frame.push_back(static_cast<int32_t>(frame));
          for (const auto& record : current) {
            change_owner.push_back(record.owner);
            change_left.push_back(record.left);
            change_right.push_back(record.right);
          }
          change_offsets.push_back(static_cast<int32_t>(change_owner.size()));
          previous = current;
          has_previous = true;
        }
        track_change_offsets.push_back(static_cast<int32_t>(change_frame.size()));
      }
    }
  }

  return {
      i32_tensor_from_vector(base_offsets),
      i32_tensor_from_vector(base_owner),
      i32_tensor_from_vector(base_left),
      i32_tensor_from_vector(base_right),
      i32_tensor_from_vector(track_change_offsets),
      i32_tensor_from_vector(change_frame),
      i32_tensor_from_vector(change_offsets),
      i32_tensor_from_vector(change_owner),
      i32_tensor_from_vector(change_left),
      i32_tensor_from_vector(change_right),
  };
}

std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor>
gate4_delta_replace_packed_from_cuts_cpu(
    const torch::Tensor& cut_depths_f64,
    const torch::Tensor& cut_ids_i64,
    const torch::Tensor& cut_offsets_i64,
    const torch::Tensor& start_segments_i64,
    const torch::Tensor& initial_owner_i64,
    const torch::Tensor& boundary_other_by_owner_i64,
    const int64_t frame_count,
    const double epsilon) {
  TORCH_CHECK(cut_depths_f64.device().is_cpu(), "cut_depths_f64 must be a CPU tensor");
  TORCH_CHECK(cut_ids_i64.device().is_cpu(), "cut_ids_i64 must be a CPU tensor");
  TORCH_CHECK(cut_offsets_i64.device().is_cpu(), "cut_offsets_i64 must be a CPU tensor");
  TORCH_CHECK(start_segments_i64.device().is_cpu(), "start_segments_i64 must be a CPU tensor");
  TORCH_CHECK(initial_owner_i64.device().is_cpu(), "initial_owner_i64 must be a CPU tensor");
  TORCH_CHECK(boundary_other_by_owner_i64.device().is_cpu(), "boundary_other_by_owner_i64 must be a CPU tensor");
  TORCH_CHECK(cut_depths_f64.scalar_type() == torch::kFloat64, "cut_depths_f64 must be float64");
  TORCH_CHECK(cut_ids_i64.scalar_type() == torch::kInt64, "cut_ids_i64 must be int64");
  TORCH_CHECK(cut_offsets_i64.scalar_type() == torch::kInt64, "cut_offsets_i64 must be int64");
  TORCH_CHECK(start_segments_i64.scalar_type() == torch::kInt64, "start_segments_i64 must be int64");
  TORCH_CHECK(initial_owner_i64.scalar_type() == torch::kInt64, "initial_owner_i64 must be int64");
  TORCH_CHECK(boundary_other_by_owner_i64.scalar_type() == torch::kInt64, "boundary_other_by_owner_i64 must be int64");
  TORCH_CHECK(cut_depths_f64.dim() == 1, "cut_depths_f64 must be rank-1");
  TORCH_CHECK(cut_ids_i64.dim() == 1, "cut_ids_i64 must be rank-1");
  TORCH_CHECK(cut_offsets_i64.dim() == 1, "cut_offsets_i64 must be rank-1");
  TORCH_CHECK(start_segments_i64.dim() == 1, "start_segments_i64 must be rank-1");
  TORCH_CHECK(initial_owner_i64.dim() == 1, "initial_owner_i64 must be rank-1");
  TORCH_CHECK(boundary_other_by_owner_i64.dim() == 2, "boundary_other_by_owner_i64 must be rank-2");
  TORCH_CHECK(cut_depths_f64.is_contiguous(), "cut_depths_f64 must be contiguous");
  TORCH_CHECK(cut_ids_i64.is_contiguous(), "cut_ids_i64 must be contiguous");
  TORCH_CHECK(cut_offsets_i64.is_contiguous(), "cut_offsets_i64 must be contiguous");
  TORCH_CHECK(start_segments_i64.is_contiguous(), "start_segments_i64 must be contiguous");
  TORCH_CHECK(initial_owner_i64.is_contiguous(), "initial_owner_i64 must be contiguous");
  TORCH_CHECK(boundary_other_by_owner_i64.is_contiguous(), "boundary_other_by_owner_i64 must be contiguous");
  TORCH_CHECK(frame_count > 0, "frame_count must be positive");

  const int64_t work_count = start_segments_i64.numel();
  TORCH_CHECK(initial_owner_i64.numel() == work_count, "initial_owner_i64 length must match start_segments_i64");
  TORCH_CHECK(cut_offsets_i64.numel() == work_count + 1, "cut_offsets_i64 length must be work_count + 1");
  TORCH_CHECK(work_count % frame_count == 0, "work_count must be track_count * frame_count");
  TORCH_CHECK(cut_depths_f64.numel() == cut_ids_i64.numel(), "cut depth/id arrays must have matching lengths");

  const int64_t track_count = work_count / frame_count;
  const int64_t site_count = boundary_other_by_owner_i64.size(0);
  const int64_t boundary_count = boundary_other_by_owner_i64.size(1);
  const double* cut_depths = cut_depths_f64.data_ptr<double>();
  const int64_t* cut_ids = cut_ids_i64.data_ptr<int64_t>();
  const int64_t* cut_offsets = cut_offsets_i64.data_ptr<int64_t>();
  const int64_t* start_segments = start_segments_i64.data_ptr<int64_t>();
  const int64_t* initial_owners = initial_owner_i64.data_ptr<int64_t>();
  const int64_t* boundary_other = boundary_other_by_owner_i64.data_ptr<int64_t>();
  validate_boundary_other_table(boundary_other, site_count, boundary_count);

  std::vector<int32_t> base_offsets;
  std::vector<int32_t> base_owner;
  std::vector<int32_t> base_left;
  std::vector<int32_t> base_right;
  std::vector<int32_t> base_record;
  std::vector<int32_t> track_change_offsets;
  std::vector<int32_t> change_frame;
  std::vector<int32_t> change_offsets;
  std::vector<int32_t> change_owner;
  std::vector<int32_t> change_left;
  std::vector<int32_t> change_right;
  std::vector<int32_t> change_record;
  base_offsets.reserve(static_cast<size_t>(track_count) + 1);
  track_change_offsets.reserve(static_cast<size_t>(track_count) + 1);
  change_offsets.push_back(0);
  base_offsets.push_back(0);
  track_change_offsets.push_back(0);

  std::vector<Gate4EndpointRecord> current;
  std::vector<Gate4EndpointRecord> previous;
  for (int64_t track = 0; track < track_count; ++track) {
    previous.clear();
    bool has_previous = false;
    for (int64_t frame = 0; frame < frame_count; ++frame) {
      const int64_t work = track * frame_count + frame;
      current.clear();
      const int64_t start_segment = start_segments[work];
      int64_t current_owner = initial_owners[work];
      const int64_t cut_begin = cut_offsets[work];
      const int64_t cut_end = cut_offsets[work + 1];
      TORCH_CHECK(cut_begin <= cut_end, "cut_offsets_i64 must be monotonic nondecreasing");
      TORCH_CHECK(cut_begin >= 0 && cut_end <= cut_depths_f64.numel(), "cut offsets out of bounds");
      validate_cut_row_arrays(cut_depths, cut_ids, cut_begin, cut_end, boundary_count);
      const int64_t segment_count = cut_end - cut_begin - 1;
      if (cut_row_is_active(start_segment, current_owner, segment_count)) {
        TORCH_CHECK(current_owner < site_count, "initial owner out of bounds");
        int64_t cursor = start_segment;
        while (cursor < segment_count) {
          int64_t next_cut_index = segment_count;
          int64_t boundary_id = -1;
          for (int64_t local_cut = cursor + 1; local_cut < segment_count; ++local_cut) {
            const int64_t candidate_boundary = cut_ids[cut_begin + local_cut];
            TORCH_CHECK(candidate_boundary >= 0 && candidate_boundary < boundary_count, "boundary id out of bounds");
            if (boundary_other[current_owner * boundary_count + candidate_boundary] >= 0) {
              next_cut_index = local_cut;
              boundary_id = candidate_boundary;
              break;
            }
          }
          if (cut_depths[cut_begin + next_cut_index] - cut_depths[cut_begin + cursor] > epsilon) {
            current.push_back(Gate4EndpointRecord{
                static_cast<int32_t>(current_owner),
                static_cast<int32_t>(cut_ids[cut_begin + cursor]),
                static_cast<int32_t>(cut_ids[cut_begin + next_cut_index])});
          }
          if (next_cut_index >= segment_count) {
            break;
          }
          const int64_t other_owner = boundary_other[current_owner * boundary_count + boundary_id];
          TORCH_CHECK(other_owner >= 0 && other_owner < site_count, "boundary owner transition out of bounds");
          current_owner = other_owner;
          cursor = next_cut_index;
        }
      }

      if (frame == 0) {
        for (const auto& record : current) {
          base_owner.push_back(record.owner);
          base_left.push_back(record.left);
          base_right.push_back(record.right);
          base_record.push_back(pack_endpoint_record_i32(record.owner, record.left, record.right));
        }
        base_offsets.push_back(static_cast<int32_t>(base_owner.size()));
        previous = current;
        has_previous = true;
        continue;
      }
      if (has_previous && same_records(current, previous)) {
        continue;
      }
      change_frame.push_back(static_cast<int32_t>(frame));
      for (const auto& record : current) {
        change_owner.push_back(record.owner);
        change_left.push_back(record.left);
        change_right.push_back(record.right);
        change_record.push_back(pack_endpoint_record_i32(record.owner, record.left, record.right));
      }
      change_offsets.push_back(static_cast<int32_t>(change_owner.size()));
      previous = current;
      has_previous = true;
    }
    track_change_offsets.push_back(static_cast<int32_t>(change_frame.size()));
  }

  return {
      i32_tensor_from_vector(base_offsets),
      i32_tensor_from_vector(base_owner),
      i32_tensor_from_vector(base_left),
      i32_tensor_from_vector(base_right),
      i32_tensor_from_vector(base_record),
      i32_tensor_from_vector(track_change_offsets),
      i32_tensor_from_vector(change_frame),
      i32_tensor_from_vector(change_offsets),
      i32_tensor_from_vector(change_owner),
      i32_tensor_from_vector(change_left),
      i32_tensor_from_vector(change_right),
      i32_tensor_from_vector(change_record),
  };
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
gate4_cut_arrays_from_sorted_cpu(
    const torch::Tensor& sorted_depths_f64,
    const torch::Tensor& sorted_ids_i64,
    const torch::Tensor& valid_counts_i64,
    const torch::Tensor& row_active_i64,
    const torch::Tensor& ray_coeff_f64,
    const torch::Tensor& frame_t_f64,
    const torch::Tensor& site_xyz_f64,
    const torch::Tensor& site_t_f64,
    const torch::Tensor& site_weight_f64,
    const int64_t frame_count,
    const double near,
    const double far,
    const double dedupe_epsilon,
    const double segment_epsilon) {
  TORCH_CHECK(sorted_depths_f64.device().is_cpu(), "sorted_depths_f64 must be a CPU tensor");
  TORCH_CHECK(sorted_ids_i64.device().is_cpu(), "sorted_ids_i64 must be a CPU tensor");
  TORCH_CHECK(valid_counts_i64.device().is_cpu(), "valid_counts_i64 must be a CPU tensor");
  TORCH_CHECK(row_active_i64.device().is_cpu(), "row_active_i64 must be a CPU tensor");
  TORCH_CHECK(ray_coeff_f64.device().is_cpu(), "ray_coeff_f64 must be a CPU tensor");
  TORCH_CHECK(frame_t_f64.device().is_cpu(), "frame_t_f64 must be a CPU tensor");
  TORCH_CHECK(site_xyz_f64.device().is_cpu(), "site_xyz_f64 must be a CPU tensor");
  TORCH_CHECK(site_t_f64.device().is_cpu(), "site_t_f64 must be a CPU tensor");
  TORCH_CHECK(site_weight_f64.device().is_cpu(), "site_weight_f64 must be a CPU tensor");
  TORCH_CHECK(sorted_depths_f64.scalar_type() == torch::kFloat64, "sorted_depths_f64 must be float64");
  TORCH_CHECK(sorted_ids_i64.scalar_type() == torch::kInt64, "sorted_ids_i64 must be int64");
  TORCH_CHECK(valid_counts_i64.scalar_type() == torch::kInt64, "valid_counts_i64 must be int64");
  TORCH_CHECK(row_active_i64.scalar_type() == torch::kInt64, "row_active_i64 must be int64");
  TORCH_CHECK(ray_coeff_f64.scalar_type() == torch::kFloat64, "ray_coeff_f64 must be float64");
  TORCH_CHECK(frame_t_f64.scalar_type() == torch::kFloat64, "frame_t_f64 must be float64");
  TORCH_CHECK(site_xyz_f64.scalar_type() == torch::kFloat64, "site_xyz_f64 must be float64");
  TORCH_CHECK(site_t_f64.scalar_type() == torch::kFloat64, "site_t_f64 must be float64");
  TORCH_CHECK(site_weight_f64.scalar_type() == torch::kFloat64, "site_weight_f64 must be float64");
  TORCH_CHECK(sorted_depths_f64.dim() == 3, "sorted_depths_f64 must be [track, candidate, frame]");
  TORCH_CHECK(sorted_ids_i64.dim() == 3, "sorted_ids_i64 must be [track, candidate, frame]");
  TORCH_CHECK(valid_counts_i64.dim() == 2, "valid_counts_i64 must be [track, frame]");
  TORCH_CHECK(row_active_i64.dim() == 1, "row_active_i64 must be [track]");
  TORCH_CHECK(ray_coeff_f64.dim() == 2 && ray_coeff_f64.size(1) == 12, "ray_coeff_f64 must be [track, 12]");
  TORCH_CHECK(frame_t_f64.dim() == 1, "frame_t_f64 must be rank-1");
  TORCH_CHECK(site_xyz_f64.dim() == 2 && site_xyz_f64.size(1) == 3, "site_xyz_f64 must be [site, 3]");
  TORCH_CHECK(site_t_f64.dim() == 1, "site_t_f64 must be rank-1");
  TORCH_CHECK(site_weight_f64.dim() == 1, "site_weight_f64 must be rank-1");
  TORCH_CHECK(sorted_depths_f64.is_contiguous(), "sorted_depths_f64 must be contiguous");
  TORCH_CHECK(sorted_ids_i64.is_contiguous(), "sorted_ids_i64 must be contiguous");
  TORCH_CHECK(valid_counts_i64.is_contiguous(), "valid_counts_i64 must be contiguous");
  TORCH_CHECK(row_active_i64.is_contiguous(), "row_active_i64 must be contiguous");
  TORCH_CHECK(ray_coeff_f64.is_contiguous(), "ray_coeff_f64 must be contiguous");
  TORCH_CHECK(frame_t_f64.is_contiguous(), "frame_t_f64 must be contiguous");
  TORCH_CHECK(site_xyz_f64.is_contiguous(), "site_xyz_f64 must be contiguous");
  TORCH_CHECK(site_t_f64.is_contiguous(), "site_t_f64 must be contiguous");
  TORCH_CHECK(site_weight_f64.is_contiguous(), "site_weight_f64 must be contiguous");
  TORCH_CHECK(frame_count > 0, "frame_count must be positive");
  TORCH_CHECK(near < far, "near must be less than far");
  TORCH_CHECK(dedupe_epsilon >= 0.0, "dedupe_epsilon must be nonnegative");
  TORCH_CHECK(segment_epsilon >= 0.0, "segment_epsilon must be nonnegative");

  const int64_t track_count = sorted_depths_f64.size(0);
  const int64_t max_candidate_count = sorted_depths_f64.size(1);
  TORCH_CHECK(sorted_depths_f64.size(2) == frame_count, "sorted_depths_f64 frame axis mismatch");
  TORCH_CHECK(sorted_ids_i64.sizes() == sorted_depths_f64.sizes(), "sorted_ids_i64 shape mismatch");
  TORCH_CHECK(valid_counts_i64.size(0) == track_count, "valid_counts_i64 track axis mismatch");
  TORCH_CHECK(valid_counts_i64.size(1) == frame_count, "valid_counts_i64 frame axis mismatch");
  TORCH_CHECK(row_active_i64.numel() == track_count, "row_active_i64 length mismatch");
  TORCH_CHECK(ray_coeff_f64.size(0) == track_count, "ray_coeff_f64 track axis mismatch");
  TORCH_CHECK(frame_t_f64.numel() == frame_count, "frame_t_f64 length mismatch");

  const int64_t site_count = site_xyz_f64.size(0);
  TORCH_CHECK(site_t_f64.numel() == site_count, "site_t_f64 length mismatch");
  TORCH_CHECK(site_weight_f64.numel() == site_count, "site_weight_f64 length mismatch");

  const double* sorted_depths = sorted_depths_f64.data_ptr<double>();
  const int64_t* sorted_ids = sorted_ids_i64.data_ptr<int64_t>();
  const int64_t* valid_counts = valid_counts_i64.data_ptr<int64_t>();
  const int64_t* row_active = row_active_i64.data_ptr<int64_t>();
  const double* ray_coeff = ray_coeff_f64.data_ptr<double>();
  const double* frame_t = frame_t_f64.data_ptr<double>();
  const double* site_xyz = site_xyz_f64.data_ptr<double>();
  const double* site_t = site_t_f64.data_ptr<double>();
  const double* site_weight = site_weight_f64.data_ptr<double>();

  std::vector<double> cut_depths;
  std::vector<int64_t> cut_ids;
  std::vector<int64_t> cut_offsets;
  std::vector<int64_t> start_segments;
  std::vector<int64_t> initial_owner;
  cut_offsets.reserve(static_cast<size_t>(track_count * frame_count) + 1);
  start_segments.reserve(static_cast<size_t>(track_count * frame_count));
  initial_owner.reserve(static_cast<size_t>(track_count * frame_count));
  cut_offsets.push_back(0);

  for (int64_t track = 0; track < track_count; ++track) {
    const bool active = sorted_row_is_active(row_active[track]);
    for (int64_t frame = 0; frame < frame_count; ++frame) {
      if (!active) {
        cut_offsets.push_back(static_cast<int64_t>(cut_depths.size()));
        start_segments.push_back(-1);
        initial_owner.push_back(-1);
        continue;
      }

      const int64_t count = valid_counts[track * frame_count + frame];
      TORCH_CHECK(count >= 0 && count <= max_candidate_count, "valid_counts_i64 value out of bounds");
      const int64_t cut_start = static_cast<int64_t>(cut_depths.size());
      cut_depths.push_back(near);
      cut_ids.push_back(-1);
      if (count > 0) {
        const int64_t first_index = (track * max_candidate_count) * frame_count + frame;
        double previous_depth = checked_sorted_depth(sorted_depths[first_index], near, far);
        cut_depths.push_back(previous_depth);
        cut_ids.push_back(checked_sorted_boundary_id_nonnegative(sorted_ids[first_index]));
        for (int64_t slot = 1; slot < count; ++slot) {
          const int64_t offset = (track * max_candidate_count + slot) * frame_count + frame;
          const double depth = checked_sorted_depth(sorted_depths[offset], near, far);
          check_sorted_depth_order(depth, previous_depth);
          const int64_t boundary_id = checked_sorted_boundary_id_nonnegative(sorted_ids[offset]);
          if (std::abs(depth - previous_depth) > dedupe_epsilon) {
            cut_depths.push_back(depth);
            cut_ids.push_back(boundary_id);
          }
          previous_depth = depth;
        }
      }
      cut_depths.push_back(far);
      cut_ids.push_back(-2);
      cut_offsets.push_back(static_cast<int64_t>(cut_depths.size()));

      const int64_t segment_count = static_cast<int64_t>(cut_depths.size()) - cut_start - 1;
      int64_t start_segment = 0;
      while (start_segment < segment_count &&
             cut_depths[static_cast<size_t>(cut_start + start_segment + 1)] -
                     cut_depths[static_cast<size_t>(cut_start + start_segment)] <=
                 segment_epsilon) {
        ++start_segment;
      }
      if (start_segment >= segment_count) {
        start_segments.push_back(-1);
        initial_owner.push_back(-1);
        continue;
      }
      start_segments.push_back(start_segment);

      const double t = frame_t[frame];
      const double* track_coeff = ray_coeff + track * 12;
      const double midpoint = 0.5 *
          (cut_depths[static_cast<size_t>(cut_start + start_segment)] +
           cut_depths[static_cast<size_t>(cut_start + start_segment + 1)]);
      const double px = track_coeff[0] + track_coeff[3] * t + (track_coeff[6] + track_coeff[9] * t) * midpoint;
      const double py = track_coeff[1] + track_coeff[4] * t + (track_coeff[7] + track_coeff[10] * t) * midpoint;
      const double pz = track_coeff[2] + track_coeff[5] * t + (track_coeff[8] + track_coeff[11] * t) * midpoint;
      int64_t best_owner = 0;
      double best_power = std::numeric_limits<double>::infinity();
      for (int64_t site = 0; site < site_count; ++site) {
        const double dx = px - site_xyz[site * 3];
        const double dy = py - site_xyz[site * 3 + 1];
        const double dz = pz - site_xyz[site * 3 + 2];
        const double dt = t - site_t[site];
        const double power = dx * dx + dy * dy + dz * dz + dt * dt - site_weight[site];
        if (power < best_power) {
          best_power = power;
          best_owner = site;
        }
      }
      initial_owner.push_back(best_owner);
    }
  }

  return {
      f64_tensor_from_vector(cut_depths),
      i64_tensor_from_vector(cut_ids),
      i64_tensor_from_vector(cut_offsets),
      i64_tensor_from_vector(start_segments),
      i64_tensor_from_vector(initial_owner),
  };
}

std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor>
gate4_delta_replace_from_sorted_cpu(
    const torch::Tensor& sorted_depths_f64,
    const torch::Tensor& sorted_ids_i64,
    const torch::Tensor& valid_counts_i64,
    const torch::Tensor& row_active_i64,
    const torch::Tensor& ray_coeff_f64,
    const torch::Tensor& frame_t_f64,
    const torch::Tensor& site_xyz_f64,
    const torch::Tensor& site_t_f64,
    const torch::Tensor& site_weight_f64,
    const torch::Tensor& boundary_other_by_owner_i64,
    const int64_t frame_count,
    const double near,
    const double far,
    const double dedupe_epsilon,
    const double segment_epsilon) {
  TORCH_CHECK(sorted_depths_f64.device().is_cpu(), "sorted_depths_f64 must be a CPU tensor");
  TORCH_CHECK(sorted_ids_i64.device().is_cpu(), "sorted_ids_i64 must be a CPU tensor");
  TORCH_CHECK(valid_counts_i64.device().is_cpu(), "valid_counts_i64 must be a CPU tensor");
  TORCH_CHECK(row_active_i64.device().is_cpu(), "row_active_i64 must be a CPU tensor");
  TORCH_CHECK(ray_coeff_f64.device().is_cpu(), "ray_coeff_f64 must be a CPU tensor");
  TORCH_CHECK(frame_t_f64.device().is_cpu(), "frame_t_f64 must be a CPU tensor");
  TORCH_CHECK(site_xyz_f64.device().is_cpu(), "site_xyz_f64 must be a CPU tensor");
  TORCH_CHECK(site_t_f64.device().is_cpu(), "site_t_f64 must be a CPU tensor");
  TORCH_CHECK(site_weight_f64.device().is_cpu(), "site_weight_f64 must be a CPU tensor");
  TORCH_CHECK(boundary_other_by_owner_i64.device().is_cpu(), "boundary_other_by_owner_i64 must be a CPU tensor");
  TORCH_CHECK(sorted_depths_f64.scalar_type() == torch::kFloat64, "sorted_depths_f64 must be float64");
  TORCH_CHECK(sorted_ids_i64.scalar_type() == torch::kInt64, "sorted_ids_i64 must be int64");
  TORCH_CHECK(valid_counts_i64.scalar_type() == torch::kInt64, "valid_counts_i64 must be int64");
  TORCH_CHECK(row_active_i64.scalar_type() == torch::kInt64, "row_active_i64 must be int64");
  TORCH_CHECK(ray_coeff_f64.scalar_type() == torch::kFloat64, "ray_coeff_f64 must be float64");
  TORCH_CHECK(frame_t_f64.scalar_type() == torch::kFloat64, "frame_t_f64 must be float64");
  TORCH_CHECK(site_xyz_f64.scalar_type() == torch::kFloat64, "site_xyz_f64 must be float64");
  TORCH_CHECK(site_t_f64.scalar_type() == torch::kFloat64, "site_t_f64 must be float64");
  TORCH_CHECK(site_weight_f64.scalar_type() == torch::kFloat64, "site_weight_f64 must be float64");
  TORCH_CHECK(boundary_other_by_owner_i64.scalar_type() == torch::kInt64, "boundary_other_by_owner_i64 must be int64");
  TORCH_CHECK(sorted_depths_f64.dim() == 3, "sorted_depths_f64 must be [track, candidate, frame]");
  TORCH_CHECK(sorted_ids_i64.dim() == 3, "sorted_ids_i64 must be [track, candidate, frame]");
  TORCH_CHECK(valid_counts_i64.dim() == 2, "valid_counts_i64 must be [track, frame]");
  TORCH_CHECK(row_active_i64.dim() == 1, "row_active_i64 must be [track]");
  TORCH_CHECK(ray_coeff_f64.dim() == 2 && ray_coeff_f64.size(1) == 12, "ray_coeff_f64 must be [track, 12]");
  TORCH_CHECK(frame_t_f64.dim() == 1, "frame_t_f64 must be rank-1");
  TORCH_CHECK(site_xyz_f64.dim() == 2 && site_xyz_f64.size(1) == 3, "site_xyz_f64 must be [site, 3]");
  TORCH_CHECK(site_t_f64.dim() == 1, "site_t_f64 must be rank-1");
  TORCH_CHECK(site_weight_f64.dim() == 1, "site_weight_f64 must be rank-1");
  TORCH_CHECK(boundary_other_by_owner_i64.dim() == 2, "boundary_other_by_owner_i64 must be rank-2");
  TORCH_CHECK(sorted_depths_f64.is_contiguous(), "sorted_depths_f64 must be contiguous");
  TORCH_CHECK(sorted_ids_i64.is_contiguous(), "sorted_ids_i64 must be contiguous");
  TORCH_CHECK(valid_counts_i64.is_contiguous(), "valid_counts_i64 must be contiguous");
  TORCH_CHECK(row_active_i64.is_contiguous(), "row_active_i64 must be contiguous");
  TORCH_CHECK(ray_coeff_f64.is_contiguous(), "ray_coeff_f64 must be contiguous");
  TORCH_CHECK(frame_t_f64.is_contiguous(), "frame_t_f64 must be contiguous");
  TORCH_CHECK(site_xyz_f64.is_contiguous(), "site_xyz_f64 must be contiguous");
  TORCH_CHECK(site_t_f64.is_contiguous(), "site_t_f64 must be contiguous");
  TORCH_CHECK(site_weight_f64.is_contiguous(), "site_weight_f64 must be contiguous");
  TORCH_CHECK(boundary_other_by_owner_i64.is_contiguous(), "boundary_other_by_owner_i64 must be contiguous");
  TORCH_CHECK(frame_count > 0, "frame_count must be positive");
  TORCH_CHECK(near < far, "near must be less than far");
  TORCH_CHECK(dedupe_epsilon >= 0.0, "dedupe_epsilon must be nonnegative");
  TORCH_CHECK(segment_epsilon >= 0.0, "segment_epsilon must be nonnegative");

  const int64_t track_count = sorted_depths_f64.size(0);
  const int64_t max_candidate_count = sorted_depths_f64.size(1);
  TORCH_CHECK(sorted_depths_f64.size(2) == frame_count, "sorted_depths_f64 frame axis mismatch");
  TORCH_CHECK(sorted_ids_i64.sizes() == sorted_depths_f64.sizes(), "sorted_ids_i64 shape mismatch");
  TORCH_CHECK(valid_counts_i64.size(0) == track_count, "valid_counts_i64 track axis mismatch");
  TORCH_CHECK(valid_counts_i64.size(1) == frame_count, "valid_counts_i64 frame axis mismatch");
  TORCH_CHECK(row_active_i64.numel() == track_count, "row_active_i64 length mismatch");
  TORCH_CHECK(ray_coeff_f64.size(0) == track_count, "ray_coeff_f64 track axis mismatch");
  TORCH_CHECK(frame_t_f64.numel() == frame_count, "frame_t_f64 length mismatch");

  const int64_t site_count = site_xyz_f64.size(0);
  TORCH_CHECK(site_t_f64.numel() == site_count, "site_t_f64 length mismatch");
  TORCH_CHECK(site_weight_f64.numel() == site_count, "site_weight_f64 length mismatch");
  TORCH_CHECK(boundary_other_by_owner_i64.size(0) == site_count, "boundary_other_by_owner_i64 site axis mismatch");
  const int64_t boundary_count = boundary_other_by_owner_i64.size(1);

  const double* sorted_depths = sorted_depths_f64.data_ptr<double>();
  const int64_t* sorted_ids = sorted_ids_i64.data_ptr<int64_t>();
  const int64_t* valid_counts = valid_counts_i64.data_ptr<int64_t>();
  const int64_t* row_active = row_active_i64.data_ptr<int64_t>();
  const double* ray_coeff = ray_coeff_f64.data_ptr<double>();
  const double* frame_t = frame_t_f64.data_ptr<double>();
  const double* site_xyz = site_xyz_f64.data_ptr<double>();
  const double* site_t = site_t_f64.data_ptr<double>();
  const double* site_weight = site_weight_f64.data_ptr<double>();
  const int64_t* boundary_other = boundary_other_by_owner_i64.data_ptr<int64_t>();
  validate_boundary_other_table(boundary_other, site_count, boundary_count);

  std::vector<int32_t> base_offsets;
  std::vector<int32_t> base_owner;
  std::vector<int32_t> base_left;
  std::vector<int32_t> base_right;
  std::vector<int32_t> track_change_offsets;
  std::vector<int32_t> change_frame;
  std::vector<int32_t> change_offsets;
  std::vector<int32_t> change_owner;
  std::vector<int32_t> change_left;
  std::vector<int32_t> change_right;
  base_offsets.reserve(static_cast<size_t>(track_count) + 1);
  track_change_offsets.reserve(static_cast<size_t>(track_count) + 1);
  change_offsets.push_back(0);
  base_offsets.push_back(0);
  track_change_offsets.push_back(0);

  std::vector<double> cut_depths;
  std::vector<int64_t> cut_ids;
  std::vector<Gate4EndpointRecord> current;
  std::vector<Gate4EndpointRecord> previous;
  for (int64_t track = 0; track < track_count; ++track) {
    const bool active = sorted_row_is_active(row_active[track]);
    previous.clear();
    bool has_previous = false;
    for (int64_t frame = 0; frame < frame_count; ++frame) {
      current.clear();
      if (active) {
        const int64_t count = valid_counts[track * frame_count + frame];
        TORCH_CHECK(count >= 0 && count <= max_candidate_count, "valid_counts_i64 value out of bounds");
        cut_depths.clear();
        cut_ids.clear();
        cut_depths.reserve(static_cast<size_t>(count) + 2);
        cut_ids.reserve(static_cast<size_t>(count) + 2);
        cut_depths.push_back(near);
        cut_ids.push_back(-1);
        if (count > 0) {
          const int64_t first_index = (track * max_candidate_count) * frame_count + frame;
          double previous_depth = checked_sorted_depth(sorted_depths[first_index], near, far);
          cut_depths.push_back(previous_depth);
          cut_ids.push_back(checked_sorted_boundary_id(sorted_ids[first_index], boundary_count));
          for (int64_t slot = 1; slot < count; ++slot) {
            const int64_t offset = (track * max_candidate_count + slot) * frame_count + frame;
            const double depth = checked_sorted_depth(sorted_depths[offset], near, far);
            check_sorted_depth_order(depth, previous_depth);
            const int64_t boundary_id = checked_sorted_boundary_id(sorted_ids[offset], boundary_count);
            if (std::abs(depth - previous_depth) > dedupe_epsilon) {
              cut_depths.push_back(depth);
              cut_ids.push_back(boundary_id);
            }
            previous_depth = depth;
          }
        }
        cut_depths.push_back(far);
        cut_ids.push_back(-2);

        const int64_t segment_count = static_cast<int64_t>(cut_depths.size()) - 1;
        int64_t start_segment = 0;
        while (start_segment < segment_count &&
               cut_depths[static_cast<size_t>(start_segment + 1)] -
                       cut_depths[static_cast<size_t>(start_segment)] <=
                   segment_epsilon) {
          ++start_segment;
        }
        if (start_segment < segment_count) {
          const double t = frame_t[frame];
          const double* track_coeff = ray_coeff + track * 12;
          const double midpoint = 0.5 *
              (cut_depths[static_cast<size_t>(start_segment)] + cut_depths[static_cast<size_t>(start_segment + 1)]);
          const double px = track_coeff[0] + track_coeff[3] * t + (track_coeff[6] + track_coeff[9] * t) * midpoint;
          const double py = track_coeff[1] + track_coeff[4] * t + (track_coeff[7] + track_coeff[10] * t) * midpoint;
          const double pz = track_coeff[2] + track_coeff[5] * t + (track_coeff[8] + track_coeff[11] * t) * midpoint;
          int64_t current_owner = 0;
          double best_power = std::numeric_limits<double>::infinity();
          for (int64_t site = 0; site < site_count; ++site) {
            const double dx = px - site_xyz[site * 3];
            const double dy = py - site_xyz[site * 3 + 1];
            const double dz = pz - site_xyz[site * 3 + 2];
            const double dt = t - site_t[site];
            const double power = dx * dx + dy * dy + dz * dz + dt * dt - site_weight[site];
            if (power < best_power) {
              best_power = power;
              current_owner = site;
            }
          }

          int64_t cursor = start_segment;
          while (cursor < segment_count) {
            int64_t next_cut_index = segment_count;
            int64_t boundary_id = -1;
            for (int64_t local_cut = cursor + 1; local_cut < segment_count; ++local_cut) {
              const int64_t candidate_boundary = cut_ids[static_cast<size_t>(local_cut)];
              TORCH_CHECK(candidate_boundary >= 0 && candidate_boundary < boundary_count, "boundary id out of bounds");
              if (boundary_other[current_owner * boundary_count + candidate_boundary] >= 0) {
                next_cut_index = local_cut;
                boundary_id = candidate_boundary;
                break;
              }
            }
            if (cut_depths[static_cast<size_t>(next_cut_index)] - cut_depths[static_cast<size_t>(cursor)] >
                segment_epsilon) {
              current.push_back(Gate4EndpointRecord{
                  static_cast<int32_t>(current_owner),
                  static_cast<int32_t>(cut_ids[static_cast<size_t>(cursor)]),
                  static_cast<int32_t>(cut_ids[static_cast<size_t>(next_cut_index)])});
            }
            if (next_cut_index >= segment_count) {
              break;
            }
            const int64_t other_owner = boundary_other[current_owner * boundary_count + boundary_id];
            TORCH_CHECK(other_owner >= 0 && other_owner < site_count, "boundary owner transition out of bounds");
            current_owner = other_owner;
            cursor = next_cut_index;
          }
        }
      }

      if (frame == 0) {
        for (const auto& record : current) {
          base_owner.push_back(record.owner);
          base_left.push_back(record.left);
          base_right.push_back(record.right);
        }
        base_offsets.push_back(static_cast<int32_t>(base_owner.size()));
        previous = current;
        has_previous = true;
        continue;
      }
      if (has_previous && same_records(current, previous)) {
        continue;
      }
      change_frame.push_back(static_cast<int32_t>(frame));
      for (const auto& record : current) {
        change_owner.push_back(record.owner);
        change_left.push_back(record.left);
        change_right.push_back(record.right);
      }
      change_offsets.push_back(static_cast<int32_t>(change_owner.size()));
      previous = current;
      has_previous = true;
    }
    track_change_offsets.push_back(static_cast<int32_t>(change_frame.size()));
  }

  return {
      i32_tensor_from_vector(base_offsets),
      i32_tensor_from_vector(base_owner),
      i32_tensor_from_vector(base_left),
      i32_tensor_from_vector(base_right),
      i32_tensor_from_vector(track_change_offsets),
      i32_tensor_from_vector(change_frame),
      i32_tensor_from_vector(change_offsets),
      i32_tensor_from_vector(change_owner),
      i32_tensor_from_vector(change_left),
      i32_tensor_from_vector(change_right),
  };
}

std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor>
gate4_delta_replace_packed_from_sorted_cpu(
    const torch::Tensor& sorted_depths_f64,
    const torch::Tensor& sorted_ids_i64,
    const torch::Tensor& valid_counts_i64,
    const torch::Tensor& row_active_i64,
    const torch::Tensor& ray_coeff_f64,
    const torch::Tensor& frame_t_f64,
    const torch::Tensor& site_xyz_f64,
    const torch::Tensor& site_t_f64,
    const torch::Tensor& site_weight_f64,
    const torch::Tensor& boundary_other_by_owner_i64,
    const int64_t frame_count,
    const double near,
    const double far,
    const double dedupe_epsilon,
    const double segment_epsilon) {
  auto result = gate4_delta_replace_from_sorted_cpu(
      sorted_depths_f64,
      sorted_ids_i64,
      valid_counts_i64,
      row_active_i64,
      ray_coeff_f64,
      frame_t_f64,
      site_xyz_f64,
      site_t_f64,
      site_weight_f64,
      boundary_other_by_owner_i64,
      frame_count,
      near,
      far,
      dedupe_epsilon,
      segment_epsilon);
  auto base_record = pack_endpoint_records_i32_cpu(std::get<1>(result), std::get<2>(result), std::get<3>(result));
  auto change_record = pack_endpoint_records_i32_cpu(std::get<7>(result), std::get<8>(result), std::get<9>(result));
  return {
      std::get<0>(result),
      std::get<1>(result),
      std::get<2>(result),
      std::get<3>(result),
      base_record,
      std::get<4>(result),
      std::get<5>(result),
      std::get<6>(result),
      std::get<7>(result),
      std::get<8>(result),
      std::get<9>(result),
      change_record,
  };
}

std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor>
gate4_delta_replace_packed_from_coeff_csr_cpu(
    const torch::Tensor& row_offsets_i64,
    const torch::Tensor& candidate_ids_i64,
    const torch::Tensor& candidate_depth_coeffs_f64,
    const torch::Tensor& row_index_i64,
    const torch::Tensor& ray_coeff_f64,
    const torch::Tensor& frame_t_f64,
    const torch::Tensor& site_xyz_f64,
    const torch::Tensor& site_t_f64,
    const torch::Tensor& site_weight_f64,
    const torch::Tensor& boundary_other_by_owner_i64,
    const int64_t frame_count,
    const double near,
    const double far,
    const double invalid_epsilon,
    const double dedupe_epsilon,
    const double segment_epsilon) {
  TORCH_CHECK(row_offsets_i64.device().is_cpu(), "row_offsets_i64 must be a CPU tensor");
  TORCH_CHECK(candidate_ids_i64.device().is_cpu(), "candidate_ids_i64 must be a CPU tensor");
  TORCH_CHECK(candidate_depth_coeffs_f64.device().is_cpu(), "candidate_depth_coeffs_f64 must be a CPU tensor");
  TORCH_CHECK(row_index_i64.device().is_cpu(), "row_index_i64 must be a CPU tensor");
  TORCH_CHECK(ray_coeff_f64.device().is_cpu(), "ray_coeff_f64 must be a CPU tensor");
  TORCH_CHECK(frame_t_f64.device().is_cpu(), "frame_t_f64 must be a CPU tensor");
  TORCH_CHECK(site_xyz_f64.device().is_cpu(), "site_xyz_f64 must be a CPU tensor");
  TORCH_CHECK(site_t_f64.device().is_cpu(), "site_t_f64 must be a CPU tensor");
  TORCH_CHECK(site_weight_f64.device().is_cpu(), "site_weight_f64 must be a CPU tensor");
  TORCH_CHECK(boundary_other_by_owner_i64.device().is_cpu(), "boundary_other_by_owner_i64 must be a CPU tensor");
  TORCH_CHECK(row_offsets_i64.scalar_type() == torch::kInt64, "row_offsets_i64 must be int64");
  TORCH_CHECK(candidate_ids_i64.scalar_type() == torch::kInt64, "candidate_ids_i64 must be int64");
  TORCH_CHECK(candidate_depth_coeffs_f64.scalar_type() == torch::kFloat64, "candidate_depth_coeffs_f64 must be float64");
  TORCH_CHECK(row_index_i64.scalar_type() == torch::kInt64, "row_index_i64 must be int64");
  TORCH_CHECK(ray_coeff_f64.scalar_type() == torch::kFloat64, "ray_coeff_f64 must be float64");
  TORCH_CHECK(frame_t_f64.scalar_type() == torch::kFloat64, "frame_t_f64 must be float64");
  TORCH_CHECK(site_xyz_f64.scalar_type() == torch::kFloat64, "site_xyz_f64 must be float64");
  TORCH_CHECK(site_t_f64.scalar_type() == torch::kFloat64, "site_t_f64 must be float64");
  TORCH_CHECK(site_weight_f64.scalar_type() == torch::kFloat64, "site_weight_f64 must be float64");
  TORCH_CHECK(boundary_other_by_owner_i64.scalar_type() == torch::kInt64, "boundary_other_by_owner_i64 must be int64");
  TORCH_CHECK(row_offsets_i64.dim() == 1, "row_offsets_i64 must be rank-1");
  TORCH_CHECK(candidate_ids_i64.dim() == 1, "candidate_ids_i64 must be rank-1");
  TORCH_CHECK(
      candidate_depth_coeffs_f64.dim() == 2 && candidate_depth_coeffs_f64.size(1) == 4,
      "candidate_depth_coeffs_f64 must be [candidate, 4]");
  TORCH_CHECK(row_index_i64.dim() == 1, "row_index_i64 must be rank-1");
  TORCH_CHECK(ray_coeff_f64.dim() == 2 && ray_coeff_f64.size(1) == 12, "ray_coeff_f64 must be [track, 12]");
  TORCH_CHECK(frame_t_f64.dim() == 1, "frame_t_f64 must be rank-1");
  TORCH_CHECK(site_xyz_f64.dim() == 2 && site_xyz_f64.size(1) == 3, "site_xyz_f64 must be [site, 3]");
  TORCH_CHECK(site_t_f64.dim() == 1, "site_t_f64 must be rank-1");
  TORCH_CHECK(site_weight_f64.dim() == 1, "site_weight_f64 must be rank-1");
  TORCH_CHECK(boundary_other_by_owner_i64.dim() == 2, "boundary_other_by_owner_i64 must be rank-2");
  TORCH_CHECK(row_offsets_i64.is_contiguous(), "row_offsets_i64 must be contiguous");
  TORCH_CHECK(candidate_ids_i64.is_contiguous(), "candidate_ids_i64 must be contiguous");
  TORCH_CHECK(candidate_depth_coeffs_f64.is_contiguous(), "candidate_depth_coeffs_f64 must be contiguous");
  TORCH_CHECK(row_index_i64.is_contiguous(), "row_index_i64 must be contiguous");
  TORCH_CHECK(ray_coeff_f64.is_contiguous(), "ray_coeff_f64 must be contiguous");
  TORCH_CHECK(frame_t_f64.is_contiguous(), "frame_t_f64 must be contiguous");
  TORCH_CHECK(site_xyz_f64.is_contiguous(), "site_xyz_f64 must be contiguous");
  TORCH_CHECK(site_t_f64.is_contiguous(), "site_t_f64 must be contiguous");
  TORCH_CHECK(site_weight_f64.is_contiguous(), "site_weight_f64 must be contiguous");
  TORCH_CHECK(boundary_other_by_owner_i64.is_contiguous(), "boundary_other_by_owner_i64 must be contiguous");
  TORCH_CHECK(frame_count > 0, "frame_count must be positive");
  TORCH_CHECK(near < far, "near must be less than far");
  TORCH_CHECK(invalid_epsilon > 0.0, "invalid_epsilon must be positive");
  TORCH_CHECK(dedupe_epsilon >= 0.0, "dedupe_epsilon must be nonnegative");
  TORCH_CHECK(segment_epsilon >= 0.0, "segment_epsilon must be nonnegative");

  const int64_t row_count = row_offsets_i64.numel() - 1;
  TORCH_CHECK(row_count >= 0, "row_offsets_i64 must contain at least one offset");
  const int64_t candidate_count = candidate_ids_i64.numel();
  TORCH_CHECK(
      candidate_depth_coeffs_f64.size(0) == candidate_count,
      "candidate_depth_coeffs_f64 candidate axis mismatch");
  TORCH_CHECK(frame_t_f64.numel() == frame_count, "frame_t_f64 length mismatch");
  const int64_t track_count = row_index_i64.numel();
  TORCH_CHECK(ray_coeff_f64.size(0) == track_count, "ray_coeff_f64 track axis mismatch");
  const int64_t site_count = site_xyz_f64.size(0);
  TORCH_CHECK(site_t_f64.numel() == site_count, "site_t_f64 length mismatch");
  TORCH_CHECK(site_weight_f64.numel() == site_count, "site_weight_f64 length mismatch");
  TORCH_CHECK(boundary_other_by_owner_i64.size(0) == site_count, "boundary_other_by_owner_i64 site axis mismatch");
  const int64_t boundary_count = boundary_other_by_owner_i64.size(1);

  const int64_t* row_offsets = row_offsets_i64.data_ptr<int64_t>();
  const int64_t* candidate_ids = candidate_ids_i64.data_ptr<int64_t>();
  const double* candidate_coeffs = candidate_depth_coeffs_f64.data_ptr<double>();
  const int64_t* row_index = row_index_i64.data_ptr<int64_t>();
  const double* ray_coeff = ray_coeff_f64.data_ptr<double>();
  const double* frame_t = frame_t_f64.data_ptr<double>();
  const double* site_xyz = site_xyz_f64.data_ptr<double>();
  const double* site_t = site_t_f64.data_ptr<double>();
  const double* site_weight = site_weight_f64.data_ptr<double>();
  const int64_t* boundary_other = boundary_other_by_owner_i64.data_ptr<int64_t>();
  validate_boundary_other_table(boundary_other, site_count, boundary_count);

  TORCH_CHECK(row_offsets[0] == 0, "row_offsets_i64[0] must be 0");
  for (int64_t row = 0; row < row_count; ++row) {
    TORCH_CHECK(row_offsets[row] <= row_offsets[row + 1], "row_offsets_i64 must be monotonic");
  }
  TORCH_CHECK(row_offsets[row_count] == candidate_count, "row_offsets_i64[-1] must match candidate count");
  for (int64_t index = 0; index < candidate_count; ++index) {
    checked_sorted_boundary_id(candidate_ids[index], boundary_count);
  }

  std::vector<int32_t> base_offsets;
  std::vector<int32_t> base_owner;
  std::vector<int32_t> base_left;
  std::vector<int32_t> base_right;
  std::vector<int32_t> base_record;
  std::vector<int32_t> track_change_offsets;
  std::vector<int32_t> change_frame;
  std::vector<int32_t> change_offsets;
  std::vector<int32_t> change_owner;
  std::vector<int32_t> change_left;
  std::vector<int32_t> change_right;
  std::vector<int32_t> change_record;
  base_offsets.reserve(static_cast<size_t>(track_count) + 1);
  track_change_offsets.reserve(static_cast<size_t>(track_count) + 1);
  change_offsets.push_back(0);
  base_offsets.push_back(0);
  track_change_offsets.push_back(0);

  std::vector<Gate4DepthCandidate> candidates;
  std::vector<double> cut_depths;
  std::vector<int64_t> cut_ids;
  std::vector<Gate4EndpointRecord> current;
  std::vector<Gate4EndpointRecord> previous;
  for (int64_t track = 0; track < track_count; ++track) {
    const int64_t row = row_index[track];
    const bool active_row = row >= 0 && row < row_count;
    int64_t row_begin = 0;
    int64_t row_end = 0;
    if (active_row) {
      row_begin = row_offsets[row];
      row_end = row_offsets[row + 1];
      TORCH_CHECK(row_begin >= 0 && row_end >= row_begin && row_end <= candidate_count, "row offset bounds");
    }
    previous.clear();
    bool has_previous = false;
    for (int64_t frame = 0; frame < frame_count; ++frame) {
      current.clear();
      if (active_row) {
        const double t = frame_t[frame];
        candidates.clear();
        candidates.reserve(static_cast<size_t>(row_end - row_begin));
        for (int64_t cursor = row_begin; cursor < row_end; ++cursor) {
          const double* coeff = candidate_coeffs + cursor * 4;
          const double denom = coeff[2] + coeff[3] * t;
          if (std::abs(denom) < invalid_epsilon) {
            continue;
          }
          const double depth = (coeff[0] + coeff[1] * t) / denom;
          if (!std::isfinite(depth) || depth < near || depth > far) {
            continue;
          }
          candidates.push_back(Gate4DepthCandidate{depth, candidate_ids[cursor]});
        }
        std::stable_sort(
            candidates.begin(),
            candidates.end(),
            [](const Gate4DepthCandidate& lhs, const Gate4DepthCandidate& rhs) {
              return lhs.depth < rhs.depth;
            });

        cut_depths.clear();
        cut_ids.clear();
        cut_depths.reserve(candidates.size() + 2);
        cut_ids.reserve(candidates.size() + 2);
        cut_depths.push_back(near);
        cut_ids.push_back(-1);
        if (!candidates.empty()) {
          double previous_depth = checked_sorted_depth(candidates[0].depth, near, far);
          cut_depths.push_back(previous_depth);
          cut_ids.push_back(checked_sorted_boundary_id(candidates[0].boundary_id, boundary_count));
          for (size_t slot = 1; slot < candidates.size(); ++slot) {
            const double depth = checked_sorted_depth(candidates[slot].depth, near, far);
            check_sorted_depth_order(depth, previous_depth);
            const int64_t boundary_id = checked_sorted_boundary_id(candidates[slot].boundary_id, boundary_count);
            if (std::abs(depth - previous_depth) > dedupe_epsilon) {
              cut_depths.push_back(depth);
              cut_ids.push_back(boundary_id);
            }
            previous_depth = depth;
          }
        }
        cut_depths.push_back(far);
        cut_ids.push_back(-2);

        const int64_t segment_count = static_cast<int64_t>(cut_depths.size()) - 1;
        int64_t start_segment = 0;
        while (start_segment < segment_count &&
               cut_depths[static_cast<size_t>(start_segment + 1)] -
                       cut_depths[static_cast<size_t>(start_segment)] <=
                   segment_epsilon) {
          ++start_segment;
        }
        if (start_segment < segment_count) {
          const double* track_coeff = ray_coeff + track * 12;
          const double midpoint = 0.5 *
              (cut_depths[static_cast<size_t>(start_segment)] + cut_depths[static_cast<size_t>(start_segment + 1)]);
          const double px = track_coeff[0] + track_coeff[3] * t + (track_coeff[6] + track_coeff[9] * t) * midpoint;
          const double py = track_coeff[1] + track_coeff[4] * t + (track_coeff[7] + track_coeff[10] * t) * midpoint;
          const double pz = track_coeff[2] + track_coeff[5] * t + (track_coeff[8] + track_coeff[11] * t) * midpoint;
          int64_t current_owner = 0;
          double best_power = std::numeric_limits<double>::infinity();
          for (int64_t site = 0; site < site_count; ++site) {
            const double dx = px - site_xyz[site * 3];
            const double dy = py - site_xyz[site * 3 + 1];
            const double dz = pz - site_xyz[site * 3 + 2];
            const double dt = t - site_t[site];
            const double power = dx * dx + dy * dy + dz * dz + dt * dt - site_weight[site];
            if (power < best_power) {
              best_power = power;
              current_owner = site;
            }
          }

          int64_t cursor = start_segment;
          while (cursor < segment_count) {
            int64_t next_cut_index = segment_count;
            int64_t boundary_id = -1;
            for (int64_t local_cut = cursor + 1; local_cut < segment_count; ++local_cut) {
              const int64_t candidate_boundary = cut_ids[static_cast<size_t>(local_cut)];
              TORCH_CHECK(candidate_boundary >= 0 && candidate_boundary < boundary_count, "boundary id out of bounds");
              if (boundary_other[current_owner * boundary_count + candidate_boundary] >= 0) {
                next_cut_index = local_cut;
                boundary_id = candidate_boundary;
                break;
              }
            }
            if (cut_depths[static_cast<size_t>(next_cut_index)] - cut_depths[static_cast<size_t>(cursor)] >
                segment_epsilon) {
              current.push_back(Gate4EndpointRecord{
                  static_cast<int32_t>(current_owner),
                  static_cast<int32_t>(cut_ids[static_cast<size_t>(cursor)]),
                  static_cast<int32_t>(cut_ids[static_cast<size_t>(next_cut_index)])});
            }
            if (next_cut_index >= segment_count) {
              break;
            }
            const int64_t other_owner = boundary_other[current_owner * boundary_count + boundary_id];
            TORCH_CHECK(other_owner >= 0 && other_owner < site_count, "boundary owner transition out of bounds");
            current_owner = other_owner;
            cursor = next_cut_index;
          }
        }
      }

      if (frame == 0) {
        for (const auto& record : current) {
          base_owner.push_back(record.owner);
          base_left.push_back(record.left);
          base_right.push_back(record.right);
          base_record.push_back(pack_endpoint_record_i32(record.owner, record.left, record.right));
        }
        base_offsets.push_back(static_cast<int32_t>(base_owner.size()));
        previous = current;
        has_previous = true;
        continue;
      }
      if (has_previous && same_records(current, previous)) {
        continue;
      }
      change_frame.push_back(static_cast<int32_t>(frame));
      for (const auto& record : current) {
        change_owner.push_back(record.owner);
        change_left.push_back(record.left);
        change_right.push_back(record.right);
        change_record.push_back(pack_endpoint_record_i32(record.owner, record.left, record.right));
      }
      change_offsets.push_back(static_cast<int32_t>(change_owner.size()));
      previous = current;
      has_previous = true;
    }
    track_change_offsets.push_back(static_cast<int32_t>(change_frame.size()));
  }

  return {
      i32_tensor_from_vector(base_offsets),
      i32_tensor_from_vector(base_owner),
      i32_tensor_from_vector(base_left),
      i32_tensor_from_vector(base_right),
      i32_tensor_from_vector(base_record),
      i32_tensor_from_vector(track_change_offsets),
      i32_tensor_from_vector(change_frame),
      i32_tensor_from_vector(change_offsets),
      i32_tensor_from_vector(change_owner),
      i32_tensor_from_vector(change_left),
      i32_tensor_from_vector(change_right),
      i32_tensor_from_vector(change_record),
  };
}

torch::Tensor metal_count_power_boundary_events(
    const torch::Tensor& boundary_f32,
    const torch::Tensor& boundary_u32,
    const torch::Tensor& beam_f32,
    const torch::Tensor& beam_u32,
    const torch::Tensor& config_i32,
    const torch::Tensor& config_f32);

std::tuple<torch::Tensor, torch::Tensor> metal_shared_signal_replay(
    const torch::Tensor& boundary_f32,
    const torch::Tensor& candidate_mask_u32,
    const torch::Tensor& sites_f32,
    const torch::Tensor& site_signal_f32,
    const torch::Tensor& beam_f32,
    const torch::Tensor& frame_t_f32,
    const torch::Tensor& grad_output_f32,
    const torch::Tensor& config_i32,
    const torch::Tensor& config_f32);

std::tuple<torch::Tensor, torch::Tensor> metal_shared_rgb_replay(
    const torch::Tensor& boundary_f32,
    const torch::Tensor& candidate_mask_u32,
    const torch::Tensor& sites_f32,
    const torch::Tensor& site_rgb_f32,
    const torch::Tensor& beam_f32,
    const torch::Tensor& frame_t_f32,
    const torch::Tensor& grad_output_rgb_f32,
    const torch::Tensor& config_i32,
    const torch::Tensor& config_f32);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> metal_shared_rgba_depth_replay(
    const torch::Tensor& boundary_f32,
    const torch::Tensor& candidate_mask_u32,
    const torch::Tensor& sites_f32,
    const torch::Tensor& site_rgba_f32,
    const torch::Tensor& beam_f32,
    const torch::Tensor& frame_t_f32,
    const torch::Tensor& config_i32,
    const torch::Tensor& config_f32);

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
    const torch::Tensor& config_f32);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> metal_realray_rgba_depth_replay(
    const torch::Tensor& boundary_f32,
    const torch::Tensor& sites_f32,
    const torch::Tensor& site_rgba_f32,
    const torch::Tensor& rays_f32,
    const torch::Tensor& frame_t_f32,
    const torch::Tensor& config_i32,
    const torch::Tensor& config_f32);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> metal_shared_realray_rgba_depth_replay(
    const torch::Tensor& boundary_f32,
    const torch::Tensor& candidate_mask_i32,
    const torch::Tensor& sites_f32,
    const torch::Tensor& site_rgba_f32,
    const torch::Tensor& track_rays_f32,
    const torch::Tensor& frame_t_f32,
    const torch::Tensor& config_i32,
    const torch::Tensor& config_f32);

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
    const torch::Tensor& config_f32);

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
    const torch::Tensor& config_f32);

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
    const torch::Tensor& config_f32);

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
    const torch::Tensor& config_f32);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> metal_fused_slab_affine_coeff_realray_rgba_depth_replay(
    const torch::Tensor& row_index_i32,
    const torch::Tensor& candidate_row_offsets_i32,
    const torch::Tensor& candidate_depth_coeff_f32,
    const torch::Tensor& sites_f32,
    const torch::Tensor& site_rgba_f32,
    const torch::Tensor& ray_coeff_f32,
    const torch::Tensor& frame_t_f32,
    const torch::Tensor& config_i32,
    const torch::Tensor& config_f32);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> metal_fused_slab_affine_coeff16_realray_rgba_depth_replay(
    const torch::Tensor& row_index_i32,
    const torch::Tensor& candidate_row_offsets_i32,
    const torch::Tensor& candidate_depth_coeff_f16,
    const torch::Tensor& sites_f32,
    const torch::Tensor& site_rgba_f32,
    const torch::Tensor& ray_coeff_f32,
    const torch::Tensor& frame_t_f32,
    const torch::Tensor& config_i32,
    const torch::Tensor& config_f32);

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
    const torch::Tensor& config_f32);

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
    const torch::Tensor& config_f32);

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
    const torch::Tensor& config_f32);

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
    const torch::Tensor& config_f32);

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
    const torch::Tensor& config_f32);

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
    const torch::Tensor& config_f32);

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
    const torch::Tensor& config_f32);

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
    const torch::Tensor& config_f32);

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
    const torch::Tensor& config_f32);

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
    const torch::Tensor& config_f32);

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
    const torch::Tensor& config_f32);

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
    const torch::Tensor& config_f32);

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
    const torch::Tensor& config_f32);

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
    const torch::Tensor& config_f32);

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
    const torch::Tensor& config_f32);

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
    const torch::Tensor& config_f32);

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
    const torch::Tensor& config_f32);

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
    const torch::Tensor& config_f32);

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
    const torch::Tensor& config_f32);

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
    const torch::Tensor& config_f32);

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
    const torch::Tensor& config_f32);

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
    const torch::Tensor& config_f32);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> metal_segment_tape_rgba_depth_replay(
    const torch::Tensor& segment_offsets_i32,
    const torch::Tensor& segment_owner_i32,
    const torch::Tensor& segment_length_f32,
    const torch::Tensor& segment_mid_f32,
    const torch::Tensor& site_rgba_f32,
    const torch::Tensor& config_i32,
    const torch::Tensor& config_f32);

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
    const torch::Tensor& config_f32);

std::tuple<torch::Tensor, torch::Tensor> metal_segment_tape_mse_vjp_direct_atomic_rgb_only(
    const torch::Tensor& segment_offsets_i32,
    const torch::Tensor& segment_owner_i32,
    const torch::Tensor& segment_length_f32,
    const torch::Tensor& site_rgba_f32,
    const torch::Tensor& target_rgb_f32,
    const torch::Tensor& config_i32,
    const torch::Tensor& config_f32);

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
    const torch::Tensor& config_f32);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> metal_endpoint_run_rgba_depth_replay(
    const torch::Tensor& run_offsets_i32,
    const torch::Tensor& run_owner_i32,
    const torch::Tensor& run_start_f32,
    const torch::Tensor& run_end_f32,
    const torch::Tensor& site_rgba_f32,
    const torch::Tensor& config_i32,
    const torch::Tensor& config_f32);

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
    const torch::Tensor& config_f32);

std::tuple<torch::Tensor, torch::Tensor> metal_endpoint_run_mse_vjp_direct_atomic_rgb_only(
    const torch::Tensor& run_offsets_i32,
    const torch::Tensor& run_owner_i32,
    const torch::Tensor& run_start_f32,
    const torch::Tensor& run_end_f32,
    const torch::Tensor& site_rgba_f32,
    const torch::Tensor& target_rgb_f32,
    const torch::Tensor& config_i32,
    const torch::Tensor& config_f32);

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
    const torch::Tensor& config_f32);

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
    const torch::Tensor& config_f32);

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
    const torch::Tensor& config_f32);

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
    const torch::Tensor& config_f32);

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
    const torch::Tensor& config_f32);

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
    const torch::Tensor& config_f32);

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
    const torch::Tensor& config_f32);

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
    const torch::Tensor& config_f32);

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
    const torch::Tensor& config_f32);

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
    const torch::Tensor& config_f32);

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
    const torch::Tensor& config_f32);

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
    const torch::Tensor& config_f32);

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
    int64_t boundary_count,
    int64_t track_count,
    int64_t frame_count,
    int64_t site_count,
    int64_t base_record_count,
    int64_t change_count,
    int64_t change_record_count);

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
    int64_t track_count,
    int64_t frame_count,
    int64_t site_count);

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
    int64_t boundary_count,
    int64_t track_count,
    int64_t frame_count,
    int64_t site_count,
    int64_t base_record_count,
    int64_t change_count,
    int64_t change_record_count);

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
    int64_t track_count,
    int64_t frame_count,
    int64_t site_count);

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
    int64_t boundary_count,
    int64_t track_count,
    int64_t frame_count,
    int64_t site_count,
    int64_t base_record_count,
    int64_t change_count,
    int64_t change_record_count);

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
    int64_t track_count,
    int64_t frame_count,
    int64_t site_count);

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
    int64_t boundary_count,
    int64_t track_count,
    int64_t frame_count,
    int64_t site_count,
    int64_t base_record_count,
    int64_t change_record_count);

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
    int64_t track_count,
    int64_t frame_count,
    int64_t site_count);

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
    int64_t boundary_count,
    int64_t track_count,
    int64_t frame_count,
    int64_t site_count,
    int64_t base_record_count,
    int64_t change_record_count);

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
    int64_t track_count,
    int64_t frame_count,
    int64_t site_count);

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
    int64_t boundary_count,
    int64_t track_count,
    int64_t frame_count,
    int64_t site_count,
    int64_t base_record_count,
    int64_t change_count,
    int64_t change_record_count);

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
    int64_t boundary_count,
    int64_t track_count,
    int64_t frame_count,
    int64_t site_count,
    int64_t base_record_count,
    int64_t change_count,
    int64_t change_record_count);

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
    int64_t boundary_count,
    int64_t track_count,
    int64_t frame_count,
    int64_t site_count,
    int64_t base_record_count,
    int64_t change_count,
    int64_t change_record_count);

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
    const torch::Tensor& config_f32);

std::tuple<torch::Tensor, torch::Tensor>
metal_endpoint_record_delta_replace_factorized_packed_framegroup16_recompute_mse_vjp_direct_atomic_rgb_only(
    const torch::Tensor& boundary_f32,
    const torch::Tensor& track_ray_coeff_f32,
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
    const torch::Tensor& config_f32);

std::tuple<torch::Tensor, torch::Tensor>
metal_endpoint_record_delta_replace_factorized_frameselect_recompute_mse_vjp_direct_atomic_rgb_only(
    const torch::Tensor& boundary_f32,
    const torch::Tensor& track_ray_coeff_f32,
    const torch::Tensor& frame_t_f32,
    const torch::Tensor& base_offsets_i32,
    const torch::Tensor& base_record_i32,
    const torch::Tensor& frame_change_index_i16,
    const torch::Tensor& change_offsets_i32,
    const torch::Tensor& change_record_i32,
    const torch::Tensor& site_rgba_f32,
    const torch::Tensor& target_rgb_f32,
    const torch::Tensor& config_i32,
    const torch::Tensor& config_f32);

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
    const torch::Tensor& config_f32);

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
    const torch::Tensor& config_f32);

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
    const torch::Tensor& config_f32);

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
    const torch::Tensor& config_f32);

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
    const torch::Tensor& config_f32);

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
    const torch::Tensor& config_f32);

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
    const torch::Tensor& config_f32);

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
    const torch::Tensor& config_f32);

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
    const torch::Tensor& config_f32);

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
    const torch::Tensor& config_f32);

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
    const torch::Tensor& config_f32);

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
    const torch::Tensor& config_f32);

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
    const torch::Tensor& config_f32);

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
    const torch::Tensor& config_f32);

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
    const torch::Tensor& config_f32);

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
    const torch::Tensor& config_f32);

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
    const torch::Tensor& config_f32);

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
    const torch::Tensor& config_f32);

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
    const torch::Tensor& config_f32);

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
    const torch::Tensor& config_f32);

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
    const torch::Tensor& config_f32);

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
    const torch::Tensor& config_f32);

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
    const torch::Tensor& config_f32);

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
    const torch::Tensor& config_f32);

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
    const torch::Tensor& config_f32);

namespace {

torch::Tensor count_power_boundary_events_dispatch(
    const torch::Tensor& boundary_f32,
    const torch::Tensor& boundary_u32,
    const torch::Tensor& beam_f32,
    const torch::Tensor& beam_u32,
    const torch::Tensor& config_i32,
    const torch::Tensor& config_f32) {
#if defined(__APPLE__)
  if (boundary_f32.device().is_mps()) {
    return metal_count_power_boundary_events(
        boundary_f32,
        boundary_u32,
        beam_f32,
        beam_u32,
        config_i32,
        config_f32);
  }
#endif
  TORCH_CHECK(
      false,
      "world_foam_lane2_fused_slab_v0.count_power_boundary_events: no backend available for device ",
      boundary_f32.device());
}

std::tuple<torch::Tensor, torch::Tensor> shared_signal_replay_dispatch(
    const torch::Tensor& boundary_f32,
    const torch::Tensor& candidate_mask_u32,
    const torch::Tensor& sites_f32,
    const torch::Tensor& site_signal_f32,
    const torch::Tensor& beam_f32,
    const torch::Tensor& frame_t_f32,
    const torch::Tensor& grad_output_f32,
    const torch::Tensor& config_i32,
    const torch::Tensor& config_f32) {
#if defined(__APPLE__)
  if (boundary_f32.device().is_mps()) {
    return metal_shared_signal_replay(
        boundary_f32,
        candidate_mask_u32,
        sites_f32,
        site_signal_f32,
        beam_f32,
        frame_t_f32,
        grad_output_f32,
        config_i32,
        config_f32);
  }
#endif
  TORCH_CHECK(
      false,
      "world_foam_lane2_fused_slab_v0.shared_signal_replay: no backend available for device ",
      boundary_f32.device());
}

std::tuple<torch::Tensor, torch::Tensor> shared_rgb_replay_dispatch(
    const torch::Tensor& boundary_f32,
    const torch::Tensor& candidate_mask_u32,
    const torch::Tensor& sites_f32,
    const torch::Tensor& site_rgb_f32,
    const torch::Tensor& beam_f32,
    const torch::Tensor& frame_t_f32,
    const torch::Tensor& grad_output_rgb_f32,
    const torch::Tensor& config_i32,
    const torch::Tensor& config_f32) {
#if defined(__APPLE__)
  if (boundary_f32.device().is_mps()) {
    return metal_shared_rgb_replay(
        boundary_f32,
        candidate_mask_u32,
        sites_f32,
        site_rgb_f32,
        beam_f32,
        frame_t_f32,
        grad_output_rgb_f32,
        config_i32,
        config_f32);
  }
#endif
  TORCH_CHECK(
      false,
      "world_foam_lane2_fused_slab_v0.shared_rgb_replay: no backend available for device ",
      boundary_f32.device());
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> shared_rgba_depth_replay_dispatch(
    const torch::Tensor& boundary_f32,
    const torch::Tensor& candidate_mask_u32,
    const torch::Tensor& sites_f32,
    const torch::Tensor& site_rgba_f32,
    const torch::Tensor& beam_f32,
    const torch::Tensor& frame_t_f32,
    const torch::Tensor& config_i32,
    const torch::Tensor& config_f32) {
#if defined(__APPLE__)
  if (boundary_f32.device().is_mps()) {
    return metal_shared_rgba_depth_replay(
        boundary_f32,
        candidate_mask_u32,
        sites_f32,
        site_rgba_f32,
        beam_f32,
        frame_t_f32,
        config_i32,
        config_f32);
  }
#endif
  TORCH_CHECK(
      false,
      "world_foam_lane2_fused_slab_v0.shared_rgba_depth_replay: no backend available for device ",
      boundary_f32.device());
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> shared_rgba_depth_vjp_dispatch(
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
#if defined(__APPLE__)
  if (boundary_f32.device().is_mps()) {
    return metal_shared_rgba_depth_vjp(
        boundary_f32,
        candidate_mask_u32,
        sites_f32,
        site_rgba_f32,
        beam_f32,
        frame_t_f32,
        grad_rgb_f32,
        grad_alpha_f32,
        grad_depth_f32,
        config_i32,
        config_f32);
  }
#endif
  TORCH_CHECK(
      false,
      "world_foam_lane2_fused_slab_v0.shared_rgba_depth_vjp: no backend available for device ",
      boundary_f32.device());
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> realray_rgba_depth_replay_dispatch(
    const torch::Tensor& boundary_f32,
    const torch::Tensor& sites_f32,
    const torch::Tensor& site_rgba_f32,
    const torch::Tensor& rays_f32,
    const torch::Tensor& frame_t_f32,
    const torch::Tensor& config_i32,
    const torch::Tensor& config_f32) {
#if defined(__APPLE__)
  if (boundary_f32.device().is_mps()) {
    return metal_realray_rgba_depth_replay(
        boundary_f32,
        sites_f32,
        site_rgba_f32,
        rays_f32,
        frame_t_f32,
        config_i32,
        config_f32);
  }
#endif
  TORCH_CHECK(
      false,
      "world_foam_lane2_fused_slab_v0.realray_rgba_depth_replay: no backend available for device ",
      boundary_f32.device());
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> shared_realray_rgba_depth_replay_dispatch(
    const torch::Tensor& boundary_f32,
    const torch::Tensor& candidate_mask_i32,
    const torch::Tensor& sites_f32,
    const torch::Tensor& site_rgba_f32,
    const torch::Tensor& track_rays_f32,
    const torch::Tensor& frame_t_f32,
    const torch::Tensor& config_i32,
    const torch::Tensor& config_f32) {
#if defined(__APPLE__)
  if (boundary_f32.device().is_mps()) {
    return metal_shared_realray_rgba_depth_replay(
        boundary_f32,
        candidate_mask_i32,
        sites_f32,
        site_rgba_f32,
        track_rays_f32,
        frame_t_f32,
        config_i32,
        config_f32);
  }
#endif
  TORCH_CHECK(
      false,
      "world_foam_lane2_fused_slab_v0.shared_realray_rgba_depth_replay: no backend available for device ",
      boundary_f32.device());
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> shared_realray_rgba_depth_vjp_dispatch(
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
#if defined(__APPLE__)
  if (boundary_f32.device().is_mps()) {
    return metal_shared_realray_rgba_depth_vjp(
        boundary_f32,
        candidate_mask_i32,
        sites_f32,
        site_rgba_f32,
        track_rays_f32,
        frame_t_f32,
        grad_rgb_f32,
        grad_alpha_f32,
        grad_depth_f32,
        config_i32,
        config_f32);
  }
#endif
  TORCH_CHECK(
      false,
      "world_foam_lane2_fused_slab_v0.shared_realray_rgba_depth_vjp: no backend available for device ",
      boundary_f32.device());
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> shared_realray_rgba_depth_vjp_reduce_dispatch(
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
#if defined(__APPLE__)
  if (boundary_f32.device().is_mps()) {
    return metal_shared_realray_rgba_depth_vjp_reduce(
        boundary_f32,
        candidate_mask_i32,
        sites_f32,
        site_rgba_f32,
        track_rays_f32,
        frame_t_f32,
        grad_rgb_f32,
        grad_alpha_f32,
        grad_depth_f32,
        config_i32,
        config_f32);
  }
#endif
  TORCH_CHECK(
      false,
      "world_foam_lane2_fused_slab_v0.shared_realray_rgba_depth_vjp_reduce: no backend available for device ",
      boundary_f32.device());
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> shared_realray_rgba_depth_vjp_reduce_csr_dispatch(
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
#if defined(__APPLE__)
  if (boundary_f32.device().is_mps()) {
    return metal_shared_realray_rgba_depth_vjp_reduce_csr(
        boundary_f32,
        row_index_i32,
        candidate_row_offsets_i32,
        candidate_boundary_ids_i32,
        sites_f32,
        site_rgba_f32,
        track_rays_f32,
        frame_t_f32,
        grad_rgb_f32,
        grad_alpha_f32,
        grad_depth_f32,
        config_i32,
        config_f32);
  }
#endif
  TORCH_CHECK(
      false,
      "world_foam_lane2_fused_slab_v0.shared_realray_rgba_depth_vjp_reduce_csr: no backend available for device ",
      boundary_f32.device());
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> fused_slab_affine_realray_rgba_depth_replay_dispatch(
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
#if defined(__APPLE__)
  if (boundary_f32.device().is_mps()) {
    return metal_fused_slab_affine_realray_rgba_depth_replay(
        boundary_f32,
        row_index_i32,
        candidate_row_offsets_i32,
        candidate_boundary_ids_i32,
        sites_f32,
        site_rgba_f32,
        ray_coeff_f32,
        frame_t_f32,
        config_i32,
        config_f32);
  }
#endif
  TORCH_CHECK(
      false,
      "world_foam_lane2_fused_slab_v0.fused_slab_affine_realray_rgba_depth_replay: no backend available for device ",
      boundary_f32.device());
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> fused_slab_affine_coeff_realray_rgba_depth_replay_dispatch(
    const torch::Tensor& row_index_i32,
    const torch::Tensor& candidate_row_offsets_i32,
    const torch::Tensor& candidate_depth_coeff_f32,
    const torch::Tensor& sites_f32,
    const torch::Tensor& site_rgba_f32,
    const torch::Tensor& ray_coeff_f32,
    const torch::Tensor& frame_t_f32,
    const torch::Tensor& config_i32,
    const torch::Tensor& config_f32) {
#if defined(__APPLE__)
  if (row_index_i32.device().is_mps()) {
    return metal_fused_slab_affine_coeff_realray_rgba_depth_replay(
        row_index_i32,
        candidate_row_offsets_i32,
        candidate_depth_coeff_f32,
        sites_f32,
        site_rgba_f32,
        ray_coeff_f32,
        frame_t_f32,
        config_i32,
        config_f32);
  }
#endif
  TORCH_CHECK(
      false,
      "world_foam_lane2_fused_slab_v0.fused_slab_affine_coeff_realray_rgba_depth_replay: no backend available for device ",
      row_index_i32.device());
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> fused_slab_affine_coeff16_realray_rgba_depth_replay_dispatch(
    const torch::Tensor& row_index_i32,
    const torch::Tensor& candidate_row_offsets_i32,
    const torch::Tensor& candidate_depth_coeff_f16,
    const torch::Tensor& sites_f32,
    const torch::Tensor& site_rgba_f32,
    const torch::Tensor& ray_coeff_f32,
    const torch::Tensor& frame_t_f32,
    const torch::Tensor& config_i32,
    const torch::Tensor& config_f32) {
#if defined(__APPLE__)
  if (row_index_i32.device().is_mps()) {
    return metal_fused_slab_affine_coeff16_realray_rgba_depth_replay(
        row_index_i32,
        candidate_row_offsets_i32,
        candidate_depth_coeff_f16,
        sites_f32,
        site_rgba_f32,
        ray_coeff_f32,
        frame_t_f32,
        config_i32,
        config_f32);
  }
#endif
  TORCH_CHECK(
      false,
      "world_foam_lane2_fused_slab_v0.fused_slab_affine_coeff16_realray_rgba_depth_replay: no backend available for device ",
      row_index_i32.device());
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> fused_slab_affine_num32_den16_realray_rgba_depth_replay_dispatch(
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
#if defined(__APPLE__)
  if (row_index_i32.device().is_mps()) {
    return metal_fused_slab_affine_num32_den16_realray_rgba_depth_replay(
        row_index_i32,
        candidate_row_offsets_i32,
        candidate_depth_num_f32,
        candidate_depth_den_f16,
        sites_f32,
        site_rgba_f32,
        ray_coeff_f32,
        frame_t_f32,
        config_i32,
        config_f32);
  }
#endif
  TORCH_CHECK(
      false,
      "world_foam_lane2_fused_slab_v0.fused_slab_affine_num32_den16_realray_rgba_depth_replay: no backend available for device ",
      row_index_i32.device());
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
fused_slab_affine_num32_den16_ownerupdate_realray_rgba_depth_replay_dispatch(
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
#if defined(__APPLE__)
  if (row_index_i32.device().is_mps()) {
    return metal_fused_slab_affine_num32_den16_ownerupdate_realray_rgba_depth_replay(
        row_index_i32,
        candidate_row_offsets_i32,
        candidate_boundary_ids_i32,
        candidate_depth_num_f32,
        candidate_depth_den_f16,
        boundary_site_pairs_i32,
        sites_f32,
        site_rgba_f32,
        ray_coeff_f32,
        frame_t_f32,
        config_i32,
        config_f32);
  }
#endif
  TORCH_CHECK(
      false,
      "world_foam_lane2_fused_slab_v0.fused_slab_affine_num32_den16_ownerupdate_realray_rgba_depth_replay: no backend available for device ",
      row_index_i32.device());
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
fused_slab_affine_num32_den16_vjp_reduce_dispatch(
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
#if defined(__APPLE__)
  if (row_index_i32.device().is_mps()) {
    return metal_fused_slab_affine_num32_den16_vjp_reduce(
        row_index_i32,
        candidate_row_offsets_i32,
        candidate_depth_num_f32,
        candidate_depth_den_f16,
        sites_f32,
        site_rgba_f32,
        ray_coeff_f32,
        frame_t_f32,
        grad_rgb_f32,
        grad_alpha_f32,
        grad_depth_f32,
        config_i32,
        config_f32);
  }
#endif
  TORCH_CHECK(
      false,
      "world_foam_lane2_fused_slab_v0.fused_slab_affine_num32_den16_vjp_reduce: no backend available for device ",
      row_index_i32.device());
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
fused_slab_affine_num32_den16_vjp_direct_atomic_dispatch(
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
#if defined(__APPLE__)
  if (row_index_i32.device().is_mps()) {
    return metal_fused_slab_affine_num32_den16_vjp_direct_atomic(
        row_index_i32,
        candidate_row_offsets_i32,
        candidate_depth_num_f32,
        candidate_depth_den_f16,
        sites_f32,
        site_rgba_f32,
        ray_coeff_f32,
        frame_t_f32,
        grad_rgb_f32,
        grad_alpha_f32,
        grad_depth_f32,
        config_i32,
        config_f32);
  }
#endif
  TORCH_CHECK(
      false,
      "world_foam_lane2_fused_slab_v0.fused_slab_affine_num32_den16_vjp_direct_atomic: no backend available for device ",
      row_index_i32.device());
}

torch::Tensor fused_slab_affine_num32_den16_vjp_direct_atomic_grad_only_dispatch(
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
#if defined(__APPLE__)
  if (row_index_i32.device().is_mps()) {
    return metal_fused_slab_affine_num32_den16_vjp_direct_atomic_grad_only(
        row_index_i32,
        candidate_row_offsets_i32,
        candidate_depth_num_f32,
        candidate_depth_den_f16,
        sites_f32,
        site_rgba_f32,
        ray_coeff_f32,
        frame_t_f32,
        grad_rgb_f32,
        grad_alpha_f32,
        grad_depth_f32,
        config_i32,
        config_f32);
  }
#endif
  TORCH_CHECK(
      false,
      "world_foam_lane2_fused_slab_v0.fused_slab_affine_num32_den16_vjp_direct_atomic_grad_only: no backend available for device ",
      row_index_i32.device());
}

torch::Tensor fused_slab_affine_num32_den16_vjp_direct_atomic_grad_only_ownerupdate_dispatch(
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
#if defined(__APPLE__)
  if (row_index_i32.device().is_mps()) {
    return metal_fused_slab_affine_num32_den16_vjp_direct_atomic_grad_only_ownerupdate(
        row_index_i32,
        candidate_row_offsets_i32,
        candidate_boundary_ids_i32,
        candidate_depth_num_f32,
        candidate_depth_den_f16,
        boundary_site_pairs_i32,
        sites_f32,
        site_rgba_f32,
        ray_coeff_f32,
        frame_t_f32,
        grad_rgb_f32,
        grad_alpha_f32,
        grad_depth_f32,
        config_i32,
        config_f32);
  }
#endif
  TORCH_CHECK(
      false,
      "world_foam_lane2_fused_slab_v0.fused_slab_affine_num32_den16_vjp_direct_atomic_grad_only_ownerupdate: no backend available for device ",
      row_index_i32.device());
}

torch::Tensor fused_slab_affine_num32_den16_vjp_direct_atomic_rgb_only_dispatch(
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
#if defined(__APPLE__)
  if (row_index_i32.device().is_mps()) {
    return metal_fused_slab_affine_num32_den16_vjp_direct_atomic_rgb_only(
        row_index_i32,
        candidate_row_offsets_i32,
        candidate_depth_num_f32,
        candidate_depth_den_f16,
        sites_f32,
        site_rgba_f32,
        ray_coeff_f32,
        frame_t_f32,
        grad_rgb_f32,
        config_i32,
        config_f32);
  }
#endif
  TORCH_CHECK(
      false,
      "world_foam_lane2_fused_slab_v0.fused_slab_affine_num32_den16_vjp_direct_atomic_rgb_only: no backend available for device ",
      row_index_i32.device());
}

std::tuple<torch::Tensor, torch::Tensor> fused_slab_affine_num32_den16_mse_vjp_direct_atomic_rgb_only_dispatch(
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
#if defined(__APPLE__)
  if (row_index_i32.device().is_mps()) {
    return metal_fused_slab_affine_num32_den16_mse_vjp_direct_atomic_rgb_only(
        row_index_i32,
        candidate_row_offsets_i32,
        candidate_depth_num_f32,
        candidate_depth_den_f16,
        sites_f32,
        site_rgba_f32,
        ray_coeff_f32,
        frame_t_f32,
        target_rgb_f32,
        config_i32,
        config_f32);
  }
#endif
  TORCH_CHECK(
      false,
      "world_foam_lane2_fused_slab_v0.fused_slab_affine_num32_den16_mse_vjp_direct_atomic_rgb_only: no backend available for device ",
      row_index_i32.device());
}

std::tuple<torch::Tensor, torch::Tensor> fused_slab_affine_num32_den16_mse_vjp_direct_atomic_track_rgb_only_dispatch(
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
#if defined(__APPLE__)
  if (row_index_i32.device().is_mps()) {
    return metal_fused_slab_affine_num32_den16_mse_vjp_direct_atomic_track_rgb_only(
        row_index_i32,
        candidate_row_offsets_i32,
        candidate_depth_num_f32,
        candidate_depth_den_f16,
        sites_f32,
        site_rgba_f32,
        ray_coeff_f32,
        frame_t_f32,
        target_rgb_f32,
        config_i32,
        config_f32);
  }
#endif
  TORCH_CHECK(
      false,
      "world_foam_lane2_fused_slab_v0.fused_slab_affine_num32_den16_mse_vjp_direct_atomic_track_rgb_only: no backend available for device ",
      row_index_i32.device());
}

std::tuple<torch::Tensor, torch::Tensor> fused_slab_affine_coeff16_mse_vjp_direct_atomic_rgb_only_dispatch(
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
#if defined(__APPLE__)
  if (row_index_i32.device().is_mps()) {
    return metal_fused_slab_affine_coeff16_mse_vjp_direct_atomic_rgb_only(
        row_index_i32,
        candidate_row_offsets_i32,
        candidate_depth_coeff_f16,
        sites_f32,
        site_rgba_f32,
        ray_coeff_f32,
        frame_t_f32,
        target_rgb_f32,
        config_i32,
        config_f32);
  }
#endif
  TORCH_CHECK(
      false,
      "world_foam_lane2_fused_slab_v0.fused_slab_affine_coeff16_mse_vjp_direct_atomic_rgb_only: no backend available for device ",
      row_index_i32.device());
}

std::tuple<torch::Tensor, torch::Tensor> fused_slab_affine_coeff16_cap224_mse_vjp_direct_atomic_rgb_only_dispatch(
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
#if defined(__APPLE__)
  if (row_index_i32.device().is_mps()) {
    return metal_fused_slab_affine_coeff16_cap224_mse_vjp_direct_atomic_rgb_only(
        row_index_i32,
        candidate_row_offsets_i32,
        candidate_depth_coeff_f16,
        sites_f32,
        site_rgba_f32,
        ray_coeff_f32,
        frame_t_f32,
        target_rgb_f32,
        config_i32,
        config_f32);
  }
#endif
  TORCH_CHECK(
      false,
      "world_foam_lane2_fused_slab_v0.fused_slab_affine_coeff16_cap224_mse_vjp_direct_atomic_rgb_only: no backend available for device ",
      row_index_i32.device());
}

std::tuple<torch::Tensor, torch::Tensor> fused_slab_affine_coeff16_densitymask_mse_vjp_direct_atomic_rgb_only_dispatch(
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
#if defined(__APPLE__)
  if (row_index_i32.device().is_mps()) {
    return metal_fused_slab_affine_coeff16_densitymask_mse_vjp_direct_atomic_rgb_only(
        row_index_i32,
        candidate_row_offsets_i32,
        candidate_depth_coeff_f16,
        sites_f32,
        site_rgba_f32,
        ray_coeff_f32,
        frame_t_f32,
        target_rgb_f32,
        config_i32,
        config_f32);
  }
#endif
  TORCH_CHECK(
      false,
      "world_foam_lane2_fused_slab_v0.fused_slab_affine_coeff16_densitymask_mse_vjp_direct_atomic_rgb_only: no backend available for device ",
      row_index_i32.device());
}

std::tuple<torch::Tensor, torch::Tensor>
fused_slab_affine_coeff16_mse_vjp_direct_atomic_sample_reduce_rgb_only_dispatch(
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
#if defined(__APPLE__)
  if (row_index_i32.device().is_mps()) {
    return metal_fused_slab_affine_coeff16_mse_vjp_direct_atomic_sample_reduce_rgb_only(
        row_index_i32,
        candidate_row_offsets_i32,
        candidate_depth_coeff_f16,
        sites_f32,
        site_rgba_f32,
        ray_coeff_f32,
        frame_t_f32,
        target_rgb_f32,
        config_i32,
        config_f32);
  }
#endif
  TORCH_CHECK(
      false,
      "world_foam_lane2_fused_slab_v0.fused_slab_affine_coeff16_mse_vjp_direct_atomic_sample_reduce_rgb_only: no backend available for device ",
      row_index_i32.device());
}

std::tuple<torch::Tensor, torch::Tensor>
fused_slab_affine_coeff16_sortnet_mse_vjp_direct_atomic_rgb_only_dispatch(
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
#if defined(__APPLE__)
  if (row_index_i32.device().is_mps()) {
    return metal_fused_slab_affine_coeff16_sortnet_mse_vjp_direct_atomic_rgb_only(
        row_index_i32,
        candidate_row_offsets_i32,
        candidate_depth_coeff_f16,
        sites_f32,
        site_rgba_f32,
        ray_coeff_f32,
        frame_t_f32,
        target_rgb_f32,
        config_i32,
        config_f32);
  }
#endif
  TORCH_CHECK(
      false,
      "world_foam_lane2_fused_slab_v0.fused_slab_affine_coeff16_sortnet_mse_vjp_direct_atomic_rgb_only: no backend available for device ",
      row_index_i32.device());
}

std::tuple<torch::Tensor, torch::Tensor>
fused_slab_affine_coeff16_mse_vjp_direct_atomic_sitecache_rgb_only_dispatch(
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
#if defined(__APPLE__)
  if (row_index_i32.device().is_mps()) {
    return metal_fused_slab_affine_coeff16_mse_vjp_direct_atomic_sitecache_rgb_only(
        row_index_i32,
        candidate_row_offsets_i32,
        candidate_depth_coeff_f16,
        sites_f32,
        site_rgba_f32,
        ray_coeff_f32,
        frame_t_f32,
        target_rgb_f32,
        config_i32,
        config_f32);
  }
#endif
  TORCH_CHECK(
      false,
      "world_foam_lane2_fused_slab_v0.fused_slab_affine_coeff16_mse_vjp_direct_atomic_sitecache_rgb_only: no backend available for device ",
      row_index_i32.device());
}

std::tuple<torch::Tensor, torch::Tensor>
fused_slab_affine_coeff16_mse_vjp_direct_atomic_framegroup16_cached_rgb_only_dispatch(
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
#if defined(__APPLE__)
  if (row_index_i32.device().is_mps()) {
    return metal_fused_slab_affine_coeff16_mse_vjp_direct_atomic_framegroup16_cached_rgb_only(
        row_index_i32,
        candidate_row_offsets_i32,
        candidate_depth_coeff_f16,
        sites_f32,
        site_rgba_f32,
        ray_coeff_f32,
        frame_t_f32,
        target_rgb_f32,
        config_i32,
        config_f32);
  }
#endif
  TORCH_CHECK(
      false,
      "world_foam_lane2_fused_slab_v0.fused_slab_affine_coeff16_mse_vjp_direct_atomic_framegroup16_cached_rgb_only: no backend available for device ",
      row_index_i32.device());
}

std::tuple<torch::Tensor, torch::Tensor>
fused_slab_affine_coeff16_ownerupdate_mse_vjp_direct_atomic_rgb_only_dispatch(
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
#if defined(__APPLE__)
  if (row_index_i32.device().is_mps()) {
    return metal_fused_slab_affine_coeff16_ownerupdate_mse_vjp_direct_atomic_rgb_only(
        row_index_i32,
        candidate_row_offsets_i32,
        candidate_boundary_ids_i32,
        candidate_depth_coeff_f16,
        boundary_site_pairs_i32,
        sites_f32,
        site_rgba_f32,
        ray_coeff_f32,
        frame_t_f32,
        target_rgb_f32,
        config_i32,
        config_f32);
  }
#endif
  TORCH_CHECK(
      false,
      "world_foam_lane2_fused_slab_v0.fused_slab_affine_coeff16_ownerupdate_mse_vjp_direct_atomic_rgb_only: no backend available for device ",
      row_index_i32.device());
}

std::tuple<torch::Tensor, torch::Tensor>
fused_slab_affine_coeff16_ownerupdate_i16_mse_vjp_direct_atomic_rgb_only_dispatch(
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
#if defined(__APPLE__)
  if (row_index_i32.device().is_mps()) {
    return metal_fused_slab_affine_coeff16_ownerupdate_i16_mse_vjp_direct_atomic_rgb_only(
        row_index_i32,
        candidate_row_offsets_i32,
        candidate_boundary_ids_i16,
        candidate_depth_coeff_f16,
        boundary_site_pairs_i16,
        sites_f32,
        site_rgba_f32,
        ray_coeff_f32,
        frame_t_f32,
        target_rgb_f32,
        config_i32,
        config_f32);
  }
#endif
  TORCH_CHECK(
      false,
      "world_foam_lane2_fused_slab_v0.fused_slab_affine_coeff16_ownerupdate_i16_mse_vjp_direct_atomic_rgb_only: no backend available for device ",
      row_index_i32.device());
}

std::tuple<torch::Tensor, torch::Tensor>
fused_slab_affine_coeff16_ownerkeep_i16_mse_vjp_direct_atomic_rgb_only_dispatch(
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
#if defined(__APPLE__)
  if (row_index_i32.device().is_mps()) {
    return metal_fused_slab_affine_coeff16_ownerkeep_i16_mse_vjp_direct_atomic_rgb_only(
        row_index_i32,
        candidate_row_offsets_i32,
        candidate_boundary_ids_i16,
        candidate_depth_coeff_f16,
        boundary_site_pairs_i16,
        sites_f32,
        site_rgba_f32,
        ray_coeff_f32,
        frame_t_f32,
        target_rgb_f32,
        config_i32,
        config_f32);
  }
#endif
  TORCH_CHECK(
      false,
      "world_foam_lane2_fused_slab_v0.fused_slab_affine_coeff16_ownerkeep_i16_mse_vjp_direct_atomic_rgb_only: no backend available for device ",
      row_index_i32.device());
}

std::tuple<torch::Tensor, torch::Tensor>
fused_slab_affine_coeff16_ownerkeep_mse_vjp_direct_atomic_rgb_only_dispatch(
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
#if defined(__APPLE__)
  if (row_index_i32.device().is_mps()) {
    return metal_fused_slab_affine_coeff16_ownerkeep_mse_vjp_direct_atomic_rgb_only(
        row_index_i32,
        candidate_row_offsets_i32,
        candidate_boundary_ids_i32,
        candidate_depth_coeff_f16,
        boundary_site_pairs_i32,
        sites_f32,
        site_rgba_f32,
        ray_coeff_f32,
        frame_t_f32,
        target_rgb_f32,
        config_i32,
        config_f32);
  }
#endif
  TORCH_CHECK(
      false,
      "world_foam_lane2_fused_slab_v0.fused_slab_affine_coeff16_ownerkeep_mse_vjp_direct_atomic_rgb_only: no backend available for device ",
      row_index_i32.device());
}

std::tuple<torch::Tensor, torch::Tensor> fused_slab_affine_coeff16_mse_vjp_direct_atomic_track_rgb_only_dispatch(
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
#if defined(__APPLE__)
  if (row_index_i32.device().is_mps()) {
    return metal_fused_slab_affine_coeff16_mse_vjp_direct_atomic_track_rgb_only(
        row_index_i32,
        candidate_row_offsets_i32,
        candidate_depth_coeff_f16,
        sites_f32,
        site_rgba_f32,
        ray_coeff_f32,
        frame_t_f32,
        target_rgb_f32,
        config_i32,
        config_f32);
  }
#endif
  TORCH_CHECK(
      false,
      "world_foam_lane2_fused_slab_v0.fused_slab_affine_coeff16_mse_vjp_direct_atomic_track_rgb_only: no backend available for device ",
      row_index_i32.device());
}

torch::Tensor fused_slab_affine_num32_den16_vjp_direct_atomic_track_dispatch(
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
#if defined(__APPLE__)
  if (row_index_i32.device().is_mps()) {
    return metal_fused_slab_affine_num32_den16_vjp_direct_atomic_track(
        row_index_i32,
        candidate_row_offsets_i32,
        candidate_depth_num_f32,
        candidate_depth_den_f16,
        sites_f32,
        site_rgba_f32,
        ray_coeff_f32,
        frame_t_f32,
        grad_rgb_f32,
        grad_alpha_f32,
        grad_depth_f32,
        config_i32,
        config_f32);
  }
#endif
  TORCH_CHECK(
      false,
      "world_foam_lane2_fused_slab_v0.fused_slab_affine_num32_den16_vjp_direct_atomic_track: no backend available for device ",
      row_index_i32.device());
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> segment_tape_rgba_depth_replay_dispatch(
    const torch::Tensor& segment_offsets_i32,
    const torch::Tensor& segment_owner_i32,
    const torch::Tensor& segment_length_f32,
    const torch::Tensor& segment_mid_f32,
    const torch::Tensor& site_rgba_f32,
    const torch::Tensor& config_i32,
    const torch::Tensor& config_f32) {
#if defined(__APPLE__)
  if (segment_offsets_i32.device().is_mps()) {
    return metal_segment_tape_rgba_depth_replay(
        segment_offsets_i32,
        segment_owner_i32,
        segment_length_f32,
        segment_mid_f32,
        site_rgba_f32,
        config_i32,
        config_f32);
  }
#endif
  TORCH_CHECK(
      false,
      "world_foam_lane2_fused_slab_v0.segment_tape_rgba_depth_replay: no backend available for device ",
      segment_offsets_i32.device());
}

torch::Tensor segment_tape_vjp_direct_atomic_grad_only_dispatch(
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
#if defined(__APPLE__)
  if (segment_offsets_i32.device().is_mps()) {
    return metal_segment_tape_vjp_direct_atomic_grad_only(
        segment_offsets_i32,
        segment_owner_i32,
        segment_length_f32,
        segment_mid_f32,
        site_rgba_f32,
        grad_rgb_f32,
        grad_alpha_f32,
        grad_depth_f32,
        config_i32,
        config_f32);
  }
#endif
  TORCH_CHECK(
      false,
      "world_foam_lane2_fused_slab_v0.segment_tape_vjp_direct_atomic_grad_only: no backend available for device ",
	      segment_offsets_i32.device());
}

std::tuple<torch::Tensor, torch::Tensor> segment_tape_mse_vjp_direct_atomic_rgb_only_dispatch(
    const torch::Tensor& segment_offsets_i32,
    const torch::Tensor& segment_owner_i32,
    const torch::Tensor& segment_length_f32,
    const torch::Tensor& site_rgba_f32,
    const torch::Tensor& target_rgb_f32,
    const torch::Tensor& config_i32,
    const torch::Tensor& config_f32) {
#if defined(__APPLE__)
  if (segment_offsets_i32.device().is_mps()) {
    return metal_segment_tape_mse_vjp_direct_atomic_rgb_only(
        segment_offsets_i32,
        segment_owner_i32,
        segment_length_f32,
        site_rgba_f32,
        target_rgb_f32,
        config_i32,
        config_f32);
  }
#endif
  TORCH_CHECK(
      false,
      "world_foam_lane2_fused_slab_v0.segment_tape_mse_vjp_direct_atomic_rgb_only: no backend available for device ",
      segment_offsets_i32.device());
}

torch::Tensor segment_tape_vjp_direct_atomic_track_dispatch(
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
#if defined(__APPLE__)
  if (segment_offsets_i32.device().is_mps()) {
    return metal_segment_tape_vjp_direct_atomic_track(
        segment_offsets_i32,
        segment_owner_i32,
        segment_length_f32,
        segment_mid_f32,
        site_rgba_f32,
        grad_rgb_f32,
        grad_alpha_f32,
        grad_depth_f32,
        config_i32,
        config_f32);
  }
#endif
  TORCH_CHECK(
      false,
      "world_foam_lane2_fused_slab_v0.segment_tape_vjp_direct_atomic_track: no backend available for device ",
      segment_offsets_i32.device());
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> endpoint_run_rgba_depth_replay_dispatch(
    const torch::Tensor& run_offsets_i32,
    const torch::Tensor& run_owner_i32,
    const torch::Tensor& run_start_f32,
    const torch::Tensor& run_end_f32,
    const torch::Tensor& site_rgba_f32,
    const torch::Tensor& config_i32,
    const torch::Tensor& config_f32) {
#if defined(__APPLE__)
  if (run_offsets_i32.device().is_mps()) {
    return metal_endpoint_run_rgba_depth_replay(
        run_offsets_i32,
        run_owner_i32,
        run_start_f32,
        run_end_f32,
        site_rgba_f32,
        config_i32,
        config_f32);
  }
#endif
  TORCH_CHECK(
      false,
      "world_foam_lane2_fused_slab_v0.endpoint_run_rgba_depth_replay: no backend available for device ",
      run_offsets_i32.device());
}

torch::Tensor endpoint_run_vjp_direct_atomic_grad_only_dispatch(
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
#if defined(__APPLE__)
  if (run_offsets_i32.device().is_mps()) {
    return metal_endpoint_run_vjp_direct_atomic_grad_only(
        run_offsets_i32,
        run_owner_i32,
        run_start_f32,
        run_end_f32,
        site_rgba_f32,
        grad_rgb_f32,
        grad_alpha_f32,
        grad_depth_f32,
        config_i32,
        config_f32);
  }
#endif
  TORCH_CHECK(
      false,
      "world_foam_lane2_fused_slab_v0.endpoint_run_vjp_direct_atomic_grad_only: no backend available for device ",
      run_offsets_i32.device());
}

std::tuple<torch::Tensor, torch::Tensor> endpoint_run_mse_vjp_direct_atomic_rgb_only_dispatch(
    const torch::Tensor& run_offsets_i32,
    const torch::Tensor& run_owner_i32,
    const torch::Tensor& run_start_f32,
    const torch::Tensor& run_end_f32,
    const torch::Tensor& site_rgba_f32,
    const torch::Tensor& target_rgb_f32,
    const torch::Tensor& config_i32,
    const torch::Tensor& config_f32) {
#if defined(__APPLE__)
  if (run_offsets_i32.device().is_mps()) {
    return metal_endpoint_run_mse_vjp_direct_atomic_rgb_only(
        run_offsets_i32,
        run_owner_i32,
        run_start_f32,
        run_end_f32,
        site_rgba_f32,
        target_rgb_f32,
        config_i32,
        config_f32);
  }
#endif
  TORCH_CHECK(
      false,
      "world_foam_lane2_fused_slab_v0.endpoint_run_mse_vjp_direct_atomic_rgb_only: no backend available for device ",
      run_offsets_i32.device());
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> endpoint_delta_replace_rgba_depth_replay_dispatch(
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
#if defined(__APPLE__)
  if (base_offsets_i32.device().is_mps()) {
    return metal_endpoint_delta_replace_rgba_depth_replay(
        base_offsets_i32,
        base_owner_i32,
        base_start_f32,
        base_end_f32,
        track_change_offsets_i32,
        change_frame_i32,
        change_offsets_i32,
        change_owner_i32,
        change_start_f32,
        change_end_f32,
        site_rgba_f32,
        config_i32,
        config_f32);
  }
#endif
  TORCH_CHECK(
      false,
      "world_foam_lane2_fused_slab_v0.endpoint_delta_replace_rgba_depth_replay: no backend available for device ",
      base_offsets_i32.device());
}

torch::Tensor endpoint_delta_replace_vjp_direct_atomic_grad_only_dispatch(
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
#if defined(__APPLE__)
  if (base_offsets_i32.device().is_mps()) {
    return metal_endpoint_delta_replace_vjp_direct_atomic_grad_only(
        base_offsets_i32,
        base_owner_i32,
        base_start_f32,
        base_end_f32,
        track_change_offsets_i32,
        change_frame_i32,
        change_offsets_i32,
        change_owner_i32,
        change_start_f32,
        change_end_f32,
        site_rgba_f32,
        grad_rgb_f32,
        grad_alpha_f32,
        grad_depth_f32,
        config_i32,
        config_f32);
  }
#endif
  TORCH_CHECK(
      false,
      "world_foam_lane2_fused_slab_v0.endpoint_delta_replace_vjp_direct_atomic_grad_only: no backend available for device ",
      base_offsets_i32.device());
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> endpoint_record_delta_replace_rgba_depth_replay_dispatch(
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
#if defined(__APPLE__)
  if (boundary_f32.device().is_mps()) {
    return metal_endpoint_record_delta_replace_rgba_depth_replay(
        boundary_f32,
        rays_f32,
        frame_t_f32,
        base_offsets_i32,
        base_owner_i32,
        base_left_i32,
        base_right_i32,
        track_change_offsets_i32,
        change_frame_i32,
        change_offsets_i32,
        change_owner_i32,
        change_left_i32,
        change_right_i32,
        site_rgba_f32,
        config_i32,
        config_f32);
  }
#endif
  TORCH_CHECK(
      false,
      "world_foam_lane2_fused_slab_v0.endpoint_record_delta_replace_rgba_depth_replay: no backend available for device ",
      boundary_f32.device());
}

torch::Tensor endpoint_record_delta_replace_vjp_direct_atomic_grad_only_dispatch(
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
#if defined(__APPLE__)
  if (boundary_f32.device().is_mps()) {
    return metal_endpoint_record_delta_replace_vjp_direct_atomic_grad_only(
        boundary_f32,
        rays_f32,
        frame_t_f32,
        base_offsets_i32,
        base_owner_i32,
        base_left_i32,
        base_right_i32,
        track_change_offsets_i32,
        change_frame_i32,
        change_offsets_i32,
        change_owner_i32,
        change_left_i32,
        change_right_i32,
        site_rgba_f32,
        grad_rgb_f32,
        grad_alpha_f32,
        grad_depth_f32,
        config_i32,
        config_f32);
  }
#endif
  TORCH_CHECK(
      false,
      "world_foam_lane2_fused_slab_v0.endpoint_record_delta_replace_vjp_direct_atomic_grad_only: no backend available for device ",
      boundary_f32.device());
}

std::tuple<torch::Tensor, torch::Tensor> endpoint_record_delta_replace_coeff16_mse_vjp_direct_atomic_rgb_only_dispatch(
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
#if defined(__APPLE__)
  if (coeff_f16.device().is_mps()) {
    return metal_endpoint_record_delta_replace_coeff16_mse_vjp_direct_atomic_rgb_only(
        coeff_f16,
        frame_t_f32,
        base_offsets_i32,
        base_owner_i32,
        base_left_i32,
        base_right_i32,
        track_change_offsets_i32,
        change_frame_i32,
        change_offsets_i32,
        change_owner_i32,
        change_left_i32,
        change_right_i32,
        site_rgba_f32,
        target_rgb_f32,
        config_i32,
        config_f32);
  }
#endif
  TORCH_CHECK(
      false,
      "world_foam_lane2_fused_slab_v0.endpoint_record_delta_replace_coeff16_mse_vjp_direct_atomic_rgb_only: no backend available for device ",
      coeff_f16.device());
}

std::tuple<torch::Tensor, torch::Tensor> endpoint_record_delta_replace_coeff16_i16x3_mse_vjp_direct_atomic_rgb_only_dispatch(
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
#if defined(__APPLE__)
  if (coeff_f16.device().is_mps()) {
    return metal_endpoint_record_delta_replace_coeff16_i16x3_mse_vjp_direct_atomic_rgb_only(
        coeff_f16,
        frame_t_f32,
        base_offsets_i32,
        base_record_i16,
        track_change_offsets_i32,
        change_frame_i32,
        change_offsets_i32,
        change_record_i16,
        site_rgba_f32,
        target_rgb_f32,
        config_i32,
        config_f32);
  }
#endif
  TORCH_CHECK(
      false,
      "world_foam_lane2_fused_slab_v0.endpoint_record_delta_replace_coeff16_i16x3_mse_vjp_direct_atomic_rgb_only: no backend available for device ",
      coeff_f16.device());
}

std::tuple<torch::Tensor, torch::Tensor>
endpoint_record_delta_replace_coeff16_i16x3_framegroup16_mse_vjp_direct_atomic_rgb_only_dispatch(
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
#if defined(__APPLE__)
  if (coeff_f16.device().is_mps()) {
    return metal_endpoint_record_delta_replace_coeff16_i16x3_framegroup16_mse_vjp_direct_atomic_rgb_only(
        coeff_f16,
        frame_t_f32,
        base_offsets_i32,
        base_record_i16,
        track_change_offsets_i32,
        track_chunk_change_offsets_i16,
        change_frame_i32,
        change_offsets_i32,
        change_record_i16,
        site_rgba_f32,
        target_rgb_f32,
        config_i32,
        config_f32);
  }
#endif
  TORCH_CHECK(
      false,
      "world_foam_lane2_fused_slab_v0.endpoint_record_delta_replace_coeff16_i16x3_framegroup16_mse_vjp_direct_atomic_rgb_only: no backend available for device ",
      coeff_f16.device());
}

std::tuple<torch::Tensor, torch::Tensor>
endpoint_record_delta_replace_coeff16_i16x3_framegroup16_ownerreduce_mse_vjp_direct_atomic_rgb_only_dispatch(
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
#if defined(__APPLE__)
  if (coeff_f16.device().is_mps()) {
    return metal_endpoint_record_delta_replace_coeff16_i16x3_framegroup16_ownerreduce_mse_vjp_direct_atomic_rgb_only(
        coeff_f16,
        frame_t_f32,
        base_offsets_i32,
        base_record_i16,
        track_change_offsets_i32,
        track_chunk_change_offsets_i16,
        track_chunk_owner_offsets_i32,
        track_chunk_owner_i16,
        change_frame_i32,
        change_offsets_i32,
        change_record_i16,
        site_rgba_f32,
        target_rgb_f32,
        config_i32,
        config_f32);
  }
#endif
  TORCH_CHECK(
      false,
      "world_foam_lane2_fused_slab_v0.endpoint_record_delta_replace_coeff16_i16x3_framegroup16_ownerreduce_mse_vjp_direct_atomic_rgb_only: no backend available for device ",
      coeff_f16.device());
}

std::tuple<torch::Tensor, torch::Tensor>
endpoint_record_delta_replace_coeff16_i16cols_framegroup16_mse_vjp_direct_atomic_rgb_only_dispatch(
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
#if defined(__APPLE__)
  if (coeff_f16.device().is_mps()) {
    return metal_endpoint_record_delta_replace_coeff16_i16cols_framegroup16_mse_vjp_direct_atomic_rgb_only(
        coeff_f16,
        frame_t_f32,
        base_offsets_i32,
        base_record_i16,
        track_change_offsets_i32,
        track_chunk_change_offsets_i16,
        change_frame_i32,
        change_offsets_i32,
        change_record_i16,
        site_rgba_f32,
        target_rgb_f32,
        config_i32,
        config_f32);
  }
#endif
  TORCH_CHECK(
      false,
      "world_foam_lane2_fused_slab_v0.endpoint_record_delta_replace_coeff16_i16cols_framegroup16_mse_vjp_direct_atomic_rgb_only: no backend available for device ",
      coeff_f16.device());
}

std::tuple<torch::Tensor, torch::Tensor>
endpoint_record_delta_replace_coeff16_i16x3_framegroup64_mse_vjp_direct_atomic_rgb_only_dispatch(
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
#if defined(__APPLE__)
  if (coeff_f16.device().is_mps()) {
    return metal_endpoint_record_delta_replace_coeff16_i16x3_framegroup64_mse_vjp_direct_atomic_rgb_only(
        coeff_f16,
        frame_t_f32,
        base_offsets_i32,
        base_record_i16,
        track_change_offsets_i32,
        track_chunk_change_offsets_i16,
        change_frame_i32,
        change_offsets_i32,
        change_record_i16,
        site_rgba_f32,
        target_rgb_f32,
        config_i32,
        config_f32);
  }
#endif
  TORCH_CHECK(
      false,
      "world_foam_lane2_fused_slab_v0.endpoint_record_delta_replace_coeff16_i16x3_framegroup64_mse_vjp_direct_atomic_rgb_only: no backend available for device ",
      coeff_f16.device());
}

std::tuple<torch::Tensor, torch::Tensor>
endpoint_record_delta_replace_coeff16_packed_mse_vjp_direct_atomic_rgb_only_dispatch(
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
#if defined(__APPLE__)
  if (coeff_f16.device().is_mps()) {
    return metal_endpoint_record_delta_replace_coeff16_packed_mse_vjp_direct_atomic_rgb_only(
        coeff_f16,
        frame_t_f32,
        base_offsets_i32,
        base_record_i32,
        track_change_offsets_i32,
        change_frame_i32,
        change_offsets_i32,
        change_record_i32,
        site_rgba_f32,
        target_rgb_f32,
        config_i32,
        config_f32);
  }
#endif
  TORCH_CHECK(
      false,
      "world_foam_lane2_fused_slab_v0.endpoint_record_delta_replace_coeff16_packed_mse_vjp_direct_atomic_rgb_only: no backend available for device ",
      coeff_f16.device());
}

std::tuple<torch::Tensor, torch::Tensor>
endpoint_record_delta_replace_coeff16_packed_framegroup16_mse_vjp_direct_atomic_rgb_only_dispatch(
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
#if defined(__APPLE__)
  if (coeff_f16.device().is_mps()) {
    return metal_endpoint_record_delta_replace_coeff16_packed_framegroup16_mse_vjp_direct_atomic_rgb_only(
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
        config_f32);
  }
#endif
  TORCH_CHECK(
      false,
      "world_foam_lane2_fused_slab_v0.endpoint_record_delta_replace_coeff16_packed_framegroup16_mse_vjp_direct_atomic_rgb_only: no backend available for device ",
      coeff_f16.device());
}

std::tuple<torch::Tensor, torch::Tensor>
endpoint_record_delta_replace_coeff16_packed_framegroup16_launch_only_mse_vjp_direct_atomic_rgb_only_dispatch(
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
#if defined(__APPLE__)
  if (coeff_f16.device().is_mps()) {
    return metal_endpoint_record_delta_replace_coeff16_packed_framegroup16_launch_only_mse_vjp_direct_atomic_rgb_only(
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
        change_record_count);
  }
#endif
  TORCH_CHECK(
      false,
      "world_foam_lane2_fused_slab_v0.endpoint_record_delta_replace_coeff16_packed_framegroup16_launch_only_mse_vjp_direct_atomic_rgb_only: no backend available for device ",
      coeff_f16.device());
}

std::tuple<torch::Tensor, torch::Tensor>
endpoint_record_delta_replace_coeff16_packed_framegroup16_unchecked_launch_only_mse_vjp_direct_atomic_rgb_only_dispatch(
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
#if defined(__APPLE__)
  if (coeff_f16.device().is_mps()) {
    return metal_endpoint_record_delta_replace_coeff16_packed_framegroup16_unchecked_launch_only_mse_vjp_direct_atomic_rgb_only(
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
        track_count,
        frame_count,
        site_count);
  }
#endif
  TORCH_CHECK(
      false,
      "world_foam_lane2_fused_slab_v0.endpoint_record_delta_replace_coeff16_packed_framegroup16_unchecked_launch_only_mse_vjp_direct_atomic_rgb_only: no backend available for device ",
      coeff_f16.device());
}

std::tuple<torch::Tensor, torch::Tensor>
endpoint_record_delta_replace_coeff16_packed_framegroup16_reduce32_launch_only_mse_vjp_direct_atomic_rgb_only_dispatch(
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
#if defined(__APPLE__)
  if (coeff_f16.device().is_mps()) {
    return metal_endpoint_record_delta_replace_coeff16_packed_framegroup16_reduce32_launch_only_mse_vjp_direct_atomic_rgb_only(
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
        change_record_count);
  }
#endif
  TORCH_CHECK(
      false,
      "world_foam_lane2_fused_slab_v0.endpoint_record_delta_replace_coeff16_packed_framegroup16_reduce32_launch_only_mse_vjp_direct_atomic_rgb_only: no backend available for device ",
      coeff_f16.device());
}

std::tuple<torch::Tensor, torch::Tensor>
endpoint_record_delta_replace_coeff16_packed_framegroup16_reduce32_unchecked_launch_only_mse_vjp_direct_atomic_rgb_only_dispatch(
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
#if defined(__APPLE__)
  if (coeff_f16.device().is_mps()) {
    return metal_endpoint_record_delta_replace_coeff16_packed_framegroup16_reduce32_unchecked_launch_only_mse_vjp_direct_atomic_rgb_only(
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
        track_count,
        frame_count,
        site_count);
  }
#endif
  TORCH_CHECK(
      false,
      "world_foam_lane2_fused_slab_v0.endpoint_record_delta_replace_coeff16_packed_framegroup16_reduce32_unchecked_launch_only_mse_vjp_direct_atomic_rgb_only: no backend available for device ",
      coeff_f16.device());
}

std::tuple<torch::Tensor, torch::Tensor>
endpoint_record_delta_replace_coeff16_packed_framegroup16_rowselect32_launch_only_mse_vjp_direct_atomic_rgb_only_dispatch(
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
#if defined(__APPLE__)
  if (coeff_f16.device().is_mps()) {
    return metal_endpoint_record_delta_replace_coeff16_packed_framegroup16_rowselect32_launch_only_mse_vjp_direct_atomic_rgb_only(
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
        change_record_count);
  }
#endif
  TORCH_CHECK(
      false,
      "world_foam_lane2_fused_slab_v0.endpoint_record_delta_replace_coeff16_packed_framegroup16_rowselect32_launch_only_mse_vjp_direct_atomic_rgb_only: no backend available for device ",
      coeff_f16.device());
}

std::tuple<torch::Tensor, torch::Tensor>
endpoint_record_delta_replace_coeff16_packed_framegroup16_rowselect32_unchecked_launch_only_mse_vjp_direct_atomic_rgb_only_dispatch(
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
#if defined(__APPLE__)
  if (coeff_f16.device().is_mps()) {
    return metal_endpoint_record_delta_replace_coeff16_packed_framegroup16_rowselect32_unchecked_launch_only_mse_vjp_direct_atomic_rgb_only(
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
        track_count,
        frame_count,
        site_count);
  }
#endif
  TORCH_CHECK(
      false,
      "world_foam_lane2_fused_slab_v0.endpoint_record_delta_replace_coeff16_packed_framegroup16_rowselect32_unchecked_launch_only_mse_vjp_direct_atomic_rgb_only: no backend available for device ",
      coeff_f16.device());
}

std::tuple<torch::Tensor, torch::Tensor>
endpoint_record_delta_replace_coeff16_packed_framegroup16_rowdesc_launch_only_mse_vjp_direct_atomic_rgb_only_dispatch(
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
#if defined(__APPLE__)
  if (coeff_f16.device().is_mps()) {
    return metal_endpoint_record_delta_replace_coeff16_packed_framegroup16_rowdesc_launch_only_mse_vjp_direct_atomic_rgb_only(
        coeff_f16,
        frame_t_f32,
        row_begin_i32,
        row_len_source_i16,
        base_record_i32,
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
        change_record_count);
  }
#endif
  TORCH_CHECK(
      false,
      "world_foam_lane2_fused_slab_v0.endpoint_record_delta_replace_coeff16_packed_framegroup16_rowdesc_launch_only_mse_vjp_direct_atomic_rgb_only: no backend available for device ",
      coeff_f16.device());
}

std::tuple<torch::Tensor, torch::Tensor>
endpoint_record_delta_replace_coeff16_packed_framegroup16_rowdesc_unchecked_launch_only_mse_vjp_direct_atomic_rgb_only_dispatch(
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
#if defined(__APPLE__)
  if (coeff_f16.device().is_mps()) {
    return metal_endpoint_record_delta_replace_coeff16_packed_framegroup16_rowdesc_unchecked_launch_only_mse_vjp_direct_atomic_rgb_only(
        coeff_f16,
        frame_t_f32,
        row_begin_i32,
        row_len_source_i16,
        base_record_i32,
        change_record_i32,
        site_rgba_f32,
        target_rgb_f32,
        config_i32,
        config_f32,
        track_count,
        frame_count,
        site_count);
  }
#endif
  TORCH_CHECK(
      false,
      "world_foam_lane2_fused_slab_v0.endpoint_record_delta_replace_coeff16_packed_framegroup16_rowdesc_unchecked_launch_only_mse_vjp_direct_atomic_rgb_only: no backend available for device ",
      coeff_f16.device());
}

std::tuple<torch::Tensor, torch::Tensor>
endpoint_record_delta_replace_coeff16_packed_framegroup16_rowdesc32_launch_only_mse_vjp_direct_atomic_rgb_only_dispatch(
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
#if defined(__APPLE__)
  if (coeff_f16.device().is_mps()) {
    return metal_endpoint_record_delta_replace_coeff16_packed_framegroup16_rowdesc32_launch_only_mse_vjp_direct_atomic_rgb_only(
        coeff_f16,
        frame_t_f32,
        row_begin_i32,
        row_len_source_i16,
        base_record_i32,
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
        change_record_count);
  }
#endif
  TORCH_CHECK(
      false,
      "world_foam_lane2_fused_slab_v0.endpoint_record_delta_replace_coeff16_packed_framegroup16_rowdesc32_launch_only_mse_vjp_direct_atomic_rgb_only: no backend available for device ",
      coeff_f16.device());
}

std::tuple<torch::Tensor, torch::Tensor>
endpoint_record_delta_replace_coeff16_packed_framegroup16_rowdesc32_unchecked_launch_only_mse_vjp_direct_atomic_rgb_only_dispatch(
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
#if defined(__APPLE__)
  if (coeff_f16.device().is_mps()) {
    return metal_endpoint_record_delta_replace_coeff16_packed_framegroup16_rowdesc32_unchecked_launch_only_mse_vjp_direct_atomic_rgb_only(
        coeff_f16,
        frame_t_f32,
        row_begin_i32,
        row_len_source_i16,
        base_record_i32,
        change_record_i32,
        site_rgba_f32,
        target_rgb_f32,
        config_i32,
        config_f32,
        track_count,
        frame_count,
        site_count);
  }
#endif
  TORCH_CHECK(
      false,
      "world_foam_lane2_fused_slab_v0.endpoint_record_delta_replace_coeff16_packed_framegroup16_rowdesc32_unchecked_launch_only_mse_vjp_direct_atomic_rgb_only: no backend available for device ",
      coeff_f16.device());
}

std::tuple<torch::Tensor, torch::Tensor>
endpoint_record_delta_replace_coeff16_packed_framegroup16_recompute_launch_only_mse_vjp_direct_atomic_rgb_only_dispatch(
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
#if defined(__APPLE__)
  if (coeff_f16.device().is_mps()) {
    return metal_endpoint_record_delta_replace_coeff16_packed_framegroup16_recompute_launch_only_mse_vjp_direct_atomic_rgb_only(
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
        change_record_count);
  }
#endif
  TORCH_CHECK(
      false,
      "world_foam_lane2_fused_slab_v0.endpoint_record_delta_replace_coeff16_packed_framegroup16_recompute_launch_only_mse_vjp_direct_atomic_rgb_only: no backend available for device ",
      coeff_f16.device());
}

std::tuple<torch::Tensor, torch::Tensor>
endpoint_record_delta_replace_coeff16_packed_framegroup16_smallrun16_launch_only_mse_vjp_direct_atomic_rgb_only_dispatch(
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
#if defined(__APPLE__)
  if (coeff_f16.device().is_mps()) {
    return metal_endpoint_record_delta_replace_coeff16_packed_framegroup16_smallrun16_launch_only_mse_vjp_direct_atomic_rgb_only(
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
        change_record_count);
  }
#endif
  TORCH_CHECK(
      false,
      "world_foam_lane2_fused_slab_v0.endpoint_record_delta_replace_coeff16_packed_framegroup16_smallrun16_launch_only_mse_vjp_direct_atomic_rgb_only: no backend available for device ",
      coeff_f16.device());
}

std::tuple<torch::Tensor, torch::Tensor>
endpoint_record_delta_replace_coeff16_packed_framegroup16_materialized_launch_only_mse_vjp_direct_atomic_rgb_only_dispatch(
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
#if defined(__APPLE__)
  if (coeff_f16.device().is_mps()) {
    return metal_endpoint_record_delta_replace_coeff16_packed_framegroup16_materialized_launch_only_mse_vjp_direct_atomic_rgb_only(
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
        change_record_count);
  }
#endif
  TORCH_CHECK(
      false,
      "world_foam_lane2_fused_slab_v0.endpoint_record_delta_replace_coeff16_packed_framegroup16_materialized_launch_only_mse_vjp_direct_atomic_rgb_only: no backend available for device ",
      coeff_f16.device());
}

std::tuple<torch::Tensor, torch::Tensor>
endpoint_record_delta_replace_coeff16_packed_framegroup16_recompute_mse_vjp_direct_atomic_rgb_only_dispatch(
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
#if defined(__APPLE__)
  if (coeff_f16.device().is_mps()) {
    return metal_endpoint_record_delta_replace_coeff16_packed_framegroup16_recompute_mse_vjp_direct_atomic_rgb_only(
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
        config_f32);
  }
#endif
  TORCH_CHECK(
      false,
      "world_foam_lane2_fused_slab_v0.endpoint_record_delta_replace_coeff16_packed_framegroup16_recompute_mse_vjp_direct_atomic_rgb_only: no backend available for device ",
      coeff_f16.device());
}

std::tuple<torch::Tensor, torch::Tensor>
endpoint_record_delta_replace_factorized_packed_framegroup16_recompute_mse_vjp_direct_atomic_rgb_only_dispatch(
    const torch::Tensor& boundary_f32,
    const torch::Tensor& track_ray_coeff_f32,
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
#if defined(__APPLE__)
  if (boundary_f32.device().is_mps()) {
    return metal_endpoint_record_delta_replace_factorized_packed_framegroup16_recompute_mse_vjp_direct_atomic_rgb_only(
        boundary_f32,
        track_ray_coeff_f32,
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
        config_f32);
  }
#endif
  TORCH_CHECK(
      false,
      "world_foam_lane2_fused_slab_v0.endpoint_record_delta_replace_factorized_packed_framegroup16_recompute_mse_vjp_direct_atomic_rgb_only: no backend available for device ",
      boundary_f32.device());
}

std::tuple<torch::Tensor, torch::Tensor>
endpoint_record_delta_replace_factorized_frameselect_recompute_mse_vjp_direct_atomic_rgb_only_dispatch(
    const torch::Tensor& boundary_f32,
    const torch::Tensor& track_ray_coeff_f32,
    const torch::Tensor& frame_t_f32,
    const torch::Tensor& base_offsets_i32,
    const torch::Tensor& base_record_i32,
    const torch::Tensor& frame_change_index_i16,
    const torch::Tensor& change_offsets_i32,
    const torch::Tensor& change_record_i32,
    const torch::Tensor& site_rgba_f32,
    const torch::Tensor& target_rgb_f32,
    const torch::Tensor& config_i32,
    const torch::Tensor& config_f32) {
#if defined(__APPLE__)
  if (boundary_f32.device().is_mps()) {
    return metal_endpoint_record_delta_replace_factorized_frameselect_recompute_mse_vjp_direct_atomic_rgb_only(
        boundary_f32,
        track_ray_coeff_f32,
        frame_t_f32,
        base_offsets_i32,
        base_record_i32,
        frame_change_index_i16,
        change_offsets_i32,
        change_record_i32,
        site_rgba_f32,
        target_rgb_f32,
        config_i32,
        config_f32);
  }
#endif
  TORCH_CHECK(
      false,
      "world_foam_lane2_fused_slab_v0.endpoint_record_delta_replace_factorized_frameselect_recompute_mse_vjp_direct_atomic_rgb_only: no backend available for device ",
      boundary_f32.device());
}

std::tuple<torch::Tensor, torch::Tensor>
endpoint_record_delta_replace_factorized_framebitmask_recompute_mse_vjp_direct_atomic_rgb_only_dispatch(
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
#if defined(__APPLE__)
  if (boundary_f32.device().is_mps()) {
    return metal_endpoint_record_delta_replace_factorized_framebitmask_recompute_mse_vjp_direct_atomic_rgb_only(
        boundary_f32,
        track_ray_coeff_f32,
        frame_t_f32,
        base_offsets_i32,
        base_record_i32,
        track_change_offsets_i32,
        track_frame_mask_i32,
        change_offsets_i32,
        change_record_i32,
        site_rgba_f32,
        target_rgb_f32,
        config_i32,
        config_f32);
  }
#endif
  TORCH_CHECK(
      false,
      "world_foam_lane2_fused_slab_v0.endpoint_record_delta_replace_factorized_framebitmask_recompute_mse_vjp_direct_atomic_rgb_only: no backend available for device ",
      boundary_f32.device());
}

std::tuple<torch::Tensor, torch::Tensor>
endpoint_record_delta_replace_coeff16_packed_framegroup16_smallrun16_mse_vjp_direct_atomic_rgb_only_dispatch(
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
#if defined(__APPLE__)
  if (coeff_f16.device().is_mps()) {
    return metal_endpoint_record_delta_replace_coeff16_packed_framegroup16_smallrun16_mse_vjp_direct_atomic_rgb_only(
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
        config_f32);
  }
#endif
  TORCH_CHECK(
      false,
      "world_foam_lane2_fused_slab_v0.endpoint_record_delta_replace_coeff16_packed_framegroup16_smallrun16_mse_vjp_direct_atomic_rgb_only: no backend available for device ",
      coeff_f16.device());
}

std::tuple<torch::Tensor, torch::Tensor>
endpoint_record_delta_replace_coeff16_packed_framegroup16_materialized_mse_vjp_direct_atomic_rgb_only_dispatch(
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
#if defined(__APPLE__)
  if (coeff_f16.device().is_mps()) {
    return metal_endpoint_record_delta_replace_coeff16_packed_framegroup16_materialized_mse_vjp_direct_atomic_rgb_only(
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
        config_f32);
  }
#endif
  TORCH_CHECK(
      false,
      "world_foam_lane2_fused_slab_v0.endpoint_record_delta_replace_coeff16_packed_framegroup16_materialized_mse_vjp_direct_atomic_rgb_only: no backend available for device ",
      coeff_f16.device());
}

std::tuple<torch::Tensor, torch::Tensor>
endpoint_record_delta_replace_coeff16_i16x4_framegroup16_mse_vjp_direct_atomic_rgb_only_dispatch(
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
#if defined(__APPLE__)
  if (coeff_f16.device().is_mps()) {
    return metal_endpoint_record_delta_replace_coeff16_i16x4_framegroup16_mse_vjp_direct_atomic_rgb_only(
        coeff_f16,
        frame_t_f32,
        base_offsets_i32,
        base_record_i16,
        track_change_offsets_i32,
        track_chunk_change_offsets_i16,
        change_frame_i32,
        change_offsets_i32,
        change_record_i16,
        site_rgba_f32,
        target_rgb_f32,
        config_i32,
        config_f32);
  }
#endif
  TORCH_CHECK(
      false,
      "world_foam_lane2_fused_slab_v0.endpoint_record_delta_replace_coeff16_i16x4_framegroup16_mse_vjp_direct_atomic_rgb_only: no backend available for device ",
      coeff_f16.device());
}

std::tuple<torch::Tensor, torch::Tensor>
endpoint_record_delta_replace_coeff16_i16x3_framegroup16_materialized_mse_vjp_direct_atomic_rgb_only_dispatch(
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
#if defined(__APPLE__)
  if (coeff_f16.device().is_mps()) {
    return metal_endpoint_record_delta_replace_coeff16_i16x3_framegroup16_materialized_mse_vjp_direct_atomic_rgb_only(
        coeff_f16,
        frame_t_f32,
        base_offsets_i32,
        base_record_i16,
        track_change_offsets_i32,
        track_chunk_change_offsets_i16,
        change_frame_i32,
        change_offsets_i32,
        change_record_i16,
        site_rgba_f32,
        target_rgb_f32,
        config_i32,
        config_f32);
  }
#endif
  TORCH_CHECK(
      false,
      "world_foam_lane2_fused_slab_v0.endpoint_record_delta_replace_coeff16_i16x3_framegroup16_materialized_mse_vjp_direct_atomic_rgb_only: no backend available for device ",
      coeff_f16.device());
}

std::tuple<torch::Tensor, torch::Tensor> endpoint_record_delta_replace_coeff16_i16x4_mse_vjp_direct_atomic_rgb_only_dispatch(
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
#if defined(__APPLE__)
  if (coeff_f16.device().is_mps()) {
    return metal_endpoint_record_delta_replace_coeff16_i16x4_mse_vjp_direct_atomic_rgb_only(
        coeff_f16,
        frame_t_f32,
        base_offsets_i32,
        base_record_i16,
        track_change_offsets_i32,
        change_frame_i32,
        change_offsets_i32,
        change_record_i16,
        site_rgba_f32,
        target_rgb_f32,
        config_i32,
        config_f32);
  }
#endif
  TORCH_CHECK(
      false,
      "world_foam_lane2_fused_slab_v0.endpoint_record_delta_replace_coeff16_i16x4_mse_vjp_direct_atomic_rgb_only: no backend available for device ",
      coeff_f16.device());
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> endpoint_record_edit_rgba_depth_replay_dispatch(
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
#if defined(__APPLE__)
  if (boundary_f32.device().is_mps()) {
    return metal_endpoint_record_edit_rgba_depth_replay(
        boundary_f32,
        rays_f32,
        frame_t_f32,
        base_offsets_i32,
        base_owner_i32,
        base_left_i32,
        base_right_i32,
        track_change_offsets_i32,
        change_frame_i32,
        op_offsets_i32,
        op_type_i32,
        op_pos_i32,
        op_owner_i32,
        op_left_i32,
        op_right_i32,
        site_rgba_f32,
        config_i32,
        config_f32);
  }
#endif
  TORCH_CHECK(
      false,
      "world_foam_lane2_fused_slab_v0.endpoint_record_edit_rgba_depth_replay: no backend available for device ",
      boundary_f32.device());
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> endpoint_record_edit_block4_rgba_depth_replay_dispatch(
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
#if defined(__APPLE__)
  if (boundary_f32.device().is_mps()) {
    return metal_endpoint_record_edit_block4_rgba_depth_replay(
        boundary_f32,
        rays_f32,
        frame_t_f32,
        anchor_offsets_i32,
        anchor_owner_i32,
        anchor_left_i32,
        anchor_right_i32,
        track_block_change_offsets_i32,
        change_frame_i32,
        op_offsets_i32,
        op_type_i32,
        op_pos_i32,
        op_owner_i32,
        op_left_i32,
        op_right_i32,
        site_rgba_f32,
        config_i32,
        config_f32);
  }
#endif
  TORCH_CHECK(
      false,
      "world_foam_lane2_fused_slab_v0.endpoint_record_edit_block4_rgba_depth_replay: no backend available for device ",
      boundary_f32.device());
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> endpoint_record_edit_block_coeff_rgba_depth_replay_dispatch(
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
#if defined(__APPLE__)
  if (coeff_f32.device().is_mps()) {
    return metal_endpoint_record_edit_block_coeff_rgba_depth_replay(
        coeff_f32,
        frame_t_f32,
        anchor_offsets_i32,
        anchor_owner_i32,
        anchor_left_i32,
        anchor_right_i32,
        track_block_change_offsets_i32,
        change_frame_i32,
        op_offsets_i32,
        op_type_i32,
        op_pos_i32,
        op_owner_i32,
        op_left_i32,
        op_right_i32,
        site_rgba_f32,
        config_i32,
        config_f32);
  }
#endif
  TORCH_CHECK(
      false,
      "world_foam_lane2_fused_slab_v0.endpoint_record_edit_block_coeff_rgba_depth_replay: no backend available for device ",
      coeff_f32.device());
}

torch::Tensor endpoint_record_edit_block_coeff_rgb_replay_dispatch(
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
#if defined(__APPLE__)
  if (coeff_f32.device().is_mps()) {
    return metal_endpoint_record_edit_block_coeff_rgb_replay(
        coeff_f32,
        frame_t_f32,
        anchor_offsets_i32,
        anchor_owner_i32,
        anchor_left_i32,
        anchor_right_i32,
        track_block_change_offsets_i32,
        change_frame_i32,
        op_offsets_i32,
        op_type_i32,
        op_pos_i32,
        op_owner_i32,
        op_left_i32,
        op_right_i32,
        site_rgba_f32,
        config_i32,
        config_f32);
  }
#endif
  TORCH_CHECK(
      false,
      "world_foam_lane2_fused_slab_v0.endpoint_record_edit_block_coeff_rgb_replay: no backend available for device ",
      coeff_f32.device());
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> endpoint_record_edit_block_coeff16_rgba_depth_replay_dispatch(
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
#if defined(__APPLE__)
  if (coeff_f16.device().is_mps()) {
    return metal_endpoint_record_edit_block_coeff16_rgba_depth_replay(
        coeff_f16,
        frame_t_f32,
        anchor_offsets_i32,
        anchor_owner_i32,
        anchor_left_i32,
        anchor_right_i32,
        track_block_change_offsets_i32,
        change_frame_i32,
        op_offsets_i32,
        op_type_i32,
        op_pos_i32,
        op_owner_i32,
        op_left_i32,
        op_right_i32,
        site_rgba_f32,
        config_i32,
        config_f32);
  }
#endif
  TORCH_CHECK(
      false,
      "world_foam_lane2_fused_slab_v0.endpoint_record_edit_block_coeff16_rgba_depth_replay: no backend available for device ",
      coeff_f16.device());
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> endpoint_record_edit_rgba_depth_replay_trackloop_dispatch(
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
#if defined(__APPLE__)
  if (boundary_f32.device().is_mps()) {
    return metal_endpoint_record_edit_rgba_depth_replay_trackloop(
        boundary_f32,
        rays_f32,
        frame_t_f32,
        base_offsets_i32,
        base_owner_i32,
        base_left_i32,
        base_right_i32,
        track_change_offsets_i32,
        change_frame_i32,
        op_offsets_i32,
        op_type_i32,
        op_pos_i32,
        op_owner_i32,
        op_left_i32,
        op_right_i32,
        site_rgba_f32,
        config_i32,
        config_f32);
  }
#endif
  TORCH_CHECK(
      false,
      "world_foam_lane2_fused_slab_v0.endpoint_record_edit_rgba_depth_replay_trackloop: no backend available for device ",
      boundary_f32.device());
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> endpoint_record_edit_rgba_depth_replay_framegroup16_dispatch(
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
#if defined(__APPLE__)
  if (boundary_f32.device().is_mps()) {
    return metal_endpoint_record_edit_rgba_depth_replay_framegroup16(
        boundary_f32,
        rays_f32,
        frame_t_f32,
        base_offsets_i32,
        base_owner_i32,
        base_left_i32,
        base_right_i32,
        track_change_offsets_i32,
        change_frame_i32,
        op_offsets_i32,
        op_type_i32,
        op_pos_i32,
        op_owner_i32,
        op_left_i32,
        op_right_i32,
        site_rgba_f32,
        config_i32,
        config_f32);
  }
#endif
  TORCH_CHECK(
      false,
      "world_foam_lane2_fused_slab_v0.endpoint_record_edit_rgba_depth_replay_framegroup16: no backend available for device ",
      boundary_f32.device());
}

torch::Tensor endpoint_record_edit_vjp_direct_atomic_grad_only_dispatch(
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
#if defined(__APPLE__)
  if (boundary_f32.device().is_mps()) {
    return metal_endpoint_record_edit_vjp_direct_atomic_grad_only(
        boundary_f32,
        rays_f32,
        frame_t_f32,
        base_offsets_i32,
        base_owner_i32,
        base_left_i32,
        base_right_i32,
        track_change_offsets_i32,
        change_frame_i32,
        op_offsets_i32,
        op_type_i32,
        op_pos_i32,
        op_owner_i32,
        op_left_i32,
        op_right_i32,
        site_rgba_f32,
        grad_rgb_f32,
        grad_alpha_f32,
        grad_depth_f32,
        config_i32,
        config_f32);
  }
#endif
  TORCH_CHECK(
      false,
      "world_foam_lane2_fused_slab_v0.endpoint_record_edit_vjp_direct_atomic_grad_only: no backend available for device ",
      boundary_f32.device());
}

torch::Tensor endpoint_record_edit_vjp_direct_atomic_rgb_only_dispatch(
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
#if defined(__APPLE__)
  if (boundary_f32.device().is_mps()) {
    return metal_endpoint_record_edit_vjp_direct_atomic_rgb_only(
        boundary_f32,
        rays_f32,
        frame_t_f32,
        base_offsets_i32,
        base_owner_i32,
        base_left_i32,
        base_right_i32,
        track_change_offsets_i32,
        change_frame_i32,
        op_offsets_i32,
        op_type_i32,
        op_pos_i32,
        op_owner_i32,
        op_left_i32,
        op_right_i32,
        site_rgba_f32,
        grad_rgb_f32,
        config_i32,
        config_f32);
  }
#endif
  TORCH_CHECK(
      false,
      "world_foam_lane2_fused_slab_v0.endpoint_record_edit_vjp_direct_atomic_rgb_only: no backend available for device ",
      boundary_f32.device());
}

std::tuple<torch::Tensor, torch::Tensor> endpoint_record_edit_mse_vjp_direct_atomic_rgb_only_dispatch(
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
#if defined(__APPLE__)
  if (boundary_f32.device().is_mps()) {
    return metal_endpoint_record_edit_mse_vjp_direct_atomic_rgb_only(
        boundary_f32,
        rays_f32,
        frame_t_f32,
        base_offsets_i32,
        base_owner_i32,
        base_left_i32,
        base_right_i32,
        track_change_offsets_i32,
        change_frame_i32,
        op_offsets_i32,
        op_type_i32,
        op_pos_i32,
        op_owner_i32,
        op_left_i32,
        op_right_i32,
        site_rgba_f32,
        target_rgb_f32,
        config_i32,
        config_f32);
  }
#endif
  TORCH_CHECK(
      false,
      "world_foam_lane2_fused_slab_v0.endpoint_record_edit_mse_vjp_direct_atomic_rgb_only: no backend available for device ",
      boundary_f32.device());
}

std::tuple<torch::Tensor, torch::Tensor> endpoint_record_edit_coeff16_mse_vjp_direct_atomic_rgb_only_dispatch(
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
#if defined(__APPLE__)
  if (coeff_f16.device().is_mps()) {
    return metal_endpoint_record_edit_coeff16_mse_vjp_direct_atomic_rgb_only(
        coeff_f16,
        frame_t_f32,
        base_offsets_i32,
        base_owner_i32,
        base_left_i32,
        base_right_i32,
        track_change_offsets_i32,
        change_frame_i32,
        op_offsets_i32,
        op_type_i32,
        op_pos_i32,
        op_owner_i32,
        op_left_i32,
        op_right_i32,
        site_rgba_f32,
        target_rgb_f32,
        config_i32,
        config_f32);
  }
#endif
  TORCH_CHECK(
      false,
      "world_foam_lane2_fused_slab_v0.endpoint_record_edit_coeff16_mse_vjp_direct_atomic_rgb_only: no backend available for device ",
      coeff_f16.device());
}

torch::Tensor endpoint_record_edit_block4_vjp_direct_atomic_rgb_only_dispatch(
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
#if defined(__APPLE__)
  if (boundary_f32.device().is_mps()) {
    return metal_endpoint_record_edit_block4_vjp_direct_atomic_rgb_only(
        boundary_f32,
        rays_f32,
        frame_t_f32,
        anchor_offsets_i32,
        anchor_owner_i32,
        anchor_left_i32,
        anchor_right_i32,
        track_block_change_offsets_i32,
        change_frame_i32,
        op_offsets_i32,
        op_type_i32,
        op_pos_i32,
        op_owner_i32,
        op_left_i32,
        op_right_i32,
        site_rgba_f32,
        grad_rgb_f32,
        config_i32,
        config_f32);
  }
#endif
  TORCH_CHECK(
      false,
      "world_foam_lane2_fused_slab_v0.endpoint_record_edit_block4_vjp_direct_atomic_rgb_only: no backend available for device ",
      boundary_f32.device());
}

torch::Tensor endpoint_record_edit_block_coeff_vjp_direct_atomic_rgb_only_dispatch(
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
#if defined(__APPLE__)
  if (coeff_f32.device().is_mps()) {
    return metal_endpoint_record_edit_block_coeff_vjp_direct_atomic_rgb_only(
        coeff_f32,
        frame_t_f32,
        anchor_offsets_i32,
        anchor_owner_i32,
        anchor_left_i32,
        anchor_right_i32,
        track_block_change_offsets_i32,
        change_frame_i32,
        op_offsets_i32,
        op_type_i32,
        op_pos_i32,
        op_owner_i32,
        op_left_i32,
        op_right_i32,
        site_rgba_f32,
        grad_rgb_f32,
        config_i32,
        config_f32);
  }
#endif
  TORCH_CHECK(
      false,
      "world_foam_lane2_fused_slab_v0.endpoint_record_edit_block_coeff_vjp_direct_atomic_rgb_only: no backend available for device ",
      coeff_f32.device());
}

std::tuple<torch::Tensor, torch::Tensor> endpoint_record_edit_block_coeff_mse_vjp_direct_atomic_rgb_only_dispatch(
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
#if defined(__APPLE__)
  if (coeff_f32.device().is_mps()) {
    return metal_endpoint_record_edit_block_coeff_mse_vjp_direct_atomic_rgb_only(
        coeff_f32,
        frame_t_f32,
        anchor_offsets_i32,
        anchor_owner_i32,
        anchor_left_i32,
        anchor_right_i32,
        track_block_change_offsets_i32,
        change_frame_i32,
        op_offsets_i32,
        op_type_i32,
        op_pos_i32,
        op_owner_i32,
        op_left_i32,
        op_right_i32,
        site_rgba_f32,
        target_rgb_f32,
        config_i32,
        config_f32);
  }
#endif
  TORCH_CHECK(
      false,
      "world_foam_lane2_fused_slab_v0.endpoint_record_edit_block_coeff_mse_vjp_direct_atomic_rgb_only: no backend available for device ",
      coeff_f32.device());
}

std::tuple<torch::Tensor, torch::Tensor> endpoint_record_edit_block_coeff16_mse_vjp_direct_atomic_rgb_only_dispatch(
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
#if defined(__APPLE__)
  if (coeff_f16.device().is_mps()) {
    return metal_endpoint_record_edit_block_coeff16_mse_vjp_direct_atomic_rgb_only(
        coeff_f16,
        frame_t_f32,
        anchor_offsets_i32,
        anchor_owner_i32,
        anchor_left_i32,
        anchor_right_i32,
        track_block_change_offsets_i32,
        change_frame_i32,
        op_offsets_i32,
        op_type_i32,
        op_pos_i32,
        op_owner_i32,
        op_left_i32,
        op_right_i32,
        site_rgba_f32,
        target_rgb_f32,
        config_i32,
        config_f32);
  }
#endif
  TORCH_CHECK(
      false,
      "world_foam_lane2_fused_slab_v0.endpoint_record_edit_block_coeff16_mse_vjp_direct_atomic_rgb_only: no backend available for device ",
      coeff_f16.device());
}

std::tuple<torch::Tensor, torch::Tensor> endpoint_record_edit_block_coeff16_packed_mse_vjp_direct_atomic_rgb_only_dispatch(
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
#if defined(__APPLE__)
  if (coeff_f16.device().is_mps()) {
    return metal_endpoint_record_edit_block_coeff16_packed_mse_vjp_direct_atomic_rgb_only(
        coeff_f16,
        frame_t_f32,
        anchor_offsets_i32,
        anchor_record_i32,
        track_block_change_offsets_i32,
        change_frame_i32,
        op_offsets_i32,
        op_type_i32,
        op_pos_i32,
        op_record_i32,
        site_rgba_f32,
        target_rgb_f32,
        config_i32,
        config_f32);
  }
#endif
  TORCH_CHECK(
      false,
      "world_foam_lane2_fused_slab_v0.endpoint_record_edit_block_coeff16_packed_mse_vjp_direct_atomic_rgb_only: no backend available for device ",
      coeff_f16.device());
}

std::tuple<torch::Tensor, torch::Tensor> endpoint_record_edit_block_coeff16_i16_mse_vjp_direct_atomic_rgb_only_dispatch(
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
#if defined(__APPLE__)
  if (coeff_f16.device().is_mps()) {
    return metal_endpoint_record_edit_block_coeff16_i16_mse_vjp_direct_atomic_rgb_only(
        coeff_f16,
        frame_t_f32,
        anchor_offsets_i32,
        anchor_owner_i16,
        anchor_left_i16,
        anchor_right_i16,
        track_block_change_offsets_i32,
        change_frame_i32,
        op_offsets_i32,
        op_type_i32,
        op_pos_i32,
        op_owner_i16,
        op_left_i16,
        op_right_i16,
        site_rgba_f32,
        target_rgb_f32,
        config_i32,
        config_f32);
  }
#endif
  TORCH_CHECK(
      false,
      "world_foam_lane2_fused_slab_v0.endpoint_record_edit_block_coeff16_i16_mse_vjp_direct_atomic_rgb_only: no backend available for device ",
      coeff_f16.device());
}

std::tuple<torch::Tensor, torch::Tensor> endpoint_record_edit_block_coeff16_i16x3_mse_vjp_direct_atomic_rgb_only_dispatch(
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
#if defined(__APPLE__)
  if (coeff_f16.device().is_mps()) {
    return metal_endpoint_record_edit_block_coeff16_i16x3_mse_vjp_direct_atomic_rgb_only(
        coeff_f16,
        frame_t_f32,
        anchor_offsets_i32,
        anchor_record_i16,
        track_block_change_offsets_i32,
        change_frame_i32,
        op_offsets_i32,
        op_type_i32,
        op_pos_i32,
        op_record_i16,
        site_rgba_f32,
        target_rgb_f32,
        config_i32,
        config_f32);
  }
#endif
  TORCH_CHECK(
      false,
      "world_foam_lane2_fused_slab_v0.endpoint_record_edit_block_coeff16_i16x3_mse_vjp_direct_atomic_rgb_only: no backend available for device ",
      coeff_f16.device());
}

torch::Tensor endpoint_record_edit_block_coeff16_vjp_direct_atomic_rgb_only_dispatch(
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
#if defined(__APPLE__)
  if (coeff_f16.device().is_mps()) {
    return metal_endpoint_record_edit_block_coeff16_vjp_direct_atomic_rgb_only(
        coeff_f16,
        frame_t_f32,
        anchor_offsets_i32,
        anchor_owner_i32,
        anchor_left_i32,
        anchor_right_i32,
        track_block_change_offsets_i32,
        change_frame_i32,
        op_offsets_i32,
        op_type_i32,
        op_pos_i32,
        op_owner_i32,
        op_left_i32,
        op_right_i32,
        site_rgba_f32,
        grad_rgb_f32,
        config_i32,
        config_f32);
  }
#endif
  TORCH_CHECK(
      false,
      "world_foam_lane2_fused_slab_v0.endpoint_record_edit_block_coeff16_vjp_direct_atomic_rgb_only: no backend available for device ",
      coeff_f16.device());
}

}  // namespace
}  // namespace world_foam_lane2_fused_slab

TORCH_LIBRARY(world_foam_lane2_fused_slab_v0, m) {
  m.def("pack_endpoint_records_i32_cpu(Tensor owner_i32, Tensor left_i32, Tensor right_i32) -> Tensor");
  m.def(
      "gate4_delta_replace_from_cuts_cpu(Tensor cut_depths_f64, Tensor cut_ids_i64, Tensor cut_offsets_i64, Tensor start_segments_i64, Tensor initial_owner_i64, Tensor boundary_other_by_owner_i64, int frame_count, float epsilon) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");
  m.def(
      "gate4_owner_run_delta_replace_from_rays_cpu(Tensor boundary_f64, Tensor site_f64, Tensor site_density_f32, Tensor rays_f32, Tensor frame_indices_i64, int frame_count, float near, float far, float invalid_epsilon, float transmittance_threshold, float dedupe_epsilon, float segment_epsilon) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");
  m.def(
      "gate4_delta_replace_packed_from_cuts_cpu(Tensor cut_depths_f64, Tensor cut_ids_i64, Tensor cut_offsets_i64, Tensor start_segments_i64, Tensor initial_owner_i64, Tensor boundary_other_by_owner_i64, int frame_count, float epsilon) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");
  m.def(
      "gate4_cut_arrays_from_sorted_cpu(Tensor sorted_depths_f64, Tensor sorted_ids_i64, Tensor valid_counts_i64, Tensor row_active_i64, Tensor ray_coeff_f64, Tensor frame_t_f64, Tensor site_xyz_f64, Tensor site_t_f64, Tensor site_weight_f64, int frame_count, float near, float far, float dedupe_epsilon, float segment_epsilon) -> (Tensor, Tensor, Tensor, Tensor, Tensor)");
  m.def(
      "gate4_delta_replace_from_sorted_cpu(Tensor sorted_depths_f64, Tensor sorted_ids_i64, Tensor valid_counts_i64, Tensor row_active_i64, Tensor ray_coeff_f64, Tensor frame_t_f64, Tensor site_xyz_f64, Tensor site_t_f64, Tensor site_weight_f64, Tensor boundary_other_by_owner_i64, int frame_count, float near, float far, float dedupe_epsilon, float segment_epsilon) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");
  m.def(
      "gate4_delta_replace_packed_from_sorted_cpu(Tensor sorted_depths_f64, Tensor sorted_ids_i64, Tensor valid_counts_i64, Tensor row_active_i64, Tensor ray_coeff_f64, Tensor frame_t_f64, Tensor site_xyz_f64, Tensor site_t_f64, Tensor site_weight_f64, Tensor boundary_other_by_owner_i64, int frame_count, float near, float far, float dedupe_epsilon, float segment_epsilon) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");
  m.def(
      "gate4_delta_replace_packed_from_coeff_csr_cpu(Tensor row_offsets_i64, Tensor candidate_ids_i64, Tensor candidate_depth_coeffs_f64, Tensor row_index_i64, Tensor ray_coeff_f64, Tensor frame_t_f64, Tensor site_xyz_f64, Tensor site_t_f64, Tensor site_weight_f64, Tensor boundary_other_by_owner_i64, int frame_count, float near, float far, float invalid_epsilon, float dedupe_epsilon, float segment_epsilon) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");
  m.def(
      "count_power_boundary_events(Tensor boundary_f32, Tensor boundary_u32, Tensor beam_f32, Tensor beam_u32, Tensor config_i32, Tensor config_f32) -> Tensor");
  m.def(
      "shared_signal_replay(Tensor boundary_f32, Tensor candidate_mask_u32, Tensor sites_f32, Tensor site_signal_f32, Tensor beam_f32, Tensor frame_t_f32, Tensor grad_output_f32, Tensor config_i32, Tensor config_f32) -> (Tensor, Tensor)");
  m.def(
      "shared_rgb_replay(Tensor boundary_f32, Tensor candidate_mask_u32, Tensor sites_f32, Tensor site_rgb_f32, Tensor beam_f32, Tensor frame_t_f32, Tensor grad_output_rgb_f32, Tensor config_i32, Tensor config_f32) -> (Tensor, Tensor)");
  m.def(
      "shared_rgba_depth_replay(Tensor boundary_f32, Tensor candidate_mask_u32, Tensor sites_f32, Tensor site_rgba_f32, Tensor beam_f32, Tensor frame_t_f32, Tensor config_i32, Tensor config_f32) -> (Tensor, Tensor, Tensor)");
  m.def(
      "shared_rgba_depth_vjp(Tensor boundary_f32, Tensor candidate_mask_u32, Tensor sites_f32, Tensor site_rgba_f32, Tensor beam_f32, Tensor frame_t_f32, Tensor grad_rgb_f32, Tensor grad_alpha_f32, Tensor grad_depth_f32, Tensor config_i32, Tensor config_f32) -> (Tensor, Tensor, Tensor, Tensor)");
  m.def(
      "realray_rgba_depth_replay(Tensor boundary_f32, Tensor sites_f32, Tensor site_rgba_f32, Tensor rays_f32, Tensor frame_t_f32, Tensor config_i32, Tensor config_f32) -> (Tensor, Tensor, Tensor)");
  m.def(
      "shared_realray_rgba_depth_replay(Tensor boundary_f32, Tensor candidate_mask_i32, Tensor sites_f32, Tensor site_rgba_f32, Tensor track_rays_f32, Tensor frame_t_f32, Tensor config_i32, Tensor config_f32) -> (Tensor, Tensor, Tensor)");
  m.def(
      "shared_realray_rgba_depth_vjp(Tensor boundary_f32, Tensor candidate_mask_i32, Tensor sites_f32, Tensor site_rgba_f32, Tensor track_rays_f32, Tensor frame_t_f32, Tensor grad_rgb_f32, Tensor grad_alpha_f32, Tensor grad_depth_f32, Tensor config_i32, Tensor config_f32) -> (Tensor, Tensor, Tensor, Tensor)");
  m.def(
      "shared_realray_rgba_depth_vjp_reduce(Tensor boundary_f32, Tensor candidate_mask_i32, Tensor sites_f32, Tensor site_rgba_f32, Tensor track_rays_f32, Tensor frame_t_f32, Tensor grad_rgb_f32, Tensor grad_alpha_f32, Tensor grad_depth_f32, Tensor config_i32, Tensor config_f32) -> (Tensor, Tensor, Tensor, Tensor)");
  m.def(
      "shared_realray_rgba_depth_vjp_reduce_csr(Tensor boundary_f32, Tensor row_index_i32, Tensor candidate_row_offsets_i32, Tensor candidate_boundary_ids_i32, Tensor sites_f32, Tensor site_rgba_f32, Tensor track_rays_f32, Tensor frame_t_f32, Tensor grad_rgb_f32, Tensor grad_alpha_f32, Tensor grad_depth_f32, Tensor config_i32, Tensor config_f32) -> (Tensor, Tensor, Tensor, Tensor)");
  m.def(
      "fused_slab_affine_realray_rgba_depth_replay(Tensor boundary_f32, Tensor row_index_i32, Tensor candidate_row_offsets_i32, Tensor candidate_boundary_ids_i32, Tensor sites_f32, Tensor site_rgba_f32, Tensor ray_coeff_f32, Tensor frame_t_f32, Tensor config_i32, Tensor config_f32) -> (Tensor, Tensor, Tensor)");
  m.def(
      "fused_slab_affine_coeff_realray_rgba_depth_replay(Tensor row_index_i32, Tensor candidate_row_offsets_i32, Tensor candidate_depth_coeff_f32, Tensor sites_f32, Tensor site_rgba_f32, Tensor ray_coeff_f32, Tensor frame_t_f32, Tensor config_i32, Tensor config_f32) -> (Tensor, Tensor, Tensor)");
  m.def(
      "fused_slab_affine_coeff16_realray_rgba_depth_replay(Tensor row_index_i32, Tensor candidate_row_offsets_i32, Tensor candidate_depth_coeff_f16, Tensor sites_f32, Tensor site_rgba_f32, Tensor ray_coeff_f32, Tensor frame_t_f32, Tensor config_i32, Tensor config_f32) -> (Tensor, Tensor, Tensor)");
  m.def(
      "fused_slab_affine_num32_den16_realray_rgba_depth_replay(Tensor row_index_i32, Tensor candidate_row_offsets_i32, Tensor candidate_depth_num_f32, Tensor candidate_depth_den_f16, Tensor sites_f32, Tensor site_rgba_f32, Tensor ray_coeff_f32, Tensor frame_t_f32, Tensor config_i32, Tensor config_f32) -> (Tensor, Tensor, Tensor)");
  m.def(
      "fused_slab_affine_num32_den16_ownerupdate_realray_rgba_depth_replay(Tensor row_index_i32, Tensor candidate_row_offsets_i32, Tensor candidate_boundary_ids_i32, Tensor candidate_depth_num_f32, Tensor candidate_depth_den_f16, Tensor boundary_site_pairs_i32, Tensor sites_f32, Tensor site_rgba_f32, Tensor ray_coeff_f32, Tensor frame_t_f32, Tensor config_i32, Tensor config_f32) -> (Tensor, Tensor, Tensor)");
  m.def(
      "fused_slab_affine_num32_den16_vjp_reduce(Tensor row_index_i32, Tensor candidate_row_offsets_i32, Tensor candidate_depth_num_f32, Tensor candidate_depth_den_f16, Tensor sites_f32, Tensor site_rgba_f32, Tensor ray_coeff_f32, Tensor frame_t_f32, Tensor grad_rgb_f32, Tensor grad_alpha_f32, Tensor grad_depth_f32, Tensor config_i32, Tensor config_f32) -> (Tensor, Tensor, Tensor, Tensor)");
  m.def(
      "fused_slab_affine_num32_den16_vjp_direct_atomic(Tensor row_index_i32, Tensor candidate_row_offsets_i32, Tensor candidate_depth_num_f32, Tensor candidate_depth_den_f16, Tensor sites_f32, Tensor site_rgba_f32, Tensor ray_coeff_f32, Tensor frame_t_f32, Tensor grad_rgb_f32, Tensor grad_alpha_f32, Tensor grad_depth_f32, Tensor config_i32, Tensor config_f32) -> (Tensor, Tensor, Tensor, Tensor)");
  m.def(
      "fused_slab_affine_num32_den16_vjp_direct_atomic_grad_only(Tensor row_index_i32, Tensor candidate_row_offsets_i32, Tensor candidate_depth_num_f32, Tensor candidate_depth_den_f16, Tensor sites_f32, Tensor site_rgba_f32, Tensor ray_coeff_f32, Tensor frame_t_f32, Tensor grad_rgb_f32, Tensor grad_alpha_f32, Tensor grad_depth_f32, Tensor config_i32, Tensor config_f32) -> Tensor");
  m.def(
      "fused_slab_affine_num32_den16_vjp_direct_atomic_grad_only_ownerupdate(Tensor row_index_i32, Tensor candidate_row_offsets_i32, Tensor candidate_boundary_ids_i32, Tensor candidate_depth_num_f32, Tensor candidate_depth_den_f16, Tensor boundary_site_pairs_i32, Tensor sites_f32, Tensor site_rgba_f32, Tensor ray_coeff_f32, Tensor frame_t_f32, Tensor grad_rgb_f32, Tensor grad_alpha_f32, Tensor grad_depth_f32, Tensor config_i32, Tensor config_f32) -> Tensor");
  m.def(
      "fused_slab_affine_num32_den16_vjp_direct_atomic_rgb_only(Tensor row_index_i32, Tensor candidate_row_offsets_i32, Tensor candidate_depth_num_f32, Tensor candidate_depth_den_f16, Tensor sites_f32, Tensor site_rgba_f32, Tensor ray_coeff_f32, Tensor frame_t_f32, Tensor grad_rgb_f32, Tensor config_i32, Tensor config_f32) -> Tensor");
  m.def(
      "fused_slab_affine_num32_den16_mse_vjp_direct_atomic_rgb_only(Tensor row_index_i32, Tensor candidate_row_offsets_i32, Tensor candidate_depth_num_f32, Tensor candidate_depth_den_f16, Tensor sites_f32, Tensor site_rgba_f32, Tensor ray_coeff_f32, Tensor frame_t_f32, Tensor target_rgb_f32, Tensor config_i32, Tensor config_f32) -> (Tensor, Tensor)");
  m.def(
      "fused_slab_affine_num32_den16_mse_vjp_direct_atomic_track_rgb_only(Tensor row_index_i32, Tensor candidate_row_offsets_i32, Tensor candidate_depth_num_f32, Tensor candidate_depth_den_f16, Tensor sites_f32, Tensor site_rgba_f32, Tensor ray_coeff_f32, Tensor frame_t_f32, Tensor target_rgb_f32, Tensor config_i32, Tensor config_f32) -> (Tensor, Tensor)");
  m.def(
      "fused_slab_affine_coeff16_mse_vjp_direct_atomic_rgb_only(Tensor row_index_i32, Tensor candidate_row_offsets_i32, Tensor candidate_depth_coeff_f16, Tensor sites_f32, Tensor site_rgba_f32, Tensor ray_coeff_f32, Tensor frame_t_f32, Tensor target_rgb_f32, Tensor config_i32, Tensor config_f32) -> (Tensor, Tensor)");
  m.def(
      "fused_slab_affine_coeff16_cap224_mse_vjp_direct_atomic_rgb_only(Tensor row_index_i32, Tensor candidate_row_offsets_i32, Tensor candidate_depth_coeff_f16, Tensor sites_f32, Tensor site_rgba_f32, Tensor ray_coeff_f32, Tensor frame_t_f32, Tensor target_rgb_f32, Tensor config_i32, Tensor config_f32) -> (Tensor, Tensor)");
  m.def(
      "fused_slab_affine_coeff16_densitymask_mse_vjp_direct_atomic_rgb_only(Tensor row_index_i32, Tensor candidate_row_offsets_i32, Tensor candidate_depth_coeff_f16, Tensor sites_f32, Tensor site_rgba_f32, Tensor ray_coeff_f32, Tensor frame_t_f32, Tensor target_rgb_f32, Tensor config_i32, Tensor config_f32) -> (Tensor, Tensor)");
  m.def(
      "fused_slab_affine_coeff16_mse_vjp_direct_atomic_sample_reduce_rgb_only(Tensor row_index_i32, Tensor candidate_row_offsets_i32, Tensor candidate_depth_coeff_f16, Tensor sites_f32, Tensor site_rgba_f32, Tensor ray_coeff_f32, Tensor frame_t_f32, Tensor target_rgb_f32, Tensor config_i32, Tensor config_f32) -> (Tensor, Tensor)");
  m.def(
      "fused_slab_affine_coeff16_sortnet_mse_vjp_direct_atomic_rgb_only(Tensor row_index_i32, Tensor candidate_row_offsets_i32, Tensor candidate_depth_coeff_f16, Tensor sites_f32, Tensor site_rgba_f32, Tensor ray_coeff_f32, Tensor frame_t_f32, Tensor target_rgb_f32, Tensor config_i32, Tensor config_f32) -> (Tensor, Tensor)");
  m.def(
      "fused_slab_affine_coeff16_mse_vjp_direct_atomic_sitecache_rgb_only(Tensor row_index_i32, Tensor candidate_row_offsets_i32, Tensor candidate_depth_coeff_f16, Tensor sites_f32, Tensor site_rgba_f32, Tensor ray_coeff_f32, Tensor frame_t_f32, Tensor target_rgb_f32, Tensor config_i32, Tensor config_f32) -> (Tensor, Tensor)");
  m.def(
      "fused_slab_affine_coeff16_mse_vjp_direct_atomic_framegroup16_cached_rgb_only(Tensor row_index_i32, Tensor candidate_row_offsets_i32, Tensor candidate_depth_coeff_f16, Tensor sites_f32, Tensor site_rgba_f32, Tensor ray_coeff_f32, Tensor frame_t_f32, Tensor target_rgb_f32, Tensor config_i32, Tensor config_f32) -> (Tensor, Tensor)");
  m.def(
      "fused_slab_affine_coeff16_ownerupdate_mse_vjp_direct_atomic_rgb_only(Tensor row_index_i32, Tensor candidate_row_offsets_i32, Tensor candidate_boundary_ids_i32, Tensor candidate_depth_coeff_f16, Tensor boundary_site_pairs_i32, Tensor sites_f32, Tensor site_rgba_f32, Tensor ray_coeff_f32, Tensor frame_t_f32, Tensor target_rgb_f32, Tensor config_i32, Tensor config_f32) -> (Tensor, Tensor)");
  m.def(
      "fused_slab_affine_coeff16_ownerupdate_i16_mse_vjp_direct_atomic_rgb_only(Tensor row_index_i32, Tensor candidate_row_offsets_i32, Tensor candidate_boundary_ids_i16, Tensor candidate_depth_coeff_f16, Tensor boundary_site_pairs_i16, Tensor sites_f32, Tensor site_rgba_f32, Tensor ray_coeff_f32, Tensor frame_t_f32, Tensor target_rgb_f32, Tensor config_i32, Tensor config_f32) -> (Tensor, Tensor)");
  m.def(
      "fused_slab_affine_coeff16_ownerkeep_i16_mse_vjp_direct_atomic_rgb_only(Tensor row_index_i32, Tensor candidate_row_offsets_i32, Tensor candidate_boundary_ids_i16, Tensor candidate_depth_coeff_f16, Tensor boundary_site_pairs_i16, Tensor sites_f32, Tensor site_rgba_f32, Tensor ray_coeff_f32, Tensor frame_t_f32, Tensor target_rgb_f32, Tensor config_i32, Tensor config_f32) -> (Tensor, Tensor)");
  m.def(
      "fused_slab_affine_coeff16_ownerkeep_mse_vjp_direct_atomic_rgb_only(Tensor row_index_i32, Tensor candidate_row_offsets_i32, Tensor candidate_boundary_ids_i32, Tensor candidate_depth_coeff_f16, Tensor boundary_site_pairs_i32, Tensor sites_f32, Tensor site_rgba_f32, Tensor ray_coeff_f32, Tensor frame_t_f32, Tensor target_rgb_f32, Tensor config_i32, Tensor config_f32) -> (Tensor, Tensor)");
  m.def(
      "fused_slab_affine_coeff16_mse_vjp_direct_atomic_track_rgb_only(Tensor row_index_i32, Tensor candidate_row_offsets_i32, Tensor candidate_depth_coeff_f16, Tensor sites_f32, Tensor site_rgba_f32, Tensor ray_coeff_f32, Tensor frame_t_f32, Tensor target_rgb_f32, Tensor config_i32, Tensor config_f32) -> (Tensor, Tensor)");
  m.def(
      "fused_slab_affine_num32_den16_vjp_direct_atomic_track(Tensor row_index_i32, Tensor candidate_row_offsets_i32, Tensor candidate_depth_num_f32, Tensor candidate_depth_den_f16, Tensor sites_f32, Tensor site_rgba_f32, Tensor ray_coeff_f32, Tensor frame_t_f32, Tensor grad_rgb_f32, Tensor grad_alpha_f32, Tensor grad_depth_f32, Tensor config_i32, Tensor config_f32) -> Tensor");
  m.def(
      "segment_tape_rgba_depth_replay(Tensor segment_offsets_i32, Tensor segment_owner_i32, Tensor segment_length_f32, Tensor segment_mid_f32, Tensor site_rgba_f32, Tensor config_i32, Tensor config_f32) -> (Tensor, Tensor, Tensor)");
	  m.def(
	      "segment_tape_vjp_direct_atomic_grad_only(Tensor segment_offsets_i32, Tensor segment_owner_i32, Tensor segment_length_f32, Tensor segment_mid_f32, Tensor site_rgba_f32, Tensor grad_rgb_f32, Tensor grad_alpha_f32, Tensor grad_depth_f32, Tensor config_i32, Tensor config_f32) -> Tensor");
	  m.def(
	      "segment_tape_mse_vjp_direct_atomic_rgb_only(Tensor segment_offsets_i32, Tensor segment_owner_i32, Tensor segment_length_f32, Tensor site_rgba_f32, Tensor target_rgb_f32, Tensor config_i32, Tensor config_f32) -> (Tensor, Tensor)");
	  m.def(
	      "segment_tape_vjp_direct_atomic_track(Tensor segment_offsets_i32, Tensor segment_owner_i32, Tensor segment_length_f32, Tensor segment_mid_f32, Tensor site_rgba_f32, Tensor grad_rgb_f32, Tensor grad_alpha_f32, Tensor grad_depth_f32, Tensor config_i32, Tensor config_f32) -> Tensor");
  m.def(
      "endpoint_run_rgba_depth_replay(Tensor run_offsets_i32, Tensor run_owner_i32, Tensor run_start_f32, Tensor run_end_f32, Tensor site_rgba_f32, Tensor config_i32, Tensor config_f32) -> (Tensor, Tensor, Tensor)");
  m.def(
      "endpoint_run_vjp_direct_atomic_grad_only(Tensor run_offsets_i32, Tensor run_owner_i32, Tensor run_start_f32, Tensor run_end_f32, Tensor site_rgba_f32, Tensor grad_rgb_f32, Tensor grad_alpha_f32, Tensor grad_depth_f32, Tensor config_i32, Tensor config_f32) -> Tensor");
  m.def(
      "endpoint_run_mse_vjp_direct_atomic_rgb_only(Tensor run_offsets_i32, Tensor run_owner_i32, Tensor run_start_f32, Tensor run_end_f32, Tensor site_rgba_f32, Tensor target_rgb_f32, Tensor config_i32, Tensor config_f32) -> (Tensor, Tensor)");
  m.def(
      "endpoint_delta_replace_rgba_depth_replay(Tensor base_offsets_i32, Tensor base_owner_i32, Tensor base_start_f32, Tensor base_end_f32, Tensor track_change_offsets_i32, Tensor change_frame_i32, Tensor change_offsets_i32, Tensor change_owner_i32, Tensor change_start_f32, Tensor change_end_f32, Tensor site_rgba_f32, Tensor config_i32, Tensor config_f32) -> (Tensor, Tensor, Tensor)");
  m.def(
      "endpoint_delta_replace_vjp_direct_atomic_grad_only(Tensor base_offsets_i32, Tensor base_owner_i32, Tensor base_start_f32, Tensor base_end_f32, Tensor track_change_offsets_i32, Tensor change_frame_i32, Tensor change_offsets_i32, Tensor change_owner_i32, Tensor change_start_f32, Tensor change_end_f32, Tensor site_rgba_f32, Tensor grad_rgb_f32, Tensor grad_alpha_f32, Tensor grad_depth_f32, Tensor config_i32, Tensor config_f32) -> Tensor");
  m.def(
      "endpoint_record_delta_replace_rgba_depth_replay(Tensor boundary_f32, Tensor rays_f32, Tensor frame_t_f32, Tensor base_offsets_i32, Tensor base_owner_i32, Tensor base_left_i32, Tensor base_right_i32, Tensor track_change_offsets_i32, Tensor change_frame_i32, Tensor change_offsets_i32, Tensor change_owner_i32, Tensor change_left_i32, Tensor change_right_i32, Tensor site_rgba_f32, Tensor config_i32, Tensor config_f32) -> (Tensor, Tensor, Tensor)");
  m.def(
      "endpoint_record_delta_replace_vjp_direct_atomic_grad_only(Tensor boundary_f32, Tensor rays_f32, Tensor frame_t_f32, Tensor base_offsets_i32, Tensor base_owner_i32, Tensor base_left_i32, Tensor base_right_i32, Tensor track_change_offsets_i32, Tensor change_frame_i32, Tensor change_offsets_i32, Tensor change_owner_i32, Tensor change_left_i32, Tensor change_right_i32, Tensor site_rgba_f32, Tensor grad_rgb_f32, Tensor grad_alpha_f32, Tensor grad_depth_f32, Tensor config_i32, Tensor config_f32) -> Tensor");
  m.def(
      "endpoint_record_delta_replace_coeff16_mse_vjp_direct_atomic_rgb_only(Tensor coeff_f16, Tensor frame_t_f32, Tensor base_offsets_i32, Tensor base_owner_i32, Tensor base_left_i32, Tensor base_right_i32, Tensor track_change_offsets_i32, Tensor change_frame_i32, Tensor change_offsets_i32, Tensor change_owner_i32, Tensor change_left_i32, Tensor change_right_i32, Tensor site_rgba_f32, Tensor target_rgb_f32, Tensor config_i32, Tensor config_f32) -> (Tensor, Tensor)");
  m.def(
      "endpoint_record_delta_replace_coeff16_i16x3_mse_vjp_direct_atomic_rgb_only(Tensor coeff_f16, Tensor frame_t_f32, Tensor base_offsets_i32, Tensor base_record_i16, Tensor track_change_offsets_i32, Tensor change_frame_i32, Tensor change_offsets_i32, Tensor change_record_i16, Tensor site_rgba_f32, Tensor target_rgb_f32, Tensor config_i32, Tensor config_f32) -> (Tensor, Tensor)");
  m.def(
      "endpoint_record_delta_replace_coeff16_i16x3_framegroup16_mse_vjp_direct_atomic_rgb_only(Tensor coeff_f16, Tensor frame_t_f32, Tensor base_offsets_i32, Tensor base_record_i16, Tensor track_change_offsets_i32, Tensor track_chunk_change_offsets_i16, Tensor change_frame_i32, Tensor change_offsets_i32, Tensor change_record_i16, Tensor site_rgba_f32, Tensor target_rgb_f32, Tensor config_i32, Tensor config_f32) -> (Tensor, Tensor)");
  m.def(
      "endpoint_record_delta_replace_coeff16_i16x3_framegroup16_ownerreduce_mse_vjp_direct_atomic_rgb_only(Tensor coeff_f16, Tensor frame_t_f32, Tensor base_offsets_i32, Tensor base_record_i16, Tensor track_change_offsets_i32, Tensor track_chunk_change_offsets_i16, Tensor track_chunk_owner_offsets_i32, Tensor track_chunk_owner_i16, Tensor change_frame_i32, Tensor change_offsets_i32, Tensor change_record_i16, Tensor site_rgba_f32, Tensor target_rgb_f32, Tensor config_i32, Tensor config_f32) -> (Tensor, Tensor)");
  m.def(
      "endpoint_record_delta_replace_coeff16_i16cols_framegroup16_mse_vjp_direct_atomic_rgb_only(Tensor coeff_f16, Tensor frame_t_f32, Tensor base_offsets_i32, Tensor base_record_i16, Tensor track_change_offsets_i32, Tensor track_chunk_change_offsets_i16, Tensor change_frame_i32, Tensor change_offsets_i32, Tensor change_record_i16, Tensor site_rgba_f32, Tensor target_rgb_f32, Tensor config_i32, Tensor config_f32) -> (Tensor, Tensor)");
  m.def(
      "endpoint_record_delta_replace_coeff16_i16x3_framegroup64_mse_vjp_direct_atomic_rgb_only(Tensor coeff_f16, Tensor frame_t_f32, Tensor base_offsets_i32, Tensor base_record_i16, Tensor track_change_offsets_i32, Tensor track_chunk_change_offsets_i16, Tensor change_frame_i32, Tensor change_offsets_i32, Tensor change_record_i16, Tensor site_rgba_f32, Tensor target_rgb_f32, Tensor config_i32, Tensor config_f32) -> (Tensor, Tensor)");
  m.def(
      "endpoint_record_delta_replace_coeff16_packed_mse_vjp_direct_atomic_rgb_only(Tensor coeff_f16, Tensor frame_t_f32, Tensor base_offsets_i32, Tensor base_record_i32, Tensor track_change_offsets_i32, Tensor change_frame_i32, Tensor change_offsets_i32, Tensor change_record_i32, Tensor site_rgba_f32, Tensor target_rgb_f32, Tensor config_i32, Tensor config_f32) -> (Tensor, Tensor)");
  m.def(
      "endpoint_record_delta_replace_coeff16_packed_framegroup16_mse_vjp_direct_atomic_rgb_only(Tensor coeff_f16, Tensor frame_t_f32, Tensor base_offsets_i32, Tensor base_record_i32, Tensor track_change_offsets_i32, Tensor track_chunk_change_offsets_i16, Tensor change_frame_i32, Tensor change_offsets_i32, Tensor change_record_i32, Tensor site_rgba_f32, Tensor target_rgb_f32, Tensor config_i32, Tensor config_f32) -> (Tensor, Tensor)");
  m.def(
      "endpoint_record_delta_replace_coeff16_packed_framegroup16_launch_only_mse_vjp_direct_atomic_rgb_only(Tensor coeff_f16, Tensor frame_t_f32, Tensor base_offsets_i32, Tensor base_record_i32, Tensor track_change_offsets_i32, Tensor track_chunk_change_offsets_i16, Tensor change_frame_i32, Tensor change_offsets_i32, Tensor change_record_i32, Tensor site_rgba_f32, Tensor target_rgb_f32, Tensor config_i32, Tensor config_f32, int boundary_count, int track_count, int frame_count, int site_count, int base_record_count, int change_count, int change_record_count) -> (Tensor, Tensor)");
  m.def(
      "endpoint_record_delta_replace_coeff16_packed_framegroup16_unchecked_launch_only_mse_vjp_direct_atomic_rgb_only(Tensor coeff_f16, Tensor frame_t_f32, Tensor base_offsets_i32, Tensor base_record_i32, Tensor track_change_offsets_i32, Tensor track_chunk_change_offsets_i16, Tensor change_frame_i32, Tensor change_offsets_i32, Tensor change_record_i32, Tensor site_rgba_f32, Tensor target_rgb_f32, Tensor config_i32, Tensor config_f32, int track_count, int frame_count, int site_count) -> (Tensor, Tensor)");
  m.def(
      "endpoint_record_delta_replace_coeff16_packed_framegroup16_reduce32_launch_only_mse_vjp_direct_atomic_rgb_only(Tensor coeff_f16, Tensor frame_t_f32, Tensor base_offsets_i32, Tensor base_record_i32, Tensor track_change_offsets_i32, Tensor track_chunk_change_offsets_i16, Tensor change_frame_i32, Tensor change_offsets_i32, Tensor change_record_i32, Tensor site_rgba_f32, Tensor target_rgb_f32, Tensor config_i32, Tensor config_f32, int boundary_count, int track_count, int frame_count, int site_count, int base_record_count, int change_count, int change_record_count) -> (Tensor, Tensor)");
  m.def(
      "endpoint_record_delta_replace_coeff16_packed_framegroup16_rowselect32_launch_only_mse_vjp_direct_atomic_rgb_only(Tensor coeff_f16, Tensor frame_t_f32, Tensor base_offsets_i32, Tensor base_record_i32, Tensor track_change_offsets_i32, Tensor track_chunk_change_offsets_i16, Tensor change_frame_i32, Tensor change_offsets_i32, Tensor change_record_i32, Tensor site_rgba_f32, Tensor target_rgb_f32, Tensor config_i32, Tensor config_f32, int boundary_count, int track_count, int frame_count, int site_count, int base_record_count, int change_count, int change_record_count) -> (Tensor, Tensor)");
  m.def(
      "endpoint_record_delta_replace_coeff16_packed_framegroup16_reduce32_unchecked_launch_only_mse_vjp_direct_atomic_rgb_only(Tensor coeff_f16, Tensor frame_t_f32, Tensor base_offsets_i32, Tensor base_record_i32, Tensor track_change_offsets_i32, Tensor track_chunk_change_offsets_i16, Tensor change_frame_i32, Tensor change_offsets_i32, Tensor change_record_i32, Tensor site_rgba_f32, Tensor target_rgb_f32, Tensor config_i32, Tensor config_f32, int track_count, int frame_count, int site_count) -> (Tensor, Tensor)");
  m.def(
      "endpoint_record_delta_replace_coeff16_packed_framegroup16_rowselect32_unchecked_launch_only_mse_vjp_direct_atomic_rgb_only(Tensor coeff_f16, Tensor frame_t_f32, Tensor base_offsets_i32, Tensor base_record_i32, Tensor track_change_offsets_i32, Tensor track_chunk_change_offsets_i16, Tensor change_frame_i32, Tensor change_offsets_i32, Tensor change_record_i32, Tensor site_rgba_f32, Tensor target_rgb_f32, Tensor config_i32, Tensor config_f32, int track_count, int frame_count, int site_count) -> (Tensor, Tensor)");
  m.def(
      "endpoint_record_delta_replace_coeff16_packed_framegroup16_rowdesc_launch_only_mse_vjp_direct_atomic_rgb_only(Tensor coeff_f16, Tensor frame_t_f32, Tensor row_begin_i32, Tensor row_len_source_i16, Tensor base_record_i32, Tensor change_record_i32, Tensor site_rgba_f32, Tensor target_rgb_f32, Tensor config_i32, Tensor config_f32, int boundary_count, int track_count, int frame_count, int site_count, int base_record_count, int change_record_count) -> (Tensor, Tensor)");
  m.def(
      "endpoint_record_delta_replace_coeff16_packed_framegroup16_rowdesc_unchecked_launch_only_mse_vjp_direct_atomic_rgb_only(Tensor coeff_f16, Tensor frame_t_f32, Tensor row_begin_i32, Tensor row_len_source_i16, Tensor base_record_i32, Tensor change_record_i32, Tensor site_rgba_f32, Tensor target_rgb_f32, Tensor config_i32, Tensor config_f32, int track_count, int frame_count, int site_count) -> (Tensor, Tensor)");
  m.def(
      "endpoint_record_delta_replace_coeff16_packed_framegroup16_rowdesc32_launch_only_mse_vjp_direct_atomic_rgb_only(Tensor coeff_f16, Tensor frame_t_f32, Tensor row_begin_i32, Tensor row_len_source_i16, Tensor base_record_i32, Tensor change_record_i32, Tensor site_rgba_f32, Tensor target_rgb_f32, Tensor config_i32, Tensor config_f32, int boundary_count, int track_count, int frame_count, int site_count, int base_record_count, int change_record_count) -> (Tensor, Tensor)");
  m.def(
      "endpoint_record_delta_replace_coeff16_packed_framegroup16_rowdesc32_unchecked_launch_only_mse_vjp_direct_atomic_rgb_only(Tensor coeff_f16, Tensor frame_t_f32, Tensor row_begin_i32, Tensor row_len_source_i16, Tensor base_record_i32, Tensor change_record_i32, Tensor site_rgba_f32, Tensor target_rgb_f32, Tensor config_i32, Tensor config_f32, int track_count, int frame_count, int site_count) -> (Tensor, Tensor)");
  m.def(
      "endpoint_record_delta_replace_coeff16_packed_framegroup16_recompute_launch_only_mse_vjp_direct_atomic_rgb_only(Tensor coeff_f16, Tensor frame_t_f32, Tensor base_offsets_i32, Tensor base_record_i32, Tensor track_change_offsets_i32, Tensor track_chunk_change_offsets_i16, Tensor change_frame_i32, Tensor change_offsets_i32, Tensor change_record_i32, Tensor site_rgba_f32, Tensor target_rgb_f32, Tensor config_i32, Tensor config_f32, int boundary_count, int track_count, int frame_count, int site_count, int base_record_count, int change_count, int change_record_count) -> (Tensor, Tensor)");
  m.def(
      "endpoint_record_delta_replace_coeff16_packed_framegroup16_smallrun16_launch_only_mse_vjp_direct_atomic_rgb_only(Tensor coeff_f16, Tensor frame_t_f32, Tensor base_offsets_i32, Tensor base_record_i32, Tensor track_change_offsets_i32, Tensor track_chunk_change_offsets_i16, Tensor change_frame_i32, Tensor change_offsets_i32, Tensor change_record_i32, Tensor site_rgba_f32, Tensor target_rgb_f32, Tensor config_i32, Tensor config_f32, int boundary_count, int track_count, int frame_count, int site_count, int base_record_count, int change_count, int change_record_count) -> (Tensor, Tensor)");
  m.def(
      "endpoint_record_delta_replace_coeff16_packed_framegroup16_materialized_launch_only_mse_vjp_direct_atomic_rgb_only(Tensor coeff_f16, Tensor frame_t_f32, Tensor base_offsets_i32, Tensor base_record_i32, Tensor track_change_offsets_i32, Tensor track_chunk_change_offsets_i16, Tensor change_frame_i32, Tensor change_offsets_i32, Tensor change_record_i32, Tensor site_rgba_f32, Tensor target_rgb_f32, Tensor config_i32, Tensor config_f32, int boundary_count, int track_count, int frame_count, int site_count, int base_record_count, int change_count, int change_record_count) -> (Tensor, Tensor)");
  m.def(
      "endpoint_record_delta_replace_coeff16_packed_framegroup16_recompute_mse_vjp_direct_atomic_rgb_only(Tensor coeff_f16, Tensor frame_t_f32, Tensor base_offsets_i32, Tensor base_record_i32, Tensor track_change_offsets_i32, Tensor track_chunk_change_offsets_i16, Tensor change_frame_i32, Tensor change_offsets_i32, Tensor change_record_i32, Tensor site_rgba_f32, Tensor target_rgb_f32, Tensor config_i32, Tensor config_f32) -> (Tensor, Tensor)");
  m.def(
      "endpoint_record_delta_replace_factorized_packed_framegroup16_recompute_mse_vjp_direct_atomic_rgb_only(Tensor boundary_f32, Tensor track_ray_coeff_f32, Tensor frame_t_f32, Tensor base_offsets_i16, Tensor base_record_i32, Tensor track_change_offsets_i16, Tensor track_chunk_change_offsets_i16, Tensor change_frame_i16, Tensor change_offsets_i16, Tensor change_record_i32, Tensor site_rgba_f32, Tensor target_rgb_f32, Tensor config_i32, Tensor config_f32) -> (Tensor, Tensor)");
  m.def(
      "endpoint_record_delta_replace_factorized_frameselect_recompute_mse_vjp_direct_atomic_rgb_only(Tensor boundary_f32, Tensor track_ray_coeff_f32, Tensor frame_t_f32, Tensor base_offsets_i16, Tensor base_record_i32, Tensor frame_change_index_i16, Tensor change_offsets_i16, Tensor change_record_i32, Tensor site_rgba_f32, Tensor target_rgb_f32, Tensor config_i32, Tensor config_f32) -> (Tensor, Tensor)");
  m.def(
      "endpoint_record_delta_replace_factorized_framebitmask_recompute_mse_vjp_direct_atomic_rgb_only(Tensor boundary_f32, Tensor track_ray_coeff_f32, Tensor frame_t_f32, Tensor base_offsets_i32, Tensor base_record_i32, Tensor track_change_offsets_i32, Tensor track_frame_mask_i32, Tensor change_offsets_i32, Tensor change_record_i32, Tensor site_rgba_f32, Tensor target_rgb_f32, Tensor config_i32, Tensor config_f32) -> (Tensor, Tensor)");
  m.def(
      "endpoint_record_delta_replace_coeff16_packed_framegroup16_smallrun16_mse_vjp_direct_atomic_rgb_only(Tensor coeff_f16, Tensor frame_t_f32, Tensor base_offsets_i32, Tensor base_record_i32, Tensor track_change_offsets_i32, Tensor track_chunk_change_offsets_i16, Tensor change_frame_i32, Tensor change_offsets_i32, Tensor change_record_i32, Tensor site_rgba_f32, Tensor target_rgb_f32, Tensor config_i32, Tensor config_f32) -> (Tensor, Tensor)");
  m.def(
      "endpoint_record_delta_replace_coeff16_packed_framegroup16_materialized_mse_vjp_direct_atomic_rgb_only(Tensor coeff_f16, Tensor frame_t_f32, Tensor base_offsets_i32, Tensor base_record_i32, Tensor track_change_offsets_i32, Tensor track_chunk_change_offsets_i16, Tensor change_frame_i32, Tensor change_offsets_i32, Tensor change_record_i32, Tensor site_rgba_f32, Tensor target_rgb_f32, Tensor config_i32, Tensor config_f32) -> (Tensor, Tensor)");
  m.def(
      "endpoint_record_delta_replace_coeff16_i16x4_framegroup16_mse_vjp_direct_atomic_rgb_only(Tensor coeff_f16, Tensor frame_t_f32, Tensor base_offsets_i32, Tensor base_record_i16, Tensor track_change_offsets_i32, Tensor track_chunk_change_offsets_i16, Tensor change_frame_i32, Tensor change_offsets_i32, Tensor change_record_i16, Tensor site_rgba_f32, Tensor target_rgb_f32, Tensor config_i32, Tensor config_f32) -> (Tensor, Tensor)");
  m.def(
      "endpoint_record_delta_replace_coeff16_i16x3_framegroup16_materialized_mse_vjp_direct_atomic_rgb_only(Tensor coeff_f16, Tensor frame_t_f32, Tensor base_offsets_i32, Tensor base_record_i16, Tensor track_change_offsets_i32, Tensor track_chunk_change_offsets_i16, Tensor change_frame_i32, Tensor change_offsets_i32, Tensor change_record_i16, Tensor site_rgba_f32, Tensor target_rgb_f32, Tensor config_i32, Tensor config_f32) -> (Tensor, Tensor)");
  m.def(
      "endpoint_record_delta_replace_coeff16_i16x4_mse_vjp_direct_atomic_rgb_only(Tensor coeff_f16, Tensor frame_t_f32, Tensor base_offsets_i32, Tensor base_record_i16, Tensor track_change_offsets_i32, Tensor change_frame_i32, Tensor change_offsets_i32, Tensor change_record_i16, Tensor site_rgba_f32, Tensor target_rgb_f32, Tensor config_i32, Tensor config_f32) -> (Tensor, Tensor)");
  m.def(
      "endpoint_record_edit_rgba_depth_replay(Tensor boundary_f32, Tensor rays_f32, Tensor frame_t_f32, Tensor base_offsets_i32, Tensor base_owner_i32, Tensor base_left_i32, Tensor base_right_i32, Tensor track_change_offsets_i32, Tensor change_frame_i32, Tensor op_offsets_i32, Tensor op_type_i32, Tensor op_pos_i32, Tensor op_owner_i32, Tensor op_left_i32, Tensor op_right_i32, Tensor site_rgba_f32, Tensor config_i32, Tensor config_f32) -> (Tensor, Tensor, Tensor)");
  m.def(
      "endpoint_record_edit_block4_rgba_depth_replay(Tensor boundary_f32, Tensor rays_f32, Tensor frame_t_f32, Tensor anchor_offsets_i32, Tensor anchor_owner_i32, Tensor anchor_left_i32, Tensor anchor_right_i32, Tensor track_block_change_offsets_i32, Tensor change_frame_i32, Tensor op_offsets_i32, Tensor op_type_i32, Tensor op_pos_i32, Tensor op_owner_i32, Tensor op_left_i32, Tensor op_right_i32, Tensor site_rgba_f32, Tensor config_i32, Tensor config_f32) -> (Tensor, Tensor, Tensor)");
  m.def(
      "endpoint_record_edit_block_coeff_rgba_depth_replay(Tensor coeff_f32, Tensor frame_t_f32, Tensor anchor_offsets_i32, Tensor anchor_owner_i32, Tensor anchor_left_i32, Tensor anchor_right_i32, Tensor track_block_change_offsets_i32, Tensor change_frame_i32, Tensor op_offsets_i32, Tensor op_type_i32, Tensor op_pos_i32, Tensor op_owner_i32, Tensor op_left_i32, Tensor op_right_i32, Tensor site_rgba_f32, Tensor config_i32, Tensor config_f32) -> (Tensor, Tensor, Tensor)");
  m.def(
      "endpoint_record_edit_block_coeff_rgb_replay(Tensor coeff_f32, Tensor frame_t_f32, Tensor anchor_offsets_i32, Tensor anchor_owner_i32, Tensor anchor_left_i32, Tensor anchor_right_i32, Tensor track_block_change_offsets_i32, Tensor change_frame_i32, Tensor op_offsets_i32, Tensor op_type_i32, Tensor op_pos_i32, Tensor op_owner_i32, Tensor op_left_i32, Tensor op_right_i32, Tensor site_rgba_f32, Tensor config_i32, Tensor config_f32) -> Tensor");
  m.def(
      "endpoint_record_edit_block_coeff16_rgba_depth_replay(Tensor coeff_f16, Tensor frame_t_f32, Tensor anchor_offsets_i32, Tensor anchor_owner_i32, Tensor anchor_left_i32, Tensor anchor_right_i32, Tensor track_block_change_offsets_i32, Tensor change_frame_i32, Tensor op_offsets_i32, Tensor op_type_i32, Tensor op_pos_i32, Tensor op_owner_i32, Tensor op_left_i32, Tensor op_right_i32, Tensor site_rgba_f32, Tensor config_i32, Tensor config_f32) -> (Tensor, Tensor, Tensor)");
  m.def(
      "endpoint_record_edit_rgba_depth_replay_trackloop(Tensor boundary_f32, Tensor rays_f32, Tensor frame_t_f32, Tensor base_offsets_i32, Tensor base_owner_i32, Tensor base_left_i32, Tensor base_right_i32, Tensor track_change_offsets_i32, Tensor change_frame_i32, Tensor op_offsets_i32, Tensor op_type_i32, Tensor op_pos_i32, Tensor op_owner_i32, Tensor op_left_i32, Tensor op_right_i32, Tensor site_rgba_f32, Tensor config_i32, Tensor config_f32) -> (Tensor, Tensor, Tensor)");
  m.def(
      "endpoint_record_edit_rgba_depth_replay_framegroup16(Tensor boundary_f32, Tensor rays_f32, Tensor frame_t_f32, Tensor base_offsets_i32, Tensor base_owner_i32, Tensor base_left_i32, Tensor base_right_i32, Tensor track_change_offsets_i32, Tensor change_frame_i32, Tensor op_offsets_i32, Tensor op_type_i32, Tensor op_pos_i32, Tensor op_owner_i32, Tensor op_left_i32, Tensor op_right_i32, Tensor site_rgba_f32, Tensor config_i32, Tensor config_f32) -> (Tensor, Tensor, Tensor)");
  m.def(
      "endpoint_record_edit_vjp_direct_atomic_grad_only(Tensor boundary_f32, Tensor rays_f32, Tensor frame_t_f32, Tensor base_offsets_i32, Tensor base_owner_i32, Tensor base_left_i32, Tensor base_right_i32, Tensor track_change_offsets_i32, Tensor change_frame_i32, Tensor op_offsets_i32, Tensor op_type_i32, Tensor op_pos_i32, Tensor op_owner_i32, Tensor op_left_i32, Tensor op_right_i32, Tensor site_rgba_f32, Tensor grad_rgb_f32, Tensor grad_alpha_f32, Tensor grad_depth_f32, Tensor config_i32, Tensor config_f32) -> Tensor");
  m.def(
      "endpoint_record_edit_vjp_direct_atomic_rgb_only(Tensor boundary_f32, Tensor rays_f32, Tensor frame_t_f32, Tensor base_offsets_i32, Tensor base_owner_i32, Tensor base_left_i32, Tensor base_right_i32, Tensor track_change_offsets_i32, Tensor change_frame_i32, Tensor op_offsets_i32, Tensor op_type_i32, Tensor op_pos_i32, Tensor op_owner_i32, Tensor op_left_i32, Tensor op_right_i32, Tensor site_rgba_f32, Tensor grad_rgb_f32, Tensor config_i32, Tensor config_f32) -> Tensor");
  m.def(
      "endpoint_record_edit_mse_vjp_direct_atomic_rgb_only(Tensor boundary_f32, Tensor rays_f32, Tensor frame_t_f32, Tensor base_offsets_i32, Tensor base_owner_i32, Tensor base_left_i32, Tensor base_right_i32, Tensor track_change_offsets_i32, Tensor change_frame_i32, Tensor op_offsets_i32, Tensor op_type_i32, Tensor op_pos_i32, Tensor op_owner_i32, Tensor op_left_i32, Tensor op_right_i32, Tensor site_rgba_f32, Tensor target_rgb_f32, Tensor config_i32, Tensor config_f32) -> (Tensor, Tensor)");
  m.def(
      "endpoint_record_edit_coeff16_mse_vjp_direct_atomic_rgb_only(Tensor coeff_f16, Tensor frame_t_f32, Tensor base_offsets_i32, Tensor base_owner_i32, Tensor base_left_i32, Tensor base_right_i32, Tensor track_change_offsets_i32, Tensor change_frame_i32, Tensor op_offsets_i32, Tensor op_type_i32, Tensor op_pos_i32, Tensor op_owner_i32, Tensor op_left_i32, Tensor op_right_i32, Tensor site_rgba_f32, Tensor target_rgb_f32, Tensor config_i32, Tensor config_f32) -> (Tensor, Tensor)");
  m.def(
      "endpoint_record_edit_block4_vjp_direct_atomic_rgb_only(Tensor boundary_f32, Tensor rays_f32, Tensor frame_t_f32, Tensor anchor_offsets_i32, Tensor anchor_owner_i32, Tensor anchor_left_i32, Tensor anchor_right_i32, Tensor track_block_change_offsets_i32, Tensor change_frame_i32, Tensor op_offsets_i32, Tensor op_type_i32, Tensor op_pos_i32, Tensor op_owner_i32, Tensor op_left_i32, Tensor op_right_i32, Tensor site_rgba_f32, Tensor grad_rgb_f32, Tensor config_i32, Tensor config_f32) -> Tensor");
  m.def(
      "endpoint_record_edit_block_coeff_vjp_direct_atomic_rgb_only(Tensor coeff_f32, Tensor frame_t_f32, Tensor anchor_offsets_i32, Tensor anchor_owner_i32, Tensor anchor_left_i32, Tensor anchor_right_i32, Tensor track_block_change_offsets_i32, Tensor change_frame_i32, Tensor op_offsets_i32, Tensor op_type_i32, Tensor op_pos_i32, Tensor op_owner_i32, Tensor op_left_i32, Tensor op_right_i32, Tensor site_rgba_f32, Tensor grad_rgb_f32, Tensor config_i32, Tensor config_f32) -> Tensor");
  m.def(
      "endpoint_record_edit_block_coeff_mse_vjp_direct_atomic_rgb_only(Tensor coeff_f32, Tensor frame_t_f32, Tensor anchor_offsets_i32, Tensor anchor_owner_i32, Tensor anchor_left_i32, Tensor anchor_right_i32, Tensor track_block_change_offsets_i32, Tensor change_frame_i32, Tensor op_offsets_i32, Tensor op_type_i32, Tensor op_pos_i32, Tensor op_owner_i32, Tensor op_left_i32, Tensor op_right_i32, Tensor site_rgba_f32, Tensor target_rgb_f32, Tensor config_i32, Tensor config_f32) -> (Tensor, Tensor)");
  m.def(
      "endpoint_record_edit_block_coeff16_mse_vjp_direct_atomic_rgb_only(Tensor coeff_f16, Tensor frame_t_f32, Tensor anchor_offsets_i32, Tensor anchor_owner_i32, Tensor anchor_left_i32, Tensor anchor_right_i32, Tensor track_block_change_offsets_i32, Tensor change_frame_i32, Tensor op_offsets_i32, Tensor op_type_i32, Tensor op_pos_i32, Tensor op_owner_i32, Tensor op_left_i32, Tensor op_right_i32, Tensor site_rgba_f32, Tensor target_rgb_f32, Tensor config_i32, Tensor config_f32) -> (Tensor, Tensor)");
  m.def(
      "endpoint_record_edit_block_coeff16_packed_mse_vjp_direct_atomic_rgb_only(Tensor coeff_f16, Tensor frame_t_f32, Tensor anchor_offsets_i32, Tensor anchor_record_i32, Tensor track_block_change_offsets_i32, Tensor change_frame_i32, Tensor op_offsets_i32, Tensor op_type_i32, Tensor op_pos_i32, Tensor op_record_i32, Tensor site_rgba_f32, Tensor target_rgb_f32, Tensor config_i32, Tensor config_f32) -> (Tensor, Tensor)");
  m.def(
      "endpoint_record_edit_block_coeff16_i16_mse_vjp_direct_atomic_rgb_only(Tensor coeff_f16, Tensor frame_t_f32, Tensor anchor_offsets_i32, Tensor anchor_owner_i16, Tensor anchor_left_i16, Tensor anchor_right_i16, Tensor track_block_change_offsets_i32, Tensor change_frame_i32, Tensor op_offsets_i32, Tensor op_type_i32, Tensor op_pos_i32, Tensor op_owner_i16, Tensor op_left_i16, Tensor op_right_i16, Tensor site_rgba_f32, Tensor target_rgb_f32, Tensor config_i32, Tensor config_f32) -> (Tensor, Tensor)");
  m.def(
      "endpoint_record_edit_block_coeff16_i16x3_mse_vjp_direct_atomic_rgb_only(Tensor coeff_f16, Tensor frame_t_f32, Tensor anchor_offsets_i32, Tensor anchor_record_i16, Tensor track_block_change_offsets_i32, Tensor change_frame_i32, Tensor op_offsets_i32, Tensor op_type_i32, Tensor op_pos_i32, Tensor op_record_i16, Tensor site_rgba_f32, Tensor target_rgb_f32, Tensor config_i32, Tensor config_f32) -> (Tensor, Tensor)");
  m.def(
      "endpoint_record_edit_block_coeff16_vjp_direct_atomic_rgb_only(Tensor coeff_f16, Tensor frame_t_f32, Tensor anchor_offsets_i32, Tensor anchor_owner_i32, Tensor anchor_left_i32, Tensor anchor_right_i32, Tensor track_block_change_offsets_i32, Tensor change_frame_i32, Tensor op_offsets_i32, Tensor op_type_i32, Tensor op_pos_i32, Tensor op_owner_i32, Tensor op_left_i32, Tensor op_right_i32, Tensor site_rgba_f32, Tensor grad_rgb_f32, Tensor config_i32, Tensor config_f32) -> Tensor");
}

TORCH_LIBRARY_IMPL(world_foam_lane2_fused_slab_v0, CompositeExplicitAutograd, m) {
  m.impl("pack_endpoint_records_i32_cpu", world_foam_lane2_fused_slab::pack_endpoint_records_i32_cpu);
  m.impl("gate4_delta_replace_from_cuts_cpu", world_foam_lane2_fused_slab::gate4_delta_replace_from_cuts_cpu);
  m.impl(
      "gate4_owner_run_delta_replace_from_rays_cpu",
      world_foam_lane2_fused_slab::gate4_owner_run_delta_replace_from_rays_cpu);
  m.impl(
      "gate4_delta_replace_packed_from_cuts_cpu",
      world_foam_lane2_fused_slab::gate4_delta_replace_packed_from_cuts_cpu);
  m.impl("gate4_cut_arrays_from_sorted_cpu", world_foam_lane2_fused_slab::gate4_cut_arrays_from_sorted_cpu);
  m.impl("gate4_delta_replace_from_sorted_cpu", world_foam_lane2_fused_slab::gate4_delta_replace_from_sorted_cpu);
  m.impl(
      "gate4_delta_replace_packed_from_sorted_cpu",
      world_foam_lane2_fused_slab::gate4_delta_replace_packed_from_sorted_cpu);
  m.impl(
      "gate4_delta_replace_packed_from_coeff_csr_cpu",
      world_foam_lane2_fused_slab::gate4_delta_replace_packed_from_coeff_csr_cpu);
  m.impl("count_power_boundary_events", world_foam_lane2_fused_slab::count_power_boundary_events_dispatch);
  m.impl("shared_signal_replay", world_foam_lane2_fused_slab::shared_signal_replay_dispatch);
  m.impl("shared_rgb_replay", world_foam_lane2_fused_slab::shared_rgb_replay_dispatch);
  m.impl("shared_rgba_depth_replay", world_foam_lane2_fused_slab::shared_rgba_depth_replay_dispatch);
  m.impl("shared_rgba_depth_vjp", world_foam_lane2_fused_slab::shared_rgba_depth_vjp_dispatch);
  m.impl("realray_rgba_depth_replay", world_foam_lane2_fused_slab::realray_rgba_depth_replay_dispatch);
  m.impl("shared_realray_rgba_depth_replay", world_foam_lane2_fused_slab::shared_realray_rgba_depth_replay_dispatch);
  m.impl("shared_realray_rgba_depth_vjp", world_foam_lane2_fused_slab::shared_realray_rgba_depth_vjp_dispatch);
  m.impl("shared_realray_rgba_depth_vjp_reduce", world_foam_lane2_fused_slab::shared_realray_rgba_depth_vjp_reduce_dispatch);
  m.impl(
      "shared_realray_rgba_depth_vjp_reduce_csr",
      world_foam_lane2_fused_slab::shared_realray_rgba_depth_vjp_reduce_csr_dispatch);
  m.impl(
      "fused_slab_affine_realray_rgba_depth_replay",
      world_foam_lane2_fused_slab::fused_slab_affine_realray_rgba_depth_replay_dispatch);
  m.impl(
      "fused_slab_affine_coeff_realray_rgba_depth_replay",
      world_foam_lane2_fused_slab::fused_slab_affine_coeff_realray_rgba_depth_replay_dispatch);
  m.impl(
      "fused_slab_affine_coeff16_realray_rgba_depth_replay",
      world_foam_lane2_fused_slab::fused_slab_affine_coeff16_realray_rgba_depth_replay_dispatch);
  m.impl(
      "fused_slab_affine_num32_den16_realray_rgba_depth_replay",
      world_foam_lane2_fused_slab::fused_slab_affine_num32_den16_realray_rgba_depth_replay_dispatch);
  m.impl(
      "fused_slab_affine_num32_den16_ownerupdate_realray_rgba_depth_replay",
      world_foam_lane2_fused_slab::fused_slab_affine_num32_den16_ownerupdate_realray_rgba_depth_replay_dispatch);
  m.impl(
      "fused_slab_affine_num32_den16_vjp_reduce",
      world_foam_lane2_fused_slab::fused_slab_affine_num32_den16_vjp_reduce_dispatch);
  m.impl(
      "fused_slab_affine_num32_den16_vjp_direct_atomic",
      world_foam_lane2_fused_slab::fused_slab_affine_num32_den16_vjp_direct_atomic_dispatch);
  m.impl(
      "fused_slab_affine_num32_den16_vjp_direct_atomic_grad_only",
      world_foam_lane2_fused_slab::fused_slab_affine_num32_den16_vjp_direct_atomic_grad_only_dispatch);
  m.impl(
      "fused_slab_affine_num32_den16_vjp_direct_atomic_grad_only_ownerupdate",
      world_foam_lane2_fused_slab::fused_slab_affine_num32_den16_vjp_direct_atomic_grad_only_ownerupdate_dispatch);
  m.impl(
      "fused_slab_affine_num32_den16_vjp_direct_atomic_rgb_only",
      world_foam_lane2_fused_slab::fused_slab_affine_num32_den16_vjp_direct_atomic_rgb_only_dispatch);
  m.impl(
      "fused_slab_affine_num32_den16_mse_vjp_direct_atomic_rgb_only",
      world_foam_lane2_fused_slab::fused_slab_affine_num32_den16_mse_vjp_direct_atomic_rgb_only_dispatch);
  m.impl(
      "fused_slab_affine_num32_den16_mse_vjp_direct_atomic_track_rgb_only",
      world_foam_lane2_fused_slab::fused_slab_affine_num32_den16_mse_vjp_direct_atomic_track_rgb_only_dispatch);
  m.impl(
      "fused_slab_affine_coeff16_mse_vjp_direct_atomic_rgb_only",
      world_foam_lane2_fused_slab::fused_slab_affine_coeff16_mse_vjp_direct_atomic_rgb_only_dispatch);
  m.impl(
      "fused_slab_affine_coeff16_cap224_mse_vjp_direct_atomic_rgb_only",
      world_foam_lane2_fused_slab::fused_slab_affine_coeff16_cap224_mse_vjp_direct_atomic_rgb_only_dispatch);
  m.impl(
      "fused_slab_affine_coeff16_densitymask_mse_vjp_direct_atomic_rgb_only",
      world_foam_lane2_fused_slab::fused_slab_affine_coeff16_densitymask_mse_vjp_direct_atomic_rgb_only_dispatch);
  m.impl(
      "fused_slab_affine_coeff16_mse_vjp_direct_atomic_sample_reduce_rgb_only",
      world_foam_lane2_fused_slab::fused_slab_affine_coeff16_mse_vjp_direct_atomic_sample_reduce_rgb_only_dispatch);
  m.impl(
      "fused_slab_affine_coeff16_sortnet_mse_vjp_direct_atomic_rgb_only",
      world_foam_lane2_fused_slab::fused_slab_affine_coeff16_sortnet_mse_vjp_direct_atomic_rgb_only_dispatch);
  m.impl(
      "fused_slab_affine_coeff16_mse_vjp_direct_atomic_sitecache_rgb_only",
      world_foam_lane2_fused_slab::fused_slab_affine_coeff16_mse_vjp_direct_atomic_sitecache_rgb_only_dispatch);
  m.impl(
      "fused_slab_affine_coeff16_mse_vjp_direct_atomic_framegroup16_cached_rgb_only",
      world_foam_lane2_fused_slab::fused_slab_affine_coeff16_mse_vjp_direct_atomic_framegroup16_cached_rgb_only_dispatch);
  m.impl(
      "fused_slab_affine_coeff16_ownerupdate_mse_vjp_direct_atomic_rgb_only",
      world_foam_lane2_fused_slab::fused_slab_affine_coeff16_ownerupdate_mse_vjp_direct_atomic_rgb_only_dispatch);
  m.impl(
      "fused_slab_affine_coeff16_ownerupdate_i16_mse_vjp_direct_atomic_rgb_only",
      world_foam_lane2_fused_slab::fused_slab_affine_coeff16_ownerupdate_i16_mse_vjp_direct_atomic_rgb_only_dispatch);
  m.impl(
      "fused_slab_affine_coeff16_ownerkeep_i16_mse_vjp_direct_atomic_rgb_only",
      world_foam_lane2_fused_slab::fused_slab_affine_coeff16_ownerkeep_i16_mse_vjp_direct_atomic_rgb_only_dispatch);
  m.impl(
      "fused_slab_affine_coeff16_ownerkeep_mse_vjp_direct_atomic_rgb_only",
      world_foam_lane2_fused_slab::fused_slab_affine_coeff16_ownerkeep_mse_vjp_direct_atomic_rgb_only_dispatch);
  m.impl(
      "fused_slab_affine_coeff16_mse_vjp_direct_atomic_track_rgb_only",
      world_foam_lane2_fused_slab::fused_slab_affine_coeff16_mse_vjp_direct_atomic_track_rgb_only_dispatch);
  m.impl(
      "fused_slab_affine_num32_den16_vjp_direct_atomic_track",
      world_foam_lane2_fused_slab::fused_slab_affine_num32_den16_vjp_direct_atomic_track_dispatch);
  m.impl(
      "segment_tape_rgba_depth_replay",
      world_foam_lane2_fused_slab::segment_tape_rgba_depth_replay_dispatch);
	  m.impl(
	      "segment_tape_vjp_direct_atomic_grad_only",
	      world_foam_lane2_fused_slab::segment_tape_vjp_direct_atomic_grad_only_dispatch);
	  m.impl(
	      "segment_tape_mse_vjp_direct_atomic_rgb_only",
	      world_foam_lane2_fused_slab::segment_tape_mse_vjp_direct_atomic_rgb_only_dispatch);
	  m.impl(
	      "segment_tape_vjp_direct_atomic_track",
	      world_foam_lane2_fused_slab::segment_tape_vjp_direct_atomic_track_dispatch);
  m.impl(
      "endpoint_run_rgba_depth_replay",
      world_foam_lane2_fused_slab::endpoint_run_rgba_depth_replay_dispatch);
  m.impl(
      "endpoint_run_vjp_direct_atomic_grad_only",
      world_foam_lane2_fused_slab::endpoint_run_vjp_direct_atomic_grad_only_dispatch);
  m.impl(
      "endpoint_run_mse_vjp_direct_atomic_rgb_only",
      world_foam_lane2_fused_slab::endpoint_run_mse_vjp_direct_atomic_rgb_only_dispatch);
  m.impl(
      "endpoint_delta_replace_rgba_depth_replay",
      world_foam_lane2_fused_slab::endpoint_delta_replace_rgba_depth_replay_dispatch);
  m.impl(
      "endpoint_delta_replace_vjp_direct_atomic_grad_only",
      world_foam_lane2_fused_slab::endpoint_delta_replace_vjp_direct_atomic_grad_only_dispatch);
  m.impl(
      "endpoint_record_delta_replace_rgba_depth_replay",
      world_foam_lane2_fused_slab::endpoint_record_delta_replace_rgba_depth_replay_dispatch);
  m.impl(
      "endpoint_record_delta_replace_vjp_direct_atomic_grad_only",
      world_foam_lane2_fused_slab::endpoint_record_delta_replace_vjp_direct_atomic_grad_only_dispatch);
  m.impl(
      "endpoint_record_delta_replace_coeff16_mse_vjp_direct_atomic_rgb_only",
      world_foam_lane2_fused_slab::endpoint_record_delta_replace_coeff16_mse_vjp_direct_atomic_rgb_only_dispatch);
  m.impl(
      "endpoint_record_delta_replace_coeff16_i16x3_mse_vjp_direct_atomic_rgb_only",
      world_foam_lane2_fused_slab::endpoint_record_delta_replace_coeff16_i16x3_mse_vjp_direct_atomic_rgb_only_dispatch);
  m.impl(
      "endpoint_record_delta_replace_coeff16_i16x3_framegroup16_mse_vjp_direct_atomic_rgb_only",
      world_foam_lane2_fused_slab::
          endpoint_record_delta_replace_coeff16_i16x3_framegroup16_mse_vjp_direct_atomic_rgb_only_dispatch);
  m.impl(
      "endpoint_record_delta_replace_coeff16_i16x3_framegroup16_ownerreduce_mse_vjp_direct_atomic_rgb_only",
      world_foam_lane2_fused_slab::
          endpoint_record_delta_replace_coeff16_i16x3_framegroup16_ownerreduce_mse_vjp_direct_atomic_rgb_only_dispatch);
  m.impl(
      "endpoint_record_delta_replace_coeff16_i16cols_framegroup16_mse_vjp_direct_atomic_rgb_only",
      world_foam_lane2_fused_slab::
          endpoint_record_delta_replace_coeff16_i16cols_framegroup16_mse_vjp_direct_atomic_rgb_only_dispatch);
  m.impl(
      "endpoint_record_delta_replace_coeff16_i16x3_framegroup64_mse_vjp_direct_atomic_rgb_only",
      world_foam_lane2_fused_slab::
          endpoint_record_delta_replace_coeff16_i16x3_framegroup64_mse_vjp_direct_atomic_rgb_only_dispatch);
  m.impl(
      "endpoint_record_delta_replace_coeff16_packed_mse_vjp_direct_atomic_rgb_only",
      world_foam_lane2_fused_slab::
          endpoint_record_delta_replace_coeff16_packed_mse_vjp_direct_atomic_rgb_only_dispatch);
  m.impl(
      "endpoint_record_delta_replace_coeff16_packed_framegroup16_mse_vjp_direct_atomic_rgb_only",
      world_foam_lane2_fused_slab::
          endpoint_record_delta_replace_coeff16_packed_framegroup16_mse_vjp_direct_atomic_rgb_only_dispatch);
  m.impl(
      "endpoint_record_delta_replace_coeff16_packed_framegroup16_launch_only_mse_vjp_direct_atomic_rgb_only",
      world_foam_lane2_fused_slab::
          endpoint_record_delta_replace_coeff16_packed_framegroup16_launch_only_mse_vjp_direct_atomic_rgb_only_dispatch);
  m.impl(
      "endpoint_record_delta_replace_coeff16_packed_framegroup16_unchecked_launch_only_mse_vjp_direct_atomic_rgb_only",
      world_foam_lane2_fused_slab::
          endpoint_record_delta_replace_coeff16_packed_framegroup16_unchecked_launch_only_mse_vjp_direct_atomic_rgb_only_dispatch);
  m.impl(
      "endpoint_record_delta_replace_coeff16_packed_framegroup16_reduce32_launch_only_mse_vjp_direct_atomic_rgb_only",
      world_foam_lane2_fused_slab::
          endpoint_record_delta_replace_coeff16_packed_framegroup16_reduce32_launch_only_mse_vjp_direct_atomic_rgb_only_dispatch);
  m.impl(
      "endpoint_record_delta_replace_coeff16_packed_framegroup16_reduce32_unchecked_launch_only_mse_vjp_direct_atomic_rgb_only",
      world_foam_lane2_fused_slab::
          endpoint_record_delta_replace_coeff16_packed_framegroup16_reduce32_unchecked_launch_only_mse_vjp_direct_atomic_rgb_only_dispatch);
  m.impl(
      "endpoint_record_delta_replace_coeff16_packed_framegroup16_rowselect32_launch_only_mse_vjp_direct_atomic_rgb_only",
      world_foam_lane2_fused_slab::
          endpoint_record_delta_replace_coeff16_packed_framegroup16_rowselect32_launch_only_mse_vjp_direct_atomic_rgb_only_dispatch);
  m.impl(
      "endpoint_record_delta_replace_coeff16_packed_framegroup16_rowselect32_unchecked_launch_only_mse_vjp_direct_atomic_rgb_only",
      world_foam_lane2_fused_slab::
          endpoint_record_delta_replace_coeff16_packed_framegroup16_rowselect32_unchecked_launch_only_mse_vjp_direct_atomic_rgb_only_dispatch);
  m.impl(
      "endpoint_record_delta_replace_coeff16_packed_framegroup16_rowdesc_launch_only_mse_vjp_direct_atomic_rgb_only",
      world_foam_lane2_fused_slab::
          endpoint_record_delta_replace_coeff16_packed_framegroup16_rowdesc_launch_only_mse_vjp_direct_atomic_rgb_only_dispatch);
  m.impl(
      "endpoint_record_delta_replace_coeff16_packed_framegroup16_rowdesc_unchecked_launch_only_mse_vjp_direct_atomic_rgb_only",
      world_foam_lane2_fused_slab::
          endpoint_record_delta_replace_coeff16_packed_framegroup16_rowdesc_unchecked_launch_only_mse_vjp_direct_atomic_rgb_only_dispatch);
  m.impl(
      "endpoint_record_delta_replace_coeff16_packed_framegroup16_rowdesc32_launch_only_mse_vjp_direct_atomic_rgb_only",
      world_foam_lane2_fused_slab::
          endpoint_record_delta_replace_coeff16_packed_framegroup16_rowdesc32_launch_only_mse_vjp_direct_atomic_rgb_only_dispatch);
  m.impl(
      "endpoint_record_delta_replace_coeff16_packed_framegroup16_rowdesc32_unchecked_launch_only_mse_vjp_direct_atomic_rgb_only",
      world_foam_lane2_fused_slab::
          endpoint_record_delta_replace_coeff16_packed_framegroup16_rowdesc32_unchecked_launch_only_mse_vjp_direct_atomic_rgb_only_dispatch);
  m.impl(
      "endpoint_record_delta_replace_coeff16_packed_framegroup16_recompute_launch_only_mse_vjp_direct_atomic_rgb_only",
      world_foam_lane2_fused_slab::
          endpoint_record_delta_replace_coeff16_packed_framegroup16_recompute_launch_only_mse_vjp_direct_atomic_rgb_only_dispatch);
  m.impl(
      "endpoint_record_delta_replace_coeff16_packed_framegroup16_smallrun16_launch_only_mse_vjp_direct_atomic_rgb_only",
      world_foam_lane2_fused_slab::
          endpoint_record_delta_replace_coeff16_packed_framegroup16_smallrun16_launch_only_mse_vjp_direct_atomic_rgb_only_dispatch);
  m.impl(
      "endpoint_record_delta_replace_coeff16_packed_framegroup16_materialized_launch_only_mse_vjp_direct_atomic_rgb_only",
      world_foam_lane2_fused_slab::
          endpoint_record_delta_replace_coeff16_packed_framegroup16_materialized_launch_only_mse_vjp_direct_atomic_rgb_only_dispatch);
  m.impl(
      "endpoint_record_delta_replace_coeff16_packed_framegroup16_recompute_mse_vjp_direct_atomic_rgb_only",
      world_foam_lane2_fused_slab::
          endpoint_record_delta_replace_coeff16_packed_framegroup16_recompute_mse_vjp_direct_atomic_rgb_only_dispatch);
  m.impl(
      "endpoint_record_delta_replace_factorized_packed_framegroup16_recompute_mse_vjp_direct_atomic_rgb_only",
      world_foam_lane2_fused_slab::
          endpoint_record_delta_replace_factorized_packed_framegroup16_recompute_mse_vjp_direct_atomic_rgb_only_dispatch);
  m.impl(
      "endpoint_record_delta_replace_factorized_frameselect_recompute_mse_vjp_direct_atomic_rgb_only",
      world_foam_lane2_fused_slab::
          endpoint_record_delta_replace_factorized_frameselect_recompute_mse_vjp_direct_atomic_rgb_only_dispatch);
  m.impl(
      "endpoint_record_delta_replace_factorized_framebitmask_recompute_mse_vjp_direct_atomic_rgb_only",
      world_foam_lane2_fused_slab::
          endpoint_record_delta_replace_factorized_framebitmask_recompute_mse_vjp_direct_atomic_rgb_only_dispatch);
  m.impl(
      "endpoint_record_delta_replace_coeff16_packed_framegroup16_smallrun16_mse_vjp_direct_atomic_rgb_only",
      world_foam_lane2_fused_slab::
          endpoint_record_delta_replace_coeff16_packed_framegroup16_smallrun16_mse_vjp_direct_atomic_rgb_only_dispatch);
  m.impl(
      "endpoint_record_delta_replace_coeff16_packed_framegroup16_materialized_mse_vjp_direct_atomic_rgb_only",
      world_foam_lane2_fused_slab::
          endpoint_record_delta_replace_coeff16_packed_framegroup16_materialized_mse_vjp_direct_atomic_rgb_only_dispatch);
  m.impl(
      "endpoint_record_delta_replace_coeff16_i16x4_framegroup16_mse_vjp_direct_atomic_rgb_only",
      world_foam_lane2_fused_slab::
          endpoint_record_delta_replace_coeff16_i16x4_framegroup16_mse_vjp_direct_atomic_rgb_only_dispatch);
  m.impl(
      "endpoint_record_delta_replace_coeff16_i16x3_framegroup16_materialized_mse_vjp_direct_atomic_rgb_only",
      world_foam_lane2_fused_slab::
          endpoint_record_delta_replace_coeff16_i16x3_framegroup16_materialized_mse_vjp_direct_atomic_rgb_only_dispatch);
  m.impl(
      "endpoint_record_delta_replace_coeff16_i16x4_mse_vjp_direct_atomic_rgb_only",
      world_foam_lane2_fused_slab::endpoint_record_delta_replace_coeff16_i16x4_mse_vjp_direct_atomic_rgb_only_dispatch);
  m.impl(
      "endpoint_record_edit_rgba_depth_replay",
      world_foam_lane2_fused_slab::endpoint_record_edit_rgba_depth_replay_dispatch);
  m.impl(
      "endpoint_record_edit_block4_rgba_depth_replay",
      world_foam_lane2_fused_slab::endpoint_record_edit_block4_rgba_depth_replay_dispatch);
  m.impl(
      "endpoint_record_edit_block_coeff_rgba_depth_replay",
      world_foam_lane2_fused_slab::endpoint_record_edit_block_coeff_rgba_depth_replay_dispatch);
  m.impl(
      "endpoint_record_edit_block_coeff_rgb_replay",
      world_foam_lane2_fused_slab::endpoint_record_edit_block_coeff_rgb_replay_dispatch);
  m.impl(
      "endpoint_record_edit_block_coeff16_rgba_depth_replay",
      world_foam_lane2_fused_slab::endpoint_record_edit_block_coeff16_rgba_depth_replay_dispatch);
  m.impl(
      "endpoint_record_edit_rgba_depth_replay_trackloop",
      world_foam_lane2_fused_slab::endpoint_record_edit_rgba_depth_replay_trackloop_dispatch);
  m.impl(
      "endpoint_record_edit_rgba_depth_replay_framegroup16",
      world_foam_lane2_fused_slab::endpoint_record_edit_rgba_depth_replay_framegroup16_dispatch);
  m.impl(
      "endpoint_record_edit_vjp_direct_atomic_grad_only",
      world_foam_lane2_fused_slab::endpoint_record_edit_vjp_direct_atomic_grad_only_dispatch);
  m.impl(
      "endpoint_record_edit_vjp_direct_atomic_rgb_only",
      world_foam_lane2_fused_slab::endpoint_record_edit_vjp_direct_atomic_rgb_only_dispatch);
  m.impl(
      "endpoint_record_edit_mse_vjp_direct_atomic_rgb_only",
      world_foam_lane2_fused_slab::endpoint_record_edit_mse_vjp_direct_atomic_rgb_only_dispatch);
  m.impl(
      "endpoint_record_edit_coeff16_mse_vjp_direct_atomic_rgb_only",
      world_foam_lane2_fused_slab::endpoint_record_edit_coeff16_mse_vjp_direct_atomic_rgb_only_dispatch);
  m.impl(
      "endpoint_record_edit_block4_vjp_direct_atomic_rgb_only",
      world_foam_lane2_fused_slab::endpoint_record_edit_block4_vjp_direct_atomic_rgb_only_dispatch);
  m.impl(
      "endpoint_record_edit_block_coeff_vjp_direct_atomic_rgb_only",
      world_foam_lane2_fused_slab::endpoint_record_edit_block_coeff_vjp_direct_atomic_rgb_only_dispatch);
  m.impl(
      "endpoint_record_edit_block_coeff_mse_vjp_direct_atomic_rgb_only",
      world_foam_lane2_fused_slab::endpoint_record_edit_block_coeff_mse_vjp_direct_atomic_rgb_only_dispatch);
  m.impl(
      "endpoint_record_edit_block_coeff16_mse_vjp_direct_atomic_rgb_only",
      world_foam_lane2_fused_slab::endpoint_record_edit_block_coeff16_mse_vjp_direct_atomic_rgb_only_dispatch);
  m.impl(
      "endpoint_record_edit_block_coeff16_packed_mse_vjp_direct_atomic_rgb_only",
      world_foam_lane2_fused_slab::endpoint_record_edit_block_coeff16_packed_mse_vjp_direct_atomic_rgb_only_dispatch);
  m.impl(
      "endpoint_record_edit_block_coeff16_i16_mse_vjp_direct_atomic_rgb_only",
      world_foam_lane2_fused_slab::endpoint_record_edit_block_coeff16_i16_mse_vjp_direct_atomic_rgb_only_dispatch);
  m.impl(
      "endpoint_record_edit_block_coeff16_i16x3_mse_vjp_direct_atomic_rgb_only",
      world_foam_lane2_fused_slab::endpoint_record_edit_block_coeff16_i16x3_mse_vjp_direct_atomic_rgb_only_dispatch);
  m.impl(
      "endpoint_record_edit_block_coeff16_vjp_direct_atomic_rgb_only",
      world_foam_lane2_fused_slab::endpoint_record_edit_block_coeff16_vjp_direct_atomic_rgb_only_dispatch);
}

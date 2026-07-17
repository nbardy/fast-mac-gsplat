#!/usr/bin/env python3
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HEADER = ROOT / "csrc" / "shared" / "world_foam_lane2_types.h"
SHADERS = (
    ROOT / "csrc" / "metal" / "world_foam_lane2_event_count.metal",
    ROOT / "csrc" / "metal" / "world_foam_lane2_power_boundary.metal",
    ROOT / "csrc" / "metal" / "world_foam_lane2_power_boundary_tensor.metal",
    ROOT / "csrc" / "metal" / "world_foam_lane2_shared_replay_tensor.metal",
)
REQUIRED_SHARED_REPLAY_KERNELS = (
    "wf2_shared_signal_replay_tensor",
    "wf2_shared_rgb_replay_tensor",
    "wf2_shared_rgba_depth_replay_tensor",
    "wf2_shared_rgba_depth_vjp_tensor",
    "wf2_realray_rgba_depth_replay_tensor",
    "wf2_shared_realray_rgba_depth_replay_tensor",
    "wf2_shared_realray_rgba_depth_vjp_tensor",
    "wf2_shared_realray_rgba_depth_vjp_reduce_tensor",
    "wf2_shared_realray_rgba_depth_vjp_partial_reduce_tensor",
    "wf2_shared_realray_rgba_depth_vjp_partial_reduce_csr_tensor",
    "wf2_shared_realray_rgba_depth_vjp_finalize_reduce_tensor",
)


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)


def validate_host_header(tmpdir: Path) -> None:
    probe = tmpdir / "host_header_probe.cpp"
    probe.write_text(
        f"""
#include <cstddef>
#include <cstdint>
#include \"{HEADER}\"

static_assert(sizeof(WF2Float3) == sizeof(float) * 3);
static_assert(offsetof(WF2ScreenTimeBeam, payload_id) > 0);
static_assert(offsetof(WF2BeamEventCount, total_crossings) > 0);
static_assert(offsetof(WF2BoundaryEvent, uvt) > 0);
static_assert(offsetof(WF2PowerBoundary3D, left_site) > 0);
static_assert(offsetof(WF2PowerBeamSlab, payload_id) > 0);
static_assert(offsetof(WF2PowerBoundaryConfig, camera_velocity_x) > 0);
static_assert(offsetof(WF2PowerBoundaryCount, boundary_event_count) > 0);

int main() {{
  WF2GridConfig config = {{}};
  config.beam_count = 1;
  WF2PowerBoundaryConfig power_config = {{}};
  power_config.boundary_count = 1;
  return config.beam_count == 1 && power_config.boundary_count == 1 ? 0 : 1;
}}
""",
        encoding="utf-8",
    )
    run(["clang++", "-std=c++17", "-fsyntax-only", str(probe)])


def validate_power_boundary_cpu_probe(tmpdir: Path) -> None:
    probe = tmpdir / "power_boundary_probe.cpp"
    binary = tmpdir / "power_boundary_probe"
    probe.write_text(
        f"""
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>
#include \"{HEADER}\"

struct Site {{
  float x;
  float z;
  float t;
  float weight;
}};

static WF2PowerBoundary3D make_boundary(const Site& a, const Site& b, uint32_t left, uint32_t right) {{
  WF2PowerBoundary3D out = {{}};
  out.nx = 2.0f * (b.x - a.x);
  out.nz = 2.0f * (b.z - a.z);
  out.nt = 2.0f * (b.t - a.t);
  out.b = a.x * a.x + a.z * a.z + a.t * a.t
      - b.x * b.x - b.z * b.z - b.t * b.t
      - a.weight + b.weight;
  out.left_site = left;
  out.right_site = right;
  return out;
}}

static bool overlaps(const WF2PowerBoundary3D& boundary, const WF2PowerBeamSlab& beam, float camera_velocity_x) {{
  if (std::fabs(boundary.nz) < 1.0e-7f) {{
    return false;
  }}
  const float x0 = beam.u_center + camera_velocity_x * beam.t0;
  const float x1 = beam.u_center + camera_velocity_x * beam.t1;
  const float s0 = -(boundary.nx * x0 + boundary.nt * beam.t0 + boundary.b) / boundary.nz;
  const float s1 = -(boundary.nx * x1 + boundary.nt * beam.t1 + boundary.b) / boundary.nz;
  const float s_min = std::fmin(s0, s1);
  const float s_max = std::fmax(s0, s1);
  return std::fmax(s_min, beam.near_depth) <= std::fmin(s_max, beam.far_depth);
}}

static uint32_t count_for_velocity(float camera_velocity_x) {{
  const Site sites[] = {{
      {{-0.75f, 0.65f, 0.08f, 0.00f}},
      {{0.10f, 1.05f, 0.28f, 0.04f}},
      {{0.72f, 1.55f, 0.58f, -0.03f}},
      {{-0.18f, 2.15f, 0.88f, 0.02f}},
      {{0.95f, 2.65f, 0.42f, 0.01f}},
  }};
  std::vector<WF2PowerBoundary3D> boundaries;
  for (uint32_t left = 0; left < 5; ++left) {{
    for (uint32_t right = left + 1; right < 5; ++right) {{
      boundaries.push_back(make_boundary(sites[left], sites[right], left, right));
    }}
  }}

  uint32_t total = 0;
  for (uint32_t i = 0; i < 17; ++i) {{
    const float u = -1.0f + (2.0f * float(i) / 16.0f);
    WF2PowerBeamSlab beam = {{}};
    beam.u_center = u;
    beam.t0 = 0.0f;
    beam.t1 = 1.0f;
    beam.near_depth = 0.25f;
    beam.far_depth = 3.0f;
    for (const auto& boundary : boundaries) {{
      total += overlaps(boundary, beam, camera_velocity_x) ? 1u : 0u;
    }}
  }}
  return total;
}}

int main() {{
  const uint32_t slow = count_for_velocity(0.35f);
  const uint32_t fast = count_for_velocity(0.70f);
  return slow == 149u && fast == 151u ? 0 : 1;
}}
""",
        encoding="utf-8",
    )
    run(["clang++", "-std=c++17", str(probe), "-o", str(binary)])
    run([str(binary)])


def validate_metal_shader(tmpdir: Path) -> None:
    xcrun = shutil.which("xcrun")
    if xcrun is None:
        print("skip: xcrun not found; host header validation passed")
        return

    metal_find = subprocess.run(
        [xcrun, "-sdk", "macosx", "-find", "metal"],
        check=False,
        capture_output=True,
        text=True,
    )
    if metal_find.returncode != 0:
        print("skip: xcrun could not find the Metal compiler; host header validation passed")
        return

    for shader in SHADERS:
        air = tmpdir / f"{shader.stem}.air"
        run([
            xcrun,
            "-sdk",
            "macosx",
            "metal",
            "-std=macos-metal2.4",
            "-c",
            str(shader),
            "-o",
            str(air),
        ])


def validate_expected_kernel_names() -> None:
    replay_source = (ROOT / "csrc" / "metal" / "world_foam_lane2_shared_replay_tensor.metal").read_text(
        encoding="utf-8"
    )
    for kernel_name in REQUIRED_SHARED_REPLAY_KERNELS:
        if kernel_name not in replay_source:
            raise RuntimeError(f"missing expected shared replay kernel {kernel_name}")


def main() -> int:
    if not HEADER.exists():
        raise FileNotFoundError(HEADER)
    for shader in SHADERS:
        if not shader.exists():
            raise FileNotFoundError(shader)
    validate_expected_kernel_names()

    with tempfile.TemporaryDirectory(prefix="wf2_static_validate_") as tmp:
        tmpdir = Path(tmp)
        validate_host_header(tmpdir)
        validate_power_boundary_cpu_probe(tmpdir)
        validate_metal_shader(tmpdir)

    print("world_foam_lane2_fused_slab_v0 static validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

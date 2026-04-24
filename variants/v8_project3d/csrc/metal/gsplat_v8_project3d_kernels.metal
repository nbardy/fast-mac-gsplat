#include <metal_stdlib>
using namespace metal;

#ifndef GSP_TILE_SIZE
#define GSP_TILE_SIZE 16u
#endif
#ifndef GSP_THREADS
#define GSP_THREADS 256u
#endif
#ifndef GSP_FAST_CAP
#define GSP_FAST_CAP 2048u
#endif
#ifndef GSP_CHUNK
#define GSP_CHUNK 64u
#endif
#ifndef GSP_SIMD_WIDTH
#define GSP_SIMD_WIDTH 32u
#endif
#ifndef GSP_SIMDGROUPS
#define GSP_SIMDGROUPS 8u
#endif

struct MetaI32 {
  int height;
  int width;
  int tiles_y;
  int tiles_x;
  int tile_size;
  int gaussians;
  int tile_count;
  int max_fast_pairs;
  int batch_size;
  int gaussians_per_batch;
  int tiles_per_image;
  int reserved;
};

struct MetaF32 {
  float alpha_threshold;
  float transmittance_threshold;
  float bg_r;
  float bg_g;
  float bg_b;
  float eps;
  float max_alpha;
};

struct ProjectMetaF32 {
  float near_plane;
};

inline float sym_bilinear(
    float3 u,
    float c00,
    float c01,
    float c02,
    float c11,
    float c12,
    float c22,
    float3 v) {
  return u.x * (c00 * v.x + c01 * v.y + c02 * v.z) +
         u.y * (c01 * v.x + c11 * v.y + c12 * v.z) +
         u.z * (c02 * v.x + c12 * v.y + c22 * v.z);
}

kernel void project_pinhole_forward(
    const device float* means3d [[buffer(0)]],
    const device float* scales [[buffer(1)]],
    const device float* quats [[buffer(2)]],
    const device float* opacities [[buffer(3)]],
    const device float* camera_to_world [[buffer(4)]],
    const device float* camera_params [[buffer(5)]],
    constant MetaI32& mi [[buffer(6)]],
    constant ProjectMetaF32& pm [[buffer(7)]],
    device float2* out_means2d [[buffer(8)]],
    device float* out_conics [[buffer(9)]],
    device float* out_opacities [[buffer(10)]],
    device float* out_depths [[buffer(11)]],
    uint gid [[thread_position_in_grid]]) {
  if (gid >= uint(mi.gaussians)) return;

  uint g3 = gid * 3u;
  uint g4 = gid * 4u;
  uint batch = gid / uint(mi.gaussians_per_batch);
  uint cb = batch * 16u;
  uint cp = batch * 4u;

  float mx = means3d[g3 + 0u];
  float my = means3d[g3 + 1u];
  float mz = means3d[g3 + 2u];

  float sx2 = scales[g3 + 0u] * scales[g3 + 0u];
  float sy2 = scales[g3 + 1u] * scales[g3 + 1u];
  float sz2 = scales[g3 + 2u] * scales[g3 + 2u];

  float qr = quats[g4 + 0u];
  float qi = quats[g4 + 1u];
  float qj = quats[g4 + 2u];
  float qk = quats[g4 + 3u];

  float gr00 = 1.0f - 2.0f * (qj * qj + qk * qk);
  float gr01 = 2.0f * (qi * qj - qr * qk);
  float gr02 = 2.0f * (qi * qk + qr * qj);
  float gr10 = 2.0f * (qi * qj + qr * qk);
  float gr11 = 1.0f - 2.0f * (qi * qi + qk * qk);
  float gr12 = 2.0f * (qj * qk - qr * qi);
  float gr20 = 2.0f * (qi * qk - qr * qj);
  float gr21 = 2.0f * (qj * qk + qr * qi);
  float gr22 = 1.0f - 2.0f * (qi * qi + qj * qj);

  float c00 = gr00 * gr00 * sx2 + gr01 * gr01 * sy2 + gr02 * gr02 * sz2;
  float c01 = gr00 * gr10 * sx2 + gr01 * gr11 * sy2 + gr02 * gr12 * sz2;
  float c02 = gr00 * gr20 * sx2 + gr01 * gr21 * sy2 + gr02 * gr22 * sz2;
  float c11 = gr10 * gr10 * sx2 + gr11 * gr11 * sy2 + gr12 * gr12 * sz2;
  float c12 = gr10 * gr20 * sx2 + gr11 * gr21 * sy2 + gr12 * gr22 * sz2;
  float c22 = gr20 * gr20 * sx2 + gr21 * gr21 * sy2 + gr22 * gr22 * sz2;

  float r00 = camera_to_world[cb + 0u];
  float r01 = camera_to_world[cb + 1u];
  float r02 = camera_to_world[cb + 2u];
  float tx = camera_to_world[cb + 3u];
  float r10 = camera_to_world[cb + 4u];
  float r11 = camera_to_world[cb + 5u];
  float r12 = camera_to_world[cb + 6u];
  float ty = camera_to_world[cb + 7u];
  float r20 = camera_to_world[cb + 8u];
  float r21 = camera_to_world[cb + 9u];
  float r22 = camera_to_world[cb + 10u];
  float tz = camera_to_world[cb + 11u];

  float dx = mx - tx;
  float dy = my - ty;
  float dz = mz - tz;
  float x = dx * r00 + dy * r10 + dz * r20;
  float y = dx * r01 + dy * r11 + dz * r21;
  float z = dx * r02 + dy * r12 + dz * r22;

  float3 col0 = float3(r00, r10, r20);
  float3 col1 = float3(r01, r11, r21);
  float3 col2 = float3(r02, r12, r22);
  float cc00 = sym_bilinear(col0, c00, c01, c02, c11, c12, c22, col0);
  float cc01 = sym_bilinear(col0, c00, c01, c02, c11, c12, c22, col1);
  float cc02 = sym_bilinear(col0, c00, c01, c02, c11, c12, c22, col2);
  float cc11 = sym_bilinear(col1, c00, c01, c02, c11, c12, c22, col1);
  float cc12 = sym_bilinear(col1, c00, c01, c02, c11, c12, c22, col2);
  float cc22 = sym_bilinear(col2, c00, c01, c02, c11, c12, c22, col2);

  float fx = camera_params[cp + 0u];
  float fy = camera_params[cp + 1u];
  float cx = camera_params[cp + 2u];
  float cy = camera_params[cp + 3u];

  bool front = z > pm.near_plane;
  float z_safe = front ? max(z, pm.near_plane) : 1.0f;
  float x_project = front ? x : 0.0f;
  float y_project = front ? y : 0.0f;

  float j00 = fx / z_safe;
  float j02 = -(fx * x_project) / (z_safe * z_safe);
  float j11 = fy / z_safe;
  float j12 = -(fy * y_project) / (z_safe * z_safe);

  float cov00 = j00 * j00 * cc00 + 2.0f * j00 * j02 * cc02 + j02 * j02 * cc22 + 0.3f;
  float cov01 = j00 * j11 * cc01 + j00 * j12 * cc02 + j02 * j11 * cc12 + j02 * j12 * cc22;
  float cov11 = j11 * j11 * cc11 + 2.0f * j11 * j12 * cc12 + j12 * j12 * cc22 + 0.3f;
  float det = max(cov00 * cov11 - cov01 * cov01, 1e-6f);

  out_means2d[gid] = float2((fx * x_project) / z_safe + cx, (fy * y_project) / z_safe + cy);
  out_conics[g3 + 0u] = cov11 / det;
  out_conics[g3 + 1u] = -cov01 / det;
  out_conics[g3 + 2u] = cov00 / det;
  out_opacities[gid] = front ? opacities[gid] : 0.0f;
  out_depths[gid] = z;
}

inline void sym_bilinear_backward(
    float g,
    float3 u,
    float c00,
    float c01,
    float c02,
    float c11,
    float c12,
    float c22,
    float3 v,
    thread float& gc00,
    thread float& gc01,
    thread float& gc02,
    thread float& gc11,
    thread float& gc12,
    thread float& gc22,
    thread float3& gu,
    thread float3& gv) {
  gc00 += g * u.x * v.x;
  gc01 += g * (u.x * v.y + u.y * v.x);
  gc02 += g * (u.x * v.z + u.z * v.x);
  gc11 += g * u.y * v.y;
  gc12 += g * (u.y * v.z + u.z * v.y);
  gc22 += g * u.z * v.z;

  float3 cv = float3(
      c00 * v.x + c01 * v.y + c02 * v.z,
      c01 * v.x + c11 * v.y + c12 * v.z,
      c02 * v.x + c12 * v.y + c22 * v.z);
  float3 cu = float3(
      c00 * u.x + c01 * u.y + c02 * u.z,
      c01 * u.x + c11 * u.y + c12 * u.z,
      c02 * u.x + c12 * u.y + c22 * u.z);
  gu += g * cv;
  gv += g * cu;
}

inline void atomic_add_float(device atomic_float* base, uint idx, float value) {
  atomic_fetch_add_explicit(&base[idx], value, memory_order_relaxed);
}

kernel void project_pinhole_backward(
    const device float2* grad_means2d [[buffer(0)]],
    const device float* grad_conics [[buffer(1)]],
    const device float* grad_projected_opacities [[buffer(2)]],
    const device float* grad_depths [[buffer(3)]],
    const device float* means3d [[buffer(4)]],
    const device float* scales [[buffer(5)]],
    const device float* quats [[buffer(6)]],
    const device float* opacities [[buffer(7)]],
    const device float* camera_to_world [[buffer(8)]],
    const device float* camera_params [[buffer(9)]],
    constant MetaI32& mi [[buffer(10)]],
    constant ProjectMetaF32& pm [[buffer(11)]],
    device float* grad_means3d [[buffer(12)]],
    device float* grad_scales [[buffer(13)]],
    device float* grad_quats [[buffer(14)]],
    device float* grad_opacities [[buffer(15)]],
    device atomic_float* grad_camera_to_world [[buffer(16)]],
    device atomic_float* grad_camera_params [[buffer(17)]],
    uint gid [[thread_position_in_grid]]) {
  if (gid >= uint(mi.gaussians)) return;

  uint g3 = gid * 3u;
  uint g4 = gid * 4u;
  uint batch = gid / uint(mi.gaussians_per_batch);
  uint cb = batch * 16u;
  uint cp = batch * 4u;

  float mx = means3d[g3 + 0u];
  float my = means3d[g3 + 1u];
  float mz = means3d[g3 + 2u];
  float sx = scales[g3 + 0u];
  float sy = scales[g3 + 1u];
  float sz = scales[g3 + 2u];
  float sx2 = sx * sx;
  float sy2 = sy * sy;
  float sz2 = sz * sz;

  float qr = quats[g4 + 0u];
  float qi = quats[g4 + 1u];
  float qj = quats[g4 + 2u];
  float qk = quats[g4 + 3u];

  float gr00 = 1.0f - 2.0f * (qj * qj + qk * qk);
  float gr01 = 2.0f * (qi * qj - qr * qk);
  float gr02 = 2.0f * (qi * qk + qr * qj);
  float gr10 = 2.0f * (qi * qj + qr * qk);
  float gr11 = 1.0f - 2.0f * (qi * qi + qk * qk);
  float gr12 = 2.0f * (qj * qk - qr * qi);
  float gr20 = 2.0f * (qi * qk - qr * qj);
  float gr21 = 2.0f * (qj * qk + qr * qi);
  float gr22 = 1.0f - 2.0f * (qi * qi + qj * qj);

  float c00 = gr00 * gr00 * sx2 + gr01 * gr01 * sy2 + gr02 * gr02 * sz2;
  float c01 = gr00 * gr10 * sx2 + gr01 * gr11 * sy2 + gr02 * gr12 * sz2;
  float c02 = gr00 * gr20 * sx2 + gr01 * gr21 * sy2 + gr02 * gr22 * sz2;
  float c11 = gr10 * gr10 * sx2 + gr11 * gr11 * sy2 + gr12 * gr12 * sz2;
  float c12 = gr10 * gr20 * sx2 + gr11 * gr21 * sy2 + gr12 * gr22 * sz2;
  float c22 = gr20 * gr20 * sx2 + gr21 * gr21 * sy2 + gr22 * gr22 * sz2;

  float r00 = camera_to_world[cb + 0u];
  float r01 = camera_to_world[cb + 1u];
  float r02 = camera_to_world[cb + 2u];
  float tx = camera_to_world[cb + 3u];
  float r10 = camera_to_world[cb + 4u];
  float r11 = camera_to_world[cb + 5u];
  float r12 = camera_to_world[cb + 6u];
  float ty = camera_to_world[cb + 7u];
  float r20 = camera_to_world[cb + 8u];
  float r21 = camera_to_world[cb + 9u];
  float r22 = camera_to_world[cb + 10u];
  float tz = camera_to_world[cb + 11u];

  float dx = mx - tx;
  float dy = my - ty;
  float dz = mz - tz;
  float x = dx * r00 + dy * r10 + dz * r20;
  float y = dx * r01 + dy * r11 + dz * r21;
  float z = dx * r02 + dy * r12 + dz * r22;

  float3 col0 = float3(r00, r10, r20);
  float3 col1 = float3(r01, r11, r21);
  float3 col2 = float3(r02, r12, r22);
  float cc00 = sym_bilinear(col0, c00, c01, c02, c11, c12, c22, col0);
  float cc01 = sym_bilinear(col0, c00, c01, c02, c11, c12, c22, col1);
  float cc02 = sym_bilinear(col0, c00, c01, c02, c11, c12, c22, col2);
  float cc11 = sym_bilinear(col1, c00, c01, c02, c11, c12, c22, col1);
  float cc12 = sym_bilinear(col1, c00, c01, c02, c11, c12, c22, col2);
  float cc22 = sym_bilinear(col2, c00, c01, c02, c11, c12, c22, col2);

  float fx = camera_params[cp + 0u];
  float fy = camera_params[cp + 1u];

  bool front = z > pm.near_plane;
  float z_safe = front ? max(z, pm.near_plane) : 1.0f;
  float x_project = front ? x : 0.0f;
  float y_project = front ? y : 0.0f;

  float inv_z = 1.0f / z_safe;
  float inv_z2 = inv_z * inv_z;
  float inv_z3 = inv_z2 * inv_z;
  float j00 = fx * inv_z;
  float j02 = -(fx * x_project) * inv_z2;
  float j11 = fy * inv_z;
  float j12 = -(fy * y_project) * inv_z2;

  float cov00 = j00 * j00 * cc00 + 2.0f * j00 * j02 * cc02 + j02 * j02 * cc22 + 0.3f;
  float cov01 = j00 * j11 * cc01 + j00 * j12 * cc02 + j02 * j11 * cc12 + j02 * j12 * cc22;
  float cov11 = j11 * j11 * cc11 + 2.0f * j11 * j12 * cc12 + j12 * j12 * cc22 + 0.3f;
  float det_raw = cov00 * cov11 - cov01 * cov01;
  float det = max(det_raw, 1e-6f);

  float2 gm = grad_means2d[gid];
  float ga = grad_conics[g3 + 0u];
  float gb = grad_conics[g3 + 1u];
  float gc = grad_conics[g3 + 2u];

  float g_cov00 = gc / det;
  float g_cov01 = -gb / det;
  float g_cov11 = ga / det;
  float g_det = -(ga * cov11 - gb * cov01 + gc * cov00) / (det * det);
  if (det_raw > 1e-6f) {
    g_cov00 += g_det * cov11;
    g_cov11 += g_det * cov00;
    g_cov01 += g_det * (-2.0f * cov01);
  }

  float g_cc00 = g_cov00 * j00 * j00;
  float g_cc02 = g_cov00 * (2.0f * j00 * j02);
  float g_cc22 = g_cov00 * j02 * j02;
  float g_j00 = g_cov00 * (2.0f * j00 * cc00 + 2.0f * j02 * cc02);
  float g_j02 = g_cov00 * (2.0f * j00 * cc02 + 2.0f * j02 * cc22);

  float g_cc01 = g_cov01 * j00 * j11;
  g_cc02 += g_cov01 * j00 * j12;
  float g_cc12 = g_cov01 * j02 * j11;
  g_cc22 += g_cov01 * j02 * j12;
  g_j00 += g_cov01 * (j11 * cc01 + j12 * cc02);
  g_j02 += g_cov01 * (j11 * cc12 + j12 * cc22);
  float g_j11 = g_cov01 * (j00 * cc01 + j02 * cc12);
  float g_j12 = g_cov01 * (j00 * cc02 + j02 * cc22);

  g_cc11 = g_cov11 * j11 * j11;
  g_cc12 += g_cov11 * (2.0f * j11 * j12);
  g_cc22 += g_cov11 * j12 * j12;
  g_j11 += g_cov11 * (2.0f * j11 * cc11 + 2.0f * j12 * cc12);
  g_j12 += g_cov11 * (2.0f * j11 * cc12 + 2.0f * j12 * cc22);

  float g_x = 0.0f;
  float g_y = 0.0f;
  float g_z = grad_depths[gid];
  float g_fx = 0.0f;
  float g_fy = 0.0f;
  float g_cx = gm.x;
  float g_cy = gm.y;

  if (front) {
    g_fx += gm.x * x * inv_z;
    g_x += gm.x * fx * inv_z;
    g_z += gm.x * (-(fx * x) * inv_z2);
    g_fy += gm.y * y * inv_z;
    g_y += gm.y * fy * inv_z;
    g_z += gm.y * (-(fy * y) * inv_z2);

    g_fx += g_j00 * inv_z;
    g_z += g_j00 * (-fx * inv_z2);
    g_fx += g_j02 * (-x * inv_z2);
    g_x += g_j02 * (-fx * inv_z2);
    g_z += g_j02 * (2.0f * fx * x * inv_z3);

    g_fy += g_j11 * inv_z;
    g_z += g_j11 * (-fy * inv_z2);
    g_fy += g_j12 * (-y * inv_z2);
    g_y += g_j12 * (-fy * inv_z2);
    g_z += g_j12 * (2.0f * fy * y * inv_z3);
  } else {
    g_fx += g_j00;
    g_fy += g_j11;
  }

  float g_c00 = 0.0f;
  float g_c01 = 0.0f;
  float g_c02 = 0.0f;
  float g_c11 = 0.0f;
  float g_c12 = 0.0f;
  float g_c22 = 0.0f;
  float3 g_col0 = float3(0.0f);
  float3 g_col1 = float3(0.0f);
  float3 g_col2 = float3(0.0f);
  float3 gu = float3(0.0f);
  float3 gv = float3(0.0f);

  gu = float3(0.0f); gv = float3(0.0f);
  sym_bilinear_backward(g_cc00, col0, c00, c01, c02, c11, c12, c22, col0, g_c00, g_c01, g_c02, g_c11, g_c12, g_c22, gu, gv);
  g_col0 += gu + gv;
  gu = float3(0.0f); gv = float3(0.0f);
  sym_bilinear_backward(g_cc01, col0, c00, c01, c02, c11, c12, c22, col1, g_c00, g_c01, g_c02, g_c11, g_c12, g_c22, gu, gv);
  g_col0 += gu; g_col1 += gv;
  gu = float3(0.0f); gv = float3(0.0f);
  sym_bilinear_backward(g_cc02, col0, c00, c01, c02, c11, c12, c22, col2, g_c00, g_c01, g_c02, g_c11, g_c12, g_c22, gu, gv);
  g_col0 += gu; g_col2 += gv;
  gu = float3(0.0f); gv = float3(0.0f);
  sym_bilinear_backward(g_cc11, col1, c00, c01, c02, c11, c12, c22, col1, g_c00, g_c01, g_c02, g_c11, g_c12, g_c22, gu, gv);
  g_col1 += gu + gv;
  gu = float3(0.0f); gv = float3(0.0f);
  sym_bilinear_backward(g_cc12, col1, c00, c01, c02, c11, c12, c22, col2, g_c00, g_c01, g_c02, g_c11, g_c12, g_c22, gu, gv);
  g_col1 += gu; g_col2 += gv;
  gu = float3(0.0f); gv = float3(0.0f);
  sym_bilinear_backward(g_cc22, col2, c00, c01, c02, c11, c12, c22, col2, g_c00, g_c01, g_c02, g_c11, g_c12, g_c22, gu, gv);
  g_col2 += gu + gv;

  float g_dx = g_x * r00 + g_y * r01 + g_z * r02;
  float g_dy = g_x * r10 + g_y * r11 + g_z * r12;
  float g_dz = g_x * r20 + g_y * r21 + g_z * r22;
  g_col0 += g_x * float3(dx, dy, dz);
  g_col1 += g_y * float3(dx, dy, dz);
  g_col2 += g_z * float3(dx, dy, dz);

  grad_means3d[g3 + 0u] = g_dx;
  grad_means3d[g3 + 1u] = g_dy;
  grad_means3d[g3 + 2u] = g_dz;

  float g_tx = -g_dx;
  float g_ty = -g_dy;
  float g_tz = -g_dz;

  float g_sx2 = g_c00 * gr00 * gr00 + g_c01 * gr00 * gr10 + g_c02 * gr00 * gr20 +
                g_c11 * gr10 * gr10 + g_c12 * gr10 * gr20 + g_c22 * gr20 * gr20;
  float g_sy2 = g_c00 * gr01 * gr01 + g_c01 * gr01 * gr11 + g_c02 * gr01 * gr21 +
                g_c11 * gr11 * gr11 + g_c12 * gr11 * gr21 + g_c22 * gr21 * gr21;
  float g_sz2 = g_c00 * gr02 * gr02 + g_c01 * gr02 * gr12 + g_c02 * gr02 * gr22 +
                g_c11 * gr12 * gr12 + g_c12 * gr12 * gr22 + g_c22 * gr22 * gr22;

  float g_gr00 = sx2 * (2.0f * g_c00 * gr00 + g_c01 * gr10 + g_c02 * gr20);
  float g_gr10 = sx2 * (g_c01 * gr00 + 2.0f * g_c11 * gr10 + g_c12 * gr20);
  float g_gr20 = sx2 * (g_c02 * gr00 + g_c12 * gr10 + 2.0f * g_c22 * gr20);
  float g_gr01 = sy2 * (2.0f * g_c00 * gr01 + g_c01 * gr11 + g_c02 * gr21);
  float g_gr11 = sy2 * (g_c01 * gr01 + 2.0f * g_c11 * gr11 + g_c12 * gr21);
  float g_gr21 = sy2 * (g_c02 * gr01 + g_c12 * gr11 + 2.0f * g_c22 * gr21);
  float g_gr02 = sz2 * (2.0f * g_c00 * gr02 + g_c01 * gr12 + g_c02 * gr22);
  float g_gr12 = sz2 * (g_c01 * gr02 + 2.0f * g_c11 * gr12 + g_c12 * gr22);
  float g_gr22 = sz2 * (g_c02 * gr02 + g_c12 * gr12 + 2.0f * g_c22 * gr22);

  grad_scales[g3 + 0u] = 2.0f * sx * g_sx2;
  grad_scales[g3 + 1u] = 2.0f * sy * g_sy2;
  grad_scales[g3 + 2u] = 2.0f * sz * g_sz2;

  float g_qr = 0.0f;
  float g_qi = 0.0f;
  float g_qj = 0.0f;
  float g_qk = 0.0f;
  g_qj += -4.0f * qj * g_gr00; g_qk += -4.0f * qk * g_gr00;
  g_qi += 2.0f * qj * g_gr01; g_qj += 2.0f * qi * g_gr01; g_qr += -2.0f * qk * g_gr01; g_qk += -2.0f * qr * g_gr01;
  g_qi += 2.0f * qk * g_gr02; g_qk += 2.0f * qi * g_gr02; g_qr += 2.0f * qj * g_gr02; g_qj += 2.0f * qr * g_gr02;
  g_qi += 2.0f * qj * g_gr10; g_qj += 2.0f * qi * g_gr10; g_qr += 2.0f * qk * g_gr10; g_qk += 2.0f * qr * g_gr10;
  g_qi += -4.0f * qi * g_gr11; g_qk += -4.0f * qk * g_gr11;
  g_qj += 2.0f * qk * g_gr12; g_qk += 2.0f * qj * g_gr12; g_qr += -2.0f * qi * g_gr12; g_qi += -2.0f * qr * g_gr12;
  g_qi += 2.0f * qk * g_gr20; g_qk += 2.0f * qi * g_gr20; g_qr += -2.0f * qj * g_gr20; g_qj += -2.0f * qr * g_gr20;
  g_qj += 2.0f * qk * g_gr21; g_qk += 2.0f * qj * g_gr21; g_qr += 2.0f * qi * g_gr21; g_qi += 2.0f * qr * g_gr21;
  g_qi += -4.0f * qi * g_gr22; g_qj += -4.0f * qj * g_gr22;

  grad_quats[g4 + 0u] = g_qr;
  grad_quats[g4 + 1u] = g_qi;
  grad_quats[g4 + 2u] = g_qj;
  grad_quats[g4 + 3u] = g_qk;
  grad_opacities[gid] = front ? grad_projected_opacities[gid] : 0.0f;

  atomic_add_float(grad_camera_params, cp + 0u, g_fx);
  atomic_add_float(grad_camera_params, cp + 1u, g_fy);
  atomic_add_float(grad_camera_params, cp + 2u, g_cx);
  atomic_add_float(grad_camera_params, cp + 3u, g_cy);

  atomic_add_float(grad_camera_to_world, cb + 0u, g_col0.x);
  atomic_add_float(grad_camera_to_world, cb + 4u, g_col0.y);
  atomic_add_float(grad_camera_to_world, cb + 8u, g_col0.z);
  atomic_add_float(grad_camera_to_world, cb + 1u, g_col1.x);
  atomic_add_float(grad_camera_to_world, cb + 5u, g_col1.y);
  atomic_add_float(grad_camera_to_world, cb + 9u, g_col1.z);
  atomic_add_float(grad_camera_to_world, cb + 2u, g_col2.x);
  atomic_add_float(grad_camera_to_world, cb + 6u, g_col2.y);
  atomic_add_float(grad_camera_to_world, cb + 10u, g_col2.z);
  atomic_add_float(grad_camera_to_world, cb + 3u, g_tx);
  atomic_add_float(grad_camera_to_world, cb + 7u, g_ty);
  atomic_add_float(grad_camera_to_world, cb + 11u, g_tz);
}

inline float3 load3_sh(const threadgroup float* base, uint idx) {
  uint b = idx * 3u;
  return float3(base[b + 0u], base[b + 1u], base[b + 2u]);
}

inline void atomic_add3(device atomic_float* base, uint idx, float3 v) {
  uint b = idx * 3u;
  atomic_fetch_add_explicit(&base[b + 0u], v.x, memory_order_relaxed);
  atomic_fetch_add_explicit(&base[b + 1u], v.y, memory_order_relaxed);
  atomic_fetch_add_explicit(&base[b + 2u], v.z, memory_order_relaxed);
}

inline uint next_pow2_u32(uint x) {
  x = max(x, 1u);
  x -= 1u;
  x |= x >> 1u;
  x |= x >> 2u;
  x |= x >> 4u;
  x |= x >> 8u;
  x |= x >> 16u;
  return x + 1u;
}

inline float safe_det(float a, float b, float c, float eps) {
  return max(a * c - b * b, eps);
}

inline bool alpha_support_params(float opacity, constant MetaF32& mf, thread float& tau) {
  if (opacity <= mf.alpha_threshold) return false;
  float ratio = max(mf.alpha_threshold / max(opacity, mf.eps), mf.eps);
  tau = -2.0f * log(ratio);
  return isfinite(tau) && (tau > 0.0f);
}

inline int4 snugbox(float2 m, float a, float b, float c, float tau, constant MetaI32& mi, constant MetaF32& mf) {
  float det = safe_det(a, b, c, mf.eps);
  float half_x = sqrt(max(tau * c / det, 0.0f));
  float half_y = sqrt(max(tau * a / det, 0.0f));
  int x0 = max(0, int(floor(m.x - half_x - 0.5f)));
  int x1 = min(mi.width - 1, int(ceil(m.x + half_x - 0.5f)));
  int y0 = max(0, int(floor(m.y - half_y - 0.5f)));
  int y1 = min(mi.height - 1, int(ceil(m.y + half_y - 0.5f)));
  return int4(x0, y0, x1, y1);
}

inline bool ellipse_intersects_rect(float2 m, float a, float b, float c, float tau, float rx0, float ry0, float rx1, float ry1) {
  float dx0 = rx0 - m.x;
  float dx1 = rx1 - m.x;
  float dy0 = ry0 - m.y;
  float dy1 = ry1 - m.y;
  if (m.x >= rx0 && m.x <= rx1 && m.y >= ry0 && m.y <= ry1) return true;
  float qmin = INFINITY;
  qmin = min(qmin, a * dx0 * dx0 + 2.0f * b * dx0 * dy0 + c * dy0 * dy0);
  qmin = min(qmin, a * dx0 * dx0 + 2.0f * b * dx0 * dy1 + c * dy1 * dy1);
  qmin = min(qmin, a * dx1 * dx1 + 2.0f * b * dx1 * dy0 + c * dy0 * dy0);
  qmin = min(qmin, a * dx1 * dx1 + 2.0f * b * dx1 * dy1 + c * dy1 * dy1);
  if (c > 1e-8f) {
    float dy = clamp(-(b / c) * dx0, dy0, dy1);
    qmin = min(qmin, a * dx0 * dx0 + 2.0f * b * dx0 * dy + c * dy * dy);
    dy = clamp(-(b / c) * dx1, dy0, dy1);
    qmin = min(qmin, a * dx1 * dx1 + 2.0f * b * dx1 * dy + c * dy * dy);
  }
  if (a > 1e-8f) {
    float dx = clamp(-(b / a) * dy0, dx0, dx1);
    qmin = min(qmin, a * dx * dx + 2.0f * b * dx * dy0 + c * dy0 * dy0);
    dx = clamp(-(b / a) * dy1, dx0, dx1);
    qmin = min(qmin, a * dx * dx + 2.0f * b * dx * dy1 + c * dy1 * dy1);
  }
  return qmin <= tau;
}

inline bool eval_alpha(
    float2 p,
    float2 m,
    float a,
    float b,
    float c,
    float opacity,
    constant MetaF32& mf,
    thread float& alpha,
    thread float& raw_alpha,
    thread float& power,
    thread float2& d) {
  d = p - m;
  power = -0.5f * (a * d.x * d.x + 2.0f * b * d.x * d.y + c * d.y * d.y);
  if (power > 0.0f) return false;
  raw_alpha = opacity * exp(power);
  alpha = min(mf.max_alpha, raw_alpha);
  return alpha >= mf.alpha_threshold;
}

inline void bitonic_sort_ids(threadgroup uint* shared_ids, uint valid_count, uint tid) {
  uint sort_n = next_pow2_u32(valid_count);
  for (uint i = tid; i < sort_n; i += GSP_THREADS) {
    if (i >= valid_count) shared_ids[i] = 0xFFFFFFFFu;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint k = 2u; k <= sort_n; k <<= 1u) {
    for (uint j = k >> 1u; j > 0u; j >>= 1u) {
      uint n_pairs = sort_n >> 1u;
      for (uint pair = tid; pair < n_pairs; pair += GSP_THREADS) {
        uint pos = 2u * j * (pair / j) + (pair % j);
        uint ixj = pos + j;
        bool ascending = ((pos & k) == 0u);
        uint va = shared_ids[pos];
        uint vb = shared_ids[ixj];
        if ((va > vb) == ascending) {
          shared_ids[pos] = vb;
          shared_ids[ixj] = va;
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  }
}

inline void load_chunk_params(
    const device float2* means2d,
    const device float* conics,
    const device float* colors,
    const device float* opacities,
    const threadgroup uint* shared_ids,
    uint chunk_start,
    uint chunk_n,
    uint tid,
    threadgroup float2* sh_means,
    threadgroup float* sh_conics,
    threadgroup float* sh_colors,
    threadgroup float* sh_opacities) {
  for (uint i = tid; i < chunk_n; i += GSP_THREADS) {
    uint g = shared_ids[chunk_start + i];
    sh_means[i] = means2d[g];
    uint b3 = i * 3u;
    uint g3 = g * 3u;
    sh_conics[b3 + 0u] = conics[g3 + 0u];
    sh_conics[b3 + 1u] = conics[g3 + 1u];
    sh_conics[b3 + 2u] = conics[g3 + 2u];
    sh_colors[b3 + 0u] = colors[g3 + 0u];
    sh_colors[b3 + 1u] = colors[g3 + 1u];
    sh_colors[b3 + 2u] = colors[g3 + 2u];
    sh_opacities[i] = opacities[g];
  }
}

inline uint reduce_alive(uint alive, uint simd_lane, uint simd_group, threadgroup uint* tg_alive) {
  uint sg_sum = simd_sum(alive);
  if (simd_lane == 0u) tg_alive[simd_group] = sg_sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_group == 0u) {
    uint v = (simd_lane < GSP_SIMDGROUPS) ? tg_alive[simd_lane] : 0u;
    uint total = simd_sum(v);
    if (simd_lane == 0u) tg_alive[0] = total;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return tg_alive[0];
}

inline uint reduce_max_u32(uint value, uint simd_lane, uint simd_group, threadgroup uint* tg_tmp) {
  uint sg_max = simd_max(value);
  if (simd_lane == 0u) tg_tmp[simd_group] = sg_max;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_group == 0u) {
    uint v = (simd_lane < GSP_SIMDGROUPS) ? tg_tmp[simd_lane] : 0u;
    uint total = simd_max(v);
    if (simd_lane == 0u) tg_tmp[0] = total;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return tg_tmp[0];
}

inline void tile_batch_local(uint global_tile, constant MetaI32& mi, thread uint& batch, thread uint& local_tile) {
  batch = global_tile / uint(mi.tiles_per_image);
  local_tile = global_tile - batch * uint(mi.tiles_per_image);
}

inline uint pixel_index(uint batch, uint x, uint y, constant MetaI32& mi) {
  return (batch * uint(mi.height) + y) * uint(mi.width) + x;
}

inline void tile_pixel_from_tid(uint global_tile, uint tid, constant MetaI32& mi, thread uint& batch, thread uint& x, thread uint& y) {
  uint local_tile;
  tile_batch_local(global_tile, mi, batch, local_tile);
  uint tile_x = local_tile % uint(mi.tiles_x);
  uint tile_y = local_tile / uint(mi.tiles_x);
  uint px = tid % GSP_TILE_SIZE;
  uint py = tid / GSP_TILE_SIZE;
  x = tile_x * GSP_TILE_SIZE + px;
  y = tile_y * GSP_TILE_SIZE + py;
}

inline uint tile_img_base(uint local_tile_idx, uint px, uint py) {
  return (((local_tile_idx * GSP_TILE_SIZE + py) * GSP_TILE_SIZE + px) * 3u);
}

kernel void count_tiles(
    const device float2* means2d [[buffer(0)]],
    const device float* conics [[buffer(1)]],
    const device float* opacities [[buffer(2)]],
    constant MetaI32& mi [[buffer(3)]],
    constant MetaF32& mf [[buffer(4)]],
    device int4* bbox_out [[buffer(5)]],
    device float* tau_out [[buffer(6)]],
    device atomic_uint* tile_counts [[buffer(7)]],
    uint gid [[thread_position_in_grid]]) {
  if (gid >= uint(mi.gaussians)) return;
  float tau;
  if (!alpha_support_params(opacities[gid], mf, tau)) {
    bbox_out[gid] = int4(1, 1, 0, 0);
    tau_out[gid] = 0.0f;
    return;
  }
  uint batch = gid / uint(mi.gaussians_per_batch);
  float2 m = means2d[gid];
  uint g3 = gid * 3u;
  float a = conics[g3 + 0u];
  float b = conics[g3 + 1u];
  float c = conics[g3 + 2u];
  int4 bb = snugbox(m, a, b, c, tau, mi, mf);
  bbox_out[gid] = bb;
  tau_out[gid] = tau;
  if (bb.x > bb.z || bb.y > bb.w) return;
  int tx0 = bb.x / mi.tile_size;
  int tx1 = bb.z / mi.tile_size;
  int ty0 = bb.y / mi.tile_size;
  int ty1 = bb.w / mi.tile_size;
  for (int ty = ty0; ty <= ty1; ++ty) {
    float ry0 = float(ty * mi.tile_size) + 0.5f;
    float ry1 = min(float(mi.height - 1) + 0.5f, float((ty + 1) * mi.tile_size - 1) + 0.5f);
    for (int tx = tx0; tx <= tx1; ++tx) {
      float rx0 = float(tx * mi.tile_size) + 0.5f;
      float rx1 = min(float(mi.width - 1) + 0.5f, float((tx + 1) * mi.tile_size - 1) + 0.5f);
      if (ellipse_intersects_rect(m, a, b, c, tau, rx0, ry0, rx1, ry1)) {
        uint local_tile = uint(ty * mi.tiles_x + tx);
        uint tile = batch * uint(mi.tiles_per_image) + local_tile;
        atomic_fetch_add_explicit(tile_counts + tile, 1u, memory_order_relaxed);
      }
    }
  }
}

kernel void emit_binned_ids(
    const device float2* means2d [[buffer(0)]],
    const device float* conics [[buffer(1)]],
    const device int4* bbox_in [[buffer(2)]],
    const device float* tau_in [[buffer(3)]],
    constant MetaI32& mi [[buffer(4)]],
    device atomic_uint* tile_cursors [[buffer(5)]],
    device uint* binned_ids [[buffer(6)]],
    uint gid [[thread_position_in_grid]]) {
  if (gid >= uint(mi.gaussians)) return;
  int4 bb = bbox_in[gid];
  if (bb.x > bb.z || bb.y > bb.w) return;
  float tau = tau_in[gid];
  uint batch = gid / uint(mi.gaussians_per_batch);
  float2 m = means2d[gid];
  uint g3 = gid * 3u;
  float a = conics[g3 + 0u];
  float b = conics[g3 + 1u];
  float c = conics[g3 + 2u];
  int tx0 = bb.x / mi.tile_size;
  int tx1 = bb.z / mi.tile_size;
  int ty0 = bb.y / mi.tile_size;
  int ty1 = bb.w / mi.tile_size;
  for (int ty = ty0; ty <= ty1; ++ty) {
    float ry0 = float(ty * mi.tile_size) + 0.5f;
    float ry1 = min(float(mi.height - 1) + 0.5f, float((ty + 1) * mi.tile_size - 1) + 0.5f);
    for (int tx = tx0; tx <= tx1; ++tx) {
      float rx0 = float(tx * mi.tile_size) + 0.5f;
      float rx1 = min(float(mi.width - 1) + 0.5f, float((tx + 1) * mi.tile_size - 1) + 0.5f);
      if (ellipse_intersects_rect(m, a, b, c, tau, rx0, ry0, rx1, ry1)) {
        uint local_tile = uint(ty * mi.tiles_x + tx);
        uint tile = batch * uint(mi.tiles_per_image) + local_tile;
        uint idx = atomic_fetch_add_explicit(tile_cursors + tile, 1u, memory_order_relaxed);
        binned_ids[idx] = gid;
      }
    }
  }
}

// eval fast path: no writeback, no stop-count save
kernel void tile_fast_forward_eval(
    const device uint* tile_counts [[buffer(0)]],
    const device int* tile_offsets [[buffer(1)]],
    const device uint* binned_ids [[buffer(2)]],
    const device float2* means2d [[buffer(3)]],
    const device float* conics [[buffer(4)]],
    const device float* colors [[buffer(5)]],
    const device float* opacities [[buffer(6)]],
    constant MetaI32& mi [[buffer(7)]],
    constant MetaF32& mf [[buffer(8)]],
    device float* out_rgb [[buffer(9)]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  if (tg_id >= uint(mi.tile_count)) return;
  uint batch, x, y;
  tile_pixel_from_tid(tg_id, tid, mi, batch, x, y);
  bool valid = (x < uint(mi.width) && y < uint(mi.height));
  uint pix = valid ? pixel_index(batch, x, y, mi) : 0u;
  uint count = tile_counts[tg_id];
  if (count == 0u || count > uint(mi.max_fast_pairs)) {
    if (valid) {
      out_rgb[pix * 3u + 0u] = mf.bg_r;
      out_rgb[pix * 3u + 1u] = mf.bg_g;
      out_rgb[pix * 3u + 2u] = mf.bg_b;
    }
    return;
  }
  threadgroup uint shared_ids[GSP_FAST_CAP];
  threadgroup float2 sh_means[GSP_CHUNK];
  threadgroup float sh_conics[GSP_CHUNK * 3u];
  threadgroup float sh_colors[GSP_CHUNK * 3u];
  threadgroup float sh_opacities[GSP_CHUNK];
  threadgroup uint tg_alive[GSP_SIMDGROUPS];
  uint start = uint(tile_offsets[tg_id]);
  for (uint i = tid; i < count; i += GSP_THREADS) shared_ids[i] = binned_ids[start + i];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bitonic_sort_ids(shared_ids, count, tid);
  float3 accum = float3(0.0f);
  float T = 1.0f;
  float2 p = float2(float(x) + 0.5f, float(y) + 0.5f);
  for (uint chunk_start = 0u; chunk_start < count; chunk_start += GSP_CHUNK) {
    uint chunk_n = min(GSP_CHUNK, count - chunk_start);
    load_chunk_params(means2d, conics, colors, opacities, shared_ids, chunk_start, chunk_n, tid, sh_means, sh_conics, sh_colors, sh_opacities);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint alive_total = reduce_alive((valid && T > mf.transmittance_threshold) ? 1u : 0u, simd_lane, simd_group, tg_alive);
    if (alive_total == 0u) break;
    if (valid && T > mf.transmittance_threshold) {
      for (uint j = 0u; j < chunk_n; ++j) {
        float alpha, raw_alpha, power; float2 d;
        float2 m = sh_means[j];
        float3 q = load3_sh(sh_conics, j);
        if (!eval_alpha(p, m, q.x, q.y, q.z, sh_opacities[j], mf, alpha, raw_alpha, power, d)) continue;
        float w = T * alpha;
        accum += w * load3_sh(sh_colors, j);
        T *= (1.0f - alpha);
        if (T <= mf.transmittance_threshold) break;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (valid) {
    out_rgb[pix * 3u + 0u] = accum.x + T * mf.bg_r;
    out_rgb[pix * 3u + 1u] = accum.y + T * mf.bg_g;
    out_rgb[pix * 3u + 2u] = accum.z + T * mf.bg_b;
  }
}

// train fast path: writes sorted IDs back into binned_ids and saves per-tile stop count
kernel void tile_fast_forward_state(
    const device uint* tile_counts [[buffer(0)]],
    const device int* tile_offsets [[buffer(1)]],
    device uint* binned_ids [[buffer(2)]],
    const device float2* means2d [[buffer(3)]],
    const device float* conics [[buffer(4)]],
    const device float* colors [[buffer(5)]],
    const device float* opacities [[buffer(6)]],
    constant MetaI32& mi [[buffer(7)]],
    constant MetaF32& mf [[buffer(8)]],
    device float* out_rgb [[buffer(9)]],
    device int* out_stop_counts [[buffer(10)]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  if (tg_id >= uint(mi.tile_count)) return;
  uint batch, x, y;
  tile_pixel_from_tid(tg_id, tid, mi, batch, x, y);
  bool valid = (x < uint(mi.width) && y < uint(mi.height));
  uint pix = valid ? pixel_index(batch, x, y, mi) : 0u;
  uint count = tile_counts[tg_id];
  uint start = uint(tile_offsets[tg_id]);
  if (count == 0u || count > uint(mi.max_fast_pairs)) {
    if (tid == 0u) out_stop_counts[tg_id] = 0;
    if (valid) {
      out_rgb[pix * 3u + 0u] = mf.bg_r;
      out_rgb[pix * 3u + 1u] = mf.bg_g;
      out_rgb[pix * 3u + 2u] = mf.bg_b;
    }
    return;
  }
  threadgroup uint shared_ids[GSP_FAST_CAP];
  threadgroup float2 sh_means[GSP_CHUNK];
  threadgroup float sh_conics[GSP_CHUNK * 3u];
  threadgroup float sh_colors[GSP_CHUNK * 3u];
  threadgroup float sh_opacities[GSP_CHUNK];
  threadgroup uint tg_alive[GSP_SIMDGROUPS];
  threadgroup uint tg_stop[GSP_SIMDGROUPS];
  for (uint i = tid; i < count; i += GSP_THREADS) shared_ids[i] = binned_ids[start + i];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bitonic_sort_ids(shared_ids, count, tid);
  for (uint i = tid; i < count; i += GSP_THREADS) binned_ids[start + i] = shared_ids[i];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float3 accum = float3(0.0f);
  float T = 1.0f;
  uint local_stop = 0u;
  float2 p = float2(float(x) + 0.5f, float(y) + 0.5f);
  for (uint chunk_start = 0u; chunk_start < count; chunk_start += GSP_CHUNK) {
    uint chunk_n = min(GSP_CHUNK, count - chunk_start);
    load_chunk_params(means2d, conics, colors, opacities, shared_ids, chunk_start, chunk_n, tid, sh_means, sh_conics, sh_colors, sh_opacities);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint alive_total = reduce_alive((valid && T > mf.transmittance_threshold) ? 1u : 0u, simd_lane, simd_group, tg_alive);
    if (alive_total == 0u) break;
    if (valid && T > mf.transmittance_threshold) {
      for (uint j = 0u; j < chunk_n; ++j) {
        local_stop = chunk_start + j + 1u;
        float alpha, raw_alpha, power; float2 d;
        float2 m = sh_means[j];
        float3 q = load3_sh(sh_conics, j);
        if (!eval_alpha(p, m, q.x, q.y, q.z, sh_opacities[j], mf, alpha, raw_alpha, power, d)) continue;
        float w = T * alpha;
        accum += w * load3_sh(sh_colors, j);
        T *= (1.0f - alpha);
        if (T <= mf.transmittance_threshold) break;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  uint tile_stop = reduce_max_u32(local_stop, simd_lane, simd_group, tg_stop);
  if (tid == 0u) out_stop_counts[tg_id] = int(tile_stop);
  if (valid) {
    out_rgb[pix * 3u + 0u] = accum.x + T * mf.bg_r;
    out_rgb[pix * 3u + 1u] = accum.y + T * mf.bg_g;
    out_rgb[pix * 3u + 2u] = accum.z + T * mf.bg_b;
  }
}

kernel void tile_fast_backward_saved(
    const device float* grad_rgb [[buffer(0)]],
    const device uint* tile_counts [[buffer(1)]],
    const device int* tile_offsets [[buffer(2)]],
    const device uint* binned_ids [[buffer(3)]],
    const device int* tile_stop_counts [[buffer(4)]],
    const device float2* means2d [[buffer(5)]],
    const device float* conics [[buffer(6)]],
    const device float* colors [[buffer(7)]],
    const device float* opacities [[buffer(8)]],
    constant MetaI32& mi [[buffer(9)]],
    constant MetaF32& mf [[buffer(10)]],
    device atomic_float* g_means2d [[buffer(11)]],
    device atomic_float* g_conics [[buffer(12)]],
    device atomic_float* g_colors [[buffer(13)]],
    device atomic_float* g_opacities [[buffer(14)]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  if (tg_id >= uint(mi.tile_count)) return;
  uint count = tile_counts[tg_id];
  uint stop_count = min(count, uint(max(tile_stop_counts[tg_id], 0)));
  if (count == 0u || count > uint(mi.max_fast_pairs) || stop_count == 0u) return;
  threadgroup uint shared_ids[GSP_FAST_CAP];
  threadgroup float2 sh_means[GSP_CHUNK];
  threadgroup float sh_conics[GSP_CHUNK * 3u];
  threadgroup float sh_colors[GSP_CHUNK * 3u];
  threadgroup float sh_opacities[GSP_CHUNK];
  threadgroup uint tg_alive[GSP_SIMDGROUPS];
  threadgroup float4 partial0[GSP_SIMDGROUPS];
  threadgroup float4 partial1[GSP_SIMDGROUPS];
  threadgroup float partial2[GSP_SIMDGROUPS];
  uint start = uint(tile_offsets[tg_id]);
  for (uint i = tid; i < stop_count; i += GSP_THREADS) shared_ids[i] = binned_ids[start + i];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint batch, x, y;
  tile_pixel_from_tid(tg_id, tid, mi, batch, x, y);
  bool valid = (x < uint(mi.width) && y < uint(mi.height));
  uint pix = valid ? pixel_index(batch, x, y, mi) : 0u;
  float2 p = float2(float(x) + 0.5f, float(y) + 0.5f);
  float3 go = valid ? float3(grad_rgb[pix * 3u + 0u], grad_rgb[pix * 3u + 1u], grad_rgb[pix * 3u + 2u]) : float3(0.0f);
  float T_final = 1.0f;
  uint end_i = stop_count;
  for (uint chunk_start = 0u; chunk_start < stop_count; chunk_start += GSP_CHUNK) {
    uint chunk_n = min(GSP_CHUNK, stop_count - chunk_start);
    load_chunk_params(means2d, conics, colors, opacities, shared_ids, chunk_start, chunk_n, tid, sh_means, sh_conics, sh_colors, sh_opacities);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint alive_total = reduce_alive((valid && T_final > mf.transmittance_threshold) ? 1u : 0u, simd_lane, simd_group, tg_alive);
    if (alive_total == 0u) break;
    if (valid && T_final > mf.transmittance_threshold) {
      for (uint j = 0u; j < chunk_n; ++j) {
        float alpha, raw_alpha, power; float2 d;
        float2 m = sh_means[j];
        float3 q = load3_sh(sh_conics, j);
        if (!eval_alpha(p, m, q.x, q.y, q.z, sh_opacities[j], mf, alpha, raw_alpha, power, d)) continue;
        T_final *= (1.0f - alpha);
        if (T_final <= mf.transmittance_threshold) { end_i = chunk_start + j + 1u; break; }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  float T_cur = T_final;
  float gT = valid ? dot(go, float3(mf.bg_r, mf.bg_g, mf.bg_b)) : 0.0f;
  // Keep barrier control uniform across the threadgroup. `end_i` is per pixel,
  // so using it as the loop bound makes saturated pixels take fewer barriers
  // than unsaturated pixels. Iterate over the tile-level stop count and mask
  // each pixel with `global_i < end_i` inside the loop.
  for (int chunk_end = int(stop_count); chunk_end > 0; chunk_end -= int(GSP_CHUNK)) {
    int chunk_start_i = max(0, chunk_end - int(GSP_CHUNK));
    uint chunk_start = uint(chunk_start_i);
    uint chunk_n = uint(chunk_end - chunk_start_i);
    load_chunk_params(means2d, conics, colors, opacities, shared_ids, chunk_start, chunk_n, tid, sh_means, sh_conics, sh_colors, sh_opacities);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int local = int(chunk_n) - 1; local >= 0; --local) {
      uint global_i = chunk_start + uint(local);
      uint g = shared_ids[global_i];
      float2 m = sh_means[uint(local)];
      float3 q = load3_sh(sh_conics, uint(local));
      float3 c = load3_sh(sh_colors, uint(local));
      float opacity = sh_opacities[uint(local)];
      float l_gmx = 0.0f, l_gmy = 0.0f, l_ga = 0.0f, l_gb = 0.0f, l_gc = 0.0f, l_gop = 0.0f;
      float3 l_gcol = float3(0.0f);
      if (valid && global_i < end_i) {
        float alpha, raw_alpha, power; float2 d;
        if (eval_alpha(p, m, q.x, q.y, q.z, opacity, mf, alpha, raw_alpha, power, d)) {
          float denom = max(1.0f - alpha, mf.eps);
          float T_prev = T_cur / denom;
          float dot_gc = dot(go, c);
          float g_alpha = T_prev * (dot_gc - gT);
          l_gcol = go * (T_prev * alpha);
          float gate = (raw_alpha < mf.max_alpha) ? 1.0f : 0.0f;
          float g_raw = g_alpha * gate;
          float g_power = g_raw * raw_alpha;
          l_ga = g_power * (-0.5f) * d.x * d.x;
          l_gb = g_power * (-1.0f) * d.x * d.y;
          l_gc = g_power * (-0.5f) * d.y * d.y;
          float g_dx = g_power * (-(q.x * d.x + q.y * d.y));
          float g_dy = g_power * (-(q.y * d.x + q.z * d.y));
          l_gmx = -g_dx;
          l_gmy = -g_dy;
          l_gop = g_raw * (raw_alpha / max(opacity, mf.eps));
          gT = alpha * dot_gc + (1.0f - alpha) * gT;
          T_cur = T_prev;
        }
      }
      float4 s0 = simd_sum(float4(l_gmx, l_gmy, l_ga, l_gb));
      float4 s1 = simd_sum(float4(l_gc, l_gcol.x, l_gcol.y, l_gcol.z));
      float s2 = simd_sum(l_gop);
      if (simd_lane == 0u) { partial0[simd_group] = s0; partial1[simd_group] = s1; partial2[simd_group] = s2; }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (simd_group == 0u) {
        float4 v0 = (simd_lane < GSP_SIMDGROUPS) ? partial0[simd_lane] : float4(0.0f);
        float4 v1 = (simd_lane < GSP_SIMDGROUPS) ? partial1[simd_lane] : float4(0.0f);
        float v2 = (simd_lane < GSP_SIMDGROUPS) ? partial2[simd_lane] : 0.0f;
        float4 t0 = simd_sum(v0);
        float4 t1 = simd_sum(v1);
        float t2 = simd_sum(v2);
        if (simd_lane == 0u) {
          atomic_fetch_add_explicit(&g_means2d[g * 2u + 0u], t0.x, memory_order_relaxed);
          atomic_fetch_add_explicit(&g_means2d[g * 2u + 1u], t0.y, memory_order_relaxed);
          atomic_fetch_add_explicit(&g_conics[g * 3u + 0u], t0.z, memory_order_relaxed);
          atomic_fetch_add_explicit(&g_conics[g * 3u + 1u], t0.w, memory_order_relaxed);
          atomic_fetch_add_explicit(&g_conics[g * 3u + 2u], t1.x, memory_order_relaxed);
          atomic_add3(g_colors, g, float3(t1.y, t1.z, t1.w));
          atomic_fetch_add_explicit(&g_opacities[g], t2, memory_order_relaxed);
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

kernel void tile_overflow_forward(
    const device uint* overflow_tile_ids [[buffer(0)]],
    const device int* overflow_tile_offsets [[buffer(1)]],
    const device uint* overflow_sorted_ids [[buffer(2)]],
    const device float2* means2d [[buffer(3)]],
    const device float* conics [[buffer(4)]],
    const device float* colors [[buffer(5)]],
    const device float* opacities [[buffer(6)]],
    constant MetaI32& mi [[buffer(7)]],
    constant MetaF32& mf [[buffer(8)]],
    device float* out_tiles [[buffer(9)]],
    uint local_tile_idx [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  uint tile_id = overflow_tile_ids[local_tile_idx];
  uint start = uint(overflow_tile_offsets[local_tile_idx]);
  uint end = uint(overflow_tile_offsets[local_tile_idx + 1]);
  uint count = end - start;
  threadgroup float2 sh_means[GSP_CHUNK];
  threadgroup float sh_conics[GSP_CHUNK * 3u];
  threadgroup float sh_colors[GSP_CHUNK * 3u];
  threadgroup float sh_opacities[GSP_CHUNK];
  threadgroup uint tg_alive[GSP_SIMDGROUPS];
  threadgroup uint sh_ids[GSP_CHUNK];
  uint batch, x, y;
  tile_pixel_from_tid(tile_id, tid, mi, batch, x, y);
  bool valid = (x < uint(mi.width) && y < uint(mi.height));
  uint px = tid % GSP_TILE_SIZE;
  uint py = tid / GSP_TILE_SIZE;
  float2 p = float2(float(x) + 0.5f, float(y) + 0.5f);
  float3 accum = float3(0.0f);
  float T = 1.0f;
  for (uint chunk_start = 0u; chunk_start < count; chunk_start += GSP_CHUNK) {
    uint chunk_n = min(GSP_CHUNK, count - chunk_start);
    for (uint i = tid; i < chunk_n; i += GSP_THREADS) sh_ids[i] = overflow_sorted_ids[start + i + chunk_start];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    load_chunk_params(means2d, conics, colors, opacities, sh_ids, 0u, chunk_n, tid, sh_means, sh_conics, sh_colors, sh_opacities);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint alive_total = reduce_alive((valid && T > mf.transmittance_threshold) ? 1u : 0u, simd_lane, simd_group, tg_alive);
    if (alive_total == 0u) break;
    if (valid && T > mf.transmittance_threshold) {
      for (uint j = 0u; j < chunk_n; ++j) {
        float alpha, raw_alpha, power; float2 d;
        float2 m = sh_means[j];
        float3 q = load3_sh(sh_conics, j);
        if (!eval_alpha(p, m, q.x, q.y, q.z, sh_opacities[j], mf, alpha, raw_alpha, power, d)) continue;
        float w = T * alpha;
        accum += w * load3_sh(sh_colors, j);
        T *= (1.0f - alpha);
        if (T <= mf.transmittance_threshold) break;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  uint base = tile_img_base(local_tile_idx, px, py);
  out_tiles[base + 0u] = accum.x + T * mf.bg_r;
  out_tiles[base + 1u] = accum.y + T * mf.bg_g;
  out_tiles[base + 2u] = accum.z + T * mf.bg_b;
}

kernel void tile_overflow_backward(
    const device float* grad_tiles [[buffer(0)]],
    const device uint* overflow_tile_ids [[buffer(1)]],
    const device int* overflow_tile_offsets [[buffer(2)]],
    const device uint* overflow_sorted_ids [[buffer(3)]],
    const device float2* means2d [[buffer(4)]],
    const device float* conics [[buffer(5)]],
    const device float* colors [[buffer(6)]],
    const device float* opacities [[buffer(7)]],
    constant MetaI32& mi [[buffer(8)]],
    constant MetaF32& mf [[buffer(9)]],
    device atomic_float* g_means2d [[buffer(10)]],
    device atomic_float* g_conics [[buffer(11)]],
    device atomic_float* g_colors [[buffer(12)]],
    device atomic_float* g_opacities [[buffer(13)]],
    uint local_tile_idx [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  uint tile_id = overflow_tile_ids[local_tile_idx];
  uint start = uint(overflow_tile_offsets[local_tile_idx]);
  uint end = uint(overflow_tile_offsets[local_tile_idx + 1]);
  uint count = end - start;
  if (count == 0u) return;
  threadgroup float2 sh_means[GSP_CHUNK];
  threadgroup float sh_conics[GSP_CHUNK * 3u];
  threadgroup float sh_colors[GSP_CHUNK * 3u];
  threadgroup float sh_opacities[GSP_CHUNK];
  threadgroup uint tg_alive[GSP_SIMDGROUPS];
  threadgroup float4 partial0[GSP_SIMDGROUPS];
  threadgroup float4 partial1[GSP_SIMDGROUPS];
  threadgroup float partial2[GSP_SIMDGROUPS];
  threadgroup uint sh_ids[GSP_CHUNK];
  uint batch, x, y;
  tile_pixel_from_tid(tile_id, tid, mi, batch, x, y);
  bool valid = (x < uint(mi.width) && y < uint(mi.height));
  uint px = tid % GSP_TILE_SIZE;
  uint py = tid / GSP_TILE_SIZE;
  float2 p = float2(float(x) + 0.5f, float(y) + 0.5f);
  uint base = tile_img_base(local_tile_idx, px, py);
  float3 go = float3(grad_tiles[base + 0u], grad_tiles[base + 1u], grad_tiles[base + 2u]);
  float T_final = 1.0f;
  uint end_i = count;
  for (uint chunk_start = 0u; chunk_start < count; chunk_start += GSP_CHUNK) {
    uint chunk_n = min(GSP_CHUNK, count - chunk_start);
    for (uint i = tid; i < chunk_n; i += GSP_THREADS) sh_ids[i] = overflow_sorted_ids[start + chunk_start + i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    load_chunk_params(means2d, conics, colors, opacities, sh_ids, 0u, chunk_n, tid, sh_means, sh_conics, sh_colors, sh_opacities);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint alive_total = reduce_alive((valid && T_final > mf.transmittance_threshold) ? 1u : 0u, simd_lane, simd_group, tg_alive);
    if (alive_total == 0u) break;
    if (valid && T_final > mf.transmittance_threshold) {
      for (uint j = 0u; j < chunk_n; ++j) {
        float alpha, raw_alpha, power; float2 d;
        float2 m = sh_means[j];
        float3 q = load3_sh(sh_conics, j);
        if (!eval_alpha(p, m, q.x, q.y, q.z, sh_opacities[j], mf, alpha, raw_alpha, power, d)) continue;
        T_final *= (1.0f - alpha);
        if (T_final <= mf.transmittance_threshold) { end_i = chunk_start + j + 1u; break; }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  float T_cur = T_final;
  float gT = valid ? dot(go, float3(mf.bg_r, mf.bg_g, mf.bg_b)) : 0.0f;
  // Keep barrier control uniform across the threadgroup. `end_i` is per pixel,
  // so using it as the loop bound makes saturated pixels take fewer barriers
  // than unsaturated pixels. Iterate over the tile count and mask each pixel
  // with `global_i < end_i` inside the loop.
  for (int chunk_end = int(count); chunk_end > 0; chunk_end -= int(GSP_CHUNK)) {
    int chunk_start_i = max(0, chunk_end - int(GSP_CHUNK));
    uint chunk_start = uint(chunk_start_i);
    uint chunk_n = uint(chunk_end - chunk_start_i);
    for (uint i = tid; i < chunk_n; i += GSP_THREADS) sh_ids[i] = overflow_sorted_ids[start + chunk_start + i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    load_chunk_params(means2d, conics, colors, opacities, sh_ids, 0u, chunk_n, tid, sh_means, sh_conics, sh_colors, sh_opacities);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int local = int(chunk_n) - 1; local >= 0; --local) {
      uint global_i = chunk_start + uint(local);
      uint g = sh_ids[uint(local)];
      float2 m = sh_means[uint(local)];
      float3 q = load3_sh(sh_conics, uint(local));
      float3 c = load3_sh(sh_colors, uint(local));
      float opacity = sh_opacities[uint(local)];
      float l_gmx = 0.0f, l_gmy = 0.0f, l_ga = 0.0f, l_gb = 0.0f, l_gc = 0.0f, l_gop = 0.0f;
      float3 l_gcol = float3(0.0f);
      if (valid && global_i < end_i) {
        float alpha, raw_alpha, power; float2 d;
        if (eval_alpha(p, m, q.x, q.y, q.z, opacity, mf, alpha, raw_alpha, power, d)) {
          float denom = max(1.0f - alpha, mf.eps);
          float T_prev = T_cur / denom;
          float dot_gc = dot(go, c);
          float g_alpha = T_prev * (dot_gc - gT);
          l_gcol = go * (T_prev * alpha);
          float gate = (raw_alpha < mf.max_alpha) ? 1.0f : 0.0f;
          float g_raw = g_alpha * gate;
          float g_power = g_raw * raw_alpha;
          l_ga = g_power * (-0.5f) * d.x * d.x;
          l_gb = g_power * (-1.0f) * d.x * d.y;
          l_gc = g_power * (-0.5f) * d.y * d.y;
          float g_dx = g_power * (-(q.x * d.x + q.y * d.y));
          float g_dy = g_power * (-(q.y * d.x + q.z * d.y));
          l_gmx = -g_dx;
          l_gmy = -g_dy;
          l_gop = g_raw * (raw_alpha / max(opacity, mf.eps));
          gT = alpha * dot_gc + (1.0f - alpha) * gT;
          T_cur = T_prev;
        }
      }
      float4 s0 = simd_sum(float4(l_gmx, l_gmy, l_ga, l_gb));
      float4 s1 = simd_sum(float4(l_gc, l_gcol.x, l_gcol.y, l_gcol.z));
      float s2 = simd_sum(l_gop);
      if (simd_lane == 0u) { partial0[simd_group] = s0; partial1[simd_group] = s1; partial2[simd_group] = s2; }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (simd_group == 0u) {
        float4 v0 = (simd_lane < GSP_SIMDGROUPS) ? partial0[simd_lane] : float4(0.0f);
        float4 v1 = (simd_lane < GSP_SIMDGROUPS) ? partial1[simd_lane] : float4(0.0f);
        float v2 = (simd_lane < GSP_SIMDGROUPS) ? partial2[simd_lane] : 0.0f;
        float4 t0 = simd_sum(v0);
        float4 t1 = simd_sum(v1);
        float t2 = simd_sum(v2);
        if (simd_lane == 0u) {
          atomic_fetch_add_explicit(&g_means2d[g * 2u + 0u], t0.x, memory_order_relaxed);
          atomic_fetch_add_explicit(&g_means2d[g * 2u + 1u], t0.y, memory_order_relaxed);
          atomic_fetch_add_explicit(&g_conics[g * 3u + 0u], t0.z, memory_order_relaxed);
          atomic_fetch_add_explicit(&g_conics[g * 3u + 1u], t0.w, memory_order_relaxed);
          atomic_fetch_add_explicit(&g_conics[g * 3u + 2u], t1.x, memory_order_relaxed);
          atomic_add3(g_colors, g, float3(t1.y, t1.z, t1.w));
          atomic_fetch_add_explicit(&g_opacities[g], t2, memory_order_relaxed);
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

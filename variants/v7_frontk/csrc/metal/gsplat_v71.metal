#include <metal_stdlib>
using namespace metal;

#define kMaxFrontK 8u

struct ProjectedGaussian {
    float4 geom;   // mean_x, mean_y, opacity, pad
    float4 conic;  // a, b, c, pad
    float4 color;  // r, g, b, pad
};

struct Uniforms {
    uint width;
    uint height;
    uint gaussians;
    uint front_k;
    float alpha_threshold;
    float bg_r;
    float bg_g;
    float bg_b;
    float eps;
};

struct VSOut {
    float4 position [[position]];
    float4 geom;
    float4 conic;
    float4 color;
};

inline float safe_det(float a, float b, float c, float eps) {
    return max(a * c - b * b, eps);
}

inline bool alpha_support_params(float opacity, constant Uniforms& u, thread float& tau) {
    if (opacity <= u.alpha_threshold) return false;
    float ratio = max(u.alpha_threshold / max(opacity, u.eps), u.eps);
    tau = -2.0f * log(ratio);
    return isfinite(tau) && tau > 0.0f;
}

inline float4 corner_px(uint vid, float x0, float y0, float x1, float y1) {
    switch (vid & 3u) {
        case 0u: return float4(x0, y0, 0.0f, 1.0f);
        case 1u: return float4(x1, y0, 0.0f, 1.0f);
        case 2u: return float4(x0, y1, 0.0f, 1.0f);
        default: return float4(x1, y1, 0.0f, 1.0f);
    }
}

inline float px_to_ndc_x(float x, uint width) {
    return 2.0f * (x / float(width)) - 1.0f;
}

inline float px_to_ndc_y(float y, uint height) {
    return 1.0f - 2.0f * (y / float(height));
}

inline bool visible_at_pixel(ProjectedGaussian g, float2 p, constant Uniforms& u, thread float& raw_alpha, thread float& alpha, thread float2& d, thread float& power) {
    d = p - g.geom.xy;
    power = -0.5f * (g.conic.x * d.x * d.x + 2.0f * g.conic.y * d.x * d.y + g.conic.z * d.y * d.y);
    if (power > 0.0f) {
        raw_alpha = 0.0f;
        alpha = 0.0f;
        return false;
    }
    raw_alpha = g.geom.z * exp(power);
    alpha = min(0.99f, raw_alpha);
    return alpha >= u.alpha_threshold;
}

vertex VSOut ellipse_vs(
    uint vid [[vertex_id]],
    uint iid [[instance_id]],
    const device ProjectedGaussian* gaussians [[buffer(0)]],
    constant Uniforms& u [[buffer(1)]]) {
    VSOut out;
    // Standard source-over hardware blending matches the front-to-back
    // reference when the already depth-sorted buffer is drawn far-to-near.
    ProjectedGaussian g = gaussians[u.gaussians - 1u - iid];
    float tau = 0.0f;
    float4 pos = float4(-2.0f, -2.0f, 0.0f, 1.0f);
    if (alpha_support_params(g.geom.z, u, tau)) {
        float a = g.conic.x;
        float b = g.conic.y;
        float c = g.conic.z;
        float det = safe_det(a, b, c, u.eps);
        float hx = sqrt(max(tau * c / det, 0.0f));
        float hy = sqrt(max(tau * a / det, 0.0f));
        float x0 = clamp(g.geom.x - hx, 0.0f, float(u.width));
        float x1 = clamp(g.geom.x + hx, 0.0f, float(u.width));
        float y0 = clamp(g.geom.y - hy, 0.0f, float(u.height));
        float y1 = clamp(g.geom.y + hy, 0.0f, float(u.height));
        float4 cpx = corner_px(vid, x0, y0, x1, y1);
        pos = float4(px_to_ndc_x(cpx.x, u.width), px_to_ndc_y(cpx.y, u.height), 0.0f, 1.0f);
    }
    out.position = pos;
    out.geom = g.geom;
    out.conic = g.conic;
    out.color = g.color;
    return out;
}

fragment float4 gaussian_fs(VSOut in [[stage_in]], constant Uniforms& u [[buffer(0)]]) {
    float2 p = in.position.xy;
    float2 d = p - in.geom.xy;
    float power = -0.5f * (in.conic.x * d.x * d.x + 2.0f * in.conic.y * d.x * d.y + in.conic.z * d.y * d.y);
    if (power > 0.0f) discard_fragment();
    float alpha = min(0.99f, in.geom.z * exp(power));
    if (alpha < u.alpha_threshold) discard_fragment();
    return float4(in.color.xyz * alpha, alpha);
}

kernel void capture_front_k(
    const device ProjectedGaussian* gaussians [[buffer(0)]],
    constant Uniforms& u [[buffer(1)]],
    device int* front_ids [[buffer(2)]],
    device float* front_raw_alpha [[buffer(3)]],
    device int* front_count [[buffer(4)]],
    device uchar* overflow_mask [[buffer(5)]],
    uint gid [[thread_position_in_grid]]) {
    uint P = u.width * u.height;
    if (gid >= P) return;
    uint base_k = gid * u.front_k;
    for (uint j = 0; j < u.front_k; ++j) {
        front_ids[base_k + j] = -1;
        front_raw_alpha[base_k + j] = 0.0f;
    }
    front_count[gid] = 0;
    overflow_mask[gid] = 0;

    float2 p = float2(float(gid % u.width) + 0.5f, float(gid / u.width) + 0.5f);
    int captured = 0;
    for (uint i = 0; i < u.gaussians; ++i) {
        float raw_alpha = 0.0f;
        float alpha = 0.0f;
        float2 d;
        float power = 0.0f;
        if (!visible_at_pixel(gaussians[i], p, u, raw_alpha, alpha, d, power)) {
            continue;
        }
        if (captured < int(u.front_k)) {
            front_ids[base_k + uint(captured)] = int(i);
            front_raw_alpha[base_k + uint(captured)] = raw_alpha;
            captured += 1;
        } else {
            overflow_mask[gid] = 1;
            break;
        }
    }
    front_count[gid] = captured;
}

kernel void backward_front_k(
    const device ProjectedGaussian* gaussians [[buffer(0)]],
    const device float* grad_image [[buffer(1)]],
    const device int* front_ids [[buffer(2)]],
    const device float* front_raw_alpha [[buffer(3)]],
    const device int* front_count [[buffer(4)]],
    const device uchar* overflow_mask [[buffer(5)]],
    constant Uniforms& u [[buffer(6)]],
    device atomic_float* g_geom [[buffer(7)]],
    device atomic_float* g_conic [[buffer(8)]],
    device atomic_float* g_color [[buffer(9)]],
    uint gid [[thread_position_in_grid]]) {
    uint P = u.width * u.height;
    if (gid >= P) return;
    if (overflow_mask[gid] != 0) return;

    int n = front_count[gid];
    if (n <= 0) return;
    if (n > int(kMaxFrontK)) return;

    uint ids[kMaxFrontK];
    float raws[kMaxFrontK];
    float alphas[kMaxFrontK];
    float3 colors[kMaxFrontK];
    float3 suffix[kMaxFrontK + 1];

    uint base_k = gid * u.front_k;
    for (int j = 0; j < n; ++j) {
        int idx = front_ids[base_k + uint(j)];
        if (idx < 0) {
            n = j;
            break;
        }
        ids[j] = uint(idx);
        raws[j] = front_raw_alpha[base_k + uint(j)];
        alphas[j] = min(0.99f, raws[j]);
        colors[j] = gaussians[ids[j]].color.xyz;
    }
    if (n <= 0) return;

    suffix[n] = float3(u.bg_r, u.bg_g, u.bg_b);
    for (int j = n - 1; j >= 0; --j) {
        suffix[j] = alphas[j] * colors[j] + (1.0f - alphas[j]) * suffix[j + 1];
    }

    uint x = gid % u.width;
    uint y = gid / u.width;
    float2 p = float2(float(x) + 0.5f, float(y) + 0.5f);
    uint grad_base = gid * 3u;
    float3 go = float3(grad_image[grad_base + 0u], grad_image[grad_base + 1u], grad_image[grad_base + 2u]);
    float T = 1.0f;

    for (int j = 0; j < n; ++j) {
        uint idx = ids[j];
        ProjectedGaussian g = gaussians[idx];
        float2 d = p - g.geom.xy;
        float alpha = alphas[j];
        float raw_alpha = raws[j];
        float w = T * alpha;
        float3 color = g.color.xyz;
        float dL_dalpha = dot(go, T * (color - suffix[j + 1]));
        float dL_draw = (raw_alpha < 0.99f) ? dL_dalpha : 0.0f;
        float g_power = dL_draw * raw_alpha;
        float exp_term = raw_alpha / max(g.geom.z, u.eps);

        atomic_fetch_add_explicit(&g_color[idx * 4u + 0u], go.x * w, memory_order_relaxed);
        atomic_fetch_add_explicit(&g_color[idx * 4u + 1u], go.y * w, memory_order_relaxed);
        atomic_fetch_add_explicit(&g_color[idx * 4u + 2u], go.z * w, memory_order_relaxed);

        atomic_fetch_add_explicit(&g_geom[idx * 4u + 2u], dL_draw * exp_term, memory_order_relaxed);
        atomic_fetch_add_explicit(&g_geom[idx * 4u + 0u], g_power * (g.conic.x * d.x + g.conic.y * d.y), memory_order_relaxed);
        atomic_fetch_add_explicit(&g_geom[idx * 4u + 1u], g_power * (g.conic.y * d.x + g.conic.z * d.y), memory_order_relaxed);

        atomic_fetch_add_explicit(&g_conic[idx * 4u + 0u], g_power * (-0.5f * d.x * d.x), memory_order_relaxed);
        atomic_fetch_add_explicit(&g_conic[idx * 4u + 1u], g_power * (-d.x * d.y), memory_order_relaxed);
        atomic_fetch_add_explicit(&g_conic[idx * 4u + 2u], g_power * (-0.5f * d.y * d.y), memory_order_relaxed);

        T *= (1.0f - alpha);
    }
}

kernel void backward_overflow_replay(
    const device ProjectedGaussian* gaussians [[buffer(0)]],
    const device float* grad_image [[buffer(1)]],
    const device float* out_image [[buffer(2)]],
    const device uchar* overflow_mask [[buffer(3)]],
    constant Uniforms& u [[buffer(4)]],
    device atomic_float* g_geom [[buffer(5)]],
    device atomic_float* g_conic [[buffer(6)]],
    device atomic_float* g_color [[buffer(7)]],
    uint gid [[thread_position_in_grid]]) {
    uint P = u.width * u.height;
    if (gid >= P) return;
    if (overflow_mask[gid] == 0) return;

    uint x = gid % u.width;
    uint y = gid / u.width;
    float2 p = float2(float(x) + 0.5f, float(y) + 0.5f);
    uint base = gid * 3u;
    float3 go = float3(grad_image[base + 0u], grad_image[base + 1u], grad_image[base + 2u]);
    float3 outc = float3(out_image[base + 0u], out_image[base + 1u], out_image[base + 2u]);
    float3 A = float3(0.0f);
    float T = 1.0f;

    for (uint i = 0; i < u.gaussians; ++i) {
        ProjectedGaussian g = gaussians[i];
        float raw_alpha = 0.0f;
        float alpha = 0.0f;
        float2 d;
        float power = 0.0f;
        if (!visible_at_pixel(g, p, u, raw_alpha, alpha, d, power)) {
            continue;
        }
        float w = T * alpha;
        float3 color = g.color.xyz;
        float denom = max(T * (1.0f - alpha), u.eps);
        float3 behind = (outc - A - w * color) / denom;
        float dL_dalpha = dot(go, T * (color - behind));
        float dL_draw = (raw_alpha < 0.99f) ? dL_dalpha : 0.0f;
        float g_power = dL_draw * raw_alpha;
        float exp_term = raw_alpha / max(g.geom.z, u.eps);

        atomic_fetch_add_explicit(&g_color[i * 4u + 0u], go.x * w, memory_order_relaxed);
        atomic_fetch_add_explicit(&g_color[i * 4u + 1u], go.y * w, memory_order_relaxed);
        atomic_fetch_add_explicit(&g_color[i * 4u + 2u], go.z * w, memory_order_relaxed);

        atomic_fetch_add_explicit(&g_geom[i * 4u + 2u], dL_draw * exp_term, memory_order_relaxed);
        atomic_fetch_add_explicit(&g_geom[i * 4u + 0u], g_power * (g.conic.x * d.x + g.conic.y * d.y), memory_order_relaxed);
        atomic_fetch_add_explicit(&g_geom[i * 4u + 1u], g_power * (g.conic.y * d.x + g.conic.z * d.y), memory_order_relaxed);

        atomic_fetch_add_explicit(&g_conic[i * 4u + 0u], g_power * (-0.5f * d.x * d.x), memory_order_relaxed);
        atomic_fetch_add_explicit(&g_conic[i * 4u + 1u], g_power * (-d.x * d.y), memory_order_relaxed);
        atomic_fetch_add_explicit(&g_conic[i * 4u + 2u], g_power * (-0.5f * d.y * d.y), memory_order_relaxed);

        A += w * color;
        T *= (1.0f - alpha);
    }
}

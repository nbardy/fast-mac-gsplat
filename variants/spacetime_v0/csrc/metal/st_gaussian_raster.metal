#include <metal_stdlib>
using namespace metal;

// Metal starter kernels for spacetime Gaussian rasterization.
// This file is a prototype: validate against st_raster_ref.py before production use.

struct ProjectedGaussian {
    float4 mean_logAmp;   // xyz = sensor-time mean: pixel_x, pixel_y, time; w = logAmp
    float4 H0;            // xyz = row 0 of 3x3 precision H
    float4 H1;            // xyz = row 1
    float4 H2;            // xyz = row 2
    float4 covDiag_depth; // xyz = diag(inv(H)); w = approximate depth for sort/debug
    float4 color_flags;   // xyz = color; w unused
};

struct RasterConfig {
    uint width;
    uint height;
    uint frameCount;
    uint primitiveCount;

    uint tileW;
    uint tileH;
    uint tilesX;
    uint tilesY;

    uint timeBins;
    uint maxRefs;       // capacity of cellIndices
    float time0;
    float time1;
    float truncR;
    uint compositingMode; // 0 = order-independent volume, 1 = sorted-alpha approximation
};

inline uint cell_id(uint cx, uint cy, uint ct, constant RasterConfig& cfg) {
    return (ct * cfg.tilesY + cy) * cfg.tilesX + cx;
}

inline float eval_d2(const device ProjectedGaussian& pg, float3 y) {
    float3 d = y - pg.mean_logAmp.xyz;
    float3 h0 = pg.H0.xyz;
    float3 h1 = pg.H1.xyz;
    float3 h2 = pg.H2.xyz;
    float3 Hd = float3(dot(h0, d), dot(h1, d), dot(h2, d));
    return dot(d, Hd);
}

inline float eval_mass(const device ProjectedGaussian& pg, float3 y, float truncR2) {
    float d2 = eval_d2(pg, y);
    if (d2 > truncR2) {
        return 0.0f;
    }
    return exp(pg.mean_logAmp.w - 0.5f * d2);
}

kernel void clear_uints(
    device uint* values [[buffer(0)]],
    constant uint& count [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) return;
    values[gid] = 0u;
}

kernel void count_cell_overlaps(
    device const ProjectedGaussian* pgs [[buffer(0)]],
    device atomic_uint* cellCounts [[buffer(1)]],
    constant RasterConfig& cfg [[buffer(2)]],
    uint pid [[thread_position_in_grid]]
) {
    if (pid >= cfg.primitiveCount) return;

    const device ProjectedGaussian& pg = pgs[pid];
    float R = cfg.truncR;
    float3 radii = R * sqrt(max(pg.covDiag_depth.xyz, float3(0.0f)));
    float3 lo = pg.mean_logAmp.xyz - radii;
    float3 hi = pg.mean_logAmp.xyz + radii;

    float dtBin = (cfg.time1 - cfg.time0) / max(float(cfg.timeBins), 1.0f);

    int minX = max(0, int(floor(lo.x / float(cfg.tileW))));
    int maxX = min(int(cfg.tilesX) - 1, int(floor(hi.x / float(cfg.tileW))));
    int minY = max(0, int(floor(lo.y / float(cfg.tileH))));
    int maxY = min(int(cfg.tilesY) - 1, int(floor(hi.y / float(cfg.tileH))));
    int minT = max(0, int(floor((lo.z - cfg.time0) / dtBin)));
    int maxT = min(int(cfg.timeBins) - 1, int(floor((hi.z - cfg.time0) / dtBin)));

    if (maxX < minX || maxY < minY || maxT < minT) return;

    for (int ct = minT; ct <= maxT; ++ct) {
        for (int cy = minY; cy <= maxY; ++cy) {
            for (int cx = minX; cx <= maxX; ++cx) {
                uint c = cell_id(uint(cx), uint(cy), uint(ct), cfg);
                atomic_fetch_add_explicit(&cellCounts[c], 1u, memory_order_relaxed);
            }
        }
    }
}

kernel void fill_cell_overlaps(
    device const ProjectedGaussian* pgs [[buffer(0)]],
    device const uint* cellOffsets [[buffer(1)]],
    device atomic_uint* cellWriteHeads [[buffer(2)]],
    device uint* cellIndices [[buffer(3)]],
    device atomic_uint* overflowCounter [[buffer(4)]],
    constant RasterConfig& cfg [[buffer(5)]],
    uint pid [[thread_position_in_grid]]
) {
    if (pid >= cfg.primitiveCount) return;

    const device ProjectedGaussian& pg = pgs[pid];
    float R = cfg.truncR;
    float3 radii = R * sqrt(max(pg.covDiag_depth.xyz, float3(0.0f)));
    float3 lo = pg.mean_logAmp.xyz - radii;
    float3 hi = pg.mean_logAmp.xyz + radii;

    float dtBin = (cfg.time1 - cfg.time0) / max(float(cfg.timeBins), 1.0f);

    int minX = max(0, int(floor(lo.x / float(cfg.tileW))));
    int maxX = min(int(cfg.tilesX) - 1, int(floor(hi.x / float(cfg.tileW))));
    int minY = max(0, int(floor(lo.y / float(cfg.tileH))));
    int maxY = min(int(cfg.tilesY) - 1, int(floor(hi.y / float(cfg.tileH))));
    int minT = max(0, int(floor((lo.z - cfg.time0) / dtBin)));
    int maxT = min(int(cfg.timeBins) - 1, int(floor((hi.z - cfg.time0) / dtBin)));

    if (maxX < minX || maxY < minY || maxT < minT) return;

    for (int ct = minT; ct <= maxT; ++ct) {
        for (int cy = minY; cy <= maxY; ++cy) {
            for (int cx = minX; cx <= maxX; ++cx) {
                uint c = cell_id(uint(cx), uint(cy), uint(ct), cfg);
                uint local = atomic_fetch_add_explicit(&cellWriteHeads[c], 1u, memory_order_relaxed);
                uint dst = cellOffsets[c] + local;
                if (dst < cfg.maxRefs) {
                    cellIndices[dst] = pid;
                } else {
                    atomic_fetch_add_explicit(overflowCounter, 1u, memory_order_relaxed);
                }
            }
        }
    }
}

kernel void render_order_independent(
    device const ProjectedGaussian* pgs [[buffer(0)]],
    device const uint* cellOffsets [[buffer(1)]],
    device const uint* cellCounts [[buffer(2)]],
    device const uint* cellIndices [[buffer(3)]],
    device float4* outRGBA [[buffer(4)]],
    constant RasterConfig& cfg [[buffer(5)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint x = gid.x;
    uint ypix = gid.y;
    uint f = gid.z;
    if (x >= cfg.width || ypix >= cfg.height || f >= cfg.frameCount) return;

    float denom = max(float(cfg.frameCount - 1u), 1.0f);
    float tau = cfg.time0 + (cfg.time1 - cfg.time0) * (float(f) / denom);
    float dtBin = (cfg.time1 - cfg.time0) / max(float(cfg.timeBins), 1.0f);

    uint cx = min(x / cfg.tileW, cfg.tilesX - 1u);
    uint cy = min(ypix / cfg.tileH, cfg.tilesY - 1u);
    uint ct = min(uint(max(0.0f, floor((tau - cfg.time0) / dtBin))), cfg.timeBins - 1u);
    uint c = cell_id(cx, cy, ct, cfg);

    uint begin = cellOffsets[c];
    uint count = cellCounts[c];
    float3 yy = float3(float(x) + 0.5f, float(ypix) + 0.5f, tau);
    float R2 = cfg.truncR * cfg.truncR;

    float totalMass = 0.0f;
    float3 weightedColor = float3(0.0f);

    for (uint j = 0; j < count; ++j) {
        uint pid = cellIndices[begin + j];
        if (pid >= cfg.primitiveCount) continue;
        const device ProjectedGaussian& pg = pgs[pid];
        float mass = eval_mass(pg, yy, R2);
        totalMass += mass;
        weightedColor += mass * pg.color_flags.xyz;
    }

    float alpha = 1.0f - exp(-totalMass);
    float3 rgb = (totalMass > 1e-8f) ? (weightedColor / totalMass) * alpha : float3(0.0f);

    uint outIdx = (f * cfg.height + ypix) * cfg.width + x;
    outRGBA[outIdx] = float4(rgb, alpha);
}

// Assumes cellIndices are already sorted front-to-back by depth for each cell.
kernel void render_sorted_alpha(
    device const ProjectedGaussian* pgs [[buffer(0)]],
    device const uint* cellOffsets [[buffer(1)]],
    device const uint* cellCounts [[buffer(2)]],
    device const uint* cellIndices [[buffer(3)]],
    device float4* outRGBA [[buffer(4)]],
    constant RasterConfig& cfg [[buffer(5)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint x = gid.x;
    uint ypix = gid.y;
    uint f = gid.z;
    if (x >= cfg.width || ypix >= cfg.height || f >= cfg.frameCount) return;

    float denom = max(float(cfg.frameCount - 1u), 1.0f);
    float tau = cfg.time0 + (cfg.time1 - cfg.time0) * (float(f) / denom);
    float dtBin = (cfg.time1 - cfg.time0) / max(float(cfg.timeBins), 1.0f);

    uint cx = min(x / cfg.tileW, cfg.tilesX - 1u);
    uint cy = min(ypix / cfg.tileH, cfg.tilesY - 1u);
    uint ct = min(uint(max(0.0f, floor((tau - cfg.time0) / dtBin))), cfg.timeBins - 1u);
    uint c = cell_id(cx, cy, ct, cfg);

    uint begin = cellOffsets[c];
    uint count = cellCounts[c];
    float3 yy = float3(float(x) + 0.5f, float(ypix) + 0.5f, tau);
    float R2 = cfg.truncR * cfg.truncR;

    float T = 1.0f;
    float3 rgb = float3(0.0f);

    for (uint j = 0; j < count; ++j) {
        uint pid = cellIndices[begin + j];
        if (pid >= cfg.primitiveCount) continue;
        const device ProjectedGaussian& pg = pgs[pid];
        float mass = eval_mass(pg, yy, R2);
        if (mass <= 0.0f) continue;
        float a = 1.0f - exp(-mass);
        rgb += T * a * pg.color_flags.xyz;
        T *= (1.0f - a);
        if (T < 1e-4f) break;
    }

    uint outIdx = (f * cfg.height + ypix) * cfg.width + x;
    outRGBA[outIdx] = float4(rgb, 1.0f - T);
}

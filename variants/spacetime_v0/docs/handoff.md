# Spacetime Gaussian Rasterization: Research Engineer Handoff
## 0. Objective

Build a Metal-based rasterizer for dynamic worlds represented by **4D spacetime Gaussian primitives**. The renderer should avoid per-frame rebuilds of splat state. Instead it builds a single **sensor-spacetime index** over `(tile_x, tile_y, time_bin)` and renders any frame by querying the relevant temporal slice.

The target representation is not:

```text
for frame f:
    evaluate/deform all 3D splats at t_f
    project all splats
    bin all splats
    rasterize
```

The target is:

```text
once per clip / keyframe window:
    project 4D primitives into sensor-time ellipsoids
    bin ellipsoids into a 3D tile-time index

for each frame f:
    for visible tile/time cell:
        evaluate only candidate spacetime footprints
```

Rendering all pixels of all frames is still output-bound: producing `F * W * H` pixels has an unavoidable `Omega(FWH)` write cost. The sublinear win is in **primitive indexing and candidate enumeration across frames**, replacing repeated per-frame primitive processing with one spacetime pass.

---

## 1. Representation

Spacetime is

\[
M = \mathbb R^3_x \times \mathbb R_t.
\]

A primitive is a 4D Gaussian density:

\[
G_i(z)
=
\frac{1}{(2\pi)^2 |\Sigma_i|^{1/2}}
\exp\left[-\frac12 (z-\mu_i)^\top Q_i (z-\mu_i)\right],
\quad
Q_i = \Sigma_i^{-1},
\quad
z=(x,y,z,t).
\]

Each primitive stores:

- `mu: float4` — center in world spacetime.
- `Q: float4x4` — inverse covariance / precision matrix.
- `logNorm: float` — usually `log(alpha) - 2 log(2pi) - 0.5 logdet(Sigma)`.
- `color: float3` — RGB radiance or albedo.
- optional view-dependent coefficients.
- optional finite support interval `(t_min, t_max)` for early rejection.

The extinction/radiance field is

\[
\sigma(z)=\sum_i \alpha_i G_i(z),
\qquad
n(z,\omega)=\sum_i \alpha_i G_i(z)c_i(\omega).
\]

---

## 2. Camera lift into spacetime

For pixel coordinate `u=(u,v)`, sensor time `tau`, and ray depth `s`, the camera ray lifted into spacetime is

\[
\gamma_{u,\tau}(s)
=
\left(o(\tau)+s d(u,\tau),\ \tau - \frac{s}{c}\right).
\]

For most initial implementations use `c = infinity`, so

\[
\gamma_{u,\tau}(s)=\left(o(\tau)+s d(u,\tau),\tau\right).
\]

For local affine projection inside a tile:

\[
\gamma_y(s)=b + A y + s v,
\qquad
 y=(u,v,\tau)\in\mathbb R^3,
\]

where:

- `b: float4` is the spacetime point at tile origin and ray depth zero.
- `A: float4x3` maps sensor-time differential coordinates into spacetime.
- `v: float4` is the ray direction in spacetime.

For an orthographic camera:

\[
A =
\begin{bmatrix}
\Delta_x & 0 & 0 \\
0 & \Delta_y & 0 \\
0 & 0 & 0 \\
0 & 0 & 1
\end{bmatrix},
\qquad
v=(0,0,1,0)^\top.
\]

For perspective cameras, use one of two modes:

1. **Exact per-pixel ray evaluation:** no local affine approximation; slower but exact.
2. **Tile-local affine approximation:** compute `b,A,v` at tile center; fast; acceptable for small tiles.

---

## 3. Projecting a 4D Gaussian into sensor-time

We need the footprint of a 4D Gaussian on `(u,v,tau)` after integrating along ray depth `s`:

\[
\rho_i(y)=\int_{-\infty}^{\infty} \alpha_i G_i(b + Ay + sv)\,ds.
\]

Let

\[
c_i=b-\mu_i,
\qquad
Q=Q_i,
\qquad
a=v^\top Qv.
\]

Then

\[
Q_\perp = Q - Qv(v^\top Qv)^{-1}v^\top Q.
\]

The ray-integrated density is

\[
\rho_i(y)
=
\exp\left(
\ell_i
-
\frac12 (y-m_i)^\top H_i (y-m_i)
\right),
\]

with

\[
H_i=A^\top Q_\perp A,
\]

\[
h_i=A^\top Q_\perp c_i,
\]

\[
m_i=-H_i^{-1}h_i,
\]

and

\[
\ell_i
=
\log\alpha_i
-2\log(2\pi)
-\frac12\log|\Sigma_i|
+\frac12\log\frac{2\pi}{a}
-rac12\left(c_i^\top Q_\perp c_i-h_i^\top H_i^{-1}h_i\right).
\]

This gives a 3D Gaussian footprint in sensor-time. Store this as:

```c
ProjectedGaussian {
    meanY = (pixel_x, pixel_y, time);
    H     = 3x3 precision in sensor-time;
    logAmp;
    color;
    depth coefficients for approximate sorting;
}
```

### Depth estimate for compositing

The most likely ray depth at sensor-time point `y` is

\[
s_i^\star(y)
=
-\frac{v^\top Q(c_i+Ay)}{v^\top Qv}.
\]

For tile-level sorting, evaluate at cell center `y_cell`:

\[
s_{i,cell}=s_i^\star(y_{cell}).
\]

---

## 4. Truncation and binning

Choose a truncation radius `R`, usually `R=3` or `R=4`.

A projected primitive contributes where

\[
(y-m_i)^\top H_i(y-m_i) \le R^2.
\]

For conservative axis-aligned bounds use covariance

\[
C_i=H_i^{-1}.
\]

The half-width in coordinate `j` is

\[
r_j=R\sqrt{(C_i)_{jj}}.
\]

Therefore the AABB is

\[
[m_x-r_x,m_x+r_x]\times[m_y-r_y,m_y+r_y]\times[m_t-r_t,m_t+r_t].
\]

Map this to cell coordinates:

\[
cell_x = \left\lfloor\frac{x}{tileW}\right\rfloor,
\quad
cell_y = \left\lfloor\frac{y}{tileH}\right\rfloor,
\quad
cell_t = \left\lfloor\frac{t-t_0}{\Delta t_{bin}}\right\rfloor.
\]

This builds a 3D sparse index over `(tile_x, tile_y, time_bin)`.

---

## 5. Complexity target

Let:

- `N` = number of spacetime primitives.
- `F` = number of output frames.
- `P = W*H` = pixels per frame.
- `C` = number of tile-time cells.
- `k_cell` = average candidates per active tile-time cell.
- `B` = number of temporal bins.

Naive dynamic splatting cost:

\[
O(FN + F P k_{pixel}).
\]

Spacetime-indexed cost:

\[
O\left(N\,\bar c_{cover}\right)
+
O\left(F P\,\bar k_{cell}\right),
\]

where `c_cover` is number of tile-time cells covered by a primitive. Crucially, the primitive indexing term uses temporal bins, not frames:

\[
\bar c_{cover} \approx \bar c_{xy}\,\bar c_t,
\qquad
\bar c_t \le B,
\qquad
B \ll F \text{ when bins cover multiple frames.}
\]

Exact frame output cannot be sublinear in `F`, but primitive work can be.

---

## 6. GPU pipeline

### Pass 0: project primitives

Input: `STGaussian[N]`.

Output: `ProjectedGaussian[N]`.

For perspective rendering, either:

- do this per large macro-tile, producing multiple projected footprints per primitive, or
- skip projection and evaluate exact ray/primitive line integral in render kernel.

### Pass 1: count cell overlaps

Input: `ProjectedGaussian[N]`.

Output: `cellCounts[numCells]`.

Each thread handles one primitive, computes its AABB in tile-time cell space, and atomically increments every overlapped cell.

### Pass 2: prefix sum

Input: `cellCounts[numCells]`.

Output:

- `cellOffsets[numCells + 1]`
- total index capacity `cellOffsets[numCells]`

Implementation options:

1. CPU scan for prototype.
2. GPU hierarchical exclusive scan.
3. MPSGraph / custom compute scan if already available in project.

### Pass 3: fill cell primitive lists

Input:

- `ProjectedGaussian[N]`
- `cellOffsets`
- zeroed `cellWriteHeads[numCells]`

Output:

- `cellIndices[totalRefs]`

Same AABB loop as Pass 1, but writes primitive IDs into compact cell lists.

### Pass 4: optional sort inside each cell

For volume-style order-independent rendering, skip this.

For alpha compositing, sort each cell's primitive IDs by approximate depth at the cell center:

\[
s_{i,cell}=s_i^\star(y_{cell}).
\]

Sorting options:

1. CPU sort per cell for prototype.
2. GPU bitonic sort for cells with bounded list length.
3. Approximate bucket sort into depth bins.

### Pass 5: render frames

Dispatch grid:

```text
(width, height, frameCount)
```

Each GPU thread renders one output pixel at one frame.

For pixel `(x,y,f)`:

1. Convert `f` to sensor time `tau_f`.
2. Compute cell `(x/tileW, y/tileH, timeBin(tau_f))`.
3. Read candidate list.
4. Evaluate sensor-time Gaussian weight:

\[
d_i^2=(y-m_i)^\top H_i(y-m_i).
\]

5. Reject if `d_i^2 > R^2`.
6. Convert mass to opacity:

\[
a_i=1-\exp(-\rho_i(y)).
\]

7. Composite.

Order-independent volume mode:

\[
\tau=\sum_i \rho_i(y),
\qquad
C=\frac{\sum_i \rho_i(y)c_i}{\tau+\epsilon}\left(1-e^{-\tau}\right).
\]

Sorted alpha mode:

\[
C=\sum_i T_i a_i c_i,
\qquad
T_i=\prod_{j<i}(1-a_j).
\]

---

## 7. Metal implementation notes

- Use `dispatchThreads(_:threadsPerThreadgroup:)` for arbitrary `(W,H,F)` render grids.
- Use `dispatchThreadgroups(_:threadsPerThreadgroup:)` when manually aligning to tile/threadgroup sizes.
- Use argument buffers once the renderer has many buffers/textures; they reduce CPU binding overhead.
- Keep structs 16-byte aligned. Prefer `float4` rows over packed matrices in device buffers.
- Avoid per-pixel dynamic allocation.
- Keep candidate list lengths bounded. Provide overflow counters for debugging.
- Start with order-independent volume compositing; add depth sorting later.
- Validate Metal against the Python CPU reference on small scenes.

---

## 8. Acceptance tests

### Math tests

1. A constant-velocity 3D Gaussian should equal one tilted 4D Gaussian.
2. Projected footprint of an orthographic primitive should match numerical line integration.
3. Integrated mass should be invariant under shifting the integration ray parameter.
4. Sensor-time AABB should contain all samples where Mahalanobis distance is below `R^2`.

### Renderer tests

1. Static scene: output should match ordinary 3D Gaussian splatting.
2. Moving dot: output trajectory should be continuous without per-frame primitive mutation.
3. Motion blur: finite shutter should smear along time dimension.
4. Rolling shutter: row-dependent `tau` should produce skew.
5. Increasing `F` while holding time bins fixed should not trigger re-binning.

### Performance metrics

Track:

- `N primitives`
- `numCells`
- `totalRefs`
- `mean candidates/cell`
- `p95 candidates/cell`
- `overflow cells`
- projection ms
- count/fill ms
- render ms/frame
- total ms for `F` frames
- speedup over per-frame binning

---

## 9. Milestone plan

### Milestone A: CPU reference

- Orthographic camera.
- 4D Gaussian projection into 3D sensor-time Gaussian.
- Tile-time binning.
- Order-independent volume compositing.

### Milestone B: Metal prototype

- Upload projected gaussians from CPU.
- GPU count/fill/render.
- CPU prefix scan.
- Buffer output `float4[W*H*F]`.

### Milestone C: Full GPU indexing

- GPU prefix scan.
- GPU cell sorting or depth bucket compositing.
- Overflow handling.

### Milestone D: Perspective + local affine

- Per-macro-tile projection.
- Exact fallback for high curvature tiles.
- Error estimate for tile-local approximation.

### Milestone E: Training integration

- Differentiable approximation path.
- Save per-pixel primitive IDs / weights for backward pass.
- Gradient kernels for `mu`, `Q`, `logAmp`, and color.

---

## 10. Known risks

1. **Occlusion correctness.** Order-independent mode is not physically exact for opaque surfaces. Use sorted alpha or layered depth bins.
2. **Perspective approximation.** Large tiles cause footprint errors. Use smaller macro-tiles or exact ray evaluation.
3. **Temporal aliasing.** Too few time bins cause missing fast primitives. Choose bin width based on maximum projected temporal covariance and motion.
4. **Atomic pressure.** Count/fill can bottleneck when many large primitives overlap the same cells. Split huge primitives or handle them in a separate global list.
5. **Covariance degeneracy.** Clamp eigenvalues of `Sigma`; regularize `H` before inversion.

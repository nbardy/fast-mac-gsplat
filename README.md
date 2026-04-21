# fast-mac-gsplat

Torch-first, Metal-backed Gaussian rasterizer fast path for Apple Silicon.

Bringing World Model Research to your mac

## Benchmarks

  | Splats | Mode | Torch | v5 | % Increase |
  |---:|---|---:|---:|---:|
  | 512 | Forward | 751.724 ms | 3.014 ms | +24,840% |
  | 512 | Forward+backward | 5109.671 ms | 7.838 ms | +65,094% |
  | 1024 | Forward | 1712.214 ms | 4.355 ms | +39,219% |
  | 1024 | Forward+backward | OOM | 11.373 ms | ♾️ %|
  | 64k | Forward | 94992.427 ms | 9.644 ms | +984,907% |
  | 64k | Forward+backward | OOM | 27.457 ms | ♾️ % |

For broad sweeps across renderers, resolutions, splat counts, and projected
splat distributions:

```bash
python benchmarks/benchmark_full_matrix.py \
  --resolutions 512x512,1024x512,1920x1080,4096x4096 \
  --splats 512,2048,65536 \
  --batch-sizes 1,4 \
  --distributions microbench_uniform_random,sparse_screen,clustered_hot_tiles,layered_depth,overflow_adversarial \
  --renderers torch_direct,v2_fastpath,v3_candidate,v5_batched,v6_direct,v6_auto,v6_upgrade_direct,v6_upgrade_auto,v7_hardware \
  --modes forward,forward_backward \
  --output-md benchmarks/full_rasterizer_benchmark.md
```

The full benchmark writes a Markdown report and marks Torch rows as skipped
when the dense reference would be too large.

## Variants

- `v2_fastpath`: low-overhead single-image compute path.
- `v3_candidate`: strongest large-scene single-image training path so far.
- `v5_batched`: native `[B,G,...]` batch API.
- `v6_direct`: batch-focused direct compute path; current default for B>1.
- `v6_auto`: v6 with conservative active-tile policy for clustered/overflow cases.
- `v6_upgrade_direct`: chief-scientist v6 upgrade handoff, direct mode.
- `v6_upgrade_auto`: chief-scientist v6 upgrade handoff, active-policy auto mode.
- `v7_hardware`: experimental Metal render-pipeline forward path with compute backward.

The current `variants/v6` branch already includes later local engineering fixes
on top of the original v6 line. `variants/v6_upgrade` is preserved separately so
the source handoff can be tested without overwriting the current v6 baseline.

## API

Install the v5 Torch + Metal package directly from GitHub:

```bash
python -m pip install "git+https://github.com/nbardy/fast-mac-gsplat.git#subdirectory=variants/v5"
```

Use projected 2D splats on MPS tensors:

```python
import torch
from torch_gsplat_bridge_v5 import RasterConfig, rasterize_projected_gaussians

device = torch.device("mps")
batch_size, gaussians = 1, 1024
height, width = 1024, 512

means2d = torch.rand((batch_size, gaussians, 2), device=device)
means2d[..., 0] *= width
means2d[..., 1] *= height

sigma = torch.full((batch_size, gaussians), 4.0, device=device)
inv_var = 1.0 / (sigma * sigma)
conics = torch.stack([inv_var, torch.zeros_like(inv_var), inv_var], dim=-1)

colors = torch.rand((batch_size, gaussians, 3), device=device)
opacities = torch.full((batch_size, gaussians), 0.5, device=device)
depths = torch.rand((batch_size, gaussians), device=device)

config = RasterConfig(height=height, width=width, background=(1.0, 1.0, 1.0))
image = rasterize_projected_gaussians(means2d, conics, colors, opacities, depths, config)

loss = image.square().mean()
loss.backward()
```

Inputs may be `[G, ...]` for one image or `[B, G, ...]` for batched rendering. The output is `[H, W, 3]` or `[B, H, W, 3]`.

## Design goals

- Keep the Python training loop in **PyTorch**.
- Use **Metal kernels** for the renderer hot path.
- Minimize memory during training by saving only compact tile bins, **not** dense `G x H x W` activations.
- Keep backward differentiable via **recompute**, not via storing per-pixel alpha / transmittance graphs.

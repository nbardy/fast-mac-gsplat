# fast-mac-gsplat

Torch-first, Metal-backed Gaussian rasterizer fast path for Apple Silicon.

Bringing World Model Research to your mac

## Benchmarks

  | Splats | Mode | Torch | v5 | % Increase |
  |---:|---|---:|---:|---:|
  | 512 | Forward | 751.724 ms | 3.014 ms | +24,840% |
  | 512 | Forward+backward | 5109.671 ms | 7.838 ms | +65,094% |
  | 1024 | Forward | 1712.214 ms | 4.355 ms | +39,219% |
  | 1024 | Forward+backward | OOM | 11.373 ms | n/a |
  | 64k | Forward | 94992.427 ms | 9.644 ms | +984,907% |
  | 64k | Forward+backward | OOM | 27.457 ms | n/a |


## Design goals

- Keep the Python training loop in **PyTorch**.
- Use **Metal kernels** for the renderer hot path.
- Minimize memory during training by saving only compact tile bins, **not** dense `G x H x W` activations.
- Keep backward differentiable via **recompute**, not via storing per-pixel alpha / transmittance graphs.



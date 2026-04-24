from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from torch_gsplat_bridge_v71 import RasterConfig, rasterize_projected_gaussians


def main() -> None:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    use_reference = device.type != "mps"
    torch.manual_seed(0)
    B, G, H, W = 1, 256, 256, 256
    means2d = torch.rand(B, G, 2, device=device, dtype=torch.float32) * torch.tensor([W, H], device=device, dtype=torch.float32)
    conics = torch.rand(B, G, 3, device=device, dtype=torch.float32)
    conics[..., 0] += 0.2
    conics[..., 2] += 0.2
    conics[..., 1] *= 0.05
    colors = torch.rand(B, G, 3, device=device, dtype=torch.float32)
    opacities = torch.rand(B, G, device=device, dtype=torch.float32) * 0.9 + 0.05
    depths = torch.rand(B, G, device=device, dtype=torch.float32)
    cfg = RasterConfig(height=H, width=W, front_k=2, use_reference_when_unavailable=use_reference, return_debug_state=False)

    t0 = time.perf_counter()
    out = rasterize_projected_gaussians(means2d, conics, colors, opacities, depths, cfg)
    if device.type == "mps":
        torch.mps.synchronize()
    t1 = time.perf_counter()
    print("forward ms:", (t1 - t0) * 1000.0)

    means2d = means2d.clone().requires_grad_(True)
    conics = conics.clone().requires_grad_(True)
    colors = colors.clone().requires_grad_(True)
    opacities = opacities.clone().requires_grad_(True)
    t2 = time.perf_counter()
    out = rasterize_projected_gaussians(means2d, conics, colors, opacities, depths, cfg)
    loss = out.square().mean()
    loss.backward()
    if device.type == "mps":
        torch.mps.synchronize()
    t3 = time.perf_counter()
    print("forward+backward ms:", (t3 - t2) * 1000.0)


if __name__ == "__main__":
    main()

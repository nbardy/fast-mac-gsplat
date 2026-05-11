from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from torch import Tensor

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from torch_gsplat_bridge_star_uvt import make_gate0_scene, stable_backward_samples  # noqa: E402


def _reference_render(
    ma: Tensor,
    q_uvt: Tensor,
    depth0: Tensor,
    depth_beta: Tensor,
    opacity: Tensor,
    color: Tensor,
    config,
) -> Tensor:
    bg = torch.tensor(config.background, dtype=torch.float32, device=ma.device)
    frames = []
    for frame in range(config.frames):
        t = float(frame) - 0.5 * float(config.frames - 1)
        rows = []
        for y in range(config.height):
            pixels = []
            for x in range(config.width):
                a = torch.tensor([x + 0.5, y + 0.5, t], dtype=torch.float32, device=ma.device)
                d = a.unsqueeze(0) - ma
                qv = (
                    q_uvt[:, 0] * d[:, 0].square()
                    + 2.0 * q_uvt[:, 1] * d[:, 0] * d[:, 1]
                    + 2.0 * q_uvt[:, 2] * d[:, 0] * d[:, 2]
                    + q_uvt[:, 3] * d[:, 1].square()
                    + 2.0 * q_uvt[:, 4] * d[:, 1] * d[:, 2]
                    + q_uvt[:, 5] * d[:, 2].square()
                )
                alpha = torch.clamp(opacity * torch.exp(-0.5 * qv), max=config.max_alpha)
                depths = depth0 + ((a.unsqueeze(0) - ma) * depth_beta).sum(dim=-1)
                order = sorted(range(ma.shape[0]), key=lambda idx: (float(depths[idx].detach().cpu()), idx))
                accum = torch.zeros((3,), dtype=torch.float32, device=ma.device)
                transmittance = torch.tensor(1.0, dtype=torch.float32, device=ma.device)
                for idx in order:
                    if float(alpha[idx].detach().cpu()) < config.alpha_threshold:
                        continue
                    accum = accum + transmittance * alpha[idx] * color[idx]
                    transmittance = transmittance * (1.0 - alpha[idx])
                    if float(transmittance.detach().cpu()) <= config.transmittance_threshold:
                        break
                pixels.append(accum + transmittance * bg)
            rows.append(torch.stack(pixels, dim=0))
        frames.append(torch.stack(rows, dim=0))
    return torch.stack(frames, dim=0)


def _reduce(ids: Tensor, samples: Tensor, tube_count: int, trailing: int | None) -> Tensor:
    ids_cpu = ids.detach().cpu().to(torch.long)
    valid = ids_cpu >= 0
    if trailing is None:
        out = torch.zeros((tube_count,), dtype=torch.float32)
        out.index_add_(0, ids_cpu[valid], samples.detach().cpu()[valid])
        return out
    out = torch.zeros((tube_count, trailing), dtype=torch.float32)
    out.index_add_(0, ids_cpu[valid], samples.detach().cpu()[valid])
    return out


def _max_abs(left: Tensor, right: Tensor) -> float:
    return float((left.detach().cpu() - right.detach().cpu()).abs().max().item())


def main() -> None:
    if not torch.backends.mps.is_available():
        print(json.dumps({"metal_skipped": "MPS is not available"}, indent=2, sort_keys=True))
        return
    ma, q_uvt, depth0, depth_beta, opacity, color, config = make_gate0_scene("crossing_depth", device="mps")
    ma = ma.detach().requires_grad_(True)
    q_uvt = q_uvt.detach().requires_grad_(True)
    depth0 = depth0.detach().requires_grad_(True)
    depth_beta = depth_beta.detach().requires_grad_(True)
    opacity = opacity.detach().requires_grad_(True)
    color = color.detach().requires_grad_(True)
    image_ref = _reference_render(ma, q_uvt, depth0, depth_beta, opacity, color, config)
    grad_image = torch.linspace(0.1, 0.9, image_ref.numel(), dtype=torch.float32, device="mps").view_as(image_ref).contiguous()
    torch.sum(image_ref * grad_image).backward()

    ids, grad_ma_samples, grad_q_samples, grad_opacity_samples, grad_color_samples, tile_unstable = stable_backward_samples(
        ma.detach(),
        q_uvt.detach(),
        depth0.detach(),
        depth_beta.detach(),
        opacity.detach(),
        color.detach(),
        grad_image,
        config,
    )
    unstable_count = int((tile_unstable.detach().cpu() > 0).sum().item())
    if unstable_count == 0:
        raise AssertionError("expected crossing_depth backward smoke to use unstable fallback")
    grad_ma = _reduce(ids, grad_ma_samples, ma.shape[0], 3)
    grad_q = _reduce(ids, grad_q_samples, ma.shape[0], 6)
    grad_opacity = _reduce(ids, grad_opacity_samples, ma.shape[0], None)
    grad_color = _reduce(ids, grad_color_samples, ma.shape[0], 3)
    errors = {
        "ma": _max_abs(grad_ma, ma.grad),
        "q_uvt": _max_abs(grad_q, q_uvt.grad),
        "opacity": _max_abs(grad_opacity, opacity.grad),
        "color": _max_abs(grad_color, color.grad),
    }
    for name, value in errors.items():
        if value > 1.0e-3:
            raise AssertionError(f"{name} unstable Metal backward mismatch: {value}")
    print(
        json.dumps(
            {
                "scene": "crossing_depth",
                "device": "mps",
                "max_abs_errors": errors,
                "unstable_tile_count": unstable_count,
                "valid_gradient_entries": int((ids.detach().cpu() >= 0).sum().item()),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

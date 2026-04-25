from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from torch_gsplat_bridge_v9_hw_output_planes import (
    probe_direct_output_formats,
    probe_hw_interop,
    render_constant_direct_format,
    render_constant_rgba,
    render_constant_rgba_direct,
    render_gaussian_eval_format,
    render_gaussian_eval_format_sorted,
    render_gaussian_eval_rgba,
    render_gaussian_eval_rgba_sorted,
    validate_direct_output_formats,
    validate_gaussian_eval_output_formats,
)


def main() -> None:
    status = probe_hw_interop(compile_pipelines=True, compile_advanced=True, run_render_probe=False)
    print(json.dumps(status.as_dict(), indent=2, sort_keys=True))
    assert status.native_extension_loaded
    assert status.native_probe_available
    assert status.metal_available
    assert status.render_pipeline_ready is True, status.render_pipeline_error

    if not torch.backends.mps.is_available():
        print("MPS unavailable; native compile probe passed, render-to-tensor validation skipped.")
        return

    rgba = (0.125, 0.5, 0.875, 1.0)
    out = render_constant_rgba(9, 7, rgba)
    assert out.device.type == "mps"
    assert tuple(out.shape) == (9, 7, 4)
    expected = torch.tensor(rgba, dtype=torch.float32).view(1, 1, 4).expand(9, 7, 4)
    max_err = float((out.detach().cpu() - expected).abs().max().item())
    assert max_err == 0.0, max_err

    direct = render_constant_rgba_direct(16, 16, rgba)
    direct_expected = torch.tensor(rgba, dtype=torch.float32).view(1, 1, 4).expand(16, 16, 4)
    direct_max_err = float((direct.detach().cpu() - direct_expected).abs().max().item())
    assert direct_max_err == 0.0, direct_max_err

    half_direct = render_constant_direct_format("rgba16f", 16, 32, rgba)
    assert half_direct.dtype == torch.float16
    assert tuple(half_direct.shape) == (16, 32, 4)
    half_expected = torch.tensor(rgba, dtype=torch.float16).view(1, 1, 4).expand(16, 32, 4).float()
    half_direct_max_err = float((half_direct.detach().cpu().float() - half_expected).abs().max().item())
    assert half_direct_max_err == 0.0, half_direct_max_err

    format_rows = probe_direct_output_formats(height=1, widths=(16, 32, 64, 1920, 3840))
    assert any(row["format"] == "rgba16f" and row["buffer_backed_texture_created"] for row in format_rows)
    validations = validate_direct_output_formats(height=16, rgba=rgba)
    assert all(row["validated"] for row in validations), validations

    means = torch.tensor([[8.5, 8.5]], dtype=torch.float32, device="mps")
    conics = torch.tensor([[1.0, 0.0, 1.0]], dtype=torch.float32, device="mps")
    colors = torch.tensor([[0.25, 0.5, 0.75]], dtype=torch.float32, device="mps")
    opacities = torch.tensor([0.8], dtype=torch.float32, device="mps")
    gaussian = render_gaussian_eval_rgba(means, conics, colors, opacities, 16, 16, direct=True)
    assert gaussian.device.type == "mps"
    assert tuple(gaussian.shape) == (16, 16, 4)
    gaussian_cpu = gaussian.detach().cpu()
    expected_center = torch.tensor([0.2, 0.4, 0.6, 0.8], dtype=torch.float32)
    center_max_err = float((gaussian_cpu[8, 8] - expected_center).abs().max().item())
    assert center_max_err <= 1.0e-5, center_max_err
    clear_max_err = float(gaussian_cpu[0, 0].abs().max().item())
    assert clear_max_err == 0.0, clear_max_err

    gaussian16 = render_gaussian_eval_format("rgba16f", means, conics, colors, opacities, 16, 32, direct=True)
    assert gaussian16.device.type == "mps"
    assert gaussian16.dtype == torch.float16
    assert tuple(gaussian16.shape) == (16, 32, 4)
    gaussian16_cpu = gaussian16.detach().cpu().float()
    expected_center16 = expected_center.half().float()
    center16_max_err = float((gaussian16_cpu[8, 8] - expected_center16).abs().max().item())
    assert center16_max_err <= 5.0e-4, center16_max_err
    clear16_max_err = float(gaussian16_cpu[0, 0].abs().max().item())
    assert clear16_max_err == 0.0, clear16_max_err

    gaussian_validations = validate_gaussian_eval_output_formats(height=16)
    assert all(row["validated"] for row in gaussian_validations), gaussian_validations

    two_means = torch.tensor([[16.5, 8.5], [16.5, 8.5]], dtype=torch.float32, device="mps")
    two_conics = torch.tensor([[1.0, 0.0, 1.0], [1.0, 0.0, 1.0]], dtype=torch.float32, device="mps")
    two_colors = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float32, device="mps")
    two_opacities = torch.tensor([0.5, 0.5], dtype=torch.float32, device="mps")
    two_depths = torch.tensor([1.0, 0.0], dtype=torch.float32, device="mps")
    reverse = torch.tensor([1, 0], dtype=torch.int64, device="mps")

    input_order = render_gaussian_eval_format(
        "rgba32f", two_means, two_conics, two_colors, two_opacities, 16, 32, direct=True
    )
    reverse_order = render_gaussian_eval_format(
        "rgba32f",
        two_means.index_select(0, reverse),
        two_conics.index_select(0, reverse),
        two_colors.index_select(0, reverse),
        two_opacities.index_select(0, reverse),
        16,
        32,
        direct=True,
    )
    order_delta = float((input_order.detach().cpu()[8, 16] - reverse_order.detach().cpu()[8, 16]).abs().max().item())
    assert order_delta > 0.1, order_delta

    sorted_asc = render_gaussian_eval_rgba_sorted(
        two_means, two_conics, two_colors, two_opacities, two_depths, 16, 32, direct=True
    )
    sorted_asc_repeat = render_gaussian_eval_rgba_sorted(
        two_means, two_conics, two_colors, two_opacities, two_depths, 16, 32, direct=True
    )
    repeat_err = float((sorted_asc.detach().cpu() - sorted_asc_repeat.detach().cpu()).abs().max().item())
    assert repeat_err == 0.0, repeat_err
    sorted_asc_manual_err = float((sorted_asc.detach().cpu() - reverse_order.detach().cpu()).abs().max().item())
    assert sorted_asc_manual_err <= 1.0e-6, sorted_asc_manual_err

    sorted_desc = render_gaussian_eval_rgba_sorted(
        two_means, two_conics, two_colors, two_opacities, two_depths, 16, 32, direct=True, descending=True
    )
    sorted_desc_manual_err = float((sorted_desc.detach().cpu() - input_order.detach().cpu()).abs().max().item())
    assert sorted_desc_manual_err <= 1.0e-6, sorted_desc_manual_err

    equal_depths = torch.zeros((2,), dtype=torch.float32, device="mps")
    stable_tie = render_gaussian_eval_rgba_sorted(
        two_means, two_conics, two_colors, two_opacities, equal_depths, 16, 32, direct=True
    )
    stable_tie_err = float((stable_tie.detach().cpu() - input_order.detach().cpu()).abs().max().item())
    assert stable_tie_err <= 1.0e-6, stable_tie_err

    sorted16 = render_gaussian_eval_format_sorted(
        "rgba16f", two_means, two_conics, two_colors, two_opacities, two_depths, 16, 32, direct=True
    )
    assert sorted16.dtype == torch.float16
    sorted16_manual_err = float((sorted16.detach().cpu().float() - reverse_order.detach().cpu().half().float()).abs().max().item())
    assert sorted16_manual_err <= 5.0e-4, sorted16_manual_err

    checked = probe_hw_interop(compile_pipelines=True, compile_advanced=True, run_render_probe=True)
    assert checked.render_to_mps_tensor_validated, checked.as_dict()
    assert checked.direct_render_to_mps_tensor_validated, checked.as_dict()
    assert checked.gaussian_eval_rgba_validated, checked.as_dict()
    assert checked.gaussian_eval_rgba16_validated, checked.as_dict()
    assert all(row["validated"] for row in checked.direct_output_format_validation), checked.as_dict()
    assert all(row["validated"] for row in checked.gaussian_output_format_validation), checked.as_dict()
    print(f"render_to_mps_tensor_max_abs_err={checked.render_to_mps_tensor_max_abs_err}")
    print(f"gaussian_eval_rgba_max_abs_err={checked.gaussian_eval_rgba_max_abs_err}")
    print(f"gaussian_eval_rgba16_max_abs_err={checked.gaussian_eval_rgba16_max_abs_err}")


if __name__ == "__main__":
    main()

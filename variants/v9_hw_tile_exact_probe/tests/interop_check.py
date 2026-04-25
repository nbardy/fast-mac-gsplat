from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from torch_gsplat_bridge_v9_hw_tile_exact import (
    estimate_tile_exact_memory,
    probe_hw_interop,
    render_constant_rgba,
    render_constant_rgba_direct,
    run_tile_exact_execution_probe,
    run_tile_exact_gaussian_probe,
    run_tile_exact_overlap_probe,
)


def _tile_max_counts(pixel_counts: torch.Tensor, tile_size: int) -> torch.Tensor:
    values = []
    height, width = pixel_counts.shape
    for tile_y in range(0, height, tile_size):
        for tile_x in range(0, width, tile_size):
            tile = pixel_counts[tile_y : tile_y + tile_size, tile_x : tile_x + tile_size]
            values.append(int(tile.max().item()))
    return torch.tensor(values, dtype=torch.int32)


def _tile_origin_transmittance(target: torch.Tensor, tile_size: int) -> torch.Tensor:
    values = []
    height, width, _ = target.shape
    for tile_y in range(0, height, tile_size):
        for tile_x in range(0, width, tile_size):
            values.append(float(target[tile_y, tile_x, 3].item()))
    return torch.tensor(values, dtype=torch.float32)


def gaussian_reference(height: int, width: int, splats) -> tuple[torch.Tensor, torch.Tensor]:
    ys = torch.arange(height, dtype=torch.float32).view(height, 1) + 0.5
    xs = torch.arange(width, dtype=torch.float32).view(1, width) + 0.5
    color_accum = torch.zeros((height, width, 3), dtype=torch.float32)
    transmittance = torch.ones((height, width), dtype=torch.float32)
    candidate_counts = torch.zeros((height, width), dtype=torch.int32)
    stopped = torch.zeros((height, width), dtype=torch.bool)
    for splat in splats:
        mean_x, mean_y, conic_x, conic_y, conic_z, opacity, red, green, blue = (float(v) for v in splat)
        active = ~stopped
        candidate_counts[active] += 1

        dx = xs - mean_x
        dy = ys - mean_y
        power = -0.5 * (conic_x * dx * dx + 2.0 * conic_y * dx * dy + conic_z * dy * dy)
        alpha = torch.clamp(opacity * torch.exp(power), max=0.99)
        alpha = torch.where((power > 0.0) | (alpha < 1.0 / 255.0), torch.zeros_like(alpha), alpha)
        alpha = torch.where(active, alpha, torch.zeros_like(alpha))

        color = torch.tensor((red, green, blue), dtype=torch.float32).view(1, 1, 3)
        color_accum = color_accum + transmittance.unsqueeze(-1) * alpha.unsqueeze(-1) * color
        transmittance = torch.where(active, transmittance * (1.0 - alpha), transmittance)
        stopped |= active & (transmittance <= 1.0e-4)
    return torch.cat([color_accum, transmittance.unsqueeze(-1)], dim=-1), candidate_counts


def main() -> None:
    status = probe_hw_interop(compile_pipelines=True, compile_advanced=True, run_render_probe=False)
    print(json.dumps(status.as_dict(), indent=2, sort_keys=True))
    assert status.native_extension_loaded
    assert status.native_probe_available
    assert status.metal_available
    assert status.render_pipeline_ready is True, status.render_pipeline_error
    assert status.tile_layouts, status.as_dict()

    if status.tile_pipeline_ready:
        assert status.tile_imageblock_sample_length is not None
        assert status.tile_imageblock_memory_16x16 is not None
        assert status.tile_imageblock_memory_32x32 is not None
        assert status.tile_execution_probe_available is True, status.tile_execution_probe_error
        assert status.tile_exact_overlap_probe_available is True, status.tile_exact_overlap_probe_error
        assert status.tile_exact_gaussian_probe_available is True, status.tile_exact_gaussian_probe_error
        assert status.tile_exact_imageblock_sample_length is not None
        assert status.tile_exact_imageblock_memory_16x16 is not None
    else:
        print(f"Tile pipeline unavailable; execution probe skipped: {status.tile_pipeline_error}")

    estimate = estimate_tile_exact_memory(height=2160, width=3840, tile_size=16)
    print(json.dumps({"tile_exact_4k_estimate": estimate}, indent=2, sort_keys=True))
    assert estimate["tile_count"] == 32400
    assert estimate["final_T_stop_count_frame_bytes"] == 2160 * 3840 * 8

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

    if status.tile_execution_probe_available:
        tile_probe = run_tile_exact_execution_probe(32, 32, 32)
        target = tile_probe["target"]
        reports = tile_probe["tile_reports"]
        assert tuple(target.shape) == (32, 32, 4)
        assert tuple(reports.shape) == (1, 4)
        expected_clear = torch.tensor((0.03125, 0.0625, 0.125, 1.0), dtype=torch.float32).view(1, 1, 4)
        target_max_err = float((target.detach().cpu() - expected_clear.expand(32, 32, 4)).abs().max().item())
        assert target_max_err == 0.0, target_max_err
        report_cpu = reports.detach().cpu()
        assert float(report_cpu[0, 0].item()) == 9013.0, report_cpu
        assert float(report_cpu[0, 1].item()) >= 32.0, report_cpu
        assert float(report_cpu[0, 2].item()) >= 32.0, report_cpu
        assert float(report_cpu[0, 3].item()) == float(0x5A5A + 1), report_cpu

    if status.tile_exact_overlap_probe_available:
        exact_probe = run_tile_exact_overlap_probe(32, 32, 16)
        target = exact_probe["target"]
        tile_stop_counts = exact_probe["tile_stop_counts"]
        tile_reports = exact_probe["tile_reports"]
        assert tuple(target.shape) == (32, 32, 4)
        assert tuple(tile_stop_counts.shape) == (4,)
        assert tuple(tile_reports.shape) == (4, 4)
        expected_rgba = tuple(float(x) for x in exact_probe["expected_rgba"])
        expected = torch.tensor(expected_rgba, dtype=torch.float32).view(1, 1, 4).expand(32, 32, 4)
        max_err = float((target.detach().cpu() - expected).abs().max().item())
        assert max_err <= 1.0e-6, (max_err, exact_probe.get("expected_semantic"))
        stop_cpu = tile_stop_counts.detach().cpu()
        expected_stop = int(exact_probe["expected_tile_stop_count"])
        assert torch.equal(stop_cpu, torch.full_like(stop_cpu, expected_stop)), stop_cpu
        reports_cpu = tile_reports.detach().cpu()
        assert torch.equal(reports_cpu[:, 0], torch.full_like(reports_cpu[:, 0], float(expected_stop))), reports_cpu
        expected_final_t = float(exact_probe["expected_final_T"])
        assert torch.allclose(reports_cpu[:, 1], torch.full_like(reports_cpu[:, 1], expected_final_t)), reports_cpu
        assert torch.equal(reports_cpu[:, 2], torch.zeros_like(reports_cpu[:, 2])), reports_cpu
        assert torch.equal(reports_cpu[:, 3], torch.arange(4, dtype=torch.float32)), reports_cpu
        print(f"tile_exact_overlap_max_abs_err={max_err}")
        print(f"tile_exact_overlap_tile_stop_counts={stop_cpu.tolist()}")
        try:
            run_tile_exact_overlap_probe(32, 32, 32)
        except ValueError as exc:
            assert "32x32" in str(exc)
        else:
            raise AssertionError("32x32 exact overlap probe should remain fail-closed")

        gaussian_probe = run_tile_exact_gaussian_probe(32, 32, 16)
        target = gaussian_probe["target"]
        tile_stop_counts = gaussian_probe["tile_stop_counts"]
        tile_reports = gaussian_probe["tile_reports"]
        assert tuple(target.shape) == (32, 32, 4)
        assert tuple(tile_stop_counts.shape) == (4,)
        assert tuple(tile_reports.shape) == (4, 4)
        expected, pixel_counts = gaussian_reference(32, 32, gaussian_probe["input_splats"])
        target_cpu = target.detach().cpu()
        gaussian_max_err = float((target_cpu - expected).abs().max().item())
        assert gaussian_max_err <= 2.0e-5, (gaussian_max_err, gaussian_probe.get("expected_semantic"))
        expected_gaussian_stop_counts = _tile_max_counts(pixel_counts, 16)
        gaussian_stop_cpu = tile_stop_counts.detach().cpu()
        assert torch.equal(gaussian_stop_cpu, expected_gaussian_stop_counts), (
            gaussian_stop_cpu,
            expected_gaussian_stop_counts,
        )
        gaussian_reports_cpu = tile_reports.detach().cpu()
        assert torch.equal(gaussian_reports_cpu[:, 0].to(torch.int32), expected_gaussian_stop_counts), (
            gaussian_reports_cpu,
            expected_gaussian_stop_counts,
        )
        expected_report_t = _tile_origin_transmittance(expected, 16)
        assert torch.allclose(gaussian_reports_cpu[:, 1], expected_report_t, atol=2.0e-5, rtol=0.0), (
            gaussian_reports_cpu,
            expected_report_t,
        )
        assert torch.equal(gaussian_reports_cpu[:, 2], torch.zeros_like(gaussian_reports_cpu[:, 2])), (
            gaussian_reports_cpu,
            target_cpu,
        )
        assert torch.equal(gaussian_reports_cpu[:, 3], torch.arange(4, dtype=torch.float32)), gaussian_reports_cpu
        print(f"tile_exact_gaussian_max_abs_err={gaussian_max_err}")
        print(f"tile_exact_gaussian_tile_stop_counts={gaussian_stop_cpu.tolist()}")
        try:
            run_tile_exact_gaussian_probe(32, 32, 32)
        except ValueError as exc:
            assert "32x32" in str(exc)
        else:
            raise AssertionError("32x32 exact gaussian probe should remain fail-closed")
    else:
        try:
            run_tile_exact_overlap_probe(32, 32, 16)
        except RuntimeError as exc:
            assert "tile exact" in str(exc).lower() or "imageblock" in str(exc).lower()
        else:
            raise AssertionError("tile exact overlap probe was unavailable but native API did not fail closed")

    checked = probe_hw_interop(compile_pipelines=True, compile_advanced=True, run_render_probe=True)
    assert checked.render_to_mps_tensor_validated, checked.as_dict()
    assert checked.direct_render_to_mps_tensor_validated, checked.as_dict()
    print(f"render_to_mps_tensor_max_abs_err={checked.render_to_mps_tensor_max_abs_err}")


if __name__ == "__main__":
    main()

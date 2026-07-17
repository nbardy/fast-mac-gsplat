from __future__ import annotations

import importlib
import sys
from pathlib import Path

import torch


VARIANT_ROOT = Path(__file__).resolve().parents[1]
if str(VARIANT_ROOT) not in sys.path:
    sys.path.insert(0, str(VARIANT_ROOT))

curve_tube = importlib.import_module("research_project.trainer_harness.curve_tube")
bridge = importlib.import_module("torch_gsplat_bridge_star_prt")
from research_project.trainer_harness.curve_tube_smoke import run_smoke  # noqa: E402


SHAPE_DOC_TOKENS = (
    "x0 [N,3]",
    "velocity [N,3]",
    "h_coeff [N,H,3]",
    "lambda_uv [N,3]",
    "depth_coeff [N,H]",
    "centers [N,F,2]",
    "depth [N,F]",
    "curve_uv_depth [F,N,3]",
    "image [F,H_px,W_px,3]",
)


def _assert_shape(tensor: torch.Tensor, expected: tuple[int, ...]) -> None:
    assert tuple(int(dim) for dim in tensor.shape) == expected


def _tiny_static_scene() -> dict[str, torch.Tensor]:
    frames = 3
    times = curve_tube.centered_frame_times(frames)
    k = torch.tensor(
        [
            [8.0, 0.0, 3.0],
            [0.0, 8.0, 2.5],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    return {
        "x0": torch.tensor([[0.0, 0.0, 4.0], [0.18, -0.12, 5.0]], dtype=torch.float32),
        "velocity": torch.tensor([[0.03, 0.02, 0.01], [-0.02, 0.01, -0.015]], dtype=torch.float32),
        "t0": torch.zeros(2, dtype=torch.float32),
        "precision_xy": torch.tensor([[2.0, 2.5], [1.8, 2.2]], dtype=torch.float32),
        "lambda_t": torch.tensor([0.02, 0.03], dtype=torch.float32),
        "opacity": torch.tensor([0.55, 0.45], dtype=torch.float32),
        "color": torch.tensor([[0.8, 0.2, 0.1], [0.1, 0.55, 0.9]], dtype=torch.float32),
        "K_seq": k.view(1, 3, 3).repeat(frames, 1, 1),
        "w2c_seq": torch.eye(4, dtype=torch.float32).view(1, 4, 4).repeat(frames, 1, 1),
        "times": times,
    }


def test_curve_tube_smoke() -> None:
    report = run_smoke(torch.device("cpu"))
    assert report["image_shape"] == [5, 32, 32, 3]
    assert report["tube_count"] == 2
    assert report["static_center_residual_px"] < 1.0e-3
    assert report["moving_center_residual_px"] < 1.0e-3
    assert report["moving_curve_fit_error"] < 1.0e-3
    assert report["prt_curve_psnr"] > 20.0
    assert report["pass"] is True


def test_documented_shape_contract_and_bridge_dense_import() -> None:
    design_doc = VARIANT_ROOT / "docs" / "curve_tube_design.md"
    text = design_doc.read_text()
    for token in SHAPE_DOC_TOKENS:
        assert token in text

    scene = _tiny_static_scene()
    prt = curve_tube.compile_projective_rational_tubes(
        x0=scene["x0"],
        velocity=scene["velocity"],
        t0=scene["t0"],
        precision_xy=scene["precision_xy"],
        lambda_t=scene["lambda_t"],
        opacity=scene["opacity"],
        color=scene["color"],
        K_seq=scene["K_seq"],
        w2c_seq=scene["w2c_seq"],
        times=scene["times"],
        camera_degree=0,
    )
    centers = curve_tube.eval_projective_centers(prt, scene["times"])
    depth = curve_tube.eval_scalar_poly(prt.depth_coeff, scene["times"], t_center=prt.t_center, t_scale=prt.t_scale)
    direct_centers, direct_depth = curve_tube.sample_world_tube_projection(
        scene["x0"],
        scene["velocity"],
        scene["t0"],
        scene["K_seq"],
        scene["w2c_seq"],
        scene["times"],
    )

    _assert_shape(prt.h_coeff, (2, 2, 3))
    _assert_shape(prt.lambda_uv, (2, 3))
    _assert_shape(prt.depth_coeff, (2, 2))
    _assert_shape(centers, (2, 3, 2))
    _assert_shape(depth, (2, 3))
    assert float((centers - direct_centers).abs().max()) < 1.0e-5
    assert float((depth - direct_depth).abs().max()) < 1.0e-5

    dense_config = curve_tube.CurveTubeRenderConfig(height=5, width=6, frames=3)
    bridge_config = bridge.PRTRenderConfig(height=5, width=6, frames=3)
    dense_image = curve_tube.dense_render_projective_rational_tubes(prt, dense_config)
    bridge_curve = bridge.compile_projective_rational_curve(prt.h_coeff, prt.center_t, bridge_config)
    bridge_image = bridge.render_projective_rational_tubes(
        prt.h_coeff,
        prt.lambda_uv,
        prt.lambda_t,
        prt.center_t,
        prt.opacity,
        prt.color,
        bridge_config,
        backend="dense",
    )
    _assert_shape(bridge_curve, (3, 2, 3))
    _assert_shape(dense_image, (3, 5, 6, 3))
    _assert_shape(bridge_image, (3, 5, 6, 3))
    assert float((dense_image - bridge_image).abs().max()) < 1.0e-5


if __name__ == "__main__":
    test_curve_tube_smoke()
    test_documented_shape_contract_and_bridge_dense_import()

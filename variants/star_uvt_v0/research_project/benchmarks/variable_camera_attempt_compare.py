from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch

from multicam_heldout_compare import (
    DEFAULT_BASELINE_CONFIG,
    ProjectedTubeSequence,
    UVTRenderConfig,
    config_data_for_run,
    load_config_file,
    load_multicam_video_bundle,
    project_world_tube_sequence,
    render_projected_sequence,
    resolve_device,
    resolve_dynaworld_path,
    resolve_variant_path,
    serialize_config_value,
    synchronize_device,
    write_json,
)
from multicam_train_step_timing_probe import (
    apply_uvt_tile_env,
    build_world_tube_model,
    camera_sequences_for_view,
    project_world_tube_sequence_dynamic_first_order,
    project_world_tube_sequence_per_frame_camera,
    project_world_tube_sequence_projective_first_order,
    project_world_tube_sequence_segmented_camera,
    render_projected_sequence_per_frame_camera,
    timed,
)


def psnr(rendered: torch.Tensor, target: torch.Tensor) -> float:
    mse = float((rendered - target).square().mean().detach().cpu())
    if mse <= 0.0:
        return float("inf")
    return -10.0 * math.log10(mse)


def l1(rendered: torch.Tensor, target: torch.Tensor) -> float:
    return float((rendered - target).abs().mean().detach().cpu())


def render_static_view(
    *,
    model,
    K_seq: torch.Tensor,
    w2c_seq: torch.Tensor,
    config: UVTRenderConfig,
    backend: str,
    reduction_mode: str,
    sample_emission_mode: str,
) -> tuple[Any, dict[str, float]]:
    projected = project_world_tube_sequence(model, K_seq[0], w2c_seq[0], config)
    rendered = render_projected_sequence(
        projected,
        config,
        backend=backend,
        reduction_mode=reduction_mode,
        sample_emission_mode=sample_emission_mode,
    )
    return rendered, {
        "projected_tube_count": float(projected.ma.shape[0]),
        "mean_segments_per_tube": 1.0,
        "temporal_chunk_count": 1.0,
    }


def render_dynamic_first_order(
    *,
    model,
    K_seq: torch.Tensor,
    w2c_seq: torch.Tensor,
    config: UVTRenderConfig,
    frames: int,
    backend: str,
    reduction_mode: str,
    sample_emission_mode: str,
) -> tuple[Any, dict[str, float]]:
    projected = project_world_tube_sequence_dynamic_first_order(
        model=model,
        K_seq=K_seq,
        w2c_seq=w2c_seq,
        config=config,
        full_frames=frames,
        frame_start=0,
    )
    rendered = render_projected_sequence(
        projected,
        config,
        backend=backend,
        reduction_mode=reduction_mode,
        sample_emission_mode=sample_emission_mode,
    )
    return rendered, {
        "projected_tube_count": float(projected.ma.shape[0]),
        "mean_segments_per_tube": 1.0,
        "temporal_chunk_count": 1.0,
    }


def render_projective_first_order(
    *,
    model,
    K_seq: torch.Tensor,
    w2c_seq: torch.Tensor,
    config: UVTRenderConfig,
    frames: int,
    backend: str,
    reduction_mode: str,
    sample_emission_mode: str,
) -> tuple[Any, dict[str, float]]:
    projected = project_world_tube_sequence_projective_first_order(
        model=model,
        K_seq=K_seq,
        w2c_seq=w2c_seq,
        config=config,
        full_frames=frames,
        frame_start=0,
    )
    rendered = render_projected_sequence(
        projected,
        config,
        backend=backend,
        reduction_mode=reduction_mode,
        sample_emission_mode=sample_emission_mode,
    )
    return rendered, {
        "projected_tube_count": float(projected.ma.shape[0]),
        "mean_segments_per_tube": 1.0,
        "temporal_chunk_count": 1.0,
    }


def render_segmented(
    *,
    model,
    K_seq: torch.Tensor,
    w2c_seq: torch.Tensor,
    config: UVTRenderConfig,
    frames: int,
    frames_per_segment: int,
    backend: str,
    reduction_mode: str,
    sample_emission_mode: str,
) -> tuple[Any, dict[str, float]]:
    projected, diagnostics = project_world_tube_sequence_segmented_camera(
        model=model,
        K_seq=K_seq,
        w2c_seq=w2c_seq,
        config=config,
        full_frames=frames,
        frame_start=0,
        frames_per_segment=frames_per_segment,
    )
    rendered = render_projected_sequence(
        projected,
        config,
        backend=backend,
        reduction_mode=reduction_mode,
        sample_emission_mode=sample_emission_mode,
    )
    return rendered, diagnostics


def render_per_frame_reference(
    *,
    model,
    bundle,
    view: int,
    K_seq: torch.Tensor,
    w2c_seq: torch.Tensor,
    config: UVTRenderConfig,
    frames: int,
    backend: str,
    reduction_mode: str,
    sample_emission_mode: str,
) -> tuple[Any, dict[str, float]]:
    projected_frames = project_world_tube_sequence_per_frame_camera(
        model=model,
        bundle=bundle,
        view=view,
        frame_start=0,
        config=config,
        full_frames=frames,
        K_seq=K_seq,
        w2c_seq=w2c_seq,
    )
    rendered = render_projected_sequence_per_frame_camera(
        projected_frames=projected_frames,
        config=config,
        backend=backend,
        reduction_mode=reduction_mode,
        sample_emission_mode=sample_emission_mode,
    )
    return rendered, {
        "projected_tube_count": float(sum(int(item.ma.shape[0]) for item in projected_frames)),
        "mean_segments_per_tube": float(frames),
        "temporal_chunk_count": float(frames),
    }


def run_compare(args: argparse.Namespace) -> dict[str, Any]:
    device = resolve_device(args.device)
    if args.uvt_render_backend == "metal_tile" and device.type != "mps":
        raise ValueError("--uvt-render-backend=metal_tile requires device=mps")
    torch.manual_seed(args.seed)
    config = load_config_file(resolve_dynaworld_path(args.baseline_config))
    data_cfg = config_data_for_run(config, target_size=args.target_size, max_frames=args.max_frames)
    bundle = load_multicam_video_bundle(
        data_cfg=data_cfg,
        camera_cfg=dict(config["camera"]),
        target_size=args.target_size,
        device=device,
    )
    _, frames, _, height, width = bundle.train_frames.shape
    if args.view < 0 or args.view >= int(bundle.train_frames.shape[0]):
        raise ValueError(f"--view={args.view} outside train view count {int(bundle.train_frames.shape[0])}")
    render_config = UVTRenderConfig(
        height=height,
        width=width,
        frames=frames,
        tile_x=args.uvt_tile_x,
        tile_y=args.uvt_tile_y,
        tile_t=args.uvt_tile_t,
        tile_capacity=args.uvt_tile_capacity,
    )
    apply_uvt_tile_env(render_config)
    model = build_world_tube_model(bundle, args, device)
    K_seq, w2c_seq = camera_sequences_for_view(bundle, args, view=args.view, frames=frames)
    target = bundle.train_frames[args.view].permute(0, 2, 3, 1).contiguous()

    rows: dict[str, dict[str, Any]] = {}
    with torch.no_grad():
        reference, reference_elapsed = timed(
            device,
            lambda: render_per_frame_reference(
                model=model,
                bundle=bundle,
                view=args.view,
                K_seq=K_seq,
                w2c_seq=w2c_seq,
                config=render_config,
                frames=frames,
                backend=args.uvt_render_backend,
                reduction_mode=args.uvt_reduction_mode,
                sample_emission_mode=args.uvt_sample_emission_mode,
            ),
        )
        reference_rendered, reference_diagnostics = reference
        rows["per_frame_loop_reference"] = {
            **reference_diagnostics,
            "render_s": reference_elapsed,
            "psnr_vs_reference": float("inf"),
            "l1_vs_reference": 0.0,
            "target_psnr": psnr(reference_rendered.rgb, target),
            "target_l1": l1(reference_rendered.rgb, target),
        }

        mode_fns = {
            "static_view": lambda: render_static_view(
                model=model,
                K_seq=K_seq,
                w2c_seq=w2c_seq,
                config=render_config,
                backend=args.uvt_render_backend,
                reduction_mode=args.uvt_reduction_mode,
                sample_emission_mode=args.uvt_sample_emission_mode,
            ),
            "dynamic_first_order": lambda: render_dynamic_first_order(
                model=model,
                K_seq=K_seq,
                w2c_seq=w2c_seq,
                config=render_config,
                frames=frames,
                backend=args.uvt_render_backend,
                reduction_mode=args.uvt_reduction_mode,
                sample_emission_mode=args.uvt_sample_emission_mode,
            ),
            "projective_first_order": lambda: render_projective_first_order(
                model=model,
                K_seq=K_seq,
                w2c_seq=w2c_seq,
                config=render_config,
                frames=frames,
                backend=args.uvt_render_backend,
                reduction_mode=args.uvt_reduction_mode,
                sample_emission_mode=args.uvt_sample_emission_mode,
            ),
        }
        for frames_per_segment in args.segment_frames:
            mode_fns[f"segmented_f{frames_per_segment}"] = (
                lambda frames_per_segment=frames_per_segment: render_segmented(
                    model=model,
                    K_seq=K_seq,
                    w2c_seq=w2c_seq,
                    config=render_config,
                    frames=frames,
                    frames_per_segment=frames_per_segment,
                    backend=args.uvt_render_backend,
                    reduction_mode=args.uvt_reduction_mode,
                    sample_emission_mode=args.uvt_sample_emission_mode,
                )
            )

        for mode, fn in mode_fns.items():
            rendered_tuple, elapsed = timed(device, fn)
            rendered, diagnostics = rendered_tuple
            tube_count = float(diagnostics["projected_tube_count"])
            rows[mode] = {
                **diagnostics,
                "render_s": elapsed,
                "speedup_vs_per_frame_reference": reference_elapsed / max(elapsed, 1.0e-12),
                "tube_count_ratio_vs_per_frame_reference": tube_count
                / max(float(rows["per_frame_loop_reference"]["projected_tube_count"]), 1.0),
                "psnr_vs_reference": psnr(rendered.rgb, reference_rendered.rgb),
                "l1_vs_reference": l1(rendered.rgb, reference_rendered.rgb),
                "target_psnr": psnr(rendered.rgb, target),
                "target_l1": l1(rendered.rgb, target),
            }

    synthetic_camera_motion_active = any(
        float(getattr(args, name))
        for name in (
            "uvt_synthetic_pan_x",
            "uvt_synthetic_pan_y",
            "uvt_synthetic_dolly_z",
            "uvt_synthetic_zoom",
            "uvt_synthetic_principal_x",
            "uvt_synthetic_principal_y",
        )
    )
    return {
        "meta": {
            "baseline_config": str(resolve_dynaworld_path(args.baseline_config)),
            "target_size": args.target_size,
            "max_frames": args.max_frames,
            "loaded_frame_count": frames,
            "device": str(device),
            "view": args.view,
            "train_camera": bundle.train_camera_names[args.view],
            "uvt_tubes": args.uvt_tubes,
            "render_backend": args.uvt_render_backend,
            "reduction_mode": args.uvt_reduction_mode,
            "sample_emission_mode": args.uvt_sample_emission_mode,
            "segment_frames": args.segment_frames,
            "synthetic_camera_motion": {
                "active": synthetic_camera_motion_active,
                "pan_x": args.uvt_synthetic_pan_x,
                "pan_y": args.uvt_synthetic_pan_y,
                "dolly_z": args.uvt_synthetic_dolly_z,
                "zoom": args.uvt_synthetic_zoom,
                "principal_x": args.uvt_synthetic_principal_x,
                "principal_y": args.uvt_synthetic_principal_y,
            },
            "config_data": serialize_config_value(data_cfg),
            "note": (
                "Projection/render accuracy probe. psnr_vs_reference compares each mode to the per-frame-loop "
                "moving-camera reference under the same untrained WorldTubeModel. target_psnr is reported for "
                "orientation only and is not a trained reconstruction metric."
            ),
        },
        "rows": rows,
    }


def parse_segment_frames(value: str) -> list[int]:
    out = [int(item) for item in value.split(",") if item.strip()]
    if not out:
        raise argparse.ArgumentTypeError("segment frame list must not be empty")
    if any(item <= 0 for item in out):
        raise argparse.ArgumentTypeError("segment frame values must be positive")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-config", type=Path, default=DEFAULT_BASELINE_CONFIG)
    parser.add_argument("--target-size", type=int, default=128)
    parser.add_argument("--max-frames", type=int, default=16)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--view", type=int, default=0)
    parser.add_argument("--uvt-tubes", type=int, default=256)
    parser.add_argument("--uvt-render-backend", choices=("dense", "metal_tile"), default="metal_tile")
    parser.add_argument("--uvt-reduction-mode", default="index_add")
    parser.add_argument("--uvt-sample-emission-mode", default="direct_atomic")
    parser.add_argument("--uvt-tile-x", type=int, default=8)
    parser.add_argument("--uvt-tile-y", type=int, default=8)
    parser.add_argument("--uvt-tile-t", type=int, default=1)
    parser.add_argument("--uvt-tile-capacity", type=int, default=256)
    parser.add_argument("--uvt-init-precision-xy", type=float, default=30.0)
    parser.add_argument("--uvt-init-lambda-t", type=float, default=0.35)
    parser.add_argument("--uvt-init-opacity", type=float, default=0.35)
    parser.add_argument("--uvt-min-precision-xy", type=float, default=1.0e-5)
    parser.add_argument("--uvt-min-lambda-t", type=float, default=1.0e-5)
    parser.add_argument("--uvt-velocity-reg", type=float, default=1.0e-4)
    parser.add_argument("--uvt-depth-velocity-reg", type=float, default=0.0)
    parser.add_argument("--uvt-position-reg", type=float, default=1.0e-6)
    parser.add_argument("--uvt-init-views", choices=("first", "all_train"), default="first")
    parser.add_argument("--uvt-init-sampling", choices=("random", "grid"), default="random")
    parser.add_argument("--uvt-init-frames", choices=("first", "all", "fit"), default="first")
    parser.add_argument("--init-depth", type=float, default=2.0)
    parser.add_argument("--segment-frames", type=parse_segment_frames, default=[4, 1])
    parser.add_argument("--uvt-synthetic-pan-x", type=float, default=0.06)
    parser.add_argument("--uvt-synthetic-pan-y", type=float, default=0.0)
    parser.add_argument("--uvt-synthetic-dolly-z", type=float, default=0.0)
    parser.add_argument("--uvt-synthetic-zoom", type=float, default=0.02)
    parser.add_argument("--uvt-synthetic-principal-x", type=float, default=0.0)
    parser.add_argument("--uvt-synthetic-principal-y", type=float, default=0.0)
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path("research_project/benchmarks/results/variable_camera_attempt_compare.json"),
    )
    args = parser.parse_args()
    report = run_compare(args)
    out_path = resolve_variant_path(args.out_json)
    write_json(out_path, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"Wrote STAR-UVT variable-camera attempt compare to {out_path}")


if __name__ == "__main__":
    main()

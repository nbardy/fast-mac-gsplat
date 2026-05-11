from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch


def find_dynaworld_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "src" / "train" / "multicam_video_data.py").exists():
            return parent
    raise FileNotFoundError("Could not find dynaworld root from STAR-UVT variant")


DYNAWORLD_ROOT = find_dynaworld_root()
TRAIN_SRC = DYNAWORLD_ROOT / "src" / "train"
for path in (TRAIN_SRC,):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from camera import CameraSpec  # noqa: E402
from config_utils import load_config_file, serialize_config_value  # noqa: E402
from multicam_video_data import (  # noqa: E402
    deepview_camera_from_models,
    deepview_lens_from_model,
    deepview_model_for_camera,
    select_multicam_record,
)
from renderers.projection import project_points_camera  # noqa: E402


DEFAULT_BASELINE_CONFIG = (
    DYNAWORLD_ROOT
    / "src"
    / "train_configs"
    / "local_mac_multicam_deepview_3cam_train2_test1_vjepa_full_relpose_features_F32_256_16f_8192splats_goodset_train0006_0014_holdout0005_alphaab_alpha1_128.jsonc"
)


def resolve_dynaworld_path(path: str | Path) -> Path:
    value = Path(path)
    if value.is_absolute():
        return value
    return DYNAWORLD_ROOT / value


def resolve_variant_path(path: str | Path) -> Path:
    value = Path(path)
    if value.is_absolute():
        return value
    return Path(__file__).resolve().parents[2] / value


def parse_target_sizes(value: str) -> list[int]:
    sizes = [int(part) for part in value.split(",") if part.strip()]
    if not sizes:
        raise ValueError("--target-sizes must include at least one integer")
    return sizes


def camera_names_from_config(data_cfg: dict[str, Any], record: dict[str, Any]) -> tuple[list[str], list[str]]:
    train_raw = data_cfg.get("multicam_train_cameras")
    train = [str(camera) for camera in train_raw] if train_raw else [str(record["source_camera"])]
    heldout_raw = data_cfg.get("multicam_heldout_cameras")
    heldout = (
        [str(camera) for camera in heldout_raw]
        if heldout_raw
        else [str(data_cfg.get("multicam_heldout_camera") or record["target_camera"])]
    )
    return train, heldout


def grid_camera_points_from_pinhole_pixels(K: torch.Tensor, *, target_size: int, grid_size: int) -> torch.Tensor:
    ys, xs = torch.meshgrid(
        torch.linspace(0, target_size - 1, grid_size),
        torch.linspace(0, target_size - 1, grid_size),
        indexing="ij",
    )
    z = torch.ones_like(xs).reshape(-1) * 2.0
    x = (xs.reshape(-1) + 0.5 - K[0, 2]) * z / K[0, 0]
    y = (ys.reshape(-1) + 0.5 - K[1, 2]) * z / K[1, 1]
    return torch.stack((x, y, z), dim=-1)


def camera_spec_from_K(
    K: torch.Tensor,
    *,
    lens_model: str,
    distortion: torch.Tensor | None,
) -> CameraSpec:
    return CameraSpec(
        fx=K[0, 0],
        fy=K[1, 1],
        cx=K[0, 2],
        cy=K[1, 2],
        camera_to_world=torch.eye(4, dtype=torch.float32),
        lens_model=lens_model,  # type: ignore[arg-type]
        distortion=distortion,
    )


def audit_camera(
    record: dict[str, Any],
    camera_name: str,
    *,
    target_size: int,
    grid_size: int,
) -> dict[str, Any]:
    K, _c2w = deepview_camera_from_models(
        record,
        camera_name,
        H=target_size,
        W=target_size,
        device=torch.device("cpu"),
    )
    model = deepview_model_for_camera(record, camera_name)
    lens_model, distortion = deepview_lens_from_model(model, device=torch.device("cpu"))
    points = grid_camera_points_from_pinhole_pixels(K, target_size=target_size, grid_size=grid_size)
    distorted_camera = camera_spec_from_K(K, lens_model=lens_model, distortion=distortion)
    pinhole_camera = camera_spec_from_K(K, lens_model="pinhole", distortion=None)
    distorted_pixels, *_ = project_points_camera(points, distorted_camera)
    pinhole_pixels, *_ = project_points_camera(points, pinhole_camera)
    shift = torch.linalg.norm(distorted_pixels - pinhole_pixels, dim=-1)
    return {
        "camera": camera_name,
        "lens_model": lens_model,
        "distortion": [] if distortion is None else [float(value) for value in distortion.tolist()],
        "mean_shift_px": float(shift.mean()),
        "p50_shift_px": float(shift.quantile(0.50)),
        "p95_shift_px": float(shift.quantile(0.95)),
        "max_shift_px": float(shift.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-config", type=Path, default=DEFAULT_BASELINE_CONFIG)
    parser.add_argument("--target-sizes", default="128,256")
    parser.add_argument("--grid-size", type=int, default=17)
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path("research_project/benchmarks/results/camera_projection_parity_audit_deepview_goodset.json"),
    )
    args = parser.parse_args()

    config = load_config_file(resolve_dynaworld_path(args.baseline_config))
    data_cfg = dict(config["data"])
    if data_cfg.get("multicam_manifest") is not None:
        data_cfg["multicam_manifest"] = str(resolve_dynaworld_path(data_cfg["multicam_manifest"]))
    record = select_multicam_record(data_cfg)
    train_cameras, heldout_cameras = camera_names_from_config(data_cfg, record)
    target_sizes = parse_target_sizes(args.target_sizes)
    rows = []
    for target_size in target_sizes:
        camera_rows = [
            {
                **audit_camera(record, camera_name, target_size=target_size, grid_size=args.grid_size),
                "split": "train" if camera_name in train_cameras else "heldout",
            }
            for camera_name in [*train_cameras, *heldout_cameras]
        ]
        rows.append({"target_size": target_size, "cameras": camera_rows})

    report = {
        "baseline_config": str(resolve_dynaworld_path(args.baseline_config)),
        "train_cameras": train_cameras,
        "heldout_cameras": heldout_cameras,
        "grid_size": args.grid_size,
        "data": serialize_config_value(data_cfg),
        "rows": rows,
    }
    out_path = resolve_variant_path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"Wrote camera projection parity audit to {out_path}")


if __name__ == "__main__":
    main()

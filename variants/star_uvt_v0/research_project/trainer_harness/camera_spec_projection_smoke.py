from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from torch_gsplat_bridge_star_uvt import brute_force_render_uvt_tubes, render_uvt_tubes  # noqa: E402

try:
    from .world_tube import make_pinhole_world_tube_demo, pinhole_from_camera_spec, project_world_tubes_pinhole
except ImportError:  # pragma: no cover - script execution fallback.
    from world_tube import make_pinhole_world_tube_demo, pinhole_from_camera_spec, project_world_tubes_pinhole


def _find_dynaworld_train_path() -> Path:
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "src" / "train"
        if (candidate / "camera.py").exists():
            return candidate
    raise FileNotFoundError("Could not find dynaworld/src/train/camera.py")


def main() -> None:
    train_path = _find_dynaworld_train_path()
    if str(train_path) not in sys.path:
        sys.path.insert(0, str(train_path))
    from camera import make_default_camera

    batch, _camera, config = make_pinhole_world_tube_demo()
    camera_spec = make_default_camera(image_size=config.width, device=torch.device("cpu"), focal_scale=42.0 / config.width)
    camera = pinhole_from_camera_spec(camera_spec)
    ma, q_uvt, depth0, depth_beta, opacity, color = project_world_tubes_pinhole(batch, camera, config)
    ref = brute_force_render_uvt_tubes(ma, q_uvt, depth0, depth_beta, opacity, color, config)
    row: dict[str, object] = {
        "cpu_reference_shape": list(ref.shape),
        "projected_tube_count": int(ma.shape[0]),
        "ma0": [float(value) for value in ma[0].tolist()],
        "camera_source": "dynaworld.src.train.camera.make_default_camera",
    }
    if torch.backends.mps.is_available():
        result = render_uvt_tubes(
            ma.to("mps"),
            q_uvt.to("mps"),
            depth0.to("mps"),
            depth_beta.to("mps"),
            opacity.to("mps"),
            color.to("mps"),
            config,
            return_aux=True,
            reference=ref,
        )
        if result.stats is None:
            raise RuntimeError("expected Metal stats")
        if result.stats.max_rgb_error is None or result.stats.max_rgb_error > 2.0e-5:
            raise AssertionError(f"CameraSpec projection Metal parity failed: {result.stats.max_rgb_error}")
        row.update(result.stats.__dict__)
    print(json.dumps(row, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

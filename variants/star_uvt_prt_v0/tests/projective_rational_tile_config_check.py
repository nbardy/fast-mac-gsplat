from __future__ import annotations

from argparse import Namespace
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from torch_gsplat_bridge_star_uvt_prt.tile_config import (  # noqa: E402
    PROJECTIVE_RATIONAL_1024_TRAIN_SPEED_SUPPORT_ALPHA_THRESHOLD,
    ProjectiveRationalTileConfig,
    ProjectiveRationalTilePolicy,
    apply_projective_rational_tile_env,
    parse_projective_rational_tile_config,
    recommend_projective_rational_train_speed_tile_policy,
    recommend_projective_rational_tile_config,
    select_projective_rational_tile_summary,
)
from research_project.benchmarks.projective_rational_metal_forward_timing_probe import (  # noqa: E402
    _resolve_tile_config,
)


def main() -> None:
    parsed = parse_projective_rational_tile_config("4x4x2:256")
    assert parsed == ProjectiveRationalTileConfig(4, 4, 2, 256)
    assert parsed.key == "4x4x2:256"
    assert parsed.as_render_kwargs() == {"tile_x": 4, "tile_y": 4, "tile_t": 2, "tile_capacity": 256}

    env: dict[str, str] = {}
    apply_projective_rational_tile_env(parsed, env)
    assert env == {
        "STAR_UVT_TILE_X": "4",
        "STAR_UVT_TILE_Y": "4",
        "STAR_UVT_TILE_T": "2",
        "STAR_UVT_TILE_CAPACITY": "256",
    }

    assert recommend_projective_rational_tile_config(tube_count=128).key == "8x8x1:128"
    assert recommend_projective_rational_tile_config(tube_count=256).key == "8x8x1:256"
    assert recommend_projective_rational_tile_config(tube_count=512).key == "4x4x2:512"
    assert recommend_projective_rational_tile_config(tube_count=512, camera_motion_scale=3.0).key == "4x4x2:512"
    assert recommend_projective_rational_tile_config(tube_count=1024, allow_unverified=True).key == "4x4x2:512"

    train_policy_512 = recommend_projective_rational_train_speed_tile_policy(tube_count=512)
    assert train_policy_512 == ProjectiveRationalTilePolicy(
        name="train_speed_verified",
        tile_config=ProjectiveRationalTileConfig(4, 4, 2, 512),
    )
    train_policy_1024 = recommend_projective_rational_train_speed_tile_policy(tube_count=1024)
    assert train_policy_1024.name == "train_speed_support32_1024"
    assert train_policy_1024.tile_config.key == "4x4x1:512"
    assert train_policy_1024.as_render_kwargs() == {"tile_x": 4, "tile_y": 4, "tile_t": 1, "tile_capacity": 512}
    assert train_policy_1024.support_alpha_threshold == PROJECTIVE_RATIONAL_1024_TRAIN_SPEED_SUPPORT_ALPHA_THRESHOLD
    assert (
        _resolve_tile_config(
            Namespace(
                tile_config="auto",
                tube_counts=[512],
                camera_motion_scale=3.0,
                tile_x=8,
                tile_y=8,
                tile_t=2,
                tile_capacity=128,
            )
        ).key
        == "4x4x2:512"
    )
    assert (
        _resolve_tile_config(
            Namespace(
                tile_config="4x4x2:256",
                tube_counts=[1024],
                camera_motion_scale=3.0,
                tile_x=8,
                tile_y=8,
                tile_t=2,
                tile_capacity=128,
            )
        ).key
        == "4x4x2:256"
    )

    try:
        recommend_projective_rational_tile_config(tube_count=1024)
    except ValueError as exc:
        assert "verified up to 512" in str(exc)
    else:
        raise AssertionError("unverified tube count must fail closed")

    try:
        recommend_projective_rational_train_speed_tile_policy(tube_count=1024, camera_motion_scale=3.0)
    except ValueError as exc:
        assert "only verified up to 1024" in str(exc)
    else:
        raise AssertionError("1024 train-speed policy must fail outside verified motion scale")

    selected, best_failed = select_projective_rational_tile_summary(
        [
            {
                "pass": True,
                "selection_score": [0, 10.0, 100, 20, 50],
                "failure_score": [0, 0, 0.0, 10.0],
            },
            {
                "pass": True,
                "selection_score": [0, 8.0, 120, 20, 40],
                "failure_score": [0, 0, 0.0, 8.0],
            },
        ]
    )
    assert selected is not None
    assert selected["selection_score"][1] == 8.0
    assert best_failed is None

    selected, best_failed = select_projective_rational_tile_summary(
        [
            {
                "pass": False,
                "selection_score": [1, 10.0, 100, 20, 50],
                "failure_score": [4, 10, 0.1, 10.0],
            },
            {
                "pass": False,
                "selection_score": [1, 8.0, 120, 20, 40],
                "failure_score": [2, 1, 0.2, 8.0],
            },
        ]
    )
    assert selected is None
    assert best_failed is not None
    assert best_failed["failure_score"][0] == 2


if __name__ == "__main__":
    main()

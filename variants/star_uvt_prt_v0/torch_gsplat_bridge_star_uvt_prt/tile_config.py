from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import MutableMapping


@dataclass(frozen=True)
class ProjectiveRationalTileConfig:
    tile_x: int
    tile_y: int
    tile_t: int
    tile_capacity: int

    def __post_init__(self) -> None:
        if self.tile_x not in (4, 8, 16):
            raise ValueError("tile_x must be 4, 8, or 16")
        if self.tile_y not in (4, 8, 16):
            raise ValueError("tile_y must be 4, 8, or 16")
        if self.tile_t not in (1, 2, 4):
            raise ValueError("tile_t must be 1, 2, or 4")
        if self.tile_capacity not in (32, 64, 128, 256, 512):
            raise ValueError("tile_capacity must be 32, 64, 128, 256, or 512")

    @property
    def key(self) -> str:
        return f"{self.tile_x}x{self.tile_y}x{self.tile_t}:{self.tile_capacity}"

    def as_dict(self) -> dict[str, int]:
        return asdict(self)

    def as_env(self) -> dict[str, str]:
        return {
            "STAR_UVT_TILE_X": str(self.tile_x),
            "STAR_UVT_TILE_Y": str(self.tile_y),
            "STAR_UVT_TILE_T": str(self.tile_t),
            "STAR_UVT_TILE_CAPACITY": str(self.tile_capacity),
        }

    def as_render_kwargs(self) -> dict[str, int]:
        return self.as_dict()


@dataclass(frozen=True)
class ProjectiveRationalTilePolicy:
    name: str
    tile_config: ProjectiveRationalTileConfig
    support_alpha_threshold: float | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("name must be non-empty")
        if self.support_alpha_threshold is not None and self.support_alpha_threshold <= 0.0:
            raise ValueError("support_alpha_threshold must be positive")

    def as_render_kwargs(self) -> dict[str, int]:
        return self.tile_config.as_render_kwargs()


PROJECTIVE_RATIONAL_1024_TRAIN_SPEED_SUPPORT_ALPHA_THRESHOLD = 32.0 / 255.0

DEFAULT_PROJECTIVE_RATIONAL_TILE_CANDIDATES: tuple[ProjectiveRationalTileConfig, ...] = (
    ProjectiveRationalTileConfig(8, 8, 1, 128),
    ProjectiveRationalTileConfig(8, 8, 1, 256),
    ProjectiveRationalTileConfig(8, 8, 2, 128),
    ProjectiveRationalTileConfig(8, 8, 2, 256),
    ProjectiveRationalTileConfig(8, 8, 2, 512),
    ProjectiveRationalTileConfig(4, 4, 2, 128),
    ProjectiveRationalTileConfig(4, 4, 2, 256),
    ProjectiveRationalTileConfig(4, 4, 2, 512),
)


def parse_projective_rational_tile_config(value: str) -> ProjectiveRationalTileConfig:
    try:
        shape, capacity_raw = value.split(":", 1)
        tile_x_raw, tile_y_raw, tile_t_raw = shape.lower().split("x", 2)
        tile_x = int(tile_x_raw)
        tile_y = int(tile_y_raw)
        tile_t = int(tile_t_raw)
        tile_capacity = int(capacity_raw)
    except ValueError as exc:
        raise ValueError("tile config must look like 8x8x2:128") from exc
    return ProjectiveRationalTileConfig(
        tile_x=tile_x,
        tile_y=tile_y,
        tile_t=tile_t,
        tile_capacity=tile_capacity,
    )


def apply_projective_rational_tile_env(
    config: ProjectiveRationalTileConfig,
    environ: MutableMapping[str, str] | None = None,
) -> None:
    target = os.environ if environ is None else environ
    target.update(config.as_env())


def recommend_projective_rational_tile_config(
    *,
    tube_count: int,
    camera_motion_scale: float = 1.0,
    allow_unverified: bool = False,
) -> ProjectiveRationalTileConfig:
    if tube_count <= 0:
        raise ValueError("tube_count must be positive")
    if camera_motion_scale <= 0.0:
        raise ValueError("camera_motion_scale must be positive")

    if tube_count <= 128 and camera_motion_scale <= 1.5:
        return ProjectiveRationalTileConfig(8, 8, 1, 128)
    if tube_count <= 256 and camera_motion_scale <= 1.5:
        return ProjectiveRationalTileConfig(8, 8, 1, 256)
    if tube_count <= 512:
        return ProjectiveRationalTileConfig(4, 4, 2, 512)
    if allow_unverified:
        return ProjectiveRationalTileConfig(4, 4, 2, 512)
    raise ValueError("PRT tile config is only verified up to 512 tubes")


def recommend_projective_rational_train_speed_tile_policy(
    *,
    tube_count: int,
    camera_motion_scale: float = 1.0,
) -> ProjectiveRationalTilePolicy:
    if tube_count <= 512:
        return ProjectiveRationalTilePolicy(
            name="train_speed_verified",
            tile_config=recommend_projective_rational_tile_config(
                tube_count=tube_count,
                camera_motion_scale=camera_motion_scale,
            ),
        )
    if tube_count <= 1024 and camera_motion_scale <= 1.5:
        return ProjectiveRationalTilePolicy(
            name="train_speed_support32_1024",
            tile_config=ProjectiveRationalTileConfig(4, 4, 1, 512),
            support_alpha_threshold=PROJECTIVE_RATIONAL_1024_TRAIN_SPEED_SUPPORT_ALPHA_THRESHOLD,
        )
    raise ValueError("PRT train-speed tile policy is only verified up to 1024 tubes at camera_motion_scale <= 1.5")


def select_projective_rational_tile_summary(
    summaries: list[dict],
) -> tuple[dict | None, dict | None]:
    if not summaries:
        raise ValueError("expected at least one tile config summary")
    passing = [item for item in summaries if bool(item.get("pass"))]
    selected = min(passing, key=lambda item: tuple(item["selection_score"])) if passing else None
    best_failed = min(summaries, key=lambda item: tuple(item["failure_score"])) if selected is None else None
    return selected, best_failed

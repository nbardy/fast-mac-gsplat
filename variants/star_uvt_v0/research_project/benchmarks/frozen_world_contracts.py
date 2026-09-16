from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any


STATIC_CAMERA_PROGRAM_MODE = "static_view"
BOUNDED_YAW_CAMERA_PROGRAM_MODE = "bounded_yaw_projective_first_order_v1"
BOUNDED_YAW_FRAME_COUNTS = (8, 16, 32, 64)
BOUNDED_YAW_TOTAL_DEGREES = 45.0
BOUNDED_YAW_IMAGE_SIZE = (256, 256)


def canonical_json_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            dict(value),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def bounded_yaw_camera_program_contract(
    *,
    frame_counts: Sequence[int] = BOUNDED_YAW_FRAME_COUNTS,
    image_size: Sequence[int] = BOUNDED_YAW_IMAGE_SIZE,
    yaw_total_degrees: float = BOUNDED_YAW_TOTAL_DEGREES,
) -> dict[str, Any]:
    counts = tuple(int(value) for value in frame_counts)
    size = tuple(int(value) for value in image_size)
    yaw_total = float(yaw_total_degrees)
    if counts != BOUNDED_YAW_FRAME_COUNTS:
        raise ValueError(
            "bounded-yaw paper sweep requires frame counts 8,16,32,64"
        )
    if size != BOUNDED_YAW_IMAGE_SIZE:
        raise ValueError("bounded-yaw paper sweep requires image size 256x256")
    if yaw_total != BOUNDED_YAW_TOTAL_DEGREES:
        raise ValueError("bounded-yaw paper sweep requires exactly 45 degrees")
    half_yaw = 0.5 * yaw_total
    return {
        "schema_version": 1,
        "mode": BOUNDED_YAW_CAMERA_PROGRAM_MODE,
        "path_scope": "bounded_open_path",
        "yaw_total_degrees": yaw_total,
        "yaw_start_degrees": -half_yaw,
        "yaw_end_degrees": half_yaw,
        "sampling": "uniform_closed_interval",
        "frame_counts": list(counts),
        "image_size": list(size),
        "compiler_chart_policy": "single_midpoint_first_order",
        "multi_chart_gauge_compiler": False,
    }


def validate_frozen_world_camera_program_request(
    *,
    mode: str,
    frame_counts: Sequence[int],
    image_size: Sequence[int],
    yaw_total_degrees: float,
    checkpoint_supplied: bool,
) -> dict[str, Any] | None:
    if mode == STATIC_CAMERA_PROGRAM_MODE:
        if checkpoint_supplied:
            raise ValueError(
                "checkpoint-only frozen-world execution is reserved for the "
                "bounded-yaw camera program"
            )
        return None
    if mode != BOUNDED_YAW_CAMERA_PROGRAM_MODE:
        raise ValueError(
            "frozen-world camera program mode must be one of: "
            f"{STATIC_CAMERA_PROGRAM_MODE}, {BOUNDED_YAW_CAMERA_PROGRAM_MODE}"
        )
    if not checkpoint_supplied:
        raise ValueError(
            "bounded-yaw frozen-world execution requires the accepted static "
            "checkpoint via --checkpoint"
        )
    return bounded_yaw_camera_program_contract(
        frame_counts=frame_counts,
        image_size=image_size,
        yaw_total_degrees=yaw_total_degrees,
    )


def validate_expected_sha256(value: str | None, *, name: str) -> str:
    if value is None:
        raise ValueError(f"{name} is required")
    normalized = value.lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError(f"{name} must be a 64-character lowercase SHA-256")
    return normalized


__all__ = [
    "BOUNDED_YAW_CAMERA_PROGRAM_MODE",
    "BOUNDED_YAW_FRAME_COUNTS",
    "BOUNDED_YAW_IMAGE_SIZE",
    "BOUNDED_YAW_TOTAL_DEGREES",
    "STATIC_CAMERA_PROGRAM_MODE",
    "bounded_yaw_camera_program_contract",
    "canonical_json_sha256",
    "validate_expected_sha256",
    "validate_frozen_world_camera_program_request",
]

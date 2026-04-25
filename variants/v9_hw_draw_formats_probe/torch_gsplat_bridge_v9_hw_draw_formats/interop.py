from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Any, Dict, Sequence, Tuple

import torch
from torch import Tensor

try:
    from . import _C
except Exception:  # pragma: no cover - import depends on local extension build.
    _C = None


RGBA = Tuple[float, float, float, float]
DIRECT_OUTPUT_FORMATS = ("rgba32f", "rgba16f", "r32f", "rg32f")


@dataclass(frozen=True)
class V9HWDrawFormatsCapabilities:
    mps_available: bool
    native_extension_loaded: bool
    native_probe_available: bool
    native_probe_error: str
    metal_available: bool
    metal_device_name: str
    has_unified_memory: bool | None
    recommended_max_working_set_size: int | None
    supports_family_apple4: bool | None
    supports_family_mac2: bool | None
    raster_order_groups_supported: bool | None
    render_pipeline_ready: bool | None
    render_pipeline_error: str
    tile_pipeline_ready: bool | None
    tile_pipeline_error: str
    tile_imageblock_sample_length: int | None
    tile_imageblock_memory_16x16: int | None
    icb_created: bool | None
    icb_error: str
    icb_execute_op_available: bool
    icb_execute_validated: bool
    icb_execute_max_abs_err: float | None
    icb_execute_error: str
    torch_mps_command_buffer_api: bool
    torch_mps_dispatch_queue_api: bool
    render_to_mps_tensor_op_available: bool
    render_to_mps_tensor_validated: bool
    render_to_mps_tensor_max_abs_err: float | None
    direct_render_to_mps_tensor_op_available: bool
    direct_render_to_mps_tensor_validated: bool
    direct_render_to_mps_tensor_max_abs_err: float | None
    direct_render_to_mps_tensor_error: str
    direct_output_format_probe_available: bool
    direct_output_format_results: list[dict[str, Any]]
    direct_output_format_validation: list[dict[str, Any]]
    validation_uses_cpu_readback: bool
    native_op_uses_cpu_readback: bool
    native_details: Dict[str, Any]

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@lru_cache(maxsize=4)
def _native_probe(compile_pipelines: bool, compile_advanced: bool) -> Dict[str, Any]:
    if _C is None:
        return {
            "native_probe_available": False,
            "native_probe_error": "native extension is not imported; build variants/v9_hw_draw_formats_probe first",
        }
    probe = getattr(_C, "probe_native", None)
    if probe is None:
        return {
            "native_probe_available": False,
            "native_probe_error": "native extension does not expose probe_native",
        }
    try:
        return dict(probe(bool(compile_pipelines), bool(compile_advanced)))
    except Exception as exc:  # pragma: no cover - depends on local Metal runtime.
        return {
            "native_probe_available": False,
            "native_probe_error": f"{type(exc).__name__}: {exc}",
        }


def _optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    return bool(value)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def render_constant_rgba(height: int, width: int, rgba: RGBA = (0.125, 0.5, 0.875, 1.0)) -> Tensor:
    """Render RGBA32F through a private Metal texture, then GPU-blit into a Torch MPS tensor."""
    if _C is None:
        raise RuntimeError("native extension is not imported; build variants/v9_hw_draw_formats_probe first")
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is not available")
    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive")
    return _C.render_constant_rgba(int(height), int(width), tuple(float(x) for x in rgba))


def render_constant_rgba_direct(height: int, width: int, rgba: RGBA = (0.125, 0.5, 0.875, 1.0)) -> Tensor:
    """Render RGBA32F directly into a buffer-backed texture over Torch MPS tensor storage."""
    if _C is None:
        raise RuntimeError("native extension is not imported; build variants/v9_hw_draw_formats_probe first")
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is not available")
    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive")
    return _C.render_constant_rgba_direct(int(height), int(width), tuple(float(x) for x in rgba))


def render_constant_rgba_direct_icb(height: int, width: int, rgba: RGBA = (0.125, 0.5, 0.875, 1.0)) -> Tensor:
    """Disabled ICB execute probe.

    A minimal ICB render execute path crashed inside Apple's AGX
    executeCommandsInBufferCommon on Apple M4. Keep this API fail-closed so
    tests and benchmarks cannot accidentally re-enter the driver crash path.
    """
    raise RuntimeError(
        "render_constant_rgba_direct_icb is disabled: minimal ICB execution crashed in "
        "AGX executeCommandsInBufferCommon on macOS/Apple M4. Treat ICB execution as unsafe "
        "until reworked in a separate isolated harness."
    )


def render_constant_direct_format(
    output_format: str,
    height: int,
    width: int,
    rgba: RGBA = (0.125, 0.5, 0.875, 1.0),
) -> Tensor:
    """Render directly into a Torch MPS tensor using rgba32f, rgba16f, r32f, or rg32f."""
    if _C is None:
        raise RuntimeError("native extension is not imported; build variants/v9_hw_draw_formats_probe first")
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is not available")
    if output_format not in DIRECT_OUTPUT_FORMATS:
        raise ValueError(f"unknown output_format {output_format!r}; expected one of {DIRECT_OUTPUT_FORMATS}")
    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive")
    return _C.render_constant_format_direct(
        str(output_format),
        int(height),
        int(width),
        tuple(float(x) for x in rgba),
    )


def direct_width_multiple(output_format: str) -> int:
    """Return the width multiple needed for a contiguous direct buffer-backed render target."""
    bytes_per_pixel = {
        "rgba32f": 16,
        "rgba16f": 8,
        "r32f": 4,
        "rg32f": 8,
    }[output_format]
    # Metal buffer-backed textures require 256-byte row alignment.
    return 256 // math_gcd(256, bytes_per_pixel)


def aligned_direct_width(output_format: str, width: int) -> int:
    multiple = direct_width_multiple(output_format)
    return ((int(width) + multiple - 1) // multiple) * multiple


def math_gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return a


def probe_direct_output_formats(
    *,
    height: int = 1,
    widths: Sequence[int] = (16, 32, 64, 1920, 3840),
) -> list[dict[str, Any]]:
    """Probe buffer-backed render target creation and row alignment for direct output formats."""
    if _C is None:
        raise RuntimeError("native extension is not imported; build variants/v9_hw_draw_formats_probe first")
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is not available")
    probe = getattr(_C, "probe_direct_output_formats", None)
    if probe is None:
        raise RuntimeError("native extension does not expose probe_direct_output_formats")
    return [dict(row) for row in probe(int(height), [int(w) for w in widths])]


def _expected_for_format(output_format: str, height: int, width: int, rgba: RGBA) -> Tensor:
    if output_format == "r32f":
        return torch.full((height, width), float(rgba[0]), dtype=torch.float32)
    if output_format == "rg32f":
        return torch.tensor(rgba[:2], dtype=torch.float32).view(1, 1, 2).expand(height, width, 2)
    if output_format == "rgba16f":
        return torch.tensor(rgba, dtype=torch.float16).view(1, 1, 4).expand(height, width, 4).float()
    return torch.tensor(rgba, dtype=torch.float32).view(1, 1, 4).expand(height, width, 4)


def validate_direct_output_formats(
    *,
    height: int = 16,
    rgba: RGBA = (0.125, 0.5, 0.875, 1.0),
) -> list[dict[str, Any]]:
    """Render one aligned image per direct output format and validate via CPU readback."""
    rows: list[dict[str, Any]] = []
    for output_format in DIRECT_OUTPUT_FORMATS:
        width = direct_width_multiple(output_format)
        row: dict[str, Any] = {
            "format": output_format,
            "height": int(height),
            "width": width,
            "validated": False,
            "max_abs_err": None,
            "error": "",
        }
        try:
            out = render_constant_direct_format(output_format, int(height), width, rgba)
            expected = _expected_for_format(output_format, int(height), width, rgba)
            got = out.detach().cpu().float()
            max_err = float((got - expected).abs().max().item())
            row["shape"] = tuple(int(x) for x in out.shape)
            row["dtype"] = str(out.dtype)
            row["max_abs_err"] = max_err
            row["validated"] = max_err == 0.0
        except Exception as exc:  # pragma: no cover - depends on local Metal runtime.
            row["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
    return rows


def probe_hw_interop(
    *,
    compile_pipelines: bool = True,
    compile_advanced: bool = True,
    run_render_probe: bool = False,
    run_icb_probe: bool = False,
    height: int = 8,
    width: int = 8,
    rgba: RGBA = (0.125, 0.5, 0.875, 1.0),
) -> V9HWDrawFormatsCapabilities:
    native = _native_probe(bool(compile_pipelines), bool(compile_advanced))
    render_validated = False
    render_err: float | None = None
    direct_validated = False
    direct_err: float | None = None
    direct_error = ""
    icb_validated = False
    icb_err: float | None = None
    icb_error = ""
    direct_format_results: list[dict[str, Any]] = []
    direct_format_validation: list[dict[str, Any]] = []
    validation_uses_cpu_readback = False

    disabled_icb_execute_message = (
        "disabled: minimal ICB execution crashed in AGX executeCommandsInBufferCommon "
        "on macOS/Apple M4; allocation-only probing remains safe"
    )
    op_available = _C is not None and hasattr(_C, "render_constant_rgba")
    direct_op_available = _C is not None and hasattr(_C, "render_constant_rgba_direct")
    icb_execute_op_available = False
    icb_error = disabled_icb_execute_message
    direct_output_format_probe_available = _C is not None and hasattr(_C, "probe_direct_output_formats")
    if run_render_probe and op_available and torch.backends.mps.is_available():
        out = render_constant_rgba(height, width, rgba)
        expected = torch.tensor(rgba, dtype=torch.float32).view(1, 1, 4).expand(height, width, 4)
        got = out.detach().cpu()
        validation_uses_cpu_readback = True
        render_err = float((got - expected).abs().max().item())
        render_validated = render_err <= 0.0
    if run_render_probe and direct_op_available and torch.backends.mps.is_available():
        direct_height = max(16, int(height))
        direct_width = max(16, aligned_direct_width("rgba32f", int(width)))
        try:
            out = render_constant_rgba_direct(direct_height, direct_width, rgba)
            expected = torch.tensor(rgba, dtype=torch.float32).view(1, 1, 4).expand(direct_height, direct_width, 4)
            got = out.detach().cpu()
            validation_uses_cpu_readback = True
            direct_err = float((got - expected).abs().max().item())
            direct_validated = direct_err <= 0.0
        except Exception as exc:  # pragma: no cover - depends on local Metal runtime.
            direct_error = f"{type(exc).__name__}: {exc}"
    if run_render_probe and direct_output_format_probe_available and torch.backends.mps.is_available():
        direct_format_results = probe_direct_output_formats(
            height=1,
            widths=(16, 32, 64, 1920, 3840),
        )
        direct_format_validation = validate_direct_output_formats(height=max(16, int(height)), rgba=rgba)
        validation_uses_cpu_readback = True
    if run_render_probe and run_icb_probe:
        icb_error = disabled_icb_execute_message

    return V9HWDrawFormatsCapabilities(
        mps_available=bool(torch.backends.mps.is_available()),
        native_extension_loaded=_C is not None,
        native_probe_available=bool(native.get("native_probe_available", False)),
        native_probe_error=str(native.get("native_probe_error", "")),
        metal_available=bool(native.get("metal_available", False)),
        metal_device_name=str(native.get("metal_device_name", "")),
        has_unified_memory=_optional_bool(native.get("has_unified_memory")),
        recommended_max_working_set_size=_optional_int(native.get("recommended_max_working_set_size")),
        supports_family_apple4=_optional_bool(native.get("supports_family_apple4")),
        supports_family_mac2=_optional_bool(native.get("supports_family_mac2")),
        raster_order_groups_supported=_optional_bool(native.get("raster_order_groups_supported")),
        render_pipeline_ready=_optional_bool(native.get("render_pipeline_ready")),
        render_pipeline_error=str(native.get("render_pipeline_error", "")),
        tile_pipeline_ready=_optional_bool(native.get("tile_pipeline_ready")),
        tile_pipeline_error=str(native.get("tile_pipeline_error", "")),
        tile_imageblock_sample_length=_optional_int(native.get("tile_imageblock_sample_length")),
        tile_imageblock_memory_16x16=_optional_int(native.get("tile_imageblock_memory_16x16")),
        icb_created=_optional_bool(native.get("icb_created")),
        icb_error=str(native.get("icb_error", "")),
        icb_execute_op_available=icb_execute_op_available,
        icb_execute_validated=icb_validated,
        icb_execute_max_abs_err=icb_err,
        icb_execute_error=icb_error,
        torch_mps_command_buffer_api=bool(native.get("torch_mps_command_buffer_api", False)),
        torch_mps_dispatch_queue_api=bool(native.get("torch_mps_dispatch_queue_api", False)),
        render_to_mps_tensor_op_available=op_available,
        render_to_mps_tensor_validated=render_validated,
        render_to_mps_tensor_max_abs_err=render_err,
        direct_render_to_mps_tensor_op_available=direct_op_available,
        direct_render_to_mps_tensor_validated=direct_validated,
        direct_render_to_mps_tensor_max_abs_err=direct_err,
        direct_render_to_mps_tensor_error=direct_error,
        direct_output_format_probe_available=direct_output_format_probe_available,
        direct_output_format_results=direct_format_results,
        direct_output_format_validation=direct_format_validation,
        validation_uses_cpu_readback=validation_uses_cpu_readback,
        native_op_uses_cpu_readback=bool(native.get("native_op_uses_cpu_readback", True)),
        native_details=native,
    )

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Any, Dict, Tuple

import torch
from torch import Tensor

try:
    from . import _C
except Exception:  # pragma: no cover - import depends on local extension build.
    _C = None


RGBA = Tuple[float, float, float, float]


@dataclass(frozen=True)
class V9HWTileExactCapabilities:
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
    tile_imageblock_memory_8x8: int | None
    tile_imageblock_memory_16x16: int | None
    tile_imageblock_memory_32x32: int | None
    tile_layouts: Tuple[Dict[str, Any], ...]
    tile_execution_probe_available: bool | None
    tile_execution_probe_error: str
    tile_exact_overlap_probe_available: bool | None
    tile_exact_overlap_probe_error: str
    tile_exact_imageblock_sample_length: int | None
    tile_exact_imageblock_memory_16x16: int | None
    tile_exact_imageblock_memory_32x32: int | None
    icb_created: bool | None
    icb_error: str
    torch_mps_command_buffer_api: bool
    torch_mps_dispatch_queue_api: bool
    render_to_mps_tensor_op_available: bool
    render_to_mps_tensor_validated: bool
    render_to_mps_tensor_max_abs_err: float | None
    direct_render_to_mps_tensor_op_available: bool
    direct_render_to_mps_tensor_validated: bool
    direct_render_to_mps_tensor_max_abs_err: float | None
    direct_render_to_mps_tensor_error: str
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
            "native_probe_error": "native extension is not imported; build variants/v9_hw_tile_exact_probe first",
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


def _layout_tuple(value: Any) -> Tuple[Dict[str, Any], ...]:
    if not value:
        return ()
    return tuple(dict(item) for item in value)


def _ceil_div(value: int, divisor: int) -> int:
    return (int(value) + int(divisor) - 1) // int(divisor)


def render_constant_rgba(height: int, width: int, rgba: RGBA = (0.125, 0.5, 0.875, 1.0)) -> Tensor:
    """Render RGBA32F through a private Metal texture, then GPU-blit into a Torch MPS tensor."""
    if _C is None:
        raise RuntimeError("native extension is not imported; build variants/v9_hw_tile_exact_probe first")
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is not available")
    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive")
    return _C.render_constant_rgba(int(height), int(width), tuple(float(x) for x in rgba))


def render_constant_rgba_direct(height: int, width: int, rgba: RGBA = (0.125, 0.5, 0.875, 1.0)) -> Tensor:
    """Render RGBA32F directly into a buffer-backed texture over Torch MPS tensor storage."""
    if _C is None:
        raise RuntimeError("native extension is not imported; build variants/v9_hw_tile_exact_probe first")
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is not available")
    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive")
    return _C.render_constant_rgba_direct(int(height), int(width), tuple(float(x) for x in rgba))


def run_tile_exact_execution_probe(height: int = 32, width: int = 32, tile_size: int = 32) -> Dict[str, Any]:
    """Dispatch the tile/imageblock probe in a render pass over a direct MPS target."""
    if _C is None:
        raise RuntimeError("native extension is not imported; build variants/v9_hw_tile_exact_probe first")
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is not available")
    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive")
    if tile_size not in (8, 16, 32):
        raise ValueError("tile_size must be 8, 16, or 32 for this probe")
    return dict(_C.run_tile_exact_execution_probe(int(height), int(width), int(tile_size)))


def run_tile_exact_overlap_probe(height: int = 32, width: int = 32, tile_size: int = 16) -> Dict[str, Any]:
    """Run the minimal exact-forward imageblock probe.

    The native path clears explicit imageblock C/T state, draws two ordered
    full-screen constant-alpha splats with blending disabled, then tile-resolves
    `float4(C.rgb, T)` into a direct Torch/MPS render target. It also emits a
    V8-shaped `tile_stop_counts` tensor for the toy overlap case.
    """
    if _C is None:
        raise RuntimeError("native extension is not imported; build variants/v9_hw_tile_exact_probe first")
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is not available")
    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive")
    if tile_size != 16:
        raise ValueError(
            "tile_size must be 16 for this exact-overlap probe; 32x32 compiled but failed encoder creation on M4"
        )
    return dict(_C.run_tile_exact_overlap_probe(int(height), int(width), int(tile_size)))


def estimate_tile_exact_memory(
    *,
    height: int = 2160,
    width: int = 3840,
    tile_size: int = 16,
    layout_name: str = "ct_stop_flags_fp32_u32x2",
    final_state_bytes_per_pixel: int = 8,
    compile_advanced: bool = True,
) -> Dict[str, Any]:
    """Estimate imageblock C/T/stop cost and optional final_T/stop_count capture cost."""
    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive")
    if tile_size <= 0:
        raise ValueError("tile_size must be positive")

    native = _native_probe(True, compile_advanced)
    layouts = _layout_tuple(native.get("tile_layouts"))
    selected = next((layout for layout in layouts if layout.get("name") == layout_name), None)
    measured_tile_key = f"imageblock_memory_{tile_size}x{tile_size}"
    measured_bytes_per_tile = None
    if selected is not None and selected.get("pipeline_ready") and measured_tile_key in selected:
        raw = selected.get(measured_tile_key)
        measured_bytes_per_tile = None if raw is None else int(raw)

    logical_bytes_per_pixel = int(selected.get("logical_bytes_per_pixel", 24)) if selected else 24
    fallback_bytes_per_tile = logical_bytes_per_pixel * tile_size * tile_size
    imageblock_bytes_per_tile = measured_bytes_per_tile or fallback_bytes_per_tile
    tiles_x = _ceil_div(width, tile_size)
    tiles_y = _ceil_div(height, tile_size)
    tile_count = tiles_x * tiles_y
    pixels = height * width
    final_state_bytes = pixels * int(final_state_bytes_per_pixel)

    return {
        "height": int(height),
        "width": int(width),
        "pixels": pixels,
        "tile_size": int(tile_size),
        "tiles_x": tiles_x,
        "tiles_y": tiles_y,
        "tile_count": tile_count,
        "layout_name": layout_name,
        "logical_bytes_per_pixel": logical_bytes_per_pixel,
        "imageblock_bytes_per_tile": imageblock_bytes_per_tile,
        "imageblock_bytes_source": "metal_measured" if measured_bytes_per_tile is not None else "logical_fallback",
        "imageblock_full_frame_equivalent_bytes": imageblock_bytes_per_tile * tile_count,
        "final_state_bytes_per_pixel": int(final_state_bytes_per_pixel),
        "final_T_stop_count_frame_bytes": final_state_bytes,
        "output_rgba32f_frame_bytes": pixels * 16,
        "selected_layout": selected or {},
        "note": "imageblock memory is transient tile-local storage; full-frame equivalent is for pressure comparison only",
    }


def probe_hw_interop(
    *,
    compile_pipelines: bool = True,
    compile_advanced: bool = True,
    run_render_probe: bool = False,
    height: int = 8,
    width: int = 8,
    rgba: RGBA = (0.125, 0.5, 0.875, 1.0),
) -> V9HWTileExactCapabilities:
    native = _native_probe(bool(compile_pipelines), bool(compile_advanced))
    render_validated = False
    render_err: float | None = None
    direct_validated = False
    direct_err: float | None = None
    direct_error = ""
    validation_uses_cpu_readback = False

    op_available = _C is not None and hasattr(_C, "render_constant_rgba")
    direct_op_available = _C is not None and hasattr(_C, "render_constant_rgba_direct")
    if run_render_probe and op_available and torch.backends.mps.is_available():
        out = render_constant_rgba(height, width, rgba)
        expected = torch.tensor(rgba, dtype=torch.float32).view(1, 1, 4).expand(height, width, 4)
        got = out.detach().cpu()
        validation_uses_cpu_readback = True
        render_err = float((got - expected).abs().max().item())
        render_validated = render_err <= 0.0
    if run_render_probe and direct_op_available and torch.backends.mps.is_available():
        direct_height = max(16, int(height))
        direct_width = max(16, int(width))
        try:
            out = render_constant_rgba_direct(direct_height, direct_width, rgba)
            expected = torch.tensor(rgba, dtype=torch.float32).view(1, 1, 4).expand(direct_height, direct_width, 4)
            got = out.detach().cpu()
            validation_uses_cpu_readback = True
            direct_err = float((got - expected).abs().max().item())
            direct_validated = direct_err <= 0.0
        except Exception as exc:  # pragma: no cover - depends on local Metal runtime.
            direct_error = f"{type(exc).__name__}: {exc}"

    return V9HWTileExactCapabilities(
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
        tile_imageblock_memory_8x8=_optional_int(native.get("tile_imageblock_memory_8x8")),
        tile_imageblock_memory_16x16=_optional_int(native.get("tile_imageblock_memory_16x16")),
        tile_imageblock_memory_32x32=_optional_int(native.get("tile_imageblock_memory_32x32")),
        tile_layouts=_layout_tuple(native.get("tile_layouts")),
        tile_execution_probe_available=_optional_bool(native.get("tile_execution_probe_available")),
        tile_execution_probe_error=str(native.get("tile_execution_probe_error", "")),
        tile_exact_overlap_probe_available=_optional_bool(native.get("tile_exact_overlap_probe_available")),
        tile_exact_overlap_probe_error=str(native.get("tile_exact_overlap_probe_error", "")),
        tile_exact_imageblock_sample_length=_optional_int(native.get("tile_exact_imageblock_sample_length")),
        tile_exact_imageblock_memory_16x16=_optional_int(native.get("tile_exact_imageblock_memory_16x16")),
        tile_exact_imageblock_memory_32x32=_optional_int(native.get("tile_exact_imageblock_memory_32x32")),
        icb_created=_optional_bool(native.get("icb_created")),
        icb_error=str(native.get("icb_error", "")),
        torch_mps_command_buffer_api=bool(native.get("torch_mps_command_buffer_api", False)),
        torch_mps_dispatch_queue_api=bool(native.get("torch_mps_dispatch_queue_api", False)),
        render_to_mps_tensor_op_available=op_available,
        render_to_mps_tensor_validated=render_validated,
        render_to_mps_tensor_max_abs_err=render_err,
        direct_render_to_mps_tensor_op_available=direct_op_available,
        direct_render_to_mps_tensor_validated=direct_validated,
        direct_render_to_mps_tensor_max_abs_err=direct_err,
        direct_render_to_mps_tensor_error=direct_error,
        validation_uses_cpu_readback=validation_uses_cpu_readback,
        native_op_uses_cpu_readback=bool(native.get("native_op_uses_cpu_readback", True)),
        native_details=native,
    )

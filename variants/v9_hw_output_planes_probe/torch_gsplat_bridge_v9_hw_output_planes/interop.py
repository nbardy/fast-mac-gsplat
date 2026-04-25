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
GAUSSIAN_OUTPUT_FORMATS = ("rgba32f", "rgba16f")


@dataclass(frozen=True)
class V9HWOutputPlanesCapabilities:
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
    gaussian_eval_pipeline_ready: bool | None
    gaussian_eval_pipeline_error: str
    tile_pipeline_ready: bool | None
    tile_pipeline_error: str
    tile_imageblock_sample_length: int | None
    tile_imageblock_memory_16x16: int | None
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
    gaussian_eval_rgba_op_available: bool
    gaussian_eval_rgba_validated: bool
    gaussian_eval_rgba_max_abs_err: float | None
    gaussian_eval_rgba_error: str
    gaussian_eval_rgba16_validated: bool
    gaussian_eval_rgba16_max_abs_err: float | None
    gaussian_eval_rgba16_error: str
    direct_output_format_probe_available: bool
    direct_output_format_results: list[dict[str, Any]]
    direct_output_format_validation: list[dict[str, Any]]
    gaussian_output_format_validation: list[dict[str, Any]]
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
            "native_probe_error": "native extension is not imported; build variants/v9_hw_output_planes_probe first",
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
        raise RuntimeError("native extension is not imported; build variants/v9_hw_output_planes_probe first")
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is not available")
    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive")
    return _C.render_constant_rgba(int(height), int(width), tuple(float(x) for x in rgba))


def render_constant_rgba_direct(height: int, width: int, rgba: RGBA = (0.125, 0.5, 0.875, 1.0)) -> Tensor:
    """Render RGBA32F directly into a buffer-backed texture over Torch MPS tensor storage."""
    if _C is None:
        raise RuntimeError("native extension is not imported; build variants/v9_hw_output_planes_probe first")
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is not available")
    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive")
    return _C.render_constant_rgba_direct(int(height), int(width), tuple(float(x) for x in rgba))


def render_constant_direct_format(
    output_format: str,
    height: int,
    width: int,
    rgba: RGBA = (0.125, 0.5, 0.875, 1.0),
) -> Tensor:
    """Render directly into a Torch MPS tensor using rgba32f, rgba16f, r32f, or rg32f."""
    if _C is None:
        raise RuntimeError("native extension is not imported; build variants/v9_hw_output_planes_probe first")
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


def _math_gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return a


def direct_width_multiple(output_format: str) -> int:
    """Return the width multiple needed for a contiguous direct buffer-backed render target."""
    bytes_per_pixel = {
        "rgba32f": 16,
        "rgba16f": 8,
        "r32f": 4,
        "rg32f": 8,
    }[output_format]
    return 256 // _math_gcd(256, bytes_per_pixel)


def aligned_direct_width(output_format: str, width: int) -> int:
    multiple = direct_width_multiple(output_format)
    return ((int(width) + multiple - 1) // multiple) * multiple


def probe_direct_output_formats(
    *,
    height: int = 1,
    widths: Sequence[int] = (16, 32, 64, 1920, 3840),
) -> list[dict[str, Any]]:
    """Probe buffer-backed render target creation and row alignment for direct output formats."""
    if _C is None:
        raise RuntimeError("native extension is not imported; build variants/v9_hw_output_planes_probe first")
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is not available")
    probe = getattr(_C, "probe_direct_output_formats", None)
    if probe is None:
        raise RuntimeError("native extension does not expose probe_direct_output_formats")
    return [dict(row) for row in probe(int(height), [int(w) for w in widths])]


def _expected_constant_for_format(output_format: str, height: int, width: int, rgba: RGBA) -> Tensor:
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
            expected = _expected_constant_for_format(output_format, int(height), width, rgba)
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


def _check_gaussian_tensor(name: str, tensor: Tensor, trailing: int) -> None:
    if tensor.device.type != "mps":
        raise RuntimeError(f"{name} must be an MPS tensor")
    if tensor.dtype != torch.float32:
        raise TypeError(f"{name} must be float32")
    if trailing == 1:
        if tensor.dim() != 1:
            raise ValueError(f"{name} must be [G]")
    elif tensor.dim() != 2 or tensor.shape[1] != trailing:
        raise ValueError(f"{name} must be [G,{trailing}]")


def render_gaussian_eval_rgba(
    means2d: Tensor,
    conics: Tensor,
    colors: Tensor,
    opacities: Tensor,
    height: int,
    width: int,
    *,
    direct: bool = True,
) -> Tensor:
    """Render screen-space Gaussian splats into an RGBA32F MPS tensor.

    This fixed-eval probe expects pixel-space `means2d` and conics `[a,b,c]`.
    The output stores premultiplied RGB plus alpha. Multiple Gaussians use
    hardware source-over blending in input order; there is no depth sort.
    """
    if _C is None:
        raise RuntimeError("native extension is not imported; build variants/v9_hw_output_planes_probe first")
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is not available")
    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive")
    _check_gaussian_tensor("means2d", means2d, 2)
    _check_gaussian_tensor("conics", conics, 3)
    _check_gaussian_tensor("colors", colors, 3)
    _check_gaussian_tensor("opacities", opacities, 1)
    g = int(means2d.shape[0])
    if int(conics.shape[0]) != g or int(colors.shape[0]) != g or int(opacities.shape[0]) != g:
        raise ValueError("all Gaussian inputs must have the same G dimension")
    return _C.render_gaussian_eval_rgba(
        means2d.contiguous(),
        conics.contiguous(),
        colors.contiguous(),
        opacities.contiguous(),
        int(height),
        int(width),
        bool(direct),
    )


def render_gaussian_eval_format(
    output_format: str,
    means2d: Tensor,
    conics: Tensor,
    colors: Tensor,
    opacities: Tensor,
    height: int,
    width: int,
    *,
    direct: bool = True,
) -> Tensor:
    """Render screen-space Gaussian splats into an rgba32f or rgba16f MPS tensor."""
    if _C is None:
        raise RuntimeError("native extension is not imported; build variants/v9_hw_output_planes_probe first")
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is not available")
    if output_format not in GAUSSIAN_OUTPUT_FORMATS:
        raise ValueError(f"unknown Gaussian output_format {output_format!r}; expected one of {GAUSSIAN_OUTPUT_FORMATS}")
    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive")
    _check_gaussian_tensor("means2d", means2d, 2)
    _check_gaussian_tensor("conics", conics, 3)
    _check_gaussian_tensor("colors", colors, 3)
    _check_gaussian_tensor("opacities", opacities, 1)
    g = int(means2d.shape[0])
    if int(conics.shape[0]) != g or int(colors.shape[0]) != g or int(opacities.shape[0]) != g:
        raise ValueError("all Gaussian inputs must have the same G dimension")
    return _C.render_gaussian_eval_format(
        str(output_format),
        means2d.contiguous(),
        conics.contiguous(),
        colors.contiguous(),
        opacities.contiguous(),
        int(height),
        int(width),
        bool(direct),
    )


def render_gaussian_eval_rgba16(
    means2d: Tensor,
    conics: Tensor,
    colors: Tensor,
    opacities: Tensor,
    height: int,
    width: int,
    *,
    direct: bool = True,
) -> Tensor:
    """Render screen-space Gaussian splats into an RGBA16F MPS tensor."""
    return render_gaussian_eval_format(
        "rgba16f",
        means2d,
        conics,
        colors,
        opacities,
        height,
        width,
        direct=direct,
    )


def validate_gaussian_eval_output_formats(*, height: int = 16) -> list[dict[str, Any]]:
    """Validate RGBA32F and RGBA16F Gaussian eval with a one-splat readback."""
    rows: list[dict[str, Any]] = []
    if not torch.backends.mps.is_available():
        return rows
    for output_format in GAUSSIAN_OUTPUT_FORMATS:
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
            means = torch.tensor([[8.5, 8.5]], dtype=torch.float32, device="mps")
            conics = torch.tensor([[1.0, 0.0, 1.0]], dtype=torch.float32, device="mps")
            colors = torch.tensor([[0.25, 0.5, 0.75]], dtype=torch.float32, device="mps")
            opacities = torch.tensor([0.8], dtype=torch.float32, device="mps")
            out = render_gaussian_eval_format(output_format, means, conics, colors, opacities, int(height), width)
            got = out.detach().cpu().float()
            center_expected = torch.tensor([0.2, 0.4, 0.6, 0.8], dtype=torch.float32)
            if output_format == "rgba16f":
                center_expected = center_expected.half().float()
            center_err = (got[8, 8] - center_expected).abs().max()
            clear_err = got[0, 0].abs().max()
            max_err = float(torch.maximum(center_err, clear_err).item())
            row["shape"] = tuple(int(x) for x in out.shape)
            row["dtype"] = str(out.dtype)
            row["max_abs_err"] = max_err
            row["validated"] = max_err <= (5.0e-4 if output_format == "rgba16f" else 1.0e-5)
        except Exception as exc:  # pragma: no cover - depends on local Metal runtime.
            row["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
    return rows


def probe_hw_interop(
    *,
    compile_pipelines: bool = True,
    compile_advanced: bool = True,
    run_render_probe: bool = False,
    height: int = 8,
    width: int = 8,
    rgba: RGBA = (0.125, 0.5, 0.875, 1.0),
) -> V9HWOutputPlanesCapabilities:
    native = _native_probe(bool(compile_pipelines), bool(compile_advanced))
    render_validated = False
    render_err: float | None = None
    direct_validated = False
    direct_err: float | None = None
    direct_error = ""
    gaussian_validated = False
    gaussian_err: float | None = None
    gaussian_error = ""
    gaussian16_validated = False
    gaussian16_err: float | None = None
    gaussian16_error = ""
    direct_format_results: list[dict[str, Any]] = []
    direct_format_validation: list[dict[str, Any]] = []
    gaussian_format_validation: list[dict[str, Any]] = []
    validation_uses_cpu_readback = False

    op_available = _C is not None and hasattr(_C, "render_constant_rgba")
    direct_op_available = _C is not None and hasattr(_C, "render_constant_rgba_direct")
    gaussian_op_available = _C is not None and hasattr(_C, "render_gaussian_eval_rgba")
    gaussian_format_op_available = _C is not None and hasattr(_C, "render_gaussian_eval_format")
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
    if run_render_probe and gaussian_op_available and torch.backends.mps.is_available():
        gaussian_height = max(16, int(height))
        gaussian_width = max(16, aligned_direct_width("rgba32f", int(width)))
        try:
            means = torch.tensor([[8.5, 8.5]], dtype=torch.float32, device="mps")
            conics = torch.tensor([[1.0, 0.0, 1.0]], dtype=torch.float32, device="mps")
            colors = torch.tensor([[0.25, 0.5, 0.75]], dtype=torch.float32, device="mps")
            opacities = torch.tensor([0.8], dtype=torch.float32, device="mps")
            out = render_gaussian_eval_rgba(means, conics, colors, opacities, gaussian_height, gaussian_width)
            got = out.detach().cpu()
            validation_uses_cpu_readback = True
            expected_center = torch.tensor([0.2, 0.4, 0.6, 0.8], dtype=torch.float32)
            center_err = (got[8, 8] - expected_center).abs().max()
            clear_err = got[0, 0].abs().max()
            gaussian_err = float(torch.maximum(center_err, clear_err).item())
            gaussian_validated = gaussian_err <= 1.0e-5
        except Exception as exc:  # pragma: no cover - depends on local Metal runtime.
            gaussian_error = f"{type(exc).__name__}: {exc}"
    if run_render_probe and direct_output_format_probe_available and torch.backends.mps.is_available():
        direct_format_results = probe_direct_output_formats(
            height=1,
            widths=(16, 32, 64, 1920, 3840),
        )
        direct_format_validation = validate_direct_output_formats(height=max(16, int(height)), rgba=rgba)
        validation_uses_cpu_readback = True
    if run_render_probe and gaussian_format_op_available and torch.backends.mps.is_available():
        gaussian_format_validation = validate_gaussian_eval_output_formats(height=max(16, int(height)))
        validation_uses_cpu_readback = True
        for row in gaussian_format_validation:
            if row.get("format") == "rgba32f":
                gaussian_validated = bool(row.get("validated"))
                gaussian_err = row.get("max_abs_err")  # type: ignore[assignment]
                gaussian_error = str(row.get("error", ""))
            elif row.get("format") == "rgba16f":
                gaussian16_validated = bool(row.get("validated"))
                gaussian16_err = row.get("max_abs_err")  # type: ignore[assignment]
                gaussian16_error = str(row.get("error", ""))

    return V9HWOutputPlanesCapabilities(
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
        gaussian_eval_pipeline_ready=_optional_bool(native.get("gaussian_eval_pipeline_ready")),
        gaussian_eval_pipeline_error=str(native.get("gaussian_eval_pipeline_error", "")),
        tile_pipeline_ready=_optional_bool(native.get("tile_pipeline_ready")),
        tile_pipeline_error=str(native.get("tile_pipeline_error", "")),
        tile_imageblock_sample_length=_optional_int(native.get("tile_imageblock_sample_length")),
        tile_imageblock_memory_16x16=_optional_int(native.get("tile_imageblock_memory_16x16")),
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
        gaussian_eval_rgba_op_available=gaussian_op_available,
        gaussian_eval_rgba_validated=gaussian_validated,
        gaussian_eval_rgba_max_abs_err=gaussian_err,
        gaussian_eval_rgba_error=gaussian_error,
        gaussian_eval_rgba16_validated=gaussian16_validated,
        gaussian_eval_rgba16_max_abs_err=gaussian16_err,
        gaussian_eval_rgba16_error=gaussian16_error,
        direct_output_format_probe_available=direct_output_format_probe_available,
        direct_output_format_results=direct_format_results,
        direct_output_format_validation=direct_format_validation,
        gaussian_output_format_validation=gaussian_format_validation,
        validation_uses_cpu_readback=validation_uses_cpu_readback,
        native_op_uses_cpu_readback=bool(native.get("native_op_uses_cpu_readback", True)),
        native_details=native,
    )

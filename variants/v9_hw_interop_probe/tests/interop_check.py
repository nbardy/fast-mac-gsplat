from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from torch_gsplat_bridge_v9_hw_interop import probe_hw_interop, render_constant_rgba, render_constant_rgba_direct


def main() -> None:
    status = probe_hw_interop(compile_pipelines=True, compile_advanced=True, run_render_probe=False)
    print(json.dumps(status.as_dict(), indent=2, sort_keys=True))
    assert status.native_extension_loaded
    assert status.native_probe_available
    assert status.metal_available
    assert status.render_pipeline_ready is True, status.render_pipeline_error

    if not torch.backends.mps.is_available():
        print("MPS unavailable; native compile probe passed, render-to-tensor validation skipped.")
        return

    rgba = (0.125, 0.5, 0.875, 1.0)
    out = render_constant_rgba(9, 7, rgba)
    assert out.device.type == "mps"
    assert tuple(out.shape) == (9, 7, 4)
    expected = torch.tensor(rgba, dtype=torch.float32).view(1, 1, 4).expand(9, 7, 4)
    max_err = float((out.detach().cpu() - expected).abs().max().item())
    assert max_err == 0.0, max_err

    direct = render_constant_rgba_direct(16, 16, rgba)
    direct_expected = torch.tensor(rgba, dtype=torch.float32).view(1, 1, 4).expand(16, 16, 4)
    direct_max_err = float((direct.detach().cpu() - direct_expected).abs().max().item())
    assert direct_max_err == 0.0, direct_max_err

    checked = probe_hw_interop(compile_pipelines=True, compile_advanced=True, run_render_probe=True)
    assert checked.render_to_mps_tensor_validated, checked.as_dict()
    assert checked.direct_render_to_mps_tensor_validated, checked.as_dict()
    print(f"render_to_mps_tensor_max_abs_err={checked.render_to_mps_tensor_max_abs_err}")


if __name__ == "__main__":
    main()

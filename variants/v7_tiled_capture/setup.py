from __future__ import annotations

import sys
from pathlib import Path

from setuptools import find_packages, setup

try:
    from torch.utils.cpp_extension import BuildExtension, CppExtension
except Exception as exc:  # pragma: no cover
    raise RuntimeError("PyTorch is required to build this extension") from exc

ROOT = Path(__file__).resolve().parent
IS_DARWIN = sys.platform == "darwin"

sources = [str(ROOT / "csrc" / "bindings.cpp")]
include_dirs = [str(ROOT / "csrc")]
extra_compile_args = {"cxx": ["-O3", "-std=c++17"]}
extra_link_args: list[str] = []

if IS_DARWIN:
    sources.append(str(ROOT / "csrc" / "metal" / "gsplat_sparse.mm"))
    extra_compile_args["cxx"] += ["-ObjC++", "-fobjc-arc"]
    extra_link_args += [
        "-framework", "Foundation",
        "-framework", "Metal",
        "-framework", "MetalPerformanceShaders",
        "-framework", "QuartzCore",
    ]

ext_modules = [
    CppExtension(
        name="torch_gsplat_bridge_v72._C",
        sources=sources,
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    )
]

setup(
    name="torch-metal-gsplat-v72-tiled-capture",
    version="0.7.2",
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
    zip_safe=False,
)

from __future__ import annotations

import sys
from pathlib import Path

from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CppExtension

ROOT = Path(__file__).resolve().parent
IS_DARWIN = sys.platform == "darwin"

sources = ["csrc/bindings.cpp"]
extra_compile_args = {"cxx": ["-O3", "-std=c++17"]}
include_dirs = [str(ROOT / "csrc")]
extra_link_args = []

define_macros = []

if IS_DARWIN:
    sources.append("csrc/metal/gsplat_metal.mm")
    extra_compile_args["cxx"] += ["-ObjC++", "-fobjc-arc"]
    extra_link_args += [
        "-framework", "Foundation",
        "-framework", "Metal",
        "-framework", "MetalPerformanceShaders",
        "-framework", "MetalPerformanceShadersGraph",
    ]
    define_macros += [("GSPAT_USE_METAL", "1")]

ext_modules = [
    CppExtension(
        name="torch_gsplat_bridge_fast._C",
        sources=sources,
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        define_macros=define_macros,
    )
]

setup(
    name="fast-mac-gsplat",
    version="0.2.0",
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
    zip_safe=False,
)

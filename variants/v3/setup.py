from __future__ import annotations

import sys
from pathlib import Path
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CppExtension

ROOT = Path(__file__).resolve().parent

extra_compile_args = ["-std=c++17"]
extra_link_args = []
if sys.platform == "darwin":
    extra_compile_args += ["-fobjc-arc"]
    extra_link_args += ["-framework", "Foundation", "-framework", "Metal"]

ext = CppExtension(
    name="torch_gsplat_bridge_v3._C",
    sources=[
        "csrc/bindings.cpp",
        "csrc/metal/gsplat_metal.mm",
    ],
    include_dirs=[str(ROOT / "csrc")],
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
)

setup(
    name="torch-metal-gsplat-v3",
    version="0.1.0",
    packages=find_packages(),
    ext_modules=[ext],
    cmdclass={"build_ext": BuildExtension},
    zip_safe=False,
)

import sys
from pathlib import Path

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

this_dir = Path(__file__).resolve().parent

sources = [str(this_dir / "csrc" / "bindings.cpp")]
extra_compile_args = ["-std=c++17"]
extra_link_args = []

if sys.platform == "darwin":
    sources.append(str(this_dir / "csrc" / "metal" / "star_prt_metal.mm"))
    extra_compile_args.append("-fobjc-arc")
    extra_link_args.extend(["-framework", "Foundation", "-framework", "Metal"])

ext_modules = [
    CppExtension(
        name="torch_gsplat_bridge_star_prt._C",
        sources=sources,
        include_dirs=[str(this_dir / "csrc")],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    )
]

setup(
    name="torch-gsplat-star-prt-v0",
    version="0.1.0",
    description="STAR Projective Rational Tube renderer scaffold",
    packages=find_packages(where="."),
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
    zip_safe=False,
)

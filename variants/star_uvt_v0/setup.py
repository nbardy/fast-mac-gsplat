from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

this_dir = Path(__file__).resolve().parent

sources = [
    str(this_dir / "csrc" / "bindings.cpp"),
    str(this_dir / "csrc" / "metal" / "star_uvt_metal.mm"),
]

ext_modules = [
    CppExtension(
        name="torch_gsplat_bridge_star_uvt._C",
        sources=sources,
        include_dirs=[str(this_dir / "csrc")],
        extra_compile_args=["-std=c++17", "-fobjc-arc"],
        extra_link_args=["-framework", "Foundation", "-framework", "Metal"],
    )
]

setup(
    name="torch-gsplat-star-uvt-v0",
    version="0.1.0",
    description="Gate 0 Metal UVT screen-time tube renderer",
    packages=["torch_gsplat_bridge_star_uvt"],
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
    zip_safe=False,
)


from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension


this_dir = Path(__file__).resolve().parent

sources = [
    str(this_dir / "csrc" / "bindings.cpp"),
    str(this_dir / "csrc" / "metal" / "world_foam_lane2_metal.mm"),
]

ext_modules = [
    CppExtension(
        name="torch_world_foam_lane2_fused_slab._C",
        sources=sources,
        include_dirs=[str(this_dir / "csrc")],
        extra_compile_args=["-std=c++17", "-fobjc-arc"],
        extra_link_args=["-framework", "Foundation", "-framework", "Metal"],
    )
]

setup(
    name="torch-world-foam-lane2-fused-slab-v0",
    version="0.1.0",
    description="Gate 0 World Foam Lane 2 Metal count kernels",
    packages=["torch_world_foam_lane2_fused_slab"],
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
    zip_safe=False,
)

from pathlib import Path
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

this_dir = Path(__file__).resolve().parent

sources = [
    str(this_dir / "csrc" / "bindings.cpp"),
    str(this_dir / "csrc" / "metal" / "gsplat_metal.mm"),
]

ext_modules = [
    CppExtension(
        name="torch_gsplat_bridge_v12a_fused_colorize_l1_no_norm._C",
        sources=sources,
        include_dirs=[str(this_dir / "csrc")],
        extra_compile_args=["-std=c++17", "-fobjc-arc"],
        extra_link_args=["-framework", "Foundation", "-framework", "Metal"],
    )
]

setup(
    name="torch-metal-gsplat-v12a-fused-colorize-l1-no-norm",
    version="0.1.0",
    description="Torch-native Metal projected Gaussian feature rasterizer plus fused no-norm colorize L1 gradient producer",
    packages=["torch_gsplat_bridge_v12a_fused_colorize_l1_no_norm"],
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
    zip_safe=False,
)

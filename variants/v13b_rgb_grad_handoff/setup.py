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
        name="torch_gsplat_bridge_v13b_rgb_grad_handoff._C",
        sources=sources,
        include_dirs=[str(this_dir / "csrc")],
        extra_compile_args=["-std=c++17", "-fobjc-arc"],
        extra_link_args=["-framework", "Foundation", "-framework", "Metal"],
    )
]

setup(
    name="torch-metal-gsplat-v13b-rgb-grad-handoff",
    version="0.1.0",
    description="Torch-native Metal projected Gaussian feature rasterizer with RGB-gradient handoff scaffold",
    packages=["torch_gsplat_bridge_v13b_rgb_grad_handoff"],
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
    zip_safe=False,
)

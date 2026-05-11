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
        name="torch_gsplat_bridge_v10_features_gradcache_zero_bg_hostmeta._C",
        sources=sources,
        include_dirs=[str(this_dir / "csrc")],
        extra_compile_args=["-std=c++17", "-fobjc-arc"],
        extra_link_args=["-framework", "Foundation", "-framework", "Metal"],
    )
]

setup(
    name="torch-metal-gsplat-v10-features-gradcache-zero-bg-hostmeta",
    version="0.1.0",
    description="Torch-native Metal projected Gaussian feature rasterizer with grad-cache and zero-background tail skip",
    packages=["torch_gsplat_bridge_v10_features_gradcache_zero_bg_hostmeta"],
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
    zip_safe=False,
)

from pathlib import Path
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

this_dir = Path(__file__).resolve().parent

sources = [
    str(this_dir / "csrc" / "metal" / "v9_hw_interop.mm"),
]

ext_modules = [
    CppExtension(
        name="torch_gsplat_bridge_v9_hw_interop._C",
        sources=sources,
        include_dirs=[str(this_dir / "csrc")],
        extra_compile_args=["-std=c++17", "-fobjc-arc"],
        extra_link_args=[
            "-framework",
            "Foundation",
            "-framework",
            "Metal",
            "-framework",
            "MetalPerformanceShaders",
            "-framework",
            "MetalPerformanceShadersGraph",
        ],
    )
]

setup(
    name="torch-metal-gsplat-v9-hw-interop",
    version="0.1.0",
    description="V9 hardware raster interop probe for Torch MPS tensors",
    packages=["torch_gsplat_bridge_v9_hw_interop"],
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
    zip_safe=False,
)

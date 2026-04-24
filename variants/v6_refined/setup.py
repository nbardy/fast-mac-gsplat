
from pathlib import Path
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

ROOT = Path(__file__).resolve().parent

ext = CppExtension(
    name='torch_gsplat_bridge_v6._C',
    sources=[
        str(ROOT / 'csrc' / 'bindings.cpp'),
        str(ROOT / 'csrc' / 'metal' / 'gsplat_metal.mm'),
    ],
    include_dirs=[str(ROOT / 'csrc')],
    extra_compile_args={'cxx': ['-O3', '-std=c++17']},
    extra_link_args=['-framework', 'Foundation', '-framework', 'Metal'],
)

setup(
    name='torch-metal-gsplat-v6-refined',
    version='0.0.1',
    description='Torch+Metal projected Gaussian rasterizer v6 refined handoff',
    packages=['torch_gsplat_bridge_v6'],
    ext_modules=[ext],
    cmdclass={'build_ext': BuildExtension},
    zip_safe=False,
)

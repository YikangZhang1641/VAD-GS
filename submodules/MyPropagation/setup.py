from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='patchmatch_cuda',
    ext_modules=[
        CUDAExtension(
            name='patchmatch_cuda',
            sources=[
                'propagation_pybind.cpp',
                'Propagation.cpp', 
                'Propagation_cuda.cu',  
            ],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': ['-O3', '--use_fast_math']
            }
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)
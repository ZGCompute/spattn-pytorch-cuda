from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='spattn_cuda',
    ext_modules=[
        CUDAExtension(
            name='spattn_cuda',
            sources=['../cuda_kernels/spattn_kernel.cu'],
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)

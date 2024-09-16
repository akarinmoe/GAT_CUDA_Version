from torch.utils.cpp_extension import BuildExtension, CUDAExtension
from setuptools import setup

setup(
    name='gat_cuda',
    ext_modules=[
        CUDAExtension(
            name='gat_cuda',
            sources=[
                'kernel_folder/Kernels/gat.cpp',
                'kernel_folder/Kernels/linear_transform.cu',
                'kernel_folder/Kernels/aggregate_features.cu',
                'kernel_folder/Kernels/softmax_kernel.cu',
                'kernel_folder/Kernels/compute_attention_coeff.cu',
            ],
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)

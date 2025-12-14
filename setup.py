from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='pointpillars',
    version='0.1',
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name='pointpillars.ops.voxel_op',
            sources=[
                'pointpillars/ops/voxelization/voxelization.cpp',
                'pointpillars/ops/voxelization/voxelization_cpu.cpp',
                'pointpillars/ops/voxelization/voxelization_cuda.cu',
            ],
            define_macros=[('WITH_CUDA', None)]
        ),
        CUDAExtension(
            name='pointpillars.ops.iou3d_op',
            sources=[
                'pointpillars/ops/iou3d/iou3d.cpp',
                'pointpillars/ops/iou3d/iou3d_kernel.cu',
            ],
            define_macros=[('WITH_CUDA', None)]
        ),
        CUDAExtension(
            name='pointpillars.ops.bev_scatter_op',
            sources=[
                'pointpillars/ops/bev_scatter/bev_scatter.cpp',
                'pointpillars/ops/bev_scatter/bev_scatter_cuda.cu',
            ],
            define_macros=[('WITH_CUDA', None)],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': ['-O3'],
            },
        )      
    ],
    cmdclass={'build_ext': BuildExtension},
    zip_safe=False
)
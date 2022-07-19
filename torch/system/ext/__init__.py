from pathlib import Path
from torch.utils.cpp_extension import load
import os

def p(rel_path):
    abs_path = Path(__file__).parent / rel_path
    return str(abs_path)


# Load in Marching cubes.
_marching_cubes_module = load(name='marching_cubes',
                              sources=[p('marching_cubes/mc.cpp'),
                                       p('marching_cubes/mc_interp_kernel.cu'),
                                       p('marching_cubes/mc_kernel.cu')],
                                       extra_cuda_cflags=['-Xcompiler -fno-gnu-unique'],
                              verbose=False)
# marching_cubes = _marching_cubes_module.marching_cubes_sparse
marching_cubes_interp = _marching_cubes_module.marching_cubes_sparse_interp
marching_cubes = _marching_cubes_module.marching_cubes_sparse
marching_cubes_dense = _marching_cubes_module.marching_cubes_dense

# Load in Image processing modules.
_imgproc_module = load(name='imgproc',
                       sources=[p('imgproc/imgproc.cu'), p('imgproc/imgproc.cpp'), p('imgproc/photometric.cu')],
                       extra_cuda_cflags=['-Xcompiler -fno-gnu-unique'],
                       verbose=False)
unproject_depth = _imgproc_module.unproject_depth
compute_normal_weight = _imgproc_module.compute_normal_weight
compute_normal_weight_robust = _imgproc_module.compute_normal_weight_robust
filter_depth = _imgproc_module.filter_depth
rgb_odometry = _imgproc_module.rgb_odometry
gradient_xy = _imgproc_module.gradient_xy

# Load in Indexing modules. (which deal with complicated indexing scheme)
_indexing_module = load(name='indexing',
                        sources=[p('indexing/indexing.cpp'), p('indexing/indexing.cu')],
                        extra_cuda_cflags=['-Xcompiler -fno-gnu-unique'],
                        verbose=False)
pack_batch = _indexing_module.pack_batch
groupby_max = _indexing_module.groupby_max
groupby_sum = _indexing_module.groupby_sum
groupby_sum_backward = _indexing_module.groupby_sum_backward
# We need point cloud processing module.
_pcproc_module = load(name='pcproc',
                      sources=[p('pcproc/pcproc.cpp'), p('pcproc/pcproc.cu'), p('pcproc/cuda_kdtree.cu'), p('pcproc/cuda_dda.cu')],
                      extra_cuda_cflags=['-Xcompiler -fno-gnu-unique']
                      ,verbose=False)
remove_radius_outlier = _pcproc_module.remove_radius_outlier
estimate_normals = _pcproc_module.estimate_normals
set_empty_voxels = _pcproc_module.setEmptyVoxels
compute_sdf = _pcproc_module.compute_sdf
valid_mask = _pcproc_module.valid_mask

_render_module = load(name='render',
                        sources = [p('render/render.cpp'),p('render/render.cu')],
                        extra_cflags = ["-O3"],
                        extra_cuda_cflags = ["-O3",'-Xcompiler -fno-gnu-unique'],
                        verbose = False)
sparse_ray_intersection = _render_module.sparse_ray_intersection
generate_rays = _render_module.generate_rays
build_octree = _render_module.build_octree


OPTIX_PATH = '/home/chx/Downloads/NVIDIA-OptiX-SDK-7.4.0-linux64-x86_64/include'
USE_OPTIX = False

# We want to isolate this build, because it uses Eigen that causes compilation nuisances.
__BUILD_SDF_FROM_MESH = USE_OPTIX

_sdfgen_module = load(name='sdfgen',
                      sources=[p('sdfgen/bind.cpp'), p('sdfgen/cuda_kdtree.cu'),
                               p('sdfgen/sdf_from_points.cu')] +
                              ([p('sdfgen/sdf_from_mesh.cu'),
                                p('sdfgen/triangle_bvh.cu')] if __BUILD_SDF_FROM_MESH else []),
                      verbose=False,
                      extra_cflags=['-O2'],
                      # https://github.com/pytorch/pytorch/issues/52663
                      #     Thrust/CUB has a bug that causes conflict w/ pytorch after cuda 11.
                      #     (e.g. xxx failed on 1st step InvalidDeviceOrdinal)
                      # Note: pytorch itself also use thrust for, e.g., sorting, but it is gradually eliminating it.
                      extra_cuda_cflags=['-O2', '-Xcompiler -fno-gnu-unique', '-DNGP_OPTIX' if USE_OPTIX else '',
                                         '-DBUILD_SDF_FROM_MESH' if __BUILD_SDF_FROM_MESH else ''],
                      extra_include_paths=[OPTIX_PATH,'/usr/local/include/eigen3'] if USE_OPTIX else [])
sdf_from_points = _sdfgen_module.sdf_from_points
# sdf_mode = _sdfgen_module.MeshSdfMode
# sdf_from_mesh = _sdfgen_module.sdf_from_mesh

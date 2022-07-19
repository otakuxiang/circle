#include "mc_data.cuh"



__device__ static inline float get_sdf(uint3 bpos, uint arx, uint ary, uint arz,
                                              const IndexerAccessor indexer,
                                              const CubeSDFAccessor cube_sdf
                                              ) {
    uint r = cube_sdf.size(1);
    if (arx >= r) { bpos.x += 1; arx = 0; }
    // if (bpos.x >= bsize.x) { bpos.x = bsize.x - 1; rpos.x = r - 1; }
    if (ary >= r) { bpos.y += 1; ary = 0; }
    // if (bpos.y >= bsize.y) { bpos.y = bsize.y - 1; rpos.y = r - 1; }
    if (arz >= r) { bpos.z += 1; arz = 0; }
    // if (bpos.z >= bsize.z) { bpos.z = bsize.z - 1; rpos.z = r - 1; }
    if (bpos.x >= indexer.size(0) || bpos.y >= indexer.size(1) || bpos.z >= indexer.size(2)) {
        return NAN;
    }
//    printf("B-Getting: %d %d %d --> %d, %d, %d\n", bx, by, bz, indexer.size(0), indexer.size(1), indexer.size(2));
    int batch_ind = indexer[bpos.x][bpos.y][bpos.z];
    if (batch_ind == -1) {
        return NAN;
    }
//    printf("Getting: %d %d %d %d --> %d %d\n", batch_ind, arx, ary, arz, cube_sdf.size(0), cube_sdf.size(1));
    float sdf = cube_sdf[batch_ind][arx][ary][arz];

    return sdf;

}


__device__ static inline float3 sdf_interp(const float3 p1, const float3 p2, float valp1, float valp2) {
    if (fabs(0.0f - valp1) < 1.0e-5f) return p1;
	if (fabs(0.0f - valp2) < 1.0e-5f) return p2;
	if (fabs(valp1 - valp2) < 1.0e-5f) return p1;

	float w2 = (0.0f - valp1) / (valp2 - valp1);
	float w1 = 1 - w2;

	return make_float3(p1.x * w1 + p2.x * w2,
	                   p1.y * w1 + p2.y * w2,
	                   p1.z * w1 + p2.z * w2);
}

__global__ static void meshing_cube(const ValidBlocksAccessor valid_blocks,
                                    const IndexerAccessor indexer,
                                    const CubeSDFAccessor cube_sdf,
                                    TrianglesAccessor triangles,
                                    TriangleVecIdAccessor triangle_flatten_id,
                                    int* __restrict__ triangles_count,
                                    int max_triangles_count,
                                    int nx, int ny, int nz
                                    ) {
    const uint r = cube_sdf.size(1);
    const uint r3 = r * r * r;
    const uint num_lif = valid_blocks.size(0);
    const float sbs = 1.0f / r;         // sub-block-size

    const uint lif_id = blockIdx.x * blockDim.x + threadIdx.x;
    const uint sub_id = blockIdx.y * blockDim.y + threadIdx.y;

    if (lif_id >= num_lif || sub_id >= r3) {
        return;
    }

    const uint3 bpos = make_uint3(
        (valid_blocks[lif_id] / (ny * nz)) % nx,
        (valid_blocks[lif_id] / nz) % ny,
        valid_blocks[lif_id] % nz);
    const uint3 bsize = make_uint3(indexer.size(0), indexer.size(1), indexer.size(2));
    const uint rx = sub_id / (r * r);
    const uint ry = (sub_id / r) % r;
    const uint rz = sub_id % r;

    // Find all 8 neighbours
    float3 points[8];
    float sdf_vals[8];

    sdf_vals[0] = get_sdf(bpos, rx, ry, rz, indexer, cube_sdf);
    if (isnan(sdf_vals[0])) return;
    points[0] = make_float3(bpos.x + rx * sbs, bpos.y + ry * sbs, bpos.z + rz * sbs);

    sdf_vals[1] = get_sdf(bpos, rx + 1, ry, rz, indexer, cube_sdf);
    if (isnan(sdf_vals[1])) return;
    points[1] = make_float3(bpos.x + (rx + 1) * sbs, bpos.y + ry * sbs, bpos.z + rz * sbs);

    sdf_vals[2] = get_sdf(bpos,rx + 1, ry + 1, rz, indexer, cube_sdf);
    if (isnan(sdf_vals[2])) return;
    points[2] = make_float3(bpos.x + (rx + 1) * sbs, bpos.y + (ry + 1) * sbs, bpos.z + rz * sbs);

    sdf_vals[3] = get_sdf(bpos, rx, ry + 1, rz, indexer, cube_sdf);
    if (isnan(sdf_vals[3])) return;
    points[3] = make_float3(bpos.x + rx * sbs, bpos.y + (ry + 1) * sbs, bpos.z + rz * sbs);

    sdf_vals[4] = get_sdf(bpos, rx, ry, rz + 1, indexer, cube_sdf);
    if (isnan(sdf_vals[4])) return;
    points[4] = make_float3(bpos.x + rx * sbs, bpos.y + ry * sbs, bpos.z + (rz + 1) * sbs);

    sdf_vals[5] = get_sdf(bpos, rx + 1, ry, rz + 1, indexer, cube_sdf);
    if (isnan(sdf_vals[5])) return;
    points[5] = make_float3(bpos.x + (rx + 1) * sbs, bpos.y + ry * sbs, bpos.z + (rz + 1) * sbs);

    sdf_vals[6] = get_sdf(bpos,rx + 1, ry + 1, rz + 1, indexer, cube_sdf);
    if (isnan(sdf_vals[6])) return;
    points[6] = make_float3(bpos.x + (rx + 1) * sbs, bpos.y + (ry + 1) * sbs, bpos.z + (rz + 1) * sbs);

    sdf_vals[7] = get_sdf(bpos, rx, ry + 1, rz + 1, indexer, cube_sdf);
    if (isnan(sdf_vals[7])) return;
    points[7] = make_float3(bpos.x + rx * sbs, bpos.y + (ry + 1) * sbs, bpos.z + (rz + 1) * sbs);

    // Find triangle config.
    int cube_type = 0;
	if (sdf_vals[0] < 0) cube_type |= 1; if (sdf_vals[1] < 0) cube_type |= 2;
	if (sdf_vals[2] < 0) cube_type |= 4; if (sdf_vals[3] < 0) cube_type |= 8;
	if (sdf_vals[4] < 0) cube_type |= 16; if (sdf_vals[5] < 0) cube_type |= 32;
	if (sdf_vals[6] < 0) cube_type |= 64; if (sdf_vals[7] < 0) cube_type |= 128;

	// Find vertex position on each edge (weighted by sdf value)
	int edge_config = edgeTable[cube_type];
	float3 vert_list[12];

	if (edge_config == 0) return;
    if (edge_config & 1) vert_list[0] = sdf_interp(points[0], points[1], sdf_vals[0], sdf_vals[1]);
	if (edge_config & 2) vert_list[1] = sdf_interp(points[1], points[2], sdf_vals[1], sdf_vals[2]);
	if (edge_config & 4) vert_list[2] = sdf_interp(points[2], points[3], sdf_vals[2], sdf_vals[3]);
	if (edge_config & 8) vert_list[3] = sdf_interp(points[3], points[0], sdf_vals[3], sdf_vals[0]);
	if (edge_config & 16) vert_list[4] = sdf_interp(points[4], points[5], sdf_vals[4], sdf_vals[5]);
	if (edge_config & 32) vert_list[5] = sdf_interp(points[5], points[6], sdf_vals[5], sdf_vals[6]);
	if (edge_config & 64) vert_list[6] = sdf_interp(points[6], points[7], sdf_vals[6], sdf_vals[7]);
	if (edge_config & 128) vert_list[7] = sdf_interp(points[7], points[4], sdf_vals[7], sdf_vals[4]);
	if (edge_config & 256) vert_list[8] = sdf_interp(points[0], points[4], sdf_vals[0], sdf_vals[4]);
	if (edge_config & 512) vert_list[9] = sdf_interp(points[1], points[5], sdf_vals[1], sdf_vals[5]);
	if (edge_config & 1024) vert_list[10] = sdf_interp(points[2], points[6], sdf_vals[2], sdf_vals[6]);
	if (edge_config & 2048) vert_list[11] = sdf_interp(points[3], points[7], sdf_vals[3], sdf_vals[7]);

    float3 vp[3];
    // Write triangles to array.
    for (int i = 0; triangleTable[cube_type][i] != -1; i += 3) {
#pragma unroll
        for (int vi = 0; vi < 3; ++vi) {
            vp[vi] = vert_list[triangleTable[cube_type][i + vi]];
        }
        int triangle_id = atomicAdd(triangles_count, 1);
        if (triangle_id < max_triangles_count) {
#pragma unroll
            for (int vi = 0; vi < 3; ++ vi) {
                triangles[triangle_id][vi][0] = vp[vi].x;
                triangles[triangle_id][vi][1] = vp[vi].y;
                triangles[triangle_id][vi][2] = vp[vi].z;
            }
            triangle_flatten_id[triangle_id] = valid_blocks[lif_id];
        }
    }

}




__inline__ __device__ bool isvalid(uint x,uint y,uint z,const MaskAccessor mask,uint nx,uint ny,uint nz){
    if (x >= nx || y >= ny || z >= nz){
        return false;
    } 
    return mask[x][y][z];
}

__global__ static void meshing_cube_dense(const CordAccessor valid_cords,
                                    const MaskAccessor mask,
                                    const TrianglesAccessor dense_sdf,
                                    TrianglesAccessor triangles,
                                    int* __restrict__ triangles_count,
                                    int max_triangles_count
                                    ) {
    const uint num_lif = valid_cords.size(0);
    const uint cord_id = blockIdx.x * blockDim.x + threadIdx.x;
    const uint nx = mask.size(0),ny = mask.size(1),nz = mask.size(2);
    if (cord_id >= num_lif) {
        return;
    }

    const uint3 bpos = make_uint3(
        valid_cords[cord_id][0],
        valid_cords[cord_id][1],
        valid_cords[cord_id][2]);

    // Find all 8 neighbours
    float3 points[8];
    float sdf_vals[8];

    if(!isvalid(bpos.x,bpos.y,bpos.z,mask,nx,ny,nz)) return;
    sdf_vals[0] = dense_sdf[bpos.x][bpos.y][bpos.z];
    points[0] = make_float3(bpos.x, bpos.y, bpos.z );

    if(!isvalid(bpos.x + 1,bpos.y,bpos.z,mask,nx,ny,nz))return;
    sdf_vals[1] = dense_sdf[bpos.x + 1][bpos.y][bpos.z];
    points[1] = make_float3(bpos.x +1, bpos.y , bpos.z );

    if(!isvalid(bpos.x + 1,bpos.y + 1,bpos.z,mask,nx,ny,nz))return;
    sdf_vals[2] = dense_sdf[bpos.x + 1][bpos.y + 1][bpos.z];
    points[2] = make_float3(bpos.x + 1, bpos.y + 1, bpos.z );

    if(!isvalid(bpos.x,bpos.y + 1,bpos.z,mask,nx,ny,nz))return;
    sdf_vals[3] = dense_sdf[bpos.x][bpos.y + 1][bpos.z];
    points[3] = make_float3(bpos.x , bpos.y + 1, bpos.z);

    if(!isvalid(bpos.x,bpos.y ,bpos.z + 1,mask,nx,ny,nz))return;
    sdf_vals[4] = dense_sdf[bpos.x][bpos.y][bpos.z + 1];
    points[4] = make_float3(bpos.x , bpos.y , bpos.z + 1);

    if(!isvalid(bpos.x + 1,bpos.y ,bpos.z + 1,mask,nx,ny,nz))return;
    sdf_vals[5] = dense_sdf[bpos.x + 1][bpos.y][bpos.z + 1];
    points[5] = make_float3(bpos.x + 1, bpos.y , bpos.z + 1);

    if(!isvalid(bpos.x + 1,bpos.y + 1,bpos.z + 1,mask,nx,ny,nz))return;
    sdf_vals[6] = dense_sdf[bpos.x + 1][bpos.y + 1][bpos.z + 1];
    points[6] = make_float3(bpos.x + 1, bpos.y + 1, bpos.z + 1);
    
    if(!isvalid(bpos.x,bpos.y + 1,bpos.z + 1,mask,nx,ny,nz))return;
    sdf_vals[7] = dense_sdf[bpos.x][bpos.y + 1][bpos.z + 1];
    points[7] = make_float3(bpos.x , bpos.y + 1, bpos.z + 1);
    // printf("%d %d %d\n",bpos.x,bpos.y,bpos.z);

    // Find triangle config.
    int cube_type = 0;
	if (sdf_vals[0] < 0) cube_type |= 1; if (sdf_vals[1] < 0) cube_type |= 2;
	if (sdf_vals[2] < 0) cube_type |= 4; if (sdf_vals[3] < 0) cube_type |= 8;
	if (sdf_vals[4] < 0) cube_type |= 16; if (sdf_vals[5] < 0) cube_type |= 32;
	if (sdf_vals[6] < 0) cube_type |= 64; if (sdf_vals[7] < 0) cube_type |= 128;

	// Find vertex position on each edge (weighted by sdf value)
	int edge_config = edgeTable[cube_type];
	float3 vert_list[12];

	if (edge_config == 0) return;
    if (edge_config & 1) vert_list[0] = sdf_interp(points[0], points[1], sdf_vals[0], sdf_vals[1]);
	if (edge_config & 2) vert_list[1] = sdf_interp(points[1], points[2], sdf_vals[1], sdf_vals[2]);
	if (edge_config & 4) vert_list[2] = sdf_interp(points[2], points[3], sdf_vals[2], sdf_vals[3]);
	if (edge_config & 8) vert_list[3] = sdf_interp(points[3], points[0], sdf_vals[3], sdf_vals[0]);
	if (edge_config & 16) vert_list[4] = sdf_interp(points[4], points[5], sdf_vals[4], sdf_vals[5]);
	if (edge_config & 32) vert_list[5] = sdf_interp(points[5], points[6], sdf_vals[5], sdf_vals[6]);
	if (edge_config & 64) vert_list[6] = sdf_interp(points[6], points[7], sdf_vals[6], sdf_vals[7]);
	if (edge_config & 128) vert_list[7] = sdf_interp(points[7], points[4], sdf_vals[7], sdf_vals[4]);
	if (edge_config & 256) vert_list[8] = sdf_interp(points[0], points[4], sdf_vals[0], sdf_vals[4]);
	if (edge_config & 512) vert_list[9] = sdf_interp(points[1], points[5], sdf_vals[1], sdf_vals[5]);
	if (edge_config & 1024) vert_list[10] = sdf_interp(points[2], points[6], sdf_vals[2], sdf_vals[6]);
	if (edge_config & 2048) vert_list[11] = sdf_interp(points[3], points[7], sdf_vals[3], sdf_vals[7]);
    float3 vp[3];
    // Write triangles to array.
    for (int i = 0; triangleTable[cube_type][i] != -1; i += 3) {
#pragma unroll
        for (int vi = 0; vi < 3; ++vi) {
            vp[vi] = vert_list[triangleTable[cube_type][i + vi]];
        }
        int triangle_id = atomicAdd(triangles_count, 1);
        if (triangle_id < max_triangles_count) {
#pragma unroll
            for (int vi = 0; vi < 3; ++ vi) {
                triangles[triangle_id][vi][0] = vp[vi].x;
                triangles[triangle_id][vi][1] = vp[vi].y;
                triangles[triangle_id][vi][2] = vp[vi].z;
            }
        }
    }

}



std::vector<torch::Tensor> marching_cubes_sparse_cuda(
    torch::Tensor valid_blocks,         // (K, )     
    torch::Tensor batch_indexer,    // (nx,ny,nz) -> batch id
    torch::Tensor cube_sdf,             // (M, rx, ry, rz)
    int max_n_triangles,                // Maximum number of triangle buffer.
    const std::vector<int>& n_xyz    // [nx, ny, nz]
) {
    CHECK_INPUT(valid_blocks);
    CHECK_INPUT(cube_sdf);
    CHECK_INPUT(batch_indexer);
    assert(max_n_triangles > 0);

    const int r = cube_sdf.size(1);
    const int r3 = r * r * r;
    const int num_lif = valid_blocks.size(0);

    torch::Tensor triangles = torch::empty({max_n_triangles, 3, 3},
                                           torch::dtype(torch::kFloat32).device(torch::kCUDA));
    torch::Tensor triangle_flatten_id = torch::empty({max_n_triangles}, torch::dtype(torch::kLong).device(torch::kCUDA));

    dim3 dimBlock = dim3(16, 16);
    uint xBlocks = (num_lif + dimBlock.x - 1) / dimBlock.x;
    uint yBlocks = (r3 + dimBlock.y - 1) / dimBlock.y;
    dim3 dimGrid = dim3(xBlocks, yBlocks);

    thrust::device_vector<int> n_output(1, 0);
    meshing_cube<<<dimGrid, dimBlock>>>(
        valid_blocks.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
        batch_indexer.packed_accessor32<int64_t, 3, torch::RestrictPtrTraits>(),
        cube_sdf.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
        triangles.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        triangle_flatten_id.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
        n_output.data().get(), max_n_triangles, 
        n_xyz[0], n_xyz[1], n_xyz[2]
    );

    int output_n_triangles = n_output[0];
    if (output_n_triangles < max_n_triangles) {
        // Trim output tensor if it is not full.
        triangles = triangles.index({torch::indexing::Slice(torch::indexing::None, output_n_triangles)});
        triangle_flatten_id = triangle_flatten_id.index({torch::indexing::Slice(torch::indexing::None, output_n_triangles)});
    } else {
        // Otherwise spawn a warning.
        std::cerr << "Warning from marching cube: the max triangle number is too small " <<
                     output_n_triangles << " vs " << max_n_triangles << std::endl;
    }

    return {triangles, triangle_flatten_id};
}


torch::Tensor marching_cubes_dense_cuda(
    torch::Tensor valid_cords,
    torch::Tensor dense_sdf,
    torch::Tensor mask,
    int max_n_triangles
) {
    CHECK_INPUT(valid_cords);
    CHECK_INPUT(dense_sdf);
    CHECK_INPUT(mask);
    assert(max_n_triangles > 0);

    const int num_lif = valid_cords.size(0);

    torch::Tensor triangles = torch::empty({max_n_triangles, 3, 3},
                                           torch::dtype(torch::kFloat32).device(torch::kCUDA));

    dim3 dimBlock = dim3(256);
    uint xBlocks = (num_lif + dimBlock.x - 1) / dimBlock.x;
    dim3 dimGrid = dim3(xBlocks);

    thrust::device_vector<int> n_output(1, 0);
    meshing_cube_dense<<<dimGrid, dimBlock>>>(
        valid_cords.packed_accessor32<int64_t, 2, torch::RestrictPtrTraits>(),
        mask.packed_accessor32<uint8_t, 3, torch::RestrictPtrTraits>(),
        dense_sdf.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        triangles.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        n_output.data().get(), max_n_triangles
    );

    int output_n_triangles = n_output[0];
    if (output_n_triangles < max_n_triangles) {
        // Trim output tensor if it is not full.
        triangles = triangles.index({torch::indexing::Slice(torch::indexing::None, output_n_triangles)});
    } else {
        // Otherwise spawn a warning.
        std::cerr << "Warning from marching cube: the max triangle number is too small " <<
                     output_n_triangles << " vs " << max_n_triangles << std::endl;
    }

    return triangles;
}

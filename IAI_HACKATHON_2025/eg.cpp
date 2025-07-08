#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>

#define N 512
#define TILE_SIZE 16  // Kích thước tile/block

__global__ void matMulShared(const float* A, const float* B, float* C, int width) {
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    // Duyệt qua các tile theo chiều ngang
    for (int t = 0; t < width / TILE_SIZE; ++t) {
        // Nạp tile A và B vào shared memory
        tileA[threadIdx.y][threadIdx.x] = A[row * width + (t * TILE_SIZE + threadIdx.x)];
        tileB[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * width + col];

        __syncthreads();  // Đồng bộ thread trước khi tính

        for (int i = 0; i < TILE_SIZE; ++i)
            sum += tileA[threadIdx.y][i] * tileB[i][threadIdx.x];

        __syncthreads();  // Đồng bộ trước khi nạp tile tiếp theo
    }

    if (row < width && col < width)
        C[row * width + col] = sum;
}

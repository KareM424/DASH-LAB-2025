#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define N 1024
#define SIZE (N * N)
#define TILE_SIZE 16

__global__ void tiledMatmul(const float *A, const float *B, float *C) {
  __shared__ float tileA[TILE_SIZE][TILE_SIZE];
  __shared__ float tileB[TILE_SIZE][TILE_SIZE];

  int row = blockIdx.y * TILE_SIZE + threadIdx.y;
  int col = blockIdx.x * TILE_SIZE + threadIdx.x;
  float temp = 0.0f;

  for (int t = 0; t < N; t += TILE_SIZE) {
    tileA[threadIdx.y][threadIdx.x] = A[row * N + t + threadIdx.x];
    tileB[threadIdx.y][threadIdx.x] = B[(t + threadIdx.y) * N + col];
    __syncthreads();

    for (int k = 0; k < TILE_SIZE; ++k) {
      temp += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
    }
    __syncthreads();
  }

  if (row < N && col < N)
    C[row * N + col] = temp;
}

void initializeMatrix(float *matrix) {
  for (int i = 0; i < SIZE; i++) {
    matrix[i] = (float)(rand() % 10);
  }
}

int main() {
  size_t sizeBytes = SIZE * sizeof(float);

  float *h_A = (float *)malloc(sizeBytes);
  float *h_B = (float *)malloc(sizeBytes);
  float *h_C = (float *)malloc(sizeBytes);

  initializeMatrix(h_A);
  initializeMatrix(h_B);

  float *d_A, *d_B, *d_C;
  cudaMalloc(&d_A, sizeBytes);
  cudaMalloc(&d_B, sizeBytes);
  cudaMalloc(&d_C, sizeBytes);

  cudaMemcpy(d_A, h_A, sizeBytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, sizeBytes, cudaMemcpyHostToDevice);

  dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
  dim3 blocksPerGrid((N + TILE_SIZE - 1) / TILE_SIZE,
                     (N + TILE_SIZE - 1) / TILE_SIZE);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  float timeMs = 0;
  for (int i = 0; i < 10; i++) {
    cudaEventRecord(start);
    tiledMatmul<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float elapsed;
    cudaEventElapsedTime(&elapsed, start, stop);
    timeMs += elapsed;
  }
  timeMs /= 10;

  cudaMemcpy(h_C, d_C, sizeBytes, cudaMemcpyDeviceToHost);

  long long totalFlops = 2LL * N * N * N;
  float gflops = (totalFlops / (timeMs / 1000.0f)) / 1e9f;

  printf("Tiled Kernel:\n");
  printf("  Time: %.3f ms\n", timeMs);
  printf("  GFLOPS: %.2f\n", gflops);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  free(h_A);
  free(h_B);
  free(h_C);

  return 0;
}

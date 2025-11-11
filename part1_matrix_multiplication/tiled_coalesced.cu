#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define N 1024
#define SIZE (N * N)
#define TILE_SIZE 16
#define REG_BLOCK 4 // Register blocking factor

__global__ void tiledCoalescedMatmul(const float *A_T, const float *B,
                                     float *C) {
  __shared__ float tileA[TILE_SIZE][TILE_SIZE];
  __shared__ float tileB[TILE_SIZE][TILE_SIZE];

  int row = blockIdx.y * TILE_SIZE + threadIdx.y;
  int col = blockIdx.x * TILE_SIZE + threadIdx.x;

  // Register blocking: each thread computes REG_BLOCK elements
  float temp[REG_BLOCK] = {0.0f};

  for (int t = 0; t < N; t += TILE_SIZE) {
    int a_idx = t + threadIdx.x;
    if (row < N && a_idx < N)
      tileA[threadIdx.y][threadIdx.x] = A_T[a_idx * N + row];
    else
      tileA[threadIdx.y][threadIdx.x] = 0.0f;

    int b_idx = t + threadIdx.y;
    if (b_idx < N && col < N)
      tileB[threadIdx.y][threadIdx.x] = B[b_idx * N + col];
    else
      tileB[threadIdx.y][threadIdx.x] = 0.0f;

    __syncthreads();

    // Unrolled dot product with register blocking
    for (int k = 0; k < TILE_SIZE; ++k) {
      float a_val = tileA[threadIdx.y][k];
      // Cache B values in registers
      float b_vals[REG_BLOCK];
      for (int i = 0; i < REG_BLOCK; ++i) {
        b_vals[i] = tileB[k][threadIdx.x * REG_BLOCK + i];
      }
      // Compute outer product
      for (int i = 0; i < REG_BLOCK; ++i) {
        temp[i] += a_val * b_vals[i];
      }
    }
    __syncthreads();
  }

  // Write results back with register blocking
  if (row < N && col < N) {
    for (int i = 0; i < REG_BLOCK; ++i) {
      if (col + i < N) {
        C[row * N + col + i] = temp[i];
      }
    }
  }
}

void initializeMatrix(float *matrix) {
  for (int i = 0; i < SIZE; i++) {
    matrix[i] = (float)(rand() % 10);
  }
}

void transpose(float *a, float *a_t) {
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      a_t[j * N + i] = a[i * N + j];
    }
  }
}

int main() {
  size_t sizeBytes = SIZE * sizeof(float);

  float *h_A = (float *)malloc(sizeBytes);
  float *h_A_T = (float *)malloc(sizeBytes);
  float *h_B = (float *)malloc(sizeBytes);
  float *h_C = (float *)malloc(sizeBytes);

  initializeMatrix(h_A);
  initializeMatrix(h_B);
  transpose(h_A, h_A_T);

  float *d_A_T, *d_B, *d_C;
  cudaMalloc(&d_A_T, sizeBytes);
  cudaMalloc(&d_B, sizeBytes);
  cudaMalloc(&d_C, sizeBytes);

  cudaMemcpy(d_A_T, h_A_T, sizeBytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, sizeBytes, cudaMemcpyHostToDevice);

  dim3 threadsPerBlock(TILE_SIZE / REG_BLOCK, TILE_SIZE);
  dim3 blocksPerGrid((N + TILE_SIZE - 1) / TILE_SIZE,
                     (N + TILE_SIZE - 1) / TILE_SIZE);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  float timeMs = 0;
  for (int i = 0; i < 10; i++) {
    cudaEventRecord(start);
    tiledCoalescedMatmul<<<blocksPerGrid, threadsPerBlock>>>(d_A_T, d_B, d_C);
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

  printf("Tiled Coalesced Kernel (Register Blocking):\n");
  printf("  Time: %.3f ms\n", timeMs);
  printf("  GFLOPS: %.2f\n", gflops);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(d_A_T);
  cudaFree(d_B);
  cudaFree(d_C);
  free(h_A);
  free(h_A_T);
  free(h_B);
  free(h_C);

  return 0;
}

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define N 1024
#define SIZE (N * N)
#define BLOCK_SIZE 16

__global__ void naiveMatmul(const float *A, const float *B, float *C) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < N && col < N) {
    float sum = 0.0f;
    for (int k = 0; k < N; ++k) {
      sum += A[row * N + k] * B[k * N + col];
    }
    C[row * N + col] = sum;
  }
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

  dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
  dim3 blocksPerGrid((N + BLOCK_SIZE - 1) / BLOCK_SIZE,
                     (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  float timeMs = 0;
  for (int i = 0; i < 10; i++) {
    cudaEventRecord(start);
    naiveMatmul<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C);
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

  printf("Naive Kernel:\n");
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

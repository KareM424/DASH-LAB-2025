#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define N 1024
#define SIZE (N * N)
#define WARMUP_RUNS 5
#define TIMING_RUNS 10
#define BLOCK_SIZE 16
#define TILE_SIZE 16
#define SHMEM_SIZE 16 * 16
// CUDA kernel for matrix multiplication (custom)
__global__ void tiled_matrixMulKernel(float *A, float *B, float *C) {
  __shared__ float shareA[SHMEM_SIZE];
  __shared__ float shareB[SHMEM_SIZE];

  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int row = by * TILE_SIZE + ty;
  int col = bx * TILE_SIZE + tx;
  float temp = 0;

  for (int i = 0; i < N / TILE_SIZE; ++i) {
    shareA[ty * TILE_SIZE + tx] = A[row * N + (i * TILE_SIZE + tx)];
    shareB[ty * TILE_SIZE + tx] = B[(i * TILE_SIZE + ty) * N + col];
    __syncthreads();

    for (int k = 0; k < TILE_SIZE; ++k) {
      temp += shareA[(ty * TILE_SIZE) + k] * shareB[(k * TILE_SIZE) + tx];
    }
    __syncthreads();
  }
  C[row * N + col] = temp;
}

// Helper to initialize matrices
void initializeMatrix(float *matrix, int rows, int cols) {
  for (int i = 0; i < rows * cols; i++) {
    matrix[i] = (float)(rand());
  }
}

int main() {
  size_t size_A = SIZE * sizeof(float);
  size_t size_B = SIZE * sizeof(float);
  size_t size_C = SIZE * sizeof(float);

  float *h_A = (float *)malloc(size_A);
  float *h_B = (float *)malloc(size_B);
  float *h_C = (float *)malloc(size_C);
  float *h_C_cublas = (float *)malloc(size_C);

  initializeMatrix(h_A, N, N);
  initializeMatrix(h_B, N, N);

  float *d_A, *d_B, *d_C, *d_C_cublas;
  cudaMalloc(&d_A, size_A);
  cudaMalloc(&d_B, size_B);
  cudaMalloc(&d_C, size_C);
  cudaMalloc(&d_C_cublas, size_C);

  cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);

  // Custom kernel launch parameters
  dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
  dim3 blocksPerGrid((N + TILE_SIZE - 1) / TILE_SIZE,
                     (N + TILE_SIZE - 1) / TILE_SIZE);

  // cuBLAS setup
  cublasHandle_t handle;
  cublasCreate(&handle);

  float alpha = 1.0f;
  float beta = 0.0f;

  // ===== WARM-UP RUNS =====
  for (int i = 0; i < WARMUP_RUNS; i++) {
    tiled_matrixMulKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C);
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_B, N, d_A,
                N, &beta, d_C_cublas, N);
  }
  cudaDeviceSynchronize();

  // ===== TIMING CUSTOM KERNEL =====
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  float customTimeMs = 0;
  for (int i = 0; i < TIMING_RUNS; i++) {
    cudaEventRecord(start);
    tiled_matrixMulKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float elapsed;
    cudaEventElapsedTime(&elapsed, start, stop);
    customTimeMs += elapsed;
  }
  customTimeMs /= TIMING_RUNS;

  cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);

  // ===== TIMING cuBLAS =====
  float cublasTimeMs = 0;
  for (int i = 0; i < TIMING_RUNS; i++) {
    cudaEventRecord(start);
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_B, N, d_A,
                N, &beta, d_C_cublas, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float elapsed;
    cudaEventElapsedTime(&elapsed, start, stop);
    cublasTimeMs += elapsed;
  }
  cublasTimeMs /= TIMING_RUNS;

  cudaMemcpy(h_C_cublas, d_C_cublas, size_C, cudaMemcpyDeviceToHost);

  // ===== CALCULATE AND DISPLAY RESULTS =====
  long long total_flops = 2LL * N * N * N;

  float customGFLOPS = (total_flops / (customTimeMs / 1000.f)) / 1e9f;
  float cublasGFLOPS = (total_flops / (cublasTimeMs / 1000.f)) / 1e9f;

  printf("Custom Kernel Performance:\n");
  printf("  Average Time: %.3f ms\n", customTimeMs);
  printf("  GFLOPS: %.2f\n\n", customGFLOPS);

  printf("cuBLAS SGEMM Performance:\n");
  printf("  Average Time: %.3f ms\n", cublasTimeMs);
  printf("  GFLOPS: %.2f\n\n", cublasGFLOPS);

  printf("Speedup (cuBLAS over custom): %.2fx\n", customTimeMs / cublasTimeMs);

  // Clean up
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cublasDestroy(handle);
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  cudaFree(d_C_cublas);
  free(h_A);
  free(h_B);
  free(h_C);
  free(h_C_cublas);

  return 0;
}

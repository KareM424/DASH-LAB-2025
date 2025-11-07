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

// CUDA kernel for matrix multiplication (custom)
__global__ void tiled_coalasced_matrixMulKernel(float *A, float *B, float *C) {
  __shared__ float shareA[TILE_SIZE][TILE_SIZE];
  __shared__ float shareB[TILE_SIZE][TILE_SIZE];

  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int row = by * TILE_SIZE + ty;
  int col = bx * TILE_SIZE + tx;
  float temp = 0;

  for (int i = 0; i < N / TILE_SIZE; ++i) {
    shareA[ty][tx] = A[row + N * (i * TILE_SIZE + ty)];
    shareB[ty][tx] = B[(i * TILE_SIZE + tx) * N + col];
    __syncthreads();

    for (int k = 0; k < TILE_SIZE; ++k) {
      temp += shareA[ty][k] * shareB[k][tx];
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
void transpose(float *a, float *a_t) {
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      a_t[j * N + i] = a[i * N + j];
    }
  }
}
bool verifyResults(float *h_C, float *h_C_cublas, int size) {
  float maxDiff = 0.0f;
  for (int i = 0; i < size; i++) {
    float diff = fabs(h_C[i] - h_C_cublas[i]);
    if (diff > maxDiff)
      maxDiff = diff;
    if (diff > 1e-2) { // Allow small numerical errors
      printf("Mismatch at index %d: custom = %.2f, cublas = %.2f\n", i, h_C[i],
             h_C_cublas[i]);
      return false;
    }
  }
  printf("Results verified! Max difference: %.6f\n\n", maxDiff);
  return true;
}
int main() {
  size_t size_A = SIZE * sizeof(float);
  size_t size_B = SIZE * sizeof(float);
  size_t size_C = SIZE * sizeof(float);
  size_t size_A_T = SIZE * sizeof(float);

  float *h_A = (float *)malloc(size_A);
  float *h_A_T = (float *)malloc(size_A_T);
  float *h_B = (float *)malloc(size_B);
  float *h_C = (float *)malloc(size_C);
  float *h_C_cublas = (float *)malloc(size_C);

  initializeMatrix(h_A, N, N);
  initializeMatrix(h_B, N, N);
  transpose(h_A, h_A_T);

  float *d_A, *d_A_T, *d_B, *d_C, *d_C_cublas;
  cudaMalloc(&d_A, size_A);
  cudaMalloc(&d_A_T, size_A_T);
  cudaMalloc(&d_B, size_B);
  cudaMalloc(&d_C, size_C);
  cudaMalloc(&d_C_cublas, size_C);

  cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
  cudaMemcpy(d_A_T, h_A_T, size_A, cudaMemcpyHostToDevice);
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
    tiled_coalasced_matrixMulKernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_A_T, d_B, d_C);
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
    tiled_coalasced_matrixMulKernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_A_T, d_B, d_C);
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
  verifyResults(h_C, h_C_cublas, SIZE);
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

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define N 1024
#define SIZE (N * N)
#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))

#define BM 64
#define BN 64
#define BLOCK_K 8
#define TM 8

__global__ void sgemm_warp_tiling(const float *A, const float *B, float *C) {
  const int K = N;
  const uint cRow = blockIdx.y;
  const uint cCol = blockIdx.x;
  const int threadCol = threadIdx.x % BN;
  const int threadRow = threadIdx.x / BN;

  __shared__ float As[BM * BLOCK_K];
  __shared__ float Bs[BLOCK_K * BN];

  const float *A_base = A + cRow * BM * K;
  const float *B_base = B + cCol * BN;
  float *C_base = C + cRow * BM * N + cCol * BN;

  const uint innerColA = threadIdx.x % BLOCK_K;
  const uint innerRowA = threadIdx.x / BLOCK_K;
  const uint innerColB = threadIdx.x % BN;
  const uint innerRowB = threadIdx.x / BN;

  float threadResults[TM] = {0.0f};

  for (uint k = 0; k < K; k += BLOCK_K) {
    As[innerRowA * BLOCK_K + innerColA] = A_base[innerRowA * K + innerColA];
    Bs[innerRowB * BN + innerColB] = B_base[innerRowB * N + innerColB];
    __syncthreads();

    A_base += BLOCK_K;
    B_base += BLOCK_K * N;

    for (uint dotIdx = 0; dotIdx < BLOCK_K; ++dotIdx) {
      float tmp = Bs[dotIdx * BN + threadCol];
      for (uint resIdx = 0; resIdx < TM; ++resIdx) {
        threadResults[resIdx] +=
            As[(threadRow * TM + resIdx) * BLOCK_K + dotIdx] * tmp;
      }
    }
    __syncthreads();
  }

  for (uint resIdx = 0; resIdx < TM; ++resIdx) {
    C_base[(threadRow * TM + resIdx) * N + threadCol] = threadResults[resIdx];
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

  dim3 threadsPerBlock((BM * BN) / TM);
  dim3 blocksPerGrid(CEIL_DIV(N, BN), CEIL_DIV(N, BM));

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  float timeMs = 0;
  for (int i = 0; i < 10; i++) {
    cudaEventRecord(start);
    sgemm_warp_tiling<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C);
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

  printf("Advanced Warp Tiling (1D):\n");
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

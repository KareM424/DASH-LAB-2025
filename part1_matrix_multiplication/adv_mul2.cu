#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define N 1024
#define SIZE (N * N)
#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))

#define BX 128
#define BY 128
#define BLOCK_K 8
#define TM 8
#define TN 8

#define TOTAL_THREADS ((BX * BY) / (TM * TN))
#define TX (BX / TN)
#define TY (BY / TM)

__global__ void sgemm2DWarpTiling(const float *A, const float *B, float *C) {
  const int K = N;
  const uint blockRow = blockIdx.y;
  const uint blockCol = blockIdx.x;

  const int threadCol = threadIdx.x % TX;
  const int threadRow = threadIdx.x / TX;

  __shared__ float As[BY * BLOCK_K];
  __shared__ float Bs[BLOCK_K * BX];

  A += blockRow * BY * K;
  B += blockCol * BX;
  C += blockRow * BY * N + blockCol * BX;

  const uint innerRowA = threadIdx.x / BLOCK_K;
  const uint innerColA = threadIdx.x % BLOCK_K;
  const uint strideA = TOTAL_THREADS / BLOCK_K;

  const uint innerRowB = threadIdx.x / BX;
  const uint innerColB = threadIdx.x % BX;
  const uint strideB = TOTAL_THREADS / BX;

  float threadResults[TM * TN] = {0.0f};
  float regM[TM] = {0.0f};
  float regN[TN] = {0.0f};

  for (uint k = 0; k < K; k += BLOCK_K) {
    for (uint loadOffset = 0; loadOffset < BY; loadOffset += strideA) {
      if (innerRowA + loadOffset < BY && innerColA < BLOCK_K) {
        As[(innerRowA + loadOffset) * BLOCK_K + innerColA] =
            A[(innerRowA + loadOffset) * K + innerColA];
      }
    }

    for (uint loadOffset = 0; loadOffset < BLOCK_K; loadOffset += strideB) {
      if (innerRowB + loadOffset < BLOCK_K && innerColB < BX) {
        Bs[(innerRowB + loadOffset) * BX + innerColB] =
            B[(innerRowB + loadOffset) * N + innerColB];
      }
    }
    __syncthreads();

    A += BLOCK_K;
    B += BLOCK_K * N;

    for (uint dotIdx = 0; dotIdx < BLOCK_K; ++dotIdx) {
      for (uint i = 0; i < TM; ++i) {
        regM[i] = As[(threadRow * TM + i) * BLOCK_K + dotIdx];
      }

      for (uint i = 0; i < TN; ++i) {
        regN[i] = Bs[dotIdx * BX + threadCol * TN + i];
      }

      for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
        for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
          threadResults[resIdxM * TN + resIdxN] +=
              regM[resIdxM] * regN[resIdxN];
        }
      }
    }
    __syncthreads();
  }

  for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
    for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
      int row = blockRow * BY + threadRow * TM + resIdxM;
      int col = blockCol * BX + threadCol * TN + resIdxN;
      if (row < N && col < N) {
        C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN] =
            threadResults[resIdxM * TN + resIdxN];
      }
    }
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

  dim3 threadsPerBlock(TOTAL_THREADS);
  dim3 blocksPerGrid(CEIL_DIV(N, BX), CEIL_DIV(N, BY));

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  float timeMs = 0;
  for (int i = 0; i < 10; i++) {
    cudaEventRecord(start);
    sgemm2DWarpTiling<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C);
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

  printf("Advanced Warp Tiling (2D Register Blocking):\n");
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

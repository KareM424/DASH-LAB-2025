#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define N 1024
#define SIZE (N * N)
#define TIMING_RUNS 10
#define SEED 42
#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))

// ============================================================================
// KERNEL 1: Naive Matrix Multiplication
// ============================================================================
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

// ============================================================================
// KERNEL 2: Basic Tiled Matrix Multiplication
// ============================================================================
#define TILE_SIZE 16

__global__ void tiledMatmul(const float *A, const float *B, float *C) {
  __shared__ float tileA[TILE_SIZE][TILE_SIZE];
  __shared__ float tileB[TILE_SIZE][TILE_SIZE];

  int row = blockIdx.y * TILE_SIZE + threadIdx.y;
  int col = blockIdx.x * TILE_SIZE + threadIdx.x;
  float temp = 0.0f;

  for (int t = 0; t < N; t += TILE_SIZE) {
    if (row < N && t + threadIdx.x < N)
      tileA[threadIdx.y][threadIdx.x] = A[row * N + t + threadIdx.x];
    else
      tileA[threadIdx.y][threadIdx.x] = 0.0f;

    if (col < N && t + threadIdx.y < N)
      tileB[threadIdx.y][threadIdx.x] = B[(t + threadIdx.y) * N + col];
    else
      tileB[threadIdx.y][threadIdx.x] = 0.0f;

    __syncthreads();

    for (int k = 0; k < TILE_SIZE; ++k) {
      temp += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
    }
    __syncthreads();
  }

  if (row < N && col < N)
    C[row * N + col] = temp;
}

// ============================================================================
// KERNEL 3: Tiled with Coalescing and Register Blocking
// ============================================================================
#define REG_BLOCK 4

__global__ void tiledCoalescedMatmul(const float *A_T, const float *B,
                                     float *C) {
  __shared__ float tileA[TILE_SIZE][TILE_SIZE];
  __shared__ float tileB[TILE_SIZE][TILE_SIZE];

  int row = blockIdx.y * TILE_SIZE + threadIdx.y;
  int col = blockIdx.x * TILE_SIZE + threadIdx.x;
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

    for (int k = 0; k < TILE_SIZE; ++k) {
      float a_val = tileA[threadIdx.y][k];
      for (int i = 0; i < REG_BLOCK; ++i) {
        temp[i] += a_val * tileB[k][threadIdx.x * REG_BLOCK + i];
      }
    }
    __syncthreads();
  }

  if (row < N) {
    for (int i = 0; i < REG_BLOCK; ++i) {
      if (col + i < N) {
        C[row * N + col + i] = temp[i];
      }
    }
  }
}

// ============================================================================
// KERNEL 4: Advanced Warp Tiling (1D)
// ============================================================================
#define BM 64
#define BN 64
#define BLOCK_K 8
#define TM 8

__global__ void advancedWarpTiling1D(const float *A, const float *B, float *C) {

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
// ============================================================================
// KERNEL 5: Advanced Warp Tiling (2D Register Blocking) -
// ============================================================================
#define BX 128
#define BY 128
#define BLOCK_K_2D 8
#define TM_2D 8
#define TN_2D 8
#define TOTAL_THREADS_2D ((BX * BY) / (TM_2D * TN_2D))
#define TX_2D (BX / TN_2D)
#define TY_2D (BY / TM_2D)

__global__ void advancedWarpTiling2D(const float *A, const float *B, float *C) {
  const int K = N;
  const uint blockRow = blockIdx.y;
  const uint blockCol = blockIdx.x;

  const int threadCol = threadIdx.x % TX_2D;
  const int threadRow = threadIdx.x / TX_2D;

  __shared__ float As[BY * BLOCK_K_2D];
  __shared__ float Bs[BLOCK_K_2D * BX];

  A += blockRow * BY * K;
  B += blockCol * BX;
  C += blockRow * BY * N + blockCol * BX;

  const uint innerRowA = threadIdx.x / BLOCK_K_2D;
  const uint innerColA = threadIdx.x % BLOCK_K_2D;
  const uint strideA = TOTAL_THREADS_2D / BLOCK_K_2D;

  const uint innerRowB = threadIdx.x / BX;
  const uint innerColB = threadIdx.x % BX;
  const uint strideB = TOTAL_THREADS_2D / BX;

  float threadResults[TM_2D * TN_2D] = {0.0f};
  float regM[TM_2D] = {0.0f};
  float regN[TN_2D] = {0.0f};

  for (uint k = 0; k < K; k += BLOCK_K_2D) {
    for (uint loadOffset = 0; loadOffset < BY; loadOffset += strideA) {
      if (innerRowA + loadOffset < BY && innerColA < BLOCK_K_2D) {
        As[(innerRowA + loadOffset) * BLOCK_K_2D + innerColA] =
            A[(innerRowA + loadOffset) * K + innerColA];
      }
    }

    for (uint loadOffset = 0; loadOffset < BLOCK_K_2D; loadOffset += strideB) {
      if (innerRowB + loadOffset < BLOCK_K_2D && innerColB < BX) {
        Bs[(innerRowB + loadOffset) * BX + innerColB] =
            B[(innerRowB + loadOffset) * N + innerColB];
      }
    }
    __syncthreads();

    A += BLOCK_K_2D;
    B += BLOCK_K_2D * N;

    for (uint dotIdx = 0; dotIdx < BLOCK_K_2D; ++dotIdx) {
      for (uint i = 0; i < TM_2D; ++i) {
        regM[i] = As[(threadRow * TM_2D + i) * BLOCK_K_2D + dotIdx];
      }

      for (uint i = 0; i < TN_2D; ++i) {
        regN[i] = Bs[dotIdx * BX + threadCol * TN_2D + i];
      }

      for (uint resIdxM = 0; resIdxM < TM_2D; ++resIdxM) {
        for (uint resIdxN = 0; resIdxN < TN_2D; ++resIdxN) {
          threadResults[resIdxM * TN_2D + resIdxN] +=
              regM[resIdxM] * regN[resIdxN];
        }
      }
    }
    __syncthreads();
  }

  for (uint resIdxM = 0; resIdxM < TM_2D; ++resIdxM) {
    for (uint resIdxN = 0; resIdxN < TN_2D; ++resIdxN) {
      int row = blockRow * BY + threadRow * TM_2D + resIdxM;
      int col = blockCol * BX + threadCol * TN_2D + resIdxN;
      if (row < N && col < N) {
        C[row * N + col] = threadResults[resIdxM * TN_2D + resIdxN];
      }
    }
  }
}

// ============================================================================
// Utility Functions
// ============================================================================
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

// ============================================================================
// Main Benchmark Suite
// ============================================================================
int main() {
  srand(SEED);

  printf("\n");
  printf("+===========================================================+\n");
  printf("|  CUDA Matrix Multiplication Benchmark Suite             |\n");
  printf("|  5 Implementations Compared                             |\n");
  printf("|                                                           |\n");
  printf("|  Matrix Size: %d x %d                                  |\n", N, N);
  printf("|  Timing Runs: %d (fixed seed: %d)                     |\n",
         TIMING_RUNS, SEED);
  printf("+===========================================================+\n");
  printf("\n");

  size_t sizeBytes = SIZE * sizeof(float);

  float *h_A = (float *)malloc(sizeBytes);
  float *h_B = (float *)malloc(sizeBytes);
  float *h_A_T = (float *)malloc(sizeBytes);

  initializeMatrix(h_A);
  initializeMatrix(h_B);
  transpose(h_A, h_A_T);

  float *d_A, *d_B, *d_A_T, *d_C;
  cudaMalloc(&d_A, sizeBytes);
  cudaMalloc(&d_B, sizeBytes);
  cudaMalloc(&d_A_T, sizeBytes);
  cudaMalloc(&d_C, sizeBytes);

  cudaMemcpy(d_A, h_A, sizeBytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, sizeBytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_A_T, h_A_T, sizeBytes, cudaMemcpyHostToDevice);

  cublasHandle_t handle;
  cublasCreate(&handle);

  float alpha = 1.0f, beta = 0.0f;

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  long long totalFlops = 2LL * N * N * N;

  // BENCHMARK 1: Naive
  printf("1. NAIVE KERNEL\n");
  dim3 naiveGrid(CEIL_DIV(N, 16), CEIL_DIV(N, 16));
  dim3 naiveBlock(16, 16);

  float naiveTimeMs = 0;
  for (int i = 0; i < TIMING_RUNS; i++) {
    cudaEventRecord(start);
    naiveMatmul<<<naiveGrid, naiveBlock>>>(d_A, d_B, d_C);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float elapsed;
    cudaEventElapsedTime(&elapsed, start, stop);
    naiveTimeMs += elapsed;
  }
  naiveTimeMs /= TIMING_RUNS;

  float naiveGFLOPS = (totalFlops / (naiveTimeMs / 1000.0f)) / 1e9f;
  printf("   Time: %.3f ms, GFLOPS: %.2f\n\n", naiveTimeMs, naiveGFLOPS);

  // BENCHMARK 2: Tiled
  printf("2. TILED KERNEL\n");
  dim3 tiledGrid(CEIL_DIV(N, TILE_SIZE), CEIL_DIV(N, TILE_SIZE));
  dim3 tiledBlock(TILE_SIZE, TILE_SIZE);

  float tiledTimeMs = 0;
  for (int i = 0; i < TIMING_RUNS; i++) {
    cudaEventRecord(start);
    tiledMatmul<<<tiledGrid, tiledBlock>>>(d_A, d_B, d_C);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float elapsed;
    cudaEventElapsedTime(&elapsed, start, stop);
    tiledTimeMs += elapsed;
  }
  tiledTimeMs /= TIMING_RUNS;

  float tiledGFLOPS = (totalFlops / (tiledTimeMs / 1000.0f)) / 1e9f;
  printf("   Time: %.3f ms, GFLOPS: %.2f\n\n", tiledTimeMs, tiledGFLOPS);

  // BENCHMARK 3: Tiled Coalesced
  printf("3. TILED COALESCED (REGISTER BLOCKING)\n");
  dim3 coalGrid(CEIL_DIV(N, TILE_SIZE), CEIL_DIV(N, TILE_SIZE));
  dim3 coalBlock(TILE_SIZE / REG_BLOCK, TILE_SIZE);

  float coalTimeMs = 0;
  for (int i = 0; i < TIMING_RUNS; i++) {
    cudaEventRecord(start);
    tiledCoalescedMatmul<<<coalGrid, coalBlock>>>(d_A_T, d_B, d_C);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float elapsed;
    cudaEventElapsedTime(&elapsed, start, stop);
    coalTimeMs += elapsed;
  }
  coalTimeMs /= TIMING_RUNS;

  float coalGFLOPS = (totalFlops / (coalTimeMs / 1000.0f)) / 1e9f;
  printf("   Time: %.3f ms, GFLOPS: %.2f\n\n", coalTimeMs, coalGFLOPS);

  // BENCHMARK 4: Advanced Warp (1D)
  printf("4. ADVANCED WARP TILING (1D THREADS)\n");
  dim3 adv1Grid(CEIL_DIV(N, 64), CEIL_DIV(N, 64));
  dim3 adv1Block(64, 8);
  printf("   Grid: (%d, %d), Block: (%d, %d)\n", adv1Grid.x, adv1Grid.y,
         adv1Block.x, adv1Block.y);

  for (int i = 0; i < 5; i++)
    advancedWarpTiling1D<<<adv1Grid, adv1Block>>>(d_A, d_B, d_C);
  cudaDeviceSynchronize();

  float adv1TimeMs = 0;
  for (int i = 0; i < TIMING_RUNS; i++) {
    cudaEventRecord(start);
    advancedWarpTiling1D<<<adv1Grid, adv1Block>>>(d_A, d_B, d_C);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float elapsed;
    cudaEventElapsedTime(&elapsed, start, stop);
    adv1TimeMs += elapsed;
  }
  adv1TimeMs /= TIMING_RUNS;

  float adv1GFLOPS = (totalFlops / (adv1TimeMs / 1000.0f)) / 1e9f;
  printf("   Time: %.3f ms, GFLOPS: %.2f\n\n", adv1TimeMs, adv1GFLOPS);

  // BENCHMARK 5: Advanced Warp (2D)
  printf("5. ADVANCED WARP TILING (2D REGISTER BLOCKING)\n");
  dim3 adv2Grid(CEIL_DIV(N, BX), CEIL_DIV(N, BY));
  dim3 adv2Block(TOTAL_THREADS_2D);

  float adv2TimeMs = 0;
  for (int i = 0; i < TIMING_RUNS; i++) {
    cudaEventRecord(start);
    advancedWarpTiling2D<<<adv2Grid, adv2Block>>>(d_A, d_B, d_C);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float elapsed;
    cudaEventElapsedTime(&elapsed, start, stop);
    adv2TimeMs += elapsed;
  }
  adv2TimeMs /= TIMING_RUNS;

  float adv2GFLOPS = (totalFlops / (adv2TimeMs / 1000.0f)) / 1e9f;
  printf("   Time: %.3f ms, GFLOPS: %.2f\n\n", adv2TimeMs, adv2GFLOPS);

  // BENCHMARK 6: cuBLAS
  printf("6. cuBLAS REFERENCE\n");
  for (int i = 0; i < 5; i++)
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_B, N, d_A,
                N, &beta, d_C, N);
  cudaDeviceSynchronize();

  float cublasTimeMs = 0;
  for (int i = 0; i < TIMING_RUNS; i++) {
    cudaEventRecord(start);
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_B, N, d_A,
                N, &beta, d_C, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float elapsed;
    cudaEventElapsedTime(&elapsed, start, stop);
    cublasTimeMs += elapsed;
  }
  cublasTimeMs /= TIMING_RUNS;

  float cublasGFLOPS = (totalFlops / (cublasTimeMs / 1000.0f)) / 1e9f;
  printf("   Time: %.3f ms, GFLOPS: %.2f\n\n", cublasTimeMs, cublasGFLOPS);

  // SUMMARY TABLE
  printf("\n+============================================================+\n");
  printf("|                    BENCHMARK SUMMARY                     |\n");
  printf("+============================================================+\n");
  printf("| %-38s | Time(ms) | GFLOPS | Eff.   |\n", "Kernel");
  printf("+============================================================+\n");
  printf("| %-38s | %7.3f | %6.2f | %5.1f%% |\n", "1. Naive", naiveTimeMs,
         naiveGFLOPS, (naiveGFLOPS / cublasGFLOPS) * 100);
  printf("| %-38s | %7.3f | %6.2f | %5.1f%% |\n", "2. Tiled", tiledTimeMs,
         tiledGFLOPS, (tiledGFLOPS / cublasGFLOPS) * 100);
  printf("| %-38s | %7.3f | %6.2f | %5.1f%% |\n",
         "3. Tiled Coalesced (Reg Block)", coalTimeMs, coalGFLOPS,
         (coalGFLOPS / cublasGFLOPS) * 100);
  printf("| %-38s | %7.3f | %6.2f | %5.1f%% |\n", "4. Advanced Warp (1D)",
         adv1TimeMs, adv1GFLOPS, (adv1GFLOPS / cublasGFLOPS) * 100);
  printf("| %-38s | %7.3f | %6.2f | %5.1f%% |\n", "5. Advanced Warp (2D)",
         adv2TimeMs, adv2GFLOPS, (adv2GFLOPS / cublasGFLOPS) * 100);
  printf("| %-38s | %7.3f | %6.2f | 100.0%% |\n", "Reference: cuBLAS",
         cublasTimeMs, cublasGFLOPS);
  printf("+============================================================+\n");
  printf("\n");

  // SPEEDUP ANALYSIS
  printf("+============================================================+\n");
  printf("|                    SPEEDUP ANALYSIS                      |\n");
  printf("+============================================================+\n");
  printf("| Improvement vs Naive:                                    |\n");
  printf("|  - Tiled:                 %.2fx                          |\n",
         naiveTimeMs / tiledTimeMs);
  printf("|  - Tiled Coalesced (Reg): %.2fx                          |\n",
         naiveTimeMs / coalTimeMs);
  printf("|  - Advanced Warp (1D):    %.2fx                          |\n",
         naiveTimeMs / adv1TimeMs);
  printf("|  - Advanced Warp (2D):    %.2fx                          |\n",
         naiveTimeMs / adv2TimeMs);
  printf("|  - cuBLAS:                %.2fx                          |\n",
         naiveTimeMs / cublasTimeMs);
  printf("|                                                            |\n");
  printf("| Advanced Warp (2D) vs (1D):  %.2fx                       |\n",
         adv1TimeMs / adv2TimeMs);
  printf("+============================================================+\n");
  printf("\n");

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cublasDestroy(handle);
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_A_T);
  cudaFree(d_C);
  free(h_A);
  free(h_B);
  free(h_A_T);

  return 0;
}

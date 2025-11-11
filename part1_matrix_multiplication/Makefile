# Makefile for CUDA Matrix Multiplication Benchmarks

NVCC = nvcc
NVCC_FLAGS = -O3
CUBLAS_FLAGS = -lcublas
LIBS = $(CUBLAS_FLAGS)

# Targets
TARGETS = naive tiled tiled_coalesced adv_mul1 adv_mul2 benchmark

all: $(TARGETS)

# Naive kernel
naive: naive.cu
	$(NVCC) $(NVCC_FLAGS) -o naive naive.cu $(LIBS)

# Basic tiled kernel
tiled: tiled.cu
	$(NVCC) $(NVCC_FLAGS) -o tiled tiled.cu $(LIBS)

# Tiled with coalescing
tiled_coalesced: tiled_coalesced.cu
	$(NVCC) $(NVCC_FLAGS) -o tiled_coalesced tiled_coalesced.cu $(LIBS)

# Advanced warp tiling (1D threads)
adv_mul1: adv_mul1.cu
	$(NVCC) $(NVCC_FLAGS) -o adv_mul1 adv_mul1.cu $(LIBS)

# Advanced warp tiling (2D register blocking)
adv_mul2: adv_mul2.cu
	$(NVCC) $(NVCC_FLAGS) -o adv_mul2 adv_mul2.cu $(LIBS)

# Benchmark suite
benchmark: benchmark.cu
	$(NVCC) $(NVCC_FLAGS) -o benchmark benchmark.cu $(LIBS)

# Run all benchmarks
run: benchmark
	@echo "========================================"
	@echo "Running All Benchmarks"
	@echo "========================================"
	./benchmark

# Run individual benchmarks
run-naive: naive
	./naive

run-tiled: tiled
	./tiled

run-tiled-coalesced: tiled_coalesced
	./tiled_coalesced

run-adv1: adv_mul1
	./adv_mul1

run-adv2: adv_mul2
	./adv_mul2

# Clean build artifacts
clean:
	rm -f $(TARGETS)
	rm -f *.o

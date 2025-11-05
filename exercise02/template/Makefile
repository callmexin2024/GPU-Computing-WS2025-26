CUDA_ROOT	=$(CUDA_HOME)
INC			=-I./inc -I$(CUDA_ROOT)/include
LIB			=-L$(CUDA_ROOT)/lib64
NVCC		=nvcc
NVCC_FLAGS	=-O2 --compiler-options "-O2 -Wall -Wextra"

.PHONY: build clean rebuild

build: ./bin/2.1_nullKernelAsync ./bin/2.1_kernel_matrix ./bin/2.2_kernel_busy_wait ./bin/2.3_memcpy_bandwidth

./bin/2.1_nullKernelAsync: ./src/2.1_nullKernelAsync.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $^ $(INC) $(LIB)

./bin/2.1_kernel_matrix: ./src/2.1_kernel_matrix.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $^ $(INC) $(LIB)

./bin/2.2_kernel_busy_wait: ./src/2.2_kernel_busy_wait.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $^ $(INC) $(LIB)

./bin/2.3_memcpy_bandwidth: ./src/2.3_memcpy_bandwidth.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $^ $(INC) $(LIB)

clean:
	rm ./bin/*

rebuild: clean build

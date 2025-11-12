NVCC=nvcc
CUDA_ROOT=$(CUDA_HOME)
INC=-I./inc -I$(CUDA_ROOT)/include
LIB=-L$(CUDA_ROOT)/lib64

.PHONY: build
build: ./bin/3.2.1 ./bin/3.1_memcpy_bandwidth

.PHONY: clean
clean:
	rm ./bin/*
	
.PHONY: rebuild
rebuild: clean build

./bin/3.2.1: ./src/3.2.1.cpp ./src/kernel.cu
	$(NVCC) -O2 --compiler-options "-O2 -Wall -Wextra" -o $@ $^ $(INC) $(LIB)

./bin/3.1_memcpy_bandwidth: ./src/3.1_memcpy_bandwidth.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $^ $(INC) $(LIB)
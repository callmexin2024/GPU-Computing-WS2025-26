#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

double benchmarkH2D(void* h_data, void* d_data, size_t size, int iterations) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; i++)
        CUDA_CHECK(cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return ms / iterations;
}

double benchmarkD2H(void* h_data, void* d_data, size_t size, int iterations) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; i++)
        CUDA_CHECK(cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return ms / iterations;
}

void runBenchmark(size_t size, int iterations, FILE* csv) {
    void* h_pageable = malloc(size);
    if (!h_pageable) { fprintf(stderr, "Failed to allocate pageable memory\n"); exit(EXIT_FAILURE); }

    void* h_pinned;
    CUDA_CHECK(cudaMallocHost(&h_pinned, size));

    void* d_data;
    CUDA_CHECK(cudaMalloc(&d_data, size));

    memset(h_pageable, 0xAB, size);
    memset(h_pinned, 0xAB, size);

    double h2d_pageable = benchmarkH2D(h_pageable, d_data, size, iterations);
    double h2d_pinned = benchmarkH2D(h_pinned, d_data, size, iterations);
    double d2h_pageable = benchmarkD2H(h_pageable, d_data, size, iterations);
    double d2h_pinned = benchmarkD2H(h_pinned, d_data, size, iterations);

    double bw_h2d_pageable = (size / (1024.0*1024.0*1024.0)) / (h2d_pageable / 1000.0);
    double bw_h2d_pinned   = (size / (1024.0*1024.0*1024.0)) / (h2d_pinned / 1000.0);
    double bw_d2h_pageable = (size / (1024.0*1024.0*1024.0)) / (d2h_pageable / 1000.0);
    double bw_d2h_pinned   = (size / (1024.0*1024.0*1024.0)) / (d2h_pinned / 1000.0);

    double size_mb = size / (1024.0*1024.0);
    printf("%10.3f MB | H2D: %8.3f GB/s (pageable) | %8.3f GB/s (pinned) | "
           "D2H: %8.3f GB/s (pageable) | %8.3f GB/s (pinned)\n",
           size_mb, bw_h2d_pageable, bw_h2d_pinned, bw_d2h_pageable, bw_d2h_pinned);

    if (csv)
        fprintf(csv, "%f,%f,%f,%f,%f\n", size_mb, bw_h2d_pageable, bw_h2d_pinned,
                                         bw_d2h_pageable, bw_d2h_pinned);

    free(h_pageable);
    CUDA_CHECK(cudaFreeHost(h_pinned));
    CUDA_CHECK(cudaFree(d_data));
}

int main() {
    FILE* csv = fopen("cuda_transfer.csv", "w");
    if (!csv) { fprintf(stderr, "Failed to open CSV file\n"); return EXIT_FAILURE; }
    fprintf(csv, "Size_MB,H2D_Pageable,H2D_Pinned,D2H_Pageable,D2H_Pinned\n");

    printf("Transfer sizes from 1 KB to 1 GB\n");
    printf("----------------------------------------------------------------------------------------------------\n");
    printf("       Size   |                  Host-to-Device (H2D)                  |                  Device-to-Host (D2H)\n");
    printf("----------------------------------------------------------------------------------------------------\n");

    size_t sizes[] = {1024, 4*1024, 16*1024, 64*1024, 256*1024, 1024*1024, 4*1024*1024,
                      16*1024*1024, 64*1024*1024, 256*1024*1024, 512*1024*1024, 1024*1024*1024};
    int num_sizes = sizeof(sizes)/sizeof(sizes[0]);

    for (int i=0;i<num_sizes;i++){
        int iter = sizes[i]<1024*1024 ? 1000 : sizes[i]<64*1024*1024 ? 100 : 20;
        runBenchmark(sizes[i], iter, csv);
    }

    fclose(csv);
    return 0;
}

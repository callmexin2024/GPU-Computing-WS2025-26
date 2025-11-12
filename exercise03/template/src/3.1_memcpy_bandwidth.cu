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
    } while (0)

double benchmarkMemcpy(void* dst, void* src, size_t size, cudaMemcpyKind kind, int iterations, bool useAsync, bool isPinned)
{
    cudaEvent_t start, stop;
    cudaStream_t stream;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaStreamCreate(&stream));
    CUDA_CHECK(cudaMemcpy(dst, src, size, kind));
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; i++) {
        if (useAsync && isPinned)
            CUDA_CHECK(cudaMemcpyAsync(dst, src, size, kind, stream));
        else
            CUDA_CHECK(cudaMemcpy(dst, src, size, kind));
    }
    if (useAsync && isPinned)
        CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaStreamDestroy(stream));
    return ms / iterations;
}

void runBenchmark(size_t size, int iterations, FILE* csv)
{
    void* h_pageable;
    if (posix_memalign(&h_pageable, 4096, size))
	{
        fprintf(stderr, "Failed to allocate aligned pageable memory\n");
        exit(EXIT_FAILURE);
    }
    void* h_pinned;
    CUDA_CHECK(cudaMallocHost(&h_pinned, size));
    void* d_data1;
    void* d_data2;
    CUDA_CHECK(cudaMalloc(&d_data1, size));
    CUDA_CHECK(cudaMalloc(&d_data2, size));
    memset(h_pageable, 0xAB, size);
    memset(h_pinned, 0xAB, size);
    CUDA_CHECK(cudaMemcpy(d_data1, h_pinned, size, cudaMemcpyHostToDevice));
    double h2d_pageable = benchmarkMemcpy(d_data1, h_pageable, size, cudaMemcpyHostToDevice, iterations, true, false);
    double h2d_pinned   = benchmarkMemcpy(d_data1, h_pinned,   size, cudaMemcpyHostToDevice, iterations, true, true);
    double d2h_pageable = benchmarkMemcpy(h_pageable, d_data1, size, cudaMemcpyDeviceToHost, iterations, true, false);
    double d2h_pinned   = benchmarkMemcpy(h_pinned,   d_data1, size, cudaMemcpyDeviceToHost, iterations, true, true);
    double d2d_time     = benchmarkMemcpy(d_data2,    d_data1, size, cudaMemcpyDeviceToDevice, iterations, true, true);
    #define BYTES_TO_GB(bytes) ((bytes) / (1024.0 * 1024.0 * 1024.0))
    #define MS_TO_SEC(ms) ((ms) / 1000.0)
    #define CALC_BW(size, time_ms) (BYTES_TO_GB(size) / MS_TO_SEC(time_ms))
    double bw_h2d_pageable = CALC_BW(size, h2d_pageable);
    double bw_h2d_pinned   = CALC_BW(size, h2d_pinned);
    double bw_d2h_pageable = CALC_BW(size, d2h_pageable);
    double bw_d2h_pinned   = CALC_BW(size, d2h_pinned);
    double bw_d2d          = CALC_BW(size, d2d_time);
    double size_mb = size / (1024.0 * 1024.0);
    printf("%10.3f MB | H2D: %8.3f GB/s (pageable) | %8.3f GB/s (pinned) | D2H: %8.3f GB/s (pageable) | %8.3f GB/s (pinned) | D2D: %8.3f GB/s\n",
           size_mb, bw_h2d_pageable, bw_h2d_pinned, bw_d2h_pageable, bw_d2h_pinned, bw_d2d);
    if (csv)
        fprintf(csv, "%.3f,%.6f,%.6f,%.6f,%.6f,%.6f\n", size_mb,
                bw_h2d_pageable, bw_h2d_pinned, bw_d2h_pageable, bw_d2h_pinned, bw_d2d);
    free(h_pageable);
    CUDA_CHECK(cudaFreeHost(h_pinned));
    CUDA_CHECK(cudaFree(d_data1));
    CUDA_CHECK(cudaFree(d_data2));
}

int main()
{
    CUDA_CHECK(cudaSetDevice(0));
    FILE* csv = fopen("cuda_transfer.csv", "w");
    if (!csv)
	{
        fprintf(stderr, "Failed to open CSV file\n");
        return EXIT_FAILURE;
    }
    fprintf(csv, "Size_MB,H2D_Pageable,H2D_Pinned,D2H_Pageable,D2H_Pinned,D2D\n");
    printf("Transfer sizes from 1 KB to 1 GB\n");
    printf("-------------------------------------------------------------------------------------------------------------------------------------------------------------\n");
    printf("       Size   |                  Host-to-Device (H2D)                  |                  Device-to-Host (D2H)                  | Device-to-Device (D2D)\n");
    printf("-------------------------------------------------------------------------------------------------------------------------------------------------------------\n");
    size_t sizes[] =
	{
        1 * 1024, 2 * 1024, 4 * 1024, 8 * 1024, 16 * 1024, 32 * 1024, 64 * 1024, 128 * 1024,
        256 * 1024, 512 * 1024,
        1 * 1024 * 1024, 2 * 1024 * 1024, (size_t)(2.25 * 1024 * 1024), (size_t)(2.5 * 1024 * 1024), (size_t)(2.75 * 1024 * 1024), 3 * 1024 * 1024,
        4 * 1024 * 1024, 16 * 1024 * 1024, 64 * 1024 * 1024, 256 * 1024 * 1024, 1024 * 1024 * 1024
    };
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    for (int i = 0; i < num_sizes; i++)
	{
        int iter;
        if (sizes[i] < 1 * 1024 * 1024)
            iter = 20000;
        else if (sizes[i] < 64 * 1024 * 1024)
            iter = 2000;
        else if (sizes[i] < 256 * 1024 * 1024)
            iter = 200;
        else
            iter = 50;
        runBenchmark(sizes[i], iter, csv);
    }
    fclose(csv);
    return 0;
}

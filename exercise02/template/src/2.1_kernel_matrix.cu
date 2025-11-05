#include <stdio.h>
#include <stdlib.h>
#include "chTimer.h"
#include <math.h>
#include <cuda_runtime.h>

__global__ void NullKernel()
{

}

typedef struct
{
    int num_blocks;
    int threads_per_block;
    float async_time_us;
    float sync_time_us;
} t_kerneltiming;

double count_kernel_time(int num_blocks, int threads_per_block, int synchronous)
{
    chTimerTimestamp start, stop;
    double elapsed_us = 0.0;

    chTimerGetTime(&start);
    NullKernel<<<num_blocks, threads_per_block>>>();
    if (synchronous)
        cudaDeviceSynchronize();
    chTimerGetTime(&stop);

    elapsed_us = 1e6 * chTimerElapsedTime(&start, &stop);
    return elapsed_us;
}

int logspace_int(int min, int max, int num_points, int idx)
{
    double log_min = log((double)min);
    double log_max = log((double)max);
    double val = log_min + (log_max - log_min) * idx / (num_points - 1);
    return (int)round(exp(val));
}

t_kerneltiming* measure_kernels_time(int num_points, int* out_count)
{
    int total = num_points * num_points;
    t_kerneltiming* results = (t_kerneltiming*)malloc(sizeof(t_kerneltiming) * total);
    if (!results) return NULL;

    NullKernel<<<1,1>>>();
    cudaDeviceSynchronize();

    int k = 0;
    for (int i = 0; i < num_points; i++)
    {
        int num_blocks = logspace_int(1, 16384, num_points, i);
        for (int j = 0; j < num_points; j++)
        {
            int threads_per_block = logspace_int(1, 1024, num_points, j);

            const int repeat = 5;
            double async_sum = 0.0;
            double sync_sum  = 0.0;
            for (int r = 0; r < repeat; r++) {
                async_sum += count_kernel_time(num_blocks, threads_per_block, 0);
                sync_sum  += count_kernel_time(num_blocks, threads_per_block, 1);
            }

            results[k].num_blocks = num_blocks;
            results[k].threads_per_block = threads_per_block;
            results[k].async_time_us = (float)(async_sum / repeat);
            results[k].sync_time_us  = (float)(sync_sum / repeat);
            k++;

            if (k % 10 == 0)
                printf(".");
            fflush(stdout);
        }
    }
    printf("\n");
    *out_count = total;
    return results;
}

void write_results_csv(const char* filename, t_kerneltiming* results, int count)
{
    FILE* fp = fopen(filename, "w");
    if (!fp)
    {
        printf("Failed to open file %s for writing.\n", filename);
        return;
    }

    fprintf(fp, "num_blocks,threads_per_block,async_time_us,sync_time_us\n");
    for (int i = 0; i < count; i++)
    {
        fprintf(fp, "%d,%d,%.6f,%.6f\n",
            results[i].num_blocks,
            results[i].threads_per_block,
            results[i].async_time_us,
            results[i].sync_time_us);
    }

    fclose(fp);
    printf("CSV file saved: %s\n", filename);
}

static void get_kernel_time_matrix()
{
    int num_points = 10;
    int count = 0;

    printf("Measuring kernel times for %d×%d configurations...\n", num_points, num_points);
    t_kerneltiming* results = measure_kernels_time(num_points, &count);
    if (results)
    {
        write_results_csv("kernel_timings.csv", results, count);
        free(results);
    }
}

int main()
{
    get_kernel_time_matrix();
    return 0;
}

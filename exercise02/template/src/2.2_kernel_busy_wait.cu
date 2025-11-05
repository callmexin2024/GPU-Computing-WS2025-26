#include <stdio.h>
#include <cuda_runtime.h>

__device__ volatile unsigned long long d_result;

__global__ void busy_wait_kernel(unsigned long long wait_cycles)
{
    unsigned long long start = clock64();
    unsigned long long now;
    do {
        now = clock64();
    } while (now - start < wait_cycles);

    d_result = now - start;
}

int main()
{
	cudaEvent_t start, stop;
    cudaSetDevice(0);

    unsigned long long wait_cycles = 5000;
    const unsigned long long max_cycles = 1 << 27;
    float ms;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    busy_wait_kernel<<<1,1>>>(wait_cycles);
    cudaDeviceSynchronize();

    cudaDeviceSynchronize();
    cudaEventRecord(start);
    busy_wait_kernel<<<1,1>>>(wait_cycles);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    float t0 = ms;
    printf("Base total_time (t0) at %llu cycles: %.6f ms\n\n", wait_cycles, t0);

    unsigned long long low = wait_cycles;
    unsigned long long high = wait_cycles * 2;
    float ms_high;

    while (high <= max_cycles) {
        cudaDeviceSynchronize();
        cudaEventRecord(start);
        busy_wait_kernel<<<1,1>>>(high);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms_high, start, stop);

        printf("Wait %10llu cycles -> total_time: %.6f ms\n", high, ms_high);

        if (ms_high >= 2.0 * t0) {
            break;
        }

        low = high;
        high *= 2;
    }

    unsigned long long best_cycles = high;
    while (high - low > 100) {
        unsigned long long mid = (low + high) / 2;

        cudaDeviceSynchronize();
        cudaEventRecord(start);
        busy_wait_kernel<<<1,1>>>(mid);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms, start, stop);

        printf("Binary search: wait_cycles %10llu -> total_time: %.6f ms\n", mid, ms);

        if (ms >= 2.0 * t0) {
            best_cycles = mid;
            high = mid;
        } else {
            low = mid;
        }
    }

    printf("\nMinimal wait_cycles to allow next kernel launch (time doubled): %llu ±100 cycles\n", best_cycles);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}

/*************************************************************************************************
 *
 *        Computer Engineering Group, Heidelberg University - GPU Computing Exercise 03
 *
 *                           Group : 06
 *
 *                            File : kernel.cu
 *
 *                         Purpose : Memory Operations Benchmark
 *
 *************************************************************************************************/

//
// Kernels
//

__global__ void
globalMemCoalescedKernel(int *d_src, int *d_dst, int numElements)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = idx; i < numElements; i += stride)
	{
		d_dst[i] = d_src[i];
	}
}

void globalMemCoalescedKernel_Wrapper(dim3 gridDim, dim3 blockDim, int *d_src, int *d_dst, int numElements)
{
	globalMemCoalescedKernel<<<gridDim, blockDim>>>(d_src, d_dst, numElements);
}

__global__ void
	globalMemStrideKernel(int *d_src, int *d_dst, int numElements, int stride_num)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx * stride_num; i < numElements; i += stride * stride_num)
    {
        d_dst[i] = d_src[i];
    }
}

void globalMemStrideKernel_Wrapper(dim3 gridDim, dim3 blockDim, int *d_src, int *d_dst, int numElements, int stride)
{
	globalMemStrideKernel<<<gridDim, blockDim>>>(d_src, d_dst, numElements, stride);
}

__global__ void
	globalMemOffsetKernel(int *d_src, int *d_dst, int numElements, int offset)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < numElements - offset; i += stride)
	{
		d_dst[i + offset] = d_src[i + offset];
	}
}

void globalMemOffsetKernel_Wrapper(dim3 gridDim, dim3 blockDim, int *d_src, int *d_dst, int numElements, int offset)
{
	globalMemOffsetKernel<<<gridDim, blockDim>>>(d_src, d_dst, numElements, offset);
}

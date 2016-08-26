#include <cuda_runtime.h>

#ifndef _softmax_kernel_
#define _softmax_kernel_

__device__ __forceinline__ float softmax(float a, float sum);

__global__ void softmax_kernel (const float * __restrict__ src,
                                                float * __restrict__ dst, int m, int n, float *sum, float *max);
__global__ void softmax_kernel2 (const float * __restrict__ src,
                                                float * __restrict__ dst, int m, int n, float *sum, float *max);

__global__ void softmax_kernel3 (const float * __restrict__ src,
                                                int m, int n, float *max);

#ifdef __cplusplus
extern "C" {
#endif
    void softmax_kernel_exec(const float *src, float *dst, int m, int n);
#ifdef __cplusplus
};
#endif

#endif

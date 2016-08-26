#include <cuda_runtime.h>

#ifndef _relu_kernel_
#define _relu_kernel_

__device__ __forceinline__ float relu(float a);

__global__ void relu_kernel (const float * __restrict__ src,
                                float * __restrict__ dst, int m, int n);
#ifdef __cplusplus
extern "C" {
#endif
    void relu_kernel_exec(const float *src, float *dst, int m, int n);
#ifdef __cplusplus
};
#endif

#endif

#include <cuda_runtime.h>

#ifndef _sigmoid_d_kernel_
#define _sigmoid_d_kernel_

__device__ __forceinline__ float sigmoid_d (float a);

__global__ void sigmoid_d_kernel (const float * __restrict__ src,
                                float * __restrict__ dst, int m, int n);
#ifdef __cplusplus
extern "C" {
#endif
    void sigmoid_d_kernel_exec(const float *src, float *dst, int m, int n);
#ifdef __cplusplus
};
#endif

#endif

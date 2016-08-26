#include <cuda_runtime.h>

#ifndef _tanh_kernel_
#define _tanh_kernel_

__device__ __forceinline__ float tanh_f(float a);

__global__ void tanh_kernel (const float * __restrict__ src,
                                float * __restrict__ dst, int m, int n);
#ifdef __cplusplus
extern "C" {
#endif
    void tanh_kernel_exec(const float *src, float *dst, int m, int n);
#ifdef __cplusplus
};
#endif

#endif

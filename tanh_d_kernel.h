
#include <cuda_runtime.h>

#ifndef _tanh_d_kernel_
#define _tanh_d_kernel_


__device__ __forceinline__ float tanh_d(float a);

__global__ void tanh_d_kernel (const float * __restrict__ src,
                                float * __restrict__ dst, int m, int n);
#ifdef __cplusplus
extern "C" {
#endif
    void tanh_d_kernel_exec(const float *src, float *dst, int m, int n);
#ifdef __cplusplus
};
#endif

#endif

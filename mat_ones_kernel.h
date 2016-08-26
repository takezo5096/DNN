#include <cuda_runtime.h>

#ifndef _mat_ones_kernel_
#define _mat_ones_kernel_

__global__ void mat_ones_kernel (const float * __restrict__ src,
                                float * __restrict__ dst, int m, int n);
#ifdef __cplusplus
extern "C" {
#endif
    void mat_ones_kernel_exec(const float *src, float *dst, int m, int n);
#ifdef __cplusplus
};
#endif

#endif

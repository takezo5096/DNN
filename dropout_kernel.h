#include <cuda_runtime.h>

#ifndef _dropout_kernel_
#define _dropout_kernel_

__global__ void dropout_kernel (const float * __restrict__ src,
                                float * __restrict__ dst, float * __restrict__ dst_idx, int m, int n, float p);
#ifdef __cplusplus
extern "C" {
#endif
    void dropout_kernel_exec(const float *src, float *dst, float *dst_idx, int m, int n, float p);
#ifdef __cplusplus
};
#endif

#endif

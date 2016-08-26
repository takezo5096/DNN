#include <cuda_runtime.h>

#ifndef _adam2_kernel_
#define _adam2_kernel_

__global__ void adam2_kernel (
                                                float * __restrict__ mm,
                                                float * __restrict__ mv,
                                                const float * __restrict__ mg,
                                                float * __restrict__ dst,
                                                float beta1, float beta2,
                                                float lr, float e, int m, int n);

#ifdef __cplusplus
extern "C" {
#endif
    void adam2_kernel_exec(float *mm, float *mv, const float *mg, float *dst, float beta1, float beta2, float lr, float e, int m, int n);

#ifdef __cplusplus
};
#endif

#endif

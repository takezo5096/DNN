#include <cuda_runtime.h>

#ifndef _softmax_cross_entropy_kernel_
#define _softmax_cross_entropy_kernel_

/*
 * softmax cross entropy kernel
 * dst = -sigma(log(src1 + 1e-8)*src2)
 *   */
__global__ void softmax_cross_entropy_kernel (
        const float * __restrict__ src1,
        const float * __restrict__ src2,
                                float * __restrict__ dst, int m, int n);
#ifdef __cplusplus
extern "C" {
#endif
    void softmax_cross_entropy_kernel_exec(const float *src1, const float *src2, float *dst, int m, int n);
#ifdef __cplusplus
};
#endif

#endif

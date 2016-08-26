#include <cuda_runtime.h>

#ifndef _mat_mul_elementwise_plus_kernel_
#define _mat_mul_elementwise_plus_kernel_

__global__ void mat_mul_elementwise_plus_kernel (
        const float * __restrict__ src1,
        const float * __restrict__ src2,
                                float * __restrict__ dst, float alpha, float beta, int m, int n);
#ifdef __cplusplus
extern "C" {
#endif
    void mat_mul_elementwise_plus_kernel_exec(const float *src1, const float *src2, float *dst, float alpha, float beta, int m, int n);
#ifdef __cplusplus
};
#endif

#endif

#include <cuda_runtime.h>

#ifndef _mat_mul_elementwise_kernel_
#define _mat_mul_elementwise_kernel_

/*
 *  * 行列要素の積を計算するカーネル
 *   */
__global__ void mat_mul_elementwise_kernel (const float * __restrict__ src1,
                                const float * __restrict__ src2,
                                float * __restrict__ dst, const int m, const int n);
#ifdef __cplusplus
extern "C" {
#endif
    void mat_mul_elementwise_kernel_exec(const float *src1, const float *src2, float *dst, const int m, const int n);
#ifdef __cplusplus
};
#endif

#endif

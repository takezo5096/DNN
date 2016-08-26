
#include <cuda_runtime.h>

#ifndef _mat_sum_kernel_
#define _mat_sum_kernel_

/*
 *  * 行列要素の合計を計算するカーネル
 *   */
__global__ void mat_sum_kernel (const float * __restrict__ src,
                                float * __restrict__ dst, int m, int n);
#ifdef __cplusplus
extern "C" {
#endif
    void mat_sum_kernel_exec(const float *src, float *dst, int m, int n);
#ifdef __cplusplus
};
#endif

#endif

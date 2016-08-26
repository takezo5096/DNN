#include <cuda_runtime.h>

#ifndef _mat_cos_kernel_
#define _mat_cos_kernel_

__device__ __forceinline__ float mat_cos(float a, float alpha);

/*
 *  * シグモイドカーネル
 *   */
__global__ void mat_cos_kernel (const float * __restrict__ src,
                                float * __restrict__ dst, int m, int n, float alpha);
#ifdef __cplusplus
extern "C" {
#endif
    void mat_cos_kernel_exec(const float *src, float *dst, int m, int n, float alpha);
#ifdef __cplusplus
};
#endif

#endif

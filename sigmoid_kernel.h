#include <cuda_runtime.h>

#ifndef _sigmoid_kernel_
#define _sigmoid_kernel_

__device__ __forceinline__ float sigmoid (float a);

/*
 *  * sigmoid kernel
 *   */
__global__ void sigmoid_kernel (const float * __restrict__ src,
                                float * __restrict__ dst, int m, int n);
#ifdef __cplusplus
extern "C" {
#endif
    void sigmoid_kernel_exec(const float *src, float *dst, int m, int n);
#ifdef __cplusplus
};
#endif

#endif

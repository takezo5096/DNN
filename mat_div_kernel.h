#include <cuda_runtime.h>

#ifndef _mat_div_kernel_
#define _mat_div_kernel_

__global__ void mat_div_kernel (
        const float * __restrict__ src1,
        const float * __restrict__ src2,
                                float * __restrict__ dst, int m, int n);
#ifdef __cplusplus
extern "C" {
#endif
    void mat_div_kernel_exec(const float *src1, const float *src2, float *dst, int m, int n);
#ifdef __cplusplus
};
#endif

#endif

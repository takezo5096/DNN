
/*
 *  * シグモイドカーネル
 *   */
__global__ void adam_kernel (
        const float * __restrict__ src1, 
        const float * __restrict__ src2, 
                                float * __restrict__ dst, float lr, float e, int m, int n);
#ifdef __cplusplus
extern "C" {
#endif
    void adam_kernel_exec(const float *src1, const float *src2, float *dst, float lr, float e, int m, int n);
#ifdef __cplusplus
};
#endif


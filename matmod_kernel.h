/*
 *  * シグモイド関数　
 *   */
__device__ __forceinline__ double matmod (float a, float p);

/*
 *  * シグモイドカーネル
 *   */
__global__ void matmod_kernel (const float * __restrict__ src,
                                float * __restrict__ dst, int m, int n, float p);
#ifdef __cplusplus
extern "C" {
#endif
    void matmod_kernel_exec(const float *src, float *dst, int m, int n, float p);
#ifdef __cplusplus
};
#endif


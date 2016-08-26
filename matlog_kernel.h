/*
 *  * シグモイド関数　
 *   */
__device__ __forceinline__ double matlog (float a, float alpha);

/*
 *  * シグモイドカーネル
 *   */
__global__ void matlog_kernel (const float * __restrict__ src, 
                                float * __restrict__ dst, int m, int n, float alpha);
#ifdef __cplusplus
extern "C" {
#endif
    void matlog_kernel_exec(const float *src, float *dst, int m, int n, float alpha);
#ifdef __cplusplus
};
#endif


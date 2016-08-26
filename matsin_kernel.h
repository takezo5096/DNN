/*
 *  * シグモイド関数　
 *   */
__device__ __forceinline__ double matsin(float a, float alpha);

/*
 *  * シグモイドカーネル
 *   */
__global__ void matsin_kernel (const float * __restrict__ src, 
                                float * __restrict__ dst, int m, int n, float alpha);
#ifdef __cplusplus
extern "C" {
#endif
    void matsin_kernel_exec(const float *src, float *dst, int m, int n, float alpha);
#ifdef __cplusplus
};
#endif


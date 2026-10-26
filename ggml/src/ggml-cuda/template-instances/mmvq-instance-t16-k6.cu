// mmvq-instance-t16-k6.cu - K=6 instance of the K2 trellis GEMV (wide path: 8K > 32 words).
// One file per K, using the fork's file layout for per-type instances: the template header is
// shared, this file fixes the rate, and t16-matmul.cu routes by K.  Unlike the iqk instances
// (which supply a device vec_dot_*), the body here is a launcher: the kernel is a standalone
// GEMV, not an MMVQ specialization.
#include "../t16-gemv.cuh"

void k2_t16_gemv_k6(int nw, int pf, dim3 grid, dim3 block, cudaStream_t stream,
                     const uint16_t * tw, const float * xf, float * yf,
                     int I, int J, size_t nb1, size_t nb2) {
    (void) pf;
    if      (nw == 8)  k2_t16_launch_wide<6, 8 >(grid, block, stream, tw, xf, yf, I, J, nb1, nb2);
    else if (nw == 32) k2_t16_launch_wide<6, 32>(grid, block, stream, tw, xf, yf, I, J, nb1, nb2);
    else               k2_t16_launch_wide<6, 16>(grid, block, stream, tw, xf, yf, I, J, nb1, nb2);
}

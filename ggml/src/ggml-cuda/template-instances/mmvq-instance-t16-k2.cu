// mmvq-instance-t16-k2.cu - K=2 instance of the K2 trellis GEMV (see ../t16-gemv.cuh).
// One file per K, using the fork's file layout for per-type instances: the template header is
// shared, this file fixes the rate, and t16-matmul.cu routes by K.  Unlike the iqk instances
// (which supply a device vec_dot_*), the body here is a launcher: the kernel is a standalone
// GEMV, not an MMVQ specialization.
#include "../t16-gemv.cuh"

void k2_t16_gemv_k2(int nw, int pf, dim3 grid, dim3 block, cudaStream_t stream,
                     const uint16_t * tw, const float * xf, float * yf,
                     int I, int J, size_t nb1, size_t nb2) {
    k2_t16_dispatch<2>(nw, pf, grid, block, stream, tw, xf, yf, I, J, nb1, nb2);
}

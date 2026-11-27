#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <type_traits>

static __device__ __constant__ uint32_t nt_tab[4][2] = {
    { 0x131e2722, 0x2d3f0119 },
    { 0x1322193f, 0x2d011e27 },
    { 0x01131922, 0x2d271e3f },
    { 0x3f22272d, 0x0113191e },
};

static __device__ __forceinline__ int nt_hdr(int ncols) { return ((4 + ncols/128 + 2 + 3)/4)*4; }

static __device__ __forceinline__ uint32_t nt_nibadd(uint32_t x, uint32_t y) {
    return ((x & 0x77777777) + (y & 0x77777777)) ^ ((x ^ y) & 0x88888888);
}

static __device__ __forceinline__ void nt_labels8(uint32_t prev, uint32_t cur, uint32_t & lo, uint32_t & hi) {
    const uint32_t Z1 = __funnelshift_r(prev, cur, 28), Z2 = __funnelshift_r(prev, cur, 24), Z3 = __funnelshift_r(prev, cur, 20);
    const uint32_t S03 = nt_nibadd(cur, Z3);
    const uint32_t i0 = (cur & 0x77777777) + (Z2 & 0x77777777), i1 = S03, i2 = nt_nibadd(S03, Z1) >> 1, i3 = nt_nibadd(S03, Z2) >> 1;
    const uint32_t s3 = __byte_perm(nt_tab[0][0], nt_tab[0][1], i0) + __byte_perm(nt_tab[1][0], nt_tab[1][1], i1)
                      + __byte_perm(nt_tab[2][0], nt_tab[2][1], i2) + __byte_perm(nt_tab[3][0], nt_tab[3][1], i3);
    const uint32_t s4 = __byte_perm(nt_tab[0][0], nt_tab[0][1], i0 >> 16) + __byte_perm(nt_tab[1][0], nt_tab[1][1], i1 >> 16)
                      + __byte_perm(nt_tab[2][0], nt_tab[2][1], i2 >> 16) + __byte_perm(nt_tab[3][0], nt_tab[3][1], i3 >> 16);
    lo = __byte_perm(s3, s4, 0x6420) ^ 0x80808080;
    hi = __byte_perm(s3, s4, 0x7531) ^ 0x80808080;
}

static __device__ __constant__ float nt_q255[256] = {
    0.000000000e+00f, 3.921568859e-03f, 7.843137719e-03f, 1.176470611e-02f, 1.568627544e-02f, 1.960784383e-02f, 2.352941222e-02f, 2.745098062e-02f,
    3.137255087e-02f, 3.529411927e-02f, 3.921568766e-02f, 4.313725606e-02f, 4.705882445e-02f, 5.098039284e-02f, 5.490196124e-02f, 5.882352963e-02f,
    6.274510175e-02f, 6.666667014e-02f, 7.058823854e-02f, 7.450980693e-02f, 7.843137532e-02f, 8.235294372e-02f, 8.627451211e-02f, 9.019608051e-02f,
    9.411764890e-02f, 9.803921729e-02f, 1.019607857e-01f, 1.058823541e-01f, 1.098039225e-01f, 1.137254909e-01f, 1.176470593e-01f, 1.215686277e-01f,
    1.254902035e-01f, 1.294117719e-01f, 1.333333403e-01f, 1.372549087e-01f, 1.411764771e-01f, 1.450980455e-01f, 1.490196139e-01f, 1.529411823e-01f,
    1.568627506e-01f, 1.607843190e-01f, 1.647058874e-01f, 1.686274558e-01f, 1.725490242e-01f, 1.764705926e-01f, 1.803921610e-01f, 1.843137294e-01f,
    1.882352978e-01f, 1.921568662e-01f, 1.960784346e-01f, 2.000000030e-01f, 2.039215714e-01f, 2.078431398e-01f, 2.117647082e-01f, 2.156862766e-01f,
    2.196078449e-01f, 2.235294133e-01f, 2.274509817e-01f, 2.313725501e-01f, 2.352941185e-01f, 2.392156869e-01f, 2.431372553e-01f, 2.470588237e-01f,
    2.509804070e-01f, 2.549019754e-01f, 2.588235438e-01f, 2.627451122e-01f, 2.666666806e-01f, 2.705882490e-01f, 2.745098174e-01f, 2.784313858e-01f,
    2.823529541e-01f, 2.862745225e-01f, 2.901960909e-01f, 2.941176593e-01f, 2.980392277e-01f, 3.019607961e-01f, 3.058823645e-01f, 3.098039329e-01f,
    3.137255013e-01f, 3.176470697e-01f, 3.215686381e-01f, 3.254902065e-01f, 3.294117749e-01f, 3.333333433e-01f, 3.372549117e-01f, 3.411764801e-01f,
    3.450980484e-01f, 3.490196168e-01f, 3.529411852e-01f, 3.568627536e-01f, 3.607843220e-01f, 3.647058904e-01f, 3.686274588e-01f, 3.725490272e-01f,
    3.764705956e-01f, 3.803921640e-01f, 3.843137324e-01f, 3.882353008e-01f, 3.921568692e-01f, 3.960784376e-01f, 4.000000060e-01f, 4.039215744e-01f,
    4.078431427e-01f, 4.117647111e-01f, 4.156862795e-01f, 4.196078479e-01f, 4.235294163e-01f, 4.274509847e-01f, 4.313725531e-01f, 4.352941215e-01f,
    4.392156899e-01f, 4.431372583e-01f, 4.470588267e-01f, 4.509803951e-01f, 4.549019635e-01f, 4.588235319e-01f, 4.627451003e-01f, 4.666666687e-01f,
    4.705882370e-01f, 4.745098054e-01f, 4.784313738e-01f, 4.823529422e-01f, 4.862745106e-01f, 4.901960790e-01f, 4.941176474e-01f, 4.980392158e-01f,
    5.019608140e-01f, 5.058823824e-01f, 5.098039508e-01f, 5.137255192e-01f, 5.176470876e-01f, 5.215686560e-01f, 5.254902244e-01f, 5.294117928e-01f,
    5.333333611e-01f, 5.372549295e-01f, 5.411764979e-01f, 5.450980663e-01f, 5.490196347e-01f, 5.529412031e-01f, 5.568627715e-01f, 5.607843399e-01f,
    5.647059083e-01f, 5.686274767e-01f, 5.725490451e-01f, 5.764706135e-01f, 5.803921819e-01f, 5.843137503e-01f, 5.882353187e-01f, 5.921568871e-01f,
    5.960784554e-01f, 6.000000238e-01f, 6.039215922e-01f, 6.078431606e-01f, 6.117647290e-01f, 6.156862974e-01f, 6.196078658e-01f, 6.235294342e-01f,
    6.274510026e-01f, 6.313725710e-01f, 6.352941394e-01f, 6.392157078e-01f, 6.431372762e-01f, 6.470588446e-01f, 6.509804130e-01f, 6.549019814e-01f,
    6.588235497e-01f, 6.627451181e-01f, 6.666666865e-01f, 6.705882549e-01f, 6.745098233e-01f, 6.784313917e-01f, 6.823529601e-01f, 6.862745285e-01f,
    6.901960969e-01f, 6.941176653e-01f, 6.980392337e-01f, 7.019608021e-01f, 7.058823705e-01f, 7.098039389e-01f, 7.137255073e-01f, 7.176470757e-01f,
    7.215686440e-01f, 7.254902124e-01f, 7.294117808e-01f, 7.333333492e-01f, 7.372549176e-01f, 7.411764860e-01f, 7.450980544e-01f, 7.490196228e-01f,
    7.529411912e-01f, 7.568627596e-01f, 7.607843280e-01f, 7.647058964e-01f, 7.686274648e-01f, 7.725490332e-01f, 7.764706016e-01f, 7.803921700e-01f,
    7.843137383e-01f, 7.882353067e-01f, 7.921568751e-01f, 7.960784435e-01f, 8.000000119e-01f, 8.039215803e-01f, 8.078431487e-01f, 8.117647171e-01f,
    8.156862855e-01f, 8.196078539e-01f, 8.235294223e-01f, 8.274509907e-01f, 8.313725591e-01f, 8.352941275e-01f, 8.392156959e-01f, 8.431372643e-01f,
    8.470588326e-01f, 8.509804010e-01f, 8.549019694e-01f, 8.588235378e-01f, 8.627451062e-01f, 8.666666746e-01f, 8.705882430e-01f, 8.745098114e-01f,
    8.784313798e-01f, 8.823529482e-01f, 8.862745166e-01f, 8.901960850e-01f, 8.941176534e-01f, 8.980392218e-01f, 9.019607902e-01f, 9.058823586e-01f,
    9.098039269e-01f, 9.137254953e-01f, 9.176470637e-01f, 9.215686321e-01f, 9.254902005e-01f, 9.294117689e-01f, 9.333333373e-01f, 9.372549057e-01f,
    9.411764741e-01f, 9.450980425e-01f, 9.490196109e-01f, 9.529411793e-01f, 9.568627477e-01f, 9.607843161e-01f, 9.647058845e-01f, 9.686274529e-01f,
    9.725490212e-01f, 9.764705896e-01f, 9.803921580e-01f, 9.843137264e-01f, 9.882352948e-01f, 9.921568632e-01f, 9.960784316e-01f, 1.000000000e+00f,
};

static __device__ __forceinline__ float nt_block_scale(float d, int q) { return __fmul_rn(d, nt_q255[q]); }

template <typename dst_t>
static __global__ void nt_dequantize_block(const void * __restrict__ vx, dst_t * __restrict__ yy, int64_t n_per_row, int64_t row_size, int64_t nunits) {
    const int64_t g = (int64_t)blockIdx.x*2 + threadIdx.x/16;
    if (g >= nunits) return;
    const int64_t nbr = n_per_row/128;
    const int64_t row = g / nbr;
    const int64_t ib  = g - row*nbr;
    const char * cx = (const char *)vx + row*row_size;
    const float d = *(const float *)cx;
    const float s = nt_block_scale(d, ((const uint8_t *)cx)[4 + ib]);
    const uint32_t * q32 = (const uint32_t *)(cx + nt_hdr((int)n_per_row) + 64*ib);
    const int w = threadIdx.x%16, j = 4*w;
    uint32_t lo, hi;
    nt_labels8(q32[w - 1], q32[w], lo, hi);
    const int c0 = (int)(128*ib) + j;
    float v[8];
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        v[k]   = __fmul_rn(s, (float)(int8_t)(lo >> 8*k));
        v[k+4] = __fmul_rn(s, (float)(int8_t)(hi >> 8*k));
    }
    dst_t * y = yy + row*n_per_row;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        if constexpr (std::is_same_v<dst_t, nv_bfloat16>) {
            y[c0 + k] = __float2bfloat16(v[k]); y[c0 + 64 + k] = __float2bfloat16(v[k+4]);
        } else if constexpr (std::is_same_v<dst_t, half>) {
            y[c0 + k] = __float2half(v[k]); y[c0 + 64 + k] = __float2half(v[k+4]);
        } else {
            y[c0 + k] = v[k]; y[c0 + 64 + k] = v[k+4];
        }
    }
}

struct nt_part {
    int wb, ia, ib;
    __device__ __forceinline__ nt_part(int iqs) {
        const int j0 = (iqs/8)*32;
        wb = j0/4; ia = j0/32; ib = ia + 2;
    }
};

template <int ncols_y, int rows, typename block_q8_1_t>
static __device__ __forceinline__ void nt_vec_dot_q8_1_multi(const void * __restrict__ vbq, const block_q8_1_t * __restrict__ y0, int stride_y,
        int kbx, int iqs, int ncols, float (&tmp)[ncols_y][rows], int i) {
    const char * cx = (const char *)vbq;
    const nt_part p(iqs);
    const uint32_t * q32 = (const uint32_t *)(cx + nt_hdr(ncols) + 64*kbx);
    uint32_t lo[8], hi[8];
    uint32_t prev = q32[p.wb - 1];
#pragma unroll
    for (int m = 0; m < 8; ++m) { const uint32_t cur = q32[p.wb + m]; nt_labels8(prev, cur, lo[m], hi[m]); prev = cur; }
    const float s = nt_block_scale(*(const float *)cx, ((const uint8_t *)cx)[4 + kbx]);
#pragma unroll
    for (int j = 0; j < ncols_y; ++j) {
        const block_q8_1_t * bq = y0 + j*stride_y;
        const int * qa = (const int *)bq[p.ia].qs, * qb = (const int *)bq[p.ib].qs;
        int sa = 0, sb = 0;
#pragma unroll
        for (int m = 0; m < 8; ++m) { sa = __dp4a((int)lo[m], qa[m], sa); sb = __dp4a((int)hi[m], qb[m], sb); }
        tmp[j][i] += s * (__low2float(bq[p.ia].ds) * sa + __low2float(bq[p.ib].ds) * sb);
    }
}

template <typename block_q8_1_t>
static __device__ __forceinline__ void nt_vec_dot_q8_1(const void * __restrict__ vbq, const block_q8_1_t * __restrict__ bq8_1,
        int kbx, int iqs, int ncols, float * result) {
    const char * cx = (const char *)vbq;
    const nt_part p(iqs);
    const uint32_t * q32 = (const uint32_t *)(cx + nt_hdr(ncols) + 64*kbx);
    const int * qa = (const int *)bq8_1[p.ia].qs, * qb = (const int *)bq8_1[p.ib].qs;
    int sa = 0, sb = 0;
    uint32_t prev = q32[p.wb - 1];
#pragma unroll
    for (int m = 0; m < 8; ++m) {
        const uint32_t cur = q32[p.wb + m];
        uint32_t lo, hi;
        nt_labels8(prev, cur, lo, hi);
        sa = __dp4a((int)lo, qa[m], sa); sb = __dp4a((int)hi, qb[m], sb);
        prev = cur;
    }
    const float s = nt_block_scale(*(const float *)cx, ((const uint8_t *)cx)[4 + kbx]);
    *result += s * (__low2float(bq8_1[p.ia].ds) * sa + __low2float(bq8_1[p.ib].ds) * sb);
}

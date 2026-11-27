#pragma once
#include "common.cuh"
#include "mmvq-args.h"
#include "nt-common.cuh"

struct nt_ycol {
    const int8_t * qs; const half * d;
    __device__ __forceinline__ nt_ycol(const char * col, int np) : qs((const int8_t *)col), d((const half *)(col + np)) {}
};
static __device__ __forceinline__ int64_t nt_ycol_bytes(int np) { return (int64_t)(np/32)*36; }

static __device__ __forceinline__ void nt_yload(const nt_ycol & y, int ib, int (&q)[8]) {
    const int4 * p = (const int4 *)(y.qs + 32*ib);
    const int4 a = p[0], b = p[1];
    q[0] = a.x; q[1] = a.y; q[2] = a.z; q[3] = a.w; q[4] = b.x; q[5] = b.y; q[6] = b.z; q[7] = b.w;
}

static __device__ __forceinline__ float nt_dot64(const char * __restrict__ cx, const block_q8_1 * __restrict__ y, int kbx, int t4, int ncols) {
    const int j0 = t4*32, wb = j0/4, ia = j0/32, ib = ia + 2;
    const uint32_t * q32 = (const uint32_t *)(cx + nt_hdr(ncols) + 64*kbx);
    const int * qa = (const int *)y[ia].qs, * qb = (const int *)y[ib].qs;
    int sa = 0, sb = 0;
    uint32_t prev = q32[wb - 1];
#pragma unroll
    for (int m = 0; m < 8; ++m) {
        const uint32_t cur = q32[wb + m];
        uint32_t lo, hi;
        nt_labels8(prev, cur, lo, hi);
        sa = __dp4a((int)lo, qa[m], sa); sb = __dp4a((int)hi, qb[m], sb);
        prev = cur;
    }
    return nt_block_scale(*(const float *)cx, ((const uint8_t *)cx)[4 + kbx]) * (__low2float(y[ia].ds) * sa + __low2float(y[ib].ds) * sb);
}

template <int NW, bool FUSED>
__launch_bounds__(32*NW, 1)
static __global__ void nt_mul_mat_vec_1(const char * __restrict__ vx_u, const char * __restrict__ vx_g, const void * __restrict__ vy,
        float * __restrict__ dst, const char * __restrict__ ids_data, const void * __restrict__ bias_u, const void * __restrict__ bias_g,
        uint64_t bias_nb1, int ncols_x, int nrows_x, int nrows_dst, int64_t row_size,
        uint64_t nb02, uint64_t nb12, uint64_t nb2, int64_t ids_nb0, ggml_unary_op unary_op, float limit) {
    const int row = blockIdx.x*NW + threadIdx.y, lane = threadIdx.x;
    if (row >= nrows_x) return;
    const int i2 = blockIdx.y;
    const int i02 = ids_data ? *(const int *)(ids_data + i2*ids_nb0) : i2;
    if (i02 < 0) return;
    const block_q8_1 * y = (const block_q8_1 *)((const char *)vy + i2*nb12);
    const char * cu = vx_u + i02*nb02 + row*row_size;
    const char * cg = FUSED ? vx_g + i02*nb02 + row*row_size : nullptr;
    const int nu = ncols_x/64;
    float su = 0, sg = 0;
    for (int u = lane; u < nu; u += 32) {
        su += nt_dot64(cu, y + (u/2)*4, u/2, u%2, ncols_x);
        if (FUSED) sg += nt_dot64(cg, y + (u/2)*4, u/2, u%2, ncols_x);
    }
    su = warp_reduce_sum(su);
    if (FUSED) sg = warp_reduce_sum(sg);
    if (lane != 0) return;
    float * d = (float *)((char *)dst + i2*nb2);
    const float * bu = bias_u ? (const float *)((const char *)bias_u + (ids_data ? i02*bias_nb1 : 0)) : nullptr;
    if (!FUSED) {
        d[row] = bu ? su + bu[row] : su;
        return;
    }
    const float * bg = bias_g ? (const float *)((const char *)bias_g + (ids_data ? i02*bias_nb1 : 0)) : nullptr;
    float u = su, g = sg, r;
    if (unary_op == GGML_UNARY_OP_SWIGLU_OAI && !bu) {
        constexpr float alpha = 1.702f, lim = 7.0f;
        g = fminf(g, lim); u = fmaxf(fminf(u, lim), -lim);
        r = g / (1.0f + expf(-g * alpha)) * (1.0f + u);
    } else {
        switch (unary_op) {
            case GGML_UNARY_OP_SILU: { g = g/(1 + expf(-g)); g = min(g, limit); r = max(-limit, min(limit, u))*g; } break;
            case GGML_UNARY_OP_RELU: r = fmaxf(g, 0.0f) * u; break;
            case GGML_UNARY_OP_GELU: {
                constexpr float GELU_COEF_A = 0.044715f, SQRT_2_OVER_PI = 0.79788456080286535587989211986876f;
                r = 0.5f*g*u*(1.0f + tanhf(SQRT_2_OVER_PI*g*(1.0f + GELU_COEF_A*g*g)));
            } break;
            default: {
                constexpr float alpha = 1.702f, lim = 7.0f;
                g += bg[row]; u += bu[row];
                g = fminf(g, lim); u = fmaxf(fminf(u, lim), -lim);
                r = g / (1.0f + expf(-g * alpha)) * (1.0f + u);
            } break;
        }
    }
    (void)nrows_dst;
    d[row] = r;
}

template <int NW>
static void nt_launch_1(const mmvq_args & a, cudaStream_t stream) {
    const int64_t row_size = ggml_row_size(GGML_TYPE_IQ4_NT, a.ncols_x);
    const dim3 grid((a.nrows_x + NW - 1)/NW, a.ne2, 1), block(32, NW, 1);
    const bool fused = a.vx_u && a.vx_g && a.unary_op != GGML_UNARY_OP_COUNT;
    if (fused) {
        nt_mul_mat_vec_1<NW, true><<<grid, block, 0, stream>>>((const char *)a.vx_u, (const char *)a.vx_g, a.vy, a.dst, a.ids_data,
                a.bias_u, a.bias_g, a.bias_nb1, a.ncols_x, a.nrows_x, a.nrows_dst, row_size, a.nb02, a.nb12, a.nb2, a.ids_nb0, a.unary_op, a.limit);
    } else {
        nt_mul_mat_vec_1<NW, false><<<grid, block, 0, stream>>>((const char *)a.vx_u, nullptr, a.vy, a.dst, a.ids_data,
                a.bias_u, nullptr, a.bias_nb1, a.ncols_x, a.nrows_x, a.nrows_dst, row_size, a.nb02, a.nb12, a.nb2, a.ids_nb0, a.unary_op, a.limit);
    }
}

template <int NC>
static __device__ __forceinline__ void nt_dot64_n(const char * __restrict__ cx, const char * __restrict__ ybase, int np,
        int kbx, int t4, int ncols, float (&s)[NC]) {
    const int j0 = t4*32, wb = j0/4, ia = 4*kbx + j0/32, ib = ia + 2;
    const uint32_t * q32 = (const uint32_t *)(cx + nt_hdr(ncols) + 64*kbx);
    uint32_t lo[8], hi[8];
    uint32_t prev = q32[wb - 1];
#pragma unroll
    for (int m = 0; m < 8; ++m) { const uint32_t cur = q32[wb + m]; nt_labels8(prev, cur, lo[m], hi[m]); prev = cur; }
    const float d = nt_block_scale(*(const float *)cx, ((const uint8_t *)cx)[4 + kbx]);
#pragma unroll
    for (int j = 0; j < NC; ++j) {
        const nt_ycol y(ybase + j*nt_ycol_bytes(np), np);
        int qa[8], qb[8];
        nt_yload(y, ia, qa); nt_yload(y, ib, qb);
        int sa = 0, sb = 0;
#pragma unroll
        for (int m = 0; m < 8; ++m) { sa = __dp4a((int)lo[m], qa[m], sa); sb = __dp4a((int)hi[m], qb[m], sb); }
        s[j] += d * (__half2float(y.d[ia]) * sa + __half2float(y.d[ib]) * sb);
    }
}

static __device__ __forceinline__ float nt_unary(ggml_unary_op unary_op, float u, float g, const float * bu, const float * bg, int row, float limit) {
    if (unary_op == GGML_UNARY_OP_SWIGLU_OAI && !bu) {
        constexpr float alpha = 1.702f, lim = 7.0f;
        g = fminf(g, lim); u = fmaxf(fminf(u, lim), -lim);
        return g / (1.0f + expf(-g * alpha)) * (1.0f + u);
    }
    switch (unary_op) {
        case GGML_UNARY_OP_SILU: { g = g/(1 + expf(-g)); g = min(g, limit); return max(-limit, min(limit, u))*g; }
        case GGML_UNARY_OP_RELU: return fmaxf(g, 0.0f) * u;
        case GGML_UNARY_OP_GELU: {
            constexpr float GELU_COEF_A = 0.044715f, SQRT_2_OVER_PI = 0.79788456080286535587989211986876f;
            return 0.5f*g*u*(1.0f + tanhf(SQRT_2_OVER_PI*g*(1.0f + GELU_COEF_A*g*g)));
        }
        default: {
            constexpr float alpha = 1.702f, lim = 7.0f;
            g += bg[row]; u += bu[row];
            g = fminf(g, lim); u = fmaxf(fminf(u, lim), -lim);
            return g / (1.0f + expf(-g * alpha)) * (1.0f + u);
        }
    }
}

template <int NW, int NC, bool FUSED>
__launch_bounds__(32*NW, 4)
static __global__ void nt_mul_mat_vec_n(const char * __restrict__ vx_u, const char * __restrict__ vx_g, const void * __restrict__ vy,
        float * __restrict__ dst, const char * __restrict__ ids_data, const void * __restrict__ bias_u, const void * __restrict__ bias_g,
        uint64_t bias_nb1, int ncols_x, int nrows_x, int nrows_y, int nrows_dst, int64_t row_size,
        uint64_t nb02, uint64_t nb12, uint64_t nb2, int64_t ids_nb0, ggml_unary_op unary_op, float limit) {
    const int row = blockIdx.x*NW + threadIdx.y, lane = threadIdx.x;
    if (row >= nrows_x) return;
    const int i2 = blockIdx.y;
    const int i02 = ids_data ? *(const int *)(ids_data + i2*ids_nb0) : i2;
    if (i02 < 0) return;
    const char * ybase = (const char *)vy + i2*nb12;
    const char * cu = vx_u + i02*nb02 + row*row_size;
    const char * cg = FUSED ? vx_g + i02*nb02 + row*row_size : nullptr;
    const int nu = ncols_x/64;
    float su[NC] = {}, sg[NC] = {};
    for (int u = lane; u < nu; u += 32) {
        nt_dot64_n<NC>(cu, ybase, nrows_y, u/2, u%2, ncols_x, su);
        if (FUSED) nt_dot64_n<NC>(cg, ybase, nrows_y, u/2, u%2, ncols_x, sg);
    }
#pragma unroll
    for (int j = 0; j < NC; ++j) {
        su[j] = warp_reduce_sum(su[j]);
        if (FUSED) sg[j] = warp_reduce_sum(sg[j]);
    }
    if (lane != 0) return;
    float * d = (float *)((char *)dst + i2*nb2);
    const float * bu = bias_u ? (const float *)((const char *)bias_u + (ids_data ? i02*bias_nb1 : 0)) : nullptr;
    const float * bg = bias_g ? (const float *)((const char *)bias_g + (ids_data ? i02*bias_nb1 : 0)) : nullptr;
#pragma unroll
    for (int j = 0; j < NC; ++j) {
        d[j*nrows_dst + row] = FUSED ? nt_unary(unary_op, su[j], sg[j], bu, bg, row, limit) : (bu ? su[j] + bu[row] : su[j]);
    }
}

template <int NW, int NC>
static void nt_launch_n(const mmvq_args & a, cudaStream_t stream) {
    const int64_t row_size = ggml_row_size(GGML_TYPE_IQ4_NT, a.ncols_x);
    const dim3 grid((a.nrows_x + NW - 1)/NW, a.ne2, 1), block(32, NW, 1);
    const bool fused = a.vx_u && a.vx_g && a.unary_op != GGML_UNARY_OP_COUNT;
    if (fused) {
        nt_mul_mat_vec_n<NW, NC, true><<<grid, block, 0, stream>>>((const char *)a.vx_u, (const char *)a.vx_g, a.vy, a.dst, a.ids_data,
                a.bias_u, a.bias_g, a.bias_nb1, a.ncols_x, a.nrows_x, a.nrows_y, a.nrows_dst, row_size, a.nb02, a.nb12, a.nb2, a.ids_nb0, a.unary_op, a.limit);
    } else {
        nt_mul_mat_vec_n<NW, NC, false><<<grid, block, 0, stream>>>((const char *)a.vx_u, nullptr, a.vy, a.dst, a.ids_data,
                a.bias_u, nullptr, a.bias_nb1, a.ncols_x, a.nrows_x, a.nrows_y, a.nrows_dst, row_size, a.nb02, a.nb12, a.nb2, a.ids_nb0, a.unary_op, a.limit);
    }
}

static inline bool nt_mul_mat_vec_1_cuda(const mmvq_args & a, cudaStream_t stream) {
    switch (a.ncols_y) {
        case 1: nt_launch_1<4>(a, stream); return true;
        case 2: nt_launch_n<4, 2>(a, stream); return true;
        case 3: nt_launch_n<4, 3>(a, stream); return true;
        case 4: nt_launch_n<4, 4>(a, stream); return true;
        case 5: nt_launch_n<4, 5>(a, stream); return true;
        case 6: nt_launch_n<4, 6>(a, stream); return true;
        case 7: nt_launch_n<4, 7>(a, stream); return true;
        case 8: nt_launch_n<4, 8>(a, stream); return true;
        default: GGML_ABORT("IQ4_NT: %d activation columns in the vector kernel (at most 8; the activations use the IQ4_NT layout)", (int)a.ncols_y);
    }
}

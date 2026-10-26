#include "t16-matmul.cuh"

#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <vector>
#include <mma.h>
#include "t16-gemv.cuh"

using namespace nvcuda;

// K2 (t16): exact EXL3 trellis decode.
//
//   window = 16 circular bits at stream position tpos (K bits per step)
//   x = window * 0x83DCD12D (u32); s = 0x6400 + byte_sum(x)
//   h = reinterpret_f16(low 16 bits of s); val = hfma(h, 0x1EEE, 0xC931)
//
// The trellis tile (i, j) is 16K little-endian uint16 words at
//   data + j*nb1 + i*nb2,
// and the decoded 16x16 tile is row-major (in, out): position 16*r + c holds the
// window inv_perm[16*r + c]. One block decodes one tile for up to K2_T16_TOK_BLK
// token columns; threads are (c = tid/16, r = tid%16) so the 16-lane reduction over
// r is a width-16 shuffle.

#define K2_T16_TOK_BLK 32

// inv_perm[q]: window stream position for the row-major tile position q = 16*r + c
// (inverse of exllamav3's tensor_core_perm, verified against k2_codec).
static __constant__ uint8_t c_k2_t16_inv_perm[256] = {
      0,  32,  64,  96, 128, 160, 192, 224,   4,  36,  68, 100, 132, 164, 196, 228,
      1,  33,  65,  97, 129, 161, 193, 225,   5,  37,  69, 101, 133, 165, 197, 229,
      8,  40,  72, 104, 136, 168, 200, 232,  12,  44,  76, 108, 140, 172, 204, 236,
      9,  41,  73, 105, 137, 169, 201, 233,  13,  45,  77, 109, 141, 173, 205, 237,
     16,  48,  80, 112, 144, 176, 208, 240,  20,  52,  84, 116, 148, 180, 212, 244,
     17,  49,  81, 113, 145, 177, 209, 241,  21,  53,  85, 117, 149, 181, 213, 245,
     24,  56,  88, 120, 152, 184, 216, 248,  28,  60,  92, 124, 156, 188, 220, 252,
     25,  57,  89, 121, 153, 185, 217, 249,  29,  61,  93, 125, 157, 189, 221, 253,
      2,  34,  66,  98, 130, 162, 194, 226,   6,  38,  70, 102, 134, 166, 198, 230,
      3,  35,  67,  99, 131, 163, 195, 227,   7,  39,  71, 103, 135, 167, 199, 231,
     10,  42,  74, 106, 138, 170, 202, 234,  14,  46,  78, 110, 142, 174, 206, 238,
     11,  43,  75, 107, 139, 171, 203, 235,  15,  47,  79, 111, 143, 175, 207, 239,
     18,  50,  82, 114, 146, 178, 210, 242,  22,  54,  86, 118, 150, 182, 214, 246,
     19,  51,  83, 115, 147, 179, 211, 243,  23,  55,  87, 119, 151, 183, 215, 247,
     26,  58,  90, 122, 154, 186, 218, 250,  30,  62,  94, 126, 158, 190, 222, 254,
     27,  59,  91, 123, 155, 187, 219, 251,  31,  63,  95, 127, 159, 191, 223, 255,
};



static __global__ void k2_t16_matmul_kernel(
        const uint16_t * __restrict__ trellis,
        const float    * __restrict__ x,
        float          * __restrict__ y,
        int K, int I, int J, int n,
        size_t nb1, size_t nb2) {
    const int j    = blockIdx.x;
    const int t0   = blockIdx.y * K2_T16_TOK_BLK;
    const int tid  = threadIdx.x;
    const int c    = tid >> 4;
    const int r    = tid & 15;
    const int q    = 16*r + c;
    const int tpos = c_k2_t16_inv_perm[q];
    const int b    = tpos*K + K - 16 + 256*K;
    const int ih   = b >> 5;
    const int il   = (b + 15) >> 5;
    const int sh   = (il + 1)*32 - (b + 16);
    const int nw   = 8*K;

    float acc[K2_T16_TOK_BLK];
#pragma unroll
    for (int t = 0; t < K2_T16_TOK_BLK; ++t) {
        acc[t] = 0.0f;
    }

    const int ntok = min(K2_T16_TOK_BLK, n - t0);
    const char * tile0 = (const char *) trellis + j*nb1;
    int buf = 0;
    for (int i = 0; i < I; ++i) {
        const uint32_t * p = (const uint32_t *)(tile0 + i*nb2);
        buf ^= 1;
        const uint32_t w1 = k2_t16_fshr(p[il % nw], p[ih % nw], (uint32_t) sh);
        const float v = k2_t16_mul1((uint16_t)(w1 & 0xFFFFu));
        const float * xr = x + (16*i + r);
        for (int t = 0; t < ntok; ++t) {
            acc[t] += v * xr[(t0 + t)*(size_t)(16*I)];
        }
    }

#pragma unroll
    for (int off = 8; off > 0; off >>= 1) {
#pragma unroll
        for (int t = 0; t < K2_T16_TOK_BLK; ++t) {
            acc[t] += __shfl_down_sync(0xFFFFFFFFu, acc[t], off, 16);
        }
    }

    if (r == 0) {
        float * yc = y + (16*j + c);
        for (int t = 0; t < ntok; ++t) {
            yc[(t0 + t)*(size_t)(16*J)] = acc[t];
        }
    }
}

// ---------------------------------------------------------------------------
// Tensor-core prefill (n > 1): f16-MMA, v2.
//
// v1 (kept in git history) ran one MMA per warp per input tile with two block-wide
// __syncthreads() per tile and re-read the activation tile once per 16-output row: 1606 t/s
// pp512, 38% of this card's f16-with-f32-accumulate ceiling (~4267 t/s) and 5.3x behind an
// IQ4_KS.  v2 fixes the structure:
//   * block = 8 warps over a 64-output x 128-token tile: warp (m, nh) owns output tile
//     j0+m and 64 tokens, so each input tile costs 4 MMA per warp (32 per block) instead of 1;
//   * every warp decodes its OWN 16x16 weight tile into its own shared slot and stages its own
//     activations, so the loop needs only __syncwarp() -- no block-wide barrier at all
//     (the duplicate decode by the two warps sharing an output tile is ~200 ops per tile, far
//     cheaper than a barrier);
//   * activations are read as float4 (16 B per lane, 8 tokens per instruction) and written to
//     shared as two half2 stores -- coalesced and 8-byte aligned by construction.
//
// Precision: k2_t16_mul1 returns __half2float of an fp16 hfma result, so the decoded weight is
// already an fp16 value and __float2half round-trips exactly; only the activations round
// (f32 -> f16).  Accumulators are f32.
// ---------------------------------------------------------------------------

#define K2_T16_MMA_WARPS   8
#define K2_T16_MMA_M       4                      // output tiles per block (16 each)
#define K2_T16_MMA_NHALF   2                      // token halves per warp
#define K2_T16_MMA_TOKENS  (K2_T16_MMA_NHALF*64)  // 128 tokens per block
#define K2_T16_MMA_NFRAG   4                      // 16-token MMA fragments per warp

// Pipeline stage: decode the block's 4 weight tiles and convert the activation tile, into the
// given shared buffers.  Kept as a device function so the loop can stage tile i+1 *before* the
// mma of tile i -- the whole point of the software pipeline below.
static __device__ __forceinline__ void k2_t16_mma_stage(
        const uint16_t * __restrict__ trellis,
        const float    * __restrict__ x,
        half           * sA_buf,
        half           * sX_buf,
        int K, int I, int n, size_t nb1, size_t nb2,
        int i, int j0, int t0, int tid, int nw) {
    const int q    = tid;                       // 256 threads cover one 16x16 tile exactly
    const int cc   = q >> 4;
    const int rr   = q & 15;
    const int tpos = c_k2_t16_inv_perm[16*rr + cc];
    const int bpos = tpos*K + K - 16 + 256*K;
    const int ih   = bpos >> 5;
    const int il   = (bpos + 15) >> 5;
    const int sh   = (il + 1)*32 - (bpos + 16);
#pragma unroll
    for (int mt = 0; mt < K2_T16_MMA_M; ++mt) {
        const uint32_t * pp = (const uint32_t *)((const char *) trellis + (j0 + mt)*nb1 + i*nb2);
        const uint32_t w1 = k2_t16_fshr(pp[il % nw], pp[ih % nw], (uint32_t) sh);
        sA_buf[mt*(16*24) + 24*cc + rr] = __float2half(k2_t16_mul1((uint16_t)(w1 & 0xFFFFu)));
    }
    // activations: 16 in x 128 tokens, 8 values per thread
#pragma unroll
    for (int e = 0; e < 2; ++e) {
        const int idx = tid + 256*e;            // float4 slot 0..511
        const int t   = idx >> 2;
        const int k4  = (idx & 3) * 4;
        const int gtok = t0 + t;
        float4 xv = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        if (gtok < n) {
            xv = *(const float4 *)(x + (16*i + k4) + (size_t)gtok*(size_t)(16*I));
        }
        half * dst = &sX_buf[t*24 + k4];
        *reinterpret_cast<half2 *>(dst)     = __floats2half2_rn(xv.x, xv.y);
        *reinterpret_cast<half2 *>(dst + 2) = __floats2half2_rn(xv.z, xv.w);
    }
}

static __global__ void __launch_bounds__(32*K2_T16_MMA_WARPS)
k2_t16_mma_kernel(
        const uint16_t * __restrict__ trellis,
        const float    * __restrict__ x,
        float          * __restrict__ y,
        int K, int I, int J, int n,
        size_t nb1, size_t nb2) {
    const int tid  = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int j0   = blockIdx.x * K2_T16_MMA_M;
    const int t0   = blockIdx.y * K2_T16_MMA_TOKENS;

    const int m  = warp >> 1;
    const int nh = warp & 1;
    const int j  = j0 + m;
    if (j >= J) {
        return;
    }

    // Double buffered: the loop stages tile i+1 into one buffer while the mma reads the other,
    // so ALU (decode + f32->f16 conversion) and tensor-core work overlap instead of alternating
    // behind a barrier -- that alternation was what capped the first versions near 40-50% of the
    // f16-with-f32-accumulate ceiling.
    __shared__ __align__(16) half sA[2][K2_T16_MMA_M][16*24];
    __shared__ __align__(16) half sX[2][K2_T16_MMA_TOKENS][24];
    __shared__ __align__(16) float sY[K2_T16_MMA_WARPS][256];   // per-warp epilogue tile

    const int nw = 8*K;
    const int tok_base = t0 + nh*64;
    const int ntok     = min(64, n - tok_base);

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> fa;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> fb[K2_T16_MMA_NFRAG];
    wmma::fragment<wmma::accumulator, 16, 16, 16, float>              fc[K2_T16_MMA_NFRAG];
#pragma unroll
    for (int f = 0; f < K2_T16_MMA_NFRAG; ++f) {
        wmma::fill_fragment(fc[f], 0.0f);
    }

    k2_t16_mma_stage(trellis, x, &sA[0][0][0], &sX[0][0][0], K, I, n, nb1, nb2, 0, j0, t0, tid, nw);
    __syncthreads();

    for (int i = 0; i < I; ++i) {
        const int cur = i & 1;
        const int nxt = cur ^ 1;
        if (i + 1 < I) {
            k2_t16_mma_stage(trellis, x, &sA[nxt][0][0], &sX[nxt][0][0], K, I, n, nb1, nb2,
                             i + 1, j0, t0, tid, nw);
        }
        if (ntok > 0) {
            wmma::load_matrix_sync(fa, &sA[cur][m][0], 24);
#pragma unroll
            for (int f = 0; f < K2_T16_MMA_NFRAG; ++f) {
                if (f*16 < ntok) {
                    wmma::load_matrix_sync(fb[f], &sX[cur][nh*64 + f*16][0], 24);
                    wmma::mma_sync(fc[f], fa, fb[f], fc[f]);
                }
            }
        }
        __syncthreads();
    }

    if (ntok > 0) {
        float * sYw = sY[warp];
#pragma unroll
        for (int f = 0; f < K2_T16_MMA_NFRAG; ++f) {
            if (f*16 >= ntok) {
                continue;
            }
            wmma::store_matrix_sync(sYw, fc[f], 16, wmma::mem_row_major);
            __syncwarp();
            const int ftok = min(16, ntok - f*16);
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const int idx = lane + 32*e;
                const int mm  = idx >> 4;
                const int t   = idx & 15;
                if (t < ftok) {
                    y[(16*j + mm) + (size_t)(tok_base + f*16 + t)*(size_t)(16*J)] = sYw[idx];
                }
            }
            __syncwarp();
        }
    }
}

static bool k2_t16_mma_enabled() {
    static int cached = -1;
    if (cached < 0) {
        const char * e = getenv("K2_T16_NOMMA");
        cached = (e != nullptr && e[0] == '1') ? 0 : 1;
    }
    return cached != 0;
}

// ---------------------------------------------------------------------------
// Fast decode GEMV (design §1/§2): one warp per k-split, 2 out-tiles per block,
// quad window extraction + half2 decode + HFMA2 dot, fp32 fold every 4 k-steps.
// Correctness oracle = the simple kernel above; enable with K2_T16_FAST=1.
// K <= 5 only (the single-funnel quad form is exact there; K=6 uses the simple path).
// ---------------------------------------------------------------------------

// Per-K instances (template-instances/mmvq-instance-t16-k*.cu). The dispatch stays here; the
// instantiation of each rate lives in its own TU, which is the upstream convention.
void k2_t16_gemv_k2(int nw, int pf, dim3 grid, dim3 block, cudaStream_t stream,
                    const uint16_t * tw, const float * xf, float * yf, int I, int J, size_t nb1, size_t nb2);
void k2_t16_gemv_k3(int nw, int pf, dim3 grid, dim3 block, cudaStream_t stream,
                    const uint16_t * tw, const float * xf, float * yf, int I, int J, size_t nb1, size_t nb2);
void k2_t16_gemv_k4(int nw, int pf, dim3 grid, dim3 block, cudaStream_t stream,
                    const uint16_t * tw, const float * xf, float * yf, int I, int J, size_t nb1, size_t nb2);
void k2_t16_gemv_k5(int nw, int pf, dim3 grid, dim3 block, cudaStream_t stream,
                    const uint16_t * tw, const float * xf, float * yf, int I, int J, size_t nb1, size_t nb2);
void k2_t16_gemv_k6(int nw, int pf, dim3 grid, dim3 block, cudaStream_t stream,
                    const uint16_t * tw, const float * xf, float * yf, int I, int J, size_t nb1, size_t nb2);

// --- MoE helpers -----------------------------------------------------------------
// 128-point Walsh-Hadamard (same as the graph's hadamard: FWHT * 1/sqrt(128)),
// one warp per 128 values, 4 values per lane.
static __device__ __forceinline__ void k2_fwht128(float v[4], int lane) {
#pragma unroll
    for (int h = 1; h <= 2; h <<= 1) {
#pragma unroll
        for (int i = 0; i < 4; i += 2 * h) {
            const float a = v[i], b = v[i + h];
            v[i] = a + b;
            v[i + h] = a - b;
        }
    }
#pragma unroll
    for (int h = 4; h < 128; h <<= 1) {
        const int lp = h >> 2;
        const bool up = ((lane & lp) == 0);
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const float o = __shfl_xor_sync(0xFFFFFFFFu, v[k], lp);
            v[k] = up ? (v[k] + o) : (o - v[k]);
        }
    }
    const float sc = 0.08838834764831845f;   // 1/sqrt(128)
#pragma unroll
    for (int k = 0; k < 4; ++k) v[k] *= sc;
}

static __global__ void k2_moe_prep_kernel(const float * __restrict__ x,
                                          const float * __restrict__ su,      // may be null (f32)
                                          const int32_t * __restrict__ ids,
                                          int64_t ids_nb1,
                                          float * __restrict__ xp,
                                          int in, int n_used, int nwarp,
                                          int64_t x_nb1, int64_t su_nb1) {
    const int gwarp = int((blockIdx.x * blockDim.x + threadIdx.x) >> 5);
    if (gwarp >= nwarp) return;
    const int lane = int(threadIdx.x & 31);
    const int blocks = in >> 7;
    const int blk = gwarp % blocks;
    const int tu = gwarp / blocks;
    const int u = tu % n_used, t = tu / n_used;
    const int32_t e = *(const int32_t *)((const char *) ids + (int64_t) t * ids_nb1 + (int64_t) u * 4);
    const int base = (blk << 7) + (lane << 2);
    float v[4];
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const int i = base + k;
        float xv = x[t * x_nb1 + i];
        if (su) {
            xv *= *(const float *)((const char *) su + (int64_t) e * su_nb1 + (int64_t) i * 4);
        }
        v[k] = xv;
    }
    k2_fwht128(v, lane);
#pragma unroll
    for (int k = 0; k < 4; ++k) xp[(int64_t) tu * in + base + k] = v[k];
}

static __global__ void k2_t16_moe_kernel(const uint16_t * __restrict__ tr,
                                         const float    * __restrict__ xp,    // [in, n_used * n]
                                         const int32_t  * __restrict__ ids,
                                         float          * __restrict__ y,
                                         int K, int I, int J, int n_used,
                                         int64_t ids_nb1, size_t nb1, size_t nb2, size_t nb3,
                                         int64_t y_nb1, int64_t y_nb2) {
    const int j    = blockIdx.x;
    const int u    = blockIdx.y;
    const int t    = blockIdx.z;
    const int tid  = threadIdx.x;
    const int c    = tid >> 4;
    const int r    = tid & 15;
    const int q    = 16 * r + c;
    const int tpos = c_k2_t16_inv_perm[q];
    const int b    = tpos * K + K - 16 + 256 * K;
    const int ih   = b >> 5, il = (b + 15) >> 5;
    const int sh   = (il + 1) * 32 - (b + 16);
    const int nw   = 8 * K;

    const int32_t e = *(const int32_t *)((const char *) ids + (int64_t) t * ids_nb1 + (int64_t) u * 4);
    const char * tile0 = (const char *) tr + (size_t) e * nb3 + (size_t) j * nb1;
    const float * xcol = xp + (int64_t)(t * n_used + u) * (16 * I);

    float acc = 0.0f;
    for (int i = 0; i < I; ++i) {
        const uint32_t * p = (const uint32_t *)(tile0 + (size_t) i * nb2);
        const uint32_t w1  = k2_t16_fshr(p[il % nw], p[ih % nw], (uint32_t) sh);
        acc += k2_t16_mul1((uint16_t)(w1 & 0xFFFFu)) * xcol[16 * i + r];
    }
#pragma unroll
    for (int off = 8; off > 0; off >>= 1) {
        acc += __shfl_down_sync(0xFFFFFFFFu, acc, off, 16);
    }
    if (r == 0) {
        float * ycol = (float *)((char *) y + (int64_t) u * y_nb1 + (int64_t) t * y_nb2);
        ycol[16 * j + c] = acc;
    }
}

static __global__ void k2_moe_post_kernel(float * __restrict__ y,
                                          const float * __restrict__ sv,      // may be null (f32)
                                          const int32_t * __restrict__ ids,
                                          int64_t ids_nb1,
                                          int out, int n_used, int nwarp,
                                          int64_t y_nb1, int64_t y_nb2, int64_t sv_nb1) {
    const int gwarp = int((blockIdx.x * blockDim.x + threadIdx.x) >> 5);
    if (gwarp >= nwarp) return;
    const int lane = int(threadIdx.x & 31);
    const int blocks = out >> 7;
    const int blk = gwarp % blocks;
    const int tu = gwarp / blocks;
    const int u = tu % n_used, t = tu / n_used;
    const int32_t e = *(const int32_t *)((const char *) ids + (int64_t) t * ids_nb1 + (int64_t) u * 4);
    const int base = (blk << 7) + (lane << 2);
    float * ycol = (float *)((char *) y + (int64_t) u * y_nb1 + (int64_t) t * y_nb2);
    float v[4];
#pragma unroll
    for (int k = 0; k < 4; ++k) v[k] = ycol[base + k];
    k2_fwht128(v, lane);
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        float val = v[k];
        if (sv) {
            val *= *(const float *)((const char *) sv + (int64_t) e * sv_nb1 + (int64_t) (base + k) * 4);
        }
        ycol[base + k] = val;
    }
}

void ggml_cuda_op_t16_mul_mat_id(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * w   = dst->src[0];
    const ggml_tensor * x   = dst->src[1];
    const ggml_tensor * ids = dst->src[2];
    const ggml_tensor * su  = dst->src[3];
    const ggml_tensor * sv  = dst->src[4];

    GGML_ASSERT(w->type == GGML_TYPE_I16 && ggml_is_contiguous(w));
    GGML_ASSERT(x->type == GGML_TYPE_F32 && ggml_is_contiguous(x));
    GGML_ASSERT(ids->type == GGML_TYPE_I32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(!su || su->type == GGML_TYPE_F32);
    GGML_ASSERT(!sv || sv->type == GGML_TYPE_F32);

    const int K = (int) (w->ne[0] / 16);
    const int J = (int) w->ne[1];
    const int I = (int) w->ne[2];
    const int E = (int) w->ne[3];
    const int in = 16 * I, out = 16 * J;
    const int n = (int) x->ne[1];
    const int n_used = (int) ids->ne[0];
    GGML_ASSERT(K >= 2 && K <= 6);
    GGML_ASSERT(x->ne[0] == in && dst->ne[0] == out && dst->ne[1] == n_used && dst->ne[2] == n);
    GGML_ASSERT(in % 128 == 0 && out % 128 == 0);

    cudaStream_t stream = ctx.stream();
    const int threads = 256;

    // xp = H(x[:,t] * su_e) per (t,u)
    ggml_cuda_pool & pool = ctx.pool();
    ggml_cuda_pool_alloc<float> xp(pool, (size_t) n * n_used * in);
    const int warps_in = n * n_used * (in / 128);
    k2_moe_prep_kernel<<<(warps_in * 32 + threads - 1) / threads, threads, 0, stream>>>(
            (const float *) x->data,
            su ? (const float *) su->data : nullptr,
            (const int32_t *) ids->data, ids->nb[1],
            xp.get(), in, n_used, warps_in,
            x->nb[1] / sizeof(float), su ? su->nb[1] / 4 : 0);

    // y = w_e^T xp (one device launch; reads ids on the GPU)
    (void) E;
    k2_t16_moe_kernel<<<dim3(J, n_used, n), 256, 0, stream>>>(
            (const uint16_t *) w->data, xp.get(), (const int32_t *) ids->data, (float *) dst->data,
            K, I, J, n_used, ids->nb[1], w->nb[1], w->nb[2], w->nb[3], dst->nb[1], dst->nb[2]);

    // y = sv_e * H(y), in place
    const int warps_out = n * n_used * (out / 128);
    k2_moe_post_kernel<<<(warps_out * 32 + threads - 1) / threads, threads, 0, stream>>>(
            (float *) dst->data,
            sv ? (const float *) sv->data : nullptr,
            (const int32_t *) ids->data, ids->nb[1],
            out, n_used, warps_out,
            dst->nb[1], dst->nb[2], sv ? sv->nb[1] / 4 : 0);

    CUDA_CHECK(cudaGetLastError());
}

void ggml_cuda_op_t16_matmul(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * w = dst->src[0];
    const ggml_tensor * x = dst->src[1];

    GGML_ASSERT(w->type == GGML_TYPE_I16 && w->ne[3] == 1);
    GGML_ASSERT(ggml_is_contiguous(w));
    GGML_ASSERT(x->type == GGML_TYPE_F32 && ggml_is_contiguous(x));
    GGML_ASSERT(dst->type == GGML_TYPE_F32 && ggml_is_contiguous(dst));

    const int K = (int) (w->ne[0] / 16);
    const int J = (int) w->ne[1];
    const int I = (int) w->ne[2];
    const int n = (int) x->ne[1];
    GGML_ASSERT(K >= 2 && K <= 6);
    GGML_ASSERT((int) w->ne[0] == 16*K);
    GGML_ASSERT(x->ne[0] == 16*I && dst->ne[0] == 16*J && dst->ne[1] == n);

    cudaStream_t stream = ctx.stream();
    if (n == 1 && K >= 2 && K <= 6 && J % 2 == 0 && k2_t16_fast_enabled()) {
        const int nw = k2_t16_nw();
        const int pf = k2_t16_pf();
        const dim3 grid(J / 2);
        const dim3 block(32, nw);
        const uint16_t * tw = (const uint16_t *) w->data;
        const float    * xf = (const float *) x->data;
        float          * yf = (float *) dst->data;
        if      (K == 2) k2_t16_gemv_k2(nw, pf, grid, block, stream, tw, xf, yf, I, J, w->nb[1], w->nb[2]);
        else if (K == 3) k2_t16_gemv_k3(nw, pf, grid, block, stream, tw, xf, yf, I, J, w->nb[1], w->nb[2]);
        else if (K == 4) k2_t16_gemv_k4(nw, pf, grid, block, stream, tw, xf, yf, I, J, w->nb[1], w->nb[2]);
        else {
            // wide path (K=5/6): two words per lane, block shape (32, nw)
            const dim3 blkW(32, nw);
            if (K == 5) k2_t16_gemv_k5(nw, pf, grid, blkW, stream, tw, xf, yf, I, J, w->nb[1], w->nb[2]);
            else        k2_t16_gemv_k6(nw, pf, grid, blkW, stream, tw, xf, yf, I, J, w->nb[1], w->nb[2]);
        }
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    if (n > 1 && k2_t16_mma_enabled()) {
        dim3 grid((J + K2_T16_MMA_M - 1) / K2_T16_MMA_M,
                  (n + K2_T16_MMA_TOKENS - 1) / K2_T16_MMA_TOKENS);
        k2_t16_mma_kernel<<<grid, 32*K2_T16_MMA_WARPS, 0, stream>>>(
                (const uint16_t *) w->data, (const float *) x->data, (float *) dst->data,
                K, I, J, n, w->nb[1], w->nb[2]);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    dim3 grid(J, (n + K2_T16_TOK_BLK - 1) / K2_T16_TOK_BLK);
    k2_t16_matmul_kernel<<<grid, 256, 0, stream>>>(
            (const uint16_t *) w->data, (const float *) x->data, (float *) dst->data,
            K, I, J, n, w->nb[1], w->nb[2]);
    CUDA_CHECK(cudaGetLastError());
}

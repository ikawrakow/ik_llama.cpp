#include "hadamard.cuh"
#include "dequantize.cuh"

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#include <intrin.h>
#include <ammintrin.h>
#include <nmmintrin.h>
#include <immintrin.h>
#include <stdlib.h>
static inline int popcount(uint32_t x) { return __popcnt(x); }
#else
static inline int popcount(uint32_t x) { return __builtin_popcount(x); }
#endif

template <int nh>
static __device__ __forceinline__ void hadamard_butterfly(float * ys, int tid, float & scale) {
    constexpr float ksqrt2 = 0.707106781f;
    #pragma unroll
    for (int h = 2; h < nh; h <<= 1) {
        __syncthreads();
        const int ii = tid/h, jj = tid%h;
        const int j  = 2*h*ii + jj;
        const float u = ys[j], v = ys[j+h];
        ys[j+0] = u + v;
        ys[j+h] = u - v;
        scale *= ksqrt2;
    }
}

template <int nh>
static __global__ void hadamard_f32(const char * src, char * dst, int ne0,
        size_t nb01, size_t nb02, size_t nb03, size_t nb1, size_t nb2, size_t nb3) {

    constexpr float ksqrt2 = 0.707106781f;
    const int nc  = ne0/nh;
    const int ii1 = blockIdx.x;
    const int i1  = ii1 / nc;
    const int ic  = ii1 % nc;
    const int i2  = blockIdx.y;
    const int i3  = blockIdx.z;
    const int tid = threadIdx.x;

    const float * x = (const float *)((const char *)src + i1*nb01 + i2*nb02 + i3*nb03) + ic*nh;
          float * y = (      float *)((const char *)dst + i1*nb1  + i2*nb2  + i3*nb3)  + ic*nh;

    __shared__ float ys[nh];
    ys[2*tid+0] = x[2*tid+0] + x[2*tid+1];
    ys[2*tid+1] = x[2*tid+0] - x[2*tid+1];
    float scale = ksqrt2;

    hadamard_butterfly<nh>(ys, tid, scale);

    __syncthreads();
    y[2*tid+0] = ys[2*tid+0] * scale;
    y[2*tid+1] = ys[2*tid+1] * scale;
}

template <int nh>
static __global__ void hadamard_f16(const char * src, char * dst, int ne0,
        size_t nb01, size_t nb02, size_t nb03, size_t nb1, size_t nb2, size_t nb3) {

    constexpr float ksqrt2 = 0.707106781f;
    const int nc  = ne0/nh;
    const int ii1 = blockIdx.x;
    const int i1  = ii1 / nc;
    const int ic  = ii1 % nc;
    const int i2  = blockIdx.y;
    const int i3  = blockIdx.z;
    const int tid = threadIdx.x;

    const half * x = (const half *)((const char *)src + i1*nb01 + i2*nb02 + i3*nb03) + ic*nh;
    float      * y = (      float *)((const char *)dst + i1*nb1  + i2*nb2  + i3*nb3)  + ic*nh;

    __shared__ float ys[nh];
    const float a = __half2float(x[2*tid + 0]);
    const float b = __half2float(x[2*tid + 1]);
    ys[2*tid + 0] = a + b;
    ys[2*tid + 1] = a - b;
    float scale = ksqrt2;

    hadamard_butterfly<nh>(ys, tid, scale);

    __syncthreads();
    y[2*tid + 0] = ys[2*tid + 0] * scale;
    y[2*tid + 1] = ys[2*tid + 1] * scale;
}

template <int nh, int qk, void (*dequant)(const void *, int64_t, int, dfloat2&), bool qr2>
static __global__ void hadamard_quant(const char * src, char * dst, int ne0,
        size_t nb01, size_t nb02, size_t nb03, size_t nb1, size_t nb2, size_t nb3) {

    constexpr float ksqrt2 = 0.707106781f;
    const int nc  = ne0/nh;
    const int ii1 = blockIdx.x;
    const int i1  = ii1 / nc;
    const int ic  = ii1 % nc;
    const int i2  = blockIdx.y;
    const int i3  = blockIdx.z;
    const int tid = threadIdx.x;

    const void * row_src = (const char *)src + i1*nb01 + i2*nb02 + i3*nb03;
    float * y = (float *)((const char *)dst + i1*nb1 + i2*nb2 + i3*nb3) + ic*nh;

    __shared__ float ys[nh];
    float scale = ksqrt2;

    if (!qr2) {
        const int abs_off = ic*nh + 2*tid;
        const int ib      = abs_off / qk;
        const int iqs     = abs_off % qk;
        dfloat2 v;
        dequant(row_src, ib, iqs, v);
        ys[2*tid + 0] = (float)v.x + (float)v.y;
        ys[2*tid + 1] = (float)v.x - (float)v.y;
    } else {
        constexpr int qk_half = qk/2;
        const int b     = tid / qk_half;
        const int iqs   = tid % qk_half;
        const int ib    = ic*(nh/qk) + b;
        dfloat2 v;
        dequant(row_src, ib, iqs, v);
        ys[b*qk + iqs + 0      ] = (float)v.x;
        ys[b*qk + iqs + qk_half] = (float)v.y;
        __syncthreads();
        const float a = ys[2*tid + 0];
        const float c = ys[2*tid + 1];
        __syncthreads();
        ys[2*tid + 0] = a + c;
        ys[2*tid + 1] = a - c;
    }

    hadamard_butterfly<nh>(ys, tid, scale);

    __syncthreads();
    y[2*tid + 0] = ys[2*tid + 0] * scale;
    y[2*tid + 1] = ys[2*tid + 1] * scale;
}

static void hadamard_f32_cuda(int nh, const char * x, char * y, int ne0, int ne1, int ne2, int ne3,
        size_t nb01, size_t nb02, size_t nb03, size_t nb1, size_t nb2, size_t nb3, cudaStream_t stream) {
    int nc = ne0/nh;
    int nrows = nc*ne1;
    dim3 num_blocks = dim3(nrows, ne2, ne3);
    switch (nh) {
        case  64: hadamard_f32< 64><<<num_blocks,  32, 0, stream>>>(x, y, ne0, nb01, nb02, nb03, nb1, nb2, nb3); break;
        case 128: hadamard_f32<128><<<num_blocks,  64, 0, stream>>>(x, y, ne0, nb01, nb02, nb03, nb1, nb2, nb3); break;
        case 256: hadamard_f32<256><<<num_blocks, 128, 0, stream>>>(x, y, ne0, nb01, nb02, nb03, nb1, nb2, nb3); break;
        case 512: hadamard_f32<512><<<num_blocks, 256, 0, stream>>>(x, y, ne0, nb01, nb02, nb03, nb1, nb2, nb3); break;
        default: GGML_ABORT("Unsupported Hadamard block size");
    }
}

#define LAUNCH_HADAMARD_F16(NH) \
    hadamard_f16<NH><<<num_blocks, NH/2, 0, stream>>>( \
            (const char *)src->data, (char *)dst->data, src->ne[0], \
            src->nb[1], src->nb[2], src->nb[3], dst->nb[1], dst->nb[2], dst->nb[3])

#define DISPATCH_HADAMARD_F16_NH \
    switch (nh) { \
        case  64: LAUNCH_HADAMARD_F16( 64); break; \
        case 128: LAUNCH_HADAMARD_F16(128); break; \
        case 256: LAUNCH_HADAMARD_F16(256); break; \
        case 512: LAUNCH_HADAMARD_F16(512); break; \
        default: GGML_ABORT("Unsupported Hadamard block size"); \
    }

#define LAUNCH_HADAMARD_QUANT(NH, DEQUANT, QK, QR2) \
    hadamard_quant<NH, QK, DEQUANT, QR2><<<num_blocks, NH/2, 0, stream>>>( \
            (const char *)src->data, (char *)dst->data, src->ne[0], \
            src->nb[1], src->nb[2], src->nb[3], dst->nb[1], dst->nb[2], dst->nb[3])

#define DISPATCH_HADAMARD_QUANT_NH(DEQUANT, QK, QR2) \
    switch (nh) { \
        case  64: LAUNCH_HADAMARD_QUANT( 64, DEQUANT, QK, QR2); break; \
        case 128: LAUNCH_HADAMARD_QUANT(128, DEQUANT, QK, QR2); break; \
        case 256: LAUNCH_HADAMARD_QUANT(256, DEQUANT, QK, QR2); break; \
        case 512: LAUNCH_HADAMARD_QUANT(512, DEQUANT, QK, QR2); break; \
        default: GGML_ABORT("Unsupported Hadamard block size"); \
    }

void ggml_cuda_op_hadamard(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src = dst->src[0];
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_are_same_shape(src, dst));

    const int nh = dst->op_params[0];
    GGML_ASSERT(dst->ne[0]%nh == 0);
    GGML_ASSERT(nh > 1 && popcount(nh) == 1);

    cudaStream_t stream = ctx.stream();

    if (src->type == GGML_TYPE_F32) {
        hadamard_f32_cuda(nh, (const char *)src->data, (char *)dst->data,
                src->ne[0], src->ne[1], src->ne[2], src->ne[3],
                src->nb[1], src->nb[2], src->nb[3], dst->nb[1], dst->nb[2], dst->nb[3], stream);
        return;
    }

    dim3 num_blocks((src->ne[0]/nh) * src->ne[1], src->ne[2], src->ne[3]);

    switch (src->type) {
        case GGML_TYPE_F16:    DISPATCH_HADAMARD_F16_NH;                                   break;
        case GGML_TYPE_Q8_0:   DISPATCH_HADAMARD_QUANT_NH(dequantize_q8_0,   QK8_0,  false); break;
        case GGML_TYPE_Q4_0:   DISPATCH_HADAMARD_QUANT_NH(dequantize_q4_0,   QK4_0,  true);  break;
        case GGML_TYPE_Q4_1:   DISPATCH_HADAMARD_QUANT_NH(dequantize_q4_1,   QK4_1,  true);  break;
        case GGML_TYPE_Q5_0:   DISPATCH_HADAMARD_QUANT_NH(dequantize_q5_0,   QK5_0,  true);  break;
        case GGML_TYPE_Q5_1:   DISPATCH_HADAMARD_QUANT_NH(dequantize_q5_1,   QK5_1,  true);  break;
        case GGML_TYPE_Q6_0:   DISPATCH_HADAMARD_QUANT_NH(dequantize_q6_0,   QK6_0,  true);  break;
        case GGML_TYPE_IQ4_NL: DISPATCH_HADAMARD_QUANT_NH(dequantize_iq4_nl, QK4_NL, true);  break;
        default:
            GGML_ABORT("hadamard: unsupported source type");
    }
}

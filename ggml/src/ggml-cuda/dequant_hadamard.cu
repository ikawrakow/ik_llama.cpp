#include "dequant_hadamard.cuh"
#include "dequantize.cuh"
#include "hadamard.cuh"

template <int nh, int qk, void (*dequant)(const void *, int64_t, int, dfloat2&), bool qr2>
static __global__ void dequant_hadamard_kernel(const char * src, char * dst, int ne0,
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

    #pragma unroll
    for (int h = 2; h < nh; h <<= 1) {
        __syncthreads();
        const int ii = tid/h;
        const int jj = tid%h;
        const int j  = 2*h*ii + jj;
        const float u = ys[j], w = ys[j+h];
        ys[j+0] = u + w;
        ys[j+h] = u - w;
        scale *= ksqrt2;
    }

    __syncthreads();
    y[2*tid + 0] = ys[2*tid + 0] * scale;
    y[2*tid + 1] = ys[2*tid + 1] * scale;
}

template <int nh>
static __global__ void dequant_hadamard_kernel_f16(const char * src, char * dst, int ne0,
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

    #pragma unroll
    for (int h = 2; h < nh; h <<= 1) {
        __syncthreads();
        const int ii = tid/h;
        const int jj = tid%h;
        const int j  = 2*h*ii + jj;
        const float u = ys[j], w = ys[j+h];
        ys[j+0] = u + w;
        ys[j+h] = u - w;
        scale *= ksqrt2;
    }

    __syncthreads();
    y[2*tid + 0] = ys[2*tid + 0] * scale;
    y[2*tid + 1] = ys[2*tid + 1] * scale;
}

#define LAUNCH_DEQHAD_QR(NH, DEQUANT, QK, QR2) \
    dequant_hadamard_kernel<NH, QK, DEQUANT, QR2><<<num_blocks, NH/2, 0, stream>>>( \
            (const char *)src->data, (char *)dst->data, src->ne[0], \
            src->nb[1], src->nb[2], src->nb[3], dst->nb[1], dst->nb[2], dst->nb[3])

#define DISPATCH_DEQHAD_NH(DEQUANT, QK, QR2) \
    switch (nh) { \
        case  64: LAUNCH_DEQHAD_QR( 64, DEQUANT, QK, QR2); break; \
        case 128: LAUNCH_DEQHAD_QR(128, DEQUANT, QK, QR2); break; \
        case 256: LAUNCH_DEQHAD_QR(256, DEQUANT, QK, QR2); break; \
        case 512: LAUNCH_DEQHAD_QR(512, DEQUANT, QK, QR2); break; \
        default: GGML_ABORT("Unsupported Hadamard block size"); \
    }

void ggml_cuda_op_dequant_hadamard(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src = dst->src[0];
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    const int nh = dst->op_params[0];
    GGML_ASSERT(nh == 64 || nh == 128 || nh == 256 || nh == 512);
    GGML_ASSERT(src->ne[0] % nh == 0);

    if (src->type == GGML_TYPE_F32) {
        ggml_cuda_op_hadamard(ctx, dst);
        return;
    }

    cudaStream_t stream = ctx.stream();
    dim3 num_blocks((src->ne[0]/nh) * src->ne[1], src->ne[2], src->ne[3]);

    switch (src->type) {
        case GGML_TYPE_F16: {
            switch (nh) {
                case  64: dequant_hadamard_kernel_f16< 64><<<num_blocks,  32, 0, stream>>>((const char *)src->data, (char *)dst->data, src->ne[0], src->nb[1], src->nb[2], src->nb[3], dst->nb[1], dst->nb[2], dst->nb[3]); break;
                case 128: dequant_hadamard_kernel_f16<128><<<num_blocks,  64, 0, stream>>>((const char *)src->data, (char *)dst->data, src->ne[0], src->nb[1], src->nb[2], src->nb[3], dst->nb[1], dst->nb[2], dst->nb[3]); break;
                case 256: dequant_hadamard_kernel_f16<256><<<num_blocks, 128, 0, stream>>>((const char *)src->data, (char *)dst->data, src->ne[0], src->nb[1], src->nb[2], src->nb[3], dst->nb[1], dst->nb[2], dst->nb[3]); break;
                case 512: dequant_hadamard_kernel_f16<512><<<num_blocks, 256, 0, stream>>>((const char *)src->data, (char *)dst->data, src->ne[0], src->nb[1], src->nb[2], src->nb[3], dst->nb[1], dst->nb[2], dst->nb[3]); break;
                default: GGML_ABORT("Unsupported Hadamard block size");
            }
        } break;
        case GGML_TYPE_Q8_0:   DISPATCH_DEQHAD_NH(dequantize_q8_0,   QK8_0, false); break;
        case GGML_TYPE_Q4_0:   DISPATCH_DEQHAD_NH(dequantize_q4_0,   QK4_0, true);  break;
        case GGML_TYPE_Q4_1:   DISPATCH_DEQHAD_NH(dequantize_q4_1,   QK4_1, true);  break;
        case GGML_TYPE_Q5_0:   DISPATCH_DEQHAD_NH(dequantize_q5_0,   QK5_0, true);  break;
        case GGML_TYPE_Q5_1:   DISPATCH_DEQHAD_NH(dequantize_q5_1,   QK5_1, true);  break;
        case GGML_TYPE_Q6_0:   DISPATCH_DEQHAD_NH(dequantize_q6_0,   QK6_0, true);  break;
        case GGML_TYPE_IQ4_NL: DISPATCH_DEQHAD_NH(dequantize_iq4_nl, QK4_NL, true); break;
        default:
            GGML_ABORT("dequant_hadamard: unsupported source type");
    }
}

#include "cpy.cuh"
#include "convert.cuh"

typedef void (*cpy_kernel_t)(const char * cx, char * cdst);

static __device__ void cpy_1_f32_f32(const char * cxi, char * cdsti) {
    const float * xi = (const float *) cxi;
    float * dsti = (float *) cdsti;

    *dsti = *xi;
}

static __device__ void cpy_1_f32_f16(const char * cxi, char * cdsti) {
    const float * xi = (const float *) cxi;
    half * dsti = (half *) cdsti;

    *dsti = __float2half(*xi);
}

static __device__ void cpy_1_f32_bf16(const char * cxi, char * cdsti) {
    const float * xi = (const float *) cxi;
    nv_bfloat16 * dsti = (nv_bfloat16 *) cdsti;

    *dsti = __float2bfloat16(*xi);
}

static __device__ void cpy_1_f16_f16(const char * cxi, char * cdsti) {
    const half * xi = (const half *) cxi;
    half * dsti = (half *) cdsti;

    *dsti = *xi;
}

static __device__ void cpy_1_f16_f32(const char * cxi, char * cdsti) {
    const half * xi = (const half *) cxi;
    float * dsti = (float *) cdsti;

    *dsti = *xi;
}

template <cpy_kernel_t cpy_1>
static __global__ void cpy_f32_f16(const char * cx, char * cdst, const int ne,
                                   const int ne00, const int ne01, const int ne02, const int nb00, const int nb01, const int nb02,
                                   const int nb03, const int ne10, const int ne11, const int ne12, const int nb10, const int nb11,
                                   const int nb12, const int nb13) {
    const int64_t i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= ne) {
        return;
    }

    // determine indices i03/i13, i02/i12, i01/i11, i00/i10 as a function of index i of flattened tensor
    // then combine those indices with the corresponding byte offsets to get the total offsets
    const int64_t i03 = i/(ne00 * ne01 * ne02);
    const int64_t i02 = (i - i03*ne00*ne01*ne02 )/ (ne00*ne01);
    const int64_t i01 = (i - i03*ne00*ne01*ne02  -  i02*ne01*ne00) / ne00;
    const int64_t i00 = i - i03*ne00*ne01*ne02 - i02*ne01*ne00 - i01*ne00;
    const int64_t x_offset = i00*nb00 + i01*nb01 + i02*nb02 + i03 * nb03;

    const int64_t i13 = i/(ne10 * ne11 * ne12);
    const int64_t i12 = (i - i13*ne10*ne11*ne12) / (ne10*ne11);
    const int64_t i11 = (i - i13*ne10*ne11*ne12 - i12*ne10*ne11) / ne10;
    const int64_t i10 = i - i13*ne10*ne11*ne12 - i12*ne10*ne11 - i11*ne10;
    const int64_t dst_offset = i10*nb10 + i11*nb11 + i12*nb12 + i13 * nb13;

    cpy_1(cx + x_offset, cdst + dst_offset);
}

template <typename dst_t>
static __global__ void k_cpy_q8_0_to_float(const char * cx, dst_t * dst, const int ne,
                                   const int ne00, const int ne01, const int ne02, const int nb01, const int nb02, const int nb03) {
    const int64_t i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= ne) {
        return;
    }

    const int64_t i03 = i/(ne00 * ne01 * ne02);
    const int64_t i02 = (i - i03*ne00*ne01*ne02) / (ne00*ne01);
    const int64_t i01 = (i - i03*ne00*ne01*ne02 - i02*ne00*ne01) / ne00;
    const int64_t i00 = i - i03*ne00*ne01*ne02 - i02*ne00*ne01 - i01*ne00;

    const block_q8_0 * q8 = (const block_q8_0 *)(cx + i01*nb01 + i02*nb02 + i03*nb03);
    const int ib = i00/QK8_0;
    const int iq = i00%QK8_0;

    if constexpr (std::is_same_v<dst_t, nv_bfloat16>) {
        dst[i00 + i01*ne00 + i02*ne00*ne01 + i03*ne00*ne01*ne02] = __float2bfloat16(__half2float(q8[ib].d)*q8[ib].qs[iq]);
    } else {
        dst[i00 + i01*ne00 + i02*ne00*ne01 + i03*ne00*ne01*ne02] = __half2float(q8[ib].d)*q8[ib].qs[iq];
    }
}

static __global__ void k_transpose_q8_0(const char * cx, char * cdst,
                                   const int ne10, const int ne11, const int ne12,
                                   const int nb01, const int nb02, const int nb03,
                                   const int nb11, const int nb12, const int nb13) {
    const int64_t i = blockDim.x*blockIdx.x + threadIdx.x;

    const int64_t i13 = i/(ne10 * ne11 * ne12);
    const int64_t i12 = (i - i13*ne10*ne11*ne12) / (ne10*ne11);
    const int64_t i11 = (i - i13*ne10*ne11*ne12 - i12*ne10*ne11) / ne10;
    const int64_t i10 = i - i13*ne10*ne11*ne12 - i12*ne10*ne11 - i11*ne10;

    //const int64_t ne00 = ne11;
    //const int64_t ne01 = ne10;
    //const int64_t ne02 = ne12;
    const int64_t i03 = i13;
    const int64_t i02 = i12;
    const int64_t i01 = i10; //(i - i03*ne00*ne01*ne02 - i02*ne00*ne01) / ne00;
    const int64_t i00 = i11; //i - i03*ne00*ne01*ne02 - i02*ne00*ne01 - i01*ne00;

    const block_q8_0 * q8 = (const block_q8_0 *)(cx + i01*nb01 + i02*nb02 + i03*nb03);
    const int ib0 = i00/QK8_0;
    const int iq0 = i00%QK8_0;

    float xi = __half2float(q8[ib0].d)*q8[ib0].qs[iq0];
    float amax = fabsf(xi);
    amax = warp_reduce_max(amax);

    //printf("%d, %d, %d: i = %ld, i11 = %ld i10 = %ld, xi = %g, amax = %g\n", blockDim.x, blockIdx.x, threadIdx.x, i, i11, i10, xi, amax);

    float d = amax/127;
    int8_t q = amax == 0.0f ? 0 : roundf(xi / d);

    block_q8_0 * dst = (block_q8_0 *)(cdst + i11*nb11 + i12*nb12 + i13*nb13);
    dst[i10 / QK8_0].qs[i10 % QK8_0] = q;

    if (threadIdx.x == 0) {
        dst[i10 / QK8_0].d = __float2half(d);
    }
}

static __device__ void cpy_blck_f32_q8_0(const char * cxi, char * cdsti) {
    const float * xi = (const float *) cxi;
    block_q8_0 * dsti = (block_q8_0 *) cdsti;

    float amax = 0.0f; // absolute max

    for (int j = 0; j < QK8_0; j++) {
        const float v = xi[j];
        amax = fmaxf(amax, fabsf(v));
    }

    const float d = amax / ((1 << 7) - 1);
    const float id = d ? 1.0f/d : 0.0f;

    dsti->d = d;

    for (int j = 0; j < QK8_0; ++j) {
        const float x0 = xi[j]*id;

        dsti->qs[j] = roundf(x0);
    }
}

static __device__ void cpy_blck_f32_q4_0(const char * cxi, char * cdsti) {
    const float * xi = (const float *) cxi;
    block_q4_0 * dsti = (block_q4_0 *) cdsti;

    float amax = 0.0f;
    float vmax = 0.0f;

    for (int j = 0; j < QK4_0; ++j) {
        const float v = xi[j];
        if (amax < fabsf(v)) {
            amax = fabsf(v);
            vmax = v;
        }
    }

    const float d  = vmax / -8;
    const float id = d ? 1.0f/d : 0.0f;

    dsti->d = d;

    for (int j = 0; j < QK4_0/2; ++j) {
        const float x0 = xi[0       + j]*id;
        const float x1 = xi[QK4_0/2 + j]*id;

        const uint8_t xi0 = min(15, (int8_t)(x0 + 8.5f));
        const uint8_t xi1 = min(15, (int8_t)(x1 + 8.5f));

        dsti->qs[j]  = xi0;
        dsti->qs[j] |= xi1 << 4;
    }
}

static __device__ void cpy_blck_f32_q4_1(const char * cxi, char * cdsti) {
    const float * xi = (const float *) cxi;
    block_q4_1 * dsti = (block_q4_1 *) cdsti;

    float vmin = FLT_MAX;
    float vmax = -FLT_MAX;

    for (int j = 0; j < QK4_1; ++j) {
        const float v = xi[j];

        if (v < vmin) vmin = v;
        if (v > vmax) vmax = v;
    }

    const float d  = (vmax - vmin) / ((1 << 4) - 1);
    const float id = d ? 1.0f/d : 0.0f;

    dsti->dm.x = d;
    dsti->dm.y = vmin;

    for (int j = 0; j < QK4_1/2; ++j) {
        const float x0 = (xi[0       + j] - vmin)*id;
        const float x1 = (xi[QK4_1/2 + j] - vmin)*id;

        const uint8_t xi0 = min(15, (int8_t)(x0 + 0.5f));
        const uint8_t xi1 = min(15, (int8_t)(x1 + 0.5f));

        dsti->qs[j]  = xi0;
        dsti->qs[j] |= xi1 << 4;
    }
}

static __device__ void cpy_blck_f32_q5_0(const char * cxi, char * cdsti) {
    const float * xi = (const float *) cxi;
    block_q5_0 * dsti = (block_q5_0 *) cdsti;

    float amax = 0.0f;
    float vmax = 0.0f;

    for (int j = 0; j < QK5_0; ++j) {
        const float v = xi[j];
        if (amax < fabsf(v)) {
            amax = fabsf(v);
            vmax = v;
        }
    }

    const float d  = vmax / -16;
    const float id = d ? 1.0f/d : 0.0f;

    dsti->d = d;

    uint32_t qh = 0;
    for (int j = 0; j < QK5_0/2; ++j) {
        const float x0 = xi[0       + j]*id;
        const float x1 = xi[QK5_0/2 + j]*id;

        const uint8_t xi0 = min(31, (int8_t)(x0 + 16.5f));
        const uint8_t xi1 = min(31, (int8_t)(x1 + 16.5f));

        dsti->qs[j]  = (xi0 & 0xf) | ((xi1 & 0xf) << 4);
        qh |= ((xi0 & 0x10u) >> 4) << (j + 0);
        qh |= ((xi1 & 0x10u) >> 4) << (j + QK5_0/2);
    }
    memcpy(dsti->qh, &qh, sizeof(qh));
}

static __device__ void cpy_blck_f32_q5_1(const char * cxi, char * cdsti) {
    const float * xi = (const float *) cxi;
    block_q5_1 * dsti = (block_q5_1 *) cdsti;

    float min = xi[0];
    float max = xi[0];

    for (int j = 1; j < QK5_1; ++j) {
        const float v = xi[j];
        min = v < min ? v : min;
        max = v > max ? v : max;
    }

    const float d  = (max - min) / 31;
    const float id = d ? 1.0f/d : 0.0f;

    dsti->dm.x = d;
    dsti->dm.y = min;

    uint32_t qh = 0;
    for (int j = 0; j < QK5_1/2; ++j) {
        const float x0 = (xi[0       + j] - min)*id;
        const float x1 = (xi[QK5_1/2 + j] - min)*id;

        const uint8_t xi0 = (uint8_t)(x0 + 0.5f);
        const uint8_t xi1 = (uint8_t)(x1 + 0.5f);

        dsti->qs[j]  = (xi0 & 0xf) | ((xi1 & 0xf) << 4);
        qh |= ((xi0 & 0x10u) >> 4) << (j + 0);
        qh |= ((xi1 & 0x10u) >> 4) << (j + QK5_1/2);
    }
    memcpy(dsti->qh, &qh, sizeof(qh));
}

static __device__ void cpy_blck_f32_q6_0(const char * cxi, char * cdsti) {
    const float * xi = (const float *) cxi;
    block_q6_0 * dsti = (block_q6_0 *) cdsti;

    float amax = 0.0f;
    float vmax = 0.0f;

    for (int j = 0; j < QK6_0; ++j) {
        const float v  = xi[j];
        const float av = fabsf(xi[j]);
        if (amax < av) {
            amax = av;
            vmax = v;
        }
    }

    const float d  = vmax / -32;
    const float id = d ? 1.0f/d : 0.0f;

    dsti->d = d;
    memset(dsti->qh, 0, QK6_0/4);

    for (int j = 0; j < QK6_0/2; ++j) {
        const float x0 = xi[0       + j]*id;
        const float x1 = xi[QK4_0/2 + j]*id;

        const uint8_t xi0 = min(63, (int8_t)(x0 + 32.5f));
        const uint8_t xi1 = min(63, (int8_t)(x1 + 32.5f));

        dsti->qs[j]  = (xi0 & 0xf) | ((xi1 & 0xf) << 4);
        const uint8_t h = (xi0 >> 4) | ((xi1 >> 4) << 2);
        dsti->qh[j%(QK6_0/4)] |= (h << 4*(j/(QK6_0/4)));
    }
}

static __device__ const int8_t iq4nl_index[241] = {
     0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 16, 16,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
     1, 17, 17,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2, 18,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,
     3,  3,  3,  3,  3,  3, 19,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4, 20,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,
     5,  5, 21, 21,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6, 22,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7, 23, 23,  8,  8,  8,  8,
     8,  8,  8,  8,  8,  8, 24,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9, 25, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 26, 26,
    11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 27, 27, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 28, 13, 13, 13,
    13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 29, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14,
    14, 14, 14, 14, 30, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15
};
static __device__ __forceinline__ int best_index_iq4nl(const int8_t * values, float x) {
    int ix = (int)x - values[0];
    if (ix < 0 || ix >= 241) return ix < 0 ? 0 : 15;
    ix = iq4nl_index[ix];
    return ix < 16 ? ix : x - values[ix-16] < values[ix-15] - x ? ix-16 : ix-15;
}

static __device__ void cpy_blck_f32_iq4_nl(const char * cxi, char * cdsti) {
    const float * xi = (const float *) cxi;
    block_iq4_nl * dsti = (block_iq4_nl *) cdsti;

    float amax = 0.0f;
    float vmax = 0.0f;

    for (int j = 0; j < QK4_NL; ++j) {
        const float v = xi[j];
        if (amax < fabsf(v)) {
            amax = fabsf(v);
            vmax = v;
        }
    }

    float d = vmax / kvalues_iq4nl[0];
    const float id = d ? 1.0f/d : 0.0f;

    //dsti->d = d;

    float sumqx = 0, sumq2 = 0;
    for (int j = 0; j < QK4_NL/2; ++j) {
        const float x0 = xi[0        + j]*id;
        const float x1 = xi[QK4_NL/2 + j]*id;
        const uint8_t xi0 = best_index_iq4nl(kvalues_iq4nl, x0);
        const uint8_t xi1 = best_index_iq4nl(kvalues_iq4nl, x1);
        dsti->qs[j] = xi0 | (xi1 << 4);
        const float v0 = kvalues_iq4nl[xi0];
        const float v1 = kvalues_iq4nl[xi1];
        const float w0 = xi[0        + j]*xi[0        + j];
        const float w1 = xi[QK4_NL/2 + j]*xi[QK4_NL/2 + j];
        sumqx += w0*v0*xi[j] + w1*v1*xi[QK4_NL/2 + j];
        sumq2 += w0*v0*v0 + w1*v1*v1;
    }

    dsti->d = sumq2 > 0 ? sumqx/sumq2 : d;
}

template <cpy_kernel_t cpy_blck, int qk>
static __global__ void cpy_f32_q(const char * cx, char * cdst, const int ne,
                                 const int ne00, const int ne01, const int ne02, const int nb00, const int nb01, const int nb02,
                                 const int nb03, const int ne10, const int ne11, const int ne12, const int nb10, const int nb11,
                                 const int nb12, const int nb13) {
    const int i = (blockDim.x*blockIdx.x + threadIdx.x)*qk;

    if (i >= ne) {
        return;
    }

    const int i03 = i/(ne00 * ne01 * ne02);
    const int i02 = (i - i03*ne00*ne01*ne02 )/ (ne00*ne01);
    const int i01 = (i - i03*ne00*ne01*ne02  -  i02*ne01*ne00) / ne00;
    const int i00 = i - i03*ne00*ne01*ne02 - i02*ne01*ne00 - i01*ne00;
    const int x_offset = i00*nb00 + i01*nb01 + i02*nb02 + i03 * nb03;

    const int i13 = i/(ne10 * ne11 * ne12);
    const int i12 = (i - i13*ne10*ne11*ne12) / (ne10*ne11);
    const int i11 = (i - i13*ne10*ne11*ne12 - i12*ne10*ne11) / ne10;
    const int i10 = i - i13*ne10*ne11*ne12 - i12*ne10*ne11 - i11*ne10;
    const int dst_offset = (i10/qk)*nb10 + i11*nb11 + i12*nb12 + i13*nb13;

    cpy_blck(cx + x_offset, cdst + dst_offset);
}

static void ggml_cpy_f16_f32_cuda(
    const char * cx, char * cdst, const int ne,
    const int ne00, const int ne01, const int ne02, const int nb00, const int nb01, const int nb02,
    const int nb03, const int ne10, const int ne11, const int ne12, const int nb10, const int nb11, const int nb12, const int nb13, cudaStream_t stream) {

    const int num_blocks = (ne + CUDA_CPY_BLOCK_SIZE - 1) / CUDA_CPY_BLOCK_SIZE;
    cpy_f32_f16<cpy_1_f16_f32><<<num_blocks, CUDA_CPY_BLOCK_SIZE, 0, stream>>>
        (cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13);
}

static void ggml_cpy_f32_f32_cuda(
    const char * cx, char * cdst, const int ne,
    const int ne00, const int ne01, const int ne02, const int nb00, const int nb01, const int nb02,
    const int nb03, const int ne10, const int ne11, const int ne12, const int nb10, const int nb11, const int nb12, const int nb13, cudaStream_t stream) {

    const int num_blocks = (ne + CUDA_CPY_BLOCK_SIZE - 1) / CUDA_CPY_BLOCK_SIZE;
    cpy_f32_f16<cpy_1_f32_f32><<<num_blocks, CUDA_CPY_BLOCK_SIZE, 0, stream>>>
        (cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13);
}

static void ggml_cpy_f32_f16_cuda(
    const char * cx, char * cdst, const int ne,
    const int ne00, const int ne01, const int ne02, const int nb00, const int nb01, const int nb02,
    const int nb03, const int ne10, const int ne11, const int ne12, const int nb10, const int nb11, const int nb12, const int nb13, cudaStream_t stream) {

    const int num_blocks = (ne + CUDA_CPY_BLOCK_SIZE - 1) / CUDA_CPY_BLOCK_SIZE;
    cpy_f32_f16<cpy_1_f32_f16><<<num_blocks, CUDA_CPY_BLOCK_SIZE, 0, stream>>>
        (cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13);
}

static void ggml_cpy_f32_bf16_cuda(
    const char * cx, char * cdst, const int ne,
    const int ne00, const int ne01, const int ne02, const int nb00, const int nb01, const int nb02,
    const int nb03, const int ne10, const int ne11, const int ne12, const int nb10, const int nb11, const int nb12, const int nb13, cudaStream_t stream) {

    const int num_blocks = (ne + CUDA_CPY_BLOCK_SIZE - 1) / CUDA_CPY_BLOCK_SIZE;
    cpy_f32_f16<cpy_1_f32_bf16><<<num_blocks, CUDA_CPY_BLOCK_SIZE, 0, stream>>>
        (cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13);
}

static void ggml_cpy_f32_q8_0_cuda(
    const char * cx, char * cdst, const int ne,
    const int ne00, const int ne01, const int ne02, const int nb00, const int nb01, const int nb02,
    const int nb03, const int ne10, const int ne11, const int ne12, const int nb10, const int nb11, const int nb12, const int nb13, cudaStream_t stream) {

    GGML_ASSERT(ne % QK8_0 == 0);
    const int num_blocks = ne / QK8_0;
    cpy_f32_q<cpy_blck_f32_q8_0, QK8_0><<<num_blocks, 1, 0, stream>>>
        (cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13);
}

static void ggml_cpy_f32_q4_0_cuda(
    const char * cx, char * cdst, const int ne,
    const int ne00, const int ne01, const int ne02, const int nb00, const int nb01, const int nb02,
    const int nb03, const int ne10, const int ne11, const int ne12, const int nb10, const int nb11, const int nb12, const int nb13, cudaStream_t stream) {

    GGML_ASSERT(ne % QK4_0 == 0);
    const int num_blocks = ne / QK4_0;
    cpy_f32_q<cpy_blck_f32_q4_0, QK4_0><<<num_blocks, 1, 0, stream>>>
        (cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13);
}

static void ggml_cpy_f32_q4_1_cuda(
    const char * cx, char * cdst, const int ne,
    const int ne00, const int ne01, const int ne02, const int nb00, const int nb01, const int nb02,
    const int nb03, const int ne10, const int ne11, const int ne12, const int nb10, const int nb11, const int nb12, const int nb13, cudaStream_t stream) {

    GGML_ASSERT(ne % QK4_1 == 0);
    const int num_blocks = ne / QK4_1;
    cpy_f32_q<cpy_blck_f32_q4_1, QK4_1><<<num_blocks, 1, 0, stream>>>
        (cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13);
}

static void ggml_cpy_f32_q5_0_cuda(
    const char * cx, char * cdst, const int ne,
    const int ne00, const int ne01, const int ne02, const int nb00, const int nb01, const int nb02,
    const int nb03, const int ne10, const int ne11, const int ne12, const int nb10, const int nb11, const int nb12, const int nb13, cudaStream_t stream) {

    GGML_ASSERT(ne % QK5_0 == 0);
    const int num_blocks = ne / QK5_0;
    cpy_f32_q<cpy_blck_f32_q5_0, QK5_0><<<num_blocks, 1, 0, stream>>>
        (cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13);
}

static void ggml_cpy_f32_q5_1_cuda(
    const char * cx, char * cdst, const int ne,
    const int ne00, const int ne01, const int ne02, const int nb00, const int nb01, const int nb02,
    const int nb03, const int ne10, const int ne11, const int ne12, const int nb10, const int nb11, const int nb12, const int nb13, cudaStream_t stream) {

    GGML_ASSERT(ne % QK5_1 == 0);
    const int num_blocks = ne / QK5_1;
    cpy_f32_q<cpy_blck_f32_q5_1, QK5_1><<<num_blocks, 1, 0, stream>>>
        (cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13);
}

static void ggml_cpy_f32_q6_0_cuda(
    const char * cx, char * cdst, const int ne,
    const int ne00, const int ne01, const int ne02, const int nb00, const int nb01, const int nb02,
    const int nb03, const int ne10, const int ne11, const int ne12, const int nb10, const int nb11, const int nb12, const int nb13, cudaStream_t stream) {

    GGML_ASSERT(ne % QK6_0 == 0);
    const int num_blocks = ne / QK6_0;
    cpy_f32_q<cpy_blck_f32_q6_0, QK6_0><<<num_blocks, 1, 0, stream>>>
        (cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13);
}

static void ggml_cpy_f32_iq4_nl_cuda(
    const char * cx, char * cdst, const int ne,
    const int ne00, const int ne01, const int ne02, const int nb00, const int nb01, const int nb02,
    const int nb03, const int ne10, const int ne11, const int ne12, const int nb10, const int nb11, const int nb12, const int nb13, cudaStream_t stream) {

    GGML_ASSERT(ne % QK4_NL == 0);
    const int num_blocks = ne / QK4_NL;
    cpy_f32_q<cpy_blck_f32_iq4_nl, QK4_NL><<<num_blocks, 1, 0, stream>>>
        (cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13);
}

static void ggml_cpy_f16_f16_cuda(
    const char * cx, char * cdst, const int ne,
    const int ne00, const int ne01, const int ne02, const int nb00, const int nb01, const int nb02,
    const int nb03, const int ne10, const int ne11, const int ne12, const int nb10, const int nb11, const int nb12, const int nb13, cudaStream_t stream) {

    const int num_blocks = (ne + CUDA_CPY_BLOCK_SIZE - 1) / CUDA_CPY_BLOCK_SIZE;
    cpy_f32_f16<cpy_1_f16_f16><<<num_blocks, CUDA_CPY_BLOCK_SIZE, 0, stream>>>
        (cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13);
}

static void transpose_q8_0(ggml_backend_cuda_context & ctx, const ggml_tensor * src, ggml_tensor * dst) {
    auto stream = ctx.stream();
    auto num_blocks = ggml_nelements(dst)/QK8_0;
    k_transpose_q8_0<<<num_blocks, QK8_0, 0, stream>>>(
            (const char *)src->data, (char *)dst->data,
            dst->ne[0], dst->ne[1], dst->ne[2], src->nb[0], src->nb[2], src->nb[3],
            dst->nb[1], dst->nb[2], dst->nb[3]);
}

static void copy_q8_0_to_float(ggml_backend_cuda_context & ctx, const ggml_tensor * src, ggml_tensor * dst) {
    auto stream = ctx.stream();
    auto num_blocks = ggml_nelements(dst)/QK8_0;
    if (dst->type == GGML_TYPE_F16) {
        k_cpy_q8_0_to_float<<<num_blocks, QK8_0, 0, stream>>>((const char *)src->data, (half *)dst->data, ggml_nelements(dst),
                src->ne[0], src->ne[1], src->ne[2], src->nb[1], src->nb[2], src->nb[3]);
    }
    else if (dst->type == GGML_TYPE_F32) {
        k_cpy_q8_0_to_float<<<num_blocks, QK8_0, 0, stream>>>((const char *)src->data, (float *)dst->data, ggml_nelements(dst),
                src->ne[0], src->ne[1], src->ne[2], src->nb[1], src->nb[2], src->nb[3]);
    }
    else if (dst->type == GGML_TYPE_BF16) {
        k_cpy_q8_0_to_float<<<num_blocks, QK8_0, 0, stream>>>((const char *)src->data, (nv_bfloat16 *)dst->data, ggml_nelements(dst),
                src->ne[0], src->ne[1], src->ne[2], src->nb[1], src->nb[2], src->nb[3]);
    }
    else {
        GGML_ABORT("fatal error");
    }
}

void ggml_cuda_cpy(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, ggml_tensor * src1) {
    const int64_t ne = ggml_nelements(src0);
    GGML_ASSERT(ne == ggml_nelements(src1));

    GGML_ASSERT(ggml_nbytes(src0) <= INT_MAX);
    GGML_ASSERT(ggml_nbytes(src1) <= INT_MAX);

    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ne02 = src0->ne[2];

    //GGML_ASSERT(src0->ne[3] == 1);

    const int64_t nb00 = src0->nb[0];
    const int64_t nb01 = src0->nb[1];
    const int64_t nb02 = src0->nb[2];
    const int64_t nb03 = src0->nb[3];

    const int64_t ne10 = src1->ne[0];
    const int64_t ne11 = src1->ne[1];
    const int64_t ne12 = src1->ne[2];

    //GGML_ASSERT(src1->ne[3] == 1);

    const int64_t nb10 = src1->nb[0];
    const int64_t nb11 = src1->nb[1];
    const int64_t nb12 = src1->nb[2];
    const int64_t nb13 = src1->nb[3];

    cudaStream_t main_stream = ctx.stream();

    char * src0_ddc = (char *) src0->data;
    char * src1_ddc = (char *) src1->data;

    if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32) {
        ggml_cpy_f32_f32_cuda (src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F16) {
        ggml_cpy_f32_f16_cuda (src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_BF16) {
        ggml_cpy_f32_bf16_cuda (src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_Q8_0) {
        ggml_cpy_f32_q8_0_cuda(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_Q4_0) {
        ggml_cpy_f32_q4_0_cuda(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_Q4_1) {
        ggml_cpy_f32_q4_1_cuda(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_Q5_0) {
        ggml_cpy_f32_q5_0_cuda(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_Q6_0) {
        ggml_cpy_f32_q6_0_cuda(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_IQ4_NL) {
        ggml_cpy_f32_iq4_nl_cuda(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_Q5_1) {
        ggml_cpy_f32_q5_1_cuda(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F16) {
        ggml_cpy_f16_f16_cuda (src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_BF16) {
        ggml_cpy_f16_f16_cuda (src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F32) {
        ggml_cpy_f16_f32_cuda (src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } else if (ggml_are_same_shape(src0, src1) && src0->type == GGML_TYPE_Q8_0 &&
            (src1->type == GGML_TYPE_F16 || src1->type == GGML_TYPE_BF16 || src1->type == GGML_TYPE_F32)) {
        copy_q8_0_to_float(ctx, src0, src1);
    } else if (ggml_is_contiguous(src0) && ggml_are_same_shape(src0, src1)) {
        if (src1->type == GGML_TYPE_F16) {
            auto to_fp16 = ggml_get_to_fp16_cuda(src0->type);
            if (to_fp16) {
                to_fp16(src0->data, (half *)src1->data, ggml_nrows(src0), src0->ne[0], main_stream);
            }
        }
        else if (src1->type == GGML_TYPE_F32) {
            auto to_fp32 = ggml_get_to_fp32_cuda(src0->type);
            if (to_fp32) {
                to_fp32(src0->data, (float *)src1->data, ggml_nrows(src0), src0->ne[0], main_stream);
            }
        }
        else if (src1->type == GGML_TYPE_BF16) {
            auto to_bf16 = ggml_get_to_bf16_cuda(src0->type);
            if (to_bf16) {
                to_bf16(src0->data, (nv_bfloat16 *)src1->data, ggml_nrows(src0), src0->ne[0], main_stream);
            }
        }
    } else if (ggml_are_same_shape(src0, src1) && src0->type == GGML_TYPE_Q8_0 && src1->type == GGML_TYPE_Q8_0) {
        transpose_q8_0(ctx, src0, src1);
    } else {
        fprintf(stderr, "%s: unsupported type combination (%s to %s)\n", __func__,
                ggml_type_name(src0->type), ggml_type_name(src1->type));
        fprintf(stderr, "%s: %ld x %ld x %ld; %zu x %zu %zu -> %ld x %ld x %ld; %zu x %zu x %zu\n", __func__,
                src0->ne[0], src0->ne[1], src0->ne[2], src0->nb[1], src0->nb[2], src0->nb[3],
                src1->ne[0], src1->ne[1], src1->ne[2], src1->nb[1], src1->nb[2], src1->nb[3]);
        GGML_ABORT("fatal error");
    }
}

void ggml_cuda_dup(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    ggml_cuda_cpy(ctx, src0, dst);
}

void* ggml_cuda_cpy_fn(const ggml_tensor * src0, ggml_tensor * src1) {
    if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32) {
            return (void*) cpy_f32_f16<cpy_1_f32_f32>;
    } else if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F16) {
            return (void*) cpy_f32_f16<cpy_1_f32_f16>;
    } else if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_BF16) {
            return (void*) cpy_f32_f16<cpy_1_f32_bf16>;
    } else if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_Q8_0) {
            return (void*) cpy_f32_q<cpy_blck_f32_q8_0, QK8_0>;
    } else if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_Q4_0) {
            return (void*) cpy_f32_q<cpy_blck_f32_q4_0, QK4_0>;
    } else if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_Q4_1) {
            return (void*) cpy_f32_q<cpy_blck_f32_q4_1, QK4_1>;
    } else if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_Q5_0) {
            return (void*) cpy_f32_q<cpy_blck_f32_q5_0, QK5_0>;
    } else if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_IQ4_NL) {
            return (void*) cpy_f32_q<cpy_blck_f32_iq4_nl, QK4_NL>;
    } else if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_Q5_1) {
            return (void*) cpy_f32_q<cpy_blck_f32_q5_1, QK5_1>;
    } else if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_Q6_0) {
            return (void*) cpy_f32_q<cpy_blck_f32_q6_0, QK6_0>;
    } else if (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F16) {
            return (void*) cpy_f32_f16<cpy_1_f16_f16>;
    } else if (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F32) {
            return (void*) cpy_f32_f16<cpy_1_f16_f32>;
    } else if (ggml_are_same_shape(src0, src1) && src0->type == GGML_TYPE_Q8_0 &&
            (src1->type == GGML_TYPE_F16 || src1->type == GGML_TYPE_BF16 || src1->type == GGML_TYPE_F32)) {
        return (void*)copy_q8_0_to_float;
    } else if (ggml_is_contiguous(src0) && ggml_are_same_shape(src0, src1)) {
        if (src1->type == GGML_TYPE_F16) {
            auto to_fp16 = ggml_get_to_fp16_cuda(src0->type);
            if (to_fp16) return (void*)to_fp16;
        }
        else if (src1->type == GGML_TYPE_F32) {
            auto to_fp32 = ggml_get_to_fp32_cuda(src0->type);
            if (to_fp32) return (void*)to_fp32;
        }
        else if (src1->type == GGML_TYPE_BF16) {
            auto to_bf16 = ggml_get_to_bf16_cuda(src0->type);
            if (to_bf16) return (void*)to_bf16;
        }
    }
    else if (ggml_are_same_shape(src0, src1) && src0->type == GGML_TYPE_Q8_0 && src1->type == GGML_TYPE_Q8_0) {
        return (void *)transpose_q8_0;
    }
    fprintf(stderr, "%s: unsupported type combination (%s to %s)\n", __func__,
            ggml_type_name(src0->type), ggml_type_name(src1->type));
    fprintf(stderr, "%s: %ld x %ld x %ld; %zu x %zu %zu -> %ld x %ld x %ld; %zu x %zu x %zu\n", __func__,
                src0->ne[0], src0->ne[1], src0->ne[2], src0->nb[1], src0->nb[2], src0->nb[3],
                src1->ne[0], src1->ne[1], src1->ne[2], src1->nb[1], src1->nb[2], src1->nb[3]);
    GGML_ABORT("fatal error");
}

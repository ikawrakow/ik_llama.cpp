#include "nvfp4.cuh"

#if defined(GGML_CUDA_NVFP4_120A) || defined(GGML_CUDA_NVFP4_121A)
#include <cuda_fp8.h>
#include <cuda_fp4.h>

#include "mma_new.cuh"

constexpr int BK = 256, BM = 128, NWARPS = 8, BM_ID = 64, NWARPS_ID = 4, SY = 36, SW = BK/8 + 12;

template<bool indexed> constexpr int nvfp4_bm = indexed ? BM_ID : BM;
template<bool indexed> constexpr int nvfp4_nwarps = indexed ? NWARPS_ID : NWARPS;

static __device__ __forceinline__ void copy4(void * dst, const void * src, bool valid) {
#if defined(__CUDA_ARCH_FEAT_SM120_ALL) || defined(__CUDA_ARCH_FEAT_SM121_ALL)
    const uint32_t address = __cvta_generic_to_shared(dst);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;" :: "r"(address), "l"(src), "r"(valid ? 4 : 0));
#endif
}

static __device__ __forceinline__ void copy16(void * dst, const void * src, bool valid) {
#if defined(__CUDA_ARCH_FEAT_SM120_ALL) || defined(__CUDA_ARCH_FEAT_SM121_ALL)
    const uint32_t address = __cvta_generic_to_shared(dst);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" :: "r"(address), "l"(src), "r"(valid ? 16 : 0));
#endif
}

static __device__ __forceinline__ void mma_nvfp4(float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2], uint32_t sa, uint32_t sb) {
#if defined(__CUDA_ARCH_FEAT_SM120_ALL) || defined(__CUDA_ARCH_FEAT_SM121_ALL)
    asm("mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, %10, {0,0}, %11, {0,0};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(sa), "r"(sb));
#endif
}

struct nvfp4_epilogue {
    const float * bias_up = nullptr, * bias_gate = nullptr;
    int stride_up = 0, stride_gate = 0;
    float alpha = 0.0f;
};

template<bool fused> struct nvfp4_fusion {};
template<> struct nvfp4_fusion<true> {
    const uint8_t * gate;
    float limit;
    nvfp4_epilogue epi;
};

template<bool indexed> struct nvfp4_routing {};
template<> struct nvfp4_routing<true> {
    const int32_t * src, * dst, * bounds;
    size_t expert_bytes;
};

template<int J, bool fused, bool tail, bool indexed>
static __global__ void __launch_bounds__(nvfp4_nwarps<indexed>*32, 1) mul_mat_nvfp4(
        const uint8_t * __restrict__ w, const uint8_t * __restrict__ xq, const uint8_t * __restrict__ xs,
        const float * __restrict__ xr, float * __restrict__ y, int m, int n, int k, size_t row_bytes, nvfp4_fusion<fused> fusion, nvfp4_routing<indexed> routing) {
#if defined(__CUDA_ARCH_FEAT_SM120_ALL) || defined(__CUDA_ARCH_FEAT_SM121_ALL)
    using namespace ggml_cuda_mma;
    constexpr int BM = nvfp4_bm<indexed>, NWARPS = nvfp4_nwarps<indexed>;
    extern __shared__ __align__(16) uint32_t smem[];
    uint32_t * sw = smem, * sy = sw + 2*BM*SW;
    constexpr int row_scale = BK/8 + BK/64;
    const int lane = threadIdx.x, warp = threadIdx.y, tid = warp*32 + lane;
    const int i0 = (warp/2)*32, j0 = (warp%2)*8;
    const int m0 = blockIdx.y*BM;
    int n0 = blockIdx.x*J, end = n;
    if constexpr (indexed) {
        n0 += routing.bounds[blockIdx.z];
        end = routing.bounds[blockIdx.z + 1];
        if (n0 >= end) return;
        w += blockIdx.z*routing.expert_bytes;
        if constexpr (fused) fusion.gate += blockIdx.z*routing.expert_bytes;
    }
    __shared__ int input_rows[indexed ? J : 1];
    if constexpr (indexed) {
        if (tid < J && n0 + tid < end) input_rows[tid] = routing.src[n0 + tid];
    }
    auto activation_row = [&](int row) {
        if constexpr (indexed) return input_rows[row - n0];
        return row;
    };
    auto weight_index = [](int row) { return fused ? (row/16)*8 + row%8 : row; };
    const size_t tile_offset = size_t(weight_index(m0))*row_bytes;
    const int stride = row_bytes;
    w += tile_offset;
    if constexpr (fused) fusion.gate += tile_offset;
    auto weight_row = [&](int row) {
        if constexpr (fused) {
            return ((row%16 < 8) ? w : fusion.gate) + weight_index(row)*stride;
        } else {
            return w + row*stride;
        }
    };
    float d[J/16][2][4] = {};
    if (tid < BM) {
        sw[tid*SW + row_scale] = (!tail || weight_index(m0 + tid) < m) ? *(const uint32_t *) weight_row(tid) : 0;
    }
    auto load_weights = [&](int kb, uint32_t * tile) {
        for (int c = tid; c < BM*(BK/8); c += NWARPS*32) {
            const int row = c/(BK/8), l = c%(BK/8), block = l/8;
            const bool valid = !tail || (weight_index(m0 + row) < m && kb/64 + block < k/64);
            const uint8_t * src = valid ? weight_row(row) + 8 + (kb/64 + block)*36 + (l%8)*4 : w;
            copy4(tile + row*SW + l, src, valid);
        }
        for (int c = tid; c < BM*(BK/64); c += NWARPS*32) {
            const int row = c/(BK/64), block = c%(BK/64);
            const bool valid = !tail || (weight_index(m0 + row) < m && kb/64 + block < k/64);
            const uint8_t * src = valid ? weight_row(row) + 4 + (kb/64 + block)*36 : w;
            copy4(tile + row*SW + BK/8 + block, src, valid);
        }
    };
    load_weights(0, sw);
    asm volatile("cp.async.commit_group;");
    for (int kb = 0; kb < k; kb += BK) {
        __syncthreads();
        uint32_t * next = sw == smem ? smem + BM*SW : smem;
        for (int c = tid; c < J*8; c += NWARPS*32) {
            const int row = c/8, l = c%8;
            const bool valid = !(tail || indexed) || n0 + row < end;
            const size_t offset = valid ? (size_t(kb/256)*n + activation_row(n0 + row))*128 + l*16 : 0;
            copy16(sy + row*SY + 4 + l*4, xq + offset, valid);
        }
        if (tid < J) {
            const bool valid = !(tail || indexed) || n0 + tid < end;
            const size_t offset = valid ? (size_t(kb/256)*n + activation_row(n0 + tid))*16 : 0;
            copy16(sy + tid*SY, xs + offset, valid);
        }
        asm volatile("cp.async.commit_group;");
        if (kb + BK < k) {
            load_weights(kb + BK, next);
            asm volatile("cp.async.commit_group;");
            asm volatile("cp.async.wait_group 1;");
        } else {
            asm volatile("cp.async.wait_group 0;");
        }
        __syncthreads();
        tile<16, 8, uint32_t> a[2][4];
        uint32_t sa[2][4];
#pragma unroll
        for (int i = 0; i < 2; ++i) {
#pragma unroll
            for (int f = 0; f < 4; ++f) {
                load_ldmatrix(a[i][f], sw + (i0 + i*16)*SW + f*8, SW);
                sa[i][f] = sw[(i0 + i*16 + lane/4 + (lane%2)*8)*SW + BK/8 + f];
            }
        }
#pragma unroll
        for (int j = 0; j < J/16; ++j) {
            tile<8, 8, uint32_t> b[4];
            uint32_t sb[4];
#pragma unroll
            for (int f = 0; f < 4; ++f) {
                load_generic(b[f], sy + (j0 + j*16)*SY + 4 + f*8, SY);
                sb[f] = sy[(j0 + j*16 + lane/4)*SY + f];
            }
#pragma unroll
            for (int i = 0; i < 2; ++i) {
#pragma unroll
                for (int f = 0; f < 4; ++f) {
                    mma_nvfp4(d[j][i], a[i][f].x, b[f].x, sa[i][f], sb[f]);
                }
            }
        }
        __syncthreads();
        sw = next;
    }
#pragma unroll
    for (int j = 0; j < J/16; ++j) {
#pragma unroll
        for (int i = 0; i < 2; ++i) {
#pragma unroll
            for (int l = 0; l < (fused ? 2 : 4); ++l) {
                const int row = i0 + i*16 + lane/4 + (l/2)*8;
                const int col = n0 + j0 + j*16 + (lane%4)*2 + l%2;
                if ((!tail || weight_index(m0 + row) < m) && (!(tail || indexed) || col < end)) {
                    const int out_col = [&]() { if constexpr (indexed) return routing.dst[col]; else return col; }();
                    const float wr = __uint_as_float(smem[row*SW + row_scale]);
                    float up = d[j][i][l]*(wr*xr[activation_row(col)]);
                    if constexpr (fused) {
                        const float gr = __uint_as_float(smem[(row + 8)*SW + row_scale]);
                        float g = d[j][i][l+2]*(gr*xr[activation_row(col)]);
                        const int out_row = (m0 + i0 + i*16)/2 + lane/4;
                        const int expert = indexed ? blockIdx.z : 0;
                        if (fusion.epi.bias_up)   up = __fadd_rn(up, fusion.epi.bias_up  [expert*fusion.epi.stride_up   + out_row]);
                        if (fusion.epi.bias_gate) g  = __fadd_rn(g,  fusion.epi.bias_gate[expert*fusion.epi.stride_gate + out_row]);
                        float value;
                        if (fusion.epi.alpha > 0.0f) {
                            const float x = min(g, fusion.limit);
                            value = x/(1.0f + expf(-fusion.epi.alpha*x))*(1.0f + max(-fusion.limit, min(fusion.limit, up)));
                        } else if (fusion.limit < 1e-6f) {
                            value = g*up/(1.0f + expf(-g));
                        } else {
                            value = min(g/(1.0f + expf(-g)), fusion.limit)*max(-fusion.limit, min(fusion.limit, up));
                        }
                        y[size_t(out_col)*m + out_row] = value;
                    } else {
                        y[size_t(out_col)*m + m0 + row] = up;
                    }
                }
            }
        }
    }
#endif
}

template<int J, bool fused, bool tail, bool indexed = false>
static void launch_nvfp4_tile(const void * w, const void * gate, const uint8_t * q, const uint8_t * s, const float * r, float * y,
        int m, int n, int k, size_t row_bytes, float limit, cudaStream_t stream,
        int tokens = 0, int experts = 1, nvfp4_routing<indexed> routing = {}, nvfp4_epilogue epi = {}) {
    constexpr int BM = nvfp4_bm<indexed>, NWARPS = nvfp4_nwarps<indexed>;
    constexpr int bytes = (2*BM*SW + J*SY)*sizeof(uint32_t);
    auto kernel = mul_mat_nvfp4<J, fused, tail, indexed>;
    nvfp4_fusion<fused> fusion;
    if constexpr (fused) fusion = { (const uint8_t *) gate, limit, epi };
    CUDA_SET_SHARED_MEMORY_LIMIT(kernel, bytes);
    kernel<<<dim3(((indexed ? tokens : n) + J - 1)/J, ((fused ? 2*m : m) + BM - 1)/BM, experts), dim3(32, NWARPS), bytes, stream>>>(
            (const uint8_t *) w, q, s, r, y, m, n, k, row_bytes, fusion, routing);
}

template<int J, bool fused, bool indexed>
static void launch_nvfp4(const void * w, const void * gate, const uint8_t * q, const uint8_t * s, const float * r, float * y,
        int m, int n, int k, size_t row_bytes, float limit, cudaStream_t stream,
        int tokens = 0, int experts = 1, nvfp4_routing<indexed> routing = {}, nvfp4_epilogue epi = {}) {
    constexpr int bm = nvfp4_bm<indexed>;
    if (m % (fused ? bm/2 : bm) == 0 && (indexed || n % J == 0) && k % BK == 0) {
        launch_nvfp4_tile<J, fused, false, indexed>(w, gate, q, s, r, y, m, n, k, row_bytes, limit, stream, tokens, experts, routing, epi);
    } else {
        launch_nvfp4_tile<J, fused, true, indexed>(w, gate, q, s, r, y, m, n, k, row_bytes, limit, stream, tokens, experts, routing, epi);
    }
}

template<bool fused>
static void dispatch_nvfp4(const void * w, const void * gate, const uint8_t * q, const uint8_t * s, const float * r, float * y,
        int m, int n, int k, size_t row_bytes, float limit, cudaStream_t stream) {
    if (n <= (fused ? 64 : 128)) {
        launch_nvfp4<64, fused, false>(w, gate, q, s, r, y, m, n, k, row_bytes, limit, stream);
    } else if (!fused && n <= 512 && ((m + BM - 1)/BM)*((n + 127)/128) < ggml_cuda_info().devices[ggml_cuda_get_device()].nsm) {
        launch_nvfp4<96, fused, false>(w, gate, q, s, r, y, m, n, k, row_bytes, limit, stream);
    } else {
        launch_nvfp4<128, fused, false>(w, gate, q, s, r, y, m, n, k, row_bytes, limit, stream);
    }
}

template<bool fused>
static void dispatch_nvfp4_id(const ggml_tensor * src0, const ggml_tensor * gate, const ggml_tensor * src1, ggml_tensor * dst,
        const nvfp4_input & input, nvfp4_routing<true> routing, float alpha, float limit, cudaStream_t stream) {
    const int m = fused ? dst->ne[0] : src0->ne[1];
    const int tokens = src1->ne[2], experts = src0->ne[2];
    const int per_expert = (tokens*dst->ne[1] + experts - 1)/experts;
    const void * up_data = src0->data;
    const void * gate_data = nullptr;
    if constexpr (fused) {
        gate_data = gate ? gate->data : src0->data;
        if (!gate) up_data = (const char *) src0->data + m*src0->nb[1];
    }
    nvfp4_epilogue epi;
    if constexpr (fused) {
        const ggml_tensor * bias_up = dst->src[4], * bias_gate = dst->src[5];
        if (bias_up) {
            epi.bias_up = (const float *) bias_up->data;
            epi.stride_up = bias_up->nb[1]/sizeof(float);
            if (!gate) {
                epi.bias_gate = epi.bias_up;
                epi.stride_gate = epi.stride_up;
                epi.bias_up += m;
            }
        }
        if (bias_gate) {
            epi.bias_gate = (const float *) bias_gate->data;
            epi.stride_gate = bias_gate->nb[1]/sizeof(float);
        }
        epi.alpha = alpha;
    }
    if (per_expert <= 16) {
        launch_nvfp4<16, fused, true>(up_data, gate_data, input.q.ptr, input.s.ptr, input.r.ptr, (float *) dst->data, m, input.n, input.k, src0->nb[1], limit, stream, tokens, experts, routing, epi);
    } else {
        launch_nvfp4<64, fused, true>(up_data, gate_data, input.q.ptr, input.s.ptr, input.r.ptr, (float *) dst->data, m, input.n, input.k, src0->nb[1], limit, stream, tokens, experts, routing, epi);
    }
}

static __device__ __forceinline__ float nvfp4_rcp(float a) {
    float b;
    asm volatile("rcp.approx.ftz.f32 %0, %1;" : "=f"(b) : "f"(a));
    return b;
}

static __global__ void quantize_nvfp4(const float * x, float * r, uint8_t * q, uint8_t * s, int k, int kp, int stride, float scale) {
#if defined(__CUDA_ARCH_FEAT_SM120_ALL) || defined(__CUDA_ARCH_FEAT_SM121_ALL)
    __shared__ float red[8];
    __shared__ float G;
    const int n = blockIdx.x, t = threadIdx.x;
    const unsigned int rows = gridDim.x;
    const float * xr = x + size_t(n)*stride;
    float rs = scale;
    if (scale < 0) {
        float a = 0;
        for (int i = t; i < k; i += blockDim.x) a = fmaxf(a, fabsf(xr[i]));
        a = warp_reduce_max(a);
        if (t%32 == 0) red[t/32] = a;
        __syncthreads();
        if (t < 32) {
            a = t < 8 ? red[t] : 0;
            a = warp_reduce_max(a);
            rs = __fdiv_rn(a, 6.0f*448.0f);
        }
    }
    if (t == 0) {
        G = rs > 0 ? __fdiv_rn(1.0f, rs) : 0.0f;
        r[n] = rs;
    }
    __syncthreads();
    // Byte u holds values i (low nibble) and i + 8 of one 16-value group, as in the NVFP4 block.
    for (int u = t; u < kp/2; u += blockDim.x) {
        const int i = (u/8)*16 + u%8;
        const float2 v = make_float2(i < k ? xr[i] : 0.0f, i < k ? xr[i + 8] : 0.0f);
        float a = fmaxf(fabsf(v.x), fabsf(v.y));
        for (int d = 4; d; d /= 2) a = fmaxf(a, __shfl_xor_sync(0xffffffff, a, d, 8));
        const float sf_value = G*(a*0x1.555556p-3f);
        const uint8_t sb = __nv_cvt_float_to_fp8(sf_value, __NV_SATFINITE, __NV_E4M3);
        const float sf = ggml_cuda_ue4m3_to_fp32(sb);
        const float inv = sb != 0 ? nvfp4_rcp(sf*nvfp4_rcp(G)) : 0.0f;
        q[(size_t(u/128)*rows + n)*128 + u%128] = __nv_cvt_float2_to_fp4x2(make_float2(v.x*inv, v.y*inv), __NV_E2M1, cudaRoundNearest);
        if (i%16 == 0) s[(size_t(i/256)*rows + n)*16 + (i%256)/16] = sb;
    }
#else
    GGML_UNUSED(x); GGML_UNUSED(r); GGML_UNUSED(q); GGML_UNUSED(s); GGML_UNUSED(k); GGML_UNUSED(kp); GGML_UNUSED(stride); GGML_UNUSED(scale);
#endif
}

#endif

bool ggml_cuda_nvfp4_available(int cc) {
#ifdef GGML_CUDA_NVFP4_120A
    if (cc == 1200) return true;
#endif
#ifdef GGML_CUDA_NVFP4_121A
    if (cc == 1210) return true;
#endif
    GGML_UNUSED(cc);
    return false;
}

bool ggml_cuda_can_mul_mat_nvfp4_id(const ggml_tensor * src0, const ggml_tensor * src1, int cc) {
    return src0->type == GGML_TYPE_NVFP4_KS && ggml_get_input_scale(src0) != 0.0f && ggml_cuda_nvfp4_available(cc)
        && src1->type == GGML_TYPE_F32
        && ggml_is_contiguous(src0) && src1->nb[0] == sizeof(float);
}

bool ggml_cuda_can_mul_mat_nvfp4(const ggml_tensor * src0, const ggml_tensor * src1, int cc) {
    return ggml_cuda_can_mul_mat_nvfp4_id(src0, src1, cc)
        && src0->ne[2]*src0->ne[3] == 1 && src1->ne[2]*src1->ne[3] == 1;
}

static void ggml_cuda_quantize_nvfp4(const float * x, uint8_t * q, uint8_t * s, float * r, int k, int kp, int n, int stride, float scale, cudaStream_t stream) {
#if defined(GGML_CUDA_NVFP4_120A) || defined(GGML_CUDA_NVFP4_121A)
    quantize_nvfp4<<<n, 256, 0, stream>>>(x, r, q, s, k, kp, stride, scale);
#else
    GGML_UNUSED(x); GGML_UNUSED(q); GGML_UNUSED(s); GGML_UNUSED(r);
    GGML_UNUSED(k); GGML_UNUSED(kp); GGML_UNUSED(n); GGML_UNUSED(stride); GGML_UNUSED(scale); GGML_UNUSED(stream);
#endif
}

static void ggml_cuda_gemm_nvfp4(const void * w, const uint8_t * q, const uint8_t * s, const float * r, float * y,
        int m, int n, int k, size_t row_bytes, cudaStream_t stream) {
#if defined(GGML_CUDA_NVFP4_120A) || defined(GGML_CUDA_NVFP4_121A)
    dispatch_nvfp4<false>(w, nullptr, q, s, r, y, m, n, k, row_bytes, 0.0f, stream);
#else
    GGML_UNUSED(w); GGML_UNUSED(q); GGML_UNUSED(s); GGML_UNUSED(r); GGML_UNUSED(y);
    GGML_UNUSED(m); GGML_UNUSED(n); GGML_UNUSED(k); GGML_UNUSED(row_bytes); GGML_UNUSED(stream);
#endif
}

static void ggml_cuda_gemm_nvfp4_glu(const void * w, const void * gate, const uint8_t * q, const uint8_t * s, const float * r, float * y,
        int m, int n, int k, size_t row_bytes, float limit, cudaStream_t stream) {
#if defined(GGML_CUDA_NVFP4_120A) || defined(GGML_CUDA_NVFP4_121A)
    dispatch_nvfp4<true>(w, gate, q, s, r, y, m, n, k, row_bytes, limit, stream);
#else
    GGML_UNUSED(w); GGML_UNUSED(gate); GGML_UNUSED(q); GGML_UNUSED(s); GGML_UNUSED(r); GGML_UNUSED(y);
    GGML_UNUSED(m); GGML_UNUSED(n); GGML_UNUSED(k); GGML_UNUSED(row_bytes); GGML_UNUSED(limit); GGML_UNUSED(stream);
#endif
}

nvfp4_input::nvfp4_input(ggml_backend_cuda_context & ctx) : q(ctx.pool()), s(ctx.pool()), r(ctx.pool()) {}

nvfp4_input::nvfp4_input(ggml_backend_cuda_context & ctx, const ggml_tensor * src1, float scale) : nvfp4_input(ctx) {
    quantize(ctx, src1, scale);
}

void nvfp4_input::quantize(ggml_backend_cuda_context & ctx, const ggml_tensor * src1, float scale) {
    k = src1->ne[0];
    n = ggml_nrows(src1);
    const int kp = GGML_PAD(k, 256);
    q.alloc(size_t(n)*kp/2);
    s.alloc(size_t(n)*kp/16);
    r.alloc(n);
    ggml_cuda_quantize_nvfp4((const float *) src1->data, q.get(), s.get(), r.get(), k, kp, n,
            src1->nb[1]/sizeof(float), scale, ctx.stream());
    CUDA_CHECK(cudaGetLastError());
}

void nvfp4_input::mul_mat(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, float * dst) {
    ggml_cuda_gemm_nvfp4(src0->data, q.get(), s.get(), r.get(), dst, src0->ne[1], n, k, src0->nb[1], ctx.stream());
    CUDA_CHECK(cudaGetLastError());
}

bool ggml_cuda_nvfp4_same_input_scale(const ggml_tensor * a, const ggml_tensor * b) {
    const float sa = ggml_get_input_scale(a), sb = ggml_get_input_scale(b);
    return sa == sb;
}

int ggml_cuda_mul_mat_nvfp4(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
        ggml_tensor * dst, const ggml_cgraph * cgraph, int node_n) {
    nvfp4_input input(ctx, src1, ggml_get_input_scale(src0));
    input.mul_mat(ctx, src0, (float *) dst->data);
    if (!cgraph) return node_n;
    while (node_n + 1 < cgraph->n_nodes) {
        dst = cgraph->nodes[node_n+1];
        if (ggml_is_noop(dst)) {
            ++node_n; continue;
        }
        if (dst->op != GGML_OP_MUL_MAT || dst->src[1] != src1 ||
                !ggml_cuda_can_mul_mat_nvfp4(dst->src[0], src1, ggml_cuda_info().devices[ctx.device].cc) ||
                !ggml_cuda_nvfp4_same_input_scale(src0, dst->src[0])) break;
        input.mul_mat(ctx, dst->src[0], (float *) dst->data);
        ++node_n;
    }
    return node_n;
}

void ggml_cuda_mul_mat_nvfp4_pair(ggml_backend_cuda_context & ctx, const ggml_tensor * up, const ggml_tensor * gate,
        const ggml_tensor * src1, float * dst_up, float * dst_gate) {
    nvfp4_input input(ctx, src1, ggml_get_input_scale(up));
    input.mul_mat(ctx, up, dst_up);
    input.mul_mat(ctx, gate, dst_gate);
}

void ggml_cuda_up_gate_nvfp4(ggml_backend_cuda_context & ctx, const ggml_tensor * up, const ggml_tensor * gate,
        const ggml_tensor * src1, ggml_tensor * dst, float limit) {
    nvfp4_input input(ctx, src1, ggml_get_input_scale(up));
    ggml_cuda_gemm_nvfp4_glu(up->data, gate->data, input.q.get(), input.s.get(), input.r.get(), (float *) dst->data,
            up->ne[1], input.n, input.k, up->nb[1], limit, ctx.stream());
    CUDA_CHECK(cudaGetLastError());
}

void ggml_cuda_gemm_nvfp4_id(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
        ggml_tensor * dst, const int32_t * ids_src, const int32_t * ids_dst, const int32_t * bounds,
        const nvfp4_input & input, bool fused, const ggml_tensor * gate, float alpha, float limit) {
#if defined(GGML_CUDA_NVFP4_120A) || defined(GGML_CUDA_NVFP4_121A)
    const nvfp4_routing<true> routing = { ids_src, ids_dst, bounds, src0->nb[2] };
    if (fused) dispatch_nvfp4_id<true>(src0, gate, src1, dst, input, routing, alpha, limit, ctx.stream());
    else dispatch_nvfp4_id<false>(src0, gate, src1, dst, input, routing, alpha, limit, ctx.stream());
    CUDA_CHECK(cudaGetLastError());
#else
    GGML_UNUSED(ctx); GGML_UNUSED(src0); GGML_UNUSED(src1); GGML_UNUSED(dst);
    GGML_UNUSED(ids_src); GGML_UNUSED(ids_dst); GGML_UNUSED(bounds); GGML_UNUSED(input);
    GGML_UNUSED(fused); GGML_UNUSED(gate); GGML_UNUSED(alpha); GGML_UNUSED(limit);
#endif
}

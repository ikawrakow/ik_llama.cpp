#include "f8.cuh"
#include "mm-f8.cuh"
#include "mm-f8-lt.cuh"
#include "convert.cuh"
#include "norm.cuh"
#include "unary.cuh"

#include "ggml-cuda.h"
#include "ggml-backend-impl.h"

// defined in ggml-cuda.cu
extern ggml_log_callback ggml_cuda_log_callback;
extern void * ggml_cuda_log_user_data;

static void f8_w8a16_notice(int device, const ggml_tensor * weight) {
    static bool done[GGML_CUDA_MAX_DEVICES] = {};
    if (done[device] || ggml_get_input_scale(weight) == 0.0f) return;
    done[device] = true;
    const int cc = ggml_cuda_info().devices[device].cc;
    if (f8_mma_available(cc) || !ggml_cuda_log_callback) return;
    char msg[160];
#if !defined(GGML_USE_HIPBLAS) && !defined(GGML_USE_MUSA) && CUDART_VERSION >= 11080
    if (cc >= CC_ADA_LOVELACE) {
        snprintf(msg, sizeof(msg), "FP8: W8A8 requested, device %d uses W8A16: this build has no code for compute capability 8.9 or higher\n", device);
    } else {
        snprintf(msg, sizeof(msg), "FP8: W8A8 requested, device %d uses W8A16: W8A8 needs compute capability 8.9, the device has %d.%d\n",
                device, cc/100, cc%100/10);
    }
#else
    snprintf(msg, sizeof(msg), "FP8: W8A8 requested, device %d uses W8A16: this build has no W8A8 kernels\n", device);
#endif
    ggml_cuda_log_callback(GGML_LOG_LEVEL_WARN, msg, ggml_cuda_log_user_data);
}

#ifdef F8_TMA_AVAILABLE
template <int RN>
static bool launch_f8_tma_dense(int n, int rows, int cols, const char * x, size_t stride, const uint8_t * y,
        const float * scales, float * z, cudaStream_t stream, const float * other = nullptr) {
    if (f8_tma_shared<RN, false, true>(n/128) > (int)ggml_cuda_info().devices[ggml_cuda_get_device()].smpbo) return false;
    const f8_moe_routing routing{nullptr, nullptr, nullptr, 1, stride, stride*rows, nullptr, 0, nullptr, nullptr, nullptr, false};
    return other ?
        launch_f8_tma<RN, false, true, true>(n, rows, cols, x, nullptr, y, scales, n/128, z, nullptr, nullptr, nullptr, 0, routing,
            nullptr, stream, 8, other) :
        launch_f8_tma<RN, false, true>(n, rows, cols, x, nullptr, y, scales, n/128, z, nullptr, nullptr, nullptr, 0, routing,
            nullptr, stream, 8);
}
#endif

static bool launch_f8_w8a8(const ggml_tensor * weight, int n, int rows, int cols, int ldc,
        const char * x, const uint8_t * y, float * z, const float * scales, cudaStream_t stream, const float * other = nullptr) {
#ifdef F8_TMA_AVAILABLE
    if (weight->type == GGML_TYPE_F8_E4M3_B128 && f8_tma_available(ggml_cuda_info().devices[ggml_cuda_get_device()].cc) &&
        n/128 <= 256 && weight->nb[1]%16 == 0 && rows%128 == 0 && ldc == rows &&
        (launch_f8_tma_dense<8>(n, rows, cols, x, weight->nb[1], y, scales, z, stream, other) ||
         launch_f8_tma_dense<6>(n, rows, cols, x, weight->nb[1], y, scales, z, stream, other))) return true;
#endif
    if (other) return false;
    if (weight->type == GGML_TYPE_F8_E4M3_KS) {
        launch_f8_pp<ggml_cuda_f8_row,128,8,2,64,f8_activation_group>(n, rows, cols, ldc, x, y, z, stream, scales);
    } else if (ggml_cuda_info().devices[ggml_cuda_get_device()].cc >= 1200) {
        launch_f8_pp<ggml_cuda_f8_block,128,8,1,128,f8_activation_group>(n, rows, cols, ldc, x, y, z, stream, scales);
    } else if (cols <= 64) {
        launch_f8_pp<ggml_cuda_f8_block,32,4,1,128,f8_activation_group>(n, rows, cols, ldc, x, y, z, stream, scales);
    } else {
        launch_f8_pp<ggml_cuda_f8_block,64,4,1,128,f8_activation_group>(n, rows, cols, ldc, x, y, z, stream, scales);
    }
    return true;
}

template <bool w8a8 = false>
static void ggml_cuda_op_mul_mat_f8_e4m3(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst,
        const char * src0_dd_i, const float * src1_ddf_i, const char * src1_ddq_i, float * dst_dd_i,
        int64_t row_low, int64_t row_high, int64_t src1_ncols, int64_t src1_padded_row_size, cudaStream_t stream) {
    const int id = ggml_cuda_get_device();
    const int cc = ggml_cuda_info().devices[id].cc;
    const int nrows = row_high - row_low;
    const int ldc = id == ctx.device ? dst->ne[0] : nrows;
    if constexpr (w8a8) {
        if (src1_ncols > 8 && dst->op_params[0] == GGML_PREC_DEFAULT && f8_mma_available(cc)) {
            constexpr int G = f8_activation_group;
            const int n = src0->ne[0];
            ggml_cuda_pool_alloc<uint8_t> activations(ctx.pool(id), size_t(src1_ncols)*n);
            ggml_cuda_pool_alloc<float> scales(ctx.pool(id), size_t(src1_ncols)*((n+G-1)/G));
            f8_quantize<G><<<dim3((n+4*G-1)/(4*G),src1_ncols),128,0,stream>>>(src1_ddf_i, activations.get(), scales.get(), n, src1->ne[0]);
            launch_f8_w8a8(src0, n, nrows, src1_ncols, ldc, src0_dd_i, activations.get(), dst_dd_i, scales.get(), stream);
            return;
        }
    }
    if (src1_ncols > 8 && dst->op_params[0] == GGML_PREC_DEFAULT && new_mma_available(cc)) {
        ggml_cuda_pool_alloc<half> activations(ctx.pool(id), src1_ncols*src1->ne[0]);
        ggml_get_to_fp16_cuda(GGML_TYPE_F32)(src1_ddf_i, activations.get(), src1_ncols, src1->ne[0], stream);
        if (src0->type == GGML_TYPE_F8_E4M3_KS) {
            launch_f8_pp<ggml_cuda_f8_row>(cc, src0->ne[0], nrows, src1_ncols, ldc, src0_dd_i, activations.get(), dst_dd_i, stream);
        } else {
            launch_f8_pp<ggml_cuda_f8_block>(cc, src0->ne[0], nrows, src1_ncols, ldc, src0_dd_i, activations.get(), dst_dd_i, stream);
        }
        return;
    }
    if (src0->type == GGML_TYPE_F8_E4M3_B128) {
        auto launch = [&](auto kernel) {
            kernel<<<dim3(nrows, src1_ncols), 256, 0, stream>>>(src0->ne[0], ldc, src0_dd_i, src1_ddf_i, dst_dd_i, src1->ne[0]);
        };
        if (src0->ne[0] % 512 == 0 && uintptr_t(src0_dd_i) % 16 == 0 && uintptr_t(src1_ddf_i) % 16 == 0) {
            launch(mul_mat_vec_f8_e4m3_b128<true>);
        } else {
            launch(mul_mat_vec_f8_e4m3_b128<false>);
        }
    } else if (src1_ncols <= 8) {
        const dim3 grid(nrows, src1_ncols);
        if (src0->ne[0] % 16 == 0 && uintptr_t(src0_dd_i) % 16 == 0 && uintptr_t(src1_ddf_i) % 16 == 0) {
            const int threads = size_t(src0->ne[0])*nrows > 32*1024*1024 ? 256 : 128;
            mul_mat_vec_f8_e4m3_wide<16, true><<<grid, threads, 0, stream>>>(src0->ne[0], ldc, src0_dd_i,
                    src1_ddf_i, dst_dd_i, src1->ne[0]);
        } else if (src0->ne[0] >= 16 && src0->ne[0] % 4 == 0 && uintptr_t(src0_dd_i) % 4 == 0 && uintptr_t(src1_ddf_i) % 16 == 0) {
            if (size_t(src0->ne[0])*nrows > 32*1024*1024) {
                mul_mat_vec_f8_e4m3_wide<16><<<grid, 256, 0, stream>>>(src0->ne[0], ldc, src0_dd_i,
                        src1_ddf_i, dst_dd_i, src1->ne[0]);
            } else {
                mul_mat_vec_f8_e4m3_wide<8><<<grid, 128, 0, stream>>>(src0->ne[0], ldc, src0_dd_i,
                        src1_ddf_i, dst_dd_i, src1->ne[0]);
            }
        } else {
            mul_mat_vec_f8_e4m3<ggml_cuda_f8_row><<<grid, 256, 0, stream>>>(src0->ne[0], ldc, (const ggml_fp8_e4m3_t *)src0_dd_i,
                    src1_ddf_i, dst_dd_i, src1->ne[0]);
        }
    } else {
        ggml_cuda_op_mul_mat_cublas(ctx, src0, src1, dst, src0_dd_i, src1_ddf_i, src1_ddq_i, dst_dd_i,
                row_low, row_high, src1_ncols, src1_padded_row_size, stream);
    }
}

static int f8_next(const ggml_cgraph * graph, int j) {
    do ++j; while (j < graph->n_nodes && ggml_is_noop(graph->nodes[j]));
    return j;
}

int f8_readers(ggml_backend_cuda_context & ctx, const ggml_cgraph * graph, const ggml_tensor * t) {
    if (t->view_src || (t->flags & (GGML_TENSOR_FLAG_OUTPUT | GGML_TENSOR_FLAG_CROSS_SPLIT))) return -1;
    auto & readers = ctx.f8_readers;
    if (readers.empty()) {
        for (int j = 0; j < graph->n_nodes; ++j) {
            const ggml_tensor * node = graph->nodes[j];
            if (node->view_src) readers[node->view_src] = -1;
            for (int k = 0; k < GGML_MAX_SRC; ++k) {
                if (!node->src[k]) continue;
                int & n = readers[node->src[k]->view_src ? node->src[k]->view_src : node->src[k]];
                if (n >= 0) ++n;
            }
        }
    }
    const auto found = readers.find(t);
    return found == readers.end() ? 0 : found->second;
}

static int ggml_cuda_mul_mat_f8_reuse(ggml_backend_cuda_context & ctx, const ggml_tensor * src1,
        ggml_tensor * dst, const ggml_cgraph * cgraph, int node_n, ggml_tensor * norm = nullptr) {
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    auto compatible = [&](const ggml_tensor * node, const ggml_tensor * input) {
        if (node->op != GGML_OP_MUL_MAT || node->src[1] != input ||
                node->op_params[0] != GGML_PREC_DEFAULT || !f8_w8a8(node->src[0], cc) ||
                !ggml_is_contiguous(node->src[0]) || !ggml_is_contiguous(node) ||
                node->src[0]->ne[2]*node->src[0]->ne[3] != 1) return false;
        for (auto t : {(const ggml_tensor *)node->src[0], (const ggml_tensor *)node->src[1], node}) {
            if (!ggml_backend_buffer_is_cuda(t->buffer)) return false;
        }
        return true;
    };
    if (src1->type != GGML_TYPE_F32 || src1->nb[0] != sizeof(float) || src1->ne[1] <= 8 ||
            src1->ne[2]*src1->ne[3] != 1 || !compatible(dst, src1)) return -1;
    constexpr int G = f8_activation_group;
    const int n = src1->ne[0], cols = src1->ne[1];
    const bool use_lt = f8_lt_supported(ctx, dst);
    float fixed = use_lt ? ggml_get_input_scale(dst->src[0]) : 0.0f;
    for (int j = node_n; fixed > 0 && cgraph;) {
        j = f8_next(cgraph, j);
        if (j >= cgraph->n_nodes || !compatible(cgraph->nodes[j], src1) || f8_lt_supported(ctx, cgraph->nodes[j]) != use_lt) break;
        const float scale = ggml_get_input_scale(cgraph->nodes[j]->src[0]);
        fixed = scale > 0 ? fmaxf(fixed, scale) : 0.0f;
    }
    if (fixed < 0) fixed = 0;
    int chain = 0;
    if (norm) {
        int last = node_n;
        for (int j = node_n; j < cgraph->n_nodes && compatible(cgraph->nodes[j], src1) && f8_lt_supported(ctx, cgraph->nodes[j]) == use_lt;) {
            ++chain;
            last = j;
            j = f8_next(cgraph, j);
        }
        const ggml_tensor * x = norm->src[0]->view_src ? norm->src[0]->view_src : norm->src[0];
        bool live = x->op == GGML_OP_NONE || (x->flags & GGML_TENSOR_FLAG_OUTPUT);
        for (int j = last + 1; j < cgraph->n_nodes && !live; ++j) {
            for (int k = 0; k < GGML_MAX_SRC; ++k) {
                const ggml_tensor * src = cgraph->nodes[j]->src[k];
                if (src && (src == x || src->view_src == x)) live = true;
            }
        }
        if (!live || f8_readers(ctx, cgraph, norm) != chain) return -1;
    }
    ggml_cuda_pool_alloc<uint8_t> activations(ctx.pool(), size_t(cols)*n);
    ggml_cuda_pool_alloc<float> scales(ctx.pool(), size_t(cols)*(use_lt ? 1 : (n+G-1)/G));
    auto stream = ctx.stream();
    if (norm) {
        float eps;
        memcpy(&eps, norm->op_params, sizeof(float));
        auto kernel = use_lt ? f8_norm_quantize<0> : f8_norm_quantize<G>;
        kernel<<<cols,1024,0,stream>>>((const float *)norm->src[0]->data, (const float *)norm->src[1]->data,
                activations.get(), scales.get(), n, norm->src[0]->nb[1]/sizeof(float), eps, fixed);
    } else
#ifdef GGML_CUDA_F8_LT
    if (use_lt) f8_lt_quantize<<<cols,256,0,stream>>>((const float *)src1->data, activations.get(), scales.get(), n, src1->nb[1]/sizeof(float), fixed);
    else
#endif
    f8_quantize<G><<<dim3((n+4*G-1)/(4*G),cols),128,0,stream>>>(
            (const float *)src1->data, activations.get(), scales.get(), n, src1->nb[1]/sizeof(float));
    auto add_after = [&](int j) {
        if (!cgraph) return -1;
        const ggml_tensor * mm = cgraph->nodes[j];
        const int ia = f8_next(cgraph, j);
        if (ia >= cgraph->n_nodes) return -1;
        const ggml_tensor * add = cgraph->nodes[ia];
        if (add->op != GGML_OP_ADD || (add->src[0] != mm && add->src[1] != mm) || add->src[0] == add->src[1]) return -1;
        const ggml_tensor * other = add->src[0] == mm ? add->src[1] : add->src[0];
        if (other->type != GGML_TYPE_F32 || add->type != GGML_TYPE_F32 || !ggml_are_same_shape(mm, other) || !ggml_are_same_shape(mm, add) ||
                !ggml_is_contiguous(other) || !ggml_is_contiguous(add) || uintptr_t(other->data)%16 != 0 || uintptr_t(add->data)%16 != 0) return -1;
        for (auto t : {other, add}) {
            if (!ggml_backend_buffer_is_cuda(t->buffer)) return -1;
        }
        return f8_readers(ctx, cgraph, mm) == 1 ? ia : -1;
    };
    auto other_of = [&](const ggml_tensor * add, const ggml_tensor * mm) { return add->src[0] == mm ? add->src[1] : add->src[0]; };
    ggml_tensor * second = nullptr, * unary = nullptr, * down = nullptr, * up = nullptr, * gate = nullptr;
    int id = 0;
    const int m = dst->ne[0];
    auto ffn = [&](auto supported) {
        const int ig = f8_next(cgraph, node_n), iu = f8_next(cgraph, ig);
        id = f8_next(cgraph, iu);
        if (id >= cgraph->n_nodes) return false;
        second = cgraph->nodes[ig];
        unary = cgraph->nodes[iu];
        down = cgraph->nodes[id];
        const bool swapped = unary->src[0] == dst && unary->src[1] == second;
        up = swapped ? second : dst;
        gate = swapped ? dst : second;
        return compatible(second, src1) && supported(second) && second->ne[0] == m &&
            unary->op == GGML_OP_FUSED_MUL_UNARY && unary->src[0] == gate && unary->src[1] == up &&
            (ggml_unary_op)unary->op_params[0] == GGML_UNARY_OP_SILU && *(const float *)(unary->op_params + 1) < 1e-6f &&
            compatible(down, unary) && supported(down) &&
            f8_readers(ctx, cgraph, dst) == 1 && f8_readers(ctx, cgraph, second) == 1 &&
            f8_readers(ctx, cgraph, unary) == 1;
    };
    auto b128 = [](const ggml_tensor * node) { return node->src[0]->type == GGML_TYPE_F8_E4M3_B128; };
#ifdef GGML_CUDA_F8_LT
    if (use_lt && cgraph && (!norm || chain == 2) && ffn([&](const ggml_tensor * node) { return f8_lt_supported(ctx, node); }) &&
        m*sizeof(float) + 64 <= (size_t)ggml_cuda_info().devices[ctx.device].smpbo) {
        auto & lt = f8_lt_context(ctx);
        ggml_cuda_pool_alloc<float> row_scales(ctx.pool(), 2*m);
        ggml_cuda_pool_alloc<uint8_t> codes(ctx.pool(), size_t(cols)*m);
        ggml_cuda_pool_alloc<float> out_scales(ctx.pool(), cols);
        const int ia = add_after(id);
        const float fixed_down = fmaxf(ggml_get_input_scale(down->src[0]), 0.0f);
        ggml_cuda_pool_alloc<nv_bfloat16> narrow(ctx.pool());
        if (ia >= 0 && lt.get(down->src[0]->ne[1], m, cols, down->ne[0], true).supported) narrow.alloc(size_t(cols)*down->ne[0]);
        if (lt.up_gate(m, n, cols, m, (const char *)up->src[0]->data, (const char *)gate->src[0]->data,
                    activations.get(), scales.get(), (float *)up->data, (float *)gate->data,
                    row_scales.get(), codes.get(), out_scales.get(), stream, fixed_down) &&
            (ia >= 0 ?
             lt.launch_add(down->src[0]->ne[1], m, cols, down->ne[0], (const char *)down->src[0]->data,
                    codes.get(), out_scales.get(), (float *)down->data, (const float *)other_of(cgraph->nodes[ia], down)->data,
                    (float *)cgraph->nodes[ia]->data, stream, narrow.get(), fixed_down > 0) :
             lt.launch(down->src[0]->ne[1], m, cols, down->ne[0], (const char *)down->src[0]->data,
                    codes.get(), out_scales.get(), (float *)down->data, stream, fixed_down > 0))) {
            CUDA_CHECK(cudaGetLastError());
            return ia >= 0 ? ia : id;
        }
    }
#endif
#ifdef F8_TMA_AVAILABLE
    if (const int limit = ggml_cuda_info().devices[ctx.device].smpbo; !use_lt && cgraph && b128(dst) && f8_tma_available(cc) &&
        n/128 <= 256 && dst->src[0]->nb[1]%16 == 0 && m%128 == 0 && m/128 <= 252 &&
        f8_tma_shared<4, true, true>(n/128) <= limit && f8_tma_shared<6, false, true>(m/128) <= limit &&
        ffn(b128) && down->src[0]->nb[1]%16 == 0 && down->ne[0]%128 == 0) {
        ggml_cuda_pool_alloc<uint8_t> codes(ctx.pool(), size_t(cols)*m);
        ggml_cuda_pool_alloc<float> out_scales(ctx.pool(), size_t(cols)*(m/128));
        const size_t stride = dst->src[0]->nb[1];
        const f8_moe_routing routing{nullptr, nullptr, nullptr, 1, stride, stride*m, nullptr, 0, nullptr, nullptr, nullptr, false};
        const int ia = add_after(id);
        const float * other = ia >= 0 ? (const float *)other_of(cgraph->nodes[ia], down)->data : nullptr;
        float * out = (float *)(ia >= 0 ? cgraph->nodes[ia] : down)->data;
        if (launch_f8_tma<4, true, true>(n, m, cols, (const char *)up->src[0]->data, (const char *)gate->src[0]->data,
                    activations.get(), scales.get(), n/128, nullptr, nullptr, codes.get(), out_scales.get(), m/128, routing,
                    nullptr, stream, 8) &&
            (launch_f8_tma_dense<8>(m, down->ne[0], cols, (const char *)down->src[0]->data, down->src[0]->nb[1], codes.get(),
                    out_scales.get(), out, stream, other) ||
             launch_f8_tma_dense<6>(m, down->ne[0], cols, (const char *)down->src[0]->data, down->src[0]->nb[1], codes.get(),
                    out_scales.get(), out, stream, other))) {
            CUDA_CHECK(cudaGetLastError());
            return ia >= 0 ? ia : id;
        }
    }
#endif
    if (!use_lt && cgraph && b128(dst) && cc < 1200 && ffn(b128)) {
        launch_f8_w8a8(up->src[0], n, m, cols, m, (const char *)up->src[0]->data, activations.get(),
                (float *)up->data, scales.get(), stream);
        launch_f8_w8a8(gate->src[0], n, m, cols, m, (const char *)gate->src[0]->data, activations.get(),
                (float *)gate->data, scales.get(), stream);
        ggml_cuda_pool_alloc<uint8_t> codes(ctx.pool(), size_t(cols)*m);
        ggml_cuda_pool_alloc<float> out_scales(ctx.pool(), size_t(cols)*((m+G-1)/G));
        f8_silu_quantize<G><<<dim3((m+4*G-1)/(4*G), cols), 128, 0, stream>>>((const float *)gate->data, (const float *)up->data,
                codes.get(), out_scales.get(), m, m);
        launch_f8_w8a8(down->src[0], m, down->ne[0], cols, down->ne[0], (const char *)down->src[0]->data,
                codes.get(), (float *)down->data, out_scales.get(), stream);
        CUDA_CHECK(cudaGetLastError());
        return id;
    }
    for (;;) {
        const auto * weight = dst->src[0];
        bool done = false;
#ifdef GGML_CUDA_F8_LT
        if (use_lt) {
            auto & lt = f8_lt_context(ctx);
            const int ia = add_after(node_n);
            if (ia >= 0) {
                ggml_cuda_pool_alloc<nv_bfloat16> narrow(ctx.pool());
                if (lt.get(weight->ne[1], n, cols, dst->ne[0], true).supported) narrow.alloc(size_t(cols)*dst->ne[0]);
                if (lt.launch_add(weight->ne[1], n, cols, dst->ne[0], (const char *)weight->data,
                        activations.get(), scales.get(), (float *)dst->data, (const float *)other_of(cgraph->nodes[ia], dst)->data,
                        (float *)cgraph->nodes[ia]->data, stream, narrow.get(), fixed > 0)) {
                    CUDA_CHECK(cudaGetLastError());
                    return ia;
                }
            }
            done = lt.launch(weight->ne[1], n, cols, dst->ne[0], (const char *)weight->data,
                    activations.get(), scales.get(), (float *)dst->data, stream, fixed > 0);
        } else
#endif
        {
            const int ia = add_after(node_n);
            if (ia >= 0 && launch_f8_w8a8(weight, n, weight->ne[1], cols, dst->ne[0],
                        (const char *)weight->data, activations.get(), (float *)cgraph->nodes[ia]->data, scales.get(), stream,
                        (const float *)other_of(cgraph->nodes[ia], dst)->data)) {
                CUDA_CHECK(cudaGetLastError());
                return ia;
            }
            launch_f8_w8a8(weight, n, weight->ne[1], cols, dst->ne[0],
                    (const char *)weight->data, activations.get(), (float *)dst->data, scales.get(), stream);
            done = true;
        }
        if (!done) {
            if (norm) {
                ggml_cuda_op_fused_rms_norm(ctx, norm);
                norm = nullptr;
            }
            ggml_cuda_op_mul_mat(ctx, weight, src1, dst, ggml_cuda_op_mul_mat_f8_e4m3<true>, nullptr);
        }
        CUDA_CHECK(cudaGetLastError());
        if (!cgraph) break;
        const int next = f8_next(cgraph, node_n);
        if (next >= cgraph->n_nodes || !compatible(cgraph->nodes[next], src1) || f8_lt_supported(ctx, cgraph->nodes[next]) != use_lt) break;
        node_n = next;
        dst = cgraph->nodes[node_n];
    }
    return node_n;
}

int ggml_cuda_f8_norm_chain(ggml_backend_cuda_context & ctx, ggml_tensor * norm, const ggml_cgraph * cgraph, int i) {
    const ggml_tensor * x = norm->src[0], * w = norm->src[1];
    if (!w || norm->op_params[1] > 1 || x->type != GGML_TYPE_F32 || norm->type != GGML_TYPE_F32 || w->type != GGML_TYPE_F32 ||
            !ggml_is_contiguous(x) || x->ne[0] < 1024 || x->ne[0] > 16384 || x->ne[0]%32 != 0 || x->ne[2]*x->ne[3] != 1) return -1;
    for (auto t : {x, w}) {
        if (!ggml_backend_buffer_is_cuda(t->buffer)) return -1;
    }
    const int j = f8_next(cgraph, i);
    if (j >= cgraph->n_nodes || cgraph->nodes[j]->op != GGML_OP_MUL_MAT || cgraph->nodes[j]->src[1] != norm ||
            !f8_w8a8(cgraph->nodes[j]->src[0], ggml_cuda_info().devices[ctx.device].cc)) return -1;
    return ggml_cuda_mul_mat_f8_reuse(ctx, norm, cgraph->nodes[j], cgraph, j, norm);
}

static int ggml_cuda_mul_mat_f8_exact(ggml_backend_cuda_context & ctx, const ggml_tensor * src1, ggml_tensor * dst,
        const ggml_cgraph * cgraph, int node_n) {
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    auto compatible = [&](const ggml_tensor * node) {
        const ggml_tensor * w = node->src[0];
        if (node->op != GGML_OP_MUL_MAT || node->src[1] != src1 || node->op_params[0] != GGML_PREC_DEFAULT ||
                (w->type != GGML_TYPE_F8_E4M3_KS && w->type != GGML_TYPE_F8_E4M3_B128) ||
                f8_w8a8(w, cc) ||
                !ggml_is_contiguous(w) || !ggml_is_contiguous(node) || w->ne[2]*w->ne[3] != 1) return false;
        for (auto t : {w, src1, node}) {
            if (!ggml_backend_buffer_is_cuda(t->buffer)) return false;
        }
        return true;
    };
    if (!cgraph || src1->type != GGML_TYPE_F32 || !ggml_is_contiguous(src1) || src1->ne[1] <= 8 || src1->ne[2]*src1->ne[3] != 1 ||
            !new_mma_available(cc) || !compatible(dst)) return -1;
    if (const int next = f8_next(cgraph, node_n); next >= cgraph->n_nodes || !compatible(cgraph->nodes[next])) return -1;
    const int n = src1->ne[0], cols = src1->ne[1];
    auto stream = ctx.stream();
    ggml_cuda_pool_alloc<half> activations(ctx.pool(), size_t(cols)*n);
    ggml_get_to_fp16_cuda(GGML_TYPE_F32)(src1->data, activations.get(), cols, n, stream);
    for (;;) {
        const ggml_tensor * w = dst->src[0];
        if (w->type == GGML_TYPE_F8_E4M3_KS) {
            launch_f8_pp<ggml_cuda_f8_row>(cc, n, w->ne[1], cols, dst->ne[0], (const char *)w->data, activations.get(), (float *)dst->data, stream);
        } else {
            launch_f8_pp<ggml_cuda_f8_block>(cc, n, w->ne[1], cols, dst->ne[0], (const char *)w->data, activations.get(), (float *)dst->data, stream);
        }
        CUDA_CHECK(cudaGetLastError());
        const int next = f8_next(cgraph, node_n);
        if (next >= cgraph->n_nodes || !compatible(cgraph->nodes[next])) break;
        node_n = next;
        dst = cgraph->nodes[node_n];
    }
    return node_n;
}

int ggml_cuda_mul_mat_f8(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst,
        const ggml_cgraph * cgraph, int node_n) {
    f8_w8a16_notice(ctx.device, src0);
    if (f8_w8a8(src0, ggml_cuda_info().devices[ctx.device].cc)) {
        const int last = ggml_cuda_mul_mat_f8_reuse(ctx, src1, dst, cgraph, node_n);
        if (last >= 0) return last;
        ggml_cuda_op_mul_mat(ctx, src0, src1, dst, ggml_cuda_op_mul_mat_f8_e4m3<true>, nullptr);
    } else {
        const int last = ggml_cuda_mul_mat_f8_exact(ctx, src1, dst, cgraph, node_n);
        if (last >= 0) return last;
        ggml_cuda_op_mul_mat(ctx, src0, src1, dst, ggml_cuda_op_mul_mat_f8_e4m3<>, nullptr);
    }
    return node_n;
}

static bool ggml_cuda_f8_aligned(const ggml_tensor * weight, const ggml_tensor * src) {
    return weight->ne[0] % 16 == 0 &&
        uintptr_t(weight->data) % 16 == 0 && weight->nb[1] % 16 == 0 && weight->nb[2] % 16 == 0 &&
        uintptr_t(src->data) % 16 == 0 && src->nb[1] % 16 == 0 && src->nb[2] % 16 == 0;
}

template <int G>
static bool ggml_cuda_mul_mat_f8_id_impl(ggml_backend_cuda_context & ctx, const ggml_tensor * weight,
        const ggml_tensor * src, const ggml_tensor * ids, ggml_tensor * dst, const ggml_tensor * gate = nullptr,
        const ggml_tensor * bias_up = nullptr, const ggml_tensor * bias_gate = nullptr, float * up = nullptr,
        ggml_tensor * next = nullptr, nv_bfloat16 ** narrow = nullptr, ggml_cuda_pool_alloc<int32_t> * routing_data = nullptr) {
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    if (src->ne[2] <= 8 || !new_mma_available(cc) || (!gate && dst->op_params[0] != GGML_PREC_DEFAULT)) return false;
    if (!ggml_is_contiguous(dst) || src->nb[0] != sizeof(float)) return false;
    for (auto tensor : {weight, gate, src, ids, (const ggml_tensor *)dst, bias_up, bias_gate}) {
        if (!tensor) continue;
        if (!ggml_backend_buffer_is_cuda(tensor->buffer)) return false;
    }
    const bool blocked = weight->type == GGML_TYPE_F8_E4M3_B128;
    const int cols = ids->ne[0]*ids->ne[1], experts = weight->ne[2];
    const bool wide = cols > 32*experts;
    auto stream = ctx.stream();
    const int chunks = (cols+255)/256;
    ggml_cuda_pool_alloc<int32_t> local_routing(ctx.pool());
    if (!routing_data) routing_data = &local_routing;
    const bool prepare = !routing_data->get();
    if (prepare) routing_data->alloc(size_t(2*cols + 4*(experts+1)) + size_t(chunks)*experts);
    auto rows = routing_data->get();
    auto bounds = rows + cols;
    auto tiles = bounds + experts + 1;
    auto counts = tiles + 3*(experts + 1);
    auto ranks = counts + size_t(chunks)*experts;
    ggml_cuda_pool_alloc<f8_activation<G>> activations(ctx.pool(), size_t(cols)*src->ne[0]);
#ifdef F8_TMA_AVAILABLE
    auto tiles_ug = tiles + experts + 1;
    auto tiles_dn = tiles_ug + experts + 1;
    bool tma = false;
    if constexpr (G != 0) {
        const int n = src->ne[0], m = dst->ne[0];
        if (next && f8_tma_available(cc) && n%128 == 0 && (n/128)%4 == 0 && n/128 <= 256 && m%128 == 0 &&
            m/128 <= 252 && next->ne[0]%128 == 0 &&
            size_t(blocked ? std::max(f8_tma_shared<4, true, true>(n/128), f8_tma_shared<8, false, true>((m/128+3)/4*4)) :
                std::max(f8_tma_shared<6, true>(n/128), f8_tma_shared<8, false>((m/128+3)/4*4))) <= ggml_cuda_info().devices[ctx.device].smpbo) {
            tma = true;
            for (auto w : {weight, gate, (const ggml_tensor *)next->src[0]}) {
                if (w->type != weight->type || uintptr_t(w->data)%16 != 0 || (!blocked && w->nb[1]%16 != 0) || w->nb[2]%w->nb[1] != 0) tma = false;
            }
        }
    }
    const int width_ug = tma ? (blocked ? 64 : 96) : 0;
#else
    const int width_ug = 0;
#endif
    if (prepare) {
        f8_moe_histogram<<<chunks,256,0,stream>>>((const char *)ids->data, counts, ranks,
                ids->ne[0], ids->ne[1], experts, ids->nb[0], ids->nb[1]);
        f8_moe_prefix<<<1,256,4*experts*sizeof(int32_t),stream>>>(counts, bounds, tiles, experts, chunks, wide ? 64 : 32, width_ug, 128);
        f8_moe_scatter<<<chunks,256,0,stream>>>((const char *)ids->data, counts, ranks, bounds, rows,
                ids->ne[0], ids->ne[1], experts, ids->nb[0], ids->nb[1]);
    }
    auto run = [&](const float * scales) {
        auto multiply = [&](const ggml_tensor * w, const ggml_tensor * bias, float * output, int n, int m,
                const f8_activation<G> * acts, const float * act_scales, int mode = 0, uint8_t * codes = nullptr, float * out_scales = nullptr) {
            f8_moe_routing routing{rows, bounds, tiles, experts, w->nb[1], w->nb[2],
                    bias ? (const char *)bias->data : nullptr, bias ? bias->nb[1] : 0, nullptr, nullptr, nullptr, false};
            routing.sorted = mode == 1;
            if (mode == 2) {
                routing.up = up;
                routing.codes = codes;
                routing.out_scales = out_scales;
            }
            auto launch = [&](auto layout) {
                using L = decltype(layout);
                if (wide) {
                    launch_f8_moe<L,64,4,1,128,G>(n, m, cols,
                            (const char *)w->data, acts, output, routing, stream, act_scales);
                } else {
                    launch_f8_moe<L,32,4,1,128,G>(n, m, cols,
                            (const char *)w->data, acts, output, routing, stream, act_scales);
                }
            };
            if (w->type == GGML_TYPE_F8_E4M3_B128) launch(ggml_cuda_f8_block{});
            else launch(ggml_cuda_f8_row{});
        };
        const int n = src->ne[0], m = dst->ne[0];
        if (next) {
#ifdef F8_TMA_AVAILABLE
            if constexpr (G != 0) {
                if (tma) {
                    const int out_stride = (m/128+3)/4*4;
                    ggml_cuda_pool_alloc<uint8_t> codes(ctx.pool(), size_t(cols)*m);
                    ggml_cuda_pool_alloc<float> out_scales(ctx.pool(), size_t(cols)*out_stride);
                    f8_moe_routing ug{rows, bounds, tiles_ug, experts, weight->nb[1], weight->nb[2],
                            bias_up ? (const char *)bias_up->data : nullptr, bias_up ? bias_up->nb[1] : 0, nullptr, nullptr, nullptr, false};
                    const ggml_tensor * down = next->src[0];
                    f8_moe_routing dn{rows, bounds, tiles_dn, experts, down->nb[1], down->nb[2], nullptr, 0, nullptr, nullptr, nullptr, false};
                    if (blocked) {
                        if (launch_f8_tma<4, true, true>(n, m, cols, (const char *)weight->data, (const char *)gate->data, activations.get(), scales, n/128,
                                    nullptr, nullptr, codes.get(), out_scales.get(), out_stride, ug,
                                    bias_gate ? (const char *)bias_gate->data : nullptr, stream) &&
                            launch_f8_tma<8, false, true>(m, next->ne[0], cols, (const char *)down->data, nullptr, codes.get(), out_scales.get(), out_stride,
                                    (float *)next->data, narrow ? *narrow : nullptr, nullptr, nullptr, 0, dn, nullptr, stream)) return;
                    } else if (launch_f8_tma<6, true>(n, m, cols, (const char *)weight->data, (const char *)gate->data, activations.get(), scales, n/128,
                                nullptr, nullptr, codes.get(), out_scales.get(), out_stride, ug,
                                bias_gate ? (const char *)bias_gate->data : nullptr, stream) &&
                        launch_f8_tma<8, false>(m, next->ne[0], cols, (const char *)down->data, nullptr, codes.get(), out_scales.get(), out_stride,
                                (float *)next->data, narrow ? *narrow : nullptr, nullptr, nullptr, 0, dn, nullptr, stream)) return;
                }
            }
#endif
            if (narrow) *narrow = nullptr;
            if constexpr (G != 0) {
                ggml_cuda_pool_alloc<float> up_data(ctx.pool(), size_t(cols)*m);
                up = up_data.get();
                ggml_cuda_pool_alloc<uint8_t> codes(ctx.pool(), size_t(cols)*m);
                ggml_cuda_pool_alloc<float> out_scales(ctx.pool(), size_t(cols)*(m/128));
                multiply(weight, bias_up, up, n, m, activations.get(), scales, 1);
                multiply(gate, bias_gate, nullptr, n, m, activations.get(), scales, 2, codes.get(), out_scales.get());
                multiply(next->src[0], nullptr, (float *)next->data, m, next->ne[0], codes.get(), out_scales.get());
            }
            return;
        }
        multiply(weight, bias_up, gate ? up : (float *)dst->data, n, m, activations.get(), scales);
        if (gate) multiply(gate, bias_gate, (float *)dst->data, n, m, activations.get(), scales);
    };
    if constexpr (G != 0) {
        ggml_cuda_pool_alloc<float> scales(ctx.pool(), size_t(cols)*((src->ne[0]+G-1)/G));
        f8_moe_convert<G><<<cols,256,0,stream>>>((const char *)src->data, activations.get(), rows,
                src->ne[0], ids->ne[0], src->ne[1], src->nb[1], src->nb[2], scales.get());
        run(scales.get());
    } else {
        f8_moe_convert<<<cols,256,0,stream>>>((const char *)src->data, activations.get(), rows,
                src->ne[0], ids->ne[0], src->ne[1], src->nb[1], src->nb[2]);
        run(nullptr);
    }
    CUDA_CHECK(cudaGetLastError());
    return true;
}

static bool ggml_cuda_mul_mat_f8_id(ggml_backend_cuda_context & ctx, const ggml_tensor * weight,
        const ggml_tensor * src, const ggml_tensor * ids, ggml_tensor * dst, const ggml_tensor * gate = nullptr,
        const ggml_tensor * bias_up = nullptr, const ggml_tensor * bias_gate = nullptr, float * up = nullptr,
        ggml_tensor * next = nullptr, nv_bfloat16 ** narrow = nullptr, ggml_cuda_pool_alloc<int32_t> * routing_data = nullptr) {
    if (f8_w8a8(weight, ggml_cuda_info().devices[ctx.device].cc) &&
            (!gate || ggml_get_input_scale(gate) != 0.0f)) {
        return ggml_cuda_mul_mat_f8_id_impl<f8_activation_group>(ctx, weight, src, ids, dst, gate, bias_up, bias_gate, up, next, narrow, routing_data);
    }
    if (next) return false;
    return ggml_cuda_mul_mat_f8_id_impl<0>(ctx, weight, src, ids, dst, gate, bias_up, bias_gate, up, nullptr, nullptr, routing_data);
}

bool ggml_cuda_mul_mat_id_f8(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const ggml_tensor * ids  = dst->src[2];
    f8_w8a16_notice(ctx.device, src0);

    if (ggml_cuda_mul_mat_f8_id(ctx, src0, src1, ids, dst)) return true;
    if (src1->ne[2] > 8 || !ggml_backend_buffer_is_cuda(src0->buffer) || !ggml_backend_buffer_is_cuda(src1->buffer) ||
        !ggml_backend_buffer_is_cuda(dst->buffer)) return false;

    const bool aligned = ggml_cuda_f8_aligned(src0, src1);
    const bool warp = aligned || src0->type == GGML_TYPE_F8_E4M3_B128;
    const dim3 grid(warp ? (src0->ne[1] + 7)/8 : src0->ne[1], ids->ne[0], ids->ne[1]);
    auto launch = [&](auto kernel) {
        kernel<<<grid, 256, 0, ctx.stream()>>>(
            src0->ne[0], src1->ne[1], (const char *)src0->data, (const char *)src1->data,
            (const char *)ids->data, (char *)dst->data, src0->nb[1], src0->nb[2], src1->nb[1], src1->nb[2],
            ids->nb[0], ids->nb[1], dst->nb[1], dst->nb[2], src0->ne[1]);
    };
    if (src0->type == GGML_TYPE_F8_E4M3_B128 && aligned) launch(mul_mat_vec_f8_e4m3_id<true, true>);
    else if (src0->type == GGML_TYPE_F8_E4M3_B128) launch(mul_mat_vec_f8_e4m3_id<true>);
    else if (aligned) launch(mul_mat_vec_f8_e4m3_id<false, true>);
    else launch(mul_mat_vec_f8_e4m3_id<false>);
    CUDA_CHECK(cudaGetLastError());
    return true;
}

int ggml_cuda_moe_up_gate_f8(ggml_backend_cuda_context & ctx, ggml_tensor * dst, const ggml_cgraph * graph, int i,
        ggml_tensor * next, const ggml_tensor * hada) {
    const ggml_tensor * src0_1 = dst->src[0];
    const ggml_tensor * src0_2 = dst->src[1];
    const ggml_tensor * src1 = dst->src[2];
    const ggml_tensor * ids  = dst->src[3];
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    f8_w8a16_notice(ctx.device, src0_1);
    if (!src0_2) {
        auto gate = *src0_1;
        gate.ne[1] /= 2;
        auto up = gate;
        up.data = (char *)up.data + up.ne[1]*up.nb[1];
        auto local_dst = *dst;
        local_dst.src[0] = &up;
        local_dst.src[1] = &gate;
        ggml_tensor bias_up, bias_gate;
        if (dst->src[4]) {
            bias_gate = *dst->src[4];
            bias_gate.ne[0] /= 2;
            bias_up = bias_gate;
            bias_up.data = (char *)bias_up.data + bias_up.ne[0]*bias_up.nb[0];
            local_dst.src[4] = &bias_up;
            local_dst.src[5] = &bias_gate;
        }
        return ggml_cuda_moe_up_gate_f8(ctx, &local_dst, graph, i, next, hada);
    }
    if (src1->ne[2] > 8) {
        const auto unary = (ggml_unary_op)dst->op_params[0];
        ggml_tensor * fused = nullptr;
        const bool down = !hada && next && next->op == GGML_OP_MUL_MAT_ID && next->src[1]->data == dst->data && next->src[2] == ids &&
            (next->src[0]->type == GGML_TYPE_F8_E4M3_KS || next->src[0]->type == GGML_TYPE_F8_E4M3_B128);
        if (down && unary == GGML_UNARY_OP_SILU && *(const float *)(dst->op_params + 1) < 1e-6f && dst->ne[0]%128 == 0 &&
            f8_w8a8(src0_1, cc) && f8_w8a8(src0_2, cc) && f8_w8a8(next->src[0], cc) &&
            ggml_backend_buffer_is_cuda(next->src[0]->buffer) && ggml_backend_buffer_is_cuda(next->buffer) &&
            ggml_is_contiguous(next) && !(dst->flags & GGML_TENSOR_FLAG_OUTPUT) &&
            f8_readers(ctx, graph, next->src[1]) == 1) {
            fused = next;
        }
        ggml_tensor * sum = nullptr, * add = nullptr;
        const ggml_tensor * residual = nullptr;
        const auto local = [&](const ggml_tensor * t) {
            return t->type == GGML_TYPE_F32 && ggml_is_contiguous(t) && ggml_backend_buffer_is_cuda(t->buffer);
        };
        const auto apart = [](const ggml_tensor * a, const ggml_tensor * b) {
            const char * pa = (const char *)a->data, * pb = (const char *)b->data;
            return pa + ggml_nbytes(a) <= pb || pb + ggml_nbytes(b) <= pa;
        };
        if (fused && i + 2 < graph->n_nodes) {
            ggml_tensor * node = graph->nodes[i + 2];
            if (node->op == GGML_OP_MUL_MULTI_ADD && node->src[0] == fused && !node->src[2] &&
                local(node) && local(node->src[1]) && f8_readers(ctx, graph, fused) == 1) {
                sum = node;
                node = i + 3 < graph->n_nodes ? graph->nodes[i + 3] : nullptr;
                if (node && node->op == GGML_OP_ADD && (node->src[0] == sum) != (node->src[1] == sum)) {
                    residual = node->src[node->src[0] == sum ? 1 : 0];
                    if (ggml_are_same_shape(residual, sum) && local(node) && local(residual) && apart(node, sum->src[1]) &&
                        f8_readers(ctx, graph, sum) == 1) {
                        add = node;
                    }
                }
            }
        }
        ggml_cuda_pool_alloc<nv_bfloat16> narrow_data(ctx.pool());
        nv_bfloat16 * narrow = sum && cc >= 1200 ? narrow_data.alloc(ggml_nelements(fused)) : nullptr;
        if (fused && ggml_cuda_mul_mat_f8_id(ctx, src0_1, src1, ids, dst, src0_2, dst->src[4], dst->src[5], nullptr, fused,
                    narrow ? &narrow : nullptr)) {
            CUDA_CHECK(cudaGetLastError());
            if (!narrow) return i + 1;
            const int n = ggml_nelements(sum);
            f8_moe_sum<<<(n + 255)/256, 256, 0, ctx.stream()>>>(fused->ne[1], sum->ne[0], n, narrow, (const float *)sum->src[1]->data,
                    add ? (const float *)residual->data : nullptr, (float *)(add ? add : sum)->data);
            CUDA_CHECK(cudaGetLastError());
            return add ? i + 3 : i + 2;
        }
        ggml_cuda_pool_alloc<float> up(ctx.pool(), ggml_nelements(dst));
        ggml_cuda_pool_alloc<int32_t> routing_data(ctx.pool());
        if (ggml_cuda_mul_mat_f8_id(ctx, src0_1, src1, ids, dst, src0_2, dst->src[4], dst->src[5], up.get(), nullptr, nullptr, &routing_data)) {
            if (unary == GGML_UNARY_OP_SWIGLU_OAI) {
                ggml_swiglu_oai_cuda_f32((const float *)dst->data, up.get(), (float *)dst->data,
                        ggml_nelements(dst), dst->ne[0], dst->ne[0], dst->ne[0], 1.702f, 7.0f, ctx.stream());
            } else {
                ggml_fused_mul_unary(ctx, unary, ggml_nelements(dst), (const float *)dst->data, up.get(),
                        (float *)dst->data, *(const float *)(dst->op_params + 1));
            }
            CUDA_CHECK(cudaGetLastError());
            if (down && ggml_cuda_mul_mat_f8_id(ctx, next->src[0], next->src[1], ids, next,
                    nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, &routing_data)) return i + 1;
            return i;
        }
    }
    if (src1->ne[2] <= 8) {
        bool same_device = true;
        for (auto tensor : {src0_1, src0_2, src1, (const ggml_tensor *)dst}) {
            same_device = same_device && ggml_backend_buffer_is_cuda(tensor->buffer);
        }
        if (same_device) {
            const auto unary = (ggml_unary_op)dst->op_params[0];
            const auto * bu = dst->src[4];
            const auto * bg = dst->src[5];
            const bool aligned = ggml_cuda_f8_aligned(src0_1, src1) && ggml_cuda_f8_aligned(src0_2, src1);
            const dim3 grid(aligned ? (dst->ne[0] + 7)/8 : dst->ne[0], ids->ne[0], ids->ne[1]);
            auto launch = [&](auto kernel) {
                kernel<<<grid, 256, 0, ctx.stream()>>>(src1->ne[0], dst->ne[0], src1->ne[1],
                        (const char *)src0_1->data, (const char *)src0_2->data, (const char *)src1->data,
                        (const char *)ids->data, bu ? (const char *)bu->data : nullptr, bg ? (const char *)bg->data : nullptr,
                        (float *)dst->data, unary, *(const float *)(dst->op_params + 1), src0_1->nb[1], src0_1->nb[2], src0_2->nb[1], src0_2->nb[2],
                        src1->nb[1], src1->nb[2], ids->nb[0], ids->nb[1], bu ? bu->nb[1] : 0, bg ? bg->nb[1] : 0);
            };
            if (src0_1->type == GGML_TYPE_F8_E4M3_B128 && aligned) launch(mul_mat_vec_f8_e4m3_up_gate_id<true, true>);
            else if (src0_1->type == GGML_TYPE_F8_E4M3_B128) launch(mul_mat_vec_f8_e4m3_up_gate_id<true>);
            else if (aligned) launch(mul_mat_vec_f8_e4m3_up_gate_id<false, true>);
            else launch(mul_mat_vec_f8_e4m3_up_gate_id<false>);
            CUDA_CHECK(cudaGetLastError());
            return i;
        }
    }
    return -1;
}

void ggml_cuda_f8_lt_prepare(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph) {
#ifdef GGML_CUDA_F8_LT
    f8_lt_prepare(ctx, cgraph);
#else
    GGML_UNUSED(ctx);
    GGML_UNUSED(cgraph);
#endif
}

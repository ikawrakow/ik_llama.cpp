#include "llama-build-context.h"
#include "llama-hparams.h"
#include "llama-cparams.h"
#include "llama-model.h"
#include "llama-context.h"

#include "ggml.h"

llm_build_context::llm_build_context(
        llama_context  & lctx,
    const llama_batch  & batch,
    const llm_build_cb & cb,
    bool   worst_case,
    bool   warmup) :
        model            (lctx.model),
        lctx             (lctx),
        hparams          (model.hparams),
        cparams          (lctx.cparams),
        batch            (batch),
        kv_self          (lctx.kv_self),
        n_embd           (hparams.n_embd),
        n_layer          (hparams.n_layer),
        n_rot            (hparams.n_rot),
        n_ctx            (cparams.n_ctx),
        n_head           (hparams.n_head()),
        n_head_kv        (hparams.n_head_kv()),
        n_embd_head_k    (hparams.n_embd_head_k),
        n_embd_k_gqa     (hparams.n_embd_k_gqa()),
        n_embd_head_v    (hparams.n_embd_head_v),
        n_embd_v_gqa     (hparams.n_embd_v_gqa()),
        n_expert         (hparams.n_expert),
        n_expert_used    (warmup ? hparams.n_expert : hparams.n_expert_used),
        freq_base        (cparams.rope_freq_base),
        freq_scale       (cparams.rope_freq_scale),
        ext_factor       (cparams.yarn_ext_factor),
        attn_factor      (cparams.yarn_attn_factor),
        beta_fast        (cparams.yarn_beta_fast),
        beta_slow        (cparams.yarn_beta_slow),
        norm_eps         (hparams.f_norm_eps),
        norm_rms_eps     (hparams.f_norm_rms_eps),
        n_tokens         (batch.n_tokens),
        n_kv             (worst_case ? kv_self.size : kv_self.n),
        n_outputs        (worst_case ? n_tokens : lctx.n_outputs),
        n_outputs_enc    (worst_case ? n_tokens : lctx.embd_enc.size() / hparams.n_embd),
        kv_head          (worst_case ? (kv_self.recurrent ? 0 : kv_self.size - n_tokens) : kv_self.head),
        n_ctx_orig       (cparams.n_ctx_orig_yarn),
        flash_attn       (cparams.flash_attn),
        mla_attn         (cparams.mla_attn),
        attn_max_batch   (cparams.attn_max_batch),
        fused_moe_up_gate(cparams.fused_moe_up_gate),
        grouped_expert_routing(cparams.grouped_expert_routing),
        fused_up_gate    (cparams.fused_up_gate),
        fused_mmad       (cparams.fused_mmad),
        rope_cache       (cparams.rope_cache),
        k_cache_hadamard (cparams.k_cache_hadamard),
        split_mode_graph_scheduling (cparams.split_mode_graph_scheduling),
        min_experts      (cparams.min_experts),
        thresh_experts   (cparams.thresh_experts),
        pooling_type     (cparams.pooling_type),
        rope_type        (hparams.rope_type),
        cb               (cb),
        buf_compute_meta (lctx.buf_compute_meta) {
            // all initializations should be done in init()
}

void llm_build_context::init() {
    struct ggml_init_params params = {
        /*.mem_size   =*/ buf_compute_meta.size(),
        /*.mem_buffer =*/ buf_compute_meta.data(),
        /*.no_alloc   =*/ true,
    };

    ctx0 = ggml_init(params);

    lctx.inp_tokens      = nullptr;
    lctx.inp_embd        = nullptr;
    lctx.inp_pos         = nullptr;
    lctx.inp_out_ids     = nullptr;
    lctx.inp_KQ_mask     = nullptr;
    lctx.inp_KQ_mask_swa = nullptr;
    lctx.inp_K_shift     = nullptr;
    lctx.inp_mean        = nullptr;
    lctx.inp_cls         = nullptr;
    lctx.inp_s_copy      = nullptr;
    lctx.inp_s_mask      = nullptr;
    lctx.inp_s_seq       = nullptr;
    lctx.inp_pos_bucket    = nullptr;
    lctx.inp_embd_enc      = nullptr;
    lctx.inp_KQ_mask_cross = nullptr;
}

void llm_build_context::free() {
    if (ctx0) {
        ggml_free(ctx0);
        ctx0 = nullptr;
    }
}

ggml_cgraph * llm_build_context::build_k_shift() {
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, model.max_nodes(), false);

    GGML_ASSERT(kv_self.size == n_ctx);

    const auto & rope_type_shift = hparams.rope_type == LLAMA_ROPE_TYPE_MROPE
        // @ngxson : this is a workaround
        // for M-RoPE, we want to rotate the whole vector when doing KV shift
        // a normal RoPE should work, we just need to use the correct ordering
        // ref: https://github.com/ggml-org/llama.cpp/pull/13870
        ? LLAMA_ROPE_TYPE_NEOX
        : hparams.rope_type;

    const float yarn_attn_factor_shift = model.arch == LLM_ARCH_DEEPSEEK2
        ? 1.0f / (1.0f + 0.1f * logf(1.0f / freq_scale))
        : cparams.yarn_attn_factor;

    lctx.inp_K_shift = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_ctx);
    cb(lctx.inp_K_shift, "K_shift", -1);
    ggml_set_input(lctx.inp_K_shift);

    for (int il = 0; il < n_layer; ++il) {
        const int64_t n_head_kv = hparams.n_head_kv(il);
        const int64_t n_embd_k_gqa = hparams.n_embd_k_gqa(il);
        struct ggml_tensor * rope_factors = build_rope_factors(il);
        struct ggml_tensor * k =
            ggml_view_3d(ctx0, kv_self.k_l[il],
                    n_embd_head_k, n_head_kv, n_ctx,
                    ggml_row_size(kv_self.k_l[il]->type, n_embd_head_k),
                    ggml_row_size(kv_self.k_l[il]->type, n_embd_k_gqa),
                    0);

        struct ggml_tensor * tmp;
        if (ggml_is_quantized(k->type)) {
            // dequantize to f32 -> RoPE -> quantize back
            tmp = ggml_cast(ctx0, k, GGML_TYPE_F32);
            cb(tmp, "K_f32", il);
            for (auto * backend : lctx.backends) {
                // Figure out which backend KV cache belongs to
                if (ggml_backend_supports_buft(backend, lctx.model.buft_layer[il].buft)) {
                    ggml_backend_sched_set_tensor_backend(lctx.sched, tmp, backend);
                    break;
                }
            }
            tmp = ggml_rope_ext_inplace(ctx0, tmp,
                    lctx.inp_K_shift, rope_factors, n_rot, rope_type_shift, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, yarn_attn_factor_shift, beta_fast, beta_slow);
            cb(tmp, "K_shifted_f32", il);
            tmp = ggml_cpy(ctx0, tmp, k);
        } else {
            // we rotate only the first n_rot dimensions
            tmp = ggml_rope_ext_inplace(ctx0, k,
                    lctx.inp_K_shift, rope_factors, n_rot, rope_type_shift, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, yarn_attn_factor_shift, beta_fast, beta_slow);
        }
        cb(tmp, "K_shifted", il);
        ggml_build_forward_expand(gf, tmp);
    }

    return gf;
}

ggml_cgraph * llm_build_context::build_s_copy() {
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, model.max_nodes(), false);

    GGML_ASSERT(kv_self.recurrent);

    struct ggml_tensor * state_copy = build_inp_s_copy();

    for (int il = 0; il < n_layer; ++il) {
        struct ggml_tensor * conv_states = ggml_reshape_2d(ctx0, kv_self.k_l[il], hparams.n_embd_k_s(), kv_self.size);
        struct ggml_tensor * ssm_states  = ggml_reshape_2d(ctx0, kv_self.v_l[il], hparams.n_embd_v_s(), kv_self.size);

        conv_states = ggml_get_rows(ctx0, conv_states, state_copy);
        ssm_states  = ggml_get_rows(ctx0,  ssm_states, state_copy);

        // TODO: name the intermediate tensors with cb()

        ggml_build_forward_expand(gf, ggml_cpy(ctx0, conv_states, kv_self.k_l[il]));
        ggml_build_forward_expand(gf, ggml_cpy(ctx0,  ssm_states, kv_self.v_l[il]));
    }

    return gf;
}

ggml_cgraph * llm_build_context::build_defrag(const std::vector<uint32_t> & ids) {
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, model.max_nodes(), false);

    for (uint32_t i = 0; i < ids.size(); ++i) {
        const uint32_t id = ids[i];

        if (i == id || id == ids.size()) {
            continue;
        }

        uint32_t nm = 1;

        while (i + nm < ids.size() && ids[i + nm] == id + nm) {
            nm++;
        }

        for (int il = 0; il < n_layer; ++il) {
            const int64_t n_embd_k_gqa = hparams.n_embd_k_gqa(il);
            const int64_t n_embd_v_gqa = hparams.n_embd_v_gqa(il);

            ggml_tensor * view_k_src = ggml_view_2d(ctx0, kv_self.k_l[il],
                    n_embd_k_gqa, nm,
                    ggml_row_size(kv_self.k_l[il]->type, n_embd_k_gqa),
                    ggml_row_size(kv_self.k_l[il]->type, n_embd_k_gqa*i));

            ggml_tensor * view_k_dst = ggml_view_2d(ctx0, kv_self.k_l[il],
                    n_embd_k_gqa, nm,
                    ggml_row_size(kv_self.k_l[il]->type, n_embd_k_gqa),
                    ggml_row_size(kv_self.k_l[il]->type, n_embd_k_gqa*id));

            ggml_tensor * view_v_src = nullptr;
            ggml_tensor * view_v_dst = nullptr;

            if (kv_self.v_l.size() > il) {
                // Note: with MLA the V cache may not be present.
                if (flash_attn) {
                    // NOTE: the V cache is not transposed when using flash attention
                    view_v_src = ggml_view_2d(ctx0, kv_self.v_l[il],
                            n_embd_v_gqa, nm,
                            ggml_row_size(kv_self.v_l[il]->type, n_embd_v_gqa),
                            ggml_row_size(kv_self.v_l[il]->type, n_embd_v_gqa*i));

                    view_v_dst = ggml_view_2d(ctx0, kv_self.v_l[il],
                            n_embd_v_gqa, nm,
                            ggml_row_size(kv_self.v_l[il]->type, n_embd_v_gqa),
                            ggml_row_size(kv_self.v_l[il]->type, n_embd_v_gqa*id));
                } else {
                    view_v_src = ggml_view_2d(ctx0, kv_self.v_l[il],
                            nm, n_embd_v_gqa,
                            ggml_row_size(kv_self.v_l[il]->type, kv_self.size),
                            ggml_row_size(kv_self.v_l[il]->type, i));

                    view_v_dst = ggml_view_2d(ctx0, kv_self.v_l[il],
                            nm, n_embd_v_gqa,
                            ggml_row_size(kv_self.v_l[il]->type, kv_self.size),
                            ggml_row_size(kv_self.v_l[il]->type, id));
                }
            }

            ggml_build_forward_expand(gf, ggml_cpy(ctx0, view_k_src, view_k_dst));
            if (view_v_src && view_v_dst) {
                ggml_build_forward_expand(gf, ggml_cpy(ctx0, view_v_src, view_v_dst));
            }
        }

        i += nm - 1;
    }

    //LLAMA_LOG_INFO("gf->n_nodes = %d\n", gf->n_nodes);

    return gf;
}

ggml_tensor * llm_build_context::build_inp_pos() {
    int n_pos_per_embd = hparams.rope_type == LLAMA_ROPE_TYPE_MROPE || hparams.rope_type == LLAMA_ROPE_TYPE_IMROPE ? 4 : 1;
    lctx.inp_pos = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, int64_t(n_tokens)*n_pos_per_embd);
    cb(lctx.inp_pos, "inp_pos", -1);
    ggml_set_input(lctx.inp_pos);
    return lctx.inp_pos;
}

ggml_tensor * llm_build_context::build_input_scale(int n_tokens) {
    int n_pos_per_token = 1;
    lctx.inp_scale = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, 1, 1, n_tokens*n_pos_per_token);
    cb(lctx.inp_scale, "inp_scale", -1);
    ggml_set_input(lctx.inp_scale);
    return lctx.inp_scale;
}

ggml_tensor * llm_build_context::build_rope_factors(int il) {
    // choose long/short freq factors based on the context size
    const auto n_ctx_pre_seq = cparams.n_ctx / cparams.n_seq_max;

    if (model.layers[il].rope_freqs != nullptr) {
        return model.layers[il].rope_freqs;
    }

    if (n_ctx_pre_seq > hparams.n_ctx_orig_yarn) {
        return model.layers[il].rope_long;
    }

    return model.layers[il].rope_short;
}

ggml_tensor * llm_build_context::build_inp_out_ids() {
    lctx.inp_out_ids = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_outputs);
    cb(lctx.inp_out_ids, "inp_out_ids", -1);
    ggml_set_input(lctx.inp_out_ids);
    return lctx.inp_out_ids;
}

ggml_tensor * llm_build_context::build_inp_KQ_mask(bool causal) {
    if (causal && flash_attn) {
        lctx.inp_KQ_mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F16, n_kv, GGML_PAD(n_tokens, GGML_KQ_MASK_PAD));
        cb(lctx.inp_KQ_mask, "KQ_mask", -1);
        ggml_set_input(lctx.inp_KQ_mask);
        return lctx.inp_KQ_mask;
    }
    lctx.inp_KQ_mask = causal
        ? ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_kv,     GGML_PAD(n_tokens, GGML_KQ_MASK_PAD))
        : ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_tokens, GGML_PAD(n_tokens, GGML_KQ_MASK_PAD));
    cb(lctx.inp_KQ_mask, "KQ_mask", -1);
    ggml_set_input(lctx.inp_KQ_mask);

    return flash_attn ? ggml_cast(ctx0, lctx.inp_KQ_mask, GGML_TYPE_F16) : lctx.inp_KQ_mask;
}

ggml_tensor * llm_build_context::build_inp_KQ_mask_swa(bool causal) {
    GGML_ASSERT(hparams.n_swa > 0);
    if (causal && flash_attn) {
        lctx.inp_KQ_mask_swa = ggml_new_tensor_2d(ctx0, GGML_TYPE_F16, n_kv, GGML_PAD(n_tokens, GGML_KQ_MASK_PAD));
        cb(lctx.inp_KQ_mask_swa, "KQ_mask_swa", -1);
        ggml_set_input(lctx.inp_KQ_mask_swa);
        return lctx.inp_KQ_mask_swa;
    }

    lctx.inp_KQ_mask_swa = causal
        ? ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_kv,     GGML_PAD(n_tokens, GGML_KQ_MASK_PAD))
        : ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_tokens, GGML_PAD(n_tokens, GGML_KQ_MASK_PAD));
    cb(lctx.inp_KQ_mask_swa, "KQ_mask_swa", -1);
    ggml_set_input(lctx.inp_KQ_mask_swa);

    return flash_attn ? ggml_cast(ctx0, lctx.inp_KQ_mask_swa, GGML_TYPE_F16) : lctx.inp_KQ_mask_swa;
}

ggml_tensor * llm_build_context::build_inp_mean() {
    lctx.inp_mean = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_tokens, n_tokens);
    cb(lctx.inp_mean, "inp_mean", -1);
    ggml_set_input(lctx.inp_mean);
    return lctx.inp_mean;
}

ggml_tensor * llm_build_context::build_inp_cls() {
    lctx.inp_cls = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    cb(lctx.inp_cls, "inp_cls", -1);
    ggml_set_input(lctx.inp_cls);
    return lctx.inp_cls;
}

ggml_tensor * llm_build_context::build_inp_s_copy() {
    lctx.inp_s_copy = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, kv_self.size);
    cb(lctx.inp_s_copy, "inp_s_copy", -1);
    ggml_set_input(lctx.inp_s_copy);
    return lctx.inp_s_copy;
}

ggml_tensor * llm_build_context::build_inp_s_mask() {
    lctx.inp_s_mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 1, n_kv);
    cb(lctx.inp_s_mask, "inp_s_mask", -1);
    ggml_set_input(lctx.inp_s_mask);
    return lctx.inp_s_mask;
}

ggml_tensor * llm_build_context::build_inp_s_seq() {
    lctx.inp_s_seq = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, n_kv, n_tokens);
    cb(lctx.inp_s_seq, "inp_s_seq", -1);
    ggml_set_input(lctx.inp_s_seq);
    return lctx.inp_s_seq;
}

ggml_cgraph * llm_build_context::append_pooling(struct ggml_cgraph * gf) {
    // find result_norm tensor for input
    struct ggml_tensor * inp = nullptr;
    for (int i = gf->n_nodes - 1; i >= 0; --i) {
        inp = gf->nodes[i];
        if (strcmp(inp->name, "result_norm") == 0 || strcmp(inp->name, "result_embd") == 0) {
            break;
        } else {
            inp = nullptr;
        }
    }
    GGML_ASSERT(inp != nullptr && "missing result_norm/result_embd tensor");

    struct ggml_tensor * cur;

    switch (pooling_type) {
        case LLAMA_POOLING_TYPE_MEAN:
            {
                struct ggml_tensor * inp_mean = build_inp_mean();
                cur = ggml_mul_mat(ctx0, ggml_cont(ctx0, ggml_transpose(ctx0, inp)), inp_mean);
            } break;
        case LLAMA_POOLING_TYPE_CLS:
        case LLAMA_POOLING_TYPE_LAST:
            {
                struct ggml_tensor * inp_cls = build_inp_cls();
                cur = ggml_get_rows(ctx0, inp, inp_cls);
            } break;
        case LLAMA_POOLING_TYPE_NONE:
            {
                cur = inp;
            } break;
        default:
            {
                GGML_ABORT("unknown pooling type");
            }
    }

    cb(cur, "result_embd_pooled", -1);

    ggml_build_forward_expand(gf, cur);

    return gf;
}

ggml_tensor * llm_build_context::llm_build_pos_bucket(bool causal) {
    if (causal) {
        lctx.inp_pos_bucket = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, n_kv,     n_tokens);
    } else {
        lctx.inp_pos_bucket = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, n_tokens, n_tokens);
    }

    ggml_set_input(lctx.inp_pos_bucket);
    cb(lctx.inp_pos_bucket, "pos_bucket", -1);

    return lctx.inp_pos_bucket;
}

ggml_tensor * llm_build_context::llm_build_pos_bias(struct ggml_tensor * pos_bucket, struct ggml_tensor * attn_rel_b) {
    struct ggml_tensor * pos_bucket_1d = ggml_view_1d(ctx0, pos_bucket, pos_bucket->ne[0] * pos_bucket->ne[1], 0);
    cb(pos_bucket_1d, "pos_bucket_1d", -1);

    struct ggml_tensor * pos_bias = ggml_get_rows(ctx0, attn_rel_b, pos_bucket_1d);
    cb(pos_bias, "pos_bias", -1);

    pos_bias = ggml_view_3d(ctx0, pos_bias, pos_bias->ne[0], lctx.inp_pos_bucket->ne[0], lctx.inp_pos_bucket->ne[1], ggml_element_size(pos_bias) * pos_bias->ne[0], ggml_element_size(pos_bias) * pos_bias->ne[0] * lctx.inp_pos_bucket->ne[0],  0);
    cb(pos_bias, "pos_bias", -1);

    pos_bias = ggml_permute(ctx0, pos_bias, 2, 0, 1, 3);
    cb(pos_bias, "pos_bias", -1);

    pos_bias = ggml_cont(ctx0, pos_bias);
    cb(pos_bias, "pos_bias", -1);

    return pos_bias;
}

ggml_tensor * llm_build_context::llm_build_inp_embd(
        struct ggml_context * ctx,
       struct llama_context & lctx,
        const llama_hparams & hparams,
          const llama_batch & batch,
         struct ggml_tensor * tok_embd,
         const llm_build_cb & cb) {
    const int64_t n_embd = hparams.n_embd;

    struct ggml_tensor * inpL;

    if (batch.token) {
        lctx.inp_tokens = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, batch.n_tokens);
        cb(lctx.inp_tokens, "inp_tokens", -1);
        ggml_set_input(lctx.inp_tokens);

        inpL = ggml_get_rows(ctx, tok_embd, lctx.inp_tokens);
    } else {
       lctx.inp_embd = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, batch.n_tokens);
        inpL = lctx.inp_embd;
        ggml_set_input(lctx.inp_embd);
    }

    // For Granite architecture
    if (hparams.f_embedding_scale != 0.0f) {
        inpL = ggml_scale(ctx, inpL, hparams.f_embedding_scale);
    }

    cb(inpL, "inp_embd", -1);

    return inpL;
}

void llm_build_context::llm_build_kv_store(
       struct llama_context & lctx,
        struct ggml_context * ctx,
        const llama_hparams & hparams,
        const llama_cparams & cparams,
       const llama_kv_cache & kv,
         struct ggml_cgraph * graph,
         struct ggml_tensor * k_cur,
         struct ggml_tensor * v_cur,
                    int32_t   n_tokens,
                    int32_t   kv_head,
         const llm_build_cb & cb,
                    int64_t   il) {
    const int64_t n_ctx = cparams.n_ctx;

    //const int64_t n_embd_k_gqa = hparams.n_embd_k_gqa(il);
    const int64_t n_embd_v_gqa = hparams.n_embd_v_gqa(il);

    const int64_t n_head_kv     = hparams.n_head_kv(il);
    const int64_t n_embd_head_k = hparams.n_embd_head_k;

    GGML_ASSERT(kv.size == n_ctx);

    //struct ggml_tensor * k_cache_view = ggml_view_1d(ctx, kv.k_l[il], n_tokens*n_embd_k_gqa,
    //        (ggml_row_size(kv.k_l[il]->type, n_embd_k_gqa))*kv_head);
    //cb(k_cache_view, "k_cache_view", il);

    GGML_ASSERT(2*il+1 < (int)lctx.cache_copies.size());
    auto k_row_size = ggml_row_size(kv.k_l[il]->type, n_embd_head_k);
    ggml_tensor * k_cache_view = ggml_view_2d(ctx, kv.k_l[il], n_embd_head_k, n_tokens*n_head_kv,
            k_row_size, k_row_size*n_head_kv*kv_head);

    lctx.cache_copies[2*il+0].cpy  = ggml_cpy(ctx, k_cur, k_cache_view);
    lctx.cache_copies[2*il+0].step = k_row_size*n_head_kv;

    // note: storing RoPE-ed version of K in the KV cache
    ggml_build_forward_expand(graph, lctx.cache_copies[2*il+0].cpy);

    struct ggml_tensor * v_cache_view = nullptr;

    if (cparams.flash_attn) {
        v_cache_view = ggml_view_1d(ctx, kv.v_l[il], n_tokens*n_embd_v_gqa,
                (kv_head)*ggml_row_size(kv.v_l[il]->type, n_embd_v_gqa));
        lctx.cache_copies[2*il+1].step = ggml_row_size(kv.v_l[il]->type, n_embd_v_gqa);
    } else {
        // note: the V cache is transposed when not using flash attention
        v_cache_view = ggml_view_2d(ctx, kv.v_l[il], n_tokens, n_embd_v_gqa,
                (  n_ctx)*ggml_element_size(kv.v_l[il]),
                (kv_head)*ggml_element_size(kv.v_l[il]));
        lctx.cache_copies[2*il+1].step = ggml_element_size(kv.v_l[il]);

        v_cur = ggml_transpose(ctx, v_cur);
    }
    cb(v_cache_view, "v_cache_view", il);

    lctx.cache_copies[2*il+1].cpy  = ggml_cpy(ctx, v_cur, v_cache_view);
    ggml_build_forward_expand(graph, lctx.cache_copies[2*il+1].cpy);
}

ggml_tensor * llm_build_context::llm_build_lora_mm(
        struct llama_context & lctx,
         struct ggml_context * ctx0,
          struct ggml_tensor * w,
          struct ggml_tensor * cur) {
    struct ggml_tensor * res = ggml_mul_mat(ctx0, w, cur);
    for (auto & it : lctx.lora_adapters) {
        struct llama_lora_weight * lora = it.first->get_weight(w);
        if (lora == nullptr) {
            continue;
        }
        const float alpha = it.first->alpha;
        const float rank  = (float) lora->b->ne[0];
        const float scale = alpha ? it.second * alpha / rank : it.second;
        struct ggml_tensor * ab_cur = ggml_mul_mat(
            ctx0, lora->b,
            ggml_mul_mat(ctx0, lora->a, cur)
        );
        ab_cur = ggml_scale(ctx0, ab_cur, scale);
        res = ggml_add(ctx0, res, ab_cur);
    }
    return res;
}

ggml_tensor * llm_build_context::llm_build_lora_mm_id(
        struct llama_context & lctx,
         struct ggml_context * ctx0,
          struct ggml_tensor * w,   // struct ggml_tensor * as
          struct ggml_tensor * cur, // struct ggml_tensor * b
          struct ggml_tensor * ids) {
    struct ggml_tensor * res = ggml_mul_mat_id(ctx0, w, cur, ids);
    for (auto & it : lctx.lora_adapters) {
        struct llama_lora_weight * lora = it.first->get_weight(w);
        if (lora == nullptr) {
            continue;
        }
        const float alpha = it.first->alpha;
        const float rank  = (float) lora->b->ne[0];
        const float scale = alpha ? it.second * alpha / rank : it.second;
        struct ggml_tensor * ab_cur = ggml_mul_mat_id(
            ctx0, lora->b,
            ggml_mul_mat_id(ctx0, lora->a, cur, ids),
            ids
        );
        ab_cur = ggml_scale(ctx0, ab_cur, scale);
        res = ggml_add(ctx0, res, ab_cur);
    }
    return res;
}

ggml_tensor * llm_build_context::llm_build_norm(
        ggml_context * ctx,
         ggml_tensor * cur,
        const llama_hparams & hparams,
         ggml_tensor * mw,
         ggml_tensor * mb,
              llm_norm_type   type,
         const llm_build_cb & cb, int il, float scale_eps) {

    if (type == LLM_NORM_RMS && mw) {
        cur = ggml_fused_rms_norm(ctx, cur, mw, scale_eps * hparams.f_norm_rms_eps);
        if (mb) {
            cb(cur, "fused_norm", il);
            cur = ggml_add(ctx, cur, mb);
        }
        return cur;
    }

    switch (type) {
        case LLM_NORM:     cur = ggml_norm    (ctx, cur, hparams.f_norm_eps);     break;
        case LLM_NORM_RMS: cur = ggml_rms_norm(ctx, cur, scale_eps * hparams.f_norm_rms_eps); break;
    }

    if (mw || mb) {
        cb(cur, "norm", il);
    }

    if (mw) {
        cur = ggml_mul(ctx, cur, mw);
        if (mb) {
            cb(cur, "norm_w", il);
        }
    }

    if (mb) {
        cur = ggml_add(ctx, cur, mb);
    }

    return cur;
}

static ggml_tensor * get_input_tensor_sm_graph(ggml_tensor * input, int id) {
    auto cur = input;
    if (input->op == GGML_OP_REDUCE) {
        auto view_src = input->view_src;
        GGML_ASSERT(view_src);
        cur = input->src[id];
        if (cur == view_src || !cur) {
            //printf("%s: Setting input to %s for id = %d\n", __func__, view_src->name, id);
            cur = input;
        }
    }
    return cur;
}

ggml_tensor * llm_build_context::llm_build_ffn(
        ggml_context * ctx,
       llama_context & lctx,
         ggml_tensor * ffn_norm,
         ggml_tensor * input,
         ggml_tensor * up,
         ggml_tensor * up_b,
         ggml_tensor * up_s,
         ggml_tensor * gate,
         ggml_tensor * gate_b,
         ggml_tensor * gate_s,
         ggml_tensor * down,
         ggml_tensor * down_b,
         ggml_tensor * down_s,
         ggml_tensor * act_scales,
            llm_ffn_op_type   type_op,
          llm_ffn_gate_type   type_gate,
         const llm_build_cb & cb, int il, ggml_cgraph * graph, bool add_input,
         bool is_norm, ggml_tensor * add_extra) {

    if (!up_b && !up_s && !gate_b && !gate_s && !down_b && !down_s &&
        up->extra && gate->extra && down->extra && type_gate == LLM_FFN_PAR &&
        (type_op == LLM_FFN_SILU || type_op == LLM_FFN_RELU || (type_op == LLM_FFN_GELU && !act_scales))) {
        //printf("%s: %s\n", __func__, ggml_op_name(input->op));
        auto unary_op = type_op == LLM_FFN_SILU ? GGML_UNARY_OP_SILU :
                        type_op == LLM_FFN_RELU ? GGML_UNARY_OP_RELU : GGML_UNARY_OP_GELU;
        auto u = (ggml_split_tensor_t *)up->extra;
        auto g = (ggml_split_tensor_t *)gate->extra;
        auto d = (ggml_split_tensor_t *)down->extra;
        GGML_ASSERT(u->n_device == g->n_device && u->n_device == d->n_device);
        std::vector<ggml_tensor *> ffn(u->n_device, nullptr);
        int id_last = -1;
        for (int id = 0; id < u->n_device; ++id) {
            int il_cb = 1000*(id+1) + il;
            auto split_u = u->splits[id];
            auto split_g = g->splits[id];
            auto split_d = d->splits[id];
            GGML_ASSERT((!split_u && !split_g && !split_d) || (split_u && split_g && split_d));
            if (!split_u) continue;
            auto cur = get_input_tensor_sm_graph(input, id);
            if (ffn_norm && ffn_norm->extra) {
                auto norm = (ggml_split_tensor_t *)ffn_norm->extra;
                GGML_ASSERT(norm->splits[id]);
                if (is_norm) {
                    cur = ggml_fused_norm(ctx, cur, norm->splits[id], lctx.model.hparams.f_norm_eps);
                } else {
                    cur = llm_build_norm(ctx, cur, lctx.model.hparams, norm->splits[id], NULL, LLM_NORM_RMS, cb, il);
                }
                cb(cur, "ffn_inp_normed", il_cb);
            }
            else if (cur->type != GGML_TYPE_F32) {
                cur = ggml_cast(ctx, cur, GGML_TYPE_F32);
            }
            cur = ggml_fused_up_gate(ctx, split_u, split_g, cur, unary_op);
            cb(cur, "ffn_up_gate", il_cb);
            cur = llm_build_lora_mm(lctx, ctx, split_d, cur);
            cb(cur, "ffn_down", il_cb);
            if (lctx.model.arch == LLM_ARCH_GLM4 || lctx.model.arch == LLM_ARCH_GLM4_MOE) {
                // GLM4 and GLM4_MOE seem to have numerical issues with half-precision accumulators
                ggml_mul_mat_set_prec(cur, GGML_PREC_F32);
            }
            if (cur->ne[1] > 32 && lctx.cparams.split_mode_f16) {
                cur = ggml_cast(ctx, cur, GGML_TYPE_F16);
            }
            if (add_extra && add_extra->op == GGML_OP_REDUCE && add_extra->op_params[3] == 1) {
                // When the reduce op is turned off via op_params[3] == 1, we need to add each src
                // rtaher than add the reduced add_extra result to the ffn reduced ffn result.
                GGML_ASSERT(add_extra->src[id]); // TODO: fix this! It can be null if the splits of the attention and ffn tensors are different
                cur = ggml_add(ctx, cur, add_extra->src[id]);
                cb(cur, "ffn_with_extra", il_cb);
            }
            if (graph) {
                ggml_build_forward_expand(graph, cur);
            }
            ffn[id] = cur;
            id_last = id;
        }
        GGML_ASSERT(id_last >= 0);
        if (add_input) {
            ffn[id_last] = ggml_add(ctx, ffn[id_last], input);
            cb(ffn[id_last], "ffn_with_inp", il);
        }
        if (add_extra && !(add_extra->op == GGML_OP_REDUCE && add_extra->op_params[3] == 1)) {
            ffn[id_last] = ggml_add(ctx, ffn[id_last], add_extra);
            cb(ffn[id_last], "ffn_with_inp", il);
        }
        auto cur = ggml_reduce(ctx, ffn.data(), u->n_device, GGML_OP_ADD);
        cb(cur, "ffn_combined", il);
        ggml_build_forward_expand(graph, cur);
        return cur;
    }

    auto cur = input;
    if (ffn_norm) {
        cur = llm_build_norm(ctx, cur, lctx.model.hparams, ffn_norm, NULL, is_norm ? LLM_NORM : LLM_NORM_RMS, cb, il);
        cb(input, "ffn_norm", il);
    }
    if (cur->type != GGML_TYPE_F32) {
        cur = ggml_cast(ctx, cur, GGML_TYPE_F32);
    }

    if (lctx.cparams.fused_up_gate &&
        up && gate && !up_b && !up_s && !gate_b && !gate_s && type_gate == LLM_FFN_PAR &&
        (type_op == LLM_FFN_SILU || type_op == LLM_FFN_RELU || (type_op == LLM_FFN_GELU && !act_scales))) {
        auto unary_op = type_op == LLM_FFN_SILU ? GGML_UNARY_OP_SILU :
                        type_op == LLM_FFN_RELU ? GGML_UNARY_OP_RELU : GGML_UNARY_OP_GELU;
        cur = ggml_fused_up_gate(ctx, up, gate, cur, unary_op);
        cb(cur, "ffn_up_gate", il);
        if (down) {
            cur = llm_build_lora_mm(lctx, ctx, down, cur);
            if (lctx.model.arch == LLM_ARCH_GLM4 || lctx.model.arch == LLM_ARCH_GLM4_MOE) {
                // GLM4 and GLM4_MOE seem to have numerical issues with half-precision accumulators
                ggml_mul_mat_set_prec(cur, GGML_PREC_F32);
            }
        }
        if (down_b) {
            cb(cur, "ffn_down", il);
        }
        if (down_b) {
            cur = ggml_add(ctx, cur, down_b);
        }
        if (down_s) {
            cur = ggml_mul(ctx, cur, down_s);
            cb(cur, "ffn_down_s", il);
        }
        if (add_input) {
            cur = ggml_add(ctx, cur, input);
            cb(cur, "ffn_out_with_inp", il);
        }
        if (add_extra) {
            cur = ggml_add(ctx, cur, add_extra);
            cb(cur, "ffn_out_with_inp", il);
        }
        return cur;
    }

    struct ggml_tensor * tmp = up ? llm_build_lora_mm(lctx, ctx, up, cur) : cur;
    cb(tmp, "ffn_up", il);

    if (up_b) {
        tmp = ggml_add(ctx, tmp, up_b);
        cb(tmp, "ffn_up_b", il);
    }

    if (up_s) {
        tmp = ggml_mul(ctx, tmp, up_s);
        cb(tmp, "ffn_up_s", il);
    }

    if (gate) {
        switch (type_gate) {
            case LLM_FFN_SEQ:
                {
                    cur = llm_build_lora_mm(lctx, ctx, gate, tmp);
                    cb(cur, "ffn_gate", il);
                } break;
            case LLM_FFN_PAR:
                {
                    cur = llm_build_lora_mm(lctx, ctx, gate, cur);
                    cb(cur, "ffn_gate", il);
                } break;
        }

        if (gate_b) {
            cur = ggml_add(ctx, cur, gate_b);
            cb(cur, "ffn_gate_b", il);
        }

        if (gate_s) {
            cur = ggml_mul(ctx, cur, gate_s);
            cb(cur, "ffn_gate_s", il);
        }

    } else {
        cur = tmp;
    }

    if (type_gate == LLM_FFN_PAR &&
       (type_op == LLM_FFN_SILU || type_op == LLM_FFN_RELU || (type_op == LLM_FFN_GELU && !act_scales))) {
        cur = ggml_fused_mul_unary(ctx, cur, tmp, type_op == LLM_FFN_SILU ? GGML_UNARY_OP_SILU :
                                                  type_op == LLM_FFN_RELU ? GGML_UNARY_OP_RELU : GGML_UNARY_OP_GELU);
    }
    else {

    switch (type_op) {
        case LLM_FFN_SILU:
            {
                cur = ggml_silu(ctx, cur);
                cb(cur, "ffn_silu", il);
            } break;
        case LLM_FFN_GELU:
            {
                cur = ggml_gelu(ctx, cur);
                cb(cur, "ffn_gelu", il);
                if (act_scales != NULL) {
                    cur = ggml_div(ctx, cur, act_scales);
                    cb(cur, "ffn_act", il);
                }
            } break;
        case LLM_FFN_RELU:
            {
                cur = ggml_relu(ctx, cur);
                cb(cur, "ffn_relu", il);
            } break;
        case LLM_FFN_RELU_SQR:
            {
                cur = ggml_relu(ctx, cur);
                cb(cur, "ffn_relu", il);

                cur = ggml_sqr(ctx, cur);
                cb(cur, "ffn_sqr(relu)", il);
            } break;
        case LLM_FFN_SWIGLU:
            {
                cur = ggml_swiglu(ctx, cur);
                cb(cur, "ffn_swiglu", il);
            } break;
        default:
            GGML_ABORT("fatal error");
    }

    if (type_gate == LLM_FFN_PAR) {
        cur = ggml_mul(ctx, cur, tmp);
        cb(cur, "ffn_gate_par", il);
    }
    }

    if (down) {
        cur = llm_build_lora_mm(lctx, ctx, down, cur);
        if (lctx.model.arch == LLM_ARCH_GLM4 || lctx.model.arch == LLM_ARCH_GLM4_MOE) {
            // GLM4 and GLM4_MOE seem to have numerical issues with half-precision accumulators
            ggml_mul_mat_set_prec(cur, GGML_PREC_F32);
        }
    }

    if (down_b) {
        cb(cur, "ffn_down", il);
    }

    if (down_b) {
        cur = ggml_add(ctx, cur, down_b);
    }

    if (down_s) {
        cur = ggml_mul(ctx, cur, down_s);
        cb(cur, "ffn_down_s", il);
    }

    if (add_input) {
        cur = ggml_add(ctx, cur, input);
        cb(cur, "ffn_out_with_inp", il);
    }
    if (add_extra) {
        cur = ggml_add(ctx, cur, add_extra);
        cb(cur, "ffn_out_with_inp", il);
    }

    return cur;
}

ggml_tensor * llm_build_context::llm_build_moe_ffn(
        ggml_context * ctx,
       llama_context & lctx,
         ggml_tensor * cur,
         ggml_tensor * gate_inp,   ggml_tensor * gate_inp_b,
         ggml_tensor * up_exps,    ggml_tensor * up_exps_b,
         ggml_tensor * gate_exps,  ggml_tensor * gate_exps_b,
         ggml_tensor * down_exps,  ggml_tensor * down_exps_b,
         ggml_tensor * exp_probs_b,
                    int64_t   n_expert,
                    int64_t   n_expert_used,
            llm_ffn_op_type   type_op,
                       bool   norm_w,
                       bool   scale_w,
                      float   w_scale,
llm_expert_gating_func_type   gating_op,
         const llm_build_cb & cb, int il, ggml_cgraph * graph, bool add_input,
         ggml_tensor * up_gate_exps, ggml_tensor * up_gate_exps_b) {

    auto input = cur;

    int64_t n_embd = cur->ne[0];
    int64_t n_tokens = cur->ne[1];
    bool weight_before_ffn = lctx.model.arch == LLM_ARCH_LLAMA4; // for llama4, we apply the sigmoid-ed weights before the FFN

    ggml_tensor * logits = llm_build_lora_mm(lctx, ctx, gate_inp, cur); // [n_expert, n_tokens]
    cb(logits, "ffn_moe_logits", il);

    if (gate_inp_b) {
        logits = ggml_add(ctx, logits, gate_inp_b);
        cb(logits, "ffn_moe_logits_biased", il);
    }


    //ggml_tensor * probs = ggml_soft_max(ctx, logits); // [n_expert, n_tokens]
    ggml_tensor * probs = nullptr;
    switch (gating_op) {
        case LLM_EXPERT_GATING_FUNC_SOFTMAX:
            {
                probs = ggml_soft_max(ctx, logits); // [n_expert, n_tokens]
            } break;
        case LLM_EXPERT_GATING_FUNC_SIGMOID:
            {
                probs = ggml_sigmoid(ctx, logits); // [n_expert, n_tokens]
            } break;
        case LLM_EXPERT_GATING_FUNC_TYPE_SOFTMAX_WEIGHT:
            {
                probs = logits; // [n_expert, n_tokens]
            } break;
        default:
            GGML_ABORT("fatal error");
    }
    cb(probs, "ffn_moe_probs", il);

    // add experts selection bias - introduced in DeepSeek V3
    // leave probs unbiased as it's later used to get expert weights
    ggml_tensor * selection_probs = probs;
    if (exp_probs_b != nullptr) {
        selection_probs = ggml_add(ctx, probs, exp_probs_b);
        cb(selection_probs, "ffn_moe_probs_biased", il);
    }

    // llama4 doesn't have exp_probs_b, and sigmoid is only used after top_k
    // see: https://github.com/meta-llama/llama-models/blob/699a02993512fb36936b1b0741e13c06790bcf98/models/llama4/moe.py#L183-L198
    if (lctx.model.arch == LLM_ARCH_LLAMA4) {
        selection_probs = logits;
    }

    // select experts
    ggml_tensor * selected_experts;
    if (lctx.cparams.grouped_expert_routing && lctx.model.arch == LLM_ARCH_BAILINGMOE2 && n_tokens > 0) {
        auto& hparams = lctx.model.hparams;
        selected_experts = ggml_grouped_topk(ctx, selection_probs, hparams.n_expert_groups, hparams.n_group_used, 2, n_expert_used);
    } else {
        //selected_experts = ggml_top_k_thresh(ctx, selection_probs, n_expert_used,
        //        lctx.cparams.min_experts, lctx.cparams.thresh_experts); // [n_expert_used, n_tokens]
        selected_experts = ggml_top_k(ctx, selection_probs, n_expert_used); // [n_expert_used, n_tokens]
    }
    cb(selected_experts, "ffn_moe_topk", il);
    ggml_tensor * weights = ggml_get_rows(ctx,
            ggml_reshape_3d(ctx, probs, 1, n_expert, n_tokens), selected_experts); // [1, n_expert_used, n_tokens]
    cb(weights, "ffn_moe_weights", il);

    if (gating_op == LLM_EXPERT_GATING_FUNC_TYPE_SOFTMAX_WEIGHT) {
        weights = ggml_reshape_2d(ctx, weights, n_expert_used, n_tokens);
        weights = ggml_soft_max(ctx, weights); // [n_expert_used, n_tokens]
        weights = ggml_reshape_3d(ctx, weights, 1, n_expert_used, n_tokens);
        cb(weights, "ffn_moe_weights_softmax", il);
    }

    if (norm_w) {
        weights = ggml_reshape_2d(ctx, weights, n_expert_used, n_tokens);

        ggml_tensor * weights_sum = ggml_sum_rows(ctx, weights); // [1, n_tokens]
        cb(weights_sum, "ffn_moe_weights_sum", il);

        if (lctx.model.arch == LLM_ARCH_BAILINGMOE2) {
            weights_sum = ggml_scale_bias(ctx, weights_sum, 1.0, 1e-20);
            cb(weights_sum, "ffn_moe_weights_sum_biased", il);
        }

        weights = ggml_div(ctx, weights, weights_sum); // [n_expert_used, n_tokens]
        cb(weights, "ffn_moe_weights_norm", il);

        weights = ggml_reshape_3d(ctx, weights, 1, n_expert_used, n_tokens);
    }
    if (scale_w && std::abs(w_scale-1) > 1e-5f) {
        weights = ggml_scale(ctx, weights, w_scale);
        cb(weights, "ffn_moe_weights_scaled", il);
    }

    if (graph) {
        ggml_build_forward_expand(graph, weights);
    }

    cur = ggml_reshape_3d(ctx, cur, n_embd, 1, n_tokens);

    if (weight_before_ffn) {
        // TODO: this is a workaround as we don't yet have a repeat op that takes custom dim (ggml_repeat_4d)
        ggml_tensor * repeated = ggml_new_tensor_3d(ctx, cur->type, n_embd, n_expert_used, n_tokens);
        repeated = ggml_repeat(ctx, cur, repeated); // [n_embd, n_expert_used, n_tokens]
        cur = ggml_mul(ctx, repeated, weights);
        cb(cur, "ffn_moe_weighted", il);
    }

    // For now we don't modify the fused up/gate op to include biases.
    // Hence, if we have biases, we cannot use fmoe.
    //
    //bool can_use_fmoe = !up_exps_b && !gate_exps_b && (type_op == LLM_FFN_SILU || type_op == LLM_FFN_GELU);
    bool can_use_fmoe = type_op == LLM_FFN_SILU || type_op == LLM_FFN_GELU || type_op == LLM_FFN_SWIGLU_OAI_MOE;

    ggml_tensor * par;
    if (can_use_fmoe && up_gate_exps) {
        if (up_gate_exps_b) {
            par = ggml_moe_up_gate_ext(ctx, up_gate_exps, nullptr, cur, selected_experts, up_gate_exps_b, nullptr,
                    type_op == LLM_FFN_SILU ? GGML_UNARY_OP_SILU :
                    type_op == LLM_FFN_GELU ? GGML_UNARY_OP_GELU : GGML_UNARY_OP_SWIGLU_OAI);
        } else {
            GGML_ASSERT(type_op != LLM_FFN_SWIGLU_OAI_MOE);
            par = ggml_moe_up_gate(ctx, up_gate_exps, nullptr, cur, selected_experts,
                    type_op == LLM_FFN_SILU ? GGML_UNARY_OP_SILU : GGML_UNARY_OP_GELU);
        }
    } else {
    GGML_ASSERT(!up_gate_exps && !up_gate_exps_b);

    if (can_use_fmoe && lctx.cparams.fused_moe_up_gate && up_exps->type == gate_exps->type) {
        if (up_exps_b || gate_exps_b) {
            par = ggml_moe_up_gate_ext(ctx, up_exps, gate_exps, cur, selected_experts, up_exps_b, gate_exps_b,
                    type_op == LLM_FFN_SILU ? GGML_UNARY_OP_SILU :
                    type_op == LLM_FFN_GELU ? GGML_UNARY_OP_GELU : GGML_UNARY_OP_SWIGLU_OAI);
        } else {
            GGML_ASSERT(type_op != LLM_FFN_SWIGLU_OAI_MOE);
            par = ggml_moe_up_gate(ctx, up_exps, gate_exps, cur, selected_experts,
                    type_op == LLM_FFN_SILU ? GGML_UNARY_OP_SILU : GGML_UNARY_OP_GELU);
        }
    } else {
        ggml_tensor * up = llm_build_lora_mm_id(lctx, ctx, up_exps, cur, selected_experts); // [n_ff, n_expert_used, n_tokens]
        cb(up, "ffn_moe_up", il);

        ggml_tensor * gate = llm_build_lora_mm_id(lctx, ctx, gate_exps, cur, selected_experts); // [n_ff, n_expert_used, n_tokens]
        cb(gate, "ffn_moe_gate", il);

        if (graph) {
            // So we can potentially fuse the up and gate mul_mat_id
            ggml_build_forward_expand(graph, up);
            ggml_build_forward_expand(graph, gate);
        }

        if (up_exps_b) {
            up = ggml_add_id(ctx, up, up_exps_b, selected_experts);
            cb(up, "ffn_moe_up_biased", il);
        }

        if (gate_exps_b) {
            gate = ggml_add_id(ctx, gate, gate_exps_b, selected_experts);
            cb(gate, "ffn_moe_gate_biased", il);
        }

        if (type_op == LLM_FFN_SILU || type_op == LLM_FFN_GELU) {
            par = ggml_fused_mul_unary(ctx, gate, up, type_op == LLM_FFN_SILU ? GGML_UNARY_OP_SILU : GGML_UNARY_OP_GELU);
        } else if (type_op == LLM_FFN_SWIGLU_OAI_MOE) {
            constexpr float alpha = 1.702f;
            constexpr float limit = 7.0f;
            par = ggml_swiglu_oai(ctx, gate, up, alpha, limit);
        }
        else {
            GGML_ABORT("fatal error");
        }

    }
    }
    cb(par, "ffn_moe_gate_par", il);

    ggml_tensor * experts = llm_build_lora_mm_id(lctx, ctx, down_exps, par, selected_experts); // [n_embd, n_expert_used, n_tokens]
    cb(experts, "ffn_moe_down", il);

    if (down_exps_b) {
        experts = ggml_add_id(ctx, experts, down_exps_b, selected_experts);
        cb(experts, "ffn_moe_down_biased", il);
    }

    if (!weight_before_ffn) {
        if (lctx.cparams.fused_mmad) {
            experts = ggml_mul_multi_add(ctx, experts, weights);
            cb(experts, "ffn_moe_weighted", il);
            if (add_input) {
                experts = ggml_add(ctx, experts, input);
                cb(experts, "ffn_out_with_inp", il);
            }
            return experts;
        }
        experts = ggml_mul(ctx, experts, weights);
        cb(experts, "ffn_moe_weighted", il);
    }

    ggml_tensor * result;
    if (n_expert_used == 1) {
        result = ggml_cont(ctx, ggml_view_2d(ctx, experts, n_embd, n_tokens, experts->nb[2], 0));
    }
    if (n_expert_used == 2) {
        result = ggml_add(ctx, ggml_view_2d(ctx, experts, n_embd, n_tokens, experts->nb[2], 0),
                             ggml_view_2d(ctx, experts, n_embd, n_tokens, experts->nb[2], experts->nb[1]));
    }
    result = ggml_multi_add(ctx, ggml_view_2d(ctx, experts, n_embd, n_tokens, experts->nb[2], 0), n_expert_used);
    if (add_input) {
        cb(result, "ffn_out", il);
        result = ggml_add(ctx, result, input);
    }
    return result;

}

ggml_tensor * llm_build_context::llm_build_std_moe_ffn(ggml_context * ctx, llama_context & lctx,
         ggml_tensor * ffn_norm,
         ggml_tensor * input,
         ggml_tensor * gate_inp,   ggml_tensor * gate_inp_b,
         ggml_tensor * up_exps,    ggml_tensor * up_exps_b,
         ggml_tensor * gate_exps,  ggml_tensor * gate_exps_b,
         ggml_tensor * down_exps,  ggml_tensor * down_exps_b,
         ggml_tensor * exp_probs_b,
         ggml_tensor * up_shexp,   ggml_tensor * up_b_shexp,
         ggml_tensor * gate_shexp, ggml_tensor * gate_b_shexp,
         ggml_tensor * down_shexp, ggml_tensor * down_b_shexp,
                    int64_t   n_expert,
                    int64_t   n_expert_used,
            llm_ffn_op_type   type_op,
                       bool   norm_w,
                       bool   scale_w,
                      float   w_scale,
llm_expert_gating_func_type   gating_op,
            llm_ffn_op_type   type_op_shexp,
         const llm_build_cb & cb, int il, ggml_cgraph * graph, bool add_input,
         ggml_tensor * up_gate_exps, ggml_tensor * up_gate_exps_b) {

    auto split_up_exps    = (ggml_split_tensor_t *)up_exps->extra;
    auto split_gate_exps  = (ggml_split_tensor_t *)gate_exps->extra;
    auto split_down_exps  = (ggml_split_tensor_t *)down_exps->extra;
    auto split_up_shexp   = up_shexp   ? (ggml_split_tensor_t *)up_shexp->extra   : nullptr;
    auto split_gate_shexp = gate_shexp ? (ggml_split_tensor_t *)gate_shexp->extra : nullptr;
    auto split_down_shexp = down_shexp ? (ggml_split_tensor_t *)down_shexp->extra : nullptr;
    auto split_up_b_shexp   = up_b_shexp   ? (ggml_split_tensor_t *)up_b_shexp   : nullptr;
    auto split_gate_b_shexp = gate_b_shexp ? (ggml_split_tensor_t *)gate_b_shexp : nullptr;
    auto split_down_b_shexp = down_b_shexp ? (ggml_split_tensor_t *)down_b_shexp : nullptr;
    if (!split_up_exps && !split_gate_exps && !split_down_exps) {
        auto cur = input;
        if (ffn_norm) {
            auto the_ffn_norm = ffn_norm->extra ? ((ggml_split_tensor_t *)ffn_norm->extra)->splits[lctx.model.main_gpu] : ffn_norm;
            GGML_ASSERT(the_ffn_norm);
            cur = llm_build_norm(ctx, cur, lctx.model.hparams, the_ffn_norm, nullptr, LLM_NORM_RMS, cb, il);
            cb(cur, "ffn_inp_normed", il);
        }
        if (cur->type != GGML_TYPE_F32) {
            cur = ggml_cast(ctx, cur, GGML_TYPE_F32);
        }
        auto the_gate_inp = gate_inp->extra ? ((ggml_split_tensor_t *)gate_inp->extra)->splits[lctx.model.main_gpu] : gate_inp;
        auto the_gate_inp_b = gate_inp_b ? gate_inp_b->extra ? ((ggml_split_tensor_t *)gate_inp_b->extra)->splits[lctx.model.main_gpu] : gate_inp_b : nullptr;
        auto the_exp_probs_b = exp_probs_b ? exp_probs_b->extra ? ((ggml_split_tensor_t *)exp_probs_b->extra)->splits[lctx.model.main_gpu] : exp_probs_b : nullptr;
        //int n_before = graph->n_nodes;
        auto routed_out = llm_build_moe_ffn(ctx, lctx, cur,
                    the_gate_inp, the_gate_inp_b,
                    up_exps,   up_exps_b,
                    gate_exps, gate_exps_b,
                    down_exps, down_exps_b,
                    the_exp_probs_b,
                    n_expert, n_expert_used,
                    type_op, norm_w, scale_w, w_scale,
                    gating_op, cb, il, graph, false, up_gate_exps, up_gate_exps_b);
        cb(routed_out, "routed_out", il);
        if (add_input) {
            routed_out = ggml_add(ctx, routed_out, input);
            cb(routed_out, "routed_out_with_inp", il);
        }
        ggml_build_forward_expand(graph, routed_out);

        if (up_shexp && gate_shexp && down_shexp) {
            if (split_up_shexp) {
                std::vector<ggml_tensor *> results; results.reserve(split_up_shexp->n_device);
                GGML_ASSERT(!split_up_b_shexp   || split_up_b_shexp->n_device   == split_up_shexp->n_device);
                GGML_ASSERT(!split_gate_b_shexp || split_gate_b_shexp->n_device == split_up_shexp->n_device);
                GGML_ASSERT(!split_down_b_shexp || split_down_b_shexp->n_device == split_up_shexp->n_device);
                for (int id = 0; id < split_up_shexp->n_device; ++id) {
                    int il_cb = 1000*id + il;
                    GGML_ASSERT((split_up_shexp->splits[id] && split_gate_shexp->splits[id] && split_down_shexp->splits[id]) ||
                                (!split_up_shexp->splits[id] && !split_gate_shexp->splits[id] && !split_down_shexp->splits[id]));
                    if (!split_up_shexp->splits[id]) continue;
                    auto the_ffn_norm = ffn_norm ? ffn_norm->extra ? ((ggml_split_tensor_t *)ffn_norm->extra)->splits[id] : ffn_norm : nullptr;
                    auto shared_out = llm_build_ffn(ctx, lctx, the_ffn_norm, input,
                            split_up_shexp->splits[id],   split_up_b_shexp   ? split_up_b_shexp->splits[id]   : nullptr, nullptr,
                            split_gate_shexp->splits[id], split_gate_b_shexp ? split_gate_b_shexp->splits[id] : nullptr, nullptr,
                            split_down_shexp->splits[id], split_down_b_shexp ? split_down_b_shexp->splits[id] : nullptr, nullptr,
                            nullptr, type_op_shexp, LLM_FFN_PAR, cb, il);
                    cb(shared_out, "ffn_shexp_out", il_cb);
                    if (shared_out->ne[1] > 32 && lctx.cparams.split_mode_f16) {
                        shared_out = ggml_cast(ctx, shared_out, GGML_TYPE_F16);
                    }
                    results.push_back(shared_out);
                }
                GGML_ASSERT(!results.empty());
                if (results.size() == 1) {
                    cur = results.front();
                } else {
                    cur = ggml_add(ctx, results[0], results[1]);
                    cur->op_params[0] = 0xff;
                    cb(cur, "ffn_shared_combined", il);
                    for (int id = 2; id < int(results.size()); ++id) {
                        cur = ggml_add(ctx, cur, results[id]);
                        cb(cur, "ffn_shared_combined", il);
                    }
                }
                if (routed_out->ne[1] > 32 && lctx.cparams.split_mode_f16) {
                    auto routed_out_f16 = ggml_cast(ctx, routed_out, GGML_TYPE_F16);
                    cur = ggml_add(ctx, routed_out_f16, cur);
                } else {
                    cur = ggml_add(ctx, routed_out, cur);
                }
                cb(cur, "ffn_out", il);
            } else {
                auto shared_out = llm_build_ffn(ctx, lctx, nullptr, cur,
                        up_shexp,   up_b_shexp,   nullptr,
                        gate_shexp, gate_b_shexp, nullptr,
                        down_shexp, down_b_shexp, nullptr,
                        nullptr, type_op_shexp, LLM_FFN_PAR, cb, il);
                cb(shared_out, "ffn_shexp_out", il);
                cur = ggml_add(ctx, routed_out, shared_out);
                cb(cur, "ffn_out", il);
            }
        } else {
            cur = routed_out;
        }
        if (cur != routed_out) {
            ggml_build_forward_expand(graph, cur);
        }
        return cur;
    }
    GGML_ASSERT(split_up_exps && split_gate_exps && split_down_exps);
    GGML_ASSERT(split_up_exps->n_device == split_gate_exps->n_device && split_up_exps->n_device == split_down_exps->n_device);
    std::vector<ggml_tensor *> results(split_up_exps->n_device, nullptr);
    GGML_ASSERT((!split_up_shexp && !split_gate_shexp && !split_down_shexp) ||
                ( split_up_shexp &&  split_gate_shexp &&  split_down_shexp));
    auto split_gate_inp = (ggml_split_tensor_t *)gate_inp->extra;
    GGML_ASSERT(split_gate_inp && split_gate_inp->n_device == split_up_exps->n_device);
    auto split_exp_probs_b = exp_probs_b ? (ggml_split_tensor_t *)exp_probs_b->extra : nullptr;
    GGML_ASSERT(!split_exp_probs_b || split_exp_probs_b->n_device == split_up_exps->n_device);

    auto split_gate_inp_b  = gate_inp_b  ? (ggml_split_tensor_t *)gate_inp_b->extra  : nullptr;
    auto split_exps_down_b = down_exps_b ? (ggml_split_tensor_t *)down_exps_b->extra : nullptr;
    auto split_exps_gate_b = gate_exps_b ? (ggml_split_tensor_t *)gate_exps_b->extra : nullptr;
    auto split_exps_up_b   = up_exps_b   ? (ggml_split_tensor_t *)up_exps_b->extra   : nullptr;
    int last_id = -1;
    bool down_bias_added = false;
    for (int id = 0; id < split_up_exps->n_device; ++id) {
        GGML_ASSERT((split_up_exps->splits[id] && split_gate_exps->splits[id] && split_down_exps->splits[id]) ||
                    (!split_up_exps->splits[id] && !split_gate_exps->splits[id] && !split_down_exps->splits[id]));
        if (!split_up_exps->splits[id]) continue;
        int il_cb = 1000*(id + 1) + il;
        auto cur = get_input_tensor_sm_graph(input, id);
        if (ffn_norm) {
            auto split_ffn_norm = (ggml_split_tensor_t *)ffn_norm->extra;
            GGML_ASSERT(split_ffn_norm && split_ffn_norm->n_device == split_up_exps->n_device);
            cur = llm_build_norm(ctx, cur, lctx.model.hparams, split_ffn_norm->splits[id], nullptr, LLM_NORM_RMS, cb, il);
            cb(cur, "ffn_inp_normed", il_cb);
        }
        if (cur->type != GGML_TYPE_F32) {
            cur = ggml_cast(ctx, cur, GGML_TYPE_F32);
        }
        GGML_ASSERT(!split_gate_inp_b  || split_gate_inp_b->splits[id]);
        GGML_ASSERT(!split_exps_down_b || split_exps_down_b->splits[id]);
        GGML_ASSERT(!split_exps_gate_b || split_exps_gate_b->splits[id]);
        GGML_ASSERT(!split_exps_up_b   || split_exps_up_b->splits[id]);
        auto routed_out = llm_build_moe_ffn(ctx, lctx, cur,
                    split_gate_inp->splits[id],  split_gate_inp_b ? split_gate_inp_b->splits[id] : nullptr,
                    split_up_exps->splits[id],   split_exps_up_b  ? split_exps_up_b->splits[id]  : nullptr,
                    split_gate_exps->splits[id], split_exps_gate_b ? split_exps_gate_b->splits[id] : nullptr,
                    split_down_exps->splits[id], !down_bias_added && split_exps_down_b ? split_exps_down_b->splits[id] : nullptr,
                    split_exp_probs_b ? split_exp_probs_b->splits[id] : nullptr,
                    n_expert, n_expert_used,
                    type_op, norm_w, scale_w, w_scale,
                    gating_op, cb, il, graph, false);
        cb(routed_out, "routed_out", il_cb);

        if (split_up_shexp) {
            GGML_ASSERT(!split_up_b_shexp   || split_up_b_shexp->n_device   == split_up_exps->n_device);
            GGML_ASSERT(!split_gate_b_shexp || split_gate_b_shexp->n_device == split_up_exps->n_device);
            GGML_ASSERT(!split_down_b_shexp || split_down_b_shexp->n_device == split_up_exps->n_device);
            auto shared_out = llm_build_ffn(ctx, lctx, nullptr, cur,
                    split_up_shexp->splits[id],   split_up_b_shexp   ? split_up_b_shexp->splits[id]   : nullptr, nullptr,
                    split_gate_shexp->splits[id], split_gate_b_shexp ? split_gate_b_shexp->splits[id] : nullptr, nullptr,
                    split_down_shexp->splits[id], !down_bias_added && split_down_b_shexp ? split_down_b_shexp->splits[id] : nullptr, nullptr,
                    nullptr, type_op_shexp, LLM_FFN_PAR, cb, il);
            cb(shared_out, "ffn_shexp_out", il_cb);

            cur = ggml_add(ctx, routed_out, shared_out);
            cb(cur, "ffn_out", il_cb);
        } else {
            cur = routed_out;
        }
        if (cur->ne[1] > 32 && lctx.cparams.split_mode_f16) {
            cur = ggml_cast(ctx, cur, GGML_TYPE_F16);
            cb(cur, "ffn_out_f16", il_cb);
        }
        ggml_build_forward_expand(graph, cur);
        results[id] = cur;
        last_id = id;
        down_bias_added = true;
    }
    GGML_ASSERT(last_id >= 0);
    if (add_input) {
        results[last_id] = ggml_add(ctx, results[last_id], input);
        cb(results[last_id], "ffn_inp_added", il);
    }

    auto cur = ggml_reduce(ctx, results.data(), split_up_exps->n_device, GGML_OP_ADD);
    cb(cur, "moe_ffn_combined", il);
    ggml_build_forward_expand(graph, cur);

    return cur;
}

static ggml_tensor * llm_build_kqv(
        struct ggml_context * ctx,
       struct llama_context & lctx,
       const llama_kv_cache & kv,
         struct ggml_cgraph * graph,
         struct ggml_tensor * wo,
         struct ggml_tensor * wo_b,
         struct ggml_tensor * q_cur,
         struct ggml_tensor * kq_mask,
                    int32_t   n_tokens,
                    int32_t   n_kv,
                    float     kq_scale,
         const llm_build_cb & cb,
                    int       il,
                ggml_tensor * sinks = nullptr, int n_swa = 0) {
    const llama_model   & model   = lctx.model;
    const llama_hparams & hparams = lctx.model.hparams;
    const llama_cparams & cparams = lctx.cparams;

    const int64_t n_ctx         = cparams.n_ctx;
    const int64_t n_head        = hparams.n_head(il);
    const int64_t n_head_kv     = hparams.n_head_kv(il);
    const int64_t n_embd_head_k = hparams.n_embd_head_k;
    //const int64_t n_embd_k_gqa  = hparams.n_embd_k_gqa(il);
    const int64_t n_embd_head_v = hparams.n_embd_head_v;
    const int64_t n_embd_v_gqa  = hparams.n_embd_v_gqa(il);

    struct ggml_tensor * q = ggml_permute(ctx, q_cur, 0, 2, 1, 3);
    cb(q, "q", il);

    struct ggml_tensor * k =
        ggml_view_3d(ctx, kv.k_l[il],
                n_embd_head_k, n_kv, n_head_kv,
                ggml_row_size(kv.k_l[il]->type, n_embd_head_k)*n_head_kv, //n_embd_k_gqa),
                ggml_row_size(kv.k_l[il]->type, n_embd_head_k),
                0);
    cb(k, "k", il);

#ifdef GGML_USE_VULKAN
    constexpr bool use_f32_precision = true;
#else
    constexpr bool use_f32_precision = false;
#endif

    struct ggml_tensor * cur;

    if (cparams.flash_attn) {
        GGML_UNUSED(model);
        GGML_UNUSED(n_ctx);

        // split cached v into n_head heads (not transposed)
        struct ggml_tensor * v =
            ggml_view_3d(ctx, kv.v_l[il],
                    n_embd_head_v, n_kv, n_head_kv,
                    ggml_row_size(kv.v_l[il]->type, n_embd_v_gqa),
                    ggml_row_size(kv.v_l[il]->type, n_embd_head_v),
                    0);
        cb(v, "v", il);

        cur = ggml_flash_attn_ext(ctx, q, k, v, kq_mask, kq_scale, hparams.f_max_alibi_bias,
                                  hparams.attn_soft_cap ? hparams.f_attn_logit_softcapping : 0.0f);
        ggml_flash_attn_ext_add_sinks(cur, sinks);
        if (n_swa > 0) {
            ((int32_t *)cur->op_params)[4] = n_swa;
        }

        // Some models produced NaNs/gibberish when FA is computed with f16 precision on CUDA
        // For DeepSeek-2, it is perfectly fine with fp16 for PP, but I get gibberish when uding fp16 for TG.
        // Not sure if it is really a matter of insufficient precision, or I have made a mistake in the fattn-vec-f16 kernel.
        if (use_f32_precision || model.arch == LLM_ARCH_PHI2 || model.arch == LLM_ARCH_PHI3 || model.arch == LLM_ARCH_GPTNEOX ||
            (model.arch == LLM_ARCH_DEEPSEEK2 && q->ne[1] <= 8) || model.arch == LLM_ARCH_COHERE2 || model.arch == LLM_ARCH_GLM4 || model.arch == LLM_ARCH_GLM4_MOE) {
            ggml_flash_attn_ext_set_prec(cur, GGML_PREC_F32);
        }
        //ggml_flash_attn_ext_set_prec(cur, GGML_PREC_F32);

        cur = ggml_reshape_2d(ctx, cur, n_embd_head_v*n_head, n_tokens);
    } else {

            // split cached v into n_head heads
        struct ggml_tensor * v =
            ggml_view_3d(ctx, kv.v_l[il],
                    n_kv, n_embd_head_v, n_head_kv,
                    ggml_element_size(kv.v_l[il])*n_ctx,
                    ggml_element_size(kv.v_l[il])*n_ctx*n_embd_head_v,
                    0);
        cb(v, "v", il);

        auto kq_size = k->ne[1]*q->ne[1]*q->ne[2]*sizeof(float)/(1024*1024);
        if (cparams.attn_max_batch == 0 || cparams.attn_max_batch >= kq_size || k->ne[2] != q->ne[2] || v->ne[2] != q->ne[2] || sinks) {
            struct ggml_tensor * kq = ggml_mul_mat(ctx, k, q);
            cb(kq, "kq", il);

            //ggml_mul_mat_set_prec(kq, GGML_PREC_F32);

            if (use_f32_precision || model.arch == LLM_ARCH_PHI2 || model.arch == LLM_ARCH_PHI3 || model.arch == LLM_ARCH_GPTNEOX || model.arch == LLM_ARCH_QWEN2 ||
                model.arch == LLM_ARCH_COHERE2 || model.arch == LLM_ARCH_GLM4 || model.arch == LLM_ARCH_GLM4_MOE || model.arch == LLM_ARCH_MIMO2) {
                // for this arch, we need to perform the KQ multiplication with F32 precision, otherwise we get NaNs
                // ref: https://github.com/ggerganov/llama.cpp/pull/4490#issuecomment-1859055847
                ggml_mul_mat_set_prec(kq, GGML_PREC_F32);
            }

            if (model.arch == LLM_ARCH_GROK) {
                // need to do the following:
                // multiply by attn_output_multiplier
                // and then :
                // kq = 30 * tanh(kq / 30)
                // before the softmax below

                //try from phi2
                //ggml_mul_mat_set_prec(kq, GGML_PREC_F32);

                //kq = ggml_tanh(ctx, ggml_scale(ctx, kq, 0.08838834764831845f/30.0f));
                //kq = ggml_scale(ctx, kq, 30);

                kq = ggml_softcap(ctx, kq, hparams.f_attn_out_scale / hparams.f_attn_logit_softcapping, hparams.f_attn_logit_softcapping);
            }

            if (hparams.attn_soft_cap) {
                //kq = ggml_softcap(ctx, kq, 1.0f / hparams.f_attn_logit_softcapping, hparams.f_attn_logit_softcapping);
                kq = ggml_softcap_max(ctx, kq, kq_mask, kq_scale, hparams.f_max_alibi_bias,
                        1.0f / hparams.f_attn_logit_softcapping, hparams.f_attn_logit_softcapping);
            } else {
                kq = ggml_soft_max_ext(ctx, kq, kq_mask, kq_scale, hparams.f_max_alibi_bias);
                ggml_soft_max_add_sinks(kq, sinks);
            }
            cb(kq, "kq_soft_max_ext", il);

            GGML_ASSERT(kv.size == n_ctx);

            struct ggml_tensor * kqv = ggml_mul_mat(ctx, v, kq);
            cb(kqv, "kqv", il);

            struct ggml_tensor * kqv_merged = ggml_permute(ctx, kqv, 0, 2, 1, 3);
            cb(kqv_merged, "kqv_merged", il);

            cur = ggml_cont_2d(ctx, kqv_merged, n_embd_head_v*n_head, n_tokens);
            cb(cur, "kqv_merged_cont", il);
        }
        else {
            // For now we will not support this option if k->ne[2] != q->ne[2] || v->ne[2] != q->ne[2];
            GGML_ASSERT(k->ne[2] == v->ne[2] && k->ne[2] == q->ne[2]);
            int n_step = (kq_size + cparams.attn_max_batch - 1)/cparams.attn_max_batch;
            n_step = std::min(n_step, int(k->ne[2]));
            int n_per_step = (q->ne[2] + n_step - 1)/n_step;
            auto r2k = q->ne[2] / k->ne[2];
            auto r2v = q->ne[2] / v->ne[2];
            n_step = q->ne[2];
            n_per_step = 1;
            ggml_tensor * kqv = nullptr;
            for (int i12 = 0; i12 < q->ne[2]; i12 += n_per_step) {
                int this_ne12 = i12 + n_per_step <= q->ne[2] ? n_per_step : q->ne[2] - i12;
                int i02 = i12/r2k;
                auto k_i = ggml_view_3d(ctx, k, k->ne[0], k->ne[1], this_ne12, k->nb[1], k->nb[2], k->nb[2]*i02);
                auto q_i = ggml_view_3d(ctx, q, q->ne[0], q->ne[1], this_ne12, q->nb[1], q->nb[2], q->nb[2]*i12);
                auto kq_i = ggml_mul_mat(ctx, k_i, q_i);
                if (model.arch == LLM_ARCH_PHI2 || model.arch == LLM_ARCH_PHI3 || model.arch == LLM_ARCH_GPTNEOX || model.arch == LLM_ARCH_QWEN2 ||
                    model.arch == LLM_ARCH_COHERE2 || model.arch == LLM_ARCH_GLM4 || model.arch == LLM_ARCH_GLM4_MOE) {
                    ggml_mul_mat_set_prec(kq_i, GGML_PREC_F32);
                }
                if (model.arch == LLM_ARCH_GROK) {
                    kq_i = ggml_softcap(ctx, kq_i, hparams.f_attn_out_scale / hparams.f_attn_logit_softcapping, hparams.f_attn_logit_softcapping);
                }
                if (hparams.attn_soft_cap) {
                    kq_i = ggml_softcap_max(ctx, kq_i, kq_mask, kq_scale, hparams.f_max_alibi_bias,
                            1.0f / hparams.f_attn_logit_softcapping, hparams.f_attn_logit_softcapping);
                } else {
                    kq_i = ggml_soft_max_ext(ctx, kq_i, kq_mask, kq_scale, hparams.f_max_alibi_bias);
                }
                i02 = i12 / r2v;
                auto v_i = ggml_view_3d(ctx, v, v->ne[0], v->ne[1], this_ne12, v->nb[1], v->nb[2], v->nb[2]*i02);
                auto kqv_i = ggml_mul_mat(ctx, v_i, kq_i);
                if (i12 == 0) {
                    kqv = kqv_i;
                } else {
                    kqv = ggml_concat(ctx, kqv, kqv_i, 2);
                }
            }
            ggml_tensor * kqv_merged = ggml_permute(ctx, kqv, 0, 2, 1, 3);
            cb(kqv_merged, "kqv_merged", il);
            cur = ggml_cont_2d(ctx, kqv_merged, n_embd_head_v*n_head, n_tokens);
            cb(cur, "kqv_merged_cont", il);
        }
    }

    ggml_build_forward_expand(graph, cur);

    if (wo) {
        cur = llm_build_context::llm_build_lora_mm(lctx, ctx, wo, cur);
        if (lctx.model.arch == LLM_ARCH_GLM4 || lctx.model.arch == LLM_ARCH_GLM4_MOE) {
            // GLM4 and GLM4_MOE seem to have numerical issues with half-precision accumulators
            ggml_mul_mat_set_prec(cur, GGML_PREC_F32);
        }
    }

    if (wo_b) {
        cb(cur, "kqv_wo", il);
    }

    if (wo_b) {
        cur = ggml_add(ctx, cur, wo_b);
    }

    return cur;
}

ggml_tensor * llm_build_context::llm_build_kv(
        ggml_context * ctx,
       llama_context & lctx,
       const llama_kv_cache & kv,
         ggml_cgraph * graph,
         ggml_tensor * wo,
         ggml_tensor * wo_b,
         ggml_tensor * k_cur,
         ggml_tensor * v_cur,
         ggml_tensor * q_cur,
         ggml_tensor * kq_mask,
                    int32_t   n_tokens,
                    int32_t   kv_head,
                    int32_t   n_kv,
                    float     kq_scale,
         const llm_build_cb & cb, int il, ggml_tensor * sinks, int n_swa) {
    const llama_hparams & hparams = lctx.model.hparams;
    const llama_cparams & cparams = lctx.cparams;

    if (cparams.k_cache_hadamard) {
        q_cur = ggml_hadamard(ctx, q_cur, hparams.n_embd_head_k);
        k_cur = ggml_hadamard(ctx, k_cur, hparams.n_embd_head_k);
        cb(q_cur, "Qcur_hadamard", il);
        cb(k_cur, "Kcur_hadamard", il);
    }

    // these nodes are added to the graph together so that they are not reordered
    // by doing so, the number of splits in the graph is reduced
    ggml_build_forward_expand(graph, q_cur);
    ggml_build_forward_expand(graph, k_cur);
    ggml_build_forward_expand(graph, v_cur);

    llm_build_kv_store(lctx, ctx, hparams, cparams, kv, graph, k_cur, v_cur, n_tokens, kv_head, cb, il);

    struct ggml_tensor * cur;

    cur  = llm_build_kqv(ctx, lctx, kv, graph, wo, wo_b,
            q_cur, kq_mask, n_tokens, n_kv, kq_scale, cb, il, sinks, n_swa);
    cb(cur, "kqv_out", il);

    return cur;
}

ggml_tensor * llm_build_context::llm_build_inp_embd_enc() {
    const int64_t n_embd = hparams.n_embd;
    lctx.inp_embd_enc = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd, n_outputs_enc);
    ggml_set_input(lctx.inp_embd_enc);
    cb(lctx.inp_embd_enc, "embd_enc", -1);
    return lctx.inp_embd_enc;
}

ggml_tensor * llm_build_context::llm_build_inp_KQ_mask_cross() {
    lctx.inp_KQ_mask_cross = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_outputs_enc, GGML_PAD(n_tokens, GGML_KQ_MASK_PAD));
    ggml_set_input(lctx.inp_KQ_mask_cross);
    cb(lctx.inp_KQ_mask_cross, "KQ_mask_cross", -1);
    return lctx.inp_KQ_mask_cross;
}

std::tuple<ggml_tensor*, ggml_tensor*, ggml_tensor*> llm_build_context::llm_build_mul_mat_qkv(ggml_cgraph * gf, ggml_tensor * cur,
            ggml_tensor * wq, ggml_tensor * bq,
            ggml_tensor * wk, ggml_tensor * bk,
            ggml_tensor * wv, ggml_tensor * bv,
            float attention_scale, int il, bool add_graph_split) const {
    auto Qcur = llm_build_lora_mm(lctx, ctx0, wq, cur);
    cb(Qcur, "Qcur", il);
    if (add_graph_split) {
        Qcur->op_params[GGML_MAX_OP_PARAMS / sizeof(int32_t) - 1] = 0xff;
    }
    auto Kcur = llm_build_lora_mm(lctx, ctx0, wk, cur);
    cb(Kcur, "Kcur", il);
    auto Vcur = llm_build_lora_mm(lctx, ctx0, wv, cur);
    cb(Vcur, "Vcur", il);
    ggml_build_forward_expand(gf, Qcur);
    ggml_build_forward_expand(gf, Kcur);
    ggml_build_forward_expand(gf, Vcur);

    if (attention_scale != 0) {
        Qcur = ggml_scale(ctx0, Qcur, attention_scale);
        cb(Qcur, "Qcur", il);
    }
    if (bq) {
        Qcur = ggml_add(ctx0, Qcur, bq);
        cb(Qcur, "Qcur", il);
        ggml_build_forward_expand(gf, Qcur);
    }
    if (bk) {
        Kcur = ggml_add(ctx0, Kcur, bk);
        cb(Kcur, "Kcur", il);
        ggml_build_forward_expand(gf, Kcur);
    }
    if (bv) {
        Vcur = ggml_add(ctx0, Vcur, bv);
        cb(Vcur, "Vcur", il);
        ggml_build_forward_expand(gf, Vcur);
    }
    return {Qcur, Kcur, Vcur};
}

std::tuple<ggml_tensor*, ggml_tensor*, ggml_tensor*> llm_build_context::llm_build_mul_mat_qkv(ggml_cgraph * gf, ggml_tensor * cur,
            ggml_tensor * wqkv, ggml_tensor * bqkv,
            ggml_tensor * wqk, ggml_tensor * bqk,
            ggml_tensor * wq, ggml_tensor * bq,
            ggml_tensor * wk, ggml_tensor * bk,
            ggml_tensor * wv, ggml_tensor * bv,
            ggml_tensor * q_norm, ggml_tensor * k_norm, float attention_scale, int il, bool add_graph_split) const {
    const int64_t n_embd_head_k = hparams.n_embd_head_k;
    const int64_t n_embd_gqa  = hparams.n_embd_v_gqa();
    if (wqkv) {
        auto qkv = llm_build_lora_mm(lctx, ctx0, wqkv, cur);
        if (add_graph_split) {
            qkv->op_params[GGML_MAX_OP_PARAMS / sizeof(int32_t) - 1] = 0xff;
        }
        cb(qkv, "qkv", il);
        if (bqkv) {
            qkv = ggml_add(ctx0, qkv, bqkv);
            cb(qkv, "qkv_b", il);
        }
        auto Qcur = ggml_view_3d(ctx0, qkv, n_embd_head_k, n_head,    n_tokens, n_embd_head_k*sizeof(float), qkv->nb[1], 0*sizeof(float)*(n_embd));
        auto Kcur = ggml_view_3d(ctx0, qkv, n_embd_head_k, n_head_kv, n_tokens, n_embd_head_k*sizeof(float), qkv->nb[1], 1*sizeof(float)*Qcur->ne[0]*Qcur->ne[1]);
        auto Vcur = ggml_view_2d(ctx0, qkv, n_embd_gqa, n_tokens, qkv->nb[1], 1*sizeof(float)*(Qcur->ne[0]*Qcur->ne[1] + Kcur->ne[0]*Kcur->ne[1]));
        cb(Qcur, "Qcur", il);
        cb(Kcur, "Kcur", il);
        cb(Vcur, "Vcur", il);
        if (q_norm) {
            Qcur = llm_build_norm(ctx0, Qcur, hparams, q_norm, NULL, LLM_NORM_RMS, cb, il);
            cb(Qcur, "Qcur_normed", il);
            ggml_build_forward_expand(gf, Qcur);
        }
        if (k_norm) {
            Kcur = llm_build_norm(ctx0, Kcur, hparams, k_norm, NULL, LLM_NORM_RMS, cb, il);
            cb(Kcur, "Kcur_normed", il);
            ggml_build_forward_expand(gf, Kcur);
        }

        return {Qcur, Kcur, Vcur};

        //ggml_build_forward_expand(gf, Qcur);
        //ggml_build_forward_expand(gf, Kcur);
        //ggml_build_forward_expand(gf, Vcur);
    }

    if (wqk) {
        auto qk = llm_build_lora_mm(lctx, ctx0, wqk, cur);
        if (add_graph_split) {
            qk->op_params[GGML_MAX_OP_PARAMS / sizeof(int32_t) - 1] = 0xff;
        }
        cb(qk, "qkv", il);
        if (bqk) {
            qk = ggml_add(ctx0, qk, bqk);
            cb(qk, "qkv_b", il);
        }
        auto Vcur = llm_build_lora_mm(lctx, ctx0, wv, cur);
        cb(Vcur, "Vcur", il);
        if (bv) {
            Vcur = ggml_add(ctx0, Vcur, bv);
            cb(Vcur, "Vcur", il);
        }
        ggml_build_forward_expand(gf, qk);
        ggml_build_forward_expand(gf, Vcur);
        auto Qcur = ggml_view_3d(ctx0, qk, n_embd_head_k, n_head,    n_tokens, n_embd_head_k*sizeof(float), qk->nb[1], 0*sizeof(float)*(n_embd));
        auto Kcur = ggml_view_3d(ctx0, qk, n_embd_head_k, n_head_kv, n_tokens, n_embd_head_k*sizeof(float), qk->nb[1], 1*sizeof(float)*Qcur->ne[0]*Qcur->ne[1]);
        cb(Qcur, "Qcur", il);
        cb(Kcur, "Kcur", il);
        if (q_norm) {
            Qcur = llm_build_norm(ctx0, Qcur, hparams, q_norm, NULL, LLM_NORM_RMS, cb, il);
            cb(Qcur, "Qcur_normed", il);
            ggml_build_forward_expand(gf, Qcur);
        }
        if (k_norm) {
            Kcur = llm_build_norm(ctx0, Kcur, hparams, k_norm, NULL, LLM_NORM_RMS, cb, il);
            cb(Kcur, "Kcur_normed", il);
            ggml_build_forward_expand(gf, Kcur);
        }

        return {Qcur, Kcur, Vcur};

    }

    auto [Q, K, V] = llm_build_mul_mat_qkv(gf, cur, wq, bq, wk, bk, wv, bv, attention_scale, il, add_graph_split);
    auto Qcur = ggml_reshape_3d(ctx0, Q, n_embd_head_k, Q->ne[0]/n_embd_head_k, n_tokens);
    if (q_norm) {
        Qcur = llm_build_norm(ctx0, Qcur, hparams, q_norm, NULL, LLM_NORM_RMS, cb, il);
        cb(Qcur, "Qcur_normed", il);
    }

    auto Kcur = ggml_reshape_3d(ctx0, K, n_embd_head_k, K->ne[0]/n_embd_head_k, n_tokens);
    if (k_norm) {
        Kcur = llm_build_norm(ctx0, Kcur, hparams, k_norm, NULL, LLM_NORM_RMS, cb, il);
        cb(Kcur, "Kcur_normed", il);
    }
    auto Vcur = V;
    return {Qcur, Kcur, Vcur};
}

static ggml_tensor * build_output(llama_context & lctx, ggml_context * ctx, ggml_tensor * cur, ggml_tensor * output, const llm_build_cb & cb) {
    // lm_head
    if (output->extra) {
        auto split_output = (ggml_split_tensor_t *)output->extra;
        std::vector<ggml_tensor *> o;
        o.reserve(split_output->n_device);
        for (int id = 0; id < split_output->n_device; ++id) {
            auto split = split_output->splits[id];
            if (!split) continue;
            o.push_back(llm_build_context::llm_build_lora_mm(lctx, ctx, split, cur));
            cb(o.back(), "output", id);
        }
        if (o.size() == 1) cur = o.front();
        cur = ggml_concat(ctx, o[0], o[1], 0);
        for (int id = 2; id < int(o.size()); ++id) {
            cur = ggml_concat(ctx, cur, o[id], 0);
        }
    } else {
        cur = llm_build_context::llm_build_lora_mm(lctx, ctx, output, cur);
    }
    return cur;
}

static ggml_tensor * build_output(llama_context & lctx, ggml_context * ctx, ggml_tensor * cur, ggml_tensor * output, ggml_tensor * output_norm, const llm_build_cb & cb) {
    // lm_head
    if (output->extra) {
        auto split_output = (ggml_split_tensor_t *)output->extra;
        auto split_output_norm = output_norm && output_norm->extra ? (ggml_split_tensor_t *)output_norm->extra : nullptr;
        std::vector<ggml_tensor *> o;
        o.reserve(split_output->n_device);
        for (int id = 0; id < split_output->n_device; ++id) {
            auto split = split_output->splits[id];
            if (!split) continue;
            if (output_norm) {
                auto the_norm = split_output_norm ? split_output_norm->splits[id] : output_norm;
                auto cur_normed = llm_build_context::llm_build_norm(ctx, cur, lctx.model.hparams, the_norm, NULL, LLM_NORM_RMS, cb, -1);
                cb(cur_normed, "output_normed", 1000*(id+1));
                o.push_back(llm_build_context::llm_build_lora_mm(lctx, ctx, split, cur_normed));
            } else {
                o.push_back(llm_build_context::llm_build_lora_mm(lctx, ctx, split, cur));
            }
            cb(o.back(), "output", id);
        }
        GGML_ASSERT(!o.empty());
        if (o.size() == 1) {
            cur = o.front();
        }
        else {
            cur = ggml_concat(ctx, o[0], o[1], 0);
            for (int id = 2; id < int(o.size()); ++id) {
                cur = ggml_concat(ctx, cur, o[id], 0);
            }
        }
    } else {
        if (output_norm) {
            cur = llm_build_context::llm_build_norm(ctx, cur, lctx.model.hparams, output_norm, NULL, LLM_NORM_RMS, cb, -1);
            cb(cur, "output_normed", -1);
        }
        cur = llm_build_context::llm_build_lora_mm(lctx, ctx, output, cur);
    }
    return cur;
}

ggml_cgraph * llm_build_context::build_llama() {
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, model.max_nodes(), false);

    // mutable variable, needed during the last layer of the computation to skip unused tokens
    int32_t n_tokens = this->n_tokens;

    const int64_t n_embd_head = hparams.n_embd_head_v;
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);
    GGML_ASSERT(n_embd_head == hparams.n_rot);

    ggml_tensor * cur;
    ggml_tensor * inpL;
    ggml_tensor * inp_attn_scale = nullptr;

    inpL = llm_build_inp_embd(ctx0, lctx, hparams, batch, model.tok_embd, cb);

    // inp_pos - contains the positions
    struct ggml_tensor * inp_pos = build_inp_pos();

    if (model.arch == LLM_ARCH_LLAMA4) {
        inp_attn_scale = build_input_scale(n_tokens);
    }

    // KQ_mask (mask for 1 head, it will be broadcasted to all heads)
    //bool is_swa = hparams.n_swa > 0 && h_params.n_swa_pattern > 0 ?
    ggml_tensor * KQ_mask = build_inp_KQ_mask();
    ggml_tensor * KQ_mask_swa = nullptr;
    if (hparams.n_swa > 0 && hparams.n_swa_pattern > 0) {
        KQ_mask_swa = build_inp_KQ_mask_swa();
    }

    //const float kq_scale = hparams.f_attention_scale == 0.0f ? 1.0f/sqrtf(float(n_embd_head)) : hparams.f_attention_scale;
    const float kq_scale = hparams.f_attention_scale == 0.0f ? 1.0f/sqrtf(float(n_embd_head)) : 1.f;
    for (int il = 0; il < n_layer; ++il) {
        struct ggml_tensor * inpSA = inpL;

        bool use_rope = model.arch == LLM_ARCH_LLAMA4 ? (il + 1) % hparams.n_no_rope_layer_step != 0 : true;
        auto this_KQ_mask = hparams.n_swa > 0 && hparams.n_swa_pattern > 0 && il % hparams.n_swa_pattern < (hparams.n_swa_pattern - 1) ?
            KQ_mask_swa : KQ_mask;
        int this_n_swa = this_KQ_mask == KQ_mask_swa ? hparams.n_swa : 0;

        // rope freq factors for llama3; may return nullptr for llama2 and other models
        //auto rope_factors = build_rope_factors(il);

        // self-attention
        if (use_rope) {
            cur = build_std_attention(gf, model.layers[il].attn_norm, inpL, inp_pos, nullptr,
                    this_KQ_mask, nullptr, nullptr, kq_scale, hparams.f_attention_scale, this_n_swa, il, true, false, true);
        }
        else {

            auto rope_factors = build_rope_factors(il);

            // norm
            cur = llm_build_norm(ctx0, inpL, hparams, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, cb, il);
            cb(cur, "attn_norm", il);

            auto [Qcur, Kcur, Vcur] = llm_build_mul_mat_qkv(gf, cur,
                    model.layers[il].wqkv, model.layers[il].bqkv,
                    model.layers[il].wqk, model.layers[il].bqk,
                    model.layers[il].wq, model.layers[il].bq,
                    model.layers[il].wk, model.layers[il].bk,
                    model.layers[il].wv, model.layers[il].bv,
                    nullptr, nullptr, hparams.f_attention_scale, il);

            if (use_rope) {
                Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, rope_factors,
                        n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                        ext_factor, attn_factor, beta_fast, beta_slow);

                Kcur = ggml_rope_ext(ctx0, Kcur, inp_pos, rope_factors,
                        n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                        ext_factor, attn_factor, beta_fast, beta_slow);
            } else if (inp_attn_scale) {
                Qcur = ggml_mul(ctx0, Qcur, inp_attn_scale);
            }

            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            if (model.arch == LLM_ARCH_LLAMA4 && use_rope && hparams.use_kq_norm) {
                // Llama4TextL2Norm
                Qcur = ggml_rms_norm(ctx0, Qcur, hparams.f_norm_rms_eps);
                Kcur = ggml_rms_norm(ctx0, Kcur, hparams.f_norm_rms_eps);
                cb(Qcur, "Qcur_normed", il);
                cb(Kcur, "Kcur_normed", il);
            }

            cur = llm_build_kv(ctx0, lctx, kv_self, gf,
                    model.layers[il].wo, model.layers[il].bo,
                    Kcur, Vcur, Qcur, this_KQ_mask, n_tokens, kv_head, n_kv, kq_scale, cb, il, nullptr,
                    this_n_swa);
        }
        //printf("%s: attn result for layer %d is %s, %s\n", __func__, il, cur->name, ggml_op_name(cur->op));

        if (il == n_layer - 1) {
            // skip computing output for unused tokens
            struct ggml_tensor * inp_out_ids = build_inp_out_ids();
            n_tokens = n_outputs;
            cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
            cb(cur, "last_attn", il);
            if (!use_rope) {
                inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
                cb(inpSA, "last_ffn_inp", il);
            }
        }

        // For Granite architecture
        if (hparams.f_residual_scale) {
            // Why is hparams.f_residual_scale not simply absorbed into model.layers[il].wv ?
            cur = ggml_scale(ctx0, cur, hparams.f_residual_scale);
        }

        ggml_tensor * ffn_inp;
        if (use_rope) {
            ffn_inp = cur;
        } else {
            ffn_inp = ggml_add(ctx0, cur, inpSA);
            cb(ffn_inp, "ffn_inp", il);
        }

        // feed-forward network
        if (model.layers[il].ffn_gate_inp == nullptr) {
            // non-MoE
            cur = llm_build_ffn(ctx0, lctx, model.layers[il].ffn_norm, ffn_inp,
                    model.layers[il].ffn_up,   model.layers[il].ffn_up_b,   NULL,
                    model.layers[il].ffn_gate, model.layers[il].ffn_gate_b, NULL,
                    model.layers[il].ffn_down, model.layers[il].ffn_down_b, NULL,
                    NULL,
                    LLM_FFN_SILU, LLM_FFN_PAR, cb, il, gf, true);
            cb(cur, "ffn_out", il);
        } else if (model.arch == LLM_ARCH_LLAMA4) {
            // llama4 MoE
            ggml_tensor * ffn_inp_normed = llm_build_norm(ctx0, ffn_inp, hparams, model.layers[il].ffn_norm, NULL, LLM_NORM_RMS, cb, il);
            cb(cur, "ffn_norm", il);

            ggml_tensor * moe_out = llm_build_moe_ffn(ctx0, lctx, ffn_inp_normed,
                    model.layers[il].ffn_gate_inp,
                    model.layers[il].ffn_up_exps,
                    model.layers[il].ffn_gate_exps,
                    model.layers[il].ffn_down_exps,
                    nullptr,
                    n_expert, n_expert_used,
                    LLM_FFN_SILU, false,
                    false, 0.0,
                    LLM_EXPERT_GATING_FUNC_SIGMOID,
                    cb, il, gf, true);

            // Shared experts
            ggml_tensor * shexp_out = llm_build_ffn(ctx0, lctx, nullptr, ffn_inp_normed,
                    model.layers[il].ffn_up_shexp,   NULL, NULL,
                    model.layers[il].ffn_gate_shexp, NULL, NULL,
                    model.layers[il].ffn_down_shexp, NULL, NULL,
                    NULL,
                    LLM_FFN_SILU, LLM_FFN_PAR, cb, il);
            cb(shexp_out, "ffn_moe_shexp", il);

            cur = ggml_add(ctx0, moe_out, shexp_out);
            cb(cur, "ffn_moe_out_merged", il);

        } else {
            // MoE branch
            cur = llm_build_norm(ctx0, ffn_inp, hparams, model.layers[il].ffn_norm, NULL, LLM_NORM_RMS, cb, il);
            cb(cur, "ffn_norm", il);

            cur = llm_build_moe_ffn(ctx0, lctx, cur,
                    model.layers[il].ffn_gate_inp,
                    model.layers[il].ffn_up_exps,
                    model.layers[il].ffn_gate_exps,
                    model.layers[il].ffn_down_exps,
                    nullptr,
                    n_expert, n_expert_used,
                    LLM_FFN_SILU, true,
                    false, 0.0,
                    LLM_EXPERT_GATING_FUNC_SOFTMAX,
                    cb, il, gf, true);
            cb(cur, "ffn_moe_out", il);
        }
        //printf("%s: ffn result for layer %d is %s, %s\n", __func__, il, cur->name, ggml_op_name(cur->op));

        // For Granite architecture
        if (hparams.f_residual_scale) {
            // Why is hparams.f_residual_scale not simply absorbed into model.layers[il].ffn_down_exps ?
            cur = ggml_scale(ctx0, cur, hparams.f_residual_scale);
        }

        //cur = ggml_add(ctx0, cur, ffn_inp);
        //cb(cur, "ffn_out", il);

        cur = lctx.cvec.apply_to(ctx0, cur, il);
        cb(cur, "l_out", il);

        // input for next layer
        inpL = cur;
    }

    cur = inpL;

    // lm_head
    cur = build_output(lctx, ctx0, cur, model.output, model.output_norm, cb);

    // For Granite architecture
    if (hparams.f_logit_scale) {
        // Why is hparams.f_logit_scale not simply absorbed into model.output ?
        cur = ggml_scale(ctx0, cur, 1.0f / hparams.f_logit_scale);
    }

    cb(cur, "result_output", -1);

    ggml_build_forward_expand(gf, cur);

    return gf;
}

ggml_cgraph * llm_build_context::build_mistral3() {
    auto gf = ggml_new_graph_custom(ctx0, model.max_nodes(), false);
    const int64_t n_embd_head = hparams.n_embd_head_v;

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);
    GGML_ASSERT(n_embd_head == hparams.n_rot);

    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = llm_build_inp_embd(ctx0, lctx, hparams, batch, model.tok_embd, cb);

    // inp_pos - contains the positions
    struct ggml_tensor * inp_pos = build_inp_pos();

    // (optional) temperature tuning
    ggml_tensor * inp_attn_scale = nullptr;
    if (hparams.f_attn_temp_scale != 0.0f) {
        inp_attn_scale = build_input_scale(n_tokens);
    }

    ggml_tensor * KQ_mask = build_inp_KQ_mask();

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    //const float kq_scale = hparams.f_attention_scale == 0.0f ? 1.0f/sqrtf(float(n_embd_head)) : hparams.f_attention_scale;
    const float kq_scale = hparams.f_attention_scale == 0.0f ? 1.0f/sqrtf(float(n_embd_head)) : 1.f;

    for (int il = 0; il < n_layer; ++il) {
        ggml_tensor * inpSA = inpL;

        auto rope_factors = build_rope_factors(il);

        cur = build_std_attention(gf, model.layers[il].attn_norm, inpL, inp_pos, rope_factors, KQ_mask,
                nullptr, inp_attn_scale, kq_scale, hparams.f_attention_scale, 0, il);

        if (il == n_layer - 1 && inp_out_ids) {
            cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
            cb(cur, "last_attn", il);
            cb(inpSA, "last_ffn_inp", il);
        }

        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        // feed-forward network (non-MoE)
        if (model.layers[il].ffn_gate_inp == nullptr) {
            // non-MoE
            cur = llm_build_ffn(ctx0, lctx, model.layers[il].ffn_norm, ffn_inp,
                    model.layers[il].ffn_up,   model.layers[il].ffn_up_b,   nullptr,
                    model.layers[il].ffn_gate, model.layers[il].ffn_gate_b, nullptr,
                    model.layers[il].ffn_down, model.layers[il].ffn_down_b, nullptr,
                    NULL,
                    LLM_FFN_SILU, LLM_FFN_PAR, cb, il, gf);
            cb(cur, "ffn_out", il);
        } else {
            // MoE branch
            cur = llm_build_std_moe_ffn(ctx0, lctx, model.layers[il].ffn_norm, ffn_inp,
                    model.layers[il].ffn_gate_inp,  nullptr,
                    model.layers[il].ffn_up_exps,   nullptr,
                    model.layers[il].ffn_gate_exps, nullptr,
                    model.layers[il].ffn_down_exps, nullptr,
                    model.layers[il].ffn_exp_probs_b,
                    nullptr,  nullptr, // we don't have shared experts
                    nullptr,  nullptr,
                    nullptr,  nullptr,
                    n_expert, n_expert_used,
                    LLM_FFN_SILU, true, false, 0.0f,
                    LLM_EXPERT_GATING_FUNC_SOFTMAX,
                    LLM_FFN_SILU, cb, il, gf);
        }
        cur = ggml_add(ctx0, cur, ffn_inp);
        cb(cur, "ffn_out", il);

        cur = lctx.cvec.apply_to(ctx0, cur, il);
        cb(cur, "l_out", il);

        // input for next layer
        inpL = cur;
    }
    cur = inpL;

    cur = build_output(lctx, ctx0, cur, model.output, model.output_norm, cb);
    cb(cur, "result_output", -1);

    ggml_build_forward_expand(gf, cur);

    return gf;
}

ggml_cgraph * llm_build_context::build_deci() {
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, model.max_nodes(), false);

    // mutable variable, needed during the last layer of the computation to skip unused tokens
    int32_t n_tokens = this->n_tokens;

    const int64_t n_embd_head = hparams.n_embd_head_v;
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);
    GGML_ASSERT(n_embd_head == hparams.n_rot);

    struct ggml_tensor * cur;
    struct ggml_tensor * inpL;

    inpL = llm_build_inp_embd(ctx0, lctx, hparams, batch, model.tok_embd, cb);

    // inp_pos - contains the positions
    struct ggml_tensor * inp_pos = build_inp_pos();

    // KQ_mask (mask for 1 head, it will be broadcasted to all heads)
    struct ggml_tensor * KQ_mask = build_inp_KQ_mask();

    const float kq_scale = hparams.f_attention_scale == 0.0f ? 1.0f/sqrtf(float(n_embd_head)) : hparams.f_attention_scale;
    for (int il = 0; il < n_layer; ++il) {
        struct ggml_tensor * inpSA = inpL;
        const int64_t n_head_kv = hparams.n_head_kv(il);
        const int64_t n_head    = hparams.n_head(il);
        const int64_t n_ff      = hparams.n_ff(il);

        if (n_head == 0) { // attention-free layer of Llama-3_1-Nemotron-51B
            cur = inpL;
        } else {
            // norm
            cur = llm_build_norm(ctx0, inpL, hparams, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, cb, il);
            cb(cur, "attn_norm", il);
        }

        if (n_head > 0 && n_head_kv == 0) { // "linear attention" of Llama-3_1-Nemotron-51B
            cur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wo, cur);
            cb(cur, "wo", il);
        } else if (n_head > 0) {
            // self-attention
            // rope freq factors for llama3; may return nullptr for llama2 and other models
            struct ggml_tensor * rope_factors = build_rope_factors(il);

            auto [Qcur, Kcur, Vcur] = llm_build_mul_mat_qkv(gf, cur, model.layers[il].wq, model.layers[il].bq,
                    model.layers[il].wk, model.layers[il].bk,
                    model.layers[il].wv, model.layers[il].bv,
                    0.f, il);

            Qcur = ggml_rope_ext(
                    ctx0, ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens), inp_pos, rope_factors,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );
            cb(Qcur, "Qcur", il);

            Kcur = ggml_rope_ext(
                    ctx0, ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens), inp_pos, rope_factors,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );
            cb(Kcur, "Kcur", il);

            cur = llm_build_kv(ctx0, lctx, kv_self, gf,
                    model.layers[il].wo, model.layers[il].bo,
                    Kcur, Vcur, Qcur, KQ_mask, n_tokens, kv_head, n_kv, kq_scale, cb, il);
        }

        if (il == n_layer - 1) {
            // skip computing output for unused tokens
            struct ggml_tensor * inp_out_ids = build_inp_out_ids();
            n_tokens = n_outputs;
            cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }

        // FFN-free layer of Llama-3_1-Nemotron-Ultra-253B
        if (n_ff == 0) {
            continue;
        }

        if (hparams.f_residual_scale) {
            cur = ggml_scale(ctx0, cur, hparams.f_residual_scale);
        }

        // modified to support attention-free layer of Llama-3_1-Nemotron-51B
        struct ggml_tensor * ffn_inp = cur;
        if (n_head > 0) {
            ffn_inp = ggml_add(ctx0, cur, inpSA);
            cb(ffn_inp, "ffn_inp", il);
        }

        // feed-forward network
        if (model.layers[il].ffn_gate_inp == nullptr) {

            cur = llm_build_ffn(ctx0, lctx, model.layers[il].ffn_norm, ffn_inp,
                    model.layers[il].ffn_up,   model.layers[il].ffn_up_b,   NULL,
                    model.layers[il].ffn_gate, model.layers[il].ffn_gate_b, NULL,
                    model.layers[il].ffn_down, model.layers[il].ffn_down_b, NULL,
                    NULL,
                    LLM_FFN_SILU, LLM_FFN_PAR, cb, il);
            cb(cur, "ffn_out", il);
        }

        if (hparams.f_residual_scale) {
            cur = ggml_scale(ctx0, cur, hparams.f_residual_scale);
        }

        cur = ggml_add(ctx0, cur, ffn_inp);
        cb(cur, "ffn_out", il);

        cur = lctx.cvec.apply_to(ctx0, cur, il);
        cb(cur, "l_out", il);

        // input for next layer
        inpL = cur;
    }

    cur = inpL;

    cur = llm_build_norm(ctx0, cur, hparams, model.output_norm, NULL, LLM_NORM_RMS, cb, -1);
    cb(cur, "result_norm", -1);

    // lm_head
    cur = llm_build_lora_mm(lctx, ctx0, model.output, cur);

    if (hparams.f_logit_scale) {
        cur = ggml_scale(ctx0, cur, 1.0f / hparams.f_logit_scale);
    }

    cb(cur, "result_output", -1);

    ggml_build_forward_expand(gf, cur);

    return gf;
}

ggml_cgraph * llm_build_context::build_baichuan() {
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, model.max_nodes(), false);

    const int64_t n_embd_head = hparams.n_embd_head_v;
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);
    GGML_ASSERT(n_embd_head == hparams.n_rot);

    struct ggml_tensor * cur;
    struct ggml_tensor * inpL;

    inpL = llm_build_inp_embd(ctx0, lctx, hparams, batch, model.tok_embd, cb);

    // inp_pos - contains the positions
    struct ggml_tensor * inp_pos = model.type == MODEL_7B ? build_inp_pos() : nullptr;

    // KQ_mask (mask for 1 head, it will be broadcasted to all heads)
    struct ggml_tensor * KQ_mask = build_inp_KQ_mask();

    for (int il = 0; il < n_layer; ++il) {
        struct ggml_tensor * inpSA = inpL;

        cur = llm_build_norm(ctx0, inpL, hparams, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, cb, il);
        cb(cur, "attn_norm", il);

        // self-attention
        {
            auto [Qcur, Kcur, Vcur] = llm_build_mul_mat_qkv(gf, cur, model.layers[il].wq, nullptr,
                    model.layers[il].wk, nullptr,
                    model.layers[il].wv, nullptr, 0, il);
            switch (model.type) {
                case MODEL_7B:
                    Qcur = ggml_rope_ext(
                            ctx0, ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens), inp_pos, nullptr,
                            n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                            ext_factor, attn_factor, beta_fast, beta_slow
                            );
                    Kcur = ggml_rope_ext(
                            ctx0, ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens), inp_pos, nullptr,
                            n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                            ext_factor, attn_factor, beta_fast, beta_slow
                            );
                    break;
                case MODEL_13B:
                    Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd/n_head, n_head, n_tokens);
                    Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd/n_head, n_head, n_tokens);
                    break;
                default:
                    GGML_ABORT("fatal error");
            }
            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);

            cur = llm_build_kv(ctx0, lctx, kv_self, gf,
                    model.layers[il].wo, NULL,
                    Kcur, Vcur, Qcur, KQ_mask, n_tokens, kv_head, n_kv, 1.0f/sqrtf(float(n_embd_head)), cb, il);
        }

        if (il == n_layer - 1) {
            // skip computing output for unused tokens
            struct ggml_tensor * inp_out_ids = build_inp_out_ids();
            cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }

        struct ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        // feed-forward network
        {
            cur = llm_build_ffn(ctx0, lctx, model.layers[il].ffn_norm, ffn_inp,
                    model.layers[il].ffn_up,   NULL, NULL,
                    model.layers[il].ffn_gate, NULL, NULL,
                    model.layers[il].ffn_down, NULL, NULL,
                    NULL,
                    LLM_FFN_SILU, LLM_FFN_PAR, cb, il);
            cb(cur, "ffn_out", il);
        }

        cur = ggml_add(ctx0, cur, ffn_inp);
        cur = lctx.cvec.apply_to(ctx0, cur, il);
        cb(cur, "l_out", il);

        // input for next layer
        inpL = cur;
    }

    cur = inpL;

    cur = llm_build_norm(ctx0, cur, hparams, model.output_norm, NULL, LLM_NORM_RMS, cb, -1);
    cb(cur, "result_norm", -1);

    // lm_head
    cur = llm_build_lora_mm(lctx, ctx0, model.output, cur);
    cb(cur, "result_output", -1);

    ggml_build_forward_expand(gf, cur);

    return gf;
}

static inline size_t llama_model_max_nodes(const llama_model & model) { return model.max_nodes(); }

ggml_cgraph * llm_build_context::build_xverse() {
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, llama_model_max_nodes(model), false);

    const int64_t n_embd_head = hparams.n_embd_head_v;
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);
    GGML_ASSERT(n_embd_head == hparams.n_rot);

    struct ggml_tensor * cur;
    struct ggml_tensor * inpL;

    inpL = llm_build_inp_embd(ctx0, lctx, hparams, batch, model.tok_embd, cb);

    // inp_pos - contains the positions
    struct ggml_tensor * inp_pos = build_inp_pos();

    // KQ_mask (mask for 1 head, it will be broadcasted to all heads)
    struct ggml_tensor * KQ_mask = build_inp_KQ_mask();

    for (int il = 0; il < n_layer; ++il) {
        struct ggml_tensor * inpSA = inpL;

        cur = llm_build_norm(ctx0, inpL, hparams, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, cb, il);
        cb(cur, "attn_norm", il);

        // self-attention
        {
            auto [Qcur, Kcur, Vcur] = llm_build_mul_mat_qkv(gf, cur, model.layers[il].wq, nullptr,
                    model.layers[il].wk, nullptr,
                    model.layers[il].wv, nullptr, 0, il);
            Qcur = ggml_rope_ext(
                    ctx0, ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens), inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );
            cb(Qcur, "Qcur", il);

            Kcur = ggml_rope_ext(
                    ctx0, ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens), inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );
            cb(Kcur, "Kcur", il);
            cur = llm_build_kv(ctx0, lctx, kv_self, gf,
                    model.layers[il].wo, NULL,
                    Kcur, Vcur, Qcur, KQ_mask, n_tokens, kv_head, n_kv, 1.0f/sqrtf(float(n_embd_head)), cb, il);
        }

        if (il == n_layer - 1) {
            // skip computing output for unused tokens
            struct ggml_tensor * inp_out_ids = build_inp_out_ids();
            cur   = ggml_get_rows(ctx0,      cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }

        struct ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        // feed-forward network
        {
            cur = llm_build_ffn(ctx0, lctx, model.layers[il].ffn_norm, ffn_inp,
                    model.layers[il].ffn_up,   NULL, NULL,
                    model.layers[il].ffn_gate, NULL, NULL,
                    model.layers[il].ffn_down, NULL, NULL,
                    NULL,
                    LLM_FFN_SILU, LLM_FFN_PAR, cb, il);
            cb(cur, "ffn_out", il);
        }

        cur = ggml_add(ctx0, cur, ffn_inp);
        cur = lctx.cvec.apply_to(ctx0, cur, il);
        cb(cur, "l_out", il);

        // input for next layer
        inpL = cur;
    }

    cur = inpL;

    cur = llm_build_norm(ctx0, cur, hparams, model.output_norm, NULL, LLM_NORM_RMS, cb, -1);
    cb(cur, "result_norm", -1);

    // lm_head
    cur = llm_build_lora_mm(lctx, ctx0, model.output, cur);
    cb(cur, "result_output", -1);

    ggml_build_forward_expand(gf, cur);

    return gf;
}

ggml_cgraph * llm_build_context::build_falcon() {
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, llama_model_max_nodes(model), false);

    const int64_t n_embd_head = hparams.n_embd_head_v;
    const int64_t n_embd_gqa  = hparams.n_embd_v_gqa();
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);
    GGML_ASSERT(n_embd_head == hparams.n_rot);

    struct ggml_tensor * cur;
    struct ggml_tensor * inpL;

    inpL = llm_build_inp_embd(ctx0, lctx, hparams, batch, model.tok_embd, cb);

    // inp_pos - contains the positions
    struct ggml_tensor * inp_pos = build_inp_pos();

    // KQ_mask (mask for 1 head, it will be broadcasted to all heads)
    struct ggml_tensor * KQ_mask = build_inp_KQ_mask();

    for (int il = 0; il < n_layer; ++il) {
        struct ggml_tensor * attn_norm;

        attn_norm = llm_build_norm(ctx0, inpL, hparams, model.layers[il].attn_norm, model.layers[il].attn_norm_b, LLM_NORM, cb, il);
        cb(attn_norm, "attn_norm", il);

        // self-attention
        {
            if (model.layers[il].attn_norm_2) {
                // Falcon-40B
                cur = llm_build_norm(ctx0, inpL, hparams, model.layers[il].attn_norm_2, model.layers[il].attn_norm_2_b, LLM_NORM, cb, il);
                cb(cur, "attn_norm_2", il);
            } else {
                cur = attn_norm;
            }

            cur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wqkv, cur);
            cb(cur, "wqkv", il);

            struct ggml_tensor * Qcur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd,     n_tokens, cur->nb[1], 0*sizeof(float)*(n_embd)));
            struct ggml_tensor * Kcur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd_gqa, n_tokens, cur->nb[1], 1*sizeof(float)*(n_embd)));
            struct ggml_tensor * Vcur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd_gqa, n_tokens, cur->nb[1], 1*sizeof(float)*(n_embd + n_embd_gqa)));

            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
            Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);

            // using mode = 2 for neox mode
            Qcur = ggml_rope_ext(
                    ctx0, Qcur, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig,
                    freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow
                    );
            cb(Qcur, "Qcur", il);

            Kcur = ggml_rope_ext(
                    ctx0, Kcur, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig,
                    freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow
                    );
            cb(Kcur, "Kcur", il);

            cur = llm_build_kv(ctx0, lctx, kv_self, gf,
                    model.layers[il].wo, NULL,
                    Kcur, Vcur, Qcur, KQ_mask, n_tokens, kv_head, n_kv, 1.0f/sqrtf(float(n_embd_head)), cb, il);
        }

        if (il == n_layer - 1) {
            // skip computing output for unused tokens
            struct ggml_tensor * inp_out_ids = build_inp_out_ids();
            cur       = ggml_get_rows(ctx0,       cur, inp_out_ids);
            inpL      = ggml_get_rows(ctx0,      inpL, inp_out_ids);
            attn_norm = ggml_get_rows(ctx0, attn_norm, inp_out_ids);
        }

        struct ggml_tensor * ffn_inp = cur;

        // feed forward
        {
            cur = llm_build_ffn(ctx0, lctx, nullptr, attn_norm, // !! use the attn norm, not the result
                    model.layers[il].ffn_up,   NULL, NULL,
                    NULL,                      NULL, NULL,
                    model.layers[il].ffn_down, NULL, NULL,
                    NULL,
                    LLM_FFN_GELU, LLM_FFN_SEQ, cb, il);
            cb(cur, "ffn_out", il);
        }

        cur = ggml_add(ctx0, cur, ffn_inp);
        cur = ggml_add(ctx0, cur, inpL);
        cur = lctx.cvec.apply_to(ctx0, cur, il);
        cb(cur, "l_out", il);

        // input for next layer
        inpL = cur;
    }

    cur = inpL;

    // norm
    cur = llm_build_norm(ctx0, cur, hparams, model.output_norm, model.output_norm_b, LLM_NORM, cb, -1);
    cb(cur, "result_norm", -1);

    cur = llm_build_lora_mm(lctx, ctx0, model.output, cur);
    cb(cur, "result_output", -1);

    ggml_build_forward_expand(gf, cur);

    return gf;
}

ggml_cgraph * llm_build_context::build_grok() {
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, llama_model_max_nodes(model), false);

    // mutable variable, needed during the last layer of the computation to skip unused tokens
    int32_t n_tokens = this->n_tokens;

    const int64_t n_embd_head = hparams.n_embd_head_v;
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);
    GGML_ASSERT(n_embd_head == hparams.n_rot);

    struct ggml_tensor * cur;
    struct ggml_tensor * inpL;

    inpL = llm_build_inp_embd(ctx0, lctx, hparams, batch, model.tok_embd, cb);

    // inp_pos - contains the positions
    struct ggml_tensor * inp_pos = build_inp_pos();

    // KQ_mask (mask for 1 head, it will be broadcasted to all heads)
    struct ggml_tensor * KQ_mask = build_inp_KQ_mask();
    for (int il = 0; il < n_layer; ++il) {
        struct ggml_tensor * inpSA = inpL;

        // norm
        cur = llm_build_norm(ctx0, inpL, hparams, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, cb, il);
        cb(cur, "attn_norm", il);


        // self-attention
        {
            auto [Qcur, Kcur, Vcur] = llm_build_mul_mat_qkv(gf, cur, model.layers[il].wq, model.layers[il].bq,
                    model.layers[il].wk, model.layers[il].bk,
                    model.layers[il].wv, model.layers[il].bv, 0.f, il);

            Qcur = ggml_rope_ext(
                    ctx0, ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens), inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );
            cb(Qcur, "Qcur", il);

            Kcur = ggml_rope_ext(
                    ctx0, ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens), inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );
            cb(Kcur, "Kcur", il);

            cur = llm_build_kv(ctx0, lctx, kv_self, gf,
                    model.layers[il].wo, model.layers[il].bo,
                    Kcur, Vcur, Qcur, KQ_mask, n_tokens, kv_head, n_kv, 1.0f, cb, il);
        }

        if (il == n_layer - 1) {
            // skip computing output for unused tokens
            struct ggml_tensor * inp_out_ids = build_inp_out_ids();
            n_tokens = n_outputs;
            cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }

        cur = llm_build_norm(ctx0, cur, hparams, model.layers[il].attn_out_norm, NULL, LLM_NORM_RMS, cb, il);
        cb(cur, "attn_out_norm", il);


        struct ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        // feed-forward network
        cur = llm_build_norm(ctx0, ffn_inp, hparams, model.layers[il].ffn_norm, NULL, LLM_NORM_RMS, cb, il);
        cb(cur, "ffn_norm", il);

        // MoE branch
        ggml_tensor* moe_out = llm_build_moe_ffn(ctx0, lctx, cur,
                model.layers[il].ffn_gate_inp,
                model.layers[il].ffn_up_exps,
                model.layers[il].ffn_gate_exps,
                model.layers[il].ffn_down_exps,
                nullptr,
                n_expert, n_expert_used,
                LLM_FFN_GELU, true,
                false, 0.0,
                LLM_EXPERT_GATING_FUNC_SOFTMAX,
                cb, il, gf);
        cb(moe_out, "ffn_moe_out", il);

        if (model.layers[il].ffn_up) {
            ggml_tensor* ffn_out = llm_build_ffn(ctx0, lctx, nullptr, cur,
                    model.layers[il].ffn_up, NULL, NULL,
                    model.layers[il].ffn_gate, NULL, NULL,
                    model.layers[il].ffn_down, NULL, NULL,
                    NULL,
                    LLM_FFN_GELU, LLM_FFN_PAR, cb, il);
            cb(ffn_out, "ffn_out", il);

            cur = ggml_scale(ctx0, ggml_add(ctx0, ffn_out, moe_out), std::sqrt(2) / 2);
            cb(cur, "ffn_out", il);
        }
        else {
            cur = moe_out;
        }

        cur = llm_build_norm(ctx0, cur, hparams, model.layers[il].ffn_post_norm, NULL, LLM_NORM_RMS, cb, il);
        cb(cur, "ffn_post_norm", il);


        cur = ggml_add(ctx0, cur, ffn_inp);
        cb(cur, "ffn_out", il);

        cur = lctx.cvec.apply_to(ctx0, cur, il);
        cb(cur, "l_out", il);

        // input for next layer
        inpL = cur;
    }

    cur = inpL;

    cur = llm_build_norm(ctx0, cur, hparams, model.output_norm, NULL, LLM_NORM_RMS, cb, -1);
    cb(cur, "result_norm", -1);

    // lm_head
    cur = llm_build_lora_mm(lctx, ctx0, model.output, cur);

    cur = ggml_scale(ctx0, cur, hparams.f_logit_scale);
    // final logit soft-capping
    if (hparams.f_final_logit_softcapping) {
        /*cur = ggml_scale(ctx0, cur, 1.0f / hparams.f_final_logit_softcapping);
          cur = ggml_tanh(ctx0, cur);
          cur = ggml_scale(ctx0, cur, hparams.f_final_logit_softcapping);*/
        cur = ggml_softcap(ctx0, cur, 1.0f / hparams.f_final_logit_softcapping, hparams.f_final_logit_softcapping);

    }
    cb(cur, "result_output", -1);

    ggml_build_forward_expand(gf, cur);

    return gf;
}

ggml_cgraph * llm_build_context::build_dbrx() {
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, llama_model_max_nodes(model), false);

    // mutable variable, needed during the last layer of the computation to skip unused tokens
    int32_t n_tokens = this->n_tokens;

    const int64_t n_embd_head = hparams.n_embd_head_v;
    const int64_t n_embd_gqa  = hparams.n_embd_v_gqa();
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);
    GGML_ASSERT(n_embd_head == hparams.n_rot);

    struct ggml_tensor * cur;
    struct ggml_tensor * inpL;

    inpL = llm_build_inp_embd(ctx0, lctx, hparams, batch, model.tok_embd, cb);

    // inp_pos - contains the positions
    struct ggml_tensor * inp_pos = build_inp_pos();

    // KQ_mask (mask for 1 head, it will be broadcasted to all heads)
    struct ggml_tensor * KQ_mask = build_inp_KQ_mask();

    for (int il = 0; il < n_layer; ++il) {
        struct ggml_tensor * inpSA = inpL;

        // norm
        cur = llm_build_norm(ctx0, inpL, hparams, model.layers[il].attn_norm, NULL, LLM_NORM, cb, il);
        cb(cur, "attn_norm", il);

        // self-attention
        {
            struct ggml_tensor * Qcur = nullptr;
            struct ggml_tensor * Kcur = nullptr;
            struct ggml_tensor * Vcur = nullptr;

            cur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wqkv, cur);
            cb(cur, "wqkv", il);

            cur = ggml_clamp(ctx0, cur, -hparams.f_clamp_kqv, hparams.f_clamp_kqv);
            cb(cur, "wqkv_clamped", il);

            Qcur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd,     n_tokens, cur->nb[1], 0*sizeof(float)*(n_embd)));
            Kcur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd_gqa, n_tokens, cur->nb[1], 1*sizeof(float)*(n_embd)));
            Vcur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd_gqa, n_tokens, cur->nb[1], 1*sizeof(float)*(n_embd + n_embd_gqa)));

            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            Qcur = ggml_rope_ext(
                    ctx0, ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens), inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );
            cb(Qcur, "Qcur", il);

            Kcur = ggml_rope_ext(
                    ctx0, ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens), inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );
            cb(Kcur, "Kcur", il);

            cur = llm_build_kv(ctx0, lctx, kv_self, gf,
                    model.layers[il].wo, NULL,
                    Kcur, Vcur, Qcur, KQ_mask, n_tokens, kv_head, n_kv, 1.0f/sqrtf(float(n_embd_head)), cb, il);
        }

        if (il == n_layer - 1) {
            // skip computing output for unused tokens
            struct ggml_tensor * inp_out_ids = build_inp_out_ids();
            n_tokens = n_outputs;
            cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }

        struct ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        // feed-forward network
        // MoE branch
        cur = llm_build_norm(ctx0, ffn_inp, hparams, model.layers[il].attn_out_norm, NULL, LLM_NORM, cb, il);
        cb(cur, "attn_out_norm", il);

        cur = llm_build_moe_ffn(ctx0, lctx, cur,
                model.layers[il].ffn_gate_inp,
                model.layers[il].ffn_up_exps,
                model.layers[il].ffn_gate_exps,
                model.layers[il].ffn_down_exps,
                nullptr,
                n_expert, n_expert_used,
                LLM_FFN_SILU, true,
                false, 0.0,
                LLM_EXPERT_GATING_FUNC_SOFTMAX,
                cb, il, gf);
        cb(cur, "ffn_moe_out", il);

        cur = ggml_add(ctx0, cur, ffn_inp);
        cb(cur, "ffn_out", il);

        cur = lctx.cvec.apply_to(ctx0, cur, il);
        cb(cur, "l_out", il);

        // input for next layer
        inpL = cur;
    }

    cur = inpL;

    cur = llm_build_norm(ctx0, cur, hparams, model.output_norm, NULL, LLM_NORM, cb, -1);
    cb(cur, "result_norm", -1);

    // lm_head
    cur = llm_build_lora_mm(lctx, ctx0, model.output, cur);

    cb(cur, "result_output", -1);

    ggml_build_forward_expand(gf, cur);

    return gf;
}

ggml_cgraph * llm_build_context::build_starcoder() {
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, llama_model_max_nodes(model), false);

    const int64_t n_embd_head = hparams.n_embd_head_v;
    const int64_t n_embd_gqa  = hparams.n_embd_v_gqa();
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);

    struct ggml_tensor * cur;
    struct ggml_tensor * inpL;

    inpL = llm_build_inp_embd(ctx0, lctx, hparams, batch, model.tok_embd, cb);

    // inp_pos - contains the positions
    struct ggml_tensor * inp_pos = build_inp_pos();

    // KQ_mask (mask for 1 head, it will be broadcasted to all heads)
    struct ggml_tensor * KQ_mask = build_inp_KQ_mask();

    struct ggml_tensor * pos = ggml_get_rows(ctx0, model.pos_embd, inp_pos);
    cb(pos, "pos_embd", -1);

    inpL = ggml_add(ctx0, inpL, pos);
    cb(inpL, "inpL", -1);

    for (int il = 0; il < n_layer; ++il) {
        cur = llm_build_norm(ctx0, inpL, hparams, model.layers[il].attn_norm, model.layers[il].attn_norm_b, LLM_NORM, cb, il);
        cb(cur, "attn_norm", il);

        // self-attention
        {
            cur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wqkv, cur);
            cb(cur, "wqkv", il);

            cur = ggml_add(ctx0, cur, model.layers[il].bqkv);
            cb(cur, "bqkv", il);

            struct ggml_tensor * Qcur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd,     n_tokens, cur->nb[1], 0*sizeof(float)*(n_embd)));
            struct ggml_tensor * Kcur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd_gqa, n_tokens, cur->nb[1], 1*sizeof(float)*(n_embd)));
            struct ggml_tensor * Vcur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd_gqa, n_tokens, cur->nb[1], 1*sizeof(float)*(n_embd + n_embd_gqa)));

            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens);

            cur = llm_build_kv(ctx0, lctx, kv_self, gf,
                    model.layers[il].wo, model.layers[il].bo,
                    Kcur, Vcur, Qcur, KQ_mask, n_tokens, kv_head, n_kv, 1.0f/sqrtf(float(n_embd_head)), cb, il);
        }

        if (il == n_layer - 1) {
            // skip computing output for unused tokens
            struct ggml_tensor * inp_out_ids = build_inp_out_ids();
            cur  = ggml_get_rows(ctx0,  cur, inp_out_ids);
            inpL = ggml_get_rows(ctx0, inpL, inp_out_ids);
        }

        // add the input
        struct ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpL);
        cb(ffn_inp, "ffn_inp", il);

        // FF
        {
            cur = llm_build_ffn(ctx0, lctx, model.layers[il].ffn_norm, ffn_inp,
                    model.layers[il].ffn_up,   model.layers[il].ffn_up_b,   NULL,
                    NULL,                      NULL,                        NULL,
                    model.layers[il].ffn_down, model.layers[il].ffn_down_b, NULL,
                    NULL,
                    LLM_FFN_GELU, LLM_FFN_SEQ, cb, il);
            cb(cur, "ffn_out", il);
        }

        cur = ggml_add(ctx0, cur, ffn_inp);
        cur = lctx.cvec.apply_to(ctx0, cur, il);
        cb(cur, "l_out", il);

        // input for next layer
        inpL = cur;
    }

    cur = llm_build_norm(ctx0, inpL, hparams, model.output_norm, model.output_norm_b, LLM_NORM, cb, -1);
    cb(cur, "result_norm", -1);

    cur = llm_build_lora_mm(lctx, ctx0, model.output, cur);
    cb(cur, "result_output", -1);

    ggml_build_forward_expand(gf, cur);

    return gf;
}

ggml_cgraph * llm_build_context::build_refact() {
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, llama_model_max_nodes(model), false);

    const int64_t n_embd_head = hparams.n_embd_head_v;
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);

    struct ggml_tensor * cur;
    struct ggml_tensor * inpL;

    inpL = llm_build_inp_embd(ctx0, lctx, hparams, batch, model.tok_embd, cb);

    // KQ_mask (mask for 1 head, it will be broadcasted to all heads)
    struct ggml_tensor * KQ_mask = build_inp_KQ_mask();

    for (int il = 0; il < n_layer; ++il) {
        struct ggml_tensor * inpSA = inpL;

        cur = llm_build_norm(ctx0, inpL, hparams, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, cb, il);
        cb(cur, "attn_norm", il);

        // self-attention
        {
            auto [Qcur, Kcur, Vcur] = llm_build_mul_mat_qkv(gf, cur, model.layers[il].wq, nullptr,
                    model.layers[il].wk, nullptr,
                    model.layers[il].wv, nullptr, 0, il);

            Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
            cb(Kcur, "Kcur", il);

            Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
            cb(Qcur, "Qcur", il);

            cur = llm_build_kv(ctx0, lctx, kv_self, gf,
                    model.layers[il].wo, NULL,
                    Kcur, Vcur, Qcur, KQ_mask, n_tokens, kv_head, n_kv, 1.0f/sqrtf(float(n_embd_head)), cb, il);
        }

        if (il == n_layer - 1) {
            // skip computing output for unused tokens
            struct ggml_tensor * inp_out_ids = build_inp_out_ids();
            cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }

        struct ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        // feed-forward network
        {
            cur = llm_build_ffn(ctx0, lctx, model.layers[il].ffn_norm, ffn_inp,
                    model.layers[il].ffn_up,   NULL, NULL,
                    model.layers[il].ffn_gate, NULL, NULL,
                    model.layers[il].ffn_down, NULL, NULL,
                    NULL,
                    LLM_FFN_SILU, LLM_FFN_PAR, cb, il);
            cb(cur, "ffn_out", il);
        }

        cur = ggml_add(ctx0, cur, ffn_inp);
        cur = lctx.cvec.apply_to(ctx0, cur, il);
        cb(cur, "l_out", il);

        // input for next layer
        inpL = cur;
    }

    cur = inpL;

    cur = llm_build_norm(ctx0, cur, hparams, model.output_norm, NULL, LLM_NORM_RMS, cb, -1);
    cb(cur, "result_norm", -1);

    // lm_head
    cur = llm_build_lora_mm(lctx, ctx0, model.output, cur);
    cb(cur, "result_output", -1);

    ggml_build_forward_expand(gf, cur);

    return gf;
}

ggml_cgraph * llm_build_context::build_bert() {
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, llama_model_max_nodes(model), false);

    const int64_t n_embd_head = hparams.n_embd_head_v;
    const int64_t n_embd_gqa  = hparams.n_embd_v_gqa();

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);

    struct ggml_tensor * cur;
    struct ggml_tensor * inpL;
    struct ggml_tensor * inp_pos = nullptr;

    if (model.arch != LLM_ARCH_JINA_BERT_V2) {
        inp_pos = build_inp_pos();
    }

    // construct input embeddings (token, type, position)
    inpL = llm_build_inp_embd(ctx0, lctx, hparams, batch, model.tok_embd, cb);

    // token types are hardcoded to zero ("Sentence A")
    struct ggml_tensor * type_row0 = ggml_view_1d(ctx0, model.type_embd, n_embd, 0);
    inpL = ggml_add(ctx0, inpL, type_row0);
    if (model.arch == LLM_ARCH_BERT) {
        inpL = ggml_add(ctx0, ggml_get_rows(ctx0, model.pos_embd, inp_pos), inpL);
    }
    cb(inpL, "inp_embd", -1);

    // embed layer norm
    inpL = llm_build_norm(ctx0, inpL, hparams, model.tok_norm, model.tok_norm_b, LLM_NORM, cb, -1);
    cb(inpL, "inp_norm", -1);

    // KQ_mask (mask for 1 head, it will be broadcasted to all heads)
    struct ggml_tensor * KQ_mask = build_inp_KQ_mask(false);

    // iterate layers
    for (int il = 0; il < n_layer; ++il) {
        struct ggml_tensor * cur = inpL;

        struct ggml_tensor * Qcur;
        struct ggml_tensor * Kcur;
        struct ggml_tensor * Vcur;

        // self-attention
        if (model.arch == LLM_ARCH_BERT || model.arch == LLM_ARCH_JINA_BERT_V2) {
            Qcur = ggml_add(ctx0, llm_build_lora_mm(lctx, ctx0, model.layers[il].wq, cur), model.layers[il].bq);
            cb(Qcur, "Qcur", il);

            if (model.layers[il].attn_q_norm) {
                Qcur = llm_build_norm(ctx0, Qcur, hparams, model.layers[il].attn_q_norm, model.layers[il].attn_q_norm_b, LLM_NORM, cb, il);
            }

            Kcur = ggml_add(ctx0, llm_build_lora_mm(lctx, ctx0, model.layers[il].wk, cur), model.layers[il].bk);
            cb(Kcur, "Kcur", il);

            if (model.layers[il].attn_k_norm) {
                Kcur = llm_build_norm(ctx0, Kcur, hparams, model.layers[il].attn_k_norm, model.layers[il].attn_k_norm_b, LLM_NORM, cb, il);
            }
            Vcur = ggml_add(ctx0, llm_build_lora_mm(lctx, ctx0, model.layers[il].wv, cur), model.layers[il].bv);
            cb(Vcur, "Vcur", il);

            Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
            Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
        } else {
            // compute Q and K and RoPE them
            cur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wqkv, cur);
            cb(cur, "wqkv", il);

            Qcur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd,     n_tokens, cur->nb[1], 0*sizeof(float)*(n_embd)));
            Kcur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd_gqa, n_tokens, cur->nb[1], 1*sizeof(float)*(n_embd)));
            Vcur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd_gqa, n_tokens, cur->nb[1], 1*sizeof(float)*(n_embd + n_embd_gqa)));

            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            Qcur = ggml_rope_ext(
                    ctx0, ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens), inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );
            cb(Qcur, "Qcur", il);

            Kcur = ggml_rope_ext(
                    ctx0, ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens), inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );
            cb(Kcur, "Kcur", il);
        }

        struct ggml_tensor * q =                 ggml_permute(ctx0, Qcur, 0, 2, 1, 3);
        struct ggml_tensor * k = ggml_cont(ctx0, ggml_permute(ctx0, Kcur, 0, 2, 1, 3));

        struct ggml_tensor * kq = ggml_mul_mat(ctx0, k, q);
        cb(kq, "kq", il);

        kq = ggml_soft_max_ext(ctx0, kq, KQ_mask, 1.0f/sqrtf(float(n_embd_head)), hparams.f_max_alibi_bias);
        cb(kq, "kq_soft_max_ext", il);

        struct ggml_tensor * v = ggml_cont(ctx0, ggml_transpose(ctx0, ggml_reshape_2d(ctx0, Vcur, n_embd_gqa, n_tokens)));
        cb(v, "v", il);

        struct ggml_tensor * kqv = ggml_mul_mat(ctx0, ggml_reshape_3d(ctx0, v, n_tokens, n_embd_head, n_head_kv), kq);
        cb(kqv, "kqv", il);

        struct ggml_tensor * kqv_merged = ggml_permute(ctx0, kqv, 0, 2, 1, 3);
        cb(kqv_merged, "kqv_merged", il);

        cur = ggml_cont_2d(ctx0, kqv_merged, n_embd_gqa, n_tokens);
        cb(cur, "kqv_merged_cont", il);

        ggml_build_forward_expand(gf, cur);

        cur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wo, cur);
        if (model.layers[il].bo) {
            cb(cur, "kqv_wo", il);
        }

        if (model.layers[il].bo) {
            cur = ggml_add(ctx0, cur, model.layers[il].bo);
        }
        cb(cur, "kqv_out", il);

        if (il == n_layer - 1 && pooling_type == LLAMA_POOLING_TYPE_NONE) {
            // skip computing output for unused tokens
            struct ggml_tensor * inp_out_ids = build_inp_out_ids();
            cur  = ggml_get_rows(ctx0,  cur, inp_out_ids);
            inpL = ggml_get_rows(ctx0, inpL, inp_out_ids);
        }

        // re-add the layer input
        cur = ggml_add(ctx0, cur, inpL);

        // attention layer norm
        cur = llm_build_norm(ctx0, cur, hparams, model.layers[il].attn_out_norm, model.layers[il].attn_out_norm_b, LLM_NORM, cb, il);

        if (model.layers[il].attn_norm_2 != nullptr) {
            cur = ggml_add(ctx0, cur, inpL); // re-add the layer input
            cur = llm_build_norm(ctx0, cur, hparams, model.layers[il].attn_norm_2, model.layers[il].attn_norm_2_b, LLM_NORM, cb, il);
        }

        struct ggml_tensor * ffn_inp = cur;
        cb(ffn_inp, "ffn_inp", il);

        // feed-forward network
        if (model.arch == LLM_ARCH_BERT) {
            cur = llm_build_ffn(ctx0, lctx, nullptr, cur,
                    model.layers[il].ffn_up,   model.layers[il].ffn_up_b,   NULL,
                    NULL,                      NULL,                        NULL,
                    model.layers[il].ffn_down, model.layers[il].ffn_down_b, NULL,
                    NULL,
                    LLM_FFN_GELU, LLM_FFN_SEQ, cb, il);
        } else if (model.arch == LLM_ARCH_JINA_BERT_V2) {
            cur = llm_build_ffn(ctx0, lctx, nullptr, cur,
                    model.layers[il].ffn_up,   NULL,                        NULL,
                    model.layers[il].ffn_gate, NULL,                        NULL,
                    model.layers[il].ffn_down, model.layers[il].ffn_down_b, NULL,
                    NULL,
                    LLM_FFN_GELU, LLM_FFN_PAR, cb, il);
        } else {
            cur = llm_build_ffn(ctx0, lctx, nullptr, cur,
                    model.layers[il].ffn_up,   NULL, NULL,
                    model.layers[il].ffn_gate, NULL, NULL,
                    model.layers[il].ffn_down, NULL, NULL,
                    NULL,
                    LLM_FFN_SILU, LLM_FFN_PAR, cb, il);
        }
        cb(cur, "ffn_out", il);

        // attentions bypass the intermediate layer
        cur = ggml_add(ctx0, cur, ffn_inp);

        // output layer norm
        cur = llm_build_norm(ctx0, cur, hparams, model.layers[il].layer_out_norm, model.layers[il].layer_out_norm_b, LLM_NORM, cb, il);

        // input for next layer
        inpL = cur;
    }

    // final output
    cur = inpL;
    cb(cur, "result_embd", -1);

    ggml_build_forward_expand(gf, cur);

    return gf;
}

ggml_cgraph * llm_build_context::build_bloom() {
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, llama_model_max_nodes(model), false);

    const int64_t n_embd_head = hparams.n_embd_head_v;
    const int64_t n_embd_gqa  = hparams.n_embd_v_gqa();
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);

    struct ggml_tensor * cur;
    struct ggml_tensor * inpL;

    inpL = llm_build_inp_embd(ctx0, lctx, hparams, batch, model.tok_embd, cb);

    // KQ_mask (mask for 1 head, it will be broadcasted to all heads)
    struct ggml_tensor * KQ_mask = build_inp_KQ_mask();

    inpL = llm_build_norm(ctx0, inpL, hparams, model.tok_norm, model.tok_norm_b, LLM_NORM, cb, -1);
    cb(inpL, "inp_norm", -1);

    for (int il = 0; il < n_layer; ++il) {
        cur = llm_build_norm(ctx0, inpL, hparams, model.layers[il].attn_norm, model.layers[il].attn_norm_b, LLM_NORM, cb, il);
        cb(cur, "attn_norm", il);

        // self-attention
        {
            cur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wqkv, cur);
            cb(cur, "wqkv", il);

            cur = ggml_add(ctx0, cur, model.layers[il].bqkv);
            cb(cur, "bqkv", il);

            struct ggml_tensor * Qcur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd,     n_tokens, cur->nb[1], 0*sizeof(float)*(n_embd)));
            struct ggml_tensor * Kcur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd_gqa, n_tokens, cur->nb[1], 1*sizeof(float)*(n_embd)));
            struct ggml_tensor * Vcur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd_gqa, n_tokens, cur->nb[1], 1*sizeof(float)*(n_embd + n_embd_gqa)));

            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens);

            cur = llm_build_kv(ctx0, lctx, kv_self, gf,
                    model.layers[il].wo, model.layers[il].bo,
                    Kcur, Vcur, Qcur, KQ_mask, n_tokens, kv_head, n_kv, 1.0f/sqrtf(float(n_embd_head)), cb, il);
        }

        if (il == n_layer - 1) {
            // skip computing output for unused tokens
            struct ggml_tensor * inp_out_ids = build_inp_out_ids();
            cur  = ggml_get_rows(ctx0,  cur, inp_out_ids);
            inpL = ggml_get_rows(ctx0, inpL, inp_out_ids);
        }

        // Add the input
        struct ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpL);
        cb(ffn_inp, "ffn_inp", il);

        // FF
        {
            cur = llm_build_ffn(ctx0, lctx, model.layers[il].ffn_norm, ffn_inp,
                    model.layers[il].ffn_up,   model.layers[il].ffn_up_b,   NULL,
                    NULL,                      NULL,                        NULL,
                    model.layers[il].ffn_down, model.layers[il].ffn_down_b, NULL,
                    NULL,
                    LLM_FFN_GELU, LLM_FFN_SEQ, cb, il);
            cb(cur, "ffn_out", il);
        }

        cur = ggml_add(ctx0, cur, ffn_inp);
        cur = lctx.cvec.apply_to(ctx0, cur, il);
        cb(cur, "l_out", il);

        // input for next layer
        inpL = cur;
    }

    cur = llm_build_norm(ctx0, inpL, hparams, model.output_norm, model.output_norm_b, LLM_NORM, cb, -1);
    cb(cur, "result_norm", -1);

    cur = llm_build_lora_mm(lctx, ctx0, model.output, cur);
    cb(cur, "result_output", -1);

    ggml_build_forward_expand(gf, cur);

    return gf;
}

ggml_cgraph * llm_build_context::build_mpt() {
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, llama_model_max_nodes(model), false);

    const int64_t n_embd_head = hparams.n_embd_head_v;
    const int64_t n_embd_gqa  = hparams.n_embd_v_gqa();
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);

    struct ggml_tensor * cur;
    struct ggml_tensor * pos;
    struct ggml_tensor * inpL;

    inpL = llm_build_inp_embd(ctx0, lctx, hparams, batch, model.tok_embd, cb);

    // KQ_mask (mask for 1 head, it will be broadcasted to all heads)
    struct ggml_tensor * KQ_mask = build_inp_KQ_mask();

    if (model.pos_embd) {
        // inp_pos - contains the positions
        struct ggml_tensor * inp_pos = build_inp_pos();
        pos = ggml_get_rows(ctx0, model.pos_embd, inp_pos);
        cb(pos, "pos_embd", -1);

        inpL = ggml_add(ctx0, inpL, pos);
        cb(inpL, "inpL", -1);
    }

    for (int il = 0; il < n_layer; ++il) {
        struct ggml_tensor * attn_norm;

        attn_norm = llm_build_norm(ctx0, inpL, hparams, model.layers[il].attn_norm, model.layers[il].attn_norm_b, LLM_NORM, cb, il);
        cb(attn_norm, "attn_norm", il);

        // self-attention
        {
            cur = attn_norm;

            cur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wqkv, cur);
            cb(cur, "wqkv", il);

            if (model.layers[il].bqkv){
                cur = ggml_add(ctx0, cur, model.layers[il].bqkv);
                cb(cur, "bqkv", il);
            }

            if (hparams.f_clamp_kqv > 0.0f) {
                cur = ggml_clamp(ctx0, cur, -hparams.f_clamp_kqv, hparams.f_clamp_kqv);
                cb(cur, "wqkv_clamped", il);
            }

            struct ggml_tensor * Qcur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd,     n_tokens, cur->nb[1], 0*sizeof(float)*(n_embd)));
            struct ggml_tensor * Kcur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd_gqa, n_tokens, cur->nb[1], 1*sizeof(float)*(n_embd)));
            struct ggml_tensor * Vcur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd_gqa, n_tokens, cur->nb[1], 1*sizeof(float)*(n_embd + n_embd_gqa)));

            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            // Q/K Layernorm
            if (model.layers[il].attn_q_norm) {
                Qcur = llm_build_norm(ctx0, Qcur, hparams, model.layers[il].attn_q_norm, model.layers[il].attn_q_norm_b, LLM_NORM, cb, il);
                cb(Qcur, "Qcur", il);

                Kcur = llm_build_norm(ctx0, Kcur, hparams, model.layers[il].attn_k_norm, model.layers[il].attn_k_norm_b, LLM_NORM, cb, il);
                cb(Kcur, "Kcur", il);

                Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
                Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);

                cur = llm_build_kv(ctx0, lctx, kv_self, gf,
                        model.layers[il].wo, model.layers[il].bo,
                        Kcur, Vcur, Qcur, KQ_mask, n_tokens, kv_head, n_kv, 1.0f/sqrtf(float(n_embd_head)), cb, il);
            } else {
                Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens);

                cur = llm_build_kv(ctx0, lctx, kv_self, gf,
                        model.layers[il].wo, model.layers[il].bo,
                        Kcur, Vcur, Qcur, KQ_mask, n_tokens, kv_head, n_kv, 1.0f/sqrtf(float(n_embd_head)), cb, il);
            }
        }

        if (il == n_layer - 1) {
            // skip computing output for unused tokens
            struct ggml_tensor * inp_out_ids = build_inp_out_ids();
            cur  = ggml_get_rows(ctx0,  cur, inp_out_ids);
            inpL = ggml_get_rows(ctx0, inpL, inp_out_ids);
        }

        // Add the input
        struct ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpL);
        cb(ffn_inp, "ffn_inp", il);

        // feed forward
        {
            cur = llm_build_ffn(ctx0, lctx, model.layers[il].ffn_norm, ffn_inp,
                    model.layers[il].ffn_up,   model.layers[il].ffn_up_b,   NULL,
                    NULL,                      NULL,                        NULL,
                    model.layers[il].ffn_down, model.layers[il].ffn_down_b, NULL,
                    model.layers[il].ffn_act,
                    LLM_FFN_GELU, LLM_FFN_SEQ, cb, il);
            cb(cur, "ffn_out", il);
        }

        cur = ggml_add(ctx0, cur, ffn_inp);
        cur = lctx.cvec.apply_to(ctx0, cur, il);
        cb(cur, "l_out", il);

        // input for next layer
        inpL = cur;
    }

    cur = inpL;

    cur = llm_build_norm(ctx0, cur, hparams, model.output_norm, model.output_norm_b, LLM_NORM, cb, -1);
    cb(cur, "result_norm", -1);

    cur = llm_build_lora_mm(lctx, ctx0, model.output, cur);
    cb(cur, "result_output", -1);

    ggml_build_forward_expand(gf, cur);

    return gf;
}

ggml_cgraph * llm_build_context::build_stablelm() {
    struct ggml_cgraph * gf = ggml_new_graph(ctx0);

    const int64_t n_embd_head = hparams.n_embd_head_v;
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);

    struct ggml_tensor * cur;
    struct ggml_tensor * inpL;

    inpL = llm_build_inp_embd(ctx0, lctx, hparams, batch, model.tok_embd, cb);

    // inp_pos - contains the positions
    struct ggml_tensor * inp_pos = build_inp_pos();

    // KQ_mask (mask for 1 head, it will be broadcasted to all heads)
    struct ggml_tensor * KQ_mask = build_inp_KQ_mask();

    for (int il = 0; il < n_layer; ++il) {


        // norm
        cur = llm_build_norm(ctx0, inpL, hparams, model.layers[il].attn_norm, model.layers[il].attn_norm_b, LLM_NORM, cb, il);
        cb(cur, "attn_norm", il);

        struct ggml_tensor * inpSA = cur;

        // self-attention
        {
            auto [Qcur, Kcur, Vcur] = llm_build_mul_mat_qkv(gf, cur, model.layers[il].wq, model.layers[il].bq,
                    model.layers[il].wk, model.layers[il].bk,
                    model.layers[il].wv, model.layers[il].bv, 0.f, il);

            Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
            cb(Qcur, "Qcur", il);
            Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
            cb(Kcur, "Kcur", il);

            if (model.layers[il].attn_q_norm) {
                Qcur = llm_build_norm(ctx0, Qcur, hparams, model.layers[il].attn_q_norm, NULL, LLM_NORM, cb, il);
                cb(Qcur, "Qcur", il);
            }
            if (model.layers[il].attn_k_norm) {
                Kcur = llm_build_norm(ctx0, Kcur, hparams, model.layers[il].attn_k_norm, NULL, LLM_NORM, cb, il);
                cb(Kcur, "Kcur", il);
            }


            Qcur = ggml_rope_ext(
                    ctx0, Qcur, inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );
            cb(Qcur, "Qcur", il);

            Kcur = ggml_rope_ext(
                    ctx0, Kcur, inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );
            cb(Kcur, "Kcur", il);

            cur = llm_build_kv(ctx0, lctx, kv_self, gf,
                    model.layers[il].wo, NULL,
                    Kcur, Vcur, Qcur, KQ_mask, n_tokens, kv_head, n_kv, 1.0f/sqrtf(float(n_embd_head)), cb, il);
        }

        if (il == n_layer - 1) {
            // skip computing output for unused tokens
            struct ggml_tensor * inp_out_ids = build_inp_out_ids();
            cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
            inpL  = ggml_get_rows(ctx0,  inpL, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }

        struct ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpL);
        cb(ffn_inp, "ffn_inp", il);

        // feed-forward network
        {
            if (model.layers[il].ffn_norm) {
                cur = llm_build_norm(ctx0, ffn_inp, hparams, model.layers[il].ffn_norm, model.layers[il].ffn_norm_b, LLM_NORM, cb, il);
                cb(cur, "ffn_norm", il);
            } else {
                // parallel residual
                cur = inpSA;
            }
            cur = llm_build_ffn(ctx0, lctx, nullptr, cur,
                    model.layers[il].ffn_up,   NULL, NULL,
                    model.layers[il].ffn_gate, NULL, NULL,
                    model.layers[il].ffn_down, NULL, NULL,
                    NULL,
                    LLM_FFN_SILU, LLM_FFN_PAR, cb, il);
            cb(cur, "ffn_out", il);
        }

        cur = ggml_add(ctx0, cur, ffn_inp);
        cur = lctx.cvec.apply_to(ctx0, cur, il);
        cb(cur, "l_out", il);

        // input for next layer
        inpL = cur;
    }

    cur = inpL;

    cur = llm_build_norm(ctx0, cur, hparams, model.output_norm, model.output_norm_b, LLM_NORM, cb, -1);
    cb(cur, "result_norm", -1);

    // lm_head
    cur = llm_build_lora_mm(lctx, ctx0, model.output, cur);
    cb(cur, "result_output", -1);

    ggml_build_forward_expand(gf, cur);

    return gf;
}

ggml_cgraph * llm_build_context::build_qwen() {
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, llama_model_max_nodes(model), false);

    const int64_t n_embd_head = hparams.n_embd_head_v;
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);

    struct ggml_tensor * cur;
    struct ggml_tensor * inpL;

    inpL = llm_build_inp_embd(ctx0, lctx, hparams, batch, model.tok_embd, cb);

    // inp_pos - contains the positions
    struct ggml_tensor * inp_pos = build_inp_pos();

    // KQ_mask (mask for 1 head, it will be broadcasted to all heads)
    struct ggml_tensor * KQ_mask = build_inp_KQ_mask();

    for (int il = 0; il < n_layer; ++il) {
        struct ggml_tensor * inpSA = inpL;

        cur = llm_build_norm(ctx0, inpL, hparams, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, cb, il);
        cb(cur, "attn_norm", il);

        // self-attention
        {
            cur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wqkv, cur);
            cb(cur, "wqkv", il);

            cur = ggml_add(ctx0, cur, model.layers[il].bqkv);
            cb(cur, "bqkv", il);

            struct ggml_tensor * Qcur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd, n_tokens, cur->nb[1], 0*sizeof(float)*(n_embd)));
            struct ggml_tensor * Kcur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd, n_tokens, cur->nb[1], 1*sizeof(float)*(n_embd)));
            struct ggml_tensor * Vcur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd, n_tokens, cur->nb[1], 2*sizeof(float)*(n_embd)));

            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
            Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);

            // using mode = 2 for neox mode
            Qcur = ggml_rope_ext(
                    ctx0, Qcur, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig,
                    freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow
                    );
            cb(Qcur, "Qcur", il);

            Kcur = ggml_rope_ext(
                    ctx0, Kcur, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig,
                    freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow
                    );
            cb(Kcur, "Kcur", il);

            cur = llm_build_kv(ctx0, lctx, kv_self, gf,
                    model.layers[il].wo, NULL,
                    Kcur, Vcur, Qcur, KQ_mask, n_tokens, kv_head, n_kv, 1.0f/sqrtf(float(n_embd_head)), cb, il);
        }

        if (il == n_layer - 1) {
            // skip computing output for unused tokens
            struct ggml_tensor * inp_out_ids = build_inp_out_ids();
            cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }

        struct ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        // feed-forward forward
        {
            cur = llm_build_ffn(ctx0, lctx, model.layers[il].ffn_norm, ffn_inp,
                    model.layers[il].ffn_up,   NULL, NULL,
                    model.layers[il].ffn_gate, NULL, NULL,
                    model.layers[il].ffn_down, NULL, NULL,
                    NULL,
                    LLM_FFN_SILU, LLM_FFN_PAR, cb, il);
            cb(cur, "ffn_out", il);
        }

        cur = ggml_add(ctx0, cur, ffn_inp);
        cur = lctx.cvec.apply_to(ctx0, cur, il);
        cb(cur, "l_out", il);

        // input for next layer
        inpL = cur;
    }

    cur = inpL;

    cur = llm_build_norm(ctx0, cur, hparams, model.output_norm, NULL, LLM_NORM_RMS, cb, -1);
    cb(cur, "result_norm", -1);

    // lm_head
    cur = llm_build_lora_mm(lctx, ctx0, model.output, cur);
    cb(cur, "result_output", -1);

    ggml_build_forward_expand(gf, cur);

    return gf;
}

ggml_cgraph * llm_build_context::build_qwen2() {
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, llama_model_max_nodes(model), false);

    const int64_t n_embd_head = hparams.n_embd_head_v;
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);
    GGML_ASSERT(n_embd_head == hparams.n_rot);

    struct ggml_tensor * cur;
    struct ggml_tensor * inpL;

    inpL = llm_build_inp_embd(ctx0, lctx, hparams, batch, model.tok_embd, cb);

    // inp_pos - contains the positions
    struct ggml_tensor * inp_pos = build_inp_pos();

    // KQ_mask (mask for 1 head, it will be broadcasted to all heads)
    struct ggml_tensor * KQ_mask = build_inp_KQ_mask();

    for (int il = 0; il < n_layer; ++il) {
        struct ggml_tensor * inpSA = inpL;

        // norm
        cur = llm_build_norm(ctx0, inpL, hparams, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, cb, il);
        cb(cur, "attn_norm", il);

        // self-attention
        {
            auto [Qcur, Kcur, Vcur] = llm_build_mul_mat_qkv(gf, cur, model.layers[il].wq, model.layers[il].bq,
                    model.layers[il].wk, model.layers[il].bk,
                    model.layers[il].wv, model.layers[il].bv, 0.f, il);

            Qcur = ggml_rope_ext(
                    ctx0, ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens), inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );
            cb(Qcur, "Qcur", il);

            Kcur = ggml_rope_ext(
                    ctx0, ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens), inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );
            cb(Kcur, "Kcur", il);

            cur = llm_build_kv(ctx0, lctx, kv_self, gf,
                    model.layers[il].wo, model.layers[il].bo,
                    Kcur, Vcur, Qcur, KQ_mask, n_tokens, kv_head, n_kv, 1.0f/sqrtf(float(n_embd_head)), cb, il);
        }

        if (il == n_layer - 1) {
            // skip computing output for unused tokens
            struct ggml_tensor * inp_out_ids = build_inp_out_ids();
            cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }

        struct ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        // feed-forward network
        cur = llm_build_ffn(ctx0, lctx, model.layers[il].ffn_norm, ffn_inp,
                model.layers[il].ffn_up,   NULL, NULL,
                model.layers[il].ffn_gate, NULL, NULL,
                model.layers[il].ffn_down, NULL, NULL,
                NULL,
                LLM_FFN_SILU, LLM_FFN_PAR, cb, il);
        cb(cur, "ffn_out", il);

        cur = ggml_add(ctx0, cur, ffn_inp);
        cur = lctx.cvec.apply_to(ctx0, cur, il);
        cb(cur, "l_out", il);

        // input for next layer
        inpL = cur;
    }

    cur = inpL;

    cur = llm_build_norm(ctx0, cur, hparams, model.output_norm, NULL, LLM_NORM_RMS, cb, -1);
    cb(cur, "result_norm", -1);

    // lm_head
    cur = llm_build_lora_mm(lctx, ctx0, model.output, cur);
    cb(cur, "result_output", -1);

    ggml_build_forward_expand(gf, cur);

    return gf;
}

ggml_cgraph * llm_build_context::build_qwen2vl() {
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, llama_model_max_nodes(model), false);

    const int64_t n_embd_head = hparams.n_embd_head_v;

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);
    GGML_ASSERT(n_embd_head == hparams.n_rot);

    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = llm_build_inp_embd(ctx0, lctx, hparams, batch, model.tok_embd, cb);

    // inp_pos - contains the positions
    struct ggml_tensor * inp_pos = build_inp_pos();

    // KQ_mask (mask for 1 head, it will be broadcasted to all heads)
    struct ggml_tensor * KQ_mask = build_inp_KQ_mask();

    //auto * inp_attn = build_attn_inp_kv();

    int sections[4];
    std::copy(std::begin(hparams.rope_sections), std::begin(hparams.rope_sections) + 4, sections);

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    for (int il = 0; il < n_layer; ++il) {
        ggml_tensor * inpSA = inpL;

        // norm
        cur = llm_build_norm(ctx0, inpL, hparams, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, cb, il);
        cb(cur, "attn_norm", il);

        // self-attention
        {
            auto [Qcur, Kcur, Vcur] = llm_build_mul_mat_qkv(gf, cur, model.layers[il].wq, model.layers[il].bq,
                    model.layers[il].wk, model.layers[il].bk,
                    model.layers[il].wv, model.layers[il].bv, 0.f, il);
            Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
            Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);

            Qcur = ggml_rope_multi(
                    ctx0, Qcur, inp_pos, nullptr,
                    n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );

            Kcur = ggml_rope_multi(
                    ctx0, Kcur, inp_pos, nullptr,
                    n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );

            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);

            cur = llm_build_kv(ctx0, lctx, kv_self, gf,
                    model.layers[il].wo, model.layers[il].bo,
                    Kcur, Vcur, Qcur, KQ_mask, n_tokens, kv_head, n_kv, 1.0f/sqrtf(float(n_embd_head)), cb, il);
        }

        if (il == n_layer - 1 && inp_out_ids) {
            cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }

        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        // feed-forward network
        cur = llm_build_ffn(ctx0, lctx, model.layers[il].ffn_norm, ffn_inp,
                model.layers[il].ffn_up,   NULL, NULL,
                model.layers[il].ffn_gate, NULL, NULL,
                model.layers[il].ffn_down, NULL, NULL,
                NULL,
                LLM_FFN_SILU, LLM_FFN_PAR, cb, il);
        cb(cur, "ffn_out", il);

        cur = ggml_add(ctx0, cur, ffn_inp);
        cur = lctx.cvec.apply_to(ctx0, cur, il);
        cb(cur, "l_out", il);

        // input for next layer
        inpL = cur;
    }

    cur = inpL;

    cur = llm_build_norm(ctx0, cur, hparams, model.output_norm, NULL, LLM_NORM_RMS, cb, -1);
    cb(cur, "result_norm", -1);

    // lm_head
    cur = llm_build_lora_mm(lctx, ctx0, model.output, cur);
    cb(cur, "result_output", -1);

    ggml_build_forward_expand(gf, cur);

    return gf;

}

ggml_cgraph * llm_build_context::build_qwen2moe() {
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, llama_model_max_nodes(model), false);

    // mutable variable, needed during the last layer of the computation to skip unused tokens
    int32_t n_tokens = this->n_tokens;

    const int64_t n_embd_head = hparams.n_embd_head_v;
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);
    GGML_ASSERT(n_embd_head == hparams.n_rot);

    struct ggml_tensor * cur;
    struct ggml_tensor * inpL;

    inpL = llm_build_inp_embd(ctx0, lctx, hparams, batch, model.tok_embd, cb);

    // inp_pos - contains the positions
    struct ggml_tensor * inp_pos = build_inp_pos();

    // KQ_mask (mask for 1 head, it will be broadcasted to all heads)
    struct ggml_tensor * KQ_mask = build_inp_KQ_mask();

    for (int il = 0; il < n_layer; ++il) {
        struct ggml_tensor * inpSA = inpL;

        // norm
        cur = llm_build_norm(ctx0, inpL, hparams, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, cb, il);
        cb(cur, "attn_norm", il);

        // self_attention
        {
            auto [Qcur, Kcur, Vcur] = llm_build_mul_mat_qkv(gf, cur, model.layers[il].wq, model.layers[il].bq,
                    model.layers[il].wk, model.layers[il].bk,
                    model.layers[il].wv, model.layers[il].bv, 0.f, il);

            Qcur = ggml_rope_ext(
                    ctx0, ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens), inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );
            cb(Qcur, "Qcur", il);

            Kcur = ggml_rope_ext(
                    ctx0, ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens), inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );
            cb(Kcur, "Kcur", il);

            cur = llm_build_kv(ctx0, lctx, kv_self, gf,
                    model.layers[il].wo, model.layers[il].bo,
                    Kcur, Vcur, Qcur, KQ_mask, n_tokens, kv_head, n_kv, 1.0f/sqrtf(float(n_embd_head)), cb, il);
        }

        if (il == n_layer - 1) {
            // skip computing output for unused tokens
            struct ggml_tensor * inp_out_ids = build_inp_out_ids();
            n_tokens = n_outputs;
            cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }

        struct ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        // MoE branch
        cur = llm_build_norm(ctx0, ffn_inp, hparams, model.layers[il].ffn_norm, NULL, LLM_NORM_RMS, cb, il);
        cb(cur, "ffn_norm", il);

        ggml_tensor * moe_out =
            llm_build_moe_ffn(ctx0, lctx, cur,
                    model.layers[il].ffn_gate_inp,
                    model.layers[il].ffn_up_exps,
                    model.layers[il].ffn_gate_exps,
                    model.layers[il].ffn_down_exps,
                    nullptr,
                    n_expert, n_expert_used,
                    LLM_FFN_SILU, false,
                    false, 0.0,
                    LLM_EXPERT_GATING_FUNC_SOFTMAX,
                    cb, il, gf);
        cb(cur, "ffn_moe_out", il);

        // FFN shared expert
        {
            ggml_tensor * cur_gate_inp = llm_build_lora_mm(lctx, ctx0, model.layers[il].ffn_gate_inp_shexp, cur);
            cb(cur_gate_inp, "ffn_shexp_gate_inp", il);

            // sigmoid
            ggml_tensor * cur_gate = ggml_div(ctx0, ggml_silu(ctx0, cur_gate_inp), cur_gate_inp);
            cb(cur_gate, "ffn_shexp_gate", il);

            ggml_tensor * cur_ffn = llm_build_ffn(ctx0, lctx, nullptr, cur,
                    model.layers[il].ffn_up_shexp,   NULL, NULL,
                    model.layers[il].ffn_gate_shexp, NULL, NULL,
                    model.layers[il].ffn_down_shexp, NULL, NULL,
                    NULL,
                    LLM_FFN_SILU, LLM_FFN_PAR, cb, il);
            cb(cur_ffn, "ffn_shexp", il);

            ggml_tensor * ffn_shexp_out = ggml_mul(ctx0, cur_ffn, cur_gate);
            cb(ffn_shexp_out, "ffn_shexp_out", il);

            moe_out = ggml_add(ctx0, moe_out, ffn_shexp_out);
            cb(moe_out, "ffn_out", il);

            cur = moe_out;
        }

        cur = ggml_add(ctx0, cur, ffn_inp);
        cur = lctx.cvec.apply_to(ctx0, cur, il);
        cb(cur, "l_out", il);

        // input for next layer
        inpL = cur;
    }

    cur = inpL;

    cur = llm_build_norm(ctx0, cur, hparams, model.output_norm, NULL, LLM_NORM_RMS, cb, -1);
    cb(cur, "result_norm", -1);

    // lm_head
    cur = llm_build_lora_mm(lctx, ctx0, model.output, cur);
    cb(cur, "result_output", -1);

    ggml_build_forward_expand(gf, cur);

    return gf;
}

ggml_cgraph * llm_build_context::build_qwen3() {
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, llama_model_max_nodes(model), false);

    const int64_t n_embd_head = hparams.n_embd_head_v;
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);
    GGML_ASSERT(n_embd_head == hparams.n_rot);

    struct ggml_tensor * cur;
    struct ggml_tensor * inpL;

    inpL = llm_build_inp_embd(ctx0, lctx, hparams, batch, model.tok_embd, cb);

    // inp_pos - contains the positions
    struct ggml_tensor * inp_pos = build_inp_pos();

    // KQ_mask (mask for 1 head, it will be broadcasted to all heads)
    struct ggml_tensor * KQ_mask = build_inp_KQ_mask();

    ggml_tensor * rope_cache = nullptr;
    if (model.split_mode != LLAMA_SPLIT_MODE_GRAPH && cparams.rope_cache &&
            (rope_type == LLAMA_ROPE_TYPE_NEOX || rope_type == LLAMA_ROPE_TYPE_NORM)) {
        rope_cache = ggml_rope_cache(ctx0, inp_pos, nullptr, n_embd_head, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                ext_factor, attn_factor, beta_fast, beta_slow);
    }

    for (int il = 0; il < n_layer; ++il) {
        struct ggml_tensor * inpSA = inpL;

        if (!rope_cache) {
            cur = build_std_attention(gf, model.layers[il].attn_norm, inpL, inp_pos, nullptr, KQ_mask, nullptr, nullptr,
                    1.0f/sqrtf(float(n_embd_head)), 0.0f, 0, il, true, false, true);
        } else {

            // norm
            cur = llm_build_norm(ctx0, inpL, hparams, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, cb, il);
            cb(cur, "attn_norm", il);

            // self-attention
            {
                auto [Qcur, Kcur, Vcur] = llm_build_mul_mat_qkv(gf, cur,
                        model.layers[il].wqkv, nullptr,
                        model.layers[il].wqk, nullptr,
                        model.layers[il].wq, nullptr,
                        model.layers[il].wk, nullptr,
                        model.layers[il].wv, nullptr,
                        model.layers[il].attn_q_norm, model.layers[il].attn_k_norm, 0, il);

                if (rope_cache) {
                    Qcur = ggml_rope_fast(ctx0, Qcur, rope_cache);
                    Kcur = ggml_rope_fast(ctx0, Kcur, rope_cache);
                } else {
                    Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                            ext_factor, attn_factor, beta_fast, beta_slow);
                    Kcur = ggml_rope_ext(ctx0, Kcur, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                            ext_factor, attn_factor, beta_fast, beta_slow);
                }
                cb(Qcur, "Qcur", il);
                cb(Kcur, "Kcur", il);

                cur = llm_build_kv(ctx0, lctx, kv_self, gf,
                        model.layers[il].wo, model.layers[il].bo,
                        Kcur, Vcur, Qcur, KQ_mask, n_tokens, kv_head, n_kv, 1.0f/sqrtf(float(n_embd_head)), cb, il);

                cur = ggml_add(ctx0, cur, inpSA);
                cb(cur, "attn_with_inp", il);
            }
        }

        if (il == n_layer - 1) {
            // skip computing output for unused tokens
            struct ggml_tensor * inp_out_ids = build_inp_out_ids();
            cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
        }

        // feed-forward network
        cur = llm_build_ffn(ctx0, lctx, model.layers[il].ffn_norm, cur,
                model.layers[il].ffn_up,   NULL, NULL,
                model.layers[il].ffn_gate, NULL, NULL,
                model.layers[il].ffn_down, NULL, NULL,
                NULL,
                LLM_FFN_SILU, LLM_FFN_PAR, cb, il, gf, true);
        cb(cur, "ffn_out", il);

        cur = lctx.cvec.apply_to(ctx0, cur, il);
        cb(cur, "l_out", il);

        // input for next layer
        inpL = cur;
    }

    cur = build_output(lctx, ctx0, inpL, model.output, model.output_norm, cb);
    cb(cur, "result_output", -1);

    ggml_build_forward_expand(gf, cur);

    return gf;
}

ggml_cgraph * llm_build_context::build_qwen3moe() {
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, llama_model_max_nodes(model), false);

    const int64_t n_embd_head = hparams.n_embd_head_v;
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);
    GGML_ASSERT(n_embd_head == hparams.n_rot);

    struct ggml_tensor * cur;
    struct ggml_tensor * inpL;

    inpL = llm_build_inp_embd(ctx0, lctx, hparams, batch, model.tok_embd, cb);

    // inp_pos - contains the positions
    struct ggml_tensor * inp_pos = build_inp_pos();

    // KQ_mask (mask for 1 head, it will be broadcasted to all heads)
    struct ggml_tensor * KQ_mask = build_inp_KQ_mask();

    for (int il = 0; il < n_layer; ++il) {
        //struct ggml_tensor * inpSA = inpL;

        // norm
        //cur = llm_build_norm(ctx0, inpL, hparams, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, cb, il);
        //cb(cur, "attn_norm", il);

        cur = build_std_attention(gf, model.layers[il].attn_norm, inpL, inp_pos, nullptr, KQ_mask, nullptr, nullptr, 1.0f/sqrtf(float(n_embd_head)), 0.0f, 0,
                il, true, false, true);
        //printf("%s: attn = %s(%s)\n", __func__, cur->name, ggml_op_name(cur->op));

        if (il == n_layer - 1) {
            // skip computing output for unused tokens
            struct ggml_tensor * inp_out_ids = build_inp_out_ids();
            cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
            //inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }

        auto ffn_inp = cur;
        //struct ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        //cb(ffn_inp, "ffn_inp", il);

        cur = llm_build_std_moe_ffn(ctx0, lctx, model.layers[il].ffn_norm, ffn_inp,
                model.layers[il].ffn_gate_inp,  nullptr,
                model.layers[il].ffn_up_exps,   nullptr,
                model.layers[il].ffn_gate_exps, nullptr,
                model.layers[il].ffn_down_exps, nullptr,
                model.layers[il].ffn_exp_probs_b,
                nullptr,  nullptr, // we don't have shared expert biases?
                nullptr,  nullptr,
                nullptr,  nullptr,
                n_expert, n_expert_used,
                LLM_FFN_SILU, true, false, 0.0f,
                LLM_EXPERT_GATING_FUNC_SOFTMAX,
                LLM_FFN_SILU, cb, il, gf, true,
                model.layers[il].ffn_up_gate_exps);

        //printf("%s: ffn = %s(%s)\n", __func__, cur->name, ggml_op_name(cur->op));

        //cur = ggml_add(ctx0, cur, ffn_inp);
        cur = lctx.cvec.apply_to(ctx0, cur, il);
        cb(cur, "l_out", il);

        // input for next layer
        inpL = cur;
    }

    cur = inpL;

    cur = build_output(lctx, ctx0, cur, model.output, model.output_norm, cb);
    cb(cur, "result_output", -1);

    ggml_build_forward_expand(gf, cur);

    return gf;
}

ggml_cgraph * llm_build_context::build_qwen3vl() {
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, llama_model_max_nodes(model), false);

    const int64_t n_embd_full = hparams.n_embd; // main embd + deepstack embds
    const size_t n_deepstack_layers = hparams.n_deepstack_layers;
    const int64_t n_embd = n_embd_full / (n_deepstack_layers + 1);
    const int64_t n_embd_head = hparams.n_embd_head_v;

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);
    GGML_ASSERT(n_embd_head == hparams.n_rot);

    struct ggml_tensor * cur;
    struct ggml_tensor * inpL;

    inpL = llm_build_inp_embd(ctx0, lctx, hparams, batch, model.tok_embd, cb);

    int sections[4];
    std::copy(std::begin(hparams.rope_sections), std::begin(hparams.rope_sections) + 4, sections);

    std::vector<struct ggml_tensor *> deepstack_features;

    if (batch.embd) {
        deepstack_features.resize(n_deepstack_layers, nullptr);
        // Image input: split main embd and deepstack embds
        struct ggml_tensor * inpL_main = ggml_view_2d(ctx0, inpL, n_embd, n_tokens, inpL->nb[1], 0);
        for (size_t i = 0; i < n_deepstack_layers; i++) {
            deepstack_features[i] = ggml_view_2d(ctx0, inpL, n_embd, n_tokens, inpL->nb[1], (i + 1) * n_embd * sizeof(float));
        }
        inpL = inpL_main;
    }

    // inp_pos - contains the positions
    struct ggml_tensor * inp_pos = build_inp_pos();

    // KQ_mask (mask for 1 head, it will be broadcasted to all heads)
    struct ggml_tensor * KQ_mask = build_inp_KQ_mask();

    for (int il = 0; il < n_layer; ++il) {

        cur = build_std_attention(gf, model.layers[il].attn_norm, inpL, inp_pos, nullptr, KQ_mask,
                nullptr, nullptr, 1.0f/sqrtf(float(n_embd_head)), 0.0f, 0, il, true, false, true, false, true);

        if (il == n_layer - 1) {
            // skip computing output for unused tokens
            struct ggml_tensor * inp_out_ids = build_inp_out_ids();
            cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
        }

        // feed-forward network
        cur = llm_build_ffn(ctx0, lctx, model.layers[il].ffn_norm, cur,
                model.layers[il].ffn_up,   NULL, NULL,
                model.layers[il].ffn_gate, NULL, NULL,
                model.layers[il].ffn_down, NULL, NULL,
                NULL,
                LLM_FFN_SILU, LLM_FFN_PAR, cb, il, gf, true, false,
                batch.embd && (size_t)il < n_deepstack_layers ? deepstack_features[il] : nullptr);

        cur = lctx.cvec.apply_to(ctx0, cur, il);
        cb(cur, "l_out", il);

        // input for next layer
        inpL = cur;
    }

    cur = build_output(lctx, ctx0, inpL, model.output, model.output_norm, cb);
    cb(cur, "result_output", -1);

    ggml_build_forward_expand(gf, cur);

    return gf;
}

ggml_cgraph * llm_build_context::build_qwen3vlmoe() {
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, llama_model_max_nodes(model), false);

    // mutable variable, needed during the last layer of the computation to skip unused tokens
    int32_t n_tokens = this->n_tokens;

    const int64_t n_embd_full = hparams.n_embd; // main embd + deepstack embds
    const size_t n_deepstack_layers = hparams.n_deepstack_layers;
    const int64_t n_embd = n_embd_full / (n_deepstack_layers + 1);
    const int64_t n_embd_head = hparams.n_embd_head_v;

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);
    GGML_ASSERT(n_embd_head == hparams.n_rot);

    struct ggml_tensor * cur;
    struct ggml_tensor * inpL;

    inpL = llm_build_inp_embd(ctx0, lctx, hparams, batch, model.tok_embd, cb);

    int sections[4];
    std::copy(std::begin(hparams.rope_sections), std::begin(hparams.rope_sections) + 4, sections);

    std::vector<struct ggml_tensor *> deepstack_features(n_deepstack_layers, nullptr);

    if (batch.embd) {
        // Image input: split main embd and deepstack embds
        struct ggml_tensor * inpL_main = ggml_view_2d(ctx0, inpL, n_embd, n_tokens, inpL->nb[1], 0);
        for (size_t i = 0; i < n_deepstack_layers; i++) {
            deepstack_features[i] = ggml_view_2d(ctx0, inpL, n_embd, n_tokens, inpL->nb[1], (i + 1) * n_embd * sizeof(float));
        }
        inpL = inpL_main;
    }

    // inp_pos - contains the positions
    struct ggml_tensor * inp_pos = build_inp_pos();

    // KQ_mask (mask for 1 head, it will be broadcasted to all heads)
    struct ggml_tensor * KQ_mask = build_inp_KQ_mask();

    for (int il = 0; il < n_layer; ++il) {
        struct ggml_tensor * inpSA = inpL;

        // norm
        cur = llm_build_norm(ctx0, inpL, hparams,
                model.layers[il].attn_norm, NULL,
                LLM_NORM_RMS, cb, il);
        cb(cur, "attn_norm", il);

        // self_attention
        {
            auto [Qcur, Kcur, Vcur] = llm_build_mul_mat_qkv(gf, cur,
                                                            model.layers[il].wq, nullptr,
                                                            model.layers[il].wk, nullptr,
                                                            model.layers[il].wv, nullptr,
                                                            0, il);

            Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens);
            Qcur = llm_build_norm(ctx0, Qcur, hparams, model.layers[il].attn_q_norm, NULL, LLM_NORM_RMS, cb, il);
            cb(Qcur, "Qcur_normed", il);

            Qcur = ggml_rope_multi(
                    ctx0, Qcur, inp_pos, nullptr,
                    n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );
            cb(Qcur, "Qcur", il);

            Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
            Kcur = llm_build_norm(ctx0, Kcur, hparams, model.layers[il].attn_k_norm, NULL, LLM_NORM_RMS, cb, il);
            cb(Kcur, "Kcur_normed", il);

            Kcur = ggml_rope_multi(
                    ctx0, Kcur, inp_pos, nullptr,
                    n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );
            cb(Kcur, "Kcur", il);

            cb(Vcur, "Vcur", il);

            cur = llm_build_kv(ctx0, lctx, kv_self, gf,
                    model.layers[il].wo, model.layers[il].bo,
                    Kcur, Vcur, Qcur, KQ_mask, n_tokens, kv_head, n_kv, 1.0f/sqrtf(float(n_embd_head)), cb, il);
        }

        if (il == n_layer - 1) {
            // skip computing output for unused tokens
            struct ggml_tensor * inp_out_ids = build_inp_out_ids();
            n_tokens = n_outputs;
            cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }

        struct ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        // MoE branch
        cur = llm_build_norm(ctx0, ffn_inp, hparams,
                model.layers[il].ffn_norm, NULL,
                LLM_NORM_RMS, cb, il);
        cb(cur, "ffn_norm", il);

        cur =
            llm_build_moe_ffn(ctx0, lctx, cur,
                    model.layers[il].ffn_gate_inp,
                    model.layers[il].ffn_up_exps,
                    model.layers[il].ffn_gate_exps,
                    model.layers[il].ffn_down_exps,
                    nullptr,
                    n_expert, n_expert_used,
                    LLM_FFN_SILU, true,
                    false, 0.0,
                    LLM_EXPERT_GATING_FUNC_SOFTMAX,
                    cb, il, gf);
        cb(cur, "ffn_moe_out", il);

        cur = ggml_add(ctx0, cur, ffn_inp);

        cur = lctx.cvec.apply_to(ctx0, cur, il);
        cb(cur, "l_out", il);

        if (batch.embd && (size_t)il < n_deepstack_layers) {
            cur = ggml_add(ctx0, cur, deepstack_features[il]);
            cb(cur, "deepstack_out", il);
        }

        // input for next layer
        inpL = cur;
    }

    cur = inpL;

    cur = llm_build_norm(ctx0, cur, hparams,
            model.output_norm, NULL,
            LLM_NORM_RMS, cb, -1);
    cb(cur, "result_norm", -1);

    // lm_head
    cur = llm_build_lora_mm(lctx, ctx0, model.output, cur);
    cb(cur, "result_output", -1);

    ggml_build_forward_expand(gf, cur);

    return gf;
}

ggml_cgraph * llm_build_context::build_phi2() {
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, llama_model_max_nodes(model), false);

    const int64_t n_embd_head = hparams.n_embd_head_v;
    const int64_t n_embd_gqa  = hparams.n_embd_v_gqa();
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);

    struct ggml_tensor * cur;
    struct ggml_tensor * attn_norm_output;
    struct ggml_tensor * ffn_output;
    struct ggml_tensor * inpL;

    inpL = llm_build_inp_embd(ctx0, lctx, hparams, batch, model.tok_embd, cb);

    // inp_pos - contains the positions
    struct ggml_tensor * inp_pos = build_inp_pos();

    // KQ_mask (mask for 1 head, it will be broadcasted to all heads)
    struct ggml_tensor * KQ_mask = build_inp_KQ_mask();

    for (int il = 0; il < n_layer; ++il) {
        attn_norm_output = llm_build_norm(ctx0, inpL, hparams, model.layers[il].attn_norm, model.layers[il].attn_norm_b, LLM_NORM, cb, il);
        cb(attn_norm_output, "attn_norm", il);

        // self-attention
        {
            struct ggml_tensor * Qcur = nullptr;
            struct ggml_tensor * Kcur = nullptr;
            struct ggml_tensor * Vcur = nullptr;

            if (model.layers[il].wqkv) {
                cur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wqkv, attn_norm_output);
                cb(cur, "wqkv", il);

                cur = ggml_add(ctx0, cur, model.layers[il].bqkv);
                cb(cur, "bqkv", il);

                Qcur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd,     n_tokens, cur->nb[1], 0*sizeof(float)*(n_embd)));
                Kcur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd_gqa, n_tokens, cur->nb[1], 1*sizeof(float)*(n_embd)));
                Vcur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd_gqa, n_tokens, cur->nb[1], 1*sizeof(float)*(n_embd + n_embd_gqa)));
            } else {
                Qcur = ggml_add(ctx0, llm_build_lora_mm(lctx, ctx0, model.layers[il].wq, attn_norm_output), model.layers[il].bq);
                Kcur = ggml_add(ctx0, llm_build_lora_mm(lctx, ctx0, model.layers[il].wk, attn_norm_output), model.layers[il].bk);
                Vcur = ggml_add(ctx0, llm_build_lora_mm(lctx, ctx0, model.layers[il].wv, attn_norm_output), model.layers[il].bv);
            }

            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
            Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);

            Qcur = ggml_rope_ext(
                    ctx0, Qcur, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig,
                    freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow
                    );
            cb(Qcur, "Qcur", il);

            // with phi2, we scale the Q to avoid precision issues
            // ref: https://github.com/ml-explore/mlx-examples/blob/08e862336ade809bc37d1035f94b359e7d1a5152/phi2/phi2.py#L64-L66
            Qcur = ggml_scale(ctx0, Qcur, 1.0f/sqrtf(float(n_embd_head)));
            cb(Qcur, "Qcur", il);

            Kcur = ggml_rope_ext(
                    ctx0, Kcur, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig,
                    freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow
                    );
            cb(Kcur, "Kcur", il);

            cur = llm_build_kv(ctx0, lctx, kv_self, gf,
                    model.layers[il].wo, model.layers[il].bo,
                    Kcur, Vcur, Qcur, KQ_mask, n_tokens, kv_head, n_kv, 1.0f, cb, il);
        }

        if (il == n_layer - 1) {
            // skip computing output for unused tokens
            struct ggml_tensor * inp_out_ids = build_inp_out_ids();
            cur              = ggml_get_rows(ctx0,              cur, inp_out_ids);
            inpL             = ggml_get_rows(ctx0,             inpL, inp_out_ids);
            attn_norm_output = ggml_get_rows(ctx0, attn_norm_output, inp_out_ids);
        }

        // FF
        {
            ffn_output = llm_build_ffn(ctx0, lctx, nullptr, attn_norm_output,
                    model.layers[il].ffn_up,   model.layers[il].ffn_up_b,   NULL,
                    NULL,                      NULL,                        NULL,
                    model.layers[il].ffn_down, model.layers[il].ffn_down_b, NULL,
                    NULL,
                    LLM_FFN_GELU, LLM_FFN_SEQ, cb, il);
            cb(ffn_output, "ffn_out", il);
        }

        cur = ggml_add(ctx0, cur, ffn_output);
        cur = ggml_add(ctx0, cur, inpL);
        cur = lctx.cvec.apply_to(ctx0, cur, il);
        cb(cur, "l_out", il);

        // input for next layer
        inpL = cur;
    }

    cur = llm_build_norm(ctx0, inpL, hparams, model.output_norm, model.output_norm_b, LLM_NORM, cb, -1);
    cb(cur, "result_norm", -1);

    cur = llm_build_lora_mm(lctx, ctx0, model.output, cur);
    cb(cur, "result_output_no_bias", -1);

    cur = ggml_add(ctx0, cur, model.output_b);
    cb(cur, "result_output", -1);
    ggml_build_forward_expand(gf, cur);
    return gf;
}

ggml_cgraph * llm_build_context::build_phi3() {
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, llama_model_max_nodes(model), false);

    const int64_t n_embd_head = hparams.n_embd_head_v;
    const int64_t n_embd_gqa = hparams.n_embd_v_gqa();
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);

    struct ggml_tensor * cur;
    struct ggml_tensor * inpL;

    inpL = llm_build_inp_embd(ctx0, lctx, hparams, batch, model.tok_embd, cb);

    // inp_pos - contains the positions
    struct ggml_tensor * inp_pos = build_inp_pos();

    // KQ_mask (mask for 1 head, it will be broadcasted to all heads)
    struct ggml_tensor * KQ_mask_swa = build_inp_KQ_mask_swa();

    for (int il = 0; il < n_layer; ++il) {
        auto residual = inpL;

        // self-attention
        {
            // rope freq factors for 128k context
            struct ggml_tensor * rope_factors = build_rope_factors(il);

            struct ggml_tensor * attn_norm_output = llm_build_norm(ctx0, inpL, hparams, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, cb, il);
            cb(attn_norm_output, "attn_norm", il);

            struct ggml_tensor * Qcur = nullptr;
            struct ggml_tensor * Kcur = nullptr;
            struct ggml_tensor * Vcur = nullptr;

            if (model.layers[il].wqkv) {
                cur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wqkv, attn_norm_output);
                cb(cur, "wqkv", il);

                Qcur = ggml_view_3d(ctx0, cur, n_embd_head,    n_head, n_tokens, n_embd_head*sizeof(float), cur->nb[1], 0 * sizeof(float) * (n_embd));
                Kcur = ggml_view_3d(ctx0, cur, n_embd_head, n_head_kv, n_tokens, n_embd_head*sizeof(float), cur->nb[1], 1 * sizeof(float) * (n_embd));
                Vcur = ggml_view_2d(ctx0, cur, n_embd_gqa, n_tokens, cur->nb[1], 1 * sizeof(float) * (n_embd + n_embd_gqa));
            }
            else {
                Qcur = ggml_add(ctx0, llm_build_lora_mm(lctx, ctx0, model.layers[il].wq, attn_norm_output), model.layers[il].bq);
                Kcur = ggml_add(ctx0, llm_build_lora_mm(lctx, ctx0, model.layers[il].wk, attn_norm_output), model.layers[il].bk);
                Vcur = ggml_add(ctx0, llm_build_lora_mm(lctx, ctx0, model.layers[il].wv, attn_norm_output), model.layers[il].bv);
                Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
                Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
            }

            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            Qcur = ggml_rope_ext(
                    ctx0, Qcur, inp_pos, rope_factors, n_rot, rope_type, n_ctx_orig,
                    freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow
                    );
            cb(Qcur, "Qcur", il);

            Qcur = ggml_scale(ctx0, Qcur, 1.0f / sqrtf(float(n_embd_head)));
            cb(Qcur, "Qcur", il);

            Kcur = ggml_rope_ext(
                    ctx0, Kcur, inp_pos, rope_factors, n_rot, rope_type, n_ctx_orig,
                    freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow
                    );
            cb(Kcur, "Kcur", il);

            cur = llm_build_kv(ctx0, lctx, kv_self, gf,
                    model.layers[il].wo, model.layers[il].bo,
                    Kcur, Vcur, Qcur, KQ_mask_swa, n_tokens, kv_head, n_kv, 1.0f, cb, il);
        }

        if (il == n_layer - 1) {
            // skip computing output for unused tokens
            struct ggml_tensor * inp_out_ids = build_inp_out_ids();
            cur = ggml_get_rows(ctx0, cur, inp_out_ids);
            residual = ggml_get_rows(ctx0, residual, inp_out_ids);
        }

        cur = ggml_add(ctx0, cur, residual);
        residual = cur;

        // FF
        // special-case: the up and gate tensors are merged into a single tensor
        // TOOD: support into llm_build_ffn
        {
            cur = llm_build_ffn(ctx0, lctx, model.layers[il].ffn_norm, cur,
                    model.layers[il].ffn_up,   NULL, NULL,
                    NULL,                      NULL, NULL,
                    model.layers[il].ffn_down, NULL, NULL,
                    NULL,
                    LLM_FFN_SWIGLU, LLM_FFN_SEQ, cb, il);
            cb(cur, "ffn_out", il);
        }

        cur = ggml_add(ctx0, residual, cur);
        cur = lctx.cvec.apply_to(ctx0, cur, il);
        cb(cur, "l_out", il);

        // input for next layer
        inpL = cur;
    }

    cur = llm_build_norm(ctx0, inpL, hparams, model.output_norm, NULL, LLM_NORM_RMS, cb, -1);
    cb(cur, "result_norm", -1);

    cur = llm_build_lora_mm(lctx, ctx0, model.output, cur);
    cb(cur, "result_output", -1);

    ggml_build_forward_expand(gf, cur);

    return gf;
}

ggml_cgraph * llm_build_context::build_plamo() {
    struct ggml_cgraph * gf = ggml_new_graph(ctx0);

    const int64_t n_embd_head = hparams.n_embd_head_v;
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);
    GGML_ASSERT(n_embd_head == hparams.n_rot);

    struct ggml_tensor * cur;
    struct ggml_tensor * inpL;

    inpL = llm_build_inp_embd(ctx0, lctx, hparams, batch, model.tok_embd, cb);

    // inp_pos - contains the positions
    struct ggml_tensor * inp_pos = build_inp_pos();

    // KQ_mask (mask for 1 head, it will be broadcasted to all heads)
    struct ggml_tensor * KQ_mask = build_inp_KQ_mask();

    for (int il = 0; il < n_layer; ++il) {

        // norm
        cur = llm_build_norm(ctx0, inpL, hparams, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, cb, il);
        cb(cur, "attn_norm", il);

        struct ggml_tensor * attention_norm = cur;

        // self-attention
        {
            auto [Qcur, Kcur, Vcur] = llm_build_mul_mat_qkv(gf, cur, model.layers[il].wq, nullptr,
                    model.layers[il].wk, nullptr,
                    model.layers[il].wv, nullptr, 0, il);
            Qcur = ggml_rope_ext(
                    ctx0, ggml_reshape_3d(ctx0, Qcur, n_rot, n_head,    n_tokens), inp_pos, nullptr,
                    n_embd_head, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow);
            cb(Qcur, "Qcur", il);

            Kcur = ggml_rope_ext(
                    ctx0, ggml_reshape_3d(ctx0, Kcur, n_rot, n_head_kv, n_tokens), inp_pos, nullptr,
                    n_embd_head, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow);
            cb(Kcur, "Kcur", il);

            cur = llm_build_kv(ctx0, lctx, kv_self, gf,
                    model.layers[il].wo, NULL,
                    Kcur, Vcur, Qcur, KQ_mask, n_tokens, kv_head, n_kv, 1.0f/sqrtf(float(n_embd_head)), cb, il);
        }
        struct ggml_tensor * sa_out = cur;

        cur = attention_norm;

        if (il == n_layer - 1) {
            // skip computing output for unused tokens
            struct ggml_tensor * inp_out_ids = build_inp_out_ids();
            cur    = ggml_get_rows(ctx0,    cur, inp_out_ids);
            sa_out = ggml_get_rows(ctx0, sa_out, inp_out_ids);
            inpL   = ggml_get_rows(ctx0,   inpL, inp_out_ids);
        }

        // feed-forward network
        {
            cur = llm_build_ffn(ctx0, lctx, nullptr, cur,
                    model.layers[il].ffn_up,   NULL, NULL,
                    model.layers[il].ffn_gate, NULL, NULL,
                    model.layers[il].ffn_down, NULL, NULL,
                    NULL,
                    LLM_FFN_SILU, LLM_FFN_PAR, cb, il);
            cb(cur, "ffn_out", il);
        }

        cur = ggml_add(ctx0, cur, sa_out);
        cur = ggml_add(ctx0, cur, inpL);
        cur = lctx.cvec.apply_to(ctx0, cur, il);
        cb(cur, "l_out", il);

        // input for next layer
        inpL = cur;
    }

    cur = inpL;

    cur = llm_build_norm(ctx0, cur, hparams, model.output_norm, NULL, LLM_NORM_RMS, cb, -1);
    cb(cur, "result_norm", -1);

    // lm_head
    cur = llm_build_lora_mm(lctx, ctx0, model.output, cur);
    cb(cur, "result_output", -1);

    ggml_build_forward_expand(gf, cur);

    return gf;
}

ggml_cgraph * llm_build_context::build_gpt2() {
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, llama_model_max_nodes(model), false);

    const int64_t n_embd_head = hparams.n_embd_head_v;
    const int64_t n_embd_gqa  = hparams.n_embd_v_gqa();
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);

    struct ggml_tensor * cur;
    struct ggml_tensor * pos;
    struct ggml_tensor * inpL;

    inpL = llm_build_inp_embd(ctx0, lctx, hparams, batch, model.tok_embd, cb);

    // inp_pos - contains the positions
    struct ggml_tensor * inp_pos = build_inp_pos();

    // KQ_mask (mask for 1 head, it will be broadcasted to all heads)
    struct ggml_tensor * KQ_mask = build_inp_KQ_mask();

    pos = ggml_get_rows(ctx0, model.pos_embd, inp_pos);
    cb(pos, "pos_embd", -1);

    inpL = ggml_add(ctx0, inpL, pos);
    cb(inpL, "inpL", -1);

    for (int il = 0; il < n_layer; ++il) {
        cur = llm_build_norm(ctx0, inpL, hparams, model.layers[il].attn_norm, model.layers[il].attn_norm_b, LLM_NORM, cb, il);
        cb(cur, "attn_norm", il);

        // self-attention
        {
            cur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wqkv, cur);
            cb(cur, "wqkv", il);

            cur = ggml_add(ctx0, cur, model.layers[il].bqkv);
            cb(cur, "bqkv", il);

            struct ggml_tensor * Qcur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd,     n_tokens, cur->nb[1], 0*sizeof(float)*(n_embd)));
            struct ggml_tensor * Kcur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd_gqa, n_tokens, cur->nb[1], 1*sizeof(float)*(n_embd)));
            struct ggml_tensor * Vcur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd_gqa, n_tokens, cur->nb[1], 1*sizeof(float)*(n_embd + n_embd_gqa)));

            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens);

            cur = llm_build_kv(ctx0, lctx, kv_self, gf,
                    model.layers[il].wo, model.layers[il].bo,
                    Kcur, Vcur, Qcur, KQ_mask, n_tokens, kv_head, n_kv, 1.0f/sqrtf(float(n_embd_head)), cb, il);
        }

        if (il == n_layer - 1) {
            // skip computing output for unused tokens
            struct ggml_tensor * inp_out_ids = build_inp_out_ids();
            cur  = ggml_get_rows(ctx0,  cur, inp_out_ids);
            inpL = ggml_get_rows(ctx0, inpL, inp_out_ids);
        }

        // add the input
        struct ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpL);
        cb(ffn_inp, "ffn_inp", il);

        // FF
        {
            cur = llm_build_ffn(ctx0, lctx, model.layers[il].ffn_norm, ffn_inp,
                    model.layers[il].ffn_up,   model.layers[il].ffn_up_b,   NULL,
                    NULL,                      NULL,                        NULL,
                    model.layers[il].ffn_down, model.layers[il].ffn_down_b, NULL,
                    NULL,
                    LLM_FFN_GELU, LLM_FFN_SEQ, cb, il);
            cb(cur, "ffn_out", il);
        }

        cur = ggml_add(ctx0, cur, ffn_inp);
        cur = lctx.cvec.apply_to(ctx0, cur, il);
        cb(cur, "l_out", il);

        // input for next layer
        inpL = cur;
    }

    cur = llm_build_norm(ctx0, inpL, hparams, model.output_norm, model.output_norm_b, LLM_NORM, cb, -1);
    cb(cur, "result_norm", -1);

    cur = llm_build_lora_mm(lctx, ctx0, model.output, cur);
    cb(cur, "result_output", -1);

    ggml_build_forward_expand(gf, cur);

    return gf;
}

ggml_cgraph * llm_build_context::build_codeshell() {
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, llama_model_max_nodes(model), false);

    const int64_t n_embd_head = hparams.n_embd_head_v;
    const int64_t n_embd_gqa  = hparams.n_embd_v_gqa();
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);
    GGML_ASSERT(n_embd_head == hparams.n_rot);

    struct ggml_tensor * cur;
    struct ggml_tensor * inpL;

    inpL = llm_build_inp_embd(ctx0, lctx, hparams, batch, model.tok_embd, cb);

    // inp_pos - contains the positions
    struct ggml_tensor * inp_pos = build_inp_pos();

    // KQ_mask (mask for 1 head, it will be broadcasted to all heads)
    struct ggml_tensor * KQ_mask = build_inp_KQ_mask();

    for (int il = 0; il < n_layer; ++il) {
        cur = llm_build_norm(ctx0, inpL, hparams, model.layers[il].attn_norm, model.layers[il].attn_norm_b, LLM_NORM, cb, il);
        cb(cur, "attn_norm", il);

        // self-attention
        {
            cur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wqkv, cur);
            cb(cur, "wqkv", il);

            cur = ggml_add(ctx0, cur, model.layers[il].bqkv);
            cb(cur, "bqkv", il);

            struct ggml_tensor * tmpq = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd,     n_tokens, cur->nb[1], 0*sizeof(float)*(n_embd)));
            struct ggml_tensor * tmpk = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd_gqa, n_tokens, cur->nb[1], 1*sizeof(float)*(n_embd)));
            struct ggml_tensor * Vcur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd_gqa, n_tokens, cur->nb[1], 1*sizeof(float)*(n_embd + n_embd_gqa)));

            cb(tmpq, "tmpq", il);
            cb(tmpk, "tmpk", il);
            cb(Vcur, "Vcur", il);

            struct ggml_tensor * Qcur = ggml_rope_ext(
                    ctx0, ggml_reshape_3d(ctx0, tmpq, n_embd_head, n_head,    n_tokens), inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );
            cb(Qcur, "Qcur", il);

            struct ggml_tensor * Kcur = ggml_rope_ext(
                    ctx0, ggml_reshape_3d(ctx0, tmpk, n_embd_head, n_head_kv, n_tokens), inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );
            cb(Kcur, "Kcur", il);

            cur = llm_build_kv(ctx0, lctx, kv_self, gf,
                    model.layers[il].wo, model.layers[il].bo,
                    Kcur, Vcur, Qcur, KQ_mask, n_tokens, kv_head, n_kv, 1.0f/sqrtf(float(n_embd_head)), cb, il);
        }

        if (il == n_layer - 1) {
            // skip computing output for unused tokens
            struct ggml_tensor * inp_out_ids = build_inp_out_ids();
            cur  = ggml_get_rows(ctx0,  cur, inp_out_ids);
            inpL = ggml_get_rows(ctx0, inpL, inp_out_ids);
        }

        // add the input
        struct ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpL);
        cb(ffn_inp, "ffn_inp", il);

        // FF
        {
            cur = llm_build_ffn(ctx0, lctx, model.layers[il].ffn_norm, ffn_inp,
                    model.layers[il].ffn_up,   model.layers[il].ffn_up_b,   NULL,
                    NULL,                      NULL,                        NULL,
                    model.layers[il].ffn_down, model.layers[il].ffn_down_b, NULL,
                    NULL,
                    LLM_FFN_GELU, LLM_FFN_SEQ, cb, il);
            cb(cur, "ffn_out", il);
        }

        cur = ggml_add(ctx0, cur, ffn_inp);
        cur = lctx.cvec.apply_to(ctx0, cur, il);
        cb(cur, "l_out", il);

        // input for next layer
        inpL = cur;
    }

    cur = llm_build_norm(ctx0, inpL, hparams, model.output_norm, model.output_norm_b, LLM_NORM, cb, -1);
    cb(cur, "result_norm", -1);

    cur = llm_build_lora_mm(lctx, ctx0, model.output, cur);
    cb(cur, "result_output", -1);

    ggml_build_forward_expand(gf, cur);

    return gf;
}

ggml_cgraph * llm_build_context::build_orion() {
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, llama_model_max_nodes(model), false);

    const int64_t n_embd_head = hparams.n_embd_head_v;
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);
    GGML_ASSERT(n_embd_head == hparams.n_rot);

    struct ggml_tensor * cur;
    struct ggml_tensor * inpL;

    inpL = llm_build_inp_embd(ctx0, lctx, hparams, batch, model.tok_embd, cb);

    // inp_pos - contains the positions
    struct ggml_tensor * inp_pos = build_inp_pos();

    // KQ_mask (mask for 1 head, it will be broadcasted to all heads)
    struct ggml_tensor * KQ_mask = build_inp_KQ_mask();

    for (int il = 0; il < n_layer; ++il) {
        struct ggml_tensor * inpSA = inpL;

        // norm
        cur = llm_build_norm(ctx0, inpL, hparams, model.layers[il].attn_norm, model.layers[il].attn_norm_b, LLM_NORM, cb, il);
        cb(cur, "attn_norm", il);

        // self-attention
        {
            auto [Qcur, Kcur, Vcur] = llm_build_mul_mat_qkv(gf, cur, model.layers[il].wq, nullptr,
                    model.layers[il].wk, nullptr,
                    model.layers[il].wv, nullptr, 0, il);
            Qcur = ggml_rope_ext(
                    ctx0, ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens), inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );
            cb(Qcur, "Qcur", il);

            Kcur = ggml_rope_ext(
                    ctx0, ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens), inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );
            cb(Kcur, "Kcur", il);

            cur = llm_build_kv(ctx0, lctx, kv_self, gf,
                    model.layers[il].wo, NULL,
                    Kcur, Vcur, Qcur, KQ_mask, n_tokens, kv_head, n_kv, 1.0f/sqrtf(float(n_embd_head)), cb, il);
        }

        if (il == n_layer - 1) {
            // skip computing output for unused tokens
            struct ggml_tensor * inp_out_ids = build_inp_out_ids();
            cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }

        struct ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        // feed-forward network
        cur = llm_build_ffn(ctx0, lctx, model.layers[il].ffn_norm, ffn_inp,
                model.layers[il].ffn_up,   NULL, NULL,
                model.layers[il].ffn_gate, NULL, NULL,
                model.layers[il].ffn_down, NULL, NULL,
                NULL,
                LLM_FFN_SILU, LLM_FFN_PAR, cb, il);
        cb(cur, "ffn_out", il);

        cur = ggml_add(ctx0, cur, ffn_inp);
        cur = lctx.cvec.apply_to(ctx0, cur, il);
        cb(cur, "l_out", il);

        // input for next layer
        inpL = cur;
    }

    cur = inpL;

    cur = llm_build_norm(ctx0, cur, hparams, model.output_norm, model.output_norm_b, LLM_NORM, cb, -1);
    cb(cur, "result_norm", -1);

    // lm_head
    cur = llm_build_lora_mm(lctx, ctx0, model.output, cur);
    cb(cur, "result_output", -1);

    ggml_build_forward_expand(gf, cur);

    return gf;
}

ggml_cgraph * llm_build_context::build_internlm2() {
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, llama_model_max_nodes(model), false);

    const int64_t n_embd_head = hparams.n_embd_head_v;
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);
    GGML_ASSERT(n_embd_head == hparams.n_rot);

    struct ggml_tensor * cur;
    struct ggml_tensor * inpL;

    inpL = llm_build_inp_embd(ctx0, lctx, hparams, batch, model.tok_embd, cb);

    // inp_pos - contains the positions
    struct ggml_tensor * inp_pos = build_inp_pos();

    // KQ_mask (mask for 1 head, it will be broadcasted to all heads)
    struct ggml_tensor * KQ_mask = build_inp_KQ_mask();

    for (int il = 0; il < n_layer; ++il) {
        struct ggml_tensor * inpSA = inpL;

        // norm
        cur = llm_build_norm(ctx0, inpL, hparams, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, cb, il);
        cb(cur, "attn_norm", il);

        // self-attention
        {
            auto [Qcur, Kcur, Vcur] = llm_build_mul_mat_qkv(gf, cur, model.layers[il].wq, model.layers[il].bq,
                    model.layers[il].wk, model.layers[il].bk,
                    model.layers[il].wv, model.layers[il].bv, 0.f, il);
            Qcur = ggml_rope_ext(
                    ctx0, ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens), inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );
            cb(Qcur, "Qcur", il);

            Kcur = ggml_rope_ext(
                    ctx0, ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens), inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );
            cb(Kcur, "Kcur", il);

            cur = llm_build_kv(ctx0, lctx, kv_self, gf,
                    model.layers[il].wo, model.layers[il].bo,
                    Kcur, Vcur, Qcur, KQ_mask, n_tokens, kv_head, n_kv, 1.0f/sqrtf(float(n_embd_head)), cb, il);
        }

        if (il == n_layer - 1) {
            // skip computing output for unused tokens
            struct ggml_tensor * inp_out_ids = build_inp_out_ids();
            cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }

        struct ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        // feed-forward network
        cur = llm_build_ffn(ctx0, lctx, model.layers[il].ffn_norm, ffn_inp,
                model.layers[il].ffn_up,   NULL, NULL,
                model.layers[il].ffn_gate, NULL, NULL,
                model.layers[il].ffn_down, NULL, NULL,
                NULL,
                LLM_FFN_SILU, LLM_FFN_PAR, cb, il);
        cb(cur, "ffn_out", il);

        cur = ggml_add(ctx0, cur, ffn_inp);
        cur = lctx.cvec.apply_to(ctx0, cur, il);
        cb(cur, "l_out", il);

        // input for next layer
        inpL = cur;
    }

    cur = inpL;

    cur = llm_build_norm(ctx0, cur, hparams, model.output_norm, NULL, LLM_NORM_RMS, cb, -1);
    cb(cur, "result_norm", -1);

    // lm_head
    cur = llm_build_lora_mm(lctx, ctx0, model.output, cur);
    cb(cur, "result_output", -1);

    ggml_build_forward_expand(gf, cur);

    return gf;
}

// ref: https://arxiv.org/abs/2203.03466
//      https://github.com/ggerganov/llama.cpp/issues/5276#issuecomment-1925774738
// based on the original build_llama() function
ggml_cgraph * llm_build_context::build_minicpm() {
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, llama_model_max_nodes(model), false);

    const int64_t n_embd_head = hparams.n_embd_head_v;
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);
    GGML_ASSERT(n_embd_head == hparams.n_rot);

    const int64_t n_embd = hparams.n_embd;
    //TODO: if the model varies, these parameters need to be read from the model
    const int64_t n_embd_base = 256;
    const float scale_embd  = 12.0f;
    const float scale_depth = 1.4f;

    struct ggml_tensor * cur;
    struct ggml_tensor * inpL;

    inpL = llm_build_inp_embd(ctx0, lctx, hparams, batch, model.tok_embd, cb);

    // scale the input embeddings
    inpL = ggml_scale(ctx0, inpL, scale_embd);
    cb(inpL, "inp_scaled", -1);

    // inp_pos - contains the positions
    struct ggml_tensor * inp_pos = build_inp_pos();

    // KQ_mask (mask for 1 head, it will be broadcasted to all heads)
    struct ggml_tensor * KQ_mask = build_inp_KQ_mask();

    for (int il = 0; il < n_layer; ++il) {
        struct ggml_tensor * inpSA = inpL;

        // norm
        cur = llm_build_norm(ctx0, inpL, hparams, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, cb, il);
        cb(cur, "attn_norm", il);

        // self-attention
        {
            auto [Qcur, Kcur, Vcur] = llm_build_mul_mat_qkv(gf, cur, model.layers[il].wq, model.layers[il].bq,
                    model.layers[il].wk, model.layers[il].bk,
                    model.layers[il].wv, model.layers[il].bv, 0.f, il);

            Qcur = ggml_rope_ext(
                    ctx0, ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens), inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );
            cb(Qcur, "Qcur", il);

            Kcur = ggml_rope_ext(
                    ctx0, ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens), inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );
            cb(Kcur, "Kcur", il);

            cur = llm_build_kv(ctx0, lctx, kv_self, gf,
                    model.layers[il].wo, model.layers[il].bo,
                    Kcur, Vcur, Qcur, KQ_mask, n_tokens, kv_head, n_kv, 1.0f/sqrtf(float(n_embd_head)), cb, il);
        }

        if (il == n_layer - 1) {
            // skip computing output for unused tokens
            struct ggml_tensor * inp_out_ids = build_inp_out_ids();
            cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }

        // scale_res - scale the hidden states for residual connection
        const float scale_res = scale_depth/sqrtf(float(n_layer));
        cur = ggml_scale(ctx0, cur, scale_res);
        cb(cur, "hidden_scaled", -1);

        struct ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        // feed-forward network
        {
            cur = llm_build_ffn(ctx0, lctx, model.layers[il].ffn_norm, ffn_inp,
                    model.layers[il].ffn_up,   NULL, NULL,
                    model.layers[il].ffn_gate, NULL, NULL,
                    model.layers[il].ffn_down, NULL, NULL,
                    NULL,
                    LLM_FFN_SILU, LLM_FFN_PAR, cb, il);
            cb(cur, "ffn_out", il);
        }

        // scale the hidden states for residual connection
        cur = ggml_scale(ctx0, cur, scale_res);
        cb(cur, "hidden_scaled_ffn", -1);

        cur = ggml_add(ctx0, cur, ffn_inp);
        cur = lctx.cvec.apply_to(ctx0, cur, il);
        cb(cur, "l_out", il);

        // input for next layer
        inpL = cur;
    }

    cur = inpL;

    cur = llm_build_norm(ctx0, cur, hparams, model.output_norm, NULL, LLM_NORM_RMS, cb, -1);
    cb(cur, "result_norm", -1);

    // lm_head scaling
    const float scale_lmhead = float(n_embd_base)/float(n_embd);
    cur = ggml_scale(ctx0, cur, scale_lmhead);
    cb(cur, "lmhead_scaling", -1);

    // lm_head
    cur = llm_build_lora_mm(lctx, ctx0, model.output, cur);
    cb(cur, "result_output", -1);

    ggml_build_forward_expand(gf, cur);

    return gf;
}

ggml_cgraph * llm_build_context::build_gemma() {
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, llama_model_max_nodes(model), false);

    const int64_t n_embd_head_k = hparams.n_embd_head_k;

    struct ggml_tensor * cur;
    struct ggml_tensor * inpL;

    inpL = llm_build_inp_embd(ctx0, lctx, hparams, batch, model.tok_embd, cb);

    inpL = ggml_scale(ctx0, inpL, sqrtf(n_embd));
    cb(inpL, "inp_scaled", -1);

    // inp_pos - contains the positions
    struct ggml_tensor * inp_pos = build_inp_pos();

    // KQ_mask (mask for 1 head, it will be broadcasted to all heads)
    struct ggml_tensor * KQ_mask = build_inp_KQ_mask();

    for (int il = 0; il < n_layer; ++il) {
        // norm
        cur = llm_build_norm(ctx0, inpL, hparams, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, cb, il);
        cb(cur, "attn_norm", il);

        // self-attention
        {
            auto [Qcur, Kcur, Vcur] = llm_build_mul_mat_qkv(gf, cur, model.layers[il].wq, nullptr,
                    model.layers[il].wk, nullptr,
                    model.layers[il].wv, nullptr, 0, il);
            Qcur = ggml_rope_ext(
                    ctx0, ggml_reshape_3d(ctx0, Qcur, n_embd_head_k, n_head,    n_tokens), inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow);
            cb(Qcur, "Qcur", il);

            Qcur = ggml_scale(ctx0, Qcur, 1.0f / sqrtf(float(n_embd_head_k)));
            cb(Qcur, "Qcur_scaled", il);

            Kcur = ggml_rope_ext(
                    ctx0, ggml_reshape_3d(ctx0, Kcur, n_embd_head_k, n_head_kv, n_tokens), inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow);
            cb(Kcur, "Kcur", il);

            cur = llm_build_kv(ctx0, lctx, kv_self, gf,
                    model.layers[il].wo, NULL,
                    Kcur, Vcur, Qcur, KQ_mask, n_tokens, kv_head, n_kv, 1.0f, cb, il);
        }

        if (il == n_layer - 1) {
            // skip computing output for unused tokens
            struct ggml_tensor * inp_out_ids = build_inp_out_ids();
            cur  = ggml_get_rows(ctx0,  cur, inp_out_ids);
            inpL = ggml_get_rows(ctx0, inpL, inp_out_ids);
        }

        struct ggml_tensor * sa_out = ggml_add(ctx0, cur, inpL);
        cb(sa_out, "sa_out", il);

        // feed-forward network
        {
            cur = llm_build_ffn(ctx0, lctx, model.layers[il].ffn_norm, sa_out,
                    model.layers[il].ffn_up,   NULL, NULL,
                    model.layers[il].ffn_gate, NULL, NULL,
                    model.layers[il].ffn_down, NULL, NULL,
                    NULL,
                    LLM_FFN_GELU, LLM_FFN_PAR, cb, il);
            cb(cur, "ffn_out", il);
        }

        cur = ggml_add(ctx0, cur, sa_out);
        cur = lctx.cvec.apply_to(ctx0, cur, il);
        cb(cur, "l_out", il);

        // input for next layer
        inpL = cur;
    }

    cur = inpL;

    cur = llm_build_norm(ctx0, cur, hparams, model.output_norm, NULL, LLM_NORM_RMS, cb, -1);
    cb(cur, "result_norm", -1);

    // lm_head
    cur = llm_build_lora_mm(lctx, ctx0, model.output, cur);
    cb(cur, "result_output", -1);

    ggml_build_forward_expand(gf, cur);

    return gf;
}

ggml_cgraph * llm_build_context::build_gemma2() {
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, llama_model_max_nodes(model), false);

    const int64_t n_embd_head_k = hparams.n_embd_head_k;

    struct ggml_tensor * cur;
    struct ggml_tensor * inpL;

    inpL = llm_build_inp_embd(ctx0, lctx, hparams, batch, model.tok_embd, cb);

    inpL = ggml_scale(ctx0, inpL, sqrtf(n_embd));
    cb(inpL, "inp_scaled", -1);

    // inp_pos - contains the positions
    struct ggml_tensor * inp_pos = build_inp_pos();

    // KQ_mask (mask for 1 head, it will be broadcasted to all heads)
    // gemma 2 requires different mask for layers using sliding window (SWA)
    struct ggml_tensor * KQ_mask     = build_inp_KQ_mask(true);
    struct ggml_tensor * KQ_mask_swa = build_inp_KQ_mask_swa(true);

    for (int il = 0; il < n_layer; ++il) {
        // (il % 2) layers use SWA
        struct ggml_tensor * KQ_mask_l = (il % 2 == 0) ? KQ_mask_swa : KQ_mask;

        // norm
        cur = llm_build_norm(ctx0, inpL, hparams, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, cb, il);
        cb(cur, "attn_norm", il);

        // self-attention
        {
            auto [Qcur, Kcur, Vcur] = llm_build_mul_mat_qkv(gf, cur, model.layers[il].wq, nullptr,
                    model.layers[il].wk, nullptr,
                    model.layers[il].wv, nullptr, 0, il);
            Qcur = ggml_rope_ext(
                    ctx0, ggml_reshape_3d(ctx0, Qcur, n_embd_head_k, n_head,    n_tokens), inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow);
            cb(Qcur, "Qcur", il);

            // ref: https://github.com/google/gemma_pytorch/commit/03e657582d17cb5a8617ebf333c1c16f3694670e
            switch (model.type) {
                case e_model::MODEL_2B:
                case e_model::MODEL_9B:  Qcur = ggml_scale(ctx0, Qcur, 1.0f / sqrtf(float(n_embd_head_k)));   break;
                case e_model::MODEL_27B: Qcur = ggml_scale(ctx0, Qcur, 1.0f / sqrtf(float(n_embd / n_head))); break;
                default: GGML_ABORT("fatal error");
            };
            cb(Qcur, "Qcur_scaled", il);

            Kcur = ggml_rope_ext(
                    ctx0, ggml_reshape_3d(ctx0, Kcur, n_embd_head_k, n_head_kv, n_tokens), inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow);
            cb(Kcur, "Kcur", il);

            cur = llm_build_kv(ctx0, lctx, kv_self, gf,
                    model.layers[il].wo, NULL,
                    Kcur, Vcur, Qcur, KQ_mask_l, n_tokens, kv_head, n_kv, 1.0f, cb, il, nullptr,
                    KQ_mask_l == KQ_mask_swa ? hparams.n_swa : 0);
        }

        cur = llm_build_norm(ctx0, cur, hparams, model.layers[il].attn_post_norm, NULL, LLM_NORM_RMS, cb, il);
        cb(cur, "attn_post_norm", il);

        if (il == n_layer - 1) {
            // skip computing output for unused tokens
            struct ggml_tensor * inp_out_ids = build_inp_out_ids();
            cur  = ggml_get_rows(ctx0,  cur, inp_out_ids);
            inpL = ggml_get_rows(ctx0, inpL, inp_out_ids);
        }

        struct ggml_tensor * sa_out = ggml_add(ctx0, cur, inpL);
        cb(sa_out, "sa_out", il);

        // feed-forward network
        {
            cur = llm_build_ffn(ctx0, lctx, model.layers[il].ffn_norm, sa_out,
                    model.layers[il].ffn_up,   NULL, NULL,
                    model.layers[il].ffn_gate, NULL, NULL,
                    model.layers[il].ffn_down, NULL, NULL,
                    NULL,
                    LLM_FFN_GELU, LLM_FFN_PAR, cb, il);
            cb(cur, "ffn_out", il);
        }

        cur = llm_build_norm(ctx0, cur, hparams, model.layers[il].ffn_post_norm, NULL, LLM_NORM_RMS, cb, -1);
        cb(cur, "ffn_post_norm", -1);

        cur = ggml_add(ctx0, cur, sa_out);
        cur = lctx.cvec.apply_to(ctx0, cur, il);
        cb(cur, "l_out", il);

        // input for next layer
        inpL = cur;
    }

    cur = inpL;

    cur = llm_build_norm(ctx0, cur, hparams, model.output_norm, NULL, LLM_NORM_RMS, cb, -1);
    cb(cur, "result_norm", -1);

    // lm_head
    cur = llm_build_lora_mm(lctx, ctx0, model.output, cur);

    // final logit soft-capping
    cur = ggml_softcap(ctx0, cur, 1.0f / hparams.f_final_logit_softcapping, hparams.f_final_logit_softcapping);
    //cur = ggml_scale(ctx0, cur, 1.0f / hparams.f_final_logit_softcapping);
    //cur = ggml_tanh(ctx0, cur);
    //cur = ggml_scale(ctx0, cur, hparams.f_final_logit_softcapping);

    cb(cur, "result_output", -1);

    ggml_build_forward_expand(gf, cur);

    return gf;
}

ggml_cgraph * llm_build_context::build_gemma3() {
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, llama_model_max_nodes(model), false);

    struct ggml_tensor * cur;
    struct ggml_tensor * inpL;

    inpL = llm_build_inp_embd(ctx0, lctx, hparams, batch, model.tok_embd, cb);

    // important: do not normalize weights for raw embeddings input (i.e. encoded image emdeddings)
    if (batch.token) {
        inpL = ggml_scale(ctx0, inpL, sqrtf(n_embd));
        cb(inpL, "inp_scaled", -1);
    }

    // inp_pos - contains the positions
    struct ggml_tensor * inp_pos = build_inp_pos();

    // KQ_mask (mask for 1 head, it will be broadcasted to all heads)
    // gemma3 requires different mask for layers using sliding window (SWA)
    struct ggml_tensor * KQ_mask     = build_inp_KQ_mask(true);
    struct ggml_tensor * KQ_mask_swa = build_inp_KQ_mask_swa(true);

    // "5-to-1 interleaved attention"
    // 5 layers of local attention followed by 1 layer of global attention
    static const int sliding_window_pattern = 6;

    ggml_tensor * rope_cache   = nullptr;
    ggml_tensor * rope_cache_l = nullptr;
    if (cparams.rope_cache && (rope_type == LLAMA_ROPE_TYPE_NEOX || rope_type == LLAMA_ROPE_TYPE_NORM)) {
        rope_cache = ggml_rope_cache(ctx0, inp_pos, nullptr, n_rot, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow);
        rope_cache_l = ggml_rope_cache(ctx0, inp_pos, nullptr, n_rot, n_rot, rope_type, n_ctx_orig, 10000.0f, 1.0f,
            ext_factor, attn_factor, beta_fast, beta_slow);
    }

    for (int il = 0; il < n_layer; ++il) {
        const bool is_sliding          = (il + 1) % sliding_window_pattern;
        const float freq_base_l        = is_sliding ? 10000.0f    : freq_base;
        const float freq_scale_l       = is_sliding ? 1.0f        : freq_scale;
        struct ggml_tensor * KQ_mask_l = is_sliding ? KQ_mask_swa : KQ_mask;

        // norm
        cur = llm_build_norm(ctx0, inpL, hparams, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, cb, il);
        cb(cur, "attn_norm", il);

        // self-attention
        {
            auto [Qcur, Kcur, Vcur] = llm_build_mul_mat_qkv(gf, cur,
                    model.layers[il].wqkv, nullptr,
                    model.layers[il].wqk, nullptr,
                    model.layers[il].wq, nullptr, model.layers[il].wk, nullptr, model.layers[il].wv, nullptr,
                    model.layers[il].attn_q_norm, model.layers[il].attn_k_norm, 0, il);

            if (rope_cache) {
                auto rcache = is_sliding ? rope_cache_l : rope_cache;
                Qcur = ggml_rope_fast(ctx0, Qcur, rcache);
                Kcur = ggml_rope_fast(ctx0, Kcur, rcache);
            } else {
                Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base_l, freq_scale_l,
                        ext_factor, attn_factor, beta_fast, beta_slow);

                Kcur = ggml_rope_ext(ctx0, Kcur, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base_l, freq_scale_l,
                        ext_factor, attn_factor, beta_fast, beta_slow);
            }
            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);

            cur = llm_build_kv(ctx0, lctx, kv_self, gf, model.layers[il].wo, NULL,
                    Kcur, Vcur, Qcur, KQ_mask_l, n_tokens, kv_head, n_kv, hparams.f_attention_scale, cb, il, nullptr,
                    KQ_mask_l == KQ_mask_swa ? hparams.n_swa : 0);
        }

        cur = llm_build_norm(ctx0, cur, hparams, model.layers[il].attn_post_norm, NULL, LLM_NORM_RMS, cb, il);
        cb(cur, "attn_post_norm", il);

        if (il == n_layer - 1) {
            // skip computing output for unused tokens
            struct ggml_tensor * inp_out_ids = build_inp_out_ids();
            cur  = ggml_get_rows(ctx0,  cur, inp_out_ids);
            inpL = ggml_get_rows(ctx0, inpL, inp_out_ids);
        }

        struct ggml_tensor * sa_out = ggml_add(ctx0, cur, inpL);
        cb(sa_out, "sa_out", il);

        // feed-forward network
        cur = llm_build_ffn(ctx0, lctx, model.layers[il].ffn_norm, sa_out,
                model.layers[il].ffn_up,   NULL, NULL,
                model.layers[il].ffn_gate, NULL, NULL,
                model.layers[il].ffn_down, NULL, NULL,
                NULL,
                LLM_FFN_GELU, LLM_FFN_PAR, cb, il);
        cb(cur, "ffn_out", il);

        cur = llm_build_norm(ctx0, cur, hparams, model.layers[il].ffn_post_norm, NULL, LLM_NORM_RMS, cb, -1);
        cb(cur, "ffn_post_norm", -1);

        cur = ggml_add(ctx0, cur, sa_out);
        cur = lctx.cvec.apply_to(ctx0, cur, il);
        cb(cur, "l_out", il);

        // input for next layer
        inpL = cur;
    }

    cur = inpL;

    cur = llm_build_norm(ctx0, cur, hparams, model.output_norm, NULL, LLM_NORM_RMS, cb, -1);
    cb(cur, "result_norm", -1);

    // lm_head
    cur = llm_build_lora_mm(lctx, ctx0, model.output, cur);

    cb(cur, "result_output", -1);

    ggml_build_forward_expand(gf, cur);

    return gf;
}

ggml_cgraph * llm_build_context::build_starcoder2() {
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, llama_model_max_nodes(model), false);

    const int64_t n_embd_head = hparams.n_embd_head_v;
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);
    GGML_ASSERT(n_embd_head == hparams.n_rot);

    struct ggml_tensor * cur;
    struct ggml_tensor * inpL;

    inpL = llm_build_inp_embd(ctx0, lctx, hparams, batch, model.tok_embd, cb);

    // inp_pos - contains the positions
    struct ggml_tensor * inp_pos = build_inp_pos();

    // KQ_mask (mask for 1 head, it will be broadcasted to all heads)
    struct ggml_tensor * KQ_mask = build_inp_KQ_mask();

    for (int il = 0; il < n_layer; ++il) {
        struct ggml_tensor * inpSA = inpL;

        // norm
        cur = llm_build_norm(ctx0, inpL, hparams, model.layers[il].attn_norm, model.layers[il].attn_norm_b, LLM_NORM, cb, il);
        cb(cur, "attn_norm", il);

        // self-attention
        {
            auto [Qcur, Kcur, Vcur] = llm_build_mul_mat_qkv(gf, cur, model.layers[il].wq, model.layers[il].bq,
                    model.layers[il].wk, model.layers[il].bk,
                    model.layers[il].wv, model.layers[il].bv, 0.f, il);
            Qcur = ggml_rope_ext(
                    ctx0, ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens), inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );
            cb(Qcur, "Qcur", il);

            Kcur = ggml_rope_ext(
                    ctx0, ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens), inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );
            cb(Kcur, "Kcur", il);

            cur = llm_build_kv(ctx0, lctx, kv_self, gf,
                    model.layers[il].wo, model.layers[il].bo,
                    Kcur, Vcur, Qcur, KQ_mask, n_tokens, kv_head, n_kv, 1.0f/sqrtf(float(n_embd_head)), cb, il);
        }

        if (il == n_layer - 1) {
            // skip computing output for unused tokens
            struct ggml_tensor * inp_out_ids = build_inp_out_ids();
            cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }

        struct ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        // feed-forward network
        cur = llm_build_ffn(ctx0, lctx, model.layers[il].ffn_norm, ffn_inp,
                model.layers[il].ffn_up,   model.layers[il].ffn_up_b,   NULL,
                NULL,                      NULL,                        NULL,
                model.layers[il].ffn_down, model.layers[il].ffn_down_b, NULL,
                NULL,
                LLM_FFN_GELU, LLM_FFN_SEQ, cb, il);
        cb(cur, "ffn_out", il);

        cur = ggml_add(ctx0, cur, ffn_inp);
        cur = lctx.cvec.apply_to(ctx0, cur, il);
        cb(cur, "l_out", il);

        // input for next layer
        inpL = cur;
    }

    cur = inpL;

    cur = llm_build_norm(ctx0, cur, hparams, model.output_norm, model.output_norm_b, LLM_NORM, cb, -1);
    cb(cur, "result_norm", -1);

    // lm_head
    cur = llm_build_lora_mm(lctx, ctx0, model.output, cur);
    cb(cur, "result_output", -1);

    ggml_build_forward_expand(gf, cur);

    return gf;
}

ggml_cgraph * llm_build_context::build_mamba() {
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, llama_model_max_nodes(model), false);

    const int64_t d_model = n_embd;
    const int64_t d_conv  = hparams.ssm_d_conv;
    const int64_t d_inner = hparams.ssm_d_inner;
    GGML_ASSERT(2 * d_model == d_inner);
    const int64_t d_state = hparams.ssm_d_state;
    const int64_t dt_rank = hparams.ssm_dt_rank;

    struct ggml_tensor * cur;
    struct ggml_tensor * inpL;

    // {n_embd, n_tokens}
    inpL = llm_build_inp_embd(ctx0, lctx, hparams, batch, model.tok_embd, cb);

    struct ggml_tensor * state_mask = build_inp_s_mask();
    struct ggml_tensor * state_seq  = build_inp_s_seq();

    for (int il = 0; il < n_layer; ++il) {
        // (ab)using the KV cache to store the states
        struct ggml_tensor * conv_states = ggml_reshape_2d(ctx0, kv_self.k_l[il], hparams.n_embd_k_s(), kv_self.size);
        struct ggml_tensor * ssm_states  = ggml_reshape_2d(ctx0, kv_self.v_l[il], hparams.n_embd_v_s(), kv_self.size);

        // clear states of sequences which are starting at the beginning of this batch
        {
            conv_states = ggml_mul(ctx0,
                    ggml_view_2d(ctx0, conv_states, conv_states->ne[0], n_kv, conv_states->nb[1], kv_head*conv_states->nb[1]),
                    state_mask);
            ssm_states  = ggml_mul(ctx0,
                    ggml_view_2d(ctx0, ssm_states, ssm_states->ne[0], n_kv, ssm_states->nb[1], kv_head*ssm_states->nb[1]),
                    state_mask);
        }

        conv_states = ggml_reshape_3d(ctx0, conv_states, d_conv - 1, d_inner, n_kv);
        ssm_states  = ggml_reshape_3d(ctx0,  ssm_states,    d_state, d_inner, n_kv);

        // norm
        cur = llm_build_norm(ctx0, inpL, hparams, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, cb, il);
        cb(cur, "attn_norm", il);

        // {n_embd, 2*d_inner} * {n_embd, n_tokens} => {2*d_inner, n_tokens}
        struct ggml_tensor * xz = llm_build_lora_mm(lctx, ctx0, model.layers[il].ssm_in, cur);
        // split the above in two
        // => {d_inner, n_tokens}
        struct ggml_tensor * x = ggml_view_2d(ctx0, xz, d_inner, xz->ne[1], xz->nb[1], 0);
        struct ggml_tensor * z = ggml_view_2d(ctx0, xz, d_inner, xz->ne[1], xz->nb[1], ggml_element_size(xz)*d_inner);

        // conv
        {
            // Custom operator which is needed only to ease simultaneous sequence processing.
            // For a single sequence, the equivalent is to concatenate the columns of conv_states and x,
            // then make a self-overlapping view of that over d_conv columns at each stride in the 3rd dimension,
            // then element-wise multiply that with the conv1d weigth,
            // then sum the elements of each row,
            // (the last two steps are a dot product over rows (also doable with mul_mat))
            // then permute away the ne[0] dimension,
            // and then you're left with the resulting x tensor.
            // The new conv_states is the last (d_conv - 1) columns
            // of the last 3rd dimensional "layer" of the self-overlapping view.
            // For simultaneous sequences, it's more complicated.
            struct ggml_tensor * x_conv = ggml_ssm_conv(ctx0, conv_states, x, model.layers[il].ssm_conv1d, state_seq);

            // store last (d_conv - 1) columns of the conv_state part of x_conv back into the KV cache
            ggml_build_forward_expand(gf,
                    ggml_cpy(ctx0,
                        ggml_view_2d(ctx0, x_conv, d_conv - 1, d_inner*n_kv, d_conv*ggml_element_size(x_conv), (1+d_inner*n_tokens)*ggml_element_size(x_conv)),
                        ggml_view_1d(ctx0, kv_self.k_l[il], (d_conv - 1)*(d_inner)*(n_kv), kv_head*(d_conv - 1)*(d_inner)*ggml_element_size(x_conv))));

            // extract x from x_conv
            x = ggml_view_2d(ctx0, x_conv, d_inner, n_tokens, d_inner*ggml_element_size(x_conv), 0);

            // bias
            x = ggml_add(ctx0, x, model.layers[il].ssm_conv1d_b);

            x = ggml_silu(ctx0, x);
        }

        // ssm
        {
            // {d_inner, dt_rank + 2*d_state} * {d_inner, n_tokens} => {dt_rank + 2*d_state, n_tokens}
            struct ggml_tensor * x_db = llm_build_lora_mm(lctx, ctx0, model.layers[il].ssm_x, x);
            // split
            struct ggml_tensor * dt = ggml_view_2d(ctx0, x_db, dt_rank, n_tokens, x_db->nb[1], 0);
            struct ggml_tensor * B  = ggml_view_2d(ctx0, x_db, d_state, n_tokens, x_db->nb[1], ggml_element_size(x_db)*dt_rank);
            struct ggml_tensor * C  = ggml_view_2d(ctx0, x_db, d_state, n_tokens, x_db->nb[1], ggml_element_size(x_db)*(dt_rank+d_state));

            // {dt_rank, d_inner} * {dt_rank, n_tokens} => {d_inner, n_tokens}
            dt = llm_build_lora_mm(lctx, ctx0, model.layers[il].ssm_dt, dt);
            dt = ggml_add(ctx0, dt, model.layers[il].ssm_dt_b);

            // Custom operator to optimize the parallel associative scan
            // as described in the Annex D of the Mamba paper.
            // => {d_inner, n_tokens} and {d_state, d_inner, n_kv} combined,
            // because only a single tensor can be returned.
            struct ggml_tensor * y_ssm_states = ggml_ssm_scan(ctx0, ssm_states, x, dt, model.layers[il].ssm_a, B, C, state_seq);

            // store last states (the second part of y_ssm_states)
            ggml_build_forward_expand(gf,
                    ggml_cpy(ctx0,
                        ggml_view_1d(ctx0, y_ssm_states, d_state*d_inner*n_kv, d_inner*n_tokens*ggml_element_size(y_ssm_states)),
                        ggml_view_1d(ctx0, kv_self.v_l[il], d_state*d_inner*n_kv, kv_head*d_state*d_inner*ggml_element_size(ssm_states))));

            struct ggml_tensor * y = ggml_view_2d(ctx0, y_ssm_states, d_inner, n_tokens, d_inner*ggml_element_size(y_ssm_states), 0);

            if (il == n_layer - 1) {
                // skip computing output for unused tokens
                struct ggml_tensor * inp_out_ids = build_inp_out_ids();
                x    = ggml_get_rows(ctx0,    x, inp_out_ids);
                y    = ggml_get_rows(ctx0,    y, inp_out_ids);
                z    = ggml_get_rows(ctx0,    z, inp_out_ids);
                inpL = ggml_get_rows(ctx0, inpL, inp_out_ids);
            }

            // {d_inner, n_tokens} * {d_inner} => {d_inner, n_tokens}
            y = ggml_add(ctx0, y, ggml_mul(ctx0, x, model.layers[il].ssm_d));
            y = ggml_mul(ctx0, y, ggml_silu(ctx0, z));

            // {d_inner, n_embd} * {d_inner, n_tokens} => {n_embd, n_tokens}
            cur = llm_build_lora_mm(lctx, ctx0, model.layers[il].ssm_out, y);
        }

        // residual
        cur = ggml_add(ctx0, cur, inpL);
        cur = lctx.cvec.apply_to(ctx0, cur, il);
        cb(cur, "l_out", il);

        // input for next layer
        inpL = cur;
    }

    // final rmsnorm
    cur = llm_build_norm(ctx0, inpL, hparams, model.output_norm, NULL, LLM_NORM_RMS, cb, -1);
    cb(cur, "result_norm", -1);

    // lm_head
    cur = llm_build_lora_mm(lctx, ctx0, model.output, cur);
    cb(cur, "result_output", -1);

    ggml_build_forward_expand(gf, cur);

    return gf;
}

ggml_cgraph * llm_build_context::build_command_r() {

    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, llama_model_max_nodes(model), false);

    const int64_t n_embd_head = hparams.n_embd_head_v;
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);
    const float f_logit_scale = hparams.f_logit_scale;

    struct ggml_tensor * cur;
    struct ggml_tensor * inpL;

    inpL = llm_build_inp_embd(ctx0, lctx, hparams, batch, model.tok_embd, cb);

    // inp_pos - contains the positions
    struct ggml_tensor * inp_pos = build_inp_pos();

    // KQ_mask (mask for 1 head, it will be broadcasted to all heads)
    struct ggml_tensor * KQ_mask = build_inp_KQ_mask();

    for (int il = 0; il < n_layer; ++il) {

        // norm
        cur = llm_build_norm(ctx0, inpL, hparams, model.layers[il].attn_norm, NULL, LLM_NORM, cb, il);
        cb(cur, "attn_norm", il);
        struct ggml_tensor * ffn_inp = cur;

        // self-attention
        {
            auto [Qcur, Kcur, Vcur] = llm_build_mul_mat_qkv(gf, cur, model.layers[il].wq, model.layers[il].bq,
                    model.layers[il].wk, model.layers[il].bk,
                    model.layers[il].wv, model.layers[il].bv, 0.f, il);

            if (model.layers[il].attn_q_norm) {
                Qcur = ggml_view_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens,
                        ggml_element_size(Qcur) * n_embd_head,
                        ggml_element_size(Qcur) * n_embd_head * n_head,
                        0);
                cb(Qcur, "Qcur", il);
                Kcur = ggml_view_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens,
                        ggml_element_size(Kcur) * n_embd_head,
                        ggml_element_size(Kcur) * n_embd_head * n_head_kv,
                        0);
                cb(Kcur, "Kcur", il);

                Qcur = llm_build_norm(ctx0, Qcur, hparams, model.layers[il].attn_q_norm, NULL, LLM_NORM, cb, il);
                cb(Qcur, "Qcur", il);

                Kcur = llm_build_norm(ctx0, Kcur, hparams, model.layers[il].attn_k_norm, NULL, LLM_NORM, cb, il);
                cb(Kcur, "Kcur", il);
            }

            Qcur = ggml_rope_ext(
                    ctx0, ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens), inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );
            cb(Qcur, "Qcur", il);

            Kcur = ggml_rope_ext(
                    ctx0, ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens), inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );
            cb(Kcur, "Kcur", il);

            cur = llm_build_kv(ctx0, lctx, kv_self, gf,
                    model.layers[il].wo, model.layers[il].bo,
                    Kcur, Vcur, Qcur, KQ_mask, n_tokens, kv_head, n_kv, 1.0f/sqrtf(float(n_embd_head)), cb, il);
        }

        if (il == n_layer - 1) {
            // skip computing output for unused tokens
            struct ggml_tensor * inp_out_ids = build_inp_out_ids();
            cur     = ggml_get_rows(ctx0,     cur, inp_out_ids);
            inpL    = ggml_get_rows(ctx0,    inpL, inp_out_ids);
            ffn_inp = ggml_get_rows(ctx0, ffn_inp, inp_out_ids);
        }

        struct ggml_tensor * attn_out = cur;

        // feed-forward network
        {
            cur = llm_build_ffn(ctx0, lctx, nullptr, ffn_inp,
                    model.layers[il].ffn_up,   NULL, NULL,
                    model.layers[il].ffn_gate, NULL, NULL,
                    model.layers[il].ffn_down, NULL, NULL,
                    NULL,
                    LLM_FFN_SILU, LLM_FFN_PAR, cb, il);
            cb(cur, "ffn_out", il);
        }

        // add together residual + FFN + self-attention
        cur = ggml_add(ctx0, cur, inpL);
        cur = ggml_add(ctx0, cur, attn_out);
        cur = lctx.cvec.apply_to(ctx0, cur, il);
        cb(cur, "l_out", il);

        // input for next layer
        inpL = cur;
    }

    cur = inpL;

    cur = llm_build_norm(ctx0, cur, hparams, model.output_norm, NULL, LLM_NORM, cb, -1);
    cb(cur, "result_norm", -1);

    // lm_head
    cur = llm_build_lora_mm(lctx, ctx0, model.output, cur);

    if (f_logit_scale) {
        cur = ggml_scale(ctx0, cur, f_logit_scale);
    }

    cb(cur, "result_output", -1);

    ggml_build_forward_expand(gf, cur);

    return gf;

}

// ref: https://allenai.org/olmo
// based on the original build_llama() function, changes:
//   * non-parametric layer norm
//   * clamp qkv
//   * removed bias
//   * removed MoE
ggml_cgraph * llm_build_context::build_olmo() {
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, llama_model_max_nodes(model), false);

    // mutable variable, needed during the last layer of the computation to skip unused tokens
    int32_t n_tokens = this->n_tokens;

    const int64_t n_embd_head = hparams.n_embd_head_v;
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);
    GGML_ASSERT(n_embd_head == hparams.n_rot);

    struct ggml_tensor * cur;
    struct ggml_tensor * inpL;

    inpL = llm_build_inp_embd(ctx0, lctx, hparams, batch, model.tok_embd, cb);

    // inp_pos - contains the positions
    struct ggml_tensor * inp_pos = build_inp_pos();

    // KQ_mask (mask for 1 head, it will be broadcasted to all heads)
    struct ggml_tensor * KQ_mask = build_inp_KQ_mask();

    for (int il = 0; il < n_layer; ++il) {
        struct ggml_tensor * inpSA = inpL;

        // norm
        cur = llm_build_norm(ctx0, inpL, hparams, NULL, NULL, LLM_NORM, cb, il);
        cb(cur, "attn_norm", il);

        // self-attention
        {
            // compute Q and K and RoPE them
            struct ggml_tensor * Qcur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wq, cur);
            cb(Qcur, "Qcur", il);
            if (hparams.f_clamp_kqv > 0.0f) {
                Qcur = ggml_clamp(ctx0, Qcur, -hparams.f_clamp_kqv, hparams.f_clamp_kqv);
                cb(Qcur, "Qcur", il);
            }

            struct ggml_tensor * Kcur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wk, cur);
            cb(Kcur, "Kcur", il);
            if (hparams.f_clamp_kqv > 0.0f) {
                Kcur = ggml_clamp(ctx0, Kcur, -hparams.f_clamp_kqv, hparams.f_clamp_kqv);
                cb(Kcur, "Kcur", il);
            }

            struct ggml_tensor * Vcur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wv, cur);
            cb(Vcur, "Vcur", il);
            if (hparams.f_clamp_kqv > 0.0f) {
                Vcur = ggml_clamp(ctx0, Vcur, -hparams.f_clamp_kqv, hparams.f_clamp_kqv);
                cb(Vcur, "Vcur", il);
            }

            Qcur = ggml_rope_ext(
                    ctx0, ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens), inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );
            cb(Qcur, "Qcur", il);

            Kcur = ggml_rope_ext(
                    ctx0, ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens), inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );
            cb(Kcur, "Kcur", il);

            cur = llm_build_kv(ctx0, lctx, kv_self, gf,
                    model.layers[il].wo, nullptr,
                    Kcur, Vcur, Qcur, KQ_mask, n_tokens, kv_head, n_kv, 1.0f/sqrtf(float(n_embd_head)), cb, il);
        }

        if (il == n_layer - 1) {
            // skip computing output for unused tokens
            struct ggml_tensor * inp_out_ids = build_inp_out_ids();
            n_tokens = n_outputs;
            cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }

        struct ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        // feed-forward network
        cur = llm_build_norm(ctx0, ffn_inp, hparams, NULL, NULL, LLM_NORM, cb, il);
        cb(cur, "ffn_norm", il);

        cur = llm_build_ffn(ctx0, lctx, nullptr, cur,
                model.layers[il].ffn_up,   NULL, NULL,
                model.layers[il].ffn_gate, NULL, NULL,
                model.layers[il].ffn_down, NULL, NULL,
                NULL,
                LLM_FFN_SILU, LLM_FFN_PAR, cb, il);
        cb(cur, "ffn_out", il);

        cur = ggml_add(ctx0, cur, ffn_inp);
        cb(cur, "ffn_out", il);

        cur = lctx.cvec.apply_to(ctx0, cur, il);
        cb(cur, "l_out", il);

        // input for next layer
        inpL = cur;
    }

    cur = inpL;

    cur = llm_build_norm(ctx0, cur, hparams, NULL, NULL, LLM_NORM, cb, -1);
    cb(cur, "result_norm", -1);

    // lm_head
    cur = llm_build_lora_mm(lctx, ctx0, model.output, cur);
    cb(cur, "result_output", -1);

    ggml_build_forward_expand(gf, cur);

    return gf;
}

ggml_cgraph * llm_build_context::build_openelm() {
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, llama_model_max_nodes(model), false);

    const int64_t n_embd_head = hparams.n_embd_head_v;
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);

    struct ggml_tensor * cur;
    struct ggml_tensor * inpL;
    inpL = llm_build_inp_embd(ctx0, lctx, hparams, batch, model.tok_embd, cb);

    // inp_pos - contains the positions
    struct ggml_tensor * inp_pos = build_inp_pos();

    // KQ_mask (mask for 1 head, it will be broadcasted to all heads)
    struct ggml_tensor * KQ_mask = build_inp_KQ_mask();

    for (int il = 0; il < n_layer; ++il) {
        const int64_t n_head    = hparams.n_head(il);
        const int64_t n_head_kv = hparams.n_head_kv(il);
        const int64_t n_head_qkv = 2*n_head_kv + n_head;

        cur = inpL;
        struct ggml_tensor * residual = cur;

        // norm
        cur = llm_build_norm(ctx0, inpL, hparams, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, cb, il);
        cb(cur, "attn_norm", il);

        // self-attention
        {
            cur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wqkv, cur);
            cb(cur, "wqkv", il);

            cur = ggml_reshape_3d(ctx0, cur, n_embd_head_k, n_head_qkv, n_tokens);

            struct ggml_tensor * Qcur = ggml_cont(ctx0, ggml_view_3d(ctx0, cur, n_embd_head, n_head, n_tokens, cur->nb[1], cur->nb[2], 0));
            cb(Qcur, "Qcur", il);

            struct ggml_tensor * Kcur = ggml_cont(ctx0, ggml_view_3d(ctx0, cur, n_embd_head, n_head_kv, n_tokens, cur->nb[1], cur->nb[2], cur->nb[1]*n_head));
            cb(Kcur, "Kcur", il);

            struct ggml_tensor * Vcur = ggml_cont(ctx0, ggml_view_3d(ctx0, cur, n_embd_head, n_head_kv, n_tokens, cur->nb[1], cur->nb[2], cur->nb[1]*(n_head+n_head_kv)));
            cb(Vcur, "Vcur", il);

            Qcur = llm_build_norm(ctx0, Qcur, hparams, model.layers[il].attn_q_norm, NULL, LLM_NORM_RMS, cb, il);
            cb(Qcur, "Qcur", il);

            Kcur = llm_build_norm(ctx0, Kcur, hparams, model.layers[il].attn_k_norm, NULL, LLM_NORM_RMS, cb, il);
            cb(Kcur, "Kcur", il);

            Qcur = ggml_rope_ext(
                    ctx0, Qcur, inp_pos, NULL, n_rot, rope_type, n_ctx_orig,
                    freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow
                    );
            cb(Qcur, "Qcur", il);

            Kcur = ggml_rope_ext(
                    ctx0, Kcur, inp_pos, NULL, n_rot, rope_type, n_ctx_orig,
                    freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow
                    );
            cb(Kcur, "Kcur", il);

            Vcur = ggml_reshape_2d(ctx0, Vcur, n_embd_head * n_head_kv, n_tokens);
            cb(Qcur, "Vcur", il);

            cur = llm_build_kv(ctx0, lctx, kv_self, gf,
                    model.layers[il].wo, NULL,
                    Kcur, Vcur, Qcur, KQ_mask, n_tokens, kv_head, n_kv, 1.0f/sqrtf(float(n_embd_head)), cb, il);
        }

        if (il == n_layer - 1) {
            // skip computing output for unused tokens
            struct ggml_tensor * inp_out_ids = build_inp_out_ids();
            residual = ggml_get_rows(ctx0, residual, inp_out_ids);
            cur = ggml_get_rows(ctx0, cur, inp_out_ids);
        }

        struct ggml_tensor * ffn_inp = ggml_add(ctx0, residual, cur);
        cb(ffn_inp, "ffn_inp", il);

        // feed-forward network
        {
            cur = llm_build_ffn(ctx0, lctx, model.layers[il].ffn_norm, ffn_inp,
                    model.layers[il].ffn_up,   NULL, NULL,
                    model.layers[il].ffn_gate, NULL, NULL,
                    model.layers[il].ffn_down, NULL, NULL,
                    NULL,
                    LLM_FFN_SILU, LLM_FFN_PAR, cb, il);
            cb(cur, "ffn_out", il);
        }

        cur = ggml_add(ctx0, cur, ffn_inp);
        cur = lctx.cvec.apply_to(ctx0, cur, il);
        cb(cur, "l_out", il);

        inpL = cur;
    }

    cur = inpL;

    // norm
    cur = llm_build_norm(ctx0, cur, hparams, model.output_norm, NULL, LLM_NORM_RMS, cb, -1);
    cb(cur, "result_norm", -1);

    cur = llm_build_lora_mm(lctx, ctx0, model.output, cur);
    cb(cur, "result_output", -1);

    ggml_build_forward_expand(gf, cur);

    return gf;
}

ggml_cgraph * llm_build_context::build_gptneox() {
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, llama_model_max_nodes(model), false);

    const int64_t n_embd_head = hparams.n_embd_head_v;
    const int64_t n_embd_gqa  = hparams.n_embd_v_gqa();
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);

    struct ggml_tensor * cur;
    struct ggml_tensor * inpL;

    inpL = llm_build_inp_embd(ctx0, lctx, hparams, batch, model.tok_embd, cb);

    // inp_pos - contains the positions
    struct ggml_tensor * inp_pos = build_inp_pos();

    // KQ_mask (mask for 1 head, it will be broadcasted to all heads)
    struct ggml_tensor * KQ_mask = build_inp_KQ_mask();

    for (int il = 0; il < n_layer; ++il) {
        cur = llm_build_norm(ctx0, inpL, hparams, model.layers[il].attn_norm, model.layers[il].attn_norm_b, LLM_NORM, cb, il);
        cb(cur, "attn_norm", il);

        // self-attention
        {
            cur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wqkv, cur);
            cb(cur, "wqkv", il);

            cur = ggml_add(ctx0, cur, model.layers[il].bqkv);
            cb(cur, "bqkv", il);

            struct ggml_tensor * Qcur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd,     n_tokens, cur->nb[1], 0*sizeof(float)*(n_embd)));
            struct ggml_tensor * Kcur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd_gqa, n_tokens, cur->nb[1], 1*sizeof(float)*(n_embd)));
            struct ggml_tensor * Vcur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd_gqa, n_tokens, cur->nb[1], 1*sizeof(float)*(n_embd + n_embd_gqa)));

            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            Qcur = ggml_rope_ext(
                    ctx0, ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens), inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );
            cb(Qcur, "Qcur", il);

            Kcur = ggml_rope_ext(
                    ctx0, ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens), inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );
            cb(Kcur, "Kcur", il);

            cur = llm_build_kv(ctx0, lctx, kv_self, gf,
                    model.layers[il].wo, model.layers[il].bo,
                    Kcur, Vcur, Qcur, KQ_mask, n_tokens, kv_head, n_kv, 1.0f/sqrtf(float(n_embd_head)), cb, il);
        }

        if (il == n_layer - 1) {
            // skip computing output for unused tokens
            struct ggml_tensor * inp_out_ids = build_inp_out_ids();
            cur  = ggml_get_rows(ctx0,  cur, inp_out_ids);
            inpL = ggml_get_rows(ctx0, inpL, inp_out_ids);
        }

        // ffn
        if (hparams.use_par_res) {
            // attention and ffn are computed in parallel
            // x = x + attn(ln1(x)) + ffn(ln2(x))

            struct ggml_tensor * attn_out = cur;

            cur = llm_build_norm(ctx0, inpL, hparams, model.layers[il].ffn_norm, model.layers[il].ffn_norm_b, LLM_NORM, cb, il);
            cb(cur, "ffn_norm", il);

            cur = llm_build_ffn(ctx0, lctx, nullptr, cur,
                    model.layers[il].ffn_up,   model.layers[il].ffn_up_b,   NULL,
                    NULL,                      NULL,                        NULL,
                    model.layers[il].ffn_down, model.layers[il].ffn_down_b, NULL,
                    NULL,
                    LLM_FFN_GELU, LLM_FFN_SEQ, cb, il);
            cb(cur, "ffn_out", il);

            cur = ggml_add(ctx0, cur, inpL);
            cb(cur, "ffn_out", il);

            cur = ggml_add(ctx0, cur, attn_out);
            cur = lctx.cvec.apply_to(ctx0, cur, il);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        } else {
            // attention and ffn are computed sequentially
            // x = x + attn(ln1(x))
            // x = x + ffn(ln2(x))

            struct ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpL);
            cb(ffn_inp, "ffn_inp", il);

            cur = llm_build_norm(ctx0, ffn_inp, hparams, model.layers[il].ffn_norm, model.layers[il].ffn_norm_b, LLM_NORM, cb, il);
            cb(cur, "ffn_norm", il);

            cur = llm_build_ffn(ctx0, lctx, nullptr, cur,
                    model.layers[il].ffn_up,   model.layers[il].ffn_up_b,   NULL,
                    NULL,                      NULL,                        NULL,
                    model.layers[il].ffn_down, model.layers[il].ffn_down_b, NULL,
                    NULL,
                    LLM_FFN_GELU, LLM_FFN_SEQ, cb, il);
            cb(cur, "ffn_out", il);

            cur = ggml_add(ctx0, cur, ffn_inp);
            cur = lctx.cvec.apply_to(ctx0, cur, il);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }
    }

    cur = llm_build_norm(ctx0, inpL, hparams, model.output_norm, model.output_norm_b, LLM_NORM, cb, -1);
    cb(cur, "result_norm", -1);

    cur = llm_build_lora_mm(lctx, ctx0, model.output, cur);
    cb(cur, "result_output", -1);

    ggml_build_forward_expand(gf, cur);

    return gf;
}

ggml_cgraph * llm_build_context::build_arctic() {
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, llama_model_max_nodes(model), false);

    // mutable variable, needed during the last layer of the computation to skip unused tokens
    int32_t n_tokens = this->n_tokens;

    const int64_t n_embd_head = hparams.n_embd_head_v;
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);
    GGML_ASSERT(n_embd_head == hparams.n_rot);

    struct ggml_tensor * cur;
    struct ggml_tensor * inpL;

    inpL = llm_build_inp_embd(ctx0, lctx, hparams, batch, model.tok_embd, cb);

    // inp_pos - contains the positions
    struct ggml_tensor * inp_pos = build_inp_pos();

    // KQ_mask (mask for 1 head, it will be broadcasted to all heads)
    struct ggml_tensor * KQ_mask = build_inp_KQ_mask();

    for (int il = 0; il < n_layer; ++il) {
        struct ggml_tensor * inpSA = inpL;

        // norm
        cur = llm_build_norm(ctx0, inpL, hparams, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, cb, il);
        cb(cur, "attn_norm", il);

        // self-attention
        {
            auto [Qcur, Kcur, Vcur] = llm_build_mul_mat_qkv(gf, cur, model.layers[il].wq, nullptr,
                    model.layers[il].wk, nullptr,
                    model.layers[il].wv, nullptr, 0, il);

            Qcur = ggml_rope_ext(
                    ctx0, ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens), inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );
            cb(Qcur, "Qcur", il);

            Kcur = ggml_rope_ext(
                    ctx0, ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens), inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );
            cb(Kcur, "Kcur", il);

            cur = llm_build_kv(ctx0, lctx, kv_self, gf,
                    model.layers[il].wo, NULL,
                    Kcur, Vcur, Qcur, KQ_mask, n_tokens, kv_head, n_kv, 1.0f/sqrtf(float(n_embd_head)), cb, il);
        }

        if (il == n_layer - 1) {
            // skip computing output for unused tokens
            struct ggml_tensor * inp_out_ids = build_inp_out_ids();
            n_tokens = n_outputs;
            cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }

        struct ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        // feed-forward network
        cur = llm_build_ffn(ctx0, lctx, model.layers[il].ffn_norm, ffn_inp,
                model.layers[il].ffn_up,   NULL, NULL,
                model.layers[il].ffn_gate, NULL, NULL,
                model.layers[il].ffn_down, NULL, NULL,
                NULL,
                LLM_FFN_SILU, LLM_FFN_PAR, cb, il);
        cb(cur, "ffn_out", il);

        struct ggml_tensor * ffn_out = ggml_add(ctx0, cur, ffn_inp);
        cb(ffn_out, "ffn_out", il);

        // MoE
        cur = llm_build_norm(ctx0, inpSA, hparams, model.layers[il].ffn_norm_exps, NULL, LLM_NORM_RMS, cb, il);
        cb(cur, "ffn_norm_exps", il);

        cur = llm_build_moe_ffn(ctx0, lctx, cur,
                model.layers[il].ffn_gate_inp,
                model.layers[il].ffn_up_exps,
                model.layers[il].ffn_gate_exps,
                model.layers[il].ffn_down_exps,
                nullptr,
                n_expert, n_expert_used,
                LLM_FFN_SILU, true,
                false, 0.0,
                LLM_EXPERT_GATING_FUNC_SOFTMAX,
                cb, il, gf);
        cb(cur, "ffn_moe_out", il);

        cur = ggml_add(ctx0, cur, ffn_out);
        cb(cur, "ffn_out", il);

        cur = lctx.cvec.apply_to(ctx0, cur, il);
        cb(cur, "l_out", il);

        // input for next layer
        inpL = cur;
    }

    cur = inpL;

    cur = llm_build_norm(ctx0, cur, hparams, model.output_norm, NULL, LLM_NORM_RMS, cb, -1);
    cb(cur, "result_norm", -1);

    // lm_head
    cur = llm_build_lora_mm(lctx, ctx0, model.output, cur);
    cb(cur, "result_output", -1);

    ggml_build_forward_expand(gf, cur);

    return gf;
}

ggml_cgraph * llm_build_context::build_deepseek2() {
#ifdef GGML_USE_VULKAN
    constexpr bool use_f32_attn_precision = true;
#else
    constexpr bool use_f32_attn_precision = false;
#endif
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, llama_model_max_nodes(model), false);

    // mutable variable, needed during the last layer of the computation to skip unused tokens
    int32_t n_tokens = this->n_tokens;

    bool is_lite = (hparams.n_layer == 27 || hparams.n_layer == 26);

    // We have to pre-scale kq_scale and attn_factor to make the YaRN RoPE work correctly.
    // See https://github.com/ggerganov/llama.cpp/discussions/7416 for detailed explanation.
    const float mscale = attn_factor * (1.0f + hparams.rope_yarn_log_mul * logf(1.0f / freq_scale));
    const float kq_scale = 1.0f*mscale*mscale/sqrtf(float(hparams.n_embd_head_k));
    const float attn_factor_scaled = 1.0f / (1.0f + 0.1f * logf(1.0f / freq_scale));

    const uint32_t n_embd_head_qk_rope = hparams.n_rot;
    const uint32_t n_embd_head_qk_nope = hparams.n_embd_head_k - hparams.n_rot;
    const uint32_t kv_lora_rank = hparams.n_lora_kv;
    const uint32_t q_lora_rank  = hparams.n_lora_q;

    struct ggml_tensor * cur;
    struct ggml_tensor * inpL;

    // {n_embd, n_tokens}
    inpL = llm_build_inp_embd(ctx0, lctx, hparams, batch, model.tok_embd, cb);

    // inp_pos - contains the positions
    struct ggml_tensor * inp_pos = build_inp_pos();

    // KQ_mask (mask for 1 head, it will be broadcasted to all heads)
    struct ggml_tensor * KQ_mask = build_inp_KQ_mask();

    // whether to use n_tokens as the matrix dimension during multiplication or n_head
    // n_tokens is higher during prompt processing, this allows to optimize for this case
    bool pp_opt = n_tokens >= 128; // Is it a fixed constant or is it somehow relared to n_head? original: n_tokens > n_head;

    auto rope_cache = cparams.rope_cache && (rope_type == LLAMA_ROPE_TYPE_NEOX || rope_type == LLAMA_ROPE_TYPE_NORM) ?
        ggml_rope_cache(ctx0, inp_pos, nullptr, n_rot, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow) : nullptr;

    for (int il = 0; il < n_layer; ++il) {
        struct ggml_tensor * inpSA = inpL;

        // norm
        cur = llm_build_norm(ctx0, inpL, hparams, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, cb, il);
        cb(cur, "attn_norm", il);

        // self_attention
        {
            ggml_tensor * q = nullptr;
            ggml_tensor * kv_rope_compressed = nullptr;
            ggml_tensor * q_rope;
            ggml_tensor * q_nope;
            ggml_tensor * k_rope;
            ggml_tensor * kv_compressed;
            if (model.layers[il].wkq_a_mqa) {
                auto mqa = ggml_mul_mat(ctx0, model.layers[il].wkq_a_mqa, cur);
                cb(mqa, "mqa", il);
                size_t qnb1;
                if (!is_lite) {
                    q = ggml_view_2d(ctx0, mqa, q_lora_rank, n_tokens, mqa->nb[1], 0);
                    q = llm_build_norm(ctx0, q, hparams, model.layers[il].attn_q_a_norm, NULL, LLM_NORM_RMS, cb, il);
                    q = ggml_mul_mat(ctx0, model.layers[il].wq_b, q);
                    qnb1 = q->nb[1];
                    cb(q, "q", il);
                    kv_rope_compressed = ggml_view_2d(ctx0, mqa, kv_lora_rank + n_embd_head_qk_rope, n_tokens, mqa->nb[1],
                            q_lora_rank*ggml_element_size(mqa));
                } else {
                    q = ggml_view_2d(ctx0, mqa, n_embd_k_gqa, n_tokens, mqa->nb[1], 0);
                    kv_rope_compressed = ggml_view_2d(ctx0, mqa, kv_lora_rank + n_embd_head_qk_rope, n_tokens, mqa->nb[1],
                            n_embd_k_gqa*ggml_element_size(mqa));
                    qnb1 = mqa->nb[1];
                }
                q_nope = ggml_view_3d(ctx0, q, n_embd_head_qk_nope, n_head, n_tokens,
                    ggml_row_size(q->type, hparams.n_embd_head_k), qnb1, 0);
                q_rope = ggml_view_3d(ctx0, q, n_embd_head_qk_rope, n_head, n_tokens,
                    ggml_row_size(q->type, hparams.n_embd_head_k), qnb1, ggml_row_size(q->type, n_embd_head_qk_nope));
                k_rope = ggml_view_3d(ctx0, kv_rope_compressed, n_embd_head_qk_rope, 1, n_tokens,
                        mqa->nb[1], mqa->nb[1], ggml_row_size(kv_rope_compressed->type, kv_lora_rank));
                kv_compressed = ggml_view_2d(ctx0, kv_rope_compressed, kv_lora_rank, n_tokens, mqa->nb[1], 0);
            }
            else {
                if (!is_lite) {
                    q = ggml_mul_mat(ctx0, model.layers[il].wq_a, cur);
                    cb(q, "q", il);

                    kv_rope_compressed = ggml_mul_mat(ctx0, model.layers[il].wkv_a_mqa, cur);
                    cb(kv_rope_compressed, "kv_rope_compressed", il);

                    ggml_build_forward_expand(gf, q);
                    ggml_build_forward_expand(gf, kv_rope_compressed);

                    q = llm_build_norm(ctx0, q, hparams, model.layers[il].attn_q_a_norm, NULL, LLM_NORM_RMS, cb, il);
                    cb(q, "q", il);

                    q = ggml_mul_mat(ctx0, model.layers[il].wq_b, q);
                    cb(q, "q", il);
                } else {
                    q = ggml_mul_mat(ctx0, model.layers[il].wq, cur);
                    cb(q, "q", il);

                    kv_rope_compressed = ggml_mul_mat(ctx0, model.layers[il].wkv_a_mqa, cur);
                    cb(kv_rope_compressed, "kv_rope_compressed", il);

                    ggml_build_forward_expand(gf, q);
                    ggml_build_forward_expand(gf, kv_rope_compressed);
                }

                q_nope = ggml_view_3d(ctx0, q, n_embd_head_qk_nope, n_head, n_tokens,
                        ggml_row_size(q->type, hparams.n_embd_head_k),
                        ggml_row_size(q->type, hparams.n_embd_head_k * n_head), 0);

                q_rope = ggml_view_3d(ctx0, q, n_embd_head_qk_rope, n_head, n_tokens,
                        ggml_row_size(q->type, hparams.n_embd_head_k),
                        ggml_row_size(q->type, hparams.n_embd_head_k * n_head),
                        ggml_row_size(q->type, n_embd_head_qk_nope));

                k_rope = ggml_view_3d(ctx0, kv_rope_compressed, n_embd_head_qk_rope, 1, n_tokens,
                        kv_rope_compressed->nb[1],
                        kv_rope_compressed->nb[1],
                        ggml_row_size(kv_rope_compressed->type, kv_lora_rank));

                kv_compressed = ggml_view_2d(ctx0, kv_rope_compressed, kv_lora_rank, n_tokens,
                        kv_rope_compressed->nb[1], 0);
            }
            cb(q_nope, "q_nope", il);
            cb(q_rope, "q_rope", il);
            cb(k_rope, "k_rope", il);
            cb(kv_compressed, "kv_compressed", il);

            ggml_build_forward_expand(gf, q_rope);
            ggml_build_forward_expand(gf, k_rope);
            if (rope_cache) {
                q_rope = ggml_rope_fast(ctx0, q_rope, rope_cache);
                k_rope = ggml_rope_fast(ctx0, k_rope, rope_cache);
            } else {
                q_rope = ggml_rope_ext(ctx0, q_rope, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                        ext_factor, attn_factor_scaled, beta_fast, beta_slow);

                k_rope = ggml_rope_ext(ctx0, k_rope, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                        ext_factor, attn_factor_scaled, beta_fast, beta_slow);
            }
            cb(q_rope, "q_rope", il);
            cb(k_rope, "k_rope", il);
            ggml_build_forward_expand(gf, q_rope);
            ggml_build_forward_expand(gf, k_rope);

            kv_compressed = llm_build_norm(ctx0, kv_compressed, hparams, model.layers[il].attn_kv_a_norm, NULL, LLM_NORM_RMS, cb, il);
            cb(kv_compressed, "kv_compressed", il);

            if (lctx.cparams.mla_attn) {

                ggml_tensor * kv_cache_trans = nullptr;

                if (lctx.cparams.mla_attn == 1 && !lctx.cparams.flash_attn) {
                    ggml_tensor * kv_cache_trans_view = ggml_view_2d(ctx0, kv_self.v_l[il], n_tokens, kv_lora_rank,
                            ggml_row_size(kv_self.v_l[il]->type, kv_self.size), ggml_row_size(kv_self.v_l[il]->type, kv_head));
                    cb(kv_cache_trans_view, "kv_cache_trans_view", il);

                    // note: storing transposed c^KV in the transposed KV cache
                    ggml_build_forward_expand(gf, ggml_cpy(ctx0, ggml_transpose(ctx0, kv_compressed), kv_cache_trans_view));

                    kv_cache_trans = ggml_view_2d(ctx0, kv_self.v_l[il],
                            n_kv, kv_lora_rank,
                            ggml_row_size(kv_self.v_l[il]->type, kv_self.size),
                            0);
                    cb(kv_cache_trans, "kv_cache_trans", il);
                }

                //ggml_tensor * kvr = ggml_concat(ctx0, kv_compressed, ggml_permute(ctx0, k_rope, 0, 2, 1, 3), 0);
                ggml_tensor * kvr = ggml_concat(ctx0, ggml_permute(ctx0, k_rope, 0, 2, 1, 3), kv_compressed, 0);
                cb(kvr, "kvr", il);

                auto row_size = ggml_row_size(kv_self.k_l[il]->type, kv_lora_rank + n_embd_head_qk_rope);
                ggml_tensor * kv_cache_view = ggml_view_2d(ctx0, kv_self.k_l[il], kv_self.k_l[il]->ne[0], n_tokens,
                        row_size, row_size*kv_head);
                lctx.cache_copies[2*il+0].cpy = ggml_cpy(ctx0, kvr, kv_cache_view);
                lctx.cache_copies[2*il+0].step = row_size;
                ggml_build_forward_expand(gf, lctx.cache_copies[2*il+0].cpy);
                ggml_tensor * kv_cache = ggml_view_2d(ctx0, kv_self.k_l[il],
                        kv_lora_rank + n_embd_head_qk_rope, n_kv,
                        ggml_row_size(kv_self.k_l[il]->type, kv_lora_rank + n_embd_head_qk_rope), 0);
                cb(kv_cache, "kv_cache", il);

                ggml_tensor * kqv;

                if (lctx.cparams.mla_attn > 1 && lctx.cparams.flash_attn && pp_opt) { // PP for mla=2,3

                    auto kv_cache_nope = ggml_view_2d(ctx0, kv_self.k_l[il], kv_lora_rank, n_kv, kv_self.k_l[il]->nb[1],
                            ggml_row_size(kv_self.k_l[il]->type, n_embd_head_qk_rope));

                    auto kv_f32_size = model.layers[il].wkv_b->ne[1] * kv_cache_nope->ne[1] * sizeof(float) / (1024*1024);
                    int n_max_head = n_head;
                    if (cparams.attn_max_batch > 0 && kv_f32_size > cparams.attn_max_batch) {
                        while (n_max_head%2 == 0 && kv_f32_size > cparams.attn_max_batch) {
                            n_max_head /= 2; kv_f32_size /= 2;
                        }
                    }
                    GGML_ASSERT(n_head % n_max_head == 0);

                    auto n_per_head = model.layers[il].wkv_b->ne[1] / n_head;

                    auto kv_cache_rope = ggml_view_3d(ctx0, kv_self.k_l[il], n_embd_head_qk_rope, n_kv, 1,
                            kv_self.k_l[il]->nb[1], kv_self.k_l[il]->nb[2], 0); //ggml_row_size(kv_self.k_l[il]->type, kv_lora_rank));

                    // There is still an issue with one or more of the ops GGML_OP_REPEAT, GGML_OP_CONCAT, GGML_OP_CPY on CUDA when
                    // the KV cache is quantized. Hence, in that case we will simply use fp16 for now.
                    // The downside of the following line is that fp16 will be used even if attention is computed on the CPU
                    // if the build is with CUDA enabled.
                    auto kv_type = lctx.backends.size() == 1 && lctx.backends.front() == lctx.backend_cpu ? kv_self.k_l[il]->type : GGML_TYPE_F16;

                    ggml_tensor repeater;
                    repeater.ne[0] = n_embd_head_qk_rope; repeater.ne[1] = n_kv; repeater.ne[2] = n_max_head; repeater.ne[3] = 1;
                    ggml_tensor * k_rope;
                    if (kv_cache_rope->type == kv_type) {
                        k_rope = ggml_repeat(ctx0, kv_cache_rope, &repeater);
                    } else {
                        auto kv_cache_rope_f16 = ggml_cast(ctx0, kv_cache_rope, GGML_TYPE_F16);
                        k_rope = ggml_repeat(ctx0, kv_cache_rope_f16, &repeater);
                    }
                    cb(k_rope, "k_rope", il);

                    //auto q = ggml_concat(ctx0, q_nope, q_rope, 0);
                    auto q = ggml_concat(ctx0, q_rope, q_nope, 0);
                    q = ggml_permute(ctx0, q, 0, 2, 1, 3);
                    cb(q, "q_concat", il);

                    ggml_build_forward_expand(gf, q);

                    for (int iter = 0; iter < n_head/n_max_head; ++iter) {

                        auto wkv_b = ggml_view_2d(ctx0, model.layers[il].wkv_b, model.layers[il].wkv_b->ne[0], n_per_head*n_max_head,
                                model.layers[il].wkv_b->nb[1], model.layers[il].wkv_b->nb[1]*n_per_head*n_max_head*iter);

                        auto kv_f32 = ggml_mul_mat(ctx0, wkv_b, kv_cache_nope);
                        cb(kv_f32, "kv_f32", il);

                        auto v_f32 = ggml_view_3d(ctx0, kv_f32, hparams.n_embd_head_v, n_kv, n_max_head,
                                ggml_row_size(kv_f32->type, n_max_head * (n_embd_head_qk_nope + hparams.n_embd_head_v)),
                                ggml_row_size(kv_f32->type, n_embd_head_qk_nope + hparams.n_embd_head_v),
                                ggml_row_size(kv_f32->type, n_embd_head_qk_nope));
                        cb(v_f32, "v_f32", il);

                        auto k_nope_f32 = ggml_view_3d(ctx0, kv_f32, n_embd_head_qk_nope, n_kv, n_max_head,
                                ggml_row_size(kv_f32->type, n_max_head * (n_embd_head_qk_nope + hparams.n_embd_head_v)),
                                ggml_row_size(kv_f32->type, n_embd_head_qk_nope + hparams.n_embd_head_v), 0);
                        cb(k_nope_f32, "k_nope_f32", il);

                        auto v = ggml_cast(ctx0, v_f32, kv_type);
                        cb(v, "v", il);

                        auto k_nope = ggml_cast(ctx0, k_nope_f32, kv_type);
                        cb(k_nope, "k_nope", il);

                        ggml_build_forward_expand(gf, k_nope);
                        ggml_build_forward_expand(gf, v);

                        //auto k = ggml_concat(ctx0, k_nope, k_rope, 0);
                        auto k = ggml_concat(ctx0, k_rope, k_nope, 0);
                        cb(k, "k", il);

                        ggml_build_forward_expand(gf, k);

                        auto q_iter = ggml_view_3d(ctx0, q, q->ne[0], q->ne[1], n_max_head,
                                q->nb[1], q->nb[2], q->nb[2]*n_max_head*iter);

                        kqv = ggml_flash_attn_ext(ctx0, q_iter, k, v, KQ_mask, kq_scale, hparams.f_max_alibi_bias, 0.f);
                        if (use_f32_attn_precision || q->ne[1] <= 8) {
                            ggml_flash_attn_ext_set_prec(kqv, GGML_PREC_F32);
                        }
                        cb(kqv, "kqv", il);

                        if (iter == 0) {
                            cur = ggml_reshape_2d(ctx0, kqv, n_embd_head_v*n_max_head, n_tokens);
                        } else {
                            cur = ggml_concat(ctx0, cur, ggml_reshape_2d(ctx0, kqv, n_embd_head_v*n_max_head, n_tokens), 0);
                        }

                    }

                }
                else {

                    ggml_tensor * kqv_compressed = nullptr;

                    //auto wkv_b = model.layers[il].wkv_b;
                    auto wk_b = model.layers[il].wk_b->ne[1] == kv_lora_rank ? model.layers[il].wk_b
                        : ggml_reshape_3d(ctx0, model.layers[il].wk_b, n_embd_head_qk_nope, kv_lora_rank, n_head);

                    q_nope = ggml_permute(ctx0, q_nope, 0, 2, 1, 3);
                    cb(q_nope, "q_nope_perm", il);

                    struct ggml_tensor * q_nope2 = ggml_mul_mat(ctx0, wk_b, q_nope);
                    cb(q_nope2, "q_nope2", il);

                    //ggml_tensor * q = ggml_concat(ctx0, q_nope2, ggml_permute(ctx0, q_rope, 0, 2, 1, 3), 0);
                    ggml_tensor * q = ggml_concat(ctx0, ggml_permute(ctx0, q_rope, 0, 2, 1, 3), q_nope2, 0);
                    cb(q, "q", il);

                    if (lctx.cparams.flash_attn && (lctx.cparams.mla_attn == 1 || lctx.cparams.mla_attn == 3)) {
                        ggml_tensor * kv_cache_lora = ggml_view_2d(ctx0, kv_self.k_l[il],
                                kv_lora_rank, n_kv,
                                ggml_row_size(kv_self.k_l[il]->type, kv_lora_rank + n_embd_head_qk_rope),
                                ggml_row_size(kv_self.k_l[il]->type, n_embd_head_qk_rope));
                        cb(kv_cache_lora, "kv_cache_lora", il);

                        kqv_compressed = ggml_flash_attn_ext(ctx0, q, kv_cache, kv_cache_lora, KQ_mask, kq_scale, hparams.f_max_alibi_bias, 0.f);
                        cb(kqv_compressed, "kqv_compressed", il);

                        if (use_f32_attn_precision) {
                            ggml_flash_attn_ext_set_prec(kqv_compressed, GGML_PREC_F32);
                        }

                        kqv_compressed = ggml_permute(ctx0, kqv_compressed, 0, 2, 1, 3);
                        cb(kqv_compressed, "kqv_compressed_perm", il);
                    }
                    else {
                        if (lctx.cparams.mla_attn > 1) {
                            ggml_tensor * kv_cache_lora = ggml_view_2d(ctx0, kv_self.k_l[il],
                                    kv_lora_rank, n_kv,
                                    ggml_row_size(kv_self.k_l[il]->type, kv_lora_rank + n_embd_head_qk_rope),
                                    ggml_row_size(kv_self.k_l[il]->type, n_embd_head_qk_rope));
                            cb(kv_cache, "kv_cache_lora", il);

                            kv_cache_trans = ggml_cont(ctx0, ggml_transpose(ctx0, kv_cache_lora));
                            cb(kv_cache_trans, "kv_cache_trans", il);
                        }

                        auto kq_size = kv_cache->ne[1]*q->ne[1]*q->ne[2]*sizeof(float)/(1024*1024); // K*Q in MiB
                        if (lctx.cparams.attn_max_batch <= 0 || lctx.cparams.attn_max_batch >= kq_size) {
                            if (!pp_opt) {
                                q = ggml_permute(ctx0, q, 0, 2, 1, 3);
                                cb(q, "q_perm", il);
                            }

                            ggml_tensor * kq = ggml_mul_mat(ctx0, kv_cache, q);
                            if (kv_cache->ne[1] < 256) {
                                ggml_mul_mat_set_prec(kq, GGML_PREC_F32);
                            }
                            cb(kq, "kq", il);

                            if (!pp_opt) {
                                kq = ggml_cont(ctx0, ggml_permute(ctx0, kq, 0, 2, 1, 3));
                                cb(kq, "kq_perm", il);
                            }

                            kq = ggml_soft_max_ext(ctx0, kq, KQ_mask, kq_scale, hparams.f_max_alibi_bias);
                            cb(kq, "kq_soft_max_ext", il);

                            if (!pp_opt) {
                                kq = ggml_permute(ctx0, kq, 0, 2, 1, 3);
                                cb(kq, "kq_soft_max_ext_perm", il);
                            }

                            kqv_compressed = ggml_mul_mat(ctx0, kv_cache_trans, kq);
                            cb(kqv_compressed, "kqv_compressed", il);

                            if (!pp_opt) {
                                kqv_compressed = ggml_permute(ctx0, kqv_compressed, 0, 2, 1, 3);
                                cb(kqv_compressed, "kqv_compressed_perm", il);
                            }

                        } else {

                            int n_step = (kq_size + lctx.cparams.attn_max_batch - 1)/lctx.cparams.attn_max_batch;
                            n_step = std::min(n_step, int(q->ne[2]));
                            int n_per_step = (q->ne[2] + n_step - 1)/n_step;

                            for (int i_head = 0; i_head < q->ne[2]; i_head += n_per_step) {
                                int this_ne12 = i_head + n_per_step <= q->ne[2] ? n_per_step : q->ne[2] - i_head;
                                ggml_tensor * q_i = ggml_view_3d(ctx0, q, q->ne[0], q->ne[1], this_ne12, q->nb[1], q->nb[2], q->nb[2]*i_head);
                                ggml_tensor * kq_i = ggml_mul_mat(ctx0, kv_cache, q_i);
                                kq_i = ggml_soft_max_ext(ctx0, kq_i, KQ_mask, kq_scale, hparams.f_max_alibi_bias);
                                ggml_tensor * kqv_i = ggml_mul_mat(ctx0, kv_cache_trans, kq_i);
                                if (i_head == 0) {
                                    kqv_compressed = kqv_i;
                                } else {
                                    kqv_compressed = ggml_concat(ctx0, kqv_compressed, kqv_i, 2);
                                }
                                ggml_build_forward_expand(gf, kqv_compressed);
                            }
                            cb(kqv_compressed, "kqv_compressed", il);
                        }
                    }

                    auto wv_b = model.layers[il].wv_b;
                    if (wv_b->ne[1] != n_embd_head_v) {
                        wv_b = ggml_reshape_3d(ctx0, wv_b, kv_lora_rank, n_embd_head_v, n_head);
                        cb(wv_b, "wv_b", il);
                    }
                    // There is an issue with quantized GEMV on CUDA when the left operand (the matrix) is
                    // not contiguous. So, for now, we create wv_b during model loading and use that
                    // instead of the commented out 3D view below.
                    //auto wv_b = ggml_view_3d(ctx0, wkv_b, kv_lora_rank, n_embd_head_v, n_head,
                    //        wkv_b->nb[1], wkv_b->nb[1]*(n_embd_head_v + n_embd_head_qk_nope),
                    //        wkv_b->nb[1]*n_embd_head_qk_nope);
                    //cb(wv_b, "wv_b", il);

                    kqv = ggml_mul_mat(ctx0, wv_b, kqv_compressed);
                    cb(kqv, "kqv", il);

                    if (n_tokens > 1) {
                        kqv = ggml_cont(ctx0, ggml_permute(ctx0, kqv, 0, 2, 1, 3));
                        cb(kqv, "kqv_perm", il);
                    }
                    cur = ggml_reshape_2d(ctx0, kqv, n_embd_head_v*n_head, n_tokens);
                    cb(cur, "kqv_2d", il);

                }

                ggml_build_forward_expand(gf, cur);

                cur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wo, cur);
                cb(cur, "kqv_out", il);

            }
            else {

                // {kv_lora_rank, n_head * (n_embd_head_qk_nope + n_embd_head_v)} * {kv_lora_rank, n_tokens} -> {n_head * (n_embd_head_qk_nope + n_embd_head_v), n_tokens}
                struct ggml_tensor * kv = ggml_mul_mat(ctx0, model.layers[il].wkv_b, kv_compressed);
                cb(kv, "kv", il);

                // split into {n_head * n_embd_head_qk_nope, n_tokens}
                struct ggml_tensor * k_nope = ggml_view_3d(ctx0, kv, n_embd_head_qk_nope, n_head, n_tokens,
                        ggml_row_size(kv->type, n_embd_head_qk_nope + hparams.n_embd_head_v),
                        ggml_row_size(kv->type, n_head * (n_embd_head_qk_nope + hparams.n_embd_head_v)),
                        0);
                cb(k_nope, "k_nope", il);

                // and {n_head * n_embd_head_v, n_tokens}
                struct ggml_tensor * v_states = ggml_view_3d(ctx0, kv, hparams.n_embd_head_v, n_head, n_tokens,
                        ggml_row_size(kv->type, (n_embd_head_qk_nope + hparams.n_embd_head_v)),
                        ggml_row_size(kv->type, (n_embd_head_qk_nope + hparams.n_embd_head_v)*n_head),
                        ggml_row_size(kv->type, (n_embd_head_qk_nope)));
                cb(v_states, "v_states", il);

                v_states = ggml_cont(ctx0, v_states);
                cb(v_states, "v_states", il);

                v_states = ggml_view_2d(ctx0, v_states, hparams.n_embd_head_v * n_head, n_tokens,
                        ggml_row_size(kv->type, hparams.n_embd_head_v * n_head),
                        0);
                cb(v_states, "v_states", il);

                struct ggml_tensor * q_states = ggml_concat(ctx0, q_nope, q_rope, 0);
                cb(q_states, "q_states", il);

                struct ggml_tensor * k_states = ggml_concat(ctx0, k_nope, ggml_repeat(ctx0, k_rope, q_rope), 0);
                cb(k_states, "k_states", il);

                cur = llm_build_kv(ctx0, lctx, kv_self, gf,
                        model.layers[il].wo, NULL,
                        k_states, v_states, q_states, KQ_mask, n_tokens, kv_head, n_kv, kq_scale, cb, il);

            }
        }

        if (il == n_layer - 1) {
            // skip computing output for unused tokens
            struct ggml_tensor * inp_out_ids = build_inp_out_ids();
            n_tokens = n_outputs;
            cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
            cb(cur, "last_attn", il);
            cb(inpSA, "last_ffn_inp", il);
        }

        struct ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        cur = llm_build_norm(ctx0, ffn_inp, hparams, model.layers[il].ffn_norm, NULL, LLM_NORM_RMS, cb, il);
        cb(cur, "ffn_norm", il);

        if ((uint32_t) il < hparams.n_layer_dense_lead) {
            cur = llm_build_ffn(ctx0, lctx, nullptr, cur,
                    model.layers[il].ffn_up,   NULL, NULL,
                    model.layers[il].ffn_gate, NULL, NULL,
                    model.layers[il].ffn_down, NULL, NULL,
                    NULL,
                    LLM_FFN_SILU, LLM_FFN_PAR, cb, il);
            cb(cur, "ffn_out", il);
        } else {
            // MoE branch
            ggml_tensor * moe_out =
                llm_build_moe_ffn(ctx0, lctx, cur,
                        model.layers[il].ffn_gate_inp,
                        model.layers[il].ffn_up_exps,
                        model.layers[il].ffn_gate_exps,
                        model.layers[il].ffn_down_exps,
                        model.layers[il].ffn_exp_probs_b,
                        n_expert, n_expert_used,
                        LLM_FFN_SILU, hparams.expert_weights_norm,
                        true, hparams.expert_weights_scale,
                        (enum llm_expert_gating_func_type) hparams.expert_gating_func,
                        cb, il, gf);
            cb(moe_out, "ffn_moe_out", il);

            // FFN shared expert
            {
                ggml_tensor * ffn_shexp = llm_build_ffn(ctx0, lctx, nullptr, cur,
                        model.layers[il].ffn_up_shexp,   NULL, NULL,
                        model.layers[il].ffn_gate_shexp, NULL, NULL,
                        model.layers[il].ffn_down_shexp, NULL, NULL,
                        NULL,
                        LLM_FFN_SILU, LLM_FFN_PAR, cb, il);
                cb(ffn_shexp, "ffn_shexp", il);

                cur = ggml_add(ctx0, moe_out, ffn_shexp);
                cb(cur, "ffn_out", il);
            }
        }

        cur = ggml_add(ctx0, cur, ffn_inp);
        cur = lctx.cvec.apply_to(ctx0, cur, il);
        cb(cur, "l_out", il);

        // input for next layer
        inpL = cur;
    }

    cur = inpL;

    cur = llm_build_norm(ctx0, cur, hparams, model.output_norm, NULL, LLM_NORM_RMS, cb, -1);
    cb(cur, "result_norm", -1);

    // lm_head
    cur = ggml_mul_mat(ctx0, model.output, cur);
    cb(cur, "result_output", -1);

    ggml_build_forward_expand(gf, cur);

    return gf;
}

ggml_cgraph * llm_build_context::build_glm4_moe() {
    // create a new graph
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, llama_model_max_nodes(model), false);

    const int64_t n_embd_head = hparams.n_embd_head_v;
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);

    struct ggml_tensor * cur;
    struct ggml_tensor * inpL;

    // input embeddings
    inpL = llm_build_inp_embd(ctx0, lctx, hparams, batch, model.tok_embd, cb);

    // position embeddings
    struct ggml_tensor * inp_pos = build_inp_pos();

    // attention KV cache input
    //auto * inp_attn = build_attn_inp_kv_unified();

    struct ggml_tensor * KQ_mask = build_inp_KQ_mask();

    // output token IDs (for last layer cropping)
    struct ggml_tensor * inp_out_ids = build_inp_out_ids();

    auto rope_cache = model.split_mode != LLAMA_SPLIT_MODE_GRAPH && cparams.rope_cache && (rope_type == LLAMA_ROPE_TYPE_NEOX || rope_type == LLAMA_ROPE_TYPE_NORM) ?
        ggml_rope_cache(ctx0, inp_pos, nullptr, n_embd_head, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow) : nullptr;

    float kq_scale = 1.0f/sqrtf(float(n_embd_head));

    // Only process up to last layer (skip final NextN layer)
    // Final layer tensors are loaded but not processed in forward pass
    const int n_transformer_layers = n_layer - hparams.nextn_predict_layers;
    for (int il = 0; il < n_transformer_layers; ++il) {
        struct ggml_tensor * inpSA = inpL;

        // self-attention
        if (rope_cache == nullptr) {
            cur = build_std_attention(gf, model.layers[il].attn_norm, inpL, inp_pos, nullptr, KQ_mask, nullptr, nullptr, kq_scale, 0.0f, 0, il, true, false, true);
        } else {
            // Pre-attention norm
            cur = llm_build_norm(ctx0, inpL, hparams, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, cb, il);
            cb(cur, "attn_norm", il);

            auto [Qcur, Kcur, Vcur] = llm_build_mul_mat_qkv(gf, cur,
                    model.layers[il].wqkv, model.layers[il].bqkv,
                    model.layers[il].wqk, model.layers[il].bqk,
                    model.layers[il].wq, model.layers[il].bq,
                    model.layers[il].wk, model.layers[il].bk,
                    model.layers[il].wv, model.layers[il].bv,
                    model.layers[il].attn_q_norm, model.layers[il].attn_k_norm, 0.f, il);

            // apply RoPE
            if (rope_cache) {
                Qcur = ggml_rope_fast(ctx0, Qcur, rope_cache);
                Kcur = ggml_rope_fast(ctx0, Kcur, rope_cache);
            } else {
                Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                        ext_factor, attn_factor, beta_fast, beta_slow);
                Kcur = ggml_rope_ext(ctx0, Kcur, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                        ext_factor, attn_factor, beta_fast, beta_slow);
            }
            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            // build attention KV (no unified cache)
            cur = llm_build_kv(ctx0, lctx, kv_self, gf,
                    model.layers[il].wo, NULL,
                    Kcur, Vcur, Qcur, KQ_mask,
                    n_tokens, kv_head, n_kv,
                    1.0f/sqrtf(float(n_embd_head)), cb, il);
        }

        // crop output on last layer
        if (il == n_transformer_layers - 1 && inp_out_ids) {
            // skip computing output for unused tokens
            cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }

        // residual connection for attention output
        ggml_tensor * ffn_inp;
        if (rope_cache) {
            ffn_inp = ggml_add(ctx0, cur, inpSA);
            cb(ffn_inp, "ffn_inp", il);
        } else {
            ffn_inp = cur;
        }

        if ((uint32_t) il < hparams.n_layer_dense_lead) {
            // dense FFN
            cur = llm_build_ffn(ctx0, lctx, model.layers[il].ffn_norm, ffn_inp,
                    model.layers[il].ffn_up,   NULL, NULL,
                    model.layers[il].ffn_gate, NULL, NULL,
                    model.layers[il].ffn_down, NULL, NULL,
                    NULL,
                    LLM_FFN_SILU, LLM_FFN_PAR, cb, il, gf, true);
            cb(cur, "ffn_out", il);
        } else {
            cur = llm_build_std_moe_ffn(ctx0, lctx, model.layers[il].ffn_norm, ffn_inp,
                    model.layers[il].ffn_gate_inp,  model.layers[il].ffn_gate_inp_b,
                    model.layers[il].ffn_up_exps,   model.layers[il].ffn_up_exps_b,
                    model.layers[il].ffn_gate_exps, model.layers[il].ffn_gate_exps_b,
                    model.layers[il].ffn_down_exps, model.layers[il].ffn_down_exps_b,
                    model.layers[il].ffn_exp_probs_b,
                    model.layers[il].ffn_up_shexp,    nullptr, // we don't have shared expert biases?
                    model.layers[il].ffn_gate_shexp,  nullptr,
                    model.layers[il].ffn_down_shexp,  nullptr,
                    n_expert, n_expert_used,
                    LLM_FFN_SILU, hparams.expert_weights_norm, true, hparams.expert_weights_scale,
                    (llm_expert_gating_func_type) hparams.expert_gating_func,
                    LLM_FFN_SILU, cb, il, gf, true);
        }

        // residual and context vector
        //cur = ggml_add(ctx0, cur, ffn_inp);
        cur = lctx.cvec.apply_to(ctx0, cur, il);
        cb(cur, "l_out", il);

        // prepare next layer input
        inpL = cur;
    }

    cur = inpL;

    // lm head
    cur = build_output(lctx, ctx0, cur, model.output, model.output_norm, cb);
    cb(cur, "result_output", -1);

    ggml_build_forward_expand(gf, cur);
    return gf;
}

ggml_cgraph * llm_build_context::build_bitnet() {
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, llama_model_max_nodes(model), false);

    const int64_t n_embd_head = hparams.n_embd_head_v;
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);

    struct ggml_tensor * cur;
    struct ggml_tensor * inpL;

    inpL = llm_build_inp_embd(ctx0, lctx, hparams, batch, model.tok_embd, cb);

    // inp_pos - contains the positions
    struct ggml_tensor * inp_pos = build_inp_pos();

    // KQ_mask (mask for 1 head, it will be broadcasted to all heads)
    struct ggml_tensor * KQ_mask = build_inp_KQ_mask();

    for (int il = 0; il < n_layer; ++il) {
        struct ggml_tensor * inpSA = inpL;

        cur = llm_build_norm(ctx0, inpL, hparams, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, cb, il);
        cb(cur, "attn_norm", il);

        // self-attention
        {
            // compute Q and K and RoPE them
            struct ggml_tensor * Qcur = ggml_mul_mat(ctx0, model.layers[il].wq, cur);
            float q_scale; std::memcpy(&q_scale, model.layers[il].wq->op_params, sizeof(float));
            // Note: we could save this scale operation by applying the Q scale on the K * Q product further down
            // (which also uses a scale). This works on the CPU and Metal backends, but produces NaNs on CUDA.
            if (fabsf(q_scale-1) > 1e-4f) Qcur = ggml_scale(ctx0, Qcur, q_scale);
            cb(Qcur, "Qcur", il);
            if (model.layers[il].bq) {
                Qcur = ggml_add(ctx0, Qcur, model.layers[il].bq);
                cb(Qcur, "Qcur", il);
            }

            // B1.K
            struct ggml_tensor * Kcur = ggml_mul_mat(ctx0, model.layers[il].wk, cur);
            float k_scale; std::memcpy(&k_scale, model.layers[il].wk->op_params, sizeof(float));
            if (fabsf(k_scale-1) > 1e-4f) Kcur = ggml_scale(ctx0, Kcur, k_scale);
            cb(Kcur, "Kcur", il);
            if (model.layers[il].bk) {
                Kcur = ggml_add(ctx0, Kcur, model.layers[il].bk);
                cb(Kcur, "Kcur", il);
            }

            // B1.V
            struct ggml_tensor * Vcur = ggml_mul_mat(ctx0, model.layers[il].wv, cur);
            float v_scale; std::memcpy(&v_scale, model.layers[il].wv->op_params, sizeof(float));
            if (model.layers[il].bv) {
                if (fabsf(v_scale-1) > 1e-4f) Vcur = ggml_scale(ctx0, Vcur, v_scale);
                v_scale = 1;
                Vcur = ggml_add(ctx0, Vcur, model.layers[il].bv);
            }
            cb(Vcur, "Vcur", il);

            Qcur = ggml_rope_ext(
                    ctx0, ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens), inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );
            cb(Qcur, "Qcur", il);

            Kcur = ggml_rope_ext(
                    ctx0, ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens), inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );
            cb(Kcur, "Kcur", il);

            ggml_tensor * cur_attn = llm_build_kv(ctx0, lctx, kv_self, gf,
                    // we cannot pass model.layers[il].wo and model.layers[il].bo because we need to do rms_norm first
                    nullptr, nullptr,
                    Kcur, Vcur, Qcur, KQ_mask, n_tokens, kv_head, n_kv, 1.0f/sqrtf(float(n_embd_head)), cb, il);

            cur_attn = llm_build_norm(ctx0, cur_attn, hparams, model.layers[il].attn_sub_norm, NULL, LLM_NORM_RMS, cb, il, 1/(v_scale*v_scale));
            cb(cur_attn, "attn_sub_norm", il);

            ggml_build_forward_expand(gf, cur_attn);

            cur = ggml_mul_mat(ctx0, model.layers[il].wo, cur_attn);
            float wo_scale; std::memcpy(&wo_scale, model.layers[il].wo->op_params, sizeof(float));
            if (fabsf(wo_scale-1) > 1e-4f) cur = ggml_scale(ctx0, cur, wo_scale);

            cb(cur, "kqv_out", il);
        }

        if (il == n_layer - 1) {
            // skip computing output for unused tokens
            struct ggml_tensor * inp_out_ids = build_inp_out_ids();
            cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }

        struct ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        // feed-forward forward
        if (model.layers[il].ffn_gate_inp == nullptr) {
            cur = llm_build_norm(ctx0, ffn_inp, hparams, model.layers[il].ffn_norm, NULL, LLM_NORM_RMS, cb, il);
            cb(cur, "ffn_norm", il);

            struct ggml_tensor *tmp = ggml_mul_mat(ctx0, model.layers[il].ffn_up, cur);
            float ffn_up_scale; std::memcpy(&ffn_up_scale, model.layers[il].ffn_up->op_params, sizeof(float));

            cb(tmp, "ffn_up", il);

            cur = ggml_mul_mat(ctx0, model.layers[il].ffn_gate, cur);
            float ffn_gate_scale; std::memcpy(&ffn_gate_scale, model.layers[il].ffn_gate->op_params, sizeof(float));
            if (fabsf(ffn_gate_scale-1) > 1e-4f) cur = ggml_scale(ctx0, cur, ffn_gate_scale);

            cb(cur, "ffn_gate", il);

            cur = ggml_fused_mul_unary(ctx0, cur, tmp, GGML_UNARY_OP_SILU);
            cb(cur, "ffn_gate_par", il);

            cur = llm_build_norm(ctx0, cur, hparams, model.layers[il].ffn_sub_norm, NULL, LLM_NORM_RMS, cb, il, 1/(ffn_up_scale*ffn_up_scale));
            cb(cur, "ffn_sub_norm", il);

            cur = ggml_mul_mat(ctx0, model.layers[il].ffn_down, cur);
            float ffn_down_scale; std::memcpy(&ffn_down_scale, model.layers[il].ffn_down->op_params, sizeof(float));
            if (fabsf(ffn_down_scale-1) > 1e-4f) cur = ggml_scale(ctx0, cur, ffn_down_scale);
            cb(cur, "ffn_down", il);
        }
        cur = ggml_add(ctx0, cur, ffn_inp);
        cb(cur, "l_out", il);

        // input for next layer
        inpL = cur;
    }

    cur = inpL;

    cur = llm_build_norm(ctx0, cur, hparams, model.output_norm, NULL, LLM_NORM_RMS, cb, -1);
    cb(cur, "result_norm", -1);

    // lm_head
    cur = llm_build_lora_mm(lctx, ctx0, model.output, cur);
    cb(cur, "result_output", -1);

    ggml_build_forward_expand(gf, cur);
    return gf;
}

ggml_cgraph * llm_build_context::build_bitnet_158() {
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, llama_model_max_nodes(model), false);

    // mutable variable, needed during the last layer of the computation to skip unused tokens
    int32_t n_tokens = this->n_tokens;

    const int64_t n_embd_head = hparams.n_embd_head_v;
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);
    GGML_ASSERT(n_embd_head == hparams.n_rot);

    struct ggml_tensor * cur;
    struct ggml_tensor * inpL;

    inpL = llm_build_inp_embd(ctx0, lctx, hparams, batch, model.tok_embd, cb);

    // inp_pos - contains the positions
    struct ggml_tensor * inp_pos = build_inp_pos();

    // KQ_mask (mask for 1 head, it will be broadcasted to all heads)
    struct ggml_tensor * KQ_mask = build_inp_KQ_mask();

    for (int il = 0; il < n_layer; ++il) {
        struct ggml_tensor * inpSA = inpL;

        // norm
        cur = llm_build_norm(ctx0, inpL, hparams, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, cb, il);
        cb(cur, "attn_norm", il);

        // self-attention
        {
            // rope freq factors for llama3; may return nullptr for llama2 and other models
            struct ggml_tensor * rope_factors = build_rope_factors(il);

            auto [Qcur, Kcur, Vcur] = llm_build_mul_mat_qkv(gf, cur, model.layers[il].wq, nullptr,
                    model.layers[il].wk, nullptr,
                    model.layers[il].wv, nullptr, 0, il);

            Qcur = ggml_rope_ext(
                    ctx0, ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens), inp_pos, rope_factors,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );
            cb(Qcur, "Qcur", il);

            Kcur = ggml_rope_ext(
                    ctx0, ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens), inp_pos, rope_factors,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );
            cb(Kcur, "Kcur", il);

            cur = llm_build_kv(ctx0, lctx, kv_self, gf,
                    NULL, NULL,
                    Kcur, Vcur, Qcur, KQ_mask, n_tokens, kv_head, n_kv, 1.0f/sqrtf(float(n_embd_head)), cb, il);

            cur = llm_build_norm(ctx0, cur, hparams, model.layers[il].attn_sub_norm, NULL, LLM_NORM_RMS, cb, il);
            cb(cur, "attn_sub_norm", il);

            cur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wo, cur);
            if (model.layers[il].wo_scale) {
                cur = ggml_mul(ctx0, cur, model.layers[il].wo_scale);
            }
            if (model.layers[il].bo) {
                cur = ggml_add(ctx0, cur, model.layers[il].bo);
            }
            cb(cur, "attn_o_out", il);
        }

        if (il == n_layer - 1) {
            // skip computing output for unused tokens
            struct ggml_tensor * inp_out_ids = build_inp_out_ids();
            // n_tokens = n_outputs;
            cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }

        struct ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        cur = llm_build_ffn(ctx0, lctx, model.layers[il].ffn_norm, ffn_inp,
                model.layers[il].ffn_up,   NULL, model.layers[il].ffn_up_scale,
                model.layers[il].ffn_gate, NULL, model.layers[il].ffn_gate_scale,
                NULL, NULL, NULL,
                NULL,
                LLM_FFN_RELU_SQR, LLM_FFN_PAR, cb, il);
        cb(cur, "ffn_out", il);

        cur = llm_build_norm(ctx0, cur, hparams, model.layers[il].ffn_sub_norm, NULL, LLM_NORM_RMS, cb, il);
        cb(cur, "ffn_sub_norm", il);

        cur = llm_build_lora_mm(lctx, ctx0, model.layers[il].ffn_down, cur);
        if (model.layers[il].ffn_down_scale) {
            cur = ggml_mul(ctx0, cur, model.layers[il].ffn_down_scale);
        }
        cb(cur, "ffn_down", il);

        cur = ggml_add(ctx0, cur, ffn_inp);
        cb(cur, "l_out", il);

        // input for next layer
        inpL = cur;
    }

    cur = inpL;

    cur = llm_build_norm(ctx0, cur, hparams, model.output_norm, NULL, LLM_NORM_RMS, cb, -1);
    cb(cur, "result_norm", -1);

    // lm_head
    cur = llm_build_lora_mm(lctx, ctx0, model.output, cur);

    cb(cur, "result_output", -1);

    ggml_build_forward_expand(gf, cur);

    return gf;
}

ggml_cgraph * llm_build_context::build_cohere2() {
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, llama_model_max_nodes(model), false);

    const int64_t n_embd_head = hparams.n_embd_head_v;
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);
    const float f_logit_scale = hparams.f_logit_scale;

    struct ggml_tensor * cur;
    struct ggml_tensor * inpL;

    inpL = llm_build_inp_embd(ctx0, lctx, hparams, batch, model.tok_embd, cb);

    // inp_pos - contains the positions
    struct ggml_tensor * inp_pos = build_inp_pos();

    // KQ_mask (mask for 1 head, it will be broadcasted to all heads)
    // cohere2 requires different mask for layers using sliding window (SWA)
    struct ggml_tensor * KQ_mask     = build_inp_KQ_mask();
    struct ggml_tensor * KQ_mask_swa = build_inp_KQ_mask_swa();

    // sliding window switch pattern
    const int32_t sliding_window_pattern = 4;

    for (int il = 0; il < n_layer; ++il) {
        // three layers sliding window attention (window size 4096) and ROPE
        // fourth layer uses global attention without positional embeddings
        const bool           is_sliding = il % sliding_window_pattern < (sliding_window_pattern - 1);
        struct ggml_tensor * KQ_mask_l = is_sliding ? KQ_mask_swa : KQ_mask;

        // self-attention
        auto attn_out = build_std_attention(gf, model.layers[il].attn_norm, inpL, inp_pos, nullptr, KQ_mask_l, nullptr, nullptr, 1.0f / sqrtf(float(n_embd_head)), 0.f,
                is_sliding ? hparams.n_swa : 0, il, is_sliding, false, true, true);
        cb(attn_out, "attn_out", il);

        if (il == n_layer - 1) {
            // skip computing output for unused tokens
            struct ggml_tensor * inp_out_ids = build_inp_out_ids();
            attn_out                         = ggml_get_rows(ctx0, attn_out, inp_out_ids);
            inpL                             = ggml_get_rows(ctx0, inpL, inp_out_ids);
        }

        attn_out->op_params[3] = 1; // i.e., turn off the reduce operation as it is not required

        // feed-forward network
        cur = llm_build_ffn(ctx0, lctx, model.layers[il].attn_norm, inpL, model.layers[il].ffn_up, NULL, NULL, model.layers[il].ffn_gate,
                    NULL, NULL, model.layers[il].ffn_down, NULL, NULL, NULL, LLM_FFN_SILU, LLM_FFN_PAR,
                    cb, il, gf, false, true, attn_out);
        cb(cur, "ffn_out", il);

        // add together residual + FFN + self-attention
        cur = lctx.cvec.apply_to(ctx0, cur, il);
        cb(cur, "l_out", il);

        // input for next layer
        inpL = cur;
    }

    cur = inpL;

    cur = llm_build_norm(ctx0, cur, hparams, model.output_norm, NULL, LLM_NORM, cb, -1);
    cb(cur, "result_norm", -1);

    if (f_logit_scale) {
        cur = ggml_scale(ctx0, cur, f_logit_scale);
        cb(cur, "result_norm_scaled", -1);
    }

    // lm_head
    cur = llm_build_lora_mm(lctx, ctx0, model.output, cur);
    cb(cur, "result_output", -1);

    ggml_build_forward_expand(gf, cur);

    return gf;
}

ggml_cgraph * llm_build_context::build_t5_encoder() {
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, llama_model_max_nodes(model), false);

    // mutable variable, needed during the last layer of the computation to skip unused tokens
    int32_t n_tokens = this->n_tokens;

    const int64_t n_embd_head = hparams.n_embd_head_v;
    const int64_t n_embd_gqa  = hparams.n_embd_v_gqa();
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);

    struct ggml_tensor * cur;
    struct ggml_tensor * inpL;

    inpL = llm_build_inp_embd(ctx0, lctx, hparams, batch, model.tok_embd, cb);

    GGML_ASSERT(lctx.is_encoding);
    struct ggml_tensor * pos_bucket_enc = llm_build_pos_bucket(false);

    // KQ_mask (mask for 1 head, it will be broadcasted to all heads)
    struct ggml_tensor * KQ_mask_enc = build_inp_KQ_mask(false);

    for (int il = 0; il < n_layer; ++il) {
        struct ggml_tensor * inpSA = inpL;

        // norm
        cur = llm_build_norm(ctx0, inpL, hparams, model.layers[il].attn_norm_enc, NULL, LLM_NORM_RMS, cb, il);
        cb(cur, "attn_norm", il);

        // self-attention
        {
            auto [Qcur, Kcur, Vcur] = llm_build_mul_mat_qkv(gf, cur, model.layers[il].wq, nullptr,
                    model.layers[il].wk, nullptr,
                    model.layers[il].wv, nullptr, 0, il);

            Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens);
            Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);

            struct ggml_tensor * q =                 ggml_permute(ctx0, Qcur, 0, 2, 1, 3);
            struct ggml_tensor * k = ggml_cont(ctx0, ggml_permute(ctx0, Kcur, 0, 2, 1, 3));

            struct ggml_tensor * kq = ggml_mul_mat(ctx0, k, q);
            cb(kq, "kq", il);

            struct ggml_tensor * attn_rel_b = model.layers[il].attn_rel_b_enc ? model.layers[il].attn_rel_b_enc : model.layers[0].attn_rel_b_enc;
            struct ggml_tensor * pos_bias = llm_build_pos_bias(pos_bucket_enc, attn_rel_b);
            struct ggml_tensor * kq_b = ggml_add(ctx0, kq, pos_bias);
            cb(kq_b, "kq_b", il);

            kq = ggml_soft_max_ext(ctx0, kq_b, KQ_mask_enc, 1.0f, hparams.f_max_alibi_bias);
            cb(kq, "kq_soft_max_ext", il);

            struct ggml_tensor * v = ggml_cont(ctx0, ggml_transpose(ctx0, ggml_reshape_2d(ctx0, Vcur, n_embd_gqa, n_tokens)));
            cb(v, "v", il);

            struct ggml_tensor * kqv = ggml_mul_mat(ctx0, ggml_reshape_3d(ctx0, v, n_tokens, n_embd_head, n_head_kv), kq);
            cb(kqv, "kqv", il);

            struct ggml_tensor * kqv_merged = ggml_permute(ctx0, kqv, 0, 2, 1, 3);
            cb(kqv_merged, "kqv_merged", il);

            cur = ggml_cont_2d(ctx0, kqv_merged, n_embd_gqa, n_tokens);
            cb(cur, "kqv_merged_cont", il);

            ggml_build_forward_expand(gf, cur);

            cur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wo_enc, cur);
            cb(cur, "kqv_out", il);
        }

        if (il == n_layer - 1) {
            // skip computing output for unused tokens
            struct ggml_tensor * inp_out_ids = build_inp_out_ids();
            n_tokens = n_outputs;
            cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }

        struct ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        // feed-forward network
        {
            // T5 uses relu, flan-T5 uses gelu-gated
            cur = llm_build_ffn(ctx0, lctx, model.layers[il].ffn_norm_enc, ffn_inp,
                    model.layers[il].ffn_up_enc,   NULL, NULL,
                    model.layers[il].ffn_gate_enc, NULL, NULL,
                    model.layers[il].ffn_down_enc, NULL, NULL,
                    NULL,
                    model.layers[il].ffn_gate_enc ? LLM_FFN_GELU : LLM_FFN_RELU,
                    model.layers[il].ffn_gate_enc ? LLM_FFN_PAR  : LLM_FFN_SEQ,
                    cb, il);
            cb(cur, "ffn_out", il);
        }

        cur = ggml_add(ctx0, cur, ffn_inp);
        cb(cur, "ffn_out", il);

        ggml_tensor * layer_dir = lctx.cvec.tensor_for(il);
        if (layer_dir != nullptr) {
            cur = ggml_add(ctx0, cur, layer_dir);
        }
        cb(cur, "l_out", il);

        // input for next layer
        inpL = cur;
    }

    cur = inpL;
    cb(cur, "result_embd", -1);

    cur = llm_build_norm(ctx0, cur, hparams, model.output_norm_enc, NULL, LLM_NORM_RMS, cb, -1);
    cb(cur, "result_norm", -1);

    ggml_build_forward_expand(gf, cur);

    return gf;
}

ggml_cgraph * llm_build_context::build_t5_decoder() {
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, llama_model_max_nodes(model), false);

    // mutable variable, needed during the last layer of the computation to skip unused tokens
    int32_t n_tokens = this->n_tokens;

    const int64_t n_embd_head = hparams.n_embd_head_v;
    const int64_t n_embd_gqa  = hparams.n_embd_v_gqa();
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);

    struct ggml_tensor * cur;
    struct ggml_tensor * inpL;

    inpL = llm_build_inp_embd(ctx0, lctx, hparams, batch, model.tok_embd, cb);

    GGML_ASSERT(!lctx.is_encoding);
    GGML_ASSERT(n_outputs_enc > 0 && "call llama_encode() first");

    struct ggml_tensor * embd_enc       = llm_build_inp_embd_enc();
    struct ggml_tensor * pos_bucket_dec = llm_build_pos_bucket(true);

    struct ggml_tensor * KQ_mask_dec   = build_inp_KQ_mask();
    struct ggml_tensor * KQ_mask_cross = llm_build_inp_KQ_mask_cross();

    for (int il = 0; il < n_layer; ++il) {
        struct ggml_tensor * inpSA = inpL;

        // norm
        cur = llm_build_norm(ctx0, inpL, hparams, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, cb, il);
        cb(cur, "attn_norm", il);

        // self-attention
        {
            auto [Qcur, Kcur, Vcur] = llm_build_mul_mat_qkv(gf, cur, model.layers[il].wq, nullptr,
                    model.layers[il].wk, nullptr,
                    model.layers[il].wv, nullptr, 0, il);

            llm_build_kv_store(lctx, ctx0, hparams, cparams, kv_self, gf, Kcur, Vcur, n_tokens, kv_head, cb, il);

            struct ggml_tensor * k =
                ggml_view_3d(ctx0, kv_self.k_l[il],
                        n_embd_head_k, n_kv, n_head_kv,
                        ggml_row_size(kv_self.k_l[il]->type, n_embd_k_gqa),
                        ggml_row_size(kv_self.k_l[il]->type, n_embd_head_k),
                        0);
            cb(k, "k", il);

            struct ggml_tensor * v =
                ggml_view_3d(ctx0, kv_self.v_l[il],
                        n_kv, n_embd_head_v, n_head_kv,
                        ggml_element_size(kv_self.v_l[il])*n_ctx,
                        ggml_element_size(kv_self.v_l[il])*n_ctx*n_embd_head_v,
                        0);
            cb(v, "v", il);

            Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens);

            struct ggml_tensor * q = ggml_permute(ctx0, Qcur, 0, 2, 1, 3);

            struct ggml_tensor * kq = ggml_mul_mat(ctx0, k, q);
            cb(kq, "kq", il);

            struct ggml_tensor * attn_rel_b = model.layers[il].attn_rel_b ? model.layers[il].attn_rel_b : model.layers[0].attn_rel_b;
            struct ggml_tensor * pos_bias = llm_build_pos_bias(pos_bucket_dec, attn_rel_b);
            struct ggml_tensor * kq_b = ggml_add(ctx0, kq, pos_bias);
            cb(kq_b, "kq_b", il);

            kq = ggml_soft_max_ext(ctx0, kq_b, KQ_mask_dec, 1.0f, hparams.f_max_alibi_bias);
            cb(kq, "kq_soft_max_ext", il);

            struct ggml_tensor * kqv = ggml_mul_mat(ctx0, v, kq);
            cb(kqv, "kqv", il);

            struct ggml_tensor * kqv_merged = ggml_permute(ctx0, kqv, 0, 2, 1, 3);
            cb(kqv_merged, "kqv_merged", il);

            cur = ggml_cont_2d(ctx0, kqv_merged, n_embd_gqa, n_tokens);
            cb(cur, "kqv_merged_cont", il);

            ggml_build_forward_expand(gf, cur);

            cur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wo, cur);
            cb(cur, "kqv_out", il);
        }

        cur = ggml_add(ctx0, cur, inpSA);
        cb(cur, "cross_inp", il);

        struct ggml_tensor * inpCA = cur;

        // norm
        cur = llm_build_norm(ctx0, cur, hparams, model.layers[il].attn_norm_cross, NULL, LLM_NORM_RMS, cb, il);
        cb(cur, "attn_norm_cross", il);

        // cross-attention
        {
            struct ggml_tensor * Qcur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wq_cross, cur);
            cb(Qcur, "Qcur", il);

            struct ggml_tensor * Kcur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wk_cross, embd_enc);
            cb(Kcur, "Kcur", il);

            struct ggml_tensor * Vcur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wv_cross, embd_enc);
            cb(Vcur, "Vcur", il);

            Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
            Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_outputs_enc);

            struct ggml_tensor * q =                 ggml_permute(ctx0, Qcur, 0, 2, 1, 3);
            struct ggml_tensor * k = ggml_cont(ctx0, ggml_permute(ctx0, Kcur, 0, 2, 1, 3));

            struct ggml_tensor * kq = ggml_mul_mat(ctx0, k, q);
            cb(kq, "kq", il);

            kq = ggml_soft_max_ext(ctx0, kq, KQ_mask_cross, 1.0f, hparams.f_max_alibi_bias);
            cb(kq, "kq_soft_max_ext", il);

            struct ggml_tensor * v = ggml_cont(ctx0, ggml_transpose(ctx0, ggml_reshape_2d(ctx0, Vcur, n_embd_gqa, n_outputs_enc)));
            cb(v, "v", il);

            struct ggml_tensor * kqv = ggml_mul_mat(ctx0, ggml_reshape_3d(ctx0, v, n_outputs_enc, n_embd_head, n_head_kv), kq);
            cb(kqv, "kqv", il);

            struct ggml_tensor * kqv_merged = ggml_permute(ctx0, kqv, 0, 2, 1, 3);
            cb(kqv_merged, "kqv_merged", il);

            cur = ggml_cont_2d(ctx0, kqv_merged, n_embd_gqa, n_tokens);
            cb(cur, "kqv_merged_cont", il);

            ggml_build_forward_expand(gf, cur);

            cur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wo_cross, cur);
            cb(cur, "kqv_out", il);
        }

        if (il == n_layer - 1) {
            // skip computing output for unused tokens
            struct ggml_tensor * inp_out_ids = build_inp_out_ids();
            n_tokens = n_outputs;
            cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
            inpCA = ggml_get_rows(ctx0, inpCA, inp_out_ids);
        }

        struct ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpCA);
        cb(ffn_inp, "ffn_inp", il);

        // feed-forward network
        {
            // T5 uses relu, flan-T5 uses gelu-gated
            cur = llm_build_ffn(ctx0, lctx, model.layers[il].ffn_norm, ffn_inp,
                    model.layers[il].ffn_up,   NULL, NULL,
                    model.layers[il].ffn_gate, NULL, NULL,
                    model.layers[il].ffn_down, NULL, NULL,
                    NULL,
                    model.layers[il].ffn_gate_enc ? LLM_FFN_GELU : LLM_FFN_RELU,
                    model.layers[il].ffn_gate_enc ? LLM_FFN_PAR : LLM_FFN_SEQ,
                    cb, il);
            cb(cur, "ffn_out", il);
        }

        cur = ggml_add(ctx0, cur, ffn_inp);
        cb(cur, "ffn_out", il);

        ggml_tensor * layer_dir = lctx.cvec.tensor_for(il);
        if (layer_dir != nullptr) {
            cur = ggml_add(ctx0, cur, layer_dir);
        }
        cb(cur, "l_out", il);

        // input for next layer
        inpL = cur;
    }

    cur = inpL;
    cb(cur, "result_embd", -1);

    cur = llm_build_norm(ctx0, cur, hparams, model.output_norm, NULL, LLM_NORM_RMS, cb, -1);
    cb(cur, "result_norm", -1);

    // lm_head
    cur = llm_build_lora_mm(lctx, ctx0, model.output, cur);
    cb(cur, "result_output", -1);

    ggml_build_forward_expand(gf, cur);

    return gf;
}

ggml_cgraph * llm_build_context::build_jais() {
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, llama_model_max_nodes(model), false);

    const int64_t n_embd_head = hparams.n_embd_head_v;
    const int64_t n_embd_gqa  = hparams.n_embd_v_gqa();
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);

    struct ggml_tensor * cur;
    struct ggml_tensor * inpL;

    inpL = llm_build_inp_embd(ctx0, lctx, hparams, batch, model.tok_embd, cb);

    // KQ_mask (mask for 1 head, it will be broadcasted to all heads)
    struct ggml_tensor * KQ_mask = build_inp_KQ_mask();

    for (int il = 0; il < n_layer; ++il) {
        cur = llm_build_norm(ctx0, inpL, hparams, model.layers[il].attn_norm, model.layers[il].attn_norm_b, LLM_NORM, cb, il);
        cb(cur, "attn_norm", il);

        // self-attention
        {
            cur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wqkv, cur);
            cb(cur, "wqkv", il);

            cur = ggml_add(ctx0, cur, model.layers[il].bqkv);
            cb(cur, "bqkv", il);

            struct ggml_tensor * Qcur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd,     n_tokens, cur->nb[1], 0*cur->nb[0]*(n_embd)));
            struct ggml_tensor * Kcur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd_gqa, n_tokens, cur->nb[1], 1*cur->nb[0]*(n_embd)));
            struct ggml_tensor * Vcur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd_gqa, n_tokens, cur->nb[1], 1*cur->nb[0]*(n_embd + n_embd_gqa)));

            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens);

            cur = llm_build_kv(ctx0, lctx, kv_self, gf,
                    model.layers[il].wo, model.layers[il].bo,
                    Kcur, Vcur, Qcur, KQ_mask, n_tokens, kv_head, n_kv, 1.0f/float(n_embd_head), cb, il);
        }

        if (il == n_layer - 1) {
            // skip computing output for unused tokens
            struct ggml_tensor * inp_out_ids = build_inp_out_ids();
            cur  = ggml_get_rows(ctx0,  cur, inp_out_ids);
            inpL = ggml_get_rows(ctx0, inpL, inp_out_ids);
        }

        // add the input
        struct ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpL);
        cb(ffn_inp, "ffn_inp", il);

        // FF
        {
            cur = llm_build_ffn(ctx0, lctx, model.layers[il].ffn_norm, ffn_inp,
                    model.layers[il].ffn_up,   model.layers[il].ffn_up_b,   NULL,
                    model.layers[il].ffn_gate, model.layers[il].ffn_gate_b, NULL,
                    model.layers[il].ffn_down, model.layers[il].ffn_down_b, NULL,
                    NULL,
                    LLM_FFN_SILU, LLM_FFN_PAR, cb, il);
            cb(cur, "ffn_out", il);
        }

        inpL = ggml_add(ctx0, cur, ffn_inp);
        cb(inpL, "l_out", il);
    }

    cur = llm_build_norm(ctx0, inpL, hparams, model.output_norm, model.output_norm_b, LLM_NORM, cb, -1);
    cb(cur, "result_norm", -1);

    cur = llm_build_lora_mm(lctx, ctx0, model.output, cur);

    cb(cur, "result_output", -1);

    ggml_build_forward_expand(gf, cur);

    return gf;
}

ggml_cgraph * llm_build_context::build_chatglm() {
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, llama_model_max_nodes(model), false);

    const int64_t n_embd_head = hparams.n_embd_head_v;
    const int64_t n_embd_gqa  = hparams.n_embd_v_gqa();
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);

    struct ggml_tensor * cur;
    struct ggml_tensor * inpL;

    inpL = llm_build_inp_embd(ctx0, lctx, hparams, batch, model.tok_embd, cb);

    // inp_pos - contains the positions
    struct ggml_tensor * inp_pos = build_inp_pos();

    // KQ_mask (mask for 1 head, it will be broadcasted to all heads)
    struct ggml_tensor * KQ_mask = build_inp_KQ_mask();

    for (int il = 0; il < n_layer; ++il) {
        struct ggml_tensor * inpSA = inpL;

        cur = llm_build_norm(ctx0, inpL, hparams, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, cb, il);
        cb(cur, "attn_norm", il);

        // self-attention
        {
            struct ggml_tensor * Qcur = nullptr;
            struct ggml_tensor * Kcur = nullptr;
            struct ggml_tensor * Vcur = nullptr;

            cur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wqkv, cur);
            cb(cur, "wqkv", il);

            cur = ggml_add(ctx0, cur, model.layers[il].bqkv);
            cb(cur, "bqkv", il);

            Qcur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd,     n_tokens, cur->nb[1], 0*sizeof(float)*(n_embd)));
            Kcur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd_gqa, n_tokens, cur->nb[1], 1*sizeof(float)*(n_embd)));
            Vcur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd_gqa, n_tokens, cur->nb[1], 1*sizeof(float)*(n_embd + n_embd_gqa)));

            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);
            //printf("freq_base: %f freq_scale: %f ext_factor: %f attn_factor: %f\n", freq_base, freq_scale, ext_factor, attn_factor);
            Qcur = ggml_rope_ext(
                    ctx0, ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens), inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );
            cb(Qcur, "Qcur_rope", il);

            Kcur = ggml_rope_ext(
                    ctx0, ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens), inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );
            cb(Kcur, "Kcur_rope", il);

            cur = llm_build_kv(ctx0, lctx, kv_self, gf,
                    model.layers[il].wo, NULL,
                    Kcur, Vcur, Qcur, KQ_mask, n_tokens, kv_head, n_kv, 1.0f/sqrtf(float(n_embd_head)), cb, il);

        }

        if (il == n_layer - 1) {
            // skip computing output for unused tokens
            struct ggml_tensor * inp_out_ids = build_inp_out_ids();
            cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }

        // Add the input
        struct ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        // FF
        {
            cur = llm_build_ffn(ctx0, lctx, model.layers[il].ffn_norm, ffn_inp,
                    model.layers[il].ffn_up,   NULL, NULL,
                    NULL,                      NULL, NULL,
                    model.layers[il].ffn_down, NULL, NULL,
                    NULL,
                    LLM_FFN_SWIGLU, LLM_FFN_SEQ, cb, il);
            cb(cur, "ffn_out", il);

        }

        inpL = ggml_add(ctx0, cur, ffn_inp);
        cb(inpL, "l_out", il);
    }

    cur = llm_build_norm(ctx0, inpL, hparams, model.output_norm, NULL, LLM_NORM_RMS, cb, -1);
    cb(cur, "result_norm", -1);

    cur = llm_build_lora_mm(lctx, ctx0, model.output, cur);
    cb(cur, "result_output", -1);

    ggml_build_forward_expand(gf, cur);

    return gf;
}

ggml_cgraph * llm_build_context::build_glm4() {
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, llama_model_max_nodes(model), false);

    const int64_t n_embd_head = hparams.n_embd_head_v;
    const int64_t n_embd_gqa  = hparams.n_embd_v_gqa();

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);

    struct ggml_tensor * cur;
    struct ggml_tensor * inpL;

    inpL = llm_build_inp_embd(ctx0, lctx, hparams, batch, model.tok_embd, cb);

    // inp_pos - contains the positions
    struct ggml_tensor * inp_pos = build_inp_pos();

    // KQ_mask (mask for 1 head, it will be broadcasted to all heads)
    struct ggml_tensor * KQ_mask     = build_inp_KQ_mask();

    for (int il = 0; il < n_layer; ++il) {
        struct ggml_tensor * inpSA = inpL;

        // Pre-attention norm
        cur = llm_build_norm(ctx0, inpL, hparams, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, cb, il);
        cb(cur, "attn_norm", il);

        // self-attention
        {
            struct ggml_tensor * Qcur = nullptr;
            struct ggml_tensor * Kcur = nullptr;
            struct ggml_tensor * Vcur = nullptr;

            if (model.layers[il].wqkv == nullptr) {
                Qcur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wq, cur);
                if (model.layers[il].bq) {
                    Qcur = ggml_add(ctx0, Qcur, model.layers[il].bq);
                }
                Kcur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wk, cur);
                if (model.layers[il].bk) {
                    Kcur = ggml_add(ctx0, Kcur, model.layers[il].bk);
                }
                Vcur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wv, cur);
                if (model.layers[il].bv) {
                    Vcur = ggml_add(ctx0, Vcur, model.layers[il].bv);
                }
            } else {
                cur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wqkv, cur);
                cb(cur, "wqkv", il);
                if (model.layers[il].bqkv) {
                    cur = ggml_add(ctx0, cur, model.layers[il].bqkv);
                    cb(cur, "bqkv", il);
                }
                Qcur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd,     n_tokens, cur->nb[1], 0*sizeof(float)*(n_embd)));
                Kcur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd_gqa, n_tokens, cur->nb[1], 1*sizeof(float)*(n_embd)));
                Vcur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd_gqa, n_tokens, cur->nb[1], 1*sizeof(float)*(n_embd + n_embd_gqa)));
            }

            Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
            Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);

            Qcur = ggml_rope_ext(
                    ctx0, Qcur, inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );

            Kcur = ggml_rope_ext(
                    ctx0, Kcur, inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );

            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            cur = llm_build_kv(ctx0, lctx, kv_self, gf,
                    model.layers[il].wo, NULL,
                    Kcur, Vcur, Qcur, KQ_mask, n_tokens, kv_head, n_kv, 1.0f / sqrtf(float(n_embd_head)), cb, il);
        }

        if (il == n_layer - 1) {
            // skip computing output for unused tokens
            struct ggml_tensor * inp_out_ids = build_inp_out_ids();
            cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }

        // Post-attention norm (new!)
        cur = llm_build_norm(ctx0, cur, hparams, model.layers[il].attn_post_norm, NULL, LLM_NORM_RMS, cb, il);
        cb(cur, "post_attn_norm", il);

        // Add the input (residual connection after post-attention norm)
        struct ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        // FF
        {
            // MLP
            cur = llm_build_ffn(ctx0, lctx, model.layers[il].ffn_norm, ffn_inp,
                    model.layers[il].ffn_up, NULL, NULL,
                    NULL, NULL, NULL,
                    model.layers[il].ffn_down, NULL, NULL,
                    NULL,
                    LLM_FFN_SWIGLU, LLM_FFN_SEQ, cb, il);
            cb(cur, "ffn_out", il);

            // Post-MLP norm
            cur = llm_build_norm(ctx0, cur, hparams, model.layers[il].ffn_post_norm, NULL, LLM_NORM_RMS, cb, il);
            cb(cur, "post_mlp_norm", il);
        }

        // Add residual connection after post-MLP norm
        inpL = ggml_add(ctx0, cur, ffn_inp);
        cb(inpL, "l_out", il);
    }

    // Final norm
    cur = llm_build_norm(ctx0, inpL, hparams, model.output_norm, NULL, LLM_NORM_RMS, cb, -1);
    cb(cur, "result_norm", -1);

    // Output projection
    cur = llm_build_lora_mm(lctx, ctx0, model.output, cur);
    cb(cur, "result_output", -1);

    ggml_build_forward_expand(gf, cur);

    return gf;
}

ggml_cgraph * llm_build_context::build_dots1() {
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, llama_model_max_nodes(model), false);

    const int64_t n_embd_head = hparams.n_embd_head_v;

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);
    GGML_ASSERT(n_embd_head == hparams.n_rot);

    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = llm_build_inp_embd(ctx0, lctx, hparams, batch, model.tok_embd, cb);

    // inp_pos - contains the positions
    ggml_tensor * inp_pos = build_inp_pos();

    struct ggml_tensor * KQ_mask = build_inp_KQ_mask();

    for (int il = 0; il < n_layer; ++il) {
        ggml_tensor * inpSA = inpL;

        // norm
        cur = llm_build_norm(ctx0, inpL, hparams, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, cb, il);
        cb(cur, "attn_norm", il);

        // self_attention
        {
            // compute Q and K and RoPE them
            ggml_tensor * Qcur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wq, cur);
            cb(Qcur, "Qcur", il);

            ggml_tensor * Kcur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wk, cur);
            cb(Kcur, "Kcur", il);

            ggml_tensor * Vcur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wv, cur);
            cb(Vcur, "Vcur", il);

            Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
            Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);

            Qcur = llm_build_norm(ctx0, Qcur, hparams, model.layers[il].attn_q_norm, NULL, LLM_NORM_RMS, cb, il);
            cb(Qcur, "Qcur_normed", il);

            Qcur = ggml_rope_ext(
                    ctx0, Qcur, inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );

            Kcur = llm_build_norm(ctx0, Kcur, hparams, model.layers[il].attn_k_norm, NULL, LLM_NORM_RMS, cb, il);
            cb(Kcur, "Kcur_normed", il);

            Kcur = ggml_rope_ext(
                    ctx0, Kcur, inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );

            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);

            cur = llm_build_kv(ctx0, lctx, kv_self, gf,
                    model.layers[il].wo, model.layers[il].bo,
                    Kcur, Vcur, Qcur, KQ_mask, n_tokens, kv_head, n_kv, 1.0f/sqrtf(float(n_embd_head)), cb, il);

        }

        if (il == n_layer - 1) {
            // skip computing output for unused tokens
            ggml_tensor * inp_out_ids = build_inp_out_ids();
            cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }

        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        // MoE branch
        cur = llm_build_norm(ctx0, ffn_inp, hparams, model.layers[il].ffn_norm, NULL, LLM_NORM_RMS, cb, il);
        cb(cur, "ffn_norm", il);

        if ((uint32_t) il < hparams.n_layer_dense_lead) {
            cur = llm_build_ffn(ctx0, lctx, nullptr, cur,
                    model.layers[il].ffn_up,   NULL, NULL,
                    model.layers[il].ffn_gate, NULL, NULL,
                    model.layers[il].ffn_down, NULL, NULL,
                    NULL,
                    LLM_FFN_SILU, LLM_FFN_PAR, cb, il);
            cb(cur, "ffn_out", il);
        } else {
            ggml_tensor * moe_out =
                llm_build_moe_ffn(ctx0, lctx, cur,
                        model.layers[il].ffn_gate_inp,
                        model.layers[il].ffn_up_exps,
                        model.layers[il].ffn_gate_exps,
                        model.layers[il].ffn_down_exps,
                        model.layers[il].ffn_exp_probs_b,
                        n_expert, n_expert_used,
                        LLM_FFN_SILU, hparams.expert_weights_norm,
                        true, hparams.expert_weights_scale,
                        (enum llm_expert_gating_func_type) hparams.expert_gating_func,
                        cb, il, gf);
            cb(moe_out, "ffn_moe_out", il);

            {
                ggml_tensor * ffn_shexp = llm_build_ffn(ctx0, lctx, nullptr, cur,
                        model.layers[il].ffn_up_shexp,   NULL, NULL,
                        model.layers[il].ffn_gate_shexp, NULL, NULL,
                        model.layers[il].ffn_down_shexp, NULL, NULL,
                        NULL,
                        LLM_FFN_SILU, LLM_FFN_PAR, cb, il);
                cb(ffn_shexp, "ffn_shexp", il);

                cur = ggml_add(ctx0, moe_out, ffn_shexp);
                cb(cur, "ffn_out", il);
            }
        }

        cur = ggml_add(ctx0, cur, ffn_inp);
        cur = lctx.cvec.apply_to(ctx0, cur, il);
        cb(cur, "l_out", il);

        // input for next layer
        inpL = cur;
    }

    cur = inpL;

    cur = llm_build_norm(ctx0, cur, hparams, model.output_norm, NULL, LLM_NORM_RMS, cb,  -1);

    cb(cur, "result_norm", -1);

    // lm_head
    cur = llm_build_lora_mm(lctx, ctx0, model.output, cur);
    cb(cur, "result_output", -1);

    ggml_build_forward_expand(gf, cur);

    return gf;
}

ggml_cgraph * llm_build_context::build_ernie4_5() {
    struct ggml_cgraph* gf = ggml_new_graph_custom(ctx0, llama_model_max_nodes(model), false);
    const int64_t n_embd_head = hparams.n_embd_head_v;

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);
    GGML_ASSERT(n_embd_head == hparams.n_rot);

    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = llm_build_inp_embd(ctx0, lctx, hparams, batch, model.tok_embd, cb);

    // inp_pos - contains the positions
    ggml_tensor * inp_pos = build_inp_pos();
    ggml_tensor * KQ_mask = build_inp_KQ_mask();

    // output token IDs (for last layer cropping)
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    for (int il = 0; il < n_layer; ++il) {
        ggml_tensor * inpSA = inpL;
        // norm
        // Pre-attention norm
        cur = llm_build_norm(ctx0, inpL, hparams, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, cb, il);
        cb(cur, "attn_norm", il);

        // self-attention
        // self-attention
        {
            // Q, K, V projections
            ggml_tensor * Qcur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wq, cur);
            cb(Qcur, "Qcur", il);
            if (model.layers[il].bq) {
                Qcur = ggml_add(ctx0, Qcur, model.layers[il].bq);
                cb(Qcur, "Qcur", il);
            }

            ggml_tensor * Kcur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wk, cur);
            cb(Kcur, "Kcur", il);
            if (model.layers[il].bk) {
                Kcur = ggml_add(ctx0, Kcur, model.layers[il].bk);
                cb(Kcur, "Kcur", il);
            }
            cb(Kcur, "Kcur", il);
            ggml_tensor * Vcur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wv, cur);
            cb(Vcur, "Vcur", il);
            if (model.layers[il].bv) {
                Vcur = ggml_add(ctx0, Vcur, model.layers[il].bv);
                cb(Vcur, "Vcur", il);
            }

            // reshape for multi-head
            Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens);
            Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
            // Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

            // apply RoPE
            Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow);
            Kcur = ggml_rope_ext(ctx0, Kcur, inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow);
            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            cur = llm_build_kv(ctx0, lctx, kv_self, gf,
                    model.layers[il].wo, NULL,
                    Kcur, Vcur, Qcur, KQ_mask,
                    n_tokens, kv_head, n_kv,
                    1.0f / sqrtf(float(n_embd_head)), cb, il);
        }

        if (il == n_layer - 1 && inp_out_ids) {
            cur = ggml_get_rows(ctx0, cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }

        // residual connection for attention output
        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        // feed-forward network
        {
            cur = llm_build_ffn(ctx0, lctx, model.layers[il].ffn_norm, ffn_inp,
                    model.layers[il].ffn_up, NULL, NULL,
                    model.layers[il].ffn_gate, NULL, NULL,
                    model.layers[il].ffn_down, NULL, NULL,
                    NULL,
                    LLM_FFN_SILU, LLM_FFN_PAR, cb, il);
            cb(cur, "ffn_out", il);
        }
        cur = ggml_add(ctx0, cur, ffn_inp);
        cb(cur, "ffn_out", il);

        cur = lctx.cvec.apply_to(ctx0, cur, il);
        cb(cur, "l_out", il);

        // input for next layer
        inpL = cur;
    }

    cur = inpL;

    cur = llm_build_norm(ctx0, cur, hparams, model.output_norm, NULL, LLM_NORM_RMS, cb, -1);

    cb(cur, "result_norm", -1);
    // lm_head
    cur = llm_build_lora_mm(lctx, ctx0, model.output, cur);

    cb(cur, "result_output", -1);
    ggml_build_forward_expand(gf, cur);
    return gf;
}

ggml_cgraph * llm_build_context::build_ernie4_5_moe() {
    struct ggml_cgraph* gf = ggml_new_graph_custom(ctx0, llama_model_max_nodes(model), false);
    const int64_t n_embd_head = hparams.n_embd_head_v;

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);
    GGML_ASSERT(n_embd_head == hparams.n_rot);

    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = llm_build_inp_embd(ctx0, lctx, hparams, batch, model.tok_embd, cb);

    // inp_pos - contains the positions
    ggml_tensor * inp_pos = build_inp_pos();

    struct ggml_tensor * KQ_mask = build_inp_KQ_mask();

    // output token IDs (for last layer cropping)
    struct ggml_tensor * inp_out_ids = build_inp_out_ids();

    GGML_ASSERT(hparams.n_moe_layer_step > 0 && "Ernie 4.5 MoE requires n_moe_layer_step > 0");
    for (int il = 0; il < n_layer; ++il) {

        cur = build_std_attention(gf, model.layers[il].attn_norm, inpL, inp_pos, nullptr, KQ_mask, nullptr, nullptr,
                    1.0f/sqrtf(float(n_embd_head)), 0.0f, 0, il, true, false, true);

        if (il == n_layer - 1 && inp_out_ids) {
            cur = ggml_get_rows(ctx0, cur, inp_out_ids);
        }

        // feed-forward network
        bool is_moe_layer = static_cast<uint32_t>(il) >= hparams.n_layer_dense_lead && (il + 1) % hparams.n_moe_layer_step == 0;

        if (!is_moe_layer) {
            // dense FFN
            cur = llm_build_ffn(ctx0, lctx, model.layers[il].ffn_norm, cur,
                    model.layers[il].ffn_up,   NULL, NULL,
                    model.layers[il].ffn_gate, NULL, NULL,
                    model.layers[il].ffn_down, NULL, NULL,
                    NULL,
                    LLM_FFN_SILU, LLM_FFN_PAR, cb, il, gf, true);
            cb(cur, "ffn_out", il);
        } else {
            cur = llm_build_std_moe_ffn(ctx0, lctx, model.layers[il].ffn_norm, cur,
                    model.layers[il].ffn_gate_inp,  model.layers[il].ffn_gate_inp_b,
                    model.layers[il].ffn_up_exps,   model.layers[il].ffn_up_exps_b,
                    model.layers[il].ffn_gate_exps, model.layers[il].ffn_gate_exps_b,
                    model.layers[il].ffn_down_exps, model.layers[il].ffn_down_exps_b,
                    model.layers[il].ffn_exp_probs_b,
                    model.layers[il].ffn_up_shexp,    nullptr, // we don't have shared expert biases?
                    model.layers[il].ffn_gate_shexp,  nullptr,
                    model.layers[il].ffn_down_shexp,  nullptr,
                    n_expert, n_expert_used,
                    LLM_FFN_SILU, true, false, 0.0f,
                    LLM_EXPERT_GATING_FUNC_SOFTMAX,
                    LLM_FFN_SILU, cb, il, gf, true);
        }

        cur = lctx.cvec.apply_to(ctx0, cur, il);
        cb(cur, "l_out", il);

        // input for next layer
        inpL = cur;
    }

    cur = build_output(lctx, ctx0, inpL, model.output, model.output_norm, cb);
    cb(cur, "result_output", -1);

    ggml_build_forward_expand(gf, cur);
    return gf;
}

ggml_cgraph * llm_build_context::build_hunyuan_moe() {
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, llama_model_max_nodes(model), false);

    const int64_t n_embd_head = hparams.n_embd_head_v;

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);
    GGML_ASSERT(n_embd_head == hparams.n_rot);

    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = llm_build_inp_embd(ctx0, lctx, hparams, batch, model.tok_embd, cb);

    ggml_tensor * inp_pos = build_inp_pos();

    struct ggml_tensor * KQ_mask = build_inp_KQ_mask();

    const float kq_scale = 1.0f / sqrtf(float(n_embd_head));

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    for (int il = 0; il < n_layer; ++il) {

        cur = build_std_attention(gf, model.layers[il].attn_norm, inpL, inp_pos, nullptr, KQ_mask,
                nullptr, nullptr, kq_scale, 0.0f, 0, il, true, false, true);

        if (il == n_layer - 1 && inp_out_ids) {
            cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
        }

        cur = llm_build_std_moe_ffn(ctx0, lctx, model.layers[il].ffn_norm, cur,
                model.layers[il].ffn_gate_inp,  nullptr,
                model.layers[il].ffn_up_exps,   nullptr,
                model.layers[il].ffn_gate_exps, nullptr,
                model.layers[il].ffn_down_exps, nullptr,
                nullptr,
                model.layers[il].ffn_up_shexp,    nullptr, // we don't have shared expert biases?
                model.layers[il].ffn_gate_shexp,  nullptr,
                model.layers[il].ffn_down_shexp,  nullptr,
                n_expert, n_expert_used,
                LLM_FFN_SILU, true, false, 0.0f,
                LLM_EXPERT_GATING_FUNC_SOFTMAX,
                LLM_FFN_SILU, cb, il, gf, true);

        cur = lctx.cvec.apply_to(ctx0, cur, il);
        cb(cur, "l_out", il);

        inpL = cur;
    }

    cur = inpL;

    cur = build_output(lctx, ctx0, cur, model.output, model.output_norm, cb);
    cb(cur, "result_output", -1);

    ggml_build_forward_expand(gf, cur);
    return gf;

}

ggml_cgraph * llm_build_context::build_mimo2() {
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, llama_model_max_nodes(model), false);

    //const int64_t n_embd_head = hparams.n_embd_head_v;
    //GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);
    //GGML_ASSERT(n_embd_head == hparams.n_rot);

    struct ggml_tensor * cur;
    struct ggml_tensor * inpL;

    inpL = llm_build_inp_embd(ctx0, lctx, hparams, batch, model.tok_embd, cb);

    // inp_pos - contains the positions
    struct ggml_tensor * inp_pos = build_inp_pos();
    struct ggml_tensor * inp_out_ids = build_inp_out_ids();

    // KQ_mask (mask for 1 head, it will be broadcasted to all heads)
    struct ggml_tensor * KQ_mask = build_inp_KQ_mask();
    struct ggml_tensor * KQ_mask_swa = build_inp_KQ_mask_swa();

    for (int il = 0; il < n_layer; ++il) {
        const bool is_sliding = model.hparams.swa_layers[il];
        auto KQ_mask_l = is_sliding ? KQ_mask_swa : KQ_mask;

        cur = build_std_attention(gf, model.layers[il].attn_norm, inpL, inp_pos, nullptr, KQ_mask_l, model.layers[il].attn_sinks,
                nullptr, 1.0f/sqrtf(float(n_embd_head_k)), 0.0f, is_sliding ? hparams.n_swa : 0, il, true, false, true);

        if (il == n_layer - 1) {
            // skip computing output for unused tokens
            cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
        }

        auto ffn_inp = cur;

        if (model.layers[il].ffn_gate_inp == nullptr) {
            cur = llm_build_ffn(ctx0, lctx, model.layers[il].ffn_norm, ffn_inp,
                    model.layers[il].ffn_up,   NULL, NULL,
                    model.layers[il].ffn_gate, NULL, NULL,
                    model.layers[il].ffn_down, NULL, NULL,
                    NULL,
                    LLM_FFN_SILU, LLM_FFN_PAR, cb, il, gf, true);
            cb(cur, "ffn_out", il);
        } else {
            cur = llm_build_std_moe_ffn(ctx0, lctx, model.layers[il].ffn_norm, ffn_inp,
                    model.layers[il].ffn_gate_inp,  nullptr,
                    model.layers[il].ffn_up_exps,   nullptr,
                    model.layers[il].ffn_gate_exps, nullptr,
                    model.layers[il].ffn_down_exps, nullptr,
                    model.layers[il].ffn_exp_probs_b,
                    nullptr,  nullptr, // we don't have shared expert biases?
                    nullptr,  nullptr,
                    nullptr,  nullptr,
                    n_expert, n_expert_used,
                    LLM_FFN_SILU, true, false, 0.0f,
                    LLM_EXPERT_GATING_FUNC_SIGMOID,
                    LLM_FFN_SILU, cb, il, gf, true);
        }

        cur = lctx.cvec.apply_to(ctx0, cur, il);
        cb(cur, "l_out", il);

        // input for next layer
        inpL = cur;
    }

    cur = inpL;

    cur = build_output(lctx, ctx0, cur, model.output, model.output_norm, cb);
    cb(cur, "result_output", -1);

    ggml_build_forward_expand(gf, cur);

    return gf;
}

ggml_cgraph * llm_build_context::build_openai_moe() {
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, llama_model_max_nodes(model), false);

    const int64_t n_embd_head = hparams.n_embd_head_v;
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);

    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = llm_build_inp_embd(ctx0, lctx, hparams, batch, model.tok_embd, cb);

    ggml_tensor * inp_pos = build_inp_pos();

    struct ggml_tensor * KQ_mask     = build_inp_KQ_mask();
    struct ggml_tensor * KQ_mask_swa = build_inp_KQ_mask_swa();
    const float kq_scale = 1.0f / sqrtf(float(n_rot));

    const int sliding_window_pattern = 2;

    for (int il = 0; il < n_layer; ++il) {
        const bool is_sliding = il % sliding_window_pattern < (sliding_window_pattern - 1);

        struct ggml_tensor * KQ_mask_l = is_sliding ? KQ_mask_swa : KQ_mask;

        cur = build_std_attention(gf, model.layers[il].attn_norm, inpL, inp_pos, nullptr, KQ_mask_l,
                model.layers[il].attn_sinks, nullptr, kq_scale, 0.0f, is_sliding ? hparams.n_swa : 0, il, true, false, true);

        if (il == n_layer - 1) {
            // skip computing output for unused tokens
            ggml_tensor * inp_out_ids = build_inp_out_ids();
            cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
        }

        bool use_dup_bias = cur->ne[1] < 32 && model.layers[il].ffn_up_exps_b_dup &&
                                               model.layers[il].ffn_gate_exps_b_dup &&
                                               model.layers[il].ffn_down_exps_b_dup;

        cur = llm_build_std_moe_ffn(ctx0, lctx, model.layers[il].ffn_norm, cur,
                model.layers[il].ffn_gate_inp,  model.layers[il].ffn_gate_inp_b,
                model.layers[il].ffn_up_exps,   use_dup_bias ? model.layers[il].ffn_up_exps_b_dup : model.layers[il].ffn_up_exps_b,
                model.layers[il].ffn_gate_exps, use_dup_bias ? model.layers[il].ffn_gate_exps_b_dup : model.layers[il].ffn_gate_exps_b,
                model.layers[il].ffn_down_exps, use_dup_bias ? model.layers[il].ffn_down_exps_b_dup : model.layers[il].ffn_down_exps_b,
                nullptr,
                nullptr,  nullptr, nullptr,  nullptr, nullptr,  nullptr, // no shared experts
                n_expert, n_expert_used,
                LLM_FFN_SWIGLU_OAI_MOE, false, false, 0.0f,
                LLM_EXPERT_GATING_FUNC_TYPE_SOFTMAX_WEIGHT,
                LLM_FFN_SWIGLU_OAI_MOE, cb, il, gf, true,
                model.layers[il].ffn_up_gate_exps, model.layers[il].ffn_up_gate_exps_b);

        cur = lctx.cvec.apply_to(ctx0, cur, il);
        cb(cur, "l_out", il);

        // input for next layer
        inpL = cur;
    }

    cur = build_output(lctx, ctx0, inpL, model.output, model.output_norm, cb);
    cb(cur, "result_output", -1);

    ggml_build_forward_expand(gf, cur);

    return gf;

}

ggml_cgraph * llm_build_context::build_bailingmoe2() {
    ggml_cgraph * gf = ggml_new_graph_custom(ctx0, llama_model_max_nodes(model), false);
    const int64_t n_embd_head = hparams.n_embd_head_v;
    //const int64_t n_embd_gqa  = hparams.n_embd_v_gqa();

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);

    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = llm_build_inp_embd(ctx0, lctx, hparams, batch, model.tok_embd, cb);

    // inp_pos - contains the positions
    ggml_tensor * inp_pos = build_inp_pos();

    //auto * inp_attn = build_attn_inp_kv();
    ggml_tensor * KQ_mask     = build_inp_KQ_mask();
    //const int64_t n_embd_head = hparams.n_embd_head_v;
    const float kq_scale = 1.0f / sqrtf(float(n_embd_head));

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    const int n_transformer_layers = n_layer - hparams.nextn_predict_layers;

    auto rope_cache = cparams.rope_cache && (rope_type == LLAMA_ROPE_TYPE_NEOX || rope_type == LLAMA_ROPE_TYPE_NORM) ?
        ggml_rope_cache(ctx0, inp_pos, nullptr, n_embd_head, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow) : nullptr;

    for (int il = 0; il < n_transformer_layers; ++il) {
        ggml_tensor * inpSA = inpL;

        // norm
        cur = llm_build_norm(ctx0, inpL, hparams, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, cb, il);
        cb(cur, "attn_norm", il);

        // self_attention
        {
            auto [Qcur, Kcur, Vcur] = llm_build_mul_mat_qkv(gf, cur, model.layers[il].wqkv, model.layers[il].bqkv,
                    nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                    model.layers[il].attn_q_norm, model.layers[il].attn_k_norm, 0.0f, il);

            if (rope_cache) {
                Qcur = ggml_rope_fast(ctx0, Qcur, rope_cache);
                Kcur = ggml_rope_fast(ctx0, Kcur, rope_cache);
            } else {
                Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                        ext_factor, attn_factor, beta_fast, beta_slow);
                Kcur = ggml_rope_ext(ctx0, Kcur, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                        ext_factor, attn_factor, beta_fast, beta_slow);
            }

            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            cur = llm_build_kv(ctx0, lctx, kv_self, gf, model.layers[il].wo, model.layers[il].bo,
                    Kcur, Vcur, Qcur, KQ_mask, n_tokens, kv_head, n_kv, kq_scale, cb, il);
        }
        if (il == n_transformer_layers - 1 && inp_out_ids) {
            cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }

        ggml_tensor * sa_out = ggml_add(ctx0, cur, inpSA);
        cb(sa_out, "sa_out", il);

        // MoE branch
        cur = llm_build_norm(ctx0, sa_out, hparams, model.layers[il].ffn_norm, NULL, LLM_NORM_RMS, cb, il);
        cb(cur, "ffn_norm", il);

        if (static_cast<uint32_t>(il) < hparams.n_layer_dense_lead) {
            cur = llm_build_ffn(ctx0, lctx, nullptr, cur,
                    model.layers[il].ffn_up,   NULL, NULL,
                    model.layers[il].ffn_gate, NULL, NULL,
                    model.layers[il].ffn_down, NULL, NULL,
                    NULL,
                    LLM_FFN_SILU, LLM_FFN_PAR, cb, il);
            cb(cur, "ffn_out", il);
        } else {

            ggml_tensor * moe_out =
                llm_build_moe_ffn(ctx0, lctx, cur,
                        model.layers[il].ffn_gate_inp,
                        model.layers[il].ffn_up_exps,
                        model.layers[il].ffn_gate_exps,
                        model.layers[il].ffn_down_exps,
                        model.layers[il].ffn_exp_probs_b,
                        n_expert, n_expert_used,
                        LLM_FFN_SILU, hparams.expert_weights_norm,
                        true, hparams.expert_weights_scale,
                        (llm_expert_gating_func_type) hparams.expert_gating_func,
                        cb, il, gf);
            cb(moe_out, "ffn_moe_out", il);

            ggml_tensor * ffn_shexp = llm_build_ffn(ctx0, lctx, nullptr, cur,
                    model.layers[il].ffn_up_shexp,   NULL, NULL,
                    model.layers[il].ffn_gate_shexp, NULL, NULL,
                    model.layers[il].ffn_down_shexp, NULL, NULL,
                    NULL,
                    LLM_FFN_SILU, LLM_FFN_PAR, cb, il);
            cb(ffn_shexp, "ffn_shexp", il);

            cur = ggml_add(ctx0, moe_out, ffn_shexp);
            cb(cur, "ffn_out", il);
        }

        cur = ggml_add(ctx0, cur, sa_out);

        cur = lctx.cvec.apply_to(ctx0, cur, il);
        cb(cur, "l_out", il);

        // input for next layer
        inpL = cur;
    }
    cur = inpL;

    cur = llm_build_norm(ctx0, cur, hparams, model.output_norm, NULL, LLM_NORM_RMS, cb, -1);

    cb(cur, "result_norm", -1);

    // lm_head
    cur = llm_build_lora_mm(lctx, ctx0, model.output, cur);

    cb(cur, "result_output", -1);

    ggml_build_forward_expand(gf, cur);
    return gf;
}

ggml_cgraph* llm_build_context::build_minimaxm2() {
    ggml_cgraph * gf = ggml_new_graph_custom(ctx0, llama_model_max_nodes(model), false);
    const int64_t n_embd_head = hparams.n_embd_head_v;
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);
    // GGML_ASSERT(n_embd_head == hparams.n_rot); this is wrong in case of minimax, head_dim = 128, n_rot = 64

    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = llm_build_inp_embd(ctx0, lctx, hparams, batch, model.tok_embd, cb);

    ggml_tensor * inp_pos = build_inp_pos();


    //auto * inp_attn = build_attn_inp_kv();
    ggml_tensor * inp_out_ids = build_inp_out_ids();
    ggml_tensor * KQ_mask = build_inp_KQ_mask();

    for (int il = 0; il < n_layer; ++il) {
        ggml_tensor* inpSA = inpL;

        cur = inpL;

        // self_attention
        {
            cur = llm_build_norm(ctx0, inpL, hparams, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, cb, il);
            cb(cur, "attn_norm", il);

            // Q, K, V projections
            ggml_tensor * Qcur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wq, cur);
            cb(Qcur, "Qcur", il);

            ggml_tensor * Kcur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wk, cur);
            cb(Kcur, "Kcur", il);

            ggml_tensor * Vcur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wv, cur);
            cb(Vcur, "Vcur", il);

            Qcur = llm_build_norm(ctx0, Qcur, hparams, model.layers[il].attn_q_norm, NULL,
                LLM_NORM_RMS, cb, il);
            cb(Qcur, "Qcur_normed", il);

            Kcur = llm_build_norm(ctx0, Kcur, hparams, model.layers[il].attn_k_norm, NULL,
                LLM_NORM_RMS, cb, il);
            cb(Kcur, "Kcur_normed", il);

            // reshape for multi-head
            Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens);
            Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
            // Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);


            // apply RoPE
            Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, nullptr,
                n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                ext_factor, attn_factor, beta_fast, beta_slow);
            Kcur = ggml_rope_ext(ctx0, Kcur, inp_pos, nullptr,
                n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                ext_factor, attn_factor, beta_fast, beta_slow);
            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            cur = llm_build_kv(ctx0, lctx, kv_self, gf,
                model.layers[il].wo, NULL,
                Kcur, Vcur, Qcur, KQ_mask,
                n_tokens, kv_head, n_kv,
                1.0f / sqrtf(float(n_embd_head)), cb, il);
        }

        if (il == n_layer - 1 && inp_out_ids) {
            cur = ggml_get_rows(ctx0, cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }

        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        // MoE branch
        cur = llm_build_norm(ctx0, ffn_inp, hparams,
            model.layers[il].ffn_norm, NULL,
            LLM_NORM_RMS,cb, il);
        cb(cur, "ffn_norm", il);

        cur = llm_build_moe_ffn(ctx0, lctx, cur,
                model.layers[il].ffn_gate_inp,
                model.layers[il].ffn_up_exps,
                model.layers[il].ffn_gate_exps,
                model.layers[il].ffn_down_exps,
                model.layers[il].ffn_exp_probs_b,
                n_expert, n_expert_used,
                LLM_FFN_SILU, true,
                false, 0,
                (llm_expert_gating_func_type)hparams.expert_gating_func,
                cb, il, gf);
        cb(cur, "ffn_moe_out", il);

        cur = ggml_add(ctx0, cur, ffn_inp);

        cur = lctx.cvec.apply_to(ctx0, cur, il);
        cb(cur, "l_out", il);

        // input for next layer
        inpL = cur;
    }

    cur = inpL;

    cur = llm_build_norm(ctx0, cur,
        hparams, model.output_norm, NULL,
        LLM_NORM_RMS, cb, -1);

    cb(cur, "result_norm", -1);

    // lm_head
    cur = llm_build_lora_mm(lctx, ctx0, model.output, cur);

    cb(cur, "result_output", -1);

    ggml_build_forward_expand(gf, cur);
    return gf;
}

ggml_cgraph* llm_build_context::build_smollm3() {
    ggml_cgraph * gf = ggml_new_graph_custom(ctx0, llama_model_max_nodes(model), false);
    const int64_t n_embd_head = hparams.n_embd_head_v;
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);
    // GGML_ASSERT(n_embd_head == hparams.n_rot); this is wrong in case of minimax, head_dim = 128, n_rot = 64

    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = llm_build_inp_embd(ctx0, lctx, hparams, batch, model.tok_embd, cb);

    ggml_tensor * inp_pos = build_inp_pos();


    //auto * inp_attn = build_attn_inp_kv();
    ggml_tensor * inp_out_ids = build_inp_out_ids();
    ggml_tensor * KQ_mask = build_inp_KQ_mask();

    const float kq_scale = hparams.f_attention_scale == 0.0f ? 1.0f/sqrtf(float(n_embd_head)) : hparams.f_attention_scale;

    for (int il = 0; il < n_layer; ++il) {
        ggml_tensor * inpSA = inpL;

        const bool use_rope = (il + 1) % hparams.n_no_rope_layer_step != 0;

        // norm
        cur = llm_build_norm(ctx0, inpL, hparams, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, cb, il);
        cb(cur, "attn_norm", il);

        // self-attention
        {
            auto [Qcur, Kcur, Vcur] = llm_build_mul_mat_qkv(gf, cur,
                    model.layers[il].wqkv, model.layers[il].bqkv,
                    model.layers[il].wqk,  model.layers[il].bqk,
                    model.layers[il].wq,   model.layers[il].bq,
                    model.layers[il].wk,   model.layers[il].bk,
                    model.layers[il].wv,   model.layers[il].bv,
                    model.layers[il].attn_q_norm, model.layers[il].attn_k_norm, 0, il);

            if (use_rope) {
                Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                        ext_factor, attn_factor, beta_fast, beta_slow);
                cb(Qcur, "Qcur", il);

                Kcur = ggml_rope_ext(ctx0, Kcur, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                        ext_factor, attn_factor, beta_fast, beta_slow);
                cb(Kcur, "Kcur", il);
            }

            cur = llm_build_kv(ctx0, lctx, kv_self, gf,
                    model.layers[il].wo, model.layers[il].bo,
                    Kcur, Vcur, Qcur, KQ_mask, n_tokens, kv_head, n_kv, kq_scale, cb, il);
            cb(cur, "attn_out", il);
        }
        if (il == n_layer - 1 && inp_out_ids) {
            cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }
        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        // feed-forward network
        cur = llm_build_ffn(ctx0, lctx, model.layers[il].ffn_norm, ffn_inp,
                model.layers[il].ffn_up,   NULL, NULL,
                model.layers[il].ffn_gate, NULL, NULL,
                model.layers[il].ffn_down, NULL, NULL,
                NULL,
                LLM_FFN_SILU, LLM_FFN_PAR, cb, il);
        cb(cur, "ffn_out", il);

        cur = ggml_add(ctx0, cur, ffn_inp);
        cur = lctx.cvec.apply_to(ctx0, cur, il);
        cb(cur, "l_out", il);

        // input for next layer
        inpL = cur;
    }
    cur = inpL;

    cur = llm_build_norm(ctx0, cur, hparams, model.output_norm, NULL, LLM_NORM_RMS, cb, -1);
    cb(cur, "result_norm", -1);

    // lm_head
    cur = llm_build_lora_mm(lctx, ctx0, model.output, cur);
    cb(cur, "result_output", -1);

    ggml_build_forward_expand(gf, cur);

    return gf;
}

ggml_cgraph * llm_build_context::llama_build_graph_defrag(llama_context & lctx, const std::vector<uint32_t> & ids) {
    llama_batch dummy;
    dummy.n_tokens = 0;

    llm_build_cb cb = [&](struct ggml_tensor * , const char * , int ) { };

    struct llm_build_context llm(lctx, dummy, cb, false, false);

    llm.init();

    struct ggml_cgraph * result = llm.build_defrag(ids);

    llm.free();

    return result;
}

ggml_cgraph * llm_build_context::llama_build_graph_k_shift(llama_context & lctx) {
    llama_batch dummy;
    dummy.n_tokens = 0;

    llm_build_cb cb = [&](struct ggml_tensor * , const char * , int ) { };

    struct llm_build_context llm(lctx, dummy, cb, false, false);

    llm.init();

    struct ggml_cgraph * result = llm.build_k_shift();

    llm.free();

    return result;
}

struct ggml_cgraph * llm_build_context::llama_build_graph_s_copy(llama_context & lctx) {
    llama_batch dummy;
    dummy.n_tokens = 0;

    llm_build_cb cb = [&](struct ggml_tensor * , const char * , int ) { };

    struct llm_build_context llm(lctx, dummy, cb, false, false);

    llm.init();

    struct ggml_cgraph * result = llm.build_s_copy();

    llm.free();

    return result;
}

ggml_cgraph * llm_build_context::llama_build_graph(
         llama_context & lctx,
     const llama_batch & batch,
                  bool   worst_case) {
    const auto & model = lctx.model;

#if IK_PRINT_TIMING
    auto tim1 = ggml_time_us();
#endif

    // this callback allows us to apply custom logic to each tensor (e.g. ggml-alloc, offloading, etc.)
    llm_build_cb cb = [&](struct ggml_tensor * cur, const char * name, int il) {
        if (il >= 0) {
            int j = 0;
            for (; j < GGML_MAX_NAME - 1; ++j) {
                cur->name[j] = name[j];
                if (!name[j]) break;
            }
            if (j < GGML_MAX_NAME - 3) {
                cur->name[j++] = '-';
                auto sil = std::to_string(il);
                for (int k = 0; k < (int)sil.size() && j < GGML_MAX_NAME - 1; ++k) {
                    cur->name[j++] = sil[k];
                }
            }
            cur->name[j] = 0;
            //ggml_format_name(cur, "%s-%d", name, il);
        } else {
            ggml_set_name(cur, name);
        }

        if (!lctx.cparams.offload_kqv) {
            if (strcmp(name, "kqv_merged_cont") == 0) {
                // all nodes between the KV store and the attention output are run on the CPU
                ggml_backend_sched_set_tensor_backend(lctx.sched, cur, lctx.backend_cpu);
            }
        }

        // norm may be automatically assigned to the backend of the previous layer, increasing data transfer between backends
        // FIXME: fix in ggml_backend_sched
        const bool full_offload = lctx.model.n_gpu_layers > (int)lctx.model.hparams.n_layer;
        if (batch.n_tokens < 32 || full_offload) {
            if (il != -1 && strcmp(name, "norm") == 0) {
                for (auto * backend : lctx.backends) {
                    if (ggml_backend_supports_buft(backend, lctx.model.buft_layer[il].buft) &&
                        (ggml_backend_supports_op(backend, cur) || ggml_backend_offload_op(backend, cur))) {
                        ggml_backend_sched_set_tensor_backend(lctx.sched, cur, backend);
                        break;
                    }
                }
            }
        }
    };

    struct ggml_cgraph * result = NULL;

    const llama_vocab * vocab = &lctx.model.vocab; //llama_get_vocab(&lctx);
    llama_token bos = vocab->token_bos();
    llama_token eos = vocab->token_eos();
    bool is_warming_up = lctx.n_eval == 0 && (batch.n_tokens == 1 && (batch.token[0] == ((bos != -1) ? bos : eos)));
    struct llm_build_context llm(lctx, batch, cb, worst_case, is_warming_up);

    llm.init();

    switch (model.arch) {
        case LLM_ARCH_LLAMA:
        case LLM_ARCH_LLAMA4:
        case LLM_ARCH_GRANITE:
        case LLM_ARCH_GRANITE_MOE:
            {
                result = llm.build_llama();
            } break;
        case LLM_ARCH_DECI:
            {
                result = llm.build_deci();
            } break;
        case LLM_ARCH_BAICHUAN:
            {
                result = llm.build_baichuan();
            } break;
        case LLM_ARCH_FALCON:
            {
                result = llm.build_falcon();
            } break;
        case LLM_ARCH_GROK:
            {
                result = llm.build_grok();
            } break;
        case LLM_ARCH_STARCODER:
            {
                result = llm.build_starcoder();
            } break;
        case LLM_ARCH_REFACT:
            {
                result = llm.build_refact();
            } break;
        case LLM_ARCH_BERT:
        case LLM_ARCH_JINA_BERT_V2:
        case LLM_ARCH_NOMIC_BERT:
            {
                result = llm.build_bert();
            } break;
        case LLM_ARCH_BLOOM:
            {
                result = llm.build_bloom();
            } break;
        case LLM_ARCH_MPT:
            {
                result = llm.build_mpt();
            } break;
         case LLM_ARCH_STABLELM:
            {
                result = llm.build_stablelm();
            } break;
        case LLM_ARCH_QWEN:
            {
                result = llm.build_qwen();
            } break;
        case LLM_ARCH_QWEN2:
            {
                result = llm.build_qwen2();
            } break;
        case LLM_ARCH_QWEN2VL:
            {
                result = llm.build_qwen2vl();
            } break;
        case LLM_ARCH_QWEN2MOE:
            {
                result = llm.build_qwen2moe();
            } break;
        case LLM_ARCH_QWEN3:
            {
                result = llm.build_qwen3();
            } break;
        case LLM_ARCH_QWEN3MOE:
            {
                result = llm.build_qwen3moe();
            } break;
        case LLM_ARCH_QWEN3VL:
            {
                result = llm.build_qwen3vl();
            } break;
        case LLM_ARCH_QWEN3VLMOE:
            {
                result = llm.build_qwen3vlmoe();
            } break;
        case LLM_ARCH_PHI2:
            {
                result = llm.build_phi2();
            } break;
        case LLM_ARCH_PHI3:
            {
                result = llm.build_phi3();
            } break;
        case LLM_ARCH_PLAMO:
            {
                result = llm.build_plamo();
            } break;
        case LLM_ARCH_GPT2:
            {
                result = llm.build_gpt2();
            } break;
        case LLM_ARCH_CODESHELL:
            {
                result = llm.build_codeshell();
            } break;
        case LLM_ARCH_ORION:
            {
                result = llm.build_orion();
            } break;
        case LLM_ARCH_INTERNLM2:
            {
                result = llm.build_internlm2();
            } break;
        case LLM_ARCH_MINICPM:
            {
                result = llm.build_minicpm();
            } break;
        case LLM_ARCH_GEMMA:
            {
                result = llm.build_gemma();
            } break;
        case LLM_ARCH_GEMMA2:
            {
                result = llm.build_gemma2();
            } break;
        case LLM_ARCH_GEMMA3:
            {
                result = llm.build_gemma3();
            } break;
        case LLM_ARCH_STARCODER2:
            {
                result = llm.build_starcoder2();
            } break;
        case LLM_ARCH_MAMBA:
            {
                result = llm.build_mamba();
            } break;
        case LLM_ARCH_XVERSE:
            {
                result = llm.build_xverse();
            } break;
        case LLM_ARCH_COMMAND_R:
            {
                result = llm.build_command_r();
            } break;
        case LLM_ARCH_DBRX:
            {
                result = llm.build_dbrx();
            } break;
        case LLM_ARCH_OLMO:
            {
                result = llm.build_olmo();
            } break;
        case LLM_ARCH_OPENELM:
            {
                result = llm.build_openelm();
            } break;
        case LLM_ARCH_GPTNEOX:
            {
                result = llm.build_gptneox();
            } break;
        case LLM_ARCH_ARCTIC:
            {
                result = llm.build_arctic();
            } break;
        case LLM_ARCH_DEEPSEEK2:
            {
                result = llm.build_deepseek2();
            } break;
        case LLM_ARCH_CHATGLM:
            {
                result = llm.build_chatglm();
            } break;
        case LLM_ARCH_GLM4:
            {
                result = llm.build_glm4();
            } break;
        case LLM_ARCH_GLM4_MOE:
            {
                result = llm.build_glm4_moe();
            } break;
        case LLM_ARCH_BITNET:
            {
                result = llm.build_bitnet();
            } break;
        case LLM_ARCH_BITNET_B158:
        case LLM_ARCH_BITNET_25:
            {
                result = llm.build_bitnet_158();
            } break;
        case LLM_ARCH_COHERE2:
            {
                result = llm.build_cohere2();
            } break;
        case LLM_ARCH_T5:
            {
                if (lctx.is_encoding) {
                    result = llm.build_t5_encoder();
                } else {
                    result = llm.build_t5_decoder();
                }
            } break;
        case LLM_ARCH_T5ENCODER:
            {
                result = llm.build_t5_encoder();
            } break;
        case LLM_ARCH_JAIS:
            {
                result = llm.build_jais();
            } break;
        case LLM_ARCH_DOTS1:
            {
                result = llm.build_dots1();
            } break;
        case LLM_ARCH_ERNIE4_5:
        {
            result = llm.build_ernie4_5();
        } break;
        case LLM_ARCH_ERNIE4_5_MOE:
        {
            result = llm.build_ernie4_5_moe();
        } break;
        case LLM_ARCH_HUNYUAN_MOE:
            {
                result = llm.build_hunyuan_moe();
            } break;
        case LLM_ARCH_OPENAI_MOE:
            {
                result = llm.build_openai_moe();
            } break;
        case LLM_ARCH_BAILINGMOE2:
            {
                result = llm.build_bailingmoe2();
            } break;
        case LLM_ARCH_MINIMAX_M2:
            {
                result = llm.build_minimaxm2();
            } break;
        case LLM_ARCH_SMOLLM3:
            {
                result = llm.build_smollm3();
            } break;
        case LLM_ARCH_MISTRAL3:
            {
                result = llm.build_mistral3();
            } break;
        case LLM_ARCH_MIMO2:
            {
                result = llm.build_mimo2();
            } break;
        default:
            GGML_ABORT("fatal error");
    }

    // add on pooling layer
    if (lctx.cparams.embeddings) {
        result = llm.append_pooling(result);
    }

    llm.free();

#if IK_PRINT_TIMING
    auto tim2 = ggml_time_us();
    printf("%s(...): %d us\n", __func__, int(tim2-tim1));
#endif

    return result;
}

ggml_tensor * llm_build_context::build_std_attention(ggml_cgraph * gf, ggml_tensor * the_attn_norm,
        ggml_tensor * input, ggml_tensor * inp_pos, ggml_tensor * rope_factors_in,
        ggml_tensor * KQ_mask, ggml_tensor * sinks, ggml_tensor * inp_attn_scale, float KQ_scale, float f_attn_scale,
        int n_swa, int il, bool do_rope, bool add_graph_split, bool add_input, bool is_norm, bool is_multi) {

    float freq_base_l  = n_swa > 0 ? hparams.rope_freq_base_train_swa : cparams.rope_freq_base;
    float freq_scale_l = n_swa > 0 ? hparams.rope_freq_scale_train_swa : hparams.rope_freq_scale_train;

    if (!model.layers[il].wqkv && !model.layers[il].wqk && cparams.flash_attn &&
         model.layers[il].wq->extra && model.layers[il].wk->extra && model.layers[il].wv->extra && model.layers[il].wo->extra) {
        if (kv_self.k_l[il]->extra && kv_self.v_l[il]->extra) {
            ggml_split_tensor_t * attn_norm = the_attn_norm ? (ggml_split_tensor_t *)the_attn_norm->extra : nullptr;
            auto wq = (ggml_split_tensor_t *)model.layers[il].wq->extra;
            auto wk = (ggml_split_tensor_t *)model.layers[il].wk->extra;
            auto wv = (ggml_split_tensor_t *)model.layers[il].wv->extra;
            auto wo = (ggml_split_tensor_t *)model.layers[il].wo->extra;
            GGML_ASSERT(wq->n_device == wk->n_device && wq->n_device == wv->n_device && wq->n_device == wo->n_device);
            auto kl = (ggml_split_tensor_t *)kv_self.k_l[il]->extra;
            auto vl = (ggml_split_tensor_t *)kv_self.v_l[il]->extra;
            GGML_ASSERT(wq->n_device == kl->n_device && wq->n_device == vl->n_device);
            ggml_split_tensor_t *bq = nullptr, *bo = nullptr, *bk = nullptr, *bv = nullptr;
            if (model.layers[il].bq && model.layers[il].bq->extra) {
                bq = (ggml_split_tensor_t *)model.layers[il].bq->extra;
                GGML_ASSERT(bq->n_device == wq->n_device);
            }
            if (model.layers[il].bo && model.layers[il].bo->extra) {
                bo = (ggml_split_tensor_t *)model.layers[il].bo->extra;
                GGML_ASSERT(bo->n_device == wq->n_device);
            }
            if (model.layers[il].bk && model.layers[il].bk->extra) {
                bk = (ggml_split_tensor_t *)model.layers[il].bk->extra;
                GGML_ASSERT(bk->n_device == wq->n_device);
            }
            if (model.layers[il].bv && model.layers[il].bv->extra) {
                bv = (ggml_split_tensor_t *)model.layers[il].bv->extra;
                GGML_ASSERT(bv->n_device == wq->n_device);
            }
            std::vector<ggml_tensor*> attn(wq->n_device, nullptr);
            int id_last = -1;
            bool output_bias_added = false;
            for (int id = 0; id < wq->n_device; ++id) {
                int il_cb = 1000*(id+1) + il;
                auto split_wq = wq->splits[id];
                auto split_wk = wk->splits[id];
                auto split_wv = wv->splits[id];
                auto split_wo = wo->splits[id];
                auto split_kl = kl->splits[id];
                auto split_vl = vl->splits[id];
                GGML_ASSERT((!split_wq && !split_wk && !split_wv && !split_wo && !split_kl && !split_vl) ||
                        (split_wq && split_wk && split_wv && split_wo && split_kl && split_vl));
                if (!split_wq) continue;
                auto cur = get_input_tensor_sm_graph(input, id);
                if (attn_norm) {
                    if (is_norm) {
                        cur = ggml_fused_norm(ctx0, cur, attn_norm->splits[id], lctx.model.hparams.f_norm_eps);
                    } else {
                        cur = llm_build_norm(ctx0, cur, lctx.model.hparams, attn_norm->splits[id], NULL, LLM_NORM_RMS, cb, il);
                    }
                }
                if (cur->type != GGML_TYPE_F32) {
                    cur = ggml_cast(ctx0, cur, GGML_TYPE_F32);
                }
                auto the_q_norm = model.layers[il].attn_q_norm ? model.layers[il].attn_q_norm->extra ?
                    ((ggml_split_tensor_t *)model.layers[il].attn_q_norm->extra)->splits[id] : model.layers[il].attn_q_norm : nullptr;
                auto the_k_norm = model.layers[il].attn_k_norm ? model.layers[il].attn_k_norm->extra ?
                    ((ggml_split_tensor_t *)model.layers[il].attn_k_norm->extra)->splits[id] : model.layers[il].attn_k_norm : nullptr;
                auto [Qcur, Kcur, Vcur] = llm_build_mul_mat_qkv(gf, cur, nullptr, nullptr, nullptr, nullptr,
                        split_wq, bq ? bq->splits[id] : nullptr,
                        split_wk, bk ? bk->splits[id] : nullptr,
                        split_wv, bv ? bv->splits[id] : nullptr,
                        the_q_norm, the_k_norm, f_attn_scale, il_cb, add_graph_split);
                auto rope_factors = rope_factors_in;
                if (!rope_factors && model.layers[il].rope_freqs && model.layers[il].rope_freqs->extra) {
                    auto extra = (ggml_split_tensor_t *)model.layers[il].rope_freqs->extra;
                    rope_factors = extra->splits[id];
                }
                if (do_rope) {
                    if (is_multi) {
                        int sections[4];
                        std::copy(hparams.rope_sections.begin(), hparams.rope_sections.begin() + GGML_MROPE_SECTIONS, sections);
                        Qcur = ggml_rope_multi(ctx0, Qcur, inp_pos, rope_factors,
                                n_rot, sections, rope_type, n_ctx_orig, freq_base_l, freq_scale_l,
                                ext_factor, attn_factor, beta_fast, beta_slow);
                        Kcur = ggml_rope_multi(ctx0, Kcur, inp_pos, rope_factors,
                                n_rot, sections, rope_type, n_ctx_orig, freq_base_l, freq_scale_l,
                                ext_factor, attn_factor, beta_fast, beta_slow);
                    } else {
                        Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, rope_factors, n_rot, rope_type, n_ctx_orig, freq_base_l, freq_scale_l,
                                ext_factor, attn_factor, beta_fast, beta_slow);
                        Kcur = ggml_rope_ext(ctx0, Kcur, inp_pos, rope_factors, n_rot, rope_type, n_ctx_orig, freq_base_l, freq_scale_l,
                                ext_factor, attn_factor, beta_fast, beta_slow);
                    }
                }
                cb(Qcur, "Qcur", il_cb);
                cb(Kcur, "Kcur", il_cb);
                if (inp_attn_scale) {
                    Qcur = ggml_mul(ctx0, Qcur, inp_attn_scale);
                    cb(Qcur, "Qcur_temp_scaled", il_cb);
                }
                if (cparams.k_cache_hadamard) {
                    Qcur = ggml_hadamard(ctx0, Qcur, hparams.n_embd_head_k);
                    Kcur = ggml_hadamard(ctx0, Kcur, hparams.n_embd_head_k);
                    cb(Qcur, "Qcur_hadamard", il_cb);
                    cb(Kcur, "Kcur_hadamard", il_cb);
                }
                ggml_build_forward_expand(gf, Qcur);
                ggml_build_forward_expand(gf, Kcur);
                ggml_build_forward_expand(gf, Vcur);

                const int64_t n_embd_head_k = hparams.n_embd_head_k;
                const int64_t n_head_kv     = split_wk->ne[1] / n_embd_head_k;

                GGML_ASSERT(kv_self.size == cparams.n_ctx);

                auto idx = 2*wq->n_device*il + 2*id;
                GGML_ASSERT(idx+1 < (int)lctx.cache_copies.size());
                auto k_row_size = ggml_row_size(split_kl->type, n_embd_head_k);
                ggml_tensor * k_cache_view = ggml_view_2d(ctx0, split_kl, n_embd_head_k, n_tokens*n_head_kv,
                        k_row_size, k_row_size*n_head_kv*kv_head);

                lctx.cache_copies[idx+0].cpy  = ggml_cpy(ctx0, Kcur, k_cache_view);
                lctx.cache_copies[idx+0].step = k_row_size*n_head_kv;

                // note: storing RoPE-ed version of K in the KV cache
                ggml_build_forward_expand(gf, lctx.cache_copies[idx+0].cpy);

                struct ggml_tensor * v_cache_view = nullptr;

                if (cparams.flash_attn) {
                    v_cache_view = ggml_view_1d(ctx0, split_vl, n_tokens*split_wv->ne[1],
                            kv_head*ggml_row_size(split_vl->type, split_wv->ne[1]));
                    lctx.cache_copies[idx+1].step = ggml_row_size(split_vl->type, split_wv->ne[1]);
                } else {
                    // note: the V cache is transposed when not using flash attention
                    v_cache_view = ggml_view_2d(ctx0, split_vl, n_tokens, split_wv->ne[1],
                            (  n_ctx)*ggml_element_size(split_vl),
                            (kv_head)*ggml_element_size(split_vl));
                    lctx.cache_copies[idx+1].step = ggml_element_size(split_vl);

                    Vcur = ggml_transpose(ctx0, Vcur);
                }
                cb(v_cache_view, "v_cache_view", il_cb);

                lctx.cache_copies[idx+1].cpy  = ggml_cpy(ctx0, Vcur, v_cache_view);
                ggml_build_forward_expand(gf, lctx.cache_copies[idx+1].cpy);

                auto q = ggml_permute(ctx0, Qcur, 0, 2, 1, 3);
                cb(q, "q", il_cb);

                auto k = ggml_view_3d(ctx0, split_kl, n_embd_head_k, n_kv, n_head_kv,
                             ggml_row_size(split_kl->type, n_embd_head_k)*n_head_kv, //n_embd_k_gqa),
                             ggml_row_size(split_kl->type, n_embd_head_k), 0);
                cb(k, "k", il_cb);

                auto v = ggml_view_3d(ctx0, split_vl, n_embd_head_v, n_kv, n_head_kv,
                             ggml_row_size(split_vl->type, split_wv->ne[1]),
                             ggml_row_size(split_vl->type, n_embd_head_v), 0);
                cb(v, "v", il_cb);

#ifdef GGML_USE_VULKAN
                constexpr bool use_f32_precision = true;
#else
                constexpr bool use_f32_precision = false;
#endif
                cur = ggml_flash_attn_ext(ctx0, q, k, v, KQ_mask, KQ_scale, hparams.f_max_alibi_bias,
                                  hparams.attn_soft_cap ? hparams.f_attn_logit_softcapping : 0.0f);
                cb(cur, "flash_attn", il_cb);
                if (model.layers[il].attn_sinks && model.layers[il].attn_sinks->extra) {
                    auto split = (ggml_split_tensor_t *)model.layers[il].attn_sinks->extra;
                    GGML_ASSERT(split->n_device == wq->n_device);
                    GGML_ASSERT(split->splits[id]);
                    ggml_flash_attn_ext_add_sinks(cur, split->splits[id]);
                } else {
                    ggml_flash_attn_ext_add_sinks(cur, sinks);
                }
                if (n_swa > 0) {
                    ((int32_t *)cur->op_params)[4] = n_swa;
                }
                // Some models produced NaNs/gibberish when FA is computed with f16 precision on CUDA
                if (use_f32_precision || model.arch == LLM_ARCH_PHI2 || model.arch == LLM_ARCH_PHI3 || model.arch == LLM_ARCH_GPTNEOX ||
                        (model.arch == LLM_ARCH_DEEPSEEK2 && q->ne[1] <= 8) || model.arch == LLM_ARCH_COHERE2 || model.arch == LLM_ARCH_GLM4 ||
                        model.arch == LLM_ARCH_GLM4_MOE) {
                    ggml_flash_attn_ext_set_prec(cur, GGML_PREC_F32);
                }

                cur = ggml_reshape_2d(ctx0, cur, split_wo->ne[0], n_tokens);
                cb(cur, "flash_attn_reshaped", il_cb);

                cur = llm_build_lora_mm(lctx, ctx0, split_wo, cur);
                if (lctx.model.arch == LLM_ARCH_GLM4 || lctx.model.arch == LLM_ARCH_GLM4_MOE) {
                    // GLM4 and GLM4_MOE seem to have numerical issues with half-precision accumulators
                    ggml_mul_mat_set_prec(cur, GGML_PREC_F32);
                }
                cb(cur, "kqv_wo", il_cb);
                if (!output_bias_added && bo) {
                    cur = ggml_add(ctx0, cur, bo->splits[id]);
                    cb(cur, "kqv_wo_biased", il_cb);
                    output_bias_added = true;
                }
                if (cur->ne[1] > 32 && lctx.cparams.split_mode_f16) {
                    cur = ggml_cast(ctx0, cur, GGML_TYPE_F16);
                }
                ggml_build_forward_expand(gf, cur);
                attn[id] = cur;
                id_last = id;
            }
            GGML_ASSERT(id_last >= 0);
            if (add_input) {
                attn[id_last] = ggml_add(ctx0, attn[id_last], input);
                cb(attn[id_last], "attn_out_with_input", il);
            }
            auto cur = ggml_reduce(ctx0, attn.data(), wq->n_device, GGML_OP_ADD);
            ggml_build_forward_expand(gf, cur);
            cb(cur, "attn_combined", il);
            return cur;
        }
    }

    auto cur = input;
    if (the_attn_norm) {
        cur = llm_build_norm(ctx0, cur, hparams, the_attn_norm, NULL, is_norm ? LLM_NORM : LLM_NORM_RMS, cb, il);
        cb(cur, "attn_norm", il);
    }

    auto [Qcur, Kcur, Vcur] = llm_build_mul_mat_qkv(gf, cur,
            model.layers[il].wqkv, model.layers[il].bqkv,
            model.layers[il].wqk,  model.layers[il].bqk,
            model.layers[il].wq,   model.layers[il].bq, model.layers[il].wk, model.layers[il].bk, model.layers[il].wv, model.layers[il].bv,
            model.layers[il].attn_q_norm, model.layers[il].attn_k_norm, f_attn_scale, il);

    if (do_rope) {
        if (is_multi) {
            int sections[4];
            std::copy(hparams.rope_sections.begin(), hparams.rope_sections.begin() + GGML_MROPE_SECTIONS, sections);
            Qcur = ggml_rope_multi(ctx0, Qcur, inp_pos, rope_factors_in,
                    n_rot, sections, rope_type, n_ctx_orig, freq_base_l, freq_scale_l,
                    ext_factor, attn_factor, beta_fast, beta_slow);
            Kcur = ggml_rope_multi(ctx0, Kcur, inp_pos, rope_factors_in,
                    n_rot, sections, rope_type, n_ctx_orig, freq_base_l, freq_scale_l,
                    ext_factor, attn_factor, beta_fast, beta_slow);
        } else {
            Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, rope_factors_in, n_rot, rope_type, n_ctx_orig, freq_base_l, freq_scale_l,
                    ext_factor, attn_factor, beta_fast, beta_slow);
            Kcur = ggml_rope_ext( ctx0, Kcur, inp_pos, rope_factors_in, n_rot, rope_type, n_ctx_orig, freq_base_l, freq_scale_l,
                    ext_factor, attn_factor, beta_fast, beta_slow);
        }
    }
    cb(Qcur, "Qcur", il);
    cb(Kcur, "Kcur", il);

    if (inp_attn_scale) {
        Qcur = ggml_mul(ctx0, Qcur, inp_attn_scale);
        cb(Qcur, "Qcur_temp_scaled", il);
    }

    cur = llm_build_kv(ctx0, lctx, kv_self, gf,
            model.layers[il].wo, model.layers[il].bo,
            Kcur, Vcur, Qcur, KQ_mask, n_tokens, kv_head, n_kv, KQ_scale, cb, il, sinks, n_swa);

    if (add_input) {
        cb(cur, "attn_out", il);
        cur = ggml_add(ctx0, cur, input);
    }

    return cur;
}

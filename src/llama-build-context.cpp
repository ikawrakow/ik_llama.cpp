#include "llama-build-context.h"
#include "llama-hparams.h"
#include "llama-cparams.h"
#include "llama-model.h"
#include "llama-context.h"
#include "llama-delta-net.h"

#include "ggml.h"

#include <unordered_set>
#include <algorithm>

uint32_t llm_build_context::llama_kv_qnext_state_slots(const llama_kv_cache & kv_self) {
    uint32_t n_slots = 0;

    for (const ggml_tensor * t : kv_self.s_l) {
        if (t == nullptr) {
            continue;
        }

        const uint32_t layer_slots = (uint32_t) t->ne[1];
        if (n_slots == 0) {
            n_slots = layer_slots;
        } else {
            GGML_ASSERT(n_slots == layer_slots);
        }
    }

    return n_slots;
}

llm_build_context::llm_build_context(
        llama_context  & lctx,
    const llama_batch  & batch,
    const llm_build_cb & cb,
    bool   worst_case,
    bool   warmup,
    int    n_outputs_) :
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
        n_embd_head_k    (hparams.n_embd_head_k(0)),
        n_embd_k_gqa     (hparams.n_embd_k_gqa()),
        n_embd_head_v    (hparams.n_embd_head_v(0)),
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
        n_outputs        (worst_case ? n_outputs_ > 0 ? n_outputs_ : n_tokens : lctx.n_outputs),
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
    lctx.inp_s_seq_qnext = nullptr;
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
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, model.max_nodes(n_tokens), false);

    GGML_ASSERT(kv_self.size == n_ctx);

    const auto & rope_type_shift = hparams.rope_type == LLAMA_ROPE_TYPE_MROPE || hparams.rope_type == LLAMA_ROPE_TYPE_IMROPE
        // @ngxson : this is a workaround
        // for M-RoPE, we want to rotate the whole vector when doing KV shift
        // a normal RoPE should work, we just need to use the correct ordering
        // ref: https://github.com/ggml-org/llama.cpp/pull/13870
        ? LLAMA_ROPE_TYPE_NEOX
        : hparams.rope_type;

    const float yarn_attn_factor_shift = model.arch == LLM_ARCH_DEEPSEEK2 || model.arch == LLM_ARCH_MISTRAL4
        ? 1.0f / (1.0f + 0.1f * logf(1.0f / freq_scale))
        : cparams.yarn_attn_factor;

    lctx.inp_K_shift = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_ctx);
    cb(lctx.inp_K_shift, "K_shift", -1);
    ggml_set_input(lctx.inp_K_shift);

    for (int il = 0; il < n_layer; ++il) {
        if (llm_arch_is_hybrid(model.arch) && hparams.is_recurrent(il)) {
            continue;
        }
        if (kv_self.k_l[il] == nullptr) {
            continue;
        }
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
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, model.max_nodes(n_tokens), false);

    const uint32_t qnext_state_slots = llama_kv_qnext_state_slots(kv_self);
    const bool has_qnext_state = qnext_state_slots > 0;
    GGML_ASSERT(kv_self.recurrent || has_qnext_state);

    struct ggml_tensor * state_copy = build_inp_s_copy();

    for (int il = 0; il < n_layer; ++il) {
        if (kv_self.recurrent) {
            struct ggml_tensor * conv_states = ggml_reshape_2d(ctx0, kv_self.k_l[il], hparams.n_embd_k_s(), kv_self.size);
            struct ggml_tensor * ssm_states  = ggml_reshape_2d(ctx0, kv_self.v_l[il], hparams.n_embd_v_s(), kv_self.size);

            conv_states = ggml_get_rows(ctx0, conv_states, state_copy);
            ssm_states  = ggml_get_rows(ctx0,  ssm_states, state_copy);

            // TODO: name the intermediate tensors with cb()

            ggml_build_forward_expand(gf, ggml_cpy(ctx0, conv_states, kv_self.k_l[il]));
            ggml_build_forward_expand(gf, ggml_cpy(ctx0,  ssm_states, kv_self.v_l[il]));
        }

        if (kv_self.s_l.size() > (size_t) il && kv_self.s_l[il] != nullptr) {
            struct ggml_tensor * qnext_states_all = ggml_reshape_2d(ctx0, kv_self.s_l[il], hparams.n_embd_v_s(), kv_self.s_l[il]->ne[1]);
            GGML_ASSERT((uint32_t) qnext_states_all->ne[1] == qnext_state_slots);
            struct ggml_tensor * qnext_state_copy = ggml_view_1d(ctx0, state_copy, qnext_state_slots, 0);
            struct ggml_tensor * qnext_states = ggml_get_rows(ctx0, qnext_states_all, qnext_state_copy);

            ggml_build_forward_expand(gf, ggml_cpy(ctx0, qnext_states, kv_self.s_l[il]));
        }
    }

    return gf;
}

ggml_cgraph * llm_build_context::build_defrag(const std::vector<uint32_t> & ids) {
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, model.max_nodes(n_tokens), false);

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
            if (llm_arch_is_hybrid(model.arch) && hparams.is_recurrent(il)) {
                continue;
            }
            if (kv_self.k_l[il] == nullptr) {
                continue;
            }
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

            if (kv_self.v_l.size() > il && kv_self.v_l[il] != nullptr) {
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

struct ggml_tensor * llm_build_context::build_inp_embd_mtp(struct ggml_tensor * mtp_tok_embd) {
    struct ggml_tensor * cur = nullptr;

    if (batch.token) {
        lctx.inp_tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, batch.n_tokens);

        cb(lctx.inp_tokens, "inp_tokens", -1);
        ggml_set_input(lctx.inp_tokens);

        cur = ggml_get_rows(ctx0, mtp_tok_embd, lctx.inp_tokens);
    } else {
        return nullptr;
    }

    cb(cur, "inp_embd", -1);

    return cur;
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

        if (strcmp(inp->name, "result_norm") == 0 || 
            strcmp(inp->name, "result_embd") == 0 || 
            strcmp(inp->name, "output_normed") == 0) { 
            break;
        }
        inp = nullptr;
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
    const int64_t n_embd_head_k = hparams.n_embd_head_k(il);

    GGML_ASSERT(kv.size == n_ctx);

    //struct ggml_tensor * k_cache_view = ggml_view_1d(ctx, kv.k_l[il], n_tokens*n_embd_k_gqa,
    //        (ggml_row_size(kv.k_l[il]->type, n_embd_k_gqa))*kv_head);
    //cb(k_cache_view, "k_cache_view", il);

    if (k_cur) {
        GGML_ASSERT(2*il+1 < (int)lctx.cache_copies.size());
        auto k_row_size = ggml_row_size(kv.k_l[il]->type, n_embd_head_k);
        ggml_tensor * k_cache_view = ggml_view_2d(ctx, kv.k_l[il], n_embd_head_k, n_tokens*n_head_kv,
                k_row_size, k_row_size*n_head_kv*kv_head);

        lctx.cache_copies[2*il+0].cpy  = ggml_cpy(ctx, k_cur, k_cache_view);
        lctx.cache_copies[2*il+0].step = k_row_size*n_head_kv;

        // note: storing RoPE-ed version of K in the KV cache
        ggml_build_forward_expand(graph, lctx.cache_copies[2*il+0].cpy);
    }

    if (v_cur) {
        ggml_tensor * v_cache_view = nullptr;
        if (!kv.v_trans) {
            v_cache_view = ggml_view_1d(ctx, kv.v_l[il], n_tokens*n_embd_v_gqa,
                    (kv_head)*ggml_row_size(kv.v_l[il]->type, n_embd_v_gqa));
            lctx.cache_copies[2*il+1].step = ggml_row_size(kv.v_l[il]->type, n_embd_v_gqa);
        } else {
            // note: the V cache is transposed for legacy non-FA layouts
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

ggml_tensor * llm_build_context::get_input_tensor_sm_graph(ggml_context * ctx, ggml_tensor * input, int id) {
    auto cur = input;
    if (input->op == GGML_OP_REDUCE) {
        auto view_src = input->view_src;
        GGML_ASSERT(view_src);
        cur = input->src[id];
        if (!cur) {
            GGML_ASSERT((input->op_params[4] & (1u << id)) == 0);
            cur = ggml_dup_tensor(ctx, input);
            input->src[id] = cur;
            input->op_params[4] |= (1u << id);
        }
        else if (cur == view_src) {
            cur = input;
        }
    }
    return cur;
}

ggml_tensor * llm_build_context::do_split_norm(ggml_context * ctx, ggml_tensor * cur, ggml_tensor * the_norm, const llama_hparams & hparams,
        const llm_build_cb & cb, int id, int il_cb, bool is_norm) {
    if (the_norm && the_norm->extra) {
        auto norm = (ggml_split_tensor_t *)the_norm->extra;
        GGML_ASSERT(norm->splits[id]);
        if (is_norm) {
            cur = ggml_fused_norm(ctx, cur, norm->splits[id], hparams.f_norm_eps);
        } else {
            cur = llm_build_context::llm_build_norm(ctx, cur, hparams, norm->splits[id], NULL, LLM_NORM_RMS, cb, il_cb);
        }
        cb(cur, "inp_normed", il_cb);
    }
    if (cur->type != GGML_TYPE_F32) {
        cur = ggml_cast(ctx, cur, GGML_TYPE_F32);
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
         bool is_norm, ggml_tensor * add_extra, ggml_tensor * post_norm) {

    if (!up_b && !up_s && !gate_b && !gate_s && !down_b && !down_s &&
        up->extra && gate->extra && down->extra && type_gate == LLM_FFN_PAR &&
        (type_op == LLM_FFN_SILU || type_op == LLM_FFN_RELU || (type_op == LLM_FFN_GELU && !act_scales))) {
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
            auto cur = get_input_tensor_sm_graph(ctx, input, id);
            cur = do_split_norm(ctx, cur, ffn_norm, lctx.model.hparams, cb, id, il_cb, is_norm);
            if (input->op != GGML_OP_REDUCE) {
                cur->op_params[GGML_MAX_OP_PARAMS / sizeof(int32_t) - 1] = 0xff;
            }
            cur = ggml_fused_up_gate(ctx, split_u, split_g, cur, unary_op);
            cb(cur, "ffn_up_gate", il_cb);
            if (lctx.model.arch == LLM_ARCH_STEP35) {
                *(float *)(cur->op_params + 1) = lctx.model.hparams.swiglu_limits[il];
            }
            cur = llm_build_lora_mm(lctx, ctx, split_d, cur);
            cb(cur, "ffn_down", il_cb);
            if (lctx.model.arch == LLM_ARCH_GLM4 || lctx.model.arch == LLM_ARCH_GLM4_MOE) {
                // GLM4 and GLM4_MOE seem to have numerical issues with half-precision accumulators
                ggml_mul_mat_set_prec(cur, GGML_PREC_F32);
            }
            if (cur->ne[1] > 32 && lctx.cparams.reduce_type != GGML_TYPE_F32) {
                cur = ggml_cast(ctx, cur, lctx.cparams.reduce_type);
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
    //if (input->op == GGML_OP_REDUCE) {
    //    if (input->src[lctx.model.main_gpu]) cur = input->src[lctx.model.main_gpu];
    //}
    if (ffn_norm) {
        auto the_ffn_norm = ffn_norm->extra ? ((ggml_split_tensor_t *)ffn_norm->extra)->splits[lctx.model.main_gpu] : ffn_norm;
        cur = llm_build_norm(ctx, cur, lctx.model.hparams, the_ffn_norm, NULL, is_norm ? LLM_NORM : LLM_NORM_RMS, cb, il);
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
        if (lctx.model.arch == LLM_ARCH_STEP35) {
            *(float *)(cur->op_params + 1) = lctx.model.hparams.swiglu_limits_shared[il];
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
        if (post_norm) {
            cur = llm_build_norm(ctx, cur, lctx.model.hparams, post_norm, NULL, LLM_NORM_RMS, cb, il);
            cb(cur, "ffn_post_normed", il);
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
        if (lctx.model.arch == LLM_ARCH_STEP35) {
            *((float *)(cur->op_params + 1)) = lctx.model.hparams.swiglu_limits_shared[il];
        }
    }
    else {

    switch (type_op) {
        case LLM_FFN_SILU:
            {
                if (lctx.model.arch == LLM_ARCH_STEP35) {
                    cur = ggml_fused_mul_unary(ctx, cur, up, GGML_UNARY_OP_SILU);
                    *(float *)(cur->op_params + 1) = lctx.model.hparams.swiglu_limits_shared[il];
                    type_gate = LLM_FFN_SEQ;
                    break;
                }
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

    if (post_norm) {
        cur = llm_build_norm(ctx, cur, lctx.model.hparams, post_norm, NULL, LLM_NORM_RMS, cb, il);
        cb(cur, "ffn_post_normed", il);
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
         ggml_tensor * up_gate_exps, ggml_tensor * up_gate_exps_b,
         ggml_tensor * input_logits, ggml_tensor * down_exps_s) {

    GGML_ASSERT(gate_inp || input_logits);

    auto input = cur;

    int64_t n_embd = cur->ne[0];
    int64_t n_tokens = cur->ne[1];
    bool weight_before_ffn = lctx.model.arch == LLM_ARCH_LLAMA4; // for llama4, we apply the sigmoid-ed weights before the FFN

    ggml_tensor * logits = gate_inp ? llm_build_lora_mm(lctx, ctx, gate_inp, cur) : input_logits; // [n_expert, n_tokens]
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

        if (lctx.model.arch == LLM_ARCH_BAILINGMOE2 || lctx.model.arch == LLM_ARCH_STEP35) {
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
    bool can_use_fmoe = (type_op == LLM_FFN_SILU || type_op == LLM_FFN_GELU || type_op == LLM_FFN_SWIGLU_OAI_MOE);

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
        if (lctx.model.arch == LLM_ARCH_STEP35) {
            *((float *)(par->op_params + 1)) = lctx.model.hparams.swiglu_limits[il];
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
        if (lctx.model.arch == LLM_ARCH_STEP35) {
            *(float *)(par->op_params + 1) = lctx.model.hparams.swiglu_limits[il];
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
            if (lctx.model.arch == LLM_ARCH_STEP35) {
                *((float *)(par->op_params + 1)) = lctx.model.hparams.swiglu_limits[il];
            }
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

    if (down_exps_s && !lctx.cparams.fused_mmad) {
        GGML_ASSERT(!weight_before_ffn);
        auto s = ggml_reshape_3d(ctx, down_exps_s, 1, n_expert, 1);
        s = ggml_repeat_4d(ctx, s, 1, n_expert, n_tokens, 1);
        s = ggml_get_rows(ctx, s, selected_experts);
        auto w_reshaped = ggml_reshape_2d(ctx, weights, n_expert_used, n_tokens);
        auto s_reshaped = ggml_reshape_2d(ctx, s, n_expert_used, n_tokens);
        w_reshaped = ggml_mul(ctx, w_reshaped, s_reshaped);
        weights = ggml_reshape_3d(ctx, w_reshaped, 1, n_expert_used, n_tokens);
    }

    if (!weight_before_ffn) {
        if (lctx.cparams.fused_mmad) {
            experts = ggml_mul_multi_add(ctx, experts, weights);
            cb(experts, "ffn_moe_weighted", il);
            if (down_exps_s) {
                experts->src[2] = down_exps_s;
                experts->src[3] = selected_experts;
            }
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
         ggml_tensor * up_gate_exps, ggml_tensor * up_gate_exps_b,
         ggml_tensor * shexp_gate) {

    auto split_up_exps    = up_exps ? (ggml_split_tensor_t *)up_exps->extra : nullptr;
    auto split_gate_exps  = gate_exps ? (ggml_split_tensor_t *)gate_exps->extra : nullptr;
    auto split_down_exps  = (ggml_split_tensor_t *)down_exps->extra;
    auto split_up_shexp   = up_shexp   ? (ggml_split_tensor_t *)up_shexp->extra   : nullptr;
    auto split_gate_shexp = gate_shexp ? (ggml_split_tensor_t *)gate_shexp->extra : nullptr;
    auto split_down_shexp = down_shexp ? (ggml_split_tensor_t *)down_shexp->extra : nullptr;
    auto split_up_b_shexp   = up_b_shexp   ? (ggml_split_tensor_t *)up_b_shexp   : nullptr;
    auto split_gate_b_shexp = gate_b_shexp ? (ggml_split_tensor_t *)gate_b_shexp : nullptr;
    auto split_down_b_shexp = down_b_shexp ? (ggml_split_tensor_t *)down_b_shexp : nullptr;
    auto split_up_gate_exps = up_gate_exps ? (ggml_split_tensor_t *)up_gate_exps->extra : nullptr;
    if (!split_up_exps && !split_gate_exps && !split_up_gate_exps && !split_down_exps) {
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
                std::vector<ggml_tensor *> results(split_up_shexp->n_device, nullptr);
                GGML_ASSERT(!split_up_b_shexp   || split_up_b_shexp->n_device   == split_up_shexp->n_device);
                GGML_ASSERT(!split_gate_b_shexp || split_gate_b_shexp->n_device == split_up_shexp->n_device);
                GGML_ASSERT(!split_down_b_shexp || split_down_b_shexp->n_device == split_up_shexp->n_device);
                bool down_bias_added = false;
                int id_add_routed = -1;
                if (split_up_shexp->splits[lctx.model.main_gpu]) {
                    id_add_routed = lctx.model.main_gpu;
                } else {
                    for (int id = 0; id < split_up_shexp->n_device; ++id) {
                        if (split_up_shexp->splits[id]) {
                            id_add_routed = id;
                            break;
                        }
                    }
                }
                GGML_ASSERT(id_add_routed >= 0);
                for (int id = 0; id < split_up_shexp->n_device; ++id) {
                    int il_cb = 1000*id + il;
                    GGML_ASSERT((split_up_shexp->splits[id] && split_gate_shexp->splits[id] && split_down_shexp->splits[id]) ||
                                (!split_up_shexp->splits[id] && !split_gate_shexp->splits[id] && !split_down_shexp->splits[id]));
                    if (!split_up_shexp->splits[id]) continue;
                    auto the_ffn_norm = ffn_norm ? ffn_norm->extra ? ((ggml_split_tensor_t *)ffn_norm->extra)->splits[id] : ffn_norm : nullptr;
                    auto this_input = input;
                    if (the_ffn_norm) {
                        this_input = llm_build_norm(ctx, input, lctx.model.hparams, the_ffn_norm, nullptr, LLM_NORM_RMS, cb, il);
                    }
                    auto shared_out = llm_build_ffn(ctx, lctx, nullptr, this_input,
                            split_up_shexp->splits[id],   split_up_b_shexp   ? split_up_b_shexp->splits[id]   : nullptr, nullptr,
                            split_gate_shexp->splits[id], split_gate_b_shexp ? split_gate_b_shexp->splits[id] : nullptr, nullptr,
                            split_down_shexp->splits[id], !down_bias_added && split_down_b_shexp ? split_down_b_shexp->splits[id] : nullptr, nullptr,
                            nullptr, type_op_shexp, LLM_FFN_PAR, cb, il, graph, false, false, nullptr);
                    cb(shared_out, "ffn_shexp_out", il_cb);
                    if (shexp_gate) {
                        auto split_shexp_gate = (ggml_split_tensor_t *)shexp_gate->extra;
                        GGML_ASSERT(split_shexp_gate && split_shexp_gate->splits[id]);
                        auto gate = llm_build_lora_mm(lctx, ctx, split_shexp_gate->splits[id], this_input);
                        if (gate->ne[1] == 1) {
                            shared_out = ggml_fused_mul_unary(ctx, gate, shared_out, GGML_UNARY_OP_SIGMOID);
                        } else {
                            gate = ggml_sigmoid(ctx, gate);
                            shared_out = ggml_mul(ctx, shared_out, gate);
                        }
                        cb(shared_out, "ffn_shexp_gated", il_cb);
                    }
                    if (id == id_add_routed) {
                        shared_out = ggml_add(ctx, shared_out, routed_out);
                        cb(shared_out, "ffn_shared_routed_added", il);
                    }
                    if (shared_out->ne[1] > 32 && lctx.cparams.reduce_type != GGML_TYPE_F32) {
                        shared_out = ggml_cast(ctx, shared_out, lctx.cparams.reduce_type);
                    }
                    ggml_build_forward_expand(graph, shared_out);
                    down_bias_added = true;
                    results[id] = shared_out;
                }
                GGML_ASSERT(!results.empty());
                cur = ggml_reduce(ctx, results.data(), split_up_shexp->n_device, GGML_OP_ADD);
                cb(cur, "ffn_out", il);
            } else {
                auto shared_out = llm_build_ffn(ctx, lctx, nullptr, cur,
                        up_shexp,   up_b_shexp,   nullptr,
                        gate_shexp, gate_b_shexp, nullptr,
                        down_shexp, down_b_shexp, nullptr,
                        nullptr, type_op_shexp, LLM_FFN_PAR, cb, il);
                cb(shared_out, "ffn_shexp_out", il);
                if (shexp_gate) {
                    auto shared_gate = llm_build_lora_mm(lctx, ctx, shexp_gate, cur);
                    cb(shared_gate, "shared_expert_gate", il);
                    if (shared_gate->ne[1] == 1) {
                        shared_out = ggml_fused_mul_unary(ctx, shared_gate, shared_out, GGML_UNARY_OP_SIGMOID);
                    } else {
                        shared_gate = ggml_sigmoid(ctx, shared_gate);
                        cb(shared_gate, "shared_expert_gate_sigmoid", il);
                        shared_out = ggml_mul(ctx, shared_out, shared_gate);
                    }
                    cb(shared_out, "ffn_shexp_gated", il);
                }
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
    GGML_ASSERT(((split_up_exps && split_gate_exps) || split_up_gate_exps) && split_down_exps);
    int n_device = split_down_exps->n_device;
    if (split_up_gate_exps) {
        GGML_ASSERT(split_up_gate_exps->n_device == n_device);
    } else {
        GGML_ASSERT(split_up_exps->n_device == n_device && split_gate_exps->n_device == n_device);
    }
    std::vector<ggml_tensor *> results(n_device, nullptr);
    GGML_ASSERT((!split_up_shexp && !split_gate_shexp && !split_down_shexp) ||
                ( split_up_shexp &&  split_gate_shexp &&  split_down_shexp));
    auto split_gate_inp = (ggml_split_tensor_t *)gate_inp->extra;
    GGML_ASSERT(split_gate_inp && split_gate_inp->n_device == n_device);
    auto split_exp_probs_b = exp_probs_b ? (ggml_split_tensor_t *)exp_probs_b->extra : nullptr;
    GGML_ASSERT(!split_exp_probs_b || split_exp_probs_b->n_device == n_device);

    auto split_gate_inp_b  = gate_inp_b  ? (ggml_split_tensor_t *)gate_inp_b->extra  : nullptr;
    auto split_exps_down_b = down_exps_b ? (ggml_split_tensor_t *)down_exps_b->extra : nullptr;
    auto split_exps_gate_b = gate_exps_b ? (ggml_split_tensor_t *)gate_exps_b->extra : nullptr;
    auto split_exps_up_b   = up_exps_b   ? (ggml_split_tensor_t *)up_exps_b->extra   : nullptr;
    auto split_exps_up_gate_b = up_gate_exps_b ? (ggml_split_tensor_t *)up_gate_exps_b->extra : nullptr;
    int last_id = -1;
    bool down_bias_added = false;
    for (int id = 0; id < n_device; ++id) {
        bool has_up_gate = split_up_gate_exps ? split_up_gate_exps->splits[id] != nullptr : split_up_exps->splits[id] != nullptr && split_gate_exps->splits[id]!= nullptr ;
        GGML_ASSERT((has_up_gate && split_down_exps->splits[id]) ||
                    (!has_up_gate && !split_down_exps->splits[id]));
        if (!has_up_gate) continue;
        int il_cb = 1000*(id + 1) + il;
        auto cur = get_input_tensor_sm_graph(ctx, input, id);
        cur = do_split_norm(ctx, cur, ffn_norm, lctx.model.hparams, cb, id, il_cb, false);
        if (cur->op != GGML_OP_REDUCE) {
            cur->op_params[GGML_MAX_OP_PARAMS / sizeof(int32_t) - 1] = 0xff;
        }
        GGML_ASSERT(!split_gate_inp_b  || split_gate_inp_b->splits[id]);
        GGML_ASSERT(!split_exps_down_b || split_exps_down_b->splits[id]);
        GGML_ASSERT(!split_exps_gate_b || split_exps_gate_b->splits[id]);
        GGML_ASSERT(!split_exps_up_b   || split_exps_up_b->splits[id]);
        auto routed_out = llm_build_moe_ffn(ctx, lctx, cur,
                    split_gate_inp->splits[id],  split_gate_inp_b ? split_gate_inp_b->splits[id] : nullptr,
                    split_up_exps ? split_up_exps->splits[id] : nullptr, split_exps_up_b  ? split_exps_up_b->splits[id]  : nullptr,
                    split_gate_exps ? split_gate_exps->splits[id] : nullptr, split_exps_gate_b ? split_exps_gate_b->splits[id] : nullptr,
                    split_down_exps->splits[id], !down_bias_added && split_exps_down_b ? split_exps_down_b->splits[id] : nullptr,
                    split_exp_probs_b ? split_exp_probs_b->splits[id] : nullptr,
                    n_expert, n_expert_used,
                    type_op, norm_w, scale_w, w_scale,
                    gating_op, cb, il, graph, false,
                    split_up_gate_exps ? split_up_gate_exps->splits[id] : nullptr,
                    split_exps_up_gate_b ? split_exps_up_gate_b->splits[id] : nullptr);
        cb(routed_out, "routed_out", il_cb);

        if (split_up_shexp) {
            GGML_ASSERT(!split_up_b_shexp   || split_up_b_shexp->n_device   == n_device);
            GGML_ASSERT(!split_gate_b_shexp || split_gate_b_shexp->n_device == n_device);
            GGML_ASSERT(!split_down_b_shexp || split_down_b_shexp->n_device == n_device);
            auto shared_out = llm_build_ffn(ctx, lctx, nullptr, cur,
                    split_up_shexp->splits[id],   split_up_b_shexp   ? split_up_b_shexp->splits[id]   : nullptr, nullptr,
                    split_gate_shexp->splits[id], split_gate_b_shexp ? split_gate_b_shexp->splits[id] : nullptr, nullptr,
                    split_down_shexp->splits[id], !down_bias_added && split_down_b_shexp ? split_down_b_shexp->splits[id] : nullptr, nullptr,
                    nullptr, type_op_shexp, LLM_FFN_PAR, cb, il);
            cb(shared_out, "ffn_shexp_out", il_cb);
            if (shexp_gate) {
                auto split_shexp_gate = (ggml_split_tensor_t *)shexp_gate->extra;
                GGML_ASSERT(split_shexp_gate && split_shexp_gate->splits[id]);
                auto gate = llm_build_lora_mm(lctx, ctx, split_shexp_gate->splits[id], cur);
                if (gate->ne[1] == 1) {
                    shared_out = ggml_fused_mul_unary(ctx, gate, shared_out, GGML_UNARY_OP_SIGMOID);
                } else {
                    gate = ggml_sigmoid(ctx, gate);
                    shared_out = ggml_mul(ctx, shared_out, gate);
                }
                cb(shared_out, "ffn_shexp_gated", il_cb);
            }

            cur = ggml_add(ctx, routed_out, shared_out);
            cb(cur, "ffn_out", il_cb);
        } else {
            cur = routed_out;
        }
        if (cur->ne[1] > 32 && lctx.cparams.reduce_type != GGML_TYPE_F32) {
            cur = ggml_cast(ctx, cur, lctx.cparams.reduce_type);
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

    auto cur = ggml_reduce(ctx, results.data(), n_device, GGML_OP_ADD);
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
    const int64_t n_embd_head_k = hparams.n_embd_head_k(il);
    //const int64_t n_embd_k_gqa  = hparams.n_embd_k_gqa(il);
    const int64_t n_embd_head_v = hparams.n_embd_head_v(il);
    const int64_t n_embd_v_gqa  = hparams.n_embd_v_gqa(il);

    struct ggml_tensor * q = ggml_permute(ctx, q_cur, 0, 2, 1, 3);
    cb(q, "q", il);

    auto k_cache = lctx.model.hparams.has_kv(il) ? kv.k_l[il]
                 : lctx.model.hparams.swa_layers[il] ? kv.k_l[hparams.n_layer_kv_from_start-2] : kv.k_l[hparams.n_layer_kv_from_start-1];
    auto v_cache = lctx.model.hparams.has_kv(il) ? kv.v_l[il]
                 : lctx.model.hparams.swa_layers[il] ? kv.v_l[hparams.n_layer_kv_from_start-2] : kv.v_l[hparams.n_layer_kv_from_start-1];

    GGML_ASSERT(k_cache != nullptr && "k_cache is null in llm_build_kqv");
    GGML_ASSERT(v_cache != nullptr && "v_cache is null in llm_build_kqv");

    struct ggml_tensor * k =
        ggml_view_3d(ctx, k_cache,
                n_embd_head_k, n_kv, n_head_kv,
                ggml_row_size(k_cache->type, n_embd_head_k)*n_head_kv, //n_embd_k_gqa),
                ggml_row_size(k_cache->type, n_embd_head_k),
                0);
    cb(k, "k", il);

#ifdef GGML_USE_VULKAN
    constexpr bool use_f32_precision = true;
#else
    constexpr bool use_f32_precision = false;
#endif

    bool should_use_f32_precision = use_f32_precision
                                  || model.arch == LLM_ARCH_PHI2
                                  || model.arch == LLM_ARCH_PHI3
                                  || model.arch == LLM_ARCH_GPTNEOX
                                  || model.arch == LLM_ARCH_QWEN2
                                  || model.arch == LLM_ARCH_COHERE2
                                  || model.arch == LLM_ARCH_COMMAND_R
                                  || model.arch == LLM_ARCH_GLM4
                                  || model.arch == LLM_ARCH_GLM4_MOE
                                  || model.arch == LLM_ARCH_MIMO2;
                               // || (model.arch == LLM_ARCH_DEEPSEEK2 && q->ne[1] <= 8);

    struct ggml_tensor * cur;

    if (cparams.flash_attn) {
        GGML_UNUSED(model);
        GGML_UNUSED(n_ctx);

        // split cached v into n_head heads (not transposed)
        struct ggml_tensor * v =
            ggml_view_3d(ctx, v_cache,
                    n_embd_head_v, n_kv, n_head_kv,
                    ggml_row_size(v_cache->type, n_embd_v_gqa),
                    ggml_row_size(v_cache->type, n_embd_head_v),
                    0);
        cb(v, "v", il);

        cur = ggml_flash_attn_ext(ctx, q, k, v, kq_mask, kq_scale, hparams.f_max_alibi_bias,
                hparams.attn_soft_cap ? hparams.f_attn_logit_softcapping : 0.0f);
        cb(cur, "fa", il);
        ggml_flash_attn_ext_add_sinks(cur, sinks);
        if (n_swa > 0) {
            ((int32_t *)cur->op_params)[4] = n_swa;
        }

        // Some models produced NaNs/gibberish when FA is computed with f16 precision on CUDA
        // For DeepSeek-2, it is perfectly fine with fp16 for PP, but I get gibberish when uding fp16 for TG.
        // Not sure if it is really a matter of insufficient precision, or I have made a mistake in the fattn-vec-f16 kernel.
        if (should_use_f32_precision) {
            ggml_flash_attn_ext_set_prec(cur, GGML_PREC_F32);
        }
        //ggml_flash_attn_ext_set_prec(cur, GGML_PREC_F32);

        if (cparams.v_cache_hadamard) {
            cur = ggml_hadamard(ctx, cur, n_embd_head_v);
            cb(cur, "fa_h", il);
        }
        cur = ggml_reshape_2d(ctx, cur, n_embd_head_v*n_head, n_tokens);
    } else {

            // split cached v into n_head heads
        struct ggml_tensor * v;
        if (kv.v_trans) {
            v = ggml_view_3d(ctx, v_cache,
                    n_kv, n_embd_head_v, n_head_kv,
                    ggml_element_size(v_cache)*n_ctx,
                    ggml_element_size(v_cache)*n_ctx*n_embd_head_v,
                    0);
        } else {
            v = ggml_view_3d(ctx, v_cache,
                    n_embd_head_v, n_kv, n_head_kv,
                    ggml_row_size(v_cache->type, n_embd_v_gqa),
                    ggml_row_size(v_cache->type, n_embd_head_v),
                    0);
            v = ggml_cont(ctx, ggml_transpose(ctx, v));
        }
        cb(v, "v", il);

        auto kq_size = k->ne[1]*q->ne[1]*q->ne[2]*sizeof(float)/(1024*1024);
        if (cparams.attn_max_batch == 0 || cparams.attn_max_batch >= kq_size || k->ne[2] != q->ne[2] || v->ne[2] != q->ne[2] || sinks) {
            struct ggml_tensor * kq = ggml_mul_mat(ctx, k, q);
            cb(kq, "kq", il);

            //ggml_mul_mat_set_prec(kq, GGML_PREC_F32);

            if (should_use_f32_precision) {
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
                    model.arch == LLM_ARCH_COHERE2 || model.arch == LLM_ARCH_COMMAND_R || model.arch == LLM_ARCH_GLM4 || model.arch == LLM_ARCH_GLM4_MOE) {
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
        q_cur = ggml_hadamard(ctx, q_cur, hparams.n_embd_head_k(il));
        if (k_cur) {
            k_cur = ggml_hadamard(ctx, k_cur, hparams.n_embd_head_k(il));
            cb(k_cur, "Kcur_hadamard", il);
        }
        cb(q_cur, "Qcur_hadamard", il);
    }
    if (cparams.v_cache_hadamard && v_cur) {
        v_cur = ggml_hadamard(ctx, v_cur, hparams.n_embd_head_v(il));
    }

    // these nodes are added to the graph together so that they are not reordered
    // by doing so, the number of splits in the graph is reduced
    ggml_build_forward_expand(graph, q_cur);
    if (k_cur) {
        ggml_build_forward_expand(graph, k_cur);
    }
    if (v_cur) {
        ggml_build_forward_expand(graph, v_cur);
    }

    if (k_cur || v_cur) {
        llm_build_kv_store(lctx, ctx, hparams, cparams, kv, graph, k_cur, v_cur, n_tokens, kv_head, cb, il);
    }

    auto cur = llm_build_kqv(ctx, lctx, kv, graph, wo, wo_b, q_cur, kq_mask, n_tokens, n_kv, kq_scale, cb, il, sinks, n_swa);
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

std::tuple<ggml_tensor*, ggml_tensor*, ggml_tensor*, ggml_tensor*> llm_build_context::llm_build_mul_mat_qkv_gated(ggml_cgraph * gf, ggml_tensor * cur,
            ggml_tensor * wq, ggml_tensor * wk, ggml_tensor * wv, ggml_tensor * q_norm, ggml_tensor * k_norm, int il) const {
    auto Qaux = llm_build_lora_mm(lctx, ctx0, wq, cur);
    cb(Qaux, "Qaux", il);
    auto Kcur = llm_build_lora_mm(lctx, ctx0, wk, cur);
    cb(Kcur, "Kcur", il);
    auto Vcur = llm_build_lora_mm(lctx, ctx0, wv, cur);
    cb(Vcur, "Vcur", il);
    ggml_build_forward_expand(gf, Qaux);
    ggml_build_forward_expand(gf, Kcur);
    ggml_build_forward_expand(gf, Vcur);
    auto row_size = ggml_row_size(Qaux->type, n_embd_head_k);
    // TODO: check why CUDA performance suffers so much if we don't make these two tensors contiguous
    auto Qcur = ggml_cont(ctx0, ggml_view_3d(ctx0, Qaux, n_embd_head_k, Qaux->ne[0]/(2*n_embd_head_k), n_tokens, 2*row_size, Qaux->nb[1], 0));
    cb(Qcur, "Qcur_cont", il);
    auto gate = ggml_cont_2d(ctx0, ggml_view_3d(ctx0, Qaux, n_embd_head_k, Qaux->ne[0]/(2*n_embd_head_k), n_tokens, 2*row_size, Qaux->nb[1], row_size), Qaux->ne[0]/2, n_tokens);
    cb(gate, "gate_cont", il);
    Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head_k, Kcur->ne[0]/n_embd_head_k, n_tokens);
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
    //gate = ggml_sigmoid(ctx0, gate);
    //gate = ggml_reshape_2d(ctx0, gate, gate->ne[0]*gate->ne[1], gate->ne[2]);
    //cb(gate, "gate", il);
    return {Qcur, Kcur, Vcur, gate};
}

std::tuple<ggml_tensor*, ggml_tensor*, ggml_tensor*> llm_build_context::llm_build_mul_mat_qkv(ggml_cgraph * gf, ggml_tensor * cur,
            ggml_tensor * wqkv, ggml_tensor * bqkv,
            ggml_tensor * wqk, ggml_tensor * bqk,
            ggml_tensor * wq, ggml_tensor * bq,
            ggml_tensor * wk, ggml_tensor * bk,
            ggml_tensor * wv, ggml_tensor * bv,
            ggml_tensor * q_norm, ggml_tensor * k_norm, float attention_scale, int il, bool add_graph_split) const {
    int n_head    = hparams.n_head(il);
    int n_head_kv = hparams.n_head_kv(il);
    const int64_t n_embd_head_k = hparams.n_embd_head_k(il);
    const int64_t n_embd_gqa  = hparams.n_embd_v_gqa(il);
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
    // Command-R/R+ uses LayerNorm (not RMSNorm) for per-head Q/K normalisation
    const auto qk_norm_type = (model.arch == LLM_ARCH_COMMAND_R) ? LLM_NORM : LLM_NORM_RMS;
    if (q_norm) {
        Qcur = llm_build_norm(ctx0, Qcur, hparams, q_norm, NULL, qk_norm_type, cb, il);
        cb(Qcur, "Qcur_normed", il);
    }

    auto Kcur = ggml_reshape_3d(ctx0, K, n_embd_head_k, K->ne[0]/n_embd_head_k, n_tokens);
    if (k_norm) {
        Kcur = llm_build_norm(ctx0, Kcur, hparams, k_norm, NULL, qk_norm_type, cb, il);
        cb(Kcur, "Kcur_normed", il);
    }
    auto Vcur = V;
    return {Qcur, Kcur, Vcur};
}

ggml_tensor * llm_build_context::build_output(llama_context & lctx, ggml_context * ctx, ggml_tensor * cur,
        ggml_tensor * output, const llm_build_cb & cb) {
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

ggml_tensor * llm_build_context::build_output(llama_context & lctx, ggml_context * ctx, ggml_tensor * cur,
        ggml_tensor * output, ggml_tensor * output_norm, const llm_build_cb & cb) {
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
                cb(cur_normed, "result_norm", 1000*(id+1));
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
        int idx = lctx.model.default_layer_device[lctx.model.hparams.n_layer];
        int idx_out = ggml_backend_sched_get_backend_idx(lctx.sched, lctx.model.output->buffer);
        if (idx_out >= 0) idx = idx_out;
        const bool is_qwen_mtp = lctx.model.arch == LLM_ARCH_QWEN35 && lctx.cparams.mtp;
        if (cur->op == GGML_OP_REDUCE && cur->src[idx] && !is_qwen_mtp) {
            // avoid copy to main GPU
            cur->view_src = cur->src[idx];
        }
        if (output_norm) {
            cur = llm_build_context::llm_build_norm(ctx, cur, lctx.model.hparams, output_norm, NULL, LLM_NORM_RMS, cb, -1);
            cb(cur, "result_norm", -1);
        }
        cur = llm_build_context::llm_build_lora_mm(lctx, ctx, output, cur);
    }
    return cur;
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
                  bool   worst_case,
                  int    n_outputs) {
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
    struct llm_build_context llm(lctx, batch, cb, worst_case, is_warming_up, n_outputs);

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
        case LLM_ARCH_QWEN3NEXT:
            {
                result = llm.build_qwen3next();
            } break;
        case LLM_ARCH_QWEN35MOE:
            {
                result = llm.build_qwen35moe();
            } break;
        case LLM_ARCH_QWEN35:
            {
                result = llm.build_qwen35();
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
        case LLM_ARCH_GEMMA4:
            {
                result = llm.build_gemma4();
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
        case LLM_ARCH_GLM_DSA:
        case LLM_ARCH_MISTRAL4:
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
        case LLM_ARCH_SEED_OSS:
            {
                result = llm.build_seedoss();
            } break;
        case LLM_ARCH_STEP35:
            {
                result = llm.build_step35();
            } break;
        default:
            GGML_ABORT("fatal error");
    }

    result->n_batch = llm.n_tokens;

    // add on pooling layer
    if (lctx.cparams.mtp_op_type == MTP_OP_NONE && (lctx.cparams.embeddings ||
        (lctx.model.hparams.nextn_predict_layers > 0 || lctx.model.mtp))) {
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
        ggml_tensor * input, ggml_tensor * inp_pos, ggml_tensor * inp_out_ids, ggml_tensor * rope_factors_in,
        ggml_tensor * KQ_mask, ggml_tensor * sinks, ggml_tensor * inp_attn_scale, float KQ_scale, float f_attn_scale,
        int n_swa, int il, bool do_rope, bool add_graph_split, bool add_input, bool is_norm, bool is_multi,
        ggml_tensor * post_norm) {

    float freq_base_l  = n_swa > 0 ? hparams.rope_freq_base_train_swa : cparams.rope_freq_base;
    float freq_scale_l = n_swa > 0 ? hparams.rope_freq_scale_train_swa : hparams.rope_freq_scale_train;
    if (hparams.has_rope_freq_base_per_layer) {
        freq_base_l = hparams.rope_freq_base_per_layer[il];
    }
    int n_rot_l = lctx.model.hparams.rope_n_rot(il);
#ifdef GGML_USE_VULKAN
    constexpr bool use_f32_precision = true;
#else
    constexpr bool use_f32_precision = false;
#endif

    bool should_use_f32_precision = use_f32_precision
                                  ||  model.arch == LLM_ARCH_PHI2
                                  || model.arch == LLM_ARCH_PHI3
                                  || model.arch == LLM_ARCH_GPTNEOX
                                  || model.arch == LLM_ARCH_QWEN2
                                  || model.arch == LLM_ARCH_COHERE2
                                  || model.arch == LLM_ARCH_COMMAND_R
                                  || model.arch == LLM_ARCH_GLM4
                               //   || model.arch == LLM_ARCH_GLM4_MOE
                                  || model.arch == LLM_ARCH_MIMO2;
                               // || (model.arch == LLM_ARCH_DEEPSEEK2 && q->ne[1] <= 8);

    if (!model.layers[il].wqkv && !model.layers[il].wqk && cparams.flash_attn &&
         model.layers[il].wq->extra && model.layers[il].wk->extra && model.layers[il].wv->extra && model.layers[il].wo->extra) {
        if (kv_self.k_l[il]->extra && kv_self.v_l[il]->extra) {
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
            bool output_bias_added = false;
            int last_id = -1;
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
                auto cur = get_input_tensor_sm_graph(ctx0, input, id);
                cur = do_split_norm(ctx0, cur, the_attn_norm, lctx.model.hparams, cb, id, il_cb, is_norm);
                auto input_normed = cur;
                auto the_q_norm = model.layers[il].attn_q_norm ? model.layers[il].attn_q_norm->extra ?
                    ((ggml_split_tensor_t *)model.layers[il].attn_q_norm->extra)->splits[id] : model.layers[il].attn_q_norm : nullptr;
                auto the_k_norm = model.layers[il].attn_k_norm ? model.layers[il].attn_k_norm->extra ?
                    ((ggml_split_tensor_t *)model.layers[il].attn_k_norm->extra)->splits[id] : model.layers[il].attn_k_norm : nullptr;
                ggml_tensor *Qcur, *Kcur, *Vcur, *gate = nullptr;
                if (model.arch == LLM_ARCH_QWEN3NEXT || model.arch == LLM_ARCH_QWEN35 || model.arch == LLM_ARCH_QWEN35MOE) {
                    auto [Q, K, V, G] = llm_build_mul_mat_qkv_gated(gf, cur, split_wq, split_wk, split_wv,
                            the_q_norm, the_k_norm, il);
                    Qcur = Q; Kcur = K; Vcur = V; gate = G;
                } else {
                    auto [Q, K, V] = llm_build_mul_mat_qkv(gf, cur, nullptr, nullptr, nullptr, nullptr,
                            split_wq, bq ? bq->splits[id] : nullptr,
                            split_wk, bk ? bk->splits[id] : nullptr,
                            split_wv, bv ? bv->splits[id] : nullptr,
                            the_q_norm, the_k_norm, f_attn_scale, il, add_graph_split);
                    Qcur = Q; Kcur = K; Vcur = V;
                }
                auto rope_factors = rope_factors_in;
                if (rope_factors) {
                    GGML_ASSERT(rope_factors->extra);
                    rope_factors = ((ggml_split_tensor_t *)rope_factors->extra)->splits[id];
                    GGML_ASSERT(rope_factors);
                }
                else if (model.layers[il].rope_freqs && model.layers[il].rope_freqs->extra) {
                    auto extra = (ggml_split_tensor_t *)model.layers[il].rope_freqs->extra;
                    rope_factors = extra->splits[id];
                }
                if (do_rope) {
                    if (is_multi) {
                        int sections[4];
                        std::copy(hparams.rope_sections.begin(), hparams.rope_sections.begin() + GGML_MROPE_SECTIONS, sections);
                        Qcur = ggml_rope_multi(ctx0, Qcur, inp_pos, rope_factors,
                                n_rot_l, sections, rope_type, n_ctx_orig, freq_base_l, freq_scale_l,
                                ext_factor, attn_factor, beta_fast, beta_slow);
                        Kcur = ggml_rope_multi(ctx0, Kcur, inp_pos, rope_factors,
                                n_rot_l, sections, rope_type, n_ctx_orig, freq_base_l, freq_scale_l,
                                ext_factor, attn_factor, beta_fast, beta_slow);
                    } else {
                        Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, rope_factors, n_rot_l, rope_type, n_ctx_orig, freq_base_l, freq_scale_l,
                                ext_factor, attn_factor, beta_fast, beta_slow);
                        Kcur = ggml_rope_ext(ctx0, Kcur, inp_pos, rope_factors, n_rot_l, rope_type, n_ctx_orig, freq_base_l, freq_scale_l,
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
                    Qcur = ggml_hadamard(ctx0, Qcur, hparams.n_embd_head_k(il));
                    Kcur = ggml_hadamard(ctx0, Kcur, hparams.n_embd_head_k(il));
                    cb(Qcur, "Qcur_hadamard", il_cb);
                    cb(Kcur, "Kcur_hadamard", il_cb);
                }
                if (cparams.v_cache_hadamard) {
                    Vcur = ggml_hadamard(ctx0, Vcur, hparams.n_embd_head_v(il));
                    cb(Vcur, "Vcur_hadamard", il_cb);
                }
                ggml_build_forward_expand(gf, Qcur);
                ggml_build_forward_expand(gf, Kcur);
                ggml_build_forward_expand(gf, Vcur);

                const int64_t n_embd_head_k = hparams.n_embd_head_k(il);
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
                if (should_use_f32_precision) {
                    ggml_flash_attn_ext_set_prec(cur, GGML_PREC_F32);
                }

                if (cparams.v_cache_hadamard) {
                    cur = ggml_hadamard(ctx0, cur, n_embd_head_v);
                    cb(cur, "flash_attn_h", il_cb);
                }

                if (model.layers[il].wqkv_gate) {
                    auto wqkv_gate = (ggml_split_tensor_t *)model.layers[il].wqkv_gate->extra;
                    GGML_ASSERT(wqkv_gate && wqkv_gate->splits[id]);
                    auto gate = llm_build_lora_mm(lctx, ctx0, wqkv_gate->splits[id], input_normed);
                    cb(gate, "attn_gate", il_cb);
                    int nh = split_wo->ne[0]/n_embd_head_v;
                    auto attn_3d = ggml_reshape_3d(ctx0, cur, n_embd_head_v, nh, n_tokens);
                    auto gate_3d = ggml_reshape_3d(ctx0, gate,            1, nh, n_tokens);
                    cur = ggml_fused_mul_unary(ctx0, gate_3d, attn_3d, GGML_UNARY_OP_SIGMOID);
                    cb(attn_3d, "attn_gated_3d", il_cb);
                }

                cur = ggml_reshape_2d(ctx0, cur, split_wo->ne[0], n_tokens);
                cb(cur, "flash_attn_reshaped", il_cb);
                if (gate) {
                    if (false && cur->ne[1] == 1) { // we need to add GGML_UNARY_OP_SIGMOID to the ops supported by ggml_fused_mul_unary
                        cur = ggml_fused_mul_unary(ctx0, cur, gate, GGML_UNARY_OP_SIGMOID);
                    } else {
                        gate = ggml_sigmoid(ctx0, gate);
                        cb(gate, "gate", il_cb);
                        cur = ggml_mul(ctx0, cur, gate);
                    }
                    cb(cur, "qkv_gated", il_cb);
                }

                if (inp_out_ids) {
                    cur = ggml_get_rows(ctx0, cur, inp_out_ids);
                    cb(cur, "fa_get_rows", il_cb);
                }

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
                if (cur->ne[1] > 32 && lctx.cparams.reduce_type != GGML_TYPE_F32) {
                    cur = ggml_cast(ctx0, cur, lctx.cparams.reduce_type);
                }
                ggml_build_forward_expand(gf, cur);
                attn[id] = cur;
                last_id = id;
            }
            GGML_ASSERT(last_id >= 0);
            if (add_input) {
                if (inp_out_ids) {
                    input = ggml_get_rows(ctx0, input, inp_out_ids);
                    cb(input, "sainp_get_rows", il);
                }
                attn[last_id] = ggml_add(ctx0, attn[last_id], input);
                cb(attn[last_id], "attn_out_with_input", il);
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
    auto input_normed = cur;

    ggml_tensor *Qcur, *Kcur, *Vcur, *gate = nullptr;
    if (model.arch == LLM_ARCH_QWEN3NEXT || model.arch == LLM_ARCH_QWEN35 || model.arch == LLM_ARCH_QWEN35MOE) {
        auto [Q, K, V, G] = llm_build_mul_mat_qkv_gated(gf, cur, model.layers[il].wq, model.layers[il].wk, model.layers[il].wv,
                model.layers[il].attn_q_norm, model.layers[il].attn_k_norm, il);
        Qcur = Q; Kcur = K; Vcur = V; gate = G;
    } else {
        auto [Q, K, V] = llm_build_mul_mat_qkv(gf, cur,
                model.layers[il].wqkv, model.layers[il].bqkv,
                model.layers[il].wqk,  model.layers[il].bqk,
                model.layers[il].wq,   model.layers[il].bq, model.layers[il].wk, model.layers[il].bk, model.layers[il].wv, model.layers[il].bv,
                model.layers[il].attn_q_norm, model.layers[il].attn_k_norm, f_attn_scale, il);
        Qcur = Q; Kcur = K; Vcur = V;
        if (model.arch == LLM_ARCH_GEMMA4) {
            Vcur = ggml_reshape_3d(ctx0, Vcur, model.hparams.n_embd_head_v(il), model.hparams.n_head_kv(il), n_tokens);
            Vcur = ggml_rms_norm(ctx0, Vcur, model.hparams.f_norm_rms_eps);
        }
    }

    if (do_rope) {
        if (is_multi) {
            int sections[4];
            std::copy(hparams.rope_sections.begin(), hparams.rope_sections.begin() + GGML_MROPE_SECTIONS, sections);
            Qcur = ggml_rope_multi(ctx0, Qcur, inp_pos, rope_factors_in,
                    n_rot_l, sections, rope_type, n_ctx_orig, freq_base_l, freq_scale_l,
                    ext_factor, attn_factor, beta_fast, beta_slow);
            Kcur = ggml_rope_multi(ctx0, Kcur, inp_pos, rope_factors_in,
                    n_rot_l, sections, rope_type, n_ctx_orig, freq_base_l, freq_scale_l,
                    ext_factor, attn_factor, beta_fast, beta_slow);
        } else {
            Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, rope_factors_in, n_rot_l, rope_type, n_ctx_orig, freq_base_l, freq_scale_l,
                    ext_factor, attn_factor, beta_fast, beta_slow);
            Kcur = ggml_rope_ext( ctx0, Kcur, inp_pos, rope_factors_in, n_rot_l, rope_type, n_ctx_orig, freq_base_l, freq_scale_l,
                    ext_factor, attn_factor, beta_fast, beta_slow);
        }
    }
    cb(Qcur, "Qcur_roped", il);
    cb(Kcur, "Kcur_roped", il);

    if (inp_attn_scale) {
        Qcur = ggml_mul(ctx0, Qcur, inp_attn_scale);
        cb(Qcur, "Qcur_temp_scaled", il);
    }

    if (auto wqkv_gate = model.layers[il].wqkv_gate; wqkv_gate != nullptr) {
        cur = llm_build_kv(ctx0, lctx, kv_self, gf,
                nullptr, nullptr,
                Kcur, Vcur, Qcur, KQ_mask, n_tokens, kv_head, n_kv, KQ_scale, cb, il, sinks, n_swa);
        cb(cur, "wqkv", il);
        auto gate = llm_build_lora_mm(lctx, ctx0, wqkv_gate, input_normed); // [n_head_l, n_tokens]
        cb(gate, "attn_gate", il);
        int n_head_l = hparams.n_head(il);
        auto attn_3d = ggml_reshape_3d(ctx0, cur, n_embd_head_v, n_head_l, n_tokens);
        auto gate_3d = ggml_reshape_3d(ctx0, gate,            1, n_head_l, n_tokens);
        cur = ggml_fused_mul_unary(ctx0, gate_3d, attn_3d, GGML_UNARY_OP_SIGMOID);
        cb(cur, "attn_gated_3d", il);
        cur = ggml_reshape_2d(ctx0, cur, n_embd_head_v * n_head_l, n_tokens);
        cb(cur, "attn_gated", il);
        cur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wo, cur);
        if (model.layers[il].bo) {
            cur = ggml_add(ctx0, cur, model.layers[il].bo);
        }
        cb(cur, "attn_out", il);
    } else {
        if (gate) {
            cur = llm_build_kv(ctx0, lctx, kv_self, gf, nullptr, nullptr,
                    Kcur, Vcur, Qcur, KQ_mask, n_tokens, kv_head, n_kv, KQ_scale, cb, il, sinks, n_swa);
            if (false && cur->ne[1] == 1) { // we need to add GGML_UNARY_OP_SIGMOID to the ops supported by ggml_fused_mul_unary
                cur = ggml_fused_mul_unary(ctx0, cur, gate, GGML_UNARY_OP_SIGMOID);
            } else {
                gate = ggml_sigmoid(ctx0, gate);
                cb(gate, "gate", il);
                cur = ggml_mul(ctx0, cur, gate);
            }
            cb(cur, "qkv_gated", il);
            cur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wo, cur);
            if (model.layers[il].bo) {
                cur = ggml_add(ctx0, cur, model.layers[il].bo);
            }
            cb(cur, "attn_out", il);
        } else {
            cur = llm_build_kv(ctx0, lctx, kv_self, gf,
                    model.layers[il].wo, model.layers[il].bo,
                    Kcur, Vcur, Qcur, KQ_mask, n_tokens, kv_head, n_kv, KQ_scale, cb, il, sinks, n_swa);
        }
    }

    if (inp_out_ids) {
        cur = ggml_get_rows(ctx0, cur, inp_out_ids);
        cb(cur, "sa_get_rows", il);
        if (add_input) {
            input = ggml_get_rows(ctx0, input, inp_out_ids);
            cb(input, "sainp_get_rows", il);
        }
    }

    if (post_norm) {
        cur = llm_build_norm(ctx0, cur, hparams, post_norm, NULL, LLM_NORM_RMS, cb, il);
        cb(cur, "sa_normed", il);
    }

    if (add_input) {
        cb(cur, "attn_out", il);
        cur = ggml_add(ctx0, cur, input);
    }

    return cur;
}

int32_t llama_model_n_nextn_layer(const llama_model * model) {
    return model->hparams.nextn_predict_layers;
}

#include "../llama-build-context.h"

#include "../llama-context.h"
#include "../llama-delta-net.h"
#include "../llama-model.h"
#include "../llama-hparams.h"

struct k3_resi_stack {
    ggml_tensor * stack = nullptr;

    void push(ggml_context * ctx0, ggml_tensor * cur, int64_t n_embd, int64_t n_tokens) {
        ggml_tensor * ckpt = ggml_reshape_3d(ctx0, cur, n_embd, 1, n_tokens);
        stack = stack ? ggml_concat(ctx0, stack, ckpt, 1) : ckpt;
    }

    ggml_tensor * mix(llm_build_context & llm, ggml_cgraph * gf, ggml_tensor * cur,
                      ggml_tensor * score_w, int64_t n_tokens, int il) {
        if (!stack) {
            return cur;
        }

        auto & ctx0    = llm.ctx0;
        auto & hparams = llm.hparams;
        auto & cb      = llm.cb;

        GGML_ASSERT(score_w && "K3: attn_res.block_size > 0 requires the *_res_score weights");

        const int64_t n_ckpt = stack->ne[1];
        const float   eps    = hparams.f_norm_rms_eps;

        ggml_tensor * scores = ggml_rms_norm(ctx0, stack, eps);
        scores = ggml_mul(ctx0, scores, score_w);
        scores = ggml_sum_rows(ctx0, scores);
        scores = ggml_reshape_2d(ctx0, scores, n_ckpt, n_tokens);

        ggml_tensor * sc_cur = ggml_rms_norm(ctx0, cur, eps);
        sc_cur = ggml_mul(ctx0, sc_cur, score_w);
        sc_cur = ggml_sum_rows(ctx0, sc_cur);

        scores = ggml_concat(ctx0, scores, sc_cur, 0);
        ggml_tensor * probs = ggml_soft_max(ctx0, scores);
        cb(probs, "res_probs", il);
        ggml_build_forward_expand(gf, probs);

        ggml_tensor * x = ggml_cont(ctx0, ggml_permute(ctx0, stack, 1, 0, 2, 3));
        ggml_tensor * p = ggml_view_3d(ctx0, probs, n_ckpt, 1, n_tokens,
                probs->nb[1], probs->nb[1], 0);
        ggml_tensor * out = ggml_sum_rows(ctx0, ggml_mul(ctx0, x, p));
        out = ggml_reshape_2d(ctx0, out, x->ne[1], n_tokens);
        cb(out, "res_mix_stack", il);

        ggml_tensor * p_cur = ggml_view_2d(ctx0, probs, 1, n_tokens, probs->nb[1],
                probs->nb[0] * n_ckpt);
        out = ggml_add(ctx0, out, ggml_mul(ctx0, cur, p_cur));
        cb(out, "res_mix", il);

        return out;
    }
};

static ggml_tensor * k3_build_mla(
        llm_build_context & llm, ggml_cgraph * gf, ggml_tensor * raw,
        ggml_tensor * KQ_mask, float kq_scale, int il) {
    auto & ctx0    = llm.ctx0;
    auto & lctx    = llm.lctx;
    auto & hparams = llm.hparams;
    auto & model   = llm.model;
    auto & kv_self = lctx.kv_self;
    auto & cb      = llm.cb;
    auto & layer   = model.layers[il];

    const int64_t n_tokens     = llm.n_tokens;
    const int64_t n_head       = llm.n_head;
    const int64_t kv_lora_rank = hparams.n_lora_kv;
    const int64_t n_rot        = hparams.n_rot;
    const int64_t head_k       = hparams.n_embd_head_k_mla;
    const int64_t head_v       = hparams.n_embd_head_v_mla;
    const int64_t qk_nope      = head_k - n_rot;
    const int64_t k_row        = kv_lora_rank + n_rot;

    GGML_ASSERT(hparams.n_embd_head_k_full == (uint32_t) k_row &&
            "K3: attention.key_length must be kv_lora_rank + qk_rope_head_dim");
    GGML_ASSERT(layer.wk_b && layer.wv_b &&
            "K3: the absorbed MLA form requires attn_k_b and attn_v_b");

    ggml_tensor * cur = llm_build_context::llm_build_norm(ctx0, raw, hparams, layer.attn_norm,
            nullptr, LLM_NORM_RMS, cb, il);
    cb(cur, "attn_norm", il);

    ggml_tensor * q = nullptr;
    if (layer.wq_a) {
        q = llm_build_context::llm_build_lora_mm(lctx, ctx0, layer.wq_a, cur);
        cb(q, "q_a", il);
        q = llm_build_context::llm_build_norm(ctx0, q, hparams, layer.attn_q_a_norm,
                nullptr, LLM_NORM_RMS, cb, il);
        cb(q, "q_a_norm", il);
        q = llm_build_context::llm_build_lora_mm(lctx, ctx0, layer.wq_b, q);
    } else {
        q = llm_build_context::llm_build_lora_mm(lctx, ctx0, layer.wq, cur);
    }
    cb(q, "q", il);

    ggml_tensor * kv = llm_build_context::llm_build_lora_mm(lctx, ctx0, layer.wkv_a_mqa, cur);
    cb(kv, "kv_a_mqa", il);

    const size_t kv_row_bytes   = ggml_row_size(kv->type, k_row);
    const size_t cache_row_bytes = ggml_row_size(kv_self.k_l[il]->type, k_row);
    ggml_tensor * kv_cmpr = ggml_cont(ctx0, ggml_view_2d(ctx0, kv, kv_lora_rank, n_tokens, kv_row_bytes, 0));
    ggml_tensor * k_pe    = ggml_cont(ctx0, ggml_view_2d(ctx0, kv, n_rot, n_tokens, kv_row_bytes,
            ggml_row_size(kv->type, kv_lora_rank)));

    kv_cmpr = llm_build_context::llm_build_norm(ctx0, kv_cmpr, hparams, layer.attn_kv_a_norm,
            nullptr, LLM_NORM_RMS, cb, il);
    cb(kv_cmpr, "kv_compressed", il);

    ggml_tensor * q3 = ggml_reshape_3d(ctx0, q, head_k, n_head, n_tokens);
    q3 = ggml_permute(ctx0, q3, 0, 2, 1, 3);
    ggml_tensor * q_nope = ggml_cont(ctx0, ggml_view_3d(ctx0, q3, qk_nope, n_tokens, n_head,
            q3->nb[1], q3->nb[2], 0));
    ggml_tensor * q_pe   = ggml_cont(ctx0, ggml_view_3d(ctx0, q3, n_rot, n_tokens, n_head,
            q3->nb[1], q3->nb[2], qk_nope * q3->nb[0]));

    ggml_tensor * Q = ggml_mul_mat(ctx0, layer.wk_b, q_nope);
    Q = ggml_cont(ctx0, ggml_concat(ctx0, Q, q_pe, 0));
    cb(Q, "Q_absorbed", il);

    ggml_tensor * kv_full = ggml_cont(ctx0, ggml_concat(ctx0, ggml_reshape_3d(ctx0, kv_cmpr,
            kv_lora_rank, 1, n_tokens), ggml_reshape_3d(ctx0, k_pe, n_rot, 1, n_tokens), 0));
    cb(kv_full, "kv_full", il);

    ggml_tensor * k_write = ggml_view_2d(ctx0, kv_self.k_l[il], k_row, n_tokens,
            cache_row_bytes, cache_row_bytes * llm.kv_head);
    lctx.cache_copies[2 * il + 0].cpy  = ggml_cpy(ctx0, kv_full, k_write);
    lctx.cache_copies[2 * il + 0].step  = cache_row_bytes;
    ggml_build_forward_expand(gf, lctx.cache_copies[2 * il + 0].cpy);

    const bool mla_v_cache = lctx.cparams.mla_attn == 1 && !lctx.cparams.flash_attn;
    if (mla_v_cache) {
        ggml_tensor * v_write = ggml_view_2d(ctx0, kv_self.v_l[il], n_tokens, kv_lora_rank,
                ggml_row_size(kv_self.v_l[il]->type, kv_self.rows(il)),
                ggml_row_size(kv_self.v_l[il]->type, llm.kv_head));
        cb(v_write, "kv_cache_trans_view", il);
        ggml_build_forward_expand(gf, ggml_cpy(ctx0, ggml_transpose(ctx0, kv_cmpr), v_write));
    }

    ggml_tensor * K = ggml_view_2d(ctx0, kv_self.k_l[il], k_row, llm.n_kv, cache_row_bytes, 0);
    ggml_tensor * V = mla_v_cache ? nullptr
            : ggml_view_2d(ctx0, kv_self.k_l[il], kv_lora_rank, llm.n_kv, cache_row_bytes, 0);

    cb(K, "kv_cache", il);

    GGML_ASSERT(!mla_v_cache || kv_self.v_l[il] != nullptr);
    GGML_ASSERT(V == nullptr || !ggml_is_quantized(kv_self.k_l[il]->type));

    ggml_tensor * attn = nullptr;
    if (lctx.cparams.flash_attn) {
        attn = ggml_flash_attn_ext(ctx0, Q, K, V, KQ_mask, kq_scale, hparams.f_max_alibi_bias, 0.f);
        if (K->ne[1] < 256) {
            ggml_flash_attn_ext_set_prec(attn, GGML_PREC_F32);
        }
        attn = ggml_permute(ctx0, attn, 0, 2, 1, 3);
    } else {
        ggml_tensor * kq = ggml_mul_mat(ctx0, K, Q);
        if (K->ne[1] < 256) {
            ggml_mul_mat_set_prec(kq, GGML_PREC_F32);
        }
        kq = ggml_soft_max_ext(ctx0, kq, KQ_mask, kq_scale, hparams.f_max_alibi_bias);
        cb(kq, "kq", il);
        ggml_tensor * VP;
        if (mla_v_cache) {
            VP = ggml_view_2d(ctx0, kv_self.v_l[il], llm.n_kv, kv_lora_rank,
                    ggml_row_size(kv_self.v_l[il]->type, kv_self.rows(il)), 0);
        } else {
            VP = ggml_cont(ctx0, ggml_transpose(ctx0, V));
        }
        cb(VP, "kv_cache_trans", il);
        attn = ggml_mul_mat(ctx0, VP, kq);
    }
    cb(attn, "mla_attn_out", il);

    ggml_tensor * wv_b = layer.wv_b;
    if (wv_b->ne[1] != head_v) {
        wv_b = ggml_reshape_3d(ctx0, wv_b, kv_lora_rank, head_v, n_head);
    }
    ggml_tensor * out = ggml_mul_mat(ctx0, wv_b, attn);
    out = ggml_reshape_2d(ctx0, ggml_cont(ctx0, ggml_permute(ctx0, out, 0, 2, 1, 3)),
            head_v * n_head, n_tokens);
    cb(out, "mla_merged", il);

    if (layer.wqkv_gate) {
        ggml_tensor * g = llm_build_context::llm_build_lora_mm(lctx, ctx0, layer.wqkv_gate, cur);
        out = ggml_fused_mul_unary(ctx0, g, out, GGML_UNARY_OP_SIGMOID);
        cb(out, "mla_gated", il);
    }

    out = llm_build_context::llm_build_lora_mm(lctx, ctx0, layer.wo, out);
    cb(out, "mla_out", il);

    return out;
}

static ggml_tensor * k3_build_latent_moe(llm_build_context & llm, ggml_cgraph * gf,
        ggml_tensor * cur, int il) {
    auto & ctx0    = llm.ctx0;
    auto & lctx    = llm.lctx;
    auto & hparams = llm.hparams;
    auto & model   = llm.model;
    auto & cb      = llm.cb;
    auto & layer   = model.layers[il];

    ggml_tensor * routed_in = layer.ffn_routed_down
        ? llm_build_context::llm_build_lora_mm(lctx, ctx0, layer.ffn_routed_down, cur)
        : cur;
    if (layer.ffn_routed_down) {
        cb(routed_in, "ffn_routed_down", il);
    }

    ggml_tensor * logits = llm_build_context::llm_build_lora_mm(lctx, ctx0, layer.ffn_gate_inp, cur);
    cb(logits, "ffn_moe_logits", il);
    ggml_build_forward_expand(gf, logits);

    ggml_tensor * moe = llm_build_context::llm_build_moe_ffn(ctx0, lctx, routed_in,
            nullptr, nullptr,
            layer.ffn_up_exps,   nullptr,
            layer.ffn_gate_exps, nullptr,
            layer.ffn_down_exps, nullptr,
            layer.ffn_exp_probs_b,
            llm.n_expert, llm.n_expert_used,
            LLM_FFN_SITU, hparams.expert_weights_norm,
            true, hparams.expert_weights_scale,
            (llm_expert_gating_func_type) hparams.expert_gating_func,
            cb, il, gf, false, layer.ffn_up_gate_exps, nullptr,
            logits, nullptr, nullptr);
    cb(moe, "ffn_moe_out", il);

    if (layer.ffn_routed_norm) {
        moe = llm_build_context::llm_build_norm(ctx0, moe, hparams, layer.ffn_routed_norm,
                nullptr, LLM_NORM_RMS, cb, il);
        cb(moe, "ffn_moe_latent_norm", il);
    }
    if (layer.ffn_routed_up) {
        moe = llm_build_context::llm_build_lora_mm(lctx, ctx0, layer.ffn_routed_up, moe);
        cb(moe, "ffn_routed_up", il);
    }

    if (layer.ffn_gate_shexp) {
        ggml_tensor * g = llm_build_context::llm_build_lora_mm(lctx, ctx0, layer.ffn_gate_shexp, cur);
        ggml_tensor * u = llm_build_context::llm_build_lora_mm(lctx, ctx0, layer.ffn_up_shexp,   cur);
        ggml_tensor * sh = llm_build_context::llm_build_situ(ctx0, hparams, g, u);
        sh = llm_build_context::llm_build_lora_mm(lctx, ctx0, layer.ffn_down_shexp, sh);
        cb(sh, "ffn_shexp", il);
        moe = ggml_add(ctx0, moe, sh);
    }

    return moe;
}

static ggml_tensor * k3_build_dense_ffn(llm_build_context & llm, ggml_tensor * cur, int il) {
    auto & ctx0  = llm.ctx0;
    auto & lctx  = llm.lctx;
    auto & layer = llm.model.layers[il];

    ggml_tensor * g = llm_build_context::llm_build_lora_mm(lctx, ctx0, layer.ffn_gate, cur);
    ggml_tensor * u = llm_build_context::llm_build_lora_mm(lctx, ctx0, layer.ffn_up,   cur);
    ggml_tensor * a = llm_build_context::llm_build_situ(ctx0, llm.hparams, g, u);
    return llm_build_context::llm_build_lora_mm(lctx, ctx0, layer.ffn_down, a);
}

ggml_cgraph * llm_build_context::build_kimi_k3() {
    ggml_cgraph * gf = new_graph_custom();

    delta_net delta(lctx, batch);

    ggml_tensor * cur  = nullptr;
    ggml_tensor * inpL = llm_build_inp_embd(ctx0, lctx, hparams, batch, model.tok_embd, cb);
    cb(inpL, "inp_embd", -1);

    ggml_tensor * KQ_mask    = build_inp_KQ_mask();
    ggml_tensor * inp_out_ids = n_tokens > 1 ? build_inp_out_ids() : nullptr;

    lctx.inp_s_seq_qnext = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, 1, n_tokens);
    cb(lctx.inp_s_seq_qnext, "inp_s_seq_qnext", -1);
    ggml_set_input(lctx.inp_s_seq_qnext);

    const bool     use_attn_res = hparams.attn_res_block_size > 0;
    const float    kq_scale     = 1.0f / sqrtf((float) hparams.n_embd_head_k_mla);
    k3_resi_stack   resi;

    GGML_ASSERT(hparams.kda_gate_lower_bound < 0.0f &&
            "K3: kda.gate_lower_bound must be < 0");

    for (int il = 0; il < n_layer; ++il) {
        auto & layer = model.layers[il];

        ggml_tensor * prefix_sum = inpL;

        cur = use_attn_res ? resi.mix(*this, gf, prefix_sum, layer.attn_res_score, n_tokens, il)
                           : prefix_sum;

        bool banked = false;
        if (use_attn_res && (uint32_t) il % hparams.attn_res_block_size == 0) {
            resi.push(ctx0, prefix_sum, n_embd, n_tokens);
            banked = true;
        }

        if (hparams.is_recurrent(il)) {
            cur = delta.build_layer_attn_kda(ctx0, gf, cur, nullptr, il, cb, false);
            cb(cur, "kda_out", il);
        } else {
            cur = k3_build_mla(*this, gf, cur, KQ_mask, kq_scale, il);
        }

        prefix_sum = banked ? cur : ggml_add(ctx0, prefix_sum, cur);
        cb(prefix_sum, "prefix_sum_attn", il);

        cur = use_attn_res ? resi.mix(*this, gf, prefix_sum, layer.ffn_res_score, n_tokens, il)
                           : prefix_sum;
        cur = llm_build_norm(ctx0, cur, hparams, layer.ffn_norm, nullptr, LLM_NORM_RMS, cb, il);
        cb(cur, "ffn_norm", il);

        if ((uint32_t) il < hparams.n_layer_dense_lead) {
            cur = k3_build_dense_ffn(*this, cur, il);
            cb(cur, "ffn_out", il);
        } else {
            cur = k3_build_latent_moe(*this, gf, cur, il);
            cb(cur, "ffn_out", il);
        }

        prefix_sum = ggml_add(ctx0, prefix_sum, cur);
        prefix_sum = lctx.cvec.apply_to(ctx0, prefix_sum, il);
        cb(prefix_sum, "l_out", il);

        inpL = prefix_sum;
    }

    cur = inpL;

    if (use_attn_res) {
        cur = resi.mix(*this, gf, cur, model.output_res_score, n_tokens, -1);
    }
    if (inp_out_ids) {
        cur = ggml_get_rows(ctx0, cur, inp_out_ids);
    }

    cur = build_output(lctx, ctx0, cur, model.output, model.output_norm, cb);
    cb(cur, "result_output", -1);

    ggml_build_forward_expand(gf, cur);
    return gf;
}

#include "../llama-build-context.h"

#include "../llama-context.h"
#include "../llama-delta-net.h"
#include "../llama-model.h"
#include "../llama-hparams.h"

struct k3_resi_stack {
    ggml_tensor * stack      = nullptr;
    ggml_tensor * stack_norm = nullptr;

    void push(ggml_context * ctx0, ggml_tensor * cur, int64_t n_embd, int64_t n_tokens, float eps) {
        ggml_tensor * ckpt = ggml_reshape_3d(ctx0, cur, n_embd, 1, n_tokens);
        ggml_tensor * norm = ggml_rms_norm(ctx0, ckpt, eps);
        stack      = stack      ? ggml_concat(ctx0, stack,      ckpt, 1) : ckpt;
        stack_norm = stack_norm ? ggml_concat(ctx0, stack_norm, norm, 1) : norm;
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

        // stack_norm is already RMS-normalized in push().
        ggml_tensor * scores = ggml_mul(ctx0, stack_norm, score_w);
        scores = ggml_sum_rows(ctx0, scores);
        scores = ggml_reshape_2d(ctx0, scores, n_ckpt, n_tokens);

        ggml_tensor * sc_cur = ggml_fused_rms_norm(ctx0, cur, score_w, eps);
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


static ggml_tensor * k3_build_latent_moe(llm_build_context & llm, ggml_cgraph * gf,
        ggml_tensor * cur, int il) {
    auto & ctx0    = llm.ctx0;
    auto & lctx    = llm.lctx;
    auto & hparams = llm.hparams;
    auto & model   = llm.model;
    auto & cb      = llm.cb;
    auto & layer   = model.layers[il];

    cur = llm_build_context::llm_build_norm(ctx0, cur, hparams, layer.ffn_norm, nullptr, LLM_NORM_RMS, cb, il);
    cb(cur, "ffn_norm", il);

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
        ggml_tensor * sh = llm_build_context::llm_build_ffn(ctx0, lctx, nullptr, cur,
                layer.ffn_up_shexp, nullptr, nullptr,
                layer.ffn_gate_shexp, nullptr, nullptr,
                layer.ffn_down_shexp, nullptr, nullptr, nullptr,
                LLM_FFN_SITU, LLM_FFN_PAR, cb, il, gf);
        cb(sh, "ffn_shexp", il);
        moe = ggml_add(ctx0, moe, sh);
    }

    return moe;
}

static ggml_tensor * k3_build_dense_ffn(llm_build_context & llm, ggml_cgraph * gf,
        ggml_tensor * cur, int il) {
    auto & ctx0  = llm.ctx0;
    auto & lctx  = llm.lctx;
    auto & layer = llm.model.layers[il];

    return llm_build_context::llm_build_ffn(ctx0, lctx, layer.ffn_norm, cur,
            layer.ffn_up, nullptr, nullptr,
            layer.ffn_gate, nullptr, nullptr,
            layer.ffn_down, nullptr, nullptr, nullptr,
            LLM_FFN_SITU, LLM_FFN_PAR, llm.cb, il, gf);
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
    const float    kq_scale     = 1.0f / sqrtf((float) hparams.n_embd_head_k_mla());
    // Expanded MLA materializes the whole KV cache; keep it for prefill only.
    const bool     pp_opt       = n_tokens >= 128 && lctx.cparams.mla_attn > 1;
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
            resi.push(ctx0, prefix_sum, n_embd, n_tokens, hparams.f_norm_rms_eps);
            banked = true;
        }

        if (hparams.is_recurrent(il)) {
            cur = delta.build_layer_attn_kda(ctx0, gf, cur, nullptr, il, cb, false);
            cb(cur, "kda_out", il);
        } else {
            cur = build_deepseek2_layer_attention(gf, il, cur, KQ_mask, nullptr, nullptr,
                    kq_scale, 1.0f, false, false, pp_opt);
        }

        prefix_sum = banked ? cur : ggml_add(ctx0, prefix_sum, cur);
        cb(prefix_sum, "prefix_sum_attn", il);

        cur = use_attn_res ? resi.mix(*this, gf, prefix_sum, layer.ffn_res_score, n_tokens, il)
                           : prefix_sum;

        if ((uint32_t) il < hparams.n_layer_dense_lead) {
            cur = k3_build_dense_ffn(*this, gf, cur, il);
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

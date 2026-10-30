#include "../llama-build-context.h"
#include "../llama-model.h"
#include "../llama-context.h"

#include <cmath>

static size_t xing4_elem_offset(const ggml_tensor * t, int64_t i) {
    return ggml_row_size(t->type, i);
}

static ggml_tensor * xing4_view_1d(ggml_context * ctx, ggml_tensor * t, int64_t ne0, int64_t i0) {
    return ggml_view_1d(ctx, t, ne0, xing4_elem_offset(t, i0));
}

static ggml_tensor * xing4_view_2d(ggml_context * ctx, ggml_tensor * t, int64_t ne0, int64_t ne1, int64_t i0) {
    return ggml_view_2d(ctx, t, ne0, ne1, t->nb[1], xing4_elem_offset(t, i0));
}

static ggml_tensor * xing4_hc_affine(ggml_context * ctx, ggml_tensor * x, ggml_tensor * scale, ggml_tensor * base) {
    return ggml_add(ctx, ggml_mul(ctx, x, scale), base);
}

static ggml_tensor * xing4_hc_pre(llm_build_context & llm, ggml_tensor * x, ggml_tensor * weights, int il) {
    const int64_t hc = llm.hparams.dsv4_hc_mult;
    const int64_t nt = x->ne[2];
    ggml_tensor * result = nullptr;
    for (int64_t ih = 0; ih < hc; ++ih) {
        auto xh = ggml_view_2d(llm.ctx0, x, llm.hparams.n_embd, nt, x->nb[2], ih*x->nb[1]);
        auto wh = ggml_view_2d(llm.ctx0, weights, 1, nt, weights->nb[1], ih*weights->nb[0]);
        auto cur = ggml_mul(llm.ctx0, xh, wh);
        result = result ? ggml_add(llm.ctx0, result, cur) : cur;
    }
    llm.cb(result, "hc_pre", il);
    return result;
}

static ggml_tensor * xing4_hc_sinkhorn(llm_build_context & llm, ggml_tensor * comb, int il) {
    auto & ctx0 = llm.ctx0;
    const auto & hp = llm.hparams;
    comb = ggml_soft_max(ctx0, comb);
    auto eps = ggml_fill(ctx0, ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, 1), hp.dsv4_hc_eps);
    auto norm_dst = [&]() {
        auto sum = ggml_cont(ctx0, ggml_permute(ctx0, comb, 1, 0, 2, 3));
        sum = ggml_sum_rows(ctx0, sum);
        sum = ggml_add(ctx0, sum, eps);
        sum = ggml_permute(ctx0, sum, 1, 0, 2, 3);
        comb = ggml_div(ctx0, comb, sum);
    };
    auto norm_src = [&]() {
        auto sum = ggml_sum_rows(ctx0, comb);
        sum = ggml_add(ctx0, sum, eps);
        comb = ggml_div(ctx0, comb, sum);
    };
    norm_dst();
    for (uint32_t i = 1; i < hp.dsv4_hc_sinkhorn_iters; ++i) {
        norm_src();
        norm_dst();
    }
    llm.cb(comb, "hc_comb", il);
    return comb;
}

static ggml_tensor * xing4_hc_pre_block(
        llm_build_context & llm, ggml_tensor * x,
        ggml_tensor * hc_fn, ggml_tensor * hc_scale, ggml_tensor * hc_base,
        ggml_tensor ** post, ggml_tensor ** comb, int il) {
    auto & ctx0 = llm.ctx0;
    const auto & hp = llm.hparams;
    const int64_t hc = hp.dsv4_hc_mult;
    const int64_t hc_dim = hc * hp.n_embd;
    const int64_t nt = x->ne[2];

    auto flat = ggml_reshape_2d(ctx0, x, hc_dim, nt);
    auto flat_norm = ggml_rms_norm(ctx0, flat, hp.f_norm_rms_eps);
    auto mixes = ggml_mul_mat(ctx0, hc_fn, flat_norm);
    llm.cb(mixes, "hc_mixes", il);

    auto pre = xing4_view_2d(ctx0, mixes, hc, nt, 0);
    pre = ggml_sigmoid(ctx0, xing4_hc_affine(ctx0, pre, xing4_view_1d(ctx0, hc_scale, 1, 0), xing4_view_1d(ctx0, hc_base, hc, 0)));
    llm.cb(pre, "hc_pre", il);

    *post = xing4_view_2d(ctx0, mixes, hc, nt, hc);
    *post = ggml_sigmoid(ctx0, xing4_hc_affine(ctx0, *post, xing4_view_1d(ctx0, hc_scale, 1, 1), xing4_view_1d(ctx0, hc_base, hc, hc)));
    *post = ggml_scale(ctx0, *post, hp.hc_magnitude);
    llm.cb(*post, "hc_post", il);

    *comb = xing4_view_2d(ctx0, mixes, hc*hc, nt, 2*hc);
    *comb = xing4_hc_affine(ctx0, *comb, xing4_view_1d(ctx0, hc_scale, 1, 2), xing4_view_1d(ctx0, hc_base, hc*hc, 2*hc));
    *comb = ggml_clamp(ctx0, *comb, -30.0f, 30.0f);
    *comb = ggml_reshape_3d(ctx0, *comb, hc, hc, nt);
    *comb = xing4_hc_sinkhorn(llm, *comb, il);
    llm.cb(*comb, "hc_comb", il);
    return xing4_hc_pre(llm, x, pre, il);
}

static ggml_tensor * xing4_hc_post_block(
        llm_build_context & llm, ggml_tensor * x, ggml_tensor * residual,
        ggml_tensor * post, ggml_tensor * comb, int il) {
    auto & ctx0 = llm.ctx0;
    const int64_t hc = llm.hparams.dsv4_hc_mult;
    const int64_t nt = x->ne[1];
    ggml_tensor * out = nullptr;
    for (int64_t dst = 0; dst < hc; ++dst) {
        auto post_dst = ggml_view_2d(ctx0, post, 1, nt, post->nb[1], dst*post->nb[0]);
        auto cur = ggml_mul(ctx0, x, post_dst);
        for (int64_t src = 0; src < hc; ++src) {
            auto res_src = ggml_view_2d(ctx0, residual, llm.hparams.n_embd, nt, residual->nb[2], src*residual->nb[1]);
            auto w = ggml_view_2d(ctx0, comb, 1, nt, comb->nb[2], src*comb->nb[0] + dst*comb->nb[1]);
            cur = ggml_add(ctx0, cur, ggml_mul(ctx0, res_src, w));
        }
        cur = ggml_reshape_3d(ctx0, cur, llm.hparams.n_embd, 1, nt);
        out = out ? ggml_concat(ctx0, out, cur, 1) : cur;
    }
    llm.cb(out, "hc_post_out", il);
    return out;
}

ggml_cgraph * llm_build_context::build_xing4() {
    const auto & hp = hparams;
    auto & model_ = model;
    const int64_t n_embd = hp.n_embd;
    const int64_t n_tokens = this->n_tokens;
    const int64_t hc = hp.dsv4_hc_mult;

    auto gf = new_graph_custom();
    auto inp = llm_build_inp_embd(ctx0, lctx, hp, batch, model_.tok_embd, cb);
    inp = ggml_reshape_3d(ctx0, inp, n_embd, 1, n_tokens);
    ggml_tensor * inpL = ggml_repeat_4d(ctx0, inp, n_embd, hc, n_tokens, 1);

    auto inp_pos = build_inp_pos();
    auto KQ_mask = build_inp_KQ_mask();
    auto * inp_out_ids = (n_outputs != n_tokens) ? build_inp_out_ids() : nullptr;

    const int64_t n_head = hp.n_head(0);
    const int64_t n_embd_head_k = hp.n_embd_head_k_mla();
    const float attn_factor_org = attn_factor * (1.0f + 0.1f * logf(1.0f / freq_scale));
    const float mscale = attn_factor_org * (1.0f + 0.1f * hp.rope_yarn_log_mul * logf(1.0f / freq_scale));
    const float kq_scale = mscale * mscale / sqrtf(float(n_embd_head_k));
    const float attn_factor_scaled = 1.0f / (1.0f + 0.1f * logf(1.0f / freq_scale));
    const bool pp_opt = n_tokens >= 128 && lctx.cparams.mla_attn > 1;

    for (int il = 0; il < (int) hp.n_layer; ++il) {
        auto & layer = model_.layers[il];
        ggml_tensor * residual = inpL;
        ggml_tensor * post = nullptr;
        ggml_tensor * comb = nullptr;

        auto cur = xing4_hc_pre_block(*this, residual, layer.hc_attn_fn, layer.hc_attn_scale, layer.hc_attn_base, &post, &comb, il);
        cur = build_deepseek2_layer_attention(gf, il, cur, KQ_mask, inp_pos, nullptr,
                kq_scale, attn_factor_scaled, false, false, pp_opt);
        inpL = xing4_hc_post_block(*this, cur, residual, post, comb, il);

        residual = inpL;
        cur = xing4_hc_pre_block(*this, residual, layer.hc_ffn_fn, layer.hc_ffn_scale, layer.hc_ffn_base, &post, &comb, il);
        cur = llm_build_norm(ctx0, cur, hp, layer.ffn_norm, nullptr, LLM_NORM_RMS, cb, il);

        if ((uint32_t) il < hp.n_layer_dense_lead) {
            cur = llm_build_ffn(ctx0, lctx, nullptr, cur,
                    layer.ffn_up, nullptr, nullptr,
                    layer.ffn_gate, nullptr, nullptr,
                    layer.ffn_down, nullptr, nullptr,
                    nullptr, LLM_FFN_SILU, LLM_FFN_PAR, cb, il);
        } else {
            auto moe = llm_build_moe_ffn(ctx0, lctx, cur,
                    layer.ffn_gate_inp,
                    layer.ffn_up_exps,
                    layer.ffn_gate_exps,
                    layer.ffn_down_exps,
                    layer.ffn_exp_probs_b,
                    hp.n_expert, hp.n_expert_used,
                    LLM_FFN_SILU, hp.expert_weights_norm,
                    true, hp.expert_weights_scale,
                    (enum llm_expert_gating_func_type) hp.expert_gating_func,
                    cb, il, gf, false, layer.ffn_up_gate_exps);
            auto shexp = llm_build_ffn(ctx0, lctx, nullptr, cur,
                    layer.ffn_up_shexp, nullptr, nullptr,
                    layer.ffn_gate_shexp, nullptr, nullptr,
                    layer.ffn_down_shexp, nullptr, nullptr,
                    nullptr, LLM_FFN_SILU, LLM_FFN_PAR, cb, il, gf);
            cur = ggml_add(ctx0, moe, shexp);
        }
        inpL = xing4_hc_post_block(*this, cur, residual, post, comb, il);
    }

    auto flat = ggml_reshape_2d(ctx0, inpL, n_embd * hc, n_tokens);
    if (inp_out_ids) {
        flat = ggml_get_rows(ctx0, flat, inp_out_ids);
    }
    const int64_t n_out = flat->ne[1];
    ggml_tensor * sums = nullptr;
    for (int64_t s = 0; s < hc; ++s) {
        auto stream = ggml_view_2d(ctx0, flat, n_embd, n_out, flat->nb[1], s * n_embd * ggml_element_size(flat));
        sums = sums ? ggml_add(ctx0, sums, stream) : stream;
    }
    auto cur = ggml_scale(ctx0, sums, 1.0f / hc);
    cur = llm_build_norm(ctx0, cur, hp, model_.output_norm, nullptr, LLM_NORM_RMS, cb, -1);
    cur = llm_build_lora_mm(lctx, ctx0, model_.output, cur);
    cb(cur, "result_output", -1);
    ggml_build_forward_expand(gf, cur);
    return gf;
}

#include "../llama-build-context.h"
#include "../llama-model.h"
#include "../llama-context.h"

#include <cmath>

static ggml_tensor * xing4_hc_pre_block(
        llm_build_context & llm, ggml_tensor * x,
        ggml_tensor * hc_fn, ggml_tensor * hc_scale, ggml_tensor * hc_base,
        ggml_tensor ** post, ggml_tensor ** comb, int il) {
    auto & ctx0 = llm.ctx0;
    const auto & hp = llm.hparams;
    const int64_t hc = hp.dsv4_hc_mult;
    const int64_t n_embd = hp.n_embd;
    const int64_t nt = x->ne[2];

    auto mixes = llm.build_mhc_pre_projection(x, hc_fn, nullptr, n_embd, hc,
                                               hp.f_norm_rms_eps, true);
    llm.cb(mixes, "hc_mixes", il);

    auto all = ggml_hc_pre(ctx0, mixes, hc_scale, hc_base, hc,
                           hp.dsv4_hc_sinkhorn_iters, 0.0f);
    auto pre  = ggml_view_2d(ctx0, all, hc, nt, hc*sizeof(float), 0);
    *post     = ggml_view_2d(ctx0, all, hc, nt, hc*sizeof(float), hc*nt*sizeof(float));
    if (hp.hc_magnitude != 2.0f) {
        *post = ggml_scale(ctx0, *post, 0.5f*hp.hc_magnitude);
    }
    *comb     = ggml_view_3d(ctx0, all, hc, hc, nt, hc*sizeof(float), hc*hc*sizeof(float), 2*hc*nt*sizeof(float));
    llm.cb(pre, "hc_pre", il);
    llm.cb(*post, "hc_post", il);
    llm.cb(*comb, "hc_comb", il);
    return llm.build_mhc_weighted_sum(x, pre, n_embd, hc);
}

static ggml_tensor * xing4_hc_post_block(
        llm_build_context & llm, ggml_tensor * x, ggml_tensor * residual,
        ggml_tensor * post, ggml_tensor * comb, int il) {
    auto & ctx0 = llm.ctx0;
    const int64_t hc = llm.hparams.dsv4_hc_mult;
    auto comb_t = ggml_cont(ctx0, ggml_permute(ctx0, comb, 1, 0, 2, 3));
    auto result = ggml_hc_post(ctx0, x, post, residual, comb_t);
    llm.cb(result, "hc_post_out", il);
    return result;
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

    const int n_main = hp.n_layer_main ? (int) hp.n_layer_main
                                       : (int) hp.n_layer - (int) hp.nextn_predict_layers;
    bool out_sliced = false;
    for (int il = 0; il < n_main; ++il) {
        auto & layer = model_.layers[il];
        ggml_tensor * residual = inpL;
        ggml_tensor * post = nullptr;
        ggml_tensor * comb = nullptr;

        auto cur = xing4_hc_pre_block(*this, residual, layer.hc_attn_fn, layer.hc_attn_scale, layer.hc_attn_base, &post, &comb, il);
        cur = build_deepseek2_layer_attention(gf, il, cur, KQ_mask, inp_pos, nullptr,
                kq_scale, attn_factor_scaled, false, false, pp_opt);
        if (il == n_main - 1 && inp_out_ids) {
            cur      = ggml_get_rows(ctx0, cur, inp_out_ids);
            residual = ggml_reshape_3d(ctx0,
                    ggml_get_rows(ctx0, ggml_reshape_2d(ctx0, residual, n_embd*hc, n_tokens), inp_out_ids),
                    n_embd, hc, n_outputs);
            post     = ggml_get_rows(ctx0, post, inp_out_ids);
            comb     = ggml_reshape_3d(ctx0,
                    ggml_get_rows(ctx0, ggml_reshape_2d(ctx0, comb, hc*hc, n_tokens), inp_out_ids),
                    hc, hc, n_outputs);
            out_sliced = true;
        }
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

    auto flat = ggml_reshape_2d(ctx0, inpL, n_embd * hc, inpL->ne[2]);
    if (!out_sliced && inp_out_ids) {
        flat = ggml_get_rows(ctx0, flat, inp_out_ids);
    }
    const int64_t n_out = flat->ne[1];
    auto sums = ggml_multi_add(ctx0, ggml_view_2d(ctx0, flat, n_embd, n_out, flat->nb[1], 0), hc);
    auto cur = ggml_scale(ctx0, sums, 1.0f / hc);
    cur = llm_build_norm(ctx0, cur, hp, model_.output_norm, nullptr, LLM_NORM_RMS, cb, -1);
    cb(cur, "result_norm", -1);
    cur = llm_build_lora_mm(lctx, ctx0, model_.output, cur);
    cb(cur, "result_output", -1);
    ggml_build_forward_expand(gf, cur);
    return gf;
}

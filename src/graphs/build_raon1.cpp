#include "../llama-build-context.h"
#include "../llama-model.h"
#include "../llama-context.h"
#include "../llama-delta-net.h"

#include <cmath>

// Raon-1 mini: 18x RDA + 7x NoPE gated MLA, 4-stream Helmert mHC, MoE/SiTU-GLU.

static ggml_tensor * raon1_build_hc_pre(
        llm_build_context & llm,
        ggml_tensor * x, ggml_tensor * fn, ggml_tensor * scale, ggml_tensor * base,
        ggml_tensor * helmert, ggml_tensor * identity, ggml_tensor * chart_base,
        ggml_tensor ** pre_out, ggml_tensor ** post_out, ggml_tensor ** comb_out, int il) {
    auto & ctx0    = llm.ctx0;
    auto & hparams = llm.hparams;
    auto & cb      = llm.cb;

    const int64_t n_embd = hparams.n_embd;
    const int64_t hc     = hparams.dsv4_hc_mult;
    const int64_t nm     = hc - 1;
    const int64_t nt     = x->ne[2];

    ggml_tensor * flat = ggml_reshape_2d(ctx0, x, n_embd * hc, nt);
    ggml_tensor * rms  = ggml_sqrt(ctx0, ggml_scale_bias(ctx0,
            ggml_mean(ctx0, ggml_sqr(ctx0, flat)), 1.0f, hparams.dsv4_hc_eps));
    ggml_tensor * mixes = ggml_div(ctx0, llm_build_context::llm_build_lora_mm(llm.lctx, ctx0, fn, flat), rms);
    cb(mixes, "hc_mixes", il);

    auto affine = [&](int64_t offset, int64_t width, int64_t si) {
        ggml_tensor * z = ggml_view_2d(ctx0, mixes, width, nt, mixes->nb[1], offset*mixes->nb[0]);
        ggml_tensor * a = ggml_view_1d(ctx0, scale, 1, si*scale->nb[0]);
        ggml_tensor * b = ggml_view_1d(ctx0, base, width, offset*base->nb[0]);
        return ggml_add(ctx0, ggml_mul(ctx0, z, a), b);
    };

    ggml_tensor * pre  = ggml_sigmoid(ctx0, affine(0, hc, 0));
    ggml_tensor * post = ggml_scale(ctx0, ggml_sigmoid(ctx0, affine(hc, hc, 1)), hparams.hc_magnitude);

    ggml_tensor * z = ggml_cont_3d(ctx0, affine(2*hc, nm*nm, 2), nm, nm, nt);
    ggml_tensor * e = ggml_mul_mat(ctx0, helmert, z);
    e = ggml_mul_mat(ctx0, helmert, ggml_cont(ctx0, ggml_transpose(ctx0, e)));
    ggml_tensor * ratio = ggml_neg(ctx0, ggml_div(ctx0, e, chart_base));
    ratio = ggml_reshape_2d(ctx0, ratio, hc*hc, nt);
    ggml_tensor * rho = ggml_pool_1d(ctx0, ratio, GGML_OP_POOL_MAX, hc*hc, hc*hc, 0);
    rho = ggml_scale_bias(ctx0, ggml_clamp(ctx0, rho, 0.0f, INFINITY), 1.0f, 1.0f);
    rho = ggml_reshape_3d(ctx0, rho, 1, 1, nt);
    ggml_tensor * phi = ggml_add(ctx0, ggml_div(ctx0, e, rho), chart_base);
    ggml_tensor * gamma = ggml_sigmoid(ctx0, ggml_view_1d(ctx0, scale, 1, 3*scale->nb[0]));
    ggml_tensor * comb = ggml_add(ctx0, ggml_mul(ctx0, ggml_sub(ctx0, phi, identity), gamma), identity);
    cb(comb, "hc_comb", il);
    cb(pre,  "hc_pre",  il);
    cb(post, "hc_post", il);

    // result[e,t] = sum_s pre[s,t] * x[e,s,t]
    ggml_tensor * pre3d = ggml_reshape_3d(ctx0, pre, hc, 1, nt);
    ggml_tensor * xh    = ggml_cont(ctx0, ggml_permute(ctx0, x, 1, 0, 2, 3)); // [hc, n_embd, nt]
    ggml_tensor * result = ggml_cont_2d(ctx0, ggml_mul_mat(ctx0, pre3d, xh), n_embd, nt);

    *pre_out  = pre;
    *post_out = post;
    *comb_out = comb;
    return result;
}

static ggml_tensor * raon1_build_hc_post(
        llm_build_context & llm,
        ggml_tensor * x, ggml_tensor * branch, ggml_tensor * post, ggml_tensor * comb, int il) {
    auto & ctx0    = llm.ctx0;
    auto & hparams = llm.hparams;
    auto & cb      = llm.cb;

    const int64_t n_embd = hparams.n_embd;
    const int64_t hc     = hparams.dsv4_hc_mult;
    const int64_t nt     = x->ne[2];

    // Batched hc x hc GEMM plus the post/branch rank-1 update.
    ggml_tensor * comb_t = ggml_cont(ctx0, ggml_permute(ctx0, comb, 1, 0, 2, 3)); // [src, dst, nt]
    ggml_tensor * xh     = ggml_cont(ctx0, ggml_permute(ctx0, x,    1, 0, 2, 3)); // [src, n_embd, nt]
    ggml_tensor * result = ggml_mul_mat(ctx0, comb_t, xh);                        // [dst, n_embd, nt]

    // post[dst,t] * branch[e,t] is a K=1 GEMM.
    ggml_tensor * post1   = ggml_reshape_3d(ctx0, post,   1, hc,     nt);
    ggml_tensor * branch1 = ggml_reshape_3d(ctx0, branch, 1, n_embd, nt);
    ggml_tensor * post_term = ggml_mul_mat(ctx0, post1, branch1);                 // [hc, n_embd, nt]
    result = ggml_add(ctx0, result, post_term);
    result = ggml_cont(ctx0, ggml_permute(ctx0, result, 1, 0, 2, 3));             // [n_embd, hc, nt]

    cb(result, "hc_post_out", il);
    return result;
}

static ggml_tensor * raon1_mla_attention(
        llm_build_context & llm, ggml_cgraph * gf, int il,
        ggml_tensor * inp, ggml_tensor * KQ_mask, float kq_scale) {
    auto & hparams = llm.hparams;
    const auto & layer = llm.model.layers[il];
    GGML_ASSERT(llm.n_head == hparams.ssm_dt_rank);
    mla_params p{layer};
    p.inp              = inp;
    p.gf               = gf;
    p.KQ_mask          = KQ_mask;
    p.output_gate      = layer.attn_v_gate;
    p.wkv_b            = layer.wkv_b;
    const size_t pp_bytes = layer.wkv_b ? (size_t) layer.wkv_b->ne[1] * llm.n_kv * sizeof(float) : 0;
    p.expanded_pp      = layer.wkv_b && (llm.lctx.cparams.attn_max_batch > 0 || pp_bytes <= 256ull*1024*1024) &&
                         llm.n_tokens >= 128 && llm.lctx.cparams.mla_attn > 1 && llm.lctx.cparams.flash_attn;
    p.il               = il;
    p.n_head           = llm.n_head;
    p.n_embd_head_k    = hparams.n_embd_head_k_mla();
    p.n_embd_head_v    = hparams.n_embd_head_v_mla();
    p.n_extra          = hparams.n_rot;
    p.kv_lora_rank     = hparams.n_lora_kv;
    p.kq_scale         = kq_scale;
    p.force_matmul_kq_f32 = true;
    p.fused_output_gate = false;
    return llm.build_mla_attention(p);
}

ggml_cgraph * llm_build_context::build_raon1() {
    ggml_cgraph * gf = new_graph_custom();

    const int64_t n_embd     = hparams.n_embd;
    const int64_t n_tokens   = this->n_tokens;
    const int64_t n_outputs  = this->n_outputs;
    const int64_t hc         = hparams.dsv4_hc_mult;
    const int64_t nm         = hc - 1;

    delta_net delta(lctx, batch);

    auto * inp_out_ids = (n_outputs != n_tokens) ? build_inp_out_ids() : nullptr;
    auto * KQ_mask = build_inp_KQ_mask();

    lctx.inp_s_seq_qnext = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, 1, n_tokens);
    cb(lctx.inp_s_seq_qnext, "inp_s_seq_qnext", -1);
    ggml_set_input(lctx.inp_s_seq_qnext);

    // fixed Helmert basis and chart for the mHC selector
    ggml_tensor * helmert = nullptr;
    for (int64_t i = 0; i < hc; ++i) {
        ggml_tensor * row = nullptr;
        for (int64_t j = 0; j < nm; ++j) {
            const float v = (i <= j ? 1.0f : i == j + 1 ? -(float) (j + 1) : 0.0f)/sqrtf((j + 1)*(j + 2));
            ggml_tensor * c = ggml_fill(ctx0, ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, 1), v);
            row = row ? ggml_concat(ctx0, row, c, 0) : c;
        }
        row = ggml_reshape_2d(ctx0, row, nm, 1);
        helmert = helmert ? ggml_concat(ctx0, helmert, row, 1) : row;
    }
    ggml_tensor * identity = ggml_diag(ctx0, ggml_fill(ctx0, ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, hc), 1.0f));
    ggml_tensor * chart_base = ggml_scale_bias(ctx0, identity, 0.5f, 0.5f/hc);

    ggml_tensor * inp = llm_build_inp_embd(ctx0, lctx, hparams, batch, model.tok_embd, cb);
    inp = ggml_reshape_3d(ctx0, inp, n_embd, 1, n_tokens);
    inp = ggml_repeat_4d(ctx0, inp, n_embd, hc, n_tokens, 1);

    ggml_tensor * inpL = inp;
    for (int il = 0; il < (int) hparams.n_layer; ++il) {
        const auto & layer = model.layers[il];

        ggml_tensor * pre = nullptr;
        ggml_tensor * post = nullptr;
        ggml_tensor * comb = nullptr;

        auto cur = raon1_build_hc_pre(*this, inpL, layer.hc_attn_fn, layer.hc_attn_scale, layer.hc_attn_base,
                helmert, identity, chart_base, &pre, &post, &comb, il);
        cb(cur, "l_in", il);

        if (hparams.is_recurrent(il)) {
            cur = delta.build_layer_attn_kda(ctx0, gf, cur, nullptr, il, cb, /*add_residual = */ false);
        } else {
            cur = raon1_mla_attention(*this, gf, il, cur, KQ_mask,
                    1.0f/sqrtf(float(hparams.n_embd_head_k_mla())));
        }
        inpL = raon1_build_hc_post(*this, inpL, cur, post, comb, il);
        inpL = lctx.cvec.apply_to(ctx0, inpL, il);

        cur = raon1_build_hc_pre(*this, inpL, layer.hc_ffn_fn, layer.hc_ffn_scale, layer.hc_ffn_base,
                helmert, identity, chart_base, &pre, &post, &comb, il);
        cur = llm_build_norm(ctx0, cur, hparams, layer.ffn_norm, nullptr, LLM_NORM_RMS, cb, il);

        if (il < (int) hparams.n_layer_dense_lead) {
            cur = llm_build_ffn(ctx0, lctx, nullptr, cur,
                    layer.ffn_up,   nullptr, nullptr,
                    layer.ffn_gate, nullptr, nullptr,
                    layer.ffn_down, nullptr, nullptr,
                    nullptr,
                    LLM_FFN_SITU, LLM_FFN_PAR, cb, il);
        } else {
            auto moe_out = llm_build_moe_ffn(ctx0, lctx, cur,
                    layer.ffn_gate_inp, nullptr,
                    layer.ffn_up_exps, nullptr,
                    layer.ffn_gate_exps, nullptr,
                    layer.ffn_down_exps, nullptr,
                    layer.ffn_exp_probs_b,
                    hparams.n_expert, hparams.n_expert_used,
                    LLM_FFN_SITU, hparams.expert_weights_norm,
                    true, hparams.expert_weights_scale,
                    (llm_expert_gating_func_type) hparams.expert_gating_func,
                    cb, il, gf, false, layer.ffn_up_gate_exps, nullptr, nullptr, nullptr, nullptr);
            ggml_build_forward_expand(gf, moe_out);

            auto ffn_shexp = llm_build_ffn(ctx0, lctx, nullptr, cur,
                    layer.ffn_up_shexp,   nullptr, nullptr,
                    layer.ffn_gate_shexp, nullptr, nullptr,
                    layer.ffn_down_shexp, nullptr, nullptr,
                    nullptr,
                    LLM_FFN_SITU, LLM_FFN_PAR, cb, il);
            cur = ggml_add(ctx0, moe_out, ffn_shexp);
            ggml_build_forward_expand(gf, cur);
        }

        inpL = raon1_build_hc_post(*this, inpL, cur, post, comb, il);
        inpL = lctx.cvec.apply_to(ctx0, inpL, il);
        auto l_out = ggml_cont_2d(ctx0, inpL, n_embd * hc, n_tokens);
        cb(l_out, "l_out", il);
        ggml_build_forward_expand(gf, l_out);
    }
    auto l_out = ggml_cont_2d(ctx0, inpL, n_embd * hc, n_tokens);
    cb(l_out, "l_out", hparams.n_layer);
    ggml_build_forward_expand(gf, l_out);

    // collapse the 4 mHC streams
    auto flat = ggml_reshape_2d(ctx0, inpL, n_embd * hc, n_tokens);
    if (inp_out_ids) {
        flat = ggml_get_rows(ctx0, flat, inp_out_ids);
    }
    const int64_t n_out = flat->ne[1];
    ggml_tensor * summed = nullptr;
    for (int64_t s = 0; s < hc; ++s) {
        auto stream = ggml_view_2d(ctx0, flat, n_embd, n_out,
                flat->nb[1], s * n_embd * ggml_element_size(flat));
        summed = summed ? ggml_add(ctx0, summed, stream) : stream;
    }
    inpL = ggml_scale(ctx0, summed, 1.0f / hc);

    auto cur = build_output(lctx, ctx0, inpL, model.output, model.output_norm, cb);
    cb(cur, "result_output", -1);
    ggml_build_forward_expand(gf, cur);
    return gf;
}

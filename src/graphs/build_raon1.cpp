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
    auto & lctx    = llm.lctx;
    auto & ctx0    = llm.ctx0;
    auto & hparams = llm.hparams;
    auto & model   = llm.model;
    auto & kv_self = lctx.kv_self;
    auto & cb      = llm.cb;
    auto & layer   = model.layers[il];

    const int64_t n_head      = llm.n_head;
    const int64_t n_head_kda  = hparams.ssm_dt_rank;
    const int64_t head_k      = hparams.n_embd_head_k_mla();
    const int64_t head_v      = hparams.n_embd_head_v_mla();
    const int64_t n_nope      = head_k - hparams.n_rot;
    const int64_t kv_lora     = hparams.n_lora_kv;
    const int64_t n_extra     = hparams.n_rot;
    const int64_t n_tokens    = llm.n_tokens;

    GGML_ASSERT(n_head == n_head_kda);

    auto cur = llm_build_context::llm_build_norm(ctx0, inp, hparams, layer.attn_norm, nullptr, LLM_NORM_RMS, cb, il);
    cb(cur, "attn_norm", il);

    auto qr = ggml_mul_mat(ctx0, layer.wq_a, cur);
    qr = llm_build_context::llm_build_norm(ctx0, qr, hparams, layer.attn_q_a_norm, nullptr, LLM_NORM_RMS, cb, il);

    auto q = ggml_mul_mat(ctx0, layer.wq_b, qr);
    q = ggml_reshape_3d(ctx0, q, head_k, n_head, n_tokens);
    q = ggml_permute(ctx0, q, 0, 2, 1, 3); // [head_k, n_tokens, n_head]

    auto q_nope = ggml_view_3d(ctx0, q, n_nope, n_tokens, n_head, q->nb[1], q->nb[2], 0);

    auto Qcur_nope = ggml_mul_mat(ctx0, layer.wk_b, q_nope); // [kv_lora, n_tokens, n_head]
    auto Qcur = Qcur_nope;
    if (n_extra > 0) {
        // Raon-1's 64 NoPE channels are real extra K/Q channels.
        auto q_extra = ggml_view_3d(ctx0, q, n_extra, n_tokens, n_head, q->nb[1], q->nb[2], n_nope*q->nb[0]);
        Qcur = ggml_concat(ctx0, q_extra, Qcur_nope, 0);      // [kv_lora + n_extra, n_tokens, n_head]
    }
    cb(Qcur, "Qcur", il);

    auto kv = ggml_mul_mat(ctx0, layer.wkv_a_mqa, cur);      // [kv_lora + n_extra, n_tokens]
    auto latent = ggml_view_2d(ctx0, kv, kv_lora, n_tokens, kv->nb[1], 0);
    latent = llm_build_context::llm_build_norm(ctx0, latent, hparams, layer.attn_kv_a_norm,
            nullptr, LLM_NORM_RMS, cb, il);

    auto k_row = latent;
    if (n_extra > 0) {
        auto extra = ggml_view_2d(ctx0, kv, n_extra, n_tokens, kv->nb[1], kv_lora*kv->nb[0]);
        k_row = ggml_concat(ctx0, extra, latent, 0);          // [kv_lora + n_extra, n_tokens]
    }

    const auto row_size = ggml_row_size(kv_self.k_l[il]->type, kv_lora + n_extra);
    auto k_cache_write = ggml_view_2d(ctx0, kv_self.k_l[il], kv_lora + n_extra, n_tokens,
            row_size, row_size * llm.kv_head);
    lctx.cache_copies[2 * il + 0].cpy  = ggml_cpy(ctx0, k_row, k_cache_write);
    lctx.cache_copies[2 * il + 0].step = row_size;
    ggml_build_forward_expand(gf, lctx.cache_copies[2 * il + 0].cpy);

    auto K_cache = ggml_view_2d(ctx0, kv_self.k_l[il], kv_lora + n_extra, llm.n_kv, row_size, 0);
    cb(K_cache, "kv_cache", il);

    ggml_tensor * kqv = nullptr;
    if (lctx.cparams.flash_attn) {
        auto V_cache = ggml_view_2d(ctx0, kv_self.k_l[il], kv_lora, llm.n_kv, row_size,
                ggml_row_size(K_cache->type, n_extra));
        kqv = ggml_flash_attn_ext(ctx0, Qcur, K_cache, V_cache, KQ_mask,
                kq_scale, hparams.f_max_alibi_bias, 0.f);
        if (K_cache->ne[1] < 256) {
            ggml_flash_attn_ext_set_prec(kqv, GGML_PREC_F32);
        }
        kqv = ggml_permute(ctx0, kqv, 0, 2, 1, 3); // [kv_lora, n_tokens, n_head]
    } else {
        auto kq = ggml_mul_mat(ctx0, K_cache, Qcur); // [n_kv, n_tokens, n_head]
        ggml_mul_mat_set_prec(kq, GGML_PREC_F32);
        kq = ggml_soft_max_ext(ctx0, kq, KQ_mask, kq_scale, hparams.f_max_alibi_bias);
        auto V_cache = ggml_view_2d(ctx0, kv_self.k_l[il], kv_lora, llm.n_kv, row_size,
                ggml_row_size(K_cache->type, n_extra));
        kqv = ggml_mul_mat(ctx0, ggml_cont(ctx0, ggml_transpose(ctx0, V_cache)), kq);
    }
    cb(kqv, "kqv_compressed", il);

    auto wv_b = layer.wv_b;
    if (wv_b->ne[1] != head_v) {
        wv_b = ggml_reshape_3d(ctx0, wv_b, kv_lora, head_v, n_head);
    }
    kqv = ggml_mul_mat(ctx0, wv_b, kqv); // [head_v, n_tokens, n_head]
    kqv = ggml_permute(ctx0, kqv, 0, 2, 1, 3);
    kqv = ggml_cont(ctx0, kqv);
    kqv = ggml_reshape_2d(ctx0, kqv, head_v * n_head, n_tokens);

    auto gate = ggml_mul_mat(ctx0, layer.attn_v_gate, cur);
    gate = ggml_reshape_2d(ctx0, gate, head_v * n_head, n_tokens);
    kqv = ggml_mul(ctx0, kqv, ggml_sigmoid(ctx0, gate));
    cb(kqv, "kqv_gated", il);

    cur = llm_build_context::llm_build_lora_mm(lctx, ctx0, layer.wo, kqv);
    cb(cur, "kqv_out", il);
    return cur;
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

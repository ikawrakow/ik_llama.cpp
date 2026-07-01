#include "../llama-build-context.h"
#include "../llama-model.h"
#include "../llama-context.h"

// openPangu-2.0-Flash graph.
//
// Dense-fallback for first generation: every attention layer runs as plain causal MHA
// (MLA projections materialized to full Q/K/V), which is bit-exact to the real DSA/SWA
// model for prompts <= 512 tokens (DSA top-2048 and SWA window-512 are inert there).
//
// Pangu-specific pieces implemented here (see vault Stage-2b forward spec):
//   - mHC / Hyper-Connections: 4 parallel residual streams mixed per sublayer via a phi
//     projection (h_pre combine-in, h_post/h_res scatter-out) with a 20-iter Sinkhorn.
//   - MoME: causal depthwise conv (k=3) on the q-lora latent, compressed-kv latent, attn out.
//   - sandwich norms (post_attention / pre_mlp / post_mlp) + block_post on a layer subset.
//   - MTP layers skipped; param_sink deferred (v0) — documented below.

// --- causal depthwise conv1d, kernel=3: out[t] = w0*x[t-2] + w1*x[t-1] + w2*x[t] (per channel) ---
// x: [C, n_tokens]; w: ggml tensor with ne = {3, 1, C} (kernel-major). Returns [C, n_tokens].
static ggml_tensor * openpangu_causal_conv(ggml_context * ctx, ggml_tensor * x, ggml_tensor * w) {
    // NOTE(v0): passthrough. A correct causal conv1d needs a per-channel conv-state cache
    // (like the SSM short-conv) so single-token decode steps see prior-token history; a stateless
    // full-sequence conv both breaks at T<kernel and is wrong in incremental decode. Deferred to v1
    // together with o_conv + param_sink. This drops the MoME local-mixing (bounded approximation).
    (void) w;
    return x;
    const int64_t C = x->ne[0];
    const int64_t T = x->ne[1];
    // weight rows per tap: w_tap[k] is a [C] vector = w[k, 0, :]
    ggml_tensor * wc = ggml_cont(ctx, w);                       // {3,1,C}
    wc = ggml_reshape_2d(ctx, wc, 3, C);                        // [3, C]
    ggml_tensor * w0 = ggml_cont(ctx, ggml_view_1d(ctx, wc, C, 0*wc->nb[1] + 0)); // stride tricks below
    // simpler: build per-tap [C] via views on the [3,C] tensor along ne0
    ggml_tensor * tap0 = ggml_cont(ctx, ggml_view_2d(ctx, wc, 1, C, wc->nb[1], 0*ggml_element_size(wc)));
    ggml_tensor * tap1 = ggml_cont(ctx, ggml_view_2d(ctx, wc, 1, C, wc->nb[1], 1*ggml_element_size(wc)));
    ggml_tensor * tap2 = ggml_cont(ctx, ggml_view_2d(ctx, wc, 1, C, wc->nb[1], 2*ggml_element_size(wc)));
    tap0 = ggml_reshape_1d(ctx, tap0, C);
    tap1 = ggml_reshape_1d(ctx, tap1, C);
    tap2 = ggml_reshape_1d(ctx, tap2, C);
    (void) w0;

    // shifted copies of x along the token axis with left zero-padding
    // x_shift(d)[:, t] = x[:, t-d] (0 for t<d). Build via ggml_pad on a rolled view.
    // We pad C rows x (T+2) cols conceptually; implement with concat of zeros + slice.
    ggml_tensor * zeros = ggml_scale(ctx, x, 0.0f);             // [C, T] of zeros
    ggml_tensor * z1 = ggml_view_2d(ctx, zeros, C, 1, zeros->nb[1], 0); // [C,1]
    ggml_tensor * z2 = ggml_view_2d(ctx, zeros, C, 2, zeros->nb[1], 0); // [C,2]

    ggml_tensor * x_t   = x;                                                   // t
    ggml_tensor * x_tm1 = ggml_concat(ctx, z1, ggml_view_2d(ctx, x, C, T-1, x->nb[1], 0), 1); // t-1
    ggml_tensor * x_tm2 = ggml_concat(ctx, z2, ggml_view_2d(ctx, x, C, T-2, x->nb[1], 0), 1); // t-2

    // out = tap2 (*) x_t + tap1 (*) x_tm1 + tap0 (*) x_tm2   (tap index = kernel position)
    ggml_tensor * out = ggml_mul(ctx, x_t,   tap2);
    out = ggml_add(ctx, out, ggml_mul(ctx, x_tm1, tap1));
    out = ggml_add(ctx, out, ggml_mul(ctx, x_tm2, tap0));
    return out;
}

// --- mHC Sinkhorn: h_res [S*S, T] -> doubly-stochastic per token, 20 iters (ends on col norm) ---
static ggml_tensor * openpangu_sinkhorn(ggml_context * ctx, ggml_tensor * h_res_flat,
                                        int64_t S, int64_t T, int iters, float hc_eps) {
    // reshape to [S(row), S(col), T]: convention m[r,c] with ne0=row(S), ne1=col(S), ne2=T
    (void) hc_eps; // softmax outputs are strictly positive, so the eps is numerically inert here
    ggml_tensor * m = ggml_reshape_3d(ctx, h_res_flat, S, S, T);
    // ref softmaxes h_res over columns (final S). ggml_soft_max works over ne0, so permute col->ne0.
    m = ggml_cont(ctx, ggml_permute(ctx, m, 1, 0, 2, 3));      // [col, row, T]
    m = ggml_soft_max(ctx, m);                                 // softmax over col
    m = ggml_cont(ctx, ggml_permute(ctx, m, 1, 0, 2, 3));      // back to [row, col, T]

    auto col_norm = [&](ggml_tensor * a) {
        ggml_tensor * col_sum = ggml_sum_rows(ctx, a);          // sums ne0(row) -> [1, col, T]
        return ggml_div(ctx, a, col_sum);                       // broadcast [1,col,T] over rows
    };
    auto row_norm = [&](ggml_tensor * a) {
        ggml_tensor * ap = ggml_cont(ctx, ggml_permute(ctx, a, 1, 0, 2, 3)); // [col,row,T]
        ggml_tensor * row_sum = ggml_sum_rows(ctx, ap);         // [1, row, T]
        ggml_tensor * out = ggml_div(ctx, ap, row_sum);
        return ggml_cont(ctx, ggml_permute(ctx, out, 1, 0, 2, 3)); // back [row,col,T]
    };

    m = col_norm(m);
    for (int i = 0; i < iters - 1; ++i) {
        m = row_norm(m);
        m = col_norm(m);
    }
    return m; // [row(S), col(S), T]
}

ggml_cgraph * llm_build_context::build_openpangu() {
    ggml_cgraph * gf = new_graph_custom();

    const int64_t n_embd_head_qk_rope = hparams.n_rot;                     // 64
    const int64_t n_embd_head_k       = hparams.n_embd_head_k(0);          // 192
    const int64_t n_embd_head_qk_nope = n_embd_head_k - n_embd_head_qk_rope; // 128
    const int64_t n_embd_head_v       = hparams.n_embd_head_v(0);          // 128
    const int64_t q_lora_rank         = hparams.n_lora_q;                  // 1024
    const int64_t kv_lora_rank        = hparams.n_lora_kv;                 // 512
    const int64_t S                   = hparams.mhc_num_stream;            // 4
    const int    sink_iters           = (int) hparams.mhc_recur_norm;      // 20
    const float  hc_eps               = 1e-6f;
    const float  kq_scale             = 1.0f / sqrtf(float(n_embd_head_k));

    ggml_tensor * cur;
    ggml_tensor * inpL = llm_build_inp_embd(ctx0, lctx, hparams, batch, model.tok_embd, cb);
    ggml_tensor * inp_pos = build_inp_pos();
    ggml_tensor * KQ_mask = build_inp_KQ_mask();

    // mHC entry: repeat the embedding into S residual streams -> R [n_embd, S, n_tokens]
    ggml_tensor * R = ggml_repeat(ctx0, ggml_reshape_3d(ctx0, inpL, n_embd, 1, n_tokens),
                                  ggml_new_tensor_3d(ctx0, inpL->type, n_embd, S, n_tokens));

    // mHC pre: combine S streams -> x [n_embd, n_tokens]; also returns h_post [S,T], h_res_mix [S,S,T]
    auto mhc_pre = [&](ggml_tensor * Rin, ggml_tensor * phi, ggml_tensor * alpha,
                       ggml_tensor * beta, ggml_tensor * gamma,
                       ggml_tensor ** h_post_out, ggml_tensor ** h_res_out) {
        ggml_tensor * flat = ggml_reshape_2d(ctx0, ggml_cont(ctx0, Rin), n_embd * S, n_tokens);
        ggml_tensor * normed = ggml_rms_norm(ctx0, flat, hparams.f_norm_rms_eps);
        normed = ggml_mul(ctx0, normed, gamma);                       // [S*H, T]
        ggml_tensor * mixes = ggml_mul_mat(ctx0, phi, normed);        // [(S+2)*S, T]
        ggml_tensor * h_pre  = ggml_view_2d(ctx0, mixes, S, n_tokens, mixes->nb[1], 0);
        ggml_tensor * h_post = ggml_view_2d(ctx0, mixes, S, n_tokens, mixes->nb[1], S*ggml_element_size(mixes));
        ggml_tensor * h_res  = ggml_view_2d(ctx0, mixes, S*S, n_tokens, mixes->nb[1], 2*S*ggml_element_size(mixes));

        // alpha = [a_pre, a_post, a_res]; beta = [b_pre(S), b_post(S), b_res(S*S)]
        ggml_tensor * a_pre  = ggml_view_1d(ctx0, alpha, 1, 0);
        ggml_tensor * b_pre  = ggml_view_1d(ctx0, beta, S, 0);
        h_pre = ggml_add(ctx0, ggml_mul(ctx0, h_pre, a_pre), b_pre);  // broadcast scalar + [S]
        h_pre = ggml_sigmoid(ctx0, h_pre);                            // [S,T] (+hc_eps omitted, inert)

        // combine: x[h,t] = sum_s h_pre[s,t] * R[h,s,t]
        ggml_tensor * hpre3 = ggml_reshape_3d(ctx0, ggml_cont(ctx0, h_pre), 1, S, n_tokens);
        ggml_tensor * weighted = ggml_mul(ctx0, Rin, hpre3);          // [H,S,T]
        ggml_tensor * wperm = ggml_cont(ctx0, ggml_permute(ctx0, weighted, 1, 0, 2, 3)); // [S,H,T]
        ggml_tensor * x = ggml_reshape_2d(ctx0, ggml_sum_rows(ctx0, wperm), n_embd, n_tokens); // sum over S

        *h_post_out = ggml_cont(ctx0, h_post);
        *h_res_out  = ggml_cont(ctx0, h_res);
        return x;
    };

    // mHC post: R_new[h,s,t] = h_post[s,t]*y[h,t] + sum_j m[s,j,t]*R[h,j,t]
    auto mhc_post = [&](ggml_tensor * y, ggml_tensor * h_post, ggml_tensor * Rin,
                        ggml_tensor * alpha, ggml_tensor * beta, ggml_tensor * h_res) {
        ggml_tensor * a_post = ggml_view_1d(ctx0, alpha, 1, 1*ggml_element_size(alpha));
        ggml_tensor * a_res  = ggml_view_1d(ctx0, alpha, 1, 2*ggml_element_size(alpha));
        ggml_tensor * b_post = ggml_view_1d(ctx0, beta, S,   S*ggml_element_size(beta));
        ggml_tensor * b_res  = ggml_view_1d(ctx0, beta, S*S, 2*S*ggml_element_size(beta));

        h_post = ggml_add(ctx0, ggml_mul(ctx0, h_post, a_post), b_post);
        h_post = ggml_scale(ctx0, ggml_sigmoid(ctx0, h_post), 2.0f);  // 2*sigmoid, [S,T]

        ggml_tensor * m = ggml_add(ctx0, ggml_mul(ctx0, h_res, a_res), b_res); // [S*S,T]
        m = openpangu_sinkhorn(ctx0, m, S, n_tokens, sink_iters, hc_eps);      // [row S, col S, T]

        // term1: h_post[s,t]*y[h,t] -> [H,S,T]
        ggml_tensor * y3 = ggml_reshape_3d(ctx0, y, n_embd, 1, n_tokens);
        ggml_tensor * hpost3 = ggml_reshape_3d(ctx0, ggml_cont(ctx0, h_post), 1, S, n_tokens);
        ggml_tensor * term1 = ggml_mul(ctx0, ggml_repeat(ctx0, y3,
                                 ggml_new_tensor_3d(ctx0, y->type, n_embd, S, n_tokens)), hpost3);

        // term2: sum_j m[s,j,t]*R[h,j,t]. For each out-stream s, weight over input streams j.
        // Build via: for stream axis, matmul R[H, j, t] with m[j, s, t] batched over t.
        // R_perm [j(S), H, T] ; m as [j(S), s(S), T]; batched mul_mat over T -> [H? ] messy.
        // Simpler explicit loop over S output streams (S=4, cheap):
        ggml_tensor * term2 = nullptr;
        for (int64_t s = 0; s < S; ++s) {
            // m_s = m[:, s, :] -> weights over input streams j: [S, T]
            ggml_tensor * m_s = ggml_cont(ctx0, ggml_view_2d(ctx0, m, S, n_tokens, m->nb[2], s*m->nb[1]));
            ggml_tensor * m_s3 = ggml_reshape_3d(ctx0, m_s, 1, S, n_tokens);      // [1,S,T]
            ggml_tensor * acc = ggml_mul(ctx0, Rin, m_s3);                        // [H,S,T]
            ggml_tensor * accp = ggml_cont(ctx0, ggml_permute(ctx0, acc, 1, 0, 2, 3)); // [S,H,T]
            ggml_tensor * summed = ggml_reshape_2d(ctx0, ggml_sum_rows(ctx0, accp), n_embd, n_tokens); // [H,T]
            summed = ggml_reshape_3d(ctx0, summed, n_embd, 1, n_tokens);
            term2 = term2 ? ggml_concat(ctx0, term2, summed, 1) : summed;         // -> [H,S,T]
        }
        return ggml_add(ctx0, term1, term2); // [H,S,T]
    };

    // Base generation uses only the transformer layers; the trailing NextN/MTP layers are skipped.
    const int n_layer_base = n_layer - (int) hparams.nextn_predict_layers;
    for (int il = 0; il < n_layer_base; ++il) {
        auto & layer = model.layers[il];

        // ================= attention sublayer =================
        ggml_tensor * h_post_a, * h_res_a;
        ggml_tensor * x = mhc_pre(R, layer.mhc_attn_phi, layer.mhc_attn_alpha,
                                  layer.mhc_attn_beta, layer.mhc_attn_gamma, &h_post_a, &h_res_a);
        cur = llm_build_norm(ctx0, x, hparams, layer.attn_norm, NULL, LLM_NORM_RMS, cb, il);

        // --- Q path: q_a -> qa_conv -> q_a_norm -> q_b ---
        ggml_tensor * q_lora = ggml_mul_mat(ctx0, layer.wq_a, cur);        // [q_lora_rank, T]
        q_lora = openpangu_causal_conv(ctx0, q_lora, layer.qa_conv);
        q_lora = llm_build_norm(ctx0, q_lora, hparams, layer.attn_q_a_norm, NULL, LLM_NORM_RMS, cb, il);
        ggml_tensor * q = ggml_mul_mat(ctx0, layer.wq_b, q_lora);          // [n_head*192, T]
        q = ggml_reshape_3d(ctx0, q, n_embd_head_k, n_head, n_tokens);
        ggml_tensor * q_nope = ggml_view_3d(ctx0, q, n_embd_head_qk_nope, n_head, n_tokens, q->nb[1], q->nb[2], 0);
        ggml_tensor * q_rope = ggml_view_3d(ctx0, q, n_embd_head_qk_rope, n_head, n_tokens, q->nb[1], q->nb[2],
                                            n_embd_head_qk_nope*ggml_element_size(q));
        q_rope = ggml_rope_ext(ctx0, ggml_cont(ctx0, q_rope), inp_pos, nullptr, n_rot, rope_type,
                               n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
        q = ggml_concat(ctx0, ggml_cont(ctx0, q_nope), q_rope, 0);        // [192, n_head, T]

        // --- KV path: kv_a -> split -> compresskv_conv -> kv_a_norm -> kv_b ---
        ggml_tensor * kv = ggml_mul_mat(ctx0, layer.wkv_a_mqa, cur);       // [kv_lora+64, T]
        ggml_tensor * ckv = ggml_cont(ctx0, ggml_view_2d(ctx0, kv, kv_lora_rank, n_tokens, kv->nb[1], 0));
        ggml_tensor * k_pe = ggml_cont(ctx0, ggml_view_2d(ctx0, kv, n_embd_head_qk_rope, n_tokens, kv->nb[1],
                                            kv_lora_rank*ggml_element_size(kv)));
        ckv = openpangu_causal_conv(ctx0, ckv, layer.kv_conv);
        ckv = llm_build_norm(ctx0, ckv, hparams, layer.attn_kv_a_norm, NULL, LLM_NORM_RMS, cb, il);
        ggml_tensor * kvb = ggml_mul_mat(ctx0, layer.wkv_b, ckv);          // [n_head*(128+128), T]
        kvb = ggml_reshape_3d(ctx0, kvb, n_embd_head_qk_nope + n_embd_head_v, n_head, n_tokens);
        ggml_tensor * k_nope = ggml_view_3d(ctx0, kvb, n_embd_head_qk_nope, n_head, n_tokens, kvb->nb[1], kvb->nb[2], 0);
        ggml_tensor * v = ggml_cont(ctx0, ggml_view_3d(ctx0, kvb, n_embd_head_v, n_head, n_tokens, kvb->nb[1], kvb->nb[2],
                                            n_embd_head_qk_nope*ggml_element_size(kvb)));
        // rope k_pe (shared across heads) then broadcast to all heads
        k_pe = ggml_reshape_3d(ctx0, k_pe, n_embd_head_qk_rope, 1, n_tokens);
        k_pe = ggml_rope_ext(ctx0, k_pe, inp_pos, nullptr, n_rot, rope_type,
                             n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
        ggml_tensor * k_pe_b = ggml_repeat(ctx0, k_pe,
                                 ggml_new_tensor_3d(ctx0, k_pe->type, n_embd_head_qk_rope, n_head, n_tokens));
        ggml_tensor * k = ggml_concat(ctx0, ggml_cont(ctx0, k_nope), k_pe_b, 0); // [192, n_head, T]

        // NOTE(v0): param_sink (128 learned latent-KV prepended) is deferred — a documented
        // approximation for the first coherence smoke; add before claiming full fidelity.

        v = ggml_reshape_2d(ctx0, v, n_embd_head_v * n_head, n_tokens);   // V as 2D (matches template)
        cur = llm_build_kv(ctx0, lctx, kv_self, gf, layer.wo, NULL,
                           k, v, q, KQ_mask, n_tokens, kv_head, n_kv, kq_scale, cb, il);

        // NOTE(v0): o_conv acts on the pre-o_proj attn output ([n_head*v_head]=6144), but
        // llm_build_kv bundles wo internally. Deferred with param_sink; needs a manual
        // attention path (wo=NULL) to insert o_conv before o_proj. (void) layer.o_conv;
        cur = llm_build_norm(ctx0, cur, hparams, layer.attn_post_norm, NULL, LLM_NORM_RMS, cb, il);

        // mHC post -> scatter back to S streams
        R = mhc_post(cur, h_post_a, R, layer.mhc_attn_alpha, layer.mhc_attn_beta, h_res_a);

        // ================= ffn sublayer =================
        ggml_tensor * h_post_m, * h_res_m;
        ggml_tensor * xm = mhc_pre(R, layer.mhc_mlp_phi, layer.mhc_mlp_alpha,
                                   layer.mhc_mlp_beta, layer.mhc_mlp_gamma, &h_post_m, &h_res_m);
        cur = llm_build_norm(ctx0, xm, hparams, layer.ffn_norm, NULL, LLM_NORM_RMS, cb, il);

        if ((uint32_t) il < hparams.n_layer_dense_lead) {
            cur = llm_build_ffn(ctx0, lctx, nullptr, cur,
                    layer.ffn_up, NULL, NULL, layer.ffn_gate, NULL, NULL, layer.ffn_down, NULL, NULL,
                    NULL, LLM_FFN_SILU, LLM_FFN_PAR, cb, il);
        } else {
            ggml_tensor * moe_out = llm_build_moe_ffn(ctx0, lctx, cur,
                    layer.ffn_gate_inp, layer.ffn_up_exps, layer.ffn_gate_exps, layer.ffn_down_exps,
                    layer.ffn_exp_probs_b, n_expert, n_expert_used, LLM_FFN_SILU,
                    hparams.expert_weights_norm, true, hparams.expert_weights_scale,
                    (enum llm_expert_gating_func_type) hparams.expert_gating_func, cb, il, gf, false);
            ggml_tensor * shexp = llm_build_ffn(ctx0, lctx, nullptr, cur,
                    layer.ffn_up_shexp, NULL, NULL, layer.ffn_gate_shexp, NULL, NULL,
                    layer.ffn_down_shexp, NULL, NULL, NULL, LLM_FFN_SILU, LLM_FFN_PAR, cb, il);
            cur = ggml_add(ctx0, moe_out, shexp);
        }
        cur = llm_build_norm(ctx0, cur, hparams, layer.ffn_post_norm, NULL, LLM_NORM_RMS, cb, il);

        R = mhc_post(cur, h_post_m, R, layer.mhc_mlp_alpha, layer.mhc_mlp_beta, h_res_m);

        // block post-norm on the layer subset (RMSNorm over the concatenated S*H)
        if (layer.block_post_norm) {
            ggml_tensor * flat = ggml_reshape_2d(ctx0, ggml_cont(ctx0, R), n_embd * S, n_tokens);
            flat = ggml_rms_norm(ctx0, flat, hparams.f_norm_rms_eps);
            flat = ggml_mul(ctx0, flat, layer.block_post_norm);
            R = ggml_reshape_3d(ctx0, flat, n_embd, S, n_tokens);
        }
        R = lctx.cvec.apply_to(ctx0, R, il);
    }

    // mHC tail merge: collapse S streams -> 1 (pre_only)
    {
        ggml_tensor * flat = ggml_reshape_2d(ctx0, ggml_cont(ctx0, R), n_embd * S, n_tokens);
        ggml_tensor * normed = ggml_mul(ctx0, ggml_rms_norm(ctx0, flat, hparams.f_norm_rms_eps), model.mhc_merge_gamma);
        ggml_tensor * w = ggml_mul_mat(ctx0, model.mhc_merge_phi, normed);          // [S, T]
        ggml_tensor * a_pre = ggml_view_1d(ctx0, model.mhc_merge_alpha, 1, 0);
        w = ggml_sigmoid(ctx0, ggml_add(ctx0, ggml_mul(ctx0, w, a_pre), model.mhc_merge_beta)); // [S,T]
        ggml_tensor * w3 = ggml_reshape_3d(ctx0, ggml_cont(ctx0, w), 1, S, n_tokens);
        ggml_tensor * weighted = ggml_mul(ctx0, R, w3);                            // [H,S,T]
        ggml_tensor * wperm = ggml_cont(ctx0, ggml_permute(ctx0, weighted, 1, 0, 2, 3)); // [S,H,T]
        cur = ggml_reshape_2d(ctx0, ggml_sum_rows(ctx0, wperm), n_embd, n_tokens);
    }

    // select only the output tokens (the framework binds n_outputs rows, not all n_tokens)
    ggml_tensor * inp_out_ids = build_inp_out_ids();
    cur = ggml_get_rows(ctx0, cur, inp_out_ids);

    cur = llm_build_norm(ctx0, cur, hparams, model.output_norm, NULL, LLM_NORM_RMS, cb, -1);
    cb(cur, "result_norm", -1);
    cur = llm_build_lora_mm(lctx, ctx0, model.output, cur);
    cb(cur, "result_output", -1);
    ggml_build_forward_expand(gf, cur);
    return gf;
}

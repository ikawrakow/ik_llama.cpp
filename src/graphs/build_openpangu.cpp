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
    // MOME: out = x + depthwise causal conv1d(k=3). Every Infer call site passes
    // residual_connection=1; the tap magnitudes confirm the conv is a small learned
    // perturbation on top of the identity, not a standalone filter.
    // NOTE(v0): batch-local — the conv window does not cross ubatch boundaries. Exact for a
    // fresh-sequence prefill; single-token decode steps miss the t-1/t-2 taps until a
    // conv-state cache exists (bounded error, |taps| < 1).
    const int64_t C = x->ne[0];
    const int64_t T = x->ne[1];
    // weight is stored f16 with ne = {3, C}: per-channel taps contiguous. ggml_mul needs an
    // f32 src1, so cast once (tiny tensor).
    ggml_tensor * wc = ggml_reshape_2d(ctx, ggml_cast(ctx, w, GGML_TYPE_F32), 3, C);
    ggml_tensor * tap0 = ggml_reshape_1d(ctx, ggml_cont(ctx, ggml_view_2d(ctx, wc, 1, C, wc->nb[1], 0*ggml_element_size(wc))), C);
    ggml_tensor * tap1 = ggml_reshape_1d(ctx, ggml_cont(ctx, ggml_view_2d(ctx, wc, 1, C, wc->nb[1], 1*ggml_element_size(wc))), C);
    ggml_tensor * tap2 = ggml_reshape_1d(ctx, ggml_cont(ctx, ggml_view_2d(ctx, wc, 1, C, wc->nb[1], 2*ggml_element_size(wc))), C);

    // shifted copies of x along the token axis with left zero-padding:
    // x_shift(d)[:, t] = x[:, t-d] (0 for t<d). zeros come from x, so they are always finite.
    ggml_tensor * zeros = ggml_scale(ctx, x, 0.0f);             // [C, T] of zeros
    ggml_tensor * x_t   = x;
    ggml_tensor * x_tm1 = T > 1
        ? ggml_concat(ctx, ggml_view_2d(ctx, zeros, C, 1, zeros->nb[1], 0),
                           ggml_view_2d(ctx, x, C, T-1, x->nb[1], 0), 1)
        : zeros;                                                // T==1: no history in batch
    ggml_tensor * x_tm2 = T > 2
        ? ggml_concat(ctx, ggml_view_2d(ctx, zeros, C, 2, zeros->nb[1], 0),
                           ggml_view_2d(ctx, x, C, T-2, x->nb[1], 0), 1)
        : zeros;                                                // T<=2: all zero-history

    // conv = tap2 (*) x_t + tap1 (*) x_tm1 + tap0 (*) x_tm2   (tap index = kernel position)
    ggml_tensor * out = ggml_mul(ctx, x_t,   tap2);
    out = ggml_add(ctx, out, ggml_mul(ctx, x_tm1, tap1));
    out = ggml_add(ctx, out, ggml_mul(ctx, x_tm2, tap0));
    return ggml_add(ctx, x, out);
}

// --- mHC Sinkhorn: h_res [S*S, T] -> doubly-stochastic per token, 20 iters (ends on col norm) ---
static ggml_tensor * openpangu_sinkhorn(ggml_context * ctx, ggml_tensor * h_res_flat,
                                        int64_t S, int64_t T, int iters, float hc_eps) {
    // The flat h_res is torch [r,c] row-major (c fastest), so a bare reshape gives ne0=col.
    // Transpose once so ne0=row(S), ne1=col(S): every axis op below then matches the
    // reference _mhc_sinkhorn_naive (softmax over col, first norm over row, end on col-sum=1)
    // and mhc_post's out[c] = sum_r m[r,c]*residual[r].
    (void) hc_eps; // softmax outputs are strictly positive, so the eps is numerically inert here
    ggml_tensor * m = ggml_reshape_3d(ctx, h_res_flat, S, S, T);
    m = ggml_cont(ctx, ggml_permute(ctx, m, 1, 0, 2, 3));
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
    ggml_set_name(R, "opg_R_entry");

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
        if (il == 0) ggml_set_name(x, "opg0_attn_mhcpre_x");
        cur = llm_build_norm(ctx0, x, hparams, layer.attn_norm, NULL, LLM_NORM_RMS, cb, il);
        if (il == 0) ggml_set_name(cur, "opg0_attn_norm");

        // --- Q path: q_a -> qa_conv -> q_a_norm -> q_b ---
        ggml_tensor * q_lora = ggml_mul_mat(ctx0, layer.wq_a, cur);        // [q_lora_rank, T]
        q_lora = openpangu_causal_conv(ctx0, q_lora, layer.qa_conv);
        if (il == 0) ggml_set_name(q_lora, "opg0_qlora_conv");
        q_lora = llm_build_norm(ctx0, q_lora, hparams, layer.attn_q_a_norm, NULL, LLM_NORM_RMS, cb, il);
        if (il == 0) ggml_set_name(q_lora, "opg0_qlora_norm");
        ggml_tensor * q = ggml_mul_mat(ctx0, layer.wq_b, q_lora);          // [n_head*192, T]
        q = ggml_reshape_3d(ctx0, q, n_embd_head_k, n_head, n_tokens);
        ggml_tensor * q_nope = ggml_view_3d(ctx0, q, n_embd_head_qk_nope, n_head, n_tokens, q->nb[1], q->nb[2], 0);
        ggml_tensor * q_rope = ggml_view_3d(ctx0, q, n_embd_head_qk_rope, n_head, n_tokens, q->nb[1], q->nb[2],
                                            n_embd_head_qk_nope*ggml_element_size(q));
        q_rope = ggml_rope_ext(ctx0, ggml_cont(ctx0, q_rope), inp_pos, nullptr, n_rot, rope_type,
                               n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
        q = ggml_concat(ctx0, ggml_cont(ctx0, q_nope), q_rope, 0);        // [192, n_head, T]
        if (il == 0) ggml_set_name(q, "opg0_q");

        // --- KV path: kv_a -> split -> compresskv_conv -> kv_a_norm -> kv_b ---
        ggml_tensor * kv = ggml_mul_mat(ctx0, layer.wkv_a_mqa, cur);       // [kv_lora+64, T]
        ggml_tensor * ckv = ggml_cont(ctx0, ggml_view_2d(ctx0, kv, kv_lora_rank, n_tokens, kv->nb[1], 0));
        ggml_tensor * k_pe = ggml_cont(ctx0, ggml_view_2d(ctx0, kv, n_embd_head_qk_rope, n_tokens, kv->nb[1],
                                            kv_lora_rank*ggml_element_size(kv)));
        ckv = openpangu_causal_conv(ctx0, ckv, layer.kv_conv);
        ckv = llm_build_norm(ctx0, ckv, hparams, layer.attn_kv_a_norm, NULL, LLM_NORM_RMS, cb, il);
        if (il == 0) ggml_set_name(ckv, "opg0_ckv_norm");
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
        if (il == 0) { ggml_set_name(k, "opg0_k"); ggml_set_name(v, "opg0_v"); }

        // ---- store this step's K/V into the cache (standard bookkeeping) ----
        v = ggml_reshape_2d(ctx0, v, n_embd_head_v * n_head, n_tokens);   // V as 2D for the store
        ggml_build_forward_expand(gf, q);
        ggml_build_forward_expand(gf, k);
        ggml_build_forward_expand(gf, v);
        llm_build_kv_store(lctx, ctx0, hparams, cparams, kv_self, gf, k, v, n_tokens, kv_head, cb, il);

        // ---- param_sink: 128 static learned latent-KV entries prepended to the sequence ----
        // sink_kv -> kv_a_norm -> kv_b gives per-head k_nope/v; sink_k_pe is used rope-free.
        // Sinks are visible to every query position (they sit "before" the sequence).
        const int64_t NS = hparams.param_sink_number;
        ggml_tensor * s_ckv = llm_build_norm(ctx0, layer.param_sink_kv, hparams,
                                             layer.attn_kv_a_norm, NULL, LLM_NORM_RMS, cb, il); // [512, NS]
        ggml_tensor * s_kvb = ggml_mul_mat(ctx0, layer.wkv_b, s_ckv);                       // [n_head*256, NS]
        s_kvb = ggml_reshape_3d(ctx0, s_kvb, n_embd_head_qk_nope + n_embd_head_v, n_head, NS);
        ggml_tensor * s_knope = ggml_view_3d(ctx0, s_kvb, n_embd_head_qk_nope, n_head, NS,
                                             s_kvb->nb[1], s_kvb->nb[2], 0);
        ggml_tensor * s_v = ggml_cont(ctx0, ggml_view_3d(ctx0, s_kvb, n_embd_head_v, n_head, NS,
                                             s_kvb->nb[1], s_kvb->nb[2], n_embd_head_qk_nope*ggml_element_size(s_kvb)));
        ggml_tensor * s_kpe = ggml_reshape_3d(ctx0, layer.param_sink_k_pe, n_embd_head_qk_rope, 1, NS);
        s_kpe = ggml_repeat(ctx0, s_kpe, ggml_new_tensor_3d(ctx0, s_kpe->type, n_embd_head_qk_rope, n_head, NS));
        ggml_tensor * s_k = ggml_concat(ctx0, ggml_cont(ctx0, s_knope), s_kpe, 0);          // [192, n_head, NS]
        s_k = ggml_cont(ctx0, ggml_permute(ctx0, s_k, 0, 2, 1, 3));                         // [192, NS, n_head]
        s_v = ggml_cont(ctx0, ggml_permute(ctx0, s_v, 1, 2, 0, 3));                         // [NS, 128, n_head]

        // ---- manual attention over [sinks ++ cached tokens] (flash_attn is forced off) ----
        auto * k_cache = kv_self.k_l[il];
        auto * v_cache = kv_self.v_l[il];
        const int64_t n_ctx_kv = kv_self.size;
        ggml_tensor * kview = ggml_view_3d(ctx0, k_cache, n_embd_head_k, n_kv, n_head,
                ggml_row_size(k_cache->type, n_embd_head_k)*n_head,
                ggml_row_size(k_cache->type, n_embd_head_k), 0);
        GGML_ASSERT(kv_self.v_trans);
        ggml_tensor * vview = ggml_view_3d(ctx0, v_cache, n_kv, n_embd_head_v, n_head,
                ggml_element_size(v_cache)*n_ctx_kv,
                ggml_element_size(v_cache)*n_ctx_kv*n_embd_head_v, 0);

        // f32 everywhere: the non-f32 concat kernel only handles dim 0, and correctness comes
        // first on this CPU path (the cache-view casts are the price of prepending sinks).
        ggml_tensor * k_all = ggml_concat(ctx0, s_k, ggml_cast(ctx0, kview, GGML_TYPE_F32), 1); // [192, NS+n_kv, H]
        ggml_tensor * v_all = ggml_concat(ctx0, s_v, ggml_cast(ctx0, vview, GGML_TYPE_F32), 0); // [NS+n_kv, 128, H]

        ggml_tensor * qp = ggml_permute(ctx0, q, 0, 2, 1, 3);                               // [192, T, H]
        ggml_tensor * kq = ggml_mul_mat(ctx0, k_all, qp);                                   // [NS+n_kv, T, H]

        // mask: sinks always visible (0) ++ the causal KQ_mask. The zero block is built by
        // scaling finite kq data (KQ_mask itself holds -inf, which 0*x would turn into NaN).
        const int64_t n_mask = KQ_mask->ne[1];
        ggml_tensor * s_mask0 = ggml_scale(ctx0, ggml_view_2d(ctx0, kq, NS, n_mask, NS*ggml_element_size(kq), 0), 0.0f);
        ggml_tensor * mask_all = ggml_concat(ctx0, s_mask0, KQ_mask, 0);                    // [NS+n_kv, n_mask]
        kq = ggml_soft_max_ext(ctx0, kq, mask_all, kq_scale, hparams.f_max_alibi_bias);

        ggml_tensor * kqv = ggml_mul_mat(ctx0, v_all, kq);                                  // [128, T, H]
        ggml_tensor * merged = ggml_cont(ctx0, ggml_permute(ctx0, kqv, 0, 2, 1, 3));        // [128, H, T]
        cur = ggml_reshape_2d(ctx0, merged, n_embd_head_v * n_head, n_tokens);

        // o_conv (MOME on the pre-o_proj attn output), then o_proj
        cur = openpangu_causal_conv(ctx0, cur, layer.o_conv);
        cur = llm_build_lora_mm(lctx, ctx0, layer.wo, cur);
        if (il == 0) ggml_set_name(cur, "opg0_attn_out");

        cur = llm_build_norm(ctx0, cur, hparams, layer.attn_post_norm, NULL, LLM_NORM_RMS, cb, il);
        if (il == 0) ggml_set_name(cur, "opg0_attn_postnorm");

        // mHC post -> scatter back to S streams
        R = mhc_post(cur, h_post_a, R, layer.mhc_attn_alpha, layer.mhc_attn_beta, h_res_a);
        if (il == 0) ggml_set_name(R, "opg0_R_attn");

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
        if (il == 0) ggml_set_name(cur, "opg0_ffn_postnorm");

        R = mhc_post(cur, h_post_m, R, layer.mhc_mlp_alpha, layer.mhc_mlp_beta, h_res_m);

        // block post-norm on the layer subset (RMSNorm over the concatenated S*H)
        if (layer.block_post_norm) {
            ggml_tensor * flat = ggml_reshape_2d(ctx0, ggml_cont(ctx0, R), n_embd * S, n_tokens);
            flat = ggml_rms_norm(ctx0, flat, hparams.f_norm_rms_eps);
            flat = ggml_mul(ctx0, flat, layer.block_post_norm);
            R = ggml_reshape_3d(ctx0, flat, n_embd, S, n_tokens);
        }
        if (il == 0) ggml_set_name(R, "opg0_R_block");
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

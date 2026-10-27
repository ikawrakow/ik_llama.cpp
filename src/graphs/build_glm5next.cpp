#include "../llama-build-context.h"
#include "../llama-context.h"
#include "../llama-delta-net.h"
#include "../llama-model.h"
#include "../llama-spec-features-dflash.h"

// Score-chunk size for the k-pool indexer: bounds the [n_pool, n_head, Tc] score tensor so it
// stays under the int32 element-count limit of the RELU kernel and the GPU compute buffer.
// Mirrors OPENPANGU_IDX_SCORE_CHUNK.
static constexpr int64_t GLM5NEXT_IDX_SCORE_CHUNK = 256;

// GLM-5.3-Flash (glm5next): hybrid trunk that alternates KDA linear attention (34 of 45
// layers) with absorbed MLA + DSA lightning k-pool indexer (11 layers), every block wrapped
// in hyper-connection streams (mHC, Sinkhorn). NoPE-only (rope.dimension_count = 0).
// The NextN/MTP block is wired through the GLM5NEXT MTP graph.

// mHC pre (copied from build_deepseek4.cpp)

static ggml_tensor * glm5next_build_hc_pre(
        ggml_context * ctx0,
        llm_build_context & llm,
        const llama_hparams & hparams,
        int64_t n_embd,
        float norm_rms_eps,
        ggml_tensor * x,
        ggml_tensor * hc_fn,
        ggml_tensor * hc_scale,
        ggml_tensor * hc_base,
        ggml_tensor ** post_out,
        ggml_tensor ** comb_out,
        const llm_build_cb & cb, int il) {
    const int64_t hc = hparams.dsv4_hc_mult;
    const int64_t nt = x->ne[2];

    if (!ggml_is_contiguous(x)) {
        x = ggml_cont(ctx0, x);
    }
    auto flat   = ggml_reshape_2d(ctx0, x, n_embd * hc, nt);
    auto normed = ggml_rms_norm(ctx0, flat, norm_rms_eps);
    cb(normed, "hc_pre", il);
    auto mixes = ggml_mul_mat(ctx0, hc_fn, normed);
    cb(mixes, "hc_pre_mixes", il);

    auto all = ggml_hc_pre(ctx0, mixes, hc_scale, hc_base, hc, hparams.dsv4_hc_sinkhorn_iters, hparams.dsv4_hc_eps);

    auto pre  = ggml_view_2d(ctx0, all, hc, nt, hc * sizeof(float), 0);
    auto post = ggml_view_2d(ctx0, all, hc, nt, hc * sizeof(float), hc * nt * sizeof(float));
    auto comb = ggml_view_3d(ctx0, all, hc, hc, nt, hc * sizeof(float), hc * hc * sizeof(float), 2 * hc * nt * sizeof(float));

    *post_out = post;
    *comb_out = comb;

    return llm.build_mhc_weighted_sum(x, pre, n_embd, hc);
}

// DSA k-pool indexer: scores k-pools of kpool tokens (a pool key is a learned convex mix of
// the member keys) and returns top-k cell indices [n_sel*kpool, n_tokens] into the KV cache,
// or nullptr for the dense fallback. The kr_l cache holds [key; gate] per token (2*d wide).

static ggml_tensor * build_glm5next_dsa_top_k(
        llm_build_context & llm,
        ggml_cgraph * gf, int il,
        ggml_tensor * cur,            // [n_embd, n_tokens] — pre-norm input to the attention block
        ggml_tensor * qr,             // [q_lora_rank, n_tokens] — q_lora latent (post wq_a + norm)
        ggml_tensor * pool_cells,     // I32 [kpool*n_pool]   — cell index of each pool member
        ggml_tensor * pool_bias,      // F32 [n_pool, n_tokens] — 0 if pool visible, else -inf
        ggml_tensor * tail_cells,     // I32 [kpool-1, n_tokens, 1, 1] or null — trailing incomplete pool
        ggml_tensor * ape_slots) {    // I32 [kpool]          — identity [0..kpool-1]
    auto & lctx    = llm.lctx;
    auto & ctx0    = llm.ctx0;
    auto & hparams = llm.hparams;
    auto & model   = llm.model;
    auto & kv_self = lctx.kv_self;
    auto & cb      = llm.cb;
    auto & layer   = model.layers[il];

    const int64_t d      = hparams.indexer_head_size;   // 128
    const int64_t nh     = hparams.indexer_n_head;       // 32
    const int64_t r      = hparams.indexer_block_size;   // 4 (kpool)
    const int64_t n_kv   = llm.n_kv;
    const int64_t n_tok  = llm.n_tokens;
    const int64_t n_pool = n_kv / r;

    if (n_pool == 0) {
        return nullptr;
    }

    // ---- indexer key + gate (per token) ----
    ggml_tensor * k = ggml_mul_mat(ctx0, layer.indexer_attn_k, cur);   // [d, n_tok]
    k = llm_build_context::llm_build_norm(ctx0, k, hparams, layer.indexer_k_norm, layer.indexer_k_norm_b, LLM_NORM, cb, il);
    cb(k, "dsa_indexer_k", il);

    ggml_tensor * g = ggml_mul_mat(ctx0, layer.indexer_comp_wgate, cur);  // [d, n_tok]
    cb(g, "dsa_indexer_gate", il);

    // pack [key; gate] into one cache row: [2*d, n_tok]
    ggml_tensor * packed = ggml_reshape_2d(ctx0, ggml_concat(ctx0, k, g, 0), 2 * d, n_tok);
    cb(packed, "dsa_indexer_packed", il);

    // write to kr_l at kv_head (key+gate packed, row = 2*d)
    const auto kr_row = ggml_row_size(kv_self.kr_l[il]->type, 2 * d);
    auto kr_view = ggml_view_2d(ctx0, kv_self.kr_l[il], 2 * d, n_tok, kr_row, kr_row * llm.kv_head);
    auto kr_cpy = ggml_cpy(ctx0, packed, kr_view);
    if ((size_t) il < lctx.dsa_cache_copies.size()) {
        lctx.dsa_cache_copies[il].cpy  = kr_cpy;
        lctx.dsa_cache_copies[il].step = kv_self.kr_l[il]->nb[1];
    }
    ggml_build_forward_expand(gf, kr_cpy);

    // ---- read back the full cached key+gate set: [2*d, n_kv] ----
    auto all = ggml_view_2d(ctx0, kv_self.kr_l[il], 2 * d, n_kv, kr_row, 0);
    cb(all, "dsa_cached_all", il);

    // ---- gather pool members: pool_cells indexes which cells belong to each pool ----
    // pool_cells is 1D [r*n_pool]: membership depends on cache positions, shared across all query tokens
    auto members = ggml_get_rows(ctx0, all, pool_cells);     // [2*d, r*n_pool]
    members = ggml_reshape_3d(ctx0, members, 2 * d, r, n_pool);
    cb(members, "dsa_pool_members", il);

    // split into keys [d, r, n_pool] and gates [d, r, n_pool] (packed along dim 0)
    auto m_k = ggml_cont(ctx0, ggml_view_3d(ctx0, members, d, r, n_pool,
            members->nb[1], members->nb[2], 0));
    auto m_g = ggml_cont(ctx0, ggml_view_3d(ctx0, members, d, r, n_pool,
            members->nb[1], members->nb[2], ggml_row_size(members->type, d)));
    cb(m_k, "dsa_pool_k", il);
    cb(m_g, "dsa_pool_g", il);

    // pool key = softmax(gate + ape) . keys over the r members, channel by channel.
    // ape is an intra-pool position bias
    m_g = ggml_add(ctx0, m_g, ggml_get_rows(ctx0, layer.indexer_comp_ape, ape_slots));

    // softmax normalizes ne[0], so bring the member axis there
    auto w = ggml_soft_max(ctx0, ggml_cont(ctx0, ggml_permute(ctx0, m_g, 1, 0, 2, 3)));
    auto v = ggml_cont(ctx0, ggml_permute(ctx0, m_k, 1, 0, 2, 3));
    auto pooled = ggml_sum_rows(ctx0, ggml_mul(ctx0, v, w));
    pooled = ggml_cont(ctx0, ggml_permute(ctx0, pooled, 1, 0, 2, 3));
    pooled = ggml_reshape_2d(ctx0, pooled, d, n_pool);
    cb(pooled, "dsa_indexer_k_pooled", il);

    // ---- indexer query (nope-only: no rope) ----
    auto q = ggml_mul_mat(ctx0, layer.indexer_attn_q_b, qr);  // [d*nh, n_tok]
    q = ggml_reshape_3d(ctx0, q, d, nh, n_tok);
    cb(q, "dsa_indexer_q", il);

    // relu(x*s) == s*relu(x) for s > 0, so both positive scalars fold into the weights
    auto wts = ggml_mul_mat(ctx0, layer.indexer_proj, cur);  // [nh, n_tok]
    wts = ggml_scale(ctx0, wts, 1.0f / sqrtf(float(d * nh)));
    wts = ggml_reshape_3d(ctx0, wts, nh, 1, n_tok);
    cb(wts, "dsa_indexer_weights", il);

    // ---- top-k over pools (cut on whole pools, never single cells) ----
    // reserve room for the tail (r-1 cells) so r*n_sel + (r-1) <= n_kv; dense fallback if the cache is too small
    const int64_t tail_cnt = tail_cells ? (r - 1) : 0;
    const int64_t n_sel = std::min<int64_t>({n_pool,
            (int64_t) hparams.indexer_top_k / r,
            (n_kv - tail_cnt) / r});
    if (n_sel < 1) {
        return nullptr;  // cache too small for the indexer — dense attention over the few keys
    }

    // ---- score: relu(pooled . q), weighted by proj, summed over heads ----
    // chunk the score over the query dim to bound the [n_pool, nh, Tc] tensor
    // (mirrors OPENPANGU_IDX_SCORE_CHUNK); pooled/q/wts are built whole, sel concatenates the
    // per-chunk top-k results along the token dim
    const bool chunk_scores = GLM5NEXT_IDX_SCORE_CHUNK > 0 && n_tok > GLM5NEXT_IDX_SCORE_CHUNK;
    ggml_tensor * sel = nullptr;
    if (!chunk_scores) {
        // small batch (decode or short prefill): no chunking needed
        auto score = ggml_mul_mat(ctx0, pooled, q);                  // [n_pool, nh, n_tok]
        score = ggml_relu(ctx0, score);
        score = ggml_cont(ctx0, ggml_permute(ctx0, score, 1, 0, 2, 3));  // [nh, n_pool, n_tok]
        score = ggml_sum_rows(ctx0, ggml_mul(ctx0, score, wts));     // [1, n_pool, n_tok]
        score = ggml_reshape_2d(ctx0, score, n_pool, n_tok);         // [n_pool, n_tok]
        score = ggml_add(ctx0, score, pool_bias);
        cb(score, "dsa_indexer_score", il);
        sel = ggml_cont(ctx0, ggml_top_k(ctx0, score, n_sel));       // [n_sel, n_tok]
        cb(sel, "dsa_top_k_pools", il);
    } else {
        for (int64_t c0 = 0; c0 < n_tok; c0 += GLM5NEXT_IDX_SCORE_CHUNK) {
            const int64_t tc = std::min<int64_t>(GLM5NEXT_IDX_SCORE_CHUNK, n_tok - c0);
            // view q [d, nh, T] → [d, nh, tc] at token offset c0
            auto q_c = ggml_cont(ctx0, ggml_view_3d(ctx0, q, d, nh, tc,
                    q->nb[1], q->nb[2], (size_t) c0 * q->nb[2]));
            // view wts [nh, 1, T] → [nh, 1, tc] at token offset c0
            auto wts_c = ggml_view_3d(ctx0, wts, nh, 1, tc,
                    wts->nb[1], wts->nb[2], (size_t) c0 * wts->nb[2]);
            // view pool_bias [n_pool, T] → [n_pool, tc] at token offset c0
            auto pb_c = ggml_cont(ctx0, ggml_view_2d(ctx0, pool_bias, n_pool, tc,
                    pool_bias->nb[1], (size_t) c0 * pool_bias->nb[1]));

            auto sc = ggml_mul_mat(ctx0, pooled, q_c);               // [n_pool, nh, tc]
            sc = ggml_relu(ctx0, sc);
            sc = ggml_cont(ctx0, ggml_permute(ctx0, sc, 1, 0, 2, 3));    // [nh, n_pool, tc]
            sc = ggml_sum_rows(ctx0, ggml_mul(ctx0, sc, wts_c));     // [1, n_pool, tc]
            sc = ggml_reshape_2d(ctx0, sc, n_pool, tc);              // [n_pool, tc]
            sc = ggml_add(ctx0, sc, pb_c);

            auto sel_c = ggml_cont(ctx0, ggml_top_k(ctx0, sc, n_sel));   // [n_sel, tc]
            sel = sel == nullptr ? sel_c : ggml_concat(ctx0, sel, sel_c, 1);
        }
        cb(sel, "dsa_top_k_pools", il);
    }

    // ---- expand each selected pool into its r member cells ----
    // pool_cells is 1D [r*n_pool] → reshape to [r, n_pool] for the gather
    auto pools = ggml_reshape_2d(ctx0, pool_cells, r, n_pool);  // [r, n_pool]
    auto top_k = ggml_get_rows(ctx0, pools,
            ggml_reshape_2d(ctx0, sel, n_sel * n_tok, 1));      // [r, n_sel*n_tok]
    top_k = ggml_reshape_2d(ctx0, top_k, r * n_sel, n_tok);     // [r*n_sel, n_tok]

    // index_kpool_always_select_tail: the trailing incomplete pool has no pool key and can
    // never be picked above, so its cells are appended instead of taking pool budget
    if (tail_cells) {
        top_k = ggml_concat(ctx0, top_k, tail_cells, 0);
    }
    cb(top_k, "dsa_top_k", il);

    return top_k;
}

// q = wq_b(wq_a(x))                        → [n_head * head_dim, n_tokens]
// Qcur = wk_b(permute(q))                  → [kv_lora_rank, n_tokens, n_head]  (absorbed)
// kv_cmpr = norm(wkv_a_mqa(x))             → [kv_lora_rank, n_tokens]
// store kv_cmpr in k_l, transposed in v_l
// Flash (-fa):  kqv_cmpr = flash_attn_ext(Qcur, K=V=K_cache, mask, scale)  (fused, no score matrix)
// Non-flash:    kq = K_cache @ Qcur → soft_max(kq, mask) → kqv_cmpr = V_cache_trans @ kq
// out = wo @ (wv_b @ kqv_cmpr)             → [n_embd, n_tokens]
//
// No rope, no q_rope/k_rope split. Returns attention output WITHOUT residual (mHC handles it).
// When the DSA k-pool indexer is active (--dsa + indexer tensors + kpool > 0), the flash path
// uses a sparse top-k mask instead of the dense KQ_mask.

static ggml_tensor * build_glm5next_mla_attention(
        llm_build_context & llm,
        ggml_cgraph * gf, int il,
        ggml_tensor * inp,             // [n_embd, n_tokens] — pre-norm
        ggml_tensor * KQ_mask,
        float kq_scale,
        ggml_tensor * pool_cells    = nullptr,
        ggml_tensor * pool_bias     = nullptr,
        ggml_tensor * tail_cells    = nullptr,
        ggml_tensor * ape_slots     = nullptr) {
    auto & lctx    = llm.lctx;
    auto & ctx0    = llm.ctx0;
    auto & hparams = llm.hparams;
    auto & model   = llm.model;
    auto & cb      = llm.cb;
    auto & layer   = model.layers[il];

    const int64_t n_head        = llm.n_head;
    const int64_t head_dim      = hparams.n_embd_head_k_full;
    const int64_t kv_lora_rank  = hparams.n_lora_kv;
    const int64_t n_embd_head_v = hparams.n_embd_head_v_full;

    auto cur = llm_build_context::llm_build_norm(ctx0, inp, hparams, layer.attn_norm, nullptr, LLM_NORM_RMS, cb, il);
    cb(cur, "attn_norm", il);

    auto qr = ggml_mul_mat(ctx0, layer.wq_a, cur);
    cb(qr, "q_a", il);
    qr = llm_build_context::llm_build_norm(ctx0, qr, hparams, layer.attn_q_a_norm, nullptr, LLM_NORM_RMS, cb, il);
    cb(qr, "q_a_norm", il);

    // DSA k-pool indexer: score k-pools and pick top_k cells (null = dense fallback).
    // Gate: --dsa flag + indexer tensors present + kpool > 0 + pool metadata created.
    ggml_tensor * top_k = nullptr;
    if (lctx.cparams.dsa && layer.indexer_attn_q_b && hparams.indexer_block_size > 0 && pool_cells) {
        top_k = build_glm5next_dsa_top_k(llm, gf, il, cur, qr, pool_cells, pool_bias, tail_cells, ape_slots);
    }

    mla_params p{layer};
    p.inp               = cur;
    p.gf                = gf;
    p.qr                = qr;
    p.KQ_mask           = KQ_mask;
    p.top_k             = top_k;
    p.il                = il;
    p.n_head            = n_head;
    p.n_embd_head_k     = head_dim;
    p.n_embd_head_v     = n_embd_head_v;
    p.n_extra           = 0;
    p.kv_lora_rank      = kv_lora_rank;
    p.kq_scale          = kq_scale;
    p.inp_is_normed     = true;
    p.use_v_cache_trans = lctx.cparams.mla_attn == 1;
    p.wkv_b             = layer.wkv_b;
    const size_t pp_bytes = layer.wkv_b ? (size_t) layer.wkv_b->ne[1] * llm.n_kv * sizeof(float) : 0;
    p.expanded_pp       = layer.wkv_b && (lctx.cparams.attn_max_batch > 0 || pp_bytes <= 256ull*1024*1024) &&
                          !lctx.cparams.dsa && lctx.cparams.mla_attn > 1 &&
                          lctx.cparams.flash_attn && llm.n_tokens >= 128;
    return llm.build_mla_attention(p);
}

ggml_cgraph * llm_build_context::build_glm5next_mtp() {
    ggml_cgraph * gf = new_graph_custom();

    const bool emit_logits = cparams.mtp_op_type == MTP_OP_DRAFT_GEN ||
            cparams.mtp_op_type == MTP_OP_UPDATE_ACCEPTED;

    GGML_ASSERT(hparams.nextn_predict_layers == 1);

    const int il = hparams.n_layer - hparams.nextn_predict_layers;
    const auto & mtp_layer = model.layers[il];
    const float kq_scale = 1.0f / sqrtf(float(hparams.n_embd_head_k_full));

    GGML_ASSERT(mtp_layer.nextn.eh_proj != nullptr);
    GGML_ASSERT(mtp_layer.nextn.enorm != nullptr);
    GGML_ASSERT(mtp_layer.nextn.hnorm != nullptr);

    auto hidden_state = build_inp_mtp_states(n_embd);
    auto token_emb = build_inp_embd_mtp(model.tok_embd);
    auto inp_pos = build_inp_pos();
    auto pos_mask = ggml_cast(ctx0, inp_pos, GGML_TYPE_F32);
    pos_mask = ggml_clamp(ctx0, pos_mask, 0.0f, 1.0f);
    pos_mask = ggml_reshape_2d(ctx0, pos_mask, 1, n_tokens);
    pos_mask = ggml_repeat(ctx0, pos_mask, token_emb);
    if (token_emb->type != GGML_TYPE_F32) {
        token_emb = ggml_cast(ctx0, token_emb, GGML_TYPE_F32);
    }
    token_emb = ggml_mul(ctx0, token_emb, pos_mask);
    cb(token_emb, "mtp_token_embd_pos_masked", -1);
    auto KQ_mask = build_inp_KQ_mask();
    auto inp_out_ids = n_tokens > 1 ? build_inp_out_ids() : nullptr;

    // Build DSA selection for the NextN MLA layer when --dsa is enabled.
    ggml_tensor * pool_cells = nullptr;
    ggml_tensor * pool_bias  = nullptr;
    ggml_tensor * tail_cells = nullptr;
    ggml_tensor * ape_slots  = nullptr;
    const bool use_dsa = lctx.cparams.dsa
        && hparams.indexer_head_size > 0
        && hparams.indexer_block_size > 0
        && il < (int) lctx.kv_self.kr_l.size()
        && lctx.kv_self.kr_l[il] != nullptr;
    if (use_dsa) {
        const int64_t r      = hparams.indexer_block_size;
        const int64_t n_kv   = this->n_kv;
        const int64_t n_pool = n_kv / r;
        if (n_pool > 0) {
            pool_cells = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, r * n_pool);
            pool_bias  = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_pool, n_tokens);
            ape_slots  = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, r);
            ggml_set_input(pool_cells);
            ggml_set_input(pool_bias);
            ggml_set_input(ape_slots);
            lctx.inp_kpool_cells     = pool_cells;
            lctx.inp_kpool_bias      = pool_bias;
            lctx.inp_kpool_ape_slots = ape_slots;
            if (r > 1) {
                tail_cells = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, r - 1, n_tokens);
                ggml_set_input(tail_cells);
                lctx.inp_kpool_tail = tail_cells;
            }
            cb(pool_cells, "inp_kpool_cells", -1);
            cb(pool_bias,  "inp_kpool_bias",  -1);
            cb(ape_slots,  "inp_kpool_ape",   -1);
            if (tail_cells) {
                cb(tail_cells, "inp_kpool_tail", -1);
            }
        }
    }

    auto cur = build_mtp_input(mtp_layer, hidden_state, token_emb, il);
    auto inpSA = cur;
    cur = build_glm5next_mla_attention(*this, gf, il, cur, KQ_mask, kq_scale,
            pool_cells, pool_bias, tail_cells, ape_slots);

    auto ffn_inp = ggml_add(ctx0, cur, inpSA);
    cb(ffn_inp, "mtp_ffn_inp", il);
    if (inp_out_ids) {
        ffn_inp = ggml_get_rows(ctx0, ffn_inp, inp_out_ids);
    }

    cur = llm_build_norm(ctx0, ffn_inp, hparams, mtp_layer.ffn_norm, nullptr, LLM_NORM_RMS, cb, il);
    cb(cur, "ffn_norm", il);

    auto moe_out = llm_build_moe_ffn(ctx0, lctx, cur,
            mtp_layer.ffn_gate_inp, nullptr,
            mtp_layer.ffn_up_exps, nullptr,
            mtp_layer.ffn_gate_exps, nullptr,
            mtp_layer.ffn_down_exps, nullptr,
            mtp_layer.ffn_exp_probs_b,
            n_expert, n_expert_used,
            LLM_FFN_SILU, hparams.expert_weights_norm,
            true, hparams.expert_weights_scale,
            (llm_expert_gating_func_type) hparams.expert_gating_func,
            cb, il, gf, false, mtp_layer.ffn_up_gate_exps, nullptr, nullptr, nullptr, nullptr);
    ggml_build_forward_expand(gf, moe_out);
    cb(moe_out, "ffn_moe_out", il);

    auto ffn_shexp = llm_build_ffn(ctx0, lctx, nullptr, cur,
            mtp_layer.ffn_up_shexp, nullptr, nullptr,
            mtp_layer.ffn_gate_shexp, nullptr, nullptr,
            mtp_layer.ffn_down_shexp, nullptr, nullptr,
            nullptr,
            LLM_FFN_SILU, LLM_FFN_PAR, cb, il);
    cb(ffn_shexp, "ffn_shexp", il);

    cur = ggml_add(ctx0, moe_out, ffn_shexp);
    cur = ggml_add(ctx0, cur, ffn_inp);
    cur = lctx.cvec.apply_to(ctx0, cur, il);
    cb(cur, "mtp_ffn_out_resid", il);

    if (!emit_logits) {
        ggml_build_forward_expand(gf, cur);
        return gf;
    }

    ggml_tensor * head_norm = mtp_layer.nextn.shared_head_norm
        ? mtp_layer.nextn.shared_head_norm : model.output_norm;
    GGML_ASSERT(head_norm != nullptr);

    cur = llm_build_norm(ctx0, cur, hparams, head_norm,
            nullptr, LLM_NORM_RMS, cb, il);
    cb(cur, "result_norm", -1);

    cur = build_output(lctx, ctx0, cur, model.output, nullptr, cb);
    cb(cur, "result_output", -1);
    ggml_build_forward_expand(gf, cur);
    return gf;
}

ggml_cgraph * llm_build_context::build_glm5next() {
    if (lctx.cparams.mtp_op_type != MTP_OP_NONE) {
        return build_glm5next_mtp();
    }

    ggml_cgraph * gf = new_graph_custom();

    const int64_t hc    = hparams.dsv4_hc_mult;

    // NoPE: no YaRN mscale correction
    const float kq_scale = 1.0f / sqrtf(float(hparams.n_embd_head_k_full));

    delta_net delta(lctx, batch);

    auto inpL = llm_build_inp_embd(ctx0, lctx, hparams, batch, model.tok_embd, cb);
    auto KQ_mask = build_inp_KQ_mask();
    auto inp_out_ids = (!lctx.cparams.mtp || n_outputs > 0) ? build_inp_out_ids() : nullptr;

    // KDA recurrent state slot routing
    lctx.inp_s_seq_qnext = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, 1, n_tokens);
    cb(lctx.inp_s_seq_qnext, "inp_s_seq_qnext", -1);
    ggml_set_input(lctx.inp_s_seq_qnext);

    // DSA k-pool indexer pool metadata: built once, shared across all MLA layers, host-filled
    // in llama_set_inputs. Allocated when --dsa is set and the model carries indexer weights.
    ggml_tensor * pool_cells = nullptr;
    ggml_tensor * pool_bias  = nullptr;
    ggml_tensor * tail_cells = nullptr;
    ggml_tensor * ape_slots  = nullptr;
    const bool use_dsa = lctx.cparams.dsa
        && hparams.indexer_head_size > 0
        && hparams.indexer_block_size > 0
        && !lctx.kv_self.kr_l.empty();
    if (use_dsa) {
        const int64_t r      = hparams.indexer_block_size;
        const int64_t n_kv   = this->n_kv;
        const int64_t n_pool = n_kv / r;
        if (n_pool > 0) {
            pool_cells = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, r * n_pool);
            pool_bias  = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_pool, n_tokens);
            ape_slots  = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, r);
            ggml_set_input(pool_cells);
            ggml_set_input(pool_bias);
            ggml_set_input(ape_slots);
            lctx.inp_kpool_cells     = pool_cells;
            lctx.inp_kpool_bias      = pool_bias;
            lctx.inp_kpool_ape_slots = ape_slots;
            if (r > 1) {
                // trailing incomplete pool cells: [kpool-1, n_tokens] (per-query)
                tail_cells = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, r - 1, n_tokens);
                ggml_set_input(tail_cells);
                lctx.inp_kpool_tail = tail_cells;
            }
            cb(pool_cells, "inp_kpool_cells", -1);
            cb(pool_bias,  "inp_kpool_bias",  -1);
            cb(ape_slots,  "inp_kpool_ape",   -1);
            if (tail_cells) {
                cb(tail_cells, "inp_kpool_tail", -1);
            }
        }
    }

    // expand embedding to hc streams: [n_embd, n_tokens] → [n_embd, hc, n_tokens]
    inpL = ggml_reshape_3d(ctx0, inpL, n_embd, 1, n_tokens);
    inpL = ggml_repeat_4d(ctx0, inpL, n_embd, hc, n_tokens, 1);
    cb(inpL, "hc_init", -1);

    const int n_trunk_layers = hparams.n_layer_kv_from_start;

    for (int il = 0; il < n_trunk_layers; ++il) {
        // attention block (hc_pre → attn → mhc_post)
        ggml_tensor * post_attn, * comb_attn;
        auto residual_attn = inpL;

        auto cur = glm5next_build_hc_pre(ctx0, *this, hparams, n_embd, hparams.f_norm_rms_eps,
                inpL,
                model.layers[il].hc_attn_fn,
                model.layers[il].hc_attn_scale,
                model.layers[il].hc_attn_base,
                &post_attn, &comb_attn, cb, il);
        cb(cur, "hc_attn_pre", il);

        if (hparams.is_recurrent(il)) {
            // KDA linear attention (applies attn_norm internally; no residual — mHC handles it)
            cur = delta.build_layer_attn_kda(ctx0, gf, cur, nullptr, il, cb, /*add_residual = */ false);
        } else {
            // NoPE absorbed MLA (applies attn_norm internally; no residual — mHC handles it).
            // When --dsa is active, the indexer builds a sparse top-k mask; else dense.
            cur = build_glm5next_mla_attention(*this, gf, il, cur, KQ_mask, kq_scale,
                    pool_cells, pool_bias, tail_cells, ape_slots);
        }

        inpL = build_mhc_post(cur, post_attn, residual_attn, comb_attn, n_embd, hc, true);
        cb(inpL, "attn_out", il);

        // FFN block (hc_pre → ffn_norm → ffn/moe → mhc_post)
        ggml_tensor * post_ffn, * comb_ffn;
        auto residual_ffn = inpL;

        cur = glm5next_build_hc_pre(ctx0, *this, hparams, n_embd, hparams.f_norm_rms_eps,
                inpL,
                model.layers[il].hc_ffn_fn,
                model.layers[il].hc_ffn_scale,
                model.layers[il].hc_ffn_base,
                &post_ffn, &comb_ffn, cb, il);
        cb(cur, "hc_ffn_pre", il);

        cur = llm_build_norm(ctx0, cur, hparams, model.layers[il].ffn_norm, nullptr, LLM_NORM_RMS, cb, il);
        cb(cur, "ffn_norm", il);

        if ((uint32_t) il < hparams.n_layer_dense_lead) {
            // dense FFN: no residual (mHC owns it); ffn_norm already applied above
            cur = llm_build_ffn(ctx0, lctx, nullptr, cur,
                    model.layers[il].ffn_up,   nullptr, nullptr,
                    model.layers[il].ffn_gate, nullptr, nullptr,
                    model.layers[il].ffn_down, nullptr, nullptr,
                    nullptr,
                    LLM_FFN_SILU, LLM_FFN_PAR, cb, il);
        } else {
            // MoE: routed experts (add_input=false, mHC owns the residual) + shared expert
            auto moe_out = llm_build_moe_ffn(ctx0, lctx, cur,
                        model.layers[il].ffn_gate_inp, nullptr,
                        model.layers[il].ffn_up_exps, nullptr,
                        model.layers[il].ffn_gate_exps, nullptr,
                        model.layers[il].ffn_down_exps, nullptr,
                        model.layers[il].ffn_exp_probs_b,
                        n_expert, n_expert_used,
                        LLM_FFN_SILU, hparams.expert_weights_norm,
                        true, hparams.expert_weights_scale,
                        (llm_expert_gating_func_type) hparams.expert_gating_func,
                        cb, il, gf, false, model.layers[il].ffn_up_gate_exps, nullptr, nullptr, nullptr, nullptr);
            ggml_build_forward_expand(gf, moe_out);
            cb(moe_out, "ffn_moe_out", il);

            ggml_tensor * ffn_shexp = llm_build_ffn(ctx0, lctx, nullptr, cur,
                    model.layers[il].ffn_up_shexp,   nullptr, nullptr,
                    model.layers[il].ffn_gate_shexp, nullptr, nullptr,
                    model.layers[il].ffn_down_shexp, nullptr, nullptr,
                    nullptr,
                    LLM_FFN_SILU, LLM_FFN_PAR, cb, il);
            cb(ffn_shexp, "ffn_shexp", il);

            cur = ggml_add(ctx0, moe_out, ffn_shexp);
            ggml_build_forward_expand(gf, cur);
        }
        cb(cur, "ffn_out", il);

        inpL = build_mhc_post(cur, post_ffn, residual_ffn, comb_ffn, n_embd, hc, true);
        inpL = lctx.cvec.apply_to(ctx0, inpL, il);
        if (lctx.dflash.capture && llama_dflash_capture_layer_requested(lctx.dflash.capture->layer_ids, il)) {
            ggml_tensor * capture = llama_dflash_hc_mean_for_capture(ctx0, inpL, hparams.dsv4_hc_mult);
            cb(capture, "l_out", il);
            ggml_build_forward_expand(gf, capture);
        } else {
            cb(inpL, "l_out", il);
        }
    }

    // Collapse the mHC streams before the shared output head.
    {
        auto flat = ggml_reshape_2d(ctx0, inpL, n_embd * hc, n_tokens);
        if (!lctx.cparams.mtp && n_outputs != n_tokens) {
            flat = ggml_get_rows(ctx0, flat, inp_out_ids);
        }
        const int64_t n_out = flat->ne[1];
        // sum the hc streams
        ggml_tensor * summed = nullptr;
        for (int64_t s = 0; s < hc; ++s) {
            auto stream = ggml_view_2d(ctx0, flat, n_embd, n_out,
                    flat->nb[1], s * n_embd * ggml_element_size(flat));
            summed = summed ? ggml_add(ctx0, summed, stream) : stream;
        }
        inpL = ggml_scale(ctx0, summed, 1.0f / hc);
        cb(inpL, "hc_collapse", -1);

        if (lctx.cparams.mtp) {
            // MTP consumes the full post-output-norm target state.
            inpL = llm_build_norm(ctx0, inpL, hparams, model.output_norm,
                    nullptr, LLM_NORM_RMS, cb, -1);
            cb(inpL, "result_mtp_embd", -1);
            ggml_set_output(inpL);
            ggml_build_forward_expand(gf, inpL);
            if (n_outputs == 0) {
                return gf;
            }
        }
    }

    if (lctx.cparams.mtp && n_outputs != n_tokens) {
        inpL = ggml_get_rows(ctx0, inpL, inp_out_ids);
    }

    auto cur = build_output(lctx, ctx0, inpL, model.output,
            lctx.cparams.mtp ? nullptr : model.output_norm, cb);
    cb(cur, "result_output", -1);
    ggml_build_forward_expand(gf, cur);
    return gf;
}

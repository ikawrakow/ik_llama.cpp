#include "../llama-build-context.h"
#include "../llama-context.h"
#include "../llama-delta-net.h"
#include "../llama-model.h"

// Score-chunk size for the k-pool indexer: bounds the [n_pool, n_head, Tc] score tensor so it
// stays under the int32 element-count limit of the RELU kernel and the GPU compute buffer.
// Mirrors OPENPANGU_IDX_SCORE_CHUNK.
static constexpr int64_t GLM5NEXT_IDX_SCORE_CHUNK = 256;

// GLM-5.3-Flash (glm5next): hybrid trunk that alternates KDA linear attention (34 of 45
// layers) with absorbed MLA + DSA lightning k-pool indexer (11 layers), every block wrapped
// in hyper-connection streams (mHC, Sinkhorn). NoPE-only (rope.dimension_count = 0).
// The NextN/MTP block is loaded but not wired for generation.

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
    auto & kv_self = lctx.kv_self;
    auto & cb      = llm.cb;
    auto & layer   = model.layers[il];

    const int64_t n_head       = llm.n_head;
    const int64_t head_dim     = hparams.n_embd_head_k_full;   // 256 (key_length_mla)
    const int64_t kv_lora_rank = hparams.n_lora_kv;            // 512
    const int64_t n_embd_head_v = hparams.n_embd_head_v_full;  // 256 (value_length_mla)

    // norm
    auto cur = llm_build_context::llm_build_norm(ctx0, inp, hparams, layer.attn_norm, nullptr, LLM_NORM_RMS, cb, il);
    cb(cur, "attn_norm", il);

    // q = wq_b(wq_a(x)) → [n_head * head_dim, n_tokens]
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

    auto q = ggml_mul_mat(ctx0, layer.wq_b, qr);
    cb(q, "q", il);

    // absorbed: Qcur = wk_b(permute(q)) → [kv_lora_rank, n_tokens, n_head]
    q = ggml_reshape_3d(ctx0, q, head_dim, n_head, llm.n_tokens);
    q = ggml_permute(ctx0, q, 0, 2, 1, 3);                         // [head_dim, n_tokens, n_head]
    auto Qcur = ggml_mul_mat(ctx0, layer.wk_b, q);                 // [kv_lora_rank, n_tokens, n_head]
    cb(Qcur, "Qcur", il);

    // kv_cmpr = norm(wkv_a_mqa(x)) → [kv_lora_rank, n_tokens]
    auto kv_cmpr = ggml_mul_mat(ctx0, layer.wkv_a_mqa, cur);
    cb(kv_cmpr, "kv_a_mqa", il);
    kv_cmpr = llm_build_context::llm_build_norm(ctx0, kv_cmpr, hparams, layer.attn_kv_a_norm, nullptr, LLM_NORM_RMS, cb, il);
    cb(kv_cmpr, "kv_compressed", il);

    // store kv_cmpr in k_l (K cache) — no rope part, just kv_lora_rank
    const auto row_size = ggml_row_size(kv_self.k_l[il]->type, kv_lora_rank);
    auto k_cache_view = ggml_view_2d(ctx0, kv_self.k_l[il], kv_lora_rank, llm.n_tokens,
            row_size, row_size * llm.kv_head);
    lctx.cache_copies[2 * il + 0].cpy  = ggml_cpy(ctx0, kv_cmpr, k_cache_view);
    lctx.cache_copies[2 * il + 0].step = row_size;
    ggml_build_forward_expand(gf, lctx.cache_copies[2 * il + 0].cpy);

    // read K cache
    auto K_cache = ggml_view_2d(ctx0, kv_self.k_l[il], kv_lora_rank, llm.n_kv, row_size, 0);
    cb(K_cache, "kv_cache", il);

    // Build the sparse attention mask when the indexer returned top_k cell indices.
    // ggml_indexer_mask takes the dense KQ_mask + I32 cell-index topk and unmask only those
    // cells (keeping future/padding masked via the base causal mask). Same op deepseek2 uses.
    ggml_tensor * attn_mask = KQ_mask;
    if (top_k) {
        attn_mask = ggml_indexer_mask(ctx0, KQ_mask, top_k);
        cb(attn_mask, "dsa_sparse_mask", il);
    }

    // Attention: flash (fused, avoids materializing [n_kv, n_tokens, n_head]) vs. soft_max path.
    // Flash is required for large contexts — the soft_max path materializes the full score
    // matrix [n_kv, n_tokens, n_head], whose compute buffer scales linearly with ctx-size.
    // Absorbed NoPE MLA: K = V = the compressed latent (kv_lora_rank, no rope part), MQA.
    ggml_tensor * kqv_cmpr = nullptr;
    if (lctx.cparams.flash_attn) {
        kqv_cmpr = ggml_flash_attn_ext(ctx0, Qcur, K_cache, K_cache, attn_mask,
                kq_scale, hparams.f_max_alibi_bias, 0.f);
        if (K_cache->ne[1] < 256) {
            ggml_flash_attn_ext_set_prec(kqv_cmpr, GGML_PREC_F32);
        }
        cb(kqv_cmpr, "kqv_compressed", il);
        // ggml_flash_attn_ext returns {v->ne[0], q->ne[2], q->ne[1], q->ne[3]} =
        // [kv_lora_rank, n_head, n_tokens, 1] — n_head and n_tokens are swapped vs. the
        // non-flash kqv_cmpr layout [kv_lora_rank, n_tokens, n_head]. Permute to match,
        // otherwise the downstream mul_mat(wv_b, kqv_cmpr) fails ggml_can_mul_mat for
        // n_tokens > 1 (batch dim n_head vs n_tokens mismatch). Mirrors build_deepseek2.
        kqv_cmpr = ggml_permute(ctx0, kqv_cmpr, 0, 2, 1, 3);
        cb(kqv_cmpr, "kqv_compressed_perm", il);
    } else {
        // V_cache_trans: with mla_attn == 1, store transposed kv_cmpr in v_l and read it back.
        // With mla_attn > 1, v_l is not allocated — transpose k_l on the fly.
        ggml_tensor * V_cache_trans = nullptr;
        if (lctx.cparams.mla_attn == 1 && kv_self.v_l[il]) {
            auto v_cache_trans_view = ggml_view_2d(ctx0, kv_self.v_l[il], llm.n_tokens, kv_lora_rank,
                    ggml_row_size(kv_self.v_l[il]->type, kv_self.size),
                    ggml_row_size(kv_self.v_l[il]->type, llm.kv_head));
            cb(v_cache_trans_view, "kv_cache_trans_view", il);
            ggml_build_forward_expand(gf, ggml_cpy(ctx0, ggml_transpose(ctx0, kv_cmpr), v_cache_trans_view));
            V_cache_trans = ggml_view_2d(ctx0, kv_self.v_l[il], llm.n_kv, kv_lora_rank,
                    ggml_row_size(kv_self.v_l[il]->type, kv_self.size), 0);
        } else {
            V_cache_trans = ggml_cont(ctx0, ggml_transpose(ctx0, K_cache));
        }
        cb(V_cache_trans, "kv_cache_trans", il);

        // kq = K_cache @ Qcur → [n_kv, n_tokens, n_head]
        auto kq = ggml_mul_mat(ctx0, K_cache, Qcur);
        if (K_cache->ne[1] < 256) {
            ggml_mul_mat_set_prec(kq, GGML_PREC_F32);
        }
        cb(kq, "kq", il);

        // soft_max with mask
        kq = ggml_soft_max_ext(ctx0, kq, attn_mask, kq_scale, hparams.f_max_alibi_bias);
        cb(kq, "kq_soft_max", il);

        // kqv_cmpr = V_cache_trans @ kq → [kv_lora_rank, n_tokens, n_head]
        kqv_cmpr = ggml_mul_mat(ctx0, V_cache_trans, kq);
        cb(kqv_cmpr, "kqv_compressed", il);
    }

    // kqv = wv_b @ kqv_cmpr → [n_embd_head_v, n_tokens, n_head]
    auto wv_b = layer.wv_b;
    if (wv_b->ne[1] != n_embd_head_v) {
        wv_b = ggml_reshape_3d(ctx0, wv_b, kv_lora_rank, n_embd_head_v, n_head);
    }
    auto kqv = ggml_mul_mat(ctx0, wv_b, kqv_cmpr);
    cb(kqv, "kqv", il);

    // reshape → [n_embd_head_v * n_head, n_tokens]
    kqv = ggml_permute(ctx0, kqv, 0, 2, 1, 3);   // [n_embd_head_v, n_head, n_tokens]
    kqv = ggml_cont(ctx0, kqv);
    kqv = ggml_reshape_2d(ctx0, kqv, n_embd_head_v * n_head, llm.n_tokens);
    cb(kqv, "kqv_2d", il);

    // out = wo @ kqv
    cur = llm_build_context::llm_build_lora_mm(lctx, ctx0, layer.wo, kqv);
    cb(cur, "kqv_out", il);

    return cur;
}


struct ggml_tensor * llm_build_context::build_glm5next_mtp(
        const llama_layer & mtp_layer,
        struct ggml_tensor * prev_embeddings,
        struct ggml_cgraph * gf) {

    const int il = hparams.n_layer - 1;

    struct ggml_tensor * KQ_mask = build_inp_KQ_mask();
    struct ggml_tensor * inp_out_ids = (n_tokens > 1 && n_outputs < n_tokens) ? build_inp_out_ids() : nullptr;

    struct ggml_tensor * token_emb = build_inp_embd_mtp(model.tok_embd);

    struct ggml_tensor * cur = build_mtp_input(mtp_layer, prev_embeddings, token_emb, il, "mtp_fused");

    // glm5next is MLA-absorbed: v_l may legitimately be shorter than k_l
    GGML_ASSERT(il < (int)kv_self.k_l.size());
    if (!kv_self.k_l[il]) {
        LLAMA_LOG_ERROR("%s: KV cache not allocated for MTP layer %d (k=%p)\n",
                __func__, il, (void*)kv_self.k_l[il]);
        GGML_ABORT("KV cache not allocated for MTP layer");
    }

    // DSA k-pool indexer pool metadata (same construction as the trunk preamble)
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
        const int64_t n_kv_  = this->n_kv;
        const int64_t n_pool = n_kv_ / r;
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

    // Attention: NoPE-absorbed MLA (+ optional DSA sparse mask). Applies attn_norm
    // internally and returns WITHOUT residual - we add it here (no mHC on this layer).
    const float kq_scale = 1.0f / sqrtf(float(hparams.n_embd_head_k_full));
    ggml_tensor * attn_out = build_glm5next_mla_attention(*this, gf, il, cur, KQ_mask, kq_scale,
            pool_cells, pool_bias, tail_cells, ape_slots);
    attn_out = ggml_add(ctx0, attn_out, cur);
    ggml_build_forward_expand(gf, attn_out);
    cb(attn_out, "mtp_attn_out", il);

    if (inp_out_ids) {
        attn_out = ggml_get_rows(ctx0, attn_out, inp_out_ids);
    }

    // FFN: ffn_norm -> routed MoE + shared expert, manual residual
    ggml_tensor * f = llm_build_norm(ctx0, attn_out, hparams, mtp_layer.ffn_norm, nullptr, LLM_NORM_RMS, cb, il);
    cb(f, "mtp_ffn_norm", il);

    auto moe_out = llm_build_moe_ffn(ctx0, lctx, f,
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
    cb(moe_out, "mtp_ffn_moe_out", il);

    ggml_tensor * ffn_shexp = llm_build_ffn(ctx0, lctx, nullptr, f,
            mtp_layer.ffn_up_shexp,   nullptr, nullptr,
            mtp_layer.ffn_gate_shexp, nullptr, nullptr,
            mtp_layer.ffn_down_shexp, nullptr, nullptr,
            nullptr,
            LLM_FFN_SILU, LLM_FFN_PAR, cb, il);
    cb(ffn_shexp, "mtp_ffn_shexp", il);

    cur = ggml_add(ctx0, ggml_add(ctx0, moe_out, ffn_shexp), attn_out);
    ggml_build_forward_expand(gf, cur);
    cur = lctx.cvec.apply_to(ctx0, cur, il);
    cb(cur, "mtp_ffn_out", il);

    // Shared output head (model.output_mtp falls back to the backbone output when
    // the GGUF carries no dedicated nextn head; shared_head_norm applies here).
    cur = build_output(lctx, ctx0, cur, model.output_mtp ? model.output_mtp : model.output, mtp_layer.nextn.shared_head_norm, cb);
    cb(cur, "result_output", -1);

    return cur;
}

ggml_cgraph * llm_build_context::build_glm5next() {
    ggml_cgraph * gf = new_graph_custom();

    if (cparams.mtp_op_type != MTP_OP_NONE) {
        // MTP tail-only graph: build just the nextn layer from main-model hidden states
        ggml_tensor * hidden_states_from_main_model = build_inp_mtp_states(hparams.n_embd);
        const int il_mtp = hparams.n_layer - 1;
        ggml_tensor * cur = build_glm5next_mtp(model.layers[il_mtp], hidden_states_from_main_model, gf);
        ggml_build_forward_expand(gf, cur);
        return gf;
    }

    const int64_t hc    = hparams.dsv4_hc_mult;

    // NoPE: no YaRN mscale correction
    const float kq_scale = 1.0f / sqrtf(float(hparams.n_embd_head_k_full));

    delta_net delta(lctx, batch);

    auto inpL = llm_build_inp_embd(ctx0, lctx, hparams, batch, model.tok_embd, cb);
    auto KQ_mask = build_inp_KQ_mask();
    auto inp_out_ids = build_inp_out_ids();

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

        ggml_tensor * prenorm_ffn = cur;
        cur = llm_build_norm(ctx0, cur, hparams, model.layers[il].ffn_norm, nullptr, LLM_NORM_RMS, cb, il);
        cb(cur, "ffn_norm", il);

        // Pre-gate instrumentation: predict layer il+1's routing from the current
        // hidden state (evaluate il+1's router one step early). Cheap: one norm +
        // one [n_expert, n_embd] GEMV per layer. Guarded by env var.
        if (getenv("IK_PREGATE") && (uint32_t)(il + 1) < hparams.n_layer &&
                (il + 1) >= (int) hparams.n_layer_dense_lead &&
                model.layers[il + 1].ffn_gate_inp) {
            auto pn = llm_build_norm(ctx0, prenorm_ffn, hparams, model.layers[il + 1].ffn_norm, nullptr, LLM_NORM_RMS, cb, il);
            auto pl = ggml_mul_mat(ctx0, model.layers[il + 1].ffn_gate_inp, pn);
            auto pk = ggml_top_k(ctx0, pl, n_expert_used);
            cb(pk, "pregate_topk", il);
            ggml_build_forward_expand(gf, pk);
        }

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
        cb(inpL, "l_out", il);
    }

    // collapse the hc streams for the head (unweighted mean; GLM-5.3-Flash has no head mHC), then
    // inp_out_ids to skip unused output tokens
    {
        auto flat = ggml_reshape_2d(ctx0, inpL, n_embd * hc, n_tokens);
        flat = ggml_get_rows(ctx0, flat, inp_out_ids);
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
    }

    auto cur = build_output(lctx, ctx0, inpL, model.output, model.output_norm, cb);
    cb(cur, "result_output", -1);
    ggml_build_forward_expand(gf, cur);
    return gf;
}

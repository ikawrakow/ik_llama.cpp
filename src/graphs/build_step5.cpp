#include "../llama-build-context.h"
#include "../llama-model.h"
#include "../llama-context.h"

// Step-5 sparse GQA indexer: proxy-key cache write + read-back scoring graph.

// IK_HADAMARD (default 1): whiten proxy keys before the quantized cache write, de-rotate at
// read-back. IK_HADAMARD=0 is for F16-cache cross-diff runs ONLY.
static bool ik_hadamard_enabled() {
    static const bool on = getenv("IK_HADAMARD") == nullptr || atoi(getenv("IK_HADAMARD")) != 0;
    return on;
}

// Materialize the reference's bf16 rounding at torch's cast points.
static ggml_tensor * step5_bf16_roundtrip(ggml_context * ctx, ggml_tensor * t) {
    return ggml_cast(ctx, ggml_cast(ctx, t, GGML_TYPE_BF16), GGML_TYPE_F32);
}

static bool ik_sparse_enabled() {
    static const bool on = getenv("IK_SPARSE") != nullptr && atoi(getenv("IK_SPARSE")) != 0;
    return on;
}

bool llama_step5_csa_z_cache() {
    return true;
}

static int64_t ik_kept_slots(const llama_hparams & hparams, int64_t n_kv_sel) {
    const int64_t Bsel = hparams.indexer_csa_block > 0 ? hparams.indexer_csa_block : 8;
    int64_t kept = (int64_t) hparams.indexer_top_k * Bsel;
    kept = ((kept + Bsel - 1) / Bsel) * Bsel;  // whole-block snap
    if (kept > n_kv_sel) kept = n_kv_sel;
    return kept;
}

void llm_build_context::build_step5_indexer_kv_write(ggml_cgraph * gf, int il, ggml_tensor * inpL) {
    const int64_t head_size = hparams.indexer_head_size;

    ggml_tensor * kr_cache = (size_t) il < kv_self.kr_l.size() ? kv_self.kr_l[il] : nullptr;
    if (!kr_cache || !hparams.indexer_is_full[il]) {
        return;
    }
    GGML_ASSERT(head_size > 0);

    const auto & layer = model.layers[il];
    ggml_tensor * src;
    GGML_ASSERT(layer.indexer_k && layer.indexer_k_norm && "STEP5 indexer k tensors missing");
    src = ggml_mul_mat(ctx0, layer.indexer_k, inpL); // {head_size, n_tokens}
    src = step5_bf16_roundtrip(ctx0, src); // torch linear materializes bf16 output
    src = llm_build_norm(ctx0, src, hparams, layer.indexer_k_norm, layer.indexer_k_norm_b, LLM_NORM, cb, il);
    src = step5_bf16_roundtrip(ctx0, src); // torch norm materializes bf16 output
    cb(src, "step5_indexer_k_raw", il);
    GGML_ASSERT(src->ne[0] == head_size && src->ne[1] == n_tokens);

    // Whiten before the cache-type cast; head_size must be a power of 2 (ggml_hadamard aborts otherwise).
    if (ik_hadamard_enabled()) {
        GGML_ASSERT(head_size > 1 && (head_size & ~(head_size - 1)) == head_size);
        src = ggml_hadamard(ctx0, src, (int) head_size);
        cb(src, "step5_indexer_k_rot", il);
    }

    src = ggml_cast(ctx0, ggml_cont(ctx0, src), kr_cache->type);
    cb(src, "step5_kr_plumb_src", il);

    ggml_tensor * kr_view = ggml_view_2d(ctx0, kr_cache, head_size, n_tokens,
            ggml_row_size(kr_cache->type, head_size),
            ggml_row_size(kr_cache->type, head_size) * kv_head);
    ggml_tensor * kr_cpy = ggml_cpy(ctx0, src, kr_view);
    cb(kr_cpy, "step5_kr_plumb_write", il);

    GGML_ASSERT((size_t) il < lctx.dsa_cache_copies.size());
    lctx.dsa_cache_copies[il].cpy  = kr_cpy;
    lctx.dsa_cache_copies[il].step = kr_cache->nb[1];

    ggml_build_forward_expand(gf, kr_cpy);

    // Cache raw z projection per cell.
    ggml_tensor * kz_cache = (size_t) il < kv_self.kz_l.size() ? kv_self.kz_l[il] : nullptr;
    if (kz_cache) {
        GGML_ASSERT(layer.indexer_z && "STEP5 indexer z tensor missing (official CSA z cache)");
        ggml_tensor * z_src = ggml_mul_mat(ctx0, layer.indexer_z, inpL); // {head_size, n_tokens}
        cb(z_src, "step5_indexer_z_cache_src", il);
        z_src = ggml_cast(ctx0, ggml_cont(ctx0, z_src), kz_cache->type);
        ggml_tensor * kz_view = ggml_view_2d(ctx0, kz_cache, head_size, n_tokens,
                ggml_row_size(kz_cache->type, head_size),
                ggml_row_size(kz_cache->type, head_size) * kv_head);
        ggml_tensor * kz_cpy = ggml_cpy(ctx0, z_src, kz_view);
        cb(kz_cpy, "step5_kz_plumb_write", il);

        GGML_ASSERT((size_t) il < lctx.dsa_z_cache_copies.size());
        lctx.dsa_z_cache_copies[il].cpy  = kz_cpy;
        lctx.dsa_z_cache_copies[il].step = kz_cache->nb[1];

        ggml_build_forward_expand(gf, kz_cpy);
    }
}

ggml_tensor * llm_build_context::build_step5_indexer_score(ggml_cgraph * gf, int il,
        ggml_tensor * inpL, ggml_tensor * inp_pos, ggml_tensor * KQ_mask,
        ggml_tensor * band01, ggml_tensor * band_neg, ggml_tensor ** fused_mask_out) {
    const int64_t head_size = hparams.indexer_head_size; // proxy_dim (HF sparse_config: 256)
    const int64_t n_ihead   = hparams.indexer_n_head;    // proxy q heads (HF num_heads: 16)

    ggml_tensor * kr_cache = (size_t) il < kv_self.kr_l.size() ? kv_self.kr_l[il] : nullptr;
    if (!kr_cache || !hparams.indexer_is_full[il]) {
        return nullptr;
    }
    GGML_ASSERT(head_size > 0 && n_ihead > 0);
    GGML_ASSERT(n_kv > 0 && n_kv <= (int64_t) kv_self.size);

    const bool fused_try = fused_mask_out != nullptr && ik_sparse_enabled() &&
            lctx.cparams.fused_idx_topk && !lctx.cparams.cb_eval;
    const int64_t kept = fused_try ? ik_kept_slots(hparams, n_kv) : 0;
    if (fused_try && kept >= n_kv) {
        return nullptr;
    }

    const auto & layer = model.layers[il];
    GGML_ASSERT(layer.indexer_q && layer.indexer_z && layer.indexer_w && layer.indexer_q_norm);

    GGML_ASSERT(inpL->ne[2] == 1);
    ggml_tensor * inpL_view = ggml_view_2d(ctx0, inpL, inpL->ne[0], inpL->ne[1], inpL->nb[1], 0);
    cb(inpL_view, "step5_indexer_inpL", il);
    if (lctx.cparams.cb_eval) {
        ggml_build_forward_expand(gf, inpL_view);
    }

    const int64_t rope_dim = std::min<int64_t>(
            hparams.indexer_rope_dim > 0 ? hparams.indexer_rope_dim : head_size, head_size);
    const int64_t nope_dim = head_size - rope_dim;

    if (!lctx.inp_kv_pos) {
        lctx.inp_kv_pos = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_kv);
        cb(lctx.inp_kv_pos, "kv_pos", -1);
        ggml_set_input(lctx.inp_kv_pos);
        ggml_build_forward_expand(gf, lctx.inp_kv_pos);
    }

    // ---- proxy q: rmsnorm(q_proj(inpL)) over proxy_dim per head, pe rope at query pos ----
    ggml_tensor * q_cur = ggml_mul_mat(ctx0, layer.indexer_q, inpL); // {head_size*n_ihead, n_tokens}
    q_cur = step5_bf16_roundtrip(ctx0, q_cur); // torch linear materializes bf16 output
    q_cur = ggml_view_3d(ctx0, q_cur, head_size, n_ihead, n_tokens,
            ggml_row_size(q_cur->type, head_size),
            ggml_row_size(q_cur->type, head_size) * n_ihead, 0);
    q_cur = llm_build_norm(ctx0, q_cur, hparams, layer.indexer_q_norm, nullptr, LLM_NORM_RMS, cb, il);
    q_cur = step5_bf16_roundtrip(ctx0, q_cur); // torch norm materializes bf16 output
    cb(q_cur, "step5_indexer_q_normed", il);

    ggml_tensor * q_pe = ggml_view_3d(ctx0, q_cur, rope_dim, n_ihead, n_tokens,
            ggml_row_size(q_cur->type, head_size),
            ggml_row_size(q_cur->type, head_size) * n_ihead, 0);
    q_pe = ggml_rope_ext(ctx0, q_pe, inp_pos, nullptr, rope_dim,
            rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow);
    if (nope_dim > 0) {
        ggml_tensor * q_nope = ggml_view_3d(ctx0, q_cur, nope_dim, n_ihead, n_tokens,
                ggml_row_size(q_cur->type, head_size),
                ggml_row_size(q_cur->type, head_size) * n_ihead,
                ggml_row_size(q_cur->type, rope_dim));
        q_cur = ggml_concat(ctx0, q_pe, q_nope, 0);
    } else {
        q_cur = q_pe;
    }
    cb(q_cur, "step5_indexer_q_cat", il);
    q_cur = step5_bf16_roundtrip(ctx0, q_cur); // torch rope materializes bf16 output
    // Official e4m3 activation round on the indexer q.
    q_cur = ggml_round_e4m3(ctx0, q_cur);
    cb(q_cur, "step5_indexer_q_e4m3", il);

    ggml_tensor * z_cur = ggml_mul_mat(ctx0, layer.indexer_z, inpL); // {head_size, n_tokens}
    z_cur = step5_bf16_roundtrip(ctx0, z_cur); // torch linear materializes bf16 output
    cb(z_cur, "step5_indexer_z_raw", il);
    if (lctx.cparams.cb_eval) {
        ggml_build_forward_expand(gf, z_cur);
    }

    ggml_tensor * w_cur = ggml_mul_mat(ctx0, layer.indexer_w, inpL); // {n_ihead, n_tokens}
    w_cur = step5_bf16_roundtrip(ctx0, w_cur); // torch linear materializes bf16 output
    w_cur = ggml_scale(ctx0, w_cur, 1.0f / sqrtf(float(head_size * n_ihead)));
    cb(w_cur, "step5_indexer_w", il);

    ggml_tensor * cached_k = ggml_view_2d(ctx0, kr_cache, head_size, n_kv,
            ggml_row_size(kr_cache->type, head_size), 0);
    // De-rotate the whitened cache rows (H^2 == I).
    ggml_tensor * k_f32;
    if (ik_hadamard_enabled()) {
        k_f32 = ggml_hadamard(ctx0, cached_k, (int) head_size);
        cb(k_f32, "step5_indexer_k_derot", il);
    } else {
        k_f32 = ggml_cast(ctx0, ggml_cont(ctx0, cached_k), GGML_TYPE_F32);
    }
    cb(k_f32, "step5_indexer_cached_k", il);

    ggml_tensor * k_pe = ggml_view_3d(ctx0, k_f32, rope_dim, 1, n_kv,
            ggml_row_size(k_f32->type, head_size),
            ggml_row_size(k_f32->type, head_size), 0);
    k_pe = ggml_rope_ext(ctx0, k_pe, lctx.inp_kv_pos, nullptr, rope_dim,
            rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow);
    ggml_tensor * indexer_k_b;
    if (nope_dim > 0) {
        ggml_tensor * k_nope = ggml_view_3d(ctx0, k_f32, nope_dim, 1, n_kv,
                ggml_row_size(k_f32->type, head_size),
                ggml_row_size(k_f32->type, head_size),
                ggml_row_size(k_f32->type, rope_dim));
        indexer_k_b = ggml_concat(ctx0, k_pe, k_nope, 0);
    } else {
        indexer_k_b = k_pe;
    }
    indexer_k_b = ggml_reshape_3d(ctx0, indexer_k_b, head_size, n_kv, 1);
    indexer_k_b = step5_bf16_roundtrip(ctx0, indexer_k_b); // torch rope materializes bf16 output
    cb(indexer_k_b, "step5_indexer_k_rope", il);

    // Scores: relu(q·k)·w summed over proxy heads at block grain, causal seed added last at slot granularity.
    ggml_tensor * mask_f32 = ggml_view_2d(ctx0, KQ_mask, n_kv, n_tokens, KQ_mask->nb[1], 0);
    if (mask_f32->type != GGML_TYPE_F32) {
        mask_f32 = ggml_cast(ctx0, mask_f32, GGML_TYPE_F32);
        cb(mask_f32, "step5_indexer_score_mask_f32", il);
    }

    // Per-indexer-head scoring on real ubatches (n_tokens > 8); fused shape capped at 8 tokens.
    auto score_chain = [&](ggml_tensor * keys, int64_t n_key) {
        if (n_tokens <= 8) {
            ggml_tensor * q2d = ggml_reshape_2d(ctx0, q_cur, head_size, n_ihead * n_tokens);
            ggml_tensor * kq = ggml_mul_mat(ctx0, keys, q2d); // {n_key, n_ihead*n_tokens}
            cb(kq, "step5_indexer_kq", il);
            kq = ggml_relu(ctx0, kq);
            cb(kq, "step5_indexer_kq_relu", il);
            kq = ggml_reshape_3d(ctx0, kq, n_key, n_ihead, n_tokens);
            kq = ggml_cont(ctx0, ggml_transpose(ctx0, kq)); // {n_ihead, n_key, n_tokens}
            ggml_tensor * w3 = ggml_reshape_3d(ctx0, w_cur, n_ihead, 1, n_tokens);
            kq = ggml_mul(ctx0, kq, w3);
            cb(kq, "step5_indexer_kq_w", il);
            ggml_tensor * s = ggml_sum_rows(ctx0, kq); // {1, n_key, n_tokens}
            return ggml_reshape_2d(ctx0, s, n_key, n_tokens);
        }
        ggml_tensor * acc = nullptr;
        for (int64_t head = 0; head < n_ihead; ++head) {
            const int il_cb = 1000 * (il + 1) + (int) head;
            // [head_size, n_tokens]: per-head slice of q_cur {head_size, n_ihead, n_tokens}
            ggml_tensor * q_h = ggml_view_2d(ctx0, q_cur, head_size, n_tokens,
                    ggml_row_size(q_cur->type, head_size) * n_ihead,
                    ggml_row_size(q_cur->type, head_size) * head);
            // [n_key, n_tokens]
            ggml_tensor * kq = ggml_mul_mat(ctx0, keys, q_h);
            cb(kq, "step5_indexer_kq", il_cb);
            kq = ggml_relu(ctx0, kq);
            cb(kq, "step5_indexer_kq_relu", il_cb);
            // [1, n_tokens] contiguous slice of w_cur {n_ihead, n_tokens}
            ggml_tensor * w_h = ggml_cont(ctx0, ggml_view_2d(ctx0, w_cur, 1, n_tokens,
                    w_cur->nb[1], w_cur->nb[0] * head));
            kq = ggml_mul(ctx0, kq, w_h); // broadcast {n_key, n_tokens} x {1, n_tokens}
            cb(kq, "step5_indexer_kq_w", il_cb);
            acc = acc ? ggml_add_inplace(ctx0, acc, kq) : kq;
        }
        return acc; // {n_key, n_tokens}
    };

    ggml_tensor * slot_score;
    const int64_t B = hparams.indexer_csa_block > 0 ? hparams.indexer_csa_block : 8;
    // Block-grain CSA with pooled block keys.
    {
        GGML_ASSERT(B > 0 && n_kv % B == 0 && "CSA slot-aligned blocks need n_kv divisible by B");
        const int64_t n_blocks = n_kv / B;

        ggml_tensor * vis = ggml_step(ctx0, ggml_scale_bias(ctx0, mask_f32, 1.0f, 0.5f));
        vis = ggml_sum_rows_ext(ctx0, vis, 1);                       // {n_kv, 1}
        vis = ggml_step(ctx0, ggml_scale_bias(ctx0, vis, 1.0f, -0.5f));
        vis = ggml_reshape_2d(ctx0, vis, n_kv, 1);
        cb(vis, "step5_indexer_slot_valid", il);
        ggml_tensor * block_k;
        {
            // Official per-dimension softmax(z)-weighted mean.
            ggml_tensor * kz_cache = (size_t) il < kv_self.kz_l.size() ? kv_self.kz_l[il] : nullptr;
            GGML_ASSERT(kz_cache && "official CSA needs the kz_l z cache (llama_step5_csa_z_cache)");
            if (kz_cache->type == GGML_TYPE_F16 || kz_cache->type == GGML_TYPE_F32) {
                ggml_tensor * z_view = ggml_view_2d(ctx0, kz_cache, head_size, n_kv,
                        ggml_row_size(kz_cache->type, head_size), 0);
                ggml_tensor * z_tap = ggml_cast(ctx0, ggml_cont(ctx0, z_view), GGML_TYPE_F32);
                cb(z_tap, "step5_indexer_cached_z", il);
                if (lctx.cparams.cb_eval) {
                    ggml_build_forward_expand(gf, z_tap);
                }
                block_k = ggml_csa_compress(ctx0, z_view, indexer_k_b, vis, (int) B, 1);
                cb(block_k, "step5_indexer_block_k", il);
                if (supports_op(block_k)) {
                    ggml_build_forward_expand(gf, block_k);
                } else {
                    // backend cannot run the fused op: fall through to the decomposed chain.
                    block_k = nullptr;
                }
            } else {
                block_k = nullptr; // Q8_0 z: decomposed path only
            }
            if (!block_k) {
                ggml_tensor * z_f32 = ggml_view_2d(ctx0, kz_cache, head_size, n_kv,
                        ggml_row_size(kz_cache->type, head_size), 0);
                z_f32 = ggml_cast(ctx0, ggml_cont(ctx0, z_f32), GGML_TYPE_F32); // {head_size, n_kv}
                cb(z_f32, "step5_indexer_cached_z", il);

            // Common layout: axis0 = token-within-region (softmax + sum axis), axis1 = dim, axis2 = region.
            ggml_tensor * z_pv = ggml_cont(ctx0, ggml_permute(ctx0,
                    ggml_reshape_3d(ctx0, z_f32, head_size, B, n_blocks), 1, 0, 2, 3));
            cb(z_pv, "step5_indexer_csa_z_blocks", il);
            ggml_tensor * k_pv = ggml_cont(ctx0, ggml_permute(ctx0,
                    ggml_reshape_3d(ctx0, indexer_k_b, head_size, B, n_blocks), 1, 0, 2, 3));
            cb(k_pv, "step5_indexer_csa_k_blocks", il);

            // Invalid slots: push z to -INF pre-softmax so garbage cache rows get zero weight.
            ggml_tensor * vis_v3 = ggml_reshape_3d(ctx0, vis, B, 1, n_blocks);
            ggml_tensor * z_masked = ggml_add(ctx0, z_pv,
                    ggml_scale_bias(ctx0, vis_v3, 1e30f, -1e30f));
            cb(z_masked, "step5_indexer_csa_z_masked", il);

            ggml_tensor * w = ggml_soft_max(ctx0, z_masked); // {B, head_size, n_blocks}
            cb(w, "step5_indexer_csa_w", il);
            ggml_tensor * num = ggml_sum_rows(ctx0, ggml_mul(ctx0, w, k_pv)); // {1, head, n_blocks}
            block_k = ggml_reshape_2d(ctx0, num, head_size, n_blocks);
            cb(block_k, "step5_indexer_block_k", il);
            }
        }

        // The reference scores against the e4m3-rounded summary.
        block_k = ggml_round_e4m3(ctx0, block_k);
        cb(block_k, "step5_indexer_block_k_e4m3", il);

        if (fused_try) {
            ggml_tensor * mseed = ggml_view_2d(ctx0, KQ_mask, n_kv, n_tokens, KQ_mask->nb[1], 0);
            if (band_neg) {
                mseed = ggml_add(ctx0, mseed, ggml_view_2d(ctx0, band_neg, n_kv, n_tokens, band_neg->nb[1], 0));
            }
            if (!lctx.inp_step5_cell_blk) {
                lctx.inp_step5_cell_blk = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_kv);
                cb(lctx.inp_step5_cell_blk, "step5_cell_blk", -1);
                ggml_set_input(lctx.inp_step5_cell_blk);
                ggml_build_forward_expand(gf, lctx.inp_step5_cell_blk);
            }
            ggml_tensor * topk = ggml_indexer_topk(ctx0, block_k, q_cur, w_cur, mseed,
                    lctx.inp_step5_cell_blk, GGML_UNARY_OP_RELU, (int) kept);
            if (supports_op(topk)) {
                cb(topk, "step5_indexer_topk", il);
                ggml_build_forward_expand(gf, topk);
                ggml_tensor * fm = ggml_indexer_mask_own(ctx0, KQ_mask, topk, band01);
                cb(fm, "step5_fused_mask", il);
                ggml_build_forward_expand(gf, fm);
                *fused_mask_out = fm;
                return nullptr;
            }
        }

        ggml_tensor * block_score = score_chain(block_k, n_blocks); // {n_blocks, n_tokens}
        cb(block_score, "step5_indexer_block_score", il);

        // broadcast to slots: slot b*B+r takes block b's score (repeat; a permute here would transpose the map).
        ggml_tensor * bs3 = ggml_reshape_3d(ctx0, block_score, 1, n_blocks, n_tokens);
        ggml_tensor * rep = ggml_repeat(ctx0, bs3,
                ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, B, n_blocks, n_tokens));
        slot_score = ggml_reshape_2d(ctx0, ggml_cont(ctx0, rep), n_kv, n_tokens);
    }

    ggml_tensor * indexer_score = ggml_add(ctx0, mask_f32, slot_score);
    cb(indexer_score, "step5_indexer_score", il);
    if (lctx.cparams.cb_eval) {
        ggml_build_forward_expand(gf, indexer_score);
    }
    return indexer_score;
}

ggml_cgraph * llm_build_context::build_step5() {
    ggml_cgraph * gf = new_graph_custom();
    ggml_tensor * cur;
    auto inp_pos     = build_inp_pos();

    std::fill(lctx.dsa_cache_copies.begin(), lctx.dsa_cache_copies.end(), llama_context::CacheCopy{});

    if (cparams.mtp_op_type != MTP_OP_NONE) {
        GGML_ASSERT(model.mtp && hparams.nextn_predict_layers > 0);
        GGML_ASSERT(batch.token && "Step5 MTP requires token batches");

        const int n_layer_base = hparams.n_layer > hparams.nextn_predict_layers
            ? hparams.n_layer - hparams.nextn_predict_layers : hparams.n_layer;
        const int n_heads_model = (int) hparams.nextn_predict_layers;
        const int n_heads = lctx.mtp_n_heads > 0
            ? std::max(1, std::min((int) lctx.mtp_n_heads, n_heads_model)) : n_heads_model;
        const int step = std::max(0, std::min((int) lctx.mtp_step_idx, n_heads - 1));
        const int il = n_layer_base + step;

        ggml_tensor * hidden_states = build_inp_mtp_states(n_embd);

        const bool step_independent_warmup = llm_arch_is_step5(model.arch) &&
            (cparams.mtp_op_type == MTP_OP_WARMUP ||
             cparams.mtp_op_type == MTP_OP_UPDATE_ACCEPTED) && n_heads > 1;
        if (step_independent_warmup) {
            for (int i = n_heads - 1; i >= 0; --i) {
                const int head_il = n_layer_base + i;
                const bool is_first = i == 0;
                const bool emit_logits = is_first && cparams.mtp_op_type == MTP_OP_UPDATE_ACCEPTED;
                cur = build_step5_mtp(model.layers[head_il], hidden_states, gf, inp_pos,
                        /*reduce_output=*/is_first, emit_logits);
                ggml_build_forward_expand(gf, cur);
            }
            return gf;
        }

        const bool reduce_mtp_output = cparams.mtp_op_type != MTP_OP_NONE;
        const bool emit_mtp_logits = cparams.mtp_op_type == MTP_OP_DRAFT_GEN ||
            cparams.mtp_op_type == MTP_OP_UPDATE_ACCEPTED;
        cur = build_step5_mtp(model.layers[il], hidden_states, gf, inp_pos,
                reduce_mtp_output, emit_mtp_logits);
        ggml_build_forward_expand(gf, cur);
        return gf;
    }

    auto inpL        = llm_build_inp_embd(ctx0, lctx, hparams, batch, model.tok_embd, cb);
    auto inp_out_ids = build_inp_out_ids();
    auto KQ_mask     = build_inp_KQ_mask();
    auto KQ_mask_swa = kv_self.any_compacted()
        ? build_swa_mask_for_graph(hparams.n_swa, true)
        : build_inp_KQ_mask_swa();
    const float kq_scale = 1.0f / sqrtf(float(n_embd_head_k));

    // Own-region band: band01 = 1.0 on the query's own-region visible causal cells, band_neg = -BIG there.
    ggml_tensor * band01 = nullptr;
    ggml_tensor * band_neg = nullptr;
    if (ik_sparse_enabled() && hparams.indexer_csa_block > 0) {
        const int64_t Bsel = hparams.indexer_csa_block;
        GGML_ASSERT((Bsel & (Bsel - 1)) == 0);  // pow2 B
        const int64_t n_kv_b  = KQ_mask->ne[0];
        const int64_t n_pad_b = KQ_mask->ne[1];
        GGML_ASSERT(n_kv_b % Bsel == 0);
        const int64_t nb_b = n_kv_b / Bsel;
        ggml_tensor * causal_b = ggml_view_2d(ctx0, KQ_mask, n_kv_b, n_pad_b, KQ_mask->nb[1], 0);
        if (causal_b->type != GGML_TYPE_F32) causal_b = ggml_cast(ctx0, ggml_cont(ctx0, causal_b), GGML_TYPE_F32);
        ggml_tensor * vis = ggml_step(ctx0, ggml_scale_bias(ctx0, causal_b, 1.0f, 1.0f));
        ggml_tensor * bsum = ggml_reshape_2d(ctx0,
            ggml_sum_rows(ctx0, ggml_reshape_3d(ctx0, vis, Bsel, nb_b, n_pad_b)), nb_b, n_pad_b);
        ggml_tensor * bvis = ggml_step(ctx0, ggml_scale_bias(ctx0, bsum, 1.0f, -0.5f));
        ggml_tensor * cbb  = ggml_cumsum(ctx0, bvis);
        ggml_tensor * nbvis = ggml_view_2d(ctx0, cbb, 1, n_pad_b, cbb->nb[1],
            (nb_b - 1)*cbb->nb[0]); // last cumsum element per column (nb-1 along dim0)
        ggml_tensor * rb = ggml_add(ctx0,
            ggml_sub(ctx0, ggml_repeat(ctx0, nbvis, bvis), cbb), bvis);
        ggml_tensor * lastb = ggml_step(ctx0, ggml_scale_bias(ctx0, rb, -1.0f, 1.5f));
        ggml_tensor * lastb_3d = ggml_reshape_3d(ctx0, lastb, nb_b, 1, n_pad_b);
        ggml_tensor * lastb_rep = ggml_repeat(ctx0, lastb_3d,
            ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, nb_b, Bsel, n_pad_b));
        band01 = ggml_mul(ctx0, vis, ggml_reshape_2d(ctx0,
            ggml_cont(ctx0, ggml_permute(ctx0, lastb_rep, 1, 0, 2, 3)), n_kv_b, n_pad_b));
        cb(band01, "step5_own_region_band", -1);
        band_neg = ggml_scale(ctx0, band01, -1e30f);
        if (KQ_mask->type != GGML_TYPE_F32) band_neg = ggml_cast(ctx0, band_neg, KQ_mask->type);
        cb(band01, "step5_own_region_band", -1);
    }

    const int n_layer_base = hparams.n_layer > hparams.nextn_predict_layers
        ? hparams.n_layer - hparams.nextn_predict_layers : hparams.n_layer;

    // Shared destination leaf for the per-layer sparse-mask scatter.
    ggml_tensor * sparse_base_leaf = nullptr;

    for (int il = 0; il < n_layer_base; ++il) {
        bool is_swa = hparams.swa_layers[il];
        auto & layer = const_cast<llama_layer&>(model.layers[il]);

        ggml_tensor * rope_factors = nullptr;
        const uint32_t apply_mask = hparams.rope_scaling_apply_mask;
        if ((is_swa && (apply_mask & 0x2)) || (!is_swa && (apply_mask & 0x1))) {
            rope_factors = build_rope_factors(il);
        }
        auto rope_freqs = layer.rope_freqs;
        layer.rope_freqs = nullptr;
        build_step5_indexer_kv_write(gf, il, inpL);
        ggml_tensor * fused_mask = nullptr;
        ggml_tensor * idx_score = build_step5_indexer_score(gf, il, inpL, inp_pos, KQ_mask, band01, band_neg, &fused_mask);

        // Selection: fused in-kernel topk+mask when available, else decomposed argsort + scatter.
        ggml_tensor * attn_mask = is_swa ? KQ_mask_swa : KQ_mask;
        if (!is_swa && ik_sparse_enabled()) {
            if (fused_mask) {
                attn_mask = fused_mask;
            } else if (idx_score) {
            ggml_tensor * sorted = ggml_argsort(ctx0, idx_score, GGML_SORT_ORDER_DESC);
            cb(sorted, "step5_indexer_sorted", il);
            // Block-grain selection: topk = 512 regions x 8 region_block_size = 4096 kept slots;
            // kept-count is a multiple of B so the rank cutoff snaps to whole blocks.
            const int64_t kept_slots = ik_kept_slots(hparams, KQ_mask->ne[0]);
            if (!sparse_base_leaf) {
                sparse_base_leaf = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, 1, KQ_mask->ne[0], idx_score->ne[1]);
            }
            ggml_tensor * sparse = build_deepseek2_dsa_sparse_mask(sorted, KQ_mask, kept_slots, sparse_base_leaf);
            // own-region force-append: zero the mask on the query's own-region causal cells.
            {
                const int64_t n_tok_sel = idx_score->ne[1];
                ggml_tensor * band_v = ggml_view_2d(ctx0, band01, KQ_mask->ne[0], n_tok_sel, band01->nb[1], 0);
                sparse = ggml_mul(ctx0, sparse, ggml_scale_bias(ctx0, band_v, -1.0f, 1.0f));
            }
            cb(sparse, "step5_sparse_mask", il);
            if (flash_attn) {
                sparse = build_deepseek2_dsa_fa_mask(sparse, KQ_mask);
                cb(sparse, "step5_sparse_mask_fa", il);
            }
            attn_mask = sparse;
            }
        }
        cur = build_std_attention(gf, model.layers[il].attn_norm, inpL,
                inp_pos, il == n_layer_base - 1 && n_tokens > 1 && !cparams.mtp ? inp_out_ids : nullptr,
                rope_factors, attn_mask, nullptr, nullptr, kq_scale, 0.0f, is_swa ? hparams.n_swa : 0,
                il, true, false, true);
        layer.rope_freqs = rope_freqs;

        if (model.layers[il].ffn_gate_inp == nullptr) {
            // dense FFN
            cur = llm_build_ffn(ctx0, lctx, model.layers[il].ffn_norm, cur,
                    model.layers[il].ffn_up,   NULL, NULL,
                    model.layers[il].ffn_gate, NULL, NULL,
                    model.layers[il].ffn_down, NULL, NULL,
                    nullptr,
                    LLM_FFN_SILU, LLM_FFN_PAR, cb, il, gf, true);
            cb(cur, "ffn_out", il);
        } else {
            const bool  norm_w  = hparams.expert_weights_norm;
            const float w_scale = hparams.expert_weights_scale;
            const bool  scale_w = w_scale != 0.0f;
            cur = llm_build_std_moe_ffn(ctx0, lctx, model.layers[il].ffn_norm, cur,
                    model.layers[il].ffn_gate_inp,  model.layers[il].ffn_gate_inp_b,
                    model.layers[il].ffn_up_exps,   model.layers[il].ffn_up_exps_b,
                    model.layers[il].ffn_gate_exps, model.layers[il].ffn_gate_exps_b,
                    model.layers[il].ffn_down_exps, model.layers[il].ffn_down_exps_b,
                    model.layers[il].ffn_exp_probs_b,
                    model.layers[il].ffn_up_shexp,    nullptr,
                    model.layers[il].ffn_gate_shexp,  nullptr,
                    model.layers[il].ffn_down_shexp,  nullptr,
                    n_expert, n_expert_used,
                    LLM_FFN_SILU, norm_w, scale_w, w_scale,
                    LLM_EXPERT_GATING_FUNC_SIGMOID,
                    LLM_FFN_SILU, cb, il, gf, true, model.layers[il].ffn_up_gate_exps);
        }

        cur = lctx.cvec.apply_to(ctx0, cur, il);
        cb(cur, "l_out", il);

        inpL = cur;
    }

    if (cparams.mtp) {
        ggml_tensor * mtp_embd = inpL->type == GGML_TYPE_F32 ? inpL : ggml_cast(ctx0, inpL, GGML_TYPE_F32);
        cb(mtp_embd, "result_mtp_embd", -1);
        ggml_set_output(mtp_embd);
        ggml_build_forward_expand(gf, mtp_embd);

        if (inp_out_ids) {
            inpL = ggml_get_rows(ctx0, inpL, inp_out_ids);
        }
    }

    cur = build_output(lctx, ctx0, inpL, model.output, model.output_norm, cb);
    cb(cur, "result_output", -1);

    ggml_build_forward_expand(gf, cur);

    return gf;
}

ggml_tensor * llm_build_context::build_step5_mtp(
        const llama_layer & mtp_layer,
        ggml_tensor * hidden_states_from_main_model,
        ggml_cgraph * gf,
        ggml_tensor * inp_pos,
        bool reduce_output,
        bool emit_logits,
        ggml_tensor ** hidden_out) {
    const int il = (int) (&mtp_layer - model.layers.data());

    GGML_ASSERT(mtp_layer.nextn.eh_proj && mtp_layer.nextn.enorm && mtp_layer.nextn.hnorm);
    GGML_ASSERT(mtp_layer.wq && mtp_layer.wk && mtp_layer.wv && mtp_layer.wo);

    ggml_tensor * inp_out_ids = (n_tokens > 1 && n_outputs < n_tokens) ? build_inp_out_ids() : nullptr;
    ggml_tensor * tok_embd_w = mtp_layer.nextn.embed_tokens ? mtp_layer.nextn.embed_tokens : model.tok_embd;
    ggml_tensor * tok_embd = build_inp_embd_mtp(tok_embd_w);
    ggml_tensor * cur = build_mtp_input(mtp_layer, hidden_states_from_main_model,
            tok_embd, il, "mtp_eh_proj");

    const bool is_swa = hparams.swa_layers[il];
    ggml_tensor * rope_factors = nullptr;
    const uint32_t apply_mask = hparams.rope_scaling_apply_mask;
    if ((is_swa && (apply_mask & 0x2)) || (!is_swa && (apply_mask & 0x1))) {
        rope_factors = build_rope_factors(il);
    }
    // MTP tail layers are excluded from compaction; the per-layer check keeps a future layout change correct.
    auto KQ_mask = is_swa
        ? (kv_self.is_compacted((int) il) ? build_swa_mask_for_graph(hparams.n_swa, true) : build_inp_KQ_mask_swa())
        : build_inp_KQ_mask();
    const float kq_scale = 1.0f / sqrtf(float(hparams.n_embd_head_k(il)));

    cur = build_std_attention(gf, mtp_layer.attn_norm, cur, inp_pos, nullptr,
            rope_factors, KQ_mask, nullptr, nullptr, kq_scale, 0.0f,
            is_swa ? hparams.n_swa : 0, il, true, false, true, false, false, nullptr, il);

    if (mtp_layer.ffn_gate_inp == nullptr) {
        cur = llm_build_ffn(ctx0, lctx, mtp_layer.ffn_norm, cur,
                mtp_layer.ffn_up, nullptr, nullptr,
                mtp_layer.ffn_gate, nullptr, nullptr,
                mtp_layer.ffn_down, nullptr, nullptr,
                nullptr, LLM_FFN_SILU, LLM_FFN_PAR, cb, il, gf, true);
    } else {
        cur = llm_build_std_moe_ffn(ctx0, lctx, mtp_layer.ffn_norm, cur,
                mtp_layer.ffn_gate_inp, nullptr,
                mtp_layer.ffn_up_exps, nullptr,
                mtp_layer.ffn_gate_exps, nullptr,
                mtp_layer.ffn_down_exps, nullptr,
                mtp_layer.ffn_exp_probs_b,
                mtp_layer.ffn_up_shexp, nullptr,
                mtp_layer.ffn_gate_shexp, nullptr,
                mtp_layer.ffn_down_shexp, nullptr,
                n_expert, n_expert_used,
                LLM_FFN_SILU, hparams.expert_weights_norm, hparams.expert_weights_scale != 0.0f,
                hparams.expert_weights_scale,
                (llm_expert_gating_func_type) hparams.expert_gating_func,
                LLM_FFN_SILU, cb, il, gf, true, mtp_layer.ffn_up_gate_exps);
    }

    cur = lctx.cvec.apply_to(ctx0, cur, il);
    cb(cur, "mtp_post_ffn", il);
    if (hidden_out) {
        *hidden_out = cur;
    }

    ggml_tensor * output_hidden = cur;
    if (reduce_output) {
        if (cparams.mtp_op_type != MTP_OP_NONE && n_tokens > 1) {
            output_hidden = ggml_view_2d(ctx0, cur, n_embd, 1,
                    cur->nb[1], (size_t) (n_tokens - 1) * cur->nb[1]);
        } else if (inp_out_ids) {
            output_hidden = ggml_get_rows(ctx0, cur, inp_out_ids);
        }
    }
    if (reduce_output) {
        ggml_tensor * mtp_embd = output_hidden->type == GGML_TYPE_F32 ? output_hidden : ggml_cast(ctx0, output_hidden, GGML_TYPE_F32);
        cb(mtp_embd, "result_mtp_embd", -1);
        ggml_set_output(mtp_embd);
        ggml_build_forward_expand(gf, mtp_embd);
    }

    if (!emit_logits) {
        return output_hidden;
    }

    ggml_tensor * head_norm = mtp_layer.nextn.shared_head_norm
        ? mtp_layer.nextn.shared_head_norm : model.output_norm;
    ggml_tensor * head = mtp_layer.nextn.shared_head_head
        ? mtp_layer.nextn.shared_head_head : model.output;
    GGML_ASSERT(head_norm && head);
    cur = llm_build_context::build_output(lctx, ctx0, output_hidden, head, head_norm, cb);
    cb(cur, "result_output", -1);
    return cur;
}

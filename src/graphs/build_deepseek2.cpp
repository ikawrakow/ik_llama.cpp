#include "../llama-build-context.h"
#include "../llama-model.h"
#include "../llama-context.h"

#include <vector>

// Per-rank attention for DEEPSEEK2 under -sm graph (requires -fa + -mla>=1).

ggml_tensor * llm_build_context::build_deepseek2_tp_attention(
        ggml_cgraph * gf, int il,
        ggml_tensor * inpL,
        ggml_tensor * KQ_mask, ggml_tensor * inp_pos,
        ggml_tensor * rope_cache,
        float kq_scale, float attn_factor_scaled,
        bool use_f32_attn_precision,
        bool is_lite,
        bool pp_opt) {
    if (!lctx.cparams.flash_attn || lctx.cparams.mla_attn < 1) {
        GGML_ABORT("-sm graph for MLA archs (DEEPSEEK2/GLM_DSA/MISTRAL4) requires -fa on and -mla >= 1. "
                   "Got mla_attn=%d, flash_attn=%d.",
                   (int)lctx.cparams.mla_attn, (int)lctx.cparams.flash_attn);
    }

    auto wo_split = (const ggml_split_tensor_t *)model.layers[il].wo->extra;
    GGML_ASSERT(wo_split);
    const int n_device = wo_split->n_device;

    const uint32_t n_embd_head_qk_rope = hparams.n_rot;
    const uint32_t n_embd_head_qk_nope = hparams.n_embd_head_k(il) - hparams.n_rot;
    const uint32_t kv_lora_rank        = hparams.n_lora_kv;
    const uint32_t n_embd_head_k       = hparams.n_embd_head_k(il);
    const uint32_t n_embd_head_v       = hparams.n_embd_head_v(il);

    auto cache_repl = (const ggml_split_tensor_t *)kv_self.k_l[il]->extra;
    if (!cache_repl) {
        LLAMA_LOG_ERROR("%s: no cache split for layer %d?\n", __func__, il);
    }
    GGML_ASSERT(cache_repl);
    GGML_ASSERT(cache_repl->n_device == n_device);

    std::vector<ggml_tensor *> attn_partials(n_device, nullptr);
    bool input_added = false;  // add residual once, on the first non-skipped rank

    // head_offset per rank: wo is row-split with splits[id]->ne[0] == n_head_local_id * n_embd_head_v.
    // wk_b/wv_b are per-head split (split_dim=2); each rank's tensor already holds only its heads.
    std::vector<int> head_offsets(n_device + 1, 0);
    for (int idx = 0; idx < n_device; ++idx) {
        int n_h_id = 0;
        if (wo_split->splits[idx]) {
            n_h_id = (int)(wo_split->splits[idx]->ne[0] / n_embd_head_v);
        }
        head_offsets[idx + 1] = head_offsets[idx] + n_h_id;
    }

    for (int id = 0; id < n_device; ++id) {
        if (!wo_split->splits[id]) continue;
        const int il_id = 1000 * il + id;  // unique cb() id per (layer, rank)

        auto input = get_input_tensor_sm_graph(ctx0, inpL, id);

        auto attn_norm_split = (const ggml_split_tensor_t *)model.layers[il].attn_norm->extra;
        GGML_ASSERT(attn_norm_split);
        ggml_tensor * cur = llm_build_norm(ctx0, input, hparams,
                attn_norm_split->splits[id], nullptr, LLM_NORM_RMS, cb, il_id);

        ggml_tensor * q;
        if (!is_lite) {
            auto wq_a_split = (const ggml_split_tensor_t *)model.layers[il].wq_a->extra;
            auto wq_b_split = (const ggml_split_tensor_t *)model.layers[il].wq_b->extra;
            GGML_ASSERT(wq_a_split && wq_b_split);
            q = ggml_mul_mat(ctx0, wq_a_split->splits[id], cur);
            ggml_build_forward_expand(gf, q);
            auto q_a_norm_split = (const ggml_split_tensor_t *)model.layers[il].attn_q_a_norm->extra;
            GGML_ASSERT(q_a_norm_split);
            q = llm_build_norm(ctx0, q, hparams, q_a_norm_split->splits[id], nullptr, LLM_NORM_RMS, cb, il_id);
            q = ggml_mul_mat(ctx0, wq_b_split->splits[id], q);
        } else {
            auto wq_split = (const ggml_split_tensor_t *)model.layers[il].wq->extra;
            GGML_ASSERT(wq_split);
            q = ggml_mul_mat(ctx0, wq_split->splits[id], cur);
            ggml_build_forward_expand(gf, q);
        }
        cb(q, "q", il_id);

        const int n_head_local = q->ne[0] / n_embd_head_k;

        const size_t row_size_q = ggml_row_size(q->type, n_embd_head_k);
        ggml_tensor * q_nope = ggml_view_3d(ctx0, q,
                n_embd_head_qk_nope, n_head_local, n_tokens,
                row_size_q, q->nb[1], 0);
        ggml_tensor * q_rope = ggml_view_3d(ctx0, q,
                n_embd_head_qk_rope, n_head_local, n_tokens,
                row_size_q, q->nb[1],
                ggml_row_size(q->type, n_embd_head_qk_nope));

        auto wkv_a_mqa_split = (const ggml_split_tensor_t *)model.layers[il].wkv_a_mqa->extra;
        GGML_ASSERT(wkv_a_mqa_split);
        ggml_tensor * kv_rope_compressed = ggml_mul_mat(ctx0, wkv_a_mqa_split->splits[id], cur);
        ggml_build_forward_expand(gf, kv_rope_compressed);

        ggml_tensor * k_rope = ggml_view_3d(ctx0, kv_rope_compressed,
                n_embd_head_qk_rope, 1, n_tokens,
                kv_rope_compressed->nb[1], kv_rope_compressed->nb[1],
                ggml_row_size(kv_rope_compressed->type, kv_lora_rank));
        ggml_tensor * kv_compressed = ggml_view_2d(ctx0, kv_rope_compressed,
                kv_lora_rank, n_tokens, kv_rope_compressed->nb[1], 0);

        if (rope_cache) {
            q_rope = ggml_rope_fast(ctx0, q_rope, rope_cache);
            k_rope = ggml_rope_fast(ctx0, k_rope, rope_cache);
        } else {
            q_rope = ggml_rope_ext(ctx0, q_rope, inp_pos, nullptr, n_rot, rope_type,
                    n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor_scaled, beta_fast, beta_slow);
            k_rope = ggml_rope_ext(ctx0, k_rope, inp_pos, nullptr, n_rot, rope_type,
                    n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor_scaled, beta_fast, beta_slow);
        }

        {
            auto kv_a_norm_split = (const ggml_split_tensor_t *)model.layers[il].attn_kv_a_norm->extra;
            GGML_ASSERT(kv_a_norm_split);
            kv_compressed = llm_build_norm(ctx0, kv_compressed, hparams,
                    kv_a_norm_split->splits[id], NULL, LLM_NORM_RMS, cb, il_id);
        }

        ggml_tensor * cache_local = cache_repl->splits[id];
        const auto row_size_cache = ggml_row_size(cache_local->type, kv_lora_rank + n_embd_head_qk_rope);
        ggml_tensor * cache_write_view = ggml_view_2d(ctx0, cache_local,
                cache_local->ne[0], n_tokens, row_size_cache, row_size_cache * kv_head);

        ggml_tensor * kvr = ggml_concat(ctx0, ggml_permute(ctx0, k_rope, 0, 2, 1, 3), kv_compressed, 0);
        if (cparams.k_cache_hadamard) {
            kvr = ggml_hadamard(ctx0, kvr, 64);
        }

        // Per-rank cache_copies slot.
        const int cc_idx = 2 * n_device * il + 2 * id;
        GGML_ASSERT(cc_idx + 1 < (int)lctx.cache_copies.size());
        lctx.cache_copies[cc_idx + 0].cpy  = ggml_cpy(ctx0, kvr, cache_write_view);
        lctx.cache_copies[cc_idx + 0].step = row_size_cache;
        ggml_build_forward_expand(gf, lctx.cache_copies[cc_idx + 0].cpy);

        ggml_tensor * kv_cache = ggml_view_2d(ctx0, cache_local,
                kv_lora_rank + n_embd_head_qk_rope, n_kv,
                row_size_cache, 0);
        cb(kv_cache, "kv_cache", il_id);

        // pp_opt (mla > 1, n_tokens >= 128, n_kv >= k_pp_opt_min_kv): materialize
        // per-rank K/V from the latent cache and use standard flash_attn instead of
        // FlashMLA-3 absorb.
        constexpr int k_pp_opt_min_kv = 1024;
        const bool tp_pp_opt = pp_opt
                            && (int)n_kv >= k_pp_opt_min_kv
                            && model.layers[il].wk_b
                            && model.layers[il].wv_b
                            && model.layers[il].wk_b_pp;

        ggml_tensor * kqv_2d;

        if (tp_pp_opt) {
            // Per-rank wk_b/wv_b slices already exist from distribute_mla_tensors:
            //   wk_b_local_pp: [n_embd_head_qk_nope, kv_lora_rank, n_head_local]
            //   wv_b_local_pp: [kv_lora_rank, n_embd_head_v,    n_head_local]
            auto wv_b_pp_split_raw = (const ggml_split_tensor_t *)model.layers[il].wv_b->extra;
            GGML_ASSERT(wv_b_pp_split_raw);
            ggml_tensor * wv_b_local_pp = wv_b_pp_split_raw->splits[id];

            ggml_tensor * kv_cache_nope = ggml_view_2d(ctx0, cache_local,
                    kv_lora_rank, n_kv,
                    row_size_cache,
                    ggml_row_size(cache_local->type, n_embd_head_qk_rope));
            cb(kv_cache_nope, "kv_cache_nope_pp", il_id);

            ggml_tensor * kv_cache_rope_view = ggml_view_3d(ctx0, cache_local,
                    n_embd_head_qk_rope, n_kv, 1,
                    row_size_cache, cache_local->nb[2], 0);
            cb(kv_cache_rope_view, "kv_cache_rope_pp", il_id);

            // Un-Hadamard the cache views via the fused dequant+hadamard kernel.
            // When khad_pretransformed is set, H was folded into wv_b/wk_b_pp at init,
            // so the cache_nope un-Hadamard is skipped (rope half still goes to FA via
            // concat — no wk_b multiply, no H to fold into).
            if (cparams.k_cache_hadamard) {
                kv_cache_rope_view = ggml_hadamard(ctx0, kv_cache_rope_view, 64);
                if (!model.khad_pretransformed) {
                    kv_cache_nope = ggml_hadamard(ctx0, kv_cache_nope, 64);
                }
            }

            // CUDA quantized-cache + REPEAT/CONCAT/CPY has known issues, so force F16 here.
            const auto kv_type = GGML_TYPE_F16;

            ggml_tensor repeater;
            repeater.ne[0] = n_embd_head_qk_rope;
            repeater.ne[1] = n_kv;
            repeater.ne[2] = n_head_local;
            repeater.ne[3] = 1;
            ggml_tensor * k_rope_rep;
            if (kv_cache_rope_view->type == kv_type) {
                k_rope_rep = ggml_repeat(ctx0, kv_cache_rope_view, &repeater);
            } else {
                auto kv_rope_f16 = ggml_cast(ctx0, kv_cache_rope_view, kv_type);
                k_rope_rep = ggml_repeat(ctx0, kv_rope_f16, &repeater);
            }
            cb(k_rope_rep, "k_rope_rep_pp", il_id);

            // V: wv_b_local viewed as 2D [kv_lora_rank, n_head_local * n_embd_head_v].
            // Per-rank, no cross-device transfer per call.
            auto wv_b_2d = ggml_reshape_2d(ctx0, wv_b_local_pp,
                    kv_lora_rank, n_head_local * n_embd_head_v);
            ggml_tensor * v_2d = ggml_mul_mat(ctx0, wv_b_2d, kv_cache_nope);
            cb(v_2d, "v_2d_pp", il_id);
            ggml_tensor * v_f32 = ggml_view_3d(ctx0, v_2d,
                    n_embd_head_v, n_kv, n_head_local,
                    v_2d->nb[1],
                    n_embd_head_v * v_2d->nb[0],
                    0);

            // wk_b_pp is transpose(wk_b) pre-materialized in llm_prepare_mla.
            // Shape: [kv_lora_rank, n_embd_head_qk_nope, n_head_local].
            auto wk_b_pp_split = (const ggml_split_tensor_t *)model.layers[il].wk_b_pp->extra;
            GGML_ASSERT(wk_b_pp_split);
            ggml_tensor * wk_b_pp_local = wk_b_pp_split->splits[id];
            GGML_ASSERT(wk_b_pp_local);
            ggml_tensor * wk_b_T_2d = ggml_reshape_2d(ctx0, wk_b_pp_local,
                    kv_lora_rank, n_head_local * n_embd_head_qk_nope);
            ggml_tensor * k_nope_2d = ggml_mul_mat(ctx0, wk_b_T_2d, kv_cache_nope);
            cb(k_nope_2d, "k_nope_2d_pp", il_id);
            ggml_tensor * k_nope_f32 = ggml_view_3d(ctx0, k_nope_2d,
                    n_embd_head_qk_nope, n_kv, n_head_local,
                    k_nope_2d->nb[1],
                    n_embd_head_qk_nope * k_nope_2d->nb[0],
                    0);

            ggml_tensor * v      = ggml_cast(ctx0, v_f32,      kv_type);
            ggml_tensor * k_nope = ggml_cast(ctx0, k_nope_f32, kv_type);
            ggml_build_forward_expand(gf, v);
            ggml_build_forward_expand(gf, k_nope);

            ggml_tensor * k = ggml_concat(ctx0, k_rope_rep, k_nope, 0);
            ggml_build_forward_expand(gf, k);
            cb(k, "k_full_pp", il_id);

            ggml_tensor * q = ggml_concat(ctx0, q_rope, q_nope, 0);
            q = ggml_permute(ctx0, q, 0, 2, 1, 3);
            ggml_build_forward_expand(gf, q);
            cb(q, "q_concat_pp", il_id);

            ggml_tensor * kqv = ggml_flash_attn_ext(ctx0, q, k, v, KQ_mask,
                    kq_scale, hparams.f_max_alibi_bias, 0.f);
            if (use_f32_attn_precision || q->ne[1] <= 8) {
                ggml_flash_attn_ext_set_prec(kqv, GGML_PREC_F32);
            }
            cb(kqv, "kqv_pp", il_id);

            kqv_2d = ggml_reshape_2d(ctx0, kqv, n_embd_head_v * n_head_local, n_tokens);
        } else {
            // Absorb path: FlashMLA-3 with the compressed latent cache, then project via wv_b.
            auto wk_b_split = (const ggml_split_tensor_t *)model.layers[il].wk_b->extra;
            GGML_ASSERT(wk_b_split);
            ggml_tensor * wk_b_local = wk_b_split->splits[id];

            ggml_tensor * q_nope_perm = ggml_permute(ctx0, q_nope, 0, 2, 1, 3);
            ggml_tensor * q_nope2     = ggml_mul_mat(ctx0, wk_b_local, q_nope_perm);

            ggml_tensor * q_combined = ggml_concat(ctx0,
                    ggml_permute(ctx0, q_rope, 0, 2, 1, 3), q_nope2, 0);
            if (cparams.k_cache_hadamard) {
                q_combined = ggml_hadamard(ctx0, q_combined, 64);
            }

            // FlashMLA-3 path: K = kv_cache (full latent + rope), V = kv_cache_lora (latent only)
            ggml_tensor * kv_cache_lora = ggml_view_2d(ctx0, cache_local,
                    kv_lora_rank, n_kv,
                    row_size_cache,
                    ggml_row_size(cache_local->type, n_embd_head_qk_rope));
            cb(kv_cache_lora, "kv_cache_lora", il_id);

            ggml_tensor * kqv_compressed = ggml_flash_attn_ext(ctx0,
                    q_combined, kv_cache, kv_cache_lora, KQ_mask,
                    kq_scale, hparams.f_max_alibi_bias, 0.f);
            cb(kqv_compressed, "kqv_compressed", il_id);
            if (use_f32_attn_precision) {
                ggml_flash_attn_ext_set_prec(kqv_compressed, GGML_PREC_F32);
            }
            // When khad_pretransformed is set, H is folded into wv_b. FA leaves
            // kqv_compressed in the H-encoded basis; the mul_mat(H@wv_b, kqv_encoded)
            // below collapses to wv_b^T @ kqv_unencoded by H @ H = I. Skip the
            // post-FA un-encode so the fold composes correctly.
            if (cparams.k_cache_hadamard && !model.khad_pretransformed) {
                kqv_compressed = ggml_hadamard(ctx0, kqv_compressed, 64);
            }
            kqv_compressed = ggml_permute(ctx0, kqv_compressed, 0, 2, 1, 3);

            auto wv_b_split = (const ggml_split_tensor_t *)model.layers[il].wv_b->extra;
            GGML_ASSERT(wv_b_split);
            ggml_tensor * wv_b_local = wv_b_split->splits[id];

            ggml_tensor * kqv = ggml_mul_mat(ctx0, wv_b_local, kqv_compressed);
            if (n_tokens > 1) {
                kqv = ggml_cont(ctx0, ggml_permute(ctx0, kqv, 0, 2, 1, 3));
            }
            kqv_2d = ggml_reshape_2d(ctx0, kqv, n_embd_head_v * n_head_local, n_tokens);
        }

        ggml_tensor * partial = llm_build_lora_mm(lctx, ctx0, wo_split->splits[id], kqv_2d);

        // Fold residual into the first non-skipped rank so the reduce result includes it.
        if (!input_added) {
            partial = ggml_add(ctx0, partial, input);
            input_added = true;
        }

        if (partial->ne[1] > 32 && lctx.cparams.reduce_type != GGML_TYPE_F32) {
            partial = ggml_cast(ctx0, partial, lctx.cparams.reduce_type);
        }
        ggml_build_forward_expand(gf, partial);
        attn_partials[id] = partial;
    }

    ggml_tensor * combined = ggml_reduce(ctx0, attn_partials.data(), n_device, GGML_OP_ADD);
    ggml_build_forward_expand(gf, combined);
    cb(combined, "attn_combined", il);
    return combined;
}

static inline int dsa_n_top_k(const llama_context & lctx) {
    int n_top_k = lctx.model.hparams.indexer_top_k;
    if (lctx.cparams.dsa_top_k >= 0) n_top_k = lctx.cparams.dsa_top_k;
    return n_top_k;
}
static inline bool dsa_should_use(const llama_context & lctx) {
    return lctx.cparams.mtp_op_type == MTP_OP_NONE && lctx.cparams.dsa && lctx.model.arch == LLM_ARCH_GLM_DSA && !lctx.kv_self.kr_l.empty();
}
static inline bool dsa_should_use(const llama_context & lctx, int il) {
    return dsa_should_use(lctx) && lctx.model.layers[il].indexer_attn_q_b && lctx.kv_self.kr_l.size() > (size_t) il && lctx.kv_self.kr_l[il];
}
static inline bool dsa_can_use_fast_path(const llama_context & lctx, int n_tokens, int n_kv) {
    auto n_top_k = dsa_n_top_k(lctx);
    if (n_tokens == 1 && n_top_k < n_kv && n_top_k%256 == 0 && lctx.cparams.flash_attn && (lctx.cparams.mla_attn == 1 || lctx.cparams.mla_attn == 3)) {
        // We need this check because we need to pretend that the KV cache is F32 so we can use ggml_get_rows to extract
        // the rows selected by the indexer. In order this to work, the KV cache row size must be a multiple of sizeof(float).
        auto k = lctx.kv_self.k_l[0];
        auto row_size = ggml_row_size(k->type, k->ne[0]);
        return row_size % sizeof(float) == 0;
    }
    return false;
}

// DSA lightning indexer (GLM-5.2 / DeepSeek-V3.2). CACHE-BACKED: the batch's indexer keys are
// (Hadamard-rotated and) written to a persistent per-layer indexer-key cache (kv_self.kr_l[il]) at
// kv_head, then the FULL [head_size, n_kv] cached key set is read back and scored against the current
// batch's indexer queries. This makes DECODE correct: a generated token (kv_head>0) scores against
// ALL past indexer keys, not just itself. Returns top_k [n_top_k, n_tokens] over the n_kv key axis.
//
// The Walsh-Hadamard rotation H (orthonormal, H^2==I) is applied to both indexer_q and indexer_k.
// (H q)*(H k) == q*k so it is score-preserving; its purpose is to improve the precision of the keys
// we store in the F16 indexer cache (matches the reference). Gated by cparams.dsa_indexer_hadamard.
//
// MULTI-SEQUENCE & FA: both are now handled.
//   - Multi-sequence batches (n_seq>1) are correct: each token is written to its own cache cell at
//     kv_self.head+i (per-sequence), the base block-diagonal KQ_mask drives cross-sequence keys to
//     -inf before argsort, and the attention sink is anchored per-sequence (inp_dsa_sink). Validated
//     n_seq=4 == n_seq=1 chunk-for-chunk (UPDATE 5).
//   - The FA path (-fa 1) consumes the sparse top-k mask too: build_deepseek2_dsa_fa_mask converts
//     the F32 sparse mask into the padded F16 mask the flash-attention kernel reads (UPDATE 4). It
//     does NOT fall back to the dense KQ_mask.
//   - Serving (context-shift / defrag / multi-turn seq_rm): the persistent indexer-key cache kr_l is
//     maintained by build_k_shift (delta-RoPE around the Hadamard), build_defrag (row move), and the
//     seq ops are metadata-only so kr_l rows stay matched to their cells (UPDATE 6).
ggml_tensor * llm_build_context::build_deepseek2_dsa_indexer(
        ggml_cgraph * gf,
        int il,
        ggml_tensor * qr,
        ggml_tensor * cur,
        ggml_tensor * KQ_mask,
        ggml_tensor * inp_pos) {
    const auto & layer = model.layers[il];

    const int64_t n_ihead   = hparams.indexer_n_head;
    const int64_t head_size = hparams.indexer_head_size;
    const int64_t rope_dim  = n_rot;              // n_embd_head_qk_rope
    const int64_t nope_dim  = head_size - rope_dim;

    int n_top_k = dsa_n_top_k(lctx);

    // ---- indexer_q : {head_size * n_ihead, n_tokens} ----
    ggml_tensor * indexer_q = nullptr;
    if (n_top_k < n_kv) {
        indexer_q = ggml_mul_mat(ctx0, layer.indexer_attn_q_b, qr);
        cb(indexer_q, "dsa_indexer_q", il);

        // split rope/nope along dim0, per head
        ggml_tensor * indexer_q_pe = ggml_view_3d(ctx0, indexer_q, rope_dim, n_ihead, n_tokens,
                ggml_row_size(indexer_q->type, head_size),
                ggml_row_size(indexer_q->type, head_size) * n_ihead, 0);
        ggml_tensor * indexer_q_nope = ggml_view_3d(ctx0, indexer_q, nope_dim, n_ihead, n_tokens,
                ggml_row_size(indexer_q->type, head_size),
                ggml_row_size(indexer_q->type, head_size) * n_ihead,
                ggml_row_size(indexer_q->type, rope_dim));

        indexer_q_pe = ggml_rope_ext(ctx0, indexer_q_pe, inp_pos, nullptr, n_rot,
                LLAMA_ROPE_TYPE_NEOX, n_ctx_orig, freq_base, freq_scale,
                ext_factor, attn_factor, beta_fast, beta_slow);

        // {head_size, n_ihead, n_tokens}
        indexer_q = ggml_concat(ctx0, indexer_q_pe, indexer_q_nope, 0);
        cb(indexer_q, "dsa_indexer_q_cat", il);
    }

    // ---- indexer_k : {head_size, n_tokens} (single key head, MQA) ----
    ggml_tensor * indexer_k = ggml_mul_mat(ctx0, layer.indexer_attn_k, cur);
    // LayerNorm (with weight + bias) over head_size
    indexer_k = llm_build_norm(ctx0, indexer_k, hparams, layer.indexer_k_norm, layer.indexer_k_norm_b, LLM_NORM, cb, il);
    cb(indexer_k, "dsa_indexer_k", il);

    ggml_tensor * indexer_k_pe = ggml_view_3d(ctx0, indexer_k, rope_dim, 1, n_tokens,
            ggml_row_size(indexer_k->type, head_size),
            ggml_row_size(indexer_k->type, head_size), 0);
    ggml_tensor * indexer_k_nope = ggml_view_3d(ctx0, indexer_k, nope_dim, 1, n_tokens,
            ggml_row_size(indexer_k->type, head_size),
            ggml_row_size(indexer_k->type, head_size),
            ggml_row_size(indexer_k->type, rope_dim));

    indexer_k_pe = ggml_rope_ext(ctx0, indexer_k_pe, inp_pos, nullptr, n_rot,
            LLAMA_ROPE_TYPE_NEOX, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow);

    // {head_size, 1, n_tokens}
    indexer_k = ggml_concat(ctx0, indexer_k_pe, indexer_k_nope, 0);
    cb(indexer_k, "dsa_indexer_k_cat", il);

    // ---- Walsh-Hadamard rotation (score-preserving; improves cached-K F16 precision) ----
    // nrot = largest power of 2 dividing head_size (== head_size for head_size = 128).
    // DSA_HADAMARD_DISABLE: DEBUG-ONLY env knob (no CLI surface). Default = rotation enabled.
    static const bool dsa_had_disable = getenv("DSA_HADAMARD_DISABLE") != nullptr;
    if (lctx.cparams.dsa_indexer_hadamard && !dsa_had_disable) {
        GGML_ASSERT((head_size & ~(head_size - 1)) == head_size);
        if (indexer_q) {
            indexer_q = ggml_hadamard(ctx0, indexer_q, head_size);
        }
        indexer_k = ggml_hadamard(ctx0, indexer_k, head_size);
    }

    // ---- write the batch's indexer keys into the persistent indexer-key cache at kv_head ----
    // kr_l[il] is [head_size, kv_size] (F16, MQA single head). Store {head_size, n_tokens}.
    ggml_tensor * kr_cache = kv_self.kr_l[il];
    GGML_ASSERT(kr_cache && "DSA indexer key cache not allocated");
    {
        ggml_tensor * indexer_k_2d = ggml_reshape_2d(ctx0, indexer_k, head_size, n_tokens);
        ggml_tensor * kr_view = ggml_view_2d(ctx0, kr_cache, head_size, n_tokens,
                ggml_row_size(kr_cache->type, head_size),
                ggml_row_size(kr_cache->type, head_size) * kv_head);
        ggml_tensor * kr_cpy = ggml_cpy(ctx0, indexer_k_2d, kr_view);
        // GRAPH-REUSE FIXUP REGISTRATION: the K/V cache_copies fixup in update_cache_copies()
        // re-points the K/V cache writes to the current kv_head when a graph is reused, but it
        // does NOT touch this indexer-key (kr_l) write, whose view bakes kv_head at build time.
        // Under FA the cache pads to 256, so consecutive decode ubatches keep the SAME n_kv and
        // can_reuse_graph() reuses the graph -- without this registration the kr_l write stays
        // baked at the first ubatch's kv_head, so later ubatches never write their recent index
        // keys (those cells read uninitialized -> block-max-pool/top-k drops the genuinely
        // attended recent block -> degraded/NaN sparse-FA decode). Register it like K/V so
        // update_cache_copies() patches view_offs = kv_head * step each reuse.
        // step = one index-key row = head_size * F16 = kr_cache->nb[1].
        if ((size_t) il < lctx.dsa_cache_copies.size()) {
            lctx.dsa_cache_copies[il].cpy  = kr_cpy;
            lctx.dsa_cache_copies[il].step = kr_cache->nb[1];
        }
        ggml_build_forward_expand(gf, kr_cpy);
    }

    if (!indexer_q) {
        return nullptr;
    }

    // ---- read back the full cached key set: {head_size, n_kv} ----
    ggml_tensor * cached_k = ggml_view_2d(ctx0, kr_cache, head_size, n_kv,
            ggml_row_size(kr_cache->type, head_size), 0);
    cb(cached_k, "dsa_cached_k", il);

    // ---- indexer weights : {n_ihead, n_tokens} ----
    ggml_tensor * indexer_weights = ggml_mul_mat(ctx0, layer.indexer_proj, cur);
    indexer_weights = ggml_scale(ctx0, indexer_weights, 1.0f / sqrtf(float(head_size * n_ihead)));
    cb(indexer_weights, "dsa_indexer_weights", il);

    // ---- scores ----
    // indexer_q : {head_size, n_ihead, n_tokens} -> {head_size, n_tokens, n_ihead}
    // cached_k : {head_size, n_kv} -> {head_size, n_kv, 1}; broadcasts over q's n_ihead dim.
    ggml_tensor * indexer_k_b = ggml_reshape_3d(ctx0, cached_k, head_size, n_kv, 1);

    ggml_tensor * indexer_score = ggml_view_2d(ctx0, KQ_mask, n_kv, n_tokens, KQ_mask->nb[1], 0);
    if (indexer_score->type != GGML_TYPE_F32) {
        indexer_score = ggml_cast(ctx0, indexer_score, GGML_TYPE_F32);
        cb(indexer_score, "indexer_score_f32", il);
    }
    if (indexer_q->ne[2] <= 8) {
        // This covers TG and small batches (as needed for instance for speculative decoding). It is quite a bit faster than
        // the loop over attention heads in the other branch. We limit it to a maximum of 8 tokens to limit compute buffer size.
        // For insrtance, at a context of 512k tokens (n_kv = 524288), with 32 indexer attention heads (GLM-5), the K*Q result is
        // 512 MiB for 8 tokens (64 MiB for 1 token). The ggml_relu op will be scheduled in-place, so no extra buffer needed there.
        // The ggml_mul op will be hopefully also in-place.
        // The ggml_cont(ctx0, ggml_transpose(ctx0, indexer_kq)) will require a second buffer. We use ggml_cont for now
        // because the CUDA ggml_mul implementation does not support non-contiguous tensors.
        // So, worse case for 512k context and 8 tokens is 1 GiB compute buffer.
        // A dedicated op that performs the relu, the multiplication, and the summation over attention heads would reduce
        // this by a factor of 32 (and most likely also bring non-negligible performance gains).
        // But for now, let's go with this.
        indexer_q = ggml_reshape_2d(ctx0, indexer_q, indexer_q->ne[0], indexer_q->ne[1]*indexer_q->ne[2]); // [head_size, n_ihead*n_tokens]
        auto indexer_kq = ggml_mul_mat(ctx0, indexer_k_b, indexer_q); // [n_kv, n_ihead*n_tokens]
        cb(indexer_kq, "dsa_indexer_kq", il);
        indexer_kq = ggml_relu(ctx0, indexer_kq);
        cb(indexer_kq, "dsa_indexer_kq_relu", il);
        indexer_kq = ggml_reshape_3d(ctx0, indexer_kq, indexer_kq->ne[0], n_ihead, n_tokens); // [n_kv, n_ihead, n_tokens]
        indexer_kq = ggml_cont(ctx0, ggml_transpose(ctx0, indexer_kq));                                                          // [n_ihead, n_kv, n_tokens]
        indexer_weights = ggml_reshape_3d(ctx0, indexer_weights, indexer_weights->ne[0], 1, indexer_weights->ne[1]);  // [n_ihead,    1, n_tokens]
        indexer_kq = ggml_mul(ctx0, indexer_kq, indexer_weights);
        cb(indexer_kq, "dsa_indexer_kq_w", il);
        auto score = ggml_sum_rows(ctx0, indexer_kq); // [1, n_kv, n_tokens]
        cb(score, "dsa_indexer_score", il);
        score = ggml_reshape_2d(ctx0, score, score->ne[1], score->ne[2]);
        indexer_score = ggml_add(ctx0, indexer_score, score);
        cb(indexer_score, "dsa_indexer_score", il);
    } else {
        // -fa 0: KQ_mask is the raw F32 input tensor, so the seed above still aliases it and the in-place
        // per-head accumulation below would corrupt the shared causal mask that the sparse-mask and later
        // softmax layers read back. Copy the seed first. On -fa 1 the F16 mask was already cast to a private
        // F32 buffer above (so this does not fire), and the small-batch path uses a non-inplace add.
        if (KQ_mask->type == GGML_TYPE_F32) {
            indexer_score = ggml_cont(ctx0, indexer_score);
        }
        for (int head = 0; head < n_ihead; ++head) {
            int il_cb = 1000*(il + 1) + head;
            // [1, n_tokens]
            auto w  = ggml_cont(ctx0, ggml_view_2d(ctx0, indexer_weights, 1, indexer_weights->ne[1], indexer_weights->nb[1], indexer_weights->nb[0]*head));
            cb(w, "iweights", il_cb);
            // [head_size, n_tokens]
            auto q = ggml_view_2d(ctx0, indexer_q, indexer_q->ne[0], indexer_q->ne[2], indexer_q->nb[2], indexer_q->nb[1]*head);
            // [n_kv, n_tokens]
            auto kq = ggml_mul_mat(ctx0, indexer_k_b, q);
            cb(kq, "ikq", il_cb);
            // [n_kv, n_tokens]
            kq = ggml_relu(ctx0, kq);
            cb(kq, "ikq_relu", il_cb);
            // [n_kv, n_tokens]
            auto score = ggml_mul(ctx0, kq, w);
            cb(score, "score", il_cb);
            indexer_score = ggml_add_inplace(ctx0, indexer_score, score);
            cb(indexer_score, "indexer_score", il_cb);
            ggml_build_forward_expand(gf, indexer_score);
        }
    }

    // Attention-sink force-inclusion: add a finite positive boost to each query's OWN SEQUENCE's
    // first n_sink present tokens so the sink token(s) always survive the top-k selection. Masking
    // the sink collapses most transformers; a heavily-quantized (IQ2) indexer does not reliably rank
    // it high on its own. The boost is finite, so it cannot un-mask future/causal -inf positions
    // (-inf + boost = -inf).
    //
    // MULTI-SEQUENCE: the boost MUST be per-(key,query), not a global per-key vector. With several
    // sequences packed into one ubatch (seq 0 at cache cells [0,n0), seq 1 at [n0,n1), ...), a global
    // "key index < n_sink" boost only protects sequence 0's sink; sequence 1's sink (at cell n0, not
    // cell 0) is left unprotected and gets dropped from top-k once the mask bites, collapsing it.
    // We therefore use a per-graph input tensor inp_dsa_sink {n_kv, n_tokens} (filled on the CPU from
    // kv_self.cells like the KQ_mask, in llama_set_inputs): inp_dsa_sink[j,i] = 1e20 iff key cell i
    // belongs to query j's sequence AND its pos is within [min present pos of that seq, +n_sink).
    //
    // SERVING: the anchor is each sequence's FIRST PRESENT pos, not absolute pos < n_sink. After
    // multi-turn seq_rm drops a sequence's early tokens its earliest survivor has pos >= n_sink; an
    // absolute test would then protect nothing and let the (now-)sink be masked out. For a fresh
    // sequence starting at pos 0, min(pos)==0 so the boosted set is exactly the old "cell pos <
    // n_sink" set with the same 1e20 magnitude — n_seq==1 from pos 0 stays byte-identical.
    // DSA_SINK: DEBUG-ONLY env knob (no CLI surface). Default = 1 (protect each sequence's first
    // present token from being masked out of top-k). Must stay in sync with the two fill sites.
    if (lctx.inp_dsa_sink) {
        indexer_score = ggml_add(ctx0, indexer_score, lctx.inp_dsa_sink);
        cb(indexer_score, "dsa_indexer_score_sink", il);
        ggml_build_forward_expand(gf, indexer_score);
    }

    // FULL descending argsort of the per-query scores over the n_kv axis: {n_kv, n_tokens} (I32).
    // We return the full ranking (not just the top-k view): the sparse-mask builder writes a value
    // into EVERY key slot keyed by its rank, which avoids relying on ggml_set_rows preserving an
    // uninitialized base for partially-written destinations (a CUDA in-place quirk that corrupted
    // decode when n_kv > top_k).
    //ggml_tensor * sorted = ggml_cont(ctx0, ggml_argsort(ctx0, indexer_score, GGML_SORT_ORDER_DESC));
    //ggml_tensor * sorted = ggml_argsort(ctx0, indexer_score, GGML_SORT_ORDER_DESC);
    ggml_tensor * sorted;
    if (cparams.flash_attn) {
        if (n_top_k > indexer_score->ne[0]) n_top_k = indexer_score->ne[0];
        sorted = ggml_top_k(ctx0, indexer_score, n_top_k);
        sorted = ggml_cont(ctx0, sorted);
    } else {
        sorted = ggml_argsort(ctx0, indexer_score, GGML_SORT_ORDER_DESC);
    }
    cb(sorted, "dsa_sorted", il);

    return sorted;
}

// Build an additive sparse causal mask {n_kv, n_tok} (F32): 0 for the top-k highest-scoring keys
// per query, a large negative value for the rest, then add the base causal KQ_mask so future/
// padding keys stay masked. ggml_soft_max_ext only requires mask->ne[1] >= q n_tokens, and
// n_tok == n_tokens here, so no padding is needed.
//
// `sorted` is the FULL descending argsort of the indexer scores: {n_kv, n_tok} (I32), where
// sorted[rank, j] = key index with the rank-th highest score for query j. We scatter a rank-based
// penalty into EVERY key slot:  pen(rank) = 0 if rank < n_top_k else -BIG. Because every key slot
// is written exactly once (sorted is a per-column permutation), the result does NOT depend on the
// scatter destination's initial contents — sidestepping the ggml in-place set_rows quirk where a
// partially-written CUDA destination keeps uninitialized (garbage) rows.
ggml_tensor * llm_build_context::build_deepseek2_dsa_sparse_mask(
        ggml_tensor * sorted,
        ggml_tensor * KQ_mask) {
    const int64_t n_kv_local = KQ_mask->ne[0];
    const int64_t n_tok      = sorted->ne[1];

    int64_t n_top_k = (int64_t) hparams.indexer_top_k;
    // Tuning knob: --dsa-top-k (cparams.dsa_top_k) lets us vary the kept-key count to characterize
    // selection quality. <0 means use the model's configured top_k. With the model's configured
    // top_k (2048) on heavily-quantized (IQ2_M) weights the indexer currently under-ranks some
    // critical keys; a near-n_kv value stays coherent.
    if (lctx.cparams.dsa_top_k >= 0) n_top_k = lctx.cparams.dsa_top_k;
    if (n_top_k > n_kv_local) n_top_k = n_kv_local;

    // Penalty magnitude for non-top-k keys. On the soft_max (-fa 0) path this F32 -BIG is added to
    // the score and softmaxed -> effectively -inf, while staying finite avoids -inf*0 = NaN. On the
    // FA (-fa 1) path this mask is cast to F16 (build_deepseek2_dsa_fa_mask): 1e30 saturates to the
    // F16 max (~6.5e4), which is still a large-enough negative bias to zero the key in the FA softmax
    // (the dense FA mask uses -INFINITY/F16 -inf there; our finite-but-huge value is equivalent in
    // effect and cannot produce NaN). So -BIG masks the key on BOTH paths.
    const float BIG = 1e30f;

    // rank-based penalty vector: pen[rank] = 0 for rank < n_top_k, else -BIG.  {n_kv}
    // sel = step(n_top_k - 0.5 - rank) = 1 for rank <= n_top_k-1, else 0
    ggml_tensor * rank = ggml_arange(ctx0, 0.0f, (float) n_kv_local, 1.0f);          // {n_kv} F32
    ggml_tensor * sel  = ggml_step(ctx0, ggml_scale_bias(ctx0, rank, -1.0f, (float) n_top_k - 0.5f));
    ggml_tensor * pen  = ggml_scale_bias(ctx0, sel, BIG, -BIG);                       // 0 or -BIG

    // shape penalty to {1, n_kv, n_tok} (broadcast the per-rank value across all query columns)
    pen = ggml_reshape_3d(ctx0, pen, 1, n_kv_local, 1);
    ggml_tensor * pen_b = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, 1, n_kv_local, n_tok);
    pen_b = ggml_repeat(ctx0, pen, pen_b);                                            // {1, n_kv, n_tok}

    // destination base {1, n_kv, n_tok} (contents irrelevant — fully overwritten by set_rows)
    ggml_tensor * base = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, 1, n_kv_local, n_tok);
    base = ggml_fill(ctx0, base, -BIG);

    // indices: {n_kv, n_tok, 1}. scatter pen_b[:, rank, j] into base[:, sorted[rank,j], j].
    ggml_tensor * idx = ggml_reshape_3d(ctx0, sorted, n_kv_local, n_tok, 1);
    ggml_tensor * scattered = ggml_set_rows(ctx0, base, pen_b, idx);

    // {n_kv, n_tok}
    ggml_tensor * sparse = ggml_reshape_2d(ctx0, ggml_cont(ctx0, scattered), n_kv_local, n_tok);

    // add base causal mask (first n_tok query columns) so future/padding keys stay masked
    ggml_tensor * causal = ggml_view_2d(ctx0, KQ_mask, n_kv_local, n_tok, KQ_mask->nb[1], 0);
    // see note in build_deepseek2_dsa_indexer: cast F16 (-fa 1) mask to F32 for the CPU add.
    if (causal->type != GGML_TYPE_F32) causal = ggml_cast(ctx0, ggml_cont(ctx0, causal), GGML_TYPE_F32);
    sparse = ggml_add(ctx0, sparse, causal);
    cb(sparse, "dsa_sparse_mask", -1);

    return sparse;
}

static ggml_tensor * build_deepseek2_dsa_fa_mask(const llama_context & lctx, ggml_context * ctx0, ggml_tensor * KQ_mask, ggml_tensor * sorted) {
    GGML_ASSERT(KQ_mask && KQ_mask->type == GGML_TYPE_F16);
    GGML_ASSERT(sorted && sorted->type == GGML_TYPE_I32);
    GGML_ASSERT(KQ_mask->ne[1] >= sorted->ne[1]);

    int n_top_k = (int64_t) lctx.model.hparams.indexer_top_k;
    if (lctx.cparams.dsa_top_k >= 0) n_top_k = lctx.cparams.dsa_top_k;

    int n_kv_local = KQ_mask->ne[0];
    if (n_top_k >= n_kv_local) {
        return KQ_mask;
    }

    GGML_ASSERT(sorted->ne[1] == lctx.inp_mask_inf->ne[1]);
    auto top_k = ggml_view_2d(ctx0, sorted, n_top_k, sorted->ne[1], sorted->nb[1], 0);
    auto mask32 = ggml_blend(ctx0, lctx.inp_mask_inf, top_k, 0.0f);
    if (KQ_mask->ne[1] == mask32->ne[1]) {
        auto mask16 = ggml_add(ctx0, KQ_mask, mask32);
        return mask16;
    }
    auto kq1 = ggml_view_2d(ctx0, KQ_mask, KQ_mask->ne[0], mask32->ne[1], KQ_mask->nb[1], 0);
    auto kq2 = ggml_view_2d(ctx0, KQ_mask, KQ_mask->ne[0], KQ_mask->ne[1] - mask32->ne[1], KQ_mask->nb[1], mask32->ne[1]*KQ_mask->nb[1]);
    kq1 = ggml_add(ctx0, kq1, mask32);
    auto mask16 = ggml_concat(ctx0, kq1, kq2, 1);
    return mask16;
}


// Adapt the (F32, unpadded {n_kv, n_tokens}) sparse mask for ggml_flash_attn_ext, which on this fork
// requires the mask to be F16, contiguous, and padded in ne[1] to GGML_PAD(n_queries, GGML_KQ_MASK_PAD)
// (build_inp_KQ_mask creates the dense -fa 1 mask exactly that way). We:
//   1) cast the sparse mask to F16,
//   2) concat the dense FA mask's padding rows [n_tok, n_pad) (already F16, causal -inf for the
//      non-existent padded queries) onto the bottom so ne[1] matches the dense mask,
//   3) ggml_cont so the result is contiguous (the FA assert requires it).
// The padded rows feed only the discarded outputs of padded query slots, so reusing the dense mask's
// padding region is both correct and the cheapest way to get the exact dense shape.
ggml_tensor * llm_build_context::build_deepseek2_dsa_fa_mask(
        ggml_tensor * sparse,
        ggml_tensor * KQ_mask) {
    const int64_t n_kv_local = KQ_mask->ne[0];
    const int64_t n_tok      = sparse->ne[1];
    const int64_t n_pad      = KQ_mask->ne[1];   // GGML_PAD(n_tokens, GGML_KQ_MASK_PAD)

    GGML_ASSERT(KQ_mask->type == GGML_TYPE_F16 && "FA dense KQ_mask expected F16 on -fa 1");

    ggml_tensor * fa_mask;
    if (n_pad > n_tok) {
        // dense padding rows: KQ_mask columns [n_tok, n_pad) -> {n_kv, n_pad - n_tok} (F16 view)
        ggml_tensor * pad = ggml_view_2d(ctx0, KQ_mask, n_kv_local, n_pad - n_tok,
                KQ_mask->nb[1], KQ_mask->nb[1] * n_tok);
        // CPU ggml_concat only supports F16 along dim 0 (concat_any); the dim-1 row concat must be
        // done in F32 (concat_f32 handles all dims), then cast the padded result to F16. On CUDA the
        // F16 dim-1 concat is supported, so this path only needed adapting for the CPU build.
        ggml_tensor * pad_f32 = ggml_cast(ctx0, ggml_cont(ctx0, pad), GGML_TYPE_F32);
        ggml_tensor * fa_f32  = ggml_concat(ctx0, sparse, pad_f32, 1);    // {n_kv, n_pad} F32
        fa_mask = ggml_cast(ctx0, fa_f32, GGML_TYPE_F16);                 // {n_kv, n_pad} F16
    } else {
        fa_mask = ggml_cast(ctx0, sparse, GGML_TYPE_F16);
    }
    fa_mask = ggml_cont(ctx0, fa_mask);
    cb(fa_mask, "dsa_fa_mask", -1);
    return fa_mask;
}

// Layer-mode attention path (non-TP). Mirrors build_deepseek2_tp_attention's interface.
ggml_tensor * llm_build_context::build_deepseek2_layer_attention(
        ggml_cgraph * gf, int il,
        ggml_tensor * inpL,
        ggml_tensor * KQ_mask, ggml_tensor * inp_pos,
        ggml_tensor * rope_cache,
        float kq_scale, float attn_factor_scaled,
        bool use_f32_attn_precision,
        bool is_lite,
        bool pp_opt) {
    const uint32_t n_embd_head_qk_rope = hparams.n_rot;
    const uint32_t n_embd_head_qk_nope = hparams.n_embd_head_k(0) - hparams.n_rot;
    const uint32_t kv_lora_rank = hparams.n_lora_kv;
    const uint32_t q_lora_rank  = hparams.n_lora_q;
    ggml_tensor * cur;

    // norm
    cur = llm_build_norm(ctx0, inpL, hparams, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, cb, il);
    cb(cur, "attn_norm", il);

    // DSA lightning indexer (GLM-5.2 / DeepSeek-V3.2). Built below from the q_lora latent
    // and used to construct a sparse causal mask. Defaults to the dense KQ_mask.
    //  - sparse_mask    : F32 additive sparse mask for the soft_max (-fa 0) path.
    //  - sparse_mask_fa : F16, padded variant for the ggml_flash_attn_ext (-fa 1) path.
    // Both default to the dense KQ_mask so non-DSA / disabled builds are unchanged.
    ggml_tensor * sparse_mask    = KQ_mask;
    ggml_tensor * sparse_mask_fa = KQ_mask;

    bool use_dsa       = dsa_should_use(lctx, il);
    bool dsa_fast_path = use_dsa && dsa_can_use_fast_path(lctx, n_tokens, n_kv);

    // self_attention
    {
        ggml_tensor * q = nullptr;
        ggml_tensor * kv_rope_compressed = nullptr;
        ggml_tensor * q_rope;
        ggml_tensor * q_nope;
        ggml_tensor * k_rope;
        ggml_tensor * kv_compressed;
        if (model.layers[il].wkq_a_mqa) {
            auto mqa = ggml_mul_mat(ctx0, model.layers[il].wkq_a_mqa, cur);
            cb(mqa, "mqa", il);
            size_t qnb1;
            if (!is_lite) {
                q = ggml_view_2d(ctx0, mqa, q_lora_rank, n_tokens, mqa->nb[1], 0);
                q = llm_build_norm(ctx0, q, hparams, model.layers[il].attn_q_a_norm, NULL, LLM_NORM_RMS, cb, il);
                q = ggml_mul_mat(ctx0, model.layers[il].wq_b, q);
                qnb1 = q->nb[1];
                cb(q, "q", il);
                kv_rope_compressed = ggml_view_2d(ctx0, mqa, kv_lora_rank + n_embd_head_qk_rope, n_tokens, mqa->nb[1],
                        q_lora_rank*ggml_element_size(mqa));
            } else {
                q = ggml_view_2d(ctx0, mqa, n_embd_k_gqa, n_tokens, mqa->nb[1], 0);
                kv_rope_compressed = ggml_view_2d(ctx0, mqa, kv_lora_rank + n_embd_head_qk_rope, n_tokens, mqa->nb[1],
                        n_embd_k_gqa*ggml_element_size(mqa));
                qnb1 = mqa->nb[1];
            }
            q_nope = ggml_view_3d(ctx0, q, n_embd_head_qk_nope, n_head, n_tokens,
                ggml_row_size(q->type, hparams.n_embd_head_k(il)), qnb1, 0);
            q_rope = ggml_view_3d(ctx0, q, n_embd_head_qk_rope, n_head, n_tokens,
                ggml_row_size(q->type, hparams.n_embd_head_k(il)), qnb1, ggml_row_size(q->type, n_embd_head_qk_nope));
            k_rope = ggml_view_3d(ctx0, kv_rope_compressed, n_embd_head_qk_rope, 1, n_tokens,
                    mqa->nb[1], mqa->nb[1], ggml_row_size(kv_rope_compressed->type, kv_lora_rank));
            kv_compressed = ggml_view_2d(ctx0, kv_rope_compressed, kv_lora_rank, n_tokens, mqa->nb[1], 0);
        }
        else {
            if (!is_lite) {
                q = ggml_mul_mat(ctx0, model.layers[il].wq_a, cur);
                cb(q, "q", il);

                kv_rope_compressed = ggml_mul_mat(ctx0, model.layers[il].wkv_a_mqa, cur);
                cb(kv_rope_compressed, "kv_rope_compressed", il);

                ggml_build_forward_expand(gf, q);
                ggml_build_forward_expand(gf, kv_rope_compressed);

                q = llm_build_norm(ctx0, q, hparams, model.layers[il].attn_q_a_norm, NULL, LLM_NORM_RMS, cb, il);
                cb(q, "q", il);

                // DSA lightning indexer (cache-backed): score the q_lora latent against the persistent
                // indexer-key cache over the full n_kv, then build a sparse top-k causal mask. Correct
                // for prefill AND decode (single sequence). Gate: --dsa opt-in (off by default) +
                // GLM_DSA arch + indexer tensors + cache. When off, the model runs the dense MLA path,
                // byte-identical to a build without this feature.
                if (use_dsa) {
                    // GLM-5.2 IndexShare: "full" layers compute their own lightning-indexer top-k;
                    // "shared" layers reuse the previous full layer's top-k (transformers reference:
                    // shared layer indexer=None, topk_indices=prev_topk_indices). The full/shared map is
                    // hparams.indexer_is_full (GGUF metadata or derived config rule). At a given step all
                    // layers share the same n_kv/n_tokens, so a full layer's argsort is valid to reuse.
                    if (hparams.indexer_is_full[il]) {
                        dsa_last_full_sorted = build_deepseek2_dsa_indexer(gf, il, q, cur, KQ_mask, inp_pos);
                        if (dsa_last_full_sorted && !dsa_fast_path) {
                            if (lctx.cparams.flash_attn) {
                                last_sparse_mask_fa = ::build_deepseek2_dsa_fa_mask(lctx, ctx0, KQ_mask, dsa_last_full_sorted);
                            } else {
                                last_sparse_mask = build_deepseek2_dsa_sparse_mask(dsa_last_full_sorted, KQ_mask);
                            }
                        }
                    }
                    if (dsa_last_full_sorted && !dsa_fast_path) {
                        if (lctx.cparams.flash_attn) {
                            GGML_ASSERT(last_sparse_mask_fa);
                            sparse_mask_fa = last_sparse_mask_fa;
                        } else {
                            GGML_ASSERT(last_sparse_mask);
                            sparse_mask = last_sparse_mask;
                        }
                    }
                }

                q = ggml_mul_mat(ctx0, model.layers[il].wq_b, q);
                cb(q, "q", il);
            } else {
                q = ggml_mul_mat(ctx0, model.layers[il].wq, cur);
                cb(q, "q", il);

                kv_rope_compressed = ggml_mul_mat(ctx0, model.layers[il].wkv_a_mqa, cur);
                cb(kv_rope_compressed, "kv_rope_compressed", il);

                ggml_build_forward_expand(gf, q);
                ggml_build_forward_expand(gf, kv_rope_compressed);
            }

            q_nope = ggml_view_3d(ctx0, q, n_embd_head_qk_nope, n_head, n_tokens,
                    ggml_row_size(q->type, hparams.n_embd_head_k(il)),
                    ggml_row_size(q->type, hparams.n_embd_head_k(il) * n_head), 0);

            q_rope = ggml_view_3d(ctx0, q, n_embd_head_qk_rope, n_head, n_tokens,
                    ggml_row_size(q->type, hparams.n_embd_head_k(il)),
                    ggml_row_size(q->type, hparams.n_embd_head_k(il) * n_head),
                    ggml_row_size(q->type, n_embd_head_qk_nope));

            k_rope = ggml_view_3d(ctx0, kv_rope_compressed, n_embd_head_qk_rope, 1, n_tokens,
                    kv_rope_compressed->nb[1],
                    kv_rope_compressed->nb[1],
                    ggml_row_size(kv_rope_compressed->type, kv_lora_rank));

            kv_compressed = ggml_view_2d(ctx0, kv_rope_compressed, kv_lora_rank, n_tokens,
                    kv_rope_compressed->nb[1], 0);
        }
        cb(q_nope, "q_nope", il);
        cb(q_rope, "q_rope", il);
        cb(k_rope, "k_rope", il);
        cb(kv_compressed, "kv_compressed", il);

        ggml_build_forward_expand(gf, q_rope);
        ggml_build_forward_expand(gf, k_rope);
        if (rope_cache) {
            q_rope = ggml_rope_fast(ctx0, q_rope, rope_cache);
            k_rope = ggml_rope_fast(ctx0, k_rope, rope_cache);
        } else {
            q_rope = ggml_rope_ext(ctx0, q_rope, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor_scaled, beta_fast, beta_slow);

            k_rope = ggml_rope_ext(ctx0, k_rope, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor_scaled, beta_fast, beta_slow);
        }
        cb(q_rope, "q_rope", il);
        cb(k_rope, "k_rope", il);
        ggml_build_forward_expand(gf, q_rope);
        ggml_build_forward_expand(gf, k_rope);

        kv_compressed = llm_build_norm(ctx0, kv_compressed, hparams, model.layers[il].attn_kv_a_norm, NULL, LLM_NORM_RMS, cb, il);
        cb(kv_compressed, "kv_compressed", il);

        if (lctx.cparams.mla_attn) {

            ggml_tensor * kv_cache_trans = nullptr;

            if (lctx.cparams.mla_attn == 1 && !lctx.cparams.flash_attn) {
                ggml_tensor * kv_cache_trans_view = ggml_view_2d(ctx0, kv_self.v_l[il], n_tokens, kv_lora_rank,
                        ggml_row_size(kv_self.v_l[il]->type, kv_self.size), ggml_row_size(kv_self.v_l[il]->type, kv_head));
                cb(kv_cache_trans_view, "kv_cache_trans_view", il);

                // note: storing transposed c^KV in the transposed KV cache
                ggml_build_forward_expand(gf, ggml_cpy(ctx0, ggml_transpose(ctx0, kv_compressed), kv_cache_trans_view));

                kv_cache_trans = ggml_view_2d(ctx0, kv_self.v_l[il],
                        n_kv, kv_lora_rank,
                        ggml_row_size(kv_self.v_l[il]->type, kv_self.size),
                        0);
                cb(kv_cache_trans, "kv_cache_trans", il);
            }

            //ggml_tensor * kvr = ggml_concat(ctx0, kv_compressed, ggml_permute(ctx0, k_rope, 0, 2, 1, 3), 0);
            ggml_tensor * kvr = ggml_concat(ctx0, ggml_permute(ctx0, k_rope, 0, 2, 1, 3), kv_compressed, 0);
            cb(kvr, "kvr", il);

            auto row_size = ggml_row_size(kv_self.k_l[il]->type, kv_lora_rank + n_embd_head_qk_rope);
            ggml_tensor * kv_cache_view = ggml_view_2d(ctx0, kv_self.k_l[il], kv_self.k_l[il]->ne[0], n_tokens,
                    row_size, row_size*kv_head);
            lctx.cache_copies[2*il+0].cpy = ggml_cpy(ctx0, kvr, kv_cache_view);
            lctx.cache_copies[2*il+0].step = row_size;
            ggml_build_forward_expand(gf, lctx.cache_copies[2*il+0].cpy);
            ggml_tensor * kv_cache = ggml_view_2d(ctx0, kv_self.k_l[il],
                    kv_lora_rank + n_embd_head_qk_rope, n_kv,
                    ggml_row_size(kv_self.k_l[il]->type, kv_lora_rank + n_embd_head_qk_rope), 0);
            cb(kv_cache, "kv_cache", il);

            ggml_tensor * kqv;

            if (lctx.cparams.mla_attn > 1 && lctx.cparams.flash_attn && pp_opt) { // PP for mla=2,3

                auto kv_cache_nope = ggml_view_2d(ctx0, kv_self.k_l[il], kv_lora_rank, n_kv, kv_self.k_l[il]->nb[1],
                        ggml_row_size(kv_self.k_l[il]->type, n_embd_head_qk_rope));

                auto kv_f32_size = model.layers[il].wkv_b->ne[1] * kv_cache_nope->ne[1] * sizeof(float) / (1024*1024);
                int n_max_head = n_head;
                if (cparams.attn_max_batch > 0 && kv_f32_size > cparams.attn_max_batch) {
                    n_max_head = 1;
                    for (int niter = 2; niter < n_head; ++niter) {
                        if (n_head % niter == 0 && kv_f32_size/niter <= cparams.attn_max_batch) {
                            n_max_head = n_head/niter;
                            break;
                        }
                    }
                }
                GGML_ASSERT(n_head % n_max_head == 0);

                auto n_per_head = model.layers[il].wkv_b->ne[1] / n_head;

                auto kv_cache_rope = ggml_view_3d(ctx0, kv_self.k_l[il], n_embd_head_qk_rope, n_kv, 1,
                        kv_self.k_l[il]->nb[1], kv_self.k_l[il]->nb[2], 0); //ggml_row_size(kv_self.k_l[il]->type, kv_lora_rank));

                // There is still an issue with one or more of the ops GGML_OP_REPEAT, GGML_OP_CONCAT, GGML_OP_CPY on CUDA when
                // the KV cache is quantized. Hence, in that case we will simply use fp16 for now.
                // The downside of the following line is that fp16 will be used even if attention is computed on the CPU
                // if the build is with CUDA enabled.
                auto kv_type = lctx.backends.size() == 1 && lctx.backends.front() == lctx.backend_cpu ? kv_self.k_l[il]->type : GGML_TYPE_F16;

                ggml_tensor repeater;
                repeater.ne[0] = n_embd_head_qk_rope; repeater.ne[1] = n_kv; repeater.ne[2] = n_max_head; repeater.ne[3] = 1;
                ggml_tensor * k_rope;
                if (kv_cache_rope->type == kv_type) {
                    k_rope = ggml_repeat(ctx0, kv_cache_rope, &repeater);
                } else {
                    auto kv_cache_rope_f16 = ggml_cast(ctx0, kv_cache_rope, GGML_TYPE_F16);
                    k_rope = ggml_repeat(ctx0, kv_cache_rope_f16, &repeater);
                }
                cb(k_rope, "k_rope", il);

                //auto q = ggml_concat(ctx0, q_nope, q_rope, 0);
                auto q = ggml_concat(ctx0, q_rope, q_nope, 0);
                q = ggml_permute(ctx0, q, 0, 2, 1, 3);
                cb(q, "q_concat", il);

                ggml_build_forward_expand(gf, q);

                for (int iter = 0; iter < n_head/n_max_head; ++iter) {

                    auto wkv_b = ggml_view_2d(ctx0, model.layers[il].wkv_b, model.layers[il].wkv_b->ne[0], n_per_head*n_max_head,
                            model.layers[il].wkv_b->nb[1], model.layers[il].wkv_b->nb[1]*n_per_head*n_max_head*iter);

                    auto kv_f32 = ggml_mul_mat(ctx0, wkv_b, kv_cache_nope);
                    cb(kv_f32, "kv_f32", il);

                    auto v_f32 = ggml_view_3d(ctx0, kv_f32, hparams.n_embd_head_v_full, n_kv, n_max_head,
                            ggml_row_size(kv_f32->type, n_max_head * (n_embd_head_qk_nope + hparams.n_embd_head_v_full)),
                            ggml_row_size(kv_f32->type, n_embd_head_qk_nope + hparams.n_embd_head_v_full),
                            ggml_row_size(kv_f32->type, n_embd_head_qk_nope));
                    cb(v_f32, "v_f32", il);

                    auto k_nope_f32 = ggml_view_3d(ctx0, kv_f32, n_embd_head_qk_nope, n_kv, n_max_head,
                            ggml_row_size(kv_f32->type, n_max_head * (n_embd_head_qk_nope + hparams.n_embd_head_v_full)),
                            ggml_row_size(kv_f32->type, n_embd_head_qk_nope + hparams.n_embd_head_v_full), 0);
                    cb(k_nope_f32, "k_nope_f32", il);

                    auto v = ggml_cast(ctx0, v_f32, kv_type);
                    cb(v, "v", il);

                    auto k_nope = ggml_cast(ctx0, k_nope_f32, kv_type);
                    cb(k_nope, "k_nope", il);

                    ggml_build_forward_expand(gf, k_nope);
                    ggml_build_forward_expand(gf, v);

                    //auto k = ggml_concat(ctx0, k_nope, k_rope, 0);
                    auto k = ggml_concat(ctx0, k_rope, k_nope, 0);
                    cb(k, "k", il);

                    ggml_build_forward_expand(gf, k);

                    auto q_iter = ggml_view_3d(ctx0, q, q->ne[0], q->ne[1], n_max_head,
                            q->nb[1], q->nb[2], q->nb[2]*n_max_head*iter);

                    kqv = ggml_flash_attn_ext(ctx0, q_iter, k, v, sparse_mask_fa, kq_scale, hparams.f_max_alibi_bias, 0.f);
                    if (use_f32_attn_precision || q->ne[1] <= 8) {
                        ggml_flash_attn_ext_set_prec(kqv, GGML_PREC_F32);
                    }
                    if (use_dsa && dsa_last_full_sorted) {
                        kqv->src[5] = dsa_last_full_sorted;
                    }
                    cb(kqv, "kqv", il);

                    if (iter == 0) {
                        cur = ggml_reshape_2d(ctx0, kqv, n_embd_head_v*n_max_head, n_tokens);
                    } else {
                        cur = ggml_concat(ctx0, cur, ggml_reshape_2d(ctx0, kqv, n_embd_head_v*n_max_head, n_tokens), 0);
                    }
                    ggml_build_forward_expand(gf, cur);

                }

            }
            else {

                ggml_tensor * kqv_compressed = nullptr;

                //auto wkv_b = model.layers[il].wkv_b;
                auto wk_b = model.layers[il].wk_b->ne[1] == kv_lora_rank ? model.layers[il].wk_b
                    : ggml_reshape_3d(ctx0, model.layers[il].wk_b, n_embd_head_qk_nope, kv_lora_rank, n_head);

                q_nope = ggml_permute(ctx0, q_nope, 0, 2, 1, 3);
                cb(q_nope, "q_nope_perm", il);

                struct ggml_tensor * q_nope2 = ggml_mul_mat(ctx0, wk_b, q_nope);
                cb(q_nope2, "q_nope2", il);

                //ggml_tensor * q = ggml_concat(ctx0, q_nope2, ggml_permute(ctx0, q_rope, 0, 2, 1, 3), 0);
                ggml_tensor * q = ggml_concat(ctx0, ggml_permute(ctx0, q_rope, 0, 2, 1, 3), q_nope2, 0);
                cb(q, "q", il);

                if (lctx.cparams.flash_attn && (lctx.cparams.mla_attn == 1 || lctx.cparams.mla_attn == 3)) {

                    if (dsa_last_full_sorted && dsa_fast_path) {
                        auto row_size = ggml_row_size(kv_self.k_l[il]->type, kv_self.k_l[il]->ne[0]);
                        GGML_ASSERT(row_size % sizeof(float) == 0);
                        GGML_ASSERT(dsa_last_full_sorted);
                        GGML_ASSERT(ggml_nrows(dsa_last_full_sorted) == 1);
                        auto k32 = ggml_reshape_4d_ext(ctx0, kv_self.k_l[il], GGML_TYPE_F32, row_size/sizeof(float),
                                kv_self.k_l[il]->ne[1], kv_self.k_l[il]->ne[2], kv_self.k_l[il]->ne[3]);
                        k32 = ggml_get_rows(ctx0, k32, dsa_last_full_sorted);
                        auto k = ggml_reshape_4d_ext(ctx0, k32, kv_self.k_l[il]->type, kv_self.k_l[il]->ne[0], k32->ne[1], k32->ne[2], k32->ne[3]);
                        auto v = ggml_view_2d(ctx0, k, kv_lora_rank, k->ne[1], k->nb[1], ggml_row_size(k->type, n_embd_head_qk_rope));
                        kqv_compressed = ggml_flash_attn_ext(ctx0, q, k, v, dsa_tg_fast_mask, kq_scale, hparams.f_max_alibi_bias, 0.f);
                    } else {
                        ggml_tensor * kv_cache_lora = ggml_view_2d(ctx0, kv_self.k_l[il],
                                kv_lora_rank, n_kv,
                                ggml_row_size(kv_self.k_l[il]->type, kv_lora_rank + n_embd_head_qk_rope),
                                ggml_row_size(kv_self.k_l[il]->type, n_embd_head_qk_rope));
                        cb(kv_cache_lora, "kv_cache_lora", il);

                        kqv_compressed = ggml_flash_attn_ext(ctx0, q, kv_cache, kv_cache_lora, sparse_mask_fa, kq_scale, hparams.f_max_alibi_bias, 0.f);
                    }
                    cb(kqv_compressed, "kqv_compressed", il);

                    if (use_f32_attn_precision) {
                        ggml_flash_attn_ext_set_prec(kqv_compressed, GGML_PREC_F32);
                    }
                    if (use_dsa && dsa_last_full_sorted) {
                        kqv_compressed->src[5] = dsa_last_full_sorted;
                    }

                    kqv_compressed = ggml_permute(ctx0, kqv_compressed, 0, 2, 1, 3);
                    cb(kqv_compressed, "kqv_compressed_perm", il);
                }
                else {
                    if (lctx.cparams.mla_attn > 1) {
                        ggml_tensor * kv_cache_lora = ggml_view_2d(ctx0, kv_self.k_l[il],
                                kv_lora_rank, n_kv,
                                ggml_row_size(kv_self.k_l[il]->type, kv_lora_rank + n_embd_head_qk_rope),
                                ggml_row_size(kv_self.k_l[il]->type, n_embd_head_qk_rope));
                        cb(kv_cache, "kv_cache_lora", il);

                        kv_cache_trans = ggml_cont(ctx0, ggml_transpose(ctx0, kv_cache_lora));
                        cb(kv_cache_trans, "kv_cache_trans", il);
                    }

                    auto kq_size = kv_cache->ne[1]*q->ne[1]*q->ne[2]*sizeof(float)/(1024*1024); // K*Q in MiB
                    if (lctx.cparams.attn_max_batch <= 0 || lctx.cparams.attn_max_batch >= kq_size) {
                        if (!pp_opt) {
                            q = ggml_permute(ctx0, q, 0, 2, 1, 3);
                            cb(q, "q_perm", il);
                        }

                        ggml_tensor * kq = ggml_mul_mat(ctx0, kv_cache, q);
                        if (kv_cache->ne[1] < 256) {
                            ggml_mul_mat_set_prec(kq, GGML_PREC_F32);
                        }
                        cb(kq, "kq", il);

                        if (!pp_opt) {
                            kq = ggml_cont(ctx0, ggml_permute(ctx0, kq, 0, 2, 1, 3));
                            cb(kq, "kq_perm", il);
                        }

                        kq = ggml_soft_max_ext(ctx0, kq, sparse_mask, kq_scale, hparams.f_max_alibi_bias);
                        cb(kq, "kq_soft_max_ext", il);

                        if (!pp_opt) {
                            kq = ggml_permute(ctx0, kq, 0, 2, 1, 3);
                            cb(kq, "kq_soft_max_ext_perm", il);
                        }

                        kqv_compressed = ggml_mul_mat(ctx0, kv_cache_trans, kq);
                        cb(kqv_compressed, "kqv_compressed", il);

                        if (!pp_opt) {
                            kqv_compressed = ggml_permute(ctx0, kqv_compressed, 0, 2, 1, 3);
                            cb(kqv_compressed, "kqv_compressed_perm", il);
                        }

                    } else {

                        int n_step = (kq_size + lctx.cparams.attn_max_batch - 1)/lctx.cparams.attn_max_batch;
                        n_step = std::min(n_step, int(q->ne[2]));
                        int n_per_step = (q->ne[2] + n_step - 1)/n_step;

                        for (int i_head = 0; i_head < q->ne[2]; i_head += n_per_step) {
                            int this_ne12 = i_head + n_per_step <= q->ne[2] ? n_per_step : q->ne[2] - i_head;
                            ggml_tensor * q_i = ggml_view_3d(ctx0, q, q->ne[0], q->ne[1], this_ne12, q->nb[1], q->nb[2], q->nb[2]*i_head);
                            ggml_tensor * kq_i = ggml_mul_mat(ctx0, kv_cache, q_i);
                            kq_i = ggml_soft_max_ext(ctx0, kq_i, sparse_mask, kq_scale, hparams.f_max_alibi_bias);
                            ggml_tensor * kqv_i = ggml_mul_mat(ctx0, kv_cache_trans, kq_i);
                            if (i_head == 0) {
                                kqv_compressed = kqv_i;
                            } else {
                                kqv_compressed = ggml_concat(ctx0, kqv_compressed, kqv_i, 2);
                            }
                            ggml_build_forward_expand(gf, kqv_compressed);
                        }
                        cb(kqv_compressed, "kqv_compressed", il);
                    }
                }

                auto wv_b = model.layers[il].wv_b;
                if (wv_b->ne[1] != n_embd_head_v) {
                    wv_b = ggml_reshape_3d(ctx0, wv_b, kv_lora_rank, n_embd_head_v, n_head);
                    cb(wv_b, "wv_b", il);
                }
                // There is an issue with quantized GEMV on CUDA when the left operand (the matrix) is
                // not contiguous. So, for now, we create wv_b during model loading and use that
                // instead of the commented out 3D view below.
                //auto wv_b = ggml_view_3d(ctx0, wkv_b, kv_lora_rank, n_embd_head_v, n_head,
                //        wkv_b->nb[1], wkv_b->nb[1]*(n_embd_head_v + n_embd_head_qk_nope),
                //        wkv_b->nb[1]*n_embd_head_qk_nope);
                //cb(wv_b, "wv_b", il);

                kqv = ggml_mul_mat(ctx0, wv_b, kqv_compressed);
                cb(kqv, "kqv", il);

                if (n_tokens > 1) {
                    kqv = ggml_cont(ctx0, ggml_permute(ctx0, kqv, 0, 2, 1, 3));
                    cb(kqv, "kqv_perm", il);
                }
                cur = ggml_reshape_2d(ctx0, kqv, n_embd_head_v*n_head, n_tokens);
                cb(cur, "kqv_2d", il);

            }

            ggml_build_forward_expand(gf, cur);

            cur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wo, cur);
            cb(cur, "kqv_out", il);

        }
        else {

            // {kv_lora_rank, n_head * (n_embd_head_qk_nope + n_embd_head_v)} * {kv_lora_rank, n_tokens} -> {n_head * (n_embd_head_qk_nope + n_embd_head_v), n_tokens}
            struct ggml_tensor * kv = ggml_mul_mat(ctx0, model.layers[il].wkv_b, kv_compressed);
            cb(kv, "kv", il);

            // split into {n_head * n_embd_head_qk_nope, n_tokens}
            struct ggml_tensor * k_nope = ggml_view_3d(ctx0, kv, n_embd_head_qk_nope, n_head, n_tokens,
                    ggml_row_size(kv->type, n_embd_head_qk_nope + hparams.n_embd_head_v_full),
                    ggml_row_size(kv->type, n_head * (n_embd_head_qk_nope + hparams.n_embd_head_v_full)),
                    0);
            cb(k_nope, "k_nope", il);

            // and {n_head * n_embd_head_v, n_tokens}
            struct ggml_tensor * v_states = ggml_view_3d(ctx0, kv, hparams.n_embd_head_v_full, n_head, n_tokens,
                    ggml_row_size(kv->type, (n_embd_head_qk_nope + hparams.n_embd_head_v_full)),
                    ggml_row_size(kv->type, (n_embd_head_qk_nope + hparams.n_embd_head_v_full)*n_head),
                    ggml_row_size(kv->type, (n_embd_head_qk_nope)));
            cb(v_states, "v_states", il);

            v_states = ggml_cont(ctx0, v_states);
            cb(v_states, "v_states", il);

            v_states = ggml_view_2d(ctx0, v_states, hparams.n_embd_head_v_full * n_head, n_tokens,
                    ggml_row_size(kv->type, hparams.n_embd_head_v_full * n_head),
                    0);
            cb(v_states, "v_states", il);

            struct ggml_tensor * q_states = ggml_concat(ctx0, q_nope, q_rope, 0);
            cb(q_states, "q_states", il);

            struct ggml_tensor * k_states = ggml_concat(ctx0, k_nope, ggml_repeat(ctx0, k_rope, q_rope), 0);
            cb(k_states, "k_states", il);

            cur = llm_build_kv(ctx0, lctx, kv_self, gf,
                    model.layers[il].wo, NULL,
                    k_states, v_states, q_states, KQ_mask, n_tokens, kv_head, n_kv, kq_scale, cb, il);

        }
    }

    return cur;
}

ggml_cgraph * llm_build_context::build_deepseek2() {
    const bool tp_mode = (model.split_mode == LLAMA_SPLIT_MODE_GRAPH ||
                          model.split_mode == LLAMA_SPLIT_MODE_ATTN);
#ifdef GGML_USE_VULKAN
    const bool use_f32_attn_precision = true;
#else
    const bool use_f32_attn_precision = lctx.cparams.graph_attn_precision == GGML_TYPE_F32;
#endif
    ggml_cgraph * gf = new_graph_custom();

    // mutable variable, needed during the last layer of the computation to skip unused tokens
    int32_t n_tokens = this->n_tokens;

    bool is_lite = (hparams.n_layer == 27 || hparams.n_layer == 26);

    // We have to pre-scale kq_scale and attn_factor to make the YaRN RoPE work correctly.
    // See https://github.com/ggerganov/llama.cpp/discussions/7416 for detailed explanation.
    const float mscale = attn_factor * (1.0f + hparams.rope_yarn_log_mul * logf(1.0f / freq_scale));
    const float kq_scale = 1.0f*mscale*mscale/sqrtf(float(hparams.n_embd_head_k(0)));
    const float attn_factor_scaled = 1.0f / (1.0f + 0.1f * logf(1.0f / freq_scale));

    struct ggml_tensor * cur;
    struct ggml_tensor * inpL;

    // {n_embd, n_tokens}
    inpL = llm_build_inp_embd(ctx0, lctx, hparams, batch, model.tok_embd, cb);

    // inp_pos - contains the positions
    struct ggml_tensor * inp_pos = build_inp_pos();

    // KQ_mask (mask for 1 head, it will be broadcasted to all heads)
    struct ggml_tensor * KQ_mask = build_inp_KQ_mask();

    if (dsa_should_use(lctx)) {
        static const int n_sink = []{ const char * e = getenv("DSA_SINK"); return e ? atoi(e) : 1; }();
        if (n_sink > 0 && n_sink < (int) n_kv) {
            lctx.inp_dsa_sink = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_kv, n_tokens);
            cb(lctx.inp_dsa_sink, "dsa_sink", -1);
            ggml_set_input(lctx.inp_dsa_sink);
            ggml_build_forward_expand(gf, lctx.inp_dsa_sink);
        }
        if (dsa_can_use_fast_path(lctx, n_tokens, n_kv)) {
            // For DSA token generation (i.e., n_tokens = 1), when we have more than n_top_k entries in the KV cache,
            // we can simply pick the selected n_top_k entries from the cache via ggml_get_rows, so the mask we
            // require is all ON (ideally we shouldn't require a mask, but I think we have places in the FA
            // implementation which assume that there is a mask).
            int n_top_k = dsa_n_top_k(lctx);
            int n_padded = GGML_PAD(n_top_k, 256);
            auto minus_inf = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, n_padded*KQ_mask->ne[1] - n_top_k);
            minus_inf = ggml_fill_inplace(ctx0, minus_inf, -INFINITY);
            auto zero = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, n_top_k);
            zero = ggml_fill_inplace(ctx0, zero, 0.0f);
            auto mask = ggml_concat(ctx0, zero, minus_inf, 0);
            mask = ggml_cast(ctx0, mask, GGML_TYPE_F16);
            mask = ggml_reshape_2d(ctx0, mask, n_padded, KQ_mask->ne[1]);
            ggml_build_forward_expand(gf, mask);
            dsa_tg_fast_mask = mask;
        } else {
            auto minus_inf = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, KQ_mask->ne[0], n_tokens);
            minus_inf = ggml_fill_inplace(ctx0, minus_inf, -INFINITY);
            ggml_build_forward_expand(gf, minus_inf);
            lctx.inp_mask_inf = minus_inf;
        }
    }

    last_sparse_mask = last_sparse_mask_fa = dsa_last_full_sorted = nullptr;

    // whether to use n_tokens as the matrix dimension during multiplication or n_head
    // n_tokens is higher during prompt processing, this allows to optimize for this case
    bool pp_opt = n_tokens >= 128 && lctx.cparams.mla_attn > 1;

    auto rope_cache = cparams.rope_cache && (rope_type == LLAMA_ROPE_TYPE_NEOX || rope_type == LLAMA_ROPE_TYPE_NORM) ?
        ggml_rope_cache(ctx0, inp_pos, nullptr, n_rot, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow) : nullptr;

    if (cparams.mtp_op_type != MTP_OP_NONE) {
        if (model.arch != LLM_ARCH_GLM_DSA || !model.mtp || hparams.nextn_predict_layers == 0) {
            GGML_ABORT("MTP tail is only wired for GLM_DSA models with NextN layers enabled");
        }

        ggml_tensor * hidden_states_from_main_model;

        if (cparams.mtp_op_type == MTP_OP_WARMUP || cparams.mtp_op_type == MTP_OP_UPDATE_ACCEPTED) {
            hidden_states_from_main_model = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hparams.n_embd, n_tokens);
        } else {
            hidden_states_from_main_model = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, hparams.n_embd);
        }
        ggml_set_name(hidden_states_from_main_model, "inp_mtp_states");
        ggml_set_input(hidden_states_from_main_model);

        lctx.inp_mtp_states = hidden_states_from_main_model;

        const int il_mtp = hparams.n_layer - 1;
        const auto & mtp_layer = model.layers[il_mtp];

        cur = build_deepseek2_mtp(mtp_layer, hidden_states_from_main_model, gf, inp_pos, rope_cache);

        ggml_build_forward_expand(gf, cur);
        return gf;
    }

    int n_active_layers = hparams.n_layer - hparams.nextn_predict_layers;
    for (int il = 0; il < n_active_layers; ++il) {
        struct ggml_tensor * inpSA = inpL;

        bool is_tp_layer = tp_mode && model.layers[il].wo && model.layers[il].wo->extra;

        if (is_tp_layer) {
            cur = build_deepseek2_tp_attention(gf, il, inpL, KQ_mask, inp_pos, rope_cache,
                                                kq_scale, attn_factor_scaled,
                                                use_f32_attn_precision, is_lite, pp_opt);
        } else {
            cur = build_deepseek2_layer_attention(gf, il, inpL, KQ_mask, inp_pos, rope_cache,
                                                  kq_scale, attn_factor_scaled,
                                                  use_f32_attn_precision, is_lite, pp_opt);
        }

        if (il == n_active_layers - 1 && !lctx.cparams.mtp) {
            // skip computing output for unused tokens
            struct ggml_tensor * inp_out_ids = build_inp_out_ids();
            n_tokens = n_outputs;
            cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
            cb(cur, "last_attn", il);
            cb(inpSA, "last_ffn_inp", il);
        }

        // TP path folds residual inside the per-rank FFN reduce; layer mode adds it here.
        struct ggml_tensor * ffn_inp;
        if (is_tp_layer) {
            ffn_inp = cur;
        } else {
            ffn_inp = ggml_add(ctx0, cur, inpSA);
        }
        cb(ffn_inp, "ffn_inp", il);

        if (is_tp_layer) {
            cur = ffn_inp;
        } else {
            cur = llm_build_norm(ctx0, ffn_inp, hparams, model.layers[il].ffn_norm, NULL, LLM_NORM_RMS, cb, il);
            cb(cur, "ffn_norm", il);
        }

        if ((uint32_t) il < hparams.n_layer_dense_lead) {
            cur = llm_build_ffn(ctx0, lctx,
                    is_tp_layer ? model.layers[il].ffn_norm : nullptr, cur,
                    model.layers[il].ffn_up,   NULL, NULL,
                    model.layers[il].ffn_gate, NULL, NULL,
                    model.layers[il].ffn_down, NULL, NULL,
                    NULL,
                    LLM_FFN_SILU, LLM_FFN_PAR, cb, il,
                    gf,
                    /*add_input=*/is_tp_layer);
            cb(cur, "ffn_out", il);
        } else if (is_tp_layer) {
            cur = llm_build_std_moe_ffn(ctx0, lctx, model.layers[il].ffn_norm, cur,
                    model.layers[il].ffn_gate_inp,    nullptr,
                    model.layers[il].ffn_up_exps,     nullptr,
                    model.layers[il].ffn_gate_exps,   nullptr,
                    model.layers[il].ffn_down_exps,   nullptr,
                    model.layers[il].ffn_exp_probs_b,
                    model.layers[il].ffn_up_shexp,    nullptr,
                    model.layers[il].ffn_gate_shexp,  nullptr,
                    model.layers[il].ffn_down_shexp,  nullptr,
                    n_expert, n_expert_used,
                    LLM_FFN_SILU, hparams.expert_weights_norm,
                    true, hparams.expert_weights_scale,
                    (enum llm_expert_gating_func_type) hparams.expert_gating_func,
                    LLM_FFN_SILU, cb, il, gf, /*add_input=*/true, model.layers[il].ffn_up_gate_exps);
            cb(cur, "ffn_out", il);
        } else {
            // MoE branch
            ggml_tensor * moe_out =
                llm_build_moe_ffn(ctx0, lctx, cur,
                        model.layers[il].ffn_gate_inp,
                        model.layers[il].ffn_up_exps,
                        model.layers[il].ffn_gate_exps,
                        model.layers[il].ffn_down_exps,
                        model.layers[il].ffn_exp_probs_b,
                        n_expert, n_expert_used,
                        LLM_FFN_SILU, hparams.expert_weights_norm,
                        true, hparams.expert_weights_scale,
                        (enum llm_expert_gating_func_type) hparams.expert_gating_func,
                        cb, il, gf, false, model.layers[il].ffn_up_gate_exps);
            cb(moe_out, "ffn_moe_out", il);

            // FFN shared expert
            ggml_tensor * ffn_shexp = llm_build_ffn(ctx0, lctx, nullptr, cur,
                    model.layers[il].ffn_up_shexp,   NULL, NULL,
                    model.layers[il].ffn_gate_shexp, NULL, NULL,
                    model.layers[il].ffn_down_shexp, NULL, NULL,
                    NULL,
                    LLM_FFN_SILU, LLM_FFN_PAR, cb, il, gf);
            cb(ffn_shexp, "ffn_shexp", il);

            cur = ggml_add(ctx0, moe_out, ffn_shexp);
            cb(cur, "ffn_out", il);
        }

        if (!is_tp_layer) {
            cur = ggml_add(ctx0, cur, ffn_inp);
        }
        cur = lctx.cvec.apply_to(ctx0, cur, il);
        cb(cur, "l_out", il);

        // input for next layer
        inpL = cur;
    }

    cur = build_output(lctx, ctx0, inpL, model.output, model.output_norm, cb);
    cb(cur, "result_output", -1);

    ggml_build_forward_expand(gf, cur);

    return gf;
}

struct ggml_tensor * llm_build_context::build_deepseek2_mtp(
    const llama_layer & mtp_layer,
    struct ggml_tensor * prev_embeddings,
    struct ggml_cgraph * gf,
    struct ggml_tensor * inp_pos,
    [[maybe_unused]] struct ggml_tensor * rope_cache) {
#ifdef GGML_USE_VULKAN
    constexpr bool use_f32_attn_precision = true;
#else
    constexpr bool use_f32_attn_precision = false;
#endif

    const int il = hparams.n_layer - 1;

    const uint32_t n_embd_head_k_mtp   = hparams.n_embd_head_k(il);

    const float mscale = attn_factor * (1.0f + hparams.rope_yarn_log_mul * logf(1.0f / freq_scale));
    const float kq_scale = 1.0f*mscale*mscale/sqrtf(float(n_embd_head_k_mtp));
    const float attn_factor_scaled = 1.0f / (1.0f + 0.1f * logf(1.0f / freq_scale));

    struct ggml_tensor * KQ_mask = build_inp_KQ_mask();
    struct ggml_tensor * inp_out_ids = n_tokens > 1 ? build_inp_out_ids() : nullptr;

    // Token embedding
    ggml_tensor * mtp_embd_weights = mtp_layer.nextn.embed_tokens;
    if (mtp_embd_weights == nullptr) {
        mtp_embd_weights = model.tok_embd;
    }
    ggml_tensor * token_emb = build_inp_embd_mtp(mtp_embd_weights);

    // Normalize and project
    ggml_tensor * token_emb_norm = llm_build_norm(ctx0, token_emb, hparams, mtp_layer.nextn.enorm, NULL, LLM_NORM_RMS, cb, il);
    ggml_tensor * hidden_state_norm = llm_build_norm(ctx0, prev_embeddings, hparams, mtp_layer.nextn.hnorm, NULL, LLM_NORM_RMS, cb, il);

    if (mtp_layer.nextn.eh_proj == nullptr) {
        GGML_ABORT("GLM_DSA MTP requires nextn.eh_proj");
    }

    ggml_tensor * combined = ggml_concat(ctx0, token_emb_norm, hidden_state_norm, 0);
    cb(combined, "mtp_concat", il);
    ggml_tensor * cur = llm_build_lora_mm(lctx, ctx0, mtp_layer.nextn.eh_proj, combined);

    struct ggml_tensor * inpSA = cur;

    cur = build_deepseek2_layer_attention(gf, il, cur, KQ_mask, inp_pos, nullptr,
                                                  kq_scale, attn_factor_scaled,
                                                  use_f32_attn_precision, false, false);

    // Residual + FFN
    ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
    cb(ffn_inp, "mtp_ffn_inp", il);

    if (inp_out_ids) {
        ffn_inp = ggml_get_rows(ctx0, ffn_inp, inp_out_ids);
    }

    cur = llm_build_norm(ctx0, ffn_inp, hparams, mtp_layer.ffn_norm, NULL, LLM_NORM_RMS, cb, il);
    cb(cur, "ffn_norm", il);

    // MoE FFN (MTP layer is always in the MoE range, not dense)
    {
        ggml_tensor * moe_out =
            llm_build_moe_ffn(ctx0, lctx, cur,
                    mtp_layer.ffn_gate_inp,
                    mtp_layer.ffn_up_exps,
                    mtp_layer.ffn_gate_exps,
                    mtp_layer.ffn_down_exps,
                    mtp_layer.ffn_exp_probs_b,
                    n_expert, n_expert_used,
                    LLM_FFN_SILU, hparams.expert_weights_norm,
                    true, hparams.expert_weights_scale,
                    (enum llm_expert_gating_func_type) hparams.expert_gating_func,
                    cb, il, gf, false, mtp_layer.ffn_up_gate_exps);
        cb(moe_out, "ffn_moe_out", il);

        // Shared Expert FFN
        ggml_tensor * ffn_shexp = llm_build_ffn(ctx0, lctx, nullptr, cur,
                mtp_layer.ffn_up_shexp,   NULL, NULL,
                mtp_layer.ffn_gate_shexp, NULL, NULL,
                mtp_layer.ffn_down_shexp, NULL, NULL,
                NULL,
                LLM_FFN_SILU, LLM_FFN_PAR, cb, il);
        cb(ffn_shexp, "ffn_shexp", il);

        cur = ggml_add(ctx0, moe_out, ffn_shexp);
        cb(cur, "ffn_out", il);
    }

    cur = ggml_add(ctx0, cur, ffn_inp);
    cur = lctx.cvec.apply_to(ctx0, cur, il);
    cb(cur, "mtp_ffn_out_resid", il);

    // Output head
    if (mtp_layer.nextn.shared_head_norm == nullptr) {
        GGML_ABORT("GLM_DSA MTP requires nextn.shared_head_norm");
    }

    cur = llm_build_norm(ctx0, cur, hparams, mtp_layer.nextn.shared_head_norm, NULL, LLM_NORM_RMS, cb, il);
    cb(cur, "result_norm", -1);

    // If nextn.shared_head_head is missing, use model.output (Main LM Head)
    ggml_tensor * mtp_head_weights = mtp_layer.nextn.shared_head_head;
    if (mtp_head_weights == nullptr) {
        mtp_head_weights = model.output;
    }
    cur = llm_build_lora_mm(lctx, ctx0, mtp_head_weights, cur);
    cb(cur, "result_output", -1);

    return cur;
}

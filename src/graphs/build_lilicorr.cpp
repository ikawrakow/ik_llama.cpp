#include "../llama-build-context.h"
#include "../llama-context.h"
#include "../llama-model.h"

#include <cmath>

// LiLiCorr correlation head: the lattice scores over the top-k draft candidates, in its own
// translation unit like the other per-family builders.

// LiLiCorr lattice scorer, ported from mainline llama.cpp PR #30506: emits ik's native draft
// lattice - scores [top_k^2, block_size] + ids [top_k, block_size], placeholder column 0 (the
// anchor) skipped by the decoder walk.
void llm_build_context::build_lilicorr_lattice(
        llama_context & lctx, ggml_context * ctx0,
        const llama_model & model, const llama_hparams & hparams,
        const llm_build_cb & cb, ggml_cgraph * gf, int64_t n_tokens,
        ggml_tensor * logits, ggml_tensor * pass_hidden,
        ggml_tensor * anchors, ggml_tensor * tok_embd) {
    const int64_t hidden     = hparams.lilicorr_hidden_size;
    const int64_t top_k      = hparams.lilicorr_candidate_topk;
    const int64_t heads      = hparams.lilicorr_num_heads;
    const int64_t block_size = hparams.dflash_block_size;
    const int64_t n_blocks   = n_tokens / block_size;
    const int64_t slots      = block_size - 1;
    const int64_t nodes      = slots * top_k;
    const int64_t n_embd     = hparams.n_embd;
    const int64_t factor_dim = model.lilicorr_out_head->ne[1];
    GGML_ASSERT(n_blocks > 0 && n_tokens % block_size == 0);
    GGML_ASSERT(logits && anchors && tok_embd && pass_hidden);
    GGML_ASSERT(logits->ne[1] == n_tokens);

    ggml_tensor * candidates = ggml_cont(ctx0, ggml_top_k(ctx0, logits, top_k));
    ggml_tensor * logits_rows = ggml_reshape_3d(ctx0, logits, 1, logits->ne[0], n_tokens);
    ggml_tensor * candidate_logits = ggml_reshape_2d(ctx0,
            ggml_get_rows(ctx0, logits_rows, candidates), top_k, n_tokens);
    // rank features and embeddings require candidates in descending logit order
    ggml_tensor * order = ggml_argsort(ctx0, candidate_logits, GGML_SORT_ORDER_DESC);
    ggml_tensor * candidate_rows = ggml_reshape_3d(ctx0, candidates, 1, top_k, n_tokens);
    candidates = ggml_reshape_2d(ctx0, ggml_get_rows(ctx0, candidate_rows, order), top_k, n_tokens);
    ggml_tensor * prob_rows = ggml_reshape_3d(ctx0, ggml_soft_max(ctx0, logits), 1, logits->ne[0], n_tokens);
    ggml_tensor * probs = ggml_reshape_2d(ctx0,
            ggml_get_rows(ctx0, prob_rows, candidates), top_k, n_tokens);

    ggml_tensor * cand_run  = ggml_reshape_3d(ctx0, candidates, top_k, n_tokens / n_blocks, n_blocks);
    ggml_tensor * probs_run = ggml_reshape_3d(ctx0, probs,      top_k, n_tokens / n_blocks, n_blocks);
    cand_run  = ggml_cont(ctx0, ggml_view_3d(ctx0, cand_run,  top_k, slots, n_blocks,
            cand_run->nb[1],  cand_run->nb[2],  cand_run->nb[1]));
    probs_run = ggml_cont(ctx0, ggml_view_3d(ctx0, probs_run, top_k, slots, n_blocks,
            probs_run->nb[1], probs_run->nb[2], probs_run->nb[1]));

    ggml_tensor * lp = ggml_log(ctx0, probs_run);
    ggml_tensor * lp_blk = ggml_reshape_3d(ctx0, lp, top_k, slots, n_blocks);
    ggml_tensor * top_lp = ggml_cont(ctx0, ggml_view_3d(ctx0, lp_blk, 1, slots, n_blocks,
            lp_blk->nb[1], lp_blk->nb[2], 0));
    // ik's binary kernels broadcast only along ne[0]; repeat the anchor row explicitly
    ggml_tensor * relative = ggml_sub(ctx0, lp_blk, ggml_repeat(ctx0, top_lp, lp_blk));
    ggml_tensor * feature = ggml_reshape_4d(ctx0, lp_blk, 1, top_k, slots, n_blocks);

    ggml_tensor * rank = ggml_arange(ctx0, 0, top_k, 1);
    rank = ggml_repeat(ctx0, ggml_reshape_3d(ctx0, rank, top_k, 1, 1), lp_blk);
    // rank >= 0 here, so sgn == step (step has no CUDA kernel in ik; sgn does)
    ggml_tensor * first = ggml_scale_bias(ctx0, ggml_sgn(ctx0, rank), -1.0f, 1.0f);

    for (ggml_tensor * item : { probs_run, relative,
            ggml_scale(ctx0, rank, top_k == 1 ? 0.0f : 1.0f / (top_k - 1)), first }) {
        feature = ggml_concat(ctx0, feature, ggml_reshape_4d(ctx0, item, 1, top_k, slots, n_blocks), 0);
    }
    // the head's feature width is whatever the trained norm tensor declares (5 in the reference
    // export: log-prob, prob, relative log-prob, rank, is-first), not a literal
    const int64_t n_features = model.lilicorr_feature_norm->ne[0];
    GGML_ASSERT(feature->ne[0] == n_features);
    feature = ggml_reshape_2d(ctx0, feature, n_features, nodes * n_blocks);

    ggml_tensor * embeds = ggml_get_rows(ctx0, tok_embd, ggml_reshape_1d(ctx0, cand_run, nodes * n_blocks));
    ggml_tensor * token_states = embeds;
    if (model.lilicorr_token_proj != nullptr) {
        GGML_ASSERT(model.lilicorr_token_proj_b != nullptr);
        token_states = ggml_add(ctx0,
                llm_build_context::llm_build_lora_mm(lctx, ctx0, model.lilicorr_token_proj, embeds,
                        model.lilicorr_token_proj_s),
                model.lilicorr_token_proj_b);
    }

    ggml_tensor * pass_blk = ggml_reshape_3d(ctx0, pass_hidden, n_embd, block_size, n_blocks);
    ggml_tensor * passes = ggml_reshape_2d(ctx0, ggml_cont(ctx0, ggml_view_3d(ctx0, pass_blk, n_embd, slots, n_blocks,
                pass_blk->nb[1], pass_blk->nb[2], pass_blk->nb[1])), n_embd, slots * n_blocks);
    ggml_tensor * pass_states = ggml_add(ctx0,
            llm_build_context::llm_build_lora_mm(lctx, ctx0, model.lilicorr_pass_hidden_proj, passes,
                    model.lilicorr_pass_hidden_proj_s), model.lilicorr_pass_hidden_proj_b);
    pass_states = ggml_reshape_4d(ctx0, pass_states, hidden, 1, slots, n_blocks);
    pass_states = ggml_reshape_2d(ctx0, ggml_repeat_4d(ctx0, pass_states, hidden, top_k, slots, n_blocks), hidden, nodes * n_blocks);

    // LayerNorm over the 5 feature channels: one GGML_OP_NORM (norm_f32 accepts narrow rows, so
    // no hand-rolled sum/repeat/scale chain), then the affine.
    ggml_tensor * feature_states = ggml_norm(ctx0, feature, 1.0e-5f);
    feature_states = ggml_add(ctx0, ggml_mul(ctx0, feature_states, model.lilicorr_feature_norm), model.lilicorr_feature_norm_b);
    feature_states = ggml_silu(ctx0, ggml_add(ctx0, llm_build_context::llm_build_lora_mm(lctx, ctx0, model.lilicorr_feature_mlp_1, feature_states), model.lilicorr_feature_mlp_1_b));
    feature_states = ggml_add(ctx0, llm_build_context::llm_build_lora_mm(lctx, ctx0, model.lilicorr_feature_mlp_3, feature_states), model.lilicorr_feature_mlp_3_b);

    ggml_tensor * slot_embd = ggml_reshape_4d(ctx0, model.lilicorr_slot_embedding, hidden, 1, slots, 1);
    ggml_tensor * rank_embd = ggml_reshape_4d(ctx0, model.lilicorr_rank_embedding, hidden, top_k, 1, 1);
    ggml_tensor * pos_embd = ggml_add(ctx0,
            ggml_repeat_4d(ctx0, slot_embd, hidden, top_k, slots, n_blocks),
            ggml_repeat_4d(ctx0, rank_embd, hidden, top_k, slots, n_blocks));
    pos_embd = ggml_reshape_2d(ctx0, pos_embd, hidden, nodes * n_blocks);

    ggml_tensor * x = ggml_add(ctx0, ggml_add(ctx0, token_states, pass_states), feature_states);
    x = ggml_add(ctx0, x, pos_embd);

    ggml_tensor * anchor_blk = ggml_reshape_3d(ctx0, anchors, n_embd, block_size, n_blocks);
    ggml_tensor * anchor = ggml_reshape_2d(ctx0,
            ggml_cont(ctx0, ggml_view_3d(ctx0, anchor_blk, n_embd, 1, n_blocks,
                    anchor_blk->nb[1], anchor_blk->nb[2], 0)), n_embd, n_blocks);
    ggml_tensor * anchor_state = ggml_add(ctx0,
            llm_build_context::llm_build_lora_mm(lctx, ctx0, model.lilicorr_context_proj, anchor,
                    model.lilicorr_context_proj_s), model.lilicorr_context_proj_b);

    const int64_t head_dim = hidden / heads;
    for (uint32_t layer = 0; layer < hparams.lilicorr_layers; ++layer) {
        const auto & weights = model.layers[layer];
        ggml_tensor * attn_input = llm_build_context::llm_build_norm(ctx0, x, hparams,
                weights.lilicorr_attn_norm, nullptr, LLM_NORM_RMS, cb, -1);
        ggml_tensor * qkv = ggml_add(ctx0,
                llm_build_context::llm_build_lora_mm(lctx, ctx0, weights.lilicorr_attn_in_proj, attn_input), weights.lilicorr_attn_in_proj_b);
        qkv = ggml_reshape_3d(ctx0, qkv, 3 * hidden, nodes, n_blocks);
        ggml_tensor * q = ggml_cont(ctx0, ggml_view_4d(ctx0, qkv, head_dim, nodes, heads, n_blocks,
                qkv->nb[1], head_dim * qkv->nb[0], qkv->nb[2], 0));
        ggml_tensor * k = ggml_cont(ctx0, ggml_view_4d(ctx0, qkv, head_dim, nodes, heads, n_blocks,
                qkv->nb[1], head_dim * qkv->nb[0], qkv->nb[2], hidden * qkv->nb[0]));
        ggml_tensor * v = ggml_view_4d(ctx0, qkv, head_dim, nodes, heads, n_blocks,
                qkv->nb[1], head_dim * qkv->nb[0], qkv->nb[2], 2 * hidden * qkv->nb[0]);
        ggml_tensor * attn = ggml_soft_max_ext(ctx0, ggml_mul_mat(ctx0, k, q), model.lilicorr_attention_bias,
                1.0f / std::sqrt(static_cast<float>(head_dim)), 0.0f);
        // attn @ v without OUT_PROD (ik's CUDA backend has no out_prod kernel):
        // mul_mat(cont(v^T), attn) contracts over the nodes dimension and yields [head_dim, nodes, ...]
        ggml_tensor * joined = ggml_mul_mat(ctx0, ggml_cont(ctx0, ggml_transpose(ctx0, v)), attn);
        joined = ggml_permute(ctx0, joined, 0, 2, 1, 3);
        ggml_tensor * attn_output = ggml_cont_2d(ctx0, joined, hidden, nodes * n_blocks);
        attn_output = ggml_add(ctx0,
                llm_build_context::llm_build_lora_mm(lctx, ctx0, weights.lilicorr_attn_out_proj, attn_output,
                        weights.lilicorr_attn_out_proj_s), weights.lilicorr_attn_out_proj_b);
        x = ggml_add(ctx0, x, attn_output);
        ggml_tensor * mlp = llm_build_context::llm_build_norm(ctx0, x, hparams,
                weights.lilicorr_mlp_norm, nullptr, LLM_NORM_RMS, cb, -1);
        mlp = ggml_silu(ctx0, ggml_add(ctx0,
                llm_build_context::llm_build_lora_mm(lctx, ctx0, weights.lilicorr_mlp_0, mlp, weights.lilicorr_mlp_0_s), weights.lilicorr_mlp_0_b));
        mlp = ggml_add(ctx0, llm_build_context::llm_build_lora_mm(lctx, ctx0, weights.lilicorr_mlp_2, mlp, weights.lilicorr_mlp_2_s), weights.lilicorr_mlp_2_b);
        x = ggml_add(ctx0, x, mlp);
    }

    x = llm_build_context::llm_build_norm(ctx0, x, hparams, model.lilicorr_output_norm,
            nullptr, LLM_NORM_RMS, cb, -1);
    anchor_state = llm_build_context::llm_build_norm(ctx0, anchor_state, hparams,
            model.lilicorr_anchor_norm, nullptr, LLM_NORM_RMS, cb, -1);
    ggml_tensor * anchor_nodes = ggml_reshape_3d(ctx0, anchor_state, hidden, 1, n_blocks);
    anchor_nodes = ggml_reshape_2d(ctx0, ggml_repeat_4d(ctx0, anchor_nodes, hidden, nodes, n_blocks, 1), hidden, nodes * n_blocks);
    ggml_tensor * factor_in = ggml_concat(ctx0, x, anchor_nodes, 0);
    factor_in = ggml_concat(ctx0, factor_in, ggml_mul(ctx0, x, anchor_nodes), 0);
    ggml_tensor * factors = ggml_add(ctx0,
            llm_build_context::llm_build_lora_mm(lctx, ctx0, model.lilicorr_factor_input_proj, factor_in), model.lilicorr_factor_input_proj_b);
    factors = ggml_silu(ctx0, factors);

    ggml_tensor * out = ggml_add(ctx0,
            llm_build_context::llm_build_lora_mm(lctx, ctx0, model.lilicorr_out_head, factors), model.lilicorr_out_head_b);
    out = ggml_reshape_4d(ctx0, ggml_l2_norm(ctx0, out, hparams.lilicorr_vector_eps), factor_dim, top_k, slots, n_blocks);
    ggml_tensor * in = ggml_add(ctx0,
            llm_build_context::llm_build_lora_mm(lctx, ctx0, model.lilicorr_in_head, factors), model.lilicorr_in_head_b);
    in = ggml_reshape_4d(ctx0, ggml_l2_norm(ctx0, in, hparams.lilicorr_vector_eps), factor_dim, top_k, slots, n_blocks);
    ggml_tensor * anchor_out = ggml_add(ctx0,
            llm_build_context::llm_build_lora_mm(lctx, ctx0, model.lilicorr_anchor_out_head, anchor_state), model.lilicorr_anchor_out_head_b);
    anchor_out = ggml_l2_norm(ctx0, anchor_out, hparams.lilicorr_vector_eps);
    anchor_out = ggml_reshape_3d(ctx0, anchor_out, factor_dim, 1, n_blocks);

    ggml_tensor * first_in = ggml_cont(ctx0, ggml_view_4d(ctx0, in, factor_dim, top_k, 1, n_blocks,
            in->nb[1], in->nb[2], in->nb[3], 0));
    ggml_tensor * first_score = ggml_scale(ctx0,
            ggml_mul_mat(ctx0, first_in, ggml_reshape_4d(ctx0, anchor_out, factor_dim, 1, 1, n_blocks)),
            hparams.lilicorr_logit_scale);
    ggml_tensor * scores = ggml_pad(ctx0,
            ggml_reshape_3d(ctx0, first_score, top_k, 1, n_blocks), top_k * top_k - top_k, 0, 0, 0);
    if (slots > 1) {
        ggml_tensor * next = ggml_cont(ctx0, ggml_view_4d(ctx0, in, factor_dim, top_k, slots - 1, n_blocks,
                in->nb[1], in->nb[2], in->nb[3], in->nb[2]));
        ggml_tensor * previous = ggml_cont(ctx0, ggml_view_4d(ctx0, out, factor_dim, top_k, slots - 1, n_blocks,
                out->nb[1], out->nb[2], out->nb[3], 0));
        ggml_tensor * score = ggml_scale(ctx0, ggml_mul_mat(ctx0, next, previous), hparams.lilicorr_logit_scale);
        scores = ggml_concat(ctx0, scores,
                ggml_reshape_3d(ctx0, score, top_k * top_k, slots - 1, n_blocks), 1);
    }

    // ik-native lattice: placeholder column 0 (the anchor; skipped by the decoder walk)
    ggml_tensor * zero_col = ggml_fill(ctx0,
            ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, top_k * top_k, 1, n_blocks), 0.0f);
    ggml_tensor * lattice = ggml_concat(ctx0, zero_col, scores, 1); // [top_k^2, block_size, n_blocks]
    ggml_tensor * ids_first = ggml_cont(ctx0, ggml_view_3d(ctx0, cand_run, top_k, 1, n_blocks,
            cand_run->nb[1], cand_run->nb[2], 0));
    ggml_tensor * ids = ggml_concat(ctx0, ids_first, cand_run, 1); // [top_k, block_size, n_blocks]

    ggml_tensor * lattice_2d = ggml_reshape_2d(ctx0, lattice, top_k * top_k, block_size * n_blocks);
    ggml_tensor * ids_2d     = ggml_reshape_2d(ctx0, ids,     top_k,     block_size * n_blocks);
    cb(lattice_2d, "lilicorr_lattice", -1);
    cb(ids_2d, "lilicorr_lattice_ids", -1);
    ggml_set_output(ids_2d);
    ggml_build_forward_expand(gf, lattice_2d);
    ggml_build_forward_expand(gf, ids_2d);
    lctx.dflash.draft_lattice_tensor = lattice_2d;
    lctx.dflash.draft_lattice_ids_tensor = ids_2d;
    lctx.dflash.draft_lattice_top_k = (int32_t) top_k;
}

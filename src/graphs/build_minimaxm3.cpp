#include "../llama-build-context.h"
#include "../llama-model.h"
#include "../llama-context.h"

// The cells of the n_sel best-scored blocks per KV head, own block included, [n_sel*block, n_head_kv, n_tokens],
// or nullptr when this ubatch attends the whole cache.
static ggml_tensor * minimaxm3_msa_cells(llm_build_context & bctx, ggml_cgraph * gf, ggml_tensor * cur,
        ggml_tensor * inp_pos, ggml_tensor * causal, int32_t n_sel, int il) {
    ggml_context * ctx0 = bctx.ctx0;
    const llama_hparams & hparams = bctx.hparams;
    const llama_layer & layer = bctx.model.layers[il];
    ggml_tensor * kr_cache = bctx.kv_self.kr_l[il];

    const int32_t n_dim    = hparams.indexer_head_size;
    const int32_t n_head   = hparams.indexer_n_head;
    const int32_t block    = hparams.indexer_block_size;
    const int32_t n_blocks = bctx.n_kv/block;
    const int32_t n_tokens = bctx.n_tokens;

    auto rope = [&](ggml_tensor * x) {
        return ggml_rope_ext(ctx0, x, inp_pos, nullptr, bctx.n_rot, bctx.rope_type, bctx.n_ctx_orig,
                bctx.freq_base, bctx.freq_scale, bctx.ext_factor, bctx.attn_factor, bctx.beta_fast, bctx.beta_slow);
    };

    ggml_tensor * k = ggml_mul_mat(ctx0, layer.indexer_k_proj, cur);
    k = llm_build_context::llm_build_norm(ctx0, k, hparams, layer.indexer_k_norm, nullptr, LLM_NORM_RMS, bctx.cb, il);
    k = ggml_reshape_2d(ctx0, rope(ggml_reshape_3d(ctx0, k, n_dim, 1, n_tokens)), n_dim, n_tokens);
    ggml_tensor * kr_cpy = ggml_cpy(ctx0, k, ggml_view_2d(ctx0, kr_cache, n_dim, n_tokens,
            kr_cache->nb[1], kr_cache->nb[1]*bctx.kv_head));
    // the view above bakes kv_head, so a reused graph needs the copy re-pointed
    bctx.lctx.dsa_cache_copies[il].cpy  = kr_cpy;
    bctx.lctx.dsa_cache_copies[il].step = kr_cache->nb[1];
    ggml_build_forward_expand(gf, kr_cpy);

    // a dense ubatch still had to cache its keys for the sparse ones that follow
    if (!causal) {
        return nullptr;
    }

    ggml_tensor * q = ggml_mul_mat(ctx0, layer.indexer_q_proj, cur);
    q = llm_build_context::llm_build_norm(ctx0, ggml_reshape_3d(ctx0, q, n_dim, n_head, n_tokens), hparams,
            layer.indexer_q_norm, nullptr, LLM_NORM_RMS, bctx.cb, il);
    q = rope(q);

    ggml_tensor * kr = ggml_view_2d(ctx0, kr_cache, n_dim, bctx.n_kv, kr_cache->nb[1], 0);
    ggml_tensor * bias = bctx.lctx.inp_msa_bias;
    ggml_tensor * sel = nullptr;
    for (int32_t r0 = 0; r0 < n_tokens; r0 += MINIMAXM3_MSA_SCORE_CHUNK) {
        const int32_t nr = std::min(MINIMAXM3_MSA_SCORE_CHUNK, n_tokens - r0);
        auto rows = [&](ggml_tensor * x) {
            return nr == n_tokens ? x : ggml_view_3d(ctx0, x, x->ne[0], x->ne[1], nr, x->nb[1], x->nb[2], r0*x->nb[2]);
        };
        ggml_tensor * score = ggml_mul_mat(ctx0, kr, rows(q));
        score = ggml_add(ctx0, ggml_reshape_3d(ctx0, score, block, n_blocks*n_head, nr), rows(causal));
        score = ggml_pool_1d(ctx0, ggml_reshape_3d(ctx0, score, block, n_blocks, n_head*nr), GGML_OP_POOL_MAX, block, block, 0);
        score = ggml_reshape_3d(ctx0, score, n_blocks, n_head, nr);
        bctx.cb(score, "msa_blk_scores", il);
        score = ggml_add(ctx0, score, rows(bias));

        ggml_tensor * sel_r = ggml_cont(ctx0, ggml_top_k(ctx0, ggml_reshape_2d(ctx0, score, n_blocks, n_head*nr), n_sel));
        sel = sel ? ggml_concat(ctx0, sel, sel_r, 1) : sel_r;
    }
    ggml_tensor * cells = ggml_get_rows(ctx0,
            ggml_reshape_2d(ctx0, bctx.lctx.inp_msa_cells, block, n_blocks),
            ggml_reshape_2d(ctx0, sel, n_sel*n_head*n_tokens, 1));
    cells = ggml_reshape_3d(ctx0, cells, n_sel*block, n_head, n_tokens);
    bctx.cb(cells, "msa_cells", il);
    return cells;
}

ggml_cgraph* llm_build_context::build_minimaxm3() {
    ggml_cgraph * gf = new_graph_custom();
    const int64_t n_embd_head = hparams.n_embd_head_v(0);
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k(0));

    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = llm_build_inp_embd(ctx0, lctx, hparams, batch, model.tok_embd, cb);

    ggml_tensor * inp_pos = build_inp_pos();
    ggml_tensor * inp_out_ids = n_tokens > 1 ? build_inp_out_ids() : nullptr;
    ggml_tensor * KQ_mask = build_inp_KQ_mask();

    // the top_k counts cells; whole blocks are selected
    const int32_t block = hparams.indexer_block_size;
    const int32_t n_sel = kv_self.kr_l.empty() ? 0 :
        ((cparams.dsa_top_k > 0 ? cparams.dsa_top_k : (int32_t) hparams.indexer_top_k) + block - 1)/block;
    ggml_tensor * msa_causal = nullptr;
    if (n_sel > 0 && n_sel*block < n_kv) {
        lctx.inp_msa_cells = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_kv);
        lctx.inp_msa_bias  = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, n_kv/block, 1, n_tokens);
        cb(lctx.inp_msa_cells, "inp_msa_cells", -1);
        cb(lctx.inp_msa_bias,  "inp_msa_bias",  -1);
        ggml_set_input(lctx.inp_msa_cells);
        ggml_set_input(lctx.inp_msa_bias);
        // a future cell must not win the max of its block
        msa_causal = ggml_cast(ctx0, ggml_view_2d(ctx0, KQ_mask, n_kv, n_tokens, KQ_mask->nb[1], 0), GGML_TYPE_F32);
        msa_causal = ggml_reshape_3d(ctx0, msa_causal, block, n_kv/block, n_tokens);
    }

    for (int il = 0; il < n_layer; ++il) {
        ggml_tensor * out_ids = il == n_layer - 1 ? inp_out_ids : nullptr;
        ggml_tensor * ffn_inp;
        if (!kv_self.kr_l.empty() && kv_self.kr_l[il]) {
            // the indexer reads the same normed input as q, k and v
            cur = llm_build_norm(ctx0, inpL, hparams, model.layers[il].attn_norm, nullptr, LLM_NORM_RMS, cb, il);
            cb(cur, "attn_norm", il);
            ggml_tensor * cells = minimaxm3_msa_cells(*this, gf, cur, inp_pos, msa_causal, n_sel, il);
            cur = build_std_attention(gf, nullptr, cur, inp_pos, out_ids, nullptr,
                    KQ_mask, nullptr, nullptr, 1.0f / sqrtf(float(n_embd_head)), 0.0f, 0,
                    il, true, false, false, false, false, nullptr, -1, 0.0f, nullptr, nullptr, nullptr, cells);
            ffn_inp = ggml_add(ctx0, cur, out_ids ? ggml_get_rows(ctx0, inpL, out_ids) : inpL);
        } else {
            ffn_inp = build_std_attention(gf, model.layers[il].attn_norm, inpL,
                    inp_pos, out_ids, nullptr,
                    KQ_mask, nullptr, nullptr, 1.0f / sqrtf(float(n_embd_head)), 0.0f, 0,
                    il, true, false, true);
        }

        if ((uint32_t) il < hparams.n_layer_dense_lead) {
            cur = llm_build_ffn(ctx0, lctx, model.layers[il].ffn_norm, ffn_inp,
                    model.layers[il].ffn_up,   nullptr, nullptr,
                    model.layers[il].ffn_gate, nullptr, nullptr,
                    model.layers[il].ffn_down, nullptr, nullptr,
                    nullptr,
                    LLM_FFN_SWIGLU_OAI, LLM_FFN_PAR, cb, il, gf, true);
        } else {
            cur = llm_build_std_moe_ffn(ctx0, lctx, model.layers[il].ffn_norm, ffn_inp,
                    model.layers[il].ffn_gate_inp,
                    nullptr,
                    model.layers[il].ffn_up_exps,
                    nullptr,
                    model.layers[il].ffn_gate_exps,
                    nullptr,
                    model.layers[il].ffn_down_exps,
                    nullptr,
                    model.layers[il].ffn_exp_probs_b,
                    model.layers[il].ffn_up_shexp,
                    nullptr,
                    model.layers[il].ffn_gate_shexp,
                    nullptr,
                    model.layers[il].ffn_down_shexp,
                    nullptr,
                    n_expert, n_expert_used,
                    LLM_FFN_SWIGLU_OAI,
                    hparams.expert_weights_norm,
                    hparams.expert_weights_scale != 0.0f, hparams.expert_weights_scale,
                    (llm_expert_gating_func_type) hparams.expert_gating_func,
                    LLM_FFN_SWIGLU_OAI,
                    cb, il, gf, true, model.layers[il].ffn_up_gate_exps);
        }

        cur = lctx.cvec.apply_to(ctx0, cur, il);
        cb(cur, "l_out", il);

        inpL = cur;
    }

    cur = build_output(lctx, ctx0, inpL, model.output, model.output_norm, cb);
    cb(cur, "result_output", -1);

    ggml_build_forward_expand(gf, cur);
    return gf;
}

#include "../llama-build-context.h"
#include "../llama-model.h"
#include "../llama-context.h"
#include "../llama-delta-net.h"

#include <optional>

// the [hc_dim] gamma is wider than the per-stream reduction, so ggml_fused_rms_norm cannot
// express this and the two ops stay separate
static ggml_tensor * qwen4exp_grouped_rms(
        ggml_context        * ctx0,
        const llama_hparams & hparams,
        ggml_tensor         * x,
        ggml_tensor         * w,
        int32_t               hc_dim,
        int32_t               nt) {
    ggml_tensor * t = ggml_rms_norm(ctx0, x, hparams.f_norm_rms_eps);
    t = ggml_reshape_2d(ctx0, t, hc_dim, nt);
    return ggml_mul(ctx0, t, w);
}

// the low-rank down/up pair is a gate over the normalised streams, not a mixing matrix
static ggml_tensor * qwen4exp_hc_mix(
        llm_build_context & bctx,
        ggml_context      * ctx0,
        llama_context     & lctx,
        const llama_hparams & hparams,
        ggml_tensor       * x,
        ggml_tensor       * w_norm,
        ggml_tensor       * w_down,
        ggml_tensor       * w_up,
        ggml_tensor       * w_inject,
        ggml_tensor      ** inject,
        int32_t             n_embd,
        int                 il,
        const llm_build_cb & cb) {
    const int32_t hc     = hparams.dsv4_hc_mult;
    const int32_t hc_dim = hc * n_embd;
    const int64_t nt     = x->ne[2];

    ggml_tensor * xn = qwen4exp_grouped_rms(ctx0, hparams, x, w_norm, hc_dim, nt);
    cb(xn, "hc_norm", il);

    ggml_tensor * lo = llm_build_context::llm_build_lora_mm(lctx, ctx0, w_down, xn);
    lo = ggml_silu(ctx0, ggml_scale(ctx0, lo, 1.0f / (float) hc));
    ggml_tensor * gate = ggml_sigmoid(ctx0, llm_build_context::llm_build_lora_mm(lctx, ctx0, w_up, lo));
    cb(gate, "hc_gate", il);

    ggml_tensor * gated = ggml_mul(ctx0, xn, gate);
    gated = ggml_reshape_3d(ctx0, gated, n_embd, hc, nt);

    ggml_tensor * mixed = ggml_multi_add(ctx0,
            ggml_view_2d(ctx0, gated, n_embd, nt, gated->nb[2], 0), hc);
    mixed = ggml_scale(ctx0, mixed, 1.0f / (float) hc);
    cb(mixed, "hc_mixed", il);

    if (inject) {
        *inject = llm_build_context::llm_build_lora_mm(lctx, ctx0, w_inject, xn);
        cb(*inject, "hc_inject", il);
    }

    GGML_UNUSED(bctx);
    return mixed;
}

// the factor of two centres the weights on one, so an untrained injection is a plain
// residual add
static ggml_tensor * qwen4exp_hc_combine(
        ggml_context        * ctx0,
        const llama_hparams & hparams,
        ggml_tensor         * residual,
        ggml_tensor         * block_out,
        ggml_tensor         * inject,
        int32_t               n_embd,
        int                   il,
        const llm_build_cb  & cb) {
    const int32_t hc = hparams.dsv4_hc_mult;
    const int64_t nt = residual->ne[2];

    ggml_tensor * w = ggml_sigmoid(ctx0, ggml_scale(ctx0, inject, 1.0f / (float) hc));
    w = ggml_scale(ctx0, w, 2.0f);
    w = ggml_reshape_3d(ctx0, w, 1, hc, nt);

    ggml_tensor * b = ggml_reshape_3d(ctx0, block_out, n_embd, 1, nt);
    b = ggml_repeat_4d(ctx0, b, n_embd, hc, nt, 1);

    ggml_tensor * cur = ggml_add(ctx0, residual, ggml_mul(ctx0, b, w));
    cb(cur, "hc_combine", il);

    return cur;
}

//   out[c, t] = sum_k w[k, c] * x[c, t - (K-1-k)*dilation]
//
// the (K-1)*dilation positions reached back live in the tail of this layer's state row.
// Prepending them puts every tap inside the tensor, so none needs a pad or a range test
static ggml_tensor * qwen4exp_ple_conv(
        ggml_context        * ctx0,
        ggml_cgraph         * gf,
        const llama_hparams & hparams,
        const llama_model   & model,
        ggml_tensor         * state_all,
        ggml_tensor         * xt,          // [n_tokens, hc_dim]
        int32_t               hc_dim,
        int32_t               n_tokens,
        uint32_t              slot,
        bool                  reset,
        int                   il,
        const llm_build_cb  & cb) {
    const int32_t kern = hparams.ple_conv_kernel;
    const int32_t dil  = hparams.ple_ngram_size;
    const int32_t hist = hparams.ple_conv_state();

    // the delta-net state occupies the front of the row; this history follows it
    const size_t esz     = ggml_element_size(state_all);
    const size_t row_off = esz * (state_all->ne[0] - hist*hc_dim);

    ggml_tensor * state = ggml_cont(ctx0,
            ggml_view_2d(ctx0, state_all, hist, hc_dim, hist*esz, slot*state_all->nb[1] + row_off));
    if (reset) {
        state = ggml_scale(ctx0, state, 0.0f);
    }
    cb(state, "ple_conv_state", il);

    ggml_tensor * conv_in = ggml_concat(ctx0, state, xt, 0); // [hist + n_tokens, hc_dim]

    ggml_tensor * conv_out = nullptr;
    for (int32_t k = 0; k < kern; ++k) {
        const int32_t start = hist - (kern - 1 - k)*dil;

        ggml_tensor * shifted = ggml_cont(ctx0,
                ggml_view_2d(ctx0, conv_in, n_tokens, hc_dim, conv_in->nb[1], start*conv_in->nb[0]));

        ggml_tensor * wk = ggml_cont(ctx0,
                ggml_view_2d(ctx0, model.layers[il].ple_conv1d, 1, hc_dim,
                        model.layers[il].ple_conv1d->nb[1],
                        k * model.layers[il].ple_conv1d->nb[0]));
        wk = ggml_reshape_1d(ctx0, wk, hc_dim);
        if (wk->type != GGML_TYPE_F32) {
            wk = ggml_cast(ctx0, wk, GGML_TYPE_F32);
        }

        ggml_tensor * term = ggml_mul(ctx0, ggml_cont(ctx0, ggml_transpose(ctx0, shifted)), wk);
        term = ggml_cont(ctx0, ggml_transpose(ctx0, term)); // [n_tokens, hc_dim]

        conv_out = conv_out ? ggml_add(ctx0, conv_out, term) : term;
    }

    // the last `hist` columns are what the next ubatch reaches back into. When the ubatch is
    // shorter than the window they still carry part of the incoming state, which is correct.
    ggml_tensor * tail = ggml_cont(ctx0,
            ggml_view_2d(ctx0, conv_in, hist, hc_dim, conv_in->nb[1],
                    (conv_in->ne[0] - hist) * conv_in->nb[0]));
    ggml_tensor * dst = ggml_view_2d(ctx0, state_all, hist, hc_dim, hist*esz,
            slot*state_all->nb[1] + row_off);
    ggml_build_forward_expand(gf, ggml_cpy(ctx0, tail, dst));

    return conv_out;
}

static ggml_tensor * qwen4exp_ple(
        llm_build_context & bctx,
        ggml_context      * ctx0,
        ggml_cgraph       * gf,
        llama_context     & lctx,
        const llama_model & model,
        const llama_hparams & hparams,
        const delta_net   & delta,
        ggml_tensor       * hidden,
        ggml_tensor       * rows,
        ggml_tensor       * state_all,
        bool                reset_state,
        const std::vector<bool> & reset_pos,
        int32_t             n_embd,
        int32_t             n_tokens,
        int                 il,
        const llm_build_cb & cb) {
    const int32_t hc      = hparams.dsv4_hc_mult;
    const int32_t hc_dim  = hc * n_embd;
    const int32_t n_heads = hparams.ple_n_heads;

    // get_rows lays the head dimension out slowest, which is the flatten order the
    // projections expect
    ggml_tensor * emb = ggml_get_rows(ctx0, model.tok_embd_per_layer, rows);
    emb = ggml_reshape_2d(ctx0, emb, hparams.ple_head_dim * n_heads, n_tokens);
    cb(emb, "ple_embd", il);

    ggml_tensor * key   = llm_build_context::llm_build_lora_mm(lctx, ctx0, model.layers[il].ple_key,   emb);
    ggml_tensor * value = llm_build_context::llm_build_lora_mm(lctx, ctx0, model.layers[il].ple_value, emb);

    auto grouped_norm = [&](ggml_tensor * x, ggml_tensor * w) {
        ggml_tensor * t = qwen4exp_grouped_rms(ctx0, hparams,
                ggml_reshape_3d(ctx0, x, n_embd, hc, n_tokens), w, hc_dim, n_tokens);
        return ggml_reshape_3d(ctx0, t, n_embd, hc, n_tokens);
    };

    key = grouped_norm(key, model.layers[il].ple_norm_key);
    ggml_tensor * query = grouped_norm(hidden, model.layers[il].ple_norm_query);

    ggml_tensor * s = ggml_sum_rows(ctx0, ggml_mul(ctx0, key, query));
    s = ggml_scale(ctx0, s, 1.0f / sqrtf((float) n_embd));

    ggml_tensor * mag  = ggml_sqrt(ctx0, ggml_clamp(ctx0, ggml_abs(ctx0, s), 1e-6f, INFINITY));
    ggml_tensor * gate = ggml_sigmoid(ctx0, ggml_mul(ctx0, ggml_sgn(ctx0, s), mag));
    cb(gate, "ple_gate", il);

    ggml_tensor * v3 = ggml_reshape_3d(ctx0, value, n_embd, 1, n_tokens);
    v3 = ggml_repeat_4d(ctx0, v3, n_embd, hc, n_tokens, 1);

    ggml_tensor * gated = ggml_mul(ctx0, v3, gate);
    cb(gated, "ple_gated_value", il);

    ggml_tensor * normalized = grouped_norm(
            ggml_reshape_2d(ctx0, gated, hc_dim, n_tokens),
            model.layers[il].ple_norm_conv);
    normalized = ggml_reshape_2d(ctx0, normalized, hc_dim, n_tokens);

    ggml_tensor * xt = ggml_cont(ctx0, ggml_transpose(ctx0, normalized)); // [n_tokens, hc_dim]

    ggml_tensor * conv_out = nullptr;
    if (delta.batch_shares_one_seq()) {
        conv_out = qwen4exp_ple_conv(ctx0, gf, hparams, model, state_all, xt, hc_dim, n_tokens,
                delta.state_slot(0), reset_state, il, cb);
    } else {
        // A mixed-sequence ubatch reads a different history per token, exactly as the
        // delta-net path splits it.
        for (int32_t i = 0; i < n_tokens; ++i) {
            ggml_tensor * x_i = ggml_cont(ctx0,
                    ggml_view_2d(ctx0, xt, 1, hc_dim, xt->nb[1], i*xt->nb[0]));
            ggml_tensor * out_i = qwen4exp_ple_conv(ctx0, gf, hparams, model, state_all, x_i, hc_dim, 1,
                    delta.state_slot(i), reset_pos[i], il, cb);
            conv_out = conv_out ? ggml_concat(ctx0, conv_out, out_i, 0) : out_i;
        }
    }

    conv_out = ggml_silu(ctx0, conv_out);
    conv_out = ggml_cont(ctx0, ggml_transpose(ctx0, conv_out)); // [hc_dim, n_tokens]
    conv_out = ggml_reshape_3d(ctx0, conv_out, n_embd, hc, n_tokens);
    cb(conv_out, "ple_conv_out", il);

    GGML_UNUSED(bctx);
    return ggml_add(ctx0, hidden, ggml_add(ctx0, gated, conv_out));
}

// a query keeps a budget of whole blocks plus the incomplete tail it sits in, where a block is
// compress_ratio consecutive cells scored through the mean of its members' indexer keys
static ggml_tensor * qwen4exp_qsa_mask(
        llm_build_context & bctx,
        ggml_context      * ctx0,
        llama_context     & lctx,
        ggml_cgraph       * gf,
        ggml_tensor       * cur,
        ggml_tensor       * inp_pos,
        ggml_tensor       * KQ_mask,
        int                 il,
        const llm_build_cb & cb) {
    const llama_hparams  & hparams = bctx.hparams;
    const llama_model    & model   = bctx.model;
    const llama_kv_cache & kv_self = bctx.kv_self;

    ggml_tensor * kr_cache = il < (int) kv_self.kr_l.size() ? kv_self.kr_l[il] : nullptr;
    ggml_tensor * kp_cache = il < (int) kv_self.kp_l.size() ? kv_self.kp_l[il] : nullptr;
    if (!kr_cache || !kp_cache || !model.layers[il].indexer_k_proj) {
        return KQ_mask;
    }

    const int32_t idx_dim  = hparams.indexer_head_size;
    const int32_t n_idx_h  = hparams.indexer_n_head;
    const int32_t r        = hparams.dsv4_compress_ratios[il];
    const int32_t n_kv     = bctx.n_kv;
    const int32_t n_tokens = bctx.n_tokens;

    // the cached indexer keys are raw: pooling precedes both the norm and the rotation
    ggml_tensor * k_raw = llm_build_context::llm_build_lora_mm(lctx, ctx0, model.layers[il].indexer_k_proj, cur);
    k_raw = ggml_reshape_2d(ctx0, k_raw, idx_dim, n_tokens);
    cb(k_raw, "qsa_k_raw", il);
    {
        ggml_tensor * kr_view = ggml_view_2d(ctx0, kr_cache, idx_dim, n_tokens,
                ggml_row_size(kr_cache->type, idx_dim),
                ggml_row_size(kr_cache->type, idx_dim) * bctx.kv_head);
        ggml_tensor * kr_cpy = ggml_cpy(ctx0, k_raw, kr_view);
        // the view above bakes kv_head, so register the copy for the offset fixup that
        // update_cache_copies() already applies to the K and V writes of a reused graph
        if (il < (int) lctx.dsa_cache_copies.size()) {
            lctx.dsa_cache_copies[il].cpy  = kr_cpy;
            lctx.dsa_cache_copies[il].step = kr_cache->nb[1];
        }
        ggml_build_forward_expand(gf, kr_cpy);
    }

    const int32_t n_blocks = (n_kv + r - 1)/r;

    // a raw key never changes once written, so only the blocks this ubatch wrote need pooling
    // again: n_tokens consecutive cells span n_tokens/r, plus one when the run straddles
    const int32_t n_win = lctx.qsa_pooled_stale ? n_blocks : std::min(n_blocks, (n_tokens + r - 1)/r + 1);

    llama_context::qsa_input * inp = nullptr;
    for (auto & q : lctx.inp_qsa) {
        if (q.ratio == (int32_t) r) {
            inp = &q;
            break;
        }
    }
    if (!inp) {
        lctx.inp_qsa.emplace_back();
        inp = &lctx.inp_qsa.back();
        inp->ratio      = (int32_t) r;
        inp->cell_blk   = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_kv);
        inp->bias       = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_kv, n_tokens);
        inp->win_blocks = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_win);
        inp->win_cells  = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, r*n_win);
        inp->win_pos    = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, GGML_MROPE_SECTIONS*n_win);
        inp->head_w     = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_idx_h, n_tokens);
        cb(inp->cell_blk,   "qsa_cell_blk",   -1);
        cb(inp->bias,       "qsa_bias",       -1);
        cb(inp->win_blocks, "qsa_win_blocks", -1);
        cb(inp->win_cells,  "qsa_win_cells",  -1);
        cb(inp->win_pos,    "qsa_win_pos",    -1);
        cb(inp->head_w,     "qsa_head_w",     -1);
        for (ggml_tensor * t : {inp->cell_blk, inp->bias, inp->win_blocks, inp->win_cells, inp->win_pos, inp->head_w}) {
            ggml_set_input(t);
            ggml_build_forward_expand(gf, t);
        }
    }

    ggml_tensor * k_all = ggml_view_2d(ctx0, kr_cache, idx_dim, n_kv,
            ggml_row_size(kr_cache->type, idx_dim), 0);

    ggml_tensor * members = ggml_get_rows(ctx0, k_all, inp->win_cells);
    members = ggml_reshape_3d(ctx0, members, idx_dim, r, n_win);

    // the gather put each block's members contiguously in one row, which is what multi_add sums
    ggml_tensor * pooled = ggml_multi_add(ctx0,
            ggml_view_2d(ctx0, members, idx_dim, n_win, members->nb[2], 0), r);
    pooled = ggml_scale(ctx0, pooled, 1.0f/(float) r);
    cb(pooled, "qsa_k_pooled", il);

    int sections[GGML_MROPE_SECTIONS];
    std::copy(hparams.rope_sections.begin(), hparams.rope_sections.begin() + GGML_MROPE_SECTIONS, sections);

    // rope reads [n_dims, n_head, n_tokens], and a block stands in for one token here
    pooled = ggml_reshape_3d(ctx0, pooled, idx_dim, 1, n_win);
    pooled = llm_build_context::llm_build_norm(ctx0, pooled, hparams,
            model.layers[il].indexer_k_norm, nullptr, LLM_NORM_RMS, cb, il);
    pooled = ggml_rope_multi(ctx0, pooled, inp->win_pos, nullptr,
            bctx.n_rot, sections, bctx.rope_type, bctx.n_ctx_orig, bctx.freq_base, bctx.freq_scale,
            bctx.ext_factor, bctx.attn_factor, bctx.beta_fast, bctx.beta_slow);
    pooled = ggml_reshape_2d(ctx0, pooled, idx_dim, n_win);
    cb(pooled, "qsa_k_win", il);

    // score against the scatter's result, not the cache, so the read depends on the write
    // rather than merely following it into the graph
    ggml_tensor * kp_all = ggml_set_rows(ctx0, kp_cache, pooled, inp->win_blocks);
    cb(kp_all, "qsa_k_scatter", il);

    pooled = ggml_view_2d(ctx0, kp_all, idx_dim, n_blocks,
            ggml_row_size(kp_cache->type, idx_dim), 0);
    cb(pooled, "qsa_k", il);

    // at n_kv the cut keeps every cell and scoring cannot change the mask. The pooling above
    // still had to run: it is the only chance these blocks get to enter the cache
    const int32_t top_k_cells = lctx.cparams.dsa_top_k > 0 ? lctx.cparams.dsa_top_k : (int32_t) hparams.indexer_top_k;
    const int32_t width = top_k_cells + r - 1;
    if (width >= n_kv) {
        ggml_build_forward_expand(gf, kp_all);
        return KQ_mask;
    }

    ggml_tensor * q = llm_build_context::llm_build_lora_mm(lctx, ctx0, model.layers[il].indexer_q_proj, cur);
    q = ggml_reshape_3d(ctx0, q, idx_dim, n_idx_h, n_tokens);
    q = llm_build_context::llm_build_norm(ctx0, q, hparams,
            model.layers[il].indexer_q_norm, nullptr, LLM_NORM_RMS, cb, il);
    q = ggml_rope_multi(ctx0, q, inp_pos, nullptr,
            bctx.n_rot, sections, bctx.rope_type, bctx.n_ctx_orig, bctx.freq_base, bctx.freq_scale,
            bctx.ext_factor, bctx.attn_factor, bctx.beta_fast, bctx.beta_slow);
    cb(q, "qsa_q", il);

    // causality and sequence membership come from KQ_mask rather than being rebuilt on the
    // host; inp->bias carries only what is specific to the block cut
    ggml_tensor * causal = ggml_view_2d(ctx0, KQ_mask, n_kv, n_tokens, KQ_mask->nb[1], 0);
    if (causal->type != GGML_TYPE_F32) {
        causal = ggml_cast(ctx0, causal, GGML_TYPE_F32);
    }
    ggml_tensor * cut_mask = ggml_add(ctx0, inp->bias, causal);
    cb(cut_mask, "qsa_cut_mask", il);

    // top-k selects cells, not blocks: every cell carries its own block's pooled key, so the
    // causal mask can still drop cells inside a selected block. The op scores and sorts
    // internally, so the n_kv by n_tokens score matrix is never materialized.
    if (lctx.cparams.fused_idx_topk) {
        ggml_tensor * k_cells = ggml_get_rows_ext(ctx0, pooled, inp->cell_blk, true, false);
        k_cells = ggml_reshape_3d(ctx0, k_cells, idx_dim, n_kv, 1);
        ggml_tensor * fused = ggml_indexer_topk(ctx0, k_cells, q, inp->head_w, cut_mask,
                GGML_UNARY_OP_RELU, width);
        cb(fused, "qsa_top_k", il);
        ggml_build_forward_expand(gf, fused);
        ggml_tensor * mask = ggml_indexer_mask(ctx0, KQ_mask, fused);
        cb(mask, "qsa_mask", il);
        return mask;
    }

    ggml_tensor * score = ggml_mul_mat(ctx0, pooled,
            ggml_reshape_2d(ctx0, ggml_cont(ctx0, q), idx_dim, n_idx_h*n_tokens));
    score = ggml_reshape_3d(ctx0, score, n_blocks, n_idx_h, n_tokens);
    score = ggml_relu(ctx0, score);
    score = ggml_cont(ctx0, ggml_permute(ctx0, score, 1, 0, 2, 3));
    score = ggml_sum_rows(ctx0, score);
    score = ggml_reshape_2d(ctx0, score, n_blocks, n_tokens);
    cb(score, "qsa_score", il);

    // expanding the block indices instead would need an integer multiply-add ggml has no op
    // for. get_rows gathers rows, so the scores are transposed first
    ggml_tensor * expanded = ggml_get_rows(ctx0,
            ggml_cont(ctx0, ggml_transpose(ctx0, score)), inp->cell_blk);
    expanded = ggml_cont(ctx0, ggml_transpose(ctx0, expanded));

    expanded = ggml_add(ctx0, expanded, cut_mask);
    cb(expanded, "qsa_score_cells", il);

    ggml_tensor * top_k = ggml_cont(ctx0, ggml_top_k(ctx0, expanded, width));
    cb(top_k, "qsa_top_k", il);

    ggml_tensor * mask = ggml_indexer_mask(ctx0, KQ_mask, top_k);
    cb(mask, "qsa_mask", il);

    return mask;
}

ggml_cgraph * llm_build_context::build_qwen4exp() {

    ggml_cgraph * gf = new_graph_custom();

    const bool is_mtp = lctx.cparams.mtp_op_type != MTP_OP_NONE;

    // the MTP pass walks only the QSA tail: no recurrent state, so the draft
    // context has zero qnext state slots and the delta-net ctor must not run
    std::optional<delta_net> delta_opt;
    if (!is_mtp) {
        delta_opt.emplace(lctx, batch);
    }

    const int32_t n_embd_head = hparams.n_embd_head_v(0);
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k(0));

    const int32_t hc = hparams.dsv4_hc_mult;

    const int n_layer_begin = is_mtp ? n_layer - hparams.nextn_predict_layers : 0;
    const int n_layer_end   = is_mtp ? n_layer : n_layer - hparams.nextn_predict_layers;

    ggml_tensor * inp_pos = build_inp_pos();
    ggml_tensor * inp_out_ids = (is_mtp || n_tokens > 1) ? build_inp_out_ids() : nullptr;
    ggml_tensor * KQ_mask = build_inp_KQ_mask();

    float KQ_scale = hparams.f_attention_scale == 0.0f ? 1.0f / sqrtf(float(n_embd_head)) : hparams.f_attention_scale;

    ggml_tensor * res_hc = nullptr;
    if (is_mtp) {
        // the fill sites assert on these buffers; the MTP graph consumes neither
        lctx.inp_s_seq_qnext = nullptr;
        lctx.inp_ple_rows    = nullptr;

        const int32_t hc_dim = hc * n_embd;
        ggml_tensor * hidden = build_inp_mtp_states(hc_dim);       // target's pre-mixer wide stream
        ggml_tensor * tok    = build_inp_embd_mtp(model.tok_embd); // [n_embd, n_tokens]
        const auto & nextn = model.layers[n_layer_begin].nextn;

        // normalize the embedding (n_embd) and the wide hidden stream (hc*n_embd)
        ggml_tensor * e = ggml_rms_norm(ctx0, tok, hparams.f_norm_rms_eps);
        e = ggml_mul(ctx0, e, nextn.enorm);

        ggml_tensor * h = ggml_rms_norm(ctx0, hidden, hparams.f_norm_rms_eps);
        h = ggml_mul(ctx0, h, nextn.hnorm);

        // eh_proj is [fc_embedding | fc_hidden] along its input dimension.
        // Concatenating e_norm and h_norm per stream therefore preserves the old
        // fc_embedding@e_norm + fc_hidden@h_norm entry result in one matmul.
        e = ggml_repeat_4d(ctx0,
                ggml_reshape_3d(ctx0, e, n_embd, 1, n_tokens), n_embd, hc, n_tokens, 1);
        h = ggml_reshape_3d(ctx0, h, n_embd, hc, n_tokens);
        ggml_tensor * eh = ggml_concat(ctx0, e, h, 0);
        cb(eh, "mtp_eh_concat", n_layer_begin);

        res_hc = llm_build_lora_mm(lctx, ctx0, nextn.eh_proj, eh);
        cb(res_hc, "mtp_eh_proj", n_layer_begin);
    } else {
        ggml_tensor * inpL = llm_build_inp_embd(ctx0, lctx, hparams, batch, model.tok_embd, cb);

        lctx.inp_s_seq_qnext = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, 1, n_tokens);
        cb(lctx.inp_s_seq_qnext, "inp_s_seq_qnext", -1);
        ggml_set_input(lctx.inp_s_seq_qnext);

        // the wide residual starts as hc identical copies of the embedding
        res_hc = ggml_repeat_4d(ctx0,
                ggml_reshape_3d(ctx0, inpL, n_embd, 1, n_tokens),
                n_embd, hc, n_tokens, 1);
        cb(res_hc, "hc_residual", -1);

        if (hparams.ple_n_heads > 0) {
            lctx.inp_ple_rows = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, hparams.ple_n_heads * n_tokens);
            cb(lctx.inp_ple_rows, "inp_ple_rows", -1);
            ggml_set_input(lctx.inp_ple_rows);
        } else {
            lctx.inp_ple_rows = nullptr;
        }
    }

    // the same test the delta-net path uses for its own state
    const bool ple_reset_state = batch.pos != nullptr && batch.pos[0] == 0;
    std::vector<bool> ple_reset_pos(n_tokens, false);
    for (int32_t i = 0; i < n_tokens && batch.pos != nullptr; ++i) {
        ple_reset_pos[i] = batch.pos[i] == 0;
    }

    for (int il = n_layer_begin; il < n_layer_end; ++il) {
        ggml_tensor * inject = nullptr;

        if (hparams.is_ple(il)) {
            res_hc = qwen4exp_ple(*this, ctx0, gf, lctx, model, hparams, *delta_opt, res_hc, lctx.inp_ple_rows,
                    kv_self.s_l[il], ple_reset_state, ple_reset_pos, n_embd, n_tokens, il, cb);
        }

        ggml_tensor * cur = qwen4exp_hc_mix(*this, ctx0, lctx, hparams, res_hc,
                model.layers[il].hc_attn_norm, model.layers[il].hc_attn_down,
                model.layers[il].hc_attn_up,   model.layers[il].hc_attn_inject,
                &inject, n_embd, il, cb);

        if (hparams.is_recurrent(il)) {
            cur = delta_opt->build_layer_attn_linear(ctx0, gf, cur, nullptr, il, cb, /* external_residual */ true,
                    GGML_UNARY_OP_SIGMOID);
        } else {
            // the indexer reads the same block input as q/k/v, and returns the causal mask
            // itself when the layer carries no compression ratio
            ggml_tensor * mask = hparams.is_qsa(il)
                ? qwen4exp_qsa_mask(*this, ctx0, lctx, gf, cur, inp_pos, KQ_mask, il, cb)
                : KQ_mask;

            cur = build_std_attention(gf, nullptr, cur, inp_pos, nullptr, nullptr,
                    mask, nullptr, nullptr, KQ_scale, 0.0f, 0, il, true, false,
                    /* add_input */ false, /* is_norm */ false, /* is_multi */ true);
        }

        res_hc = qwen4exp_hc_combine(ctx0, hparams, res_hc, cur, inject, n_embd, il, cb);

        cur = qwen4exp_hc_mix(*this, ctx0, lctx, hparams, res_hc,
                model.layers[il].hc_ffn_norm, model.layers[il].hc_ffn_down,
                model.layers[il].hc_ffn_up,   model.layers[il].hc_ffn_inject,
                &inject, n_embd, il, cb);

        cur = llm_build_std_moe_ffn(ctx0, lctx, nullptr, cur,
                model.layers[il].ffn_gate_inp,  nullptr,
                model.layers[il].ffn_up_exps,   nullptr,
                model.layers[il].ffn_gate_exps, nullptr,
                model.layers[il].ffn_down_exps, nullptr,
                nullptr,
                model.layers[il].ffn_up_shexp,    nullptr,
                model.layers[il].ffn_gate_shexp,  nullptr,
                model.layers[il].ffn_down_shexp,  nullptr,
                n_expert, n_expert_used,
                LLM_FFN_SILU, true, false, 0.0f,
                LLM_EXPERT_GATING_FUNC_SOFTMAX,
                LLM_FFN_SILU, cb, il, gf, /* add_input */ false, model.layers[il].ffn_up_gate_exps, nullptr,
                model.layers[il].ffn_gate_inp_shexp);

        res_hc = qwen4exp_hc_combine(ctx0, hparams, res_hc, cur, inject, n_embd, il, cb);
        res_hc = lctx.cvec.apply_to(ctx0, res_hc, il);
        cb(res_hc, "l_out", il);
    }

    if (is_mtp) {
        // wide stream out: the next draft step's hidden input (scheme A)
        ggml_tensor * flat = ggml_reshape_2d(ctx0, res_hc, hc * n_embd, n_tokens);
        ggml_tensor * h_next = inp_out_ids ? ggml_get_rows(ctx0, flat, inp_out_ids) : flat;
        cb(h_next, "result_mtp_embd", -1);
        ggml_set_output(h_next);
        ggml_build_forward_expand(gf, h_next);

        // exit mixer is hc_head-shaped: per-stream norm + low-rank gate, no inject
        const auto & nextn = model.layers[n_layer_begin].nextn;
        ggml_tensor * cur = qwen4exp_hc_mix(*this, ctx0, lctx, hparams,
                ggml_reshape_3d(ctx0, h_next, n_embd, hc, h_next->ne[1]),
                nextn.hc_head_norm, nextn.hc_head_down, nextn.hc_head_up,
                nullptr, nullptr, n_embd, -1, cb);

        cur = llm_build_lora_mm(lctx, ctx0, model.output, cur);
        cb(cur, "result_output", -1);
        ggml_build_forward_expand(gf, cur);
        return gf;
    }

    if (lctx.cparams.mtp && hparams.n_embd_out > hparams.n_embd) {
        // hand the draft the pre-mixer wide stream for its first step
        ggml_tensor * flat = ggml_reshape_2d(ctx0, res_hc, hc * n_embd, n_tokens);
        ggml_tensor * h_nextn = inp_out_ids ? ggml_get_rows(ctx0, flat, inp_out_ids) : flat;
        cb(h_nextn, "result_mtp_embd", -1);
        ggml_set_output(h_nextn);
        ggml_build_forward_expand(gf, h_nextn);
    }

    ggml_tensor * cur = qwen4exp_hc_mix(*this, ctx0, lctx, hparams, res_hc,
            model.hc_head_norm, model.hc_head_down, model.hc_head_up,
            nullptr, nullptr, n_embd, -1, cb);

    if (inp_out_ids) {
        cur = ggml_get_rows(ctx0, cur, inp_out_ids);
    }

    // name the mixed stream for append_pooling (the nextn tail needs a named embd)
    cb(cur, "result_embd", -1);

    cur = llm_build_lora_mm(lctx, ctx0, model.output, cur);
    cb(cur, "result_output", -1);

    ggml_build_forward_expand(gf, cur);

    return gf;
}

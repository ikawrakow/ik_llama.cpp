#include "../llama-build-context.h"
#include "../llama-model.h"
#include "../llama-context.h"

#include <algorithm>
#include <vector>

// build one LFM2 short-convolution block; the state holds the last l_cache - 1 input vectors
static ggml_tensor * build_lfm2_shortconv(
        llm_build_context & bctx,
        ggml_cgraph *       gf,
        ggml_tensor *       input,
        ggml_tensor *       state_seq,
        uint32_t            state_seq_id,
        bool                reset_state,
        int                 il) {
    const auto & hparams = bctx.hparams;
    auto &       lctx    = bctx.lctx;
    auto &       layer   = bctx.model.layers[il];

    const int64_t n_tokens = input->ne[1];
    const int64_t d_conv   = hparams.n_shortconv_l_cache;
    const int64_t state_dim = (d_conv - 1) * hparams.n_embd;

    GGML_ASSERT(d_conv > 1);
    GGML_ASSERT(layer.ssm_in != nullptr);
    GGML_ASSERT(layer.ssm_conv1d != nullptr);
    GGML_ASSERT(layer.ssm_out != nullptr);
    GGML_ASSERT(state_seq != nullptr);
    GGML_ASSERT(state_seq->ne[0] == 1 && state_seq->ne[1] == n_tokens);
    GGML_ASSERT(lctx.kv_self.s_l[il] != nullptr);
    GGML_ASSERT(state_seq_id < (uint32_t) lctx.kv_self.s_l[il]->ne[1]);

    // state row for this sequence; a reset is a graph op so it is ordered before SSM_CONV
    auto state_all = ggml_reshape_2d(bctx.ctx0, lctx.kv_self.s_l[il], state_dim,
            lctx.kv_self.s_l[il]->ne[1]);
    auto state_dst = ggml_view_2d(bctx.ctx0, state_all, state_dim, 1,
            state_all->nb[1], (size_t) state_seq_id * state_all->nb[1]);
    auto state_src = state_dst;
    if (reset_state) {
        state_src = ggml_scale(bctx.ctx0, state_src, 0.0f);
        bctx.cb(state_src, "state_reset", il);
    }

    auto conv_state = ggml_reshape_3d(bctx.ctx0, state_src, d_conv - 1, hparams.n_embd, 1);

    // {n_embd, n_tokens} -> {3*n_embd, n_tokens}, split into B, C and X
    auto bcx = llm_build_context::llm_build_lora_mm(lctx, bctx.ctx0, layer.ssm_in, input);
    bctx.cb(bcx, "model.layers.{}.conv.in_proj", il);

    const int64_t chunk = bcx->ne[0] / 3;
    GGML_ASSERT(chunk == hparams.n_embd && bcx->ne[0] % 3 == 0);
    auto b = ggml_view_2d(bctx.ctx0, bcx, chunk, n_tokens, bcx->nb[1], 0);
    auto c = ggml_view_2d(bctx.ctx0, bcx, chunk, n_tokens, bcx->nb[1],
            (size_t) chunk * ggml_element_size(bcx));
    auto x = ggml_view_2d(bctx.ctx0, bcx, chunk, n_tokens, bcx->nb[1],
            (size_t) 2 * chunk * ggml_element_size(bcx));

    // the short-conv gate is elementwise, before the causal convolution
    auto bx = ggml_mul(bctx.ctx0, b, x);
    bctx.cb(bx, "model.layers.{}.conv.gated_input", il);

    // SSM_CONV returns [convolved input | d_conv columns of the new state]
    auto conv_raw = ggml_ssm_conv(bctx.ctx0, conv_state, bx, layer.ssm_conv1d, state_seq, nullptr);
    bctx.cb(conv_raw, "model.layers.{}.conv.conv", il);

    // persist columns 1..d_conv-1 of the new state (column 0 is the current input)
    auto new_state = ggml_view_2d(bctx.ctx0, conv_raw, d_conv - 1, hparams.n_embd,
            d_conv * ggml_element_size(conv_raw),
            (size_t) (hparams.n_embd * n_tokens + 1) * ggml_element_size(conv_raw));
    auto state_copy = ggml_cpy(bctx.ctx0, new_state, state_dst);
    bctx.cb(state_copy, "shortconv_state_cpy", il);
    ggml_build_forward_expand(gf, state_copy);

    auto conv_out = ggml_view_2d(bctx.ctx0, conv_raw, hparams.n_embd, n_tokens,
            hparams.n_embd * ggml_element_size(conv_raw), 0);
    auto y = ggml_mul(bctx.ctx0, c, conv_out);
    y = llm_build_context::llm_build_lora_mm(lctx, bctx.ctx0, layer.ssm_out, y);
    bctx.cb(y, "model.layers.{}.conv.out_proj", il);

    return y;
}

ggml_cgraph * llm_build_context::build_lfm2() {
    ggml_cgraph * gf = new_graph_custom();

    auto inpL = llm_build_inp_embd(ctx0, lctx, hparams, batch, model.tok_embd, cb);

    auto inp_pos = build_inp_pos();
    auto KQ_mask = build_inp_KQ_mask();
    auto inp_out_ids = n_tokens > 1 ? build_inp_out_ids() : nullptr;

    // one-slot sequence map; the state row is selected from s_l below
    lctx.inp_s_seq_qnext = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, 1, n_tokens);
    cb(lctx.inp_s_seq_qnext, "inp_s_seq_qnext", -1);
    ggml_set_input(lctx.inp_s_seq_qnext);

    std::vector<llama_seq_id> seq_ids(n_tokens, 0);
    if (batch.n_seq_id != nullptr && batch.seq_id != nullptr) {
        for (int64_t i = 0; i < n_tokens; ++i) {
            GGML_ASSERT(batch.n_seq_id[i] > 0);
            // one-sequence-per-token contract, same as the qnext hybrid path
            GGML_ASSERT(batch.n_seq_id[i] == 1);
            seq_ids[i] = batch.seq_id[i][0];
        }
    }

    const uint32_t qnext_slots = llama_kv_qnext_state_slots(kv_self);
    GGML_ASSERT(qnext_slots > 0);
    for (const auto seq_id : seq_ids) {
        GGML_ASSERT(seq_id >= 0 && (uint32_t) seq_id < qnext_slots);
    }

    const int64_t n_embd_head = hparams.n_embd_head_k(0);
    const float KQ_scale = hparams.f_attention_scale == 0.0f
        ? 1.0f / sqrtf(float(n_embd_head)) : hparams.f_attention_scale;

    for (int il = 0; il < n_layer; ++il) {
        auto prev = inpL;
        ggml_tensor * op_out;

        auto norm = llm_build_norm(ctx0, inpL, hparams, model.layers[il].attn_norm,
                nullptr, LLM_NORM_RMS, cb, il);
        cb(norm, "operator_norm", il);

        if (hparams.is_recurrent(il)) {
            const bool all_same = std::all_of(seq_ids.begin() + 1, seq_ids.end(),
                    [&](llama_seq_id id) { return id == seq_ids.front(); });
            if (all_same) {
                op_out = build_lfm2_shortconv(*this, gf, norm, lctx.inp_s_seq_qnext,
                        (uint32_t) seq_ids.front(), batch.pos != nullptr && batch.pos[0] == 0, il);
            } else {
                // mixed sequences: one token at a time
                op_out = nullptr;
                for (int64_t i = 0; i < n_tokens; ++i) {
                    auto norm_i = ggml_view_2d(ctx0, norm, norm->ne[0], 1, norm->nb[1],
                            (size_t) i * norm->nb[1]);
                    auto seq_i = ggml_view_2d(ctx0, lctx.inp_s_seq_qnext, 1, 1,
                            lctx.inp_s_seq_qnext->nb[1], (size_t) i * lctx.inp_s_seq_qnext->nb[1]);
                    auto out_i = build_lfm2_shortconv(*this, gf, norm_i, seq_i,
                            (uint32_t) seq_ids[i], batch.pos != nullptr && batch.pos[i] == 0, il);
                    op_out = op_out == nullptr ? out_i : ggml_concat(ctx0, op_out, out_i, 1);
                }
            }
        } else {
            // the shared operator norm above is the input to GQA (null attn norm: no double norm)
            op_out = build_std_attention(gf, nullptr, norm,
                    inp_pos, nullptr, nullptr, KQ_mask, nullptr, nullptr,
                    KQ_scale, 0.0f, hparams.n_swa, il,
                    true, false, false, false, false);
        }

        if (il == n_layer - 1 && inp_out_ids != nullptr) {
            op_out = ggml_get_rows(ctx0, op_out, inp_out_ids);
            prev = ggml_get_rows(ctx0, prev, inp_out_ids);
        }

        inpL = ggml_add(ctx0, prev, op_out);
        cb(inpL, "operator_residual", il);

        inpL = llm_build_ffn(ctx0, lctx, model.layers[il].ffn_norm, inpL,
                model.layers[il].ffn_up, nullptr, nullptr,
                model.layers[il].ffn_gate, nullptr, nullptr,
                model.layers[il].ffn_down, nullptr, nullptr,
                nullptr, LLM_FFN_SILU, LLM_FFN_PAR, cb, il, gf, true, false);
        inpL = lctx.cvec.apply_to(ctx0, inpL, il);
        cb(inpL, "l_out", il);
    }

    // the embedding norm is the final norm; the LM head is tied to the token embeddings
    inpL = llm_build_norm(ctx0, inpL, hparams, model.tok_norm, nullptr,
            LLM_NORM_RMS, cb, -1);
    cb(inpL, "embedding_norm", -1);
    auto cur = build_output(lctx, ctx0, inpL, model.output, nullptr, cb);
    cb(cur, "result_output", -1);
    ggml_build_forward_expand(gf, cur);
    return gf;
}

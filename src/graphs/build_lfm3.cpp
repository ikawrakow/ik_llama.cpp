#include "../llama-build-context.h"
#include "../llama-model.h"
#include "../llama-context.h"

#include <algorithm>
#include <cmath>
#include <vector>

// LFM3 dense: shortconv + gated attention (full / sliding-window).
ggml_cgraph * llm_build_context::build_lfm3() {
    ggml_cgraph * gf = new_graph_custom();

    auto inpL = llm_build_inp_embd(ctx0, lctx, hparams, batch, model.tok_embd, cb);

    auto inp_pos = build_inp_pos();
    auto KQ_mask = build_inp_KQ_mask();
    auto KQ_mask_swa = hparams.n_swa > 0
        ? (kv_self.any_compacted() ? build_swa_mask_for_graph(hparams.n_swa, true)
                                   : build_inp_KQ_mask_swa())
        : nullptr;
    auto inp_out_ids = n_tokens > 1 ? build_inp_out_ids() : nullptr;

    lctx.inp_s_seq_qnext = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, 1, n_tokens);
    cb(lctx.inp_s_seq_qnext, "inp_s_seq_qnext", -1);
    ggml_set_input(lctx.inp_s_seq_qnext);

    std::vector<llama_seq_id> seq_ids(n_tokens, 0);
    if (batch.n_seq_id != nullptr && batch.seq_id != nullptr) {
        for (int64_t i = 0; i < n_tokens; ++i) {
            GGML_ASSERT(batch.n_seq_id[i] > 0);
            GGML_ASSERT(batch.n_seq_id[i] == 1);
            seq_ids[i] = batch.seq_id[i][0];
        }
    }

    const uint32_t qnext_slots = llama_kv_qnext_state_slots(kv_self);
    GGML_ASSERT(qnext_slots > 0);
    for (const auto seq_id : seq_ids) {
        GGML_ASSERT(seq_id >= 0 && (uint32_t) seq_id < qnext_slots);
    }

    const bool all_same = std::all_of(seq_ids.begin() + 1, seq_ids.end(),
            [&](llama_seq_id id) { return id == seq_ids.front(); });

    for (int il = 0; il < n_layer; ++il) {
        auto prev = inpL;
        ggml_tensor * op_out;

        auto norm = llm_build_norm(ctx0, inpL, hparams, model.layers[il].attn_norm,
                nullptr, LLM_NORM_RMS, cb, il);
        cb(norm, "operator_norm", il);

        if (hparams.is_recurrent(il)) {
            if (all_same) {
                op_out = build_lfm_shortconv(gf, norm, lctx.inp_s_seq_qnext,
                        (uint32_t) seq_ids.front(), batch.pos != nullptr && batch.pos[0] == 0, il);
            } else {
                op_out = nullptr;
                for (int64_t i = 0; i < n_tokens; ++i) {
                    auto norm_i = ggml_view_2d(ctx0, norm, norm->ne[0], 1, norm->nb[1],
                            (size_t) i * norm->nb[1]);
                    auto seq_i = ggml_view_2d(ctx0, lctx.inp_s_seq_qnext, 1, 1,
                            lctx.inp_s_seq_qnext->nb[1], (size_t) i * lctx.inp_s_seq_qnext->nb[1]);
                    auto out_i = build_lfm_shortconv(gf, norm_i, seq_i,
                            (uint32_t) seq_ids[i], batch.pos != nullptr && batch.pos[i] == 0, il);
                    op_out = op_out == nullptr ? out_i : ggml_concat(ctx0, op_out, out_i, 1);
                }
            }

            if (model.layers[il].attn_post_norm != nullptr) {
                op_out = llm_build_norm(ctx0, op_out, hparams, model.layers[il].attn_post_norm,
                        nullptr, LLM_NORM_RMS, cb, il);
            }
        } else {
            const bool use_swa = hparams.swa_layers[il] != 0;
            const int  n_swa   = use_swa ? (int) hparams.n_swa : 0;
            const float KQ_scale = hparams.f_attention_scale == 0.0f
                ? 1.0f/std::sqrt(float(hparams.n_embd_head_k(il)))
                : hparams.f_attention_scale;

            op_out = build_std_attention(gf, nullptr, norm,
                    inp_pos, nullptr, use_swa ? nullptr : model.layers[il].rope_freqs,
                    use_swa ? KQ_mask_swa : KQ_mask, nullptr, nullptr,
                    KQ_scale, 0.0f, n_swa, il,
                    true, false, false, false, false,
                    model.layers[il].attn_post_norm, -1, hparams.f_norm_rms_eps);
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
                nullptr, LLM_FFN_SILU, LLM_FFN_PAR, cb, il, gf, true, false,
                nullptr, model.layers[il].ffn_post_norm, hparams.f_norm_rms_eps);
        inpL = lctx.cvec.apply_to(ctx0, inpL, il);
        cb(inpL, "l_out", il);
    }

    auto cur = build_output(lctx, ctx0, inpL, model.output, model.output_norm, cb);
    cb(cur, "result_output", -1);
    ggml_build_forward_expand(gf, cur);
    return gf;
}

#include "../llama-build-context.h"
#include "../llama-model.h"
#include "../llama-context.h"

ggml_cgraph* llm_build_context::build_spark2_5() {
    ggml_cgraph * gf = new_graph_custom();

    const int64_t n_embd_head = hparams.n_embd_head_v(0);
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k(0));

    ggml_tensor * inpL = llm_build_inp_embd(ctx0, lctx, hparams, batch, model.tok_embd, cb);
    ggml_tensor * inp_pos = build_inp_pos();

    ggml_tensor * inp_out_ids = n_tokens > 1 ? build_inp_out_ids() : nullptr;

    // KQ_mask (mask for 1 head, it will be broadcasted to all heads)
    ggml_tensor * KQ_mask = build_inp_KQ_mask();

    // SWA layers use the compacted-window mask when available.
    ggml_tensor * KQ_mask_swa = nullptr;
    if (hparams.n_swa > 0) {
        KQ_mask_swa = kv_self.any_compacted() ? build_swa_mask_for_graph(hparams.n_swa, true)
                                              : build_inp_KQ_mask_swa();
    }

    // f_attention_scale is applied to Q by build_std_attention().
    const float kq_scale = hparams.f_attention_scale == 0.0f ? 1.0f/sqrtf(float(n_embd_head)) : 1.f;

    for (int il = 0; il < n_layer; ++il) {
        const bool is_swa = KQ_mask_swa != nullptr && hparams.swa_layers[il] != 0;

        ggml_tensor * KQ_mask_l = is_swa ? KQ_mask_swa : KQ_mask;
        const int     n_swa_l   = is_swa ? (int) hparams.n_swa : 0;

        ggml_tensor * rope_factors = is_swa ? nullptr : build_rope_factors(il);

        // attention + residual (build_std_attention reduces inp_out_ids on the last layer)
        ggml_tensor * cur = build_std_attention(gf, model.layers[il].attn_norm, inpL,
                inp_pos, il == n_layer - 1 ? inp_out_ids : nullptr, rope_factors,
                KQ_mask_l, nullptr, nullptr, kq_scale, hparams.f_attention_scale,
                n_swa_l, il, true, false, true);

        cur = llm_build_ffn(ctx0, lctx, model.layers[il].ffn_norm, cur,
                model.layers[il].ffn_up,   nullptr, nullptr,
                model.layers[il].ffn_gate, nullptr, nullptr,
                model.layers[il].ffn_down, nullptr, nullptr,
                nullptr,
                LLM_FFN_GELU, LLM_FFN_PAR, cb, il, gf, true);
        cb(cur, "ffn_out", il);

        cur = lctx.cvec.apply_to(ctx0, cur, il);
        cb(cur, "l_out", il);

        inpL = cur;
    }

    ggml_tensor * cur = build_output(lctx, ctx0, inpL, model.output, model.output_norm, cb);
    cb(cur, "result_output", -1);

    ggml_build_forward_expand(gf, cur);

    return gf;
}

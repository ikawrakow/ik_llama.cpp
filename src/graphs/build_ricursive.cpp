#include "../llama-build-context.h"
#include "../llama-model.h"
#include "../llama-context.h"

ggml_cgraph * llm_build_context::build_ricursive() {
    ggml_cgraph * gf = new_graph_custom();

    ggml_tensor * inpL = llm_build_inp_embd(ctx0, lctx, hparams, batch, model.tok_embd, cb);
    ggml_tensor * inp_vres = inpL;

    ggml_tensor * inp_pos     = build_inp_pos();
    ggml_tensor * inp_out_ids = n_tokens > 1 ? build_inp_out_ids() : nullptr;

    ggml_tensor * KQ_mask     = build_inp_KQ_mask();
    ggml_tensor * KQ_mask_swa = kv_self.any_compacted()
        ? build_swa_mask_for_graph(hparams.n_swa, true)
        : build_inp_KQ_mask_swa();

    const float kq_scale = 1.0f/sqrtf(float(n_embd_head_k));

    for (int il = 0; il < n_layer; ++il) {
        const bool is_swa = hparams.swa_layers[il];

        auto cur = build_std_attention(gf, model.layers[il].attn_norm, inpL,
                inp_pos, il == n_layer - 1 ? inp_out_ids : nullptr, nullptr,
                is_swa ? KQ_mask_swa : KQ_mask, nullptr, nullptr, kq_scale, 0.0f,
                is_swa ? hparams.n_swa : 0, il, hparams.has_rope(il), false, true, false, false,
                model.layers[il].attn_post_norm, -1, hparams.f_norm_rms_eps, nullptr,
                nullptr, nullptr, nullptr, inp_vres);

        cur = llm_build_ffn(ctx0, lctx, model.layers[il].ffn_norm, cur,
                model.layers[il].ffn_up,   nullptr, nullptr,
                model.layers[il].ffn_gate, nullptr, nullptr,
                model.layers[il].ffn_down, nullptr, nullptr,
                nullptr,
                LLM_FFN_SILU, LLM_FFN_PAR, cb, il, gf, true, false, nullptr,
                model.layers[il].ffn_post_norm, hparams.f_norm_rms_eps);
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

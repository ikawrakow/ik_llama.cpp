#include "../llama-build-context.h"
#include "../llama-model.h"
#include "../llama-context.h"

ggml_cgraph * llm_build_context::build_cohere2_moe() {
    ggml_cgraph * gf = new_graph_custom();

    const int64_t n_embd_head = hparams.n_embd_head_v(0);
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k(0));
    const float kq_scale = 1.0f / sqrtf(float(n_embd_head));

    ggml_tensor * inpL = llm_build_inp_embd(ctx0, lctx, hparams, batch, model.tok_embd, cb);
    ggml_tensor * inp_pos = build_inp_pos();
    ggml_tensor * KQ_mask = build_inp_KQ_mask();
    ggml_tensor * KQ_mask_swa = build_inp_KQ_mask_swa();

    for (int il = 0; il < n_layer; ++il) {
        const bool is_sliding = hparams.swa_layers[il];
        const bool force_rope = il < (int) hparams.n_layer_dense_lead;
        ggml_tensor * KQ_mask_l = is_sliding ? KQ_mask_swa : KQ_mask;

        ggml_tensor * attn_out = build_std_attention(gf, model.layers[il].attn_norm, inpL, inp_pos, nullptr, nullptr,
                KQ_mask_l, nullptr, nullptr, kq_scale, 0.f,
                is_sliding ? hparams.n_swa : 0, il, is_sliding || force_rope, false, true, false);
        cb(attn_out, "attn_out", il);

        if (il == n_layer - 1 && n_tokens > 1) {
            ggml_tensor * inp_out_ids = build_inp_out_ids();
            attn_out = ggml_get_rows(ctx0, attn_out, inp_out_ids);
            inpL = ggml_get_rows(ctx0, inpL, inp_out_ids);
        }

        attn_out->op_params[3] = 1;

        ggml_tensor * cur;
        if (model.layers[il].ffn_gate_inp == nullptr) {
            cur = llm_build_ffn(ctx0, lctx, model.layers[il].attn_norm, inpL,
                    model.layers[il].ffn_up,   nullptr, nullptr,
                    model.layers[il].ffn_gate, nullptr, nullptr,
                    model.layers[il].ffn_down, nullptr, nullptr,
                    nullptr, LLM_FFN_SILU, LLM_FFN_PAR,
                    cb, il, gf, false, false, attn_out);
        } else {
            cur = llm_build_std_moe_ffn(ctx0, lctx, model.layers[il].attn_norm, inpL,
                    model.layers[il].ffn_gate_inp,  nullptr,
                    model.layers[il].ffn_up_exps,   nullptr,
                    model.layers[il].ffn_gate_exps, nullptr,
                    model.layers[il].ffn_down_exps, nullptr,
                    nullptr,
                    nullptr, nullptr,
                    nullptr, nullptr,
                    nullptr, nullptr,
                    n_expert, n_expert_used,
                    LLM_FFN_SILU, hparams.expert_weights_norm, false, 0.0f,
                    (llm_expert_gating_func_type) hparams.expert_gating_func,
                    LLM_FFN_SILU, cb, il, gf, false, model.layers[il].ffn_up_gate_exps, nullptr, nullptr,
                    attn_out);
        }
        cb(cur, "ffn_out", il);

        cur = lctx.cvec.apply_to(ctx0, cur, il);
        cb(cur, "l_out", il);

        inpL = cur;
    }

    ggml_tensor * cur = inpL;

    cur = llm_build_norm(ctx0, cur, hparams, model.output_norm, nullptr, LLM_NORM_RMS, cb, -1);
    cb(cur, "result_norm", -1);

    if (hparams.f_logit_scale) {
        cur = ggml_scale(ctx0, cur, hparams.f_logit_scale);
        cb(cur, "result_norm_scaled", -1);
    }

    cur = llm_build_lora_mm(lctx, ctx0, model.output, cur);
    cb(cur, "result_output", -1);

    ggml_build_forward_expand(gf, cur);

    return gf;
}

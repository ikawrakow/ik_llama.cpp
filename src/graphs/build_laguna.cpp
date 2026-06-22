#include "../llama-build-context.h"
#include "../llama-model.h"
#include "../llama-context.h"

ggml_cgraph * llm_build_context::build_laguna() {
    ggml_cgraph * gf = new_graph_custom();

    ggml_tensor * inpL        = llm_build_inp_embd(ctx0, lctx, hparams, batch, model.tok_embd, cb);
    ggml_tensor * inp_pos     = build_inp_pos();
    ggml_tensor * inp_out_ids = n_tokens > 1 ? build_inp_out_ids() : nullptr;
    ggml_tensor * KQ_mask     = build_inp_KQ_mask();
    // Laguna M.1 has only global-attention layers and leaves n_swa at zero; building
    // the SWA mask in that case trips the generic SWA precondition.
    ggml_tensor * KQ_mask_swa = hparams.n_swa > 0 ? build_inp_KQ_mask_swa() : nullptr;

    for (int il = 0; il < n_layer; ++il) {
        const bool is_swa = hparams.swa_layers[il];
        const int n_swa_l = is_swa ? hparams.n_swa : 0;

        auto KQ_mask_l = is_swa ? KQ_mask_swa : KQ_mask;
        // If a future Laguna GGUF marks SWA layers, it must also carry a real
        // sliding-window size so those layers get an SWA mask.
        GGML_ASSERT(KQ_mask_l != nullptr);
        auto rope_factors = is_swa ? nullptr : build_rope_factors(il);

        auto cur = build_std_attention(gf, model.layers[il].attn_norm, inpL,
                        inp_pos, il == n_layer - 1 ? inp_out_ids : nullptr, rope_factors,
                        KQ_mask_l, nullptr, nullptr, 1.0f / sqrtf(float(n_embd_head_k)), 0.0f, n_swa_l, il, true, false, true);

        if (model.layers[il].ffn_gate_inp == nullptr) {
            cur = llm_build_ffn(ctx0, lctx, model.layers[il].ffn_norm, cur, //ffn_inp,
                    model.layers[il].ffn_up,   nullptr, nullptr,
                    model.layers[il].ffn_gate, nullptr, nullptr,
                    model.layers[il].ffn_down, nullptr, nullptr,
                    nullptr,
                    LLM_FFN_SILU, LLM_FFN_PAR, cb, il, gf, true);
        } else {
            cur = llm_build_std_moe_ffn(ctx0, lctx, model.layers[il].ffn_norm, cur, //ffn_inp,
                    model.layers[il].ffn_gate_inp,  model.layers[il].ffn_gate_inp_b,
                    model.layers[il].ffn_up_exps,   model.layers[il].ffn_up_exps_b,
                    model.layers[il].ffn_gate_exps, model.layers[il].ffn_gate_exps_b,
                    model.layers[il].ffn_down_exps, model.layers[il].ffn_down_exps_b,
                    model.layers[il].ffn_exp_probs_b,
                    model.layers[il].ffn_up_shexp,    nullptr,
                    model.layers[il].ffn_gate_shexp,  nullptr,
                    model.layers[il].ffn_down_shexp,  nullptr,
                    n_expert, n_expert_used,
                    LLM_FFN_SILU, hparams.expert_weights_norm, hparams.expert_weights_scale != 0.0f, hparams.expert_weights_scale,
                    (llm_expert_gating_func_type) hparams.expert_gating_func,
                    LLM_FFN_SILU, cb, il, gf, true, model.layers[il].ffn_up_gate_exps);
        }

        cur = lctx.cvec.apply_to(ctx0, cur, il);
        cb(cur, "l_out", il);

        inpL = cur;
    }

    ggml_tensor * cur = build_output(lctx, ctx0, inpL, model.output, model.output_norm, cb);
    cb(cur, "result_output", -1);

    ggml_build_forward_expand(gf, cur);

    return gf;
}

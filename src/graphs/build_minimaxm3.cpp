#include "../llama-build-context.h"
#include "../llama-model.h"
#include "../llama-context.h"

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

    for (int il = 0; il < n_layer; ++il) {
        ggml_tensor * ffn_inp = build_std_attention(gf, model.layers[il].attn_norm, inpL,
                inp_pos, il == n_layer - 1 ? inp_out_ids : nullptr, nullptr,
                KQ_mask, nullptr, nullptr, 1.0f / sqrtf(float(n_embd_head)), 0.0f, 0,
                il, true, false, true);

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

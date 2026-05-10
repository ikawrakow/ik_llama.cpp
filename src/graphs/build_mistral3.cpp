#include "../llama-build-context.h"
#include "../llama-model.h"
#include "../llama-context.h"

ggml_cgraph * llm_build_context::build_mistral3() {
    auto gf = ggml_new_graph_custom(ctx0, model.max_nodes(n_tokens), false);
    const int64_t n_embd_head = hparams.n_embd_head_v(0);

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k(0));
    GGML_ASSERT(n_embd_head == hparams.n_rot);

    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = llm_build_inp_embd(ctx0, lctx, hparams, batch, model.tok_embd, cb);

    // inp_pos - contains the positions
    struct ggml_tensor * inp_pos = build_inp_pos();

    // (optional) temperature tuning
    ggml_tensor * inp_attn_scale = nullptr;
    if (hparams.f_attn_temp_scale != 0.0f) {
        inp_attn_scale = build_input_scale(n_tokens);
    }

    ggml_tensor * KQ_mask = build_inp_KQ_mask();

    ggml_tensor * inp_out_ids = n_tokens > 1 ? build_inp_out_ids() : nullptr;

    //const float kq_scale = hparams.f_attention_scale == 0.0f ? 1.0f/sqrtf(float(n_embd_head)) : hparams.f_attention_scale;
    const float kq_scale = hparams.f_attention_scale == 0.0f ? 1.0f/sqrtf(float(n_embd_head)) : 1.f;

    for (int il = 0; il < n_layer; ++il) {

        auto rope_factors = build_rope_factors(il);

        cur = build_std_attention(gf, model.layers[il].attn_norm, inpL, inp_pos, il == n_layer - 1 ? inp_out_ids : nullptr,
                rope_factors, KQ_mask, nullptr, inp_attn_scale, kq_scale, hparams.f_attention_scale, 0, il, true, false, true);

        // feed-forward network (non-MoE)
        if (model.layers[il].ffn_gate_inp == nullptr) {
            // non-MoE
            cur = llm_build_ffn(ctx0, lctx, model.layers[il].ffn_norm, cur,
                    model.layers[il].ffn_up,   model.layers[il].ffn_up_b,   nullptr,
                    model.layers[il].ffn_gate, model.layers[il].ffn_gate_b, nullptr,
                    model.layers[il].ffn_down, model.layers[il].ffn_down_b, nullptr,
                    NULL,
                    LLM_FFN_SILU, LLM_FFN_PAR, cb, il, gf, true);
            cb(cur, "ffn_out", il);
        } else {
            // MoE branch
            cur = llm_build_std_moe_ffn(ctx0, lctx, model.layers[il].ffn_norm, cur,
                    model.layers[il].ffn_gate_inp,  nullptr,
                    model.layers[il].ffn_up_exps,   nullptr,
                    model.layers[il].ffn_gate_exps, nullptr,
                    model.layers[il].ffn_down_exps, nullptr,
                    model.layers[il].ffn_exp_probs_b,
                    nullptr,  nullptr, // we don't have shared experts
                    nullptr,  nullptr,
                    nullptr,  nullptr,
                    n_expert, n_expert_used,
                    LLM_FFN_SILU, true, false, 0.0f,
                    LLM_EXPERT_GATING_FUNC_SOFTMAX,
                    LLM_FFN_SILU, cb, il, gf, true);
        }
        cb(cur, "ffn_out", il);

        cur = lctx.cvec.apply_to(ctx0, cur, il);
        cb(cur, "l_out", il);

        // input for next layer
        inpL = cur;
    }
    cur = inpL;

    cur = build_output(lctx, ctx0, cur, model.output, model.output_norm, cb);
    cb(cur, "result_output", -1);

    ggml_build_forward_expand(gf, cur);

    return gf;
}


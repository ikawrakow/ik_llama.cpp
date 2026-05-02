#include "../llama-build-context.h"
#include "../llama-model.h"
#include "../llama-context.h"

ggml_cgraph * llm_build_context::build_openai_moe() {
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, model.max_nodes(n_tokens), false);

    const int64_t n_embd_head = hparams.n_embd_head_v(0);
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k(0));

    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = llm_build_inp_embd(ctx0, lctx, hparams, batch, model.tok_embd, cb);

    ggml_tensor * inp_pos = build_inp_pos();
    auto inp_out_ids = n_tokens > 1 ? build_inp_out_ids() : nullptr;

    struct ggml_tensor * KQ_mask     = build_inp_KQ_mask();
    struct ggml_tensor * KQ_mask_swa = build_inp_KQ_mask_swa();
    const float kq_scale = 1.0f / sqrtf(float(n_rot));

    const int sliding_window_pattern = 2;

    for (int il = 0; il < n_layer; ++il) {
        const bool is_sliding = il % sliding_window_pattern < (sliding_window_pattern - 1);

        struct ggml_tensor * KQ_mask_l = is_sliding ? KQ_mask_swa : KQ_mask;

        cur = build_std_attention(gf, model.layers[il].attn_norm, inpL,
                inp_pos, il == n_layer - 1 ? inp_out_ids : nullptr, nullptr, KQ_mask_l,
                model.layers[il].attn_sinks, nullptr, kq_scale, 0.0f, is_sliding ? hparams.n_swa : 0, il, true, false, true);

        bool use_dup_bias = cur->ne[1] < 32 && model.layers[il].ffn_up_exps_b_dup &&
                                               model.layers[il].ffn_gate_exps_b_dup &&
                                               model.layers[il].ffn_down_exps_b_dup;

        cur = llm_build_std_moe_ffn(ctx0, lctx, model.layers[il].ffn_norm, cur,
                model.layers[il].ffn_gate_inp,  model.layers[il].ffn_gate_inp_b,
                model.layers[il].ffn_up_exps,   use_dup_bias ? model.layers[il].ffn_up_exps_b_dup : model.layers[il].ffn_up_exps_b,
                model.layers[il].ffn_gate_exps, use_dup_bias ? model.layers[il].ffn_gate_exps_b_dup : model.layers[il].ffn_gate_exps_b,
                model.layers[il].ffn_down_exps, use_dup_bias ? model.layers[il].ffn_down_exps_b_dup : model.layers[il].ffn_down_exps_b,
                nullptr,
                nullptr,  nullptr, nullptr,  nullptr, nullptr,  nullptr, // no shared experts
                n_expert, n_expert_used,
                LLM_FFN_SWIGLU_OAI_MOE, false, false, 0.0f,
                LLM_EXPERT_GATING_FUNC_TYPE_SOFTMAX_WEIGHT,
                LLM_FFN_SWIGLU_OAI_MOE, cb, il, gf, true,
                model.layers[il].ffn_up_gate_exps, model.layers[il].ffn_up_gate_exps_b);

        cur = lctx.cvec.apply_to(ctx0, cur, il);
        cb(cur, "l_out", il);

        // input for next layer
        inpL = cur;
    }

    cur = build_output(lctx, ctx0, inpL, model.output, model.output_norm, cb);
    cb(cur, "result_output", -1);

    ggml_build_forward_expand(gf, cur);

    return gf;

}

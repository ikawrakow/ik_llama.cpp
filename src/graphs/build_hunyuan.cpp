#include "../llama-build-context.h"
#include "../llama-model.h"
#include "../llama-context.h"

ggml_cgraph * llm_build_context::build_hunyuan_moe() {
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, model.max_nodes(n_tokens), false);

    const int64_t n_embd_head = hparams.n_embd_head_v(0);

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k(0));
    GGML_ASSERT(n_embd_head == hparams.n_rot);

    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = llm_build_inp_embd(ctx0, lctx, hparams, batch, model.tok_embd, cb);

    ggml_tensor * inp_pos = build_inp_pos();

    struct ggml_tensor * KQ_mask = build_inp_KQ_mask();

    const float kq_scale = 1.0f / sqrtf(float(n_embd_head));

    ggml_tensor * inp_out_ids = n_tokens > 1 ? build_inp_out_ids() : nullptr;

    for (int il = 0; il < n_layer; ++il) {

        cur = build_std_attention(gf, model.layers[il].attn_norm, inpL,
                inp_pos, il == n_layer - 1 ? inp_out_ids : nullptr, nullptr, KQ_mask,
                nullptr, nullptr, kq_scale, 0.0f, 0, il, true, false, true);

        cur = llm_build_std_moe_ffn(ctx0, lctx, model.layers[il].ffn_norm, cur,
                model.layers[il].ffn_gate_inp,  nullptr,
                model.layers[il].ffn_up_exps,   nullptr,
                model.layers[il].ffn_gate_exps, nullptr,
                model.layers[il].ffn_down_exps, nullptr,
                nullptr,
                model.layers[il].ffn_up_shexp,    nullptr, // we don't have shared expert biases?
                model.layers[il].ffn_gate_shexp,  nullptr,
                model.layers[il].ffn_down_shexp,  nullptr,
                n_expert, n_expert_used,
                LLM_FFN_SILU, true, false, 0.0f,
                LLM_EXPERT_GATING_FUNC_SOFTMAX,
                LLM_FFN_SILU, cb, il, gf, true, model.layers[il].ffn_up_gate_exps);

        cur = lctx.cvec.apply_to(ctx0, cur, il);
        cb(cur, "l_out", il);

        inpL = cur;
    }

    cur = inpL;

    cur = build_output(lctx, ctx0, cur, model.output, model.output_norm, cb);
    cb(cur, "result_output", -1);

    ggml_build_forward_expand(gf, cur);
    return gf;

}

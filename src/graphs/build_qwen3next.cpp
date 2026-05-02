#include "../llama-build-context.h"
#include "../llama-model.h"
#include "../llama-context.h"
#include "../llama-delta-net.h"

ggml_cgraph * llm_build_context::build_qwen3next() {

    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, model.max_nodes(n_tokens), false);

    delta_net delta(lctx, batch);

    const int64_t n_embd_head = hparams.n_embd_head_v(0);
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k(0));

    ggml_tensor * inpL = llm_build_inp_embd(ctx0, lctx, hparams, batch, model.tok_embd, cb);
    ggml_tensor * inp_pos = build_inp_pos();
    ggml_tensor * inp_out_ids = n_tokens > 1 ? build_inp_out_ids() : nullptr;
    ggml_tensor * KQ_mask = build_inp_KQ_mask();

    lctx.inp_s_seq_qnext = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, 1, n_tokens);
    cb(lctx.inp_s_seq_qnext, "inp_s_seq_qnext", -1);
    ggml_set_input(lctx.inp_s_seq_qnext);

    float KQ_scale = hparams.f_attention_scale == 0.0f ? 1.0f / sqrtf(float(n_embd_head)) : hparams.f_attention_scale;

    ggml_tensor * cur = nullptr;

    for (int il = 0; il < n_layer; ++il) {

        GGML_ASSERT(model.layers[il].attn_norm != nullptr);
        GGML_ASSERT(model.layers[il].attn_post_norm != nullptr);

        const bool has_moe = model.layers[il].ffn_gate_inp != nullptr;
        const bool has_dense = model.layers[il].ffn_gate != nullptr &&
                               model.layers[il].ffn_up != nullptr &&
                               model.layers[il].ffn_down != nullptr;
        GGML_ASSERT(has_moe || has_dense);
        if (has_moe) {
            GGML_ASSERT(model.layers[il].ffn_up_exps != nullptr);
            GGML_ASSERT(model.layers[il].ffn_gate_exps != nullptr);
            GGML_ASSERT(model.layers[il].ffn_down_exps != nullptr);
        }


        if (hparams.is_recurrent(il)) {
            cur = delta.build_layer_attn_linear(ctx0, gf, inpL, il == n_layer - 1 ? inp_out_ids : nullptr, il, cb);
        } else {
            cur = build_std_attention(gf, model.layers[il].attn_norm, inpL, inp_pos, il == n_layer - 1 ? inp_out_ids : nullptr, nullptr,
                    KQ_mask, nullptr, nullptr, KQ_scale, 0.0f, 0, il, true, false, true, false, false);
        }

        if (!model.layers[il].ffn_gate_inp) {
            // dense FFN
            cur = llm_build_ffn(ctx0, lctx, model.layers[il].ffn_norm, cur,
                    model.layers[il].ffn_up,   nullptr, nullptr,
                    model.layers[il].ffn_gate, nullptr, nullptr,
                    model.layers[il].ffn_down, nullptr, nullptr,
                    nullptr,
                    LLM_FFN_SILU, LLM_FFN_PAR, cb, il, gf, true);
            cb(cur, "ffn_out", il);
        } else {
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
                    LLM_FFN_SILU, cb, il, gf, true, model.layers[il].ffn_up_gate_exps, nullptr, model.layers[il].ffn_gate_inp_shexp);
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

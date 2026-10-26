#include "../llama-build-context.h"
#include "../llama-model.h"
#include "../llama-context.h"

// Flint (RecursiveSI): Qwen2-MoE with leading dense layers, SWA/NoPE and attention sinks.
ggml_cgraph * llm_build_context::build_flint() {
    ggml_cgraph * gf = new_graph_custom();

    struct ggml_tensor * cur;
    struct ggml_tensor * inpL;

    inpL = llm_build_inp_embd(ctx0, lctx, hparams, batch, model.tok_embd, cb);

    // inp_pos - contains the positions
    struct ggml_tensor * inp_pos = build_inp_pos();
    struct ggml_tensor * inp_out_ids = n_tokens > 1 ? build_inp_out_ids() : nullptr;

    // KQ_mask (mask for 1 head, it will be broadcasted to all heads)
    struct ggml_tensor * KQ_mask     = build_inp_KQ_mask();
    // Sliding-window mask (windowed cache when --swa-compress)
    struct ggml_tensor * KQ_mask_swa = hparams.n_swa > 0
        ? kv_self.any_compacted()
            ? build_swa_mask_for_graph(hparams.n_swa, true)
            : build_inp_KQ_mask_swa()
        : KQ_mask;

    for (int il = 0; il < n_layer; ++il) {
        const bool is_sliding = hparams.swa_layers[il];
        auto KQ_mask_l = is_sliding ? KQ_mask_swa : KQ_mask;

        const bool is_final_layer = il == n_layer - 1;
        struct ggml_tensor * attn_out_ids = is_final_layer ? inp_out_ids : nullptr;

        // global-attention layers are NoPE
        const bool do_rope = hparams.has_rope(il);

        cur = build_std_attention(gf, model.layers[il].attn_norm, inpL,
                do_rope ? inp_pos : nullptr, attn_out_ids, nullptr,
                KQ_mask_l, model.layers[il].attn_sinks,
                nullptr, 1.0f/sqrtf(float(n_embd_head_k)), 0.0f,
                is_sliding ? hparams.n_swa : 0, il, do_rope, false, true);

        auto ffn_inp = cur;

        if (model.layers[il].ffn_gate_inp == nullptr) {
            // leading dense layers
            cur = llm_build_ffn(ctx0, lctx, model.layers[il].ffn_norm, ffn_inp,
                    model.layers[il].ffn_up,   NULL, NULL,
                    model.layers[il].ffn_gate, NULL, NULL,
                    model.layers[il].ffn_down, NULL, NULL,
                    NULL,
                    LLM_FFN_SILU, LLM_FFN_PAR, cb, il, gf, true);
            cb(cur, "ffn_out", il);
        } else {
            cur = llm_build_std_moe_ffn(ctx0, lctx, model.layers[il].ffn_norm, ffn_inp,
                    model.layers[il].ffn_gate_inp,  nullptr,
                    model.layers[il].ffn_up_exps,   nullptr,
                    model.layers[il].ffn_gate_exps, nullptr,
                    model.layers[il].ffn_down_exps, nullptr,
                    model.layers[il].ffn_exp_probs_b,
                    model.layers[il].ffn_up_shexp,   nullptr,
                    model.layers[il].ffn_gate_shexp, nullptr,
                    model.layers[il].ffn_down_shexp, nullptr,
                    n_expert, n_expert_used,
                    LLM_FFN_SILU, hparams.expert_weights_norm, true, hparams.expert_weights_scale,
                    (llm_expert_gating_func_type) hparams.expert_gating_func,
                    LLM_FFN_SILU, cb, il, gf, true,
                    model.layers[il].ffn_up_gate_exps, nullptr,
                    model.layers[il].ffn_gate_inp_shexp);
        }

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

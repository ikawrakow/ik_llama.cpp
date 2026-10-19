#include "../llama-build-context.h"
#include "../llama-model.h"
#include "../llama-context.h"

// Maple (deepgrove): 20B-A1B ternary MoE (per-head Q/K norm, interleaved SWA, MoE SiLU clamp).
ggml_cgraph * llm_build_context::build_maple() {
    ggml_cgraph * gf = new_graph_custom();

    const int64_t n_embd_head = hparams.n_embd_head_v(0);
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k(0));
    // Maple uses partial rotary, so no n_embd_head == n_rot assert here.

    struct ggml_tensor * cur;
    struct ggml_tensor * inpL;

    inpL = llm_build_inp_embd(ctx0, lctx, hparams, batch, model.tok_embd, cb);

    // inp_pos - contains the positions
    struct ggml_tensor * inp_pos = build_inp_pos();

    // KQ_mask (mask for 1 head, it will be broadcasted to all heads)
    struct ggml_tensor * KQ_mask     = build_inp_KQ_mask();
    // Sliding-window mask (windowed cache when --swa-compress)
    struct ggml_tensor * KQ_mask_swa = hparams.n_swa > 0
        ? kv_self.any_compacted()
            ? build_swa_mask_for_graph(hparams.n_swa, true)
            : build_inp_KQ_mask_swa()
        : nullptr;

    ggml_tensor * inp_out_ids = nullptr;

    for (int il = 0; il < n_layer; ++il) {

        if (il == n_layer - 1 && n_tokens > 1) {
            inp_out_ids = build_inp_out_ids();
        }

        const bool is_swa = hparams.swa_layers[il];
        ggml_tensor * KQ_mask_l = is_swa ? KQ_mask_swa : KQ_mask;
        ggml_tensor * inp_pos_l  = is_swa ? inp_pos : nullptr; // global layers are NoPE

        cur = build_std_attention(gf, model.layers[il].attn_norm, inpL, inp_pos_l, inp_out_ids, nullptr,
                KQ_mask_l, nullptr, nullptr, 1.0f/sqrtf(float(n_embd_head)), 0.0f,
                is_swa ? hparams.n_swa : 0, il, is_swa, false, true);

        auto ffn_inp = cur;

        cur = llm_build_std_moe_ffn(ctx0, lctx, model.layers[il].ffn_norm, ffn_inp,
                model.layers[il].ffn_gate_inp,  nullptr,
                model.layers[il].ffn_up_exps,   nullptr,
                model.layers[il].ffn_gate_exps, nullptr,
                model.layers[il].ffn_down_exps, nullptr,
                model.layers[il].ffn_exp_probs_b,
                nullptr,  nullptr,
                nullptr,  nullptr,
                nullptr,  nullptr,
                n_expert, n_expert_used,
                LLM_FFN_SILU, true, false, 0.0f,
                LLM_EXPERT_GATING_FUNC_SOFTMAX,
                LLM_FFN_SILU, cb, il, gf, true,
                model.layers[il].ffn_up_gate_exps);

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

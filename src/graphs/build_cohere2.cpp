#include "../llama-build-context.h"
#include "../llama-model.h"
#include "../llama-context.h"

ggml_cgraph * llm_build_context::build_cohere2() {
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, model.max_nodes(n_tokens), false);

    const int64_t n_embd_head = hparams.n_embd_head_v(0);
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k(0));
    const float f_logit_scale = hparams.f_logit_scale;

    struct ggml_tensor * cur;
    struct ggml_tensor * inpL;

    inpL = llm_build_inp_embd(ctx0, lctx, hparams, batch, model.tok_embd, cb);

    // inp_pos - contains the positions
    struct ggml_tensor * inp_pos = build_inp_pos();

    // KQ_mask (mask for 1 head, it will be broadcasted to all heads)
    // cohere2 requires different mask for layers using sliding window (SWA)
    struct ggml_tensor * KQ_mask     = build_inp_KQ_mask();
    struct ggml_tensor * KQ_mask_swa = build_inp_KQ_mask_swa();

    // sliding window switch pattern
    const int32_t sliding_window_pattern = 4;

    for (int il = 0; il < n_layer; ++il) {
        // three layers sliding window attention (window size 4096) and ROPE
        // fourth layer uses global attention without positional embeddings
        const bool           is_sliding = il % sliding_window_pattern < (sliding_window_pattern - 1); 
        struct ggml_tensor * KQ_mask_l = is_sliding ? KQ_mask_swa : KQ_mask;

        // self-attention
        auto attn_out = build_std_attention(gf, model.layers[il].attn_norm, inpL, inp_pos, nullptr, nullptr,
                KQ_mask_l, nullptr, nullptr, 1.0f / sqrtf(float(n_embd_head)), 0.f,
                is_sliding ? hparams.n_swa : 0, il, is_sliding, false, true, true);
        cb(attn_out, "attn_out", il);

        if (il == n_layer - 1 && n_tokens > 1) {
            // skip computing output for unused tokens
            struct ggml_tensor * inp_out_ids = build_inp_out_ids();
            attn_out                         = ggml_get_rows(ctx0, attn_out, inp_out_ids);
            inpL                             = ggml_get_rows(ctx0, inpL, inp_out_ids);
        }

        attn_out->op_params[3] = 1; // i.e., turn off the reduce operation as it is not required

        // feed-forward network
        cur = llm_build_ffn(ctx0, lctx, model.layers[il].attn_norm, inpL, model.layers[il].ffn_up, NULL, NULL, model.layers[il].ffn_gate,
                    NULL, NULL, model.layers[il].ffn_down, NULL, NULL, NULL, LLM_FFN_SILU, LLM_FFN_PAR,
                    cb, il, gf, false, true, attn_out);
        cb(cur, "ffn_out", il);

        // add together residual + FFN + self-attention
        cur = lctx.cvec.apply_to(ctx0, cur, il);
        cb(cur, "l_out", il);

        // input for next layer
        inpL = cur;
    }

    cur = inpL;

    cur = llm_build_norm(ctx0, cur, hparams, model.output_norm, NULL, LLM_NORM, cb, -1);
    cb(cur, "result_norm", -1);

    if (f_logit_scale) {
        cur = ggml_scale(ctx0, cur, f_logit_scale);
        cb(cur, "result_norm_scaled", -1);
    }

    // lm_head
    cur = llm_build_lora_mm(lctx, ctx0, model.output, cur);
    cb(cur, "result_output", -1);

    ggml_build_forward_expand(gf, cur);

    return gf;
}

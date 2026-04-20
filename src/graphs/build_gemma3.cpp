#include "../llama-build-context.h"
#include "../llama-model.h"
#include "../llama-context.h"

ggml_cgraph * llm_build_context::build_gemma3() {
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, model.max_nodes(n_tokens), false);

    struct ggml_tensor * cur;
    struct ggml_tensor * inpL;

    inpL = llm_build_inp_embd(ctx0, lctx, hparams, batch, model.tok_embd, cb);

    // important: do not normalize weights for raw embeddings input (i.e. encoded image emdeddings)
    if (batch.token) {
        inpL = ggml_scale(ctx0, inpL, sqrtf(n_embd));
        cb(inpL, "inp_scaled", -1);
    }

    // inp_pos - contains the positions
    struct ggml_tensor * inp_pos = build_inp_pos();

    // KQ_mask (mask for 1 head, it will be broadcasted to all heads)
    // gemma3 requires different mask for layers using sliding window (SWA)
    struct ggml_tensor * KQ_mask     = build_inp_KQ_mask(true);
    struct ggml_tensor * KQ_mask_swa = build_inp_KQ_mask_swa(true);

    // "5-to-1 interleaved attention"
    // 5 layers of local attention followed by 1 layer of global attention
    static const int sliding_window_pattern = 6;

    ggml_tensor * rope_cache   = nullptr;
    ggml_tensor * rope_cache_l = nullptr;
    if (cparams.rope_cache && (rope_type == LLAMA_ROPE_TYPE_NEOX || rope_type == LLAMA_ROPE_TYPE_NORM)) {
        rope_cache = ggml_rope_cache(ctx0, inp_pos, nullptr, n_rot, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow);
        rope_cache_l = ggml_rope_cache(ctx0, inp_pos, nullptr, n_rot, n_rot, rope_type, n_ctx_orig, 10000.0f, 1.0f,
            ext_factor, attn_factor, beta_fast, beta_slow);
    }

    for (int il = 0; il < n_layer; ++il) {
        const bool is_sliding          = (il + 1) % sliding_window_pattern;
        const float freq_base_l        = is_sliding ? 10000.0f    : freq_base;
        const float freq_scale_l       = is_sliding ? 1.0f        : freq_scale;
        struct ggml_tensor * KQ_mask_l = is_sliding ? KQ_mask_swa : KQ_mask;

        // norm
        cur = llm_build_norm(ctx0, inpL, hparams, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, cb, il);
        cb(cur, "attn_norm", il);

        // self-attention
        {
            auto [Qcur, Kcur, Vcur] = llm_build_mul_mat_qkv(gf, cur,
                    model.layers[il].wqkv, nullptr,
                    model.layers[il].wqk, nullptr,
                    model.layers[il].wq, nullptr, model.layers[il].wk, nullptr, model.layers[il].wv, nullptr,
                    model.layers[il].attn_q_norm, model.layers[il].attn_k_norm, 0, il);

            if (rope_cache) {
                auto rcache = is_sliding ? rope_cache_l : rope_cache;
                Qcur = ggml_rope_fast(ctx0, Qcur, rcache);
                Kcur = ggml_rope_fast(ctx0, Kcur, rcache);
            } else {
                Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base_l, freq_scale_l,
                        ext_factor, attn_factor, beta_fast, beta_slow);

                Kcur = ggml_rope_ext(ctx0, Kcur, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base_l, freq_scale_l,
                        ext_factor, attn_factor, beta_fast, beta_slow);
            }
            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);

            cur = llm_build_kv(ctx0, lctx, kv_self, gf, model.layers[il].wo, NULL,
                    Kcur, Vcur, Qcur, KQ_mask_l, n_tokens, kv_head, n_kv, hparams.f_attention_scale, cb, il, nullptr,
                    KQ_mask_l == KQ_mask_swa ? hparams.n_swa : 0);
        }

        cur = llm_build_norm(ctx0, cur, hparams, model.layers[il].attn_post_norm, NULL, LLM_NORM_RMS, cb, il);
        cb(cur, "attn_post_norm", il);

        if (il == n_layer - 1) {
            // skip computing output for unused tokens
            struct ggml_tensor * inp_out_ids = build_inp_out_ids();
            cur  = ggml_get_rows(ctx0,  cur, inp_out_ids);
            inpL = ggml_get_rows(ctx0, inpL, inp_out_ids);
        }

        struct ggml_tensor * sa_out = ggml_add(ctx0, cur, inpL);
        cb(sa_out, "sa_out", il);

        // feed-forward network
        cur = llm_build_ffn(ctx0, lctx, model.layers[il].ffn_norm, sa_out,
                model.layers[il].ffn_up,   NULL, NULL,
                model.layers[il].ffn_gate, NULL, NULL,
                model.layers[il].ffn_down, NULL, NULL,
                NULL,
                LLM_FFN_GELU, LLM_FFN_PAR, cb, il);
        cb(cur, "ffn_out", il);

        cur = llm_build_norm(ctx0, cur, hparams, model.layers[il].ffn_post_norm, NULL, LLM_NORM_RMS, cb, -1);
        cb(cur, "ffn_post_norm", -1);

        cur = ggml_add(ctx0, cur, sa_out);
        cur = lctx.cvec.apply_to(ctx0, cur, il);
        cb(cur, "l_out", il);

        // input for next layer
        inpL = cur;
    }

    cur = inpL;

    cur = llm_build_norm(ctx0, cur, hparams, model.output_norm, NULL, LLM_NORM_RMS, cb, -1);
    cb(cur, "result_norm", -1);

    // lm_head
    cur = llm_build_lora_mm(lctx, ctx0, model.output, cur);

    cb(cur, "result_output", -1);

    ggml_build_forward_expand(gf, cur);

    return gf;
}


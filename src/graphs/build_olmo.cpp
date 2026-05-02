#include "../llama-build-context.h"
#include "../llama-model.h"
#include "../llama-context.h"

// ref: https://allenai.org/olmo
// based on the original build_llama() function, changes:
//   * non-parametric layer norm
//   * clamp qkv
//   * removed bias
//   * removed MoE
ggml_cgraph * llm_build_context::build_olmo() {
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, model.max_nodes(n_tokens), false);

    // mutable variable, needed during the last layer of the computation to skip unused tokens
    int32_t n_tokens = this->n_tokens;

    const int64_t n_embd_head = hparams.n_embd_head_v(0);
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k(0));
    GGML_ASSERT(n_embd_head == hparams.n_rot);

    struct ggml_tensor * cur;
    struct ggml_tensor * inpL;

    inpL = llm_build_inp_embd(ctx0, lctx, hparams, batch, model.tok_embd, cb);

    // inp_pos - contains the positions
    struct ggml_tensor * inp_pos = build_inp_pos();

    // KQ_mask (mask for 1 head, it will be broadcasted to all heads)
    struct ggml_tensor * KQ_mask = build_inp_KQ_mask();

    for (int il = 0; il < n_layer; ++il) {
        struct ggml_tensor * inpSA = inpL;

        // norm
        cur = llm_build_norm(ctx0, inpL, hparams, NULL, NULL, LLM_NORM, cb, il);
        cb(cur, "attn_norm", il);

        // self-attention
        {
            // compute Q and K and RoPE them
            struct ggml_tensor * Qcur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wq, cur);
            cb(Qcur, "Qcur", il);
            if (hparams.f_clamp_kqv > 0.0f) {
                Qcur = ggml_clamp(ctx0, Qcur, -hparams.f_clamp_kqv, hparams.f_clamp_kqv);
                cb(Qcur, "Qcur", il);
            }

            struct ggml_tensor * Kcur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wk, cur);
            cb(Kcur, "Kcur", il);
            if (hparams.f_clamp_kqv > 0.0f) {
                Kcur = ggml_clamp(ctx0, Kcur, -hparams.f_clamp_kqv, hparams.f_clamp_kqv);
                cb(Kcur, "Kcur", il);
            }

            struct ggml_tensor * Vcur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wv, cur);
            cb(Vcur, "Vcur", il);
            if (hparams.f_clamp_kqv > 0.0f) {
                Vcur = ggml_clamp(ctx0, Vcur, -hparams.f_clamp_kqv, hparams.f_clamp_kqv);
                cb(Vcur, "Vcur", il);
            }

            Qcur = ggml_rope_ext(
                    ctx0, ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens), inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );
            cb(Qcur, "Qcur", il);

            Kcur = ggml_rope_ext(
                    ctx0, ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens), inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );
            cb(Kcur, "Kcur", il);

            Kcur = ggml_rope_ext(
                    ctx0, ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens), inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );
            cb(Kcur, "Kcur", il);

            cur = llm_build_kv(ctx0, lctx, kv_self, gf,
                    model.layers[il].wo, nullptr,
                    Kcur, Vcur, Qcur, KQ_mask, n_tokens, kv_head, n_kv, 1.0f/sqrtf(float(n_embd_head)), cb, il);
        }

        if (il == n_layer - 1) {
            // skip computing output for unused tokens
            struct ggml_tensor * inp_out_ids = build_inp_out_ids();
            n_tokens = n_outputs;
            cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }

        struct ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        // feed-forward network
        cur = llm_build_norm(ctx0, ffn_inp, hparams, NULL, NULL, LLM_NORM, cb, il);
        cb(cur, "ffn_norm", il);

        cur = llm_build_ffn(ctx0, lctx, nullptr, cur,
                model.layers[il].ffn_up,   NULL, NULL,
                model.layers[il].ffn_gate, NULL, NULL,
                model.layers[il].ffn_down, NULL, NULL,
                NULL,
                LLM_FFN_SILU, LLM_FFN_PAR, cb, il);
        cb(cur, "ffn_out", il);

        cur = ggml_add(ctx0, cur, ffn_inp);
        cb(cur, "ffn_out", il);

        cur = lctx.cvec.apply_to(ctx0, cur, il);
        cb(cur, "l_out", il);

        // input for next layer
        inpL = cur;
    }

    cur = inpL;

    cur = llm_build_norm(ctx0, cur, hparams, NULL, NULL, LLM_NORM, cb, -1);
    cb(cur, "result_norm", -1);

    // lm_head
    cur = llm_build_lora_mm(lctx, ctx0, model.output, cur);
    cb(cur, "result_output", -1);

    ggml_build_forward_expand(gf, cur);

    return gf;
}

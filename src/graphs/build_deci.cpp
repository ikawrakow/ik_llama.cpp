#include "../llama-build-context.h"
#include "../llama-model.h"
#include "../llama-context.h"

ggml_cgraph * llm_build_context::build_deci() {
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

    const float kq_scale = hparams.f_attention_scale == 0.0f ? 1.0f/sqrtf(float(n_embd_head)) : hparams.f_attention_scale;
    for (int il = 0; il < n_layer; ++il) {
        struct ggml_tensor * inpSA = inpL;
        const int64_t n_head_kv = hparams.n_head_kv(il);
        const int64_t n_head    = hparams.n_head(il);
        const int64_t n_ff      = hparams.n_ff(il);

        if (n_head == 0) { // attention-free layer of Llama-3_1-Nemotron-51B
            cur = inpL;
        } else {
            // norm
            cur = llm_build_norm(ctx0, inpL, hparams, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, cb, il);
            cb(cur, "attn_norm", il);
        }

        if (n_head > 0 && n_head_kv == 0) { // "linear attention" of Llama-3_1-Nemotron-51B
            cur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wo, cur);
            cb(cur, "wo", il);
        } else if (n_head > 0) {
            // self-attention
            // rope freq factors for llama3; may return nullptr for llama2 and other models
            struct ggml_tensor * rope_factors = build_rope_factors(il);

            auto [Qcur, Kcur, Vcur] = llm_build_mul_mat_qkv(gf, cur, model.layers[il].wq, model.layers[il].bq,
                    model.layers[il].wk, model.layers[il].bk,
                    model.layers[il].wv, model.layers[il].bv,
                    0.f, il);

            Qcur = ggml_rope_ext(
                    ctx0, ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens), inp_pos, rope_factors,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );
            cb(Qcur, "Qcur", il);

            Kcur = ggml_rope_ext(
                    ctx0, ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens), inp_pos, rope_factors,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );
            cb(Kcur, "Kcur", il);

            cur = llm_build_kv(ctx0, lctx, kv_self, gf,
                    model.layers[il].wo, model.layers[il].bo,
                    Kcur, Vcur, Qcur, KQ_mask, n_tokens, kv_head, n_kv, kq_scale, cb, il);
        }

        if (il == n_layer - 1) {
            // skip computing output for unused tokens
            struct ggml_tensor * inp_out_ids = build_inp_out_ids();
            n_tokens = n_outputs;
            cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }

        // FFN-free layer of Llama-3_1-Nemotron-Ultra-253B
        if (n_ff == 0) {
            continue;
        }

        if (hparams.f_residual_scale) {
            cur = ggml_scale(ctx0, cur, hparams.f_residual_scale);
        }

        // modified to support attention-free layer of Llama-3_1-Nemotron-51B
        struct ggml_tensor * ffn_inp = cur;
        if (n_head > 0) {
            ffn_inp = ggml_add(ctx0, cur, inpSA);
            cb(ffn_inp, "ffn_inp", il);
        }

        // feed-forward network
        if (model.layers[il].ffn_gate_inp == nullptr) {

            cur = llm_build_ffn(ctx0, lctx, model.layers[il].ffn_norm, ffn_inp,
                    model.layers[il].ffn_up,   model.layers[il].ffn_up_b,   NULL,
                    model.layers[il].ffn_gate, model.layers[il].ffn_gate_b, NULL,
                    model.layers[il].ffn_down, model.layers[il].ffn_down_b, NULL,
                    NULL,
                    LLM_FFN_SILU, LLM_FFN_PAR, cb, il);
            cb(cur, "ffn_out", il);
        }

        if (hparams.f_residual_scale) {
            cur = ggml_scale(ctx0, cur, hparams.f_residual_scale);
        }

        cur = ggml_add(ctx0, cur, ffn_inp);
        cb(cur, "ffn_out", il);

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

    if (hparams.f_logit_scale) {
        cur = ggml_scale(ctx0, cur, 1.0f / hparams.f_logit_scale);
    }

    cb(cur, "result_output", -1);

    ggml_build_forward_expand(gf, cur);

    return gf;
}


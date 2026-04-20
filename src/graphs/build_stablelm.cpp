#include "../llama-build-context.h"
#include "../llama-model.h"
#include "../llama-context.h"

ggml_cgraph * llm_build_context::build_stablelm() {
    struct ggml_cgraph * gf = ggml_new_graph(ctx0);

    const int64_t n_embd_head = hparams.n_embd_head_v(0);
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k(0));

    struct ggml_tensor * cur;
    struct ggml_tensor * inpL;

    inpL = llm_build_inp_embd(ctx0, lctx, hparams, batch, model.tok_embd, cb);

    // inp_pos - contains the positions
    struct ggml_tensor * inp_pos = build_inp_pos();

    // KQ_mask (mask for 1 head, it will be broadcasted to all heads)
    struct ggml_tensor * KQ_mask = build_inp_KQ_mask();

    for (int il = 0; il < n_layer; ++il) {


        // norm
        cur = llm_build_norm(ctx0, inpL, hparams, model.layers[il].attn_norm, model.layers[il].attn_norm_b, LLM_NORM, cb, il);
        cb(cur, "attn_norm", il);

        struct ggml_tensor * inpSA = cur;

        // self-attention
        {
            auto [Qcur, Kcur, Vcur] = llm_build_mul_mat_qkv(gf, cur, model.layers[il].wq, model.layers[il].bq,
                    model.layers[il].wk, model.layers[il].bk,
                    model.layers[il].wv, model.layers[il].bv, 0.f, il);

            Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
            cb(Qcur, "Qcur", il);
            Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
            cb(Kcur, "Kcur", il);

            if (model.layers[il].attn_q_norm) {
                Qcur = llm_build_norm(ctx0, Qcur, hparams, model.layers[il].attn_q_norm, NULL, LLM_NORM, cb, il);
                cb(Qcur, "Qcur", il);
            }
            if (model.layers[il].attn_k_norm) {
                Kcur = llm_build_norm(ctx0, Kcur, hparams, model.layers[il].attn_k_norm, NULL, LLM_NORM, cb, il);
                cb(Kcur, "Kcur", il);
            }


            Qcur = ggml_rope_ext(
                    ctx0, Qcur, inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );
            cb(Qcur, "Qcur", il);

            Kcur = ggml_rope_ext(
                    ctx0, Kcur, inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );
            cb(Kcur, "Kcur", il);

            cur = llm_build_kv(ctx0, lctx, kv_self, gf,
                    model.layers[il].wo, NULL,
                    Kcur, Vcur, Qcur, KQ_mask, n_tokens, kv_head, n_kv, 1.0f/sqrtf(float(n_embd_head)), cb, il);
        }

        if (il == n_layer - 1) {
            // skip computing output for unused tokens
            struct ggml_tensor * inp_out_ids = build_inp_out_ids();
            cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
            inpL  = ggml_get_rows(ctx0,  inpL, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }

        struct ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpL);
        cb(ffn_inp, "ffn_inp", il);

        // feed-forward network
        {
            if (model.layers[il].ffn_norm) {
                cur = llm_build_norm(ctx0, ffn_inp, hparams, model.layers[il].ffn_norm, model.layers[il].ffn_norm_b, LLM_NORM, cb, il);
                cb(cur, "ffn_norm", il);
            } else {
                // parallel residual
                cur = inpSA;
            }
            cur = llm_build_ffn(ctx0, lctx, nullptr, cur,
                    model.layers[il].ffn_up,   NULL, NULL,
                    model.layers[il].ffn_gate, NULL, NULL,
                    model.layers[il].ffn_down, NULL, NULL,
                    NULL,
                    LLM_FFN_SILU, LLM_FFN_PAR, cb, il);
            cb(cur, "ffn_out", il);
        }

        cur = ggml_add(ctx0, cur, ffn_inp);
        cur = lctx.cvec.apply_to(ctx0, cur, il);
        cb(cur, "l_out", il);

        // input for next layer
        inpL = cur;
    }

    cur = inpL;

    cur = llm_build_norm(ctx0, cur, hparams, model.output_norm, model.output_norm_b, LLM_NORM, cb, -1);
    cb(cur, "result_norm", -1);

    // lm_head
    cur = llm_build_lora_mm(lctx, ctx0, model.output, cur);
    cb(cur, "result_output", -1);

    ggml_build_forward_expand(gf, cur);

    return gf;
}


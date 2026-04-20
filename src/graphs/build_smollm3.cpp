#include "../llama-build-context.h"
#include "../llama-model.h"
#include "../llama-context.h"

ggml_cgraph* llm_build_context::build_smollm3() {
    ggml_cgraph * gf = ggml_new_graph_custom(ctx0, model.max_nodes(n_tokens), false);
    const int64_t n_embd_head = hparams.n_embd_head_v(0);
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k(0));
    // GGML_ASSERT(n_embd_head == hparams.n_rot); this is wrong in case of minimax, head_dim = 128, n_rot = 64

    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = llm_build_inp_embd(ctx0, lctx, hparams, batch, model.tok_embd, cb);

    ggml_tensor * inp_pos = build_inp_pos();


    //auto * inp_attn = build_attn_inp_kv();
    ggml_tensor * inp_out_ids = build_inp_out_ids();
    ggml_tensor * KQ_mask = build_inp_KQ_mask();

    const float kq_scale = hparams.f_attention_scale == 0.0f ? 1.0f/sqrtf(float(n_embd_head)) : hparams.f_attention_scale;

    for (int il = 0; il < n_layer; ++il) {
        ggml_tensor * inpSA = inpL;

        const bool use_rope = (il + 1) % hparams.n_no_rope_layer_step != 0;

        // norm
        cur = llm_build_norm(ctx0, inpL, hparams, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, cb, il);
        cb(cur, "attn_norm", il);

        // self-attention
        {
            auto [Qcur, Kcur, Vcur] = llm_build_mul_mat_qkv(gf, cur,
                    model.layers[il].wqkv, model.layers[il].bqkv,
                    model.layers[il].wqk,  model.layers[il].bqk,
                    model.layers[il].wq,   model.layers[il].bq,
                    model.layers[il].wk,   model.layers[il].bk,
                    model.layers[il].wv,   model.layers[il].bv,
                    model.layers[il].attn_q_norm, model.layers[il].attn_k_norm, 0, il);

            if (use_rope) {
                Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                        ext_factor, attn_factor, beta_fast, beta_slow);
                cb(Qcur, "Qcur", il);

                Kcur = ggml_rope_ext(ctx0, Kcur, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                        ext_factor, attn_factor, beta_fast, beta_slow);
                cb(Kcur, "Kcur", il);
            }

            cur = llm_build_kv(ctx0, lctx, kv_self, gf,
                    model.layers[il].wo, model.layers[il].bo,
                    Kcur, Vcur, Qcur, KQ_mask, n_tokens, kv_head, n_kv, kq_scale, cb, il);
            cb(cur, "attn_out", il);
        }
        if (il == n_layer - 1 && inp_out_ids) {
            cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }
        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        // feed-forward network
        cur = llm_build_ffn(ctx0, lctx, model.layers[il].ffn_norm, ffn_inp,
                model.layers[il].ffn_up,   NULL, NULL,
                model.layers[il].ffn_gate, NULL, NULL,
                model.layers[il].ffn_down, NULL, NULL,
                NULL,
                LLM_FFN_SILU, LLM_FFN_PAR, cb, il);
        cb(cur, "ffn_out", il);

        cur = ggml_add(ctx0, cur, ffn_inp);
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

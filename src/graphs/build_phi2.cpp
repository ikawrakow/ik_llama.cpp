#include "../llama-build-context.h"
#include "../llama-model.h"
#include "../llama-context.h"

ggml_cgraph * llm_build_context::build_phi2() {
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, model.max_nodes(n_tokens), false);

    const int64_t n_embd_head = hparams.n_embd_head_v(0);
    const int64_t n_embd_gqa  = hparams.n_embd_v_gqa();
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k(0));

    struct ggml_tensor * cur;
    struct ggml_tensor * attn_norm_output;
    struct ggml_tensor * ffn_output;
    struct ggml_tensor * inpL;

    inpL = llm_build_inp_embd(ctx0, lctx, hparams, batch, model.tok_embd, cb);

    // inp_pos - contains the positions
    struct ggml_tensor * inp_pos = build_inp_pos();

    // KQ_mask (mask for 1 head, it will be broadcasted to all heads)
    struct ggml_tensor * KQ_mask = build_inp_KQ_mask();

    for (int il = 0; il < n_layer; ++il) {
        attn_norm_output = llm_build_norm(ctx0, inpL, hparams, model.layers[il].attn_norm, model.layers[il].attn_norm_b, LLM_NORM, cb, il);
        cb(attn_norm_output, "attn_norm", il);

        // self-attention
        {
            struct ggml_tensor * Qcur = nullptr;
            struct ggml_tensor * Kcur = nullptr;
            struct ggml_tensor * Vcur = nullptr;

            if (model.layers[il].wqkv) {
                cur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wqkv, attn_norm_output);
                cb(cur, "wqkv", il);

                cur = ggml_add(ctx0, cur, model.layers[il].bqkv);
                cb(cur, "bqkv", il);

                Qcur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd,     n_tokens, cur->nb[1], 0*sizeof(float)*(n_embd)));
                Kcur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd_gqa, n_tokens, cur->nb[1], 1*sizeof(float)*(n_embd)));
                Vcur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd_gqa, n_tokens, cur->nb[1], 1*sizeof(float)*(n_embd + n_embd_gqa)));
            } else {
                Qcur = ggml_add(ctx0, llm_build_lora_mm(lctx, ctx0, model.layers[il].wq, attn_norm_output), model.layers[il].bq);
                Kcur = ggml_add(ctx0, llm_build_lora_mm(lctx, ctx0, model.layers[il].wk, attn_norm_output), model.layers[il].bk);
                Vcur = ggml_add(ctx0, llm_build_lora_mm(lctx, ctx0, model.layers[il].wv, attn_norm_output), model.layers[il].bv);
            }

            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
            Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);

            Qcur = ggml_rope_ext(
                    ctx0, Qcur, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig,
                    freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow
                    );
            cb(Qcur, "Qcur", il);

            // with phi2, we scale the Q to avoid precision issues
            // ref: https://github.com/ml-explore/mlx-examples/blob/08e862336ade809bc37d1035f94b359e7d1a5152/phi2/phi2.py#L64-L66
            Qcur = ggml_scale(ctx0, Qcur, 1.0f/sqrtf(float(n_embd_head)));
            cb(Qcur, "Qcur", il);

            Kcur = ggml_rope_ext(
                    ctx0, Kcur, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig,
                    freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow
                    );
            cb(Kcur, "Kcur", il);

            cur = llm_build_kv(ctx0, lctx, kv_self, gf,
                    model.layers[il].wo, model.layers[il].bo,
                    Kcur, Vcur, Qcur, KQ_mask, n_tokens, kv_head, n_kv, 1.0f, cb, il);
        }

        if (il == n_layer - 1) {
            // skip computing output for unused tokens
            struct ggml_tensor * inp_out_ids = build_inp_out_ids();
            cur              = ggml_get_rows(ctx0,              cur, inp_out_ids);
            inpL             = ggml_get_rows(ctx0,             inpL, inp_out_ids);
            attn_norm_output = ggml_get_rows(ctx0, attn_norm_output, inp_out_ids);
        }

        // FF
        {
            ffn_output = llm_build_ffn(ctx0, lctx, nullptr, attn_norm_output,
                    model.layers[il].ffn_up,   model.layers[il].ffn_up_b,   NULL,
                    NULL,                      NULL,                        NULL,
                    model.layers[il].ffn_down, model.layers[il].ffn_down_b, NULL,
                    NULL,
                    LLM_FFN_GELU, LLM_FFN_SEQ, cb, il);
            cb(ffn_output, "ffn_out", il);
        }

        cur = ggml_add(ctx0, cur, ffn_output);
        cur = ggml_add(ctx0, cur, inpL);
        cur = lctx.cvec.apply_to(ctx0, cur, il);
        cb(cur, "l_out", il);

        // input for next layer
        inpL = cur;
    }

    cur = llm_build_norm(ctx0, inpL, hparams, model.output_norm, model.output_norm_b, LLM_NORM, cb, -1);
    cb(cur, "result_norm", -1);

    cur = llm_build_lora_mm(lctx, ctx0, model.output, cur);
    cb(cur, "result_output_no_bias", -1);

    cur = ggml_add(ctx0, cur, model.output_b);
    cb(cur, "result_output", -1);
    ggml_build_forward_expand(gf, cur);
    return gf;
}


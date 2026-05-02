#include "../llama-build-context.h"
#include "../llama-model.h"
#include "../llama-context.h"

ggml_cgraph * llm_build_context::build_openelm() {
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, model.max_nodes(n_tokens), false);

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
        const int64_t n_head    = hparams.n_head(il);
        const int64_t n_head_kv = hparams.n_head_kv(il);
        const int64_t n_head_qkv = 2*n_head_kv + n_head;

        cur = inpL;
        struct ggml_tensor * residual = cur;

        // norm
        cur = llm_build_norm(ctx0, inpL, hparams, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, cb, il);
        cb(cur, "attn_norm", il);

        // self-attention
        {
            cur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wqkv, cur);
            cb(cur, "wqkv", il);

            cur = ggml_reshape_3d(ctx0, cur, n_embd_head_k, n_head_qkv, n_tokens);

            struct ggml_tensor * Qcur = ggml_cont(ctx0, ggml_view_3d(ctx0, cur, n_embd_head, n_head, n_tokens, cur->nb[1], cur->nb[2], 0));
            cb(Qcur, "Qcur", il);

            struct ggml_tensor * Kcur = ggml_cont(ctx0, ggml_view_3d(ctx0, cur, n_embd_head, n_head_kv, n_tokens, cur->nb[1], cur->nb[2], cur->nb[1]*n_head));
            cb(Kcur, "Kcur", il);

            struct ggml_tensor * Vcur = ggml_cont(ctx0, ggml_view_3d(ctx0, cur, n_embd_head, n_head_kv, n_tokens, cur->nb[1], cur->nb[2], cur->nb[1]*(n_head+n_head_kv)));
            cb(Vcur, "Vcur", il);

            Qcur = llm_build_norm(ctx0, Qcur, hparams, model.layers[il].attn_q_norm, NULL, LLM_NORM_RMS, cb, il);
            cb(Qcur, "Qcur", il);

            Kcur = llm_build_norm(ctx0, Kcur, hparams, model.layers[il].attn_k_norm, NULL, LLM_NORM_RMS, cb, il);
            cb(Kcur, "Kcur", il);

            Qcur = ggml_rope_ext(
                    ctx0, Qcur, inp_pos, NULL, n_rot, rope_type, n_ctx_orig,
                    freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow
                    );
            cb(Qcur, "Qcur", il);

            Kcur = ggml_rope_ext(
                    ctx0, Kcur, inp_pos, NULL, n_rot, rope_type, n_ctx_orig,
                    freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow
                    );
            cb(Kcur, "Kcur", il);

            Vcur = ggml_reshape_2d(ctx0, Vcur, n_embd_head * n_head_kv, n_tokens);
            cb(Qcur, "Vcur", il);

            cur = llm_build_kv(ctx0, lctx, kv_self, gf,
                    model.layers[il].wo, NULL,
                    Kcur, Vcur, Qcur, KQ_mask, n_tokens, kv_head, n_kv, 1.0f/sqrtf(float(n_embd_head)), cb, il);
        }

        if (il == n_layer - 1) {
            // skip computing output for unused tokens
            struct ggml_tensor * inp_out_ids = build_inp_out_ids();
            residual = ggml_get_rows(ctx0, residual, inp_out_ids);
            cur = ggml_get_rows(ctx0, cur, inp_out_ids);
        }

        struct ggml_tensor * ffn_inp = ggml_add(ctx0, residual, cur);
        cb(ffn_inp, "ffn_inp", il);

        // feed-forward network
        {
            cur = llm_build_ffn(ctx0, lctx, model.layers[il].ffn_norm, ffn_inp,
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

        inpL = cur;
    }

    cur = inpL;

    // norm
    cur = llm_build_norm(ctx0, cur, hparams, model.output_norm, NULL, LLM_NORM_RMS, cb, -1);
    cb(cur, "result_norm", -1);

    cur = llm_build_lora_mm(lctx, ctx0, model.output, cur);
    cb(cur, "result_output", -1);

    ggml_build_forward_expand(gf, cur);

    return gf;
}

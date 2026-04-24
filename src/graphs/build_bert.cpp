#include "../llama-build-context.h"
#include "../llama-model.h"
#include "../llama-context.h"

ggml_cgraph * llm_build_context::build_bert() {
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, model.max_nodes(n_tokens), false);

    const int64_t n_embd_head = hparams.n_embd_head_v(0);
    const int64_t n_embd_gqa  = hparams.n_embd_v_gqa();

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k(0));

    struct ggml_tensor * cur;
    struct ggml_tensor * inpL;
    struct ggml_tensor * inp_pos = nullptr;

    if (model.arch != LLM_ARCH_JINA_BERT_V2) {
        inp_pos = build_inp_pos();
    }

    // construct input embeddings (token, type, position)
    inpL = llm_build_inp_embd(ctx0, lctx, hparams, batch, model.tok_embd, cb);

    // token types are hardcoded to zero ("Sentence A")
    struct ggml_tensor * type_row0 = ggml_view_1d(ctx0, model.type_embd, n_embd, 0);
    inpL = ggml_add(ctx0, inpL, type_row0);
    if (model.arch == LLM_ARCH_BERT) {
        inpL = ggml_add(ctx0, ggml_get_rows(ctx0, model.pos_embd, inp_pos), inpL);
    }
    cb(inpL, "inp_embd", -1);

    // embed layer norm
    inpL = llm_build_norm(ctx0, inpL, hparams, model.tok_norm, model.tok_norm_b, LLM_NORM, cb, -1);
    cb(inpL, "inp_norm", -1);

    // KQ_mask (mask for 1 head, it will be broadcasted to all heads)
    struct ggml_tensor * KQ_mask = build_inp_KQ_mask(false);

    // iterate layers
    for (int il = 0; il < n_layer; ++il) {
        struct ggml_tensor * cur = inpL;

        struct ggml_tensor * Qcur;
        struct ggml_tensor * Kcur;
        struct ggml_tensor * Vcur;

        // self-attention
        if (model.arch == LLM_ARCH_BERT || model.arch == LLM_ARCH_JINA_BERT_V2) {
            Qcur = ggml_add(ctx0, llm_build_lora_mm(lctx, ctx0, model.layers[il].wq, cur), model.layers[il].bq);
            cb(Qcur, "Qcur", il);

            if (model.layers[il].attn_q_norm) {
                Qcur = llm_build_norm(ctx0, Qcur, hparams, model.layers[il].attn_q_norm, model.layers[il].attn_q_norm_b, LLM_NORM, cb, il);
            }

            Kcur = ggml_add(ctx0, llm_build_lora_mm(lctx, ctx0, model.layers[il].wk, cur), model.layers[il].bk);
            cb(Kcur, "Kcur", il);

            if (model.layers[il].attn_k_norm) {
                Kcur = llm_build_norm(ctx0, Kcur, hparams, model.layers[il].attn_k_norm, model.layers[il].attn_k_norm_b, LLM_NORM, cb, il);
            }
            Vcur = ggml_add(ctx0, llm_build_lora_mm(lctx, ctx0, model.layers[il].wv, cur), model.layers[il].bv);
            cb(Vcur, "Vcur", il);

            Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
            Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
        } else {
            // compute Q and K and RoPE them
            cur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wqkv, cur);
            cb(cur, "wqkv", il);

            Qcur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd,     n_tokens, cur->nb[1], 0*sizeof(float)*(n_embd)));
            Kcur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd_gqa, n_tokens, cur->nb[1], 1*sizeof(float)*(n_embd)));
            Vcur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd_gqa, n_tokens, cur->nb[1], 1*sizeof(float)*(n_embd + n_embd_gqa)));

            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            Qcur = ggml_rope_ext(
                    ctx0, ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens), inp_pos, nullptr,
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
        }

        struct ggml_tensor * q =                 ggml_permute(ctx0, Qcur, 0, 2, 1, 3);
        struct ggml_tensor * k = ggml_cont(ctx0, ggml_permute(ctx0, Kcur, 0, 2, 1, 3));

        struct ggml_tensor * kq = ggml_mul_mat(ctx0, k, q);
        cb(kq, "kq", il);

        kq = ggml_soft_max_ext(ctx0, kq, KQ_mask, 1.0f/sqrtf(float(n_embd_head)), hparams.f_max_alibi_bias);
        cb(kq, "kq_soft_max_ext", il);

        struct ggml_tensor * v = ggml_cont(ctx0, ggml_transpose(ctx0, ggml_reshape_2d(ctx0, Vcur, n_embd_gqa, n_tokens)));
        cb(v, "v", il);

        struct ggml_tensor * kqv = ggml_mul_mat(ctx0, ggml_reshape_3d(ctx0, v, n_tokens, n_embd_head, n_head_kv), kq);
        cb(kqv, "kqv", il);

        struct ggml_tensor * kqv_merged = ggml_permute(ctx0, kqv, 0, 2, 1, 3);
        cb(kqv_merged, "kqv_merged", il);

        cur = ggml_cont_2d(ctx0, kqv_merged, n_embd_gqa, n_tokens);
        cb(cur, "kqv_merged_cont", il);

        ggml_build_forward_expand(gf, cur);

        cur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wo, cur);
        if (model.layers[il].bo) {
            cb(cur, "kqv_wo", il);
        }

        if (model.layers[il].bo) {
            cur = ggml_add(ctx0, cur, model.layers[il].bo);
        }
        cb(cur, "kqv_out", il);

        if (il == n_layer - 1 && pooling_type == LLAMA_POOLING_TYPE_NONE) {
            // skip computing output for unused tokens
            struct ggml_tensor * inp_out_ids = build_inp_out_ids();
            cur  = ggml_get_rows(ctx0,  cur, inp_out_ids);
            inpL = ggml_get_rows(ctx0, inpL, inp_out_ids);
        }

        // re-add the layer input
        cur = ggml_add(ctx0, cur, inpL);

        // attention layer norm
        cur = llm_build_norm(ctx0, cur, hparams, model.layers[il].attn_out_norm, model.layers[il].attn_out_norm_b, LLM_NORM, cb, il);

        if (model.layers[il].attn_norm_2 != nullptr) {
            cur = ggml_add(ctx0, cur, inpL); // re-add the layer input
            cur = llm_build_norm(ctx0, cur, hparams, model.layers[il].attn_norm_2, model.layers[il].attn_norm_2_b, LLM_NORM, cb, il);
        }

        struct ggml_tensor * ffn_inp = cur;
        cb(ffn_inp, "ffn_inp", il);

        // feed-forward network
        if (model.arch == LLM_ARCH_BERT) {
            cur = llm_build_ffn(ctx0, lctx, nullptr, cur,
                    model.layers[il].ffn_up,   model.layers[il].ffn_up_b,   NULL,
                    NULL,                      NULL,                        NULL,
                    model.layers[il].ffn_down, model.layers[il].ffn_down_b, NULL,
                    NULL,
                    LLM_FFN_GELU, LLM_FFN_SEQ, cb, il);
        } else if (model.arch == LLM_ARCH_JINA_BERT_V2) {
            cur = llm_build_ffn(ctx0, lctx, nullptr, cur,
                    model.layers[il].ffn_up,   NULL,                        NULL,
                    model.layers[il].ffn_gate, NULL,                        NULL,
                    model.layers[il].ffn_down, model.layers[il].ffn_down_b, NULL,
                    NULL,
                    LLM_FFN_GELU, LLM_FFN_PAR, cb, il);
        } else {
            cur = llm_build_ffn(ctx0, lctx, nullptr, cur,
                    model.layers[il].ffn_up,   NULL, NULL,
                    model.layers[il].ffn_gate, NULL, NULL,
                    model.layers[il].ffn_down, NULL, NULL,
                    NULL,
                    LLM_FFN_SILU, LLM_FFN_PAR, cb, il);
        }
        cb(cur, "ffn_out", il);

        // attentions bypass the intermediate layer
        cur = ggml_add(ctx0, cur, ffn_inp);

        // output layer norm
        cur = llm_build_norm(ctx0, cur, hparams, model.layers[il].layer_out_norm, model.layers[il].layer_out_norm_b, LLM_NORM, cb, il);

        // input for next layer
        inpL = cur;
    }

    // final output
    cur = inpL;
    cb(cur, "result_embd", -1);

    ggml_build_forward_expand(gf, cur);

    return gf;
}


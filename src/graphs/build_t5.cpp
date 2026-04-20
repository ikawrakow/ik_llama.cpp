#include "../llama-build-context.h"
#include "../llama-model.h"
#include "../llama-context.h"

ggml_cgraph * llm_build_context::build_t5_encoder() {
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, model.max_nodes(n_tokens), false);

    // mutable variable, needed during the last layer of the computation to skip unused tokens
    int32_t n_tokens = this->n_tokens;

    const int64_t n_embd_head = hparams.n_embd_head_v(0);
    const int64_t n_embd_gqa  = hparams.n_embd_v_gqa();
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k(0));

    struct ggml_tensor * cur;
    struct ggml_tensor * inpL;

    inpL = llm_build_inp_embd(ctx0, lctx, hparams, batch, model.tok_embd, cb);

    GGML_ASSERT(lctx.is_encoding);
    struct ggml_tensor * pos_bucket_enc = llm_build_pos_bucket(false);

    // KQ_mask (mask for 1 head, it will be broadcasted to all heads)
    struct ggml_tensor * KQ_mask_enc = build_inp_KQ_mask(false);

    for (int il = 0; il < n_layer; ++il) {
        struct ggml_tensor * inpSA = inpL;

        // norm
        cur = llm_build_norm(ctx0, inpL, hparams, model.layers[il].attn_norm_enc, NULL, LLM_NORM_RMS, cb, il);
        cb(cur, "attn_norm", il);

        // self-attention
        {
            auto [Qcur, Kcur, Vcur] = llm_build_mul_mat_qkv(gf, cur, model.layers[il].wq, nullptr,
                    model.layers[il].wk, nullptr,
                    model.layers[il].wv, nullptr, 0, il);

            Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens);
            Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);

            struct ggml_tensor * q =                 ggml_permute(ctx0, Qcur, 0, 2, 1, 3);
            struct ggml_tensor * k = ggml_cont(ctx0, ggml_permute(ctx0, Kcur, 0, 2, 1, 3));

            struct ggml_tensor * kq = ggml_mul_mat(ctx0, k, q);
            cb(kq, "kq", il);

            struct ggml_tensor * attn_rel_b = model.layers[il].attn_rel_b_enc ? model.layers[il].attn_rel_b_enc : model.layers[0].attn_rel_b_enc;
            struct ggml_tensor * pos_bias = llm_build_pos_bias(pos_bucket_enc, attn_rel_b);
            struct ggml_tensor * kq_b = ggml_add(ctx0, kq, pos_bias);
            cb(kq_b, "kq_b", il);

            kq = ggml_soft_max_ext(ctx0, kq_b, KQ_mask_enc, 1.0f, hparams.f_max_alibi_bias);
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

            cur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wo_enc, cur);
            cb(cur, "kqv_out", il);
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
        {
            // T5 uses relu, flan-T5 uses gelu-gated
            cur = llm_build_ffn(ctx0, lctx, model.layers[il].ffn_norm_enc, ffn_inp,
                    model.layers[il].ffn_up_enc,   NULL, NULL,
                    model.layers[il].ffn_gate_enc, NULL, NULL,
                    model.layers[il].ffn_down_enc, NULL, NULL,
                    NULL,
                    model.layers[il].ffn_gate_enc ? LLM_FFN_GELU : LLM_FFN_RELU,
                    model.layers[il].ffn_gate_enc ? LLM_FFN_PAR  : LLM_FFN_SEQ,
                    cb, il);
            cb(cur, "ffn_out", il);
        }

        cur = ggml_add(ctx0, cur, ffn_inp);
        cb(cur, "ffn_out", il);

        ggml_tensor * layer_dir = lctx.cvec.tensor_for(il);
        if (layer_dir != nullptr) {
            cur = ggml_add(ctx0, cur, layer_dir);
        }
        cb(cur, "l_out", il);

        // input for next layer
        inpL = cur;
    }

    cur = inpL;
    cb(cur, "result_embd", -1);

    cur = llm_build_norm(ctx0, cur, hparams, model.output_norm_enc, NULL, LLM_NORM_RMS, cb, -1);
    cb(cur, "result_norm", -1);

    ggml_build_forward_expand(gf, cur);

    return gf;
}

ggml_cgraph * llm_build_context::build_t5_decoder() {
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, model.max_nodes(n_tokens), false);

    // mutable variable, needed during the last layer of the computation to skip unused tokens
    int32_t n_tokens = this->n_tokens;

    const int64_t n_embd_head = hparams.n_embd_head_v(0);
    const int64_t n_embd_gqa  = hparams.n_embd_v_gqa();
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k(0));

    struct ggml_tensor * cur;
    struct ggml_tensor * inpL;

    inpL = llm_build_inp_embd(ctx0, lctx, hparams, batch, model.tok_embd, cb);

    GGML_ASSERT(!lctx.is_encoding);
    GGML_ASSERT(n_outputs_enc > 0 && "call llama_encode() first");

    struct ggml_tensor * embd_enc       = llm_build_inp_embd_enc();
    struct ggml_tensor * pos_bucket_dec = llm_build_pos_bucket(true);

    struct ggml_tensor * KQ_mask_dec   = build_inp_KQ_mask();
    struct ggml_tensor * KQ_mask_cross = llm_build_inp_KQ_mask_cross();

    for (int il = 0; il < n_layer; ++il) {
        struct ggml_tensor * inpSA = inpL;

        // norm
        cur = llm_build_norm(ctx0, inpL, hparams, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, cb, il);
        cb(cur, "attn_norm", il);

        // self-attention
        {
            auto [Qcur, Kcur, Vcur] = llm_build_mul_mat_qkv(gf, cur, model.layers[il].wq, nullptr,
                    model.layers[il].wk, nullptr,
                    model.layers[il].wv, nullptr, 0, il);

            llm_build_kv_store(lctx, ctx0, hparams, cparams, kv_self, gf, Kcur, Vcur, n_tokens, kv_head, cb, il);

            struct ggml_tensor * k =
                ggml_view_3d(ctx0, kv_self.k_l[il],
                        n_embd_head_k, n_kv, n_head_kv,
                        ggml_row_size(kv_self.k_l[il]->type, n_embd_k_gqa),
                        ggml_row_size(kv_self.k_l[il]->type, n_embd_head_k),
                        0);
            cb(k, "k", il);

            struct ggml_tensor * v =
                ggml_view_3d(ctx0, kv_self.v_l[il],
                        n_kv, n_embd_head_v, n_head_kv,
                        ggml_element_size(kv_self.v_l[il])*n_ctx,
                        ggml_element_size(kv_self.v_l[il])*n_ctx*n_embd_head_v,
                        0);
            cb(v, "v", il);

            Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens);

            struct ggml_tensor * q = ggml_permute(ctx0, Qcur, 0, 2, 1, 3);

            struct ggml_tensor * kq = ggml_mul_mat(ctx0, k, q);
            cb(kq, "kq", il);

            struct ggml_tensor * attn_rel_b = model.layers[il].attn_rel_b ? model.layers[il].attn_rel_b : model.layers[0].attn_rel_b;
            struct ggml_tensor * pos_bias = llm_build_pos_bias(pos_bucket_dec, attn_rel_b);
            struct ggml_tensor * kq_b = ggml_add(ctx0, kq, pos_bias);
            cb(kq_b, "kq_b", il);

            kq = ggml_soft_max_ext(ctx0, kq_b, KQ_mask_dec, 1.0f, hparams.f_max_alibi_bias);
            cb(kq, "kq_soft_max_ext", il);

            struct ggml_tensor * kqv = ggml_mul_mat(ctx0, v, kq);
            cb(kqv, "kqv", il);

            struct ggml_tensor * kqv_merged = ggml_permute(ctx0, kqv, 0, 2, 1, 3);
            cb(kqv_merged, "kqv_merged", il);

            cur = ggml_cont_2d(ctx0, kqv_merged, n_embd_gqa, n_tokens);
            cb(cur, "kqv_merged_cont", il);

            ggml_build_forward_expand(gf, cur);

            cur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wo, cur);
            cb(cur, "kqv_out", il);
        }

        cur = ggml_add(ctx0, cur, inpSA);
        cb(cur, "cross_inp", il);

        struct ggml_tensor * inpCA = cur;

        // norm
        cur = llm_build_norm(ctx0, cur, hparams, model.layers[il].attn_norm_cross, NULL, LLM_NORM_RMS, cb, il);
        cb(cur, "attn_norm_cross", il);

        // cross-attention
        {
            struct ggml_tensor * Qcur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wq_cross, cur);
            cb(Qcur, "Qcur", il);

            struct ggml_tensor * Kcur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wk_cross, embd_enc);
            cb(Kcur, "Kcur", il);

            struct ggml_tensor * Vcur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wv_cross, embd_enc);
            cb(Vcur, "Vcur", il);

            Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
            Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_outputs_enc);

            struct ggml_tensor * q =                 ggml_permute(ctx0, Qcur, 0, 2, 1, 3);
            struct ggml_tensor * k = ggml_cont(ctx0, ggml_permute(ctx0, Kcur, 0, 2, 1, 3));

            struct ggml_tensor * kq = ggml_mul_mat(ctx0, k, q);
            cb(kq, "kq", il);

            kq = ggml_soft_max_ext(ctx0, kq, KQ_mask_cross, 1.0f, hparams.f_max_alibi_bias);
            cb(kq, "kq_soft_max_ext", il);

            struct ggml_tensor * v = ggml_cont(ctx0, ggml_transpose(ctx0, ggml_reshape_2d(ctx0, Vcur, n_embd_gqa, n_outputs_enc)));
            cb(v, "v", il);

            struct ggml_tensor * kqv = ggml_mul_mat(ctx0, ggml_reshape_3d(ctx0, v, n_outputs_enc, n_embd_head, n_head_kv), kq);
            cb(kqv, "kqv", il);

            struct ggml_tensor * kqv_merged = ggml_permute(ctx0, kqv, 0, 2, 1, 3);
            cb(kqv_merged, "kqv_merged", il);

            cur = ggml_cont_2d(ctx0, kqv_merged, n_embd_gqa, n_tokens);
            cb(cur, "kqv_merged_cont", il);

            ggml_build_forward_expand(gf, cur);

            cur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wo_cross, cur);
            cb(cur, "kqv_out", il);
        }

        if (il == n_layer - 1) {
            // skip computing output for unused tokens
            struct ggml_tensor * inp_out_ids = build_inp_out_ids();
            n_tokens = n_outputs;
            cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
            inpCA = ggml_get_rows(ctx0, inpCA, inp_out_ids);
        }

        struct ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpCA);
        cb(ffn_inp, "ffn_inp", il);

        // feed-forward network
        {
            // T5 uses relu, flan-T5 uses gelu-gated
            cur = llm_build_ffn(ctx0, lctx, model.layers[il].ffn_norm, ffn_inp,
                    model.layers[il].ffn_up,   NULL, NULL,
                    model.layers[il].ffn_gate, NULL, NULL,
                    model.layers[il].ffn_down, NULL, NULL,
                    NULL,
                    model.layers[il].ffn_gate_enc ? LLM_FFN_GELU : LLM_FFN_RELU,
                    model.layers[il].ffn_gate_enc ? LLM_FFN_PAR : LLM_FFN_SEQ,
                    cb, il);
            cb(cur, "ffn_out", il);
        }

        cur = ggml_add(ctx0, cur, ffn_inp);
        cb(cur, "ffn_out", il);

        ggml_tensor * layer_dir = lctx.cvec.tensor_for(il);
        if (layer_dir != nullptr) {
            cur = ggml_add(ctx0, cur, layer_dir);
        }
        cb(cur, "l_out", il);

        // input for next layer
        inpL = cur;
    }

    cur = inpL;
    cb(cur, "result_embd", -1);

    cur = llm_build_norm(ctx0, cur, hparams, model.output_norm, NULL, LLM_NORM_RMS, cb, -1);
    cb(cur, "result_norm", -1);

    // lm_head
    cur = llm_build_lora_mm(lctx, ctx0, model.output, cur);
    cb(cur, "result_output", -1);

    ggml_build_forward_expand(gf, cur);

    return gf;
}

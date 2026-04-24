#include "../llama-build-context.h"
#include "../llama-model.h"
#include "../llama-context.h"

ggml_cgraph * llm_build_context::build_bitnet() {
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
        struct ggml_tensor * inpSA = inpL;

        cur = llm_build_norm(ctx0, inpL, hparams, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, cb, il);
        cb(cur, "attn_norm", il);

        // self-attention
        {
            // compute Q and K and RoPE them
            struct ggml_tensor * Qcur = ggml_mul_mat(ctx0, model.layers[il].wq, cur);
            float q_scale; std::memcpy(&q_scale, model.layers[il].wq->op_params, sizeof(float));
            // Note: we could save this scale operation by applying the Q scale on the K * Q product further down
            // (which also uses a scale). This works on the CPU and Metal backends, but produces NaNs on CUDA.
            if (fabsf(q_scale-1) > 1e-4f) Qcur = ggml_scale(ctx0, Qcur, q_scale);
            cb(Qcur, "Qcur", il);
            if (model.layers[il].bq) {
                Qcur = ggml_add(ctx0, Qcur, model.layers[il].bq);
                cb(Qcur, "Qcur", il);
            }

            // B1.K
            struct ggml_tensor * Kcur = ggml_mul_mat(ctx0, model.layers[il].wk, cur);
            float k_scale; std::memcpy(&k_scale, model.layers[il].wk->op_params, sizeof(float));
            if (fabsf(k_scale-1) > 1e-4f) Kcur = ggml_scale(ctx0, Kcur, k_scale);
            cb(Kcur, "Kcur", il);
            if (model.layers[il].bk) {
                Kcur = ggml_add(ctx0, Kcur, model.layers[il].bk);
                cb(Kcur, "Kcur", il);
            }

            // B1.V
            struct ggml_tensor * Vcur = ggml_mul_mat(ctx0, model.layers[il].wv, cur);
            float v_scale; std::memcpy(&v_scale, model.layers[il].wv->op_params, sizeof(float));
            if (model.layers[il].bv) {
                if (fabsf(v_scale-1) > 1e-4f) Vcur = ggml_scale(ctx0, Vcur, v_scale);
                v_scale = 1;
                Vcur = ggml_add(ctx0, Vcur, model.layers[il].bv);
            }
            cb(Vcur, "Vcur", il);

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

            ggml_tensor * cur_attn = llm_build_kv(ctx0, lctx, kv_self, gf,
                    // we cannot pass model.layers[il].wo and model.layers[il].bo because we need to do rms_norm first
                    nullptr, nullptr,
                    Kcur, Vcur, Qcur, KQ_mask, n_tokens, kv_head, n_kv, 1.0f/sqrtf(float(n_embd_head)), cb, il);

            cur_attn = llm_build_norm(ctx0, cur_attn, hparams, model.layers[il].attn_sub_norm, NULL, LLM_NORM_RMS, cb, il, 1/(v_scale*v_scale));
            cb(cur_attn, "attn_sub_norm", il);

            ggml_build_forward_expand(gf, cur_attn);

            cur = ggml_mul_mat(ctx0, model.layers[il].wo, cur_attn);
            float wo_scale; std::memcpy(&wo_scale, model.layers[il].wo->op_params, sizeof(float));
            if (fabsf(wo_scale-1) > 1e-4f) cur = ggml_scale(ctx0, cur, wo_scale);

            cb(cur, "kqv_out", il);
        }

        if (il == n_layer - 1) {
            // skip computing output for unused tokens
            struct ggml_tensor * inp_out_ids = build_inp_out_ids();
            cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }

        struct ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        // feed-forward forward
        if (model.layers[il].ffn_gate_inp == nullptr) {
            cur = llm_build_norm(ctx0, ffn_inp, hparams, model.layers[il].ffn_norm, NULL, LLM_NORM_RMS, cb, il);
            cb(cur, "ffn_norm", il);

            struct ggml_tensor *tmp = ggml_mul_mat(ctx0, model.layers[il].ffn_up, cur);
            float ffn_up_scale; std::memcpy(&ffn_up_scale, model.layers[il].ffn_up->op_params, sizeof(float));

            cb(tmp, "ffn_up", il);

            cur = ggml_mul_mat(ctx0, model.layers[il].ffn_gate, cur);
            float ffn_gate_scale; std::memcpy(&ffn_gate_scale, model.layers[il].ffn_gate->op_params, sizeof(float));
            if (fabsf(ffn_gate_scale-1) > 1e-4f) cur = ggml_scale(ctx0, cur, ffn_gate_scale);

            cb(cur, "ffn_gate", il);

            cur = ggml_fused_mul_unary(ctx0, cur, tmp, GGML_UNARY_OP_SILU);
            cb(cur, "ffn_gate_par", il);

            cur = llm_build_norm(ctx0, cur, hparams, model.layers[il].ffn_sub_norm, NULL, LLM_NORM_RMS, cb, il, 1/(ffn_up_scale*ffn_up_scale));
            cb(cur, "ffn_sub_norm", il);

            cur = ggml_mul_mat(ctx0, model.layers[il].ffn_down, cur);
            float ffn_down_scale; std::memcpy(&ffn_down_scale, model.layers[il].ffn_down->op_params, sizeof(float));
            if (fabsf(ffn_down_scale-1) > 1e-4f) cur = ggml_scale(ctx0, cur, ffn_down_scale);
            cb(cur, "ffn_down", il);
        }
        cur = ggml_add(ctx0, cur, ffn_inp);
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

ggml_cgraph * llm_build_context::build_bitnet_158() {
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
        cur = llm_build_norm(ctx0, inpL, hparams, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, cb, il);
        cb(cur, "attn_norm", il);

        // self-attention
        {
            // rope freq factors for llama3; may return nullptr for llama2 and other models
            struct ggml_tensor * rope_factors = build_rope_factors(il);

            auto [Qcur, Kcur, Vcur] = llm_build_mul_mat_qkv(gf, cur, model.layers[il].wq, nullptr,
                    model.layers[il].wk, nullptr,
                    model.layers[il].wv, nullptr, 0, il);

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
                    NULL, NULL,
                    Kcur, Vcur, Qcur, KQ_mask, n_tokens, kv_head, n_kv, 1.0f/sqrtf(float(n_embd_head)), cb, il);

            cur = llm_build_norm(ctx0, cur, hparams, model.layers[il].attn_sub_norm, NULL, LLM_NORM_RMS, cb, il);
            cb(cur, "attn_sub_norm", il);

            cur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wo, cur);
            if (model.layers[il].wo_scale) {
                cur = ggml_mul(ctx0, cur, model.layers[il].wo_scale);
            }
            if (model.layers[il].bo) {
                cur = ggml_add(ctx0, cur, model.layers[il].bo);
            }
            cb(cur, "attn_o_out", il);
        }

        if (il == n_layer - 1) {
            // skip computing output for unused tokens
            struct ggml_tensor * inp_out_ids = build_inp_out_ids();
            // n_tokens = n_outputs;
            cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }

        struct ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        cur = llm_build_ffn(ctx0, lctx, model.layers[il].ffn_norm, ffn_inp,
                model.layers[il].ffn_up,   NULL, model.layers[il].ffn_up_scale,
                model.layers[il].ffn_gate, NULL, model.layers[il].ffn_gate_scale,
                NULL, NULL, NULL,
                NULL,
                LLM_FFN_RELU_SQR, LLM_FFN_PAR, cb, il);
        cb(cur, "ffn_out", il);

        cur = llm_build_norm(ctx0, cur, hparams, model.layers[il].ffn_sub_norm, NULL, LLM_NORM_RMS, cb, il);
        cb(cur, "ffn_sub_norm", il);

        cur = llm_build_lora_mm(lctx, ctx0, model.layers[il].ffn_down, cur);
        if (model.layers[il].ffn_down_scale) {
            cur = ggml_mul(ctx0, cur, model.layers[il].ffn_down_scale);
        }
        cb(cur, "ffn_down", il);

        cur = ggml_add(ctx0, cur, ffn_inp);
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

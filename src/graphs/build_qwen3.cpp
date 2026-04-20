#include "../llama-build-context.h"
#include "../llama-model.h"
#include "../llama-context.h"

ggml_cgraph * llm_build_context::build_qwen3() {
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, model.max_nodes(n_tokens), false);

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

    ggml_tensor * rope_cache = nullptr;
    if (model.split_mode != LLAMA_SPLIT_MODE_GRAPH && cparams.rope_cache &&
            (rope_type == LLAMA_ROPE_TYPE_NEOX || rope_type == LLAMA_ROPE_TYPE_NORM)) {
        rope_cache = ggml_rope_cache(ctx0, inp_pos, nullptr, n_embd_head, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                ext_factor, attn_factor, beta_fast, beta_slow);
    }

    auto inp_out_ids = n_tokens > 1 ? build_inp_out_ids() : nullptr;

    for (int il = 0; il < n_layer; ++il) {
        struct ggml_tensor * inpSA = inpL;

        if (!rope_cache) {
            cur = build_std_attention(gf, model.layers[il].attn_norm, inpL,
                    inp_pos, il == n_layer-1 && n_tokens > 1 ? inp_out_ids : nullptr, nullptr,
                    KQ_mask, nullptr, nullptr, 1.0f/sqrtf(float(n_embd_head)), 0.0f, 0, il, true, false, true);
        } else {

            // norm
            cur = llm_build_norm(ctx0, inpL, hparams, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, cb, il);
            cb(cur, "attn_norm", il);

            // self-attention
            {
                auto [Qcur, Kcur, Vcur] = llm_build_mul_mat_qkv(gf, cur,
                        model.layers[il].wqkv, nullptr,
                        model.layers[il].wqk, nullptr,
                        model.layers[il].wq, nullptr,
                        model.layers[il].wk, nullptr,
                        model.layers[il].wv, nullptr,
                        model.layers[il].attn_q_norm, model.layers[il].attn_k_norm, 0, il);

                if (rope_cache) {
                    Qcur = ggml_rope_fast(ctx0, Qcur, rope_cache);
                    Kcur = ggml_rope_fast(ctx0, Kcur, rope_cache);
                } else {
                    Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                            ext_factor, attn_factor, beta_fast, beta_slow);
                    Kcur = ggml_rope_ext(ctx0, Kcur, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                            ext_factor, attn_factor, beta_fast, beta_slow);
                }
                cb(Qcur, "Qcur", il);
                cb(Kcur, "Kcur", il);

                cur = llm_build_kv(ctx0, lctx, kv_self, gf,
                        model.layers[il].wo, model.layers[il].bo,
                        Kcur, Vcur, Qcur, KQ_mask, n_tokens, kv_head, n_kv, 1.0f/sqrtf(float(n_embd_head)), cb, il);

                cur = ggml_add(ctx0, cur, inpSA);
                cb(cur, "attn_with_inp", il);
            }
        }

        if (il == n_layer - 1 && rope_cache && inp_out_ids) {
            // skip computing output for unused tokens
            struct ggml_tensor * inp_out_ids = build_inp_out_ids();
            cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
        }

        // feed-forward network
        cur = llm_build_ffn(ctx0, lctx, model.layers[il].ffn_norm, cur,
                model.layers[il].ffn_up,   NULL, NULL,
                model.layers[il].ffn_gate, NULL, NULL,
                model.layers[il].ffn_down, NULL, NULL,
                NULL,
                LLM_FFN_SILU, LLM_FFN_PAR, cb, il, gf, true);
        cb(cur, "ffn_out", il);

        cur = lctx.cvec.apply_to(ctx0, cur, il);
        cb(cur, "l_out", il);

        // input for next layer
        inpL = cur;
    }

    cur = build_output(lctx, ctx0, inpL, model.output, model.output_norm, cb);
    cb(cur, "result_output", -1);

    ggml_build_forward_expand(gf, cur);

    return gf;
}

ggml_cgraph * llm_build_context::build_qwen3moe() {
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, model.max_nodes(n_tokens), false);

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

    ggml_tensor * inp_out_ids = nullptr; //build_inp_out_ids();

    for (int il = 0; il < n_layer; ++il) {

        if (il == n_layer - 1 && n_tokens > 1) {
            inp_out_ids = build_inp_out_ids();
        }

        cur = build_std_attention(gf, model.layers[il].attn_norm, inpL, inp_pos, inp_out_ids, nullptr,
                KQ_mask, nullptr, nullptr, 1.0f/sqrtf(float(n_embd_head)), 0.0f, 0, il, true, false, true);

        auto ffn_inp = cur;

        cur = llm_build_std_moe_ffn(ctx0, lctx, model.layers[il].ffn_norm, ffn_inp,
                model.layers[il].ffn_gate_inp,  nullptr,
                model.layers[il].ffn_up_exps,   nullptr,
                model.layers[il].ffn_gate_exps, nullptr,
                model.layers[il].ffn_down_exps, nullptr,
                model.layers[il].ffn_exp_probs_b,
                nullptr,  nullptr, // we don't have shared expert biases?
                nullptr,  nullptr,
                nullptr,  nullptr,
                n_expert, n_expert_used,
                LLM_FFN_SILU, true, false, 0.0f,
                LLM_EXPERT_GATING_FUNC_SOFTMAX,
                LLM_FFN_SILU, cb, il, gf, true,
                model.layers[il].ffn_up_gate_exps);

        cur = lctx.cvec.apply_to(ctx0, cur, il);
        cb(cur, "l_out", il);

        // input for next layer
        inpL = cur;
    }

    cur = inpL;

    cur = build_output(lctx, ctx0, cur, model.output, model.output_norm, cb);
    cb(cur, "result_output", -1);

    ggml_build_forward_expand(gf, cur);

    return gf;
}

ggml_cgraph * llm_build_context::build_qwen3vl() {
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, model.max_nodes(n_tokens), false);

    const int64_t n_embd_full = hparams.n_embd; // main embd + deepstack embds
    const size_t n_deepstack_layers = hparams.n_deepstack_layers;
    const int64_t n_embd = n_embd_full / (n_deepstack_layers + 1);
    const int64_t n_embd_head = hparams.n_embd_head_v(0);

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k(0));
    GGML_ASSERT(n_embd_head == hparams.n_rot);

    struct ggml_tensor * cur;
    struct ggml_tensor * inpL;

    inpL = llm_build_inp_embd(ctx0, lctx, hparams, batch, model.tok_embd, cb);

    int sections[4];
    std::copy(std::begin(hparams.rope_sections), std::begin(hparams.rope_sections) + 4, sections);

    std::vector<struct ggml_tensor *> deepstack_features;

    if (batch.embd) {
        deepstack_features.resize(n_deepstack_layers, nullptr);
        // Image input: split main embd and deepstack embds
        struct ggml_tensor * inpL_main = ggml_view_2d(ctx0, inpL, n_embd, n_tokens, inpL->nb[1], 0);
        for (size_t i = 0; i < n_deepstack_layers; i++) {
            deepstack_features[i] = ggml_view_2d(ctx0, inpL, n_embd, n_tokens, inpL->nb[1], (i + 1) * n_embd * sizeof(float));
        }
        inpL = inpL_main;
    }

    // inp_pos - contains the positions
    struct ggml_tensor * inp_pos = build_inp_pos();

    // KQ_mask (mask for 1 head, it will be broadcasted to all heads)
    struct ggml_tensor * KQ_mask = build_inp_KQ_mask();

    auto inp_out_ids = n_tokens > 1 ? build_inp_out_ids() : nullptr;

    for (int il = 0; il < n_layer; ++il) {

        cur = build_std_attention(gf, model.layers[il].attn_norm, inpL,
                inp_pos, il == n_layer - 1 ? inp_out_ids : nullptr, nullptr, KQ_mask,
                nullptr, nullptr, 1.0f/sqrtf(float(n_embd_head)), 0.0f, 0, il, true, false, true, false, true);

        // feed-forward network
        cur = llm_build_ffn(ctx0, lctx, model.layers[il].ffn_norm, cur,
                model.layers[il].ffn_up,   NULL, NULL,
                model.layers[il].ffn_gate, NULL, NULL,
                model.layers[il].ffn_down, NULL, NULL,
                NULL,
                LLM_FFN_SILU, LLM_FFN_PAR, cb, il, gf, true, false,
                batch.embd && (size_t)il < n_deepstack_layers ? deepstack_features[il] : nullptr);

        cur = lctx.cvec.apply_to(ctx0, cur, il);
        cb(cur, "l_out", il);

        // input for next layer
        inpL = cur;
    }

    cur = build_output(lctx, ctx0, inpL, model.output, model.output_norm, cb);
    cb(cur, "result_output", -1);

    ggml_build_forward_expand(gf, cur);

    return gf;
}

ggml_cgraph * llm_build_context::build_qwen3vlmoe() {
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, model.max_nodes(n_tokens), false);

    // mutable variable, needed during the last layer of the computation to skip unused tokens
    int32_t n_tokens = this->n_tokens;

    const int64_t n_embd_full = hparams.n_embd; // main embd + deepstack embds
    const size_t n_deepstack_layers = hparams.n_deepstack_layers;
    const int64_t n_embd = n_embd_full / (n_deepstack_layers + 1);
    const int64_t n_embd_head = hparams.n_embd_head_v(0);

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k(0));
    GGML_ASSERT(n_embd_head == hparams.n_rot);

    struct ggml_tensor * cur;
    struct ggml_tensor * inpL;

    inpL = llm_build_inp_embd(ctx0, lctx, hparams, batch, model.tok_embd, cb);

    int sections[4];
    std::copy(std::begin(hparams.rope_sections), std::begin(hparams.rope_sections) + 4, sections);

    std::vector<struct ggml_tensor *> deepstack_features(n_deepstack_layers, nullptr);

    if (batch.embd) {
        // Image input: split main embd and deepstack embds
        struct ggml_tensor * inpL_main = ggml_view_2d(ctx0, inpL, n_embd, n_tokens, inpL->nb[1], 0);
        for (size_t i = 0; i < n_deepstack_layers; i++) {
            deepstack_features[i] = ggml_view_2d(ctx0, inpL, n_embd, n_tokens, inpL->nb[1], (i + 1) * n_embd * sizeof(float));
        }
        inpL = inpL_main;
    }

    // inp_pos - contains the positions
    struct ggml_tensor * inp_pos = build_inp_pos();

    // KQ_mask (mask for 1 head, it will be broadcasted to all heads)
    struct ggml_tensor * KQ_mask = build_inp_KQ_mask();

    for (int il = 0; il < n_layer; ++il) {
        struct ggml_tensor * inpSA = inpL;

        // norm
        cur = llm_build_norm(ctx0, inpL, hparams,
                model.layers[il].attn_norm, NULL,
                LLM_NORM_RMS, cb, il);
        cb(cur, "attn_norm", il);

        // self_attention
        {
            auto [Qcur, Kcur, Vcur] = llm_build_mul_mat_qkv(gf, cur,
                                                            model.layers[il].wq, nullptr,
                                                            model.layers[il].wk, nullptr,
                                                            model.layers[il].wv, nullptr,
                                                            0, il);

            Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens);
            Qcur = llm_build_norm(ctx0, Qcur, hparams, model.layers[il].attn_q_norm, NULL, LLM_NORM_RMS, cb, il);
            cb(Qcur, "Qcur_normed", il);

            Qcur = ggml_rope_multi(
                    ctx0, Qcur, inp_pos, nullptr,
                    n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );
            cb(Qcur, "Qcur", il);

            Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
            Kcur = llm_build_norm(ctx0, Kcur, hparams, model.layers[il].attn_k_norm, NULL, LLM_NORM_RMS, cb, il);
            cb(Kcur, "Kcur_normed", il);


            Kcur = ggml_rope_multi(
                    ctx0, Kcur, inp_pos, nullptr,
                    n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );
            cb(Kcur, "Kcur", il);

            cb(Vcur, "Vcur", il);

            cur = llm_build_kv(ctx0, lctx, kv_self, gf,
                    model.layers[il].wo, model.layers[il].bo,
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

        // MoE branch
        cur = llm_build_norm(ctx0, ffn_inp, hparams,
                model.layers[il].ffn_norm, NULL,
                LLM_NORM_RMS, cb, il);
        cb(cur, "ffn_norm", il);

        cur =
            llm_build_moe_ffn(ctx0, lctx, cur,
                    model.layers[il].ffn_gate_inp,
                    model.layers[il].ffn_up_exps,
                    model.layers[il].ffn_gate_exps,
                    model.layers[il].ffn_down_exps,
                    nullptr,
                    n_expert, n_expert_used,
                    LLM_FFN_SILU, true,
                    false, 0.0,
                    LLM_EXPERT_GATING_FUNC_SOFTMAX,
                    cb, il, gf, false, model.layers[il].ffn_up_gate_exps);
        cb(cur, "ffn_moe_out", il);

        cur = ggml_add(ctx0, cur, ffn_inp);

        cur = lctx.cvec.apply_to(ctx0, cur, il);
        cb(cur, "l_out", il);

        if (batch.embd && (size_t)il < n_deepstack_layers) {
            cur = ggml_add(ctx0, cur, deepstack_features[il]);
            cb(cur, "deepstack_out", il);
        }

        // input for next layer
        inpL = cur;
    }

    cur = inpL;

    cur = llm_build_norm(ctx0, cur, hparams,
            model.output_norm, NULL,
            LLM_NORM_RMS, cb, -1);
    cb(cur, "result_norm", -1);

    // lm_head
    cur = llm_build_lora_mm(lctx, ctx0, model.output, cur);
    cb(cur, "result_output", -1);

    ggml_build_forward_expand(gf, cur);

    return gf;
}

#include "../llama-build-context.h"
#include "../llama-model.h"
#include "../llama-context.h"
#include "../llama-delta-net.h"

ggml_cgraph * llm_build_context::build_qwen35moe() {

    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, model.max_nodes(n_tokens), false);

    delta_net delta(lctx, batch);

    const int64_t n_embd_head = hparams.n_embd_head_v(0);
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k(0));

    ggml_tensor * inpL = llm_build_inp_embd(ctx0, lctx, hparams, batch, model.tok_embd, cb);
    ggml_tensor * inp_pos = build_inp_pos();
    ggml_tensor * inp_out_ids = n_tokens > 1 ? build_inp_out_ids() : nullptr;
    ggml_tensor * KQ_mask = build_inp_KQ_mask();

    lctx.inp_s_seq_qnext = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, 1, n_tokens);
    cb(lctx.inp_s_seq_qnext, "inp_s_seq_qnext", -1);
    ggml_set_input(lctx.inp_s_seq_qnext);

    float KQ_scale = hparams.f_attention_scale == 0.0f ? 1.0f / sqrtf(float(n_embd_head)) : hparams.f_attention_scale;

    ggml_tensor * cur = nullptr;

    for (int il = 0; il < n_layer; ++il) {

        if (hparams.is_recurrent(il)) {
            cur = delta.build_layer_attn_linear(ctx0, gf, inpL, il == n_layer - 1 ? inp_out_ids : nullptr, il, cb);
        } else {
            cur = build_std_attention(gf, model.layers[il].attn_norm, inpL, inp_pos, il == n_layer - 1 ? inp_out_ids : nullptr, nullptr,
                    KQ_mask, nullptr, nullptr, KQ_scale, 0.0f, 0, il, true, false, true, false, true);
        }

        cur = llm_build_std_moe_ffn(ctx0, lctx, model.layers[il].ffn_norm, cur,
                model.layers[il].ffn_gate_inp,  nullptr,
                model.layers[il].ffn_up_exps,   nullptr,
                model.layers[il].ffn_gate_exps, nullptr,
                model.layers[il].ffn_down_exps, nullptr,
                nullptr,
                model.layers[il].ffn_up_shexp,    nullptr, // we don't have shared expert biases?
                model.layers[il].ffn_gate_shexp,  nullptr,
                model.layers[il].ffn_down_shexp,  nullptr,
                n_expert, n_expert_used,
                LLM_FFN_SILU, true, false, 0.0f,
                LLM_EXPERT_GATING_FUNC_SOFTMAX,
                LLM_FFN_SILU, cb, il, gf, true, model.layers[il].ffn_up_gate_exps, nullptr, model.layers[il].ffn_gate_inp_shexp);

        cur = lctx.cvec.apply_to(ctx0, cur, il);
        cb(cur, "l_out", il);

        inpL = cur;
    }

    cur = build_output(lctx, ctx0, inpL, model.output, model.output_norm, cb);
    cb(cur, "result_output", -1);

    ggml_build_forward_expand(gf, cur);

    return gf;
}

ggml_cgraph * llm_build_context::build_qwen35() {

    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, model.max_nodes(n_tokens), false);

    const int64_t n_embd_head = hparams.n_embd_head_v(0);
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k(0));

    ggml_tensor * cur;

    ggml_tensor * inp_pos = build_inp_pos();

    if (cparams.mtp_op_type != MTP_OP_NONE) {
        // MTP tail-only graph
        ggml_tensor * hidden_states_from_main_model;
        if (cparams.mtp_op_type == MTP_OP_WARMUP || cparams.mtp_op_type == MTP_OP_UPDATE_ACCEPTED) {
            hidden_states_from_main_model = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hparams.n_embd, n_tokens);
        } else {
            hidden_states_from_main_model = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, hparams.n_embd);
        }
        ggml_set_name(hidden_states_from_main_model, "inp_mtp_states");
        ggml_set_input(hidden_states_from_main_model);
        lctx.inp_mtp_states = hidden_states_from_main_model;

        const int il_mtp = hparams.n_layer - 1;
        const auto & mtp_layer = model.layers[il_mtp];

        cur = build_qwen35_mtp(mtp_layer, hidden_states_from_main_model, n_embd_head, gf, inp_pos);
    } else {
        delta_net delta(lctx, batch);

        ggml_tensor * inpL = llm_build_inp_embd(ctx0, lctx, hparams, batch, model.tok_embd, cb);
        ggml_tensor * inp_out_ids = (n_tokens > 1 && !lctx.cparams.mtp) ? build_inp_out_ids() : nullptr;
        ggml_tensor * KQ_mask = build_inp_KQ_mask();

        lctx.inp_s_seq_qnext = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, 1, n_tokens);
        cb(lctx.inp_s_seq_qnext, "inp_s_seq_qnext", -1);
        ggml_set_input(lctx.inp_s_seq_qnext);

        float KQ_scale = hparams.f_attention_scale == 0.0f ? 1.0f / sqrtf(float(n_embd_head)) : hparams.f_attention_scale;

        cur = nullptr;

        const int n_transformer_layers = n_layer - hparams.nextn_predict_layers;
        for (int il = 0; il < n_transformer_layers; ++il) {

            if (hparams.is_recurrent(il)) {
                cur = delta.build_layer_attn_linear(ctx0, gf, inpL, il == n_transformer_layers - 1 ? inp_out_ids : nullptr, il, cb);
            } else {
                cur = build_std_attention(gf, model.layers[il].attn_norm, inpL, inp_pos, il == n_transformer_layers - 1 ? inp_out_ids : nullptr, nullptr,
                        KQ_mask, nullptr, nullptr, KQ_scale, 0.0f, 0, il, true, false, true, false, true);
            }

            cur = llm_build_ffn(ctx0, lctx, model.layers[il].ffn_norm, cur,
                    model.layers[il].ffn_up,   NULL, NULL,
                    model.layers[il].ffn_gate, NULL, NULL,
                    model.layers[il].ffn_down, NULL, NULL,
                    NULL,
                    LLM_FFN_SILU, LLM_FFN_PAR, cb, il, gf, true, false);

            cur = lctx.cvec.apply_to(ctx0, cur, il);
            cb(cur, "l_out", il);

            inpL = cur;
        }

        if (lctx.cparams.mtp) {
            //struct ggml_tensor * embd_copy = ggml_dup(ctx0, inpL);
            //cb(embd_copy, "result_mtp_embd", -1);
            //ggml_set_output(embd_copy);
            cb(inpL, "result_mtp_embd", -1);
            ggml_set_output(inpL);
        }

        cur = build_output(lctx, ctx0, inpL, model.output, model.output_norm, cb);
        cb(cur, "result_output", -1);
    }

    ggml_build_forward_expand(gf, cur);

    return gf;
}

struct ggml_tensor * llm_build_context::build_qwen35_mtp(
    const llama_layer & mtp_layer,
    struct ggml_tensor * prev_embeddings,
    int64_t n_embd_head,
    struct ggml_cgraph * gf,
    struct ggml_tensor * inp_pos) {

    const int il = hparams.n_layer - 1;

    struct ggml_tensor * KQ_mask = build_inp_KQ_mask();

    struct ggml_tensor * inp_out_ids = (n_tokens > 1 && n_outputs < n_tokens) ? build_inp_out_ids() : nullptr;

    ggml_tensor * token_emb = build_inp_embd_mtp(model.tok_embd);

    ggml_tensor * token_emb_norm = llm_build_norm(ctx0, token_emb, hparams, mtp_layer.nextn.enorm, NULL, LLM_NORM_RMS, cb, il);
    ggml_tensor * hidden_state_norm = llm_build_norm(ctx0, prev_embeddings, hparams, mtp_layer.nextn.hnorm, NULL, LLM_NORM_RMS, cb, il);

    ggml_tensor * cur;
    if (mtp_layer.nextn.eh_proj != nullptr) {
        // Full fusion: concat + project (27B, 4B, 2B, 0.8B)
        ggml_tensor * combined = ggml_concat(ctx0, token_emb_norm, hidden_state_norm, 0);
        cb(combined, "mtp_concat", il);
        cur = llm_build_lora_mm(lctx, ctx0, mtp_layer.nextn.eh_proj, combined);
    } else {
        // 9B — no fc/eh_proj
        cur = ggml_add(ctx0, token_emb_norm, hidden_state_norm);
    }
    cb(cur, "mtp_fused", il);

    // Self-Attention (wq may be shared from main model's last layer)
    GGML_ASSERT(il < (int)kv_self.k_l.size() && il < (int)kv_self.v_l.size());
    if (!kv_self.k_l[il] || !kv_self.v_l[il]) {
        LLAMA_LOG_ERROR("%s: KV cache not allocated for MTP layer %d (k=%p, v=%p)\n",
                __func__, il, (void*)kv_self.k_l[il], (void*)kv_self.v_l[il]);
        GGML_ABORT("KV cache not allocated for MTP layer");
    }
    if (!model.layers[il].wq || !model.layers[il].wk || !model.layers[il].wv || !model.layers[il].wo) {
        LLAMA_LOG_ERROR("%s: Missing attention weights for MTP layer %d (wq=%p, wk=%p, wv=%p, wo=%p)\n",
                __func__, il, (void*)model.layers[il].wq, (void*)model.layers[il].wk,
                (void*)model.layers[il].wv, (void*)model.layers[il].wo);
        GGML_ABORT("Missing attention weights for MTP layer");
    }

    const float kq_scale = 1.0f / sqrtf(float(n_embd_head));

    cur = build_std_attention(gf, mtp_layer.attn_norm, cur,
            inp_pos, nullptr, nullptr,
            KQ_mask, nullptr, nullptr,
            kq_scale, 0.0f, 0, il, true, false, true, false, true, nullptr);

    if (inp_out_ids) {
        cur = ggml_get_rows(ctx0, cur, inp_out_ids);
    }

    // Dense FFN — optional (9B and 4B don't have FFN in MTP layer)
    if (mtp_layer.ffn_gate != nullptr) {
        cur = llm_build_ffn(ctx0, lctx, mtp_layer.ffn_norm, cur,
                mtp_layer.ffn_up,   NULL, NULL,
                mtp_layer.ffn_gate, NULL, NULL,
                mtp_layer.ffn_down, NULL, NULL,
                NULL,
                LLM_FFN_SILU, LLM_FFN_PAR, cb, il, gf, true, false);
    }

    cur = lctx.cvec.apply_to(ctx0, cur, il);
    cb(cur, "ffn_out", il);

    // As far as I can tell this was wrong. We need the FFN output, and not the normalized result.
    //cur = llm_build_norm(ctx0, cur, hparams, mtp_layer.nextn.shared_head_norm, NULL, LLM_NORM_RMS, cb, il);
    cb(cur, "result_norm", -1);

    //cur = build_output(lctx, ctx0, cur, model.output, nullptr, cb);
    cur = build_output(lctx, ctx0, cur, model.output, mtp_layer.nextn.shared_head_norm, cb);
    cb(cur, "result_output", -1);

    return cur;
}

#include "../llama-build-context.h"
#include "../llama-model.h"
#include "../llama-context.h"

ggml_cgraph * llm_build_context::build_bailingmoe2() {
    ggml_cgraph * gf = ggml_new_graph_custom(ctx0, model.max_nodes(n_tokens), false);
    const int64_t n_embd_head = hparams.n_embd_head_v(0);
    //const int64_t n_embd_gqa  = hparams.n_embd_v_gqa();

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k(0));

    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = llm_build_inp_embd(ctx0, lctx, hparams, batch, model.tok_embd, cb);

    // inp_pos - contains the positions
    ggml_tensor * inp_pos = build_inp_pos();

    //auto * inp_attn = build_attn_inp_kv();
    ggml_tensor * KQ_mask     = build_inp_KQ_mask();
    //const int64_t n_embd_head = hparams.n_embd_head_v;
    const float kq_scale = 1.0f / sqrtf(float(n_embd_head));

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    const int n_transformer_layers = n_layer - hparams.nextn_predict_layers;

    auto rope_cache = cparams.rope_cache && (rope_type == LLAMA_ROPE_TYPE_NEOX || rope_type == LLAMA_ROPE_TYPE_NORM) ?
        ggml_rope_cache(ctx0, inp_pos, nullptr, n_embd_head, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow) : nullptr;

    for (int il = 0; il < n_transformer_layers; ++il) {
        ggml_tensor * inpSA = inpL;

        // norm
        cur = llm_build_norm(ctx0, inpL, hparams, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, cb, il);
        cb(cur, "attn_norm", il);

        // self_attention
        {
            auto [Qcur, Kcur, Vcur] = llm_build_mul_mat_qkv(gf, cur, model.layers[il].wqkv, model.layers[il].bqkv,
                    nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                    model.layers[il].attn_q_norm, model.layers[il].attn_k_norm, 0.0f, il);

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
            cb(Vcur, "Vcur", il);

            cur = llm_build_kv(ctx0, lctx, kv_self, gf, model.layers[il].wo, model.layers[il].bo,
                    Kcur, Vcur, Qcur, KQ_mask, n_tokens, kv_head, n_kv, kq_scale, cb, il);
        }
        if (il == n_transformer_layers - 1 && inp_out_ids) {
            cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }

        ggml_tensor * sa_out = ggml_add(ctx0, cur, inpSA);
        cb(sa_out, "sa_out", il);

        // MoE branch
        cur = llm_build_norm(ctx0, sa_out, hparams, model.layers[il].ffn_norm, NULL, LLM_NORM_RMS, cb, il);
        cb(cur, "ffn_norm", il);

        // MoE branch
        cur = llm_build_norm(ctx0, sa_out, hparams, model.layers[il].ffn_norm, NULL, LLM_NORM_RMS, cb, il);
        cb(cur, "ffn_norm", il);

        if (static_cast<uint32_t>(il) < hparams.n_layer_dense_lead) {
            cur = llm_build_ffn(ctx0, lctx, nullptr, cur,
                    model.layers[il].ffn_up,   NULL, NULL,
                    model.layers[il].ffn_gate, NULL, NULL,
                    model.layers[il].ffn_down, NULL, NULL,
                    NULL,
                    LLM_FFN_SILU, LLM_FFN_PAR, cb, il);
            cb(cur, "ffn_out", il);
        } else {

            ggml_tensor * moe_out =
                llm_build_moe_ffn(ctx0, lctx, cur,
                        model.layers[il].ffn_gate_inp,
                        model.layers[il].ffn_up_exps,
                        model.layers[il].ffn_gate_exps,
                        model.layers[il].ffn_down_exps,
                        model.layers[il].ffn_exp_probs_b,
                        n_expert, n_expert_used,
                        LLM_FFN_SILU, hparams.expert_weights_norm,
                        true, hparams.expert_weights_scale,
                        (llm_expert_gating_func_type) hparams.expert_gating_func,
                        cb, il, gf, false, model.layers[il].ffn_up_gate_exps);
            cb(moe_out, "ffn_moe_out", il);

            ggml_tensor * ffn_shexp = llm_build_ffn(ctx0, lctx, nullptr, cur,
                    model.layers[il].ffn_up_shexp,   NULL, NULL,
                    model.layers[il].ffn_gate_shexp, NULL, NULL,
                    model.layers[il].ffn_down_shexp, NULL, NULL,
                    NULL,
                    LLM_FFN_SILU, LLM_FFN_PAR, cb, il);
            cb(ffn_shexp, "ffn_shexp", il);

            cur = ggml_add(ctx0, moe_out, ffn_shexp);
            cb(cur, "ffn_out", il);
        }

        cur = ggml_add(ctx0, cur, sa_out);

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

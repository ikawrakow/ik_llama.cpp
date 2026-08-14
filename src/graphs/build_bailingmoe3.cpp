#include "../llama-build-context.h"
#include "../llama-context.h"
#include "../llama-delta-net.h"
#include "../llama-model.h"

ggml_cgraph * llm_build_context::build_bailingmoe3() {
    const bool tp_mode = model.split_mode == LLAMA_SPLIT_MODE_GRAPH || model.split_mode == LLAMA_SPLIT_MODE_ATTN;
#ifdef GGML_USE_VULKAN
    const bool use_f32_attn_precision = true;
#else
    const bool use_f32_attn_precision = lctx.cparams.graph_attn_precision == GGML_TYPE_F32;
#endif

    ggml_cgraph * gf = new_graph_custom();
    delta_net delta(lctx, batch);

    auto inpL = llm_build_inp_embd(ctx0, lctx, hparams, batch, model.tok_embd, cb);
    auto inp_pos = build_inp_pos();
    auto inp_out_ids = build_inp_out_ids();
    auto KQ_mask = build_inp_KQ_mask();

    lctx.inp_s_seq_qnext = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, 1, n_tokens);
    cb(lctx.inp_s_seq_qnext, "inp_s_seq_qnext", -1);
    ggml_set_input(lctx.inp_s_seq_qnext);

    const float mscale = attn_factor * (1.0f + hparams.rope_yarn_log_mul * logf(1.0f / freq_scale));
    const float kq_scale = mscale * mscale / sqrtf(float(hparams.n_embd_head_k(0)));
    const float attn_factor_scaled = 1.0f / (1.0f + 0.1f * logf(1.0f / freq_scale));
    const bool pp_opt = n_tokens >= 128 && lctx.cparams.mla_attn > 1;
    auto rope_cache = cparams.rope_cache && (rope_type == LLAMA_ROPE_TYPE_NEOX || rope_type == LLAMA_ROPE_TYPE_NORM)
        ? ggml_rope_cache(ctx0, inp_pos, nullptr, n_rot, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                ext_factor, attn_factor, beta_fast, beta_slow)
        : nullptr;

    const int n_transformer_layers = hparams.n_layer_kv_from_start;
    ggml_tensor * cur = nullptr;
    for (int il = 0; il < n_transformer_layers; ++il) {
        const bool last_layer = il == n_transformer_layers - 1;

        if (hparams.is_recurrent(il)) {
            cur = delta.build_layer_attn_kda(ctx0, gf, inpL, last_layer ? inp_out_ids : nullptr, il, cb);
        } else {
            auto inpSA = inpL;
            const bool is_tp_layer = tp_mode && model.layers[il].wo->extra;
            const bool direct_q = hparams.n_lora_q == 0;
            if (is_tp_layer) {
                cur = build_deepseek2_tp_attention(gf, il, inpL, KQ_mask, inp_pos, rope_cache,
                        kq_scale, attn_factor_scaled, use_f32_attn_precision, direct_q, pp_opt);
            } else {
                cur = build_deepseek2_layer_attention(gf, il, inpL, KQ_mask, inp_pos, rope_cache,
                        kq_scale, attn_factor_scaled, use_f32_attn_precision, direct_q, pp_opt);
            }
            if (last_layer) {
                cur = ggml_get_rows(ctx0, cur, inp_out_ids);
                if (!is_tp_layer) {
                    inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
                }
            }
            if (!is_tp_layer) {
                cur = ggml_add(ctx0, cur, inpSA);
            }
            cb(cur, "ffn_inp", il);
        }

        if ((uint32_t) il < hparams.n_layer_dense_lead) {
            cur = llm_build_ffn(ctx0, lctx, model.layers[il].ffn_norm, cur,
                    model.layers[il].ffn_up,   nullptr, nullptr,
                    model.layers[il].ffn_gate, nullptr, nullptr,
                    model.layers[il].ffn_down, nullptr, nullptr,
                    nullptr,
                    LLM_FFN_SILU, LLM_FFN_PAR, cb, il, gf, true);
        } else {
            cur = llm_build_std_moe_ffn(ctx0, lctx, model.layers[il].ffn_norm, cur,
                    model.layers[il].ffn_gate_inp, nullptr,
                    model.layers[il].ffn_up_exps, nullptr,
                    model.layers[il].ffn_gate_exps, nullptr,
                    model.layers[il].ffn_down_exps, nullptr,
                    model.layers[il].ffn_exp_probs_b,
                    model.layers[il].ffn_up_shexp, nullptr,
                    model.layers[il].ffn_gate_shexp, nullptr,
                    model.layers[il].ffn_down_shexp, nullptr,
                    n_expert, n_expert_used,
                    LLM_FFN_SILU, hparams.expert_weights_norm,
                    true, hparams.expert_weights_scale,
                    (llm_expert_gating_func_type) hparams.expert_gating_func,
                    LLM_FFN_SILU, cb, il, gf, true, model.layers[il].ffn_up_gate_exps);
        }
        cb(cur, "ffn_out", il);

        cur = lctx.cvec.apply_to(ctx0, cur, il);
        cb(cur, "l_out", il);
        inpL = cur;
    }

    cur = build_output(lctx, ctx0, inpL, model.output, model.output_norm, cb);
    cb(cur, "result_output", -1);
    ggml_build_forward_expand(gf, cur);
    return gf;
}

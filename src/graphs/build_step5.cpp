#include "../llama-build-context.h"
#include "../llama-model.h"
#include "../llama-context.h"

// Dense fallback: sparse_config.enabled=False in the Step5 reference. Sliding
// layers retain their ordinary attention window; full layers attend to all history.
ggml_cgraph * llm_build_context::build_step5() {
    GGML_ASSERT(!cparams.mtp && cparams.mtp_op_type == MTP_OP_NONE);
    ggml_cgraph * gf = new_graph_custom();
    auto inp_pos = build_inp_pos();
    auto inpL = llm_build_inp_embd(ctx0, lctx, hparams, batch, model.tok_embd, cb);
    auto inp_out_ids = build_inp_out_ids();
    auto mask = build_inp_KQ_mask();
    auto mask_swa = build_inp_KQ_mask_swa();
    const int n_trunk = hparams.n_layer - hparams.nextn_predict_layers;

    // Keep Step5's asymmetric SwiGLU clamp local instead of changing the shared
    // FFN helpers' architecture-dependent activation policy.
    auto mlp = [&](ggml_tensor * input, ggml_tensor * gate, ggml_tensor * up,
                   ggml_tensor * down, float limit) {
        auto g = llm_build_lora_mm(lctx, ctx0, gate, input);
        auto u = llm_build_lora_mm(lctx, ctx0, up, input);
        auto act = ggml_fused_mul_unary(ctx0, g, u, GGML_UNARY_OP_SILU);
        memcpy(act->op_params + 1, &limit, sizeof(limit));
        return llm_build_lora_mm(lctx, ctx0, down, act);
    };

    for (int il = 0; il < n_trunk; ++il) {
        const auto & layer = model.layers[il];
        const bool swa = hparams.swa_layers[il];
        // The reference applies scaling only to the configured layer types.
        auto rope_factors = ((swa ? 2u : 1u) & hparams.rope_scaling_apply_mask)
            ? build_rope_factors(il) : nullptr;
        auto cur = build_std_attention(gf, layer.attn_norm, inpL, inp_pos,
                il == n_trunk - 1 && n_tokens > 1 ? inp_out_ids : nullptr,
                rope_factors, swa ? mask_swa : mask, nullptr, nullptr,
                1.0f / sqrtf(float(hparams.n_embd_head_k(il))), 0.0f,
                swa ? hparams.n_swa : 0, il, true, false, true);

        auto input = llm_build_norm(ctx0, cur, hparams, layer.ffn_norm, nullptr,
                LLM_NORM_RMS, cb, il);
        ggml_tensor * ffn;
        if (layer.ffn_gate) {
            ffn = mlp(input, layer.ffn_gate, layer.ffn_up, layer.ffn_down,
                    hparams.swiglu_limits_shared[il]);
        } else {
            // Selection-only router bias; mixing weights use the unbiased sigmoid.
            auto logits = llm_build_lora_mm(lctx, ctx0, layer.ffn_gate_inp, input);
            ggml_mul_mat_set_prec(logits, GGML_PREC_F32);
            auto probs = ggml_sigmoid(ctx0, logits);
            auto selection = ggml_add(ctx0, probs, layer.ffn_exp_probs_b);
            auto ids = ggml_top_k(ctx0, selection, n_expert_used);
            auto unbiased = ggml_sub(ctx0, selection, layer.ffn_exp_probs_b);
            auto weights = ggml_get_rows(ctx0,
                    ggml_reshape_3d(ctx0, unbiased, 1, n_expert, input->ne[1]), ids);
            weights = ggml_reshape_2d(ctx0, weights, n_expert_used, input->ne[1]);
            if (hparams.expert_weights_norm) {
                auto sum = ggml_scale_bias(ctx0, ggml_sum_rows(ctx0, weights), 1.0f, 1e-20f);
                weights = ggml_div(ctx0, weights, sum);
            }
            weights = ggml_scale(ctx0, weights, hparams.expert_weights_scale);
            weights = ggml_reshape_3d(ctx0, weights, 1, n_expert_used, input->ne[1]);
            cb(ids, "ffn_moe_topk", il);
            cb(weights, "ffn_moe_weights", il);

            auto expert_in = ggml_reshape_3d(ctx0, input, n_embd, 1, input->ne[1]);
            auto act = layer.ffn_up_gate_exps
                ? ggml_moe_up_gate(ctx0, layer.ffn_up_gate_exps, nullptr, expert_in, ids, GGML_UNARY_OP_SILU)
                : ggml_moe_up_gate(ctx0, layer.ffn_up_exps, layer.ffn_gate_exps, expert_in, ids, GGML_UNARY_OP_SILU);
            const float limit = hparams.swiglu_limits[il];
            memcpy(act->op_params + 1, &limit, sizeof(limit));
            auto experts = ggml_mul_mat_id(ctx0, layer.ffn_down_exps, act, ids);
            auto routed = ggml_mul_multi_add(ctx0, experts, weights);
            auto shared = mlp(input, layer.ffn_gate_shexp, layer.ffn_up_shexp,
                    layer.ffn_down_shexp, hparams.swiglu_limits_shared[il]);
            ffn = ggml_add(ctx0, routed, shared);
        }
        cb(ffn, "ffn_out", il);
        inpL = lctx.cvec.apply_to(ctx0, ggml_add(ctx0, cur, ffn), il);
        cb(inpL, "l_out", il);
    }
    auto output = build_output(lctx, ctx0, inpL, model.output, model.output_norm, cb);
    cb(output, "result_output", -1);
    ggml_build_forward_expand(gf, output);
    return gf;
}

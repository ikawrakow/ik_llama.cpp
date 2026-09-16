#include "../llama-build-context.h"
#include "../llama-model.h"
#include "../llama-context.h"

// K2 Horizon grouped RMS norm: splits embedding into n_groups groups,
// applies RMS norm to each group independently, then reassembles.
static ggml_tensor * k2_horizon_group_rms_norm(
    ggml_context * ctx,
    ggml_tensor * cur,
    ggml_tensor * weight,
    int64_t n_groups,
    float eps)
{
    GGML_ASSERT(n_groups > 0);
    GGML_ASSERT(cur->ne[0] % n_groups == 0);

    const int64_t n_embd   = cur->ne[0];
    const int64_t n_tokens = cur->ne[1];

    // reshape: (n_embd, n_tokens) -> (n_embd/n_groups, n_groups, n_tokens)
    cur = ggml_reshape_3d(ctx, cur, n_embd / n_groups, n_groups, n_tokens);

    // RMS norm per group
    cur = ggml_rms_norm(ctx, cur, eps);

    // reshape back: (n_embd, n_tokens)
    cur = ggml_reshape_2d(ctx, cur, n_embd, n_tokens);

    // apply learned weights
    if (weight != nullptr) {
        cur = ggml_mul(ctx, cur, weight);
    }

    return cur;
}

// K2 Horizon MoVA: Mixture of Value Attention
static ggml_tensor * k2_horizon_routed_value(
    ggml_context * ctx,
    llama_context & lctx,
    const llama_layer & layer,
    ggml_tensor * cur,
    int il,
    const llama_hparams & hparams,
    const llm_build_cb & cb)
{
    const int64_t n_embd   = cur->ne[0];
    const int64_t n_tokens = cur->ne[1];
    const int64_t n_embd_v_gqa = hparams.n_embd_v_gqa(il);
    const int64_t n_values = hparams.n_value_expert;
    const int64_t n_used   = hparams.n_value_expert_used;

    GGML_ASSERT(layer.attn_v_gate != nullptr);
    GGML_ASSERT(layer.attn_v_exps != nullptr);

    // router logits: (n_embd) . (n_embd, n_values) -> (n_values, n_tokens)
    ggml_tensor * logits = llm_build_context::llm_build_lora_mm(lctx, ctx, layer.attn_v_gate, cur);

    // gating function
    ggml_tensor * probs;
    switch (hparams.expert_gating_func) {
        case LLM_EXPERT_GATING_FUNC_SOFTMAX:
            probs = ggml_soft_max(ctx, logits);
            break;
        case LLM_EXPERT_GATING_FUNC_SIGMOID:
            probs = ggml_sigmoid(ctx, logits);
            break;
        default:
            GGML_ABORT("Unsupported K2 Horizon value-router gating function");
    }

    // optional bias
    if (layer.attn_v_gate_b != nullptr) {
        probs = ggml_add(ctx, probs, layer.attn_v_gate_b);
        cb(probs, "v_moe_probs_biased", il);
    }

    cb(logits, "v_moe_logits", il);
    cb(probs, "v_moe_probs", il);

    // top-k selection
    ggml_tensor * selected_experts = ggml_top_k(ctx, probs, n_used); // [n_used, n_tokens]

    // extract selected weights via argsort-style indexing
    ggml_tensor * selection_probs = ggml_reshape_3d(ctx, probs, 1, n_values, n_tokens);
    ggml_tensor * selected_weights = ggml_get_rows(ctx, selection_probs, selected_experts);
    // [1, n_used, n_tokens]

    // normalize weights (conditional on expert_weights_norm, matching upstream)
    if (hparams.expert_weights_norm) {
        selected_weights = ggml_reshape_2d(ctx, selected_weights, n_used, n_tokens);
        ggml_tensor * wsum = ggml_sum_rows(ctx, selected_weights);
        wsum = ggml_clamp(ctx, wsum, 6.103515625e-5f, INFINITY);
        selected_weights = ggml_div(ctx, selected_weights, wsum);
        selected_weights = ggml_reshape_3d(ctx, selected_weights, 1, n_used, n_tokens);
        cb(selected_weights, "v_moe_weights_norm", il);
    }

    // expert weights scaling (matching upstream)
    if (hparams.expert_weights_scale != 0.0f && hparams.expert_weights_scale != 1.0f) {
        selected_weights = ggml_scale(ctx, selected_weights, hparams.expert_weights_scale);
        cb(selected_weights, "v_moe_weights_scaled", il);
    }

    // compute routed values: indexed matmul + silu + weighted sum
    // attn_v_exps: (n_embd, n_embd_v_gqa, n_values)
    // value_inp:   (n_embd, 1, n_tokens)
    ggml_tensor * value_inp = ggml_reshape_3d(ctx, cur, n_embd, 1, n_tokens);
    ggml_tensor * values = llm_build_context::llm_build_lora_mm_id(lctx, ctx, layer.attn_v_exps, value_inp, selected_experts);
    // values: (n_embd_v_gqa, n_used, n_tokens)
    values = ggml_silu(ctx, values);
    values = ggml_mul(ctx, values, selected_weights);

    // sum across selected experts
    ggml_tensor * value_out = ggml_view_2d(ctx, values, n_embd_v_gqa, n_tokens, values->nb[2], 0);
    for (int64_t i = 1; i < n_used; ++i) {
        ggml_tensor * part = ggml_view_2d(ctx, values, n_embd_v_gqa, n_tokens, values->nb[2], i * values->nb[1]);
        value_out = ggml_add(ctx, value_out, part);
    }

    // making it contiguous in case it isn't (for one expert only)
    if (n_used == 1) value_out = ggml_cont(ctx, value_out);

    cb(value_out, "Vcur_routed", il);
    return value_out;
}

ggml_cgraph * llm_build_context::build_k2horizon() {
    ggml_cgraph * gf = new_graph_custom();

    int32_t n_tokens = this->n_tokens;

    const int64_t n_embd_head = hparams.n_embd_head_v(0);
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k(0));

    ggml_tensor * cur;
    ggml_tensor * inpL;

    // 1. Embedding
    inpL = llm_build_inp_embd(ctx0, lctx, hparams, batch, model.tok_embd, cb);

    // 2. Position
    struct ggml_tensor * inp_pos = build_inp_pos();

    // 3. Attention mask
    ggml_tensor * KQ_mask = build_inp_KQ_mask();

    // 4. Output IDs
    auto inp_out_ids = n_tokens > 1 ? build_inp_out_ids() : nullptr;

    // 5. Scale
    const float kq_scale = 1.0f / sqrtf(float(n_embd_head));

    // 6. RoPE cache (precomputed lookup, avoids per-layer recomputation)
    ggml_tensor * rope_cache = nullptr;
    if (cparams.rope_cache && (hparams.rope_type == LLAMA_ROPE_TYPE_NEOX || hparams.rope_type == LLAMA_ROPE_TYPE_NORM)) {
        const int64_t n_rot = hparams.n_embd_head_k(0);
        rope_cache = ggml_rope_cache(ctx0, inp_pos, nullptr, n_rot, n_rot, hparams.rope_type,
                hparams.n_ctx_orig_yarn, hparams.rope_freq_base_train, hparams.rope_freq_scale_train,
                hparams.yarn_ext_factor, hparams.rope_attn_factor, hparams.yarn_beta_fast, hparams.yarn_beta_slow);
    }

    // 7. Layer loop
    for (int il = 0; il < n_layer; ++il) {
        struct ggml_tensor * inpSA = inpL;

        const bool is_moe_layer = hparams.n_expert > 0 &&
            static_cast<uint32_t>(il) >= hparams.n_layer_dense_lead;
        const bool is_mova_layer = is_moe_layer && hparams.n_value_expert > 0;

        // === grouped RMS norm before attention ===
        cur = k2_horizon_group_rms_norm(ctx0, inpL, model.layers[il].attn_norm,
                hparams.n_norm_groups, hparams.f_norm_rms_eps);
        cb(cur, "attn_norm", il);

        ggml_tensor * attn_inp = cur;

        // === Q ===
        ggml_tensor * Qcur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wq, cur);
        if (model.layers[il].attn_q_norm != nullptr) {
            Qcur = k2_horizon_group_rms_norm(ctx0, Qcur, model.layers[il].attn_q_norm,
                    hparams.n_head(il), hparams.f_norm_rms_eps);
        }

        // === K ===
        ggml_tensor * Kcur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wk, cur);
        if (model.layers[il].attn_k_norm != nullptr) {
            Kcur = k2_horizon_group_rms_norm(ctx0, Kcur, model.layers[il].attn_k_norm,
                    hparams.n_head_kv(il), hparams.f_norm_rms_eps);
        }

        // === V: standard or MoVA routed ===
        ggml_tensor * Vcur;
        if (is_mova_layer) {
            Vcur = k2_horizon_routed_value(ctx0, lctx, model.layers[il], cur, il, hparams, cb);
        } else {
            Vcur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wv, cur);
        }

        // reshape Q/K/V
        Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, hparams.n_head(il), n_tokens);
        Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, hparams.n_head_kv(il), n_tokens);
        Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, hparams.n_head_kv(il), n_tokens);

        // RoPE — use fast path when rope_cache is available
        if (rope_cache) {
            Qcur = ggml_rope_fast(ctx0, Qcur, rope_cache);
            Kcur = ggml_rope_fast(ctx0, Kcur, rope_cache);
        } else {
            Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, nullptr, hparams.rope_n_rot(il), hparams.rope_type,
                    hparams.n_ctx_orig_yarn, hparams.rope_freq_base_train, hparams.rope_freq_scale_train,
                    hparams.yarn_ext_factor, hparams.rope_attn_factor, hparams.yarn_beta_fast, hparams.yarn_beta_slow);
            Kcur = ggml_rope_ext(ctx0, Kcur, inp_pos, nullptr, hparams.rope_n_rot(il), hparams.rope_type,
                    hparams.n_ctx_orig_yarn, hparams.rope_freq_base_train, hparams.rope_freq_scale_train,
                    hparams.yarn_ext_factor, hparams.rope_attn_factor, hparams.yarn_beta_fast, hparams.yarn_beta_slow);
        }

        cb(Qcur, "Qcur", il);
        cb(Kcur, "Kcur", il);
        cb(Vcur, "Vcur", il);

        // === Attention via llm_build_kv (handles KV store + attention in one call) ===
        // Replaces direct ggml_flash_attn_ext() which crashes on k2-horizon's
        // 128 head_dim with quantized KV cache (IQK FA unsupported K-type).
        // llm_build_kv stores K/V into cache AND computes Q*K attention internally.
        // n_kv is the class member (KV cache size), not n_head_kv
        if (model.layers[il].wqkv_gate == nullptr) {
            // standard attention
            cur = llm_build_kv(ctx0, lctx, kv_self, gf,
                    model.layers[il].wo, model.layers[il].wo_b,
                    Kcur, Vcur, Qcur,
                    KQ_mask, n_tokens, kv_head, n_kv,
                    kq_scale, cb, il);
        } else {
            // attention with softplus gating — no output projection yet
            cur = llm_build_kv(ctx0, lctx, kv_self, gf,
                    nullptr, nullptr,
                    Kcur, Vcur, Qcur,
                    KQ_mask, n_tokens, kv_head, n_kv,
                    kq_scale, cb, il);

            // softplus gate
            constexpr float LN2 = 0.6931471805599453f;
            constexpr float ONE_OVER_LN2 = 1.4426950408889634f;

            ggml_tensor * gate = llm_build_lora_mm(lctx, ctx0, model.layers[il].wqkv_gate, attn_inp);
            gate = ggml_scale(ctx0, gate, LN2);
            gate = ggml_softplus(ctx0, gate);
            gate = ggml_scale(ctx0, gate, ONE_OVER_LN2);

            cur = ggml_mul(ctx0, cur, gate);
            cur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wo, cur);
            if (model.layers[il].wo_b != nullptr) {
                cur = ggml_add(ctx0, cur, model.layers[il].wo_b);
            }
        }

        // last token selection
        if (il == n_layer - 1 && inp_out_ids != nullptr) {
            cur = ggml_get_rows(ctx0, cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }

        // residual
        cur = ggml_add(ctx0, cur, inpSA);
        ggml_tensor * ffn_inp = cur;
        cb(ffn_inp, "ffn_inp", il);

        // === grouped RMS norm before FFN ===
        cur = k2_horizon_group_rms_norm(ctx0, ffn_inp, model.layers[il].ffn_norm,
                hparams.n_norm_groups, hparams.f_norm_rms_eps);
        cb(cur, "ffn_norm", il);

        // === FFN (dense or MoE) ===
        if (is_moe_layer) {
            ggml_tensor * moe_out = llm_build_moe_ffn(ctx0, lctx, cur,
                    model.layers[il].ffn_gate_inp,
                    model.layers[il].ffn_up_exps,
                    model.layers[il].ffn_gate_exps,
                    model.layers[il].ffn_down_exps,
                    model.layers[il].ffn_exp_probs_b,
                    hparams.n_expert, hparams.n_expert_used,
                    LLM_FFN_SILU, hparams.expert_weights_norm, true, hparams.expert_weights_scale,
                    (llm_expert_gating_func_type)hparams.expert_gating_func,
                    cb, il, gf, false,
                    model.layers[il].ffn_up_gate_exps);

            if (model.layers[il].ffn_gate_shexp != nullptr) {
                ggml_tensor * shared_out = llm_build_ffn(ctx0, lctx, nullptr, cur,
                        model.layers[il].ffn_up_shexp,   nullptr, nullptr,
                        model.layers[il].ffn_gate_shexp, nullptr, nullptr,
                        model.layers[il].ffn_down_shexp, nullptr, nullptr,
                        nullptr,
                        LLM_FFN_SILU, LLM_FFN_PAR, cb, il, gf);
                cur = ggml_add(ctx0, moe_out, shared_out);
            } else {
                cur = moe_out;
            }
        } else {
            cur = llm_build_ffn(ctx0, lctx, nullptr, cur,
                    model.layers[il].ffn_up,   nullptr, nullptr,
                    model.layers[il].ffn_gate, nullptr, nullptr,
                    model.layers[il].ffn_down, nullptr, nullptr,
                    nullptr,
                    LLM_FFN_SILU, LLM_FFN_PAR, cb, il, gf);
        }
        cb(cur, "ffn_out", il);

        // FFN residual
        cur = ggml_add(ctx0, cur, ffn_inp);
        cur = lctx.cvec.apply_to(ctx0, cur, il);
        cb(cur, "l_out", il);

        inpL = cur;
    }

    // === Final grouped RMS norm ===
    cur = k2_horizon_group_rms_norm(ctx0, inpL, model.output_norm,
            hparams.n_norm_groups, hparams.f_norm_rms_eps);
    cb(cur, "result_norm", -1);

    // === Vocab projection ===
    cur = llm_build_lora_mm(lctx, ctx0, model.output, cur);
    cb(cur, "result_output", -1);

    ggml_build_forward_expand(gf, cur);
    return gf;
}

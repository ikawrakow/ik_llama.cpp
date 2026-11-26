#include "../llama-build-context.h"
#include "../llama-context.h"
#include "../llama-model.h"

#include <cmath>

// Dory 1.0 (NVIDIA Nemotron Labs): a looped nGPT stack. Physical modules alternate attention
// (even blk index) and MLP (odd). The recurrent group shares weights across loop passes but
// owns an independent KV cache slot per visit. Weights are aliased by expand_looped_layers();
// hparams.phys_layer(il) recovers the shared physical module for logic.
// Reference: transformers/models/dory/modeling_dory.py.

namespace {

// L2Norm projects onto the unit sphere: x / sqrt(sum_rows(x^2) + 1e-6), not RMS.
// ggml_l2_norm clamps the denominator instead, so use the reference form.
// Identity (exact): x/sqrt(S + eps) = (1/sqrt(n)) * rms_norm(x, eps/n), n = ne[0],
// so the 5-op sqr/sum_rows/scale_bias/sqrt/div chain is rms_norm + scale (2 ops).
// dory_rms_norm exposes the unscaled rms form: sqrt(n) * dory_l2_norm(x). The FFN
// consumes it directly because its suv_scale = sqrt(n_embd) cancels the factor.
ggml_tensor * dory_rms_norm(ggml_context * ctx, ggml_tensor * x) {
    static constexpr float eps = 1e-6f;
    return ggml_rms_norm(ctx, x, eps / (float) x->ne[0]);
}

ggml_tensor * dory_l2_norm(ggml_context * ctx, ggml_tensor * x) {
    return ggml_scale(ctx, dory_rms_norm(ctx, x), 1.0f / std::sqrt((float) x->ne[0]));
}

// GGUF scale vectors may be BF16; ggml binary ops do not mix BF16 with F32 activations.
ggml_tensor * dory_f32(ggml_context * ctx, ggml_tensor * t) {
    return t->type == GGML_TYPE_F32 ? t : ggml_cast(ctx, t, GGML_TYPE_F32);
}

// The residual rate |alpha * alpha_init/init_std| is folded into the F32
// residual_alpha weight at load (llm_fold_dory_scales); the graph reads it directly.

// Interpolate from the normalized input: h_norm + rate * (cur - h_norm).
ggml_tensor * dory_residual(ggml_context * ctx, ggml_tensor * h_norm, ggml_tensor * cur, ggml_tensor * rate) {
    // ggml takes the result shape from the first operand, so the token-shaped difference must
    // precede the per-channel rate.
    ggml_tensor * delta = ggml_sub(ctx, cur, h_norm);
    return ggml_add(ctx, h_norm, ggml_mul(ctx, delta, rate));
}

} // namespace

ggml_cgraph * llm_build_context::build_dory() {
    ggml_cgraph * gf = new_graph_custom();

    // The constant nGPT rescales — sqk', s_gate', the residual rate
    // |alpha*alpha_init/init_std|, and the logit scale output_s/init_std — are folded
    // into their F32 weights at load (llm_fold_dory_scales). suv_scale = sqrt(n_embd)
    // is cancelled by the FFN's rms_norm input form; only its reciprocal survives,
    // recovering the unit-sphere h for the residual.
    const float suv_scale = std::sqrt((float) n_embd);

    // Attention uses head_dim^-0.5 after q was scaled by the folded sqk'.
    const float kq_scale = 1.0f / std::sqrt((float) n_embd_head_k);

    const int region = (int) (hparams.loop_end - hparams.loop_start);

    // masks: sliding modules use the SWA mask, full-attention modules the causal mask
    bool has_full = false, has_swa = false;
    for (int il = 0; il < n_layer; ++il) {
        if (hparams.phys_layer(il) % 2 == 0) {
            if (hparams.swa_layers[il]) {
                has_swa = true;
            } else {
                has_full = true;
            }
        }
    }
    const bool compact = kv_self.any_compacted();
    ggml_tensor * KQ_mask  = has_full ? build_inp_KQ_mask() : nullptr;
    ggml_tensor * swa_mask = nullptr;
    if (has_swa) {
        swa_mask = compact ? build_swa_mask_for_graph(hparams.n_swa, true) : build_inp_KQ_mask_swa();
    }

    ggml_tensor * inpL = llm_build_inp_embd(ctx0, lctx, hparams, batch, model.tok_embd, cb);
    ggml_tensor * inp_pos = build_inp_pos();
    ggml_tensor * inp_out_ids = n_tokens > 1 ? build_inp_out_ids() : nullptr;

    // il selects the KV cache slot, attention mask and RoPE base; the aliased model.layers[il]
    // holds the shared physical weights.
    auto build_module = [&](int il, ggml_tensor * inp) -> ggml_tensor * {
        const auto & layer = model.layers[il];
        const bool is_attn = hparams.phys_layer(il) % 2 == 0;

        ggml_tensor * h;
        ggml_tensor * cur;
        ggml_tensor * rate;

        if (is_attn) {
            h = dory_l2_norm(ctx0, inp);
            cb(h, "module_inp_norm", il);

            ggml_tensor * Qcur = llm_build_lora_mm(lctx, ctx0, layer.wq, h);
            ggml_tensor * Kcur = llm_build_lora_mm(lctx, ctx0, layer.wk, h);
            ggml_tensor * Vcur = llm_build_lora_mm(lctx, ctx0, layer.wv, h);
            ggml_tensor * Gcur = llm_build_lora_mm(lctx, ctx0, layer.wqkv_gate, h); // [n_embd_q, n_tokens]

            Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head_k, n_head,    n_tokens);
            Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head_k, n_head_kv, n_tokens);
            Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head_v, n_head_kv, n_tokens);

            // unit-normalize q/k over head_dim before RoPE
            Qcur = dory_l2_norm(ctx0, Qcur);
            Kcur = dory_l2_norm(ctx0, Kcur);

            const int n_swa = hparams.swa_layers[il] ? (int) hparams.n_swa : 0;
            float freq_base_l  = n_swa > 0 ? hparams.rope_freq_base_train_swa  : cparams.rope_freq_base;
            float freq_scale_l = n_swa > 0 ? hparams.rope_freq_scale_train_swa : hparams.rope_freq_scale_train;
            const int n_rot_l = hparams.rope_n_rot(il);
            ggml_tensor * rope_factors = build_rope_factors(il);
            Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, rope_factors, n_rot_l, rope_type, n_ctx_orig,
                    freq_base_l, freq_scale_l, ext_factor, attn_factor, beta_fast, beta_slow);
            Kcur = ggml_rope_ext(ctx0, Kcur, inp_pos, rope_factors, n_rot_l, rope_type, n_ctx_orig,
                    freq_base_l, freq_scale_l, ext_factor, attn_factor, beta_fast, beta_slow);
            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            // q = q * sqk', per (head, channel); sqk' = (sqk * sqk_scale)^2 was
            // folded into the F32 weight at load (llm_fold_dory_scales)
            ggml_tensor * qk_scale = ggml_reshape_2d(ctx0, layer.attn_qk_scale, n_embd_head_k, n_head);
            Qcur = ggml_mul(ctx0, Qcur, qk_scale);
            cb(Qcur, "Qcur_scaled", il);

            // merged attention output (wo applied after gating below)
            ggml_tensor * mask = n_swa > 0 ? swa_mask : KQ_mask;
            cur = llm_build_kv(ctx0, lctx, kv_self, gf, nullptr, nullptr, Kcur, Vcur, Qcur,
                    mask, n_tokens, kv_head, n_kv, kq_scale, cb, il, nullptr, n_swa);
            cb(cur, "attn_out", il);

            // gate = sigmoid(g_proj(x) * s_gate'); s_gate' = (s_gate * s_gate_scale)^2
            // was folded into the F32 weight at load (llm_fold_dory_scales)
            ggml_tensor * gate = ggml_sigmoid(ctx0, ggml_mul(ctx0, Gcur, layer.attn_gate_scale));
            cb(gate, "attn_gate", il);
            cur = ggml_mul(ctx0, cur, gate);
            cb(cur, "attn_gated", il);

            cur = dory_l2_norm(ctx0, cur);                 // attn_sub_norm
            cur = llm_build_lora_mm(lctx, ctx0, layer.wo, cur); // o_proj
            cur = dory_l2_norm(ctx0, cur);                 // post_attention_layernorm
            cb(cur, "attn_out_norm", il);

            rate = dory_f32(ctx0, layer.attn_resid_alpha); // |alpha*lambda| folded at load
        } else {
            // h_rms = sqrt(n_embd) * l2_norm(inp). The ffn_gate/ffn_up GEMMs consume
            // it unscaled: the inherited sqrt(n_embd) cancels the suv_scale that used
            // to multiply their results, so the scale vectors apply plain (2 ggml_scale
            // saved). The residual interpolation still needs the unit-sphere h.
            ggml_tensor * h_rms = dory_rms_norm(ctx0, inp);
            h = ggml_scale(ctx0, h_rms, 1.0f / suv_scale);
            cb(h, "module_inp_norm", il);

            ggml_tensor * gate = llm_build_lora_mm(lctx, ctx0, layer.ffn_gate, h_rms);
            gate = ggml_mul(ctx0, gate, dory_f32(ctx0, layer.ffn_gate_scale));
            ggml_tensor * up = llm_build_lora_mm(lctx, ctx0, layer.ffn_up, h_rms);
            up = ggml_mul(ctx0, up, dory_f32(ctx0, layer.ffn_up_scale));

            cur = ggml_swiglu_split(ctx0, gate, up);       // silu(gate) * up
            cb(cur, "ffn_swiglu", il);

            cur = dory_l2_norm(ctx0, cur);                 // ffn_sub_norm
            cur = llm_build_lora_mm(lctx, ctx0, layer.ffn_down, cur);
            cur = dory_l2_norm(ctx0, cur);                 // post_feedforward_layernorm
            cb(cur, "ffn_out", il);

            rate = dory_f32(ctx0, layer.ffn_resid_alpha); // |alpha*lambda| folded at load
        }

        ggml_tensor * out = dory_residual(ctx0, h, cur, rate);
        cb(out, "l_out", il);
        return out;
    };

    ggml_tensor * input_hidden = nullptr;
    const int rec_end_slot = (int) hparams.loop_start + region * (int) hparams.n_loops;

    for (int il = 0; il < n_layer; ++il) {
        // entering the recurrent group: capture the normalized input-phase output once
        if (il == (int) hparams.loop_start) {
            input_hidden = dory_l2_norm(ctx0, inpL);
            cb(input_hidden, "input_hidden", -1);
        }
        // reinject before every recurrent pass, including the first:
        // hidden = L2Norm(hidden) + input_hidden (pass 0 therefore sees 2 * input_hidden)
        if (il >= (int) hparams.loop_start && il < rec_end_slot &&
                (il - (int) hparams.loop_start) % region == 0) {
            inpL = ggml_add(ctx0, dory_l2_norm(ctx0, inpL), input_hidden);
            cb(inpL, "loop_reinject", il);
        }

        inpL = build_module(il, inpL);
        inpL = lctx.cvec.apply_to(ctx0, inpL, il);
        cb(inpL, "l_out", il);
    }

    if (n_outputs == 0 && !cparams.embeddings) {
        ggml_build_forward_expand(gf, inpL);
        return gf;
    }

    if (inp_out_ids) {
        inpL = ggml_get_rows(ctx0, inpL, inp_out_ids);
    }

    // final L2Norm
    ggml_tensor * cur = dory_l2_norm(ctx0, inpL);
    cb(cur, "result_norm", -1);

    if (cparams.embeddings) {
        ggml_build_forward_expand(gf, cur);
        return gf;
    }

    // logits = (h @ W) * logit_scale'; logit_scale' = output_s/init_std folded at load
    cur = build_output(lctx, ctx0, cur, model.output, nullptr, cb);
    cur = ggml_mul(ctx0, cur, dory_f32(ctx0, model.output_s));
    cb(cur, "result_output", -1);
    ggml_build_forward_expand(gf, cur);

    return gf;
}

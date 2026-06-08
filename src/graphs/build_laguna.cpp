#include "../llama-build-context.h"
#include "../llama-model.h"
#include "../llama-context.h"

ggml_cgraph * llm_build_context::build_laguna() {
    ggml_cgraph * gf = new_graph_custom();

    ggml_tensor * inpL        = llm_build_inp_embd(ctx0, lctx, hparams, batch, model.tok_embd, cb);
    ggml_tensor * inp_pos     = build_inp_pos();
    ggml_tensor * inp_out_ids = build_inp_out_ids();
    ggml_tensor * KQ_mask     = build_inp_KQ_mask();
    ggml_tensor * KQ_mask_swa = build_inp_KQ_mask_swa();

    for (int il = 0; il < n_layer; ++il) {
        const bool is_swa = hparams.swa_layers[il];
        const int n_swa_l = is_swa ? hparams.n_swa : 0;

        ggml_tensor * inpSA = inpL;

        ggml_tensor * cur = llm_build_norm(ctx0, inpL, hparams, model.layers[il].attn_norm, nullptr, LLM_NORM_RMS, cb, il);
        cb(cur, "attn_norm", il);
        ggml_tensor * input_normed = cur;

        ggml_tensor * Qcur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wq, cur);
        cb(Qcur, "Qcur", il);
        ggml_tensor * Kcur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wk, cur);
        cb(Kcur, "Kcur", il);
        ggml_tensor * Vcur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wv, cur);
        cb(Vcur, "Vcur", il);
        ggml_build_forward_expand(gf, Qcur);
        ggml_build_forward_expand(gf, Kcur);
        ggml_build_forward_expand(gf, Vcur);

        const int64_t n_head_l      = hparams.n_head(il);
        const int64_t n_head_kv_l   = hparams.n_head_kv(il);
        const int64_t n_embd_head_k = hparams.n_embd_head_k(il);
        const int64_t n_embd_head_v = hparams.n_embd_head_v(il);

        Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head_k, n_head_l, n_tokens);
        Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head_k, n_head_kv_l, n_tokens);

        if (model.layers[il].attn_q_norm) {
            Qcur = llm_build_norm(ctx0, Qcur, hparams, model.layers[il].attn_q_norm, nullptr, LLM_NORM_RMS, cb, il);
            cb(Qcur, "Qcur_normed", il);
            ggml_build_forward_expand(gf, Qcur);
        }
        if (model.layers[il].attn_k_norm) {
            Kcur = llm_build_norm(ctx0, Kcur, hparams, model.layers[il].attn_k_norm, nullptr, LLM_NORM_RMS, cb, il);
            cb(Kcur, "Kcur_normed", il);
            ggml_build_forward_expand(gf, Kcur);
        }

        const float freq_base_l  = is_swa ? hparams.rope_freq_base_train_swa  : cparams.rope_freq_base;
        const float freq_scale_l = is_swa ? hparams.rope_freq_scale_train_swa : cparams.rope_freq_scale;
        const float ext_factor_l  = is_swa ? 0.0f  : ext_factor;
        const float attn_factor_l = is_swa ? 1.0f  : attn_factor;
        const float beta_fast_l   = is_swa ? 32.0f : beta_fast;
        const float beta_slow_l   = is_swa ? 1.0f  : beta_slow;
        ggml_tensor * rope_factors = is_swa ? nullptr : build_rope_factors(il);
        const int n_rot_l = hparams.rope_n_rot(il);

        Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, rope_factors,
                n_rot_l, rope_type, n_ctx_orig, freq_base_l, freq_scale_l,
                ext_factor_l, attn_factor_l, beta_fast_l, beta_slow_l);
        Kcur = ggml_rope_ext(ctx0, Kcur, inp_pos, rope_factors,
                n_rot_l, rope_type, n_ctx_orig, freq_base_l, freq_scale_l,
                ext_factor_l, attn_factor_l, beta_fast_l, beta_slow_l);
        cb(Qcur, "Qcur_roped", il);
        cb(Kcur, "Kcur_roped", il);

        cur = llm_build_kv(ctx0, lctx, kv_self, gf,
                nullptr, nullptr,
                Kcur, Vcur, Qcur,
                is_swa ? KQ_mask_swa : KQ_mask,
                n_tokens, kv_head, n_kv,
                1.0f / sqrtf(float(n_embd_head_k)), cb, il, nullptr, n_swa_l);
        cb(cur, "attn_out", il);

        if (model.layers[il].wqkv_gate) {
            ggml_tensor * gate = llm_build_lora_mm(lctx, ctx0, model.layers[il].wqkv_gate, input_normed);
            cb(gate, "attn_gate", il);
            gate = ggml_softplus(ctx0, gate);
            cb(gate, "attn_gate_softplus", il);

            ggml_tensor * attn_3d = ggml_reshape_3d(ctx0, cur, n_embd_head_v, n_head_l, n_tokens);
            ggml_tensor * gate_3d = ggml_reshape_3d(ctx0, gate, 1, n_head_l, n_tokens);
            cb(gate_3d, "attn_gate_3d", il);

            cur = ggml_mul(ctx0, attn_3d, gate_3d);
            cb(cur, "attn_gated_3d", il);
            cur = ggml_reshape_2d(ctx0, cur, n_embd_head_v * n_head_l, n_tokens);
            cb(cur, "attn_gated", il);
        }

        cur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wo, cur);
        cb(cur, "attn_proj", il);

        if (il == n_layer - 1 && inp_out_ids) {
            cur   = ggml_get_rows(ctx0, cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }

        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        if (model.layers[il].ffn_gate_inp == nullptr) {
            cur = llm_build_ffn(ctx0, lctx, model.layers[il].ffn_norm, ffn_inp,
                    model.layers[il].ffn_up,   nullptr, nullptr,
                    model.layers[il].ffn_gate, nullptr, nullptr,
                    model.layers[il].ffn_down, nullptr, nullptr,
                    nullptr,
                    LLM_FFN_SILU, LLM_FFN_PAR, cb, il);
            cb(cur, "ffn_out", il);
            cur = ggml_add(ctx0, cur, ffn_inp);
        } else {
            cur = llm_build_std_moe_ffn(ctx0, lctx, model.layers[il].ffn_norm, ffn_inp,
                    model.layers[il].ffn_gate_inp,  model.layers[il].ffn_gate_inp_b,
                    model.layers[il].ffn_up_exps,   model.layers[il].ffn_up_exps_b,
                    model.layers[il].ffn_gate_exps, model.layers[il].ffn_gate_exps_b,
                    model.layers[il].ffn_down_exps, model.layers[il].ffn_down_exps_b,
                    model.layers[il].ffn_exp_probs_b,
                    model.layers[il].ffn_up_shexp,    nullptr,
                    model.layers[il].ffn_gate_shexp,  nullptr,
                    model.layers[il].ffn_down_shexp,  nullptr,
                    n_expert, n_expert_used,
                    LLM_FFN_SILU, hparams.expert_weights_norm, hparams.expert_weights_scale != 0.0f, hparams.expert_weights_scale,
                    (llm_expert_gating_func_type) hparams.expert_gating_func,
                    LLM_FFN_SILU, cb, il, gf, true, model.layers[il].ffn_up_gate_exps);
        }

        cur = lctx.cvec.apply_to(ctx0, cur, il);
        cb(cur, "l_out", il);

        inpL = cur;
    }

    ggml_tensor * cur = build_output(lctx, ctx0, inpL, model.output, model.output_norm, cb);
    cb(cur, "result_output", -1);

    ggml_build_forward_expand(gf, cur);

    return gf;
}

#include "../llama-build-context.h"
#include "../llama-context.h"
#include "../llama-model.h"

#include <cmath>

ggml_cgraph * llm_build_context::build_dflash() {
    const int64_t n_embd_head_k = hparams.n_embd_head_k(0);
    const int64_t n_embd_head_v = hparams.n_embd_head_v(0);
    const int64_t n_target_features = hparams.dflash_n_target_features;
        const int64_t ctx_len = std::max<int64_t>(1, (int64_t) cparams.n_ctx - (int64_t) hparams.dflash_block_size);
    const int64_t n_kv_total = ctx_len + n_tokens;

    GGML_ASSERT(n_embd_head_k == n_embd_head_v);
    GGML_ASSERT(n_target_features > 0);

    ggml_cgraph * gf = ggml_new_graph_custom(ctx0, model.max_nodes((int) std::max<int64_t>(n_tokens, ctx_len)) + 32 * n_layer, false);

    lctx.inp_dflash_target_features = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_target_features, ctx_len);
    ggml_set_input(lctx.inp_dflash_target_features);
    cb(lctx.inp_dflash_target_features, "dflash_target_features", -1);

    lctx.inp_dflash_pos_ctx = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, ctx_len);
    ggml_set_input(lctx.inp_dflash_pos_ctx);
    cb(lctx.inp_dflash_pos_ctx, "dflash_pos_ctx", -1);

    lctx.inp_dflash_kq_mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_kv_total, GGML_PAD(n_tokens, GGML_KQ_MASK_PAD));
    ggml_set_input(lctx.inp_dflash_kq_mask);
    cb(lctx.inp_dflash_kq_mask, "dflash_kq_mask", -1);

    ggml_tensor * dflash_kq_mask = flash_attn ? ggml_cast(ctx0, lctx.inp_dflash_kq_mask, GGML_TYPE_F16) : lctx.inp_dflash_kq_mask;

    ggml_tensor * tok_embd = model.tok_embd;
    if (tok_embd == nullptr) {
        tok_embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_Q4_0, n_embd, hparams.n_vocab);
    }

    ggml_tensor * inpL = llm_build_inp_embd(ctx0, lctx, hparams, batch, tok_embd, cb);
    ggml_tensor * inp_pos = build_inp_pos();
    ggml_tensor * inp_out_ids = (n_tokens > 1 && n_outputs < n_tokens) ? build_inp_out_ids() : nullptr;

    ggml_tensor * fused_target = llm_build_lora_mm(lctx, ctx0, model.dflash_fc, lctx.inp_dflash_target_features);
    fused_target = llm_build_norm(ctx0, fused_target, hparams, model.dflash_hidden_norm, nullptr, LLM_NORM_RMS, cb, -1);
    cb(fused_target, "dflash_target_fused", -1);

    const float kq_scale = 1.0f / std::sqrt((float) n_embd_head_k);

    for (int il = 0; il < n_layer; ++il) {
        ggml_tensor * inpSA = inpL;

        ggml_tensor * cur = llm_build_norm(ctx0, inpL, hparams, model.layers[il].attn_norm, nullptr, LLM_NORM_RMS, cb, il);
        cb(cur, "attn_norm", il);

        ggml_tensor * Qcur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wq, cur);
        Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head_k, n_head, n_tokens);
        Qcur = llm_build_norm(ctx0, Qcur, hparams, model.layers[il].attn_q_norm, nullptr, LLM_NORM_RMS, cb, il);
        Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, nullptr,
                n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                ext_factor, attn_factor, beta_fast, beta_slow);
        cb(Qcur, "Qcur", il);

        ggml_tensor * Kcur_noise = llm_build_lora_mm(lctx, ctx0, model.layers[il].wk, cur);
        Kcur_noise = ggml_reshape_3d(ctx0, Kcur_noise, n_embd_head_k, n_head_kv, n_tokens);
        Kcur_noise = llm_build_norm(ctx0, Kcur_noise, hparams, model.layers[il].attn_k_norm, nullptr, LLM_NORM_RMS, cb, il);
        Kcur_noise = ggml_rope_ext(ctx0, Kcur_noise, inp_pos, nullptr,
                n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                ext_factor, attn_factor, beta_fast, beta_slow);
        cb(Kcur_noise, "Kcur_noise", il);

        ggml_tensor * Vcur_noise = llm_build_lora_mm(lctx, ctx0, model.layers[il].wv, cur);
        Vcur_noise = ggml_reshape_3d(ctx0, Vcur_noise, n_embd_head_v, n_head_kv, n_tokens);
        cb(Vcur_noise, "Vcur_noise", il);

        ggml_tensor * Kcur_ctx = llm_build_lora_mm(lctx, ctx0, model.layers[il].wk, fused_target);
        Kcur_ctx = ggml_reshape_3d(ctx0, Kcur_ctx, n_embd_head_k, n_head_kv, ctx_len);
        Kcur_ctx = llm_build_norm(ctx0, Kcur_ctx, hparams, model.layers[il].attn_k_norm, nullptr, LLM_NORM_RMS, cb, il);
        Kcur_ctx = ggml_rope_ext(ctx0, Kcur_ctx, lctx.inp_dflash_pos_ctx, nullptr,
                n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                ext_factor, attn_factor, beta_fast, beta_slow);
        cb(Kcur_ctx, "Kcur_ctx", il);

        ggml_tensor * Vcur_ctx = llm_build_lora_mm(lctx, ctx0, model.layers[il].wv, fused_target);
        Vcur_ctx = ggml_reshape_3d(ctx0, Vcur_ctx, n_embd_head_v, n_head_kv, ctx_len);
        cb(Vcur_ctx, "Vcur_ctx", il);

        ggml_tensor * Kcur = ggml_concat(ctx0, Kcur_ctx, Kcur_noise, 2);
        ggml_tensor * Vcur = ggml_concat(ctx0, Vcur_ctx, Vcur_noise, 2);
        cb(Kcur, "Kcur", il);
        cb(Vcur, "Vcur", il);

        Kcur = ggml_cast(ctx0, Kcur, GGML_TYPE_F16);
        Vcur = ggml_cast(ctx0, Vcur, GGML_TYPE_F16);
        cb(Qcur, "Qcur", il);
        cb(Kcur, "Kcur_f16", il);
        cb(Vcur, "Vcur_f16", il);

        ggml_tensor * q = ggml_permute(ctx0, Qcur, 0, 2, 1, 3);
        ggml_tensor * k = ggml_cont(ctx0, ggml_permute(ctx0, Kcur, 0, 2, 1, 3));
        ggml_tensor * v = ggml_cont(ctx0, ggml_permute(ctx0, Vcur, 0, 2, 1, 3));
        cb(q, "q", il);
        cb(k, "k", il);
        cb(v, "v", il);

        cur = ggml_flash_attn_ext(ctx0, q, k, v, dflash_kq_mask, kq_scale, hparams.f_max_alibi_bias,
                hparams.attn_soft_cap ? hparams.f_attn_logit_softcapping : 0.0f);
        cb(cur, "flash_attn", il);
        ggml_build_forward_expand(gf, cur);

        cur = ggml_reshape_2d(ctx0, cur, model.layers[il].wo->ne[0], n_tokens);
        cb(cur, "flash_attn_reshaped", il);

        cur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wo, cur);
        cb(cur, "kqv_out", il);

        cur = ggml_add(ctx0, cur, inpSA);
        cb(cur, "attn_residual", il);

        ggml_tensor * ffn_residual = cur;
        cur = llm_build_norm(ctx0, cur, hparams, model.layers[il].attn_post_norm, nullptr, LLM_NORM_RMS, cb, il);
        cb(cur, "attn_post_norm", il);

        cur = llm_build_ffn(ctx0, lctx, nullptr, cur,
                model.layers[il].ffn_up, nullptr, nullptr,
                model.layers[il].ffn_gate, nullptr, nullptr,
                model.layers[il].ffn_down, nullptr, nullptr,
                nullptr,
                LLM_FFN_SILU, LLM_FFN_PAR, cb, il, gf, false, false);
        cb(cur, "ffn_out", il);

        cur = ggml_add(ctx0, cur, ffn_residual);
        cb(cur, "l_out", il);

        inpL = cur;
    }

    ggml_tensor * output = model.output;
    if (output == nullptr) {
        output = ggml_new_tensor_2d(ctx0, GGML_TYPE_Q4_0, n_embd, hparams.n_vocab);
    }

    ggml_tensor * result_input = inpL;
    if (inp_out_ids) {
        result_input = ggml_get_rows(ctx0, result_input, inp_out_ids);
        cb(result_input, "result_output_rows", -1);
    }

    ggml_tensor * result = build_output(lctx, ctx0, result_input, output, model.output_norm, cb);
    cb(result, "result_output", -1);
    ggml_build_forward_expand(gf, result);

    return gf;
}

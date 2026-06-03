#include "../llama-build-context.h"
#include "../llama-model.h"
#include "../llama-context.h"

ggml_cgraph * llm_build_context::build_mellum() {
    ggml_cgraph * gf = new_graph_custom();

    struct ggml_tensor * cur;
    struct ggml_tensor * inpL;

    inpL = llm_build_inp_embd(ctx0, lctx, hparams, batch, model.tok_embd, cb);

    struct ggml_tensor * inp_pos     = build_inp_pos();
    struct ggml_tensor * inp_out_ids = n_tokens > 1 ? build_inp_out_ids() : nullptr;
    struct ggml_tensor * KQ_mask     = build_inp_KQ_mask();
    struct ggml_tensor * KQ_mask_swa = build_inp_KQ_mask_swa();

    for (int il = 0; il < n_layer; ++il) {
        const bool is_swa = hparams.swa_layers[il];
        const int64_t n_embd_head = hparams.n_embd_head_v(il);
        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k(il));
        GGML_ASSERT(n_embd_head == hparams.n_rot);

        struct ggml_tensor * inpSA = inpL;

        cur = llm_build_norm(ctx0, inpL, hparams, model.layers[il].attn_norm, nullptr, LLM_NORM_RMS, cb, il);
        cb(cur, "attn_norm", il);

        auto [Qcur, Kcur, Vcur] = llm_build_mul_mat_qkv(gf, cur,
                model.layers[il].wqkv, nullptr,
                model.layers[il].wqk,  nullptr,
                model.layers[il].wq,   nullptr,
                model.layers[il].wk,   nullptr,
                model.layers[il].wv,   nullptr,
                model.layers[il].attn_q_norm, model.layers[il].attn_k_norm, 0.0f, il);

        const float freq_base_l   = is_swa ? hparams.rope_freq_base_train_swa : freq_base;
        const float freq_scale_l  = is_swa ? 1.0f : freq_scale;
        const float ext_factor_l  = is_swa ? 0.0f : ext_factor;
        const float attn_factor_l = is_swa ? 1.0f : attn_factor;

        Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base_l, freq_scale_l,
                ext_factor_l, attn_factor_l, beta_fast, beta_slow);
        Kcur = ggml_rope_ext(ctx0, Kcur, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base_l, freq_scale_l,
                ext_factor_l, attn_factor_l, beta_fast, beta_slow);

        cb(Qcur, "Qcur", il);
        cb(Kcur, "Kcur", il);
        cb(Vcur, "Vcur", il);

        cur = llm_build_kv(ctx0, lctx, kv_self, gf,
                model.layers[il].wo, model.layers[il].bo,
                Kcur, Vcur, Qcur, is_swa ? KQ_mask_swa : KQ_mask,
                n_tokens, kv_head, n_kv, 1.0f/sqrtf(float(n_embd_head)), cb, il, nullptr, is_swa ? hparams.n_swa : 0);

        if (il == n_layer - 1 && inp_out_ids) {
            cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }

        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        cur = llm_build_std_moe_ffn(ctx0, lctx, model.layers[il].ffn_norm, ffn_inp,
                model.layers[il].ffn_gate_inp,  nullptr,
                model.layers[il].ffn_up_exps,   nullptr,
                model.layers[il].ffn_gate_exps, nullptr,
                model.layers[il].ffn_down_exps, nullptr,
                nullptr,
                nullptr, nullptr,
                nullptr, nullptr,
                nullptr, nullptr,
                n_expert, n_expert_used,
                LLM_FFN_SILU, true, false, 0.0f,
                LLM_EXPERT_GATING_FUNC_SOFTMAX,
                LLM_FFN_SILU, cb, il, gf, true,
                model.layers[il].ffn_up_gate_exps);

        cur = lctx.cvec.apply_to(ctx0, cur, il);
        cb(cur, "l_out", il);

        inpL = cur;
    }

    cur = build_output(lctx, ctx0, inpL, model.output, model.output_norm, cb);
    cb(cur, "result_output", -1);

    ggml_build_forward_expand(gf, cur);

    return gf;
}

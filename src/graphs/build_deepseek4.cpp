#include "../llama-model.h"
#include "../llama-context.h"
#include "../llama-build-context.h"

#include <cmath>

ggml_cgraph * llm_build_context::build_deepseek4() {
    ggml_cgraph * gf = new_graph_custom();

    const int64_t n_embd_head = hparams.n_embd_head_k(0);
    const int64_t n_embd_head_rope = hparams.n_rot;
    const int64_t n_embd_head_nope = n_embd_head - n_embd_head_rope;

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_v(0));
    GGML_ASSERT(n_embd_head_nope > 0);

    ggml_tensor * inpL = llm_build_inp_embd(ctx0, lctx, hparams, batch, model.tok_embd, cb);
    ggml_tensor * inp_pos = build_inp_pos();
    ggml_tensor * KQ_mask = build_inp_KQ_mask();

    for (int il = 0; il < n_layer; ++il) {
        ggml_tensor * inpSA = inpL;

        ggml_tensor * cur = llm_build_norm(ctx0, inpL, hparams, model.layers[il].attn_norm, nullptr, LLM_NORM_RMS, cb, il);
        cb(cur, "attn_norm", il);

        ggml_tensor * qr = llm_build_lora_mm(lctx, ctx0, model.layers[il].wq_a, cur);
        cb(qr, "qr", il);

        qr = llm_build_norm(ctx0, qr, hparams, model.layers[il].attn_q_a_norm, nullptr, LLM_NORM_RMS, cb, il);
        cb(qr, "qr_norm", il);

        ggml_tensor * q = llm_build_lora_mm(lctx, ctx0, model.layers[il].wq_b, qr);
        q = ggml_reshape_3d(ctx0, q, n_embd_head, n_head, n_tokens);
        q = ggml_rms_norm(ctx0, q, hparams.f_norm_rms_eps);
        cb(q, "q_norm", il);

        ggml_tensor * q_nope = ggml_view_3d(ctx0, q, n_embd_head_nope, n_head, n_tokens,
                ggml_row_size(q->type, n_embd_head),
                ggml_row_size(q->type, n_embd_head) * n_head,
                0);
        ggml_tensor * q_pe = ggml_view_3d(ctx0, q, n_embd_head_rope, n_head, n_tokens,
                ggml_row_size(q->type, n_embd_head),
                ggml_row_size(q->type, n_embd_head) * n_head,
                ggml_row_size(q->type, n_embd_head_nope));
        q_pe = ggml_rope_ext(ctx0, q_pe, inp_pos, nullptr, n_embd_head_rope, rope_type, n_ctx_orig,
                freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
        cb(q_pe, "q_pe", il);
        q = ggml_concat(ctx0, q_nope, q_pe, 0);
        cb(q, "q", il);

        ggml_tensor * kv = llm_build_lora_mm(lctx, ctx0, model.layers[il].wkv_latent, cur);
        kv = llm_build_norm(ctx0, kv, hparams, model.layers[il].attn_kv_norm, nullptr, LLM_NORM_RMS, cb, il);
        kv = ggml_reshape_3d(ctx0, kv, n_embd_head, 1, n_tokens);
        cb(kv, "kv_norm", il);

        ggml_tensor * kv_nope = ggml_view_3d(ctx0, kv, n_embd_head_nope, 1, n_tokens,
                ggml_row_size(kv->type, n_embd_head),
                ggml_row_size(kv->type, n_embd_head),
                0);
        ggml_tensor * kv_pe = ggml_view_3d(ctx0, kv, n_embd_head_rope, 1, n_tokens,
                ggml_row_size(kv->type, n_embd_head),
                ggml_row_size(kv->type, n_embd_head),
                ggml_row_size(kv->type, n_embd_head_nope));
        kv_pe = ggml_rope_ext(ctx0, kv_pe, inp_pos, nullptr, n_embd_head_rope, rope_type, n_ctx_orig,
                freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
        cb(kv_pe, "kv_pe", il);
        kv = ggml_concat(ctx0, kv_nope, kv_pe, 0);
        cb(kv, "kv", il);

        ggml_tensor * attn = llm_build_kv(ctx0, lctx, kv_self, gf,
                nullptr, nullptr,
                kv, kv, q, KQ_mask, n_tokens, kv_head, n_kv,
                1.0f / std::sqrt(float(n_embd_head)), cb, il);
        cb(attn, "attn_raw", il);

        attn = ggml_reshape_3d(ctx0, attn, n_embd_head, n_head, n_tokens);
        ggml_tensor * attn_nope = ggml_view_3d(ctx0, attn, n_embd_head_nope, n_head, n_tokens,
                ggml_row_size(attn->type, n_embd_head),
                ggml_row_size(attn->type, n_embd_head) * n_head,
                0);
        ggml_tensor * attn_pe = ggml_view_3d(ctx0, attn, n_embd_head_rope, n_head, n_tokens,
                ggml_row_size(attn->type, n_embd_head),
                ggml_row_size(attn->type, n_embd_head) * n_head,
                ggml_row_size(attn->type, n_embd_head_nope));
        cb(attn_pe, "attn_derope", il);
        attn = ggml_concat(ctx0, attn_nope, attn_pe, 0);

        const int64_t o_group_dim = model.layers[il].wo_a->ne[0];
        const int64_t n_groups = (n_head * n_embd_head) / o_group_dim;
        const int64_t o_lora_rank = model.layers[il].wo_b->ne[0] / n_groups;

        GGML_ASSERT((n_head * n_embd_head) % o_group_dim == 0);
        GGML_ASSERT(model.layers[il].wo_b->ne[0] % n_groups == 0);

        attn = ggml_reshape_3d(ctx0, attn, o_group_dim, n_groups, n_tokens);
        attn = ggml_permute(ctx0, attn, 0, 2, 1, 3);

        ggml_tensor * oa = ggml_mul_mat(ctx0,
                ggml_reshape_3d(ctx0, model.layers[il].wo_a, model.layers[il].wo_a->ne[0], o_lora_rank, n_groups),
                attn);
        cb(oa, "attn_wo_a", il);
        oa = ggml_permute(ctx0, oa, 0, 2, 1, 3);
        oa = ggml_cont_2d(ctx0, oa, o_lora_rank * n_groups, n_tokens);

        cur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wo_b, oa);
        cb(cur, "attn_out", il);

        if (il == n_layer - 1) {
            ggml_tensor * inp_out_ids = build_inp_out_ids();
            cur = ggml_get_rows(ctx0, cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }

        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        cur = llm_build_norm(ctx0, ffn_inp, hparams, model.layers[il].ffn_norm, nullptr, LLM_NORM_RMS, cb, il);
        cb(cur, "ffn_norm", il);

        if ((uint32_t) il < hparams.n_layer_dense_lead) {
            cur = llm_build_ffn(ctx0, lctx, nullptr, cur,
                    model.layers[il].ffn_up,   nullptr, nullptr,
                    model.layers[il].ffn_gate, nullptr, nullptr,
                    model.layers[il].ffn_down, nullptr, nullptr,
                    nullptr,
                    LLM_FFN_SILU, LLM_FFN_PAR, cb, il);
        } else {
            ggml_tensor * moe_out = llm_build_moe_ffn(ctx0, lctx, cur,
                    model.layers[il].ffn_gate_inp,
                    model.layers[il].ffn_up_exps,
                    model.layers[il].ffn_gate_exps,
                    model.layers[il].ffn_down_exps,
                    model.layers[il].ffn_exp_probs_b,
                    n_expert, n_expert_used,
                    LLM_FFN_SILU, hparams.expert_weights_norm,
                    true, hparams.expert_weights_scale,
                    (enum llm_expert_gating_func_type) hparams.expert_gating_func,
                    cb, il, gf, false, model.layers[il].ffn_up_gate_exps);
            cb(moe_out, "ffn_moe_out", il);

            ggml_tensor * ffn_shexp = llm_build_ffn(ctx0, lctx, nullptr, cur,
                    model.layers[il].ffn_up_shexp,   nullptr, nullptr,
                    model.layers[il].ffn_gate_shexp, nullptr, nullptr,
                    model.layers[il].ffn_down_shexp, nullptr, nullptr,
                    nullptr,
                    LLM_FFN_SILU, LLM_FFN_PAR, cb, il);
            cb(ffn_shexp, "ffn_shexp", il);

            cur = ggml_add(ctx0, moe_out, ffn_shexp);
        }

        cb(cur, "ffn_out", il);

        cur = ggml_add(ctx0, cur, ffn_inp);
        cur = lctx.cvec.apply_to(ctx0, cur, il);
        cb(cur, "l_out", il);

        inpL = cur;
    }

    ggml_tensor * out = build_output(lctx, ctx0, inpL, model.output, model.output_norm, cb);
    cb(out, "result_output", -1);

    ggml_build_forward_expand(gf, out);

    return gf;
}

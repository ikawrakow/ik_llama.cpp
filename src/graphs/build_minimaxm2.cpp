#include "../llama-build-context.h"
#include "../llama-model.h"
#include "../llama-context.h"

ggml_cgraph* llm_build_context::build_minimaxm2() {
    ggml_cgraph * gf = ggml_new_graph_custom(ctx0, model.max_nodes(n_tokens), false);
    const int64_t n_embd_head = hparams.n_embd_head_v(0);
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k(0));
    // GGML_ASSERT(n_embd_head == hparams.n_rot); this is wrong in case of minimax, head_dim = 128, n_rot = 64

    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = llm_build_inp_embd(ctx0, lctx, hparams, batch, model.tok_embd, cb);

    ggml_tensor * inp_pos = build_inp_pos();


    //auto * inp_attn = build_attn_inp_kv();
    ggml_tensor * inp_out_ids = build_inp_out_ids();
    ggml_tensor * KQ_mask = build_inp_KQ_mask();

    for (int il = 0; il < n_layer; ++il) {
        ggml_tensor* inpSA = inpL;

        cur = inpL;

        // self_attention
        if (model.split_mode == LLAMA_SPLIT_MODE_GRAPH || model.split_mode == LLAMA_SPLIT_MODE_ATTN) {
            // Unfortunately we cannot use build_std_attention because Q and K get normed before being RoPE'd,
            // but the RMS norm is applied on the whole row, and not per head as it is normally done.
            // Hence, we need to keep a copy of wq and wk on each device, do the whole matrix multiplications
            // on each device, apply the norm, and only then take from the result the self attention portion
            // being processed on the given device. If we would split wq and wk, we would need to reassemble
            // the whole Q and K via reduce-concat to apply the RMS norm, and that would kill performance.
            // Alternatively, we would need to add an extra reduce op, which computes the squared sum on each device,
            // than does a reduce-add operation to compute the total sum (per row) of Q and K, and then
            // it performs RMS norm using that. This would be possibly better, but let's leave it for another day.
            auto wq = (ggml_split_tensor_t *)model.layers[il].wq->extra;
            auto wk = (ggml_split_tensor_t *)model.layers[il].wk->extra;
            auto wv = (ggml_split_tensor_t *)model.layers[il].wv->extra;
            auto wo = (ggml_split_tensor_t *)model.layers[il].wo->extra;
            GGML_ASSERT(wq && wk && wv && wo);
            GGML_ASSERT(wq->n_device == wk->n_device && wq->n_device == wv->n_device && wq->n_device == wo->n_device);
            auto q_norm = (ggml_split_tensor_t *)model.layers[il].attn_q_norm->extra;
            auto k_norm = (ggml_split_tensor_t *)model.layers[il].attn_k_norm->extra;
            auto attn_norm = (ggml_split_tensor_t *)model.layers[il].attn_norm->extra;
            GGML_ASSERT(attn_norm && q_norm && k_norm);
            GGML_ASSERT(wq->n_device == q_norm->n_device && wq->n_device == k_norm->n_device && wq->n_device == attn_norm->n_device);
            auto kl = (ggml_split_tensor_t *)kv_self.k_l[il]->extra;
            auto vl = (ggml_split_tensor_t *)kv_self.v_l[il]->extra;
            GGML_ASSERT(wq->n_device == kl->n_device && wq->n_device == vl->n_device);
            int head_count    = 0;
            int head_count_kv = 0;
            int n_device = wq->n_device;
            std::vector<ggml_tensor *> attn(n_device, nullptr);
            bool input_added = false;
            for (int id = 0; id < n_device; ++id) {
                if (!wq->splits[id]) continue;
                int il_id = 1000*il + id;
                auto input = get_input_tensor_sm_graph(ctx0, inpL, id);
                cur = llm_build_norm(ctx0, input, hparams, attn_norm->splits[id], nullptr, LLM_NORM_RMS, cb, il_id);

                auto Qcur = llm_build_lora_mm(lctx, ctx0, wq->splits[id], cur);
                cb(Qcur, "Qcur", il_id);

                auto Kcur = llm_build_lora_mm(lctx, ctx0, wk->splits[id], cur);
                cb(Kcur, "Kcur", il_id);

                auto Vcur = llm_build_lora_mm(lctx, ctx0, wv->splits[id], cur);
                cb(Vcur, "Vcur", il_id);

                // Do this here so Q, K, V matrix multiplications may be fused
                ggml_build_forward_expand(gf, Qcur);
                ggml_build_forward_expand(gf, Kcur);
                ggml_build_forward_expand(gf, Vcur);

                Qcur = llm_build_norm(ctx0, Qcur, hparams, q_norm->splits[id], nullptr, LLM_NORM_RMS, cb, il_id);
                cb(Qcur, "Qcur_normed", il_id);

                Kcur = llm_build_norm(ctx0, Kcur, hparams, k_norm->splits[id], nullptr, LLM_NORM_RMS, cb, il_id);
                cb(Kcur, "Kcur_normed", il_id);

                // reshape for multi-head
                Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens);
                Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);

                int gqa_ratio   = n_head / n_head_kv;
                int nhead_kv_id = Vcur->ne[0] / n_embd_head_v;
                int nhead_id    = nhead_kv_id * gqa_ratio;
                GGML_ASSERT(nhead_kv_id > 0 && nhead_kv_id <= n_head_kv);

                Qcur = ggml_view_3d(ctx0, Qcur, n_embd_head_k, nhead_id,    n_tokens, Qcur->nb[1], Qcur->nb[2], head_count*Qcur->nb[1]);
                Kcur = ggml_view_3d(ctx0, Kcur, n_embd_head_k, nhead_kv_id, n_tokens, Kcur->nb[1], Kcur->nb[2], head_count_kv*Kcur->nb[1]);
                head_count    += nhead_id;
                head_count_kv += nhead_kv_id;

                Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                        ext_factor, attn_factor, beta_fast, beta_slow);
                Kcur = ggml_rope_ext(ctx0, Kcur, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                        ext_factor, attn_factor, beta_fast, beta_slow);
                cb(Qcur, "Qcur_roped", il_id);
                cb(Kcur, "Kcur_roped", il_id);

                if (cparams.k_cache_hadamard) {
                    Qcur = ggml_hadamard(ctx0, Qcur, n_embd_head_k);
                    Kcur = ggml_hadamard(ctx0, Kcur, n_embd_head_k);
                    cb(Qcur, "Qcur_hadamard", il_id);
                    cb(Kcur, "Kcur_hadamard", il_id);
                }
                ggml_build_forward_expand(gf, Qcur);
                ggml_build_forward_expand(gf, Kcur);
                if (cparams.v_cache_hadamard) {
                    Vcur = ggml_hadamard(ctx0, Vcur, n_embd_head_v);
                    cb(Vcur, "Vcur_hadamard", il_id);
                    ggml_build_forward_expand(gf, Vcur);
                }

                // Store K, V in KV cache
                auto idx = 2*wq->n_device*il + 2*id;
                GGML_ASSERT(idx+1 < (int)lctx.cache_copies.size());
                auto k_row_size = ggml_row_size(kl->splits[id]->type, n_embd_head_k);
                auto k_cache_view = ggml_view_2d(ctx0, kl->splits[id], n_embd_head_k, n_tokens*nhead_kv_id,
                        k_row_size, k_row_size*nhead_kv_id*kv_head);
                lctx.cache_copies[idx+0].cpy  = ggml_cpy(ctx0, Kcur, k_cache_view);
                lctx.cache_copies[idx+0].step = k_row_size*nhead_kv_id;

                auto v_cache_view = ggml_view_1d(ctx0, vl->splits[id], n_tokens*wv->splits[id]->ne[1],
                            kv_head*ggml_row_size(vl->splits[id]->type, wv->splits[id]->ne[1]));
                lctx.cache_copies[idx+1].cpy  = ggml_cpy(ctx0, Vcur, v_cache_view);
                lctx.cache_copies[idx+1].step = ggml_row_size(vl->splits[id]->type, wv->splits[id]->ne[1]);

                ggml_build_forward_expand(gf, lctx.cache_copies[idx+0].cpy);
                ggml_build_forward_expand(gf, lctx.cache_copies[idx+1].cpy);

                auto q = ggml_permute(ctx0, Qcur, 0, 2, 1, 3);
                cb(q, "q", il_id);

                auto k = ggml_view_3d(ctx0, kl->splits[id], n_embd_head_k, n_kv, nhead_kv_id,
                             ggml_row_size(kl->splits[id]->type, n_embd_head_k)*nhead_kv_id,
                             ggml_row_size(kl->splits[id]->type, n_embd_head_k), 0);
                cb(k, "k", il_id);

                auto v = ggml_view_3d(ctx0, vl->splits[id], n_embd_head_v, n_kv, nhead_kv_id,
                             ggml_row_size( vl->splits[id]->type, wv->splits[id]->ne[1]),
                             ggml_row_size( vl->splits[id]->type, n_embd_head_v), 0);
                cb(v, "v", il_id);

                cur = ggml_flash_attn_ext(ctx0, q, k, v, KQ_mask, 1.0f / sqrtf(float(n_embd_head)), hparams.f_max_alibi_bias, 0.0f);
                cb(cur, "fa", il_id);

                if (cparams.v_cache_hadamard) {
                    cur = ggml_hadamard(ctx0, cur, n_embd_head_v);
                    cb(cur, "fa_h", il_id);
                }

                cur = ggml_reshape_2d(ctx0, cur, wo->splits[id]->ne[0], n_tokens);
                cb(cur, "fa_reshaped", il_id);

                if (il == n_layer - 1 && n_tokens > 1 && inp_out_ids) {
                    cur = ggml_get_rows(ctx0, cur, inp_out_ids);
                    cb(cur, "fa_get_rows", il_id);
                    if (!input_added) {
                        input = ggml_get_rows(ctx0, input, inp_out_ids);
                        cb(cur, "sainp_get_rows", il_id);
                    }
                }

                cur = llm_build_lora_mm(lctx, ctx0, wo->splits[id], cur);
                cb(cur, "kqv_wo", il_id);

                if (!input_added) {
                    cur = ggml_add(ctx0, cur, input);
                    cb(cur, "attn_out_with_input", il);
                    input_added = true;
                }

                if (cur->ne[1] > 32 && lctx.cparams.reduce_type != GGML_TYPE_F32) {
                    cur = ggml_cast(ctx0, cur, lctx.cparams.reduce_type);
                }
                ggml_build_forward_expand(gf, cur);
                attn[id] = cur;
            }

            cur = ggml_reduce(ctx0, attn.data(), n_device, GGML_OP_ADD);
            ggml_build_forward_expand(gf, cur);
            cb(cur, "attn_combined", il);

        } else {
            cur = llm_build_norm(ctx0, inpL, hparams, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, cb, il);
            cb(cur, "attn_norm", il);

            // Q, K, V projections
            ggml_tensor * Qcur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wq, cur);
            cb(Qcur, "Qcur", il);

            ggml_tensor * Kcur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wk, cur);
            cb(Kcur, "Kcur", il);

            ggml_tensor * Vcur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wv, cur);
            cb(Vcur, "Vcur", il);

            Qcur = llm_build_norm(ctx0, Qcur, hparams, model.layers[il].attn_q_norm, NULL,
                    LLM_NORM_RMS, cb, il);
            cb(Qcur, "Qcur_normed", il);

            Kcur = llm_build_norm(ctx0, Kcur, hparams, model.layers[il].attn_k_norm, NULL,
                    LLM_NORM_RMS, cb, il);
            cb(Kcur, "Kcur_normed", il);

            // reshape for multi-head
            Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens);
            Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);

            // apply RoPE
            Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow);
            Kcur = ggml_rope_ext(ctx0, Kcur, inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow);
            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            cur = llm_build_kv(ctx0, lctx, kv_self, gf,
                    model.layers[il].wo, NULL,
                    Kcur, Vcur, Qcur, KQ_mask,
                    n_tokens, kv_head, n_kv,
                    1.0f / sqrtf(float(n_embd_head)), cb, il);

            if (il == n_layer - 1 && inp_out_ids) {
                cur = ggml_get_rows(ctx0, cur, inp_out_ids);
                inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
            }

            cur = ggml_add(ctx0, cur, inpSA);
            cb(cur, "ffn_inp", il);
        }

        cur = llm_build_std_moe_ffn(ctx0, lctx, model.layers[il].ffn_norm, cur,
                model.layers[il].ffn_gate_inp,  nullptr,
                model.layers[il].ffn_up_exps,   nullptr,
                model.layers[il].ffn_gate_exps, nullptr,
                model.layers[il].ffn_down_exps, nullptr,
                model.layers[il].ffn_exp_probs_b,
                nullptr,  nullptr, nullptr,  nullptr, nullptr,  nullptr, // no shared experts
                n_expert, n_expert_used,
                LLM_FFN_SILU, true, false, 0.0f,
                (llm_expert_gating_func_type)hparams.expert_gating_func,
                LLM_FFN_SILU, cb, il, gf, true, model.layers[il].ffn_up_gate_exps);

        cur = lctx.cvec.apply_to(ctx0, cur, il);
        cb(cur, "l_out", il);

        // input for next layer
        inpL = cur;
    }

    cur = build_output(lctx, ctx0, inpL, model.output, model.output_norm, cb);
    cb(cur, "result_output", -1);

    ggml_build_forward_expand(gf, cur);
    return gf;
}

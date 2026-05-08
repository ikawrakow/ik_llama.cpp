#include "../llama-build-context.h"
#include "../llama-model.h"
#include "../llama-context.h"

static int gemma4_mtp_target_kv_layer(const llama_hparams & mtp_hparams, const llama_hparams & target_hparams, int mtp_il) {
    GGML_ASSERT(mtp_il >= 0 && mtp_il < (int) mtp_hparams.n_layer);

    const bool is_sliding = mtp_hparams.swa_layers[mtp_il] != 0;
    const int target_n_kv_layer = target_hparams.n_layer_kv_from_start > 0
        ? std::min<int>((int) target_hparams.n_layer, target_hparams.n_layer_kv_from_start)
        : (int) target_hparams.n_layer;

    int target_il = target_n_kv_layer - 1;
    for (; target_il >= 0; --target_il) {
        if ((target_hparams.swa_layers[target_il] != 0) == is_sliding) {
            break;
        }
    }

    GGML_ASSERT(target_il >= 0 && "Gemma4 MTP could not find a matching target KV layer");
    return target_il;
}

static void gemma4_mtp_prepare_frozen_kv_views(
        ggml_context * ctx0,
        llama_context & lctx,
        const llama_kv_cache & target_kv,
        int assistant_il,
        int target_il,
        int32_t target_n_kv,
        ggml_tensor ** frozen_k,
        ggml_tensor ** frozen_v,
        const llm_build_cb & cb) {
    if (*frozen_k || *frozen_v) {
        GGML_ASSERT(*frozen_k && *frozen_v);
        return;
    }

    if (!lctx.cparams.flash_attn) {
        return;
    }

    GGML_ASSERT(target_il >= 0 && target_il < (int) target_kv.k_l.size() && target_il < (int) target_kv.v_l.size());

    ggml_tensor * k_cache = target_kv.k_l[target_il];
    ggml_tensor * v_cache = target_kv.v_l[target_il];
    if (!k_cache || !v_cache || !k_cache->extra || !v_cache->extra) {
        return;
    }

    auto * split_k = (ggml_split_tensor_t *) k_cache->extra;
    auto * split_v = (ggml_split_tensor_t *) v_cache->extra;

    GGML_ASSERT(split_k && split_v);
    GGML_ASSERT(split_k->n_device == split_v->n_device);

    const llama_hparams & assistant_hparams = lctx.model.hparams;
    const int64_t n_embd_head_k = assistant_hparams.n_embd_head_k(assistant_il);
    const int64_t n_embd_head_v = assistant_hparams.n_embd_head_v(assistant_il);

    std::vector<ggml_tensor *> k_parts;
    std::vector<ggml_tensor *> v_parts;
    k_parts.reserve(split_k->n_device);
    v_parts.reserve(split_v->n_device);

    for (int id = 0; id < split_k->n_device; ++id) {
        ggml_tensor * split_kl = split_k->splits[id];
        ggml_tensor * split_vl = split_v->splits[id];

        GGML_ASSERT((split_kl && split_vl) || (!split_kl && !split_vl));
        if (!split_kl) {
            continue;
        }

        GGML_ASSERT(target_kv.size > 0);
        GGML_ASSERT(split_kl->ne[1] % target_kv.size == 0);

        const int64_t split_n_head_kv = split_kl->ne[1] / target_kv.size;

        ggml_tensor * k_part = ggml_view_3d(ctx0, split_kl,
                n_embd_head_k, target_n_kv, split_n_head_kv,
                ggml_row_size(split_kl->type, n_embd_head_k) * split_n_head_kv,
                ggml_row_size(split_kl->type, n_embd_head_k),
                0);
        if (k_part->type != GGML_TYPE_F32) {
            k_part = ggml_cast(ctx0, k_part, GGML_TYPE_F32);
        }
        cb(k_part, "mtp_frozen_k_split", 1000 * (assistant_il + 1) + id);

        ggml_tensor * v_part = ggml_view_3d(ctx0, split_vl,
                n_embd_head_v, target_n_kv, split_n_head_kv,
            ggml_row_size(split_vl->type, split_n_head_kv * n_embd_head_v),
                ggml_row_size(split_vl->type, n_embd_head_v),
                0);
        if (v_part->type != GGML_TYPE_F32) {
            v_part = ggml_cast(ctx0, v_part, GGML_TYPE_F32);
        }
        cb(v_part, "mtp_frozen_v_split", 1000 * (assistant_il + 1) + id);

        k_parts.push_back(k_part);
        v_parts.push_back(v_part);
    }

    GGML_ASSERT(!k_parts.empty() && k_parts.size() == v_parts.size());

    ggml_tensor * k_full = k_parts[0];
    ggml_tensor * v_full = v_parts[0];
    for (size_t i = 1; i < k_parts.size(); ++i) {
        k_full = ggml_concat(ctx0, k_full, k_parts[i], 2);
        v_full = ggml_concat(ctx0, v_full, v_parts[i], 2);
    }

    if (k_full->type != GGML_TYPE_F16) {
        k_full = ggml_cast(ctx0, k_full, GGML_TYPE_F16);
    }
    if (v_full->type != GGML_TYPE_F16) {
        v_full = ggml_cast(ctx0, v_full, GGML_TYPE_F16);
    }

    cb(k_full, "mtp_frozen_k", assistant_il);
    cb(v_full, "mtp_frozen_v", assistant_il);

    *frozen_k = k_full;
    *frozen_v = v_full;
}

static ggml_cgraph * build_gemma4_graph_parallel(llm_build_context & llm, llama_context & lctx, ggml_context * ctx0,
        ggml_tensor * inpL, ggml_tensor * inp_pos, ggml_tensor * inp_out_ids,
        ggml_tensor * KQ_mask, ggml_tensor * KQ_mask_swa, int n_tokens,  const llm_build_cb & cb) {
    auto & model   = lctx.model;
    auto & hparams = model.hparams;
    auto & cparams = lctx.cparams;
    auto & kv_self = lctx.kv_self;
    int n_device = model.splits.size();
    GGML_ASSERT(n_device > 1);
    GGML_ASSERT(cparams.flash_attn);
    auto gf = ggml_new_graph_custom(ctx0, model.max_nodes(n_tokens), false);

    bool is_moe = hparams.n_expert > 0;

    std::vector<ggml_tensor *> sa_inp(n_device, nullptr);
    std::vector<ggml_tensor *> sa_out(n_device, nullptr);
    std::vector<ggml_tensor *> ffn_inp(n_device, nullptr);
    std::vector<ggml_tensor *> ffn_out(n_device, nullptr);
    std::vector<ggml_tensor *> ffn_out_moe;
    if (is_moe) {
        ffn_out_moe.resize(n_device, nullptr);
    }

    ggml_tensor * inpL_moe = nullptr;

    for (int il = 0; il < hparams.n_layer; ++il) {
        auto & l = model.layers[il];
        const bool is_sliding    = hparams.swa_layers[il] ? true : false;
        const float freq_base_l  = is_sliding ? hparams.rope_freq_base_train_swa  : cparams.rope_freq_base;
        const float freq_scale_l = is_sliding ? hparams.rope_freq_scale_train_swa : cparams.rope_freq_scale;
        const int   n_rot_l      = is_sliding ? hparams.n_rot_swa : hparams.n_rot;
        const int   n_swa        = is_sliding ? hparams.n_swa : 0;

        struct ggml_tensor * KQ_mask_l = is_sliding ? KQ_mask_swa : KQ_mask;

        auto freq_factors = !is_sliding ? model.layers[il].rope_freqs : nullptr;
        if (freq_factors) {
            GGML_ASSERT(freq_factors->extra);
        }

        auto wq = (const ggml_split_tensor_t *)l.wq->extra;
        auto wk = (const ggml_split_tensor_t *)l.wk->extra;
        auto wv = l.wv ? (const ggml_split_tensor_t *)l.wv->extra : nullptr;
        auto wo = (const ggml_split_tensor_t *)l.wo->extra;
        GGML_ASSERT(wq && wk && wo);

        auto q_norm = (const ggml_split_tensor_t *)l.attn_q_norm->extra;
        auto k_norm = (const ggml_split_tensor_t *)l.attn_k_norm->extra;
        GGML_ASSERT(q_norm && k_norm);

        auto kl = (ggml_split_tensor_t *)kv_self.k_l[il]->extra;
        auto vl = (ggml_split_tensor_t *)kv_self.v_l[il]->extra;
        GGML_ASSERT(kl && vl);

        for (int id = 0; id < n_device; ++id) {
            GGML_ASSERT((wq->splits[id] && wk->splits[id] && (!wv || wv->splits[id]) && wo->splits[id]) ||
                    (!wq->splits[id] && !wk->splits[id] && (!wv || !wv->splits[id]) && !wo->splits[id]));
            if (!wq->splits[id]) {
                sa_inp[id] = sa_out[id] = nullptr;
                continue;
            }
            GGML_ASSERT(kl->splits[id] && vl->splits[id]);

            int il_cb = 1000*(il + 1) + id;

            if (il == 0) {
                sa_inp[id] = inpL;
            } else {
                GGML_ASSERT(inpL->op == GGML_OP_REDUCE);
                auto cur = llm_build_context::get_input_tensor_sm_graph(ctx0, inpL, id);
                if (is_moe) {
                    GGML_ASSERT(inpL_moe && inpL_moe->op == GGML_OP_REDUCE);
                    auto cur_moe = llm_build_context::get_input_tensor_sm_graph(ctx0, inpL_moe, id);
                    auto post_norm_1 = (ggml_split_tensor_t *)model.layers[il-1].ffn_post_norm_1->extra;
                    auto post_norm_2 = (ggml_split_tensor_t *)model.layers[il-1].ffn_post_norm_2->extra;
                    cur = ggml_fused_rms_rms_add(ctx0, cur, post_norm_1->splits[id], cur_moe, post_norm_2->splits[id], hparams.f_norm_rms_eps);
                    cb(cur, "ffn_combined", il_cb);
                }
                cur = llm_build_context::do_split_norm(ctx0, cur, model.layers[il-1].ffn_post_norm, hparams, cb, id, il_cb, false);
                cb(cur, "ffn_normed", il_cb);
                auto add = ffn_inp[id];
                if (!add) {
                    for (int j = 0; j < n_device; ++j) {
                        if (ffn_inp[j]) {
                            add = ffn_inp[j]; break;
                        }
                    }
                    GGML_ASSERT(add);
                }
                sa_inp[id] = ggml_add(ctx0, cur, add);
                cb(sa_inp[id], "sa_inp", il_cb);
                if (model.layers[il-1].out_scale) {
                    auto scale = (const ggml_split_tensor_t *)model.layers[il-1].out_scale->extra;
                    sa_inp[id] = ggml_mul(ctx0, sa_inp[id], scale->splits[id]);
                    cb(sa_inp[id], "sa_inp_scaled", il_cb);
                }
            }
            auto cur = llm_build_context::do_split_norm(ctx0, sa_inp[id], model.layers[il].attn_norm, hparams, cb, id, il_cb, false);
            cb(cur, "sa_inp_normed", il_cb);
            auto Qcur = llm.llm_build_lora_mm(lctx, ctx0, wq->splits[id], cur);
            cb(Qcur, "Qcur", il_cb);
            auto Kcur = llm.llm_build_lora_mm(lctx, ctx0, wk->splits[id], cur);
            cb(Kcur, "Kcur", il_cb);
            ggml_tensor * Vcur = nullptr;
            if (wv) {
                Vcur = llm.llm_build_lora_mm(lctx, ctx0, wv->splits[id], cur);
                cb(Vcur, "Vcur", il_cb);
            }
            ggml_build_forward_expand(gf, Qcur);
            ggml_build_forward_expand(gf, Kcur);
            if (Vcur) {
                ggml_build_forward_expand(gf, Vcur);
                Vcur = ggml_reshape_3d(ctx0, Vcur, hparams.n_embd_head_v(il), Vcur->ne[0]/hparams.n_embd_head_v(il), n_tokens);
                cb(Vcur, "Vcur", il_cb);
            }
            Qcur = ggml_reshape_3d(ctx0, Qcur, hparams.n_embd_head_k(il), Qcur->ne[0]/hparams.n_embd_head_k(il), n_tokens);
            cb(Qcur, "Qcur", il_cb);
            Kcur = ggml_reshape_3d(ctx0, Kcur, hparams.n_embd_head_k(il), Kcur->ne[0]/hparams.n_embd_head_k(il), n_tokens);
            cb(Kcur, "Kcur", il_cb);
            if (!Vcur) {
                Vcur = Kcur;
            }
            Qcur = llm.llm_build_norm(ctx0, Qcur, hparams, q_norm->splits[id], NULL, LLM_NORM_RMS, cb, il_cb);
            cb(Qcur, "Qcur_n", il_cb);
            Kcur = llm.llm_build_norm(ctx0, Kcur, hparams, k_norm->splits[id], NULL, LLM_NORM_RMS, cb, il_cb);
            cb(Kcur, "Kcur_n", il_cb);
            Vcur = ggml_rms_norm(ctx0, Vcur, hparams.f_norm_rms_eps);
            cb(Vcur, "Vcur_n", il_cb);

            auto rope_factors = freq_factors ? ((const ggml_split_tensor_t *)freq_factors->extra)->splits[id] : nullptr;
            Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, rope_factors, n_rot_l, llm.rope_type, llm.n_ctx_orig, freq_base_l, freq_scale_l,
                                llm.ext_factor, llm.attn_factor, llm.beta_fast, llm.beta_slow);
            Kcur = ggml_rope_ext(ctx0, Kcur, inp_pos, rope_factors, n_rot_l, llm.rope_type, llm.n_ctx_orig, freq_base_l, freq_scale_l,
                                llm.ext_factor, llm.attn_factor, llm.beta_fast, llm.beta_slow);
            cb(Qcur, "Qcur_rope", il_cb);
            cb(Kcur, "Kcur_rope", il_cb);

            const int64_t n_embd_head_k = hparams.n_embd_head_k(il);
            const int64_t n_embd_head_v = hparams.n_embd_head_v(il);
            const int64_t n_head_kv     = wk->splits[id]->ne[1] / n_embd_head_k;

            if (cparams.k_cache_hadamard) {
                Qcur = ggml_hadamard(ctx0, Qcur, n_embd_head_k);
                Kcur = ggml_hadamard(ctx0, Kcur, n_embd_head_k);
                cb(Qcur, "Qcur_h", il_cb);
                cb(Kcur, "Kcur_h", il_cb);
            }
            if (cparams.v_cache_hadamard) {
                Vcur = ggml_hadamard(ctx0, Vcur, n_embd_head_v);
                cb(Vcur, "Vcur_h", il_cb);
            }

            GGML_ASSERT(kv_self.size == cparams.n_ctx);

            ggml_build_forward_expand(gf, Qcur);
            ggml_build_forward_expand(gf, Kcur);
            ggml_build_forward_expand(gf, Vcur);

            auto idx = 2*n_device*il + 2*id;
            GGML_ASSERT(idx+1 < (int)lctx.cache_copies.size());
            auto k_row_size = ggml_row_size(kl->splits[id]->type, n_embd_head_k);
            ggml_tensor * k_cache_view = ggml_view_2d(ctx0, kl->splits[id], n_embd_head_k, n_tokens*n_head_kv,
                    k_row_size, k_row_size*n_head_kv*llm.kv_head);

            lctx.cache_copies[idx+0].cpy  = ggml_cpy(ctx0, Kcur, k_cache_view);
            cb(lctx.cache_copies[idx+0].cpy, "k_cache", il_cb);
            lctx.cache_copies[idx+0].step = k_row_size*n_head_kv;
            ggml_build_forward_expand(gf, lctx.cache_copies[idx+0].cpy);

            if (!wv) {
                wv = wk;
            }
            auto v_cache_view = ggml_view_1d(ctx0, vl->splits[id], n_tokens*wv->splits[id]->ne[1],
                    llm.kv_head*ggml_row_size(vl->splits[id]->type, wv->splits[id]->ne[1]));
            lctx.cache_copies[idx+1].step = ggml_row_size(vl->splits[id]->type, wv->splits[id]->ne[1]);
            lctx.cache_copies[idx+1].cpy  = ggml_cpy(ctx0, Vcur, v_cache_view);
            cb(lctx.cache_copies[idx+1].cpy, "v_cache", il_cb);
            ggml_build_forward_expand(gf, lctx.cache_copies[idx+1].cpy);

            auto split_kl = kl->splits[id];
            auto split_vl = vl->splits[id];

            auto q = ggml_permute(ctx0, Qcur, 0, 2, 1, 3);
            cb(q, "q", il_cb);
            auto k = ggml_view_3d(ctx0, split_kl, n_embd_head_k, llm.n_kv, n_head_kv,
                    ggml_row_size(split_kl->type, n_embd_head_k)*n_head_kv,
                    ggml_row_size(split_kl->type, n_embd_head_k), 0);
            cb(k, "k", il_cb);
            auto v = ggml_view_3d(ctx0, split_vl, n_embd_head_v, llm.n_kv, n_head_kv,
                    ggml_row_size(split_vl->type, wv->splits[id]->ne[1]),
                    ggml_row_size(split_vl->type, n_embd_head_v), 0);
            cb(v, "v", il_cb);

            cur = ggml_flash_attn_ext(ctx0, q, k, v, KQ_mask_l, hparams.f_attention_scale, hparams.f_max_alibi_bias,
                    hparams.attn_soft_cap ? hparams.f_attn_logit_softcapping : 0.0f);
            cb(cur, "fa", il_cb);
            cur->op_params[4] = n_swa;
            if (cparams.v_cache_hadamard) {
                cur = ggml_hadamard(ctx0, cur, n_embd_head_v);
                cb(cur, "fa_h", il_cb);
            }
            cur = ggml_reshape_2d(ctx0, cur, wo->splits[id]->ne[0], n_tokens);
            if (il == hparams.n_layer-1 && inp_out_ids) {
                cur = ggml_get_rows(ctx0, cur, inp_out_ids);
                sa_inp[id] = ggml_get_rows(ctx0, sa_inp[id], inp_out_ids);
            }
            cur = llm.llm_build_lora_mm(lctx, ctx0, wo->splits[id], cur);
            cb(cur, "qkv", il_cb);
            if (cur->ne[1] > 32 && cparams.reduce_type != GGML_TYPE_F32) {
                cur = ggml_cast(ctx0, cur, cparams.reduce_type);
                cb(cur, "qkv_cast", il_cb);
            }
            ggml_build_forward_expand(gf, cur);
            sa_out[id] = cur;

        }

        auto last_ffn_inp = ggml_reduce(ctx0, sa_out.data(), n_device, GGML_OP_ADD);
        ggml_build_forward_expand(gf, last_ffn_inp);
        cb(last_ffn_inp, "sa_reduce", il);

        auto ffn_up   = (const ggml_split_tensor_t *)l.ffn_up->extra;
        auto ffn_gate = (const ggml_split_tensor_t *)l.ffn_gate->extra;
        auto ffn_down = (const ggml_split_tensor_t *)l.ffn_down->extra;
        GGML_ASSERT(ffn_up && ffn_gate && ffn_down);

        for (int id = 0; id < n_device; ++id) {
            GGML_ASSERT((ffn_up->splits[id] && ffn_gate->splits[id] && ffn_down->splits[id]) ||
                    (!ffn_up->splits[id] && !ffn_gate->splits[id] && !ffn_down->splits[id]));
            if (!ffn_up->splits[id]) {
                ffn_inp[id] = ffn_out[id] = nullptr;
                if (is_moe) {
                    ffn_out_moe[id] = nullptr;
                }
                continue;
            }
            int il_cb = 1000*(il + 1) + id;

            GGML_ASSERT(last_ffn_inp && last_ffn_inp->op == GGML_OP_REDUCE);
            auto cur = llm_build_context::get_input_tensor_sm_graph(ctx0, last_ffn_inp, id);
            cur = llm_build_context::do_split_norm(ctx0, cur, model.layers[il].attn_post_norm, hparams, cb, id, il_cb, false);
            cb(cur, "sa_post", il_cb);
            auto add = sa_inp[id];
            if (!add) {
                for (int j = 0; j < n_device; ++j) {
                    if (sa_inp[j]) {
                        add = sa_inp[j]; break;
                    }
                }
            }
            ffn_inp[id] = ggml_add(ctx0, cur, add);
            cb(ffn_inp[id], "ffn_inp", il_cb);
            cur = llm_build_context::do_split_norm(ctx0, ffn_inp[id], model.layers[il].ffn_norm, hparams, cb, id, il_cb, false);
            cb(cur, "ffn_inp_normed", il_cb);
            cur = llm.llm_build_ffn(ctx0, lctx, nullptr, cur,
                    ffn_up->splits[id], nullptr, nullptr,
                    ffn_gate->splits[id], nullptr, nullptr,
                    ffn_down->splits[id], nullptr, nullptr,
                    nullptr,
                    LLM_FFN_GELU, LLM_FFN_PAR, cb, il, gf, false, false, nullptr, nullptr);
            cb(cur, "ffn", il_cb);
            if (cur->ne[1] > 32 && cparams.reduce_type != GGML_TYPE_F32) {
                cur = ggml_cast(ctx0, cur, cparams.reduce_type);
                cb(cur, "ffn_cast", il_cb);
            }
            ggml_build_forward_expand(gf, cur);
            ffn_out[id] = cur;

            if (is_moe) {
                cur = llm_build_context::do_split_norm(ctx0, ffn_inp[id], model.layers[il].ffn_pre_norm_2, hparams, cb, id, il_cb, false);
                cb(cur, "ffn_moe_inp", il_cb);
                auto tmp = ggml_fused_rms_norm(ctx0, ffn_inp[id],
                        ((const ggml_split_tensor_t *)model.layers[il].ffn_gate_inp_s->extra)->splits[id], hparams.f_norm_rms_eps);
                cb(tmp, "tmp", il_cb);
                auto logits = llm.llm_build_lora_mm(lctx, ctx0, ((const ggml_split_tensor_t *)model.layers[il].ffn_gate_inp->extra)->splits[id], tmp);
                cb(logits, "logits", il_cb);
                ggml_build_forward_expand(gf, logits);

                auto moe = llm. llm_build_moe_ffn(ctx0, lctx, cur,
                        nullptr, nullptr, nullptr,
                        ((const ggml_split_tensor_t *)model.layers[il].ffn_down_exps->extra)->splits[id], nullptr,
                        llm.n_expert, llm.n_expert_used,
                        LLM_FFN_GELU, true, false, 0.0f,
                        LLM_EXPERT_GATING_FUNC_SOFTMAX,
                        cb, il, gf, false,
                        ((const ggml_split_tensor_t *)model.layers[il].ffn_up_gate_exps->extra)->splits[id],
                        nullptr, logits, ((const ggml_split_tensor_t *)model.layers[il].ffn_down_exps_s->extra)->splits[id]);
                if (moe->ne[1] > 32 && cparams.reduce_type != GGML_TYPE_F32) {
                    moe = ggml_cast(ctx0, moe, cparams.reduce_type);
                    cb(moe, "ffn_moe_cast", il_cb);
                }
                ggml_build_forward_expand(gf, moe);
                ffn_out_moe[id] = moe;
            }

        }

        inpL = ggml_reduce(ctx0, ffn_out.data(), n_device, GGML_OP_ADD);
        cb(inpL, "ffn_reduce", il);
        ggml_build_forward_expand(gf, inpL);

        if (is_moe) {
            inpL_moe = ggml_reduce(ctx0, ffn_out_moe.data(), n_device, GGML_OP_ADD);
            cb(inpL_moe, "ffn_moe_reduce", il);
            ggml_build_forward_expand(gf, inpL_moe);
        }

    }

    int idx = lctx.model.default_layer_device[lctx.model.hparams.n_layer];
    int idx_out = ggml_backend_sched_get_backend_idx(lctx.sched, lctx.model.output->buffer);
    if (idx_out >= 0) idx = idx_out;
    auto cur = inpL->src[idx];
    if (!cur) {
        cur = inpL->view_src;
    }

    auto post_norm   = (const ggml_split_tensor_t *)model.layers[hparams.n_layer-1].ffn_post_norm->extra;
    if (is_moe) {
        auto cur_moe = inpL_moe->src[idx];
        if (!cur_moe) {
            cur_moe = inpL_moe->view_src;
        }
        auto post_norm_1 = (const ggml_split_tensor_t *)model.layers[hparams.n_layer-1].ffn_post_norm_1->extra;
        auto post_norm_2 = (const ggml_split_tensor_t *)model.layers[hparams.n_layer-1].ffn_post_norm_2->extra;
        cur = ggml_fused_rms_rms_add(ctx0, cur, post_norm_1->splits[idx], cur_moe, post_norm_2->splits[idx], hparams.f_norm_rms_eps);
        cur->op_params[GGML_MAX_OP_PARAMS / sizeof(int32_t) - 1] = 0xff;
        ggml_build_forward_expand(gf, cur);
        cb(cur, "ffn_combined", hparams.n_layer-1);
    }
    cur = llm.llm_build_norm(ctx0, cur, hparams, post_norm->splits[idx], NULL, LLM_NORM_RMS, cb, -1);
    cb(cur, "ffn_normed", hparams.n_layer-1);
    auto add = ffn_inp[idx];
    if (!add) {
        for (int j = 0; j < n_device; ++j) {
            if (ffn_inp[j]) {
                add = ffn_inp[j]; break;
            }
        }
    }
    cur = ggml_add(ctx0, cur, add);
    cb(cur, "ffn_out", hparams.n_layer-1);

    if (model.layers[hparams.n_layer-1].out_scale) {
        auto scale = (const ggml_split_tensor_t *)model.layers[hparams.n_layer-1].out_scale->extra;
        cur = ggml_mul(ctx0, cur, scale->splits[idx]);
        cb(cur, "ffn_out_scaled", hparams.n_layer-1);
    }

    cur = llm_build_context::build_output(lctx, ctx0, cur, model.output, model.output_norm, cb);
    cb(cur, "almost_result", -1);
    if (hparams.f_final_logit_softcapping > 0) {
        cur = ggml_softcap(ctx0, cur, 1.0f / hparams.f_final_logit_softcapping, hparams.f_final_logit_softcapping);
    }
    cb(cur, "result_output", -1);

    ggml_build_forward_expand(gf, cur);

    return gf;
}

ggml_cgraph * llm_build_context::build_gemma4_mtp() {
    ggml_cgraph * gf = ggml_new_graph_custom(ctx0, model.max_nodes(n_tokens), false);

    const int64_t n_embd          = hparams.n_embd;
    const int64_t n_vocab         = hparams.n_vocab;
    const int64_t n_backbone      = hparams.mtp_backbone_n_embd;
    const int32_t n_layer         = hparams.n_layer;
    const bool    has_target_ctx  = lctx.mtp_target_ctx != nullptr;

    GGML_ASSERT(n_backbone > 0);

    lctx.inp_tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, batch.n_tokens);
    cb(lctx.inp_tokens, "inp_tokens", -1);
    ggml_set_input(lctx.inp_tokens);

    ggml_tensor * hidden_state = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_backbone, n_tokens);
    ggml_set_name(hidden_state, "inp_mtp_states");
    ggml_set_input(hidden_state);
    lctx.inp_mtp_states = hidden_state;

    if (!has_target_ctx || !batch.token) {
        ggml_tensor * cur = ggml_view_2d(ctx0, hidden_state, n_embd, n_tokens,
                ggml_row_size(hidden_state->type, n_backbone), 0);
        cb(cur, "mtp_init_hidden_view", -1);

        ggml_tensor * mtp_embd = ggml_dup(ctx0, hidden_state);
        cb(mtp_embd, "result_mtp_embd", -1);
        ggml_build_forward_expand(gf, mtp_embd);

        ggml_tensor * logits = build_output(lctx, ctx0, cur, model.output, model.output_norm, cb);
        cb(logits, "result_output", -1);
        ggml_build_forward_expand(gf, logits);

        GGML_UNUSED(n_vocab);
        return gf;
    }

    const llama_model   & target_model   = lctx.mtp_target_ctx->model;
    const llama_hparams & target_hparams = target_model.hparams;
    const llama_cparams & target_cparams = lctx.mtp_target_ctx->cparams;
    const llama_kv_cache & target_kv     = lctx.mtp_target_ctx->kv_self;

    GGML_ASSERT(n_tokens <= target_kv.n);

    ggml_tensor * inp_pos = build_inp_pos();

    ggml_tensor * token_embd = ggml_get_rows(ctx0, target_model.tok_embd, lctx.inp_tokens);
    cb(token_embd, "inp_embd_target", -1);
    token_embd = ggml_scale(ctx0, token_embd, std::sqrt(float(n_backbone)));
    cb(token_embd, "inp_embd_scaled", -1);

    ggml_tensor * cur = ggml_concat(ctx0, token_embd, hidden_state, 0);
    cb(cur, "inp_mtp_combined", -1);
    cur = llm_build_lora_mm(lctx, ctx0, model.mtp_pre_proj, cur);
    cb(cur, "mtp_pre_proj", -1);

    const int32_t target_n_kv = target_kv.n;
    const int32_t target_kv_head = target_kv.head;

    ggml_tensor * KQ_mask = nullptr;
    ggml_tensor * KQ_mask_swa = nullptr;
    ggml_tensor * frozen_k_swa = nullptr;
    ggml_tensor * frozen_v_swa = nullptr;
    ggml_tensor * frozen_k_full = nullptr;
    ggml_tensor * frozen_v_full = nullptr;
    {
        const int64_t n_mask_tokens = GGML_PAD(n_tokens, GGML_KQ_MASK_PAD);
        lctx.inp_KQ_mask = ggml_new_tensor_2d(ctx0, flash_attn ? GGML_TYPE_F16 : GGML_TYPE_F32, target_n_kv, n_mask_tokens);
        cb(lctx.inp_KQ_mask, "KQ_mask", -1);
        ggml_set_input(lctx.inp_KQ_mask);
        KQ_mask = lctx.inp_KQ_mask;

        if (target_hparams.n_swa > 0) {
            lctx.inp_KQ_mask_swa = ggml_new_tensor_2d(ctx0, flash_attn ? GGML_TYPE_F16 : GGML_TYPE_F32, target_n_kv, n_mask_tokens);
            cb(lctx.inp_KQ_mask_swa, "KQ_mask_swa", -1);
            ggml_set_input(lctx.inp_KQ_mask_swa);
            KQ_mask_swa = lctx.inp_KQ_mask_swa;
        }
    }

    for (int il = 0; il < n_layer; ++il) {
        ggml_tensor * inpL = cur;

        const bool is_sliding    = hparams.swa_layers[il] ? true : false;
        const float freq_base_l  = is_sliding ? target_hparams.rope_freq_base_train_swa  : target_cparams.rope_freq_base;
        const float freq_scale_l = is_sliding ? target_hparams.rope_freq_scale_train_swa : target_cparams.rope_freq_scale;
        const int   n_rot_l      = is_sliding ? target_hparams.n_rot_swa : target_hparams.n_rot;
        const int   n_swa        = is_sliding ? target_hparams.n_swa : 0;
        const int   n_embd_head  = hparams.n_embd_head_k(il);
        const int   n_head       = hparams.n_head(il);
        ggml_tensor * KQ_mask_l  = is_sliding ? KQ_mask_swa : KQ_mask;

        cur = llm_build_norm(ctx0, inpL, hparams, model.layers[il].attn_norm, nullptr, LLM_NORM_RMS, cb, il);
        cb(cur, "attn_norm", il);

        ggml_tensor * Qcur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wq, cur);
        cb(Qcur, "Qcur", il);
        Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens);
        Qcur = llm_build_norm(ctx0, Qcur, hparams, model.layers[il].attn_q_norm, nullptr, LLM_NORM_RMS, cb, il);
        cb(Qcur, "Qcur_normed", il);
        Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, nullptr, n_rot_l, rope_type, n_ctx_orig, freq_base_l, freq_scale_l,
                ext_factor, attn_factor, beta_fast, beta_slow);
        cb(Qcur, "Qcur_rope", il);

        const int target_il = gemma4_mtp_target_kv_layer(hparams, target_hparams, il);
        ggml_tensor *& frozen_k = is_sliding ? frozen_k_swa : frozen_k_full;
        ggml_tensor *& frozen_v = is_sliding ? frozen_v_swa : frozen_v_full;
        gemma4_mtp_prepare_frozen_kv_views(ctx0, lctx, target_kv, il, target_il, target_n_kv, &frozen_k, &frozen_v, cb);
        cur = llm_build_kv(ctx0, lctx, target_kv, gf, model.layers[il].wo, model.layers[il].bo,
            nullptr, nullptr, Qcur, KQ_mask_l, n_tokens, target_kv_head, target_n_kv, hparams.f_attention_scale, cb, il, nullptr, n_swa, target_il,
            &frozen_k, &frozen_v);

        cur = llm_build_norm(ctx0, cur, hparams, model.layers[il].attn_post_norm, nullptr, LLM_NORM_RMS, cb, il);
        cb(cur, "attn_post_norm", il);
        cur = ggml_add(ctx0, cur, inpL);
        cb(cur, "attn_out", il);

        ggml_tensor * ffn = llm_build_ffn(ctx0, lctx, model.layers[il].ffn_norm, cur,
                model.layers[il].ffn_up,   nullptr, nullptr,
                model.layers[il].ffn_gate, nullptr, nullptr,
                model.layers[il].ffn_down, nullptr, nullptr,
                nullptr,
                LLM_FFN_GELU, LLM_FFN_PAR, cb, il, gf, true, false, nullptr, model.layers[il].ffn_post_norm);
        cb(ffn, "ffn_out", il);

        cur = ffn;
        if (model.layers[il].out_scale) {
            cur = ggml_mul(ctx0, cur, model.layers[il].out_scale);
            cb(cur, "out_scaled", il);
        }
        cur = lctx.cvec.apply_to(ctx0, cur, il);
        cb(cur, "l_out", il);
    }

    ggml_tensor * mtp_embd = llm_build_lora_mm(lctx, ctx0, model.mtp_post_proj, cur);
    cb(mtp_embd, "result_mtp_embd", -1);
    ggml_build_forward_expand(gf, mtp_embd);

    ggml_tensor * logits;
    // E2B/E4B: The centroid/token-ordering tensors are kept in the GGUF for future use but
    // not required for correct inference — the full-vocab matmul against the tied output
    // weight still yields valid per-token logits.
    {
        logits = build_output(lctx, ctx0, cur, model.output, model.output_norm, cb);
        cb(logits, "result_output", -1);
    }
    ggml_build_forward_expand(gf, logits);

    GGML_UNUSED(n_embd);
    GGML_UNUSED(n_vocab);

    return gf;
}

static ggml_tensor * gemma4_project_per_layer_inputs(ggml_context * ctx0, const llama_model & model, const llm_build_cb & cb,
        int n_embd, int n_embd_per_layer, int n_layer, int n_tokens,
        ggml_tensor * inputs_embeds, ggml_tensor * inp_per_layer) {
    const float per_layer_projection_scale = 1.0f / sqrtf((float) n_embd);
    const float per_layer_input_scale      = 1.0f / sqrtf(2.0f);

    ggml_tensor * per_layer_proj = ggml_mul_mat(ctx0, model.per_layer_model_proj, inputs_embeds);
    per_layer_proj               = ggml_scale(ctx0, per_layer_proj, per_layer_projection_scale);
    per_layer_proj               = ggml_reshape_3d(ctx0, per_layer_proj, n_embd_per_layer, n_layer, n_tokens);
    per_layer_proj               = llm_build_context::llm_build_norm(ctx0, per_layer_proj, model.hparams,
            model.per_layer_proj_norm, nullptr, LLM_NORM_RMS, cb, -1);  // [n_embd_per_layer, n_layer, n_tokens]
    cb(per_layer_proj, "per_layer_proj", -1);

    inp_per_layer = ggml_add(ctx0, per_layer_proj, inp_per_layer);
    inp_per_layer = ggml_scale(ctx0, inp_per_layer, per_layer_input_scale);
    cb(inp_per_layer, "inp_per_layer", -1);

    // permute to shape: [n_embd_per_layer, n_tokens, n_layer]
    inp_per_layer = ggml_cont(ctx0, ggml_permute(ctx0, inp_per_layer, 0, 2, 1, 3));
    return inp_per_layer;
}

ggml_cgraph * llm_build_context::build_gemma4() {

    struct ggml_tensor * cur;
    struct ggml_tensor * inpL;

    inpL = llm_build_inp_embd(ctx0, lctx, hparams, batch, model.tok_embd, cb);
    cb(inpL, "tok_embd", -1);

    // important: do not normalize weights for raw embeddings input (i.e. encoded image emdeddings)
    if (batch.token) {
        inpL = ggml_scale(ctx0, inpL, sqrtf(n_embd));
        cb(inpL, "inp_scaled", -1);
    }

    // inp_pos - contains the positions
    struct ggml_tensor * inp_pos = build_inp_pos();

    // KQ_mask (mask for 1 head, it will be broadcasted to all heads)
    // gemma3 requires different mask for layers using sliding window (SWA)
    struct ggml_tensor * KQ_mask     = build_inp_KQ_mask(true);
    struct ggml_tensor * KQ_mask_swa = build_inp_KQ_mask_swa(true);

    auto inp_out_ids = n_tokens > 1 ? build_inp_out_ids() : nullptr;

    ggml_tensor * inp_per_layer = nullptr;
    if (model.tok_embd_per_layer) {
        if (batch.token) {
            inp_per_layer = ggml_get_rows(ctx0, model.tok_embd_per_layer, lctx.inp_tokens);
            inp_per_layer = ggml_reshape_3d(ctx0, inp_per_layer, hparams.n_embd_per_layer, n_layer, n_tokens);
            inp_per_layer = ggml_scale(ctx0, inp_per_layer, sqrtf((float) hparams.n_embd_per_layer));
            cb(inp_per_layer, "inp_per_layer_selected", -1);
        } else {
            // Vision embedding path: use padding token (ID=0) embedding
            // TODO: verify if this is the correct behavior in transformers implementation
            const int64_t embd_size = model.tok_embd_per_layer->ne[0];  // n_embd_per_layer * n_layer

            // Extract and dequantize padding token embedding (row 0)
            auto padding = ggml_view_1d(ctx0, model.tok_embd_per_layer, embd_size, 0);
            inp_per_layer = ggml_cast(ctx0, padding, GGML_TYPE_F32);

            // Reshape to [n_embd_per_layer, n_layer, 1]
            inp_per_layer = ggml_reshape_3d(ctx0, inp_per_layer, hparams.n_embd_per_layer, n_layer, 1);
            cb(inp_per_layer, "inp_per_layer_vision", -1);
        }
        inp_per_layer = gemma4_project_per_layer_inputs(ctx0, model, cb, n_embd,
                model.hparams.n_embd_per_layer, n_layer, n_tokens, inpL, inp_per_layer);

    }

    if (model.split_mode == LLAMA_SPLIT_MODE_GRAPH) {
        return build_gemma4_graph_parallel(*this, lctx, ctx0, inpL, inp_pos, inp_out_ids,
                                     KQ_mask, KQ_mask_swa, n_tokens,  cb);
    }

    auto gf = ggml_new_graph_custom(ctx0, model.max_nodes(n_tokens), false);

    // "5-to-1 interleaved attention"
    // 5 layers of local attention followed by 1 layer of global attention
    // we do this via swa_layers now
    // static const int sliding_window_pattern = 6;

    for (int il = 0; il < n_layer; ++il) {
        const bool is_sliding    = hparams.swa_layers[il] ? true : false;
        const float freq_base_l  = is_sliding ? hparams.rope_freq_base_train_swa  : cparams.rope_freq_base;
        const float freq_scale_l = is_sliding ? hparams.rope_freq_scale_train_swa : cparams.rope_freq_scale;
        const int   n_rot_l      = is_sliding ? hparams.n_rot_swa : hparams.n_rot;
        const int   n_swa        = is_sliding ? hparams.n_swa : 0;
        const int   n_embd_head  = hparams.n_embd_head_k(il);
        const int   n_head       = hparams.n_head(il);
        const int   n_head_kv    = hparams.n_head_kv(il);

        struct ggml_tensor * KQ_mask_l = is_sliding ? KQ_mask_swa : KQ_mask;

        auto freq_factors = !is_sliding ? model.layers[il].rope_freqs : nullptr;

        ggml_tensor * attn_out;

        if (hparams.has_kv(il) && model.layers[il].wv) {
            attn_out = build_std_attention(gf, model.layers[il].attn_norm, inpL, inp_pos, il == n_layer - 1 ? inp_out_ids : nullptr, freq_factors,
                    KQ_mask_l, nullptr, nullptr, hparams.f_attention_scale, 0.0f, n_swa, il, true, false, true, false, false, model.layers[il].attn_post_norm);
        } else {
            cur = llm_build_norm(ctx0, inpL, hparams, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, cb, il);
            cb(cur, "attn_norm", il);

            ggml_tensor *Qcur, *Kcur = nullptr, *Vcur = nullptr;
            Qcur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wq, cur);
            cb(Qcur, "Qcur", il);
            if (hparams.has_kv(il)) {
                Kcur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wk, cur);
                cb(Kcur, "Kcur", il);
                if (model.layers[il].wv) {
                    Vcur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wv, cur);
                    cb(Vcur, "Vcur", il);
                } else {
                    Vcur = Kcur;
                }
            }
            Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens);
            Qcur = llm_build_norm(ctx0, Qcur, hparams, model.layers[il].attn_q_norm, nullptr, LLM_NORM_RMS, cb, il);
            if (hparams.has_kv(il)) {
                Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
                Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);
                Kcur = llm_build_norm(ctx0, Kcur, hparams, model.layers[il].attn_k_norm, nullptr, LLM_NORM_RMS, cb, il);
                Vcur = ggml_rms_norm(ctx0, Vcur, hparams.f_norm_rms_eps);
                cb(Kcur, "Kcur_normed", il);
                cb(Vcur, "Vcur_normed", il);
            }
            Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, freq_factors, n_rot_l, rope_type, n_ctx_orig, freq_base_l, freq_scale_l,
                    ext_factor, attn_factor, beta_fast, beta_slow);
            cb(Qcur, "Qcur_rope", il);
            if (hparams.has_kv(il)) {
                Kcur = ggml_rope_ext(ctx0, Kcur, inp_pos, freq_factors, n_rot_l, rope_type, n_ctx_orig, freq_base_l, freq_scale_l,
                        ext_factor, attn_factor, beta_fast, beta_slow);
                cb(Kcur, "Kcur_rope", il);
            }
            cur = llm_build_kv(ctx0, lctx, kv_self, gf, model.layers[il].wo, model.layers[il].bo,
                Kcur, Vcur, Qcur, KQ_mask_l, n_tokens, kv_head, n_kv, hparams.f_attention_scale, cb, il, nullptr, n_swa);


            if (il == n_layer - 1 && inp_out_ids) {
                // skip computing output for unused tokens
                cur  = ggml_get_rows(ctx0,  cur, inp_out_ids);
                inpL = ggml_get_rows(ctx0, inpL, inp_out_ids);
            }

            cur = llm_build_norm(ctx0, cur, hparams, model.layers[il].attn_post_norm, NULL, LLM_NORM_RMS, cb, il);
            cb(cur, "attn_post_norm", il);

            attn_out = ggml_add(ctx0, cur, inpL);
            cb(attn_out, "attn_out", il);
        }

        if (model.layers[il].ffn_gate_inp) {

            auto cur_mlp = llm_build_norm(ctx0, attn_out, hparams, model.layers[il].ffn_norm, nullptr, LLM_NORM_RMS, cb, il);
            cb(cur_mlp, "ffn_norm_1", il);

            cur_mlp = llm_build_ffn(ctx0, lctx, nullptr, cur_mlp,
                    model.layers[il].ffn_up,   nullptr, nullptr,
                    model.layers[il].ffn_gate, nullptr, nullptr,
                    model.layers[il].ffn_down, nullptr, nullptr,
                    nullptr,
                    LLM_FFN_GELU, LLM_FFN_PAR, cb, il, gf);

            // Expert FFN
            auto cur_moe = llm_build_norm(ctx0, attn_out, hparams, model.layers[il].ffn_pre_norm_2, nullptr, LLM_NORM_RMS, cb, il);
            cb(cur_moe, "ffn_norm_2", il);

            // custom MoE logits calculation (router operates on attn_out, not cur)
            auto tmp = ggml_fused_rms_norm(ctx0, attn_out, model.layers[il].ffn_gate_inp_s, hparams.f_norm_rms_eps);
            cb(tmp, "tmp", il);
            auto logits = llm_build_lora_mm(lctx, ctx0, model.layers[il].ffn_gate_inp, tmp); // [n_expert, n_tokens]
            cb(logits, "ffn_moe_logits", il);

            // TODO: we need to pass in the above logits
            cur_moe = llm_build_moe_ffn(ctx0, lctx, cur_moe,
                    nullptr, // gate_inp
                    nullptr, // up_exps
                    nullptr, // gate_exps
                    model.layers[il].ffn_down_exps,
                    nullptr, // exp_probs_b (not used for gemma4)
                    n_expert, n_expert_used,
                    LLM_FFN_GELU, true, false, 0.0f,
                    LLM_EXPERT_GATING_FUNC_SOFTMAX,
                    cb, il, gf,
                    false,
                    model.layers[il].ffn_up_gate_exps,
                    nullptr, logits, model.layers[il].ffn_down_exps_s);

            cur = ggml_fused_rms_rms_add(ctx0, cur_mlp, model.layers[il].ffn_post_norm_1, cur_moe, model.layers[il].ffn_post_norm_2, hparams.f_norm_rms_eps);
            cb(cur, "ffn_moe_combined", il);
            ggml_build_forward_expand(gf, cur);

            cur = llm_build_norm(ctx0, cur, hparams, model.layers[il].ffn_post_norm, NULL, LLM_NORM_RMS, cb, -1);
            cb(cur, "ffn_post_norm", -1);

            cur = ggml_add(ctx0, cur, attn_out);

        } else {

            cur = llm_build_ffn(ctx0, lctx, model.layers[il].ffn_norm, attn_out,
                    model.layers[il].ffn_up,   nullptr, nullptr,
                    model.layers[il].ffn_gate, nullptr, nullptr,
                    model.layers[il].ffn_down, nullptr, nullptr,
                    nullptr,
                    LLM_FFN_GELU, LLM_FFN_PAR, cb, il, gf, true, false, nullptr, model.layers[il].ffn_post_norm);
            cb(cur, "ffn_out", il);

        }

        if (inp_per_layer) {
            ggml_tensor * pe_in = cur;
            cb(cur, "pe_in", il);

            cur = llm_build_lora_mm(lctx, ctx0, model.layers[il].per_layer_inp_gate, cur); // [n_embd_per_layer, n_tokens]
            cur = ggml_gelu(ctx0, cur);
            ggml_tensor * inp_this_layer = ggml_view_2d(ctx0, inp_per_layer, inp_per_layer->ne[0], inp_per_layer->ne[1],
                    ggml_row_size(inp_per_layer->type, inp_per_layer->ne[0]),
                    il*inp_per_layer->ne[0]*inp_per_layer->ne[1]*ggml_element_size(inp_per_layer)); // [n_embd_per_layer, n_tokens]

            // TODO @ngxson : improve this
            if (il == n_layer - 1 && inp_out_ids) {
                inp_this_layer = ggml_get_rows(ctx0, inp_this_layer, inp_out_ids);
            }

            cur = ggml_mul(ctx0, cur, inp_this_layer);
            cur = llm_build_lora_mm(lctx, ctx0, model.layers[il].per_layer_proj, cur); // [n_embd, n_tokens]
            cur = llm_build_norm(ctx0, cur, hparams, model.layers[il].per_layer_post_norm, nullptr, LLM_NORM_RMS, cb, il);
            cb(cur, "per_layer_embd_out", il);

            // residual connection
            cur = ggml_add(ctx0, pe_in, cur);
        }

        // layer_scalar
        if (model.layers[il].out_scale) {
            cur = ggml_mul(ctx0, cur, model.layers[il].out_scale);
            cb(cur, "out_scaled", il);
        }

        cur = lctx.cvec.apply_to(ctx0, cur, il);
        cb(cur, "l_out", il);

        // input for next layer
        inpL = cur;
    }

    cur = inpL;

    if (cparams.mtp) {
        ggml_tensor * mtp_embd = ggml_dup(ctx0, cur);
        cb(mtp_embd, "result_mtp_embd", -1);
        ggml_build_forward_expand(gf, mtp_embd);
    }

    cur = llm_build_norm(ctx0, cur, hparams, model.output_norm, NULL, LLM_NORM_RMS, cb, -1);
    cb(cur, "result_norm", -1);

    // lm_head
    cur = llm_build_lora_mm(lctx, ctx0, model.output, cur);

    if (hparams.f_final_logit_softcapping > 0) {
        cur = ggml_softcap(ctx0, cur, 1.0f / hparams.f_final_logit_softcapping, hparams.f_final_logit_softcapping);
    }

    cb(cur, "result_output", -1);

    ggml_build_forward_expand(gf, cur);

    return gf;
}

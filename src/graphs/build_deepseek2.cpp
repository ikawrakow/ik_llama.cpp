#include "../llama-build-context.h"
#include "../llama-model.h"
#include "../llama-context.h"

ggml_cgraph * llm_build_context::build_deepseek2() {
#ifdef GGML_USE_VULKAN
    constexpr bool use_f32_attn_precision = true;
#else
    constexpr bool use_f32_attn_precision = false;
#endif
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, model.max_nodes(n_tokens), false);

    // mutable variable, needed during the last layer of the computation to skip unused tokens
    int32_t n_tokens = this->n_tokens;

    bool is_lite = (hparams.n_layer == 27 || hparams.n_layer == 26);

    // We have to pre-scale kq_scale and attn_factor to make the YaRN RoPE work correctly.
    // See https://github.com/ggerganov/llama.cpp/discussions/7416 for detailed explanation.
    const float mscale = attn_factor * (1.0f + hparams.rope_yarn_log_mul * logf(1.0f / freq_scale));
    const float kq_scale = 1.0f*mscale*mscale/sqrtf(float(hparams.n_embd_head_k(0)));
    const float attn_factor_scaled = 1.0f / (1.0f + 0.1f * logf(1.0f / freq_scale));

    const uint32_t n_embd_head_qk_rope = hparams.n_rot;
    const uint32_t n_embd_head_qk_nope = hparams.n_embd_head_k(0) - hparams.n_rot;
    const uint32_t kv_lora_rank = hparams.n_lora_kv;
    const uint32_t q_lora_rank  = hparams.n_lora_q;

    struct ggml_tensor * cur;
    struct ggml_tensor * inpL;

    // {n_embd, n_tokens}
    inpL = llm_build_inp_embd(ctx0, lctx, hparams, batch, model.tok_embd, cb);

    // inp_pos - contains the positions
    struct ggml_tensor * inp_pos = build_inp_pos();

    // KQ_mask (mask for 1 head, it will be broadcasted to all heads)
    struct ggml_tensor * KQ_mask = build_inp_KQ_mask();

    // whether to use n_tokens as the matrix dimension during multiplication or n_head
    // n_tokens is higher during prompt processing, this allows to optimize for this case
    bool pp_opt = n_tokens >= 128; // Is it a fixed constant or is it somehow relared to n_head? original: n_tokens > n_head;

    auto rope_cache = cparams.rope_cache && (rope_type == LLAMA_ROPE_TYPE_NEOX || rope_type == LLAMA_ROPE_TYPE_NORM) ?
        ggml_rope_cache(ctx0, inp_pos, nullptr, n_rot, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow) : nullptr;

    int n_active_layers = hparams.n_layer - hparams.nextn_predict_layers;
    for (int il = 0; il < n_active_layers; ++il) {
        struct ggml_tensor * inpSA = inpL;

        // norm
        cur = llm_build_norm(ctx0, inpL, hparams, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, cb, il);
        cb(cur, "attn_norm", il);

        // self_attention
        {
            ggml_tensor * q = nullptr;
            ggml_tensor * kv_rope_compressed = nullptr;
            ggml_tensor * q_rope;
            ggml_tensor * q_nope;
            ggml_tensor * k_rope;
            ggml_tensor * kv_compressed;
            if (model.layers[il].wkq_a_mqa) {
                auto mqa = ggml_mul_mat(ctx0, model.layers[il].wkq_a_mqa, cur);
                cb(mqa, "mqa", il);
                size_t qnb1;
                if (!is_lite) {
                    q = ggml_view_2d(ctx0, mqa, q_lora_rank, n_tokens, mqa->nb[1], 0);
                    q = llm_build_norm(ctx0, q, hparams, model.layers[il].attn_q_a_norm, NULL, LLM_NORM_RMS, cb, il);
                    q = ggml_mul_mat(ctx0, model.layers[il].wq_b, q);
                    qnb1 = q->nb[1];
                    cb(q, "q", il);
                    kv_rope_compressed = ggml_view_2d(ctx0, mqa, kv_lora_rank + n_embd_head_qk_rope, n_tokens, mqa->nb[1],
                            q_lora_rank*ggml_element_size(mqa));
                } else {
                    q = ggml_view_2d(ctx0, mqa, n_embd_k_gqa, n_tokens, mqa->nb[1], 0);
                    kv_rope_compressed = ggml_view_2d(ctx0, mqa, kv_lora_rank + n_embd_head_qk_rope, n_tokens, mqa->nb[1],
                            n_embd_k_gqa*ggml_element_size(mqa));
                    qnb1 = mqa->nb[1];
                }
                q_nope = ggml_view_3d(ctx0, q, n_embd_head_qk_nope, n_head, n_tokens,
                    ggml_row_size(q->type, hparams.n_embd_head_k(il)), qnb1, 0);
                q_rope = ggml_view_3d(ctx0, q, n_embd_head_qk_rope, n_head, n_tokens,
                    ggml_row_size(q->type, hparams.n_embd_head_k(il)), qnb1, ggml_row_size(q->type, n_embd_head_qk_nope));
                k_rope = ggml_view_3d(ctx0, kv_rope_compressed, n_embd_head_qk_rope, 1, n_tokens,
                        mqa->nb[1], mqa->nb[1], ggml_row_size(kv_rope_compressed->type, kv_lora_rank));
                kv_compressed = ggml_view_2d(ctx0, kv_rope_compressed, kv_lora_rank, n_tokens, mqa->nb[1], 0);
            }
            else {
                if (!is_lite) {
                    q = ggml_mul_mat(ctx0, model.layers[il].wq_a, cur);
                    cb(q, "q", il);

                    kv_rope_compressed = ggml_mul_mat(ctx0, model.layers[il].wkv_a_mqa, cur);
                    cb(kv_rope_compressed, "kv_rope_compressed", il);

                    ggml_build_forward_expand(gf, q);
                    ggml_build_forward_expand(gf, kv_rope_compressed);

                    q = llm_build_norm(ctx0, q, hparams, model.layers[il].attn_q_a_norm, NULL, LLM_NORM_RMS, cb, il);
                    cb(q, "q", il);

                    q = ggml_mul_mat(ctx0, model.layers[il].wq_b, q);
                    cb(q, "q", il);
                } else {
                    q = ggml_mul_mat(ctx0, model.layers[il].wq, cur);
                    cb(q, "q", il);

                    kv_rope_compressed = ggml_mul_mat(ctx0, model.layers[il].wkv_a_mqa, cur);
                    cb(kv_rope_compressed, "kv_rope_compressed", il);

                    ggml_build_forward_expand(gf, q);
                    ggml_build_forward_expand(gf, kv_rope_compressed);
                }

                q_nope = ggml_view_3d(ctx0, q, n_embd_head_qk_nope, n_head, n_tokens,
                        ggml_row_size(q->type, hparams.n_embd_head_k(il)),
                        ggml_row_size(q->type, hparams.n_embd_head_k(il) * n_head), 0);

                q_rope = ggml_view_3d(ctx0, q, n_embd_head_qk_rope, n_head, n_tokens,
                        ggml_row_size(q->type, hparams.n_embd_head_k(il)),
                        ggml_row_size(q->type, hparams.n_embd_head_k(il) * n_head),
                        ggml_row_size(q->type, n_embd_head_qk_nope));

                k_rope = ggml_view_3d(ctx0, kv_rope_compressed, n_embd_head_qk_rope, 1, n_tokens,
                        kv_rope_compressed->nb[1],
                        kv_rope_compressed->nb[1],
                        ggml_row_size(kv_rope_compressed->type, kv_lora_rank));

                kv_compressed = ggml_view_2d(ctx0, kv_rope_compressed, kv_lora_rank, n_tokens,
                        kv_rope_compressed->nb[1], 0);
            }
            cb(q_nope, "q_nope", il);
            cb(q_rope, "q_rope", il);
            cb(k_rope, "k_rope", il);
            cb(kv_compressed, "kv_compressed", il);

            ggml_build_forward_expand(gf, q_rope);
            ggml_build_forward_expand(gf, k_rope);
            if (rope_cache) {
                q_rope = ggml_rope_fast(ctx0, q_rope, rope_cache);
                k_rope = ggml_rope_fast(ctx0, k_rope, rope_cache);
            } else {
                q_rope = ggml_rope_ext(ctx0, q_rope, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                        ext_factor, attn_factor_scaled, beta_fast, beta_slow);

                k_rope = ggml_rope_ext(ctx0, k_rope, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                        ext_factor, attn_factor_scaled, beta_fast, beta_slow);
            }
            cb(q_rope, "q_rope", il);
            cb(k_rope, "k_rope", il);
            ggml_build_forward_expand(gf, q_rope);
            ggml_build_forward_expand(gf, k_rope);

            kv_compressed = llm_build_norm(ctx0, kv_compressed, hparams, model.layers[il].attn_kv_a_norm, NULL, LLM_NORM_RMS, cb, il);
            cb(kv_compressed, "kv_compressed", il);

            if (lctx.cparams.mla_attn) {

                ggml_tensor * kv_cache_trans = nullptr;

                if (lctx.cparams.mla_attn == 1 && !lctx.cparams.flash_attn) {
                    ggml_tensor * kv_cache_trans_view = ggml_view_2d(ctx0, kv_self.v_l[il], n_tokens, kv_lora_rank,
                            ggml_row_size(kv_self.v_l[il]->type, kv_self.size), ggml_row_size(kv_self.v_l[il]->type, kv_head));
                    cb(kv_cache_trans_view, "kv_cache_trans_view", il);

                    // note: storing transposed c^KV in the transposed KV cache
                    ggml_build_forward_expand(gf, ggml_cpy(ctx0, ggml_transpose(ctx0, kv_compressed), kv_cache_trans_view));

                    kv_cache_trans = ggml_view_2d(ctx0, kv_self.v_l[il],
                            n_kv, kv_lora_rank,
                            ggml_row_size(kv_self.v_l[il]->type, kv_self.size),
                            0);
                    cb(kv_cache_trans, "kv_cache_trans", il);
                }

                //ggml_tensor * kvr = ggml_concat(ctx0, kv_compressed, ggml_permute(ctx0, k_rope, 0, 2, 1, 3), 0);
                ggml_tensor * kvr = ggml_concat(ctx0, ggml_permute(ctx0, k_rope, 0, 2, 1, 3), kv_compressed, 0);
                cb(kvr, "kvr", il);

                auto row_size = ggml_row_size(kv_self.k_l[il]->type, kv_lora_rank + n_embd_head_qk_rope);
                ggml_tensor * kv_cache_view = ggml_view_2d(ctx0, kv_self.k_l[il], kv_self.k_l[il]->ne[0], n_tokens,
                        row_size, row_size*kv_head);
                lctx.cache_copies[2*il+0].cpy = ggml_cpy(ctx0, kvr, kv_cache_view);
                lctx.cache_copies[2*il+0].step = row_size;
                ggml_build_forward_expand(gf, lctx.cache_copies[2*il+0].cpy);
                ggml_tensor * kv_cache = ggml_view_2d(ctx0, kv_self.k_l[il],
                        kv_lora_rank + n_embd_head_qk_rope, n_kv,
                        ggml_row_size(kv_self.k_l[il]->type, kv_lora_rank + n_embd_head_qk_rope), 0);
                cb(kv_cache, "kv_cache", il);

                ggml_tensor * kqv;

                if (lctx.cparams.mla_attn > 1 && lctx.cparams.flash_attn && pp_opt) { // PP for mla=2,3

                    auto kv_cache_nope = ggml_view_2d(ctx0, kv_self.k_l[il], kv_lora_rank, n_kv, kv_self.k_l[il]->nb[1],
                            ggml_row_size(kv_self.k_l[il]->type, n_embd_head_qk_rope));

                    auto kv_f32_size = model.layers[il].wkv_b->ne[1] * kv_cache_nope->ne[1] * sizeof(float) / (1024*1024);
                    int n_max_head = n_head;
                    if (cparams.attn_max_batch > 0 && kv_f32_size > cparams.attn_max_batch) {
                        n_max_head = 1;
                        for (int niter = 2; niter < n_head; ++niter) {
                            if (n_head % niter == 0 && kv_f32_size/niter <= cparams.attn_max_batch) {
                                n_max_head = n_head/niter;
                                break;
                            }
                        }
                    }
                    GGML_ASSERT(n_head % n_max_head == 0);

                    auto n_per_head = model.layers[il].wkv_b->ne[1] / n_head;

                    auto kv_cache_rope = ggml_view_3d(ctx0, kv_self.k_l[il], n_embd_head_qk_rope, n_kv, 1,
                            kv_self.k_l[il]->nb[1], kv_self.k_l[il]->nb[2], 0); //ggml_row_size(kv_self.k_l[il]->type, kv_lora_rank));

                    // There is still an issue with one or more of the ops GGML_OP_REPEAT, GGML_OP_CONCAT, GGML_OP_CPY on CUDA when
                    // the KV cache is quantized. Hence, in that case we will simply use fp16 for now.
                    // The downside of the following line is that fp16 will be used even if attention is computed on the CPU
                    // if the build is with CUDA enabled.
                    auto kv_type = lctx.backends.size() == 1 && lctx.backends.front() == lctx.backend_cpu ? kv_self.k_l[il]->type : GGML_TYPE_F16;

                    ggml_tensor repeater;
                    repeater.ne[0] = n_embd_head_qk_rope; repeater.ne[1] = n_kv; repeater.ne[2] = n_max_head; repeater.ne[3] = 1;
                    ggml_tensor * k_rope;
                    if (kv_cache_rope->type == kv_type) {
                        k_rope = ggml_repeat(ctx0, kv_cache_rope, &repeater);
                    } else {
                        auto kv_cache_rope_f16 = ggml_cast(ctx0, kv_cache_rope, GGML_TYPE_F16);
                        k_rope = ggml_repeat(ctx0, kv_cache_rope_f16, &repeater);
                    }
                    cb(k_rope, "k_rope", il);

                    //auto q = ggml_concat(ctx0, q_nope, q_rope, 0);
                    auto q = ggml_concat(ctx0, q_rope, q_nope, 0);
                    q = ggml_permute(ctx0, q, 0, 2, 1, 3);
                    cb(q, "q_concat", il);

                    ggml_build_forward_expand(gf, q);

                    for (int iter = 0; iter < n_head/n_max_head; ++iter) {

                        auto wkv_b = ggml_view_2d(ctx0, model.layers[il].wkv_b, model.layers[il].wkv_b->ne[0], n_per_head*n_max_head,
                                model.layers[il].wkv_b->nb[1], model.layers[il].wkv_b->nb[1]*n_per_head*n_max_head*iter);

                        auto kv_f32 = ggml_mul_mat(ctx0, wkv_b, kv_cache_nope);
                        cb(kv_f32, "kv_f32", il);

                        auto v_f32 = ggml_view_3d(ctx0, kv_f32, hparams.n_embd_head_v_full, n_kv, n_max_head,
                                ggml_row_size(kv_f32->type, n_max_head * (n_embd_head_qk_nope + hparams.n_embd_head_v_full)),
                                ggml_row_size(kv_f32->type, n_embd_head_qk_nope + hparams.n_embd_head_v_full),
                                ggml_row_size(kv_f32->type, n_embd_head_qk_nope));
                        cb(v_f32, "v_f32", il);

                        auto k_nope_f32 = ggml_view_3d(ctx0, kv_f32, n_embd_head_qk_nope, n_kv, n_max_head,
                                ggml_row_size(kv_f32->type, n_max_head * (n_embd_head_qk_nope + hparams.n_embd_head_v_full)),
                                ggml_row_size(kv_f32->type, n_embd_head_qk_nope + hparams.n_embd_head_v_full), 0);
                        cb(k_nope_f32, "k_nope_f32", il);

                        auto v = ggml_cast(ctx0, v_f32, kv_type);
                        cb(v, "v", il);

                        auto k_nope = ggml_cast(ctx0, k_nope_f32, kv_type);
                        cb(k_nope, "k_nope", il);

                        ggml_build_forward_expand(gf, k_nope);
                        ggml_build_forward_expand(gf, v);

                        //auto k = ggml_concat(ctx0, k_nope, k_rope, 0);
                        auto k = ggml_concat(ctx0, k_rope, k_nope, 0);
                        cb(k, "k", il);

                        ggml_build_forward_expand(gf, k);

                        auto q_iter = ggml_view_3d(ctx0, q, q->ne[0], q->ne[1], n_max_head,
                                q->nb[1], q->nb[2], q->nb[2]*n_max_head*iter);

                        kqv = ggml_flash_attn_ext(ctx0, q_iter, k, v, KQ_mask, kq_scale, hparams.f_max_alibi_bias, 0.f);
                        if (use_f32_attn_precision || q->ne[1] <= 8) {
                            ggml_flash_attn_ext_set_prec(kqv, GGML_PREC_F32);
                        }
                        cb(kqv, "kqv", il);

                        if (iter == 0) {
                            cur = ggml_reshape_2d(ctx0, kqv, n_embd_head_v*n_max_head, n_tokens);
                        } else {
                            cur = ggml_concat(ctx0, cur, ggml_reshape_2d(ctx0, kqv, n_embd_head_v*n_max_head, n_tokens), 0);
                        }
                        ggml_build_forward_expand(gf, cur);

                    }

                }
                else {

                    ggml_tensor * kqv_compressed = nullptr;

                    //auto wkv_b = model.layers[il].wkv_b;
                    auto wk_b = model.layers[il].wk_b->ne[1] == kv_lora_rank ? model.layers[il].wk_b
                        : ggml_reshape_3d(ctx0, model.layers[il].wk_b, n_embd_head_qk_nope, kv_lora_rank, n_head);

                    q_nope = ggml_permute(ctx0, q_nope, 0, 2, 1, 3);
                    cb(q_nope, "q_nope_perm", il);

                    struct ggml_tensor * q_nope2 = ggml_mul_mat(ctx0, wk_b, q_nope);
                    cb(q_nope2, "q_nope2", il);

                    //ggml_tensor * q = ggml_concat(ctx0, q_nope2, ggml_permute(ctx0, q_rope, 0, 2, 1, 3), 0);
                    ggml_tensor * q = ggml_concat(ctx0, ggml_permute(ctx0, q_rope, 0, 2, 1, 3), q_nope2, 0);
                    cb(q, "q", il);

                    if (lctx.cparams.flash_attn && (lctx.cparams.mla_attn == 1 || lctx.cparams.mla_attn == 3)) {
                        ggml_tensor * kv_cache_lora = ggml_view_2d(ctx0, kv_self.k_l[il],
                                kv_lora_rank, n_kv,
                                ggml_row_size(kv_self.k_l[il]->type, kv_lora_rank + n_embd_head_qk_rope),
                                ggml_row_size(kv_self.k_l[il]->type, n_embd_head_qk_rope));
                        cb(kv_cache_lora, "kv_cache_lora", il);

                        kqv_compressed = ggml_flash_attn_ext(ctx0, q, kv_cache, kv_cache_lora, KQ_mask, kq_scale, hparams.f_max_alibi_bias, 0.f);
                        cb(kqv_compressed, "kqv_compressed", il);

                        if (use_f32_attn_precision) {
                            ggml_flash_attn_ext_set_prec(kqv_compressed, GGML_PREC_F32);
                        }

                        kqv_compressed = ggml_permute(ctx0, kqv_compressed, 0, 2, 1, 3);
                        cb(kqv_compressed, "kqv_compressed_perm", il);
                    }
                    else {
                        if (lctx.cparams.mla_attn > 1) {
                            ggml_tensor * kv_cache_lora = ggml_view_2d(ctx0, kv_self.k_l[il],
                                    kv_lora_rank, n_kv,
                                    ggml_row_size(kv_self.k_l[il]->type, kv_lora_rank + n_embd_head_qk_rope),
                                    ggml_row_size(kv_self.k_l[il]->type, n_embd_head_qk_rope));
                            cb(kv_cache, "kv_cache_lora", il);

                            kv_cache_trans = ggml_cont(ctx0, ggml_transpose(ctx0, kv_cache_lora));
                            cb(kv_cache_trans, "kv_cache_trans", il);
                        }

                        auto kq_size = kv_cache->ne[1]*q->ne[1]*q->ne[2]*sizeof(float)/(1024*1024); // K*Q in MiB
                        if (lctx.cparams.attn_max_batch <= 0 || lctx.cparams.attn_max_batch >= kq_size) {
                            if (!pp_opt) {
                                q = ggml_permute(ctx0, q, 0, 2, 1, 3);
                                cb(q, "q_perm", il);
                            }

                            ggml_tensor * kq = ggml_mul_mat(ctx0, kv_cache, q);
                            if (kv_cache->ne[1] < 256) {
                                ggml_mul_mat_set_prec(kq, GGML_PREC_F32);
                            }
                            cb(kq, "kq", il);

                            if (!pp_opt) {
                                kq = ggml_cont(ctx0, ggml_permute(ctx0, kq, 0, 2, 1, 3));
                                cb(kq, "kq_perm", il);
                            }

                            kq = ggml_soft_max_ext(ctx0, kq, KQ_mask, kq_scale, hparams.f_max_alibi_bias);
                            cb(kq, "kq_soft_max_ext", il);

                            if (!pp_opt) {
                                kq = ggml_permute(ctx0, kq, 0, 2, 1, 3);
                                cb(kq, "kq_soft_max_ext_perm", il);
                            }

                            kqv_compressed = ggml_mul_mat(ctx0, kv_cache_trans, kq);
                            cb(kqv_compressed, "kqv_compressed", il);

                            if (!pp_opt) {
                                kqv_compressed = ggml_permute(ctx0, kqv_compressed, 0, 2, 1, 3);
                                cb(kqv_compressed, "kqv_compressed_perm", il);
                            }

                        } else {

                            int n_step = (kq_size + lctx.cparams.attn_max_batch - 1)/lctx.cparams.attn_max_batch;
                            n_step = std::min(n_step, int(q->ne[2]));
                            int n_per_step = (q->ne[2] + n_step - 1)/n_step;

                            for (int i_head = 0; i_head < q->ne[2]; i_head += n_per_step) {
                                int this_ne12 = i_head + n_per_step <= q->ne[2] ? n_per_step : q->ne[2] - i_head;
                                ggml_tensor * q_i = ggml_view_3d(ctx0, q, q->ne[0], q->ne[1], this_ne12, q->nb[1], q->nb[2], q->nb[2]*i_head);
                                ggml_tensor * kq_i = ggml_mul_mat(ctx0, kv_cache, q_i);
                                kq_i = ggml_soft_max_ext(ctx0, kq_i, KQ_mask, kq_scale, hparams.f_max_alibi_bias);
                                ggml_tensor * kqv_i = ggml_mul_mat(ctx0, kv_cache_trans, kq_i);
                                if (i_head == 0) {
                                    kqv_compressed = kqv_i;
                                } else {
                                    kqv_compressed = ggml_concat(ctx0, kqv_compressed, kqv_i, 2);
                                }
                                ggml_build_forward_expand(gf, kqv_compressed);
                            }
                            cb(kqv_compressed, "kqv_compressed", il);
                        }
                    }

                    auto wv_b = model.layers[il].wv_b;
                    if (wv_b->ne[1] != n_embd_head_v) {
                        wv_b = ggml_reshape_3d(ctx0, wv_b, kv_lora_rank, n_embd_head_v, n_head);
                        cb(wv_b, "wv_b", il);
                    }
                    // There is an issue with quantized GEMV on CUDA when the left operand (the matrix) is
                    // not contiguous. So, for now, we create wv_b during model loading and use that
                    // instead of the commented out 3D view below.
                    //auto wv_b = ggml_view_3d(ctx0, wkv_b, kv_lora_rank, n_embd_head_v, n_head,
                    //        wkv_b->nb[1], wkv_b->nb[1]*(n_embd_head_v + n_embd_head_qk_nope),
                    //        wkv_b->nb[1]*n_embd_head_qk_nope);
                    //cb(wv_b, "wv_b", il);

                    kqv = ggml_mul_mat(ctx0, wv_b, kqv_compressed);
                    cb(kqv, "kqv", il);

                    if (n_tokens > 1) {
                        kqv = ggml_cont(ctx0, ggml_permute(ctx0, kqv, 0, 2, 1, 3));
                        cb(kqv, "kqv_perm", il);
                    }
                    cur = ggml_reshape_2d(ctx0, kqv, n_embd_head_v*n_head, n_tokens);
                    cb(cur, "kqv_2d", il);

                }

                ggml_build_forward_expand(gf, cur);

                cur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wo, cur);
                cb(cur, "kqv_out", il);

            }
            else {

                // {kv_lora_rank, n_head * (n_embd_head_qk_nope + n_embd_head_v)} * {kv_lora_rank, n_tokens} -> {n_head * (n_embd_head_qk_nope + n_embd_head_v), n_tokens}
                struct ggml_tensor * kv = ggml_mul_mat(ctx0, model.layers[il].wkv_b, kv_compressed);
                cb(kv, "kv", il);

                // split into {n_head * n_embd_head_qk_nope, n_tokens}
                struct ggml_tensor * k_nope = ggml_view_3d(ctx0, kv, n_embd_head_qk_nope, n_head, n_tokens,
                        ggml_row_size(kv->type, n_embd_head_qk_nope + hparams.n_embd_head_v_full),
                        ggml_row_size(kv->type, n_head * (n_embd_head_qk_nope + hparams.n_embd_head_v_full)),
                        0);
                cb(k_nope, "k_nope", il);

                // and {n_head * n_embd_head_v, n_tokens}
                struct ggml_tensor * v_states = ggml_view_3d(ctx0, kv, hparams.n_embd_head_v_full, n_head, n_tokens,
                        ggml_row_size(kv->type, (n_embd_head_qk_nope + hparams.n_embd_head_v_full)),
                        ggml_row_size(kv->type, (n_embd_head_qk_nope + hparams.n_embd_head_v_full)*n_head),
                        ggml_row_size(kv->type, (n_embd_head_qk_nope)));
                cb(v_states, "v_states", il);

                v_states = ggml_cont(ctx0, v_states);
                cb(v_states, "v_states", il);

                v_states = ggml_view_2d(ctx0, v_states, hparams.n_embd_head_v_full * n_head, n_tokens,
                        ggml_row_size(kv->type, hparams.n_embd_head_v_full * n_head),
                        0);
                cb(v_states, "v_states", il);

                struct ggml_tensor * q_states = ggml_concat(ctx0, q_nope, q_rope, 0);
                cb(q_states, "q_states", il);

                struct ggml_tensor * k_states = ggml_concat(ctx0, k_nope, ggml_repeat(ctx0, k_rope, q_rope), 0);
                cb(k_states, "k_states", il);

                cur = llm_build_kv(ctx0, lctx, kv_self, gf,
                        model.layers[il].wo, NULL,
                        k_states, v_states, q_states, KQ_mask, n_tokens, kv_head, n_kv, kq_scale, cb, il);

            }
        }

        if (il == n_active_layers - 1) {
            // skip computing output for unused tokens
            struct ggml_tensor * inp_out_ids = build_inp_out_ids();
            n_tokens = n_outputs;
            cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
            cb(cur, "last_attn", il);
            cb(inpSA, "last_ffn_inp", il);
        }

        struct ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        cur = llm_build_norm(ctx0, ffn_inp, hparams, model.layers[il].ffn_norm, NULL, LLM_NORM_RMS, cb, il);
        cb(cur, "ffn_norm", il);

        if ((uint32_t) il < hparams.n_layer_dense_lead) {
            cur = llm_build_ffn(ctx0, lctx, nullptr, cur,
                    model.layers[il].ffn_up,   NULL, NULL,
                    model.layers[il].ffn_gate, NULL, NULL,
                    model.layers[il].ffn_down, NULL, NULL,
                    NULL,
                    LLM_FFN_SILU, LLM_FFN_PAR, cb, il);
            cb(cur, "ffn_out", il);
        } else {
            // MoE branch
            ggml_tensor * moe_out =
                llm_build_moe_ffn(ctx0, lctx, cur,
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

            // FFN shared expert
            {
                ggml_tensor * ffn_shexp = llm_build_ffn(ctx0, lctx, nullptr, cur,
                        model.layers[il].ffn_up_shexp,   NULL, NULL,
                        model.layers[il].ffn_gate_shexp, NULL, NULL,
                        model.layers[il].ffn_down_shexp, NULL, NULL,
                        NULL,
                        LLM_FFN_SILU, LLM_FFN_PAR, cb, il);
                cb(ffn_shexp, "ffn_shexp", il);

                cur = ggml_add(ctx0, moe_out, ffn_shexp);
                cb(cur, "ffn_out", il);
            }
        }

        cur = ggml_add(ctx0, cur, ffn_inp);
        cur = lctx.cvec.apply_to(ctx0, cur, il);
        cb(cur, "l_out", il);

        // input for next layer
        inpL = cur;
    }

    cur = inpL;

    cur = llm_build_norm(ctx0, cur, hparams, model.output_norm, NULL, LLM_NORM_RMS, cb, -1);
    cb(cur, "result_norm", -1);

    // lm_head
    cur = ggml_mul_mat(ctx0, model.output, cur);
    cb(cur, "result_output", -1);

    ggml_build_forward_expand(gf, cur);

    return gf;
}

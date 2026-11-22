#include "../llama-build-context.h"
#include "../llama-context.h"
#include "../llama-model.h"

#include <algorithm>
#include <cmath>
#include <vector>

// Saved sources, each RMS-normalized once: keys [n_embd, n, n_tokens]; values, their transpose, for the loop bank.
struct inkling_stack {
    std::vector<ggml_tensor *> rows;
    ggml_tensor * keys = nullptr;
    ggml_tensor * values = nullptr;
};

static ggml_tensor * inkling_norm(ggml_context * ctx, ggml_tensor * value) {
    return ggml_rms_norm(ctx, ggml_reshape_3d(ctx, value, value->ne[0], 1, value->ne[1]), 1e-6f);
}

static ggml_tensor * inkling_append(ggml_context * ctx, ggml_tensor * keys, ggml_tensor * row) {
    return keys ? ggml_concat(ctx, keys, row, 1) : row;
}

static void inkling_push(ggml_context * ctx, inkling_stack & stack, ggml_tensor * value) {
    stack.rows.push_back(inkling_norm(ctx, value));
    stack.keys = inkling_append(ctx, stack.keys, stack.rows.back());
}

static inkling_stack inkling_sources(ggml_context * ctx, const inkling_stack & blocks, ggml_tensor * partial, int first) {
    inkling_stack stack;
    for (int i = first; i < (int) blocks.rows.size(); ++i) {
        stack.rows.push_back(blocks.rows[i]);
        stack.keys = inkling_append(ctx, stack.keys, blocks.rows[i]);
    }
    inkling_push(ctx, stack, partial);
    stack.values = ggml_cont(ctx, ggml_transpose(ctx, stack.keys));
    return stack;
}

static ggml_tensor * inkling_mix(ggml_context * ctx, const inkling_stack & stack, ggml_tensor * partial, ggml_tensor * query) {
    auto keys = partial ? inkling_append(ctx, stack.keys, inkling_norm(ctx, partial)) : stack.keys;
    auto scores = ggml_mul_mat(ctx, query, keys);
    ggml_mul_mat_set_prec(scores, GGML_PREC_F32);
    scores = ggml_cont(ctx, ggml_transpose(ctx, scores));
    auto probs = ggml_soft_max_ext(ctx, scores, nullptr, 1.0f / std::sqrt(float(keys->ne[0])), 0.0f);
    auto values = partial || !stack.values ? ggml_cont(ctx, ggml_transpose(ctx, keys)) : stack.values;
    auto result = ggml_mul_mat(ctx, values, probs);
    ggml_mul_mat_set_prec(result, GGML_PREC_F32);
    return result;
}

static ggml_tensor * inkling_row(ggml_context * ctx, ggml_tensor * mixed, int index) {
    auto view = ggml_view_2d(ctx, mixed, mixed->ne[0], mixed->ne[2], mixed->nb[2], index * mixed->nb[1]);
    return ggml_is_contiguous(view) ? view : ggml_cont(ctx, view);
}

static ggml_tensor * inkling_control(ggml_context * ctx, ggml_tensor * dest, ggml_tensor * message, ggml_tensor * down,
        ggml_tensor * forget, ggml_tensor * gain, float eps) {
    auto d = ggml_rms_norm(ctx, dest, eps);
    auto m = ggml_rms_norm(ctx, message, eps);
    auto h = ggml_mul_mat(ctx, down, ggml_concat(ctx, d, m, 0));
    ggml_mul_mat_set_prec(h, GGML_PREC_F32);
    auto f = ggml_mul_mat(ctx, forget, h);
    auto g = ggml_mul_mat(ctx, gain, h);
    ggml_mul_mat_set_prec(f, GGML_PREC_F32);
    ggml_mul_mat_set_prec(g, GGML_PREC_F32);
    auto keep = ggml_scale_bias(ctx, ggml_tanh(ctx, f), -1.0f, 1.0f);
    return ggml_add(ctx, ggml_mul(ctx, keep, dest), ggml_mul(ctx, ggml_tanh(ctx, g), message));
}

static ggml_tensor * inkling_conv(llm_build_context & llm, ggml_cgraph * gf, int il, ggml_tensor * x,
        int channel_offset, ggml_tensor * weight, ggml_tensor * delta = nullptr) {
    auto ctx = llm.ctx0;
    auto state = llm.kv_self.s_l[il];
    const int kernel = weight->ne[0];
    const int history = kernel - 1;
    const int channels = x->ne[0];
    const int tokens = x->ne[1];
    const auto & batch = llm.batch;
    auto state_view = [&](llama_seq_id seq) {
        return ggml_view_2d(ctx, state, history, channels, history * sizeof(float),
                seq * state->nb[1] + (size_t) channel_offset * history * sizeof(float));
    };
    auto past = state_view(batch.seq_id ? batch.seq_id[0][0] : batch.all_seq_id);
    if (batch.pos && batch.pos[0] == 0) {
        past = ggml_fill(ctx, past, 0.0f);
    }
    auto window = ggml_concat(ctx, past, ggml_cont(ctx, ggml_transpose(ctx, x)), 0);
    auto tail = ggml_view_2d(ctx, window, history, channels, window->nb[1], tokens * window->nb[0]);
    const int sequences = batch.n_seq_id ? batch.n_seq_id[0] : 1;
    for (int i = 0; i < sequences; ++i) {
        const llama_seq_id seq = batch.seq_id ? batch.seq_id[0][i] : batch.all_seq_id;
        ggml_build_forward_expand(gf, ggml_cpy(ctx, tail, state_view(seq)));
    }

    ggml_tensor * sum = nullptr;
    if (!delta) {
        sum = ggml_ssm_conv_m2(ctx, window, weight);
    } else {
        const int heads = llm.hparams.n_head_kv(il);
        const int width = channels / heads;
        delta = ggml_reshape_3d(ctx, delta, kernel, heads, tokens);
        auto columns = ggml_cont(ctx, ggml_transpose(ctx, window));
        for (int lag = 0; lag < kernel; ++lag) {
            const int tap = history - lag;
            auto samples = ggml_view_3d(ctx, columns, width, heads, tokens, columns->nb[0] * width, columns->nb[1], tap * columns->nb[1]);
            auto base = ggml_view_2d(ctx, weight, 1, channels, weight->nb[1], tap * weight->nb[0]);
            base = ggml_reshape_2d(ctx, ggml_cont(ctx, base), width, heads);
            auto changes = ggml_view_3d(ctx, delta, 1, heads, tokens, delta->nb[1], delta->nb[2], lag * delta->nb[0]);
            auto taps = ggml_add(ctx, ggml_repeat(ctx, base, samples), changes);
            auto product = ggml_mul(ctx, taps, samples);
            sum = sum ? ggml_add(ctx, sum, product) : product;
        }
        sum = ggml_reshape_2d(ctx, sum, channels, tokens);
    }
    return ggml_scale(ctx, ggml_add(ctx, sum, x), 1.0f / std::sqrt(float((delta ? 2 : 1) * kernel + 1)));
}

static ggml_tensor * inkling_clip(ggml_context * ctx, ggml_tensor * x, ggml_tensor * weight, float limit) {
    if (weight) {
        x = ggml_mul(ctx, x, weight);
    }
    auto length = ggml_sqrt(ctx, ggml_sum_rows(ctx, ggml_sqr(ctx, x)));
    length = ggml_scale_bias(ctx, length, 1.0f, 1e-6f);
    auto factor = ggml_div(ctx, ggml_fill(ctx, length, limit), length);
    auto flat = ggml_reshape_1d(ctx, factor, ggml_nelements(factor));
    return ggml_mul(ctx, x, ggml_reshape(ctx, ggml_clamp(ctx, flat, 0.0f, 1.0f), factor));
}

static ggml_tensor * inkling_attention(llm_build_context & llm, ggml_cgraph * gf, int il,
        ggml_tensor * uq, ggml_tensor * uk, ggml_tensor * uv, ggml_tensor * mask,
        ggml_tensor * tau) {
    auto ctx = llm.ctx0;
    const auto & layer = llm.model.layers[il];
    const auto & hp = llm.hparams;
    const auto & kv = llm.kv_self;
    const int heads = hp.n_head(il);
    const int kv_heads = hp.n_head_kv(il);
    const int dk = hp.n_embd_head_k(il);
    const int dv = hp.n_embd_head_v(il);
    const int tokens = llm.n_tokens;
    auto q = ggml_reshape_3d(ctx, llm.llm_build_lora_mm(llm.lctx, ctx, layer.wq, uq), dk, heads, tokens);
    auto r = ggml_reshape_3d(ctx, llm.llm_build_lora_mm(llm.lctx, ctx, layer.attn_rel, uq), hp.n_relative, heads, tokens);
    q = inkling_clip(ctx, q, layer.attn_q_clip, hp.f_qk_clip);
    r = inkling_clip(ctx, r, layer.attn_rel_clip, hp.f_relative_clip);
    if (tau) {
        q = ggml_mul(ctx, q, tau);
        r = ggml_mul(ctx, r, tau);
    }

    auto k = inkling_conv(llm, gf, il, llm.llm_build_lora_mm(llm.lctx, ctx, layer.wk, uk), 0, layer.attn_k_conv,
            tau ? llm.llm_build_lora_mm(llm.lctx, ctx, layer.attn_k_conv_proj, uk) : nullptr);
    k = inkling_clip(ctx, ggml_reshape_3d(ctx, k, dk, kv_heads, tokens), nullptr, hp.f_qk_clip);
    auto v = inkling_conv(llm, gf, il, llm.llm_build_lora_mm(llm.lctx, ctx, layer.wv, uv), dk * kv_heads, layer.attn_v_conv,
            tau ? llm.llm_build_lora_mm(llm.lctx, ctx, layer.attn_v_conv_proj, uv) : nullptr);
    v = ggml_rms_norm(ctx, ggml_reshape_3d(ctx, v, dv, kv_heads, tokens), llm.norm_rms_eps);
    if (llm.cparams.k_cache_hadamard) {
        if (int block_size = llm.model.hadamard_size_k(il); block_size > 0) {
            q = ggml_hadamard(ctx, q, block_size);
            k = ggml_hadamard(ctx, k, block_size);
        }
    }
    const int v_hadamard = llm.cparams.v_cache_hadamard ? llm.model.hadamard_size_v(il) : 0;
    if (v_hadamard > 0) {
        v = ggml_hadamard(ctx, v, v_hadamard);
    }
    const bool compact = kv.is_compacted(il);
    llm.llm_build_kv_store(llm.lctx, ctx, hp, llm.cparams, kv, gf, k, v, tokens,
            compact ? llm.swa_head : llm.kv_head, llm.cb, il);

    const int count = mask->ne[0];
    const int offset = compact ? llm.lctx.swa_window_view.win_off : 0;
    auto cache_view = [&](ggml_tensor * cache, int dim) {
        const size_t stride = ggml_row_size(cache->type, dim);
        const size_t row = kv_heads * stride;
        return ggml_view_3d(ctx, cache, dim, kv_heads, count, stride, row, offset * row);
    };
    auto keys = ggml_permute(ctx, cache_view(kv.k_l[il], dk), 0, 2, 1, 3);
    q = ggml_permute(ctx, q, 0, 2, 1, 3);
    const float scale = 1.0f / std::sqrt(float(dk));

    auto table = ggml_mul_mat(ctx, layer.attn_rel_bias, r);
    table = ggml_pad(ctx, table, hp.inkling_relative_extent_max() + 1 - table->ne[0], 0, 0, 0);
    auto sinks = ggml_reshape_3d(ctx, llm.llm_build_lora_mm(llm.lctx, ctx, layer.attn_sink_proj, uq), 1, heads, tokens);
    table = ggml_add(ctx, table, ggml_scale(ctx, sinks, -1.0f));
    auto zero = ggml_fill(ctx, ggml_new_tensor_1d(ctx, GGML_TYPE_F32, heads), 0.0f);
    ggml_tensor * result;
    if (llm.flash_attn) {
        auto v_cache = kv.v_l[il];
        const size_t row = ggml_row_size(v_cache->type, dv * kv_heads);
        auto values = ggml_view_3d(ctx, v_cache, dv, count, kv_heads, row, ggml_row_size(v_cache->type, dv), offset * row);
        auto pos_k = compact ? llm.lctx.inp_pos_k_swa : llm.lctx.inp_pos_k;
        result = ggml_flash_attn_ext(ctx, q, keys, values, mask, scale, 0.0f, 0.0f);
        ggml_flash_attn_ext_add_rel_bias(result, table, llm.lctx.inp_pos_q, pos_k);
        ggml_flash_attn_ext_add_sinks(result, zero);
        const int n_swa = hp.swa_layers[il] ? hp.n_swa : 0;
        if (n_swa > 0 && !kv.cells_disordered && llm.cparams.n_seq_max == 1) {
            result->op_params[4] = n_swa;
        }
        llm.cb(result, "fa", il);
    } else {
        auto kq = ggml_mul_mat(ctx, keys, q);
        ggml_mul_mat_set_prec(kq, GGML_PREC_F32);
        kq = ggml_scale(ctx, kq, scale);
        auto indices = compact ? llm.lctx.inp_rel_pos_swa : llm.lctx.inp_rel_pos;
        auto bias = ggml_get_rows(ctx, ggml_cont(ctx, ggml_transpose(ctx, table)), indices);
        bias = ggml_cont(ctx, ggml_permute(ctx, bias, 2, 0, 1, 3));
        auto probs = ggml_soft_max_ext(ctx, ggml_add(ctx, kq, bias), mask, 1.0f, 0.0f);
        ggml_soft_max_add_sinks(probs, zero);

        auto values = ggml_cont(ctx, ggml_permute(ctx, cache_view(kv.v_l[il], dv), 1, 2, 0, 3));
        result = ggml_mul_mat(ctx, values, probs);
        result = ggml_cont(ctx, ggml_permute(ctx, result, 0, 2, 1, 3));
    }
    if (v_hadamard > 0) {
        result = ggml_hadamard(ctx, result, v_hadamard);
    }
    result = ggml_reshape_2d(ctx, result, dv * heads, tokens);
    llm.cb(result, "kqv_merged_cont", il);
    auto gated = ggml_fused_mul_unary(ctx, llm.llm_build_lora_mm(llm.lctx, ctx, layer.wqkv_gate, uq), result, GGML_UNARY_OP_SIGMOID);
    return inkling_conv(llm, gf, il, llm.llm_build_lora_mm(llm.lctx, ctx, layer.wo, gated), (dk + dv) * kv_heads, layer.attn_post_conv);
}

ggml_cgraph * llm_build_context::build_inkling2_loop() {
    ggml_cgraph * gf = new_graph_custom();

    const auto first = hparams.swa_layers.begin();
    const auto last = first + n_layer;
    const bool has_full = std::find(first, last, 0u) != last;
    const bool has_swa = std::any_of(first, last, [](uint32_t swa) { return swa != 0; });
    const bool compact = kv_self.any_compacted();
    lctx.inp_scale = has_full ? build_input_scale(n_tokens) : nullptr;
    auto mask = has_full ? build_inp_KQ_mask() : nullptr;
    ggml_tensor * swa_mask = nullptr;
    if (has_swa) {
        swa_mask = compact ? build_swa_mask_for_graph(hparams.n_swa, true) : build_inp_KQ_mask_swa();
    }
    auto input = [&](int width, int rows, const char * name) {
        auto ids = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, width, rows);
        cb(ids, name, -1);
        ggml_set_input(ids);
        return ids;
    };
    auto full_width = mask ? mask : swa_mask;
    if (flash_attn) {
        lctx.inp_pos_q = input(full_width->ne[1], 1, "inp_pos_q");
        lctx.inp_pos_k = has_full || !compact ? input(full_width->ne[0], 1, "inp_pos_k") : nullptr;
        lctx.inp_pos_k_swa = compact ? input(swa_mask->ne[0], 1, "inp_pos_k_swa") : nullptr;
    } else {
        lctx.inp_rel_pos = has_full || !compact ? input(full_width->ne[0], n_tokens, "inp_rel_pos") : nullptr;
        lctx.inp_rel_pos_swa = compact ? input(swa_mask->ne[0], n_tokens, "inp_rel_pos_swa") : nullptr;
    }

    auto inp_out_ids = n_tokens > 1 ? build_inp_out_ids() : nullptr;
    auto x = llm_build_inp_embd(ctx0, lctx, hparams, batch, model.tok_embd, cb);
    if (batch.token) {
        x = ggml_fused_rms_norm(ctx0, x, model.tok_norm, norm_rms_eps);
    }
    inkling_stack blocks, pre_blocks, bank;
    ggml_tensor * pre = nullptr;
    for (int il = 0; il < n_layer; ++il) {
        const uint32_t ip = hparams.phys_layer(il);
        const auto & layer = model.layers[il];
        if (ip == hparams.loop_start) {
            if (pre) {
                bank = inkling_sources(ctx0, blocks, x, (int) pre_blocks.rows.size() + 1);
                blocks = pre_blocks;
                x = pre;
            } else {
                pre = x;
                pre_blocks = blocks;
            }
        }
        ggml_tensor * messages = nullptr;
        if (bank.keys && ip < hparams.loop_end) {
            auto query = layer.loop_msg_query;
            if (model.loop_res_query) {
                const auto row = 2 * (ip - hparams.loop_start);
                auto c = model.loop_res_query;
                auto k = model.loop_kv_res_query;
                auto q = ggml_view_1d(ctx0, c, n_embd, row * c->nb[1]);
                auto kv = ggml_view_2d(ctx0, k, n_embd, 2, k->nb[1], row * k->nb[1]);
                auto mlp = ggml_view_1d(ctx0, c, n_embd, (row + 1) * c->nb[1]);
                query = ggml_concat(ctx0, ggml_concat(ctx0, q, kv, 1), mlp, 1);
            }
            messages = inkling_mix(ctx0, bank, nullptr, query);
        }
        auto blend = [&](ggml_tensor * dest, int stream) {
            return messages ? inkling_control(ctx0, dest, inkling_row(ctx0, messages, stream), layer.loop_ctrl_down[stream],
                    layer.loop_ctrl_forget[stream], layer.loop_ctrl_gain[stream], norm_rms_eps) : dest;
        };

        const bool full = !hparams.swa_layers[il];
        ggml_tensor * q = x;
        ggml_tensor * k = x;
        ggml_tensor * v = x;
        if (ip != 0) {
            auto mixed = inkling_mix(ctx0, blocks, x, layer.attn_res_query);
            q = inkling_row(ctx0, mixed, 0);
            k = full ? inkling_row(ctx0, mixed, 1) : q;
            v = full ? inkling_row(ctx0, mixed, 2) : q;
        }
        q = ggml_fused_rms_norm(ctx0, blend(q, 0), full ? layer.attn_q_inp_norm : layer.attn_norm, norm_rms_eps);
        if (full || messages) {
            k = ggml_fused_rms_norm(ctx0, blend(k, 1), full ? layer.attn_k_inp_norm : layer.attn_norm, norm_rms_eps);
            v = ggml_fused_rms_norm(ctx0, blend(v, 2), full ? layer.attn_v_inp_norm : layer.attn_norm, norm_rms_eps);
        } else {
            k = v = q;
        }
        auto residual = inkling_attention(*this, gf, il, q, k, v, full ? mask : swa_mask, full ? lctx.inp_scale : nullptr);
        if (ip % hparams.attn_res_block_size == 0) {
            inkling_push(ctx0, blocks, x);
        } else {
            residual = ggml_add(ctx0, x, residual);
        }

        auto ffn_input = inkling_row(ctx0, inkling_mix(ctx0, blocks, residual, layer.ffn_res_query), 0);
        ffn_input = ggml_fused_rms_norm(ctx0, blend(ffn_input, 3), layer.ffn_norm, norm_rms_eps);
        auto ffn = llm_build_moe_ffn(ctx0, lctx, ffn_input, layer.ffn_gate_inp,
                nullptr, nullptr, layer.ffn_down_exps, layer.ffn_exp_probs_b,
                n_expert, n_expert_used < n_expert ? hparams.n_expert_used_layer(il) : n_expert_used, LLM_FFN_SITU, true, true, hparams.expert_weights_scale,
                LLM_EXPERT_GATING_FUNC_SIGMOID, cb, il, gf, false, layer.ffn_up_gate_exps);
        const int offset = (hparams.n_embd_k_gqa(il) + hparams.n_embd_v_gqa(il) + hparams.n_embd);
        x = ggml_add(ctx0, residual, inkling_conv(*this, gf, il, ffn, offset, layer.ffn_post_conv));
        x = lctx.cvec.apply_to(ctx0, x, il);
        cb(x, "l_out", il);
    }

    if (n_outputs == 0 && !cparams.embeddings) {
        ggml_build_forward_expand(gf, x);
        return gf;
    }
    x = inkling_row(ctx0, inkling_mix(ctx0, blocks, x, model.output_res_query), 0);
    if (inp_out_ids) {
        x = ggml_get_rows(ctx0, x, inp_out_ids);
    }
    auto logits = build_output(lctx, ctx0, x, model.output, model.output_norm, cb, true);
    const float cap = hparams.f_final_logit_softcapping;
    logits = ggml_softcap(ctx0, logits, 1.0f / cap, cap);
    cb(logits, "result_output", -1);
    ggml_build_forward_expand(gf, logits);
    return gf;
}

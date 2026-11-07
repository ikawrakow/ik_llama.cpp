#include "../llama-build-context.h"
#include "../llama-model.h"
#include "../llama-context.h"

// Nemotron-H: Mamba-2 + NoPE attention + relu^2 FFN, single sequence per context.
// Ported from upstream llama.cpp (src/models/nemotron-h.cpp, mamba-base.cpp).

ggml_tensor * llm_build_context::build_nemotron_mamba2(ggml_cgraph * gf, ggml_tensor * cur, ggml_tensor * ssm_ids, int il) {
    const auto & layer = model.layers[il];

    const int64_t d_conv   = hparams.ssm_d_conv;
    const int64_t d_inner  = hparams.ssm_d_inner;
    const int64_t d_state  = hparams.ssm_d_state;
    const int64_t n_head   = hparams.ssm_dt_rank;   // Mamba-2: dt_rank == n_ssm_heads
    const int64_t head_dim = d_inner / n_head;
    const int64_t n_group  = hparams.ssm_n_group;
    const int64_t n_seqs   = ssm_ids->ne[0];
    const int64_t n_seq_tokens = n_tokens / n_seqs;
    const int64_t ch       = d_inner + 2 * n_group * d_state; // wide conv channel count

    if (n_seqs != 1) {
        // llama_kv_cache_init already rejects n_seq_max > 1 for this arch with a clean message;
        // this is the backstop for a context built some other way (a throw would escape through
        // llama_build_graph's extern "C" boundary and terminate without the message)
        GGML_ABORT("Nemotron-H: the Mamba-2 path supports a single sequence per batch "
                   "(use -np 1 / n_seq_max = 1; multi-sequence state slots are not implemented)");
    }
    GGML_ASSERT(n_seq_tokens == n_tokens);
    GGML_ASSERT(d_inner % n_head  == 0);
    GGML_ASSERT(d_inner % n_group == 0);

    // recurrent state: rolling conv state in k_l, SSM state in s_l
    ggml_tensor * conv_states_all = kv_self.k_l[il]; // 2D {embd_r, slots}
    ggml_tensor * ssm_states_all  = kv_self.s_l[il]; // 2D {embd_s, slots}
    GGML_ASSERT(conv_states_all != nullptr && ssm_states_all != nullptr);
    const int64_t state_slots = ssm_states_all->ne[1];
    GGML_ASSERT((int64_t) conv_states_all->ne[0] == (d_conv - 1) * ch);
    GGML_ASSERT((int64_t) conv_states_all->ne[1] == state_slots);
    GGML_ASSERT((int64_t) ssm_states_all->ne[0] == d_state * d_inner);
    if (state_slots != 1) {
        GGML_ABORT("Nemotron-H: the recurrent state cache has %d slots but the Mamba-2 graph "
                   "implements one (use -np 1 / n_seq_max = 1)", (int) state_slots);
    }

    // conv state -> {d_conv - 1, ch, n_seqs}
    ggml_tensor * conv = ggml_reshape_3d(ctx0, conv_states_all, d_conv - 1, ch, n_seqs);

    // d_in_proj = 2 * d_inner + 2 * n_group * d_state + n_head
    // {n_embd, d_in_proj} @ {n_embd, n_tokens} => {d_in_proj, n_tokens}
    ggml_tensor * zxBCdt = llm_build_lora_mm(lctx, ctx0, layer.ssm_in, cur);
    cb(zxBCdt, "ssm_in", il);

    // split into z / xBC / dt
    ggml_tensor * z   = ggml_view_2d(ctx0, zxBCdt, d_inner, n_tokens, zxBCdt->nb[1], 0);
    ggml_tensor * xBC = ggml_view_3d(ctx0, zxBCdt, ch, n_tokens, 1, zxBCdt->nb[1], zxBCdt->nb[1]*n_tokens,
                                     d_inner * ggml_element_size(zxBCdt));
    ggml_tensor * dt  = ggml_view_2d(ctx0, zxBCdt, n_head, n_tokens, zxBCdt->nb[1],
                                     (2 * d_inner + 2 * n_group * d_state) * ggml_element_size(zxBCdt));

    // conv
    {
        // => {d_conv - 1 + n_seq_tokens, ch, n_seqs}
        ggml_tensor * conv_x = ggml_concat(ctx0, conv, ggml_transpose(ctx0, xBC), 0);

        // copy the last (d_conv - 1) columns back into the state cache
        ggml_tensor * last_conv = ggml_view_3d(ctx0, conv_x, d_conv - 1, ch, n_seqs, conv_x->nb[1], conv_x->nb[2],
                                               n_seq_tokens * conv_x->nb[0]);
        ggml_build_forward_expand(gf, ggml_cpy(ctx0, last_conv, ggml_view_1d(ctx0, conv_states_all, (d_conv - 1)*ch*n_seqs, 0)));

        // 1D convolution over the sliding window (Mamba-2 layout)
        xBC = ggml_ssm_conv_m2(ctx0, conv_x, layer.ssm_conv1d);
        cb(xBC, "ssm_conv", il);

        // bias
        if (layer.ssm_conv1d_b) {
            xBC = ggml_add(ctx0, xBC, layer.ssm_conv1d_b);
        }

        xBC = ggml_silu(ctx0, xBC);
        cb(xBC, "ssm_xBC", il);
    }

    // ssm
    {
        ggml_tensor * x = ggml_view_4d(ctx0, xBC, head_dim, n_head, n_seq_tokens, n_seqs,
                                       head_dim * xBC->nb[0], xBC->nb[1], xBC->nb[2], 0);
        ggml_tensor * B = ggml_view_4d(ctx0, xBC, d_state, n_group, n_seq_tokens, n_seqs,
                                       d_state * xBC->nb[0], xBC->nb[1], xBC->nb[2], d_inner * ggml_element_size(xBC));
        ggml_tensor * C = ggml_view_4d(ctx0, xBC, d_state, n_group, n_seq_tokens, n_seqs,
                                       d_state * xBC->nb[0], xBC->nb[1], xBC->nb[2], (d_inner + n_group*d_state) * ggml_element_size(xBC));

        // {n_head, n_seq_tokens, n_seqs}
        dt = ggml_reshape_3d(ctx0, ggml_cont(ctx0, dt), n_head, n_seq_tokens, n_seqs);
        dt = ggml_add(ctx0, dt, layer.ssm_dt_b);

        // states: {embd_s, slots} -> {d_state, head_dim, n_head, slots}
        ggml_tensor * ssm = ggml_reshape_4d(ctx0, ssm_states_all, d_state, head_dim, n_head, state_slots);

        struct ggml_tensor * y_ssm = ggml_ssm_scan_m2(ctx0, ssm, x, dt, layer.ssm_a, B, C, ssm_ids, 1);
        cb(y_ssm, "ssm_scan", il);

        // write the new SSM state back into the cache slot
        const int64_t d_ssm = d_state * d_inner;
        ggml_build_forward_expand(gf, ggml_cpy(ctx0,
                ggml_view_1d(ctx0, y_ssm, d_ssm, ggml_nelements(x) * ggml_element_size(x)),
                ggml_view_1d(ctx0, ssm_states_all, d_ssm, 0)));

        // y = first nelements(x) values of y_ssm
        ggml_tensor * y = ggml_view_4d(ctx0, y_ssm, head_dim, n_head, n_seq_tokens, n_seqs,
                                       x->nb[1], n_head * x->nb[1], n_seq_tokens * n_head * x->nb[1], 0);

        // y = y + x * D
        y = ggml_add(ctx0, y, ggml_mul(ctx0, x, layer.ssm_d));

        // y = swiglu(z, y)
        y = ggml_swiglu_split(ctx0, ggml_reshape_4d(ctx0, ggml_cont(ctx0, z), head_dim, n_head, n_seq_tokens, n_seqs), y);

        // grouped RMS norm.
        // The weight is one gamma per (group, channel), i.e. {d_inner/n_group, n_group}, so the
        // generic fused norm (which needs a single row of gammas) falls back to RMS_NORM + MUL.
        // Fold the group axis into the row width and use the grouped fused kernel instead, the
        // same way the qwen4exp graph does: one GGML_OP_FUSED_RMS_NORM per layer.
        if (layer.ssm_norm) {
            y = ggml_reshape_2d(ctx0, y, d_inner, n_seq_tokens * n_seqs);
            if (n_group > 1) {
                ggml_tensor * gamma = ggml_reshape_1d(ctx0, layer.ssm_norm, d_inner);
                y = ggml_fused_grouped_rms_norm(ctx0, y, gamma, hparams.f_norm_rms_eps, (int) n_group);
            } else {
                y = llm_build_norm(ctx0, y, hparams, layer.ssm_norm, NULL, LLM_NORM_RMS, cb, il);
            }
            cb(y, "ssm_norm", il);
            y = ggml_reshape_4d(ctx0, y, d_inner / n_group, n_group, n_seq_tokens, n_seqs);
        }

        y = ggml_reshape_2d(ctx0, y, d_inner, n_seq_tokens * n_seqs);

        // {d_inner, n_embd} @ {d_inner, n_tokens} => {n_embd, n_tokens}
        cur = llm_build_lora_mm(lctx, ctx0, layer.ssm_out, y);
    }

    return cur;
}

ggml_cgraph * llm_build_context::build_nemotron_h() {
    ggml_cgraph * gf = new_graph_custom();

    struct ggml_tensor * cur;
    struct ggml_tensor * inpL;

    // {n_embd, n_tokens}
    inpL = llm_build_inp_embd(ctx0, lctx, hparams, batch, model.tok_embd, cb);
    ggml_build_forward_expand(gf, inpL);

    // source state ids for the ssm scan: one slot, filled with 0 at runtime
    lctx.inp_ssm_ids = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 1);
    cb(lctx.inp_ssm_ids, "inp_ssm_ids", -1);
    ggml_set_input(lctx.inp_ssm_ids);

    // KQ_mask for the attention layers
    struct ggml_tensor * KQ_mask = build_inp_KQ_mask();

    // output row selection (skipped for single-token decodes, matching the other builders)
    struct ggml_tensor * inp_out_ids = (!llm_arch_requires_all_graph_output_rows(model.arch) &&
            n_tokens > 1 && n_outputs < n_tokens) ? build_inp_out_ids() : nullptr;

    for (int il = 0; il < n_layer; ++il) {
        struct ggml_tensor * inpSA = inpL;
        const bool is_attention = !hparams.is_recurrent(il) && hparams.n_ff(il) == 0;
        struct ggml_tensor * attn_out_ids = is_attention && il == n_layer - 1 ? inp_out_ids : nullptr;

        if (is_attention) {
            // attention layer without positional encoding: the shared builder covers the norm,
            // the QKV projections (and biases), the KV-cache write/read and the output projection
            const int64_t n_embd_head = hparams.n_embd_head_k(il);
            const float kq_scale_l = hparams.f_attention_scale == 0.0f
                ? 1.0f / sqrtf(float(n_embd_head))
                : hparams.f_attention_scale;
            cur = build_std_attention(gf, model.layers[il].attn_norm, inpL,
                    /*inp_pos*/ nullptr, attn_out_ids, /*rope_factors*/ nullptr,
                    KQ_mask, model.layers[il].attn_sinks,
                    /*inp_attn_scale*/ nullptr, kq_scale_l, /*f_attn_scale*/ 0.0f,
                    /*n_swa*/ 0, il, /*do_rope*/ false, /*add_graph_split*/ false,
                    /*add_input*/ false);
            cb(cur, "attn_out", il);
        } else {
            // pre-block norm, shared by the SSM and FFN layer kinds
            cur = llm_build_norm(ctx0, inpL, hparams, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, cb, il);
            cb(cur, "attn_norm", il);

            if (hparams.is_recurrent(il)) {
                // SSM (mamba2) layer
                cur = build_nemotron_mamba2(gf, cur, lctx.inp_ssm_ids, il);
            } else {
                // FFN layer: dense relu^2, or routed relu^2 experts (+ shared expert)
                if (model.layers[il].ffn_gate_inp != nullptr) {
                    const auto & layer = model.layers[il];

                    ggml_tensor * inp_emb = cur;
                    ggml_tensor * inp_latent = cur;
                    if (layer.ffn_latent_down) {
                        inp_latent = llm_build_lora_mm(lctx, ctx0, layer.ffn_latent_down, cur);
                        cb(inp_latent, "ffn_moe_latent_down", il);
                    }

                    // the router reads the full-width input; the experts may read the latent
                    ggml_tensor * router_logits = nullptr;
                    ggml_tensor * gate_inp = layer.ffn_gate_inp;
                    if (layer.ffn_latent_down) {
                        router_logits = llm_build_lora_mm(lctx, ctx0, layer.ffn_gate_inp, cur);
                        gate_inp = nullptr;
                    }

                    const int64_t n_expert_il      = layer.ffn_up_exps->ne[2]; // per-layer expert count
                    const int64_t n_expert_used_il = (int64_t) hparams.n_expert_used_layer(il);

                    ggml_tensor * moe_out = llm_build_moe_ffn(ctx0, lctx, inp_latent,
                            gate_inp,            nullptr,
                            layer.ffn_up_exps,   nullptr,
                            nullptr,             nullptr, // no gate — relu^2(up)
                            layer.ffn_down_exps, nullptr,
                            layer.ffn_exp_probs_b,
                            n_expert_il, n_expert_used_il,
                            LLM_FFN_RELU_SQR,
                            hparams.expert_weights_norm,
                            hparams.expert_weights_scale > 0.0f && hparams.expert_weights_scale != 1.0f,
                            hparams.expert_weights_scale,
                            LLM_EXPERT_GATING_FUNC_SIGMOID,
                            cb, il, gf, false,
                            nullptr, nullptr,
                            router_logits, nullptr, nullptr);
                    cb(moe_out, "ffn_moe_out", il);

                    if (layer.ffn_latent_up) {
                        moe_out = llm_build_lora_mm(lctx, ctx0, layer.ffn_latent_up, moe_out);
                        cb(moe_out, "ffn_moe_latent_up", il);
                    }

                    if (layer.ffn_up_shexp && layer.ffn_down_shexp) {
                        ggml_tensor * shexp = llm_build_ffn(ctx0, lctx, /*ffn_norm*/ nullptr, inp_emb,
                                layer.ffn_up_shexp,   nullptr, nullptr,
                                nullptr,              nullptr, nullptr,
                                layer.ffn_down_shexp, nullptr, nullptr,
                                nullptr,
                                LLM_FFN_RELU_SQR, LLM_FFN_PAR, cb, il);
                        cb(shexp, "ffn_shexp", il);
                        cur = ggml_add(ctx0, moe_out, shexp);
                    } else {
                        cur = moe_out;
                    }
                    cb(cur, "ffn_out", il);
                } else {
                    // dense FFN: up/down only, relu^2 (no gate)
                    cur = llm_build_ffn(ctx0, lctx, /*ffn_norm*/ nullptr, cur,
                            model.layers[il].ffn_up,   model.layers[il].ffn_up_b,   nullptr,
                            nullptr,                    nullptr,                    nullptr,
                            model.layers[il].ffn_down, model.layers[il].ffn_down_b, nullptr,
                            nullptr,
                            LLM_FFN_RELU_SQR, LLM_FFN_PAR, cb, il);
                    cb(cur, "ffn_out", il);
                }
            }

        } // !is_attention

        if (il == n_layer - 1 && inp_out_ids != nullptr) {
            // skip computing output for unused tokens (the attention builder already applied
            // attn_out_ids inside its own graph)
            if (!is_attention) {
                cur = ggml_get_rows(ctx0, cur, inp_out_ids);
            }
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }

        // residual
        cur = ggml_add(ctx0, cur, inpSA);
        cur = lctx.cvec.apply_to(ctx0, cur, il);
        cb(cur, "l_out", il);

        // input for next layer
        inpL = cur;
    }

    // final rmsnorm
    cur = llm_build_norm(ctx0, inpL, hparams, model.output_norm, NULL, LLM_NORM_RMS, cb, -1);
    cb(cur, "result_norm", -1);

    // lm_head
    cur = llm_build_lora_mm(lctx, ctx0, model.output, cur);
    cb(cur, "result_output", -1);

    ggml_build_forward_expand(gf, cur);

    return gf;
}

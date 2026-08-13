#include "llama-kda.h"
#include "llama-hparams.h"
#include "llama-cparams.h"
#include "llama-model.h"
#include "llama-context.h"

#include "ggml.h"

static std::pair<ggml_tensor *, ggml_tensor *> build_kda_qkvz(llama_context & lctx, ggml_context * ctx0,
        ggml_tensor * wq, ggml_tensor * wk, ggml_tensor * wv, ggml_tensor * ssm_g_a,
        ggml_tensor * input, int il, const llm_build_cb & cb, ggml_cgraph * gf) {
    auto q = llm_build_context::llm_build_lora_mm(lctx, ctx0, wq, input);
    auto k = llm_build_context::llm_build_lora_mm(lctx, ctx0, wk, input);
    auto v = llm_build_context::llm_build_lora_mm(lctx, ctx0, wv, input);
    auto z = llm_build_context::llm_build_lora_mm(lctx, ctx0, ssm_g_a, input);
    cb(q, "q", il);
    cb(k, "k", il);
    cb(v, "v", il);
    cb(z, "z", il);

    auto qkv = ggml_concat(ctx0, q, k, 0);
    qkv = ggml_concat(ctx0, qkv, v, 0);
    cb(qkv, "qkv_mixed", il);
    ggml_build_forward_expand(gf, qkv);
    ggml_build_forward_expand(gf, z);
    return { qkv, z };
}

static std::pair<ggml_tensor *, ggml_tensor *> build_kda_beta_gate(llama_context & lctx, ggml_context * ctx0,
        ggml_tensor * ssm_beta, ggml_tensor * ssm_f_a, ggml_tensor * ssm_dt_b, ggml_tensor * ssm_a,
        ggml_tensor * input, int64_t head_dim, int64_t n_head, float lower_bound,
        int il, const llm_build_cb & cb, ggml_cgraph * gf) {
    const int64_t n_tok = input->ne[1];

    auto beta = llm_build_context::llm_build_lora_mm(lctx, ctx0, ssm_beta, input);
    beta = ggml_reshape_4d(ctx0, beta, n_head, 1, n_tok, 1);
    cb(beta, "beta", il);

    auto raw = llm_build_context::llm_build_lora_mm(lctx, ctx0, ssm_f_a, input);
    raw = ggml_reshape_4d(ctx0, raw, head_dim, n_head, n_tok, 1);
    cb(raw, "decay_raw", il);

    auto dt = ggml_reshape_4d(ctx0, ssm_dt_b, head_dim, n_head, 1, 1);
    auto a  = ggml_reshape_4d(ctx0, ssm_a, 1, n_head, 1, 1);
    auto log_decay = ggml_add(ctx0, raw, dt);
    log_decay = ggml_mul(ctx0, log_decay, a);
    log_decay = ggml_sigmoid(ctx0, log_decay);
    log_decay = ggml_scale(ctx0, log_decay, lower_bound);
    cb(log_decay, "log_decay", il);

    ggml_build_forward_expand(gf, beta);
    ggml_build_forward_expand(gf, log_decay);
    return { beta, log_decay };
}

static ggml_tensor * build_kda_conv(ggml_context * ctx0,
        ggml_tensor * ssm_conv1d_q, ggml_tensor * ssm_conv1d_k, ggml_tensor * ssm_conv1d_v) {
    const int64_t d_conv = ssm_conv1d_q->ne[0];
    auto q = ggml_reshape_2d(ctx0, ssm_conv1d_q, d_conv, ggml_nrows(ssm_conv1d_q));
    auto k = ggml_reshape_2d(ctx0, ssm_conv1d_k, d_conv, ggml_nrows(ssm_conv1d_k));
    auto v = ggml_reshape_2d(ctx0, ssm_conv1d_v, d_conv, ggml_nrows(ssm_conv1d_v));
    auto qkv = ggml_concat(ctx0, q, k, 1);
    return ggml_concat(ctx0, qkv, v, 1);
}

static ggml_tensor * build_kda_gated_output(llama_context & lctx, ggml_context * ctx0, ggml_tensor * ssm_norm, ggml_tensor * ssm_out, ggml_tensor * output, ggml_tensor * z,
        int64_t head_v_dim, int64_t num_v_heads, int64_t n_tok, int il, const llm_build_cb & cb) {

    ggml_tensor * attn_out_2d = ggml_reshape_2d(ctx0, output, head_v_dim, num_v_heads * n_tok);
    ggml_tensor * z_2d        = ggml_reshape_2d(ctx0, z,      head_v_dim, num_v_heads * n_tok);

    ggml_tensor * attn_out_norm = llm_build_context::llm_build_norm(ctx0, attn_out_2d, lctx.model.hparams, ssm_norm, nullptr, LLM_NORM_RMS, cb, il);
    cb(attn_out_norm, "attn_rms_norm", il);
    attn_out_norm = ggml_mul(ctx0, ggml_sigmoid(ctx0, z_2d), attn_out_norm);
    cb(attn_out_norm, "attn_out_norm", il);

    ggml_tensor * final_output = ggml_reshape_2d(ctx0, attn_out_norm, head_v_dim*num_v_heads, n_tok);
    cb(final_output, "final_output", il);

    ggml_tensor * out = llm_build_context::llm_build_lora_mm(lctx, ctx0, ssm_out, final_output);
    cb(out, "linear_attn_out", il);

    return ggml_reshape_2d(ctx0, out, lctx.model.hparams.n_embd, n_tok);
}

ggml_tensor * delta_net::build_layer_attn_kda_core(ggml_context * ctx0, ggml_cgraph * gf,
        ggml_tensor * delta_input, ggml_tensor * inp_s_seq_qnext, ggml_tensor * inp_out_ids,
        uint32_t state_seq_id_local, bool reset_state_local, int il, const llm_build_cb & cb) const {
    const int64_t n_tok = delta_input->ne[1];
    const int64_t head_dim = lctx.model.hparams.ssm_d_state;

    auto & model   = lctx.model;
    auto & hparams = model.hparams;
    auto & kv_self = lctx.kv_self;
    auto & layer   = model.layers[il];

    if (model.split_mode == LLAMA_SPLIT_MODE_GRAPH && kv_self.s_l[il]->extra) {
        auto split_s_l = (ggml_split_tensor_t *) kv_self.s_l[il]->extra;
        GGML_ASSERT(split_s_l && split_s_l->n_device > 1);

        std::vector<ggml_tensor *> results(split_s_l->n_device, nullptr);
        bool input_added = false;
        for (int id = 0; id < split_s_l->n_device; ++id) {
            if (!split_s_l->splits[id]) {
                continue;
            }

            auto split = [id](ggml_tensor * tensor) {
                auto data = (ggml_split_tensor_t *) tensor->extra;
                GGML_ASSERT(data && data->splits[id]);
                return data->splits[id];
            };

            const int il_cb = 1000 * il + id;
            auto input = llm_build_context::get_input_tensor_sm_graph(ctx0, delta_input, id);
            auto cur = llm_build_context::llm_build_norm(ctx0, input, hparams,
                    split(layer.attn_norm), nullptr, LLM_NORM_RMS, cb, il_cb);

            auto ssm_out = split(layer.ssm_out);
            const int64_t n_head = ssm_out->ne[0] / head_dim;
            auto [qkv_mixed, z] = build_kda_qkvz(lctx, ctx0,
                    split(layer.wq), split(layer.wk), split(layer.wv), split(layer.ssm_g_a),
                    cur, il_cb, cb, gf);
            auto [beta, log_decay] = build_kda_beta_gate(lctx, ctx0,
                    split(layer.ssm_beta), split(layer.ssm_f_a), split(layer.ssm_dt_b), split(layer.ssm_a),
                    cur, head_dim, n_head, hparams.kda_gate_lower_bound, il_cb, cb, gf);
            auto conv = build_kda_conv(ctx0,
                    split(layer.ssm_conv1d_q), split(layer.ssm_conv1d_k), split(layer.ssm_conv1d_v));

            ggml_tensor * per_step_ckpt = nullptr;
            if (save_per_step_states && il < (int) kv_self.ckpt.per_step_ssm.size()) {
                per_step_ckpt = kv_self.ckpt.per_step_ssm[il][id];
            }
            auto per_step_conv = save_per_step_states && il < (int) kv_self.ckpt.per_step_conv.size() &&
                                 id < (int) kv_self.ckpt.per_step_conv[il].size()
                               ? kv_self.ckpt.per_step_conv[il][id] : nullptr;

            const uint32_t qnext_state_slots = split_s_l->splits[id]->ne[1];
            auto output = build_qkv(ctx0, split_s_l->splits[id], conv, qkv_mixed,
                    inp_s_seq_qnext, beta, log_decay,
                    head_dim, n_head, head_dim, n_head, hparams.ssm_d_conv,
                    state_seq_id_local, qnext_state_slots, reset_state_local,
                    hparams.f_norm_rms_eps, 1, il_cb, cb, gf, per_step_ckpt, per_step_conv);

            auto gated_output = build_kda_gated_output(lctx, ctx0, split(layer.ssm_norm), ssm_out, output, z,
                    head_dim, n_head, n_tok, il_cb, cb);
            if (inp_out_ids) {
                gated_output = ggml_get_rows(ctx0, gated_output, inp_out_ids);
            }
            if (!input_added) {
                if (inp_out_ids) {
                    input = ggml_get_rows(ctx0, input, inp_out_ids);
                }
                gated_output = ggml_add(ctx0, gated_output, input);
                input_added = true;
            }
            if (gated_output->ne[1] > 32 && lctx.cparams.reduce_type != GGML_TYPE_F32) {
                gated_output = ggml_cast(ctx0, gated_output, lctx.cparams.reduce_type);
            }
            ggml_build_forward_expand(gf, gated_output);
            results[id] = gated_output;
        }

        auto output = ggml_reduce(ctx0, results.data(), split_s_l->n_device, GGML_OP_ADD);
        ggml_build_forward_expand(gf, output);
        return output;
    }

    const uint32_t qnext_state_slots = llm_build_context::llama_kv_qnext_state_slots(kv_self);
    int idx = model.default_layer_device[il];
    auto input = delta_input;
    if (input->op == GGML_OP_REDUCE) {
        const int idx_s_l = ggml_backend_sched_get_backend_idx(lctx.sched, kv_self.s_l[il]->buffer);
        if (idx_s_l >= 0) {
            idx = idx_s_l;
        }
        if (input->src[idx]) {
            input->view_src = input->src[idx];
        }
    }

    auto norm = layer.attn_norm->extra
              ? ((ggml_split_tensor_t *) layer.attn_norm->extra)->splits[idx]
              : layer.attn_norm;
    auto cur = llm_build_context::llm_build_norm(ctx0, input, hparams, norm, nullptr, LLM_NORM_RMS, cb, il);

    const int64_t n_head = hparams.ssm_dt_rank;
    auto [qkv_mixed, z] = build_kda_qkvz(lctx, ctx0,
            layer.wq, layer.wk, layer.wv, layer.ssm_g_a, cur, il, cb, gf);
    auto [beta, log_decay] = build_kda_beta_gate(lctx, ctx0,
            layer.ssm_beta, layer.ssm_f_a, layer.ssm_dt_b, layer.ssm_a,
            cur, head_dim, n_head, hparams.kda_gate_lower_bound, il, cb, gf);
    auto conv = build_kda_conv(ctx0, layer.ssm_conv1d_q, layer.ssm_conv1d_k, layer.ssm_conv1d_v);

    ggml_tensor * per_step_ckpt = nullptr;
    if (save_per_step_states && il < (int) kv_self.ckpt.per_step_ssm.size()) {
        per_step_ckpt = kv_self.ckpt.per_step_ssm[il].front();
    }
    auto per_step_conv = save_per_step_states && il < (int) kv_self.ckpt.per_step_conv.size() &&
                         !kv_self.ckpt.per_step_conv[il].empty()
                       ? kv_self.ckpt.per_step_conv[il].front() : nullptr;

    auto output = build_qkv(ctx0, kv_self.s_l[il], conv, qkv_mixed,
            inp_s_seq_qnext, beta, log_decay,
            head_dim, n_head, head_dim, n_head, hparams.ssm_d_conv,
            state_seq_id_local, qnext_state_slots, reset_state_local,
            hparams.f_norm_rms_eps, 1, il, cb, gf, per_step_ckpt, per_step_conv);
    auto gated_output = build_kda_gated_output(lctx, ctx0, layer.ssm_norm, layer.ssm_out, output, z,
            head_dim, n_head, n_tok, il, cb);

    if (inp_out_ids) {
        gated_output = ggml_get_rows(ctx0, gated_output, inp_out_ids);
        input = ggml_get_rows(ctx0, input, inp_out_ids);
    }
    output = ggml_add(ctx0, gated_output, input);
    cb(output, "ssm_output", il);
    return output;
}

ggml_tensor * delta_net::build_layer_attn_kda(ggml_context * ctx0, ggml_cgraph * gf,
        ggml_tensor * cur, ggml_tensor * inp_out_ids, int il, const llm_build_cb & cb) const {
    GGML_ASSERT(lctx.inp_s_seq_qnext != nullptr);

    auto & layer = lctx.model.layers[il];
    GGML_ASSERT(lctx.model.hparams.is_recurrent(il));
    GGML_ASSERT(layer.wq && layer.wk && layer.wv);
    GGML_ASSERT(layer.ssm_conv1d_q && layer.ssm_conv1d_k && layer.ssm_conv1d_v);
    GGML_ASSERT(layer.ssm_f_a && layer.ssm_g_a && layer.ssm_beta);
    GGML_ASSERT(layer.ssm_a && layer.ssm_dt_b && layer.ssm_norm && layer.ssm_out);

    if (all_same_seq) {
        const bool reset_state = batch.pos != nullptr && batch.pos[0] == 0;
        return build_layer_attn_kda_core(ctx0, gf, cur, lctx.inp_s_seq_qnext, inp_out_ids,
                token_seq_ids.front(), reset_state, il, cb);
    }

    GGML_ASSERT(has_unique_seq_ids && "bailingmoe3 mixed-sequence batches require unique sequence IDs per token");

    ggml_tensor * out = nullptr;
    for (int64_t i = 0; i < batch.n_tokens; ++i) {
        auto cur_i = ggml_view_2d(ctx0, cur, cur->ne[0], 1, cur->nb[1], (size_t) i * cur->nb[1]);
        auto inp_s_seq_qnext_i = ggml_view_2d(ctx0, lctx.inp_s_seq_qnext, 1, 1,
                lctx.inp_s_seq_qnext->nb[1], (size_t) i * lctx.inp_s_seq_qnext->nb[1]);
        const bool reset_state = batch.pos != nullptr && batch.pos[i] == 0;
        auto out_i = build_layer_attn_kda_core(ctx0, gf, cur_i, inp_s_seq_qnext_i, inp_out_ids,
                (uint32_t) token_seq_ids[i], reset_state, il, cb);
        out = out == nullptr ? out_i : ggml_concat(ctx0, out, out_i, 1);
    }
    return out;
}

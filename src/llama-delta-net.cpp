#include "llama-delta-net.h"
#include "llama-hparams.h"
#include "llama-cparams.h"
#include "llama-model.h"
#include "llama-context.h"

#include "ggml.h"

#include <algorithm>
#include <unordered_set>

#define QWEN3NEXT_CHUNK_SIZE 64

delta_net::delta_net(llama_context & _lctx, const llama_batch & _batch) : lctx(_lctx), batch(_batch) {
    auto & model = lctx.model;
    auto & hparams = model.hparams;

    GGML_ASSERT(batch.n_tokens > 0);
    GGML_ASSERT(hparams.ssm_n_group > 0);
    GGML_ASSERT(hparams.ssm_dt_rank > 0);
    GGML_ASSERT(hparams.ssm_d_conv > 0);
    GGML_ASSERT(hparams.ssm_d_inner % hparams.ssm_dt_rank == 0);

    const int64_t head_k_dim     = hparams.ssm_d_state;
    const int64_t num_k_heads    = hparams.ssm_n_group;
    const int64_t num_v_heads    = hparams.ssm_dt_rank;
    const int64_t head_v_dim     = hparams.ssm_d_inner / num_v_heads;
    const int64_t key_dim        = head_k_dim * num_k_heads;
    const int64_t value_dim      = head_v_dim * num_v_heads;
    const int64_t ssm_state_dim  = head_v_dim * head_v_dim * num_v_heads;
    const int64_t conv_dim       = key_dim * 2 + value_dim;
    const int64_t conv_state_dim = (hparams.ssm_d_conv - 1) * conv_dim;
    const int64_t state_dim      = conv_state_dim + ssm_state_dim;
    GGML_ASSERT(hparams.n_embd_v_s() == (uint32_t) state_dim);

    const bool has_explicit_seq_info = batch.n_seq_id != nullptr && batch.seq_id != nullptr;
    token_seq_ids.resize(batch.n_tokens, 0);
    for (int i = 0; i < batch.n_tokens; ++i) {
        if (has_explicit_seq_info) {
            GGML_ASSERT(batch.n_seq_id[i] > 0 && "qwen3next expects each token to belong to at least one sequence");
            GGML_ASSERT(batch.n_seq_id[i] == 1 && "qwen3next does not support multi-sequence tokens yet");
            token_seq_ids[i] = batch.seq_id[i][0];
        } else {
            token_seq_ids[i] = 0;
        }
    }

    auto seq_id = token_seq_ids[0];
    all_same_seq = std::all_of(token_seq_ids.begin(), token_seq_ids.end(), [seq_id](llama_seq_id s) { return s == seq_id; });

    has_unique_seq_ids = true;
    if (!all_same_seq) {
        std::unordered_set<llama_seq_id> seen;
        seen.reserve(token_seq_ids.size());
        for (auto s : token_seq_ids) {
            if (!seen.insert(s).second) {
                has_unique_seq_ids = false;
                break;
            }
        }
    }

    const uint32_t qnext_state_slots = llm_build_context::llama_kv_qnext_state_slots(lctx.kv_self);
    GGML_ASSERT(qnext_state_slots > 0);

    // Reserve-graph builds may not carry explicit sequence IDs, in which case
    // the fallback sequence slot is 0.
    for (llama_seq_id s : token_seq_ids) {
        GGML_ASSERT(s >= 0);
        GGML_ASSERT((uint32_t) s < qnext_state_slots);
    }

}

delta_net::~delta_net() = default;

std::pair<ggml_tensor *, ggml_tensor *> delta_net::build_delta_net_chunking(ggml_context * ctx0,
                      ggml_tensor * q, ggml_tensor * k, ggml_tensor * v,
                      ggml_tensor * g, ggml_tensor * beta, ggml_tensor * state,
                      ggml_tensor * causal_mask, ggml_tensor * identity,
                      ggml_tensor * diag_mask, int il, const llm_build_cb & cb) {

    const int64_t S_k      = q->ne[0];
    const int64_t H_k      = q->ne[1];
    const int64_t n_tokens = q->ne[2];
    const int64_t n_seqs   = q->ne[3];

    const int64_t S_v = v->ne[0];
    const int64_t H_v = v->ne[1];

    GGML_ASSERT(n_seqs == 1);
    GGML_ASSERT(v->ne[2] == n_tokens);
    GGML_ASSERT(k->ne[2] == n_tokens);
    GGML_ASSERT(g->ne[0] == H_v && g->ne[1] == n_tokens && g->ne[2] == n_seqs);
    if (beta->ne[0] != H_v || beta->ne[2] != n_tokens || beta->ne[3] != n_seqs) {
        printf("beta: %ld x %ld x %ld, expected %ld x %ld x %ld\n", beta->ne[0], beta->ne[2], beta->ne[3], H_v, n_tokens, n_seqs);
    }
    GGML_ASSERT(beta->ne[0] == H_v && beta->ne[2] == n_tokens && beta->ne[3] == n_seqs);
    GGML_ASSERT(state->ne[0] == S_v && state->ne[1] == S_v && state->ne[2] == H_v && state->ne[3] == n_seqs);
    GGML_ASSERT(H_k == H_v);

    const float scale = 1.0f / sqrtf(S_v);
    q = ggml_scale(ctx0, q, scale);

    beta = ggml_sigmoid(ctx0, beta);

    cb(q,    "q_in", il);
    cb(k,    "k_in", il);
    cb(v,    "v_in", il);
    cb(beta, "beta_in", il);
    cb(g,    "g_in", il);
    cb(state,"state_in", il);

    const int64_t chunk_size = QWEN3NEXT_CHUNK_SIZE;
    const int64_t pad = (chunk_size - n_tokens % chunk_size) % chunk_size;
    const int64_t n_chunks = (n_tokens + pad) / chunk_size;

    q    = ggml_permute(ctx0, q, 0, 2, 1, 3);
    k    = ggml_permute(ctx0, k, 0, 2, 1, 3);
    v    = ggml_permute(ctx0, v, 0, 2, 1, 3);
    g    = ggml_permute(ctx0, g, 2, 0, 3, 1);
    beta = ggml_permute(ctx0, beta, 2, 0, 1, 3);

    q    = ggml_pad(ctx0, q, 0, pad, 0, 0);
    k    = ggml_pad(ctx0, k, 0, pad, 0, 0);
    v    = ggml_pad(ctx0, v, 0, pad, 0, 0);
    beta = ggml_pad(ctx0, beta, 0, pad, 0, 0);
    g    = ggml_pad(ctx0, g, pad, 0, 0, 0);

    cb(q,    "q_pad", il);
    cb(k,    "k_pad", il);
    cb(v,    "v_pad", il);
    cb(beta, "beta_pad", il);
    cb(g,    "g_pad", il);

    ggml_tensor * v_beta = ggml_mul(ctx0, v, beta);
    ggml_tensor * k_beta = ggml_mul(ctx0, k, beta);

    cb(v_beta, "v_beta", il);
    cb(k_beta, "k_beta", il);

    q      = ggml_reshape_4d(ctx0, q,      S_k, chunk_size, n_chunks, H_k * n_seqs);
    k      = ggml_reshape_4d(ctx0, k,      S_k, chunk_size, n_chunks, H_k * n_seqs);
    k_beta = ggml_reshape_4d(ctx0, k_beta, S_k, chunk_size, n_chunks, H_v * n_seqs);
    v      = ggml_reshape_4d(ctx0, v,      S_v, chunk_size, n_chunks, H_v * n_seqs);
    v_beta = ggml_reshape_4d(ctx0, v_beta, S_v, chunk_size, n_chunks, H_v * n_seqs);

    g    = ggml_reshape_4d(ctx0, g, chunk_size, 1, n_chunks, H_v * n_seqs);
    beta = ggml_reshape_4d(ctx0, beta, 1, chunk_size, n_chunks, H_v * n_seqs);

    ggml_tensor * g_cumsum = ggml_cumsum(ctx0, g);
    cb(g_cumsum, "g_cumsum", il);

    ggml_tensor * gcs_i =
        ggml_repeat_4d(ctx0, g_cumsum, chunk_size, chunk_size, n_chunks, H_v * n_seqs);
    ggml_tensor * gcs_j = ggml_reshape_4d(ctx0, g_cumsum, 1, chunk_size, n_chunks, H_v * n_seqs);

    ggml_tensor * gcs_j_broadcast =
        ggml_repeat_4d(ctx0, gcs_j, chunk_size, chunk_size, n_chunks, H_v * n_seqs);
    ggml_tensor * decay_mask = ggml_sub(ctx0, gcs_j_broadcast, gcs_i);
    cb(decay_mask, "decay_mask", il);

    decay_mask = ggml_mul(ctx0, decay_mask, diag_mask);
    cb(decay_mask, "decay_mask_1", il);
    decay_mask = ggml_exp(ctx0, decay_mask);
    cb(decay_mask, "decay_mask_exp", il);
    decay_mask = ggml_mul(ctx0, decay_mask, diag_mask);
    cb(decay_mask, "decay_mask_2", il);

    ggml_tensor * kmulkbeta = ggml_mul_mat(ctx0, k, k_beta);
    cb(kmulkbeta, "kk_beta", il);

    ggml_tensor * k_decay = ggml_mul(ctx0, kmulkbeta, decay_mask);
    cb(k_decay, "k_decay_1", il);
    k_decay = ggml_mul(ctx0, k_decay, causal_mask);
    cb(k_decay, "k_decay_2", il);
    ggml_tensor * attn    = ggml_neg(ctx0, k_decay);
    cb(attn, "attn_pre_solve", il);

    ggml_tensor * attn_lower = ggml_mul(ctx0, attn, causal_mask);
    cb(attn_lower, "attn_lower", il);
    ggml_tensor * identity_repeat =
        ggml_repeat_4d(ctx0, identity, attn_lower->ne[0], attn_lower->ne[1], attn_lower->ne[2], attn_lower->ne[3]);
    ggml_tensor * lhs        = ggml_neg(ctx0, ggml_sub(ctx0, attn_lower, identity_repeat));

    ggml_tensor * lin_solve  = ggml_solve_tri(ctx0, lhs, attn, true, true, false);
    attn                     = ggml_mul(ctx0, lin_solve, causal_mask);
    cb(attn, "attn_mul", il);
    attn                     = ggml_add(ctx0, attn, identity);
    cb(attn, "attn_solved", il);

    auto v_beta_t = ggml_cont(ctx0, ggml_transpose(ctx0, v_beta));
    cb(v_beta_t, "v_beta_t", il);
    v = ggml_mul_mat(ctx0, v_beta_t, attn);
    cb(v, "v_beta", il);

    ggml_tensor * g_cumsum_t = ggml_cont(ctx0, ggml_transpose(ctx0, g_cumsum));
    cb(g_cumsum_t, "g_cumsum_t", il);
    ggml_tensor * gexp       = ggml_exp(ctx0, g_cumsum_t);
    cb(gexp, "gexp", il);

    ggml_tensor * kbeta_gexp = ggml_mul(ctx0, k_beta, gexp);
    cb(kbeta_gexp, "kbeta_gexp", il);

    auto kbeta_gexp_t = ggml_cont(ctx0, ggml_transpose(ctx0, kbeta_gexp));
    cb(kbeta_gexp_t, "kbeta_gexp_t", il);
    auto attn_kbeta = ggml_mul_mat(ctx0, attn, kbeta_gexp_t);
    cb(attn_kbeta, "attn_kbeta", il);
    ggml_tensor * k_cumdecay = ggml_cont(ctx0, ggml_transpose(ctx0, attn_kbeta));
    cb(k_cumdecay, "k_cumdecay", il);

    ggml_tensor * attn_kq = ggml_mul_mat(ctx0, k, q);
    cb(attn_kq, "attn_kq_pre", il);
    attn_kq = ggml_mul(ctx0, decay_mask, attn_kq);
    cb(attn_kq, "attn_kq_0", il);
    attn_kq = ggml_mul(ctx0, attn_kq,    diag_mask);
    cb(attn_kq, "attn_kq", il);

    ggml_tensor * g_last = ggml_view_4d(ctx0, g_cumsum, 1, 1, g_cumsum->ne[2], g_cumsum->ne[3],
            g_cumsum->nb[1], g_cumsum->nb[2], g_cumsum->nb[3],
            (g_cumsum->ne[0] - 1) * ggml_element_size(g_cumsum));
    g_last = ggml_cont(ctx0, g_last);
    cb(g_last, "g_last", il);

    ggml_tensor * g_last_exp = ggml_exp(ctx0, g_last);
    cb(g_last_exp, "g_last_exp", il);

    ggml_tensor * g_last_repeat =
        ggml_repeat_4d(ctx0, g_last, chunk_size, 1, n_chunks, H_v * n_seqs);
    ggml_tensor * g_diff = ggml_neg(ctx0, ggml_sub(ctx0, g_cumsum, g_last_repeat));
    cb(g_diff, "g_diff", il);

    ggml_tensor * g_diff_exp = ggml_exp(ctx0, g_diff);
    cb(g_diff_exp, "g_diff_exp", il);
    ggml_tensor * g_diff_exp_t = ggml_reshape_4d(ctx0, g_diff_exp, 1, chunk_size, n_chunks, g_diff_exp->ne[3]);

    ggml_tensor * key_gdiff = ggml_mul(ctx0, k, g_diff_exp_t);
    cb(key_gdiff, "key_gdiff", il);

    ggml_tensor * key_gdiff_t = ggml_cont(ctx0, ggml_transpose(ctx0, key_gdiff));
    cb(key_gdiff_t, "key_gdiff_t", il);

    cb(state, "new_state", il);

    auto get_slice_2d = [ctx0](ggml_tensor * t, int64_t c) -> ggml_tensor * {
        return ggml_view_4d(ctx0, t, t->ne[0], t->ne[1], 1, t->ne[3],
                t->nb[1], t->nb[2], t->nb[3], t->nb[2] * c);
    };

    ggml_tensor * core_attn_out = nullptr;

    for (int64_t chunk = 0; chunk < n_chunks; chunk++) {
        ggml_tensor * q_chunk          = get_slice_2d(q, chunk);
        ggml_tensor * v_chunk          = get_slice_2d(v, chunk);
        ggml_tensor * gexp_chunk       = get_slice_2d(gexp, chunk);
        ggml_tensor * k_cumdecay_chunk = get_slice_2d(k_cumdecay, chunk);
        ggml_tensor * attn_chunk       = get_slice_2d(attn_kq, chunk);
        cb(attn_chunk, "attn_chunk", il);

        ggml_tensor * state_t = ggml_cont_4d(ctx0, ggml_permute(ctx0, state, 1, 0, 2, 3), S_v, S_v, 1, H_v * n_seqs);
        cb(state_t, "state_t", il);

        ggml_tensor * v_prime = ggml_mul_mat(ctx0, state_t, k_cumdecay_chunk);
        cb(v_prime, "v_prime_chunk", il);

        ggml_tensor * v_new   = ggml_sub(ctx0, v_prime, v_chunk);
        ggml_tensor * v_new_t = ggml_cont(ctx0, ggml_transpose(ctx0, v_new));
        cb(v_new, "v_new_chunk", il);

        ggml_tensor * q_g_exp    = ggml_mul(ctx0, q_chunk, gexp_chunk);
        cb(q_g_exp, "q_g_exp", il);
        ggml_tensor * attn_inter = ggml_mul_mat(ctx0, state_t, q_g_exp);
        cb(attn_inter, "attn_inter_chunk", il);

        ggml_tensor * v_attn = ggml_mul_mat(ctx0, v_new_t, attn_chunk);
        cb(v_attn, "v_attn_chunk", il);

        ggml_tensor * core_attn_out_chunk = ggml_sub(ctx0, attn_inter, v_attn);
        cb(core_attn_out_chunk, "core_attn_out_chunk", il);

        core_attn_out = core_attn_out == nullptr
            ? core_attn_out_chunk
            : ggml_concat(ctx0, core_attn_out, core_attn_out_chunk, 2);

        ggml_tensor * k_gdiff_t = get_slice_2d(key_gdiff_t, chunk);
        ggml_tensor * kgdmulvnew = ggml_mul_mat(ctx0, v_new_t, k_gdiff_t);
        cb(kgdmulvnew, "kgdmulvnew", il);

        ggml_tensor * gexp_last_chunk = ggml_cont(ctx0, get_slice_2d(g_last_exp, chunk));
        cb(gexp_last_chunk, "gexp_last_chunk", il);
        auto s_mul = ggml_mul(ctx0, state, ggml_reshape_4d(ctx0, gexp_last_chunk, gexp_last_chunk->ne[0], gexp_last_chunk->ne[1], H_v, n_seqs));
        cb(s_mul, "s_mul", il);
        state = ggml_sub(ctx0, s_mul, ggml_reshape_4d(ctx0, kgdmulvnew, kgdmulvnew->ne[0], kgdmulvnew->ne[1], H_v, n_seqs));
    }

    ggml_tensor * output_tokens = ggml_view_4d(ctx0, core_attn_out,
            S_v, n_tokens, H_v, n_seqs,
            ggml_row_size(core_attn_out->type, S_v),
            ggml_row_size(core_attn_out->type, S_v * QWEN3NEXT_CHUNK_SIZE * n_chunks),
            ggml_row_size(core_attn_out->type, S_v * QWEN3NEXT_CHUNK_SIZE * n_chunks * H_v), 0);
    cb(output_tokens, "output_tokens", il);

    output_tokens = ggml_permute(ctx0, output_tokens, 0, 2, 1, 3);
    output_tokens = ggml_cont(ctx0, output_tokens);
    cb(output_tokens, "output_tokens", il);

    return {output_tokens, state};
}

std::pair<ggml_tensor *, ggml_tensor *> delta_net::build_delta_net_autoregressive(ggml_context * ctx0,
        ggml_tensor * q, ggml_tensor * k, ggml_tensor * v,
        ggml_tensor * g, ggml_tensor * beta, ggml_tensor * state,
        int il, const llm_build_cb & cb) {
    const int64_t H_k      = q->ne[1];
    const int64_t n_tokens = q->ne[2];
    const int64_t n_seqs   = q->ne[3];

    const int64_t S_v = v->ne[0];
    const int64_t H_v = v->ne[1];

    GGML_ASSERT(n_tokens == 1);
    GGML_ASSERT(n_seqs == 1);
    GGML_ASSERT(H_k == H_v);
    GGML_ASSERT(state->ne[0] == S_v && state->ne[1] == S_v && state->ne[2] == H_v && state->ne[3] == n_seqs);

    const float scale = 1.0f / sqrtf(S_v);

    q    = ggml_scale(ctx0, q, scale);
    beta = ggml_sigmoid(ctx0, beta);

    cb(q,    "q_in", il);
    cb(k,    "k_in", il);
    cb(v,    "v_in", il);
    cb(beta, "beta_in", il);
    cb(g,    "g_in", il);

    ggml_tensor * g_t    = ggml_reshape_4d(ctx0, ggml_transpose(ctx0, g), 1, 1, H_k, n_seqs);
    ggml_tensor * beta_t = ggml_reshape_4d(ctx0, ggml_transpose(ctx0, beta), 1, 1, H_k, n_seqs);

    g_t = ggml_exp(ctx0, g_t);
    cb(g_t, "g_t", il);
    state = ggml_mul(ctx0, state, g_t);
    cb(state, "state", il);

    ggml_tensor * k_t_unsqueezed = ggml_reshape_4d(ctx0, k, 1, S_v, H_v, n_seqs);
    ggml_tensor * kv_mem         = ggml_mul(ctx0, state, k_t_unsqueezed);
    cb(kv_mem, "kv_mem", il);
    kv_mem = ggml_cont(ctx0, ggml_transpose(ctx0, kv_mem));
    cb(kv_mem, "kv_mem_t_cont", il);
    kv_mem = ggml_transpose(ctx0, ggml_sum_rows(ctx0, kv_mem));

    ggml_tensor * v_t    = ggml_reshape_4d(ctx0, v, S_v, 1, H_v, n_seqs);
    ggml_tensor * v_diff = ggml_sub(ctx0, v_t, kv_mem);
    cb(v_diff, "v_diff", il);
    ggml_tensor * delta  = ggml_mul(ctx0, v_diff, beta_t);
    cb(delta, "delta", il);

    ggml_tensor * k_t_delta = ggml_mul(ctx0, ggml_repeat_4d(ctx0, k_t_unsqueezed, S_v, S_v, H_v, n_seqs), delta);
    cb(k_t_delta, "k_t_delta", il);
    state                   = ggml_add(ctx0, state, k_t_delta);

    ggml_tensor * q_t_unsqueezed = ggml_reshape_4d(ctx0, q, 1, S_v, H_v, n_seqs);
    ggml_tensor * state_q        = ggml_mul(ctx0, state, q_t_unsqueezed);
    cb(state_q, "state_q", il);
    state_q = ggml_cont(ctx0, ggml_transpose(ctx0, state_q));
    cb(state_q, "state_q_t_cont", il);
    ggml_tensor * core_attn_out = ggml_transpose(ctx0, ggml_sum_rows(ctx0, state_q));

    cb(core_attn_out, "output_tokens", il);
    cb(state,         "new_state", il);

    return {core_attn_out, state};
}

std::pair<ggml_tensor *, ggml_tensor *> delta_net::build_qkvz(ggml_context * ctx0, ggml_tensor * input, int il, const llm_build_cb & cb) const {
    auto & model = lctx.model;
    const int64_t n_tok = input->ne[1];
    if (model.layers[il].wqkv) {
        ggml_tensor * qkv_mixed = llm_build_context::llm_build_lora_mm(lctx, ctx0, model.layers[il].wqkv, input);
        cb(qkv_mixed, "qkv_mixed", il);
        qkv_mixed = ggml_reshape_3d(ctx0, qkv_mixed, qkv_mixed->ne[0], n_tok, 1);
        cb(qkv_mixed, "linear_attn_qkv_mixed", il);

        ggml_tensor * z = llm_build_context::llm_build_lora_mm(lctx, ctx0, model.layers[il].wqkv_gate, input);
        cb(z, "z", il);

        return { qkv_mixed, z };
    }

    auto & hparams = model.hparams;
    const int64_t head_k_dim  = hparams.ssm_d_state;
    const int64_t num_k_heads = hparams.ssm_n_group;
    const int64_t num_v_heads = hparams.ssm_dt_rank;
    const int64_t head_v_dim  = hparams.ssm_d_inner / num_v_heads;

    ggml_tensor * mixed_qkvz = llm_build_context::llm_build_lora_mm(lctx, ctx0, model.layers[il].ssm_in, input);
    cb(mixed_qkvz, "linear_attn_mixed_qkvz", il);

    const int64_t qkvz_new_dim = 2 * head_k_dim + 2 * head_v_dim * (num_v_heads / num_k_heads);
    ggml_tensor * mixed_qkvz_reshaped = ggml_reshape_4d(ctx0, mixed_qkvz, qkvz_new_dim, num_k_heads, n_tok, 1);

    int64_t split_sizes_qkvz[4] = {
        head_k_dim,
        head_k_dim,
        head_v_dim * num_v_heads / num_k_heads,
        head_v_dim * num_v_heads / num_k_heads
    };

    ggml_tensor * query = ggml_view_4d(ctx0, mixed_qkvz_reshaped, split_sizes_qkvz[0], num_k_heads, n_tok, 1,
            mixed_qkvz_reshaped->nb[1], mixed_qkvz_reshaped->nb[2], mixed_qkvz_reshaped->nb[3], 0);
    cb(query, "q", il);

    ggml_tensor * key = ggml_view_4d(ctx0, mixed_qkvz_reshaped, split_sizes_qkvz[1], num_k_heads, n_tok, 1,
            mixed_qkvz_reshaped->nb[1], mixed_qkvz_reshaped->nb[2], mixed_qkvz_reshaped->nb[3],
            split_sizes_qkvz[0] * ggml_element_size(mixed_qkvz_reshaped));
    cb(key, "k", il);

    ggml_tensor * value = ggml_view_4d(ctx0, mixed_qkvz_reshaped, split_sizes_qkvz[2], num_k_heads, n_tok, 1,
            mixed_qkvz_reshaped->nb[1], mixed_qkvz_reshaped->nb[2], mixed_qkvz_reshaped->nb[3],
            (split_sizes_qkvz[0] + split_sizes_qkvz[1]) * ggml_element_size(mixed_qkvz_reshaped));
    cb(value, "v", il);

    ggml_tensor * z = ggml_view_4d(ctx0, mixed_qkvz_reshaped, split_sizes_qkvz[3], num_k_heads, n_tok, 1,
            mixed_qkvz_reshaped->nb[1], mixed_qkvz_reshaped->nb[2], mixed_qkvz_reshaped->nb[3],
            (split_sizes_qkvz[0] + split_sizes_qkvz[1] + split_sizes_qkvz[2]) * ggml_element_size(mixed_qkvz_reshaped));
    z = ggml_cont(ctx0, z);
    cb(z, "z", il);

    ggml_tensor * query_flat = ggml_cont_3d(ctx0, query, head_k_dim * num_k_heads, n_tok, 1);
    cb(query_flat, "query_flat", il);

    ggml_tensor * key_flat = ggml_cont_3d(ctx0, key, head_k_dim * num_k_heads, n_tok, 1);
    cb(key_flat, "key_flat", il);

    ggml_tensor * value_flat = ggml_cont_3d(ctx0, value, head_v_dim * num_v_heads, n_tok, 1);
    cb(value_flat, "value_flat", il);

    ggml_tensor * qkv_mixed = ggml_concat(ctx0, query_flat, key_flat, 0);
    qkv_mixed = ggml_concat(ctx0, qkv_mixed, value_flat, 0);
    cb(qkv_mixed, "qkv_mixed", il);

    return { qkv_mixed, z };
}

ggml_tensor * delta_net::build_layer_attn_linear_core(ggml_context * ctx0, ggml_cgraph * gf,
            ggml_tensor * cur, ggml_tensor * causal_mask, ggml_tensor * identity,
            ggml_tensor * diag_mask, ggml_tensor * inp_s_seq_qnext,
            uint32_t state_seq_id_local, bool reset_state_local, int il, const llm_build_cb & cb) const {

    auto & model = lctx.model;
    auto & hparams = model.hparams;
    auto & kv_self = lctx.kv_self;
    const int64_t head_k_dim  = hparams.ssm_d_state;
    const int64_t num_k_heads = hparams.ssm_n_group;
    const int64_t num_v_heads = hparams.ssm_dt_rank;
    const int64_t head_v_dim  = hparams.ssm_d_inner / num_v_heads;
    const int64_t key_dim        = head_k_dim * num_k_heads;
    const int64_t value_dim      = head_v_dim * num_v_heads;
    const int64_t conv_dim       = key_dim * 2 + value_dim;
    const int64_t conv_state_dim = (hparams.ssm_d_conv - 1) * conv_dim;
    const int64_t ssm_state_dim  = head_v_dim * head_v_dim * num_v_heads;
    const int64_t state_dim      = conv_state_dim + ssm_state_dim;
    const uint32_t qnext_state_slots = llm_build_context::llama_kv_qnext_state_slots(kv_self);
    GGML_ASSERT(qnext_state_slots > 0);

    const int64_t n_tok = cur->ne[1];
    const int64_t n_seqs = 1;
    const int64_t n_seq_tokens = n_tok;

    auto qkvz = build_qkvz(ctx0, cur, il, cb);
    ggml_tensor * qkv_mixed = qkvz.first;
    ggml_tensor * z         = qkvz.second;

    ggml_tensor *alpha, *beta;
    if (model.layers[il].ssm_beta_alpha) {
        ggml_tensor * mixed_ba = llm_build_context::llm_build_lora_mm(lctx, ctx0, model.layers[il].ssm_beta_alpha, cur);
        cb(mixed_ba, "linear_attn_mixed_ba", il);

        int64_t ba_new_dim = 2 * num_v_heads / num_k_heads;
        ggml_tensor * mixed_ba_reshaped = ggml_reshape_4d(ctx0, mixed_ba, ba_new_dim, num_k_heads, n_tok, 1);

        int64_t split_sizes_ba[2] = {
            num_v_heads / num_k_heads,
            num_v_heads / num_k_heads
        };

        ggml_tensor * b = ggml_view_4d(ctx0, mixed_ba_reshaped, split_sizes_ba[0], num_k_heads, n_tok, 1,
                mixed_ba_reshaped->nb[1], mixed_ba_reshaped->nb[2], mixed_ba_reshaped->nb[3], 0);
        cb(b, "b", il);

        ggml_tensor * a = ggml_view_4d(ctx0, mixed_ba_reshaped, split_sizes_ba[1], num_k_heads, n_tok, 1,
                mixed_ba_reshaped->nb[1], mixed_ba_reshaped->nb[2], mixed_ba_reshaped->nb[3],
                split_sizes_ba[0] * ggml_element_size(mixed_ba_reshaped));
        cb(a, "a", il);

        beta  = ggml_cont_4d(ctx0, b, num_v_heads, 1, n_tok, 1);
        alpha = ggml_cont_3d(ctx0, a, num_v_heads, n_tok, 1);
    } else {
        beta = llm_build_context::llm_build_lora_mm(lctx, ctx0, model.layers[il].ssm_beta, cur);
        beta = ggml_reshape_4d(ctx0, beta, num_v_heads, 1, n_tok, 1);
        alpha = llm_build_context::llm_build_lora_mm(lctx, ctx0, model.layers[il].ssm_alpha, cur);
        // Why???
        alpha = ggml_cont_3d(ctx0, alpha, num_v_heads, n_seq_tokens, n_seqs);
    }
    cb(beta, "beta", il);
    cb(alpha, "alpha", il);

    ggml_tensor * alpha_biased   = ggml_add(ctx0, alpha, model.layers[il].ssm_dt);
    ggml_tensor * alpha_softplus = ggml_softplus(ctx0, alpha_biased);
    cb(alpha_softplus, "a_softplus", il);
    ggml_tensor * gate = ggml_mul(ctx0, alpha_softplus, model.layers[il].ssm_a);
    cb(gate, "gate", il);

    size_t state_row_size = 0;
    ggml_tensor * state_all = nullptr;
    GGML_ASSERT((size_t) il < kv_self.s_l.size() && kv_self.s_l[il] != nullptr);
    ggml_tensor * state_storage = kv_self.s_l[il];
    GGML_ASSERT(state_storage->type == GGML_TYPE_F32);
    GGML_ASSERT(state_storage->ne[0] >= state_dim);
    GGML_ASSERT((uint32_t) state_storage->ne[1] == qnext_state_slots);
    state_row_size = state_storage->nb[1];
    GGML_ASSERT(ggml_nbytes(state_storage) >= state_row_size * qnext_state_slots);
    state_all = ggml_view_2d(ctx0, state_storage, state_dim, qnext_state_slots, state_row_size, 0);

    ggml_tensor * state_dst = ggml_view_2d(ctx0, state_all, state_dim, 1, state_row_size, state_seq_id_local * state_row_size);
    ggml_tensor * state_f32 = state_dst;
    if (state_f32->type != GGML_TYPE_F32) {
        state_f32 = ggml_cast(ctx0, state_f32, GGML_TYPE_F32);
    }
    if (reset_state_local) {
        state_f32 = ggml_scale(ctx0, state_f32, 0.0f);
    }

    ggml_tensor * conv_state_flat = ggml_view_2d(ctx0, state_f32, conv_state_dim, 1, state_f32->nb[1], 0);
    ggml_tensor * ssm_state_flat  = ggml_view_2d(ctx0, state_f32, ssm_state_dim, 1, state_f32->nb[1],
            conv_state_dim * ggml_element_size(state_f32));

    ggml_tensor * conv_states = ggml_reshape_3d(ctx0, conv_state_flat, hparams.ssm_d_conv - 1, conv_dim, 1);
    ggml_tensor * state       = ggml_reshape_4d(ctx0, ssm_state_flat, head_v_dim, head_v_dim, num_v_heads, 1);
    cb(conv_states, "conv_states", il);
    cb(state, "state_predelta", il);

    ggml_tensor * conv_output_raw = ggml_ssm_conv(ctx0, conv_states, qkv_mixed, model.layers[il].ssm_conv1d, inp_s_seq_qnext);
    cb(conv_output_raw, "conv_output_raw", il);

    //ggml_tensor * conv_output = ggml_view_2d(ctx0, conv_output_raw, conv_dim, n_tok, conv_dim * ggml_element_size(conv_output_raw), 0);
    //ggml_tensor * conv_output_silu = ggml_silu(ctx0, conv_output);
    ggml_tensor * conv_output_silu = ggml_silu(ctx0, conv_output_raw);
    cb(conv_output_silu, "conv_output_silu", il);

    // Calculate the total conv dimension
    int64_t qkv_dim = head_k_dim * num_k_heads * 2 + head_v_dim * num_v_heads;
    int64_t nb1_qkv = ggml_row_size(conv_output_silu->type, qkv_dim);

    // Extract the convolved Q, K, V from conv_output
    ggml_tensor * q_conv = ggml_view_4d(ctx0, conv_output_silu, head_k_dim, num_k_heads, n_tok, 1,
            ggml_row_size(conv_output_silu->type, head_k_dim),
            nb1_qkv, nb1_qkv * n_tok, 0);

    ggml_tensor * k_conv = ggml_view_4d(ctx0, conv_output_silu, head_k_dim, num_k_heads, n_tok, 1,
            ggml_row_size(conv_output_silu->type, head_k_dim),
            nb1_qkv, nb1_qkv * n_tok,
            head_k_dim * num_k_heads * ggml_element_size(conv_output_silu));

    ggml_tensor * v_conv = ggml_view_4d(ctx0, conv_output_silu, head_v_dim, num_v_heads, n_tok, 1,
            ggml_row_size(conv_output_silu->type, head_v_dim),
            nb1_qkv, nb1_qkv * n_tok,
            ggml_row_size(conv_output_silu->type, 2 * head_k_dim * num_k_heads));

    cb(q_conv, "q_conv", il);
    cb(k_conv, "k_conv", il);
    cb(v_conv, "v_conv", il);

    const float eps_norm = hparams.f_norm_rms_eps;

    q_conv = ggml_l2_norm(ctx0, q_conv, eps_norm);
    k_conv = ggml_l2_norm(ctx0, k_conv, eps_norm);

    if (num_k_heads != num_v_heads) {
        GGML_ASSERT(num_v_heads % num_k_heads == 0);
        if (model.layers[il].ssm_beta_alpha) {
            const int64_t repeat_factor = num_v_heads / num_k_heads;

            ggml_tensor * q_reshaped = ggml_reshape_3d(ctx0, q_conv, head_k_dim, 1, num_k_heads * n_tok);
            ggml_tensor * k_reshaped = ggml_reshape_3d(ctx0, k_conv, head_k_dim, 1, num_k_heads * n_tok);

            ggml_tensor * q_repeated = ggml_repeat_4d(ctx0, q_reshaped, head_k_dim, repeat_factor, num_k_heads * n_tok, 1);
            ggml_tensor * k_repeated = ggml_repeat_4d(ctx0, k_reshaped, head_k_dim, repeat_factor, num_k_heads * n_tok, 1);

            q_conv = ggml_reshape_4d(ctx0, q_repeated, head_k_dim, num_k_heads * repeat_factor, n_tok, 1);
            k_conv = ggml_reshape_4d(ctx0, k_repeated, head_k_dim, num_k_heads * repeat_factor, n_tok, 1);
        } else {
            q_conv = ggml_repeat_4d(ctx0, q_conv, head_k_dim, num_v_heads, n_seq_tokens, n_seqs);
            k_conv = ggml_repeat_4d(ctx0, k_conv, head_k_dim, num_v_heads, n_seq_tokens, n_seqs);
        }
    }

    cb(q_conv, "q_conv_predelta", il);
    cb(k_conv, "k_conv_predelta", il);
    cb(v_conv, "v_conv_predelta", il);

    std::pair<ggml_tensor *, ggml_tensor *> attn_out;

    GGML_ASSERT(causal_mask != nullptr);
    GGML_ASSERT(identity    != nullptr);
    GGML_ASSERT(diag_mask   != nullptr);

    attn_out = n_tok == 1
        ? build_delta_net_autoregressive(ctx0, q_conv, k_conv, v_conv, gate, beta, state, il, cb)
        : build_delta_net_chunking(ctx0, q_conv, k_conv, v_conv, gate, beta, state, causal_mask, identity, diag_mask, il, cb);
    ggml_tensor * output    = attn_out.first;
    ggml_tensor * new_state = attn_out.second;
    cb(output, "attn_output", il);
    cb(new_state, "new_state", il);

    ggml_tensor * new_conv_states = ggml_view_2d(ctx0, conv_output_raw, hparams.ssm_d_conv - 1, conv_dim,
            hparams.ssm_d_conv * ggml_element_size(conv_output_raw),
            (1 + conv_dim * n_tok) * ggml_element_size(conv_output_raw));
    auto new_conv_states_cont = ggml_cont(ctx0, new_conv_states);
    cb(new_conv_states_cont, "new_conv_states_cont", il);
    ggml_tensor * new_conv_flat = ggml_reshape_2d(ctx0, new_conv_states_cont, conv_state_dim, 1);
    ggml_tensor * new_ssm_flat  = ggml_reshape_2d(ctx0, new_state, ssm_state_dim, 1);
    ggml_tensor * new_state_flat = ggml_concat(ctx0, new_conv_flat, new_ssm_flat, 0);

    ggml_tensor * state_update = new_state_flat;
    if (state_dst->type != GGML_TYPE_F32) {
        state_update = ggml_cast(ctx0, state_update, state_dst->type);
    }
    ggml_build_forward_expand(gf, ggml_cpy(ctx0, state_update, state_dst));

    ggml_tensor * attn_out_2d = ggml_reshape_2d(ctx0, output, head_v_dim, num_v_heads * n_tok);
    ggml_tensor * z_2d        = ggml_reshape_2d(ctx0, z,      head_v_dim, num_v_heads * n_tok);

    ggml_tensor * attn_out_norm = llm_build_context::llm_build_norm(ctx0, attn_out_2d, hparams, model.layers[il].ssm_norm, nullptr, LLM_NORM_RMS, cb, il);
    ggml_tensor * gated_silu    = ggml_silu(ctx0, z_2d);
    cb(gated_silu, "gated_silu", il);
    attn_out_norm = ggml_mul(ctx0, attn_out_norm, gated_silu);
    cb(attn_out_norm, "attn_out_norm", il);

    ggml_tensor * final_output = ggml_reshape_2d(ctx0, attn_out_norm, value_dim, n_tok);
    cb(final_output, "final_output", il);

    ggml_tensor * out = llm_build_context::llm_build_lora_mm(lctx, ctx0, model.layers[il].ssm_out, final_output);
    cb(out, "linear_attn_out", il);

    return ggml_reshape_2d(ctx0, out, hparams.n_embd, n_tok);

}

ggml_tensor * delta_net::build_layer_attn_linear(ggml_context * ctx0, ggml_cgraph * gf,
        ggml_tensor * cur, ggml_tensor * causal_mask, ggml_tensor * identity,
        ggml_tensor * diag_mask, int il, const llm_build_cb & cb) const {
    GGML_ASSERT(lctx.inp_s_seq_qnext != nullptr);

    auto & model = lctx.model;
    auto & hparams = model.hparams;
    GGML_ASSERT(hparams.is_recurrent(il));

    GGML_ASSERT(model.layers[il].ssm_conv1d != nullptr);
    GGML_ASSERT(model.layers[il].ssm_dt != nullptr);
    GGML_ASSERT(model.layers[il].ssm_a != nullptr);
    GGML_ASSERT(model.layers[il].ssm_beta_alpha != nullptr || (model.layers[il].ssm_alpha != nullptr && model.layers[il].ssm_beta != nullptr));
    GGML_ASSERT(model.layers[il].ssm_norm != nullptr);
    GGML_ASSERT(model.layers[il].ssm_out != nullptr);
    GGML_ASSERT(model.layers[il].wqkv != nullptr || model.layers[il].ssm_in != nullptr);
    GGML_ASSERT(model.layers[il].wqkv_gate != nullptr || model.layers[il].ssm_in != nullptr);

    if (all_same_seq) {
        bool reset_state = batch.pos != nullptr && batch.pos[0] == 0;
        return build_layer_attn_linear_core(ctx0, gf, cur, causal_mask, identity, diag_mask, lctx.inp_s_seq_qnext, token_seq_ids.front(), reset_state, il, cb);
    }

    GGML_ASSERT(has_unique_seq_ids && "qwen3next mixed-sequence batches require unique sequence IDs per token");

    ggml_tensor * out = nullptr;
    for (int64_t i = 0; i < batch.n_tokens; ++i) {
        ggml_tensor * cur_i = ggml_view_2d(ctx0, cur, cur->ne[0], 1, cur->nb[1], (size_t) i * cur->nb[1]);
        ggml_tensor * inp_s_seq_qnext_i = ggml_view_2d(ctx0, lctx.inp_s_seq_qnext, 1, 1, lctx.inp_s_seq_qnext->nb[1], (size_t) i * lctx.inp_s_seq_qnext->nb[1]);

        const bool reset_state_i = batch.pos != nullptr && batch.pos[i] == 0;
        const uint32_t state_seq_id_i = (uint32_t) token_seq_ids[i];
        ggml_tensor * out_i = build_layer_attn_linear_core(ctx0, gf, cur_i, causal_mask, identity, diag_mask, inp_s_seq_qnext_i, state_seq_id_i, reset_state_i, il, cb);

        out = out == nullptr ? out_i : ggml_concat(ctx0, out, out_i, 1);
    }

    return out;

}


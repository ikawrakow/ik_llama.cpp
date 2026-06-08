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

    // PXA_LLAMA_MTP_FIX: distinct-sequence decomposition. Generalizes the v4 "n_seqs == n_tok"
    // (one token per seq) assumption. For an MTP verify batch each seq carries (n_max+1) tokens
    // with the SAME seq_id on consecutive positions; this routes one recurrent row per *sequence*
    // (gathered/scattered once) while the conv/delta-net scan iterates n_seq_tokens per seq.
    // pxa_decompose_seqs requires per-seq contiguity + position-monotonicity (true for plain
    // decode AND slot-major MTP verify) and rejects genuinely interleaved batches.
    seq_decomp = pxa_decompose_seqs(batch);

    const uint32_t qnext_state_slots = llm_build_context::llama_kv_qnext_state_slots(lctx.kv_self);
    GGML_ASSERT(qnext_state_slots > 0);

    // Reserve-graph builds may not carry explicit sequence IDs, in which case
    // the fallback sequence slot is 0.
    for (llama_seq_id s : token_seq_ids) {
        GGML_ASSERT(s >= 0);
        GGML_ASSERT((uint32_t) s < qnext_state_slots);
    }

    int max_per_step = lctx.kv_self.save_per_step_ssm
        ? lctx.kv_self.ckpt.per_step_max_allocated
        : 0;
    save_per_step_states = lctx.kv_self.save_per_step_ssm && batch.n_tokens > 1 && batch.n_tokens <= max_per_step;
}

delta_net::~delta_net() = default;

std::pair<ggml_tensor *, ggml_tensor *> delta_net::build_fused_delta_net(ggml_context * ctx0,
        ggml_tensor * q, ggml_tensor * k, ggml_tensor * v,
        ggml_tensor * g, ggml_tensor * beta, ggml_tensor * state,
        int il, const llm_build_cb & cb, int repeat_type,
        ggml_tensor * per_step_ckpt) {

    const int64_t S_k      = q->ne[0];
    const int64_t H_k      = q->ne[2];
    const int64_t n_tokens = q->ne[1];
    const int64_t n_seqs   = q->ne[3];

    const int64_t S_v = v->ne[0];
    const int64_t H_v = v->ne[1];

    GGML_ASSERT(q->ne[0] == S_k && q->ne[2] == H_k && q->ne[1] == n_tokens && q->ne[3] == n_seqs);
    GGML_ASSERT(k->ne[0] == S_k && k->ne[2] == H_k && k->ne[1] == n_tokens && k->ne[3] == n_seqs);
    GGML_ASSERT(v->ne[2] == n_tokens);
    GGML_ASSERT(g->ne[0] == H_v && g->ne[1] == n_tokens && g->ne[2] == n_seqs);
    GGML_ASSERT(beta->ne[0] == H_v && beta->ne[2] == n_tokens && beta->ne[3] == n_seqs);
    GGML_ASSERT(state->ne[0] == S_v && state->ne[1] == S_v && state->ne[2] == H_v && state->ne[3] == n_seqs);
    //GGML_ASSERT(H_k == H_v);
    GGML_ASSERT(H_v % H_k == 0);

    cb(q,    "q_in", il);
    cb(k,    "k_in", il);
    cb(v,    "v_in", il);
    cb(beta, "beta_in", il);
    cb(g,    "g_in", il);
    cb(state,"state_in", il);

    v = ggml_permute(ctx0, v, 0, 2, 1, 3);
    g = ggml_permute(ctx0, g, 2, 0, 3, 1);
    beta = ggml_permute(ctx0, beta, 2, 0, 1, 3);

    ggml_tensor * state_flat = ggml_reshape_4d(ctx0, state, S_v, S_v * H_v, 1, n_seqs);
    if (!ggml_is_contiguous(state_flat)) {
        state_flat = ggml_cont_4d(ctx0, state_flat, S_v, S_v * H_v, 1, n_seqs);
    }

    cb(q,         "q_fused", il);
    cb(k,         "k_fused", il);
    cb(v,         "v_fused", il);
    cb(g,         "g_fused", il);
    cb(beta,      "beta_fused", il);
    cb(state_flat,"state_fused", il);

    ggml_tensor * fused_result = ggml_delta_net(ctx0, q, k, v, g, beta, state_flat, per_step_ckpt);
    cb(fused_result, "delta_net_fused_raw", il);
    fused_result->op_params[0] = repeat_type;

    const int64_t output_size = S_v * H_v * n_tokens * n_seqs;
    const int64_t state_size  = S_v * S_v * H_v * n_seqs;

    auto output_tokens = ggml_view_4d(ctx0, fused_result,
            S_v, H_v, n_tokens, n_seqs,
            ggml_row_size(fused_result->type, S_v),
            ggml_row_size(fused_result->type, S_v * H_v),
            ggml_row_size(fused_result->type, S_v * H_v * n_tokens), 0);
    //output_tokens = ggml_cont_4d(ctx0, output_tokens, S_v, H_v, n_tokens, n_seqs);

    ggml_tensor * new_state_flat = ggml_view_1d(ctx0, fused_result, state_size,
            output_size * ggml_element_size(fused_result));
    ggml_tensor * new_state = ggml_reshape_4d(ctx0, new_state_flat, S_v, S_v, H_v, n_seqs);

    cb(output_tokens, "output_tokens", il);
    cb(new_state,     "new_state", il);

    return {output_tokens, new_state};
}

std::pair<ggml_tensor *, ggml_tensor *> delta_net::build_qkvz(llama_context & lctx, ggml_context * ctx0,
        ggml_tensor * wqkv, ggml_tensor * wqkv_gate,
        ggml_tensor * input, int il, const llm_build_cb & cb, ggml_cgraph * gf) {

    const int64_t n_tok = input->ne[1];
    auto qkv_mixed = llm_build_context::llm_build_lora_mm(lctx, ctx0, wqkv, input);
    cb(qkv_mixed, "qkv_mixed", il);
    ggml_tensor * z = llm_build_context::llm_build_lora_mm(lctx, ctx0, wqkv_gate, input);
    cb(z, "z", il);
    ggml_build_forward_expand(gf, qkv_mixed);
    ggml_build_forward_expand(gf, z);
    qkv_mixed = ggml_reshape_3d(ctx0, qkv_mixed, qkv_mixed->ne[0], n_tok, 1);
    cb(qkv_mixed, "linear_attn_qkv_mixed", il);
    return { qkv_mixed, z };
}

std::pair<ggml_tensor *, ggml_tensor *> delta_net::build_qkvz(llama_context & lctx, ggml_context * ctx0, ggml_tensor * ssm_in,
        int64_t head_k_dim, int64_t num_k_heads, int64_t head_v_dim, int64_t num_v_heads,
        ggml_tensor * input, int il, const llm_build_cb & cb) {

    const int64_t n_tok = input->ne[1];

    ggml_tensor * mixed_qkvz = llm_build_context::llm_build_lora_mm(lctx, ctx0, ssm_in, input);
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

std::pair<ggml_tensor *, ggml_tensor *> delta_net::build_qkvz(llama_context & lctx, ggml_context * ctx0, ggml_tensor * wqkv, ggml_tensor * wqkv_gate, ggml_tensor * ssm_in,
            int64_t head_k_dim, int64_t num_k_heads, int64_t head_v_dim, int64_t num_v_heads, ggml_tensor * input, int il, const llm_build_cb & cb, ggml_cgraph * gf) {
    GGML_ASSERT((wqkv && wqkv_gate) || ssm_in);
    return wqkv && wqkv_gate ? build_qkvz(lctx, ctx0, wqkv, wqkv_gate, input, il, cb, gf)
                             : build_qkvz(lctx, ctx0, ssm_in, head_k_dim, num_k_heads, head_v_dim, num_v_heads, input, il, cb);
}

std::pair<ggml_tensor *, ggml_tensor *> delta_net::build_beta_gate(llama_context & lctx, ggml_context * ctx0,
        ggml_tensor * ssm_beta_alpha, ggml_tensor * ssm_beta, ggml_tensor * ssm_alpha,
        ggml_tensor * ssm_dt, ggml_tensor * ssm_a, int64_t num_k_heads, int64_t num_v_heads, int64_t n_seqs,
        ggml_tensor * cur, int il, const llm_build_cb & cb, ggml_cgraph * gf) {

    auto n_tok = cur->ne[1];
    auto n_seq_tokens = n_tok / n_seqs;

    ggml_tensor *alpha, *beta;
    if (ssm_beta_alpha) {
        ggml_tensor * mixed_ba = llm_build_context::llm_build_lora_mm(lctx, ctx0, ssm_beta_alpha, cur);
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

        // PXA_LLAMA_FIX_v4: split the token axis into (n_seq_tokens, n_seqs) so beta/gate carry the per-seq batch dim
        // that build_fused_delta_net expects (beta [H_v,1,n_seq_tokens,n_seqs], gate [H_v,n_seq_tokens,n_seqs]).
        beta  = ggml_cont_4d(ctx0, b, num_v_heads, 1, n_seq_tokens, n_seqs);
        alpha = ggml_cont_3d(ctx0, a, num_v_heads, n_seq_tokens, n_seqs);
    } else {
        beta = llm_build_context::llm_build_lora_mm(lctx, ctx0, ssm_beta, cur);
        cb(beta, "beta", il);
        beta = ggml_reshape_4d(ctx0, beta, num_v_heads, 1, n_seq_tokens, n_seqs);
        cb(beta, "beta_reshaped", il);
        alpha = llm_build_context::llm_build_lora_mm(lctx, ctx0, ssm_alpha, cur);
        cb(alpha, "alpha", il);
        alpha = ggml_reshape_3d(ctx0, alpha, num_v_heads, n_seq_tokens, n_seqs);
        cb(alpha, "alpha_reshaped", il);
    }
    cb(beta, "beta", il);
    cb(alpha, "alpha", il);
    ggml_build_forward_expand(gf, beta);
    ggml_build_forward_expand(gf, alpha);

    ggml_tensor * alpha_biased   = ggml_add(ctx0, alpha, ssm_dt);
    cb(alpha_biased, "alpha_biased", il);
    ggml_tensor * alpha_softplus = ggml_softplus(ctx0, alpha_biased);
    cb(alpha_softplus, "a_softplus", il);
    ggml_tensor * gate = ggml_mul(ctx0, alpha_softplus, ssm_a);
    cb(gate, "gate", il);

    return {beta, gate};
}

ggml_tensor * delta_net::build_qkv(ggml_context * ctx0, ggml_tensor * state_storage, ggml_tensor * ssm_conv1d,
        ggml_tensor * qkv_mixed, ggml_tensor * state_row_idx, ggml_tensor * conv_seq_map, ggml_tensor * state_mask, ggml_tensor * beta, ggml_tensor * gate,
        int64_t head_k_dim, int64_t num_k_heads, int64_t head_v_dim, int64_t num_v_heads, int64_t ssm_d_conv,
        int64_t n_seqs_in, uint32_t qnext_state_slots, bool reset_state_local,
        float eps_norm, int repeat_type, int il, const llm_build_cb & cb, ggml_cgraph * gf,
        ggml_tensor * per_step_ckpt, ggml_tensor * per_step_conv, int64_t pxa_static_slot) {
    const int64_t key_dim        = head_k_dim * num_k_heads;
    const int64_t value_dim      = head_v_dim * num_v_heads;
    const int64_t conv_dim       = key_dim * 2 + value_dim;
    const int64_t conv_state_dim = (ssm_d_conv - 1) * conv_dim;
    const int64_t ssm_state_dim  = head_v_dim * head_v_dim * num_v_heads;
    const int64_t state_dim      = conv_state_dim + ssm_state_dim;
    GGML_ASSERT(qnext_state_slots > 0);

    // PXA_LLAMA_FIX_v4: n_seqs is passed explicitly. The mixed (concurrent) path calls this ONCE with
    // n_seqs = batch.n_tokens, n_seq_tokens = 1 (each token its own seq) -- a single batched delta-net,
    // NOT a per-token loop of N subgraphs (that interleave was the recurrent-state corruption at np>=3).
    // qkv_mixed is [conv_dim, n_tok] (or [conv_dim, n_tok, 1]); the conv consumes it token-major as a matrix.
    const int64_t n_tok        = qkv_mixed->ne[1];
    const int64_t n_seqs       = n_seqs_in;
    GGML_ASSERT(n_seqs > 0 && n_tok % n_seqs == 0);
    const int64_t n_seq_tokens = n_tok / n_seqs;
    GGML_ASSERT(ggml_is_matrix(qkv_mixed)); // [conv_dim, n_tok] (ne[2]==1) -- consumed token-major by ssm_conv

    size_t state_row_size = 0;
    ggml_tensor * state_all = nullptr;
    GGML_ASSERT(state_storage->type == GGML_TYPE_F32);
    GGML_ASSERT(state_storage->ne[0] >= state_dim);
    GGML_ASSERT((uint32_t) state_storage->ne[1] == qnext_state_slots);
    state_row_size = state_storage->nb[1];
    GGML_ASSERT(ggml_nbytes(state_storage) >= state_row_size * qnext_state_slots);

    state_all = ggml_view_2d(ctx0, state_storage, state_dim, qnext_state_slots, state_row_size, 0);

    // PXA_LLAMA_FIX_v4: gather ALL participating sequences' recurrent rows ONCE, in token order, via state_row_idx
    // (absolute seq_id per seq). all_same -> [n_seqs=1]; mixed -> [n_seqs=n_tok]. ONE get_rows -> [state_dim, n_seqs],
    // ONE set_rows at the end => a SINGLE subgraph (the per-token N-subgraph interleave was the allocator-corruption bug).
    GGML_ASSERT(state_row_idx != nullptr);
    GGML_ASSERT(state_row_idx->ne[0] == n_seqs);
    ggml_tensor * pxa_row_idx = state_row_idx;

    // PXA_PERF_NP1_FASTPATH: for a single sequence (n_seqs==1) whose state row is known at graph-build time
    // (pxa_static_slot>=0), access the recurrent row IN PLACE via a static-offset view -- the pre-PR (main)
    // path -- instead of get_rows/set_rows. The state row is ~state_dim floats (MBs) per layer; the
    // gather+mask+scatter copies it ~3x/layer/token and cost ~10% TG / ~4% PP. Graph reuse stays correct
    // because pxa_seq_sig() encodes the slot for the all_same case, so any slot change rebuilds the graph
    // (a reused graph never keeps a stale static offset). The batched n_seqs>1 path is unchanged.
    const bool pxa_fast = (pxa_static_slot >= 0);
    GGML_ASSERT(!pxa_fast || n_seqs == 1);
    ggml_tensor * state_dst = nullptr;
    ggml_tensor * state_f32 = nullptr;
    if (pxa_fast) {
        state_dst = ggml_view_2d(ctx0, state_all, state_dim, 1, state_row_size, (size_t) pxa_static_slot * state_row_size);
        state_f32 = state_dst;
        if (state_f32->type != GGML_TYPE_F32) {
            state_f32 = ggml_cast(ctx0, state_f32, GGML_TYPE_F32);
        }
        if (reset_state_local) {
            state_f32 = ggml_scale(ctx0, state_f32, 0.0f);
            cb(state_f32, "state_reset", il);
        }
    } else {
        state_dst = ggml_get_rows(ctx0, state_all, pxa_row_idx); // [state_dim, n_seqs]
        state_f32 = state_dst;
        if (state_f32->type != GGML_TYPE_F32) {
            state_f32 = ggml_cast(ctx0, state_f32, GGML_TYPE_F32);
        }
        // PXA_LLAMA_FIX_v4: per-seq reset via mask multiply ([1,n_seqs] broadcast over state_dim): 0 zeros a fresh seq's
        // carried state, 1 keeps it. Handles mixed batches where some seqs start (pos 0) and others continue. Falls back to
        // the scalar reset_state_local only if no mask is supplied (keeps non-qwen3next callers, if any, working).
        if (state_mask != nullptr) {
            GGML_ASSERT(state_mask->ne[1] == n_seqs);
            state_f32 = ggml_mul(ctx0, state_f32, state_mask);
            cb(state_f32, "state_reset_masked", il);
        } else if (reset_state_local) {
            state_f32 = ggml_scale(ctx0, state_f32, 0.0f);
            cb(state_f32, "state_reset", il);
        }
    }

    // [state_dim, n_seqs] -> conv part [conv_state_dim, n_seqs] and ssm part [ssm_state_dim, n_seqs]
    ggml_tensor * conv_state_flat = ggml_view_2d(ctx0, state_f32, conv_state_dim, n_seqs, state_f32->nb[1], 0);
    ggml_tensor * ssm_state_flat  = ggml_view_2d(ctx0, state_f32, ssm_state_dim, n_seqs, state_f32->nb[1],
            conv_state_dim * ggml_element_size(state_f32));
    if (n_seqs > 1) {
        // ssm_conv requires a contiguous 3d conv_states; the views above are strided across seqs when n_seqs>1.
        conv_state_flat = ggml_cont(ctx0, conv_state_flat);
        ssm_state_flat  = ggml_cont(ctx0, ssm_state_flat);
    }

    ggml_tensor * conv_states = ggml_reshape_3d(ctx0, conv_state_flat, ssm_d_conv - 1, conv_dim, n_seqs);
    ggml_tensor * state       = ggml_reshape_4d(ctx0, ssm_state_flat, head_v_dim, head_v_dim, num_v_heads, n_seqs);
    cb(conv_states, "conv_states", il);
    cb(state, "state_predelta", il);
    ggml_build_forward_expand(gf, state);

    // PXA_LLAMA_FIX_v4: conv_seq_map is the [n_kv=n_seqs, n_tok] seq->state-column map. States are gathered in token
    // order so it is identity in row 0 (sq[0][t]=t) with rows>=1 out-of-range -> hits ik's multi-seq-UNIQUE conv fast
    // path. all_same passes a [1,n_tok] view (n_kv=1 -> single-seq fast path, content ignored).
    ggml_tensor * conv_output_raw = ggml_ssm_conv(ctx0, conv_states, qkv_mixed, ssm_conv1d, conv_seq_map, per_step_conv);
    cb(conv_output_raw, "conv_output_raw", il);

    ggml_tensor * conv_output = ggml_view_2d(ctx0, conv_output_raw, conv_dim, n_tok, conv_dim * ggml_element_size(conv_output_raw), 0);
    ggml_tensor * conv_output_silu = ggml_silu(ctx0, conv_output);
    cb(conv_output_silu, "conv_output_silu", il);
    ggml_build_forward_expand(gf, conv_output_silu);

    // Calculate the total conv dimension
    int64_t qkv_dim = head_k_dim * num_k_heads * 2 + head_v_dim * num_v_heads;
    int64_t nb1_qkv = ggml_row_size(conv_output_silu->type, qkv_dim);

    // PXA_LLAMA_FIX_v4: seq stride is nb1_qkv * n_seq_tokens (tokens of a seq are contiguous in token order).
    // all_same: n_seq_tokens=n_tok,n_seqs=1 (== old nb1_qkv*n_tok). mixed: n_seq_tokens=1,n_seqs=n_tok -> nb1_qkv.
    const size_t nb3_qkv = (size_t) nb1_qkv * n_seq_tokens;

    // Extract the convolved Q, K, V from conv_output
    ggml_tensor * q_conv = ggml_view_4d(ctx0, conv_output_silu, head_k_dim, num_k_heads, n_seq_tokens, n_seqs,
            ggml_row_size(conv_output_silu->type, head_k_dim), nb1_qkv, nb3_qkv, 0);

    ggml_tensor * k_conv = ggml_view_4d(ctx0, conv_output_silu, head_k_dim, num_k_heads, n_seq_tokens, n_seqs,
            ggml_row_size(conv_output_silu->type, head_k_dim), nb1_qkv, nb3_qkv,
            head_k_dim * num_k_heads * ggml_element_size(conv_output_silu));

    ggml_tensor * v_conv = ggml_view_4d(ctx0, conv_output_silu, head_v_dim, num_v_heads, n_seq_tokens, n_seqs,
            ggml_row_size(conv_output_silu->type, head_v_dim), nb1_qkv, nb3_qkv,
            ggml_row_size(conv_output_silu->type, 2 * head_k_dim * num_k_heads));

    cb(q_conv, "q_conv", il);
    cb(k_conv, "k_conv", il);
    cb(v_conv, "v_conv", il);

    if (n_seq_tokens > 1) {
        q_conv = ggml_permute(ctx0, q_conv, 0, 2, 1, 3);
        k_conv = ggml_permute(ctx0, k_conv, 0, 2, 1, 3);
        q_conv = ggml_l2_norm(ctx0, q_conv, eps_norm);
        k_conv = ggml_l2_norm(ctx0, k_conv, eps_norm);
    } else {
        q_conv = ggml_l2_norm(ctx0, q_conv, eps_norm);
        k_conv = ggml_l2_norm(ctx0, k_conv, eps_norm);
        q_conv = ggml_permute(ctx0, q_conv, 0, 2, 1, 3);
        k_conv = ggml_permute(ctx0, k_conv, 0, 2, 1, 3);
    }
    cb(q_conv, "q_conv_normed", il);
    cb(k_conv, "k_conv_normed", il);

    auto [output, new_state] = build_fused_delta_net(ctx0, q_conv, k_conv, v_conv, gate, beta, state, il, cb, repeat_type,
            per_step_ckpt);

    cb(output, "attn_output", il);
    cb(new_state, "new_state", il);

    // PXA_LLAMA_FIX_v4: the new conv state lives in conv_output_raw's state region [d_conv, conv_dim, n_seqs] at flat
    // offset conv_dim*n_tok; the next-step state is the LAST (d_conv-1) columns of each. View it generalized over n_seqs.
    ggml_tensor * new_conv_states = ggml_view_3d(ctx0, conv_output_raw, ssm_d_conv - 1, conv_dim, n_seqs,
            ssm_d_conv * ggml_element_size(conv_output_raw),
            ssm_d_conv * conv_dim * ggml_element_size(conv_output_raw),
            (conv_dim * n_tok + 1) * ggml_element_size(conv_output_raw));
    auto new_conv_states_cont = ggml_cont(ctx0, new_conv_states);
    cb(new_conv_states_cont, "new_conv_states_cont", il);
    ggml_tensor * new_conv_flat = ggml_reshape_2d(ctx0, new_conv_states_cont, conv_state_dim, n_seqs);
    ggml_tensor * new_ssm_flat  = ggml_reshape_2d(ctx0, new_state, ssm_state_dim, n_seqs);
    ggml_tensor * state_cpy;
    if (pxa_fast) {
        // PXA_PERF_NP1_FASTPATH: write the new [state_dim,1] row straight back into the static-offset view,
        // in place -- no scatter copy (mirrors the pre-PR path).
        state_cpy = ggml_concat_inplace(ctx0, new_conv_flat, new_ssm_flat, state_dst, 0);
    } else {
        // PXA_LLAMA_FIX_v4: pack the new rows [state_dim, n_seqs] and SCATTER them back to their absolute seq rows with ONE
        // ggml_set_rows. Single producer + single scatter => no per-token in-place aliasing => allocator-safe at any np.
        ggml_tensor * pxa_new_row = ggml_concat(ctx0, new_conv_flat, new_ssm_flat, 0); // [state_dim, n_seqs]
        state_cpy = ggml_set_rows(ctx0, state_storage, pxa_new_row, pxa_row_idx);
    }
    cb(state_cpy, "state_cpy", il);
    ggml_build_forward_expand(gf, state_cpy);

    return output;
}

ggml_tensor * delta_net::build_gated_output(llama_context & lctx, ggml_context * ctx0, ggml_tensor * ssm_norm, ggml_tensor * ssm_out, ggml_tensor * output, ggml_tensor * z,
        int64_t head_v_dim, int64_t num_v_heads, int64_t n_tok, int il, const llm_build_cb & cb) {

    ggml_tensor * attn_out_2d = ggml_reshape_2d(ctx0, output, head_v_dim, num_v_heads * n_tok);
    ggml_tensor * z_2d        = ggml_reshape_2d(ctx0, z,      head_v_dim, num_v_heads * n_tok);

    ggml_tensor * attn_out_norm = llm_build_context::llm_build_norm(ctx0, attn_out_2d, lctx.model.hparams, ssm_norm, nullptr, LLM_NORM_RMS, cb, il);
    cb(attn_out_norm, "attn_rms_norm", il);
    attn_out_norm = ggml_fused_mul_unary(ctx0, z_2d, attn_out_norm, GGML_UNARY_OP_SILU);
    cb(attn_out_norm, "attn_out_norm", il);

    ggml_tensor * final_output = ggml_reshape_2d(ctx0, attn_out_norm, head_v_dim*num_v_heads, n_tok);
    cb(final_output, "final_output", il);

    ggml_tensor * out = llm_build_context::llm_build_lora_mm(lctx, ctx0, ssm_out, final_output);
    cb(out, "linear_attn_out", il);

    return ggml_reshape_2d(ctx0, out, lctx.model.hparams.n_embd, n_tok);
}

static ggml_tensor * get_input_tensor_sm_graph(ggml_context * ctx, ggml_tensor * input, int id) {
    auto cur = input;
    if (input->op == GGML_OP_REDUCE) {
        auto view_src = input->view_src;
        GGML_ASSERT(view_src);
        cur = input->src[id];
        if (!cur) {
            GGML_ASSERT((input->op_params[4] & (1u << id)) == 0);
            cur = ggml_dup_tensor(ctx, input);
            input->src[id] = cur;
            input->op_params[4] |= (1u << id);
        }
        else if (cur == view_src) {
            cur = input;
        }
    }
    return cur;
}

ggml_tensor * delta_net::build_layer_attn_linear_core(ggml_context * ctx0, ggml_cgraph * gf,
            ggml_tensor * delta_input, ggml_tensor * state_row_idx, ggml_tensor * conv_seq_map, ggml_tensor * state_mask, ggml_tensor * inp_out_ids,
            int64_t n_seqs, bool reset_state_local, int il, const llm_build_cb & cb, int64_t pxa_static_slot) const {

    const int64_t n_tok = delta_input->ne[1];
    GGML_ASSERT(n_seqs > 0 && n_tok % n_seqs == 0);
    // PXA_LLAMA_FIX_v4: state routing is now by the runtime state_row_idx tensor (state_seq_id_local removed).

    auto & model   = lctx.model;
    auto & hparams = model.hparams;
    auto & kv_self = lctx.kv_self;

    int64_t head_k_dim  = hparams.ssm_d_state;
    int64_t num_k_heads = hparams.ssm_n_group;
    int64_t num_v_heads = hparams.ssm_dt_rank;
    int64_t head_v_dim  = hparams.ssm_d_inner / num_v_heads;
    GGML_ASSERT(num_v_heads % num_k_heads == 0);
    int64_t gqa_ratio   = num_v_heads / num_k_heads;

    if (model.split_mode == LLAMA_SPLIT_MODE_GRAPH && kv_self.s_l[il]->extra) {
        GGML_ASSERT(head_k_dim == head_v_dim);
        auto split_s_l = (ggml_split_tensor_t *)kv_self.s_l[il]->extra;
        GGML_ASSERT(split_s_l);
        int n_device = split_s_l->n_device;
        ggml_split_tensor_t *split_wqkv = nullptr, *split_wqkv_gate = nullptr, *split_smm_in = nullptr;
        auto & l = model.layers[il];
        if (l.wqkv && l.wqkv_gate) {
            split_wqkv = (ggml_split_tensor_t *)l.wqkv->extra;
            split_wqkv_gate = (ggml_split_tensor_t *)l.wqkv_gate->extra;
            GGML_ASSERT(split_wqkv && split_wqkv_gate);
            GGML_ASSERT(split_wqkv->n_device == n_device);
            GGML_ASSERT(split_wqkv_gate->n_device == n_device);
        } else {
            split_smm_in = (ggml_split_tensor_t *)l.ssm_in->extra;
            GGML_ASSERT(split_smm_in);
            GGML_ASSERT(split_smm_in->n_device == n_device);
        }
        GGML_ASSERT(n_device > 1);
        std::vector<ggml_tensor *> results(n_device, nullptr);
        bool input_added = false;
        for (int id = 0; id < n_device; ++id) {
            if (!split_s_l->splits[id]) continue;
            auto input = get_input_tensor_sm_graph(ctx0, delta_input, id);
            auto split_norm = (ggml_split_tensor_t *)l.attn_norm->extra;
            GGML_ASSERT(split_norm && split_norm->splits[id]);
            auto cur = llm_build_context::llm_build_norm(ctx0, input, hparams, split_norm->splits[id], nullptr, LLM_NORM_RMS, cb, il);
            int qnext_state_slots = split_s_l->splits[id]->ne[1];
            int il_cb = 1000*il + id;
            int64_t num_k_heads_id, num_v_heads_id;
            ggml_tensor *qkv_mixed, *z;
            if (split_wqkv && split_wqkv_gate) {
                num_k_heads_id = split_wqkv->splits[id]->ne[1]/(head_k_dim*(2 + gqa_ratio));
                num_v_heads_id = num_k_heads_id * gqa_ratio;
                auto p = build_qkvz(lctx, ctx0, split_wqkv->splits[id], split_wqkv_gate->splits[id], cur, il_cb, cb, gf);
                qkv_mixed = p.first;
                z = p.second;
            } else {
                num_k_heads_id = split_smm_in->splits[id]->ne[1]/(2*head_k_dim*(1 + gqa_ratio));
                num_v_heads_id = num_k_heads_id * gqa_ratio;
                auto p = build_qkvz(lctx, ctx0, nullptr, nullptr, split_smm_in->splits[id],
                        head_k_dim, num_k_heads_id, head_v_dim, num_v_heads_id, cur, il, cb, gf);
                qkv_mixed = p.first;
                z = p.second;
            }
            auto split_ssm_dt = (ggml_split_tensor_t *)l.ssm_dt->extra;
            GGML_ASSERT(split_ssm_dt && split_ssm_dt->splits[id] && split_ssm_dt->splits[id]->ne[0] == num_v_heads_id);
            auto split_ssm_a  = (ggml_split_tensor_t *)l.ssm_a->extra;
            GGML_ASSERT(split_ssm_a && split_ssm_a->splits[id] && split_ssm_a->splits[id]->ne[0] == num_v_heads_id);
            ggml_tensor *beta, *gate;
            if (l.ssm_beta_alpha) {
                auto split_ssm_beta_alpha = (ggml_split_tensor_t *)l.ssm_beta_alpha->extra;
                GGML_ASSERT(split_ssm_beta_alpha && split_ssm_beta_alpha->splits[id]);
                auto p = build_beta_gate(lctx, ctx0, split_ssm_beta_alpha->splits[id], nullptr, nullptr, split_ssm_dt->splits[id], split_ssm_a->splits[id],
                        num_k_heads_id, num_v_heads_id, n_seqs, cur, il, cb, gf);
                beta = p.first; gate = p.second;
            } else {
                auto split_ssm_beta = (ggml_split_tensor_t *)l.ssm_beta->extra;
                GGML_ASSERT(split_ssm_beta && split_ssm_beta->splits[id]);
                auto split_ssm_alpha = (ggml_split_tensor_t *)l.ssm_alpha->extra;
                GGML_ASSERT(split_ssm_alpha && split_ssm_alpha->splits[id]);
                auto p = build_beta_gate(lctx, ctx0, nullptr, split_ssm_beta->splits[id], split_ssm_alpha->splits[id], split_ssm_dt->splits[id], split_ssm_a->splits[id],
                        num_k_heads_id, num_v_heads_id, n_seqs, cur, il, cb, gf);
                beta = p.first; gate = p.second;
            }
            auto split_ssm_conv1d = (ggml_split_tensor_t *)l.ssm_conv1d->extra;
            GGML_ASSERT(split_ssm_conv1d && split_ssm_conv1d->splits[id]);
            ggml_tensor * per_step_ckpt = nullptr;
            if (save_per_step_states && il < (int)kv_self.ckpt.per_step_ssm.size()) {
                per_step_ckpt = kv_self.ckpt.per_step_ssm[il][id];
            }
            auto per_step_conv = save_per_step_states && il < (int)kv_self.ckpt.per_step_conv.size() &&
                                 id < (int)kv_self.ckpt.per_step_conv[il].size()
                               ? kv_self.ckpt.per_step_conv[il][id] : nullptr;

            auto output = build_qkv(ctx0, split_s_l->splits[id], split_ssm_conv1d->splits[id], qkv_mixed, state_row_idx, conv_seq_map, state_mask, beta, gate,
                               head_k_dim, num_k_heads_id, head_v_dim, num_v_heads_id, hparams.ssm_d_conv,
                               n_seqs, qnext_state_slots, reset_state_local, hparams.f_norm_rms_eps,
                               l.ssm_beta_alpha ? 0 : 1, il, cb, gf, per_step_ckpt, per_step_conv, pxa_static_slot);
            split_norm = (ggml_split_tensor_t *)l.ssm_norm->extra;
            GGML_ASSERT(split_norm && split_norm->splits[id]);
            auto split_ssm_out = (ggml_split_tensor_t *)l.ssm_out->extra;
            GGML_ASSERT(split_ssm_out && split_ssm_out->splits[id] && split_ssm_out->splits[id]->ne[0] == head_k_dim*num_v_heads_id);
            auto gated_output = build_gated_output(lctx, ctx0, split_norm->splits[id], split_ssm_out->splits[id], output, z, head_v_dim, num_v_heads_id, n_tok, il_cb, cb);
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
        auto cur = ggml_reduce(ctx0, results.data(), n_device, GGML_OP_ADD);
        ggml_build_forward_expand(gf, cur);
        return cur;
    }

    const uint32_t qnext_state_slots = llm_build_context::llama_kv_qnext_state_slots(kv_self);
    GGML_ASSERT(qnext_state_slots > 0);

    int idx = model.default_layer_device[il];
    auto input = delta_input;
    if (input->op == GGML_OP_REDUCE) {
        if (kv_self.s_l[il]) {
            int idx_s_l = ggml_backend_sched_get_backend_idx(lctx.sched, kv_self.s_l[il]->buffer);
            if (idx_s_l >= 0) idx = idx_s_l;
        }
        if (input->src[idx]) {
            input->view_src = input->src[idx];
        }
    }
    auto norm = model.layers[il].attn_norm->extra ? ((ggml_split_tensor_t *)model.layers[il].attn_norm->extra)->splits[idx] : model.layers[il].attn_norm;
    auto cur = llm_build_context::llm_build_norm(ctx0, input, hparams, norm, nullptr, LLM_NORM_RMS, cb, il);

    auto [qkv_mixed, z] = build_qkvz(lctx, ctx0, model.layers[il].wqkv, model.layers[il].wqkv_gate, model.layers[il].ssm_in,
            head_k_dim, num_k_heads, head_v_dim, num_v_heads, cur, il, cb, gf);

    auto [beta, gate] = build_beta_gate(lctx, ctx0, model.layers[il].ssm_beta_alpha, model.layers[il].ssm_beta, model.layers[il].ssm_alpha,
            model.layers[il].ssm_dt, model.layers[il].ssm_a, num_k_heads, num_v_heads, n_seqs, cur, il, cb, gf);

    // Get per-step checkpoint tensor if available
    ggml_tensor * per_step_ckpt = nullptr;
    if (save_per_step_states && il < (int)kv_self.ckpt.per_step_ssm.size()) {
        per_step_ckpt = kv_self.ckpt.per_step_ssm[il].front();
    }

    auto per_step_conv = save_per_step_states && il < (int)kv_self.ckpt.per_step_conv.size() && !kv_self.ckpt.per_step_conv[il].empty()
                       ? kv_self.ckpt.per_step_conv[il].front() : nullptr;

    auto output = build_qkv(ctx0, kv_self.s_l[il], model.layers[il].ssm_conv1d,
        qkv_mixed, state_row_idx, conv_seq_map, state_mask, beta, gate,
        head_k_dim, num_k_heads, head_v_dim, num_v_heads, hparams.ssm_d_conv,
        n_seqs, qnext_state_slots, reset_state_local, hparams.f_norm_rms_eps,
        model.layers[il].ssm_beta_alpha ? 0 : 1, il, cb, gf, per_step_ckpt, per_step_conv, pxa_static_slot);

    auto gated_output = build_gated_output(lctx, ctx0, model.layers[il].ssm_norm, model.layers[il].ssm_out, output, z, head_v_dim, num_v_heads, n_tok, il, cb);
    if (inp_out_ids) {
        gated_output = ggml_get_rows(ctx0, gated_output, inp_out_ids);
        input        = ggml_get_rows(ctx0, input, inp_out_ids);
    }
    output = ggml_add(ctx0, gated_output, input);
    cb(output, "ssm_output", il);
    return output;

}

ggml_tensor * delta_net::build_layer_attn_linear(ggml_context * ctx0, ggml_cgraph * gf,
        ggml_tensor * cur, ggml_tensor * inp_out_ids, int il, const llm_build_cb & cb) const {
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

    GGML_ASSERT(lctx.inp_conv_seq_map != nullptr);
    GGML_ASSERT(lctx.inp_qnext_state_mask != nullptr);
    const int64_t n_tok = batch.n_tokens;

    // PXA_LLAMA_FIX_v4: state-reset mask [1, n_seqs] (per-seq), broadcast over state_dim inside build_qkv.
    auto make_state_mask = [&](int64_t n_seqs) {
        return ggml_view_2d(ctx0, lctx.inp_qnext_state_mask, 1, n_seqs,
                lctx.inp_qnext_state_mask->nb[1], 0);
    };

    if (all_same_seq) {
        // SINGLE sequence: n_seqs = 1. state_row_idx = [1] (this seq's absolute row), conv map = [1, n_tok] view
        // (n_kv=1 -> conv single-seq fast path). One batched call over all n_tok tokens (unchanged prompt/decode path).
        ggml_tensor * state_row_idx = ggml_view_1d(ctx0, lctx.inp_s_seq_qnext, 1, 0);
        ggml_tensor * conv_seq_map  = ggml_view_2d(ctx0, lctx.inp_conv_seq_map, 1, n_tok,
                lctx.inp_conv_seq_map->nb[1], 0);
        ggml_tensor * state_mask = make_state_mask(1);
        bool reset_state = batch.pos != nullptr && batch.pos[0] == 0;
        // PXA_PERF_NP1_FASTPATH: the single sequence's absolute state row is known at build time -- it is exactly
        // what llama_set_inputs writes to inp_s_seq_qnext[0] (batch.seq_id[0][0], clamped >=0). Pass it so build_qkv
        // accesses the row in place (static-offset view) instead of get_rows/set_rows. pxa_seq_sig() encodes this
        // slot for the all_same case, so graph reuse is invalidated if the active slot changes.
        int64_t pxa_static_slot = token_seq_ids.empty() ? 0 : (int64_t) token_seq_ids.front();
        if (pxa_static_slot < 0) pxa_static_slot = 0;
        return build_layer_attn_linear_core(ctx0, gf, cur, state_row_idx, conv_seq_map, state_mask, inp_out_ids, 1, reset_state, il, cb, pxa_static_slot);
    }

    // PXA_LLAMA_MTP_FIX: generalized mixed/MTP path. The batch is decomposed into n_seqs DISTINCT
    // sequences (first-seen order), each with n_seq_tokens contiguous, position-monotonic tokens.
    //   - plain concurrent decode: n_seqs == n_tok, n_seq_tokens == 1 (the old v4 behavior).
    //   - MTP verify (np>1): n_seqs == #slots, n_seq_tokens == n_max+1 (same seq_id repeated).
    // ONE recurrent row is gathered/scattered PER SEQUENCE (no duplicate get_rows/set_rows on the
    // same row -> no racing writes). conv_seq_map is the [n_kv=n_seqs, n_tok] map (row 0 = seq_slot(t),
    // rest -1); the general ssm_conv kernel scans tokens in order, carrying each seq's conv window.
    GGML_ASSERT(seq_decomp.ok && "qwen3next mixed batch is not decomposable (interleaved sequences)");
    GGML_ASSERT(seq_decomp.uniform_tok &&
        "qwen3next batched delta-net requires a uniform tokens-per-sequence count (decode or MTP verify)");
    const int64_t n_seqs = seq_decomp.n_seqs;
    GGML_ASSERT(n_seqs >= 1 && n_tok % n_seqs == 0);

    // state_row_idx: first n_seqs entries of inp_s_seq_qnext hold the DISTINCT absolute seq rows.
    ggml_tensor * state_row_idx = ggml_view_1d(ctx0, lctx.inp_s_seq_qnext, n_seqs, 0);
    // conv_seq_map: view the first n_seqs rows (n_kv) of the full [n_tok, n_tok] map, keeping the
    // full per-token row stride (nb[1]) so the host-filled rows line up. -> [n_kv=n_seqs, n_tok].
    ggml_tensor * conv_seq_map  = ggml_view_2d(ctx0, lctx.inp_conv_seq_map, n_seqs, n_tok,
            lctx.inp_conv_seq_map->nb[1], 0);
    GGML_ASSERT(conv_seq_map->ne[0] == n_seqs && conv_seq_map->ne[1] == n_tok);
    ggml_tensor * state_mask = make_state_mask(n_seqs);

    return build_layer_attn_linear_core(ctx0, gf, cur, state_row_idx, conv_seq_map, state_mask, inp_out_ids, n_seqs, /*reset_state_local*/ false, il, cb);
}


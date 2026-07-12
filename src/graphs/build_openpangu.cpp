#include "../llama-build-context.h"
#include "../llama-model.h"
#include "../llama-context.h"

#include <algorithm>
#include <climits>
#include <vector>

static constexpr int OPENPANGU_CACHE_COPIES_PER_LAYER = 1;
static constexpr int OPENPANGU_COPY_K_CKV = 0;
static constexpr int64_t OPENPANGU_DSA_GATHER_MIN_RATIO = 2;
// Keep these fixed constants in sync with llama_openpangu_chunked_graph_nodes()
// in llama.cpp; the scheduler budget mirrors the chunk loops below.
static constexpr int64_t OPENPANGU_IDX_SCORE_CHUNK = 256;
static constexpr int64_t OPENPANGU_ATT_SCORE_CHUNK = 256;
static constexpr int64_t OPENPANGU_ATT_FULL_KQ_MAX_MIB = 1024;
static constexpr int64_t OPENPANGU_CUDA_GET_ROWS_GRID_Y_MAX = 65535;

static std::vector<llama_context::CacheCopy> & openpangu_cache_copies(llama_context & lctx) {
    return lctx.cparams.mtp_op_type == MTP_OP_NONE ? lctx.openpangu_cache_copies : lctx.openpangu_cache_copies_mtp;
}

static void openpangu_clear_cache_copies(llama_context & lctx) {
    auto & copies = openpangu_cache_copies(lctx);
    std::fill(copies.begin(), copies.end(), llama_context::CacheCopy{});
    std::fill(lctx.dsa_cache_copies.begin(), lctx.dsa_cache_copies.end(), llama_context::CacheCopy{});
}

static void openpangu_register_cache_copy(
        llama_context & lctx, int il, int slot, ggml_tensor * cpy, size_t step) {
    GGML_ASSERT(slot >= 0 && slot < OPENPANGU_CACHE_COPIES_PER_LAYER);
    auto & copies = openpangu_cache_copies(lctx);
    const size_t idx = (size_t) OPENPANGU_CACHE_COPIES_PER_LAYER*il + slot;
    GGML_ASSERT(idx < copies.size());
    copies[idx].cpy = cpy;
    copies[idx].step = step;
}

static uint32_t openpangu_kv_cache_pad(const llama_cparams & cparams) {
    return cparams.flash_attn ? 256u : 32u;
}

static bool openpangu_idx_score_should_chunk(int64_t n_tokens, int64_t chunk) {
    return chunk > 0 && n_tokens > 14 && n_tokens > chunk;
}

static bool openpangu_att_score_should_chunk(
        int64_t n_kv_eff, int64_t n_sinks, int64_t n_head, int64_t n_tokens,
        int64_t chunk, int64_t cap_mib) {
    if (chunk <= 0 || cap_mib <= 0 || n_tokens <= 14 || n_tokens <= chunk) {
        return false;
    }

    const double full_kq_bytes =
        (double) (n_kv_eff + n_sinks) * (double) n_head * (double) n_tokens * (double) sizeof(float);
    const double cap_bytes = (double) cap_mib * 1024.0 * 1024.0;
    return full_kq_bytes > cap_bytes;
}

static bool openpangu_dsa_gather_should_engage(int64_t n_kv, int64_t n_tokens, int64_t topk, int64_t pad) {
    // Gather must prune at least half the cache to beat get_rows/cast/cont-transpose copy
    // overhead; measured 2K regression at ratio ~1 where top_k covers nearly all cache rows.
    return n_tokens <= 14 && n_kv >= OPENPANGU_DSA_GATHER_MIN_RATIO*topk + pad + n_tokens;
}

static bool openpangu_dsa_prefill_gather_should_engage(
        int64_t n_kv, int64_t n_tokens, int64_t chunk_start, int64_t chunk_size, int64_t topk, int64_t pad) {
    GGML_UNUSED(chunk_size);
    if (n_tokens <= 14 || topk <= 0 || chunk_start < 0 || chunk_start >= n_tokens || n_kv < n_tokens) {
        return false;
    }

    const int64_t min_causal_kv = n_kv - n_tokens + chunk_start;
    return min_causal_kv >= OPENPANGU_DSA_GATHER_MIN_RATIO*topk + pad;
}

static bool openpangu_dsa_gather_rows_fit_cuda(int64_t topk, int64_t n_tokens) {
    return topk > 0 && n_tokens > 0 && topk <= OPENPANGU_CUDA_GET_ROWS_GRID_Y_MAX / n_tokens;
}

static int64_t openpangu_dsa_gather_tokens_per_get_rows(int64_t topk) {
    GGML_ASSERT(topk > 0 && topk <= OPENPANGU_CUDA_GET_ROWS_GRID_Y_MAX);
    return std::max<int64_t>(1, OPENPANGU_CUDA_GET_ROWS_GRID_Y_MAX / topk);
}

static ggml_tensor * openpangu_build_v_latent_from_k(
        ggml_context * ctx, const llama_kv_cache & kv_self, int il,
        int64_t kv_lora_rank, int64_t n_kv_view, int64_t win_off) {
    ggml_tensor * kl = kv_self.k_l[il];
    if (ggml_is_quantized(kl->type)) {
        ggml_tensor * full_view = ggml_view_2d(ctx, kl, kl->ne[0], n_kv_view,
                                               kl->nb[1], (size_t) win_off*kl->nb[1]);
        ggml_tensor * full_f32 = ggml_cast(ctx, full_view, GGML_TYPE_F32);
        ggml_tensor * v_src = ggml_view_2d(ctx, full_f32, kv_lora_rank, n_kv_view,
                                           full_f32->nb[1], 0);
        return ggml_cont(ctx, ggml_transpose(ctx, v_src));
    }

    ggml_tensor * v_src = ggml_view_2d(ctx, kl, kv_lora_rank, n_kv_view,
                                       kl->nb[1], (size_t) win_off*kl->nb[1]);
    return ggml_cont(ctx, ggml_transpose(ctx, v_src));
}

static ggml_tensor * openpangu_build_k_latent_for_read(
        ggml_context * ctx, const llama_kv_cache & kv_self, int il,
        int64_t n_kv_view, int64_t win_off) {
    ggml_tensor * kl = kv_self.k_l[il];
    ggml_tensor * k_view = ggml_view_2d(ctx, kl, kl->ne[0], n_kv_view,
                                        kl->nb[1], (size_t) win_off*kl->nb[1]);
    if (!ggml_is_quantized(kl->type)) {
        return k_view;
    }

    return ggml_cast(ctx, k_view, GGML_TYPE_F32);
}

static ggml_tensor * openpangu_cast_for_latent_cache_write(ggml_context * ctx, ggml_tensor * src, ggml_tensor * kl) {
    return ggml_is_quantized(kl->type) && src->type != GGML_TYPE_F32 ? ggml_cast(ctx, src, GGML_TYPE_F32) : src;
}

static ggml_tensor * openpangu_cast_gathered_latent_for_cache_type(ggml_context * ctx, ggml_tensor * src, ggml_tensor * kl) {
    return !ggml_is_quantized(kl->type) && src->type != kl->type ? ggml_cast(ctx, src, kl->type) : src;
}

static ggml_tensor * openpangu_build_swa_mask_for_graph(llm_build_context & llm, uint32_t window, bool * windowed) {
    *windowed = false;
    llm.lctx.openpangu_swa_window_view = {};

    if (window == 0) {
        return nullptr;
    }

    const uint32_t pad = openpangu_kv_cache_pad(llm.cparams);
    const llama_openpangu_swa_window_view view =
        llama_openpangu_calc_swa_window_view(llm.n_kv, llm.n_tokens, window, pad);

    if (!view.engaged) {
        return llm.build_inp_KQ_mask_swa();
    }

    llm.lctx.openpangu_swa_window_view = {
        true,
        llm.n_kv,
        llm.n_tokens,
        window,
        pad,
        view.w_view,
        view.win_off,
    };
    *windowed = true;
    return llm.build_inp_KQ_mask_swa_win(view.w_view);
}

// openPangu-2.0-Flash graph.
//
// Attention runs absorbed MLA over a latent KV cache: per position the cache stores only
// the 512-d compressed latent plus the 64-d roped k_pe (k_l, straight layout);
// q_nope is projected into latent space through attn_k_b and the
// attention output is up-projected through attn_v_b after the weighted sum. Per-head K/V
// never materialize.
//
// Pangu-specific pieces implemented here (see vault Stage-2b forward spec):
//   - mHC / Hyper-Connections: 4 parallel residual streams mixed per sublayer via a phi
//     projection (h_pre combine-in, h_post/h_res scatter-out) with a 20-iter Sinkhorn.
//   - MoME: causal depthwise conv (k=3) on the q-lora latent, compressed-kv latent, attn out;
//     decode taps come from a recurrent ggml_ssm_conv state slot (openpangu_causal_conv below).
//   - param_sink: 128 learned latent-space KV entries per layer, prepended to every query's
//     attention span outside the causal/window/top-k masks.
//   - DSA + SWA schedule: windowed base layers use the SWA mask; windowless base layers run
//     the lightning indexer over a per-position indexer-key cache and restrict
//     attention to the top-k scored positions plus the sinks. Schedule-less GGUFs run dense.
//     For prompts <= 512 tokens both mechanisms are inert and output is bit-exact to dense.
//   - sandwich norms (post_attention / pre_mlp / post_mlp) + block_post on a layer subset.
//   - NextN/MTP layers (build_openpangu_mtp below) drive the mtp speculative framework,
//     chaining conv state through the same recurrent slot.

// --- causal depthwise conv1d, kernel=3: out[t] = w0*x[t-2] + w1*x[t-1] + w2*x[t] (per channel) ---
// x: [C, n_tokens]; w: ggml tensor with ne = {3, 1, C} (kernel-major). Returns [C, n_tokens].
// MOME: out = x + depthwise causal conv1d(k=3). Every Infer call site passes
// residual_connection=1; the tap magnitudes confirm the conv is a small learned
// perturbation on top of the identity, not a standalone filter.
//
// Conv state: `state_all` is this layer's cache_s_l recurrent slot table
// [2*col_ne, qnext_state_slots]. Slot 0 holds the single-sequence decode state, with
// this site's two taps packed at float offset 2*site_off. The buffer is zeroed at cache
// allocation and reset at pos 0, preserving zero history at sequence start (pos-0 graphs
// are discarded from reuse via reset_previous, so the baked reset never runs at pos > 0).
// Speculative rollback snapshots/restores the whole slot via the spec checkpoint.
static ggml_tensor * openpangu_causal_conv(ggml_context * ctx, ggml_cgraph * gf,
                                           ggml_tensor * x, ggml_tensor * w,
                                           ggml_tensor * state_all, int64_t site_off,
                                           ggml_tensor * seq_qnext, bool reset_state) {
    const int64_t C = x->ne[0];
    const int64_t T = x->ne[1];
    // weight is stored f16 with ne = {3, C}: per-channel taps contiguous. ggml_ssm_conv
    // expects an f32 kernel, so cast once (tiny tensor).
    ggml_tensor * wc = ggml_reshape_2d(ctx, ggml_cast(ctx, w, GGML_TYPE_F32), 3, C);

    GGML_ASSERT(state_all != nullptr);
    GGML_ASSERT(state_all->type == GGML_TYPE_F32);
    GGML_ASSERT(state_all->ne[0] >= 2*(site_off + C));
    GGML_ASSERT(seq_qnext != nullptr);
    GGML_ASSERT(seq_qnext->type == GGML_TYPE_I32);
    GGML_ASSERT(seq_qnext->ne[0] == 1);
    GGML_ASSERT(seq_qnext->ne[1] == T);

    ggml_tensor * state_flat = ggml_view_2d(ctx, state_all, 2*C, 1, state_all->nb[1],
            2*site_off*ggml_element_size(state_all));
    ggml_tensor * state_in = reset_state ? ggml_scale(ctx, state_flat, 0.0f) : state_flat;
    ggml_tensor * states = ggml_reshape_3d(ctx, state_in, 2, C, 1);

    ggml_tensor * conv_raw = ggml_ssm_conv(ctx, states, x, wc, seq_qnext, nullptr);
    ggml_tensor * conv = ggml_view_2d(ctx, conv_raw, C, T, C*ggml_element_size(conv_raw), 0);
    ggml_tensor * out = ggml_add(ctx, x, conv);

    ggml_tensor * new_states = ggml_view_2d(ctx, conv_raw, 2, C,
            3*ggml_element_size(conv_raw),
            (1 + C*T)*ggml_element_size(conv_raw));
    ggml_tensor * new_states_cont = ggml_cont(ctx, new_states);
    ggml_tensor * new_state_flat = ggml_reshape_2d(ctx, new_states_cont, 2*C, 1);
    ggml_build_forward_expand(gf, ggml_cpy(ctx, new_state_flat, state_flat));
    return out;
}

// Attention sublayer body, shared by the base layers and the NextN/MTP head.
// x_normed = input-layernormed hidden [n_embd, T]; returns post-o_proj output [n_embd, T].
// conv_state is the recurrent MoME state slot. seq_qnext is the [1, T] sequence-id input
// used by ggml_ssm_conv and is shared by all three conv sites.
ggml_tensor * llm_build_context::build_openpangu_attention(
        ggml_cgraph * gf, const llama_layer & layer, int il, ggml_tensor * x_normed,
        ggml_tensor * KQ_mask, ggml_tensor * inp_pos,
        ggml_tensor * conv_state, ggml_tensor * seq_qnext,
        float kq_scale, bool KQ_mask_swa_windowed) {
    const int64_t n_embd_head_qk_rope = hparams.n_rot;                       // 64
    const int64_t n_embd_head_k       = hparams.n_embd_head_k(0);            // 192
    const int64_t n_embd_head_qk_nope = n_embd_head_k - n_embd_head_qk_rope; // 128
    const int64_t n_embd_head_v       = hparams.n_embd_head_v(0);            // 128
    const int64_t kv_lora_rank        = hparams.n_lora_kv;                   // 512
    const int64_t q_lora_rank         = hparams.n_lora_q;                    // 1024

    // MoME conv-state site offsets within one slot (channels): [qa | compresskv | o]
    const int64_t conv_off_qa  = 0;
    const int64_t conv_off_ckv = q_lora_rank;
    const int64_t conv_off_o   = q_lora_rank + kv_lora_rank;
    const bool reset_conv_state = batch.pos && batch.n_tokens > 0 && batch.pos[0] == 0;

    ggml_tensor * cur = x_normed;

    // --- Q path: q_a -> qa_conv -> q_a_norm -> q_b ---
    ggml_tensor * q_lora = ggml_mul_mat(ctx0, layer.wq_a, cur);        // [q_lora_rank, T]
    q_lora = openpangu_causal_conv(ctx0, gf, q_lora, layer.qa_conv, conv_state, conv_off_qa, seq_qnext, reset_conv_state);
    if (il == 0) ggml_set_name(q_lora, "opg0_qlora_conv");
    q_lora = llm_build_norm(ctx0, q_lora, hparams, layer.attn_q_a_norm, NULL, LLM_NORM_RMS, cb, il);
    if (il == 0) ggml_set_name(q_lora, "opg0_qlora_norm");
    ggml_tensor * q = ggml_mul_mat(ctx0, layer.wq_b, q_lora);          // [n_head*192, T]
    q = ggml_reshape_3d(ctx0, q, n_embd_head_k, n_head, n_tokens);
    ggml_tensor * q_nope = ggml_view_3d(ctx0, q, n_embd_head_qk_nope, n_head, n_tokens, q->nb[1], q->nb[2], 0);
    ggml_tensor * q_rope = ggml_view_3d(ctx0, q, n_embd_head_qk_rope, n_head, n_tokens, q->nb[1], q->nb[2],
                                        n_embd_head_qk_nope*ggml_element_size(q));
    q_rope = ggml_rope_ext(ctx0, ggml_cont(ctx0, q_rope), inp_pos, nullptr, n_rot, rope_type,
                           n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
    // absorbed (MLA-latent) queries: q_nope projected into the 512-latent through the
    // load-derived wk_b, then ++ roped q_pe -> one [576, T, H] query against the latent cache
    ggml_tensor * qn = ggml_cont(ctx0, ggml_permute(ctx0, ggml_cont(ctx0, q_nope), 0, 2, 1, 3)); // [128, T, H]
    ggml_tensor * qp = ggml_cont(ctx0, ggml_permute(ctx0, q_rope, 0, 2, 1, 3));                  // [64, T, H]
    // wk_b loads 2D [128, H*512] from the converter split (head-major rows) or 3D from the
    // load-time derivation; either way the data is [d, r, h] — reshape to the batched form
    ggml_tensor * wk_b3 = ggml_reshape_3d(ctx0, layer.wk_b, n_embd_head_qk_nope, kv_lora_rank, n_head);
    ggml_tensor * q_lat = ggml_mul_mat(ctx0, wk_b3, qn);                                          // [512, T, H]
    ggml_tensor * q_all = ggml_concat(ctx0, q_lat, qp, 0);                                       // [576, T, H]
    if (il == 0) ggml_set_name(q_all, "opg0_q_lat");

    // --- KV path: kv_a -> split -> compresskv_conv -> kv_a_norm -> kv_b ---
    ggml_tensor * kv = ggml_mul_mat(ctx0, layer.wkv_a_mqa, cur);       // [kv_lora+64, T]
    ggml_tensor * ckv = ggml_cont(ctx0, ggml_view_2d(ctx0, kv, kv_lora_rank, n_tokens, kv->nb[1], 0));
    ggml_tensor * k_pe = ggml_cont(ctx0, ggml_view_2d(ctx0, kv, n_embd_head_qk_rope, n_tokens, kv->nb[1],
                                        kv_lora_rank*ggml_element_size(kv)));
    ckv = openpangu_causal_conv(ctx0, gf, ckv, layer.kv_conv, conv_state, conv_off_ckv, seq_qnext, reset_conv_state);
    ckv = llm_build_norm(ctx0, ckv, hparams, layer.attn_kv_a_norm, NULL, LLM_NORM_RMS, cb, il);
    if (il == 0) ggml_set_name(ckv, "opg0_ckv_norm");
    // rope k_pe (shared across heads)
    k_pe = ggml_reshape_3d(ctx0, k_pe, n_embd_head_qk_rope, 1, n_tokens);
    k_pe = ggml_rope_ext(ctx0, k_pe, inp_pos, nullptr, n_rot, rope_type,
                         n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
    ggml_tensor * k_pe2d = ggml_reshape_2d(ctx0, k_pe, n_embd_head_qk_rope, n_tokens);

    // ---- latent cache store: per position [ckv 512 | roped k_pe 64] straight into k_l.
    // The value-side latent is rebuilt from k_l per graph; per-head K/V are never materialized.
    {
        ggml_tensor * kl = kv_self.k_l[il];
        ggml_tensor * ckv_store = ckv;
        ggml_tensor * kpe_store = k_pe2d;
        if (ggml_is_quantized(kl->type)) {
            ckv_store = openpangu_cast_for_latent_cache_write(ctx0, ckv_store, kl);
            kpe_store = openpangu_cast_for_latent_cache_write(ctx0, kpe_store, kl);
        }
        ggml_tensor * k_latent = ggml_concat(ctx0, ckv_store, kpe_store, 0);
        ggml_tensor * kl_full = ggml_view_2d(ctx0, kl, kv_lora_rank + n_embd_head_qk_rope,
                                             n_tokens, kl->nb[1], kv_head*kl->nb[1]);
        ggml_tensor * cpy_kl = ggml_cpy(ctx0, k_latent, kl_full);
        openpangu_register_cache_copy(lctx, il, OPENPANGU_COPY_K_CKV, cpy_kl, kl->nb[1]);
        ggml_build_forward_expand(gf, cpy_kl);
    }

    // ---- DSA lightning indexer: per-query top-k selection mask (DSA layers only) ----
    // Reference (Infer _pangu_torch_calib): q_idx = wq_b(q_lora_normed) [24 heads x 128],
    // k_idx = rms(k_norm)(wk(x_normed)) [128, shared across heads], both NEOX-roped on the
    // FIRST n_rot channels (opposite order vs the main head's [nope|rope] split);
    // score[t,s] = sum_g weights_proj(x)[t,g] * relu(q_idx[t,g]·k_idx[s]) in f32, causal,
    // then top-k. Selection only prunes when the causal window exceeds top_k; below that it
    // covers everything, so the mask is skipped and the layer is exactly dense.
    ggml_tensor * sel_mask = nullptr;   // [n_kv, T] additive mask: 0 = selected, -1e30 = pruned
    ggml_tensor * sel_idx  = nullptr;   // [topk, T] per-token top-k rows, reused by gathered DSA
    bool dsa_gather_engaged = false;
    int64_t dsa_topk = 0;
    ggml_tensor * idx_cache = (size_t) il < kv_self.kr_l.size() ? kv_self.kr_l[il] : nullptr;
    if (idx_cache && layer.indexer_attn_q_b) {
        const int64_t n_ihead = hparams.indexer_n_head;    // 24
        const int64_t d_idx   = hparams.indexer_head_size; // 128
        int64_t topk = hparams.indexer_top_k;              // 2048
        if (lctx.cparams.dsa_top_k > 0) {
            topk = lctx.cparams.dsa_top_k;
        }
        GGML_ASSERT(topk > 0 && topk <= INT_MAX);
        dsa_topk = topk;

        const uint32_t pad = openpangu_kv_cache_pad(cparams);
        const bool is_base_graph = cparams.mtp_op_type == MTP_OP_NONE;
        const bool dsa_gather_predicate =
            openpangu_dsa_gather_should_engage(n_kv, n_tokens, topk, pad);
        const bool dsa_gather_allowed =
            is_base_graph && dsa_gather_predicate &&
            openpangu_dsa_gather_rows_fit_cuda(topk, n_tokens);

        // indexer keys for this batch -> position-indexed cache (write-before-read holds by
        // graph order, same as the kv store; committed columns never change -> rollback-safe)
        ggml_tensor * k_idx = ggml_mul_mat(ctx0, layer.indexer_attn_k, x_normed);   // [d_idx, T]
        k_idx = llm_build_norm(ctx0, k_idx, hparams, layer.indexer_k_norm, layer.indexer_k_norm_b,
                               layer.indexer_k_norm_b ? LLM_NORM : LLM_NORM_RMS, cb, il);
        ggml_tensor * k_idx_rope = ggml_view_2d(ctx0, k_idx, n_embd_head_qk_rope, n_tokens, k_idx->nb[1], 0);
        ggml_tensor * k_idx_pass = ggml_view_2d(ctx0, k_idx, d_idx - n_embd_head_qk_rope, n_tokens, k_idx->nb[1],
                                                n_embd_head_qk_rope*ggml_element_size(k_idx));
        k_idx_rope = ggml_rope_ext(ctx0, ggml_reshape_3d(ctx0, ggml_cont(ctx0, k_idx_rope), n_embd_head_qk_rope, 1, n_tokens),
                                   inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                                   ext_factor, attn_factor, beta_fast, beta_slow);
        k_idx = ggml_concat(ctx0, ggml_reshape_2d(ctx0, k_idx_rope, n_embd_head_qk_rope, n_tokens),
                            ggml_cont(ctx0, k_idx_pass), 0);                        // [d_idx, T]
        if (il == 0) ggml_set_name(k_idx, "opg0_idx_k");
        ggml_tensor * idx_w = ggml_view_2d(ctx0, idx_cache, d_idx, n_tokens, idx_cache->nb[1], kv_head*idx_cache->nb[1]);
        ggml_tensor * cpy_idx = ggml_cpy(ctx0, k_idx, idx_w);
        if ((size_t) il < lctx.dsa_cache_copies.size()) {
            lctx.dsa_cache_copies[il].cpy  = cpy_idx;
            lctx.dsa_cache_copies[il].step = idx_cache->nb[1];
        }
        ggml_build_forward_expand(gf, cpy_idx);

        if (n_kv > topk) {
            // indexer queries
            ggml_tensor * q_idx = ggml_mul_mat(ctx0, layer.indexer_attn_q_b, q_lora); // [n_ihead*d_idx, T]
            q_idx = ggml_reshape_3d(ctx0, q_idx, d_idx, n_ihead, n_tokens);
            ggml_tensor * q_idx_rope = ggml_view_3d(ctx0, q_idx, n_embd_head_qk_rope, n_ihead, n_tokens,
                                                    q_idx->nb[1], q_idx->nb[2], 0);
            ggml_tensor * q_idx_pass = ggml_view_3d(ctx0, q_idx, d_idx - n_embd_head_qk_rope, n_ihead, n_tokens,
                                                    q_idx->nb[1], q_idx->nb[2], n_embd_head_qk_rope*ggml_element_size(q_idx));
            q_idx_rope = ggml_rope_ext(ctx0, ggml_cont(ctx0, q_idx_rope), inp_pos, nullptr, n_rot, rope_type,
                                       n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
            q_idx = ggml_concat(ctx0, q_idx_rope, ggml_cont(ctx0, q_idx_pass), 0);   // [d_idx, n_ihead, T]
            if (il == 0) ggml_set_name(q_idx, "opg0_idx_q");

            ggml_tensor * k_all_idx = ggml_view_2d(ctx0, idx_cache, d_idx, n_kv, idx_cache->nb[1], 0);
            ggml_tensor * w_idx = ggml_mul_mat(ctx0, layer.indexer_proj, x_normed);  // [n_ihead, T]

            const bool chunk_scores = openpangu_idx_score_should_chunk(n_tokens, OPENPANGU_IDX_SCORE_CHUNK);
            const bool defer_sel_mask_to_att_chunks =
                !dsa_gather_allowed &&
                openpangu_att_score_should_chunk(n_kv, hparams.param_sink_number, n_head, n_tokens,
                        OPENPANGU_ATT_SCORE_CHUNK, OPENPANGU_ATT_FULL_KQ_MAX_MIB);
            if (lctx.cparams.fused_idx_topk) {
                // Fused indexer top-k (CPU-only op): one op computes sum_g w * relu(q.k) + causal
                // mask -> top-k without materializing the [n_kv, n_ihead, T] score tensor, so no
                // score chunking is needed. CUDA backends do not implement GGML_OP_INDEXER_TOPK;
                // the scheduler runs it on CPU and copies the operands across the backend boundary.
                // The op reads the mask row-strided, so the raw view suffices.
                ggml_tensor * idx_mask = ggml_view_2d(ctx0, KQ_mask, n_kv, n_tokens,
                                                      KQ_mask->nb[1], 0);
                sel_idx = ggml_indexer_topk(ctx0, k_all_idx, q_idx, w_idx, idx_mask,
                                            GGML_UNARY_OP_RELU, (int) topk);          // [topk, T] i32
                if (il == 0) ggml_set_name(sel_idx, "opg0_idx_sel");
                dsa_gather_engaged = dsa_gather_allowed;
                if (dsa_gather_engaged) {
                    GGML_ASSERT(n_kv >= topk + (int64_t) pad + n_tokens);
                } else if (defer_sel_mask_to_att_chunks) {
                    sel_mask = nullptr;
                } else {
                    ggml_tensor * base = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, 1, n_kv, n_tokens);
                    base = ggml_fill(ctx0, base, -1e30f);
                    ggml_tensor * zeros = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, 1, topk, n_tokens);
                    zeros = ggml_fill(ctx0, zeros, 0.0f);
                    sel_mask = ggml_set_rows(ctx0, base, zeros, sel_idx);
                    sel_mask = ggml_reshape_2d(ctx0, sel_mask, n_kv, n_tokens);
                }
            } else if (chunk_scores) {
                ggml_tensor * sel_mask_parts = nullptr;
                for (int64_t c0 = 0; c0 < n_tokens; c0 += OPENPANGU_IDX_SCORE_CHUNK) {
                    const int64_t tc = std::min<int64_t>(OPENPANGU_IDX_SCORE_CHUNK, n_tokens - c0);
                    ggml_tensor * q_idx_c = ggml_view_3d(ctx0, q_idx, d_idx, n_ihead, tc,
                                                         q_idx->nb[1], q_idx->nb[2], (size_t) c0*q_idx->nb[2]);
                    q_idx_c = ggml_cont(ctx0, q_idx_c);
                    ggml_tensor * w_idx_c = ggml_view_2d(ctx0, w_idx, n_ihead, tc, w_idx->nb[1], (size_t) c0*w_idx->nb[1]);
                    w_idx_c = ggml_cont(ctx0, w_idx_c);

                    // Scores over the whole cache window, causal-masked before selection.
                    // The chunked path never materializes the legacy [n_kv, n_ihead, T] score tensor.
                    ggml_tensor * sc_c = ggml_mul_mat(ctx0, k_all_idx, q_idx_c);              // [n_kv, n_ihead, Tc]
                    sc_c = ggml_relu(ctx0, sc_c);
                    sc_c = ggml_mul(ctx0, sc_c, ggml_reshape_3d(ctx0, w_idx_c, 1, n_ihead, tc));
                    sc_c = ggml_cont(ctx0, ggml_permute(ctx0, sc_c, 1, 0, 2, 3));             // [n_ihead, n_kv, Tc]
                    sc_c = ggml_reshape_2d(ctx0, ggml_sum_rows(ctx0, sc_c), n_kv, tc);        // [n_kv, Tc]

                    sc_c = ggml_add(ctx0, sc_c, ggml_cont(ctx0, ggml_view_2d(ctx0, KQ_mask, n_kv, tc,
                                    KQ_mask->nb[1], (size_t) c0*KQ_mask->nb[1])));
                    ggml_tensor * sel_idx_c = ggml_top_k(ctx0, sc_c, (int) topk);             // [topk, Tc] i32
                    sel_idx = sel_idx == nullptr ? sel_idx_c : ggml_concat(ctx0, sel_idx, sel_idx_c, 1);

                    if (!dsa_gather_allowed && !defer_sel_mask_to_att_chunks) {
                        ggml_tensor * base_c = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, 1, n_kv, tc);
                        base_c = ggml_fill(ctx0, base_c, -1e30f);
                        ggml_tensor * zeros_c = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, 1, topk, tc);
                        zeros_c = ggml_fill(ctx0, zeros_c, 0.0f);
                        ggml_tensor * sel_mask_c = ggml_set_rows(ctx0, base_c, zeros_c, sel_idx_c);
                        sel_mask_c = ggml_reshape_2d(ctx0, sel_mask_c, n_kv, tc);
                        sel_mask_parts = sel_mask_parts == nullptr ? sel_mask_c : ggml_concat(ctx0, sel_mask_parts, sel_mask_c, 1);
                    }
                }
                if (il == 0) ggml_set_name(sel_idx, "opg0_idx_sel");
                dsa_gather_engaged = dsa_gather_allowed;
                if (dsa_gather_engaged) {
                    GGML_ASSERT(n_kv >= topk + (int64_t) pad + n_tokens);
                } else if (defer_sel_mask_to_att_chunks) {
                    sel_mask = nullptr;
                } else {
                    sel_mask = sel_mask_parts;
                }
            } else {
                GGML_ASSERT(n_tokens <= 14 || OPENPANGU_IDX_SCORE_CHUNK == 0 || n_tokens <= OPENPANGU_IDX_SCORE_CHUNK);
                ggml_tensor * sc = ggml_mul_mat(ctx0, k_all_idx, q_idx);                 // [n_kv, n_ihead, T]
                sc = ggml_relu(ctx0, sc);
                sc = ggml_mul(ctx0, sc, ggml_reshape_3d(ctx0, w_idx, 1, n_ihead, n_tokens));
                sc = ggml_cont(ctx0, ggml_permute(ctx0, sc, 1, 0, 2, 3));                // [n_ihead, n_kv, T]
                sc = ggml_reshape_2d(ctx0, ggml_sum_rows(ctx0, sc), n_kv, n_tokens);     // [n_kv, T]
                sc = ggml_add(ctx0, sc, ggml_cont(ctx0, ggml_view_2d(ctx0, KQ_mask, n_kv, n_tokens, KQ_mask->nb[1], 0)));
                if (il == 0) ggml_set_name(sc, "opg0_idx_scores");

                // Exact top-k -> additive mask: scatter zeros into a -1e30 base at the selected
                // positions ([1, n_kv, T] row layout makes set_rows a per-query scatter).
                // The legacy scatter path consumes this strided top_k view directly; the gathered
                // path below flattens a tiny contiguous copy for one get_rows.
                sel_idx = ggml_top_k(ctx0, sc, (int) topk);                              // [topk, T] i32
                if (il == 0) ggml_set_name(sel_idx, "opg0_idx_sel");
                dsa_gather_engaged = dsa_gather_allowed;
                if (dsa_gather_engaged) {
                    GGML_ASSERT(n_kv >= topk + (int64_t) pad + n_tokens);
                } else {
                    ggml_tensor * base = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, 1, n_kv, n_tokens);
                    base = ggml_fill(ctx0, base, -1e30f);
                    ggml_tensor * zeros = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, 1, topk, n_tokens);
                    zeros = ggml_fill(ctx0, zeros, 0.0f);
                    sel_mask = ggml_set_rows(ctx0, base, zeros, sel_idx);
                    sel_mask = ggml_reshape_2d(ctx0, sel_mask, n_kv, n_tokens);
                }
            }
        }
    }

    // ---- param_sink: 128 learned latent-KV entries prepended to the sequence ----
    // The sinks are native latent-space entries: kv_a_norm of the learned latent IS the
    // key/value latent; sink_k_pe is used rope-free. Sinks are visible to every query.
    const int64_t NS = hparams.param_sink_number;
    GGML_ASSERT(layer.param_sink_blk && layer.param_sink_lat_t);
    ggml_tensor * sink_blk = layer.param_sink_blk;                                          // [576, NS]
    ggml_tensor * s_lat_t = layer.param_sink_lat_t;                                        // [NS, 512]

    // ---- latent attention over [sinks ++ cached tokens] (flash_attn is forced off) ----
    // Keep sinks and cached tokens separate until after KQ so f16 latent caches do not need
    // unsupported non-f32 concat along dim1.
    const bool use_swa_window = KQ_mask_swa_windowed && lctx.openpangu_swa_window_view.active;
    const bool use_dsa_gather = dsa_gather_engaged && sel_idx != nullptr;
    const int64_t n_kv_attn   = use_dsa_gather ? dsa_topk :
                                use_swa_window ? lctx.openpangu_swa_window_view.w_view  : n_kv;
    const int64_t win_off     = use_swa_window ? lctx.openpangu_swa_window_view.win_off : 0;
    if (sel_mask) {
        GGML_ASSERT(!use_swa_window && "openPangu DSA/indexer layers must not use SWA window views");
    }
    if (use_dsa_gather) {
        GGML_ASSERT(!use_swa_window && "openPangu gathered DSA layers must not use SWA window views");
        GGML_ASSERT(dsa_topk > 0 && n_kv_attn == dsa_topk);
        GGML_ASSERT(hparams.f_max_alibi_bias == 0.0f && "maskless gathered DSA cannot carry ALiBi bias");
    }

    ggml_tensor * kl_all = nullptr;
    ggml_tensor * vl_all = nullptr;
    if (!use_dsa_gather) {
        kl_all = openpangu_build_k_latent_for_read(ctx0, kv_self, il, n_kv_attn, win_off);
    }
    auto get_vl_all = [&]() -> ggml_tensor * {
        if (vl_all == nullptr) {
            vl_all = openpangu_build_v_latent_from_k(ctx0, kv_self, il, kv_lora_rank, n_kv_attn, win_off);
        }
        return vl_all;
    };

    ggml_tensor * k_gath = nullptr;
    ggml_tensor * kq_cache = nullptr;
    if (use_dsa_gather) {
        ggml_tensor * kq_sinks = ggml_mul_mat(ctx0, sink_blk, q_all);                       // [NS, T, H]
        if (n_tokens == 1) {
            ggml_tensor * kl_full = ggml_view_2d(ctx0, kv_self.k_l[il], kv_lora_rank + n_embd_head_qk_rope, n_kv,
                                                 kv_self.k_l[il]->nb[1], 0);
            ggml_tensor * sel_idx_flat = ggml_cont_2d(ctx0, sel_idx, dsa_topk, 1);            // [topk] i32
            k_gath = ggml_get_rows(ctx0, kl_full, sel_idx_flat);                              // [576, topk] f32
            k_gath = openpangu_cast_gathered_latent_for_cache_type(ctx0, k_gath, kv_self.k_l[il]);
            k_gath = ggml_reshape_3d(ctx0, k_gath, kv_lora_rank + n_embd_head_qk_rope, dsa_topk, 1);
            kq_cache = ggml_mul_mat(ctx0,
                    ggml_reshape_2d(ctx0, k_gath, kv_lora_rank + n_embd_head_qk_rope, dsa_topk),
                    q_all);                                                                   // [topk, 1, H]
        } else {
            ggml_tensor * kl_full = ggml_view_2d(ctx0, kv_self.k_l[il],
                                                 kv_lora_rank + n_embd_head_qk_rope, n_kv,
                                                 kv_self.k_l[il]->nb[1], 0);
            ggml_tensor * sel_idx_flat = ggml_cont_2d(ctx0, sel_idx, dsa_topk*n_tokens, 1);   // [topk*T] i32
            k_gath = ggml_get_rows(ctx0, kl_full, sel_idx_flat);                              // [576, topk*T] f32
            k_gath = openpangu_cast_gathered_latent_for_cache_type(ctx0, k_gath, kv_self.k_l[il]);
            k_gath = ggml_reshape_3d(ctx0, k_gath, kv_lora_rank + n_embd_head_qk_rope,
                                     dsa_topk, n_tokens);                                    // [576, topk, T]
            ggml_tensor * q_gath = ggml_cont(ctx0, ggml_permute(ctx0, q_all, 0, 2, 1, 3));   // [576, H, T]
            kq_cache = ggml_mul_mat(ctx0, k_gath, q_gath);                                  // [topk, H, T]
            kq_cache = ggml_cont(ctx0, ggml_permute(ctx0, kq_cache, 0, 2, 1, 3));            // [topk, T, H]
        }
        ggml_tensor * kq = ggml_concat(ctx0, kq_sinks, kq_cache, 0);                         // [NS+topk, T, H]
        // sel_idx came from scores after adding the causal KQ_mask. The engagement bound gives
        // every token at least topk valid positions, so all gathered rows are visible here.
        kq = ggml_soft_max_ext(ctx0, kq, nullptr, kq_scale, hparams.f_max_alibi_bias);

        ggml_tensor * kq_s = ggml_view_3d(ctx0, kq, NS, n_tokens, n_head, kq->nb[1], kq->nb[2], 0);
        ggml_tensor * kq_c = ggml_view_3d(ctx0, kq, n_kv_attn, n_tokens, n_head, kq->nb[1], kq->nb[2],
                                          NS*ggml_element_size(kq));
        ggml_tensor * kqv_cache = nullptr;
        GGML_ASSERT(k_gath != nullptr);
        if (n_tokens == 1) {
            ggml_tensor * v_gath = ggml_view_2d(ctx0, k_gath, kv_lora_rank, dsa_topk,
                                                k_gath->nb[1], 0);                       // [512, topk]
            ggml_tensor * v_gath_t = ggml_cont(ctx0, ggml_transpose(ctx0, v_gath));       // [topk, 512]
            kqv_cache = ggml_mul_mat(ctx0, v_gath_t, kq_c);                              // [512, 1, H]
        } else {
            ggml_tensor * v_gath = ggml_view_3d(ctx0, k_gath, kv_lora_rank, dsa_topk, n_tokens,
                                                k_gath->nb[1], k_gath->nb[2], 0);          // [512, topk, T]
            ggml_tensor * v_gath_t = ggml_cont(ctx0, ggml_permute(ctx0, v_gath, 1, 0, 2, 3)); // [topk, 512, T]
            ggml_tensor * kq_c_gath = ggml_cont(ctx0, ggml_permute(ctx0, kq_c, 0, 2, 1, 3));  // [topk, H, T]
            kqv_cache = ggml_mul_mat(ctx0, v_gath_t, kq_c_gath);                           // [512, H, T]
            kqv_cache = ggml_cont(ctx0, ggml_permute(ctx0, kqv_cache, 0, 2, 1, 3));         // [512, T, H]
        }
        ggml_tensor * kqv = ggml_add(ctx0,
                ggml_mul_mat(ctx0, s_lat_t, kq_s),
                kqv_cache);                                                                // [512, T, H]
        ggml_tensor * wv_b3 = ggml_reshape_3d(ctx0, layer.wv_b, kv_lora_rank, n_embd_head_v, n_head);
        ggml_tensor * out_h = ggml_mul_mat(ctx0, wv_b3, kqv);                               // [128, T, H]
        ggml_tensor * merged = ggml_cont(ctx0, ggml_permute(ctx0, out_h, 0, 2, 1, 3));      // [128, H, T]
        cur = ggml_reshape_2d(ctx0, merged, n_embd_head_v * n_head, n_tokens);
    } else {
        const bool chunk_att = openpangu_att_score_should_chunk(n_kv_attn, NS, n_head, n_tokens,
                OPENPANGU_ATT_SCORE_CHUNK, OPENPANGU_ATT_FULL_KQ_MAX_MIB);
        const bool use_dsa_sel_idx_mask = sel_idx != nullptr && dsa_topk > 0;
        ggml_tensor * kqv = nullptr;
        if (chunk_att) {
            const bool can_prefill_gather =
                use_dsa_sel_idx_mask && !use_swa_window &&
                hparams.f_max_alibi_bias == 0.0f &&
                openpangu_dsa_gather_rows_fit_cuda(dsa_topk, 1);
            ggml_tensor * kl_full = nullptr;
            if (can_prefill_gather) {
                kl_full = ggml_view_2d(ctx0, kv_self.k_l[il], kv_lora_rank + n_embd_head_qk_rope, n_kv,
                                       kv_self.k_l[il]->nb[1], 0);
            }

            for (int64_t c0 = 0; c0 < n_tokens; c0 += OPENPANGU_ATT_SCORE_CHUNK) {
                const int64_t tc = std::min<int64_t>(OPENPANGU_ATT_SCORE_CHUNK, n_tokens - c0);
                ggml_tensor * q_all_c = ggml_view_3d(ctx0, q_all, kv_lora_rank + n_embd_head_qk_rope,
                                                     tc, n_head, q_all->nb[1], q_all->nb[2],
                                                     (size_t) c0*q_all->nb[1]);
                q_all_c = ggml_cont(ctx0, q_all_c);

                ggml_tensor * kq_sinks_c = ggml_mul_mat(ctx0, sink_blk, q_all_c);            // [NS, Tc, H]

                const bool prefill_gather_chunk =
                    can_prefill_gather &&
                    openpangu_dsa_prefill_gather_should_engage(n_kv, n_tokens, c0, tc, dsa_topk,
                                                               openpangu_kv_cache_pad(cparams));
                ggml_tensor * kqv_c = nullptr;
                if (prefill_gather_chunk) {
                    const int64_t gather_token_chunk = openpangu_dsa_gather_tokens_per_get_rows(dsa_topk);
                    for (int64_t g0 = 0; g0 < tc; g0 += gather_token_chunk) {
                        const int64_t tg = std::min<int64_t>(gather_token_chunk, tc - g0);
                        ggml_tensor * q_all_g = ggml_view_3d(ctx0, q_all_c,
                                                             kv_lora_rank + n_embd_head_qk_rope,
                                                             tg, n_head, q_all_c->nb[1], q_all_c->nb[2],
                                                             (size_t) g0*q_all_c->nb[1]);
                        q_all_g = ggml_cont(ctx0, q_all_g);
                        ggml_tensor * kq_sinks_g = ggml_view_3d(ctx0, kq_sinks_c, NS, tg, n_head,
                                                                kq_sinks_c->nb[1], kq_sinks_c->nb[2],
                                                                (size_t) g0*kq_sinks_c->nb[1]);
                        ggml_tensor * sel_idx_g = ggml_view_2d(ctx0, sel_idx, dsa_topk, tg,
                                                               sel_idx->nb[1], (size_t) (c0 + g0)*sel_idx->nb[1]);
                        ggml_tensor * sel_idx_flat_g = ggml_cont_2d(ctx0, sel_idx_g, dsa_topk*tg, 1);
                        ggml_tensor * k_gath_g = ggml_get_rows(ctx0, kl_full, sel_idx_flat_g);        // [576, topk*tg]
                        k_gath_g = openpangu_cast_gathered_latent_for_cache_type(ctx0, k_gath_g, kv_self.k_l[il]);
                        k_gath_g = ggml_reshape_3d(ctx0, k_gath_g, kv_lora_rank + n_embd_head_qk_rope,
                                                   dsa_topk, tg);                                    // [576, topk, tg]
                        ggml_tensor * q_gath_g = ggml_cont(ctx0, ggml_permute(ctx0, q_all_g, 0, 2, 1, 3)); // [576, H, tg]
                        ggml_tensor * kq_cache_g = ggml_mul_mat(ctx0, k_gath_g, q_gath_g);           // [topk, H, tg]
                        kq_cache_g = ggml_cont(ctx0, ggml_permute(ctx0, kq_cache_g, 0, 2, 1, 3));    // [topk, tg, H]
                        ggml_tensor * kq_g_all = ggml_concat(ctx0, kq_sinks_g, kq_cache_g, 0);       // [NS+topk, tg, H]
                        kq_g_all = ggml_soft_max_ext(ctx0, kq_g_all, nullptr, kq_scale, hparams.f_max_alibi_bias);

                        ggml_tensor * kq_s_g = ggml_view_3d(ctx0, kq_g_all, NS, tg, n_head,
                                                            kq_g_all->nb[1], kq_g_all->nb[2], 0);
                        ggml_tensor * kq_cache_soft_g = ggml_view_3d(ctx0, kq_g_all, dsa_topk, tg, n_head,
                                                                     kq_g_all->nb[1], kq_g_all->nb[2],
                                                                     NS*ggml_element_size(kq_g_all));
                        ggml_tensor * v_gath_g = ggml_view_3d(ctx0, k_gath_g, kv_lora_rank, dsa_topk, tg,
                                                              k_gath_g->nb[1], k_gath_g->nb[2], 0);
                        ggml_tensor * v_gath_t_g = ggml_cont(ctx0, ggml_permute(ctx0, v_gath_g, 1, 0, 2, 3)); // [topk, 512, tg]
                        ggml_tensor * kq_cache_gath_g = ggml_cont(ctx0, ggml_permute(ctx0, kq_cache_soft_g, 0, 2, 1, 3)); // [topk, H, tg]
                        ggml_tensor * kqv_cache_g = ggml_mul_mat(ctx0, v_gath_t_g, kq_cache_gath_g); // [512, H, tg]
                        kqv_cache_g = ggml_cont(ctx0, ggml_permute(ctx0, kqv_cache_g, 0, 2, 1, 3));  // [512, tg, H]
                        ggml_tensor * kqv_g = ggml_add(ctx0,
                                ggml_mul_mat(ctx0, s_lat_t, kq_s_g),
                                kqv_cache_g);                                                       // [512, tg, H]
                        kqv_c = kqv_c == nullptr ? kqv_g : ggml_concat(ctx0, kqv_c, kqv_g, 1);
                    }
                } else {
                    ggml_tensor * kq_cache_c = ggml_mul_mat(ctx0, kl_all, q_all_c);              // [n_kv_attn, Tc, H]
                    ggml_tensor * kq_c_all = ggml_concat(ctx0, kq_sinks_c, kq_cache_c, 0);       // [NS+n_kv_attn, Tc, H]

                    ggml_tensor * kq_mask_eff = ggml_view_2d(ctx0, KQ_mask, n_kv_attn, tc,
                                                             KQ_mask->nb[1], (size_t) c0*KQ_mask->nb[1]);
                    if (sel_mask) {
                        ggml_tensor * sel_mask_c = ggml_view_2d(ctx0, sel_mask, n_kv_attn, tc,
                                                                sel_mask->nb[1], (size_t) c0*sel_mask->nb[1]);
                        kq_mask_eff = ggml_add(ctx0, kq_mask_eff, sel_mask_c);
                    } else if (use_dsa_sel_idx_mask) {
                        ggml_tensor * sel_idx_c = ggml_view_2d(ctx0, sel_idx, dsa_topk, tc,
                                                               sel_idx->nb[1], (size_t) c0*sel_idx->nb[1]);
                        ggml_tensor * base_src_c = ggml_view_2d(ctx0, kq_cache_c, n_kv_attn, tc, kq_cache_c->nb[1], 0);
                        ggml_tensor * base_c = ggml_scale_bias(ctx0, base_src_c, 0.0f, -1e30f);
                        base_c = ggml_reshape_3d(ctx0, base_c, 1, n_kv_attn, tc);
                        ggml_tensor * zeros_src_c = ggml_view_2d(ctx0, kq_cache_c, dsa_topk, tc,
                                                                 dsa_topk*ggml_element_size(kq_cache_c), 0);
                        ggml_tensor * zeros_c = ggml_scale(ctx0, zeros_src_c, 0.0f);
                        zeros_c = ggml_reshape_3d(ctx0, zeros_c, 1, dsa_topk, tc);
                        ggml_tensor * sel_mask_c = ggml_set_rows(ctx0, base_c, zeros_c, sel_idx_c);
                        sel_mask_c = ggml_reshape_2d(ctx0, sel_mask_c, n_kv_attn, tc);
                        kq_mask_eff = ggml_add(ctx0, kq_mask_eff, sel_mask_c);
                    }
                    kq_mask_eff = ggml_cont(ctx0, kq_mask_eff);
                    ggml_tensor * s_mask0 = ggml_scale(ctx0,
                            ggml_view_2d(ctx0, kq_c_all, NS, tc, NS*ggml_element_size(kq_c_all), 0), 0.0f);
                    ggml_tensor * mask_all = ggml_concat(ctx0, s_mask0, kq_mask_eff, 0);         // [NS+n_kv_attn, Tc]
                    kq_c_all = ggml_soft_max_ext(ctx0, kq_c_all, mask_all, kq_scale, hparams.f_max_alibi_bias);

                    ggml_tensor * kq_s_c = ggml_view_3d(ctx0, kq_c_all, NS, tc, n_head,
                                                        kq_c_all->nb[1], kq_c_all->nb[2], 0);
                    ggml_tensor * kq_cache_soft_c = ggml_view_3d(ctx0, kq_c_all, n_kv_attn, tc, n_head,
                                                                 kq_c_all->nb[1], kq_c_all->nb[2],
                                                                 NS*ggml_element_size(kq_c_all));
                    ggml_tensor * kqv_cache_c = ggml_mul_mat(ctx0, get_vl_all(), kq_cache_soft_c); // [512, Tc, H]
                    kqv_c = ggml_add(ctx0,
                            ggml_mul_mat(ctx0, s_lat_t, kq_s_c),
                            kqv_cache_c);                                                       // [512, Tc, H]
                }
                kqv = kqv == nullptr ? kqv_c : ggml_concat(ctx0, kqv, kqv_c, 1);
            }
        } else {
            GGML_ASSERT(!use_dsa_sel_idx_mask || sel_mask != nullptr);
            ggml_tensor * kq_sinks = ggml_mul_mat(ctx0, sink_blk, q_all);                   // [NS, T, H]
            kq_cache = ggml_mul_mat(ctx0, kl_all, q_all);                                   // [n_kv_attn, T, H]
            ggml_tensor * kq = ggml_concat(ctx0, kq_sinks, kq_cache, 0);                    // [NS+n_kv_attn, T, H]

            // mask: sinks always visible (0) ++ the causal/SWA KQ_mask (+ the DSA top-k selection
            // mask on indexer layers). The zero block is built by scaling finite kq data (KQ_mask
            // itself holds -inf, which 0*x would turn into NaN).
            ggml_tensor * kq_mask_eff = ggml_view_2d(ctx0, KQ_mask, n_kv_attn, n_tokens, KQ_mask->nb[1], 0);
            if (sel_mask) {
                kq_mask_eff = ggml_add(ctx0, kq_mask_eff, sel_mask);
            }
            kq_mask_eff = ggml_cont(ctx0, kq_mask_eff);
            ggml_tensor * s_mask0 = ggml_scale(ctx0, ggml_view_2d(ctx0, kq, NS, n_tokens, NS*ggml_element_size(kq), 0), 0.0f);
            ggml_tensor * mask_all = ggml_concat(ctx0, s_mask0, kq_mask_eff, 0);            // [NS+n_kv, T]
            kq = ggml_soft_max_ext(ctx0, kq, mask_all, kq_scale, hparams.f_max_alibi_bias);

            ggml_tensor * kq_s = ggml_view_3d(ctx0, kq, NS, n_tokens, n_head, kq->nb[1], kq->nb[2], 0);
            ggml_tensor * kq_c = ggml_view_3d(ctx0, kq, n_kv_attn, n_tokens, n_head, kq->nb[1], kq->nb[2],
                                              NS*ggml_element_size(kq));
            ggml_tensor * kqv_cache = ggml_mul_mat(ctx0, get_vl_all(), kq_c);              // [512, T, H]
            kqv = ggml_add(ctx0,
                    ggml_mul_mat(ctx0, s_lat_t, kq_s),
                    kqv_cache);                                                           // [512, T, H]
        }

        ggml_tensor * wv_b3 = ggml_reshape_3d(ctx0, layer.wv_b, kv_lora_rank, n_embd_head_v, n_head);
        ggml_tensor * out_h = ggml_mul_mat(ctx0, wv_b3, kqv);                              // [128, T, H]
        ggml_tensor * merged = ggml_cont(ctx0, ggml_permute(ctx0, out_h, 0, 2, 1, 3));      // [128, H, T]
        cur = ggml_reshape_2d(ctx0, merged, n_embd_head_v * n_head, n_tokens);
    }

    // o_conv (MOME on the pre-o_proj attn output), then o_proj
    cur = openpangu_causal_conv(ctx0, gf, cur, layer.o_conv, conv_state, conv_off_o, seq_qnext, reset_conv_state);
    cur = llm_build_lora_mm(lctx, ctx0, layer.wo, cur);
    return cur;
}

// NextN/MTP head: eh_proj stitching -> one plain-residual Pangu block (sandwich norms,
// NO mHC, no block_post_norm) -> shared head. Mirrors OpenPanguV2MultiTokenPredictorLayer:
//   x = eh_proj(cat(enorm(embed(tok)), hnorm(prev_hidden)))
//   x = x + post_attn_ln(attn(input_ln(x)))
//   x = x + post_mlp_ln(moe(pre_mlp_ln(x)))
//   logits = shared_head.head(shared_head.norm(x))
// The MTP context allocates recurrent conv-state slots for the NextN layers too, so draft
// convs chain real t-1/t-2 taps across warmup and sequential draft steps.
ggml_tensor * llm_build_context::build_openpangu_mtp(
        const llama_layer & mtp_layer,
        ggml_tensor * prev_embeddings,
        ggml_cgraph * gf,
        int il,
        ggml_tensor * inp_pos,
        ggml_tensor * KQ_mask,
        ggml_tensor * inp_out_ids,
        ggml_tensor * inp_tokens,
        ggml_tensor * seq_qnext,
        ggml_tensor ** full_hidden_out,
        bool select_outputs,
        bool build_logits,
        bool cache_writes_only,
        bool KQ_mask_swa_windowed) {
    const float kq_scale = 1.0f / sqrtf(float(hparams.n_embd_head_k(0)));

    // same position-addressing invariant as build_openpangu (worst-case builds exempt)
    if (batch.pos && batch.n_tokens > 0) {
        GGML_ASSERT((llama_pos) kv_head == batch.pos[0] &&
                    "openPangu KV cache is position-addressed; kv head must equal the first batch position");
    }

    // the batch inputs (tokens, positions, masks, output selection, conv sequence ids) are
    // created ONCE by the caller and shared by every head built into the graph:
    // llama_set_inputs fills the tensors the lctx.inp_* pointers reference, so per-head
    // creation would leave every head but the last reading unwritten memory
    ggml_tensor * mtp_embd_weights = mtp_layer.nextn.embed_tokens
        ? mtp_layer.nextn.embed_tokens : model.tok_embd;
    ggml_tensor * token_emb = ggml_get_rows(ctx0, mtp_embd_weights, inp_tokens);
    cb(token_emb, "inp_embd", il);

    ggml_tensor * emb_norm = llm_build_norm(ctx0, token_emb,       hparams, mtp_layer.nextn.enorm, NULL, LLM_NORM_RMS, cb, il);
    ggml_tensor * hid_norm = llm_build_norm(ctx0, prev_embeddings, hparams, mtp_layer.nextn.hnorm, NULL, LLM_NORM_RMS, cb, il);

    // reference order: cat([inputs_embeds, previous_hidden_states], -1)
    ggml_tensor * combined = ggml_concat(ctx0, emb_norm, hid_norm, 0);
    ggml_tensor * cur = llm_build_lora_mm(lctx, ctx0, mtp_layer.nextn.eh_proj, combined);
    cb(cur, "mtp_eh_proj", il);

    // --- attention sublayer (plain residual) ---
    ggml_tensor * inpSA = cur;
    cur = llm_build_norm(ctx0, cur, hparams, mtp_layer.attn_norm, NULL, LLM_NORM_RMS, cb, il);
    // the MTP context allocates a recurrent conv-state slot for the NextN layers, so the
    // draft head chains real t-1/t-2 taps across warmup and sequential draft steps
    ggml_tensor * mtp_conv_state = (size_t) il < kv_self.s_l.size() ? kv_self.s_l[il] : nullptr;
    cur = build_openpangu_attention(gf, mtp_layer, il, cur, KQ_mask, inp_pos,
                                    mtp_conv_state, seq_qnext, kq_scale, KQ_mask_swa_windowed);
    if (cache_writes_only) {
        // only this head's latent-cache and conv-slot writes matter at this site (the
        // update chain's last head and the draft-time row fill); the FFN, norms, and
        // shared head would compute values nobody consumes
        cb(cur, "mtp_cache_write_anchor", il);
        return cur;
    }
    cur = llm_build_norm(ctx0, cur, hparams, mtp_layer.attn_post_norm, NULL, LLM_NORM_RMS, cb, il);
    ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
    cb(ffn_inp, "mtp_ffn_inp", il);

    const bool keep_full_hidden = full_hidden_out != nullptr;
    if (select_outputs && inp_out_ids && !keep_full_hidden) {
        ffn_inp = ggml_get_rows(ctx0, ffn_inp, inp_out_ids);
    }

    // --- ffn sublayer: MoE (routed + shared expert), plain residual ---
    cur = llm_build_norm(ctx0, cur = ffn_inp, hparams, mtp_layer.ffn_norm, NULL, LLM_NORM_RMS, cb, il);
    {
        ggml_tensor * moe_out = llm_build_moe_ffn(ctx0, lctx, cur,
                mtp_layer.ffn_gate_inp, mtp_layer.ffn_up_exps, mtp_layer.ffn_gate_exps, mtp_layer.ffn_down_exps,
                mtp_layer.ffn_exp_probs_b, n_expert, n_expert_used, LLM_FFN_SILU,
                hparams.expert_weights_norm, true, hparams.expert_weights_scale,
                (enum llm_expert_gating_func_type) hparams.expert_gating_func, cb, il, gf, false);
        ggml_tensor * shexp = llm_build_ffn(ctx0, lctx, nullptr, cur,
                mtp_layer.ffn_up_shexp, NULL, NULL, mtp_layer.ffn_gate_shexp, NULL, NULL,
                mtp_layer.ffn_down_shexp, NULL, NULL, NULL, LLM_FFN_SILU, LLM_FFN_PAR, cb, il);
        cur = ggml_add(ctx0, moe_out, shexp);
    }
    cur = llm_build_norm(ctx0, cur, hparams, mtp_layer.ffn_post_norm, NULL, LLM_NORM_RMS, cb, il);
    cur = ggml_add(ctx0, cur, ffn_inp);
    cb(cur, "mtp_out_resid", il);

    // --- shared head ---
    cur = llm_build_norm(ctx0, cur, hparams, mtp_layer.nextn.shared_head_norm, NULL, LLM_NORM_RMS, cb, -1);
    if (full_hidden_out) {
        *full_hidden_out = cur;
    }
    if (select_outputs && inp_out_ids && keep_full_hidden) {
        cur = ggml_get_rows(ctx0, cur, inp_out_ids);
    }
    if (!build_logits) {
        cb(cur, "mtp_hidden", il);
        return cur;
    }
    cb(cur, "result_norm", -1);
    ggml_tensor * head = mtp_layer.nextn.shared_head_head
        ? mtp_layer.nextn.shared_head_head : model.output;
    cur = llm_build_lora_mm(lctx, ctx0, head, cur);
    cb(cur, "result_output", -1);
    return cur;
}

ggml_cgraph * llm_build_context::build_openpangu() {
    ggml_cgraph * gf = new_graph_custom();
    openpangu_clear_cache_copies(lctx);

    // the indexer and latent stores are addressed by absolute position through kv_head;
    // enforce head == first batch position on real builds so any future cache
    // plumbing that breaks the append-only invariant fails here instead of corrupting
    // (worst-case measurement builds pass pos = null and are exempt)
    if (batch.pos && batch.n_tokens > 0) {
        GGML_ASSERT((llama_pos) kv_head == batch.pos[0] &&
                    "openPangu KV cache is position-addressed; kv head must equal the first batch position");
    }

    const int64_t n_embd_head_k = hparams.n_embd_head_k(0);                // 192
    const int64_t S             = hparams.mhc_num_stream;                  // 4
    const int    sink_iters     = (int) hparams.mhc_recur_norm;            // 20
    const float  kq_scale       = 1.0f / sqrtf(float(n_embd_head_k));


    // NextN/MTP graph (speculative decoding): draft generation selects the
    // requested head by depth; warmup/update chains all active heads so their conv slots and
    // latent caches hold real committed rows, exposing head 1 as the one-token shortcut.
    // Chaining convention (mirrors how head 1 consumes the target's shifted hidden rows):
    // head k+1's row at position p consumes head k's output row at position p-1; the p-1 of
    // the first batch row lives in the previous warmup/update batch and crosses decodes
    // through the inp_mtp_carry input / lctx.mtp_carry storage.
    if (cparams.mtp_op_type != MTP_OP_NONE) {
        GGML_ASSERT(model.mtp && hparams.nextn_predict_layers > 0 &&
                    "OpenPangu MTP graph requested without NextN layers loaded");
        GGML_ASSERT(batch.token && "openPangu MTP graphs decode token batches");

        ggml_tensor * hidden_states_from_main_model;
        if (cparams.mtp_op_type == MTP_OP_WARMUP || cparams.mtp_op_type == MTP_OP_UPDATE_ACCEPTED) {
            hidden_states_from_main_model = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hparams.n_embd, n_tokens);
        } else {
            hidden_states_from_main_model = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, hparams.n_embd);
        }
        ggml_set_name(hidden_states_from_main_model, "inp_mtp_states");
        ggml_set_input(hidden_states_from_main_model);
        lctx.inp_mtp_states = hidden_states_from_main_model;

        // shared batch inputs, created exactly once per graph (see build_openpangu_mtp)
        ggml_tensor * inp_pos = build_inp_pos();
        // the NextN/MTP layers are SWA layers with their own window (2048); the mask fill
        // uses hparams.n_swa_mtp when the graph is built with an MTP op type
        bool KQ_mask_swa_windowed = false;
        ggml_tensor * KQ_mask = hparams.n_swa_mtp > 0 && hparams.n_swa > 0
            ? openpangu_build_swa_mask_for_graph(*this, hparams.n_swa_mtp, &KQ_mask_swa_windowed)
            : build_inp_KQ_mask();
        ggml_tensor * inp_out_ids = n_tokens > 1 ? build_inp_out_ids() : nullptr;
        lctx.inp_tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, batch.n_tokens);
        cb(lctx.inp_tokens, "inp_tokens", -1);
        ggml_set_input(lctx.inp_tokens);
        ggml_tensor * inp_tokens = lctx.inp_tokens;
        lctx.inp_s_seq_qnext = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, 1, n_tokens);
        cb(lctx.inp_s_seq_qnext, "inp_s_seq_qnext", -1);
        ggml_set_input(lctx.inp_s_seq_qnext);
        ggml_tensor * seq_qnext = lctx.inp_s_seq_qnext;

        const int il_mtp_first = (int) (hparams.n_layer - hparams.nextn_predict_layers);
        const int n_mtp_heads_model = (int) hparams.nextn_predict_layers;
        const int n_mtp_heads = lctx.mtp_n_heads > 0
            ? std::max(1, std::min((int) lctx.mtp_n_heads, n_mtp_heads_model))
            : n_mtp_heads_model;
        const int step_idx = (int) std::min<int32_t>(std::max<int32_t>(0, lctx.mtp_step_idx), n_mtp_heads - 1);

        // carry input: head k's output at the last committed position (k = 1..n_heads_model-1),
        // fixed at model width so the storage layout is head-count independent
        ggml_tensor * inp_carry = nullptr;
        const bool is_cache_update = cparams.mtp_op_type == MTP_OP_WARMUP || cparams.mtp_op_type == MTP_OP_UPDATE_ACCEPTED;
        const bool needs_carry = n_mtp_heads_model > 1 &&
            ((is_cache_update && n_mtp_heads > 1) ||
             (cparams.mtp_op_type == MTP_OP_DRAFT_GEN && step_idx == 1 && n_mtp_heads > 2));
        if (needs_carry) {
            inp_carry = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hparams.n_embd, n_mtp_heads_model - 1);
            ggml_set_name(inp_carry, "inp_mtp_carry");
            ggml_set_input(inp_carry);
            lctx.inp_mtp_carry = inp_carry;
        }

        ggml_tensor * mtp_out = nullptr;
        if (cparams.mtp_op_type == MTP_OP_DRAFT_GEN) {
            const int il_mtp = il_mtp_first + step_idx;
            mtp_out = build_openpangu_mtp(model.layers[il_mtp],
                                          hidden_states_from_main_model, gf, il_mtp,
                                          inp_pos, KQ_mask, inp_out_ids, inp_tokens,
                                          seq_qnext,
                                          nullptr, true, true, false,
                                          KQ_mask_swa_windowed);
            // each draft step runs one head, so a deeper head has no cache row at this
            // position when its own decode comes later. Head h first decodes at step h+1 and
            // the update batches cover everything below the draft base, which with three
            // heads leaves exactly one gap: head 3's row at draft step 2. Pre-write it here
            // from the committed carry (head 2's output at the last committed position).
            if (step_idx == 1 && n_mtp_heads > 2) {
                ggml_tensor * fill_hidden = ggml_view_2d(ctx0, inp_carry, hparams.n_embd, 1,
                                                         inp_carry->nb[1], (size_t) 1 * inp_carry->nb[1]);
                ggml_tensor * fill = build_openpangu_mtp(model.layers[il_mtp_first + 2],
                                                         fill_hidden, gf, il_mtp_first + 2,
                                                         inp_pos, KQ_mask, nullptr, inp_tokens,
                                                         seq_qnext,
                                                         nullptr, false, false,
                                                         /*cache_writes_only=*/true,
                                                         KQ_mask_swa_windowed);
                ggml_build_forward_expand(gf, fill);
            }
        } else if (n_mtp_heads == 1) {
            mtp_out = build_openpangu_mtp(model.layers[il_mtp_first],
                                          hidden_states_from_main_model, gf, il_mtp_first,
                                          inp_pos, KQ_mask, inp_out_ids, inp_tokens,
                                          seq_qnext,
                                          nullptr, true, true, false,
                                          KQ_mask_swa_windowed);
        } else {
            ggml_tensor * prev_full = hidden_states_from_main_model;
            ggml_tensor * head1_hidden = nullptr;
            std::vector<ggml_tensor *> carry_out_rows;
            for (int i = 0; i < n_mtp_heads; ++i) {
                const int il_mtp = il_mtp_first + i;
                // the last head's block output feeds nothing (no carry, no logits) - only
                // its cache writes matter, so skip its FFN and norms
                const bool is_last_head = i + 1 == n_mtp_heads;
                ggml_tensor * full_hidden = nullptr;
                ggml_tensor * out = build_openpangu_mtp(model.layers[il_mtp],
                                                        prev_full, gf, il_mtp,
                                                        inp_pos, KQ_mask, inp_out_ids, inp_tokens,
                                                        seq_qnext,
                                                        is_last_head ? nullptr : &full_hidden,
                                                        i == 0,
                                                        false,
                                                        /*cache_writes_only=*/is_last_head && i > 0,
                                                        KQ_mask_swa_windowed);
                if (i == 0) {
                    head1_hidden = out;
                } else {
                    ggml_build_forward_expand(gf, out);
                }
                if (i + 1 < n_mtp_heads) {
                    // shift: head i+1's row p consumes this head's row p-1; row 0's
                    // predecessor comes from the carry
                    ggml_tensor * carry_col = ggml_view_2d(ctx0, inp_carry, hparams.n_embd, 1,
                                                           inp_carry->nb[1], (size_t) i * inp_carry->nb[1]);
                    if (n_tokens > 1) {
                        ggml_tensor * shifted = ggml_view_2d(ctx0, full_hidden, hparams.n_embd, n_tokens - 1,
                                                             full_hidden->nb[1], 0);
                        prev_full = ggml_concat(ctx0, carry_col, shifted, 1);
                    } else {
                        prev_full = carry_col;
                    }
                    carry_out_rows.push_back(ggml_view_2d(ctx0, full_hidden, hparams.n_embd, 1,
                                                          full_hidden->nb[1],
                                                          (size_t) (n_tokens - 1) * full_hidden->nb[1]));
                }
            }
            GGML_ASSERT(head1_hidden != nullptr);

            // committed carries for the next warmup/update and for draft-time row fills
            ggml_tensor * carry_out = carry_out_rows[0];
            for (size_t k = 1; k < carry_out_rows.size(); ++k) {
                carry_out = ggml_concat(ctx0, carry_out, carry_out_rows[k], 1);
            }
            carry_out = ggml_cont(ctx0, carry_out);
            ggml_set_name(carry_out, "mtp_carry_out");
            ggml_set_output(carry_out);
            ggml_build_forward_expand(gf, carry_out);

            const auto & head1_layer = model.layers[il_mtp_first];
            cb(head1_hidden, "result_norm", -1);
            ggml_tensor * head = head1_layer.nextn.shared_head_head
                ? head1_layer.nextn.shared_head_head : model.output;
            mtp_out = llm_build_lora_mm(lctx, ctx0, head, head1_hidden);
            cb(mtp_out, "result_output", -1);
        }

        ggml_build_forward_expand(gf, mtp_out);
        return gf;
    }

    ggml_tensor * cur;
    ggml_tensor * inpL = llm_build_inp_embd(ctx0, lctx, hparams, batch, model.tok_embd, cb);
    ggml_tensor * inp_pos = build_inp_pos();
    ggml_tensor * KQ_mask = build_inp_KQ_mask();
    // SWA layers get the windowed mask (window 512 base); DSA layers keep the plain causal
    // mask and add the indexer's top-k selection inside the attention builder. Absent
    // schedule keys (n_swa == 0) keep every layer dense (pre-DSA GGUF fallback).
    bool KQ_mask_swa_windowed = false;
    ggml_tensor * KQ_mask_swa = hparams.n_swa > 0 ? openpangu_build_swa_mask_for_graph(*this, hparams.n_swa, &KQ_mask_swa_windowed) : nullptr;
    lctx.inp_s_seq_qnext = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, 1, n_tokens);
    cb(lctx.inp_s_seq_qnext, "inp_s_seq_qnext", -1);
    ggml_set_input(lctx.inp_s_seq_qnext);
    ggml_tensor * seq_qnext = lctx.inp_s_seq_qnext;

    // mHC entry: repeat the embedding into S residual streams -> R [n_embd, S, n_tokens]
    ggml_tensor * R = ggml_repeat(ctx0, ggml_reshape_3d(ctx0, inpL, n_embd, 1, n_tokens),
                                  ggml_new_tensor_3d(ctx0, inpL->type, n_embd, S, n_tokens));
    ggml_set_name(R, "opg_R_entry");

    // mHC pre: combine S streams -> x [n_embd, n_tokens]; also returns h_post [S,T], h_res_mix [S,S,T]
    auto mhc_pre = [&](ggml_tensor * Rin, ggml_tensor * phi, ggml_tensor * alpha,
                       ggml_tensor * beta, ggml_tensor * gamma,
                       ggml_tensor ** h_post_out, ggml_tensor ** h_res_out) {
        ggml_tensor * flat = ggml_reshape_2d(ctx0, ggml_cont(ctx0, Rin), n_embd * S, n_tokens);
        ggml_tensor * normed = ggml_rms_norm(ctx0, flat, hparams.f_norm_rms_eps);
        normed = ggml_mul(ctx0, normed, gamma);                       // [S*H, T]
        ggml_tensor * mixes = ggml_mul_mat(ctx0, phi, normed);        // [(S+2)*S, T]
        ggml_tensor * h_pre  = ggml_view_2d(ctx0, mixes, S, n_tokens, mixes->nb[1], 0);
        ggml_tensor * h_post = ggml_view_2d(ctx0, mixes, S, n_tokens, mixes->nb[1], S*ggml_element_size(mixes));
        ggml_tensor * h_res  = ggml_view_2d(ctx0, mixes, S*S, n_tokens, mixes->nb[1], 2*S*ggml_element_size(mixes));

        // alpha = [a_pre, a_post, a_res]; beta = [b_pre(S), b_post(S), b_res(S*S)]
        ggml_tensor * a_pre  = ggml_view_1d(ctx0, alpha, 1, 0);
        ggml_tensor * b_pre  = ggml_view_1d(ctx0, beta, S, 0);
        // cont is required: the CUDA broadcast-mul path misreads strided views (h_pre is a
        // row-slice of mixes), while CPU handles the strides — token 0 right, tokens 1+ garbage
        h_pre = ggml_add(ctx0, ggml_mul(ctx0, ggml_cont(ctx0, h_pre), a_pre), b_pre);  // broadcast scalar + [S]
        h_pre = ggml_sigmoid(ctx0, h_pre);                            // [S,T] (+eps omitted, inert)

        // combine: x[h,t] = sum_s h_pre[s,t] * R[h,s,t]
        ggml_tensor * hpre3 = ggml_reshape_3d(ctx0, ggml_cont(ctx0, h_pre), 1, S, n_tokens);
        ggml_tensor * weighted = ggml_mul(ctx0, Rin, hpre3);          // [H,S,T]
        ggml_tensor * wperm = ggml_cont(ctx0, ggml_permute(ctx0, weighted, 1, 0, 2, 3)); // [S,H,T]
        ggml_tensor * x = ggml_reshape_2d(ctx0, ggml_sum_rows(ctx0, wperm), n_embd, n_tokens); // sum over S

        *h_post_out = ggml_cont(ctx0, h_post);
        *h_res_out  = ggml_cont(ctx0, h_res);
        return x;
    };

    // mHC post: R_new[h,s,t] = h_post[s,t]*y[h,t] + sum_j m[s,j,t]*R[h,j,t]
    auto mhc_post = [&](ggml_tensor * y, ggml_tensor * h_post, ggml_tensor * Rin,
                        ggml_tensor * alpha, ggml_tensor * beta, ggml_tensor * h_res) {
        ggml_tensor * a_post = ggml_view_1d(ctx0, alpha, 1, 1*ggml_element_size(alpha));
        ggml_tensor * a_res  = ggml_view_1d(ctx0, alpha, 1, 2*ggml_element_size(alpha));
        ggml_tensor * b_post = ggml_view_1d(ctx0, beta, S,   S*ggml_element_size(beta));
        ggml_tensor * b_res  = ggml_view_1d(ctx0, beta, S*S, 2*S*ggml_element_size(beta));

        h_post = ggml_add(ctx0, ggml_mul(ctx0, h_post, a_post), b_post);
        h_post = ggml_scale(ctx0, ggml_sigmoid(ctx0, h_post), 2.0f);  // 2*sigmoid, [S,T]

        ggml_tensor * m = ggml_add(ctx0, ggml_mul(ctx0, h_res, a_res), b_res); // [S*S,T]
        m = ggml_sinkhorn(ctx0, m, (int) S, sink_iters, 0.0f, /*output_transposed=*/true); // [row S, col S, T]

        // term1: h_post[s,t]*y[h,t] -> [H,S,T]
        ggml_tensor * y3 = ggml_reshape_3d(ctx0, y, n_embd, 1, n_tokens);
        ggml_tensor * hpost3 = ggml_reshape_3d(ctx0, ggml_cont(ctx0, h_post), 1, S, n_tokens);
        ggml_tensor * term1 = ggml_mul(ctx0, ggml_repeat(ctx0, y3,
                                 ggml_new_tensor_3d(ctx0, y->type, n_embd, S, n_tokens)), hpost3);

        // term2: sum_j m[s,j,t]*R[h,j,t]. For each out-stream s, weight over input streams j.
        // Build via: for stream axis, matmul R[H, j, t] with m[j, s, t] batched over t.
        // R_perm [j(S), H, T] ; m as [j(S), s(S), T]; batched mul_mat over T -> [H? ] messy.
        // Simpler explicit loop over S output streams (S=4, cheap):
        ggml_tensor * term2 = nullptr;
        for (int64_t s = 0; s < S; ++s) {
            // m_s = m[:, s, :] -> weights over input streams j: [S, T]
            ggml_tensor * m_s = ggml_cont(ctx0, ggml_view_2d(ctx0, m, S, n_tokens, m->nb[2], s*m->nb[1]));
            ggml_tensor * m_s3 = ggml_reshape_3d(ctx0, m_s, 1, S, n_tokens);      // [1,S,T]
            ggml_tensor * acc = ggml_mul(ctx0, Rin, m_s3);                        // [H,S,T]
            ggml_tensor * accp = ggml_cont(ctx0, ggml_permute(ctx0, acc, 1, 0, 2, 3)); // [S,H,T]
            ggml_tensor * summed = ggml_reshape_2d(ctx0, ggml_sum_rows(ctx0, accp), n_embd, n_tokens); // [H,T]
            summed = ggml_reshape_3d(ctx0, summed, n_embd, 1, n_tokens);
            term2 = term2 ? ggml_concat(ctx0, term2, summed, 1) : summed;         // -> [H,S,T]
        }
        return ggml_add(ctx0, term1, term2); // [H,S,T]
    };

    // Base generation uses only the transformer layers; the trailing NextN/MTP layers are skipped.
    const int n_layer_base = n_layer - (int) hparams.nextn_predict_layers;
    for (int il = 0; il < n_layer_base; ++il) {
        auto & layer = model.layers[il];
        ggml_tensor * conv_state = (size_t) il < kv_self.s_l.size() ? kv_self.s_l[il] : nullptr;

        // ================= attention sublayer =================
        ggml_tensor * h_post_a, * h_res_a;
        ggml_tensor * x = mhc_pre(R, layer.mhc_attn_phi, layer.mhc_attn_alpha,
                                  layer.mhc_attn_beta, layer.mhc_attn_gamma, &h_post_a, &h_res_a);
        if (il == 0) ggml_set_name(x, "opg0_attn_mhcpre_x");
        cur = llm_build_norm(ctx0, x, hparams, layer.attn_norm, NULL, LLM_NORM_RMS, cb, il);
        if (il == 0) ggml_set_name(cur, "opg0_attn_norm");

        const bool layer_swa = KQ_mask_swa && hparams.openpangu_window[il] > 0;
        ggml_tensor * layer_mask = layer_swa ? KQ_mask_swa : KQ_mask;
        cur = build_openpangu_attention(gf, layer, il, cur, layer_mask, inp_pos,
                                        conv_state, seq_qnext, kq_scale,
                                        layer_swa && KQ_mask_swa_windowed);
        if (il == 0) ggml_set_name(cur, "opg0_attn_out");

        cur = llm_build_norm(ctx0, cur, hparams, layer.attn_post_norm, NULL, LLM_NORM_RMS, cb, il);
        if (il == 0) ggml_set_name(cur, "opg0_attn_postnorm");

        // mHC post -> scatter back to S streams
        R = mhc_post(cur, h_post_a, R, layer.mhc_attn_alpha, layer.mhc_attn_beta, h_res_a);
        if (il == 0) ggml_set_name(R, "opg0_R_attn");

        // ================= ffn sublayer =================
        ggml_tensor * h_post_m, * h_res_m;
        ggml_tensor * xm = mhc_pre(R, layer.mhc_mlp_phi, layer.mhc_mlp_alpha,
                                   layer.mhc_mlp_beta, layer.mhc_mlp_gamma, &h_post_m, &h_res_m);
        cur = llm_build_norm(ctx0, xm, hparams, layer.ffn_norm, NULL, LLM_NORM_RMS, cb, il);

        if ((uint32_t) il < hparams.n_layer_dense_lead) {
            cur = llm_build_ffn(ctx0, lctx, nullptr, cur,
                    layer.ffn_up, NULL, NULL, layer.ffn_gate, NULL, NULL, layer.ffn_down, NULL, NULL,
                    NULL, LLM_FFN_SILU, LLM_FFN_PAR, cb, il);
        } else {
            ggml_tensor * moe_out = llm_build_moe_ffn(ctx0, lctx, cur,
                    layer.ffn_gate_inp, layer.ffn_up_exps, layer.ffn_gate_exps, layer.ffn_down_exps,
                    layer.ffn_exp_probs_b, n_expert, n_expert_used, LLM_FFN_SILU,
                    hparams.expert_weights_norm, true, hparams.expert_weights_scale,
                    (enum llm_expert_gating_func_type) hparams.expert_gating_func, cb, il, gf, false);
            ggml_tensor * shexp = llm_build_ffn(ctx0, lctx, nullptr, cur,
                    layer.ffn_up_shexp, NULL, NULL, layer.ffn_gate_shexp, NULL, NULL,
                    layer.ffn_down_shexp, NULL, NULL, NULL, LLM_FFN_SILU, LLM_FFN_PAR, cb, il);
            cur = ggml_add(ctx0, moe_out, shexp);
        }
        cur = llm_build_norm(ctx0, cur, hparams, layer.ffn_post_norm, NULL, LLM_NORM_RMS, cb, il);
        if (il == 0) ggml_set_name(cur, "opg0_ffn_postnorm");

        R = mhc_post(cur, h_post_m, R, layer.mhc_mlp_alpha, layer.mhc_mlp_beta, h_res_m);

        // block post-norm on the layer subset (RMSNorm over the concatenated S*H)
        if (layer.block_post_norm) {
            ggml_tensor * flat = ggml_reshape_2d(ctx0, ggml_cont(ctx0, R), n_embd * S, n_tokens);
            flat = ggml_rms_norm(ctx0, flat, hparams.f_norm_rms_eps);
            flat = ggml_mul(ctx0, flat, layer.block_post_norm);
            R = ggml_reshape_3d(ctx0, flat, n_embd, S, n_tokens);
        }
        if (il == 0) ggml_set_name(R, "opg0_R_block");
        R = lctx.cvec.apply_to(ctx0, R, il);
    }

    // mHC tail merge: collapse S streams -> 1 (pre_only)
    {
        ggml_tensor * flat = ggml_reshape_2d(ctx0, ggml_cont(ctx0, R), n_embd * S, n_tokens);
        ggml_tensor * normed = ggml_mul(ctx0, ggml_rms_norm(ctx0, flat, hparams.f_norm_rms_eps), model.mhc_merge_gamma);
        ggml_tensor * w = ggml_mul_mat(ctx0, model.mhc_merge_phi, normed);          // [S, T]
        ggml_tensor * a_pre = ggml_view_1d(ctx0, model.mhc_merge_alpha, 1, 0);
        w = ggml_sigmoid(ctx0, ggml_add(ctx0, ggml_mul(ctx0, w, a_pre), model.mhc_merge_beta)); // [S,T]
        ggml_tensor * w3 = ggml_reshape_3d(ctx0, ggml_cont(ctx0, w), 1, S, n_tokens);
        ggml_tensor * weighted = ggml_mul(ctx0, R, w3);                            // [H,S,T]
        ggml_tensor * wperm = ggml_cont(ctx0, ggml_permute(ctx0, weighted, 1, 0, 2, 3)); // [S,H,T]
        cur = ggml_reshape_2d(ctx0, ggml_sum_rows(ctx0, wperm), n_embd, n_tokens);
    }

    // select only the output tokens (the framework binds n_outputs rows, not all n_tokens).
    // With MTP enabled, keep every token: the speculative framework consumes per-token
    // hidden states (result_norm via pooling) to warm up / feed the NextN head.
    if (!cparams.mtp) {
        ggml_tensor * inp_out_ids = build_inp_out_ids();
        cur = ggml_get_rows(ctx0, cur, inp_out_ids);
    }

    cur = llm_build_norm(ctx0, cur, hparams, model.output_norm, NULL, LLM_NORM_RMS, cb, -1);
    cb(cur, "result_norm", -1);
    cur = llm_build_lora_mm(lctx, ctx0, model.output, cur);
    cb(cur, "result_output", -1);
    ggml_build_forward_expand(gf, cur);
    return gf;
}

#include "../llama-build-context.h"
#include "../llama-model.h"
#include "../llama-context.h"
#include "openpangu-op-policy.h"

#include <algorithm>
#include <climits>
#include <vector>

static constexpr int OPENPANGU_CACHE_WRITES_PER_LAYER = 1;
static constexpr int OPENPANGU_WRITE_K_CKV = 0;
static constexpr int64_t OPENPANGU_DSA_GATHER_MIN_RATIO = 2;
// Keep these fixed constants and the loop-node accounting in sync with
// llama_openpangu_chunked_graph_nodes() in llama.cpp. A full-span fused dense/SWA/MTP
// call contributes raw-cache view (1) + mask view (1) + latent-attn op (1) = 3 graph
// nodes (a materialized legacy DSA mask adds one mask-add node), none repeated per
// token chunk. For C chunks the explicit path is 16*C + (C - 1) result concats +
// at most 7 shared cache nodes = 17*C + 6 <= 32*C; deferred DSA adds 10*C, so
// 27*C + 6 <= 48*C. For I indexer chunks, 16*I common + up to 2*I concats + 4*I
// mask ops = 22*I <= 24*I. For G legacy gather subchunks, 3 outer + 26*G core +
// up to G casts + (G - 1) inner concats = 28*G + 2, plus at most 2 shared-view/
// outer-concat nodes = 28*G + 4 <= 32 + 32*G. When all C outer chunks gather,
// mode-1 emits 2*C query view/cont nodes + 2*C index view/cont nodes + C indexed
// ops + (C - 1) output concats + 1 shared raw-cache view = 6*C. The opt-out still
// emits the legacy chain, so its larger allowance remains an upper bound for both.
// Tensor-storage leaves are excluded from gf->n_nodes. The estimator keeps an
// additional 64 + 5% margin.
static constexpr int64_t OPENPANGU_IDX_SCORE_CHUNK = 256;
static constexpr int64_t OPENPANGU_ATT_SCORE_CHUNK = 256;
static constexpr int64_t OPENPANGU_ATT_FULL_KQ_MAX_MIB = 1024;
static constexpr int64_t OPENPANGU_CUDA_GET_ROWS_GRID_Y_MAX = 65535;

// Production default: the fused/specialized OpenPangu ops are on and capability-gated at
// each call site. The single documented compile-time switch GGML_OPENPANGU_LEGACY_OPS
// (build with -DGGML_OPENPANGU_LEGACY_OPS=1) forces every legacy chain at once (the A/B
// baseline), so the selection is fixed for the whole binary and graph reuse cannot switch
// implementations.
static bool openpangu_fused_ops_enabled() {
    return !openpangu_legacy_ops_forced();
}

static bool openpangu_rope_offset_enabled() {
    return openpangu_fused_ops_enabled();
}

static bool openpangu_batched_mix_enabled() {
    return openpangu_fused_ops_enabled();
}

static bool openpangu_backend_supports_batched_mix(
        llama_context & lctx, ggml_tensor * placement, ggml_tensor * candidate) {
    if (!openpangu_batched_mix_enabled() || placement == nullptr || candidate == nullptr) {
        return false;
    }

    // alpha is a stable per-layer model weight and anchors the scheduler backend
    // for the entire mHC post block. Never insert a CUDA-unsupported candidate
    // between CUDA-resident neighbors and create a silent CPU island.
    ggml_backend_t backend = ggml_backend_sched_get_tensor_backend(lctx.sched, placement);
    return backend != nullptr && ggml_backend_supports_op(backend, candidate);
}

static bool openpangu_backend_supports_rope_offset(
        llama_context & lctx, ggml_tensor * placement, ggml_tensor * rope_offset) {
    if (!openpangu_rope_offset_enabled() || placement == nullptr || rope_offset == nullptr) {
        return false;
    }

    // The projection weight anchors the scheduler placement of the indexer path. Do not insert an
    // accelerator-unsupported op between accelerator-supported neighbors and force a CPU island.
    ggml_backend_t backend = ggml_backend_sched_get_tensor_backend(lctx.sched, placement);
    return backend != nullptr && ggml_backend_supports_op(backend, rope_offset);
}

static std::vector<llama_context::CacheWrite> & openpangu_cache_writes(llama_context & lctx) {
    return lctx.cparams.mtp_op_type == MTP_OP_NONE ? lctx.openpangu_cache_writes : lctx.openpangu_cache_writes_mtp;
}

static bool openpangu_indexer_topk_auto_enabled() {
    return openpangu_fused_ops_enabled();
}

static bool openpangu_use_indexer_topk(
        llama_context & lctx,
        ggml_tensor * placement,
        ggml_tensor * candidate) {
    // Legacy-forced bring-up mode dominates even an explicit -fidx request, so the
    // load-time "legacy-forced" report stays truthful. Automatic mode already respects
    // it through openpangu_indexer_topk_auto_enabled(); this closes the explicit path.
    if (openpangu_legacy_ops_forced()) {
        return false;
    }
    ggml_backend_t backend = placement == nullptr ? nullptr :
        ggml_backend_sched_get_tensor_backend(lctx.sched, placement);
    // Split buffer types do not resolve to one scheduled backend here. That is
    // intentionally treated like any unknown placement: automatic mode keeps
    // the legacy graph, while explicit -fidx retains force-on semantics.
    return openpangu_should_use_indexer_topk(
        lctx.cparams.fused_idx_topk,
        openpangu_indexer_topk_auto_enabled(),
        backend,
        candidate);
}

static bool openpangu_consider_indexer_topk(llama_context & lctx, ggml_tensor * placement) {
    if (openpangu_legacy_ops_forced()) {
        return false;
    }
    if (lctx.cparams.fused_idx_topk) {
        return true;
    }
    if (!openpangu_indexer_topk_auto_enabled() || placement == nullptr) {
        return false;
    }
    // A split buffer type has no single scheduled backend anchor. Preserve the
    // legacy graph exactly instead of constructing an unused fused candidate.
    return ggml_backend_sched_get_tensor_backend(lctx.sched, placement) != nullptr;
}

static void openpangu_clear_cache_writes(llama_context & lctx) {
    auto & writes = openpangu_cache_writes(lctx);
    std::fill(writes.begin(), writes.end(), llama_context::CacheWrite{});
    std::fill(lctx.dsa_cache_copies.begin(), lctx.dsa_cache_copies.end(), llama_context::CacheCopy{});
}

static void openpangu_register_cache_write(
        llama_context & lctx, int il, int slot,
        ggml_tensor * node, ggml_tensor * root, size_t step,
        llama_openpangu_cache_write_kind kind) {
    GGML_ASSERT(slot >= 0 && slot < OPENPANGU_CACHE_WRITES_PER_LAYER);
    auto & writes = openpangu_cache_writes(lctx);
    const size_t idx = (size_t) OPENPANGU_CACHE_WRITES_PER_LAYER*il + slot;
    GGML_ASSERT(idx < writes.size());
    writes[idx].node = node;
    writes[idx].root = root;
    writes[idx].step = step;
    writes[idx].kind = kind;
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

// Raw latent-cache view (native type: f32/f16/q8_0) for the fused latent-attention op,
// which dequantizes internally -- no F32 cast, no value transpose.
static ggml_tensor * openpangu_build_k_latent_raw(
        ggml_context * ctx, const llama_kv_cache & kv_self, int il,
        int64_t n_kv_view, int64_t win_off) {
    ggml_tensor * kl = kv_self.k_l[il];
    return ggml_view_2d(ctx, kl, kl->ne[0], n_kv_view, kl->nb[1], (size_t) win_off*kl->nb[1]);
}

// Whether to route dense/SWA spans or indexed DSA gathers through latent-attention ops.
// Capability-gated at the call site (openpangu_backend_supports_fused_attn); a
// GGML_OPENPANGU_LEGACY_OPS build forces the explicit chain. The v1 op has no
// ALiBi, so a configured max-bias also forces the explicit chain.
static bool openpangu_fused_attn_enabled(const llama_hparams & hparams) {
    return openpangu_fused_ops_enabled() && hparams.f_max_alibi_bias == 0.0f;
}

// Latent attention is implemented on CPU and CUDA only. Confirm the scheduled backend for
// this layer's attention output projection can execute the fused op before adopting it, so
// an unsupported backend (or a CUDA layout the op rejects) keeps the exact legacy chain
// instead of forcing a scheduler CPU island. The projection weight anchors the scheduled
// placement, mirroring the batched-mix and rope-offset gates.
static bool openpangu_backend_supports_fused_attn(
        llama_context & lctx, const llama_hparams & hparams,
        ggml_tensor * placement, ggml_tensor * candidate) {
    if (!openpangu_fused_attn_enabled(hparams) || placement == nullptr || candidate == nullptr) {
        return false;
    }
    ggml_backend_t backend = ggml_backend_sched_get_tensor_backend(lctx.sched, placement);
    return backend != nullptr && ggml_backend_supports_op(backend, candidate);
}

// Fused RMSNorm+add is used at the MTP post-norm sites. ggml_fused_rms_norm_add already
// decomposes non-canonical weight layouts internally, so the only remaining risk is a
// scheduled backend that cannot execute the produced FUSED_RMS_NORM_ADD op. Confirm support
// against the norm weight's placement before adopting the candidate; otherwise the caller
// builds the explicit decomposed norm+add. A candidate the constructor already decomposed is
// universally supported and reduces to that same legacy chain.
static bool openpangu_backend_supports_fused_rms_norm_add(
        llama_context & lctx, ggml_tensor * placement, ggml_tensor * candidate) {
    if (!openpangu_fused_ops_enabled() || placement == nullptr || candidate == nullptr) {
        return false;
    }
    ggml_backend_t backend = ggml_backend_sched_get_tensor_backend(lctx.sched, placement);
    return backend != nullptr && ggml_backend_supports_op(backend, candidate);
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
// Pangu-specific pieces implemented here:
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

static void openpangu_build_latent_cache_write(
        llm_build_context & llm, ggml_cgraph * gf, int il,
        ggml_tensor * ckv, ggml_tensor * k_pe,
        int64_t kv_lora_rank, int64_t n_embd_head_qk_rope) {
    ggml_context * ctx = llm.ctx0;
    ggml_tensor * kl = llm.kv_self.k_l[il];
    ggml_tensor * cache_write = nullptr;
    llama_openpangu_cache_write_kind write_kind = llama_openpangu_cache_write_kind::legacy_cpy;

    // The opt-out is authoritative: do not even construct a candidate in that process.
    // Otherwise validate the exact proposed node, then anchor the placement decision on the
    // persistent cache root. An unsupported accelerator keeps the established chain instead
    // of letting the scheduler create a CPU island around the cache mutation.
    if (llama_openpangu_pack_cache_rows_enabled()) {
        const size_t dst_offset = llm.kv_head*kl->nb[1];
        // Precheck the operand contract so a layout the pack op cannot represent routes to the
        // legacy concat+cpy write below instead of aborting in the constructor.
        ggml_tensor * candidate = nullptr;
        bool backend_present = false;
        bool backend_supports_pack = false;
        if (ggml_pack_cache_rows_valid(ckv, k_pe, kl, dst_offset)) {
            candidate = ggml_pack_cache_rows(ctx, ckv, k_pe, kl, dst_offset);
            GGML_ASSERT(llama_openpangu_pack_candidate_matches(
                    candidate, ckv, k_pe, kl, dst_offset));
            ggml_backend_t backend = ggml_backend_sched_get_tensor_backend(llm.lctx.sched, kl);
            backend_present = backend != nullptr;
            backend_supports_pack = backend_present && ggml_backend_supports_op(backend, candidate);
        }
        write_kind = llama_openpangu_select_cache_write(
                true, true, backend_present, backend_supports_pack);
        if (write_kind == llama_openpangu_cache_write_kind::pack_cache_rows) {
            cache_write = candidate;
        }
    }

    if (cache_write == nullptr) {
        ggml_tensor * ckv_store = ckv;
        ggml_tensor * kpe_store = k_pe;
        if (ggml_is_quantized(kl->type)) {
            ckv_store = openpangu_cast_for_latent_cache_write(ctx, ckv_store, kl);
            kpe_store = openpangu_cast_for_latent_cache_write(ctx, kpe_store, kl);
        }
        ggml_tensor * k_latent = ggml_concat(ctx, ckv_store, kpe_store, 0);
        ggml_tensor * kl_full = ggml_view_2d(ctx, kl, kv_lora_rank + n_embd_head_qk_rope,
                                             llm.n_tokens, kl->nb[1], llm.kv_head*kl->nb[1]);
        cache_write = ggml_cpy(ctx, k_latent, kl_full);
    }
    GGML_ASSERT(write_kind == llama_openpangu_cache_write_kind::pack_cache_rows ||
                write_kind == llama_openpangu_cache_write_kind::legacy_cpy);
    openpangu_register_cache_write(
            llm.lctx, il, OPENPANGU_WRITE_K_CKV, cache_write, kl, kl->nb[1], write_kind);
    ggml_build_forward_expand(gf, cache_write);
}

struct openpangu_dsa_selection {
    ggml_tensor * mask = nullptr;
    ggml_tensor * indices = nullptr;
    bool gather_engaged = false;
    int64_t topk = 0;
};

static openpangu_dsa_selection openpangu_build_dsa_selection(
        llm_build_context & llm, ggml_cgraph * gf, const llama_layer & layer, int il,
        ggml_tensor * x_normed, ggml_tensor * q_lora,
        ggml_tensor * KQ_mask, ggml_tensor * inp_pos) {
    // Reference (Infer _pangu_torch_calib): q_idx = wq_b(q_lora_normed) [24 heads x 128],
    // k_idx = rms(k_norm)(wk(x_normed)) [128, shared across heads], both NEOX-roped on the
    // first n_rot channels (opposite order from the main head's [nope|rope] split).
    // score[t,s] = sum_g weights_proj(x)[t,g] * relu(q_idx[t,g] dot k_idx[s]), then causal
    // masking and top-k. Selection is inert while the causal window is no larger than top-k.
    openpangu_dsa_selection selection;
    ggml_tensor * idx_cache = (size_t) il < llm.kv_self.kr_l.size() ? llm.kv_self.kr_l[il] : nullptr;
    if (idx_cache == nullptr || layer.indexer_attn_q_b == nullptr) {
        return selection;
    }

    ggml_context * ctx = llm.ctx0;
    const int64_t n_ihead = llm.hparams.indexer_n_head;    // 24
    const int64_t d_idx   = llm.hparams.indexer_head_size; // 128
    const int64_t n_embd_head_qk_rope = llm.hparams.n_rot; // 64
    int64_t topk = llm.hparams.indexer_top_k;              // 2048
    if (llm.lctx.cparams.dsa_top_k > 0) {
        topk = llm.lctx.cparams.dsa_top_k;
    }
    GGML_ASSERT(topk > 0 && topk <= INT_MAX);
    selection.topk = topk;

    const uint32_t pad = openpangu_kv_cache_pad(llm.cparams);
    const bool is_base_graph = llm.cparams.mtp_op_type == MTP_OP_NONE;
    const bool dsa_gather_predicate =
        openpangu_dsa_gather_should_engage(llm.n_kv, llm.n_tokens, topk, pad);
    const bool dsa_gather_allowed =
        is_base_graph && dsa_gather_predicate &&
        openpangu_dsa_gather_rows_fit_cuda(topk, llm.n_tokens);

    // Indexer keys for this batch -> position-indexed cache (write-before-read holds by
    // graph order, same as the kv store; committed columns never change -> rollback-safe).
    ggml_tensor * k_idx = ggml_mul_mat(ctx, layer.indexer_attn_k, x_normed);   // [d_idx, T]
    k_idx = llm.llm_build_norm(ctx, k_idx, llm.hparams, layer.indexer_k_norm, layer.indexer_k_norm_b,
                              layer.indexer_k_norm_b ? LLM_NORM : LLM_NORM_RMS, llm.cb, il);
    ggml_tensor * k_idx_offset = nullptr;
    if (openpangu_rope_offset_enabled()) {
        ggml_tensor * k_idx_3d = ggml_reshape_3d(ctx, k_idx, d_idx, 1, llm.n_tokens);
        // Precheck the operand contract so an unsupported rope layout leaves the offset op
        // unbuilt and the gate below falls back to the legacy split-rope chain.
        if (ggml_rope_ext_offset_valid(k_idx_3d, inp_pos, nullptr, llm.n_rot, 0, llm.rope_type)) {
            k_idx_offset = ggml_rope_ext_offset(ctx, k_idx_3d, inp_pos, nullptr, llm.n_rot, 0, llm.rope_type,
                                               llm.n_ctx_orig, llm.freq_base, llm.freq_scale,
                                               llm.ext_factor, llm.attn_factor, llm.beta_fast, llm.beta_slow);
        }
    }
    if (openpangu_backend_supports_rope_offset(llm.lctx, layer.indexer_attn_k, k_idx_offset)) {
        k_idx = ggml_reshape_2d(ctx, k_idx_offset, d_idx, llm.n_tokens);
    } else {
        ggml_tensor * k_idx_rope = ggml_view_2d(ctx, k_idx, n_embd_head_qk_rope,
                                                llm.n_tokens, k_idx->nb[1], 0);
        ggml_tensor * k_idx_pass = ggml_view_2d(ctx, k_idx, d_idx - n_embd_head_qk_rope,
                                                llm.n_tokens, k_idx->nb[1],
                                                n_embd_head_qk_rope*ggml_element_size(k_idx));
        k_idx_rope = ggml_rope_ext(ctx,
                ggml_reshape_3d(ctx, ggml_cont(ctx, k_idx_rope), n_embd_head_qk_rope, 1, llm.n_tokens),
                inp_pos, nullptr, llm.n_rot, llm.rope_type, llm.n_ctx_orig,
                llm.freq_base, llm.freq_scale, llm.ext_factor, llm.attn_factor,
                llm.beta_fast, llm.beta_slow);
        k_idx = ggml_concat(ctx,
                ggml_reshape_2d(ctx, k_idx_rope, n_embd_head_qk_rope, llm.n_tokens),
                ggml_cont(ctx, k_idx_pass), 0);                                      // [d_idx, T]
    }
    if (il == 0) ggml_set_name(k_idx, "opg0_idx_k");
    ggml_tensor * idx_w = ggml_view_2d(ctx, idx_cache, d_idx, llm.n_tokens,
                                       idx_cache->nb[1], llm.kv_head*idx_cache->nb[1]);
    ggml_tensor * cpy_idx = ggml_cpy(ctx, k_idx, idx_w);
    if ((size_t) il < llm.lctx.dsa_cache_copies.size()) {
        llm.lctx.dsa_cache_copies[il].cpy  = cpy_idx;
        llm.lctx.dsa_cache_copies[il].step = idx_cache->nb[1];
    }
    ggml_build_forward_expand(gf, cpy_idx);

    if (llm.n_kv <= topk) {
        return selection;
    }

    // Indexer queries.
    ggml_tensor * q_idx = ggml_mul_mat(ctx, layer.indexer_attn_q_b, q_lora); // [n_ihead*d_idx, T]
    q_idx = ggml_reshape_3d(ctx, q_idx, d_idx, n_ihead, llm.n_tokens);
    ggml_tensor * q_idx_offset = nullptr;
    if (openpangu_rope_offset_enabled() &&
        ggml_rope_ext_offset_valid(q_idx, inp_pos, nullptr, llm.n_rot, 0, llm.rope_type)) {
        q_idx_offset = ggml_rope_ext_offset(ctx, q_idx, inp_pos, nullptr, llm.n_rot, 0, llm.rope_type,
                                           llm.n_ctx_orig, llm.freq_base, llm.freq_scale,
                                           llm.ext_factor, llm.attn_factor, llm.beta_fast, llm.beta_slow);
    }
    if (openpangu_backend_supports_rope_offset(llm.lctx, layer.indexer_attn_q_b, q_idx_offset)) {
        q_idx = q_idx_offset;
    } else {
        ggml_tensor * q_idx_rope = ggml_view_3d(ctx, q_idx, n_embd_head_qk_rope, n_ihead, llm.n_tokens,
                                                q_idx->nb[1], q_idx->nb[2], 0);
        ggml_tensor * q_idx_pass = ggml_view_3d(ctx, q_idx, d_idx - n_embd_head_qk_rope,
                                                n_ihead, llm.n_tokens, q_idx->nb[1], q_idx->nb[2],
                                                n_embd_head_qk_rope*ggml_element_size(q_idx));
        q_idx_rope = ggml_rope_ext(ctx, ggml_cont(ctx, q_idx_rope), inp_pos, nullptr,
                                   llm.n_rot, llm.rope_type, llm.n_ctx_orig,
                                   llm.freq_base, llm.freq_scale, llm.ext_factor,
                                   llm.attn_factor, llm.beta_fast, llm.beta_slow);
        q_idx = ggml_concat(ctx, q_idx_rope, ggml_cont(ctx, q_idx_pass), 0); // [d_idx, n_ihead, T]
    }
    if (il == 0) ggml_set_name(q_idx, "opg0_idx_q");

    ggml_tensor * k_all_idx = ggml_view_2d(ctx, idx_cache, d_idx, llm.n_kv, idx_cache->nb[1], 0);
    ggml_tensor * w_idx = ggml_mul_mat(ctx, layer.indexer_proj, x_normed);  // [n_ihead, T]
    const bool chunk_scores = openpangu_idx_score_should_chunk(llm.n_tokens, OPENPANGU_IDX_SCORE_CHUNK);
    const bool defer_sel_mask_to_att_chunks =
        !dsa_gather_allowed &&
        openpangu_att_score_should_chunk(llm.n_kv, llm.hparams.param_sink_number,
                llm.n_head, llm.n_tokens, OPENPANGU_ATT_SCORE_CHUNK, OPENPANGU_ATT_FULL_KQ_MAX_MIB);

    // Automatic adoption is OpenPangu-only and follows the indexer projection's scheduled
    // backend. An unsupported accelerator therefore retains the exact legacy chain instead
    // of acquiring a CPU island. Explicit -fidx preserves its historical force-on behavior.
    ggml_tensor * idx_topk_candidate = nullptr;
    if (openpangu_consider_indexer_topk(llm.lctx, layer.indexer_proj)) {
        ggml_tensor * idx_mask = ggml_view_2d(ctx, KQ_mask, llm.n_kv, llm.n_tokens,
                                              KQ_mask->nb[1], 0);
        idx_topk_candidate = ggml_indexer_topk(
            ctx, k_all_idx, q_idx, w_idx, idx_mask, GGML_UNARY_OP_RELU, (int) topk);
        if (il == 0) ggml_set_name(idx_topk_candidate, "opg0_idx_sel_fused_candidate");
    }

    if (idx_topk_candidate != nullptr &&
        openpangu_use_indexer_topk(llm.lctx, layer.indexer_proj, idx_topk_candidate)) {
        selection.indices = idx_topk_candidate;                                  // [topk, T] i32
        if (il == 0) ggml_set_name(selection.indices, "opg0_idx_sel_fused");
        selection.gather_engaged = dsa_gather_allowed;
        if (selection.gather_engaged) {
            GGML_ASSERT(llm.n_kv >= topk + (int64_t) pad + llm.n_tokens);
        } else if (!defer_sel_mask_to_att_chunks) {
            ggml_tensor * base = ggml_fill(ctx,
                    ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, llm.n_kv, llm.n_tokens), -1e30f);
            ggml_tensor * zeros = ggml_fill(ctx,
                    ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, topk, llm.n_tokens), 0.0f);
            selection.mask = ggml_reshape_2d(ctx,
                    ggml_set_rows(ctx, base, zeros, selection.indices), llm.n_kv, llm.n_tokens);
        }
        return selection;
    }

    if (chunk_scores) {
        ggml_tensor * sel_mask_parts = nullptr;
        for (int64_t c0 = 0; c0 < llm.n_tokens; c0 += OPENPANGU_IDX_SCORE_CHUNK) {
            const int64_t tc = std::min<int64_t>(OPENPANGU_IDX_SCORE_CHUNK, llm.n_tokens - c0);
            ggml_tensor * q_idx_c = ggml_view_3d(ctx, q_idx, d_idx, n_ihead, tc,
                                                 q_idx->nb[1], q_idx->nb[2], (size_t) c0*q_idx->nb[2]);
            q_idx_c = ggml_cont(ctx, q_idx_c);
            ggml_tensor * w_idx_c = ggml_view_2d(ctx, w_idx, n_ihead, tc,
                                                 w_idx->nb[1], (size_t) c0*w_idx->nb[1]);
            w_idx_c = ggml_cont(ctx, w_idx_c);

            // Scores over the whole cache window, causal-masked before selection.
            // The chunked path never materializes the legacy [n_kv, n_ihead, T] score tensor.
            ggml_tensor * sc_c = ggml_mul_mat(ctx, k_all_idx, q_idx_c);              // [n_kv, n_ihead, Tc]
            sc_c = ggml_relu(ctx, sc_c);
            sc_c = ggml_mul(ctx, sc_c, ggml_reshape_3d(ctx, w_idx_c, 1, n_ihead, tc));
            sc_c = ggml_cont(ctx, ggml_permute(ctx, sc_c, 1, 0, 2, 3));             // [n_ihead, n_kv, Tc]
            sc_c = ggml_reshape_2d(ctx, ggml_sum_rows(ctx, sc_c), llm.n_kv, tc);    // [n_kv, Tc]
            sc_c = ggml_add(ctx, sc_c, ggml_cont(ctx,
                    ggml_view_2d(ctx, KQ_mask, llm.n_kv, tc,
                                 KQ_mask->nb[1], (size_t) c0*KQ_mask->nb[1])));
            ggml_tensor * sel_idx_c = ggml_top_k(ctx, sc_c, (int) topk);            // [topk, Tc] i32
            selection.indices = selection.indices == nullptr ? sel_idx_c :
                ggml_concat(ctx, selection.indices, sel_idx_c, 1);

            if (!dsa_gather_allowed && !defer_sel_mask_to_att_chunks) {
                ggml_tensor * base_c = ggml_fill(ctx,
                        ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, llm.n_kv, tc), -1e30f);
                ggml_tensor * zeros_c = ggml_fill(ctx,
                        ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, topk, tc), 0.0f);
                ggml_tensor * sel_mask_c = ggml_reshape_2d(ctx,
                        ggml_set_rows(ctx, base_c, zeros_c, sel_idx_c), llm.n_kv, tc);
                sel_mask_parts = sel_mask_parts == nullptr ? sel_mask_c :
                    ggml_concat(ctx, sel_mask_parts, sel_mask_c, 1);
            }
        }
        if (il == 0) ggml_set_name(selection.indices, "opg0_idx_sel");
        selection.gather_engaged = dsa_gather_allowed;
        if (selection.gather_engaged) {
            GGML_ASSERT(llm.n_kv >= topk + (int64_t) pad + llm.n_tokens);
        } else if (!defer_sel_mask_to_att_chunks) {
            selection.mask = sel_mask_parts;
        }
        return selection;
    }

    GGML_ASSERT(llm.n_tokens <= 14 || OPENPANGU_IDX_SCORE_CHUNK == 0 ||
                llm.n_tokens <= OPENPANGU_IDX_SCORE_CHUNK);
    ggml_tensor * sc = ggml_mul_mat(ctx, k_all_idx, q_idx);                 // [n_kv, n_ihead, T]
    sc = ggml_relu(ctx, sc);
    sc = ggml_mul(ctx, sc, ggml_reshape_3d(ctx, w_idx, 1, n_ihead, llm.n_tokens));
    sc = ggml_cont(ctx, ggml_permute(ctx, sc, 1, 0, 2, 3));                // [n_ihead, n_kv, T]
    sc = ggml_reshape_2d(ctx, ggml_sum_rows(ctx, sc), llm.n_kv, llm.n_tokens); // [n_kv, T]
    sc = ggml_add(ctx, sc, ggml_cont(ctx,
            ggml_view_2d(ctx, KQ_mask, llm.n_kv, llm.n_tokens, KQ_mask->nb[1], 0)));
    if (il == 0) ggml_set_name(sc, "opg0_idx_scores");

    // Exact top-k -> additive mask: scatter zeros into a -1e30 base at the selected
    // positions ([1, n_kv, T] row layout makes set_rows a per-query scatter). The legacy
    // scatter consumes this strided top_k view directly; gathered attention makes it contiguous.
    selection.indices = ggml_top_k(ctx, sc, (int) topk);                  // [topk, T] i32
    if (il == 0) ggml_set_name(selection.indices, "opg0_idx_sel");
    selection.gather_engaged = dsa_gather_allowed;
    if (selection.gather_engaged) {
        GGML_ASSERT(llm.n_kv >= topk + (int64_t) pad + llm.n_tokens);
    } else {
        ggml_tensor * base = ggml_fill(ctx,
                ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, llm.n_kv, llm.n_tokens), -1e30f);
        ggml_tensor * zeros = ggml_fill(ctx,
                ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, topk, llm.n_tokens), 0.0f);
        selection.mask = ggml_reshape_2d(ctx,
                ggml_set_rows(ctx, base, zeros, selection.indices), llm.n_kv, llm.n_tokens);
    }
    return selection;
}

struct openpangu_mhc_pre_result {
    ggml_tensor * x;
    ggml_tensor * h_post;
    ggml_tensor * h_res;
};

static openpangu_mhc_pre_result openpangu_build_mhc_pre(
        llm_build_context & llm, ggml_cgraph * gf,
        ggml_tensor * Rin, ggml_tensor * phi, ggml_tensor * alpha,
        ggml_tensor * beta, ggml_tensor * gamma, int64_t n_streams) {
    ggml_context * ctx = llm.ctx0;
    if (!ggml_is_contiguous(Rin)) {
        Rin = ggml_cont(ctx, Rin);
    }
    ggml_tensor * flat = ggml_reshape_2d(ctx, Rin, llm.n_embd*n_streams, llm.n_tokens);
    ggml_tensor * normed = ggml_fused_rms_norm(
            ctx, flat, gamma, llm.hparams.f_norm_rms_eps);
    ggml_tensor * mixes = ggml_mul_mat(ctx, phi, normed); // [(S+2)*S, T]
    ggml_tensor * h_pre = ggml_view_2d(ctx, mixes, n_streams, llm.n_tokens,
                                       mixes->nb[1], 0);
    ggml_tensor * h_post = ggml_view_2d(ctx, mixes, n_streams, llm.n_tokens,
                                        mixes->nb[1], n_streams*ggml_element_size(mixes));
    ggml_tensor * h_res = ggml_view_2d(ctx, mixes, n_streams*n_streams, llm.n_tokens,
                                       mixes->nb[1], 2*n_streams*ggml_element_size(mixes));

    // alpha = [a_pre, a_post, a_res]; beta = [b_pre(S), b_post(S), b_res(S*S)]
    ggml_tensor * a_pre = ggml_view_1d(ctx, alpha, 1, 0);
    ggml_tensor * b_pre = ggml_view_1d(ctx, beta, n_streams, 0);
    // cont is required: the CUDA broadcast-mul path misreads strided views (h_pre is a
    // row-slice of mixes), while CPU handles the strides: token 0 right, tokens 1+ garbage.
    h_pre = ggml_add(ctx, ggml_mul(ctx, ggml_cont(ctx, h_pre), a_pre), b_pre);
    h_pre = ggml_sigmoid(ctx, h_pre); // [S,T] (+eps omitted, inert)

    // combine: x[h,t] = sum_s h_pre[s,t] * R[h,s,t]
    ggml_tensor * hpre3 = ggml_reshape_3d(ctx, h_pre, 1, n_streams, llm.n_tokens);
    ggml_tensor * weighted = ggml_mul(ctx, Rin, hpre3); // [H,S,T]
    ggml_tensor * x = ggml_reshape_2d(
            ctx, ggml_sum_rows_ext(ctx, weighted, 1), llm.n_embd, llm.n_tokens);
    ggml_build_forward_expand(gf, x);

    ggml_tensor * h_post_cont = ggml_cont(ctx, h_post);
    ggml_tensor * h_res_cont = ggml_cont(ctx, h_res);
    return { x, h_post_cont, h_res_cont };
}

static ggml_tensor * openpangu_build_mhc_post(
        llm_build_context & llm,
        ggml_tensor * y, ggml_tensor * h_post, ggml_tensor * Rin,
        ggml_tensor * alpha, ggml_tensor * beta, ggml_tensor * h_res,
        int64_t n_streams, int sink_iters) {
    ggml_context * ctx = llm.ctx0;
    ggml_tensor * a_post = ggml_view_1d(ctx, alpha, 1, ggml_element_size(alpha));
    ggml_tensor * a_res  = ggml_view_1d(ctx, alpha, 1, 2*ggml_element_size(alpha));
    ggml_tensor * b_post = ggml_view_1d(ctx, beta, n_streams,
                                        n_streams*ggml_element_size(beta));
    ggml_tensor * b_res = ggml_view_1d(ctx, beta, n_streams*n_streams,
                                       2*n_streams*ggml_element_size(beta));

    h_post = ggml_add(ctx, ggml_mul(ctx, h_post, a_post), b_post);
    h_post = ggml_scale(ctx, ggml_sigmoid(ctx, h_post), 2.0f); // [S,T]

    ggml_tensor * m = ggml_add(ctx, ggml_mul(ctx, h_res, a_res), b_res); // [S*S,T]
    m = ggml_sinkhorn(ctx, m, (int) n_streams, sink_iters, 0.0f,
                      /*output_transposed=*/true); // [row S, col S, T]

    // term1: h_post[s,t]*y[h,t] -> [H,S,T]
    ggml_tensor repeater;
    repeater.ne[0] = llm.n_embd;
    repeater.ne[1] = n_streams;
    repeater.ne[2] = llm.n_tokens;
    repeater.ne[3] = 1;
    ggml_tensor * y3 = ggml_reshape_3d(ctx, y, llm.n_embd, 1, llm.n_tokens);
    ggml_tensor * hpost3 = ggml_reshape_3d(
            ctx, ggml_cont(ctx, h_post), 1, n_streams, llm.n_tokens);
    ggml_tensor * term1 = ggml_mul(ctx, ggml_repeat(ctx, y3, &repeater), hpost3);

    // term2: sum_j m[j,s,t]*R[h,j,t], directly consuming Sinkhorn's
    // output-transposed [input-row j, output-column s, token t] layout.
    ggml_tensor * term2 = nullptr;
    // Precheck the operand contract so an unsupported layout leaves the candidate unbuilt and
    // the gate below falls back to the explicit per-slot mix loop instead of aborting.
    ggml_tensor * batched_mix = ggml_batched_mix_ext_valid(Rin, m)
            ? ggml_batched_mix_ext(ctx, Rin, m) : nullptr;
    if (batched_mix != nullptr) {
        ggml_set_name(batched_mix, "opg_mhc_batched_mix");
    }
    if (openpangu_backend_supports_batched_mix(llm.lctx, alpha, batched_mix)) {
        term2 = batched_mix;
    } else {
        for (int64_t s = 0; s < n_streams; ++s) {
            ggml_tensor * m_s = ggml_cont(ctx,
                    ggml_view_2d(ctx, m, n_streams, llm.n_tokens,
                                 m->nb[2], s*m->nb[1]));
            ggml_tensor * m_s3 = ggml_reshape_3d(ctx, m_s, 1, n_streams, llm.n_tokens);
            ggml_tensor * acc = ggml_mul(ctx, Rin, m_s3);
            ggml_tensor * summed = ggml_sum_rows_ext(ctx, acc, 1);
            term2 = term2 ? ggml_concat(ctx, term2, summed, 1) : summed;
        }
    }
    return ggml_add(ctx, term1, term2); // [H,S,T]
}

static ggml_tensor * openpangu_build_rms_norm_add(
        llm_build_context & llm, ggml_tensor * x,
        ggml_tensor * weight, ggml_tensor * residual, int il) {
    if (openpangu_fused_ops_enabled()) {
        ggml_tensor * candidate = ggml_fused_rms_norm_add(
                llm.ctx0, x, weight, residual, llm.hparams.f_norm_rms_eps);
        if (openpangu_backend_supports_fused_rms_norm_add(llm.lctx, weight, candidate)) {
            return candidate;
        }
    }
    x = llm.llm_build_norm(
            llm.ctx0, x, llm.hparams, weight, NULL, LLM_NORM_RMS, llm.cb, il);
    return ggml_add(llm.ctx0, x, residual);
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
    openpangu_build_latent_cache_write(
            *this, gf, il, ckv, k_pe2d, kv_lora_rank, n_embd_head_qk_rope);

    // ---- DSA lightning indexer: per-query top-k selection mask (DSA layers only) ----
    const openpangu_dsa_selection dsa = openpangu_build_dsa_selection(
            *this, gf, layer, il, x_normed, q_lora, KQ_mask, inp_pos);
    ggml_tensor * sel_mask = dsa.mask;       // [n_kv, T] additive mask
    ggml_tensor * sel_idx  = dsa.indices;    // [topk, T] selected cache rows
    const bool dsa_gather_engaged = dsa.gather_engaged;
    const int64_t dsa_topk = dsa.topk;
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

    // Capability probe: build a representative dense latent-attention candidate and adopt the
    // fused path only when this layer's scheduled backend can execute it. When the legacy
    // chain is chosen (unset probe, unsupported backend, or forced legacy) the probe node is
    // left unreferenced, so it never enters the built graph.
    ggml_tensor * fused_attn_probe = nullptr;
    if (openpangu_fused_attn_enabled(hparams)) {
        ggml_tensor * kl_raw_probe = openpangu_build_k_latent_raw(ctx0, kv_self, il, n_kv_attn, win_off);
        ggml_tensor * mask_probe = KQ_mask ?
            ggml_view_2d(ctx0, KQ_mask, n_kv_attn, n_tokens, KQ_mask->nb[1], 0) : nullptr;
        // Operand-contract precheck on the representative candidate: if the fused op cannot
        // represent this layer's cache/mask layout, leave the probe unset so the gate declines
        // to the legacy chain instead of aborting in the constructor. The adopted per-path ops
        // share this cache/dv/dv_off/mask contract (the indexed paths' selection indices are
        // I32-contiguous by construction), so a valid probe implies a valid adopted op.
        if (ggml_latent_attn_ext_valid(q_all, kl_raw_probe, sink_blk, s_lat_t, mask_probe,
                                       nullptr, kv_lora_rank, 0, hparams.f_max_alibi_bias, 0)) {
            fused_attn_probe = ggml_latent_attn_prefix_ext(
                    ctx0, q_all, kl_raw_probe, sink_blk, s_lat_t, mask_probe,
                    kv_lora_rank, 0, kq_scale, hparams.f_max_alibi_bias);
        }
    }
    const bool use_fused_attn =
        openpangu_backend_supports_fused_attn(lctx, hparams, layer.wv_b, fused_attn_probe);
    if (use_dsa_gather) {
        ggml_tensor * kqv = nullptr;
        if (use_fused_attn) {
            // sel_idx contains absolute cache rows selected after the causal indexer mask,
            // so gathered attention is maskless just like the legacy chain.
            ggml_tensor * kl_raw = openpangu_build_k_latent_raw(ctx0, kv_self, il, n_kv, 0);
            ggml_tensor * sel_idx_cont = ggml_cont(ctx0, sel_idx);                           // [topk, T] i32
            kqv = ggml_latent_attn_indexed_ext(ctx0, q_all, kl_raw, sink_blk, s_lat_t, nullptr,
                                                sel_idx_cont, kv_lora_rank, 0,
                                                kq_scale, 0.0f);                              // [512, T, H]
        } else {
            // Bring-up opt-out: preserve the complete pre-gather + explicit attention chain.
            ggml_tensor * k_gath = nullptr;
            ggml_tensor * kq_cache = nullptr;
            ggml_tensor * kq_sinks = ggml_mul_mat(ctx0, sink_blk, q_all);                   // [NS, T, H]
            if (n_tokens == 1) {
                ggml_tensor * kl_full = ggml_view_2d(ctx0, kv_self.k_l[il],
                                                     kv_lora_rank + n_embd_head_qk_rope, n_kv,
                                                     kv_self.k_l[il]->nb[1], 0);
                ggml_tensor * sel_idx_flat = ggml_cont_2d(ctx0, sel_idx, dsa_topk, 1);       // [topk] i32
                k_gath = ggml_get_rows(ctx0, kl_full, sel_idx_flat);                         // [576, topk] f32
                k_gath = openpangu_cast_gathered_latent_for_cache_type(
                    ctx0, k_gath, kv_self.k_l[il]);
                k_gath = ggml_reshape_3d(ctx0, k_gath,
                                         kv_lora_rank + n_embd_head_qk_rope, dsa_topk, 1);
                kq_cache = ggml_mul_mat(ctx0,
                        ggml_reshape_2d(ctx0, k_gath,
                                        kv_lora_rank + n_embd_head_qk_rope, dsa_topk),
                        q_all);                                                               // [topk, 1, H]
            } else {
                ggml_tensor * kl_full = ggml_view_2d(ctx0, kv_self.k_l[il],
                                                     kv_lora_rank + n_embd_head_qk_rope, n_kv,
                                                     kv_self.k_l[il]->nb[1], 0);
                ggml_tensor * sel_idx_flat = ggml_cont_2d(
                    ctx0, sel_idx, dsa_topk*n_tokens, 1);                                    // [topk*T] i32
                k_gath = ggml_get_rows(ctx0, kl_full, sel_idx_flat);                         // [576, topk*T] f32
                k_gath = openpangu_cast_gathered_latent_for_cache_type(
                    ctx0, k_gath, kv_self.k_l[il]);
                k_gath = ggml_reshape_3d(ctx0, k_gath,
                                         kv_lora_rank + n_embd_head_qk_rope,
                                         dsa_topk, n_tokens);                                // [576, topk, T]
                ggml_tensor * q_gath = ggml_cont(ctx0,
                        ggml_permute(ctx0, q_all, 0, 2, 1, 3));                              // [576, H, T]
                kq_cache = ggml_mul_mat(ctx0, k_gath, q_gath);                              // [topk, H, T]
                kq_cache = ggml_cont(ctx0,
                        ggml_permute(ctx0, kq_cache, 0, 2, 1, 3));                           // [topk, T, H]
            }
            ggml_tensor * kq = ggml_concat(ctx0, kq_sinks, kq_cache, 0);                    // [NS+topk, T, H]
            // sel_idx came from scores after adding the causal KQ_mask. The engagement bound
            // gives every token at least topk valid positions, so all gathered rows are visible.
            kq = ggml_soft_max_ext(ctx0, kq, nullptr, kq_scale, hparams.f_max_alibi_bias);

            ggml_tensor * kq_s = ggml_view_3d(ctx0, kq, NS, n_tokens, n_head,
                                               kq->nb[1], kq->nb[2], 0);
            ggml_tensor * kq_c = ggml_view_3d(ctx0, kq, n_kv_attn, n_tokens, n_head,
                                               kq->nb[1], kq->nb[2],
                                               NS*ggml_element_size(kq));
            ggml_tensor * kqv_cache = nullptr;
            GGML_ASSERT(k_gath != nullptr);
            if (n_tokens == 1) {
                ggml_tensor * v_gath = ggml_view_2d(ctx0, k_gath, kv_lora_rank, dsa_topk,
                                                    k_gath->nb[1], 0);                       // [512, topk]
                ggml_tensor * v_gath_t = ggml_cont(ctx0, ggml_transpose(ctx0, v_gath));      // [topk, 512]
                kqv_cache = ggml_mul_mat(ctx0, v_gath_t, kq_c);                             // [512, 1, H]
            } else {
                ggml_tensor * v_gath = ggml_view_3d(ctx0, k_gath, kv_lora_rank, dsa_topk,
                                                    n_tokens, k_gath->nb[1], k_gath->nb[2], 0); // [512, topk, T]
                ggml_tensor * v_gath_t = ggml_cont(ctx0,
                        ggml_permute(ctx0, v_gath, 1, 0, 2, 3));                             // [topk, 512, T]
                ggml_tensor * kq_c_gath = ggml_cont(ctx0,
                        ggml_permute(ctx0, kq_c, 0, 2, 1, 3));                               // [topk, H, T]
                kqv_cache = ggml_mul_mat(ctx0, v_gath_t, kq_c_gath);                        // [512, H, T]
                kqv_cache = ggml_cont(ctx0,
                        ggml_permute(ctx0, kqv_cache, 0, 2, 1, 3));                          // [512, T, H]
            }
            kqv = ggml_add(ctx0,
                    ggml_mul_mat(ctx0, s_lat_t, kq_s),
                    kqv_cache);                                                              // [512, T, H]
        }
        ggml_tensor * wv_b3 = ggml_reshape_3d(ctx0, layer.wv_b, kv_lora_rank, n_embd_head_v, n_head);
        ggml_tensor * out_h = ggml_mul_mat(ctx0, wv_b3, kqv);                               // [128, T, H]
        ggml_tensor * merged = ggml_cont(ctx0, ggml_permute(ctx0, out_h, 0, 2, 1, 3));      // [128, H, T]
        cur = ggml_reshape_2d(ctx0, merged, n_embd_head_v * n_head, n_tokens);
    } else {
        const bool use_dsa_sel_idx_mask = sel_idx != nullptr && dsa_topk > 0;
        const bool chunk_att =
            openpangu_att_score_should_chunk(n_kv_attn, NS, n_head, n_tokens,
                    OPENPANGU_ATT_SCORE_CHUNK, OPENPANGU_ATT_FULL_KQ_MAX_MIB) &&
            // Dense/SWA and MTP fused attention tiles the full T internally. DSA prefill
            // keeps this loop because it also owns the gathered path, the opt-out's legacy
            // subchunks, and deferred selection-mask construction for non-gather chunks.
            (!use_fused_attn || use_dsa_sel_idx_mask);
        ggml_tensor * kqv = nullptr;
        if (chunk_att) {
            const bool can_prefill_gather =
                use_dsa_sel_idx_mask && !use_swa_window &&
                hparams.f_max_alibi_bias == 0.0f &&
                openpangu_dsa_gather_rows_fit_cuda(dsa_topk, 1);
            ggml_tensor * kl_full = nullptr;
            ggml_tensor * kl_raw_full = nullptr;
            if (can_prefill_gather && use_fused_attn) {
                kl_raw_full = openpangu_build_k_latent_raw(ctx0, kv_self, il, n_kv, 0);
            } else if (can_prefill_gather) {
                kl_full = ggml_view_2d(ctx0, kv_self.k_l[il],
                                       kv_lora_rank + n_embd_head_qk_rope, n_kv,
                                       kv_self.k_l[il]->nb[1], 0);
            }

            for (int64_t c0 = 0; c0 < n_tokens; c0 += OPENPANGU_ATT_SCORE_CHUNK) {
                const int64_t tc = std::min<int64_t>(OPENPANGU_ATT_SCORE_CHUNK, n_tokens - c0);
                ggml_tensor * q_all_c = ggml_view_3d(ctx0, q_all, kv_lora_rank + n_embd_head_qk_rope,
                                                     tc, n_head, q_all->nb[1], q_all->nb[2],
                                                     (size_t) c0*q_all->nb[1]);
                q_all_c = ggml_cont(ctx0, q_all_c);

                ggml_tensor * kq_sinks_c = use_fused_attn ? nullptr :
                    ggml_mul_mat(ctx0, sink_blk, q_all_c);                                   // [NS, Tc, H]

                const bool prefill_gather_chunk =
                    can_prefill_gather &&
                    openpangu_dsa_prefill_gather_should_engage(n_kv, n_tokens, c0, tc, dsa_topk,
                                                               openpangu_kv_cache_pad(cparams));
                ggml_tensor * kqv_c = nullptr;
                if (prefill_gather_chunk) {
                    if (use_fused_attn) {
                        GGML_ASSERT(kl_raw_full != nullptr);
                        ggml_tensor * sel_idx_c = ggml_view_2d(ctx0, sel_idx, dsa_topk, tc,
                                                               sel_idx->nb[1],
                                                               (size_t) c0*sel_idx->nb[1]);
                        sel_idx_c = ggml_cont(ctx0, sel_idx_c);                               // [topk, Tc] i32
                        // Selection consumed the causal mask before top-k, matching the
                        // legacy maskless softmax. The indexed op therefore receives no mask.
                        kqv_c = ggml_latent_attn_indexed_ext(ctx0, q_all_c, kl_raw_full,
                                                             sink_blk, s_lat_t, nullptr,
                                                             sel_idx_c, kv_lora_rank, 0,
                                                             kq_scale, 0.0f);                 // [512, Tc, H]
                    } else {
                        // Bring-up opt-out: retain the CUDA-grid-safe get_rows subchunks.
                        GGML_ASSERT(kl_full != nullptr && kq_sinks_c != nullptr);
                        const int64_t gather_token_chunk =
                            openpangu_dsa_gather_tokens_per_get_rows(dsa_topk);
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
                                                                   sel_idx->nb[1],
                                                                   (size_t) (c0 + g0)*sel_idx->nb[1]);
                            ggml_tensor * sel_idx_flat_g = ggml_cont_2d(ctx0, sel_idx_g, dsa_topk*tg, 1);
                            ggml_tensor * k_gath_g = ggml_get_rows(ctx0, kl_full, sel_idx_flat_g); // [576, topk*tg]
                            k_gath_g = openpangu_cast_gathered_latent_for_cache_type(
                                ctx0, k_gath_g, kv_self.k_l[il]);
                            k_gath_g = ggml_reshape_3d(ctx0, k_gath_g,
                                                       kv_lora_rank + n_embd_head_qk_rope,
                                                       dsa_topk, tg);                         // [576, topk, tg]
                            ggml_tensor * q_gath_g = ggml_cont(ctx0,
                                ggml_permute(ctx0, q_all_g, 0, 2, 1, 3));                    // [576, H, tg]
                            ggml_tensor * kq_cache_g = ggml_mul_mat(ctx0, k_gath_g, q_gath_g); // [topk, H, tg]
                            kq_cache_g = ggml_cont(ctx0,
                                ggml_permute(ctx0, kq_cache_g, 0, 2, 1, 3));                 // [topk, tg, H]
                            ggml_tensor * kq_g_all = ggml_concat(ctx0, kq_sinks_g,
                                                                 kq_cache_g, 0);             // [NS+topk, tg, H]
                            kq_g_all = ggml_soft_max_ext(ctx0, kq_g_all, nullptr,
                                                         kq_scale, hparams.f_max_alibi_bias);

                            ggml_tensor * kq_s_g = ggml_view_3d(ctx0, kq_g_all, NS, tg, n_head,
                                                                kq_g_all->nb[1], kq_g_all->nb[2], 0);
                            ggml_tensor * kq_cache_soft_g = ggml_view_3d(
                                ctx0, kq_g_all, dsa_topk, tg, n_head,
                                kq_g_all->nb[1], kq_g_all->nb[2],
                                NS*ggml_element_size(kq_g_all));
                            ggml_tensor * v_gath_g = ggml_view_3d(ctx0, k_gath_g,
                                                                  kv_lora_rank, dsa_topk, tg,
                                                                  k_gath_g->nb[1], k_gath_g->nb[2], 0);
                            ggml_tensor * v_gath_t_g = ggml_cont(ctx0,
                                ggml_permute(ctx0, v_gath_g, 1, 0, 2, 3));                   // [topk, 512, tg]
                            ggml_tensor * kq_cache_gath_g = ggml_cont(ctx0,
                                ggml_permute(ctx0, kq_cache_soft_g, 0, 2, 1, 3));             // [topk, H, tg]
                            ggml_tensor * kqv_cache_g = ggml_mul_mat(
                                ctx0, v_gath_t_g, kq_cache_gath_g);                           // [512, H, tg]
                            kqv_cache_g = ggml_cont(ctx0,
                                ggml_permute(ctx0, kqv_cache_g, 0, 2, 1, 3));                // [512, tg, H]
                            ggml_tensor * kqv_g = ggml_add(ctx0,
                                    ggml_mul_mat(ctx0, s_lat_t, kq_s_g),
                                    kqv_cache_g);                                              // [512, tg, H]
                            kqv_c = kqv_c == nullptr ? kqv_g :
                                ggml_concat(ctx0, kqv_c, kqv_g, 1);
                        }
                    }
                } else if (use_fused_attn) {
                    // DSA legacy fallback inside the gather-owned outer loop. Dense/SWA/MTP
                    // fused attention bypasses the loop and uses the full-span call below.
                    ggml_tensor * kl_raw = openpangu_build_k_latent_raw(ctx0, kv_self, il, n_kv_attn, win_off);
                    ggml_tensor * mask_eff = ggml_view_2d(ctx0, KQ_mask, n_kv_attn, tc,
                                                          KQ_mask->nb[1], (size_t) c0*KQ_mask->nb[1]);
                    if (sel_mask) {
                        ggml_tensor * sel_mask_c = ggml_view_2d(ctx0, sel_mask, n_kv_attn, tc,
                                                                sel_mask->nb[1], (size_t) c0*sel_mask->nb[1]);
                        mask_eff = ggml_add(ctx0, mask_eff, sel_mask_c);
                    } else if (use_dsa_sel_idx_mask) {
                        ggml_tensor * sel_idx_c = ggml_view_2d(ctx0, sel_idx, dsa_topk, tc,
                                                               sel_idx->nb[1], (size_t) c0*sel_idx->nb[1]);
                        ggml_tensor * base_c = ggml_fill(ctx0, ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, 1, n_kv_attn, tc), -1e30f);
                        ggml_tensor * zeros_c = ggml_fill(ctx0, ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, 1, dsa_topk, tc), 0.0f);
                        ggml_tensor * sel_mask_c = ggml_reshape_2d(ctx0, ggml_set_rows(ctx0, base_c, zeros_c, sel_idx_c), n_kv_attn, tc);
                        mask_eff = ggml_add(ctx0, mask_eff, sel_mask_c);
                    }
                    kqv_c = ggml_latent_attn_prefix_ext(ctx0, q_all_c, kl_raw, sink_blk, s_lat_t, mask_eff,
                                                        kv_lora_rank, 0, kq_scale, hparams.f_max_alibi_bias); // [512, Tc, H]
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
        } else if (use_fused_attn) {
            // Fused latent attention: one op replaces [prefix|cache] QK, joint softmax, the
            // two value contractions, the zero-prefix mask, and the value transpose. The op
            // reads the raw (possibly q8) cache and dequantizes internally.
            GGML_ASSERT(!use_dsa_sel_idx_mask || sel_mask != nullptr);
            ggml_tensor * kl_raw = openpangu_build_k_latent_raw(ctx0, kv_self, il, n_kv_attn, win_off);
            ggml_tensor * mask_eff = ggml_view_2d(ctx0, KQ_mask, n_kv_attn, n_tokens, KQ_mask->nb[1], 0);
            if (sel_mask) {
                mask_eff = ggml_add(ctx0, mask_eff, sel_mask);
            }
            kqv = ggml_latent_attn_prefix_ext(ctx0, q_all, kl_raw, sink_blk, s_lat_t, mask_eff,
                                              kv_lora_rank, 0, kq_scale, hparams.f_max_alibi_bias); // [512, T, H]
        } else {
            GGML_ASSERT(!use_dsa_sel_idx_mask || sel_mask != nullptr);
            ggml_tensor * kq_sinks = ggml_mul_mat(ctx0, sink_blk, q_all);                   // [NS, T, H]
            ggml_tensor * kq_cache = ggml_mul_mat(ctx0, kl_all, q_all);                     // [n_kv_attn, T, H]
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
    ggml_tensor * ffn_inp = openpangu_build_rms_norm_add(
            *this, cur, mtp_layer.attn_post_norm, inpSA, il);
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
    cur = openpangu_build_rms_norm_add(
            *this, cur, mtp_layer.ffn_post_norm, ffn_inp, il);
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
    openpangu_clear_cache_writes(lctx);

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

    // Base generation uses only the transformer layers; the trailing NextN/MTP layers are skipped.
    const int n_layer_base = n_layer - (int) hparams.nextn_predict_layers;
    for (int il = 0; il < n_layer_base; ++il) {
        auto & layer = model.layers[il];
        ggml_tensor * conv_state = (size_t) il < kv_self.s_l.size() ? kv_self.s_l[il] : nullptr;

        // ================= attention sublayer =================
        const openpangu_mhc_pre_result attn_mhc = openpangu_build_mhc_pre(
                *this, gf, R, layer.mhc_attn_phi, layer.mhc_attn_alpha,
                layer.mhc_attn_beta, layer.mhc_attn_gamma, S);
        ggml_tensor * x = attn_mhc.x;
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
        R = openpangu_build_mhc_post(
                *this, cur, attn_mhc.h_post, R, layer.mhc_attn_alpha,
                layer.mhc_attn_beta, attn_mhc.h_res, S, sink_iters);
        if (il == 0) ggml_set_name(R, "opg0_R_attn");

        // ================= ffn sublayer =================
        const openpangu_mhc_pre_result mlp_mhc = openpangu_build_mhc_pre(
                *this, gf, R, layer.mhc_mlp_phi, layer.mhc_mlp_alpha,
                layer.mhc_mlp_beta, layer.mhc_mlp_gamma, S);
        ggml_tensor * xm = mlp_mhc.x;
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

        R = openpangu_build_mhc_post(
                *this, cur, mlp_mhc.h_post, R, layer.mhc_mlp_alpha,
                layer.mhc_mlp_beta, mlp_mhc.h_res, S, sink_iters);

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
        cur = ggml_reshape_2d(ctx0, ggml_sum_rows_ext(ctx0, weighted, 1), n_embd, n_tokens);
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

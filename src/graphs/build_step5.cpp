#include "../llama-build-context.h"
#include "../llama-model.h"
#include "../llama-context.h"

// Step-5 sparse GQA indexer: persistent proxy-key cache plumbing + proxy forward &
// read-back scoring.
//
// kv_self.kr_l[il] ([indexer_head_size == proxy_dim, kv_size], idx_type_k) holds one proxy
// indexer key per cached cell on FULL-attn layers only (SWA layers carry no indexer):
//   - write:  ggml_cpy per graph, rows kv_head..kv_head+n_tokens, registered in
//             lctx.dsa_cache_copies so llama_context::update_cache_copies() re-points the
//             baked view_offs when a graph is reused (GLM-DSA dsa_cache_copies pattern).
//             Without that registration a reused decode graph keeps scattering this ubatch's
//             keys into the first ubatch's slot.
//   - defrag: rows follow their cells (generic kr_l block in llm_build_context::build_defrag).
//   - shift:  rows carry NO position encoding (RoPE is applied at READ-BACK), so a K-shift
//             leaves them untouched -- "copy-through" by construction (build_k_shift()).
//
// The write source is the REAL proxy key: k_norm(k_proj(inpL)) over proxy_dim, cached RAW.
// k_norm is position-independent so it belongs at write time; only the pe RoPE (pe/nope split
// over indexer_rope_dim) is deferred to read-back, where the cached key is rotated by its cell's
// absolute position (lctx.inp_kv_pos, host-filled from kv_self.cells in llama_set_inputs) and the
// proxy query by inp_pos. RoPE-at-read-back keeps the cached key position-free, and keeps block
// pooling over cached keys CSA-friendly.
// Shapes/score logic ported INLINE from build_deepseek2_dsa_indexer (build_deepseek2.cpp:374;
// k_norm layernorm+bias :417, pe/nope split rope_dim 32, per-head w + sum :488) -- port logic,
// not dependencies.

// ---- semantic env flags (read ONCE per process) ----
// Precedent: DSA_HADAMARD_DISABLE frozen at first build (build_deepseek2.cpp:440). Frozen
// values keep cache contents self-consistent within a process (flipping IK_HADAMARD mid-process
// would mix rotated/unrotated cached rows).
// IK_HADAMARD (default 1; IK_HADAMARD=0 disables): Walsh-Hadamard whitening of proxy keys
// BEFORE the idx_type_k cache write, de-rotated at read-back (ggml_hadamard is orthonormal,
// H^2 == I -- hadamard.cu accumulates 1/sqrt(2) per butterfly stage). Quantizing whitened keys
// spreads each dim's energy over all quant blocks -> cache-Q8 error decorrelates from the score
// (Q8 ~= F16 KV-cache precision win). Exact-arithmetic scores identical to IK_HADAMARD=0:
// rope(H(H k_raw)) == rope(k_raw), so all plumbing invariants hold under either value.
static bool ik_hadamard_enabled() {
    // vLLM #61727 cross-diff: the reference has NO key rotation -- Hadamard existed solely to
    // decorrelate quantized-(Q8) cache error. Default caches are F16 (more mantissa than the
    // reference's bf16), so whitening is pointless overhead + a semantics divergence.
    // Default stays ON regardless: with quantized indexer-K caches (-ctv q6_0) un-whitened
    // keys let cache-quant error correlate with selection scores -> deep-context decode
    // corruption. IK_HADAMARD=0 is for F16-cache cross-diff runs ONLY.
    static const bool on = getenv("IK_HADAMARD") == nullptr || atoi(getenv("IK_HADAMARD")) != 0;
    return on;
}

// vLLM #61727 / official cross-diff: the reference model runs ALL indexer activations in bf16
// (torch: every matmul/norm/rope output materializes bf16); only the score matmul sees e4m3.
// Our ggml graph computes F32 throughout -- materialize the bf16 rounding at the same points
// torch does (cast F32->BF16->F32 is exact apart from the intended rounding).
static ggml_tensor * step5_bf16_roundtrip(ggml_context * ctx, ggml_tensor * t) {
    return ggml_cast(ctx, ggml_cast(ctx, t, GGML_TYPE_BF16), GGML_TYPE_F32);
}
// IK_SPARSE (default 0 = dense telemetry-only; IK_SPARSE=1 activates indexer selection):
// top-M rank-penalty sparse mask -> build_std_attention additive mask on FULL attn layers (SWA
// unchanged). Frozen per process like IK_HADAMARD (selection must stay consistent with the
// cached-key set across decode steps). Dense-equivalence property: at ctx <= topk the
// builder's clamp makes pen(rank)==0 for every key, so the sparse mask == the causal mask BITWISE
// and IK_SPARSE=1 reproduces IK_SPARSE=0 logits EXACTLY.
static bool ik_sparse_enabled() {
    static const bool on = getenv("IK_SPARSE") != nullptr && atoi(getenv("IK_SPARSE")) != 0;
    return on;
}
// The builder implements ONLY the official reference semantics: CSA compression = per-dim
// softmax(z)-weighted mean (kernel.py:283) via the FUSED ggml_csa_compress op with the
// BF16->E4M3 double-round epilogue (round=1), official e4m3 activation rounding of the
// indexer q/k before the score matmul, and BLOCK-grain hard top-512-region selection.
// The decomposed chain and the F32-summary fused variant SURVIVE only as backend
// fallbacks (Q8_0 z cache / supports_op failure), not as modes. Former env vars
// (IK_CSA / IK_CSA_E4M3 / IK_QK_E4M3 / IK_SEL_GRAIN) are silently ignored.

// Official CSA needs the persistent z cache (kz_l). Read by llama_kv_cache_init() and
// llama_model::cache_size() via the llama-context.h decl.
bool llama_step5_csa_z_cache() {
    return true;
}

// Selection is OFFICIAL hard top-512-regions ALWAYS -- no soft gate, no n_kv-proportional
// scaling. Kept-slot capacity for the selection cutoff, SHARED by the fused (ggml_indexer_topk
// width) and decomposed (rank-penalty scatter) paths so the two cannot drift. BLOCK-grain ALWAYS:
// counts BLOCKS (topk x Bsel slots); whole-block snap; clamp at n_kv
// (== exact dense -- the decomposed clamp makes pen(rank)==0 for every key, the fused path
// short-circuits).
static int64_t ik_kept_slots(const llama_hparams & hparams, int64_t n_kv_sel) {
    const int64_t Bsel = hparams.indexer_csa_block > 0 ? hparams.indexer_csa_block : 8;
    int64_t kept = (int64_t) hparams.indexer_top_k * Bsel;
    kept = ((kept + Bsel - 1) / Bsel) * Bsel;  // whole-block snap
    if (kept > n_kv_sel) kept = n_kv_sel;
    return kept;
}

void llm_build_context::build_step5_indexer_kv_write(ggml_cgraph * gf, int il, ggml_tensor * inpL) {
    const int64_t head_size = hparams.indexer_head_size;

    ggml_tensor * kr_cache = (size_t) il < kv_self.kr_l.size() ? kv_self.kr_l[il] : nullptr;
    if (!kr_cache || !hparams.indexer_is_full[il]) {
        return;
    }
    GGML_ASSERT(head_size > 0);

    // The real proxy key = k_norm(k_proj(inpL)) over proxy_dim, cached RAW (no RoPE).
    // Ported inline from build_deepseek2_dsa_indexer (build_deepseek2.cpp:417).
    const auto & layer = model.layers[il];
    ggml_tensor * src;
    GGML_ASSERT(layer.indexer_k && layer.indexer_k_norm && "STEP5 indexer k tensors missing");
    src = ggml_mul_mat(ctx0, layer.indexer_k, inpL); // {head_size, n_tokens}
    src = step5_bf16_roundtrip(ctx0, src); // torch linear materializes bf16 output
    src = llm_build_norm(ctx0, src, hparams, layer.indexer_k_norm, layer.indexer_k_norm_b, LLM_NORM, cb, il);
    src = step5_bf16_roundtrip(ctx0, src); // torch norm materializes bf16 output
    cb(src, "step5_indexer_k_raw", il);
    GGML_ASSERT(src->ne[0] == head_size && src->ne[1] == n_tokens);

    // IK_HADAMARD whitening BEFORE the cache-type cast: the quantizer sees rotated values --
    // that is where the precision win lives. Row blocks of head_size (power of 2; ggml_hadamard
    // aborts otherwise -- IK_HADAMARD=0 for exotic toy dims).
    if (ik_hadamard_enabled()) {
        GGML_ASSERT(head_size > 1 && (head_size & ~(head_size - 1)) == head_size);
        src = ggml_hadamard(ctx0, src, (int) head_size);
        cb(src, "step5_indexer_k_rot", il);
    }

    src = ggml_cast(ctx0, ggml_cont(ctx0, src), kr_cache->type);
    cb(src, "step5_kr_plumb_src", il);

    ggml_tensor * kr_view = ggml_view_2d(ctx0, kr_cache, head_size, n_tokens,
            ggml_row_size(kr_cache->type, head_size),
            ggml_row_size(kr_cache->type, head_size) * kv_head);
    ggml_tensor * kr_cpy = ggml_cpy(ctx0, src, kr_view);
    cb(kr_cpy, "step5_kr_plumb_write", il);

    // graph-reuse fixup registration (dsa_cache_copies pattern): kr_view bakes kv_head at
    // build time; update_cache_copies() patches view_offs = kv_head*nb[1] on graph reuse.
    GGML_ASSERT((size_t) il < lctx.dsa_cache_copies.size());
    lctx.dsa_cache_copies[il].cpy  = kr_cpy;
    lctx.dsa_cache_copies[il].step = kr_cache->nb[1];

    ggml_build_forward_expand(gf, kr_cpy);

    // Also cache the RAW z projection per cell -- the softmax logit
    // of the region summary (kernel.py:283). No norm (csa_z_norm_type="none"), no RoPE, no
    // Hadamard: z is consumed only through exp(), and whitening it would be a (score-neutral
    // here, but semantics-divergent) transform the reference does not perform. Same
    // view/registration discipline as the kr write above (graph-reuse offset patch via
    // dsa_z_cache_copies; defrag row-move in llama-build-context.cpp).
    ggml_tensor * kz_cache = (size_t) il < kv_self.kz_l.size() ? kv_self.kz_l[il] : nullptr;
    if (kz_cache) {
        GGML_ASSERT(layer.indexer_z && "STEP5 indexer z tensor missing (official CSA z cache)");
        ggml_tensor * z_src = ggml_mul_mat(ctx0, layer.indexer_z, inpL); // {head_size, n_tokens}
        cb(z_src, "step5_indexer_z_cache_src", il);
        z_src = ggml_cast(ctx0, ggml_cont(ctx0, z_src), kz_cache->type);
        ggml_tensor * kz_view = ggml_view_2d(ctx0, kz_cache, head_size, n_tokens,
                ggml_row_size(kz_cache->type, head_size),
                ggml_row_size(kz_cache->type, head_size) * kv_head);
        ggml_tensor * kz_cpy = ggml_cpy(ctx0, z_src, kz_view);
        cb(kz_cpy, "step5_kz_plumb_write", il);

        GGML_ASSERT((size_t) il < lctx.dsa_z_cache_copies.size());
        lctx.dsa_z_cache_copies[il].cpy  = kz_cpy;
        lctx.dsa_z_cache_copies[il].step = kz_cache->nb[1];

        ggml_build_forward_expand(gf, kz_cpy);
    }
}

// Proxy forward (q/z/w proj + q_norm rmsnorm) + indexer-key read-back scoring.
// Scores = relu(q·k)·w summed over proxy heads, seeded with the causal KQ_mask -- the DSA
// precedent (build_deepseek2.cpp:488). Skipped on SWA layers / MTP tail (write and score guards
// mirror each other).
// Returns the causal-seeded score {n_kv, n_tokens} (F32) for the selection path,
// nullptr on layers without an indexer cache entry.
// Perf note: the cached span is cast to F32 for the pe RoPE; fine at gate contexts,
// revisit (quantized nope matmul + cast only the pe slice) before 1M-ctx decode work.
ggml_tensor * llm_build_context::build_step5_indexer_score(ggml_cgraph * gf, int il,
        ggml_tensor * inpL, ggml_tensor * inp_pos, ggml_tensor * KQ_mask,
        ggml_tensor * band01, ggml_tensor * band_neg, ggml_tensor ** fused_mask_out) {
    const int64_t head_size = hparams.indexer_head_size; // proxy_dim (HF sparse_config: 256)
    const int64_t n_ihead   = hparams.indexer_n_head;    // proxy q heads (HF num_heads: 16)

    ggml_tensor * kr_cache = (size_t) il < kv_self.kr_l.size() ? kv_self.kr_l[il] : nullptr;
    if (!kr_cache || !hparams.indexer_is_full[il]) {
        return nullptr;
    }
    GGML_ASSERT(head_size > 0 && n_ihead > 0);
    GGML_ASSERT(n_kv > 0 && n_kv <= (int64_t) kv_self.size);

    // ---- fused selection path decision (built BEFORE the heavy chains so the exact-dense
    // short-circuit can skip them entirely) ----
    // Active iff selection is on (IK_SPARSE=1) and the fused ops are enabled
    // (cparams.fused_idx_topk, CLI -fidx/-no-fidx, default on); NOT under tap capture
    // (cb_eval -> decomposed chain keeps the score taps byte-identical).
    const bool fused_try = fused_mask_out != nullptr && ik_sparse_enabled() &&
            lctx.cparams.fused_idx_topk && !lctx.cparams.cb_eval;
    const int64_t kept = fused_try ? ik_kept_slots(hparams, n_kv) : 0;
    if (fused_try && kept >= n_kv) {
        // kept == n_kv: selection keeps every key == EXACT DENSE (the decomposed path reaches
        // the same mask through the pen(rank)==0 clamp). Skip the whole indexer scoring chain;
        // the caller keeps the dense KQ_mask. Dense-equivalence holds by construction.
        return nullptr;
    }

    const auto & layer = model.layers[il];
    GGML_ASSERT(layer.indexer_q && layer.indexer_z && layer.indexer_w && layer.indexer_q_norm);

    // Debug tap: the layer input itself (view, no copy -- avoids renaming inpL which
    // may alias an earlier cb()-named tensor). Replays the whole indexer
    // chain from this tap + mini weights (build_step5_indexer_inpL).
    GGML_ASSERT(inpL->ne[2] == 1);
    ggml_tensor * inpL_view = ggml_view_2d(ctx0, inpL, inpL->ne[0], inpL->ne[1], inpL->nb[1], 0);
    cb(inpL_view, "step5_indexer_inpL", il);
    // ROOT only under tap capture (cb_eval): a rooted tensor is never consumed, so gallocr pins
    // it to graph end -- unrooted, this alias costs ZERO (nothing reads it). Unconditional rooting
    // pinned {n_embd, n_tokens} F32 x n_layer (measured: the dominant share of the 7382 MiB
    // reserve at kv 16128 / ub 4096).
    if (lctx.cparams.cb_eval) {
        ggml_build_forward_expand(gf, inpL_view);
    }

    // pe/nope split over indexer_rope_dim (HF sparse_indexer_rope_dim: 32); clamped so toy
    // tiny test dims (proxy_dim <= rope_dim) rope the whole proxy instead of building empty views.
    const int64_t rope_dim = std::min<int64_t>(
            hparams.indexer_rope_dim > 0 ? hparams.indexer_rope_dim : head_size, head_size);
    const int64_t nope_dim = head_size - rope_dim;

    // Cached-cell positions for the read-back pe RoPE (mirrors inp_dsa_sink: per-graph input,
    // host-filled in llama_set_inputs from kv_self.cells). RAW cached keys + this tensor are what
    // keeps the position-free cached-key invariant: cached rows depend on position only at read-back.
    // The fill runs right before graph compute and ggml_rope_ext reads it DURING compute, so the
    // positions are always fresh; after decode the compute buffer is clobbered (not observable
    // post-hoc).
    if (!lctx.inp_kv_pos) {
        lctx.inp_kv_pos = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_kv);
        cb(lctx.inp_kv_pos, "kv_pos", -1);
        ggml_set_input(lctx.inp_kv_pos);
        ggml_build_forward_expand(gf, lctx.inp_kv_pos); // match inp_dsa_sink (build_deepseek2.cpp:1274)
    }

    // ---- proxy q: rmsnorm(q_proj(inpL)) over proxy_dim per head, pe rope at query pos ----
    ggml_tensor * q_cur = ggml_mul_mat(ctx0, layer.indexer_q, inpL); // {head_size*n_ihead, n_tokens}
    q_cur = step5_bf16_roundtrip(ctx0, q_cur); // torch linear materializes bf16 output
    q_cur = ggml_view_3d(ctx0, q_cur, head_size, n_ihead, n_tokens,
            ggml_row_size(q_cur->type, head_size),
            ggml_row_size(q_cur->type, head_size) * n_ihead, 0);
    q_cur = llm_build_norm(ctx0, q_cur, hparams, layer.indexer_q_norm, nullptr, LLM_NORM_RMS, cb, il);
    q_cur = step5_bf16_roundtrip(ctx0, q_cur); // torch norm materializes bf16 output
    cb(q_cur, "step5_indexer_q_normed", il);

    ggml_tensor * q_pe = ggml_view_3d(ctx0, q_cur, rope_dim, n_ihead, n_tokens,
            ggml_row_size(q_cur->type, head_size),
            ggml_row_size(q_cur->type, head_size) * n_ihead, 0);
    q_pe = ggml_rope_ext(ctx0, q_pe, inp_pos, nullptr, rope_dim,
            rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow);
    if (nope_dim > 0) {
        ggml_tensor * q_nope = ggml_view_3d(ctx0, q_cur, nope_dim, n_ihead, n_tokens,
                ggml_row_size(q_cur->type, head_size),
                ggml_row_size(q_cur->type, head_size) * n_ihead,
                ggml_row_size(q_cur->type, rope_dim));
        q_cur = ggml_concat(ctx0, q_pe, q_nope, 0);
    } else {
        q_cur = q_pe;
    }
    cb(q_cur, "step5_indexer_q_cat", il);
    q_cur = step5_bf16_roundtrip(ctx0, q_cur); // torch rope materializes bf16 output
    // Official e4m3 activation round on the indexer q (kernel.py:546) -- every score
    // consumer below (decomposed score_chain AND the fused ggml_indexer_topk kernel) must
    // see the rounded q, exactly as the reference rounds index_q inside indexer_logits.
    q_cur = ggml_round_e4m3(ctx0, q_cur);
    cb(q_cur, "step5_indexer_q_e4m3", il);

    // ---- z/w projections (z ↔ csa_block_compress (z_norm_type=none),
    // w ≡ DSA per-head indexer weights) ----
    ggml_tensor * z_cur = ggml_mul_mat(ctx0, layer.indexer_z, inpL); // {head_size, n_tokens}
    z_cur = step5_bf16_roundtrip(ctx0, z_cur); // torch linear materializes bf16 output
    cb(z_cur, "step5_indexer_z_raw", il);
    // z is telemetry-only: unconsumed cb() tensors never enter the graph and cb_eval never
    // fires (inp_dsa_sink precedent) -- root it explicitly, but ONLY under tap capture
    // (unrooted z_cur would otherwise be pruned).
    if (lctx.cparams.cb_eval) {
        ggml_build_forward_expand(gf, z_cur);
    }

    ggml_tensor * w_cur = ggml_mul_mat(ctx0, layer.indexer_w, inpL); // {n_ihead, n_tokens}
    w_cur = step5_bf16_roundtrip(ctx0, w_cur); // torch linear materializes bf16 output
    w_cur = ggml_scale(ctx0, w_cur, 1.0f / sqrtf(float(head_size * n_ihead))); // DSA :488 port
    cb(w_cur, "step5_indexer_w", il);

    // ---- read back the cached RAW proxy keys ({head_size, n_kv}), rope pe at read-back ----
    ggml_tensor * cached_k = ggml_view_2d(ctx0, kr_cache, head_size, n_kv,
            ggml_row_size(kr_cache->type, head_size), 0);
    // IK_HADAMARD de-rotation (H^2 == I; ggml_hadamard outputs F32 so it slots in for the cast).
    // Result = the raw proxy key modulo cache quantization noise -> pe RoPE below sees the same
    // raw-domain key as IK_HADAMARD=0.
    ggml_tensor * k_f32;
    if (ik_hadamard_enabled()) {
        k_f32 = ggml_hadamard(ctx0, cached_k, (int) head_size);
        cb(k_f32, "step5_indexer_k_derot", il);
    } else {
        k_f32 = ggml_cast(ctx0, ggml_cont(ctx0, cached_k), GGML_TYPE_F32);
    }
    cb(k_f32, "step5_indexer_cached_k", il);

    ggml_tensor * k_pe = ggml_view_3d(ctx0, k_f32, rope_dim, 1, n_kv,
            ggml_row_size(k_f32->type, head_size),
            ggml_row_size(k_f32->type, head_size), 0);
    k_pe = ggml_rope_ext(ctx0, k_pe, lctx.inp_kv_pos, nullptr, rope_dim,
            rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow);
    ggml_tensor * indexer_k_b;
    if (nope_dim > 0) {
        ggml_tensor * k_nope = ggml_view_3d(ctx0, k_f32, nope_dim, 1, n_kv,
                ggml_row_size(k_f32->type, head_size),
                ggml_row_size(k_f32->type, head_size),
                ggml_row_size(k_f32->type, rope_dim));
        indexer_k_b = ggml_concat(ctx0, k_pe, k_nope, 0);
    } else {
        indexer_k_b = k_pe;
    }
    indexer_k_b = ggml_reshape_3d(ctx0, indexer_k_b, head_size, n_kv, 1);
    indexer_k_b = step5_bf16_roundtrip(ctx0, indexer_k_b); // torch rope materializes bf16 output
    cb(indexer_k_b, "step5_indexer_k_rope", il);

    // ---- scores: relu(q·k)·w summed over proxy heads (DSA :488 port) ----
    // Official semantics: scores are computed at BLOCK grain (pooled block keys, n_kv/B)
    // and broadcast per-block scores back to slots. The causal mask seed is added LAST at slot
    // granularity. Non-inplace add on purpose: the F32 mask seed aliases
    // inp_KQ_mask (-fa 0), which later softmax layers read back (build_deepseek2.cpp:540 caveat).
    ggml_tensor * mask_f32 = ggml_view_2d(ctx0, KQ_mask, n_kv, n_tokens, KQ_mask->nb[1], 0);
    if (mask_f32->type != GGML_TYPE_F32) {
        mask_f32 = ggml_cast(ctx0, mask_f32, GGML_TYPE_F32);
        cb(mask_f32, "step5_indexer_score_mask_f32", il);
    }

    // scoring chain reused VERBATIM for token keys or pooled block keys (n_key = n_kv / n_blocks)
    //
    // Compute-buffer shape discipline (build_deepseek2.cpp:513 two-branch precedent): the fused
    // {n_key, n_ihead*n_tokens} shape is deliberately capped at 8 tokens -- at llama-server
    // ubatches its kq/relu/cont/mul temporaries are EACH ~{n_key, 16*n_tokens} F32
    // (== 2 x kv_size*n_ubatch*4 B at CSA block grain) and dominate the CUDA compute buffer
    // (measured reserve law: ~29 x {kv,ub} F32 equivalents, kv_size*n_ubatch*117 B total).
    // Real ubatches run the math PER INDEXER HEAD instead (build_deepseek2.cpp:552 branch):
    // {n_key, n_tokens} F32 temporaries (~kv*n_ubatch/8 B), no transpose/cont anywhere
    // (broadcast mul keeps both operands contiguous), heads accumulated ascending.
    // NUMERIC NOTE: sequential add-accumulate over heads is NOT bitwise == the fused chain's
    // ggml_sum_rows reduction -- accepted drift on n_tokens>8 graphs only (ingest/reserve);
    // decode/MTP graphs (<=8 tokens) keep the fused path byte-identical. Per-head taps key on
    // il_cb = 1000*(il+1)+head (deepseek2 head-loop naming precedent).
    auto score_chain = [&](ggml_tensor * keys, int64_t n_key) {
        if (n_tokens <= 8) {
            ggml_tensor * q2d = ggml_reshape_2d(ctx0, q_cur, head_size, n_ihead * n_tokens);
            ggml_tensor * kq = ggml_mul_mat(ctx0, keys, q2d); // {n_key, n_ihead*n_tokens}
            cb(kq, "step5_indexer_kq", il);
            kq = ggml_relu(ctx0, kq);
            cb(kq, "step5_indexer_kq_relu", il);
            kq = ggml_reshape_3d(ctx0, kq, n_key, n_ihead, n_tokens);
            kq = ggml_cont(ctx0, ggml_transpose(ctx0, kq)); // {n_ihead, n_key, n_tokens}
            ggml_tensor * w3 = ggml_reshape_3d(ctx0, w_cur, n_ihead, 1, n_tokens);
            kq = ggml_mul(ctx0, kq, w3);
            cb(kq, "step5_indexer_kq_w", il);
            ggml_tensor * s = ggml_sum_rows(ctx0, kq); // {1, n_key, n_tokens}
            return ggml_reshape_2d(ctx0, s, n_key, n_tokens);
        }
        ggml_tensor * acc = nullptr;
        for (int64_t head = 0; head < n_ihead; ++head) {
            const int il_cb = 1000 * (il + 1) + (int) head;
            // [head_size, n_tokens]: per-head slice of contiguous q_cur {head_size, n_ihead,
            // n_tokens} -- nb1 stride between heads, nb2 stride between tokens (deepseek2 :555)
            ggml_tensor * q_h = ggml_view_2d(ctx0, q_cur, head_size, n_tokens,
                    ggml_row_size(q_cur->type, head_size) * n_ihead,
                    ggml_row_size(q_cur->type, head_size) * head);
            // [n_key, n_tokens]
            ggml_tensor * kq = ggml_mul_mat(ctx0, keys, q_h);
            cb(kq, "step5_indexer_kq", il_cb);
            kq = ggml_relu(ctx0, kq);
            cb(kq, "step5_indexer_kq_relu", il_cb);
            // [1, n_tokens] contiguous slice of w_cur {n_ihead, n_tokens}; cont because the CUDA
            // ggml_mul path wants contiguous tensors (build_deepseek2.cpp:520 caveat)
            ggml_tensor * w_h = ggml_cont(ctx0, ggml_view_2d(ctx0, w_cur, 1, n_tokens,
                    w_cur->nb[1], w_cur->nb[0] * head));
            kq = ggml_mul(ctx0, kq, w_h); // broadcast {n_key, n_tokens} x {1, n_tokens}
            cb(kq, "step5_indexer_kq_w", il_cb);
            acc = acc ? ggml_add_inplace(ctx0, acc, kq) : kq;
        }
        return acc; // {n_key, n_tokens}
    };

    ggml_tensor * slot_score;
    const int64_t B = hparams.indexer_csa_block > 0 ? hparams.indexer_csa_block : 8;
    // official BLOCK-grain CSA ALWAYS: pooled block
    // keys per the official softmax(z)-weighted summary, scores at block grain, broadcast back
    // to slots. Slot-aligned reshape pooling needs contiguous b*B..b*B+B-1 groups (==
    // position-aligned blocks in steady single-sequence decode; multi-sequence/defrag
    // alignment TODO).
    {
        GGML_ASSERT(B > 0 && n_kv % B == 0 && "CSA slot-aligned blocks need n_kv divisible by B");
        const int64_t n_blocks = n_kv / B;

        // validity weights v[slot] = 1 iff visible to SOME query of this ubatch: step(mask+0.5)
        // per (slot,query) turns {0,-inf} into {1,0}, summed over query columns and re-stepped.
        // Empty padded cells carry allocator garbage kr_l rows -- zero-weighted OUT of the pool
        // so their garbage cannot contaminate block means (and 0*garbage stays a finite 0).
        ggml_tensor * vis = ggml_step(ctx0, ggml_scale_bias(ctx0, mask_f32, 1.0f, 0.5f));
        vis = ggml_sum_rows_ext(ctx0, vis, 1);                       // {n_kv, 1}
        vis = ggml_step(ctx0, ggml_scale_bias(ctx0, vis, 1.0f, -0.5f));
        vis = ggml_reshape_2d(ctx0, vis, n_kv, 1);
        cb(vis, "step5_indexer_slot_valid", il);
        ggml_tensor * block_k;
        {
            // ---- OFFICIAL: per-dimension softmax(z)-weighted mean (kernel.py:283/:301) ----
            // summary_{b,d} = sum_t softmax_t(z_{t,d}) * k_{t,d} over the region's tokens:
            // an element-wise weighted mean, NOT a convex combination of whole key vectors.
            // The softmax normalizes over tokens WITHIN each (region, dim) only -- one Triton
            // program per (region, head) in the reference, no cross-head/region reduction; the
            // ggml chain below reduces exactly that one axis. softmax's internal row-max == the
            // kernel's shared scalar shift up to softmax invariance, and sum(softmax) == 1
            // exactly, so the kernel's denominator division drops out.
            ggml_tensor * kz_cache = (size_t) il < kv_self.kz_l.size() ? kv_self.kz_l[il] : nullptr;
            GGML_ASSERT(kz_cache && "official CSA needs the kz_l z cache (llama_step5_csa_z_cache)");
            // ---- FUSED: ggml_csa_compress op -- reads the kz_l view + indexer_k_b
            // directly (kills the permute/cont/masked/softmax/mul/sum_rows chain), one
            // (region, dim) thread, BF16->E4M3 double-round epilogue per kernel.py:345.
            // Fallbacks (backend capability, not modes): Q8_0 z (-ictk q8_0) and
            // supports_op failure ride the decomposed cast path below.
            if (kz_cache->type == GGML_TYPE_F16 || kz_cache->type == GGML_TYPE_F32) {
                ggml_tensor * z_view = ggml_view_2d(ctx0, kz_cache, head_size, n_kv,
                        ggml_row_size(kz_cache->type, head_size), 0);
                // tap on the fused path too (decomposed branch dumps the same;
                // without it the softmax(z) summary is unreplayable from taps alone).
                ggml_tensor * z_tap = ggml_cast(ctx0, ggml_cont(ctx0, z_view), GGML_TYPE_F32);
                cb(z_tap, "step5_indexer_cached_z", il);
                if (lctx.cparams.cb_eval) {
                    ggml_build_forward_expand(gf, z_tap); // unconsumed tap: root or it is pruned
                }
                block_k = ggml_csa_compress(ctx0, z_view, indexer_k_b, vis, (int) B, 1);
                cb(block_k, "step5_indexer_block_k", il);
                if (supports_op(block_k)) {
                    ggml_build_forward_expand(gf, block_k);
                } else {
                    // backend cannot run the fused op: fall through to the decomposed chain
                    // (deepseek2.cpp:506 supports_op fallback precedent) -- CPU-only runs etc.
                    block_k = nullptr;
                }
            } else {
                block_k = nullptr; // Q8_0 z: decomposed path only
            }
            if (!block_k) {
                ggml_tensor * z_f32 = ggml_view_2d(ctx0, kz_cache, head_size, n_kv,
                        ggml_row_size(kz_cache->type, head_size), 0);
                z_f32 = ggml_cast(ctx0, ggml_cont(ctx0, z_f32), GGML_TYPE_F32); // {head_size, n_kv}
                cb(z_f32, "step5_indexer_cached_z", il);

            // Common layout for the element-wise pairing: axis0 = token-within-region (the
            // softmax AND the sum axis), axis1 = dim, axis2 = region. z {head,n_kv} flat =
            // d + head*t -> reshape {head,B,n_blocks} (d,r,b) -> permute (r,d,b) -> cont.
            // k side identical off indexer_k_b {head,n_kv,1}. ggml_sum_rows reduces ne[0].
            ggml_tensor * z_pv = ggml_cont(ctx0, ggml_permute(ctx0,
                    ggml_reshape_3d(ctx0, z_f32, head_size, B, n_blocks), 1, 0, 2, 3));
            cb(z_pv, "step5_indexer_csa_z_blocks", il);
            ggml_tensor * k_pv = ggml_cont(ctx0, ggml_permute(ctx0,
                    ggml_reshape_3d(ctx0, indexer_k_b, head_size, B, n_blocks), 1, 0, 2, 3));
            cb(k_pv, "step5_indexer_csa_k_blocks", il);

            // Padded/invalid slots: push z to -INF pre-softmax (w == exp(-inf) == 0) so their
            // garbage cache rows are zero-weighted out of the summary -- the graph-level analog
            // of the kernel's count==0 region skip (kernel.py:355). vis {n_kv,1} reshapes to
            // {B,1,n_blocks} (flat r + B*b == t = b*B+r) and broadcasts over the dim axis.
            ggml_tensor * vis_v3 = ggml_reshape_3d(ctx0, vis, B, 1, n_blocks);
            ggml_tensor * z_masked = ggml_add(ctx0, z_pv,
                    ggml_scale_bias(ctx0, vis_v3, 1e30f, -1e30f));
            cb(z_masked, "step5_indexer_csa_z_masked", il);

            ggml_tensor * w = ggml_soft_max(ctx0, z_masked); // {B, head_size, n_blocks}
            cb(w, "step5_indexer_csa_w", il);
            ggml_tensor * num = ggml_sum_rows(ctx0, ggml_mul(ctx0, w, k_pv)); // {1, head, n_blocks}
            block_k = ggml_reshape_2d(ctx0, num, head_size, n_blocks);
            cb(block_k, "step5_indexer_block_k", il);
            }
        }

        // The reference scores against the e4m3-rounded summary (cache.summary is fp8;
        // indexer_logits rounds it again -- an identity on already-quantized values, which is
        // what the fused round=1 summary already is). The raw k that fed csa_compress /
        // the decomposed chain above stays unrounded.
        block_k = ggml_round_e4m3(ctx0, block_k);
        cb(block_k, "step5_indexer_block_k_e4m3", il);

        // ---- FUSED selection path: block keys + cell_blk expand + ggml_indexer_topk +
        // ggml_indexer_mask (qwen4exp.cpp:419 wiring precedent). The topk op scores and sorts
        // IN-KERNEL (relu(q·k)·w summed over proxy heads + causal mask seed + descending sort;
        // the {n_kv, n_tokens} score matrix NEVER materializes in the compute buffer) and the
        // mask op writes the final attention mask in one pass (topk cells 0, every other cell
        // -INF cliff (hard selection), dense mask added on top -> future/padding stay -INF). Together they
        // replace score_chain + slot broadcast + argsort + rank-penalty scatter + fa_mask
        // adapter -- the ~25 F32 {kv,ub} tensor-equivalents that dominate the sparse-mode
        // compute buffer collapse to the pooling chain (O(kv/8)) + one mask-sized output.
        // Semantics == decomposed: same pooled block keys, same w-weighted scores (F32 k takes
        // the F32 GEMM branch -- no FP16 rounding of the scores), same kept capacity
        // (ik_kept_slots), hard -INF cliff; the CUB radix sort is
        // STABLE, so intra-block score ties stay contiguous in rank order and the kept cutoff
        // (a multiple of B) still snaps to whole blocks.
        if (fused_try) {
            // causal seed: RAW KQ_mask view (F16 on -fa 1 / F32 on -fa 0 -- the op templates
            // both; no F32 cast needed here, unlike the decomposed add below). The own-region
            // band_neg (-BIG on the query token's own-region causal cells) EXCLUDES the own region from
            // top-k competition -- official semantics (kernel.py top-k over regions EXCLUDING
            // own, then _append_current_region_kernel). Only the topk seed sees the band: the
            // mask op below gets the RAW KQ_mask + the band01 force map.
            ggml_tensor * mseed = ggml_view_2d(ctx0, KQ_mask, n_kv, n_tokens, KQ_mask->nb[1], 0);
            if (band_neg) {
                mseed = ggml_add(ctx0, mseed, ggml_view_2d(ctx0, band_neg, n_kv, n_tokens, band_neg->nb[1], 0));
            }
            if (!lctx.inp_step5_cell_blk) {
                // cell -> block map, slot-aligned i/B (same grouping as the pooling reshape
                // above); host-filled in llama_set_inputs. One shared input per graph -- every
                // full-attn layer consumes the same map (inp_kv_pos lazy-create pattern).
                lctx.inp_step5_cell_blk = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_kv);
                cb(lctx.inp_step5_cell_blk, "step5_cell_blk", -1);
                ggml_set_input(lctx.inp_step5_cell_blk);
                ggml_build_forward_expand(gf, lctx.inp_step5_cell_blk);
            }
            // own-region force-append: band01 (1.0 on the own-region causal cells, derived
            // graph-side from the dense causal mask) force-unmasks them even when the
            // exclusion above kept them out of the top-k -- kernel.py
            // _append_current_region_kernel. Kept capacity stays topk x B; the own region
            // rides on top (the reference's topk+1 packed slots).
            ggml_tensor * topk = ggml_indexer_topk(ctx0, block_k, q_cur, w_cur, mseed,
                    lctx.inp_step5_cell_blk, GGML_UNARY_OP_RELU, (int) kept);
            if (supports_op(topk)) {
                cb(topk, "step5_indexer_topk", il);
                ggml_build_forward_expand(gf, topk);
                // dup of KQ_mask: F16 {n_kv, padded n_tok} on -fa 1 (FA-ready, no adapter),
                // F32 {n_kv, n_tok} additive on -fa 0. pen = 0 -> hard -INF cliff.
                ggml_tensor * fm = ggml_indexer_mask_own(ctx0, KQ_mask, topk, band01, 0.0f);
                cb(fm, "step5_fused_mask", il);
                ggml_build_forward_expand(gf, fm);
                *fused_mask_out = fm;
                return nullptr; // no decomposed score tensor -- caller consumes *fused_mask_out
            }
            // backend cannot run the fused op: fall through to the decomposed chain below
            // (deepseek2.cpp:506 supports_op fallback precedent)
        }

        ggml_tensor * block_score = score_chain(block_k, n_blocks); // {n_blocks, n_tokens}
        cb(block_score, "step5_indexer_block_score", il);

        // broadcast to slots: slot b*B+r takes block b's score. repeat alone does it: rep
        // {B,n_blocks,n_tokens} with rep[r,b,j] = block_score[b,j] flattens col-major to
        // linear r + B*b (+ B*n_blocks*j) -- reshaped {n_kv,n_tokens} that is exactly slot
        // i = b*B+r <- block b (the inverse of the pooling reshape's slot map). A permute
        // here TRANSPOSES the map (slot i <- block i%n_blocks -- bitwise-verified against
        // the score tap via the broadcast check); it must NOT run.
        ggml_tensor * bs3 = ggml_reshape_3d(ctx0, block_score, 1, n_blocks, n_tokens);
        ggml_tensor * rep = ggml_repeat(ctx0, bs3,
                ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, B, n_blocks, n_tokens));
        slot_score = ggml_reshape_2d(ctx0, ggml_cont(ctx0, rep), n_kv, n_tokens);
    }

    ggml_tensor * indexer_score = ggml_add(ctx0, mask_f32, slot_score);
    cb(indexer_score, "step5_indexer_score", il);
    // ROOT only under tap capture: with IK_SPARSE=1 the argsort consumes indexer_score (expand
    // redundant), with IK_SPARSE=0 it is telemetry-only -- unconditional rooting pinned one
    // {n_kv, n_tokens} F32 PER LAYER to graph end (23 layers x 1 equiv ~= 6 GiB at typical
    // large-context configs).
    if (lctx.cparams.cb_eval) {
        ggml_build_forward_expand(gf, indexer_score);
    }
    return indexer_score;
}

ggml_cgraph * llm_build_context::build_step5() {
    ggml_cgraph * gf = new_graph_custom();
    ggml_tensor * cur;
    auto inp_pos     = build_inp_pos();

    // indexer-write registrations are per-graph: stale entries from a previous build must not
    // be patched against a graph that no longer contains them (build_openpangu.cpp precedent).
    std::fill(lctx.dsa_cache_copies.begin(), lctx.dsa_cache_copies.end(), llama_context::CacheCopy{});

    if (cparams.mtp_op_type != MTP_OP_NONE) {
        GGML_ASSERT(model.mtp && hparams.nextn_predict_layers > 0);
        GGML_ASSERT(batch.token && "Step5 MTP requires token batches");

        const int n_layer_base = hparams.n_layer > hparams.nextn_predict_layers
            ? hparams.n_layer - hparams.nextn_predict_layers : hparams.n_layer;
        const int n_heads_model = (int) hparams.nextn_predict_layers;
        const int n_heads = lctx.mtp_n_heads > 0
            ? std::max(1, std::min((int) lctx.mtp_n_heads, n_heads_model)) : n_heads_model;
        const int step = std::max(0, std::min((int) lctx.mtp_step_idx, n_heads - 1));
        const int il = n_layer_base + step;

        ggml_tensor * hidden_states = build_inp_mtp_states(n_embd);

        const bool step_independent_warmup = llm_arch_is_step5(model.arch) &&
            (cparams.mtp_op_type == MTP_OP_WARMUP ||
             cparams.mtp_op_type == MTP_OP_UPDATE_ACCEPTED) && n_heads > 1;
        if (step_independent_warmup) {
            for (int i = n_heads - 1; i >= 0; --i) {
                const int head_il = n_layer_base + i;
                const bool is_first = i == 0;
                const bool emit_logits = is_first && cparams.mtp_op_type == MTP_OP_UPDATE_ACCEPTED;
                cur = build_step5_mtp(model.layers[head_il], hidden_states, gf, inp_pos,
                        /*reduce_output=*/is_first, emit_logits);
                ggml_build_forward_expand(gf, cur);
            }
            return gf;
        }

        const bool reduce_mtp_output = cparams.mtp_op_type != MTP_OP_NONE;
        const bool emit_mtp_logits = cparams.mtp_op_type == MTP_OP_DRAFT_GEN ||
            cparams.mtp_op_type == MTP_OP_UPDATE_ACCEPTED;
        cur = build_step5_mtp(model.layers[il], hidden_states, gf, inp_pos,
                reduce_mtp_output, emit_mtp_logits);
        ggml_build_forward_expand(gf, cur);
        return gf;
    }

    auto inpL        = llm_build_inp_embd(ctx0, lctx, hparams, batch, model.tok_embd, cb);
    auto inp_out_ids = build_inp_out_ids();
    auto KQ_mask     = build_inp_KQ_mask();
    // --swa-compress: compacted sliding layers store KV rows in the [sinks|window] layout and
    // llm_build_kv reads lctx.swa_window_view (set inside build_swa_mask_for_graph) for the kv
    // view/store head -- MIMO2/DeepSeek4 consumption pattern. All main-loop sliding layers share
    // one compacted row count (kv-cache assert), so one any_compacted() check picks the mask;
    // the MTP tail layers are NEVER compacted (llama_kv_layer_rows keeps the nextn tail at
    // kv_size) and are matched per-layer in build_step5_mtp. Indexer/full-attn layers stay
    // uncompacted => kr_l rows, inp_kv_pos and the IK_CSA validity/block machinery are
    // untouched by compaction.
    auto KQ_mask_swa = kv_self.any_compacted()
        ? build_swa_mask_for_graph(hparams.n_swa, true)
        : build_inp_KQ_mask_swa();
    //const float kq_scale = 1.0f / sqrtf(float(n_rot));
    const float kq_scale = 1.0f / sqrtf(float(n_embd_head_k));

    // Shared own-region band (computed ONCE per graph, consumed by every full-attn
    // indexer layer): band01 = 1.0 on the query token's own-region VISIBLE causal cells,
    // 0 elsewhere; band_neg = -BIG there (fused topk competition-exclusion seed). Own cell
    // per column = the MAX VISIBLE cell of the dense causal mask column (same-seq cache
    // cells are pos-ordered, so the last visible cell is the token's own kv cell), and the
    // own region [B*floor(own/B), own] == exactly the VISIBLE cells of the LAST block that
    // contains a visible cell (any visible cell there is <= own and >= its block start).
    // Computed WITHOUT argmax (argmax emits I32; I32->F32 dup is unsupported on both
    // backends -- ggml.c dup ABORT on those):
    //   vis     = step(causal + 1)             1 on visible, 0 on masked/padded
    //   bvis_b  = step(block_sum - 0.5)        block contains a visible cell
    //   rb_b    = nbvis - cumsum_b + bvis_b    visible blocks in [b, nb)
    //   lastb_b = step(1.5 - rb_b)             b is the LAST visible block
    //   band01  = vis * expand_b(lastb_b)      per-cell own-region map (block k -> cells
    //                                           [k*B, (k+1)*B): reshape {nb,1,pad} -> repeat
    //                                           to {nb,B,pad} -> permute {B,nb,pad} -> cont+reshape
    //                                           {n_kv,pad}, giving flat index t + B*k == cell i)
    // Padded mask columns (all -INF): vis == 0 everywhere => band01 == 0 (cleaner than an
    // argmax-on-all--INF formulation, which pinned a spurious own=0 cell). All ops F32,
    // CPU + CUDA native. B pow2 (asserted; block reshape alignment).
    ggml_tensor * band01 = nullptr;
    ggml_tensor * band_neg = nullptr;
    if (ik_sparse_enabled() && hparams.indexer_csa_block > 0) {
        const int64_t Bsel = hparams.indexer_csa_block;
        GGML_ASSERT((Bsel & (Bsel - 1)) == 0);  // pow2 B
        const int64_t n_kv_b  = KQ_mask->ne[0];
        const int64_t n_pad_b = KQ_mask->ne[1];
        GGML_ASSERT(n_kv_b % Bsel == 0);
        const int64_t nb_b = n_kv_b / Bsel;
        ggml_tensor * causal_b = ggml_view_2d(ctx0, KQ_mask, n_kv_b, n_pad_b, KQ_mask->nb[1], 0);
        if (causal_b->type != GGML_TYPE_F32) causal_b = ggml_cast(ctx0, ggml_cont(ctx0, causal_b), GGML_TYPE_F32);
        ggml_tensor * vis = ggml_step(ctx0, ggml_scale_bias(ctx0, causal_b, 1.0f, 1.0f));
        ggml_tensor * bsum = ggml_reshape_2d(ctx0,
            ggml_sum_rows(ctx0, ggml_reshape_3d(ctx0, vis, Bsel, nb_b, n_pad_b)), nb_b, n_pad_b);
        ggml_tensor * bvis = ggml_step(ctx0, ggml_scale_bias(ctx0, bsum, 1.0f, -0.5f));
        ggml_tensor * cbb  = ggml_cumsum(ctx0, bvis);
        ggml_tensor * nbvis = ggml_view_2d(ctx0, cbb, 1, n_pad_b, cbb->nb[1],
            (nb_b - 1)*cbb->nb[0]); // last cumsum element per column (nb-1 along dim0)
        ggml_tensor * rb = ggml_add(ctx0,
            ggml_sub(ctx0, ggml_repeat(ctx0, nbvis, bvis), cbb), bvis);
        // (rb == 1) via step(1.5 - rb); rb == 0 (trailing empty blocks) also passes here but
        // vis == 0 on every cell of an empty block, so the band contribution is still 0.
        ggml_tensor * lastb = ggml_step(ctx0, ggml_scale_bias(ctx0, rb, -1.0f, 1.5f));
        ggml_tensor * lastb_3d = ggml_reshape_3d(ctx0, lastb, nb_b, 1, n_pad_b);
        ggml_tensor * lastb_rep = ggml_repeat(ctx0, lastb_3d,
            ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, nb_b, Bsel, n_pad_b));
        band01 = ggml_mul(ctx0, vis, ggml_reshape_2d(ctx0,
            ggml_cont(ctx0, ggml_permute(ctx0, lastb_rep, 1, 0, 2, 3)), n_kv_b, n_pad_b));
        cb(band01, "step5_own_region_band", -1);
        band_neg = ggml_scale(ctx0, band01, -1e30f);
        // fused-path seed add needs matching types with the raw KQ_mask view (F16 on -fa 1);
        // -1e30 saturates to F16 -inf, which is the intended -BIG exclusion value there.
        if (KQ_mask->type != GGML_TYPE_F32) band_neg = ggml_cast(ctx0, band_neg, KQ_mask->type);
        cb(band01, "step5_own_region_band", -1);
    }

    const int n_layer_base = hparams.n_layer > hparams.nextn_predict_layers
        ? hparams.n_layer - hparams.nextn_predict_layers : hparams.n_layer;

    // Shared destination leaf for the per-layer sparse-mask scatter (build_deepseek2_dsa_sparse_mask
    // base_leaf param). A fresh leaf per layer is pinned by gallocr from graph start (leaves are
    // allocated up-front there), costing n_full_layers x {n_kv, n_tokens} F32 ~= 6 GiB at
    // kv 16128 / ub 4096 even though each layer's consumer runs in the same layer — measured
    // linear growth in leaf-liveness tests + the persistent sparse-mode compute buffer.
    // One shared leaf = 1 equiv persistent; graph order serializes the fills (fill fully
    // overwrites before set_rows reads). Created lazily: only the sparse path consumes it.
    ggml_tensor * sparse_base_leaf = nullptr;

    for (int il = 0; il < n_layer_base; ++il) {
        bool is_swa = hparams.swa_layers[il];
        auto & layer = const_cast<llama_layer&>(model.layers[il]);

        ggml_tensor * rope_factors = nullptr;
        const uint32_t apply_mask = hparams.rope_scaling_apply_mask;
        if ((is_swa && (apply_mask & 0x2)) || (!is_swa && (apply_mask & 0x1))) {
            rope_factors = build_rope_factors(il);
        }
        auto rope_freqs = layer.rope_freqs;
        layer.rope_freqs = nullptr;
        // Indexer plumbing: proxy-key cache write (real proxy key: k_norm(k_proj), RAW) +
        // read-back scoring.
        // fused_mask: set when the fused selection path ran (in-kernel score/sort + mask write;
        // build_step5_indexer_score) -- see the selection comment below.
        build_step5_indexer_kv_write(gf, il, inpL);
        ggml_tensor * fused_mask = nullptr;
        ggml_tensor * idx_score = build_step5_indexer_score(gf, il, inpL, inp_pos, KQ_mask, band01, band_neg, &fused_mask);

        // ---- Selection (mask-based sparse attention FIRST) ----
        // Two equivalent implementations, same semantics (same pooled keys, same kept capacity
        // ik_kept_slots, hard -INF cliff):
        //   FUSED (default under IK_SPARSE=1 + -fidx + no tap capture): ggml_indexer_topk +
        //     ggml_indexer_mask inside build_step5_indexer_score -- no {n_kv, n_tokens} score
        //     matrix, no argsort/scatter chain, no fa_mask adapter; the compute buffer collapses
        //     from ~29 to ~4 F32 {kv,ub} equivalents (pooling + vis + mask output).
        //   DECOMPOSED (cb_eval tap capture, -no-fidx, token-grain anchor, or backend without
        //     the fused ops): full descending argsort over the n_kv axis -> rank-penalty scatter
        //     mask (build_deepseek2_dsa_sparse_mask; the --dsa-top-k/-dsatk kept-key-count
        //     characterization flag applies here too) -> FA adapter (..._dsa_fa_mask) when -fa 1
        //     needs the F16 padded contiguous mask shape. ALWAYS built under IK_SPARSE=1,
        //     deliberately NO n_kv <= topk short-circuit: at ctx <= topk the clamp makes
        //     pen(rank)==0 for every key, so the mask == the causal mask BITWISE and logits ==
        //     IK_SPARSE=0 dense EXACTLY -- the dense-equivalence property therefore exercises
        //     the whole argsort/scatter path instead of bypassing it (the fused path reaches the
        //     same exact-dense result via its kept==n_kv short-circuit).
        // KV-row gather (computing attention over kept rows only) stays on the 1M-ctx decode
        // TODO list -- both paths are mask-only today.
        ggml_tensor * attn_mask = is_swa ? KQ_mask_swa : KQ_mask;
        if (!is_swa && ik_sparse_enabled()) {
            if (fused_mask) {
                // dup of the dense KQ_mask shape/type (F16 padded on -fa 1, F32 on -fa 0):
                // consumed directly on BOTH attention paths, no adapter.
                attn_mask = fused_mask;
            } else if (idx_score) {
            ggml_tensor * sorted = ggml_argsort(ctx0, idx_score, GGML_SORT_ORDER_DESC);
            cb(sorted, "step5_indexer_sorted", il);
            // Block-grain selection: topk counts BLOCKS (sparse_config topk=512 x region_block_size=8
            // = 4096 kept slots). Kept-count must be a multiple of B so the rank cutoff snaps to
            // whole blocks (intra-block broadcast scores tie contiguously in the argsort).
            // kept capacity: ik_kept_slots (SHARED with the fused path -- fixed topk x Bsel
            // floor; no n_kv-proportional scaling).
            const int64_t kept_slots = ik_kept_slots(hparams, KQ_mask->ne[0]);
            if (!sparse_base_leaf) {
                sparse_base_leaf = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, 1, KQ_mask->ne[0], idx_score->ne[1]);
            }
            ggml_tensor * sparse = build_deepseek2_dsa_sparse_mask(sorted, KQ_mask, kept_slots, sparse_base_leaf);
            // own-region force-append: zero the mask on the query's own-region causal cells
            // (shared band01, derived graph-side from the dense causal mask: own cell = max
            // visible cell of the column, region = [B*floor(own/B), own]). sparse * (1 - band)
            // -- 1 - t is exactly 0/1, so no -INF*0 NaN and cells outside the band (incl. -INF
            // future/padding) pass through untouched; band cells are causal-visible so their
            // sparse value is 0 or finite -BIG. NOTE (documented compromise): the decomposed
            // argsort does NOT exclude the own region from competition (the fused path does,
            // via band_neg on its causal seed) -- when the own region ranks inside the kept
            // window one competing region is displaced vs the fused path.
            {
                const int64_t n_tok_sel = idx_score->ne[1];
                ggml_tensor * band_v = ggml_view_2d(ctx0, band01, KQ_mask->ne[0], n_tok_sel, band01->nb[1], 0);
                sparse = ggml_mul(ctx0, sparse, ggml_scale_bias(ctx0, band_v, -1.0f, 1.0f));
            }
            cb(sparse, "step5_sparse_mask", il);
            if (flash_attn) {
                sparse = build_deepseek2_dsa_fa_mask(sparse, KQ_mask);
                cb(sparse, "step5_sparse_mask_fa", il);
            }
            attn_mask = sparse;
            }
            // both null (fused kept>=n_kv short-circuit): attn_mask stays the dense KQ_mask
            // == exact dense semantics
        }
        // ssmax scale dropped (reference zero-init vestige) -- no inp_attn_scale.
        cur = build_std_attention(gf, model.layers[il].attn_norm, inpL,
                inp_pos, il == n_layer_base - 1 && n_tokens > 1 && !cparams.mtp ? inp_out_ids : nullptr,
                rope_factors, attn_mask, nullptr, nullptr, kq_scale, 0.0f, is_swa ? hparams.n_swa : 0,
                il, true, false, true);
        layer.rope_freqs = rope_freqs;

        if (model.layers[il].ffn_gate_inp == nullptr) {
            // dense FFN
            cur = llm_build_ffn(ctx0, lctx, model.layers[il].ffn_norm, cur,
                    model.layers[il].ffn_up,   NULL, NULL,
                    model.layers[il].ffn_gate, NULL, NULL,
                    model.layers[il].ffn_down, NULL, NULL,
                    nullptr,
                    LLM_FFN_SILU, LLM_FFN_PAR, cb, il, gf, true);
            cb(cur, "ffn_out", il);
        } else {
            const bool  norm_w  = hparams.expert_weights_norm;
            const float w_scale = hparams.expert_weights_scale;
            const bool  scale_w = w_scale != 0.0f;
            cur = llm_build_std_moe_ffn(ctx0, lctx, model.layers[il].ffn_norm, cur,
                    model.layers[il].ffn_gate_inp,  model.layers[il].ffn_gate_inp_b,
                    model.layers[il].ffn_up_exps,   model.layers[il].ffn_up_exps_b,
                    model.layers[il].ffn_gate_exps, model.layers[il].ffn_gate_exps_b,
                    model.layers[il].ffn_down_exps, model.layers[il].ffn_down_exps_b,
                    model.layers[il].ffn_exp_probs_b,
                    model.layers[il].ffn_up_shexp,    nullptr, // we don't have shared expert biases?
                    model.layers[il].ffn_gate_shexp,  nullptr,
                    model.layers[il].ffn_down_shexp,  nullptr,
                    n_expert, n_expert_used,
                    LLM_FFN_SILU, norm_w, scale_w, w_scale,
                    LLM_EXPERT_GATING_FUNC_SIGMOID,
                    //(llm_expert_gating_func_type) hparams.expert_gating_func,
                    LLM_FFN_SILU, cb, il, gf, true, model.layers[il].ffn_up_gate_exps);
        }

        cur = lctx.cvec.apply_to(ctx0, cur, il);
        cb(cur, "l_out", il);

        inpL = cur;
    }

    if (cparams.mtp) {
        ggml_tensor * mtp_embd = inpL->type == GGML_TYPE_F32 ? inpL : ggml_cast(ctx0, inpL, GGML_TYPE_F32);
        cb(mtp_embd, "result_mtp_embd", -1);
        ggml_set_output(mtp_embd);
        ggml_build_forward_expand(gf, mtp_embd);

        if (inp_out_ids) {
            inpL = ggml_get_rows(ctx0, inpL, inp_out_ids);
        }
    }

    cur = build_output(lctx, ctx0, inpL, model.output, model.output_norm, cb);
    cb(cur, "result_output", -1);

    ggml_build_forward_expand(gf, cur);

    return gf;
}

ggml_tensor * llm_build_context::build_step5_mtp(
        const llama_layer & mtp_layer,
        ggml_tensor * hidden_states_from_main_model,
        ggml_cgraph * gf,
        ggml_tensor * inp_pos,
        bool reduce_output,
        bool emit_logits,
        ggml_tensor ** hidden_out) {
    const int il = (int) (&mtp_layer - model.layers.data());

    GGML_ASSERT(mtp_layer.nextn.eh_proj && mtp_layer.nextn.enorm && mtp_layer.nextn.hnorm);
    GGML_ASSERT(mtp_layer.wq && mtp_layer.wk && mtp_layer.wv && mtp_layer.wo);

    ggml_tensor * inp_out_ids = (n_tokens > 1 && n_outputs < n_tokens) ? build_inp_out_ids() : nullptr;
    ggml_tensor * tok_embd_w = mtp_layer.nextn.embed_tokens ? mtp_layer.nextn.embed_tokens : model.tok_embd;
    ggml_tensor * tok_embd = build_inp_embd_mtp(tok_embd_w);
    ggml_tensor * cur = build_mtp_input(mtp_layer, hidden_states_from_main_model,
            tok_embd, il, "mtp_eh_proj");

    const bool is_swa = hparams.swa_layers[il];
    ggml_tensor * rope_factors = nullptr;
    const uint32_t apply_mask = hparams.rope_scaling_apply_mask;
    if ((is_swa && (apply_mask & 0x2)) || (!is_swa && (apply_mask & 0x1))) {
        rope_factors = build_rope_factors(il);
    }
    // MTP tail layers are excluded from compaction (llama_kv_layer_rows nextn guard), but keep
    // the check per-layer so a future layout change stays correct (DeepSeek4 pattern).
    auto KQ_mask = is_swa
        ? (kv_self.is_compacted((int) il) ? build_swa_mask_for_graph(hparams.n_swa, true) : build_inp_KQ_mask_swa())
        : build_inp_KQ_mask();
    const float kq_scale = 1.0f / sqrtf(float(hparams.n_embd_head_k(il)));

    cur = build_std_attention(gf, mtp_layer.attn_norm, cur, inp_pos, nullptr,
            rope_factors, KQ_mask, nullptr, nullptr, kq_scale, 0.0f,
            is_swa ? hparams.n_swa : 0, il, true, false, true, false, false, nullptr, il);

    if (mtp_layer.ffn_gate_inp == nullptr) {
        cur = llm_build_ffn(ctx0, lctx, mtp_layer.ffn_norm, cur,
                mtp_layer.ffn_up, nullptr, nullptr,
                mtp_layer.ffn_gate, nullptr, nullptr,
                mtp_layer.ffn_down, nullptr, nullptr,
                nullptr, LLM_FFN_SILU, LLM_FFN_PAR, cb, il, gf, true);
    } else {
        cur = llm_build_std_moe_ffn(ctx0, lctx, mtp_layer.ffn_norm, cur,
                mtp_layer.ffn_gate_inp, nullptr,
                mtp_layer.ffn_up_exps, nullptr,
                mtp_layer.ffn_gate_exps, nullptr,
                mtp_layer.ffn_down_exps, nullptr,
                mtp_layer.ffn_exp_probs_b,
                mtp_layer.ffn_up_shexp, nullptr,
                mtp_layer.ffn_gate_shexp, nullptr,
                mtp_layer.ffn_down_shexp, nullptr,
                n_expert, n_expert_used,
                LLM_FFN_SILU, hparams.expert_weights_norm, hparams.expert_weights_scale != 0.0f,
                hparams.expert_weights_scale,
                (llm_expert_gating_func_type) hparams.expert_gating_func,
                LLM_FFN_SILU, cb, il, gf, true, mtp_layer.ffn_up_gate_exps);
    }

    cur = lctx.cvec.apply_to(ctx0, cur, il);
    cb(cur, "mtp_post_ffn", il);
    if (hidden_out) {
        *hidden_out = cur;
    }

    ggml_tensor * output_hidden = cur;
    if (reduce_output) {
        if (cparams.mtp_op_type != MTP_OP_NONE && n_tokens > 1) {
            output_hidden = ggml_view_2d(ctx0, cur, n_embd, 1,
                    cur->nb[1], (size_t) (n_tokens - 1) * cur->nb[1]);
        } else if (inp_out_ids) {
            output_hidden = ggml_get_rows(ctx0, cur, inp_out_ids);
        }
    }
    if (reduce_output) {
        ggml_tensor * mtp_embd = output_hidden->type == GGML_TYPE_F32 ? output_hidden : ggml_cast(ctx0, output_hidden, GGML_TYPE_F32);
        cb(mtp_embd, "result_mtp_embd", -1);
        ggml_set_output(mtp_embd);
        ggml_build_forward_expand(gf, mtp_embd);
    }

    if (!emit_logits) {
        return output_hidden;
    }

    ggml_tensor * head_norm = mtp_layer.nextn.shared_head_norm
        ? mtp_layer.nextn.shared_head_norm : model.output_norm;
    ggml_tensor * head = mtp_layer.nextn.shared_head_head
        ? mtp_layer.nextn.shared_head_head : model.output;
    GGML_ASSERT(head_norm && head);
    cur = llm_build_context::build_output(lctx, ctx0, output_hidden, head, head_norm, cb);
    cb(cur, "result_output", -1);
    return cur;
}

# terrain — ik_lcpp_laguna (fork of ik_llama.cpp)
base commit: 80dd1895 (2026-07-25), branch: laguna-swa-kv
(committed through 80dd1895: opt-in --swa-compress, arch-generalized gate,
swa_layers population, --fit estimator fix, -mqkv wrap fix, dual-codification
removal. UNCOMMITTED on top: W5 ring-aware state IO + its tests. See PR.md for
the narrative, git diff --stat HEAD for the live file list.)
purpose: SWA ring KV — sliding-window layers allocate window-sized KV
instead of dense n_ctx (Blocker A: 256k context under -sm graph bf16).

## feature map (all uncommitted on branch)
- decision: src/llama-model.cpp `supports_swa_ring()` (~2249) — any
  `hparams.n_swa > 0` arch except LLAMA4 (chunk sentinel)/OPENPANGU
  (per-layer varying windows)/DEEPSEEK4/DFLASH_DRAFT (both unaudited against
  per-layer swa_layers-keyed sizing). Activation: src/llama.cpp, after the
  Mamba kv_size special case — n_seq_max==1, defrag off, --swa-compress
  opt-in (cparams.swa_compress).
- per-layer eligibility: hparams.swa_layers[il], populated per-arch in
  src/llama-hparams.cpp via `llama_hparams_set_swa_layers_periodic()`
  (GEMMA2 period 2, GEMMA3/COHERE2 period hparams.n_swa_pattern, PHI3/
  OPENAI_MOE uniform/period-2 guarded on n_swa>0). Resolved: GEMMA3/COHERE2/
  OPENAI_MOE graph builders now read hparams.n_swa_pattern instead of each
  re-declaring the period (guard leg in the shell test). GEMMA2 still keeps its
  own separate `il % 2` literal in build_gemma2.cpp — covered by a functional
  ppl-parity fixture, not by the guard.
- cache state: src/llama-context.h llama_kv_cache — swa_ring, size_swa,
  ring_occ[] (slot -> occupant cell, -1 empty). occ updated in
  llama_kv_cache_find_slot; reset in llama_kv_cache_clear.
- allocation: llama_kv_cache_init per-layer kv_size_l (dense k_l/v_l AND
  -sm graph split_k_l/split_v_l).
- writes: llm_build_kv_store + split-tensor duplicate (~3145-3220) — offset
  cell % size_swa; wrap -> two copies, cache_copies entry nulled (blocks
  reuse). FIXED (W3): -mqkv fused-QKV k_cur/v_cur are strided (non-flat)
  views into the combined qkv tensor; the wrap-split byte-flatten path now
  ggml_cont()s them first instead of asserting contiguity and aborting.
  Transposed-V wrap already sliced the CONTIGUOUS source per token BEFORE
  ggml_transpose (pre-existing fix, unaffected).
- reads: llm_build_kqv + split path — n_kv_l = size_swa, v_trans stride size_swa.
- mask: build_inp_KQ_mask_swa width size_swa; occupancy GGML_ABORT guard;
  non-causal swa-mask path aborts.
- graph reuse: llama_context::update_cache_copies patches head % size_swa,
  refuses reuse when a write would wrap.
- fattn: op_params[4]=n_swa NOT set for ring layers (CUDA wrapper tail-slices
  by index assuming newest-at-end: ggml/src/ggml-cuda/fattn.cu:44-71).
- guards: K-shift (update_internal, returns 1), defrag (ABORT), seq_cp/seq_keep
  (refuse_if_ring). seq_add/seq_div have NO entrypoint guard by design — they set
  has_shift and update_internal refuses there (an earlier entrypoint guard made
  generation continue on stale positions; the shell test pins that).
- state IO (W5, src/llama.cpp llama_data_write/read): a ring layer serializes the
  last min(size_swa, cell_count) cells oldest-first via write_ring_rows /
  write_ring_elems (transposed V), restored at (dst_cell % size_swa) by
  read_ring_rows / read_ring_elems. Blob carries LLAMA_KV_RING_MAGIC + size_swa +
  n_swa, written ONLY when swa_ring (dense blobs stay byte-compatible with pre-W5
  builds); mismatched geometry and cross-mode restores are refused.
  PARTIAL_ONLY *restore* is REFUSED under the ring (llama_state_seq_set_data_internal):
  it rewinds cells without rewriting rows, and find_slot then re-records those cells
  as their slots' owners, so the occupancy guard would confirm an occupancy the rows
  no longer match — a silent wrong-attention path, not an abort. PARTIAL_ONLY *save*
  still works (writes no rows); the server therefore still builds checkpoints it can
  no longer restore under the ring (wasteful, not incorrect — candidate follow-up).
  ring_occ is
  replayed from the restored cell indices (the whole-cache path clears it, the
  seq path gets it from find_slot). Fragmented cell layout -> throw. read_kv_cache
  discards a partial restore on the throwing paths too.
- seq_rm (llama.cpp ~2148): partial TAIL removal returns false when the
  rewind exceeds R - n_swa; server/spec-trim callers fall back to
  full-clear + reprocess.
- estimator: llama_model::cache_size (~2267) window-aware, real n_ubatch,
  gated on n_seq_max<=1 to match runtime fallback. UNFIXED: no path threads
  cparams.defrag_thold into this model-load-time function, so --fit +
  --swa-compress + non-default --defrag-thold can still under-budget.
- test: tests/test-laguna-swa-ring.sh (Laguna + GEMMA2 fixtures, -mqkv leg,
  guard/server legs, --prompt-cache round-trip, server slot save/restore)
  + tests/test-swa-ring-state.cpp (ring blob smaller than dense, byte-exact
  re-serialize, identical continuation logits, cross-mode/geometry/truncation
  refusals — locates the ring descriptor by computed blob offset rather than
  re-declaring the magic)
  + tests/test-cache-size-estimator.cpp (standalone unit
  test, casts public llama_model* to internal type, calls cache_size()
  directly — the --fit multi-device path itself is unreachable on <2 devices).
  UNTESTED: MELLUM, GEMMA4, GEMMA4_MTP/ASSISTANT, COHERE2_MOE, MIMO2, STEP35
  (ring-eligible, structurally sound, zero coverage).

## invariants
- ring exact ONLY for append-only single-seq; occupancy guard aborts otherwise.
- R >= n_swa + n_ubatch guarantees overwritten slots are out-of-window.
- CUDA/-sm graph path VERIFIED on r02-rds01 (2026-07-23, pre-generalization
  Laguna-only build) — real-model KL parity at bf16 noise floor. NOT
  re-verified against this session's arch-generalization work.

## open (see task tracker: W6 test sweep, W7 docs/PR, W8 deploy)
- W5 DONE (uncommitted): ring state IO + tests, suite green, ctest 3/3.
- still open: six ring-eligible archs with no fixture (list above); CUDA/-sm graph
  not re-verified since arch-generalization; --fit + non-default --defrag-thold
  under-budget.
- W6 candidates found by the W5 hunt (all PRE-EXISTING, none introduced by W5):
  - examples/server/server-context.cpp:3296 (discard_n_kv_and_cache_tokens) and
    :4469 (rewind_context) call llama_kv_cache_seq_rm with a nonzero p0 and IGNORE
    the return. Under the ring that refusal means the cells stay while the server
    believes they are gone. 3296 is context-shift-only (already refused downstream
    in update_internal); 4469 is on the DRY/saturate-predict rewind path and has no
    downstream refusal — the occupancy guard is the only backstop.
  - llama_kv_cache_seq_add / seq_div still have no entrypoint ring guard by design
    (an earlier guard there made generation continue on stale positions); the shell
    test pins that the refusal must come from update_internal instead.
  - read_kv_cache_meta's single-sequence branch never range-checks dest_seq_id
    against n_seq_max (the whole-cache branch does). Pre-existing upstream, dense
    too.

# SWA ring KV cache (`--swa-compress` to opt in)

## Problem

Laguna marks most layers as sliding-window attention (`n_swa = 512`; 36 of 48
on S-2.1), but the KV cache allocates dense `n_ctx` cells on every layer, so
SWA layers pay full-context memory for tokens they can never attend to. On
Laguna-S at 256k context that is 41 GB of f16 KV where 10.4 GB suffices, and
long contexts do not fit under `-sm graph` at all (bf16 reduce is forced for
this arch). At 1M (the model's trained length) dense f16 KV would be ~384 GB;
the ring brings it to 48 GB.

## Change

SWA layers store K/V in a ring of `R = GGML_PAD(n_swa + n_ubatch,
max(kv_pad, 256))` cells, written at `cell % R`. Cell metadata and slot search
are unchanged; only tensor storage, read views, and the SWA mask are
ring-aware. `R >= n_swa + n_ubatch` makes the ring exact for append-only
single-sequence use: every overwritten slot is already out-of-window for all
queries in flight.

- Writes that cross the ring end split into two copies; such graphs are not
  registered for graph reuse (the reuse offset patch refuses wraps and forces
  a rebuild). Transposed-V wrap writes slice the contiguous source before
  transposing, in both the normal and the `-sm graph` split path.
- The SWA mask is `[R, n_tokens]`, filled from an occupancy array
  (`ring_occ[slot] = cell`); a guard aborts if an in-window cell is not its
  slot's occupant, so non-append-only use can never corrupt silently.
- CUDA flash-attention `op_params[4]` stays 0 for ring layers (that kernel
  slices the K tail by index and assumes position-ordered cells).
- `llama_model::cache_size()` is window-aware for Laguna so `-sm graph`
  auto-fit accepts long contexts.
- `llama_kv_cache_seq_rm` returns false for a partial tail removal that
  rewinds past the resident window (`rewind > R - n_swa`); the server and
  speculative trim already fall back to clearing the sequence and
  reprocessing. Smaller rewinds keep partial cache reuse.
- State IO (checkpointing) is ring-aware, not refused. A ring layer serializes
  the last `min(size_swa, cell_count)` cells in oldest-to-newest order instead
  of one row per cell; replaying them into the destination ring at the slots its
  own cell indices imply reproduces the source exactly, and the cells that fall
  out are outside every future query's window by construction (`size_swa =
  pad(n_swa + n_ubatch)`). The blob carries the ring geometry so a restore into a
  differently sized window -- or into a dense cache, or a dense blob into a ring
  -- is refused instead of restored wrong. A blob saved *without*
  `--swa-compress` is byte-identical to what earlier builds wrote, so no existing
  session or slot-save file is invalidated. This covers `--prompt-cache`, the
  server's RAM prompt cache (no longer auto-disabled) and
  `/slots/{id}?action=save|restore`.
- `llama_state_seq_get_size` no longer lets an exception escape into C callers,
  and a restore that fails part-way now discards what it wrote on the throwing
  paths too (a half-restored ring would otherwise abort the next decode in the
  occupancy guard).
- `main.cpp` honors `llama_kv_cache_seq_rm`'s refusal when trimming a reloaded
  session: it reprocesses the prompt instead of continuing on cells whose K/V
  the ring no longer holds.
- `--dry-*` sampling flags are now documented in `--help` (the parsers
  existed; the help entries did not).

## Scope and limitations

The ring is opt-in via `--swa-compress` (default off, matching prior dense
behavior) and engages only for architectures that pass
`llama_model::supports_swa_ring()` (`hparams.n_swa > 0`, excluding
`LLM_ARCH_LLAMA4` (its `n_swa` is a chunked-attention sentinel, not a real
window), `LLM_ARCH_OPENPANGU` (per-layer *varying* window sizes, incompatible
with the ring's uniform invariant), `LLM_ARCH_DEEPSEEK4` and
`LLM_ARCH_DFLASH_DRAFT` (both apply/store SWA in ways not audited against the
ring's per-layer `swa_layers[il]`-keyed sizing)), with `n_seq_max == 1` and
defrag off, and only when the ring is smaller than the full context. Verified
end-to-end on Laguna and (as a real non-Laguna arch) GEMMA2, whose alternating
SWA layers are populated via a new periodic-pattern helper
(`llama_hparams_set_swa_layers_periodic()`); several other archs generalize
structurally the same way but are not yet test-covered (tracked follow-up).
Context shift (K-shift), defrag, `seq_cp` and `seq_keep` are incompatible with a
ring and fail with errors pointing at `--swa-compress` (disengage to restore the
dense behavior). State save/load is supported, but only for the single-sequence
append-only layout the ring itself supports: a fragmented cell layout is refused
rather than serialized in the wrong order. Metadata-only restores
(`LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY`, used by the server's in-context rewind
checkpoints) are also refused: they rewind cells without rewriting rows, which a
ring cannot honor once generation past the rewind point has overwritten those
slots — the server falls back to a reset, as it already does when a checkpoint
cannot be restored. `-sm graph` without flash attention crashes
upstream even with dense KV, so the split-path no-FA wrap handling is
consistency hardening, not a reachable path today.

## Verification

- `tests/test-laguna-swa-ring.sh` (registered in ctest; generates tiny
  synthetic Laguna and GEMMA2 GGUFs via gguf-py): perplexity parity
  `--swa-compress` (ring) vs dense at rel. diff 0.00e+00 (no-FA, `-ub 48` —
  deliberately not a divisor of the ring size, so wrap-split writes are
  exercised), 0.00e+00 (`-mqkv` fused-QKV, same wrap-forcing `-ub 48` —
  regression check for the non-contiguous-view wrap fix), 1.56e-06 (FA),
  0.00e+00 (`-ub 1` decode-shaped with graph reuse), 0.00e+00 (GEMMA2, a real
  non-Laguna arch with alternating SWA layers, 1.5x KV shrink); a KV-shrink
  assertion; guard legs for state save and context shift; server legs proving
  an unsafe cache rewind falls back to a full reprocess (previously an abort)
  and the prompt cache auto-disables. Generation is a smoke check only: tiny
  random-weight models have near-flat logits, so greedy text equality flips
  under any float-reduction-order change; scoring parity is the oracle.
- `tests/test-cache-size-estimator.cpp` (standalone unit test, registered in
  ctest): pins `cache_size()`'s SWA-ring sizing formula and its
  `n_seq_max <= 1` dense-fallback agreement with the runtime, independent of
  the `--fit` multi-device path (which requires 2+ devices to reach and is
  otherwise unexercised on a single-device build).
- Real model (Laguna-XS Q4_K_M, 3x NVIDIA L40, `-sm graph -fa on`):
  ring-vs-dense KL divergence sits at the bf16 numerics noise floor measured
  between two dense configs differing only in ubatch — mean KLD 0.050 vs
  0.055 control at `-ub 512` (no ring wraps), 0.046 vs 0.054 control at
  `-ub 384` (wrap-crossing writes on the split path).
- Serving (Laguna-S-2.1 117B, 2x L40, 180k context, f16 ring KV 8.8 GB):
  175k-token prefill at 1800 t/s; decode 95 -> 54 t/s from empty to full
  window; VRAM flat under sustained full-window load; cache rewinds, prompt
  switches, and slot reuse exercised against the exact traffic that
  previously aborted the server.

Refs #1607: implements the window-sized SWA KV storage requested there. The
ring approach (single shared cache, per-layer ring storage, occupancy-guarded
mask) now generalizes to any arch passing `supports_swa_ring()`, not just
Laguna; sequence-state checkpointing is implemented window-aware, so slot
save/restore and prompt caching work with the ring engaged.

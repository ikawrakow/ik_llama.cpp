# SWA ring KV cache (`--swa-compress` to opt in)

**Ready for review.** Opt-in and default-off, so a build without the flag behaves
exactly as before. Verified on 2x L40 under `-sm graph -fa on` on real weights
across Laguna, GEMMA2/3, cohere2 and mellum, and serving Laguna-S at 180k
context in production. Two disclosures a reviewer should weigh rather than
discover: the GEMMA4 `op_params[4]` gate is defensive, not measured -- `-sm graph`
downgrades to `layer` for GEMMA4 whenever `n_embd_per_layer > 0`, true of every
GGUF available to me, so its hand-rolled builder could not be reached to
demonstrate the bug -- and the graph audit has no negative test; its refusal
behaviour is evidenced only by the episode where it correctly refused GEMMA2
until fused `SOFT_CAP_MAX` was added to the walk.

## Problem

Laguna marks most layers as sliding-window attention (`n_swa = 512`; 36 of 48
on S-2.1), but the KV cache allocates dense `n_ctx` cells on every layer, so
SWA layers pay full-context memory for tokens they can never attend to. On
Laguna-S at 256k context that is 41 GB of f16 KV where 10.4 GB suffices, and
long contexts do not fit under `-sm graph` at all (bf16 reduce is forced for
this arch). At 1M (the model's trained length) dense f16 KV would be ~384 GB;
the ring brings it to 48 GB.

## Change

SWA layers store K/V in `n_seq_max` stripes of `W = GGML_PAD(n_swa + n_ubatch,
max(kv_pad, 256))` rows each. A token of sequence `s` at position `p` is written
at row `s*W + p % W`. Cell metadata and slot search are unchanged; only tensor
storage, write offsets, read views, the SWA mask and state IO are ring-aware.

`W >= n_swa + n_ubatch` makes the ring exact for append-only use: a row is only
overwritten by a token `W` positions later *in the same sequence*, by which point
the oldest query in that ubatch is already more than `n_swa` past it. Deriving the
row from `(sequence, position)` rather than from the cell index is what makes this
hold per sequence -- eviction distance is then measured in each sequence's own
positions, so a parked sequence cannot have its window evicted by a busy one --
and it also makes tail rewinds self-consistent: re-appending at position `p`
rewrites exactly the row `p` held before, whatever cell it lands in.

- `find_slot` precomputes the ubatch's destination row runs (one per sequence,
  split again where a run wraps its stripe). One contiguous run covering the whole
  ubatch is the fast path (a single view copy, registered for graph reuse);
  anything else -- a mixed-sequence ubatch, or one that wraps -- is emitted as one
  copy per run with reuse refused. That refusal is load-bearing: the reuse key does
  not distinguish sequence mixes, so a reused graph could otherwise be patched with
  one sequence's row offset while the ubatch holds two. Transposed-V writes slice
  the contiguous source before transposing, in both the normal and the `-sm graph`
  split path.
- The SWA mask is `[n_seq_max*W, n_tokens]`; a token sees only its own stripe (the
  rest is masked out wholesale). It is filled from an occupancy array
  (`ring_occ[row] = cell`); a guard, comparing positions *per sequence*, aborts if
  an in-window cell is not its row's occupant, so non-append-only use can never
  corrupt silently.
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
  of one row per cell; replaying them into the destination sequence's stripe at
  the offsets the restored positions imply reproduces the source exactly, and the cells that fall
  out are outside every future query's window by construction (`size_swa =
  pad(n_swa + n_ubatch)`). The blob carries the PER-SEQUENCE window rather than the
  total row count, so a checkpoint moves freely between `--parallel` values and
  between slots, while a restore into a differently sized window -- or into a dense cache, or a dense blob into a ring
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
ring's per-layer `swa_layers[il]`-keyed sizing)), and only when
the ring (`n_seq_max * W` rows) is smaller than the full context -- so a small
context with many slots correctly declines to engage and stays dense. Verified
end-to-end on Laguna, on GEMMA3 (`gemma-3-4b-it`, real weights, CUDA) and on
(as a synthetic non-Laguna fixture) GEMMA2, whose alternating
SWA layers are populated via a new periodic-pattern helper
(`llama_hparams_set_swa_layers_periodic()`).

Archs beyond those are no longer admitted on the strength of "they route
through the same helpers" -- that claim turned out to be false. Several
builders hand-roll their KV access instead (`grep cache_copies src/graphs`),
and `build_gemma4_graph_parallel()` writes at a dense `llm.kv_head` offset and
reads back `llm.n_kv` rows, neither of which the ring's row geometry survives.
A hand-rolled *write* is also the one violation the occupancy guard cannot
catch: `ring_occ` is recorded by `find_slot` wherever the write was *meant* to
land, so a write that goes elsewhere is silently wrong rather than loud.

So the graph is audited instead of the arch name being trusted
(`llama_kv_ring_audit_graph()`, once, at reservation): for every attention node
whose K roots at a ring layer's cache tensor, the mask must root at the SWA
mask and the K view must span all `size_swa` rows, and every ring layer must be
reached by at least one such node. Operands are traced through `view_src` plus
the materialization/shape hops; the layer map includes each per-device split, so
`-sm graph` resolves. It covers `FLASH_ATTN_EXT`, `SOFT_MAX`, the fused
`SOFT_CAP_MAX` (gemma-family logit softcapping -- a `SOFT_MAX`-only walk sees
*no* attention node on those layers and the audit correctly refused GEMMA2
until it was added) and `LATENT_ATTN`. A failed audit refuses context creation:
the ring's tensors are allocated by then, so there is no falling back to dense.

The denylist stays, because it encodes what a graph walk cannot see: LLAMA4's
chunk sentinel and openPangu's per-layer windows differ at *mask-fill* time with
identical wiring, and DFLASH_DRAFT never touches `kv_self` at all.
Context shift (K-shift) and `seq_keep` are incompatible with a ring and
fail with errors pointing at `--swa-compress` (disengage to restore the dense
behavior).

`seq_cp` **is** supported, for a full range only, and its semantics necessarily
differ from dense: a ring cell's row is derived from its sequence and only one
sequence can own it, so the tokens are COPIED into the destination's own rows
(the cache grows) and the destination is REPLACED rather than united with the
source. A sub-range is refused -- it would have to rewind the destination's rows
without rewriting them, the same reason a `PARTIAL_ONLY` state restore is
refused. Both ids must be `< n_seq_max`, since an out-of-range id addresses a
stripe the ring does not own (dense tolerates it: it only grows a `std::set`).
`seq_cp` returns void, so a caller that needs to know tests
`llama_kv_cache_seq_pos_min(ctx, dst) != -1` -- note `seq_pos_max` returns 0 for
an empty sequence and cannot serve here.

`seq_keep` stays refused deliberately, not for want of a mechanism: under the
ring it is just a whole-sequence `seq_rm` of every other sequence, which the
ring already supports. Its only in-tree callers are `examples/lookahead` and
`examples/speculative`, and unlocking it would make them *correct but unusable*.
`lookahead` fans out to `W + G + 1 == 31` sequences per generated token, each a
full-range `seq_cp`, and the copy above moves a stripe through
`llama_state_seq_get_data`/`set_data` -- device to host to device. An honest
refusal beats roughly a gigabyte of PCIe traffic per token. The escape, if those
programs are ever wanted under `--swa-compress`, is an internal device-side
stripe copy (contiguous `[seq*ring_w, (seq+1)*ring_w)` slabs, `ggml_cpy` for K, a
strided view over `size_swa` for transposed V) behind the same void API, with no
blob; `seq_keep` then follows for free.

Defrag is incompatible too, but it LOSES to the ring rather than
changing the layout: `--defrag-thold` is ignored with a warning when the ring
engages, and an explicit `llama_kv_cache_defrag()` is dropped with an error rather
than aborting the process. Keeping the layout independent of a context-time-only
setting is what lets the model-load-time `--fit` estimator stay correct without
being told about it. Server system prompts work, now that the fan-out's `seq_cp` does: the server
verifies the fan-out landed by counting the seq_ids it added, and drops the
prompt with a reported reason rather than serving slots whose K/V never received
it. State save/load is supported per sequence, but
only for the append-only layout the ring itself supports: a fragmented cell layout,
or a whole-context save holding several sequences (its cells are packed from index
0 with no sequence of their own), is refused rather than serialized in the wrong
order. Metadata-only restores
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
  `n_seq_max` scaling -- and the clamp where the striped ring reaches the dense
  size -- in lockstep with the runtime, independent of
  the `--fit` multi-device path (which requires 2+ devices to reach and is
  otherwise unexercised on a single-device build).
- **CUDA flash attention's SWA tail slice is unsound with `--parallel > 1`** -- found while
  benchmarking this branch, fixed here, filed upstream as ikawrakow/ik_llama.cpp#2186.
  `ggml_cuda_flash_attn_ext` keeps the last `pad(n_swa + n_tokens)` **cells** and drops the
  rest from the tensor (`ggml-cuda/fattn.cu:52-66`); with several slots interleaving, one
  sequence's window spans ~`n_seq_max` times as many cells as the slice keeps, so in-window
  cells become unattendable rather than masked. `op_params[4]` was set on every dense SWA
  layer with no sequence-count condition; it is now also gated on `n_seq_max == 1`
  (`llama-build-context.cpp:2038`, `:3303`). The ring already left it at zero, since its
  rows are not position-ordered.

  gemma-2-2b-it (`n_swa = 4096`), 2x L40, `-sm graph`, four concurrent 3,313-token prompts,
  greedy, top-5 logprobs of the first 12 generated tokens:

  | comparison | result |
  |---|---|
  | `-np 1`, `-fa on` vs `-fa off` (control: the slice is a superset) | text identical, max abs logprob diff **0.31** |
  | `-np 4`, `-fa on` vs `-fa off` | **2 of 4 slots produce different text**, max diff **7.11** |
  | `-np 4`, **ring** vs `-fa off` | **4 of 4 identical**, max diff **0.47** |

  The control bounds FA arithmetic noise at ~0.3; the `-np 4` divergence is ~23x that and
  changes sampled tokens. The ring agrees with the unsliced reference.

- Throughput, measured (2x L40, `-sm graph`, `gemma-2-2b-it`, 64k context, 14,002-token
  prompts, 256 tokens generated each, aggregate over slots), **both sides correct** (i.e.
  after the fix above):

  | config | ring pp / tg | dense pp / tg | KV |
  |---|---|---|---|
  | `-np 1`, 1 request  | 21,392 / 154.6 tok/s | 20,701 / 163.2 tok/s | 3,562 vs 6,656 MiB |
  | `-np 4`, 4 concurrent | 9,138 / **160.5** tok/s | 9,616 / 139.1 tok/s | **4,264** vs 6,656 MiB |

  At `-np 1` it is a wash for roughly half the KV. At `-np 4` the ring is ~5% behind on
  prompt processing and **~15% ahead on generation**, at 0.64x the KV -- so the striped ring
  is the faster *and* smaller multi-sequence configuration, not a memory-for-speed trade.

  Two earlier readings of this are worth recording as corrections, since both were wrong in
  the same direction. Against dense `-fa on` BEFORE the fix the ring looked ~16-24% slower;
  that baseline was the unsound one. Against dense `-fa off` (correct, but no flash
  attention at all) the ring looks ~1.9x faster on both axes, which flatters it. The table
  above is the comparison that holds both sides to the same standard. Graph rebuilds were a
  third false lead: an A/B with `-no-gr` moves both sides ~1%.

- Multi-sequence (`-np 4`) on real weights, 2x NVIDIA L40 under `-sm graph -fa on`
  (`gemma-2-2b-it`, `n_swa = 4096`, 64k context):
  - `tests/test-swa-ring-multiseq.cpp` passes 24/24 with `LLAMACPP_TEST_NGL=99`,
    so the striped write offsets, the mixed-ubatch multi-run writes and the
    cross-stripe state IO are exercised on device buffers and on the split-tensor
    path, not only on CPU.
  - The ring engages and is sized per sequence: 4,608 rows at `-np 1`, 18,432 at
    `-np 4` (KV 4,264 vs 6,656 MiB dense). With a context too small for the
    stripes (`-c 16384`, `-np 4`) it correctly declines and stays dense.
  - A 4-slot server (`--swa-compress -np 4 -ub 48`, ring 17,408 rows, KV 4,212 vs
    6,656 MiB) answered four concurrent requests with text IDENTICAL to the same
    dense server, each slot recalling its own planted token. Then the skew case a
    shared ring cannot survive: one slot generated a long run while the others sat
    idle, and every idle slot's continuation was still identical to dense.
  - `/slots/2?action=save|restore` on a non-zero slot round-tripped 884 cells and
    continued identically -- the case that failed before the serializer followed
    position order rather than cell order.
  - Zero `GGML_ASSERT`/`GGML_ABORT` on either server. `--fit` with `-np 4` did not
    OOM; transposed V (`-fa off`) and `-mqkv` both ran clean at `-np 4`.
- CUDA / real weights, current build (2x NVIDIA L40, GPU-offloaded):
  - `tests/test-swa-ring-state.cpp` passes 31/31 with `LLAMACPP_TEST_NGL=99`
    (Laguna-XS Q4_K_M, all 40 layers on GPU), so the ring state IO -- including
    the cross-mode and geometry refusals -- is exercised on device buffers, not
    only on CPU. `LLAMACPP_TEST_NGL` defaults to 0, so ctest is unchanged.
  - Server slot save/restore round-trip under `-sm graph` (Laguna-S-2.1
    UD-Q5_K_S, 180k ctx, 18,690-token prompt, i.e. far past both `n_swa` and the
    ring): `/slots/0?action=save` wrote 700,059,692 B for 11,168 cells in 244 ms
    -- the window-aware size (12 global layers x 11,168 + 36 ring layers x 1,024
    cells; a dense blob would be ~2.09 GB) -- and after poisoning the cache with
    an unrelated prompt, `action=restore` read back the same byte count and the
    continuation was IDENTICAL to the pre-save baseline. This is the split-tensor
    read path (`read_kv_cache_data_split`), which is unreachable on a single
    device and therefore untestable on CPU.
  - GEMMA3 (`gemma-3-4b-it` Q4_K_M, a real non-Laguna SWA arch from another
    publisher): ring engages at `n_swa = 1024`, pattern 6, ring 1536 cells; KV
    544 -> 225 MiB (2.4x). Perplexity ring vs dense 3.1249 vs 3.1192 (rel.
    1.8e-03) against a dense-vs-dense ubatch control spread of 7.2e-03 -- i.e.
    inside the numerics noise floor; with `-fa off` the gap tightens to 6.4e-04.
  - Laguna-XS Q4_K_M re-measured on this build: ring 2.7070 vs dense 2.7120 over
    20 chunks (rel. 1.8e-03, ring marginally lower, +/- 3.2e-02 stderr),
    bit-identical across repeated runs. Note that at only 6 chunks the same
    comparison reads +1.1e-02: ring and dense are not bit-identical under CUDA
    flash attention (different `n_kv`, hence different kernel tiling and
    reduction order), so short perplexity samples are not a usable oracle here.
  - A non-SWA arch from another publisher (`qwen35`, `n_swa = 0`) refuses the
    ring cleanly: "`--swa-compress` requested but this arch does not support the
    SWA ring -> using full-size KV", correct generation.
- Per-arch sweep on real weights, all under `-sm graph -fa on` on 2x L40 (each arch
  run through: ring engagement + KV shrink, perplexity ring vs dense at a
  wrap-forcing `-ub 48`, `-mqkv` fused QKV, transposed V (`-fa off`, single-GPU
  since `-sm graph` requires FA), decode-shaped `-ub 1` graph reuse, the defrag
  and context-shift guards, `--prompt-cache` round-trip, `--fit`, and a server
  slot save/restore over a prompt longer than the window):

  | arch | model | ring cells | KV ring/dense | ppl ring vs dense | slot save/restore |
  |---|---|---|---|---|---|
  | gemma2   | gemma-2-2b-it            | 4608 | 650 / 832 MiB   | 2.8392 / 2.8392 (exact) | identical |
  | gemma3   | gemma-3-4b-it            | 1536 | 225 / 544 MiB   | 3.1249 / 3.1192 | (cli only) |
  | cohere2  | c4ai-command-r7b         | 4608 | 688 / 1024 MiB  | 2.5108 / 2.5108 (exact) | identical |
  | gpt-oss  | gpt-oss-20b              | 768  | 210 / 384 MiB   | 2.2062 / 2.2060 | identical |
  | gemma4   | gemma-4-E2B-it           | 1536 | -- / 144 MiB    | 5.7570 / 5.7314 | identical |
  | laguna   | Laguna-S-2.1 / XS        | 1024 | 8784 MiB @ 180k | within noise    | identical |
  | mellum   | Mellum-4b-sft-all        | 1536 | 175 / 448 MiB   | 2.2557 / 2.2584 | (single GPU) |

  `-mqkv` and transposed V reproduce the ring column exactly on gemma2 and cohere2;
  zero `GGML_ASSERT`/`GGML_ABORT` anywhere in the sweep; `--fit`'s multi-device
  estimator path (unreachable below 2 devices, hence never previously executed) ran
  without OOM on every arch.

  Two findings from the sweep are NOT ring defects and are filed upstream: Phi-3
  GGUFs lacking `phi3.attention.sliding_window` fail to load at all (the compat
  fallback naming those models sits below a required `get_key`, ikawrakow/ik_llama.cpp#2183),
  and cohere2 under `-sm graph` aborts with a CUDA illegal memory access on context
  shift with `--swa-compress` absent and the ring never engaged (#2184). `mellum`
  aborts under `-sm graph` at graph build (`GGML_ASSERT(nhave > 1)` in `ggml_reduce`,
  #2185) with dense KV and no fork flags, so its numbers above are single-GPU;
  `-mqkv` (2.2613), `-ub 1` (2.2549) and transposed V (2.2577 vs dense 2.2584) all
  land inside the same band there, and `-np 4` engages with zero aborts. `gemma4`
  with `-fa off` returns garbage perplexity for dense as well as ring, so the
  transposed-V leg is not measurable there.

  The context-shift guard leg is only meaningful where the ring actually engages
  (it cannot when `n_swa >= n_ctx`); it is pinned on Laguna and gpt-oss, and the
  guard itself is arch-independent (`update_internal` refusing on `has_shift`).
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

### Graph audit, verified on 2x L40 (`-sm graph`, CUDA)

The audit's own risk is a *false* refusal -- an opt-in user whose setup worked
yesterday getting a refused context today. Swept under both `-sm graph` and
`-sm layer` with `--swa-compress -fa on`: Laguna-S 36 SWA layers audited,
GEMMA2 13, GEMMA4 12, cohere2 24, zero refusals, and the live 180k-context
production config starts and serves unchanged. (phi3 still fails to load at all
-- #2183, unrelated, ring never engages.)

Two honest limits on this. First, `build_gemma4_graph_parallel()` is a *latent*
defect, not a demonstrated one: `-sm graph` is downgraded to `layer` for
GEMMA4 whenever `hparams.n_embd_per_layer > 0` (`llama.cpp:4081`), which is true
of every GEMMA4 GGUF I have, so I could not reach the hand-rolled builder to
show it misbehaving. It is reachable in principle by a GEMMA4 variant without
per-layer embeddings; the audit is what would refuse it there. Second, the
`op_params[4]` gate added to that same builder is therefore **defensive, not
measured** -- it applies the identical single-sequence contract the shared
builders already carry (upstream #2186, which *was* measured on gemma-2), to a
path I could not exercise. It changes dense behaviour only for
`--parallel > 1` on that unreachable-here variant.

Refs #1607: implements the window-sized SWA KV storage requested there. The
ring approach (single shared cache, per-layer ring storage, occupancy-guarded
mask) now generalizes to any arch passing `supports_swa_ring()`, not just
Laguna; sequence-state checkpointing is implemented window-aware, so slot
save/restore and prompt caching work with the ring engaged.

# GLM-5.2 / DeepSeek-V3.2 DSA "Lightning Indexer" — Implementation Progress

Branch: `glm-dsa-indexer`  Build: `build-idx` (CUDA sm_60, 3x P100)
Model: `/mnt/optane0/GLM-5.2-UD-IQ2_M` (arch `glm-dsa`, key_length=576, value_length=512, indexer top_k=2048, 32 indexer heads, head_size=128)

The DSA lightning indexer scores each query against the (Hadamard-rotated) indexer keys, keeps the
top-k highest-scoring keys per query, and masks the rest out of attention. It is implemented for
`LLM_ARCH_GLM_DSA` inside ik's deepseek2 graph (`src/graphs/build_deepseek2.cpp`,
`build_deepseek2_dsa_indexer` / `build_deepseek2_dsa_sparse_mask` / `build_deepseek2_dsa_fa_mask`).
Reference (verbatim mainline source) in `DSA_REFERENCE.md`.

Escape hatches: `DSA_INDEXER_DISABLE=1` (dense fallback), `DSA_HADAMARD_DISABLE=1`,
`DSA_TOPK_OVERRIDE=N`, `DSA_SINK=N` (force-include first N sink keys in top-k).

## History (commits on the branch)

- `587351ca` — scaffold: batch-local, single-seq **prefill** only. c512 PPL exact no-op vs dense
  (2.7760), prefill coherent. Decode degenerated (each generated token saw only itself as an
  indexer key). -fa 1 not handled.
- `f03f5ed4` — **decode-correct via a persistent per-layer indexer-K cache** (`kv_self.kr_l[il]`,
  F16, `[head_size, kv_size]`). A decoded token now scores against ALL past indexer keys. Uses a
  full-coverage rank-scatter mask (writes every key slot exactly once) to dodge a CUDA in-place
  `set_rows` quirk.
- `b3cce6c2` — **wire the sparse mask into the flash-attention path** (`-fa 1`, our serving config),
  not just the `-fa 0` soft_max path. c512 -fa1 PPL exact vs dense. BUT long-ctx -fa1 decode was
  **unvalidatable** because of a pre-existing P100 MLA-FA vec-decode bug.
- `5f18dcc0` — **(UPDATE 4) cherry-pick the MLA-FA vec-decode fix** (`391eb467` from
  `consol-canonical`). See below.

---

## UPDATE 4 (2026-06-25): MLA-FA fix merged; FA path re-validated; multi-seq characterized

### 1. MLA-FA vec-decode fix merged (`5f18dcc0`)

Cherry-picked `391eb467` (branch `cuda-mla-fa-vec-decode` / `consol-canonical`) into the indexer
branch — clean apply, no conflicts (the two touched files were byte-identical to the fix's parent).
The fix: in `fattn-vec-f16.cuh` / `fattn-vec-f32.cuh` the FA vec kernel's V loop and V pointer must
step **Dk** (not Dv) for asymmetric MLA head sizes (Dk=576 K, Dv=512 V); threads `tid>=Dv` read 0.
Keyed on compile-time `Dk!=Dv`, so symmetric kernels are byte-identical. Rebuilt `llama-cli` and
`llama-perplexity` clean.

### 2. FA path re-validation — **THE FA PATH IS ALIVE.**

All runs: 3x P100 (`-ngl 99 --cpu-moe`), `numactl --interleave=all`, `GGML_CUDA_NO_PINNED=1`,
wikitext-2 `wiki.test.raw`. Long-ctx generation uses a 2521-token prompt (> top_k 2048, so the mask
**actively bites**) and decodes 129 tokens at temp 0.

**c512 PPL (mask is a no-op at n_ctx<top_k — proves correctness-preservation):**

| Config | PPL (8 chunks) |
|---|---|
| Indexer ON,  `-mla 3 -fa 1` | **2.0854** |
| Dense (DISABLE), `-mla 3 -fa 1` | **2.0854** (byte-identical, all 8 chunks) |

ON == dense exactly → indexer is an exact no-op when `n_kv <= top_k`, no regression on the FA path.
(The shift from the previously-recorded 2.0743 to 2.0854 is entirely due to the cherry-picked
MLA-FA fix changing the V accumulation; since ON==dense it is the new correct dense baseline, not an
indexer artifact.)

**Long-context decode (2521-tok prompt, mask actively bites during prefill AND decode):**

| Config | Result |
|---|---|
| **Pre-fix (587351ca, -fa 0)** | DEGENERATE: collapsed to `0.0.0.0.0...` after the prompt |
| Indexer ON, `-mla 1 -fa 1` | **COHERENT** — accurate summary of the prompt |
| Indexer ON, `-mla 3 -fa 1` | **COHERENT** — even recalls "alongside Ben Whishaw" (deep-context detail) |
| Dense (DISABLE), `-mla 1 -fa 1` | COHERENT (control) |
| Indexer ON, `-mla 1 -fa 0` | COHERENT (control) |

Every run generated all 129 decode tokens with no NaN/abort. **The MLA-FA fix fully unblocked the
long-context `-fa 1` decode path that was the open blocker.** The indexer is now feature-complete and
validated for single-sequence prefill + decode on both the soft_max (`-fa 0`) and flash-attention
(`-fa 1`) paths, at `-mla 1` and `-mla 3`.

### 3. Multi-sequence (n_seq>1) — characterized; root cause found; NOT yet fixed

Tested with llama-perplexity packing 2 sequences per batch (`-c 4096 -b 8192` → n_seq=2), at
n_ctx=4096 > top_k=2048 so the mask actively bites:

| Config | n_seq=1 | n_seq=2 |
|---|---|---|
| Dense (indexer OFF) | — | **2.5406** (healthy) |
| Indexer ON | **3.0524** | **62.6474** (broken, ~20x worse) |

- Dense multi-seq is healthy → the fork's MLA multi-seq path is fine.
- Indexer single-seq is healthy.
- **Indexer + multi-seq is numerically broken** (no NaN/crash anymore — the persistent cache + full
  scatter mask removed the old hard crash — but the top-k selection is wrong). The fault is isolated
  to the indexer's single-sequence assumption.
- Note: at `n_ctx <= top_k` (mask is a no-op) multi-seq is fine (n_seq=2 c512 == single-seq). The
  break only appears once the mask bites.

**Root cause** (`build_deepseek2_dsa_indexer`): the indexer uses the graph's single scalar `kv_head`
and `n_kv` for the whole ubatch. In a multi-seq ubatch, perplexity packs seq 0 at cache slots
`[0, n_ctx)` and seq 1 at `[n_ctx, 2*n_ctx)`, but the indexer:
  1. writes the entire ubatch's keys at one `kv_head` offset (line ~427), and
  2. reads back `[0, n_kv)` and scores/argsorts every query against the full `n_kv` key span.
The base block-diagonal KQ_mask is added before argsort, so cross-sequence keys *should* sort to the
bottom — but the cache-write offset and the single contiguous read-back corrupt the per-sequence key
layout once the mask bites, giving wrong top-k sets. (The exact interaction needs a per-key dump to
pin down whether the dominant error is the write offset or the cross-seq argsort tie-breaking; both
are consequences of the same single-`kv_head`/single-`n_kv` assumption.)

**Required work** (deferred — structural, not safely landable+testable in this session):
  - Plumb per-token `seq_id` (from the ubatch) and per-sequence `kv_head` into the indexer graph
    builder; today only scalar `kv_head`/`n_kv` reach it.
  - Write each sequence's indexer keys to its own cache slot range, and run the score/argsort/top-k
    **per sequence** (or mask cross-seq keys to a true -inf *before* argsort so they can never enter
    top-k, and confirm the argsort is stable w.r.t. ties at the -inf floor).
  - Re-run the n_seq=2 c4096 PPL; target ≈ the dense multi-seq value (2.54) and ≈ single-seq indexer
    (3.05), not 62.6.

### 4. deepseek32 arch wiring — N/A in this fork

The mainline reference (`DSA_REFERENCE.md`) implements DSA only under arch `deepseek32`
(`LLM_ARCH_DEEPSEEK32`, DeepSeek-V3.2); mainline's `glm-dsa` is a stub that loads indexer tensors but
runs the plain deepseek2 MLA graph. **This fork has no `LLM_ARCH_DEEPSEEK32` enum** — DSA is
implemented entirely under `LLM_ARCH_GLM_DSA`, wired into ik's `build_deepseek2.cpp`. So "wire the
deepseek32 path" is not applicable here as written: our single source of truth is `glm-dsa`, and the
deepseek32 reference graph has already been mirrored into the glm-dsa code path. If a real
DeepSeek-V3.2 GGUF (`general.architecture == "deepseek32"`) needs to be served later, the work is to
add the `LLM_ARCH_DEEPSEEK32` enum + arch-name mapping + tensor-name table and route it through the
same `build_deepseek2_dsa_*` helpers (the indexer logic is arch-agnostic; only the arch gate at
`build_deepseek2_layer_attention` line ~671 and the kr_l-cache gate would need to include the new
enum). Untested without a deepseek32 GGUF on disk.

---

## UPDATE 5 (2026-06-25): multi-sequence FIXED — per-sequence attention sink; n_seq>1 now healthy

Branch: `glm-dsa-multiseq` (isolated sub-branch off `glm-dsa-indexer`; the validated single-seq branch
is untouched). Build `build-idx`.

### Root cause (refined from UPDATE 4)

The break was **not** the cache write offset or the cross-seq argsort. The persistent indexer-K cache
write at `kv_head` and the score/argsort are already per-sequence correct: in a multi-seq ubatch every
token is placed contiguously at `kv_self.head + i` (exactly like the main K cache), and the base
KQ_mask added before argsort already drives every cross-sequence key to `-inf` (it is filled from
`kv_self.cells[i].has_seq_id(seq_id)` per query), so cross-seq keys can never enter a query's top-k.

The actual fault was the **attention-sink force-include**. The old code boosted the *global* key range
`[0, n_sink)` by `+1e20` (a per-key `{n_kv}` vector via `ggml_arange`). With several sequences packed
into one ubatch — seq 0 at cache cells `[0, n0)`, seq 1 at `[n0, n1)`, … — this only protects sequence
0's sink. Sequence 1's sink lives at cell `n0`, not cell 0, so it received no boost and was dropped from
top-k once the mask bites (`n_kv > top_k`), collapsing that sequence. This is exactly the documented
"masking the sink collapses the transformer," but only for the non-first sequences. Empirically: at
c4096 n_seq=2, chunk[1] (seq 0) = 2.33 healthy, chunk[2] (seq 1) = 61.2 broken — the break is isolated
to the *second* sequence, the smoking gun for a sink anchored at the wrong (global) cell.

### Fix: per-sequence sink as a CPU-filled input tensor

Replaced the global arange sink boost with a per-graph input tensor `inp_dsa_sink` `{n_kv, n_tokens}`
(F32), filled on the CPU in `llama_set_inputs` from `kv_self.cells` exactly like the KQ_mask:

    inp_dsa_sink[j, i] = 1e20  iff  cell[i].pos in [0, n_sink) AND cell[i].has_seq_id(seq_of_query_j)
                       = 0     otherwise

so each query's own sequence's sink is force-included, and a query never boosts another sequence's
keys. The boost is still finite, so it cannot un-mask causal/future `-inf` positions.

Files:
  - `src/graphs/build_deepseek2.cpp` `build_deepseek2_dsa_indexer`: build/add `lctx.inp_dsa_sink`
    (lazily created on first layer, reused across layers) in place of the arange boost.
  - `src/llama-context.h`: new member `inp_dsa_sink`.
  - `src/llama-build-context.cpp`: reset `inp_dsa_sink = nullptr` per graph build.
  - `src/llama.cpp` `llama_set_inputs`: fill it from `kv_self.cells` + `batch.seq_id`.

**Single-seq is byte-identical.** For a single contiguous sequence starting at pos 0, cells `[0,n_sink)`
have `pos < n_sink` and the same seq_id, so the set boosted is exactly the old "cell index < n_sink"
set with the same `1e20` magnitude. n_seq==1 numerics are unchanged (verified below, byte-identical).

### Validation (3x P100, `-ngl 99 --cpu-moe -mla 3 -fa 1`, `numactl --interleave=all`,
`GGML_CUDA_NO_PINNED=1`, wikitext-2 `wiki.test.raw`)

**Multi-seq correctness (mask actively bites, n_kv > top_k):**

| Config | Before fix | After fix |
|---|---|---|
| c4096 n_seq=2, indexer ON, chunk[2] (seq 1) | **61.2** (broken) | **3.07** (healthy) |
| c4096 n_seq=2, indexer ON, chunk[1] (seq 0) | 2.33 | 2.33 (unchanged) |
| single-seq indexer reference (UPDATE 4) | 3.05 | — |

chunk[2] 61.2 → 3.07, matching the single-seq indexer value (~3.05). Fixed.

**n_seq=4 == n_seq=1, chunk-for-chunk (the strongest correctness proof):** at c2048,
`DSA_TOPK_OVERRIDE=1024` (mask bites: 1024 < 2048 keys/seq):

| chunk | n_seq=1 indexer ON | n_seq=4 indexer ON |
|---|---|---|
| [1] | 2.5005 | 2.5005 |
| [2] | 2.6080 | 2.6080 |
| [3] | 2.7759 | 2.7759 |
| [4] | 3.1138 | 3.1137 |
| Final | **3.1138** | **3.1137** |

Multi-sequence batched processing is now numerically identical (to FP rounding) to processing each
sequence on its own. The indexer is sequence-correct.

**No regression (single-seq):** c512 n_seq=1, 4 chunks, indexer ON vs dense (`DSA_INDEXER_DISABLE=1`):

| chunk | Indexer ON | Dense |
|---|---|---|
| [1]..[4] | 2.2770 / 2.8741 / 2.3956 / 2.1957 | 2.2770 / 2.8741 / 2.3956 / 2.1957 |
| Final | **2.1957** | **2.1957** (byte-identical) |

Indexer ON == dense, all chunks exact → single-seq no-op preserved, no regression.

### Known limitation (capacity, not correctness)

n_seq=4 at the full c4096 (n_kv=16384) OOMs the P100 compute buffer: the indexer's argsort + score over
`16384 keys × n_tokens` per layer exceeds 16 GB VRAM during graph reservation. This is a memory-capacity
ceiling of the 3x P100 rig with a large packed batch, **not** an indexer correctness issue — n_seq=4 is
proven correct at c2048 (n_kv=8192). Larger packed multi-seq batches need either more VRAM, a smaller
ubatch, or a future memory optimization of the indexer score path (e.g. chunked argsort).

**Multi-sequence (n_seq>1) is now fixed and validated.** The GLM-5.2 DSA lightning indexer is feature-
complete and sequence-correct for prefill + decode, soft_max + flash-attention, `-mla 1`/`-mla 3`, and
n_seq>=1. **Fully general and PR-ready.**

---

## UPDATE 6 (2026-06-25): serving-correctness — kr_l maintained across shift/defrag/seq-ops; per-seq sink anchored on first-present pos; context-shift on MLA characterized

Branch `glm-dsa-multiseq`. An adversarial review (verified) flagged that the indexer was proven on the
**perplexity** path but not the **serving** path: the persistent indexer-K cache `kr_l` was written and
read but never *maintained* by the KV-cache mutators (K-shift, defrag, seq-ops), and the attention sink
anchored on absolute `pos < n_sink` (wrong for a sequence whose early tokens were `seq_rm`'d). This
update closes those gaps and — importantly — pins down what is actually reachable on our MLA model.

### 1. kr_l now wired into every KV-cache mutator

- **`build_k_shift`** (`src/llama-build-context.cpp`): after the main-K RoPE-delta loop, a new block
  rotates the indexer keys by the **same per-cell delta**. The cached key is `H·concat(RoPE(k_pe,pos),
  k_nope)`, so we **un-Hadamard (H·kr, H symmetric/orthonormal ⇒ H·H=I) → RoPE-delta the pe sub-block →
  re-Hadamard**. Exact because GLM-DSA carries **no rope-scaling metadata** ⇒ `ext_factor==0`,
  `attn_factor==1`, `freq_scale==1` (confirmed at runtime), so NEOX RoPE is a pure, composable rotation:
  `RoPE(x,pos+delta)==RoPE(RoPE(x,pos),delta)`. Params (`rope_factors=nullptr`, `n_rot`, NEOX,
  `freq_base`, `ext_factor`, `attn_factor`) mirror the forward indexer RoPE byte-for-byte; the
  DEEPSEEK2-only `yarn_attn_factor_shift` does NOT leak in (GLM_DSA≠DEEPSEEK2). Non-in-place
  (cont→rope→concat→re-Had→cpy), no aliasing. The k-shift Hadamard input is filled in
  `llama_set_k_shift` with the identical Sylvester construction.
- **`build_defrag`** (`src/llama-build-context.cpp`): a `kr_l` row-move `ggml_cpy` mirrors the `k_l`
  move (defrag does **not** change `pos`, so no re-RoPE — a plain row follow is correct). `max_moves`
  divisor bumped 6→9 `*n_layer` when the indexer cache is present (kr_l adds 3 nodes/layer/move).
- **seq-ops** (`seq_rm`/`seq_cp`/`seq_keep`): verified **metadata-only** — they touch
  `cells[].seq_id`/`pos`/`used`/`head` and never move K/V/kr_l tensor data, so a cell keeps its physical
  index and its kr_l row stays matched. `seq_add`/`seq_div` change `pos` and set `has_shift=true`,
  routing through K-shift. **No kr_l action needed in the seq-ops themselves.**

### 2. Per-sequence sink anchored on first-present pos (not absolute pos<n_sink)

`llama_set_inputs` now boosts each query's own sequence's first `n_sink` **present** tokens:
`inp_dsa_sink[j,i]=1e20 iff cell i has query j's seq AND cell.pos ∈ [min_pos_of_seq, min_pos_of_seq +
n_sink)`, where `min_pos_of_seq` is the per-sequence minimum present `pos` over the scored `n_kv` span.
After multi-turn `seq_rm` drops a sequence's early tokens, its earliest survivor has `pos ≥ n_sink`; the
old absolute test would have protected *nothing* and let the (now-)sink be masked out. For a fresh
sequence at pos 0, `min_pos==0` ⇒ the boosted set is **byte-identical** to the old behaviour.

### 3. The serving-scenario test — and what it actually found

The whole point: force a real context-shift (llama-cli infinite-gen path: `seq_rm` then
`seq_add(-n_discard)` → `has_shift`) on GLM-5.2-IQ2_M with the indexer mask biting, and confirm
post-shift coherence. **Result: a RoPE context-shift on this model is REFUSED BY THE ENGINE.**
`llama_kv_cache_update_internal` calls `get_can_shift()`, which returns **false for all MLA models**
(`is_mla_model()` includes `GLM_DSA`); the update returns 1 and `llama_decode` propagates it as
`main : failed to eval`. Empirically reproduced and **isolated with a dense control**:

| Config (`-c 256 -n 400`, prompt overflows ctx, shift fires) | At the context-shift |
|---|---|
| Indexer ON (`-mla 3 -fa 1`) | coherent up to the shift, then **`failed to eval`** |
| Dense (`DSA_INDEXER_DISABLE=1`), same prompt/seed | coherent up to the shift, then **`failed to eval`** (identical) |

The failure is **pre-existing MLA engine behaviour, independent of the indexer** — dense fails the same
way at the same token. The indexer neither causes nor worsens it; on the MLA path the shift simply never
happens, so the indexer's kr_l can never desync via K-shift. The `build_k_shift` kr_l block is therefore
**correct-and-dormant**: it is never reached on the current MLA path (guarded by `get_can_shift`), but
keeps the indexer keys consistent the instant MLA K-shift is ever enabled. This is documented loudly in
the code (build-context.cpp k-shift block) and is the honest serving-correctness status, not a paper-over.

**Serving implication (R740/llama-swap):** GLM-5.2 (being MLA) cannot do an in-place RoPE context-shift
at all — n_ctx must be sized to the workload, or the server must truncate/restart on overflow, exactly
as for any MLA model (DeepSeek-V3, etc.). This is orthogonal to and predates the indexer.

### 4. Validation

- **No regression (single-seq, mask is a no-op):** c512 n_seq=1, 4 chunks, **indexer ON == dense ==
  `2.1957 +/- 0.12031`**, byte-identical every chunk (2.2770/2.8741/2.3956/2.1957) — re-run post-change,
  matches the UPDATE 5 baseline exactly. The kr_l/shift/sink changes are a true no-op when no
  shift/defrag fires and the sequence starts at pos 0.
- **Multi-seq (mask bites):** c4096 n_seq=2 healthy (see table below), confirming the per-seq sink
  change did not regress the UPDATE 5 multi-seq fix.
- **Serving shift:** characterized as engine-refused for MLA (above), with a dense control proving
  indexer-independence.
- **Adversarial review:** independent reviewer (did not write the code) verified the K-shift math
  (Hadamard inverse, RoPE composability, param match), aliasing safety, inp_dsa_hadamard reuse, defrag
  shape/budget, seq-op metadata-only claim, and per-seq sink fill — verdict **GO**, no correctness
  defect in the diff; its #1 must-do (run the real shift) is what surfaced the MLA `get_can_shift` gate.
- **Build:** clean (`llama-cli`, `llama-perplexity`, sm_60).

### Honest serving-correctness status

| Serving path | Reachable on MLA GLM-DSA? | kr_l correct? |
|---|---|---|
| prefill + decode (single & multi-seq) | yes | yes — validated, byte-identical no-op vs dense |
| seq_rm / seq_cp / seq_keep (multi-turn) | yes (metadata-only) | yes — kr_l rows stay matched to cells; per-seq sink re-anchors |
| defrag | yes (when `do_defrag` fires) | yes — kr_l row-move mirrors k_l; wired + budgeted |
| RoPE context-shift (K-shift) | **NO — engine `get_can_shift` refuses MLA** | wiring correct but **dormant/unreachable**; dense fails identically |

The indexer is now **serving-general for every path the MLA engine actually executes** (prefill, decode,
multi-turn seq-ops, defrag, multi-seq), not merely perplexity-general. The only path the review worried
about, RoPE K-shift, is not an MLA path at all (refused upstream, dense and indexer alike); its kr_l
wiring is in place and correct for the day MLA shift is enabled.

---

## Status summary

| Capability | Status |
|---|---|
| Single-seq prefill, `-fa 0` | DONE, validated |
| Single-seq decode, `-fa 0` | DONE, validated (persistent kr_l cache) |
| Single-seq prefill+decode, `-fa 1` (`-mla 1` and `-mla 3`) | **DONE, validated (UPDATE 4)** |
| c512 exact no-op vs dense (no regression) | DONE (2.0854 == dense, byte-identical) |
| Hadamard rotation, attention-sink force-include | DONE (gated, on by default) |
| Multi-seq (n_seq>1) with active mask | **DONE, validated (UPDATE 5)** — per-sequence sink; n_seq=4==n_seq=1 |
| Serving: kr_l maintained on defrag + seq-ops; per-seq sink on first-present pos | **DONE (UPDATE 6)** — defrag row-move + seq-ops metadata-only; c512 ON==dense byte-identical |
| Serving: RoPE context-shift (K-shift) on MLA | **ENGINE-GATED OFF for all MLA** (`get_can_shift`); kr_l wiring correct-but-dormant; dense fails identically (UPDATE 6) |
| deepseek32 arch | N/A in this fork (DSA lives under glm-dsa) |

**Bottom line:** the indexer is feature-complete, sequence-correct, and serving-general for **every path
the MLA engine actually executes** — prefill + decode, soft_max + flash-attention, `-mla 1`/`-mla 3`,
n_seq>=1, multi-turn seq-ops, and defrag — on the R740 serving target. UPDATE 5 closed multi-seq
(per-sequence sink: n_seq=2 c4096 61.2→3.07, n_seq=4==n_seq=1). UPDATE 6 closed serving-correctness:
`kr_l` is now maintained by defrag (row-move) and stays matched across the metadata-only seq-ops, the
attention sink anchors on each sequence's first-present pos (correct after multi-turn `seq_rm`), and
single-seq is **byte-identical to dense (2.1957, no regression)**. The one path the review worried about,
RoPE **context-shift**, is **refused by the engine for all MLA models** (`get_can_shift`) — a dense
control fails identically, proving it pre-existing and indexer-independent; the `kr_l` K-shift wiring is
in place and correct but dormant until/unless MLA K-shift is enabled. Remaining ceilings are
operational, not correctness: VRAM for very large packed batches (n_seq=4 at full c4096 OOMs the 3x P100
rig), and MLA's inability to in-place context-shift (size n_ctx to the workload, as for any MLA model).

---

## UPDATE 7 — latent graph-reuse cache-fixup bug for `kr_l` (found via MiniMax MSA), FIXED (2026-06-27)

While porting the indexer-cache work to MiniMax-M3 MSA, an adversarial review of the MSA path found a
graph-reuse cache-fixup omission. The **same class of bug exists here**: the persistent indexer-key cache
write (`kr_l`) is a bare `ggml_cpy` whose destination view bakes `kv_head` at graph-build time, and it was
**never registered in `update_cache_copies()`**. That function re-points the K/V cache writes to the current
`kv_head` whenever a compute graph is REUSED, but it did not touch the `kr_l` write.

### 7.1 Reachability (why it is a real defect)
`graph_reuse` defaults true (`common/common.h`, `llama.cpp` cparams). `can_reuse_graph()` reuses a graph iff
`kv_self.n == prev->n_kv`. Under **FA the cache pads to 256**, so consecutive single-token decode ubatches
share the same padded `n_kv` and the graph IS reused. With `kr_l` unregistered, the reused graph keeps
writing this ubatch's index keys into the FIRST ubatch's slot; later ubatches never populate their own
recent index-key cells (those cells stay at the allocation-zeroed 0.0), so the indexer scores against stale
keys. Structurally identical to the MSA bug (reference fork commit `133d14c9`).

### 7.2 The fix (mirrors the K/V fixup; same shape as MSA `133d14c9`)
* `src/llama-context.h`: new `std::vector<CacheCopy> dsa_cache_copies;`
* `src/llama.cpp` ctor: `dsa_cache_copies.resize(hparams.n_layer)` (null entries -> no-op when DSA off).
* `src/graphs/build_deepseek2.cpp` (the `kr_l` write): register the `ggml_cpy` as
  `lctx.dsa_cache_copies[il] = { kr_cpy, kr_cache->nb[1] }` (step = one index-key row = `head_size`*F16).
* `src/llama.cpp` `update_cache_copies()`: a new loop re-points each registered cpy's
  `view_offs = kv_self.head * step` and patches `src[1]->data` / `data`, exactly like the K/V loop, with the
  `c.cpy->view_src == kv_self.kr_l[il]` + null/op guard (the MSA fix omitted that guard; included here).
The soft_max / non-DSA paths are byte-identical (soft_max pads to 32 -> `n_kv` changes each ubatch -> never
reuses; and even on reuse the patch reproduces the exact offset a fresh build would bake).

### 7.3 Validation (GLM-5.2-UD-IQ2_M, 3x P100 `-ngl 99 --cpu-moe -t 32`, NO_PINNED, `numactl --interleave`)

**FIRST: a platform-fix prerequisite.** This worktree's `glm-dsa-upstream` was rebased to upstream and **lost
the local R740 P2P-disable patch** (`ggml_cuda_set_peer_access` -> false; fork commit `b78ea479`). On this
Sky Lake-E box GPU P2P DMA is silently corrupt, so WITHOUT that patch every multi-GPU GLM run is garbage
(c512 PPL = 154880 = n_vocab; decode = `!!!!`), indexer ON **or** OFF (dense `DSA_INDEXER_DISABLE=1` is
identically broken; DeepSeek-V2-Lite 3-GPU aborts with an illegal memory access while 1-GPU is clean at PPL
5.4454). The P2P patch was re-applied to the working tree to obtain a working baseline; it is orthogonal to
the `kr_l` fix and belongs in its own commit/flag. **None of the #2040 numbers are reproducible on current
HEAD until that patch is restored.**

With the P2P patch in place:

| config | result | verdict |
|---|---|---|
| c512 `-fa 1 -mla 3` indexer ON (no-op floor) | **2.1983** | == #2040 baseline; healthy build confirmed |
| long-ctx FA decode, 2735-tok recall prompt, `-mla 3 -fa 1` temp0, **reuse ON (default), FIXED** | coherent, recalls "Dr. Mariana Velasquez ... Daniel Okonkwo" verbatim | deep-context recall correct |
| same, **reuse ON (default), UNFIXED** | **also coherent**, same correct deep-context recall | the bug is **LATENT** here |
| ub128 PPL `-fa 1 -mla 3`, reuse ON, UNFIXED (4 chunks) | **1.7239 / 1.8211 / 2.1888 / 2.4517** (Final 2.4517) | healthy, no PPL inflation |

**Honest finding: the bug is real in code but does not OBSERVABLY manifest for GLM-DSA at its configured
`top_k = 2048`.** top_k=2048 is permissive (at a 2735-token prompt it keeps 2048 of ~2735 keys), so even
when reuse leaves some recent indexer-key cells stale, the genuinely-attended recent blocks still clear the
top-k and decode stays coherent. This is unlike MSA, whose tighter selection flipped the top-k and inflated
PPL 9.6 -> ~20. The fix is still correct and necessary (it prevents the latent corruption from biting at any
tighter top_k, longer context, or future serving config), but it is a latent-bug fix here, not a visible
regression fix. The earlier ub128 "nan" seen before the P2P patch was P2P corruption, not this bug.

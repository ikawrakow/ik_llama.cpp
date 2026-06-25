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

## Status summary

| Capability | Status |
|---|---|
| Single-seq prefill, `-fa 0` | DONE, validated |
| Single-seq decode, `-fa 0` | DONE, validated (persistent kr_l cache) |
| Single-seq prefill+decode, `-fa 1` (`-mla 1` and `-mla 3`) | **DONE, validated (UPDATE 4)** |
| c512 exact no-op vs dense (no regression) | DONE (2.0854 == dense, byte-identical) |
| Hadamard rotation, attention-sink force-include | DONE (gated, on by default) |
| Multi-seq (n_seq>1) with active mask | **BROKEN — root-caused, fix deferred** |
| deepseek32 arch | N/A in this fork (DSA lives under glm-dsa) |

**Bottom line:** the indexer is feature-complete and PR-worthy for **single-sequence** serving
(prefill + decode, soft_max + flash-attention, `-mla 1`/`-mla 3`), which is the R740 serving target.
The one real remaining gap for a fully general PR is **multi-sequence batched** support, which is
characterized and root-caused here but requires per-sequence cache/score plumbing not safely
landable in this session.

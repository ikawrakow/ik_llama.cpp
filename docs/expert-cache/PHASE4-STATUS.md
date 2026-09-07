# Phase 4 — Dynamic LRU Expert Cache: STATUS / handoff

> Session handoff for continuing `PHASE4-DYNAMIC-EXPERT-CACHE-PLAN.md` implementation.
> Read that plan first; this doc is the delta: M0 results (done), M1 code (written, in
> validation), debugging findings, and what remains. Repo: `ik_llama.cpp/`, branch
> **`expert-cache`** (created from `expert-trace` @ a1882ce which absorbed the prior WIP:
> THP madvise mmap + windowed-kpool plumbing — commit a1882ce, intentional).

## ⏭ RESUME HERE (checkpoint 2026-09-06)

**Where we are: M1 ✅ · M2 ✅ (validated) · M3a ✅ (sim validated) · M3b ✅ (validated) · M3c ✅ (validated; speed gate FAIL — anomaly explained, staging fix in).**
Branch `expert-cache`: M3c + staged promotion copies committed and pushed (see
`PHASE4-M3C-FINDINGS.md` for the full handoff). Headline numbers at 19.7k
(IK_PRINT_TIMING=1 build): base 7.86 → M3c unstaged 8.11 → **M3c staged 9.98 t/s**;
hit 0.46 @step 100 (stories converged 0.524). TG ≥ 12 t/s gate still FAILS.
- **The hit↔speed anomaly is solved.** Hits always converted (−30.6 ms/step CPU at
  hit 0.44); pageable-source promotion copies clawed ~27 ms/step back via driver
  staging-lock serialization with critical-path input copies (KQ_mask sets 0.04 →
  28.5 ms/step). Fix: pinned staging ring in the copy engine
  (`IK_EXP_CACHE_STAGING_MB` 256, `IK_EXP_CACHE_STAGING_THREADS` 2) → contention
  gone (0.10 ms/step), CUDA splits back to the topology floor.
- **Next attack for the 12 t/s bar:** the +11.5 ms/step cache-topology/classify
  stall on the CUDA side (admission-independent; batched readbacks), then M4's
  H-sweep. Profiling tooling: `run_m3c_profile.sh`, `analyze_m3c_profile.py`,
  `ring_bench.cu`; A/B knobs: `IK_EXP_CACHE_ADMISSION=0`, `IK_EXP_CACHE_STAGING_MB=0`.

## ⏭ older checkpoint (2026-09-02 late)
Branch `expert-cache` @ **249e0f4, pushed to fork** (= M3b: promotion worker; ddfc556 = M3a:
shadow classic-LRU sim — classify stages routed ids at k,ntok≤8,
`llama_expert_cache_step_boundary` applies touches/misses/promotions at TG step
boundaries). Both build trees green at 249e0f4.
- **M3b validation (2026-09-02)**: promotion worker live and proven. Smoke clean
  (queue step 2 → publish step 3, 167 graphs, no CUDA errors, launch-blocking+timeout).
  1.1k snap A/B × 2 (`run_m3b_snap.sh`, `analyze_m3b.py`; snaps parsed **order='F'** —
  ne0-fastest!): forced promotion `IK_EXP_CACHE_FORCE_PROMOTE=il:slot:expert:step` —
  (a) expert 29 @step 4: positions ≤ publish bitwise-identical; expert-0 eviction row
  hot→cold at rel p50 **2.25e-2** (exact M2 envelope), 4 more noise-scale positions at
  later expert-0 routings; (b) expert 176 @step 1: first post-publish routing served
  **cold→hot from the copied slot** at rel p50 **2.22e-2** / abs max 1.47e-2 — publish
  side proven. `IK_EXP_CACHE_VERIFY_COPIES=1` readback+memcmp at publish: **bitwise
  PASS** in all runs. No early diffs, no NaN, no CUDA errors anywhere.
- **M3b design (as built)**: `ggml_cuda_copy_engine` (ggml-cuda.h/.cu): dedicated
  non-blocking copy stream per CUDA backend; `sync_compute()` at queue time records on
  the compute stream + waitEvent on the copy stream + immediate event destroy
  (waitEvent snapshots — no event lifetime ties; review caught a fence-retirement race
  in the original split fence_compute/wait design); `h2d()` ×3 slices (8.56 MB) from the
  worker thread; `fence_copy()`/`poll()` with in-order retirement (single stream ⇒ FIFO
  completion). llama.cpp: `expert_cache_state::expert_cache_promoter` (thread + queue +
  inflight FIFO, cribbed from ggml-moe-prefetch pool shape; host-slots mode = plain
  memcpy, complete at issue). Tearing-safe protocol enforced:
  `llama_expert_cache_queue_promotion` (llama.cpp:7096) sets remap[old]=-1 + slot
  PENDING at queue time (hard guards: max 1 pending/layer, slot < 64, src host-resident,
  dst slice size == src); classify free-slot scan excludes pending slots (llama.cpp:6898,
  warn gated to k≤8 — k=288 warmup always exhausts, benign); publish
  (remap[new]=slot) only after the copy fence completes, polled at the next TG boundary
  by `llama_expert_cache_publish` (llama.cpp:7186). All policy state touched by the
  decode thread only; worker touches queue/pending FIFOs under mtx.
- **M3 next: M3c** — wire admission (2nd miss in window, N=64) + `--expert-cache-promote-gbps`
  cap (default 8) + global in-flight cap (4) + PP read-only. Admission/cap ordering
  DECIDED: **the routing stream is the queue** — promote-at-miss-if-budget, miss count
  saturates (no drop, no explicit queue); see `PHASE4-M3C-ADMISSION.md` (for review).
  The forced-promotion hook + verify-copies stay as debug tooling.
- **Bisect builds DONE (bash-12, green)** but `bash run_bisect.sh` still not run
  (sequential; needs ~100 GiB-free machine). Still blocks M4's absolute bars only.
- Older checkpoints (M1/M2 evidence chain, freeze RCAs, key code facts) below.

## Motivation (30 s version)

GLM-5.3-Flash 321B IQ3_XXS: TG speed is bounded by reading ~2.9 GB of expert weights per
token from host RAM (42 MoE layers × 8 experts × 8.56 MB). Phase 0 showed routing has strong
**within-session temporal locality** but no cross-workload stability → only a **dynamic
per-session LRU cache** of expert slices in VRAM works. Target: keep H≈32 slices/layer
resident (12.3 GB, replacing today's 5-layer static banks), get ~40–45% of expert reads from
VRAM → TG 10.16 → 14–16 t/s. Traffic model TG ≈ 9.01/(1−hit) validated.

## M0 — feasibility gates: DONE (all four)

Artifacts: `pcie_bench.cu`/`pcie_bench`, `run_m0.sh`, `diff_traces.py`,
`logs/m0-{parity-cpu,parity-gpu5,nographs}.log`, `traces/m0-parity-{cpu,gpu5}.bin`.

1. **Numeric parity (CPU iqk vs CUDA mmvq on same IQ3_XXS slice): NOT bitwise.**
   Ran champion-style configs A (all experts CPU) vs B (#12: layers 39–43 experts on CUDA0),
   greedy 128 TG tokens on the 19.7k creature prompt, `--expert-trace` both:
   - Every token's router weights differ downstream of the first GPU layer (p50 6e-4,
     p99 6e-2, max 0.66 abs — driven by near-tie top-k order/set flips; 24,136/830k records
     have id-order or 8th-expert set differences).
   - **BUT all 19,782 token ids incl. all 128 generated greedy tokens match; TG text
     byte-identical.** So GPU≠CPU bitwise (the plan's "PP traces were byte-identical"
     parenthetical was wrong — that had been a CPU-vs-CPU comparison), yet greedy output is
     robust.
   - **Decision (validation standard)**: cache-ON cannot be trace-byte-identical by
     construction → use the plan's fallback: greedy text diff over the battery + logit
     max-abs-diff budget + trace id-set agreement modulo near-ties. H=0/off stays
     byte-exact always. NOTE: M1 (all-CPU hot path) *is* bitwise-exact vs baseline by
     construction (see "bitwise-exact tail" below) — so the M1 gates can still demand
     byte-identical traces.
2. **PCIe microbenchmark: 26.2 GB/s sustained** pinned HtoD on a dedicated non-blocking copy
   stream with 8.6 MB slot-sized chunks (idle GPU), **26.15 GB/s while the compute stream is
   saturated** (spin kernel, 2048×256) — copy engine is independent of SMs, contention ≈ 0.
   Pageable memory halves it (14.4 GB/s) → pinned host RAM mandatory (champion already pins).
   Single-slice burst latency 0.53 ms, amortized 0.34 ms/slice at 16. DtoH same speed.
   PCIe link idles at gen1 but ramps to gen4x16 under load (verified via nvidia-smi polling).
   → promotion-cap default `--expert-cache-promote-gbps 8` ≈ 30% of realized bandwidth: sane.
3. **Capture-break cost (GGML_CUDA_DISABLE_GRAPHS=1, champion #12 @19.7k): TG 8.76 vs 10.16
   (−13.8%), PP 68.98 vs 77.3 (−10.7%).** Worst-case price if the classify step kills CUDA
   graph capture. (Logs confirm 0 vs 75 captured graphs.) The chosen classify mechanism
   (eval-callback range-cutting) may be cheaper than full graphs-off — real number lands in
   M2/M3 gates. If intolerable → strategy (b) in-graph remap (plan §masked-execution).
4. **--expert-cache-off guard**: all new code is behind `--expert-cache*` flags; H=0 or no
   flag ⇒ no slot tensors, no graph change, no callback ⇒ byte-identical by construction.
   (Verified implicitly: baseline node count 4183 unchanged when off.)

## M1 — load-time split + two-path masked execution, all-CPU: **GATES PASSED 2026-09-02** (hybrid pair byte-identical; CPU pair byte-identical with F1/6097707 — see root-cause section)

### What's implemented (branch `expert-cache`, builds green: build-novlk + build-cuda)

- **Flags** (`common/common.{h,cpp}` → `include/llama.h` params → `src/llama-cparams.h`):
  `--expert-cache H` (slots/layer), `--expert-cache-gb B` (budget→H resolved at load from
  real tensor sizes), `--expert-cache-promote-gbps R` (cap, plumbed, used in M3).
  Also `llama_model_params.expert_cache_{h,gb}`, defaults wired in
  `llama_model/context_default_params` (positional init lists — keep order!).
- **Slot allocation** (`llama_model::load_tensors`, after repack/modify passes, before
  overrides freed): for every layer with separate `ffn_{up,gate,down}_exps` (merged
  `ffn_up_gate_exps` variant skipped), creates `ffn_*_exps_hot` tensors `[ne0, ne1, H+1]`
  same type, one pinned host buffer (`llama_default_buffer_type_cpu(true)`; M2 → CUDA0),
  `GGML_BACKEND_BUFFER_USAGE_WEIGHTS`, registered in `model.ctxs/bufs/tensors_by_name`.
  Fill: slot s ← expert s (s<H), trash slot H ← expert 0 (finite garbage); slice = contiguous
  `memcpy` (expert = last dim, `nb[2]` stride; host src) or `ggml_backend_tensor_get`.
  Guards: xor of H/gb flags; incompatible with `-rtr` repack (warn+disable); clamps
  H ≤ n_expert−1; no eligible layers → warn+disable. H=32 × 42 layers = 11.6 GiB pinned.
- **Model/context fields**: `llama_model.expert_cache_{h,gb}`; `llama_layer.ffn_*_exps_hot`;
  `llama_context::expert_cache_state` (per-layer `remap[288]`, `slot_expert[H+1]`, current
  graph input tensor ptrs) as `lctx.expert_cache`; `chained_eval_cb/ud` + `expert_cache_pending`.
- **Two-path graph** (`llm_build_moe_ffn`, llama-build-context.cpp ~1593-1685): when
  `lctx.expert_cache` has the layer and the config matches the champion variant (separate
  up/gate, `-fmoe`, SILU/GELU, no biases, no `down_exps_s`, not llama4-weight-before),
  emits per layer: input tensors `hot_ids`/`cold_ids` (I32 [8, n]) + `hot_mask` (F32 [8, n])
  (`ggml_set_input`, named `ffn_exp_cache_*-N`, pointers stashed in the layer state), then
  hot `ggml_moe_up_gate`+`mul_mat_id` on slot tensors, cold same on full tensors,
  **bitwise-exact tail**: `mask3d` multiply on `experts_hot` (1.0/0.0 — neutral), then
  `experts = add(experts_hot, experts_cold)` — exactly one side nonzero per (j, token), so
  the sum equals the single path's `experts` **bit for bit** — then the single-path tail
  verbatim (`ggml_mul_multi_add` when `-fmmad` (default on!), else `mul`+`multi_add`),
  `add_input` last. Any config mismatch → silent single path.
- **Classify callback** (`llama_expert_cache_eval_cb` + `_dispatch_cb`, llama.cpp ~6670-6800):
  hooks the sched eval callback (`ggml_backend_sched_eval` range-cutting: ask=true → need=1
  on `ffn_moe_topk-N` ⇒ compute range, **backend sync**, ask=false → ids valid). Reads ids
  pitch-aware (like the tracer), writes per-column `hot_ids = slot | -1 (CPU) / H (CUDA
  trash)`, `cold_ids = -1 | id`, `hot_mask = 1|0`. Writes direct-memcpy for host-resident
  input tensors, per-column `ggml_backend_tensor_set` otherwise. Chains any previous
  `cparams.cb_eval` (incl. tracer's IK_EXPVERIFY snap, which was re-pointed at the chain).
  Context init creates state (static dummy: remap[e]=e for e<H) and installs the dispatcher
  **after** `cparams.cb_eval = params.cb_eval` (see bug #2). Debug: `IK_EXP_CACHE_DEBUG=1`
  prints firings + cumulative hit rate every 4200 classifies.

### Bugs found & fixed during bring-up

1. **`fused_mmad` defaults to TRUE** — the single-path tail is `ggml_mul_multi_add`
   (fused FMA chain), so my first mul+`multi_add` two-path tail could never be bitwise.
   Fixed by the add-before-weight structure above (same op, same values, same order).
2. **Callback stomped**: my init installed the dispatcher into `cparams.cb_eval`, but
   `cparams.cb_eval = params.cb_eval` runs ~60 lines *later* in context init → callback
   never fired → masks stayed garbage → "Nth Nth Nth…" degenerate output. Fixed via
   `expert_cache_pending` + install after the params copy. Lesson: with no classify the
   masks are garbage — NOT graceful. (M3 note: failure modes must degrade to *slow*, not
   wrong — consider initializing cold_ids=ids-passthrough semantics impossible w/o readback;
   instead the reserve-graph compute + first real classify always precede consumption, so
   this is safe by construction once installed correctly.)
3. **Name collision**: TWO graph nodes are named exactly `ffn_moe_topk-N`: the
   `ggml_argsort` node is `ffn_moe_probs_biased-N (sort)`, `ggml_top_k`'s [8] view gets
   renamed by `cb` — but a second VIEW node of the sort result with **ne[0]=288** also
   carries `ffn_moe_topk-N` and fires the callback **once per run during the first
   (warmup/reserve-graph) pass**, before the [8] one. The shape guard
   (`t->ne[0] == hot_ids->ne[0]` i.e. 8) skips it safely. OPEN: if it ever fires mid-decode
   it costs a wasted sync/layer; consider gating ask=true on the shape check (needs
   hot_ids already set — true after first build). Root cause of the [288] view's existence
   not fully pinned down (likely reserve/warmup graph artifact) — revisit if it recurs.

### Validation state

- Smoke tests (tiny prompt): cache engages (nodes 4183→4393, splits 86→170, CUDA graphs
  still captured: "have 85 graphs"), classify fires per layer/step, host-resident mask
  writes, coherent text. The earlier degenerate output was bug #2, not the graph math.
- **Hybrid gates: PASS (2026-09-01 evening, H=8, build 55 = 90e34b5+M2+offload-fix)**:
  - `traces/m1-base.bin` vs `traces/m1-cache8.bin`: **byte-identical** (53,333,452 B each).
  - TG text: **identical** (both traced and clean pairs).
  - Perf (clean pair): TG 7.35 vs 7.41 → ratio **0.992 ≥ 0.98** PASS; PP 59.71 vs 65.15
    (−8.3%, dual CPU matmul traversal — acceptable for M1; hot path leaves CPU in M2).
  - Cache-on captures 168 CUDA graph segments (vs 85 base) — capture survives the classify
    range-cutting; TG cost ≤ 0.8%.
  - Meta-scan found 43 candidates; layer 45 (MTP tail) correctly dropped at tensor-creation
    re-verification → 42 cached layers, as designed.
- **Pure-CPU pair: TEXT identical, traces NOT byte-identical** (same size 53,333,452 B):
  divergence onset **token 17408 (= 34×512, exactly an ubatch boundary; 88% through PP),
  layer 27**; 0 final-token mismatches; TG text identical; 47,378 routed-id order/set flips
  (near-tie top-k signature, 8th-slot swaps), 64,453 weight mismatches p50 5.9e-3.
  **Anatomy** (diff_traces.py): first mismatch is a *weight* diff at (17408, L27); layers
  3–26 bitwise-equal at that token ⇒ divergence introduced by **block-26** (attn-26 or
  MoE-26) for that token; later tokens mismatch at progressively *earlier* layers (global
  min L8) ⇒ **re-triggering at multiple (token,layer) points**, not pure attention cascade.
  Token 17408's L26 ids are `[214 56 232 125 63 106 49 206]` — **all ≥ 8, i.e. all-cold**,
  so its hot path is fully masked (`mask=0`) and inert by construction.
  **Verdicts so far (2026-09-01 late eve)**:
  - `m1-cpu-base` exact rerun (job bash-3): **BASELINE SELF-IDENTICAL** — the pure-CPU
    config is deterministic; the pair divergence is real and cache-induced.
  - `m1-cpu-cache8` exact rerun (run_narrow.sh run 1): **IDENTICAL** — the cache-path
    divergence is **deterministic**, not a race.
  - run 2 (cache8 **without** `--prefetch-experts`): **STILL DIVERGES, same byte offset
    (46,938,289)**; noprefetch trace == prefetch trace byte-for-byte ⇒ fully deterministic,
    prefetch inert (confirmed empirically; engine does POPULATE_READ/WILLNEED/MADV_COLD only).
  - **EXPVERIFY run (IK_EXPVERIFY=1, cache8+trace, novlk): trace byte-identical to the
    original cache8 trace; id overlap vs compute-time probs = clean 8.000/8 at ubatches ≥ 34**
    (30 scattered single-column 7.998/8 drops across ALL ubatches incl. 2–33 = known near-tie
    recompute noise, no clustering at 34). ⇒ **traced ids are faithful; the compute genuinely
    diverges (deterministic).** Trace-artifact theory DEAD.
  - Forensic: B's divergent (17408, L27) weight vector is a valid normalized router output
    (sums to 2.5 like all records) but matches NO other record in either trace (nearest:
    A's own value 2.6e-3 away, next 10× farther) ⇒ not a stale-copy artifact either.
  - **Localization**: attention-26 exonerated (router-26 input is post-attention and bitwise
    equal) ⇒ divergence lives downstream of ffn_norm-26: MoE-26 two-path / ffn_shexp /
    mHC-post. Token 17408 @L26 is all-cold (ids [214 56 232 125 63 106 49 206], all ≥ 8) so
    its hot path is masked-inert; the cold path is mathematically forced-bitwise (verified
    kernel-by-kernel: zero-fill `-1`, expert-grouped IQK `iqk_mul_mat_moe`/`_fused_up_gate`
    per-(expert,chunk) determinism, Ny-dependent dispatch identical for groups with identical
    counts) ⇒ contradiction ⇒ the flaw is inside some kernel's detail not visible from
    orchestration code. **Empirical localization DONE (bash-7)**: IK_DUMP_TENSORS
    instrumentation (env-gated exact-name node dumps, OUTPUT-flagged at build, read back at
    tracer-record site; run_dump26.sh + diff_dumps.py) dumped the MoE-26 chain at ubatches
    33–35 for base vs cache; both traces reproduced byte-identical under the dump flags.
  - **DUMP RESULTS (m1d-cpu-{base,cache8}, traces/dump-{base,cache}-26.bin)**:
    - **Ubatch 33: the two-path math is PERFECT.** cold vs base diffs are exclusively
      B==0.0 at hit columns (masked by construction: 83 hit columns at gate_par-26);
      the ADD result ("ffn_moe_down-26") is BITWISE-EQUAL to baseline experts;
      ffn_shexp/ffn_out bitwise-equal. The mechanism is correct when fed correct inputs.
    - **Ubatch 34: global perturbation.** ffn_norm-26, gate_par_cold, down_cold, shexp,
      ffn_out all differ for ~all 512 tokens with close-but-different values (rel p50
      3.7e-2 norm / 1.1e-1 gate_par; zero-frac ≈ 0 — genuine numeric perturbation, NOT
      wrong-expert swaps and NOT stale-column copies).
    - **Trace per-ubatch map** (ubatch_map.py): ubatches 32–33 fully clean; ubatch 34
      mismatched-token counts per layer 20..27 = [56,56,491,507,507,507,510,512];
      first-mismatch layer histogram = {8:1, 16:54, 17:1, 22:435, 23:16, 26:3, 27:2}.
      ⇒ poison enters at MoE outputs of layers 7/15/16/21/22/25/26 for different tokens
      (first visible router = poisoned MoE layer + 1), i.e. a per-(token,layer) lottery
      that switches on EXACTLY at ubatch 34 and ramps with layer (~0% L3, 10% L15,
      85% L21, ~100% L26). Ubatch 35+: all tokens all layers.
    - **masked_hot dump anomaly**: dumped "ffn_moe_masked_hot-26" is nonzero at ALL 4096
      (j,c) positions including true-miss columns — even at ubatch 33 where the ADD result
      is bitwise-equal. Classify code verified (llama.cpp:6855-6885): mask = hit?1:0,
      hot = hit?slot:(cuda?trash:-1), cold = hit?-1:id — so live masked_hot MUST be zero
      at miss columns. ⇒ **the dumped masked_hot content is unreliable (buffer overwritten
      in-graph before end-of-graph readback despite OUTPUT flag, or a double-named node)**.
      Same caveat applies to the ffn_norm-26 dump: token 17408's traced L26 router output
      is bitwise-equal while dumped ffn_norm-26 col 0 differs — a deterministic router
      cannot do that, so dumped ffn_norm-26 ≠ what the router consumed (overwritten
      in-graph, e.g. mHC/mixer aliasing). TRUSTWORTHY dumps (validated by the exact
      ubatch-33 pattern): gate_par_cold, down_cold (op intermediates consumed immediately).
      Input tensors (hot_ids/cold_ids/hot_mask) would also be trustworthy end-of-graph
      (only the callback writes them) — NOT YET DUMPED.
    - Reconciled picture: mid-graph ffn_norm-26 genuinely differs for ~510/512 tokens of
      ubatch 34 (trace L26 weights differ for 510), token 17408's L26 output survived
      bitwise by luck of a tiny perturbation. The two-path math per se is exonerated
      (ubatch 33); something perturbs the HIDDEN STATES entering MoE layers 21+ (and
      sporadically 7/15/16) precisely from ubatch 34 on.
  - **2026-09-02: ROOT CAUSE FOUND — compute-time MoE-router fusion asymmetry, not the
  - **2026-09-02 (final): ROOT CAUSE = compute-time glm45 router-fusion asymmetry,
    broken BY THE CLASSIFY CALLBACK'S RANGE CUT, plus a residual nonlocal
    perturbation (M2) still under dissection.** Measured directly with an
    IK_LOG_FUSION firing counter (1-token runs, `run_fusiontest.sh`):
    **base graph fuses 126x (42 layers x 3 computes); cache graph fuses 0x.**
    The glm45 pattern [SIGMOID, RESHAPE, ADD, ARGSORT, VIEW, GET_ROWS]
    (ggml.c:26950) fires only if all six nodes sit in ONE scheduler range; the
    classify callback cut at the `ffn_moe_topk-N` VIEW ends its range before
    GET_ROWS, so every cache run computes the router UNFUSED. Unfused selection
    scores = scalar `expf` sigmoid; fused = AVX512 `v_expf` (~1 ULP off) ->
    near-tie top-8 orderings flip (e.g. token 17864/L8: e71=7.4874253 vs
    e258=7.4874249 -- unfused orders 71-first (cache trace), fused orders
    258-first (base trace)). The first flips land at ubatch 34 (token 17408+)
    simply because that's where the data's first near-ties are. Weights are
    scalar `1/(1+expf(-x))` in BOTH paths (bitwise-equal given same ids), so
    fused-vs-unfused changes ONLY near-tie selections.
  - The failed kill-switch test, explained: `IK_DISABLE_ROUTER_FUSION=1`
    (3fdeec7) cache run came out byte-identical to the fused cache run because
    cache runs never fuse anyway (the callback cut, not node order, is what
    breaks the pattern -- the cache graph's uncut node order matches the
    pattern just like base). The earlier "DFS-order asymmetry" story was wrong.
  - snap4 evidence chain: cutting L8's router chain in a BASE run (snap
    callback) reproduces the cache run's divergence from the original base
    EXACTLY (same 2374 mismatch tokens, same first cell 17408/L27), while cuts
    at non-router nodes (snap3: down/out/norm chain) leave base numerics
    untouched (== orig base). orig-cache == snap4-cache bitwise. And within the
    snap window the L8-cut base run is trace-identical to the full cache run --
    fused-vs-unfused flips at L9+ first appear at 17920 (L29).
  - **Residual mechanism M2 (OPEN, snap5 in flight):** the first divergence
    cell (17408, L27) is a WEIGHTS-only ~1e-3-relative diff with IDENTICAL ids
    -- impossible from router fused-vs-unfused (weights are bitwise given same
    ids+input), so L27's router INPUT differs, i.e. block-27 attention output
    differs with bitwise-clean input, nonlocally triggered by cutting L8's
    router chain (snap4-base shows it; snap3-base with non-router cuts does
    not). Prime suspect: a scratch (params->wdata) read-before-write in the
    MLA flash-attention or KDA recurrence path whose stale content depends on
    range structure. snap5 (`run_snap5.sh`) localizes it: safe-cut snaps of
    ffn_norm-26 -> l_out-26 -> hc_attn_pre-27 -> kqv_out-27 -> ffn_norm-27 at
    ubatch 34, base vs cache.
  - **FIX F1 (6097707) + M1 CPU GATE: BYTE-IDENTICAL PASS.** classify trigger
    moved from `ffn_moe_topk-N` to `ffn_moe_weights-N` (the router get_rows;
    topk view read via `t->src[1]`). The cut lands AFTER the whole pattern, so
    the fusion stays alive in cache runs -- VERIFIED: cache run with F1 fuses
    378x over 9 computes (42 layers each), output coherent. The full 19.7k PP
    + 128 TG cache8+trace run (`run_m1f1.sh`, traces/m1f1-cpu-cache8.bin) is
    BYTE-IDENTICAL to traces/m1-cpu-base.bin. Conclusion: BOTH the near-tie id
    flips AND the snap5 "M2" perturbation (MoE-26 token-17408 j1) were the
    fused-vs-unfused router asymmetry; the MoE-26 cell was a consumed-id flip
    the end-of-graph trace readback could not see (trace = live-faithful at
    most cells, but NOT guaranteed -- treat end-of-graph topk readbacks as
    advisory). Perf (traced runs): F1 TG 2.80/PP 25.86 vs old-trigger cache
    TG 2.71/PP 25.16 vs base TG 2.67/PP 25.52 -- F1 slightly faster than the
    old trigger (fused router back on).
  - **snap5 verdict (M2 localization):** the ubatch-34 perturbation is BORN in
    the routed MoE-26 compute: at token 17408, ffn_norm-26 row 0 bitwise-clean,
    shared-expert output clean, but ffn_moe_down-26 row 1 (token 17408, j=1)
    differs at rel 1.6e-2; everything downstream (l_out-26 7e-5, kqv_out-27
    2.7e-2, ffn_norm-27 1.4e-2, router-27 weights 1e-3) is its echo. The
    trace's "(17408,L27) weights" first cell was merely the first
    router-visible echo. Open question the F1 gate answers: was the MoE-26/j1
    difference a consumed-id near-tie flip the end-of-graph trace could not
    see (fusion mechanism; F1 fixes it), or a genuine two-path kernel
    difference (F1 won't)?
  - **ANALYSIS FOOTGUN (cost a day)**: numpy reshape(n0,n1,n2) is C-order (last axis
    fastest) but ggml ne0 is FASTEST-varying -- 3D snap/dump records must be parsed as
    rows-of-n0 (or order='F'). All whole-tensor equality verdicts were
    permutation-invariant and remain valid; all per-slot hit/miss correlation analyses
    from .tmp/consistency_test.py / l8_forensics.py / stale_test.py were scrambled and
    are DISCARDED (their "masked_hot nonzero at miss" alarm was an artifact).
  - **Latent engine bug found & fixed**: `biased_sigmoid` 5-arg scalar tail read
    uninitialized y[i] (`y[i] = y[i] + bias[i]` -> now `z[i] + bias[i]`); never triggers
    for n_expert%16==0 (GLM-5.3: 288=18x16), would bite other models.
  - Superseded narrowing narrative (kept for the tooling caveats): end-of-graph dumps are
    UNRELIABLE for late-reused buffers (masked_hot garbage, dumped topk ≠ traced ids —
    dump channel broken, NOT the trace); the sched-eval compute-time snapshot channel
    (llama_snap_cb, IK_SNAP_*) is clobber-immune and was the decisive tool; DSA pooling
    is off in all M1 runs (no --dsa → kp_l/windowed-pooling class dead). The trace's
    fine-grained attribution turned out FAITHFUL after all (its L8/token-17864 id swap
    matched the snap-validated cold-op output swap bitwise); mid-hunt doubts about it
    came from the broken dump channel and the numpy-axis scramble.
  - **Exonerated by code inspection + compute-time measurement**: classify callback
    (static dummy remap), `-1` zero-fill in both CPU kernels (ggml.c:18286-18295
    mul_mat_id, 18614-18624 fused up_gate), expert-grouped IQK chunking
    per-(expert,chunk) deterministic, sched range-cutting + sync identical across
    configs, identical build flags, two-path masked-add exactness (ub33 bitwise-perfect:
    hot dense only at hits, cold == base at all misses, ADD == base 4096/4096),
    wdata/work-size plan-vs-runtime match for MUL_MAT_ID and MOE_FUSED_UP_GATE
    (ggml.c:28949-28977 vs runtime), DSA/qsa paths (dead).

## 2026-09-01 evening: box freeze incident + RCA

**Event**: first cache32 clean run (`--expert-cache 32`, host slots) → machine fully froze
(desktop, harness), hard reboot required. Two stacked causes:

1. **REAL M1 BUG — MoE offload theft**: ik's `ggml_backend_cuda_offload_op`
   (ggml-cuda.cu:5200) offloads `MUL_MAT_ID`/`MOE_FUSED_UP_GATE` to GPU when
   `batch×n_active ≥ min_batch×n_experts_tot`. Full expert tensors (288) never trigger it at
   ubatch 512 (512×8 < 32×288); the hot slot tensors have only H+1=33 "experts" → PP batches
   ≥ ~132 tokens get stolen onto CUDA **with weights in host-pinned RAM and `ids` in an
   unpinned sched host tensor** → kernel dereferences unpinned host pointer → **illegal
   memory access** in `launch_mul_mat_q_id` → abort. Smoke tests (2-token PP) and `-ngl 0`
   runs never hit it. **Fix**: `ggml_backend_cuda_offload_op` returns false when
   `src0->name` contains `_exps.hot` (slot residency is cache-managed; device-resident M2
   slots reach CUDA via the weights rule, bypassing offload entirely).
2. **Amplifier**: `ulimit -c unlimited` + `core_pattern=|systemd-coredump` → the ~120 GiB
   pinned process tried to dump core → total meltdown. Plus the raw RAM math: 105.3 pinned
   experts + 11.6 slots = 120/125 GiB, no headroom for desktop/harness.

**New standing protocol** (in `run_m1.sh`, reuse for future runners): `ulimit -c 0`;
gates at H=8 (3.2 GiB slots; gates are H-independent); MemAvailable ≥ 100 GiB guard before
every run; never compile during cache-bearing runs.

## Baseline-regression finding (matters for M4's absolute bars, NOT for M1's ratio gate)

Same #9-style command (`-ngl 99 -ot ffn_.*_exps=CPU`), same workload, greedy:

| binary | config | graphs | TG t/s | PP t/s |
|---|---|---|---|---|
| BENCHMARKS-era (#9 row) | #9 | 75-era | **9.01** | 69.7 |
| 06069ec (+trace) | #9 | 85 | 7.71 | 66.3 |
| 06069ec (+trace) | #12 banks | 75 | 8.75 | 73.1 |
| 06069ec (graphs OFF) | #12 banks | 0 | 8.76 | 69.0 |
| a1882ce+M1 (clean) | #9 | 85 | **6.25** | 65.6 |

- The tree regressed #9-config TG by ~31% vs the BENCHMARKS table (cache OFF — not the
  cache code). Suspects by delta: tracer commits (`llama_expert_trace_mark_outputs`
  flag-chains topk/weights tensors as OUTPUT on **every** graph build, no early-out — 85 vs
  75 capture segments) and a1882ce (mmap anon-THP path + glm5next DSA windowing; DSA inactive
  without `--dsa` here, so suspect the mmap/kpool parts).
- **M0.3's "graphs-off costs −13.8%" was cross-binary contaminated**: same-binary #12-style
  shows graphs on/off = 8.75 vs 8.76 ≈ free. M0.3 conclusion superseded; the true
  capture-break price on the current tree is ~0 for the champion shape.
- TODO (after M1 gates): bisect with a `git worktree` build of c4b2232 (pre-trace) and
  a1882ce-only, same #9 command, sequential. Until resolved, treat BENCHMARKS.md rows as
  measured on an older binary; M4's "TG ≥ 14 vs champion 10.16" bar may need re-baselining.
- **Bisect staged (2026-09-01 eve)**: worktrees `ik_bisect/` (c4b2232) + `ik_bisect2/`
  (06069ec) created; build dirs configured (needs `-DCUDAToolkit_ROOT=$CUDA_PATH
  -DCMAKE_CUDA_ARCHITECTURES=86 -DGGML_CUDA=ON -DGGML_VULKAN=OFF -DGGML_RPC=OFF
  -DCMAKE_BUILD_TYPE=Release` and `CPATH=$CUDA_PATH/include` or nvcc's compiler-id test
  fails on cuda_runtime.h). `run_bisect.sh` ready. Remaining: build both (ccache-warm),
  run `bash run_bisect.sh` sequentially.
- Extra data point: on build 55, the TRACED #9 run (m1-base, 8.17) is ~10% FASTER than the
  clean one (m1-base-clean, 7.41) — trace-on beating clean is anomalous; feeds the
  mark_outputs/graph-shape suspicion. Bisect should confirm or refute.

## M2 — hot slots in VRAM: **DONE 2026-09-02** (F3 b617ae0 + F4/F5 f15ae25 + F6 654a057; validated below)

**2026-09-02 freeze #2 + RCA + F3:** the first M2 smoke (CUDA-resident slots) wedged the
box at the first PP ubatch: `CUDA error: an illegal memory access` right after the L3
classify (`EXP_CACHE L3 ntok=5 k=8 ... host=111` in logs/m2-smoke.log). Root cause: the
three graph inputs (`ffn_exp_cache_hot_ids/hot_mask/cold_ids-N`) are INPUT-flagged →
ggml-backend.cpp:1334 hardcodes graph inputs to the LAST (CPU) backend, and per-backend
copies refresh only at SPLIT STARTS. With slots on CUDA0, the hot MoE ops sit in the SAME
CUDA split as the router (attention→router→hot-fork, no backend change between them), so
the split-start copy of hot_ids/hot_mask happened BEFORE the classify's mid-graph write →
CUDA hot ops gathered with uninitialized/stale ids → OOB gather → fault → GPU wedge → box
freeze (same failure class as freeze #1, new vector). Warmup survived only because fresh
buffers were zero-filled. Fix **F3 (b617ae0, both trees green; smoke PASSED 2026-09-02: slots CUDA0, classify
host=001 = hot inputs device/cold host, coherent text, no CUDA errors)**: new sched flag
`ggml_backend_sched_set_expert_cache_cuda_inputs` (ggml-backend.h/.cpp) set from llama at
ctx init when `expert_cache->slots_on_cuda` (llama.cpp:10156 area, both sched_new sites);
the placement rule special-cases `ffn_exp_cache_hot_ids-*`/`ffn_exp_cache_hot_mask-*`
inputs onto backend 0 (CUDA0) so hot ops read the ORIGINAL (no stale copy) and the
classify writes them via its existing `ggml_backend_tensor_set` (non-host) path mid-graph,
post-sync, stream-ordered. `cold_ids` stays host (cold ops are CPU). M1/host-slot runs
unaffected (flag false). Also learned: the warmup graph's `ffn_moe_weights-N` is
[1,288,1] (src1 = the 288-wide argsort parent — TG-variant router shape); the classify
shape guard handled it (k=288 == warmup hot_ids width; real PP/TG graphs are k=8 — M1 TG
perf proves it). Retest protocol agreed with user: CUDA_LAUNCH_BLOCKING=1 + timeout watchdog, user
accepts residual wedge risk.

**19.7k A/B #1 (b617ae0): base clean (PP 67.10 / TG 7.52); cache run ABORTED at first PP
ubatch** (clean SIGABRT, NO wedge — F3 changed the failure mode). Repro: 1.5k-token prompt
faults identically (`run_m2_repro.sh`, logs/m2-repro.log). Bug hunt on the hot-op PP path
(9-slot weights ⇒ 512 tokens > 32×9 ⇒ the GENERAL moe_up_gate path that upstream never
exercises at PP — real models always take mmq_id since 512 ≤ 32×288):
- **F4 (in tree, uncommitted): three row-count mis-sized pool allocs** — the general path
  sizes src1 buffers by ne12 (token count) but a single expert slot can hold n_ids×ne12
  rows (trash slot gets ~everything with the static dummy remap): src1_quantized
  (:3398→rows×n_ids), src1_contiguous (:3405→×n_ids), and the same in the standalone
  mul_mat_id general path (:2989→×n_ids).
- **F5 (in tree, uncommitted): `final_dst_contiguous.alloc(ggml_nelements(next))` —
  pool_alloc<char> takes BYTES, so it got 16.7MB for 16.7M floats (67MB needed)** —
  slots ≤36 rows fit silently; the trash slot's ~3956 rows (64.8MB) blew past → the
  scatter kernel (k_copy_dst_from_contiguous) read off the pool → illegal access.
  Fix: alloc(sizeof(float)*nelements). Verified: fault moved PAST the scatter.
- **F6 (654a057, THE crasher)**: compute-sanitizer memcheck named it: wild gather in
  quantize_mmq_q8_1 (quantize_id.cu:38) reading garbage ids_src1. Root cause:
  launch_mmq_ids_helper sizes its shared-memory pair store by n_tokens (valid for
  distinct top-k: <=1 (expert,token) pair per expert), but the cache's CUDA miss
  sentinel routed EVERY miss to the trash slot -> up to k*ntok pairs in one slot ->
  shared-store overflow -> garbage ids_src1. Fix: classify assigns misses DISTINCT
  free slots per token (bitmask; masked to zero downstream, so semantics unchanged;
  trash slot = fallback only for the >32-wide warmup graph). mmvq TG path never used
  the helper, which is why smoke/warmup passed and only PP crashed.
- Commits f15ae25 (F4+F5) + 654a057 (F6) pushed to fork. NOTE: the sanitizer's
  --destroy-on-device-error RELAUNCHED the payload mid-death -> two 100+ GiB
  processes coexisted (1 GiB free) — when killing sanitizer runs, kill the whole
  PID tree and verify MemAvailable before trusting job_kill.
- **19.7k A/B #2 (all fixes): BOTH RUNS CLEAN.** PP 64.30 vs 68.41 (0.94), TG 7.94 vs
  8.36 (0.95) at H=8 all-miss (pure overhead, no benefit expected). TG text diverges
  at token ~1 (19.7k) / token ~1-2 (1.1k) — under M0.1's standard (GPU!=CPU bitwise,
  greedy text over battery) this needs quantification: snap-based ubatch-0 L3 MoE
  decomposition comparison in flight (`run_m2snap.sh` + .tmp/m2snap_cmp.py): miss rows
  must be bitwise vs baseline (same CPU cold path), hit rows small-rel (kernel swap).
- **NUMERICS VALIDATED (snap @ ubatch-0 L3, traces/m2snap-{base,cache}.bin)**: 2191/2264
  down rows bitwise-equal (= all miss pairs, CPU cold path identical); the 73 differing
  rows are exactly the ~3.2% HIT pairs (H=8 ⇒ 8/288) with a tight uniform norm-rel diff
  p50 2.3e-2 / max 2.8e-2 (q8_1-vs-iqk activation-quantizer noise; cosine 1.0, norms
  match). masked_hot zero rows = 2191 exactly (mask exact); hot+cold == output ✓.
  ⟹ CUDA hot path correct; 19.7k token-1 text flip = accumulated quantizer chaos within
  the M0.1 budget (documented standard: GPU≠CPU bitwise; battery text-diff + this
  decomposition evidence). M2 sign-off: functional + validated; perf at H=8 static dummy
  = pure overhead (TG 0.95×, PP 0.94×) — benefit arrives with M3's real hit rate.


- `llama-model.h`: `expert_cache_layers` (meta-resolved candidates), `expert_cache_device`
  (index into `devices[]`, −1 = host), `expert_cache_bytes` (for the carve).
- `llm_load_tensors` early block (before `device_mem` accounting): eligibility from GGUF
  meta (`get_tensor_meta`, separate up/gate/down, no merged, uniform expert count), gb→H
  resolution + clamp, `expert_cache_bytes` = (H+1)×bytes/slot, placement = `devices[0]`
  buft when `n_gpu_layers>0 && devices non-empty && !IK_EXP_CACHE_HOST`, then **carve**
  `device_mem[dev] -= bytes` so auto-fit/splits accounting sees it; falls back to host
  slots with a warning if it can't fit. New incompat guards: `--merge-up-gate-exps`,
  `defer_experts` (both warn+disable).
- Late block (end of load): re-verifies per-layer eligibility against created tensors,
  allocs slot buffer on the chosen buft, fills device-aware (`ggml_backend_tensor_set` per
  slice for device dst; handles non-host src via scratch bounce).
- Context init: `slots_on_cuda` derived from the actual buffer (not hardcoded); classify
  already writes trash-slot `H` (vs `-1`) on miss when `slots_on_cuda`.
- Graph: **unchanged** — hot ops follow their WEIGHTS-usage tensors to CUDA0
  (ggml-backend.cpp:1347), and ik's CUDA backend supports all ops used
  (`MOE_FUSED_UP_GATE` fast-TG mmvq ggml-cuda.cu:3083, mmq_id PP path :3248,
  `MUL_MULTI_ADD` multiadd.cu — all list IQ3_XXS). Trash slot covers CUDA's missing
  bounds check. ids/mask host inputs cross via split input copies (to verify in smoke).
- Offload-theft guard (above) also protects the M1/host mode going forward.
- NEXT: commit after M1 gates pass; smoke (`-c 2048 -n 8`, watch "have N graphs",
  IK_EXP_CACHE_DEBUG host=... lines), then #9-style 19.7k A/B (cache8 on CUDA0 vs
  base) for parity (text diff) + capture status, then M2 perf read.
- OPEN: `ggml_backend_cuda_offload_op`'s heuristic could still steal *legit* CPU-expert
  MoE ops at ubatch ≥ 2048 with unpinned ids — pre-existing ik hazard, not ours; noted.


### Command/env cheatsheet

```bash
# builds (both green): ccache'd, ~1-4 min incremental
export CCACHE_DIR=/home/bepis/prog/llm-tests/.ccache NIX_ENFORCE_NO_NATIVE=0 \
       TMPDIR=/home/bepis/prog/llm-tests/.tmp   # stale /tmp/nix-shell-* breaks nvcc/gpp temps
cmake --build ik_llama.cpp/build-novlk -j32 --target llama-cli
export LD_LIBRARY_PATH=/nix/store/pp1xkyx8s2i28x38ipp0775z6llqy9gj-cuda-merged-12.8/lib:/run/opengl-driver/lib
cmake --build ik_llama.cpp/build-cuda  -j32 --target llama-cli
# CUDA runs need that LD_LIBRARY_PATH; python/numpy needs zlib:
export LD_LIBRARY_PATH=/nix/store/78x9i5x1wpqw4kq0h39b8f35abcv156h-zlib-1.3.2/lib:$LD_LIBRARY_PATH
# nvcc: /nix/store/pp1xkyx8s2i28x38ipp0775z6llqy9gj-cuda-merged-12.8/bin/nvcc -I<cuda>/include
# nvcc with system gcc: worked out of the box for sm_86
# GPU: RTX 3090 24 GiB, idle between runs; check nvidia-smi before benching.
# RAM: one hybrid run pins ~105 GiB; NEVER overlap two model runs (also contamination rule).
# /tmp is a FRESH tmpfs per bash call in the sandbox — put scratch scripts in .tmp/ (workspace).
# Dump tooling: IK_DUMP_TENSORS="exact-node-name,..." + IK_DUMP_POS=<token-pos> (±512 ubatch
#   window) + IK_DUMP_FILE=<path>; nodes are OUTPUT-flagged at build (mark_outputs) and read
#   back post-sync at the tracer-record site. CAVEAT: tensors overwritten later in-graph
#   (e.g. "ffn_norm-N", "ffn_moe_masked_hot-N") read back clobbered — only op intermediates
#   consumed immediately (gate_par/down) and graph-input tensors are trustworthy.
#   Analyze: diff_dumps.py (fixed token-axis: dim 2 for 3D, dim 1 for 2D), .tmp/mask_audit.py,
#   .tmp/ubatch_map.py (recreate from history if wiped).
# text extraction from logs: awk '/^generate: n_ctx/{found=1; next} /^main: prompt eval time/{exit} found'
```

## What remains (plan milestones)

- **M1 gates** (above) → hybrid PASS; CPU pair: baseline deterministic, cache divergence
  deterministic, prefetch-inert, traced ids EXPVERIFY-faithful, two-path math proven
  perfect at ubatch 33 (dump); poison enters at scattered (token,layer) MoE outputs from
  ubatch 34 on (histogram {8:1, 16:54, 17:1, 22:435, 23:16, 26:3, 27:2}).
  **Next run (decisive)**: dump the INPUT tensors + layer-21 chain at ubatch 34 —
  IK_DUMP_TENSORS="ffn_moe_logits-21,ffn_moe_topk-21,ffn_moe_gate_par_cold-21,
  ffn_moe_down_cold-21,ffn_moe_masked_hot-21" PLUS the graph-input tensors — need their
  exact node names first (grep llama-build-context.cpp for where cl.hot_ids/cold_ids/
  hot_mask enter the graph — likely "hot_ids-21" etc.); also add a one-line n_win/qsa_
  pooled_stale log per graph build to verify ubatch 34 isn't a wide-variant rebuild.
  Input-tensor dumps are trustworthy end-of-graph (only the callback writes them).
  If cold_ids-21@ub34 ≠ base ids-21 → callback/readback bug; if equal → compute/kernel bug
  (then dump layer-7 chain for the single L8-first token and compare per-expert groups).
- **Commits**: 6183a49 M1+M2+offload-theft fix; 8d32aa7 debug instrumentation (env-gated,
  off by default); 3fdeec7 router-fusion kill-switch + biased_sigmoid tail fix; **6097707
  F1 (classify trigger at ffn_moe_weights-N) — the M1 CPU gate fix**. Both build trees
  (build-novlk, build-cuda) at 6097707, green.
- **M1 — DONE 2026-09-02 (all gates green on 6097707)**: CPU pair byte-identical
  (traces/m1f1-cpu-cache8.bin == traces/m1-cpu-base.bin) + TG text identical; hybrid pair
  byte-identical WITH F1 (traces/m1f1-cache8.bin == traces/m1f1-base.bin) + TG text
  identical + traced TG ratio 0.985 (8.48 vs 8.61 t/s; PP 62.15 vs 67.40 traced).
- **M2** (hot slots → CUDA0, static content): committed in 6183a49, build-cuda at 6097707,
  runtime-untested. NEXT: smoke (`-c 2048 -n 8`, no IK_EXP_CACHE_HOST → "target GPU",
  buffer name CUDA0, graphs count, coherent text), then 19.7k A/B `m2-cache8cuda` vs a
  FRESH same-binary (6097707) `m2-base-clean` baseline (the old m1-base-clean was a
  different binary) → greedy text diff + perf.
- **Baseline-regression bisect** (not a milestone, blocks M4 absolute bars): build
  `ik_bisect/` + `ik_bisect2/`, `bash run_bisect.sh` (sequential), attribute the #9-config
  9.01→6.25/7.41 regression to tracer commits vs a1882ce. NOTE: with the fusion root cause
  now known, part of the "regression" might be fusion-related graph-order changes from the
  tracer's OUTPUT flagging — worth one extra bisect data point: current tree, cache off,
  with/without --expert-trace (the tracer flags change allocator layout AND the 85-vs-168
  graph-count anomaly is still unexplained).
- **M3** (dynamic LRU + promotion) — DESIGN NOTES (2026-09-02, pre-implementation):
  - Mutation protocol: remap/LRU/pending mutated ONLY at TG step boundaries (a hook after
    the step's compute in llama_decode_internal); the classify stays read-only w.r.t.
    remap and collects per-layer miss counts + hit touches into staging applied at the
    boundary. PP (n_tokens>8): fully read-only (no promotion/eviction).
  - Eviction/publish safety: at queue time, remap[old_expert]=-1 IMMEDIATELY (old goes
    cold) and the slot is marked pending; the copy stream (event-ordered after the
    boundary's compute) then overwrites the slot; remap[new]=s only after the copy event
    completes. **Pending slots are excluded from F6's free-slot assignment** — compute
    reading a slot mid-overwrite could produce NaN/Inf and 0×NaN=NaN poisons the sum.
    Max 1 pending slot per layer (k=8 needs ≤8 free of 9 slots).
  - Promotion worker: 1 thread + dedicated CUDA copy stream; crib ggml-moe-prefetch.cpp
    pool shape + only_active_experts tensor_set_async; rate cap --expert-cache-promote-gbps
    (default 8 ≈ 30% of the measured 26.2 GB/s); admission = promote on 2nd miss in window.
  - Gates: hit rate 0.40–0.45 @H=32 stories (Phase-0 prediction), TG ≥ 12 t/s @19.7k.
  Original plan text:
  step boundaries; promotion thread + **dedicated CUDA copy stream** (infra:
  `common.cuh:71/862/879`, `ggml_backend_cuda_event_*` ggml-cuda.cu:5224-5258; crib
  `only_active_experts` sched path ggml-backend.cpp:2052-2143 for mid-compute
  readback/`tensor_set_async`, and `ggml-moe-prefetch.cpp:58-167` for thread-pool shape);
  admission filter (promote on 2nd miss in window) + `--expert-cache-promote-gbps` cap +
  PP read-only (n_tokens>8: no promotion/eviction); `IK_EXP_CACHE_DEBUG` counters exist;
  add `IK_EXP_CACHE_VERIFY=1` both-paths-compare mode. Gates: hit rate ≈ Phase-0 prediction
  (0.40–0.45 @H=32 stories), TG ≥ 12 t/s @19.7k.
- **M4**: H sweep {16,24,32,43} @19.7k/58k, admission A/B, pick default, update
  `BENCHMARKS.md`/`HOT-EXPERT-REPACK-PLAN.md`/plan doc; success bar TG ≥ 14 t/s @19.7k.
- MTP tail MoE: skipped v1 (no hot tensors for it; document).

## Key code facts learned this session (beyond the plan's digest)

- Trace bin format: hdr `IKEXP001` + n_layers + 42×(layer,n_expert=288,n_topk=8); then per
  ubatch: n_tokens, (token_id,pos)×n, then n×42×{ids[8] i32, weights[8] f32} (router weights,
  NOT expert outputs). `diff_traces.py` compares two bins (ids/weights/phase breakdown).
- CPU fused `moe_up_gate` **does** skip `id<0` (zeroes dst row + skips grouping;
  ggml.c:18579-18600) — `-1` masking works natively on both cold ops on CPU.
- `ggml_backend_sched_eval` with `callback_eval` set: computes node ranges, `need==1` ⇒
  `ggml_backend_synchronize` + ask=false with valid data; **flagged node gets recomputed**
  once (idempotent ops only!); works per-split, CPU+CUDA. This is the classify hook.
- `only_active_experts` (auto-enabled when experts host-resident; saw "enabling
  only_active_experts scheduling" in logs) already does ids-readback + async HtoD of active
  expert ranges per split — the closest precedent and proof the readback timing works.
- `fused_mmad` (`-fmmad`) and `fused_moe_up_gate` (`-fmoe`) both default ON; single-path
  MoE tail = `ggml_mul_multi_add` (FMA chain) — any two-path restructure must feed it
  bitwise-identical inputs (add-before-weight) to keep byte-parity on CPU.
- `llama_context_params.cb_eval` is copied into cparams LATE in context init (llama.cpp
  ~9404) — anything installing into cparams must go after it (or be re-stomped).
- `load_all_data` asserts every ctx tensor exists in the GGUF → synthetic slot tensors must
  live in their own ggml_context + buffer (done), registered post-load.
- GGUF: separate `ffn_{up,gate,down}_exps` (no merged `ffn_gate_up_exps`); per-expert slice
  2.5625+2.5625+3.4375 = 8.5625 MB; 288 experts, top-8, 42 MoE layers (L3–L44).
- Smoke-test regime that catches everything fast: `-c 2048 -n 8 -p "hi"` (3 min load).

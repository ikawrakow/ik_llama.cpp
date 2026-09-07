# Phase 4 / M3 Implementation Plan — Dynamic LRU Expert Cache + Promotion

Status: M0/M1/M2 done (see PHASE4-STATUS.md for the full evidence chain). M3 makes the
cache live: slots start as the static dummy (experts 0..H-1) and adapt at TG step
boundaries via an LRU policy with rate-capped PCIe promotion.

## Proven foundations (do not re-litigate)

- Two-path masked MoE is bitwise-exact on CPU (M1 gate: byte-identical, 6097707/F1).
- CUDA-resident slots work (M2: F3 placement + F4/F5 general-path alloc fixes + F6
  distinct-free-slot invariant; snap-validated: cold path bitwise, hot rows = uniform
  q8-vs-iqk quantizer noise p50 2.3e-2, mask exact).
- Classify callback: fires at `ffn_moe_weights-N` (after the router fusion pattern —
  never cut inside it again), writes hot_ids/cold_ids/hot_mask mid-graph, post-sync.
- PCIe: 26.2 GB/s pinned HtoD on a dedicated copy stream, contention ≈ 0 with compute.
- GPU-path bring-up protocol (two freezes + one near-miss): CUDA_LAUNCH_BLOCKING=1 +
  timeout + IK_* debug env on first runs of any new device path; kill by PID tree and
  verify MemAvailable + nvidia-smi before declaring a GPU job dead (the sanitizer
  relaunch incident); expect "clean SIGABRT" to be the failure mode of remaining bugs.

## Design (locked decisions)

1. **Mutation only at TG step boundaries.** A hook after the step's compute (in
   llama_decode_internal) applies staged changes: LRU touches, admission decisions,
   eviction/promotion queueing, publish-on-event-complete. The classify callback stays
   read-only w.r.t. remap; it stages per-layer miss counts + hit slot touches.
   PP (n_tokens > 8): fully read-only — no promotion, no eviction, no LRU mutation.
2. **Eviction/publish protocol (tearing-safe):** at queue time remap[old]=-1 (old expert
   goes cold immediately) and the slot becomes PENDING; the copy thread (dedicated CUDA
   copy stream, event-ordered after the boundary's compute stream point) overwrites the
   slot's 3 slices (up/gate/down, 8.56 MB total); remap[new]=s + LRU insert only after
   the copy event completes (polled at the next boundary). **Pending slots are excluded
   from F6's free-slot assignment** — compute reading a mid-overwrite slot can yield
   NaN/Inf and 0×NaN=NaN poisons the MoE sum. Max 1 pending slot per layer (k=8 ids need
   ≤8 free of H+1=9 slots at H=8; at H=32 there's slack).
3. **Admission filter:** promote an expert on its 2nd miss within a rolling window
   (per-layer miss-count map, window = N TG steps, start N=64); rate cap
   `--expert-cache-promote-gbps` (default 8 ≈ 30% of 26.2 GB/s realized); also a global
   in-flight cap (start: 4 copies).
4. **Victim selection:** LRU among non-pending slots in that layer.

## Work breakdown (in order)

- [x] M3a. State + boundary hook (no promotion yet): per-layer `lru` list,
      `miss_count[288]`/window bookkeeping, staged hit-touches, `IK_EXP_CACHE_DEBUG`
      counters (hit rate per layer + global, printed every 4200 classify calls — exists;
      extend with promotion counters). Validate: counters vs Phase-0 predictions on the
      stories workload at H=32 (STATIC content — the counters measure what the hit rate
      WOULD be; zero behavior change). Pure CPU-side code — no GPU risk.
      **DONE 2026-09-02** (uncommitted diff: llama-context.h +20, llama.cpp +140; shadow
      classic-LRU sim staged by classify at k,ntok≤8, applied at TG boundaries).
      Validated: smoke clean (167 graphs, no CUDA errors); 1.1k A/B vs pre-M3a binary TG
      text BYTE-IDENTICAL; stories H=32 (lighthouse s43): sim hit converged **0.5375** —
      above the 0.40–0.45 gate (sequential LRU tracks intra-session drift, beating
      Phase-0's static oracle 0.44–0.48; per-layer shape matches: L3–6 ≈ 0.15–0.18,
      L18–27 ≈ 0.65–0.75). Promo telemetry: ~160 promo/step ≈ 12 GB/s @8.8 t/s classic —
      the 8 GB/s cap will bind; adm2 (2nd-miss filter) ≈ 113/step ≈ 8.5 GB/s.
- [x] M3b. Promotion worker: thread + copy stream + events (crib
      ggml-moe-prefetch.cpp:58-167 pool shape; event infra common.cuh:71/862/879,
      ggml-cuda.cu:5224-5258; HtoD via tensor_set_async on the copy stream).
      Host-slots mode: plain memcpy publish (also useful for CPU-only configs).
      Validate: single forced promotion on the tiny repro (1.1k prompt), snap L3 MoE
      before/after to prove the new expert's rows match the cold-path values bitwise-
      mod-kernel (same test shape as the M2 snap validation).
      **DONE 2026-09-02** (ggml-cuda.h +22, ggml-cuda.cu +101, llama-context.h +14,
      llama.cpp +~300): `ggml_cuda_copy_engine` = dedicated non-blocking copy stream +
      fence/poll; boundary-time `sync_compute()` orders copies after the just-submitted
      step's compute (review caught+fixed a fence-retirement race in the initial
      split fence/wait design). Promoter thread (prefetch-pool shape), tearing-safe
      queue (remap[old]=-1 + slot PENDING at queue time, pending excluded from the
      classify free-slot scan, publish remap[new]=slot on copy-fence completion at the
      next boundary; hard guards: max 1 pending/layer, slot < 64). Host-slots mode:
      memcpy publish. Validation hooks: `IK_EXP_CACHE_FORCE_PROMOTE=il:slot:expert:step`
      (-1 = self-select first staged miss), `IK_EXP_CACHE_VERIFY_COPIES=1` (readback +
      memcmp at publish). Validated: smoke clean (queue step 2 → publish step 3, 167
      graphs, no CUDA errors); 1.1k snap A/B ×2 (`run_m3b_snap.sh`, `analyze_m3b.py` —
      snaps parsed order='F'): eviction direction (expert 0 hot→cold) and publish
      direction (expert 176 cold→hot from copied content) both diff at exactly the M2
      quantizer-noise envelope (rel p50 2.2e-2, abs max ≤1.6e-2), everything else
      bitwise-identical; VERIFY_COPIES bitwise PASS in all runs.
- [x] M3c. Wire admission + caps + PP read-only; pending-aware free-slot exclusion in
      the classify (F6 site). Validate: 19.7k A/B vs m2-base-clean — greedy text budget
      per M0.1 + hit-rate trace.
      **DONE 2026-09-06** (committed + pushed; handoff: `PHASE4-M3C-FINDINGS.md`).
      Hit rate PASS (stories 0.524 ≥ 0.40–0.45); text budget PASS per M0.1 (early
      coherent divergence = M2's documented quantizer noise); **TG gate FAIL** —
      anomaly traced to pageable promotion copies serializing with input copies;
      staged-copy fix (pinned ring + split memcpy) lands 8.11 → 9.98 t/s at 19.7k
      (timing build, base 7.86). Remaining headroom before M4: +11.5 ms/step
      topology/classify stall.
- [ ] M3d. IK_EXP_CACHE_VERIFY=1 (both-paths compare) if needed for debugging;
      snap-based tests may suffice — decide when we get there.
- [ ] Gates: hit rate 0.40–0.45 @H=32 stories; TG ≥ 12 t/s @19.7k (champion 10.16).

## Test ladder for any new GPU-path code (hard rule)

1. build-novlk compile check (fast) → 2. build-cuda compile → 3. tiny smoke
   (-c 2048 -n 8) with CUDA_LAUNCH_BLOCKING=1 + timeout → 4. 1.1k repro A/B →
   5. 19.7k A/B. Never skip 3. Sequential model runs only; MemAvailable ≥ 100 GiB guard;
   ulimit -c 0; no compiles while a cache-bearing run is in flight.

## Open threads carried into M3

- Baseline TG regression (9.01 → 7.4–8.6 #9-config): bisect trees ik_bisect (c4b2232),
  ik_bisect2 (06069ec) — CPATH fix applied, builds in flight at checkpoint; then
  `bash run_bisect.sh`. Blocks M4's absolute bars only.
- Unexplained: graph count 85 (base) vs 168 (cache) — benign so far, revisit if M3 perf
  is off. m2 smoke showed "have 167 graphs" (capture active).
- Warmup graph quirk: ffn_moe_weights-N is [1,288,1] there (TG-variant router); the
  classify shape-guards it. Benign.

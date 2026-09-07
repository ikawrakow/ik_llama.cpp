# Phase 4 / M3c — implementation findings & handoff (2026-09-06)

Status: M3c **implemented, reviewed, functionally validated, committed**; the
speed gate **failed**. Follow-ups done 2026-09-06 (items 1–5 below): the
anomaly is explained (hits convert to CPU savings but pageable promotion
copies clawed them back via driver staging-lock contention on critical-path
input copies) and the copy-path fix is in — **staged promotions: 8.11 → 9.98
t/s at 19.7k** (timing build; base 7.86). TG ≥ 12 t/s still fails; the next
attack is the +11.5 ms/step cache-topology/classify stall. Everything below
is reproduced from `logs/m3c-*.log` (`run_m3c_gates.sh`) and
`logs/prof-19k-*.log` (`run_m3c_profile.sh`, `run_m3c_staged_ab.sh`).

## What M3c is

Live admission for the dynamic LRU expert cache (spec:
`PHASE4-M3C-ADMISSION.md`, option C "the routing stream is the queue"). The
cache now adapts at TG step boundaries: 2nd-miss-in-window (N=64) admission
filter, wall-time token bucket (`--expert-cache-promote-gbps`, default 8 GB/s,
burst = 2 steps ≈ 1.8 GB, first boundary seeds one burst), in-flight cap 128,
per-layer pending cap H+1−k, candidates sorted hottest-first into an ephemeral
scratch array, deferral telemetry split by reason. PP stays read-only.

Uncommitted diff in `ik_llama.cpp` (on top of M3b commit 249e0f4):
`src/llama.cpp`, `src/llama-context.h`, one comment touch in
`ggml/src/ggml-cuda.cu`. Builds: `build-novlk` + `build-cuda` green, zero new
warnings. **Build gotcha that cost us one misleading "green": stale TMPDIR
(`/tmp/nix-shell-*`) makes nvcc fail while make falls back; always build with
`TMPDIR=/tmp` and verify `libllama.so` mtime, not the exit code.**

## Code review (3 independent reviewers + author pass)

Fixed before validation:

- **Blocker:** the M3c-relaxed pending cap (`pending ≤ H+1−k`) could starve the
  classify free-slot scan, which was hard-capped at 32 slots (`uint32` mask) —
  at H=32 up to 25 pending ⇒ < 8 free ⇒ `1u << trash_slot` UB + duplicate-slot
  assignment. Fixed by widening the scan to the 64-wide `pending_mask` domain
  (`uint64_t used`, `s < 64`, guarded fallback) and clamping H ≤ 63 at load.
- **Concurrency hole:** worker pop→pending-push window made a mid-copy job
  invisible to `inflight()`/`expert_in_flight()` ⇒ same-expert double queue,
  `remap`↔`slot_expert` desync. Fixed with sentinel-fence registration
  (`EXP_CACHE_FENCE_PENDING`) in the same critical section as the pop; publish
  checks the sentinel before polling (poll returns true for *unknown* fences).
- **Bookkeeping:** `miss_count ← 0` on promotion left stale deque entries that
  would eat post-reset misses at window expiry. Fixed: erase the id from all
  `miss_log` entries at reset (count ≡ log content).
- `IK_EXP_CACHE_WINDOW ≤ 0` → empty-deque UB → clamped to ≥ 1 (also fixes the
  identical pre-existing sim loop).
- LRU touch no longer appends unknown slots (trash-slot H via FORCE hook).
- Telemetry: added `qfail` / `awaiting` counters; budget prints `uncapped` when
  the cap is off.

Accepted (documented, no code change): recency tiebreak is inert (all
candidates just missed this step — the count key does the ordering); effective
window is N+1-ish by the shared push-then-expire idiom (M3a sim identical →
comparisons stay apples-to-apples); FORCE-hook LRU staleness (validation-only);
`hit_count` survives eviction (only consumed by the future victim-aware
follow-on — remember this when building it); H ≤ 7 configs silently disable
admission (safe, invariant unsatisfiable there).

## Validation ladder results

Run chain `run_m3c_gates.sh` (sequential, memory-guarded), all runs exit 0,
zero CUDA-error lines:

| stage | result |
|---|---|
| smoke H=8 (`run_m3c_smoke.sh`) | 549/549 VERIFY_COPIES bitwise PASS; deterministic; `qfail=0 awaiting=0` |
| 1.1k A/B H=32 | 3219/3219 VERIFY_COPIES bitwise PASS; TG text cmp **inconclusive** (compare script artifact, see below) |
| stories H=32 (`story-lighthouse`, seed 43, `-c 8192 -n 3072`) | live hit **0.524** converged (sim classic-LRU ref 0.5415); 153k promotions; TG **9.04 t/s** |
| 19.7k A/B (creature-run-py, `-n 128`) | base TG **7.79 t/s** → M3c TG **8.09 t/s** (+3.9%); PP 68.02 → 67.14 (−1.3%) |

**Gates: hit rate 0.524 ≥ 0.40–0.45 PASS. TG ≥ 12 t/s @19.7k FAIL (8.09).**

### The central finding: hit rate does not convert to speed

The plan's model `TG ≈ 9.01/(1−hit)` predicts ~18 t/s at hit=0.52. Observed:
9.04 t/s on stories (~+3% vs the ~8.8 t/s M3a-era story number). So the hot
path saves only a few percent of TG time per hit. This is not an admission
artifact:

- `deferred=0/0/0` everywhere — the 8 GB/s bucket never bound
  (`budget` stayed ~1–1.2 GB, `inflight_hw=127` rides under the 128 cap). Per
  the admission doc's own framing, this closes the "which cap binds" question
  for these workloads: **none**.
- M2 measured all-miss cache-on overhead 0.95×; we now know the inverse
  (hit ⇒ savings) is small, so most TG time is outside the CPU cold expert
  path — or promotion/copy/classify overheads eat the savings.

Known confounders to quantify before touching policy again:

1. Promotion traffic: stories ran ~72 promotions/step ≈ 0.6 GB HtoD per step
   (contention measured ≈0 at M0, but that was isolated PCIe, not vs CPU
   cold-path DRAM reads).
2. `IK_EXP_CACHE_VERIFY_COPIES` was ON for the 1.1k run only — stories/19.7k
   were clean, so readback sync is not the explanation.
3. The 19.7k text compare ~~is currently broken~~ **was fixed 2026-09-06**
   (see What's-next item 1): the reported "first diff" had been an EXP_CACHE
   log line, not model text; the real answer is genuine early divergence,
   within the M0.1 budget per M2's signed-off numerics evidence.

## What's next (ordered)

1. **Fix the text compare** — **DONE 2026-09-06.** Root cause was not a
   prefix-filter gap but the tail-of-lines approach itself: async EXP_CACHE_*
   telemetry is injected mid-text (splitting generated lines) and floods the
   log tail. `run_m3c_gates.sh` now anchors on the sampler `generate:` line and
   compares the reconstructed generation stream (telemetry segments removed
   incl. their newline, which rejoins split text fragments). Re-diff of the
   existing logs: **text NOT identical** — 1.1k diverges at stream char 7
   (~token 2), 19.7k at char 2; both continuations coherent. This matches M2's
   signed-off token~1 flip (q8_1-vs-iqk quantizer noise, snap-validated), so
   the M0.1 standard (GPU≠CPU bitwise; battery text-diff + decomposition
   evidence) is met — the earlier "inconclusive" was purely the artifact.
2. **Profile one TG step** — **DONE 2026-09-06** (`run_m3c_profile.sh` →
   `logs/prof-19k-{base,admoff,admon}.log`, `analyze_m3c_profile.py`;
   IK_PRINT_TIMING=1 in `llama.cpp` + `ggml-backend.cpp`, the latter needed
   two dead `tim1` sites fixed to compile; **the working tree still has
   timing ON — flip both defines back to 0 before the M3c commit**).
   Per-TG-step graph_compute (timing build, ms; CUDA splits / CPU splits):
   base 127.3 (12.1 / 114.4) · admoff 125.5 (23.6 / 101.1) · admon 123.1
   (50.5 / 70.5). **The Phase-4 premise holds: the CPU cold expert path is
   ~90% of the base TG step at 19.7k** (114/127 ms); attention/KV is not
   dominant. prelude/set_inputs/get_result are all < 0.4 ms.
3. **Isolate promotion overhead from hit savings** — **DONE 2026-09-06, and it
   explains the hit↔speed anomaly.** Hits DO convert: admon's CPU cold path is
   −30.6 ms/step vs admoff at hit≈0.44 (still converging: 76 promotions/step,
   7607 total = 5.5× the 1376 slots — churn; deferred 0/4/0, budget never
   binds). But +26.9 ms/step comes back on the CUDA side, signature:
   `set CUDA0#KQ_mask` 0.04 → **28.5 ms/step** (131 calls, 3.6 s over 128
   steps). Promotions already use a dedicated copy stream + fences
   (`ggml_cuda_copy_engine`), so the contention is not stream sharing — it is
   **pageable-source `cudaMemcpyAsync`**: expert slices come from the mmap'd
   GGUF (pageable), pageable HtoD degenerates to staged/synchronous copies
   behind a driver staging lock, and the main thread's input copies serialize
   behind ~0.68 GB/step of promotion traffic (≈28 ms at the ~21-24 GB/s
   pageable ceiling — matches exactly). Net policy effect: −30.6 + 26.9 ≈
   **−4 ms/step (+3.3%)**, matching the observed +3.9% gate A/B. Also
   measured: cache-topology fixed cost (two-path masked MoE + classify) is
   +11.5 ms/step CUDA-side with admission OFF, while the CPU side gets 13 ms
   *cheaper* — admoff is slightly net-positive vs base even at hit≈0.03.
   Direct classify readback cost is negligible (43 × ~11 µs/step).
4. **Premise re-examination: not needed — the premise is confirmed.** The
   follow-up is not policy, it's the copy path: **pin the promotion sources**
   (`cudaHostRegister` on the GGUF expert mapping, or a pinned staging pool in
   the promoter) so copy-engine HtoD is truly async and stops serializing
   with critical-path input copies. Counterfactual: admon without the 26.9 ms
   contention ≈ 96-100 ms/step ≈ **10-10.4 t/s at hit 0.44**, more at the
   converged stories hit 0.52 — but the 12 t/s bar likely also needs the
   +11.5 ms topology/classify stall attacked (batched readbacks) and/or
   higher converged hit at 19.7k. Note 19.7k with `-n 128` is the policy's
   worst case: the cache never converges inside 128 steps.
5. **Pinning fix: IMPLEMENTED + MEASURED 2026-09-06** (`run_m3c_staged_ab.sh`
   → `logs/prof-19k-{staged,staging0,staged2}.log`). Staged promotion copies:
   pinned ring in the copy engine (`ggml_backend_cuda_copy_engine_set_staging`),
   per-chunk memcpy split across N helper threads; env `IK_EXP_CACHE_STAGING_MB`
   (default 256; 0 = legacy path A/B knob), `IK_EXP_CACHE_STAGING_THREADS`
   (default 2, clamp 1–8). Microbench `ring_bench.cu` (pcie_bench discipline):
   memcpy 8.5 / 12.5 / 12.6 / 12.9 GB/s at T=1/2/3/4 (saturates at ~13);
   staged end-to-end 6.5 GB/s single-producer (under pageable's 14.4 — the
   ring buys lock-avoidance, not throughput); contention probe p50 **948 →
   26 µs = idle** (the pathology reproduced and cured in isolation). 19.7k
   A/B (timing build): **unstaged 8.11 → staged 9.92 (T=1) → 9.98 t/s (T=2)**;
   KQ_mask contention 28.5 → 0.10 ms/step; CUDA splits back to the admoff
   floor (~24-25 ms). Worker is memcpy-bound in-app: 28→42 promotions/step at
   T=1→2, hit 0.39→0.46 @step 100; the inflight cap still binds
   (deferred=0/5036/0 at T=2) — demand exceeds supply, the filter re-presents,
   so convergence is slower, not broken. Control arm (STAGING_MB=0) reproduced
   the pre-staging run exactly (hit 0.4377, 7607 promoted, 8.18 t/s) ⇒ the
   staging diff itself is behavior-neutral. **TG ≥ 12 t/s still FAILS (9.98)**;
   remaining known headroom: the +11.5 ms/step cache-topology/classify stall
   (CUDA-side, admission-independent — batched readbacks is the attack), and
   transient-vs-converged hit at 19.7k.
6. **Deferred by design:** M3d (`IK_EXP_CACHE_VERIFY` both-paths compare) —
   only if debugging demands it; victim-aware admission (TinyLFU) — M3c already
   collects the windowed hit counts it needs, but with no cap binding its value
   is unproven; MTP tail MoE — documented out of scope for v1.
7. **Bookkeeping: DONE 2026-09-06** — `PHASE4-STATUS.md` /
   `PHASE4-M3-PLAN.md` updated, reviewed diff committed to `expert-cache`
   (IK_PRINT_TIMING flipped back to 0 in both files first) and pushed to fork.
   (Baseline-regression bisect — `ik_bisect`/`ik_bisect2` + `run_bisect.sh` —
   still open; blocks M4's absolute bars only.)

## Reference: logs & scripts

- `run_m3c_smoke.sh` → `logs/m3c-smoke.log`
- `run_m3c_gates.sh` → `logs/m3c-1k-{base,m3c}.log`, `logs/m3c-story.log`,
  `logs/m3c-19k-{base,m3c}.log`
- `run_m3c_profile.sh` → `logs/prof-19k-{base,admoff,admon}.log` (IK_PRINT_TIMING=1
  build; `analyze_m3c_profile.py` decomposes per-TG-step phases/splits/tensors)
- `run_m3c_staged_ab.sh` → `logs/prof-19k-{staged,staging0}.log`;
  `logs/prof-19k-staged2.log` (2-thread memcpy build); `ring_bench.cu` /
  `ring_bench` (staged-ring microbench: memcpy scaling, staged rate,
  contention probe)
- Telemetry lines: `EXP_CACHE_LIVE` (live policy: hit, promoted/step,
  deferrals budget/inflight/layer, qfail, awaiting, inflight high-water,
  budget) and `EXP_CACHE_SIM` (classic-LRU reference sim, unchanged).

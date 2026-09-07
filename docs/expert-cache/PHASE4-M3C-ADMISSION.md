# M3c admission under a bandwidth cap — design decision (for review)

Status: PROPOSED v2 (revised after review), not yet implemented. Decides how the
live LRU expert cache admits promotions when promotion demand exceeds the PCIe
bandwidth budget. Self-contained; `PHASE4-M3-PLAN.md` has milestone context.

## The problem

The expert cache keeps H expert slices per MoE layer resident in VRAM and
promotes/evicts dynamically at TG step boundaries. One promotion = copying an
expert's 3 slices (up/gate/down) from host RAM to a VRAM slot = **8.56 MB** of
PCIe HtoD traffic.

Two mechanisms collide:

1. **Admission filter.** An expert becomes a promotion candidate only on its
   **2nd miss within a rolling window** of N=64 TG steps (per-layer windowed
   miss counts, already maintained by the M3a shadow sim).
2. **Rate cap.** `--expert-cache-promote-gbps`, default **8 GB/s**. Plus a
   global in-flight cap and a per-layer pending (mid-overwrite) slot limit.

M3a telemetry on the stories workload (H=32, 8.8 t/s, 42 MoE layers L3–L44):

| policy | promotions/step | bandwidth |
|---|---|---|
| classic LRU (promote every miss) | ~160 | ~12 GB/s |
| 2nd-miss filtered (admission demand) | ~113 | **~8.5 GB/s** |
| cap | — | **8 GB/s** |

## Capacity ceilings: which cap actually binds (review point 1, verified)

v1 of this doc analyzed only the bandwidth numbers and concluded "the cap
binds." Review forced the arithmetic on the structural conjuncts; v1 was wrong.
Ceilings at 8.8 t/s (113.6 ms/step, 0.33 ms per 8.56 MB copy at 26.2 GB/s):

| constraint | max promotions/step | effective GB/s | vs demand 113/step |
|---|---|---|---|
| token bucket @ 8 GB/s | 106 | 8.0 | ~94% of demand |
| 1 pending slot/layer (M3b hard guard) | 42 | 3.2 | 37% — **binds first** |
| in-flight < 4 (plan's starting value) | **4** | 0.30 | **3.5% — binds hard** |

As written in v1, the in-flight cap would have throttled admissions to 4% of
demand and the whole bandwidth discussion would have been moot. Resolutions:

- **In-flight cap re-based on drain time.** Its purpose is bounding the copy
  backlog so publishes land ~1 step after issue, not rate-limiting. Copies are
  FIFO on one stream: 128 outstanding ≈ 42 ms of drain, comfortably inside a
  113.6 ms step, and 128/step ≈ 9.6 GB/s > the 8 GB/s bucket, so the bucket
  stays the binding constraint. **Set in-flight cap = 128.** (4 was a
  needlessly conservative starting guess.)
- **Per-layer pending limit generalized.** The classify free-slot scan needs
  k=8 distinct free non-pending slots out of H+1 ⇒ the true invariant is
  `pending ≤ H+1−k`, not 1. M3c relaxes the M3b guard accordingly: H=8 → 1
  (42/step ceiling — fine, H=8 is a test config), H=32 → 25/layer ⇒ 1050/step,
  non-binding.
- **Telemetry must say which cap binds.** Deferral counters split by reason:
  `deferred_budget / deferred_inflight / deferred_layer_pending`. If
  budget isn't the dominant reason in the stories gate run, the framing is
  wrong again and we revisit.

## How misses are served (review point 4 — answers the cap's true rationale)

A miss is **computed on CPU** from the host-resident full expert tensors (the
cold path; the whole config is `-ot ffn_.*_exps=CPU`). Miss bytes never cross
PCIe — so promotion cannot be "retarget the miss's HtoD stream"; promotion
bytes are always *additional* traffic.

That restates the cap's mechanism: promotion DMA reads host DRAM concurrently
with the CPU cold path reading experts from the same DRAM (8ch DDR4-3200,
~200 GB/s theoretical; the TG cold path pulls ~26 GB/s effective and is
CPU-dequant-bound, not DRAM-bound). M0 measured HtoD-vs-compute contention
≈ 0 at the full 26.2 GB/s. So the 8 GB/s cap is **not protecting compute** —
it is a policy backstop that limits spend on marginal promotions once the
2nd-miss filter has already cut demand from 12 to 8.5 GB/s. Keep the knob for
headroom on weaker systems, but expect hit-rate sensitivity to it to be mild.
(For the future SSD tier this picture changes entirely — SSD reads are the
expensive resource there, and admission strictness matters much more.)

## What "good" looks like

- TG speed is a function of hit rate (model: TG ≈ 9.01/(1−hit)); admission
  policy affects *convergence speed* and *drift tracking*. All sane variants
  share the same steady state when no cap binds — differences are in
  burst/overload behavior.
- Scarce promotion bandwidth should flow to the **hottest** candidates first.
- Never promote an expert that has stopped being routed.
- Never silently lose a valuable admission.
- Minimal new persistent state (F4–F6 history).

## The options

### A. Cap-then-filter (drop) — rejected

2nd miss fires with an empty bucket → admission dropped. Standard for
CPU-cache prefetch hints (free re-presentation), a bad fit here: murky
interaction with the window bookkeeping (reset → re-qualify from scratch;
don't reset → re-fire every step), and after a topic shift the losers of the
budget race are picked essentially at random. Jittery, path-dependent
convergence.

### B. Filter-then-cap (explicit FIFO queue + token bucket) — rejected

The networking/storage-tiering textbook pattern. No lost admissions, bursts
absorbed — but a new persistent data structure with staleness rules (victim
slots move while an admission waits; gone-cold candidates must be re-validated
at dequeue), a depth cap for sustained overload, and — decisive — FIFO grants
*admission order*, not *value order*. B only beats C if you also sort the
queue, at which point it is C with persistence and revalidation debt.

### C. The routing stream is the queue (chosen)

On each staged miss at a TG boundary, the expert is a candidate iff its
windowed miss count ≥ 2. Candidates are promoted subject to budget /
in-flight / per-layer-pending conditions; candidates that fail a condition
simply **stay candidates** (their windowed count is untouched) and re-present
on their next miss. No queue object: the set of experts with count ≥ 2 *is*
the queue, and the routing stream replays it in demand order.

**Review amendment (point 2): demand order must be enforced, not hoped for.**
C-as-written drained the bucket in layer-iteration order within a boundary —
under a binding cap that's a persistent systematic bias toward low layer
indices, and "saturate the count at 2" discarded exactly the signal that
establishes priority. Fix, keeping the no-persistent-state property:

- miss counters count the true windowed total (no saturation at 2);
- at each boundary, collect qualifying candidates into an **ephemeral
  scratch array** (~42 layers × 8 ids, discarded at the end of the boundary),
  **sort by windowed count descending** (ties: most-recent miss first), then
  drain against the bucket in that order.

Now "hottest first" holds both across steps (re-presentation frequency) and
within a step (the sort). If the sort ever showed up in profiles it could be
replaced by layer-rotation, but at ~340 entries it is microseconds.

### D. Adaptive threshold (3rd miss when budget is empty, 1st when full) — rejected

A control loop with more state and tuning surface. The sorted ephemeral
candidate list captures most of the benefit without it.

## Follow-on, named now (review point 3): victim-aware admission

Under a binding cap the true admission test is not "candidate is hot" but
**"candidate is hotter than the victim it displaces"** — the TinyLFU insight
(`admit iff freq(candidate) > freq(victim)`). We already have windowed miss
counts for candidates; a windowed *hit* count per resident slot is one more
counter in the existing per-layer structure. Not in M3c — but it is the first
thing to try if the hit-rate gate lands at the low end of 0.40–0.45, because
it addresses the root cause (bandwidth spent on zero-gain swaps) rather than
reshuffling who waits.

## M3c algorithm spec (intended diff shape)

Per TG step boundary, in order:

1. **Publish** completed promotions (M3b `llama_expert_cache_publish`).
2. Bucket refill by **wall time**:
   `budget = min(budget + cap_gbps × step_walltime, burst_cap)`,
   burst_cap ≈ 2 steps' accrual (~1.8 GB at defaults); first boundary seeds
   one burst cap so session-start convergence isn't serialized.
   *Hidden feature, do not "fix" to per-step accrual:* low hit rate → slower
   steps → more budget per step → faster convergence exactly when the cache is
   performing worst.
3. Per staged layer, per staged routed id (dedup within the step): hit → LRU
   touch + windowed hit count (needed by the follow-on; cheap now); miss →
   bump windowed miss count (true count, window expiry via the M3a deque);
   collect candidates (count ≥ 2) into the scratch array.
4. Sort candidates (count desc, recency tiebreak); drain in order while
   bucket ≥ 8.56 MB ∧ in-flight < 128 ∧ layer pending < H+1−k: evict LRU
   victim among non-pending slots, `queue_promotion` (tearing-safe, M3b),
   bucket -= 8.56 MB, candidate's miss count ← 0 (a promotee later evicted
   must re-qualify — cheap thrash damping).
   Each skipped candidate records its deferral reason
   (`deferred_budget / deferred_inflight / deferred_layer_pending`).
5. PP (n_tokens > 8): fully read-only — no counts, no promotions (plan
   decision 1; the M3a staging guard already enforces the shape).

Also in the diff: relax M3b's `pending_mask != 0` guard in
`llama_expert_cache_queue_promotion` to the true invariant
(pending popcount < H+1−k). Env/flags: `--expert-cache-promote-gbps` (exists,
default 8), `IK_EXP_CACHE_WINDOW` (exists, default 64),
`IK_EXP_CACHE_ADMISSION=0` (added 2026-09-06 post-review: explicit admission
off-switch for clean on/off A/Bs — accounting/telemetry still run, candidates
never generated);
`IK_EXP_CACHE_STAGING_MB` / `IK_EXP_CACHE_STAGING_THREADS` (added 2026-09-06:
pinned staging ring for promotion copies, default 256 MB / 2 memcpy threads;
see PHASE4-M3C-FINDINGS.md item 5); new debug counters:
promoted/step, deferrals by reason, in-flight high-water.

## Validation gates (unchanged)

- Hit rate 0.40–0.45 @H=32 stories — how far below the M3a sim's 0.5375
  (classic, unfiltered, uncapped) the live filtered+capped policy lands, and
  whether deferral telemetry says budget is the dominant reason.
- TG ≥ 12 t/s @19.7k A/B vs m2-base-clean, greedy text budget per M0.1.
- If hit rate lands low: victim-aware admission (above) is the first follow-on.
- If `deferred_budget` dominates: the cap itself is the constraint — revisit
  the default or the pinned-source question, not the design.

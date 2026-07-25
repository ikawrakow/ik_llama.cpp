# Context pack — Laguna SWA ring KV (ik_llama.cpp fork)

## Mission
Fork of ikawrakow/ik_llama.cpp to fix **Blocker A**: main-model KV must honor
`sliding_window=512` so Poolside Laguna's SWA layers stop allocating dense
`n_ctx` KV. Target: **256k context fits under `-sm graph`** (which forces
bf16 reduce_type for Laguna, src/llama.cpp ~7884). KV shrink ≈ 256× per SWA
layer at 256k (window 1024 cells vs 262144).

- Local: `/home/jgonc/Personal/projects/ik_lcpp_laguna`, branch `laguna-swa-kv`,
  base upstream `31018dc5` (2026-07-23), SQUASHED to single commit `acf9a11b`
  (ring + seq_rm rewind refusal + prompt-cache gate + llama_kv_self_is_swa_ring
  + --dry-* help entries + split-path transposed-V wrap hardening). Remote
  git-am twin on r02-rds01: `d5ef9fc6`. PR text: `PR.md` (untracked).
  Adversarial review findings: split transposed-V wrap was missing (fixed;
  note `-sm graph -fa off` crashes upstream even dense, so path is hardening);
  --dry-* parsers already existed upstream (duplicate removed, help kept);
  ub=512 never wraps R=1024 — wrap coverage on GPU needs -ub 384 (KL-verified
  at noise floor: 0.046 vs 0.054 control).
- Laguna arch was already in upstream (`LLM_ARCH_LAGUNA`, XS mixes SWA+global
  via `hparams.swa_layers[il]`, window `hparams.n_swa`); SWA was mask-only.

## Design (ruled): modulo ring in the single shared cache
NOT mainline's dual-cache iswa (too invasive here; and this fork's `-sm graph`
split-KV tensors need the same treatment, which mainline never had).
- Ring size `R = GGML_PAD(n_swa + n_ubatch, max(kv_pad,256))`; exact because
  R ≥ n_swa + n_ubatch ⇒ every overwritten slot is out-of-window for all
  in-flight queries (append-only, single-seq).
- Cell bookkeeping (cells/head/n/find_slot) unchanged; only tensor storage,
  views, and the SWA mask are ring-aware. `ring_occ[slot] = cell` maintained in
  `llama_kv_cache_find_slot`, reset in `..._clear`.
- Writes at `cell % R`; a ubatch straddling the ring end emits TWO copies and
  nulls its `cache_copies` entry (graph-reuse then refuses & rebuilds).
- Mask `[R, n_tokens_pad]` filled from ring_occ + cells (seq/causal/window).
- Occupancy guard in mask fill: any in-window cell that is not its slot's
  occupant → `GGML_ABORT` (non-append-only use never corrupts silently).

## Change map (all in commit 4f3398bf)
- `include/llama.h` + `common/*`: `swa_full` flag (`--swa-full` = dense fallback).
  **Since superseded**: flag flipped to opt-in `swa_compress`/`--swa-compress`
  (default off = dense, matching pre-existing behavior); the ring gate also
  generalized from Laguna-only to `llama_model::supports_swa_ring()` (any
  `n_swa > 0` arch, with a small excluded list). PR.md tracks current state.
- `src/llama-context.h`: cache fields `swa_ring/size_swa/ring_occ`.
- `src/llama.cpp`: ring decision (after Mamba kv_size case; guards at the time:
  Laguna-only, `n_seq_max==1`, defrag off -- **since generalized**, see line 43
  above: any `supports_swa_ring()` arch, not just Laguna); per-layer `kv_size_l`
  in `llama_kv_cache_init`
  (dense k_l/v_l AND split_k_l/split_v_l); find_slot occ; mask fill ring branch
  + guard; non-causal swa-mask path aborts; `update_cache_copies` patches
  `head % R`, refuses wrap; K-shift returns error; defrag/state-save abort,
  state-load errors (all messages point at `--swa-compress` now, post-flip).
  **Since superseded**: rows are (seq, pos)-derived and striped per sequence, the
  write follows a per-ubatch run plan, state IO round-trips, and defrag is
  disabled-with-a-warning rather than aborting.
- `src/llama-build-context.cpp`: `llm_build_kv_store` ring writes (wrap-split);
  `llm_build_kqv` ring reads (`n_kv_l=R`, v_trans stride R); same in
  `build_std_attention` split path (`-sm graph`); `build_inp_KQ_mask_swa`
  width R; fattn `op_params[4]` NOT set for ring layers.
- `src/llama-model.cpp`: `cache_size()` window-aware (bound n_swa+4096) so
  `-sm graph` auto-fit accepts long contexts.
- `gguf-py/gguf/constants.py`: upstream dup INDEXER_* enum members removed
  (import TypeError on py3.14).
- `tests/`: `test-laguna-swa-ring.sh` + `make-tiny-laguna-gguf.py` (+ ctest reg).

## Hard-won gotchas
1. **Transposed-V wrap writes**: slice the CONTIGUOUS source per token BEFORE
   `ggml_transpose`. `ggml_view_2d` of an already-transposed tensor assumes
   contiguous rows → silent wrong data (was a real 0.9% PPL bug).
2. **fattn op_params[4]**: CUDA wrapper (ggml-cuda/fattn.cu:44-71) slices the
   K tail by INDEX assuming newest-at-end — wrong for ring; leave 0.
3. **Greedy-text oracle is invalid on tiny random models**: flat logits ⇒
   argmax flips under any reduction-order change (dense-vs-dense diverged from
   token 2 with just `-t 4` vs `-t 8`). Scaling output weights doesn't help
   (argmax-invariant). Use scoring parity; decode-shaped coverage via
   `llama-perplexity -ub 1 -gr`.
4. ik quirks: `-fa` consumes a value (`-fa on`); ppl prints
   "Final estimate: PPL over N chunks for n_ctx=... = X +/- Y";
   graph_reuse defaults ON; defrag_thold defaults -1 (off).
5. Test ubatch = 48 ON PURPOSE (doesn't divide 256 ⇒ wrap-split exercised).

## Verification (clean-build close-out, CPU)
- ppl ring vs full: ub48 no-fa **0.00e+00**, fa on 1.56e-06, ub1+reuse
  (decode-shaped) **0.00e+00**; KV shrink 2.0× on tiny model (3/4 layers SWA);
  state-save + context-shift guards fail loudly. Code-level hunt: 8/9 HOLD,
  1 AMBIGUOUS (max_nodes headroom for +2 wrap copies — loud failure if ever hit).

## Remote test-drive (r02-rds01, DONE 2026-07-23 — CUDA/-sm graph VERIFIED)
- Host: 4× NVIDIA L40 (46GB each), 56 cores, CUDA 13.1 (`/usr/local/cuda/bin`,
  NOT in default PATH), no numpy. Ansible: `ansible r02-rds01 -b ...` works.
- Models: `/data/hf_models/Laguna-XS-2.1-Q4_K_M.gguf` (19G, vocab 100352),
  `/data/hf_models/Laguna-S-2.1-UD-Q5_K_S/` (3-part).
- Their checkout `/root/ik_llama.cpp` @ 9d07d868 (untouched); ours:
  `/root/ik_lcpp_laguna-swa` @ `abd0f54a` (= 4f3398bf via git am), built OK.
- GPU 0 runs someone's llama-server (untouched). The server that held GPUs
  2+3 (mainline llama.cpp serving Laguna-S @ 256k, port 8080, nohup) was
  STOPPED with user approval; restart: `/root/restart-laguna-server.sh`
  (cmdline also in `/root/laguna-server-cmdline.txt`). All tests ran with
  `CUDA_VISIBLE_DEVICES=1,2,3` (-sm graph needs ≥2 devices).
- Results (XS-2.1 Q4_K_M, `-sm graph -ngl 999 -fa on`):
  1. Smoke -c 16384: "SWA ring KV: n_swa = 512 -> ring size 1024 cells",
     reduce_type bf16 confirmed, KV 760 MiB, coherent text, 151 tok/s.
  2. PPL parity -c 4096 ×8 chunks (corpus = head of src/llama.cpp): ring
     2.5141 ±0.046 (repeat bit-identical) vs full 2.5167 ±0.046 → Δ 0.10%.
     KL check: ring-vs-full mean KLD 0.0501, same-top-p 96.18% vs NUMERICS
     CONTROL (full-ub256 vs full-ub512, both provably correct) 0.0548 /
     96.25% → ring divergence AT/BELOW the bf16 noise floor. Not a defect.
  3. Blocker A -c 262144: ring KV 10360 MiB vs dense 40960 MiB (SWA layers
     20G→0.12G ≈170×; residual = global layers, 1 in 4). Ring gen coherent.
     Layer mix implied 3:1 SWA:global (derived from KV scaling).
- Logs on host: smoke1.log, ppl-{ring,full,ring2}.log, big-{ring,full}.log,
  kld-{save,ring,ctrl}.log, kld-base.dat (3.3G, deletable).
- SERVING (production, since 2026-07-23): deployed via the Aircentre Ansible
  repo (`datacenter-ansible-playbooks/plays/services/lcpp_server/
  deploy_laguna.yml`, updated for the fork): Laguna-S-2.1-UD-Q5_K_S at ctx
  184320 (180k) on GPUs 2,3, port 8080, `-sm graph`, f16 ring KV 8.8G,
  margin 1.2/0.9G per GPU, --no-context-shift, alias + DRY tune (0.8/1.75/2/
  -1) via the fork's new --dry-* flags. Log /var/log/lcpp_server_laguna.log,
  pid /var/run/lcpp_server_laguna.pid. GPU 1 free; GPU 0 has an unrelated
  server. Measured: ~25 MiB per 1k ctx per GPU; ~215k absolute 2-GPU max.
  Earlier proofs: 1M ctx on 3 GPUs (KV 48.1G; dense f16 ~384G), 256k on 3
  GPUs (KV 12.4G). Rollback to mainline: `/root/restart-laguna-server.sh`.
- STRESS (fixed binary, 180k): 175k-token pp at 1800 t/s (97s), 2048 tg at
  55.7 t/s at full depth; cache-reuse: unsafe rewind (2048) → refused →
  full reprocess 108s; safe rewind (=512 slack) → partial reuse ~10s.
  VRAM flat across the whole run. Pre-fix, the unsafe rewind ABORTED the
  server via the occupancy guard (this motivated 265486db).
- SECOND production crash (motivated 0b5e5f07): ik server's RAM prompt
  cache (--cache-ram, default 8192 MiB) snapshots slot KV via
  llama_state_seq_get_size/get_data on slot swap → hit the state-save
  GGML_ABORT on ordinary small-prompt traffic. The 175k stress dodged it
  ONLY because that slot state exceeded the cache size limit. Now: server
  auto-disables the cache under ring (log line asserted in the test), and
  seq-state serialization warns + returns 0. Verified live: repeat/switch/
  switch-back chat traffic, all finish:stop, health ok.

- THIRD production incident (2026-07-24, NOT a fork bug): deep opencode
  sessions produced 8-16k-token "Wait/Actually" reasoning spirals. Root
  cause: the GGUF-embedded Laguna chat template renders client-echoed
  reasoning_content into <think> blocks for ALL historical assistant turns
  whenever enable_thinking is on; opencode re-sends reasoning each turn, and
  the model (trained with prior thinking stripped) degenerates when
  conditioned on its own past CoT. Confirmed by replaying the real failing
  session (opencode sqlite → exact messages): echoed-reasoning render →
  spiral (2/2 seeds, marker density matches real sample); stripped → clean
  tool calls at every depth/sampling tested. Fix: patched template
  (history gated on preserve_thinking only; enable_thinking still opens
  fresh <think> at generation) deployed via --chat-template-file, file
  lives at plays/services/lcpp_server/files/laguna-chat-template.jinja in
  the Aircentre ansible repo. Post-fix reproducer: prompt 23498→11458,
  clean tool calls. FOLLOW-UP 1: the gate must keep `loop.last and
  reasoning_content` rendering — common/chat-diff-analyzer.cpp
  compare_reasoning_presence() probes [user, assistant+reasoning] with
  assistant LAST to auto-detect <think> parsing; hiding reasoning there
  set reasoning.mode=NONE and leaked raw </think> into content (v1 of the
  patch did this; v2 fixed it, both behaviors verified live).
  FOLLOW-UP 2 (thinking-vs-history tension, measured 2026-07-24): ANY
  reasoning-free assistant turn in history suppresses generation thinking
  entirely (1868 rsn chars -> 0 from one such turn; both '<think></think>'
  and bare '</think>' forms). Explored: newest-turn full primer -> spiral
  returns when that chain is a runaway (29KB rsn, 2/2 seeds); truncated/
  placeholder primers -> tool calls emitted INSIDE the think block,
  unparseable (also malformed args); natural small primer -> 50% in-think
  tool calls. FINAL deployed form: strip all history reasoning (loop.last
  kept for the detector) — thinking on fresh/history-free turns only,
  agent tool calls reliable, no spirals. Path to thinking+agents: fork
  parser change making a tool-call opener an alternative reasoning
  terminator (vLLM parsers do this), then re-test primed variants. Also learned: deployed DRY (breakers \n : " *) is
  near-inert on agent text (matches reset at every colon/quote); with DRY
  fully off the model loops verbatim even at 4k ctx, so keep DRY on.

## Open risks
- bf16 reduce under -sm graph is inherently noisy on this model+quant
  (control max KLD ~14 between two CORRECT configs) — expected, not ring's.
- **Resolved (W4)**: estimator now takes the real `n_ubatch` instead of a
  hardcoded `n_swa+4096` bound; scales by `n_seq_max` (one striped window per
  sequence) in lockstep with the runtime, and clamps to dense where the striped
  ring would reach the full context. The old `--defrag-thold` under-budget gap is
  closed from the other side: defrag no longer changes the layout (it is disabled
  when the ring engages), so the estimator never needs to see it.
- Multi-seq IS supported: rows are striped per sequence
  (`row = seq*ring_w + pos % ring_w`), so `-np > 1` engages the ring. Context
  shift / defrag / `seq_cp` / `seq_keep` (and hence server system prompts, which
  fan out via `seq_cp`) remain deliberately unsupported with ring.
- **Resolved (W5)**: state IO is ring-aware. A ring layer serializes the last
  `min(size_swa, cell_count)` cells oldest-first; the blob carries a `RING`
  descriptor (`size_swa`, `n_swa`) so mismatched-geometry and cross-mode
  (ring<->dense) restores are refused. Written only when the ring is engaged, so
  dense blobs stay byte-compatible with pre-W5 builds. `--prompt-cache`, the
  server RAM prompt cache and `/slots/{id}?action=save|restore` all work now;
  pinned by `tests/test-swa-ring-state.cpp` plus new legs in
  `tests/test-laguna-swa-ring.sh`.
- 256k KV is now global-layer-bound (~10G). Further shrink needs KV quant
  (-ctk/-ctv) or global-layer work — out of scope.

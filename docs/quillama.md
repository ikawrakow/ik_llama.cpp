# quillama.cpp: technical notes

quillama.cpp is [ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp) (full history kept, so
upstream ik changes can be merged) tuned for **CPU-only servers**, such as a VPS with lots of RAM, a
fast SSD and no GPU. This page lists every change, what it does, how it was measured, and its limits.

- [Build and run](#build-and-run)
- [Docker](#docker)
- [What changed](#what-changed)
  - [1. JSON schema / structured output now always constrains the output](#1-json-schema--structured-output-now-always-constrains-the-output)
  - [2. Faster grammar-constrained sampling](#2-faster-grammar-constrained-sampling)
  - [3. Intel AMX for prompt processing](#3-intel-amx-for-prompt-processing)
  - [4. AMX flash attention](#4-amx-flash-attention)
  - [5. Run-time repack on by default when the model fits in RAM](#5-run-time-repack-on-by-default-when-the-model-fits-in-ram)
  - [6. Models larger than RAM: expert streaming from SSD](#6-models-larger-than-ram-expert-streaming-from-ssd)
  - [7. Grammar jump-forward drafting](#7-grammar-jump-forward-drafting)
  - [8. Transparent huge pages that actually work on a VPS](#8-transparent-huge-pages-that-actually-work-on-a-vps)
- [Benchmarks](#benchmarks)
- [Ideas taken from Strata and the MoE-cache fork](#ideas-taken-from-strata-and-the-moe-cache-fork)
- [Environment variables](#environment-variables)
- [Tests](#tests)

## Build and run

Build on the machine that will run it: `GGML_NATIVE` targets the local CPU. The AMX code is compiled
in any x86-64 Linux build and enabled at run time only when the CPU and kernel support it.

```bash
./scripts/quill-build-cpu.sh            # cmake -DGGML_NATIVE=ON -DGGML_CUDA=OFF ..., Release
./build/bin/llama-server -m model.gguf -t $(nproc) -c 16384 --jinja
```

Nothing else needs to be configured. On start-up the server prints which mode it picked:

```
quillama: model (17.3 GiB) fits in RAM (58.1 GiB available): run-time repack enabled (--no-repack to disable)
```

or, for a model larger than RAM:

```
quillama: model (380.0 GiB) does not fit comfortably in RAM (120.4 GiB available): keeping mmap, no run-time repack, MoE expert prefetch from storage enabled
```

Use a physical-core thread count (`-t`). On a CPU with AMX, prompts are processed in micro-batches
of 2048 tokens (`-ub 2048`, instead of ik's 512) unless `-ub` is given. Each MoE expert then sees
more tokens per batch, which keeps the tile units busy: MoE prompt speed 648 → 865 t/s at 2048
tokens. The compute buffer grows accordingly (a few hundred MB to ~2 GB, depending on the model).

## Docker

The CPU image is built from `docker/ik_llama-cpu.Containerfile`. It is compiled for the CPU of the
machine that builds it (`GGML_NATIVE=ON`), so build it on the server that will run it:

```bash
docker build -f docker/ik_llama-cpu.Containerfile --target server -t quillama:cpu-server .
docker run -d --name quillama -p 8080:8080 -v /path/to/models:/models:ro quillama:cpu-server \
    --model /models/model.gguf --ctx-size 32768 --threads 8 --jinja
```

or with Compose (edit the model file name and thread count first):

```bash
MODELS_DIR=/path/to/models docker compose -f docker/quillama-cpu.compose.yaml up -d --build
docker logs -f quillama          # shows whether the model was repacked into RAM
```

- Inside a container, quillama reads the container's memory limit (cgroup v1 or v2), not the
  host's RAM, when it decides between repacking into RAM and streaming from disk. If you set
  `mem_limit`, leave room for the KV cache.
- AMX works in containers with Docker's default seccomp profile. The log line
  `quillama: ...` and the speed of a long prompt tell whether it is in use.
- The healthcheck allows 15 minutes for loading, because repacking a large model takes a while.


### 1. JSON schema / structured output now always constrains the output

Three bugs made ik_llama.cpp emit output that ignores the requested JSON format, while upstream
llama.cpp, with the same model and prompt, follows it:

| Request | ik_llama.cpp | quillama.cpp |
| --- | --- | --- |
| `response_format: {"type": "json_object"}` (no schema): what the OpenAI SDKs send for "JSON mode" | **no constraint at all**: the empty schema was treated as "no response format" | constrained to a JSON object (as in upstream) |
| `json_schema` with a template that uses `force_pure_content` | schema silently dropped | schema grammar applied |
| reasoning models with a lazy grammar (tool calls after `</think>`) | the end-of-reasoning tag was never fed to the grammar, so a trigger inside it was lost | replayed into the grammar (ported from upstream) |

Measured end-to-end with `scripts/quill-json-schema-test.py` against a **random-weight** model:
such a model emits noise, so every valid output is the grammar's doing. ik_llama.cpp produced
unconstrained garbage for every `json_object` request; quillama.cpp produced valid JSON for all of
them.

The grammar converter itself is now checked by upstream llama.cpp's behavioural suite
(`tests/test-grammar-integration-upstream.cpp`): around 100 grammars and JSON schemas, each with
strings that must or must not match. ik's own grammar tests were disabled because they no longer
compiled. ik's converter passes all of it. The one intended difference is that the empty schema
`{}` means "any object" rather than "any value", which keeps parameter-less tool calls valid.

### 2. Faster grammar-constrained sampling

Whenever a sampled token violates the grammar, the sampler masks the whole vocabulary (150k+ tokens
for Qwen-style models) with the grammar. ik decoded every token's text into a fresh vector on every
call. Now:

- the decoded code points of every token are cached once per vocabulary;
- a token whose first character no parse stack accepts is rejected without walking the grammar.
  This is exact: such a token can never match. At structural JSON positions (`{`, `"key"`, `:`,
  `,`) only a handful of tokens remain for the full check.

`tests/test-quill-grammar.cpp` compares the new mask with the original algorithm over 985 grammar
states × 151,936 tokens: identical. JSON-schema generation (synthetic 1.6B Q4_K_M, 4 threads):
12.8 t/s with the original masking → **21.9 t/s**, token-identical (ik_llama.cpp: 10.3 t/s; the
rest of the difference is the run-time repack default, section 5).

### 3. Intel AMX for prompt processing

Sapphire Rapids and newer Xeons (common on cloud VPS: AWS m7i/c7i/r7i, GCP c3, Azure Dv6/Ev6, ...)
have AMX tile units that do 16×16×32 BF16 matrix products per instruction. ik_llama.cpp did not use
them. `ggml/src/quill/quill-amx.cpp` adds an AMX BF16 GEMM, used for batches of at least 32 tokens:

- weights are dequantized panel by panel to BF16: fast AVX-512 paths for Q4_0, Q8_0, IQ4_NL,
  IQ4_XS, Q4_K, Q5_K, Q6_K, Q4_K_R4 and Q6_K_R4, and `to_float()` for every other type, including
  ik's row-interleaved `_R4/_R8/_R16` repacks;
- activations are packed once per matmul (shared by all threads) into the AMX tile layout;
- a 2×2 tile micro-kernel accumulates over the full K dimension in the tile registers;
- it covers plain matmuls, ik's fused FFN up/gate (SiLU/GELU/ReLU), MoE expert matmuls
  (`MUL_MAT_ID`; tokens are gathered per expert and outputs scattered back) and the fused MoE
  up/gate.

Token generation (batch 1) is memory-bound and keeps ik's kernels. On CPUs without AMX (AMD EPYC,
older Xeons, ARM) nothing changes. `QUILL_AMX=0` disables it.

Accuracy: activations and weights are rounded to BF16, while ik rounds activations to 8-bit. The
KL divergence between the AMX and iqk paths over whole models is 0.0003 (dense) and 0.0009 (MoE).

### 4. AMX flash attention

ik's CPU flash attention took 21-25% of a 512-token prompt and grows with the context.
`quill_amx_flash_attn`:

- computes `S = Q·Kᵀ` and `O += P·V` on AMX tiles, with an online softmax in AVX-512;
- uses a work unit of (sequence, KV head, query range), so all query heads that share a KV head
  (GQA) reuse each packed 256-key K/V chunk;
- skips fully masked 32×32 blocks of the causal mask;
- supports F16, BF16, Q8_0 and F32 K/V caches and softcap.

ALiBi, attention sinks and other special cases keep the iqk path. Unit-tested against a
double-precision reference.

### 5. Run-time repack on by default when the model fits in RAM

ik's `-rtr` repacks weights at load into row-interleaved layouts that its CPU kernels process
faster: **+25-40% token generation** in our measurements. It was opt-in, and turning it on used to
cost prompt speed. Now that the AMX path handles the repacked types, quillama turns it on
automatically, on a CPU-only run, when the model file is below 70% of `MemAvailable`. The weights
are then held in RAM instead of the page cache.

- `--no-repack` (`-nrtr`) disables it.
- `-rtr` and `--no-mmap` keep their usual meaning.
- Models that do not fit keep `mmap` (see next section).

### 6. Models larger than RAM: expert streaming from SSD

When the model does not fit, quillama keeps `mmap` and enables ik's MoE expert read-ahead
(`--prefetch-experts`). A worker pool faults in the selected experts' pages with
`MADV_POPULATE_READ` while the CPU works, instead of taking page faults one by one. Under a
1.6 GB memory cap with a 2.6 GB MoE model, this measured +3% over plain demand paging.

`--moe-lookahead` (experimental, off by default) adds the "early router" of Strata and of the
MoE-cache llama.cpp fork, moved to the SSD → RAM tier:

1. While layer *L* computes, layer *L+1*'s router runs on layer *L*'s FFN input.
2. The predicted experts are queued for read-ahead, under a separate ticket so the real selection
   is still fetched on demand.

On synthetic models it lost to plain read-ahead (2.6 vs 2.9 t/s, even with correlated routing), so
it stays opt-in until it can be measured on real MoE models.

### 7. Grammar jump-forward drafting

When every parse stack of the grammar agrees on the next character, that text is forced: in a
strict schema, the keys, quotes, colons and braces. `--grammar-draft N` (default 16) drafts the
tokenization of the forced text and lets the normal grammar-constrained sampling verify it in one
batch.

- The output is identical to normal decoding.
- On CPU, a batch of a few tokens costs about as much as one token, since decode is memory-bound.
- Used only when no other speculative method is configured and the KV cache can drop a rejected
  tail (not with recurrent/hybrid models).

The acceptance rate depends on how strongly the model prefers the canonical tokenization of the
forced text. A random-weight model accepts only ~28% (it picks among all valid splits at random),
so the speed-up could not be measured meaningfully here.

### 8. Transparent huge pages that actually work on a VPS

ik's `-thp` maps the model with `MAP_HUGETLB`, which needs pages reserved in `vm.nr_hugepages`. A
typical VPS has none, so the flag silently fell back to 4 KiB pages. quillama falls back to an
anonymous 2 MiB-aligned mapping with `MADV_HUGEPAGE`, which works with the default THP setting
(`madvise`). It measured +2% token generation on a 1 GB model; larger models gain more from fewer
TLB misses.

### 9. Huge pages for repacked weights and other large CPU buffers

With `-rtr` (on by default when the model fits in RAM) the weights are not served from the mmap.
They live in an anonymous CPU backend buffer, and `-thp` never applied to that buffer. quillama
allocates every CPU buffer of 32 MiB or more as a 2 MiB-aligned mapping with `MADV_HUGEPAGE`. This
needs no flag and no reserved huge pages. On an AVX2-only build (`-march=znver3`-class), a
Qwen3-30B-A3B-shaped Q4_K_M MoE generated tokens **~20% faster**, because decode streams
gigabytes of expert weights and 4 KiB pages make it TLB-bound. `QUILL_THP=0` turns it off.

### 10. MoE work split for decode

ik splits each expert's matmul into chunks of at least 32 rows. When decoding a few tokens, a
Qwen3-30B-A3B expert (768 rows for up/gate) became up to 24 tiny streams per expert, and every
stream paid for its own prefetcher ramp-up. For batches of 4 tokens or fewer, quillama uses a
128-row minimum (+2% TG on AVX2). Prompt processing is unchanged. To tune it, set
`QUILL_MOE_MIN_CHUNK=N` (minimum rows per chunk), or `QUILL_MOE_CHUNKS=N` to force N chunks per
expert.

### 11. Sparse output layer under a grammar

When a grammar is active and few tokens can come next (inside a JSON key, after `"`, a number,
`true`/`false`, ...), the server computes logits only for the tokens the grammar allows.
Everything else is masked to -inf anyway, so the result is exact. Greedy outputs are
byte-identical with and without it. The 151936 x 2048 output matrix is ~10% of the per-token
weight traffic of Qwen3-30B-A3B, and schema-constrained generation on the AVX2 build went from
**28.1 to 33.4 tokens/s**. The limit `--grammar-sparse-logits N` defaults to 4096 allowed tokens;
0 disables it. It is skipped when `n_probs` is requested, during speculative drafts, and while a
lazy grammar waits for its trigger.

### 12. JSON schema: non-negative numbers

`{"type": "number", "minimum": 0}` (or `exclusiveMinimum >= 0`) now forbids a leading `-`. Upstream
llama.cpp ignores bounds on `number`, so models could emit negative values for such fields.

### 13. Faster attention during generation at long context (AVX2)

On an AVX2 CPU at 8k context, flash attention takes about half of each generated token's time
for a Qwen3-30B-A3B-shaped model (1.7 ms per layer, against 1.2 ms for the MoE FFN). It is limited
by compute, not memory: f16, q8_0 and q6_0 KV caches all decode at the same speed. Of that
attention time, 53% goes to accumulating V. quillama converts each 32-row V tile to f32 once,
keeps 64 output columns per query head in registers across the tile, and feeds V to the FMAs
straight from L1. This takes 1 broadcast + 8 FMAs per V row, where ik reloaded and stored the
accumulators every 4 rows. The products and their summation order are unchanged, and outputs are
bit-identical, tested across KV types, context lengths, batch sizes and thread counts.

Results: the attention kernel is 18% faster on one thread at 8k context, and generation at 8k
depth is about 5% faster end to end. Prompt processing is unchanged within noise.
`QUILL_FA_VBLOCK=0` restores the original loop. AVX-512 builds keep ik's AVX-512 path.

TurboQuant (KV-cache compression to 2-4 bits) was evaluated for CPU and not adopted. Because
CPU attention is compute-bound, a smaller KV cache does not decode faster (measured: q4_0 KV was
4% faster than q8_0 at 8k, f16 the same as q8_0). Its CPU grouped-query decode change is
already in ik's kernel.

### 14. Speculative decoding on CPU, and prompt-prefix reuse

Verifying draft tokens in one batch is cheap on a CPU MoE: on the Qwen3-30B-A3B-shaped model,
4 tokens cost 1.67x one token and 8 tokens 2.4x. Drafting therefore pays off as soon as more than
~0.7 tokens per step are accepted. The server has n-gram and suffix ("prompt lookup") drafting,
which suits extraction workloads that copy names from the prompt: `--spec-type suffix` (or
`ngram-simple`). Speculative decoding currently needs `-np 1`. With grammar drafting off, outputs
are byte-identical to normal decoding. Check `timings.draft_n_accepted` on your own model to see
whether it helps.

`scripts/quill-prefix-reuse-test.py` checks that a prompt sharing 98% of its prefix with an
earlier one processes only the new tokens, also after an unrelated request moved it out of the
slot into `--cache-ram`. It passes with 1 and 2 slots and with speculative decoding on.

### Recommended settings for an AVX2 EPYC VPS (Zen 2/3, no AVX-512)

* `-ctk q8_0 -ctv q8_0`: at long context, decode is bandwidth-bound on the KV cache as well as on
  the weights. q8_0 halves that traffic, and its quality loss is negligible.
* `--threads` close to the number of physical cores that actually deliver memory bandwidth. On a
  12-vCPU VPS, try 6, 8 and 12 with `llama-bench -t 6,8,12`; decode often peaks below the core
  count. Prompt processing is compute-bound, so use all cores with `--threads-batch 12`.
* Keep `-fa on` and `-rtr` (default). Leave THP at the kernel default (`madvise`) or `always`.

## Benchmarks

Machine: cloud VM with 4 vCPUs of an Intel Xeon (Sapphire Rapids class, AMX-BF16/INT8), 15 GB RAM.
Models: random-weight GGUFs with a real Qwen2 tokenizer (`scripts/quill-synthetic-model.py`;
Hugging Face was not reachable from the build environment). Speed does not depend on the weight
values. Both builds use `GGML_NATIVE=ON`, 4 threads, ik_llama.cpp at commit `3d27f5bb` (main, Oct
2026). The VM is shared, so runs vary by about ±5-10%.

Three configurations: ik_llama.cpp with its defaults; ik_llama.cpp hand-tuned with `-rtr 1 -ub 2048`;
and quillama.cpp with the settings it picks by itself on this machine (repack on because the model
fits in RAM; `-ub 2048` because the CPU has AMX). llama-bench does not apply the automatic choices,
so they are passed explicitly there.

**Dense, 1.6B parameters, Q4_K_M** (llama architecture, 16 layers, 2048 hidden, 8192 FFN, 152k vocab)

| | ik defaults | ik tuned | quillama.cpp | vs ik tuned |
| --- | ---: | ---: | ---: | ---: |
| prompt, 512 tokens | 383 t/s | 468 t/s | **974 t/s** | 2.1× |
| prompt, 4096 tokens | 287 t/s | 313 t/s | **888 t/s** | 2.8× |
| generation | 17.2 t/s | 27.2 t/s | **27.9 t/s** | same kernels |

**MoE, 4.5B total parameters, Q4_K_M** (qwen3moe architecture, 12 layers, 64 experts, top-8)

| | ik defaults | ik tuned | quillama.cpp | vs ik tuned |
| --- | ---: | ---: | ---: | ---: |
| prompt, 512 tokens | 419 t/s | 418 t/s | **700 t/s** | 1.7× |
| prompt, 4096 tokens | 257 t/s | 267 t/s | **727 t/s** | 2.7× |
| generation | 25.7 t/s | 34.8 t/s | **32.0 t/s** | same kernels |

Token generation uses ik's own kernels in both cases, so a hand-tuned ik matches quillama there (the
differences are within this shared VM's run-to-run noise). The generation gain over ik's defaults
comes from enabling ik's run-time repack automatically (section 5).

**JSON-schema generation** (dense model, server `/completion` with a nested schema): 10.3 t/s →
21.9 t/s.

For comparison, upstream llama.cpp (Oct 2026) on the same machine: dense prompt 512 tokens 330 t/s,
generation 20.4 t/s; MoE prompt 512 tokens 185 t/s, generation 25.1 t/s.

How the gains split (prompt numbers at 512 tokens unless noted):

| Change | Prompt (dense / MoE) | Generation |
| --- | --- | --- |
| AMX GEMM + fast dequantization | 2.2× / 1.3-1.5× | n/a |
| AMX flash attention | grows with context: 2.8× / 2.7× vs tuned ik at 4096 tokens | n/a |
| `-ub 2048` by default on AMX (no gain for ik's kernels) | +10% / +35-50% | n/a |
| run-time repack by default (ik's `-rtr`) | unchanged | +25-60% vs ik defaults |
| THP fallback (`-thp`) | n/a | +2% |
| grammar mask | n/a | 1.7× for JSON-schema output (2.1× vs ik together with repack) |

On an AMD EPYC or another CPU without AMX, the prompt-speed gains do not apply. The repack default,
grammar speed, JSON fixes and THP do.

Reproduce:

```bash
python3 scripts/quill-synthetic-model.py --arch llama    --out dense-f16.gguf
python3 scripts/quill-synthetic-model.py --arch qwen3moe --out moe-f16.gguf
./build/bin/llama-quantize dense-f16.gguf dense-q4km.gguf Q4_K_M
./build/bin/llama-bench -m dense-q4km.gguf -t 4 -p 512,2048,4096 -n 64 -rtr 1
```

## Ideas taken from Strata and the MoE-cache fork

[Strata](https://github.com/Niko1221/Strata) runs a 125B MoE on a gaming PC by tiering the model
over VRAM, RAM and SSD. The [MoE-cache llama.cpp fork](https://github.com/GenerelSchwerz/llama.cpp)
keeps hot experts in VRAM. Both are GPU-centric. On a CPU-only server their ideas map as follows:

| GPU idea | CPU-only counterpart in quillama |
| --- | --- |
| hot experts in VRAM, cold in RAM | weights repacked into the fast CPU layout in RAM when they fit (5); `mmap` + expert read-ahead from SSD when they do not (6) |
| early router: predict and pre-stage the next layer's experts | `--moe-lookahead` (6), experimental |
| large prefill chunks so streamed experts are reused by many tokens | AMX tiles (3, 4) make large micro-batches compute-efficient; raise `-ub` for MoE |
| MTP / prompt-lookup speculation | already in ik (`-mtp`, n-gram / suffix-tree drafts); quillama adds grammar-forced drafts (7) |
| structured output by schema prompting + validation (Strata) | grammar-constrained decoding, which guarantees the schema instead of validating it afterwards (1, 2) |

## Environment variables

| Variable | Effect |
| --- | --- |
| `QUILL_AMX=0` | disable all AMX kernels |
| `QUILL_AMX_MIN_BATCH=N` | smallest batch for the AMX GEMM (default 32) |
| `QUILL_AMX_SLOW_DQ=1` | use `to_float()` instead of the AVX-512 dequantizers (debugging) |
| `QUILL_AMX_NO_REPACKED=1` | AMX skips the row-interleaved types (they go to ik's kernels) |
| `QUILL_GRAMMAR_REFERENCE=1` | original grammar masking (debugging) |
| `QUILL_GRAMMAR_TOPK=1` | exact top-k-aware grammar masking (experimental) |
| `QUILL_MOE_LOOKAHEAD=1` | same as `--moe-lookahead` |
| `QUILL_THP=0` | no huge pages for large CPU buffers |
| `QUILL_MOE_MIN_CHUNK=N` | minimum rows per expert chunk at decode (default 128) |
| `QUILL_MOE_CHUNKS=N` | force N chunks per expert |
| `QUILL_FA_VBLOCK=0` | original AVX2 flash-attention V accumulation |

## Tests

```bash
cmake -B build -DLLAMA_BUILD_TESTS=ON && cmake --build build -j
cd build && ctest -R "quill|grammar-integration-upstream"
```

- `test-quill-amx`: AMX GEMM, fused up/gate and flash attention against float/double references,
  for 18 weight types, odd sizes, multi-thread splits, GQA, causal masks and quantized KV caches.
- `test-quill-grammar`: the fast grammar mask equals the reference algorithm.
- `test-grammar-integration-upstream`: upstream llama.cpp's grammar / JSON-schema suite.
- `scripts/quill-json-schema-test.py`: end-to-end structured output through the server (run it
  against a random-weight model).

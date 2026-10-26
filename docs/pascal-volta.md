# Pascal and Volta GPUs

This page lists what the CUDA backend does on NVIDIA Pascal and Volta GPUs: which kernels run, what falls back
to the CPU, and what fails. It describes commit `3dbc2c9f8` (2026-10-13). Most of it is about flash attention
(FA), because that is where these GPUs differ most from newer ones.

## Quick answer: will my model run on the GPU?

With all layers offloaded (`-ngl 99`), the matrix multiplications run on the GPU. Flash attention (FA) is on by
default (`-fa on`). On these GPUs it runs on the GPU only for some models, KV cache types and batch sizes.
Otherwise the FA step runs on the CPU. The model still works, but more slowly: K and V are copied to host
memory and the CPU runs that step. If the CPU cannot run the KV cache type either, the program stops.

The table is for an f16 (default) or q8_0 KV cache. "GPU" means that every FA step runs on the GPU. "Partly
CPU" means that the steps in the notes run on the CPU.

The attention kernel depends on how many tokens one step processes, so the notes use token counts. A step
of 1 to 8 tokens is a generation step, and a step of more than 8 tokens is a prompt step. While generating,
a step has one token for each request served at once (up to `-np`), plus any draft tokens from speculative
decoding. More than 8 in total, for example 9 requests at once, makes it a prompt step. A prompt is read in
batches of `-ub` tokens (512 by default), and the last batch can be short. The MLA models also switch to their
second head size at 128 tokens (default `-mla 3`), so their notes mention batches under 128 tokens.

| model family (head size) | P100 | P40, GTX 10xx ‡ | V100, Titan V | notes |
|---|---|---|---|---|
| Llama 3.1, 3.3, Qwen2.5 1.5B and larger, Qwen3, Qwen3 MoE, Mistral-Small-3.x, Mistral-Medium-3.5, Devstral, GLM-4.5-Air, MiniMax-M2, Gemma 2 and 3 27B (128) | GPU † | GPU | GPU | |
| Gemma 1, Gemma 2 and 3 up to 12B, Qwen3.5, Qwen3.6, Qwen3-Next, Falcon3 (256) | GPU | GPU | GPU † | |
| Gemma-4 (512 in the global layers, 256 in the others) | GPU | GPU | partly CPU | V100: the global layers, every step |
| gpt-oss, Llama-3.2-1B, Qwen2.5-0.5B, SmolLM2 (64) | GPU | GPU | GPU | P100, P40 with a q8_0 cache: generation on the CPU |
| DeepSeek-V2, V2-Lite, Coder-V2, Kimi-K2.6, K2.7 (MLA: 576/512, 192/128) | partly CPU | partly CPU | partly CPU | P100, P40: prompts. V100: generation, and prompt batches under 128 tokens |
| GLM-4.7-Flash (MLA: 576/512, 256) | partly CPU † | partly CPU | partly CPU † | P100, P40: prompt batches of 9 to 127 tokens. V100: generation, and prompt batches under 128 tokens |
| Mistral-Small-4 (MLA: 320/256, 128) | partly CPU † | partly CPU | partly CPU † | generation, and prompt batches under 128 tokens |
| MiMo-V2, V2.5 (192/128) | partly CPU † | partly CPU | GPU † | P100, P40: prompts |
| DeepSeek-V4 (512, with sinks) | partly CPU † | partly CPU | partly CPU † | every FA step |

† From the tables below and the head sizes of the model, not run on a model of this family.

‡ Placement from a P100 that reports compute capability 6.1, as a P40 does. No model was run this way. See
[Not yet verified](#not-yet-verified).

What to do:

- For a "partly CPU" model, `-fa 0` turns FA off. Attention then runs as matrix multiplications and a softmax,
  which the GPU runs at every head size. A quantized V cache needs FA, so use `-fa 0` with an f16 cache. With
  several GPUs and `-sm graph`, the MLA models and Gemma-4 need FA. `-fa 0` was run on a P100 only. Its speed
  against FA on the CPU was not measured.
- Use an f16 or q8_0 KV cache. Other types can move FA to the CPU, also in the "GPU" rows. With q5_0, q5_1,
  bf16 or f32, the program then stops (see [When FA runs on the CPU](#when-fa-runs-on-the-cpu)). q8_KV crashes
  on these GPUs.

This table describes commit `3dbc2c9f8`.

## GPUs

| compute capability | architecture | cards | tested on |
|---|---|---|---|
| 6.0 (sm_60) | Pascal | Tesla P100 | P100 16 GB |
| 6.1 (sm_61) | Pascal | Tesla P40, Tesla P4, GeForce GTX 10xx | P40, in the tests of #2509, #2580 and #2647. FA placement: see [Not yet verified](#not-yet-verified) |
| 7.0 (sm_70) | Volta | Tesla V100, Titan V | Titan V |

The code paths depend on these checks (`ggml/src/ggml-cuda/common.cuh`):

| check | sm_60 | sm_61 | sm_70 | used for |
|---|---|---|---|---|
| `fast_fp16_available` | yes | no | yes | fp16 FA kernels |
| `__dp4a` (cc >= `MIN_CC_DP4A` = 610) | no | yes | yes | MMQ |
| `fp16_mma_available` (cc >= 700) | no | no | yes | wmma FA kernel |
| `new_mma_available` (Turing and newer) | no | no | no | mma FA kernels |

CUDA graphs are not used below Ampere (`cc < CC_AMPERE` in `ggml-cuda.cu`).

## Matrix multiplication

Quantized weights, `MUL_MAT`. n is the number of activation columns (the number of tokens in the batch).

| n | sm_60 | sm_61 | sm_70 |
|---|---|---|---|
| 1-8 | MMVQ | MMVQ | MMVQ |
| > 8 | dequantize to f32 + cuBLAS SGEMM | MMQ | MMQ |

- MMVQ covers the types in `ggml_cuda_mmvq_type_supported` (`mmvq.cu`). MMQ covers the types in
  `ggml_cuda_should_use_mmq` (`mmq.cu`). A quantized type that is not in the list for its n goes to cuBLAS.
- `ggml_cuda_should_use_mmq` returns false below cc 610 (`MIN_CC_DP4A`), so sm_60 never takes MMQ.
  `GGML_CUDA_FORCE_MMQ` does not change this, because this check comes first. On sm_61 and sm_70, MMQ is used
  for every n > 8.
- The code uses fp16 cuBLAS (`CUBLAS_COMPUTE_16F`) only at cc >= 700 and with `GGML_PREC_DEFAULT`. On Pascal,
  F16 weights and quantized weights that reach cuBLAS are converted to f32 and use SGEMM. On Volta they use
  fp16 cuBLAS.
- BF16 weights use `cublasGemmEx` with BF16 inputs and fp32 compute on all three architectures.

## Flash attention

For every FA node, the CUDA backend reports whether it can run it (`ggml_cuda_fattn_is_supported`,
`fattn.cu`). If it cannot, the scheduler runs that node on the CPU and copies its K and V to host memory.
This is decided per node, so in one model some layers can run FA on the GPU and others on the CPU.
`GGML_SCHED_DEBUG=1` prints the backend of each node.

n_q below is the number of query rows in the batch: 1 for single-sequence decode, 2-8 for small batches
(several slots, speculative decoding), more than 8 for prompt processing.

### Pascal (sm_60, sm_61), f16 KV cache

| Dk/Dv | n_q <= 8 | n_q > 8, sm_60 | n_q > 8, sm_61 |
|---|---|---|---|
| 64, 128 | vec_f32 | tile_f16 (tile_f32 with `GGML_PREC_F32`) | tile_f32 |
| 256 | vec_f32 | tile_f32 (#2580) | tile_f32 (#2580) |
| 512/512 | tile_new (#2647) | tile_new | tile_new |
| 192/128 | vec_f32 | CPU | CPU |
| 576/512 | vec_f32 (#2079) | CPU | CPU |
| 320/256, 192/192 | CPU | CPU | CPU |
| 32, 40, 72, 80, 88, 96, 112, 160 | CPU | CPU | CPU |

- sm_60 decode uses vec_f32, not the fp16 vec kernel (#2144). sm_61 is excluded from `fast_fp16_available`,
  so it always uses the f32 kernels.
- `-cuda fa-fabsum=1` replaces tile_f16 with its FABsum variant on sm_60 (#2510, see
  [parameters](parameters.md)).
- tile_new (head size 512) needs a mask, a GQA ratio that is a multiple of 4, and no sinks, ALiBi or softcap.
  If one of these is not met, the node runs on the CPU.
- For the other rows, mask, ALiBi, softcap, sinks and the GQA ratio do not change where the node runs.
  Sinks (gpt-oss) are applied in vec_f32 and in the tile kernels (tile: #2087).

### Volta (sm_70), f16 KV cache

| Dk/Dv | n_q = 1 | n_q 2-8 | n_q > 8 |
|---|---|---|---|
| 64, 128, 256 | vec_f32 | wmma | wmma |
| 80, 96, 112 | wmma | wmma | wmma |
| 192/128 | wmma | wmma, **wrong output** with `GGML_PREC_DEFAULT` (see below) | wmma |
| 512/512, 576/512, 320/256, 192/192 | CPU | CPU | CPU |
| 32, 40, 72, 88, 160 | CPU | CPU | CPU |

- wmma applies attention sinks (#2646) and reads the mask by its row stride (#2566).
- Mask, ALiBi, softcap, sinks, the GQA ratio and the precision do not change where the node runs.

### KV cache types

Pascal, n_q <= 8 (vec_f32, default build):

| Dk/Dv | K/V types that run on the GPU |
|---|---|
| 64 | f16 |
| 128 | f16, q8_0, q4_0, iq4_nl, and the pairs q8_0/iq4_nl, q6_0/q5_0, q8_0/q6_0 |
| 256 | f16, q8_0 |
| 192/128, 576/512 | f16, q8_0 (K and V of the same type) |

Every other type runs FA on the CPU at n_q <= 8, except at head size 512 (below). The tile kernels (n_q > 8 at
head sizes 64, 128 and 256) and tile_new (head size 512, every n_q) convert K and V to f16 before the launch.
So f32, f16, bf16, q8_0, q6_0, q5_0, q5_1, q4_0, q4_1 and iq4_nl run on the GPU there, in any pair.

Volta: wmma uses the same conversion. At head sizes 64, 96, 128, 256 and 192/128, all of these types run on the
GPU at every n_q. At 80 and 112 only f32, f16 and bf16 apply, because the quantized blocks are 32 wide. A quantized
cache at n_q = 1 also uses wmma. There is one exception, at n_q = 1 and head size 64, 128 or 256. If K and V are
each f32, f16 or bf16, but not both f16, the node goes to vec_f32. vec_f32 has no f32 or bf16 code, so the node
runs on the CPU.

q8_KV has no conversion to f16. See [What does not work](#what-does-not-work).

`GGML_CUDA_FA_ALL_QUANTS=ON` changes the vec_f32 type list at head sizes 64 and 128. Builds with this option
were not checked.

### When FA runs on the CPU

The CPU FA kernel (`iqk_flash_attn.cpp`) accepts f16, q8_0, q8_KV, q6_0, q4_0, q4_1 and iq4_nl for K and V.
q4_0, q4_1 and iq4_nl need `GGML_IQK_FA_ALL_QUANTS`, which is ON by default. For other types it stops the
process with `K cache ... coupled with V cache ... is not a supported combination on the CPU backend`:

- q5_0 or q5_1, as K or V
- f32, as K or V
- bf16, unless K and V are both bf16 and the CPU has AVX512-BF16.

So with these cache types, an FA node that the GPU does not take ends the run. A few nodes skip this check:
ALiBi, and some DeepSeek-V4 and DSA nodes.

### Model runs

Backend of the FA nodes, from `GGML_SCHED_DEBUG=1` with `-ngl 99 -fa 1`, all layers and the KV cache on the
GPU.

| model | KV cache | P100 | Titan V |
|---|---|---|---|
| gpt-oss-20b (head size 64, sinks) | f16 | GPU | GPU |
| gpt-oss-20b | q8_0 | prompt GPU, decode CPU | GPU |
| Llama-3.1-8B (head size 128) | q5_0 | prompt GPU, decode CPU, aborts | GPU |
| Llama-3.1-8B | bf16 | prompt GPU, decode CPU, aborts | prompt GPU, decode CPU, aborts |
| Llama-3.1-8B | `-ctk q8_KV -ctv f16` | crash (SIGSEGV) | crash (SIGSEGV) |
| DeepSeek-V2-Lite, `-mla 3`, 409-token prompt | f16 | prompt (192/128) CPU, decode (576/512) GPU | prompt GPU, decode CPU |
| DeepSeek-V2-Lite, `-mla 3`, 43-token prompt | f16 | prompt (576/512) CPU, decode GPU | prompt CPU, decode CPU |
| Gemma-4-E4B | f16 | global layers (512/512) GPU (#2647), SWA layers GPU | global layers CPU, SWA layers GPU |

## What does not work

| problem | GPUs | result | cause |
|---|---|---|---|
| FA 192/128 and 576/512 at n_q > 8: prompts of MLA models with `-mla 3`, for example DeepSeek-V2-Lite | Pascal | CPU | The tile kernels accept only Dk = Dv. vec_f32 has these shapes but is used only for n_q <= 8 |
| FA 576/512 at every n_q: decode and short prompts of MLA models with `-mla 3` | Volta | CPU | The vec_f32 route for this shape (n_q <= 8) is only in the pre-Volta branch of `fattn.cu`. wmma takes Dk <= 256 |
| FA 512/512: Gemma-4 global layers | Volta | CPU | The tile_new route is limited to cc 6.x (`is_pascal_hs512`, `fattn.cu`) |
| FA 512/512 with sinks | all | CPU | tile_new does not accept sinks |
| FA 320/256 | all | CPU | No kernel accepts it before Turing: vec_f32 accepts only 192/128 and 576/512 with Dk != Dv, tile accepts only Dk = Dv, wmma needs Dk <= 256 |
| FA 192/192, 32, 40, 72, 88, 160, and 80, 96, 112 on Pascal | all | CPU | No kernel case for these head sizes |
| FA decode with a quantized KV type not in the vec_f32 list, for example gpt-oss-20b with q8_0 | Pascal | CPU, or abort for q5_0 and q5_1 | No vec_f32 instance |
| FA decode with a bf16 or f32 KV cache | all | CPU, abort (bf16 K and V run on a CPU with AVX512-BF16) | Pascal: vec_f32 has no bf16 or f32 code. Volta: at n_q = 1 and head size 64, 128 or 256, an unquantized cache is sent to vec_f32 |
| `-ctk q8_KV` | all | crash (SIGSEGV) | CUDA has no f32 -> q8_KV copy, so the K-cache write runs on the CPU into the GPU KV buffer. There is also no q8_KV -> f16 conversion for the FA kernels |
| FA with softcap at a head size other than 128 or 256 | all | kernel trap, abort | The softcap variants of the kernels are compiled only for head sizes 128 and 256 |
| FA 192/128 at n_q 2-8 with `GGML_PREC_DEFAULT`: DeepSeek-V2 and Kimi-K2 with `-mla 0`, in a batch or a last prompt ubatch of 2-8 tokens | Volta | wrong output | The 8-column wmma kernel stores the result up to Dk = 192 instead of Dv = 128 and overwrites the next column |

## In progress

- Faster batched decode on P100 for batches of 9-16 tokens: MMVQ for up to 16 columns on sm_60 (#2668).
- A tile FA kernel for Volta.
- The FA cases above that still run on the CPU on these GPUs.

## Not yet verified

- sm_61 FA placement comes from a P100 that reports compute capability 6.1 to the backend. On a real sm_61 card
  (P40), FA kernels ran only in the tests of #2580 (head size 256) and #2647 (head size 512). For the other sm_61
  cells, the kernel names come from the source.
- Volta was tested on a Titan V only, not on a V100. sm_62, the P4 and GeForce GTX 10xx cards were not tested.
- Models that were not run, but whose FA shapes (from GGUF metadata and graph code) are in the CPU rows above:
  Mistral-Small-4 (320/256), DeepSeek-V4 (512/512 with sinks), MiMo-V2 (192/128, n_q > 8 on Pascal) and
  GLM-4.7-Flash (576/512).
- The q5_0 and q5_1 aborts on Volta come from the same CPU code as on Pascal. They were not run on Volta.
- On Volta, a quantized KV cache is converted to f16 on every FA call. How this cost grows with context was
  not measured.

## Build

```bash
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="60;61;70"
cmake --build build --config Release
```

- Use the architectures of the GPUs you have.
- Without `CMAKE_CUDA_ARCHITECTURES`, the build uses `native` when `GGML_NATIVE` is on (the default unless
  cross-compiling), the CUDA toolkit is 11.6 or newer, and CMake is 3.24 or newer. Otherwise it uses
  `50;61;70;75;80`, or `60;61;70;75;80` with `GGML_CUDA_F16`. The first list has no `60`, so for a P100 set
  the list.
- CUDA 13.0 removed offline compilation for Maxwell, Pascal and Volta. Use a CUDA 12.x toolkit. NVIDIA
  recommends 12.9 and the 580 driver branch, which is the last branch that supports these GPUs. The P100 and
  Titan V tests for this page used CUDA 12.4.
- `GGML_CUDA_FA_ALL_QUANTS` (default OFF) adds vec FA kernels for more KV types. `GGML_IQK_FA_ALL_QUANTS`
  (default ON) adds q4_0, q4_1 and iq4_nl to the CPU FA kernel.

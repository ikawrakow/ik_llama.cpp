# Parameters Documentation

Overview of the most common command-line parameters in `ik_llama.cpp`. 

## Table of Contents

- [General Parameters](#general-parameters)

- [Speculative Decoding](#speculative-decoding)

- [Cache Prompt to Host Memory](#cache-prompt-to-host-memory)

- [Sampling](#sampling)

- [Template](#prompt-template)

- [Context Hacking](#context-hacking)

- [Parallel Processing](#parallel-processing)

- [Backend Options](#backend-options)

- [GPU Offload](#gpu-offload)

- [Model Options](#model-options)

- [Server Options](#server-options)

- [Other Tools](#other-tools)

## General Parameters

| Parameter | Description | Default | Notes/Examples |
| - | - | - | - |
| `-h, --help, --usage` | Print usage and exit | - | - |
| `-t, --threads N` | Number of threads to use during generation | 4 | Try to match the number of physical CPU cores. Avoid odd numbers (e.g. 1,3,...). |
| `-tb, --threads-batch N` | Number of threads to use during batch and prompt processing | Same as `--threads` | Same as `--threads` When doing full GPU offload, use a lower number (e.g. 2) |
| `-c, --ctx-size N` | Size of the prompt context | 0 (loaded from model) | Influences the size of KV size (memory) therefore look for a value that fits your system then increase as needed (2048, 4096,…). If you use parallel slots, this context size will be split across the slots. |
| `-n, --predict N` | Number of tokens to predict | -1 (infinity), -2 (until context filled) | -1 (infinity), -2 (until context filled). Safe to leave default. |
| `-b, --batch-size N` | Logical maximum batch size | 2048 | Safe to leave default. Higher values may give better t/s especially on GPU, while using more memory. |
| `-ub, --ubatch-size N` | Physical maximum batch size | 512 | Safe to leave default. Similar to `--batch-size N` |
| `--keep N` | Number of tokens to keep from the initial prompt | 0 | (-1 = all) |
| `--chunks N` | Max number of chunks to process | -1 (all) |  |
| `-fa, --flash-attn` | Enables Flash Attention | on | auto / on / off Improves t/s and reduces memory usage. |
| `--no-fa, --no-flash-attn` | Disable Flash Attention |  | Alternative parameter to turn of FA. See `--flash-attn` |
| `-mla, --mla-use` | Enable MLA | 3 | 0 / 1 / 2 / 3 For DeepSeek models, and other recent models that are using MLA. [PR 188](https://github.com/ikawrakow/ik_llama.cpp/pull/188) [PR 205](https://github.com/ikawrakow/ik_llama.cpp/pull/205) [PR 235](https://github.com/ikawrakow/ik_llama.cpp/pull/235) [PR 243](https://github.com/ikawrakow/ik_llama.cpp/pull/243) [PR 252](https://github.com/ikawrakow/ik_llama.cpp/pull/252) [PR 253](https://github.com/ikawrakow/ik_llama.cpp/pull/253) [PR 273](https://github.com/ikawrakow/ik_llama.cpp/pull/273) [PR 386](https://github.com/ikawrakow/ik_llama.cpp/pull/386) [PR 497](https://github.com/ikawrakow/ik_llama.cpp/pull/497) [PR 943](https://github.com/ikawrakow/ik_llama.cpp/pull/943)|
| `-amb, --attention-max-batch` | Max batch size for attention computations | 0 | Specifies the maximum K*Q size in MB we want to tolerate. [PR 237](https://github.com/ikawrakow/ik_llama.cpp/pull/237) |
| `-fmoe or --fused-moe` | Fused MoE ffn_up and ffn_gate | - | Speedup for MoE models. [PR 229](https://github.com/ikawrakow/ik_llama.cpp/pull/229) |
| `--no-fmoe, --no-fused-moe` | Disable fused MoE | Enabled | See `--fused-moe` |
| `-ger, --grouped-expert-routing` | Enable grouped expert routing | Disabled | For BailingMoeV2 architecture (Ling/Ring models). [PR 836](https://github.com/ikawrakow/ik_llama.cpp/pull/836) [PR 838](https://github.com/ikawrakow/ik_llama.cpp/pull/838) |
| `--no-fug, --no-fused-up-gate` | Disable fused up-gate | Enabled | Turn off the speedup for dense models. [PR 741](https://github.com/ikawrakow/ik_llama.cpp/pull/741) |
| `--no-mmad, --no-fused-mul-multiadd` | Disable fused mul-multi_add | Enabled | [PR 858](https://github.com/ikawrakow/ik_llama.cpp/pull/858) |
| `-gr, --graph-reuse` | Enable graph reuse | Enabled | For models with fast TG inference (100+ t/s). [PR 947](https://github.com/ikawrakow/ik_llama.cpp/pull/947) |
| `--no-gr, --no-graph-reuse` | Disable graph reuse | Disabled | Option to turn off graph reuse. [PR 1094](https://github.com/ikawrakow/ik_llama.cpp/pull/1094) |
| `-ser, --smart-expert-reduction` | Experts reduction Kmin,t | -1, 0 | Use a custom number of active experts. Powerful, basically REAP from just command line. If we set t = 1, we use a fixed number of experts  K_min (`-ser 1,6` will use 6 experts instead of the model default). [PR 239](https://github.com/ikawrakow/ik_llama.cpp/pull/239) |
| `-mqkv, --merge-qkv` | Merge Q,K,V | 0 | Downside: mmap cannot be used. [PR 878](https://github.com/ikawrakow/ik_llama.cpp/pull/878) [PR 892](https://github.com/ikawrakow/ik_llama.cpp/pull/892) |
| `-muge, --merge-up-gate-experts` | Merge ffn_up/gate_exps | 0 | Speed up on some models. [PR 1137](https://github.com/ikawrakow/ik_llama.cpp/pull/1137) [PR 1139](https://github.com/ikawrakow/ik_llama.cpp/pull/1139) |
| `-khad, --k-cache-hadamard` | Use Hadamard transform for K-cache | 0 | May improve KV quality when heavily quantized. [PR 1033](https://github.com/ikawrakow/ik_llama.cpp/pull/1033) [PR 1034](https://github.com/ikawrakow/ik_llama.cpp/pull/1034) |
| `-sas, --scheduler_async` | Async evaluation of compute graphs | 0 | [PR 1089](https://github.com/ikawrakow/ik_llama.cpp/pull/1089) |
| `-vq, --validate-quants` | Validate quantized data while loading the model | 0 | If there are NaNs in the model, you will get info about the tensors containing NaNs. [PR 977](https://github.com/ikawrakow/ik_llama.cpp/pull/977) |
| `-sp, --special` | Special tokens output enabled | false |  |
| `--no-warmup` | Skip warming up the model with an empty run | - |  |

## Speculative Decoding

| Parameter | Description | Default | Notes/Examples |
| - | - | - | - |
| `-td, --threads-draft N` | Number of threads to use during generation | Same as `--threads` |  |
| `-tbd, --threads-batch-draft N` | Number of threads to use during batch and prompt processing | Same as `--threads-draft` |  |
| `-ps, --p-split N` | Speculative decoding split probability | 0.1 |  |
| `-cd, --ctx-size-draft N` | Size of the prompt context for the draft model | 0 (loaded from model) | Similar to `--ctx-size` but applied to the draft model, if used. |
| `-ctkd, --cache-type-k-draft TYPE` | KV cache data type for K for the draft model | - | For draft model, see: `-ctk` |
| `-ctvd, --cache-type-v-draft TYPE` | KV cache data type for V for the draft model | - | For draft model, see: `-ctk` |
| `-draft, --draft-params` | Comma-separated list of draft model parameters | - |  |

## Cache Prompt to Host Memory

| Parameter | Description | Default | Notes/Examples |
| - | - | - | - |
| `-cram, --cache-ram N` | Set the maximum cache size in MiB | 8192 (-1 = no limit, 0 = disable) | Very useful when the variations of the same prompt are re-sent to the model (coding agents, etc.). [PR 954](https://github.com/ikawrakow/ik_llama.cpp/pull/954) |
| `-crs, --cache-ram-similarity N` | Max similarity of prompt tokens to cache tokens that triggers prompt cache | 0.50 |  |
| `-cram-n-min, --cache-ram-n-min N` | Minimum number of cached tokens that triggers prompt cache | 0 |  |

## Sampling

| Parameter | Description | Default | Notes/Examples |
| - | - | - | - |
| `--samplers SAMPLERS` | Samplers used for generation in order, separated by `;` | dry;top_k;tfs_z;typical_p;top_p;min_p;xtc;top_n_sigma;temperature;adaptive_p | Powerful option to customize samplers. Try to keep the order otherwise. Example to use only min_p and temperature: `--samplers min_p;temperature` |
| `--sampling-seq SEQUENCE` | Simplified sequence for samplers | dkfypmxntw | Same as `--samplers` , just shorter format. |

## Prompt Template

| Parameter | Description | Default | Notes/Examples |
| - | - | - | - |
| `--jinja` | Set custom jinja chat template | Template taken from model's metadata | Mandatory for Tool Calling. |
| `--chat-template JINJA_TEMPLATE` | Use jinja template for chat | Disabled | If there is no official `tool_use` Jinja template, you may want to set `--chat-template chatml` to use a default that works with many models |
| `--chat-template-file file_with_JINJA_TEMPLATE` | Load jinja template for chat from the file | - | Sometimes the model producer or community fixes the template after the GGUF files are released, therefore it’ metadata contains buggy version. To avoid re-downloading the entire model file, download only the .jinja file the use it (`--chat-template-file /models/Qwen_Qwen3-Coder-30B-A3B-Instruct-fixed.jinja`). |
| `--reasoning-format FORMAT` | Controls whether thought tags are allowed and/or extracted from the response | none | One of:  - none: leaves thoughts unparsed in `message.content`  - deepseek: puts thoughts in `message.reasoning_content` (except in streaming mode, which behaves as `none`)  - deepseek-legacy: keeps `<think>` tags in `message.content` while also populating `message.reasoning_content` |
| `--chat-template-kwargs JSON` | Sets additional params for the json template parser | - | Example for gpt-oss: `--chat-template-kwargs '{"reasoning_effort": "medium"}'` |
| `--reasoning-budget N` | Controls the amount of thinking allowed | -1 (unrestricted), 0 (disable thinking) |  |
| `--reasoning-tokens FORMAT` | Exclude reasoning tokens to select the slot more accurately | auto |  |

## Context Hacking

| Parameter | Description | Default | Notes/Examples |
| - | - | - | - |
| `-dkvc, --dump-kv-cache` | Verbose print of the KV cache | - |  |
| `-nkvo, --no-kv-offload` | Disable KV offload | - | Keep KV on CPU. |
| `-ctk, --cache-type-k TYPE` | KV cache data type for K | f16 | Reduces K size in KV which improves speed and reduces memory requirements, but may reduce output quality. |
| `-ctv, --cache-type-v TYPE` | KV cache data type for V | f16 | See: `-ctk` |
| `--no-context-shift` | Disable context-shift | - |  |
| `--context-shift` | Set context-shift | on | auto / on / off / 0 / 1 |

## Parallel Processing

| Parameter | Description | Default | Notes/Examples |
| - | - | - | - |
| `-np, --parallel N` | Number of parallel sequences to decode | 1 | Useful when fronted support it. See `--ctx-size` |

## Backend Options

| Parameter | Description | Default | Notes/Examples |
| - | - | - | - |
| `--mlock` | Force system to keep model in RAM rather than swapping or compressing | - |  |
| `--no-mmap` | Do not memory-map model (slower load but may reduce pageouts) | - |  |
| `-rtr, --run-time-repack` | Repack tensors if interleaved variant is available | - | May improve performance on some systems. [PR 147](https://github.com/ikawrakow/ik_llama.cpp/pull/147) |

## GPU Offload

| Parameter | Description | Default | Notes/Examples |
| - | - | - | - |
| `-ngl, --gpu-layers N` | Number of layers to store in VRAM | - | For better speed you aim to offload the entire model in GPU memory. To identify how many layers (also shape and more metadata) open the GGUF model file on browser [bartowski/Qwen_Qwen3-0.6B-IQ4_NL.gguf](https://huggingface.co/bartowski/Qwen_Qwen3-0.6B-GGUF/blob/main/Qwen_Qwen3-0.6B-IQ4_NL.gguf) then scroll down to the Tensors table. Use a number higher than the numbers of model layers to full offload (`--gpu-layers` 99, for a model with less than 99 layers). See `--ctx-size` and reduce it to the minimum needed. If model fails to load due to the insufficient GPU memory, reduce the number of layers (`--gpu-layers 20`, for a model with 40 layers will offload only the first 20 layers). |
| `-ngld, --gpu-layers-draft N` | Number of layers to store in VRAM for the draft model | - | For draft model, see `--gpu-layers` |
| `--cpu-moe` | Keep all MoE weights in CPU memory | - | Simple offload mode for MoE. [PR 841](https://github.com/ikawrakow/ik_llama.cpp/pull/841) |
| `--n-cpu-moe N` | Keep MoE weights of the first N layers in CPU memory | - | Similar to `--cpu-moe` but when some GPU memory is available to store some layers. |
| `-sm, --split-mode SPLIT_MODE` | How to split the model across multiple GPUs | none | When you have more than one GPU, how to split the model across multiple GPUs, one of: - none: use one GPU only. - graph: split model tensors and computation graph across GPUs. `graph` is exclusive here and extremely effective for dense and MoE [PR 1080](https://github.com/ikawrakow/ik_llama.cpp/pull/1080). - layer: split layers and KV across GPUs Example: `-sm graph ` |
| `-ts, --tensor-split SPLIT` | Fraction of the model to offload to each GPU (comma-separated) | - | Powerful for tweaking. Example: `-ts 3,1` |
| `-dev, --device dev1,dev2` | Comma-separated list of devices to use for offloading | none | If there are many GPUs available on the system and only selected ones need to be used. Example: `-dev  CUDA0,CUDA1` |
| `-devd, --device-draft dev1,dev2` | Comma-separated list of devices for draft model | none | For draft model, see `--device` |
| `-mg, --main-gpu i` | The GPU to use for the model (with split-mode = none) | - |  |
| `-cuda fa-offset=value` | FP16 precision offset for FA calculation | 0 | Rarely, fp16 precision is inadequate, at least for some models, when computing FA for very long contexts. Value must be a valid floating point number in the interval [0...3] (this is checked and if the supplied value is outside this interval it is ignored). By the default the offset is zero. If you find that a model works up to a given context length but then starts producing gibberish/incoherent output/endless repetitions, it is very likely it is due to f16 overflow in the FA calculation, and using this command line option is likely to solve it. [PR 1198](https://github.com/ikawrakow/ik_llama.cpp/pull/1198) |
| `-ot or --override-tensor` | Override where model weights are stored | - | Override where model weights are stored using regular expressions. This allows for example to keep the MoE experts on the CPU and to offload only the attention and not repeating layers to the GPU. Example: `\.ffn_.*_exps\.=CPU` [PR 232](https://github.com/ikawrakow/ik_llama.cpp/pull/232) |
| `-op or --offload-policy a,b` | Manually define the offload policy | - | a and b are integers. One can have multiple pairs following the -op or --offload-policy argument (i.e., -op a1,b1,a2,b2,a3,b3...). The first integer defines the op (see below). The second integer is 0 or 1 and defines if the op should be offloaded (1) or not offloaded (0) to the GPU. The first integer is simply the `enum` value in the `ggml_op enum`. If the op is set to -1, then all op offloads are set to enabled or disabled.  Examples: `-op -1,0`: disable all offload to the GPU `-op 26,0`: disable offload of matrix multiplications to the GPU `-op 27,0`: disable offload of indirect matrix multiplications to the GPU (used for the experts in a MoE model) `-op 29,0`: disable fused up-gate-unary op offload to the GPU (applied to MoE models with `-fmoe`) [PR 405](https://github.com/ikawrakow/ik_llama.cpp/pull/405) |
| `--offload-only-active-experts or -ooae` | On MOE offload only active experts | - | [PR 698](https://github.com/ikawrakow/ik_llama.cpp/pull/698) |
| `-smf16, --split-mode-f16` | Use f16 for data exchange between GPUs | 1 | [PR 1087](https://github.com/ikawrakow/ik_llama.cpp/pull/1087) |
| `-smf32, --split-mode-f32` | Use f32 for data exchange between GPUs | 0 | [PR 1087](https://github.com/ikawrakow/ik_llama.cpp/pull/1087) |
| `-grt, --graph-reduce-type` | Type for data exchange between GPUs | f32 | q8_0 / bf16 / f16 / f32 Reduce the data transferred between GPUs [PR 1154](https://github.com/ikawrakow/ik_llama.cpp/pull/1154) |
| `-smgs, --split-mode-graph-scheduling` | Force Split Mode Graph Scheduling | 0 | [PR 1068](https://github.com/ikawrakow/ik_llama.cpp/pull/1068) |
| `--max-gpu N` | Define (and use) a maximum number of GPUs per layer with split mode "graph" |  | This is of interest when there are more than 2 GPUs available, but using all of them leads to a lower performance than using just 2 (or using the default split mode "layer") [PR 1051](https://github.com/ikawrakow/ik_llama.cpp/pull/1051) |
| `-cuda, --cuda-params` | Comma-separated list of cuda parameters | - | Powerful way to tweak Fusion, GPU offload threshold, and MMQ-ID threshold. [PR 910](https://github.com/ikawrakow/ik_llama.cpp/pull/910) |

## Model Options

| Parameter | Description | Default | Notes/Examples |
| - | - | - | - |
| `--check-tensors` | Check model tensor data for invalid values | false |  |
| `--override-kv KEY=TYPE:VALUE` | Override model metadata by key | - | Advanced option to override model metadata by key. May be specified multiple times. types: int, float, bool, str. Example: `--override-kv tokenizer.ggml.add_bos_token=bool:false` |
| `-m, --model FNAME` | Model path | models/$filename | Mandatory, the GGUF model file to be served. |
| `-md, --model-draft FNAME` | Draft model for speculative decoding | unused |  |
| `--draft-max, --draft, --draft-n N` | Number of tokens to draft for speculative decoding | 16 |  |
| `--draft-min, --draft-n-min N` | Minimum number of draft tokens to use for speculative decoding | - |  |
| `--draft-p-min P` | Minimum speculative decoding probability (greedy) | 0.8 |  |

## Server Options

| Parameter | Description | Default | Notes/Examples |
| - | - | - | - |
| `--host HOST` | IP address to listen | 127.0.0.1 | Change to 0.0.0.0 when endpoint will be accessed from another computer. Keep in mind to never expose the server to Internet. |
| `--port PORT` | Port to listen | 8080 |   |
| `--webui NAME` | Controls which webui to server | auto | Flexibility in choosing the integrated powerful WebUIs:  - `none`: disable webui  - `auto`: default webui   - `llamacpp`: llamacpp webui |
| `--api-key KEY` | API key to use for authentication | none | Add a custom API KEY. Clients will need to specify it when connecting. |

## Other Tools

### sweep_bench

Benchmark utility that performs a series of prompt processing batches followed by TG.
The KV cache is not cleared, so the N_KV columns tells you how many tokens were in the KV cache when the PP/TG was processed.

```
llama-sweep-bench -m /models/model.gguf -c 12288 -ub 512 -rtr -fa -ctk q8_0 -ctv q8_0
```

| Parameter | Description | Default | Notes/Examples |
| - | - | - | - |
| `-nrep N, --n-repetitions N` | Define the number of repetitions used at zero context | - | [PR 1176](https://github.com/ikawrakow/ik_llama.cpp/pull/1176) |
| `-n` | Specifies he number of TG tokens  | - | If not specified, it is set to u-batch/4 [PR 897](https://github.com/ikawrakow/ik_llama.cpp/pull/897) | 

### llama-bench

Benchmark utility.

```
llama-bench -tgb 4,16 -p 512 -n 128 other_arguments
```

| Parameter | Description | Default | Notes/Examples |
| - | - | - | - |
| `-tgb (or --threads-gen-batch)` | Enable having different number of threads for generation and batch processing | - | [PR 284](https://github.com/ikawrakow/ik_llama.cpp/pull/284) |

### Imatrix

Create imatrix from calibration dataset.

```
llama-imatrix -m /models/model-bf16.gguf -f /models/calibration_data_v5_rc.txt -o /models/model.imatrix
```

| Parameter | Description | Default | Notes/Examples |
| - | - | - | - |
| `--layer-similarity or -lsim` | Collect statistics about activations change caused by a layer using cosine similarity | - | [PR 328](https://github.com/ikawrakow/ik_llama.cpp/pull/328) |
| `--hide-imatrix` | Store "top_secret" in the imatrix data file name | - | And in calibration dataset fields, and zeros in the batch size and number of chunks used to compute the imatrix. [PR 329](https://github.com/ikawrakow/ik_llama.cpp/pull/329) |

### Quantization

Quantize models to reduce size and improve speed.

```
llama-quantize --imatrix /models/model.imatrix /models/model-bf16.gguf /models/model-IQ4_NL.gguf IQ4_NL
```

| Parameter | Description | Default | Notes/Examples |
| - | - | - | - |
| `--custom-q` | Custom quantization rules with regular expressions | - | Example: `llama-quantize --imatrix some_imatrix --custom-q "regex1=typ1,regex2=type2..." some_model some_output_file some_base_quant` [PR 244](https://github.com/ikawrakow/ik_llama.cpp/pull/244) |

### Build Arguments

Build with `cmake`.

```
cmake -B build -DGGML_NATIVE=ON

cmake --build build --config Release -j$(nproc)
```

| Argument | Notes/Examples |
| - | - |
| `-DGGML_ARCH_FLAGS="-march=armv8.2-a+dotprod+fp16"` | Direct access to ARCH options. |
| `-DGGML_CUDA=ON` | Build with CUDA support. By default it builds to native CUDA. |
| `-DCMAKE_CUDA_ARCHITECTURES=86` | Build for specific CUDA GPU Compute Capability, e.g. 8.6 for RTX30*0 |
| `-DGGML_RPC=ON` | Build the RPC backend. |
| `-DGGML_IQK_FA_ALL_QUANTS=ON` | More KV quantization types |
| `-DLLAMA_SERVER_SQLITE3=ON` | Sqlite3 for mikupad |
| `-DCMAKE_TOOLCHAIN_FILE=[...]` | Example: on Windows tells `cmake` where is `sqlite3`. |
| `-DGGML_NATIVE=ON` | Turn off when cross-compiling. |
| `-DGGML_NCCL=OFF` | To disable usage of NCCL. |

### Environment variables

Use them on the command line.

```
CUDA_VISIBLE_DEVICES=0,2 llama-server -m /models/model-bf16.gguf
```

| Name | Notes/Examples |
| - | - |
| CUDA_VISIBLE_DEVICES | Use only specified GPUs. Example: Use first and 3rd `CUDA_VISIBLE_DEVICES=0,2` |


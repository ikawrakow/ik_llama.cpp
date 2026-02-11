# General
-h, --help, --usage
print usage and exit

-t, --threads N
number of threads to use during generation (default: 4)

-tb, --threads-batch N
number of threads to use during batch and prompt processing (default: same as --threads)

-c, --ctx-size N
size of the prompt context (default: 0, 0 = loaded from model)

-cd, --ctx-size-draft N
size of the prompt context for the draft model (default: 0, 0 = loaded from model)

-n, --predict N
number of tokens to predict (default: -1, -1 = infinity, -2 = until context filled)

-b, --batch-size N
logical maximum batch size (default: 2048)

-ub, --ubatch-size N
physical maximum batch size (default: 512)

--keep N
number of tokens to keep from the initial prompt (default: 0, -1 = all)

--chunks N
max number of chunks to process (default: -1, -1 = all)

-no-fa, --no-flash-attn
disable Flash Attention (default: enabled)

-fa, --flash-attn (auto|on|off|0|1)
set Flash Attention (default: on)

-mla, --mla-use
enable MLA (default: 3)

-amb, --attention-max-batch
max batch size for attention computations (default: 0)

-no-fmoe, --no-fused-moe
disable fused MoE (default: enabled)

-ger, --grouped-expert-routing 
enable grouped expert routing (default: disabled)

-no-fug, --no-fused-up-gate
disable fused up-gate (default: enabled)

-no-mmad, --no-fused-mul-multiadd
disable fused mul-multi_add (default: enabled)

-gr, --graph-reuse
enable graph reuse (default: enabled)

-no-gr, --no-graph-reuse
disable graph reuse (default: disabled)

-ser, --smart-expert-reduction 
experts reduction (default: -1,0)

-mqkv, --merge-qkv
merge Q,K,V (default: 0) (downside is that mmap cannot be used)

-muge, --merge-up-gate-experts,
merge ffn_up/gate_exps (default: 0)

-khad, --k-cache-hadamard
Use Hadamard transform for K-cache (default: 0)

-smf16, --split-mode-f16
Use f16 for data exchange between GPUs (default: 1)

-smf32, --split-mode-f32
Use f32 for data exchange between GPUs (default: 0)

-grt, --graph-reduce-type
Type for data exchange between GPUs (default: f32)

-smgs, --split-mode-graph-scheduling
Force Split Mode Graph Scheduling (default: 0)

-sas, --scheduler_async
Async evaluation of compute graphs: 0

-vq, --validate-quants
validate quantized data while loading the model (default: 0)

-sp, --special
special tokens output enabled (default: false)

--no-warmup
skip warming up the model with an empty run

-fmoe or --fused-moe
Fused MoE ffn_up and ffn_gate

# Speculative
-td, --threads-draft N
number of threads to use during generation (default: same as --threads)

-tbd, --threads-batch-draft N
number of threads to use during batch and prompt processing (default: same as --threads-draft)

-ps, --p-split N
speculative decoding split probability (default: 0.1)

# Cache prompt to host memory
-cram, --cache-ram N
set the maximum cache size in MiB (default: 8192, -1 - no limit, 0 - disable)

-crs, --cache-ram-similarity N
max of similarity of prompt tokens to cache tokens that triggers prompt cache (default: 0.50).

-cram-n-min --cache-ram-n-min N
minimum number of the cached tokens that triggers prompt cache (default: 0).

# Sampling
--samplers SAMPLERS
samplers that will be used for generation in the order, separated by ';'
(default: dry;top_k;tfs_z;typical_p;top_p;min_p;xtc;top_n_sigma;temperature;adaptive_p)

--sampling-seq SEQUENCE
simplified sequence for samplers that will be used (default: dkfypmxntw)

# Template
--jinja
set custom jinja chat template (default: template taken from model's metadata)

--chat-template JINJA_TEMPLATE
use jinja template for chat (default: disabled)

--chat-template-file file_with_JINJA_TEMPLATE
load jinja template for chat from the file
                        
--reasoning-format FORMAT
controls whether thought tags are allowed and/or extracted from the response, and in which format they're returned; one of:
 - none: leaves thoughts unparsed in `message.content`
 - deepseek: puts thoughts in `message.reasoning_content` (except in streaming mode, which behaves as `none`)
 - deepseek-legacy: keeps `<think>` tags in `message.content` while also populating `message.reasoning_content`
 - (default: none)

--chat-template-kwargs JSON
sets additional params for the json template parser

--reasoning-budget N
controls the amount of thinking allowed.
currently only one of: -1 for unrestricted thinking budget, or 0 to disable thinking(default: -1)

--reasoning-tokens FORMAT
exclude reasoning tokens to select the slot more accurately.
 - none: include all tokens
 - auto: exclude all tokens between <think> and </think>
 - Or comma separated start and end tokens such as [THINK],[/THINK]
 - (default: auto)

# Context hacking
-dkvc, --dump-kv-cache
verbose print of the KV cache

-nkvo, --no-kv-offload
disable KV offload

-ctk, --cache-type-k TYPE
KV cache data type for K (default: f16)

-ctv, --cache-type-v TYPE
KV cache data type for V (default: f16)
-ctkd, --cache-type-k-draft TYPE
KV cache data type for K for the draft model

-ctvd, --cache-type-v-draft TYPE
KV cache data type for V for the draft model

--no-context-shift
disable context-shift.

--context-shift (auto|on|off|0|1)
set context-shift (default: on)

# Parallel
-np, --parallel N
number of parallel sequences to decode (default: 1)

# Backend
-cuda, --cuda-params
comma separate list of cuda parameters

-draft, --draft-params
comma separate list of draft model parameters

--mlock
force system to keep model in RAM rather than swapping or compressing

--no-mmap
do not memory-map model (slower load but may reduce pageouts if not using mlock)

--run-time-repack
repack tensors if interleaved variant is available

--cpu-moe
keep all MoE weights in CPU memory

--n-cpu-moe N
keep MoE weights of the first N layers in CPU memory

# GPU offload
-ngl, --gpu-layers N
number of layers to store in VRAM

-ngld, --gpu-layers-draft N
number of layers to store in VRAM for the draft model

-sm, --split-mode SPLIT_MODE
how to split the model across multiple GPUs, one of:
 - none: use one GPU only
 - graph: split model tensors and computation graph across GPUs
 - layer (default): split layers and KV across GPUs

-ts, --tensor-split SPLIT
fraction of the model to offload to each GPU, comma-separated list of proportions, e.g. 3,1

-dev, --device dev1,dev2
comma-separated list of devices to use for offloading (none = don't offload)
Example: CUDA0,CUDA1,RPC[192.168.0.1:8080]

-devd, --device-draft dev1,dev2
comma-separated list of devices to use for offloading for the draft model (none = don't offload)
Example: CUDA0,CUDA1,RPC[192.168.0.1:8080]

-mg, --main-gpu i
the GPU to use for the model (with split-mode = none), 

-cuda fa-offset=value
Rarely, fp16 precision is inadequate, at least for some models, when computing FA for very long contexts.
Value must be a valid floating point number in the interval [0...3] (this is checked and if the supplied value is outside this interval it is ignored).
By the default the offset is zero. If you find that a model works up to a given context length but then starts producing gibberish/incoherent output/endless repetitions, it is very likely it is due to f16 overflow in the FA calculation, and using this command line option is likely to solve it.

-ot or --override-tensor
Override where model weights are stored

-op or --offload-policy a,b 
manually define the offload policy
 - where a and b are integers. One can have multiple pairs following the -op or --offload-policy argument (i.e., -op a1,b1,a2,b2,a3,b3...). The first integer defines the op (see below). The second integer is 0 or 1 and defines if the op should be offloaded (1) or not offloaded (0) to the GPU. The first integer is simply the enum value in the ggml_op enum.
 - If the op is set to -1, then all op offloads are set to enabled or disabled.
Examples:
-op -1,0: disable all offload to the GPU
-op 26,0: disable offload of matrix multiplications to the GPU
-op 27,0: disable offload of indirect matrix multiplications to the GPU (used for the experts in a MoE model)
-op 29,0: disable fused up-gate-unary op offload to the GPU (applied to MoE models with -fmoe)

--offload-only-active-experts or -ooae
On MOE offload only active experts

# Model
--check-tensors
check model tensor data for invalid values (default: false)

--override-kv KEY=TYPE:VALUE
advanced option to override model metadata by key. may be specified multiple times.
types: int, float, bool, str. example: --override-kv tokenizer.ggml.add_bos_token=bool:false

-m, --model FNAME
model path (default: models/$filename with filename from --hf-file
or --model-url if set, otherwise models/7B/ggml-model-f16.gguf)

-md, --model-draft FNAME
draft model for speculative decoding (default: unused)

--draft-max, --draft, --draft-n N
number of tokens to draft for speculative decoding (default: 16)

--draft-min, --draft-n-min N
minimum number of draft tokens to use for speculative decoding

--draft-p-min P
minimum speculative decoding probability (greedy) (default: 0.8)

# Server
--host HOST
ip address to listen (default: 127.0.0.1)

--port PORT
port to listen (default: 8080)

--webui NAME
controls which webui to server:
 - none: disable webui
 - auto: default webui 
 - llamacpp: llamacpp webui 
 - (default: auto)

--api-key KEY
API key to use for authentication (default: none)

# Split mode graph
-grt or --graph-reduce-type q8_0 | bf16 | f16 | f32
choose the type for as data exchange / graph reduce type for exchanging data between GPUs when using split mode graph.

# sweep_bench
-nrep N | --n-repetitions N
Define th number of repetitions used at zero context.

# llama-bench
-tgb (or --threads-gen-batch)
enable having different number of threads for tg and pp

# Imatrix
--layer-similarity or -lsim
collect statistics about the activations change caused by a layer using cosine similarity 

--hide-imatrix
This will store "top_secret" in the imatrix data file name and calibration dataset fields, and zeros in the batch size and number of chunks used to compute the imatrix.

# Quantization
--custom-q
Custom quantization rules with regular expressions
Example:
llama-quantize --imatrix some_imatrix --custom-q "regex1=typ1,regex2=type2..." some_model some_output_file some_base_quant

# Build arguments
cmake -DGGML_ARCH_FLAGS="-march=armv8.2-a+dotprod+fp16" (plus other things you want to add)

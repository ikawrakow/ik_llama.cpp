### 🔀 [#612](https://github.com/ikawrakow/ik_llama.cpp/pull/612) - kimi-k2 convert script and chat template

| **Author** | `ubergarm` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-07-15 |
| **Updated** | 2025-07-17 |

---

#### Description

1. Add convert script changes from @gabriellarson on mainline PR https://github.com/ggml-org/llama.cpp/pull/14654
2. Add kimi-k2 chat template to support chat endpoint (not sure if this is needed or if the gguf supplies the chat template via jinja or whatnot somehow lol)

Marking this draft for now. I'm about done with testing convert after getting sidetracked with an unrelated technical issue. Then I can roll a Q8_0, do imatrix, and make some small enough quants to test the chat template better.

The workflow for converting Kimi-K2-Instruct is roughly documented here: https://huggingface.co/gabriellarson/Kimi-K2-Instruct-GGUF/discussions/1#68746feb3c3f2a7b1e8541ff

*UPDATE*
My first convert_hf_to_gguf.py just finished and cooking first Q8_0 that seems to have proper tensors to support fast MLA:

```
blk.0.attn_kv_b.weight - [  512, 16384,     1,     1], type =   bf16, converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
```

---

#### 💬 Conversation

👤 **ubergarm** commented the **2025-07-15** at **04:17:47**:<br>

Okay just got the Q8_0 started up and seems coherent in short inferences. Also with this PR it does detect the chat template as such now:
```
INFO [                    main] model loaded | tid="123282723551424" timestamp=1752553001
INFO [                    main] chat template | tid="123282723551424" timestamp=1752553001 chat_example="<|im_system|>system<|im_middle|>You are a helpful assistant<|im_end|><|im_assistant|>assistant<|im_middle|>Hello<|im_end|><|im_user|>user<|im_middle|>Hi there<|im_end|><|im_assistant|>assistant<|im_middle|>How are you?<|im_end|>" built_in=true
```

---

👤 **ikawrakow** submitted a review the **2025-07-15** at **06:01:35**: ✅ `APPROVED`

---

👤 **ubergarm** commented the **2025-07-15** at **16:13:31**:<br>

Thanks!

Continuing testing this morning, rolled first test quant `Kimi-K2-Instruct-IQ2_KL.gguf`. 

Also updated chat template a bit as moonshot seems to have added carriage returns overnight: https://huggingface.co/moonshotai/Kimi-K2-Instruct/blob/main/tokenizer_config.json#L154

```
llm_load_print_meta: model type       = 671B
llm_load_print_meta: model ftype      = IQ2_KL - 2.6875 bpw
llm_load_print_meta: model params     = 1.027 T
llm_load_print_meta: model size       = 345.687 GiB (2.892 BPW)
llm_load_print_meta: repeating layers = 344.166 GiB (2.885 BPW, 1024.571 B parameters)
llm_load_print_meta: general.name     = Kimi K2 Instruct Bf16 Safetensors
```

<details>

<summary>👈 Recipe Details</summary>

```bash
#!/usr/bin/env bash

# Quantizing MLA Notes
# https://github.com/ikawrakow/ik_llama.cpp/issues/601#issuecomment-3070185792

# [0,60] Layers
# First Layer has dense ffn_(gate|up|down)
# Remaining layers have 384x exps and 1x shexp

#           token_embd.weight - [ 7168, 163840,     1,     1], type =   bf16, converting to q8_0 .. size =  2240.00 MiB ->  1190.00 MiB

#       blk.0.ffn_down.weight - [18432,  7168,     1,     1], type =   bf16, converting to q8_0 .. size =   252.00 MiB ->   133.88 MiB
#       blk.0.ffn_gate.weight - [ 7168, 18432,     1,     1], type =   bf16, converting to q8_0 .. size =   252.00 MiB ->   133.88 MiB
#         blk.0.ffn_up.weight - [ 7168, 18432,     1,     1], type =   bf16, converting to q8_0 .. size =   252.00 MiB ->   133.88 MiB
#      blk.0.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
#       blk.0.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
# blk.0.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
#  blk.0.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
#  blk.0.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   bf16, converting to q8_0 .. size =     7.88 MiB ->     4.18 MiB
#      blk.0.attn_kv_b.weight - [  512, 16384,     1,     1], type =   bf16, converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
#       blk.0.attn_k_b.weight - [  128, 32768,     1,     1], type =   bf16, converting to q8_0 .. size =     8.00 MiB ->     4.25 MiB
#       blk.0.attn_v_b.weight - [  512,  8192,     1,     1], type =   bf16, converting to q8_0 .. size =     8.00 MiB ->     4.25 MiB
#    blk.0.attn_output.weight - [ 8192,  7168,     1,     1], type =   bf16, converting to q8_0 .. size =   112.00 MiB ->    59.50 MiB
#       blk.0.attn_q_a.weight - [ 7168,  1536,     1,     1], type =   bf16, converting to q8_0 .. size =    21.00 MiB ->    11.16 MiB
#       blk.0.attn_q_b.weight - [ 1536, 12288,     1,     1], type =   bf16, converting to q8_0 .. size =    36.00 MiB ->    19.12 MiB

#  blk.9.ffn_down_exps.weight - [ 2048,  7168,   384,     1], type =   bf16, converting to q8_0 .. size = 10752.00 MiB ->  5712.00 MiB
#  blk.9.ffn_gate_exps.weight - [ 7168,  2048,   384,     1], type =   bf16, converting to q8_0 .. size = 10752.00 MiB ->  5712.00 MiB
#    blk.9.ffn_up_exps.weight - [ 7168,  2048,   384,     1], type =   bf16, converting to q8_0 .. size = 10752.00 MiB ->  5712.00 MiB
#      blk.9.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
#      blk.9.exp_probs_b.bias - [  384,     1,     1,     1], type =    f32, size =    0.001 MB
#   blk.9.ffn_gate_inp.weight - [ 7168,   384,     1,     1], type =    f32, size =   10.500 MB
#       blk.9.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
#  blk.9.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
# blk.9.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
# blk.9.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =   bf16, converting to q8_0 .. size =    28.00 MiB ->    14.88 MiB
# blk.9.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =   bf16, converting to q8_0 .. size =    28.00 MiB ->    14.88 MiB
#   blk.9.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =   bf16, converting to q8_0 .. size =    28.00 MiB ->    14.88 MiB
#  blk.9.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   bf16, converting to q8_0 .. size =     7.88 MiB ->     4.18 MiB
#      blk.9.attn_kv_b.weight - [  512, 16384,     1,     1], type =   bf16, converting to q8_0 .. size =    16.00 MiB ->     8.50 MiB
#       blk.9.attn_k_b.weight - [  128, 32768,     1,     1], type =   bf16, converting to q8_0 .. size =     8.00 MiB ->     4.25 MiB
#       blk.9.attn_v_b.weight - [  512,  8192,     1,     1], type =   bf16, converting to q8_0 .. size =     8.00 MiB ->     4.25 MiB
#    blk.9.attn_output.weight - [ 8192,  7168,     1,     1], type =   bf16, converting to q8_0 .. size =   112.00 MiB ->    59.50 MiB
#       blk.9.attn_q_a.weight - [ 7168,  1536,     1,     1], type =   bf16, converting to q8_0 .. size =    21.00 MiB ->    11.16 MiB
#       blk.9.attn_q_b.weight - [ 1536, 12288,     1,     1], type =   bf16, converting to q8_0 .. size =    36.00 MiB ->    19.12 MiB

#               output.weight - [ 7168, 163840,     1,     1], type =   bf16, converting to q8_0 .. size =  2240.00 MiB ->  1190.00 MiB

#!/usr/bin/env bash

custom="
## Attention [0-60] (GPU)
# Only ik's fork uses this, keep it q8_0 as its only for PP with -mla 3
blk\..*\.attn_kv_b\.weight=q8_0

# ideally k_b and v_b are smaller than q8_0 as they are is used for TG with -mla 3 (and ik's imatrix supports it)
# blk.*.attn_k_b.weight is not divisible by 256 so only supports qN_0 or iq4_nl
blk\..*\.attn_k_b\.weight=q5_0

# Balance of attn tensors
blk\..*\.attn_.*=iq5_ks

## First Single Dense Layer [0] (GPU)
blk\..*\.ffn_down\.weight=iq5_ks
blk\..*\.ffn_(gate|up)\.weight=iq4_ks

## Shared Expert (1-60) (GPU)
blk\..*\.ffn_down_shexp\.weight=iq5_ks
blk\..*\.ffn_(gate|up)_shexp\.weight=iq4_ks

## Routed Experts (1-60) (CPU)
blk\..*\.ffn_down_exps\.weight=iq3_ks
blk\..*\.ffn_(gate|up)_exps\.weight=iq2_kl

## Token embedding and output tensors (GPU)
token_embd\.weight=iq4_k
output\.weight=iq6_k
"

custom=$(
  echo "$custom" | grep -v '^#' | \
  sed -Ez 's:\n+:,:g;s:,$::;s:^,::'
)

numactl -N 1 -m 1 \
./build/bin/llama-quantize \
    --custom-q "$custom" \
    --imatrix /mnt/raid/models/ubergarm/Kimi-K2-Instruct-GGUF/imatrix-Kimi-K2-Instruct-Q8_0.dat \
    /mnt/raid/models/ubergarm/Kimi-K2-Instruct-GGUF/Kimi-K2-384x15B-Instruct-safetensors-BF16-00001-of-00045.gguf \
    /mnt/raid/models/ubergarm/Kimi-K2-Instruct-GGUF/Kimi-K2-Instruct-IQ2_KL.gguf \
    IQ2_KL \
    192
```

</details>

Currently testing perplexity to make sure it runs clean.

Also working with the AIBeaverClub folks to test the API endpoint, and having some kind of issue. The model will reply okay sometimes, but other times it takes a little time and returns empty response and the server logs have really high TG when it happens:

```
INFO [           print_timings] prompt eval time     =     115.07 ms /     1 tokens (  115.07 ms per token,     8.69 tokens per second) | tid="134826401728704" timestamp=1752595857 id_slot=0 id_task=1550 t_prompt_processing=115.067 n_prompt_tokens_processed=1 t_token=115.067 n_tokens_second=8.690588961213901
INFO [           print_timings] generation eval time =       0.02 ms /     1 runs   (    0.02 ms per token, 45454.55 tokens per second) | tid="134826401728704" timestamp=1752595857 id_slot=0 id_task=1550 t_token_generation=0.022 n_decoded=1 t_token=0.022 n_tokens_second=45454.545454545456
```

But then other times it does respond okay, well formatted, coherent... 

So hoping maybe just the chat template is off and will hack on it some more before marking ready.

@anikifoss 

No pressure, but happy to hear if you manage to use this convert script on the original fp8 safetensors to get your good MLA bf16 GGUFs (with the attn_kv_b tensor).

---

👤 **anikifoss** commented the **2025-07-15** at **17:01:52**:<br>

@ubergarm I can test the `convert_hf_to_gguf.py` from this PR to convert unsloth's BF16 `safetensors` to GGUF.

---

👤 **ubergarm** commented the **2025-07-15** at **17:07:54**:<br>

> @ubergarm I can test the `convert_hf_to_gguf.py` from this PR to convert unsloth's BF16 `safetensors` to GGUF.

Oh I didn't realize they uploaded the bf16 safetensors that must be just the output of fp8_cast_bf16.py yes that should work as that step does not strip the `attn_kv_b` so should work out! Thanks for testing, I know this thing is a monster. Working with this 1TB+ model feels like driving a barge lol...

So far so good, the updated chat template `add_ass` fixed the generation issue. So as soon as my perplexity comes back clean I'll start uploading and be ready to merge this.

---

👤 **ikawrakow** commented the **2025-07-15** at **17:13:34**:<br>

> So as soon as my perplexity comes back clean I'll start uploading and be ready to merge this.

How quickly, or rather how slowly, does it go?

---

👤 **ikawrakow** commented the **2025-07-15** at **17:19:00**:<br>

Btw., I have decided to add a sub-2 bpw quant, `IQ1_KT`, at 1.75 bpw (so same as `IQ1_M`). It is Trellis, but my guess is that with Kimi-2 even more people will reach to the lowest possible bpw models. Desperate times call for desperate action! It is shaping up to be nearly on par with `IQ2_XXS` (2.0625 bpw), and certainly much better than `IQ1_M`. CUDA is done with very decent performance. I'll do the CPU tomorrow.

---

👤 **ubergarm** commented the **2025-07-15** at **17:46:20**:<br>

Okay perplexity ran clean on CPU only implementation:

```
model=/mnt/raid/hf/Kimi-K2-Instruct-GGUF/IQ2_KL/Kimi-K2-Instruct-IQ2_KL-00001-of-00008.gguf
numactl -N 1 -m 1 \
./build/bin/llama-perplexity \
    -m "$model" \
    -f wiki.test.raw \
    --seed 1337 \
    -fa -fmoe \
    -mla 3 \
    --ctx-size 512 \
    --numa numactl \
    --threads 192

Final estimate: PPL = 3.2741 +/- 0.01689
```

Happy to merge this now and model will land in hugging face in 10 minutes.

---

👤 **anikifoss** commented the **2025-07-15** at **19:10:17**:<br>

> Oh I didn't realize they uploaded the bf16 safetensors that must be just the output of fp8_cast_bf16.py yes that should work as that step does not strip the attn_kv_b so should work out! Thanks for testing, I know this thing is a monster. Working with this 1TB+ model feels like driving a barge lol...

@ubergarm I don't see the `attn_kv_b` in GGUFs created from the unloth's BF16 safetensors, so I assume it's already removed. Do you still want me to test the conversion, or start over from the FP8 safetensors (will likely take me a couple of days to set up triton and run the intermediate conversion step)

---

👤 **saood06** commented the **2025-07-16** at **00:29:19**:<br>

> TODO: find a safetensor viewer...

HF has one built in just like for GGUF.

---

👤 **ubergarm** commented the **2025-07-16** at **02:53:08**:<br>

@ikawrakow 

> How quickly, or rather how slowly, does it go?

I finally got to some sweep benches feeling out this big dual socket AMD EPYC 9965 192-Core rig in NPS1 with ~768GB RAM per socket. mlc clocks it at around 256GiB/s RAM bandwidth per socket. The "smaller" Kimi-K2-Instruct quants will fit on a single socket. Given I believe this is Zen5 I tried out #610  and did see around 8% boost in PP with that AVX512 kernel. Also increasing `-ub 4096 -b 4096` and omitting `-rtr` a valid option even on this MoE.

<img width="4176" height="2217" alt="kimi-k2-instruct-amdvolcano-iq2_kl" src="https://github.com/user-attachments/assets/9512e535-046f-496c-9b3b-e65074d90f5a" />

<details>

<summary>👈 Command and Data</summary>

```bash
# IQ2_KL 345.687 GiB (2.892 BPW)
model=/mnt/raid/hf/Kimi-K2-Instruct-GGUF/IQ2_KL/Kimi-K2-Instruct-IQ2_KL-00001-of-00008.gguf
numactl -N 0 -m 0 \
./build/bin/llama-sweep-bench \
    --model "$model"\
    --ctx-size 12288 \
    -ctk q8_0 \
    -fa -fmoe \
    -mla 3 \
    --threads 128 \
    --threads-batch 192 \
    -ub 4096 -b 4096 \
    --no-mmap \
    --numa numactl \
    --warmup-batch
```
# IQ2_KL --no-mmap -ub 512 -b 2048
|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|   512 |    128 |      0 |    4.757 |   107.63 |    9.559 |    13.39 |
|   512 |    128 |    512 |    2.947 |   173.75 |    9.396 |    13.62 |
|   512 |    128 |   1024 |    4.313 |   118.71 |    9.448 |    13.55 |
|   512 |    128 |   1536 |    3.477 |   147.27 |    9.589 |    13.35 |
|   512 |    128 |   2048 |    3.495 |   146.49 |    9.726 |    13.16 |
|   512 |    128 |   2560 |    3.666 |   139.66 |    9.777 |    13.09 |
|   512 |    128 |   3072 |    3.568 |   143.51 |    9.899 |    12.93 |
|   512 |    128 |   3584 |    3.590 |   142.61 |    9.998 |    12.80 |
|   512 |    128 |   4096 |    4.052 |   126.34 |   10.100 |    12.67 |
|   512 |    128 |   4608 |    4.661 |   109.85 |   10.212 |    12.53 |
|   512 |    128 |   5120 |    4.912 |   104.23 |   10.200 |    12.55 |
|   512 |    128 |   5632 |    5.023 |   101.94 |   10.319 |    12.40 |
|   512 |    128 |   6144 |    4.372 |   117.10 |   10.387 |    12.32 |
|   512 |    128 |   6656 |    4.393 |   116.55 |   10.526 |    12.16 |
|   512 |    128 |   7168 |    4.757 |   107.64 |   10.537 |    12.15 |
|   512 |    128 |   7680 |    4.561 |   112.27 |   10.516 |    12.17 |
|   512 |    128 |   8192 |    4.554 |   112.43 |   10.611 |    12.06 |
|   512 |    128 |   8704 |    4.806 |   106.54 |   10.575 |    12.10 |
|   512 |    128 |   9216 |    4.494 |   113.93 |   10.754 |    11.90 |

# IQ2_KL -rtr -ub 512 -b 2048 -rtr
|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|   512 |    128 |      0 |    3.185 |   160.74 |    9.178 |    13.95 |
|   512 |    128 |    512 |    3.397 |   150.71 |    9.229 |    13.87 |
|   512 |    128 |   1024 |    3.479 |   147.17 |    9.399 |    13.62 |
|   512 |    128 |   1536 |    3.392 |   150.96 |    9.353 |    13.69 |
|   512 |    128 |   2048 |    3.946 |   129.75 |    9.507 |    13.46 |
|   512 |    128 |   2560 |    3.952 |   129.55 |    9.600 |    13.33 |
|   512 |    128 |   3072 |    3.639 |   140.69 |    9.705 |    13.19 |
|   512 |    128 |   3584 |    3.766 |   135.95 |    9.689 |    13.21 |
|   512 |    128 |   4096 |    3.835 |   133.49 |    9.840 |    13.01 |
|   512 |    128 |   4608 |    4.312 |   118.74 |    9.814 |    13.04 |
|   512 |    128 |   5120 |    4.104 |   124.76 |   10.159 |    12.60 |
|   512 |    128 |   5632 |    4.257 |   120.27 |   10.044 |    12.74 |
|   512 |    128 |   6144 |    4.343 |   117.89 |   10.312 |    12.41 |
|   512 |    128 |   6656 |    4.435 |   115.46 |   10.186 |    12.57 |
|   512 |    128 |   7168 |    4.783 |   107.06 |   10.240 |    12.50 |
|   512 |    128 |   7680 |    4.670 |   109.63 |   10.351 |    12.37 |
|   512 |    128 |   8192 |    4.627 |   110.66 |   10.374 |    12.34 |

# IQ2_KL --no-mmap -ub 4096 -b 4096
|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|  4096 |   1024 |      0 |   17.240 |   237.58 |   78.567 |    13.03 |
|  4096 |   1024 |   4096 |   20.060 |   204.19 |   81.596 |    12.55 |
|  4096 |   1024 |   8192 |   22.211 |   184.42 |   84.820 |    12.07 |

# IQ2_KL -rtr -ub 4096 -b 4096
|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|  4096 |   1024 |      0 |   20.563 |   199.19 |   78.669 |    13.02 |
|  4096 |   1024 |   4096 |   21.216 |   193.06 |   83.873 |    12.21 |
|  4096 |   1024 |   8192 |   24.440 |   167.60 |   87.510 |    11.70 |

# IQ2_KL PR610 ik/q8_k_r8_avx512 --no-mmap -ub 4096 -b 4096
|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|  4096 |   1024 |      0 |   15.844 |   258.53 |   79.230 |    12.92 |
|  4096 |   1024 |   4096 |   17.343 |   236.18 |   83.245 |    12.30 |
|  4096 |   1024 |   8192 |   21.132 |   193.83 |   86.125 |    11.89 |

</details>

---

👤 **ubergarm** commented the **2025-07-16** at **03:39:04**:<br>

> Btw., I have decided to add a sub-2 bpw quant, IQ1_KT, at 1.75 bpw (so same as IQ1_M). It is Trellis, but my guess is that with Kimi-2 even more people will reach to the lowest possible bpw models. Desperate times call for desperate action! It is shaping up to be nearly on par with IQ2_XXS (2.0625 bpw), and certainly much better than IQ1_M. CUDA is done with very decent performance. I'll do the CPU tomorrow.

I had a few hours on a dual RTX 6000 Pro (Max-Q 300W version maybe as each GPU was under 300W despite 600W cap shown in `nvidia-smi`) with 198GB VRAM total and that DeepSeek-TNG-R1T2-Chimera-IQ2_KT fully offloaded with `-ub 4096 -b 4096` with over 40k context available at f16 or more at q8_0

<img width="4176" height="2161" alt="dual-6000-take-3" src="https://github.com/user-attachments/assets/6f008912-a5ec-4049-94b4-d10f89e22de3" />

Curious to see how the IQ1_KT comes along as competition for the IQ1_S and IQ1_M is indeed welcome with these ridiculous 1TB models!

---

👤 **ubergarm** commented the **2025-07-16** at **12:48:15**:<br>

Thanks @anikifoss I opened a PR here https://github.com/ikawrakow/ik_llama.cpp/pull/617 with the fixup, let us know how it looks in the morning!

---

👤 **anikifoss** commented the **2025-07-16** at **23:58:27**:<br>

Done:
```
Writing: 100%|███████████████████████████████████████████████████████████████████████████████████| 2.05T/2.05T [16:59:41<00:00, 33.6Mbyte/s]
```

HDDs are not fast :roll_eyes:

---

👤 **anikifoss** commented the **2025-07-17** at **17:32:59**:<br>

@ubergarm quantized to Q4 for down_exp nd Q3 for the other exps. It runs, was able to produce spinning hexagon with 3 tries (Q4/Q3 mix is just under 512GB, but noticably worse than Q6/Q4).
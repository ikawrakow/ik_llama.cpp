## 🔀 [Pull Request #612](https://github.com/ikawrakow/ik_llama.cpp/pull/612) - kimi-k2 convert script and chat template

| **Author** | `ubergarm` |
| :--- | :--- |
| **State** | 🔀 **Merged** |
| **Source Branch** | `ug/convert-kimi-k2` |
| **Target Branch** | `main` |
| **Created** | 2025-07-15 |
| **Updated** | 2025-07-17 |
| **Merged** | 2025-07-15 |

---

## 📄 Description

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

## 💬 Conversation

👤 **ubergarm** commented on **2025-07-15** at **04:17:47**

Okay just got the Q8_0 started up and seems coherent in short inferences. Also with this PR it does detect the chat template as such now:
```
INFO [                    main] model loaded | tid="123282723551424" timestamp=1752553001
INFO [                    main] chat template | tid="123282723551424" timestamp=1752553001 chat_example="<|im_system|>system<|im_middle|>You are a helpful assistant<|im_end|><|im_assistant|>assistant<|im_middle|>Hello<|im_end|><|im_user|>user<|im_middle|>Hi there<|im_end|><|im_assistant|>assistant<|im_middle|>How are you?<|im_end|>" built_in=true
```

Gonna let this imatrix run and get some sleep. I added specifically `-mla 1` based on [this discussion](https://github.com/ikawrakow/ik_llama.cpp/issues/601#issuecomment-3070185792). Historically I leave off `-fa` as well during imatrix but not sure best practice or if it matters much.
```
model=/mnt/raid/models/ubergarm/Kimi-K2-Instruct-GGUF/Kimi-K2-Instruct-Q8_0.gguf
numactl --interleave=all \
./build/bin/llama-imatrix \
    -m "$model" \
    -f ubergarm-imatrix-calibration-corpus-v02.txt \
    -o /mnt/raid/models/ubergarm/Kimi-K2-Instruct-GGUF/imatrix-Kimi-K2-Instruct-Q8_0.dat \
    -mla 1 \
    --verbosity 1 \
    --ctx-size 512 \
    --layer-similarity \
    --numa distribute \
    --threads 384
```

Thanks!

---

👤 **ikawrakow** approved this pull request ✅ on **2025-07-15** at **06:01:35**

---

👤 **ubergarm** commented on **2025-07-15** at **16:13:31**

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

llama_model_loader: - type  f32:  365 tensors                                                                                       11:59:08 [72/1848]
llama_model_loader: - type q5_0:   61 tensors
llama_model_loader: - type q8_0:   61 tensors
llama_model_loader: - type iq4_k:    1 tensors
llama_model_loader: - type iq6_k:    1 tensors
llama_model_loader: - type iq4_ks:  122 tensors
llama_model_loader: - type iq5_ks:  366 tensors
llama_model_loader: - type iq3_ks:   60 tensors
llama_model_loader: - type iq2_kl:  120 tensors
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

*EDIT*
Maybe I just need to `add_ass` to get it to reply:

```bash
$ python chat_template_tester.py moonshotai/Kimi-K2-Instruct
>> chat template <<
<|im_system|>system<|im_middle|>example system prompt<|im_end|><|im_user|>user<|im_middle|>example user turn 1<|im_end|><|im_assistant|>assistant<|im_middle|>example assistant turn 1<|im_end|><|im_user|>user<|im_middle|>example user turn 2<|im_end|><|im_assistant|>assistant<|im_middle|>
>> end of chat template <<
```


@anikifoss 

No pressure, but happy to hear if you manage to use this convert script on the original fp8 safetensors to get your good MLA bf16 GGUFs (with the attn_kv_b tensor).

---

👤 **anikifoss** commented on **2025-07-15** at **17:01:52**

@ubergarm I can test the `convert_hf_to_gguf.py` from this PR to convert unsloth's BF16 `safetensors` to GGUF.

---

👤 **ubergarm** commented on **2025-07-15** at **17:07:54**

> @ubergarm I can test the `convert_hf_to_gguf.py` from this PR to convert unsloth's BF16 `safetensors` to GGUF.

Oh I didn't realize they uploaded the bf16 safetensors that must be just the output of fp8_cast_bf16.py yes that should work as that step does not strip the `attn_kv_b` so should work out! Thanks for testing, I know this thing is a monster. Working with this 1TB+ model feels like driving a barge lol...

So far so good, the updated chat template `add_ass` fixed the generation issue. So as soon as my perplexity comes back clean I'll start uploading and be ready to merge this.

---

👤 **ikawrakow** commented on **2025-07-15** at **17:13:34**

> So as soon as my perplexity comes back clean I'll start uploading and be ready to merge this.

How quickly, or rather how slowly, does it go?

---

👤 **ikawrakow** commented on **2025-07-15** at **17:19:00**

Btw., I have decided to add a sub-2 bpw quant, `IQ1_KT`, at 1.75 bpw (so same as `IQ1_M`). It is Trellis, but my guess is that with Kimi-2 even more people will reach to the lowest possible bpw models. Desperate times call for desperate action! It is shaping up to be nearly on par with `IQ2_XXS` (2.0625 bpw), and certainly much better than `IQ1_M`. CUDA is done with very decent performance. I'll do the CPU tomorrow.

---

👤 **ubergarm** commented on **2025-07-15** at **17:27:41**

> How quickly, or rather how slowly, does it go?

I hope to get some sweep-benches in eventually, anecdotally on short prompts with llama-server seeing around 130\~150 tok/sec PP and 10\~12 tok/sec TG running CPU-only on a single socket of a `AMD EPYC 9965 192-Core Processor` with 768GB DDR5@6400MT/s clocked around 260GiB/sec RAM bandwidth *per socket*

Running like so on a single socket. I haven't found the sweet spot for threads given this rig is new to me.

```bash
model=/mnt/raid/hf/Kimi-K2-Instruct-GGUF/IQ2_KL/Kimi-K2-Instruct-IQ2_KL-00001-of-00008.gguf
numactl -N 0 -m 0 \
./build/bin/llama-server \
    --model "$model"\
    --alias ubergarm/Kimi-K2-Instruct \
    --ctx-size 32768 \
    -ctk q8_0 \
    -fa -fmoe \
    -mla 3 \
    --parallel 1 \
    --threads 64 \
    --threads-batch 192 \
    --numa numactl \
    --host 127.0.0.1 \
    --port 8080
```

---

👤 **ubergarm** commented on **2025-07-15** at **17:46:20**

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

👤 **anikifoss** commented on **2025-07-15** at **19:10:17**

> Oh I didn't realize they uploaded the bf16 safetensors that must be just the output of fp8_cast_bf16.py yes that should work as that step does not strip the attn_kv_b so should work out! Thanks for testing, I know this thing is a monster. Working with this 1TB+ model feels like driving a barge lol...

@ubergarm I don't see the `attn_kv_b` in GGUFs created from the unloth's BF16 safetensors, so I assume it's already removed. Do you still want me to test the conversion, or start over from the FP8 safetensors (will likely take me a couple of days to set up triton and run the intermediate conversion step)

---

👤 **ubergarm** commented on **2025-07-15** at **19:22:13**

@anikifoss 

Thanks for giving it a try, at least it sounds like this `convert_hf_to_gguf.py` "worked" on the unsloth BF16 safetensors? Hrmm... I think there is a safetensor viewer script let me see... In the mean time I'll check with my remote rig guy and ask him if its cool to just upload the bf16 GGUFs to make it easier for ya! Will update soon.

<details>

<summary>👈 gguf dump of my bf16 GGUFs</summary>

```bash
$ python ./gguf-py/scripts/gguf_dump.py /mnt/raid/models/ubergarm/Kimi-K2-Instruct-GGUF/Kimi-K2-384x15B-Instruct-safetensors-BF16-00001-of-00045.gguf

INFO:gguf-dump:* Loading: /mnt/raid/models/ubergarm/Kimi-K2-Instruct-GGUF/Kimi-K2-384x15B-Instruct-safetensors-BF16-00001-of-00045.gguf
* File is LITTLE endian, script is running on a LITTLE endian host.
* Dumping 48 key/value pair(s)
      1: UINT32     |        1 | GGUF.version = 3
      2: UINT64     |        1 | GGUF.tensor_count = 36
      3: UINT64     |        1 | GGUF.kv_count = 45
      4: STRING     |        1 | general.architecture = 'deepseek2'
      5: STRING     |        1 | general.type = 'model'
      6: STRING     |        1 | general.name = 'Kimi K2 Instruct Bf16 Safetensors'
      7: STRING     |        1 | general.finetune = 'Instruct-safetensors'
      8: STRING     |        1 | general.basename = 'Kimi-K2'
      9: STRING     |        1 | general.size_label = '384x15B'
.
.
.

* Dumping 36 tensor(s)
      1: 1174405120 |  7168, 163840,     1,     1 | BF16    | token_embd.weight
      2:       7168 |  7168,     1,     1,     1 | F32     | blk.0.attn_norm.weight
      3:  132120576 | 18432,  7168,     1,     1 | BF16    | blk.0.ffn_down.weight
      4:  132120576 |  7168, 18432,     1,     1 | BF16    | blk.0.ffn_gate.weight
      5:  132120576 |  7168, 18432,     1,     1 | BF16    | blk.0.ffn_up.weight
      6:       7168 |  7168,     1,     1,     1 | F32     | blk.0.ffn_norm.weight
      7:        512 |   512,     1,     1,     1 | F32     | blk.0.attn_kv_a_norm.weight
      8:    4128768 |  7168,   576,     1,     1 | BF16    | blk.0.attn_kv_a_mqa.weight
      9:    8388608 |   512, 16384,     1,     1 | BF16    | blk.0.attn_kv_b.weight
     10:    4194304 |   128, 32768,     1,     1 | BF16    | blk.0.attn_k_b.weight
     11:    4194304 |   512,  8192,     1,     1 | BF16    | blk.0.attn_v_b.weight
     12:   58720256 |  8192,  7168,     1,     1 | BF16    | blk.0.attn_output.weight
     13:       1536 |  1536,     1,     1,     1 | F32     | blk.0.attn_q_a_norm.weight
     14:   11010048 |  7168,  1536,     1,     1 | BF16    | blk.0.attn_q_a.weight
     15:   18874368 |  1536, 12288,     1,     1 | BF16    | blk.0.attn_q_b.weight
     16:       7168 |  7168,     1,     1,     1 | F32     | blk.9.attn_norm.weight
     17: 5637144576 |  2048,  7168,   384,     1 | BF16    | blk.9.ffn_down_exps.weight
     18: 5637144576 |  7168,  2048,   384,     1 | BF16    | blk.9.ffn_gate_exps.weight
     19: 5637144576 |  7168,  2048,   384,     1 | BF16    | blk.9.ffn_up_exps.weight
     20:        384 |   384,     1,     1,     1 | F32     | blk.9.exp_probs_b.bias
     21:    2752512 |  7168,   384,     1,     1 | F32     | blk.9.ffn_gate_inp.weight
     22:   14680064 |  2048,  7168,     1,     1 | BF16    | blk.9.ffn_down_shexp.weight
     23:   14680064 |  7168,  2048,     1,     1 | BF16    | blk.9.ffn_gate_shexp.weight
     24:   14680064 |  7168,  2048,     1,     1 | BF16    | blk.9.ffn_up_shexp.weight
     25:       7168 |  7168,     1,     1,     1 | F32     | blk.9.ffn_norm.weight
     26:        512 |   512,     1,     1,     1 | F32     | blk.9.attn_kv_a_norm.weight
     27:    4128768 |  7168,   576,     1,     1 | BF16    | blk.9.attn_kv_a_mqa.weight
     28:    8388608 |   512, 16384,     1,     1 | BF16    | blk.9.attn_kv_b.weight
     29:    4194304 |   128, 32768,     1,     1 | BF16    | blk.9.attn_k_b.weight
     30:    4194304 |   512,  8192,     1,     1 | BF16    | blk.9.attn_v_b.weight
     31:   58720256 |  8192,  7168,     1,     1 | BF16    | blk.9.attn_output.weight
     32:       1536 |  1536,     1,     1,     1 | F32     | blk.9.attn_q_a_norm.weight
     33:   11010048 |  7168,  1536,     1,     1 | BF16    | blk.9.attn_q_a.weight
     34:   18874368 |  1536, 12288,     1,     1 | BF16    | blk.9.attn_q_b.weight
.
.
.
```

</details>

TODO: find a safetensor viewer...

---

👤 **anikifoss** commented on **2025-07-15** at **19:25:08**

> Thanks for giving it a try, at least it sounds like this convert_hf_to_gguf.py "worked" on the unsloth BF16 safetensors?

@ubergarm I haven't run it on your branch. What I'm saying is [this quant](https://huggingface.co/anikifoss/Kimi-K2-Instruct-DQ4_K), created from unsloth's BF16 safetensors and converted to GGUF using llama.cpp does not have `attn_kv_b`. So, most likely, unsloth's BF16 safetensors does not have `attn_kv_b`.

---

👤 **ubergarm** commented on **2025-07-15** at **19:33:06**

@anikifoss 

> converted to GGUF using llama.cpp

:point_up: that is the step which I believe munges up and omits the `attn_kv_b`.

If you use the freshly merged ik_llama.cpp/convert_hf_to_gguf.py on those bf16 safetensors, I believe you will get the attn_kv_b tensors in your bf16 GGUF.

afaik going from fp8 safetensors upcasting via fp8_cast_bf16.py bf16 safetensors does *not* mess with the actual tensors.

> So, most likely, unsloth's BF16 safetensors does not have attn_kv_b.

Unless they did something strange, I believe they should be okay to use with this new convert script.

Probably easy enough to test if you have the disk space as nothing more required to download.

*EDIT*

I believe this is the code that is munging it up in [mainline convert_hf_to_gguf.py](https://github.com/ggml-org/llama.cpp/blob/master/convert_hf_to_gguf.py#L5832-L5852).

---

👤 **saood06** commented on **2025-07-16** at **00:29:19**

> TODO: find a safetensor viewer...

HF has one built in just like for GGUF.

---

👤 **ubergarm** commented on **2025-07-16** at **02:53:08**

@ikawrakow 

> How quickly, or rather how slowly, does it go?

I finally got to some sweep benches feeling out this big dual socket AMD EPYC 9965 192-Core rig in NPS1 with ~768GB RAM per socket. mlc clocks it at around 256GiB/s RAM bandwidth per socket. The "smaller" Kimi-K2-Instruct quants will fit on a single socket. Given I believe this is Zen5 I tried out [#610](https://github.com/ikawrakow/ik_llama.cpp/issues/610)  and did see around 8% boost in PP with that AVX512 kernel. Also increasing `-ub 4096 -b 4096` and omitting `-rtr` a valid option even on this MoE.

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

# IQ4_KS PR610 ik/q8_k_r8_avx512 --no-mmap -ub 4096 -b 4096
|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|  4096 |   1024 |      0 |   16.169 |   253.33 |   89.668 |    11.42 |
|  4096 |   1024 |   4096 |   18.017 |   227.34 |   96.703 |    10.59 |
|  4096 |   1024 |   8192 |   20.752 |   197.38 |   99.845 |    10.26 |

</details>

I compared the larger IQ4_KS 550.428 GiB (4.604 BPW)  and its remarkably similar performance.

<img width="4176" height="2161" alt="kimi-k2-instruct-amdvolcano-iq2_kl-vs-iq4_ks" src="https://github.com/user-attachments/assets/a3c44574-8649-4564-85c6-3550f4da7890" />

---

👤 **ubergarm** commented on **2025-07-16** at **03:39:04**

> Btw., I have decided to add a sub-2 bpw quant, IQ1_KT, at 1.75 bpw (so same as IQ1_M). It is Trellis, but my guess is that with Kimi-2 even more people will reach to the lowest possible bpw models. Desperate times call for desperate action! It is shaping up to be nearly on par with IQ2_XXS (2.0625 bpw), and certainly much better than IQ1_M. CUDA is done with very decent performance. I'll do the CPU tomorrow.

I had a few hours on a dual RTX 6000 Pro (Max-Q 300W version maybe as each GPU was under 300W despite 600W cap shown in `nvidia-smi`) with 198GB VRAM total and that DeepSeek-TNG-R1T2-Chimera-IQ2_KT fully offloaded with `-ub 4096 -b 4096` with over 40k context available at f16 or more at q8_0

<img width="4176" height="2161" alt="dual-6000-take-3" src="https://github.com/user-attachments/assets/6f008912-a5ec-4049-94b4-d10f89e22de3" />

Curious to see how the IQ1_KT comes along as competition for the IQ1_S and IQ1_M is indeed welcome with these ridiculous 1TB models!

---

👤 **anikifoss** commented on **2025-07-16** at **04:25:58**

@ubergarm looks like you're missing an indent:
```
python convert_hf_to_gguf.py \
    --outtype bf16 \
    --split-max-size 50G \
    /mnt/data/Models/unsloth/Kimi-K2-Instruct-BF16
  File "convert_hf_to_gguf.py", line 3439
    self._set_vocab_gpt2()
    ^
IndentationError: expected an indented block after 'else' statement on line 3438
```

I fixed it locally, so it can run overnight.

---

👤 **ubergarm** commented on **2025-07-16** at **12:48:15**

Thanks @anikifoss I opened a PR here https://github.com/ikawrakow/ik_llama.cpp/pull/617 with the fixup, let us know how it looks in the morning!

---

👤 **anikifoss** commented on **2025-07-16** at **23:58:27**

Done:
```
Writing: 100%|███████████████████████████████████████████████████████████████████████████████████| 2.05T/2.05T [16:59:41<00:00, 33.6Mbyte/s]
```

HDDs are not fast :roll_eyes:

---

👤 **anikifoss** commented on **2025-07-16** at **23:59:05**

I'll quantize overnight and will let you know how it works tomorrow.

---

👤 **anikifoss** commented on **2025-07-17** at **17:32:59**

@ubergarm quantized converted GGUF to Q4_K for down_exps and Q3_K for the other exps. It runs, was able to produce spinning hexagon with 3 tries (Q4/Q3 mix is just under 512GB, but noticably worse than Q6/Q4).
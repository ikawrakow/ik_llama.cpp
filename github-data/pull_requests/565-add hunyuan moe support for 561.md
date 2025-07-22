### 🔀 [#565](https://github.com/ikawrakow/ik_llama.cpp/pull/565) - add hunyuan moe support for 561

| **Author** | `ubergarm` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-06-30 |
| **Updated** | 2025-07-15 |

---

#### Description

Based this PR on mainline https://github.com/ggml-org/llama.cpp/pull/14425. Didn't merge any python stuff (used mainline convert script). Tested with bf16 on hybrid CUDA+CPU.

```bash
model=/mnt/raid/models/ubergarm/Hunyuan-A13B-Instruct-GGUF/Hunyuan-A13B-Instruct-BF16-00001-of-00004.gguf
./build/bin/llama-server \
  --model "$model" \
  --alias ubergarm/Hunyuan-A13B-Instruct-bf16 \
  -fa \
  -ctk q8_0 -ctv q8_0 \
  -c 8192 \
  --temp 0.6 \
  --presence-penalty 0.7 \
  --min-p 0.1 \
  -ts 48,48 \
  -ngl 16 \
  --threads 24 \
  --host 127.0.0.1 \
  --port 8080
```

Would be great if anyone else could test e.g. @Downtown-Case as per #561 

I haven't yet made imatrix nor tried to quantize further.

Might be able to use one of the following if was converted recently enough:
* https://huggingface.co/bullerwins/Hunyuan-A13B-Instruct-GGUF
* https://huggingface.co/qwp4w3hyb/Hunyuan-A13B-Instruct-hf-WIP-GGUF

The behavior seems a bit odd and will answer in chinese if I don't use some kind of system prompt or explicitly say speak in english. Mainline seems to use some kind of `--jinja` thing which isn't supported here psure. So ymmv.

---

#### 💬 Conversation

👤 **ubergarm** commented the **2025-06-30** at **18:28:48**:<br>

I'm currently processing an imatrix and noticed that it *requires* `-fa` or will have very large numbers.

This seems to be working so far, though still seems a higher than I expected which could be indicative of an problem:

```bash
./build/bin/llama-imatrix \
    --verbosity 1 \
    --layer-similarity \
    -m /mnt/raid/models/ubergarm/Hunyuan-A13B-Instruct-GGUF/Hunyuan-A13B-Instruct-BF16-00001-of-00004.gguf \
    -f ubergarm-imatrix-calibration-corpus-v02.txt \
    -o /mnt/raid/models/ubergarm/Hunyuan-A13B-Instruct-GGUF/imatrix-Hunyuan-A13B-Instruct-BF16.dat \
    -fa \
    --ctx-size 512 \
    -ts 48,48 \
    -ngl 18 \
    --threads 24

system_info: n_threads = 24 / 48 | AVX = 1 | AVX_VNNI = 0 | AVX2 = 1 | AVX512 = 1 | AVX512_VBMI = 1 | AVX512_VNNI = 1 | AVX512_BF16 = 1 | FMA = 1 | NEON = 0 | SVE = 0 | ARM_FMA = 0 | F16C = 1 | FP16_VA = 0 | WASM_SIMD = 0 | BLAS = 1 | SSE3 = 1 | SSSE3 = 1 | VSX = 0 | MATMUL_INT8 = 0 | LLAMAFILE = 1 |
compute_imatrix: tokenizing the input ..
compute_imatrix: tokenization took 701.577 ms
compute_imatrix: computing over 865 chunks with batch_size 512
compute_imatrix: 5.03 seconds per pass - ETA 1 hours 12.48 minutes
[1]12.7104,[2]14.8010,[3]14.3374,[4]30.5778,[5]17.4738,[6]14.5285,[7]20.2402,[8]14.9318,[9]11.7604,
save_imatrix: stored collected data after 10 chunks in /mnt/raid/models/ubergarm/Hunyuan-A13B-Instruct-GGUF/imatrix-Hunyuan-A13B-Instruct-BF16.dat
[10]12.0205,[11]10.2799,[12]12.3863,[13]14.9808,[14]16.1885,[15]16.6677,[16]20.9547,[17]19.1613,[18]17.4531,[19]15.5200,

...
```

---

👤 **ikawrakow** commented the **2025-06-30** at **20:20:40**:<br>

No FA and FA giving very different PPL values is not a good sign.

PPL of 60 is not a good sign either, especially for a model of that size.

---

👤 **ubergarm** commented the **2025-06-30** at **20:36:19**:<br>

I'm going to leave an endpoint up for a little bit if anyone wants to try the first experimental quant.. No promises lol

## Endpoint
WebUI: https://llm.ubergarm.com/
APIEndpoint: https://llm.ubergarm.com/ (it is llama-server API endpoint with no API key)

There are 8 concurrent slots each with 64k prompt limit.

## Test Quant
I just rolled an imatrix.dat and made my first quant for testing.
```
llm_load_print_meta: model type       = 80B.A13B
llm_load_print_meta: model ftype      = IQ4_K - 4.5 bpw
llm_load_print_meta: model params     = 80.393 B
llm_load_print_meta: model size       = 48.581 GiB (5.191 BPW)
llm_load_print_meta: general.name     = Hunyuan A13B Instruct
```

```
blk\..*\.attn_k.*=iq6_k
blk\..*\.attn_v.*=iq6_k

blk\..*\.attn_q.*=iq5_k
blk\..*\.attn_o.*=iq5_k

# 1x Shared Expert
blk\..*\.ffn_(gate|up)_shexp.*=iq6_k
blk\..*\.ffn_(down)_shexp.*=iq5_k

# 64x Routed Experts
blk\..*\.ffn_(gate|up)_exps.*=iq5_k
blk\..*\.ffn_(down)_exps.*=iq4_k

# Token Embedding
token_embd\.weight=iq4_k
```

How I ran it:
```bash
model=/mnt/raid/models/ubergarm/Hunyuan-A13B-Instruct-GGUF/Hunyuan-A13B-Instruct-IQ4_K.gguf
./build/bin/llama-server \
  --model "$model" \
  --alias ubergarm/Hunyuan-A13B-Instruct-IQ4_K \
  -fa \
  -ctk q8_0 -ctv q8_0 \
  -c 524288 \
  --temp 0.6 \
  --presence-penalty 0.7 \
  --min-p 0.1 \
  -ts 48,48 \
  -ngl 99 \
  --parallel 8 \
  --threads 1 \
  --host 127.0.0.1 \
  --port 8080
```

---

👤 **ikawrakow** submitted a review the **2025-07-01** at **06:00:36**: 💬 `COMMENTED`

---

👤 **ikawrakow** commented during a code review the **2025-07-01** at **06:00:36** on `src/llama.cpp`:<br>

If you check your previous PR about GLM4 you will see that you had to remove the `Vcur` reshaping. It is the same here. Remove this line and it is likely the difference between FA and no FA will go away.

---

👤 **ubergarm** submitted a review the **2025-07-01** at **23:54:30**: 💬 `COMMENTED`

---

👤 **ubergarm** commented the **2025-07-02** at **04:03:30**:<br>

> run on wsl I got a error: Floating point exception (core dumped), in the initial procress of ik_llama.cpp

Its becase I'm a madman and released a quant depending on two unmerged PRs. Check here for instructions how to get the IQ3_KS PR here: https://huggingface.co/ubergarm/Hunyuan-A13B-Instruct-GGUF#note-building-experimental-prs

---

👤 **ubergarm** commented the **2025-07-02** at **18:58:03**:<br>

> The PPL of 500+ is not very promising. I suspect this is because of the not implemented technique to reduce the importance of recently used experts, which completely modifies the inference compared to how the model was trained, that was discussed in the mainline PR

Looking more closely, yes I see that the [official pytorch reference MoE routing "capacity" mechanism](https://github.com/Tencent-Hunyuan/Hunyuan-A13B/blob/95becb636c3ab95f203e10c51c5f090040886577/models/modeling_hunyuan.py#L74) is not seem implemented in [the build_moe_ffn() code](https://github.com/ubergarm/ik_llama.cpp/blob/ug/hunyuan-moe-2/src/llama.cpp#L9855).

The mainline PR `https://github.com/ggml-org/llama.cpp/pull/14425` seems still open for now, and yes no rush to merge this. (I've updated instructions on the hugginface model if any brave souls want to test the current implementation.)

I'll try quanting from the Pretrain version just to see how it performs, given that bf16 scores much lower PPL oddly enough:
```
model=Hunyuan-A13B-Pretrain-BF16-00001-of-00004.gguf
./build/bin/llama-perplexity \
        --model "$model" \
        -f wiki.test.raw \
        --seed 1337 \
        -ts 48,48 \
        -ngl 18 \
        --threads 24

Final estimate: PPL = 5.2880 +/- 0.03236
```

---

👤 **ikawrakow** submitted a review the **2025-07-09** at **08:29:32**: ✅ `APPROVED`<br>

OK, lets merge this.
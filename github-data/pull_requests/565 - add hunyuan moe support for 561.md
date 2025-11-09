## 🔀 [Pull Request #565](https://github.com/ikawrakow/ik_llama.cpp/pull/565) - add hunyuan moe support for 561

| **Author** | `ubergarm` |
| :--- | :--- |
| **State** | 🔀 **Merged** |
| **Source Branch** | `ug/hunyuan-moe-2` |
| **Target Branch** | `main` |
| **Created** | 2025-06-30 |
| **Updated** | 2025-07-15 |
| **Merged** | 2025-07-09 |

---

## 📄 Description

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

Would be great if anyone else could test e.g. @Downtown-Case as per [#561](https://github.com/ikawrakow/ik_llama.cpp/issues/561) 

I haven't yet made imatrix nor tried to quantize further.

Might be able to use one of the following if was converted recently enough:
* https://huggingface.co/bullerwins/Hunyuan-A13B-Instruct-GGUF
* https://huggingface.co/qwp4w3hyb/Hunyuan-A13B-Instruct-hf-WIP-GGUF

The behavior seems a bit odd and will answer in chinese if I don't use some kind of system prompt or explicitly say speak in english. Mainline seems to use some kind of `--jinja` thing which isn't supported here psure. So ymmv.

---

## 💬 Conversation

👤 **ubergarm** commented on **2025-06-30** at **18:28:48**

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
compute_imatrix: tokenization took 709.171 ms
compute_imatrix: computing over 865 chunks with batch_size 512
compute_imatrix: 4.37 seconds per pass - ETA 1 hours 3.07 minutes
[1]12.7104,[2]14.8010,[3]14.3374,[4]30.5778,[5]17.4738,[6]14.5285,[7]20.2402,[8]14.9318,[9]11.7604,
save_imatrix: stored collected data after 10 chunks in /mnt/raid/models/ubergarm/Hunyuan-A13B-Instruct-GGUF/imatrix-Hunyuan-A13B-Instruct-BF16.dat
[10]12.0205,[11]10.2799,[12]12.3863,[13]14.9808,[14]16.1885,[15]16.6677,[16]20.9547,[17]19.1613,[18]17.4531,[19]15.5200,
save_imatrix: stored collected data after 20 chunks in /mnt/raid/models/ubergarm/Hunyuan-A13B-Instruct-GGUF/imatrix-Hunyuan-A13B-Instruct-BF16.dat
[20]14.7222,[21]13.4574,[22]12.5603,[23]11.8334,[24]11.1943,[25]10.7840,[26]10.5614,[27]10.8168,[28]11.2630,[29]11.9753,
save_imatrix: stored collected data after 30 chunks in /mnt/raid/models/ubergarm/Hunyuan-A13B-Instruct-GGUF/imatrix-Hunyuan-A13B-Instruct-BF16.dat
[30]12.7904,[31]12.8568,[32]12.7520,[33]13.2066,[34]13.7438,[35]14.3701,[36]15.2825,[37]16.4474,[38]17.2615,[39]17.7246,
save_imatrix: stored collected data after 40 chunks in /mnt/raid/models/ubergarm/Hunyuan-A13B-Instruct-GGUF/imatrix-Hunyuan-A13B-Instruct-BF16.dat
[40]20.3797,[41]22.3074,[42]22.9196,[43]23.5967,[44]24.9652,[45]26.3450,[46]28.0728,[47]28.1975,[48]27.9526,[49]31.3467,
save_imatrix: stored collected data after 50 chunks in /mnt/raid/models/ubergarm/Hunyuan-A13B-Instruct-GGUF/imatrix-Hunyuan-A13B-Instruct-BF16.dat
[50]30.1730,[51]31.2195,[52]30.6089,[53]30.0938,[54]29.5127,[55]29.9680,[56]29.2944,[57]28.2416,[58]27.2467,[59]26.2110,
save_imatrix: stored collected data after 60 chunks in /mnt/raid/models/ubergarm/Hunyuan-A13B-Instruct-GGUF/imatrix-Hunyuan-A13B-Instruct-BF16.dat
[60]25.3394,[61]24.4437,[62]23.7538,[63]25.8637,[64]27.0096,[65]28.0507,[66]27.7521,[67]29.0344,[68]29.8659,[69]30.3886,
save_imatrix: stored collected data after 70 chunks in /mnt/raid/models/ubergarm/Hunyuan-A13B-Instruct-GGUF/imatrix-Hunyuan-A13B-Instruct-BF16.dat
[70]31.4350,[71]31.8531,[72]31.7906,[73]31.7912,[74]32.9230,[75]34.9214,[76]37.0384,[77]38.7590,[78]38.9847,[79]40.2656,
save_imatrix: stored collected data after 80 chunks in /mnt/raid/models/ubergarm/Hunyuan-A13B-Instruct-GGUF/imatrix-Hunyuan-A13B-Instruct-BF16.dat
[80]41.5627,[81]41.0075,[82]42.5855,[83]44.5075,[84]43.9110,[85]43.3078,[86]42.7130,[87]41.7924,[88]41.2850,[89]41.5686,
save_imatrix: stored collected data after 90 chunks in /mnt/raid/models/ubergarm/Hunyuan-A13B-Instruct-GGUF/imatrix-Hunyuan-A13B-Instruct-BF16.dat
[90]40.8182,[91]41.2610,[92]42.4782,[93]44.0758,[94]43.5943,[95]43.7613,[96]43.0079,[97]42.6615,[98]43.6499,[99]43.1762,
save_imatrix: stored collected data after 100 chunks in /mnt/raid/models/ubergarm/Hunyuan-A13B-Instruct-GGUF/imatrix-Hunyuan-A13B-Instruct-BF16.dat
[100]42.4092,[101]43.1918,[102]44.5605,[103]44.1737,[104]44.2998,[105]45.3024,[106]45.5803,[107]45.3388,[108]45.5154,[109]45.8490,
save_imatrix: stored collected data after 110 chunks in /mnt/raid/models/ubergarm/Hunyuan-A13B-Instruct-GGUF/imatrix-Hunyuan-A13B-Instruct-BF16.dat
[110]45.6819,[111]46.1607,[112]46.8070,[113]47.5833,[114]48.5492,[115]48.9797,[116]49.6842,[117]49.8659,[118]51.1640,[119]51.3824,
save_imatrix: stored collected data after 120 chunks in /mnt/raid/models/ubergarm/Hunyuan-A13B-Instruct-GGUF/imatrix-Hunyuan-A13B-Instruct-BF16.dat
[120]52.0141,[121]53.6073,[122]55.3684,[123]56.2596,[124]56.0548,[125]56.1662,[126]56.3532,[127]57.2403,[128]56.6770,[129]58.3851,
save_imatrix: stored collected data after 130 chunks in /mnt/raid/models/ubergarm/Hunyuan-A13B-Instruct-GGUF/imatrix-Hunyuan-A13B-Instruct-BF16.dat
[130]58.2333,[131]59.2614,[132]60.7497,[133]62.4619,[134]63.7352,[135]64.8522,[136]66.5478,[137]64.9457,[138]63.5455,[139]63.2199,
save_imatrix: stored collected data after 140 chunks in /mnt/raid/models/ubergarm/Hunyuan-A13B-Instruct-GGUF/imatrix-Hunyuan-A13B-Instruct-BF16.dat

...
```

*EDIT*: I also tried adding this model to the list for `ggml_flash_attn_ext_set_prec(cur, GGML_PREC_F32);` but the imatrix PPL values look about the same. Seems to be about the same with or without `-fmoe` as well.

fwiw this seems to be similar numbers as I'm getting using mainline llama-imatrix:

```
[1]12.7998,[2]14.9052,[3]14.4276,[4]30.9156,[5]17.5724,[6]14.6579,[7]20.3671,[8]15.0254,[9]11.8121,[10]12.0809,[11]10.3416,[12]12.4422,[13]15.1108,^C
```

---

👤 **ikawrakow** commented on **2025-06-30** at **20:20:40**

No FA and FA giving very different PPL values is not a good sign.

PPL of 60 is not a good sign either, especially for a model of that size.

---

👤 **ubergarm** commented on **2025-06-30** at **20:36:19**

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

👤 **ikawrakow** started a conversation on `src/llama.cpp` on **2025-07-01** at **06:00:36**

If you check your previous PR about GLM4 you will see that you had to remove the `Vcur` reshaping. It is the same here. Remove this line and it is likely the difference between FA and no FA will go away.

> 👤 **ubergarm** replied on **2025-07-01** at **23:54:30**
> 
> Yup, thanks for the reminder! The two trickiest parts of porting an architecture is remembering to:
> 
> 1. Remove the `Vcur` reshaping.
> 2. On mainline `build_attn()` the argument order goes `Qcur, Kcur, Vcur,`, but here with `llm_build_kv()` the order goes `Kcur, Vcur, Qcur,`.
> 
> Just re-downloaded the new .safetensors, converted, and built a fresh quant to test:
> 
> * `FA=1` Final estimate: PPL = 522.7473 +/- 5.68072
> * `FA=0` Final estimate: PPL = 527.6625 +/- 5.73144
> 
> So looks "good" now haha... I didn't wait to find the bf16's PPL but this lines up in the ball-park with what [mainline is seeing around ~500](https://github.com/ggml-org/llama.cpp/pull/14425#issuecomment-3024357323).
> 
> Of course I couldn't help myself and had to try out [the new IQ3_KS quant](https://github.com/ikawrakow/ik_llama.cpp/pull/566) as well lol...
> 
> So far so good!
> ```
> llm_load_print_meta: model type       = 80B.A13B
> llm_load_print_meta: model ftype      = IQ3_KS - 3.1875 bpw
> llm_load_print_meta: model params     = 80.393 B
> llm_load_print_meta: model size       = 34.088 GiB (3.642 BPW)
> llm_load_print_meta: general.name     = Hunyuan A13B Instruct
> 
> # Attention
> blk\..*\.attn_k.*=iq6_k
> blk\..*\.attn_v.*=iq6_k
> 
> blk\..*\.attn_q.*=iq5_k
> blk\..*\.attn_o.*=iq5_k
> 
> # 1x Shared Expert
> blk\..*\.ffn_(down)_shexp.*=iq6_k
> blk\..*\.ffn_(gate|up)_shexp.*=iq5_k
> 
> # 64x Routed Experts
> blk\..*\.ffn_(down)_exps.*=iq4_ks
> blk\..*\.ffn_(gate|up)_exps.*=iq3_ks # let's live dangerously
> 
> # Token Embedding
> token_embd\.weight=iq6_k # splurged here a bit as this model's tokenization seems wierd
> ```

---

👤 **kiron111** commented on **2025-07-02** at **02:42:43**

> I'm going to leave an endpoint up for a little bit if anyone wants to try the first experimental quant.. No promises lol
> 
> ## Endpoint
> WebUI: https://llm.ubergarm.com/ APIEndpoint: https://llm.ubergarm.com/ (it is llama-server API endpoint with no API key)
> 
> There are 8 concurrent slots each with 64k prompt limit.
> 
> ## Test Quant
> I just rolled an imatrix.dat and made my first quant for testing.
> 
> ```
> llm_load_print_meta: model type       = 80B.A13B
> llm_load_print_meta: model ftype      = IQ4_K - 4.5 bpw
> llm_load_print_meta: model params     = 80.393 B
> llm_load_print_meta: model size       = 48.581 GiB (5.191 BPW)
> llm_load_print_meta: general.name     = Hunyuan A13B Instruct
> ```
> 
> ```
> blk\..*\.attn_k.*=iq6_k
> blk\..*\.attn_v.*=iq6_k
> 
> blk\..*\.attn_q.*=iq5_k
> blk\..*\.attn_o.*=iq5_k
> 
> # 1x Shared Expert
> blk\..*\.ffn_(gate|up)_shexp.*=iq6_k
> blk\..*\.ffn_(down)_shexp.*=iq5_k
> 
> # 64x Routed Experts
> blk\..*\.ffn_(gate|up)_exps.*=iq5_k
> blk\..*\.ffn_(down)_exps.*=iq4_k
> 
> # Token Embedding
> token_embd\.weight=iq4_k
> ```
> 
> How I ran it:
> 
> ```shell
> model=/mnt/raid/models/ubergarm/Hunyuan-A13B-Instruct-GGUF/Hunyuan-A13B-Instruct-IQ4_K.gguf
> ./build/bin/llama-server \
>   --model "$model" \
>   --alias ubergarm/Hunyuan-A13B-Instruct-IQ4_K \
>   -fa \
>   -ctk q8_0 -ctv q8_0 \
>   -c 524288 \
>   --temp 0.6 \
>   --presence-penalty 0.7 \
>   --min-p 0.1 \
>   -ts 48,48 \
>   -ngl 99 \
>   --parallel 8 \
>   --threads 1 \
>   --host 127.0.0.1 \
>   --port 8080
> ```

tested on your api, it works for Chinese Q&A.

---

👤 **ubergarm** commented on **2025-07-02** at **02:44:39**

> tested on your api, it works for Chinese Q&A.

Ahh very good, thanks you. Tonight I was running my updated [experimental IQ3_KS](https://huggingface.co/ubergarm/Hunyuan-A13B-Instruct-GGUF/blob/main/Hunyuan-A13B-Instruct-IQ3_KS.gguf) which I went ahead and released prematurely because oh well it seems okay lol...

Thanks for testing!

![swee-bench-Hunyuan-A13B-Instruct-IQ3_KS](https://github.com/user-attachments/assets/9f9fb8a0-8e1a-4798-b5db-549490010fed)

It can fit 256k context in under 24GB VRAM when not offloading additional exps and with `-ub 1024` get over 500 tok/sec PP and about 17 tok/sec TG. So quite a flexible model in terms of size at least.

---

👤 **kiron111** commented on **2025-07-02** at **03:54:01**

run on wsl I got a error: Floating point exception (core dumped), in the initial procress of ik_llama.cpp

```
model=/mnt/g/lm-studio/models/ubergarm/Hunyuan-A13B-Instruct-GGUF/Hunyuan-A13B-Instruct-IQ3_KS.gguf
./build/bin/llama-server \
  --model "$model" \
  --alias ubergarm/Hunyuan-A13B-Instruct-IQ4_K \
  -fa \
  -ctk q8_0 -ctv q8_0 \
  -c 4096 \
  --temp 0.6 \
  --presence-penalty 0.7 \
  --min-p 0.1 \
  -ts 48,48 \
  -ngl 99 \
  --parallel 8 \
  --threads 1 \
  --host 127.0.0.1 \
  --port 8080
INFO [                    main] build info | tid="140543167610880" timestamp=1751427934 build=3776 commit="c6c23fa4"
INFO [                    main] system info | tid="140543167610880" timestamp=1751427934 n_threads=24 n_threads_batch=-1 total_threads=16 system_info="AVX = 1 | AVX_VNNI = 0 | AVX2 = 1 | AVX512 = 0 | AVX512_VBMI = 0 | AVX512_VNNI = 0 | AVX512_BF16 = 0 | FMA = 1 | NEON = 0 | SVE = 0 | ARM_FMA = 0 | F16C = 1 | FP16_VA = 0 | WASM_SIMD = 0 | BLAS = 1 | SSE3 = 1 | SSSE3 = 1 | VSX = 0 | MATMUL_INT8 = 0 | LLAMAFILE = 1 | "
Floating point exception (core dumped)
```
OS: Win 11 +WSL
branch: Uberarm/huanyuan-moe2 ( I use this command to download the source code: git clone --branch ug/hunyuan-moe-2 https://github.com/ubergarm/ik_llama.cpp.git)
Cmake config: cmake -B ./build -DGGML_CUDA=ON -DGGML_BLAS=OFF -DGGML_SCHED_MAX_COPIES=1 -DGGML_CUDA_IQK_FORCE_BF16=1 -DGGML_CUDA_F16=ON -DGGML_CCACHE=OFF
Hardware: AMD Ryzen 5700X, 4070 12GB, WSL ram limit = 42 ( physical ram 48 Gb in total)

---

👤 **ubergarm** commented on **2025-07-02** at **04:03:30**

> run on wsl I got a error: Floating point exception (core dumped), in the initial procress of ik_llama.cpp

Its becase I'm a madman and released a quant depending on two unmerged PRs. Check here for instructions how to get the IQ3_KS PR here: https://huggingface.co/ubergarm/Hunyuan-A13B-Instruct-GGUF#note-building-experimental-prs

Also @kiron111 look at the examples on the model card you will need to use `-ngl 99 -ot exps=CPU` remove that `-ts` stuff that was specific to my test rig.

This model is great for low VRAM machines and can probably run in 6GB VRAM with some usable context.

---

👤 **kiron111** commented on **2025-07-02** at **04:08:50**

> > run on wsl I got a error: Floating point exception (core dumped), in the initial procress of ik_llama.cpp
> 
> Its becase I'm a madman and released a quant depending on two unmerged PRs. Check here for instructions how to get the IQ3_KS PR here: https://huggingface.co/ubergarm/Hunyuan-A13B-Instruct-GGUF#note-building-experimental-prs
> 
> Also @kiron111 look at the examples on the model card you will need to use `-ngl 99 -ot exps=CPU` remove that `-ts` stuff that was specific to my test rig.
> 
> This model is great for low VRAM machines and can probably run in 6GB VRAM with some usable context.

thankyou
oh....I miss so many points...let me redownload and recompile first

---

👤 **ikawrakow** commented on **2025-07-02** at **07:34:31**

The PPL of 500+ is not very promising. I suspect this is because of the not implemented technique to reduce the importance of recently used experts, which completely modifies the inference compared to how the model was trained, that was discussed in the mainline PR. Hence still wondering if to merge. They have merged as is in mainline, but `ik_llama.cpp` tries to be better than mainline.

---

👤 **ubergarm** commented on **2025-07-02** at **18:58:03**

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

*EDIT*
```
model=Hunyuan-A13B-Pretrain-IQ3_KS.gguf
# model type       = 80B.A13B
# model ftype      = IQ3_KS - 3.1875 bpw
# model params     = 80.393 B
# model size       = 34.088 GiB (3.642 BPW)
# general.name     = Hunyuan A13B Pretrain

Final estimate: PPL = 5.4382 +/- 0.03349
```

---

👤 **ikawrakow** approved this pull request ✅ on **2025-07-09** at **08:29:32**

OK, lets merge this.
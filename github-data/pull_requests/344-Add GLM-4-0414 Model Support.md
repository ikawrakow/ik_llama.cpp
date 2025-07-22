### 🔀 [#344](https://github.com/ikawrakow/ik_llama.cpp/pull/344) - Add GLM-4-0414 Model Support

| **Author** | `ubergarm` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-04-24 |
| **Updated** | 2025-05-08 |

---

#### Description

This is my second attempt which still has some issues. Original attempt was #333. This one is based on https://github.com/ggml-org/llama.cpp/pull/12867 . However, this PR does not bring over any of the python stuff.

In limited testing with of [bartowski/THUDM_GLM-Z1-32B-0414-GGUF](https://huggingface.co/bartowski/THUDM_GLM-Z1-32B-0414-GGUF/blob/main/THUDM_GLM-Z1-32B-0414-IQ4_XS.gguf) on CPU only and CUDA backends it seems to work as long as:

1. Flash Attention must be explicitly enabled e.g. `-fa`
2. If using CUDA, cannot offload >= 60 layers. (works up to -ngl 59).

## Example Command
This is one way to run it on CUDA that seems to work:
```
./build/bin/llama-server \
    --alias "bartowski/THUDM_GLM-Z1-32B-0414-IQ4_XS" \
    --model /mnt/astrodata/llm/models/bartowski/THUDM_GLM-Z1-32B-0414-GGUF/THUDM_GLM-Z1-32B-0414-IQ4_XS.gguf \
    -fa \
    --ctx-size 8192 \
    --n-gpu-layers 59 \
    --threads 8 \
    --host 127.0.0.1 \
    --port 8080
```

If I increase `--n-gpu-layers 60` or higher, it outputs `GGGGGGGGGGGGGGG`.

It also seems okay to add `-amb 512 -ctk q8_0 -ctv q8_0`...

fwiw there seems to be some issues still on mainline implementation possibly related:

* https://github.com/ggml-org/llama.cpp/issues/12946#issuecomment-2824836433
* https://github.com/ggml-org/llama.cpp/pull/13099

So I'll mark this as draft for now and see how things are looking soon.

---

#### 💬 Conversation

👤 **ikawrakow** commented the **2025-04-25** at **07:29:50**:<br>

> If I increase --n-gpu-layers 60 or higher, it outputs GGGGGGGGGGGGGGG.

Does it also happen when you use `-ctk q8_0 -ctv q8_0`? There is [this PR](https://github.com/ggml-org/llama.cpp/pull/13101) in mainline where they want to force `f32` for cuBLAS matrix multiplications (those get used for attention calculations when KV cache is `f16`) to get meaningful results out of GLM-4. This indicates that `f16` may not have enough range to accommodate the GLM-4 numerical range. In such cases using quantized cache may help.

---

👤 **ubergarm** commented the **2025-04-25** at **14:37:32**:<br>

Hrrm, unfortunately no using `-ctk q8_0 -ctv q8_0` with `-ngl 60` (or higher) still throws `GGGGGGGG`... 

Without `-fa` and under <60 layers offload it looks like this: `转 Cuomo. кури....bl的话 the", E neuronal,,-T...� l -氏 Blarnalc见�.flow总 in杯商house^C`

I also tried [city96's patch to force `f32` dtype](https://github.com/ggml-org/llama.cpp/pull/13101/files) as shown below, but that didn't seem to fix this issue either:

```
--- a/src/llama.cpp
+++ b/src/llama.cpp
@@ -9188,6 +9188,10 @@ static struct ggml_tensor * llm_build_ffn(

     if (down) {
         cur = llm_build_lora_mm(lctx, ctx, down, cur);
+        if (lctx.model.arch == LLM_ARCH_GLM4) {
+            // GLM4 seems to have numerical issues with half-precision accumulators
+            ggml_mul_mat_set_prec(cur, GGML_PREC_F32);
+        }
     }
```

Could be that I made a mistake in the `build_glm4()` the attention cgraph? Interestingly this invocation seems to works fine too:
```
./build/bin/llama-server \
    --alias "bartowski/THUDM_GLM-Z1-32B-0414-IQ4_XS" \
    --model /mnt/astrodata/llm/models/bartowski/THUDM_GLM-Z1-32B-0414-GGUF/THUDM_GLM-Z1-32B-0414-IQ4_XS.gguf \
    -fa \
    --ctx-size 8192 \
    --n-gpu-layers 99 \
    -ot attn=CPU \
    -nkvo \
    --threads 8 \
    --host 127.0.0.1 \
    --port 8080
```

Last observations are that mainline seems to work fine with or without `-fa` and also mainline is *much slower* even fully offloaded e.g. 20 tok/sec PP and 5 tok/s TG. Compared to `ik_llama.cp` getting 163 tok/sec PP and 17 tok/sec TG with `-ot attn=CPU -nkvo` and even faster at 271 tok/sec PP and 25 tok/sec TG with `-ngl 59`...

Not sure what to try next other than dig in deeper to how `build_inp_KQ_mask()` and `llm_build_kv` have changed with mainline refactors or something...

---

👤 **ikawrakow** commented the **2025-04-25** at **14:43:20**:<br>

> Could be that I made a mistake in the build_glm4() the attention cgraph? Interestingly this invocation seems to works fine too:

If you make a mistake with building the graph, this invocation wouldn't be working. If it works for all layers offloaded to the GPU except attention tensors and KV cache, it means there is a precision issue in the attention calculation on CUDA (on the CPU everything is computed with `fp32` precision).

---

👤 **ubergarm** commented the **2025-04-25** at **14:48:42**:<br>

I just noticed one more odd thing trying `-ot attn=CPU -ot .*=CUDA0` on `ik_llama.cpp` it prints this out on startup then crashes. There are two `__missing__` types per layer it seems... 

```
Tensor token_embd.weight buffer type overriden to CPU
Tensor output_norm.weight buffer type overriden to CPU
Tensor output.weight buffer type overriden to CPU
Tensor blk.0.attn_norm.weight buffer type overriden to CPU
Tensor __missing__ buffer type overriden to CPU
Tensor __missing__ buffer type overriden to CPU
Tensor blk.0.attn_q.weight buffer type overriden to CPU
Tensor blk.0.attn_k.weight buffer type overriden to CPU
Tensor blk.0.attn_v.weight buffer type overriden to CPU
Tensor blk.0.attn_q.bias buffer type overriden to CPU
Tensor blk.0.attn_k.bias buffer type overriden to CPU
Tensor blk.0.attn_v.bias buffer type overriden to CPU
Tensor blk.0.attn_output.weight buffer type overriden to CPU
Tensor blk.0.post_attention_norm.weight buffer type overriden to CPU
Tensor blk.0.ffn_norm.weight buffer type overriden to CPU
Tensor blk.0.ffn_down.weight buffer type overriden to CPU
Tensor blk.0.ffn_up.weight buffer type overriden to CPU
Tensor blk.0.post_ffw_norm.weight buffer type overriden to CPU
```

---

👤 **ikawrakow** commented the **2025-04-25** at **14:50:36**:<br>

Try this: in the function `llm_build_kqv()`, on all lines that have
```
if (model.arch == LLM_ARCH_PHI2 || model.arch == LLM_ARCH_PHI3 || model.arch == LLM_ARCH_GPTNEOX etc
```
add `|| model.arch == LLM_ARCH_GLM4`.

This will set the precision of the `K*Q` calculation to `fp32`, and hopefully fix the issue when all layers are offloaded to the GPU.

---

👤 **ikawrakow** commented the **2025-04-25** at **14:57:27**:<br>

I see in mainline `llama.cpp` they have become tired of setting the `K*Q` calculation for `fp32` precision for specific models, and now have this
```c++
        ggml_tensor * kq = ggml_mul_mat(ctx0, k, q);

        // note: this op tends to require high floating point range
        //       while for some models F16 is enough, for others it is not, so we default to F32 here
        ggml_mul_mat_set_prec(kq, GGML_PREC_F32);
```

This is why mainline may be working for this model. I still refuse to set that generically for all models as this hurts performance for long contexts quite a bit. The downside is that one needs to explicitly enable `fp32` precision when necessary for a model.

---

👤 **ubergarm** commented the **2025-04-25** at **15:01:49**:<br>

> add || model.arch == LLM_ARCH_GLM4

Yes, this fixed the issue, I can fully offload now!

```
--- a/src/llama.cpp
+++ b/src/llama.cpp
@@ -9415,7 +9415,7 @@ static struct ggml_tensor * llm_build_kqv(
         // For DeepSeek-2, it is perfectly fine with fp16 for PP, but I get gibberish when uding fp16 for TG.
         // Not sure if it is really a matter of insufficient precision, or I have made a mistake in the fattn-vec-f16 kernel.
         if (model.arch == LLM_ARCH_PHI2 || model.arch == LLM_ARCH_PHI3 || model.arch == LLM_ARCH_GPTNEOX ||
-            (model.arch == LLM_ARCH_DEEPSEEK2 && q->ne[1] <= 8)) {
+            model.arch == LLM_ARCH_GLM4 || (model.arch == LLM_ARCH_DEEPSEEK2 && q->ne[1] <= 8)) {
             ggml_flash_attn_ext_set_prec(cur, GGML_PREC_F32);
         }
```

I'll push this up.

Remaining questions:
* Is it okay that it does *not* work without `-fa` ?
* I didn't test on other hardware nor include the latest mainline patch `ggml_mul_mat_set_prec(cur, GGML_PREC_F32);`.

---

👤 **ubergarm** commented the **2025-04-25** at **15:35:45**:<br>

Okay, so now without `-fa` it no longer produces `GGGGGGG` but it is back to this kinda stuff:

```
arsTab�.^rellsúng pacirc Pepper九龙每:室hlt一层avit学isi� cé个义项PMC\":列为� friAZalyrátolpanies�formanceInvoke9不足 Cornel Naz/Rkoz�koz�INFedomaidaporaidariantchartôaid
```

I'll look for a reference, I thought I've seen others mentioning this kinda output before.

---

👤 **ikawrakow** submitted a review the **2025-04-25** at **16:58:38**: 💬 `COMMENTED`

---

👤 **ikawrakow** commented during a code review the **2025-04-25** at **16:58:38** on `src/llama.cpp`:<br>

Add
```c++
                if (model.arch == LLM_ARCH_GLM4) {
                    ggml_mul_mat_set_prec(kqv_i, GGML_PREC_F32);
                }
```
after line 9515

---

👤 **ikawrakow** submitted a review the **2025-04-25** at **17:01:07**: 💬 `COMMENTED`

---

👤 **ikawrakow** commented during a code review the **2025-04-25** at **17:01:07** on `src/llama.cpp`:<br>

Add
```c++
if ( model.arch == LLM_ARCH_GLM4) {
     ggml_mul_mat_set_prec(kqv, GGML_PREC_F32);
}
```
after line 9475

---

👤 **ikawrakow** commented the **2025-04-25** at **17:07:32**:<br>

I don't think any of the suggestions you are finding around the Internet are going to help. Just think about it:
* It works on the CPU (calculation done with `fp32`)
* It works with FA after setting precision to `fp32`
* You set the precision of the `K*Q` matrix multiplication to `fp32` and it improved things, but did not fix. Getting `GGGGG...` basically means there are NaNs in the result. Getting gibberish output means the values are finite but not meaningful.

The only logical conclusion from these 3 observations is that you also need to set the precision to `fp32` for the `kqv = V*softmax(K*Q)` matrix multiplication. The other option would be that things go wrong in the `softmax` calculation on CUDA.  But looking at the CUDA `softmax` implementation, it is already done using `fp32` arithmetic. Hence, it must be the `kqv` matrix multiplication.

---

👤 **ubergarm** commented the **2025-04-25** at **18:31:20**:<br>

Thanks, I appreciate you helping me learn on this.

Just to be clear I'm getting the gibberish output without `-fa` on *both* CPU only as well as CUDA backend.

I tried setting precision to fp32 as you describe, but still get the same gibberish.
<details>

<summary>The patch you suggested above.</summary>
I went ahead and tried this and it seems to be taking the `kqv` path and not the `kqv_i` but still giving same gibberish.
```
--- a/src/llama.cpp
+++ b/src/llama.cpp
@@ -9473,6 +9473,9 @@ static struct ggml_tensor * llm_build_kqv(
             GGML_ASSERT(kv.size == n_ctx);

             struct ggml_tensor * kqv = ggml_mul_mat(ctx, v, kq);
+            if (model.arch == LLM_ARCH_GLM4) {
+                ggml_mul_mat_set_prec(kqv, GGML_PREC_F32);
+            }
             cb(kqv, "kqv", il);

             struct ggml_tensor * kqv_merged = ggml_permute(ctx, kqv, 0, 2, 1, 3);
@@ -9513,6 +9516,9 @@ static struct ggml_tensor * llm_build_kqv(
                 i02 = i12 / r2v;
                 auto v_i = ggml_view_3d(ctx, v, v->ne[0], v->ne[1], this_ne12, v->nb[1], v->nb[2], v->nb[2]*i02);
                 auto kqv_i = ggml_mul_mat(ctx, v_i, kq_i);
+                if (model.arch == LLM_ARCH_GLM4) {
+                    ggml_mul_mat_set_prec(kqv_i, GGML_PREC_F32);
+                }
                 if (i12 == 0) {
                     kqv = kqv_i;
                 } else {
```
</details>

I'll dig into the differences between mainline non flash attention and this forks non flash attention path more to see if anything else sticks out to me.

---

👤 **ikawrakow** commented the **2025-04-26** at **06:02:40**:<br>

> Just to be clear I'm getting the gibberish output without -fa on both CPU only as well as CUDA backend.

Sorry, I missed the fact that it is not working on the CPU without FA. If I had paid better attention, I would have diagnosed the problem much earlier.

Simply remove the line (line 15686 in the version I just cloned from your repository)
```c++
Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);
```

In mainline they have reorganized how attention is built. Reshaping `V` to 3D at this point fits with their `build_attn` function, but not with the way things are done here (and were formerly done in mainline). I tested and it works!

---

👤 **ikawrakow** commented the **2025-04-26** at **07:19:17**:<br>

Here a quick CPU only `sweep-bench` performance comparison to mainline for the [bartowski/THUDM_GLM-Z1-32B-0414-GGUF](https://huggingface.co/bartowski/THUDM_GLM-Z1-32B-0414-GGUF/blob/main/THUDM_GLM-Z1-32B-0414-IQ4_XS.gguf) model you are using

### Mainline
```
./bin/llama-sweep-bench -m THUDM_GLM-Z1-32B-0414-IQ4_XS.gguf -c 8192 -t 32 -fa -ctk q8_0 -ctv q8_0
```

|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|   512 |    128 |      0 |   19.729 |    25.95 |   30.953 |     4.14 |
|   512 |    128 |    512 |   20.930 |    24.46 |   31.639 |     4.05 |
|   512 |    128 |   1024 |   22.138 |    23.13 |   32.156 |     3.98 |
|   512 |    128 |   1536 |   23.310 |    21.96 |   32.627 |     3.92 |
|   512 |    128 |   2048 |   24.451 |    20.94 |   33.047 |     3.87 |
|   512 |    128 |   2560 |   25.607 |    19.99 |   33.452 |     3.83 |
|   512 |    128 |   3072 |   26.732 |    19.15 |   33.765 |     3.79 |
|   512 |    128 |   3584 |   27.819 |    18.40 |   34.119 |     3.75 |
|   512 |    128 |   4096 |   28.965 |    17.68 |   34.460 |     3.71 |
|   512 |    128 |   4608 |   30.076 |    17.02 |   34.823 |     3.68 |
|   512 |    128 |   5120 |   31.207 |    16.41 |   35.184 |     3.64 |
|   512 |    128 |   5632 |   32.371 |    15.82 |   35.544 |     3.60 |
|   512 |    128 |   6144 |   33.485 |    15.29 |   35.917 |     3.56 |
|   512 |    128 |   6656 |   34.627 |    14.79 |   36.275 |     3.53 |
|   512 |    128 |   7168 |   35.749 |    14.32 |   36.641 |     3.49 |
|   512 |    128 |   7680 |   36.891 |    13.88 |   37.006 |     3.46 |


### ik_llama.cpp

```
./bin/llama-sweep-bench -m THUDM_GLM-Z1-32B-0414-IQ4_XS.gguf -c 8192 -t 32 -fa -ctk q8_0 -ctv q8_0 -rtr
```
(but I needed the changes in PR #349 to make FA work on the CPU).

|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|   512 |    128 |      0 |    7.275 |    70.38 |   30.690 |     4.17 |
|   512 |    128 |    512 |    7.445 |    68.77 |   31.104 |     4.12 |
|   512 |    128 |   1024 |    7.608 |    67.30 |   31.206 |     4.10 |
|   512 |    128 |   1536 |    7.778 |    65.83 |   31.421 |     4.07 |
|   512 |    128 |   2048 |    7.929 |    64.57 |   31.559 |     4.06 |
|   512 |    128 |   2560 |    8.087 |    63.31 |   31.746 |     4.03 |
|   512 |    128 |   3072 |    8.243 |    62.11 |   31.883 |     4.01 |
|   512 |    128 |   3584 |    8.405 |    60.91 |   32.053 |     3.99 |
|   512 |    128 |   4096 |    8.545 |    59.92 |   32.169 |     3.98 |
|   512 |    128 |   4608 |    8.706 |    58.81 |   32.351 |     3.96 |
|   512 |    128 |   5120 |    8.855 |    57.82 |   32.398 |     3.95 |
|   512 |    128 |   5632 |    9.025 |    56.73 |   32.591 |     3.93 |
|   512 |    128 |   6144 |    9.164 |    55.87 |   32.655 |     3.92 |
|   512 |    128 |   6656 |    9.316 |    54.96 |   32.838 |     3.90 |
|   512 |    128 |   7168 |    9.476 |    54.03 |   32.902 |     3.89 |
|   512 |    128 |   7680 |    9.635 |    53.14 |   33.091 |     3.87 |

---

👤 **ubergarm** commented the **2025-04-26** at **14:53:00**:<br>

Sweeet that fixes up the non-flash-attention case! This model is quite efficient, I just ran it with 128k context and only using `21194MiB` VRAM ?? Looking forward to some testing and benchmarking soon.

For now I'll fixup this PR and put it after your recent cohere2 additions and set it to review ready afterwards.

Thanks again really appreciate your time looking at this! Cheers!

---

👤 **ubergarm** commented the **2025-04-26** at **15:23:46**:<br>

Okay got it rebased, gonna force push it up after quick final test!!!

---

👤 **ikawrakow** submitted a review the **2025-04-26** at **15:33:46**: ✅ `APPROVED`

---

👤 **ubergarm** commented the **2025-04-26** at **15:41:12**:<br>

Yaay!! Feels good to finally get that model working haha... Thanks again for your patience and guidance! Have a g'night!

---

👤 **ubergarm** commented the **2025-04-26** at **20:04:37**:<br>

> Here a quick CPU only sweep-bench performance comparison to mainline for the [bartowski/THUDM_GLM-Z1-32B-0414-GGUF](https://huggingface.co/bartowski/THUDM_GLM-Z1-32B-0414-GGUF/blob/main/THUDM_GLM-Z1-32B-0414-IQ4_XS.gguf) model you are using

I followed your lead and ran some `llama-sweep-bench` comparisons too. My CPU-only benchmarks line up with yours, but my GPU results surprised me and didn't look as good assuming my quick fixup of @saood06's [llama-sweep-bench](https://github.com/ubergarm/llama.cpp/blob/ug/port-sweep-bench/examples/sweep-bench/sweep-bench.cpp) back to mainline isn't introducing some issue.

## ik's CPU-only test

![thud-sweep-03-ik-CPU](https://github.com/user-attachments/assets/50ce592f-33b8-4a46-9f68-a92c8101ba00)

## my CPU-only test

![thud-sweep-01-CPU](https://github.com/user-attachments/assets/69292d5f-3be9-45f4-b66f-9a8bae445385)

<details>

<summary>Logs</summary>

## `llama.cpp@558a76`
Plus github.com/ubergarm/llama.cpp `ug/port-sweep-bench` branch.
```
$ ./build/bin/llama-sweep-bench \
    --model /mnt/astrodata/llm/models/bartowski/THUDM_GLM-Z1-32B-0414-GGUF/THUDM_GLM-Z1-32B-0414-IQ4_XS.gguf \
    -fa -ctk q8_0 -ctv q8_0 \
    -c 5120 \
    --no-mmap \
    --threads 16

build: 5192 (e59a5f1e) with cc (GCC) 14.2.1 20250128 for x86_64-pc-linux-gnu
llama_model_loader: loaded meta data with 37 key-value pairs and 613 tensors from /mnt/astrodata/llm/models/bartowski/THUDM_GLM-Z1-32B-0414-GGUF/THUDM_GLM-Z1-32B-0414-IQ4_XS.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = glm4
llama_model_loader: - kv   1:                               general.type str              = model
llama_model_loader: - kv   2:                               general.name str              = GLM Z1 32B 0414
llama_model_loader: - kv   3:                            general.version str              = 0414
llama_model_loader: - kv   4:                           general.basename str              = GLM-Z1
llama_model_loader: - kv   5:                         general.size_label str              = 32B
llama_model_loader: - kv   6:                            general.license str              = mit
llama_model_loader: - kv   7:                               general.tags arr[str,1]       = ["text-generation"]
llama_model_loader: - kv   8:                          general.languages arr[str,2]       = ["zh", "en"]
llama_model_loader: - kv   9:                           glm4.block_count u32              = 61
llama_model_loader: - kv  10:                        glm4.context_length u32              = 32768
llama_model_loader: - kv  11:                      glm4.embedding_length u32              = 6144
llama_model_loader: - kv  12:                   glm4.feed_forward_length u32              = 23040
llama_model_loader: - kv  13:                  glm4.attention.head_count u32              = 48
llama_model_loader: - kv  14:               glm4.attention.head_count_kv u32              = 2
llama_model_loader: - kv  15:                        glm4.rope.freq_base f32              = 10000.000000
llama_model_loader: - kv  16:      glm4.attention.layer_norm_rms_epsilon f32              = 0.000010
llama_model_loader: - kv  17:                  glm4.attention.key_length u32              = 128
llama_model_loader: - kv  18:                glm4.attention.value_length u32              = 128
llama_model_loader: - kv  19:                  glm4.rope.dimension_count u32              = 64
llama_model_loader: - kv  20:                       tokenizer.ggml.model str              = gpt2
llama_model_loader: - kv  21:                         tokenizer.ggml.pre str              = glm4
llama_model_loader: - kv  22:                      tokenizer.ggml.tokens arr[str,151552]  = ["!", "\"", "#", "$", "%", "&", "'", ...
llama_model_loader: - kv  23:                  tokenizer.ggml.token_type arr[i32,151552]  = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...
llama_model_loader: - kv  24:                      tokenizer.ggml.merges arr[str,318088]  = ["Ġ Ġ", "Ġ ĠĠĠ", "ĠĠ ĠĠ", "...
llama_model_loader: - kv  25:                tokenizer.ggml.eos_token_id u32              = 151329
llama_model_loader: - kv  26:            tokenizer.ggml.padding_token_id u32              = 151329
llama_model_loader: - kv  27:                tokenizer.ggml.eot_token_id u32              = 151336
llama_model_loader: - kv  28:            tokenizer.ggml.unknown_token_id u32              = 151329
llama_model_loader: - kv  29:                tokenizer.ggml.bos_token_id u32              = 151331
llama_model_loader: - kv  30:                    tokenizer.chat_template str              = [gMASK]<sop>{%- if tools -%}<|system|...
llama_model_loader: - kv  31:               general.quantization_version u32              = 2
llama_model_loader: - kv  32:                          general.file_type u32              = 30
llama_model_loader: - kv  33:                      quantize.imatrix.file str              = /models_out/GLM-Z1-32B-0414-GGUF/THUD...
llama_model_loader: - kv  34:                   quantize.imatrix.dataset str              = /training_dir/calibration_datav3.txt
llama_model_loader: - kv  35:             quantize.imatrix.entries_count i32              = 366
llama_model_loader: - kv  36:              quantize.imatrix.chunks_count i32              = 125
llama_model_loader: - type  f32:  245 tensors
llama_model_loader: - type q5_K:   61 tensors
llama_model_loader: - type q6_K:    1 tensors
llama_model_loader: - type iq4_xs:  306 tensors
print_info: file format = GGUF V3 (latest)
print_info: file type   = IQ4_XS - 4.25 bpw
print_info: file size   = 16.38 GiB (4.32 BPW)
load: special_eot_id is not in special_eog_ids - the tokenizer config may be incorrect
load: special tokens cache size = 14
load: token to piece cache size = 0.9710 MB
print_info: arch             = glm4
print_info: vocab_only       = 0
print_info: n_ctx_train      = 32768
print_info: n_embd           = 6144
print_info: n_layer          = 61
print_info: n_head           = 48
print_info: n_head_kv        = 2
print_info: n_rot            = 64
print_info: n_swa            = 0
print_info: n_swa_pattern    = 1
print_info: n_embd_head_k    = 128
print_info: n_embd_head_v    = 128
print_info: n_gqa            = 24
print_info: n_embd_k_gqa     = 256
print_info: n_embd_v_gqa     = 256
print_info: f_norm_eps       = 0.0e+00
print_info: f_norm_rms_eps   = 1.0e-05
print_info: f_clamp_kqv      = 0.0e+00
print_info: f_max_alibi_bias = 0.0e+00
print_info: f_logit_scale    = 0.0e+00
print_info: f_attn_scale     = 0.0e+00
print_info: n_ff             = 23040
print_info: n_expert         = 0
print_info: n_expert_used    = 0
print_info: causal attn      = 1
print_info: pooling type     = 0
print_info: rope type        = 0
print_info: rope scaling     = linear
print_info: freq_base_train  = 10000.0
print_info: freq_scale_train = 1
print_info: n_ctx_orig_yarn  = 32768
print_info: rope_finetuned   = unknown
print_info: ssm_d_conv       = 0
print_info: ssm_d_inner      = 0
print_info: ssm_d_state      = 0
print_info: ssm_dt_rank      = 0
print_info: ssm_dt_b_c_rms   = 0
print_info: model type       = 32B
print_info: model params     = 32.57 B
print_info: general.name     = GLM Z1 32B 0414
print_info: vocab type       = BPE
print_info: n_vocab          = 151552
print_info: n_merges         = 318088
print_info: BOS token        = 151331 '[gMASK]'
print_info: EOS token        = 151329 '<|endoftext|>'
print_info: EOT token        = 151336 '<|user|>'
print_info: UNK token        = 151329 '<|endoftext|>'
print_info: PAD token        = 151329 '<|endoftext|>'
print_info: LF token         = 198 'Ċ'
print_info: EOG token        = 151329 '<|endoftext|>'
print_info: EOG token        = 151336 '<|user|>'
print_info: max token length = 1024
load_tensors: loading model tensors, this can take a while... (mmap = false)
load_tensors: offloading 0 repeating layers to GPU
load_tensors: offloaded 0/62 layers to GPU
load_tensors:          CPU model buffer size = 16775.23 MiB
...............................................................................................
llama_context: constructing llama_context
llama_context: n_seq_max     = 1
llama_context: n_ctx         = 5120
llama_context: n_ctx_per_seq = 5120
llama_context: n_batch       = 2048
llama_context: n_ubatch      = 512
llama_context: causal_attn   = 1
llama_context: flash_attn    = 1
llama_context: freq_base     = 10000.0
llama_context: freq_scale    = 1
llama_context: n_ctx_per_seq (5120) < n_ctx_train (32768) -- the full capacity of the model will not be utilized
llama_context:        CPU  output buffer size =     0.58 MiB
init: kv_size = 5120, offload = 1, type_k = 'q8_0', type_v = 'q8_0', n_layer = 61, can_shift = 1
init:        CPU KV buffer size =   162.03 MiB
llama_context: KV self size  =  162.03 MiB, K (q8_0):   81.02 MiB, V (q8_0):   81.02 MiB
llama_context:        CPU compute buffer size =   308.00 MiB
llama_context: graph nodes  = 2264
llama_context: graph splits = 1
common_init_from_params: setting dry_penalty_last_n to ctx_size = 5120
common_init_from_params: warming up the model with an empty run - please wait ... (--no-warmup to disable)

system_info: n_threads = 16 (n_threads_batch = 16) / 32 | CPU : SSE3 = 1 | SSSE3 = 1 | AVX = 1 | AVX_VNNI = 1 | AVX2 = 1 | F16C = 1 | FMA = 1 | BMI2 = 1 | AVX512 = 1 | AVX512_VBMI = 1 | AVX512_VNNI = 1 | AVX512_BF16 = 1 | LLAMAFILE = 1 | OPENMP = 1 | AARCH64_REPACK = 1 |


main: n_kv_max = 5120, n_batch = 2048, n_ubatch = 512, flash_attn = 1, n_gpu_layers = -1, n_threads = 16, n_threads_batch = 16

|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|   512 |    128 |      0 |   25.444 |    20.12 |   25.742 |     4.97 |
|   512 |    128 |    512 |   28.640 |    17.88 |   26.082 |     4.91 |
|   512 |    128 |   1024 |   33.622 |    15.23 |   26.430 |     4.84 |
|   512 |    128 |   1536 |   39.245 |    13.05 |   27.190 |     4.71 |
|   512 |    128 |   2048 |   45.237 |    11.32 |   27.152 |     4.71 |
|   512 |    128 |   2560 |   51.249 |     9.99 |   27.521 |     4.65 |
|   512 |    128 |   3072 |   57.110 |     8.97 |   27.905 |     4.59 |
|   512 |    128 |   3584 |   62.143 |     8.24 |   28.275 |     4.53 |
|   512 |    128 |   4096 |   67.889 |     7.54 |   28.630 |     4.47 |
|   512 |    128 |   4608 |   72.920 |     7.02 |   29.034 |     4.41 |
```

## `ik_llama.cpp@baeefb47`
```
$ ./build/bin/llama-sweep-bench \
    --model /mnt/astrodata/llm/models/bartowski/THUDM_GLM-Z1-32B-0414-GGUF/THUDM_GLM-Z1-32B-0414-IQ4_XS.gguf \
    -rtr -fa -ctk q8_0 -ctv q8_0 \
    -c 5120 \
    --threads 16

llama_model_loader: loaded meta data with 37 key-value pairs and 613 tensors from /mnt/astrodata/llm/models/bartowski/THUDM_GLM-Z1-32B-0414-GGUF/THUDM_GLM-Z1-32B-0414-IQ4_XS.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = glm4
llama_model_loader: - kv   1:                               general.type str              = model
llama_model_loader: - kv   2:                               general.name str              = GLM Z1 32B 0414
llama_model_loader: - kv   3:                            general.version str              = 0414
llama_model_loader: - kv   4:                           general.basename str              = GLM-Z1
llama_model_loader: - kv   5:                         general.size_label str              = 32B
llama_model_loader: - kv   6:                            general.license str              = mit
llama_model_loader: - kv   7:                               general.tags arr[str,1]       = ["text-generation"]
llama_model_loader: - kv   8:                          general.languages arr[str,2]       = ["zh", "en"]
llama_model_loader: - kv   9:                           glm4.block_count u32              = 61
llama_model_loader: - kv  10:                        glm4.context_length u32              = 32768
llama_model_loader: - kv  11:                      glm4.embedding_length u32              = 6144
llama_model_loader: - kv  12:                   glm4.feed_forward_length u32              = 23040
llama_model_loader: - kv  13:                  glm4.attention.head_count u32              = 48
llama_model_loader: - kv  14:               glm4.attention.head_count_kv u32              = 2
llama_model_loader: - kv  15:                        glm4.rope.freq_base f32              = 10000.000000
llama_model_loader: - kv  16:      glm4.attention.layer_norm_rms_epsilon f32              = 0.000010
llama_model_loader: - kv  17:                  glm4.attention.key_length u32              = 128
llama_model_loader: - kv  18:                glm4.attention.value_length u32              = 128
llama_model_loader: - kv  19:                  glm4.rope.dimension_count u32              = 64
llama_model_loader: - kv  20:                       tokenizer.ggml.model str              = gpt2
llama_model_loader: - kv  21:                         tokenizer.ggml.pre str              = glm4
llama_model_loader: - kv  22:                      tokenizer.ggml.tokens arr[str,151552]  = ["!", "\"", "#", "$", "%", "&", "'", ...
llama_model_loader: - kv  23:                  tokenizer.ggml.token_type arr[i32,151552]  = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...
llama_model_loader: - kv  24:                      tokenizer.ggml.merges arr[str,318088]  = ["Ġ Ġ", "Ġ ĠĠĠ", "ĠĠ ĠĠ", "...
llama_model_loader: - kv  25:                tokenizer.ggml.eos_token_id u32              = 151329
llama_model_loader: - kv  26:            tokenizer.ggml.padding_token_id u32              = 151329
llama_model_loader: - kv  27:                tokenizer.ggml.eot_token_id u32              = 151336
llama_model_loader: - kv  28:            tokenizer.ggml.unknown_token_id u32              = 151329
llama_model_loader: - kv  29:                tokenizer.ggml.bos_token_id u32              = 151331
llama_model_loader: - kv  30:                    tokenizer.chat_template str              = [gMASK]<sop>{%- if tools -%}<|system|...
llama_model_loader: - kv  31:               general.quantization_version u32              = 2
llama_model_loader: - kv  32:                          general.file_type u32              = 30
llama_model_loader: - kv  33:                      quantize.imatrix.file str              = /models_out/GLM-Z1-32B-0414-GGUF/THUD...
llama_model_loader: - kv  34:                   quantize.imatrix.dataset str              = /training_dir/calibration_datav3.txt
llama_model_loader: - kv  35:             quantize.imatrix.entries_count i32              = 366
llama_model_loader: - kv  36:              quantize.imatrix.chunks_count i32              = 125
llama_model_loader: - type  f32:  245 tensors
llama_model_loader: - type q5_K:   61 tensors
llama_model_loader: - type q6_K:    1 tensors
llama_model_loader: - type iq4_xs:  306 tensors
llm_load_vocab: special tokens cache size = 14
llm_load_vocab: token to piece cache size = 0.9710 MB
llm_load_print_meta: format           = GGUF V3 (latest)
llm_load_print_meta: arch             = glm4
llm_load_print_meta: vocab type       = BPE
llm_load_print_meta: n_vocab          = 151552
llm_load_print_meta: n_merges         = 318088
llm_load_print_meta: vocab_only       = 0
llm_load_print_meta: n_ctx_train      = 32768
llm_load_print_meta: n_embd           = 6144
llm_load_print_meta: n_layer          = 61
llm_load_print_meta: n_head           = 48
llm_load_print_meta: n_head_kv        = 2
llm_load_print_meta: n_rot            = 64
llm_load_print_meta: n_swa            = 0
llm_load_print_meta: n_swa_pattern    = 1
llm_load_print_meta: n_embd_head_k    = 128
llm_load_print_meta: n_embd_head_v    = 128
llm_load_print_meta: n_gqa            = 24
llm_load_print_meta: n_embd_k_gqa     = 256
llm_load_print_meta: n_embd_v_gqa     = 256
llm_load_print_meta: f_norm_eps       = 0.0e+00
llm_load_print_meta: f_norm_rms_eps   = 1.0e-05
llm_load_print_meta: f_clamp_kqv      = 0.0e+00
llm_load_print_meta: f_max_alibi_bias = 0.0e+00
llm_load_print_meta: f_logit_scale    = 0.0e+00
llm_load_print_meta: n_ff             = 23040
llm_load_print_meta: n_expert         = 0
llm_load_print_meta: n_expert_used    = 0
llm_load_print_meta: causal attn      = 1
llm_load_print_meta: pooling type     = 0
llm_load_print_meta: rope type        = 0
llm_load_print_meta: rope scaling     = linear
llm_load_print_meta: freq_base_train  = 10000.0
llm_load_print_meta: freq_scale_train = 1
llm_load_print_meta: n_ctx_orig_yarn  = 32768
llm_load_print_meta: rope_finetuned   = unknown
llm_load_print_meta: ssm_d_conv       = 0
llm_load_print_meta: ssm_d_inner      = 0
llm_load_print_meta: ssm_d_state      = 0
llm_load_print_meta: ssm_dt_rank      = 0
llm_load_print_meta: model type       = 32B
llm_load_print_meta: model ftype      = IQ4_XS - 4.25 bpw
llm_load_print_meta: model params     = 32.566 B
llm_load_print_meta: model size       = 16.382 GiB (4.321 BPW)
llm_load_print_meta: repeating layers = 15.210 GiB (4.255 BPW, 30.704 B parameters)
llm_load_print_meta: general.name     = GLM Z1 32B 0414
llm_load_print_meta: BOS token        = 151331 '[gMASK]'
llm_load_print_meta: EOS token        = 151329 '<|endoftext|>'
llm_load_print_meta: UNK token        = 151329 '<|endoftext|>'
llm_load_print_meta: PAD token        = 151329 '<|endoftext|>'
llm_load_print_meta: LF token         = 128 'Ä'
llm_load_print_meta: EOT token        = 151336 '<|user|>'
llm_load_print_meta: max token length = 1024
llm_load_tensors: ggml ctx size =    0.28 MiB
llm_load_tensors:        CPU buffer size = 16775.23 MiB
...............................................................................................
llama_new_context_with_model: n_ctx      = 5120
llama_new_context_with_model: n_batch    = 2048
llama_new_context_with_model: n_ubatch   = 512
llama_new_context_with_model: flash_attn = 1
llama_new_context_with_model: mla_attn   = 0
llama_new_context_with_model: attn_max_b = 0
llama_new_context_with_model: fused_moe  = 0
llama_new_context_with_model: ser        = -1, 0
llama_new_context_with_model: freq_base  = 10000.0
llama_new_context_with_model: freq_scale = 1
llama_kv_cache_init:        CPU KV buffer size =   162.03 MiB
llama_new_context_with_model: KV self size  =  162.03 MiB, K (q8_0):   81.02 MiB, V (q8_0):   81.02 MiB
llama_new_context_with_model:        CPU  output buffer size =     0.58 MiB
llama_new_context_with_model:        CPU compute buffer size =   308.00 MiB
llama_new_context_with_model: graph nodes  = 1592
llama_new_context_with_model: graph splits = 1

main: n_kv_max = 5120, n_batch = 2048, n_ubatch = 512, flash_attn = 1, n_gpu_layers = -1, n_threads = 16, n_threads_batch = 16

============ Repacked 367 tensors
```
|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|   512 |    128 |      0 |    6.188 |    82.74 |   25.659 |     4.99 |
|   512 |    128 |    512 |    6.286 |    81.46 |   25.741 |     4.97 |
|   512 |    128 |   1024 |    6.383 |    80.21 |   25.814 |     4.96 |
|   512 |    128 |   1536 |    6.478 |    79.04 |   25.871 |     4.95 |
|   512 |    128 |   2048 |    6.559 |    78.06 |   25.941 |     4.93 |
|   512 |    128 |   2560 |    6.651 |    76.98 |   26.026 |     4.92 |
|   512 |    128 |   3072 |    6.734 |    76.03 |   26.051 |     4.91 |
|   512 |    128 |   3584 |    6.815 |    75.12 |   26.110 |     4.90 |
|   512 |    128 |   4096 |    6.902 |    74.18 |   26.160 |     4.89 |
|   512 |    128 |   4608 |    7.007 |    73.07 |   26.232 |     4.88 |

</details>

## my CUDA GPU test

![thud-sweep-02-GPU](https://github.com/user-attachments/assets/c9207bfb-bf41-439d-acf2-0e5e75c40890)

<details>

<summary>Logs</summary>

## `llama.cpp@558a76`
Plus github.com/ubergarm/llama.cpp `ug/port-sweep-bench` branch.
```
$ CUDA_VISIBLE_DEVICE=0 \
./build/bin/llama-sweep-bench \
    --model /mnt/astrodata/llm/models/bartowski/THUDM_GLM-Z1-32B-0414-GGUF/THUDM_GLM-Z1-32B-0414-IQ4_XS.gguf \
    -fa -ctk f16 -ctv f16 \
    -c 32768 \
    -ngl 99 \
    --threads 16

ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 1 CUDA devices:
  Device 0: NVIDIA GeForce RTX 3090 Ti, compute capability 8.6, VMM: yes
build: 5192 (e59a5f1e) with cc (GCC) 14.2.1 20250128 for x86_64-pc-linux-gnu
llama_model_load_from_file_impl: using device CUDA0 (NVIDIA GeForce RTX 3090 Ti) - 22895 MiB free
llama_model_loader: loaded meta data with 37 key-value pairs and 613 tensors from /mnt/astrodata/llm/models/bartowski/THUDM_GLM-Z1-32B-0414-GGUF/THUDM_GLM-Z1-32B-0414-IQ4_XS.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = glm4
llama_model_loader: - kv   1:                               general.type str              = model
llama_model_loader: - kv   2:                               general.name str              = GLM Z1 32B 0414
llama_model_loader: - kv   3:                            general.version str              = 0414
llama_model_loader: - kv   4:                           general.basename str              = GLM-Z1
llama_model_loader: - kv   5:                         general.size_label str              = 32B
llama_model_loader: - kv   6:                            general.license str              = mit
llama_model_loader: - kv   7:                               general.tags arr[str,1]       = ["text-generation"]
llama_model_loader: - kv   8:                          general.languages arr[str,2]       = ["zh", "en"]
llama_model_loader: - kv   9:                           glm4.block_count u32              = 61
llama_model_loader: - kv  10:                        glm4.context_length u32              = 32768
llama_model_loader: - kv  11:                      glm4.embedding_length u32              = 6144
llama_model_loader: - kv  12:                   glm4.feed_forward_length u32              = 23040
llama_model_loader: - kv  13:                  glm4.attention.head_count u32              = 48
llama_model_loader: - kv  14:               glm4.attention.head_count_kv u32              = 2
llama_model_loader: - kv  15:                        glm4.rope.freq_base f32              = 10000.000000
llama_model_loader: - kv  16:      glm4.attention.layer_norm_rms_epsilon f32              = 0.000010
llama_model_loader: - kv  17:                  glm4.attention.key_length u32              = 128
llama_model_loader: - kv  18:                glm4.attention.value_length u32              = 128
llama_model_loader: - kv  19:                  glm4.rope.dimension_count u32              = 64
llama_model_loader: - kv  20:                       tokenizer.ggml.model str              = gpt2
llama_model_loader: - kv  21:                         tokenizer.ggml.pre str              = glm4
llama_model_loader: - kv  22:                      tokenizer.ggml.tokens arr[str,151552]  = ["!", "\"", "#", "$", "%", "&", "'", ...
llama_model_loader: - kv  23:                  tokenizer.ggml.token_type arr[i32,151552]  = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...
llama_model_loader: - kv  24:                      tokenizer.ggml.merges arr[str,318088]  = ["Ġ Ġ", "Ġ ĠĠĠ", "ĠĠ ĠĠ", "...
llama_model_loader: - kv  25:                tokenizer.ggml.eos_token_id u32              = 151329
llama_model_loader: - kv  26:            tokenizer.ggml.padding_token_id u32              = 151329
llama_model_loader: - kv  27:                tokenizer.ggml.eot_token_id u32              = 151336
llama_model_loader: - kv  28:            tokenizer.ggml.unknown_token_id u32              = 151329
llama_model_loader: - kv  29:                tokenizer.ggml.bos_token_id u32              = 151331
llama_model_loader: - kv  30:                    tokenizer.chat_template str              = [gMASK]<sop>{%- if tools -%}<|system|...
llama_model_loader: - kv  31:               general.quantization_version u32              = 2
llama_model_loader: - kv  32:                          general.file_type u32              = 30
llama_model_loader: - kv  33:                      quantize.imatrix.file str              = /models_out/GLM-Z1-32B-0414-GGUF/THUD...
llama_model_loader: - kv  34:                   quantize.imatrix.dataset str              = /training_dir/calibration_datav3.txt
llama_model_loader: - kv  35:             quantize.imatrix.entries_count i32              = 366
llama_model_loader: - kv  36:              quantize.imatrix.chunks_count i32              = 125
llama_model_loader: - type  f32:  245 tensors
llama_model_loader: - type q5_K:   61 tensors
llama_model_loader: - type q6_K:    1 tensors
llama_model_loader: - type iq4_xs:  306 tensors
print_info: file format = GGUF V3 (latest)
print_info: file type   = IQ4_XS - 4.25 bpw
print_info: file size   = 16.38 GiB (4.32 BPW)
load: special_eot_id is not in special_eog_ids - the tokenizer config may be incorrect
load: special tokens cache size = 14
load: token to piece cache size = 0.9710 MB
print_info: arch             = glm4
print_info: vocab_only       = 0
print_info: n_ctx_train      = 32768
print_info: n_embd           = 6144
print_info: n_layer          = 61
print_info: n_head           = 48
print_info: n_head_kv        = 2
print_info: n_rot            = 64
print_info: n_swa            = 0
print_info: n_swa_pattern    = 1
print_info: n_embd_head_k    = 128
print_info: n_embd_head_v    = 128
print_info: n_gqa            = 24
print_info: n_embd_k_gqa     = 256
print_info: n_embd_v_gqa     = 256
print_info: f_norm_eps       = 0.0e+00
print_info: f_norm_rms_eps   = 1.0e-05
print_info: f_clamp_kqv      = 0.0e+00
print_info: f_max_alibi_bias = 0.0e+00
print_info: f_logit_scale    = 0.0e+00
print_info: f_attn_scale     = 0.0e+00
print_info: n_ff             = 23040
print_info: n_expert         = 0
print_info: n_expert_used    = 0
print_info: causal attn      = 1
print_info: pooling type     = 0
print_info: rope type        = 0
print_info: rope scaling     = linear
print_info: freq_base_train  = 10000.0
print_info: freq_scale_train = 1
print_info: n_ctx_orig_yarn  = 32768
print_info: rope_finetuned   = unknown
print_info: ssm_d_conv       = 0
print_info: ssm_d_inner      = 0
print_info: ssm_d_state      = 0
print_info: ssm_dt_rank      = 0
print_info: ssm_dt_b_c_rms   = 0
print_info: model type       = 32B
print_info: model params     = 32.57 B
print_info: general.name     = GLM Z1 32B 0414
print_info: vocab type       = BPE
print_info: n_vocab          = 151552
print_info: n_merges         = 318088
print_info: BOS token        = 151331 '[gMASK]'
print_info: EOS token        = 151329 '<|endoftext|>'
print_info: EOT token        = 151336 '<|user|>'
print_info: UNK token        = 151329 '<|endoftext|>'
print_info: PAD token        = 151329 '<|endoftext|>'
print_info: LF token         = 198 'Ċ'
print_info: EOG token        = 151329 '<|endoftext|>'
print_info: EOG token        = 151336 '<|user|>'
print_info: max token length = 1024
load_tensors: loading model tensors, this can take a while... (mmap = true)
load_tensors: offloading 61 repeating layers to GPU
load_tensors: offloading output layer to GPU
load_tensors: offloaded 62/62 layers to GPU
load_tensors:        CUDA0 model buffer size = 16303.48 MiB
load_tensors:   CPU_Mapped model buffer size =   471.75 MiB
...............................................................................................
llama_context: constructing llama_context
llama_context: n_seq_max     = 1
llama_context: n_ctx         = 32768
llama_context: n_ctx_per_seq = 32768
llama_context: n_batch       = 2048
llama_context: n_ubatch      = 512
llama_context: causal_attn   = 1
llama_context: flash_attn    = 1
llama_context: freq_base     = 10000.0
llama_context: freq_scale    = 1
llama_context:  CUDA_Host  output buffer size =     0.58 MiB
init: kv_size = 32768, offload = 1, type_k = 'f16', type_v = 'f16', n_layer = 61, can_shift = 1
init:      CUDA0 KV buffer size =  1952.00 MiB
llama_context: KV self size  = 1952.00 MiB, K (f16):  976.00 MiB, V (f16):  976.00 MiB
llama_context:      CUDA0 compute buffer size =   353.00 MiB
llama_context:  CUDA_Host compute buffer size =    76.01 MiB
llama_context: graph nodes  = 2264
llama_context: graph splits = 2
common_init_from_params: setting dry_penalty_last_n to ctx_size = 32768
common_init_from_params: warming up the model with an empty run - please wait ... (--no-warmup to disable)

system_info: n_threads = 16 (n_threads_batch = 16) / 32 | CUDA : ARCHS = 860 | USE_GRAPHS = 1 | PEER_MAX_BATCH_SIZE = 128 | CPU : SSE3 = 1 | SSSE3 = 1 | AVX = 1 | AVX_VNNI = 1 | AVX2 = 1 | F16C = 1 | FMA = 1 | BMI2 = 1 | AVX512 = 1 | AVX512_VBMI = 1 | AVX512_VNNI = 1 | AVX512_BF16 = 1 | LLAMAFILE = 1 | OPENMP = 1 | AARCH64_REPACK = 1 |


main: n_kv_max = 32768, n_batch = 2048, n_ubatch = 512, flash_attn = 1, n_gpu_layers = 99, n_threads = 16, n_threads_batch = 16

|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|   512 |    128 |      0 |    0.375 |  1365.61 |    3.339 |    38.33 |
|   512 |    128 |    512 |    0.377 |  1356.98 |    3.373 |    37.94 |
|   512 |    128 |   1024 |    0.383 |  1337.96 |    3.386 |    37.80 |
|   512 |    128 |   1536 |    0.389 |  1316.12 |    3.426 |    37.36 |
|   512 |    128 |   2048 |    0.395 |  1296.18 |    3.419 |    37.44 |
|   512 |    128 |   2560 |    0.400 |  1280.80 |    3.444 |    37.17 |
|   512 |    128 |   3072 |    0.405 |  1265.46 |    3.457 |    37.03 |
|   512 |    128 |   3584 |    0.410 |  1248.46 |    3.475 |    36.84 |
|   512 |    128 |   4096 |    0.416 |  1229.54 |    3.488 |    36.70 |
|   512 |    128 |   4608 |    0.422 |  1212.10 |    3.504 |    36.53 |
|   512 |    128 |   5120 |    0.428 |  1197.32 |    3.520 |    36.37 |
|   512 |    128 |   5632 |    0.433 |  1181.44 |    3.538 |    36.18 |
|   512 |    128 |   6144 |    0.438 |  1168.89 |    3.553 |    36.03 |
|   512 |    128 |   6656 |    0.444 |  1154.19 |    3.567 |    35.89 |
|   512 |    128 |   7168 |    0.449 |  1141.06 |    3.616 |    35.40 |
|   512 |    128 |   7680 |    0.454 |  1126.73 |    3.625 |    35.31 |
|   512 |    128 |   8192 |    0.460 |  1114.03 |    3.755 |    34.09 |
|   512 |    128 |   8704 |    0.466 |  1098.92 |    3.668 |    34.90 |
|   512 |    128 |   9216 |    0.471 |  1088.14 |    3.668 |    34.90 |
|   512 |    128 |   9728 |    0.476 |  1076.44 |    3.671 |    34.86 |
|   512 |    128 |  10240 |    0.482 |  1062.75 |    3.676 |    34.82 |
|   512 |    128 |  10752 |    0.487 |  1051.19 |    3.687 |    34.72 |
|   512 |    128 |  11264 |    0.491 |  1042.43 |    3.692 |    34.67 |
|   512 |    128 |  11776 |    0.505 |  1013.60 |    3.720 |    34.41 |
|   512 |    128 |  12288 |    0.504 |  1014.87 |    3.784 |    33.82 |
|   512 |    128 |  12800 |    0.539 |   950.02 |    3.833 |    33.39 |
|   512 |    128 |  13312 |    0.516 |   991.65 |    3.909 |    32.74 |
|   512 |    128 |  13824 |    0.522 |   981.09 |    3.873 |    33.05 |
|   512 |    128 |  14336 |    0.539 |   949.82 |    4.010 |    31.92 |
|   512 |    128 |  14848 |    0.569 |   899.85 |    3.995 |    32.04 |
|   512 |    128 |  15360 |    0.534 |   958.25 |    3.950 |    32.40 |
|   512 |    128 |  15872 |    0.539 |   949.11 |    3.824 |    33.47 |
|   512 |    128 |  16384 |    0.547 |   936.62 |    3.832 |    33.41 |
|   512 |    128 |  16896 |    0.555 |   922.31 |    3.827 |    33.45 |
|   512 |    128 |  17408 |    0.559 |   915.85 |    3.858 |    33.18 |
|   512 |    128 |  17920 |    0.561 |   913.36 |    3.847 |    33.27 |
|   512 |    128 |  18432 |    0.567 |   902.43 |    3.863 |    33.13 |
|   512 |    128 |  18944 |    0.571 |   895.97 |    3.864 |    33.12 |
|   512 |    128 |  19456 |    0.575 |   891.16 |    3.899 |    32.83 |
|   512 |    128 |  19968 |    0.580 |   882.92 |    3.857 |    33.18 |
|   512 |    128 |  20480 |    0.585 |   875.26 |    3.863 |    33.14 |
|   512 |    128 |  20992 |    0.590 |   867.25 |    3.871 |    33.07 |
|   512 |    128 |  21504 |    0.595 |   860.29 |    3.917 |    32.68 |
|   512 |    128 |  22016 |    0.600 |   853.53 |    3.921 |    32.64 |
|   512 |    128 |  22528 |    0.605 |   846.56 |    3.927 |    32.60 |
|   512 |    128 |  23040 |    0.609 |   840.50 |    3.931 |    32.56 |
|   512 |    128 |  23552 |    0.615 |   832.38 |    3.941 |    32.48 |
|   512 |    128 |  24064 |    0.620 |   825.45 |    3.945 |    32.44 |
|   512 |    128 |  24576 |    0.626 |   818.16 |    3.948 |    32.42 |
|   512 |    128 |  25088 |    0.630 |   812.67 |    3.956 |    32.36 |
|   512 |    128 |  25600 |    0.637 |   804.33 |    3.962 |    32.31 |
|   512 |    128 |  26112 |    0.640 |   800.21 |    3.967 |    32.26 |
|   512 |    128 |  26624 |    0.646 |   792.11 |    3.974 |    32.21 |
|   512 |    128 |  27136 |    0.650 |   787.81 |    3.984 |    32.13 |
|   512 |    128 |  27648 |    0.656 |   781.05 |    3.989 |    32.09 |
|   512 |    128 |  28160 |    0.663 |   771.82 |    4.086 |    31.33 |
|   512 |    128 |  28672 |    0.665 |   769.50 |    4.039 |    31.69 |
|   512 |    128 |  29184 |    0.671 |   763.01 |    4.043 |    31.66 |
|   512 |    128 |  29696 |    0.676 |   757.73 |    4.051 |    31.60 |
|   512 |    128 |  30208 |    0.680 |   752.57 |    4.054 |    31.58 |
|   512 |    128 |  30720 |    0.686 |   746.34 |    4.065 |    31.49 |
|   512 |    128 |  31232 |    0.690 |   741.72 |    4.067 |    31.47 |
|   512 |    128 |  31744 |    0.697 |   734.83 |    4.074 |    31.42 |
|   512 |    128 |  32256 |    0.701 |   730.49 |    4.083 |    31.35 |
```

## `ik_llama.cpp@baeefb47`
```
CUDA_VISIBLE_DEVICE=0 \
./build/bin/llama-sweep-bench \
    --model /mnt/astrodata/llm/models/bartowski/THUDM_GLM-Z1-32B-0414-GGUF/THUDM_GLM-Z1-32B-0414-IQ4_XS.gguf \
    -fa -ctk f16 -ctv f16 \
    -c 32768 \
    -ngl 99 \
    --threads 16

llama_model_loader: loaded meta data with 37 key-value pairs and 613 tensors from /mnt/astrodata/llm/models/bartowski/THUDM_GLM-Z1-32B-0414-GGUF/THUDM_GLM-Z1-32B-0414-IQ4_XS.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = glm4
llama_model_loader: - kv   1:                               general.type str              = model
llama_model_loader: - kv   2:                               general.name str              = GLM Z1 32B 0414
llama_model_loader: - kv   3:                            general.version str              = 0414
llama_model_loader: - kv   4:                           general.basename str              = GLM-Z1
llama_model_loader: - kv   5:                         general.size_label str              = 32B
llama_model_loader: - kv   6:                            general.license str              = mit
llama_model_loader: - kv   7:                               general.tags arr[str,1]       = ["text-generation"]
llama_model_loader: - kv   8:                          general.languages arr[str,2]       = ["zh", "en"]
llama_model_loader: - kv   9:                           glm4.block_count u32              = 61
llama_model_loader: - kv  10:                        glm4.context_length u32              = 32768
llama_model_loader: - kv  11:                      glm4.embedding_length u32              = 6144
llama_model_loader: - kv  12:                   glm4.feed_forward_length u32              = 23040
llama_model_loader: - kv  13:                  glm4.attention.head_count u32              = 48
llama_model_loader: - kv  14:               glm4.attention.head_count_kv u32              = 2
llama_model_loader: - kv  15:                        glm4.rope.freq_base f32              = 10000.000000
llama_model_loader: - kv  16:      glm4.attention.layer_norm_rms_epsilon f32              = 0.000010
llama_model_loader: - kv  17:                  glm4.attention.key_length u32              = 128
llama_model_loader: - kv  18:                glm4.attention.value_length u32              = 128
llama_model_loader: - kv  19:                  glm4.rope.dimension_count u32              = 64
llama_model_loader: - kv  20:                       tokenizer.ggml.model str              = gpt2
llama_model_loader: - kv  21:                         tokenizer.ggml.pre str              = glm4
llama_model_loader: - kv  22:                      tokenizer.ggml.tokens arr[str,151552]  = ["!", "\"", "#", "$", "%", "&", "'", ...
llama_model_loader: - kv  23:                  tokenizer.ggml.token_type arr[i32,151552]  = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...
llama_model_loader: - kv  24:                      tokenizer.ggml.merges arr[str,318088]  = ["Ġ Ġ", "Ġ ĠĠĠ", "ĠĠ ĠĠ", "...
llama_model_loader: - kv  25:                tokenizer.ggml.eos_token_id u32              = 151329
llama_model_loader: - kv  26:            tokenizer.ggml.padding_token_id u32              = 151329
llama_model_loader: - kv  27:                tokenizer.ggml.eot_token_id u32              = 151336
llama_model_loader: - kv  28:            tokenizer.ggml.unknown_token_id u32              = 151329
llama_model_loader: - kv  29:                tokenizer.ggml.bos_token_id u32              = 151331
llama_model_loader: - kv  30:                    tokenizer.chat_template str              = [gMASK]<sop>{%- if tools -%}<|system|...
llama_model_loader: - kv  31:               general.quantization_version u32              = 2
llama_model_loader: - kv  32:                          general.file_type u32              = 30
llama_model_loader: - kv  33:                      quantize.imatrix.file str              = /models_out/GLM-Z1-32B-0414-GGUF/THUD...
llama_model_loader: - kv  34:                   quantize.imatrix.dataset str              = /training_dir/calibration_datav3.txt
llama_model_loader: - kv  35:             quantize.imatrix.entries_count i32              = 366
llama_model_loader: - kv  36:              quantize.imatrix.chunks_count i32              = 125
llama_model_loader: - type  f32:  245 tensors
llama_model_loader: - type q5_K:   61 tensors
llama_model_loader: - type q6_K:    1 tensors
llama_model_loader: - type iq4_xs:  306 tensors
llm_load_vocab: special tokens cache size = 14
llm_load_vocab: token to piece cache size = 0.9710 MB
llm_load_print_meta: format           = GGUF V3 (latest)
llm_load_print_meta: arch             = glm4
llm_load_print_meta: vocab type       = BPE
llm_load_print_meta: n_vocab          = 151552
llm_load_print_meta: n_merges         = 318088
llm_load_print_meta: vocab_only       = 0
llm_load_print_meta: n_ctx_train      = 32768
llm_load_print_meta: n_embd           = 6144
llm_load_print_meta: n_layer          = 61
llm_load_print_meta: n_head           = 48
llm_load_print_meta: n_head_kv        = 2
llm_load_print_meta: n_rot            = 64
llm_load_print_meta: n_swa            = 0
llm_load_print_meta: n_swa_pattern    = 1
llm_load_print_meta: n_embd_head_k    = 128
llm_load_print_meta: n_embd_head_v    = 128
llm_load_print_meta: n_gqa            = 24
llm_load_print_meta: n_embd_k_gqa     = 256
llm_load_print_meta: n_embd_v_gqa     = 256
llm_load_print_meta: f_norm_eps       = 0.0e+00
llm_load_print_meta: f_norm_rms_eps   = 1.0e-05
llm_load_print_meta: f_clamp_kqv      = 0.0e+00
llm_load_print_meta: f_max_alibi_bias = 0.0e+00
llm_load_print_meta: f_logit_scale    = 0.0e+00
llm_load_print_meta: n_ff             = 23040
llm_load_print_meta: n_expert         = 0
llm_load_print_meta: n_expert_used    = 0
llm_load_print_meta: causal attn      = 1
llm_load_print_meta: pooling type     = 0
llm_load_print_meta: rope type        = 0
llm_load_print_meta: rope scaling     = linear
llm_load_print_meta: freq_base_train  = 10000.0
llm_load_print_meta: freq_scale_train = 1
llm_load_print_meta: n_ctx_orig_yarn  = 32768
llm_load_print_meta: rope_finetuned   = unknown
llm_load_print_meta: ssm_d_conv       = 0
llm_load_print_meta: ssm_d_inner      = 0
llm_load_print_meta: ssm_d_state      = 0
llm_load_print_meta: ssm_dt_rank      = 0
llm_load_print_meta: model type       = 32B
llm_load_print_meta: model ftype      = IQ4_XS - 4.25 bpw
llm_load_print_meta: model params     = 32.566 B
llm_load_print_meta: model size       = 16.382 GiB (4.321 BPW)
llm_load_print_meta: repeating layers = 15.210 GiB (4.255 BPW, 30.704 B parameters)
llm_load_print_meta: general.name     = GLM Z1 32B 0414
llm_load_print_meta: BOS token        = 151331 '[gMASK]'
llm_load_print_meta: EOS token        = 151329 '<|endoftext|>'
llm_load_print_meta: UNK token        = 151329 '<|endoftext|>'
llm_load_print_meta: PAD token        = 151329 '<|endoftext|>'
llm_load_print_meta: LF token         = 128 'Ä'
llm_load_print_meta: EOT token        = 151336 '<|user|>'
llm_load_print_meta: max token length = 1024
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 1 CUDA devices:
  Device 0: NVIDIA GeForce RTX 3090 Ti, compute capability 8.6, VMM: yes
llm_load_tensors: ggml ctx size =    0.56 MiB
llm_load_tensors: offloading 61 repeating layers to GPU
llm_load_tensors: offloading non-repeating layers to GPU
llm_load_tensors: offloaded 62/62 layers to GPU
llm_load_tensors:        CPU buffer size =   471.75 MiB
llm_load_tensors:      CUDA0 buffer size = 16303.48 MiB
...............................................................................................
llama_new_context_with_model: n_ctx      = 32768
llama_new_context_with_model: n_batch    = 2048
llama_new_context_with_model: n_ubatch   = 512
llama_new_context_with_model: flash_attn = 1
llama_new_context_with_model: mla_attn   = 0
llama_new_context_with_model: attn_max_b = 0
llama_new_context_with_model: fused_moe  = 0
llama_new_context_with_model: ser        = -1, 0
llama_new_context_with_model: freq_base  = 10000.0
llama_new_context_with_model: freq_scale = 1
llama_kv_cache_init:      CUDA0 KV buffer size =  1952.00 MiB
llama_new_context_with_model: KV self size  = 1952.00 MiB, K (f16):  976.00 MiB, V (f16):  976.00 MiB
llama_new_context_with_model:  CUDA_Host  output buffer size =     0.58 MiB
llama_new_context_with_model:      CUDA0 compute buffer size =   308.00 MiB
llama_new_context_with_model:  CUDA_Host compute buffer size =    76.01 MiB
llama_new_context_with_model: graph nodes  = 1592
llama_new_context_with_model: graph splits = 2

main: n_kv_max = 32768, n_batch = 2048, n_ubatch = 512, flash_attn = 1, n_gpu_layers = 99, n_threads = 16, n_threads_batch = 16
```

|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|   512 |    128 |      0 |    0.333 |  1538.22 |    3.212 |    39.85 |
|   512 |    128 |    512 |    0.343 |  1492.07 |    3.276 |    39.07 |
|   512 |    128 |   1024 |    0.354 |  1447.26 |    3.339 |    38.34 |
|   512 |    128 |   1536 |    0.363 |  1410.13 |    3.398 |    37.67 |
|   512 |    128 |   2048 |    0.373 |  1373.33 |    3.456 |    37.03 |
|   512 |    128 |   2560 |    0.384 |  1332.96 |    3.523 |    36.33 |
|   512 |    128 |   3072 |    0.394 |  1298.19 |    3.583 |    35.72 |
|   512 |    128 |   3584 |    0.405 |  1265.49 |    3.640 |    35.17 |
|   512 |    128 |   4096 |    0.415 |  1233.24 |    3.697 |    34.62 |
|   512 |    128 |   4608 |    0.426 |  1202.42 |    3.754 |    34.10 |
|   512 |    128 |   5120 |    0.436 |  1174.72 |    3.820 |    33.51 |
|   512 |    128 |   5632 |    0.446 |  1147.10 |    3.876 |    33.02 |
|   512 |    128 |   6144 |    0.457 |  1119.58 |    3.931 |    32.56 |
|   512 |    128 |   6656 |    0.468 |  1094.31 |    3.987 |    32.11 |
|   512 |    128 |   7168 |    0.477 |  1073.21 |    4.042 |    31.67 |
|   512 |    128 |   7680 |    0.487 |  1050.31 |    4.098 |    31.23 |
|   512 |    128 |   8192 |    0.500 |  1023.63 |    4.154 |    30.82 |
|   512 |    128 |   8704 |    0.511 |  1002.28 |    4.222 |    30.32 |
|   512 |    128 |   9216 |    0.521 |   982.66 |    4.278 |    29.92 |
|   512 |    128 |   9728 |    0.531 |   963.76 |    4.335 |    29.53 |
|   512 |    128 |  10240 |    0.541 |   946.41 |    4.391 |    29.15 |
|   512 |    128 |  10752 |    0.551 |   928.41 |    4.445 |    28.80 |
|   512 |    128 |  11264 |    0.561 |   912.12 |    4.502 |    28.43 |
|   512 |    128 |  11776 |    0.570 |   897.92 |    4.555 |    28.10 |
|   512 |    128 |  12288 |    0.579 |   883.61 |    4.612 |    27.76 |
|   512 |    128 |  12800 |    0.590 |   867.46 |    4.667 |    27.43 |
|   512 |    128 |  13312 |    0.601 |   852.49 |    4.720 |    27.12 |
|   512 |    128 |  13824 |    0.610 |   839.39 |    4.776 |    26.80 |
|   512 |    128 |  14336 |    0.621 |   824.14 |    4.828 |    26.51 |
|   512 |    128 |  14848 |    0.631 |   811.64 |    4.885 |    26.20 |
|   512 |    128 |  15360 |    0.642 |   797.72 |    4.934 |    25.94 |
|   512 |    128 |  15872 |    0.652 |   785.82 |    4.989 |    25.66 |
|   512 |    128 |  16384 |    0.662 |   773.33 |    5.043 |    25.38 |
|   512 |    128 |  16896 |    0.672 |   762.26 |    5.099 |    25.10 |
|   512 |    128 |  17408 |    0.681 |   751.45 |    5.153 |    24.84 |
|   512 |    128 |  17920 |    0.692 |   740.14 |    5.206 |    24.59 |
|   512 |    128 |  18432 |    0.702 |   729.58 |    5.260 |    24.33 |
|   512 |    128 |  18944 |    0.711 |   719.91 |    5.313 |    24.09 |
|   512 |    128 |  19456 |    0.720 |   710.66 |    5.371 |    23.83 |
|   512 |    128 |  19968 |    0.731 |   700.28 |    5.423 |    23.60 |
|   512 |    128 |  20480 |    0.740 |   691.88 |    5.482 |    23.35 |
|   512 |    128 |  20992 |    0.750 |   682.74 |    5.536 |    23.12 |
|   512 |    128 |  21504 |    0.761 |   673.13 |    5.591 |    22.89 |
|   512 |    128 |  22016 |    0.770 |   664.83 |    5.641 |    22.69 |
|   512 |    128 |  22528 |    0.781 |   655.60 |    5.699 |    22.46 |
|   512 |    128 |  23040 |    0.790 |   648.12 |    5.749 |    22.26 |
|   512 |    128 |  23552 |    0.800 |   639.76 |    5.804 |    22.05 |
|   512 |    128 |  24064 |    0.811 |   631.16 |    5.860 |    21.84 |
|   512 |    128 |  24576 |    0.820 |   624.55 |    5.915 |    21.64 |
|   512 |    128 |  25088 |    0.830 |   616.63 |    5.970 |    21.44 |
|   512 |    128 |  25600 |    0.840 |   609.34 |    6.028 |    21.24 |
|   512 |    128 |  26112 |    0.850 |   602.01 |    6.084 |    21.04 |
|   512 |    128 |  26624 |    0.860 |   595.01 |    6.139 |    20.85 |
|   512 |    128 |  27136 |    0.870 |   588.30 |    6.197 |    20.66 |
|   512 |    128 |  27648 |    0.880 |   582.14 |    6.251 |    20.48 |
|   512 |    128 |  28160 |    0.890 |   575.38 |    6.308 |    20.29 |
|   512 |    128 |  28672 |    0.900 |   569.14 |    6.361 |    20.12 |
|   512 |    128 |  29184 |    0.912 |   561.64 |    6.416 |    19.95 |
|   512 |    128 |  29696 |    0.920 |   556.31 |    6.472 |    19.78 |
|   512 |    128 |  30208 |    0.930 |   550.65 |    6.527 |    19.61 |
|   512 |    128 |  30720 |    0.940 |   544.53 |    6.586 |    19.44 |
|   512 |    128 |  31232 |    0.951 |   538.41 |    6.633 |    19.30 |
|   512 |    128 |  31744 |    0.961 |   532.89 |    6.693 |    19.12 |
|   512 |    128 |  32256 |    0.970 |   527.77 |    6.744 |    18.98 |


</details>

I didn't yet try comparing running with non-flash-attention.

---

👤 **saood06** commented the **2025-04-27** at **08:48:11**:<br>

> > This model is quite efficient, I just ran it with 128k context and only using 21194MiB VRAM ??
> 
> Yes, it has a very high GQA factor of 24

This caught my eye, and was glad they had a prior work dedicated to long context training of LLMs, that they referenced in the GQA part of their technical report, [LongAlign: A Recipe for Long Context Alignment of Large Language Models](https://arxiv.org/abs/2401.18058)

---

👤 **saood06** commented the **2025-05-08** at **22:44:40**:<br>

I found [this](https://adamniederer.com/blog/llm-context-benchmarks.html) where someone uses NoLiMa to test the long context performance and they did notice lower performance (which I believe is because of the very high GQA factor).
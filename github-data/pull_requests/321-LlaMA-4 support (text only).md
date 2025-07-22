### 🔀 [#321](https://github.com/ikawrakow/ik_llama.cpp/pull/321) - LlaMA-4 support (text only)

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-04-09 |
| **Updated** | 2025-04-11 |

---

#### Description

It seems the initial reactions to LlaMA-4 are mostly negative. Nevertheless, quantized LlaMA-Scout is something I can run on one of my systems, so here it is.

Derived from [PR 12791](https://github.com/ggml-org/llama.cpp/pull/12791) in mainline. But the code bases have diverged so much by now that it did take some effort to port the PR.

As with Gemma-3, I did not add the necessary modifications to `convert_hf_to_gguf.py`, so mainline is required to generate the model GGUF.

Did a quick test with a `Q6_K` model (no imatrix yet, so wanted to use more bits to not worry about quantization effects). Ryzen-5975WX CPU, RTX-4080 GPU, using
```
-ot exps=CPU -rtr -fmoe -t 32 -ngl 100
```
I got 221 t/s in the perplexity run, and 10.5 t/s for 128 tokens asking the standard question about the meaning of life. This is not bad at all. 

As mentioned in [PR 12791](https://github.com/ggml-org/llama.cpp/pull/12791), the model fails the ultimate AGI test:
```
> How many r's are there in strawberry?
There are 2 R's in the word "strawberry".
```

Closes #314

---

#### 💬 Conversation

👤 **ikawrakow** commented the **2025-04-09** at **15:02:02**:<br>

So, using a single active expert as prescribed by the model parameters, I get
```
PPL(Q8_0, n_ctx = 512) = 9.0644
```
Activating 2 experts using `--override-kv "llama4.expert_used_count=int:2"` I get
```
PPL(Q8_0, n_ctx = 512) = 8.7030
```

It is of course slower (133 t/s vs 211 t/s with the setup described above), but it is kind of strange that 2 experts produce a lower PPL. This wasn't the case for Mixtral8x7B where 3 experts were worse than 2 (unless one was using a very low bpw quantization).

---

👤 **ikawrakow** commented the **2025-04-10** at **05:59:25**:<br>

Here some quantization experiments with LlaMA-4-Scout

* UD-Q2_K_XL.gguf - downloaded from [Huggingface](https://huggingface.co/unsloth/Llama-4-Scout-17B-16E-Instruct-GGUF): `PPL(n_ctx = 512) = 9.6535`
* Same quantization mix as UD-Q2_K_XL.gguf, but quantized with `ik_llama.cpp`<sup>1</sup>: `PPL(n_ctx = 512) = 9.5668`
* Replace `q4_K` with `iq4_K` for `ffn_down_exps` tensors: `PPL(n_ctx = 512) = 9.4895`
* Strangely enough, replacing `q4_K` with `iq4_K` in the attention tensors leads to higher PPL

___
<sup>1</sup> Unsloth's `Q2_K_XL` mix is obtained without any code changes using
```
./bin/llama-quantize --imatrix $imatrix --custom-q "ffn_gate_shexp=q4_K,ffn_up_shexp=q4_K,ffn_down_shexp=q6_K,attn=q4_K,token_embd.weight=q4_K,output.weight=q6_K,blk\.[0-5]\.ffn_down_exps=q4_K,ffn_down_exps=q3_K,ffn_up_exps=q2_K,ffn_gate_exps=q2_K" $model $output_file q2_K
```

---

👤 **saood06** commented the **2025-04-10** at **06:13:30**:<br>

> Strangely enough, replacing `q4_K` with `iq4_K` in the attention tensors leads to higher PPL

Do you think this could affect other architectures?

---

👤 **ikawrakow** commented the **2025-04-10** at **06:18:31**:<br>

> Do you think this could affect other architectures?

I have noticed in the past that `iq4_k/iq5_k/iq6_k` for the attention tensors does not have a clear advantage compared to `q4_K/q5_K/q6_K`. They are much better for the FFN portion and that's where the quality gains come from. But this is the first time when it became worse. So, in your case, if you are looking to optimize performance (and have time/energy to experiment), you can try replacing `iq4_k` with `q4_K` in the attention tensors as this will improve inference speed.

---

👤 **ikawrakow** commented the **2025-04-10** at **06:20:02**:<br>

Oh, for token embeddings I had a few cases where it was better to use the corresponding k-quant instead of the `iqk` quant.

---

👤 **saood06** commented the **2025-04-10** at **06:46:32**:<br>

> I have noticed in the past that `iq4_k/iq5_k/iq6_k` for the attention tensors does not have a clear advantage compared to `q4_K/q5_K/q6_K`. They are much better for the FFN portion and that's where the quality gains come from. But this is the first time when it became worse. So, in your case, if you are looking to optimize performance (and have time/energy to experiment), you can try replacing `iq4_k` with `q4_K` in the attention tensors as this will improve inference speed.

>Oh, for token embeddings I had a few cases where it was better to use the corresponding k-quant instead of the iqk quant.

Interesting to hear. I will take all this into account next time I make quants.

---

👤 **ikawrakow** commented the **2025-04-10** at **06:57:24**:<br>

> Have you tried even higher numbers? Does it peak at 2 experts?

Just tried. Did not run `Wikitext2` to completion, but after 172 chunks PPL is 0.1 higher than 2 experts, so it is very unlikely it will be better at the end. Still better than a single expert, but 2 experts seems to be the sweet spot (at the expense of a hit of performance).

---

👤 **ikawrakow** commented the **2025-04-10** at **07:05:15**:<br>

This seems solid enough, merging it.

---

👤 **saood06** commented the **2025-04-10** at **08:20:34**:<br>

> Just tried. Did not run `Wikitext2` to completion, but after 172 chunks PPL with 3 experts is 0.1 higher than 2 experts, so it is very unlikely it will be better at the end. Still better than a single expert, but 2 experts seems to be the sweet spot (at the expense of a hit in performance).

If I ever try Maverick will see if it is replicable there.

---

👤 **ikawrakow** commented the **2025-04-10** at **15:11:51**:<br>

So, L4-Scout seems to quantize pretty well.

### 4-bit (IQ4_KS)

* `PPL = 9.0554` (better than `Q8_0`, so no need to go beyond that)
* Quantized model size: 54.003 GiB
* Recipe
```
./bin/llama-quantize --imatrix l4_scout_imat_512.out --custom-q "ffn_gate_shexp=iq4_ks,ffn_up_shexp=iq4_ks,ffn_down_shexp=iq5_k,attn=iq4_ks,token_embd.weight=q4_K,output.weight=q6_K,ffn_.*_exps=iq4_ks" ../../iquants/models/l4_109B/Llama4-Scout-16x17B-BF16.gguf junk1.bin iq4_ks
```
(so basically everything with `IQ4_KS`, except for `ffn_down_shexp` (`IQ5_K`), `token_embd` (`Q4_K`) and `output.weight` (`Q6_K`)) gives a Wikitext2 PPL of `9.0554` (better than `Q8_0`).

### Beating Unsloth's UD-Q2_K_XL

* `PPL = 9.4736` vs theirs `PPL = 9.6535`
* Model size: 39.090 GiB vs Unsloth's 39.654 GiB
* Recipe
```
./bin/llama-quantize --imatrix l4_scout_imat_512.out --custom-q "ffn_gate_shexp=iq4_ks,ffn_up_shexp=iq4_ks,ffn_down_shexp=iq5_k,attn=iq4_ks,token_embd.weight=q4_K,output.weight=q6_K,blk\.[0-5]\.ffn_down_exps=iq4_ks,ffn_down_exps=q3_K,ffn_up_exps=q2_K,ffn_gate_exps=q2_K" ../../iquants/models/l4_109B/Llama4-Scout-16x17B-BF16.gguf junk1.bin q2_K
```

### Beating Unsloth's UD-IQ2_XXS

* `PPL = 10.1506` vs theirs `PPL = 10.3454`
* Model size: 34.871 GiB vs theirs 35.904 GiB
* Recipe:
```
./bin/llama-quantize --imatrix l4_scout_imat_512.out --custom-q "ffn_gate_shexp=iq4_ks,ffn_up_shexp=iq4_ks,ffn_down_shexp=iq5_k,attn=iq4_ks,token_embd.weight=q4_K,output.weight=q6_K,blk\.[0-5]\.ffn_down_exps=iq4_ks,ffn_down_exps=q3_K,ffn_up_exps=iq1_s,ffn_gate_exps=iq1_s" ../../iquants/models/l4_109B/Llama4-Scout-16x17B-BF16.gguf junk1.bin iq1_s
```

### Beating Unsloth's UD-IQ1_S

* `PPL = 10.9640` vs theirs `PPL = 11.0173`
* Model size: 31.121 GiB vs theirs 31.510 GiB 
* Recipe:
```
./bin/llama-quantize --imatrix l4_scout_imat_512.out --custom-q "ffn_gate_shexp=iq4_ks,ffn_up_shexp=iq4_ks,ffn_down_shexp=iq5_k,attn=iq4_ks,token_embd.weight=q4_K,output.weight=q6_K,blk\.[0-5]\.ffn_down_exps=iq4_ks,ffn_down_exps=iq3_k,ffn_up_exps=iq1_s,ffn_gate_exps=iq1_s" ../../iquants/models/l4_109B/Llama4-Scout-16x17B-BF16.gguf junk1.bin iq1_s
```

---

👤 **ikawrakow** commented the **2025-04-11** at **16:01:10**:<br>

Here another recipe for `iq3_xxs`:
```
./bin/llama-quantize --imatrix l4_scout_imat_512.out --custom-q "ffn_gate_shexp=iq4_ks,ffn_up_shexp=iq4_ks,ffn_down_shexp=iq5_k,attn=iq4_ks,token_embd.weight=q4_K,output.weight=q6_K,ffn_down_exps=iq4_ks,ffn_.*_exps=iq3_xxs" ../../iquants/models/l4_109B/Llama4-Scout-16x17B-BF16.gguf junk1.bin iq3_xxs
```

The model ends up being 45.05 GiB (48.38 GB), so qualifies for this "under 50 GB" [shoot-out](https://huggingface.co/blog/bartowski/llama4-scout-off). Final Wiki2 PPL is `9.2462` (so just 2% higher than `Q8_0`). PPL after 300 chunks (as used in [the shoot-out](https://huggingface.co/blog/bartowski/llama4-scout-off)) is `8.8937`.  If I then go through the trouble of running `llama-perplexity` with the `--kl-divergence` option, I get this
```
====== Perplexity statistics ======
Mean PPL(Q)                   :   8.894160 ±   0.099641
Cor(ln(PPL(Q)), ln(PPL(base))):  97.61%
Mean ln(PPL(Q)/PPL(base))     :   0.030502 ±   0.002438

====== KL divergence statistics ======
Mean    KLD:   0.106186 ±   0.001075
99.0%   KLD:   1.098310
Median  KLD:   0.033228

====== Token probability statistics ======
Mean    Δp: -0.695 ± 0.033 %
90.0%   Δp:  5.221%
Median  Δp: -0.002%

RMS Δp    :  9.177 ± 0.076 %
Same top p: 87.280 ± 0.120 %
```
So, a different league than the shoot-out models.
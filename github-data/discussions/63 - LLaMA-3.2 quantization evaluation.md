### 🗣️ [#63](https://github.com/ikawrakow/ik_llama.cpp/discussions/63) - LLaMA-3.2 quantization evaluation

| **Author** | `ikawrakow` |
| :--- | :--- |
| **Created** | 2024-09-26 |
| **Updated** | 2024-09-26 |

---

#### Description

LLaMA-3.2 is out. `llama.cpp` does not yet support the vision models, so this post focuses on the 1B ad 3B text models that could be very handy for local usage on low-end devices. The models are small enough even with full precision (`bf16`) but I think it is still interesting to look at quantization as token generation is significantly faster with quantized models.

To reproduce the results reported here
1. Clone my validation dataset repository
```
git clone git@hf.co:datasets/ikawrakow/validation-datasets-for-llama.cpp
cd validation-datasets-for-llama.cpp
gunzip wiki.test.raw.gz
gunzip wiki.train.raw.gz
```

2. Get one or more LLaMA-3.2 models. E.g.
```
git clone git@hf.co:meta-llama/Llama-3.2-3B
```

3. Convert to GGUF. E.g.
```
python3 convert_hf_to_gguf.py --outtype bf16 Llama-3.2-3B/
```

4. Create imatrix data. E.g.
```
./bin/llama-imatrix -m Llama-3.2-3B/Llama-3.2-3B-BF16.gguf -f validation-datasets-for-llama.cpp/wiki.train.raw --chunks 1000 -o l32_imatrix_c512.out
```

5. Quantize. E.g.
```
./bin/llama-quantize --imatrix l32_imatrix_c512.out Llama-3.2-3B/Llama-3.2-3B-BF16.gguf iq4k.gguf iq4_k
```
6. Compute perplexity
```
./bin/llama-perplexity -m iq4k.gguf -f validation-datasets-for-llama.cpp/wiki.test.raw -t 1 -ngl 100
```

7. Compute HellaSwag
```
./bin/llama-perplexity -m iq4k.gguf -bf validation-datasets-for-llama.cpp/hellaswag-validation.bin --multiple-choice -t 1 -ngl 100 -c 2048
```

8. Compute MMLU
```
./bin/llama-perplexity -m iq4k.gguf -bf validation-datasets-for-llama.cpp/mmlu-test.bin --multiple-choice -t 1 -ngl 100 -c 2048
```

### Perplexity

Perplexity (`PPL` in what follows) is not the best measure to compare *different* models, but it is extremely useful when comparing a quantized version of a model to the *same* full precision model. In the graphs below I use the quantization error defined as
```
quantization error = PPL(Q)/PPL(bf16) - 1
```
where `PPL(Q)` is the perplexity of quantization `Q` and `PPL(bf16)` is the perplexity of the full model (the 3.2 models are released as `bf16`, so I use `bf16` throughout as `bf16` support has been added here in PR #39, #40, #41, #56). 

The following graph shows quantization error of LLaMA-3.2-3B as a function of bits-per-weight (bpw) for (almost) all quantization types supported here. Note that this is the effective bpw that includes the `token_embedding.weight` tensor, which is quantized with more bits (typically `Q6_K`), and this has a significant impact on the overall bpw balance as this tensor represents a significant fraction of the overall model size. The y-axis is logarithmic, so differences can be quite large even if data points look relatively close. The cyan circles are for the new quants `IQ2_K, IQ3_K, IQ4_K, IQ5_K` and `IQ6_K` that are not available in mainline `llama.cpp`. The black symbols are for i-quants, the red for k-quants, and the blue symbols are legacy quants (`Q4_0, Q4_1, Q5_0`, Q5_1`). 

![l32_ppl_3B](https://github.com/user-attachments/assets/602e5623-6a90-4c74-82ef-26dca80c4a86)

The next graph shows results for LLaMA-3.2-3B-Instruct. The results are qualitatively very similar to the base model, with the quantization error being slightly lower compared to the base model.
![l32_it_ppl_3B](https://github.com/user-attachments/assets/91929ff8-f456-4d37-bce1-0105bfc79d7c)

My conclusion from these two graphs are
1. Going below 3 bpw with these models is not useful - the quantization error becomes too large. This is similar to the 3.1 LlaMA models
2. The new iqk-quants `IQ4_K` and `IQ5_K` are significantly better than k- or legacy quants in this bpw range
3. Legacy quants are mostly useless as it is so often the case

The next graph is for the base LLaMA-3.2-1B model

![l32_ppl_1B](https://github.com/user-attachments/assets/3918f73f-f7d4-4a66-80df-16c6dc9d5fcf)

Here the quantization error is significantly larger, going below 2% only for 5+ bpw.  At about 4.95 bpw `IQ4_K` has a quantization error of 3%, `Q4_K_S` is at 4.3%, and `Q4_0` at 12.5% (!), nearly the same as `IQ3_K` at 3.68 bpw.

### HellaSwag

The HellaSwag 0-shot score of 74.34 for the 3B base model is surprisingly high for a model of this size. But here we are more interested in looking at the impact of quantization, so I'll focus on that. The following graph shows
```
HellaSwag(bf16) - HellaSwag(Q)
```
for LLaMA-3.2-3B.
 
![hella_3B](https://github.com/user-attachments/assets/06f69a2f-48e2-440a-876a-2cb5b960ae71)

As one could have expected from the perplexity results, sub-3-bpw quantization destroys the model utility. Hence, it is more useful to focus on the 3+ bpw range, which is the purpose of the next graph
 
![hella_3B_a](https://github.com/user-attachments/assets/b49e6b58-362e-4844-982b-89c211000df0)

We see that `IQ4_K, IQ5_K, IQ6_K` and `Q6_K` are basically indistinguishable from the `bf16` model for the HellaSwag metrics. But at less than 2 points below `bf16`, even `IQ3_K` and `IQ3_S` could be useful if HellaSwag is representative for the kind of tasks one intends to tackle.

### MMLU

Here I show only results for the 3+ bpw range for LLaMA-3.2-3B in the following graph
 
![mmlu_3B_a](https://github.com/user-attachments/assets/5562b55f-f2aa-4ee5-b32f-023e698fb22d)

All quantizations above `IQ3_K` (3.6 bpw) are (nearly) indistinguishable from the full `bf16` model according to this metrics.

---

#### 🗣️ Discussion

👤 **ikawrakow** replied the **2024-09-26** at **16:11:00**:<br>

Here some performance numbers for the 1B model on a Ryzen-7950X CPU

| model           |       size | backend    | threads |          test |              t/s |
| --------------- | ---------: | ---------- | ------: | ------------: | ---------------: |
| llama 1B BF16   |   2.79 GiB | CPU        |      16 |         pp512 |  1217.13 ± 18.31 |
| llama 1B BF16   |   2.79 GiB | CPU        |       1 |         tg128 |     15.31 ± 0.19 |
| llama 1B BF16   |   2.79 GiB | CPU        |       2 |         tg128 |     22.97 ± 0.04 |
| llama 1B BF16   |   2.79 GiB | CPU        |       4 |         tg128 |     23.86 ± 0.08 |
| llama 1B BF16   |   2.79 GiB | CPU        |       8 |         tg128 |     23.45 ± 0.32 |
| llama 1B Q8_0   |   1.48 GiB | CPU        |      16 |         pp512 |  1109.36 ± 24.77 |
| llama 1B Q8_0   |   1.48 GiB | CPU        |       1 |         tg128 |     38.57 ± 0.24 |
| llama 1B Q8_0   |   1.48 GiB | CPU        |       2 |         tg128 |     46.86 ± 0.04 |
| llama 1B Q8_0   |   1.48 GiB | CPU        |       4 |         tg128 |     46.42 ± 0.11 |
| llama 1B Q8_0   |   1.48 GiB | CPU        |       8 |         tg128 |     44.41 ± 0.07 |
| llama 1B IQ4_K  | 935.24 MiB | CPU        |      16 |         pp512 |  1211.41 ± 12.99 |
| llama 1B IQ4_K  | 935.24 MiB | CPU        |       1 |         tg128 |     30.81 ± 0.04 |
| llama 1B IQ4_K  | 935.24 MiB | CPU        |       2 |         tg128 |     57.37 ± 0.17 |
| llama 1B IQ4_K  | 935.24 MiB | CPU        |       4 |         tg128 |     76.93 ± 0.14 |
| llama 1B IQ4_K  | 935.24 MiB | CPU        |       8 |         tg128 |     74.61 ± 0.09 |
| llama 1B IQ5_K  |   1.02 GiB | CPU        |      16 |         pp512 |   982.76 ± 16.70 |
| llama 1B IQ5_K  |   1.02 GiB | CPU        |       1 |         tg128 |     24.76 ± 0.04 |
| llama 1B IQ5_K  |   1.02 GiB | CPU        |       2 |         tg128 |     46.39 ± 0.06 |
| llama 1B IQ5_K  |   1.02 GiB | CPU        |       4 |         tg128 |     66.47 ± 0.23 |
| llama 1B IQ5_K  |   1.02 GiB | CPU        |       8 |         tg128 |     64.73 ± 0.10 |
| llama 1B Q5_K_S |   1.03 GiB | CPU        |      16 |         pp512 |  1257.38 ± 13.08 |
| llama 1B Q5_K_S |   1.03 GiB | CPU        |       1 |         tg128 |     31.56 ± 0.55 |
| llama 1B Q5_K_S |   1.03 GiB | CPU        |       2 |         tg128 |     55.68 ± 0.28 |
| llama 1B Q5_K_S |   1.03 GiB | CPU        |       4 |         tg128 |     66.34 ± 0.27 |
| llama 1B Q5_K_S |   1.03 GiB | CPU        |       8 |         tg128 |     65.35 ± 0.23 |
| llama 1B Q6_K   |   1.15 GiB | CPU        |      16 |         pp512 |  1271.25 ± 12.18 |
| llama 1B Q6_K   |   1.15 GiB | CPU        |       1 |         tg128 |     31.43 ± 0.21 |
| llama 1B Q6_K   |   1.15 GiB | CPU        |       2 |         tg128 |     51.40 ± 0.22 |
| llama 1B Q6_K   |   1.15 GiB | CPU        |       4 |         tg128 |     58.25 ± 0.13 |
| llama 1B Q6_K   |   1.15 GiB | CPU        |       8 |         tg128 |     57.64 ± 0.02 |

---

👤 **ikawrakow** replied the **2024-09-26** at **16:18:44**:<br>

Here some performance numbers for the 3B model on a Ryzen-7950X CPU

| model           |       size | backend    | threads |          test |              t/s |          
| --------------- | ---------: | ---------- | ------: | ------------: | ---------------: |
| llama 3B BF16   |   6.72 GiB | CPU        |      16 |         pp512 |   482.81 ± 16.34 |
| llama 3B BF16   |   6.72 GiB | CPU        |       1 |         tg128 |      5.53 ± 0.05 |  
| llama 3B BF16   |   6.72 GiB | CPU        |       2 |         tg128 |      8.65 ± 0.01 |  
| llama 3B BF16   |   6.72 GiB | CPU        |       4 |         tg128 |      9.35 ± 0.02 |  
| llama 3B BF16   |   6.72 GiB | CPU        |       8 |         tg128 |      9.14 ± 0.05 |  
| llama 3B Q8_0   |   3.57 GiB | CPU        |      16 |         pp512 |    383.82 ± 1.85 |
| llama 3B Q8_0   |   3.57 GiB | CPU        |       1 |         tg128 |     14.93 ± 0.30 | 
| llama 3B Q8_0   |   3.57 GiB | CPU        |       2 |         tg128 |     18.66 ± 0.04 | 
| llama 3B Q8_0   |   3.57 GiB | CPU        |       4 |         tg128 |     18.03 ± 0.13 | 
| llama 3B Q8_0   |   3.57 GiB | CPU        |       8 |         tg128 |     17.20 ± 0.03 | 
| llama 3B IQ3_K  |   1.55 GiB | CPU        |      16 |         pp512 |    409.30 ± 3.79 |
| llama 3B IQ3_K  |   1.55 GiB | CPU        |       1 |         tg128 |     11.58 ± 0.01 | 
| llama 3B IQ3_K  |   1.55 GiB | CPU        |       2 |         tg128 |     22.28 ± 0.02 | 
| llama 3B IQ3_K  |   1.55 GiB | CPU        |       4 |         tg128 |     39.25 ± 0.18 | 
| llama 3B IQ3_K  |   1.55 GiB | CPU        |       8 |         tg128 |     37.45 ± 0.08 | 
| llama 3B IQ4_K  |   2.09 GiB | CPU        |      16 |         pp512 |    418.06 ± 2.13 |
| llama 3B IQ4_K  |   2.09 GiB | CPU        |       1 |         tg128 |     12.23 ± 0.04 | 
| llama 3B IQ4_K  |   2.09 GiB | CPU        |       2 |         tg128 |     23.16 ± 0.07 | 
| llama 3B IQ4_K  |   2.09 GiB | CPU        |       4 |         tg128 |     30.55 ± 0.02 | 
| llama 3B IQ4_K  |   2.09 GiB | CPU        |       8 |         tg128 |     29.41 ± 0.16 | 
| llama 3B Q4_K_S |   2.09 GiB | CPU        |      16 |         pp512 |   445.79 ± 15.41 |
| llama 3B Q4_K_S |   2.09 GiB | CPU        |       1 |         tg128 |     13.85 ± 0.03 | 
| llama 3B Q4_K_S |   2.09 GiB | CPU        |       2 |         tg128 |     22.74 ± 0.09 | 
| llama 3B Q4_K_S |   2.09 GiB | CPU        |       4 |         tg128 |     30.74 ± 0.09 | 
| llama 3B Q4_K_S |   2.09 GiB | CPU        |       8 |         tg128 |     29.77 ± 0.02 | 
| llama 3B IQ5_K  |   2.41 GiB | CPU        |      16 |         pp512 |    338.86 ± 7.69 |
| llama 3B IQ5_K  |   2.41 GiB | CPU        |       1 |         tg128 |      9.70 ± 0.12 |  
| llama 3B IQ5_K  |   2.41 GiB | CPU        |       2 |         tg128 |     18.31 ± 0.02 | 
| llama 3B IQ5_K  |   2.41 GiB | CPU        |       4 |         tg128 |     26.21 ± 0.03 | 
| llama 3B IQ5_K  |   2.41 GiB | CPU        |       8 |         tg128 |     25.18 ± 0.10 | 
| llama 3B Q5_K_S |   2.41 GiB | CPU        |      16 |         pp512 |    432.96 ± 2.83 |
| llama 3B Q5_K_S |   2.41 GiB | CPU        |       1 |         tg128 |     12.89 ± 0.15 | 
| llama 3B Q5_K_S |   2.41 GiB | CPU        |       2 |         tg128 |     22.54 ± 0.09 | 
| llama 3B Q5_K_S |   2.41 GiB | CPU        |       4 |         tg128 |     26.37 ± 0.07 | 
| llama 3B Q5_K_S |   2.41 GiB | CPU        |       8 |         tg128 |     25.55 ± 0.02 | 
| llama 3B Q6_K   |   2.76 GiB | CPU        |      16 |         pp512 |    439.73 ± 5.86 |
| llama 3B Q6_K   |   2.76 GiB | CPU        |       1 |         tg128 |     12.90 ± 0.19 | 
| llama 3B Q6_K   |   2.76 GiB | CPU        |       2 |         tg128 |     21.05 ± 0.01 | 
| llama 3B Q6_K   |   2.76 GiB | CPU        |       4 |         tg128 |     22.97 ± 0.01 | 
| llama 3B Q6_K   |   2.76 GiB | CPU        |       8 |         tg128 |     22.20 ± 0.01 |
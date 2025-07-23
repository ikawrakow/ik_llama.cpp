### 🔀 [#295](https://github.com/ikawrakow/ik_llama.cpp/pull/295) - Quantization improvements

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-03-28 |
| **Updated** | 2025-03-30 |

---

#### Description

It is now more than a year since I added the imatrix to `llama.cpp`.  I think we can say that imatrix based quantization is now the standard. Hence, I believe it is no longer necessary to make quantization robust against failure modes that can be triggered when quantizing without an imatrix.

Based on this consideration, this PR adds improved versions of `make_qx_quants`, used to quantize `Q4_0, Q5_0, Q6_0, Q3_K, Q6_K`, and `quantize_row_iq4_nl_impl`, used to quantize `IQ4_NL` and `IQ4_XS`.

The following table shows PPL comparisons between the mai branch, this PR, and [PR 12557](https://github.com/ggml-org/llama.cpp/pull/12557) in mainline `llama.cpp` for LLaMA-v1-7B<sup>1</sup>(L1-7B in the table), LLaMA-v2-7B<sup>1</sup> (L2-7B), Mistral-7B<sup>1</sup> (M-7B), LLaMA-3.1-8B-Instruct (L3-8B), and DeepSeek-V2-Lite (DSL). Context is always 512 tokens. Also given are the quantization times (Q-time for short in the table) in seconds on a Ryzen-7950X CPU. Tested is "pure" quantization (i.e., using the `--pure` option of `llama-quantize`) with token embeddings and output tensor set to `Q8_0`. The quantization command line is
```
./bin/llama-quantize --imatrix $imatrix --token-embedding-type q8_0 --output-tensor-type q8_0 --pure $model $output $quant
```

| Model |  Quantization |  PPL (main) |  PPL (PR 12557) | PPL (this PR) | Q-time (main) | Q-time (PR 12557) | Q-time (this PR) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| L1-7B | Q4_0 | 6.1684 | 6.0276 | 6.0247 | N/A<sup>2</sup> |  N/A<sup>2</sup> |  N/A<sup>2</sup> |
| L2-7B | Q4_0 | 5.9364 | 5.9037 | 5.9056 | 15.1 | 35.2 | 19.7 |
|M-7B | Q4_0 | 5.7924 | 5.7900 | 5.7879 | 16.0 | 44.0 | 22.0 |
| L3-8B | Q4_0 | 7.7039 | 7.5873 | 7.6132 | 17.4 | 46.2 | 23.6 |
| DSL | Q4_0 | 6.9684 | 6.9120 | 6.9286 | 39.5 | 102.8 | 50.7 |
| L1-7B | Q5_0 | 6.0946 | 5.9333 | 5.9320 | N/A<sup>2</sup> |  N/A<sup>2</sup> |  N/A<sup>2</sup> |
| L2-7B | Q5_0 | 5.8228 | 5.8132 | 5.8128 | 15.7 | 56.2 | 20.8 |
|M-7B | Q5_0 | 5.7105 | 5.7113 | 5.7121 | 17.2 | 64.0 | 22.3 |
| L3-8B | Q5_0 | 7.4153 | 7.3829 | 7.3809 | 18.4 | 65.0 | 24.6 |
| DSL | Q5_0 | 6.8160 | 6.8087 | 6.8157 | 41.1 | 144.0 | 52.1 |
| L1-7B | Q6_0 | 5.9183 | N/A<sup>3</sup> | 5.9151 | N/A<sup>2,3</sup> |  N/A<sup>2</sup> |  N/A<sup>2</sup> |
| L2-7B | Q6_0 | 5.8067 | N/A<sup>3</sup> | 5.8039 | 15.8 | N/A<sup>3</sup> | 19.8 |
|M-7B | Q6_0 | 5.6971 | N/A<sup>3</sup> | 5.6962 | 17.7 | N/A<sup>3</sup> | 23.3 |
| L3-8B | Q6_0 | 7.3507 | N/A<sup>3</sup> | 7.3437 | 19.3 | N/A<sup>3</sup> | 25.2 |
| DSL | Q6_0 | 6.7752 | N/A<sup>3</sup> |  6.7779 | 41.8 | N/A<sup>3</sup> | 53.1 |
| L1-7B | Q3_K | 6.4003 | 6.2943 | 6.2865 | N/A<sup>2</sup> |  N/A<sup>2</sup> |  N/A<sup>2</sup> |
| L2-7B | Q3_K | 6.2069 | 6.1678 | 6.1594 |  15.7 | 37.0 | 17.1 |
|M-7B | Q3_K | 5.9961 |  5.9896 | 5.9908 | 16.9 | 41.2 | 18.4 |
| L3-8B | Q3_K | 8.8509 | 8.2609 | 8.2799 | 18.5 | 42.4 | 20.2 |
| DSL | Q3_K | 7.3065 | N/A<sup>4</sup> | 7.2488 | 46.5 | N/A<sup>4</sup> | 57.2 |
| L1-7B | Q6_K | 5.9124 | 5.9122<sup>5</sup> | 5.9110 | N/A<sup>2</sup> |  N/A<sup>2</sup> |  N/A<sup>2</sup> |
| L2-7B | Q6_K | 5.8045 | 5.8050<sup>5</sup>  | 5.8039 |  17.0 | 20.2<sup>5</sup>  | 22.3 |
|M-7B | Q6_K | 5.6995 |  5.6992<sup>5</sup>  | 5.6998 | 18.4 | 22.0<sup>5</sup>  | 25.0 |
| L3-8B | Q6_K | 7.3461 | 7.3463<sup>5</sup>  | 7.3421 |  20.5 | 23.8<sup>5</sup>  | 27.1 |
| DSL | Q6_K | 6.7775 | N/A<sup>4</sup> | 6.7735 | 42.2 | N/A<sup>4</sup> | 51.2 |
| L1-7B | IQ4_NL | 5.9965 | 5.9919 | 5.9889 | N/A<sup>2</sup> |  N/A<sup>2</sup> |  N/A<sup>2</sup> |
| L2-7B | IQ4_NL | 5.8725 | 5.8772 | 5.8729 |  24.3 | 125.6 | 35.4 |
|M-7B | IQ4_NL | 5.7581 |  5.7658 | 5.7600 |  26.1 |  134.7 | 38.6 |
| L3-8B | IQ4_NL | 7.5388 | 7.5260 | 7.5261 |  27.6 | 136.3 | 39.1 |
| DSL | IQ4_NL | 6.8795 | 6.8599 | 6.8700 | 53.1 | 315.7 | 87.2 |
| L1-7B | IQ4_XS | 5.9929 | 5.9914 | 5.9875 | N/A<sup>2</sup> |  N/A<sup>2</sup> |  N/A<sup>2</sup> |
| L2-7B | IQ4_XS | 5.8731 | 5.8801 | 5.8721 |  22.8  | 124.9 | 29.3 |
|M-7B | IQ4_XS | 5.7586 |  5.7694 | 5.7622 |  24.2 |  134.1 | 38.0 |
| L3-8B | IQ4_XS | 7.5515 | 7.5515 | 7.5417 |  25.7 | 135.9 | 39.0 |
| DSL | IQ4_XS | 6.8832 | N/A<sup>4</sup> | 6.8774 | 57.5 | N/A<sup>4</sup> | 88.8 |
___
<sup>1</sup> Why use such ancient models? The LLaMA-v1 models were the basis for k-quants development. I-quants were developed using LLaMA-v1, LLaMA-v2 and Mistral-7B. In my experience, if a quantization technique does well on all 3 of these, it is (almost) guaranteed to do well on any other model out there. 

<sup>2</sup> I have this model on an old HDD. In this case quantization time is dominated by the time needed to read the data from the HDD. I could have copied the model to the SSD drive, but I think the timing for the other models give enough indication of the relative performance of the various quantization techniques.

<sup>3</sup> This quantization type is not available in mainline `llama.cpp`.

<sup>4</sup> Some of the tensor row size are not divisible by the k- and i-quants super-block size of 256. In mainline `llama.cpp` the quantization fails in that case when using `--pure`. I have changed `ik_llama.cpp` to use the fallback quantization type in that case in PR #294.

<sup>5</sup> PR 12557 does not change `Q6_K` quantization. 

### Some background

Quantization involves a mixed-integer optimization problem, which is hard to solve in general. But in the case of block-wise quantization, where each block is quantized independently, and hence one has to deal with just 16 or 32 variables, an exact solution is feasible without very long computation times. However, the experience with the LLaMA-v1 series of models collected while developing k-quants showed that the exact solution can often lead to disastrous results in observed quantization quality (e.g., a much higher perplexity or lower HellaSwag score). Hence, k-quants and later i-quants used heuristics to search for a solution only within a carefully tuned range of scales around the round-to-nearest (RTN) value. When I added the i-matrix, the hope was that one can discard the heuristics and use the exact solution instead. But even with an imatrix, it was possible to arrive at a catastrophic failure (see, e.g., the results of the main branch for `Q4_0` and `Q5_0`. To avoid such failures, when quantizing without `--pure`, a different quantization type is used for the `ffn_down` tensors in the first few layers). In addition, often quantizations were prepared without an imatrix, so the quantization technique had to be made robust also for this use case. Hence, the heuristics remained.

In [PR 12557](https://github.com/ggml-org/llama.cpp/pull/12557) in mainline `llama.cpp` @compilade uses a (nearly) exhaustive search for optimality, whith correspondingly very long quantization times. One can arrive at about the same result much quicker as follows. To minimize the weighted-mean-square-error (WMSE) between the original model weights $x_i$ and the integer quants $q_i$, one needs to maximize

$$F = \frac{\left(\sum w_i x_i q_i\right)^2}{\sum w_i q_i^2}$$

where the `w_i` are importances given by, e.g., an imatrix (but can also be defined in some different way when no matrix is available), and the summation is over the elements of a quantization block. The above equation is for a "Type-0" quantization where the quantized model weight $\tilde{x}_i$ is give by $\tilde{x}_i = d q_i$, and where $d$ is the float block scale. The block scale that minimizes $WMSE$ is given by

$$d =  \frac{\sum w_i x_i q_i}{\sum w_i q_i^2}$$

The gradient $g_j$ of the integer quant $q_j$ is given by

$$g_j = \frac{\partial F}{\partial q_j} = 2 d w_j (x_j - d q_j)$$

If we take a step along the gradient (we are maximizing $F$, so need to go along the gradient), the quant with the maximum $|g_j|$ will be first to change to the next integer value ($q_j + \Delta_j$, where $\Delta_j = 1$ if $g_j > 0, -1$ otherwise). Hence we can compute the new value of $F$ by just adding $w_j x_j \Delta_j$ to the numerator and $w_j (2 q_j \Delta_j + 1)$ to the denominator. If the new value of $F$ is greater than the previous highest value, we accept the change, set $q_j \to q_j + \Delta_j$, compute the new optimum scale $d$, and repeat the previous steps. If the new value of $F$ is lower than the previous highest $F$, we break out from the iteration. This is very similar to the exact solution technique,  except that there one doesn't check just the quant with the maximum gradient, but adds all possible steps along the gradient that change the quants to the next integer value along the gradient while the quants are within the allowed range, sorts the steps in increasing order, and then goes over the steps updating one quant at a time, computing the updated $F$, and picking the step that resulted in the maximum value for $F$. Because of that, this kind of "first order" approximation is much faster than exhaustive search, as can be seen in the above table by comparing quantization run times between this PR and @compilade's PR 12557, while achieving effectively the same quantization accuracy as measured by PPL.

Extending the above algorithm to the non-linear quants `IQ4_XS` and `IQ4_NL` is trivial. One just needs to replace $q_i$ with $T(q_i)$ in the above equations, where $T(q_i)$ is the non-linear mapping function (lookup table), i.e., we have $\tilde{x}_i = d T(q_i)$

---

#### 💬 Conversation

👤 **compilade** commented the **2025-03-28** at **15:35:37**:<br>

Nice! It seems like your improved `make_qx_quants` is extremely similar to `make_qkxh_quants` when starting the search from `MIN(abs(nmin), abs(nmax)) - 1` instead of `MIN(abs(nmin), abs(nmax)) / 2` (when comparing the equirectangular projections). This would also make `make_qkxh_quants` faster (though I don't know by how much).

Here's your improved `make_qx_quants` with settings from `Q4_0`:

![equirectangular-tmp-2048](https://github.com/user-attachments/assets/3b0c3d0e-92c7-43f9-b498-2bb3adf4143c)

And your improved `quantize_row_iq4_nl_impl` looks like this:

![equirectangular-tmp2-2048](https://github.com/user-attachments/assets/855d814b-15bd-46b8-8546-42ed2f71f4b5)


Very interesting approach with the gradient.

---

👤 **ikawrakow** commented the **2025-03-28** at **19:44:43**:<br>

To be honest I don't understand these plots. I know yellow is good and blue is bad, and there is a lot of blue, so they must be pretty bad?

---

👤 **compilade** commented the **2025-03-28** at **19:59:47**:<br>

> To be honest I don't understand these plots. I know yellow is good and blue is bad, and there is a lot of blue, so they must be pretty bad? 

No, the plots of your algorithms are not bad. Blue is simply the color of the max error. I did also include the min mean and max cosine similarities of the plots.

If an algorithm had a very big error in one spot, everything else would be yellow. This means the colors can't really be compared directly.

The information which can be gotten out of those plots is whether the algorithms have spots where a transition between representable values is very harsh, which can indicate either instability in the algorithm or non-idealness.

In this case, the modifications you propose here **do improve** how the plots look like (for `IQ4_NL` there was otherwise a lot of sudden changes in the error in the original version).

---

👤 **ikawrakow** commented the **2025-03-28** at **20:03:32**:<br>

And what are the two coordinates of the plot? I understand it is a projection, but what is it that is being projected?

> Very interesting approach with the gradient.

That would be the standard way to approach an optimization problem, no?

---

👤 **compilade** commented the **2025-03-28** at **20:55:13**:<br>

> And what are the two coordinates of the plot? I understand it is a projection, but what is it that is being projected?

The horizontal coordinates is `theta` which goes from 0 to 2*π radians, while the vertical coordinates is `phi`, which goes from 0 to π radians.

The vectors tested have the form $[\sin(\phi) \cdot \cos(\theta), \sin(\phi) \cdot \sin(\theta), \cos(\phi)]$

The script which I'm using is <https://github.com/compilade/rounding-experiments/blob/main/equirectangular.py>, although I have some local modifications to make it use other rounding algorithms, which are defined in [`rounding-impl.c`](https://github.com/compilade/rounding-experiments/blob/main/rounding-impl.c) with Python bindings in [`rounding_c.py`](https://github.com/compilade/rounding-experiments/blob/main/rounding_c.py).

> > Very interesting approach with the gradient.
> 
> That would be the standard way to approach an optimization problem, no?

Sure. Being standard doesn't mean it's not interesting. You have made the gradients explicit, which I appreciate.

And ***if*** your gradient search and my cumulative search (once the range is reduced) are equivalent (or close enough), that in itself is interesting, since I did not explicitly use gradients.

I really like when different approaches end up being equivalent (or close enough) because this makes them easier to understand, explain and generalize to other cases (notably, my approach might be harder to adapt to grid-restricted i-quants).

(If they are not equivalent, this is still very cool, even if this is technically using a standard approach)

I will compare the speed and perplexity of narrower cumulative search with this once I have some spare time, since I do think reducing the searched range will greatly improve the speed of my (currently quite slow) proposed algorithms.

---

👤 **saood06** commented the **2025-03-28** at **23:16:13**:<br>

>Tested is "pure" quantization (i.e., using the `--pure` option of `llama-quantize`) with token embeddings and output tensor set to `Q8_0`. 

Was this needed for some quants of DSL to function? As I ran into issues with a pure iq4_k_r4 quant for the new Deepseek V3 0324 (as my first mix of this finetune was noticeably slower than my first and fastest mix of R1).

The pure ran at about the same speed as that R1 mix (I think it should have been a bit faster than it is and the speed loss may be from #259 since for this model I did not convert it myself and grabbed a conversion that was done with mainline), but it was not functional (I forgot to test perplexity before unloading it), either giving a few incomprehensible tokens or just straight to an EOS token from my brief usage.

Comparing the quant logs for both, the only different tensors of the functional R1 mix were the following 5:

```
blk.X.attn_k_b.weight - [  128, 65536,     1,     1]
llama_tensor_get_type : tensor cols 128 x 65536 are not divisible by 256, required for iq4_k_r4 - using fallback quantization q5_0
====== llama_model_quantize_internal: did not find weights for blk.X.attn_k_b.weight
converting to q5_0 .. size =    16.00 MiB ->     5.50 MiB
```

```
blk.X.attn_v_b.weight - [  512, 16384,     1,     1]
====== llama_model_quantize_internal: did not find weights for blk.X.attn_v_b.weight
converting to iq4_k_r4 .. size =    16.00 MiB ->     4.50 MiB
```

These two tensors were not in my new mix as mentioned above, being computed (`Computed blk.X.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CPU`)

```
 blk.X.attn_output.weight - [16384,  7168,     1,     1], converting to q5_K .. size =   224.00 MiB ->    77.00 MiB
```

```
output.weight - [ 7168, 129280,     1,     1],
====== llama_model_quantize_internal: did not find weights for output.weight
converting to q6_K .. size =  1767.50 MiB ->   724.95 MiB
```

```
token_embd.weight - [ 7168, 129280,     1,     1], type =    f16,
====== llama_model_quantize_internal: did not find weights for token_embd.weight
converting to iq4_k .. size =  1767.50 MiB ->   497.11 MiB
```

The new pure V3 had all of the three of the above set to iq4_k_r4.

Also for reference the full tensor breakdown of both mixes:

R1 fast and functional:
```
llama_model_loader: - type f32: 361 tensors
llama_model_loader: - type q5_0: 61 tensors
llama_model_loader: - type q5_K: 61 tensors
llama_model_loader: - type q6_K: 1 tensors
llama_model_loader: - type iq4_k: 1 tensors
llama_model_loader: - type iq4_k_r4: 662 tensors
llm_load_print_meta: model params = 672.050 B //this is higher because of MLA tensor inclusion
```

Pure mix of V3_0324:
```
llama_model_loader: - type  f32:  361 tensors
llama_model_loader: - type iq4_k_r4:  664 tensors
llm_load_print_meta: model params     = 671.026 B //this is lower because of MLA tensor exclusion
```

Do you think that setting output.weight to iq6_k and leaving the rest completely pure would work? 

When I do make this next quant I might end up converting the model myself to see if #259 was costing me performance (even if I won't be comparing the exact same mix, I think it would still answer that question).

---

👤 **ikawrakow** commented the **2025-03-29** at **06:53:18**:<br>

> When I do make this next quant I might end up converting the model myself to see if https://github.com/ikawrakow/ik_llama.cpp/pull/259 was costing me performance

#259 creates `attn_k_b` and `attn_v_b` as `Q8_0`, so this can have impact on TG performance compared to a model where these tensors were created with lower bpw. Apart from this, your system seems to be extremely sensitive to how things are laid out in memory, and creating `attn_k_b` and `attn_v_b` on the fly will lead to a different memory layout.

 >  but it was not functional (I forgot to test perplexity before unloading it), either giving a few incomprehensible tokens or just straight to an EOS token from my brief usage.

Not sure about this one.

---

👤 **saood06** commented the **2025-03-29** at **07:36:32**:<br>

> > When I do make this next quant I might end up converting the model myself to see if #259 was costing me performance
> 
> #259 creates `attn_k_b` and `attn_v_b` as `Q8_0`, so this can have impact on TG performance compared to a model where these tensors were created with lower bpw.

Yes I experimented with some quant mixes with those at Q8_0 before to see how much impact they had on PPL (but never isolated effects as the change in PPL was too minor and the TG impact too large for my preferences).

>Apart from this, your system seems to be extremely sensitive to how things are laid out in memory, and creating `attn_k_b` and `attn_v_b` on the fly will lead to a different memory layout.

Yes it is unfortunately very sensitive to that, I even considered #259 before I downloaded this preconverted model but decided to try it anyway.

> > but it was not functional (I forgot to test perplexity before unloading it), either giving a few incomprehensible tokens or just straight to an EOS token from my brief usage.
> 
> Not sure about this one.

I'll test attn_output.weight set to iq6_k and report back when I get a chance (will first have to download and convert the model so that I can also test #259 ).

---

👤 **saood06** commented the **2025-03-30** at **08:44:47**:<br>

> I'll test attn_output.weight set to iq6_k and report back when I get a chance (will first have to download and convert the model so that I can also test #259 ).

This was also outputting gibberish.
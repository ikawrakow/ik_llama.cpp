### 🐛 [#367](https://github.com/ikawrakow/ik_llama.cpp/issues/367) - Bug: IQ1_S_R4, IQ1_M_R4 failed on Qwen3-235B-A22B

| **Author** | `Flying-Cloud` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-05-03 |
| **Updated** | 2025-05-04 |

---

#### Description

### What happened?

I was trying to quantize Qwen3-235B-A22B using IQ1_M_R4 quantization type. I found that it fails on quantizing blk.1.ffn_gate_exps.weight - [ 4096,  1536,   128,     1] The main issue seems in 
```python
size_t quantize_iq1_m_r4(const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix)
                ...
                iq1m_process_1block(xb+ 0, weight+ 0, L, scales.data() + 8*ibl + 2*k+0, index+0, &shift1, pairs);
                iq1m_process_1block(xb+16, weight+16, L, scales.data() + 8*ibl + 2*k+1, index+2, &shift2, pairs);
                ...
```
I have tried IQ1_M, it works well on Qwen3-235B-A22B. Only IQ1_M_R4 fails
I tried IQ1_S_R4, it also fails.

### Name and Version

./build/bin/llama-quantize --ignore-imatrix-rules --imatrix ./Qwen3-235B.imatrix  /models/Qwen3-235B-A22B/BF16/Qwen3-235B-A22B-BF16-00001-of-00010.gguf  /models/Qwen3-235B-A22B/gguf_new/Qwen3-235B-A22B.gguf IQ1_M_R4

### What operating system are you seeing the problem on?

Linux

### Relevant log output

```shell
[  15/1131]             blk.1.attn_k_norm.weight - [  128,     1,     1,     1], type =    f32, size =    0.000 MB                                                                                     [17/1919]
[  16/1131]                  blk.1.attn_k.weight - [ 4096,   512,     1,     1], type =   bf16, converting to q4_k_r4 .. size =     4.00 MiB ->     1.12 MiB                                                    
[  17/1131]             blk.1.attn_output.weight - [ 8192,  4096,     1,     1], type =   bf16, converting to q5_k_r4 .. size =    64.00 MiB ->    22.00 MiB                                                    
[  18/1131]             blk.1.attn_q_norm.weight - [  128,     1,     1,     1], type =    f32, size =    0.000 MB                                                                                              
[  19/1131]                  blk.1.attn_q.weight - [ 4096,  8192,     1,     1], type =   bf16, converting to q4_k_r4 .. size =    64.00 MiB ->    18.00 MiB                                                    
[  20/1131]                  blk.1.attn_v.weight - [ 4096,   512,     1,     1], type =   bf16, converting to iq4_k_r4 .. size =     4.00 MiB ->     1.12 MiB                                                   
[  21/1131]               blk.1.attn_norm.weight - [ 4096,     1,     1,     1], type =    f32, size =    0.016 MB                                                                                              
[  22/1131]           blk.1.ffn_down_exps.weight - [ 1536,  4096,   128,     1], type =   bf16, converting to q2_k_r4 .. size =  1536.00 MiB ->   252.00 MiB                                                    
[  23/1131]           blk.1.ffn_gate_exps.weight - [ 4096,  1536,   128,     1], type =   bf16, converting to iq1_m_r4 .. /home2/llm/llama_project/llama_cpp/ggml/src/ggml-quants.c:14324: GGML_ASSERT(be
sti1 >= 0 && besti2 >= 0 && best_k >= 0) failed                                                                                                                                                                 
/home2/llm/llama_project/llama_cpp/ggml/src/ggml-quants.c:14324: GGML_ASSERT(besti1 >= 0 && besti2 >= 0 && best_k >= 0) failed                                                                           
/home2/llm/llama_project/llama_cpp/ggml/src/ggml-quants.c:14324: GGML_ASSERT(besti1 >= 0 && besti2 >= 0 && best_k >= 0) failed                                                                           
/home2/llm/llama_project/llama_cpp/ggml/src/ggml-quants.c:14324: GGML_ASSERT(besti1 >= 0 && besti2 >= 0 && best_k >= 0) failed
```

---

#### 💬 Conversation

👤 **Flying-Cloud** commented the **2025-05-03** at **10:26:11**:<br>

Oh I guess it because 1536 / 256 = 6 which is not divisible by 4?

---

👤 **ikawrakow** commented the **2025-05-03** at **10:29:06**:<br>

The number of rows must be a multiple of 4, not the number of blocks. Qwen3-235B-A22B should work with any `_R4` or `_R8` quant. The issue is in the quantization function itself. I'll look into it.

---

👤 **ikawrakow** commented the **2025-05-03** at **11:01:47**:<br>

There is PR #368. Does it fix it? I cannot actually run such a large model (not enough RAM, not enough disk space), so it is a bit if a guessing game.

---

👤 **Flying-Cloud** commented the **2025-05-03** at **11:32:22**:<br>

> There is PR [#368](https://github.com/ikawrakow/ik_llama.cpp/pull/368). Does it fix it? I cannot actually run such a large model (not enough RAM, not enough disk space), so it is a bit if a guessing game.

It works! No longer error displayed. So what's the matter here. It seems like there are some near-zero weights in gate_proj weights?

---

👤 **ikawrakow** commented the **2025-05-03** at **11:35:12**:<br>

Either near zero weights, or the more tricky one, mismatching imatrix. Mismatching in the sense that the imatrix importances are zero where the model weights are not zero.

---

👤 **Flying-Cloud** commented the **2025-05-03** at **11:37:10**:<br>

Got it. It make sense since I notice the imatrix I downloaded from unsloth is computed through only 46 chunks. Thanks for your quick reply!

---

👤 **Flying-Cloud** commented the **2025-05-03** at **15:36:56**:<br>

Sorry to bother you again. I just found that IQ1_M_R4 fail in the deep layer of Qwen3-235B-A22B:  blk.18.ffn_down_exps.weight
I try to revise the code from:
```python
float sumwx = 0;
                    for (int j = 0; j < kBlockSize; ++j) sumwx += weight[j]*std::abs(xb[j]);
                    if (!sumwx) {
                        for (int j = 0; j < kBlockSize; ++j) weight[j] = sqrt(sigma2 + xb[j]*xb[j]);
                    }
```
to
```python
float sumwx = 0;
                    for (int j = 0; j < kBlockSize; ++j) sumwx += weight[j];
                    if (sumwx < 1e-3) {
                        for (int j = 0; j < kBlockSize; ++j) weight[j] = sqrt(sigma2 + xb[j]*xb[j]);
                    }
```
Still same Error as the issue begins.

---

👤 **Flying-Cloud** commented the **2025-05-03** at **15:36:56**:<br>

Sorry to bother you again. I just found that IQ1_M_R4 fail in the deep layer of Qwen3-235B-A22B:  blk.18.ffn_down_exps.weight
I try to revise to code from:
```python
float sumwx = 0;
                    for (int j = 0; j < kBlockSize; ++j) sumwx += weight[j]*std::abs(xb[j]);
                    if (!sumwx) {
                        for (int j = 0; j < kBlockSize; ++j) weight[j] = sqrt(sigma2 + xb[j]*xb[j]);
                    }
```
to
```python
float sumwx = 0;
                    for (int j = 0; j < kBlockSize; ++j) sumwx += weight[j];
                    if (sumwx < 1e-3) {
                        for (int j = 0; j < kBlockSize; ++j) weight[j] = sqrt(sigma2 + xb[j]*xb[j]);
                    }
```
Still same Error as the issue begins.

---

👤 **ikawrakow** commented the **2025-05-03** at **15:49:48**:<br>

So, we need to see what these values are that cause the assert.
Just before
```c++
GGML_ASSERT(besti1 >= 0 && besti2 >= 0 && best_k >= 0);
```
you can add
```c++
if (besti1 < 0 || besti2 < 0 || best_k < 0) {
    printf("Failed to find optimum division\nValues:\n");
    for (int i = 0; i < block_size; ++i) {
        printf("%d  %g  %g\n", i, weight[i], xb[i]);
    }
}
```

The strange part if that in the log that you posted above the assert is on line 14324, but I don't have an assert on that line. Instead, if it fails for `iq1_m_r4`, the assert should be on line 14466.

---

👤 **Flying-Cloud** commented the **2025-05-03** at **16:13:39**:<br>

I apply this code, and the results are:
```
[ 166/1131]          blk.13.ffn_down_exps.weight - [ 1536,  4096,   128,     1], type =   bf16, converting to iq1_m_r4 .. Failed to find optimum division                          
Values:                      
0  5.21497e-22  5.55515e-05 
1  1.7415e-21  9.20296e-05                                                                                                                                       
2  2.79688e-21  -6.91414e-05
3  1.52191e-21  0.000104427                                                                                                                                      
4  3.59385e-21  -2.22921e-05
5  5.47448e-21  -9.39369e-05                                                                                                                                     
6  2.96794e-22  0.000101566  
7  1.15378e-20  -9.25064e-05                                                                                                                                     
8  3.73609e-23  2.36034e-05  
9  1.50841e-21  7.96318e-05                                                                                                                                      
10  4.79334e-17  -3.07336e-07
11  2.84946e-22  9.72748e-05
12  2.6887e-21  2.8491e-05                                                                                                                                       
13  1.21816e-21  0.00011301 
14  2.37663e-19  2.96831e-05                                                                                                                                     
15  3.55494e-22  0.000113487  
```

---

👤 **ikawrakow** commented the **2025-05-03** at **16:15:48**:<br>

Oh, I see. Give me a minute, I'll push a fix.

---

👤 **ikawrakow** commented the **2025-05-03** at **16:31:08**:<br>

See #371.

The issue was I checked for very small values in a block of 32 quants. But then we quantize 2 blocks of 16 each. Hence, it can happen that the block of 32 has non-zero values, but one of the blocks of 16 does not.

---

👤 **Flying-Cloud** commented the **2025-05-03** at **16:50:07**:<br>

```
[  22/1131]           blk.1.ffn_down_exps.weight - [ 1536,  4096,   128,     1], type =   bf16, converting to iq1_m_r4 .. Failed to find optimum division
Values:                                                                                                                                                                                      
Failed to find optimum division
Values:                     
0  0  -0.000106335           
1  2.70895e-19  -6.03199e-05
2  1.64793e-32  -4.22001e-05
3  0  2.47955e-05                                                                                                                                                                            
4  0  7.00951e-05 
5  2.11893e-21  -8.52346e-06
6  3.84517e-21  3.38554e-05
7  5.97258e-30  -0.000101566 
8  2.10669e-23  -9.10759e-05
9  2.90266e-25  2.70605e-05
10  0  5.55515e-05                                                                                                                                                                           
11  0  4.95911e-05
12  0  -0.000106335
13  2.25005e-22  -4.86374e-05
14  9.6013e-28  8.4877e-05
15  0  -9.72748e-05
```
Still error, now it just fails in blk.1.
I guess should change "!sumwx" to "sumwx < {a small threshold}"?

---

👤 **ikawrakow** commented the **2025-05-03** at **17:01:35**:<br>

I pushed another attempt.

---

👤 **Flying-Cloud** commented the **2025-05-03** at **17:29:44**:<br>

I tried the new attempt and it overcomes the barrier of  "blk.13 down_exps" and "blk.18. down_exps"
If this success with whole quantization process for Qwen3-235B, I will check the ppl to ensure that it functions well.
It might takes a few time and I will let you know right away

---

👤 **whatever1983** commented the **2025-05-03** at **20:24:08**:<br>

Seriously, are you guys crazy to quant the Qwen3 series with IQ1S?  I am having trouble generating a working python Tetris game using 30B-A3B using IQ5K that I am forced to use IQ6K.   The Qwen3 is a regression many ways trying to use too little active parameters, the end result is that any quantization at all wrecks coding performance. 

Just a interesting observation, DS 0324 IQ2M is able to generate a fully working Tetris that's way more beautiful.  

Jack Ma is too focused on proving to the market that making active parameters as little as possible is the way to greater AI, which is totally wrong.  You know, shorting the US market as a way of payment for releasing shitty little models is not the way forward for better AI.

---

👤 **whatever1983** commented the **2025-05-03** at **20:24:08**:<br>

Seriously, are you guys crazy to quant the Qwen3 series with IQ1S?  I am having trouble generating a working python Tetris game using 30B-A3B using IQ5K that I am forced to use IQ6K.   The Qwen3 is a regression many ways trying to use too little active parameters, the end result is that quanting at all recks coding performance. 

Just a interesting observation, DS 0324 IQ2M is able to generate a fully working Tetris that's way more beautiful.  

Jack Ma is too focused on proving to the market that making active parameters as little as possible is the way to greater AI, which is totally wrong.  You know, shorting the US market as a way of payment for releasing shitty little models is not the way forward for better AI.

---

👤 **Flying-Cloud** commented the **2025-05-04** at **04:11:04**:<br>

> I tried the new attempt and it overcomes the barrier of "blk.13 down_exps" and "blk.18. down_exps" If this success with whole quantization process for Qwen3-235B, I will check the ppl to ensure that it functions well. It might takes a few time and I will let you know right away

I test the ppl by comparing the first 20 chunks. Models are quantized with all ffn layers in IQ1_M_R4 except layer 0. The results show that imatrix works well, improving ppl from 8.94 -> 7.79
```
IQ1_M_R4 with Imatrix
[1]5.4013,[2]7.1211,[3]6.7454,[4]6.2828,[5]6.6146,[6]6.7710,[7]6.8578,[8]7.1599,[9]7.5218,[10]7.8782,[11]7.8624,[12]8.0117,[13]8.3022,[14]8.1145,[15]8.0912,[16]8.3432,[17]7.8917,[18]7.9417,[19]7.8582,[20]7.7944,
IQ1_M_R4 without imatrix
[1]6.2115,[2]7.8978,[3]7.3012,[4]7.1488,[5]7.2871,[6]7.5437,[7]7.6070,[8]8.0672,[9]8.5137,[10]8.9069,[11]8.9250,[12]9.1706,[13]9.4891,[14]9.2762,[15]9.2846,[16]9.5792,[17]9.0253,[18]9.1182,[19]9.0367,[20]8.9418,
```

> I pushed another attempt.

BTW, thish push has a minor typo: "1e-14f" instead of "1e-14"


> Seriously, are you guys crazy to quant the Qwen3 series with IQ1S? I am having trouble generating a working python Tetris game using 30B-A3B using IQ5K that I am forced to use IQ6K. The Qwen3 is a regression many ways trying to use too little active parameters, the end result is that any quantization at all wrecks coding performance.
> 
> Just a interesting observation, DS 0324 IQ2M is able to generate a fully working Tetris that's way more beautiful.
> 
> Jack Ma is too focused on proving to the market that making active parameters as little as possible is the way to greater AI, which is totally wrong. You know, shorting the US market as a way of payment for releasing shitty little models is not the way forward for better AI.

I test the IQ1_S/IQ1_M on Qwen3 just for research purpose. It is interesting for me if a large-scale moe model can be downsized through SOTA low-bits quantization type. But I will never try it on a small model like 30B-3B, which can more easily fit in with workstation or personal PC environment through current reasonable techniques like AWQ, Q4 Quant.
I agree with you that Qwen3 series have spent too much effort on chasing high scores across various leaderboards. It is revealed that Qwen3 series models even worse than O1-mini in SimpleQA, which prove that these models are lack in world knowledge.
I suspect that Jack Ma kept the more powerful Qwen-Max model internally while choosing to open-source the Qwen3 series, which performs better on leaderboards, to attract attention.

---

👤 **ikawrakow** commented the **2025-05-04** at **04:19:01**:<br>

> BTW, thish push has a minor typo: "1e-14f" instead of "1e-14"

`1e-14f` is how you write this value as float. `1e-14` is a double.

---

👤 **ikawrakow** commented the **2025-05-04** at **04:27:20**:<br>

> Seriously, are you guys crazy to quant the Qwen3 series with IQ1S? I am having trouble generating a working python Tetris game using 30B-A3B using IQ5K that I am forced to use IQ6K.

For my part, I'm here to make the tools and not the rules. 😄 

But people do use LLMs for many different things, not just for coding. People also have very different systems where often they can only run the largest models using low-bit quantization. Hence, for some people such craziness is useful.
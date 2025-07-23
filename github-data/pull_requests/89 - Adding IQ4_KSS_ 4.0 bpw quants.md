### 🔀 [#89](https://github.com/ikawrakow/ik_llama.cpp/pull/89) - Adding IQ4_KSS: 4.0 bpw quants

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2024-10-16 |
| **Updated** | 2024-10-17 |

---

#### Description

@Nexesenex has been asking for a 4.0 bpw quantization here and in `llama.cpp`. Well, here it is.

It uses the same non-linear grid as `IQ4_K` and `IQ4_KS`. Compared to `IQ4_KS`, we save 0.25 bpw by enforcing that the number of set bits in a group of 4 quants is even (i.e., we need 15 bits for 4 quants, so 3.75 bpw). Combined with 7+1 bits per block of 32 weights (7 bits for the scale + 1 bit indicating if there is a grid shift), we arrive at exactly 4.0 bpw. (well, there is also one float per tensor row, but that is < 0.01 bpw for 7B+ parameter models, so negligible). The best way I was able to come up with for packing the bits is to combine the 15 bits needed for the quants with the one extra bit per group of 4, needed for the block scale/grid shift, into a 16 bit unsigned integer. If prepared appropriately, the 15 quant bits can be converted to 16 bits for easier unpacking by just using `v ^ (v >> 1)` where `v` contains the 15 bits shifted 1 bit to the left. Assembling the scale from single bits stored in the `uint16_t` packed data is computationally more costly. My RTX-4080 GPU handles it gracefully, without noticeable impact on inference performance. Zen4 is also mostly OK as one can use the `_mm512_cmpeq_epi16_mask` instruction to pack the scale/shift bits back together. But on `AVX2`, `ARM_NEON`, and `Metal`, performance is noticeably lower compared to, say, `IQ4_KS`.

My initial idea for implementing the quantization function was to simply first quantize to `IQ4_KS`, and then prune to `IQ4_KSS` by flipping one bit per group of 4 (if number of set bits is odd), where the bit to be flipped is selected to minimize the difference to the original model weights. This kind of worked, but the resulting quantization error was higher than I was hoping for, so I ended up writing a dedicated `IQ4_KSS` method, where enforcing even number of set bits per group of 4 is incorporated into the block scale search. This makes quantization significantly slower than `IQ4_KS` (e.g., about 113 seconds vs 51 seconds for `IQ4_KS` to quantize a 7B parameter model on a Ryzen-7950X CPU). 

In terms of quantization accuracy, these new quants mostly end up where one would expect them to be from the bpw vs quantization error curve established by other iqk-quants.

The first graph is for LLaMA-3.1-8B instruct. As I had recently done these calculation to compare with VPTQ, the new quantization approach from the Microsoft team claiming to be SOTA, the token embedding and output tensor are left as `fp16`, and the bpw only includes the tensors from the repeating layers. I have added labels to the 4-bit quants for easier disambiguation.  

 
![il31](https://github.com/user-attachments/assets/f0c0f60c-fa71-48c9-a082-94e7a61eb80e)

In all following graph the token embedding and output tensors are quantized, and the bpw is for the total model (i.e., total number of bits, including embeddding and output tensors, divided by total number of model parameters).

   
![inemo2407_ppl](https://github.com/user-attachments/assets/a38361c6-72d0-4154-800e-4d2b4a4fbac1)

![iphi3 5_ppl](https://github.com/user-attachments/assets/ba509eb2-13c4-423e-a98e-8097b4141fb5)

![iqwen2 5_ppl](https://github.com/user-attachments/assets/4698c60a-b0da-4990-bffd-74df3e657d57)

![g9](https://github.com/user-attachments/assets/30fa38a5-0b8a-4120-9cd5-10a03c0e505a)

---

#### 💬 Conversation

👤 **Nexesenex** commented the **2024-10-16** at **20:38:07**:<br>

Hey IK,

Congratulations and thank you. Now, I'm gonna try to make all of this work, because I ideally don't want to ever touch 3 bits quants ever again (except for attn_q.weight :P). I'll report my progresses. :D

---

👤 **Nexesenex** commented the **2024-10-16** at **23:20:20**:<br>

The new IQ4_KSS quant is really SOTA imo, and thank you very much. You're rocking the place, as usual.

Now, I see that IQ3_K is at 3.43bpw, and is close to IQ3_S, which was itself a bit better in its first version than its second one back when you launched it on official. Is there room to progress on IQ3_K?

I have already what I need for my own use now, but would you be willing to crack a IQ3_KM 3.65-3.75bpw, midrange between IQ3_K and IQ4_KSS. There might be a sweet spot for your maths around there, way below the usual IQ "line".

Also, I observed how Exllama v2 quantizes. Turboderp's tool calculates something akin to what quantize stats does in order to decide, in respect for a broad quant strategy, what tensor to quantize at which bpw, am I correct?

With an IQ3_KM and an IQ3_KSS, you might be able to drop down a bit (attn_q wise, and ffn_gate wise) the bpw of the quant strategies revolving in the 3 to 4.5 bpw bracket. Ofc, the logic applies on the whole scope, but that's a work I'm only able to suggest, not to do myself lol.

Then, if you were willing to code an automatic quantization system akin to Exllama v2, but maybe more rigorous on the skeleton "ftype" strategy employed (due to the knowledge gained in all the experimentation with FTYPES) and an automatic upscale or downscale (compared to the skeleton 'ftype" strategy) of the quant of a given tensor accordingly to its "error rate", then the process of strategization of the quants would be greatly helped, and the FTYPES also could be SOTA, on the top of your SOTA GGML_TYPES.

On my side, I ponder seriously about trying to rebase my KoboldCPP fork on your LlamaCPP clone, to offer the benefit of your quants to myself and others in daily use.

---

👤 **Nexesenex** commented the **2024-10-17** at **03:30:26**:<br>

I tested your IQ6_K quant on Nemo 12b on ST/llama-server, and it indeed feels very like a Q8_0.
Your quants are amazing.
This night, I'm gonna quant a IQ4_KSS modified ftype for Mistral 123b. I can't wait ! :D
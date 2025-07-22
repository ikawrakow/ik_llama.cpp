### ✨ [#293](https://github.com/ikawrakow/ik_llama.cpp/issues/293) - Feature Request: IQ6_K row interleaved quant

| **Author** | `saood06` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-03-27 |
| **Updated** | 2025-04-24 |

---

#### Description

### Prerequisites

- [x] I am running the latest code. Mention the version if possible as well.
- [x] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [x] I searched using keywords relevant to my issue to make sure that I am creating a new issue that is not already open (or closed).
- [x] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new and useful enhancement to share.

### Feature Description

Quantizing models you have to choose between IQ6_K and Q6_K_R4 as IQ6_K does not have a row interleaved version.

### Motivation

I think a row interleaved version of IQ6_K would be helpful as IQ6_K has a nice quality improvement over #130.

### Possible Implementation

I'm not sure if 4 or 8 rows would be better.

---

#### 💬 Conversation

👤 **ikawrakow** commented the **2025-03-27** at **07:26:10**:<br>

Using lookup tables with more than 16 entries is a nightmere on `AVX2`. If `N` is the size of the lookup table, for `N > 16` it requires `N/16` shuffles, and `N/16-1` blends of the `N/16` shuffle results. Not sure what the Intel engineers were thinking when they specified the shuffle instructions that way. I did `IQ5_K` (2 shuffles, 1 blend), but for `IQ6_K` it becomes 4 shuffles and 3 blends. I think any benefit one may have from the interleaving will go away. That's why I didn't do `IQ6_K_R4`.

---

👤 **saood06** commented the **2025-03-27** at **07:32:12**:<br>

>Using lookup tables with more than 16 entries is a nightmere on AVX2. If N is the size of the lookup table, for N > 16 it requires N/16 shuffles, and N/16-1 blends of the N/16 shuffle results. Not sure what the Intel engineers were thinking when they specified the shuffle instructions that way. I did IQ5_K (2 shuffles, 1 blend), but for IQ6_K it becomes 4 shuffles and 3 blends. I think any benefit one may have from the interleaving will go away.

Is the situation better with AVX-512 (as Zen 4 and newer Intel is) or NEON?

> That's why I didn't do `IQ6_K_R4`.

Thank you for the explanation.

---

👤 **ikawrakow** commented the **2025-03-27** at **07:42:54**:<br>

`NEON` is OK up to 6 bits because the shuffle instruction there allows up to 64 entries in the lookup table.

On `Zen4` one may do better using masked instructions, but I haven't taken the time to investigate.

---

👤 **saood06** commented the **2025-03-27** at **11:16:34**:<br>

Another question on the interleaved quants, do you mind explaining when 8 (or 16 in the case of BF16_R16) rows can be used beneficially, since you applied that to a few quants such as IQ4_XS and Q8_K, Q8_0. Would an IQ4_K_R8 be better than the  IQ4_K_R4 that exists?

---

👤 **ikawrakow** commented the **2025-03-27** at **11:50:39**:<br>

It depends on how many vector registers are available and how much bit twiddling one needs to do to unpack the quants into `int8` for multiply-adds with the activations. Zen4 (or in general, `AVX512`) has a big advantage here with 32 vector registers of 512 bits (so 4X the amount of data compared to what one can store in vector registers on `AVX2`). `NEON` also has 32 registers but they are 128 bits, so same total amount as `AVX2`.  It is difficult to predict in advance if going to 8 interleaved rows will be beneficial. On Zen4 it will be most of the time, but on `AVX2` or `NEON` it is hard to tell. Hence, one needs to implement and see what happens. But implementing takes time, so I didn't feel I wanted to spend the time to try it for all quantization types.

In mainline they also do row interleaving now for a small group of quantization types. They have decided to tie it to the backend (i.e., each new repacked quant becomes a new backend). The advantage of doing this is that one does not need new quantization types as it is the case here. But there are many disadvantages as well. For one, `mmap` is no longer an option. Then, from usability point of view, it is kind of stupid to be spending the time to repack each time one is loading the model. This is why I didn't take that route. But with additional quantization types it becomes a nightmare to maintain multiple types for the same quant (if, for instance, one wanted to have 8 interleaved rows on Zen4 but 4 on `AVX2`). Hence, to change from 4 to 8, all platforms need to benefit, so we are where we are.

---

👤 **saood06** commented the **2025-03-27** at **12:30:21**:<br>

> It depends on how many vector registers are available and how much bit twiddling one needs to do to unpack the quants into `int8` for multiply-adds with the activations. Zen4 (or in general, `AVX512`) has a big advantage here with 32 vector registers of 512 bits (so 4X the amount of data compared to what one can store in vector registers on `AVX2`). `NEON` also has 32 registers but they are 128 bits, so same total amount as `AVX2`. It is difficult to predict in advance if going to 8 interleaved rows will be beneficial. On Zen4 it will be most of the time, but on `AVX2` or `NEON` it is hard to tell. Hence, one needs to implement and see what happens. But implementing takes time, so I didn't feel I wanted to spend the time to try it for all quantization types.

I see. If you do attempt more of them I'd test them as I find them interesting like I remember that the R4 I tested was worse in almost every way to the R8 that replaced it except at a batch size of 4 (it was worse at every other batch size and also peaked at a lower number of batches and with less throughput). I may end up adding a plot.py to batched bench as well since the tables are a bit hard to read (especially since you have to do math to find out that `-pps` was turned on).

> In mainline they also do row interleaving now for a small group of quantization types.

Yes I saw, was going to mention it in your discussion comparing to llama.cpp.

>They have decided to tie it to the backend (i.e., each new repacked quant becomes a new backend). The advantage of doing this is that one does not need new quantization types as it is the case here. But there are many disadvantages as well. For one, `mmap` is no longer an option. Then, from usability point of view, it is kind of stupid to be spending the time to repack each time one is loading the model.This is why I didn't take that route. 

Yes those tradeoffs are the reason I manually pack my quants instead of using -rtr to do that even here.

>But with additional quantization types it becomes a nightmare to maintain multiple types for the same quant (if, for instance, one wanted to have 8 interleaved rows on Zen4 but 4 on `AVX2`). Hence, to change from 4 to 8, all platforms need to benefit, so we are where we are.

That makes sense, thank you. If you want to close this then that is fine.
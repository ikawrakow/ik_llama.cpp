## 🔀 [Pull Request #617](https://github.com/ikawrakow/ik_llama.cpp/pull/617) - Fixup kimi-k2 convert indentation

| **Author** | `ubergarm` |
| :--- | :--- |
| **State** | 🔀 **Merged** |
| **Source Branch** | `ug/kimi-k2-convert-fixup` |
| **Target Branch** | `main` |
| **Created** | 2025-07-16 |
| **Updated** | 2025-07-16 |
| **Merged** | 2025-07-16 |

---

## 📄 Description

Fixup a copy-paste python indent bug on the convert_hf_to_gguf.py script for kimi-k2-instruct. Thanks @anikifoss for testing and if you have success let me know here to confirm this patch is good.

https://github.com/ikawrakow/ik_llama.cpp/pull/612#issuecomment-3076684820

---

## 💬 Conversation

👤 **anikifoss** commented on **2025-07-16** at **13:10:21**

Still running, 8 hours later at 50%. There is `attn_kv_b` in the output GGUF.

Why do you need `attn_kv_b` anyway?

---

👤 **ikawrakow** approved this pull request ✅ on **2025-07-16** at **13:24:15**

---

👤 **ubergarm** commented on **2025-07-16** at **13:30:08**

> Still running, 8 hours later at 50%. There is `attn_kv_b` in the output GGUF.
> 
> Why do you need `attn_kv_b` anyway?

@anikifoss 

Thanks for running this long job and testing!

Check here for some more info: https://github.com/ikawrakow/ik_llama.cpp/issues/601#issuecomment-3070185792

Based on that discussion I've changed my recipes a bit for Kimi and future deepseek models.

---

👤 **anikifoss** commented on **2025-07-16** at **13:38:55**

Thanks, you already pointed to that PR. Looks like it's for imatrix. There is so much activity I'm having hard time keeping up :sweat_smile:

---

👤 **ubergarm** commented on **2025-07-16** at **14:25:05**

@anikifoss 

Ooops, I'm so scattered sometimes! I've been trying to understand more clearly myself as well!

## tl;dr;

given you use q8_0 for all attn in your quants, it probably doesn't matter to you much. haha... also i think you are the source of `-ot attn_kv_b=CPU` maybe? I thought I tried that once and it wasn't running for me, but other people using it. Maybe it depends on which `-mla ` you're using? I only use 3 now since it got CUDA support.

## ramblings

The reason I pointed at that comment again is this specific bit, regarding "Why do you need attn_kv_b anyway":

> This gives you imatrix data for the wk_b and wv_b tensors, which is good. It is good because these two get used for TG, so you want them quantized with fewer bits if possible. If wkv_b is added to the GGUF, it should be quantized with Q8_0. If it is not added, ik_llama.cpp will (nearly) losslessly create wkv_b tensors as Q8_0 from wk_b and wv_b while loading the model. wkv_b being Q8_0 is fine because tit only gets used for PP, so the more bits don't matter for performance. -ik

Also from https://github.com/ikawrakow/ik_llama.cpp/discussions/477#discussioncomment-13733184

> The attn_k_b and attn_v_b tensors get used for TG. The attn_kv_b tensors that ik_llama.cpp creates on-the-fly are used for PP (when MLA = 2, 3). To avoid potential accuracy loss due to re-quantization, the attn_kv_b tensors get created as Q8_0. -ik

Also this discussion on MLA and comments: https://github.com/ikawrakow/ik_llama.cpp/discussions/354#discussioncomment-13054586

There is a little bit about it too in one of the original mainline MLA PRs by fairydreaming which is was not merged, but possibly a bit more similar to how it is done here psure: https://github.com/ggml-org/llama.cpp/pull/11446

So all that to say my limited understanding of having the both the `attn_kv_b` allow this fork to use "the best of both worlds" for `-mla 3` which uses:
* q8_0 attn_kv_b tensor for PP (its fine to be big given PP is CPU bound)
* quantized attn_k_b attn_v_b tensors (preferably with correct imatrix for lower bpws) for TG (memory bound so smaller size is faster)

But yeah as you use q8_0 for all of it, probably not a bit deal on your quants and also why mainline uses q8_0 for all that as compilades new imatrix/gguf stuff that properly handles those tensors is not yet merged.

My latest recipes I've been leaving attn_kv_b at q8_0 now and only quantizing attn_k_b and attn_v_b.  Unfortunately though, attn_k_b is not divisible by 256 so I'm stuck with q5_0 or iq4_nl.

I hope this is somewhat accurate :sweat_smile:

---

👤 **ikawrakow** commented on **2025-07-16** at **14:43:46**

> I hope this is somewhat accurate

It is. Basically, you don't need to have the `attn_kv_b` tensors to create imatrix data and a good quantized model for `ik_llama.cpp`. The only potential benefit from having `attn_kv_b` in the GGUF is that then these tensors becomes part of the contiguously allocated (or mmap'ed) tensor data storage, while if they are not present in the GGUF, memory is allocated separately for them (but still on the same device that stores the corresponding `attn_k` and `attn_v` tensors). Considering how sensitive the big NUMA systems are to the way the tensors are stored in RAM, this may have some performance implications. But nobody has studied this effect in detail yet, so we don't really know.

---

👤 **saood06** commented on **2025-07-16** at **20:13:58**

>Considering how sensitive the big NUMA systems are to the way the tensors are stored in RAM, this may have some performance implications. But nobody has studied this effect in detail yet, so we don't really know.

I did some direct comparisons a long while back, and there was a measurable (but small) impact on my system (and this was with q8_0 attn tensors which matches the size they are created at if not present). So I can say that when it comes to my system it matters enough to be measured.
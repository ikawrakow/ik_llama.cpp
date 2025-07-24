### 📝 [#432](https://github.com/ikawrakow/ik_llama.cpp/issues/432) - Refactor: GGUF v14 broke compatibility with IQx_KS quants

| **Author** | `Nexesenex` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-05-18 |
| **Updated** | 2025-05-19 |

---

#### Description

### Background Description

Example :

Loading Text Model: E:\text-generation-webui\models\calme-3.3-llamaloi-3b.Q8_0-iMat-IQ4_KS.gguf
gguf_init_from_file_impl: tensor 'blk.0.attn_norm.weight' has offset 272564480, expected 272560384
gguf_init_from_file_impl: failed to read tensor data
Traceback (most recent call last):
  File "Q:\GitHub\croco.cpp\koboldcpp.py", line 8505, in <module>
    main(launch_args=parser.parse_args(),default_args=parser.parse_args([]))
  File "Q:\GitHub\croco.cpp\koboldcpp.py", line 7419, in main
    kcpp_main_process(args,global_memory,using_gui_launcher)
  File "Q:\GitHub\croco.cpp\koboldcpp.py", line 7859, in kcpp_main_process
    loadok = load_model(modelname)
  File "Q:\GitHub\croco.cpp\koboldcpp.py", line 1965, in load_model
    ret = handle.load_model(inputs)
OSError: exception: access violation reading 0x0000000000000008

(Croco.cpp is my fork of KoboldCPP, itself based on Llama.cpp mainline, with some additions merged from IK_LLama, notably the IQ_K Quants.

The GGUF format evolved quite a lot, and since rev14, some flexibility of use might have been tightened by JG, breaking compatibility with the IQx_KS quants, possibly due to the template introduced in https://github.com/ikawrakow/ik_llama.cpp/pull/45  .

I know it's not related to IK_Llama.cpp per-se, rather with mainline, but I don't expect mainline to make any move to maintain even GGUF compatibility with IK_Llama.cpp's quants despite all the work you authored for mainline. It's.. frustrating and disappointing, to put it mildly.

So, it's either up to JG, either up to you, IK.

### Possible Refactor Approaches

Well, I tried to check that by myself when GGUF v14 was out, where was the introduced limitation provoking the problem with the memory offset, but it's beyond what I can remotely spot and fix by myself in a reasonable amount of trial and error.

---

#### 💬 Conversation

👤 **ikawrakow** commented the **2025-05-18** at **15:07:29**:<br>

#45 in this repository or a PR somewhere else?

What is GGUF v14 anyway and why should we care about it here?

---

👤 **Nexesenex** commented the **2025-05-18** at **15:21:31**:<br>

Yes, PR 45 in the IK Llama repo.

Since the 14th revision of the GGUF format went out on mainline, it seems that some screws got tightened.

https://github.com/ggml-org/llama.cpp/pull/11030

Maybe one of those 2 "restrictions" :
```
- Restricted the key general.alignment to uint32_t and powers of 2. On master this key can be set to other types (allowing users to write a file that then causes an error on read) and other values (which don't work correctly with GGML_PAD). There is now a macro GGUF_KEY_GENERAL_ALIGNMENT since this key has a special meaning.
- If user code tries to call gguf_get_arr_data on a string array an error is raised. On master this returns a pointer of type gguf_str, a type defined in ggml.c. I would consider this a misuse of the API.
```

Before that mainline GGUF refactor, I could use all your quants on my KoboldCPP fork after merging your commits (at the time, IQ_K, IQ_KS, and IQ_KT). After that, only the first gen of IQ_K quants (2,3,4,5,6) are functioning on my fork of KCPP, the rest produce offset errors.

You have absolutely no reason to help me on this, except to maintain some relative compatibility between the quantized models produced by IK_LLama and a fork of mainline implementing the IK quants.
But I understand perfectly that you most likely will not want to waste your time trying to fix compatibility with some - potentially adverse - or at least factually incompatible mainline coding and refactoring which is unrelated to IK_LLama.

I just wanted to point out what happened, because I spent a few hours trying to figure this out a few months ago before giving up, and deciding to follow the mainline move to avoid a growing merge-hell later on.

---

👤 **Nexesenex** commented the **2025-05-18** at **15:21:31**:<br>

Yes, PR 45 in the IK Llama repo.

Since the 14th revision of the GGUF format went out on mainline, it seems that some screws got tightened.

https://github.com/ggml-org/llama.cpp/pull/11030

Maybe one of those 2 "restrictions" :
```
- Restricted the key general.alignment to uint32_t and powers of 2. On master this key can be set to other types (allowing users to write a file that then causes an error on read) and other values (which don't work correctly with GGML_PAD). There is now a macro GGUF_KEY_GENERAL_ALIGNMENT since this key has a special meaning.
- If user code tries to call gguf_get_arr_data on a string array an error is raised. On master this returns a pointer of type gguf_str, a type defined in ggml.c. I would consider this a misuse of the API.
```

Before that PR, I could use all your quants (at the time, IQ_K, IQ_KS, and IQ_KT). After that, only the first gen of IQ_K quants (2,3,4,5,6) are functioning, the rest produce offset errors.

You have absolutely no reason to help me on this, except to maintain some relative compatibility between the quants produced by IK_LLama and a fork of mainline implementing the IK quants.
But I understand perfectly that you most likely will not want to waste your time trying to fix compatibility with some - potentially adverse - or at least factually incompatible mainline coding and refactoring which is unrelated to IK_LLama.

I just wanted to point out what happened, because I spent a few hours trying to figure this out a few months ago before giving up, and deciding to follow the mainline move to avoid a growing merge-hell later on.

---

👤 **ikawrakow** commented the **2025-05-18** at **15:44:53**:<br>

@Nexesenex 

It is because of this code block
```c++
        {
            ok = ok && gr.read(info.t.type);

            // check that tensor type is within defined range
            if (info.t.type < 0 || info.t.type >= GGML_TYPE_COUNT) {
                fprintf(stderr, "%s: tensor '%s' has invalid ggml type %d (%s)\n",
                    __func__, info.t.name, info.t.type, ggml_type_name(info.t.type));
                ok = false;
                break;
            }
            const size_t  type_size = ggml_type_size(info.t.type);
            const int64_t blck_size = ggml_blck_size(info.t.type);

            // check that row size is divisible by block size
            if (blck_size == 0 || info.t.ne[0] % blck_size != 0) {
                fprintf(stderr, "%s: tensor '%s' of type %d (%s) has %" PRId64 " elements per row, "
                    "not a multiple of block size (%" PRId64 ")\n",
                    __func__, info.t.name, (int) info.t.type, ggml_type_name(info.t.type), info.t.ne[0], blck_size);
                ok = false;
                break;
            }

            // calculate byte offsets given the tensor shape and type
            info.t.nb[0] = type_size;
            info.t.nb[1] = info.t.nb[0]*(info.t.ne[0]/blck_size);
            for (int j = 2; j < GGML_MAX_DIMS; ++j) {
                info.t.nb[j] = info.t.nb[j - 1]*info.t.ne[j - 1];
            }
        }
        if (!ok) {
            break;
        }

        // tensor data offset within buffer
        ok = ok && gr.read(info.offset);

        ctx->info.push_back(info);
    }
```

I had the concept that a GGUF is a general storage format for LLM models and similar. With that block it isn't. It wants the data type type to be one of the data types in `ggml`, so clearly does not work to store anything else. But even if the data type is a `ggml` type (as it is in your fork), it still uses the faulty assumption that the tensor row size is going to be determined by the block size, type size, and number of elements in the row. That is a bug. There is the function `ggml_row_size(enum ggml_type type, int64_t nelemenets)`, which is supposed to be used instead of the above code. But yes, the same mistake can be found many times over in the CUDA code. Unless there are other assumtions such as these, you can fix it by replacing the line
```c++
info.t.nb[1] = info.t.nb[0]*(info.t.ne[0]/blck_size);
```
with
```c++
info.t.nb[1] = ggml_row_size(info.t.type, info.t.ne[0]);
```
Let me know how it goes.

@JohannesGaessler  FYI

---

👤 **JohannesGaessler** commented the **2025-05-18** at **16:28:45**:<br>

On the mainline repository the implementation is

```C
size_t ggml_row_size(enum ggml_type type, int64_t ne) {
    assert(ne % ggml_blck_size(type) == 0);
    return ggml_type_size(type)*ne/ggml_blck_size(type);
}
```

Doing this calculation manually can be seen as a defect but it only manifests as a bug if `ggml_row_size` is modified as was presumably done for this fork. I will accept PRs to fix such defects on mainline.

---

👤 **ikawrakow** commented the **2025-05-18** at **16:38:18**:<br>

> On the mainline repository the implementation is

Yes, this is the current implementation. But that implementation can change, and that's why there is the `ggml_row_size` function that has been around for quite some time. It has nothing to do with forks. It can also change in mainline, and then one wouldn't want to go and hunt down all places in the code where `ne*ts/bs` is used. 

@Nexesenex has a simple fix that I suggested above. Mainline can keep it the way it is, or change it. That's up to you and the other mainline maintainers.

---

👤 **JohannesGaessler** commented the **2025-05-18** at **16:52:34**:<br>

Yes, I agree that it's better to use `ggml_row_size`. If I write new code or touch existing code I will replace it as appropriate. It's a defect. But as there are no inputs that can provoke incorrect results on the mainline repository this defect is not manifesting as a bug and it is fairly low-priority. If this issue is of higher priority for someone else they will need to go through the code and fix the defect where applicable themself.

---

👤 **Nexesenex** commented the **2025-05-18** at **17:43:13**:<br>

@ikawrakow : it works. Tyvm!

@JohannesGaessler : now that the issue you yourself acknowledged as a defect has been elucidated, maybe, just maybe it would be simpler to fix it in mainline and be done with it while it's fresh?

I might seem obnoxious, but.. what matters above all other considerations is that things are working, especially when the ratio result/effort is high for a skilled dev like you, and I guess IK's work deserve the courtesy of not being made unworkable on mainline and its forks out of mere coding orthodoxy, especially considering that his former gens of quants are still one of the backbone of the mainline project, and that his new ones are SOTA, simple and straight.
A mainline project which, by the way, sorely misses the new ones and drifts slowly, quant-wise, towards.. how to put it? "belatedness", maybe, considering the price of the hardware, and the colossal storage, bandwidth, and compute taken by obsolete GGUF quantizations still produced nowadays?
It's a no-brainer, really.

Note : I speak on my own and sole behalf, but I needed to say this.

---

👤 **ikawrakow** commented the **2025-05-19** at **14:11:02**:<br>

@Nexesenex I think I can close this now.

---

👤 **Nexesenex** commented the **2025-05-19** at **15:03:12**:<br>

Yep. Thank again, @ikawrakow.
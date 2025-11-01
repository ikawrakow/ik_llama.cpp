## 🔀 [Pull Request #298](https://github.com/ikawrakow/ik_llama.cpp/pull/298) - Update gguf-py constants

| **Author** | `saood06` |
| :--- | :--- |
| **State** | 🔀 **Merged** |
| **Source Branch** | `s6/fix_python` |
| **Target Branch** | `main` |
| **Created** | 2025-03-31 |
| **Updated** | 2025-04-24 |
| **Merged** | 2025-04-24 |

---

## 📄 Description

As reported in [#297](https://github.com/ikawrakow/ik_llama.cpp/issues/297) the constants.py file needs to be updated. 

Testing the command that errored it now gets further.

Command: `python gguf-py/scripts/gguf_dump.py --markdown /mnt/sda/DeepSeek-V3-0324-IQ4_K_R4.gguf`

```
Traceback (most recent call last):
  File "/home/saood06/ik_main/ik_llama.cpp/gguf-py/scripts/gguf_dump.py", line 454, in <module>
    main()
    ~~~~^^
  File "/home/saood06/ik_main/ik_llama.cpp/gguf-py/scripts/gguf_dump.py", line 439, in main
    reader = GGUFReader(args.model, 'r')
  File "/home/saood06/ik_main/ik_llama.cpp/gguf-py/gguf/gguf_reader.py", line 130, in __init__
    self._build_tensors(offs, tensors_fields)
    ~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^
  File "/home/saood06/ik_main/ik_llama.cpp/gguf-py/gguf/gguf_reader.py", line 278, in _build_tensors
    block_size, type_size = GGML_QUANT_SIZES[ggml_type]
                            ~~~~~~~~~~~~~~~~^^^^^^^^^^^
KeyError: <GGMLQuantizationType.IQ5_K_R4: 340>
```

This is because GGML_QUANT_SIZES ([code](https://github.com/ikawrakow/ik_llama.cpp/blob/4819257ce66a680608cf9c7871156041d00eb7da/gguf-py/gguf/constants.py#L1292)) still needs to be updated, not sure of the values for the new quant types. @ikawrakow could you give me a hint at how to update this?

- [x] I have read the [contributing guidelines](https://github.com/ggerganov/llama.cpp/blob/master/CONTRIBUTING.md)
- Self-reported review complexity:
  - [X] Low
  - [ ] Medium
  - [ ] High

---

## 💬 Conversation

👤 **ikawrakow** commented on **2025-03-31** at **08:03:08**

> could you give me a hint at how to update this?

Search in `ggml-common.h` for the quantization types missing in `constants.py` and use the static asserts in `ggml-common.h` to see the expected size. Alternatively, the `type_traits` structure in `ggml.c` defines everything needed in `contants.py` in one place.

The python stuff is in desperate need of sync with mainline. But the difference is now so large that one needs time and focus to merge the changes. Alternatively, one just copies over everything python script related from mainline and adds the few changes that I have made. IIRC, the changes I made were related to Bitnet models, and more recently the MLA stuff for DeepSeek models (but one may consider removing that as the additional tensors can be generated on-the-fly when loading the model).

---

👤 **saood06** commented on **2025-03-31** at **09:07:46**

> > could you give me a hint at how to update this?
> 
> Search in `ggml-common.h` for the quantization types missing in `constants.py` and use the static asserts in `ggml-common.h` to see the expected size. Alternatively, the `type_traits` structure in `ggml.c` defines everything needed in `contants.py` in one place.
> 

Thanks, I see what I need to do.

> (but one may consider removing that as the additional tensors can be generated on-the-fly when loading the model).

I'm still testing the performance implications of that on my system, it seems like it may have mattered.

---

👤 **saood06** commented on **2025-03-31** at **09:10:53**

>The python stuff is in desperate need of sync with mainline.

What went wrong with the Gemma changes, I noticed you reverted grabbing them and said to use mainline for conversions. The deepseek associated stuff including the MLA changes to the python were all grabbed when I ported it over I think.

This GGML_QUANT_SIZES is the only thing I know that is missing besides the Gemma stuff, is there anything else. If there is I can look into it.

---

👤 **ikawrakow** commented on **2025-03-31** at **09:15:43**

> What went wrong with the Gemma changes

It wasn't working. I copy-pasted the Gemma3 portion, but it started throwing exceptions. I didn't spend the time to understand why and fix it.

---

👤 **saood06** commented on **2025-04-24** at **04:23:34**

@ikawrakow 

Thanks for the hint. I was able to update GGML_QUANT_SIZES and this should be ready for review now.


Running `python gguf-py/scripts/gguf_dump.py --markdown /mnt/sda/DeepSeek-V3-0324-IQ4_K_R4.gguf`  works now. Output of the command attached below.

[gguf_dump1.md](https://github.com/user-attachments/files/19884332/gguf_dump1.md)

Edit: Something seems wrong with I2_S, trying to use dump the model it runs into this error.

```
  File "/home/saood06/ik_main/ik_llama.cpp/gguf-py/gguf/gguf_reader.py", line 130, in __init__
    self._build_tensors(offs, tensors_fields)
    ~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^
  File "/home/saood06/ik_main/ik_llama.cpp/gguf-py/gguf/gguf_reader.py", line 325, in _build_tensors
    data = self._get(data_offs, item_type, item_count).reshape(np_dims),
           ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^
ValueError: cannot reshape array of size 17425632 into shape (2560,6912)
```

I added a print statement to see which tensor it was hitting and all it's values:

```
Processing tensor: blk.9.ffn_down.weight
  dims: [6912, 2560]
  raw_dtype: [36]
  ggml_type: I2_S
  n_elems: 17694720
  np_dims: (2560, 6912)
  block_size: 1, type_size: 1
  n_bytes: 17694720
  data_offs: 1827046400
  item_count: 17694720
  item_type: <class 'numpy.uint8'>
```

Interestingly enough  the `iq2_bn_r4` and `iq2_bn` converted version does not error and I can gguf-dump them.

---

👤 **ikawrakow** approved this pull request ✅ on **2025-04-24** at **05:33:08**
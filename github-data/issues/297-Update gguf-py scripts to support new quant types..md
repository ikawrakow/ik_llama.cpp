### 📝 [#297](https://github.com/ikawrakow/ik_llama.cpp/issues/297) - Update gguf-py scripts to support new quant types.

| **Author** | `ubergarm` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-03-31 |
| **Updated** | 2025-04-24 |

---

#### Description

This is more of a convenience and lower priority. I wanted to print out some info with `gguf_dump.py` but looks like possibly just need to add latest quant enum constants into `GGMLQuantizationType` etc...

Here is how to recreate:
```bash
$ cd ik_llama.cpp
$ uv venv ./venv --python 3.12 --python-preference=only-managed
$ source venv/bin/activate
$ uv pip install 'numpy<2.0.0' sentencepiece pyyaml
$ python gguf-py/scripts/gguf_dump.py --markdown /mnt/raid/models/ubergarm/DeepSeek-V3-0324-GGUF/DeepSeek-V3-0324-Q8_0_R8.gguf

Traceback (most recent call last):
  File "/home/w/projects/ik_llama.cpp/gguf-py/scripts/gguf_dump.py", line 454, in <module>
    main()
  File "/home/w/projects/ik_llama.cpp/gguf-py/scripts/gguf_dump.py", line 439, in main
    reader = GGUFReader(args.model, 'r')
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/w/projects/ik_llama.cpp/gguf-py/gguf/gguf_reader.py", line 130, in __init__
    self._build_tensors(offs, tensors_fields)
  File "/home/w/projects/ik_llama.cpp/gguf-py/gguf/gguf_reader.py", line 275, in _build_tensors
    ggml_type = GGMLQuantizationType(raw_dtype[0])
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/w/.local/share/uv/python/cpython-3.12.9-linux-x86_64-gnu/lib/python3.12/enum.py", line 751, in __call__
    return cls.__new__(cls, value)
           ^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/w/.local/share/uv/python/cpython-3.12.9-linux-x86_64-gnu/lib/python3.12/enum.py", line 1165, in __new__
    raise ve_exc
ValueError: 208 is not a valid GGMLQuantizationType
```

Maybe me or @saood06 will take a look at it eventually. Just recording it here now before I forget.

Thanks!

---

#### 💬 Conversation

👤 **saood06** commented the **2025-04-24** at **05:55:57**:<br>

@ubergarm 

#298 is now merged in which addressed it.

---

👤 **ubergarm** commented the **2025-04-24** at **14:35:23**:<br>

Sweet! Appreciate the update and confirming gguf dump works now with your `V3-0324-IQ4_K_R4` quant!
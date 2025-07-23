### 🐛 [#605](https://github.com/ikawrakow/ik_llama.cpp/issues/605) - Bug: IQ3_KS missing from GGMLQuantizationType - gguf_reader.py script cannot process IQ3_KS tensors

| **Author** | `Thireus` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-07-13 |
| **Updated** | 2025-07-13 |

---

#### Description

### What happened?

The https://github.com/ikawrakow/ik_llama.cpp/blob/e2b1a5e1fcb3ad55eae03c58c986a21e842ff7a4/gguf-py/gguf/gguf_reader.py script cannot process `IQ3_KS` tensors because this type is missing from `GGMLQuantizationType`:

### Name and Version

https://github.com/ikawrakow/ik_llama.cpp/blob/e2b1a5e1fcb3ad55eae03c58c986a21e842ff7a4/gguf-py/gguf/constants.py#L1265

### What operating system are you seeing the problem on?

_No response_

### Relevant log output

```shell
  File "/home/thireus/AI/venv/lib/python3.11/site-packages/gguf/gguf_reader.py", line 130, in __init__
    self._build_tensors(offs, tensors_fields)
  File "/home/thireus/AI/venv/lib/python3.11/site-packages/gguf/gguf_reader.py", line 275, in _build_tensors
    ggml_type = GGMLQuantizationType(raw_dtype[0])
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3.11/enum.py", line 717, in __call__
    return cls.__new__(cls, value)
           ^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3.11/enum.py", line 1133, in __new__
    raise ve_exc
ValueError: 156 is not a valid GGMLQuantizationType
```
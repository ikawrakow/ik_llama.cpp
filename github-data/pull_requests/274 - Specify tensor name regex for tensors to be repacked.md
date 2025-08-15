### 🔀 [#274](https://github.com/ikawrakow/ik_llama.cpp/pull/274) - Specify tensor name regex for tensors to be repacked

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-03-21 |
| **Updated** | 2025-03-21 |

---

#### Description

This PR follows in the footsteps of #272 and adds the ability to specify one or more regular expressions to use for matching tensor names to be repacked. This is useful for hybrid GPU/CPU inference where one will want to repack only the tensors that stay on the CPU.

Usage
```
./bin/llama-quantize --repack --repack-pattern regex1,regex2,... some_model output_file_name quant_type
```

E.g., if one uses tensor override `-ot exps=CPU` for inference to have the DeepSeek MoE experts stay on the CPU, one would use
```
./bin/llama-quantize --repack --repack-pattern exps some_model output_file_name quant_type
```
to repack an existing model.
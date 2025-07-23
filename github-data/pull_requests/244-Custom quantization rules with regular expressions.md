### 🔀 [#244](https://github.com/ikawrakow/ik_llama.cpp/pull/244) - Custom quantization rules with regular expressions

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-03-06 |
| **Updated** | 2025-03-07 |

---

#### Description

For DeepSeekV3/R1 it is handy to be able to define custom rules for picking quantization types for the various tensors. Well, this is useful in general, but particularly useful for very large models where one wants to squeeze the last bit of quantized model quality for the smallest possible model size.

This PR adds this ability. Using

```
./bin/llama-quantize --imatrix some_imatrix --custom-q "regex1=typ1,regex2=type2..." some_model some_output_file some_base_quant
```
one can pass custom rules to the quantization function. The rules are comma separated (but one can also use multiple `--custom-q` arguments). The custom rules are processed in order and the first match is taken. So, for instance, if I use
```
--custom-q "\.ffn_down_exps\.weight=iq4_nl,\.ffn_.*_exps\.weight=iq1_s_r4"
```
the second rule matches the `ffn_down` experts, but because a match was found in the first rule, `IQ4_NL` will get used for `blk.*.ffn_down_exps.weight`, and `IQ1_S_R4` will get used for the `ffn_up` and `ffn_gate` experts tensors. 

To summarize how the quantization type is determined:
1. The type is set to the quantization type specified on the command line as last argument
2. If there are rules added via `--attn-q-type, --attn-k-type, --attn-v-type, --attn-qkv-type, --attn-output-type, --ffn-gate-type,  --ffn-down-type, --ffn-up-type`, and the tensor is one of those, the type specified that way gets used (for now)
3. Else, the built-in rules get applied.
4. If there are custom rules provided and the tensor name matches one of the regular expressions in the custom rules, the type specified in the first match found becomes the selected quantization type for the tensor, retrospectively of what might have happened in steps 1-3.
5. If the tensor row size is not a multiple of the block size of the type selected in 1-4, the type is overridden with a built-in rule that maps quants with bock sizes > 32 to one of the quants with block size 32.

---

#### 💬 Conversation

👤 **davidsyoung** commented the **2025-03-06** at **17:58:36**:<br>

This is awesome. It’ll come in really useful!
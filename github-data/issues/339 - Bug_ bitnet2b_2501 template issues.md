### 🐛 [#339](https://github.com/ikawrakow/ik_llama.cpp/issues/339) - Bug: bitnet2b_2501 template issues

| **Author** | `saood06` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-04-22 |
| **Updated** | 2025-04-22 |

---

#### Description

### What happened?

The model would not output the EOS token resulting in it endlessly continuing generation, often taking over both user and assistant roles. This is because the attached chat template is wrong. The following example from the transformer's PR is correct as I can get it to function properly using a template derived from it.

`<|begin_of_text|>User: Hey, are you conscious? Can you talk to me?<|eot_id|>Assistant:`

### Name and Version

35691804

### What operating system are you seeing the problem on?

_No response_

### Relevant log output

```shell

```

---

#### 💬 Conversation

👤 **saood06** commented the **2025-04-22** at **07:51:57**:<br>

I think this can actually be closed, the llama_chat_apply_template_internal code looks correct, and I would just need to update the model's GGUF file. I don't use the CLI mode enough to know why it wasn't working there, but now I can get it to function properly in server when I use the correct template.

---

👤 **saood06** commented the **2025-04-22** at **07:51:57**:<br>

I think this can actually be closed, the llama_chat_apply_template_internal code looks correct, and I would just need to update the model's GGUF file. I don't use the CLI mode enough to know why it wasn't working there.
### 🐛 [#281](https://github.com/ikawrakow/ik_llama.cpp/issues/281) - Bug: Strange dips in TG performance

| **Author** | `saood06` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-03-22 |
| **Updated** | 2025-03-23 |

---

#### Description

### What happened?

As mentioned in https://github.com/ikawrakow/ik_llama.cpp/pull/273 I've seen this behavior occur with llama-server (sorry, I never really noted the configurations or models it occurs with), and I can usually mitigate it by canceling and then restarting generation until TG performance goes back to the expected value, the chart below shows this behavior captured in a benchmark.

![Image](https://github.com/user-attachments/assets/3e788edb-c182-40fa-943b-17ab011ee91f)

Also I'm fairly certain I've never encountered this bug in batched-bench only in server and sweep-bench both of which manipulate the KV more than batched-bench.

### Name and Version

Graph capturing this behavior was on https://github.com/ikawrakow/ik_llama.cpp/commit/3d6e25c82db5510df483185b8a20f0ce01136dd7

### What operating system are you seeing the problem on?

Linux

### Relevant log output

```shell

```

---

#### 💬 Conversation

👤 **saood06** commented the **2025-03-23** at **13:11:13**:<br>

Closing via #282 

![Image](https://github.com/user-attachments/assets/728a3265-82e8-4817-9ebf-a8165dc63205)

PP performance for those options:

![Image](https://github.com/user-attachments/assets/533d51dc-cc13-4c19-babd-b88173760e00)

For my primary use case MLA-3 on is the best with nice PP and TG, it seems like though for tasks with very small PP and TG keeping context under 8K MLA-1 off is useful.

Thank you for the quick find and fix.
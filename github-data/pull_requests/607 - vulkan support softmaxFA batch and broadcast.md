## 🔀 [Pull Request #607](https://github.com/ikawrakow/ik_llama.cpp/pull/607) - vulkan: support softmax/FA batch and broadcast

| **Author** | `firecoperana` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Source Branch** | `fcp/vulkan_fa_fix_dsv` |
| **Target Branch** | `main` |
| **Created** | 2025-07-13 |
| **Updated** | 2025-07-16 |
| **Assignees** | `firecoperana` |

---

## 📄 Description

vulkan: support softmax/FA batch and broadcast 
https://github.com/ggml-org/llama.cpp/pull/14449
Fix gibberish output when FA is enabled for some model

The new FA for deepseek MLA PR is missing this, which caused gibberish output in some models.

- [x] I have read the [contributing guidelines](https://github.com/ggerganov/llama.cpp/blob/master/CONTRIBUTING.md)
- Self-reported review complexity:
  - [ ] Low
  - [x] Medium
  - [ ] High

---

## 💬 Conversation

👤 **ubergarm** commented on **2025-07-13** at **19:09:26**

Great, this fixes the gibberish issue we were seeing over on [#598](https://github.com/ikawrakow/ik_llama.cpp/issues/598) when I run with `KHR_coopmat` and `-fa` enabled:
```
ggml_vulkan: 0 = NVIDIA GeForce RTX 3090 Ti (NVIDIA) | uma: 0 | fp16: 1 | warp size: 32 | shared memory: 49152 | int dot: 1 | matrix cores: KHR_coopmat
```

However, on the AMD GPU rig it no longer outputs that same looking gibberish, but now kinda chokes/freezes up around the same point where it used to throw gibberish. Then it very slowly outputs `3333`
```
$ ./build/bin/llama-server --version
version: 3796 (69ab6921)
built with cc (Ubuntu 13.3.0-6ubuntu2~24.04) 13.3.0 for x86_64-linux-gnu

ggml_vulkan: 0 = Radeon RX 7900 XTX (AMD open-source driver) | uma: 0 | fp16: 1 | warp size: 64 | shared memory: 32768 | int dot: 1 | matrix cores: KHR_coopmat

... For example, in French, numbers from  to 10 are all irregular except for 11-16 which333^C
Response cancelled.
```

Also, I get a similar behavior where it starts out okay then goes to `33333` on my nvidia GPU when running with `NV_coopmat2`

```bash
ggml_vulkan: 0 = NVIDIA GeForce RTX 3090 Ti (NVIDIA) | uma: 0 | fp16: 1 | warp size: 32 | shared memory: 49152 | int dot: 1 | matrix cores: NV_coopmat2

...Maybe the user is learning French or needs it for a specific purpose. They might be preparing for a trip, studying, or33333333333333333333333333333333333333333333333333333333333333333333333333333333333^C
Response cancelled.
```

So this PR does seem to fix the NVIDIA `KHR_coopmat` `-fa` enabled path, but not on the NVIDIA `NV_coopmat2` nor AMD `KHR_coopmat` `libvulkan.so (found version "1.4.313")` path.

---

👤 **firecoperana** commented on **2025-07-13** at **23:46:43**

Can you try again?

---

👤 **ubergarm** commented on **2025-07-14** at **01:38:51**

Hey thanks a lot for working on this stuff! I just tried again with dba868a8 with the three cases:

### NVIDIA 3090TI FE
* `KHR_coopmat` is still working okay it seems
* `NV_coopmat2` still glitches out similarly.

### AMD RX 7900 XTX
* `NV_coopmat2` still glitches out

Yeah so seems unchanged with two cases still suddnely outputing just 3 `so cardinal numbers33^C` after about ~225ish tokens into the reply. I have some time tomorrow to test anything else, thanks!

---

👤 **ikawrakow** commented on **2025-07-15** at **06:04:07**

@firecoperana

Is this necessary after [#608](https://github.com/ikawrakow/ik_llama.cpp/issues/608)?

---

👤 **firecoperana** commented on **2025-07-15** at **12:30:20**

Already included in the main.
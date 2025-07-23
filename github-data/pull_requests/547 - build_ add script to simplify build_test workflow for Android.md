### 🔀 [#547](https://github.com/ikawrakow/ik_llama.cpp/pull/547) - build: add script to simplify build&test workflow for Android

| **Author** | `jeffzhou2000` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-06-22 |
| **Updated** | 2025-07-04 |

---

#### Description

- [x] I have read the [contributing guidelines](https://github.com/ggerganov/llama.cpp/blob/master/CONTRIBUTING.md)
- Self-reported review complexity:
  - [x] Low
  - [ ] Medium
  - [ ] High

### purpose

add script to simplify build & test workflow of ik_llama.cpp for Android

---

#### 💬 Conversation

👤 **ikawrakow** submitted a review the **2025-06-23** at **10:05:44**: 🔄 `CHANGES_REQUESTED`

---

👤 **jeffzhou2000** submitted a review the **2025-06-23** at **10:20:45**: 💬 `COMMENTED`

---

👤 **jeffzhou2000** submitted a review the **2025-06-23** at **10:24:07**: 💬 `COMMENTED`

---

👤 **jeffzhou2000** submitted a review the **2025-06-23** at **10:42:21**: 💬 `COMMENTED`

---

👤 **zhouwg** submitted a review the **2025-06-23** at **10:42:21**: 💬 `COMMENTED`

---

👤 **zhouwg** commented during a code review the **2025-06-23** at **10:42:21** on `CMakeLists.txt`:<br>

refined according to your comment, pls take a look if you have time.

---

👤 **zhouwg** submitted a review the **2025-06-23** at **11:19:16**: 💬 `COMMENTED`

---

👤 **zhouwg** commented during a code review the **2025-06-23** at **11:19:16** on `CMakeLists.txt`:<br>

> Your measurements clearly indicate that these are **not the best** compiler settings. 
the potential best compiler settings for ik_llama.cpp on Snapdragon 8Elite might-be:

-march=armv8.7-a+dotprod+fp16

or 

-march=armv8.7-a+dotprod+fp16 -mcpu=cortex-x1 -mtune=cortex-x1

or 


-march=armv8.7-a+dotprod+fp16+i8mm -mcpu=cortex-x1 -mtune=cortex-x1

or 

-march=armv8.7-a+dotprod+fp16+i8mm -mcpu=cortex-x1 -mtune=cortex-x1 -D_GNU_SOURCE -ffp-model=fast -fno-finite-math-only

depend on workload of Android OS, this is my personal opinion, might-be not very exactly correct.

---

👤 **jeffzhou2000** submitted a review the **2025-06-23** at **11:34:06**: 💬 `COMMENTED`

---

👤 **zhouwg** submitted a review the **2025-06-23** at **13:15:25**: 💬 `COMMENTED`

---

👤 **zhouwg** commented during a code review the **2025-06-23** at **13:15:25** on `scripts/build-run-android.sh`:<br>

YES, you are right.

I'm not sure because it's a script to simplify workflow of build ik_llama.cpp on Linux for Android.

I'd like to close this PR accordingly and it doesn't matter.

thanks for your time to review this PR and have a good day.

---

👤 **jeffzhou2000** submitted a review the **2025-07-04** at **09:11:51**: 💬 `COMMENTED`

---

👤 **ikawrakow** submitted a review the **2025-07-04** at **09:16:24**: 💬 `COMMENTED`

---

👤 **jeffzhou2000** submitted a review the **2025-07-04** at **09:18:14**: 💬 `COMMENTED`

---

👤 **zhouwg** commented during a code review the **2025-07-04** at **09:18:14** on `scripts/build-run-android.sh`:<br>

yes, you are absolutely correct: they are totally off-topic in a discussion about new SOTA quantization types in ik_llama.cpp. thanks for your understanding!
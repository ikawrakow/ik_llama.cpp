## 🔀 [Pull Request #547](https://github.com/ikawrakow/ik_llama.cpp/pull/547) - build: add script to simplify build&test workflow for Android

| **Author** | `jeffzhou2000` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Source Branch** | `fix-build-android` |
| **Target Branch** | `main` |
| **Created** | 2025-06-22 |
| **Updated** | 2025-07-04 |

---

## 📄 Description

- [x] I have read the [contributing guidelines](https://github.com/ggerganov/llama.cpp/blob/master/CONTRIBUTING.md)
- Self-reported review complexity:
  - [x] Low
  - [ ] Medium
  - [ ] High

### purpose

add script to simplify build & test workflow of ik_llama.cpp for Android

---

## 💬 Conversation

👤 **jeffzhou2000** commented on **2025-06-22** at **13:11:26**

comparison of llama_bench on Android phone equipped with Qualcomm Snapdragon 8Elite(one of the most advanced mobile SoCs on our planet at the moment) + Android NDK r28(the following benchmark data might-be depend on the workload of Android OS):

1. both build with " -O3 -flto -D_GNU_SOURCE -ffp-model=fast -fno-finite-math-only " 

upstream llama.cpp with latest codes:
llama-bench:
![Screenshot from 2025-06-22 12-58-28](https://github.com/user-attachments/assets/84381046-de4a-4c54-9aac-5c81c04d15e6)
llama-cli:
![Screenshot from 2025-06-22 15-12-04](https://github.com/user-attachments/assets/ac3644a1-0db7-46d2-b4ce-6b8e514bd8ef)

ik_llama.cpp with latest codes:

![Screenshot from 2025-06-22 13-08-16](https://github.com/user-attachments/assets/a2383ac7-617b-46e0-a5ab-ff907c733cb1)

![Screenshot from 2025-06-22 15-09-01](https://github.com/user-attachments/assets/4b2b2aa9-3cae-4e1b-937b-2fe62ac84dc6)

llama-cli(the inference result is incorrect)
![Screenshot from 2025-06-22 15-12-20](https://github.com/user-attachments/assets/db2bc851-84e5-4a20-9de6-b1ede74e1972)


2. both build with " -O3 -march=armv8.2-a+dotprod+fp16 -flto -D_GNU_SOURCE -ffp-model=fast -fno-finite-math-only"

upstream llama.cpp with latest codes:

![Screenshot from 2025-06-22 15-55-05](https://github.com/user-attachments/assets/a65da566-955f-4510-94b4-cb0b1f50dbca)

ik_llama.cpp with latest codes:
![Screenshot from 2025-06-22 15-47-34](https://github.com/user-attachments/assets/cd6d0b39-2c0e-4d07-959e-bfc9d1620ca0)

3. both build with  " -O3 -march=armv8.7-a -mcpu=cortex-x1 -mtune=cortex-x1 -flto -D_GNU_SOURCE  -ffp-model=fast -fno-finite-math-only ".

upstream llama.cpp with latest codes:
![Screenshot from 2025-06-22 16-16-13](https://github.com/user-attachments/assets/6d38c68d-0827-4a44-b84a-cbd1aa3f3412)

ik_llama.cpp with latest codes:
![Screenshot from 2025-06-22 16-22-37](https://github.com/user-attachments/assets/825d3aa6-049f-4a0c-81b0-89f2dad4ba9e)

4. both build with " -O3 -march=armv8.7-a+dotprod+fp16 -mcpu=cortex-x1 -mtune=cortex-x1  -flto -D_GNU_SOURCE -fvectorize -ffp-model=fast -fno-finite-math-only"

upstream llama.cpp with latest codes:
![Screenshot from 2025-06-22 21-47-06](https://github.com/user-attachments/assets/e287d710-cd3a-42ae-bbc1-bf254fbccb64)

![Screenshot from 2025-06-22 17-30-43](https://github.com/user-attachments/assets/96389f4e-8961-4995-9424-e2804ee146d1)

the following is a screenshot when I helped troubleshooting a performance regression issue in the upstream llama.cpp project. as well known, there are so many approved PRs in the upstream llama.cpp project and some approved PRs might-be brings regression issues in the upstream llama.cpp project. sometimes I can't reproduce the same benchmark result with the upstream llama.cpp's latest codes.

![455784182-f30ce0c8-5528-44fe-8be3-213ebaf4e730](https://github.com/user-attachments/assets/bc182761-acd1-4aeb-9da8-8bce36b9e15e)

ik_llama.cpp with latest codes:
![Screenshot from 2025-06-22 21-29-08](https://github.com/user-attachments/assets/3ef41de6-e1bc-4a03-9ea8-a76e2223de51)

![Screenshot from 2025-06-22 17-45-34](https://github.com/user-attachments/assets/00cde394-87f7-4851-bec7-7b27dea9c16d)


----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
after enable GGML_IQK_FLASH_ATTENTION

build with " -O3 -march=armv8.7-a+dotprod+fp16 -mcpu=cortex-x1 -mtune=cortex-x1  -flto -D_GNU_SOURCE -fvectorize -ffp-model=fast -fno-finite-math-only"
![Screenshot from 2025-06-22 18-09-57](https://github.com/user-attachments/assets/0ca053b7-1aa9-4201-8d3b-ea4771b0d636)


build with " -O3 -march=armv8.7-a+dotprod+fp16 -mcpu=cortex-x1 -mtune=cortex-x1  -flto -D_GNU_SOURCE  -ffp-model=fast -fno-finite-math-only"

![Screenshot from 2025-06-22 18-18-55](https://github.com/user-attachments/assets/f6de8cc6-03b2-4955-bb8a-f4877c3b9226)

build with " -O3 -march=armv8.7-a -mcpu=cortex-x1 -mtune=cortex-x1 -flto -D_GNU_SOURCE -ffp-model=fast -fno-finite-math-only "

![Screenshot from 2025-06-22 18-24-55](https://github.com/user-attachments/assets/f388d84e-59e3-48c1-aacb-bfd25c06449c)

build failed with " -O3 -march=armv8.7-a -flto -D_GNU_SOURCE -ffp-model=fast -fno-finite-math-only "

build with " -O3 -march=armv8.7-a+dotprod+fp16 -flto -D_GNU_SOURCE -ffp-model=fast -fno-finite-math-only"
![Screenshot from 2025-06-22 18-33-45](https://github.com/user-attachments/assets/d92f5f3f-283b-46ee-98cc-472d3f968a65)

build with " -O3 -march=armv8.2-a+dotprod+fp16 -flto -D_GNU_SOURCE -ffp-model=fast -fno-finite-math-only"
![Screenshot from 2025-06-22 18-46-51](https://github.com/user-attachments/assets/1d4e6165-3ef6-4c2a-9525-20123f381880)


build with "-O3 -flto -D_GNU_SOURCE -ffp-model=fast -fno-finite-math-only"
![Screenshot from 2025-06-22 18-56-27](https://github.com/user-attachments/assets/8bccc65c-90c8-4382-bb47-0dc9e115eca4)



in my opinion/personal perspective, the upstream llama.cpp can get much performance gains from optimization of Google's state-of-the-art toolchain(as well known, there are many top world-class compiler experts and engineers in Google). at the same time, the hand-written codes in this project runs faster than the neon codes in the upstream.

---

👤 **ikawrakow** started a conversation on `CMakeLists.txt` on **2025-06-23** at **10:02:36**

Your measurements clearly indicate that these are **not the best** compiler settings. Apart from not being best for `ik_llama.cpp`, there are a lot of Android phones out there that only support `armv8.2-a`, which is the minimum required for `ik_llama.cpp` to build correctly.

More generally, `ik_llama.cpp` allows to manually set `GGML_ARCH_FLAGS`, exactly for the purpose of building on Android when the compiler for whatever reason does not use correct settings with `GGML_NATIVE`.

> 👤 **jeffzhou2000** replied on **2025-06-23** at **10:24:07**
> 
> yes, you are right.
> 
> 1. I have tried GGML_ARCH_FLAGS in the CMakeLists.txt and it works fine as expected although the GGML_NATIVE can't works for Android.
> 2. I can remove this changes in the toplevel CMakeLists.txt accordingly.

> 👤 **jeffzhou2000** replied on **2025-06-23** at **10:42:21**
> 
> refined according to your comment, pls take a look if you have time.

> 👤 **jeffzhou2000** replied on **2025-06-23** at **11:34:06**
> 
> > Your measurements clearly indicate that these are **not the best** compiler settings. 
> 
> 
> sorry I just noticed this.
> 
> the potential best compiler settings for ik_llama.cpp on Snapdragon 8Elite with NDK r28 might-be:
> 
> -march=armv8.7-a+dotprod+fp16
> 
> or 
> 
> -march=armv8.7-a+dotprod+fp16 -mcpu=cortex-x1 -mtune=cortex-x1
> 
> or 
> 
> 
> -march=armv8.7-a+dotprod+fp16+i8mm -mcpu=cortex-x1 -mtune=cortex-x1
> 
> or 
> 
> -march=armv8.7-a+dotprod+fp16+i8mm -mcpu=cortex-x1 -mtune=cortex-x1 -D_GNU_SOURCE -ffp-model=fast -fno-finite-math-only
> 
> depend on workload of Android OS, this is my personal opinion, might-be not very exactly correct.
> 
> the performance of upstream llama.cpp on Android is also weird: sometimes very good, sometimes not good as expected, as I said before: I have very very very limited knowledge about hardcore AI tech, or the codes in ${llama.cpp_src_rootdirectory}/src, so I don't know how to troubleshoot the performance issue of the upstream llama.cpp on Android thoroughly. one more thing, it seems the Android is not the point of upstream llama.cpp although it's an on-device inference framework: 
> - it seems cuda is the key-point of upstream llama.cpp.
> - I know gg is also familiar with Android dev but it seems he focus on metal and obviously he is very busy.

---

👤 **ikawrakow** started a conversation on `examples/quantize-stats/CMakeLists.txt` on **2025-06-23** at **10:03:29**

This can easily be combined with `OR` with the above condition.

> 👤 **jeffzhou2000** replied on **2025-06-23** at **10:20:45**
> 
> thanks for reminder.
> 
> if (NOT MSVC OR NOT (CMAKE_SYSTEM_NAME STREQUAL "Android"))
>     list(APPEND ARCH_FLAGS -march=native)
> endif()

---

👤 **ikawrakow** started a conversation on `scripts/build-run-android.sh` on **2025-06-23** at **10:04:53**

To me this looks a lot like a script that will only work for your specific setup. Is it really useful for others?

> 👤 **jeffzhou2000** replied on **2025-07-04** at **09:11:51**
> 
> YES, you are right.
> 
> I'm not sure because it's a script to simplify workflow of build ik_llama.cpp on Linux for Android.
> 
> I'd like to close this PR accordingly and it doesn't matter because I know this project is a playground for AI expert.
> 
> btw, I deleted my inappropriate comments(they are marked off-topic) in your excellent project today, thanks for your understanding. as I said two weeks ago: I still think the upstream llama.cpp project need your unique and important ideas and codes because you are a truly AI expert and already did an unique and important contribution to the llama.cpp project.

> 👤 **ikawrakow** replied on **2025-07-04** at **09:16:24**
> 
> > btw, I deleted my inappropriate comments(they are marked off-topic) in your excellent project today, 
> 
> Well, they are totally off-topic in a discussion about new SOTA quantization types in `ik_llama.cpp`. If you want to discuss stuff related to Android, you can open a new discussion.

> 👤 **jeffzhou2000** replied on **2025-07-04** at **09:18:14**
> 
> yes, you are absolutely correct: they are totally off-topic in a discussion about new SOTA quantization types in ik_llama.cpp. thanks for your understanding!

---

👤 **ikawrakow** requested changes on this pull request 🔄 on **2025-06-23** at **10:05:44**
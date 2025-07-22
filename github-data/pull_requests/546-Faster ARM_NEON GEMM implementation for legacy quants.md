### 🔀 [#546](https://github.com/ikawrakow/ik_llama.cpp/pull/546) - Faster ARM_NEON GEMM implementation for legacy quants

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-06-21 |
| **Updated** | 2025-06-22 |

---

#### Description

It is time to give some attention to the `ARM_NEON` back-end, which has fallen behind quite a bit.

This PR corresponds to PRs #531, #533, #534 and applies the on-the-fly repacking technique to `Q4_0, Q4_1, Q5_0, Q5_1, Q6_0, Q8_0, IQ4_NL` for the `ARM_NEON` implementation.

Here is a PP-512 performance comparison between the main branch and this PR for LlaMA-3.1-8B-Instruct on M2-Max

| type |  t/s (main) | t/s (PR) | Speedup |
| ---: | ---: | ---: | ---: |
| Q4_0 | 83.58 | 128.41 | 1.536 |
| Q5_0 | 74.20 |  128.57 | 1.733 |
| Q6_0 | 74.25 | 128.79 | 1.735 |
| Q8_0 | 84.45 | 128.63 | 1.523 |
| IQ4_NL | 84.47 | 128.09 | 1.516 |
| Q4_1 | 74.44 | 115.36 | 1.550 |
| Q5_1 | 64.16 | 114.89 | 1.791 |

---

#### 💬 Conversation

👤 **zhouwg** commented the **2025-06-22** at **07:22:29**:<br>

I tried your ik_llamacpp on Android phone equipped with Qualcomm Snapdragon 8Elite(one of the most advanced mobile SoCs on our planet at the moment) today, the **performance of your excellent ik_llamacpp is impressive(faster than the upstream llama.cpp)** .

both build with " -O3 -flto -D_GNU_SOURCE -ffp-model=fast -fno-finite-math-only " because " -O3 -march=armv8.7-a -mcpu=cortex-x1 -mtune=cortex-x1 -flto -D_GNU_SOURCE -fvectorize -ffp-model=fast -fno-finite-math-only " can't works with ik_llama.cpp cause of some compile error with inline assemble codes.

upstream llama.cpp:
llama-bench:
![Screenshot from 2025-06-22 12-58-28](https://github.com/user-attachments/assets/84381046-de4a-4c54-9aac-5c81c04d15e6)
llama-cli:
![Screenshot from 2025-06-22 15-12-04](https://github.com/user-attachments/assets/ac3644a1-0db7-46d2-b4ce-6b8e514bd8ef)

ik_llama.cpp:
llama-bench:
![Screenshot from 2025-06-22 13-08-16](https://github.com/user-attachments/assets/a2383ac7-617b-46e0-a5ab-ff907c733cb1)
![Screenshot from 2025-06-22 15-09-01](https://github.com/user-attachments/assets/4b2b2aa9-3cae-4e1b-937b-2fe62ac84dc6)

llama-cli(the inference result is incorrect and don't know why)
![Screenshot from 2025-06-22 15-12-20](https://github.com/user-attachments/assets/db2bc851-84e5-4a20-9de6-b1ede74e1972)

---

👤 **zhouwg** commented the **2025-06-22** at **07:24:04**:<br>

I tried ik_llamacpp on Android phone equipped with Qualcomm Snapdragon 8Elite(one of the most advanced mobile SoCs on our planet at the moment) today, the **performance of your excellent ik_llamacpp is impressive** .

both build with " -O3 -flto -D_GNU_SOURCE -ffp-model=fast -fno-finite-math-only " because " -O3 -march=armv8.7-a -mcpu=cortex-x1 -mtune=cortex-x1 -flto -D_GNU_SOURCE -fvectorize -ffp-model=fast -fno-finite-math-only " can't works with ik_llama.cpp cause of some compile error with inline assemble codes.

upstream llama.cpp with latest codes:
llama-bench:
![Screenshot from 2025-06-22 12-58-28](https://github.com/user-attachments/assets/84381046-de4a-4c54-9aac-5c81c04d15e6)
llama-cli:
![Screenshot from 2025-06-22 15-12-04](https://github.com/user-attachments/assets/ac3644a1-0db7-46d2-b4ce-6b8e514bd8ef)

ik_llama.cpp with latest codes:
llama-bench:
![Screenshot from 2025-06-22 13-08-16](https://github.com/user-attachments/assets/a2383ac7-617b-46e0-a5ab-ff907c733cb1)
![Screenshot from 2025-06-22 15-09-01](https://github.com/user-attachments/assets/4b2b2aa9-3cae-4e1b-937b-2fe62ac84dc6)

llama-cli(the inference result is incorrect and don't know why)
![Screenshot from 2025-06-22 15-12-20](https://github.com/user-attachments/assets/db2bc851-84e5-4a20-9de6-b1ede74e1972)

---

👤 **zhouwg** commented the **2025-06-22** at **08:36:03**:<br>

comparison of llama_bench on Android phone equipped with Qualcomm Snapdragon 8Elite(one of the most advanced mobile SoCs on our planet at the moment) + Android NDK r28:

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

4. both build with " -O3 -march=armv8.7-a -mcpu=cortex-x1 -mtune=cortex-x1  -flto -D_GNU_SOURCE -fvectorize -ffp-model=fast -fno-finite-math-only"

upstream llama.cpp with latest codes:
the following is a screenshot when I helped troubleshooting a performance regression issue in the upstream llama.cpp project. as well known, there are so many approved PRs in the upstream llama.cpp project and some approved PRs might-be brings regression issues in the upstream llama.cpp project. sometimes I can't reproduce the same benchmark result with the upstream llama.cpp's latest codes.

![455784182-f30ce0c8-5528-44fe-8be3-213ebaf4e730](https://github.com/user-attachments/assets/bc182761-acd1-4aeb-9da8-8bce36b9e15e)

ik_llama.cpp with latest codes:

---

👤 **zhouwg** commented the **2025-06-22** at **09:46:28**:<br>

comparison of llama_bench on Android phone equipped with Qualcomm Snapdragon 8Elite(one of the most advanced mobile SoCs on our planet at the moment) + Android NDK r28:

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

![Screenshot from 2025-06-22 17-30-43](https://github.com/user-attachments/assets/96389f4e-8961-4995-9424-e2804ee146d1)

the following is a screenshot when I helped troubleshooting a performance regression issue in the upstream llama.cpp project. as well known, there are so many approved PRs in the upstream llama.cpp project and some approved PRs might-be brings regression issues in the upstream llama.cpp project. sometimes I can't reproduce the same benchmark result with the upstream llama.cpp's latest codes.

![455784182-f30ce0c8-5528-44fe-8be3-213ebaf4e730](https://github.com/user-attachments/assets/bc182761-acd1-4aeb-9da8-8bce36b9e15e)

ik_llama.cpp with latest codes:
![Screenshot from 2025-06-22 17-45-34](https://github.com/user-attachments/assets/00cde394-87f7-4851-bec7-7b27dea9c16d)

---

👤 **zhouwg** commented the **2025-06-22** at **10:58:12**:<br>

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

![Screenshot from 2025-06-22 17-30-43](https://github.com/user-attachments/assets/96389f4e-8961-4995-9424-e2804ee146d1)

the following is a screenshot when I helped troubleshooting a performance regression issue in the upstream llama.cpp project. as well known, there are so many approved PRs in the upstream llama.cpp project and some approved PRs might-be brings regression issues in the upstream llama.cpp project. sometimes I can't reproduce the same benchmark result with the upstream llama.cpp's latest codes.

![455784182-f30ce0c8-5528-44fe-8be3-213ebaf4e730](https://github.com/user-attachments/assets/bc182761-acd1-4aeb-9da8-8bce36b9e15e)

ik_llama.cpp with latest codes:
![Screenshot from 2025-06-22 17-45-34](https://github.com/user-attachments/assets/00cde394-87f7-4851-bec7-7b27dea9c16d)

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
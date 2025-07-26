### [Discussion #591](https://github.com/ikawrakow/ik_llama.cpp/discussions/591) - I dont see any speed improvement in generation, so want to understand if i am missing something

| **Author** | `Greatz08` |
| :--- | :--- |
| **Created** | 2025-07-07 |
| **Updated** | 2025-07-08 |

---

#### Description

First of all thank you very much for your contribution in quantization which helps GPU poor people like us to enjoy LLM's :-)) . I recently compiled llama.cpp with these commands : 

`
cmake -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES="89" \
    -DGGML_CUDA_F16=ON \
    -DGGML_CUDA_FA_ALL_QUANTS=ON \
    -DGGML_BLAS=ON \
    -DGGML_BLAS_VENDOR=OpenBLAS \
    -DLLAMA_LLGUIDANCE=ON \
`

`cmake --build build --config Release -j`

I have RTX 4060 8GB VRAM, so i asked gemini 2.5 pro latest to guide me. I feeded him all docs context with project gitingest and then i asked it to generate best build command and it did which i pasted above, so do let me know if i have to make some more changes or not, because i used same commands to build the fork version (this project).

I get same speed in both llama.cpp version and this fork version. I used following command to run model.

`GGML_CUDA_ENABLE_UNIFIED_MEMORY=1 ./build/bin/llama-server --device CUDA0 \
    -m ~/models/Qwen3-30B-A3B-128K-UD-Q2_K_XL.gguf \
    -c 32000 \
    -ngl 48 \
    -t 4 \
    -ot '.*\.ffn_down_exps\.weight=CPU' \
    -ot '.*\.ffn_up_exps\.weight=CPU' \
    -ub 256 -b 512 \
    --host 0.0.0.0 \
    --port 8009 \
    --flash-attn \
    --cache-type-k q8_0 \
    --cache-type-v q8_0 \
`

I am getting 20-23 token/s , so i wanted to know if i can improve it further with re compiling or you can guide me to improve this command further. I am asking for much more improvement because i want to go for IQ3_XXS Quant which people reported works great and that's will be my end limit.

---

#### 🗣️ Discussion

👤 **ikawrakow** commented on **2025-07-07** at **16:24:50**

* Remove `-DGGML_BLAS=ON  -DGGML_BLAS_VENDOR=OpenBLAS` from the build command
* I wouldn't know what `-DLLAMA_LLGUIDANCE=ON` does, so just remove from the build command
* You can reduce your build time by not using `-DGGML_CUDA_FA_ALL_QUANTS=ON`, which is only necessary if you want to use more exotic KV cache quantization types (not needed with the `Q8_0` that you have used)
* Does your RTX 4060 support unified memory? If not, remove the `GGML_CUDA_ENABLE_UNIFIED_MEMORY=1` from your server command
* What is your CPU? Does it only have 4 cores? All operations with tensors that were not offloaded to the GPU run on the CPU for token generation, so that's important
*  If you are leaving 2 of the 3 FFN tensors on the CPU, I think it is better to have `ffn_up_exps` and `ffn_gate_exps` on the CPU
* Use `-ngl 100` or some such. IIRC Qwen3-30B-A3B has 48 repeating layers, so with `-ngl 48` you are not offloading the output tensor to the GPU. This slows down promo processing and token generation. Or was that your intent? 
* You definitely want to add `-fmoe` to your server command
* For better prompt processing speed, you should try to use larger `-b` and `-ub` (if VRAM permits). Given enough VRAM, best prompt processing speed for MoE models such as Qwen3-30B-A3B is obtained with `-b 4096 -ub 4096` (but this requires larger CUDA compute buffers)

Having said all that, token generation speed in the case of CPU-only or hybrid GPU/CPU inference is limited by CPU memory bandwidth, so performance gains compared to mainline `llama.cpp` tend to be smaller. The big advantage of `ik_llama.cpp` is in prompt processing speed. You may also see larger performance gains for token generation with a long context stored in the KV cache. 

After you get going with Unsloth's quantized models, you may also want to look into some of the quantized models with `ik_llama.cpp` specific quants, but let's not throw too much information your way all at once.
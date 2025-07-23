### 🗣️ [#591](https://github.com/ikawrakow/ik_llama.cpp/discussions/591) - I dont see any speed improvement in generation, so want to understand if i am missing something

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

👤 **ikawrakow** replied the **2025-07-07** at **16:24:50**:<br>

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

> 👤 **Greatz08** replied the **2025-07-08** at **00:59:56**:<br>
> > Does your RTX 4060 support unified memory? If not, remove the GGML_CUDA_ENABLE_UNIFIED_MEMORY=1 from your server command
> 
> I dont think so, i will remove it.
> 
> 
> 
> > What is your CPU? Does it only have 4 cores? All operations with tensors that were not offloaded to the GPU run on the CPU for token generation, so that's important
> 
> I forgot to mention any info about my CPU. My cpu is AMD Ryzen 7840HS (8 core,16 threads). I btw tested both t 4 and t 8, i pasted t 4 version command in my previous message. I was just testing both values for observing inference speed differences.
> 
> 
> 
> 
> > If you are leaving 2 of the 3 FFN tensors on the CPU, I think it is better to have ffn_up_exps and ffn_gate_exps on the CPU
> 
> Ok, this was interesting thing to know and i will try with these two tensor layers. If possible do share your wisdom on this, like why you think these two will be better (just interested to learn and understand more :-) ). 
> 
> ![image](https://github.com/user-attachments/assets/8bfe6500-309a-496f-af06-9eafcd108597)
> blk.1.ffn_down_exps.weight  - 0.66 % of model param
> blk.1.ffn_gate_exps.weight  - 0.66 % of model param
> blk.1.ffn_gate_inp.weight  - <0.01 % of model param
> blk.1.ffn_norm.weight  - <0.01 % of model param
> blk.1.ffn_up_exps.weight  -   0.66 % of model param
> 
> On the basis of this i thought two layers would be sufficient to save enough vram space to load all attention layers in GPU VRAM ( https://reddit..com/r/LocalLLaMA/comments/1ki7tg7/dont_offload_gguf_layers_offload_tensors_200_gen/ ) . From this reddit post i got know about this awesome trick of override-tensor.
> 
> 
> 
> > Use -ngl 100 or some such. IIRC Qwen3-30B-A3B has 48 repeating layers, so with -ngl 48 you are not offloading the output tensor to the GPU. This slows down promo processing and token generation. Or was that your intent?
> 
> ![image](https://github.com/user-attachments/assets/2d14c597-30d8-48d5-9e50-8d3474d30a19)
> 
> Number of Layers: 48  - After seeing this i thought i should be loading all 48 layers in GPU VRAM (for that only i saved VRAM space by offloading specific tensor layers) , because of this i choose 48 layers. I dont know about 'repeating layer' , so i think either i missed a key concept or you might be referring to another model layers ? ( Do let me know about this)
> 
> 
> > For better prompt processing speed, you should try to use larger -b and -ub (if VRAM permits). Given enough VRAM, best prompt processing speed for MoE models such as Qwen3-30B-A3B is obtained with -b 4096 -ub 4096 (but this requires larger CUDA compute buffers)
> 
> I will see how much i can increment those numbers for both params, and will test with longer context. I will also follow rest of your suggestions and will test things out.
> 
> 
> Thank you very much for your guidance on this matter @ikawrakow   :-))
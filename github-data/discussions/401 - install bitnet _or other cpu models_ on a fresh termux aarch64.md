### 🗣️ [#401](https://github.com/ikawrakow/ik_llama.cpp/discussions/401) - install bitnet (or other cpu models) on a fresh termux aarch64

| **Author** | `Benjamin-Wegener` |
| :--- | :--- |
| **Created** | 2025-05-09 |
| **Updated** | 2025-06-21 |

---

#### Description

just for convenience all subsequential commands to install bitnet (or other cpu models) on a fresh termux aarch64:
```bash
apt update && apt install wget cmake git -y
git clone https://github.com/ikawrakow/ik_llama.cpp
cd ik_llama.cpp
cmake -B ./build -DGGML_CUDA=OFF -DGGML_BLAS=OFF -DGGML_ARCH_FLAGS="-march=armv8.2-a+dotprod+fp16" -DGGML_IQK_FLASH_ATTENTION=OFF
cmake --build ./build --config Release -j $(nproc)
wget https://huggingface.co/microsoft/bitnet-b1.58-2B-4T-gguf/resolve/main/ggml-model-i2_s.gguf?download=true -O ./models/ggml-model-i2_s.gguf
./build/bin/llama-quantize --allow-requantize ./models/ggml-model-i2_s.gguf ./models/bitnet.gguf iq2_bn_r4
./build/bin/llama-server -mla 3--model ./models/bitnet.gguf
```
the template for the model in chat prompt under 127.0.0.1:8080 should be
```
<|begin_of_text|>{{prompt}}<|eot_id|>
{{history}}
{{char}}:
```

thanks for the help @ikawrakow @RobertAgee @saood06 
edit: sometimes its producing  nonsense output
reverted to old prompt template

---

#### 🗣️ Discussion

👤 **VinnyG9** replied the **2025-05-14** at **12:07:00**:<br>

what is a termux?

> 👤 **saood06** replied the **2025-05-14** at **12:25:00**:<br>
> > what is a termux?
> 
> Android terminal emulator: https://termux.dev/en/

---

👤 **Benjamin-Wegener** replied the **2025-05-15** at **14:23:33**:<br>

using the built in llama-server standard and pasting that in prompt template field to get correct chat format
<|begin_of_text|>{{prompt}}<|eot_id|>

{{history}}
{{char}}:

> 👤 **saood06** replied the **2025-05-16** at **06:01:00**:<br>
> Just to be clear the proper template is:
> 
> <|begin_of_text|>System: {system_message}<|eot_id|>
> User: {user_message_1}<|eot_id|>
> Assistant: {assistant_message_1}<|eot_id|>
> User: {user_message_2}<|eot_id|>
> Assistant: {assistant_message_2}<|eot_id|>
> 
> It's been a while since I've used the server's template field but my testing using an alternative front-end following this was successful.
> 
> 👤 **saood06** replied the **2025-05-18** at **12:42:54**:<br>
> @Benjamin-Wegener 
> 
> The template above is grabbed from the paper. It isn't what is meant to actually go into the template field under the server's built in front-end. 
> 
> That uses the following variables: {{prompt}}, {{history}}, {{char}}, {{name}}, {{message}} and has sections for the System Prompt, Prompt template, and Chat history template, along with names for the user and the AI.
> 
> Even when I used the bundled front-end I still basically never used the "Chat" section where those fields existed. I used the completions section where I would manually conform to a template, but I can see why on a mobile device the Chat endpoint would be far more convenient.
> 
> Also I have uploaded already converted models [here](https://huggingface.co/tdh111/bitnet-b1.58-2B-4T-GGUF) which might be useful if space is limited (the actual time to convert is minor for this model so unlike other models that benefit doesn't exist for it).
> 
> 👤 **RobertAgee** replied the **2025-05-18** at **12:59:53**:<br>
> FWIW, once i got the server running, I was able to confirm it was working with this curl request. Alternatively, you could send this like a regular JSON webhook of course:
> 
> ```
> curl http://127.0.0.1:8080/completion -X POST \
>   -H "Content-Type: application/json" \
>   -d '{
>     "prompt": "<|im_start|>system\nYou are a helpful assistant<|im_end|>\n<|im_start|>user\nHello, who are you?<|im_end|>\n<|im_start|>assistant\n",
>     "temperature": 0.7,
>     "n_predict": 128,
>     "stop": ["<|im_end|>"]
>   }'
>   ```
>   
> Also, I was able to connect [ChatterUI's](https://github.com/Vali-98/ChatterUI) (free and oss) mobile app to my termux server with a config file and now I have a superfast, local, AI with TTS, chat interface, and convo history.
> 
> Setting up the connection took me awhile to figure out, so if anyone's interested, I'll share the config file and settings. But yeah, all things said Bitnet is rough but shows promise. Would love to try out an abliterated version and Falcon 3 to see if either of those would help it have a little more conversational flow.
> 
> 👤 **Benjamin-Wegener** replied the **2025-05-18** at **13:44:35**:<br>
> so we revert that back to what i posted earlier for the server? what do you think?
> 
> ```
> <|begin_of_text|>{{prompt}}<|eot_id|>
> 
> {{history}}
> {{char}}:
> ```
> @saood06

---

👤 **RobertAgee** replied the **2025-05-16** at **05:26:44**:<br>

Didn't work for me in my case. Stayed hung up at compilation forever
![1000035416](https://github.com/user-attachments/assets/0b55130a-1964-44fb-8f44-da2bd2557b84)

> 👤 **ikawrakow** replied the **2025-05-16** at **05:30:51**:<br>
> You have to be patient. The file is 18k LOC of heavily templated C++ code. It takes a while to compile even on a fast desktop CPU. I know it needs to get refactored into multiple files (#183), but I haven't come around to do it.
> 
> 👤 **ikawrakow** replied the **2025-05-16** at **06:21:47**:<br>
> Just measured: it takes 2 minutes on my M2-Max CPU to compile this file. Based on this, my guess is that it is in the 5-10 minutes range on a phone.
> 
> 👤 **saood06** replied the **2025-05-16** at **06:26:21**:<br>
> > Just measured: it takes 2 minutes on my M2-Max CPU to compile this file. Based on this, my guess is that it is in the 5-10 minutes range on a phone.
> 
> I feel like it took longer when I tested it, and the person reporting the clashing .so files reported around half an hour, but yes the solution is to just be patient.
> 
> 👤 **RobertAgee** replied the **2025-05-16** at **06:27:06**:<br>
> I waited more than 10 minutes, without competing processes open. in htop, no rw was happening so there's something causing it to hang idk
> 
> 👤 **saood06** replied the **2025-05-16** at **06:29:17**:<br>
> > I waited more than 10 minutes, without competing processes open. in htop, no rw was happening so there's something causing it to hang idk
> 
> But was there still CPU usage? Also if you don't mind sharing what device it was on it would help estimate how long it would take. ( I may be able to time a compile on the device I use to test Android on but that may be a while as I have to borrow that device).
> 
> 👤 **RobertAgee** replied the **2025-05-17** at **14:17:34**:<br>
> Hi @saood06 I appreciate your patience and willingness to help. I have a Samsung a71 5g
> 
> ```
> PLATFORM
> OS	Android 10, upgradable to Android 13, One UI 5
> Chipset	Exynos 980 (8 nm)
> CPU	Octa-core (2x2.2 GHz Cortex-A77 & 6x1.8 GHz Cortex A55)
> GPU	Mali-G76 MP5
> ```
> 
> I did get it to compile and successfully run with the new FA kernels OFF flag at the compilation step.
> 
> 👤 **saood06** replied the **2025-05-18** at **02:49:19**:<br>
> >Hi @saood06 I appreciate your patience and willingness to help
> >I did get it to compile and successfully run with the new FA kernels OFF flag at the compilation step.
> 
> I'm glad you were able to get it working. I don't think the new flag is necessary but it definitely would speed things up, which could matter a lot (especially as a lot of users won't have the patience and understanding to just wait).

---

👤 **ikawrakow** replied the **2025-05-17** at **08:24:16**:<br>

You can now disable building the templated flash attention (FA) kernels. Disabling FA should massively improve build times.

See PR #429

> 👤 **RobertAgee** replied the **2025-05-17** at **10:00:36**:<br>
> Thanks @ikawrakow for the fast PR! I was able to successfully get it running and make a call to get a response! :) 
> 
> For anyone in my situation, it did have a few what looked like errors in the console during the build process, but it was successful, as I said, so no worries. Here's the list of commands with the speed up (disabling flash attention kernels):
> 
> ```apt update && apt install wget cmake git -y
> 
> git clone https://github.com/ikawrakow/ik_llama.cpp
> 
> cd ik_llama.cpp
> 
> cmake -B ./build -DGGML_CUDA=OFF -DGGML_BLAS=OFF -DGGML_ARCH_FLAGS="-march=armv8.2-a+dotprod+fp16" -DGGML_IQK_FLASH_ATTENTION=OFF
> 
> cmake --build ./build --config Release -j $(nproc)
> 
> wget https://huggingface.co/microsoft/bitnet-b1.58-2B-4T-gguf/resolve/main/ggml-model-i2_s.gguf?download=true -O ./models/ggml-model-i2_s.gguf
> 
> ./build/bin/llama-quantize --allow-requantize ./models/ggml-model-i2_s.gguf ./models/bitnet.gguf iq2_bn_r4
> 
> ./build/bin/llama-server -mla 3 --model ./models/bitnet.gguf
> ```
> 
> Sample call I made from my API tester app to the server to test it.
> 
> ```
> curl http://127.0.0.1:8080/completion -X POST \
>   -H "Content-Type: application/json" \
>   -d '{
>     "prompt": "<|im_start|>system\nYou are a helpful assistant<|im_end|>\n<|im_start|>user\nHello, who are you?<|im_end|>\n<|im_start|>assistant\n",
>     "temperature": 0.7,
>     "n_predict": 128,
>     "stop": ["<|im_end|>"]
>   }'
> ```

---

👤 **ikawrakow** replied the **2025-05-20** at **09:48:56**:<br>

There is now PR #435 that significantly reduces build time. I cannot test on Android myself, so would appreciate if someone did and reported
* New vs old build time (with CPU model)
* Does it still work correctly?
* Is the inference performance affected?

> 👤 **aezendc** replied the **2025-06-02** at **15:30:06**:<br>
> > There is now PR #435 that significantly reduces build time. I cannot test on Android myself, so would appreciate if someone did and reported
> > 
> > * New vs old build time (with CPU model)
> > * Does it still work correctly?
> > * Is the inference performance affected?
> 
> HI ikawrakow do we have a step by step running microsoft/bitnet-b1.58-2B-4T-gguf in windows?
> 
> 👤 **ikawrakow** replied the **2025-06-02** at **15:36:51**:<br>
> There are no prebuild packages, so you need to follow the [above instructions](https://github.com/ikawrakow/ik_llama.cpp/discussions/401#discussioncomment-13178115) and build yourself. They don't work (with small adjustments)?
> 
> 👤 **aezendc** replied the **2025-06-02** at **15:45:42**:<br>
> > There are no prebuild packages, so you need to follow the [above instructions](https://github.com/ikawrakow/ik_llama.cpp/discussions/401#discussioncomment-13178115) and build yourself. They don't work (with small adjustments)?
> 
> I made it work I use [saood06](https://github.com/saood06) converted model https://huggingface.co/tdh111/bitnet-b1.58-2B-4T-GGUF. I will create a basic commands
> 
> 👤 **saood06** replied the **2025-06-03** at **00:51:30**:<br>
> > do we have a step by step running microsoft/bitnet-b1.58-2B-4T-gguf in windows?
> 
> There are build instructions with a lot more details for Windows [here](https://github.com/ikawrakow/ik_llama.cpp/blob/main/docs/build.md).  Once it is built you can just grab the model either pre-converted one like [this](https://huggingface.co/tdh111/bitnet-b1.58-2B-4T-GGUF) or convert one yourself and just launch server. Which is covered in the above instructions.
> 
> It seems like you have already figured it out, but just wanted to link the Windows build instructions in case anyone else finds this and wants to follow along.
> 
> 👤 **aezendc** replied the **2025-06-03** at **03:34:32**:<br>
> > > do we have a step by step running microsoft/bitnet-b1.58-2B-4T-gguf in windows?
> > 
> > There are build instructions with a lot more details for Windows [here](https://github.com/ikawrakow/ik_llama.cpp/blob/main/docs/build.md). Once it is built you can just grab the model either pre-converted one like [this](https://huggingface.co/tdh111/bitnet-b1.58-2B-4T-GGUF) or convert one yourself and just launch server. Which is covered in the above instructions.
> > 
> > It seems like you have already figured it out, but just wanted to link the Windows build instructions in case anyone else finds this and wants to follow along.
> 
> Thanks for this @saood06 very helpful and a very detailed one. One thing I have a problem accessing the llama-server ui and its just keep loading.
> 
> 👤 **saood06** replied the **2025-06-03** at **07:11:46**:<br>
> > Thanks for this @saood06 very helpful and a very detailed one. One thing I have a problem accessing the llama-server ui and its just keep loading.
> 
> Just to be sure, are you making sure to access the server using the port passed in when launching (or 8080 if not set as that is the default), and are you setting the host address (if needed) since it defaults to 127.0.0.1 (AKA localhost) which is only accessible on that machine.
> 
> 👤 **aezendc** replied the **2025-06-03** at **12:28:17**:<br>
> > > Thanks for this @saood06 very helpful and a very detailed one. One thing I have a problem accessing the llama-server ui and its just keep loading.
> > 
> > Just to be sure, are you making sure to access the server using the port passed in when launching (or 8080 if not set as that is the default), and are you setting the host address (if needed) since it defaults to 127.0.0.1 (AKA localhost) which is only accessible on that machine.
> 
> i am using the default http://127.0.0.1:8080/ but somehow it works now. Thanks for the info
> 
> 👤 **aezendc** replied the **2025-06-04** at **14:40:21**:<br>
> > > Thanks for this @saood06 very helpful and a very detailed one. One thing I have a problem accessing the llama-server ui and its just keep loading.
> > 
> > Just to be sure, are you making sure to access the server using the port passed in when launching (or 8080 if not set as that is the default), and are you setting the host address (if needed) since it defaults to 127.0.0.1 (AKA localhost) which is only accessible on that machine.
> 
> How you do make the the model to respond longer?
> 
> 👤 **saood06** replied the **2025-06-21** at **16:33:44**:<br>
> >How you do make the the model to respond longer?
> 
> I don't have much specific advice for using this model. Beyond benchmarking and minor curiosity of the ability of a model this small, I haven't used it much.
> 
> I'd be curious to hear what your experience with it has been? Is it useful (even if the responses are a bit short for your liking)?
> 
> I've never actually found a great model and prompt context agnostic way to increase the length of a response without reducing the quality of the response, but my strategies are (in order of least effort to highest effort), are:
> 
> * add context specific details or changes to the prompt given
> * break the task apart and only allow it to respond to a fraction at a time
> * manually steer the model to avoid skipping or missing out on details (often is easier with a thinking model as you often only have to steer during thinking tokens).
> 
> 👤 **aezendc** replied the **2025-06-21** at **16:46:12**:<br>
> I fix it now. The only problem of mine is the libomp.so build and I do not have a file of it. I set it the openmp off because libggml.so needs the libomp.so an when I build llama-server using windows and transfer the binaries to my android phone and the model is hallucinating.
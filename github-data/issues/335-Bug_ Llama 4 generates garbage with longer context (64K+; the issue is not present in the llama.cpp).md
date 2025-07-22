### 🐛 [#335](https://github.com/ikawrakow/ik_llama.cpp/issues/335) - Bug: Llama 4 generates garbage with longer context (64K+; the issue is not present in the llama.cpp)

| **Author** | `Lissanro` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-04-20 |
| **Updated** | 2025-04-25 |

---

#### Description

### What happened?

ik_llama.cpp works fine with Llama 4 (Maverick and Scout) at lower context (like few thousand tokens).

But with 64K long context, it seems to completely fail. Prompt content does not matter as far as I can tell, can be as simple as pasting a long snippet from a random book and asking question about it. Not sure exact threshold failure, so I recommend testing with at least 70K to reproduce.

Example output from ik_llama.cpp:

```
00 + 0: 0: 0: 0:00:0: // 0:00: 0: 0:00: 0: 0 0:0: //: 0:00:00: "1: (data: 0:00:00 + 0:00 (1: 0: 0000: 0
:0: "C: 0 + 0: 0:0:00: 0:0:00:0: :0: 0:0000: 0:00:00: 0:0: 0:00
:00:17:00: 0: "1: 0: //: 0, 0: 0:00: "data: 0: 0 0:0000:00
:00: //: 0: 0 : 0: //0:00:0:00:0:00 + 0
... (very long garbage output continues for a while) ...
```

In contrast, with llama.cpp I get coherent output:

```
To address your concerns about the potential connection between... (long normal output that addresses the question)
```

This is how I started with ik_llama.cpp (where the issue occurs):

```
~/pkgs/ik_llama.cpp/build/bin/llama-server \
--model /home/lissanro/neuro/Llama-4-Maverick-17B-128E-Instruct-GGUF-UD-Q4_K_XL-1048576seq/Llama-4-Maverick-17B-128E-Instruct-UD-Q4_K_XL-00001-of-00005.gguf \
--ctx-size 524288 --n-gpu-layers 49 --tensor-split 25,25,25,25 -fa -ctk q8_0 -ctv q8_0 \
-rtr -amb 1024 --override-tensor "exps=CPU" --threads 64 --host 0.0.0.0 --port 5000
```

This is how I started llama.cpp (which works fine; had to use smaller ctx-size but still fits the same prompt I used for the test):

```
~/pkgs/llama.cpp/build/bin/llama-server \
--model /home/lissanro/neuro/Llama-4-Maverick-17B-128E-Instruct-GGUF-UD-Q4_K_XL-1048576seq/Llama-4-Maverick-17B-128E-Instruct-UD-Q4_K_XL-00001-of-00005.gguf \
--ctx-size 80000 --n-gpu-layers 4 --tensor-split 25,25,25,25 -fa -ctk q8_0 -ctv q8_0 \
--threads 64 --host 0.0.0.0 --port 5000
```

Please let me know if I am doing something wrong or did I encountered a bug?

### Name and Version

I am using latest git version.

### What operating system are you seeing the problem on?

Linux

### Relevant log output

```shell

```

---

#### 💬 Conversation

👤 **ikawrakow** commented the **2025-04-20** at **05:54:22**:<br>

What happens if you don't use the `-amb 1024` command line argument? You may need to reduce the max. context size without that. I'm trying to pinpoint the problem, and two things come to mind:
* I have a bug when computing attention in chunks. If so, removing `-amb 1024` will make it work correctly
* I have a bug in the RoPE implementation. If so, removing `-amb 1024` will still not work.

---

👤 **Lissanro** commented the **2025-04-20** at **14:02:44**:<br>

Unfortunately removing `-amb 1024 `did not help, I still get very long bad reply like `0: "0000: 0:00: 0:00: //:0:00:00:` - I let it run for a while, then stopped it because otherwise it probably would have continued until running out of output token limit. Here is full log without `-amb 1024` option in case it is useful: https://pastebin.com/hE8kP3Sn

---

👤 **ikawrakow** commented the **2025-04-20** at **14:44:19**:<br>

OK, thanks. I'll take a closer look when I come back from a short break.

---

👤 **Lissanro** commented the **2025-04-23** at **05:40:29**:<br>

Some additional information about reproducing the issue with a smaller Scout model and maybe help to narrow down possible causes:

I tested with Scout ([Unsloth quant](https://huggingface.co/unsloth/Llama-4-Scout-17B-16E-Instruct-GGUF/tree/main/UD-Q4_K_XL)). It starts to breakdown at around 10K-14K range; 10K seems to produce mostly good output, not sure if the quality is 100% but it seems to be coherent. At 14K quality drops significantly, sometimes I get obvious garbage, sometimes something semi-coherent. It becomes increasingly worse beyond that point. For example, at 32K+ context length, bad output is obvious.

I thought maybe llama.cpp had similar issue in the past, but when I reverted it to the initial match that added the llama 4 text-only support, output with both Scout and Maverick was fine, even at larger context (tested with up to 48K input prompt). So, it seems to be ik_llama.cpp specific issue.

I tested both ik_llama.cpp and llama.cpp with identical command:

./build/bin/llama-server --model ~/neuro/Llama-4-Scout-17B-16E-Instruct-GGUF-UD-Q4_K_XL-10485760seq/Llama-4-Scout-17B-16E-Instruct-UD-Q4_K_XL-00001-of-00002.gguf --ctx-size 81920 --n-gpu-layers 49 --tensor-split 25,25,25,25 -fa -ctk q8_0 -ctv q8_0 --threads 64 --host 0.0.0.0 --port 5000

I also tried ik_llama.cpp without "-fa -ctk q8_0 -ctv q8_0" but still got bad output.

---

👤 **ikawrakow** commented the **2025-04-23** at **06:16:05**:<br>

Thanks, this is useful. I think I can run Scout with 16k context, so this will make debugging easier.

---

👤 **ikawrakow** commented the **2025-04-23** at **08:29:12**:<br>

Perplexity for context of 16k tokens seems fine:
```
./bin/llama-perplexity -m Llama-4-Scout-17B-16E-Instruct-UD-Q2_K_XL.gguf -f ../../iquants/tests/wiki.test.raw -ub 2048 -t 32 -ngl 100 -c 16384 -ot "blk\.[0-8]\.ffn_up_exps=CUDA0,blk\.[0-8]\.ffn_down_exps=CUDA0,exps=CPU" -rtr -fmoe -fa
perplexity: 53.95 seconds per pass - ETA 15.28 minutes
[1]5.1728,[2]7.0584,[3]7.3954,[4]6.8851,[5]6.2507,[6]6.6663,[7]6.4059,[8]6.5071,[9]6.6680,[10]6.7368,[11]6.8609,[12]7.0999,[13]7.1736,[14]7.1565,[15]7.1548,[16]7.1633,[17]7.1819,
Final estimate: PPL = 7.1819 +/- 0.04765
```
 
I also spent some time talking to it using `llama-server`, seemed also fine. I thought the answers were often too short and lacked detail, but I didn't see the symptoms that you are having.
  
Can you attach the specific prompt that triggers the bug?

---

👤 **Lissanro** commented the **2025-04-24** at **06:22:02**:<br>

I decided to test with your exact quant, I downloaded it here:

https://huggingface.co/unsloth/Llama-4-Scout-17B-16E-Instruct-GGUF/resolve/main/Llama-4-Scout-17B-16E-Instruct-UD-Q2_K_XL.gguf

After testing with it, I noticed that at 18K input, it still may produce coherent output in many cases, even though quality may be reduced. For example, a prompt to summaries Wikipedia article about AI, truncated to about 18K tokens:

```
## Summary

Artificial intelligence (AI) refers to the capability of computational systems to perform tasks typically associated with human intelligence, such as learning, reasoning, problem-solving, perception, and decision-making. It is a field of research in computer science that develops and studies methods and software that enable machines to perceive their environment and use learning and intelligence to take actions that maximize their chances of achieving defined goals.
...
[few more paragraphs of text that provide seemingly normal summary of the article]
```

But when I increase input length further (around 23K toknes), it starts to breakdown:

```
The emergence of generative artificial intelligence (AI) has been seen as a significant breakthrough in the field of artificial intelligence (AI) behavior prediction prediction prediction patterns prediction analysis prediction analysis prediction vehicles and criticism criticism of the behavior of vehicles vehicles vehicles vehicles vehicles vehicles vehicles vehicles vehicles vehicles vehicles vehicles vehicles vehicles vehicles vehicles vehicles vehicles vehicles vehicles vehicles vehicles vehicles vehicles vehicles vehicles vehicles vehicles vehicles
...
[word "vehicles" is repeated until running out of the token limit]
```

However, the very beginning still may look OK, and there is still a possibility that it may provide semi-coherent replies to some prompts. But I am pretty sure that using full size article about AI (around 72K) will reliably break it no matter what settings. Using full 72K token long that I share below, you can truncate it to the maximum context window you can run for the best reproducibility.

For reference, here is output with full 72K tokens long prompt:

```
�

###iom..

  |

.Imageoboxiom

.Imageoboxiom

.Imageobox Gmoboxiom

###iomobox Gmobox Hectometers Hectometers Hectometers Hectometers Hectometers
...
[word "Hectometers" is repeated until running out of token limit]
```

Here are exact prompts used that reproduce the issue on my side:

https://dragon.studio/2025/04/prompt-23K.txt (truncated Wikipedia article, around 23K tokens long, the result shown above)

https://dragon.studio/2025/04/prompt-76K.txt (full Wikipedia article, around 76K tokens long)

I think just by using long enough prompt it should be possible to reproduce the issue - the longer the prompt, the more reproducible it should be (as shown in the examples, it still starts semi-coherent for 23K long prompt for this combination of quant and prompt).

For full reproducibility, I also provide exact setting I used:

https://dragon.studio/2025/04/send_prompt.py - running this script like this will use fixed seed and determenistic temperature setting for the best reproducibility:

```
python3 send_prompt.py --temp=0 --seed=0 --port=5000 prompt-23K.txt
```

You do not really need to use the script -  it is quite short and does nothing fancy, just sets basic parameters and sends the prompt, then prints out the result. So probably you can just use the prompt in UI of your choice to get the same or similar result by just setting temperature and seed to 0 (not sure if it matters, but my test script by default sets top-k=40, top-p=0.9, min-p=0.1, max-tokens=1024).

This is how I compiled ik_llama.cpp (after running "git clone" in the ~/pkgs folder):

```
cd ~/pkgs && cmake ik_llama.cpp -B ik_llama.cpp/build -DGGML_CUDA_FA_ALL_QUANTS=ON -DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=ON -DLLAMA_CURL=ON && cmake --build ik_llama.cpp/build --config Release -j --clean-first --target llama-quantize llama-cli llama-server
```

This is how I run it:

```
~/pkgs/ik_llama.cpp/build/bin/llama-server \
--model /mnt/secondary/neuro/Llama-4-Scout-17B-16E-Instruct-UD-Q2_K_XL.gguf \
--ctx-size 81920 --n-gpu-layers 49 --tensor-split 25,25,25,25 -fa -ctk q8_0 -ctv q8_0 \
--threads 64 --host 0.0.0.0 --port 5000
```

---

👤 **Lissanro** commented the **2025-04-24** at **06:22:02**:<br>

I decided to test with your exact quant, I download it here:

https://huggingface.co/unsloth/Llama-4-Scout-17B-16E-Instruct-GGUF/resolve/main/Llama-4-Scout-17B-16E-Instruct-UD-Q2_K_XL.gguf

After testing with it, I noticed that at 18K input, it still may produce coherent output in many cases, even though quality may be reduced. For example, a prompt to summaries Wikipedia article about AI, truncated to about 18K tokens:

```
## Summary

Artificial intelligence (AI) refers to the capability of computational systems to perform tasks typically associated with human intelligence, such as learning, reasoning, problem-solving, perception, and decision-making. It is a field of research in computer science that develops and studies methods and software that enable machines to perceive their environment and use learning and intelligence to take actions that maximize their chances of achieving defined goals.
...
[few more paragraphs of text that provide seemingly normal summary of the article]
```

But when I increase input length further (around 23K toknes), it starts to breakdown:

```
The emergence of generative artificial intelligence (AI) has been seen as a significant breakthrough in the field of artificial intelligence (AI) behavior prediction prediction prediction patterns prediction analysis prediction analysis prediction vehicles and criticism criticism of the behavior of vehicles vehicles vehicles vehicles vehicles vehicles vehicles vehicles vehicles vehicles vehicles vehicles vehicles vehicles vehicles vehicles vehicles vehicles vehicles vehicles vehicles vehicles vehicles vehicles vehicles vehicles vehicles vehicles vehicles
...
[word "vehicles" is repeated until running out of the token limit]
```

However, the very beginning still may look OK, and there is still a possibility that it may provide semi-coherent replies to some prompts. But I am pretty sure that using full size article about AI (around 72K) will reliably break it no matter what settings. Using full 72K token long that I share below, you can truncate it to the maximum context window you can run for the best reproducibility.

Here are exact prompts used that reproduce the issue on my side:

https://dragon.studio/2025/04/prompt-23K.txt (truncated Wikipedia article, around 23K tokens long, the result shown above)

https://dragon.studio/2025/04/prompt-76K.txt (full Wikipedia article, around 76K tokens long)

I think just by using long enough prompt it should be possible to reproduce the issue - the longer the prompt, the more reproducible it should be (as shown in the examples, it still starts semi-coherent for 23K long prompt for this combination of quant and prompt).

For full reproducibility, I also provide exact setting I used:

https://dragon.studio/2025/04/send_prompt.py - running this script like this will use fixed seed and determenistic temperature setting for the best reproducibility:

```
python3 send_prompt.py --temp=0 --seed=0 prompt-23.txt
```

You do not really need to use the script -  it is quite short and does nothing fancy, just sets basic parameters and sends the prompt, then prints out the result. So probably you can just use the prompt in UI of your choice to get the same or similar result by just setting temperature and seed to 0 (not sure if it matters, but my test script by default sets top-k=40, top-p=0.9, min-p=0.1, max-tokens=1024).

This is how I compiled ik_llama.cpp (after running "git clone" in the ~/pkgs folder):

```
cd ~/pkgs && cmake ik_llama.cpp -B ik_llama.cpp/build -DGGML_CUDA_FA_ALL_QUANTS=ON -DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=ON -DLLAMA_CURL=ON && cmake --build ik_llama.cpp/build --config Release -j --clean-first --target llama-quantize llama-cli llama-server
```

This is how I run it:

```
~/pkgs/ik_llama.cpp/build/bin/llama-server \
--model /mnt/secondary/neuro/Llama-4-Scout-17B-16E-Instruct-UD-Q2_K_XL.gguf \
--ctx-size 81920 --n-gpu-layers 49 --tensor-split 25,25,25,25 -fa -ctk q8_0 -ctv q8_0 \
--threads 64 --host 0.0.0.0 --port 5000
```

---

👤 **ikawrakow** commented the **2025-04-24** at **08:53:02**:<br>

Thank you for this! I can now reproduce it with my setup (single GPU). I was concerned that the bug was somehow related to splitting the model, which would have made it impossible for me to debug. I can now try to find the issue.

---

👤 **ikawrakow** commented the **2025-04-24** at **11:22:27**:<br>

@Lissanro 

#342 should fix it. Can you confirm that it works on your end? Thanks.

---

👤 **Lissanro** commented the **2025-04-25** at **00:35:45**:<br>

It seems to fix it.

I noticed that output is not identical between llama.cpp and ik_llama.cpp given exactly the same deterministic settings and seed, but perhaps this is normal and caused by different implementation. But I though I share this observation just in case.

ik_llama.cpp output:
https://pastebin.com/c8vKhm69

llama.cpp output:
https://pastebin.com/SXi15Dh5

By the way, can you please share an exact command to measure perplexity? I could run it on my side to see if there is a difference in perplexity between ik_llama.cpp and llama.cpp, if this a potentially useful information.

I also tested scout with longer 200K+ input prompt: https://dragon.studio/2025/04/prompt-long.txt - basically, I just added few more AI related Wikipedia articles, and then one long bat-related article (also from Wikipedia, to see if Scout can pay attention to the content, and if there is a difference between llama.cpp and ik_llama.cpp in output quality.

[llama.cpp output](https://pastebin.com/0xZAAkaH) and [ik_llama.cpp output](https://pastebin.com/nY7MTyTT) is different but it seems to be of similar quality (in both cases Scout completely missed my prompt at the beginning and all AI related articles).

My prompt was:

```txt
Provide a brief summary for articles below. First, list all article titles that I shared below, then, for each article, write a brief few paragraps summary.

[many long articles about AI, then one long article about bats]
```

I also tested with UD-Q4_K_XL quant and it also produced output of similar quality in both llama.cpp and ik_llama.cpp, missing the prompt in the beginning and AI related articles, focusing only on the bat article at the end.

If ik_llama.cpp is expected to generated different output given the same seed and zero temperature, then I think this bug can be considered fixed, since as far as I can tell both llama.cpp and ik_llama.cpp produce output of similar quality (after applying the patch you just shared).

---

👤 **ikawrakow** commented the **2025-04-25** at **07:00:27**:<br>

Thank you for testing.

The output of `llama.cpp` and `ik_llama.cpp` cannot be identical because the calculation is done in a different way, and floating point operations are not associative.

---

👤 **ikawrakow** commented the **2025-04-25** at **07:06:23**:<br>

> By the way, can you please share an exact command to measure perplexity? I could run it on my side to see if there is a difference in perplexity between ik_llama.cpp and llama.cpp, if this a potentially useful information.

To measure perplexity you use
```
./bin/llama-perplexity -m $your_model $other_parameters_you_use_for_server -f $file_containing_text -c context_length
```
The above perplexity values refer to `wiki.test.raw`, which is the test corpus everybody in the `llama.cpp` uses when referring to perplexity, and the command was
```
./bin/llama-perplexity -m Llama-4-Scout-17B-16E-Instruct-UD-Q2_K_XL.gguf -f ../../iquants/tests/wiki.test.raw -ub 2048 -t 32 -ngl 100 -c 16384 -ot "blk\.[0-8]\.ffn_up_exps=CUDA0,blk\.[0-8]\.ffn_down_exps=CUDA0,exps=CPU" -rtr -fmoe -fa
```
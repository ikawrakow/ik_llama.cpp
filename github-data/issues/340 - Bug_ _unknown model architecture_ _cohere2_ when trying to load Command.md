### 🐛 [#340](https://github.com/ikawrakow/ik_llama.cpp/issues/340) - Bug: \"unknown model architecture: 'cohere2'\" when trying to load Command A model

| **Author** | `Alexey-Akishin` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-04-22 |
| **Updated** | 2025-04-26 |

---

#### Description

### What happened?

It would be great if it was possible to run Command A in ik_llama.cpp (it works in llama.cpp). Currently when I try to load it, I get this:

```
llama_model_load: error loading model: error loading model architecture: unknown model architecture: 'cohere2'
llama_load_model_from_file: failed to load model
```

### Name and Version

I tried with newest version from this repo

### What operating system are you seeing the problem on?

Linux

### Relevant log output

```shell

```

---

#### 💬 Conversation

👤 **ikawrakow** commented the **2025-04-22** at **17:19:53**:<br>

I can look into adding it, but I don't have the bandwidth to test every model. Are you willing to test?

---

👤 **saood06** commented the **2025-04-22** at **17:29:22**:<br>

I could test, there is a [small model](https://huggingface.co/dranger003/c4ai-command-r7b-12-2024-GGUF) for it as well. I looked into the code, port looked simple (but would need to be redone because of their refactorings).

---

👤 **Alexey-Akishin** commented the **2025-04-22** at **17:34:13**:<br>

I will be more than happy to test, I build ik_llama.cpp from source, so for example I can test a patch when it is available, no problem.

---

👤 **mcm007** commented the **2025-04-25** at **07:23:19**:<br>

Tested on CPU only, the small 7B model works OK with #341 .

---

👤 **Alexey-Akishin** commented the **2025-04-25** at **09:25:05**:<br>

Unfortunately it did not work for me with Command A. I just asked it to summarize first few paragraphs from wiki article about "dog":

```
# Article Summary?

## Introduction?
-? Dogs? (Canis familiaris or Canis lupus familiaris) are? domesticated? descendants? of?? gray? wolves.
-? First?? species? domesticated? by?? humans? over?? 14,000?? years?? ago.
?
## Domestication? and? Diet?
-? Selectively??? bred? from? an?? extinct? population? of? wolves? during? the?? Late? Pleistocene.
-? Adapted??? to? thrive? on?????????????? starch-rich??????????????????????? diet? due? to? long??? association? with?????????? humans.
?
## Physical? Attributes? and? Senses?
-? Varied? breeds? with?? different?? shapes,??? sizes,? and?????????????????? colors.
-? Possess??? powerful? jaws? with????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
```

Question marks did not want to stop so I interrupted generation.

For comparison, llama.cpp result:

```
The article provides an overview of the domestication, evolution, and roles of dogs in human society. Here's a concise summary:

Dogs, descended from gray wolves, were domesticated over 14,000 years ago, making them the first species tamed by humans. They have adapted to thrive on a starch-rich diet and possess enhanced senses of smell and hearing. Bred for various traits, dogs serve multiple purposes, including companionship, therapy, and assistance. The strong human-canine bond has led to extensive research, solidifying dogs' status as "man's best friend." Globally, the dog population is estimated at 700 million to 1 billion, with most living in developing countries as feral or community animals.
```

Model I used for testing: https://huggingface.co/bartowski/CohereForAI_c4ai-command-a-03-2025-GGUF/tree/main/CohereForAI_c4ai-command-a-03-2025-IQ4_NL

---

👤 **ikawrakow** commented the **2025-04-25** at **09:52:50**:<br>

It looks like something not quite right with the vocabulary.  So, I guess, I need to test with this specific model.

---

👤 **ikawrakow** commented the **2025-04-25** at **10:29:47**:<br>

@Alexey-Akishin 

Can you also provide the specific command line you are using? And the details of the system you are running on (GPU(s), CPU).

Thanks.

---

👤 **ikawrakow** commented the **2025-04-25** at **11:16:54**:<br>

So, downloaded this specific model. Works fine on the CPU. Produces gibberish on the GPU with partial offload.Is this model another one of those where one needs `fp32` precision for it to work?

---

👤 **ikawrakow** commented the **2025-04-25** at **13:00:29**:<br>

> Is this model another one of those where one needs fp32 precision for it to work?

Yes, it is. Setting the precision of the `K*Q` matrix multiplication to `fp32` fixes the gibberish on CUDA. The current state of #341 should also work with the 111B parameter Command-A model.

---

👤 **Alexey-Akishin** commented the **2025-04-25** at **21:38:17**:<br>

I just tried latest #341 patch and it works well now! You are right, I was using CUDA (loading the whole model to GPUs). Thank you so much for adding support for Command A!

---

👤 **ikawrakow** commented the **2025-04-26** at **06:12:51**:<br>

OK, thanks for testing. I'll merge the PR and close the issue.
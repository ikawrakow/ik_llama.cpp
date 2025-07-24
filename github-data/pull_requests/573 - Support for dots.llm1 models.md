### 🔀 [#573](https://github.com/ikawrakow/ik_llama.cpp/pull/573) - Support for dots.llm1 models

| **Author** | `saood06` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-07-03 |
| **Updated** | 2025-07-10 |

---

#### Description

Port of https://github.com/ggml-org/llama.cpp/pull/14118

It compiles. Testers welcome.

Edit: Tested myself a tiny bit (more testers still welcome), see comment below.

Huggingface link to models: [instruct](https://huggingface.co/rednote-hilab/dots.llm1.inst), [base](https://huggingface.co/rednote-hilab/dots.llm1.base)

---

#### 💬 Conversation

👤 **saood06** commented the **2025-07-03** at **04:59:14**:<br>

> I am testing using UD-Q4_K_XL, and it is working.

Thanks.

>I notice an issue that if I leave system prompt empty, sometimes the response becomes unrelated to my question. With system prompt, it is fine. Do you also see this? I have the same issue when I run it from mainline.

If it exists in mainline then maybe it is a problem with the model? I haven't seen it but I haven't tested the model further than my comment above.

---

👤 **ikawrakow** submitted a review the **2025-07-03** at **06:19:04**: 🔄 `CHANGES_REQUESTED`

---

👤 **saood06** commented the **2025-07-04** at **00:05:25**:<br>

>  Not sure if there is better way.

That fix is only for the incorrect BOS token, which to me seems like an issue with existing models caused by the convert script which is where the fix should happen (with workarounds like [this](https://huggingface.co/gghfez/dots.llm1.inst-GGUF/discussions/1 for existing models) .

Both the `config.json` and `tokenizer_config.json` are set to null, which makes it take the default, but that doesn't seem to be correct for this model at least.

---

👤 **firecoperana** commented the **2025-07-04** at **00:10:41**:<br>

Without the fix, the model uses comma as BOS token that causes the pause, as least for the quant I'm using. See the screenshot I posted. Id 11 is the comma. After I set to null, comma is not used as BOS token.

---

👤 **saood06** commented the **2025-07-04** at **00:24:53**:<br>

> Without the fix, the model uses comma as BOS token that causes the pause, as least for the quant I'm using. See the screenshot I posted. Id 11 is the comma. After I set to null, comma is not used as BOS token.

Well the comma still causes a pause (I'm assuming) even if you avoid encountering it from the BOS token by setting the BOS token.

I've seen the screenshot you posted, and I also see the wrong BOS token in my own GGUF that I converted as part of the testing here (from safetensors to BF16 GGUF). Using `--override-kv tokenizer.ggml.bos_token_id=int:-1` like you linked above fixes it for affected models, but for future models to not be affected I think the convert script needs to explicitly set it, without changing the default like the `llama.cpp` change you suggested does.

---

👤 **saood06** submitted a review the **2025-07-09** at **17:29:30**: 💬 `COMMENTED`

---

👤 **saood06** commented the **2025-07-09** at **17:45:47**:<br>

> @saood06 What are your plans with this PR?

Sorry kept pushing off testing this more, but I just pushed a commit with both the recommended changes.

>You are disagreeing [...] about the `BOS` token

I still think the better solution would have been for the convert script to set it to `-1` when config.json has it set to `NULL` instead of leaving it to be set to default and changing the default for this architecture, but given the fact that every GGUF I saw on huggingface has this issue, changing the default so that users don't have to set `--override-kv tokenizer.ggml.bos_token_id=int:-1` (assuming they know to do that) or some other workaround makes sense.

I also changed the warmup behavior to work with this model (a MoE without a BOS token), it is still the same hacky solution but now it does account for models without a BOS token, and it did warmup for me now (not sure why it wasn't with BOS set to [token id 11/`,`]).

---

👤 **ikawrakow** submitted a review the **2025-07-10** at **06:31:53**: ✅ `APPROVED`
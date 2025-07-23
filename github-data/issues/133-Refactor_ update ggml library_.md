### 📝 [#133](https://github.com/ikawrakow/ik_llama.cpp/issues/133) - Refactor: update ggml library?

| **Author** | `Nexesenex` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2024-12-11 |
| **Updated** | 2025-03-21 |

---

#### Description

### Background Description

Hey IK,

It becomes harder and harder to merge your work into my fork of KoboldCPP. I'm advancing well, but now I'm hitting the ggml_library barrier.

For example, to merge :
https://github.com/ikawrakow/ik_llama.cpp/pull/9/files#diff-f028a352a33ee20b42faca7dcc389e8f0f9c9a55e016cccffed45fe90bcc13f8R5907

into a current version of KoboldCPP,
I need :

https://github.com/ggerganov/ggml/pull/988

because

"grad" is not a member of ggml_tensor anymore

```
"static struct ggml_tensor * ggml_softcap_impl(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        float                 s_before,
        float                 s_after,
        bool inplace) {
    GGML_ASSERT(ggml_is_padded_1d(a));

    bool is_node = false;

    if (a->grad) {   // <---------------------------
        is_node = true;
    }"
```

I merged and made work on my KCPP fork your first batch of IK quants (2,3,4,5,6) on Cuda, but I also meet trouble to refactor properly the cuda side of things for your more recent quants (specifically on the dot product template modification, even if I might be able to handle that one by myself with more digging into the factoring, I'm not sure).

Anyway, do you have plans to update IK_Llama's GGML Library, or even the whole Llama.CPP (I'm not asking for that last one, though) in the future? I'd love to keep using your work, and integrating it into my KCPP fork is a very good exercise for me to learn, but integrating your work into KCPP without the current ggml library is just too much for me to handle, as is to rebase everything on IK_Llama considering that KCPP mainline follows the developments of Llama.CPP, and thus of the ggml library.

### Possible Refactor Approaches

For you to decide!

---

#### 💬 Conversation

👤 **ikawrakow** commented the **2024-12-11** at **10:18:44**:<br>

Well, it is hopelessly diverged now.

---

👤 **Nexesenex** commented the **2024-12-11** at **16:32:40**:<br>

Yeah, i quite guessed so..

Then, could you maybe look up at the commits of the ggml library since you last updated LCPP mainline, and then update IK LCPP to the last state the ggml library didn't become too divergent yet to be a real hassle for you to handle, so backtracking to a previous point of the library doesn't bring me (or whoever would like to integrate your work into his own inference software) back too far?

---

👤 **ikawrakow** commented the **2024-12-11** at **17:27:55**:<br>

> Then, could you maybe look up at the commits of the ggml library since you last updated LCPP mainline...

Most of the changes I have made are in `ggml`, not `llama.cpp`. So, no, picking up mainline `ggml` changes cannot be done quickly. They added a thread pool (I didn't like it, and, predictable, there were bugs related to that for quite some time), they refactored the back-end for the 77'th time, they started working on turning `ggml` into an actual machine learning library rather than the inference framework it actually is (the PR you are referring to above is one of the massive changes related to that), there was a massive change in the Metal objective-C code for no real gain (my fork is still faster), etc.

---

👤 **Nexesenex** commented the **2024-12-11** at **18:24:44**:<br>

Yeah, I'm quite upset about this. I feel like the coyote chasing speedy gonzales with all these refactors. Llama.CPP is first and foremost inference software (if I'm not mistaken), and integrating fully into it what is becoming a dual use library make things very complex for fork maintainers.

So, I reverted back my KCPP fork to pre ggml 988, and i could integrate your PRs https://github.com/ikawrakow/ik_llama.cpp/pull/9/files#diff-f028a352a33ee20b42faca7dcc389e8f0f9c9a55e016cccffed45fe90bcc13f8
and https://github.com/ikawrakow/ik_llama.cpp/pull/24/files#diff-f028a352a33ee20b42faca7dcc389e8f0f9c9a55e016cccffed45fe90bcc13f8

I'm onto merging https://github.com/ikawrakow/ik_llama.cpp/pull/28/files#diff-f028a352a33ee20b42faca7dcc389e8f0f9c9a55e016cccffed45fe90bcc13f8
right now, because I use long context and I want the speed bump.

Next, I will attack your IQ_K post 1st gen quants and the cuda refactor problematic for me, because you made a IQ4_KSS for me, and I want to use it in my own preferred inference software, as well as your other quants (trellis quants are interesting for me to test, because with proper sampling, lower bpw SOTA quants of very big models can become quite usable, and I need that to fully offload Mistral 123b with a huge context, image gen models, & so on). :)

---

👤 **Nexesenex** commented the **2024-12-11** at **18:24:44**:<br>

Yeah, I'm quite upset about this. I feel like the coyote chasing speedy gonzales with all these refactors. Llama.CPP is first and foremost inference software (if I'm not mistaken), and integrating fully into it what is becoming a dual use library make things very complex for fork maintainers.

So, I reverted back my KCPP fork to pre ggml 988, and i could integrate your PRs https://github.com/ikawrakow/ik_llama.cpp/pull/9/files#diff-f028a352a33ee20b42faca7dcc389e8f0f9c9a55e016cccffed45fe90bcc13f8
and https://github.com/ikawrakow/ik_llama.cpp/pull/24/files#diff-f028a352a33ee20b42faca7dcc389e8f0f9c9a55e016cccffed45fe90bcc13f8
I'm on https://github.com/ikawrakow/ik_llama.cpp/pull/28/files#diff-f028a352a33ee20b42faca7dcc389e8f0f9c9a55e016cccffed45fe90bcc13f8
right now, because I use long context and I want the speed bump.
Next, I will attack your post IQ_K 1st gen quants and the cuda refactor problematic for me, because you made a IQ4_KSS for me, and I want to use it in my own preferred inference software, as well as your other quants (trellis quants are interesting for me to test, because with proper sampling, lower bpw SOTA quants of very big models can become quite usable, and I need that to fully offload Mistral 123b with a huge context, image gen models, & so on). :)

---

👤 **ikawrakow** commented the **2025-03-21** at **12:40:05**:<br>

I guess this will not happen. It will be easier to take current `llama.cpp` and apply the changes I have done here than to try syncing this totally diverged fork with upstream.
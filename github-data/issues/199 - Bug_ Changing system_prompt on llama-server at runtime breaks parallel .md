### 🐛 [#199](https://github.com/ikawrakow/ik_llama.cpp/issues/199) - Bug: Changing system_prompt on llama-server at runtime breaks parallel processing

| **Author** | `saood06` |
| :--- | :--- |
| **State** | ✅ **Open** |
| **Created** | 2025-02-09 |
| **Updated** | 2025-04-25 |

---

#### Description

The motivation for me testing batched performance was to have multiple streams of completion from the same prompt. Sharing a prompt via system_prompt saves allocating KV.

Setting system_prompt at launch does work to allow this with high performance, but changing it at runtime which is needed in order to keep KV cache allocation low results in slots processing sequentially. I did come up with some workarounds but none that were viable, restarting the server with the new prompt did work but has the major downside of having to reprocess the entire prompt. Saving and restoring KV cache using the ```/slots/{id_slot}?action=save``` was thought of but not implemented.

---

#### 💬 Conversation

👤 **ikawrakow** commented the **2025-02-11** at **06:02:32**:<br>

Is this something that has been fixed in mainline but does not work here?

---

👤 **saood06** commented the **2025-02-11** at **10:33:25**:<br>

It does not exist in mainline because mainline removed system_prompt support, although they plan to add support for a new feature that accomplishes the same thing that I am (multiple options in parallel for a single completion in a KV efficient way). I don't think they realized system_prompt (if it didn't have this bug) could be used this way as no front end implemented it to do so ( I had to mod support into one to test it).

---

👤 **saood06** commented the **2025-02-11** at **10:33:25**:<br>

It does not exist in mainline because mainline removed system_prompt
support, although they plan to add support for a new feature that
accomplishes the same thing that I am (multiple options in parallel for a
single completion in a KV efficient way). I don't think they realized
system_prompt (if it didn't have this bug) could be used this way as no
front end implemented it to do so ( I had to mod support into one to test
it).


On Tue, Feb 11, 2025, 12:02 AM Kawrakow ***@***.***> wrote:

> Is this something that has been fixed in mainline but does not work here?
>
> —
> Reply to this email directly, view it on GitHub
> <https://github.com/ikawrakow/ik_llama.cpp/issues/199#issuecomment-2649876588>,
> or unsubscribe
> <https://github.com/notifications/unsubscribe-auth/ADLOJ3JHME265HPEN3Q7EH32PGHA3AVCNFSM6AAAAABWZKEUOWVHI2DSMVQWIX3LMV43OSLTON2WKQ3PNVWWK3TUHMZDMNBZHA3TMNJYHA>
> .
> You are receiving this because you authored the thread.Message ID:
> ***@***.***>
>

---

👤 **VinnyG9** commented the **2025-04-25** at **06:01:15**:<br>

> It does not exist in mainline because mainline removed system_prompt support, although they plan to add support for a new feature that accomplishes the same thing that I am (multiple options in parallel for a single completion in a KV efficient way). I don't think they realized system_prompt (if it didn't have this bug) could be used this way as no front end implemented it to do so ( I had to mod support into one to test it).

isn't this the same as chat template flag?

---

👤 **saood06** commented the **2025-04-25** at **06:26:12**:<br>

> > It does not exist in mainline because mainline removed system_prompt support, although they plan to add support for a new feature that accomplishes the same thing that I am (multiple options in parallel for a single completion in a KV efficient way). I don't think they realized system_prompt (if it didn't have this bug) could be used this way as no front end implemented it to do so ( I had to mod support into one to test it).
> 
> isn't this the same as chat template flag?

No, this allows you to more efficiently use the KV for multiple slots as the system_prompt is only allocated once and is used in all slots. For example if you store 30,000 tokens in system_prompt and then use 10 slots you can set KV cache to 40,000 and each slot would get 31,000 tokens (30K shared, 1K unique), and without using the system_prompt to get 31,000 tokens per slot would need a KV of 310,000 tokens which with most models is resource intensive, but this is only useful if you have a use for a large shared prefix between slots. 

I do plan to improve the KV situation in server, but right now I am leaning toward doing something else though and not starting from system_prompt.

---

👤 **saood06** commented the **2025-04-25** at **06:26:12**:<br>

> > It does not exist in mainline because mainline removed system_prompt support, although they plan to add support for a new feature that accomplishes the same thing that I am (multiple options in parallel for a single completion in a KV efficient way). I don't think they realized system_prompt (if it didn't have this bug) could be used this way as no front end implemented it to do so ( I had to mod support into one to test it).
> 
> isn't this the same as chat template flag?

No, this allows you to more efficiently use the KV for multiple slots as the system_prompt is only allocated once and is used in all slots. For example if you store 30,000 tokens in system_prompt and then use 10 slots you can set KV cache to 40,000 and each slot would get 31,000 tokens (30K shared, 1K unique), and without using the system_prompt to get 31,000 tokens per slot would need a KV of 310,000 tokens which with most models is resource intensive. 

I do plan to improve the KV situation in server, but right now I am leaning toward doing something else though and not starting from system_prompt.
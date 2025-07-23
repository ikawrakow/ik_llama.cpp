### 🗣️ [#451](https://github.com/ikawrakow/ik_llama.cpp/discussions/451) - Context reuse / context shift for long prompts

| **Author** | `SamuelOliveirads` |
| :--- | :--- |
| **Created** | 2025-05-23 |
| **Updated** | 2025-06-10 |

---

#### Description

Hi! — I'm coming from koboldcpp, and I've been testing this fork due to its optimizations.

One feature I found very useful in koboldcpp was the context shift functionality, which helps when working with very long context windows.

I noticed that `llama.cpp` implemented something similar in [PR #9866](https://github.com/ggml-org/llama.cpp/pull/9866), which allows for reusing the prompt cache more efficiently instead of regenerating the entire prompt every time the context overflows.

I searched through this repo but couldn’t find an equivalent implementation.

Here’s the issue I’m currently facing:
- I'm using a 62k context in Qwen 3.
- When the context overflows, the cache keeps my system prompt, but discards the conversation history.
- That leads to reprocessing ~58k tokens from scratch each time, which at ~40 tokens/sec takes several minutes per new message.
- With proper cache reuse (like in llama.cpp), this would take just seconds.

My question is:  
- Is there already something similar to context reuse implemented here?
- If not, would this be something feasible to implement, perhaps inspired by how llama.cpp did it?

Thanks!

---

#### 🗣️ Discussion

👤 **mtcl** replied the **2025-05-30** at **16:47:09**:<br>

This is a very useful usecase because of which I have been switching back and forth between ik_llama.cpp and llama.cpp. This works seamlessly with llama.cpp i have noticed. I always thought I am doing something wrong here and it is my user error, but apparantly it is not! Thank you for mentioning it here.

---

👤 **cmoncure** replied the **2025-05-30** at **19:51:44**:<br>

This would be a massive win for me.  Currently PP is the millstone around the neck (for which you have had to endure many of my ignorant comments in support of a solution).

KV Cache reuse and tool calling would open up whole new worlds.

> 👤 **mtcl** replied the **2025-06-05** at **02:26:48**:<br>
> I agree 100% with you. Given that I built my own tool calling solution for ik_llama.cpp, at this point of time kv cache reuse would mean an instant switch for me to this!

---

👤 **SamuelOliveirads** replied the **2025-06-03** at **21:52:10**:<br>

Glad to see that others are also interested in this feature! I was about to open an issue myself, but I noticed that @saood06 is already looking into something similar [here](https://github.com/ikawrakow/ik_llama.cpp/issues/455#issuecomment-2917718499) — so now it’s just a matter of waiting.

By the way, @saood06, if you need any help with testing, I’d be happy to assist.

> 👤 **saood06** replied the **2025-06-06** at **09:16:14**:<br>
> Since there does seem to be demand, and people waiting, I'll provide an update which explains what my plan is (and the benefits, but also the limitations), and the current status.
> 
> The goal is to create a new mechanism where if enabled a [trie](https://en.wikipedia.org/wiki/Trie) of all processed tokens is kept that can be saved and restored to a file. This should allow you to keep every explored branch of a session (or multiple if you share a large initial prompt between sessions) with the least amount of space and no quality loss.
> 
> This may only be viable on MLA models as they are extremely light for KV cache, and this method does not degrade quality like chunking or shifting, but for that reason this does not handle the common case of shifting the cache when you want to remove the thought tokens without having to reprocess as there is no way to do that without losing (at least some) quality. 
> 
> I was stalled because of #436 but now that saving and loading works I am now unblocked, but this still seems like a large undertaking and may take some time.
> 
> I may end up porting the chunk/shift method (or @cmoncure is welcome to do it) anyway (even before I finish), since as I said they have different tradeoffs, but integrating the two fully as nice as it sounds (which would let you be able to chunk and shift from the trie) seems way too difficult.
> 
> 👤 **cmoncure** replied the **2025-06-06** at **15:16:33**:<br>
> Do you have any insight into the nature or mechanism behind the quality loss with chunking?
> 
> 👤 **ikawrakow** replied the **2025-06-06** at **15:29:13**:<br>
> Are we talking about the `llama.cpp` feature (taken from kobold.cpp) where if I have
> ```
> aaaaccccbbbb
> ```
> in the KV cache, and the new context is
> ```
> aaaabbbb
> ```
> I can reuse the full `aaaabbbb` (mainline `llama.cpp`) instead of just reusing `aaaa` as it happens here?
> 
> If so, here is an example:
> 
> **KV cache:** Yesterday I saw a movie. I absolutely enjoyed it. The main actor was ... 
> **New context:** Yesterday I saw a movie. The main actor was
> 
> Suppose **New context** is in the context of the worst movie you have ever seen, so you expect "a disaster" or some such.
> The existing KV cache, despite context shifting and all that, will be heavily biased towards "brilliant", "amazing" and such.
> 
> Do you see the problem? You cannot undo the impact of the skipped tokens by just changing the position encoding via RoPE.
> 
> 👤 **saood06** replied the **2025-06-06** at **15:41:47**:<br>
> > Are we talking about the `llama.cpp` feature (taken from kobold.cpp) where if I have
> 
> Yes that is what we are talking about. Thank you for the very clear example (so much better than what I was typing out).
> 
> I'm not sure this is from kobold.cpp. I know they offer a much better context shift where they effectively keep the context full at all times once you hit the limit unlike llama.cpp and here where the context shift unnecessarily removes far more tokens than is needed (I think half) and thus shifts are less frequent. Kobold.cpp on the other hand shifts every token which keeps the maximum information allowed at all times.
> 
> 👤 **cmoncure** replied the **2025-06-06** at **19:40:13**:<br>
> >You cannot undo the impact of the skipped tokens by just changing the position encoding via RoPE.
> 
> So...
> 
> 1. KV Cache is a Key-Value cache
> 2. KV Cache as a "memoization" technique stores the results of the expensive PP computation for reuse.
> 3. But the PP computation is cumulative in such a way that the presence and order of tokens matters.
> 4. Once a token has acted on the KV cache, its effect poisons the KV cache indelibly.
> 
> Questions:
> 
> 1. Is the effect of tokens on the KV cache _additive_ or _multiplicative_ (or something else)?  If additive, can the effect of tokens removed from the prompt be recalculated and their effect subtracted?
> 2. If the presence of token PP computation in the KV cache poisons it forever, then doesn't that imply that tokens outside the context window can continue to affect generation? That would contradict my mental model of how all this is supposed to work. Edit: I suppose that's why the whole thing must be scrapped each time when the context window fills up. It makes sense.
> 
> 👤 **saood06** replied the **2025-06-07** at **06:17:39**:<br>
> >     4. Once a token has acted on the KV cache, its effect poisons the KV cache indelibly.
> > 
> > 
> > Questions:
> > 
> >     2. If the presence of token PP computation in the KV cache poisons it forever, then doesn't that imply that tokens outside the context window can continue to affect generation? That would contradict my mental model of how all this is supposed to work. Edit: I suppose that's why the whole thing must be scrapped each time when the context window fills up. It makes sense.
> 
> No. If that were the case then you could not have multiple slots which serve independent users that share the KV cache, but that is a well supported use case.
> 
> The tokens do not "poison" the cache, it is just that a token holds the information of all prior tokens from that sequence when it was calculated. If you get rid of tokens and then shift tokens that had come after the now deleted tokens in order to re-use them the shifted tokens will still contain the information from the deleted tokens.
> 
> To add to the the example given above with the movie, even though you removed the tokens "I absolutely enjoyed it.", their influence is not gone if you keep the tokens after and shift them.
> 
> If you shift "The main actor was" then you will see the influence of the removed tokens (but it will be much faster as you are not recomputing those tokens).
> 
> If you do recompute the tokens "The main actor was"  and do not shift then it will be slower (as you have to actually compute the tokens again) but you will not experience the lingering impact of "I absolutely enjoyed it."
> 
> 👤 **cmoncure** replied the **2025-06-10** at **02:35:21**:<br>
> >If you do recompute the tokens "The main actor was" and do not shift then it will be slower (as you have to actually compute the tokens again) but you will not experience the lingering impact of "I absolutely enjoyed it."
> 
> Forgive me if I've misunderstood.  Suppose we have the following prompt:
> 
> `AAAABBBBCCCC`
> 
> Then we can understand the state of the fully processed KV cache to be something like the following, where some function `f(X) :-> x` gives the "effect" of the token on subsequent tokens:
> 
> `A A A A Ba Ba Ba Ba Cab Cab Cab Cab`
> 
> I'm stretching the truth a bit here for the purposes of a convenient representation. But the above illustrates that each part of the prompt carries with it information about the previous parts.
> 
> Suppose that our context grows and our `A` tokens must be pushed off the top of the context window. Then we have some intermediate state
> 
> `Ba Ba Ba Ba Cab Cab Cab Cab D D D D`
> 
> In order to create a properly functioning KV cache, we have to effectuate the following:
> 
> 1. The effect of `A` tokens must be removed from `B` and `C`
> 2. D tokens must take into account `B` and `C`
> 
> So that finally, we have
> 
> `B B B B Cb Cb Cb Cb Dbc Dbc Dbc Dbc`
> 
> The way this is currently achieved is (if I am not mistaken) by dropping and re-processing the entire cache pertaining to the prompt, which is expensive, suggesting an algorithmic complexity of O(n^2). Can we not instead of re-processing the entire prompt, simply calculate f(A) and subtract it from the following tokens (or the inverse f'(A) and add it):
> 
> `Ba Ba Ba Ba Cab Cab Cab Cab` - f(A) => `B B B B Cb Cb Cb Cb`
> 
> Finally computing the rest of the prompt only against D: 
> 
> `D D D D` + f(B) + F(C) => `Dbc Dbc Dbc Dbc`
> 
> Then concatenate the two to get the desired state? I'm still reading through llama.cpp... it's a lot.

---

👤 **cmoncure** replied the **2025-06-05** at **18:35:28**:<br>

Might have to do it myself.
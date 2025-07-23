### 🗣️ [#403](https://github.com/ikawrakow/ik_llama.cpp/discussions/403) - Tool Calling and Structured Response (Json Mode) support

| **Author** | `mtcl` |
| :--- | :--- |
| **Created** | 2025-05-10 |
| **Updated** | 2025-05-30 |

---

#### Description

Hey Team,

Amazing work here. as compared to llama.cpp the biggest feature that I see missing is support for tool calling. D oyou have any plans to include it in the future roadmap? Or am i missing something and it alredy exists?

I am forced to use other frameworks, even though i like inferencing speeds from ik_llama.cpp, just beacuse i cant live without these features and want to swap it out natively in the openai's python client in my project implementation. 

I know tha i can prompt the model in a particular way to force it to produce a json response. I am not looking for that.

Thank you in advance!

---

#### 🗣️ Discussion

👤 **ikawrakow** replied the **2025-05-10** at **08:30:16**:<br>

Hey @mtcl,

we are a very small team, so cannot do everything that `llama.cpp` does. Hence, the strategy is to focus on few things, but do these things really well.

Please enter a feature request in the Issues. I'll label it with "help wanted" and we will see what happens.

> 👤 **mtcl** replied the **2025-05-10** at **08:33:02**:<br>
> No worries my friend. I have a workaround here that I've written.
> 
> https://github.com/Teachings/FastAgentAPI
> 
> It acts as a wrapper and get me by. Thank you for your hard work!
> 
> 👤 **cmoncure** replied the **2025-05-30** at **19:58:13**:<br>
> Before I try and get this running, can you educate me on the mechanics of tool calling within the LLM response? I understand that the LLM may request a call as part of its TG phase, and then the call runner injects the result into the LLM response. Is this correct? 
> 
> I have some questions about this.  Suppose I want to ask the LLM a question about a long document.
> 
> What's the difference in outcome between:
> 1) Including the question and document in the prompt, and enduring the long PP time
> 2) Including the question in the prompt, and having the LLM retrieve the document instantly via tool call during TG, then going on to complete the response?
> 
> Do all injected tokens need to undergo a form of 'PP during TG'?  That would make the most sense, actually...

---

👤 **KCS-Mack** replied the **2025-05-18** at **22:28:59**:<br>

This is great, will give it a try!
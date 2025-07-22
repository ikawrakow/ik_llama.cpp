### 📝 [#362](https://github.com/ikawrakow/ik_llama.cpp/issues/362) - README language is vague wrt. \"quantization improvements\"

| **Author** | `usrlocalben` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-04-30 |
| **Updated** | 2025-05-13 |

---

#### Description

### What happened?

The new README commit text indicates recent _quantization improvements_ but it's not clear what that means.

e.g.,
- Are they now correct? (previously in error?)
- Are they more accurate? (previously out of spec?)
- Is the implementation more efficient?
  - ...during inference?
  - ...during quantization?
- ...or more memory efficient?

And similarly,
- Are old quants compatible? (or even valid?)
- Should they be recomputed? 


### Name and Version

https://github.com/ikawrakow/ik_llama.cpp/commit/98d1626469879d35faba9cb7e9d0b1ddaf853eee

---

#### 💬 Conversation

👤 **saood06** commented the **2025-04-30** at **23:21:14**:<br>

As the README mentions you can often find detailed information in PRs. https://github.com/ikawrakow/ik_llama.cpp/pull/295 and  https://github.com/ikawrakow/ik_llama.cpp/pull/302 are the related PRs

---

👤 **ikawrakow** commented the **2025-05-01** at **16:41:52**:<br>

Would you like to have links to the specific PR's in the News section? I did try this along with a short description initially, but then it becomes kind of too long for a News section.

But to address your points:

> Are they now correct? (previously in error?)

That would a be a fix, not an improvement

> Are they more accurate? (previously out of spec?)

There isn't such a thing as a spec for a quantization method. You can never predict in advance how accurate the method is going to be, and then it also differs from model to model. Not to mention the fact that people haven't even agreed on the right way to measure the accuracy of a quantization method. So, basically, it is impossible to write a spec so that the method and the implementation can be determined to meet or not  meet the spec.

But yes, improving accuracy is one of the ways how one can improve quantization.

> Is the implementation more efficient?
> ...during quantization?

That's the only other thing that comes to mind when thinking about quantization improvements. I wouldn't consider making inference more efficient for certain quantization types as quantization improvement, but rather as a performance improvement for certain quantization types.  

> Are old quants compatible? (or even valid?)

Breaking changes are clearly indicated

> Should they be recomputed?

Here is where the user needs to understand what the improvement was so they can decide if it is worth re-quantizing their model(s). And for that, one needs to find the PR's by typing "is:pr quantization improvements" in the search box. For instance, I tend to measure quantization accuracy using perplexity, but there are a lot of people out there who disagree that this is the right way. So, as a user making their own quants, you do really need to read what was improved and decide for yourself. And providing enough information so the user can do that is way out of scope for a News section.

---

👤 **usrlocalben** commented the **2025-05-13** at **13:16:29**:<br>

Thanks for the commentary and also the README updates w/PR links on the line-items. I now resolve the language this way: To Quantize is a verb/action and therefore strongly refers to _computing_ the quant, i.e. llama-quantize. Closing
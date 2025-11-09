## 🔀 [Pull Request #502](https://github.com/ikawrakow/ik_llama.cpp/pull/502) - Add an endpoint that lists all the saved prompt caches to server

| **Author** | `saood06` |
| :--- | :--- |
| **State** | 🔀 **Merged** |
| **Source Branch** | `s6/list_prompt_cache` |
| **Target Branch** | `main` |
| **Created** | 2025-06-06 |
| **Updated** | 2025-06-11 |
| **Merged** | 2025-06-07 |

---

## 📄 Description

Now that saving the prompt cache works this adds a way to query all the currently saved prompt caches.

This should be enough to be used by any front end. The only thing that may potentially be useful to be added is giving the prompt in an array based on how the prompt is tokenized.

---

## 💬 Conversation

👤 **ikawrakow** approved this pull request ✅ on **2025-06-07** at **05:18:57**

---

👤 **saood06** commented on **2025-06-11** at **06:50:30**

>The only thing that may potentially be useful to be added is giving the prompt in an array based on how the prompt is tokenized.

Using it more that isn't nearly as useful as a timestamp (created? as these are write once), and some information about the model (architecture could work but even though you could share prompts between different models that share an architecture [and have the same number of layers], I'm pretty sure it can have bad results if the models differ enough).

I'm alleviating both of these by putting info about the model and numbering my saves but it would be better if the info above was returned and that way a frontend could also make use of it and improve ergonomics, and not all users will think to follow the approach I am.

The timestamp can be included trivially, but the model information as far as I can tell will be a breaking change to the session save format (there is some metadata included that prevents you from loading incompatible saves, but for the reasons listed above I don't think it is the best choice to output and use those, and they really aren't very human friendly).

I really don't want to make a breaking change (not just because it would break old saves [unless converted] but it would also break support with mainline, unless they also chooses to adopt it).

Edit: forgot to mention an endpoint allowing you to delete saved prompts might be worth adding.
### 🐛 [#342](https://github.com/ikawrakow/ik_llama.cpp/pull/342) - Fix LLaMA-4 attention

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-04-24 |
| **Updated** | 2025-04-25 |

---

#### Description

Closes #335 

I had missed the SWA part. As SWA only has a real impact past 8k tokens, and as the impact of not using SWA is relatively small for the next 8k tokens, the model appeared coherent up to 16k tokens.

It now produces the following summary of the first 23.5k tokens of the Wikipedia article on Artificial Intelligence:
```
Here is a brief summary of the article on Artificial Intelligence (AI):

**What is AI?**
Artificial intelligence refers to the capability of computer systems to perform tasks typically associated with human intelligence, such as learning, reasoning, problem-solving, perception, and decision-making.

**Applications of AI**
High-profile applications of AI include advanced web search engines, recommendation systems, virtual assistants, autonomous vehicles, generative and creative tools, and superhuman play and analysis in strategy games.

**Goals and Tools of AI**
The traditional goals of AI research include learning, reasoning, knowledge representation, planning, natural language processing, perception, and support for robotics. AI researchers have adapted and integrated various techniques, including search, mathematical optimization, formal logic, artificial neural networks, and methods based on statistics, operations research, and economics.

**Subfields of AI**
Subfields of AI research include machine learning, natural language processing, computer vision, and robotics. Machine learning is a study of programs that can improve their performance on a given task automatically.

**Techniques Used in AI**
Techniques used in AI include search and optimization, logic, probabilistic methods for uncertain reasoning, classifiers and statistical learning methods, artificial neural networks, and deep learning.

**Applications of AI in Various Industries**
AI is used in various industries, including healthcare, medicine, games, mathematics, finance, military, and education. AI has helped farmers identify areas that need irrigation, fertilization, pesticide treatments, or increasing yield. AI is also used in astronomy to analyze increasing amounts of available data and applications.

**Ethics of AI**
AI has potential benefits and potential risks. AI may be able to advance science and find solutions for serious problems, but as the use of AI has become widespread, several unintended consequences and risks have been identified. In-production systems can sometimes not factor ethics and bias into their AI training processes, especially when the AI algorithms are inherently unexplainable in deep learning.
```

Interestingly enough, PPL for a context of 16k tokens goes up after this change (7.27 vs 7.18). We are trading predictive power for the ability to process longer contexts.
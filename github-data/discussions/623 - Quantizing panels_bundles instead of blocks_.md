### 🗣️ [#623](https://github.com/ikawrakow/ik_llama.cpp/discussions/623) - Quantizing panels/bundles instead of blocks?

| **Author** | `jubruckne` |
| :--- | :--- |
| **Created** | 2025-07-17 |
| **Updated** | 2025-07-17 |

---

#### Description

Hi there! I much admire your work in this project. 

One thing I’ve been wondering… I believe weights are already repacked to make MatMul more efficient for the ffn... now I don’t understand the code well enough… are we (or could we possibly) also interleaving weight of w1,w2,w3 into panels? And then quantize based on this panels structures instead of individual blocked weight matrixes?

Maybe this doesn’t make my sense at all..  but I’ve been thinking about it for a while now, and it seems to me this could also open other possibilities like selecting variable Bitrate for each panel. Or sorting the panels by importance (derived from imatrix), and only calculating the most important ones (like top 50%). 

I apologize if some of this seems stupid, it probably is 🙈…

---

#### 🗣️ Discussion

👤 **ikawrakow** replied the **2025-07-17** at **12:19:22**:<br>

You mean, instead of having 256 weights from the same row in a block of 256, we could have used 32 x 8 from 8 different consecutive rows?
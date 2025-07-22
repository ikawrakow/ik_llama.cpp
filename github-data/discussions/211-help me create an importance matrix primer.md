### 🗣️ [#211](https://github.com/ikawrakow/ik_llama.cpp/discussions/211) - help me create an importance matrix primer

| **Author** | `robbiemu` |
| :--- | :--- |
| **Created** | 2025-02-19 |
| **Updated** | 2025-02-22 |

---

#### Description

this primer, if I am honest is mostly about the related main stream llama.cpp project, but the details are so general I think it generally applies. I was hoping @ikawrakow you might review this and help me to track down gaps and errors, before I release a final version. (I'm the [llama-gguf-optimize](https://github.com/robbiemu/llama-gguf-optimize) guy interested in language preservation, btw -- hello again! ).

(version: 0.3)

# importance matrices in Llama.cpp

## Architectural Design of Importance Matrices in Llama.cpp

Quantization reduces the precision of neural network weights and activations, lowering memory usage and computational costs. Early calibration methods, such as min-max scaling, determined quantization ranges based on observed activation values. Modern calibration-based methods typically select quantization parameters, such as scaling factors and offsets, by analyzing the network’s data distributions to improve accuracy.

### Background: On Quantization

The development of techniques to quantify weight importance in neural networks has roots in **network pruning**. This will introduce a Hessian related to the model's weights and performance, so it should be defined first.

The Hessian matrix $H$ is defined as the matrix of **second-order partial derivatives** of the loss $\mathcal{L}$ (like MSE, minimized during training, which compares _model outputs_ to target values) with respect to the model’s weights., composed of second-order partial derivatives $H_{ij} = \frac{\partial^2 \mathcal{L}}{\partial w_i \partial w_j}$. This Hessian effectively measures the local curvature of the error surface during training. Its eigenvalues and eigenvectors reveal the directions of greatest sensitivity in parameter space. A large value means the loss changes rapidly when that weight is modified (high curvature), while a small value indicates the loss is relatively flat with respect to that weight.

#### Network Pruning: Optimal Brain Damage and Optimal Brain Surgeon

Network pruning aims to remove redundant or non-essential weights without significantly degrading model performance. Early foundational work, such as **Optimal Brain Damage (OBD)** (LeCun et al., 1990) and **Optimal Brain Surgeon (OBS)** (Hassibi & Stork, 1993), formalized this process using second-order derivatives of the loss function. 

1. **Optimal Brain Damage (OBD):**  
   OBD approximates the sensitivity of the loss to weight removal by leveraging a **diagonal Hessian matrix**. The importance of a weight $w_i$ is computed as:

$$
\mathcal{I}_i = \frac{1}{2} w_i^2 \cdot H_{ii},
$$

   where $H_{ii}$ is the second derivative of the loss with respect to $w_i$. This diagonal approximation assumes that interactions between weights (off-diagonal Hessian terms) are negligible, drastically reducing computational complexity.

2. **Optimal Brain Surgeon (OBS):**  
   OBS generalizes OBD by incorporating the **full Hessian matrix**, capturing cross-interactions between weights. The saliency $\mathcal{S}_q$ of removing weight $w_q$ is given by:

$$
\mathcal{S}_q = \frac{w_q^2}{2 [H^{-1}]_{qq}},
$$

   where $[H^{-1}]_{qq}$ is the inverse Hessian’s diagonal entry for $w_q$. While more accurate, computing and inverting the full Hessian is computationally prohibitive for modern deep networks, limiting OBS’s practicality.

Both methods link weight importance to the curvature of the loss landscape in a global matrix of model weights. A weight with a large $H_{ii}$ (steep curvature) is highly sensitive—even small perturbations may destabilize the model. Conversely, a flat curvature ($H_{ii} \approx 0$) implies robustness to changes.

#### Hessian-Based Sensitivity Analysis

Exact Hessian computation is often infeasible for large networks due to its $O(N^2)$ memory cost (where $N$ is the number of weights).

In quantization, the goal is analogous to pruning: allocate higher precision (bits) to weights that most influence model output.
- **Sensitivity Metric for Quantization:**  
   The expected change to the loss from quantizing $w_i$ can be approximated as:

$$
\Delta \mathcal{L} \approx \frac{1}{2} \sum_i H_{ii} (\Delta w_i)^2,
$$

   where $\Delta w_i$ is the quantization error (essentially $q_i - w_i$ in the llama.cpp-specific formulation discussed later). To minimize $\Delta \mathcal{L}$, weights with large $H_{ii}$ (high sensitivity) should have smaller $\Delta w_i$, achieved by allocating more bits. 

In practice, gradient methods such as the **Fisher information matrix** (computed from first-order gradients as $F = \mathbb{E}[\nabla \mathcal{L} \nabla \mathcal{L}^T]$) are often used instead. The FIM avoids second-derivative computations but assumes the loss is well-approximated by a probabilistic model (it equals the Hessian exactly when the loss is the negative log-likelihood of a probabilistic model, like cross-entropy loss. For other losses, it's an approximation). In such a framework, a small gradient for a given weight indicates that even a large change in that weight has little effect on the model’s performance. Conversely, a large gradient suggests that even a small change could have a significant impact. Squaring these gradients provides a measure of importance for each weight. However, there are two major drawbacks when applying this approach to llama.cpp:

1. **Limited Training Capabilities:**  
   llama.cpp does not currently support the full training regime required to reliably compute these gradients, which includes both the activation and the loss’s error signal.

2. **Memory Overhead:**  
   The resulting importance matrix is large — at minimum, its size matches that of the model, and when using fp32 gradients, it can be nearly twice as large.

## Llama.cpp fundamentals

To overcome these challenges, llama.cpp employs an alternative that leverages readily available activation statistics rather than gradients. Consider a single row from a model tensor, whose weights are denoted by $w_j$. This row interacts with a column of activations (or embeddings) $a_j$ produced by preceding network layers. The dot product of the weight row with the activation column yields one element of the subsequent activation matrix.

Now, suppose we quantize this tensor row to obtain quantized weights $q_j$. To minimize the quantization error on the resulting activations, we define an error function:

$$
F = \left(\sum_{j} (q_j - w_j) \, a_j\right)^2.
$$

Taking the derivative of $F$ with respect to a particular quantized weight $q_i$ gives:

$$
\frac{\partial F}{\partial q_i} = \sum_{j} a_i \, a_j \, (q_j - w_j).
$$

Averaging this expression over a representative dataset, we obtain:

$$
\sum_{j} \langle a_i a_j \rangle \, (q_j - w_j),
$$

where $\langle \cdot \rangle$ denotes the expectation value over the data.

Because activations can take on both positive and negative values, the cross terms $\langle a_i a_j \rangle$ for $i \neq j$ are likely to cancel out (unless there is a strong correlation). This means the diagonal elements $\langle a_i^2 \rangle$ dominate. Therefore, the approach can be simplified by using:

$$
\mathcal{I}_i = \langle a_i^2 \rangle,
$$

This design enables hardware-aware optimizations while maintaining model accuracy through these core mechanisms:

- **Importance Matrix**:
	As discussed above, this is a mathematical construct that assigns **sensitivity scores** to columns of neural network weights, repeated row by row. Columns with higher scores (indicating greater impact on model outputs) retain higher numerical precision during quantization, while less critical columns undergo more aggressive compression. 
-  **Precision Allocation Strategy**:
A base strategy to adjust is required. The standard quantization methods in `llama.cpp` (like `Q4_0`, `Q5_K`, etc.) generally use a linear mapping, ie: $x = a * q$ or $x = a*q + b$ (see [Even more quantization types?](https://github.com/ggml-org/llama.cpp/discussions/5063)). More details on this approach is provided later in this article. Some _i-quants_ in llama.cpp employ **3rd-order polynomial dequantization**:
	
$$
W_{quant} = aq^3 + bq^2 + cq + d
$$
	
  This non-linear mapping can provide better compression than equivalent linear methods while maintaining accuracy. The use of importance matrices introduces a more sophisticated strategy, biasing the quantization scale for blocks of weights.
	
### Matrix Representation

 A naive conceptualization to the creation of an importance matrix would be to divide the entire model up into columns per weight as if it were one giant matrix, thus producing one importance matrix. For reasons previously mentioned, this is not the case. Instead, each layer in the network is given its own importance matrix.
- **1D Tensor of Weights**:
	 - Each layer in a neural network can be thought of as a vector (1D tensor) of weights. This is essentially a flat list of all the weights in that layer.
- **Block-Wise Grouping**:
	- For quantization, weights are logically partitioned into **fixed-size blocks**. These blocks are not a literal reshaping of the tensor into 2D space but instead represent computational groupings.
- **Columns in the Importance Matrix**:
	- Each column in the importance matrix corresponds to one of these groups of weights. 
	- The importance score for a column is derived from the **variance of the weight's associated activations**.
#### Application

The framework introduces a bias for each weight's parameters (eg, _scale_) based on each value — also in the source code called a "weight" — in the importance matrix. This is implemented with **Hardware-Agnostic Vectorization** implemented through an abstracted SIMD interface, which leverages compile-time intrinsics to generate optimized code paths for multiple instruction sets: x86 (AVX2), ARM (NEON), and RISC-V (V extension).

## Quantization Workflow Implementation

_A comparison of the approaches used in all of the different quantizations available in llama.cpp is beyond the scope of this article. Here, approaches similar to some Q4 approaches are discussed. This is partially applicable to many other bit depths and quantization types._

### Core Algorithmic Steps

1. **Importance matrix column scores** 
2. **Block-Wise Processing**
    - 32-element blocks align to reduce quantization error, and 32 is a good choice because all transformer models in existence have row sizes that are divisible by 32, so one does not need to deal with partial blocks.
    - 256-element superblocks used in k-quants

#### Block-level quantization of the row

Quantization maps a range of floating-point values to a smaller set of integers. This process relies on two key parameters:

1. **Scale** (multiplier): Determines how much to multiply quantized integers to approximate original values.
    
2. **Minimum** (offset): Defines the starting point of the quantization range. _In symmetric quantization (e.g., Q4_0), the minimum is omitted, as the range is centered at zero._
    

The reconstructed value is calculated as:  
`original ≈ q * scale + minimum`

##### Example: Q4_0 Quantization

In llama.cpp’s **Q4_0** format, quantization simplifies to **symmetric scaling** (no minimum term):  
`original ≈ q * scale`.

**Key Properties of Q4_0**:  
- **Per block of 32 weights**:  
  - Each weight is stored as a 4-bit integer (`q`).  
  - A single **6-bit scale** (`d`) is shared across the block.  
  - Total overhead: 6 bits (scale) + 0 bits (minimum) = **6 bits per block**.  
- **Optimization objective**:  
  Minimize the weighted reconstruction error:  

$$  
\sum_{i} w_i (x_i - \text{scale} \cdot q_i)^2  
$$ 

  - $x_i$: Original floating-point weights.  
  - $q_i$: 4-bit integers (range: -8 to 7).  
  - $w_i$: Importance weights (derived from the importance matrix).  

**Role of the Importance Matrix**:  
When provided, the algorithm prioritizes minimizing errors for high-importance weights by:  
  1. **Weighting the error terms**: Errors at positions with larger `quant_weights[i]` contribute more to the loss.  
  2. **Iterative scale refinement**: Tests candidate scales to find the one that minimizes importance-weighted error (see `make_qx_quants` code).  
- Without an importance matrix, the scale is determined by the **maximum absolute weight** in the block (`d = max / -8`), treating all weights equally.  

##### Comparison with Q4_K quants

Briefly, **Q4_K** introduces additional complexity to improve accuracy at the cost of storage, using both the scale and minimum parameters and 256 weight _superblocks_ with their own parameters (the importance matrix biases error minimization at **both levels** in this case).

### Execution Flow

#### Phase 1: Importance Matrix Generation

The workflow initiates with `llama-imatrix` execution, which performs forward passes through the model using calibration data. Key implementation steps include:

8. **Chunk Processing**: Input text is divided into configurable-length segments (default 512 tokens, configurable to match context size) to be processed sequentially. Each chunk undergoes full model inference while tracking activation patterns.
9. **Tensor Significance Accumulation**: The `llama-imatrix` tool aggregates importance metrics across all processed chunks, maintaining running totals for each weight tensor. GPU offloading via `-ngl` parameter accelerates this computation through parallel processing.
10. **Output Serialization**: Final importance values are normalized and stored in binary format (`imatrix.dat` by default) with metadata including processing timestamps and chunk statistics.

#### Phase 2: Quantization Application

The `llama-quantize` tool consumes the generated *imatrix* through several critical code paths:

11. **Matrix Loading**: During quantization initialization, the specified imatrix file is memory-mapped and validated against the target model architecture. The `prepare_imatrix()` function handles format compatibility checks and memory allocation.
12. **Weight Prioritization**: The quantization algorithm uses quantized weights modified by parameters such as scale that are adjusted with importance scores. High-importance weights receive larger bit allocations within mixed-precision quantization blocks. 

## Calibration Process Specifications

### Data Selection Recommendations

The users define calibration corpora. Discussions on llama.cpp's implementation suggest:

- **Domain Alignment**
    - Technical models: 40% code (GitHub), 30% math (arXiv), 30% general text
    - Conversational models: 60% dialogue datasets, 40% Wikipedia
- **Entropy Filtering**
	- Some form of filtering of data may improve quality.  

---

This documentation introduces general approaches to quantization and then llama.cpp's  approach to importance-based quantization, emphasizing major technical implementation details. This approach demonstrates quantization efficiency across several hardware platforms, with calibration data selection remaining the primary user-controlled quality factor.

---

#### 🗣️ Discussion

👤 **ikawrakow** replied the **2025-02-21** at **06:51:45**:<br>

1. Many equations do not show in my Browsers (Firefox, Safari)
2. You are trying to describe the imatrix as used in llama.cpp. Hence, it would be better to use the mathematical foundation of that instead of the LeanQuants paper.
3. You could start by referring to the imatrix PR in `llama.cpp` (https://github.com/ggml-org/llama.cpp/pull/4861)
4. Only `IQ4_XS` and `IQ4_NL` use a non-linear mapping from quantized values to dequantized model weights. All other i-quants in `llama.cpp` use points on a lattice to map a group of 8 (`IQ2_XXS, IQ2_XS, IQ2_S`, E8 lattice) or 4 (`IQ3_XXS, IQ3_S`, D4 lattice) quants to corresponding model values.
5. Blocks of 32 have nothing to do with `AVX2`. They are there to reduce quantization error, and 32 is a good choice because all transformer models in existence have row sizes that are divisible by 32, so one does not need to deal with partial blocks. Blocks of 256 are there to reduce storage requirements spent on block scales. E.g., `Q4_K` uses 6 bits for scale/minimum in blocks of 32, ending up with `256/32*(6+6) = 96` bits for the block scale. Add  `2*16` bits for the super-block `fp16` scale/minimu and you end up with 128 bits or 0.5 bits per weight. In comparison, `Q4_1` which would be the corresponding legacy quantization type uses 5 bits per weight.  
6. Legacy quants do not support imatrix: wrong. See e.g. [this function](https://github.com/ggml-org/llama.cpp/blob/ee02ad02c56ff36a5edd22d8617ab3f9546ce7fe/ggml/src/ggml-quants.c#L1849), which gets called when quantizing a model to `Q4_0`. From there one goes to [this function](https://github.com/ggml-org/llama.cpp/blob/ee02ad02c56ff36a5edd22d8617ab3f9546ce7fe/ggml/src/ggml-quants.c#L1821), which explicitly uses an importance matrix.
7. Phase 2: wrong
8. Dynamic bitwidth allocation: wrong
9. Chunk processing: the division is not "for sequential processing" but to have the ability to generate imatrix data for different **context lengths**. 

Etc. Sorry @robbiemu, but this is just too far from representing the actual imatrix fundamentals and the imatrix use for guiding quantization.

> 👤 **robbiemu** replied the **2025-02-21** at **11:55:47**:<br>
> thank you for that :) Its a draft, of course there are things going to be wrong, its a big project that I've worked _with_ much more than _in_, and I need and appreciate the help identifying where I need to correct. 
> 
> especially things like simple errata like Github's markdown not rendering latex and my confusing at one point blocks of 32 for superblocks of 256 vis-a-vis AVX2 are little burden. But there were a couple of points that I dont feel confident how to process.
> 
> At the beginning, I did transclude in sections from another document I have on LeanQuants specifically because in our conversation where I felt you were the one to equate the imatrix to the hessian approach. And they have a very natural way of expressing the relationship to quantization decisions so .. I took pains to show the approximate relationship. That and, if you search/read about llama.cpp importance matrices online now, you will often see this relationship indicated. In reading your PR comment I see that you don't even explicitly mention it, so maybe inclusion was misguided. Yet, you also don't directly ground quantization decisions to using an importance matrix. In other words, the "how did we get here" that this section currently provides .. I'll need to add that still. Do you prefer another formulation rather than what I used from LeanQuant? If I were to keep it: What is glossed over as essentially a given, that you can calculate only the diagonal, and the fact that you can treat a block-diagonal matrix here as a collection of smaller matrices (so you can break up the model's quantization row-wise, as is done in llama.cpp) -- those can be simplified or removed and replaced with the derivation you spell out in your PR.
> 
> What really interests me is # 7. after generating your imatrix the next step, in practice, is to use the quantization tool. So it must be in the details it is incorrect. I got this from perplexity (I've not been working very much in the llama.cpp source code, except in regards YaRN). If it is not too much to ask, could I ask you to help correct that into a high level description. I'm trying to avoid an exact correspondence here (phase 1 also does not live up to that), I just want a simple conceptual description of the execution graph.
> 
> 👤 **robbiemu** replied the **2025-02-21** at **12:28:24**:<br>
> On one other point:
> 
> "for sequential processing"  -- this is just a lack of clarity, it I guess should be "to then be processed sequentially" maybe. I was never describing the reasoning, just the application, not getting into the details. Maybe I could add something about matching the max_positional_embeddings though, sure. batch and ubatch currently under the lens for change, there's a draft PR to make ubatch functionally different from batch in imatrix generation (ie computing multiple chunks per batch in https://github.com/ggml-org/llama.cpp/pull/9400 ) - as the nature and intent are perhaps changing, describing the intent is something I am not interested in adding to the document.

---

👤 **ikawrakow** replied the **2025-02-21** at **16:20:18**:<br>

If this was a draft that had the occasional mistake here or there, I would try to help you. But the content is so far away from reality that I wouldn't know where to begin (short of completely rewriting it).

As an example, let's look at the section "Phase 2" (point 7 i my initial response that really interests you):

> During quantization initialization, the specified imatrix file is memory-mapped

No, it isn't. It is small and there is no need to complicate things with `mmap`. The data is simply loaded into memory using a standard C++ file stream.

> The quantization algorithm scales compression aggressiveness inversely with importance scores...

Absolutely not. Everything is quantized with the same number of bits, so the "compression aggressiveness" is the same. Instead, when the difference between the original and the quantized model is minimized, the importance matrix enters as a weighting factor in the optimization objective (a.k.a. "loss" these days).

>  the quantization resolution R is determined by: [followed by bogus equation]

Where did you even get this equation from? It certainly is not used anywhere in `llama.cpp` or `ik_llama.cpp` 

> High-importance weights receive larger bit allocations within mixed-precision quantization ...

No. All model weights in a tensor use the exact same amount of bits per weight.

> 👤 **robbiemu** replied the **2025-02-21** at **19:03:42**:<br>
> Ok hold on, please understand I'm just trying to essentially describe this, using tools to help me avoid reading the code was probably a mistake but, in my defense, its a big project that I am trying to elaborate. :) I'll apply the changes, this will get better. Maybe I should seek help from others instead... if so my apologies. I dont want to address the entire reply you gave me there just now, but something you said really gave me doubt.
> 
> >> The quantization algorithm scales compression aggressiveness inversely with importance scores...
> >
> > Absolutely not. Everything is quantized with the same number of bits, so the "compression aggressiveness" is the same. Instead, when the difference between the original and the quantized model is minimized, the importance matrix enters as a weighting factor in the optimization objective (a.k.a. "loss" these days).
> 
> Wow that is a surprise.  So for example, in your earlier reference to the `quantize_row_q4_0_impl()` function, the loop is not assigning a different number of bits to each column of weights within the row? If it is applying the same value throughout, why is it using a for loop for each column of weights from the row?
> 
> edit: ooh, I forgot about this! I had known it at some level before, but it was never necessary in discussing it so I forgot and went back to my original understanding. It is basically a lot more computation to use a different number of bits, but there are other details that go into extracting the original value. the multiplier and the offset.
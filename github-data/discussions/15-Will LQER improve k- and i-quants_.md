### 🗣️ [#15](https://github.com/ikawrakow/ik_llama.cpp/discussions/15) - Will LQER improve k- and i-quants?

| **Author** | `ikawrakow` |
| :--- | :--- |
| **Created** | 2024-08-09 |
| **Updated** | 2025-07-12 |

---

#### Description

[LQER/L²QER](https://arxiv.org/pdf/2402.02446) is the latest hype about LLM quantization. Promptly, there is an [issue](https://github.com/ggerganov/llama.cpp/discussions/8831) in `llama.cpp` to use that to improve the existing quantization methods because, you know, the gras is always greener on the other side of the road. But, unlike many earlier calls to improve quantization with the latest "SOTA" quantization advertisement, err, scientific paper, on arXiv, there are already efforts underway to actually implement this. E.g., [this PR](https://github.com/ggerganov/llama.cpp/pull/8939) adds Numpy dequantization so one can use Numpy to do the SVD of the difference between the full model and a quantized model.

People are of course free to spend their energy any way they see fit, and I should rather mind my own business, but I couldn't help myself but put this prediction on the record:

**LQER/L²QER will not help to improve any of the k- or I-quants in `llama.cpp`.**

Why do I think so?

Having spent so much time on developing all k- and i-quants in `llama.cpp`, I basically remember perplexity (PPL) values for a lot of models, especially the early once such as LLaMA-v1 and LLaMA-v2. And these are exactly the models the LQER authors compare their quantization against in Table 3 of the paper. So, for me, just a quick look was sufficient to see that the results of the paper are nowhere near being SOTA as they are being advertised. But let's do the comparison. I reproduce the Table 3.1 here for convenience:
 
<img width="1561" alt="Screenshot 2024-08-09 at 1 39 34 PM" src="https://github.com/user-attachments/assets/92f1e85f-83a8-4f51-bc6a-0f7ebfeb21d8">

Activation quantization is not quite there yet in `llama.cpp`, so we will focus on the upper part of the table, which shows results when only the model weights are quantized. Let us do some comparisons. I'll use `Q4_K_S`, `IQ4_XS`, and the newly added `IQ4_K` and `IQ3_K`. The L²QER quantization is 4.3 bpw, so it is in the same range as `IQ3_XS` (4.25 bpw) and `Q4_K_S/IQ4_K` (4.5 bpw). `IQ3_K` (3.4 bpw) is there to put things into perspective.

I have archived my LLaMA-v1 models and didn't feel like restoring (or re-downloading) the 33B and 65B models, so we will look at 7B and 13B. The PPL results in the paper are computed with standard Python tooling, and it is known that perplexities computed with `llama.cpp` can be quite different from people get in the Python Universe. But the ratio of the quantized PPL to the PPL of the `f16` model is nearly independent of the way PPL has been computed. The authors of the LQER paper have chosen to use the difference `PPL(Q) - PPL(f16)` (the ∆PPL column in Table 3), which is basically the same thing. Nevertheless, let's put some effort into making `llama.cpp` PPL more comparable to Python tooling. As far as I can tell, there are two main differences how PPL is computed:
* In `llama.cpp` PPL is evaluated by sequentially going over the provided evaluation text, while in Python samples of the given context length are selected at random. This should not result in a different result, at least not beyond the statistical uncertainty of the PPL estimate, so I did not change `llama.cpp`. 
* In `llama.cpp` the mean log probability is evaluated over the second half of the context window `n_ctx`, while in Python the whole context window is used. Both are approximations to PPL for a context `n_ctx`. The `llama.cpp` approximation is better (to first order, it reports PPL for `3/4 n_ctx`, while the Python estimate is for `1/2 n_ctx`. Nevertheless, let's just change it in `llama.cpp` by adjusting [this line](https://github.com/ikawrakow/ik_llama.cpp/blob/a9f302ebe2373321c12b01d8760904901aa064a4/examples/perplexity/perplexity.cpp#L567). But instead of just using `first = 1`, I adjusted a bit around and ended up using `first = std::max(1, n_ctx/128)`, which gave the closest match between `llama.cpp` and the values reported in Table 3 of the LQER paper (which are for a context of 2048. I know this based on other quantization papers, which quote the same `f16` `PPL` values and explicitly state the context window used)

The following table shows the `llama.cpp` `f16` perplexities for the full models computed with this modification:

|  LLaMA-v1-7B | LLaMA-v1-13B | LLaMA-v2-7B | LLaMA-v2-13B |
| -------------- | --------------  | -------------- | --------------- |
|  5.6291 +/- 0.02202 | 5.0172 +/- 0.01893 | 5.4802 +/- 0.02128  |   4.8706 +/- 0.01824 |

OK, we can now do the comparison. The table shows  ∆PPL for the 4 LLaMA models and the 4 different quantization types. For more convenient comparison I have also added the  L²QER result.  

| Quantization |  bpw | LLaMA-v1-7B | LLaMA-v1-13B | LLaMA-v2-7B | LLaMA-v2-13B |
| ------- | ----- | --- | ---- | ---- | ---- |
| L²QER            |  4.30 | 0.220 |  0.100  | 0.100 | 0.060 |
| IQ3_K           |  3.43 | 0.220 |   0.142  | 0.114 | 0.144 | 
| IQ4_XS         |  4.25 | 0.075 |  0.054  | 0.065 | 0.048 |
| Q4_K_S        |  4.50 | 0.065 | 0.041  | 0.063 | 0.044 |
| IQ4_K           |  4.50 | 0.041  | 0.033  | 0.043 | 0.034 |

I think the difference in performance is clear, and no further discussion is required.

I made [this comment](https://github.com/ggerganov/llama.cpp/pull/729#issuecomment-1519038289) back in April of 2023. I had just gotten involved with `llama.cpp` and had started thinking about the quantization of LLMs. With SVD being a standard tool in the toolbox of an ML practitioner, it was one of the first things that came to mind. Did I try? Of course I did - with disappointing results: one needed way too many terms to be competitive with block-wise quantization (I had already started working on k-quants). It is of course possible that my SVD attempts weren't good and, and the LQER authors were able to get something out of SVD. But my guess is it is a matter of the quality of the quantization to begin with: if the quality is low, then perhaps one can improve with just the first few components of the singular value decomposition. But if one still has a 2X - 5X larger quantization error **after** having done that, it is extremely unlikely that one can improve the much better quants by using just a few SVD terms. So, based on this, I reach the above conclusion.

Pinging @compilade who seems to be the main driving force behind implementing LQER in `llama.cpp` just in case this is somehow useful.

---

#### 🗣️ Discussion

👤 **compilade** replied the **2024-08-09** at **15:12:32**:<br>

Thanks for pinging me, it's interesting to learn about your past attempts with SVD.

In the LQER paper they don't seem to use it on top of SOTA quantization methods (they seem to use it on top of MXINT), so I'm simply curious to see if it's viable to apply it on top of k-quants and i-quants.

It might not be worth it, though, as you say.

But there's also something else which they did not try in the paper: subtracting a low-rank decomposition of the weights to then quantize only what remains, while the LoRA adapter of the quantization error should be able to recover it. I did not yet experiment with different ranks for both of theses low-rank approximations.

And in my preliminary tests this *does* help with pure `Q2_K` compared to plain LQER, but wasn't really better than the default `Q2_K` mix (which also uses `Q3_K` in some places), at least on a small model (OpenELM-270M), and with F16 LoRA and a rank of 32.

It's possible that a specialized quantization type for the not-low-rank part of weights could be useful, but I did not yet study how the distribution is changed when subtracting a low-rank approximation. My hypothesis is that non-linear assymetric quant types have an advantage for this, so the new `IQ2_K` and `IQ3_K` *might* be well suited for this.

I did not yet implement L²QER, so I dont know how it would perform yet. You're likely very right that it won't be good, but I want to try, because it will enable other experiments like different error-minimization objectives for the quantized dense tensor and the low-rank adapter.

Also, I have not yet implemented Numpy dequantization for most of the `IQ` types, only `IQ4_NL` and `IQ4_XS`, because the grids for the others are a bit large. Ideally, they should be generated at runtime with a minimal amount of magic numbers. Is that possible?

---

👤 **ikawrakow** replied the **2024-08-09** at **16:01:22**:<br>

> Also, I have not yet implemented Numpy dequantization for most of the IQ types, only IQ4_NL and IQ4_XS, because the grids for the others are a bit large. Ideally, they should be generated at runtime with a minimal amount of magic numbers. Is that possible?

Perhaps you should ask Georgi? According to `git blame` he is the author of most of the `IQ` tables.

But more seriously: the short answer is 'no'. To generate these tables, I quantized a bunch of models using the full E8 or D4 lattice, and collected statistics how often each lattice point is being used. This data is already orders of magnitude larger than the final `IQ` tables (and it takes quite some tome to generate). I then ran an optimization that attempts to a) Maximize the use count of selected lattice points and b) Minimize the maximum (or count averaged) distance between not selected lattice points to the nearest selected lattice point. I haven't published the code that does these things. But even if I had, the run time of the optimization is much too long to be invoked each time (and the lattice point use statistics is much bigger than the tables). I'm also not sure why you think the tables are too large? The data fits in L1 cache, no? Or are we running this on computers with 16 kB of RAM?

> And in my preliminary tests this does help with pure Q2_K compared to plain LQER, but wasn't really better than the default Q2_K mix (which also uses Q3_K in some places), at least on a small model (OpenELM-270M), and with F16 LoRA and a rank of 32.

If you use enough principle components you will eventually get an improvement, of course. But the question is, given the extra bits spent, is the improvement better than what is achievable by using a different quant, using quantization mixes, etc., with the same extra bits spent. Also, as demonstrated by `IQ2_S` (and `IQ2_K` in this repo), `Q2_K` is far from optimal in terms of the compromise between quantization accuracy and quantized model size, so perhaps one could get something there.

> But there's also something else which they did not try in the paper: subtracting a low-rank decomposition of the weights to then quantize only what remains, while the LoRA adapter of the quantization error should be able to recover it. I did not yet experiment with different ranks for both of theses low-rank approximations.

This is the first thing I tried. If that had been successful, we would have gotten not just a model compression, but a massive increase in performance too as matrix multiplications with a low rank decomposition are much faster than using the full matrix. I did have moderate success with the `K` and `Q` tensors in the early layers of LLaMA-1, but anything else was just hopeless until you started approaching full SVD.

But then again, I'm one of those people suffering from the NIH syndrome, so used my own hand-rolled tools for this investigation. Perhaps you will be more lucky just using standard tooling.

---

👤 **ikawrakow** replied the **2024-08-27** at **15:11:01**:<br>

Btw, on [this branch](https://github.com/ikawrakow/ik_llama.cpp/tree/ik/try_svd) there is some exploration of using SVD before or after the quantization. I have misused the `quantize-stats` tool to look at how the root-mean-square-error (rmse) behaves as a function of the number of SVD components. One can do the SVD before or after quantization. Certainly not production quality, AVX2-only vectorization, very simple multi-threading, but still enough to see that SVD does not add any value to LLMs quantization when the quantization works reasonably well. I know it works because full SVD reduces rmse to zero.

> 👤 **compilade** replied the **2024-08-27** at **16:59:19**:<br>
> Thanks!
> 
> I see that when `SVD_BEFORE` is `false`, the initial output fed into `try_svd` is non-zero, and SVD is [done on the subtraction of input and output](https://github.com/ikawrakow/ik_llama.cpp/blob/63fc8014a25e5192b618e0d8f869f8c507c99793/examples/quantize-stats/quantize-stats.cpp#L317), which means this does look similar to LQER (while also quantizing the low-rank tensor?) if I understand it correctly. Still feels like a good proof of concept, even though it doesn't test using SVD both before quantization (to remove low-rank components from the input) *and* after (to then correct both the additional low-rank error and the quantization error) at the same time. It's helpful to know that plain LQER is worse than better quantization.
> 
> I didn't really do any experiment lately towards LQER (and L²QER) because I was busy with other things, but this SVD implementation could likely be eventually useful for control vectors according to <https://github.com/ggerganov/llama.cpp/discussions/8831#discussioncomment-10227359> (cc @ngxson)
> 
> For L²QER, I think `imatrix` files will probably need to use a less bespoke format, which means I think they could be GGUF files with `general.type` equal to `imatrix` (a bit like LoRA adapters have `general.type` equal to `adapter` since <https://github.com/ggerganov/llama.cpp/pull/8332>).

---

👤 **ikawrakow** replied the **2024-09-11** at **14:31:14**:<br>

@compilade With your PR-9400 in `llama.cpp` I now have to write GGUF loading and link against `ggml` when I want to take a quick look at an imatrix? Instead of just copy/pasting the 20 LOC of imatrix structure definition and (de-)serialization into a `.cpp` file and being done in 5 minutes? Ouch. And no, HF tools will with 99.99% probability not help me with what I'm interested in. I mean, having a Python imatrix to GGUF converter is I guess great for those who want to look at imatrix files on HF, but changing the imatrix tool to output GGUFs is a bit too much afaik.

Oh well, I'll need to keep my own copy of the `imatrix` and `quantize` tools.

> 👤 **ngxson** replied the **2024-09-11** at **15:17:56**:<br>
> Hi and sorry if this change disrupts your workflow.
> 
> The main reason behind this change was that we want to unify file formats in llama.cpp. From the perspective of software engineering, is needed because it could help abstract out some parts of the implementation, thus provide a better code base for more features to come in the future.
> 
> Contrary to what you said (to have HF to visualize the GGUF file), in fact, this change does introduce a headache to HF backend, since now we have to distinguish between GGUF model file and GGUF other-files (i.e. imatrix, cvector, lora). This is just to clarify to you that the main motivation of the change is about refactoring code in llama.cpp.
> 
> Beside that, I'm wondering if it could help you: there is `gguf-py` package that allow GGUF file to be loaded into python. You can then use `torch` to investigate the imatrix tensors.
> 
> Another option would be have a CLI arg in imatrix to select the output file format, although this may make the code a bit harder to maintain.
> 
> In anyway, I appreciate your work and would love to know if we can do anything to help you.
> 
> 👤 **ikawrakow** replied the **2024-09-11** at **16:01:09**:<br>
> > In anyway, I appreciate your work and would love to know if we can do anything to help you.
> 
> Not merge PR-9400? Or just merge the imatrix to GGUF Python conversion script?
> 
> I have written many tools that are not for public consumption but I have used (and still use occasionally) to investigate various quantization strategies. They are nice, simple, stand-alone programs where I don't even need a Makefile or a CMakeLists.txt but can just do `g++ -O3 some_too.cpp && ./a.out some_imatrix some_other_input`. They all become useless with this commit.
> 
> > The main reason behind this change was that we want to unify file formats in llama.cpp. 
> > Contrary to what you said (to have HF to visualize the GGUF file), in fact, this change does introduce a headache to HF backend, 
> 
> I see. We make a change that introduces headaches, triples or quadruples the code required to load/save such files thus magnifying the probability for bugs, and mandates linking against `libggml.so` for any tool that wants to operate with such files, to gain the benefit of "unifying file formats in llama.cpp"? Where the thing being unified is not some monstrous code with thousands of lines of code and massive maintenance burden but a 20 LOC thing that defines the format and implements (de-)serialization? Cool.
> 
> 👤 **ikawrakow** replied the **2024-09-11** at **16:19:12**:<br>
> > From the perspective of software engineering, is needed because it could help abstract out some parts of the implementation, thus provide a better code base for more features to come in the future.
> ```
> ls -al ./ggml/src/libggml.so
> -rwxrwxr-x 1 iwan iwan 369408304 Sep  9 20:11 ./ggml/src/libggml.so
> ```
> Don't know about you, but having to link against a 370 MB `.so` to abstract 20 LoC does not add up afaik.
> 
> 👤 **ngxson** replied the **2024-09-11** at **16:57:26**:<br>
> Regarding the merge decision, I can't determine whether it will be merged or not. My role is to provide clarity and explore options to help.
> 
> The abstraction here isn't just about code length, but about creating a unified approach for tensor save/load operations within llama.cpp. In the future, this could also make it easier to add more parameters to imatrix.gguf file. It also allows more users to experiment with imatrix directly in the GGUF format, without needing conversions.
> 
> I completely agree that linking against a 370 MB .so file is not desirable. However, it's worth noting that your `libggml.so` is likely built with CUDA support, which significantly increases its size. Also, the GGUF-related code is actually a small fraction of the whole ggml library.
> 
> To address your specific workflow needs, I have a suggestion that might help: What if I provide you a header-only GGUF loader? This could potentially allow you to work with GGUF files without the need for linking against the full `libggml.so`. I've been considering this idea for a while, but couldn't find a valid usage for it.
> 
> 👤 **compilade** replied the **2024-09-12** at **02:48:39**:<br>
> @ikawrakow Thanks for expressing concern about the format change.
> 
> The main reason for it is that there doesn't seem to be a backward-compatible way to make the non-GGUF-based `imatrix` format work with many ubatches per chunk, or many chunks per ubatches (in the simple format, ncalls is tied to the ubatch size but is also somehow used as the number of chunks). It's also impossible to get the chunk size used to make a non-GGUF `imatrix` file from its metadata. (The convert script assumes 512 was used, but that's not always true. This is mostly relevant when merging `imatrix` files with `--in-file`)
> 
> The non-GGUF `imatrix` files *are* simpler to deserialize, *but* that format has no way to be extended backward-compatibly, except by adding more stuff at the end and never ever removing any field. (And that format also doesn't have any magic number at the start, so not particularly easy to identify)
> 
> I don't really want to break your scripts, though. Would a reverse convert script, from new to old format help (round-trip conversion tests can be used to test for correctness), or do you categorically oppose using GGUF for `imatrix` files? Should `llama-quantize` be able to load both formats instead of only one?

---

👤 **ikawrakow** replied the **2024-09-12** at **13:16:15**:<br>

@compilade Thank you for responding to my concerns.

> The main reason for it is that there doesn't seem to be a backward-compatible way to make the non-GGUF-based imatrix format work with many ubatches per chunk, or many chunks per ubatches (in the simple format, ncalls is tied to the ubatch size but is also somehow used as the number of chunks). 

I must admit I don't understand the concerns. The issue is that one cannot (correctly) combine imatrices computed with different `u_batch` sizes? (One can always combine them, but the files will not contribute to the combined imatrix with the correct weight). Why would one want to do that?  AFAIK, not needing to worry about batch and u-batch sizes is a feature, not a bug.

> It's also impossible to get the chunk size used to make a non-GGUF imatrix file from its metadata. (The convert script assumes 512 was used, but that's not always true. This is mostly relevant when merging imatrix files with --in-file)

Here is what I do
```
./bin/llama-imatrix -m some_model -f some+_training_data -c some_context --chunks N -o some_imatrix_c${some_context}.out
```
I.e., my imatrix files always carry the context length that was used in their name. Worth noting that a) The context length has a surprisingly small influence on the quantization results b) One may want to combine imatrices computed with a different context length to see what happens (what context length are you going to record for the combined imatrix file?)

> The non-GGUF imatrix files are simpler to deserialize, but that format has no way to be extended backward-compatibly, except by adding more stuff at the end and never ever removing any field. (And that format also doesn't have any magic number at the start, so not particularly easy to identify)

The imatrix is one and only one thing. I wouldn't know how one wants to "extend" it without it no longer being an imatrix. But suppose we **really** wanted to extend it. Here is what I would do
```
void read_imatrix(std::istream in, ...) {
    int  n_entries;
    VersionInfo vinfo = {}; // default constructor makes sure we are dealing with a "legacy" imatrix file.
    in.read((char *)&n_entries, sizeof(n_entries);
    if (n_entries == std::numeric_limits<int>::max()) {
        // imatrices with that many entries definitely do not exist
        // => we are dealing with an "extended" imatrix
        // read actual number of entries
        in.read((char *)&n_entries, sizeof(n_entries);
        // read version info
        read_version_info(vinfo);
    }
    ...
}
``` 
Voila, all existing imatrices continue to work, you can add whatever extensions you like (anywhere you like, not just at the end), we don't need to include `ggml/gguf` headers and link against a 370 MB `libggml.so`, etc.

> 👤 **compilade** replied the **2024-09-13** at **01:56:41**:<br>
> > I must admit I don't understand the concerns. The issue is that one cannot (correctly) combine imatrices computed with different `u_batch` sizes? (One can always combine them, but the files will not contribute to the combined imatrix with the correct weight). Why would one want to do that? AFAIK, not needing to worry about batch and u-batch sizes is a feature, not a bug.
> 
> The sanest way to both not worry about batch sizes and correctly combine `imatrix` files is to store the number of tokens (or activations in this case) instead of the number of "chunks". This is what is done in the GGUF-based format. You're right that the chunk size in the metadata isn't really necessary. I *think* it would be possible to make it work that way in the simper format, but there would still be some weirdness with MoE tensors.
> 
> I know using GGUF would make the `imatrix` format more complicated, but interoperability with existing and future GGUF tooling would be useful. For example, I'm working on some kind `gguf-diff` tool which will compare tensors between GGUF files (dequantizing if necessary), and making `imatrix` data stored as GGUF would make that tool work on `imatrix` files too without having to specially handle them.
> 
> > what context length are you going to record for the combined imatrix file?
> 
> The one used at the time of merging them (the last one). It seems like there is no good choice for the context length in that case.
> 
> > Voila, all existing imatrices continue to work, you can add whatever extensions you like (anywhere you like, not just at the end)
> 
> But the extensions would still break your scripts, so I don't see how it makes it better? It seems like all you want from this is that `imatrix` remains a trivially parsable format, even if it's changed?
> 
> > we don't need to include `ggml/gguf` headers and link against a 370 MB `libggml.so`, etc.
> 
> You're still using `llama-imatrix` (which does link against `libggml.so`) to generate those files.
> 
> You know what, I think you're right to want to keep it simple. But GGUF-based `imatrix` also enables a bunch of stuff which is otherwise not possible. I will set <https://github.com/ggerganov/llama.cpp/pull/9400> as a draft, and then I'll try to make a compromise by making `llama-imatrix` *both* able to output the simple (somewhat backward-compatibly, but by storing the number of tokens as `ncall` instead of the number of chunks (the division by `ncall` will still result in a mean (of squares), so your existing scripts *should* continue to work)) *and* GGUF-based format (so that bidirectional conversion will be directly possible with `--in-file`. The GGUF-based `imatrix` format would only be used when the `--output` ends with `.gguf`, which it will by default), while I'll also try to make `llama-quantize` read both (basically falling back when loading as GGUF fails).
> 
> It's gonna take me *at least* another week to implement that (not much free time this month, and lots of good conferences in my area).
> 
> Not sure if supporting both formats will be viable long-term, though. But from this discussion I gather that both have reasons to exist.
> 
> Basically, I think these are the arguments **for** each format:
> 
> - Keeping the "simpler" `imatrix.dat` format
>   - Simpler to parse
>     - Only 20 LOC (closer to 50 LOC with proper error handling)
>   - No need to link to `ggml` to load it
>   - Allows self-contained programs to do experiments with it
> - Using GGUF for `imatrix`
>   - Reduces the need for special-purpose formats
>   - More interoperability with existing and future GGUF tooling
>     - `gguf_dump.py`
>     - HF previews
>     - (eventually) `gguf-diff`
>   - Trivially extensible
>     - More metadata
>     - Easier type changes for metadata (e.g. `int32` vs `int64`)
>   - Counts are multi-dimensional
>     - For stacked MoE tensors, each expert gets its own activation count
>     - Allows keeping the sums intact
>       - Great for merging `imatrix` files
> 
> And the arguments against:
> 
> - Keeping the "simpler" `imatrix.dat` format
>   - Not trivially identifiable (no magic number as file header)
>   - Weird serialization of MoE activation sums (scaled to use the same chunk count for the whole tensor)
>   - Hard to backward-compatibly extend
>     - (although some kind of extension *is* possible, it will pretty much always cause breaking changes)
>   - Need to write a special-purpose `imatrix_reader` in `gguf-py`
> - Using GGUF for `imatrix`:
>   - Depends on more code to load/save such files
>     - which means more probability for bugs
>       - (although since that code is shared with model loading, noticing/fixing bugs there benefit everything which uses it)
>   - Can't make stand-alone programs for quantization experiments like before
>     - Need to link to `libggml.so` to use GGUF-based `imatrix` files
>     - Or need to include some `gguf.h` header-only library
> 
> 👤 **compilade** replied the **2025-07-12** at **14:18:22**:<br>
> @ikawrakow
> 
> I made changes to <https://github.com/ggml-org/llama.cpp/pull/9400> since last time.
> 
> Is it sufficient for `llama-imatrix` to use the GGUF format only when the output filename ends with `.gguf`, so that if you keep using old output names, you'll still get the same format your scripts can work with?
> 
> Similarly, conversion back to the previous format is now implemented, and is used like resuming an `imatrix` file but without a dataset, and where the output format ends with anything else than `.gguf`:
> 
> ```console
> $ ./bin/llama-imatrix --in-file imatrix.gguf -o imatrix.dat
> ```
> 
> `imatrix.gguf` files can always be converted to the `imatrix.dat` format, but the reverse lacks some shape information for 3d tensor evaluation counts (which is necessary to handle partial data gracefully in stacked MoE tensors). Both directions still work, though. `llama-quantize` can read both formats.
> 
> I've had some complaints regarding using the filename extension to select the imatrix format. The alternative would be a format flag, but you would need to know about it (especially if the default isn't the format you're used to).
> 
> It's still not completely clear to me what or how strict your requirements are. Is it closer to "GGUF imatrix files should not exist", "GGUF imatrix should only be used deliberately" (e.g. by using the `.gguf` suffix), or "a format flag for the previous format would be enough, even if the default is GGUF"?
> 
> 👤 **ikawrakow** replied the **2025-07-12** at **17:19:43**:<br>
> @compilade 
> 
> Thank you for letting me know. I basically never use `llama.cpp` now, so the imatrix GG-ification is no longer relevant for my needs. The imatrix tool in mainline has been broken for MLA models for quite some time now, so I guess it is time to fix  that by merging your PR.
> 
> I'm of course looking forward to all the imatrix improvements that have been discussed, but never materialized because their implementation was inhibited by the inferior data format. Now, with the imatrix GG-ified, its future is looking really bright!
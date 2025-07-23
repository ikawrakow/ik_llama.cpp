### 🗣️ [#140](https://github.com/ikawrakow/ik_llama.cpp/discussions/140) - Questions about weight[j]

| **Author** | `DavidZyy` |
| :--- | :--- |
| **Created** | 2024-12-13 |
| **Updated** | 2025-02-11 |

---

#### Description

Hi @ikawrakow, your work on quantization is amazing and I really admire them. Recently, I am reading codes about this and have some questions. 
For example, at funtion `quantize_row_q4_0_impl` and other places,  `weight[j]` is: 
```cpp
weight[j] = qw[j] * sqrtf(sigma2 + xb[j]*xb[j]);
```
I already see some discussions at [here](https://github.com/ggerganov/llama.cpp/discussions/5263#discussioncomment-11511794), but  I still don't quite understand, Can you give me some guidance? Why do not use the following directly?
```cpp
weight[j] = qw[j]
```

---

#### 🗣️ Discussion

👤 **ikawrakow** replied the **2024-12-14** at **08:13:19**:<br>

Hi @DavidZyy,

this is simply an empirical correction, there is no science behind it (and it was amusing to observe people trying to make scientific sense out of it). From the pre-imatrix days we have learned that it is better to assign higher weights (importance) to model weights with larger magnitudes in a weighted RMSE minimization. As there is no precise science behind that, it was just a matter of experimentation to determine how this higher importance should look like ($x^2$, $|x|$, $\sigma^2 + x^2$, $\sigma + |x|$, etc., are all variations that have been tried). When I introduced the imatrix, the hope was of course that one can get rid of such non-scientific stuff and just use the diagonal elements of the Hessian. But in practice it is rarely as simple as that. Having the $\sqrt{\sigma^2 + x^2}$ in there does improve quantization accuracy, at least as measured by perplexity or KL-divergence.

Why $\sqrt{\sigma^2 + x^2}$ and not something else?
* As the Hessian already gives a lot of information about model weight importance, at some level it should be clear that the empirical correction cannot be as strongly magnitude dependent as it was without the imatrix
* We definitely do not want to have the importance of small-magnitude weights become (nearly) zero
* Based on the above two bullet points, and the experience from pre-imatrix quantization, $\sqrt{\sigma^2 + x^2}$ was an obvious choice that turned out to work better than anything else I tried

Why the need for correcting the Hessian in the first place?
* We are using just the diagonal elements, which is an approximation. In my experience adding a correction to an approximation often improves things
* From a more conceptual point of view, even if we did use the full Hessian, we still don't know if RMSE between the quantized and the full model weights is the similarity measure that we should be minimizing. RMSE is of course very convenient (expressions are very simple), so not knowing what to minimize we just use that. But in reality another similarity measure may be better, and it will have a different Hessian, so a different importance matrix, so we are back to square one where the importances being used are just a matter of empirical experimentation.

---

👤 **DavidZyy** replied the **2024-12-14** at **13:58:43**:<br>

Thanks for taking time to answer this question and share information, I learned a lot from your answers.
Yes, it's very interesting :)
> (and it was amusing to observe people trying to make scientific sense out of it)

---

👤 **jukofyork** replied the **2025-02-10** at **17:03:34**:<br>

Oh shit, I just realised I totally forgot to reply to this post! @ikawrakow Thanks for the explanation!

FWIW, I actually tested a couple of different schemes that were more grounded in regularisation theory, but they performed worse than your empirical method. It would still be nice to find some way to interpolate between the two extremes; the recent 256-expert being a good case in point!

I did manage to fix some of this back when `dbrx` first dropped:

https://github.com/ggerganov/llama.cpp/pull/7099

IIRC, all the main discussion is in this PR:

https://github.com/ggerganov/llama.cpp/pull/6387#issuecomment-2094926182

but I still suspect that for these new very-high-expert-MoEs it should really be down-regularised compared to non-MoE or older low-expert-count-MoEs.

---

👤 **ikawrakow** replied the **2025-02-10** at **18:07:55**:<br>

@jukofyork So, I have used regularization in a variety of contexts. Sadly, having spent the better part of my career in Medical Device where everything is closed source, there aren't many examples of that in the open. [This repository](https://github.com/ikawrakow/mnist) uses Tikhonov regularization for the training of an SVM model to recognize hand written digits. I put it out there because I find it funny that with fewer lines of code I can beet the [ggml mnist example](https://github.com/ggml-org/ggml/tree/master/examples/mnist) by a huge margin (0.4% vs 2% error rate, so 5X lower). But having used ragularization techniques in deformable image registration, large scale optimization of radiation therapy treatments, real-time target and/or critical organ tracking on live MRI images, MR and PET image reconstruction, etc., I think I know quite well when regularization is required, and LLM quantization is not one of the cases where it is, at least not in the classical sense of adding penalty term(s) to the optimization objective. For instance, Tikhonov regularization that was being proposed in one of the discussions, is pretty much the last thing we want to do when quantizing because we definitely do not want to make the quantized values as small as possible, which is the goal of the Tikhonov regularization term. At some level, one can consider i-quants as using "regularization" via forcing groups of quants to fall on a finite set of grid points, the set being much smaller than all possible grid points for the given number of bits per quant. E.g., `IQ2_XXS` uses 256 out of 6561 points on the E8 lattice. This prevents overfitting, thus can be considered as "regularization". 

The other thing I have learned is that theories are rarely useful in their pure form. More often than not, you start with this beautiful theory to only find that it does not work very well in practice. So, you start adding fudge factors, and things get better. And then you add even more fudge factors and it gets better. When you are done with it you have something that works really well, but you barely recognize the beautiful pure theory you started from.

Just my 2 cents

> 👤 **jukofyork** replied the **2025-02-10** at **19:26:00**:<br>
> > For instance, Tikhonov regularization that was being proposed in one of the discussions, is pretty much the last thing we want to do when quantizing because we definitely do not want to make the quantized values as small as possible, which is the goal of the Tikhonov regularization term.
> 
> I was late to that discussion, but it was possibly me who mentioned this.
> 
> If it was, then I wouldn't have been proposing to use Tikhonov regularization on the weighting factors themselves to drive them to zero, as I agree this makes no sense. I would have suggested regularising the log of the weighting factors towards zero, which in turn regularises the weighting factors to 1 (ie: all equally weighted), whilst retaining the multiplicative symmetry around 1 and enforcing the non-negativity.
> 
> From a Bayesian perspective:
> 
> - Tikhonov regularization of the weights assumes some Gaussian prior centred around zero with lambda controlling the scale (which is obviously not what we want here).
> - Tikhonov regularization of the log of the weights assumes some [log-normal](https://en.wikipedia.org/wiki/Log-normal_distribution) prior centred around 1 with lambda controlling the (log) scale.
> 
> I'm pretty sure I tried this way back when I mentioned this in that thread and it did turn out to be slightly worse than your empirically derived method on whatever model I tried it on.
> 
> ---
> 
> I still think this is an important area to consider (whatever the chosen regularization method is):
> 
> #### (A) I see people still using using bartowski's same ~250kb `calibration_datav3.txt` file on `Deepseek-V3` as on fully-dense models. 
> 
> IMO, this has two huge problems:
> 
> 1. The effective sample size is *at best* 1/32 = ~3% compared to a dense model.
> 2. If the router penalty hasn't done a good job during training, the effective sample size is potentially (much) lower than 3%.
> 
> This can be corrected by either increasing the sample size, or where not possible (say due to the model being too large); adjusting the regularisation factor appropriately.
> 
> #### (B) I see people using `wiki.train.raw` for the `imatrix` and then testing on `wiki.test.raw` (not so much now thankfully).
> 
> Thinking they are getting an unbiased estimate of the `imatrix`'s perplexity improvement:
> 
> ##### wiki.train.raw
> ```
>   = Valkyria Chronicles III = 
>  
>  Senjō no Valkyria 3 : Unrecorded Chronicles ( Japanese : 戦場のヴァルキュリア3 , lit . Valkyria of the Battlefield 3 ) , commonly referred to as Valkyria Chronicles III outside Japan , is a tactical role @-@ playing video game developed by Sega and Media.Vision for the PlayStation Portable . Released in January 2011 in Japan , it is the third game in the Valkyria series . Employing the same fusion of tactical and real @-@ time gameplay as its predecessors , the story runs parallel to the first game and follows the " Nameless " , a penal military unit serving the nation of Gallia during the Second Europan War who perform secret black operations and are pitted against the Imperial unit " Calamaty Raven " . 
>  The game began development in 2010 , carrying over a large portion of the work done on Valkyria Chronicles II . While it retained the standard features of the series , it also underwent multiple adjustments , such as making the game more forgiving for series newcomers . Character designer Raita Honjou and composer Hitoshi Sakimoto both returned from previous entries , along with Valkyria Chronicles II director Takeshi Ozawa . A large team of writers handled the script . The game 's opening theme was sung by May 'n . 
>  It met with positive sales in Japan , and was praised by both Japanese and western critics . After release , it received downloadable content , along with an expanded edition in November of that year . It was also adapted into manga and an original video animation series . Due to low sales of Valkyria Chronicles II , Valkyria Chronicles III was not localized , but a fan translation compatible with the game 's expanded edition was released in 2014 . Media.Vision would return to the franchise with the development of Valkyria : Azure Revolution for the PlayStation 4 . 
>  
>  = = Gameplay = = 
> ```
> 
> ##### wiki.test.raw
> 
> ```
>   = Robert Boulter = 
>  
>  Robert Boulter is an English film , television and theatre actor . He had a guest @-@ starring role on the television series The Bill in 2000 . This was followed by a starring role in the play Herons written by Simon Stephens , which was performed in 2001 at the Royal Court Theatre . He had a guest role in the television series Judge John Deed in 2002 . In 2004 Boulter landed a role as " Craig " in the episode " Teddy 's Story " of the television series The Long Firm ; he starred alongside actors Mark Strong and Derek Jacobi . He was cast in the 2005 theatre productions of the Philip Ridley play Mercury Fur , which was performed at the Drum Theatre in Plymouth and the Menier Chocolate Factory in London . He was directed by John Tiffany and starred alongside Ben Whishaw , Shane Zaza , Harry Kent , Fraser Ayres , Sophie Stanton and Dominic Hall . 
>  In 2006 , Boulter starred alongside Whishaw in the play Citizenship written by Mark Ravenhill . He appeared on a 2006 episode of the television series , Doctors , followed by a role in the 2007 theatre production of How to Curse directed by Josie Rourke . How to Curse was performed at Bush Theatre in the London Borough of Hammersmith and Fulham . Boulter starred in two films in 2008 , Daylight Robbery by filmmaker Paris Leonti , and Donkey Punch directed by Olly Blackburn . In May 2008 , Boulter made a guest appearance on a two @-@ part episode arc of the television series Waking the Dead , followed by an appearance on the television series Survivors in November 2008 . He had a recurring role in ten episodes of the television series Casualty in 2010 , as " Kieron Fletcher " . Boulter starred in the 2011 film Mercenaries directed by Paris Leonti . 
>  
>  = = Career = = 
> ```
> 
> It should be really clear why this is a bad idea.
> 
> #### (C) I see people running the `imatrix` calculation on only the first `512` tokens of models with huge contexts.
> 
> This is clearly a *very* bad idea for several reasons related to the transformer architecture, likely biases the weighting factors to short sequences and also under-represents (part of) the tensors in the transformer blocks vs the MLP blocks.
> 
> ---
> 
> I am certainly no "Bayesian purist" and will happily tune the prior to get the best observed results too!
> 
> BUT: I strongly believe the effectiveness of the `imatrix` calculations could be vastly improved by adding some method of interpolation/regularisation/whatever to allow for informed tuning of the weighting factors! :smile:
> 
> 👤 **saood06** replied the **2025-02-10** at **20:23:18**:<br>
> > I still think this is an important area to consider (whatever the chosen regularization method is):
> > #### (A) I see people still using using bartowski's same ~250kb `calibration_datav3.txt` file on `Deepseek-V3` as on fully-dense models.
> > 
> > IMO, this has two huge problems:
> > 
> >     1. The effective sample size is _at best_ 1/32 = ~3% compared to a dense model.
> > 
> >     2. If the router penalty hasn't done a good job during training, the effective sample size is potentially (much) lower than 3%.
> > 
> > 
> > This can be corrected by either increasing the sample size, or where not possible (say due to the model being too large); adjusting the regularisation factor appropriately.
> 
> There is some discussion among a huggingface quant maker about imatrixing arctic-instruct ( another large MoE), where they talked about how since the experts are stored together in one tensor if for a layer only 1 expert is missing the entire layer can't be quantized, also while investigating this trying to get that expert to activate they observation something that shows size alone doesn't matter as the diversity of data did.
> 
> "the only ones that has 127 out of 128 experts other than yours was "calibration_datav3" from bartowski and " imatrix-with-rp-format-data". Many datasets got way less experts than that. It clearly is the quality of training data and not the amount that matters. 4chan pol_062016-112019_labeled is massive but when I aborted it, it only had 122 out of 128 experts on layer 0. MMLU which I though is really diverse only managed to trigger 121 out of 121 experts on layer 0. "Tech-Awesome-Hub/mix-data" was with just 120 out of 128 experts on layer 0 even worse than that."
> 
> From: https://huggingface.co/mradermacher/BabyHercules-4x150M-GGUF/discussions/3#6758d52499eea0c4b65d0475
> 
> They do discuss the idea of needing more data because of MoE in that thread. I use their imatrix.dat files, and my ppl numbers I gave you are for IQ4_K_R4.
> 
> 👤 **ikawrakow** replied the **2025-02-11** at **06:01:32**:<br>
> Is the inability to activate al experts observed just for layer 0 or for all layers?
> 
> Are people aware of the fact that one can run the model with more active experts than specified by the meta data?
> ```
> ./bin/llama-imatrix -m some_model -f some_training --override-kv deepseek2.expert_used_count=int:N
> ```
> I think doing that will likely help activate more experts.
> 
> I also don't understand why the entire experts tensor cannot be imatrix-quantized if just one expert is missing. If that's what we ended up with, it definitely needs fixing.
> 
> 👤 **saood06** replied the **2025-02-11** at **15:17:30**:<br>
> > Is the inability to activate al experts observed just for layer 0 or for all layers?
> 
> Just layer 0.
> 
> > Are people aware of the fact that one can run the model with more active experts than specified by the meta data?
> > 
> > ```
> > ./bin/llama-imatrix -m some_model -f some_training --override-kv deepseek2.expert_used_count=int:N
> > ```
> > 
> > I think doing that will likely help activate more experts.
> 
> Yes, people are aware of that (not sure if these people are) since I've seen plenty of testing every time a popular MoE comes out of people testing with that override to various values, but are you sure that is recommended? LLM performance tends to drop if you activate more or less than experts than the trained upon amount.
> 
> 
> > I also don't understand why the entire experts tensor cannot be imatrix-quantized if just one expert is missing. If that's what we ended up with, it definitely needs fixing.
> 
> That is what happens. When doing imatrix they hit this (happened with other layers and tensors but this is the only one that persisted through the entire imatrix run.
> 
> ```save_imatrix: entry '              blk.0.ffn_gate_exps.weight' has partial data (99.22%) - skipping```
> 
> This lead to them not releasing IQ1 quants as it runs into this:
> 
> ```llama_model_quantize: failed to quantize: Missing importance matrix for tensor blk.0.ffn_gate_exps.weight in a very low-bit quantization```
> 
> 
> They never reported that for any of the Deepseek models so I'm assuming they only encountered it with arctic and no matter what they did they were never able to activate that expert so I'm giving some credence to their theory that "There indeed could be an issue in the model router that makes it impossible to ever get routed to this specific expert which would be really unfortunate."
> 
> Looking at the files in safetensors each expert is stored separately but with a GGUF that is not the case and they are all stored together.
> 
> 👤 **ikawrakow** replied the **2025-02-11** at **16:33:38**:<br>
> Thanks for making me aware of this situation. I prepared PR #202 to deal with it.
> 
> 👤 **ikawrakow** replied the **2025-02-11** at **17:11:08**:<br>
> > but are you sure that is recommended? 
> 
> I don't know if it is recommended. What I do know is that one can improve low bpw quantization by using a slightly higher number of active experts. E.g., for DeepSeek-Lite, 8 instead of 6 active experts is distinctly better for `IQ1_S` and `IQ1_M`. IIRC, 3 instead of 2 active experts did improve `IQ1_S` and `IQ1_M` quantized Mixtral8x7. As you increase the bpw the advantage goes away and eventually becomes counter productive. Using 3 instead of 2 experts for Mixtral8x7 was futile at 4+ bpw. But these new models have way more experts and more active experts, so activating additional experts is more forgiving.  A quick check with DeepSeek-Lite (6 active experts as per meta data):
> * For 7 experts PPL is slightly lower (-0.2%)
> * For 8 and 9 experts it is about the same
> * For 10 experts PPL is ~0.3% higher.
> 
> 👤 **saood06** replied the **2025-02-11** at **17:27:49**:<br>
> With R1 I've come across a person saying "I tried with 10 and 12 experts and generating perplexity failed with NaNs." and this same person tested 2,3,4,6,8,16 of unsloth's IQ1_M. His results below.
> 
> Experts | PPL
> -- | --
> 8 | 3.4155, 4.2311, 3.0817, 2.8601, 2.6933, 2.5792, 2.5123, 2.5239
> 16 | 3.5350, 4.3594, 3.0307, 2.8619, 2.7227, 2.6664, 2.6288, 2.6568
> 6 | 3.4227, 4.2400, 3.1610, 2.9933, 2.8307, 2.7110, 2.6253, 2.6488
> 4 | 3.5790, 4.5984, 3.5135, 3.4490, 3.2952, 3.2563, 3.1883, 3.2978
> 3 | 3.9209, 4.9318, 4.0944, 4.2450, 4.2071, 4.3095, 4.3150, 4.6082
> 2 | 6.2387, 7.7455
> 
> Here's another user who reported only lower expert usage.
> 
> 
> Model | [1] | [2] | [3] | [4] | [5] | [6] | [7] | [8]
> -- | -- | -- | -- | -- | -- | -- | -- | --
> IQ2_XXS | 3.39 | 4.56 | 3.44 | 3.27 | 3.27 | 3.20 | 3.12 | 3.12
> IQ3_XXS (exp=3) | 3.12 | 4.03 | 2.93 | 2.63 | 2.52 | 2.48 | 2.45 | 2.48
> IQ3_XXS (exp=4) | 2.87 | 3.61 | 2.60 | 2.25 | 2.09 | 1.97 | 1.89 | 1.87
> IQ3_XXS (exp=6) | 2.67 | 3.53 | 2.53 | 2.13 | 1.94 | 1.80 | 1.71 | 1.65
> IQ3_XXS (def) | 2.69 | 3.53 | 2.51 | 2.11 | 1.91 | 1.78 | 1.69 | 1.62
> 
> 👤 **jukofyork** replied the **2025-02-11** at **19:22:47**:<br>
> > > but are you sure that is recommended?
> > 
> > I don't know if it is recommended. What I do know is that one can improve low bpw quantization by using a slightly higher number of active experts. E.g., for DeepSeek-Lite, 8 instead of 6 active experts is distinctly better for `IQ1_S` and `IQ1_M`. IIRC, 3 instead of 2 active experts did improve `IQ1_S` and `IQ1_M` quantized Mixtral8x7. As you increase the bpw the advantage goes away and eventually becomes counter productive. Using 3 instead of 2 experts for Mixtral8x7 was futile at 4+ bpw. But these new models have way more experts and more active experts, so activating additional experts is more forgiving. A quick check with DeepSeek-Lite (6 active experts as per meta data):
> > 
> >     * For 7 experts PPL is slightly lower (-0.2%)
> > 
> >     * For 8 and 9 experts it is about the same
> > 
> >     * For 10 experts PPL is ~0.3% higher.
> 
> Yeah, I managed to do this with `dbrx` before the PR that fixes the divisors for the experts separately. IIRC, I actually activated all the experts for `dbrx` and it got a better resulting `imatrix` than the pre-PR code did, and was quite usable.
> 
> 👤 **jukofyork** replied the **2025-02-11** at **19:24:47**:<br>
> > With R1 I've come across a person saying "I tried with 10 and 12 experts and generating perplexity failed with NaNs." and this same person tested 2,3,4,6,8,16 of unsloth's IQ1_M. His results below.
> 
> This could be because most previous MoEs use softmax to gate/weight with, so as you add more experts is scales down the weights, but `deepseek-v3` uses sigmoids, so the sum getting added into the hidden state will get larger and larger (you can probably also hack the weights and bias to counter this though).
> 
> EDIT:
> 
> ```
> INFO:hf-to-gguf:blk.11.exp_probs_b.bias,      torch.float32 --> F32, shape = {256}
> INFO:hf-to-gguf:blk.11.ffn_gate_inp.weight,   torch.bfloat16 --> F32, shape = {7168, 256}
> ```
> 
> 👤 **saood06** replied the **2025-02-11** at **20:24:39**:<br>
> > `deepseek-v3` uses sigmoids, so the sum getting added into the hidden state will get larger and larger
> 
> Then why does 16 experts work, but not 10/12?
> 
> 👤 **jukofyork** replied the **2025-02-11** at **20:33:32**:<br>
> > > `deepseek-v3` uses sigmoids, so the sum getting added into the hidden state will get larger and larger
> > 
> > Then why does 16 experts work, but not 10/12?
> 
> Not sure - seems very strange!
> 
> Only thing i can think of is some have negatively correlated outputs, and the sum of 16 cancels out the error that overflows whereas with 10 or 12 it doesn't?
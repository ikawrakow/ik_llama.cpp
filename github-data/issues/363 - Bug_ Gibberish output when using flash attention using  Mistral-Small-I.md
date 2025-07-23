### 🐛 [#363](https://github.com/ikawrakow/ik_llama.cpp/issues/363) - Bug: Gibberish output when using flash attention using  Mistral-Small-Instruct-2409-Q6_K and Gemma-3-12b-it-q4_0 on CPU

| **Author** | `djg26` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-05-01 |
| **Updated** | 2025-05-09 |

---

#### Description

### What happened?

Whenever I use the flash attention flag the models always output incoherent text. This happens with both mistral small 2409 and gemma 3 12b. If I don't use flash attention they work fine. Normal llama.cpp works fine when I use flash attention on these models.
Command used for gemma: ./llama-server -m ~/Downloads/gemma-3-12b-it-q4_0.gguf -ctk q8_0 -ctv q8_0  -fa -c 32768
My system is a Core-i7 10700 with 32GB of RAM.
![Image](https://github.com/user-attachments/assets/d3470ebb-a2e5-4b68-b8d7-6b67e343addb)

### Name and Version

./llama-server --version
version: 3657 (98d16264)
built with cc (GCC) 14.2.1 20250207 for x86_64-pc-linux-gnu


### What operating system are you seeing the problem on?

Linux

### Relevant log output

```shell

```

---

#### 💬 Conversation

👤 **ikawrakow** commented the **2025-05-01** at **06:59:00**:<br>

Thank you for the bug report. Can you confirm that #364 fixes it?

---

👤 **djg26** commented the **2025-05-01** at **12:07:51**:<br>

It's a little better, however it still breaks down after a few tokens, becoming gibberish again with a longer context. Tested with  Mistral-Small-Instruct-2409-Q6_K with the following command ./llama-server -m ~/Downloads/Mistral-Small-Instruct-2409-Q6_K.gguf -ctk q8_0 -ctv q8_0 -fa -c 32768
With flash attention:

![Image](https://github.com/user-attachments/assets/590cfa10-21ac-47fa-9f5d-7ee682622d61)

Without flash attention:

![Image](https://github.com/user-attachments/assets/38a5fdef-bf0b-4d38-925f-48498324e0cc)

---

👤 **ikawrakow** commented the **2025-05-01** at **13:38:15**:<br>

What about Gemma?

---

👤 **djg26** commented the **2025-05-01** at **13:53:04**:<br>

Same issue with Gemma as well.

With flash attention:

![Image](https://github.com/user-attachments/assets/559480f1-e29c-496c-bb07-494ec217b930)

Without flash attention:

![Image](https://github.com/user-attachments/assets/3125466a-f470-4278-bc65-696b43ce9569)

---

👤 **ikawrakow** commented the **2025-05-01** at **15:45:01**:<br>

So, `f16` and `Q6_0` KV cache works. Here is the 5 paragraph story I get with `Q6_0`:
```
> Write a five paragraph story
The rain hammered against the windows of the antique bookstore, a relentless rhythm mirroring the turmoil in Eleanor’s heart. She’d come to "The Dusty Tome" seeking solace, a refuge from the wreckage of her recent engagement. The scent of aged paper and leather was usually comforting, a familiar hug for her soul, but today it only amplified the hollow ache in her chest. She drifted between towering shelves, her fingers tracing the spines of forgotten stories, hoping to find a narrative that could somehow explain her own shattered plot. A first edition of "Wuthering Heights" caught her eye, a book about passionate, destructive love – a cruel irony, she thought with a bitter smile.

Lost in thought, she hadn't noticed the elderly gentleman meticulously dusting a shelf nearby. He was a small man, with a cloud of white hair and eyes that twinkled with a quiet wisdom. "Lost, are you, dear?" he asked, his voice soft and laced with a gentle curiosity. Eleanor startled, embarrassed to be caught in her melancholic reverie. "Just… browsing," she mumbled, avoiding his gaze. He chuckled, a warm, comforting sound. "Browsing can be a powerful form of healing. Sometimes, a good story is all we need to realign ourselves." He paused, then added, "Everyone has a story, you know. And every story, even the sad ones, holds a thread of hope."

He led her to a small, secluded alcove filled with lesser-known poets and philosophers. "Have you ever read Rilke?" he asked, pulling a slim volume from the shelf. Eleanor shook her head. "He writes of acceptance, of embracing the totality of experience, even the pain. He says that out of suffering, beauty can emerge." He opened the book, pointing to a passage. "’Let everything happen to you: beauty and terror. Just keep going. No feeling is final.’" The words resonated within her, a small, flickering ember of understanding igniting in the darkness of her grief.

Eleanor spent the next hour lost in Rilke’s poetry, the rain outside fading into a gentle hum. She didn’t feel magically healed, but the sharp edges of her sadness had softened, replaced by a quiet sense of possibility. She realized she wasn’t just mourning the loss of a relationship, but the loss of a future she’d envisioned. And perhaps, she thought, a new, unexpected future awaited, a story yet unwritten. 

As she prepared to leave, she turned to thank the gentleman. He was gone, vanished as silently as he'd appeared. The alcove felt empty, the book of Rilke resting on the shelf, waiting for another seeker. Eleanor smiled, a genuine smile this time, and stepped back into the rain, no longer feeling quite so lost. The rain still fell, but it felt different now, washing away the debris and leaving behind a space for something new to grow.
```

In the case of `Q8_0` KV cache, it does start OK, but transitions to repetitions in the 4th paragraph. I'll need to investigate.

---

👤 **ikawrakow** commented the **2025-05-01** at **16:15:55**:<br>

OK, this should work now with `Q8_0` KV cache. Here the 5 paragraph story I get with
```
./bin/llama-cli -m gemma3-12B-q4_0.gguf  -t 32 -fa  -rtr -ctk q8_0 -ctv q8_0 -c 16384 -cnv
```

Write a five paragraph story

The old lighthouse keeper, Silas, squinted at the churning grey sea. He'd seen a thousand storms lash against the craggy coast, each one a symphony of wind and wave, but this one felt different. A primal unease settled in his gut, a feeling he’d learned to trust over his fifty years tending the beacon. The rhythmic sweep of the lamp, a comforting pulse in the darkness, couldn’t quite dispel the sense of foreboding. He checked the barometer, its needle plummeting with alarming speed. Tonight, he knew, would be a night to remember.

As darkness deepened, the storm hit with brutal force. Rain lashed against the thick glass of the lighthouse windows, blurring the world into a chaotic swirl of grey. The wind howled like a banshee, rattling the metal framework and threatening to tear the very building from its foundations. Silas, a solitary figure against the tempest, methodically checked each lamp and mechanism, his movements practiced and sure. He'd seen ships founder in calmer seas, and he wouldn't let this storm claim another. The rhythmic flash of the light, a desperate plea in the roaring darkness, was all that stood between the rocky coast and unsuspecting vessels.

Suddenly, amidst the cacophony, Silas heard it - a faint, desperate cry carried on the wind. He strained his ears, battling the storm's fury. It came again, clearer this time, a child's voice, choked with fear. He rushed to the window, peering through the rain-streaked glass. Through a momentary break in the storm, he saw it – a small sailboat, tossed about like a toy, its mast broken, a tiny figure clinging to the wreckage.

Without hesitation, Silas activated the emergency beacon and grabbed his oilskins. He knew venturing out in such a storm was madness, but leaving a child to the mercy of the sea was unthinkable. He fought his way down the winding staircase, the lighthouse groaning around him. The waves crashed against the rocks, threatening to sweep him away, but he pressed on, guided by the child’s cries and the unwavering beam of the light he'd so diligently maintained.

Hours later, soaked and shivering, Silas stood on the beach, watching as the coast guard hauled a small, terrified girl from the wreckage. Relief washed over him, stronger than any wave. The storm had subsided, leaving behind a trail of debris and a sky slowly clearing to reveal a sliver of moon. He turned back towards the steadfast lighthouse, its beam still cutting through the lingering darkness, a silent guardian, a beacon of hope in a world often shrouded in storms.

---

👤 **djg26** commented the **2025-05-01** at **17:04:17**:<br>

It does work better now, but I'm still having repetition issues when the context has a few 1,000 tokens in it, both with Gemma3 and Mistral small. This is with the KV cache both on Q8_0.

---

👤 **ikawrakow** commented the **2025-05-01** at **17:14:24**:<br>

You need to give me a way to reproduce it.

---

👤 **djg26** commented the **2025-05-01** at **17:35:28**:<br>

With q8_0 on  kv cache
./llama-server -m ~/Downloads/gemma-3-12b-it-q4_0.gguf -ctk q8_0 -ctv q8_0 -fa -c 32768
![Image](https://github.com/user-attachments/assets/a5858715-79fb-4745-9b79-58615bd33339)

Without q8_0 on kv cache (but flash attention still enabled)
./llama-server -m ~/Downloads/gemma-3-12b-it-q4_0.gguf -fa -c 32768
![Image](https://github.com/user-attachments/assets/12152bae-b4fd-47c3-8a17-28e762045612)

I'm not very used to llama-cli so I've been doing it via llama-server's webui.

---

👤 **ikawrakow** commented the **2025-05-01** at **18:18:58**:<br>

Can you post `cat /proc/cpuinfo`? Thanks.

Here is what I get:

<img width="657" alt="Image" src="https://github.com/user-attachments/assets/86f03895-0973-41f9-9445-796aa881bdae" />


Can you post `cat /proc/cpuinfo`?

---

👤 **ikawrakow** commented the **2025-05-01** at **18:18:58**:<br>

Here is what I get:

<img width="657" alt="Image" src="https://github.com/user-attachments/assets/86f03895-0973-41f9-9445-796aa881bdae" />


Can you post `cat /proc/cpuinfo`?

---

👤 **djg26** commented the **2025-05-01** at **18:31:10**:<br>

```
processor	: 0
vendor_id	: GenuineIntel
cpu family	: 6
model		: 165
model name	: Intel(R) Core(TM) i7-10700 CPU @ 2.90GHz
stepping	: 5
microcode	: 0xfc
cpu MHz		: 3591.565
cache size	: 16384 KB
physical id	: 0
siblings	: 16
core id		: 0
cpu cores	: 8
apicid		: 0
initial apicid	: 0
fpu		: yes
fpu_exception	: yes
cpuid level	: 22
wp		: yes
flags		: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush dts acpi mmx fxsr sse sse2 ss ht tm pbe syscall nx pdpe1gb rdtscp lm constant_tsc art arch_perfmon pebs bts rep_good nopl xtopology nonstop_tsc cpuid aperfmperf pni pclmulqdq dtes64 monitor ds_cpl vmx smx est tm2 ssse3 sdbg fma cx16 xtpr pdcm pcid sse4_1 sse4_2 x2apic movbe popcnt tsc_deadline_timer aes xsave avx f16c rdrand lahf_lm abm 3dnowprefetch cpuid_fault epb ssbd ibrs ibpb stibp ibrs_enhanced tpr_shadow flexpriority ept vpid ept_ad fsgsbase tsc_adjust bmi1 avx2 smep bmi2 erms invpcid mpx rdseed adx smap clflushopt intel_pt xsaveopt xsavec xgetbv1 xsaves dtherm ida arat pln pts hwp hwp_notify hwp_act_window hwp_epp vnmi pku ospke md_clear flush_l1d arch_capabilities
vmx flags	: vnmi preemption_timer posted_intr invvpid ept_x_only ept_ad ept_1gb flexpriority apicv tsc_offset vtpr mtf vapic ept vpid unrestricted_guest vapic_reg vid ple shadow_vmcs pml ept_violation_ve ept_mode_based_exec
bugs		: spectre_v1 spectre_v2 spec_store_bypass swapgs itlb_multihit srbds mmio_stale_data retbleed eibrs_pbrsb gds bhi
bogomips	: 5799.77
clflush size	: 64
cache_alignment	: 64
address sizes	: 39 bits physical, 48 bits virtual
power management:

processor	: 1
vendor_id	: GenuineIntel
cpu family	: 6
model		: 165
model name	: Intel(R) Core(TM) i7-10700 CPU @ 2.90GHz
stepping	: 5
microcode	: 0xfc
cpu MHz		: 1853.498
cache size	: 16384 KB
physical id	: 0
siblings	: 16
core id		: 1
cpu cores	: 8
apicid		: 2
initial apicid	: 2
fpu		: yes
fpu_exception	: yes
cpuid level	: 22
wp		: yes
flags		: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush dts acpi mmx fxsr sse sse2 ss ht tm pbe syscall nx pdpe1gb rdtscp lm constant_tsc art arch_perfmon pebs bts rep_good nopl xtopology nonstop_tsc cpuid aperfmperf pni pclmulqdq dtes64 monitor ds_cpl vmx smx est tm2 ssse3 sdbg fma cx16 xtpr pdcm pcid sse4_1 sse4_2 x2apic movbe popcnt tsc_deadline_timer aes xsave avx f16c rdrand lahf_lm abm 3dnowprefetch cpuid_fault epb ssbd ibrs ibpb stibp ibrs_enhanced tpr_shadow flexpriority ept vpid ept_ad fsgsbase tsc_adjust bmi1 avx2 smep bmi2 erms invpcid mpx rdseed adx smap clflushopt intel_pt xsaveopt xsavec xgetbv1 xsaves dtherm ida arat pln pts hwp hwp_notify hwp_act_window hwp_epp vnmi pku ospke md_clear flush_l1d arch_capabilities
vmx flags	: vnmi preemption_timer posted_intr invvpid ept_x_only ept_ad ept_1gb flexpriority apicv tsc_offset vtpr mtf vapic ept vpid unrestricted_guest vapic_reg vid ple shadow_vmcs pml ept_violation_ve ept_mode_based_exec
bugs		: spectre_v1 spectre_v2 spec_store_bypass swapgs itlb_multihit srbds mmio_stale_data retbleed eibrs_pbrsb gds bhi
bogomips	: 5799.77
clflush size	: 64
cache_alignment	: 64
address sizes	: 39 bits physical, 48 bits virtual
power management:

processor	: 2
vendor_id	: GenuineIntel
cpu family	: 6
model		: 165
model name	: Intel(R) Core(TM) i7-10700 CPU @ 2.90GHz
stepping	: 5
microcode	: 0xfc
cpu MHz		: 800.000
cache size	: 16384 KB
physical id	: 0
siblings	: 16
core id		: 2
cpu cores	: 8
apicid		: 4
initial apicid	: 4
fpu		: yes
fpu_exception	: yes
cpuid level	: 22
wp		: yes
flags		: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush dts acpi mmx fxsr sse sse2 ss ht tm pbe syscall nx pdpe1gb rdtscp lm constant_tsc art arch_perfmon pebs bts rep_good nopl xtopology nonstop_tsc cpuid aperfmperf pni pclmulqdq dtes64 monitor ds_cpl vmx smx est tm2 ssse3 sdbg fma cx16 xtpr pdcm pcid sse4_1 sse4_2 x2apic movbe popcnt tsc_deadline_timer aes xsave avx f16c rdrand lahf_lm abm 3dnowprefetch cpuid_fault epb ssbd ibrs ibpb stibp ibrs_enhanced tpr_shadow flexpriority ept vpid ept_ad fsgsbase tsc_adjust bmi1 avx2 smep bmi2 erms invpcid mpx rdseed adx smap clflushopt intel_pt xsaveopt xsavec xgetbv1 xsaves dtherm ida arat pln pts hwp hwp_notify hwp_act_window hwp_epp vnmi pku ospke md_clear flush_l1d arch_capabilities
vmx flags	: vnmi preemption_timer posted_intr invvpid ept_x_only ept_ad ept_1gb flexpriority apicv tsc_offset vtpr mtf vapic ept vpid unrestricted_guest vapic_reg vid ple shadow_vmcs pml ept_violation_ve ept_mode_based_exec
bugs		: spectre_v1 spectre_v2 spec_store_bypass swapgs itlb_multihit srbds mmio_stale_data retbleed eibrs_pbrsb gds bhi
bogomips	: 5799.77
clflush size	: 64
cache_alignment	: 64
address sizes	: 39 bits physical, 48 bits virtual
power management:

processor	: 3
vendor_id	: GenuineIntel
cpu family	: 6
model		: 165
model name	: Intel(R) Core(TM) i7-10700 CPU @ 2.90GHz
stepping	: 5
microcode	: 0xfc
cpu MHz		: 800.000
cache size	: 16384 KB
physical id	: 0
siblings	: 16
core id		: 3
cpu cores	: 8
apicid		: 6
initial apicid	: 6
fpu		: yes
fpu_exception	: yes
cpuid level	: 22
wp		: yes
flags		: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush dts acpi mmx fxsr sse sse2 ss ht tm pbe syscall nx pdpe1gb rdtscp lm constant_tsc art arch_perfmon pebs bts rep_good nopl xtopology nonstop_tsc cpuid aperfmperf pni pclmulqdq dtes64 monitor ds_cpl vmx smx est tm2 ssse3 sdbg fma cx16 xtpr pdcm pcid sse4_1 sse4_2 x2apic movbe popcnt tsc_deadline_timer aes xsave avx f16c rdrand lahf_lm abm 3dnowprefetch cpuid_fault epb ssbd ibrs ibpb stibp ibrs_enhanced tpr_shadow flexpriority ept vpid ept_ad fsgsbase tsc_adjust bmi1 avx2 smep bmi2 erms invpcid mpx rdseed adx smap clflushopt intel_pt xsaveopt xsavec xgetbv1 xsaves dtherm ida arat pln pts hwp hwp_notify hwp_act_window hwp_epp vnmi pku ospke md_clear flush_l1d arch_capabilities
vmx flags	: vnmi preemption_timer posted_intr invvpid ept_x_only ept_ad ept_1gb flexpriority apicv tsc_offset vtpr mtf vapic ept vpid unrestricted_guest vapic_reg vid ple shadow_vmcs pml ept_violation_ve ept_mode_based_exec
bugs		: spectre_v1 spectre_v2 spec_store_bypass swapgs itlb_multihit srbds mmio_stale_data retbleed eibrs_pbrsb gds bhi
bogomips	: 5799.77
clflush size	: 64
cache_alignment	: 64
address sizes	: 39 bits physical, 48 bits virtual
power management:

processor	: 4
vendor_id	: GenuineIntel
cpu family	: 6
model		: 165
model name	: Intel(R) Core(TM) i7-10700 CPU @ 2.90GHz
stepping	: 5
microcode	: 0xfc
cpu MHz		: 4614.727
cache size	: 16384 KB
physical id	: 0
siblings	: 16
core id		: 4
cpu cores	: 8
apicid		: 8
initial apicid	: 8
fpu		: yes
fpu_exception	: yes
cpuid level	: 22
wp		: yes
flags		: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush dts acpi mmx fxsr sse sse2 ss ht tm pbe syscall nx pdpe1gb rdtscp lm constant_tsc art arch_perfmon pebs bts rep_good nopl xtopology nonstop_tsc cpuid aperfmperf pni pclmulqdq dtes64 monitor ds_cpl vmx smx est tm2 ssse3 sdbg fma cx16 xtpr pdcm pcid sse4_1 sse4_2 x2apic movbe popcnt tsc_deadline_timer aes xsave avx f16c rdrand lahf_lm abm 3dnowprefetch cpuid_fault epb ssbd ibrs ibpb stibp ibrs_enhanced tpr_shadow flexpriority ept vpid ept_ad fsgsbase tsc_adjust bmi1 avx2 smep bmi2 erms invpcid mpx rdseed adx smap clflushopt intel_pt xsaveopt xsavec xgetbv1 xsaves dtherm ida arat pln pts hwp hwp_notify hwp_act_window hwp_epp vnmi pku ospke md_clear flush_l1d arch_capabilities
vmx flags	: vnmi preemption_timer posted_intr invvpid ept_x_only ept_ad ept_1gb flexpriority apicv tsc_offset vtpr mtf vapic ept vpid unrestricted_guest vapic_reg vid ple shadow_vmcs pml ept_violation_ve ept_mode_based_exec
bugs		: spectre_v1 spectre_v2 spec_store_bypass swapgs itlb_multihit srbds mmio_stale_data retbleed eibrs_pbrsb gds bhi
bogomips	: 5799.77
clflush size	: 64
cache_alignment	: 64
address sizes	: 39 bits physical, 48 bits virtual
power management:

processor	: 5
vendor_id	: GenuineIntel
cpu family	: 6
model		: 165
model name	: Intel(R) Core(TM) i7-10700 CPU @ 2.90GHz
stepping	: 5
microcode	: 0xfc
cpu MHz		: 800.000
cache size	: 16384 KB
physical id	: 0
siblings	: 16
core id		: 5
cpu cores	: 8
apicid		: 10
initial apicid	: 10
fpu		: yes
fpu_exception	: yes
cpuid level	: 22
wp		: yes
flags		: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush dts acpi mmx fxsr sse sse2 ss ht tm pbe syscall nx pdpe1gb rdtscp lm constant_tsc art arch_perfmon pebs bts rep_good nopl xtopology nonstop_tsc cpuid aperfmperf pni pclmulqdq dtes64 monitor ds_cpl vmx smx est tm2 ssse3 sdbg fma cx16 xtpr pdcm pcid sse4_1 sse4_2 x2apic movbe popcnt tsc_deadline_timer aes xsave avx f16c rdrand lahf_lm abm 3dnowprefetch cpuid_fault epb ssbd ibrs ibpb stibp ibrs_enhanced tpr_shadow flexpriority ept vpid ept_ad fsgsbase tsc_adjust bmi1 avx2 smep bmi2 erms invpcid mpx rdseed adx smap clflushopt intel_pt xsaveopt xsavec xgetbv1 xsaves dtherm ida arat pln pts hwp hwp_notify hwp_act_window hwp_epp vnmi pku ospke md_clear flush_l1d arch_capabilities
vmx flags	: vnmi preemption_timer posted_intr invvpid ept_x_only ept_ad ept_1gb flexpriority apicv tsc_offset vtpr mtf vapic ept vpid unrestricted_guest vapic_reg vid ple shadow_vmcs pml ept_violation_ve ept_mode_based_exec
bugs		: spectre_v1 spectre_v2 spec_store_bypass swapgs itlb_multihit srbds mmio_stale_data retbleed eibrs_pbrsb gds bhi
bogomips	: 5799.77
clflush size	: 64
cache_alignment	: 64
address sizes	: 39 bits physical, 48 bits virtual
power management:

processor	: 6
vendor_id	: GenuineIntel
cpu family	: 6
model		: 165
model name	: Intel(R) Core(TM) i7-10700 CPU @ 2.90GHz
stepping	: 5
microcode	: 0xfc
cpu MHz		: 800.000
cache size	: 16384 KB
physical id	: 0
siblings	: 16
core id		: 6
cpu cores	: 8
apicid		: 12
initial apicid	: 12
fpu		: yes
fpu_exception	: yes
cpuid level	: 22
wp		: yes
flags		: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush dts acpi mmx fxsr sse sse2 ss ht tm pbe syscall nx pdpe1gb rdtscp lm constant_tsc art arch_perfmon pebs bts rep_good nopl xtopology nonstop_tsc cpuid aperfmperf pni pclmulqdq dtes64 monitor ds_cpl vmx smx est tm2 ssse3 sdbg fma cx16 xtpr pdcm pcid sse4_1 sse4_2 x2apic movbe popcnt tsc_deadline_timer aes xsave avx f16c rdrand lahf_lm abm 3dnowprefetch cpuid_fault epb ssbd ibrs ibpb stibp ibrs_enhanced tpr_shadow flexpriority ept vpid ept_ad fsgsbase tsc_adjust bmi1 avx2 smep bmi2 erms invpcid mpx rdseed adx smap clflushopt intel_pt xsaveopt xsavec xgetbv1 xsaves dtherm ida arat pln pts hwp hwp_notify hwp_act_window hwp_epp vnmi pku ospke md_clear flush_l1d arch_capabilities
vmx flags	: vnmi preemption_timer posted_intr invvpid ept_x_only ept_ad ept_1gb flexpriority apicv tsc_offset vtpr mtf vapic ept vpid unrestricted_guest vapic_reg vid ple shadow_vmcs pml ept_violation_ve ept_mode_based_exec
bugs		: spectre_v1 spectre_v2 spec_store_bypass swapgs itlb_multihit srbds mmio_stale_data retbleed eibrs_pbrsb gds bhi
bogomips	: 5799.77
clflush size	: 64
cache_alignment	: 64
address sizes	: 39 bits physical, 48 bits virtual
power management:

processor	: 7
vendor_id	: GenuineIntel
cpu family	: 6
model		: 165
model name	: Intel(R) Core(TM) i7-10700 CPU @ 2.90GHz
stepping	: 5
microcode	: 0xfc
cpu MHz		: 800.000
cache size	: 16384 KB
physical id	: 0
siblings	: 16
core id		: 7
cpu cores	: 8
apicid		: 14
initial apicid	: 14
fpu		: yes
fpu_exception	: yes
cpuid level	: 22
wp		: yes
flags		: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush dts acpi mmx fxsr sse sse2 ss ht tm pbe syscall nx pdpe1gb rdtscp lm constant_tsc art arch_perfmon pebs bts rep_good nopl xtopology nonstop_tsc cpuid aperfmperf pni pclmulqdq dtes64 monitor ds_cpl vmx smx est tm2 ssse3 sdbg fma cx16 xtpr pdcm pcid sse4_1 sse4_2 x2apic movbe popcnt tsc_deadline_timer aes xsave avx f16c rdrand lahf_lm abm 3dnowprefetch cpuid_fault epb ssbd ibrs ibpb stibp ibrs_enhanced tpr_shadow flexpriority ept vpid ept_ad fsgsbase tsc_adjust bmi1 avx2 smep bmi2 erms invpcid mpx rdseed adx smap clflushopt intel_pt xsaveopt xsavec xgetbv1 xsaves dtherm ida arat pln pts hwp hwp_notify hwp_act_window hwp_epp vnmi pku ospke md_clear flush_l1d arch_capabilities
vmx flags	: vnmi preemption_timer posted_intr invvpid ept_x_only ept_ad ept_1gb flexpriority apicv tsc_offset vtpr mtf vapic ept vpid unrestricted_guest vapic_reg vid ple shadow_vmcs pml ept_violation_ve ept_mode_based_exec
bugs		: spectre_v1 spectre_v2 spec_store_bypass swapgs itlb_multihit srbds mmio_stale_data retbleed eibrs_pbrsb gds bhi
bogomips	: 5799.77
clflush size	: 64
cache_alignment	: 64
address sizes	: 39 bits physical, 48 bits virtual
power management:

processor	: 8
vendor_id	: GenuineIntel
cpu family	: 6
model		: 165
model name	: Intel(R) Core(TM) i7-10700 CPU @ 2.90GHz
stepping	: 5
microcode	: 0xfc
cpu MHz		: 4604.265
cache size	: 16384 KB
physical id	: 0
siblings	: 16
core id		: 0
cpu cores	: 8
apicid		: 1
initial apicid	: 1
fpu		: yes
fpu_exception	: yes
cpuid level	: 22
wp		: yes
flags		: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush dts acpi mmx fxsr sse sse2 ss ht tm pbe syscall nx pdpe1gb rdtscp lm constant_tsc art arch_perfmon pebs bts rep_good nopl xtopology nonstop_tsc cpuid aperfmperf pni pclmulqdq dtes64 monitor ds_cpl vmx smx est tm2 ssse3 sdbg fma cx16 xtpr pdcm pcid sse4_1 sse4_2 x2apic movbe popcnt tsc_deadline_timer aes xsave avx f16c rdrand lahf_lm abm 3dnowprefetch cpuid_fault epb ssbd ibrs ibpb stibp ibrs_enhanced tpr_shadow flexpriority ept vpid ept_ad fsgsbase tsc_adjust bmi1 avx2 smep bmi2 erms invpcid mpx rdseed adx smap clflushopt intel_pt xsaveopt xsavec xgetbv1 xsaves dtherm ida arat pln pts hwp hwp_notify hwp_act_window hwp_epp vnmi pku ospke md_clear flush_l1d arch_capabilities
vmx flags	: vnmi preemption_timer posted_intr invvpid ept_x_only ept_ad ept_1gb flexpriority apicv tsc_offset vtpr mtf vapic ept vpid unrestricted_guest vapic_reg vid ple shadow_vmcs pml ept_violation_ve ept_mode_based_exec
bugs		: spectre_v1 spectre_v2 spec_store_bypass swapgs itlb_multihit srbds mmio_stale_data retbleed eibrs_pbrsb gds bhi
bogomips	: 5799.77
clflush size	: 64
cache_alignment	: 64
address sizes	: 39 bits physical, 48 bits virtual
power management:

processor	: 9
vendor_id	: GenuineIntel
cpu family	: 6
model		: 165
model name	: Intel(R) Core(TM) i7-10700 CPU @ 2.90GHz
stepping	: 5
microcode	: 0xfc
cpu MHz		: 800.000
cache size	: 16384 KB
physical id	: 0
siblings	: 16
core id		: 1
cpu cores	: 8
apicid		: 3
initial apicid	: 3
fpu		: yes
fpu_exception	: yes
cpuid level	: 22
wp		: yes
flags		: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush dts acpi mmx fxsr sse sse2 ss ht tm pbe syscall nx pdpe1gb rdtscp lm constant_tsc art arch_perfmon pebs bts rep_good nopl xtopology nonstop_tsc cpuid aperfmperf pni pclmulqdq dtes64 monitor ds_cpl vmx smx est tm2 ssse3 sdbg fma cx16 xtpr pdcm pcid sse4_1 sse4_2 x2apic movbe popcnt tsc_deadline_timer aes xsave avx f16c rdrand lahf_lm abm 3dnowprefetch cpuid_fault epb ssbd ibrs ibpb stibp ibrs_enhanced tpr_shadow flexpriority ept vpid ept_ad fsgsbase tsc_adjust bmi1 avx2 smep bmi2 erms invpcid mpx rdseed adx smap clflushopt intel_pt xsaveopt xsavec xgetbv1 xsaves dtherm ida arat pln pts hwp hwp_notify hwp_act_window hwp_epp vnmi pku ospke md_clear flush_l1d arch_capabilities
vmx flags	: vnmi preemption_timer posted_intr invvpid ept_x_only ept_ad ept_1gb flexpriority apicv tsc_offset vtpr mtf vapic ept vpid unrestricted_guest vapic_reg vid ple shadow_vmcs pml ept_violation_ve ept_mode_based_exec
bugs		: spectre_v1 spectre_v2 spec_store_bypass swapgs itlb_multihit srbds mmio_stale_data retbleed eibrs_pbrsb gds bhi
bogomips	: 5799.77
clflush size	: 64
cache_alignment	: 64
address sizes	: 39 bits physical, 48 bits virtual
power management:

processor	: 10
vendor_id	: GenuineIntel
cpu family	: 6
model		: 165
model name	: Intel(R) Core(TM) i7-10700 CPU @ 2.90GHz
stepping	: 5
microcode	: 0xfc
cpu MHz		: 4603.834
cache size	: 16384 KB
physical id	: 0
siblings	: 16
core id		: 2
cpu cores	: 8
apicid		: 5
initial apicid	: 5
fpu		: yes
fpu_exception	: yes
cpuid level	: 22
wp		: yes
flags		: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush dts acpi mmx fxsr sse sse2 ss ht tm pbe syscall nx pdpe1gb rdtscp lm constant_tsc art arch_perfmon pebs bts rep_good nopl xtopology nonstop_tsc cpuid aperfmperf pni pclmulqdq dtes64 monitor ds_cpl vmx smx est tm2 ssse3 sdbg fma cx16 xtpr pdcm pcid sse4_1 sse4_2 x2apic movbe popcnt tsc_deadline_timer aes xsave avx f16c rdrand lahf_lm abm 3dnowprefetch cpuid_fault epb ssbd ibrs ibpb stibp ibrs_enhanced tpr_shadow flexpriority ept vpid ept_ad fsgsbase tsc_adjust bmi1 avx2 smep bmi2 erms invpcid mpx rdseed adx smap clflushopt intel_pt xsaveopt xsavec xgetbv1 xsaves dtherm ida arat pln pts hwp hwp_notify hwp_act_window hwp_epp vnmi pku ospke md_clear flush_l1d arch_capabilities
vmx flags	: vnmi preemption_timer posted_intr invvpid ept_x_only ept_ad ept_1gb flexpriority apicv tsc_offset vtpr mtf vapic ept vpid unrestricted_guest vapic_reg vid ple shadow_vmcs pml ept_violation_ve ept_mode_based_exec
bugs		: spectre_v1 spectre_v2 spec_store_bypass swapgs itlb_multihit srbds mmio_stale_data retbleed eibrs_pbrsb gds bhi
bogomips	: 5799.77
clflush size	: 64
cache_alignment	: 64
address sizes	: 39 bits physical, 48 bits virtual
power management:

processor	: 11
vendor_id	: GenuineIntel
cpu family	: 6
model		: 165
model name	: Intel(R) Core(TM) i7-10700 CPU @ 2.90GHz
stepping	: 5
microcode	: 0xfc
cpu MHz		: 4600.024
cache size	: 16384 KB
physical id	: 0
siblings	: 16
core id		: 3
cpu cores	: 8
apicid		: 7
initial apicid	: 7
fpu		: yes
fpu_exception	: yes
cpuid level	: 22
wp		: yes
flags		: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush dts acpi mmx fxsr sse sse2 ss ht tm pbe syscall nx pdpe1gb rdtscp lm constant_tsc art arch_perfmon pebs bts rep_good nopl xtopology nonstop_tsc cpuid aperfmperf pni pclmulqdq dtes64 monitor ds_cpl vmx smx est tm2 ssse3 sdbg fma cx16 xtpr pdcm pcid sse4_1 sse4_2 x2apic movbe popcnt tsc_deadline_timer aes xsave avx f16c rdrand lahf_lm abm 3dnowprefetch cpuid_fault epb ssbd ibrs ibpb stibp ibrs_enhanced tpr_shadow flexpriority ept vpid ept_ad fsgsbase tsc_adjust bmi1 avx2 smep bmi2 erms invpcid mpx rdseed adx smap clflushopt intel_pt xsaveopt xsavec xgetbv1 xsaves dtherm ida arat pln pts hwp hwp_notify hwp_act_window hwp_epp vnmi pku ospke md_clear flush_l1d arch_capabilities
vmx flags	: vnmi preemption_timer posted_intr invvpid ept_x_only ept_ad ept_1gb flexpriority apicv tsc_offset vtpr mtf vapic ept vpid unrestricted_guest vapic_reg vid ple shadow_vmcs pml ept_violation_ve ept_mode_based_exec
bugs		: spectre_v1 spectre_v2 spec_store_bypass swapgs itlb_multihit srbds mmio_stale_data retbleed eibrs_pbrsb gds bhi
bogomips	: 5799.77
clflush size	: 64
cache_alignment	: 64
address sizes	: 39 bits physical, 48 bits virtual
power management:

processor	: 12
vendor_id	: GenuineIntel
cpu family	: 6
model		: 165
model name	: Intel(R) Core(TM) i7-10700 CPU @ 2.90GHz
stepping	: 5
microcode	: 0xfc
cpu MHz		: 800.000
cache size	: 16384 KB
physical id	: 0
siblings	: 16
core id		: 4
cpu cores	: 8
apicid		: 9
initial apicid	: 9
fpu		: yes
fpu_exception	: yes
cpuid level	: 22
wp		: yes
flags		: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush dts acpi mmx fxsr sse sse2 ss ht tm pbe syscall nx pdpe1gb rdtscp lm constant_tsc art arch_perfmon pebs bts rep_good nopl xtopology nonstop_tsc cpuid aperfmperf pni pclmulqdq dtes64 monitor ds_cpl vmx smx est tm2 ssse3 sdbg fma cx16 xtpr pdcm pcid sse4_1 sse4_2 x2apic movbe popcnt tsc_deadline_timer aes xsave avx f16c rdrand lahf_lm abm 3dnowprefetch cpuid_fault epb ssbd ibrs ibpb stibp ibrs_enhanced tpr_shadow flexpriority ept vpid ept_ad fsgsbase tsc_adjust bmi1 avx2 smep bmi2 erms invpcid mpx rdseed adx smap clflushopt intel_pt xsaveopt xsavec xgetbv1 xsaves dtherm ida arat pln pts hwp hwp_notify hwp_act_window hwp_epp vnmi pku ospke md_clear flush_l1d arch_capabilities
vmx flags	: vnmi preemption_timer posted_intr invvpid ept_x_only ept_ad ept_1gb flexpriority apicv tsc_offset vtpr mtf vapic ept vpid unrestricted_guest vapic_reg vid ple shadow_vmcs pml ept_violation_ve ept_mode_based_exec
bugs		: spectre_v1 spectre_v2 spec_store_bypass swapgs itlb_multihit srbds mmio_stale_data retbleed eibrs_pbrsb gds bhi
bogomips	: 5799.77
clflush size	: 64
cache_alignment	: 64
address sizes	: 39 bits physical, 48 bits virtual
power management:

processor	: 13
vendor_id	: GenuineIntel
cpu family	: 6
model		: 165
model name	: Intel(R) Core(TM) i7-10700 CPU @ 2.90GHz
stepping	: 5
microcode	: 0xfc
cpu MHz		: 4598.921
cache size	: 16384 KB
physical id	: 0
siblings	: 16
core id		: 5
cpu cores	: 8
apicid		: 11
initial apicid	: 11
fpu		: yes
fpu_exception	: yes
cpuid level	: 22
wp		: yes
flags		: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush dts acpi mmx fxsr sse sse2 ss ht tm pbe syscall nx pdpe1gb rdtscp lm constant_tsc art arch_perfmon pebs bts rep_good nopl xtopology nonstop_tsc cpuid aperfmperf pni pclmulqdq dtes64 monitor ds_cpl vmx smx est tm2 ssse3 sdbg fma cx16 xtpr pdcm pcid sse4_1 sse4_2 x2apic movbe popcnt tsc_deadline_timer aes xsave avx f16c rdrand lahf_lm abm 3dnowprefetch cpuid_fault epb ssbd ibrs ibpb stibp ibrs_enhanced tpr_shadow flexpriority ept vpid ept_ad fsgsbase tsc_adjust bmi1 avx2 smep bmi2 erms invpcid mpx rdseed adx smap clflushopt intel_pt xsaveopt xsavec xgetbv1 xsaves dtherm ida arat pln pts hwp hwp_notify hwp_act_window hwp_epp vnmi pku ospke md_clear flush_l1d arch_capabilities
vmx flags	: vnmi preemption_timer posted_intr invvpid ept_x_only ept_ad ept_1gb flexpriority apicv tsc_offset vtpr mtf vapic ept vpid unrestricted_guest vapic_reg vid ple shadow_vmcs pml ept_violation_ve ept_mode_based_exec
bugs		: spectre_v1 spectre_v2 spec_store_bypass swapgs itlb_multihit srbds mmio_stale_data retbleed eibrs_pbrsb gds bhi
bogomips	: 5799.77
clflush size	: 64
cache_alignment	: 64
address sizes	: 39 bits physical, 48 bits virtual
power management:

processor	: 14
vendor_id	: GenuineIntel
cpu family	: 6
model		: 165
model name	: Intel(R) Core(TM) i7-10700 CPU @ 2.90GHz
stepping	: 5
microcode	: 0xfc
cpu MHz		: 4599.937
cache size	: 16384 KB
physical id	: 0
siblings	: 16
core id		: 6
cpu cores	: 8
apicid		: 13
initial apicid	: 13
fpu		: yes
fpu_exception	: yes
cpuid level	: 22
wp		: yes
flags		: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush dts acpi mmx fxsr sse sse2 ss ht tm pbe syscall nx pdpe1gb rdtscp lm constant_tsc art arch_perfmon pebs bts rep_good nopl xtopology nonstop_tsc cpuid aperfmperf pni pclmulqdq dtes64 monitor ds_cpl vmx smx est tm2 ssse3 sdbg fma cx16 xtpr pdcm pcid sse4_1 sse4_2 x2apic movbe popcnt tsc_deadline_timer aes xsave avx f16c rdrand lahf_lm abm 3dnowprefetch cpuid_fault epb ssbd ibrs ibpb stibp ibrs_enhanced tpr_shadow flexpriority ept vpid ept_ad fsgsbase tsc_adjust bmi1 avx2 smep bmi2 erms invpcid mpx rdseed adx smap clflushopt intel_pt xsaveopt xsavec xgetbv1 xsaves dtherm ida arat pln pts hwp hwp_notify hwp_act_window hwp_epp vnmi pku ospke md_clear flush_l1d arch_capabilities
vmx flags	: vnmi preemption_timer posted_intr invvpid ept_x_only ept_ad ept_1gb flexpriority apicv tsc_offset vtpr mtf vapic ept vpid unrestricted_guest vapic_reg vid ple shadow_vmcs pml ept_violation_ve ept_mode_based_exec
bugs		: spectre_v1 spectre_v2 spec_store_bypass swapgs itlb_multihit srbds mmio_stale_data retbleed eibrs_pbrsb gds bhi
bogomips	: 5799.77
clflush size	: 64
cache_alignment	: 64
address sizes	: 39 bits physical, 48 bits virtual
power management:

processor	: 15
vendor_id	: GenuineIntel
cpu family	: 6
model		: 165
model name	: Intel(R) Core(TM) i7-10700 CPU @ 2.90GHz
stepping	: 5
microcode	: 0xfc
cpu MHz		: 800.000
cache size	: 16384 KB
physical id	: 0
siblings	: 16
core id		: 7
cpu cores	: 8
apicid		: 15
initial apicid	: 15
fpu		: yes
fpu_exception	: yes
cpuid level	: 22
wp		: yes
flags		: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush dts acpi mmx fxsr sse sse2 ss ht tm pbe syscall nx pdpe1gb rdtscp lm constant_tsc art arch_perfmon pebs bts rep_good nopl xtopology nonstop_tsc cpuid aperfmperf pni pclmulqdq dtes64 monitor ds_cpl vmx smx est tm2 ssse3 sdbg fma cx16 xtpr pdcm pcid sse4_1 sse4_2 x2apic movbe popcnt tsc_deadline_timer aes xsave avx f16c rdrand lahf_lm abm 3dnowprefetch cpuid_fault epb ssbd ibrs ibpb stibp ibrs_enhanced tpr_shadow flexpriority ept vpid ept_ad fsgsbase tsc_adjust bmi1 avx2 smep bmi2 erms invpcid mpx rdseed adx smap clflushopt intel_pt xsaveopt xsavec xgetbv1 xsaves dtherm ida arat pln pts hwp hwp_notify hwp_act_window hwp_epp vnmi pku ospke md_clear flush_l1d arch_capabilities
vmx flags	: vnmi preemption_timer posted_intr invvpid ept_x_only ept_ad ept_1gb flexpriority apicv tsc_offset vtpr mtf vapic ept vpid unrestricted_guest vapic_reg vid ple shadow_vmcs pml ept_violation_ve ept_mode_based_exec
bugs		: spectre_v1 spectre_v2 spec_store_bypass swapgs itlb_multihit srbds mmio_stale_data retbleed eibrs_pbrsb gds bhi
bogomips	: 5799.77
clflush size	: 64
cache_alignment	: 64
address sizes	: 39 bits physical, 48 bits virtual
power management:

```

---

👤 **djg26** commented the **2025-05-01** at **18:49:24**:<br>

I get the same issue using mistral small as well.
Command used: ./llama-server -m ~/Downloads/Mistral-Small-Instruct-2409-Q6_K.gguf -ctk q8_0 -ctv q8_0 -fa -c 32768
KV cache q8_0
![Image](https://github.com/user-attachments/assets/35e8b1f5-3476-4e92-9f9e-a6b1e0f4adfc)
./llama-server -m ~/Downloads/Mistral-Small-Instruct-2409-Q6_K.gguf  -fa -c 32768
no Q8_0 KV cache

![Image](https://github.com/user-attachments/assets/dbc37878-80d8-471b-97fc-bece9459cc58)

---

👤 **djg26** commented the **2025-05-01** at **19:32:56**:<br>

Running with K at f16 and V at q8_0 seems to work fine.
 ./llama-server -m ~/Downloads/Mistral-Small-Instruct-2409-Q6_K.gguf -ctv q8_0 -fa -c 32768
![Image](https://github.com/user-attachments/assets/665cd3d2-2c25-4bad-8c82-2c4304b21c51)

---

👤 **ikawrakow** commented the **2025-05-02** at **05:10:26**:<br>

I'll have to investigate in more detail then. In the meantime, just don't use `Q8_0` for K-cache.

---

👤 **ikawrakow** commented the **2025-05-04** at **06:04:20**:<br>

Not sure what to do with this one. It works fine on both of my systems (Zen4 and vanilla AVX2) after #364

---

👤 **djg26** commented the **2025-05-04** at **14:39:17**:<br>

It seems like having K be ```q8_0``` and flash attention turned on doesn't work, at least for me. I've tested it on another computer with a ryzen 5 5600x and it still starts getting repetitive like before after some tokens. If I don't enable ```-fa``` and just have the K be ```q8_0``` then the models work fine. It also works fine if K is something like ```q6_0``` with the ```-fa``` flag.

---

👤 **djg26** commented the **2025-05-04** at **17:18:01**:<br>

Qwen3-30B-A3B also works fine when using ```q8_0``` KV and flash attention.

---

👤 **djg26** commented the **2025-05-09** at **17:16:43**:<br>

Closing as after doing another git pull to the latest version it seems to work fine now.
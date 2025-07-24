### 🗣️ [#621](https://github.com/ikawrakow/ik_llama.cpp/discussions/621) - Deepseek v3/r1 poisoned prompt?

| **Author** | `magikRUKKOLA` |
| :--- | :--- |
| **Created** | 2025-07-17 |
| **Updated** | 2025-07-17 |

---

#### Description

[**UPDATE**]: the current bug IS UNRELATED to the Deepseek inference implementation!  Its ~~a bug of the llm itself~~ just a stupidity of mine.
[**UPDATE2**:] solved! https://github.com/ikawrakow/ik_llama.cpp/discussions/621#discussioncomment-13784692  the issue is related to the usage of the special tokens in the prompt.

I tried different quants and settings.  The answer of Deepseek V3 or R1 is surprisingly stupid.  Initially I thought its a bug in a RAM then I realized I have ECC ram.  So ... its about 20k context prompt and Deepseek answers more or less the following absolute nonsense:

```
  I need to solve the equation ( x + 5 = -2 ) for x. This is a simple linear
  equation.
  First, I want to isolate x. To do that, I need to get rid of the 5 that's
  being added to x. The opposite of addition is subtraction, so I should
  subtract 5 from both sides of the equation.

  So, starting with:

  ( x + 5 = -2 )

  Subtract 5 from both sides:

  ( x + 5 - 5 = -2 - 5 )

  Simplify:

  On the left side, ( +5 - 5 ) cancels out, leaving just x.

  On the right side, ( -2 - 5 ) is ( -7 ).

  So, ( x = -7 ).

  I should verify this. If I plug ( x = -7 ) back into the original equation:

  ( -7 + 5 = -2 )

  ( -7 + 5 ) is indeed ( -2 ), so it checks out.

  Therefore, the solution is ( x = -7 ).  To solve the equation (x + 5 = -2)
  for
  (x), follow these steps:

  1. Isolate (x) by eliminating the constant term (+5) on the left side.
  Subtract 5 from both sides of the equation:
  [
  x + 5 - 5 = -2 - 5
  ]
  2. Simplify both sides:
    • Left side: (x + 0 = x)
    • Right side: (-2 - 5 = -7)
  So, the equation simplifies to:
  [
  x = -7
  ]

  ### Verification

  Substitute (x = -7) back into the original equation: [ (-7) + 5 = -2 ] [ -2
  = -2
  \quad \text{(True)} ]

  ### Final Answer

  [ \boxed{x = -7} ]
```

Any idea what the hell is going on?

basically you can dump the prompt and send it like (if you use mods):

```bash
cat ~/deepseek-bug.txt | mods -m r1q4
```

lol i never had such problems with this model at all.  All I have done is downloaded some content from @Thireus repo and threw it onto the llm.   The result is quite surprising.  Let me try to reduce the prompt to figure out what may cause a bug.

the prompt:
```
**Prompt**: can you check how exactly this quantization framework works -- does it select a specific sections of the tensors which are the most responsible for the perplexity?  can you explain?

	File: /opt/GGUF-Tool-Suite/GGUF-Tool-Suite/quant_assign.py
	```
	#!/usr/bin/env python3
	#***************************************************************#
	#** This script is part of Thireus' GGUF Tool Suite.          **#
	#** quant_assign.py the recipe maker tool of choice! Use it   **#
	#** to produce recipes that can be cooked and used by others. **#
	#**                                                           **#
	#** ********************************************************* **#
	#** --------------- Updated: Jul-11-2025 -------------------- **#
	#** ********************************************************* **#
	#**                                                           **#
	#** Author: Thireus <gguf@thireus.com>                        **#
	#**                                                           **#
	#** https://gguf.thireus.com/                                 **#
	#** Thireus' GGUF Tool Suite - Quantize LLMs Like a Chef       **#
	#**                                  ·     ·       ·~°          **#
	#**     Λ,,Λ             ₚₚₗ  ·° ᵍᵍᵐˡ   · ɪᴋ_ʟʟᴀᴍᴀ.ᴄᴘᴘ°   ᴮᶠ¹⁶ ·  **#
	#**    (:·ω·)       。··°      ·   ɢɢᴜғ   ·°·  ₕᵤ𝓰𝓰ᵢₙ𝓰𝒻ₐ𝒸ₑ   ·°   **#
	#**    /    o―ヽニニフ))             · · ɪǫ3_xxs      ~·°        **#
	#**    し―-J                                                   **#
	#**                                                           **#
	#** Copyright © 2025 - Thireus.          Zₑᵣₒ₋ₛₕₒₜ, 𝒻ᵤₗₗ ₙₒₙₛₑₙₛₑ **#
	#***************************************************************#
	#**PLEASE REFER TO THE README FILE FOR ADDITIONAL INFORMATION!**#
	#***************************************************************#
	
	# Requires: pip install pandas numpy argparse
	
	# Tip: You can pipe the output of this script (as long as no warning or debug logs are present) to quants_regex_merger like so: | tee /dev/tty | ./quants_regex_merger.sh
	# python quant_assign.py ppl_results_guessed.csv --gpu-tensors 'blk\.([3-9]|[1-5][0-9]|60)\.ffn_down_shexp\.weight' 'blk\.([3-9]|[1-5][0-9]|60)\.ffn_gate_shexp\.weight' 'blk\.([3-9]|[1-5][0-9]|60)\.ffn_up_shexp\.weight' --cpu-tensors 'blk\.([3-9]|[1-5][0-9]|60)\.ffn_down_exps\.weight' 'blk\.([3-9]|[1-5][0-9]|60)\.ffn_up_exps\.weight' 'blk\.([3-9]|[1-5][0-9]|60)\.ffn_gate_exps\.weight' --cpu-quants iq4_ks iq3_k iq2_k iq1_m_r4 --gpu-quants q8_0 iq6_k iq5_k_r4 --cpu-tensors-max-size 230 --tolerance 0.01 --exponential-factor 8 | ./quants_regex_merger.sh --model-name DeepSeek-R1-0528
	# python quant_assign.py 'ppl_results.csv' --gpu-tensors '.*' --cpu-tensors 'blk\.([3-9]|[1-5][0-9]|60)\.ffn_down_exps\.weight' 'blk\.([3-9]|[1-5][0-9]|60)\.ffn_up_exps\.weight' 'blk\.([3-9]|[1-5][0-9]|60)\.ffn_gate_exps\.weight' --cpu-quants iq4_ks iq3_k iq2_k iq1_m_r4 --gpu-quants q8_0 iq5_k_r4 iq6_k --cpu-tensors-max-size 230 --gpu-tensors-max-size 90% --tolerance 0.01 --exponential-factor 8 | ./quants_regex_merger.sh --model-name DeepSeek-R1-0528
	
	from datetime import datetime
	import time
	import os
	import shlex
	import argparse
	import pandas as pd
	import numpy as np
	import re
	import sys
	import hashlib
	import functools
	import subprocess
	import tempfile
	from collections import Counter
	import textwrap
	
	# Global default quants list
	DEFAULT_QUANTS = ['q8', 'q4']
	
	# Default reducing factors when data not available
	DEFAULT_REDUCE = {
	    32: 1.000,
	    16: 0.999,
	     8: 0.9998,
	     6: 0.9967,
	     4: 0.9763,
	     3: 0.918,
	     2: 0.878,
	     1: 0.395,
	}
	
	# Remote connection settings for tensor_downloader.sh:
	# Please edit tensor_downloader.sh!
	# Resolve script directory for locating tensor_downloader.sh
	script_dir = os.path.dirname(os.path.realpath(__file__))
	tensor_downloader = os.path.join(script_dir, 'tensor_downloader.sh')
	
	if not os.path.isfile(tensor_downloader) or not os.access(tensor_downloader, os.X_OK):
	    print(f"Error: tensor_downloader.sh not found or not executable at {tensor_downloader}", file=sys.stderr)
	    sys.exit(1)
	
	# Cache for fetched map files and parsed maps per quant
	_fetched_maps = set()
	_quant_maps = {}
	
	# Verbosity flags
	DEBUG = False
	INFO = False
	
	# Constants
	GIB = 1024**3 # for GiB-to-bytes conversion
	STRETCH_MIN = 1.0
	STRETCH_MAX = 10.0
	STRETCH_STEP = 0.01
	
	
	def extract_quant_num(qtype):
	    """
	    Extract the first integer in a qtype string.
	    """
	    m = re.search(r"(\d+)", qtype)
	    return int(m.group(1)) if m else float('inf')
	
	
	# Cache for factors loaded via normalised_ppl.py
	_factor_cache = {}
	
	
	def compute_iqr_bounds(values, k):
	    """
	    Compute robust IQR bounds for outlier detection.
	    """
	    arr = np.array(list(values.values()))
	    Q1, Q3 = np.percentile(arr, [25, 75])
	    IQR = Q3 - Q1
	    lower = Q1 - k * IQR
	    upper = Q3 + k * IQR
	    return lower, upper
	
	
	def _call_normalised_ppl(keys):
	    """
	    Call the normalised_ppl.py script for a list of keys, using edges 1 and 32.
	    Returns a dict mapping each numeric key to its fetched factor (float).
	    Raises RuntimeError on parse failure for a key, or subprocess errors.
	    """
	    script_path = os.path.join(os.path.dirname(__file__), 'normalised_ppl.py')
	    keys_list = list(keys)
	    if INFO:
	        print(f"[Info] Calling normalised_ppl.py for keys: {keys_list}")
	    # Compose command: include 1 and 32 as edge values
	    bpw_args = ['1'] + [str(k) for k in keys_list] + ['32']
	    cmd = ['python', script_path, '--bpw-list'] + bpw_args
	    if DEBUG:
	        print(f"[Debug] Running command: {' '.join(shlex.quote(c) for c in cmd)}")
	    try:
	        output = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)
	        if DEBUG:
	            print(f"[Debug] normalised_ppl.py output:\n{output}")
	    except Exception as e:
	        if INFO:
	            print(f"[Warning] normalised_ppl.py call failed: {e}")
	        raise
	
	    # Parse output lines like 'KEY: VALUE'
	    results = {}
	    for line in output.splitlines():
	        parts = line.strip().split(':')
	        if len(parts) != 2:
	            continue
	        try:
	            bpw = float(parts[0])
	            val = float(parts[1])
	        except ValueError:
	            continue
	        # Only collect requested keys
	        if bpw in keys_list:
	            results[bpw] = val
	    # Ensure all requested keys are found
	    missing = set(keys_list) - set(results.keys())
	    if missing:
	        raise RuntimeError(f"Keys {missing} not found in normalised_ppl output")
	    return results
	
	
	def obtain_quants_bpw(qtype):
	    """
	    Infer bits-per-weight (bpw) for each tensor of a quantized map.
	    Compares sizes in the bf16 base map to the qtype map.
	    Returns: dict tensor_name -> bpw (float)
	    """
	    # load base sizes and types from bf16 map
	    base_sizes, base_actual_qtypes = get_map_sizes('bf16')
	    # load quantized sizes (map returns tuple even if actual_qtypes unused)
	    if qtype == 'f32':
	        _qtype = 'bf16'
	    else:
	        _qtype = qtype
	    quant_sizes, quant_actual_qtypes = get_map_sizes(_qtype)
	    bpw_map = {}
	    for name, Sq in quant_sizes.items():
	        Sbase = base_sizes.get(name)
	        if Sbase is None or Sbase == 0:
	            if DEBUG:
	                print(f"[Debug] No base size for tensor {name}, skipping")
	            continue
	        dtype_base = base_actual_qtypes.get(name, 'bf16')
	        # bits per weight in base format
	        if dtype_base in ('bf16', 'fp16'):
	            bbase = 16
	        else:
	            bbase = 32
	        dtype_quant = {quant_actual_qtypes.get(name, qtype)}
	        bpw = bbase * (Sq / Sbase)
	        if quant_actual_qtypes.get(name, qtype) == qtype:
	            bpw_map[name] = bpw
	        else:
	            if DEBUG:
	                print(f"[Debug] Skipping tensor {name} because dtype_quant={dtype_quant} mismatch with exepcted qtype: {qtype}")
	        if DEBUG:
	            print(f"[Debug] Tensor {name}: base dtype={dtype_base}, Sbase={Sbase}, dtype_quant={dtype_quant}, Sq={Sq}, bpw={bpw:.3f}")
	    return bpw_map
	
	
	@functools.lru_cache(maxsize=None)
	def get_bpw(qtype):
	    """
	    Return the bpw for a given qtype, caching results for performance.
	    """
	    # infer bits-per-weight from data instead of hardcoding
	    bpw_map = obtain_quants_bpw(qtype)
	    # compute average bpw across all tensors, fallback if empty
	    if bpw_map:
	        return sum(bpw_map.values()) / len(bpw_map)
	    else:
	        if DEBUG:
	            print(f"[Debug] Could not infer bpw for qtype {qtype}, using extract_quant_num fallback")
	        return extract_quant_num(qtype)
	
	@functools.lru_cache(maxsize=None)
	def get_default_factor(qtype):
	    """
	    Return reducing factor based on bit-width.
	    Attempts to fetch a better factor using normalised_ppl.py, falling back to DEFAULT_REDUCE.
	    Results are cached per bpw.
	    """
	    bpw = get_bpw(qtype)
	    try:
	        if INFO:
	            print(f"[Info] bpw for qtype {qtype}: {bpw}")
	        key = bpw
	    except Exception:
	        if DEBUG:
	            print(f"[Debug] Could not parse bpw from qtype '{qtype}', returning 1.0")
	        return 1.0
	
	    # fallback default
	    default_value = DEFAULT_REDUCE.get(int(key), 1.0)
	
	    # return cached if available
	    if bpw in _factor_cache:
	        if DEBUG:
	            print(f"[Debug] Returning cached factor for bpw {bpw}: {_factor_cache[bpw]}")
	        return _factor_cache[bpw]
	
	    # try to fetch from script for this single key
	    try:
	        fetched = _call_normalised_ppl([bpw])
	        factor = fetched.get(bpw, default_value)
	    except Exception:
	        factor = default_value
	    else:
	        if DEBUG:
	            print(f"[Debug] Caching factor for bpw {bpw}: {factor}")
	        _factor_cache[bpw] = factor
	
	    return factor
	
	
	def parse_value(val):
	    """
	    Parse a PPL string or number, stripping '%' if present.
	    """
	    if pd.isna(val):
	        return np.nan
	    s = str(val).strip()
	    if s.endswith('%'):
	        s = s[:-1]
	    try:
	        return float(s)
	    except:
	        return np.nan
	
	
	def classify_tensors(columns, cpu_patterns, gpu_patterns):
	    """
	    Classify tensor names into CPU/GPU based on regex lists.
	    """
	    classes = {'cpu': [], 'gpu': []}
	    for name in columns:
	        assigned = False
	        for pat in cpu_patterns:
	            if re.fullmatch(pat, name):
	                classes['cpu'].append(name)
	                assigned = True
	                break
	        if assigned:
	            continue
	        for pat in gpu_patterns:
	            if re.fullmatch(pat, name):
	                classes['gpu'].append(name)
	                assigned = True
	                break
	        if not assigned:
	            classes['gpu'].append(name)
	    return classes
	
	
	def group_tensors(names):
	    """
	    Group tensor names by base name (strip leading layer indices).
	    """
	    groups = {}
	    for name in names:
	        m = re.match(r"blk\.\d+\.(.*)", name)
	        base = m.group(1) if m else name
	        groups.setdefault(base, []).append(name)
	    return groups
	
	
	def select_qtype(df, qtype_arg):
	    """
	    Select the row for given QTYPE or lowest quant.
	    """
	    if qtype_arg:
	        if qtype_arg not in df['QTYPE'].values:
	            print(f"Error: qtype '{qtype_arg}' not found in CSV.")
	            sys.exit(1)
	        return df[df['QTYPE'] == qtype_arg].iloc[0]
	    df['__quant_num__'] = df['QTYPE'].map(extract_quant_num)
	    sel = df.nsmallest(1, '__quant_num__').iloc[0]
	    df.drop(columns='__quant_num__', inplace=True)
	    return sel
	
	
	def fetch_map_for_qtype(qtype):
	    """
	    Fetch and cache tensors.{qtype}.map via tensor_downloader.sh.
	    """
	    if qtype in _fetched_maps:
	        return True
	    tmpdir = tempfile.gettempdir()
	    local_map = os.path.join(tmpdir, f"tensors.{qtype}.map")
	    cmd = [tensor_downloader, qtype.upper(), "0", tmpdir, f"tensors.{qtype}.map"]
	    if INFO: print(f"[Info] Fetching map for {qtype}...")
	    try:
	        if DEBUG or INFO:
	            subprocess.run(cmd, check=True)
	        else:
	            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
	        if INFO: print(f"[Info] Saved map to {local_map}")
	        _fetched_maps.add(qtype)
	        return True
	    except subprocess.CalledProcessError as e:
	        print(f"[Warning] failed to fetch tensors.map: {e}")
	        return False
	
	
	def get_map_sizes(qtype):
	    """
	    Return parsed map sizes for given qtype, caching results.
	    """
	    if qtype not in _quant_maps:
	        fetch_map_for_qtype(qtype)
	        # parse_map_file now returns tuple
	        _quant_maps[qtype] = parse_map_file(qtype)
	    return _quant_maps[qtype]
	
	
	def parse_map_file(qtype):
	    """
	    Parse local tensors.{qtype}.map into:
	      - sizes: dict tensor_name -> bytes_size
	      - actual_qtypes: dict tensor_name -> dtype (e.g., 'bf16', 'f32')
	    """
	    tmpdir = tempfile.gettempdir()
	    path = os.path.join(tmpdir, f"tensors.{qtype}.map")
	    sizes = {}
	    actual_qtypes = {}
	    if not os.path.exists(path):
	        return sizes, actual_qtypes
	    with open(path) as f:
	        for line in f:
	            parts = line.strip().split(':')
	            if len(parts) < 5:
	                continue
	            # parts example:
	            # [file, checksum, tensor_name, shape=..., dtype=f32, elements=..., bytes=...]
	            tensor_name = parts[2]
	            # find dtype and bytes fields
	            dtype = None
	            size_bytes = None
	            for p in parts:
	                if p.startswith('dtype='):
	                    dtype = p.split('=')[1]
	                elif p.startswith('bytes='):
	                    size_bytes = int(p.split('=')[1])
	            if dtype is None or size_bytes is None:
	                continue
	            sizes[tensor_name] = size_bytes
	            actual_qtypes[tensor_name] = dtype
	    return sizes, actual_qtypes
	
	
	def load_sample_ppl_table(path):
	    """
	    Load sample PPL CSV and compute reduction factors per base name.
	    """
	    sample_df = pd.read_csv(path, index_col=0)
	    sample_df = sample_df.replace(['404','404.0'], np.nan)
	    dropped = [c for c in sample_df.columns if sample_df[c].isna().any()]
	    if dropped and INFO:
	        print(f"[Info] Dropping sample PPL columns with missing values: {dropped}")
	    sample_df = sample_df.drop(columns=dropped)
	    max_vals = sample_df.max()
	    red = sample_df.div(max_vals)
	    return {col: red[col].to_dict() for col in sample_df.columns}
	
	# --- New spread-based assignment logic ---
	
	def compute_class_midpoint(class_values, forced_mid=None):
	    """
	    Compute mean PPL or use forced midpoint.
	    """
	    if forced_mid is not None:
	        mid = forced_mid
	        if DEBUG: print(f"[Debug] Forced midpoint: {mid:.4f}")
	    else:
	        mid = np.mean(list(class_values.values()))
	        if DEBUG: print(f"[Debug] Class midpoint (mean PPL): {mid:.4f}")
	    return mid
	
	
	def compute_group_spreads(class_values, forced_mid=None):
	    """
	    Compute each tensor's spread in [-1,1], corrected formula for upper side.
	    """
	    mid = compute_class_midpoint(class_values, forced_mid)
	    vals = list(class_values.values())
	    min_ppl, max_ppl = min(vals), max(vals)
	    down = abs(min_ppl - mid) or 1e-6
	    up = abs(max_ppl - mid) or 1e-6
	    spreads = {}
	    for name, ppl in class_values.items():
	        if ppl < mid:
	            rel = (ppl - min_ppl) / down
	            spread = -(1 - min(1, rel))
	        else:
	            rel = (max_ppl - ppl) / up  # corrected
	            spread = 1 - min(1, rel)
	        spreads[name] = spread
	        if DEBUG: print(f"[Debug] Tensor {name}: PPL={ppl:.4f}, spread={spread:.4f}")
	    return spreads
	
	
	def compute_quant_intervals(quants, stretch=1.0):
	    """
	    Compute normalized spread intervals from 1 to -1 per quant,
	    applying stretching factor to reducing factors.
	    """
	    # apply stretching: new_factor = old_factor ** (1/stretch)
	    widths = {}
	    for q in quants:
	        orig = get_default_factor(q)
	        stretched = orig * (1.0 / stretch)
	        #print("orig:", orig, "stretch:", stretch, "stretched:", stretched)
	        widths[q] = (1 - stretched)
	    total = sum(widths.values()) or 1e-6
	    norm = total / 2
	    intervals = []
	    top = 1.0
	    for q in quants:
	        span = widths[q] / norm
	        bottom = top - span
	        intervals.append((q, top, bottom))
	        if DEBUG:
	            print(f"[Debug] Quant {q} @stretch={stretch:.2f}: interval ({bottom:.4f}, {top:.4f}]")
	        top = bottom
	    return intervals
	
	
	def assign_quants(quants, _, class_values, forced_mid=None, stretch=1.0):
	    """
	    Assign quants based on spread intervals and fetch correct tensor sizes.
	    """
	    if INFO:
	        print(f"[Info] Performing spread-based quant assignment (stretch={stretch:.2f})...")
	    spreads = compute_group_spreads(class_values, forced_mid)
	    intervals = compute_quant_intervals(quants, stretch)
	    assignment = {}
	    sizes = {}
	    for name in class_values:
	        spread = spreads[name]
	        for q, top, bottom in intervals:
	            if bottom < spread <= top:
	                assignment[name] = q
	                break
	        else:
	            assignment[name] = quants[-1]
	        sizes[name], _ = get_map_sizes(assignment[name])
	        sizes[name] = sizes[name].get(name, 0)
	        if INFO:
	            print(f"[Info] Assigned {assignment[name]} to {name} (spread={spread:.4f}) size={sizes[name]}")
	    return assignment, sizes
	
	
	def total_size_for_quant(names, qtype):
	    """
	    Sum the map sizes for the given tensor names under the specified quant.
	    """
	    sizes_map, _ = get_map_sizes(qtype)
	    return sum(sizes_map.get(name, 0) for name in names)
	
	
	def optimize_midpoint_and_assign(quants, _, class_values,
	                                 max_bytes, tolerance=0.05, exp_factor=1.0):
	    """
	    Loop over stretch factors and perform midpoint optimization using class mean with dichotomy.
	    exp_factor controls exponent in stretch calculation: higher = more aggressive extremes.
	    """
	    if INFO:
	        print(f"[Info] Starting optimization for target size {max_bytes} bytes ±{tolerance*100}% with exp_factor={exp_factor:.2f}...")
	    best_assign, best_size = {}, float('inf')
	    # compute initial midpoint as class mean
	    class_mid = compute_class_midpoint(class_values)
	    # outer loop: stretch factor sweep
	    stretch = STRETCH_MIN
	    while stretch <= STRETCH_MAX:
	        if INFO and stretch > STRETCH_MIN:
	            print(f"[Info] Trying stretch factor {stretch:.2f}...")
	        # reset bisection bounds for each stretch
	        low_val, high_val = min(class_values.values()), max(class_values.values())
	        # compute exponential boundary modifier
	        exponential_factor = (STRETCH_MAX/stretch) ** exp_factor
	        low_val *= exponential_factor
	        high_val *= exponential_factor
	        # start midpoint clamped to [low_val, high_val]
	        mid = max(low_val, min(high_val, class_mid))
	        prev_mid = None
	        change = None
	        change_min_threshold = 0.0001
	        mid_min_threshold = 0.00001
	        if INFO:
	            print(f"[Info] Progress: {stretch/STRETCH_MAX*100:.2f}%")
	        # inner loop: dichotomy until converged
	        while (prev_mid == None or prev_mid > mid_min_threshold) and (change == None or change >= change_min_threshold):
	            if INFO:
	                print(f"[Info] Evaluating midpoint={mid:.6f}, stretch={stretch:.2f}...")
	            assignment, sizes = assign_quants(quants, None,
	                                             class_values,
	                                             forced_mid=mid, stretch=stretch)
	            size = sum(sizes.values())
	            # tolerance check
	            if abs(size - max_bytes) / max_bytes <= tolerance:
	                if INFO:
	                    print(f"[Info] Found acceptable size {size} at midpoint={mid:.6f}, stretch={stretch:.2f}.")
	                return assignment, size
	            # check midpoint change
	            if prev_mid is not None:
	                change = abs(mid - prev_mid) / prev_mid
	                if change < change_min_threshold:  # less than 0.01%
	                    if INFO:
	                        print(f"[Info] Midpoint change {change*100:.4f}% below threshold; breaking inner loop.")
	                    break
	            prev_mid = mid
	            # decide direction and update bounds
	            if size < max_bytes:
	                high_val = mid
	            else:
	                low_val = mid
	            if INFO:
	                reason = 'too small' if size < max_bytes else 'too large'
	                direction = 'down' if size < max_bytes else 'up'
	                print(f"[Info] Size {size} is {reason}; moving midpoint {direction}.")
	            # compute next midpoint by dichotomy
	            mid = (low_val + high_val) / 2
	            # track best
	            if abs(size - max_bytes) < abs(best_size - max_bytes):
	                best_size, best_assign = size, assignment.copy()
	        # increment stretch factor
	        stretch = round(stretch + STRETCH_STEP, 2)
	    if INFO:
	        print("[Warning] Optimization finished; using best found assignment.")
	    return best_assign, best_size
	
	def scale_for_size(assignment, sizes, quants, max_size_bytes):
	    """
	    Fallback simple scaling if optimized assignment not used.
	    """
	    total = sum(sizes.values())
	    if INFO: print(f"[Info] Starting fallback scaling: current total {total}, target {max_size_bytes}")
	    if total <= max_size_bytes:
	        return assignment, total
	    items = list(assignment.items())
	    while total > max_size_bytes:
	        made_change = False
	        for name, q in items:
	            idx = quants.index(q)
	            if idx + 1 < len(quants):
	                new_q = quants[idx+1]
	                assignment[name] = new_q
	                sizes[name], _ = get_map_sizes(new_q)
	                sizes[name] = sizes[name].get(name, 0)
	                made_change = True
	                total = sum(sizes.values())
	                if INFO: print(f"[Info] Scaling {name} from {q} to {new_q}, new total {total}")
	                if total <= max_size_bytes:
	                    return assignment, total
	        if not made_change:
	            if INFO: print("[Warning] Cannot reduce size further via fallback scaling.")
	            break
	    return assignment, total
	
	
	def _convert_value(v):
	    """
	    Convert a CSV cell value v to float, handling percentage strings.
	    """
	    if isinstance(v, str) and v.endswith('%'):
	        try:
	            return float(v.rstrip('%')) / 100.0
	        except ValueError:
	            return np.nan
	    try:
	        return float(v)
	    except (TypeError, ValueError):
	        return np.nan
	
	def assign_qtype(default_qtype, regex_assign_list, quants, names):
	    """
	    Build a dict mapping each tensor in `names` to a QTYPE.
	    - If regex_assign_list is non-empty, scan in order, first match wins.
	    - Otherwise fall back to default_qtype (or highest‑bpw if default_qtype is None).
	    """
	    # Resolve ultimate default
	    if default_qtype:
	        base_q = default_qtype
	    else:
	        base_q = max(quants, key=get_bpw)
	
	    out = {}
	    for name in names:
	        assigned = None
	        # Try regex overrides first
	        for pat, qt in regex_assign_list:
	            if pat.fullmatch(name):
	                assigned = qt
	                break
	        if assigned is None:
	            assigned = base_q
	        out[name] = assigned
	    return out
	
	
	def main():
	    global DEBUG, INFO
	    parser = argparse.ArgumentParser(description="Assign optimal quants per tensor based on PPL CSV.")
	    parser.add_argument('--debug', action='store_true', help='Show debug logs')
	    parser.add_argument('--info', action='store_true', help='Show info logs')
	    parser.add_argument('--tolerance', type=float, default=0.05,
	                        help='Relative GiB tolerance for size optimization')
	    parser.add_argument('--cpu-irq-k', type=float, default=1.5,
	                        help='IQR multiplier k for CPU outlier detection')
	    parser.add_argument('--gpu-irq-k', type=float, default=1.5,
	                        help='IQR multiplier k for GPU outlier detection')
	    parser.add_argument('csv_file', help='Input CSV file')
	    parser.add_argument('--qtype', help='QTYPE to analyze (default: lowest quant)')
	    parser.add_argument('--cpu-assign-qtype', help='QTYPE to assign to non-measured CPU tensors or tensors missing from csv (default: highest quant)')
	    parser.add_argument('--gpu-assign-qtype', help='QTYPE to assign to non-measured GPU tensors or tensors missing from csv (default: highest quant)')
	    parser.add_argument('--cpu-assign-tensors', nargs='+', default=[], help="List of regex=QTYPE patterns for CPU tensors to force-assign")
	    parser.add_argument('--gpu-assign-tensors', nargs='+', default=[], help="List of regex=QTYPE patterns for GPU tensors to force-assign")
	    #parser.add_argument('--sample-ppl', help='CSV sample PPL file path', required=True)
	    parser.add_argument('--cpu-tensors', nargs='+', default=[], help='Regex patterns for CPU tensors')
	    parser.add_argument('--gpu-tensors', nargs='+', default=[], help='Regex patterns for GPU tensors')
	    parser.add_argument('--cpu-quants', nargs='+', help='Ordered list of CPU quants')
	    parser.add_argument('--gpu-quants', nargs='+', help='Ordered list of GPU quants')
	    parser.add_argument('--cpu-tensors-max-size', type=str, help='Max CPU tensors size in GiB or percent (e.g., 80%)')
	    parser.add_argument('--gpu-tensors-max-size', type=str, help='Max GPU tensors size in GiB or percent (e.g., 80%)')
	    parser.add_argument('--exponential-factor', type=float, default=1.0,
	                        help='Exponent controlling midpoint adjustment aggressiveness during stretch sweeps. '
	                             'Higher values push quantization toward extremes; default is 1.0.')
	    parser.add_argument('--ignore-f32', action='store_true', help='Ignore f32 tensors (default: not ignored)')
	    parser.add_argument('--tensors-from-csv', action='store_true', help='Obtains list of tensors from csv file only (default: tensors are obtained from map file)')
	    args = parser.parse_args()
	
	    def parse_regex_assign_list(raw_list):
	        parsed = []
	        for item in raw_list:
	            try:
	                pat, qt = item.split('=', 1)
	            except ValueError:
	                parser.error(f"Invalid regex‑assign spec: {item}. Must be PATTERN=QTYPE")
	            parsed.append((re.compile(pat), qt))
	        return parsed
	
	    cpu_regex_assign = parse_regex_assign_list(args.cpu_assign_tensors)
	    gpu_regex_assign = parse_regex_assign_list(args.gpu_assign_tensors)
	
	    DEBUG = args.debug
	    INFO = args.info or DEBUG
	
	    if args.cpu_tensors and not args.cpu_quants:
	        parser.error("--cpu-quants is required when --cpu-tensors is used")
	    if args.gpu_tensors and not args.gpu_quants:
	        parser.error("--gpu-quants is required when --gpu-tensors is used")
	
	    cpu_quants = args.cpu_quants or DEFAULT_QUANTS
	    # Reorder cpu_quants from highest to lowest bpw
	    try:
	        cpu_quants = sorted(cpu_quants, key=get_bpw, reverse=True)
	        if INFO: print(f"[Info] CPU quants reordered by bpw: {cpu_quants}")
	    except Exception:
	        pass
	
	    gpu_quants = args.gpu_quants or DEFAULT_QUANTS
	    # Reorder gpu_quants from highest to lowest bpw
	    try:
	        gpu_quants = sorted(gpu_quants, key=get_bpw, reverse=True)
	        if INFO: print(f"[Info] GPU quants reordered by bpw: {gpu_quants}")
	    except Exception:
	        pass
	
	    if INFO: print(f"[Info] Loading CSV: {args.csv_file}")
	    df = pd.read_csv(args.csv_file)
	    if 'QTYPE' not in df.columns:
	        print("Error: CSV must have 'QTYPE' as first column.")
	        sys.exit(1)
	
	    #reduction_factors = load_sample_ppl_table(args.sample_ppl)
	    row = select_qtype(df, args.qtype)
	    qtype = row['QTYPE']
	    if INFO: print(f"[Info] Selected QTYPE: {qtype}")
	
	    # Pre-fetch maps
	    fetch_map_for_qtype(qtype)
	    _, items = get_map_sizes(qtype)
	    _, items_bf16 = get_map_sizes('bf16')
	    _items = items_bf16
	    if not _items:
	        _items = items
	
	    # Collect tensor names (either from csv or from map file)
	    if INFO: print(f"[Info] Get all tensor names")
	    if args.tensors_from_csv:
	        tensor_names = [c for c in df.columns if c != 'QTYPE']
	    else:
	        tensor_names = [n for n,d in _items.items()]
	
	    # Identify all f32 tensors once
	    if INFO: print(f"[Info] Get f32 tensor names")
	    # get_map_sizes returns (sizes, actual_qtypes)
	    f32_names = [n for n,d in _items.items() if d == 'f32']
	
	    # Classify tensors
	    classes = classify_tensors(tensor_names, args.cpu_tensors, args.gpu_tensors)
	
	    subclasses_to_assign = {'cpu': [], 'gpu': []}
	    subclasses_assigned = {'cpu': [], 'gpu': []}
	
	    # Build values dict, converting strings (e.g. '0.0653%') properly and pre-assign tensors that haven't been measured
	    values = {}
	    pre_assignments = {}
	    pre_assignments_offset = {'cpu': 0, 'gpu': 0}
	    for cls in ['cpu', 'gpu']:
	        quants = cpu_quants if cls == 'cpu' else gpu_quants
	        names = classes.get(cls, [])
	        if cls == 'cpu':
	            _assign_qtype = assign_qtype(args.cpu_assign_qtype, cpu_regex_assign, quants, names)
	        else:
	            _assign_qtype = assign_qtype(args.gpu_assign_qtype, gpu_regex_assign, quants, names)
	
	        # skip if nothing for this cls
	        if not names:
	            continue
	
	        for name in names:
	            # missing measurement → pre‑assign
	            if name not in row or pd.isna(row.at[name]):
	                if name in f32_names:
	                    # This is a f32 tensor which we must skip
	                    continue
	                pre_assignments[name] = _assign_qtype[name]
	
	                subclasses_assigned[cls].append(name)
	                if INFO: print(f"[Info] Assigning {name!r} → {pre_assignments[name]!r} (missing metrics)")
	
	                # jump to next tensor
	                continue
	
	            # got a raw value → convert and store
	            raw = row[name]
	            conv = _convert_value(raw)
	            if np.isnan(conv):
	                print(f"Error: could not parse numeric value for tensor {name!r}: {raw!r}")
	                sys.exit(1)
	
	            values[name] = conv
	            subclasses_to_assign[cls].append(name)
	
	        # 1. Get all unique q‑types
	        _assign_qtype_qtypes = set(_assign_qtype.values())
	
	        # 2. Loop over each q‑type
	        for _qtype in _assign_qtype_qtypes:
	            # 2a. Collect all tensor names that were assigned this qtype
	            _tensor_subgroup_names = [
	                name
	                for name, assigned_q in _assign_qtype.items()
	                if assigned_q == _qtype and name in pre_assignments
	            ]
	
	            # 2b. Compute the total size for this group
	            size = total_size_for_quant(_tensor_subgroup_names, _qtype)
	
	            # 2c. Add it into your pre_assignments_offset for whatever class 'cls' is
	            #     (you’ll need to define or look up `cls` in your context)
	            pre_assignments_offset[cls] += size
	
	    totals = {}
	    # Create separate assignment storage per class to avoid mixing identical qnames
	    assignments = {'cpu': {}, 'gpu': {}}
	
	    # prepare per-class f32 offsets (skip if user manually included 'f32' as a quant)
	    f32_offset = {'cpu': 0, 'gpu': 0}
	    f32_classes = {}
	    add_f32 = not args.ignore_f32
	    if add_f32:
	        f32_classes = classify_tensors(f32_names, args.cpu_tensors, args.gpu_tensors)
	        for cls in ['gpu', 'cpu']:
	            # if user did *not* list 'f32' in 'cls'_quants, add to cls offset
	            if 'f32' not in (cpu_quants if cls=='cpu' else gpu_quants):
	                f32_offset[cls] = total_size_for_quant(f32_classes.get(cls, []), 'bf16')
	                if f32_offset[cls] == 0:
	                    f32_offset[cls] = total_size_for_quant(f32_classes.get(cls, []), qtype)
	
	    # Track precomputed extremes per class
	    extremes = {}
	
	    # Process GPU and CPU classes
	    for cls in ['gpu', 'cpu']:
	        quants = cpu_quants if cls == 'cpu' else gpu_quants
	        names = classes.get(cls, [])
	        names_to_assign = subclasses_to_assign.get(cls, [])
	        names_assigned = subclasses_assigned.get(cls, [])
	        if not names:
	            continue
	   
	        print(f"\n## {'CPU' if cls=='cpu' else 'GPU'}-loaded tensors")
	        class_vals = {n: values[n] for n in names_to_assign}
	
	        # Determine bounds and outliers
	        k_val = args.cpu_irq_k if cls=='cpu' else args.gpu_irq_k
	        lower, upper = compute_iqr_bounds(class_vals, k_val)
	        if INFO: print(f"[Info] {cls.upper()} outlier bounds: lower={lower:.4f}, upper={upper:.4f}")
	        out_low = [n for n,v in class_vals.items() if v < lower]
	        out_high = [n for n,v in class_vals.items() if v > upper]
	        if DEBUG: print(f"[Debug] {cls.upper()} low outliers: {out_low}")
	        if DEBUG: print(f"[Debug] {cls.upper()} high outliers: {out_high}")
	
	        # Assign extremes and compute outlier size deduction
	        outlier_bytes = 0
	        for n in out_low:
	            assignments[cls][n] = quants[-1]
	            size = total_size_for_quant([n], quants[-1])
	            outlier_bytes += size
	            if INFO: print(f"[Info] Assigned lowest quant {quants[-1]} to low outlier {n}, size={size/GIB:.3f} GiB")
	        for n in out_high:
	            assignments[cls][n] = quants[0]
	            size = total_size_for_quant([n], quants[0])
	            outlier_bytes += size
	            if INFO: print(f"[Info] Assigned highest quant {quants[0]} to high outlier {n}, size={size/GIB:.3f} GiB")
	        for n in out_low + out_high:
	            class_vals.pop(n, None)
	
	        # Normal assignment on remaining
	        
	        # Determine max-size argument, allowing percent
	        raw_max = args.cpu_tensors_max_size if cls == 'cpu' else args.gpu_tensors_max_size
	        max_arg_bytes = None
	        # Precompute extremes once
	        highest_q = max(quants, key=get_bpw)
	        lowest_q = min(quants, key=get_bpw)
	        max_ref = total_size_for_quant(names_to_assign, highest_q) + f32_offset[cls] + pre_assignments_offset[cls]
	        min_ref = total_size_for_quant(names_to_assign, lowest_q) + f32_offset[cls] + pre_assignments_offset[cls]
	        extremes[cls] = {
	            'highest_q': highest_q, 'lowest_q': lowest_q,
	            'max_ref': max_ref, 'min_ref': min_ref
	        }
	
	        _max_arg_bytes = 0
	        if raw_max:
	            if isinstance(raw_max, str) and raw_max.endswith('%'):
	                pct = float(raw_max.rstrip('%')) / 100.0
	                _max_arg_bytes = pct * max_ref
	                if INFO: print(f"[Info] {cls.upper()} max-size set to {raw_max} of {highest_q} total ({max_ref/GIB:.3f} GiB) = {_max_arg_bytes/GIB:.3f} GiB")
	            else:
	                _max_arg_bytes = float(raw_max) * GIB
	            max_arg_bytes = _max_arg_bytes
	            max_arg_bytes -= outlier_bytes # deduct outliers
	            max_arg_bytes -= f32_offset.get(cls, 0) # deduct f32 offset
	            max_arg_bytes -= pre_assignments_offset.get(cls, 0) # deduct pre-assigned offset
	            if INFO: print(f"[Info] Deducted outliers and f32 total {outlier_bytes/GIB:.3f} GiB from target, adjusted max={max_arg_bytes/GIB:.3f} GiB")
	
	        if _max_arg_bytes >= (max_ref - max_ref*0.0001):
	            # Assign highest quant to all (except extremes)
	            if INFO: print(f"[Info] Reasonably assigning highest quant to all tensors...")
	            assignment, sizes = assign_quants(
	                [highest_q], None, class_vals)
	            total_bytes = sum(sizes.values())
	        elif _max_arg_bytes == 0:
	            # Assign lowest quant to all (except extremes)
	            if INFO: print(f"[Info] Reasonably assigning lowest quant to all tensors...")
	            assignment, sizes = assign_quants(
	                [lowest_q], None, class_vals)
	            total_bytes = sum(sizes.values())
	        elif max_arg_bytes:
	            assignment, total_bytes = optimize_midpoint_and_assign(
	                quants, None, class_vals,
	                max_arg_bytes, args.tolerance, args.exponential_factor)
	            #print(f"# Optimized sub-total {cls.upper()} size excluding outliers and f32: {total_bytes/GIB:.3f} GiB")
	        else:
	            assignment, sizes = assign_quants(
	                quants, None, class_vals)
	            total_bytes = sum(sizes.values())
	
	        assignments[cls].update(assignment)  # Store per-class assignments
	        totals[cls] = total_bytes + outlier_bytes # add outliers back
	        totals[cls] += f32_offset.get(cls, 0) # add f32 offset to the grand total
	        totals[cls] += pre_assignments_offset.get(cls, 0) # add pre-assigned offset to the grand total
	        print(f"# Total {cls.upper()} size: {totals[cls]/GIB:.3f} GiB")
	        print(f"# Outlier tensors total size: {outlier_bytes/GIB:.3f} GiB")
	        print(f"# f32 tensors total size: {f32_offset.get(cls, 0)/GIB:.3f} GiB")
	        print(f"# Pre-assigned tensors total size: {pre_assignments_offset.get(cls, 0)/GIB:.3f} GiB")
	        if max_arg_bytes:
	            print(f"# Optimized sub-total {cls.upper()} size excluding outliers and f32: {total_bytes/GIB:.3f} GiB")
	
	        # List any auto‑added f32 tensors in the output
	        if add_f32 and f32_offset.get(cls,0) > 0:
	            print(f"# Auto‑included f32 tensors for {cls.upper()}:")
	            for n in sorted(f32_names):
	                # _size = total_size_for_quant([n], 'bf16')
	                # if _size == 0:
	                #     _size = total_size_for_quant([n], qtype)
	                # size = _size / GIB
	                print(f"{re.escape(n)}=f32")
	
	        groups = group_tensors(names)
	        for base, full in groups.items():
	            displayed_already = False
	            for name in sorted((n for n in full if n in pre_assignments), key=lambda n: pre_assignments[n], reverse=True):
	                if not displayed_already:
	                    print(f"# Group: {re.escape(base)}")
	                    displayed_already = True
	                print(f"{re.escape(name)}={pre_assignments.get(name,'')}")
	            for name in sorted((n for n in full if n in values), key=lambda n: values[n], reverse=True):
	                if not displayed_already:
	                    print(f"# Group: {re.escape(base)}")
	                    displayed_already = True
	                print(f"{re.escape(name)}={assignments[cls].get(name,'')}")
	
	    # Summary of tensor sizes per class
	    print("\n## Summary of tensor sizes per class")
	    _tb = 0
	    _pct = 0
	    for cls, tb in totals.items():
	        # Retrieve extremes
	        ext = extremes.get(cls, {})
	        highest_q = ext.get('highest_q')
	        lowest_q = ext.get('lowest_q')
	        max_size = ext.get('max_ref', 0) / GIB
	        min_size = ext.get('min_ref', 0) / GIB
	        # Percentage of max q-size
	        pct = (tb / (max_size * GIB)) * 100 if max_size > 0 else 0
	        _tb += tb
	        _pct += pct
	        print(f"#{cls.upper():>4} Total: {tb/GIB:.3f} GiB ({pct:.1f}%) | {max_size:.2f} GiB max, if all were {highest_q} | {min_size:.2f} GiB min, if all were {lowest_q}")
	    
	    print(f"# GPU+CPU Total: {_tb/GIB:.3f} GiB ({_pct/2:.1f}%)")
	
	    # Summary tensor counts and bits-per-weight per qtype
	    print("\n## Summary of tensor counts and bpw per qtype")
	
	    # Build a combined list of all qtypes, maintaining order
	    all_qtypes = []
	    if cpu_quants:
	        all_qtypes.extend(cpu_quants)
	    if gpu_quants:
	        all_qtypes.extend(gpu_quants)
	    seen = set()
	    ordered_qtypes = []
	    for qt in all_qtypes:
	        if qt not in seen:
	            seen.add(qt)
	            ordered_qtypes.append(qt)
	
	    # Use separate assignment maps per class (already handled earlier)
	    quant_counts_by_class = {
	        'cpu': Counter(assignments.get('cpu', {}).values()),
	        'gpu': Counter(assignments.get('gpu', {}).values())
	    }
	
	    _bytes = 0
	    _bpw = 0
	    _w = 0
	    for cls in ['gpu', 'cpu']:
	        quants_list = gpu_quants if cls == 'gpu' else cpu_quants
	        _quants_list = quants_list
	        if add_f32:
	            if 'f32' not in (cpu_quants if cls=='cpu' else gpu_quants):
	                quants_list = ['f32'] + _quants_list
	        names = classes.get(cls, [])
	        names_assigned = subclasses_assigned.get(cls, [])
	        names_post_assigned = subclasses_to_assign.get(cls, [])
	        if not quants_list:
	            continue
	        # Section header per class
	        print(f"#\n# {cls.upper()}-loaded quants:")
	        print(f"# QTYPE\t\tCount\tBPW\tAssigned GiB\t% Assigned\tMax GiB (all)")
	        if cls == 'cpu':
	            _assign_qtype = assign_qtype(args.cpu_assign_qtype, cpu_regex_assign, _quants_list, names_assigned)
	        else:
	            _assign_qtype = assign_qtype(args.gpu_assign_qtype, gpu_regex_assign, _quants_list, names_assigned)
	        if _assign_qtype not in (cpu_quants if cls=='cpu' else gpu_quants):
	            unique_qtypes = set(_assign_qtype.values())
	            quants_list = list(set(quants_list).union(unique_qtypes))
	        # Sort quants by bits-per-weight descending
	        sorted_quants = sorted(quants_list, key=lambda q: get_bpw(q) or 0, reverse=True)
	        for qt in sorted_quants:
	            try:
	                bpw_val = get_bpw(qt)
	            except:
	                bpw_val = 0
	            if qt == 'f32':
	                sizes_map, _ = get_map_sizes('bf16')
	            else:
	                sizes_map, _ = get_map_sizes(qt)
	
	            # Pre-assigned tensors
	            if qt in _assign_qtype.values() or qt == 'f32':
	                # If we’re in the f32‐override case, swap in the f32 list; otherwise keep the original.
	                if add_f32 and qt == 'f32':
	                    # Overwrite names_assigned list
	                    names_assigned = f32_classes.get(cls, [])
	                    cnt = len(names_assigned)
	                    if cnt > 0:
	                        assigned_bytes = sum(sizes_map.get(n, 0) for n in names_assigned)
	                        assigned_gib = assigned_bytes / GIB
	                        _bytes += assigned_bytes
	                        _bpw += bpw_val * assigned_bytes
	                        _w += assigned_bytes / bpw_val
	                        print(f"# +{qt:<10}\t{cnt:<3}\t{bpw_val:<6}\t{assigned_gib:>6.2f} GiB\t-\t\t-")
	                else:
	                    # for each qt in whatever loop you have:
	                    # collect the names whose assigned q‐type is qt
	                    names_assigned = [name 
	                                    for name, q in _assign_qtype.items() 
	                                    if q == qt]
	                    # now cnt and assigned_bytes only refer to that subset:
	                    cnt = len(names_assigned)
	                    assigned_bytes = sum(sizes_map.get(n, 0) for n in names_assigned)
	                    assigned_gib = assigned_bytes / GIB
	                    _bytes += assigned_bytes
	                    _bpw   += bpw_val * assigned_bytes
	                    _w += assigned_bytes / bpw_val
	                    print(f"# +{qt:<10}\t{cnt:<3}\t{bpw_val:<6}\t{assigned_gib:>6.2f} GiB\t-\t\t-")
	            
	            # Post-assigned tensors
	            cnt = Counter(assignments[cls].values()).get(qt, 0)
	            if (cnt > 0 or qt in _quants_list) and qt != 'f32':
	                # Assigned size
	                assigned = [n for n, q in assignments[cls].items() if q == qt]
	                assigned_bytes = sum(sizes_map.get(n, 0) for n in assigned)
	                assigned_gib = assigned_bytes / GIB
	                _bytes += assigned_bytes
	                _bpw += bpw_val * assigned_bytes
	                _w += assigned_bytes / bpw_val
	                # Max size if all
	                max_gib = total_size_for_quant(names_post_assigned, qt) / GIB
	                pct = (assigned_bytes / (max_gib * GIB) * 100) if max_gib > 0 else 0
	                print(f"# {qt:<10}\t{cnt:<3}\t{bpw_val:<6}\t{assigned_gib:>6.2f} GiB\t{pct:>3.1f}%\t\t{max_gib:.2f}")
	    
	    print(f"#\n# -Average BPW: {_bytes/_w:.4f}")
	
	    print(f"#\n# -Notes:\n# - '+' means user-defined pre-assigned tensors and f32 tensors")
	
	    now = datetime.now().astimezone()  # Gets local time with tzinfo if available
	    current_time = now.strftime("%Y-%m-%d %H:%M:%S %Z%z")
	    print(f"# - Recipe produced on the {current_time} using Thireus' GGUF tools (https://gguf.thireus.com/)")
	    # Compute SHA-256 of the current script (if readable)
	    script_path = sys.argv[0]
	    if os.path.isfile(script_path):
	        try:
	            with open(script_path, 'rb') as f:
	                sha256 = hashlib.sha256(f.read()).hexdigest()
	        except Exception:
	            sha256 = "ERROR"
	    else:
	        sha256 = "N/A"
	    print(f"# - Script SHA-256: {sha256}")
	    # Reconstruct a safely quoted command‐line
	    quoted_args = [shlex.quote(arg) for arg in sys.argv]
	    command_line = ' '.join(quoted_args)
	
	    # Wrap the command into lines starting with "# "
	    wrapped_lines = textwrap.wrap(
	        command_line,
	        width=115,  # 80 - len("# ") - len(" \\")
	        break_long_words=False,
	        break_on_hyphens=False
	    )
	    # Add "# " prefix and " \\" suffix to each line, except the last one
	    formatted_lines = [
	        f"# {line} \\" if i < len(wrapped_lines) - 1 else f"# {line}"
	        for i, line in enumerate(wrapped_lines)
	    ]
	    print(f"# - Command used:")
	    print('\n'.join(formatted_lines))
	
	    if all(tb == 0 for tb in totals.values()):
	        print("\n[Warning] All tensor sizes are zero—did you fetch the map files correctly?")
	
	if __name__ == '__main__':
	    main()
	```
	
	File: /opt/GGUF-Tool-Suite/GGUF-Tool-Suite/models/DeepSeek-R1-0528/ppl_results.csv
	```
	QTYPE,blk.0.ffn_down.weight,blk.0.ffn_gate.weight,blk.0.ffn_up.weight,blk.1.ffn_down.weight,blk.1.ffn_gate.weight,blk.1.ffn_up.weight,blk.2.ffn_down.weight,blk.2.ffn_gate.weight,blk.2.ffn_up.weight,blk.3.ffn_down_exps.weight,blk.3.ffn_down_shexp.weight,blk.3.ffn_gate_exps.weight,blk.3.ffn_gate_shexp.weight,blk.3.ffn_up_exps.weight,blk.3.ffn_up_shexp.weight,blk.4.ffn_down_exps.weight,blk.4.ffn_down_shexp.weight,blk.4.ffn_gate_exps.weight,blk.4.ffn_gate_shexp.weight,blk.4.ffn_up_exps.weight,blk.4.ffn_up_shexp.weight,blk.5.ffn_down_exps.weight,blk.5.ffn_down_shexp.weight,blk.5.ffn_gate_exps.weight,blk.5.ffn_gate_shexp.weight,blk.5.ffn_up_exps.weight,blk.5.ffn_up_shexp.weight,blk.6.ffn_down_exps.weight,blk.6.ffn_down_shexp.weight,blk.6.ffn_gate_exps.weight,blk.6.ffn_gate_shexp.weight,blk.6.ffn_up_exps.weight,blk.6.ffn_up_shexp.weight,blk.7.ffn_down_exps.weight,blk.7.ffn_down_shexp.weight,blk.7.ffn_gate_exps.weight,blk.7.ffn_gate_shexp.weight,blk.7.ffn_up_exps.weight,blk.7.ffn_up_shexp.weight,blk.8.ffn_down_exps.weight,blk.8.ffn_down_shexp.weight,blk.8.ffn_gate_exps.weight,blk.8.ffn_gate_shexp.weight,blk.8.ffn_up_exps.weight,blk.8.ffn_up_shexp.weight,blk.9.ffn_down_exps.weight,blk.9.ffn_down_shexp.weight,blk.9.ffn_gate_exps.weight,blk.9.ffn_gate_shexp.weight,blk.9.ffn_up_exps.weight,blk.9.ffn_up_shexp.weight,blk.10.ffn_down_exps.weight,blk.10.ffn_down_shexp.weight,blk.10.ffn_gate_exps.weight,blk.10.ffn_gate_shexp.weight,blk.10.ffn_up_exps.weight,blk.10.ffn_up_shexp.weight,blk.11.ffn_down_exps.weight,blk.11.ffn_down_shexp.weight,blk.11.ffn_gate_exps.weight,blk.11.ffn_gate_shexp.weight,blk.11.ffn_up_exps.weight,blk.11.ffn_up_shexp.weight,blk.12.ffn_down_exps.weight,blk.12.ffn_down_shexp.weight,blk.12.ffn_gate_exps.weight,blk.12.ffn_gate_shexp.weight,blk.12.ffn_up_exps.weight,blk.12.ffn_up_shexp.weight,blk.13.ffn_down_exps.weight,blk.13.ffn_down_shexp.weight,blk.13.ffn_gate_exps.weight,blk.13.ffn_gate_shexp.weight,blk.13.ffn_up_exps.weight,blk.13.ffn_up_shexp.weight,blk.14.ffn_down_exps.weight,blk.14.ffn_down_shexp.weight,blk.14.ffn_gate_exps.weight,blk.14.ffn_gate_shexp.weight,blk.14.ffn_up_exps.weight,blk.14.ffn_up_shexp.weight,blk.15.ffn_down_exps.weight,blk.15.ffn_down_shexp.weight,blk.15.ffn_gate_exps.weight,blk.15.ffn_gate_shexp.weight,blk.15.ffn_up_exps.weight,blk.15.ffn_up_shexp.weight,blk.16.ffn_down_exps.weight,blk.16.ffn_down_shexp.weight,blk.16.ffn_gate_exps.weight,blk.16.ffn_gate_shexp.weight,blk.16.ffn_up_exps.weight,blk.16.ffn_up_shexp.weight,blk.17.ffn_down_exps.weight,blk.17.ffn_down_shexp.weight,blk.17.ffn_gate_exps.weight,blk.17.ffn_gate_shexp.weight,blk.17.ffn_up_exps.weight,blk.17.ffn_up_shexp.weight,blk.18.ffn_down_exps.weight,blk.18.ffn_down_shexp.weight,blk.18.ffn_gate_exps.weight,blk.18.ffn_gate_shexp.weight,blk.18.ffn_up_exps.weight,blk.18.ffn_up_shexp.weight,blk.19.ffn_down_exps.weight,blk.19.ffn_down_shexp.weight,blk.19.ffn_gate_exps.weight,blk.19.ffn_gate_shexp.weight,blk.19.ffn_up_exps.weight,blk.19.ffn_up_shexp.weight,blk.20.ffn_down_exps.weight,blk.20.ffn_down_shexp.weight,blk.20.ffn_gate_exps.weight,blk.20.ffn_gate_shexp.weight,blk.20.ffn_up_exps.weight,blk.20.ffn_up_shexp.weight,blk.21.ffn_down_exps.weight,blk.21.ffn_down_shexp.weight,blk.21.ffn_gate_exps.weight,blk.21.ffn_gate_shexp.weight,blk.21.ffn_up_exps.weight,blk.21.ffn_up_shexp.weight,blk.22.ffn_down_exps.weight,blk.22.ffn_down_shexp.weight,blk.22.ffn_gate_exps.weight,blk.22.ffn_gate_shexp.weight,blk.22.ffn_up_exps.weight,blk.22.ffn_up_shexp.weight,blk.23.ffn_down_exps.weight,blk.23.ffn_down_shexp.weight,blk.23.ffn_gate_exps.weight,blk.23.ffn_gate_shexp.weight,blk.23.ffn_up_exps.weight,blk.23.ffn_up_shexp.weight,blk.24.ffn_down_exps.weight,blk.24.ffn_down_shexp.weight,blk.24.ffn_gate_exps.weight,blk.24.ffn_gate_shexp.weight,blk.24.ffn_up_exps.weight,blk.24.ffn_up_shexp.weight,blk.25.ffn_down_exps.weight,blk.25.ffn_down_shexp.weight,blk.25.ffn_gate_exps.weight,blk.25.ffn_gate_shexp.weight,blk.25.ffn_up_exps.weight,blk.25.ffn_up_shexp.weight,blk.26.ffn_down_exps.weight,blk.26.ffn_down_shexp.weight,blk.26.ffn_gate_exps.weight,blk.26.ffn_gate_shexp.weight,blk.26.ffn_up_exps.weight,blk.26.ffn_up_shexp.weight,blk.27.ffn_down_exps.weight,blk.27.ffn_down_shexp.weight,blk.27.ffn_gate_exps.weight,blk.27.ffn_gate_shexp.weight,blk.27.ffn_up_exps.weight,blk.27.ffn_up_shexp.weight,blk.28.ffn_down_exps.weight,blk.28.ffn_down_shexp.weight,blk.28.ffn_gate_exps.weight,blk.28.ffn_gate_shexp.weight,blk.28.ffn_up_exps.weight,blk.28.ffn_up_shexp.weight,blk.29.ffn_down_exps.weight,blk.29.ffn_down_shexp.weight,blk.29.ffn_gate_exps.weight,blk.29.ffn_gate_shexp.weight,blk.29.ffn_up_exps.weight,blk.29.ffn_up_shexp.weight,blk.30.ffn_down_exps.weight,blk.30.ffn_down_shexp.weight,blk.30.ffn_gate_exps.weight,blk.30.ffn_gate_shexp.weight,blk.30.ffn_up_exps.weight,blk.30.ffn_up_shexp.weight,blk.31.ffn_down_exps.weight,blk.31.ffn_down_shexp.weight,blk.31.ffn_gate_exps.weight,blk.31.ffn_gate_shexp.weight,blk.31.ffn_up_exps.weight,blk.31.ffn_up_shexp.weight,blk.32.ffn_down_exps.weight,blk.32.ffn_down_shexp.weight,blk.32.ffn_gate_exps.weight,blk.32.ffn_gate_shexp.weight,blk.32.ffn_up_exps.weight,blk.32.ffn_up_shexp.weight,blk.33.ffn_down_exps.weight,blk.33.ffn_down_shexp.weight,blk.33.ffn_gate_exps.weight,blk.33.ffn_gate_shexp.weight,blk.33.ffn_up_exps.weight,blk.33.ffn_up_shexp.weight,blk.34.ffn_down_exps.weight,blk.34.ffn_down_shexp.weight,blk.34.ffn_gate_exps.weight,blk.34.ffn_gate_shexp.weight,blk.34.ffn_up_exps.weight,blk.34.ffn_up_shexp.weight,blk.35.ffn_down_exps.weight,blk.35.ffn_down_shexp.weight,blk.35.ffn_gate_exps.weight,blk.35.ffn_gate_shexp.weight,blk.35.ffn_up_exps.weight,blk.35.ffn_up_shexp.weight,blk.36.ffn_down_exps.weight,blk.36.ffn_down_shexp.weight,blk.36.ffn_gate_exps.weight,blk.36.ffn_gate_shexp.weight,blk.36.ffn_up_exps.weight,blk.36.ffn_up_shexp.weight,blk.37.ffn_down_exps.weight,blk.37.ffn_down_shexp.weight,blk.37.ffn_gate_exps.weight,blk.37.ffn_gate_shexp.weight,blk.37.ffn_up_exps.weight,blk.37.ffn_up_shexp.weight,blk.38.ffn_down_exps.weight,blk.38.ffn_down_shexp.weight,blk.38.ffn_gate_exps.weight,blk.38.ffn_gate_shexp.weight,blk.38.ffn_up_exps.weight,blk.38.ffn_up_shexp.weight,blk.39.ffn_down_exps.weight,blk.39.ffn_down_shexp.weight,blk.39.ffn_gate_exps.weight,blk.39.ffn_gate_shexp.weight,blk.39.ffn_up_exps.weight,blk.39.ffn_up_shexp.weight,blk.40.ffn_down_exps.weight,blk.40.ffn_down_shexp.weight,blk.40.ffn_gate_exps.weight,blk.40.ffn_gate_shexp.weight,blk.40.ffn_up_exps.weight,blk.40.ffn_up_shexp.weight,blk.41.ffn_down_exps.weight,blk.41.ffn_down_shexp.weight,blk.41.ffn_gate_exps.weight,blk.41.ffn_gate_shexp.weight,blk.41.ffn_up_exps.weight,blk.41.ffn_up_shexp.weight,blk.42.ffn_down_exps.weight,blk.42.ffn_down_shexp.weight,blk.42.ffn_gate_exps.weight,blk.42.ffn_gate_shexp.weight,blk.42.ffn_up_exps.weight,blk.42.ffn_up_shexp.weight,blk.43.ffn_down_exps.weight,blk.43.ffn_down_shexp.weight,blk.43.ffn_gate_exps.weight,blk.43.ffn_gate_shexp.weight,blk.43.ffn_up_exps.weight,blk.43.ffn_up_shexp.weight,blk.44.ffn_down_exps.weight,blk.44.ffn_down_shexp.weight,blk.44.ffn_gate_exps.weight,blk.44.ffn_gate_shexp.weight,blk.44.ffn_up_exps.weight,blk.44.ffn_up_shexp.weight,blk.45.ffn_down_exps.weight,blk.45.ffn_down_shexp.weight,blk.45.ffn_gate_exps.weight,blk.45.ffn_gate_shexp.weight,blk.45.ffn_up_exps.weight,blk.45.ffn_up_shexp.weight,blk.46.ffn_down_exps.weight,blk.46.ffn_down_shexp.weight,blk.46.ffn_gate_exps.weight,blk.46.ffn_gate_shexp.weight,blk.46.ffn_up_exps.weight,blk.46.ffn_up_shexp.weight,blk.47.ffn_down_exps.weight,blk.47.ffn_down_shexp.weight,blk.47.ffn_gate_exps.weight,blk.47.ffn_gate_shexp.weight,blk.47.ffn_up_exps.weight,blk.47.ffn_up_shexp.weight,blk.48.ffn_down_exps.weight,blk.48.ffn_down_shexp.weight,blk.48.ffn_gate_exps.weight,blk.48.ffn_gate_shexp.weight,blk.48.ffn_up_exps.weight,blk.48.ffn_up_shexp.weight,blk.49.ffn_down_exps.weight,blk.49.ffn_down_shexp.weight,blk.49.ffn_gate_exps.weight,blk.49.ffn_gate_shexp.weight,blk.49.ffn_up_exps.weight,blk.49.ffn_up_shexp.weight,blk.50.ffn_down_exps.weight,blk.50.ffn_down_shexp.weight,blk.50.ffn_gate_exps.weight,blk.50.ffn_gate_shexp.weight,blk.50.ffn_up_exps.weight,blk.50.ffn_up_shexp.weight,blk.51.ffn_down_exps.weight,blk.51.ffn_down_shexp.weight,blk.51.ffn_gate_exps.weight,blk.51.ffn_gate_shexp.weight,blk.51.ffn_up_exps.weight,blk.51.ffn_up_shexp.weight,blk.52.ffn_down_exps.weight,blk.52.ffn_down_shexp.weight,blk.52.ffn_gate_exps.weight,blk.52.ffn_gate_shexp.weight,blk.52.ffn_up_exps.weight,blk.52.ffn_up_shexp.weight,blk.53.ffn_down_exps.weight,blk.53.ffn_down_shexp.weight,blk.53.ffn_gate_exps.weight,blk.53.ffn_gate_shexp.weight,blk.53.ffn_up_exps.weight,blk.53.ffn_up_shexp.weight,blk.54.ffn_down_exps.weight,blk.54.ffn_down_shexp.weight,blk.54.ffn_gate_exps.weight,blk.54.ffn_gate_shexp.weight,blk.54.ffn_up_exps.weight,blk.54.ffn_up_shexp.weight,blk.55.ffn_down_exps.weight,blk.55.ffn_down_shexp.weight,blk.55.ffn_gate_exps.weight,blk.55.ffn_gate_shexp.weight,blk.55.ffn_up_exps.weight,blk.55.ffn_up_shexp.weight,blk.56.ffn_down_exps.weight,blk.56.ffn_down_shexp.weight,blk.56.ffn_gate_exps.weight,blk.56.ffn_gate_shexp.weight,blk.56.ffn_up_exps.weight,blk.56.ffn_up_shexp.weight,blk.57.ffn_down_exps.weight,blk.57.ffn_down_shexp.weight,blk.57.ffn_gate_exps.weight,blk.57.ffn_gate_shexp.weight,blk.57.ffn_up_exps.weight,blk.57.ffn_up_shexp.weight,blk.58.ffn_down_exps.weight,blk.58.ffn_down_shexp.weight,blk.58.ffn_gate_exps.weight,blk.58.ffn_gate_shexp.weight,blk.58.ffn_up_exps.weight,blk.58.ffn_up_shexp.weight,blk.59.ffn_down_exps.weight,blk.59.ffn_down_shexp.weight,blk.59.ffn_gate_exps.weight,blk.59.ffn_gate_shexp.weight,blk.59.ffn_up_exps.weight,blk.59.ffn_up_shexp.weight,blk.60.ffn_down_exps.weight,blk.60.ffn_down_shexp.weight,blk.60.ffn_gate_exps.weight,blk.60.ffn_gate_shexp.weight,blk.60.ffn_up_exps.weight,blk.60.ffn_up_shexp.weight,output.weight,token_embd.weight
	iq1_m_r4,3.2026,3.1999,3.1999,3.2006,3.2017,3.1995,3.1996,3.2033,3.2008,3.1998,3.2016,3.2008,3.1999,3.2022,3.2003,3.2000,3.2016,3.2014,3.2006,3.2004,3.2001,3.2015,3.2010,3.2025,3.2008,3.2032,3.2016,3.2010,3.2002,3.2008,3.1992,3.2022,3.2019,3.2013,3.1999,3.2004,3.2017,3.2009,3.1989,3.2002,3.2016,3.1997,3.2005,3.2025,3.2012,3.2002,3.1996,3.2018,3.2004,3.2021,3.2007,3.2038,3.2002,3.2016,3.1980,3.1997,3.2013,3.2046,3.2027,3.2016,3.2002,3.2023,3.2016,3.2059,3.2011,3.2030,3.2007,3.2037,3.2009,3.2005,3.1994,3.2031,3.2013,3.2025,3.2010,3.2048,3.2008,3.2022,3.2015,3.2026,3.2009,3.1992,3.2014,3.2028,3.2013,3.2044,3.2027,3.2023,3.2011,3.2032,3.2027,3.2019,3.2001,3.2011,3.2045,3.2015,3.2012,3.2034,3.2006,3.2047,3.1992,3.2054,3.2030,3.2035,3.2034,3.2045,3.2018,3.2047,3.2022,3.2016,3.1999,3.2063,3.2027,3.2044,3.2023,3.2033,3.2012,3.2083,3.2040,3.2029,3.2011,3.2057,3.2055,3.2076,3.2021,3.2051,3.1989,3.2034,3.2018,3.2082,3.2043,3.2006,3.2030,3.2041,3.2022,3.2097,3.2022,3.2070,3.2034,3.2070,3.2030,3.2086,3.2011,3.2048,3.2032,3.2046,3.2003,3.2081,3.2032,3.2023,3.2023,3.2060,3.2048,3.2087,3.2023,3.2059,3.2012,3.2071,3.2018,3.2069,3.2017,3.2068,3.2014,3.2041,3.2029,3.2083,3.2035,3.2070,3.2024,3.2043,3.2033,3.2071,3.2032,3.2061,3.2012,3.2034,3.2021,3.2062,3.2027,3.2081,3.2010,3.2046,3.2012,3.2160,3.2013,3.2055,3.2006,3.2081,3.2022,3.2158,3.2038,3.2080,3.2018,3.2080,3.2029,3.2122,3.2021,3.2077,3.2022,3.2078,3.2032,3.2170,3.2008,3.2101,3.2009,3.2105,3.2017,3.2155,3.2026,3.2081,3.2011,3.2074,3.2011,3.2130,3.2011,3.2079,3.2009,3.2088,3.2028,3.2111,3.2007,3.2081,3.2011,3.2088,3.2014,3.2173,3.2019,3.2099,3.2017,3.2090,3.2021,3.2166,3.2015,3.2088,3.2000,3.2088,3.2015,3.2198,3.2006,3.2124,3.2021,3.2055,3.2022,3.2139,3.2012,3.2097,3.2019,3.2096,3.2017,3.2106,3.2017,3.2107,3.2013,3.2111,3.2017,3.2207,3.2019,3.2128,3.2024,3.2127,3.2004,3.2144,3.2010,3.2101,3.2020,3.2094,3.2014,3.2117,3.2012,3.2145,3.2023,3.2093,3.2007,3.2130,3.2016,3.2123,3.2018,3.2119,3.2012,3.2172,3.2006,3.2144,3.2026,3.2106,3.2013,3.2135,3.2016,3.2128,3.2031,3.2084,3.2015,3.2092,3.2006,3.2106,3.2032,3.2129,3.2018,3.2114,3.2011,3.2107,3.2022,3.2105,3.2013,3.2069,3.2006,3.2108,3.2021,3.2083,3.2005,3.2103,3.2008,3.2105,3.2014,3.2107,3.2008,3.2087,3.2008,3.2120,3.2024,3.2090,3.2023,3.2096,3.1998,3.2121,3.2017,3.2112,3.2002,3.2063,3.2004,3.2081,3.2023,3.2062,3.2007,3.2075,3.2013,3.2088,3.2028,3.2082,3.1993,3.2055,3.2007,3.2071,3.2024,3.2052,3.2009,3.2037,3.1997,3.2081,3.1995,3.2044,3.1985,3.2077,3.2952,3.2099,3.2286,3.2059,3.2474,3.7143,3.2256
	```
	
	File: /opt/GGUF-Tool-Suite/GGUF-Tool-Suite/README.md
	```
	# GGUF Tool Suite
	
	**GGUF Tool Suite** is a set of flexible utilities that enables users to experiment with and create custom GGUF quantization blends. It simplifies the process of mixing quant formats (like `iq3_xxs`, `iq4_nl`, etc.) to:
	
	- Optimize performance  
	- Reduce model size  
	- Preserve accuracy across different hardware and use cases
	
	---
	
	## 🚀 MVP1 Release
	
	> This release is MVP1 and has only been tested on  
	> **[DeepSeek-R1-0528](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528)**.
	
	### ⚠️ Requirements
	
	- You **must use `ik_llama.cpp`** with [this patch](https://github.com/Thireus/ik_llama.cpp/commit/a66490410a366a9605234b94d67f3d9b7b389140) (compatibility with `llama.cpp` is **not guaranteed**).
	- Source code and Windows builds of `ik_llama.cpp` with pre-patched `GGML_MAX_CONTEXTS` and `ulimit` are available at:  
	  👉 https://github.com/Thireus/ik_llama.cpp  
	- Official repo:  
	  👉 https://github.com/ikawrakow/ik_llama.cpp  
	
	### 🧠 Important: Linux `ulimit` Patch
	
	Split models with a large number of files may **fail to load** unless you increase file descriptor limits.  
	Run the following command **before launching llama binaries**:
	
	```bash
	# Lifts "too many open files" limitation
	ulimit -n 99999
	```
	
	---
	
	## 📁 Recipe Examples
	
	Examples are included in the `recipe_examples` folder. Have a look at the file name or inside the recipe files to see the VRAM and RAM requirements of each.
	
	> ⚠️ You’re encouraged to build your own recipes tailored to your setup rather than relying on others'.
	
	---
	
	## 📥 Download Model Shards from a Recipe
	
	```bash
	git clone https://github.com/Thireus/GGUF-Tool-Suite
	cd GGUF-Tool-Suite
	mkdir -p kitchen && cd kitchen
	../quant_downloader.sh ../recipe_examples/DeepSeek-R1-0528.THIREUS-3.4064bpw-3.3372ppl.242GB-GGUF_11GB-GPU_231GB-CPU.254e1cf_c044584.recipe
	```
	
	> 💡 **Pro tip**: Re-running `quant_downloader.sh` in the same directory will only download the **missing/different shards** from your current quant mix.
	
	---
	
	## 🧠 Run a Downloaded Model (Example)
	
	```bash
	~/ik_llama-main-b3904-41a9c8a-bin-win-cuda-12.8-x64-avx512/llama-cli \
	  -m DeepSeek-R1-0528-THIREUS-BF16-SPECIAL_TENSOR-00001-of-01148.gguf \
	  -mla 3 -fa -amb 1024 -fmoe -ctk f16 -c 16384 -ngl 99 \
	  -ot "blk\.(3|4|5|6)\.ffn_.*=CUDA0" \
	  -ot "blk\.(7|8|9)\.ffn_.*=CUDA1" \
	  -ot "blk\.(10|11|12)\.ffn_.*=CUDA2" \
	  -ot exps=CPU -b 4096 -ub 4096 --warmup-batch --no-mmap --threads 36 \
	  --main-gpu 0 \
	  -p '<｜begin▁of▁sentence｜><｜User｜>What is the solution of x+5=-2?<｜Assistant｜><think>\n'
	```
	
	---
	
	## 🛠️ Generate a Custom Recipe for Your Config
	
	[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Thireus/GGUF-Tool-Suite/blob/main/quant_recipe_pipeline.ipynb)
	
	```bash
	python quant_assign.py ppl_results.csv \
	  --gpu-tensors '.*' \
	  --cpu-tensors 'blk\.([3-9]|[1-5][0-9]|60)\.ffn_down_exps\.weight' \
	                 'blk\.([3-9]|[1-5][0-9]|60)\.ffn_up_exps\.weight' \
	                 'blk\.([3-9]|[1-5][0-9]|60)\.ffn_gate_exps\.weight' \
	  --cpu-quants iq4_ks iq3_k iq2_k iq1_m_r4 \
	  --gpu-quants q8_0 iq5_k_r4 iq6_k \
	  --cpu-tensors-max-size 230 \
	  --gpu-tensors-max-size 95% \
	  --tolerance 0.01 \
	  --exponential-factor 8 \
	  --gpu-assign-qtype iq4_xs \
	  --gpu-assign-tensors 'blk\.([0-9]|[1-5][0-9]|60)\.attn_k_b\.weight=q8_0' \
	  | ./quants_regex_merger.sh \
	    --model-name "recipe_examples/DeepSeek-R1-0528" \
	    --add-ppl 0 \
	    --model-link "https://huggingface.co/deepseek-ai/DeepSeek-R1-0528"
	```
	
	> 🔧 **Adjust parameters** such as `--cpu-tensors-max-size` or `--gpu-quants` as needed for your specific hardware.
	
	---
	
	## 📊 About `ppl_results.csv`
	
	The file `ppl_results.csv` contains **individual tensor-level PPL benchmarks** for:
	
	- `Q8_0` (GPU tensors) + `IQ3-XXS` (CPU tensors)
	- Target model: **DeepSeek-R1-0528**
	- Quantization degradation reference: `IQ1-M-R4`
	
	This is the **core file** used to determine optimal quant mix strategies.  
	> ⚠️ Generating this CSV took **several days of GPU + CPU compute time**.
	
	- `IQ3-XXS` was chosen for CPU tensors as it fits within **256GB RAM**
	- Scripts used to generate:
	
	```bash
	./benchmark_each_tensor.sh --qtypes iq1_m_r4
	./collect_ppl_results.sh --chunks 250 --qtypes iq1_m_r4
	```
	
	📄 An article explaining the methodology is **coming soon**.
	
	---
	
	## 🙏 Acknowledgements
	
	Big thanks to **ubergarm** for his support and for providing the invaluable **`imatrix` files**.
	
	📄 Ubergarm's `imatrix` for DeepSeek-R1-0528 can be found here:  
	🔗 [imatrix_ubergarm.dat](https://huggingface.co/ubergarm/DeepSeek-R1-0528-GGUF/blob/main/imatrix-DeepSeek-R1-0528.dat)
	
	📄 Ubergarm's `imatrix` for DeepSeek-TNG-R1T2-Chimera can be found here:  
	🔗 [imatrix_r1t2_ubergarm.dat](https://huggingface.co/ubergarm/DeepSeek-TNG-R1T2-Chimera-GGUF/blob/main/imatrix-DeepSeek-TNG-R1T2-Chimera-Q8_0.dat)
	
	Also sincere thanks to **ikawrakow** and all **co-authors** of `ik_llama.cpp` for making this entire toolchain possible.
	
	---
	
	## 📜 License & Attribution
	
	Any **use, reproduction, or modification** of this software **must give clear and visible credit** to **Thireus** and the **GGUF Tool Suite**.  
	See the [LICENSE](./LICENSE) file for more details.
	
	🔗 https://gguf.thireus.com/
	```
```

---

#### 🗣️ Discussion

👤 **saood06** replied the **2025-07-17** at **01:19:14**:<br>

>I tried different quants and settings. The answer of Deepseek V3 or R1 is surprisingly stupid. Initially I thought its a bug in a RAM then I realized I have ECC ram. So ... its about 20k context prompt and Deepseek answers more or less the following absolute nonsense:

What sampler settings do you use?

> 👤 **magikRUKKOLA** replied the **2025-07-17** at **01:20:56**:<br>
> > What sampler settings do you use?
> 
> Unfortunately its irrelevant.  Try throwing the prompt into the original inference provider (the deepseek app or whatever).  The result will be the same -- nonsense related to the stupid equation.
> 
> the settings are default ones:
> 
> ```
> CUDA_VISIBLE_DEVICES="0,1,2" \
> /opt/ik_llama.cpp/ik_llama.cpp/build/bin/llama-server \
>     --model /opt/ubergarm/DeepSeek-R1-0528-GGUF/IQ2_K_R4/DeepSeek-R1-0528-IQ2_K_R4-00001-of-00005.gguf \
>     --alias ubergarm/DeepSeek-R1-0528-IQ2_K_R4-160k \
>     --seed 3407 \
>     -mla 3 -fa \
>     --ctx-size $((160 * 1024)) \
>     --temp 0.5 --top-k 0 --top-p 1.0 --min-p 0.1 --repeat-penalty 1.0 \
>     -ctk q8_0 \
>     -fmoe \
>     --n-gpu-layers 99 \
>     -b $((4 * 1024)) -ub $((2 * 1024)) \
>     -amb 512 \
>     --override-tensor exps=CPU \
>     --threads $(grep ^cpu\\scores /proc/cpuinfo | uniq | awk '{print $4}' | xargs -I{} echo "{}-0" | bc) \
>     --host 0.0.0.0 \
>     --port 8080 \
>     --lookup-cache-dynamic /mnt/data/ik_llama.kv.dump
> ```
> 
> 👤 **saood06** replied the **2025-07-17** at **01:29:34**:<br>
> >Try throwing the prompt into the original inference provider (the deepseek app or whatever). The result will be the same -- nonsense related to the stupid equation.
> 
> I did (used my choice of a third party API provider instead of the official one or my own quantized local instance), and saw the same result as you, but was able to fix it by removing the special tokens embedded in the prompt here:
> 
> `-p '<｜begin▁of▁sentence｜><｜User｜>What is the solution of x+5=-2?<｜Assistant｜><think>\n'`
> 
> Slightly mangled it to remove the special tokens to:
> 
> `<｜begin_of_sentence｜><|User|>What is the solution of x+5=-2?<|Assistant|>`
> 
> And now the output is relevant to the question.
> 
> I'm not sure it is a poisoned prompt, because I don't think you are supposed to use those tokens in the user prompt.
> 
> <details>
> <summary>Full reply</summary>
> <details>
> <summary>Thought process</summary>
> Okay, let's try to figure out how this quantization framework works. The user is asking if it selects specific sections of tensors that are most responsible for perplexity. They provided some code and data files, so I'll need to go through them to understand the process.
> 
> First, looking at the quant_assign.py script. It seems to be a Python script that assigns quantization types to different tensors based on their perplexity (PPL) values from the CSV file. The main idea is to optimize the model size while keeping the perplexity increase within a certain tolerance.
> 
> The script uses command-line arguments to specify which tensors go to CPU or GPU, the quantization types for each, and size constraints. For example, in the sample command, they're using --gpu-tensors and --cpu-tensors with regex patterns to categorize the tensors. Then they specify --cpu-quants and --gpu-quants to list the quantization methods for each.
> 
> In the code, the classify_tensors function uses regex patterns to group tensors into CPU and GPU classes. The main logic for assigning quants is in the compute_group_spreads and compute_quant_intervals functions. The spread is calculated based on how much each tensor's PPL deviates from the mean. The quant intervals are determined by stretching factors, which adjust how aggressively different quantization types are applied.
> 
> The optimize_midpoint_and_assign function seems to adjust the midpoint of PPL values and stretch factor to find a balance between model size and perplexity. It tries different stretch factors to meet the size constraints while minimizing the impact on PPL.
> 
> The CSV file ppl_results.csv contains PPL values for each tensor under different quantization types. The script likely uses these values to determine which quantization method to apply to each tensor. Tensors with higher PPL degradation would be assigned higher bitwidth quantizations to preserve accuracy, while those less sensitive could use lower bitwidths to save space.
> 
> The README mentions that the framework generates a recipe that specifies which quantization to use for each tensor. The quant_downloader.sh then downloads the necessary shards based on this recipe.
> 
> So, putting it all together: the framework analyzes the PPL impact of quantizing each tensor, categorizes them into CPU/GPU groups, and selects the most appropriate quantization type for each tensor within the size constraints. It prioritizes tensors that contribute more to perplexity by assigning them higher precision (less aggressive quantization) to minimize the overall impact on model performance.
> </details>
> 
> 
> The quantization framework uses a **spread-based assignment algorithm** that analyzes tensor-level perplexity (PPL) metrics to strategically select quantization formats while balancing model size and accuracy. Here's how it works:
> 
> ### 1. **Tensor Classification**
> - Uses regex patterns to categorize tensors into CPU/GPU groups:
>   ```python
>   --gpu-tensors 'blk\.([3-9]|[1-5][0-9]|60)\.ffn_down_shexp\.weight'
>   --cpu-tensors 'blk\.([3-9]|[1-5][0-9]|60)\.ffn_down_exps\.weight'
>   ```
> - Unmatched tensors default to GPU
> 
> ### 2. **Per-Tensor PPL Analysis**
> - Reads PPL degradation metrics from CSV (e.g., `iq1_m_r4` column):
>   ```csv
>   QTYPE,blk.0.ffn_down.weight,blk.0.ffn_gate.weight...
>   iq1_m_r4,3.2026,3.1999,3.1999...
>   ```
> 
> ### 3. **Spread Calculation**
> - Computes relative PPL impact using IQR-based outlier detection:
>   ```python
>   lower, upper = compute_iqr_bounds(class_vals, k_val)
>   spreads = compute_group_spreads(class_vals, forced_mid)
>   ```
> - Spread values range [-1,1] where:
>   - -1 = Most PPL-sensitive tensors
>   - +1 = Least PPL-sensitive tensors
> 
> ### 4. **Quantization Band Allocation**
> - Creates quantization bands using exponential stretching:
>   ```python
>   compute_quant_intervals(quants, stretch=1.0)
>   ```
>   Example output for stretch=8:
>   ```
>   Quant q8_0 @stretch=8.00: interval (0.8724, 1.0000]
>   Quant iq6_k @stretch=8.00: interval (0.7448, 0.8724]
>   ```
> 
> ### 5. **Optimal Assignment**
> - Uses dichotomic search to find the best quantization mix:
>   ```python
>   optimize_midpoint_and_assign(quants, class_vals, max_bytes)
>   ```
> - Balances between:
>   - Size constraints (`--cpu-tensors-max-size 230`)
>   - Tolerance thresholds (`--tolerance 0.01`)
>   - Exponential aggression (`--exponential-factor 8`)
> 
> ### Key Insight
> The framework **doesn't directly select "most responsible" sections**, but uses PPL spreads to:
> 1. Assign high-precision quants (like `q8_0`) to tensors with:
>    - Negative spreads (PPL degradation outliers)
>    - Left side of spread distribution
> 2. Assign aggressive quants (like `iq1_m_r4`) to:
>    - Positive spread tensors
>    - Right side of distribution
> 
> This creates a **PPL-aware quantization gradient** across model layers while respecting hardware constraints. The exponential factor controls how aggressively it pushes quantization toward less sensitive tensors.
> </details>
> 
> 👤 **magikRUKKOLA** replied the **2025-07-17** at **01:35:13**:<br>
> > I'm not sure it is a poisoned prompt, because I don't think you are supposed to use those tokens in the user prompt.
> 
> Uh oh!  You're absolutely right lol!  Thanks for the help!
> 
> Is there any option in ik_llama.cpp to automatically filter out the tokens from the prompt?  Or its supposed to be the job of the client?  I am talking about where exactly the user's input sanitation supposed to be done?
> 
> 👤 **Thireus** replied the **2025-07-17** at **01:46:48**:<br>
> Not a bad answer at all. Glad to see DeepSeek understands the code.
> 
> 👤 **saood06** replied the **2025-07-17** at **01:49:05**:<br>
> > > I'm not sure it is a poisoned prompt, because I don't think you are supposed to use those tokens in the user prompt.
> > 
> > Uh oh! You're absolutely right lol! Thanks for the help!
> 
> Glad to help.
> 
> > Is there any option in ik_llama.cpp to automatically filter out the tokens from the prompt? Or its supposed to be the job of the client? I am talking about where exactly the user's input sanitation supposed to be done?
> 
> Well I prefer and know more about the `/completion` endpoint where this would definitely be the responsibility of the client. 
> 
> The `/v1/chat/completions` endpoint on the other hand I'm not sure about. It being done here could make sense but from what you saw it is not doing that. This is meant to be an "OpenAI-compatible Chat Completions API" but the docs also say "no strong claims of compatibility with OpenAI API spec is being made", so I'm not sure whether this is intended behavior or not.
> 
> 👤 **saood06** replied the **2025-07-17** at **01:52:25**:<br>
> > Not a bad answer at all. Glad to see DeepSeek understands the code.
> 
> Glad to hear a review of the response from the author of the code. (Deepseek is still my go to model currently).
> 
> I understand what your code does, but I still want to go through it all to see the full implementation details.
> 
> 👤 **saood06** replied the **2025-07-17** at **02:04:06**:<br>
> @magikRUKKOLA 
> 
> I noticed you use `lookup-cache-dynamic`
> 
> I remember reading about this in some old PR's but I forgot about it, and never tested it. In your experience how much does it help?
> 
> 👤 **magikRUKKOLA** replied the **2025-07-17** at **02:12:09**:<br>
> @saood06 
> > I noticed you use `lookup-cache-dynamic`
> > 
> > I remember reading about this in some old PR's but I forgot about it, and never tested it. In your experience how much does it help?
> 
> Nope, it doesn't work in my case at all.  I looked up the code -- it doesn't seem to be implemented.
> I came from ktransformers and noticed that they implemented the on-storage prefix caching -- https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/prefix_cache.md (which works great so all the prefills are saved).  So I thought the similar option exists here but no, it doesn't.  I left that option as a reminder to myself to figure this out later on, sorry for a confusion. :)
> 
> 👤 **saood06** replied the **2025-07-17** at **02:22:43**:<br>
> >So I thought the similar option exists here but no, it doesn't. I left that option as a reminder to myself to figure this out later on, sorry for a confusion. :)
> 
> Similar options do exist. You can set a save path with: `--slot-save-path /mnt/sda/slotKVcache/`. 
> 
> Make saves with:
> ```
> curl --header "Content-Type: application/json" \
>    --request POST \
>    --data '{"filename":"test.bin"}' [...]:8080/slots/0?action=save
> ```
> 
> Restore is similar:
> ```
> curl --header "Content-Type: application/json" \
>    --request POST \
>    --data '{"filename":"test.bin"}' [...]:8080/slots/0?action=restore
> ```
> 
> One of my long term goals has been to improve the way the KV cache is used in the server (implementing a trie based solution that should offer similar benefits to PagedAttention). But a more medium term (and what should be much simpler to do) goal that I had was to automatically load saved prompts into the cache whenever they would save computations.
> 
> Edit:
> This does not work with models that use MLA and are using the `-mla 2` flag (and there is almost no reason to use that over  `-mla 3` or `-mla 1`). For more info see: https://github.com/ikawrakow/ik_llama.cpp/pull/473#discussion_r2116130404
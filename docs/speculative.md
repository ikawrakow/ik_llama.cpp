# Speculative Decoding

llama.cpp supports speculative decoding, a technique that can significantly accelerate token generation by predicting multiple tokens ahead of the main model.

[Speculative decoding](https://en.wikipedia.org/wiki/Transformer_(deep_learning)#Speculative_decoding) leverages the fact that computing n tokens in a batch (as in prompt processing) is more efficient than computing n sequentially (as in response generation). By generating draft tokens quickly and then verifying them with the target model in a single batch, this approach can achieve substantial speedups when the draft predictions are frequently correct.

## Implementations

The `llama-server` application supports several implementations of speculative decoding. An implementation with draft model can be mixed with an implementation without draft model.

### Draft Model (`draft`)

A much smaller model (called the _draft model_) generates drafts.
A draft model is the most used approach in speculative decoding.

### n-gram Cache (`ngram-cache`)

An n-gram is a sequence of n tokens. The n-gram cache implementation maintains statistics about short n-gram sequences.
A draft is computed using probabilities derived from these statistics. External statistics can also be loaded from files for improved accuracy.

See:

- #5479, #6828, #6848

### n-gram Map (`ngram-simple`, `ngram-map-*`)

These implementations search the token history for patterns and use matching sequences as draft candidates.
They require no additional model but rely on patterns that have already appeared in the generated text.
An example to use this approach can be the rewriting of source code by a LLM.

#### n-gram Map (`ngram-simple`)

This implementation looks for the last n-gram in history that matches the current n-gram and creates a draft using the m tokens following the matched n-gram. It is the simplest self-speculative approach with minimal overhead.

```
llama-server [...] --spec-type ngram-simple:n_max=64
```

#### n-gram Map Key (`ngram-map-k`)

This implementation looks for the current n-gram of size n (called the _key_) in the token history. If the key n-gram is followed by the same m tokens (called the _mgram_) multiple times, it creates a draft using these m tokens. This approach requires a minimum number of occurrences (stage key `ngram_min_hits`, default is 1) before generating drafts.

The number of accepted tokens is stored for each used n-gram.

**Example:**
```
llama-server [...] --spec-type ngram-map-k:n_max=64,ngram_min_hits=1
```

#### n-gram Map Key-4-Values (`ngram-map-k4v`)

This experimental implementation looks for the current n-gram of size n (called the _key_) in the token history. For each key, up to four _values_ (n-grams of size m, called _mgrams_) are tracked. An internal statistic counts the occurrences of each mgram after the key n-gram. If one mgram is significantly more frequent than the others, it is used as the draft.

The number of accepted tokens is stored for each used n-gram.

**Example:** Server options to be used if there are a lot of longer repetitions.
```
llama-server [...] --spec-type ngram-map-k4v:n_max=64,ngram_size_n=8,ngram_size_m=8,ngram_min_hits=2
```

### n-gram Mod (`ngram-mod`)

Add basic ngram hasher for speculative decoding:

- For each ngram, compute a hash using LCG
- For each computed hash, store the next token
- During speculation, iteratively compute the rolling hash of the last n tokens and pick the next token from the storage

Some characteristics:

- Lightweight (~16 MB)
- Constant memory and complexity
- Can generate variable draft lengths (i.e. m is not fixed)

Currently, a single hash pool is shared across all server slots, so different requests can benefit from each other.

**Sample usage:**

```
# notes:
# - small `n` are not recommended
# - MoEs require long drafts
# - dense models: can reduce `n_min` and `n_max`

llama-server ... --spec-type ngram-mod:n_max=64,n_min=48,ngram_size_n=24
```

Applications:

- Iterating over a block of text/code (e.g. in llama.vim)
- Reasoning models (when they have to repeat their thinking in the final answer)
- Summarization

Example Video:

- See #19164

### Differences between ngram-simple, ngram-map and ngram-mod

- ngram-simple looks for a previous matching n-gram and inserts the following m-gram.
- ngram-map-k looks for a previous matching n-gram and inserts the following m-gram but uses an internal hash-map of n-grams in the current context window.
- ngram-mod uses a hash pool which is shared across all server slots. The hash pool is a map from n-gram hash to the next token (not the next m-gram as in ngram-map).

## Command-Line Options

The canonical startup surface is repeated `--spec-type SPEC[:k=v,...]`. Legacy `--spec-stage`, `--draft-*`, `--spec-ngram-*`, `--suffix-*`, and `-mtp` flags are rejected with replacement guidance.

### `--spec-type SPEC[:k=v,...]`

Each `--spec-type` entry defines one speculative stage. Repeat it to configure the supported two-stage path.

| Type | Description |
|------|-------------|
| `none` | No speculative decoding |
| `draft` | Draft-model speculative decoding; pair with `-md/--model-draft` |
| `mtp` | Embedded or assistant-backed MTP |
| `ngram-cache` | Use n-gram cache lookup |
| `ngram-simple` | Use simple n-gram pattern matching |
| `ngram-map-k` | Use n-gram pattern matching with n-gram keys |
| `ngram-map-k4v` | Use n-gram pattern matching with n-gram keys and up to four m-gram values |
| `ngram-mod` | Use the shared n-gram hasher |
| `suffix` | Use suffix-tree speculative decoding |

Canonical stage keys:

| Key | Meaning |
|-----|---------|
| `n_max` | Maximum drafted tokens for that stage |
| `n_min` | Minimum usable drafted tokens for that stage |
| `p_min` | Minimum speculative probability threshold |
| `heads` | MTP heads to use; `1` is the default, while values above `1` and `0` (all model heads) are experimental |
| `ngram_size_n` | Lookup n-gram size |
| `ngram_size_m` | Draft m-gram size |
| `ngram_min_hits` | Minimum matching hits for n-gram map stages |
| `suffix_min_match_len` | Minimum suffix context match length |
| `suffix_max_depth` | Maximum suffix-tree depth |
| `suffix_corpus` | Optional suffix corpus file for pre-warming |

String-valued stage keys such as `suffix_corpus` need shell-safe quoting when the value contains commas. From a normal shell, quote the value inside the stage payload so the parser sees the comma as part of the string value.

Example shell-safe form:

```bash
./llama-server [...] \
    --spec-type "suffix:n_max=16,n_min=2,suffix_min_match_len=5,suffix_max_depth=64,suffix_corpus='/tmp/spec,type-corpus.json'"
```

If you are constructing `argv` directly without shell unescaping, the parser also accepts escaped commas as `\,`.

Examples:

```bash
# Single-stage MTP
./llama-server [...] --spec-type mtp:n_max=1,p_min=0.0

# Single-stage ngram-mod
./llama-server [...] --spec-type ngram-mod:n_max=64,n_min=48,ngram_size_n=24

# Draft-model speculation
./llama-server [...] --model-draft draft.gguf --spec-type draft:n_max=4,p_min=0.0

# Two-stage self-spec -> MTP fallback
./llama-server [...] \
    --spec-type ngram-mod:n_max=64,n_min=2,ngram_size_n=8 \
    --spec-type mtp:n_max=1,p_min=0.0

# Suffix stage with pre-warmed corpus
./llama-server [...] \
    --spec-type suffix:n_max=16,n_min=2,suffix_min_match_len=5,suffix_max_depth=64,suffix_corpus=/path/to/corpus.json

# Suffix stage with a comma-bearing corpus path from a normal shell
./llama-server [...] \
    --spec-type "suffix:n_max=16,n_min=2,suffix_min_match_len=5,suffix_max_depth=64,suffix_corpus='/tmp/spec,type-corpus.json'"
```

### `--spec-autotune`

Autotunes the active stage parameters and reports the best configuration back as a canonical `--spec-type ...` snippet.

## Statistics
Each speculative decoding implementation prints statistics.

```
draft acceptance rate = 0.57576 (  171 accepted /   297 generated)
statistics ngram_simple: #calls = 15, #gen drafts = 5, #acc drafts = 5, #gen tokens = 187, #acc tokens = 73
statistics draft: #calls = 10, #gen drafts = 10, #acc drafts = 10, #gen tokens = 110, #acc tokens = 98
```

```
draft acceptance rate = 0.70312 (   90 accepted /   128 generated)
statistics ngram_mod: #calls = 810, #gen drafts = 15, #acc drafts = 15, #gen tokens = 960, #acc tokens = 730, dur(b,g,a) = 0.149, 0.347, 0.005 ms
```

```
statistics ngram_map_k: #calls(b,g,a) = 6 1690 26, #gen drafts = 26, #acc drafts = 26, #gen tokens = 1248, #acc tokens = 968, dur(b,g,a) = 2.234, 1.427, 0.016 ms
```


- `#calls(b,g,a)`: number of calls of begin (new prompt), generation and accumulation of this implementations
- `#gen drafts`: number of drafts generated by this implementation
- `#acc drafts`: number of drafts accepted (partially) by the main model
- `#gen tokens`: number of tokens generated by this implementation (including rejected tokens)
- `#acc tokens`: number of tokens accepted by the main model
- `dur(b,g,a): durations of begin (new prompt), generation and accumulation (process acceptance).

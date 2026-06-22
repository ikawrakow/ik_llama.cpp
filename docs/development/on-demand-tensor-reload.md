# On-Demand Tensor Reload

## Overview

This patch introduces **selective tensor hot-swapping** for `ik_llama.cpp` models, now with full support for `graph`/`layer` split mode.
It allows individual tensors (or groups of tensors) to be reloaded from their original on-disk GGUF files **without tearing down the process, the `llama_model`, or the `llama_context`**. Tensors may reside on any backend—GPU, CPU, or split across multiple GPUs—and the reload logic preserves that placement.

This is primarily intended for:

* Iterative experimentation and LoRA-like surgical updates.
* Dynamic MoE (Mixture-of-Experts) expert swapping.
* **Mixed-quantization perplexity benchmarks**, where the bulk of a model lives in one quant (e.g., Q4_X) on GPU while individual experts are hot-swapped one-by-one into a different quant (e.g., IQ1_KT) to measure isolated quality impact.

---

## Motivation

Standard `ik_llama.cpp` workflows require restarting the entire executable to pick up new weights. For large models distributed across multiple GPUs—or models that spill into CPU memory—this incurs significant downtime. This patch solves that by:

1. **Tracking provenance**: At load time, every tensor is mapped back to its source GGUF shard, byte offset, and modification time.
2. **Detecting changes**: At runtime, it cheaply `stat()`s the source files to see if a tensor’s backing data has changed.
3. **Surgical replacement**: Only the changed tensors are re-mapped/re-allocated. The rest of the model stays resident in GPU/CPU memory.
4. **Graph safety**: Cached CUDA graphs are invalidated and the context’s cached compute graphs (`ctx->prev` / `ctx->prev_mtp`) are reset so that the next evaluation rebuilds the graph with the new buffer pointers, sizes, or types.

---

## High-Level Architecture

The patch adds a `reload_info` registry to `llama_model` (defined in `src/llama-reload-info.h`). The lifecycle has five phases:

### 1. Registration Phase (`llama_model_load`)
During model loading, every weight that is successfully mapped gets an entry in `model.reload->tensor_reload_sources` **only when the environment variable `LLAMA_HOTSWAP_ENABLED` is set**:

```cpp
struct tensor_reload_source {
    std::string   path;          // Absolute path to the GGUF shard
    size_t        data_offset;   // Byte offset of the tensor data in the file
    size_t        nbytes;        // Current byte size
    int64_t       last_mtime;    // Last modification time (seconds)
    int64_t       last_mtime_ns; // Nanosecond precision on Linux

    // Snapshots of the *original* loaded state so we can reattach later
    ggml_backend_buffer_t original_buffer;
    void                * original_data;
    ggml_type             original_type;
    int64_t               original_ne[GGML_MAX_DIMS];
    size_t                original_nb[GGML_MAX_DIMS];
    ggml_split_tensor_t * original_extra;
    std::vector<split_info> original_splits;
    std::vector<std::string> sibling_names; // MoE siblings
    reload_state          state;
};
```

### 2. Snapshot Phase (`snapshot_all_reload_tensors`)
The first time a reload is requested, an **eager snapshot** is taken of every registered tensor and its MoE siblings. This captures the original buffer handles, split descriptors, and strides. This snapshot is essential for:

* **Reattachment**: If a tensor was detached to a private buffer because it grew, but later shrinks back to its original size/type, it can be reattached to the original shared buffer, avoiding memory fragmentation.
* **MoE consistency**: MoE layers often have three sibling tensors (`ffn_down_exps`, `ffn_up_exps`, `ffn_gate_exps`) that must share the same split topology across GPUs.

### 3. Detection Phase (`reload_changed_tensors`)
When the user (or the server health-check loop) calls `llama_reload_changed_tensors()`:

1. It iterates over the registry and `stat()`s each source file.
2. If `mtime` (or `mtime_ns`) differs, it re-parses the GGUF header (`gguf_find_tensor_meta`) to get the new `offset`, `nbytes`, `ggml_type`, and on-disk shape (`ne`).
3. **Shape verification**: If the on-disk dimensions differ from the model tensor (`file_ne[i] != tensor->ne[i]`), the tensor is skipped entirely; the reload logic refuses to change logical shapes.
4. It builds a **sorted job list**: tensors that are **returning to their original snapshot** are processed first. This maximizes the chance of freeing private buffers before allocating new ones, reducing memory pressure.

### 4. Reload Phase (`reload_tensor`)
For each changed tensor, the patch performs a careful in-place update.

#### 0. Shape Verification
Before any metadata or buffer changes, the code verifies that the on-disk `ne[0..3]` exactly match the current model tensor. If any dimension differs, the reload is aborted with a log message and the tensor is left untouched.

#### A. Returning Check
The first decision is whether the tensor's new on-disk type matches its **original** snapshot type (`curr_type == src.original_type`).

* **Returning to original**: The tensor is reattached to its original shared buffer and original split descriptors. Any private buffer allocated during a previous reload is freed (only if the tensor's state is `DETACHED` or `FALLBACK_CPU`). State becomes `ON_ORIGINAL`.
* **Changed**: Proceed to metadata update and buffer reallocation.

#### B. Metadata Update & Block-Size Alignment
If the tensor’s `ggml_type` changed (e.g., Q4_X → IQ1_KT), the main tensor descriptor and all its split descriptors are updated with new `type` and `nb` values. The logical shape (`ne`) is guaranteed unchanged by the preceding shape verification. However, for fused/multi-GPU splits the per-device boundaries must be recalculated.

**Critical constraint for fused/multi-GPU splits:**  
Different quants use different block sizes:
* **Q4_X / Q4_0**: block size **32**
* **IQ1_KT**: block size **256**

When a tensor changes between these types, `apply_tensor_type_change()` re-rounds every GPU slice’s `ne[0]` to the nearest multiple of the new block size. If this redistribution is not propagated to all siblings in the same MoE layer, the CUDA split backend dispatches rows to the wrong devices and **matmul fails**.

#### C. Buffer Lifecycle
The patch tracks each tensor with a `reload_state` enum (`UNINITIALIZED`, `ON_ORIGINAL`, `DETACHED`, `FALLBACK_CPU`). Buffers are only freed if the state is not `ON_ORIGINAL`, ensuring shared original buffers are never corrupted.

| Scenario | Action |
|----------|--------|
| Returning to original snapshot | **Reattach** to `original_buffer`, restore original splits, free old private buffer if any. |
| Changed type/size while previously on original | **Detach** from the shared buffer to a newly allocated private buffer so the shared region isn’t corrupted for other tensors. |
| Changed type/size while already detached | Free old private buffer, allocate new one. |
| Allocation fails on target backend | **CPU fallback**: allocate on `ggml_backend_cpu_buffer_type()` and clear split metadata. State becomes `FALLBACK_CPU`. |

#### D. Split Tensor (Multi-GPU) Handling
For split tensors, the patch:
- Recomputes per-device bounds using the new block-size alignment.
- Reallocates per-device split buffers if necessary.
- **Resyncs MoE siblings**: If `ffn_down_exps` changes its split topology, `ffn_up_exps` and `ffn_gate_exps` in the same layer are forced to adopt identical per-device `ne[0]` distributions and strides. This is required by the CUDA split-backend contract.

#### E. Data Copy
Finally, the tensor bytes are read from the updated file and copied into the (possibly new) backend buffer via `ggml_backend_tensor_set`.

---

## Hybrid CPU/GPU Inference

When running with `--split-mode layer --fit --gpu-layers 99` (or any configuration where the model does not fully fit in VRAM), some tensors naturally land in CPU memory. The hot-swap system fully supports this:

* **CPU tensors are reloadable**: The reload logic reads the new data from disk and copies it into the CPU backend buffer exactly as it would for CUDA buffers.
* **Fallback allocator**: If a GPU buffer allocation fails during a reload (e.g., because an IQ1_KT expert is larger than the original Q4_X expert), the system automatically falls back to a CPU buffer for that tensor.

This allows you to keep, for example, 90 % of an MoE model on 13 GPUs while a few large expert tensors cycle through CPU RAM, or to benchmark quants that vary in size per-expert without worrying about exact VRAM fitting.

---

## API & Environment Variables

### Public C API
```cpp
// include/llama.h
LLAMA_API bool llama_reload_changed_tensors(struct llama_context * ctx);
```

Returns `true` if at least one tensor was reloaded. When this happens, the function also resets the context’s cached compute graphs (`ctx->prev` and `ctx->prev_mtp`) so that the next evaluation performs a full graph rebuild with the new tensor pointers.

### Environment Variables

| Variable | Purpose |
|----------|---------|
| `LLAMA_HOTSWAP_ENABLED` | Enables the hot-swap loop in `perplexity` and the health-check hook in `server`. |
| `LLAMA_PERPLEXITY_PRE_RELOAD_SCRIPT` | Path to an executable script run between perplexity iterations (e.g., to regenerate/re-quantize a tensor file). |

---

## Integration Points

### `examples/perplexity/perplexity.cpp`
When `LLAMA_HOTSWAP_ENABLED` is set, the tool runs in a loop:

1. Perform an initial `llama_reload_changed_tensors()` to apply any pending changes before the first evaluation.
2. Compute perplexity (or Hellaswag, etc.).
3. Print timings and write logs.
4. Execute the optional pre-reload script.
5. Call `llama_reload_changed_tensors(ctx)`. If no tensors changed, exit; otherwise repeat from step 2.

### `examples/server/server.cpp`
On every health-check (`/health`) request, if `LLAMA_HOTSWAP_ENABLED` is set, the server calls `llama_reload_changed_tensors()`. This provides a convenient, external trigger: simply `touch` or overwrite a tensor’s source GGUF file and poll `/health` to apply the change.

---

## MoE Sibling Resync

MoE weights are often stored as three separate tensors that must be split identically across GPUs. The patch automatically detects these families by suffix:

- `.ffn_down_exps.weight`
- `.ffn_up_exps.weight`
- `.ffn_gate_exps.weight`

When one member of the family is reloaded and its per-device split dimensions change—especially when crossing quant types with different block sizes (Q4_X=32 vs IQ1_KT=256)—`resync_moe_sibling_splits()` is invoked. The logic follows these steps:

1. **Fast path**: If the reference tensor is returning to its original snapshot, the siblings are also reattached to their original snapshots via `reattach_split_tensor_to_shared()`—no data movement is required.
2. **Phase A – Detach**: Siblings are detached from shared buffers (freeing only non-original buffers) and new main handles are allocated. Split tensors receive a dummy `data` pointer because the split backend uses `extra->splits`.
3. **Phase B – Propagate dimensions**: The reference tensor’s per-device `ne[0]` distribution is copied to the siblings, and strides (`nb[]`) are recomputed using a temporary `ggml_context`. This step is mandatory because the valid split boundaries depend on the quantization block size.
4. **Phase C – Allocate GPU splits**: New per-device GPU buffers are allocated for each sibling split.
5. **Phase D – CPU fallback (if needed)**: If any GPU allocation fails, the **entire** sibling group is moved to CPU buffers to maintain consistency.
6. **Phase E – Write back**: The original sibling data (which has not changed, only the layout) is written back into the new buffers via `ggml_backend_tensor_set`.

---

## Buffer Lifecycle Details

### Reattachment to Shared Buffers
If a tensor was originally loaded in a large shared GGUF buffer alongside other tensors, and it was previously detached because it grew, the patch attempts to **reattach** it when it returns to its original size and type. This is done by restoring:

- `tensor->buffer = original_buffer`
- `tensor->data   = original_data`
- `tensor->extra  = original_extra` (restoring all split descriptors)

This prevents unbounded memory growth during iterative experiments where tensors oscillate between two states.

### State Machine
Because `ggml` does not provide native reference counting on buffers, the patch uses a per-tensor state machine to avoid corrupting shared allocations:

* `ON_ORIGINAL`: The tensor still lives in its initial shared buffer. This buffer is **never** freed during reload.
* `DETACHED`: The tensor was moved to a privately allocated buffer. This buffer **is** freed before the next reload.
* `FALLBACK_CPU`: The tensor was moved to CPU memory after a GPU allocation failure.

Only buffers belonging to tensors in the `DETACHED` or `FALLBACK_CPU` states are released, ensuring that shared original buffers remain valid for all other tensors that still reference them.

---

## Limitations & Safety Notes

1. **File path stability**: The source file must remain at the same path. Renaming or removing shards will cause `stat()` or `open()` to fail.
2. **No locking**: There is no file-locking protocol. The user must ensure the GGUF file is not being written to while `ik_llama.cpp` is reading it.
3. **Graph rebuild cost**: While cheaper than a full process restart, rebuilding the CUDA graph (or CPU graph) incurs a one-time latency spike after a reload.
4. **Platform specifics**: Nanosecond mtime checks use `st_mtim.tv_nsec` and are guarded by `#ifdef __linux__`.
5. **Thread safety**: `llama_reload_changed_tensors` is **not** thread-safe with active inference. Ensure the context is idle before calling (the perplexity example naturally guarantees this; the server example only invokes it during the synchronous `/health` handler).

---

## Usage Example: Per-Expert Quantization Sweep (Q4_X ↔ IQ1_KT)

This example benchmarks a massive MoE model where the base weights are **Q4_X**. The tool iteratively replaces individual `ffn_down_exps.weight` tensors with **IQ1_KT** equivalents to measure the isolated perplexity impact of each expert's quantization level.

A sanity check is embedded in the source directory: one of the "IQ1_KT" shard files is actually the original **Q4_X** tensor. When the rotation reaches that slot, the reloaded tensor is byte-for-byte identical to the baseline, so the PPL must match exactly—confirming that the hot-swap machinery introduces no loss.

### 1. Helper script (`tensor-swap.sh`)
Place the rotation script in your model directory (e.g., `/opt/THIREUS/Kimi-K2.6/Q4_X/`). It maintains `.bak` files so that each iteration restores the previous tensor before installing the next candidate.

```bash
#!/bin/bash
set -euo pipefail

TARGET_GLOB="*Q4_X*gguf"
SOURCE_DIR="../smol-IQ1-KT-mist.bin"
TENSOR_NAME_PATTERN="blk\.[0-9]+\.ffn_down_exps\.weight"

# ... (see full script in patch) ...
```

The script scans for target files matching `*Q4_X*gguf` containing `blk.[N].ffn_down_exps.weight`, then pulls replacements from `../smol-IQ1-KT-mist.bin/` by matching the `SPECIAL_TENSOR-NNNN-of-XXXX.gguf` shard number.

### 2. Launch perplexity with hot-swap enabled

```bash
ulimit -n 9999
ulimit -l unlimited

export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7,8,9,10,11,12"
export LLAMA_HOTSWAP_ENABLED=1
export LLAMA_PERPLEXITY_PRE_RELOAD_SCRIPT=./tensor-swap.sh
export LLAMA_DEBUG=1

# --offload-policy -1,off \

GGML_CUDA_NO_PINNED=1 \
/opt/ik_llama.cpp/ik_llama.cpp/build/bin/llama-perplexity \
    --chunks 8 \
    -f /opt/ik_llama.cpp/wiki.test.raw \
    --model /opt/THIREUS/Kimi-K2.6/Q4_X/Kimi-K2.6-THIREUS-Q4_X-SPECIAL_TENSOR-00001-of-01097.gguf \
    --alias THIREUS/Kimi-K2.6-Q4_X.bin \
    -b 512 -ub 512 \
    --ctx-size 512 \
    --fit \
    --fit-margin 4200 \
    --gpu-fit-margin 0,4400,12,4400 \
    --temp 0.0 --top-k 0 --top-p 1.0 \
    -ctk f16 \
    -ctv q8_0 \
    -amb 128 \
    -mea 128 \
    -wgt 1 \
    --mlock \
    --split-mode layer \
    --graph-reduce-type f16 \
    --threads $(grep ^cpu\\scores /proc/cpuinfo | uniq | awk '{print $4}' | xargs -I{} echo "{}-0" | bc) \
    -sas \
    --gpu-layers 99 \
    --no-offload-only-active-experts \
    --host 0.0.0.0 \
    --port 8080 \
    --log-enable \
    --logdir /var/log/ \
    --jinja \
    --special \
    --prompt-cache "$HOME/.cache/ik_llama.cpp/prompt-cache.bin" --prompt-cache-all \
    --slot-save-path "$HOME/.cache/ik_llama.cpp/slot.bin" \
    --lookup-cache-dynamic "$HOME/.cache/ik_llama.cpp/slot.bin" \
    --keep -1 \
    --slot-prompt-similarity 0.35 \
    --metrics \
    -cuda fusion=1
```

### 3. What happens

1. The model loads with **Q4_X** weights distributed across 13 GPUs using layer splitting.
2. The first pass computes the baseline perplexity over 8 chunks.
3. `tensor-swap.sh` runs between iterations:
   * Restores the previously swapped tensor from `.bak` to its original Q4_X state.
   * Copies the next IQ1_KT expert shard into place.
4. `llama_reload_changed_tensors()` detects the `mtime` changes, re-parses the GGUF headers, and reloads the affected `ffn_down_exps.weight` tensor(s).
   * The restored tensor **returns to its original Q4_X snapshot** and reattaches to its shared buffer.
   * The newly swapped tensor is loaded into a private buffer with the new IQ1_KT data.
   * Because Q4_X and IQ1_KT have different block sizes (32 vs 256), the split backend redistributes per-device boundaries and resyncs the MoE siblings (`ffn_up_exps` and `ffn_gate_exps`) to the same layout.
5. The CUDA graphs are invalidated and the next perplexity iteration begins.
6. When the rotation hits the sanity-check slot (where the source file is actually the original Q4_X tensor), the perplexity returns to the exact baseline value, confirming the reload is lossless.

### 4. Expected behavior

```text
snapshot_all_reload_tensors: eager snapshot of all reload tensors + siblings
perplexity: calculating perplexity over 8 chunks, n_ctx=512, batch_size=512, n_seq=1
[1]1.0622,[2]1.2068,[3]1.2327,[4]1.1873,[5]1.1487,[6]1.1283,[7]1.1214,[8]1.1109,
Final estimate: PPL = 1.1109

main: executing pre-reload script: ./tensor-swap.sh
main: [pre-reload] Swapped index 0 (tensor #00918)
reloaded tensor 'blk.1.ffn_down_exps.weight'

perplexity: calculating perplexity over 8 chunks ...
Final estimate: PPL = 1.1105

main: executing pre-reload script: ./tensor-swap.sh
main: [pre-reload] Restored index 0. Advancing to index 1.
main: [pre-reload] Swapped index 1 (tensor #00921)
reloaded tensor 'blk.1.ffn_down_exps.weight'
reloaded tensor 'blk.2.ffn_down_exps.weight'

perplexity: calculating perplexity over 8 chunks ...
Final estimate: PPL = 1.1080
```

Notice that when the script restores a tensor to its original Q4_X shard, the reload reattaches it to the shared buffer with zero copy. When the sanity-check slot is reached, the PPL returns to the exact baseline, proving the mechanism is sound.

---

## Summary of Changed Files

| File | Change |
|------|--------|
| `examples/perplexity/perplexity.cpp` | Hot-swap loop + pre-reload script execution. |
| `examples/server/server.cpp` | Trigger reload on `/health` when env var is set. |
| `ggml/include/ggml-cuda.h` | Add `ggml_backend_cuda_invalidate_graphs()`. |
| `ggml/include/ggml.h` | Conditional `GGML_MAX_SRC` override. |
| `ggml/src/CMakeLists.txt` | Propagate `GGML_MAX_SRC` compile definition. |
| `ggml/src/ggml-cuda.cu` | Implement graph invalidation; debug prints for split tensors. |
| `ggml/src/ggml.c` | Debug print in `ggml_mul_mat_id` for shape mismatches. |
| `include/llama.h` | Declare `llama_reload_changed_tensors()`. |
| `src/llama-mmap.cpp/h` | Expose `llama_file::get_path()` so reload registry knows the source file path. |
| `src/llama-model.h` | Add `std::unique_ptr<reload_info> reload` to `llama_model`. |
| `src/llama-reload-info.h` | **New.** Defines `tensor_reload_source`, `reload_state`, and `reload_info` registry. |
| `src/llama-reload.cpp` | **New.** Core implementation: GGUF header parser, snapshot, reload, MoE resync, buffer management, CPU fallback, shape verification. |
| `src/llama.cpp` | Wire reload registry into `llama_model_load`; reset cached compute graphs (`ctx->prev` / `ctx->prev_mtp`) on reload; export C API. |
| `src/CMakeLists.txt` | Propagate `GGML_MAX_SRC` compile definition. |

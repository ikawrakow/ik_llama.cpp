# CUDA Bit-Exact Quantization for GGUF

> Adaptation note (Oct 2026, onto Joel's PR 2572 `d9e2868` lead): production
> dispatch is Joel's single `ggml_cuda_quantize(device,type,...,nslice,imatrix)`
> entry in `ggml/src/ggml-cuda/kt-encoder.cu` (KT + legacy). Legacy order:
> **Q8_0, Q6_0, Q5_0, Q4_0, Q5_1, Q4_1, IQ4_NL, IQ4_XS**; Q5_0/Q4_0 sections are
> bannered for easy removal once Ikawrakow confirms. **Q6_0 OLS is KEPT** (make_qx + fudge,
> weight=x*x without imatrix) to match HEAD CPU `quantize_q6_0`; the old plain
> max/-32 path is removed. All plain/imatrix scales store `FP16(fudge*d)`
> (Q8_0 uses the HEAD `Q6_0`-fudge quirk to stay bit-exact). Q8_0 ignores
> imatrix (CPU does); symmetric Q4_0 stays on CPU. `FP_CONTRACT OFF` guards in
> `ggml-quants.c` keep CPU/GPU bit-exact. The `custom_pre_ols` scope note below
> is historical (Aug research base); HEAD OLS is kept.

## 1. Goal

while producing **byte-for-byte identical** GGUF tensor payloads to the CPU
reference quantizers. `Q8_0`, `Q4_0`, `Q5_0` and `Q6_0` are implemented and enabled with
`--cuda-quantize` (off by default); every other quant type in the target set is
*provably bit-exact* under the legacy (non-OLS) algorithms and can be added
behind the same entry.

This branch (`custom_pre_ols`) is scoped so that the GGUF-facing reference
quantizers do **not** use the OLS `d = sumqx/sumq2` scale refinement. OLS is
retained only for the KV-cache CUDA kernels (which never touch a GGUF), as
agreed.

## 2. Bit-exactness model

A quantized block is byte-for-byte portable across backends if and only if
every stored byte depends only on:

1. **max / min reductions** — exact and order-independent, so any parallel
   reduction (any thread count, any tree shape) produces the identical scalar;
2. **per-element rounding** — `t = f(x_i)` applied to a single value, with a
   fixed, backend-independent rounding mode (truncate / round-half-up /
   round-half-even / round-half-away);
3. **FP16 conversion** — `GGML_FP32_TO_FP16` is round-to-nearest-even
   (`_cvtss_sh` on MSVC, software nearest-even elsewhere); CUDA `__float2half_rn`
   matches bit-for-bit.

There must be **no cross-element floating-point accumulation** in the stored
values (no `sumqx/sumq2`, no `sums` used as scales). Under the legacy
algorithms this holds for the seven "simple block" quants.

## 3. Target set

### 3.1 Bit-exact legacy block quants

All block sizes are 32 values. `d`/`m` are derived only from max/min. The CPU
reference line numbers are from `ggml/src/ggml-quants.c` on `custom_pre_ols`.

| Type | CPU scale (`_ref`) | CPU quant idiom | Stored fields |
|------|--------------------|-----------------|---------------|
| `Q4_0` ✓ | `amax = max\|v\|; d = max/-8; id = d?1/d:0` (`:673`) | `(int8_t)(x*id + 8.5f)`, clamp `MIN(15,…)` | `d`(fp16), 16 nibbles `qs[j] = xi0 \| xi1<<4` |
| `Q4_1` | `min/max; d=(max-min)/15; m=min` (`:716`) | `(int8_t)((x-min)*id + 0.5f)`, clamp `MIN(15,…)` | `d,m`(fp16), nibbles |
| `Q5_0` ✓ | `amax; d=max/-16` (`:757`) | `(int8_t)(x*id + 16.5f)`, clamp `MIN(31,…)` | `d`(fp16), nibbles, `qh`(32b) |
| `Q5_1` | `min/max; d=(max-min)/31; m=min` (`:805`) | `(uint8_t)((x-min)*id + 0.5f)` (no clamp) | `d,m`(fp16), nibbles, `qh`(32b) |
| `Q6_0` ✓ | `amax; d=max/-32` (`:853`) | `(int8_t)(x*id + 32.5f)`, clamp `MIN(63,…)` | `d`(fp16), nibbles, `qh`(8B) |
Not implemented for now : | `Q6_1` | `min/max; d=(max-min)/63; m=min` (`:897`) | `(int)((x-min)*id + 0.5f)`, clamp `MIN(63,…)` | `d,m`(fp16), nibbles, `qh`(8B) |
| `Q8_0` ✓ | `amax; d=amax/127; id=d?1/d:0` (`:943`) | `roundf(x*id)` (round-half-away) | `d`(fp16), 32 int8 |

✓ = implemented on CUDA. `Q4_0`, `Q5_0` and `Q6_0` with an importance matrix (the
`make_qx_quants` path, `quantize_row_q4_0_impl` / `quantize_row_q5_0_impl` /
`quantize_row_q6_0_impl`) are likewise CUDA-implemented, see §4.4. `Q8_0`'s
importance-matrix path is a weighted least-squares scale
(`quantize_row_q8_0_impl`), see §4.5.

Round-trip rule: the CPU `(int8_t)/(int)(f)` cast **truncates toward zero**;
CUDA `(signed char)/(int)(f)` truncates identically. `roundf()` in both C and
CUDA is round-half-away-from-zero. FP16 is round-to-nearest-even on both sides.

### 3.2 Non-target types (stay CPU)

- `Q2_K/Q3_K/Q4_K/Q5_K/Q6_K`: scale selection uses `make_qkx2_quants` grid
  searches with float comparison tie-breaking tied to a fixed scan order; not
  provably backend-stable without replicating that order.
- `IQ1_S…IQ4_KT`: trellis/IQ search with `best_scale = sumqx/sumq2`
  (`ggml-quants.c` ~`13262+`). Same reason. Future work if exactness can be
  pinned to a single canonical scan order.
- `Q4_0` fast path: the standalone `quantize_row_q4_0` still dispatches to
  `iqk_quantize_q4_0` whose stored scale is `d = sumqx/sumq2`
  (`iqk_quantize.cpp:849`). That is not the GGUF-chunk entry: the fork's
  `quantize_q4_0` (`ggml-quants.c:3509`), used by `ggml_quantize_chunk`, already
  calls the legacy `quantize_row_q4_0_ref` (no weights) or
  `quantize_row_q4_0_impl` (with weights), which is what the CUDA set matches.

## 4. Kernel design

### 4.1 Flat block tiling

The F32 tensor is a row-major buffer of `nrows * n_per_row` floats with
`n_per_row % 32 == 0`, so the per-row segmentation into 32-value quant blocks
equals the flat segmentation. A quant block is just `[ib*32, (ib+1)*32)` over
the flat buffer, `ib = 0 … nblocks-1`. No row bookkeeping is needed; this also
makes row splitting across multiple GPUs trivially safe.

### 4.2 Warp-per-quant-block

- `blockDim.x = 32` (one warp), `gridDim.x = nblocks`.
- **Phase A (scale):** lane `l` loads `x[ib*32 + l]`; warp-reduce
  `fmaxf(|x|, …)` via `__shfl_xor_sync` (exact, order-independent). All lanes
  recompute `d`/`id`/`m` from the reduced scalar with the exact CPU arithmetic
  (`max/-8`, `(max-min)/15`, …). Lane 0 stores `d`,`m` with `__float2half_rn`.
- **Phase B (quant):** each lane rounds its own element with the type's exact
  idiom (`t = __fmul_rn(x*id)` then `(int32_t)(t + 8.5f / 16.5f)`, truncation
  toward zero like the CPU `(int8_t)/(int)` cast; `__fmul_rn` forces a single
  rounding so a `-use_fast_math` build cannot FMA-contract the product+add).
  Nibble/high-bit packing uses the per-element values held across the warp:
  - `Q4_0/Q4_1`: lane `j<16` fetches lane `j+16`'s quant via
    `__shfl_xor_sync(...,16)` and writes `qs[j] = q_j | q_{j+16}<<4`.
  - `Q5_0/Q5_1`: same nibble exchange; the `qh` 5-th-bit bitmap is assembled
    with `__ballot_sync(..., (q>>4)&1)` over the full warp (executed uniformly
    by every lane — calling it from `if (lane==0)` is undefined) so bit `e` is
    set iff element `e`'s 5-th bit is set, then stored little-endian by lane 0
    (the ref memcpys a native `uint32`).
  - `Q6_0/Q6_1 (Q6_1 is not implemented)`: `qs[j]` as above; `qh[k] = hi2(xi[k]) | hi2(xi[k+16])<<2 |
    hi2(xi[k+8])<<4 | hi2(xi[k+24])<<6` with `hi2(e)=xi[e]>>4` (CPU `:887-888`).
  - `Q8_0`: lane `l` writes `qs[l] = (int8_t)roundf(x[ib*32+l]*id)` directly —
    no packing step.

### 4.3 Determinism

- max/min reduce: exact, any order → same scalar.
- division `1/d`, `max/-8`, `(max-min)/15`: IEEE-correctly-rounded scalar ops,
  identical on CPU and GPU.
- rounding and FP16: per-element, matched semantics (§2, §3.1).
- No cross-lane FP accumulation anywhere.

### 4.4 `Q4_0` / `Q5_0` / `Q6_0` with importance matrix (`make_qx_quants`)

The imatrix quantizers (`quantize_row_q4_0_impl` `:3429`,
`quantize_row_q5_0_impl` `:3577`, `quantize_row_q6_0_impl` `:3698`) are
sequential algorithms — a per-block greedy `make_qx_quants` search — so they
cannot use the warp-per-block tiling of §4.2. Instead:

- **One thread per quant block** replays `make_qx_quants` (`ggml-quants.c:1786`)
  in the exact CPU order, including the greedy `-(nmax + 0.1*is)/max` iscale
  sweep and the coordinate-descent loop. Every float op is a correctly-rounded
  intrinsic (`__fdiv_rn` / `__fmul_rn` / `__fadd_rn` / `__fsqrt_rn`, nearest-int
  via the 2^23 + 2^22 magic), so the `-use_fast_math` build cannot change a
  single bit.
- **Row `sigma2` is host-computed.** `sum_x2 += x[j]*x[j]` is order-dependent,
  so it is summed on the host in the exact CPU summation order and uploaded once
  per tensor; the kernel computes only the per-block weight
  `qw[j]*sqrtf(sigma2 + x[j]^2)`.
- **Chunked driver with a per-chunk `base`.** Like §4.1 the tensor is processed
  in `1<<20`-block device chunks; the kernel maps block `gb = base + ib` to row
  `gb/blocks_per_row` and weight block `gb % blocks_per_row`, so the
  row/weight indexing stays correct even when a chunk boundary is not a multiple
  of `blocks_per_row`.
- **Byte-exact fp16.** `GGML_FP32_TO_FP16` here resolves to the bit-twiddle
  `ggml_compute_fp32_to_fp16` (`ggml-impl.h:595`, no `__F16C__` in this build),
  ported as `fp32_to_fp16_ggml` and stored via `__ushort_as_half` — a
  `(ggml_half)` cast would run the bit pattern through `__half(float)` and
  corrupt the value. Degenerate blocks can yield a NaN scale; see §7.
- **CPU must not FMA-contract.** `make_qx_quants`, `quantize_row_q4_0_impl`,
  `quantize_row_q5_0_impl` and `quantize_row_q6_0_impl` are compiled with
  `#pragma STDC FP_CONTRACT OFF` in `ggml-quants.c`; a `/arch:AVX2` release
  build otherwise contracts `sumlx += w*x*l` (in `make_qx_quants`) and the
  per-block weight's `sigma2 + x[j]*x[j]` (in the `_impl` functions), drifting
  ~1 ulp from the GPU.
- `Q5_0` additionally uses `nmax == 16` (vs 8 for `Q4_0`) and, after the
  optimizer, packs each `L`'s 5-th bit into the `qh` bitmap exactly like
  `quantize_row_q5_0_impl`; the device kernel and the host driver mirror the
  `Q4_0` imatrix shapes with `QK5_0` / `sizeof(block_q5_0)`.
- `Q6_0` uses `nmax == 32`, so the optimizer emits `L` in `[0, 63]` (6-bit) and
  the packing writes two `qh` bits per element — `qh[k] = h_k | h_{k+8}<<4` with
  `h_e = (L_e>>4) | (L_{e+16}>>4)<<2` — mirroring `quantize_row_q6_0_impl`. The
`_impl` early-returns to `quantize_row_q6_0_ref` when no `quant_weights` are
   given (matching `Q4_0`/`Q5_0`), so the plain kernel stays the §4.2 warp port;
   the imatrix driver mirrors the `Q4_0`/`Q5_0` shapes with `QK6_0` /
   `sizeof(block_q6_0)`.

### 4.5 `Q8_0` with importance matrix (weighted least-squares)

`Q8_0` cannot use `make_qx_quants`: its codes are signed `int8` that fit a
full 8 bits, but `make_qx_quants` shifts by `nmax` (`q = l + nmax`) and for
`nmax == 128` an `int8_t` would overflow (and the offset encoding would not
reproduce the reference's sign). `quantize_row_q8_0_impl` (`ggml-quants.c`)
therefore keeps the plain `roundf(x*id)` codes and refines **only the scale**
with the weighted least-squares

```
d  = sumqx/sumq2,   sumqx = sum w[j] * q[j] * x[j],   sumq2 = sum w[j] * q[j]^2
w[j] = qw[j] * sqrtf(sigma2 + x[j]^2)     (same weight convention as §4.4)
```

This is the fork's earlier Q8_0 "OLS scale refinement"
(`d = sumqx/sumq2`, weight `v^2` in the reverted commit) re-weighted by the
importance matrix. The device kernel `quantize_q8_0_imatrix_kernel` replays it
one thread per quant block, with the LS products/sums expressed as
`__fmul_rn`/`__fadd_rn` (and `__fsqrt_rn`, `__fdiv_rn`) so a `-use_fast_math`
build cannot FMA-contract them; the row `sigma2` is again host-computed in the
exact CPU order. `quantize_q8_0_impl` is guarded by `#pragma STDC FP_CONTRACT OFF`
and early-returns to `quantize_row_q8_0_ref` when no `quant_weights` are given
(plain warp kernel unchanged). `d` may be NaN for degenerate blocks; see §7.

## 5. Integration into `llama-quantize`

Implemented as follows:

1. New CUDA translation unit `ggml/src/ggml-cuda/quantize_gguf.cu`
   (+ `.cuh`), auto-picked by the `ggml-cuda/*.cu` GLOB in
   `ggml/src/CMakeLists.txt`.
2. Public host entry exposed via `ggml-cuda.h` under `GGML_USE_CUDA`:
   `ggml_cuda_quantize_q8_0(const float * src, void * dst, int64_t nrows, int64_t n_per_row)`.
   It owns a device context (device 0 for the POC), allocates device buffers,
   copies in, launches, copies out, frees, and checks every CUDA return code
   (returning `0` on failure so the caller aborts). Returns
   `nrows * ggml_row_size(Q8_0, n_per_row)`.
3. `llama_model_quantize_params` gains `bool cuda_quantize` (default false),
   set by the `--cuda-quantize` CLI flag in `examples/quantize/quantize.cpp`
   (ignored with a warning in non-CUDA builds).
4. `do_quantize()`: when `params->cuda_quantize` and the new type is eligible
   (`Q8_0`, `Q5_0` and `Q6_0` always; `Q4_0` unless `--symmetric-q4-0` is
   requested), routes each `ne[2]` (expert) slice through the CUDA entry and
   skips the CPU chunk loop. The imatrix slice follows the CPU convention
   `imatrix + i02*ne[0]` (one weight per column, reused for every row of the
   expert). All other types (and non-eligible `Q4_0`) fall back to the CPU
   `ggml_quantize_chunk` path untouched.
5. When eligibility holds the per-tensor progress log reads
   `converts to %s ..` instead of `converting to %s ..`, so a `--cuda-quantize`
   run can be told apart from the CPU path in the output.
6. `ggml_validate_row_data` still runs on the CUDA output (it must pass, since
   the bytes equal the CPU bytes).

## 6. Validation harness

`examples/unit_test_cuda` (target `unit_test_cuda`) verifies byte-for-byte
parity. It compares three producers of Q8_0 GGUF bytes on identical input:

1. **GPU:** `ggml_cuda_quantize_q8_0` (`ggml/src/ggml-cuda/quantize_gguf.cu`);
2. **CPU:** `ggml_quantize_chunk` — the fork's real llama-quantize path
   (`quantize_q8_0 -> quantize_row_q8_0_ref`);
3. **REF:** a local copy of `quantize_row_q8_0_ref` (`ggml-quants.c:943`) with
   the fp16 step done by `__float2half_rn`, no ggml internals.

`gpu vs cpu` answers "is the kernel byte-exact?", `cpu vs ref` answers "is the
fork's CPU path still the vanilla reference?", `gpu vs ref` ties the two
together with an independent baseline. The harness is spec-driven: a
`quant_spec` table lists each covered type (currently `Q8_0`, `Q4_0`, `Q5_0`,
`Q6_0`) with its `QK`, block size, CUDA entry, local `ref_*` copy and fill set,
so adding a type is one table entry. Fills are `random-uniform`, `weight-like` (90%
of 32-value blocks tight around zero, 10% wide), crafted `edge-cases`
(all-zero, single outlier, ±max, exact `.5` rounding ties, denormals,
huge/tiny magnitudes, mixed signs), a `q5-boundary` fill that drives the
5-th-bit / `qh` bitmap and the `x*id + 16.5f` truncation ties, and a
`q6-boundary` fill that likewise drives `Q6_0`'s `qh` and its `x*id + 32.5f`
truncation ties; a `> 1<<20`-block tensor exercises the CUDA host wrapper's
chunk loop. `test_slices`
reproduces `do_quantize`'s `ne[2]` slicing exactly: the CUDA and CPU branches
both quantize one expert slice at a time into consecutive slots, and both must
equal a single whole-tensor call.

`q4_0-imatrix`, `q5_0-imatrix`, `q6_0-imatrix` and `q8_0-imatrix` add imatrix
specs: the local
`ref_*` copies replay the imatrix quantizer (
`make_qx_quants` + `quantize_row_q4_0_impl`/`quantize_row_q5_0_impl`/
`quantize_row_q6_0_impl`, or the weighted-LS `quantize_row_q8_0_impl` from
`ggml-quants.c`) with the fp16 step done by a host port of
`ggml_compute_fp32_to_fp16`'s bit-twiddle, and a synthetic importance matrix is
generated with one weight per column, reused for every row (the CPU convention).
Their scales can legitimately
be NaN for degenerate blocks, so these specs set `nan_d_equal`: two fp16 `d`
values that are both NaN compare equal even if their sign/payload differs (see
§7). Every finite block must still match byte-for-byte.

End-to-end: `llama-quantize` twice on a small F32 model (`Q8_0`), once CPU,
once `--cuda-quantize`; `memcmp` the resulting tensor payloads.

Build: `cmake --build build --target unit_test_cuda -j`. Run:
`unit_test_cuda [--seed N] [--device N] [--all-devices] [--big] [--huge] [--quick]`.

## 7. Known gaps / follow-ups

- ~~**`Q4_0` dispatch:** switch `quantize_row_q4_0` to the legacy
  `quantize_row_q4_0_ref`, then add the CUDA `Q4_0` kernel.~~ Done: the fork's
  GGUF-chunk path `quantize_q4_0` already calls the legacy ref / `_impl` and the
  CUDA `Q4_0` kernel is in, including the importance-matrix path (`§4.4`).
  `Q4_0` stays on the CPU only when `--symmetric-q4-0` is requested.
- Add `Q4_1, Q5_1 (Q6_1 is not implemented)` kernels behind the same entry, one commit
  per type, each validated by the harness. (`Q5_0`, `Q6_0` and their imatrix paths, and the `Q8_0` imatrix path, are done.)
- Multi-GPU: rows are independent; later split row ranges across the 3 devices
  and add pinned-memory async upload. Does not affect bit-exactness.
- Reproducibility statement: CUDA kernels must never introduce an FMA or a
  reduction into a stored scale; the validation harness is the guard.
- **NaN sign (accepted, not a porting gap):** the imatrix scale can be NaN for
  degenerate blocks (e.g. huge `|x|` makes `x*x` overflow to `inf`, so the
  per-block weights become `0*inf`/`inf` and `sumlx/suml2` is NaN). x86 SSE sets
  the sign of such a NaN from the operand signs (renders `-NaN`, fp16 `0xfe00`),
  whereas NVIDIA always emits the default quiet NaN (`+NaN`, `0x7e00`). This is a
  CPU-vendor semantic difference — even CPU-only llama.cpp is not NaN-sign-stable
  across CPU vendors — so the harness treats any NaN `d` as equal (`nan_d_equal`)
  and the imatrix blocks at issue only ever differ by that single bit.

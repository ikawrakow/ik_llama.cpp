//
// Copyright (C) 2023-2024 The ggml authors
// Copyright (C) 2026 Nexesenex
// MIT license
// SPDX-License-Identifier: MIT
//
// Bit-exact CUDA quantization of legacy block quants for GGUF, dispatched
// through Joel's single entry ggml_cuda_quantize() (kt-encoder.cu lead).
// Logical order after Joel's KT: Q8_0, Q6_0, Q5_0, Q4_0. Q5_0 and Q4_0
// sections are bannered so they can be removed with minimal work once
// Ikawrakow confirms (delete their kernels + helpers + dispatcher cases).
//
// Q8_0 target: quantize_q8_0 (ggml/src/ggml-quants.c:3681) via
// quantize_row_q8_0_ref (ggml/src/ggml-quants.c:911):
//   amax = max(|x_j|) over the 32-value block
//   d    = fudge*amax/127      (__fdiv_rn: exact under -use_fast_math)
//   id   = d ? 1/d : 0         (__fdiv_rn: exact under -use_fast_math)
//   q_j  = roundf(x_j*id)      // round half away from zero
//   d    = FP16(fudge*d)       // round to nearest even
//   NOTE: HEAD CPU uses ggml_get_quantize_fudge_factor(GGML_TYPE_Q6_0) for
//   Q8_0 (ggml-quants.c:915, copy-paste quirk). CUDA matches it exactly so
//   the bytes stay identical; fix the CPU quirk separately if desired.
//   Q8_0 ignores imatrix on the CPU, so CUDA also ignores it.
//
// Q6_0 OLS is KEPT (not reverted): quantize_q6_0 (ggml-quants.c:3668) always
// runs quantize_row_q6_0_impl via make_qx_quants with fudge:
//   weight = x*x without imatrix, qw*sqrt(sigma2+x*x) with imatrix
//   d = make_qx_quants(QK6_0, 32, xb, L, 1, weight) * fudge
//   FP16(d*fudge). The old plain max/-32 kernel is NOT used for GGUF.
//
// Q4_0 target: quantize_q4_0 without imatrix uses quantize_row_q4_0_ref
// (ggml/src/ggml-quants.c:673):
//   max  = signed value with max |x| (first occurrence = |x| ties)
//   d    = fudge*max/-8        (__fdiv_rn: exact under -use_fast_math)
//   id   = d ? 1/d : 0         (__fdiv_rn: exact under -use_fast_math)
//   q_j  = MIN(15, (int8_t)(x_j*id + 8.5f))  // truncation toward zero
//   d    = FP16(fudge*d)
//   byte j (0..15) = low nibble q_j | high nibble q_{j+16} << 4
//   Symmetric Q4_0 (--symmetric-q4-0, d=amax/7, no fudge) stays on CPU.
//
// Q5_0 target: quantize_q5_0 without imatrix uses quantize_row_q5_0_ref:
//   max  = signed value with max |x| (first occurrence wins |x| ties)
//   d    = fudge*max/-16       (__fdiv_rn: exact under -use_fast_math)
//   id   = d ? 1/d : 0         (__fdiv_rn: exact under -use_fast_math)
//   q_j  = MIN(31, (int8_t)(x_j*id + 16.5f))  // truncation toward zero
//   d    = FP16(fudge*d)
//   byte j (0..15) = low nibble q_j | high nibble q_{j+16} << 4
//   5-th bit of q_j and q_{j+16} -> qh bit j and bit j+16 (4-byte LE uint32)
//
// Q4_0 / Q5_0 / Q6_0 with importance matrix: quantize_row_q4_0_impl,
// quantize_row_q5_0_impl, quantize_row_q6_0_impl. Each 32-value block is
// quantized by make_qx_quants, a deterministic sequential greedy optimizer.
// One thread per block replays it in the exact CPU order with
// correctly-rounded intrinsics, so it is byte-identical too, with
// FP16(d*fudge). The row-level sigma2 sum is order-dependent, so it is
// pre-computed on the host in the exact CPU summation order.
//
//
// Q5_0 also stores a 4-byte qh bitmap that carries each quant's 5-th bit.
// Like qs/d it is an exact, order-independent function of the block (bit e =
// (q_e >> 4) & 1), so it is assembled with __ballot_sync over the warp and
// stored little-endian, reproducing the reference's `memcpy(&qh, 4)` bytes.
//
// The 32-value blocks tile the tensor row-major buffer contiguously
// (n_per_row % 32 == 0), so rows need no explicit bookkeeping. Each warp
// quantizes one block independently; the max reduction via shuffles is exact
// and order-independent, the argmax tie-break (lowest index wins) matches the
// reference's sequential scan, and the per-element rounding is
// backend-deterministic. The result is byte-for-byte identical to the
// quantize_row_*_ref implementations on any GPU.
//
// Both public entry points share one chunked host driver (fixed ~128 MiB F32
// device chunks) so single large tensors never need a large contiguous VRAM
// allocation; every CUDA call is checked, and on failure the error is printed
// and 0 returned (the caller aborts the quantization).

#include "quantize_gguf.cuh"

#include <cinttypes>
#include <cstdio>
#include <cmath>
#include <algorithm>
#include <vector>

// ---------------------------------------------------------------------------
// kernels
// ---------------------------------------------------------------------------

static __global__ void quantize_q8_0_kernel(
        const float * __restrict__ x, void * __restrict__ vy, const int64_t nblocks, const float fudge) {
    const int32_t lane = threadIdx.x; // 0 .. 31 == QK8_0

    // grid-stride over quant blocks (nblocks can exceed UINT_MAX on big models)
    for (int64_t ib = blockIdx.x; ib < nblocks; ib += gridDim.x) {
        const float xi = x[ib*QK8_0 + lane];

        // exact, order-independent max reduction (max is associative/exact)
        float amax = fabsf(xi);
#pragma unroll
        for (int m = 16; m > 0; m >>= 1) {
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, m));
        }

        // __fdiv_rn: correctly-rounded IEEE division. The build uses
        // -use_fast_math, which makes plain '/' approximate (reciprocal +
        // multiply); the CPU ref divides with exact rounding, and an ulp
        // difference in d/id flips roundf() exactly on k+0.5 ties.
        const float d  = __fmul_rn(fudge, __fdiv_rn(amax, 127.0f));
        const float id = d ? __fdiv_rn(1.0f, d) : 0.0f;

        block_q8_0 * y = (block_q8_0 *)vy;
        y[ib].qs[lane] = (int8_t)roundf(xi*id);

        if (lane == 0) {
            // store the __half directly. Do NOT round-trip through
            // __half_as_ushort + assignment: block_q8_0.d is __half, and
            // `half = unsigned short` converts the ushort as a *number*
            // (half(float(ushort))), corrupting the scale bits (e.g. 0x29f2
            // -> 0x713e). Assigning the __half copies the raw bits.
            y[ib].d = __float2half_rn(d);
        }
    }
}

static __global__ void quantize_q4_0_kernel(
        const float * __restrict__ x, void * __restrict__ vy, const int64_t nblocks, const float fudge) {
    const int32_t lane = threadIdx.x; // 0 .. 31 == QK4_0

    for (int64_t ib = blockIdx.x; ib < nblocks; ib += gridDim.x) {
        const float xi = x[ib*QK4_0 + lane];

        // argmax of |x|. On |x| ties the *first* (lowest index) element wins,
        // exactly like the ref's sequential scan: quantize_row_q4_0_ref keeps
        // the signed value `max` of the first max-magnitude element, which
        // fixes the sign of d.
        float   bval = fabsf(xi);
        int32_t bidx = lane;
#pragma unroll
        for (int m = 16; m > 0; m >>= 1) {
            const float   oval = __shfl_xor_sync(0xffffffffu, bval, m);
            const int32_t oidx = __shfl_xor_sync(0xffffffffu, bidx, m);
            if (oval > bval || (oval == bval && oidx < bidx)) {
                bval = oval;
                bidx = oidx;
            }
        }

        // signed value of the argmax element, broadcast to the warp
        const float max = __shfl_sync(0xffffffffu, xi, bidx);

        // __fdiv_rn: correctly-rounded IEEE division (see Q8_0 kernel). max/-8
        // is a power-of-2 division (exact anyway); 1/d is the general case.
        // CPU stores FP16(fudge*d), and codes use id=1/(fudge*d).
        const float d  = __fmul_rn(fudge, __fdiv_rn(max, -8.0f));
        const float id = d ? __fdiv_rn(1.0f, d) : 0.0f;

        // MIN(15, (int8_t)(x_j*id + 8.5f)): truncation toward zero, then clamp.
        // |x_j*id| <= 8 so x_j*id + 8.5 is in [-0.5, 16.5] and the truncation
        // is always representable (matches the ref's (int8_t) cast exactly).
        // __fmul_rn forces the multiply to round once: with -use_fast_math nvcc
        // otherwise contracts `xi*id + 8.5f` into one FMA (single rounding)
        // while the CPU rounds the product and the add separately, and that
        // 1-ulp difference flips the truncation at integer thresholds.
        const float   t = __fmul_rn(xi, id);
        const int32_t v = (int32_t)(t + 8.5f);
        const int32_t q = v > 15 ? 15 : v;

        block_q4_0 * y = (block_q4_0 *)vy;
        if (lane == 0) {
            y[ib].d = __float2half_rn(d); // store the __half, see Q8_0 kernel
        }

        // byte j (0..15): low nibble = element j, high nibble = element j+16.
        // lane j<16 writes its byte using the nibble received from lane j+16.
        const uint32_t my   = (uint32_t)q & 0xF;
        const uint32_t pair = __shfl_xor_sync(0xffffffffu, my, 16);
        if (lane < 16) {
            y[ib].qs[lane] = (uint8_t)(my | (pair << 4));
        }
    }
}

// ---------------------------------------------------------------------------
// Q5_0: quantize_row_q5_0_ref (ggml-quants.c:757)
// ---------------------------------------------------------------------------

static __global__ void quantize_q5_0_kernel(
        const float * __restrict__ x, void * __restrict__ vy, const int64_t nblocks, const float fudge) {
    const int32_t lane = threadIdx.x; // 0 .. 31 == QK5_0

    for (int64_t ib = blockIdx.x; ib < nblocks; ib += gridDim.x) {
        const float xi = x[ib*QK5_0 + lane];

        // argmax of |x|, first (lowest index) element wins |x| ties (see Q4_0)
        float   bval = fabsf(xi);
        int32_t bidx = lane;
#pragma unroll
        for (int m = 16; m > 0; m >>= 1) {
            const float   oval = __shfl_xor_sync(0xffffffffu, bval, m);
            const int32_t oidx = __shfl_xor_sync(0xffffffffu, bidx, m);
            if (oval > bval || (oval == bval && oidx < bidx)) {
                bval = oval;
                bidx = oidx;
            }
        }

        // signed value of the argmax element, broadcast to the warp
        const float max = __shfl_sync(0xffffffffu, xi, bidx);

        // correctly-rounded integer division (see Q8_0 kernel); max/-16 is a
        // power-of-2 division (exact anyway), 1/d is the general case.
        // CPU stores FP16(fudge*d), codes use id=1/(fudge*d).
        const float d  = __fmul_rn(fudge, __fdiv_rn(max, -16.0f));
        const float id = d ? __fdiv_rn(1.0f, d) : 0.0f;

        // MIN(31, (int8_t)(x_j*id + 16.5f)): truncation toward zero, then
        // clamp. |x_j*id| <= 16 so x_j*id + 16.5 is in [-0.5, 32.5]. __fmul_rn
        // forces the product to round once (no FMA contraction) so the result
        // matches the CPU's separate product + add roundings, like Q4_0.
        const float   t = __fmul_rn(xi, id);
        const int32_t v = (int32_t)(t + 16.5f);
        const int32_t q = v > 31 ? 31 : v;

        block_q5_0 * y = (block_q5_0 *)vy;
        if (lane == 0) {
            y[ib].d = __float2half_rn(d); // store the __half, see Q8_0 kernel
        }

        // byte j (0..15): low nibble = element j, high nibble = element j+16.
        const uint32_t my   = (uint32_t)q & 0xF;
        const uint32_t pair = __shfl_xor_sync(0xffffffffu, my, 16);
        if (lane < 16) {
            y[ib].qs[lane] = (uint8_t)(my | (pair << 4));
        }

        // 5-th bit of every element -> qh bit e (= element index e, because
        // element j<16 maps to bit j and element j+16 maps to bit j+16).
        // __ballot_sync must be executed by every lane (uniform), so it is done
        // here for the whole warp: bit e is set iff lane e's element has its
        // 5th bit set, exactly reproducing the reference bit layout. Store the
        // 4 bytes little-endian (the ref memcpys a native uint32).
        const uint32_t qh = __ballot_sync(0xffffffffu, (q >> 4) & 1);
        if (lane == 0) {
            y[ib].qh[0] = (uint8_t)(qh >>  0);
            y[ib].qh[1] = (uint8_t)(qh >>  8);
            y[ib].qh[2] = (uint8_t)(qh >> 16);
            y[ib].qh[3] = (uint8_t)(qh >> 24);
        }
    }
}

// ---------------------------------------------------------------------------
// Q5_1: quantize_row_q5_1_ref (ggml-quants.c:809). No fudge. No clamp
// (bug-compatible: codes are raw (uint8_t)(x+0.5f), matching cpy-utils).
// ---------------------------------------------------------------------------

static __global__ void quantize_q5_1_kernel(
        const float * __restrict__ x, void * __restrict__ vy, const int64_t nblocks, const float fudge) {
    (void) fudge; // no fudge for Q5_1 (matches CPU ref)
    const int32_t lane = threadIdx.x; // 0 .. 31 == QK5_1

    for (int64_t ib = blockIdx.x; ib < nblocks; ib += gridDim.x) {
        const float xi = x[ib*QK5_1 + lane];

        // min/max reductions (values only; order-independent).
        float bmin = xi;
        float bmax = xi;
#pragma unroll
        for (int m = 16; m > 0; m >>= 1) {
            bmin = fminf(bmin, __shfl_xor_sync(0xffffffffu, bmin, m));
            bmax = fmaxf(bmax, __shfl_xor_sync(0xffffffffu, bmax, m));
        }

        const float d  = __fdiv_rn(bmax - bmin, 31.0f);
        const float id = d ? __fdiv_rn(1.0f, d) : 0.0f;

        // (x-min)*id + 0.5f truncation toward zero, no clamp (matches ref).
        const float t = __fmul_rn(__fsub_rn(xi, bmin), id);
        const uint32_t q = (uint32_t)(int32_t)(t + 0.5f);

        block_q5_1 * y = (block_q5_1 *)vy;
        if (lane == 0) {
            y[ib].d = __float2half_rn(d);
            y[ib].m = __float2half_rn(bmin);
        }

        // byte j (0..15): low nibble = element j, high nibble = element j+16.
        const uint32_t my   = q & 0xF;
        const uint32_t pair = __shfl_xor_sync(0xffffffffu, my, 16);
        if (lane < 16) {
            y[ib].qs[lane] = (uint8_t)(my | (pair << 4));
        }

        // 5th bit -> qh bitmap, LE uint32 (same as Q5_0 kernel).
        const uint32_t qhb = __ballot_sync(0xffffffffu, (q >> 4) & 1);
        if (lane == 0) {
            y[ib].qh[0] = (uint8_t)(qhb >>  0);
            y[ib].qh[1] = (uint8_t)(qhb >>  8);
            y[ib].qh[2] = (uint8_t)(qhb >> 16);
            y[ib].qh[3] = (uint8_t)(qhb >> 24);
        }
    }
}

// ---------------------------------------------------------------------------
// Q4_1: quantize_row_q4_1_ref (ggml-quants.c:718). No fudge. MIN(15) clamp.
// Q4_1 ignores symmetric_q4_0 on the CPU, so no symmetric guard is needed.
// ---------------------------------------------------------------------------

static __global__ void quantize_q4_1_kernel(
        const float * __restrict__ x, void * __restrict__ vy, const int64_t nblocks, const float fudge) {
    (void) fudge; // no fudge for Q4_1 (matches CPU ref)
    const int32_t lane = threadIdx.x; // 0 .. 31 == QK4_1

    for (int64_t ib = blockIdx.x; ib < nblocks; ib += gridDim.x) {
        const float xi = x[ib*QK4_1 + lane];

        // min/max reductions (values only; order-independent).
        float bmin = xi;
        float bmax = xi;
#pragma unroll
        for (int m = 16; m > 0; m >>= 1) {
            bmin = fminf(bmin, __shfl_xor_sync(0xffffffffu, bmin, m));
            bmax = fmaxf(bmax, __shfl_xor_sync(0xffffffffu, bmax, m));
        }

        const float d  = __fdiv_rn(bmax - bmin, 15.0f);
        const float id = d ? __fdiv_rn(1.0f, d) : 0.0f;

        // MIN(15, (int8_t)((x-min)*id + 0.5f)) truncation toward zero.
        const float t = __fmul_rn(__fsub_rn(xi, bmin), id);
        const int32_t v = (int32_t)(t + 0.5f);
        const uint32_t q = (uint32_t)(v > 15 ? 15 : v);

        block_q4_1 * y = (block_q4_1 *)vy;
        if (lane == 0) {
            y[ib].d = __float2half_rn(d);
            y[ib].m = __float2half_rn(bmin);
        }

        // byte j (0..15): low nibble = element j, high nibble = element j+16.
        const uint32_t my   = q & 0xF;
        const uint32_t pair = __shfl_xor_sync(0xffffffffu, my, 16);
        if (lane < 16) {
            y[ib].qs[lane] = (uint8_t)(my | (pair << 4));
        }
    }
}

// ---------------------------------------------------------------------------
// Q6_0 OLS: quantize_q6_0 (ggml-quants.c:3668) via quantize_row_q6_0_impl.
// OLS is KEPT. The kernel lives below with the make_qx_quants device port
// (one thread per block, weight = x*x, FP16(d*fudge)). The old plain
// max/-32 kernel was removed: HEAD CPU never uses it for GGUF.
// ---------------------------------------------------------------------------
// (Q6_0 OLS kernel + driver defined after make_qx_quants_device.)

// ---------------------------------------------------------------------------
// Q4_0 with importance matrix: make_qx_quants (ggml-quants.c:1786)
// ---------------------------------------------------------------------------

// Byte-exact port of ggml_compute_fp32_to_fp16 (ggml/src/ggml-impl.h:595), the
// fp16 conversion the CPU reference uses when __F16C__ is off (as in this
// build). __float2half_rn agrees with it on finite values and overflow to inf
// (both round to nearest even), but encodes NaN differently: the hardware
// conversion emits the canonical 0x7fff, while this bit-mask path emits
// (sign ? 0xfe00 : 0x7e00). The make_qx_quants scale can be NaN for degenerate
// imatrix blocks, so this must be byte-exact too.
//
// Note: only the NaN *sign/payload* is not normalized here, and that is a
// CPU-vendor semantic difference, not a porting gap: x86 SSE sets the sign of
// a NaN result from the operand signs (so e.g. -inf/+inf and 0*inf render as
// -NaN -> 0xfe00), whereas NVIDIA hardware always emits the default quiet NaN
// (+NaN -> 0x7e00). Even CPU-only llama.cpp produces different bytes for these
// degenerate, NaN-scale blocks across CPU vendors. The harness therefore treats
// any NaN d as equal (spec.nan_d_equal); every finite block must still match
// byte-for-byte.
static __device__ __forceinline__ uint16_t fp32_to_fp16_ggml(float f) {
    const float scale_to_inf  = __int_as_float(0x77800000u);
    const float scale_to_zero = __int_as_float(0x08800000u);
    float base = __fmul_rn(__fmul_rn(fabsf(f), scale_to_inf), scale_to_zero);

    const uint32_t w      = __float_as_uint(f);
    const uint32_t shl1_w = w + w;
    const uint32_t sign   = w & 0x80000000u;
    uint32_t bias = shl1_w & 0xFF000000u;
    if (bias < 0x71000000u) {
        bias = 0x71000000u;
    }

    base = __fadd_rn(__int_as_float((bias >> 1) + 0x07800000u), base);
    const uint32_t bits          = __float_as_uint(base);
    const uint32_t exp_bits      = (bits >> 13) & 0x00007C00u;
    const uint32_t mantissa_bits = bits & 0x00000FFFu;
    const uint32_t nonsign       = exp_bits + mantissa_bits;
    return (uint16_t)((sign >> 16) | (shl1_w > 0xFF000000u ? 0x7E00u : nonsign));
}

// round-half-to-even via the 2^23 + 2^22 magic constant (ggml-quants.c:1779)
static __device__ int nearest_int_device(float fval) {
    const unsigned int u = __float_as_uint(__fadd_rn(fval, 12582912.0f));
    return (int)((u & 0x007fffffu) - 0x00400000u);
}

// clamp_l_device(l, nmax) as in make_qx_quants
static __device__ int clamp_l_device(int l, int nmax) {
    return l > nmax-1 ? nmax-1 : (l < -nmax ? -nmax : l);
}

// Byte-exact device port of make_qx_quants, restricted to the path the Q4_0
// imatrix quantizer uses: rmse_type == 1 with a non-null weight vector. One
// thread replays the whole sequential algorithm in the exact CPU order, so
// the greedy search and coordinate-descent loop take the identical sequence
// of steps. Every float op is a correctly-rounded intrinsic so the
// -use_fast_math build (approximate sqrt/div) and FMA contraction cannot
// change a single bit: `sumlx += w*x*l` is (w*x)*l + sumlx with separate
// roundings, exactly like the non-contracting host compiler.
static __device__ float make_qx_quants_device(int n, int nmax, const float * x, int8_t * L, const float * qw) {
    float max  = 0.0f;
    float amax = 0.0f;
    for (int i = 0; i < n; ++i) {
        const float ax = fabsf(x[i]);
        if (ax > amax) { amax = ax; max = x[i]; }
    }
    if (amax < 1e-15f) { // GROUP_MAX_EPS: all zero
        for (int i = 0; i < n; ++i) L[i] = 0;
        return 0.0f;
    }
    float iscale = __fdiv_rn(-(float)nmax, max);

    float sumlx = 0.0f;
    float suml2 = 0.0f;
    for (int i = 0; i < n; ++i) {
        int l = nearest_int_device(__fmul_rn(iscale, x[i]));
        l = clamp_l_device(l, nmax);
        L[i] = (int8_t)(l + nmax);
        const float w = qw[i]; // rmse_type == 1, qw always provided
        sumlx = __fadd_rn(sumlx, __fmul_rn(__fmul_rn(w, x[i]), (float)l));
        suml2 = __fadd_rn(suml2, __fmul_rn(__fmul_rn(w, (float)l), (float)l));
    }
    float scale = suml2 != 0.0f ? __fdiv_rn(sumlx, suml2) : 0.0f;
    float best = __fmul_rn(scale, sumlx);
    float best_sumlx = sumlx, best_suml2 = suml2;

    for (int is = -9; is <= 9; ++is) {
        // iscale = -(nmax + 0.1*is)/max
        iscale = __fdiv_rn(-__fadd_rn((float)nmax, __fmul_rn(0.1f, (float)is)), max);
        sumlx = suml2 = 0.0f;
        for (int i = 0; i < n; ++i) {
            int l = nearest_int_device(__fmul_rn(iscale, x[i]));
            l = clamp_l_device(l, nmax);
            const float w = qw[i];
            sumlx = __fadd_rn(sumlx, __fmul_rn(__fmul_rn(w, x[i]), (float)l));
            suml2 = __fadd_rn(suml2, __fmul_rn(__fmul_rn(w, (float)l), (float)l));
        }
        if (suml2 > 0.0f && __fmul_rn(sumlx, sumlx) > __fmul_rn(best, suml2)) {
            for (int i = 0; i < n; ++i) {
                int l = nearest_int_device(__fmul_rn(iscale, x[i]));
                L[i] = (int8_t)(nmax + clamp_l_device(l, nmax));
            }
            scale = __fdiv_rn(sumlx, suml2);
            best = __fmul_rn(scale, sumlx);
            best_sumlx = sumlx; best_suml2 = suml2;
        }
        // iscale = (nmax-1 + 0.1*is)/max
        iscale = __fdiv_rn(__fadd_rn((float)(nmax-1), __fmul_rn(0.1f, (float)is)), max);
        sumlx = suml2 = 0.0f;
        for (int i = 0; i < n; ++i) {
            int l = nearest_int_device(__fmul_rn(iscale, x[i]));
            l = clamp_l_device(l, nmax);
            const float w = qw[i];
            sumlx = __fadd_rn(sumlx, __fmul_rn(__fmul_rn(w, x[i]), (float)l));
            suml2 = __fadd_rn(suml2, __fmul_rn(__fmul_rn(w, (float)l), (float)l));
        }
        if (suml2 > 0.0f && __fmul_rn(sumlx, sumlx) > __fmul_rn(best, suml2)) {
            for (int i = 0; i < n; ++i) {
                int l = nearest_int_device(__fmul_rn(iscale, x[i]));
                L[i] = (int8_t)(nmax + clamp_l_device(l, nmax));
            }
            scale = __fdiv_rn(sumlx, suml2);
            best = __fmul_rn(scale, sumlx);
            best_sumlx = sumlx; best_suml2 = suml2;
        }
    }

    // coordinate descent; identical step sequence to the reference
    sumlx = best_sumlx; suml2 = best_suml2;
    for (int iter = 0; iter < n*(2*nmax-1); ++iter) {
        float abs_gmax = 0.0f, gmax = 0.0f;
        int best_j = -1;
        for (int j = 0; j < n; ++j) {
            const float w = qw[j];
            const int l = (int)L[j] - nmax;
            // g = scale*w*(x[j] - scale*l), each op rounded separately
            const float g = __fmul_rn(__fmul_rn(scale, w),
                    __fadd_rn(x[j], -__fmul_rn(scale, (float)l)));
            if ((g > 0.0f && l < nmax-1) || (g < 0.0f && l > -nmax)) {
                const float ag = fabsf(g);
                if (ag > abs_gmax) { abs_gmax = ag; gmax = g; best_j = j; }
            }
        }
        if (best_j < 0) break;

        float new_sumlx = sumlx, new_suml2 = suml2;
        const float w = qw[best_j];
        int l = (int)L[best_j] - nmax;
        if (gmax > 0.0f) {
            new_sumlx = __fadd_rn(new_sumlx, __fmul_rn(w, x[best_j]));
            new_suml2 = __fadd_rn(new_suml2, __fmul_rn(w, (float)(2*l + 1)));
            l += 1;
        } else {
            new_sumlx = __fsub_rn(new_sumlx, __fmul_rn(w, x[best_j]));
            new_suml2 = __fsub_rn(new_suml2, __fmul_rn(w, (float)(2*l - 1)));
            l -= 1;
        }
        if (new_suml2 > 0.0f && __fmul_rn(new_sumlx, new_sumlx) > __fmul_rn(best, new_suml2)) {
            sumlx = new_sumlx; suml2 = new_suml2;
            scale = __fdiv_rn(sumlx, suml2);
            best = __fmul_rn(scale, sumlx);
            L[best_j] = (int8_t)(l + nmax);
        } else {
            break;
        }
    }
    return scale;
}

// Byte-exact device port of make_qkx3_quants (ggml-quants.c:2211), used by the
// Q4_1/Q5_1 imatrix quantizers (rmse path with min). One thread replays the
// whole sequential algorithm in the exact CPU order. Double accumulators use
// correctly-rounded intrinsics (__dadd_rn/__dmul_rn/__ddiv_rn) so the
// -use_fast_math build cannot change a bit; float parts use __fdiv_rn/
// __fmul_rn/__fadd_rn/__fsub_rn. Requires callers in ggml-quants.c to compile
// with #pragma STDC FP_CONTRACT OFF (as they now do).
static __device__ float make_qkx3_quants_device(int n, int nmax, const float * x, const float * weights,
        uint8_t * L, float * the_min, uint8_t * Laux,
        float rmin, float rdelta, int nstep, bool use_mad) {
    float min = x[0];
    float max = x[0];
    double sum_w = weights ? (double)weights[0] : (double)(x[0]*x[0]);
    double sum_x = sum_w * (double)x[0];
    double sum_x2 = sum_w * (double)x[0] * (double)x[0];
    for (int i = 1; i < n; ++i) {
        if (x[i] < min) min = x[i];
        if (x[i] > max) max = x[i];
        float w = weights ? weights[i] : x[i]*x[i];
        sum_w = __dadd_rn(sum_w, (double)w);
        sum_x = __dadd_rn(sum_x, __dmul_rn((double)w, (double)x[i]));
        sum_x2 = __dadd_rn(sum_x2, __dmul_rn(__dmul_rn((double)w, (double)x[i]), (double)x[i]));
    }
    if (min > 0) {
        min = 0;
    }
    if (max - min < 1e-10f) {
        for (int i = 0; i < n; ++i) L[i] = 0;
        *the_min = -min;
        return 0.f;
    }
    float iscale = __fdiv_rn((float)nmax, __fsub_rn(max, min));
    float scale = __fdiv_rn(1.0f, iscale);
    double best_mad = 0;
    for (int i = 0; i < n; ++i) {
        int l = nearest_int_device(__fmul_rn(iscale, __fsub_rn(x[i], min)));
        l = l > nmax ? nmax : (l < 0 ? 0 : l);
        L[i] = (uint8_t)l;
        double diff = __dadd_rn(__dmul_rn((double)scale, (double)L[i]), (double)min);
        diff = __dsub_rn(diff, (double)x[i]);
        diff = use_mad ? fabs(diff) : __dmul_rn(diff, diff);
        double w = weights ? (double)weights[i] : (double)(x[i]*x[i]);
        best_mad = __dadd_rn(best_mad, __dmul_rn(w, diff));
    }
    if (nstep < 1) {
        *the_min = -min;
        return scale;
    }
    for (int is = 0; is <= nstep; ++is) {
        iscale = __fdiv_rn(__fadd_rn(__fadd_rn(rmin, __fmul_rn(rdelta, (float)is)), (float)nmax), __fsub_rn(max, min));
        double sum_l = 0, sum_l2 = 0, sum_xl = 0;
        for (int i = 0; i < n; ++i) {
            int l = nearest_int_device(__fmul_rn(iscale, __fsub_rn(x[i], min)));
            l = l > nmax ? nmax : (l < 0 ? 0 : l);
            Laux[i] = (uint8_t)l;
            float w = weights ? weights[i] : x[i]*x[i];
            sum_l  = __dadd_rn(sum_l, __dmul_rn((double)w, (double)l));
            sum_l2 = __dadd_rn(sum_l2, __dmul_rn(__dmul_rn((double)w, (double)l), (double)l));
            sum_xl = __dadd_rn(sum_xl, __dmul_rn(__dmul_rn((double)w, (double)l), (double)x[i]));
        }
        double D = __dsub_rn(__dmul_rn(sum_w, sum_l2), __dmul_rn(sum_l, sum_l));
        if (D > 0) {
            double this_scale = __ddiv_rn(__dsub_rn(__dmul_rn(sum_w, sum_xl), __dmul_rn(sum_x, sum_l)), D);
            double this_min   = __ddiv_rn(__dsub_rn(__dmul_rn(sum_l2, sum_x), __dmul_rn(sum_l, sum_xl)), D);
            if (this_min > 0) {
                this_min = 0;
                this_scale = __ddiv_rn(sum_xl, sum_l2);
            }
            double mad = 0;
            if (use_mad) {
                for (int i = 0; i < n; ++i) {
                    double diff = __dadd_rn(__dmul_rn((double)this_scale, (double)Laux[i]), (double)this_min);
                    diff = __dsub_rn(diff, (double)x[i]);
                    diff = fabs(diff);
                    double w = weights ? (double)weights[i] : (double)(x[i]*x[i]);
                    mad = __dadd_rn(mad, __dmul_rn(w, diff));
                }
            } else {
                mad = __dsub_rn(sum_x2, __dmul_rn(2*this_scale, sum_xl));
                mad = __dsub_rn(mad, __dmul_rn(2*this_min, sum_x));
                mad = __dadd_rn(mad, __dmul_rn(__dmul_rn(2*this_scale, this_min), sum_l));
                mad = __dadd_rn(mad, __dmul_rn(__dmul_rn(this_scale, this_scale), sum_l2));
                mad = __dadd_rn(mad, __dmul_rn(__dmul_rn(this_min, this_min), sum_w));
            }
            if (mad < best_mad) {
                for (int i = 0; i < n; ++i) {
                    L[i] = Laux[i];
                }
                best_mad = mad;
                scale = (float)this_scale;
                min = (float)this_min;
            }
        }
    }
    if (use_mad) {
        *the_min = -min;
        return scale;
    }

    double sum_l = 0, sum_l2 = 0, sum_xl = 0;
    for (int i = 0; i < n; ++i) {
        int l = L[i];
        double w = weights ? (double)weights[i] : (double)(x[i]*x[i]);
        sum_l  = __dadd_rn(sum_l, __dmul_rn(w, (double)l));
        sum_l2 = __dadd_rn(sum_l2, __dmul_rn(__dmul_rn(w, (double)l), (double)l));
        sum_xl = __dadd_rn(sum_xl, __dmul_rn(__dmul_rn(w, (double)l), (double)x[i]));
    }
    double best = __dsub_rn(__dadd_rn(__dmul_rn(__dmul_rn(2.0, (double)scale), sum_xl), __dmul_rn(__dmul_rn(2.0, (double)min), sum_x)),
        __dmul_rn(__dmul_rn(__dmul_rn(2.0, (double)scale), (double)min), sum_l));
    best = __dsub_rn(best, __dmul_rn(__dmul_rn((double)scale, (double)scale), sum_l2));
    best = __dsub_rn(best, __dmul_rn(__dmul_rn((double)min, (double)min), sum_w));
    int last_j = -1, last_dir = 0;
    for (int itry = 0; itry < nmax*n; ++itry) {
        float gmax = 0;
        int best_j = -1, dir = 0;
        for (int j = 0; j < n; ++j) {
            float g = __double2float_rn(__dsub_rn(__dsub_rn((double)x[j], __dmul_rn((double)scale, (double)L[j])), (double)min));
            if (g > 0 && L[j] < nmax && g > gmax) {
                gmax = g; best_j = j; dir = 1;
            }
            else if (g < 0 && L[j] > 0 && -g > gmax) {
                gmax = -g; best_j = j; dir = -1;
            }
        }
        if (best_j < 0 || (best_j == last_j && dir == -last_dir)) break;
        double w = weights ? (double)weights[best_j] : (double)(x[best_j]*x[best_j]);
        sum_l  = __dadd_rn(sum_l, __dmul_rn(w, (double)dir));
        sum_l2 = __dadd_rn(sum_l2, __dmul_rn(w, (double)(2*L[best_j]*dir + 1)));
        sum_xl = __dadd_rn(sum_xl, __dmul_rn(__dmul_rn(w, (double)x[best_j]), (double)dir));
        double D = __dsub_rn(__dmul_rn(sum_w, sum_l2), __dmul_rn(sum_l, sum_l));
        if (D <= 0) break;
        double this_scale = __ddiv_rn(__dsub_rn(__dmul_rn(sum_w, sum_xl), __dmul_rn(sum_x, sum_l)), D);
        double this_min   = __ddiv_rn(__dsub_rn(__dmul_rn(sum_l2, sum_x), __dmul_rn(sum_l, sum_xl)), D);
        if (this_min > 0) {
            this_min = 0;
            this_scale = __ddiv_rn(sum_xl, sum_l2);
        }
        if (this_scale < 0) break;
        double score = __dadd_rn(__dmul_rn(2*this_scale, sum_xl), __dmul_rn(2*this_min, (double)sum_x));
        score = __dsub_rn(score, __dmul_rn(__dmul_rn(2*this_scale, this_min), sum_l));
        score = __dsub_rn(score, __dmul_rn(__dmul_rn(this_scale, this_scale), sum_l2));
        score = __dsub_rn(score, __dmul_rn(__dmul_rn(this_min, this_min), sum_w));
        if (score <= best) break;
        best = score;
        scale = (float)this_scale;
        min = (float)this_min;
        L[best_j] += (uint8_t)dir;
        last_j = best_j; last_dir = dir;
    }
    *the_min = -min;
    return scale;
}

// One thread per quant block: computes the per-block weights from the shared
// row sigma2 and the (row-reused) importance weights, then runs
// make_qx_quants. `base` is the chunk's first global quant block, so the
// row/weight indexing stays correct when a tensor spans several device chunks
// and the chunk base is not a multiple of blocks_per_row. Block gb = base + ib
// belongs to row gb/blocks_per_row and uses weight block gb%blocks_per_row,
// matching quantize_row_q4_0_impl which is called once per row with the same
// quant_weights pointer.
static __global__ void quantize_q4_0_imatrix_kernel(
        const float * __restrict__ x, const float * __restrict__ qw, const float * __restrict__ sigma2,
        void * __restrict__ vy, const int64_t base, const int64_t nblocks, const int32_t blocks_per_row,
        const float fudge) {
    const int64_t ib = (int64_t)blockIdx.x*blockDim.x + threadIdx.x;
    if (ib >= nblocks) {
        return;
    }
    const int64_t gb = base + ib;
    const float * xb = x + ib*QK4_0;
    const float * qb = qw + (int32_t)(gb % blocks_per_row)*QK4_0;
    const float s2   = sigma2[gb / blocks_per_row];

    float weight[QK4_0];
    int8_t L[QK4_0];
    for (int j = 0; j < QK4_0; ++j) {
        // weight[j] = qw[j]*sqrtf(sigma2 + xb[j]^2); __fsqrt_rn keeps the
        // square root exact under -use_fast_math, __fadd_rn/__fmul_rn keep
        // the sum/multiply uncontracted.
        weight[j] = __fmul_rn(qb[j], __fsqrt_rn(__fadd_rn(s2, __fmul_rn(xb[j], xb[j]))));
    }

    const float d = make_qx_quants_device(QK4_0, 8, xb, L, weight);

    block_q4_0 * y = (block_q4_0 *)vy;
    y[ib].d = __ushort_as_half(fp32_to_fp16_ggml(__fmul_rn(fudge, d)));
    for (int j = 0; j < QK4_0/2; ++j) {
        y[ib].qs[j] = (uint8_t)(L[j] | (L[j + QK4_0/2] << 4));
    }
}

// Q4_1 imatrix kernel: quantize_row_q4_1_impl via make_qkx3_quants (nmax=15).
// No fudge. Stores d and -the_min like the CPU (y.m = FP16(-min)).
static __global__ void quantize_q4_1_imatrix_kernel(
        const float * __restrict__ x, const float * __restrict__ qw, const float * __restrict__ sigma2,
        void * __restrict__ vy, const int64_t base, const int64_t nblocks, const int32_t blocks_per_row) {
    const int64_t ib = (int64_t)blockIdx.x*blockDim.x + threadIdx.x;
    if (ib >= nblocks) {
        return;
    }
    const int64_t gb = base + ib;
    const float * xb = x + ib*QK4_1;
    const float * qb = qw + (int32_t)(gb % blocks_per_row)*QK4_1;
    const float s2   = sigma2[gb / blocks_per_row];

    float weight[QK4_1];
    uint8_t L[QK4_1], Laux[QK4_1];
    for (int j = 0; j < QK4_1; ++j) {
        weight[j] = __fmul_rn(qb[j], __fsqrt_rn(__fadd_rn(s2, __fmul_rn(xb[j], xb[j]))));
    }

    float the_min;
    const float d = make_qkx3_quants_device(QK4_1, 15, xb, weight, L, &the_min, Laux, -0.9f, 0.05f, 36, false);

    block_q4_1 * y = (block_q4_1 *)vy;
    y[ib].d = __float2half_rn(d);
    y[ib].m = __float2half_rn(-the_min);
    for (int j = 0; j < QK4_1/2; ++j) {
        y[ib].qs[j] = (uint8_t)(L[j] | (L[j + QK4_1/2] << 4));
    }
}

// Q5_0 imatrix kernel: identical shape to quantize_q4_0_imatrix_kernel, but
// quantizes 32-value blocks with nmax == 16 and additionally packs each L's
// 5th bit into the qh bitmap like quantize_row_q5_0_impl (ggml-quants.c:3577).
static __global__ void quantize_q5_0_imatrix_kernel(
        const float * __restrict__ x, const float * __restrict__ qw, const float * __restrict__ sigma2,
        void * __restrict__ vy, const int64_t base, const int64_t nblocks, const int32_t blocks_per_row,
        const float fudge) {
    const int64_t ib = (int64_t)blockIdx.x*blockDim.x + threadIdx.x;
    if (ib >= nblocks) {
        return;
    }
    const int64_t gb = base + ib;
    const float * xb = x + ib*QK5_0;
    const float * qb = qw + (int32_t)(gb % blocks_per_row)*QK5_0;
    const float s2   = sigma2[gb / blocks_per_row];

    float weight[QK5_0];
    int8_t L[QK5_0];
    for (int j = 0; j < QK5_0; ++j) {
        weight[j] = __fmul_rn(qb[j], __fsqrt_rn(__fadd_rn(s2, __fmul_rn(xb[j], xb[j]))));
    }

    const float d = make_qx_quants_device(QK5_0, 16, xb, L, weight);

    block_q5_0 * y = (block_q5_0 *)vy;
    y[ib].d = __ushort_as_half(fp32_to_fp16_ggml(__fmul_rn(fudge, d)));

    // qs low nibbles + qh 5th-bit bitmap; byte-exact clone of the packing loop
    // in quantize_row_q5_0_impl (L is already +nmax, so L in [0, 31]).
    uint32_t qh = 0;
    for (int j = 0; j < QK5_0/2; ++j) {
        const uint8_t xi0 = (uint8_t)L[j];
        const uint8_t xi1 = (uint8_t)L[j + QK5_0/2];
        y[ib].qs[j] = (uint8_t)((xi0 & 0x0F) | ((xi1 & 0x0F) << 4));
        qh |= ((uint32_t)((xi0 & 0x10u) >> 4)) << (j + 0);
        qh |= ((uint32_t)((xi1 & 0x10u) >> 4)) << (j + QK5_0/2);
    }
    y[ib].qh[0] = (uint8_t)(qh >>  0);
    y[ib].qh[1] = (uint8_t)(qh >>  8);
    y[ib].qh[2] = (uint8_t)(qh >> 16);
    y[ib].qh[3] = (uint8_t)(qh >> 24);
}

// Q5_1 imatrix kernel: quantize_row_q5_1_impl via make_qkx3_quants (nmax=31).
// No fudge. Stores d and -the_min like the CPU (y.m = FP16(-min)).
static __global__ void quantize_q5_1_imatrix_kernel(
        const float * __restrict__ x, const float * __restrict__ qw, const float * __restrict__ sigma2,
        void * __restrict__ vy, const int64_t base, const int64_t nblocks, const int32_t blocks_per_row) {
    const int64_t ib = (int64_t)blockIdx.x*blockDim.x + threadIdx.x;
    if (ib >= nblocks) {
        return;
    }
    const int64_t gb = base + ib;
    const float * xb = x + ib*QK5_1;
    const float * qb = qw + (int32_t)(gb % blocks_per_row)*QK5_1;
    const float s2   = sigma2[gb / blocks_per_row];

    float weight[QK5_1];
    uint8_t L[QK5_1], Laux[QK5_1];
    for (int j = 0; j < QK5_1; ++j) {
        weight[j] = __fmul_rn(qb[j], __fsqrt_rn(__fadd_rn(s2, __fmul_rn(xb[j], xb[j]))));
    }

    float the_min;
    const float d = make_qkx3_quants_device(QK5_1, 31, xb, weight, L, &the_min, Laux, -0.9f, 0.05f, 36, false);

    block_q5_1 * y = (block_q5_1 *)vy;
    y[ib].d = __float2half_rn(d);
    y[ib].m = __float2half_rn(-the_min);

    uint32_t qh = 0;
    for (int j = 0; j < QK5_1/2; ++j) {
        const uint8_t xi0 = L[j];
        const uint8_t xi1 = L[j + QK5_1/2];
        y[ib].qs[j] = (uint8_t)((xi0 & 0x0F) | ((xi1 & 0x0F) << 4));
        qh |= ((uint32_t)((xi0 & 0x10u) >> 4)) << (j + 0);
        qh |= ((uint32_t)((xi1 & 0x10u) >> 4)) << (j + QK5_1/2);
    }
    y[ib].qh[0] = (uint8_t)(qh >>  0);
    y[ib].qh[1] = (uint8_t)(qh >>  8);
    y[ib].qh[2] = (uint8_t)(qh >> 16);
    y[ib].qh[3] = (uint8_t)(qh >> 24);
}

// Q6_0 imatrix kernel: identical shape to quantize_q5_0_imatrix_kernel, but
// quantizes with nmax == 32 (L in [0, 63], bits 0-5) and packs the 2-bit qh
// (bits 4-5) exactly like quantize_row_q6_0_impl (ggml-quants.c:3697).
static __global__ void quantize_q6_0_imatrix_kernel(
        const float * __restrict__ x, const float * __restrict__ qw, const float * __restrict__ sigma2,
        void * __restrict__ vy, const int64_t base, const int64_t nblocks, const int32_t blocks_per_row,
        const float fudge) {
    const int64_t ib = (int64_t)blockIdx.x*blockDim.x + threadIdx.x;
    if (ib >= nblocks) {
        return;
    }
    const int64_t gb = base + ib;
    const float * xb = x + ib*QK6_0;
    const float * qb = qw + (int32_t)(gb % blocks_per_row)*QK6_0;
    const float s2   = sigma2[gb / blocks_per_row];

    float weight[QK6_0];
    int8_t L[QK6_0];
    for (int j = 0; j < QK6_0; ++j) {
        weight[j] = __fmul_rn(qb[j], __fsqrt_rn(__fadd_rn(s2, __fmul_rn(xb[j], xb[j]))));
    }

    const float d = make_qx_quants_device(QK6_0, 32, xb, L, weight);

    block_q6_0 * y = (block_q6_0 *)vy;
    y[ib].d = __ushort_as_half(fp32_to_fp16_ggml(__fmul_rn(fudge, d)));

    // Byte-exact clone of the packing loop in quantize_row_q6_0_impl: L is
    // already +nmax so in [0, 63]; low nibbles -> qs, bits 4-5 -> the 2-bit qh.
    for (int j = 0; j < QK6_0/2; ++j) {
        const uint8_t xi0 = (uint8_t)L[j];
        const uint8_t xi1 = (uint8_t)L[j + QK6_0/2];
        y[ib].qs[j] = (uint8_t)((xi0 & 0x0F) | ((xi1 & 0x0F) << 4));
    }
    for (int k = 0; k < QK6_0/4; ++k) {
        // qh[k] = h_k | (h_{k+8} << 4), h_e = (q_e>>4)|((q_{e+16}>>4)<<2)
        const uint8_t a0 = (uint8_t)L[k];
        const uint8_t a1 = (uint8_t)L[k + QK6_0/2];
        const uint8_t b0 = (uint8_t)L[k + QK6_0/4];
        const uint8_t b1 = (uint8_t)L[k + 3*(QK6_0/4)];
        y[ib].qh[k] = (uint8_t)((a0 >> 4) | ((a1 >> 4) << 2)
                              | (((b0 >> 4) | ((b1 >> 4) << 2)) << 4));
    }
}

// Q6_0 OLS without imatrix: quantize_row_q6_0_impl with quant_weights == NULL.
// CPU sets weight[j] = xb[j]*xb[j], then d = make_qx_quants(QK6_0, 32, ...)
// with fudge. One thread per block replays it exactly. OLS is KEPT.
static __global__ void quantize_q6_0_ols_kernel(
        const float * __restrict__ x,
        void * __restrict__ vy, const int64_t nblocks, const float fudge) {
    const int64_t ib = (int64_t)blockIdx.x*blockDim.x + threadIdx.x;
    if (ib >= nblocks) {
        return;
    }
    const float * xb = x + ib*QK6_0;

    float weight[QK6_0];
    int8_t L[QK6_0];
    for (int j = 0; j < QK6_0; ++j) {
        weight[j] = __fmul_rn(xb[j], xb[j]);
    }

    const float d = make_qx_quants_device(QK6_0, 32, xb, L, weight);

    block_q6_0 * y = (block_q6_0 *)vy;
    y[ib].d = __ushort_as_half(fp32_to_fp16_ggml(__fmul_rn(fudge, d)));

    for (int j = 0; j < QK6_0/2; ++j) {
        const uint8_t xi0 = (uint8_t)L[j];
        const uint8_t xi1 = (uint8_t)L[j + QK6_0/2];
        y[ib].qs[j] = (uint8_t)((xi0 & 0x0F) | ((xi1 & 0x0F) << 4));
    }
    for (int k = 0; k < QK6_0/4; ++k) {
        const uint8_t a0 = (uint8_t)L[k];
        const uint8_t a1 = (uint8_t)L[k + QK6_0/2];
        const uint8_t b0 = (uint8_t)L[k + QK6_0/4];
        const uint8_t b1 = (uint8_t)L[k + 3*(QK6_0/4)];
        y[ib].qh[k] = (uint8_t)((a0 >> 4) | ((a1 >> 4) << 2)
                              | (((b0 >> 4) | ((b1 >> 4) << 2)) << 4));
    }
}

// Q8_0 imatrix kernel (kept for research, UNUSED for GGUF: HEAD CPU
// quantize_q8_0 ignores imatrix, so the helper routes to plain).
// Unlike Q4_0/Q5_0/Q6_0, Q8_0 cannot use make_qx_quants (its offset encoding
// would overflow int8_t for 8-bit codes; block_q8_0 stores signed int8
// directly).
// block_q8_0 stores signed int8 directly). The CPU impl
// quantize_row_q8_0_impl (ggml-quants.c:3799) instead keeps the plain
// `roundf(x*id)` codes and refines only the scale via weighted least-squares
// d = sumqx/sumq2, weighted per element w[j] = qw[j]*sqrt(sigma2 + x[j]^2).
// This kernel replays that loop in one thread per block with correctly-rounded
// intrinsics (__fdiv_rn/__fadd_rn/__fmul_rn/__fsqrt_rn and roundf), so it is
// byte-identical. The row-level sigma2 is pre-computed on the host in the
// exact CPU summation order (see the driver).
static __global__ void quantize_q8_0_imatrix_kernel(
        const float * __restrict__ x, const float * __restrict__ qw, const float * __restrict__ sigma2,
        void * __restrict__ vy, const int64_t base, const int64_t nblocks, const int32_t blocks_per_row) {
    const int64_t ib = (int64_t)blockIdx.x*blockDim.x + threadIdx.x;
    if (ib >= nblocks) {
        return;
    }
    const int64_t gb = base + ib;
    const float * xb = x + ib*QK8_0;
    const float * qb = qw + (int32_t)(gb % blocks_per_row)*QK8_0;
    const float s2   = sigma2[gb / blocks_per_row];

    // per-element importance weight, byte-mirroring quantize_row_q8_0_impl
    float weight[QK8_0];
    for (int j = 0; j < QK8_0; ++j) {
        weight[j] = __fmul_rn(qb[j], __fsqrt_rn(__fadd_rn(s2, __fmul_rn(xb[j], xb[j]))));
    }

    // plain top magnitude -> base scale (exact under -use_fast_math)
    float amax = 0.0f;
    for (int j = 0; j < QK8_0; ++j) {
        amax = fmaxf(amax, fabsf(xb[j]));
    }
    const float d0  = __fdiv_rn(amax, 127.0f);
    const float id0 = d0 ? __fdiv_rn(1.0f, d0) : 0.0f;

    // weighted least-squares scale over the plain signed codes
    block_q8_0 * y = (block_q8_0 *)vy;
    float sumqx = 0.0f, sumq2 = 0.0f;
    for (int j = 0; j < QK8_0; ++j) {
        const float v = xb[j];
        const int8_t q = (int8_t)roundf(v*id0);
        y[ib].qs[j] = q;

        const float wq = __fmul_rn(weight[j], (float)q);
        sumqx = __fadd_rn(sumqx, __fmul_rn(wq, v));        // w*q*x
        sumq2 = __fadd_rn(sumq2, __fmul_rn(wq, (float)q)); // w*q*q
    }
    const float d = sumq2 > 0.0f ? __fdiv_rn(sumqx, sumq2) : d0;
    y[ib].d = __ushort_as_half(fp32_to_fp16_ggml(d));
}

// ---------------------------------------------------------------------------
// host driver (shared by all block quants)
// ---------------------------------------------------------------------------

using quantize_kernel_t = void (*)(const float *, void *, int64_t, float);

static size_t ggml_cuda_quantize_generic(const float * src, void * dst, int64_t nrows, int64_t n_per_row,
        int64_t qk, size_t blk_size, quantize_kernel_t kernel, const char * name, float fudge) {
    GGML_ASSERT(nrows > 0);
    GGML_ASSERT(n_per_row % qk == 0);

    const int64_t nblocks_total = nrows*(n_per_row/qk);

    int n_devices = 0;
    if (cudaGetDeviceCount(&n_devices) != cudaSuccess || n_devices == 0) {
        return 0;
    }
    if (cudaSetDevice(0) != cudaSuccess) { // POC: device 0 only
        return 0;
    }

    // Fixed-size device buffers; the tensor is processed in chunks so that a
    // single large tensor never needs a huge VRAM allocation (which can fail
    // silently while another context holds most of the memory, e.g. a loaded
    // model in llama-server/webui, and then produce a confusing illegal-access
    // error from the NULL pointers). ~128 MiB of F32 input per chunk.
    const int64_t chunk_blocks = 1 << 20;                  // quant blocks per chunk
    const int64_t chunk_x      = chunk_blocks*qk;          // floats per chunk
    const int64_t chunk_y      = chunk_blocks*blk_size;    // bytes per chunk

    float   * x_dev = nullptr;
    uint8_t * y_dev = nullptr;

    cudaError_t err = cudaMalloc(&x_dev, chunk_x*sizeof(float));
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: %s: cudaMalloc(x_dev, %" PRId64 "): %s\n",
                __func__, name, (int64_t)(chunk_x*sizeof(float)), cudaGetErrorString(err));
        return 0;
    }
    err = cudaMalloc(&y_dev, chunk_y);
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: %s: cudaMalloc(y_dev, %" PRId64 "): %s\n",
                __func__, name, (int64_t)chunk_y, cudaGetErrorString(err));
        cudaFree(x_dev);
        return 0;
    }

    // one warp per quant block
    const int64_t block_size = qk;

    for (int64_t base = 0; base < nblocks_total; base += chunk_blocks) {
        const int64_t nblocks = std::min(chunk_blocks, nblocks_total - base);

        err = cudaMemcpy(x_dev, src + base*qk, nblocks*qk*sizeof(float), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            fprintf(stderr, "%s: %s: cudaMemcpy H2D: %s\n", __func__, name, cudaGetErrorString(err));
            break;
        }

        kernel<<<(unsigned)nblocks, (unsigned)block_size>>>(x_dev, y_dev, nblocks, fudge);

        err = cudaMemcpy((char *)dst + base*blk_size, y_dev, nblocks*blk_size, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            fprintf(stderr, "%s: %s: cudaMemcpy D2H: %s\n", __func__, name, cudaGetErrorString(err));
            break;
        }
    }

    cudaFree(x_dev);
    cudaFree(y_dev);

    if (err != cudaSuccess) {
        return 0;
    }

    return nblocks_total*blk_size;
}

// Order after Joel's KT: Q8_0, Q6_0, Q5_0, Q4_0. Q8_0 matches HEAD CPU
// including its Q6_0-fudge quirk (ggml-quants.c:915).
size_t ggml_cuda_quantize_q8_0(const float * src, void * dst, int64_t nrows, int64_t n_per_row) {
    const float fudge = ggml_get_quantize_fudge_factor(GGML_TYPE_Q6_0); // match CPU quirk
    return ggml_cuda_quantize_generic(src, dst, nrows, n_per_row,
            QK8_0, sizeof(block_q8_0), quantize_q8_0_kernel, "q8_0", fudge);
}

// Q6_0 OLS entry: replaced below by the OLS driver (make_qx + fudge).
// Declared here for ordering; defined after the OLS kernel.
size_t ggml_cuda_quantize_q6_0(const float * src, void * dst, int64_t nrows, int64_t n_per_row);

// --- Removable Q5_0 section ---
size_t ggml_cuda_quantize_q5_0(const float * src, void * dst, int64_t nrows, int64_t n_per_row) {
    const float fudge = ggml_get_quantize_fudge_factor(GGML_TYPE_Q5_0);
    return ggml_cuda_quantize_generic(src, dst, nrows, n_per_row,
            QK5_0, sizeof(block_q5_0), quantize_q5_0_kernel, "q5_0", fudge);
}

// --- Removable Q4_0 section ---
size_t ggml_cuda_quantize_q4_0(const float * src, void * dst, int64_t nrows, int64_t n_per_row) {
    const float fudge = ggml_get_quantize_fudge_factor(GGML_TYPE_Q4_0);
    return ggml_cuda_quantize_generic(src, dst, nrows, n_per_row,
            QK4_0, sizeof(block_q4_0), quantize_q4_0_kernel, "q4_0", fudge);
}

// Q4_1 without imatrix (quantize_row_q4_1_ref). No fudge; ignores symmetric.
size_t ggml_cuda_quantize_q4_1(const float * src, void * dst, int64_t nrows, int64_t n_per_row) {
    return ggml_cuda_quantize_generic(src, dst, nrows, n_per_row,
            QK4_1, sizeof(block_q4_1), quantize_q4_1_kernel, "q4_1", 1.0f);
}

// Q4_0 with an importance matrix. `imatrix` holds n_per_row weights and is
// reused for every row, exactly like the CPU quantize_row_q4_0_impl which is
// called once per row with the same quant_weights pointer.
size_t ggml_cuda_quantize_q4_0_imatrix(const float * src, void * dst, int64_t nrows, int64_t n_per_row,
        const float * imatrix) {
    GGML_ASSERT(nrows > 0);
    GGML_ASSERT(n_per_row % QK4_0 == 0);

    const int64_t nblocks_total = nrows*(n_per_row/QK4_0);
    const int32_t blocks_per_row = (int32_t)(n_per_row/QK4_0);

    int n_devices = 0;
    if (cudaGetDeviceCount(&n_devices) != cudaSuccess || n_devices == 0) {
        return 0;
    }
    if (cudaSetDevice(0) != cudaSuccess) { // POC: device 0 only
        return 0;
    }

    // Per-row sigma2 = sum_x2/n_per_row, summed sequentially in the exact
    // order of quantize_row_q4_0_impl. The host compiler is cl with default
    // /fp:precise (no fast-math, no FMA contraction), so this loop produces
    // the identical float as the CPU reference. A row sum cannot be reduced
    // in parallel on the GPU: different summation order -> different bits.
    std::vector<float> sigma2(nrows);
    for (int64_t irow = 0; irow < nrows; ++irow) {
        const float * xr = src + irow*n_per_row;
        float sum_x2 = 0.0f;
        for (int64_t j = 0; j < n_per_row; ++j) {
            sum_x2 += xr[j]*xr[j];
        }
        sigma2[irow] = sum_x2/n_per_row;
    }

    // Fixed-size device chunks, same rationale as the generic driver.
    const int64_t chunk_blocks = 1 << 20;
    const int64_t chunk_x      = chunk_blocks*QK4_0;
    const int64_t chunk_y      = chunk_blocks*sizeof(block_q4_0);

    float   * x_dev = nullptr;
    float   * q_dev = nullptr;
    float   * s_dev = nullptr;
    uint8_t * y_dev = nullptr;

    cudaError_t err = cudaMalloc(&x_dev, chunk_x*sizeof(float));
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: q4_0_imatrix: cudaMalloc(x_dev, %" PRId64 "): %s\n",
                __func__, (int64_t)(chunk_x*sizeof(float)), cudaGetErrorString(err));
        return 0;
    }
    err = cudaMalloc(&q_dev, n_per_row*sizeof(float));
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: q4_0_imatrix: cudaMalloc(q_dev, %" PRId64 "): %s\n",
                __func__, (int64_t)(n_per_row*sizeof(float)), cudaGetErrorString(err));
        cudaFree(x_dev);
        return 0;
    }
    err = cudaMalloc(&s_dev, nrows*sizeof(float));
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: q4_0_imatrix: cudaMalloc(s_dev, %" PRId64 "): %s\n",
                __func__, (int64_t)(nrows*sizeof(float)), cudaGetErrorString(err));
        cudaFree(x_dev);
        cudaFree(q_dev);
        return 0;
    }
    err = cudaMalloc(&y_dev, chunk_y);
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: q4_0_imatrix: cudaMalloc(y_dev, %" PRId64 "): %s\n",
                __func__, (int64_t)chunk_y, cudaGetErrorString(err));
        cudaFree(x_dev);
        cudaFree(q_dev);
        cudaFree(s_dev);
        return 0;
    }

    err = cudaMemcpy(q_dev, imatrix, n_per_row*sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: q4_0_imatrix: cudaMemcpy imatrix H2D: %s\n", __func__, cudaGetErrorString(err));
        cudaFree(x_dev);
        cudaFree(q_dev);
        cudaFree(s_dev);
        cudaFree(y_dev);
        return 0;
    }
    err = cudaMemcpy(s_dev, sigma2.data(), nrows*sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: q4_0_imatrix: cudaMemcpy sigma2 H2D: %s\n", __func__, cudaGetErrorString(err));
        cudaFree(x_dev);
        cudaFree(q_dev);
        cudaFree(s_dev);
        cudaFree(y_dev);
        return 0;
    }

    for (int64_t base = 0; base < nblocks_total; base += chunk_blocks) {
        const int64_t nblocks = std::min(chunk_blocks, nblocks_total - base);

        err = cudaMemcpy(x_dev, src + base*QK4_0, nblocks*QK4_0*sizeof(float), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            fprintf(stderr, "%s: q4_0_imatrix: cudaMemcpy H2D: %s\n", __func__, cudaGetErrorString(err));
            break;
        }

        // 256 threads per block, one quant block per thread
        const unsigned int block_size = 256;
        const float fudge_q4_0 = ggml_get_quantize_fudge_factor(GGML_TYPE_Q4_0);
        quantize_q4_0_imatrix_kernel<<<(unsigned)((nblocks + block_size - 1)/block_size), block_size>>>(
                x_dev, q_dev, s_dev, y_dev, base, nblocks, blocks_per_row, fudge_q4_0);

        err = cudaMemcpy((char *)dst + base*sizeof(block_q4_0), y_dev, nblocks*sizeof(block_q4_0), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            fprintf(stderr, "%s: q4_0_imatrix: cudaMemcpy D2H: %s\n", __func__, cudaGetErrorString(err));
            break;
        }
    }

    cudaFree(x_dev);
    cudaFree(q_dev);
    cudaFree(s_dev);
    cudaFree(y_dev);

    if (err != cudaSuccess) {
        return 0;
    }

    return nblocks_total*sizeof(block_q4_0);
}

// Q5_0 with an importance matrix. Mirror of ggml_cuda_quantize_q4_0_imatrix:
// same chunked driver, same host-computed row sigma2 in the exact CPU order of
// quantize_row_q5_0_impl (ggml-quants.c:3577).
size_t ggml_cuda_quantize_q5_0_imatrix(const float * src, void * dst, int64_t nrows, int64_t n_per_row,
        const float * imatrix) {
    GGML_ASSERT(nrows > 0);
    GGML_ASSERT(n_per_row % QK5_0 == 0);

    const int64_t nblocks_total = nrows*(n_per_row/QK5_0);
    const int32_t blocks_per_row = (int32_t)(n_per_row/QK5_0);

    int n_devices = 0;
    if (cudaGetDeviceCount(&n_devices) != cudaSuccess || n_devices == 0) {
        return 0;
    }
    if (cudaSetDevice(0) != cudaSuccess) { // POC: device 0 only
        return 0;
    }

    // Per-row sigma2 = sum_x2/n_per_row, summed sequentially in the exact
    // order of quantize_row_q5_0_impl. See the Q4_0 driver for why this sum is
    // computed on the host and not reduced in parallel.
    std::vector<float> sigma2(nrows);
    for (int64_t irow = 0; irow < nrows; ++irow) {
        const float * xr = src + irow*n_per_row;
        float sum_x2 = 0.0f;
        for (int64_t j = 0; j < n_per_row; ++j) {
            sum_x2 += xr[j]*xr[j];
        }
        sigma2[irow] = sum_x2/n_per_row;
    }

    // Fixed-size device chunks, same rationale as the generic driver.
    const int64_t chunk_blocks = 1 << 20;
    const int64_t chunk_x      = chunk_blocks*QK5_0;
    const int64_t chunk_y      = chunk_blocks*sizeof(block_q5_0);

    float   * x_dev = nullptr;
    float   * q_dev = nullptr;
    float   * s_dev = nullptr;
    uint8_t * y_dev = nullptr;

    cudaError_t err = cudaMalloc(&x_dev, chunk_x*sizeof(float));
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: q5_0_imatrix: cudaMalloc(x_dev, %" PRId64 "): %s\n",
                __func__, (int64_t)(chunk_x*sizeof(float)), cudaGetErrorString(err));
        return 0;
    }
    err = cudaMalloc(&q_dev, n_per_row*sizeof(float));
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: q5_0_imatrix: cudaMalloc(q_dev, %" PRId64 "): %s\n",
                __func__, (int64_t)(n_per_row*sizeof(float)), cudaGetErrorString(err));
        cudaFree(x_dev);
        return 0;
    }
    err = cudaMalloc(&s_dev, nrows*sizeof(float));
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: q5_0_imatrix: cudaMalloc(s_dev, %" PRId64 "): %s\n",
                __func__, (int64_t)(nrows*sizeof(float)), cudaGetErrorString(err));
        cudaFree(x_dev);
        cudaFree(q_dev);
        return 0;
    }
    err = cudaMalloc(&y_dev, chunk_y);
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: q5_0_imatrix: cudaMalloc(y_dev, %" PRId64 "): %s\n",
                __func__, (int64_t)chunk_y, cudaGetErrorString(err));
        cudaFree(x_dev);
        cudaFree(q_dev);
        cudaFree(s_dev);
        return 0;
    }

    err = cudaMemcpy(q_dev, imatrix, n_per_row*sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: q5_0_imatrix: cudaMemcpy imatrix H2D: %s\n", __func__, cudaGetErrorString(err));
        cudaFree(x_dev);
        cudaFree(q_dev);
        cudaFree(s_dev);
        cudaFree(y_dev);
        return 0;
    }
    err = cudaMemcpy(s_dev, sigma2.data(), nrows*sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: q5_0_imatrix: cudaMemcpy sigma2 H2D: %s\n", __func__, cudaGetErrorString(err));
        cudaFree(x_dev);
        cudaFree(q_dev);
        cudaFree(s_dev);
        cudaFree(y_dev);
        return 0;
    }

    for (int64_t base = 0; base < nblocks_total; base += chunk_blocks) {
        const int64_t nblocks = std::min(chunk_blocks, nblocks_total - base);

        err = cudaMemcpy(x_dev, src + base*QK5_0, nblocks*QK5_0*sizeof(float), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            fprintf(stderr, "%s: q5_0_imatrix: cudaMemcpy H2D: %s\n", __func__, cudaGetErrorString(err));
            break;
        }

        // 256 threads per block, one quant block per thread
        const unsigned int block_size = 256;
        const float fudge_q5_0 = ggml_get_quantize_fudge_factor(GGML_TYPE_Q5_0);
        quantize_q5_0_imatrix_kernel<<<(unsigned)((nblocks + block_size - 1)/block_size), block_size>>>(
                x_dev, q_dev, s_dev, y_dev, base, nblocks, blocks_per_row, fudge_q5_0);

        err = cudaMemcpy((char *)dst + base*sizeof(block_q5_0), y_dev, nblocks*sizeof(block_q5_0), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            fprintf(stderr, "%s: q5_0_imatrix: cudaMemcpy D2H: %s\n", __func__, cudaGetErrorString(err));
            break;
        }
    }

    cudaFree(x_dev);
    cudaFree(q_dev);
    cudaFree(s_dev);
    cudaFree(y_dev);

    if (err != cudaSuccess) {
        return 0;
    }

    return nblocks_total*sizeof(block_q5_0);
}

// Q6_0 with an importance matrix. Mirror of ggml_cuda_quantize_q5_0_imatrix:
// same chunked driver, same host-computed row sigma2 in the exact CPU order of
// quantize_row_q6_0_impl (ggml-quants.c:3697), one thread per quant block
// replaying make_qx_quants with nmax == 32.
size_t ggml_cuda_quantize_q6_0_imatrix(const float * src, void * dst, int64_t nrows, int64_t n_per_row,
        const float * imatrix) {
    GGML_ASSERT(nrows > 0);
    GGML_ASSERT(n_per_row % QK6_0 == 0);

    const int64_t nblocks_total = nrows*(n_per_row/QK6_0);
    const int32_t blocks_per_row = (int32_t)(n_per_row/QK6_0);

    int n_devices = 0;
    if (cudaGetDeviceCount(&n_devices) != cudaSuccess || n_devices == 0) {
        return 0;
    }
    if (cudaSetDevice(0) != cudaSuccess) { // POC: device 0 only
        return 0;
    }

    // Per-row sigma2 = sum_x2/n_per_row, summed sequentially in the exact
    // order of quantize_row_q6_0_impl. See the Q4_0 driver for why this sum is
    // computed on the host and not reduced in parallel.
    std::vector<float> sigma2(nrows);
    for (int64_t irow = 0; irow < nrows; ++irow) {
        const float * xr = src + irow*n_per_row;
        float sum_x2 = 0.0f;
        for (int64_t j = 0; j < n_per_row; ++j) {
            sum_x2 += xr[j]*xr[j];
        }
        sigma2[irow] = sum_x2/n_per_row;
    }

    // Fixed-size device chunks, same rationale as the generic driver.
    const int64_t chunk_blocks = 1 << 20;
    const int64_t chunk_x      = chunk_blocks*QK6_0;
    const int64_t chunk_y      = chunk_blocks*sizeof(block_q6_0);

    float   * x_dev = nullptr;
    float   * q_dev = nullptr;
    float   * s_dev = nullptr;
    uint8_t * y_dev = nullptr;

    cudaError_t err = cudaMalloc(&x_dev, chunk_x*sizeof(float));
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: q6_0_imatrix: cudaMalloc(x_dev, %" PRId64 "): %s\n",
                __func__, (int64_t)(chunk_x*sizeof(float)), cudaGetErrorString(err));
        return 0;
    }
    err = cudaMalloc(&q_dev, n_per_row*sizeof(float));
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: q6_0_imatrix: cudaMalloc(q_dev, %" PRId64 "): %s\n",
                __func__, (int64_t)(n_per_row*sizeof(float)), cudaGetErrorString(err));
        cudaFree(x_dev);
        return 0;
    }
    err = cudaMalloc(&s_dev, nrows*sizeof(float));
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: q6_0_imatrix: cudaMalloc(s_dev, %" PRId64 "): %s\n",
                __func__, (int64_t)(nrows*sizeof(float)), cudaGetErrorString(err));
        cudaFree(x_dev);
        cudaFree(q_dev);
        return 0;
    }
    err = cudaMalloc(&y_dev, chunk_y);
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: q6_0_imatrix: cudaMalloc(y_dev, %" PRId64 "): %s\n",
                __func__, (int64_t)chunk_y, cudaGetErrorString(err));
        cudaFree(x_dev);
        cudaFree(q_dev);
        cudaFree(s_dev);
        return 0;
    }

    err = cudaMemcpy(q_dev, imatrix, n_per_row*sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: q6_0_imatrix: cudaMemcpy imatrix H2D: %s\n", __func__, cudaGetErrorString(err));
        cudaFree(x_dev);
        cudaFree(q_dev);
        cudaFree(s_dev);
        cudaFree(y_dev);
        return 0;
    }
    err = cudaMemcpy(s_dev, sigma2.data(), nrows*sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: q6_0_imatrix: cudaMemcpy sigma2 H2D: %s\n", __func__, cudaGetErrorString(err));
        cudaFree(x_dev);
        cudaFree(q_dev);
        cudaFree(s_dev);
        cudaFree(y_dev);
        return 0;
    }

    for (int64_t base = 0; base < nblocks_total; base += chunk_blocks) {
        const int64_t nblocks = std::min(chunk_blocks, nblocks_total - base);

        err = cudaMemcpy(x_dev, src + base*QK6_0, nblocks*QK6_0*sizeof(float), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            fprintf(stderr, "%s: q6_0_imatrix: cudaMemcpy H2D: %s\n", __func__, cudaGetErrorString(err));
            break;
        }

        // 256 threads per block, one quant block per thread
        const unsigned int block_size = 256;
        const float fudge_q6_0 = ggml_get_quantize_fudge_factor(GGML_TYPE_Q6_0);
        quantize_q6_0_imatrix_kernel<<<(unsigned)((nblocks + block_size - 1)/block_size), block_size>>>(
                x_dev, q_dev, s_dev, y_dev, base, nblocks, blocks_per_row, fudge_q6_0);

        err = cudaMemcpy((char *)dst + base*sizeof(block_q6_0), y_dev, nblocks*sizeof(block_q6_0), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            fprintf(stderr, "%s: q6_0_imatrix: cudaMemcpy D2H: %s\n", __func__, cudaGetErrorString(err));
            break;
        }
    }

    cudaFree(x_dev);
    cudaFree(q_dev);
    cudaFree(s_dev);
    cudaFree(y_dev);

    if (err != cudaSuccess) {
        return 0;
    }

    return nblocks_total*sizeof(block_q6_0);
}

// Q5_1 without imatrix (quantize_row_q5_1_ref). No fudge.
size_t ggml_cuda_quantize_q5_1(const float * src, void * dst, int64_t nrows, int64_t n_per_row) {
    return ggml_cuda_quantize_generic(src, dst, nrows, n_per_row,
            QK5_1, sizeof(block_q5_1), quantize_q5_1_kernel, "q5_1", 1.0f);
}

// Q5_1 with an importance matrix (quantize_row_q5_1_impl via make_qkx3).
// Same chunked driver shape as the Q5_0 imatrix path, same host-computed row
// sigma2 in the exact CPU summation order. No fudge.
size_t ggml_cuda_quantize_q5_1_imatrix(const float * src, void * dst, int64_t nrows, int64_t n_per_row,
        const float * imatrix) {
    GGML_ASSERT(nrows > 0);
    GGML_ASSERT(n_per_row % QK5_1 == 0);

    const int64_t nblocks_total = nrows*(n_per_row/QK5_1);
    const int32_t blocks_per_row = (int32_t)(n_per_row/QK5_1);

    int n_devices = 0;
    if (cudaGetDeviceCount(&n_devices) != cudaSuccess || n_devices == 0) {
        return 0;
    }
    if (cudaSetDevice(0) != cudaSuccess) { // device 0 only
        return 0;
    }

    std::vector<float> sigma2(nrows);
    for (int64_t irow = 0; irow < nrows; ++irow) {
        const float * xr = src + irow*n_per_row;
        float sum_x2 = 0.0f;
        for (int64_t j = 0; j < n_per_row; ++j) {
            sum_x2 += xr[j]*xr[j];
        }
        sigma2[irow] = sum_x2/n_per_row;
    }

    const int64_t chunk_blocks = 1 << 20;
    const int64_t chunk_x      = chunk_blocks*QK5_1;
    const int64_t chunk_y      = chunk_blocks*sizeof(block_q5_1);

    float   * x_dev = nullptr;
    float   * q_dev = nullptr;
    float   * s_dev = nullptr;
    uint8_t * y_dev = nullptr;

    cudaError_t err = cudaMalloc(&x_dev, chunk_x*sizeof(float));
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: q5_1_imatrix: cudaMalloc(x_dev): %s\n", __func__, cudaGetErrorString(err));
        return 0;
    }
    err = cudaMalloc(&q_dev, n_per_row*sizeof(float));
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: q5_1_imatrix: cudaMalloc(q_dev): %s\n", __func__, cudaGetErrorString(err));
        cudaFree(x_dev);
        return 0;
    }
    err = cudaMalloc(&s_dev, nrows*sizeof(float));
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: q5_1_imatrix: cudaMalloc(s_dev): %s\n", __func__, cudaGetErrorString(err));
        cudaFree(x_dev);
        cudaFree(q_dev);
        return 0;
    }
    err = cudaMalloc(&y_dev, chunk_y);
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: q5_1_imatrix: cudaMalloc(y_dev): %s\n", __func__, cudaGetErrorString(err));
        cudaFree(x_dev);
        cudaFree(q_dev);
        cudaFree(s_dev);
        return 0;
    }

    err = cudaMemcpy(q_dev, imatrix, n_per_row*sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: q5_1_imatrix: cudaMemcpy imatrix H2D: %s\n", __func__, cudaGetErrorString(err));
        cudaFree(x_dev);
        cudaFree(q_dev);
        cudaFree(s_dev);
        cudaFree(y_dev);
        return 0;
    }
    err = cudaMemcpy(s_dev, sigma2.data(), nrows*sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: q5_1_imatrix: cudaMemcpy sigma2 H2D: %s\n", __func__, cudaGetErrorString(err));
        cudaFree(x_dev);
        cudaFree(q_dev);
        cudaFree(s_dev);
        cudaFree(y_dev);
        return 0;
    }

    const unsigned int block_size = 256;
    for (int64_t base = 0; base < nblocks_total; base += chunk_blocks) {
        const int64_t nblocks = std::min(chunk_blocks, nblocks_total - base);

        err = cudaMemcpy(x_dev, src + base*QK5_1, nblocks*QK5_1*sizeof(float), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            fprintf(stderr, "%s: q5_1_imatrix: cudaMemcpy H2D: %s\n", __func__, cudaGetErrorString(err));
            break;
        }

        quantize_q5_1_imatrix_kernel<<<(unsigned)((nblocks + block_size - 1)/block_size), block_size>>>(
                x_dev, q_dev, s_dev, y_dev, base, nblocks, blocks_per_row);

        err = cudaMemcpy((char *)dst + base*sizeof(block_q5_1), y_dev, nblocks*sizeof(block_q5_1), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            fprintf(stderr, "%s: q5_1_imatrix: cudaMemcpy D2H: %s\n", __func__, cudaGetErrorString(err));
            break;
        }
    }

    cudaFree(x_dev);
    cudaFree(q_dev);
    cudaFree(s_dev);
    cudaFree(y_dev);

    if (err != cudaSuccess) {
        return 0;
    }

    return nblocks_total*sizeof(block_q5_1);
}

// Q4_1 with an importance matrix (quantize_row_q4_1_impl via make_qkx3).
// Same chunked driver shape as Q5_1 imatrix. No fudge.
size_t ggml_cuda_quantize_q4_1_imatrix(const float * src, void * dst, int64_t nrows, int64_t n_per_row,
        const float * imatrix) {
    GGML_ASSERT(nrows > 0);
    GGML_ASSERT(n_per_row % QK4_1 == 0);

    const int64_t nblocks_total = nrows*(n_per_row/QK4_1);
    const int32_t blocks_per_row = (int32_t)(n_per_row/QK4_1);

    int n_devices = 0;
    if (cudaGetDeviceCount(&n_devices) != cudaSuccess || n_devices == 0) {
        return 0;
    }
    if (cudaSetDevice(0) != cudaSuccess) { // device 0 only
        return 0;
    }

    std::vector<float> sigma2(nrows);
    for (int64_t irow = 0; irow < nrows; ++irow) {
        const float * xr = src + irow*n_per_row;
        float sum_x2 = 0.0f;
        for (int64_t j = 0; j < n_per_row; ++j) {
            sum_x2 += xr[j]*xr[j];
        }
        sigma2[irow] = sum_x2/n_per_row;
    }

    const int64_t chunk_blocks = 1 << 20;
    const int64_t chunk_x      = chunk_blocks*QK4_1;
    const int64_t chunk_y      = chunk_blocks*sizeof(block_q4_1);

    float   * x_dev = nullptr;
    float   * q_dev = nullptr;
    float   * s_dev = nullptr;
    uint8_t * y_dev = nullptr;

    cudaError_t err = cudaMalloc(&x_dev, chunk_x*sizeof(float));
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: q4_1_imatrix: cudaMalloc(x_dev): %s\n", __func__, cudaGetErrorString(err));
        return 0;
    }
    err = cudaMalloc(&q_dev, n_per_row*sizeof(float));
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: q4_1_imatrix: cudaMalloc(q_dev): %s\n", __func__, cudaGetErrorString(err));
        cudaFree(x_dev);
        return 0;
    }
    err = cudaMalloc(&s_dev, nrows*sizeof(float));
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: q4_1_imatrix: cudaMalloc(s_dev): %s\n", __func__, cudaGetErrorString(err));
        cudaFree(x_dev);
        cudaFree(q_dev);
        return 0;
    }
    err = cudaMalloc(&y_dev, chunk_y);
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: q4_1_imatrix: cudaMalloc(y_dev): %s\n", __func__, cudaGetErrorString(err));
        cudaFree(x_dev);
        cudaFree(q_dev);
        cudaFree(s_dev);
        return 0;
    }

    err = cudaMemcpy(q_dev, imatrix, n_per_row*sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: q4_1_imatrix: cudaMemcpy imatrix H2D: %s\n", __func__, cudaGetErrorString(err));
        cudaFree(x_dev);
        cudaFree(q_dev);
        cudaFree(s_dev);
        cudaFree(y_dev);
        return 0;
    }
    err = cudaMemcpy(s_dev, sigma2.data(), nrows*sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: q4_1_imatrix: cudaMemcpy sigma2 H2D: %s\n", __func__, cudaGetErrorString(err));
        cudaFree(x_dev);
        cudaFree(q_dev);
        cudaFree(s_dev);
        cudaFree(y_dev);
        return 0;
    }

    const unsigned int block_size = 256;
    for (int64_t base = 0; base < nblocks_total; base += chunk_blocks) {
        const int64_t nblocks = std::min(chunk_blocks, nblocks_total - base);

        err = cudaMemcpy(x_dev, src + base*QK4_1, nblocks*QK4_1*sizeof(float), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            fprintf(stderr, "%s: q4_1_imatrix: cudaMemcpy H2D: %s\n", __func__, cudaGetErrorString(err));
            break;
        }

        quantize_q4_1_imatrix_kernel<<<(unsigned)((nblocks + block_size - 1)/block_size), block_size>>>(
                x_dev, q_dev, s_dev, y_dev, base, nblocks, blocks_per_row);

        err = cudaMemcpy((char *)dst + base*sizeof(block_q4_1), y_dev, nblocks*sizeof(block_q4_1), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            fprintf(stderr, "%s: q4_1_imatrix: cudaMemcpy D2H: %s\n", __func__, cudaGetErrorString(err));
            break;
        }
    }

    cudaFree(x_dev);
    cudaFree(q_dev);
    cudaFree(s_dev);
    cudaFree(y_dev);

    if (err != cudaSuccess) {
        return 0;
    }

    return nblocks_total*sizeof(block_q4_1);
}

// Q6_0 OLS without imatrix (quantize_row_q6_0_impl, quant_weights == NULL).
// OLS is KEPT: weight = x*x, d = make_qx_quants * fudge. Chunked, one thread
// per block, 256 threads per launch block.
size_t ggml_cuda_quantize_q6_0(const float * src, void * dst, int64_t nrows, int64_t n_per_row) {
    GGML_ASSERT(nrows > 0);
    GGML_ASSERT(n_per_row % QK6_0 == 0);

    const int64_t nblocks_total = nrows*(n_per_row/QK6_0);

    int n_devices = 0;
    if (cudaGetDeviceCount(&n_devices) != cudaSuccess || n_devices == 0) {
        return 0;
    }
    if (cudaSetDevice(0) != cudaSuccess) { // device 0 only (matches dispatcher)
        return 0;
    }

    const float fudge = ggml_get_quantize_fudge_factor(GGML_TYPE_Q6_0);

    const int64_t chunk_blocks = 1 << 20;
    const int64_t chunk_x      = chunk_blocks*QK6_0;
    const int64_t chunk_y      = chunk_blocks*sizeof(block_q6_0);

    float   * x_dev = nullptr;
    uint8_t * y_dev = nullptr;

    cudaError_t err = cudaMalloc(&x_dev, chunk_x*sizeof(float));
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: q6_0_ols: cudaMalloc(x_dev): %s\n", __func__, cudaGetErrorString(err));
        return 0;
    }
    err = cudaMalloc(&y_dev, chunk_y);
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: q6_0_ols: cudaMalloc(y_dev): %s\n", __func__, cudaGetErrorString(err));
        cudaFree(x_dev);
        return 0;
    }

    const unsigned int block_size = 256;
    for (int64_t base = 0; base < nblocks_total; base += chunk_blocks) {
        const int64_t nblocks = std::min(chunk_blocks, nblocks_total - base);

        err = cudaMemcpy(x_dev, src + base*QK6_0, nblocks*QK6_0*sizeof(float), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            fprintf(stderr, "%s: q6_0_ols: cudaMemcpy H2D: %s\n", __func__, cudaGetErrorString(err));
            break;
        }

        quantize_q6_0_ols_kernel<<<(unsigned)((nblocks + block_size - 1)/block_size), block_size>>>(
                x_dev, y_dev, nblocks, fudge);

        err = cudaMemcpy((char *)dst + base*sizeof(block_q6_0), y_dev, nblocks*sizeof(block_q6_0), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            fprintf(stderr, "%s: q6_0_ols: cudaMemcpy D2H: %s\n", __func__, cudaGetErrorString(err));
            break;
        }
    }

    cudaFree(x_dev);
    cudaFree(y_dev);

    if (err != cudaSuccess) {
        return 0;
    }

    return nblocks_total*sizeof(block_q6_0);
}

// Q8_0 with an importance matrix. HEAD CPU quantize_q8_0 ignores imatrix, so
// this helper routes to plain (same bytes). The weighted-LS kernel below is
// kept for research but UNUSED for GGUF.
size_t ggml_cuda_quantize_q8_0_imatrix(const float * src, void * dst, int64_t nrows, int64_t n_per_row,
        const float * imatrix) {
    (void) imatrix; // CPU ignores it; match exactly
    return ggml_cuda_quantize_q8_0(src, dst, nrows, n_per_row);
}

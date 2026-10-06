#pragma once

// quillama.cpp: Intel AMX (Advanced Matrix Extensions) BF16 GEMM for CPU prompt processing.
//
// Weights of any type that has a to_float() (all ggml / ik quants except the row-interleaved
// *_R4/_R8/_R16 repacks) are dequantized panel by panel to BF16 and multiplied with the
// activations on the AMX tile unit. Used only for large batches (prompt processing), where
// the matmuls are compute bound; token generation keeps ik's memory-bound kernels.
//
// Disable at run time with QUILL_AMX=0; QUILL_AMX_MIN_BATCH sets the smallest batch (default 32).

#include "ggml.h"

#include <stddef.h>
#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// true when the CPU has AMX-BF16 + AVX512-BF16, the OS granted tile state, and QUILL_AMX != 0
bool quill_amx_available(void);

// can this weight type be dequantized row-wise to BF16 for AMX?
bool quill_amx_type_supported(enum ggml_type type);

// rows interleaved by a row-repacked type (*_R4/_R8/_R16), else 1
int64_t quill_amx_interleave(enum ggml_type type);

// minimum number of activation rows (tokens) for which AMX is used
int  quill_amx_min_batch(void);

// bytes of work buffer needed to hold T activation rows of length K packed for AMX
size_t quill_amx_packed_size(int64_t K, int64_t T);

// pack T rows of f32 activations (row i at x + i*x_stride bytes) into the AMX B-operand layout.
// Rows are taken from row_ids[i] when row_ids != NULL (gather). Work is split over (ith, nth).
void quill_amx_pack(const void * x, size_t x_stride, const int32_t * row_ids, int64_t K, int64_t T,
                    void * packed, int ith, int nth);

// pack only 16-row group g (rows 16g .. 16g+15, zero-padded past T) of the above
void quill_amx_pack_group(const void * x, size_t x_stride, const int32_t * row_ids, int64_t K, int64_t T,
                          void * packed, int64_t g);

// dst[t][m] = sum_k W[m][k] * X[t][k] for m in [0, M), t in [0, T).
// W rows of type `type` with stride w_stride bytes; dst row t at dst + t*dst_stride bytes
// (or at dst_rows[t] when dst_rows != NULL, scatter). Work is split over (ith, nth).
void quill_amx_gemm(enum ggml_type type, const void * w, size_t w_stride,
                    int64_t K, int64_t M, int64_t T, const void * packed,
                    float * dst, size_t dst_stride, float * const * dst_rows,
                    int ith, int nth);

// fused FFN: dst = act(gate . x) * (up . x), act in {SILU, RELU, GELU}, optional clamp limit as in ik.
// Returns false (nothing done) for unsupported activations.
bool quill_amx_gemm_up_gate(enum ggml_type type, const void * w_up, const void * w_gate, size_t w_stride,
                            int64_t K, int64_t M, int64_t T, const void * packed,
                            float * dst, size_t dst_stride, float * const * dst_rows,
                            int unary_op, float limit, int ith, int nth);

// flash attention for prompt processing (F32 Q; F16/BF16/Q8_0/F32 K and V; F16 mask; no ALiBi/sinks)
bool quill_amx_flash_attn_eligible(const struct ggml_tensor * dst);
void quill_amx_flash_attn(struct ggml_tensor * dst, int ith, int nth);

#ifdef __cplusplus
}
#endif

#!/usr/bin/env python3
"""
TDD-driven tests for PR #1654 corrections to the Qwen3-Next / Qwen3.5 converter.

Bug 1: QKVZ slicing in convert_hf_to_gguf.py
  HF stores QKVZ grouped per-K-head:
      [G0_q, G0_k, G0_v_0..v_{r-1}, G0_z_0..z_{r-1},
       G1_q, G1_k, G1_v_0..v_{r-1}, G1_z_0..z_{r-1}, ...]
  where r = num_v_per_k. The current converter does flat-block slicing
  ([Q_block; K_block; V_block; Z_block]) which is wrong, and additionally
  has an n_z size typo at line 2347.

Bug 2: in_proj_ba mapping
  Should map to a SINGLE SSM_BETA_ALPHA tensor per upstream
  tensor_mapping.py:864-866. The current converter splits BA into b_t/a_t.

Bug 3: quants.py 1/d math
  np.where(d == 0, 0, 1/d) is NOT lazy — `1/d` is fully evaluated for ALL
  elements, including subnormal d (e.g. 1e-40) where 1/d overflows to inf
  and propagates through np.where (because subnormal != 0). Downstream
  inf*0 = NaN, then astype(uint8) raises "invalid value in cast" or
  produces garbage. Fix: guard with abs(d) <= eps.

These tests intentionally use stdlib unittest + numpy only — no torch, no
pytest — because the box running the test suite has neither installed.
"""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

import numpy as np

# Make the local gguf package importable.
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "gguf-py"))


# ---------------------------------------------------------------------------
# Bug 1 — QKVZ per-K-head split contract
# ---------------------------------------------------------------------------

def _build_synthetic_qkvz(num_k_heads: int, head_k_dim: int, num_v_per_k: int,
                          head_v_dim: int, hidden: int) -> np.ndarray:
    """
    Build a synthetic in_proj_qkvz weight in HF's per-K-head layout:

        For each K-head g in 0..num_k_heads-1:
            q rows: head_k_dim rows tagged value 1000*(g+1) + r,         r in [0, head_k_dim)
            k rows: head_k_dim rows tagged value 2000*(g+1) + r
            v rows: num_v_per_k * head_v_dim rows tagged 3000*(g+1) + idx
            z rows: num_v_per_k * head_v_dim rows tagged 4000*(g+1) + idx

    The total row count matches HF's actual layout for Qwen3-Next:
        rows = num_k_heads * (2*head_k_dim + 2*num_v_per_k*head_v_dim)
             = 2*head_k_dim*num_k_heads + 2*num_v_heads*head_v_dim
    """
    per_group = 2 * head_k_dim + 2 * num_v_per_k * head_v_dim
    total_rows = num_k_heads * per_group
    out = np.zeros((total_rows, hidden), dtype=np.float32)
    row = 0
    for g in range(num_k_heads):
        for r in range(head_k_dim):
            out[row, :] = 1000 * (g + 1) + r
            row += 1
        for r in range(head_k_dim):
            out[row, :] = 2000 * (g + 1) + r
            row += 1
        for idx in range(num_v_per_k * head_v_dim):
            out[row, :] = 3000 * (g + 1) + idx
            row += 1
        for idx in range(num_v_per_k * head_v_dim):
            out[row, :] = 4000 * (g + 1) + idx
            row += 1
    assert row == total_rows
    return out


def _expected_q_block(num_k_heads, head_k_dim, hidden):
    rows = []
    for g in range(num_k_heads):
        for r in range(head_k_dim):
            rows.append(np.full((hidden,), 1000 * (g + 1) + r, dtype=np.float32))
    return np.stack(rows, axis=0)


def _expected_k_block(num_k_heads, head_k_dim, hidden):
    rows = []
    for g in range(num_k_heads):
        for r in range(head_k_dim):
            rows.append(np.full((hidden,), 2000 * (g + 1) + r, dtype=np.float32))
    return np.stack(rows, axis=0)


def _expected_v_block(num_k_heads, num_v_per_k, head_v_dim, hidden):
    rows = []
    for g in range(num_k_heads):
        for idx in range(num_v_per_k * head_v_dim):
            rows.append(np.full((hidden,), 3000 * (g + 1) + idx, dtype=np.float32))
    return np.stack(rows, axis=0)


def _expected_z_block(num_k_heads, num_v_per_k, head_v_dim, hidden):
    rows = []
    for g in range(num_k_heads):
        for idx in range(num_v_per_k * head_v_dim):
            rows.append(np.full((hidden,), 4000 * (g + 1) + idx, dtype=np.float32))
    return np.stack(rows, axis=0)


class Qwen3NextQKVZSplitTests(unittest.TestCase):
    """Bug 1: verify per-K-head QKVZ split is correct."""

    NUM_K_HEADS = 2
    HEAD_K_DIM = 4
    NUM_V_PER_K = 2
    HEAD_V_DIM = 4
    HIDDEN = 8

    def test_qkvz_split_matches_per_head_groupings(self):
        """
        The split helper should pull Q/K/V/Z out of the per-K-head-grouped
        HF layout and return them in the canonical
        [head0_q, head1_q, ...; head0_k, head1_k, ...; etc] flat order.
        """
        from gguf.qwen3next_split import split_qkvz

        qkvz = _build_synthetic_qkvz(
            num_k_heads=self.NUM_K_HEADS,
            head_k_dim=self.HEAD_K_DIM,
            num_v_per_k=self.NUM_V_PER_K,
            head_v_dim=self.HEAD_V_DIM,
            hidden=self.HIDDEN,
        )

        q, k, v, z = split_qkvz(
            qkvz,
            num_k_heads=self.NUM_K_HEADS,
            head_k_dim=self.HEAD_K_DIM,
            num_v_heads=self.NUM_K_HEADS * self.NUM_V_PER_K,
            head_v_dim=self.HEAD_V_DIM,
        )

        np.testing.assert_array_equal(
            q, _expected_q_block(self.NUM_K_HEADS, self.HEAD_K_DIM, self.HIDDEN),
            err_msg="Q block: per-head grouping not preserved",
        )
        np.testing.assert_array_equal(
            k, _expected_k_block(self.NUM_K_HEADS, self.HEAD_K_DIM, self.HIDDEN),
            err_msg="K block: per-head grouping not preserved",
        )
        np.testing.assert_array_equal(
            v, _expected_v_block(self.NUM_K_HEADS, self.NUM_V_PER_K, self.HEAD_V_DIM, self.HIDDEN),
            err_msg="V block: per-head grouping not preserved",
        )
        np.testing.assert_array_equal(
            z, _expected_z_block(self.NUM_K_HEADS, self.NUM_V_PER_K, self.HEAD_V_DIM, self.HIDDEN),
            err_msg="Z block: per-head grouping not preserved",
        )

    def test_qkvz_split_z_size_matches_v_size(self):
        """
        Z must have shape (num_v_heads * head_v_dim, hidden) — same as V.
        The buggy n_z = data_torch.shape[1] (= hidden) is wrong; this test
        catches the size typo at line 2347.
        """
        from gguf.qwen3next_split import split_qkvz

        qkvz = _build_synthetic_qkvz(
            num_k_heads=self.NUM_K_HEADS,
            head_k_dim=self.HEAD_K_DIM,
            num_v_per_k=self.NUM_V_PER_K,
            head_v_dim=self.HEAD_V_DIM,
            hidden=self.HIDDEN,
        )

        _, _, v, z = split_qkvz(
            qkvz,
            num_k_heads=self.NUM_K_HEADS,
            head_k_dim=self.HEAD_K_DIM,
            num_v_heads=self.NUM_K_HEADS * self.NUM_V_PER_K,
            head_v_dim=self.HEAD_V_DIM,
        )
        self.assertEqual(z.shape, v.shape, "Z block must be the same shape as V block")
        expected_v_rows = self.NUM_K_HEADS * self.NUM_V_PER_K * self.HEAD_V_DIM
        self.assertEqual(v.shape[0], expected_v_rows)
        self.assertEqual(z.shape[0], expected_v_rows)


# ---------------------------------------------------------------------------
# Bug 2 — SSM_BETA_ALPHA tensor mapping
# ---------------------------------------------------------------------------

class SSMBetaAlphaMappingTests(unittest.TestCase):
    """Bug 2: linear_attn.in_proj_ba should land as a single combined tensor."""

    def test_ssm_beta_alpha_enum_exists(self):
        from gguf.constants import MODEL_TENSOR
        self.assertTrue(
            hasattr(MODEL_TENSOR, "SSM_BETA_ALPHA"),
            "MODEL_TENSOR.SSM_BETA_ALPHA must exist for combined in_proj_ba mapping",
        )

    def test_ssm_beta_alpha_has_canonical_name(self):
        from gguf.constants import MODEL_TENSOR, TENSOR_NAMES
        self.assertIn(MODEL_TENSOR.SSM_BETA_ALPHA, TENSOR_NAMES)
        self.assertEqual(
            TENSOR_NAMES[MODEL_TENSOR.SSM_BETA_ALPHA],
            "blk.{bid}.ssm_ba",
            "SSM_BETA_ALPHA must serialize as blk.{bid}.ssm_ba per upstream",
        )

    def test_in_proj_ba_maps_to_ssm_beta_alpha(self):
        """
        Per upstream tensor_mapping.py:864-866, `linear_attn.in_proj_ba` must
        map to MODEL_TENSOR.SSM_BETA_ALPHA — not split into b_t/a_t.
        """
        from gguf.constants import MODEL_TENSOR
        from gguf.tensor_mapping import TensorNameMap, get_tensor_name_map

        # Use Qwen3Next arch (the only one that ships in_proj_ba).
        from gguf.constants import MODEL_ARCH
        tmap = get_tensor_name_map(MODEL_ARCH.QWEN3NEXT, n_blocks=4)

        mapped = tmap.get_type_and_name("model.layers.0.linear_attn.in_proj_ba.weight")
        self.assertIsNotNone(mapped, "in_proj_ba.weight must be mapped")
        ttype, name = mapped
        self.assertEqual(
            ttype,
            MODEL_TENSOR.SSM_BETA_ALPHA,
            f"in_proj_ba should map to SSM_BETA_ALPHA, got {ttype}",
        )
        self.assertEqual(name, "blk.0.ssm_ba.weight")


# ---------------------------------------------------------------------------
# Bug 3 — quants.py 1/d subnormal-overflow guard
# ---------------------------------------------------------------------------

class QuantsSubnormalGuardTests(unittest.TestCase):
    """Bug 3: subnormal d must not produce inf/NaN in the quantize path."""

    def _run_quant(self, qtype_name):
        """
        Build a synthetic blocks array where one row's max yields a subnormal
        d (e.g. tiny weights) that would overflow under 1/d but not be
        caught by `d == 0`. Verify the quantize path produces no inf/NaN.
        """
        from gguf.constants import GGMLQuantizationType, GGML_QUANT_SIZES
        from gguf.quants import quantize

        qtype = getattr(GGMLQuantizationType, qtype_name)
        block_size, _ = GGML_QUANT_SIZES[qtype]

        # Three rows:
        #   row 0: all tiny subnormal-ish values → d will be subnormal
        #   row 1: all zero                    → d == 0 (already handled)
        #   row 2: normal values               → d normal
        rows = np.zeros((3, block_size), dtype=np.float32)
        rows[0, :] = 1e-40       # subnormal float32; |1/d| ≈ 1e40 → finite but huge
        rows[0, 0] = 5e-40       # ensure at least one nonzero entry per row
        rows[1, :] = 0.0
        rows[2, :] = 1.0

        # quantize() raises "invalid value encountered in cast" if any inf/NaN
        # leaks through. Capture FloatingPointError too via errstate.
        with np.errstate(invalid="raise", over="raise"):
            out = quantize(rows, qtype)

        # Output must be uint8 bytes, no NaN possible by dtype, but verify shape
        self.assertEqual(out.dtype, np.uint8)
        # Re-verify: dequantize and ensure no inf/NaN propagated. Skip qtypes
        # that don't ship a Python dequantize_blocks (Q6_0 is pre-existing
        # quantize-only); the quantize-without-overflow assertion above is
        # already what the fix guarantees.
        from gguf.quants import dequantize
        try:
            deq = dequantize(out, qtype)
        except NotImplementedError:
            return
        self.assertFalse(np.any(np.isnan(deq)), f"{qtype_name}: NaN leaked into dequantized output")
        self.assertFalse(np.any(np.isinf(deq)), f"{qtype_name}: Inf leaked into dequantized output")

    def test_q4_0_handles_subnormal_d(self):
        self._run_quant("Q4_0")

    def test_q4_1_handles_subnormal_d(self):
        self._run_quant("Q4_1")

    def test_q5_0_handles_subnormal_d(self):
        self._run_quant("Q5_0")

    def test_q5_1_handles_subnormal_d(self):
        self._run_quant("Q5_1")

    def test_q6_0_handles_subnormal_d(self):
        self._run_quant("Q6_0")

    def test_q8_0_handles_subnormal_d(self):
        self._run_quant("Q8_0")


if __name__ == "__main__":
    unittest.main(verbosity=2)

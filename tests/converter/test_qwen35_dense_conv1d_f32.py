#!/usr/bin/env python3
"""
TDD test: Qwen3_5TextModel must force the SSM conv1d weight to F32.

The CPU kernel ggml_compute_forward_ssm_conv_f32 asserts that the conv1d
weight tensor is contiguous F32 (nb[0] == sizeof(float)). HF ships the
Qwen3.6 dense (27B) conv1d weight as bf16, so a GGUF whose conv1d weight
follows --outtype (bf16) aborts on the first decode:

    GGML_ASSERT(src2->nb[0] == sizeof(float)) failed

The MoE sibling Qwen3_5MoeTextModel already overrides tensor_force_quant to
return GGMLQuantizationType.F32 for MODEL_TENSOR.SSM_CONV1D. The dense
Qwen3_5TextModel had no such override, so its conv1d weight was emitted at
bf16 and the resulting GGUF crashed at runtime.

This test asserts the dense class carries the same conv1d F32 force.

AST-only — no torch, no checkpoint.
"""
from __future__ import annotations

import ast
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CONVERT = REPO_ROOT / "convert_hf_to_gguf.py"


def _find_class(tree: ast.AST, name: str) -> ast.ClassDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise AssertionError(f"class {name} not found in {CONVERT}")


def _find_method(cls: ast.ClassDef, name: str):
    for node in cls.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    return None


class Qwen35DenseConv1dF32Tests(unittest.TestCase):
    """Qwen3_5TextModel.tensor_force_quant must force conv1d weight to F32."""

    @classmethod
    def setUpClass(cls):
        with open(CONVERT) as f:
            cls.tree = ast.parse(f.read())
        cls.dense = _find_class(cls.tree, "Qwen3_5TextModel")

    def test_dense_forces_conv1d_weight_to_f32(self):
        """Qwen3_5TextModel must override tensor_force_quant and return
        GGMLQuantizationType.F32 for MODEL_TENSOR.SSM_CONV1D. Without it the
        dense conv1d weight follows --outtype (bf16) and the GGUF aborts in
        ggml_compute_forward_ssm_conv_f32 on the first decode."""
        m = _find_method(self.dense, "tensor_force_quant")
        self.assertIsNotNone(
            m,
            "Qwen3_5TextModel.tensor_force_quant not found — the dense conv1d "
            "weight is not forced to F32, so the GGUF crashes at runtime in "
            "ggml_compute_forward_ssm_conv_f32.",
        )
        body = ast.unparse(m)
        self.assertIn(
            "SSM_CONV1D",
            body,
            "Qwen3_5TextModel.tensor_force_quant does not reference "
            "MODEL_TENSOR.SSM_CONV1D — the conv1d weight will not be forced F32.",
        )
        self.assertIn(
            "F32",
            body,
            "Qwen3_5TextModel.tensor_force_quant does not return "
            "GGMLQuantizationType.F32 for the conv1d weight.",
        )

    def test_dense_tensor_force_quant_preserves_super(self):
        """The override must fall through to super().tensor_force_quant for
        all other tensors, otherwise it strips force-quant logic the base
        class applies (norms, 1-D tensors, etc.)."""
        m = _find_method(self.dense, "tensor_force_quant")
        self.assertIsNotNone(m, "Qwen3_5TextModel.tensor_force_quant not found.")
        body = ast.unparse(m)
        self.assertIn(
            "super().tensor_force_quant",
            body,
            "Qwen3_5TextModel.tensor_force_quant must end with "
            "super().tensor_force_quant(...) so non-conv1d tensors keep the "
            "base class force-quant behaviour.",
        )


if __name__ == "__main__":
    unittest.main()

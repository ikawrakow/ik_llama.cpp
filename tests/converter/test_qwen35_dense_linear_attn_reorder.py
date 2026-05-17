#!/usr/bin/env python3
"""
TDD test: Qwen3_5TextModel.modify_tensors must V-head-reorder every
linear-attention projection tensor.

Qwen3.6-27B's linear-attention layers are asymmetric: 16 key heads, 48
value heads. HF stores V heads grouped-by-K-head; the ggml runtime's
shared delta_net::build_layer_attn_linear expects them tiled. The
reference Qwen3NextModel.modify_tensors (used by the working
Qwen3_5MoeTextModel) reorders five projection tensors via
_reorder_v_heads:

    in_proj_qkv (V part), in_proj_z, in_proj_a, in_proj_b, out_proj

Qwen3_5TextModel (dense) re-implements modify_tensors from a Qwen2Model
base and originally reordered only A_log / dt_bias / conv1d -- it had no
branch for the five projection tensors, so they were emitted in HF's
grouped V-head order. 48 of 64 layers then computed linear attention on
misaligned V weights and decode produced garbage / invalid UTF-8.

This test asserts the dense modify_tensors dispatches every V-space
projection so the reorder is applied.

AST-only -- no torch, no checkpoint.
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


# Linear-attention projection tensors whose V heads must be reordered from
# HF's grouped-by-K-head layout to the runtime's tiled layout.
V_SPACE_PROJECTIONS = [
    ".linear_attn.in_proj_qkv.weight",
    ".linear_attn.in_proj_z.weight",
    ".linear_attn.in_proj_a.weight",
    ".linear_attn.in_proj_b.weight",
    ".linear_attn.out_proj.weight",
]


class Qwen35DenseLinearAttnReorderTests(unittest.TestCase):
    """Qwen3_5TextModel.modify_tensors must reorder every linear-attn projection."""

    @classmethod
    def setUpClass(cls):
        raw = CONVERT.read_text()
        tree = ast.parse(raw)
        dense = _find_class(tree, "Qwen3_5TextModel")
        mt = _find_method(dense, "modify_tensors")
        assert mt is not None, "Qwen3_5TextModel.modify_tensors not found"
        lines = raw.splitlines(keepends=True)
        cls.method_src = "".join(lines[mt.lineno - 1:mt.end_lineno])

    def test_modify_tensors_dispatches_every_v_space_projection(self):
        """Each linear-attn V-space projection must have a dispatch branch in
        Qwen3_5TextModel.modify_tensors. A missing branch means the tensor is
        emitted in HF's grouped V-head order, misaligning the runtime
        delta_net and producing garbage decode output."""
        for tensor in V_SPACE_PROJECTIONS:
            with self.subTest(tensor=tensor):
                self.assertIn(
                    tensor,
                    self.method_src,
                    f"Qwen3_5TextModel.modify_tensors has no branch for "
                    f"'{tensor}' -- it is emitted in HF grouped V-head order, "
                    f"misaligning the runtime delta_net linear-attention and "
                    f"producing garbage decode output.",
                )


if __name__ == "__main__":
    unittest.main()

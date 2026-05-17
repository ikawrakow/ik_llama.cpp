#!/usr/bin/env python3
"""
TDD test: Qwen3_5TextModel.modify_tensors must V-head-reorder every
linear-attention projection tensor, with the correct reorder arguments.

Qwen3.6-27B's linear-attention layers are asymmetric: 16 key heads, 48
value heads. HF stores V heads grouped-by-K-head; the ggml runtime's
shared delta_net::build_layer_attn_linear expects them tiled. The
reference Qwen3NextModel.modify_tensors (used by the working
Qwen3_5MoeTextModel) reorders five projection tensors via
_reorder_v_heads:

    in_proj_qkv (V part), in_proj_z, in_proj_a, in_proj_b, out_proj

Qwen3_5TextModel (dense) re-implements modify_tensors from a Qwen2Model
base. A missing branch -- or a branch wired with the wrong reorder axis
(dim) or head_dim -- emits V weights in the wrong layout and decode
produces garbage / invalid UTF-8.

This test verifies, per tensor, that the branch exists AND calls
_reorder_v_heads with the correct dim and head_dim. It also verifies the
dense converter fails loudly on the *fused* linear-attn layout it does
not support, rather than silently emitting un-reordered tensors.

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


def _find_branch(method: ast.FunctionDef, tensor: str):
    """Return the If node in `method` whose test references `tensor`."""
    for node in ast.walk(method):
        if isinstance(node, ast.If) and tensor in ast.unparse(node.test):
            return node
    return None


def _reorder_calls_in_body(if_node: ast.If):
    """_reorder_v_heads calls in the branch BODY only (not its orelse, which
    is the next elif arm)."""
    calls = []
    for stmt in if_node.body:
        for n in ast.walk(stmt):
            if (
                isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute)
                and n.func.attr == "_reorder_v_heads"
            ):
                calls.append(n)
    return calls


# tensor -> (expected reorder dim, expected head_dim arg as source text).
# dim 0 reorders rows (output channels); dim 1 reorders columns (out_proj's
# V-space input). head_dim is head_v_dim for V-channel tensors and 1 for the
# per-head-scalar in_proj_a / in_proj_b.
EXPECTED = {
    ".linear_attn.in_proj_qkv.weight": ("0", "head_v_dim"),
    ".linear_attn.in_proj_z.weight": ("0", "head_v_dim"),
    ".linear_attn.in_proj_a.weight": ("0", "1"),
    ".linear_attn.in_proj_b.weight": ("0", "1"),
    ".linear_attn.out_proj.weight": ("1", "head_v_dim"),
}

# Fused linear-attn layouts the dense converter does NOT handle. A checkpoint
# shipping these must fail loudly, not fall through un-reordered.
FUSED_LAYOUTS = [
    ".linear_attn.in_proj_qkvz.weight",
    ".linear_attn.in_proj_ba.weight",
]


class Qwen35DenseLinearAttnReorderTests(unittest.TestCase):
    """Qwen3_5TextModel.modify_tensors must reorder every linear-attn projection."""

    @classmethod
    def setUpClass(cls):
        tree = ast.parse(CONVERT.read_text())
        dense = _find_class(tree, "Qwen3_5TextModel")
        mt = _find_method(dense, "modify_tensors")
        assert mt is not None, "Qwen3_5TextModel.modify_tensors not found"
        cls.mt = mt

    def test_each_v_space_branch_reorders_with_correct_args(self):
        """Each linear-attn V-space projection must have a dispatch branch that
        calls _reorder_v_heads with the correct reorder axis and head_dim. A
        missing branch, or one wired with the wrong dim/head_dim, mis-lays-out
        the V heads and decode produces garbage."""
        for tensor, (exp_dim, exp_head_dim) in EXPECTED.items():
            with self.subTest(tensor=tensor):
                branch = _find_branch(self.mt, tensor)
                self.assertIsNotNone(
                    branch,
                    f"Qwen3_5TextModel.modify_tensors has no branch for "
                    f"'{tensor}' -- it is emitted in HF grouped V-head order, "
                    f"misaligning the runtime delta_net and producing garbage.",
                )
                calls = _reorder_calls_in_body(branch)
                self.assertTrue(
                    calls,
                    f"branch for '{tensor}' never calls _reorder_v_heads -- "
                    f"the tensor is passed through un-reordered.",
                )
                call = calls[0]
                self.assertGreaterEqual(
                    len(call.args), 5,
                    f"'{tensor}': _reorder_v_heads(tensor, dim, num_k_heads, "
                    f"num_v_per_k, head_dim) needs 5 args.",
                )
                self.assertEqual(
                    ast.unparse(call.args[1]), exp_dim,
                    f"'{tensor}': reorder dim should be {exp_dim} "
                    f"(0 = rows, 1 = columns).",
                )
                self.assertEqual(
                    ast.unparse(call.args[4]), exp_head_dim,
                    f"'{tensor}': reorder head_dim should be {exp_head_dim}.",
                )

    def test_fused_linear_attn_layout_fails_loudly(self):
        """The dense converter handles only the separate in_proj_qkv/z/a/b
        form. A checkpoint shipping the fused in_proj_qkvz / in_proj_ba layout
        must hit an explicit NotImplementedError at convert time -- otherwise
        the fused tensors fall through to super().modify_tensors un-reordered
        and silently produce a broken GGUF."""
        for tensor in FUSED_LAYOUTS:
            with self.subTest(tensor=tensor):
                branch = _find_branch(self.mt, tensor)
                self.assertIsNotNone(
                    branch,
                    f"no guard for fused layout '{tensor}' -- a fused "
                    f"checkpoint would fall through un-reordered and produce a "
                    f"broken GGUF with no error.",
                )
                body = ast.unparse(branch.body)
                self.assertIn(
                    "NotImplementedError",
                    body,
                    f"the '{tensor}' guard must raise NotImplementedError so a "
                    f"fused checkpoint fails loudly at convert time.",
                )


if __name__ == "__main__":
    unittest.main()

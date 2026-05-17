"""
Regression test for ubergarm's PR #1654 Apr-28 retest report on
Qwen3.6-35B-A3B.

Symptom: converter crashes with `RuntimeError: shape '[16, 2, 128, 8192]' is
invalid for input of size 0` at `_reorder_v_heads` while processing
`blk.0.linear_attn.conv1d.weight`.

Root cause: `Qwen3_5MoeTextModel.modify_tensors` duplicates the parent
`Qwen3NextModel.modify_tensors` linear_attn / norm / numeric-transform
handling. Subclass falls through to `yield from super().modify_tensors(...)`
so every transformation runs TWICE:

  - `data_torch + 1` for norm.weight on the general fall-through path runs
    twice -> wrong scale (+2 not +1).  NOTE: the mtp. branch applies its own
    +1 to standalone nextn norms before returning -- that is intentional and
    correct (commit ea5bffe3); the mtp. branch never reaches super().
  - `-torch.exp(data_torch)` for A_log runs twice -> -exp(-exp(x)) garbage
  - linear_attn V-head reorder runs twice -> mis-permuted weights
  - linear_attn.conv1d.weight: subclass produces [4, 8192] (transposed),
    parent re-slices [0:4096] then [4096:] -> empty v_part -> reshape crash

Fix: remove the duplicated handling from the subclass so the parent (which
has identical handling) processes the tensors exactly once. The subclass
retains only the genuinely subclass-specific MoE expert reshaping and MTP
standalone-tensor dispatch (which returns before reaching super()).

These tests use AST inspection on convert_hf_to_gguf.py -- stdlib unittest
only, no torch/pytest needed.
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


def _find_method(cls: ast.ClassDef, name: str) -> ast.FunctionDef:
    for node in cls.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"method {cls.name}.{name} not found")


def _find_mtp_branch(method: ast.FunctionDef) -> ast.If:
    """Return the `if name.startswith("mtp."):` top-level if-statement."""
    for stmt in method.body:
        if isinstance(stmt, ast.If) and "mtp" in ast.unparse(stmt.test):
            return stmt
    raise AssertionError(
        "if name.startswith('mtp.') branch not found in "
        f"method {method.name}"
    )


class Qwen35MoeNoDoubleProcessingTests(unittest.TestCase):
    """Qwen3_5MoeTextModel must not duplicate transformations its parent
    (Qwen3NextModel) already performs on the general fall-through path, since
    the subclass yields to super() at the end and any duplicated transforms
    would apply twice.

    Exception: the `if name.startswith("mtp."):` branch applies its own
    `data_torch + 1` to standalone nextn norm tensors, then returns -- it
    never reaches super(), so there is no double-processing there."""

    @classmethod
    def setUpClass(cls):
        with open(CONVERT) as f:
            cls.tree = ast.parse(f.read())
        cls.subclass = _find_class(cls.tree, "Qwen3_5MoeTextModel")
        cls.parent = _find_class(cls.tree, "Qwen3NextModel")
        cls.sub_mt = _find_method(cls.subclass, "modify_tensors")
        cls.parent_mt = _find_method(cls.parent, "modify_tensors")
        cls.sub_src = ast.unparse(cls.sub_mt)
        cls.parent_src = ast.unparse(cls.parent_mt)

    def test_parent_has_the_transforms_subclass_must_not_redo(self):
        """Sanity: the parent class is what we expect to be doing the work.
        If this fails the regression hunt has to start with the parent."""
        self.assertIn("data_torch + 1", self.parent_src,
                      "parent should keep norm.weight + 1 transform")
        self.assertIn("-torch.exp(data_torch)", self.parent_src,
                      "parent should keep A_log -> -exp transform")
        self.assertIn(".linear_attn.", self.parent_src,
                      "parent should keep linear_attn V-reorder block")

    def test_subclass_does_not_redo_norm_plus_one(self):
        """The general fall-through path of Qwen3_5MoeTextModel.modify_tensors
        must not apply `data_torch + 1`, because the parent already does it
        and yielding to super() would apply it twice.

        The `if name.startswith("mtp."):` branch is explicitly exempt: it
        applies +1 to standalone nextn norm tensors (enorm/hnorm/
        shared_head_norm) and then returns before reaching super(). This is
        correct and intentional (commit ea5bffe3) -- without it the MTP block
        runs with off-scale norms and produces garbage outputs.
        """
        mtp_branch = _find_mtp_branch(self.sub_mt)
        mtp_src = ast.unparse(mtp_branch)

        # Confirm the mtp. branch carries the intentional +1 for standalone norms.
        # This is correct: the mtp. standalone-tensor path yields and returns,
        # never reaching super(), so there is no double-processing.
        self.assertIn(
            "data_torch + 1",
            mtp_src,
            "mtp. branch should apply +1 to standalone nextn norm tensors "
            "(enorm/hnorm/shared_head_norm) before returning -- removing it "
            "would break MTP norm scale (commit ea5bffe3).",
        )

        # The general fall-through path (everything outside the mtp. branch)
        # must NOT contain data_torch + 1 -- the parent handles it there.
        general_path_stmts = [
            s for s in self.sub_mt.body
            if not (isinstance(s, ast.If) and "mtp" in ast.unparse(s.test))
        ]
        general_path_src = " ".join(ast.unparse(s) for s in general_path_stmts)
        self.assertNotIn(
            "data_torch + 1",
            general_path_src,
            "Qwen3_5MoeTextModel duplicates parent's norm.weight + 1 transform "
            "on the general fall-through path. Yielding to super() applies it "
            "AGAIN -> norms get wrong scale. Only the mtp. standalone-norm "
            "branch may apply +1 (it returns before reaching super()).",
        )

    def test_subclass_does_not_redo_numeric_transforms(self):
        self.assertNotIn(
            "-torch.exp(data_torch)",
            self.sub_src,
            "Qwen3_5MoeTextModel duplicates the A_log -> -exp transform. "
            "Yielding to super applies it twice -> -exp(-exp(A_log)) -> garbage.",
        )

    def test_subclass_does_not_redo_linear_attn_block(self):
        # ast.unparse normalizes quote style -- accept either form
        self.assertFalse(
            ("if '.linear_attn.' in name:" in self.sub_src
             or 'if ".linear_attn." in name:' in self.sub_src),
            msg=("Qwen3_5MoeTextModel duplicates parent's linear_attn V-head "
                 "reorder block. Yielding to super reapplies the V-reorder; "
                 "for conv1d.weight the subclass also transposes the tensor, "
                 "so the parent's slicing on [4, 8192] gives empty v_part and "
                 "_reorder_v_heads raises 'shape [16, 2, 128, 8192] invalid "
                 "for input of size 0'."),
        )

    def test_subclass_keeps_moe_expert_handling(self):
        """The MoE expert reshaping (packed down_proj / gate_up_proj) is
        genuinely subclass-specific and must remain -- Qwen2MoeModel parent
        expects per-expert layout, ours stores them packed."""
        self.assertIn("mlp.experts.down_proj", self.sub_src,
                      "subclass must keep packed-expert down_proj handling")
        self.assertIn("mlp.experts.gate_up_proj", self.sub_src,
                      "subclass must keep packed-expert gate_up_proj handling")


if __name__ == "__main__":
    unittest.main()

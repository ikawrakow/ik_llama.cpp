"""
Regression test: PR #1654 commit 070a6102 added a `data_torch = data_torch.T`
transpose at the end of the linear_attn.conv1d.weight branch in two classes
(Qwen3NextModel and Qwen3_5TextModel). The transpose stores conv1d weights
in the wrong GGUF orientation: ik_llama check_tensor_dims expects [4, 8192]
(kernel x channels for Qwen3.6) but the transpose produces [8192, 4],
blocking model load:

    check_tensor_dims: tensor 'blk.0.ssm_conv1d.weight' has wrong shape;
    expected     4,  8192, got  8192,     4,     1,     1

The pre-070a6102 converter had no transpose; the pre-PR Qwen3.6 GGUFs in
production load correctly. The fix removes the transpose from both classes.
"""
from __future__ import annotations

import ast
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CONVERT = REPO_ROOT / "convert_hf_to_gguf.py"


def _find_class(tree, name):
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise AssertionError(f"class {name} not found")


def _override(cls, method_name):
    for node in cls.body:
        if isinstance(node, ast.FunctionDef) and node.name == method_name:
            return node
    return None


class Qwen3NextConv1dNoTransposeTests(unittest.TestCase):
    """The pre-070a6102 converter wrote linear_attn.conv1d.weight without a
    transpose. ik_llama's check_tensor_dims expects [kernel, channels]
    (== [4, 8192] for Qwen3.6); GGUF stores numpy shapes reversed so the
    correct torch input shape is [channels, kernel] (== [8192, 4]).

    pschou's 070a6102 fold added `data_torch = data_torch.T` based on a
    misread of the GGUF storage convention. With the transpose, the GGUF
    ends up with [8192, 4] storage and load fails:

        check_tensor_dims: tensor 'blk.0.ssm_conv1d.weight' has wrong shape;
        expected     4,  8192, got  8192,     4,     1,     1

    Removing the transpose restores the working orientation.
    """

    @classmethod
    def setUpClass(cls):
        with open(CONVERT) as f:
            cls.tree = ast.parse(f.read())

    def _conv1d_block_text(self, class_name):
        cls = _find_class(self.tree, class_name)
        m = _override(cls, "modify_tensors")
        if m is None:
            return ""
        return ast.unparse(m)

    def _has_transpose_after_conv1d(self, body):
        # Look for the conv1d branch followed by a torch.cat assignment then a .T
        # Heuristic: the transpose was always written as `data_torch = data_torch.T`
        # at the tail of the conv1d.weight elif branch.
        if ".linear_attn.conv1d.weight" not in body:
            return False
        return "data_torch = data_torch.T" in body

    def test_qwen3next_no_conv1d_transpose(self):
        body = self._conv1d_block_text("Qwen3NextModel")
        self.assertFalse(
            self._has_transpose_after_conv1d(body),
            "Qwen3NextModel.modify_tensors has `data_torch = data_torch.T` in "
            "its linear_attn.conv1d.weight branch. ik_llama check_tensor_dims "
            "expects [4, 8192] (kernel x channels) and the transpose produces "
            "the wrong orientation, blocking model load.",
        )

    def test_qwen3_5_text_no_conv1d_transpose(self):
        body = self._conv1d_block_text("Qwen3_5TextModel")
        self.assertFalse(
            self._has_transpose_after_conv1d(body),
            "Qwen3_5TextModel (dense) has the same incorrect transpose in its "
            "conv1d.weight branch. Same load-blocking issue.",
        )


if __name__ == "__main__":
    unittest.main()

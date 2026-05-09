"""
Regression test: PR #1654 commit 070a6102 introduced an override of
Qwen3_5MoeTextModel.set_gguf_parameters that does not call super(),
silently dropping every architectural hyperparameter except
full_attention_interval. The resulting GGUF aborts at quantize time:

    key not found in model: qwen35moe.block_count
    GGML_ASSERT(... && "n_attention_wv is unexpected") failed

and at load time the model can't be initialized.

The fix is to remove the override entirely so the parent's full
set_gguf_parameters chain runs.
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


class Qwen35MoeSetGgufParametersTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        with open(CONVERT) as f:
            cls.tree = ast.parse(f.read())
        cls.subclass = _find_class(cls.tree, "Qwen3_5MoeTextModel")
        cls.parent = _find_class(cls.tree, "Qwen3NextModel")

    def test_parent_writes_block_count_via_chain(self):
        """Sanity: parent has set_gguf_parameters and calls super so the
        block_count / embedding_length / etc. propagate up the chain."""
        m = _override(self.parent, "set_gguf_parameters")
        self.assertIsNotNone(
            m,
            "Qwen3NextModel.set_gguf_parameters missing — without it none of "
            "the qwen35moe.* hparams are written.",
        )
        body = ast.unparse(m)
        self.assertIn(
            "super().set_gguf_parameters()",
            body,
            "Qwen3NextModel.set_gguf_parameters must call super() so the base "
            "chain writes block_count, embedding_length, head_count, expert_count, "
            "etc. Otherwise the converter produces a GGUF that aborts at "
            "quantize/load time with 'key not found in model: qwen35moe.block_count'.",
        )

    def test_subclass_does_not_override_set_gguf_parameters_without_super(self):
        """If the subclass overrides set_gguf_parameters, it must call super,
        otherwise the parent's full hparam-writing chain is skipped."""
        m = _override(self.subclass, "set_gguf_parameters")
        if m is None:
            return  # no override is fine — parent runs directly
        body = ast.unparse(m)
        self.assertIn(
            "super().set_gguf_parameters()",
            body,
            "Qwen3_5MoeTextModel.set_gguf_parameters override is missing a "
            "super() call, so the parent's hparam chain does not run. The "
            "produced GGUF will be missing every qwen35moe.* hparam and will "
            "abort at llama-quantize and llama-perplexity.",
        )


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
"""
TDD test for Task 3: Qwen3_5TextModel.__init__ bumps block_count for MTP layers.

The dense Qwen3.6-27B model has a config field mtp_num_hidden_layers which
indicates the number of MTP tail layers. When mtp_num_hidden_layers > 0, the
converter must bump block_count by that amount and rebuild tensor_map so that
blk.{n_main + i}.* tensor names resolve during conversion.

This test uses AST inspection (no torch needed) to verify:
  1. Qwen3_5TextModel defines its own __init__ method (not inherited from
     Qwen2Model, which does not handle MTP).
  2. The __init__ reads mtp_num_hidden_layers from self.hparams and stores it
     as self._nextn_layers.
  3. The __init__ bumps self.block_count when nextn > 0.
  4. The __init__ rebuilds self.tensor_map when nextn > 0.
  5. The pattern is structurally equivalent to Qwen3_5MoeTextModel.__init__
     (the proven MoE reference implementation).

AST inspection was chosen over instantiation (which requires torch, not
available on this box) and over --vocab-only conversion (which requires a
full model write pass and is ~10 seconds). AST inspection is instantaneous,
has no external dependencies, and directly checks the code structure that
determines runtime behavior.
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


class Qwen35DenseBlockCountTests(unittest.TestCase):
    """Qwen3_5TextModel.__init__ must bump block_count for MTP layers."""

    @classmethod
    def setUpClass(cls):
        with open(CONVERT) as f:
            cls.tree = ast.parse(f.read())
        cls.dense = _find_class(cls.tree, "Qwen3_5TextModel")
        cls.moe = _find_class(cls.tree, "Qwen3_5MoeTextModel")
        cls.moe_init = _find_method(cls.moe, "__init__")

    def test_dense_has_own_init(self):
        """Qwen3_5TextModel must define its own __init__, not rely on
        Qwen2Model which does not handle mtp_num_hidden_layers."""
        init = _find_method(self.dense, "__init__")
        self.assertIsNotNone(
            init,
            "Qwen3_5TextModel is missing __init__. Without it, block_count is "
            "never bumped for mtp_num_hidden_layers, so MTP tensor names "
            "blk.64.nextn.* are unmapped and the converter drops them silently.",
        )

    def test_dense_init_reads_mtp_num_hidden_layers(self):
        """__init__ must read mtp_num_hidden_layers from self.hparams and
        store it as self._nextn_layers (matching the MoE reference)."""
        init = _find_method(self.dense, "__init__")
        self.assertIsNotNone(init, "Qwen3_5TextModel.__init__ not found")
        src = ast.unparse(init)
        self.assertIn(
            "mtp_num_hidden_layers",
            src,
            "Qwen3_5TextModel.__init__ must read mtp_num_hidden_layers from "
            "self.hparams; block_count bump is never triggered without it.",
        )
        self.assertIn(
            "_nextn_layers",
            src,
            "Qwen3_5TextModel.__init__ must store result as self._nextn_layers "
            "(other methods, e.g. set_gguf_parameters, read this attribute).",
        )

    def test_dense_init_bumps_block_count(self):
        """When nextn > 0, __init__ must set self.block_count to include MTP."""
        init = _find_method(self.dense, "__init__")
        self.assertIsNotNone(init, "Qwen3_5TextModel.__init__ not found")
        src = ast.unparse(init)
        self.assertIn(
            "block_count",
            src,
            "Qwen3_5TextModel.__init__ must set self.block_count to include "
            "MTP layers; without this blk.64.* names are out of range and the "
            "converter cannot map the MTP tensors.",
        )

    def test_dense_init_rebuilds_tensor_map(self):
        """When nextn > 0, __init__ must rebuild self.tensor_map."""
        init = _find_method(self.dense, "__init__")
        self.assertIsNotNone(init, "Qwen3_5TextModel.__init__ not found")
        src = ast.unparse(init)
        self.assertIn(
            "tensor_map",
            src,
            "Qwen3_5TextModel.__init__ must rebuild self.tensor_map after "
            "bumping block_count so that blk.{64}.nextn.* lookups succeed.",
        )

    def test_dense_init_calls_super(self):
        """__init__ must call super().__init__ so Qwen2Model initialization
        (block_count, tensor_map, hparams) runs first."""
        init = _find_method(self.dense, "__init__")
        self.assertIsNotNone(init, "Qwen3_5TextModel.__init__ not found")
        src = ast.unparse(init)
        self.assertIn(
            "super().__init__",
            src,
            "Qwen3_5TextModel.__init__ must call super().__init__ before the "
            "MTP block_count bump; otherwise self.hparams is not populated.",
        )

    def test_dense_init_matches_moe_structure(self):
        """Dense __init__ must mirror Qwen3_5MoeTextModel.__init__."""
        self.assertIsNotNone(
            self.moe_init,
            "Qwen3_5MoeTextModel.__init__ not found -- cannot cross-check.",
        )
        moe_src = ast.unparse(self.moe_init)
        init = _find_method(self.dense, "__init__")
        self.assertIsNotNone(init, "Qwen3_5TextModel.__init__ not found")
        dense_src = ast.unparse(init)

        for attr in ("mtp_num_hidden_layers", "_nextn_layers", "block_count", "tensor_map"):
            self.assertIn(attr, moe_src, f"MoE __init__ must reference {attr} (sanity).")
            self.assertIn(attr, dense_src, f"Dense __init__ must reference {attr}.")


if __name__ == "__main__":
    unittest.main(verbosity=2)

#!/usr/bin/env python3
"""
TDD test for Task 5: Qwen3_5TextModel.modify_tensors dispatches MTP tensors.

Previously the method silently dropped all mtp.* tensors with `return []`.
After Task 5, it must:
  (a) no longer contain the bare `return []` drop for mtp.* (has dispatch logic
      referencing `mtp.layers` instead);
  (b) emit the 4 standalone MTP target names:
        blk.{nextn_bid}.nextn.eh_proj.weight
        blk.{nextn_bid}.nextn.enorm.weight
        blk.{nextn_bid}.nextn.hnorm.weight
        blk.{nextn_bid}.nextn.shared_head_norm.weight
  (c) applies data_torch + 1 within the mtp branch for the standalone norms;
  (d) re-routes mtp.layers.* via type(self).modify_tensors (same dense method)
      so normal dense handling (norm+1, attention/MLP via super()) applies once.

AST-only — no torch, no pytest.
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


class Qwen35MtpDispatchTests(unittest.TestCase):
    """Qwen3_5TextModel.modify_tensors must dispatch mtp.* tensors."""

    @classmethod
    def setUpClass(cls):
        with open(CONVERT) as f:
            raw = f.read()
        cls.tree = ast.parse(raw)
        # raw_lines: used to extract the method's raw source text for checks
        # where ast.unparse normalizes regex strings (e.g. escaping backslashes)
        # and plain string matching on the original source is more reliable.
        cls.raw_src = raw
        cls.dense = _find_class(cls.tree, "Qwen3_5TextModel")
        mt = _find_method(cls.dense, "modify_tensors")
        assert mt is not None, "Qwen3_5TextModel.modify_tensors not found"
        cls.mt = mt
        cls.src = ast.unparse(mt)
        # Extract the raw method text using line numbers from the AST node.
        lines = raw.splitlines(keepends=True)
        cls.raw_method_src = "".join(lines[mt.lineno - 1:mt.end_lineno])

    # ------------------------------------------------------------------
    # (a) No bare drop — must have dispatch logic referencing mtp.layers
    # ------------------------------------------------------------------

    def test_mtp_prefix_not_bare_dropped(self):
        """The method must no longer silently drop all mtp.X with 'return []'.
        After the fix it must have dispatch logic for mtp.layers.X tensors.
        We check the raw method source (not ast.unparse output) because
        ast.unparse escapes backslashes inside regex string constants, turning
        the literal r"mtp.layers..." into 'mtp\\.layers...' and hiding
        the plain 'mtp.layers' substring."""
        # The old code had: if name.startswith("mtp."): return []
        # After the fix the branch must contain "mtp.layers" in the regex.
        self.assertIn(
            "mtp.layers",
            self.raw_method_src,
            "modify_tensors must dispatch mtp.layers.* tensors (dispatch logic "
            "not found in raw source). Was the mtp. drop block replaced with "
            "actual routing?",
        )

    def test_mtp_layers_dispatch_uses_type_self_modify_tensors(self):
        """mtp.layers.* must be re-routed via type(self).modify_tensors so
        the dense norm+1 and standard attention handling apply exactly once."""
        self.assertIn(
            "type(self).modify_tensors",
            self.src,
            "mtp.layers.* dispatch must use type(self).modify_tensors(self, ...) "
            "to re-enter the dense modify_tensors and apply norm+1 once.",
        )

    def test_nextn_layers_zero_guard(self):
        """When _nextn_layers == 0, the method must short-circuit without
        emitting any tensor (return, not return [])."""
        self.assertIn(
            "_nextn_layers",
            self.src,
            "modify_tensors must guard on self._nextn_layers == 0 for models "
            "that ship no MTP tail layer.",
        )

    def test_uses_num_hidden_layers_for_base(self):
        """The base layer index must be read from hparams['num_hidden_layers']
        so the MTP block maps to blk.{n_main + i}.*."""
        self.assertIn(
            "num_hidden_layers",
            self.src,
            "modify_tensors must read num_hidden_layers from self.hparams to "
            "compute the base block index for MTP tensors.",
        )

    # ------------------------------------------------------------------
    # (b) 4 standalone target names present in source
    # ------------------------------------------------------------------

    def test_standalone_eh_proj_target(self):
        self.assertIn(
            "nextn.eh_proj.weight",
            self.src,
            "modify_tensors must map mtp.fc.weight -> blk.N.nextn.eh_proj.weight",
        )

    def test_standalone_enorm_target(self):
        self.assertIn(
            "nextn.enorm.weight",
            self.src,
            "modify_tensors must map mtp.pre_fc_norm_embedding.weight -> "
            "blk.N.nextn.enorm.weight",
        )

    def test_standalone_hnorm_target(self):
        self.assertIn(
            "nextn.hnorm.weight",
            self.src,
            "modify_tensors must map mtp.pre_fc_norm_hidden.weight -> "
            "blk.N.nextn.hnorm.weight",
        )

    def test_standalone_shared_head_norm_target(self):
        self.assertIn(
            "nextn.shared_head_norm.weight",
            self.src,
            "modify_tensors must map mtp.norm.weight -> "
            "blk.N.nextn.shared_head_norm.weight",
        )

    # ------------------------------------------------------------------
    # (c) norm+1 applied inside the mtp standalone branch
    # ------------------------------------------------------------------

    def test_standalone_norm_plus_one_applied(self):
        """The standalone branch must apply data_torch + 1 for norm tensors
        because standalone tensors are yielded directly without passing through
        the general-path norm+1 code below the mtp block."""
        # Find the if name.startswith("mtp.") subtree in the AST and check
        # that data_torch + 1 appears within it, not just anywhere in the method.
        mtp_block = None
        for node in ast.walk(self.mt):
            if isinstance(node, ast.If):
                # Look for: if name.startswith("mtp."):
                test = node.test
                if (
                    isinstance(test, ast.Call)
                    and isinstance(test.func, ast.Attribute)
                    and test.func.attr == "startswith"
                ):
                    args = test.args
                    if args and isinstance(args[0], ast.Constant) and args[0].value == "mtp.":
                        mtp_block = node
                        break

        self.assertIsNotNone(
            mtp_block,
            "Could not find `if name.startswith('mtp.')` block in modify_tensors. "
            "The mtp dispatch block is missing or malformed.",
        )

        mtp_src = ast.unparse(mtp_block)
        self.assertIn(
            "data_torch + 1",
            mtp_src,
            "The mtp standalone branch must apply data_torch + 1 for norm tensors "
            "(enorm, hnorm, shared_head_norm) — they are yielded directly and "
            "miss the general-path norm+1 below the mtp block.",
        )

    # ------------------------------------------------------------------
    # (d) re-route uses type(self).modify_tensors — already covered in (a)
    #     but also verify yield from is used (generator protocol)
    # ------------------------------------------------------------------

    def test_mtp_layers_uses_yield_from(self):
        """The mtp.layers.* dispatch must use yield from so this generator
        correctly delegates to the recursive type(self).modify_tensors call."""
        self.assertIn(
            "yield from type(self).modify_tensors",
            self.src,
            "mtp.layers.* dispatch must use `yield from type(self).modify_tensors` "
            "to forward tensors through the generator chain.",
        )

    def test_mtp_branch_uses_re_match(self):
        """mtp.layers.{i}.{rest} pattern must be parsed with re.match so the
        block index i can be extracted and mapped to base + i."""
        self.assertIn(
            "re.match",
            self.src,
            "The mtp.layers.* dispatch must use re.match to parse the layer "
            "index from the tensor name.",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)

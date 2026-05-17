#!/usr/bin/env python3
"""
TDD test for Task 2: register NEXTN tensors on MODEL_ARCH.QWEN35 dense arch.

The dense QWEN35 model (Qwen3.6-27B) has multi-token-prediction (MTP)
support via a tail layer with 4 tensor types. Currently these are registered
only on QWEN35MOE (the MoE variant); they must also be registered on QWEN35
(the dense variant) so the converter can name the MTP tensors during
conversion.

This test asserts:
  1. QWEN35 includes all 4 NEXTN tensor types
  2. get_tensor_name_map() succeeds for 65 blocks (64 main + 1 MTP)
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

# Make the local gguf package importable.
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "gguf-py"))

import gguf


class Qwen35NextnTests(unittest.TestCase):

    def test_qwen35_has_all_nextn_tensors(self):
        """QWEN35 must include all 4 NEXTN tensor types."""
        ts = gguf.MODEL_TENSORS[gguf.MODEL_ARCH.QWEN35]
        expected = [
            "NEXTN_EH_PROJ",
            "NEXTN_ENORM",
            "NEXTN_HNORM",
            "NEXTN_SHARED_HEAD_NORM",
        ]
        for tensor_name in expected:
            tensor_enum = getattr(gguf.MODEL_TENSOR, tensor_name)
            self.assertIn(
                tensor_enum,
                ts,
                f"QWEN35 missing {tensor_name}",
            )

    def test_qwen35_name_map_builds_for_mtp_block(self):
        """get_tensor_name_map() must handle 65 blocks (64 main + 1 MTP)."""
        # This should not raise an exception. The name_map validates
        # that all required tensor types can be mapped for 65 blocks.
        name_map = gguf.get_tensor_name_map(gguf.MODEL_ARCH.QWEN35, 65)
        self.assertIsNotNone(name_map)


if __name__ == "__main__":
    unittest.main()

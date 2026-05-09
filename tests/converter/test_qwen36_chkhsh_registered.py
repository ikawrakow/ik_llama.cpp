"""
Regression test: PR #1654 claims to support Qwen3.6-35B-A3B end-to-end.
That requires its tokenizer chkhsh be in get_vocab_base_pre()'s dispatch
table; otherwise conversion raises `NotImplementedError: BPE pre-tokenizer
was not recognized` at vocab setup time.

Two snapshots have been observed in the wild for Qwen/Qwen3.6-35B-A3B:

  - d30d75d9059f1aa2c19359de71047b3ae408c70875e8a3ccf8c5fba56c9d8af4
    (registered when PR #1654 landed model class support)
  - 1444df51289cfa8063b96f0e62b1125440111bc79a52003ea14b6eac7016fd5f
    (current HF snapshot used by ubergarm's and our prod environments)

Both should resolve to the `qwen35` pre-tokenizer alias. Failure mode if
either is missing: the converter aborts on the very first vocab setup
call before writing any tensors.
"""
from __future__ import annotations

import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CONVERT = REPO_ROOT / "convert_hf_to_gguf.py"


class Qwen36ChkhshRegisteredTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.src = CONVERT.read_text()

    def _registered_to_qwen35(self, chkhsh: str) -> bool:
        # Match the if-block: chkhsh entry then a "qwen35" assignment within
        # ~3 lines (covers the comment + res = "qwen35" body).
        idx = self.src.find(f'chkhsh == "{chkhsh}"')
        if idx < 0:
            return False
        return 'res = "qwen35"' in self.src[idx:idx + 400]

    def test_d30d75d9_snapshot_registered(self):
        self.assertTrue(
            self._registered_to_qwen35(
                "d30d75d9059f1aa2c19359de71047b3ae408c70875e8a3ccf8c5fba56c9d8af4"
            ),
            "The d30d75d9 Qwen3.6-35B-A3B chkhsh entry was removed. The "
            "vocab dispatch needs at least one Qwen3.6 hash entry mapping "
            "to qwen35; without it conversion aborts in get_vocab_base_pre.",
        )

    def test_1444df51_snapshot_registered(self):
        self.assertTrue(
            self._registered_to_qwen35(
                "1444df51289cfa8063b96f0e62b1125440111bc79a52003ea14b6eac7016fd5f"
            ),
            "Our prod Qwen3.6-35B-A3B HF download has chkhsh "
            "1444df51289cfa8063b96f0e62b1125440111bc79a52003ea14b6eac7016fd5f. "
            "Without an entry mapping it to qwen35, get_vocab_base_pre "
            "raises NotImplementedError on the very first vocab setup call.",
        )


if __name__ == "__main__":
    unittest.main()

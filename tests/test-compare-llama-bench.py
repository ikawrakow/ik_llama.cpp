#!/usr/bin/env python3

import os
import pathlib
import sqlite3
import subprocess
import sys
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "compare-llama-bench.py"

# Keep this fixture schema at the compare script's cross-version contract, not
# at either printer's full schema. New printer-only fields must not be required
# to compare legacy results.
KEY_PROPERTIES = [
    "cpu_info", "gpu_info", "n_gpu_layers", "cuda", "vulkan", "kompute", "metal", "sycl", "rpc", "gpu_blas",
    "blas", "model_filename", "model_type", "model_size", "model_n_params", "n_batch", "n_ubatch", "embeddings", "n_threads",
    "type_k", "type_v", "use_mmap", "no_kv_offload", "split_mode", "main_gpu", "tensor_split", "flash_attn", "n_prompt", "n_gen",
]
SOURCE_COLUMNS = list(dict.fromkeys(["build_commit", "test_time", "avg_ts"] + KEY_PROPERTIES))


class CompareLlamaBenchTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.tmp.name)
        self.deps = self.root / "deps"
        self.deps.mkdir()

        # The script normally imports GitPython and tabulate. The fixture uses
        # explicit commits and only verifies the data path, so small stubs keep
        # this regression test hermetic without changing script dependencies.
        (self.deps / "git.py").write_text(
            "class InvalidGitRepositoryError(Exception): pass\n"
            "class Repo:\n"
            "    def __init__(self, *args, **kwargs):\n"
            "        raise InvalidGitRepositoryError()\n"
            "class Commit: pass\n",
            encoding="utf-8",
        )
        (self.deps / "tabulate.py").write_text(
            "def tabulate(rows, headers, **kwargs):\n"
            "    return repr((headers, rows))\n",
            encoding="utf-8",
        )

    def tearDown(self):
        self.tmp.cleanup()

    def create_db(self, table_names):
        path = self.root / "llama-bench.sqlite"
        connection = sqlite3.connect(path)
        try:
            columns = ", ".join(f"{name} TEXT" for name in SOURCE_COLUMNS)
            for table in table_names:
                connection.execute(f"CREATE TABLE {table} ({columns})")
            connection.commit()
        finally:
            connection.close()
        return path

    def insert_run(self, path, table, commit, avg_ts):
        values = {name: "0" for name in SOURCE_COLUMNS}
        values.update({
            "build_commit": commit,
            "test_time": commit,
            "avg_ts": str(avg_ts),
            "cpu_info": "fixture cpu",
            "gpu_info": "fixture gpu",
            "model_filename": "fixture.gguf",
            "model_type": "fixture",
            "model_size": "1",
            "model_n_params": "1",
            "n_gpu_layers": "0",
            "n_batch": "1",
            "n_ubatch": "1",
            "n_threads": "1",
            "type_k": "f16",
            "type_v": "f16",
            "use_mmap": "1",
            "split_mode": "none",
            "main_gpu": "0",
            "tensor_split": "0",
            "n_prompt": "16",
            "n_gen": "0",
        })
        connection = sqlite3.connect(path)
        try:
            connection.execute(
                f"INSERT INTO {table} ({', '.join(SOURCE_COLUMNS)}) "
                f"VALUES ({', '.join('?' for _ in SOURCE_COLUMNS)})",
                [values[column] for column in SOURCE_COLUMNS],
            )
            connection.commit()
        finally:
            connection.close()

    def run_compare(self, path):
        env = os.environ.copy()
        env["PYTHONPATH"] = str(self.deps) + os.pathsep + env.get("PYTHONPATH", "")
        return subprocess.run(
            [sys.executable, str(SCRIPT), "-i", str(path), "-b", "base0001", "-c", "comp0002", "-s", "model_type", "-o", "plain"],
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )

    def assert_successful_comparison(self, path):
        result = self.run_compare(path)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("fixture", result.stdout)
        self.assertIn("pp16", result.stdout)
        self.assertIn("10.0", result.stdout)
        self.assertIn("20.0", result.stdout)
        self.assertIn("2.0", result.stdout)

    def test_legacy_test_table(self):
        path = self.create_db(["test"])
        self.insert_run(path, "test", "base0001", 10.0)
        self.insert_run(path, "test", "comp0002", 20.0)
        self.assert_successful_comparison(path)

    def test_v2_table(self):
        path = self.create_db(["test_v2"])
        self.insert_run(path, "test_v2", "base0001", 10.0)
        self.insert_run(path, "test_v2", "comp0002", 20.0)
        self.assert_successful_comparison(path)

    def test_mixed_tables_compare_across_versions(self):
        path = self.create_db(["test", "test_v2"])
        self.insert_run(path, "test", "base0001", 10.0)
        self.insert_run(path, "test_v2", "comp0002", 20.0)
        self.assert_successful_comparison(path)

    def test_empty_database_reports_expected_tables(self):
        path = self.root / "empty.sqlite"
        sqlite3.connect(path).close()
        result = self.run_compare(path)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("expected `test` or `test_v2`", result.stderr)


if __name__ == "__main__":
    unittest.main()

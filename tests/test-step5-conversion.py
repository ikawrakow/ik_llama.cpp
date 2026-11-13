"""Small CPU fixtures for Step5 conversion; no model weights or inference."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "gguf-py"))
spec = importlib.util.spec_from_file_location("converter", ROOT / "convert_hf_to_gguf.py")
converter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(converter)
torch = converter.torch
gguf = converter.gguf


class Step5ConversionTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        directory = Path(self.tmp.name)
        hp = dict(num_hidden_layers=4, hidden_size=16, intermediate_size=32,
                  head_dim=12, num_attention_heads=4, num_attention_groups=2,
                  max_position_embeddings=8192, rms_norm_eps=1e-5, sliding_window=512,
                  layer_types=["sliding_attention"] * 3 + ["full_attention"],
                  partial_rotary_factors=[1, 1, 1, 1/3], rope_theta=[10000] * 3 + [10000000],
                  swiglu_limits=[7] * 4, swiglu_limits_shared=[7] * 4,
                  moe_num_experts=8, num_experts_per_tok=2, moe_intermediate_size=24,
                  share_expert_dim=24, moe_router_scaling_factor=3, norm_expert_weight=True)
        (directory / "config.json").write_text(json.dumps({"text_config": hp}))
        self.model = converter.Step5Model(directory, gguf.LlamaFileType.MOSTLY_BF16,
                                          directory / "unused.gguf")

    def test_dense_metadata_preserves_layer_geometry(self):
        self.model.set_gguf_parameters()
        kv = self.model.gguf_writer.kv_data[0]
        self.assertEqual(kv["step5.block_count"].value, 4)
        self.assertEqual(kv["step5.rope.dimension_count_per_layer"].value, [12, 12, 12, 4])
        self.assertEqual(kv["step5.rope.freq_base_per_layer"].value, [10000., 10000., 10000., 10000000.])
        self.assertEqual(kv["step5.attention.sliding_window_pattern"].value, [True, True, True, False])
        self.assertNotIn("step5.nextn_predict_layers", kv)
        self.assertFalse(any("sparse" in key for key in kv))

    def test_zero_centered_norms_and_plain_projection(self):
        raw = torch.tensor([-0.25, 0.0, 0.5])
        for source, target, bid in [
            ("model.norm.weight", "output_norm.weight", None),
            ("model.layers.0.input_layernorm.weight", "blk.0.attn_norm.weight", 0),
            ("model.layers.0.self_attn.q_norm.weight", "blk.0.attn_q_norm.weight", 0),
            ("model.layers.0.self_attn.k_norm.weight", "blk.0.attn_k_norm.weight", 0),
            ("model.layers.0.post_attention_layernorm.weight", "blk.0.ffn_norm.weight", 0),
        ]:
            with self.subTest(source=source):
                [(name, data)] = self.model.modify_tensors(raw, source, bid)
                self.assertEqual(name, target)
                torch.testing.assert_close(data, raw + 1, rtol=0, atol=0)
        [(name, data)] = self.model.modify_tensors(raw, "model.layers.0.self_attn.q_proj.weight", 0)
        self.assertEqual(name, "blk.0.attn_q.weight")
        self.assertIs(data, raw)

    def test_router_and_expert_names(self):
        raw = torch.zeros(1)
        for source, target in [("moe.router_bias", "exp_probs_b.bias"),
                               ("moe.gate.weight", "ffn_gate_inp.weight"),
                               ("moe.up_proj.weight", "ffn_up_exps.weight"),
                               ("share_expert.up_proj.weight", "ffn_up_shexp.weight")]:
            [(name, data)] = self.model.modify_tensors(raw, f"model.layers.3.{source}", 3)
            self.assertEqual(name, f"blk.3.{target}")
            self.assertIs(data, raw)

    def test_preserved_indexer_norm_conventions(self):
        raw = torch.tensor([-0.25, 0.0, 0.5])
        for part, suffix in [("q", ".weight"), ("k", ".weight"), ("z", ".weight"),
                             ("w", ".weight"), ("q_norm", ".weight"),
                             ("k_norm", ".weight"), ("k_norm", ".bias"), ("ssmax_s", "")]:
            source = "ssmax_s" if part == "ssmax_s" else f"sparse_indexer_{part}{suffix}"
            [(name, data)] = self.model.modify_tensors(raw, f"model.layers.3.self_attn.{source}", 3)
            self.assertEqual(name, f"blk.3.indexer.{part}{suffix}")
            torch.testing.assert_close(data, raw + 1 if part == "q_norm" else raw, rtol=0, atol=0)

    def test_sparse_metadata_is_preserved(self):
        self.model.hparams["sparse_config"] = {"enabled": True, "topk": 512, "proxy_dim": 256}
        self.model.set_gguf_parameters()
        kv = self.model.gguf_writer.kv_data[0]
        self.assertTrue(kv["step5.sparse.enabled"].value)
        self.assertEqual(kv["step5.sparse.topk"].value, 512)
        self.assertEqual(kv["step5.sparse.proxy_dim"].value, 256)

    def test_mtp_and_vision_are_not_exported(self):
        for name, bid in [("model.layers.4.transformer.shared_head.output.weight", 4),
                          ("vit.patch_embedding.weight", None)]:
            self.assertEqual(self.model.modify_tensors(torch.zeros(1), name, bid), [])

    def test_unhandled_text_weights_and_rope_scaling_fail(self):
        with self.assertRaises(ValueError):
            self.model.modify_tensors(torch.zeros(1), "model.layers.0.mlp.unknown.weight", 0)
        self.model.hparams["rope_scaling"] = {"factor": 8}
        with self.assertRaises(ValueError):
            self.model.set_gguf_parameters()


if __name__ == "__main__":
    unittest.main()

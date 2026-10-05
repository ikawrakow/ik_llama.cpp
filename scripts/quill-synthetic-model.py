#!/usr/bin/env python3
"""
Create a random-weight GGUF model with a real tokenizer.

Used for reproducible CPU speed benchmarks and for grammar / JSON-schema
conformance tests (with random weights the model "wants" to emit garbage, so
any schema-valid output is entirely the grammar's doing).

  python3 scripts/quill-synthetic-model.py --arch llama    --out dense.gguf
  python3 scripts/quill-synthetic-model.py --arch qwen3moe --out moe.gguf
"""
import argparse, os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "gguf-py"))
import gguf  # noqa: E402

PRESETS = {
    # dense llama-style, ~2B params with a 152k vocab
    "llama":    dict(n_embd=2048, n_layer=16, n_ff=8192, n_head=16, n_head_kv=4, head_dim=128),
    # Qwen3-30B-A3B shaped MoE, scaled down in depth / expert count
    "qwen3moe": dict(n_embd=2048, n_layer=12, n_ff=6144, n_head=32, n_head_kv=4, head_dim=128,
                     n_expert=64, n_expert_used=8, n_ff_exp=768),
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=list(PRESETS), default="llama")
    ap.add_argument("--out", required=True)
    ap.add_argument("--vocab", default=os.path.join(os.path.dirname(__file__), "..", "models", "ggml-vocab-qwen2.gguf"))
    ap.add_argument("--layers", type=int, default=0)
    ap.add_argument("--experts", type=int, default=0)
    ap.add_argument("--embd", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    p = dict(PRESETS[args.arch])
    if args.layers:  p["n_layer"]  = args.layers
    if args.experts: p["n_expert"] = args.experts
    if args.embd:    p["n_embd"]   = args.embd
    arch = args.arch
    rng = np.random.default_rng(args.seed)

    vr = gguf.GGUFReader(args.vocab)
    def field(name):
        f = vr.fields[name]
        if f.types[0] == gguf.GGUFValueType.ARRAY:
            if f.types[1] == gguf.GGUFValueType.STRING:
                return [bytes(f.parts[i]).decode("utf-8", errors="replace") for i in f.data]
            return [int(f.parts[i][0]) for i in f.data]
        if f.types[0] == gguf.GGUFValueType.STRING:
            return bytes(f.parts[f.data[0]]).decode("utf-8")
        return f.parts[f.data[0]][0]
    tokens = field("tokenizer.ggml.tokens")
    n_vocab = len(tokens)

    w = gguf.GGUFWriter(args.out, arch)
    w.add_name(f"quill-synthetic-{arch}")
    w.add_block_count(p["n_layer"])
    w.add_context_length(32768)
    w.add_embedding_length(p["n_embd"])
    w.add_feed_forward_length(p["n_ff"])
    w.add_head_count(p["n_head"])
    w.add_head_count_kv(p["n_head_kv"])
    w.add_key_length(p["head_dim"])
    w.add_value_length(p["head_dim"])
    w.add_rope_freq_base(1e6)
    w.add_layer_norm_rms_eps(1e-6)
    w.add_file_type(gguf.LlamaFileType.MOSTLY_F16)
    if arch == "llama":
        w.add_rope_dimension_count(p["head_dim"])
        w.add_vocab_size(n_vocab)
    if "n_expert" in p:
        w.add_expert_count(p["n_expert"])
        w.add_expert_used_count(p["n_expert_used"])
        w.add_expert_feed_forward_length(p["n_ff_exp"])
    w.add_tokenizer_model(field("tokenizer.ggml.model"))
    w.add_tokenizer_pre(field("tokenizer.ggml.pre"))
    w.add_token_list(tokens)
    w.add_token_types(field("tokenizer.ggml.token_type"))
    w.add_token_merges(field("tokenizer.ggml.merges"))
    w.add_eos_token_id(int(field("tokenizer.ggml.eos_token_id")))
    w.add_bos_token_id(int(field("tokenizer.ggml.bos_token_id")))
    w.add_pad_token_id(int(field("tokenizer.ggml.padding_token_id")))
    w.add_chat_template(field("tokenizer.chat_template"))

    E, H, HKV, D = p["n_embd"], p["n_head"], p["n_head_kv"], p["head_dim"]
    # (name, numpy shape (rows, ..., cols) == reversed ggml ne)
    specs = [("token_embd.weight", (n_vocab, E)), ("output_norm.weight", (E,)), ("output.weight", (n_vocab, E))]
    for i in range(p["n_layer"]):
        b = f"blk.{i}."
        specs += [(b + "attn_norm.weight", (E,)),
                  (b + "attn_q.weight", (H * D, E)),
                  (b + "attn_k.weight", (HKV * D, E)),
                  (b + "attn_v.weight", (HKV * D, E)),
                  (b + "attn_output.weight", (E, H * D)),
                  (b + "ffn_norm.weight", (E,))]
        if arch == "qwen3moe":
            X, F = p["n_expert"], p["n_ff_exp"]
            specs += [(b + "attn_q_norm.weight", (D,)), (b + "attn_k_norm.weight", (D,)),
                      (b + "ffn_gate_inp.weight", (X, E)),
                      (b + "ffn_gate_exps.weight", (X, F, E)),
                      (b + "ffn_up_exps.weight", (X, F, E)),
                      (b + "ffn_down_exps.weight", (X, E, F))]
        else:
            F = p["n_ff"]
            specs += [(b + "ffn_gate.weight", (F, E)), (b + "ffn_up.weight", (F, E)), (b + "ffn_down.weight", (E, F))]

    def make(name, shape):
        if len(shape) == 1:
            return np.ones(shape, dtype=np.float32)
        fan_in = shape[-1]
        a = rng.standard_normal(shape, dtype=np.float32)
        a *= (1.0 / np.sqrt(fan_in)) if "token_embd" not in name else 1.0
        return a.astype(np.float16)

    for name, shape in specs:
        dt = np.float32 if len(shape) == 1 else np.float16
        w.add_tensor_info(name, shape, np.dtype(dt), int(np.prod(shape)) * np.dtype(dt).itemsize)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_ti_data_to_file()
    for name, shape in specs:
        w.write_tensor_data(make(name, shape))
    w.close()
    print(f"wrote {args.out}")

if __name__ == "__main__":
    main()

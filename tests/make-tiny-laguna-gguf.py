#!/usr/bin/env python3
"""Generate a tiny random Laguna GGUF for tests/test-laguna-swa-ring.sh.

Dense-FFN Laguna (no MoE, no attention gate) with an explicit sliding-window
pattern: 3 SWA layers + 1 global layer, n_swa = 64. Deterministic given --seed.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "gguf-py"))
import gguf  # noqa: E402

N_LAYER   = 4
N_EMBD    = 64
N_HEAD    = 4
N_HEAD_KV = 2
HEAD_DIM  = 16
N_FF      = 128
N_SWA     = 64
SWA_PATTERN = [1, 1, 1, 0]   # blk.3 is global attention
ROPE_DIM  = 16
N_CTX_TRAIN = 4096


def bytes_to_unicode():
    # standard GPT-2 byte -> unicode table
    bs = list(range(ord("!"), ord("~") + 1)) + \
         list(range(ord("\xa1"), ord("\xac") + 1)) + \
         list(range(ord("\xae"), ord("\xff") + 1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    b2u = bytes_to_unicode()
    tokens = [b2u[i] for i in range(256)] + ["ab", "cd", "<s>", "</s>"]
    merges = ["a b", "c d"]
    n_vocab = len(tokens)

    w = gguf.GGUFWriter(args.out, "laguna")
    w.add_context_length(N_CTX_TRAIN)
    w.add_embedding_length(N_EMBD)
    w.add_block_count(N_LAYER)
    w.add_feed_forward_length(N_FF)
    w.add_head_count([N_HEAD] * N_LAYER)
    w.add_head_count_kv(N_HEAD_KV)
    w.add_key_length(HEAD_DIM)
    w.add_value_length(HEAD_DIM)
    w.add_layer_norm_rms_eps(1e-5)
    w.add_sliding_window(N_SWA)
    w.add_rope_dimension_count(ROPE_DIM)
    w.add_rope_freq_base(10000.0)
    # laguna-specific keys (see src/llama-hparams.cpp LLM_ARCH_LAGUNA)
    w.add_array("laguna.attention.sliding_window_pattern", SWA_PATTERN)
    w.add_uint32("laguna.rope.dimension_count_swa", ROPE_DIM)
    w.add_uint32("laguna.expert_feed_forward_length", N_FF)

    w.add_tokenizer_model("gpt2")
    # note: the tokenizer MODEL is "gpt2" but the accepted PRE string is "gpt-2"
    # (src/llama-vocab.cpp); "gpt2" as pre is rejected at model load
    if hasattr(w, "add_tokenizer_pre"):
        w.add_tokenizer_pre("gpt-2")
    else:
        w.add_string("tokenizer.ggml.pre", "gpt-2")
    w.add_token_list(tokens)
    w.add_token_merges(merges)
    w.add_bos_token_id(n_vocab - 2)
    w.add_eos_token_id(n_vocab - 1)

    def wt(name, shape, norm=False):
        if norm:
            data = np.ones(shape, dtype=np.float32)
        else:
            data = (rng.standard_normal(shape) * 0.02).astype(np.float32)
        w.add_tensor(name, data)

    wt("token_embd.weight", (n_vocab, N_EMBD))
    wt("output_norm.weight", (N_EMBD,), norm=True)
    wt("output.weight", (n_vocab, N_EMBD))
    for i in range(N_LAYER):
        wt(f"blk.{i}.attn_norm.weight", (N_EMBD,), norm=True)
        wt(f"blk.{i}.attn_q.weight", (N_HEAD * HEAD_DIM, N_EMBD))
        wt(f"blk.{i}.attn_k.weight", (N_HEAD_KV * HEAD_DIM, N_EMBD))
        wt(f"blk.{i}.attn_v.weight", (N_HEAD_KV * HEAD_DIM, N_EMBD))
        wt(f"blk.{i}.attn_output.weight", (N_EMBD, N_HEAD * HEAD_DIM))
        wt(f"blk.{i}.ffn_norm.weight", (N_EMBD,), norm=True)
        wt(f"blk.{i}.ffn_gate.weight", (N_FF, N_EMBD))
        wt(f"blk.{i}.ffn_up.weight", (N_FF, N_EMBD))
        wt(f"blk.{i}.ffn_down.weight", (N_EMBD, N_FF))

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {args.out} (n_vocab={n_vocab})")


if __name__ == "__main__":
    main()

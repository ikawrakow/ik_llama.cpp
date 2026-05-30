"""
Pure-NumPy helpers for Qwen3-Next / Qwen3.5 in_proj_qkvz tensor splitting.

HF stores `linear_attn.in_proj_qkvz.weight` with rows grouped per K-head:

    [G0_q (head_k_dim rows),
     G0_k (head_k_dim rows),
     G0_v (num_v_per_k * head_v_dim rows),
     G0_z (num_v_per_k * head_v_dim rows),
     G1_q ..., G1_k ..., G1_v ..., G1_z ...,
     ...]

The previous converter assumed flat-block layout
([Q_block; K_block; V_block; Z_block]) and additionally had an n_z size typo
(used `data.shape[1]` = hidden_size instead of the V-slice size).

The split here matches the upstream llama.cpp pattern in
convert_hf_to_gguf.py:4788-4816: reshape rows to
(num_k_heads, 2*head_k_dim + 2*v_per_k*head_v_dim, hidden), then split along
the middle axis into Q/K/V/Z partitions, then flatten back to 2D.

This module is a stdlib + NumPy only port — it does not require torch — so
the converter can call it on `data_torch.numpy()` and re-wrap the result.
The semantic split contract is identical.
"""
from __future__ import annotations

import numpy as np


def split_qkvz(
    qkvz: np.ndarray,
    *,
    num_k_heads: int,
    head_k_dim: int,
    num_v_heads: int,
    head_v_dim: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split a Qwen3-Next-style in_proj_qkvz weight (per-K-head-grouped) into
    its Q, K, V, Z components.

    Args:
        qkvz: 2-D array of shape (rows, hidden), where
            rows = num_k_heads * (2*head_k_dim + 2*num_v_per_k*head_v_dim)
        num_k_heads: number of K (and Q) heads in linear-attn block.
        head_k_dim:  per-head dim for Q and K.
        num_v_heads: number of V (and Z) heads. Must be divisible by num_k_heads.
        head_v_dim:  per-head dim for V and Z.

    Returns:
        (q, k, v, z), each 2-D, in canonical
        [head0, head1, ..., head_{N-1}] flat order along axis 0.
    """
    if qkvz.ndim != 2:
        raise ValueError(f"split_qkvz expects 2-D input, got shape {qkvz.shape}")
    if num_v_heads % num_k_heads != 0:
        raise ValueError(
            f"num_v_heads ({num_v_heads}) must be divisible by num_k_heads ({num_k_heads})"
        )
    num_v_per_k = num_v_heads // num_k_heads
    per_group = 2 * head_k_dim + 2 * num_v_per_k * head_v_dim
    expected_rows = num_k_heads * per_group
    if qkvz.shape[0] != expected_rows:
        raise ValueError(
            f"split_qkvz: expected {expected_rows} rows "
            f"(num_k_heads={num_k_heads}, head_k_dim={head_k_dim}, "
            f"num_v_per_k={num_v_per_k}, head_v_dim={head_v_dim}), "
            f"got {qkvz.shape[0]}"
        )

    hidden = qkvz.shape[1]

    # Reshape rows to (num_k_heads, per_group, hidden) so each group's
    # contiguous Q/K/V/Z slice can be addressed cleanly.
    grouped = qkvz.reshape(num_k_heads, per_group, hidden)

    q_end = head_k_dim
    k_end = q_end + head_k_dim
    v_end = k_end + num_v_per_k * head_v_dim
    z_end = v_end + num_v_per_k * head_v_dim
    assert z_end == per_group

    q = grouped[:, :q_end, :]                     # (num_k_heads, head_k_dim, hidden)
    k = grouped[:, q_end:k_end, :]                # (num_k_heads, head_k_dim, hidden)
    v = grouped[:, k_end:v_end, :]                # (num_k_heads, v_per_k*head_v_dim, hidden)
    z = grouped[:, v_end:z_end, :]                # (num_k_heads, v_per_k*head_v_dim, hidden)

    # Flatten the head axis back into the row axis so the result is 2-D in
    # canonical [head0_rows, head1_rows, ...] order.
    q = np.ascontiguousarray(q.reshape(num_k_heads * head_k_dim, hidden))
    k = np.ascontiguousarray(k.reshape(num_k_heads * head_k_dim, hidden))
    v = np.ascontiguousarray(v.reshape(num_v_heads * head_v_dim, hidden))
    z = np.ascontiguousarray(z.reshape(num_v_heads * head_v_dim, hidden))
    return q, k, v, z

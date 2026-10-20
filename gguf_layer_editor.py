#!/usr/bin/env python3
"""
GGUF Layer Copy Tool v4 - Preserves original file layout, fixes metadata update
Supports multiple duplication ranges in a single run.
"""
from __future__ import annotations

import argparse
import logging
import os
import re
import struct
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

if "NO_LOCAL_GGUF" not in os.environ and (Path(__file__).parent.parent / "gguf").exists():
    sys.path.insert(0, str(Path(__file__).parent.parent))

from gguf import (
    GGUFReader,
    ReaderTensor,
    GGUFValueType,
)

logger = logging.getLogger("gguf-copy-layers-v4")


def rename_layer_tensor(name: str, old_idx: int, new_idx: int) -> str:
    old_prefix = f"blk.{old_idx}."
    new_prefix = f"blk.{new_idx}."
    if name.startswith(old_prefix):
        return new_prefix + name[len(old_prefix):]
    return name


def write_string(f, s: bytes) -> None:
    f.write(struct.pack("<Q", len(s)))
    f.write(s)


def write_val(f, val, vtype: GGUFValueType) -> None:
    if vtype == GGUFValueType.UINT8:
        f.write(struct.pack("B", val))
    elif vtype == GGUFValueType.INT8:
        f.write(struct.pack("b", val))
    elif vtype == GGUFValueType.UINT16:
        f.write(struct.pack("<H", val))
    elif vtype == GGUFValueType.INT16:
        f.write(struct.pack("<h", val))
    elif vtype == GGUFValueType.UINT32:
        f.write(struct.pack("<I", val))
    elif vtype == GGUFValueType.INT32:
        f.write(struct.pack("<i", val))
    elif vtype == GGUFValueType.UINT64:
        f.write(struct.pack("<Q", val))
    elif vtype == GGUFValueType.INT64:
        f.write(struct.pack("<q", val))
    elif vtype == GGUFValueType.FLOAT32:
        f.write(struct.pack("<f", val))
    elif vtype == GGUFValueType.FLOAT64:
        f.write(struct.pack("<d", val))
    elif vtype == GGUFValueType.BOOL:
        f.write(struct.pack("B", 1 if val else 0))
    elif vtype == GGUFValueType.STRING:
        write_string(f, val.encode("utf-8") if isinstance(val, str) else val)
    else:
        raise ValueError(f"Unsupported type: {vtype}")


# Row meta size for quantized types (from ggml type traits)
_ROW_META_SIZES = {
    144: 4,   # GGML_TYPE_IQ4_KS
    145: 4,   # GGML_TYPE_IQ2_KS
    146: 4,   # GGML_TYPE_IQ4_KSS
    147: 0,   # GGML_TYPE_Q8_K16
    148: 0,   # GGML_TYPE_Q8_K32
}


def compute_ggml_nbytes(t: ReaderTensor) -> int:
    """Compute the actual memory size that ggml_nbytes would return for this tensor."""
    shape = list(t.shape)
    tensor_type = int(t.tensor_type)

    if tensor_type not in _ROW_META_SIZES or len(shape) <= 1:
        return t.n_bytes

    row_meta_size = _ROW_META_SIZES[tensor_type]
    if row_meta_size == 0:
        return t.n_bytes

    n_rows = 1
    for dim in shape[1:]:
        n_rows *= dim

    return t.n_bytes + n_rows * row_meta_size


def _get_full_attention_interval(reader: GGUFReader) -> int:
    """Read full_attention_interval from metadata, default to 4."""
    for key in reader.fields:
        if "full_attention_interval" in key.lower():
            val = reader.fields[key].parts[-1].tolist()
            if isinstance(val, list) and len(val) == 1:
                return int(val[0])
            return int(val)
    return 4


def _is_recurrent_layer(layer_idx: int, interval: int) -> bool:
    """Check if a layer is recurrent based on hybrid architecture pattern.

    In qwen3next/qwen35/qwen35moe, layers where (layer_idx + 1) % interval == 0
    are full-attention layers, all others are recurrent (linear attention) layers.
    """
    return (layer_idx + 1) % interval != 0


def _get_layer_type_name(layer_idx: int, interval: int) -> str:
    return "recurrent" if _is_recurrent_layer(layer_idx, interval) else "full-attention"


def copy_layers_v4(
    input_path: str,
    output_path: str,
    duplication_ranges: list[list[int]] | None = None,
    delete_layers: list[int] | None = None,
) -> None:
    """Copy, insert, or delete transformer layers inside a GGUF file.

    Supports multiple duplication ranges processed in a single pass.
    Each range is a list of layer indices to duplicate.
    Also supports deleting specific layers.
    """
    reader = GGUFReader(input_path, "r")
    logger.info("Loaded %d tensors, %d metadata fields", len(reader.tensors), len(reader.fields))

    n_layers = max(
        int(t.name.split(".")[1])
        for t in reader.tensors
        if t.name.startswith("blk.")
    ) + 1

    full_attn_interval = _get_full_attention_interval(reader)
    logger.info("Using full_attention_interval=%d", full_attn_interval)

    # Handle deletions first
    delete_set = set(delete_layers) if delete_layers else set()
    if delete_set:
        logger.info("Will delete layers: %s", sorted(delete_set))

    # Validate and filter duplication ranges to ensure type compatibility
    # Layer type is determined by (layer_idx + 1) % full_attn_interval
    # A duplicated layer must end up at a position with the same type
    filtered_ranges: list[list[int]] = []

    # Simpler approach: apply duplications sequentially but in memory
    # Start with original layer indices 0 to n_layers-1
    current_layers = list(range(n_layers))

    # Remove deleted layers from current_layers
    if delete_set:
        current_layers = [idx for idx in current_layers if idx not in delete_set]
        logger.info("After deletions: %d layers", len(current_layers))

    for range_idx, layers in enumerate(duplication_ranges or []):
        dup_layers = sorted(set(layers))
        insert_after = max(dup_layers)

        # Find where insert_after is in current_layers
        insert_pos = current_layers.index(insert_after)

        # Iteratively find valid layers: for each position, find a layer that matches
        # We try to place layers in order, skipping those that don't match
        valid_dup_layers = []
        next_pos = insert_pos + 1

        for dup_idx in dup_layers:
            new_type = _is_recurrent_layer(next_pos, full_attn_interval)
            orig_type = _is_recurrent_layer(dup_idx, full_attn_interval)
            if orig_type == new_type:
                valid_dup_layers.append(dup_idx)
                next_pos += 1
            else:
                logger.warning(
                    "Skipping layer %d duplication: original is %s, new position %d would be %s",
                    dup_idx, _get_layer_type_name(dup_idx, full_attn_interval), next_pos, _get_layer_type_name(next_pos, full_attn_interval)
                )
        
        if not valid_dup_layers:
            logger.warning("Duplication %d: all layers skipped due to type mismatch", range_idx + 1)
            continue
        
        logger.info("Duplication %d: layers %s (insert after %d)", range_idx + 1, valid_dup_layers, insert_after)
        
        # Find positions in current_layers
        new_layers = []
        for idx in current_layers:
            new_layers.append(idx)
            if idx == insert_after:
                # Insert duplicates after this layer
                for dup_idx in valid_dup_layers:
                    new_layers.append(dup_idx)
        
        current_layers = new_layers
        logger.info("After duplication %d: %d layers", range_idx + 1, len(current_layers))
        filtered_ranges.append(valid_dup_layers)
    
    # Build final layer mapping
    # output_layer_idx -> (orig_layer_idx, is_duplicated)
    output_layer_map: dict[int, tuple[int, bool]] = {}
    for out_idx, orig_idx in enumerate(current_layers):
        # Check if this layer is a duplicate
        is_dup = False
        for layers in filtered_ranges:
            if orig_idx in layers:
                # Check if this output position is a duplicate insertion
                # A layer is duplicated if it appears after its original position
                # Find first occurrence
                first_occurrence = current_layers.index(orig_idx)
                if out_idx > first_occurrence:
                    is_dup = True
                    break
        output_layer_map[out_idx] = (orig_idx, is_dup)

    final_n_layers = len(current_layers)
    total_dup = final_n_layers - n_layers + len(delete_set)

    # Validate that all output layers match the expected type at their position
    type_errors = []
    for out_idx, (orig_idx, _) in output_layer_map.items():
        expected_type = _is_recurrent_layer(out_idx, full_attn_interval)
        actual_type = _is_recurrent_layer(orig_idx, full_attn_interval)
        if expected_type != actual_type:
            type_errors.append(
                f"  Position {out_idx}: expected {_get_layer_type_name(out_idx, full_attn_interval)}, "
                f"got layer {orig_idx} which is {_get_layer_type_name(orig_idx, full_attn_interval)}"
            )

    if type_errors:
        logger.error("Layer type mismatch after modifications:")
        for err in type_errors:
            logger.error(err)
        raise ValueError(
            "Cannot produce valid model: layer types don't match expected pattern. "
            "This usually happens when deletions or duplications shift layers to positions "
            "with incompatible types. Try deleting multiples of 4 layers, or adjust duplication ranges."
        )

    logger.info("Final model: %d layers (%d duplicated, %d deleted)", final_n_layers, total_dup - len(delete_set), len(delete_set))
    
    # Gather tensors by layer
    layer_tensors: dict[int, list[ReaderTensor]] = {}
    non_layer_tensors: list[ReaderTensor] = []
    for t in reader.tensors:
        m = re.match(r"blk\.(\d+)\.", t.name)
        if m:
            idx = int(m.group(1))
            layer_tensors.setdefault(idx, []).append(t)
        else:
            non_layer_tensors.append(t)
    
    # Build ordered list of output tensors
    output_tensors: list[tuple[str, ReaderTensor, bool]] = []
    for out_idx in sorted(output_layer_map.keys()):
        src_idx, is_dup = output_layer_map[out_idx]
        for t in layer_tensors.get(src_idx, []):
            new_name = rename_layer_tensor(t.name, src_idx, out_idx)
            output_tensors.append((new_name, t, is_dup))
    
    # Add non-layer tensors at the start
    for t in non_layer_tensors:
        output_tensors.insert(0, (t.name, t, False))
    
    logger.info("Output will have %d tensors (was %d)", len(output_tensors), len(reader.tensors))
    
    # Open files
    with open(input_path, "rb") as fin:
        with open(output_path, "wb") as fout:
            # Read original header
            magic = fin.read(4)
            version = struct.unpack("<I", fin.read(4))[0]
            orig_tensor_count = struct.unpack("<Q", fin.read(8))[0]
            kv_count = struct.unpack("<Q", fin.read(8))[0]
            
            logger.info("Original header: version=%d, tensors=%d, kv=%d", version, orig_tensor_count, kv_count)
            
            # Write new header
            fout.write(magic)
            fout.write(struct.pack("<I", version))
            fout.write(struct.pack("<Q", len(output_tensors)))
            fout.write(struct.pack("<Q", kv_count))
            
            # Find KV data end
            if len(reader.tensors) == 0:
                raise ValueError("No tensors found in input file")
            
            kv_data_end = reader.tensors[0].field.offset
            kv_data_size = kv_data_end - 24
            
            logger.info("KV data section: offset=24, size=%d, end=%d", kv_data_size, kv_data_end)
            
            fin.seek(24)
            kv_data = bytearray(fin.read(kv_data_size))
            
            if len(kv_data) != kv_data_size:
                raise ValueError(f"Could not read full KV data: expected {kv_data_size}, got {len(kv_data)}")
            
            # Update block_count metadata
            block_count_updated = False
            net_change = final_n_layers - n_layers
            for key, field in reader.fields.items():
                if "block_count" in key.lower():
                    value_offset = field.offset - 24
                    for part in field.parts[:-1]:
                        value_offset += part.nbytes
                    value_part = field.parts[-1]
                    value_size = value_part.nbytes
                    old_val = int(value_part.tolist()[0])
                    new_val = old_val + net_change
                    
                    if value_offset < 0 or value_offset + value_size > len(kv_data):
                        logger.error("Field %s: value offset %d out of bounds", key, value_offset)
                        continue
                    
                    if value_size == 4:
                        current_val = struct.unpack_from("<I", kv_data, value_offset)[0]
                    elif value_size == 8:
                        current_val = struct.unpack_from("<Q", kv_data, value_offset)[0]
                    else:
                        logger.warning("Field %s: unsupported value size %d, skipping", key, value_size)
                        continue
                    
                    if current_val != old_val:
                        logger.warning("Field %s: value mismatch! expected %d, found %d", key, old_val, current_val)
                        continue
                    
                    logger.info("Updating metadata %s: %d -> %d", key, old_val, new_val)
                    
                    if value_size == 4:
                        struct.pack_into("<I", kv_data, value_offset, new_val)
                    elif value_size == 8:
                        struct.pack_into("<Q", kv_data, value_offset, new_val)
                    
                    block_count_updated = True
            
            if not block_count_updated:
                logger.warning("No block_count field was updated!")
            
            fout.write(kv_data)
            
            # Build offset map
            offset_map: dict[str, int] = {}
            
            # Map original tensors (non-duplicated)
            for name, t, is_dup in output_tensors:
                if not name.startswith("blk."):
                    orig_t = next((ot for ot in reader.tensors if ot.name == name), None)
                    if orig_t:
                        offset_map[name] = orig_t.data_offset - reader.data_offset
                else:
                    layer_idx = int(name.split(".")[1])
                    src_idx, _ = output_layer_map[layer_idx]
                    orig_name = rename_layer_tensor(name, layer_idx, src_idx)
                    orig_t = next((ot for ot in reader.tensors if ot.name == orig_name), None)
                    if orig_t:
                        offset_map[name] = orig_t.data_offset - reader.data_offset
            
            # Find end of original tensor data
            orig_data_end = 0
            for t in reader.tensors:
                t_end = t.data_offset + compute_ggml_nbytes(t)
                if t_end > orig_data_end:
                    orig_data_end = t_end
            logger.info("Original data end (including padding): %d", orig_data_end)
            
            # Alignment
            alignment = reader.alignment
            logger.info("Using alignment=%d", alignment)
            
            # New duplicated tensors get offsets starting from after original data
            current_offset = orig_data_end - reader.data_offset
            padding = (alignment - (current_offset % alignment)) % alignment
            current_offset += padding
            
            for name, t, is_dup in output_tensors:
                if is_dup:
                    # This tensor needs a new offset
                    if name not in offset_map or offset_map[name] < current_offset:
                        offset_map[name] = current_offset
                        nbytes = compute_ggml_nbytes(t)
                        current_offset += nbytes
                        padding = (alignment - (current_offset % alignment)) % alignment
                        current_offset += padding
            
            logger.info("Original data size: %d, new total data size: %d",
                        orig_data_end - reader.data_offset, current_offset)
            
            # Write tensor info table
            for name, t, is_dup in output_tensors:
                write_string(fout, name.encode("utf-8"))
                shape = list(t.shape)
                fout.write(struct.pack("<I", len(shape)))
                for dim in shape:
                    fout.write(struct.pack("<Q", int(dim)))
                fout.write(struct.pack("<I", int(t.tensor_type)))
                offset = offset_map[name]
                fout.write(struct.pack("<Q", offset))
            
            # Pad to alignment before tensor data
            ti_end = fout.tell()
            padding = (alignment - (ti_end % alignment)) % alignment
            if padding > 0:
                fout.write(b"\x00" * padding)
            
            # Now copy original tensor data
            new_data_start = fout.tell()
            logger.info("Tensor data section starts at offset %d", new_data_start)
            
            fin.seek(reader.data_offset)
            orig_data = fin.read(orig_data_end - reader.data_offset)
            fout.write(orig_data)
            
            # Pad to where new tensors should start
            current_pos = fout.tell()
            expected_new_start = new_data_start + (orig_data_end - reader.data_offset)
            padding = expected_new_start - current_pos
            if padding > 0:
                fout.write(b"\x00" * padding)
            
            # Write duplicated tensor data
            for name, t, is_dup in output_tensors:
                if is_dup:
                    expected_offset = new_data_start + offset_map[name]
                    current_pos = fout.tell()
                    if current_pos != expected_offset:
                        padding = expected_offset - current_pos
                        if padding > 0:
                            fout.write(b"\x00" * padding)
                    t.data.tofile(fout)
                    ggml_nbytes = compute_ggml_nbytes(t)
                    if ggml_nbytes > t.n_bytes:
                        padding = ggml_nbytes - t.n_bytes
                        fout.write(b"\x00" * padding)
            
            final_size = fout.tell()
            logger.info("Final file size: %d", final_size)
    
    logger.info("Done! Output: %s", output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Copy, insert, or delete transformer layers inside a GGUF file.")
    parser.add_argument("input", type=str, help="Source GGUF model path")
    parser.add_argument("output", type=str, help="Output GGUF model path")
    parser.add_argument("--duplicate", type=int, nargs="+", action="append", metavar="N", help="Duplicate layer(s) immediately after their original position. Can be specified multiple times for multiple duplications.")
    parser.add_argument("--delete", type=int, nargs="+", metavar="N", help="Delete layer(s) from the model.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()

    if not args.duplicate and not args.delete:
        parser.error("At least one of --duplicate or --delete must be specified.")

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    copy_layers_v4(args.input, args.output, args.duplicate, args.delete)


if __name__ == "__main__":
    main()

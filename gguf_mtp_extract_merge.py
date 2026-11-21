#!/usr/bin/env python3
"""
GGUF MTP (Multi-Token Prediction) Extract and Merge Tool

Extract NextN/MTP layers from a model that has them, and merge them into
a model that doesn't have them.

Usage:
    # Extract MTP from source model and merge into target model
    python gguf_mtp_extract_merge.py extract-merge \
        ~/source-with-mtp.gguf \
        ~/target-without-mtp.gguf \
        ~/output-with-mtp.gguf

    # Just extract MTP tensors to a standalone file
    python gguf_mtp_extract_merge.py extract \
        ~/source-with-mtp.gguf \
        ~/mtp-layers.gguf

    # Just merge extracted MTP tensors into target
    python gguf_mtp_extract_merge.py merge \
        ~/mtp-layers.gguf \
        ~/target-without-mtp.gguf \
        ~/output-with-mtp.gguf
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

logger = logging.getLogger("gguf-mtp-tool")


def write_string(f, s: bytes) -> None:
    f.write(struct.pack("<Q", len(s)))
    f.write(s)


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


def _get_arch_prefix(reader: GGUFReader) -> str:
    """Get the architecture prefix from metadata (e.g., 'qwen35', 'qwen3next')."""
    arch = reader.fields.get("general.architecture")
    if arch:
        if len(arch.parts) >= 2:
            last_part = arch.parts[-1]
            data = bytes(last_part)
            try:
                decoded = data.decode("utf-8")
                cleaned = "".join(c for c in decoded if c.isalnum() or c in "._-")
                if cleaned and len(cleaned) >= 3:
                    return cleaned
            except UnicodeDecodeError:
                pass
        for part in reversed(arch.parts):
            data = bytes(part)
            try:
                decoded = data.decode("utf-8")
                cleaned = "".join(c for c in decoded if c.isalnum() or c in "._-")
                if cleaned and len(cleaned) >= 3 and "." not in cleaned:
                    return cleaned
            except UnicodeDecodeError:
                continue
    return "unknown"


def _get_block_count(reader: GGUFReader) -> int:
    """Get block_count from metadata."""
    for key in reader.fields:
        if "block_count" in key.lower():
            val = reader.fields[key].parts[-1].tolist()
            if isinstance(val, list) and len(val) == 1:
                return int(val[0])
            return int(val)
    return max(
        int(t.name.split(".")[1])
        for t in reader.tensors
        if t.name.startswith("blk.")
    ) + 1


def _get_nextn_predict_layers(reader: GGUFReader) -> int:
    """Get nextn_predict_layers from metadata."""
    for key in reader.fields:
        if "nextn_predict_layers" in key.lower():
            val = reader.fields[key].parts[-1].tolist()
            if isinstance(val, list) and len(val) == 1:
                return int(val[0])
            return int(val)
    return 0


def _rename_tensor(name: str, old_idx: int, new_idx: int) -> str:
    """Rename a tensor from old layer index to new layer index."""
    old_prefix = f"blk.{old_idx}."
    new_prefix = f"blk.{new_idx}."
    if name.startswith(old_prefix):
        return new_prefix + name[len(old_prefix):]
    return name


def extract_mtp(source_path: str, output_path: str) -> dict:
    """Extract MTP/NextN tensors from a model that has them."""
    reader = GGUFReader(source_path, "r")
    logger.info("Loaded %d tensors from %s", len(reader.tensors), source_path)

    arch_prefix = _get_arch_prefix(reader)
    block_count = _get_block_count(reader)
    nextn_layers = _get_nextn_predict_layers(reader)

    logger.info("Architecture: %s, block_count: %d, nextn_predict_layers: %d",
                arch_prefix, block_count, nextn_layers)

    if nextn_layers == 0:
        raise ValueError("Source model does not have MTP/NextN layers (nextn_predict_layers=0)")

    mtp_start_layer = block_count - nextn_layers
    logger.info("MTP layers start at layer %d", mtp_start_layer)

    mtp_tensors: list[ReaderTensor] = []
    for t in reader.tensors:
        m = re.match(r"blk\.(\d+)\.", t.name)
        if m:
            layer_idx = int(m.group(1))
            if layer_idx >= mtp_start_layer:
                mtp_tensors.append(t)

    if not mtp_tensors:
        raise ValueError("No MTP/NextN tensors found in source model")

    logger.info("Found %d MTP tensors", len(mtp_tensors))
    for t in mtp_tensors:
        logger.info("  %s", t.name)

    import pickle
    mtp_data = {
        "arch_prefix": arch_prefix,
        "block_count": block_count,
        "nextn_predict_layers": nextn_layers,
        "mtp_start_layer": mtp_start_layer,
        "tensors": [],
    }
    for t in mtp_tensors:
        mtp_data["tensors"].append({
            "name": t.name,
            "shape": list(t.shape),
            "tensor_type": int(t.tensor_type),
            "data": np.array(t.data),
            "n_bytes": t.n_bytes,
        })

    with open(output_path, "wb") as fout:
        pickle.dump(mtp_data, fout, protocol=pickle.HIGHEST_PROTOCOL)

    logger.info("Extracted %d MTP tensors to %s", len(mtp_tensors), output_path)
    return mtp_data


def merge_mtp(mtp_path: str, target_path: str, output_path: str) -> None:
    """Merge MTP/NextN tensors into a target model."""
    import pickle

    with open(mtp_path, "rb") as fin:
        mtp_data = pickle.load(fin)

    target_reader = GGUFReader(target_path, "r")

    logger.info("MTP file: %d tensors", len(mtp_data["tensors"]))
    logger.info("Target file: %d tensors", len(target_reader.tensors))

    target_arch = _get_arch_prefix(target_reader)
    target_block_count = _get_block_count(target_reader)
    target_nextn = _get_nextn_predict_layers(target_reader)

    logger.info("Target: arch=%s, block_count=%d, nextn_predict_layers=%d",
                target_arch, target_block_count, target_nextn)

    if target_nextn > 0:
        logger.warning("Target model already has MTP layers (nextn_predict_layers=%d)", target_nextn)

    new_mtp_layer_idx = target_block_count
    logger.info("MTP tensors will be placed at layer %d", new_mtp_layer_idx)

    mtp_tensors: list[tuple[str, dict]] = []
    for t in mtp_data["tensors"]:
        m = re.match(r"blk\.(\d+)\.", t["name"])
        if m:
            old_idx = int(m.group(1))
            new_name = _rename_tensor(t["name"], old_idx, new_mtp_layer_idx)
            mtp_tensors.append((new_name, t))
        else:
            mtp_tensors.append((t["name"], t))

    output_target_tensors = [(t.name, t) for t in target_reader.tensors]
    output_mtp_tensors = mtp_tensors

    logger.info("Output will have %d tensors (target=%d + mtp=%d)",
                len(output_target_tensors) + len(output_mtp_tensors),
                len(target_reader.tensors), len(mtp_tensors))

    # Use target model's original KV data and patch only necessary fields
    with open(target_path, "rb") as fin:
        with open(output_path, "wb") as fout:
            magic = fin.read(4)
            version = struct.unpack("<I", fin.read(4))[0]
            orig_tensor_count = struct.unpack("<Q", fin.read(8))[0]
            kv_count = struct.unpack("<Q", fin.read(8))[0]

            logger.info("Original header: version=%d, tensors=%d, kv=%d",
                        version, orig_tensor_count, kv_count)

            # Read original KV data
            if len(target_reader.tensors) == 0:
                raise ValueError("No tensors found in target file")
            kv_data_end = target_reader.tensors[0].field.offset
            kv_data_size = kv_data_end - 24
            fin.seek(24)
            kv_data = bytearray(fin.read(kv_data_size))

            # Update block_count metadata
            block_count_updated = False
            for key, field in target_reader.fields.items():
                if "block_count" in key.lower():
                    value_offset = field.offset - 24
                    for part in field.parts[:-1]:
                        value_offset += part.nbytes
                    value_part = field.parts[-1]
                    value_size = value_part.nbytes
                    old_val = int(value_part.tolist()[0])
                    new_val = old_val + 1

                    if value_size == 4:
                        current_val = struct.unpack_from("<I", kv_data, value_offset)[0]
                    elif value_size == 8:
                        current_val = struct.unpack_from("<Q", kv_data, value_offset)[0]
                    else:
                        continue

                    if current_val != old_val:
                        logger.warning("Field %s: value mismatch! expected %d, found %d",
                                       key, old_val, current_val)
                        continue

                    logger.info("Updating metadata %s: %d -> %d", key, old_val, new_val)

                    if value_size == 4:
                        struct.pack_into("<I", kv_data, value_offset, new_val)
                    elif value_size == 8:
                        struct.pack_into("<Q", kv_data, value_offset, new_val)

                    block_count_updated = True

            if not block_count_updated:
                logger.warning("No block_count field was updated!")

            # Update or add nextn_predict_layers metadata
            nextn_updated = False
            for key, field in target_reader.fields.items():
                if "nextn_predict_layers" in key.lower():
                    value_offset = field.offset - 24
                    for part in field.parts[:-1]:
                        value_offset += part.nbytes
                    value_part = field.parts[-1]
                    value_size = value_part.nbytes
                    old_val = int(value_part.tolist()[0])
                    new_val = old_val + mtp_data["nextn_predict_layers"]

                    if value_size == 4:
                        current_val = struct.unpack_from("<I", kv_data, value_offset)[0]
                    elif value_size == 8:
                        current_val = struct.unpack_from("<Q", kv_data, value_offset)[0]
                    else:
                        continue

                    if current_val != old_val:
                        logger.warning("Field %s: value mismatch! expected %d, found %d",
                                       key, old_val, current_val)
                        continue

                    logger.info("Updating metadata %s: %d -> %d", key, old_val, new_val)

                    if value_size == 4:
                        struct.pack_into("<I", kv_data, value_offset, new_val)
                    elif value_size == 8:
                        struct.pack_into("<Q", kv_data, value_offset, new_val)

                    nextn_updated = True

            if not nextn_updated:
                logger.info("Adding nextn_predict_layers metadata (was missing)")
                arch_prefix = target_arch
                new_key = f"{arch_prefix}.nextn_predict_layers".encode("utf-8")
                kv_data.extend(struct.pack("<Q", len(new_key)))
                kv_data.extend(new_key)
                kv_data.extend(struct.pack("<I", int(GGUFValueType.UINT32)))
                kv_data.extend(struct.pack("<I", mtp_data["nextn_predict_layers"]))
                kv_count += 1
                nextn_updated = True

            if not nextn_updated:
                logger.warning("Failed to add nextn_predict_layers!")

            # Write new header
            fout.write(magic)
            fout.write(struct.pack("<I", version))
            fout.write(struct.pack("<Q", len(output_target_tensors) + len(output_mtp_tensors)))
            fout.write(struct.pack("<Q", kv_count))

            fout.write(kv_data)

            alignment = target_reader.alignment
            logger.info("Using alignment=%d", alignment)

            # Build offset map
            offset_map: dict[str, int] = {}
            for name, t in output_target_tensors:
                offset_map[name] = t.data_offset - target_reader.data_offset

            orig_data_end = 0
            for t in target_reader.tensors:
                t_end = t.data_offset + compute_ggml_nbytes(t)
                if t_end > orig_data_end:
                    orig_data_end = t_end
            logger.info("Original data end (including padding): %d", orig_data_end)

            current_offset = orig_data_end - target_reader.data_offset
            padding = (alignment - (current_offset % alignment)) % alignment
            current_offset += padding

            for name, t_dict in output_mtp_tensors:
                offset_map[name] = current_offset
                tensor_type = t_dict["tensor_type"]
                shape = t_dict["shape"]
                n_bytes = t_dict["n_bytes"]
                if tensor_type in _ROW_META_SIZES and len(shape) > 1:
                    row_meta_size = _ROW_META_SIZES[tensor_type]
                    if row_meta_size > 0:
                        n_rows = 1
                        for dim in shape[1:]:
                            n_rows *= dim
                        n_bytes += n_rows * row_meta_size
                current_offset += n_bytes
                padding = (alignment - (current_offset % alignment)) % alignment
                current_offset += padding

            logger.info("Original data size: %d, new total data size: %d",
                        orig_data_end - target_reader.data_offset, current_offset)

            # Compute tensor info table size
            ti_size = 0
            for name, t in output_target_tensors:
                ti_size += 8 + len(name.encode("utf-8"))
                ti_size += 4
                ti_size += len(t.shape) * 8
                ti_size += 4
                ti_size += 8
            for name, t_dict in output_mtp_tensors:
                ti_size += 8 + len(name.encode("utf-8"))
                ti_size += 4
                ti_size += len(t_dict["shape"]) * 8
                ti_size += 4
                ti_size += 8

            tensor_info_start = 24 + len(kv_data)
            tensor_info_end = tensor_info_start + ti_size
            data_offset_padding = (alignment - (tensor_info_end % alignment)) % alignment
            data_start_abs = tensor_info_end + data_offset_padding

            logger.info("Tensor info table: offset=%d, size=%d", tensor_info_start, ti_size)
            logger.info("Tensor data starts at absolute offset %d", data_start_abs)

            # Write tensor info with RELATIVE offsets
            for name, t in output_target_tensors:
                write_string(fout, name.encode("utf-8"))
                shape = list(t.shape)
                fout.write(struct.pack("<I", len(shape)))
                for dim in shape:
                    fout.write(struct.pack("<Q", int(dim)))
                fout.write(struct.pack("<I", int(t.tensor_type)))
                rel_offset = offset_map[name]
                fout.write(struct.pack("<Q", rel_offset))

            for name, t_dict in output_mtp_tensors:
                write_string(fout, name.encode("utf-8"))
                shape = t_dict["shape"]
                fout.write(struct.pack("<I", len(shape)))
                for dim in shape:
                    fout.write(struct.pack("<Q", int(dim)))
                fout.write(struct.pack("<I", int(t_dict["tensor_type"])))
                rel_offset = offset_map[name]
                fout.write(struct.pack("<Q", rel_offset))

            # Pad to alignment before tensor data
            ti_end = fout.tell()
            padding = (alignment - (ti_end % alignment)) % alignment
            if padding > 0:
                fout.write(b"\x00" * padding)

            # Copy original tensor data
            new_data_start = fout.tell()
            logger.info("Tensor data section starts at offset %d (expected %d)", new_data_start, data_start_abs)

            fin.seek(target_reader.data_offset)
            orig_data = fin.read(orig_data_end - target_reader.data_offset)
            fout.write(orig_data)

            # Pad to where new tensors should start
            current_pos = fout.tell()
            expected_new_start = data_start_abs + (orig_data_end - target_reader.data_offset)
            padding = expected_new_start - current_pos
            if padding > 0:
                fout.write(b"\x00" * padding)

            # Write MTP tensor data
            for name, t_dict in output_mtp_tensors:
                expected_offset = data_start_abs + offset_map[name]
                current_pos = fout.tell()
                if current_pos != expected_offset:
                    padding = expected_offset - current_pos
                    if padding > 0:
                        fout.write(b"\x00" * padding)
                t_dict["data"].tofile(fout)
                tensor_type = t_dict["tensor_type"]
                shape = t_dict["shape"]
                n_bytes = t_dict["n_bytes"]
                if tensor_type in _ROW_META_SIZES and len(shape) > 1:
                    row_meta_size = _ROW_META_SIZES[tensor_type]
                    if row_meta_size > 0:
                        n_rows = 1
                        for dim in shape[1:]:
                            n_rows *= dim
                        extra = n_rows * row_meta_size
                        if extra > 0:
                            fout.write(b"\x00" * extra)

            final_size = fout.tell()
            logger.info("Final file size: %d", final_size)

    logger.info("Done! Output: %s", output_path)


def extract_merge(source_path: str, target_path: str, output_path: str) -> None:
    """Extract MTP from source and merge into target in one step."""
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as tmp:
        mtp_temp_path = tmp.name

    try:
        extract_mtp(source_path, mtp_temp_path)
        merge_mtp(mtp_temp_path, target_path, output_path)
    finally:
        if os.path.exists(mtp_temp_path):
            os.remove(mtp_temp_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract and/or merge MTP (Multi-Token Prediction) / NextN layers in GGUF models."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    extract_parser = subparsers.add_parser("extract", help="Extract MTP layers from a model")
    extract_parser.add_argument("source", type=str, help="Source GGUF model with MTP layers")
    extract_parser.add_argument("output", type=str, help="Output file for extracted MTP layers")

    merge_parser = subparsers.add_parser("merge", help="Merge extracted MTP layers into a model")
    merge_parser.add_argument("mtp", type=str, help="Extracted MTP layers file")
    merge_parser.add_argument("target", type=str, help="Target GGUF model without MTP")
    merge_parser.add_argument("output", type=str, help="Output GGUF model with MTP")

    em_parser = subparsers.add_parser("extract-merge", help="Extract MTP from source and merge into target")
    em_parser.add_argument("source", type=str, help="Source GGUF model with MTP layers")
    em_parser.add_argument("target", type=str, help="Target GGUF model without MTP")
    em_parser.add_argument("output", type=str, help="Output GGUF model with MTP")

    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s"
    )

    if args.command == "extract":
        extract_mtp(args.source, args.output)
    elif args.command == "merge":
        merge_mtp(args.mtp, args.target, args.output)
    elif args.command == "extract-merge":
        extract_merge(args.source, args.target, args.output)


if __name__ == "__main__":
    main()

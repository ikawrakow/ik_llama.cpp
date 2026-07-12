#!/usr/bin/env python3
import argparse
import json
import os
import re
from pathlib import Path

import torch
from gguf import GGUFEndian, GGUFWriter
from safetensors.torch import load_file


VISION_PREFIX = "vision_tower.vision_model."


def load_index(model_dir: Path) -> dict[str, str]:
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        with index_path.open("r", encoding="utf-8") as f:
            return json.load(f)["weight_map"]

    shards = sorted(model_dir.glob("*.safetensors"))
    if len(shards) == 1:
        tensors = load_file(str(shards[0]), device="cpu")
        return {name: shards[0].name for name in tensors}

    raise FileNotFoundError(f"unable to find safetensors index in {model_dir}")


def rename_tensor(name: str) -> str | None:
    if name == "vision_tower.vision_model.embeddings.patch_embedding.weight":
        return "v.patch_embd.weight"
    if name in (
        "vision_tower.vision_model.pre_layrnorm.weight",
        "vision_tower.vision_model.pre_layrnorm.bias",
    ):
        return name.replace("vision_tower.vision_model.pre_layrnorm", "v.pre_ln")

    if name.startswith("multi_modal_projector."):
        name = name.replace("multi_modal_projector.linear_1", "mm.0")
        name = name.replace("multi_modal_projector.linear_2", "mm.2")
        return name
    if name.startswith("patch_merge_mlp."):
        name = name.replace("patch_merge_mlp.linear_1", "mm.4")
        name = name.replace("patch_merge_mlp.linear_2", "mm.6")
        return name

    if not name.startswith(VISION_PREFIX + "encoder.layers."):
        return None

    name = name[len(VISION_PREFIX):]
    name = name.replace("encoder.layers", "blk")
    name = name.replace("layer_norm1", "ln1")
    name = name.replace("layer_norm2", "ln2")
    name = name.replace("self_attn.q_proj", "attn_q")
    name = name.replace("self_attn.k_proj", "attn_k")
    name = name.replace("self_attn.v_proj", "attn_v")
    name = name.replace("self_attn.out_proj", "attn_out")
    name = name.replace("mlp.fc1", "ffn_up")
    name = name.replace("mlp.fc2", "ffn_down")
    return "v." + name


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert MiniMax-M3 vision encoder/projector to GGUF")
    parser.add_argument("-m", "--model-dir", required=True, help="Path to MiniMax-M3 HF model directory")
    parser.add_argument("-o", "--output", default=None, help="Output GGUF path")
    parser.add_argument("--use-f32", action="store_true", help="Write tensors as f32 instead of f16")
    parser.add_argument("--bigendian", action="store_true", help="Write big-endian GGUF")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    config = read_json(model_dir / "config.json")
    vision_config = config["vision_config"]

    preprocessor_path = model_dir / "preprocessor_config.json"
    preprocessor = read_json(preprocessor_path) if preprocessor_path.exists() else {}
    compression = vision_config.get("img_token_compression_config", {})

    output = Path(args.output) if args.output else model_dir / "mmproj-minimax-m3-vl.gguf"
    ftype = 0 if args.use_f32 else 1

    writer = GGUFWriter(
        path=str(output),
        arch="clip",
        endianess=GGUFEndian.BIG if args.bigendian else GGUFEndian.LITTLE,
    )
    writer.add_bool("clip.has_text_encoder", False)
    writer.add_bool("clip.has_vision_encoder", True)
    writer.add_bool("clip.has_audio_encoder", False)
    writer.add_string("clip.projector_type", "minimax_m3_vl")
    writer.add_string("general.name", "MiniMax-M3 vision projector")
    writer.add_uint32("general.file_type", ftype)

    writer.add_uint32("clip.vision.image_size", vision_config["image_size"])
    writer.add_uint32("clip.vision.patch_size", vision_config["patch_size"])
    writer.add_uint32("clip.vision.embedding_length", vision_config["hidden_size"])
    writer.add_uint32("clip.vision.feed_forward_length", vision_config["intermediate_size"])
    writer.add_uint32("clip.vision.projection_dim", vision_config["projection_dim"])
    writer.add_uint32("clip.vision.attention.head_count", vision_config["num_attention_heads"])
    writer.add_uint32("clip.vision.block_count", vision_config["num_hidden_layers"])
    writer.add_float32("clip.vision.attention.layer_norm_epsilon", vision_config.get("layer_norm_eps", 1e-5))
    writer.add_uint32("clip.vision.spatial_merge_size", compression.get("spatial_merge_size", 2))
    writer.add_uint32("clip.vision.temporal_patch_size", compression.get("temporal_patch_size", 2))
    writer.add_uint32("clip.vision.image_min_pixels", preprocessor.get("min_pixels", 4 * 28 * 28))
    writer.add_uint32("clip.vision.image_max_pixels", preprocessor.get("max_pixels", 451584))
    writer.add_array("clip.vision.image_mean", preprocessor.get("image_mean", [0.48145466, 0.4578275, 0.40821073]))
    writer.add_array("clip.vision.image_std", preprocessor.get("image_std", [0.26862954, 0.26130258, 0.27577711]))
    writer.add_bool("clip.use_gelu", True)

    weight_map = load_index(model_dir)
    shard_cache: dict[str, dict[str, torch.Tensor]] = {}

    for src_name in sorted(weight_map):
        dst_name = rename_tensor(src_name)
        if dst_name is None:
            continue

        shard_name = weight_map[src_name]
        if shard_name not in shard_cache:
            shard_cache[shard_name] = load_file(str(model_dir / shard_name), device="cpu")

        data = shard_cache[shard_name][src_name]
        if src_name.endswith("patch_embedding.weight") and data.ndim == 5:
            if data.shape[2] != 2:
                raise ValueError(f"expected temporal_patch_size 2, got {data.shape[2]}")
            for i in range(data.shape[2]):
                patch_name = dst_name if i == 0 else f"{dst_name}.{i}"
                patch_data = data[:, :, i]
                if args.use_f32:
                    patch_data = patch_data.float()
                else:
                    patch_data = patch_data.half()
                writer.add_tensor(patch_name, patch_data.numpy())
            continue
        if args.use_f32:
            data = data.float()
        elif data.ndim == 2 and dst_name.endswith(".weight"):
            data = data.half()
        else:
            data = data.float()
        writer.add_tensor(dst_name, data.numpy())

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()

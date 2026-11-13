# Step-5-Preview dense text

This implements the reference's `sparse_config.enabled=False` path: full causal
attention on full-attention layers and ordinary 512-token sliding attention on
sliding layers. It does not implement sparse selection, MTP, SWA cache compression,
vision, or graph/attention tensor splitting. Dense attention uses more compute
than the released model's sparse path at long contexts.

Existing `step5` GGUFs can be used directly. Indexer tensors and appended MTP
blocks are explicitly skipped. The BF16 text converter preserves trunk indexer
tensors and sparse metadata for future use, while omitting MTP and vision weights.
It folds zero-centered RMSNorm weights into the engine's `w + 1` convention;
the indexer's plain LayerNorm weights and bias remain unchanged. Preserving sparse
metadata does not enable sparse execution on this dense-only branch.
Raw FP8 checkpoint conversion and nonidentity RoPE scaling are not supported here.

The new graph uses existing backend operations. Scheduler/offload policies,
existing architecture builders, and public APIs are unchanged. New architecture
registration, dispatch, metadata loading, and tensor maps are additive.

## Validation

Conversion fixtures require the converter's Python dependencies but no weights:

```sh
python tests/test-step5-conversion.py -v
```

Validated with a Linux/CUDA build, seven small conversion fixtures and manual
text generation with clean shutdown. A recall probe is a quality smoke test,
not proof of reference numerical parity. Windows and full CI are not yet tested.

#!/bin/bash

set -euo pipefail

if [ $# -lt 1 ]
then
    echo "usage:   $0 path_to_build_binary [path_to_temp_folder]"
    echo "example: $0 ../../build/bin ../../tmp"
    exit 1
fi

if [ $# -gt 1 ]
then
    TMP_DIR=$2
else
    TMP_DIR=/tmp
fi

set -x

SPLIT=$1/llama-gguf-split
QUANTIZE=$1/llama-quantize
MAIN=$1/llama-cli
WORK_PATH=$TMP_DIR/quantize
ROOT_DIR=$(realpath $(dirname $0)/../../)

mkdir -p "$WORK_PATH"

# Clean up in case of previously failed test
rm -f $WORK_PATH/ggml-model-split*.gguf $WORK_PATH/ggml-model-requant*.gguf

# 1. Get a model
(
cd $WORK_PATH
"$ROOT_DIR"/scripts/hf.sh --repo ggml-org/gemma-1.1-2b-it-Q8_0-GGUF --file gemma-1.1-2b-it.Q8_0.gguf
)
echo PASS

# 2. Split model
$SPLIT --split-max-tensors 28  $WORK_PATH/gemma-1.1-2b-it.Q8_0.gguf $WORK_PATH/ggml-model-split
echo PASS
echo

# 3. Requant model with '--keep-split'
$QUANTIZE --allow-requantize --keep-split $WORK_PATH/ggml-model-split-00001-of-00006.gguf $WORK_PATH/ggml-model-requant.gguf Q4_K
echo PASS
echo

# 3a. Test the requanted model is loading properly
$MAIN --model $WORK_PATH/ggml-model-requant-00001-of-00006.gguf --n-predict 32
echo PASS
echo

# 4. Requant mode without '--keep-split'
$QUANTIZE --allow-requantize $WORK_PATH/ggml-model-split-00001-of-00006.gguf $WORK_PATH/ggml-model-requant-merge.gguf Q4_K
echo PASS
echo

# 4b. Test the requanted model is loading properly
$MAIN --model $WORK_PATH/ggml-model-requant-merge.gguf --n-predict 32
echo PASS
echo

# 5. Warn when custom quantization rules are ignored for an unquantized default type
CUSTOM_Q_LOG=$WORK_PATH/custom-q-unquantized.log
$QUANTIZE --allow-requantize --dry-run --custom-q 'blk.*=q4_K' \
    $WORK_PATH/gemma-1.1-2b-it.Q8_0.gguf $WORK_PATH/ggml-model-custom-q.gguf F16 2>&1 | tee "$CUSTOM_Q_LOG"
grep -F "ignoring --custom-q rules because default type f16 is not quantized" "$CUSTOM_Q_LOG"
if grep -F "Using custom type" "$CUSTOM_Q_LOG"; then
    echo "custom quantization unexpectedly applied for an unquantized default type" >&2
    exit 1
fi
echo PASS

# 6. Apply custom quantization rules when the default type is quantized
CUSTOM_Q_APPLY_LOG=$WORK_PATH/custom-q-quantized.log
$QUANTIZE --allow-requantize --dry-run --custom-q '.*=q4_K' \
    $WORK_PATH/gemma-1.1-2b-it.Q8_0.gguf $WORK_PATH/ggml-model-custom-q-quantized.gguf Q4_K 2>&1 | tee "$CUSTOM_Q_APPLY_LOG"
if grep -F "ignoring --custom-q rules" "$CUSTOM_Q_APPLY_LOG"; then
    echo "custom quantization unexpectedly ignored for a quantized default type" >&2
    exit 1
fi
grep -F "Using custom type q4_K for tensor" "$CUSTOM_Q_APPLY_LOG"
echo PASS

# Clean up
rm -f $WORK_PATH/ggml-model-split*.gguf $WORK_PATH/ggml-model-requant*.gguf \
    $WORK_PATH/custom-q-unquantized.log $WORK_PATH/custom-q-quantized.log

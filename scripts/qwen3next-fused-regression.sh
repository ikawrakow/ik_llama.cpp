#!/usr/bin/env bash
set -euo pipefail

BIN="${BIN:-./build/bin/llama-perplexity}"
MODEL="${MODEL:-}"
INPUT_FILE="${INPUT_FILE:-/tmp/qwen3next_fused_regression_input.txt}"
OUT_FILE="${OUT_FILE:-/tmp/qwen3next_fused_regression_$(date +%Y%m%d_%H%M%S).md}"

CUDA_DEVICE="${CUDA_DEVICE:-0}"
THREADS="${THREADS:-8}"
CTX="${CTX:-2048}"
FA="${FA:-on}"
NGL="${NGL:-47}"
N_CPU_MOE="${N_CPU_MOE:-40}"
CHUNKS="${CHUNKS:-1}"

DECODE_B="${DECODE_B:-1}"
DECODE_UB="${DECODE_UB:-1}"
PREFILL_B="${PREFILL_B:-2048}"
PREFILL_UB="${PREFILL_UB:-512}"

# Mandatory safety checks:
# 1) mode=1 decode should stay aligned with mode=0 decode.
# 2) mode=1 prefill should stay aligned with mode=0 prefill.
MAX_DECODE_DELTA_01="${MAX_DECODE_DELTA_01:-0.10}"
MAX_PREFILL_DELTA_01="${MAX_PREFILL_DELTA_01:-0.10}"

usage() {
    cat <<'USAGE'
Usage:
  scripts/qwen3next-fused-regression.sh --model /path/to/model.gguf [options]

Options:
  --model PATH             GGUF model path (required)
  --bin PATH               llama-perplexity binary (default: ./build/bin/llama-perplexity)
  --input PATH             input text file; auto-generated if missing
  --out PATH               markdown output file
  --cuda-device ID         CUDA_VISIBLE_DEVICES value (default: 0)
  --threads N              -t value (default: 8)
  --ctx N                  -c value (default: 2048)
  --fa on|off              -fa value (default: on)
  --ngl N                  -ngl value (default: 47)
  --n-cpu-moe N            --n-cpu-moe value (default: 40)
  --chunks N               --chunks value (default: 1)
  --decode-b N             decode batch size (default: 1)
  --decode-ub N            decode ubatch size (default: 1)
  --prefill-b N            prefill batch size (default: 2048)
  --prefill-ub N           prefill ubatch size (default: 512)
  --max-decode-delta-01 X  fail threshold for |PPL(mode1)-PPL(mode0)| in decode (default: 0.10)
  --max-prefill-delta-01 X fail threshold for |PPL(mode1)-PPL(mode0)| in prefill (default: 0.10)
  -h, --help               show this help
USAGE
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model) MODEL="$2"; shift 2 ;;
        --bin) BIN="$2"; shift 2 ;;
        --input) INPUT_FILE="$2"; shift 2 ;;
        --out) OUT_FILE="$2"; shift 2 ;;
        --cuda-device) CUDA_DEVICE="$2"; shift 2 ;;
        --threads) THREADS="$2"; shift 2 ;;
        --ctx) CTX="$2"; shift 2 ;;
        --fa) FA="$2"; shift 2 ;;
        --ngl) NGL="$2"; shift 2 ;;
        --n-cpu-moe) N_CPU_MOE="$2"; shift 2 ;;
        --chunks) CHUNKS="$2"; shift 2 ;;
        --decode-b) DECODE_B="$2"; shift 2 ;;
        --decode-ub) DECODE_UB="$2"; shift 2 ;;
        --prefill-b) PREFILL_B="$2"; shift 2 ;;
        --prefill-ub) PREFILL_UB="$2"; shift 2 ;;
        --max-decode-delta-01) MAX_DECODE_DELTA_01="$2"; shift 2 ;;
        --max-prefill-delta-01) MAX_PREFILL_DELTA_01="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *)
            echo "unknown option: $1" >&2
            usage
            exit 2
            ;;
    esac
done

if [[ -z "$MODEL" ]]; then
    echo "--model is required" >&2
    exit 2
fi
if [[ ! -x "$BIN" ]]; then
    echo "binary not executable: $BIN" >&2
    exit 1
fi
if [[ ! -f "$MODEL" ]]; then
    echo "model not found: $MODEL" >&2
    exit 1
fi

if [[ ! -f "$INPUT_FILE" ]]; then
    cat > "$INPUT_FILE" <<'TXT'
Regression text for Qwen3Next fused DeltaNet checks.
This text is deterministic and intentionally repetitive.
TXT
    for _ in $(seq 1 500); do
        echo "the model should keep stable perplexity under consistent settings" >> "$INPUT_FILE"
    done
fi

log_dir="${OUT_FILE}.logs"
mkdir -p "$log_dir"

extract_ppl() {
    local file="$1"
    local line val
    line="$(grep -E 'Final estimate:' "$file" | tail -n1 || true)"
    if [[ -z "$line" ]]; then
        echo "NA"
        return
    fi
    val="$(echo "$line" | sed -nE 's/.*= ([0-9]+\.[0-9]+).*/\1/p')"
    if [[ -z "$val" ]]; then
        val="$(echo "$line" | grep -Eo '[0-9]+\.[0-9]+' | head -n1 || true)"
    fi
    if [[ -z "$val" ]]; then
        echo "NA"
    else
        echo "$val"
    fi
}

abs_delta() {
    awk -v a="$1" -v b="$2" 'BEGIN { d = a - b; if (d < 0) d = -d; printf "%.6f", d }'
}

run_ppl() {
    local mode="$1"
    local b="$2"
    local ub="$3"
    local label="$4"
    local log="${log_dir}/${label}_m${mode}.log"

    echo "running ${label} mode=${mode} (b=${b} ub=${ub})" >&2
    CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" \
    LLAMA_QWEN3NEXT_FUSED_DELTA="$mode" \
    "$BIN" -m "$MODEL" -f "$INPUT_FILE" \
        -c "$CTX" -b "$b" -ub "$ub" -t "$THREADS" \
        -fa "$FA" -ngl "$NGL" --n-cpu-moe "$N_CPU_MOE" \
        --chunks "$CHUNKS" --no-warmup >"$log" 2>&1

    extract_ppl "$log"
}

decode_0="$(run_ppl 0 "$DECODE_B" "$DECODE_UB" decode)"
decode_1="$(run_ppl 1 "$DECODE_B" "$DECODE_UB" decode)"
decode_2="$(run_ppl 2 "$DECODE_B" "$DECODE_UB" decode)"

prefill_0="$(run_ppl 0 "$PREFILL_B" "$PREFILL_UB" prefill)"
prefill_1="$(run_ppl 1 "$PREFILL_B" "$PREFILL_UB" prefill)"
prefill_2="$(run_ppl 2 "$PREFILL_B" "$PREFILL_UB" prefill)"

if [[ "$decode_0" == "NA" || "$decode_1" == "NA" || "$decode_2" == "NA" || \
      "$prefill_0" == "NA" || "$prefill_1" == "NA" || "$prefill_2" == "NA" ]]; then
    echo "failed to extract one or more perplexity values; see logs in ${log_dir}" >&2
    exit 1
fi

decode_delta_01="$(abs_delta "$decode_0" "$decode_1")"
decode_delta_02="$(abs_delta "$decode_0" "$decode_2")"
prefill_delta_01="$(abs_delta "$prefill_0" "$prefill_1")"
prefill_delta_02="$(abs_delta "$prefill_0" "$prefill_2")"

decode_ok="$(awk -v d="$decode_delta_01" -v t="$MAX_DECODE_DELTA_01" 'BEGIN { print(d <= t ? "yes" : "no") }')"
prefill_ok="$(awk -v d="$prefill_delta_01" -v t="$MAX_PREFILL_DELTA_01" 'BEGIN { print(d <= t ? "yes" : "no") }')"

{
    echo "# Qwen3Next Fused DeltaNet Regression Report"
    echo
    echo "- date: \`$(date -u +%Y-%m-%dT%H:%M:%SZ)\`"
    echo "- bin: \`$BIN\`"
    echo "- model: \`$MODEL\`"
    echo "- input: \`$INPUT_FILE\`"
    echo "- cuda_device: \`$CUDA_DEVICE\`"
    echo "- ctx: \`$CTX\`"
    echo "- fa: \`$FA\`"
    echo "- ngl: \`$NGL\`"
    echo "- n_cpu_moe: \`$N_CPU_MOE\`"
    echo "- chunks: \`$CHUNKS\`"
    echo
    echo "## Perplexity"
    echo
    echo "| Path | mode=0 | mode=1 | mode=2 | |delta|(1-0) | |delta|(2-0) |"
    echo "|---|---:|---:|---:|---:|---:|"
    echo "| decode (b=${DECODE_B},ub=${DECODE_UB}) | ${decode_0} | ${decode_1} | ${decode_2} | ${decode_delta_01} | ${decode_delta_02} |"
    echo "| prefill (b=${PREFILL_B},ub=${PREFILL_UB}) | ${prefill_0} | ${prefill_1} | ${prefill_2} | ${prefill_delta_01} | ${prefill_delta_02} |"
    echo
    echo "## Safety Checks"
    echo
    echo "- decode safety (mode1 ~= mode0): \`${decode_ok}\` (threshold \`${MAX_DECODE_DELTA_01}\`)"
    echo "- prefill safety (mode1 ~= mode0): \`${prefill_ok}\` (threshold \`${MAX_PREFILL_DELTA_01}\`)"
    echo
    echo "## Logs"
    echo
    echo "- raw logs dir: \`${log_dir}\`"
    echo "- decode mode0: \`${log_dir}/decode_m0.log\`"
    echo "- decode mode1: \`${log_dir}/decode_m1.log\`"
    echo "- decode mode2: \`${log_dir}/decode_m2.log\`"
    echo "- prefill mode0: \`${log_dir}/prefill_m0.log\`"
    echo "- prefill mode1: \`${log_dir}/prefill_m1.log\`"
    echo "- prefill mode2: \`${log_dir}/prefill_m2.log\`"
} > "$OUT_FILE"

echo "wrote report: $OUT_FILE"

if [[ "$decode_ok" != "yes" || "$prefill_ok" != "yes" ]]; then
    echo "regression check failed; see report: $OUT_FILE" >&2
    exit 1
fi

echo "regression check passed"

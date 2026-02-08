#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04}"
IK_REPO="${IK_REPO:-/home/yurko/Code/ik_llama.cpp}"
IK_BUILD_DIR="${IK_BUILD_DIR:-build}"
MODEL_HOST="${MODEL_HOST:-/home/yurko/.cache/llama.cpp/qwen3-next-coder.gguf}"
OUT_ROOT="${OUT_ROOT:-/tmp/qwen3next-regression}"
GPU_DEVICE="${GPU_DEVICE:-0}"

THREADS="${THREADS:-8}"
FA="${FA:-on}"
NGL="${NGL:-999}"

PROXY_CTX="${PROXY_CTX:-8192}"
PROXY_B="${PROXY_B:-3072}"
PROXY_UB="${PROXY_UB:-768}"
PROXY_N="${PROXY_N:-128}"
PROXY_N_CPU_MOE="${PROXY_N_CPU_MOE:-40}"

REG_CTX="${REG_CTX:-2048}"
REG_NGL="${REG_NGL:-47}"
REG_DECODE_B="${REG_DECODE_B:-1}"
REG_DECODE_UB="${REG_DECODE_UB:-1}"
REG_PREFILL_B="${REG_PREFILL_B:-2048}"
REG_PREFILL_UB="${REG_PREFILL_UB:-512}"

WITH_FIT=1
FIT_CTX="${FIT_CTX:-65536}"
FIT_N_CPU_MOE="${FIT_N_CPU_MOE:-47}"
FIT_N="${FIT_N:-1}"

usage() {
    cat <<'USAGE'
Usage:
  scripts/qwen3next-regression.sh [options]

Options:
  --image IMAGE              Docker image to run checks in (default: nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04)
  --ik-repo PATH             ik repo path (default: /home/yurko/Code/ik_llama.cpp)
  --ik-build-dir NAME        Build dir under ik repo (default: build)
  --model PATH               Host path to model GGUF file
  --out-root PATH            Output root directory (default: /tmp/qwen3next-regression)
  --gpu-device ID            CUDA device id (default: 0)
  --threads N                Threads (default: 8)
  --fa on|off                Flash attention mode (default: on)
  --ngl N                    -ngl value (default: 999)

  --proxy-ctx N              Proxy sweep context (default: 8192)
  --proxy-b N                Proxy sweep batch size (default: 3072)
  --proxy-ub N               Proxy sweep ubatch size (default: 768)
  --proxy-n N                Proxy sweep generation tokens (default: 128)
  --proxy-n-cpu-moe N        Proxy sweep --n-cpu-moe (default: 40)

  --reg-ctx N                Fused regression context (default: 2048)
  --reg-ngl N                Fused regression -ngl (default: 47)
  --reg-decode-b N           Fused regression decode b (default: 1)
  --reg-decode-ub N          Fused regression decode ub (default: 1)
  --reg-prefill-b N          Fused regression prefill b (default: 2048)
  --reg-prefill-ub N         Fused regression prefill ub (default: 512)

  --fit-ctx N                Long-context fit sanity context (default: 65536)
  --fit-n-cpu-moe N          Long-context fit sanity --n-cpu-moe (default: 47)
  --fit-n N                  Long-context fit sanity generation tokens (default: 1)
  --no-fit                   Skip long-context fit sanity
  -h, --help                 Show this help

Runs:
  1) Fused-delta regression guard (mode0/mode1/mode2 + sanity thresholds)
  2) Single-GPU proxy sweep benchmark
  3) Optional long-context fit sanity
USAGE
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --image) IMAGE="$2"; shift 2 ;;
        --ik-repo) IK_REPO="$2"; shift 2 ;;
        --ik-build-dir) IK_BUILD_DIR="$2"; shift 2 ;;
        --model) MODEL_HOST="$2"; shift 2 ;;
        --out-root) OUT_ROOT="$2"; shift 2 ;;
        --gpu-device) GPU_DEVICE="$2"; shift 2 ;;
        --threads) THREADS="$2"; shift 2 ;;
        --fa) FA="$2"; shift 2 ;;
        --ngl) NGL="$2"; shift 2 ;;
        --proxy-ctx) PROXY_CTX="$2"; shift 2 ;;
        --proxy-b) PROXY_B="$2"; shift 2 ;;
        --proxy-ub) PROXY_UB="$2"; shift 2 ;;
        --proxy-n) PROXY_N="$2"; shift 2 ;;
        --proxy-n-cpu-moe) PROXY_N_CPU_MOE="$2"; shift 2 ;;
        --reg-ctx) REG_CTX="$2"; shift 2 ;;
        --reg-ngl) REG_NGL="$2"; shift 2 ;;
        --reg-decode-b) REG_DECODE_B="$2"; shift 2 ;;
        --reg-decode-ub) REG_DECODE_UB="$2"; shift 2 ;;
        --reg-prefill-b) REG_PREFILL_B="$2"; shift 2 ;;
        --reg-prefill-ub) REG_PREFILL_UB="$2"; shift 2 ;;
        --fit-ctx) FIT_CTX="$2"; shift 2 ;;
        --fit-n-cpu-moe) FIT_N_CPU_MOE="$2"; shift 2 ;;
        --fit-n) FIT_N="$2"; shift 2 ;;
        --no-fit) WITH_FIT=0; shift ;;
        -h|--help) usage; exit 0 ;;
        *)
            echo "Unknown option: $1" >&2
            usage
            exit 2
            ;;
    esac
done

if [[ ! -d "$IK_REPO" ]]; then
    echo "ik repo does not exist: $IK_REPO" >&2
    exit 1
fi
if [[ ! -f "$MODEL_HOST" ]]; then
    echo "Model file does not exist: $MODEL_HOST" >&2
    exit 1
fi

run_id="$(date +%Y%m%d_%H%M%S)"
out_dir="${OUT_ROOT%/}/${run_id}"
mkdir -p "$out_dir"

cat > "${out_dir}/run_inside.sh" <<'BASH'
#!/usr/bin/env bash
set -euo pipefail

IK_BUILD_DIR="${IK_BUILD_DIR:-build}"
GPU_DEVICE="${GPU_DEVICE:-0}"
THREADS="${THREADS:-8}"
FA="${FA:-on}"
NGL="${NGL:-999}"

PROXY_CTX="${PROXY_CTX:-8192}"
PROXY_B="${PROXY_B:-3072}"
PROXY_UB="${PROXY_UB:-768}"
PROXY_N="${PROXY_N:-128}"
PROXY_N_CPU_MOE="${PROXY_N_CPU_MOE:-40}"

REG_CTX="${REG_CTX:-2048}"
REG_NGL="${REG_NGL:-47}"
REG_DECODE_B="${REG_DECODE_B:-1}"
REG_DECODE_UB="${REG_DECODE_UB:-1}"
REG_PREFILL_B="${REG_PREFILL_B:-2048}"
REG_PREFILL_UB="${REG_PREFILL_UB:-512}"

WITH_FIT="${WITH_FIT:-1}"
FIT_CTX="${FIT_CTX:-65536}"
FIT_N_CPU_MOE="${FIT_N_CPU_MOE:-47}"
FIT_N="${FIT_N:-1}"

IK_BIN="/ik/${IK_BUILD_DIR}/bin"
IK_LD="/ik/${IK_BUILD_DIR}/bin:/ik/${IK_BUILD_DIR}/src:/ik/${IK_BUILD_DIR}/ggml/src:/ik/${IK_BUILD_DIR}/examples/mtmd"
MODEL="/model.gguf"

RUN_LOG="/out/run.log"
STATUS_FILE="/out/status.tsv"

touch "$RUN_LOG"
printf "name\tstatus\texit_code\n" > "$STATUS_FILE"

log() {
    local msg="$1"
    printf "[%s] %s\n" "$(date +%H:%M:%S)" "$msg" | tee -a "$RUN_LOG"
}

run_cmd() {
    local name="$1"
    shift
    local out_file="/out/${name}.out"
    local err_file="/out/${name}.err"
    local ec

    log "RUN: $name"
    set +e
    "$@" >"$out_file" 2>"$err_file"
    ec=$?
    set -e

    if [[ $ec -eq 0 ]]; then
        printf "%s\tOK\t0\n" "$name" >> "$STATUS_FILE"
        log "OK: $name"
    else
        printf "%s\tFAIL\t%d\n" "$name" "$ec" >> "$STATUS_FILE"
        log "FAIL($ec): $name"
    fi
    return $ec
}

require_bin() {
    local path="$1"
    if [[ ! -x "$path" ]]; then
        log "MISSING: $path"
        exit 1
    fi
}

extract_best_metric() {
    local out_file="$1"
    local err_file="$2"
    local col="$3"
    awk -F'|' -v c="$col" '
        /^\|[[:space:]]*[0-9]+[[:space:]]*\|/ {
            v = $c
            gsub(/[[:space:]]/, "", v)
            if ((v + 0) > best) {
                best = v + 0
                row = $0
            }
        }
        END {
            if (best > 0) {
                printf "%.2f\t%s\n", best, row
            } else {
                print "NA\tNA"
            }
        }
    ' < <(cat "$out_file" "$err_file")
}

require_bin "$IK_BIN/llama-perplexity"
require_bin "$IK_BIN/llama-sweep-bench"
require_bin "$IK_BIN/llama-cli"
require_bin "/ik/scripts/qwen3next-fused-regression.sh"

export CUDA_VISIBLE_DEVICES="$GPU_DEVICE"
log "GPU checks on CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

run_cmd "fused_regression" \
    env LD_LIBRARY_PATH="$IK_LD" /ik/scripts/qwen3next-fused-regression.sh \
        --model "$MODEL" \
        --bin "$IK_BIN/llama-perplexity" \
        --out /out/fused_regression.md \
        --cuda-device "$GPU_DEVICE" \
        --threads "$THREADS" \
        --ctx "$REG_CTX" \
        --fa "$FA" \
        --ngl "$REG_NGL" \
        --n-cpu-moe "$PROXY_N_CPU_MOE" \
        --chunks 1 \
        --decode-b "$REG_DECODE_B" \
        --decode-ub "$REG_DECODE_UB" \
        --prefill-b "$REG_PREFILL_B" \
        --prefill-ub "$REG_PREFILL_UB" || true

run_cmd "proxy_sweep" \
    env LD_LIBRARY_PATH="$IK_LD" "$IK_BIN/llama-sweep-bench" \
        -m "$MODEL" \
        -c "$PROXY_CTX" \
        -b "$PROXY_B" \
        -ub "$PROXY_UB" \
        -n "$PROXY_N" \
        -t "$THREADS" \
        -fa "$FA" \
        --jinja \
        -ngl "$NGL" \
        --n-cpu-moe "$PROXY_N_CPU_MOE" \
        -rtr \
        --temp 1 \
        --top-p 0.95 \
        --top-k 40 \
        --min-p 0.01 || true

if [[ "$WITH_FIT" == "1" ]]; then
    run_cmd "fit_sanity" \
        env LD_LIBRARY_PATH="$IK_LD" "$IK_BIN/llama-cli" \
            -m "$MODEL" \
            -c "$FIT_CTX" \
            -n "$FIT_N" \
            -t "$THREADS" \
            -fa "$FA" \
            -ngl "$NGL" \
            --n-cpu-moe "$FIT_N_CPU_MOE" \
            -rtr \
            --temp 0 \
            --top-k 1 \
            --simple-io \
            --no-display-prompt \
            -p "ping" || true
else
    printf "%s\tSKIP\t0\n" "fit_sanity" >> "$STATUS_FILE"
    log "SKIP: fit_sanity"
fi

fused_decode_safe="NA"
fused_prefill_safe="NA"
fused_mode0_decode_sane="NA"
fused_mode0_prefill_sane="NA"
if [[ -f /out/fused_regression.md ]]; then
    fused_decode_safe="$(sed -nE 's/^- decode safety .*: `([^`]+)`.*/\1/p' /out/fused_regression.md | tail -n1 || true)"
    fused_prefill_safe="$(sed -nE 's/^- prefill safety .*: `([^`]+)`.*/\1/p' /out/fused_regression.md | tail -n1 || true)"
    fused_mode0_decode_sane="$(sed -nE 's/^- mode0 decode sanity: `([^`]+)`.*/\1/p' /out/fused_regression.md | tail -n1 || true)"
    fused_mode0_prefill_sane="$(sed -nE 's/^- mode0 prefill sanity: `([^`]+)`.*/\1/p' /out/fused_regression.md | tail -n1 || true)"
    if [[ -z "$fused_decode_safe" ]]; then fused_decode_safe="NA"; fi
    if [[ -z "$fused_prefill_safe" ]]; then fused_prefill_safe="NA"; fi
    if [[ -z "$fused_mode0_decode_sane" ]]; then fused_mode0_decode_sane="NA"; fi
    if [[ -z "$fused_mode0_prefill_sane" ]]; then fused_mode0_prefill_sane="NA"; fi
fi

best_pp_tsv="$(extract_best_metric /out/proxy_sweep.out /out/proxy_sweep.err 6)"
best_tg_tsv="$(extract_best_metric /out/proxy_sweep.out /out/proxy_sweep.err 8)"
best_pp="${best_pp_tsv%%$'\t'*}"
best_pp_row="${best_pp_tsv#*$'\t'}"
best_tg="${best_tg_tsv%%$'\t'*}"
best_tg_row="${best_tg_tsv#*$'\t'}"

{
    echo "# Qwen3Next Regression Summary"
    echo
    echo "## Fused Regression"
    echo "- config: \`ctx=${REG_CTX}, decode(b=${REG_DECODE_B},ub=${REG_DECODE_UB}), prefill(b=${REG_PREFILL_B},ub=${REG_PREFILL_UB}), n-cpu-moe=${PROXY_N_CPU_MOE}\`"
    echo "- decode safety: \`$fused_decode_safe\`"
    echo "- prefill safety: \`$fused_prefill_safe\`"
    echo "- mode0 decode sanity: \`$fused_mode0_decode_sane\`"
    echo "- mode0 prefill sanity: \`$fused_mode0_prefill_sane\`"
    echo "- report: \`/out/fused_regression.md\`"
    echo
    echo "## Proxy Sweep"
    echo "- config: \`c=${PROXY_CTX}, b=${PROXY_B}, ub=${PROXY_UB}, n=${PROXY_N}, n-cpu-moe=${PROXY_N_CPU_MOE}\`"
    echo "- best PP t/s: \`$best_pp\`"
    echo "- best TG t/s: \`$best_tg\`"
    echo "- best PP row: \`$best_pp_row\`"
    echo "- best TG row: \`$best_tg_row\`"
    echo
    echo "## Long-Context Fit"
    if [[ "$WITH_FIT" == "1" ]]; then
        echo "- config: \`c=${FIT_CTX}, n-cpu-moe=${FIT_N_CPU_MOE}, n=${FIT_N}\`"
        echo "- output: \`/out/fit_sanity.out\`"
    else
        echo "- skipped"
    fi
    echo
    echo "## Command Status"
    echo '```'
    cat "$STATUS_FILE"
    echo '```'
} > /out/SUMMARY.md

log "Summary written to /out/SUMMARY.md"
BASH

chmod +x "${out_dir}/run_inside.sh"

docker_cmd=(
    docker run --rm --gpus all
    -e IK_BUILD_DIR="${IK_BUILD_DIR}"
    -e GPU_DEVICE="${GPU_DEVICE}"
    -e THREADS="${THREADS}"
    -e FA="${FA}"
    -e NGL="${NGL}"
    -e PROXY_CTX="${PROXY_CTX}"
    -e PROXY_B="${PROXY_B}"
    -e PROXY_UB="${PROXY_UB}"
    -e PROXY_N="${PROXY_N}"
    -e PROXY_N_CPU_MOE="${PROXY_N_CPU_MOE}"
    -e REG_CTX="${REG_CTX}"
    -e REG_NGL="${REG_NGL}"
    -e REG_DECODE_B="${REG_DECODE_B}"
    -e REG_DECODE_UB="${REG_DECODE_UB}"
    -e REG_PREFILL_B="${REG_PREFILL_B}"
    -e REG_PREFILL_UB="${REG_PREFILL_UB}"
    -e WITH_FIT="${WITH_FIT}"
    -e FIT_CTX="${FIT_CTX}"
    -e FIT_N_CPU_MOE="${FIT_N_CPU_MOE}"
    -e FIT_N="${FIT_N}"
    -v "${IK_REPO}:/ik"
    -v "${MODEL_HOST}:/model.gguf:ro"
    -v "${out_dir}:/out"
    "${IMAGE}" /bin/bash /out/run_inside.sh
)

echo "Running regression in container: ${IMAGE}"
echo "Output directory: ${out_dir}"
"${docker_cmd[@]}"

echo
echo "Done. Summary:"
echo "  ${out_dir}/SUMMARY.md"
echo "Raw logs:"
echo "  ${out_dir}/*.out"
echo "  ${out_dir}/*.err"

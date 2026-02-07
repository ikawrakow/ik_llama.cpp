#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-iktest-dev:latest}"
MAIN_REPO="${MAIN_REPO:-/home/yurko/Code/llama.cpp}"
IK_REPO="${IK_REPO:-/home/yurko/Code/ik_llama.cpp}"
MAIN_BUILD_DIR="${MAIN_BUILD_DIR:-build}"
IK_BUILD_DIR="${IK_BUILD_DIR:-build}"
MODEL_HOST="${MODEL_HOST:-/home/yurko/.cache/llama.cpp/qwen3-next-coder.gguf}"
OUT_ROOT="${OUT_ROOT:-/tmp/qwen3next-eval}"
WITH_GPU=0
GPU_DEVICE="${GPU_DEVICE:-0}"
SWEEP_CTX="${SWEEP_CTX:-2048}"
SWEEP_N="${SWEEP_N:-32}"

usage() {
    cat <<'USAGE'
Usage:
  scripts/qwen3next-eval.sh [options]

Options:
  --with-gpu                 Enable GPU checks in addition to CPU checks.
  --gpu-device ID            CUDA device id to use for GPU sanity checks (default: 0).
  --image IMAGE              Docker image to run checks in (default: iktest-dev:latest).
  --main-repo PATH           Mainline repo path (default: /home/yurko/Code/llama.cpp).
  --ik-repo PATH             ik repo path (default: /home/yurko/Code/ik_llama.cpp).
  --main-build-dir NAME      Mainline build dir under main repo (default: build).
  --ik-build-dir NAME        ik build dir under ik repo (default: build).
  --model PATH               Host path to model GGUF file.
  --out-root PATH            Output root directory (default: /tmp/qwen3next-eval).
  --sweep-ctx N              Sweep context size for PP/TG check (default: 2048).
  --sweep-n N                Sweep generation tokens (default: 32).
  -h, --help                 Show this help.

What this script runs (in this order):
  1) CPU perplexity parity (chunks=1)      mainline -> ik
  2) CPU perplexity parity (chunks=2)      mainline -> ik
  3) CPU short generation smoke quality    mainline -> ik
  4) Optional GPU sanity checks            mainline -> ik

Output:
  A timestamped folder is created under OUT_ROOT with:
  - SUMMARY.md
  - run.log
  - *.out / *.err logs for each command
USAGE
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --with-gpu)
            WITH_GPU=1
            shift
            ;;
        --gpu-device)
            GPU_DEVICE="$2"
            shift 2
            ;;
        --image)
            IMAGE="$2"
            shift 2
            ;;
        --main-repo)
            MAIN_REPO="$2"
            shift 2
            ;;
        --ik-repo)
            IK_REPO="$2"
            shift 2
            ;;
        --main-build-dir)
            MAIN_BUILD_DIR="$2"
            shift 2
            ;;
        --ik-build-dir)
            IK_BUILD_DIR="$2"
            shift 2
            ;;
        --model)
            MODEL_HOST="$2"
            shift 2
            ;;
        --out-root)
            OUT_ROOT="$2"
            shift 2
            ;;
        --sweep-ctx)
            SWEEP_CTX="$2"
            shift 2
            ;;
        --sweep-n)
            SWEEP_N="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage
            exit 2
            ;;
    esac
done

if [[ ! -d "$MAIN_REPO" ]]; then
    echo "Mainline repo does not exist: $MAIN_REPO" >&2
    exit 1
fi
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

cat > "${out_dir}/ppl_input.txt" <<'TXT'
Deterministic evaluation text for quick perplexity parity checks.
The next lines intentionally repeat a simple pattern to reduce variance.
TXT
for _ in $(seq 1 400); do
    echo "the system writes logs and the system reads logs" >> "${out_dir}/ppl_input.txt"
done

cat > "${out_dir}/gen_prompt.txt" <<'TXT'
Write a concise Python function that returns the first n Fibonacci numbers iteratively, and then give one sentence explaining time complexity.
TXT

cat > "${out_dir}/run_inside.sh" <<'BASH'
#!/usr/bin/env bash
set -euo pipefail

WITH_GPU="${WITH_GPU:-0}"
GPU_DEVICE="${GPU_DEVICE:-0}"
SWEEP_CTX="${SWEEP_CTX:-2048}"
SWEEP_N="${SWEEP_N:-32}"
MAIN_BUILD_DIR="${MAIN_BUILD_DIR:-build}"
IK_BUILD_DIR="${IK_BUILD_DIR:-build}"

MAIN_BIN="/mainline/${MAIN_BUILD_DIR}/bin"
IK_BIN="/ik/${IK_BUILD_DIR}/bin"
MAIN_LD="/mainline/${MAIN_BUILD_DIR}/bin:/mainline/${MAIN_BUILD_DIR}/src:/mainline/${MAIN_BUILD_DIR}/ggml/src:/mainline/${MAIN_BUILD_DIR}/examples/mtmd"
IK_LD="/ik/${IK_BUILD_DIR}/bin:/ik/${IK_BUILD_DIR}/src:/ik/${IK_BUILD_DIR}/ggml/src:/ik/${IK_BUILD_DIR}/examples/mtmd"
MODEL="/model.gguf"

RUN_LOG="/out/run.log"
STATUS_FILE="/out/status.tsv"

touch "$RUN_LOG"
printf "name\tstatus\texit_code\thost_mem_used_before_mib\thost_mem_used_after_mib\tgpu_mem_used_before_mib\tgpu_mem_used_after_mib\tmax_rss_kib\telapsed\n" > "$STATUS_FILE"

log() {
    local msg="$1"
    printf "[%s] %s\n" "$(date +%H:%M:%S)" "$msg" | tee -a "$RUN_LOG"
}

require_bin() {
    local path="$1"
    if [[ ! -x "$path" ]]; then
        log "MISSING: $path"
        return 1
    fi
}

host_mem_used_mib() {
    awk '
        /MemTotal:/     { mt = $2 }
        /MemAvailable:/ { ma = $2 }
        END {
            if (mt > 0 && ma >= 0) {
                printf "%.1f", (mt - ma) / 1024.0
            } else {
                print "NA"
            }
        }
    ' /proc/meminfo
}

gpu_mem_used_mib() {
    if [[ "$WITH_GPU" != "1" ]]; then
        echo "NA"
        return
    fi
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "NA"
        return
    fi
    local used
    used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr '\n' ',' | sed 's/,$//' || true)"
    if [[ -z "$used" ]]; then
        echo "NA"
    else
        echo "$used"
    fi
}

extract_max_rss_kib() {
    local time_file="$1"
    if [[ ! -f "$time_file" ]]; then
        echo "NA"
        return
    fi
    local rss
    rss="$(grep -E '^Maximum resident set size' "$time_file" | awk '{print $6}' | tail -n1 || true)"
    if [[ -z "$rss" ]]; then
        echo "NA"
    else
        echo "$rss"
    fi
}

extract_elapsed() {
    local time_file="$1"
    if [[ ! -f "$time_file" ]]; then
        echo "NA"
        return
    fi
    local elapsed
    elapsed="$(grep -E '^Elapsed \(wall clock\) time' "$time_file" | sed -E 's/^[^:]+:[[:space:]]*//' | tail -n1 || true)"
    if [[ -z "$elapsed" ]]; then
        echo "NA"
    else
        echo "$elapsed"
    fi
}

run_cmd() {
    local name="$1"
    shift
    local out_file="/out/${name}.out"
    local err_file="/out/${name}.err"
    local time_file="/out/${name}.time"
    local ec
    local host_before host_after gpu_before gpu_after max_rss elapsed

    host_before="$(host_mem_used_mib)"
    gpu_before="$(gpu_mem_used_mib)"
    log "RUN: $name"

    set +e
    if [[ -x /usr/bin/time ]]; then
        /usr/bin/time -v -o "$time_file" "$@" >"$out_file" 2>"$err_file"
        ec=$?
    else
        "$@" >"$out_file" 2>"$err_file"
        ec=$?
    fi
    set -e

    host_after="$(host_mem_used_mib)"
    gpu_after="$(gpu_mem_used_mib)"
    max_rss="$(extract_max_rss_kib "$time_file")"
    elapsed="$(extract_elapsed "$time_file")"

    if [[ $ec -eq 0 ]]; then
        printf "%s\tOK\t0\t%s\t%s\t%s\t%s\t%s\t%s\n" \
            "$name" "$host_before" "$host_after" "$gpu_before" "$gpu_after" "$max_rss" "$elapsed" >> "$STATUS_FILE"
        log "OK: $name"
    else
        printf "%s\tFAIL\t%d\t%s\t%s\t%s\t%s\t%s\t%s\n" \
            "$name" "$ec" "$host_before" "$host_after" "$gpu_before" "$gpu_after" "$max_rss" "$elapsed" >> "$STATUS_FILE"
        log "FAIL($ec): $name"
    fi
    return $ec
}

extract_ppl() {
    local out_file="$1"
    local err_file="$2"
    local line num

    line="$(cat "$out_file" "$err_file" 2>/dev/null | grep -E "Final estimate:" | tail -n1 || true)"
    if [[ -z "$line" ]]; then
        echo "NA"
        return
    fi

    num="$(echo "$line" | sed -nE 's/.*= ([0-9]+\.[0-9]+).*/\1/p')"
    if [[ -z "$num" ]]; then
        num="$(echo "$line" | grep -Eo '[0-9]+\.[0-9]+' | head -n1 || true)"
    fi
    if [[ -z "$num" ]]; then
        echo "NA"
    else
        echo "$num"
    fi
}

abs_delta() {
    local a="$1"
    local b="$2"
    awk -v a="$a" -v b="$b" 'BEGIN { d = a - b; if (d < 0) d = -d; printf "%.6f", d }'
}

has_token() {
    local file="$1"
    local pattern="$2"
    if grep -Eiq "$pattern" "$file"; then
        echo "yes"
    else
        echo "no"
    fi
}

main_ppl() {
    LD_LIBRARY_PATH="$MAIN_LD" "$MAIN_BIN/llama-perplexity" "$@"
}

ik_ppl() {
    LD_LIBRARY_PATH="$IK_LD" "$IK_BIN/llama-perplexity" "$@"
}

main_cli() {
    LD_LIBRARY_PATH="$MAIN_LD" "$MAIN_BIN/llama-cli" "$@"
}

main_completion() {
    LD_LIBRARY_PATH="$MAIN_LD" "$MAIN_BIN/llama-completion" "$@"
}

ik_cli() {
    LD_LIBRARY_PATH="$IK_LD" "$IK_BIN/llama-cli" "$@"
}

main_sweep() {
    LD_LIBRARY_PATH="$MAIN_LD" "$MAIN_BIN/llama-sweep-bench" "$@"
}

ik_sweep() {
    LD_LIBRARY_PATH="$IK_LD" "$IK_BIN/llama-sweep-bench" "$@"
}

require_bin "$MAIN_BIN/llama-perplexity"
require_bin "$MAIN_BIN/llama-cli"
require_bin "$MAIN_BIN/llama-completion"
require_bin "$IK_BIN/llama-perplexity"
require_bin "$IK_BIN/llama-cli"

if [[ "$WITH_GPU" != "1" ]]; then
    export CUDA_VISIBLE_DEVICES=""
    log "GPU checks disabled (CPU-only mode)"
else
    export CUDA_VISIBLE_DEVICES="$GPU_DEVICE"
    log "GPU checks enabled on CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
fi

PPL_INPUT="/out/ppl_input.txt"
GEN_PROMPT="$(cat /out/gen_prompt.txt)"

# CPU perplexity: chunks=1 (mainline -> ik)
run_cmd "cpu_ppl_chunks1_mainline" \
    main_ppl -m "$MODEL" -f "$PPL_INPUT" -c 256 -b 64 -ub 64 --chunks 1 --no-warmup -ngl 0 || true
run_cmd "cpu_ppl_chunks1_ik" \
    ik_ppl -m "$MODEL" -f "$PPL_INPUT" -c 256 -b 64 -ub 64 --chunks 1 --no-warmup -ngl 0 || true

# CPU perplexity: chunks=2 (mainline -> ik)
run_cmd "cpu_ppl_chunks2_mainline" \
    main_ppl -m "$MODEL" -f "$PPL_INPUT" -c 256 -b 64 -ub 64 --chunks 2 --no-warmup -ngl 0 || true
run_cmd "cpu_ppl_chunks2_ik" \
    ik_ppl -m "$MODEL" -f "$PPL_INPUT" -c 256 -b 64 -ub 64 --chunks 2 --no-warmup -ngl 0 || true

# CPU short generation smoke quality (mainline -> ik)
run_cmd "cpu_gen_mainline" \
    main_completion -m "$MODEL" --cpu-moe -ngl 0 -c 512 -n 64 --seed 123 --temp 0 --top-k 1 --simple-io --no-display-prompt -p "$GEN_PROMPT" || true
run_cmd "cpu_gen_ik" \
    ik_cli -m "$MODEL" --cpu-moe -ngl 0 -c 512 -n 64 --seed 123 --temp 0 --top-k 1 --simple-io --no-display-prompt -p "$GEN_PROMPT" || true

if [[ "$WITH_GPU" == "1" ]]; then
    # CUDA sanity perplexity: chunks=1 (mainline -> ik)
    run_cmd "gpu_ppl_chunks1_mainline" \
        main_ppl -m "$MODEL" -f "$PPL_INPUT" -c 256 -b 64 -ub 64 --chunks 1 --no-warmup -ngl 1 || true
    run_cmd "gpu_ppl_chunks1_ik" \
        ik_ppl -m "$MODEL" -f "$PPL_INPUT" -c 256 -b 64 -ub 64 --chunks 1 --no-warmup -ngl 1 || true

    # Quick sweep sanity (mainline -> ik)
    if [[ -x "$MAIN_BIN/llama-sweep-bench" ]]; then
        run_cmd "gpu_sweep_mainline" \
            main_sweep -m "$MODEL" --cpu-moe -ngl 999 -c "$SWEEP_CTX" -b 1024 -ub 128 -n "$SWEEP_N" -ctk f16 -ctv f16 || true
    else
        printf "%s\tSKIP\t0\tNA\tNA\tNA\tNA\tNA\tNA\n" "gpu_sweep_mainline" >> "$STATUS_FILE"
        log "SKIP: gpu_sweep_mainline (missing $MAIN_BIN/llama-sweep-bench)"
    fi
    if [[ -x "$IK_BIN/llama-sweep-bench" ]]; then
        run_cmd "gpu_sweep_ik" \
            ik_sweep -m "$MODEL" --cpu-moe -ngl 999 -c "$SWEEP_CTX" -b 1024 -ub 128 -n "$SWEEP_N" -ctk f16 -ctv f16 || true
    else
        printf "%s\tSKIP\t0\tNA\tNA\tNA\tNA\tNA\tNA\n" "gpu_sweep_ik" >> "$STATUS_FILE"
        log "SKIP: gpu_sweep_ik (missing $IK_BIN/llama-sweep-bench)"
    fi
fi

# Aggregate summary
cpu_c1_main="$(extract_ppl /out/cpu_ppl_chunks1_mainline.out /out/cpu_ppl_chunks1_mainline.err)"
cpu_c1_ik="$(extract_ppl /out/cpu_ppl_chunks1_ik.out /out/cpu_ppl_chunks1_ik.err)"
cpu_c2_main="$(extract_ppl /out/cpu_ppl_chunks2_mainline.out /out/cpu_ppl_chunks2_mainline.err)"
cpu_c2_ik="$(extract_ppl /out/cpu_ppl_chunks2_ik.out /out/cpu_ppl_chunks2_ik.err)"

cpu_c1_delta="NA"
cpu_c2_delta="NA"
if [[ "$cpu_c1_main" != "NA" && "$cpu_c1_ik" != "NA" ]]; then
    cpu_c1_delta="$(abs_delta "$cpu_c1_main" "$cpu_c1_ik")"
fi
if [[ "$cpu_c2_main" != "NA" && "$cpu_c2_ik" != "NA" ]]; then
    cpu_c2_delta="$(abs_delta "$cpu_c2_main" "$cpu_c2_ik")"
fi

main_has_fib="$(has_token /out/cpu_gen_mainline.out 'fibonacci|fibs|fib')"
ik_has_fib="$(has_token /out/cpu_gen_ik.out 'fibonacci|fibs|fib')"
main_has_complexity="$(has_token /out/cpu_gen_mainline.out 'complexity|O\(')"
ik_has_complexity="$(has_token /out/cpu_gen_ik.out 'complexity|O\(')"

{
    echo "# Qwen3Next Eval Summary"
    echo
    echo "Mode: $( [[ "$WITH_GPU" == "1" ]] && echo "CPU+GPU" || echo "CPU-only" )"
    echo "- Sweep config: c=\`$SWEEP_CTX\`, n=\`$SWEEP_N\`"
    echo
    echo "## CPU Perplexity"
    echo "- chunks=1 mainline: \`$cpu_c1_main\`"
    echo "- chunks=1 ik: \`$cpu_c1_ik\`"
    echo "- chunks=1 |delta|: \`$cpu_c1_delta\`"
    echo "- chunks=2 mainline: \`$cpu_c2_main\`"
    echo "- chunks=2 ik: \`$cpu_c2_ik\`"
    echo "- chunks=2 |delta|: \`$cpu_c2_delta\`"
    echo
    echo "## CPU Short Generation Smoke"
    echo "- mainline has Fibonacci token(s): \`$main_has_fib\`"
    echo "- ik has Fibonacci token(s): \`$ik_has_fib\`"
    echo "- mainline has complexity token(s): \`$main_has_complexity\`"
    echo "- ik has complexity token(s): \`$ik_has_complexity\`"
    echo
    echo "## Command Status + Memory"
    echo '```'
    cat "$STATUS_FILE"
    echo '```'
    echo
    echo "## First Non-empty Lines (Generation)"
    echo "### mainline"
    awk 'NF { print; c++; if (c == 20) exit }' /out/cpu_gen_mainline.out
    echo
    echo "### ik"
    awk 'NF { print; c++; if (c == 20) exit }' /out/cpu_gen_ik.out
} > /out/SUMMARY.md

log "Summary written to /out/SUMMARY.md"
BASH

chmod +x "${out_dir}/run_inside.sh"

docker_cmd=(
    docker run --rm
    -e WITH_GPU="${WITH_GPU}"
    -e GPU_DEVICE="${GPU_DEVICE}"
    -e SWEEP_CTX="${SWEEP_CTX}"
    -e SWEEP_N="${SWEEP_N}"
    -e MAIN_BUILD_DIR="${MAIN_BUILD_DIR}"
    -e IK_BUILD_DIR="${IK_BUILD_DIR}"
    -v "${MAIN_REPO}:/mainline"
    -v "${IK_REPO}:/ik"
    -v "${MODEL_HOST}:/model.gguf:ro"
    -v "${out_dir}:/out"
)

if [[ "$WITH_GPU" -eq 1 ]]; then
    docker_cmd+=(--gpus all)
fi

docker_cmd+=("${IMAGE}" /bin/bash /out/run_inside.sh)

echo "Running eval in container: ${IMAGE}"
echo "Output directory: ${out_dir}"
"${docker_cmd[@]}"

echo
echo "Done. Summary:"
echo "  ${out_dir}/SUMMARY.md"
echo "Raw logs:"
echo "  ${out_dir}/*.out"
echo "  ${out_dir}/*.err"

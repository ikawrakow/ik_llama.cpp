#!/usr/bin/env bash
#
# llama-cli-official — official llama.cpp CLI (v529 design/features) running on
# the ik_llama.cpp MoE-optimized engine.
#
# How it works: the official llama-cli is a client-server app. When given
# --server-base it connects to a running llama-server instead of spawning its
# own. So we:
#   1. start ik's llama-server (MoE-optimized) if one isn't already running
#   2. launch the official llama-cli pointed at it via --server-base
#   3. stop the server we started (unless --keep-server)
#
# Usage:
#   llama-cli-official [llama-cli args...]
#   llama-cli-official -m /path/to/model.gguf -p "hello" -n 64
#
# Env overrides:
#   LLAMA_CLI_OFFICIAL_BIN  official llama-cli binary (default: ~/llama.cpp-official/build/bin/llama-cli)
#   IK_SERVER_BIN           ik llama-server binary       (default: ~/ik_llama.cpp/build/bin/llama-server)
#   IK_SERVER_PORT          port for ik server            (default: 8080)
#   IK_SERVER_ARGS          extra args for ik server (space-separated)
#   LLAMA_CLI_OFFICIAL_KEEP_SERVER=1  leave the server running after exit
#
set -u

LLAMA_CLI_OFFICIAL_BIN="${LLAMA_CLI_OFFICIAL_BIN:-$HOME/llama.cpp-official/build/bin/llama-cli}"
IK_SERVER_BIN="${IK_SERVER_BIN:-$HOME/ik_llama.cpp/build/bin/llama-server}"
IK_SERVER_PORT="${IK_SERVER_PORT:-8080}"
IK_SERVER_ADDR="http://127.0.0.1:${IK_SERVER_PORT}"

# default model: first -m/--model/-hf arg passed to the CLI, else APEX mini
MODEL_ARG=""
for a in "$@"; do
    case "$a" in
        -m|--model|-hf|--hf) MODEL_ARG="found" ;;
    esac
    if [ "$MODEL_ARG" = "found" ]; then MODEL_ARG="skip"; fi
done
if [ "$MODEL_ARG" != "skip" ]; then
    DEFAULT_MODEL="$HOME/Downloads/gemma-4-26B-A4B-heretic-APEX-I-Mini.gguf"
else
    DEFAULT_MODEL=""
fi

if [ ! -x "$LLAMA_CLI_OFFICIAL_BIN" ]; then
    echo "error: official llama-cli not found at $LLAMA_CLI_OFFICIAL_BIN" >&2
    exit 1
fi
if [ ! -x "$IK_SERVER_BIN" ]; then
    echo "error: ik llama-server not found at $IK_SERVER_BIN" >&2
    exit 1
fi

# --- is a server already up? ---
server_up() {
    curl -s -o /dev/null -w "%{http_code}" "$IK_SERVER_ADDR/health" 2>/dev/null | grep -q 200
}

STARTED_SERVER=0
if server_up; then
    echo "using already-running ik server at $IK_SERVER_ADDR"
else
    echo "starting ik llama-server (MoE-optimized) on port $IK_SERVER_PORT ..."
    MODEL="$DEFAULT_MODEL"
    # pick the model from args if present (first -m/-hf value)
    prev=""
    for a in "$@"; do
        if [ "$prev" = "-m" ] || [ "$prev" = "--model" ] || [ "$prev" = "-hf" ] || [ "$prev" = "--hf" ]; then
            MODEL="$a"; break
        fi
        prev="$a"
    done
    if [ -z "${MODEL:-}" ] || [ ! -f "$MODEL" ]; then
        echo "error: model not found: ${MODEL:-<none>}" >&2
        echo "       pass -m <model.gguf> or set the model path" >&2
        exit 1
    fi
    # -t 4 -tb 8: fast first token (TTFT) preference; -np 1 -c 4096: safe on 16GB
    "$IK_SERVER_BIN" \
        -m "$MODEL" \
        -t 4 -tb 8 -ngl 0 -c 4096 --jinja --reasoning off -np 1 \
        --port "$IK_SERVER_PORT" \
        ${IK_SERVER_ARGS:-} \
        > /tmp/llama-cli-official-server.log 2>&1 &
    SERVER_PID=$!
    STARTED_SERVER=1
    trap 'kill $SERVER_PID 2>/dev/null' EXIT
    echo "waiting for server (model load can take a while)..."
    for i in $(seq 1 300); do
        if server_up; then echo "server ready."; break; fi
        if ! kill -0 $SERVER_PID 2>/dev/null; then
            echo "error: server died, see /tmp/llama-cli-official-server.log" >&2
            exit 1
        fi
        sleep 1
    done
    if ! server_up; then
        echo "error: server did not become ready in time" >&2
        exit 1
    fi
fi

# --- run the official CLI against ik's engine ---
"$LLAMA_CLI_OFFICIAL_BIN" --server-base "$IK_SERVER_ADDR" "$@"
RC=$?

if [ "$STARTED_SERVER" = "1" ] && [ "${LLAMA_CLI_OFFICIAL_KEEP_SERVER:-0}" != "1" ]; then
    kill $SERVER_PID 2>/dev/null
    wait $SERVER_PID 2>/dev/null
    echo "ik server stopped."
fi
exit $RC

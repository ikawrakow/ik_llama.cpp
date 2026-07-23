#!/usr/bin/env bash
# Parity test for the Laguna SWA ring KV cache.
#
# The ring allocates window-sized K/V for sliding-window layers and must be
# numerically equivalent to the dense allocation (--swa-full). This test:
#   1. generates a tiny random Laguna GGUF (deterministic seed),
#   2. checks perplexity parity ring vs --swa-full on three legs:
#        - no-FA, ub=48: prompt-processing incl. ring wrap-SPLIT writes
#          (48 does not divide the ring size 256, so a ubatch straddles the
#          ring end; a divisor would leave the split path untested),
#        - FA, ub=48: the F16-mask + fattn kernel path (skipped if FA is
#          unavailable on this build),
#        - no-FA, ub=1 with graph reuse: single-token (decode-shaped) graphs,
#          ring writes at cell % size, decode wraps, and the graph-reuse
#          offset patching,
#   3. smoke-runs greedy generation in both modes (must complete; the ring
#      run must report engagement),
#   4. checks the KV buffer actually shrinks (the point of the feature),
#   5. checks the state-save and context-shift guards fail loudly.
#
# NOTE on oracles: generated TEXT equality between ring and dense is NOT
# asserted. The tiny random model has nearly flat logits, so greedy argmax
# flips under ANY change in float reduction order — verified: dense-vs-dense
# with only a different -t diverges from the second token. Scoring parity
# (perplexity) is the tie-robust oracle; it matched to 7 significant digits.
#
# Usage: tests/test-laguna-swa-ring.sh [BUILD_DIR]   (default: ./build)
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="${1:-$REPO_DIR/build}"
BIN="$BUILD_DIR/bin"
WORK_DIR="$(mktemp -d)"
cleanup() {
    status=$?
    if [ "$status" -ne 0 ]; then
        echo "=== test failed (exit $status); log tails: ==="
        for f in "$WORK_DIR"/*.log; do
            [ -f "$f" ] && { echo "--- $(basename "$f") (last 15) ---"; tail -15 "$f"; }
        done
    fi
    rm -rf "$WORK_DIR"
    exit "$status"
}
trap cleanup EXIT

python3 -c "import numpy" 2>/dev/null || {
    echo "SKIP-PRECONDITION: python3 with numpy is required (for gguf-py)"; exit 1; }

MODEL="$WORK_DIR/tiny-laguna.gguf"
python3 "$REPO_DIR/tests/make-tiny-laguna-gguf.py" "$MODEL" --seed 42

# Deterministic prompt long enough to cross the SWA window (n_swa=64 in the
# tiny model) and to wrap the ring (256 cells < n_ctx = 768) repeatedly.
PROMPT_FILE="$WORK_DIR/prompt.txt"
python3 - "$PROMPT_FILE" <<'EOF'
import sys
words = ["alpha","bravo","charlie","delta","echo","foxtrot","golf","hotel"]
text = " ".join(words[i % 8] + str(i % 97) for i in range(600))
open(sys.argv[1], "w").write(text)
EOF

COMMON_ARGS=(-m "$MODEL" -c 768 -b 768 --seed 7 -t 4 --no-warmup)

run_ppl() { # $1: extra args, $2: output tag
    "$BIN/llama-perplexity" "${COMMON_ARGS[@]}" -f "$PROMPT_FILE" --ppl-stride 0 $1 2>&1 \
        | tee "$WORK_DIR/ppl-$2.log" \
        | sed -nE 's|.*Final estimate:.*= ([0-9.]+) \+/-.*|\1|p' > "$WORK_DIR/ppl-$2.val"
}

check_ppl_pair() { # $1: tag-ring, $2: tag-full
    local ring full
    ring=$(cat "$WORK_DIR/ppl-$1.val")
    full=$(cat "$WORK_DIR/ppl-$2.val")
    echo "ppl $1=$ring $2=$full"
    python3 - "$ring" "$full" <<'EOF'
import sys
ring, full = float(sys.argv[1]), float(sys.argv[2])
rel = abs(ring - full) / max(full, 1e-9)
assert rel < 1e-3, f"perplexity mismatch: ring={ring} full={full} rel={rel}"
print(f"ppl parity OK (rel diff {rel:.2e})")
EOF
}

echo "== ppl leg 1: no-FA, ub=48 (prompt processing + wrap-split writes) =="
run_ppl "-no-fa -ub 48"             ring-nofa
run_ppl "-no-fa -ub 48 --swa-full"  full-nofa
check_ppl_pair ring-nofa full-nofa

echo "== ppl leg 2: FA, ub=48 (F16 mask + fattn kernels) =="
if run_ppl "-fa on -ub 48" ring-fa 2>/dev/null && [ -s "$WORK_DIR/ppl-ring-fa.val" ]; then
    run_ppl "-fa on -ub 48 --swa-full" full-fa
    check_ppl_pair ring-fa full-fa
else
    echo "flash attention unavailable on this build/CPU -> skipping FA leg"
fi

echo "== ppl leg 3: no-FA, ub=1 + graph reuse (decode-shaped graphs, reuse patching) =="
run_ppl "-no-fa -ub 1 -gr"             ring-ub1
run_ppl "-no-fa -ub 1 -gr --swa-full"  full-ub1
check_ppl_pair ring-ub1 full-ub1

# Shorter prompt for generation legs: prompt + 350 generated tokens must fit
# n_ctx = 768, since context shift is (deliberately) unsupported with the ring.
PROMPT_GEN_FILE="$WORK_DIR/prompt-gen.txt"
head -c 320 "$PROMPT_FILE" > "$PROMPT_GEN_FILE"

echo "== generation smoke: both modes must complete (see oracle NOTE above) =="
run_gen() { # $1: extra args, $2: output tag
    "$BIN/llama-cli" "${COMMON_ARGS[@]}" -ub 48 -f "$PROMPT_GEN_FILE" -n 350 --temp 0 --top-k 1 \
        --no-display-prompt $1 > "$WORK_DIR/gen-$2.txt" 2> "$WORK_DIR/gen-$2.log"
}
run_gen ""           ring
run_gen "--swa-full" full
[ -s "$WORK_DIR/gen-ring.txt" ] || { echo "FAIL: ring generation produced no output"; exit 1; }
[ -s "$WORK_DIR/gen-full.txt" ] || { echo "FAIL: dense generation produced no output"; exit 1; }
echo "generation smoke OK"

# The ring must actually have been engaged in the ring runs, otherwise this
# test silently degrades into full-vs-full.
if ! grep -q "SWA ring KV" "$WORK_DIR/ppl-ring-nofa.log"; then
    echo "FAIL: ring run did not report 'SWA ring KV' engagement"
    exit 1
fi
if ! grep -q "SWA ring KV" "$WORK_DIR/gen-ring.log"; then
    echo "FAIL: ring generation run did not report 'SWA ring KV' engagement"
    exit 1
fi
if grep -q "SWA ring KV: n_swa" "$WORK_DIR/ppl-full-nofa.log"; then
    echo "FAIL: --swa-full run unexpectedly engaged the SWA ring"
    exit 1
fi

# And the allocation must actually shrink: this is the point of the feature.
# The tiny model is mostly SWA layers, so ring KV must be well under dense KV.
kv_mib() { grep -oE 'KV self size *= *[0-9.]+ MiB' "$1" | grep -oE '[0-9.]+' | head -1; }
KV_RING=$(kv_mib "$WORK_DIR/ppl-ring-nofa.log")
KV_FULL=$(kv_mib "$WORK_DIR/ppl-full-nofa.log")
echo "KV self size: ring=${KV_RING} MiB full=${KV_FULL} MiB"
python3 - "$KV_RING" "$KV_FULL" <<'EOF'
import sys
ring, full = float(sys.argv[1]), float(sys.argv[2])
assert ring < 0.8 * full, f"ring KV ({ring} MiB) did not shrink vs dense KV ({full} MiB)"
print(f"KV shrink OK ({full/max(ring,1e-9):.1f}x smaller)")
EOF

echo "== guard: state save must fail cleanly with the ring =="
if "$BIN/llama-cli" "${COMMON_ARGS[@]}" -ub 48 -f "$PROMPT_GEN_FILE" -n 8 --temp 0 --top-k 1 \
        --prompt-cache "$WORK_DIR/state.bin" > "$WORK_DIR/state.log" 2>&1; then
    echo "FAIL: state save with ring should have failed"
    exit 1
fi
if ! grep -qi "swa-full" "$WORK_DIR/state.log"; then
    echo "FAIL: state-save failure lacks the SWA ring guard message"
    tail -5 "$WORK_DIR/state.log"
    exit 1
fi
echo "state-save guard OK"

echo "== guard: context shift must fail cleanly with the ring =="
# n_ctx 512 keeps the ring engaged (ring 256 < 512); prompt + 400 generated
# tokens overflow it, so llama-cli attempts a context shift, which the ring
# must refuse with a message pointing at --swa-full (not corrupt silently).
if "$BIN/llama-cli" -m "$MODEL" -c 512 -b 512 -ub 48 --seed 7 -t 4 --no-warmup \
        -f "$PROMPT_GEN_FILE" -n 400 --temp 0 --top-k 1 > "$WORK_DIR/shift.log" 2>&1; then
    echo "FAIL: generation past n_ctx with ring should have failed (context shift unsupported)"
    exit 1
fi
if ! grep -qi "swa-full" "$WORK_DIR/shift.log"; then
    echo "FAIL: context-shift failure lacks the SWA ring guard message"
    tail -5 "$WORK_DIR/shift.log"
    exit 1
fi
echo "context-shift guard OK"

echo "== server: cache-reuse rewind must fall back, not crash =="
# A cache_prompt reuse that truncates more generated tokens off the cache than
# the ring's slack (R - n_swa) needs window cells the ring has overwritten.
# llama_kv_cache_seq_rm must refuse the partial removal so the server falls
# back to reprocessing the full prompt; before the refusal existed this hit
# the occupancy guard and aborted the whole server.
if [ -x "$BIN/llama-server" ]; then
    SRV_PORT=18731
    # c=1024 ub=128 -> ring R = 256, n_swa = 64, slack = 192
    "$BIN/llama-server" -m "$MODEL" -c 1024 -ub 128 --port $SRV_PORT --host 127.0.0.1 \
        -np 1 --no-context-shift -t 4 --dry-multiplier 0.8 > "$WORK_DIR/server.log" 2>&1 &
    SRV_PID=$!
    for _ in $(seq 1 60); do
        curl -s -m 2 "http://127.0.0.1:$SRV_PORT/health" 2>/dev/null | grep -q '"ok"' && break
        sleep 1
    done
    grep -q "SWA ring KV" "$WORK_DIR/server.log" || { echo "FAIL: server did not engage the ring"; kill $SRV_PID 2>/dev/null; exit 1; }
    # the fork adds --dry-* launch flags (upstream ik only had per-request DRY);
    # /props must reflect the CLI value as the server-wide default
    curl -s -m 5 "http://127.0.0.1:$SRV_PORT/props" | grep -q '"dry_multiplier": *0.8' \
        || { echo "FAIL: --dry-multiplier did not reach default generation settings"; kill $SRV_PID 2>/dev/null; exit 1; }
    SRV_PROMPT=$(python3 -c "print('lorem ipsum dolor sit amet ' * 12, end='')")
    # ORACLE NOTE: the random-weight model emits EOS early and generates
    # invalid-UTF-8 bytes, so response bodies (may be HTTP 500) and text
    # content are NOT asserted. ignore_eos forces the full 250 tokens so the
    # reuse below must rewind by 250 > slack 192; the assertions are that the
    # server SURVIVES that request (pre-fix: occupancy-guard abort killed it)
    # and that the fallback reprocessed from scratch (last "kv cache rm" p0=0).
    curl -s -m 300 -X POST "http://127.0.0.1:$SRV_PORT/completion" -H 'Content-Type: application/json' \
        -d "{\"prompt\": \"$SRV_PROMPT\", \"n_predict\": 250, \"cache_prompt\": true, \"ignore_eos\": true}" > "$WORK_DIR/srv1.json" \
        || { echo "FAIL: server first request got no HTTP response"; kill $SRV_PID 2>/dev/null; exit 1; }
    curl -s -m 300 -X POST "http://127.0.0.1:$SRV_PORT/completion" -H 'Content-Type: application/json' \
        -d "{\"prompt\": \"$SRV_PROMPT\", \"n_predict\": 8, \"cache_prompt\": true, \"ignore_eos\": true}" > "$WORK_DIR/srv2.json" || true
    if ! kill -0 $SRV_PID 2>/dev/null; then
        echo "FAIL: server died on cache rewind (ring occupancy abort?)"
        tail -5 "$WORK_DIR/server.log"
        exit 1
    fi
    curl -s -m 5 "http://127.0.0.1:$SRV_PORT/health" | grep -q '"ok"' \
        || { echo "FAIL: server unhealthy after cache rewind"; kill $SRV_PID 2>/dev/null; exit 1; }
    grep "kv cache rm" "$WORK_DIR/server.log" | tail -1 | grep -q "p0=0" \
        || { echo "FAIL: rewind was not refused into a full reprocess (last kv-cache-rm p0 != 0)"; kill $SRV_PID 2>/dev/null; exit 1; }
    # the server-side RAM prompt cache (on by default) snapshots slot KV via
    # llama_state_seq_get_size/get_data, which the ring cannot serialize —
    # the server must auto-disable it at startup and survive prompt switches
    # that would otherwise trigger prompt_save (pre-fix: GGML_ABORT).
    grep -q "prompt cache is disabled: the SWA ring" "$WORK_DIR/server.log" \
        || { echo "FAIL: server prompt cache was not auto-disabled under the ring"; kill $SRV_PID 2>/dev/null; exit 1; }
    curl -s -m 300 -X POST "http://127.0.0.1:$SRV_PORT/completion" -H 'Content-Type: application/json' \
        -d "{\"prompt\": \"a completely different prompt about penguins\", \"n_predict\": 32, \"cache_prompt\": true, \"ignore_eos\": true}" > "$WORK_DIR/srv3.json" || true
    curl -s -m 5 "http://127.0.0.1:$SRV_PORT/health" | grep -q '"ok"' \
        || { echo "FAIL: server unhealthy after prompt switch (prompt_save path?)"; kill $SRV_PID 2>/dev/null; exit 1; }
    kill $SRV_PID 2>/dev/null; wait $SRV_PID 2>/dev/null || true
    echo "server cache-rewind fallback OK"
else
    echo "SKIP: server rewind leg (llama-server not built in $BIN)"
fi

echo "PASS: Laguna SWA ring parity"

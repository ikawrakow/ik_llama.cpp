#!/usr/bin/env bash
# Parity test for the SWA ring KV cache.
#
# The ring allocates window-sized K/V for sliding-window layers and must be
# numerically equivalent to the dense (default) allocation. Ring is opt-in via
# --swa-compress. This test:
#   1. generates a tiny random Laguna GGUF (deterministic seed),
#   2. checks perplexity parity --swa-compress (ring) vs default (dense) on three legs:
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
# Usage: tests/test-swa-ring.sh [BUILD_DIR]   (default: ./build)
# All ring-mode invocations below pass --swa-compress explicitly.
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
# The ring gate is now arch-generic (supports_swa_ring()); this Laguna GGUF
# still exercises it since Laguna is not in the LLAMA4/OPENPANGU exclusion set.

# Deterministic prompt long enough to cross the SWA window (n_swa=64 in the
# tiny model) and to wrap the ring (256 cells < n_ctx = 768) repeatedly.
PROMPT_FILE="$WORK_DIR/prompt.txt"
python3 - "$PROMPT_FILE" <<'EOF'
import sys
words = ["alpha","bravo","charlie","delta","echo","foxtrot","golf","hotel"]
text = " ".join(words[i % 8] + str(i % 97) for i in range(600))
open(sys.argv[1], "w").write(text)
EOF

# NOTE: ring is opt-in via --swa-compress; omitting it (the default) must stay
# byte-identical to the project's pre-ring dense behavior.
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
run_ppl "-no-fa -ub 48 --swa-compress"  ring-nofa
run_ppl "-no-fa -ub 48"                 full-nofa
check_ppl_pair ring-nofa full-nofa

# Pins the generalized ring-size formula (n_swa=64, ub=48 -> GGML_PAD(64+48,256)=256):
# a regression to a hardcoded ubatch assumption (the --fit estimator bug) would not
# show up here since this is the real allocation path, but the ring size logged must
# track the actual ubatch, not a fixed constant.
if ! grep -q "ring size 256 cells" "$WORK_DIR/ppl-ring-nofa.log"; then
    echo "FAIL: ring size did not match GGML_PAD(n_swa + n_ubatch, 256) for n_swa=64, ub=48"
    tail -5 "$WORK_DIR/ppl-ring-nofa.log"
    exit 1
fi

echo "== ppl leg 1b: -mqkv, no-FA, ub=48 (fused-QKV strided-view wrap split, v_trans branch) =="
# -mqkv fuses Q/K/V into one weight, making Kcur/Vcur strided views into the combined
# qkv tensor (row stride spans Q+K+V) -- non-contiguous by ggml's definition. Under
# -no-fa the V cache is transposed (v_trans=true), so this leg pins the transposed
# wrap-split store against a source whose row stride it does not own. K is RoPE'd back
# to contiguous before storage, so the non-transposed strided store is pinned by leg 2b
# below instead.
run_ppl "-no-fa -ub 48 -mqkv --swa-compress"  ring-mqkv
run_ppl "-no-fa -ub 48 -mqkv"                 full-mqkv
check_ppl_pair ring-mqkv full-mqkv

echo "== ppl leg 2: FA, ub=48 (F16 mask + fattn kernels) =="
if run_ppl "-fa on -ub 48 --swa-compress" ring-fa 2>/dev/null && [ -s "$WORK_DIR/ppl-ring-fa.val" ]; then
    run_ppl "-fa on -ub 48" full-fa
    check_ppl_pair ring-fa full-fa

    echo "== ppl leg 2b: FA + -mqkv, ub=48 (fused-QKV strided store, non-transposed V) =="
    # With flash attention, v_trans is false (src/llama.cpp: cache.v_trans =
    # !cache.recurrent && !cparams.flash_attn && ...), so llm_build_kv_store's
    # NON-transposed V branch handles the wrap split. That branch slices -mqkv's
    # strided Vcur view along the token axis using Vcur's own nb[] rather than
    # materializing a contiguous copy, and this is the leg that pins it -- leg 1b
    # (no-FA) reaches the transposed branch instead.
    run_ppl "-fa on -ub 48 -mqkv --swa-compress" ring-fa-mqkv
    run_ppl "-fa on -ub 48 -mqkv"                full-fa-mqkv
    check_ppl_pair ring-fa-mqkv full-fa-mqkv
else
    echo "flash attention unavailable on this build/CPU -> skipping FA leg (and FA+mqkv leg)"
fi

echo "== ppl leg 3: no-FA, ub=1 + graph reuse (decode-shaped graphs, reuse patching) =="
run_ppl "-no-fa -ub 1 -gr --swa-compress"  ring-ub1
run_ppl "-no-fa -ub 1 -gr"                 full-ub1
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
run_gen "--swa-compress" ring
run_gen ""               full
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
    # supports_swa_ring() must stay opt-in-gated: dense run (no --swa-compress)
    # must never engage the ring regardless of arch eligibility.
    echo "FAIL: default (no --swa-compress) run unexpectedly engaged the SWA ring"
    exit 1
fi

# The graph audit must RUN and PASS whenever the ring engages: it is what catches a
# builder that hand-rolls its cache access and reads a ring layer at the wrong width
# (an arch denylist cannot see that, and the occupancy guard cannot either -- ring_occ
# is recorded wherever the write was MEANT to land). A ring run without the audit line
# means the audit was bypassed, which is as bad as the audit failing.
for tag in ppl-ring-nofa gen-ring ring-mqkv; do
    log="$WORK_DIR/${tag}.log"
    [ -f "$log" ] || log="$WORK_DIR/ppl-${tag}.log"
    [ -f "$log" ] || continue
    if ! grep -q "SWA ring graph audit passed" "$log"; then
        echo "FAIL: ring run $tag engaged the ring without passing the graph audit"
        exit 1
    fi
    # A vacuous audit (zero layers examined) would "pass" while checking nothing.
    if grep -q "SWA ring graph audit passed (0 sliding-window layers)" "$log"; then
        echo "FAIL: graph audit examined no sliding-window layers in $tag"
        exit 1
    fi
done
# ... and must not run at all when the ring is off.
if grep -q "SWA ring graph audit" "$WORK_DIR/ppl-full-nofa.log"; then
    echo "FAIL: graph audit ran on a dense (non-ring) context"
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

echo "== --fit estimator unit test: cache_size() SWA formula (no multi-device needed) =="
# get_layer_sizes()/cache_size() are only ever called from the --fit multi-device
# split-mode-graph path (src/llama.cpp), which throws below 2 devices -- unreachable
# on this single-device build. test-cache-size-estimator loads $MODEL directly and
# calls cache_size() itself, pinning the formula independent of that gate.
if [ -x "$BIN/test-cache-size-estimator" ]; then
    "$BIN/test-cache-size-estimator" "$MODEL" 2>&1 | tee "$WORK_DIR/cache-size-estimator.log"
    grep -q "cache_size() estimator OK" "$WORK_DIR/cache-size-estimator.log" \
        || { echo "FAIL: cache_size() estimator unit test did not report OK"; exit 1; }
else
    echo "SKIP: test-cache-size-estimator not built in $BIN"
fi

echo "== state save/restore unit test (ring blob layout, byte-exact round-trip, refusals) =="
# Ring layers serialize only the window (min(size_swa, cell_count) rows, oldest first)
# instead of one row per cell, so this pins: the blob is smaller than the dense one, a
# restore re-serializes byte-for-byte, the continuation logits match, and cross-mode /
# mismatched-geometry blobs are refused with 0 rather than restored wrong.
if [ -x "$BIN/test-swa-ring-state" ]; then
    "$BIN/test-swa-ring-state" "$MODEL" 2>&1 | tee "$WORK_DIR/ring-state.log"
    grep -q "SWA ring state save/restore OK" "$WORK_DIR/ring-state.log" \
        || { echo "FAIL: SWA ring state save/restore unit test did not report OK"; exit 1; }
else
    echo "SKIP: test-swa-ring-state not built in $BIN"
fi

echo "== multi-sequence unit test (per-sequence row stripes, mixed ubatches, cross-stripe state) =="
# The ring stripes rows per sequence (row = seq*ring_w + pos % ring_w), which is what makes
# -np > 1 sound: eviction is measured in each sequence's own positions. This pins that a
# parked sequence survives another running a full window past it, that mixed-sequence
# ubatches write the right stripes (row-major and transposed V), that a blob moves between
# stripes and across --parallel values, and that a whole-context save of two striped
# sequences is refused rather than written wrong.
if [ -x "$BIN/test-swa-ring-multiseq" ]; then
    "$BIN/test-swa-ring-multiseq" "$MODEL" 2>&1 | tee "$WORK_DIR/ring-multiseq.log"
    grep -q "SWA ring multi-sequence OK" "$WORK_DIR/ring-multiseq.log" \
        || { echo "FAIL: SWA ring multi-sequence unit test did not report OK"; exit 1; }
else
    echo "SKIP: test-swa-ring-multiseq not built in $BIN"
fi

echo "== end-to-end: the ring must engage with -np > 1 =="
# The gate used to refuse n_seq_max > 1 outright. A 4-slot server must now get a striped
# ring (4 x the per-sequence window) instead of silently falling back to dense KV.
NP_LOG="$WORK_DIR/np4.log"
"$BIN/llama-cli" -m "$MODEL" -c 2048 -b 512 -ub 48 --swa-compress -np 4 \
    --seed 7 -t 4 --no-warmup -n 8 -p "hello" > "$NP_LOG" 2>&1 || {
    echo "FAIL: -np 4 with --swa-compress did not run"; tail -20 "$NP_LOG"; exit 1; }
grep -q "SWA ring KV: n_swa" "$NP_LOG" \
    || { echo "FAIL: ring did not engage with -np 4"; grep -i swa "$NP_LOG"; exit 1; }
grep -q "per sequence x 4 sequences" "$NP_LOG" \
    || { echo "FAIL: ring was not sized for 4 sequences"; grep -i "SWA ring" "$NP_LOG"; exit 1; }
echo "-np 4 ring engagement OK"

echo "== end-to-end: --prompt-cache must round-trip with the ring =="
# Until the ring-aware serialization landed, every llama_state_* call refused on a ring
# context and this leg asserted the refusal instead. Now the first run must WRITE a
# session file and the second must load it and match the prompt exactly.
rm -f "$WORK_DIR/state.bin"
if ! "$BIN/llama-cli" "${COMMON_ARGS[@]}" -ub 48 --swa-compress -f "$PROMPT_GEN_FILE" -n 8 --temp 0 --top-k 1 \
        --prompt-cache "$WORK_DIR/state.bin" > "$WORK_DIR/state-save.log" 2>&1; then
    echo "FAIL: state save with the ring should exit cleanly"
    tail -20 "$WORK_DIR/state-save.log"
    exit 1
fi
if grep -qi "cannot be serialized with the SWA ring\|not supported with SWA ring" "$WORK_DIR/state-save.log"; then
    echo "FAIL: state save still refuses under the ring"
    grep -i "SWA ring" "$WORK_DIR/state-save.log" | tail -5
    exit 1
fi
[ -s "$WORK_DIR/state.bin" ] || { echo "FAIL: --prompt-cache wrote no session file under the ring"; exit 1; }

if ! "$BIN/llama-cli" "${COMMON_ARGS[@]}" -ub 48 --swa-compress -f "$PROMPT_GEN_FILE" -n 8 --temp 0 --top-k 1 \
        --prompt-cache "$WORK_DIR/state.bin" > "$WORK_DIR/state-load.log" 2>&1; then
    echo "FAIL: reloading the session file under the ring should exit cleanly"
    tail -20 "$WORK_DIR/state-load.log"
    exit 1
fi
if ! grep -q "exact match for prompt\|using full prompt from session file" "$WORK_DIR/state-load.log"; then
    echo "FAIL: reloaded session did not match the prompt (restore silently produced nothing usable)"
    grep -i "session" "$WORK_DIR/state-load.log" | tail -8
    exit 1
fi
if grep -qi "a cell inside the attention window was overwritten" "$WORK_DIR/state-load.log"; then
    echo "FAIL: restored ring occupancy disagrees with the restored rows (mask guard fired)"
    exit 1
fi
# "exact match" alone only proves the token list matched. The restored KV was actually KEPT
# only if the session trim succeeded -- if it had been refused, main.cpp clears the cache and
# reprocesses, which would make this leg pass while reusing nothing.
if grep -q "cannot trim the session" "$WORK_DIR/state-load.log"; then
    echo "FAIL: exact-match reload fell back to reprocessing, so no restored KV was reused"
    grep -i "session" "$WORK_DIR/state-load.log" | tail -8
    exit 1
fi

echo "== end-to-end: a partially matching session must not leave the ring incoherent =="
# main.cpp trims the non-matching tail with llama_kv_cache_seq_rm(p0 = n_matching), which
# the ring refuses when the cells just behind p0 are no longer resident. That refusal must
# turn into a full reprocess, never a silent continue on stale cells (which the ring's own
# occupancy guard would then turn into a process abort).
cat "$PROMPT_GEN_FILE" > "$WORK_DIR/prompt-ext.txt"
printf ' and then something entirely different about penguins in the desert\n' >> "$WORK_DIR/prompt-ext.txt"
if ! "$BIN/llama-cli" "${COMMON_ARGS[@]}" -ub 48 --swa-compress -f "$WORK_DIR/prompt-ext.txt" -n 8 --temp 0 --top-k 1 \
        --prompt-cache "$WORK_DIR/state.bin" > "$WORK_DIR/state-partial.log" 2>&1; then
    echo "FAIL: partially matching session reload should exit cleanly"
    tail -20 "$WORK_DIR/state-partial.log"
    exit 1
fi
if grep -qi "a cell inside the attention window was overwritten" "$WORK_DIR/state-partial.log"; then
    echo "FAIL: partial session reuse continued on stale cells under the ring"
    exit 1
fi
echo "state save/restore OK"

echo "== guard: context shift must fail cleanly with the ring (error-return, not a crash) =="
# n_ctx 512 keeps the ring engaged (ring 256 < 512); prompt + 400 generated
# tokens overflow it, so llama-cli attempts a context shift. The ring must
# refuse via llama_kv_cache_update_internal's has_shift-gated check (a clean
# error return, exit 1) -- NOT via a process abort. A clean-exit-only check
# (nonzero status) cannot tell these apart: a SIGABRT core dump is also
# nonzero. This regression-tests exactly that: an earlier attempt to guard
# llama_kv_cache_seq_add() at the entrypoint (refusing the mutation instead of
# letting it set has_shift for update_internal to detect) silently skipped the
# shift, generation continued on stale positions, and the run crashed in the
# ring's own out-of-window overwrite guard instead of stopping cleanly.
set +e
"$BIN/llama-cli" -m "$MODEL" -c 512 -b 512 -ub 48 --swa-compress --seed 7 -t 4 --no-warmup \
        -f "$PROMPT_GEN_FILE" -n 400 --temp 0 --top-k 1 > "$WORK_DIR/shift.log" 2>&1
SHIFT_RC=$?
set -e
if [ "$SHIFT_RC" -eq 0 ]; then
    echo "FAIL: generation past n_ctx with ring should have failed (context shift unsupported)"
    exit 1
fi
if [ "$SHIFT_RC" -gt 128 ]; then
    echo "FAIL: context-shift refusal crashed the process (signal $((SHIFT_RC - 128))) instead of a clean error return"
    tail -20 "$WORK_DIR/shift.log"
    exit 1
fi
if ! grep -qi "context shift is not supported with SWA ring" "$WORK_DIR/shift.log"; then
    echo "FAIL: context-shift failure lacks the specific SWA ring guard message"
    tail -5 "$WORK_DIR/shift.log"
    exit 1
fi
echo "context-shift guard OK"

echo "== guard: --defrag-thold is ignored, the ring stays engaged =="
# Defrag moves cells, which the ring cannot follow. Rather than silently swapping the
# layout for a dense one (which used to force the --fit estimator to know about a
# context-time-only setting), the ring wins and defrag is disabled with a warning: the
# KV size a caller budgeted for is the KV size it gets, whatever --defrag-thold says.
"$BIN/llama-cli" -m "$MODEL" -c 768 -b 768 -ub 48 --swa-compress --defrag-thold 0.5 \
        --seed 7 -t 4 --no-warmup -p "hello" -n 4 --temp 0 --top-k 1 \
        > "$WORK_DIR/defrag.log" 2>&1
if ! grep -qi "disabling KV defrag" "$WORK_DIR/defrag.log"; then
    echo "FAIL: --swa-compress with --defrag-thold did not report that defrag was disabled"
    tail -10 "$WORK_DIR/defrag.log"
    exit 1
fi
if ! grep -qi "SWA ring KV: n_swa" "$WORK_DIR/defrag.log"; then
    echo "FAIL: the ring did not engage; --defrag-thold must not change the KV layout"
    tail -10 "$WORK_DIR/defrag.log"
    exit 1
fi
echo "defrag guard OK"

echo "== server: cache-reuse rewind must fall back, not crash =="
# A cache_prompt reuse that truncates more generated tokens off the cache than
# the ring's slack (R - n_swa) needs window cells the ring has overwritten.
# llama_kv_cache_seq_rm must refuse the partial removal so the server falls
# back to reprocessing the full prompt; before the refusal existed this hit
# the occupancy guard and aborted the whole server.
if [ -x "$BIN/llama-server" ]; then
    SRV_PORT=18731
    # c=1024 ub=128 -> ring R = 256, n_swa = 64, slack = 192
    mkdir -p "$WORK_DIR/slots"
    "$BIN/llama-server" -m "$MODEL" -c 1024 -ub 128 --swa-compress --port $SRV_PORT --host 127.0.0.1 \
        -np 1 --no-context-shift -t 4 --dry-multiplier 0.8 --slot-save-path "$WORK_DIR/slots" \
        > "$WORK_DIR/server.log" 2>&1 &
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
    # llama_state_seq_get_size/get_data. The ring used to be unable to serialize that, so
    # the server auto-disabled the cache at startup; now that ring state round-trips it
    # must stay ENABLED and survive prompt switches that trigger prompt_save.
    if grep -q "prompt cache is disabled: the SWA ring" "$WORK_DIR/server.log"; then
        echo "FAIL: server still auto-disables the prompt cache under the ring"
        kill $SRV_PID 2>/dev/null; exit 1
    fi
    grep -q "prompt cache is enabled" "$WORK_DIR/server.log" \
        || { echo "FAIL: server prompt cache not enabled under the ring"; kill $SRV_PID 2>/dev/null; exit 1; }
    curl -s -m 300 -X POST "http://127.0.0.1:$SRV_PORT/completion" -H 'Content-Type: application/json' \
        -d "{\"prompt\": \"a completely different prompt about penguins\", \"n_predict\": 32, \"cache_prompt\": true, \"ignore_eos\": true}" > "$WORK_DIR/srv3.json" || true
    curl -s -m 5 "http://127.0.0.1:$SRV_PORT/health" | grep -q '"ok"' \
        || { echo "FAIL: server unhealthy after prompt switch (prompt_save path?)"; kill $SRV_PID 2>/dev/null; exit 1; }
    # A PARTIAL_ONLY (metadata-only) checkpoint restore cannot be honored by a ring: the rows
    # behind the rewind point have been overwritten, and find_slot would re-record the
    # restored cells as their slots' owners, making the occupancy guard confirm a lie. The
    # refusal must be graceful -- the server falls back to a reset, never generates on it.
    if grep -q "failed to restore context checkpoint" "$WORK_DIR/server.log"; then
        grep -qi "PARTIAL_ONLY) restore is not possible with the SWA ring" "$WORK_DIR/server.log" \
            || { echo "FAIL: checkpoint restore failed for a reason other than the ring refusal"; kill $SRV_PID 2>/dev/null; exit 1; }
    fi
    grep -qi "a cell inside the attention window was overwritten" "$WORK_DIR/server.log" \
        && { echo "FAIL: a metadata-only checkpoint restore was honored under the ring"; kill $SRV_PID 2>/dev/null; exit 1; }

    # slot checkpointing: /slots/N?action=save|restore is the "SWA checkpoint" the ring
    # could not provide before (llama_state_seq_save_file returned 0). Both directions must
    # report a nonzero token count and leave the server healthy.
    curl -s -m 30 -X POST "http://127.0.0.1:$SRV_PORT/slots/0?action=save" -H 'Content-Type: application/json' \
        -d '{"filename": "ring-slot0.bin"}' > "$WORK_DIR/slot-save.json" || true
    grep -q '"n_saved": *[1-9]' "$WORK_DIR/slot-save.json" \
        || { echo "FAIL: slot save under the ring stored no tokens"; cat "$WORK_DIR/slot-save.json"; kill $SRV_PID 2>/dev/null; exit 1; }
    [ -s "$WORK_DIR/slots/ring-slot0.bin" ] \
        || { echo "FAIL: slot save wrote no file under the ring"; kill $SRV_PID 2>/dev/null; exit 1; }
    curl -s -m 30 -X POST "http://127.0.0.1:$SRV_PORT/slots/0?action=restore" -H 'Content-Type: application/json' \
        -d '{"filename": "ring-slot0.bin"}' > "$WORK_DIR/slot-restore.json" || true
    grep -q '"n_restored": *[1-9]' "$WORK_DIR/slot-restore.json" \
        || { echo "FAIL: slot restore under the ring loaded no tokens"; cat "$WORK_DIR/slot-restore.json"; kill $SRV_PID 2>/dev/null; exit 1; }
    kill -0 $SRV_PID 2>/dev/null || { echo "FAIL: server died during slot save/restore"; tail -20 "$WORK_DIR/server.log"; exit 1; }
    # a restored slot must still generate -- this is where a wrongly placed ring row shows
    # up as the occupancy-guard abort rather than as bad text
    curl -s -m 300 -X POST "http://127.0.0.1:$SRV_PORT/completion" -H 'Content-Type: application/json' \
        -d "{\"prompt\": \"$SRV_PROMPT\", \"n_predict\": 16, \"cache_prompt\": true, \"ignore_eos\": true}" > "$WORK_DIR/srv4.json" || true
    curl -s -m 5 "http://127.0.0.1:$SRV_PORT/health" | grep -q '"ok"' \
        || { echo "FAIL: server unhealthy after generating from a restored slot"; tail -20 "$WORK_DIR/server.log"; kill $SRV_PID 2>/dev/null; exit 1; }
    # NOTE: match the abort's own wording. The guard prints "cell %d (seq %u, pos %d) inside the
    # attention window [%d, %d] was overwritten", so a pattern that spells out a sentence around
    # "attention window" can never match and the check is vacuous.
    grep -aq "was overwritten" "$WORK_DIR/server.log" \
        && { echo "FAIL: restored slot left the ring occupancy inconsistent"; kill $SRV_PID 2>/dev/null; exit 1; }
    # A server system prompt is fanned out to every slot with llama_kv_cache_seq_cp, which the
    # ring implements as a stripe clone, so it must be ACCEPTED -- and the slot must keep
    # generating afterwards. That second part is the real assertion: a generated token's position
    # is system_tokens.size() + the slot-relative one, and dropping that offset placed every
    # generated token on the ring rows the system prompt itself occupies. Here the occupancy
    # guard turns that into an abort; on the dense path it is silently wrong attention instead,
    # which no oracle available to this suite can see.
    curl -s -m 300 -X POST "http://127.0.0.1:$SRV_PORT/completion" -H 'Content-Type: application/json' \
        -d '{"prompt": "hello", "system_prompt": "You are a helpful assistant.", "n_predict": 8, "ignore_eos": true}' \
        > "$WORK_DIR/srv-sysprompt.json" || true
    kill -0 $SRV_PID 2>/dev/null \
        || { echo "FAIL: server died generating with a system prompt under the ring"; tail -30 "$WORK_DIR/server.log"; exit 1; }
    grep -aq "was overwritten" "$WORK_DIR/server.log" \
        && { echo "FAIL: generated tokens landed on the system prompt's ring rows"; tail -30 "$WORK_DIR/server.log"; kill $SRV_PID 2>/dev/null; exit 1; }
    # NOTE: do NOT assert on the response body -- format_error_response emits
    # "type": "invalid_request_error", and this random-weight model answers ordinary requests
    # with an error envelope too, so such a check passes whatever the server does. The server log
    # is the discriminating signal: release_slots reports n_system_tokens, nonzero only if the
    # prompt was encoded AND survived the fan-out verification (system_prompt_disable clears it).
    grep -aq "n_system_tokens=[1-9]" "$WORK_DIR/server.log" \
        || { echo "FAIL: ring server served slots with no system prompt resident"; tail -30 "$WORK_DIR/server.log"; kill $SRV_PID 2>/dev/null; exit 1; }
    grep -aq 'swa-compress' "$WORK_DIR/srv-sysprompt.json" \
        && { echo "FAIL: a system prompt is still refused under --swa-compress"; cat "$WORK_DIR/srv-sysprompt.json"; kill $SRV_PID 2>/dev/null; exit 1; }
    curl -s -m 5 "http://127.0.0.1:$SRV_PORT/health" | grep -q '"ok"' \
        || { echo "FAIL: server unhealthy after a system prompt under the ring"; tail -20 "$WORK_DIR/server.log"; kill $SRV_PID 2>/dev/null; exit 1; }
    # a further request must still be served: system_prompt_update() opens with an unconditional
    # kv_cache_clear() and re-runs on every update_slots() tick while system_need_update stays
    # set, so a prompt left half-applied would wedge the server.
    curl -s -m 300 -X POST "http://127.0.0.1:$SRV_PORT/completion" -H 'Content-Type: application/json' \
        -d '{"prompt": "hello again", "n_predict": 4, "ignore_eos": true}' > "$WORK_DIR/srv-postsys.json" \
        || { echo "FAIL: server stopped serving after a system prompt"; tail -20 "$WORK_DIR/server.log"; kill $SRV_PID 2>/dev/null; exit 1; }
    kill -0 $SRV_PID 2>/dev/null \
        || { echo "FAIL: server died after a system prompt"; tail -20 "$WORK_DIR/server.log"; exit 1; }

    kill $SRV_PID 2>/dev/null; wait $SRV_PID 2>/dev/null || true
    echo "server cache-rewind fallback + slot checkpointing + system prompt under the ring OK"

    # DENSE control for the fan-out check: the ring leg above can only prove a REFUSAL is
    # reported. system_prompt_update() now verifies the fan-out landed by counting the seq_ids
    # it added (n_cells_seq0 * (n_parallel + 1)); if that arithmetic is wrong it would reject
    # perfectly good system prompts on the dense path, which is every ordinary user. This leg
    # is the one that catches that -- multi-slot, no --swa-compress, prompt must be ACCEPTED.
    # It also covers the fan-out loop bound: slot ids are 0..n_parallel-1 and n_seq_max is
    # n_parallel, so copying to 1..n_parallel tagged every cell with an out-of-range seq_id.
    SRV_PORT2=18732
    "$BIN/llama-server" -m "$MODEL" -c 2048 -ub 128 --port $SRV_PORT2 --host 127.0.0.1 \
        -np 2 --no-context-shift -t 4 > "$WORK_DIR/server-dense-sys.log" 2>&1 &
    SRV2_PID=$!
    for _ in $(seq 1 60); do
        curl -s -m 2 "http://127.0.0.1:$SRV_PORT2/health" 2>/dev/null | grep -q '"ok"' && break
        sleep 1
    done
    curl -s -m 5 "http://127.0.0.1:$SRV_PORT2/health" | grep -q '"ok"' \
        || { echo "FAIL: dense multi-slot server did not come up"; tail -20 "$WORK_DIR/server-dense-sys.log"; kill $SRV2_PID 2>/dev/null; exit 1; }
    curl -s -m 300 -X POST "http://127.0.0.1:$SRV_PORT2/completion" -H 'Content-Type: application/json' \
        -d '{"prompt": "hello", "system_prompt": "You are a helpful assistant.", "n_predict": 4, "ignore_eos": true}' \
        > "$WORK_DIR/srv-dense-sys.json" || true
    # ORACLE NOTE: per the note on the ring leg above, this random-weight model returns HTTP
    # 500 bodies, so the RESPONSE cannot say whether the prompt was accepted -- a body check
    # here is red for every implementation, correct or not. The server log is the discriminating
    # signal: release_slots reports n_system_tokens, which is nonzero only if the system prompt
    # was encoded AND survived the fan-out verification (system_prompt_disable clears it).
    grep -aq "n_system_tokens=[1-9]" "$WORK_DIR/server-dense-sys.log" \
        || { echo "FAIL: dense server served slots with no system prompt resident"; tail -20 "$WORK_DIR/server-dense-sys.log"; kill $SRV2_PID 2>/dev/null; exit 1; }
    grep -aq "system prompt fan-out did not reach every slot" "$WORK_DIR/server-dense-sys.log" \
        && { echo "FAIL: fan-out verification miscounted on the dense path"; tail -20 "$WORK_DIR/server-dense-sys.log"; kill $SRV2_PID 2>/dev/null; exit 1; }
    grep -aq "system prompt disabled" "$WORK_DIR/server-dense-sys.log" \
        && { echo "FAIL: dense server disabled a valid system prompt"; tail -20 "$WORK_DIR/server-dense-sys.log"; kill $SRV2_PID 2>/dev/null; exit 1; }
    # both slots must still serve with the system prompt resident
    curl -s -m 300 -X POST "http://127.0.0.1:$SRV_PORT2/completion" -H 'Content-Type: application/json' \
        -d '{"prompt": "second slot please", "n_predict": 4, "ignore_eos": true}' > "$WORK_DIR/srv-dense-sys2.json" || true
    curl -s -m 5 "http://127.0.0.1:$SRV_PORT2/health" | grep -q '"ok"' \
        || { echo "FAIL: dense server unhealthy after a system-prompt fan-out"; tail -20 "$WORK_DIR/server-dense-sys.log"; kill $SRV2_PID 2>/dev/null; exit 1; }
    kill $SRV2_PID 2>/dev/null; wait $SRV2_PID 2>/dev/null || true
    echo "dense multi-slot system-prompt fan-out OK"

    # RING multi-slot: the leg above runs the fan-out on the dense path, where seq_cp only tags
    # cells. Under the ring it CLONES the source stripe into each destination's own rows, and a
    # clone that lands on the wrong rows shows up as the occupancy guard aborting once a slot
    # generates -- so both slots must serve after the fan-out, not just accept it.
    SRV_PORT3=18733
    "$BIN/llama-server" -m "$MODEL" -c 1024 -ub 128 --swa-compress --port $SRV_PORT3 --host 127.0.0.1 \
        -np 2 --no-context-shift -t 4 > "$WORK_DIR/server-ring-sys.log" 2>&1 &
    SRV3_PID=$!
    for _ in $(seq 1 60); do
        curl -s -m 2 "http://127.0.0.1:$SRV_PORT3/health" 2>/dev/null | grep -q '"ok"' && break
        sleep 1
    done
    curl -s -m 5 "http://127.0.0.1:$SRV_PORT3/health" | grep -q '"ok"' \
        || { echo "FAIL: ring multi-slot server did not come up"; tail -20 "$WORK_DIR/server-ring-sys.log"; kill $SRV3_PID 2>/dev/null; exit 1; }
    curl -s -m 300 -X POST "http://127.0.0.1:$SRV_PORT3/completion" -H 'Content-Type: application/json' \
        -d '{"prompt": "hello", "system_prompt": "You are a helpful assistant.", "n_predict": 8, "ignore_eos": true}' \
        > "$WORK_DIR/srv-ring-sys.json" || true
    curl -s -m 300 -X POST "http://127.0.0.1:$SRV_PORT3/completion" -H 'Content-Type: application/json' \
        -d '{"prompt": "second slot please", "n_predict": 8, "ignore_eos": true}' > "$WORK_DIR/srv-ring-sys2.json" || true
    kill -0 $SRV3_PID 2>/dev/null \
        || { echo "FAIL: ring server died serving slots after a system-prompt fan-out"; tail -30 "$WORK_DIR/server-ring-sys.log"; exit 1; }
    grep -aq "was overwritten" "$WORK_DIR/server-ring-sys.log" \
        && { echo "FAIL: system-prompt stripe clone left the ring occupancy inconsistent"; tail -30 "$WORK_DIR/server-ring-sys.log"; kill $SRV3_PID 2>/dev/null; exit 1; }
    grep -aq "system prompt fan-out did not reach every slot" "$WORK_DIR/server-ring-sys.log" \
        && { echo "FAIL: the ring stripe clone did not reach every slot"; tail -30 "$WORK_DIR/server-ring-sys.log"; kill $SRV3_PID 2>/dev/null; exit 1; }
    grep -aq "n_system_tokens=[1-9]" "$WORK_DIR/server-ring-sys.log" \
        || { echo "FAIL: ring multi-slot server served slots with no system prompt resident"; tail -30 "$WORK_DIR/server-ring-sys.log"; kill $SRV3_PID 2>/dev/null; exit 1; }
    kill $SRV3_PID 2>/dev/null; wait $SRV3_PID 2>/dev/null || true
    echo "ring multi-slot system-prompt stripe clone OK"
else
    echo "SKIP: server rewind leg (llama-server not built in $BIN)"
fi

echo "== arch-generalization leg: real non-Laguna SWA arch (GEMMA2) must engage + shrink the ring =="
# supports_swa_ring() is arch-generic, but the ring's per-layer routing keys off
# hparams.swa_layers, not n_swa alone. GEMMA2 alternates SWA on even layers via a
# hardcoded graph pattern (build_gemma2.cpp: il % 2 == 0) and, until fixed, never
# populated swa_layers to match -- so the ring silently never shrank any GEMMA2
# layer despite passing the arch gate. This leg pins that swa_layers now tracks
# the real per-layer pattern for a non-Laguna arch.
GEMMA2_MODEL="$WORK_DIR/tiny-gemma2.gguf"
python3 "$REPO_DIR/tests/make-tiny-gemma2-gguf.py" "$GEMMA2_MODEL" --seed 42
COMMON_ARGS=(-m "$GEMMA2_MODEL" -c 768 -b 768 --seed 7 -t 4 --no-warmup)
run_ppl "-no-fa -ub 48 --swa-compress" g2-ring
run_ppl "-no-fa -ub 48"                g2-full
check_ppl_pair g2-ring g2-full
if ! grep -q "SWA ring KV" "$WORK_DIR/ppl-g2-ring.log"; then
    echo "FAIL: gemma2 (non-Laguna arch) did not engage the SWA ring under --swa-compress"
    exit 1
fi
KV_RING_G2=$(kv_mib "$WORK_DIR/ppl-g2-ring.log")
KV_FULL_G2=$(kv_mib "$WORK_DIR/ppl-g2-full.log")
echo "gemma2 KV self size: ring=${KV_RING_G2} MiB full=${KV_FULL_G2} MiB"
python3 - "$KV_RING_G2" "$KV_FULL_G2" <<'EOF'
import sys
ring, full = float(sys.argv[1]), float(sys.argv[2])
assert ring < 0.9 * full, f"gemma2 ring KV ({ring} MiB) did not shrink vs dense KV ({full} MiB)"
print(f"gemma2 KV shrink OK ({full/max(ring,1e-9):.2f}x smaller)")
EOF

echo "== guard: no dual-codified SWA periodicity constants (GEMMA3/COHERE2/OPENAI_MOE) =="
# These three archs used to hardcode their own copy of the sliding-window
# period as a graph-builder-local constant, duplicating hparams.n_swa_pattern
# set in llama-hparams.cpp -- the two could silently drift apart. The fix made
# each graph builder read hparams.n_swa_pattern instead of re-declaring the
# literal. This leg pins that the literal re-declaration doesn't come back.
for f in build_gemma3.cpp build_cohere2.cpp build_openai.cpp; do
    path="$REPO_DIR/src/graphs/$f"
    if grep -qE 'sliding_window_pattern\s*=\s*[0-9]+\s*;' "$path"; then
        echo "FAIL: $f re-declares a hardcoded sliding_window_pattern literal instead of reading hparams.n_swa_pattern"
        exit 1
    fi
    if ! grep -q 'sliding_window_pattern = hparams.n_swa_pattern' "$path"; then
        echo "FAIL: $f does not read sliding_window_pattern from hparams.n_swa_pattern"
        exit 1
    fi
done
echo "dual-codification guard OK (gemma3/cohere2/openai_moe all read hparams.n_swa_pattern)"

echo "== end-to-end: -mqkv with -np 2 (strided fused-QKV source, multi-sequence stores) =="
# Two slots interleaved in one ubatch produce several ring_parts per store, and -mqkv
# makes the source a strided view into the fused QKV tensor. Together that is the
# multi-part strided store path; the perplexity legs above only reach it with a single
# sequence, where the parts come from a wrap split alone.
MQKV_NP_LOG="$WORK_DIR/cli-mqkv-np2.log"
"$BIN/llama-cli" -m "$MODEL" -c 2048 -b 512 -ub 48 -mqkv --swa-compress -np 2 \
    --seed 7 -t 4 --no-warmup -n 24 -p "alpha bravo charlie delta" > "$MQKV_NP_LOG" 2>&1 || {
    echo "FAIL: -mqkv with -np 2 and --swa-compress did not run"; tail -20 "$MQKV_NP_LOG"; exit 1; }
grep -q "SWA ring KV: n_swa" "$MQKV_NP_LOG" || {
    echo "FAIL: the ring did not engage for -mqkv -np 2"; grep -i swa "$MQKV_NP_LOG"; exit 1; }
echo "-mqkv -np 2 multi-part strided store OK"

echo "== guard: ring K/V stores share one helper and slice strided sources in place =="
# A ring write is a scatter: one ggml_cpy per destination row run. Every source part
# is a view built from the source tensor's OWN strides, so a fused-QKV (-mqkv) Kcur
# -- a strided view into the combined QKV tensor -- is sliced without materializing a
# contiguous copy first. This leg pins both properties: the four store branches route
# through the single helper, and none of them reintroduces a ggml_cont on the source.
BC="$REPO_DIR/src/llama-build-context.cpp"
N_RING_STORE=$(grep -c 'llm_build_ring_store(' "$BC" || true)
# 1 definition + 6 call sites (K, V-flat, V-transposed) x (llm_build_kv_store,
# build_std_attention); a store branch that stops calling it is a silent duplicate.
if [ "$N_RING_STORE" -lt 7 ]; then
    echo "FAIL: expected the definition + 6 ring store call sites of llm_build_ring_store(), found $N_RING_STORE"
    exit 1
fi
# note: match 'ggml_cont(' with the paren -- a bare 'ggml_cont' also hits the
# 'ggml_context * ctx' parameter, which no correct implementation can drop
if awk '/^static void llm_build_ring_store\(/,/^}/' "$BC" | grep -q 'ggml_cont('; then
    echo "FAIL: llm_build_ring_store() materializes its source with ggml_cont instead of slicing it with the source's own strides"
    exit 1
fi
for fn in llm_build_kv_store build_std_attention; do
    if awk -v fn="$fn" '$0 ~ ("^(void|ggml_tensor \\* )?llm_build_context::" fn "\\(") ,/^}/' "$BC" \
            | grep -q 'ggml_cont(ctx0*, *[KkVv]'; then
        echo "FAIL: $fn still calls ggml_cont on a K/V source; ring parts must be strided views"
        exit 1
    fi
done
echo "ring store guard OK ($N_RING_STORE llm_build_ring_store references, no source ggml_cont)"

echo "== guard: every CUDA fattn n_swa cell-slice goes through can_use_kv_swa_reduction() =="
# ggml-cuda's fattn wrapper reads op_params[4] (n_swa) as licence to keep only the
# newest pad(n_swa + n_tokens) CELLS of the cache and drop the rest outright. That
# is a superset of the window only for ONE append-only sequence over a
# position-ordered cache: ring layers are not position-ordered (rows are
# seq*ring_w + pos % ring_w), and with n_seq_max > 1 interleaved slots spread a
# sequence's window over more cells than the slice keeps -- those cells are absent
# from the tensor rather than masked, i.e. silently wrong output (upstream #2186).
# The predicate lives in exactly one place; this leg pins that no graph builder
# sets the slice without asking it.
mapfile -t SLICE_SITES < <(grep -rnE 'op_params\)?\[4\] = n_swa' "$REPO_DIR/src" || true)
if [ ${#SLICE_SITES[@]} -eq 0 ]; then
    echo "FAIL: no op_params[4] = n_swa site found at all -- this guard has gone stale"
    exit 1
fi
for site in "${SLICE_SITES[@]}"; do
    f="${site%%:*}"; rest="${site#*:}"; ln="${rest%%:*}"
    guard=$(sed -n "$((ln-1))p" "$f")
    case "$guard" in
        *can_use_kv_swa_reduction*) ;;
        *) echo "FAIL: $f:$ln sets the fattn n_swa cell-slice without can_use_kv_swa_reduction()"
           echo "      guard line was: $guard"
           exit 1 ;;
    esac
done
echo "n_swa cell-slice guard OK (${#SLICE_SITES[@]} sites, all gated)"

echo "PASS: SWA ring (--swa-compress) parity"

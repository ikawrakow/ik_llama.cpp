#!/usr/bin/env bash
set -euo pipefail

SERVER="${SERVER:-http://localhost:8080}"
INPUT_FILE="${1:-}"
STATE_FILE="${2:-/tmp/perplexity_state.bin}"
N_CTX="${N_CTX:-512}"
SAVE_EVERY="${SAVE_EVERY:-10}"
TMPDIR="${TMPDIR:-/tmp}"

if [[ -z "$INPUT_FILE" ]]; then
    echo "Usage: $0 <text-file> [state-file]" >&2
    exit 1
fi

TOKEN_FILE="$TMPDIR/perplexity_tokens_$$.json"
REQ_FILE="$TMPDIR/perplexity_req_$$.json"
trap 'rm -f "$TOKEN_FILE" "$REQ_FILE"' EXIT

jq -s -R '{"content": ., "add_special": true}' < "$INPUT_FILE" > "$REQ_FILE"
curl -sf -X POST "$SERVER/tokenize" -H "Content-Type: application/json" -d @"$REQ_FILE" > "$TOKEN_FILE" || {
    echo "[!] Tokenization failed" >&2; exit 1
}

TOTAL_TOKENS=$(jq -e '.tokens | length' "$TOKEN_FILE")
N_CHUNKS=$(( TOTAL_TOKENS / N_CTX ))

print_final() {
    curl -sf "$SERVER/perplexity/state" | jq -r '[.nll, .nll2, .count, .perplexity, .chunk_index] | @tsv' | awk -v n_ctx="$N_CTX" '
    {
        nll=$1; nll2=$2; count=$3; ppl=$4; chunk_index=$5;
        if (count>1) {
            mean=nll/count;
            var=(nll2/count)-(mean*mean);
            if (var>0) {
                unc=(sqrt(var/(count-1)))*ppl;
                printf("Final estimate: PPL over %d chunks for n_ctx=%d = %.4lf +/- %.5lf\n", chunk_index, n_ctx, ppl, unc);
            } else {
                printf("Final estimate: PPL over %d chunks for n_ctx=%d = %.4lf\n", chunk_index, n_ctx, ppl);
            }
        }
    }'
}

save_checkpoint() {
    curl -sf -X POST "$SERVER/perplexity/save" -H "Content-Type: application/json" -d "{\"filename\":\"$STATE_FILE\"}" > /dev/null 2>&1 || true
}

START_CHUNK=0
if [[ -f "$STATE_FILE" ]]; then
    curl -sf -X POST "$SERVER/perplexity/load" -H "Content-Type: application/json" -d "{\"filename\":\"$STATE_FILE\"}" > /dev/null 2>&1 || true
    START_CHUNK=$(curl -sf "$SERVER/perplexity/state" | jq -r '.chunk_index // 0')
fi

if (( START_CHUNK >= N_CHUNKS )); then
    echo "perplexity: calculating perplexity over $N_CHUNKS chunks, n_ctx=$N_CTX, batch_size=$N_CTX, n_seq=1"
    print_final
    read -p "Invalidate state and start over? [y/N] " answer < /dev/tty
    if [[ "$answer" == [yY] ]]; then
        rm -f "$STATE_FILE"
        curl -sf -X POST "$SERVER/perplexity/reset" -H "Content-Type: application/json" -d '{}' > /dev/null 2>&1 || true
        echo "[*] State invalidated. Restart the script to begin from scratch."
    fi
    exit 0
fi

echo "perplexity: calculating perplexity over $N_CHUNKS chunks, n_ctx=$N_CTX, batch_size=$N_CTX, n_seq=1"

shutdown() {
    echo ""
    save_checkpoint
    exit 0
}
trap shutdown SIGINT SIGTERM

chunk_idx=$START_CHUNK
while (( chunk_idx < N_CHUNKS )); do
    start=$(( chunk_idx * N_CTX ))
    jq "{\"tokens\": .tokens[$start:$((start + N_CTX))], \"n_ctx\": $N_CTX}" "$TOKEN_FILE" > "$REQ_FILE"

    resp=$(curl -sf -X POST "$SERVER/perplexity" -H "Content-Type: application/json" -d @"$REQ_FILE") || {
        echo "[!] Request failed at chunk $((chunk_idx + 1))" >&2; exit 1
    }

    if echo "$resp" | jq -e '.error' > /dev/null 2>&1; then
        echo "[!] Server error: $(echo "$resp" | jq -c '.error')" >&2; exit 1
    fi

    idx=$(echo "$resp" | jq -r '.chunk_index')
    ppl=$(echo "$resp" | jq -r '.perplexity')
    printf "[%d]%.4lf," "$idx" "$ppl"

    if (( (chunk_idx + 1) % SAVE_EVERY == 0 )); then
        save_checkpoint
    fi

    chunk_idx=$((chunk_idx + 1))
done

echo ""
print_final
save_checkpoint

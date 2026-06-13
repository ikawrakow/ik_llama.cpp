#pragma once
// PXA_LLAMA_MTP_FIX: shared distinct-sequence decomposition for the batched delta-net.
//
// Both the delta_net graph builder (src/llama-delta-net.cpp) and the input filler
// (llama_set_inputs in src/llama.cpp) must agree EXACTLY on:
//   - the number of distinct sequences in the ubatch (n_seqs),
//   - the first-seen order of those distinct seq_ids (seqs[]),
//   - the per-token slot index seq_slot[t] in [0, n_seqs),
//   - the per-seq token count n_seq_tokens[s] (and whether it is uniform),
//   - per-seq "starts at pos 0" (reset) flag.
//
// v4 hardwired n_seqs == n_tok (one token per seq) which is correct ONLY for plain
// np>1 single-token decode. An MTP *verify* batch carries n_seqs sequences x (n_max+1)
// tokens, the SAME seq_id repeated on consecutive positions. This decomposition
// generalizes to that: one recurrent state row per sequence (gathered once /
// scattered once), with the in-kernel conv/delta-net scan iterating n_seq_tokens
// within each sequence.

#include "llama.h"

#include <cstdint>
#include <vector>

struct pxa_seq_decomp {
    bool                       ok            = false; // batch is acceptable for the batched path
    bool                       all_same      = false; // single distinct sequence
    bool                       uniform_tok   = false; // every seq has the same n_seq_tokens
    int64_t                    n_seqs        = 0;
    int64_t                    n_seq_tokens  = 0;      // valid iff uniform_tok
    std::vector<llama_seq_id>  seqs;                   // distinct absolute seq_ids, first-seen order
    std::vector<int32_t>       seq_slot;               // per-token: index into seqs[] (size n_tokens)
    std::vector<int32_t>       seq_count;              // per-distinct-seq token count (size n_seqs)
    std::vector<llama_pos>     seq_first_pos;          // per-distinct-seq first token pos (size n_seqs)
    std::vector<char>          seq_reset;              // per-distinct-seq: first token pos == 0
};

// Build the decomposition. Acceptance rule: each sequence's tokens must be
// CONTIGUOUS in batch order and POSITION-MONOTONIC (strictly increasing pos).
// This is true for plain decode (1 tok/seq) AND for an MTP verify batch
// (slot-major: seqs assembled one after another, positions increasing).
// Genuinely interleaved batches (a seq_id reappearing after a different seq) are
// rejected (ok=false) so the caller can fall back / assert.
static inline pxa_seq_decomp pxa_decompose_seqs(const struct llama_batch & batch) {
    pxa_seq_decomp d;
    const int n = batch.n_tokens;
    if (n <= 0) { return d; }

    const bool has_seq = (batch.n_seq_id != nullptr && batch.seq_id != nullptr);
    d.seq_slot.resize(n, 0);

    auto tok_seq = [&](int i) -> llama_seq_id {
        if (has_seq && batch.seq_id[i]) return batch.seq_id[i][0];
        return 0;
    };
    auto tok_pos = [&](int i) -> llama_pos {
        return batch.pos ? batch.pos[i] : 0;
    };

    // If multi-seq-per-token is present we cannot decompose (not supported here).
    if (has_seq) {
        for (int i = 0; i < n; ++i) {
            if (batch.n_seq_id[i] != 1) { return d; }
        }
    }

    // First-seen distinct sequence list + per-token slot, requiring contiguity.
    // We track the slot of the immediately-preceding token; a seq may only extend
    // the current run or open a brand-new (never-before-seen) run.
    int prev_slot = -1;
    for (int i = 0; i < n; ++i) {
        const llama_seq_id s = tok_seq(i);
        int slot = -1;
        if (prev_slot >= 0 && d.seqs[prev_slot] == s) {
            slot = prev_slot; // extend current run
        } else {
            // must be a NEW sequence (contiguity): reject if seen before
            for (size_t k = 0; k < d.seqs.size(); ++k) {
                if (d.seqs[k] == s) { return d; } // reappeared non-contiguously -> reject
            }
            slot = (int) d.seqs.size();
            d.seqs.push_back(s);
            d.seq_count.push_back(0);
            d.seq_first_pos.push_back(tok_pos(i));
        }
        // position-monotonic within a run
        if (slot == prev_slot) {
            if (tok_pos(i) <= tok_pos(i-1)) { return d; }
        }
        d.seq_slot[i]      = slot;
        d.seq_count[slot] += 1;
        prev_slot          = slot;
    }

    d.n_seqs   = (int64_t) d.seqs.size();
    d.all_same = (d.n_seqs == 1);

    d.uniform_tok = true;
    for (int64_t s = 0; s < d.n_seqs; ++s) {
        if (d.seq_count[s] != d.seq_count[0]) { d.uniform_tok = false; break; }
    }
    if (d.uniform_tok) {
        d.n_seq_tokens = d.seq_count[0];
    }

    d.seq_reset.resize(d.n_seqs, 0);
    for (int64_t s = 0; s < d.n_seqs; ++s) {
        d.seq_reset[s] = (d.seq_first_pos[s] == 0) ? 1 : 0;
    }

    d.ok = true;
    return d;
}

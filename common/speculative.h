#pragma once

#include "llama.h"
#include "common.h"

#include <vector>
#include <string>
#include <memory>
#include <cmath>
#include <cstdint>

struct common_speculative;

// comma separated list of all types
std::string common_speculative_type_name_str();

// convert string to type
enum common_speculative_type common_speculative_type_from_name(const std::string & name);

// convert type to string
std::string common_speculative_type_to_str(enum common_speculative_type type);

// check if the llama_context is compatible for speculative decoding
// note: clears the memory of the context
bool common_speculative_is_compat(llama_context * ctx_tgt);

common_speculative * common_speculative_init(
        common_params_speculative & params,
        llama_context             * ctx_tgt);

void common_speculative_free(common_speculative * spec);

// optionally call once at the beginning of a new generation
void common_speculative_begin(common_speculative * spec, const llama_tokens & prompt);

// sample up to n_draft tokens and add them to the batch using the draft model
llama_tokens common_speculative_draft(
                     common_speculative * spec,
                     common_params_speculative & params,
                     const llama_tokens & prompt,
                            llama_token   id_last);

// informs the speculative decoder that n_accepted tokens were accepted by the target model
void common_speculative_accept(common_speculative * spec, uint16_t n_accepted);

void common_speculative_print_stats(const common_speculative * spec, double slot_tps = 0.0, int n_decoded = 0, int n_past = 0, common_params_speculative * active_params = nullptr);

// Speculative Auto-Tuner
struct spec_tuner_arm {
    float  value;
    double Q     = 0.0; // mean per-step Tokens-Per-Second (TPS)
    int    N     = 0;
};

struct spec_tuner_coord {
    std::string name;
    std::vector<spec_tuner_arm> arms;
    int current_idx = 0;
    int best_idx    = 0;
    int user_idx    = 0;

    int select_epsilon_greedy(double epsilon) const;

    void update(double reward);

    void reset_scores();

    void build_grid_float(float lo, float hi, int n_points, float user_value);
    void build_grid_int(int lo, int hi, int step, int user_value);
    int  find_nearest_arm(float value) const;
};

struct spec_tuner {
    bool     enabled    = false;

    double   epsilon    = 0.15;  // 15% explore, 85% exploit

    // task-change detection (per-call)
    // If tuner goes bad for 30 consecutive calls, reset the tuner.
    double   step_ema        = 0.0;
    double   step_ema_alpha  = 0.05;
    double   step_drop_pct   = 0.30;
    int      n_low           = 0; 
    int      reset_after     = 30;
    int      cooldown        = 0;
    int      cooldown_max    = 50;
    int      n_resets        = 0;

    int      last_n_drafted  = 0;
    uint64_t n_calls         = 0;
    int      log_every       = 50;

    // per-request tracking
    uint64_t n_requests      = 0;
    int64_t  t_tuner_us      = 0;
    double   ema_tps         = 0.0;
    double   ema_alpha       = 0.3;

    common_speculative_type spec_type = COMMON_SPECULATIVE_TYPE_NONE;
    std::vector<spec_tuner_coord> coords;

    void init(common_speculative_type type, const common_params_speculative & user_params);
    void propose(common_params_speculative & params);
    void accept_feedback(int n_accepted, int n_drafted, double step_tps);
    void end_of_request(double slot_tps, int n_past, common_params_speculative & active_params);
    void enforce_constraints(common_params_speculative & params);
    void print_best() const;
    void reset_exploration();

    void write_best(common_params_speculative & params) const;
};

// Generates speculative draft tokens using the Multi-Token Prediction (MTP) architecture.
std::vector<llama_token> mtp_speculative_gen_draft(
    struct common_sampler * smpl,
    struct llama_context * ctx,
    int n_draft,
    float p_min,
    llama_token id_last,
    int32_t n_past,
    llama_seq_id seq_id);

void mtp_update_kv_cache(struct llama_context * ctx, const llama_batch& batch, bool is_prompt_warmup);

void mtp_accept_tokens(
    struct llama_context * ctx,
    const std::vector<llama_token> & ids,
    int32_t n_past_base,
    llama_seq_id seq_id
);

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

void common_speculative_print_stats(const common_speculative * spec, double slot_tps = 0.0, int n_decoded = 0);


// Speculative Auto-Tuner (Coordinate UCB with Discounting)
struct spec_tuner_arm {
    float  value;
    double Q       = 0.0;   // discounted mean reward (t/s)
    double N       = 1e-9;  // discounted visit count
};

struct spec_tuner_coord {
    std::string name;
    std::vector<spec_tuner_arm> arms;
    int current_idx = 0;

    float  best_value = 0.0f;
    double best_Q     = 0.0;

    int select_ucb(double c, double N_total) const;

    void update(double reward, double gamma);

    void warm_start(float user_value, double init_Q, double init_N);

    void build_grid_float(float lo, float hi, int n_points, float user_value);
    void build_grid_int(int lo, int hi, int step, int user_value);
};

struct spec_tuner {
    bool     enabled     = false;
    bool     proposed    = false;
    double   gamma       = 0.85;    // aggressive discount factor
    double   c           = 2.0;
    double   ema_tps     = 0.0;     // exponential moving avg of absolute t/s (baseline for relative reward)
    double   ema_alpha   = 0.5;
    uint64_t total_n     = 0;
    int      min_tokens  = 10;
    int64_t  t_tuner_us  = 0;

    double   tps_history[8] = {};
    int      tps_idx        = 0;
    bool     boosted        = false;
    double   c_base         = 2.0;
    double   c_boost        = 4.0;

    common_speculative_type spec_type = COMMON_SPECULATIVE_TYPE_NONE;
    std::vector<spec_tuner_coord> coords;

    void init(common_speculative_type type, const common_params_speculative & user_params);

    void propose(common_params_speculative & params);

    void feedback(double slot_tps, int n_decoded, common_params_speculative & active_params);

    void enforce_constraints(common_params_speculative & params);

    void print_best() const;
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

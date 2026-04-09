#pragma once

#include "common.h"

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

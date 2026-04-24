#include "spec-tuner.h"

#include "ggml.h"
#include "log.h"

#include <algorithm>
#include <iomanip>
#include <random>

int spec_tuner_coord::find_nearest_arm(float value) const {
    int idx = 0;
    float best_dist = 1e30f;
    for (int i = 0; i < (int)arms.size(); i++) {
        float dist = std::fabs(arms[i].value - value);
        if (dist < best_dist) {
            best_dist = dist;
            idx       = i;
        }
    }
    return idx;
}

int spec_tuner_coord::select_epsilon_greedy(double epsilon) const {
    static thread_local std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<double> coin(0.0, 1.0);

    if (coin(rng) < epsilon) {
        std::uniform_int_distribution<int> dist(0, (int)arms.size() - 1);
        return dist(rng);
    }
    return best_idx;
}

void spec_tuner_coord::update(double reward) {
    auto & arm = arms[current_idx];
    arm.N += 1;
    arm.Q += (reward - arm.Q) / arm.N;

    double best_Q = -1e30;
    for (int i = 0; i < (int)arms.size(); i++) {
        if (arms[i].N > 0 && arms[i].Q > best_Q) {
            best_Q   = arms[i].Q;
            best_idx = i;
        }
    }
}

void spec_tuner_coord::reset_scores() {
    for (auto & arm : arms) {
        arm.Q = 0.0;
        arm.N = 0;
    }
    current_idx = user_idx;
    best_idx    = user_idx;
}

void spec_tuner_coord::build_grid_float(float lo, float hi, int n_points, float user_value) {
    arms.clear();
    for (int i = 0; i < n_points; i++) {
        float v = lo + (hi - lo) * i / std::max(1, n_points - 1);
        arms.push_back({v, 0.0, 0});
    }
    bool found = false;
    for (auto & a : arms) {
        if (std::fabs(a.value - user_value) < 1e-6f) { found = true; break; }
    }
    if (!found) {
        arms.push_back({user_value, 0.0, 0});
        std::sort(arms.begin(), arms.end(), [](const spec_tuner_arm & a, const spec_tuner_arm & b) {
            return a.value < b.value;
        });
    }
}

void spec_tuner_coord::build_grid_int(int lo, int hi, int step, int user_value) {
    arms.clear();
    for (int v = lo; v <= hi; v += step) {
        arms.push_back({(float)v, 0.0, 0});
    }
    if (arms.empty() || (int)arms.back().value != hi) {
        arms.push_back({(float)hi, 0.0, 0});
    }
    bool found = false;
    for (auto & a : arms) {
        if ((int)a.value == user_value) { found = true; break; }
    }
    if (!found && user_value >= lo && user_value <= hi) {
        arms.push_back({(float)user_value, 0.0, 0});
        std::sort(arms.begin(), arms.end(), [](const spec_tuner_arm & a, const spec_tuner_arm & b) {
            return a.value < b.value;
        });
    }
}

void spec_tuner::reset_exploration() {
    n_resets++;
    LOG_DBG("Autotune task change detected (n_low=%d) — resetting MAB (reset #%d)\n", n_low, n_resets);
    for (auto & coord : coords) {
        coord.reset_scores();
    }
    n_low = 0;
    cooldown = cooldown_max;
    step_ema = 0.0;
    n_calls = 0;
}

void spec_tuner::write_best(common_params_speculative & params) const {
    for (const auto & coord : coords) {
        float val = coord.arms[coord.best_idx].value;
        if      (coord.name == "n_max")                params.n_max                = (int32_t)val;
        else if (coord.name == "p_min")                params.p_min                = val;
        else if (coord.name == "n_min")                params.n_min                = (int32_t)val;
        else if (coord.name == "ngram_size_n")         params.ngram_size_n         = (uint16_t)val;
        else if (coord.name == "ngram_size_m")         params.ngram_size_m         = (uint16_t)val;
        else if (coord.name == "ngram_min_hits")       params.ngram_min_hits       = (uint16_t)val;
        else if (coord.name == "suffix_min_match_len") params.suffix_min_match_len = (int32_t)val;
    }
}

void spec_tuner::init(common_speculative_type type, const common_params_speculative & user_params) {
    enabled    = true;
    spec_type  = type;
    coords.clear();
    n_calls    = 0;
    n_requests = 0;
    ema_tps    = 0.0;
    step_ema   = 0.0;
    n_low      = 0;
    cooldown   = 0;
    n_resets   = 0;
    t_tuner_us = 0;
    last_n_drafted = 0;

    // all types get n_max
    // For simplicity we will create a fixed grid of possible values
    {
        spec_tuner_coord coord;
        coord.name = "n_max";
        int hi = std::max(16, (int)user_params.n_max);
        coord.build_grid_int(1, hi, 1, user_params.n_max);
        coords.push_back(std::move(coord));
    }

    if (type == COMMON_SPECULATIVE_TYPE_DRAFT) {
        {
            spec_tuner_coord coord;
            coord.name = "p_min";
            coord.build_grid_float(0.0f, 0.95f, 11, user_params.p_min);
            coords.push_back(std::move(coord));
        }
        {
            spec_tuner_coord coord;
            coord.name = "n_min";
            coord.build_grid_int(0, 6, 1, user_params.n_min);
            coords.push_back(std::move(coord));
        }
    }

    if (type == COMMON_SPECULATIVE_TYPE_SUFFIX) {
        {
            spec_tuner_coord coord;
            coord.name = "p_min";
            coord.build_grid_float(0.0f, 0.95f, 11, user_params.p_min);
            coords.push_back(std::move(coord));
        }
        {
            spec_tuner_coord coord;
            coord.name = "suffix_min_match_len";
            coord.build_grid_int(1, 12, 1, user_params.suffix_min_match_len);
            coords.push_back(std::move(coord));
        }
    }

    // Ngram can change only n_max/n_min per call
    if (type == COMMON_SPECULATIVE_TYPE_NGRAM_MOD) {
        {
            spec_tuner_coord coord;
            coord.name = "n_min";
            int hi = std::max(0, std::min(4, (int)user_params.n_max - 1));
            coord.build_grid_int(0, hi, 1, user_params.n_min);
            coords.push_back(std::move(coord));
        }
    }

    for (auto & coord : coords) {
        float user_val = 0.0f;
        if      (coord.name == "n_max")                user_val = (float)user_params.n_max;
        else if (coord.name == "p_min")                user_val = user_params.p_min;
        else if (coord.name == "n_min")                user_val = (float)user_params.n_min;
        else if (coord.name == "ngram_size_n")         user_val = (float)user_params.ngram_size_n;
        else if (coord.name == "ngram_size_m")         user_val = (float)user_params.ngram_size_m;
        else if (coord.name == "ngram_min_hits")       user_val = (float)user_params.ngram_min_hits;
        else if (coord.name == "suffix_min_match_len") user_val = (float)user_params.suffix_min_match_len;

        coord.user_idx    = coord.find_nearest_arm(user_val);
        coord.best_idx    = 0;
        coord.current_idx = 0;
    }

    LOG_DBG("Autotune ε-greedy (ε=%.2f) per-draft-call, reward=per-step TPS\n", epsilon);
    for (const auto & coord : coords) {
        std::ostringstream oss;
        oss << "  " << coord.name << ": [";
        for (size_t i = 0; i < coord.arms.size(); i++) {
            if (i > 0) oss << ", ";
            oss << coord.arms[i].value;
        }
        oss << "] (user=" << coord.arms[coord.user_idx].value << ")";
        LOG_DBG("%s\n", oss.str().c_str());
    }
}

void spec_tuner::propose(common_params_speculative & params) {
    int64_t t_start = ggml_time_us();

    // always select fresh arm for every draft call
    for (auto & coord : coords) {
        coord.current_idx = coord.select_epsilon_greedy(epsilon);

        float val = coord.arms[coord.current_idx].value;
        if      (coord.name == "n_max")                params.n_max                = (int32_t)val;
        else if (coord.name == "p_min")                params.p_min                = val;
        else if (coord.name == "n_min")                params.n_min                = (int32_t)val;
        else if (coord.name == "ngram_size_n")         params.ngram_size_n         = (uint16_t)val;
        else if (coord.name == "ngram_size_m")         params.ngram_size_m         = (uint16_t)val;
        else if (coord.name == "ngram_min_hits")       params.ngram_min_hits       = (uint16_t)val;
        else if (coord.name == "suffix_min_match_len") params.suffix_min_match_len = (int32_t)val;
    }

    enforce_constraints(params);
    t_tuner_us += (ggml_time_us() - t_start);
}

void spec_tuner::enforce_constraints(common_params_speculative & params) {
    if (params.n_min < 0)    params.n_min = 0;
    if (params.n_max < 1)    params.n_max = 1;
    if (params.n_min > params.n_max) params.n_min = params.n_max;

    if (params.p_min < 0.0f)  params.p_min = 0.0f;
    if (params.p_min > 0.95f) params.p_min = 0.95f;

    if (params.ngram_size_n < 1)   params.ngram_size_n = 1;
    if (params.ngram_size_m < 1)   params.ngram_size_m = 1;
    if (params.ngram_min_hits < 1) params.ngram_min_hits = 1;
}

void spec_tuner::accept_feedback(int n_accepted, int n_drafted, double step_tps) {
    int64_t t_start = ggml_time_us();
    n_calls++;

    // per-step TPS as reward: captures draft cost, verification cost, and acceptance benefit
    double reward = step_tps;

    for (auto & coord : coords) {
        coord.update(reward);
    }

    if (cooldown > 0) {
        cooldown--;
        if (step_ema <= 0.0) {
            step_ema = step_tps;
        } else {
            step_ema = step_ema_alpha * step_tps + (1.0 - step_ema_alpha) * step_ema;
        }
    } else if (step_ema <= 0.0) {
        step_ema = step_tps;
    } else {
        if (step_tps < step_ema * (1.0 - step_drop_pct)) {
            n_low++;
            if (n_low >= reset_after) {
                reset_exploration();
                t_tuner_us += (ggml_time_us() - t_start);
                return;
            }
        } else {
            n_low = 0;
        }
        step_ema = step_ema_alpha * step_tps + (1.0 - step_ema_alpha) * step_ema;
    }

    if (n_calls <= 5 || (n_calls % log_every == 0)) {
        std::ostringstream oss;
        oss << "Autotune call=" << n_calls
            << " n_drafted=" << n_drafted
            << " n_accepted=" << n_accepted
            << " step_tps=" << std::fixed << std::setprecision(1) << step_tps
            << " ema=" << std::fixed << std::setprecision(1) << step_ema;
        for (const auto & coord : coords) {
            bool is_int = (coord.name != "p_min");
            oss << " " << coord.name << "=";
            if (is_int) oss << (int)coord.arms[coord.current_idx].value;
            else oss << std::fixed << std::setprecision(2) << coord.arms[coord.current_idx].value;
            oss << "→best=";
            if (is_int) oss << (int)coord.arms[coord.best_idx].value;
            else oss << std::fixed << std::setprecision(2) << coord.arms[coord.best_idx].value;
            oss << "(Q=" << std::fixed << std::setprecision(1) << coord.arms[coord.best_idx].Q
                << ",N=" << coord.arms[coord.best_idx].N << ")";
        }
        LOG_DBG("%s\n", oss.str().c_str());
    }

    t_tuner_us += (ggml_time_us() - t_start);
}

void spec_tuner::end_of_request(double slot_tps, int n_past, common_params_speculative & active_params) {
    int64_t t_start = ggml_time_us();
    n_requests++;

    GGML_UNUSED(n_past);

    if (ema_tps <= 0.0) {
        ema_tps = slot_tps;
    } else {
        ema_tps = ema_alpha * slot_tps + (1.0 - ema_alpha) * ema_tps;
    }

    write_best(active_params);
    enforce_constraints(active_params);

    t_tuner_us += (ggml_time_us() - t_start);
    print_best();
}

void spec_tuner::print_best() const {
    {
        std::ostringstream oss;
        oss << "Autotune req=" << n_requests
            << " calls=" << n_calls
            << " tps=" << std::fixed << std::setprecision(2) << ema_tps;

        if (n_resets > 0) oss << " resets=" << n_resets;
        if (n_low > 0)    oss << " n_low=" << n_low;

        oss << " best:";
        for (const auto & coord : coords) {
            bool is_int = (coord.name != "p_min");
            oss << " " << coord.name << "=";
            if (is_int) oss << (int)coord.arms[coord.best_idx].value;
            else oss << std::fixed << std::setprecision(2) << coord.arms[coord.best_idx].value;
            oss << "(Q=" << std::fixed << std::setprecision(2) << coord.arms[coord.best_idx].Q
                << ",N=" << coord.arms[coord.best_idx].N << ")";
        }

        if (!coords.empty()) {
            oss << " | n_max arms:";
            for (const auto & arm : coords[0].arms) {
                oss << " " << (int)arm.value << "(Q=" << std::fixed << std::setprecision(2) << arm.Q
                    << ",N=" << arm.N << ")";
            }
        }

        oss << " tuner=" << std::fixed << std::setprecision(3) << t_tuner_us / 1000.0 << "ms";
        LOG_DBG("%s\n", oss.str().c_str());
    }

    {
        std::ostringstream oss;
        oss << "Autotune reuse: ";
        for (const auto & coord : coords) {
            bool is_int = (coord.name != "p_min");
            if      (coord.name == "n_max")             oss << "--draft-max ";
            else if (coord.name == "p_min")             oss << "--draft-p-min ";
            else if (coord.name == "n_min")             oss << "--draft-min ";
            else if (coord.name == "ngram_size_n")      oss << "--spec-ngram-size-n ";
            else if (coord.name == "ngram_size_m")      oss << "--spec-ngram-size-m ";
            else if (coord.name == "ngram_min_hits")    oss << "--spec-ngram-min-hits ";
            else if (coord.name == "suffix_min_match_len") oss << "--suffix-pattern-len ";
            else                                        oss << "--" << coord.name << " ";

            if (is_int) oss << (int)coord.arms[coord.best_idx].value << " ";
            else oss << std::fixed << std::setprecision(2) << coord.arms[coord.best_idx].value << " ";
        }
        LOG_INF("%s\n", oss.str().c_str());
    }
}

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace llama_moe_cache_detail {

inline std::vector<double> device_shares(const std::vector<float> & cumulative,
        const std::vector<bool> & eligible) {
    if (cumulative.size() != eligible.size()) {
        throw std::invalid_argument("MoE cache split metadata has inconsistent device counts");
    }

    std::vector<double> shares(cumulative.size(), 0.0);
    double previous = 0.0;
    for (std::size_t i = 0; i < cumulative.size(); ++i) {
        const double point = std::isfinite(cumulative[i]) ? std::max(0.0, double(cumulative[i])) : previous;
        shares[i] = std::max(0.0, point - previous);
        previous = std::max(previous, point);
    }

    double total = 0.0;
    std::size_t eligible_count = 0;
    for (std::size_t i = 0; i < shares.size(); ++i) {
        if (eligible[i]) {
            total += shares[i];
            ++eligible_count;
        } else {
            shares[i] = 0.0;
        }
    }

    if (eligible_count == 0) {
        return shares;
    }
    if (!std::isfinite(total) || total <= 0.0) {
        const double equal_share = 1.0 / eligible_count;
        for (std::size_t i = 0; i < shares.size(); ++i) {
            shares[i] = eligible[i] ? equal_share : 0.0;
        }
        return shares;
    }

    for (std::size_t i = 0; i < shares.size(); ++i) {
        shares[i] = eligible[i] ? shares[i] / total : 0.0;
    }
    return shares;
}

} // namespace llama_moe_cache_detail

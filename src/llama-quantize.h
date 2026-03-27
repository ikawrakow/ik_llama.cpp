#pragma once

#include "ggml.h"

#include <utility>

std::pair<ggml_type, int> interleaved_properties(ggml_type type);

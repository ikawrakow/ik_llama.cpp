#include "json-schema-to-grammar.h"
#include "common.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <map>
#include <regex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using json = nlohmann::ordered_json;

static std::string build_repetition(const std::string & item_rule, int min_items, int max_items, const std::string & separator_rule = "") {
    auto has_max = max_items != std::numeric_limits<int>::max();

    if (max_items == 0) {
        return "";
    }
    if (min_items == 0 && max_items == 1) {
        return item_rule + "?";
    }

    if (separator_rule.empty()) {
        if (min_items == 1 && !has_max) {
            return item_rule + "+";
        }
        if (min_items == 0 && !has_max) {
            return item_rule + "*";
        }
        return item_rule + "{" + std::to_string(min_items) + "," + (has_max ? std::to_string(max_items) : "") + "}";
    }

    auto result = item_rule + " " + build_repetition("(" + separator_rule + " " + item_rule + ")", min_items == 0 ? 0 : min_items - 1, has_max ? max_items - 1 : max_items);
    if (min_items == 0) {
        result = "(" + result + ")?";
    }
    return result;
}

static void build_min_max_int(int64_t min_value, int64_t max_value, std::stringstream & out, int decimals_left = 16, bool top_level = true) {
    auto has_min = min_value != std::numeric_limits<int64_t>::min();
    auto has_max = max_value != std::numeric_limits<int64_t>::max();

    auto digit_range = [&](char from, char to) {
        out << "[";
        if (from == to) {
            out << from;
        } else {
            out << from << "-" << to;
        }
        out << "]";
    };
    auto more_digits = [&](int min_digits, int max_digits) {
        out << "[0-9]";
        if (min_digits == max_digits && min_digits == 1) {
            return;
        }
        out << "{";
        out << min_digits;
        if (max_digits != min_digits) {
            out << ",";
            if (max_digits != std::numeric_limits<int>::max()) {
                out << max_digits;
            }
        }
        out << "}";
    };
    std::function<void(const std::string_view &, const std::string_view &)> uniform_range =
        [&](const std::string_view & from, const std::string_view & to) {
            size_t i = 0;
            while (i < from.length() && i < to.length() && from[i] == to[i]) {
                i++;
            }
            if (i > 0) {
                out << "\"" << from.substr(0, i) << "\"";
            }
            if (i < from.length() && i < to.length()) {
                if (i > 0) {
                    out << " ";
                }
                auto sub_len = from.length() - i - 1;
                if (sub_len > 0) {
                    auto from_sub = from.substr(i + 1);
                    auto to_sub = to.substr(i + 1);
                    auto sub_zeros = string_repeat("0", sub_len);
                    auto sub_nines = string_repeat("9", sub_len);

                    auto to_reached = false;
                    out << "(";
                    if (from_sub == sub_zeros) {
                        digit_range(from[i], to[i] - 1);
                        out << " ";
                        more_digits(sub_len, sub_len);
                    } else {
                        out << "[" << from[i] << "] ";
                        out << "(";
                        uniform_range(from_sub, sub_nines);
                        out << ")";
                        if (from[i] < to[i] - 1) {
                            out << " | ";
                            if (to_sub == sub_nines) {
                                digit_range(from[i] + 1, to[i]);
                                to_reached = true;
                            } else {
                                digit_range(from[i] + 1, to[i] - 1);
                            }
                            out << " ";
                            more_digits(sub_len, sub_len);
                        }
                    }
                    if (!to_reached) {
                        out << " | ";
                        digit_range(to[i], to[i]);
                        out << " ";
                        uniform_range(sub_zeros, to_sub);
                    }
                    out << ")";
                } else {
                    out << "[" << from[i] << "-" << to[i] << "]";
                }
            }
        };

    if (has_min && has_max) {
        if (min_value < 0 && max_value < 0) {
            out << "\"-\" (";
            build_min_max_int(-max_value, -min_value, out, decimals_left, /* top_level= */ true);
            out << ")";
            return;
        }

        if (min_value < 0) {
            out << "\"-\" (";
            build_min_max_int(0, -min_value, out, decimals_left, /* top_level= */ true);
            out << ") | ";
            min_value = 0;
        }

        auto min_s = std::to_string(min_value);
        auto max_s = std::to_string(max_value);
        auto min_digits = min_s.length();
        auto max_digits = max_s.length();

        for (auto digits = min_digits; digits < max_digits; digits++) {
            uniform_range(min_s, string_repeat("9", digits));
            min_s = "1" + string_repeat("0", digits);
            out << " | ";
        }
        uniform_range(min_s, max_s);
        return;
    }

    auto less_decimals = std::max(decimals_left - 1, 1);

    if (has_min) {
        if (min_value < 0) {
            out << "\"-\" (";
            build_min_max_int(std::numeric_limits<int64_t>::min(), -min_value, out, decimals_left, /* top_level= */ false);
            out << ") | [0] | [1-9] ";
            more_digits(0, decimals_left - 1);
        } else if (min_value == 0) {
            if (top_level) {
                out << "[0] | [1-9] ";
                more_digits(0, less_decimals);
            } else {
                more_digits(1, decimals_left);
            }
        } else if (min_value <= 9) {
            char c = '0' + min_value;
            auto range_start = top_level ? '1' : '0';
            if (c > range_start) {
                digit_range(range_start, c - 1);
                out << " ";
                more_digits(1, less_decimals);
                out << " | ";
            }
            digit_range(c, '9');
            out << " ";
            more_digits(0, less_decimals);
        } else {
            auto min_s = std::to_string(min_value);
            auto len = min_s.length();
            auto c = min_s[0];

            if (c > '1') {
                digit_range(top_level ? '1' : '0', c - 1);
                out << " ";
                more_digits(len, less_decimals);
                out << " | ";
            }
            digit_range(c, c);
            out << " (";
            build_min_max_int(std::stoll(min_s.substr(1)), std::numeric_limits<int64_t>::max(), out, less_decimals, /* top_level= */ false);
            out << ")";
            if (c < '9') {
                out << " | ";
                digit_range(c + 1, '9');
                out << " ";
                more_digits(len - 1, less_decimals);
            }
        }
        return;
    }

    if (has_max) {
        if (max_value >= 0) {
            if (top_level) {
                out << "\"-\" [1-9] ";
                more_digits(0, less_decimals);
                out << " | ";
            }
            build_min_max_int(0, max_value, out, decimals_left, /* top_level= */ true);
        } else {
            out << "\"-\" (";
            build_min_max_int(-max_value, std::numeric_limits<int64_t>::max(), out, decimals_left, /* top_level= */ false);
            out << ")";
        }
        return;
    }

    throw std::runtime_error("At least one of min_value or max_value must be set");
}

const std::string SPACE_RULE = "| \" \" | \"\\n\"{1,2} [ \\t]{0,20}";

struct BuiltinRule {
    std::string content;
    std::vector<std::string> deps;
};

static std::unordered_map<std::string, BuiltinRule> PRIMITIVE_RULES = {
    {"boolean", {"(\"true\" | \"false\") space", {}}},
    {"decimal-part", {"[0-9]{1,16}", {}}},
    {"integral-part", {"[0] | [1-9] [0-9]{0,15}", {}}},
    {"number", {"(\"-\"? integral-part) (\".\" decimal-part)? ([eE] [-+]? integral-part)? space", {"integral-part", "decimal-part"}}},
    {"integer", {"(\"-\"? integral-part) space", {"integral-part"}}},
    {"value", {"object | array | string | number | boolean | null", {"object", "array", "string", "number", "boolean", "null"}}},
    {"object", {"\"{\" space ( string \":\" space value (\",\" space string \":\" space value)* )? \"}\" space", {"string", "value"}}},
    {"array", {"\"[\" space ( value (\",\" space value)* )? \"]\" space", {"value"}}},
    {"uuid", {"\"\\\"\" [0-9a-fA-F]{8} \"-\" [0-9a-fA-F]{4} \"-\" [0-9a-fA-F]{4} \"-\" [0-9a-fA-F]{4} \"-\" [0-9a-fA-F]{12} \"\\\"\" space", {}}},
    {"char",   {"[^\"\\\\\\x7F\\x00-\\x1F] | [\\\\] ([\"\\\\bfnrt] | \"u\" [0-9a-fA-F]{4})", {}}},
    {"string", {"\"\\\"\" char* \"\\\"\" space", {"char"}}},
    {"null", {"\"null\" space", {}}},
};

static std::unordered_map<std::string, BuiltinRule> STRING_FORMAT_RULES = {
    {"date", {"[0-9]{4} \"-\" ( \"0\" [1-9] | \"1\" [0-2] ) \"-\" ( \"0\" [1-9] | [1-2] [0-9] | \"3\" [0-1] )", {}}},
    {"time", {"([01] [0-9] | \"2\" [0-3]) \":\" [0-5] [0-9] \":\" [0-5] [0-9] ( \".\" [0-9]{3} )? ( \"Z\" | ( \"+\" | \"-\" ) ( [01] [0-9] | \"2\" [0-3] ) \":\" [0-5] [0-9] )", {}}},
    {"date-time", {"date \"T\" time", {"date", "time"}}},
    {"date-string", {"\"\\\"\" date \"\\\"\" space", {"date"}}},
    {"time-string", {"\"\\\"\" time \"\\\"\" space", {"time"}}},
    {"date-time-string", {"\"\\\"\" date-time \"\\\"\" space", {"date-time"}}}
};

static bool is_reserved_name(const std::string & name) {
    static const std::unordered_set<std::string> RESERVED_NAMES = [] {
        std::unordered_set<std::string> s;
        s.insert("root");
        for (const auto & p : PRIMITIVE_RULES) {
            s.insert(p.first);
        }
        for (const auto & p : STRING_FORMAT_RULES) {
            s.insert(p.first);
        }
        return s;
    }();
    return RESERVED_NAMES.find(name) != RESERVED_NAMES.end();
}

static std::regex INVALID_RULE_CHARS_RE("[^a-zA-Z0-9-]+");
static std::regex GRAMMAR_LITERAL_ESCAPE_RE("[\r\n\"\\\\]");
static std::regex GRAMMAR_RANGE_LITERAL_ESCAPE_RE("[\r\n\"\\]\\-\\\\]");
static std::unordered_map<char, std::string> GRAMMAR_LITERAL_ESCAPES = {
    {'\r', "\\r"}, {'\n', "\\n"}, {'"', "\\\""}, {'-', "\\-"}, {']', "\\]"}, {'\\', "\\\\"}
};

static std::unordered_set<char> NON_LITERAL_SET = {'|', '.', '(', ')', '[', ']', '{', '}', '*', '+', '?'};
static std::unordered_set<char> ESCAPED_IN_REGEXPS_BUT_NOT_IN_LITERALS = {'^', '$', '.', '[', ']', '(', ')', '|', '{', '}', '*', '+', '?'};

static std::string replacePattern(const std::string & input, const std::regex & regex, const std::function<std::string(const std::smatch  &)> & replacement) {
    std::smatch match;
    std::string result;

    std::string::const_iterator searchStart(input.cbegin());
    std::string::const_iterator searchEnd(input.cend());

    while (std::regex_search(searchStart, searchEnd, match, regex)) {
        result.append(searchStart, searchStart + match.position());
        result.append(replacement(match));
        searchStart = match.suffix().first;
    }

    result.append(searchStart, searchEnd);

    return result;
}

static std::string format_literal(const std::string & literal) {
    std::string escaped = replacePattern(literal, GRAMMAR_LITERAL_ESCAPE_RE, [&](const std::smatch & match) {
        char c = match.str()[0];
        return GRAMMAR_LITERAL_ESCAPES.at(c);
    });
    return "\"" + escaped + "\"";
}

std::string gbnf_format_literal(const std::string & literal) { return format_literal(literal); }

// quillama.cpp: exact numeric bounds. A bound is parsed from its JSON text into a decimal
// (sign, integer part, fraction digits without trailing zeros), so 1.2, -0.5 and 1e-05 are
// compared digit by digit and never rounded.
namespace {
struct quill_dec {
    bool        neg  = false;
    int64_t     ip   = 0;  // integer part of the magnitude
    std::string frac;      // fraction digits, no trailing zeros
    bool is_zero() const { return ip == 0 && frac.empty(); }
};

static bool quill_dec_parse(const json & v, quill_dec & out) {
    if (!v.is_number()) {
        return false;
    }
    const std::string s = v.dump();
    quill_dec d;
    size_t i = 0;
    if (i < s.size() && (s[i] == '-' || s[i] == '+')) {
        d.neg = s[i] == '-';
        ++i;
    }
    std::string mant;
    int point = -1;
    for (; i < s.size() && (isdigit((unsigned char) s[i]) || s[i] == '.'); ++i) {
        if (s[i] == '.') {
            point = (int) mant.size();
        } else {
            mant += s[i];
        }
    }
    if (point < 0) {
        point = (int) mant.size();
    }
    if (i < s.size() && (s[i] == 'e' || s[i] == 'E')) {
        point += atoi(s.c_str() + i + 1);
    }
    if (mant.empty() || point > 15 || point < -64) {
        return false;  // beyond what the grammar's integral-part can write, or absurdly small
    }
    std::string ipart = point > 0 ? mant.substr(0, std::min<size_t>(point, mant.size())) : "";
    while ((int) ipart.size() < point) {
        ipart += '0';
    }
    std::string fpart = point >= 0 ? (point < (int) mant.size() ? mant.substr(point) : "")
                                   : std::string(-point, '0') + mant;
    while (!fpart.empty() && fpart.back() == '0') {
        fpart.pop_back();
    }
    d.ip   = ipart.empty() ? 0 : std::stoll(ipart);
    d.frac = fpart;
    if (d.is_zero()) {
        d.neg = false;
    }
    out = d;
    return true;
}

// -1, 0, 1 like a <=> b
static int quill_dec_cmp(const quill_dec & a, const quill_dec & b) {
    if (a.neg != b.neg) {
        return a.neg ? -1 : 1;
    }
    int r = 0;
    if (a.ip != b.ip) {
        r = a.ip < b.ip ? -1 : 1;
    } else {
        const size_t n = std::max(a.frac.size(), b.frac.size());
        for (size_t k = 0; k < n && r == 0; ++k) {
            const char ca = k < a.frac.size() ? a.frac[k] : '0';
            const char cb = k < b.frac.size() ? b.frac[k] : '0';
            r = ca < cb ? -1 : (ca > cb ? 1 : 0);
        }
    }
    return a.neg ? -r : r;
}

// smallest integer >= d (> d when exclusive), largest integer <= d (< d when exclusive)
static int64_t quill_dec_ceil(const quill_dec & d, bool exclusive) {
    const bool integral = d.frac.empty();
    const int64_t c = d.neg ? -d.ip : (integral ? d.ip : d.ip + 1);
    return exclusive && integral ? c + 1 : c;
}
static int64_t quill_dec_floor(const quill_dec & d, bool exclusive) {
    const bool integral = d.frac.empty();
    const int64_t f = d.neg ? (integral ? -d.ip : -d.ip - 1) : d.ip;
    return exclusive && integral ? f - 1 : f;
}

struct quill_frac_rule {
    std::string ne;        // grammar for a non-empty digit sequence, "" if none fits
    bool        empty_ok;  // whether ending here (all remaining digits zero) fits
};

// Fraction digits d[k..] given that d[..k) equals lo[..k) (tight_lo) and hi[..k) (tight_hi).
// lo / hi are fraction digit strings with implicit trailing zeros.
static quill_frac_rule quill_frac_between(const std::string * lo, bool lo_ex, const std::string * hi, bool hi_ex,
                                          size_t k, bool tight_lo, bool tight_hi) {
    auto rest_nonzero = [](const std::string & s, size_t from) {
        for (size_t j = from; j < s.size(); ++j) {
            if (s[j] != '0') {
                return true;
            }
        }
        return false;
    };
    if (tight_lo && k >= lo->size()) {
        // lo exhausted: its remaining digits are zeros, so every continuation is >= lo; > lo needs a
        // nonzero digit
        if (!lo_ex) {
            tight_lo = false;
        } else if (!tight_hi) {
            return { "[0-9]* [1-9] [0-9]*", false };
        }
    }
    if (tight_hi && k >= hi->size()) {
        // hi exhausted: only zeros keep the value <= hi, and nothing keeps it < hi
        if (hi_ex) {
            return { "", false };
        }
        if (!tight_lo) {
            return { "\"0\"+", true };
        }
        // tight on both with lo exclusive and hi exhausted: lo exhausted too (lo <= hi), so equal: none
        return { "", false };
    }
    if (!tight_lo && !tight_hi) {
        return { "[0-9]+", true };
    }
    bool empty_ok = true;
    if (tight_lo) {
        empty_ok = empty_ok && !lo_ex && !rest_nonzero(*lo, k);
    }
    if (tight_hi) {
        empty_ok = empty_ok && (!hi_ex || rest_nonzero(*hi, k));
    }
    const char l = tight_lo ? (k < lo->size() ? (*lo)[k] : '0') : '0';  // past lo's end: implicit zeros
    const char h = tight_hi ? (*hi)[k] : '9';
    auto range = [](char a, char b) {
        return a == b ? std::string("\"") + a + "\"" : std::string("[") + a + "-" + b + "]";
    };
    std::vector<std::string> alts;
    auto add = [&](char a, bool tl, bool th) {
        const quill_frac_rule r = quill_frac_between(lo, lo_ex, hi, hi_ex, k + 1, tl, th);
        if (r.ne.empty() && !r.empty_ok) {
            return;
        }
        const std::string tail = r.ne.empty() ? "" : (r.empty_ok ? " (" + r.ne + ")?" : " (" + r.ne + ")");
        alts.push_back(range(a, a) + tail);
    };
    if (tight_lo && tight_hi && l == h) {
        add(l, true, true);
    } else {
        if (tight_lo) {
            add(l, true, false);
        }
        const char a = tight_lo ? (char) (l + 1) : '0';
        const char b = tight_hi ? (char) (h - 1) : '9';
        if (a <= b) {
            alts.push_back(range(a, b) + " [0-9]*");
        }
        if (tight_hi) {
            add(h, false, true);
        }
    }
    std::string ne;
    for (size_t j = 0; j < alts.size(); ++j) {
        ne += (j ? " | " : "") + alts[j];
    }
    return { ne, empty_ok };
}

// optional "." and fraction digits between lo and hi (nullptr = unbounded); ok = false if none fits
static std::string quill_fraction(const std::string * lo, bool lo_ex, const std::string * hi, bool hi_ex, bool & ok) {
    const quill_frac_rule r = quill_frac_between(lo, lo_ex, hi, hi_ex, 0, lo != nullptr, hi != nullptr);
    ok = !r.ne.empty() || r.empty_ok;
    if (r.ne.empty()) {
        return "";
    }
    return "(\".\" (" + r.ne + "))" + (r.empty_ok ? "?" : "");
}

// magnitudes u >= 0 with lo <= u <= hi (exclusive flags; hi == nullptr: unbounded)
static void quill_magnitude_alts(const quill_dec & lo, bool lo_ex, const quill_dec * hi, bool hi_ex,
                                 std::vector<std::string> & alts) {
    auto lit = [](int64_t v) { return "\"" + std::to_string(v) + "\""; };
    auto with = [](const std::string & a, const std::string & f) { return f.empty() ? a : a + " " + f; };
    bool ok = false;
    if (hi && hi->ip == lo.ip) {
        const std::string f = quill_fraction(&lo.frac, lo_ex, &hi->frac, hi_ex, ok);
        if (ok) {
            alts.push_back(with(lit(lo.ip), f));
        }
        return;
    }
    const std::string f = quill_fraction(&lo.frac, lo_ex, nullptr, false, ok);
    if (ok) {
        alts.push_back(with(lit(lo.ip), f));
    }
    const int64_t a = lo.ip + 1;
    const int64_t b = hi ? hi->ip - 1 : std::numeric_limits<int64_t>::max();
    if (a <= b) {
        std::stringstream out;
        out << "(";
        build_min_max_int(a, b, out);
        out << ") (\".\" [0-9]+)?";
        alts.push_back(out.str());
    }
    if (hi) {
        const std::string fh = quill_fraction(nullptr, false, &hi->frac, hi_ex, ok);
        if (ok) {
            alts.push_back(with(lit(hi->ip), fh));
        }
    }
}

// a JSON number v with lo <= v <= hi (nullptr = unbounded), written without an exponent
static std::string quill_number_range(const quill_dec * lo, bool lo_ex, const quill_dec * hi, bool hi_ex) {
    std::vector<std::string> alts;
    const quill_dec zero;
    // v >= 0
    if ((!hi || !hi->neg) && !(hi && hi->is_zero() && hi_ex)) {
        const bool lo_pos = lo && !lo->neg;
        quill_magnitude_alts(lo_pos ? *lo : zero, lo_pos && lo_ex, hi, hi_ex, alts);
    }
    // v < 0, written as "-" u with u = -v > 0
    if (!lo || lo->neg) {
        quill_dec ulo = zero;
        bool ulo_ex = true;
        if (hi && hi->neg) {
            ulo = *hi;
            ulo.neg = false;
            ulo_ex = hi_ex;
        }
        quill_dec uhi;
        if (lo) {
            uhi = *lo;
            uhi.neg = false;
        }
        std::vector<std::string> neg;
        quill_magnitude_alts(ulo, ulo_ex, lo ? &uhi : nullptr, lo_ex, neg);
        for (const auto & a : neg) {
            alts.push_back("\"-\" " + a);
        }
    }
    std::string r;
    for (size_t j = 0; j < alts.size(); ++j) {
        r += (j ? " | " : "") + std::string("(") + alts[j] + ")";
    }
    return r;
}

// reads minimum / exclusiveMinimum / maximum / exclusiveMaximum (also draft-4 boolean exclusives);
// false if a bound is present but cannot be represented exactly
static bool quill_read_bounds(const json & schema, bool & has_lo, quill_dec & lo, bool & lo_ex,
                              bool & has_hi, quill_dec & hi, bool & hi_ex) {
    has_lo = has_hi = lo_ex = hi_ex = false;
    auto take = [&](const char * key, bool exclusive, bool is_lower) -> bool {
        if (!schema.contains(key) || schema[key].is_boolean()) {
            return true;
        }
        quill_dec d;
        if (!quill_dec_parse(schema[key], d)) {
            return false;
        }
        bool &      has = is_lower ? has_lo : has_hi;
        quill_dec & cur = is_lower ? lo : hi;
        bool &      ex  = is_lower ? lo_ex : hi_ex;
        const int c = has ? quill_dec_cmp(d, cur) : 0;
        if (!has || (is_lower ? c > 0 : c < 0) || (c == 0 && exclusive)) {  // keep the stricter bound
            cur = d;
            ex  = exclusive;
            has = true;
        }
        return true;
    };
    auto flag = [&](const char * key) {
        return schema.contains(key) && schema[key].is_boolean() && schema[key].get<bool>();
    };
    return take("minimum", flag("exclusiveMinimum"), true) && take("exclusiveMinimum", true, true) &&
           take("maximum", flag("exclusiveMaximum"), false) && take("exclusiveMaximum", true, false);
}
} // namespace

class common_schema_converter {
private:
    friend class common_schema_info;
    friend std::string build_grammar(const std::function<void(const common_grammar_builder &)> & cb, const common_grammar_options & options);
    std::function<json(const std::string &)> _fetch_json;
    bool _dotall;
    std::map<std::string, std::string> _rules;
    std::unordered_map<std::string, json> _refs;
    std::unordered_set<std::string> _refs_being_resolved;
    std::vector<std::string> _errors;
    std::vector<std::string> _warnings;

    std::string _add_rule(const std::string & name, const std::string & rule) {
        std::string esc_name = regex_replace(name, INVALID_RULE_CHARS_RE, "-");
        if (_rules.find(esc_name) == _rules.end() || _rules[esc_name] == rule) {
            _rules[esc_name] = rule;
            return esc_name;
        }
        int i = 0;
        while (_rules.find(esc_name + std::to_string(i)) != _rules.end() && _rules[esc_name + std::to_string(i)] != rule) {
            i++;
        }
        std::string key = esc_name + std::to_string(i);
        _rules[key] = rule;
        return key;
    }

    std::string _generate_union_rule(const std::string & name, const std::vector<json> & alt_schemas) {
        std::vector<std::string> rules;
        rules.reserve(alt_schemas.size());
        for (size_t i = 0; i < alt_schemas.size(); i++) {
            rules.push_back(visit(alt_schemas[i], name + (name.empty() ? "alternative-" : "-") + std::to_string(i)));
        }
        return string_join(rules, " | ");
    }

    std::string _visit_pattern(const std::string & pattern, const std::string & name) {
        if (!(pattern.front() == '^' && pattern.back() == '$')) {
            _errors.push_back("Pattern must start with '^' and end with '$'");
            return "";
        }
        std::string sub_pattern = pattern.substr(1, pattern.length() - 2);
        std::unordered_map<std::string, std::string> sub_rule_ids;

        size_t i = 0;
        size_t length = sub_pattern.length();

        using literal_or_rule = std::pair<std::string, bool>;
        auto to_rule = [&](const literal_or_rule & ls) {
            auto is_literal = ls.second;
            auto s = ls.first;
            return is_literal ? "\"" + s + "\"" : s;
        };
        std::function<literal_or_rule()> transform = [&]() -> literal_or_rule {
            size_t start = i;
            std::vector<literal_or_rule> seq;

            auto get_dot = [&]() {
                std::string rule;
                if (_dotall) {
                    rule = "[\\U00000000-\\U0010FFFF]";
                } else {
                    rule = "[^\\x0A\\x0D]";
                }
                return _add_rule("dot", rule);
            };

            // Joins the sequence, merging consecutive literals together.
            auto join_seq = [&]() {
                std::vector<literal_or_rule> ret;

                std::string literal;
                auto flush_literal = [&]() {
                    if (literal.empty()) {
                        return false;
                    }
                    ret.emplace_back(literal, true);
                    literal.clear();
                    return true;
                };

                for (const auto & item : seq) {
                    auto is_literal = item.second;
                    if (is_literal) {
                        literal += item.first;
                    } else {
                        flush_literal();
                        ret.push_back(item);
                    }
                }
                flush_literal();

                std::vector<std::string> results;
                results.reserve(ret.size());
                for (const auto & item : ret) {
                    results.push_back(to_rule(item));
                }
                return std::make_pair(string_join(results, " "), false);
            };

            while (i < length) {
                char c = sub_pattern[i];
                if (c == '.') {
                    seq.emplace_back(get_dot(), false);
                    i++;
                } else if (c == '(') {
                    i++;
                    if (i < length && sub_pattern[i] == '?') {
                        if (i + 1 < length && sub_pattern[i + 1] == ':') {
                            i += 2; // skip "?:" for non-capturing group, treat as regular group
                        } else {
                            // lookahead/lookbehind (?=, ?!, ?<=, ?<!) - not supported
                            _warnings.push_back("Unsupported pattern syntax");
                            // skip to matching ')' to avoid UB on empty seq
                            int depth = 1;
                            while (i < length && depth > 0) {
                                if (sub_pattern[i] == '\\' && i + 1 < length) {
                                    i += 2; // skip escaped character
                                } else {
                                    if (sub_pattern[i] == '(') depth++;
                                    else if (sub_pattern[i] == ')') depth--;
                                    i++;
                                }
                            }
                            continue;
                        }
                    }
                    seq.emplace_back("(" + to_rule(transform()) + ")", false);
                } else if (c == ')') {
                    i++;
                    if (start > 0 && sub_pattern[start - 1] != '(' && (start < 2 || sub_pattern[start - 2] != '?' || sub_pattern[start - 1] != ':')) {
                        _errors.push_back("Unbalanced parentheses");
                    }
                    return join_seq();
                } else if (c == '[') {
                    std::string square_brackets = std::string(1, c);
                    i++;
                    while (i < length && sub_pattern[i] != ']') {
                        if (sub_pattern[i] == '\\') {
                            square_brackets += sub_pattern.substr(i, 2);
                            i += 2;
                        } else {
                            square_brackets += sub_pattern[i];
                            i++;
                        }
                    }
                    if (i >= length) {
                        _errors.push_back("Unbalanced square brackets");
                    }
                    square_brackets += ']';
                    i++;
                    seq.emplace_back(square_brackets, false);
                } else if (c == '|') {
                    seq.emplace_back("|", false);
                    i++;
                } else if (c == '*' || c == '+' || c == '?') {
                    seq.back() = std::make_pair(to_rule(seq.back()) + c, false);
                    i++;
                } else if (c == '{') {
                    std::string curly_brackets = std::string(1, c);
                    i++;
                    while (i < length && sub_pattern[i] != '}') {
                        curly_brackets += sub_pattern[i];
                        i++;
                    }
                    if (i >= length) {
                        _errors.push_back("Unbalanced curly brackets");
                    }
                    curly_brackets += '}';
                    i++;
                    auto nums = string_split(curly_brackets.substr(1, curly_brackets.length() - 2), ",");
                    int min_times = 0;
                    int max_times = std::numeric_limits<int>::max();
                    try {
                        if (nums.size() == 1) {
                            min_times = max_times = std::stoi(nums[0]);
                        } else if (nums.size() != 2) {
                            _errors.push_back("Wrong number of values in curly brackets");
                        } else {
                            if (!nums[0].empty()) {
                                min_times = std::stoi(nums[0]);
                            }
                            if (!nums[1].empty()) {
                                max_times = std::stoi(nums[1]);
                            }
                        }
                    } catch (const std::invalid_argument & e) {
                        _errors.push_back("Invalid number in curly brackets");
                        return std::make_pair("", false);
                    }
                    auto &last = seq.back();
                    auto &sub = last.first;
                    auto sub_is_literal = last.second;

                    if (!sub_is_literal) {
                        std::string & sub_id = sub_rule_ids[sub];
                        if (sub_id.empty()) {
                            sub_id = _add_rule(name + "-" + std::to_string(sub_rule_ids.size()), sub);
                        }
                        sub = sub_id;
                    }
                    seq.back().first = build_repetition(
                        sub_is_literal ? "\"" + sub + "\"" : sub,
                        min_times,
                        max_times,
                        ""
                    );
                    seq.back().second = false;
                } else {
                    std::string literal;
                    auto is_non_literal = [&](char c) {
                        return NON_LITERAL_SET.find(c) != NON_LITERAL_SET.end();
                    };
                    while (i < length) {
                        if (sub_pattern[i] == '\\' && i < length - 1) {
                            char next = sub_pattern[i + 1];
                            if (ESCAPED_IN_REGEXPS_BUT_NOT_IN_LITERALS.find(next) != ESCAPED_IN_REGEXPS_BUT_NOT_IN_LITERALS.end()) {
                                i++;
                                literal += sub_pattern[i];
                                i++;
                            } else {
                                literal += sub_pattern.substr(i, 2);
                                i += 2;
                            }
                        } else if (sub_pattern[i] == '"') {
                            literal += "\\\"";
                            i++;
                        } else if (!is_non_literal(sub_pattern[i]) &&
                                (i == length - 1 || literal.empty() || sub_pattern[i + 1] == '.' || !is_non_literal(sub_pattern[i + 1]))) {
                            literal += sub_pattern[i];
                            i++;
                        } else {
                            break;
                        }
                    }
                    if (!literal.empty()) {
                        seq.emplace_back(literal, true);
                    }
                }
            }
            return join_seq();
        };
        return _add_rule(name, "\"\\\"\" (" + to_rule(transform()) + ") \"\\\"\" space");
    }

    /*
        Returns a rule that matches a JSON string that is none of the provided strings

        not_strings({"a"})
            -> ["] ( [a] char+ | [^"a] char* )? ["] space
        not_strings({"and", "also"})
            -> ["] ( [a] ([l] ([s] ([o] char+ | [^"o] char*) | [^"s] char*) | [n] ([d] char+ | [^"d] char*) | [^"ln] char*) | [^"a] char* )? ["] space
    */
    std::string _not_strings(const std::vector<std::string> & strings) {

        struct TrieNode {
            std::map<char, TrieNode> children;
            bool is_end_of_string;

            TrieNode() : is_end_of_string(false) {}

            void insert(const std::string & string) {
                auto *node = this;
                for (char c : string) {
                    node = &node->children[c];
                }
                node->is_end_of_string = true;
            }
        };

        TrieNode trie;
        for (const auto & s : strings) {
            trie.insert(s);
        }

        std::string char_rule = _add_primitive("char", PRIMITIVE_RULES.at("char"));
        std::ostringstream out;
        out << "[\"] ( ";
        std::function<void(const TrieNode &)> visit = [&](const TrieNode & node) {
            std::ostringstream rejects;
            auto first = true;
            for (const auto & kv : node.children) {
                rejects << kv.first;
                if (first) {
                    first = false;
                } else {
                    out << " | ";
                }
                out << "[" << kv.first << "]";
                if (!kv.second.children.empty()) {
                    out << " (";
                    visit(kv.second);
                    out << ")";
                } else if (kv.second.is_end_of_string) {
                    out << " " << char_rule << "+";
                }
            }
            if (!node.children.empty()) {
                if (!first) {
                    out << " | ";
                }
                out << "[^\"" << rejects.str() << "] " << char_rule << "*";
            }
        };
        visit(trie);

        out << " )";
        if (!trie.is_end_of_string) {
            out << "?";
        }
        out << " [\"] space";
        return out.str();
    }

    std::string _resolve_ref(const std::string & ref) {
        auto it = ref.find('#');
        std::string ref_fragment = it != std::string::npos ? ref.substr(it + 1) : ref;
        static const std::regex nonalphanumeric_regex(R"([^a-zA-Z0-9-]+)");
        std::string ref_name = "ref" + std::regex_replace(ref_fragment, nonalphanumeric_regex, "-");
        if (_rules.find(ref_name) == _rules.end() && _refs_being_resolved.find(ref) == _refs_being_resolved.end()) {
            _refs_being_resolved.insert(ref);
            json resolved = _refs[ref];
            ref_name = visit(resolved, ref_name);
            _refs_being_resolved.erase(ref);
        }
        return ref_name;
    }

    std::string _build_object_rule(
        const std::vector<std::pair<std::string, json>> & properties,
        const std::unordered_set<std::string> & required,
        const std::string & name,
        const json & additional_properties,
        int min_props = 0,
        int max_props = std::numeric_limits<int>::max())
    {
        std::vector<std::string> required_props;
        std::vector<std::string> optional_props;
        std::unordered_map<std::string, std::string> prop_kv_rule_names;
        std::vector<std::string> prop_names;
        for (const auto & kv : properties) {
            const auto &prop_name = kv.first;
            const auto &prop_schema = kv.second;

            std::string prop_rule_name = visit(prop_schema, name + (name.empty() ? "" : "-") + prop_name);
            prop_kv_rule_names[prop_name] = _add_rule(
                name + (name.empty() ? "" : "-") + prop_name + "-kv",
                format_literal(json(prop_name).dump()) + " space \":\" space " + prop_rule_name
            );
            if (required.find(prop_name) != required.end()) {
                required_props.push_back(prop_name);
            } else {
                optional_props.push_back(prop_name);
            }
            prop_names.push_back(prop_name);
        }
        if ((additional_properties.is_boolean() && additional_properties.get<bool>()) || additional_properties.is_object()) {
            std::string sub_name = name + (name.empty() ? "" : "-") + "additional";
            std::string value_rule =
                additional_properties.is_object() ? visit(additional_properties, sub_name + "-value")
                : _add_primitive("value", PRIMITIVE_RULES.at("value"));

            auto key_rule =
                prop_names.empty() ? _add_primitive("string", PRIMITIVE_RULES.at("string"))
                : _add_rule(sub_name + "-k", _not_strings(prop_names));
            std::string kv_rule = _add_rule(sub_name + "-kv", key_rule + " \":\" space " + value_rule);
            prop_kv_rule_names["*"] = kv_rule;
            optional_props.push_back("*");
        }

        std::string rule = "\"{\" space ";
        for (size_t i = 0; i < required_props.size(); i++) {
            if (i > 0) {
                rule += " \",\" space ";
            }
            rule += prop_kv_rule_names[required_props[i]];
        }

        // quillama.cpp: minProperties / maxProperties. The optional properties are emitted in
        // declaration order, each present or absent, then the additional properties ("*", last), so
        // the grammar can count: state (i, c) = optional property i is next and c properties have
        // been emitted so far. Without these keywords the rule below is unchanged.
        const bool counted = min_props > 0 || max_props != std::numeric_limits<int>::max();
        if (counted && (int) required_props.size() <= max_props) {
            const bool has_max = max_props != std::numeric_limits<int>::max();
            static const std::string FAIL = "\x01";
            std::map<std::pair<size_t, int>, std::string> memo;
            std::function<std::string(size_t, int)> rest = [&](size_t i, int c) -> std::string {
                // without a maximum, counts at or above the minimum behave alike
                const int key_c = has_max ? c : std::min(c, std::max(min_props, 1));
                const auto key = std::make_pair(i, key_c);
                auto it = memo.find(key);
                if (it != memo.end()) {
                    return it->second;
                }
                const std::string comma = c > 0 ? "\",\" space " : "";
                std::vector<std::string> alts;
                bool has_empty = false;
                if (i == optional_props.size()) {
                    has_empty = c >= min_props;
                } else if (optional_props[i] == "*") {
                    const std::string & kv = prop_kv_rule_names["*"];
                    const int lo = std::max(0, min_props - c);
                    const int hi = has_max ? max_props - c : std::numeric_limits<int>::max();
                    if (hi >= lo) {
                        const std::string rep = c > 0
                            ? build_repetition("( \",\" space " + kv + " )", lo, hi)
                            : build_repetition(kv, lo, hi, "\",\" space");
                        if (rep.empty()) {
                            has_empty = true;
                        } else {
                            alts.push_back(rep);
                        }
                    }
                } else {
                    const std::string skip = rest(i + 1, c);
                    if (skip == "") {
                        has_empty = true;
                    } else if (skip != FAIL) {
                        alts.push_back(skip);
                    }
                    if (c < max_props) {
                        const std::string next = rest(i + 1, c + 1);
                        if (next != FAIL) {
                            alts.push_back(comma + prop_kv_rule_names[optional_props[i]] + (next.empty() ? "" : " " + next));
                        }
                    }
                }
                std::string res;
                if (alts.empty()) {
                    res = has_empty ? "" : FAIL;
                } else {
                    res = _add_rule(name + (name.empty() ? "" : "-") + "props-" + std::to_string(i) + "-" + std::to_string(key_c),
                                    alts.size() == 1 && !has_empty ? alts[0] : "(" + string_join(alts, " | ") + ")" + (has_empty ? "?" : ""));
                }
                memo[key] = res;
                return res;
            };
            const std::string tail = rest(0, (int) required_props.size());
            if (tail != FAIL) {
                return rule + (tail.empty() ? "" : " " + tail) + " \"}\" space";
            }
            // unsatisfiable bounds: fall back to the unbounded object
        }

        if (!optional_props.empty()) {
            rule += " (";
            if (!required_props.empty()) {
                rule += " \",\" space ( ";
            }

            std::function<std::string(const std::vector<std::string> &, bool)> get_recursive_refs = [&](const std::vector<std::string> & ks, bool first_is_optional) {
                std::string res;
                if (ks.empty()) {
                    return res;
                }
                const std::string& k = ks[0];
                std::string kv_rule_name = prop_kv_rule_names[k];
                std::string comma_ref = "( \",\" space " + kv_rule_name + " )";
                if (first_is_optional) {
                    res = comma_ref + (k == "*" ? "*" : "?");
                } else {
                    res = kv_rule_name + (k == "*" ? " " + comma_ref + "*" : "");
                }
                if (ks.size() > 1) {
                    res += " " + _add_rule(
                        name + (name.empty() ? "" : "-") + k + "-rest",
                        get_recursive_refs(std::vector<std::string>(ks.begin() + 1, ks.end()), true)
                    );
                }
                return res;
            };

            for (size_t i = 0; i < optional_props.size(); i++) {
                if (i > 0) {
                    rule += " | ";
                }
                rule += get_recursive_refs(std::vector<std::string>(optional_props.begin() + i, optional_props.end()), false);
            }
            if (!required_props.empty()) {
                rule += " )";
            }
            rule += " )?";
        }

        rule += " \"}\" space";

        return rule;
    }

    std::string _add_primitive(const std::string & name, const BuiltinRule & rule) {
        auto n = _add_rule(name, rule.content);
        for (const auto & dep : rule.deps) {
            BuiltinRule dep_rule;
            auto it = PRIMITIVE_RULES.find(dep);
            if (it == PRIMITIVE_RULES.end()) {
                it = STRING_FORMAT_RULES.find(dep);
                if (it == STRING_FORMAT_RULES.end()) {
                    _errors.push_back("Rule " + dep + " not known");
                    continue;
                }
            }
            if (_rules.find(dep) == _rules.end()) {
                _add_primitive(dep, it->second);
            }
        }
        return n;
    }

public:
    common_schema_converter(
        const std::function<json(const std::string &)> & fetch_json,
        bool dotall)
          : _fetch_json(fetch_json), _dotall(dotall)
    {
        _rules["space"] = SPACE_RULE;
    }

    void resolve_refs(json & schema, const std::string & url) {
        /*
        * Resolves all $ref fields in the given schema, fetching any remote schemas,
        * replacing each $ref with absolute reference URL and populates _refs with the
        * respective referenced (sub)schema dictionaries.
        */
        std::function<void(json &)> visit_refs = [&](json & n) {
            if (n.is_array()) {
                for (auto & x : n) {
                    visit_refs(x);
                }
            } else if (n.is_object()) {
                if (n.contains("$ref")) {
                    std::string ref = n["$ref"];
                    if (_refs.find(ref) == _refs.end()) {
                        json target;
                        if (ref.find("https://") == 0) {
                            std::string base_url = ref.substr(0, ref.find('#'));
                            auto it = _refs.find(base_url);
                            if (it != _refs.end()) {
                                target = it->second;
                            } else {
                                // Fetch the referenced schema and resolve its refs
                                auto referenced = _fetch_json(ref);
                                resolve_refs(referenced, base_url);
                                _refs[base_url] = referenced;
                            }
                            if (ref.find('#') == std::string::npos || ref.substr(ref.find('#') + 1).empty()) {
                                return;
                            }
                        } else if (ref.find("#/") == 0) {
                            target = schema;
                            n["$ref"] = url + ref;
                            ref = url + ref;
                        } else {
                            _errors.push_back("Unsupported ref: " + ref);
                            return;
                        }
                        std::string pointer = ref.substr(ref.find('#') + 1);
                        std::vector<std::string> tokens = string_split(pointer, "/");
                        for (size_t i = 1; i < tokens.size(); ++i) {
                            const std::string& sel = tokens[i];
                            if (target.is_object() && target.contains(sel)) {
                                target = target[sel];
                            } else if (target.is_array()) {
                                size_t sel_index;
                                try {
                                    sel_index = std::stoull(sel);
                                } catch (const std::invalid_argument & e) {
                                    sel_index = target.size();
                                }
                                if (sel_index >= target.size()) {
                                    _errors.push_back("Error resolving ref " + ref + ": " + sel + " not in " + target.dump());
                                    return;
                                }
                                target = target[sel_index];
                            } else {
                                _errors.push_back("Error resolving ref " + ref + ": " + sel + " not in " + target.dump());
                                return;
                            }
                        }
                        _refs[ref] = target;
                    }
                } else {
                    for (const auto & kv : n.items()) {
                        visit_refs(kv.value());
                    }
                }
            }
        };

        visit_refs(schema);
    }

    static std::string _generate_constant_rule(const json & value) {
        return format_literal(value.dump());
    }

    std::string visit(const json & schema, const std::string & name) {
        json schema_type = schema.contains("type") ? schema["type"] : json();
        std::string schema_format = schema.contains("format") ? schema["format"].get<std::string>() : "";
        std::string rule_name = is_reserved_name(name) ? name + "-" : name.empty() ? "root" : name;

        if (schema.contains("$ref")) {
            return _add_rule(rule_name, _resolve_ref(schema["$ref"]));
        }
        if (schema.contains("oneOf") || schema.contains("anyOf")) {
            std::vector<json> alt_schemas = schema.contains("oneOf") ? schema["oneOf"].get<std::vector<json>>() : schema["anyOf"].get<std::vector<json>>();
            return _add_rule(rule_name, _generate_union_rule(name, alt_schemas));
        }
        if (schema_type.is_array()) {
            std::vector<json> schema_types;
            for (const auto & t : schema_type) {
                json schema_copy(schema);
                schema_copy["type"] = t;
                schema_types.push_back(schema_copy);
            }
            return _add_rule(rule_name, _generate_union_rule(name, schema_types));
        }
        if (schema.contains("const")) {
            return _add_rule(rule_name, _generate_constant_rule(schema["const"]) + " space");
        }
        if (schema.contains("enum")) {
            std::vector<std::string> enum_values;
            for (const auto & v : schema["enum"]) {
                enum_values.push_back(_generate_constant_rule(v));
            }
            return _add_rule(rule_name, "(" + string_join(enum_values, " | ") + ") space");
        }
        const bool has_prop_count = schema.contains("minProperties") || schema.contains("maxProperties");
        if ((schema_type.is_null() || schema_type == "object")
                && (schema.contains("properties") ||
                    (schema.contains("additionalProperties") && schema["additionalProperties"] != true) ||
                    (schema_type == "object" && has_prop_count))) {
            std::unordered_set<std::string> required;
            if (schema.contains("required") && schema["required"].is_array()) {
                for (const auto & item : schema["required"]) {
                    if (item.is_string()) {
                        required.insert(item.get<std::string>());
                    }
                }
            }
            std::vector<std::pair<std::string, json>> properties;
            if (schema.contains("properties")) {
                for (const auto & prop : schema["properties"].items()) {
                    properties.emplace_back(prop.key(), prop.value());
                }
            }
            const auto count = [&](const char * k, int dflt) {
                return schema.contains(k) && schema[k].is_number_integer() ? std::max(0, schema[k].get<int>()) : dflt;
            };
            // a bare {"type": "object", "maxProperties": N} is a map of any values
            json additional = schema.contains("additionalProperties") ? schema["additionalProperties"]
                            : (!schema.contains("properties") ? json(true) : json());
            return _add_rule(rule_name,
                _build_object_rule(
                    properties, required, name, additional,
                    count("minProperties", 0), count("maxProperties", std::numeric_limits<int>::max())));
        }
        if ((schema_type.is_null() || schema_type == "object" || schema_type == "string") && schema.contains("allOf")) {
            std::unordered_set<std::string> required;
            std::vector<std::pair<std::string, json>> properties;
            std::map<std::string, size_t> enum_values;
            const std::string& hybrid_name = name;
            std::function<void(const json &, bool)> add_component = [&](const json & comp_schema, bool is_required) {
                if (comp_schema.contains("$ref")) {
                    add_component(_refs[comp_schema["$ref"]], is_required);
                } else if (comp_schema.contains("properties")) {
                    for (const auto & prop : comp_schema["properties"].items()) {
                        properties.emplace_back(prop.key(), prop.value());
                        if (is_required) {
                            required.insert(prop.key());
                        }
                    }
                } else if (comp_schema.contains("enum")) {
                    for (const auto & v : comp_schema["enum"]) {
                        const auto rule = _generate_constant_rule(v);
                        if (enum_values.find(rule) == enum_values.end()) {
                            enum_values[rule] = 0;
                        }
                        enum_values[rule] += 1;
                    }
                } else {
                  // todo warning
                }
            };
            for (const auto & t : schema["allOf"]) {
                if (t.contains("anyOf")) {
                    for (const auto & tt : t["anyOf"]) {
                        add_component(tt, false);
                    }
                } else {
                    add_component(t, true);
                }
            }
            if (!enum_values.empty()) {
                std::vector<std::string> enum_intersection;
                for (const auto & p : enum_values) {
                    if (p.second == schema["allOf"].size()) {
                        enum_intersection.push_back(p.first);
                    }
                }
                if (!enum_intersection.empty()) {
                    return _add_rule(rule_name, "(" + string_join(enum_intersection, " | ") + ") space");
                }
            }
            return _add_rule(rule_name, _build_object_rule(properties, required, hybrid_name, json()));
        }
        if ((schema_type.is_null() || schema_type == "array") && (schema.contains("items") || schema.contains("prefixItems"))) {
            json items = schema.contains("items") ? schema["items"] : schema["prefixItems"];
            if (items.is_array()) {
                std::string rule = "\"[\" space ";
                for (size_t i = 0; i < items.size(); i++) {
                    if (i > 0) {
                        rule += " \",\" space ";
                    }
                    rule += visit(items[i], name + (name.empty() ? "" : "-") + "tuple-" + std::to_string(i));
                }
                rule += " \"]\" space";
                return _add_rule(rule_name, rule);
            }
            std::string item_rule_name = visit(items, name + (name.empty() ? "" : "-") + "item");
            int min_items = schema.contains("minItems") ? schema["minItems"].get<int>() : 0;
            json max_items_json = schema.contains("maxItems") ? schema["maxItems"] : json();
            int max_items = max_items_json.is_number_integer() ? max_items_json.get<int>() : std::numeric_limits<int>::max();

            return _add_rule(rule_name, "\"[\" space " + build_repetition(item_rule_name, min_items, max_items, "\",\" space") + " \"]\" space");
        }
        if ((schema_type.is_null() || schema_type == "string") && schema.contains("pattern")) {
            return _visit_pattern(schema["pattern"], rule_name);
        }
        if ((schema_type.is_null() || schema_type == "string") && std::regex_match(schema_format, std::regex("^uuid[1-5]?$"))) {
            return _add_primitive(rule_name == "root" ? "root" : schema_format, PRIMITIVE_RULES.at("uuid"));
        }
        if ((schema_type.is_null() || schema_type == "string") && STRING_FORMAT_RULES.find(schema_format + "-string") != STRING_FORMAT_RULES.end()) {
            auto prim_name = schema_format + "-string";
            return _add_rule(rule_name, _add_primitive(prim_name, STRING_FORMAT_RULES.at(prim_name)));
        }
        if (schema_type == "string" && (schema.contains("minLength") || schema.contains("maxLength"))) {
            std::string char_rule = _add_primitive("char", PRIMITIVE_RULES.at("char"));
            int min_len = schema.contains("minLength") ? schema["minLength"].get<int>() : 0;
            int max_len = schema.contains("maxLength") ? schema["maxLength"].get<int>() : std::numeric_limits<int>::max();
            return _add_rule(rule_name, "\"\\\"\" " + build_repetition(char_rule, min_len, max_len) + " \"\\\"\" space");
        }
        if (schema_type == "integer" && (schema.contains("minimum") || schema.contains("exclusiveMinimum") || schema.contains("maximum") || schema.contains("exclusiveMaximum"))) {
            int64_t min_value = std::numeric_limits<int64_t>::min();
            int64_t max_value = std::numeric_limits<int64_t>::max();
            bool has_lo, lo_ex, has_hi, hi_ex;
            quill_dec lo, hi;
            if (quill_read_bounds(schema, has_lo, lo, lo_ex, has_hi, hi, hi_ex)) {
                // quillama.cpp: fractional bounds round inward (minimum 1.2 -> 2, maximum -1.5 -> -2)
                if (has_lo) {
                    min_value = quill_dec_ceil(lo, lo_ex);
                }
                if (has_hi) {
                    max_value = quill_dec_floor(hi, hi_ex);
                }
            } else {
                _warnings.push_back("integer bounds beyond 15 digits were not enforced: " + schema.dump());
            }
            if (min_value > max_value) {
                _errors.push_back("no integer satisfies the bounds of " + schema.dump());
                return "";
            }
            std::stringstream out;
            out << "(";
            build_min_max_int(min_value, max_value, out);
            out << ") space";
            return _add_rule(rule_name, out.str());
        }
        if (schema.empty() || schema_type == "object") {
            return _add_rule(rule_name, _add_primitive("object", PRIMITIVE_RULES.at("object")));
        }
        if (schema_type.is_null() && schema.is_object()) {
            // No type constraint and no recognized structural keywords (e.g. {"description": "..."}).
            // Per JSON Schema semantics this is equivalent to {} and accepts any value.
            return _add_rule(rule_name, _add_primitive("value", PRIMITIVE_RULES.at("value")));
        }
        if (!schema_type.is_string() || PRIMITIVE_RULES.find(schema_type.get<std::string>()) == PRIMITIVE_RULES.end()) {
            _errors.push_back("Unrecognized schema: " + schema.dump());
            return "";
        }
        // quillama.cpp: "number" bounds, exact for any decimal bound (the value is written without an
        // exponent)
        if (schema_type == "number" && (schema.contains("minimum") || schema.contains("exclusiveMinimum") || schema.contains("maximum") || schema.contains("exclusiveMaximum"))) {
            bool has_lo, lo_ex, has_hi, hi_ex;
            quill_dec lo, hi;
            if (quill_read_bounds(schema, has_lo, lo, lo_ex, has_hi, hi, hi_ex) && (has_lo || has_hi)) {
                if (has_lo && has_hi) {
                    const int c = quill_dec_cmp(lo, hi);
                    if (c > 0 || (c == 0 && (lo_ex || hi_ex))) {
                        _errors.push_back("no number satisfies the bounds of " + schema.dump());
                        return "";
                    }
                }
                const std::string body = quill_number_range(has_lo ? &lo : nullptr, lo_ex, has_hi ? &hi : nullptr, hi_ex);
                if (!body.empty()) {
                    return _add_rule(rule_name, "(" + body + ") space");
                }
            }
        }
        // a "number" whose bounds could not be represented exactly: minimum / exclusiveMinimum >= 0
        // still forbids a leading '-'
        if (schema_type == "number") {
            auto non_negative = [&](const char * key) {
                return schema.contains(key) && schema[key].is_number() && schema[key].get<double>() >= 0.0;
            };
            if (non_negative("minimum") || non_negative("exclusiveMinimum")) {
                return _add_primitive(rule_name == "root" ? "root" : "nonneg-number",
                    BuiltinRule{"integral-part (\".\" decimal-part)? ([eE] [-+]? integral-part)? space", {"integral-part", "decimal-part"}});
            }
        }
        return _add_primitive(rule_name == "root" ? "root" : schema_type.get<std::string>(), PRIMITIVE_RULES.at(schema_type.get<std::string>()));
    }

    void check_errors() {
        if (!_errors.empty()) {
            throw std::invalid_argument("JSON schema conversion failed:\n" + string_join(_errors, "\n"));
        }
        if (!_warnings.empty()) {
            fprintf(stderr, "WARNING: JSON schema conversion was incomplete: %s\n", string_join(_warnings, "; ").c_str());
        }
    }

    std::string format_grammar() {
        std::stringstream ss;
        for (const auto & kv : _rules) {
            ss << kv.first << " ::= " << kv.second << '\n';
        }
        return ss.str();
    }
};

// common_schema_info implementation (pimpl)

common_schema_info::common_schema_info()
    : impl_(std::make_unique<common_schema_converter>(
        [](const std::string &) { return json(); },
        false)) {}

common_schema_info::~common_schema_info() = default;

common_schema_info::common_schema_info(common_schema_info &&) noexcept = default;
common_schema_info & common_schema_info::operator=(common_schema_info &&) noexcept = default;

void common_schema_info::resolve_refs(nlohmann::ordered_json & schema) {
    impl_->resolve_refs(schema, "");
}

// Determines if a JSON schema can resolve to a string type through any path.
// Some models emit raw string values rather than JSON-encoded strings for string parameters.
// If any branch of the schema (via oneOf, anyOf, $ref, etc.) permits a string, this returns
// true, allowing callers to handle the value as a raw string for simplicity.
bool common_schema_info::resolves_to_string(const nlohmann::ordered_json & schema) {
    std::unordered_set<std::string> visited_refs;

    std::function<bool(const json &)> check = [&](const json & s) -> bool {
        if (!s.is_object()) {
            return false;
        }

        // Handle $ref
        if (s.contains("$ref")) {
            const std::string & ref = s["$ref"];
            if (visited_refs.find(ref) != visited_refs.end()) {
                // Circular reference, assume not a string to be safe
                return false;
            }
            visited_refs.insert(ref);
            auto it = impl_->_refs.find(ref);
            if (it != impl_->_refs.end()) {
                return check(it->second);
            }
            return false;
        }

        // Check type field
        if (s.contains("type")) {
            const json & schema_type = s["type"];
            if (schema_type.is_string()) {
                if (schema_type == "string") {
                    return true;
                }
            } else if (schema_type.is_array()) {
                // Type can be an array like ["string", "null"]
                for (const auto & t : schema_type) {
                    if (t == "string") {
                        return true;
                    }
                }
            }
        }

        // Check oneOf/anyOf - if any alternative can be a string
        if (s.contains("oneOf")) {
            for (const auto & alt : s["oneOf"]) {
                if (check(alt)) {
                    return true;
                }
            }
        }
        if (s.contains("anyOf")) {
            for (const auto & alt : s["anyOf"]) {
                if (check(alt)) {
                    return true;
                }
            }
        }

        // Check allOf - all components must be compatible with string type
        if (s.contains("allOf")) {
            bool all_string = true;
            for (const auto & component : s["allOf"]) {
                if (!check(component)) {
                    all_string = false;
                    break;
                }
            }
            if (all_string) {
                return true;
            }
        }

        // Check const - if the constant value is a string
        if (s.contains("const")) {
            if (s["const"].is_string()) {
                return true;
            }
        }

        // Check enum - if any enum value is a string
        if (s.contains("enum")) {
            for (const auto & val : s["enum"]) {
                if (val.is_string()) {
                    return true;
                }
            }
        }

        // String-specific keywords imply string type
        if (s.contains("pattern") || s.contains("minLength") || s.contains("maxLength")) {
            return true;
        }

        // Check format - many formats imply string
        if (s.contains("format")) {
            const std::string & fmt = s["format"];
            if (fmt == "date" || fmt == "time" || fmt == "date-time" ||
                fmt == "uri" || fmt == "email" || fmt == "hostname" ||
                fmt == "ipv4" || fmt == "ipv6" || fmt == "uuid" ||
                fmt.find("uuid") == 0) {
                return true;
            }
        }

        return false;
    };

    return check(schema);
}

std::string json_schema_to_grammar(const json & schema, bool force_gbnf) {
#ifdef LLAMA_USE_LLGUIDANCE
    if (!force_gbnf) {
        return "%llguidance {}\nstart: %json " + schema.dump();
    }
#else
    (void)force_gbnf;
#endif // LLAMA_USE_LLGUIDANCE
    return build_grammar([&](const common_grammar_builder & callbacks) {
        auto copy = schema;
        callbacks.resolve_refs(copy);
        callbacks.add_schema("", copy);
    });
}

std::string build_grammar(const std::function<void(const common_grammar_builder &)> & cb, const common_grammar_options & options) {
    common_schema_converter converter([&](const std::string &) { return json(); }, options.dotall);
    common_grammar_builder builder {
        /* .add_rule = */ [&](const std::string & name, const std::string & rule) {
            return converter._add_rule(name, rule);
        },
        /* .add_schema = */ [&](const std::string & name, const nlohmann::ordered_json & schema) {
            return converter.visit(schema, name == "root" ? "" : name);
        },
        /* .resolve_refs = */ [&](nlohmann::ordered_json & schema) {
            converter.resolve_refs(schema, "");
        }
    };
    cb(builder);
    converter.check_errors();
    return converter.format_grammar();
}

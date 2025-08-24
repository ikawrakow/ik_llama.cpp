#pragma once

#include <locale>
#include <string>
#include <algorithm>

#include "../errors.h"

namespace sqlite {
	namespace utility {
		inline std::string utf16_to_utf8(u16str_ref input) {
			struct : std::codecvt<char16_t, char, std::mbstate_t> {
			} codecvt;
			std::mbstate_t state{};
			std::string result((std::max)(input.size() * 3 / 2, std::size_t(4)), '\0');
			const char16_t *remaining_input = input.data();
			std::size_t produced_output = 0;
			while(true) {
				char *used_output;
				switch(codecvt.out(state, remaining_input, input.data() + input.size(),
				                   remaining_input, &result[produced_output],
				                   &result[result.size() - 1] + 1, used_output)) {
				case std::codecvt_base::ok:
					result.resize(used_output - result.data());
					return result;
				case std::codecvt_base::noconv:
					// This should be unreachable
				case std::codecvt_base::error:
					throw errors::invalid_utf16("Invalid UTF-16 input", "");
				case std::codecvt_base::partial:
					if(used_output == result.data() + produced_output)
						throw errors::invalid_utf16("Unexpected end of input", "");
					produced_output = used_output - result.data();
					result.resize(
							result.size()
							+ (std::max)((input.data() + input.size() - remaining_input) * 3 / 2,
							           std::ptrdiff_t(4)));
				}
			}
		}
	} // namespace utility
} // namespace sqlite

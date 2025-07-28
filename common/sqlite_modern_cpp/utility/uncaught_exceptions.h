#pragma once

#include <cassert>
#include <exception>
#include <iostream>

// Consider that std::uncaught_exceptions is available if explicitly indicated
// by the standard library, if compiler advertises full C++17 support or, as a
// special case, for MSVS 2015+ (which doesn't define __cplusplus correctly by
// default as of 2017.7 version and couldn't do it at all until it).
#ifndef MODERN_SQLITE_UNCAUGHT_EXCEPTIONS_SUPPORT
#ifdef __cpp_lib_uncaught_exceptions
#define MODERN_SQLITE_UNCAUGHT_EXCEPTIONS_SUPPORT
#elif __cplusplus >= 201703L
#define MODERN_SQLITE_UNCAUGHT_EXCEPTIONS_SUPPORT
#elif defined(_MSC_VER) && _MSC_VER >= 1900
#define MODERN_SQLITE_UNCAUGHT_EXCEPTIONS_SUPPORT
#endif
#endif

namespace sqlite {
	namespace utility {
#ifdef MODERN_SQLITE_UNCAUGHT_EXCEPTIONS_SUPPORT
		class UncaughtExceptionDetector {
		public:
			operator bool() {
				return count != std::uncaught_exceptions();
			}
		private:
			int count = std::uncaught_exceptions();
		};
#else
		class UncaughtExceptionDetector {
		public:
			operator bool() {
				return std::uncaught_exception();
			}
		};
#endif
	}
}

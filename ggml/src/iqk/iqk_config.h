//
// Copyright (C) 2024-2025 Iwan Kawrakow
// MIT license
// SPDX-License-Identifier: MIT
//

#pragma once

#if defined IQK_IMPLEMENT
#undef IQK_IMPLEMENT
#endif

#if defined __AVX2__ || defined __ARM_FEATURE_DOTPROD
#define IQK_IMPLEMENT
#endif

#ifdef GGML_SHARED
#    if defined(_WIN32) && !defined(__MINGW32__)
#        ifdef GGML_BUILD
#            define IQK_API __declspec(dllexport)
#        else
#            define IQK_API __declspec(dllimport)
#        endif
#    else
#        define IQK_API __attribute__ ((visibility ("default")))
#    endif
#else
#    define IQK_API
#endif

#ifdef _MSC_VER
#define IQK_NOINLINE __declspec(noinline)
#define IQK_ALWAYS_INLINE inline
#if !defined __x86_64__ && defined _M_X64
#define __x86_64__
#endif
#else
#define IQK_NOINLINE __attribute__((__noinline__))
#define IQK_ALWAYS_INLINE __attribute__((__always_inline__))
#endif

#if defined __x86_64__
#if defined HAVE_FANCY_SIMD
    #undef HAVE_FANCY_SIMD
#endif
#if defined(__AVX512F__) && defined(__AVX512VNNI__) && defined(__AVX512VL__) && defined(__AVX512BW__) && defined(__AVX512DQ__)
    #define HAVE_FANCY_SIMD
#endif
#endif


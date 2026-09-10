include(CheckCSourceRuns)

set(AVX_CODE "
    #include <immintrin.h>
    int main()
    {
        __m256 a;
        a = _mm256_set1_ps(0);
        return 0;
    }
")

set(AVX512_CODE "
    #include <immintrin.h>
    int main()
    {
        __m512i a = _mm512_set_epi8(0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0);
        __m512i b = a;
        __mmask64 equality_mask = _mm512_cmp_epi8_mask(a, b, _MM_CMPINT_EQ);
        return 0;
    }
")

set(AVX512VNNI_CODE "
    #include <immintrin.h>
    int main()
    {
        __m512i acc = _mm512_setzero_si512();
        __m512i u   = _mm512_set1_epi8(1);
        __m512i s   = _mm512_set1_epi8(1);
        acc = _mm512_dpbusd_epi32(acc, u, s);
        return _mm512_reduce_add_epi32(acc) == 64 ? 0 : 1;
    }
")

set(AVX512VBMI_CODE "
    #include <immintrin.h>
    int main()
    {
        __m512i a   = _mm512_set1_epi8(1);
        __m512i idx = _mm512_setzero_si512();
        __m512i r   = _mm512_permutexvar_epi8(idx, a);
        return _mm512_reduce_add_epi32(r) == 16 * 0x01010101 ? 0 : 1;
    }
")

set(AVX512BF16_CODE "
    #include <immintrin.h>
    int main()
    {
        __m512   acc = _mm512_setzero_ps();
        __m512   a   = _mm512_set1_ps(1.0f);
        __m512bh b   = _mm512_cvtne2ps_pbh(a, a);
        acc = _mm512_dpbf16_ps(acc, b, b);
        return _mm512_reduce_add_ps(acc) == 32.0f ? 0 : 1;
    }
")

set(AVX2_CODE "
    #include <immintrin.h>
    int main()
    {
        __m256i a = {0};
        a = _mm256_abs_epi16(a);
        __m256i x;
        _mm256_extract_epi64(x, 0); // we rely on this in our AVX2 code
        return 0;
    }
")

set(FMA_CODE "
    #include <immintrin.h>
    int main()
    {
        __m256 acc = _mm256_setzero_ps();
        const __m256 d = _mm256_setzero_ps();
        const __m256 p = _mm256_setzero_ps();
        acc = _mm256_fmadd_ps( d, p, acc );
        return 0;
    }
")

macro(check_sse type flags)
    set(__FLAG_I 1)
    set(CMAKE_REQUIRED_FLAGS_SAVE ${CMAKE_REQUIRED_FLAGS})
    foreach (__FLAG ${flags})
        if (NOT ${type}_FOUND)
            set(CMAKE_REQUIRED_FLAGS ${__FLAG})
            check_c_source_runs("${${type}_CODE}" HAS_${type}_${__FLAG_I})
            if (HAS_${type}_${__FLAG_I})
                set(${type}_FOUND TRUE CACHE BOOL "${type} support")
                set(${type}_FLAGS "${__FLAG}" CACHE STRING "${type} flags")
            endif()
            math(EXPR __FLAG_I "${__FLAG_I}+1")
        endif()
    endforeach()
    set(CMAKE_REQUIRED_FLAGS ${CMAKE_REQUIRED_FLAGS_SAVE})

    if (NOT ${type}_FOUND)
        set(${type}_FOUND FALSE CACHE BOOL "${type} support")
        set(${type}_FLAGS "" CACHE STRING "${type} flags")
    endif()

    mark_as_advanced(${type}_FOUND ${type}_FLAGS)
endmacro()

# flags are for MSVC only!
check_sse("AVX" " ;/arch:AVX")
if (NOT ${AVX_FOUND})
    set(GGML_AVX OFF)
else()
    set(GGML_AVX ON)
endif()

check_sse("AVX2" " ;/arch:AVX2")
check_sse("FMA" " ;/arch:AVX2")
if ((NOT ${AVX2_FOUND}) OR (NOT ${FMA_FOUND}))
    set(GGML_AVX2 OFF)
else()
    set(GGML_AVX2 ON)
endif()

check_sse("AVX512" " ;/arch:AVX512")
if (NOT ${AVX512_FOUND})
    set(GGML_AVX512 OFF)
else()
    set(GGML_AVX512 ON)
endif()

# MSVC has no /arch: flag for the individual AVX-512 extensions and does not
# define their macros, so ggml/src/CMakeLists.txt sets them manually from these
# options. Probe each extension the same way the base sets are probed: the test
# program is compiled *and run*, so a CPU lacking the extension fails it.
#
# Each probe feeds the result of the instruction under test into the exit code.
# A probe that discards its result can be optimised away entirely, and then it
# passes because nothing is left to execute rather than because the CPU has the
# extension. GGML_NATIVE is a detection path, so a probe that fails here turns
# the option off even if it was requested on the command line, the same way the
# base AVX-512 check does.
if (GGML_AVX512)
    check_sse("AVX512VNNI" " ;/arch:AVX512")
    if (NOT ${AVX512VNNI_FOUND})
        set(GGML_AVX512_VNNI OFF)
    else()
        set(GGML_AVX512_VNNI ON)
    endif()

    check_sse("AVX512VBMI" " ;/arch:AVX512")
    if (NOT ${AVX512VBMI_FOUND})
        set(GGML_AVX512_VBMI OFF)
    else()
        set(GGML_AVX512_VBMI ON)
    endif()

    check_sse("AVX512BF16" " ;/arch:AVX512")
    if (NOT ${AVX512BF16_FOUND})
        set(GGML_AVX512_BF16 OFF)
    else()
        set(GGML_AVX512_BF16 ON)
    endif()
endif()

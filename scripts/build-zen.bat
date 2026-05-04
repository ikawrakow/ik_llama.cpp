@echo off
REM CPU-only build helper for AVX-512-capable CPUs (AMD Zen4 / Intel
REM Sapphire Rapids+) on Windows + MSVC. Enables the IQK GEMM kernels
REM gated by HAVE_FANCY_SIMD (see docs\build.md "CPU build flags for AVX-512").
REM
REM Run from a Visual Studio "x64 Native Tools Command Prompt" so that
REM cl.exe and the rest of the MSVC toolchain are on PATH.
REM
REM Usage:
REM   scripts\build-zen.bat [build-dir]
REM
REM Default build directory is "build".

setlocal

if "%~1"=="" (set BUILD_DIR=build) else (set BUILD_DIR=%~1)

cmake -B "%BUILD_DIR%" -G "NMake Makefiles" ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DGGML_NATIVE=ON ^
    -DGGML_AVX512=ON ^
    -DGGML_AVX512_VBMI=ON ^
    -DGGML_AVX512_VNNI=ON ^
    -DGGML_AVX512_BF16=ON
if errorlevel 1 exit /b 1

cmake --build "%BUILD_DIR%" --config Release

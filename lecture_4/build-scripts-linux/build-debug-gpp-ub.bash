#!/usr/bin/env bash
set -euo pipefail

cmake -S . -B build/debug-gcc-asan-ubsan -G Ninja \
    -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_CXX_COMPILER=g++ \
    -DENABLE_SANITIZERS=ON

cmake --build build/debug-gcc-asan-ubsan --verbose

ASAN_OPTIONS=detect_leaks=1:halt_on_error=1:abort_on_error=1 \
UBSAN_OPTIONS=print_stacktrace=1:halt_on_error=1 \
./build/debug-gcc-asan-ubsan/myapp


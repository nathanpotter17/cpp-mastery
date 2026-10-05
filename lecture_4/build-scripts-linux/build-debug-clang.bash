#!/usr/bin/env bash
set -euo pipefail

cmake -S . -B build/debug-clang -G Ninja \
    -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DCMAKE_EXE_LINKER_FLAGS="-fuse-ld=lld-21"

cmake --build build/debug-clang --verbose

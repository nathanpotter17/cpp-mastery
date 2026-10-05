#!/usr/bin/env bash
set -euo pipefail

cmake -S . -B build/release-clang -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DCMAKE_EXE_LINKER_FLAGS="-fuse-ld=lld-21"

cmake --build build/release-clang

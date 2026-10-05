#!/usr/bin/env bash
set -euo pipefail

cmake -S . -B build/release-gcc-full -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_COMPILER=g++ \
    -DCMAKE_CXX_FLAGS_RELEASE="-O3 -DNDEBUG"

cmake --build build/release-gcc-full --verbose

#!/usr/bin/env bash
set -euo pipefail

cmake -S . -B build/debug-gpp -G Ninja \
    -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_CXX_COMPILER=g++-14

cmake --build build/debug-gpp --verbose

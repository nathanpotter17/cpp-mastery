#!/usr/bin/env bash
set -euo pipefail

cmake -S . -B build/debug-gpp -G Ninja \
    -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_CXX_COMPILER=g++

cmake --build build/debug-gpp --verbose

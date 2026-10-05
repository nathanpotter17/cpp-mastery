#!/usr/bin/env bash
set -euo pipefail

cmake -S . -B build/release-gpp -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_COMPILER=g++

cmake --build build/release-gpp

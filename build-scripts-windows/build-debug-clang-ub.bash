#!/usr/bin/env bash
set -euo pipefail

LECTURE="${1:?Usage: $0 <lecture_N>}"

if [[ ! -f "$LECTURE/main.cpp" ]]; then
    echo "No such lecture: $LECTURE/main.cpp" >&2
    exit 1
fi

BUILD_DIR="$LECTURE/build/debug-clang-asan-ubsan"

cmake -S . -B "$BUILD_DIR" -G Ninja \
    -DLECTURE="$LECTURE" \
    -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_CXX_COMPILER=clang++-17 \
    -DENABLE_SANITIZERS=ON

cmake --build "$BUILD_DIR" --verbose

ASAN_OPTIONS=detect_leaks=1:halt_on_error=1:abort_on_error=1 \
UBSAN_OPTIONS=print_stacktrace=1:halt_on_error=1 \
"./$BUILD_DIR/$LECTURE"

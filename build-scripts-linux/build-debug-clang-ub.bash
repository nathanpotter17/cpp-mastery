#!/usr/bin/env bash
set -euo pipefail

LECTURE="${1:?Usage: $0 <lecture_N>}"

if [[ ! -f "$LECTURE/main.cpp" && ! -f "$LECTURE/src/main.cpp" ]]; then
    echo "No such lecture: $LECTURE/main.cpp or $LECTURE/src/main.cpp" >&2
    exit 1
fi

BUILD_DIR="$LECTURE/build/debug-clang-asan-ubsan"

cmake -S . -B "$BUILD_DIR" -G Ninja \
    -DLECTURE="$LECTURE" \
    -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DENABLE_SANITIZERS=ON

cmake --build "$BUILD_DIR" --verbose

# A lecture can list leaks to ignore (e.g. inside system libraries) in lsan.supp.
LSAN_OPTIONS=""
if [[ -f "$LECTURE/lsan.supp" ]]; then
    LSAN_OPTIONS="suppressions=$LECTURE/lsan.supp"
fi
export LSAN_OPTIONS

ASAN_OPTIONS=detect_leaks=1:halt_on_error=1:abort_on_error=1 \
UBSAN_OPTIONS=print_stacktrace=1:halt_on_error=1 \
"./$BUILD_DIR/$LECTURE"

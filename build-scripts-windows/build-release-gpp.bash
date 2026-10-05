#!/usr/bin/env bash
set -euo pipefail

LECTURE="${1:?Usage: $0 <lecture_N>}"

if [[ ! -f "$LECTURE/main.cpp" && ! -f "$LECTURE/src/main.cpp" ]]; then
    echo "No such lecture: $LECTURE/main.cpp or $LECTURE/src/main.cpp" >&2
    exit 1
fi

BUILD_DIR="$LECTURE/build/release-gpp"

cmake -S . -B "$BUILD_DIR" -G Ninja \
    -DLECTURE="$LECTURE" \
    -DCMAKE_BUILD_TYPE=Release \
    -DSTATIC_RUNTIME=ON \
    -DCMAKE_C_COMPILER=gcc-14 \
    -DCMAKE_CXX_COMPILER=g++-14

cmake --build "$BUILD_DIR"

echo
echo "Build succeeded: $BUILD_DIR/$LECTURE"

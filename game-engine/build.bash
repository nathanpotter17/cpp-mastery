#!/usr/bin/env bash
set -euo pipefail

# Configures and builds game-engine into build/<config>/, and optionally runs it.

# CMake paths below are relative to this script's directory.
cd "$(dirname "$0")"

# --- Menu --------------------------------------------------------------------

echo "
Build                            Build & run
 1) Debug Clang                   7) Debug Clang
 2) Debug G++                     8) Debug G++
 3) Release Clang                 9) Release Clang
 4) Release G++                  10) Release G++
 5) Debug Clang + Asan & UbSan   11) Debug Clang + Asan & UbSan
 6) Debug G++ + Asan & UbSan     12) Debug G++ + Asan & UbSan
"
read -rp "Choose [1-12, Enter for 7]: " choice
choice="${choice:-7}"

if ! [[ "$choice" =~ ^[0-9]+$ ]] || (( choice < 1 || choice > 12 )); then
  echo "Invalid choice: $choice" >&2
  exit 1
fi

# 7-12 are 1-6 plus a run afterwards.
run=false
if (( choice > 6 )); then
  run=true
  choice=$(( choice - 6 ))
fi

# --- Configurations ----------------------------------------------------------

clang=(-DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_EXE_LINKER_FLAGS=-fuse-ld=lld)
gcc=(-DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++)

# Release is already -O3 -DNDEBUG, plus LTO from CMakeLists.txt.
case "$choice" in
  1) config=debug-clang;      flags=(-DCMAKE_BUILD_TYPE=Debug "${clang[@]}") ;;
  2) config=debug-gcc;        flags=(-DCMAKE_BUILD_TYPE=Debug "${gcc[@]}") ;;
  3) config=release-clang;    flags=(-DCMAKE_BUILD_TYPE=Release "${clang[@]}") ;;
  4) config=release-gcc;      flags=(-DCMAKE_BUILD_TYPE=Release "${gcc[@]}") ;;
  5) config=debug-clang-asan; flags=(-DCMAKE_BUILD_TYPE=Debug "${clang[@]}" -DENABLE_SANITIZERS=ON) ;;
  6) config=debug-gcc-asan;   flags=(-DCMAKE_BUILD_TYPE=Debug "${gcc[@]}" -DENABLE_SANITIZERS=ON) ;;
esac

# --- Build -------------------------------------------------------------------

build_dir="build/$config"

cmake -S . -B "$build_dir" -G Ninja "${flags[@]}"
cmake --build "$build_dir"

echo
echo "Build succeeded: game-engine/$build_dir/game-engine"

if [[ "$run" == false ]]; then
  exit 0
fi

# --- Run ---------------------------------------------------------------------

# The sanitizer options only affect sanitizer builds. lsan.supp lists leaks
# inside system libraries (libdecor's GTK plugin on Wayland) that aren't ours.
echo
LSAN_OPTIONS="suppressions=$PWD/lsan.supp" \
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1:abort_on_error=1 \
UBSAN_OPTIONS=print_stacktrace=1:halt_on_error=1 \
"./$build_dir/game-engine"

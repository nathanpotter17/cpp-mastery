#!/usr/bin/env bash
set -euo pipefail

# Configures and builds one lecture (or all of them) into lecture_N/build/<config>/,
# and optionally runs it.

# The build scripts use `-S .`, so always run from the repo root.
cd "$(dirname "$0")"

OS=$(uname -s)
TARGET_DIR=""

case "$OS" in
  Linux)
    echo "Linux"
    TARGET_DIR="build-scripts-linux"
    ;;
  Darwin)
    echo "macOS"
    echo "No macOS support"
    exit 1
    ;;
  CYGWIN*|MINGW32*|MSYS*|MINGW*)
    echo "Windows"
    TARGET_DIR="build-scripts-windows"
    ;;
  *)
    echo "Unknown OS: $OS"
    exit 1
    ;;
esac

# --- Lecture -----------------------------------------------------------------

mapfile -t lectures < <(
  for main in lecture_*/main.cpp lecture_*/src/main.cpp; do
    [[ -f "$main" ]] && echo "${main%%/*}"
  done | sort -uV
)

echo
echo "0) All lectures"
for i in "${!lectures[@]}"; do
  echo "$((i + 1))) ${lectures[$i]}"
done
echo

read -rp "Choose a lecture [0-${#lectures[@]}]: " lecture_choice

if [[ "$lecture_choice" == 0 ]]; then
  TARGET="all"
elif [[ "$lecture_choice" =~ ^[0-9]+$ ]] && (( lecture_choice >= 1 && lecture_choice <= ${#lectures[@]} )); then
  TARGET="${lectures[$((lecture_choice - 1))]}"
else
  echo "Invalid choice"
  exit 1
fi

# --- Configuration -----------------------------------------------------------

# Running needs a single lecture, so "All lectures" only offers the builds.
if [[ "$TARGET" == all ]]; then
  echo "
Build
 1) Debug Clang
 2) Debug G++
 3) Release Clang
 4) Release G++
 5) Debug Clang + Asan & UbSan
 6) Debug G++ + Asan & UbSan
"
  read -rp "Choose [1-6, Enter for 1]: " choice
  choice="${choice:-1}"
  max=6
else
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
  max=12
fi

if ! [[ "$choice" =~ ^[0-9]+$ ]] || (( choice < 1 || choice > max )); then
  echo "Invalid choice: $choice" >&2
  exit 1
fi

# 7-12 are 1-6 plus a run afterwards.
run=false
if (( choice > 6 )); then
  run=true
  choice=$(( choice - 6 ))
fi

# Each script builds into lecture_N/build/<config>/.
case "$choice" in
  1) SCRIPT="build-debug-clang.bash";    CONFIG="debug-clang" ;;
  2) SCRIPT="build-debug-gpp.bash";      CONFIG="debug-gpp" ;;
  3) SCRIPT="build-release-clang.bash";  CONFIG="release-clang" ;;
  4) SCRIPT="build-release-gpp.bash";    CONFIG="release-gpp" ;;
  5) SCRIPT="build-debug-clang-ub.bash"; CONFIG="debug-clang-asan-ubsan" ;;
  6) SCRIPT="build-debug-gpp-ub.bash";   CONFIG="debug-gcc-asan-ubsan" ;;
esac

# --- Build -------------------------------------------------------------------

if [[ "$TARGET" == all ]]; then
  for lecture in "${lectures[@]}"; do
    echo
    echo "=== $lecture ==="
    "./${TARGET_DIR}/${SCRIPT}" "$lecture"
  done
  exit 0
fi

"./${TARGET_DIR}/${SCRIPT}" "$TARGET"

if [[ "$run" == false ]]; then
  exit 0
fi

# --- Run ---------------------------------------------------------------------

# The sanitizer options only affect sanitizer builds. A lecture can list leaks
# to ignore (e.g. inside system libraries) in lsan.supp.
LSAN_OPTIONS=""
if [[ -f "$TARGET/lsan.supp" ]]; then
  LSAN_OPTIONS="suppressions=$PWD/$TARGET/lsan.supp"
fi

echo
LSAN_OPTIONS="$LSAN_OPTIONS" \
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1:abort_on_error=1 \
UBSAN_OPTIONS=print_stacktrace=1:halt_on_error=1 \
"./$TARGET/build/$CONFIG/$TARGET"

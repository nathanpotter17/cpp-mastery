#!/usr/bin/env bash
set -euo pipefail

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
elif [[ "$lecture_choice" =~ ^[0-9]+$ ]] && (( lecture_choice <= ${#lectures[@]} )); then
  TARGET="${lectures[$((lecture_choice - 1))]}"
else
  echo "Invalid choice"
  exit 1
fi

echo "
1) Debug Clang
2) Debug G++

3) Release Clang
4) Release G++
"

# Sanitizer builds run the program afterwards, so only offer them for one lecture.
if [[ "$TARGET" != all ]]; then
  echo "5) Debug Clang with Asan & UbSan
6) Debug G++ with Asan & UbSan
"
fi

echo "7) Full Release Clang
8) Full Release G++
"

read -rp "Choose a script [1-8]: " choice

case "$choice" in
  1) SCRIPT="build-debug-clang.bash" ;;
  2) SCRIPT="build-debug-gpp.bash" ;;
  3) SCRIPT="build-release-clang.bash" ;;
  4) SCRIPT="build-release-gpp.bash" ;;
  5) SCRIPT="build-debug-clang-ub.bash" ;;
  6) SCRIPT="build-debug-gpp-ub.bash" ;;
  7) SCRIPT="build-release-clang-full.bash" ;;
  8) SCRIPT="build-release-gpp-full.bash" ;;
  *) echo "Invalid choice"; exit 1 ;;
esac

if [[ "$TARGET" == all ]]; then
  if [[ "$choice" == 5 || "$choice" == 6 ]]; then
    echo "Sanitizer builds are only available for a single lecture"
    exit 1
  fi

  # Each lecture gets its own build tree under lecture_N/build/.
  for lecture in "${lectures[@]}"; do
    echo
    echo "=== $lecture ==="
    "./${TARGET_DIR}/${SCRIPT}" "$lecture"
  done
else
  "./${TARGET_DIR}/${SCRIPT}" "$TARGET"
fi

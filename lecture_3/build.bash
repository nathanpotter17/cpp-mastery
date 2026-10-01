#!/usr/bin/env bash

echo "
1) Debug Clang
2) Debug G++
3) Release Clang
4) Release G++
5) Debug Clang with Asan & UbSan
6) Debug G++ with Asan & UbSan
7) Full Release Clang
8) Full Release G++
"

read -rp "Choose a script [1-N]: " choice

case "$choice" in
  1) ./build-scripts/build-debug-clang.bash ;;
  2) ./build-scripts/build-debug-gpp.bash ;;
  3) ./build-scripts/build-release-clang.bash ;;
  4) ./build-scripts/build-release-gpp.bash ;;
  5) ./build-scripts/build-debug-clang-ub.bash ;;
  6) ./build-scripts/build-debug-gpp-ub.bash ;;
  7) ./build-scripts/build-release-clang-full.bash ;;
  8) ./build-scripts/build-release-gpp-full.bash ;;
  *) echo "Invalid choice" ;;
esac

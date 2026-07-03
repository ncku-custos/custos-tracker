#!/usr/bin/env bash
# Pre-commit gate: configure + build + test + format check.
set -euo pipefail
cd "$(dirname "$0")/.."

BUILD_DIR=${BUILD_DIR:-build}
cmake -S . -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE:-Release}" "$@" >/dev/null
cmake --build "$BUILD_DIR" -j"$(nproc)"
ctest --test-dir "$BUILD_DIR" --output-on-failure

if command -v clang-format >/dev/null 2>&1; then
  git ls-files '*.cpp' '*.hpp' | xargs -r clang-format --dry-run -Werror
else
  echo "check: clang-format not found, skipping format check" >&2
fi

echo "check: OK"

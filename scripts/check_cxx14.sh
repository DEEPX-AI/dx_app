#!/usr/bin/env bash
# C++14 conformance guard (spec B1/B2): the four headers in HEADERS - the
# three that own the shared g_interrupted() flag, plus ordered_queue.hpp -
# must each compile standalone under -std=gnu++14 with -Werror. This catches:
#   - a C++17 extension (e.g. a structured binding) sneaking back into any of
#     them, since -Werror=c++17-extensions would fail;
#   - a broken forward-declaration/definition split of g_interrupted() (the
#     original B2 bug: declared in common_util.hpp, defined only in
#     run_dir.hpp, so any TU including just common_util.hpp warned "used but
#     never defined" under plain -Werror).
#
#   bash scripts/check_cxx14.sh [--root <dir>]
#
# --root checks the headers of another tree (a copy under test) instead of
# this repository's: tests/scripts/test_check_cxx14.py breaks a copy, never
# the real file (U-74).
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd -P)"
if [ "${1:-}" = "--root" ]; then
  if [ -z "${2:-}" ] || [ ! -d "${2}/src/cpp_example/common/utility" ]; then
    echo "check_cxx14: --root needs a tree holding src/cpp_example/common/utility" >&2
    exit 2
  fi
  if [ $# -gt 2 ]; then
    echo "check_cxx14: unknown argument: $3 (usage: check_cxx14.sh [--root <dir>])" >&2
    exit 2
  fi
  ROOT="$(cd "$2" && pwd -P)"
elif [ $# -gt 0 ]; then
  echo "check_cxx14: unknown argument: $1 (usage: check_cxx14.sh [--root <dir>])" >&2
  exit 2
fi
CE="$ROOT/src/cpp_example"
W="$(mktemp -d)"
trap 'rm -rf "$W"' EXIT

FLAGS=(-std=gnu++14 -W -Wall -Wextra -Werror -O0 -fPIC -fsyntax-only)
INC=(-I"$CE" -isystem /usr/include/opencv4)
DEFS=(-DPROJECT_ROOT_DIR="\"$ROOT\"")

HEADERS=(
  "common/utility/interrupt_flag.hpp"
  "common/utility/common_util.hpp"
  "common/utility/run_dir.hpp"
  "common/utility/ordered_queue.hpp"
)

status=0
for h in "${HEADERS[@]}"; do
  echo "checking $h ..."
  src="$W/$(echo "$h" | tr '/.' '__').cpp"
  cat > "$src" <<EOF
#include "$h"
int main() { return 0; }
EOF
  if ! g++ "${FLAGS[@]}" "${INC[@]}" "${DEFS[@]}" -c "$src" -o "$src.o" > "$W/out.log" 2>&1; then
    cat "$W/out.log"
    echo "FAILED: $h is not a self-contained, C++14-conformant header" >&2
    status=1
  fi
done

if [ "$status" -ne 0 ]; then
  exit 1
fi
echo "check_cxx14 OK: all headers compile standalone under -std=gnu++14 -Werror"

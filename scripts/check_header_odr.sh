#!/usr/bin/env bash
# Two translation units including every shared header must link.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd -P)"
CE="$ROOT/src/cpp_example"
W="$(mktemp -d)"
trap 'rm -rf "$W"' EXIT

FLAGS="-O0 -std=gnu++14 -w -pthread -fPIC -DPROJECT_ROOT_DIR=\"$ROOT\""   # common_util.hpp reads it
FLAGS="$FLAGS -DDXRT_LEGACY_HEADER_OK"   # silences dxrt's "<dxrt/...> is obsolete" #pragma notes
INC="-I/usr/include/opencv4 -I$ROOT/extern -I$ROOT/src/utility \
     -I$ROOT/src/postprocess/sfa3d -I$ROOT/src/postprocess/superpoint \
     -I$ROOT/src/postprocess/dope -I$ROOT/src/postprocess/yolopv2 \
     -I$ROOT/src/postprocess/vitpose \
     -I$CE -I$CE/common/processors"

# Every shared header, not one: a function or object defined without
# `inline` in any of them is a "multiple definition" link error here instead
# of in the first program that includes it from two files (U-53).
DIRS="common/base common/config common/processors common/utility common/visualizers"
EXCLUDE=""   # "<header> <reason>" per line; empty unless a header cannot compile here
mapfile -t HEADERS < <(cd "$CE" && find $DIRS -maxdepth 1 -name '*.hpp' | sort)
[ "${#HEADERS[@]}" -gt 0 ] || { echo "check_header_odr: no headers found under $CE" >&2; exit 1; }

for i in a b; do
  {
    for h in "${HEADERS[@]}"; do
      case $'\n'"$EXCLUDE" in *$'\n'"$h "*) continue ;; esac
      echo "#include \"$h\""
    done
    echo "namespace { int anchor_$i() { return 0; } }"
  } > "$W/tu_$i.cpp"
  g++ $FLAGS $INC -c "$W/tu_$i.cpp" -o "$W/tu_$i.o"
done

echo 'int main() { return 0; }' > "$W/main.cpp"
g++ $FLAGS $INC -c "$W/main.cpp" -o "$W/main.o"
g++ "$W"/tu_a.o "$W"/tu_b.o "$W"/main.o -o "$W/app" \
    $(pkg-config --libs opencv4 2>/dev/null || echo "-lopencv_core -lopencv_imgproc") -ldxrt -lstdc++fs
echo "two-TU link OK (${#HEADERS[@]} headers)"

# Last phase: two variants of one family, linked into one program, keep their
# own factory and postprocessor (per-variant namespaces, check_variant_odr.sh).
# The probe's compiler notes go to a log that is printed only on failure.
if ! bash "$ROOT/scripts/check_variant_odr.sh" > "$W/variant_odr.log" 2>&1; then
  cat "$W/variant_odr.log"
  echo "check_header_odr: the variant ODR probe failed" >&2
  exit 1
fi
grep -E '^(link order|check_variant_odr)' "$W/variant_odr.log"

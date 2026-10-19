#!/usr/bin/env bash
# Two variants of one family, linked into one program, must keep their own
# postprocessor. Before per-variant namespaces the linker silently kept one
# Yolo11Factory::createPostprocessor for both (weak symbols, no diagnostic).
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd); T=$ROOT/src/cpp_example
W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
F=(-std=c++14 -O0 "-DPROJECT_ROOT_DIR=\"$ROOT\"" -I"$ROOT/extern" -I"$ROOT/src/utility" -I"$T" -I"$T/common/processors")
for d in "$ROOT"/src/postprocess/*/; do F+=(-I"$d"); done
F+=($(pkg-config --cflags opencv4))
probe() {  # $1 tag  $2 variant header
  local cls; cls=$(grep -oE 'class \w+Factory' "$T/$2" | head -1 | cut -d' ' -f2)
  local q; q=$(grep -oE '^namespace v_\w+' "$T/$2" | head -1 | cut -d' ' -f2)
  printf '#include "%s"\n#include <string>\n#include <typeinfo>\nstd::string from_%s() { dxapp::%s%s f; auto p = f.createPostprocessor(640, 640); return typeid(*p).name(); }\n' \
    "$2" "$1" "${q:+$q::}" "$cls" > "$W/$1.cpp"
  g++ "${F[@]}" -c "$W/$1.cpp" -o "$W/$1.o"
}
probe n object_detection/yolo11/yolo11-n_640x640/factory/yolo11-n_640x640_factory.hpp
probe s object_detection/yolo11/yolo11-s_640x640/factory/yolo11-s_640x640_factory.hpp
printf '#include <iostream>\n#include <string>\nstd::string from_n(); std::string from_s();\nint main() { std::cout << from_n() << "\\n" << from_s() << "\\n"; }\n' > "$W/main.cpp"
g++ "${F[@]}" -c "$W/main.cpp" -o "$W/main.o"
g++ "${F[@]}" -c "$ROOT/src/utility/common_util.cpp" -o "$W/cu.o"
L=($(pkg-config --libs opencv4) -ldxrt -lstdc++fs)
for order in "n s" "s n"; do
  set -- $order
  g++ "$W/$1.o" "$W/$2.o" "$W/main.o" "$W/cu.o" -o "$W/probe" "${L[@]}"
  out=$("$W/probe" | c++filt -t)
  a=$(echo "$out" | sed -n 1p); b=$(echo "$out" | sed -n 2p)
  echo "link order $order: yolo11-n -> $a ; yolo11-s -> $b"
  case "$a" in *YOLOv11Postprocessor) ;; *) echo "FAIL: yolo11-n built $a"; exit 1;; esac
  case "$b" in *YOLOv8Postprocessor) ;; *) echo "FAIL: yolo11-s built $b"; exit 1;; esac
done
echo "check_variant_odr: OK"

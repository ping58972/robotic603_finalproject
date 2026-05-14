#!/usr/bin/env bash

source /opt/ros/noetic/setup.bash
source /catkin_ws/devel/setup.bash

target_symbol=""
target_seek_enabled="true"

for arg in "$@"; do
  case "$arg" in
    target_symbol:=*)
      target_symbol="${arg#target_symbol:=}"
      ;;
    target_seek:=false|target_seek:=False|target_seek:=0)
      target_seek_enabled="false"
      ;;
  esac
done

normalize_symbol() {
  case "$(echo "$1" | tr '[:upper:]' '[:lower:]' | xargs)" in
    star|stars) echo "star" ;;
    square|squares) echo "square" ;;
    circle|circles) echo "circle" ;;
    triangle|triangles|triange|trianlge) echo "triangle" ;;
    *) return 1 ;;
  esac
}

if [[ "$target_seek_enabled" == "true" && -z "$target_symbol" ]]; then
  if [[ ! -t 0 ]]; then
    echo "No interactive stdin available. Pass target_symbol:=star, square, circle, or triangle." >&2
    exit 2
  fi

  while true; do
    read -r -p "Target symbol [star, square, circle, triangle]: " requested_symbol
    if normalized_symbol="$(normalize_symbol "$requested_symbol")"; then
      set -- "$@" "target_symbol:=${normalized_symbol}"
      break
    fi
    echo "Invalid target symbol: ${requested_symbol}" >&2
  done
fi

roslaunch symbols_recognition real_robot.launch "$@"

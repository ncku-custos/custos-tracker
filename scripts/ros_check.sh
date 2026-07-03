#!/usr/bin/env bash
# ROS2-side gate: build+install the core to a local prefix, then colcon
# build + test the ros2/ packages against it. Core-only checkouts keep using
# scripts/check.sh; this script additionally needs ROS2 Lyrical
# (ros-lyrical-ros-base, ament, ros-lyrical-vision-msgs) and colcon.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ ! -f /opt/ros/lyrical/setup.bash ]; then
  echo "ros_check: /opt/ros/lyrical not found — install ROS2 Lyrical first" >&2
  exit 1
fi
# ROS setup scripts are not nounset-clean.
set +u
# shellcheck disable=SC1091
source /opt/ros/lyrical/setup.bash
set -u

BUILD_DIR=${BUILD_DIR:-build}
BUILD_TYPE=${CMAKE_BUILD_TYPE:-Release}

cmake -S . -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE="$BUILD_TYPE" >/dev/null
cmake --build "$BUILD_DIR" -j"$(nproc)"
cmake --install "$BUILD_DIR" --prefix ros2/install-ctrk >/dev/null

# colcon runs with cwd ros2/ so its build/install/log dirs stay out of the
# root cmake build tree.
cd ros2
CMAKE_PREFIX_PATH="$PWD/install-ctrk${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}" \
  colcon build --cmake-args -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
colcon test
colcon test-result --verbose

echo "ros_check: OK"

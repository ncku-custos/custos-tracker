cmake_minimum_required(VERSION 3.28)
project(ctrk_ros VERSION 0.4.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
add_compile_options(-Wall -Wextra -Wpedantic)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_lifecycle REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(vision_msgs REQUIRED)
find_package(ctrk_interfaces REQUIRED)
# Not a rosdep key: installed by scripts/ros_check.sh into ros2/install-ctrk
# and found through CMAKE_PREFIX_PATH (docs/DECISIONS.md D-0015).
find_package(ctrk 0.4 REQUIRED)

# Support layer (param->config mapping, msg conversion, sink bridge) as an
# object collection the component library and the unit tests share.
add_library(ctrk_ros_support STATIC
  src/param_config.cpp
  src/msg_convert.cpp
  src/sink_bridge.cpp
)
target_include_directories(ctrk_ros_support PUBLIC src)
target_link_libraries(ctrk_ros_support PUBLIC
  ctrk::sot
  ctrk::tbd
  rclcpp::rclcpp
  rclcpp_lifecycle::rclcpp_lifecycle
  ${sensor_msgs_TARGETS}
  ${std_srvs_TARGETS}
  ${vision_msgs_TARGETS}
  ${ctrk_interfaces_TARGETS}
)
set_target_properties(ctrk_ros_support PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_params test/test_params.cpp)
  target_link_libraries(test_params ctrk_ros_support)
  ament_add_gtest(test_msg_convert test/test_msg_convert.cpp)
  target_link_libraries(test_msg_convert ctrk_ros_support)
endif()

ament_package()

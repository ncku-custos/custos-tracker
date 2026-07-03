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
# frames_player reads image sequences, draw_node rasterizes boxes; the ctrk
# Config only pulls the components the core libs link.
find_package(OpenCV 4 REQUIRED COMPONENTS core imgproc imgcodecs)

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

# All components live in ONE shared library: the ctrk log/profile sinks are
# process-wide globals inside the statically linked core — one .so means one
# copy shared by every composed node (docs/DECISIONS.md D-0016).
add_library(ctrk_ros_components SHARED
  src/tbd_node.cpp
  src/sot_node.cpp
  src/frames_player_node.cpp
  src/draw_node.cpp
)
target_link_libraries(ctrk_ros_components PUBLIC
  ctrk_ros_support
  rclcpp_components::component
  opencv_core
  opencv_imgproc
  opencv_imgcodecs
)
rclcpp_components_register_node(ctrk_ros_components
  PLUGIN "ctrk_ros::TbdNode"
  EXECUTABLE ctrk_tbd_node
)
rclcpp_components_register_node(ctrk_ros_components
  PLUGIN "ctrk_ros::SotNode"
  EXECUTABLE ctrk_sot_node
)
rclcpp_components_register_node(ctrk_ros_components
  PLUGIN "ctrk_ros::FramesPlayerNode"
  EXECUTABLE ctrk_frames_player
)
rclcpp_components_register_node(ctrk_ros_components
  PLUGIN "ctrk_ros::DrawNode"
  EXECUTABLE ctrk_draw_node
)

install(TARGETS ctrk_ros_components
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)
install(DIRECTORY launch params DESTINATION share/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  # Tests locate repo models the same way core tests do (skip when absent).
  get_filename_component(CTRK_REPO_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../.. ABSOLUTE)

  ament_add_gtest(test_params test/test_params.cpp)
  target_link_libraries(test_params ctrk_ros_support)
  ament_add_gtest(test_msg_convert test/test_msg_convert.cpp)
  target_link_libraries(test_msg_convert ctrk_ros_support)
  ament_add_gtest(test_lifecycle_tbd test/test_lifecycle_tbd.cpp)
  target_link_libraries(test_lifecycle_tbd ctrk_ros_components)
  target_compile_definitions(test_lifecycle_tbd PRIVATE CTRK_REPO_ROOT="${CTRK_REPO_ROOT}")
  # The CI anchor: full pub/sub e2e with the model-free MOSSE backend.
  ament_add_gtest(test_lifecycle_sot_mosse test/test_lifecycle_sot_mosse.cpp TIMEOUT 120)
  target_link_libraries(test_lifecycle_sot_mosse ctrk_ros_components)
  # Full composed graph (player -> sot -> draw), plugin-loaded, IPC on.
  find_package(class_loader REQUIRED)
  find_package(ament_index_cpp REQUIRED)
  ament_add_gtest(test_composition test/test_composition.cpp TIMEOUT 120)
  target_link_libraries(test_composition ctrk_ros_components class_loader::class_loader
    ament_index_cpp::ament_index_cpp)
endif()

ament_package()

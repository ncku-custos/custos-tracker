cmake_minimum_required(VERSION 3.28)
project(ctrk_interfaces VERSION 0.4.0)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  msg/SotStatus.msg
  srv/SetTarget.srv
  DEPENDENCIES std_msgs
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()

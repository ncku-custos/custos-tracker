cmake_minimum_required(VERSION 3.28)
project(ctrk VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(CTRK_BUILD_TESTS "Build unit tests" ON)
option(CTRK_BUILD_APPS "Build CLI apps" ON)
option(CTRK_WITH_ORT "Build the ONNX Runtime inference backend" ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

add_compile_options(-Wall -Wextra -Wpedantic)

find_package(OpenCV 4 REQUIRED COMPONENTS
  core imgproc imgcodecs videoio video dnn highgui)

if(CTRK_WITH_ORT)
  include(cmake/ort.cmake)
endif()

add_subdirectory(core)

if(CTRK_BUILD_APPS AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/apps/CMakeLists.txt)
  add_subdirectory(apps)
endif()

if(CTRK_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

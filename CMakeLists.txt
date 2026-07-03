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
# Opt-in host tuning (docs/DECISIONS.md D-0009). The portable default stays
# plain -O3: the deployment target is an ARM SoC, so ISA-specific flags are a
# benchmarking tool here, never the shipped configuration.
option(CTRK_MARCH_NATIVE "Tune for the build host CPU (-march=native)" OFF)
option(CTRK_LTO "Enable link-time optimization for release builds" OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

add_compile_options(-Wall -Wextra -Wpedantic)

if(CTRK_MARCH_NATIVE)
  add_compile_options(-march=native)
endif()
if(CTRK_LTO)
  include(CheckIPOSupported)
  check_ipo_supported(RESULT ctrk_ipo_ok OUTPUT ctrk_ipo_msg)
  if(ctrk_ipo_ok)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
  else()
    message(WARNING "CTRK_LTO requested but IPO unsupported: ${ctrk_ipo_msg}")
  endif()
endif()

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

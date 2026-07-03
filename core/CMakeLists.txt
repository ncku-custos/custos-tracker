configure_file(src/common/version.cpp.in ${CMAKE_CURRENT_BINARY_DIR}/version.cpp @ONLY)

add_library(ctrk_common STATIC
  ${CMAKE_CURRENT_BINARY_DIR}/version.cpp
  src/common/geometry.cpp
  src/common/kalman.cpp
  src/common/munkres.cpp
  src/common/subwindow.cpp
  src/common/appearance.cpp
  src/common/log.cpp
  src/common/profile.cpp
)
add_library(ctrk::common ALIAS ctrk_common)
# Exported name matches the in-tree alias, so find_package(ctrk) consumers
# and in-tree code link the same ctrk::<name> targets.
set_target_properties(ctrk_common PROPERTIES EXPORT_NAME common)

target_include_directories(ctrk_common
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  PRIVATE
    src)
target_link_libraries(ctrk_common PRIVATE opencv_core opencv_imgproc)

if(CTRK_WITH_ORT)
  add_library(ctrk_infer STATIC src/infer/ort_engine.cpp)
  add_library(ctrk::infer ALIAS ctrk_infer)
  set_target_properties(ctrk_infer PROPERTIES EXPORT_NAME infer)
  target_include_directories(ctrk_infer
    PUBLIC
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
      $<INSTALL_INTERFACE:include>
    PRIVATE
      src)
  target_link_libraries(ctrk_infer PUBLIC ctrk::common PRIVATE ctrk::ort)
  install(TARGETS ctrk_infer EXPORT ctrkTargets)

  add_library(ctrk_tbd STATIC
    src/tbd/yolov8.cpp
    src/tbd/byte_tracker.cpp
    src/tbd/multi_tracker.cpp
    src/tbd/gmc.cpp
  )
  add_library(ctrk::tbd ALIAS ctrk_tbd)
  set_target_properties(ctrk_tbd PROPERTIES EXPORT_NAME tbd)
  target_include_directories(ctrk_tbd
    PUBLIC
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
      $<INSTALL_INTERFACE:include>
    PRIVATE
      src)
  target_link_libraries(ctrk_tbd PUBLIC ctrk::common ctrk::infer
    PRIVATE opencv_core opencv_imgproc opencv_video opencv_calib3d)
  install(TARGETS ctrk_tbd EXPORT ctrkTargets)

  add_library(ctrk_sot STATIC
    src/sot/nanotrack.cpp
    src/sot/mosse.cpp
    src/sot/sot_tracker.cpp
  )
  add_library(ctrk::sot ALIAS ctrk_sot)
  set_target_properties(ctrk_sot PROPERTIES EXPORT_NAME sot)
  target_include_directories(ctrk_sot
    PUBLIC
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
      $<INSTALL_INTERFACE:include>
    PRIVATE
      src)
  target_link_libraries(ctrk_sot PUBLIC ctrk::common ctrk::infer PRIVATE opencv_core opencv_imgproc)
  install(TARGETS ctrk_sot EXPORT ctrkTargets)
endif()

install(TARGETS ctrk_common EXPORT ctrkTargets)
install(DIRECTORY include/ DESTINATION include)
install(EXPORT ctrkTargets NAMESPACE ctrk:: DESTINATION lib/cmake/ctrk)

include(CMakePackageConfigHelpers)
configure_package_config_file(${PROJECT_SOURCE_DIR}/cmake/ctrkConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/ctrkConfig.cmake
  INSTALL_DESTINATION lib/cmake/ctrk)
write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/ctrkConfigVersion.cmake
  COMPATIBILITY SameMajorVersion)
install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/ctrkConfig.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/ctrkConfigVersion.cmake
  DESTINATION lib/cmake/ctrk)

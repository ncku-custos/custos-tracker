configure_file(src/common/version.cpp.in ${CMAKE_CURRENT_BINARY_DIR}/version.cpp @ONLY)

add_library(ctrk_common STATIC
  ${CMAKE_CURRENT_BINARY_DIR}/version.cpp
)
add_library(ctrk::common ALIAS ctrk_common)

target_include_directories(ctrk_common
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  PRIVATE
    src)
target_link_libraries(ctrk_common PRIVATE opencv_core opencv_imgproc)

install(TARGETS ctrk_common EXPORT ctrkTargets)
install(DIRECTORY include/ DESTINATION include)
install(EXPORT ctrkTargets NAMESPACE ctrk:: DESTINATION lib/cmake/ctrk)

add_library(ctrk_app_common STATIC app_common.cpp)
target_include_directories(ctrk_app_common
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/core/src)
target_link_libraries(ctrk_app_common
  PUBLIC ctrk::tbd
  PUBLIC ctrk::common opencv_core opencv_imgproc opencv_videoio opencv_highgui)

add_executable(track_tbd track_tbd.cpp)
target_link_libraries(track_tbd PRIVATE ctrk_app_common)

add_executable(track_sot track_sot.cpp)
target_link_libraries(track_sot PRIVATE ctrk_app_common)

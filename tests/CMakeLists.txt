include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.17.0.zip
  URL_HASH SHA256=40d4ec942217dcc84a9ebe2a68584ada7d4a33a8ee958755763278ea1c5e18ff
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
FetchContent_MakeAvailable(googletest)

# One test binary per component; ctrk_add_test(<name> <sources...>)
function(ctrk_add_test name)
  add_executable(${name} ${ARGN})
  target_link_libraries(${name} PRIVATE ctrk::common GTest::gtest_main opencv_core opencv_imgproc opencv_videoio)
  # Tests may reach into internal headers under core/src.
  target_include_directories(${name} PRIVATE ${CMAKE_SOURCE_DIR}/core/src ${CMAKE_SOURCE_DIR}/tests)
  target_compile_definitions(${name} PRIVATE CTRK_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
  gtest_discover_tests(${name})
endfunction()

include(GoogleTest)

ctrk_add_test(test_version test_version.cpp)
ctrk_add_test(test_geometry test_geometry.cpp)
ctrk_add_test(test_kalman test_kalman.cpp)
ctrk_add_test(test_munkres test_munkres.cpp)
ctrk_add_test(test_common_infra test_common_infra.cpp)
ctrk_add_test(test_synth_e2e test_synth_e2e.cpp)
if(CTRK_WITH_ORT)
  ctrk_add_test(test_infer test_infer.cpp)
  target_link_libraries(test_infer PRIVATE ctrk::infer)
  ctrk_add_test(test_yolo_decode test_yolo_decode.cpp)
  target_link_libraries(test_yolo_decode PRIVATE ctrk::tbd)
  ctrk_add_test(test_byte_tracker test_byte_tracker.cpp)
  target_link_libraries(test_byte_tracker PRIVATE ctrk::tbd)
  ctrk_add_test(test_nanotrack_oracle test_nanotrack_oracle.cpp)
  target_link_libraries(test_nanotrack_oracle PRIVATE ctrk::sot opencv_video opencv_dnn)
  ctrk_add_test(test_mosse test_mosse.cpp)
  target_link_libraries(test_mosse PRIVATE ctrk::sot)
  ctrk_add_test(test_subwindow_exact test_subwindow_exact.cpp)
  target_link_libraries(test_subwindow_exact PRIVATE ctrk::sot)
  ctrk_add_test(test_sot_state test_sot_state.cpp)
  target_link_libraries(test_sot_state PRIVATE ctrk::sot)
  ctrk_add_test(test_reacquire test_reacquire.cpp)
  target_link_libraries(test_reacquire PRIVATE ctrk::sot)
endif()

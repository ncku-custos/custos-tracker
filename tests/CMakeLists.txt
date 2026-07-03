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
  target_include_directories(${name} PRIVATE ${CMAKE_SOURCE_DIR}/core/src)
  gtest_discover_tests(${name})
endfunction()

include(GoogleTest)

ctrk_add_test(test_version test_version.cpp)
ctrk_add_test(test_geometry test_geometry.cpp)
ctrk_add_test(test_kalman test_kalman.cpp)
ctrk_add_test(test_munkres test_munkres.cpp)

# ONNX Runtime discovery. Preference order:
#   1. CTRK_ORT_ROOT — an unpacked official release tarball (CI / non-Ubuntu hosts)
#   2. System package (Ubuntu libonnxruntime-dev ships onnxruntimeConfig.cmake)
# Exposes target ctrk::ort either way.

set(CTRK_ORT_ROOT "" CACHE PATH "Path to an unpacked official onnxruntime release tarball")

add_library(ctrk_ort INTERFACE)
add_library(ctrk::ort ALIAS ctrk_ort)
# Part of the install export set: ctrk_infer PRIVATE-links it, which still
# lands in the static lib's LINK_ONLY interface. Consumers must find_package
# onnxruntime themselves (proper Config-file plumbing is a step-4 task).
install(TARGETS ctrk_ort EXPORT ctrkTargets)

if(CTRK_ORT_ROOT)
  if(NOT EXISTS ${CTRK_ORT_ROOT}/include/onnxruntime_cxx_api.h)
    message(FATAL_ERROR "CTRK_ORT_ROOT=${CTRK_ORT_ROOT} does not look like an onnxruntime release tarball")
  endif()
  target_include_directories(ctrk_ort INTERFACE ${CTRK_ORT_ROOT}/include)
  target_link_directories(ctrk_ort INTERFACE ${CTRK_ORT_ROOT}/lib)
  target_link_libraries(ctrk_ort INTERFACE onnxruntime)
  message(STATUS "ONNX Runtime: tarball at ${CTRK_ORT_ROOT}")
else()
  find_package(onnxruntime CONFIG REQUIRED)
  target_link_libraries(ctrk_ort INTERFACE onnxruntime::onnxruntime)
  message(STATUS "ONNX Runtime: system package ${onnxruntime_VERSION}")
endif()

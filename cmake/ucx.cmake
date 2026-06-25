if(NOT UCX)
  return()
endif()

execute_process(
  COMMAND "${Python3_EXECUTABLE}" -c
          "import commux, pathlib; p=pathlib.Path(commux.__file__).resolve().parent; print(p / 'include'); print(p / 'lib'); print(p / 'lib' / 'libcommux.so')"
  OUTPUT_VARIABLE _commux_info
  OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE _commux_probe)

if(NOT _commux_probe EQUAL 0)
  message(FATAL_ERROR
          "UCX support now requires the Python package 'commux' to be "
          "installed in ${Python3_EXECUTABLE}'s environment")
endif()

string(REPLACE "\n" ";" _commux_lines "${_commux_info}")
list(GET _commux_lines 0 COMMUX_INCLUDE_DIR)
list(GET _commux_lines 1 COMMUX_LIBRARY_DIR)
list(GET _commux_lines 2 COMMUX_LIBRARY)

if(NOT EXISTS "${COMMUX_INCLUDE_DIR}/commux/process_group_ucx.hpp")
  message(FATAL_ERROR "commux header not found under ${COMMUX_INCLUDE_DIR}")
endif()

if(NOT EXISTS "${COMMUX_LIBRARY}")
  message(FATAL_ERROR "commux library not found at ${COMMUX_LIBRARY}")
endif()

find_path(
  UCX_INCLUDE_DIR
  NAMES ucp/api/ucp.h
  HINTS
    "${COMMUX_INCLUDE_DIR}"
    "${COMMUX_INCLUDE_DIR}/.."
    "$ENV{UCX_ROOT}/include"
    "${CMAKE_BINARY_DIR}/include"
    "/opt/nvidia/hpc_sdk/Linux_x86_64/26.3/comm_libs/13.1/hpcx/hpcx-2.25.1/ucx/include")

if(NOT UCX_INCLUDE_DIR)
  message(FATAL_ERROR
          "commux header requires UCX headers, but ucp/api/ucp.h was not found")
endif()

execute_process(
  COMMAND ldd "${COMMUX_LIBRARY}"
  OUTPUT_VARIABLE _commux_ldd
  ERROR_VARIABLE _commux_ldd_error
  RESULT_VARIABLE _commux_ldd_result)
if(_commux_ldd_result EQUAL 0 AND _commux_ldd MATCHES "libc10_cuda")
  set(COMMUX_CUDA_FOUND TRUE)
else()
  set(COMMUX_CUDA_FOUND FALSE)
endif()

add_library(commux::commux SHARED IMPORTED GLOBAL)
set_target_properties(
  commux::commux PROPERTIES
  IMPORTED_LOCATION "${COMMUX_LIBRARY}"
  INTERFACE_INCLUDE_DIRECTORIES "${COMMUX_INCLUDE_DIR};${UCX_INCLUDE_DIR}")

set(UCX_FOUND TRUE)
set(COMMUX_FOUND TRUE)
message(STATUS "UCX backend: commux at ${COMMUX_LIBRARY}")
message(STATUS "UCX backend: commux CUDA support ${COMMUX_CUDA_FOUND}")

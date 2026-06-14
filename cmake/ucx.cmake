include(FetchContent)
include(ExternalProject)

if(NOT UCX)
  return()
endif()

set(UCX_VERSION "v1.20.1" CACHE STRING "Pinned OpenUCX version")
set(UCX_SOURCE_CACHE
    "${CMAKE_SOURCE_DIR}/.cache/ucx-${UCX_VERSION}.tar.gz")
set(UCX_INSTALL_DIR "${CMAKE_BINARY_DIR}")
file(MAKE_DIRECTORY "${UCX_INSTALL_DIR}/include" "${UCX_INSTALL_DIR}/lib")

if(EXISTS "${UCX_SOURCE_CACHE}")
  FetchContent_Declare(
    ucx
    URL "${UCX_SOURCE_CACHE}"
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
else()
  FetchContent_Declare(
    ucx
    GIT_REPOSITORY https://github.com/openucx/ucx.git
    GIT_TAG "${UCX_VERSION}"
    GIT_SHALLOW TRUE
    UPDATE_DISCONNECTED TRUE)
endif()

FetchContent_GetProperties(ucx)
if(NOT ucx_POPULATED)
  if(POLICY CMP0169)
    cmake_policy(SET CMP0169 OLD)
  endif()
  FetchContent_Populate(ucx)
  if(NOT EXISTS "${UCX_SOURCE_CACHE}")
    file(MAKE_DIRECTORY "${CMAKE_SOURCE_DIR}/.cache")
    execute_process(
      COMMAND "${CMAKE_COMMAND}" -E tar czf "${UCX_SOURCE_CACHE}" .
      WORKING_DIRECTORY "${ucx_SOURCE_DIR}"
      COMMAND_ERROR_IS_FATAL ANY)
  endif()
endif()

set(_ucx_cuda_arg "--without-cuda")
if(CMAKE_CUDA_COMPILER)
  if(CUDA_TOOLKIT_ROOT_DIR)
    set(_cuda_root "${CUDA_TOOLKIT_ROOT_DIR}")
  elseif(CUDAToolkit_ROOT)
    set(_cuda_root "${CUDAToolkit_ROOT}")
  else()
    get_filename_component(_cuda_root "${CMAKE_CUDA_COMPILER}" DIRECTORY)
    get_filename_component(_cuda_root "${_cuda_root}" DIRECTORY)
  endif()
  set(_ucx_cuda_arg "--with-cuda=${_cuda_root}")
endif()

ExternalProject_Add(
  ucx_external
  SOURCE_DIR "${ucx_SOURCE_DIR}"
  INSTALL_DIR "${UCX_INSTALL_DIR}"
  BUILD_IN_SOURCE TRUE
  CONFIGURE_COMMAND
    /bin/sh -c
    "./autogen.sh && ./contrib/configure-release --prefix=${UCX_INSTALL_DIR} --with-pic --enable-mt --disable-static --enable-shared --without-java --without-go ${_ucx_cuda_arg}"
  BUILD_COMMAND "${CMAKE_MAKE_PROGRAM}" -j2
  INSTALL_COMMAND "${CMAKE_MAKE_PROGRAM}" install
  BUILD_BYPRODUCTS
    "${UCX_INSTALL_DIR}/lib/libucp.so"
    "${UCX_INSTALL_DIR}/lib/libuct.so"
    "${UCX_INSTALL_DIR}/lib/libucs.so"
    "${UCX_INSTALL_DIR}/lib/libucm.so")

add_library(UCX::ucp SHARED IMPORTED GLOBAL)
set_target_properties(
  UCX::ucp PROPERTIES
  IMPORTED_LOCATION "${UCX_INSTALL_DIR}/lib/libucp.so"
  INTERFACE_INCLUDE_DIRECTORIES "${UCX_INSTALL_DIR}/include")
add_dependencies(UCX::ucp ucx_external)

add_library(UCX::uct SHARED IMPORTED GLOBAL)
set_target_properties(
  UCX::uct PROPERTIES
  IMPORTED_LOCATION "${UCX_INSTALL_DIR}/lib/libuct.so"
  INTERFACE_INCLUDE_DIRECTORIES "${UCX_INSTALL_DIR}/include")
add_dependencies(UCX::uct ucx_external)

add_library(UCX::ucs SHARED IMPORTED GLOBAL)
set_target_properties(
  UCX::ucs PROPERTIES
  IMPORTED_LOCATION "${UCX_INSTALL_DIR}/lib/libucs.so"
  INTERFACE_INCLUDE_DIRECTORIES "${UCX_INSTALL_DIR}/include")
add_dependencies(UCX::ucs ucx_external)

add_library(UCX::ucm SHARED IMPORTED GLOBAL)
set_target_properties(
  UCX::ucm PROPERTIES
  IMPORTED_LOCATION "${UCX_INSTALL_DIR}/lib/libucm.so"
  INTERFACE_INCLUDE_DIRECTORIES "${UCX_INSTALL_DIR}/include")
add_dependencies(UCX::ucm ucx_external)

set(UCX_FOUND TRUE)

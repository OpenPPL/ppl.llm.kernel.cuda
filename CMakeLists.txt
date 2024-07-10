cmake_minimum_required(VERSION 3.14)
project(ppl.llm.kernel.cuda)
# --------------------------------------------------------------------------- #

# options
option(PPLNN_CUDA_ENABLE_NCCL "enable nccl" OFF)
option(PPLNN_CUDA_ENABLE_CUDNN "enable cudnn, if it is disabled, some kernel will not work" OFF)
option(PPLNN_INSTALL "" ON)
option(PPLNN_CUDA_DISABLE_CONV_ALL "" ON)

if(MSVC)
    option(PPLNN_USE_MSVC_STATIC_RUNTIME "" ON)
endif()

if(NOT CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 17)
endif()

if(NOT CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 17)
endif()

# --------------------------------------------------------------------------- #

# dependencies
set(PPLCOMMON_USE_CUDA ON)
if (PPLNN_CUDA_ENABLE_NCCL)
    set(PPLCOMMON_ENABLE_NCCL ON)
endif()
include(cmake/deps.cmake)

include(${HPCC_DEPS_DIR}/hpcc/cmake/cuda-common.cmake)

if(CUDA_VERSION VERSION_LESS "9.0")
    message(FATAL_ERROR "cuda verson [${CUDA_VERSION}] < min required [9.0]")
elseif(CUDA_VERSION VERSION_LESS "10.2")
    message(WARNNING " strongly recommend cuda >= 10.2, now is [${CUDA_VERSION}]")
endif()

# --------------------------------------------------------------------------- #

if (PPLNN_CUDA_ENABLE_CUDNN)
    list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules)
    find_package(CUDNN)
    if(NOT CUDNN_FOUND)
        message(FATAL_ERROR "cudnn is required.")
    endif()
    message(STATUS "CUDNN_VERSION: ${CUDNN_VERSION}")
    list(APPEND PPLNN_CUDA_BINARY_DEFINITIONS PPLNN_CUDA_ENABLE_CUDNN)
    list(APPEND PPLNN_CUDA_EXTERNAL_INCLUDE_DIRECTORIES ${CUDNN_INCLUDE_PATH})
    list(APPEND PPLNN_CUDA_EXTERNAL_LINK_LIBRARIES ${CUDNN_LIBRARY})
else()
    message(WARNING "cudnn is disabled, some kernel will not avaliable.")
endif()

# --------------------------------------------------------------------------- #

# compiler related

if(PPLNN_USE_MSVC_STATIC_RUNTIME)
    hpcc_use_msvc_static_runtime()
else()
    hpcc_use_msvc_dynamic_runtime()
endif()

# --------------------------------------------------------------------------- #

set(CUTLASS_ENABLE_EXAMPLES OFF CACHE BOOL "")
set(CUTLASS_ENABLE_PROFILER OFF CACHE BOOL "")
set(CUTLASS_ENABLE_TOOLS OFF CACHE BOOL "")
# disable installing of cutlass
FetchContent_GetProperties(cutlass)
if(NOT cutlass_POPULATED)
    FetchContent_Populate(cutlass)
    add_subdirectory(${cutlass_SOURCE_DIR} ${cutlass_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()

# --------------------------------------------------------------------------- #

get_filename_component(__CUDA_LLM_SRC_DIR__ "${CMAKE_CURRENT_SOURCE_DIR}/src" ABSOLUTE)
get_filename_component(__CUDA_LLM_INCLUDE_DIR__ "${CMAKE_CURRENT_SOURCE_DIR}/include" ABSOLUTE)
file(GLOB_RECURSE __CUDA_LLM_SRC__ ${__CUDA_LLM_SRC_DIR__}/*.cu)
list(APPEND PPLNN_CUDA_KERNEL_GPU_SRC ${__CUDA_LLM_SRC__})
list(APPEND PPLNN_CUDA_EXTERNAL_INCLUDE_DIRECTORIES ${__CUDA_LLM_INCLUDE_DIR__})
unset(__CUDA_LLM_SRC__)
unset(__CUDA_LLM_SRC_DIR__)
unset(__CUDA_LLM_INCLUDE_DIR__)

get_filename_component(__CUTLASS_SOURCE_DIR__ "${HPCC_DEPS_DIR}/cutlass" ABSOLUTE)
set(__PPLNN_CUTLASS_INCLUDE_DIRECTORIES__
    ${__CUTLASS_SOURCE_DIR__}/include
    ${__CUTLASS_SOURCE_DIR__}/tools/util/include
    ${__CUTLASS_SOURCE_DIR__}/tools/library/include
    ${__CUTLASS_SOURCE_DIR__}/tools/library/src)
list(APPEND PPLNN_CUDA_EXTERNAL_INCLUDE_DIRECTORIES ${__PPLNN_CUTLASS_INCLUDE_DIRECTORIES__})
unset(__CUTLASS_SOURCE_DIR__)
unset(__PPLNN_CUTLASS_INCLUDE_DIRECTORIES__)

list(APPEND PPLNN_CUDA_EXTERNAL_LINK_LIBRARIES cublasLt)

set(PPLNN_CUDA_KERNEL_COMMON_NVCC_FLAGS "${PPLNN_CUDA_KERNEL_COMMON_NVCC_FLAGS} --expt-relaxed-constexpr")

hpcc_populate_dep(ppl.kernel.cuda)

# installations
if(PPLNN_INSTALL)
    install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/ DESTINATION include)
endif()

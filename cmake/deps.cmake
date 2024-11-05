if(NOT HPCC_DEPS_DIR)
    set(HPCC_DEPS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/deps)
endif()

# forces to install libraries to `lib`, not `lib64` or others
set(CMAKE_INSTALL_LIBDIR lib)

# --------------------------------------------------------------------------- #

if(CMAKE_COMPILER_IS_GNUCC)
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 4.9.0)
        message(FATAL_ERROR "gcc >= 4.9.0 is required.")
    endif()
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 6.0.0)
        message(FATAL_ERROR "clang >= 6.0.0 is required.")
    endif()
endif()

# --------------------------------------------------------------------------- #

if(APPLE)
    if(CMAKE_C_COMPILER_ID MATCHES "Clang")
        set(OpenMP_C "${CMAKE_C_COMPILER}")
        set(OpenMP_C_FLAGS "-Xclang -fopenmp -I/usr/local/opt/libomp/include -Wno-unused-command-line-argument")
    endif()
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        set(OpenMP_CXX "${CMAKE_CXX_COMPILER}")
        set(OpenMP_CXX_FLAGS "-Xclang -fopenmp -I/usr/local/opt/libomp/include -Wno-unused-command-line-argument")
    endif()
endif()

# --------------------------------------------------------------------------- #

include(FetchContent)

set(FETCHCONTENT_BASE_DIR ${HPCC_DEPS_DIR})
set(FETCHCONTENT_QUIET OFF)

if(PPLNN_HOLD_DEPS)
    set(FETCHCONTENT_UPDATES_DISCONNECTED ON)
endif()

# --------------------------------------------------------------------------- #

find_package(Git QUIET)
if(NOT Git_FOUND)
    message(FATAL_ERROR "git is required.")
endif()

if(NOT PPLNN_DEP_HPCC_VERSION)
    set(PPLNN_DEP_HPCC_VERSION master)
endif()

if(PPLNN_DEP_HPCC_PKG)
    FetchContent_Declare(hpcc
        URL ${PPLNN_DEP_HPCC_PKG}
        SOURCE_DIR ${HPCC_DEPS_DIR}/hpcc
        BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/hpcc-build
        SUBBUILD_DIR ${HPCC_DEPS_DIR}/hpcc-subbuild)
else()
    if(NOT PPLNN_DEP_HPCC_GIT)
        set(PPLNN_DEP_HPCC_GIT "https://github.com/OpenPPL/hpcc.git")
    endif()
    FetchContent_Declare(hpcc
        GIT_REPOSITORY ${PPLNN_DEP_HPCC_GIT}
        GIT_TAG ${PPLNN_DEP_HPCC_VERSION}
        SOURCE_DIR ${HPCC_DEPS_DIR}/hpcc
        BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/hpcc-build
        SUBBUILD_DIR ${HPCC_DEPS_DIR}/hpcc-subbuild)
endif()

FetchContent_GetProperties(hpcc)
if(NOT hpcc_POPULATED)
    FetchContent_Populate(hpcc)
    include(${hpcc_SOURCE_DIR}/cmake/hpcc-common.cmake)
endif()

# --------------------------------------------------------------------------- #

if(NOT PPLNN_DEP_PPLCUDAKERNEL_VERSION)
    set(PPLNN_DEP_PPLCUDAKERNEL_VERSION master)
endif()

if(PPLNN_DEP_PPLCUDAKERNEL_PKG)
    hpcc_declare_pkg_dep(ppl.kernel.cuda
        ${PPLNN_DEP_PPLCUDAKERNEL_PKG})
else()
    if(NOT PPLNN_DEP_PPLCUDAKERNEL_GIT)
        set(PPLNN_DEP_PPLCUDAKERNEL_GIT "https://github.com/OpenPPL/ppl.kernel.cuda.git")
    endif()
    hpcc_declare_git_dep(ppl.kernel.cuda
        ${PPLNN_DEP_PPLCUDAKERNEL_GIT}
        ${PPLNN_DEP_PPLCUDAKERNEL_VERSION})
endif()

# -----------------------------test------------------------------------------ #
if(PPLNN_BUILD_TESTS)
    file(GLOB_RECURSE TEST_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/test/*.cu ${CMAKE_CURRENT_SOURCE_DIR}/test/*.h)

    add_executable(test_gemm ${TEST_SOURCES})
    target_compile_options(test_gemm PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)
    target_include_directories(test_gemm PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/test)
    target_link_libraries(test_gemm PRIVATE pplkernelcuda_static ${PPLKERNELCUDA_LINK_LIBRARIES} -lcublasLt)
endif()


# --------------------------------------------------------------------------- #

set(CUTLASS_ENABLE_HEADERS_ONLY ON CACHE BOOL "Enable only the header library")

if(NOT PPLNN_DEP_CUTLASS_VERSION)
    set(PPLNN_DEP_CUTLASS_VERSION v3.4.1)
endif()

if(PPLNN_DEP_CUTLASS_PKG)
    hpcc_declare_pkg_dep(cutlass
        ${PPLNN_DEP_CUTLASS_PKG})
else()
    if(NOT PPLNN_DEP_CUTLASS_GIT)
        set(PPLNN_DEP_CUTLASS_GIT "https://github.com/NVIDIA/cutlass.git")
    endif()
    hpcc_declare_git_dep(cutlass
        ${PPLNN_DEP_CUTLASS_GIT}
        ${PPLNN_DEP_CUTLASS_VERSION})
endif()

# --------------------------------------------------------------------------- #

set(PYBIND11_INSTALL OFF CACHE BOOL "disable pybind11 installation")
set(PYBIND11_TEST OFF CACHE BOOL "disable pybind11 tests")
set(PYBIND11_NOPYTHON ON CACHE BOOL "do not find python")
set(PYBIND11_FINDPYTHON OFF CACHE BOOL "do not find python")

set(__PYBIND11_TAG__ v2.9.2)

if(PPLNN_DEP_PYBIND11_PKG)
    hpcc_declare_pkg_dep(pybind11
        ${PPLNN_DEP_PYBIND11_PKG})
elseif(PPLNN_DEP_PYBIND11_GIT)
    hpcc_declare_git_dep(pybind11
        ${PPLNN_DEP_PYBIND11_GIT}
        ${__PYBIND11_TAG__})
else()
    hpcc_declare_pkg_dep(pybind11
        "https://github.com/pybind/pybind11/archive/refs/tags/${__PYBIND11_TAG__}.zip")
endif()

unset(__PYBIND11_TAG__)

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

set(__HPCC_COMMIT__ master)

if(PPLNN_DEP_HPCC_PKG)
    FetchContent_Declare(hpcc
        URL ${PPLNN_DEP_HPCC_PKG}
        SOURCE_DIR ${HPCC_DEPS_DIR}/hpcc
        BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/hpcc-build
        SUBBUILD_DIR ${HPCC_DEPS_DIR}/hpcc-subbuild)
else()
    if(NOT PPLNN_DEP_HPCC_GIT)
        set(PPLNN_DEP_HPCC_GIT "https://github.com/openppl-public/hpcc.git")
    endif()
    FetchContent_Declare(hpcc
        GIT_REPOSITORY ${PPLNN_DEP_HPCC_GIT}
        GIT_TAG ${__HPCC_COMMIT__}
        SOURCE_DIR ${HPCC_DEPS_DIR}/hpcc
        BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/hpcc-build
        SUBBUILD_DIR ${HPCC_DEPS_DIR}/hpcc-subbuild)
endif()

unset(__HPCC_COMMIT__)

FetchContent_GetProperties(hpcc)
if(NOT hpcc_POPULATED)
    FetchContent_Populate(hpcc)
    include(${hpcc_SOURCE_DIR}/cmake/hpcc-common.cmake)
endif()

# --------------------------------------------------------------------------- #

if(PPLNN_DEP_PPLCUDAKERNEL_PKG)
    hpcc_declare_pkg_dep(ppl.kernel.cuda
        ${PPLNN_DEP_PPLCUDAKERNEL_PKG})
else()
    if(NOT PPLNN_DEP_PPLCUDAKERNEL_GIT)
        set(PPLNN_DEP_PPLCUDAKERNEL_GIT "https://github.com/openppl-public/ppl.kernel.cuda.git")
    endif()
    hpcc_declare_git_dep(ppl.kernel.cuda
        ${PPLNN_DEP_PPLCUDAKERNEL_GIT}
        master)
endif()

# --------------------------------------------------------------------------- #

set(CUTLASS_ENABLE_HEADERS_ONLY ON CACHE BOOL "Enable only the header library")

hpcc_declare_git_dep(cutlass
    https://github.com/NVIDIA/cutlass.git v3.2.2)

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

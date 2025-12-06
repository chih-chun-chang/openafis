cmake_minimum_required (VERSION 3.9)
project(openafis CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CXX_EXTENSIONS NO)

# --- Taskflow header-only target ---------------------------------------
# Assumes: ${PROJECT_SOURCE_DIR}/3rdparty/taskflow/taskflow.hpp
add_library(taskflow INTERFACE)

# We add 3rdparty as an include dir so we can #include <taskflow/taskflow.hpp>
target_include_directories(taskflow INTERFACE
    ${PROJECT_SOURCE_DIR}/3rdparty/taskflow
)

target_compile_features(taskflow INTERFACE cxx_std_17)
# Make sure any target using taskflow is at least C++17 (you already are)
target_compile_features(taskflow INTERFACE cxx_std_17)

enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 86)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules")
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(CheckIPOSupported)
check_ipo_supported(RESULT ipo_supported)
if (ipo_supported AND (CMAKE_BUILD_TYPE STREQUAL "Release" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo"))
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
else()
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION OFF)
endif()


include(CheckCXXCompilerFlag)
include(AddCXXCompilerFlag)

# Choose default build type if not set from command line
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type (Debug/Release/RelWithDebInfo/MinSizeRel)" FORCE)
endif()
message(STATUS "CMAKE_BUILD_TYPE = ${CMAKE_BUILD_TYPE}")


if (MSVC)
    string(REGEX REPLACE "[-/]W[1-4]" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4")

    add_cxx_compiler_flag(/GS-)
    add_cxx_compiler_flag(/Ob2)
    add_cxx_compiler_flag(/Oi)
    add_cxx_compiler_flag(/Ot)
    add_cxx_compiler_flag(/Oy)
    add_cxx_compiler_flag(/GT)
    add_cxx_compiler_flag(/Zc:__cplusplus)
    
    if (NOT "${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
        add_cxx_compiler_flag(/QIntel-jcc-erratum)
        add_cxx_compiler_flag(/fp:fast)
    else()
        add_cxx_compiler_flag(-march=native)
    endif()    

    foreach(flag_var CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO)
        string(REPLACE /GR /GR- "${flag_var}" "${${flag_var}}")
        string(REPLACE /EHsc /EHsc- "${flag_var}" "${${flag_var}}")
    endforeach()
    
    add_definitions(-D_HAS_EXCEPTIONS=0)
    add_definitions(-D_ITERATOR_DEBUG_LEVEL=0)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
else()
    add_cxx_compiler_flag(-Wall)
    add_cxx_compiler_flag(-Wextra)
    add_cxx_compiler_flag(-Wshadow)
    add_cxx_compiler_flag(-Werror)
    add_cxx_compiler_flag(-pedantic)
    add_cxx_compiler_flag(-pedantic-errors)
    add_cxx_compiler_flag(-Wshorten-64-to-32)
    add_cxx_compiler_flag(-fstrict-aliasing)
    add_cxx_compiler_flag(-Wno-deprecated)
    #add_cxx_compiler_flag(-fno-exceptions)
    add_cxx_compiler_flag(-Wthread-safety)

    # Different flags for Debug vs Release-like builds
    if (CMAKE_BUILD_TYPE STREQUAL "Debug")
        add_cxx_compiler_flag(-g)
        add_cxx_compiler_flag(-O0)
        # Optional: avoid architecture-specific optimizations in debug
        # add_cxx_compiler_flag(-fno-omit-frame-pointer)
    else()
        add_cxx_compiler_flag(-O3)
        add_cxx_compiler_flag(-march=native)
        add_cxx_compiler_flag(-mtune=native)
    endif()

    if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
      find_program(GCC_AR gcc-ar)
      if (GCC_AR)
        set(CMAKE_AR ${GCC_AR})
      endif()
      find_program(GCC_RANLIB gcc-ranlib)
      if (GCC_RANLIB)
        set(CMAKE_RANLIB ${GCC_RANLIB})
      endif()
    elseif("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
      include(llvm-toolchain)
    endif()
endif()

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

add_subdirectory(lib)
add_subdirectory(cli)

cmake_minimum_required (VERSION 3.9)

set(VERSION "0.0.1")

file(GLOB
    LIB_FILES
    *.cpp
    *.h)

# Add CUDA source explicitly
set(LIB_CUDA_FILES
    MatchManyGPU.cu
)

include_directories(${PROJECT_SOURCE_DIR}/3rdparty)
include_directories(${PROJECT_SOURCE_DIR}/lib)
add_library(openafis STATIC ${LIB_FILES} ${LIB_CUDA_FILES})

set_target_properties(openafis PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

target_include_directories(openafis PRIVATE
    ${PROJECT_SOURCE_DIR}/3rdparty
    ${PROJECT_SOURCE_DIR}/lib
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

target_link_libraries(openafis PUBLIC taskflow)

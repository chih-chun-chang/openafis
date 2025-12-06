cmake_minimum_required (VERSION 3.9)

set(VERSION "0.0.1")

enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(CXXFeatureCheck)
cxx_feature_check(STEADY_CLOCK)

include_directories(${PROJECT_SOURCE_DIR}/lib)
add_executable(openafis-cli Main.cpp)
# Include lib/ headers AND CUDA toolkit headers for the CLI target
target_include_directories(openafis-cli PRIVATE
    ${PROJECT_SOURCE_DIR}/lib
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

target_link_libraries(openafis-cli PRIVATE 
    openafis
    taskflow
    Threads::Threads 
    CUDA::cudart
)

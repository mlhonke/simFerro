cmake_minimum_required(VERSION 3.12)

SET(CMAKE_C_COMPILER /usr/bin/gcc-8)
SET(CMAKE_CXX_COMPILER /usr/bin/g++-8)

project(simFerro LANGUAGES C CXX)

# Load the libSimFluid library.
message("Loading the libSimFluid library.")
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/external/libSimFluid)

# Points CMAKE to directory containing FindLIBIGL.cmake and other CMAKE find files.
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# Find and prepare to use CUDA
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
enable_language(CUDA)
include(CheckLanguage)
check_language(CUDA)

include_directories(/usr/local/cuda/samples/common/inc/)
#include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

set(CMAKE_CXX_STANDARD 11)
SET(CMAKE_CXX_FLAGS "-std=c++11 -g -fopenmp")
SET(CMAKE_EXE_LINKER_FLAGS  "-lX11 -lXi -lXrandr -lXxf86vm -lXinerama -lXcursor -lrt -lm -L/usr/local/cuda/lib64 -L/usr/local/cuda/lib64/cublas -L/usr/local/cuda/lib64/libcusparse.so -L/usr/local/cuda/lib64/culibos -L/usr/local/cuda/lib64/cudart -ldl")
set(CUDA_SEPARATE_COMPILATION ON)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -ccbin gcc-8 -arch=sm_60")
#set_directory_properties( PROPERTIES COMPILE_DEFINITIONS "" )

find_package(Armadillo REQUIRED QUIET)
include_directories(${ARMADILLO_INCLUDE_DIRS})

file(GLOB simFerroHeaders ${CMAKE_CURRENT_SOURCE_DIR}/include/*)
file(GLOB simFerroSources ${CMAKE_CURRENT_SOURCE_DIR}/src/*)
add_library(simFerro STATIC
        ${simFerroHeaders}
        ${simFerroSources})
target_include_directories(simFerro PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(simFerro PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/external/libSimFluid/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/external/libSimFluid/includeCUDA)
target_link_libraries(simFerro simFluid)

target_link_libraries(simFerro Eigen3::Eigen)

find_library(CUDART_LIBRARY cudart ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
target_include_directories(simFerro PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(simFerro ${CUDART_LIBRARY})

find_package(CUDA)
set_target_properties(simFerro PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(simFerro PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_link_libraries(simFerro ${CUDA_LIBRARY}) # unnecessary?
target_link_libraries(simFerro ${CUDA_cusparse_LIBRARY})
target_link_libraries(simFerro ${CUDA_cublas_LIBRARY})

target_link_libraries(simFerro ${ARMADILLO_LIBRARIES})
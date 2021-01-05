cmake_minimum_required(VERSION 3.12)
project(de_c_pure)

list(APPEND CMAKE_PREFIX_PATH "/usr/lib")
list(APPEND CMAKE_PREFIX_PATH "/opt/libtorch/share/cmake/Torch") 

find_package(Armadillo REQUIRED)
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)
find_package(Torch REQUIRED)
include_directories(${ARMADILLO_INCLUDE_DIRS})

set(CMAKE_CXX_STANDARD 20)

add_executable(de_c_pure main.cpp lattice_base.h analysis.h)
target_link_libraries(de_c_pure ${LAPACK_LIBRARIES} ${BLAS_LIBARIES} ${ARMADILLO_LIBRARIES} ${TORCH_LIBRARIES})

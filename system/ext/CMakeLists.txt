# Note: This cmake list serve only as linting purpose in Clion.
project(ext LANGUAGES CXX CUDA)
cmake_minimum_required(VERSION 3.8)


set(PYTHON_INCLUDE_DIR /usr/include/python3.8)
set(PYTORCH_INCLUDE_DIR /usr/local/lib/python3.8/dist-packages/torch/include)

function(PYTORCH_EXT LIB_NAME LIB_PATH)
    file(GLOB LIB_SOURCES ${LIB_PATH}/*.cpp ${LIB_PATH}/*.cu ${LIB_PATH}/*.cuh)
    add_library(${LIB_NAME} ${LIB_SOURCES})
    target_include_directories(${LIB_NAME} PRIVATE
            ${PYTHON_INCLUDE_DIR}
            ${PYTORCH_INCLUDE_DIR}
            ${PYTORCH_INCLUDE_DIR}/torch/csrc/api/include
            ${PYTORCH_INCLUDE_DIR}/TH
            ${PYTORCH_INCLUDE_DIR}/THC
    )
endfunction()

pytorch_ext(imgproc imgproc/)
pytorch_ext(marching_cubes marching_cubes/)
pytorch_ext(indexing indexing/)
pytorch_ext(pcproc pcproc/)

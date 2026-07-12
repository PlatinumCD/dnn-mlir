cmake_minimum_required(VERSION 3.22)

project(dnn-mlir VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(MLIR REQUIRED CONFIG)
message(STATUS "Found MLIR at: ${MLIR_DIR}")
message(STATUS "Found LLVM at: ${LLVM_DIR}")

list(APPEND CMAKE_MODULE_PATH
  "${MLIR_CMAKE_DIR}"
  "${LLVM_CMAKE_DIR}"
  "${PROJECT_SOURCE_DIR}/cmake"
)
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)
include(TorchMLIR)

set(LLVM_RUNTIME_OUTPUT_INTDIR "${CMAKE_BINARY_DIR}/bin")
set(LLVM_LIBRARY_OUTPUT_INTDIR "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

include_directories(SYSTEM ${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})
add_definitions(${LLVM_DEFINITIONS})
include_directories("${PROJECT_SOURCE_DIR}/include")
include_directories("${PROJECT_BINARY_DIR}/include")

dnn_mlir_configure_torch_mlir()

option(DNN_MLIR_ENABLE_TESTS "Build the DNN-MLIR lit test suite" ON)

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(tools)
if(DNN_MLIR_ENABLE_TESTS)
  add_subdirectory(test)
endif()

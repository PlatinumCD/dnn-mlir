cmake_minimum_required(VERSION 3.22)

project(dnn-mlir VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(MLIR REQUIRED CONFIG)
message(STATUS "Found MLIR at: ${MLIR_DIR}")
message(STATUS "Found LLVM at: ${LLVM_DIR}")

list(APPEND CMAKE_MODULE_PATH ${MLIR_CMAKE_DIR} ${LLVM_CMAKE_DIR})
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

# Keep every generated tool and library in predictable top-level directories.
set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

include_directories(SYSTEM ${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})
add_definitions(${LLVM_DEFINITIONS})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include)

set(DNN_MLIR_TORCH_MLIR_SOURCE_DIR
    "${PROJECT_SOURCE_DIR}/../torch-mlir"
    CACHE PATH "Path to the Torch-MLIR source tree")
set(DNN_MLIR_TORCH_MLIR_BINARY_DIR
    "${PROJECT_SOURCE_DIR}/../build-torch-mlir"
    CACHE PATH "Path to the Torch-MLIR build tree")

set(_torch_source_include "${DNN_MLIR_TORCH_MLIR_SOURCE_DIR}/include")
set(_torch_binary_include
    "${DNN_MLIR_TORCH_MLIR_BINARY_DIR}/tools/torch-mlir/include")
set(_torch_library_dir "${DNN_MLIR_TORCH_MLIR_BINARY_DIR}/lib")

foreach(_required_path
    "${_torch_source_include}/torch-mlir/Dialect/Torch/IR/TorchOps.h"
    "${_torch_binary_include}/torch-mlir/Dialect/Torch/IR/TorchOps.h.inc"
    "${_torch_library_dir}/libTorchMLIRTorchDialect.a"
    "${_torch_library_dir}/libTorchMLIRTorchUtils.a"
    "${_torch_library_dir}/libTorchMLIRTorchPasses.a"
    "${_torch_library_dir}/libTorchMLIRTorchOnnxToTorch.a"
    "${_torch_library_dir}/libTorchMLIRConversionUtils.a"
    "${_torch_library_dir}/libTorchMLIRTorchConversionDialect.a"
    "${_torch_library_dir}/libTorchMLIRTMTensorDialect.a"
    "${_torch_library_dir}/libTorchMLIRConversionPasses.a"
    "${_torch_library_dir}/libTorchMLIRTorchConversionPasses.a"
    "${_torch_library_dir}/libTorchMLIRTorchToArith.a"
    "${_torch_library_dir}/libTorchMLIRTorchToLinalg.a"
    "${_torch_library_dir}/libTorchMLIRTorchToSCF.a"
    "${_torch_library_dir}/libTorchMLIRTorchToTensor.a"
    "${_torch_library_dir}/libTorchMLIRTorchToTMTensor.a"
    "${_torch_library_dir}/libTorchMLIRTorchConversionToMLProgram.a")
  if(NOT EXISTS "${_required_path}")
    message(FATAL_ERROR
      "Required Torch-MLIR build artifact not found: ${_required_path}")
  endif()
endforeach()

# Torch-MLIR does not currently provide a build-tree package config. Import the
# Torch-MLIR dialect, transformation, and conversion archives used by DNN-MLIR.
add_library(TorchMLIRTorchDialect STATIC IMPORTED GLOBAL)
set_target_properties(TorchMLIRTorchDialect PROPERTIES
  IMPORTED_LOCATION "${_torch_library_dir}/libTorchMLIRTorchDialect.a"
  INTERFACE_INCLUDE_DIRECTORIES
    "${_torch_source_include};${_torch_binary_include}"
  INTERFACE_LINK_LIBRARIES
    "MLIRBytecodeOpInterface;MLIRBytecodeReader;MLIRBytecodeWriter;MLIRFuncDialect;MLIRIR;MLIRSupport;MLIRControlFlowInterfaces;MLIRInferTypeOpInterface;MLIRSideEffectInterfaces;LLVMSupport")

add_library(TorchMLIRTorchConversionDialect STATIC IMPORTED GLOBAL)
set_target_properties(TorchMLIRTorchConversionDialect PROPERTIES
  IMPORTED_LOCATION
    "${_torch_library_dir}/libTorchMLIRTorchConversionDialect.a"
  INTERFACE_INCLUDE_DIRECTORIES
    "${_torch_source_include};${_torch_binary_include}"
  INTERFACE_LINK_LIBRARIES
    "TorchMLIRTorchDialect;MLIRIR;MLIRSupport;MLIRSideEffectInterfaces;LLVMSupport")

add_library(TorchMLIRTorchUtils STATIC IMPORTED GLOBAL)
set_target_properties(TorchMLIRTorchUtils PROPERTIES
  IMPORTED_LOCATION "${_torch_library_dir}/libTorchMLIRTorchUtils.a"
  INTERFACE_INCLUDE_DIRECTORIES
    "${_torch_source_include};${_torch_binary_include}"
  INTERFACE_LINK_LIBRARIES
    "TorchMLIRTorchDialect;MLIRIR;MLIRSupport;LLVMSupport")

add_library(TorchMLIRTorchOnnxToTorch STATIC IMPORTED GLOBAL)
set_target_properties(TorchMLIRTorchOnnxToTorch PROPERTIES
  IMPORTED_LOCATION "${_torch_library_dir}/libTorchMLIRTorchOnnxToTorch.a"
  INTERFACE_INCLUDE_DIRECTORIES
    "${_torch_source_include};${_torch_binary_include}"
  INTERFACE_LINK_LIBRARIES
    "TorchMLIRTorchDialect;TorchMLIRConversionUtils;MLIRIR;MLIRPass;LLVMSupport")

add_library(TorchMLIRConversionUtils STATIC IMPORTED GLOBAL)
set_target_properties(TorchMLIRConversionUtils PROPERTIES
  IMPORTED_LOCATION "${_torch_library_dir}/libTorchMLIRConversionUtils.a"
  INTERFACE_INCLUDE_DIRECTORIES
    "${_torch_source_include};${_torch_binary_include}"
  INTERFACE_LINK_LIBRARIES
    "TorchMLIRTorchDialect;MLIRArithDialect;MLIRLinalgDialect;MLIRIR;LLVMSupport")

add_library(TorchMLIRTorchPasses STATIC IMPORTED GLOBAL)
set_target_properties(TorchMLIRTorchPasses PROPERTIES
  IMPORTED_LOCATION "${_torch_library_dir}/libTorchMLIRTorchPasses.a"
  INTERFACE_INCLUDE_DIRECTORIES
    "${_torch_source_include};${_torch_binary_include}"
  INTERFACE_LINK_LIBRARIES
    "TorchMLIRTorchDialect;TorchMLIRTorchUtils;TorchMLIRTorchOnnxToTorch;MLIRIR;MLIRPass;MLIRTransforms;LLVMSupport")

add_library(TorchMLIRTMTensorDialect STATIC IMPORTED GLOBAL)
set_target_properties(TorchMLIRTMTensorDialect PROPERTIES
  IMPORTED_LOCATION "${_torch_library_dir}/libTorchMLIRTMTensorDialect.a"
  INTERFACE_INCLUDE_DIRECTORIES
    "${_torch_source_include};${_torch_binary_include}"
  INTERFACE_LINK_LIBRARIES
    "MLIRIR;MLIRInferTypeOpInterface;MLIRSideEffectInterfaces;LLVMSupport")

add_library(TorchMLIRTorchToArith STATIC IMPORTED GLOBAL)
set_target_properties(TorchMLIRTorchToArith PROPERTIES
  IMPORTED_LOCATION "${_torch_library_dir}/libTorchMLIRTorchToArith.a"
  INTERFACE_INCLUDE_DIRECTORIES
    "${_torch_source_include};${_torch_binary_include}"
  INTERFACE_LINK_LIBRARIES
    "TorchMLIRTorchDialect;MLIRFuncDialect;MLIRIR;MLIRPass;LLVMSupport")

add_library(TorchMLIRTorchToLinalg STATIC IMPORTED GLOBAL)
set_target_properties(TorchMLIRTorchToLinalg PROPERTIES
  IMPORTED_LOCATION "${_torch_library_dir}/libTorchMLIRTorchToLinalg.a"
  INTERFACE_INCLUDE_DIRECTORIES
    "${_torch_source_include};${_torch_binary_include}"
  INTERFACE_LINK_LIBRARIES
    "TorchMLIRTorchDialect;MLIRIR;MLIRLinalgDialect;MLIRMathDialect;MLIRPass;LLVMSupport")

add_library(TorchMLIRTorchToSCF STATIC IMPORTED GLOBAL)
set_target_properties(TorchMLIRTorchToSCF PROPERTIES
  IMPORTED_LOCATION "${_torch_library_dir}/libTorchMLIRTorchToSCF.a"
  INTERFACE_INCLUDE_DIRECTORIES
    "${_torch_source_include};${_torch_binary_include}"
  INTERFACE_LINK_LIBRARIES
    "TorchMLIRTorchConversionDialect;TorchMLIRTorchDialect;MLIRFuncDialect;MLIRIR;MLIRPass;MLIRSCFDialect;LLVMSupport")

add_library(TorchMLIRTorchToTensor STATIC IMPORTED GLOBAL)
set_target_properties(TorchMLIRTorchToTensor PROPERTIES
  IMPORTED_LOCATION "${_torch_library_dir}/libTorchMLIRTorchToTensor.a"
  INTERFACE_INCLUDE_DIRECTORIES
    "${_torch_source_include};${_torch_binary_include}"
  INTERFACE_LINK_LIBRARIES
    "TorchMLIRConversionUtils;TorchMLIRTorchDialect;MLIRIR;MLIRPass;MLIRTensorDialect;LLVMSupport")

add_library(TorchMLIRTorchToTMTensor STATIC IMPORTED GLOBAL)
set_target_properties(TorchMLIRTorchToTMTensor PROPERTIES
  IMPORTED_LOCATION "${_torch_library_dir}/libTorchMLIRTorchToTMTensor.a"
  INTERFACE_INCLUDE_DIRECTORIES
    "${_torch_source_include};${_torch_binary_include}"
  INTERFACE_LINK_LIBRARIES
    "TorchMLIRTMTensorDialect;TorchMLIRTorchDialect;TorchMLIRTorchUtils;MLIRIR;MLIRLinalgDialect;MLIRMathDialect;MLIRPass;LLVMSupport")

add_library(TorchMLIRTorchConversionToMLProgram STATIC IMPORTED GLOBAL)
set_target_properties(TorchMLIRTorchConversionToMLProgram PROPERTIES
  IMPORTED_LOCATION
    "${_torch_library_dir}/libTorchMLIRTorchConversionToMLProgram.a"
  INTERFACE_INCLUDE_DIRECTORIES
    "${_torch_source_include};${_torch_binary_include}"
  INTERFACE_LINK_LIBRARIES
    "TorchMLIRTorchDialect;MLIRIR;MLIRLinalgDialect;MLIRMathDialect;MLIRMLProgramDialect;MLIRPass;LLVMSupport")

add_library(TorchMLIRConversionPasses STATIC IMPORTED GLOBAL)
set_target_properties(TorchMLIRConversionPasses PROPERTIES
  IMPORTED_LOCATION "${_torch_library_dir}/libTorchMLIRConversionPasses.a"
  INTERFACE_INCLUDE_DIRECTORIES
    "${_torch_source_include};${_torch_binary_include}"
  INTERFACE_LINK_LIBRARIES
    "TorchMLIRTorchToArith;TorchMLIRTorchToLinalg;TorchMLIRTorchToSCF;TorchMLIRTorchToTensor;TorchMLIRTorchToTMTensor;TorchMLIRTorchConversionToMLProgram;TorchMLIRConversionUtils")

add_library(TorchMLIRTorchConversionPasses STATIC IMPORTED GLOBAL)
set_target_properties(TorchMLIRTorchConversionPasses PROPERTIES
  IMPORTED_LOCATION
    "${_torch_library_dir}/libTorchMLIRTorchConversionPasses.a"
  INTERFACE_INCLUDE_DIRECTORIES
    "${_torch_source_include};${_torch_binary_include}"
  INTERFACE_LINK_LIBRARIES
    "TorchMLIRTorchConversionDialect;TorchMLIRTorchDialect;TorchMLIRTorchPasses;TorchMLIRConversionPasses;MLIRControlFlowTransforms;MLIRFuncTransforms;MLIRIR;MLIRLinalgTransforms;MLIRMemRefTransforms;MLIRPass;MLIRVectorTransforms;LLVMSupport")

option(DNN_MLIR_ENABLE_TESTS "Build the DNN-MLIR lit test suite" ON)

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(tools)
if(DNN_MLIR_ENABLE_TESTS)
  add_subdirectory(test)
endif()

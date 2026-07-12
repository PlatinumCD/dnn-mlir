include_guard(GLOBAL)

include(CMakeParseArguments)

function(_dnn_mlir_import_torch_library target archive)
  cmake_parse_arguments(ARG "" "" "LINK_LIBRARIES" ${ARGN})
  add_library(${target} STATIC IMPORTED GLOBAL)
  set_target_properties(${target} PROPERTIES
    IMPORTED_LOCATION "${_torch_library_dir}/${archive}"
    INTERFACE_INCLUDE_DIRECTORIES
      "${_torch_source_include};${_torch_binary_include}"
    INTERFACE_LINK_LIBRARIES "${ARG_LINK_LIBRARIES}"
  )
endfunction()

function(dnn_mlir_configure_torch_mlir)
  set(DNN_MLIR_TORCH_MLIR_SOURCE_DIR
      "${PROJECT_SOURCE_DIR}/externals/torch-mlir"
      CACHE PATH "Path to the Torch-MLIR source tree")

  if(NOT DEFINED DNN_MLIR_TORCH_MLIR_BINARY_DIR)
    get_filename_component(_default_torch_binary_dir
      "${MLIR_DIR}/../../.." ABSOLUTE)
    set(DNN_MLIR_TORCH_MLIR_BINARY_DIR
        "${_default_torch_binary_dir}"
        CACHE PATH "Path to the Torch-MLIR build tree")
  endif()

  if(NOT DNN_MLIR_TORCH_MLIR_BINARY_DIR)
    message(FATAL_ERROR
      "DNN_MLIR_TORCH_MLIR_BINARY_DIR must name a Torch-MLIR build tree")
  endif()

  set(_torch_source_include
      "${DNN_MLIR_TORCH_MLIR_SOURCE_DIR}/include")
  set(_torch_binary_include
      "${DNN_MLIR_TORCH_MLIR_BINARY_DIR}/tools/torch-mlir/include")
  set(_torch_library_dir "${DNN_MLIR_TORCH_MLIR_BINARY_DIR}/lib")

  set(_required_paths
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
    "${_torch_library_dir}/libTorchMLIRTorchConversionToMLProgram.a"
  )
  foreach(required_path IN LISTS _required_paths)
    if(NOT EXISTS "${required_path}")
      message(FATAL_ERROR
        "Required Torch-MLIR artifact not found: ${required_path}\n"
        "Set DNN_MLIR_TORCH_MLIR_BINARY_DIR to a compatible build tree.")
    endif()
  endforeach()

  # Torch-MLIR does not currently export a build-tree package containing these
  # targets, so DNN-MLIR defines the imported archives it directly consumes.
  _dnn_mlir_import_torch_library(
    TorchMLIRTorchDialect libTorchMLIRTorchDialect.a
    LINK_LIBRARIES
      MLIRBytecodeOpInterface MLIRBytecodeReader MLIRBytecodeWriter
      MLIRFuncDialect MLIRIR MLIRSupport MLIRControlFlowInterfaces
      MLIRInferTypeOpInterface MLIRSideEffectInterfaces LLVMSupport)

  _dnn_mlir_import_torch_library(
    TorchMLIRTorchConversionDialect libTorchMLIRTorchConversionDialect.a
    LINK_LIBRARIES
      TorchMLIRTorchDialect MLIRIR MLIRSupport MLIRSideEffectInterfaces
      LLVMSupport)

  _dnn_mlir_import_torch_library(
    TorchMLIRTorchUtils libTorchMLIRTorchUtils.a
    LINK_LIBRARIES TorchMLIRTorchDialect MLIRIR MLIRSupport LLVMSupport)

  _dnn_mlir_import_torch_library(
    TorchMLIRConversionUtils libTorchMLIRConversionUtils.a
    LINK_LIBRARIES
      TorchMLIRTorchDialect MLIRArithDialect MLIRLinalgDialect MLIRIR
      LLVMSupport)

  _dnn_mlir_import_torch_library(
    TorchMLIRTorchOnnxToTorch libTorchMLIRTorchOnnxToTorch.a
    LINK_LIBRARIES
      TorchMLIRTorchDialect TorchMLIRConversionUtils MLIRIR MLIRPass
      LLVMSupport)

  _dnn_mlir_import_torch_library(
    TorchMLIRTorchPasses libTorchMLIRTorchPasses.a
    LINK_LIBRARIES
      TorchMLIRTorchDialect TorchMLIRTorchUtils TorchMLIRTorchOnnxToTorch
      MLIRIR MLIRPass MLIRTransforms LLVMSupport)

  _dnn_mlir_import_torch_library(
    TorchMLIRTMTensorDialect libTorchMLIRTMTensorDialect.a
    LINK_LIBRARIES
      MLIRIR MLIRInferTypeOpInterface MLIRSideEffectInterfaces LLVMSupport)

  _dnn_mlir_import_torch_library(
    TorchMLIRTorchToArith libTorchMLIRTorchToArith.a
    LINK_LIBRARIES
      TorchMLIRTorchDialect MLIRFuncDialect MLIRIR MLIRPass LLVMSupport)

  _dnn_mlir_import_torch_library(
    TorchMLIRTorchToLinalg libTorchMLIRTorchToLinalg.a
    LINK_LIBRARIES
      TorchMLIRTorchDialect MLIRIR MLIRLinalgDialect MLIRMathDialect
      MLIRPass LLVMSupport)

  _dnn_mlir_import_torch_library(
    TorchMLIRTorchToSCF libTorchMLIRTorchToSCF.a
    LINK_LIBRARIES
      TorchMLIRTorchConversionDialect TorchMLIRTorchDialect MLIRFuncDialect
      MLIRIR MLIRPass MLIRSCFDialect LLVMSupport)

  _dnn_mlir_import_torch_library(
    TorchMLIRTorchToTensor libTorchMLIRTorchToTensor.a
    LINK_LIBRARIES
      TorchMLIRConversionUtils TorchMLIRTorchDialect MLIRIR MLIRPass
      MLIRTensorDialect LLVMSupport)

  _dnn_mlir_import_torch_library(
    TorchMLIRTorchToTMTensor libTorchMLIRTorchToTMTensor.a
    LINK_LIBRARIES
      TorchMLIRTMTensorDialect TorchMLIRTorchDialect TorchMLIRTorchUtils
      MLIRIR MLIRLinalgDialect MLIRMathDialect MLIRPass LLVMSupport)

  _dnn_mlir_import_torch_library(
    TorchMLIRTorchConversionToMLProgram
    libTorchMLIRTorchConversionToMLProgram.a
    LINK_LIBRARIES
      TorchMLIRTorchDialect MLIRIR MLIRLinalgDialect MLIRMathDialect
      MLIRMLProgramDialect MLIRPass LLVMSupport)

  _dnn_mlir_import_torch_library(
    TorchMLIRConversionPasses libTorchMLIRConversionPasses.a
    LINK_LIBRARIES
      TorchMLIRTorchToArith TorchMLIRTorchToLinalg TorchMLIRTorchToSCF
      TorchMLIRTorchToTensor TorchMLIRTorchToTMTensor
      TorchMLIRTorchConversionToMLProgram TorchMLIRConversionUtils)

  _dnn_mlir_import_torch_library(
    TorchMLIRTorchConversionPasses libTorchMLIRTorchConversionPasses.a
    LINK_LIBRARIES
      TorchMLIRTorchConversionDialect TorchMLIRTorchDialect
      TorchMLIRTorchPasses TorchMLIRConversionPasses
      MLIRControlFlowTransforms MLIRFuncTransforms MLIRIR
      MLIRLinalgTransforms MLIRMemRefTransforms MLIRPass
      MLIRVectorTransforms LLVMSupport)
endfunction()

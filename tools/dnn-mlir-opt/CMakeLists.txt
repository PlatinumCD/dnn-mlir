add_llvm_executable(dnn-mlir-opt
  dnn-mlir-opt.cpp
)

llvm_update_compile_flags(dnn-mlir-opt)

target_link_libraries(dnn-mlir-opt PRIVATE
  DNNIR
  DNNTorchToDNN
  MLIRArithDialect
  MLIRFuncDialect
  MLIRFuncInlinerExtension
  MLIRIR
  MLIRLinalgDialect
  MLIROptLib
  MLIRParser
  MLIRPass
  MLIRSupport
  MLIRTensorDialect
  MLIRTransforms
  TorchMLIRConversionPasses
  TorchMLIRTorchConversionPasses
  TorchMLIRTMTensorDialect
  TorchMLIRTorchConversionDialect
  TorchMLIRTorchDialect
  TorchMLIRTorchPasses
  TorchMLIRTorchOnnxToTorch
  TorchMLIRConversionUtils
  TorchMLIRTorchUtils
)

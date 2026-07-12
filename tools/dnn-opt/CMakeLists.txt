add_llvm_executable(dnn-opt
  dnn-opt.cpp
)

llvm_update_compile_flags(dnn-opt)

target_link_libraries(dnn-opt PRIVATE
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

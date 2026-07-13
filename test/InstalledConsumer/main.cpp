#include "dnn-mlir/Conversion/TorchToDNN/Passes.h"
#include "dnn-mlir/Dialect/DNN/IR/DNNDialect.h"

#include "mlir/IR/MLIRContext.h"

int main() {
  mlir::MLIRContext context;
  auto *dialect = context.getOrLoadDialect<mlir::dnn::DNNDialect>();
  auto capturePass = mlir::dnn::createConvertTorchToDNNPass();
  auto verifierPass = mlir::dnn::createVerifyDNNBackendContractPass();
  if (!dialect || !capturePass || !verifierPass)
    return 1;
  return dialect->getNamespace() == "dnn" ? 0 : 1;
}

#ifndef DNN_MLIR_CONVERSION_TORCHTODNN_PIPELINES_H
#define DNN_MLIR_CONVERSION_TORCHTODNN_PIPELINES_H

#include "llvm/ADT/StringRef.h"
#include "mlir/Pass/PassManager.h"

namespace mlir::dnn {

void addTorchValueSemanticsNormalization(OpPassManager &pm,
                                         llvm::StringRef extraLibrary = {});
void registerDNNBackendToLinalgPipeline();

} // namespace mlir::dnn

#endif // DNN_MLIR_CONVERSION_TORCHTODNN_PIPELINES_H

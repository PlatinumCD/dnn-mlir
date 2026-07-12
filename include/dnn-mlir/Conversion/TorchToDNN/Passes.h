#ifndef DNN_MLIR_CONVERSION_TORCHTODNN_PASSES_H
#define DNN_MLIR_CONVERSION_TORCHTODNN_PASSES_H

#include <memory>

#include "llvm/ADT/ArrayRef.h"
#include "mlir/Pass/Pass.h"

namespace mlir::dnn {

std::unique_ptr<Pass> createConvertTorchToDNNPass();
std::unique_ptr<Pass>
createConvertTorchToDNNPass(llvm::ArrayRef<std::string> queries,
                            llvm::ArrayRef<std::string> captures);
void registerTorchToDNNPasses();

} // namespace mlir::dnn

#endif // DNN_MLIR_CONVERSION_TORCHTODNN_PASSES_H

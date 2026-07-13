#include "dnn-mlir/Conversion/TorchToDNN/Pooling/PoolingPatterns.h"

#include "dnn-mlir/Conversion/TorchToDNN/TorchToDNNPatterns.h"

namespace mlir::dnn::torch_to_dnn {

void populateMaxPool2dPattern(
    RewritePatternSet &patterns,
    llvm::ArrayRef<std::string> selectedOperations) {
  populateNamedStructuredPattern(patterns, selectedOperations,
                                 "aten.max_pool2d");
}

} // namespace mlir::dnn::torch_to_dnn

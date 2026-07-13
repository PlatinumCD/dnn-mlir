#include "dnn-mlir/Conversion/TorchToDNN/Pooling/PoolingPatterns.h"

#include "dnn-mlir/Conversion/TorchToDNN/TorchToDNNPatterns.h"

namespace mlir::dnn::torch_to_dnn {

void populateAdaptiveAvgPool2dPattern(
    RewritePatternSet &patterns,
    llvm::ArrayRef<std::string> selectedOperations) {
  populateNamedStructuredPattern(patterns, selectedOperations,
                                 "aten.adaptive_avg_pool2d");
}

} // namespace mlir::dnn::torch_to_dnn

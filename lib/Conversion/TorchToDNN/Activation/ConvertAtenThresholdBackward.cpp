#include "dnn-mlir/Conversion/TorchToDNN/Activation/ActivationPatterns.h"

namespace mlir::dnn::torch_to_dnn {

void populateThresholdBackwardPattern(
    RewritePatternSet &patterns,
    llvm::ArrayRef<std::string> selectedOperations) {
  populateNamedActivationPattern(patterns, selectedOperations, "aten.threshold_backward");
}

} // namespace mlir::dnn::torch_to_dnn


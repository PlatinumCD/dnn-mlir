#include "dnn-mlir/Conversion/TorchToDNN/Activation/ActivationPatterns.h"

namespace mlir::dnn::torch_to_dnn {

void populateSoftmaxBackwardDataPattern(
    RewritePatternSet &patterns,
    llvm::ArrayRef<std::string> selectedOperations) {
  populateNamedActivationPattern(patterns, selectedOperations, "aten._softmax_backward_data");
}

} // namespace mlir::dnn::torch_to_dnn


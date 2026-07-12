#include "dnn-mlir/Conversion/TorchToDNN/Activation/ActivationPatterns.h"

namespace mlir::dnn::torch_to_dnn {

void populateTanhPattern(
    RewritePatternSet &patterns,
    llvm::ArrayRef<std::string> selectedOperations) {
  populateNamedActivationPattern(patterns, selectedOperations, "aten.tanh");
}

} // namespace mlir::dnn::torch_to_dnn


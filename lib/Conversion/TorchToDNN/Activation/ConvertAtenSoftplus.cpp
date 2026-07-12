#include "dnn-mlir/Conversion/TorchToDNN/Activation/ActivationPatterns.h"

namespace mlir::dnn::torch_to_dnn {

void populateSoftplusPattern(
    RewritePatternSet &patterns,
    llvm::ArrayRef<std::string> selectedOperations) {
  populateNamedActivationPattern(patterns, selectedOperations, "aten.softplus");
}

} // namespace mlir::dnn::torch_to_dnn


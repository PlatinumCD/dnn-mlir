#include "dnn-mlir/Conversion/TorchToDNN/Activation/ActivationPatterns.h"

namespace mlir::dnn::torch_to_dnn {

void populateSiluPattern(
    RewritePatternSet &patterns,
    llvm::ArrayRef<std::string> selectedOperations) {
  populateNamedActivationPattern(patterns, selectedOperations, "aten.silu");
}

} // namespace mlir::dnn::torch_to_dnn


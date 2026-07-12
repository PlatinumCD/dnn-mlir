#include "dnn-mlir/Conversion/TorchToDNN/Recurrent/RecurrentPatterns.h"

namespace mlir::dnn::torch_to_dnn {

// Uses the same input-schema operand mapping as rnn_tanh.input. Only the
// recurrent activation parameter differs.
void populateAtenRnnReluInputPattern(
    RewritePatternSet &patterns,
    llvm::ArrayRef<std::string> selectedOperations) {
  populateNamedRnnPattern(patterns, selectedOperations,
                          "aten.rnn_relu.input", "relu");
}

} // namespace mlir::dnn::torch_to_dnn

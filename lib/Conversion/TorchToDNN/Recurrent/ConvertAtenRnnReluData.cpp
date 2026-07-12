#include "dnn-mlir/Conversion/TorchToDNN/Recurrent/RecurrentPatterns.h"

namespace mlir::dnn::torch_to_dnn {

// Uses the same packed-data operand mapping as rnn_tanh.data. Only the
// recurrent activation parameter differs.
void populateAtenRnnReluDataPattern(
    RewritePatternSet &patterns,
    llvm::ArrayRef<std::string> selectedOperations) {
  populateNamedRnnPattern(patterns, selectedOperations,
                          "aten.rnn_relu.data", "relu");
}

} // namespace mlir::dnn::torch_to_dnn

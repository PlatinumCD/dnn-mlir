#include "dnn-mlir/Conversion/TorchToDNN/Recurrent/RecurrentPatterns.h"

namespace mlir::dnn::torch_to_dnn {

// aten::rnn_tanh.input operands:
//   0 input, 1 hx, 2 params, 3 has_biases, 4 num_layers, 5 dropout,
//   6 train, 7 bidirectional, 8 batch_first.
// A one-layer forward RNN with bias produces operand groups
// [1, 1, 4, 0, 0, 0, 0, 0, 0] and parameter indices [3, 4, 5, 6, 7, 8].
void populateAtenRnnTanhInputPattern(
    RewritePatternSet &patterns,
    llvm::ArrayRef<std::string> selectedOperations) {
  populateNamedRnnPattern(patterns, selectedOperations,
                          "aten.rnn_tanh.input", "tanh");
}

} // namespace mlir::dnn::torch_to_dnn

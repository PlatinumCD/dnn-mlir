#include "dnn-mlir/Conversion/TorchToDNN/Recurrent/RecurrentPatterns.h"

namespace mlir::dnn::torch_to_dnn {

// aten::rnn_tanh.data operands:
//   0 data, 1 batch_sizes, 2 hx, 3 params, 4 has_biases, 5 num_layers,
//   6 dropout, 7 train, 8 bidirectional.
// A one-layer forward packed RNN with bias produces operand groups
// [1, 1, 1, 4, 0, 0, 0, 0, 0] and parameter indices [4, 5, 6, 7, 8].
void populateAtenRnnTanhDataPattern(
    RewritePatternSet &patterns,
    llvm::ArrayRef<std::string> selectedOperations) {
  populateNamedRnnPattern(patterns, selectedOperations,
                          "aten.rnn_tanh.data", "tanh");
}

} // namespace mlir::dnn::torch_to_dnn

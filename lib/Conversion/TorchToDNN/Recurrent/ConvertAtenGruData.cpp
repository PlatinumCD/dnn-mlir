#include "dnn-mlir/Conversion/TorchToDNN/Recurrent/RecurrentPatterns.h"

namespace mlir::dnn::torch_to_dnn {

// Packed-sequence GRU schema:
//
//   aten::gru.data(
//       Tensor data, Tensor batch_sizes, Tensor hx, Tensor[] params,
//       bool has_biases, int num_layers, float dropout, bool train,
//       bool bidirectional) -> (Tensor output, Tensor h_n)
//
// Original operand indices:
//   0 data, 1 batch_sizes, 2 hx, 3 params, 4 has_biases, 5 num_layers,
//   6 dropout, 7 train, 8 bidirectional.
//
// A one-layer, forward, biased packed GRU produces:
//   operand_groups = [1, 1, 1, 4, 0, 0, 0, 0, 0]
//   parameter_indices = [4, 5, 6, 7, 8]
//
// The packed overload has no batch_first parameter. Result order is packed
// output followed by the final hidden state h_n.
void populateAtenGruDataPattern(
    RewritePatternSet &patterns,
    llvm::ArrayRef<std::string> selectedOperations) {
  populateNamedGruPattern(patterns, selectedOperations, "aten.gru.data");
}

} // namespace mlir::dnn::torch_to_dnn

#include "dnn-mlir/Conversion/TorchToDNN/Recurrent/RecurrentPatterns.h"

namespace mlir::dnn::torch_to_dnn {

// Padded-input GRU schema:
//
//   aten::gru.input(
//       Tensor input, Tensor hx, Tensor[] params, bool has_biases,
//       int num_layers, float dropout, bool train, bool bidirectional,
//       bool batch_first) -> (Tensor output, Tensor h_n)
//
// Original operand indices:
//   0 input, 1 hx, 2 params, 3 has_biases, 4 num_layers, 5 dropout,
//   6 train, 7 bidirectional, 8 batch_first.
//
// A one-layer, forward, biased GRU flattens to:
//   operands = input, h0, weight_ih, weight_hh, bias_ih, bias_hh
//   operand_groups = [1, 1, 4, 0, 0, 0, 0, 0, 0]
//   parameter_indices = [3, 4, 5, 6, 7, 8]
//
// GRU gate parameters use three hidden-size blocks. For hidden size H,
// weight and bias leading dimensions are therefore 3H.
void populateAtenGruInputPattern(
    RewritePatternSet &patterns,
    llvm::ArrayRef<std::string> selectedOperations) {
  populateNamedGruPattern(patterns, selectedOperations, "aten.gru.input");
}

} // namespace mlir::dnn::torch_to_dnn

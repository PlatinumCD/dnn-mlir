#include "dnn-mlir/Conversion/TorchToDNN/Recurrent/RecurrentPatterns.h"

namespace mlir::dnn::torch_to_dnn {

// Packed-sequence LSTM schema:
//
//   aten::lstm.data(
//       Tensor data, Tensor batch_sizes, Tensor[] hx, Tensor[] params,
//       bool has_biases, int num_layers, float dropout, bool train,
//       bool bidirectional) -> (Tensor output, Tensor h_n, Tensor c_n)
//
// Original operand indices:
//
//   0: data           Packed sequence values.
//   1: batch_sizes    Active batch size at each packed time step.
//   2: hx             Two-element list containing h0 and c0.
//   3: params         Flat parameter list ordered by layer and direction.
//   4: has_biases     Whether bias tensors are present in params.
//   5: num_layers     Number of recurrent layers.
//   6: dropout        Inter-layer dropout probability.
//   7: train          Whether training behavior is enabled.
//   8: bidirectional  Whether both directions are used.
//
// For a one-layer, forward, biased LSTM, flattening produces:
//
//   operands = data, batch_sizes, h0, c0,
//              weight_ih, weight_hh, bias_ih, bias_hh
//   operand_groups = [1, 1, 2, 4, 0, 0, 0, 0, 0]
//   parameter_indices = [4, 5, 6, 7, 8]
//
// Unlike aten.lstm.input, the packed-data overload has no batch_first
// parameter. Result order remains output, h_n, c_n.
void populateAtenLstmDataPattern(
    RewritePatternSet &patterns,
    llvm::ArrayRef<std::string> selectedOperations) {
  populateNamedLstmPattern(patterns, selectedOperations, "aten.lstm.data");
}

} // namespace mlir::dnn::torch_to_dnn

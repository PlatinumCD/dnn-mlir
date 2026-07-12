#include "dnn-mlir/Conversion/TorchToDNN/Recurrent/RecurrentPatterns.h"

namespace mlir::dnn::torch_to_dnn {

// FX imports the following ATen schema as
// `torch.operator "torch.aten.lstm.input"`:
//
//   aten::lstm.input(
//       Tensor input, Tensor[] hx, Tensor[] params, bool has_biases,
//       int num_layers, float dropout, bool train, bool bidirectional,
//       bool batch_first) -> (Tensor output, Tensor h_n, Tensor c_n)
//
// Original operand indices:
//
//   0: input          Input sequence tensor.
//   1: hx             Two-element list: initial hidden state h0 and initial
//                     cell state c0.
//   2: params         Flat parameter list ordered by layer and direction.
//                     A one-layer, forward, biased LSTM normally contains
//                     weight_ih, weight_hh, bias_ih, and bias_hh. Additional
//                     layers, reverse directions, or projection weights add
//                     more tensors to this list.
//   3: has_biases     Whether bias tensors are present in params.
//   4: num_layers     Number of recurrent layers.
//   5: dropout        Inter-layer dropout probability.
//   6: train          Whether training behavior is enabled.
//   7: bidirectional  Whether both forward and reverse directions are used.
//   8: batch_first    Whether input/output use [batch, sequence, feature].
//
// The conversion flattens tensor lists into dnn.lstm operands. For the common
// one-layer, forward, biased case, the flattened operand order is:
//
//   input, h0, c0, weight_ih, weight_hh, bias_ih, bias_hh
//
// `operand_groups` contains one entry per original operand. Its value is the
// number of flattened tensor operands contributed by that operand. The common
// case therefore produces:
//
//   operand_groups = [1, 2, 4, 0, 0, 0, 0, 0, 0]
//
// The params group is computed from the actual list length; it is not always
// four. Scalar operands do not become SSA operands. They are recorded using
// their zero-based source indices and values:
//
//   parameter_indices = [3, 4, 5, 6, 7, 8]
//   parameters = [has_biases, num_layers, dropout, train,
//                 bidirectional, batch_first]
//
// The three dnn.lstm results preserve ATen result order: output, h_n, c_n.
void populateAtenLstmPattern(
    RewritePatternSet &patterns,
    llvm::ArrayRef<std::string> selectedOperations) {
  populateNamedLstmPattern(patterns, selectedOperations, "aten.lstm.input");
}

} // namespace mlir::dnn::torch_to_dnn

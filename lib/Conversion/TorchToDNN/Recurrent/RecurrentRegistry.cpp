#include "dnn-mlir/Conversion/TorchToDNN/Recurrent/RecurrentPatterns.h"

#include "dnn-mlir/Conversion/TorchToDNN/TorchToDNNPatterns.h"

using namespace mlir;

namespace mlir::dnn::torch_to_dnn {

void populateRecurrentPatterns(RewritePatternSet &patterns,
                               ArrayRef<std::string> selectedOperations) {
#define DNN_REGISTER_RECURRENT(Name, Operation, Capture) \
  populate##Name##Pattern(patterns, selectedOperations);
  DNN_RECURRENT_PATTERN_LIST(DNN_REGISTER_RECURRENT)
#undef DNN_REGISTER_RECURRENT
}

} // namespace mlir::dnn::torch_to_dnn

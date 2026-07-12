#include "dnn-mlir/Conversion/TorchToDNN/Activation/ActivationPatterns.h"

#include "dnn-mlir/Conversion/TorchToDNN/TorchToDNNPatterns.h"

using namespace mlir;

namespace mlir::dnn::torch_to_dnn {

void populateActivationPatterns(RewritePatternSet &patterns,
                                ArrayRef<std::string> selectedOperations) {
#define DNN_REGISTER_ACTIVATION(Name, Operation, Target) \
  populate##Name##Pattern(patterns, selectedOperations);
  DNN_ACTIVATION_PATTERN_LIST(DNN_REGISTER_ACTIVATION)
#undef DNN_REGISTER_ACTIVATION
}

} // namespace mlir::dnn::torch_to_dnn

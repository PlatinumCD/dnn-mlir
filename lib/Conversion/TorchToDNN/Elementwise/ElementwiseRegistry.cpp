#include "dnn-mlir/Conversion/TorchToDNN/Elementwise/ElementwisePatterns.h"

#include "dnn-mlir/Conversion/TorchToDNN/TorchToDNNPatterns.h"

using namespace mlir;

namespace mlir::dnn::torch_to_dnn {

void populateElementwisePatterns(RewritePatternSet &patterns,
                                 ArrayRef<std::string> selectedOperations) {
#define DNN_REGISTER_ELEMENTWISE(Name, Operation, Capture) \
  populate##Name##Pattern(patterns, selectedOperations);
  DNN_ELEMENTWISE_PATTERN_LIST(DNN_REGISTER_ELEMENTWISE)
#undef DNN_REGISTER_ELEMENTWISE
}

} // namespace mlir::dnn::torch_to_dnn

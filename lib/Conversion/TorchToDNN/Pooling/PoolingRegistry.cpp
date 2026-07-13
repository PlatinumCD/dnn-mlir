#include "dnn-mlir/Conversion/TorchToDNN/Pooling/PoolingPatterns.h"

#include "dnn-mlir/Conversion/TorchToDNN/TorchToDNNPatterns.h"

using namespace mlir;

namespace mlir::dnn::torch_to_dnn {

void populatePoolingPatterns(RewritePatternSet &patterns,
                             ArrayRef<std::string> selectedOperations) {
#define DNN_REGISTER_POOLING(Name, Operation, Capture) \
  populate##Name##Pattern(patterns, selectedOperations);
  DNN_POOLING_PATTERN_LIST(DNN_REGISTER_POOLING)
#undef DNN_REGISTER_POOLING
}

} // namespace mlir::dnn::torch_to_dnn

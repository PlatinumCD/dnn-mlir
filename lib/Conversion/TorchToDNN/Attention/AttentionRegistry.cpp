#include "dnn-mlir/Conversion/TorchToDNN/Attention/AttentionPatterns.h"

#include "dnn-mlir/Conversion/TorchToDNN/TorchToDNNPatterns.h"

using namespace mlir;

namespace mlir::dnn::torch_to_dnn {

void populateAttentionPatterns(RewritePatternSet &patterns,
                               ArrayRef<std::string> selectedOperations) {
#define DNN_REGISTER_ATTENTION(Name, Operation, Capture) \
  populate##Name##Pattern(patterns, selectedOperations);
  DNN_ATTENTION_PATTERN_LIST(DNN_REGISTER_ATTENTION)
#undef DNN_REGISTER_ATTENTION
}

} // namespace mlir::dnn::torch_to_dnn

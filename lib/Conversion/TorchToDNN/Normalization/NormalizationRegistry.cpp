#include "dnn-mlir/Conversion/TorchToDNN/Normalization/NormalizationPatterns.h"

#include "dnn-mlir/Conversion/TorchToDNN/TorchToDNNPatterns.h"

using namespace mlir;

namespace mlir::dnn::torch_to_dnn {

void populateNormalizationPatterns(RewritePatternSet &patterns,
                                   ArrayRef<std::string> selectedOperations) {
#define DNN_REGISTER_NORMALIZATION(Name, Operation, Capture) \
  populate##Name##Pattern(patterns, selectedOperations);
  DNN_NORMALIZATION_PATTERN_LIST(DNN_REGISTER_NORMALIZATION)
#undef DNN_REGISTER_NORMALIZATION
}

} // namespace mlir::dnn::torch_to_dnn

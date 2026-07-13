#include "dnn-mlir/Conversion/TorchToDNN/Normalization/NormalizationPatterns.h"

#include "dnn-mlir/Conversion/TorchToDNN/TorchToDNNPatterns.h"

namespace mlir::dnn::torch_to_dnn {

void populateBatchNormPattern(
    RewritePatternSet &patterns,
    llvm::ArrayRef<std::string> selectedOperations) {
  populateNamedStructuredPattern(patterns, selectedOperations,
                                 "aten.batch_norm", true);
}

} // namespace mlir::dnn::torch_to_dnn

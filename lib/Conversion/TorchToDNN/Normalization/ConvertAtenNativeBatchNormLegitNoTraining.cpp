#include "dnn-mlir/Conversion/TorchToDNN/Normalization/NormalizationPatterns.h"

#include "dnn-mlir/Conversion/TorchToDNN/TorchToDNNPatterns.h"

namespace mlir::dnn::torch_to_dnn {

void populateNativeBatchNormLegitNoTrainingPattern(
    RewritePatternSet &patterns,
    llvm::ArrayRef<std::string> selectedOperations) {
  populateNamedStructuredOperatorPattern(
      patterns, selectedOperations,
      "aten._native_batch_norm_legit_no_training", true);
}

} // namespace mlir::dnn::torch_to_dnn

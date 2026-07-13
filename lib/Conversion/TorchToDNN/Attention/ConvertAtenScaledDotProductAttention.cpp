#include "dnn-mlir/Conversion/TorchToDNN/Attention/AttentionPatterns.h"

#include "dnn-mlir/Conversion/TorchToDNN/TorchToDNNPatterns.h"

namespace mlir::dnn::torch_to_dnn {

void populateScaledDotProductAttentionPattern(
    RewritePatternSet &patterns,
    llvm::ArrayRef<std::string> selectedOperations) {
  populateNamedStructuredPattern(patterns, selectedOperations,
                                 "aten.scaled_dot_product_attention");
}

} // namespace mlir::dnn::torch_to_dnn

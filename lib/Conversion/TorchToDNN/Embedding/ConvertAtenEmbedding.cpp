#include "dnn-mlir/Conversion/TorchToDNN/Embedding/EmbeddingPatterns.h"

#include "dnn-mlir/Conversion/TorchToDNN/TorchToDNNPatterns.h"

namespace mlir::dnn::torch_to_dnn {

void populateEmbeddingPattern(
    RewritePatternSet &patterns,
    llvm::ArrayRef<std::string> selectedOperations) {
  populateNamedStructuredPattern(patterns, selectedOperations,
                                 "aten.embedding");
}

} // namespace mlir::dnn::torch_to_dnn

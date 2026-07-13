#include "dnn-mlir/Conversion/TorchToDNN/Embedding/EmbeddingPatterns.h"

#include "dnn-mlir/Conversion/TorchToDNN/TorchToDNNPatterns.h"

using namespace mlir;

namespace mlir::dnn::torch_to_dnn {

void populateEmbeddingPatterns(RewritePatternSet &patterns,
                               ArrayRef<std::string> selectedOperations) {
#define DNN_REGISTER_EMBEDDING(Name, Operation, Capture) \
  populate##Name##Pattern(patterns, selectedOperations);
  DNN_EMBEDDING_PATTERN_LIST(DNN_REGISTER_EMBEDDING)
#undef DNN_REGISTER_EMBEDDING
}

} // namespace mlir::dnn::torch_to_dnn

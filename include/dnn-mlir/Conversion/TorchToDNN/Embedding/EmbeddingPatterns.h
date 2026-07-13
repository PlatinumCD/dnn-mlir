#ifndef DNN_MLIR_CONVERSION_TORCHTODNN_EMBEDDING_PATTERNS_H
#define DNN_MLIR_CONVERSION_TORCHTODNN_EMBEDDING_PATTERNS_H

#include "llvm/ADT/ArrayRef.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir::dnn::torch_to_dnn {

#define DNN_EMBEDDING_PATTERN_LIST(X) \
  X(Embedding, "aten.embedding", "dnn.embedding")

#define DNN_DECLARE_EMBEDDING_PATTERN(Name, Operation, Capture) \
  void populate##Name##Pattern(RewritePatternSet &, llvm::ArrayRef<std::string>);
DNN_EMBEDDING_PATTERN_LIST(DNN_DECLARE_EMBEDDING_PATTERN)
#undef DNN_DECLARE_EMBEDDING_PATTERN

} // namespace mlir::dnn::torch_to_dnn

#endif

#ifndef DNN_MLIR_CONVERSION_TORCHTODNN_ATTENTION_PATTERNS_H
#define DNN_MLIR_CONVERSION_TORCHTODNN_ATTENTION_PATTERNS_H

#include "llvm/ADT/ArrayRef.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir::dnn::torch_to_dnn {

#define DNN_ATTENTION_PATTERN_LIST(X) \
  X(ScaledDotProductAttention, "aten.scaled_dot_product_attention", \
    "dnn.scaled_dot_product_attention")

#define DNN_DECLARE_ATTENTION_PATTERN(Name, Operation, Capture) \
  void populate##Name##Pattern(RewritePatternSet &, llvm::ArrayRef<std::string>);
DNN_ATTENTION_PATTERN_LIST(DNN_DECLARE_ATTENTION_PATTERN)
#undef DNN_DECLARE_ATTENTION_PATTERN

} // namespace mlir::dnn::torch_to_dnn

#endif

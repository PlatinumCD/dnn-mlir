#ifndef DNN_MLIR_CONVERSION_TORCHTODNN_POOLING_PATTERNS_H
#define DNN_MLIR_CONVERSION_TORCHTODNN_POOLING_PATTERNS_H

#include "llvm/ADT/ArrayRef.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir::dnn::torch_to_dnn {

#define DNN_POOLING_PATTERN_LIST(X) \
  X(MaxPool2d, "aten.max_pool2d", "dnn.max_pool2d") \
  X(AdaptiveAvgPool2d, "aten.adaptive_avg_pool2d", \
    "dnn.adaptive_avg_pool2d")

#define DNN_DECLARE_POOLING_PATTERN(Name, Operation, Capture) \
  void populate##Name##Pattern(RewritePatternSet &, llvm::ArrayRef<std::string>);
DNN_POOLING_PATTERN_LIST(DNN_DECLARE_POOLING_PATTERN)
#undef DNN_DECLARE_POOLING_PATTERN

} // namespace mlir::dnn::torch_to_dnn

#endif

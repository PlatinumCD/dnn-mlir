#ifndef DNN_MLIR_CONVERSION_TORCHTODNN_SHAPE_PATTERNS_H
#define DNN_MLIR_CONVERSION_TORCHTODNN_SHAPE_PATTERNS_H

#include "llvm/ADT/ArrayRef.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir::dnn::torch_to_dnn {

#define DNN_SHAPE_PATTERN_LIST(X) \
  X(FlattenUsingInts, "aten.flatten.using_ints", "dnn.flatten")

#define DNN_DECLARE_SHAPE_PATTERN(Name, Operation, Capture) \
  void populate##Name##Pattern(RewritePatternSet &, llvm::ArrayRef<std::string>);
DNN_SHAPE_PATTERN_LIST(DNN_DECLARE_SHAPE_PATTERN)
#undef DNN_DECLARE_SHAPE_PATTERN

} // namespace mlir::dnn::torch_to_dnn

#endif

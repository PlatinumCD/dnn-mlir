#ifndef DNN_MLIR_CONVERSION_TORCHTODNN_ELEMENTWISE_PATTERNS_H
#define DNN_MLIR_CONVERSION_TORCHTODNN_ELEMENTWISE_PATTERNS_H

#include "llvm/ADT/ArrayRef.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir::dnn::torch_to_dnn {

#define DNN_ELEMENTWISE_PATTERN_LIST(X) \
  X(AddTensor, "aten.add.Tensor", "dnn.add") \
  X(MulTensor, "aten.mul.Tensor", "dnn.mul")

#define DNN_DECLARE_ELEMENTWISE_PATTERN(Name, Operation, Capture) \
  void populate##Name##Pattern(RewritePatternSet &, llvm::ArrayRef<std::string>);
DNN_ELEMENTWISE_PATTERN_LIST(DNN_DECLARE_ELEMENTWISE_PATTERN)
#undef DNN_DECLARE_ELEMENTWISE_PATTERN

} // namespace mlir::dnn::torch_to_dnn

#endif

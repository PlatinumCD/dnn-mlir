#ifndef DNN_MLIR_CONVERSION_TORCHTODNN_NORMALIZATION_PATTERNS_H
#define DNN_MLIR_CONVERSION_TORCHTODNN_NORMALIZATION_PATTERNS_H

#include "llvm/ADT/ArrayRef.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir::dnn::torch_to_dnn {

#define DNN_NORMALIZATION_PATTERN_LIST(X) \
  X(BatchNorm, "aten.batch_norm", "dnn.batch_norm") \
  X(NativeBatchNorm, "aten.native_batch_norm", "dnn.batch_norm") \
  X(NativeBatchNormLegitNoTraining, \
    "aten._native_batch_norm_legit_no_training", "dnn.batch_norm") \
  X(LayerNorm, "aten.layer_norm", "dnn.layer_norm")

#define DNN_DECLARE_NORMALIZATION_PATTERN(Name, Operation, Capture) \
  void populate##Name##Pattern(RewritePatternSet &, llvm::ArrayRef<std::string>);
DNN_NORMALIZATION_PATTERN_LIST(DNN_DECLARE_NORMALIZATION_PATTERN)
#undef DNN_DECLARE_NORMALIZATION_PATTERN

void populateDecomposedLayerNormFusionPatterns(
    RewritePatternSet &patterns,
    llvm::ArrayRef<std::string> selectedOperations);

} // namespace mlir::dnn::torch_to_dnn

#endif

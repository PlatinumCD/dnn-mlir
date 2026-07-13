#ifndef DNN_MLIR_CONVERSION_TORCHTODNN_MATRIX_MATRIXPATTERNS_H
#define DNN_MLIR_CONVERSION_TORCHTODNN_MATRIX_MATRIXPATTERNS_H

#include "llvm/ADT/ArrayRef.h"
#include "mlir/IR/PatternMatch.h"

#define DNN_MATRIX_PATTERN_LIST(X)                                          \
  X(Mm, "aten.mm", "dnn.mm")                                              \
  X(Matmul, "aten.matmul", "dnn.matmul")

namespace mlir::dnn::torch_to_dnn {

void populateAtenMmPattern(RewritePatternSet &patterns,
                           llvm::ArrayRef<std::string> selectedOperations);
void populateAtenMatmulPattern(
    RewritePatternSet &patterns,
    llvm::ArrayRef<std::string> selectedOperations);

} // namespace mlir::dnn::torch_to_dnn

#endif // DNN_MLIR_CONVERSION_TORCHTODNN_MATRIX_MATRIXPATTERNS_H

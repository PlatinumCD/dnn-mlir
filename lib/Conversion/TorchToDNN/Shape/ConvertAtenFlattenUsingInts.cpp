#include "dnn-mlir/Conversion/TorchToDNN/Shape/ShapePatterns.h"

#include "dnn-mlir/Conversion/TorchToDNN/TorchToDNNPatterns.h"

namespace mlir::dnn::torch_to_dnn {

void populateFlattenUsingIntsPattern(
    RewritePatternSet &patterns,
    llvm::ArrayRef<std::string> selectedOperations) {
  populateNamedStructuredPattern(patterns, selectedOperations,
                                 "aten.flatten.using_ints");
}

} // namespace mlir::dnn::torch_to_dnn

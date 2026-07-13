#include "dnn-mlir/Conversion/TorchToDNN/Elementwise/ElementwisePatterns.h"

#include "dnn-mlir/Conversion/TorchToDNN/TorchToDNNPatterns.h"

namespace mlir::dnn::torch_to_dnn {

void populateAddTensorPattern(
    RewritePatternSet &patterns,
    llvm::ArrayRef<std::string> selectedOperations) {
  populateNamedStructuredPattern(patterns, selectedOperations,
                                 "aten.add.Tensor");
}

} // namespace mlir::dnn::torch_to_dnn

#include "dnn-mlir/Conversion/TorchToDNN/Convolution/ConvolutionPatterns.h"

namespace mlir::dnn::torch_to_dnn {

void populateConvTranspose1dPattern(
    RewritePatternSet &patterns,
    llvm::ArrayRef<std::string> selectedOperations) {
  populateNamedConvolutionPattern(patterns, selectedOperations, "aten.conv_transpose1d");
}

} // namespace mlir::dnn::torch_to_dnn


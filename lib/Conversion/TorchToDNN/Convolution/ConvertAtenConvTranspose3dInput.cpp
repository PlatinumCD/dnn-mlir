#include "dnn-mlir/Conversion/TorchToDNN/Convolution/ConvolutionPatterns.h"

namespace mlir::dnn::torch_to_dnn {

void populateConvTranspose3dInputPattern(
    RewritePatternSet &patterns,
    llvm::ArrayRef<std::string> selectedOperations) {
  populateNamedConvolutionPattern(patterns, selectedOperations, "aten.conv_transpose3d.input");
}

} // namespace mlir::dnn::torch_to_dnn


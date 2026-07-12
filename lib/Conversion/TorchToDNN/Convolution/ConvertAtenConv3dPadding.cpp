#include "dnn-mlir/Conversion/TorchToDNN/Convolution/ConvolutionPatterns.h"

namespace mlir::dnn::torch_to_dnn {

void populateConv3dPaddingPattern(
    RewritePatternSet &patterns,
    llvm::ArrayRef<std::string> selectedOperations) {
  populateNamedConvolutionPattern(patterns, selectedOperations, "aten.conv3d.padding");
}

} // namespace mlir::dnn::torch_to_dnn


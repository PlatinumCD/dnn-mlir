#include "dnn-mlir/Conversion/TorchToDNN/Matrix/MatrixPatterns.h"
#include "dnn-mlir/Conversion/TorchToDNN/TorchToDNNPatterns.h"

using namespace mlir;

namespace mlir::dnn::torch_to_dnn {

void populateMatrixPatterns(RewritePatternSet &patterns,
                            ArrayRef<std::string> selectedOperations) {
  populateAtenMmPattern(patterns, selectedOperations);
  populateAtenMatmulPattern(patterns, selectedOperations);
}

} // namespace mlir::dnn::torch_to_dnn

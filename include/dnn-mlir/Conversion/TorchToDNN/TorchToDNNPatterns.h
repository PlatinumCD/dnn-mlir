#ifndef DNN_MLIR_CONVERSION_TORCHTODNN_TORCHTODNNPATTERNS_H
#define DNN_MLIR_CONVERSION_TORCHTODNN_TORCHTODNNPATTERNS_H

#include <optional>
#include <string>

#include "llvm/ADT/ArrayRef.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "torch-mlir/Dialect/Torch/IR/TorchTypes.h"

namespace mlir::dnn::torch_to_dnn {

struct TensorBridgeType {
  Type originalType;
  torch::Torch::ValueTensorType valueTensorType;
  RankedTensorType builtinType;
};

std::optional<TensorBridgeType> getTensorBridgeType(Type type);

std::string canonicalizeOperationName(llvm::StringRef operation);
bool isOperationSelected(llvm::ArrayRef<std::string> selectedOperations,
                         llvm::StringRef operation);
bool hasValueSemanticsForCapture(Operation *operation);
bool hasSupportedValuesForCapture(Operation *operation);
std::optional<Attribute> getConstantTorchAttribute(Value value);

Value materializeBuiltinTensor(PatternRewriter &rewriter, Location loc,
                               Value value,
                               const TensorBridgeType &bridgeType);

Value materializeTorchTensor(PatternRewriter &rewriter, Location loc,
                             Value value,
                             const TensorBridgeType &bridgeType);

void populateNamedStructuredPattern(
    RewritePatternSet &patterns,
    llvm::ArrayRef<std::string> selectedOperations,
    llvm::StringRef operationName, bool preserveKind = false);
void populateNamedStructuredOperatorPattern(
    RewritePatternSet &patterns,
    llvm::ArrayRef<std::string> selectedOperations,
    llvm::StringRef operationName, bool preserveKind = false);

void populateMatrixPatterns(RewritePatternSet &patterns,
                            llvm::ArrayRef<std::string> selectedOperations);
void populateAffinePatterns(RewritePatternSet &patterns,
                            llvm::ArrayRef<std::string> selectedOperations);
void populateActivationPatterns(
    RewritePatternSet &patterns,
    llvm::ArrayRef<std::string> selectedOperations);
void populateAttentionPatterns(
    RewritePatternSet &patterns,
    llvm::ArrayRef<std::string> selectedOperations);
void populateConvolutionPatterns(
    RewritePatternSet &patterns,
    llvm::ArrayRef<std::string> selectedOperations);
void populateEmbeddingPatterns(
    RewritePatternSet &patterns,
    llvm::ArrayRef<std::string> selectedOperations);
void populateRecurrentPatterns(
    RewritePatternSet &patterns,
    llvm::ArrayRef<std::string> selectedOperations);
void populateNormalizationPatterns(
    RewritePatternSet &patterns,
    llvm::ArrayRef<std::string> selectedOperations);
void populatePoolingPatterns(
    RewritePatternSet &patterns,
    llvm::ArrayRef<std::string> selectedOperations);
void populateElementwisePatterns(
    RewritePatternSet &patterns,
    llvm::ArrayRef<std::string> selectedOperations);
void populateShapePatterns(
    RewritePatternSet &patterns,
    llvm::ArrayRef<std::string> selectedOperations);

} // namespace mlir::dnn::torch_to_dnn

#endif // DNN_MLIR_CONVERSION_TORCHTODNN_TORCHTODNNPATTERNS_H

#include "dnn-mlir/Conversion/TorchToDNN/Convolution/ConvolutionPatterns.h"

#include "dnn-mlir/Conversion/TorchToDNN/TorchToDNNPatterns.h"
#include "dnn-mlir/Dialect/DNN/IR/DNNOps.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"
#include "mlir/IR/Operation.h"

using namespace mlir;

namespace mlir::dnn::torch_to_dnn {
namespace {

class ConvertAtenConvolutionPattern : public RewritePattern {
public:
  ConvertAtenConvolutionPattern(StringRef operationName, MLIRContext *context)
      : RewritePattern((Twine("torch.") + operationName).str(), 1, context),
        operationName(operationName.str()) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    SmallVector<Value> operands;
    SmallVector<int32_t> parameterIndices;
    SmallVector<Attribute> parameters;
    operands.reserve(op->getNumOperands());
    for (auto [index, operand] : llvm::enumerate(op->getOperands())) {
      auto bridgeType = getTensorBridgeType(operand.getType());
      if (bridgeType)
        operands.push_back(materializeBuiltinTensor(
            rewriter, op->getLoc(), operand, *bridgeType));
      else if (auto attribute = getConstantTorchAttribute(operand)) {
        parameterIndices.push_back(static_cast<int32_t>(index));
        parameters.push_back(*attribute);
      }
      else
        operands.push_back(operand);
    }

    SmallVector<Type> resultTypes;
    SmallVector<std::optional<TensorBridgeType>> resultBridges;
    resultTypes.reserve(op->getNumResults());
    resultBridges.reserve(op->getNumResults());
    for (Value result : op->getResults()) {
      auto bridgeType = getTensorBridgeType(result.getType());
      if (bridgeType) {
        resultTypes.push_back(bridgeType->builtinType);
        resultBridges.push_back(std::move(bridgeType));
      } else {
        resultTypes.push_back(result.getType());
        resultBridges.push_back(std::nullopt);
      }
    }

    OperationState state(op->getLoc(), ConvolutionOp::getOperationName());
    state.addOperands(operands);
    state.addTypes(resultTypes);
    state.addAttribute("kind", rewriter.getStringAttr(operationName));
    if (!parameters.empty()) {
      state.addAttribute("parameter_indices",
                        rewriter.getI32ArrayAttr(parameterIndices));
      state.addAttribute("parameters", rewriter.getArrayAttr(parameters));
    }
    Operation *convolution = rewriter.create(state);

    SmallVector<Value> replacements;
    replacements.reserve(op->getNumResults());
    for (auto [result, bridgeType] :
         llvm::zip(convolution->getResults(), resultBridges)) {
      if (bridgeType)
        replacements.push_back(materializeTorchTensor(
            rewriter, op->getLoc(), result, *bridgeType));
      else
        replacements.push_back(result);
    }
    rewriter.replaceOp(op, replacements);
    return success();
  }

private:
  std::string operationName;
};

} // namespace

void populateNamedConvolutionPattern(
    RewritePatternSet &patterns, ArrayRef<std::string> selectedOperations,
    StringRef operationName) {
  if (!isOperationSelected(selectedOperations, operationName))
    return;
  patterns.add<ConvertAtenConvolutionPattern>(operationName,
                                              patterns.getContext());
}

} // namespace mlir::dnn::torch_to_dnn

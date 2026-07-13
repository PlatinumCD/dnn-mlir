#include "dnn-mlir/Conversion/TorchToDNN/TorchToDNNPatterns.h"

#include <cassert>

#include "dnn-mlir/Conversion/TorchToDNN/CaptureRegistry.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"
#include "mlir/IR/Operation.h"

using namespace mlir;

namespace mlir::dnn::torch_to_dnn {
namespace {

class ConvertStructuredTorchOpPattern : public RewritePattern {
public:
  ConvertStructuredTorchOpPattern(StringRef operationName,
                                  StringRef targetOperationName,
                                  bool preserveKind, bool opaqueOperator,
                                  MLIRContext *context)
      : RewritePattern(opaqueOperator
                           ? "torch.operator"
                           : (Twine("torch.") + operationName).str(),
                       1, context),
        operationName(operationName.str()),
        targetOperationName(targetOperationName.str()),
        preserveKind(preserveKind), opaqueOperator(opaqueOperator) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (!hasSupportedValuesForCapture(op))
      return rewriter.notifyMatchFailure(
          op, "DNN capture requires value-semantic, representable values");

    if (opaqueOperator) {
      auto name = op->getAttrOfType<StringAttr>("name");
      StringRef candidate = name ? name.getValue() : StringRef();
      candidate.consume_front("torch.");
      if (!name || candidate != operationName)
        return failure();
    }
    SmallVector<Value> operands;
    SmallVector<int32_t> parameterIndices;
    SmallVector<Attribute> parameters;
    operands.reserve(op->getNumOperands());
    for (auto [index, operand] : llvm::enumerate(op->getOperands())) {
      if (auto bridgeType = getTensorBridgeType(operand.getType())) {
        operands.push_back(materializeBuiltinTensor(
            rewriter, op->getLoc(), operand, *bridgeType));
      } else if (auto attribute = getConstantTorchAttribute(operand)) {
        parameterIndices.push_back(static_cast<int32_t>(index));
        parameters.push_back(*attribute);
      } else {
        operands.push_back(operand);
      }
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

    OperationState state(op->getLoc(), targetOperationName);
    state.addOperands(operands);
    state.addTypes(resultTypes);
    if (preserveKind)
      state.addAttribute("kind", rewriter.getStringAttr(operationName));
    if (!parameters.empty()) {
      state.addAttribute("parameter_indices",
                         rewriter.getDenseI32ArrayAttr(parameterIndices));
      state.addAttribute("parameters", rewriter.getArrayAttr(parameters));
    }
    Operation *converted = rewriter.create(state);

    SmallVector<Value> replacements;
    replacements.reserve(op->getNumResults());
    for (auto [result, bridgeType] :
         llvm::zip(converted->getResults(), resultBridges)) {
      replacements.push_back(bridgeType ? materializeTorchTensor(
                                             rewriter, op->getLoc(), result,
                                             *bridgeType)
                                        : result);
    }
    rewriter.replaceOp(op, replacements);
    return success();
  }

private:
  std::string operationName;
  std::string targetOperationName;
  bool preserveKind;
  bool opaqueOperator;
};

} // namespace

void populateNamedStructuredPattern(
    RewritePatternSet &patterns, ArrayRef<std::string> selectedOperations,
    StringRef operationName, bool preserveKind) {
  if (!isOperationSelected(selectedOperations, operationName))
    return;
  std::optional<StringRef> targetOperationName =
      getCaptureForQuery(operationName);
  assert(targetOperationName && "structured query is missing its capture");
  if (!targetOperationName)
    return;
  patterns.add<ConvertStructuredTorchOpPattern>(
      operationName, *targetOperationName, preserveKind,
      /*opaqueOperator=*/false, patterns.getContext());
}

void populateNamedStructuredOperatorPattern(
    RewritePatternSet &patterns, ArrayRef<std::string> selectedOperations,
    StringRef operationName, bool preserveKind) {
  if (!isOperationSelected(selectedOperations, operationName))
    return;
  std::optional<StringRef> targetOperationName =
      getCaptureForQuery(operationName);
  assert(targetOperationName && "structured query is missing its capture");
  if (!targetOperationName)
    return;
  patterns.add<ConvertStructuredTorchOpPattern>(
      operationName, *targetOperationName, preserveKind,
      /*opaqueOperator=*/true, patterns.getContext());
}

} // namespace mlir::dnn::torch_to_dnn

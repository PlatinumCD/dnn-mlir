#include "dnn-mlir/Conversion/TorchToDNN/Recurrent/RecurrentPatterns.h"

#include "dnn-mlir/Conversion/TorchToDNN/TorchToDNNPatterns.h"
#include "dnn-mlir/Dialect/DNN/IR/DNNOps.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"

using namespace mlir;

namespace mlir::dnn::torch_to_dnn {
namespace {

struct RecurrentOperandState {
  SmallVector<Value> tensorOperands;
  SmallVector<int32_t> operandGroups;
  SmallVector<int32_t> parameterIndices;
  SmallVector<Attribute> parameters;
};

enum class RecurrentKind { Lstm, Gru, Rnn };

StringRef normalizeOperationName(StringRef operation) {
  operation = operation.trim();
  if (operation.starts_with("torch."))
    operation = operation.drop_front(6);
  return operation;
}

LogicalResult appendTensorElement(Value value, PatternRewriter &rewriter,
                                  Location loc,
                                  RecurrentOperandState &state) {
  auto bridgeType = getTensorBridgeType(value.getType());
  if (!bridgeType)
    return failure();
  state.tensorOperands.push_back(
      materializeBuiltinTensor(rewriter, loc, value, *bridgeType));
  return success();
}

LogicalResult flattenOperand(Value operand, unsigned operandIndex,
                             PatternRewriter &rewriter, Location loc,
                             RecurrentOperandState &state) {
  if (auto bridgeType = getTensorBridgeType(operand.getType())) {
    state.tensorOperands.push_back(
        materializeBuiltinTensor(rewriter, loc, operand, *bridgeType));
    state.operandGroups.push_back(1);
    return success();
  }

  if (Operation *definingOp = operand.getDefiningOp()) {
    if (definingOp->getName().getStringRef() ==
        "torch.prim.ListConstruct") {
      int32_t groupSize = 0;
      for (Value element : definingOp->getOperands()) {
        if (failed(appendTensorElement(element, rewriter, loc, state)))
          return failure();
        ++groupSize;
      }
      state.operandGroups.push_back(groupSize);
      return success();
    }
  }

  if (auto attribute = getConstantTorchAttribute(operand)) {
    state.parameterIndices.push_back(static_cast<int32_t>(operandIndex));
    state.parameters.push_back(*attribute);
    state.operandGroups.push_back(0);
    return success();
  }

  return failure();
}

class ConvertRecurrentPattern : public RewritePattern {
public:
  ConvertRecurrentPattern(StringRef operationName,
                          RecurrentKind recurrentKind,
                          StringRef activation, MLIRContext *context)
      : RewritePattern("torch.operator", 1, context),
        operationName(normalizeOperationName(operationName).str()),
        recurrentKind(recurrentKind),
        activation(activation.str()) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto name = op->getAttrOfType<StringAttr>("name");
    if (!name || normalizeOperationName(name.getValue()) != operationName)
      return failure();

    RecurrentOperandState operandState;
    for (auto [index, operand] : llvm::enumerate(op->getOperands())) {
      if (failed(flattenOperand(operand, index, rewriter, op->getLoc(),
                                operandState)))
        return failure();
    }

    SmallVector<Type> resultTypes;
    SmallVector<TensorBridgeType> resultBridges;
    resultTypes.reserve(op->getNumResults());
    resultBridges.reserve(op->getNumResults());
    for (Value result : op->getResults()) {
      auto bridgeType = getTensorBridgeType(result.getType());
      if (!bridgeType)
        return failure();
      resultTypes.push_back(bridgeType->builtinType);
      resultBridges.push_back(*bridgeType);
    }

    DenseI32ArrayAttr parameterIndices;
    ArrayAttr parameters;
    if (!operandState.parameters.empty()) {
      parameterIndices =
          rewriter.getDenseI32ArrayAttr(operandState.parameterIndices);
      parameters = rewriter.getArrayAttr(operandState.parameters);
    }

    Operation *recurrent;
    if (recurrentKind == RecurrentKind::Lstm) {
      recurrent = LstmOp::create(
                      rewriter, op->getLoc(), resultTypes,
                      operandState.tensorOperands, operationName,
                      operandState.operandGroups, parameterIndices, parameters)
                      .getOperation();
    } else if (recurrentKind == RecurrentKind::Gru) {
      recurrent = GruOp::create(
                      rewriter, op->getLoc(), resultTypes,
                      operandState.tensorOperands, operationName,
                      operandState.operandGroups, parameterIndices, parameters)
                      .getOperation();
    } else {
      recurrent = RnnOp::create(
                      rewriter, op->getLoc(), resultTypes,
                      operandState.tensorOperands, activation, operationName,
                      operandState.operandGroups, parameterIndices, parameters)
                      .getOperation();
    }

    // Preserve source-specific discardable metadata on the DNN operation.
    for (NamedAttribute attribute : op->getAttrs()) {
      StringRef attributeName = attribute.getName();
      if (attributeName == "name" || attributeName == "activation" ||
          attributeName == "kind" ||
          attributeName == "operand_groups" ||
          attributeName == "parameter_indices" ||
          attributeName == "parameters")
        continue;
      recurrent->setAttr(attributeName, attribute.getValue());
    }
    SmallVector<Value> replacements;
    replacements.reserve(recurrent->getNumResults());
    for (auto [result, bridgeType] :
         llvm::zip(recurrent->getResults(), resultBridges)) {
      replacements.push_back(
          materializeTorchTensor(rewriter, op->getLoc(), result, bridgeType));
    }
    rewriter.replaceOp(op, replacements);
    return success();
  }

private:
  std::string operationName;
  RecurrentKind recurrentKind;
  std::string activation;
};

} // namespace

void populateNamedLstmPattern(RewritePatternSet &patterns,
                              ArrayRef<std::string> selectedOperations,
                              StringRef operationName) {
  if (!isOperationSelected(selectedOperations, operationName))
    return;
  patterns.add<ConvertRecurrentPattern>(
      operationName, RecurrentKind::Lstm, /*activation=*/"",
      patterns.getContext());
}

void populateNamedGruPattern(RewritePatternSet &patterns,
                             ArrayRef<std::string> selectedOperations,
                             StringRef operationName) {
  if (!isOperationSelected(selectedOperations, operationName))
    return;
  patterns.add<ConvertRecurrentPattern>(
      operationName, RecurrentKind::Gru, /*activation=*/"",
      patterns.getContext());
}

void populateNamedRnnPattern(RewritePatternSet &patterns,
                             ArrayRef<std::string> selectedOperations,
                             StringRef operationName, StringRef activation) {
  if (!isOperationSelected(selectedOperations, operationName))
    return;
  patterns.add<ConvertRecurrentPattern>(
      operationName, RecurrentKind::Rnn, activation,
      patterns.getContext());
}

} // namespace mlir::dnn::torch_to_dnn

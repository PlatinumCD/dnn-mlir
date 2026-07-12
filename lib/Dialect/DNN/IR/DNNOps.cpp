#include "dnn-mlir/Dialect/DNN/IR/DNNOps.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace mlir::dnn;

namespace {

bool areCompatibleStaticDims(int64_t lhs, int64_t rhs) {
  return ShapedType::isDynamic(lhs) || ShapedType::isDynamic(rhs) ||
         lhs == rhs;
}

LogicalResult verifyCommonElementType(Operation *op, TypeRange types) {
  Type elementType = cast<ShapedType>(types.front()).getElementType();
  for (Type type : types.drop_front()) {
    if (cast<ShapedType>(type).getElementType() != elementType)
      return op->emitOpError("requires matching tensor element types");
  }
  return success();
}

} // namespace

LogicalResult MmOp::verify() {
  auto lhsType = cast<RankedTensorType>(getLhs().getType());
  auto rhsType = cast<RankedTensorType>(getRhs().getType());
  auto resultType = cast<RankedTensorType>(getResult().getType());

  if (lhsType.getRank() != 2 || rhsType.getRank() != 2 ||
      resultType.getRank() != 2)
    return emitOpError("requires rank-two operands and result");

  if (failed(verifyCommonElementType(
          *this, TypeRange{lhsType, rhsType, resultType})))
    return failure();

  if (!areCompatibleStaticDims(lhsType.getDimSize(1),
                               rhsType.getDimSize(0)))
    return emitOpError("has mismatching contracting dimensions");
  if (!areCompatibleStaticDims(resultType.getDimSize(0),
                               lhsType.getDimSize(0)) ||
      !areCompatibleStaticDims(resultType.getDimSize(1),
                               rhsType.getDimSize(1)))
    return emitOpError("result shape must be [lhs_rows, rhs_columns]");
  return success();
}

LogicalResult LinearOp::verify() {
  auto inputType = cast<RankedTensorType>(getInput().getType());
  auto weightType = cast<RankedTensorType>(getWeight().getType());
  auto resultType = cast<RankedTensorType>(getResult().getType());

  if (inputType.getRank() < 1)
    return emitOpError("requires input rank of at least one");
  if (weightType.getRank() != 2)
    return emitOpError("requires rank-two weight");
  if (resultType.getRank() != inputType.getRank())
    return emitOpError("requires result rank to match input rank");

  SmallVector<Type> elementTypes{inputType, weightType, resultType};
  if (Value bias = getBias()) {
    auto biasType = cast<RankedTensorType>(bias.getType());
    if (biasType.getRank() != 1)
      return emitOpError("requires rank-one bias");
    if (!areCompatibleStaticDims(biasType.getDimSize(0),
                                 weightType.getDimSize(0)))
      return emitOpError("bias size must equal out_features");
    elementTypes.push_back(biasType);
  }
  if (failed(verifyCommonElementType(*this, elementTypes)))
    return failure();

  if (!areCompatibleStaticDims(inputType.getDimSize(inputType.getRank() - 1),
                               weightType.getDimSize(1)))
    return emitOpError("input feature size must equal weight in_features");
  for (int64_t dim = 0; dim < inputType.getRank() - 1; ++dim) {
    if (!areCompatibleStaticDims(inputType.getDimSize(dim),
                                 resultType.getDimSize(dim)))
      return emitOpError("result batch dimensions must match the input");
  }
  if (!areCompatibleStaticDims(
          resultType.getDimSize(resultType.getRank() - 1),
          weightType.getDimSize(0)))
    return emitOpError("result final dimension must equal out_features");
  return success();
}

LogicalResult RnnOp::verify() {
  StringRef activation = getActivation();
  if (activation != "tanh" && activation != "relu")
    return emitOpError("requires activation to be 'tanh' or 'relu'");
  if (getNumResults() != 2)
    return emitOpError("requires output and final hidden-state results");
  return success();
}

#define GET_OP_CLASSES
#include "dnn-mlir/Dialect/DNN/IR/DNNOps.cpp.inc"

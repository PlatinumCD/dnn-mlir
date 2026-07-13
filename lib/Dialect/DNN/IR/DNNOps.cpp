#include "dnn-mlir/Dialect/DNN/IR/DNNOps.h"

#include "llvm/ADT/STLExtras.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"

#include <algorithm>

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

LogicalResult MatmulOp::verify() {
  auto lhsType = cast<RankedTensorType>(getLhs().getType());
  auto rhsType = cast<RankedTensorType>(getRhs().getType());
  auto resultType = cast<RankedTensorType>(getResult().getType());
  int64_t lhsRank = lhsType.getRank();
  int64_t rhsRank = rhsType.getRank();

  if (lhsRank < 1 || rhsRank < 1)
    return emitOpError("requires operand ranks of at least one");
  if (failed(verifyCommonElementType(
          *this, TypeRange{lhsType, rhsType, resultType})))
    return failure();

  int64_t lhsContracting = lhsType.getDimSize(lhsRank - 1);
  int64_t rhsContracting =
      rhsType.getDimSize(rhsRank == 1 ? 0 : rhsRank - 2);
  if (!areCompatibleStaticDims(lhsContracting, rhsContracting))
    return emitOpError("has mismatching contracting dimensions");

  int64_t lhsBatchRank = std::max<int64_t>(lhsRank - 2, 0);
  int64_t rhsBatchRank = std::max<int64_t>(rhsRank - 2, 0);
  int64_t resultBatchRank = std::max(lhsBatchRank, rhsBatchRank);
  SmallVector<int64_t> expectedShape;
  expectedShape.reserve(resultBatchRank + (lhsRank > 1) + (rhsRank > 1));

  // Align batch dimensions from the right and apply standard broadcasting.
  // Missing leading dimensions behave as dimensions of size one.
  for (int64_t resultDim = 0; resultDim < resultBatchRank; ++resultDim) {
    int64_t lhsOffset = resultBatchRank - lhsBatchRank;
    int64_t rhsOffset = resultBatchRank - rhsBatchRank;
    int64_t lhsDim = resultDim < lhsOffset
                         ? 1
                         : lhsType.getDimSize(resultDim - lhsOffset);
    int64_t rhsDim = resultDim < rhsOffset
                         ? 1
                         : rhsType.getDimSize(resultDim - rhsOffset);

    if (!ShapedType::isDynamic(lhsDim) && !ShapedType::isDynamic(rhsDim) &&
        lhsDim != rhsDim && lhsDim != 1 && rhsDim != 1)
      return emitOpError("has non-broadcastable batch dimensions ")
             << lhsDim << " and " << rhsDim;

    int64_t broadcastDim = ShapedType::kDynamic;
    if (lhsDim == 1)
      broadcastDim = rhsDim;
    else if (rhsDim == 1 || lhsDim == rhsDim)
      broadcastDim = lhsDim;
    else if (ShapedType::isDynamic(lhsDim) &&
             !ShapedType::isDynamic(rhsDim))
      broadcastDim = rhsDim;
    else if (!ShapedType::isDynamic(lhsDim) &&
             ShapedType::isDynamic(rhsDim))
      broadcastDim = lhsDim;
    expectedShape.push_back(broadcastDim);
  }

  if (lhsRank > 1)
    expectedShape.push_back(lhsType.getDimSize(lhsRank - 2));
  if (rhsRank > 1)
    expectedShape.push_back(rhsType.getDimSize(rhsRank - 1));

  if (resultType.getRank() != static_cast<int64_t>(expectedShape.size()))
    return emitOpError("has an incorrect result rank");
  for (auto [actual, expected] :
       llvm::zip(resultType.getShape(), expectedShape))
    if (!areCompatibleStaticDims(actual, expected))
      return emitOpError(
          "result shape does not match the broadcasted matmul shape");
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

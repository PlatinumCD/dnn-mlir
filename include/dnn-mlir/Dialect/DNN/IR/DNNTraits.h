#ifndef DNN_MLIR_DIALECT_DNN_IR_DNNTRAITS_H
#define DNN_MLIR_DIALECT_DNN_IR_DNNTRAITS_H

#include "llvm/ADT/STLExtras.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/OpDefinition.h"

namespace mlir::dnn::OpTrait {

// DNN operations sit beyond the Torch mutation and aliasing boundary. Torch
// values must be converted to builtin or other standard MLIR types before
// entering the dialect.
template <typename ConcreteType>
class ValueSemantics
    : public ::mlir::OpTrait::TraitBase<ConcreteType, ValueSemantics> {
public:
  static LogicalResult verifyTrait(Operation *operation) {
    auto isTorchType = [](Type type) {
      return type.getDialect().getNamespace() == "torch";
    };
    if (llvm::any_of(operation->getOperandTypes(), isTorchType) ||
        llvm::any_of(operation->getResultTypes(), isTorchType))
      return operation->emitOpError(
          "requires value-semantic non-Torch operand and result types");
    return success();
  }
};

// Verifies the positional encoding used to retain constant Torch operands.
// For ordinary operations, live operands and parameters partition the original
// operand list. Recurrent operations additionally use operand_groups to record
// how flattened tensor-list operands map back to that list.
template <typename ConcreteType>
class ParameterMetadata
    : public ::mlir::OpTrait::TraitBase<ConcreteType, ParameterMetadata> {
public:
  static LogicalResult verifyTrait(Operation *operation) {
    auto indices = operation->getAttrOfType<DenseI32ArrayAttr>(
        "parameter_indices");
    auto parameters = operation->getAttrOfType<ArrayAttr>("parameters");
    if (static_cast<bool>(indices) != static_cast<bool>(parameters))
      return operation->emitOpError(
          "requires parameter_indices and parameters to appear together");

    int64_t originalOperandCount = operation->getNumOperands();
    DenseI32ArrayAttr operandGroups =
        operation->getAttrOfType<DenseI32ArrayAttr>("operand_groups");
    int64_t zeroGroupCount = 0;
    if (operandGroups) {
      originalOperandCount = operandGroups.size();
      int64_t flattenedOperandCount = 0;
      for (int32_t groupSize : operandGroups.asArrayRef()) {
        if (groupSize < 0)
          return operation->emitOpError(
              "requires nonnegative operand group sizes");
        flattenedOperandCount += groupSize;
        zeroGroupCount += groupSize == 0;
      }
      if (flattenedOperandCount != operation->getNumOperands())
        return operation->emitOpError(
            "requires operand_groups to account for every operand");
    } else if (parameters) {
      originalOperandCount += parameters.size();
    }

    if (!indices) {
      if (zeroGroupCount != 0)
        return operation->emitOpError(
            "requires parameters for zero-sized operand groups");
      return success();
    }
    int64_t parameterCount = static_cast<int64_t>(parameters.size());
    if (indices.size() != parameterCount)
      return operation->emitOpError(
          "requires equal parameter_indices and parameters lengths");
    if (operandGroups && zeroGroupCount != parameterCount)
      return operation->emitOpError(
          "requires one parameter for every zero-sized operand group");

    int32_t previous = -1;
    for (int32_t index : indices.asArrayRef()) {
      if (index < 0 || index >= originalOperandCount)
        return operation->emitOpError(
            "has a parameter index outside the original operand range");
      if (index <= previous)
        return operation->emitOpError(
            "requires strictly increasing, unique parameter indices");
      if (operandGroups && operandGroups[index] != 0)
        return operation->emitOpError(
            "requires parameter indices to identify zero-sized operand groups");
      previous = index;
    }
    return success();
  }
};

} // namespace mlir::dnn::OpTrait

#endif // DNN_MLIR_DIALECT_DNN_IR_DNNTRAITS_H

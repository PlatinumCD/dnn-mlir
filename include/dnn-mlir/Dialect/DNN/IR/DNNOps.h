#ifndef DNN_MLIR_DIALECT_DNN_IR_DNNOPS_H
#define DNN_MLIR_DIALECT_DNN_IR_DNNOPS_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "dnn-mlir/Dialect/DNN/IR/DNNDialect.h"
#include "dnn-mlir/Dialect/DNN/IR/DNNTraits.h"

#define GET_OP_CLASSES
#include "dnn-mlir/Dialect/DNN/IR/DNNOps.h.inc"

#endif // DNN_MLIR_DIALECT_DNN_IR_DNNOPS_H

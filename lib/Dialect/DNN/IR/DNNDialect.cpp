#include "dnn-mlir/Dialect/DNN/IR/DNNDialect.h"
#include "dnn-mlir/Dialect/DNN/IR/DNNOps.h"

using namespace mlir;
using namespace mlir::dnn;

#include "dnn-mlir/Dialect/DNN/IR/DNNDialect.cpp.inc"

void DNNDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "dnn-mlir/Dialect/DNN/IR/DNNOps.cpp.inc"
      >();
}


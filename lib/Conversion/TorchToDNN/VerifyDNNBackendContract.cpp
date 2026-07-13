#include "dnn-mlir/Conversion/TorchToDNN/Passes.h"

#include "mlir/IR/AttrTypeSubElements.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace mlir::dnn {
namespace {

bool isTorchDialect(StringRef dialectNamespace) {
  return dialectNamespace == "torch" || dialectNamespace == "torch_c";
}

bool containsTorchType(Type type) {
  return type
      .walk([&](Type nestedType) -> WalkResult {
        if (isTorchDialect(nestedType.getDialect().getNamespace()))
          return WalkResult::interrupt();
        return WalkResult::advance();
      })
      .wasInterrupted();
}

WalkResult verifyBackendOperation(Operation *op) {
  StringRef dialectNamespace = op->getName().getDialectNamespace();
  if (isTorchDialect(dialectNamespace)) {
    op->emitOpError() << "is illegal in the DNN backend contract: residual '"
                      << dialectNamespace << "' operation";
    return WalkResult::interrupt();
  }

  if (isa<UnrealizedConversionCastOp>(op)) {
    op->emitOpError() << "is illegal in the DNN backend contract: unresolved "
                        "conversion cast";
    return WalkResult::interrupt();
  }

  for (Type type : op->getOperandTypes()) {
    if (containsTorchType(type)) {
      op->emitOpError()
          << "is illegal in the DNN backend contract: operand has residual "
             "Torch type "
          << type;
      return WalkResult::interrupt();
    }
  }
  for (Type type : op->getResultTypes()) {
    if (containsTorchType(type)) {
      op->emitOpError()
          << "is illegal in the DNN backend contract: result has residual "
             "Torch type "
          << type;
      return WalkResult::interrupt();
    }
  }

  // Function result types are stored in a type attribute rather than as SSA
  // results. Walking every attribute also covers nested TypeAttrs on other
  // symbol and metadata operations.
  for (NamedAttribute namedAttribute : op->getAttrs()) {
    WalkResult attributeResult =
        namedAttribute.getValue().walk([&](Type type) -> WalkResult {
          if (isTorchDialect(type.getDialect().getNamespace()))
            return WalkResult::interrupt();
          return WalkResult::advance();
        });
    if (attributeResult.wasInterrupted()) {
      op->emitOpError()
          << "is illegal in the DNN backend contract: attribute '"
          << namedAttribute.getName().getValue()
          << "' contains a residual Torch type";
      return WalkResult::interrupt();
    }
  }

  for (Region &region : op->getRegions()) {
    for (Block &block : region) {
      for (BlockArgument argument : block.getArguments()) {
        if (containsTorchType(argument.getType())) {
          op->emitOpError()
              << "is illegal in the DNN backend contract: region block "
                 "argument has residual Torch type "
              << argument.getType();
          return WalkResult::interrupt();
        }
      }
    }
  }

  return WalkResult::advance();
}

class VerifyDNNBackendContractPass
    : public PassWrapper<VerifyDNNBackendContractPass,
                         OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VerifyDNNBackendContractPass)

  StringRef getArgument() const final {
    return "verify-dnn-backend-contract";
  }
  StringRef getDescription() const final {
    return "Verify that no source-level Torch artifacts remain in DNN backend "
           "IR";
  }

  void runOnOperation() final {
    WalkResult result = getOperation().walk<WalkOrder::PreOrder>(
        [](Operation *op) { return verifyBackendOperation(op); });

    if (result.wasInterrupted())
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> createVerifyDNNBackendContractPass() {
  return std::make_unique<VerifyDNNBackendContractPass>();
}

} // namespace mlir::dnn

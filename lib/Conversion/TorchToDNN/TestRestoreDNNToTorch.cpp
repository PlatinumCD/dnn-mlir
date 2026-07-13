#include "dnn-mlir/Conversion/TorchToDNN/Passes.h"

#include "dnn-mlir/Dialect/DNN/IR/DNNDialect.h"
#include "dnn-mlir/Dialect/DNN/IR/DNNOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/IR/TorchTypes.h"
#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionDialect.h"
#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionOps.h"

using namespace mlir;

namespace mlir::dnn {
namespace {

namespace Torch = torch::Torch;
namespace TorchConversion = torch::TorchConversion;

Torch::ValueTensorType getTorchTensorType(RankedTensorType builtinType) {
  return Torch::ValueTensorType::get(
      builtinType.getContext(), builtinType.getShape(),
      builtinType.getElementType(), builtinType.getEncoding());
}

FailureOr<Value> restoreTorchTensor(Value builtinTensor) {
  auto bridge =
      builtinTensor.getDefiningOp<TorchConversion::ToBuiltinTensorOp>();
  if (!bridge)
    return failure();
  return bridge->getOperand(0);
}

LogicalResult replaceCapturedResult(PatternRewriter &rewriter, Operation *op,
                                    Value builtinResult, Value torchResult) {
  SmallVector<TorchConversion::FromBuiltinTensorOp> bridges;
  for (Operation *user : builtinResult.getUsers()) {
    auto bridge = dyn_cast<TorchConversion::FromBuiltinTensorOp>(user);
    if (!bridge)
      return rewriter.notifyMatchFailure(
          op, "test restoration requires captured results to return through "
              "Torch tensor bridges");
    bridges.push_back(bridge);
  }
  for (TorchConversion::FromBuiltinTensorOp bridge : bridges)
    rewriter.replaceOp(bridge, torchResult);
  rewriter.eraseOp(op);
  return success();
}

class RestoreMmPattern : public OpRewritePattern<MmOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(MmOp op,
                                PatternRewriter &rewriter) const override {
    Location location = op.getLoc();
    FailureOr<Value> lhs = restoreTorchTensor(op.getLhs());
    FailureOr<Value> rhs = restoreTorchTensor(op.getRhs());
    if (failed(lhs) || failed(rhs))
      return rewriter.notifyMatchFailure(
          op, "test restoration requires operands produced by Torch tensor "
              "bridges");
    auto resultType = getTorchTensorType(
        cast<RankedTensorType>(op.getResult().getType()));
    Value result = Torch::AtenMmOp::create(rewriter, location, resultType,
                                           *lhs, *rhs);
    return replaceCapturedResult(rewriter, op, op.getResult(), result);
  }
};

class RestoreReluPattern : public OpRewritePattern<ReluOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ReluOp op,
                                PatternRewriter &rewriter) const override {
    if (op->getNumOperands() != 1 || op->getNumResults() != 1 ||
        op->getAttr("parameter_indices") || op->getAttr("parameters"))
      return rewriter.notifyMatchFailure(
          op, "test restoration only supports parameter-free dnn.relu");

    Location location = op.getLoc();
    FailureOr<Value> input = restoreTorchTensor(op->getOperand(0));
    if (failed(input))
      return rewriter.notifyMatchFailure(
          op, "test restoration requires an operand produced by a Torch "
              "tensor bridge");
    Type builtinResultType = op->getResult(0).getType();
    auto resultType =
        getTorchTensorType(cast<RankedTensorType>(builtinResultType));
    Value result =
        Torch::AtenReluOp::create(rewriter, location, resultType, *input);
    return replaceCapturedResult(rewriter, op, op->getResult(0), result);
  }
};

class TestRestoreDNNToTorchPass
    : public PassWrapper<TestRestoreDNNToTorchPass,
                         OperationPass<func::FuncOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestRestoreDNNToTorchPass)

  StringRef getArgument() const final { return "test-restore-dnn-to-torch"; }
  StringRef getDescription() const final {
    return "Test-only restoration of supported DNN captures to equivalent "
           "Torch operations";
  }

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<DNNDialect, Torch::TorchDialect,
                    TorchConversion::TorchConversionDialect>();
  }

  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<RestoreMmPattern, RestoreReluPattern>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
      return;
    }

    WalkResult unsupported =
        getOperation().walk([](Operation *operation) -> WalkResult {
          if (operation->getName().getDialectNamespace() != "dnn")
            return WalkResult::advance();
          operation->emitOpError(
              "is not supported by the test-only semantic restoration pass");
          return WalkResult::interrupt();
        });
    if (unsupported.wasInterrupted())
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> createTestRestoreDNNToTorchPass() {
  return std::make_unique<TestRestoreDNNToTorchPass>();
}

} // namespace mlir::dnn

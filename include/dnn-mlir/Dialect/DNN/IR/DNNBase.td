#ifndef DNN_MLIR_DIALECT_DNN_IR_DNNBASE_TD
#define DNN_MLIR_DIALECT_DNN_IR_DNNBASE_TD

include "mlir/IR/OpBase.td"

def DNN_Dialect : Dialect {
  let name = "dnn";
  let cppNamespace = "::mlir::dnn";
  let summary = "High-level neural-network operations";
  let description = [{
    The DNN dialect preserves important neural-network structure while
    unrelated Torch operations continue lowering to standard MLIR dialects.
  }];
}

class DNN_Op<string mnemonic, list<Trait> traits = []>
    : Op<DNN_Dialect, mnemonic, traits>;

#endif // DNN_MLIR_DIALECT_DNN_IR_DNNBASE_TD


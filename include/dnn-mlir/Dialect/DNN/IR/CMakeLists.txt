set(LLVM_TARGET_DEFINITIONS DNNBase.td)
mlir_tablegen(DNNDialect.h.inc -gen-dialect-decls -dialect=dnn)
mlir_tablegen(DNNDialect.cpp.inc -gen-dialect-defs -dialect=dnn)

set(LLVM_TARGET_DEFINITIONS DNNOps.td)
mlir_tablegen(DNNOps.h.inc -gen-op-decls)
mlir_tablegen(DNNOps.cpp.inc -gen-op-defs)

add_public_tablegen_target(DNNIRIncGen)


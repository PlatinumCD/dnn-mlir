add_mlir_dialect_library(DNNIR
  DNNDialect.cpp
  DNNOps.cpp

  DEPENDS
  DNNIRIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSideEffectInterfaces
  MLIRSupport
)


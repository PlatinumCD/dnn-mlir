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

target_include_directories(DNNIR PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

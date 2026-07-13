// RUN: dnn-mlir-opt %s | FileCheck %s
//
// Structural Hugging Face BERT Base encoder imported through Torch-MLIR's FX
// path without decomposition. It uses sequence length 128, deterministic splat
// parameters, and an explicit 4-D boolean attention mask to bypass frontend
// mask construction while preserving BERT attention semantics.
//
// CHECK-LABEL: func.func @bert_base_encoder
// CHECK-DAG: dnn.linear
// CHECK-DAG: dnn.add
// CHECK-DAG: dnn.embedding
// CHECK-DAG: dnn.layer_norm
// CHECK-DAG: dnn.scaled_dot_product_attention
// CHECK-DAG: dnn.gelu
// CHECK-DAG: dnn.tanh
// CHECK-NOT: torch.aten.embedding

module {
  func.func @bert_base_encoder(%arg0: !torch.vtensor<[1,128],si64>, %arg1: !torch.vtensor<[1,1,128,128],i1>) -> !torch.vtensor<[1,128,768],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %cst = arith.constant dense<0.000000e+00> : tensor<512x768xf32>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<2x768xf32>
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<30522x768xf32>
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<768x3072xf32>
    %cst_3 = arith.constant dense<0.000000e+00> : tensor<3072xf32>
    %cst_4 = arith.constant dense<0.000000e+00> : tensor<3072x768xf32>
    %cst_5 = arith.constant dense<0.000000e+00> : tensor<768x768xf32>
    %cst_6 = arith.constant dense<0.000000e+00> : tensor<768xf32>
    %int2 = torch.constant.int 2
    %int64 = torch.constant.int 64
    %float1.000000e-01 = torch.constant.float 1.000000e-01
    %false = torch.constant.bool false
    %int-1 = torch.constant.int -1
    %0 = torch.vtensor.literal(dense<0> : tensor<1x512xi64>) : !torch.vtensor<[1,512],si64>
    %int1 = torch.constant.int 1
    %int0 = torch.constant.int 0
    %int128 = torch.constant.int 128
    %1 = torch.aten.slice.Tensor %0, %int1, %int0, %int128, %int1 : !torch.vtensor<[1,512],si64>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,128],si64>
    %2 = torch.prim.ListConstruct %int1, %int-1 : (!torch.int, !torch.int) -> !torch.list<int>
    %3 = torch.aten.expand %0, %2, %false : !torch.vtensor<[1,512],si64>, !torch.list<int>, !torch.bool -> !torch.vtensor<[1,512],si64>
    %4 = torch.aten.gather %3, %int1, %1, %false : !torch.vtensor<[1,512],si64>, !torch.int, !torch.vtensor<[1,128],si64>, !torch.bool -> !torch.vtensor<[1,128],si64>
    %5 = torch.prim.ListConstruct %int1, %int128 : (!torch.int, !torch.int) -> !torch.list<int>
    %6 = torch.aten.expand %4, %5, %false : !torch.vtensor<[1,128],si64>, !torch.list<int>, !torch.bool -> !torch.vtensor<[1,128],si64>
    %7 = torch_c.to_builtin_tensor %arg0 : !torch.vtensor<[1,128],si64> -> tensor<1x128xi64>
    %8 = dnn.embedding %cst_1, %7 {parameter_indices = array<i32: 2, 3, 4>, parameters = [0, false, false]} : (tensor<30522x768xf32>, tensor<1x128xi64>) -> tensor<1x128x768xf32>
    %9 = torch_c.from_builtin_tensor %8 : tensor<1x128x768xf32> -> !torch.vtensor<[1,128,768],f32>
    %10 = torch_c.to_builtin_tensor %6 : !torch.vtensor<[1,128],si64> -> tensor<1x128xi64>
    %11 = dnn.embedding %cst_0, %10 {parameter_indices = array<i32: 2, 3, 4>, parameters = [-1, false, false]} : (tensor<2x768xf32>, tensor<1x128xi64>) -> tensor<1x128x768xf32>
    %12 = torch_c.from_builtin_tensor %11 : tensor<1x128x768xf32> -> !torch.vtensor<[1,128,768],f32>
    %13 = torch_c.to_builtin_tensor %9 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %14 = torch_c.to_builtin_tensor %12 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %15 = dnn.add %13, %14 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<1x128x768xf32>, tensor<1x128x768xf32>) -> tensor<1x128x768xf32>
    %16 = torch_c.from_builtin_tensor %15 : tensor<1x128x768xf32> -> !torch.vtensor<[1,128,768],f32>
    %17 = torch_c.to_builtin_tensor %1 : !torch.vtensor<[1,128],si64> -> tensor<1x128xi64>
    %18 = dnn.embedding %cst, %17 {parameter_indices = array<i32: 2, 3, 4>, parameters = [-1, false, false]} : (tensor<512x768xf32>, tensor<1x128xi64>) -> tensor<1x128x768xf32>
    %19 = torch_c.from_builtin_tensor %18 : tensor<1x128x768xf32> -> !torch.vtensor<[1,128,768],f32>
    %20 = torch_c.to_builtin_tensor %16 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %21 = torch_c.to_builtin_tensor %19 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %22 = dnn.add %20, %21 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<1x128x768xf32>, tensor<1x128x768xf32>) -> tensor<1x128x768xf32>
    %23 = torch_c.from_builtin_tensor %22 : tensor<1x128x768xf32> -> !torch.vtensor<[1,128,768],f32>
    %24 = torch_c.to_builtin_tensor %23 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %25 = dnn.layer_norm %24, %cst_6, %cst_6 {parameter_indices = array<i32: 1, 4, 5>, parameters = [[768], 9.9999999999999998E-13, false]} : (tensor<1x128x768xf32>, tensor<768xf32>, tensor<768xf32>) -> tensor<1x128x768xf32>
    %26 = torch_c.from_builtin_tensor %25 : tensor<1x128x768xf32> -> !torch.vtensor<[1,128,768],f32>
    %27 = torch.aten.dropout %26, %float1.000000e-01, %false : !torch.vtensor<[1,128,768],f32>, !torch.float, !torch.bool -> !torch.vtensor<[1,128,768],f32>
    %28 = torch_c.to_builtin_tensor %27 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %29 = dnn.linear %28, %cst_5, %cst_6 : tensor<1x128x768xf32>, tensor<768x768xf32>, tensor<768xf32> -> tensor<1x128x768xf32>
    %30 = torch_c.from_builtin_tensor %29 : tensor<1x128x768xf32> -> !torch.vtensor<[1,128,768],f32>
    %31 = torch.prim.ListConstruct %int1, %int128, %int-1, %int64 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %32 = torch.aten.view %30, %31 : !torch.vtensor<[1,128,768],f32>, !torch.list<int> -> !torch.vtensor<[1,128,12,64],f32>
    %33 = torch.aten.transpose.int %32, %int1, %int2 : !torch.vtensor<[1,128,12,64],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,12,128,64],f32>
    %34 = torch_c.to_builtin_tensor %27 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %35 = dnn.linear %34, %cst_5, %cst_6 : tensor<1x128x768xf32>, tensor<768x768xf32>, tensor<768xf32> -> tensor<1x128x768xf32>
    %36 = torch_c.from_builtin_tensor %35 : tensor<1x128x768xf32> -> !torch.vtensor<[1,128,768],f32>
    %37 = torch.prim.ListConstruct %int1, %int128, %int-1, %int64 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %38 = torch.aten.view %36, %37 : !torch.vtensor<[1,128,768],f32>, !torch.list<int> -> !torch.vtensor<[1,128,12,64],f32>
    %39 = torch.aten.transpose.int %38, %int1, %int2 : !torch.vtensor<[1,128,12,64],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,12,128,64],f32>
    %40 = torch_c.to_builtin_tensor %27 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %41 = dnn.linear %40, %cst_5, %cst_6 : tensor<1x128x768xf32>, tensor<768x768xf32>, tensor<768xf32> -> tensor<1x128x768xf32>
    %42 = torch_c.from_builtin_tensor %41 : tensor<1x128x768xf32> -> !torch.vtensor<[1,128,768],f32>
    %43 = torch.prim.ListConstruct %int1, %int128, %int-1, %int64 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %44 = torch.aten.view %42, %43 : !torch.vtensor<[1,128,768],f32>, !torch.list<int> -> !torch.vtensor<[1,128,12,64],f32>
    %45 = torch.aten.transpose.int %44, %int1, %int2 : !torch.vtensor<[1,128,12,64],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,12,128,64],f32>
    %46 = torch_c.to_builtin_tensor %33 : !torch.vtensor<[1,12,128,64],f32> -> tensor<1x12x128x64xf32>
    %47 = torch_c.to_builtin_tensor %39 : !torch.vtensor<[1,12,128,64],f32> -> tensor<1x12x128x64xf32>
    %48 = torch_c.to_builtin_tensor %45 : !torch.vtensor<[1,12,128,64],f32> -> tensor<1x12x128x64xf32>
    %49 = torch_c.to_builtin_tensor %arg1 : !torch.vtensor<[1,1,128,128],i1> -> tensor<1x1x128x128xi1>
    %50 = dnn.scaled_dot_product_attention %46, %47, %48, %49 {parameter_indices = array<i32: 4, 5, 6, 7>, parameters = [0.000000e+00, false, 1.250000e-01, false]} : (tensor<1x12x128x64xf32>, tensor<1x12x128x64xf32>, tensor<1x12x128x64xf32>, tensor<1x1x128x128xi1>) -> tensor<1x12x128x64xf32>
    %51 = torch_c.from_builtin_tensor %50 : tensor<1x12x128x64xf32> -> !torch.vtensor<[1,12,128,64],f32>
    %52 = torch.aten.transpose.int %51, %int1, %int2 : !torch.vtensor<[1,12,128,64],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,128,12,64],f32>
    %53 = torch.prim.ListConstruct %int1, %int128, %int-1 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %54 = torch.aten.reshape %52, %53 : !torch.vtensor<[1,128,12,64],f32>, !torch.list<int> -> !torch.vtensor<[1,128,768],f32>
    %55 = torch_c.to_builtin_tensor %54 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %56 = dnn.linear %55, %cst_5, %cst_6 : tensor<1x128x768xf32>, tensor<768x768xf32>, tensor<768xf32> -> tensor<1x128x768xf32>
    %57 = torch_c.from_builtin_tensor %56 : tensor<1x128x768xf32> -> !torch.vtensor<[1,128,768],f32>
    %58 = torch.aten.dropout %57, %float1.000000e-01, %false : !torch.vtensor<[1,128,768],f32>, !torch.float, !torch.bool -> !torch.vtensor<[1,128,768],f32>
    %59 = torch_c.to_builtin_tensor %58 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %60 = torch_c.to_builtin_tensor %27 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %61 = dnn.add %59, %60 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<1x128x768xf32>, tensor<1x128x768xf32>) -> tensor<1x128x768xf32>
    %62 = torch_c.from_builtin_tensor %61 : tensor<1x128x768xf32> -> !torch.vtensor<[1,128,768],f32>
    %63 = torch_c.to_builtin_tensor %62 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %64 = dnn.layer_norm %63, %cst_6, %cst_6 {parameter_indices = array<i32: 1, 4, 5>, parameters = [[768], 9.9999999999999998E-13, false]} : (tensor<1x128x768xf32>, tensor<768xf32>, tensor<768xf32>) -> tensor<1x128x768xf32>
    %65 = torch_c.from_builtin_tensor %64 : tensor<1x128x768xf32> -> !torch.vtensor<[1,128,768],f32>
    %66 = torch_c.to_builtin_tensor %65 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %67 = dnn.linear %66, %cst_4, %cst_3 : tensor<1x128x768xf32>, tensor<3072x768xf32>, tensor<3072xf32> -> tensor<1x128x3072xf32>
    %68 = torch_c.from_builtin_tensor %67 : tensor<1x128x3072xf32> -> !torch.vtensor<[1,128,3072],f32>
    %69 = torch_c.to_builtin_tensor %68 : !torch.vtensor<[1,128,3072],f32> -> tensor<1x128x3072xf32>
    %70 = dnn.gelu %69 {parameter_indices = array<i32: 1>, parameters = ["none"]} : (tensor<1x128x3072xf32>) -> tensor<1x128x3072xf32>
    %71 = torch_c.from_builtin_tensor %70 : tensor<1x128x3072xf32> -> !torch.vtensor<[1,128,3072],f32>
    %72 = torch_c.to_builtin_tensor %71 : !torch.vtensor<[1,128,3072],f32> -> tensor<1x128x3072xf32>
    %73 = dnn.linear %72, %cst_2, %cst_6 : tensor<1x128x3072xf32>, tensor<768x3072xf32>, tensor<768xf32> -> tensor<1x128x768xf32>
    %74 = torch_c.from_builtin_tensor %73 : tensor<1x128x768xf32> -> !torch.vtensor<[1,128,768],f32>
    %75 = torch.aten.dropout %74, %float1.000000e-01, %false : !torch.vtensor<[1,128,768],f32>, !torch.float, !torch.bool -> !torch.vtensor<[1,128,768],f32>
    %76 = torch_c.to_builtin_tensor %75 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %77 = torch_c.to_builtin_tensor %65 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %78 = dnn.add %76, %77 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<1x128x768xf32>, tensor<1x128x768xf32>) -> tensor<1x128x768xf32>
    %79 = torch_c.from_builtin_tensor %78 : tensor<1x128x768xf32> -> !torch.vtensor<[1,128,768],f32>
    %80 = torch_c.to_builtin_tensor %79 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %81 = dnn.layer_norm %80, %cst_6, %cst_6 {parameter_indices = array<i32: 1, 4, 5>, parameters = [[768], 9.9999999999999998E-13, false]} : (tensor<1x128x768xf32>, tensor<768xf32>, tensor<768xf32>) -> tensor<1x128x768xf32>
    %82 = torch_c.from_builtin_tensor %81 : tensor<1x128x768xf32> -> !torch.vtensor<[1,128,768],f32>
    %83 = torch_c.to_builtin_tensor %82 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %84 = dnn.linear %83, %cst_5, %cst_6 : tensor<1x128x768xf32>, tensor<768x768xf32>, tensor<768xf32> -> tensor<1x128x768xf32>
    %85 = torch_c.from_builtin_tensor %84 : tensor<1x128x768xf32> -> !torch.vtensor<[1,128,768],f32>
    %86 = torch.prim.ListConstruct %int1, %int128, %int-1, %int64 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %87 = torch.aten.view %85, %86 : !torch.vtensor<[1,128,768],f32>, !torch.list<int> -> !torch.vtensor<[1,128,12,64],f32>
    %88 = torch.aten.transpose.int %87, %int1, %int2 : !torch.vtensor<[1,128,12,64],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,12,128,64],f32>
    %89 = torch_c.to_builtin_tensor %82 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %90 = dnn.linear %89, %cst_5, %cst_6 : tensor<1x128x768xf32>, tensor<768x768xf32>, tensor<768xf32> -> tensor<1x128x768xf32>
    %91 = torch_c.from_builtin_tensor %90 : tensor<1x128x768xf32> -> !torch.vtensor<[1,128,768],f32>
    %92 = torch.prim.ListConstruct %int1, %int128, %int-1, %int64 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %93 = torch.aten.view %91, %92 : !torch.vtensor<[1,128,768],f32>, !torch.list<int> -> !torch.vtensor<[1,128,12,64],f32>
    %94 = torch.aten.transpose.int %93, %int1, %int2 : !torch.vtensor<[1,128,12,64],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,12,128,64],f32>
    %95 = torch_c.to_builtin_tensor %82 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %96 = dnn.linear %95, %cst_5, %cst_6 : tensor<1x128x768xf32>, tensor<768x768xf32>, tensor<768xf32> -> tensor<1x128x768xf32>
    %97 = torch_c.from_builtin_tensor %96 : tensor<1x128x768xf32> -> !torch.vtensor<[1,128,768],f32>
    %98 = torch.prim.ListConstruct %int1, %int128, %int-1, %int64 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %99 = torch.aten.view %97, %98 : !torch.vtensor<[1,128,768],f32>, !torch.list<int> -> !torch.vtensor<[1,128,12,64],f32>
    %100 = torch.aten.transpose.int %99, %int1, %int2 : !torch.vtensor<[1,128,12,64],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,12,128,64],f32>
    %101 = torch_c.to_builtin_tensor %88 : !torch.vtensor<[1,12,128,64],f32> -> tensor<1x12x128x64xf32>
    %102 = torch_c.to_builtin_tensor %94 : !torch.vtensor<[1,12,128,64],f32> -> tensor<1x12x128x64xf32>
    %103 = torch_c.to_builtin_tensor %100 : !torch.vtensor<[1,12,128,64],f32> -> tensor<1x12x128x64xf32>
    %104 = torch_c.to_builtin_tensor %arg1 : !torch.vtensor<[1,1,128,128],i1> -> tensor<1x1x128x128xi1>
    %105 = dnn.scaled_dot_product_attention %101, %102, %103, %104 {parameter_indices = array<i32: 4, 5, 6, 7>, parameters = [0.000000e+00, false, 1.250000e-01, false]} : (tensor<1x12x128x64xf32>, tensor<1x12x128x64xf32>, tensor<1x12x128x64xf32>, tensor<1x1x128x128xi1>) -> tensor<1x12x128x64xf32>
    %106 = torch_c.from_builtin_tensor %105 : tensor<1x12x128x64xf32> -> !torch.vtensor<[1,12,128,64],f32>
    %107 = torch.aten.transpose.int %106, %int1, %int2 : !torch.vtensor<[1,12,128,64],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,128,12,64],f32>
    %108 = torch.prim.ListConstruct %int1, %int128, %int-1 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %109 = torch.aten.reshape %107, %108 : !torch.vtensor<[1,128,12,64],f32>, !torch.list<int> -> !torch.vtensor<[1,128,768],f32>
    %110 = torch_c.to_builtin_tensor %109 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %111 = dnn.linear %110, %cst_5, %cst_6 : tensor<1x128x768xf32>, tensor<768x768xf32>, tensor<768xf32> -> tensor<1x128x768xf32>
    %112 = torch_c.from_builtin_tensor %111 : tensor<1x128x768xf32> -> !torch.vtensor<[1,128,768],f32>
    %113 = torch.aten.dropout %112, %float1.000000e-01, %false : !torch.vtensor<[1,128,768],f32>, !torch.float, !torch.bool -> !torch.vtensor<[1,128,768],f32>
    %114 = torch_c.to_builtin_tensor %113 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %115 = torch_c.to_builtin_tensor %82 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %116 = dnn.add %114, %115 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<1x128x768xf32>, tensor<1x128x768xf32>) -> tensor<1x128x768xf32>
    %117 = torch_c.from_builtin_tensor %116 : tensor<1x128x768xf32> -> !torch.vtensor<[1,128,768],f32>
    %118 = torch_c.to_builtin_tensor %117 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %119 = dnn.layer_norm %118, %cst_6, %cst_6 {parameter_indices = array<i32: 1, 4, 5>, parameters = [[768], 9.9999999999999998E-13, false]} : (tensor<1x128x768xf32>, tensor<768xf32>, tensor<768xf32>) -> tensor<1x128x768xf32>
    %120 = torch_c.from_builtin_tensor %119 : tensor<1x128x768xf32> -> !torch.vtensor<[1,128,768],f32>
    %121 = torch_c.to_builtin_tensor %120 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %122 = dnn.linear %121, %cst_4, %cst_3 : tensor<1x128x768xf32>, tensor<3072x768xf32>, tensor<3072xf32> -> tensor<1x128x3072xf32>
    %123 = torch_c.from_builtin_tensor %122 : tensor<1x128x3072xf32> -> !torch.vtensor<[1,128,3072],f32>
    %124 = torch_c.to_builtin_tensor %123 : !torch.vtensor<[1,128,3072],f32> -> tensor<1x128x3072xf32>
    %125 = dnn.gelu %124 {parameter_indices = array<i32: 1>, parameters = ["none"]} : (tensor<1x128x3072xf32>) -> tensor<1x128x3072xf32>
    %126 = torch_c.from_builtin_tensor %125 : tensor<1x128x3072xf32> -> !torch.vtensor<[1,128,3072],f32>
    %127 = torch_c.to_builtin_tensor %126 : !torch.vtensor<[1,128,3072],f32> -> tensor<1x128x3072xf32>
    %128 = dnn.linear %127, %cst_2, %cst_6 : tensor<1x128x3072xf32>, tensor<768x3072xf32>, tensor<768xf32> -> tensor<1x128x768xf32>
    %129 = torch_c.from_builtin_tensor %128 : tensor<1x128x768xf32> -> !torch.vtensor<[1,128,768],f32>
    %130 = torch.aten.dropout %129, %float1.000000e-01, %false : !torch.vtensor<[1,128,768],f32>, !torch.float, !torch.bool -> !torch.vtensor<[1,128,768],f32>
    %131 = torch_c.to_builtin_tensor %130 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %132 = torch_c.to_builtin_tensor %120 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %133 = dnn.add %131, %132 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<1x128x768xf32>, tensor<1x128x768xf32>) -> tensor<1x128x768xf32>
    %134 = torch_c.from_builtin_tensor %133 : tensor<1x128x768xf32> -> !torch.vtensor<[1,128,768],f32>
    %135 = torch_c.to_builtin_tensor %134 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %136 = dnn.layer_norm %135, %cst_6, %cst_6 {parameter_indices = array<i32: 1, 4, 5>, parameters = [[768], 9.9999999999999998E-13, false]} : (tensor<1x128x768xf32>, tensor<768xf32>, tensor<768xf32>) -> tensor<1x128x768xf32>
    %137 = torch_c.from_builtin_tensor %136 : tensor<1x128x768xf32> -> !torch.vtensor<[1,128,768],f32>
    %138 = torch_c.to_builtin_tensor %137 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %139 = dnn.linear %138, %cst_5, %cst_6 : tensor<1x128x768xf32>, tensor<768x768xf32>, tensor<768xf32> -> tensor<1x128x768xf32>
    %140 = torch_c.from_builtin_tensor %139 : tensor<1x128x768xf32> -> !torch.vtensor<[1,128,768],f32>
    %141 = torch.prim.ListConstruct %int1, %int128, %int-1, %int64 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %142 = torch.aten.view %140, %141 : !torch.vtensor<[1,128,768],f32>, !torch.list<int> -> !torch.vtensor<[1,128,12,64],f32>
    %143 = torch.aten.transpose.int %142, %int1, %int2 : !torch.vtensor<[1,128,12,64],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,12,128,64],f32>
    %144 = torch_c.to_builtin_tensor %137 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %145 = dnn.linear %144, %cst_5, %cst_6 : tensor<1x128x768xf32>, tensor<768x768xf32>, tensor<768xf32> -> tensor<1x128x768xf32>
    %146 = torch_c.from_builtin_tensor %145 : tensor<1x128x768xf32> -> !torch.vtensor<[1,128,768],f32>
    %147 = torch.prim.ListConstruct %int1, %int128, %int-1, %int64 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %148 = torch.aten.view %146, %147 : !torch.vtensor<[1,128,768],f32>, !torch.list<int> -> !torch.vtensor<[1,128,12,64],f32>
    %149 = torch.aten.transpose.int %148, %int1, %int2 : !torch.vtensor<[1,128,12,64],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,12,128,64],f32>
    %150 = torch_c.to_builtin_tensor %137 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %151 = dnn.linear %150, %cst_5, %cst_6 : tensor<1x128x768xf32>, tensor<768x768xf32>, tensor<768xf32> -> tensor<1x128x768xf32>
    %152 = torch_c.from_builtin_tensor %151 : tensor<1x128x768xf32> -> !torch.vtensor<[1,128,768],f32>
    %153 = torch.prim.ListConstruct %int1, %int128, %int-1, %int64 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %154 = torch.aten.view %152, %153 : !torch.vtensor<[1,128,768],f32>, !torch.list<int> -> !torch.vtensor<[1,128,12,64],f32>
    %155 = torch.aten.transpose.int %154, %int1, %int2 : !torch.vtensor<[1,128,12,64],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,12,128,64],f32>
    %156 = torch_c.to_builtin_tensor %143 : !torch.vtensor<[1,12,128,64],f32> -> tensor<1x12x128x64xf32>
    %157 = torch_c.to_builtin_tensor %149 : !torch.vtensor<[1,12,128,64],f32> -> tensor<1x12x128x64xf32>
    %158 = torch_c.to_builtin_tensor %155 : !torch.vtensor<[1,12,128,64],f32> -> tensor<1x12x128x64xf32>
    %159 = torch_c.to_builtin_tensor %arg1 : !torch.vtensor<[1,1,128,128],i1> -> tensor<1x1x128x128xi1>
    %160 = dnn.scaled_dot_product_attention %156, %157, %158, %159 {parameter_indices = array<i32: 4, 5, 6, 7>, parameters = [0.000000e+00, false, 1.250000e-01, false]} : (tensor<1x12x128x64xf32>, tensor<1x12x128x64xf32>, tensor<1x12x128x64xf32>, tensor<1x1x128x128xi1>) -> tensor<1x12x128x64xf32>
    %161 = torch_c.from_builtin_tensor %160 : tensor<1x12x128x64xf32> -> !torch.vtensor<[1,12,128,64],f32>
    %162 = torch.aten.transpose.int %161, %int1, %int2 : !torch.vtensor<[1,12,128,64],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,128,12,64],f32>
    %163 = torch.prim.ListConstruct %int1, %int128, %int-1 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %164 = torch.aten.reshape %162, %163 : !torch.vtensor<[1,128,12,64],f32>, !torch.list<int> -> !torch.vtensor<[1,128,768],f32>
    %165 = torch_c.to_builtin_tensor %164 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %166 = dnn.linear %165, %cst_5, %cst_6 : tensor<1x128x768xf32>, tensor<768x768xf32>, tensor<768xf32> -> tensor<1x128x768xf32>
    %167 = torch_c.from_builtin_tensor %166 : tensor<1x128x768xf32> -> !torch.vtensor<[1,128,768],f32>
    %168 = torch.aten.dropout %167, %float1.000000e-01, %false : !torch.vtensor<[1,128,768],f32>, !torch.float, !torch.bool -> !torch.vtensor<[1,128,768],f32>
    %169 = torch_c.to_builtin_tensor %168 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %170 = torch_c.to_builtin_tensor %137 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %171 = dnn.add %169, %170 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<1x128x768xf32>, tensor<1x128x768xf32>) -> tensor<1x128x768xf32>
    %172 = torch_c.from_builtin_tensor %171 : tensor<1x128x768xf32> -> !torch.vtensor<[1,128,768],f32>
    %173 = torch_c.to_builtin_tensor %172 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %174 = dnn.layer_norm %173, %cst_6, %cst_6 {parameter_indices = array<i32: 1, 4, 5>, parameters = [[768], 9.9999999999999998E-13, false]} : (tensor<1x128x768xf32>, tensor<768xf32>, tensor<768xf32>) -> tensor<1x128x768xf32>
    %175 = torch_c.from_builtin_tensor %174 : tensor<1x128x768xf32> -> !torch.vtensor<[1,128,768],f32>
    %176 = torch_c.to_builtin_tensor %175 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %177 = dnn.linear %176, %cst_4, %cst_3 : tensor<1x128x768xf32>, tensor<3072x768xf32>, tensor<3072xf32> -> tensor<1x128x3072xf32>
    %178 = torch_c.from_builtin_tensor %177 : tensor<1x128x3072xf32> -> !torch.vtensor<[1,128,3072],f32>
    %179 = torch_c.to_builtin_tensor %178 : !torch.vtensor<[1,128,3072],f32> -> tensor<1x128x3072xf32>
    %180 = dnn.gelu %179 {parameter_indices = array<i32: 1>, parameters = ["none"]} : (tensor<1x128x3072xf32>) -> tensor<1x128x3072xf32>
    %181 = torch_c.from_builtin_tensor %180 : tensor<1x128x3072xf32> -> !torch.vtensor<[1,128,3072],f32>
    %182 = torch_c.to_builtin_tensor %181 : !torch.vtensor<[1,128,3072],f32> -> tensor<1x128x3072xf32>
    %183 = dnn.linear %182, %cst_2, %cst_6 : tensor<1x128x3072xf32>, tensor<768x3072xf32>, tensor<768xf32> -> tensor<1x128x768xf32>
    %184 = torch_c.from_builtin_tensor %183 : tensor<1x128x768xf32> -> !torch.vtensor<[1,128,768],f32>
    %185 = torch.aten.dropout %184, %float1.000000e-01, %false : !torch.vtensor<[1,128,768],f32>, !torch.float, !torch.bool -> !torch.vtensor<[1,128,768],f32>
    %186 = torch_c.to_builtin_tensor %185 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %187 = torch_c.to_builtin_tensor %175 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %188 = dnn.add %186, %187 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<1x128x768xf32>, tensor<1x128x768xf32>) -> tensor<1x128x768xf32>
    %189 = torch_c.from_builtin_tensor %188 : tensor<1x128x768xf32> -> !torch.vtensor<[1,128,768],f32>
    %190 = torch_c.to_builtin_tensor %189 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %191 = dnn.layer_norm %190, %cst_6, %cst_6 {parameter_indices = array<i32: 1, 4, 5>, parameters = [[768], 9.9999999999999998E-13, false]} : (tensor<1x128x768xf32>, tensor<768xf32>, tensor<768xf32>) -> tensor<1x128x768xf32>
    %192 = torch_c.from_builtin_tensor %191 : tensor<1x128x768xf32> -> !torch.vtensor<[1,128,768],f32>
    %193 = torch_c.to_builtin_tensor %192 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %194 = dnn.linear %193, %cst_5, %cst_6 : tensor<1x128x768xf32>, tensor<768x768xf32>, tensor<768xf32> -> tensor<1x128x768xf32>
    %195 = torch_c.from_builtin_tensor %194 : tensor<1x128x768xf32> -> !torch.vtensor<[1,128,768],f32>
    %196 = torch.prim.ListConstruct %int1, %int128, %int-1, %int64 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %197 = torch.aten.view %195, %196 : !torch.vtensor<[1,128,768],f32>, !torch.list<int> -> !torch.vtensor<[1,128,12,64],f32>
    %198 = torch.aten.transpose.int %197, %int1, %int2 : !torch.vtensor<[1,128,12,64],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,12,128,64],f32>
    %199 = torch_c.to_builtin_tensor %192 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %200 = dnn.linear %199, %cst_5, %cst_6 : tensor<1x128x768xf32>, tensor<768x768xf32>, tensor<768xf32> -> tensor<1x128x768xf32>
    %201 = torch_c.from_builtin_tensor %200 : tensor<1x128x768xf32> -> !torch.vtensor<[1,128,768],f32>
    %202 = torch.prim.ListConstruct %int1, %int128, %int-1, %int64 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %203 = torch.aten.view %201, %202 : !torch.vtensor<[1,128,768],f32>, !torch.list<int> -> !torch.vtensor<[1,128,12,64],f32>
    %204 = torch.aten.transpose.int %203, %int1, %int2 : !torch.vtensor<[1,128,12,64],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,12,128,64],f32>
    %205 = torch_c.to_builtin_tensor %192 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %206 = dnn.linear %205, %cst_5, %cst_6 : tensor<1x128x768xf32>, tensor<768x768xf32>, tensor<768xf32> -> tensor<1x128x768xf32>
    %207 = torch_c.from_builtin_tensor %206 : tensor<1x128x768xf32> -> !torch.vtensor<[1,128,768],f32>
    %208 = torch.prim.ListConstruct %int1, %int128, %int-1, %int64 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %209 = torch.aten.view %207, %208 : !torch.vtensor<[1,128,768],f32>, !torch.list<int> -> !torch.vtensor<[1,128,12,64],f32>
    %210 = torch.aten.transpose.int %209, %int1, %int2 : !torch.vtensor<[1,128,12,64],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,12,128,64],f32>
    %211 = torch_c.to_builtin_tensor %198 : !torch.vtensor<[1,12,128,64],f32> -> tensor<1x12x128x64xf32>
    %212 = torch_c.to_builtin_tensor %204 : !torch.vtensor<[1,12,128,64],f32> -> tensor<1x12x128x64xf32>
    %213 = torch_c.to_builtin_tensor %210 : !torch.vtensor<[1,12,128,64],f32> -> tensor<1x12x128x64xf32>
    %214 = torch_c.to_builtin_tensor %arg1 : !torch.vtensor<[1,1,128,128],i1> -> tensor<1x1x128x128xi1>
    %215 = dnn.scaled_dot_product_attention %211, %212, %213, %214 {parameter_indices = array<i32: 4, 5, 6, 7>, parameters = [0.000000e+00, false, 1.250000e-01, false]} : (tensor<1x12x128x64xf32>, tensor<1x12x128x64xf32>, tensor<1x12x128x64xf32>, tensor<1x1x128x128xi1>) -> tensor<1x12x128x64xf32>
    %216 = torch_c.from_builtin_tensor %215 : tensor<1x12x128x64xf32> -> !torch.vtensor<[1,12,128,64],f32>
    %217 = torch.aten.transpose.int %216, %int1, %int2 : !torch.vtensor<[1,12,128,64],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,128,12,64],f32>
    %218 = torch.prim.ListConstruct %int1, %int128, %int-1 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %219 = torch.aten.reshape %217, %218 : !torch.vtensor<[1,128,12,64],f32>, !torch.list<int> -> !torch.vtensor<[1,128,768],f32>
    %220 = torch_c.to_builtin_tensor %219 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %221 = dnn.linear %220, %cst_5, %cst_6 : tensor<1x128x768xf32>, tensor<768x768xf32>, tensor<768xf32> -> tensor<1x128x768xf32>
    %222 = torch_c.from_builtin_tensor %221 : tensor<1x128x768xf32> -> !torch.vtensor<[1,128,768],f32>
    %223 = torch.aten.dropout %222, %float1.000000e-01, %false : !torch.vtensor<[1,128,768],f32>, !torch.float, !torch.bool -> !torch.vtensor<[1,128,768],f32>
    %224 = torch_c.to_builtin_tensor %223 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %225 = torch_c.to_builtin_tensor %192 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %226 = dnn.add %224, %225 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<1x128x768xf32>, tensor<1x128x768xf32>) -> tensor<1x128x768xf32>
    %227 = torch_c.from_builtin_tensor %226 : tensor<1x128x768xf32> -> !torch.vtensor<[1,128,768],f32>
    %228 = torch_c.to_builtin_tensor %227 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %229 = dnn.layer_norm %228, %cst_6, %cst_6 {parameter_indices = array<i32: 1, 4, 5>, parameters = [[768], 9.9999999999999998E-13, false]} : (tensor<1x128x768xf32>, tensor<768xf32>, tensor<768xf32>) -> tensor<1x128x768xf32>
    %230 = torch_c.from_builtin_tensor %229 : tensor<1x128x768xf32> -> !torch.vtensor<[1,128,768],f32>
    %231 = torch_c.to_builtin_tensor %230 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %232 = dnn.linear %231, %cst_4, %cst_3 : tensor<1x128x768xf32>, tensor<3072x768xf32>, tensor<3072xf32> -> tensor<1x128x3072xf32>
    %233 = torch_c.from_builtin_tensor %232 : tensor<1x128x3072xf32> -> !torch.vtensor<[1,128,3072],f32>
    %234 = torch_c.to_builtin_tensor %233 : !torch.vtensor<[1,128,3072],f32> -> tensor<1x128x3072xf32>
    %235 = dnn.gelu %234 {parameter_indices = array<i32: 1>, parameters = ["none"]} : (tensor<1x128x3072xf32>) -> tensor<1x128x3072xf32>
    %236 = torch_c.from_builtin_tensor %235 : tensor<1x128x3072xf32> -> !torch.vtensor<[1,128,3072],f32>
    %237 = torch_c.to_builtin_tensor %236 : !torch.vtensor<[1,128,3072],f32> -> tensor<1x128x3072xf32>
    %238 = dnn.linear %237, %cst_2, %cst_6 : tensor<1x128x3072xf32>, tensor<768x3072xf32>, tensor<768xf32> -> tensor<1x128x768xf32>
    %239 = torch_c.from_builtin_tensor %238 : tensor<1x128x768xf32> -> !torch.vtensor<[1,128,768],f32>
    %240 = torch.aten.dropout %239, %float1.000000e-01, %false : !torch.vtensor<[1,128,768],f32>, !torch.float, !torch.bool -> !torch.vtensor<[1,128,768],f32>
    %241 = torch_c.to_builtin_tensor %240 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %242 = torch_c.to_builtin_tensor %230 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %243 = dnn.add %241, %242 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<1x128x768xf32>, tensor<1x128x768xf32>) -> tensor<1x128x768xf32>
    %244 = torch_c.from_builtin_tensor %243 : tensor<1x128x768xf32> -> !torch.vtensor<[1,128,768],f32>
    %245 = torch_c.to_builtin_tensor %244 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %246 = dnn.layer_norm %245, %cst_6, %cst_6 {parameter_indices = array<i32: 1, 4, 5>, parameters = [[768], 9.9999999999999998E-13, false]} : (tensor<1x128x768xf32>, tensor<768xf32>, tensor<768xf32>) -> tensor<1x128x768xf32>
    %247 = torch_c.from_builtin_tensor %246 : tensor<1x128x768xf32> -> !torch.vtensor<[1,128,768],f32>
    %248 = torch_c.to_builtin_tensor %247 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %249 = dnn.linear %248, %cst_5, %cst_6 : tensor<1x128x768xf32>, tensor<768x768xf32>, tensor<768xf32> -> tensor<1x128x768xf32>
    %250 = torch_c.from_builtin_tensor %249 : tensor<1x128x768xf32> -> !torch.vtensor<[1,128,768],f32>
    %251 = torch.prim.ListConstruct %int1, %int128, %int-1, %int64 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %252 = torch.aten.view %250, %251 : !torch.vtensor<[1,128,768],f32>, !torch.list<int> -> !torch.vtensor<[1,128,12,64],f32>
    %253 = torch.aten.transpose.int %252, %int1, %int2 : !torch.vtensor<[1,128,12,64],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,12,128,64],f32>
    %254 = torch_c.to_builtin_tensor %247 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %255 = dnn.linear %254, %cst_5, %cst_6 : tensor<1x128x768xf32>, tensor<768x768xf32>, tensor<768xf32> -> tensor<1x128x768xf32>
    %256 = torch_c.from_builtin_tensor %255 : tensor<1x128x768xf32> -> !torch.vtensor<[1,128,768],f32>
    %257 = torch.prim.ListConstruct %int1, %int128, %int-1, %int64 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %258 = torch.aten.view %256, %257 : !torch.vtensor<[1,128,768],f32>, !torch.list<int> -> !torch.vtensor<[1,128,12,64],f32>
    %259 = torch.aten.transpose.int %258, %int1, %int2 : !torch.vtensor<[1,128,12,64],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,12,128,64],f32>
    %260 = torch_c.to_builtin_tensor %247 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %261 = dnn.linear %260, %cst_5, %cst_6 : tensor<1x128x768xf32>, tensor<768x768xf32>, tensor<768xf32> -> tensor<1x128x768xf32>
    %262 = torch_c.from_builtin_tensor %261 : tensor<1x128x768xf32> -> !torch.vtensor<[1,128,768],f32>
    %263 = torch.prim.ListConstruct %int1, %int128, %int-1, %int64 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %264 = torch.aten.view %262, %263 : !torch.vtensor<[1,128,768],f32>, !torch.list<int> -> !torch.vtensor<[1,128,12,64],f32>
    %265 = torch.aten.transpose.int %264, %int1, %int2 : !torch.vtensor<[1,128,12,64],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,12,128,64],f32>
    %266 = torch_c.to_builtin_tensor %253 : !torch.vtensor<[1,12,128,64],f32> -> tensor<1x12x128x64xf32>
    %267 = torch_c.to_builtin_tensor %259 : !torch.vtensor<[1,12,128,64],f32> -> tensor<1x12x128x64xf32>
    %268 = torch_c.to_builtin_tensor %265 : !torch.vtensor<[1,12,128,64],f32> -> tensor<1x12x128x64xf32>
    %269 = torch_c.to_builtin_tensor %arg1 : !torch.vtensor<[1,1,128,128],i1> -> tensor<1x1x128x128xi1>
    %270 = dnn.scaled_dot_product_attention %266, %267, %268, %269 {parameter_indices = array<i32: 4, 5, 6, 7>, parameters = [0.000000e+00, false, 1.250000e-01, false]} : (tensor<1x12x128x64xf32>, tensor<1x12x128x64xf32>, tensor<1x12x128x64xf32>, tensor<1x1x128x128xi1>) -> tensor<1x12x128x64xf32>
    %271 = torch_c.from_builtin_tensor %270 : tensor<1x12x128x64xf32> -> !torch.vtensor<[1,12,128,64],f32>
    %272 = torch.aten.transpose.int %271, %int1, %int2 : !torch.vtensor<[1,12,128,64],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,128,12,64],f32>
    %273 = torch.prim.ListConstruct %int1, %int128, %int-1 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %274 = torch.aten.reshape %272, %273 : !torch.vtensor<[1,128,12,64],f32>, !torch.list<int> -> !torch.vtensor<[1,128,768],f32>
    %275 = torch_c.to_builtin_tensor %274 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %276 = dnn.linear %275, %cst_5, %cst_6 : tensor<1x128x768xf32>, tensor<768x768xf32>, tensor<768xf32> -> tensor<1x128x768xf32>
    %277 = torch_c.from_builtin_tensor %276 : tensor<1x128x768xf32> -> !torch.vtensor<[1,128,768],f32>
    %278 = torch.aten.dropout %277, %float1.000000e-01, %false : !torch.vtensor<[1,128,768],f32>, !torch.float, !torch.bool -> !torch.vtensor<[1,128,768],f32>
    %279 = torch_c.to_builtin_tensor %278 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %280 = torch_c.to_builtin_tensor %247 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %281 = dnn.add %279, %280 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<1x128x768xf32>, tensor<1x128x768xf32>) -> tensor<1x128x768xf32>
    %282 = torch_c.from_builtin_tensor %281 : tensor<1x128x768xf32> -> !torch.vtensor<[1,128,768],f32>
    %283 = torch_c.to_builtin_tensor %282 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %284 = dnn.layer_norm %283, %cst_6, %cst_6 {parameter_indices = array<i32: 1, 4, 5>, parameters = [[768], 9.9999999999999998E-13, false]} : (tensor<1x128x768xf32>, tensor<768xf32>, tensor<768xf32>) -> tensor<1x128x768xf32>
    %285 = torch_c.from_builtin_tensor %284 : tensor<1x128x768xf32> -> !torch.vtensor<[1,128,768],f32>
    %286 = torch_c.to_builtin_tensor %285 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %287 = dnn.linear %286, %cst_4, %cst_3 : tensor<1x128x768xf32>, tensor<3072x768xf32>, tensor<3072xf32> -> tensor<1x128x3072xf32>
    %288 = torch_c.from_builtin_tensor %287 : tensor<1x128x3072xf32> -> !torch.vtensor<[1,128,3072],f32>
    %289 = torch_c.to_builtin_tensor %288 : !torch.vtensor<[1,128,3072],f32> -> tensor<1x128x3072xf32>
    %290 = dnn.gelu %289 {parameter_indices = array<i32: 1>, parameters = ["none"]} : (tensor<1x128x3072xf32>) -> tensor<1x128x3072xf32>
    %291 = torch_c.from_builtin_tensor %290 : tensor<1x128x3072xf32> -> !torch.vtensor<[1,128,3072],f32>
    %292 = torch_c.to_builtin_tensor %291 : !torch.vtensor<[1,128,3072],f32> -> tensor<1x128x3072xf32>
    %293 = dnn.linear %292, %cst_2, %cst_6 : tensor<1x128x3072xf32>, tensor<768x3072xf32>, tensor<768xf32> -> tensor<1x128x768xf32>
    %294 = torch_c.from_builtin_tensor %293 : tensor<1x128x768xf32> -> !torch.vtensor<[1,128,768],f32>
    %295 = torch.aten.dropout %294, %float1.000000e-01, %false : !torch.vtensor<[1,128,768],f32>, !torch.float, !torch.bool -> !torch.vtensor<[1,128,768],f32>
    %296 = torch_c.to_builtin_tensor %295 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %297 = torch_c.to_builtin_tensor %285 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %298 = dnn.add %296, %297 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<1x128x768xf32>, tensor<1x128x768xf32>) -> tensor<1x128x768xf32>
    %299 = torch_c.from_builtin_tensor %298 : tensor<1x128x768xf32> -> !torch.vtensor<[1,128,768],f32>
    %300 = torch_c.to_builtin_tensor %299 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %301 = dnn.layer_norm %300, %cst_6, %cst_6 {parameter_indices = array<i32: 1, 4, 5>, parameters = [[768], 9.9999999999999998E-13, false]} : (tensor<1x128x768xf32>, tensor<768xf32>, tensor<768xf32>) -> tensor<1x128x768xf32>
    %302 = torch_c.from_builtin_tensor %301 : tensor<1x128x768xf32> -> !torch.vtensor<[1,128,768],f32>
    %303 = torch_c.to_builtin_tensor %302 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %304 = dnn.linear %303, %cst_5, %cst_6 : tensor<1x128x768xf32>, tensor<768x768xf32>, tensor<768xf32> -> tensor<1x128x768xf32>
    %305 = torch_c.from_builtin_tensor %304 : tensor<1x128x768xf32> -> !torch.vtensor<[1,128,768],f32>
    %306 = torch.prim.ListConstruct %int1, %int128, %int-1, %int64 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %307 = torch.aten.view %305, %306 : !torch.vtensor<[1,128,768],f32>, !torch.list<int> -> !torch.vtensor<[1,128,12,64],f32>
    %308 = torch.aten.transpose.int %307, %int1, %int2 : !torch.vtensor<[1,128,12,64],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,12,128,64],f32>
    %309 = torch_c.to_builtin_tensor %302 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %310 = dnn.linear %309, %cst_5, %cst_6 : tensor<1x128x768xf32>, tensor<768x768xf32>, tensor<768xf32> -> tensor<1x128x768xf32>
    %311 = torch_c.from_builtin_tensor %310 : tensor<1x128x768xf32> -> !torch.vtensor<[1,128,768],f32>
    %312 = torch.prim.ListConstruct %int1, %int128, %int-1, %int64 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %313 = torch.aten.view %311, %312 : !torch.vtensor<[1,128,768],f32>, !torch.list<int> -> !torch.vtensor<[1,128,12,64],f32>
    %314 = torch.aten.transpose.int %313, %int1, %int2 : !torch.vtensor<[1,128,12,64],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,12,128,64],f32>
    %315 = torch_c.to_builtin_tensor %302 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %316 = dnn.linear %315, %cst_5, %cst_6 : tensor<1x128x768xf32>, tensor<768x768xf32>, tensor<768xf32> -> tensor<1x128x768xf32>
    %317 = torch_c.from_builtin_tensor %316 : tensor<1x128x768xf32> -> !torch.vtensor<[1,128,768],f32>
    %318 = torch.prim.ListConstruct %int1, %int128, %int-1, %int64 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %319 = torch.aten.view %317, %318 : !torch.vtensor<[1,128,768],f32>, !torch.list<int> -> !torch.vtensor<[1,128,12,64],f32>
    %320 = torch.aten.transpose.int %319, %int1, %int2 : !torch.vtensor<[1,128,12,64],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,12,128,64],f32>
    %321 = torch_c.to_builtin_tensor %308 : !torch.vtensor<[1,12,128,64],f32> -> tensor<1x12x128x64xf32>
    %322 = torch_c.to_builtin_tensor %314 : !torch.vtensor<[1,12,128,64],f32> -> tensor<1x12x128x64xf32>
    %323 = torch_c.to_builtin_tensor %320 : !torch.vtensor<[1,12,128,64],f32> -> tensor<1x12x128x64xf32>
    %324 = torch_c.to_builtin_tensor %arg1 : !torch.vtensor<[1,1,128,128],i1> -> tensor<1x1x128x128xi1>
    %325 = dnn.scaled_dot_product_attention %321, %322, %323, %324 {parameter_indices = array<i32: 4, 5, 6, 7>, parameters = [0.000000e+00, false, 1.250000e-01, false]} : (tensor<1x12x128x64xf32>, tensor<1x12x128x64xf32>, tensor<1x12x128x64xf32>, tensor<1x1x128x128xi1>) -> tensor<1x12x128x64xf32>
    %326 = torch_c.from_builtin_tensor %325 : tensor<1x12x128x64xf32> -> !torch.vtensor<[1,12,128,64],f32>
    %327 = torch.aten.transpose.int %326, %int1, %int2 : !torch.vtensor<[1,12,128,64],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,128,12,64],f32>
    %328 = torch.prim.ListConstruct %int1, %int128, %int-1 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %329 = torch.aten.reshape %327, %328 : !torch.vtensor<[1,128,12,64],f32>, !torch.list<int> -> !torch.vtensor<[1,128,768],f32>
    %330 = torch_c.to_builtin_tensor %329 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %331 = dnn.linear %330, %cst_5, %cst_6 : tensor<1x128x768xf32>, tensor<768x768xf32>, tensor<768xf32> -> tensor<1x128x768xf32>
    %332 = torch_c.from_builtin_tensor %331 : tensor<1x128x768xf32> -> !torch.vtensor<[1,128,768],f32>
    %333 = torch.aten.dropout %332, %float1.000000e-01, %false : !torch.vtensor<[1,128,768],f32>, !torch.float, !torch.bool -> !torch.vtensor<[1,128,768],f32>
    %334 = torch_c.to_builtin_tensor %333 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %335 = torch_c.to_builtin_tensor %302 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %336 = dnn.add %334, %335 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<1x128x768xf32>, tensor<1x128x768xf32>) -> tensor<1x128x768xf32>
    %337 = torch_c.from_builtin_tensor %336 : tensor<1x128x768xf32> -> !torch.vtensor<[1,128,768],f32>
    %338 = torch_c.to_builtin_tensor %337 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %339 = dnn.layer_norm %338, %cst_6, %cst_6 {parameter_indices = array<i32: 1, 4, 5>, parameters = [[768], 9.9999999999999998E-13, false]} : (tensor<1x128x768xf32>, tensor<768xf32>, tensor<768xf32>) -> tensor<1x128x768xf32>
    %340 = torch_c.from_builtin_tensor %339 : tensor<1x128x768xf32> -> !torch.vtensor<[1,128,768],f32>
    %341 = torch_c.to_builtin_tensor %340 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %342 = dnn.linear %341, %cst_4, %cst_3 : tensor<1x128x768xf32>, tensor<3072x768xf32>, tensor<3072xf32> -> tensor<1x128x3072xf32>
    %343 = torch_c.from_builtin_tensor %342 : tensor<1x128x3072xf32> -> !torch.vtensor<[1,128,3072],f32>
    %344 = torch_c.to_builtin_tensor %343 : !torch.vtensor<[1,128,3072],f32> -> tensor<1x128x3072xf32>
    %345 = dnn.gelu %344 {parameter_indices = array<i32: 1>, parameters = ["none"]} : (tensor<1x128x3072xf32>) -> tensor<1x128x3072xf32>
    %346 = torch_c.from_builtin_tensor %345 : tensor<1x128x3072xf32> -> !torch.vtensor<[1,128,3072],f32>
    %347 = torch_c.to_builtin_tensor %346 : !torch.vtensor<[1,128,3072],f32> -> tensor<1x128x3072xf32>
    %348 = dnn.linear %347, %cst_2, %cst_6 : tensor<1x128x3072xf32>, tensor<768x3072xf32>, tensor<768xf32> -> tensor<1x128x768xf32>
    %349 = torch_c.from_builtin_tensor %348 : tensor<1x128x768xf32> -> !torch.vtensor<[1,128,768],f32>
    %350 = torch.aten.dropout %349, %float1.000000e-01, %false : !torch.vtensor<[1,128,768],f32>, !torch.float, !torch.bool -> !torch.vtensor<[1,128,768],f32>
    %351 = torch_c.to_builtin_tensor %350 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %352 = torch_c.to_builtin_tensor %340 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %353 = dnn.add %351, %352 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<1x128x768xf32>, tensor<1x128x768xf32>) -> tensor<1x128x768xf32>
    %354 = torch_c.from_builtin_tensor %353 : tensor<1x128x768xf32> -> !torch.vtensor<[1,128,768],f32>
    %355 = torch_c.to_builtin_tensor %354 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %356 = dnn.layer_norm %355, %cst_6, %cst_6 {parameter_indices = array<i32: 1, 4, 5>, parameters = [[768], 9.9999999999999998E-13, false]} : (tensor<1x128x768xf32>, tensor<768xf32>, tensor<768xf32>) -> tensor<1x128x768xf32>
    %357 = torch_c.from_builtin_tensor %356 : tensor<1x128x768xf32> -> !torch.vtensor<[1,128,768],f32>
    %358 = torch_c.to_builtin_tensor %357 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %359 = dnn.linear %358, %cst_5, %cst_6 : tensor<1x128x768xf32>, tensor<768x768xf32>, tensor<768xf32> -> tensor<1x128x768xf32>
    %360 = torch_c.from_builtin_tensor %359 : tensor<1x128x768xf32> -> !torch.vtensor<[1,128,768],f32>
    %361 = torch.prim.ListConstruct %int1, %int128, %int-1, %int64 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %362 = torch.aten.view %360, %361 : !torch.vtensor<[1,128,768],f32>, !torch.list<int> -> !torch.vtensor<[1,128,12,64],f32>
    %363 = torch.aten.transpose.int %362, %int1, %int2 : !torch.vtensor<[1,128,12,64],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,12,128,64],f32>
    %364 = torch_c.to_builtin_tensor %357 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %365 = dnn.linear %364, %cst_5, %cst_6 : tensor<1x128x768xf32>, tensor<768x768xf32>, tensor<768xf32> -> tensor<1x128x768xf32>
    %366 = torch_c.from_builtin_tensor %365 : tensor<1x128x768xf32> -> !torch.vtensor<[1,128,768],f32>
    %367 = torch.prim.ListConstruct %int1, %int128, %int-1, %int64 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %368 = torch.aten.view %366, %367 : !torch.vtensor<[1,128,768],f32>, !torch.list<int> -> !torch.vtensor<[1,128,12,64],f32>
    %369 = torch.aten.transpose.int %368, %int1, %int2 : !torch.vtensor<[1,128,12,64],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,12,128,64],f32>
    %370 = torch_c.to_builtin_tensor %357 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %371 = dnn.linear %370, %cst_5, %cst_6 : tensor<1x128x768xf32>, tensor<768x768xf32>, tensor<768xf32> -> tensor<1x128x768xf32>
    %372 = torch_c.from_builtin_tensor %371 : tensor<1x128x768xf32> -> !torch.vtensor<[1,128,768],f32>
    %373 = torch.prim.ListConstruct %int1, %int128, %int-1, %int64 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %374 = torch.aten.view %372, %373 : !torch.vtensor<[1,128,768],f32>, !torch.list<int> -> !torch.vtensor<[1,128,12,64],f32>
    %375 = torch.aten.transpose.int %374, %int1, %int2 : !torch.vtensor<[1,128,12,64],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,12,128,64],f32>
    %376 = torch_c.to_builtin_tensor %363 : !torch.vtensor<[1,12,128,64],f32> -> tensor<1x12x128x64xf32>
    %377 = torch_c.to_builtin_tensor %369 : !torch.vtensor<[1,12,128,64],f32> -> tensor<1x12x128x64xf32>
    %378 = torch_c.to_builtin_tensor %375 : !torch.vtensor<[1,12,128,64],f32> -> tensor<1x12x128x64xf32>
    %379 = torch_c.to_builtin_tensor %arg1 : !torch.vtensor<[1,1,128,128],i1> -> tensor<1x1x128x128xi1>
    %380 = dnn.scaled_dot_product_attention %376, %377, %378, %379 {parameter_indices = array<i32: 4, 5, 6, 7>, parameters = [0.000000e+00, false, 1.250000e-01, false]} : (tensor<1x12x128x64xf32>, tensor<1x12x128x64xf32>, tensor<1x12x128x64xf32>, tensor<1x1x128x128xi1>) -> tensor<1x12x128x64xf32>
    %381 = torch_c.from_builtin_tensor %380 : tensor<1x12x128x64xf32> -> !torch.vtensor<[1,12,128,64],f32>
    %382 = torch.aten.transpose.int %381, %int1, %int2 : !torch.vtensor<[1,12,128,64],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,128,12,64],f32>
    %383 = torch.prim.ListConstruct %int1, %int128, %int-1 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %384 = torch.aten.reshape %382, %383 : !torch.vtensor<[1,128,12,64],f32>, !torch.list<int> -> !torch.vtensor<[1,128,768],f32>
    %385 = torch_c.to_builtin_tensor %384 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %386 = dnn.linear %385, %cst_5, %cst_6 : tensor<1x128x768xf32>, tensor<768x768xf32>, tensor<768xf32> -> tensor<1x128x768xf32>
    %387 = torch_c.from_builtin_tensor %386 : tensor<1x128x768xf32> -> !torch.vtensor<[1,128,768],f32>
    %388 = torch.aten.dropout %387, %float1.000000e-01, %false : !torch.vtensor<[1,128,768],f32>, !torch.float, !torch.bool -> !torch.vtensor<[1,128,768],f32>
    %389 = torch_c.to_builtin_tensor %388 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %390 = torch_c.to_builtin_tensor %357 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %391 = dnn.add %389, %390 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<1x128x768xf32>, tensor<1x128x768xf32>) -> tensor<1x128x768xf32>
    %392 = torch_c.from_builtin_tensor %391 : tensor<1x128x768xf32> -> !torch.vtensor<[1,128,768],f32>
    %393 = torch_c.to_builtin_tensor %392 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %394 = dnn.layer_norm %393, %cst_6, %cst_6 {parameter_indices = array<i32: 1, 4, 5>, parameters = [[768], 9.9999999999999998E-13, false]} : (tensor<1x128x768xf32>, tensor<768xf32>, tensor<768xf32>) -> tensor<1x128x768xf32>
    %395 = torch_c.from_builtin_tensor %394 : tensor<1x128x768xf32> -> !torch.vtensor<[1,128,768],f32>
    %396 = torch_c.to_builtin_tensor %395 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %397 = dnn.linear %396, %cst_4, %cst_3 : tensor<1x128x768xf32>, tensor<3072x768xf32>, tensor<3072xf32> -> tensor<1x128x3072xf32>
    %398 = torch_c.from_builtin_tensor %397 : tensor<1x128x3072xf32> -> !torch.vtensor<[1,128,3072],f32>
    %399 = torch_c.to_builtin_tensor %398 : !torch.vtensor<[1,128,3072],f32> -> tensor<1x128x3072xf32>
    %400 = dnn.gelu %399 {parameter_indices = array<i32: 1>, parameters = ["none"]} : (tensor<1x128x3072xf32>) -> tensor<1x128x3072xf32>
    %401 = torch_c.from_builtin_tensor %400 : tensor<1x128x3072xf32> -> !torch.vtensor<[1,128,3072],f32>
    %402 = torch_c.to_builtin_tensor %401 : !torch.vtensor<[1,128,3072],f32> -> tensor<1x128x3072xf32>
    %403 = dnn.linear %402, %cst_2, %cst_6 : tensor<1x128x3072xf32>, tensor<768x3072xf32>, tensor<768xf32> -> tensor<1x128x768xf32>
    %404 = torch_c.from_builtin_tensor %403 : tensor<1x128x768xf32> -> !torch.vtensor<[1,128,768],f32>
    %405 = torch.aten.dropout %404, %float1.000000e-01, %false : !torch.vtensor<[1,128,768],f32>, !torch.float, !torch.bool -> !torch.vtensor<[1,128,768],f32>
    %406 = torch_c.to_builtin_tensor %405 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %407 = torch_c.to_builtin_tensor %395 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %408 = dnn.add %406, %407 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<1x128x768xf32>, tensor<1x128x768xf32>) -> tensor<1x128x768xf32>
    %409 = torch_c.from_builtin_tensor %408 : tensor<1x128x768xf32> -> !torch.vtensor<[1,128,768],f32>
    %410 = torch_c.to_builtin_tensor %409 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %411 = dnn.layer_norm %410, %cst_6, %cst_6 {parameter_indices = array<i32: 1, 4, 5>, parameters = [[768], 9.9999999999999998E-13, false]} : (tensor<1x128x768xf32>, tensor<768xf32>, tensor<768xf32>) -> tensor<1x128x768xf32>
    %412 = torch_c.from_builtin_tensor %411 : tensor<1x128x768xf32> -> !torch.vtensor<[1,128,768],f32>
    %413 = torch_c.to_builtin_tensor %412 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %414 = dnn.linear %413, %cst_5, %cst_6 : tensor<1x128x768xf32>, tensor<768x768xf32>, tensor<768xf32> -> tensor<1x128x768xf32>
    %415 = torch_c.from_builtin_tensor %414 : tensor<1x128x768xf32> -> !torch.vtensor<[1,128,768],f32>
    %416 = torch.prim.ListConstruct %int1, %int128, %int-1, %int64 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %417 = torch.aten.view %415, %416 : !torch.vtensor<[1,128,768],f32>, !torch.list<int> -> !torch.vtensor<[1,128,12,64],f32>
    %418 = torch.aten.transpose.int %417, %int1, %int2 : !torch.vtensor<[1,128,12,64],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,12,128,64],f32>
    %419 = torch_c.to_builtin_tensor %412 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %420 = dnn.linear %419, %cst_5, %cst_6 : tensor<1x128x768xf32>, tensor<768x768xf32>, tensor<768xf32> -> tensor<1x128x768xf32>
    %421 = torch_c.from_builtin_tensor %420 : tensor<1x128x768xf32> -> !torch.vtensor<[1,128,768],f32>
    %422 = torch.prim.ListConstruct %int1, %int128, %int-1, %int64 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %423 = torch.aten.view %421, %422 : !torch.vtensor<[1,128,768],f32>, !torch.list<int> -> !torch.vtensor<[1,128,12,64],f32>
    %424 = torch.aten.transpose.int %423, %int1, %int2 : !torch.vtensor<[1,128,12,64],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,12,128,64],f32>
    %425 = torch_c.to_builtin_tensor %412 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %426 = dnn.linear %425, %cst_5, %cst_6 : tensor<1x128x768xf32>, tensor<768x768xf32>, tensor<768xf32> -> tensor<1x128x768xf32>
    %427 = torch_c.from_builtin_tensor %426 : tensor<1x128x768xf32> -> !torch.vtensor<[1,128,768],f32>
    %428 = torch.prim.ListConstruct %int1, %int128, %int-1, %int64 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %429 = torch.aten.view %427, %428 : !torch.vtensor<[1,128,768],f32>, !torch.list<int> -> !torch.vtensor<[1,128,12,64],f32>
    %430 = torch.aten.transpose.int %429, %int1, %int2 : !torch.vtensor<[1,128,12,64],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,12,128,64],f32>
    %431 = torch_c.to_builtin_tensor %418 : !torch.vtensor<[1,12,128,64],f32> -> tensor<1x12x128x64xf32>
    %432 = torch_c.to_builtin_tensor %424 : !torch.vtensor<[1,12,128,64],f32> -> tensor<1x12x128x64xf32>
    %433 = torch_c.to_builtin_tensor %430 : !torch.vtensor<[1,12,128,64],f32> -> tensor<1x12x128x64xf32>
    %434 = torch_c.to_builtin_tensor %arg1 : !torch.vtensor<[1,1,128,128],i1> -> tensor<1x1x128x128xi1>
    %435 = dnn.scaled_dot_product_attention %431, %432, %433, %434 {parameter_indices = array<i32: 4, 5, 6, 7>, parameters = [0.000000e+00, false, 1.250000e-01, false]} : (tensor<1x12x128x64xf32>, tensor<1x12x128x64xf32>, tensor<1x12x128x64xf32>, tensor<1x1x128x128xi1>) -> tensor<1x12x128x64xf32>
    %436 = torch_c.from_builtin_tensor %435 : tensor<1x12x128x64xf32> -> !torch.vtensor<[1,12,128,64],f32>
    %437 = torch.aten.transpose.int %436, %int1, %int2 : !torch.vtensor<[1,12,128,64],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,128,12,64],f32>
    %438 = torch.prim.ListConstruct %int1, %int128, %int-1 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %439 = torch.aten.reshape %437, %438 : !torch.vtensor<[1,128,12,64],f32>, !torch.list<int> -> !torch.vtensor<[1,128,768],f32>
    %440 = torch_c.to_builtin_tensor %439 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %441 = dnn.linear %440, %cst_5, %cst_6 : tensor<1x128x768xf32>, tensor<768x768xf32>, tensor<768xf32> -> tensor<1x128x768xf32>
    %442 = torch_c.from_builtin_tensor %441 : tensor<1x128x768xf32> -> !torch.vtensor<[1,128,768],f32>
    %443 = torch.aten.dropout %442, %float1.000000e-01, %false : !torch.vtensor<[1,128,768],f32>, !torch.float, !torch.bool -> !torch.vtensor<[1,128,768],f32>
    %444 = torch_c.to_builtin_tensor %443 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %445 = torch_c.to_builtin_tensor %412 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %446 = dnn.add %444, %445 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<1x128x768xf32>, tensor<1x128x768xf32>) -> tensor<1x128x768xf32>
    %447 = torch_c.from_builtin_tensor %446 : tensor<1x128x768xf32> -> !torch.vtensor<[1,128,768],f32>
    %448 = torch_c.to_builtin_tensor %447 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %449 = dnn.layer_norm %448, %cst_6, %cst_6 {parameter_indices = array<i32: 1, 4, 5>, parameters = [[768], 9.9999999999999998E-13, false]} : (tensor<1x128x768xf32>, tensor<768xf32>, tensor<768xf32>) -> tensor<1x128x768xf32>
    %450 = torch_c.from_builtin_tensor %449 : tensor<1x128x768xf32> -> !torch.vtensor<[1,128,768],f32>
    %451 = torch_c.to_builtin_tensor %450 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %452 = dnn.linear %451, %cst_4, %cst_3 : tensor<1x128x768xf32>, tensor<3072x768xf32>, tensor<3072xf32> -> tensor<1x128x3072xf32>
    %453 = torch_c.from_builtin_tensor %452 : tensor<1x128x3072xf32> -> !torch.vtensor<[1,128,3072],f32>
    %454 = torch_c.to_builtin_tensor %453 : !torch.vtensor<[1,128,3072],f32> -> tensor<1x128x3072xf32>
    %455 = dnn.gelu %454 {parameter_indices = array<i32: 1>, parameters = ["none"]} : (tensor<1x128x3072xf32>) -> tensor<1x128x3072xf32>
    %456 = torch_c.from_builtin_tensor %455 : tensor<1x128x3072xf32> -> !torch.vtensor<[1,128,3072],f32>
    %457 = torch_c.to_builtin_tensor %456 : !torch.vtensor<[1,128,3072],f32> -> tensor<1x128x3072xf32>
    %458 = dnn.linear %457, %cst_2, %cst_6 : tensor<1x128x3072xf32>, tensor<768x3072xf32>, tensor<768xf32> -> tensor<1x128x768xf32>
    %459 = torch_c.from_builtin_tensor %458 : tensor<1x128x768xf32> -> !torch.vtensor<[1,128,768],f32>
    %460 = torch.aten.dropout %459, %float1.000000e-01, %false : !torch.vtensor<[1,128,768],f32>, !torch.float, !torch.bool -> !torch.vtensor<[1,128,768],f32>
    %461 = torch_c.to_builtin_tensor %460 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %462 = torch_c.to_builtin_tensor %450 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %463 = dnn.add %461, %462 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<1x128x768xf32>, tensor<1x128x768xf32>) -> tensor<1x128x768xf32>
    %464 = torch_c.from_builtin_tensor %463 : tensor<1x128x768xf32> -> !torch.vtensor<[1,128,768],f32>
    %465 = torch_c.to_builtin_tensor %464 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %466 = dnn.layer_norm %465, %cst_6, %cst_6 {parameter_indices = array<i32: 1, 4, 5>, parameters = [[768], 9.9999999999999998E-13, false]} : (tensor<1x128x768xf32>, tensor<768xf32>, tensor<768xf32>) -> tensor<1x128x768xf32>
    %467 = torch_c.from_builtin_tensor %466 : tensor<1x128x768xf32> -> !torch.vtensor<[1,128,768],f32>
    %468 = torch_c.to_builtin_tensor %467 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %469 = dnn.linear %468, %cst_5, %cst_6 : tensor<1x128x768xf32>, tensor<768x768xf32>, tensor<768xf32> -> tensor<1x128x768xf32>
    %470 = torch_c.from_builtin_tensor %469 : tensor<1x128x768xf32> -> !torch.vtensor<[1,128,768],f32>
    %471 = torch.prim.ListConstruct %int1, %int128, %int-1, %int64 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %472 = torch.aten.view %470, %471 : !torch.vtensor<[1,128,768],f32>, !torch.list<int> -> !torch.vtensor<[1,128,12,64],f32>
    %473 = torch.aten.transpose.int %472, %int1, %int2 : !torch.vtensor<[1,128,12,64],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,12,128,64],f32>
    %474 = torch_c.to_builtin_tensor %467 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %475 = dnn.linear %474, %cst_5, %cst_6 : tensor<1x128x768xf32>, tensor<768x768xf32>, tensor<768xf32> -> tensor<1x128x768xf32>
    %476 = torch_c.from_builtin_tensor %475 : tensor<1x128x768xf32> -> !torch.vtensor<[1,128,768],f32>
    %477 = torch.prim.ListConstruct %int1, %int128, %int-1, %int64 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %478 = torch.aten.view %476, %477 : !torch.vtensor<[1,128,768],f32>, !torch.list<int> -> !torch.vtensor<[1,128,12,64],f32>
    %479 = torch.aten.transpose.int %478, %int1, %int2 : !torch.vtensor<[1,128,12,64],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,12,128,64],f32>
    %480 = torch_c.to_builtin_tensor %467 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %481 = dnn.linear %480, %cst_5, %cst_6 : tensor<1x128x768xf32>, tensor<768x768xf32>, tensor<768xf32> -> tensor<1x128x768xf32>
    %482 = torch_c.from_builtin_tensor %481 : tensor<1x128x768xf32> -> !torch.vtensor<[1,128,768],f32>
    %483 = torch.prim.ListConstruct %int1, %int128, %int-1, %int64 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %484 = torch.aten.view %482, %483 : !torch.vtensor<[1,128,768],f32>, !torch.list<int> -> !torch.vtensor<[1,128,12,64],f32>
    %485 = torch.aten.transpose.int %484, %int1, %int2 : !torch.vtensor<[1,128,12,64],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,12,128,64],f32>
    %486 = torch_c.to_builtin_tensor %473 : !torch.vtensor<[1,12,128,64],f32> -> tensor<1x12x128x64xf32>
    %487 = torch_c.to_builtin_tensor %479 : !torch.vtensor<[1,12,128,64],f32> -> tensor<1x12x128x64xf32>
    %488 = torch_c.to_builtin_tensor %485 : !torch.vtensor<[1,12,128,64],f32> -> tensor<1x12x128x64xf32>
    %489 = torch_c.to_builtin_tensor %arg1 : !torch.vtensor<[1,1,128,128],i1> -> tensor<1x1x128x128xi1>
    %490 = dnn.scaled_dot_product_attention %486, %487, %488, %489 {parameter_indices = array<i32: 4, 5, 6, 7>, parameters = [0.000000e+00, false, 1.250000e-01, false]} : (tensor<1x12x128x64xf32>, tensor<1x12x128x64xf32>, tensor<1x12x128x64xf32>, tensor<1x1x128x128xi1>) -> tensor<1x12x128x64xf32>
    %491 = torch_c.from_builtin_tensor %490 : tensor<1x12x128x64xf32> -> !torch.vtensor<[1,12,128,64],f32>
    %492 = torch.aten.transpose.int %491, %int1, %int2 : !torch.vtensor<[1,12,128,64],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,128,12,64],f32>
    %493 = torch.prim.ListConstruct %int1, %int128, %int-1 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %494 = torch.aten.reshape %492, %493 : !torch.vtensor<[1,128,12,64],f32>, !torch.list<int> -> !torch.vtensor<[1,128,768],f32>
    %495 = torch_c.to_builtin_tensor %494 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %496 = dnn.linear %495, %cst_5, %cst_6 : tensor<1x128x768xf32>, tensor<768x768xf32>, tensor<768xf32> -> tensor<1x128x768xf32>
    %497 = torch_c.from_builtin_tensor %496 : tensor<1x128x768xf32> -> !torch.vtensor<[1,128,768],f32>
    %498 = torch.aten.dropout %497, %float1.000000e-01, %false : !torch.vtensor<[1,128,768],f32>, !torch.float, !torch.bool -> !torch.vtensor<[1,128,768],f32>
    %499 = torch_c.to_builtin_tensor %498 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %500 = torch_c.to_builtin_tensor %467 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %501 = dnn.add %499, %500 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<1x128x768xf32>, tensor<1x128x768xf32>) -> tensor<1x128x768xf32>
    %502 = torch_c.from_builtin_tensor %501 : tensor<1x128x768xf32> -> !torch.vtensor<[1,128,768],f32>
    %503 = torch_c.to_builtin_tensor %502 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %504 = dnn.layer_norm %503, %cst_6, %cst_6 {parameter_indices = array<i32: 1, 4, 5>, parameters = [[768], 9.9999999999999998E-13, false]} : (tensor<1x128x768xf32>, tensor<768xf32>, tensor<768xf32>) -> tensor<1x128x768xf32>
    %505 = torch_c.from_builtin_tensor %504 : tensor<1x128x768xf32> -> !torch.vtensor<[1,128,768],f32>
    %506 = torch_c.to_builtin_tensor %505 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %507 = dnn.linear %506, %cst_4, %cst_3 : tensor<1x128x768xf32>, tensor<3072x768xf32>, tensor<3072xf32> -> tensor<1x128x3072xf32>
    %508 = torch_c.from_builtin_tensor %507 : tensor<1x128x3072xf32> -> !torch.vtensor<[1,128,3072],f32>
    %509 = torch_c.to_builtin_tensor %508 : !torch.vtensor<[1,128,3072],f32> -> tensor<1x128x3072xf32>
    %510 = dnn.gelu %509 {parameter_indices = array<i32: 1>, parameters = ["none"]} : (tensor<1x128x3072xf32>) -> tensor<1x128x3072xf32>
    %511 = torch_c.from_builtin_tensor %510 : tensor<1x128x3072xf32> -> !torch.vtensor<[1,128,3072],f32>
    %512 = torch_c.to_builtin_tensor %511 : !torch.vtensor<[1,128,3072],f32> -> tensor<1x128x3072xf32>
    %513 = dnn.linear %512, %cst_2, %cst_6 : tensor<1x128x3072xf32>, tensor<768x3072xf32>, tensor<768xf32> -> tensor<1x128x768xf32>
    %514 = torch_c.from_builtin_tensor %513 : tensor<1x128x768xf32> -> !torch.vtensor<[1,128,768],f32>
    %515 = torch.aten.dropout %514, %float1.000000e-01, %false : !torch.vtensor<[1,128,768],f32>, !torch.float, !torch.bool -> !torch.vtensor<[1,128,768],f32>
    %516 = torch_c.to_builtin_tensor %515 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %517 = torch_c.to_builtin_tensor %505 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %518 = dnn.add %516, %517 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<1x128x768xf32>, tensor<1x128x768xf32>) -> tensor<1x128x768xf32>
    %519 = torch_c.from_builtin_tensor %518 : tensor<1x128x768xf32> -> !torch.vtensor<[1,128,768],f32>
    %520 = torch_c.to_builtin_tensor %519 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %521 = dnn.layer_norm %520, %cst_6, %cst_6 {parameter_indices = array<i32: 1, 4, 5>, parameters = [[768], 9.9999999999999998E-13, false]} : (tensor<1x128x768xf32>, tensor<768xf32>, tensor<768xf32>) -> tensor<1x128x768xf32>
    %522 = torch_c.from_builtin_tensor %521 : tensor<1x128x768xf32> -> !torch.vtensor<[1,128,768],f32>
    %523 = torch_c.to_builtin_tensor %522 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %524 = dnn.linear %523, %cst_5, %cst_6 : tensor<1x128x768xf32>, tensor<768x768xf32>, tensor<768xf32> -> tensor<1x128x768xf32>
    %525 = torch_c.from_builtin_tensor %524 : tensor<1x128x768xf32> -> !torch.vtensor<[1,128,768],f32>
    %526 = torch.prim.ListConstruct %int1, %int128, %int-1, %int64 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %527 = torch.aten.view %525, %526 : !torch.vtensor<[1,128,768],f32>, !torch.list<int> -> !torch.vtensor<[1,128,12,64],f32>
    %528 = torch.aten.transpose.int %527, %int1, %int2 : !torch.vtensor<[1,128,12,64],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,12,128,64],f32>
    %529 = torch_c.to_builtin_tensor %522 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %530 = dnn.linear %529, %cst_5, %cst_6 : tensor<1x128x768xf32>, tensor<768x768xf32>, tensor<768xf32> -> tensor<1x128x768xf32>
    %531 = torch_c.from_builtin_tensor %530 : tensor<1x128x768xf32> -> !torch.vtensor<[1,128,768],f32>
    %532 = torch.prim.ListConstruct %int1, %int128, %int-1, %int64 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %533 = torch.aten.view %531, %532 : !torch.vtensor<[1,128,768],f32>, !torch.list<int> -> !torch.vtensor<[1,128,12,64],f32>
    %534 = torch.aten.transpose.int %533, %int1, %int2 : !torch.vtensor<[1,128,12,64],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,12,128,64],f32>
    %535 = torch_c.to_builtin_tensor %522 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %536 = dnn.linear %535, %cst_5, %cst_6 : tensor<1x128x768xf32>, tensor<768x768xf32>, tensor<768xf32> -> tensor<1x128x768xf32>
    %537 = torch_c.from_builtin_tensor %536 : tensor<1x128x768xf32> -> !torch.vtensor<[1,128,768],f32>
    %538 = torch.prim.ListConstruct %int1, %int128, %int-1, %int64 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %539 = torch.aten.view %537, %538 : !torch.vtensor<[1,128,768],f32>, !torch.list<int> -> !torch.vtensor<[1,128,12,64],f32>
    %540 = torch.aten.transpose.int %539, %int1, %int2 : !torch.vtensor<[1,128,12,64],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,12,128,64],f32>
    %541 = torch_c.to_builtin_tensor %528 : !torch.vtensor<[1,12,128,64],f32> -> tensor<1x12x128x64xf32>
    %542 = torch_c.to_builtin_tensor %534 : !torch.vtensor<[1,12,128,64],f32> -> tensor<1x12x128x64xf32>
    %543 = torch_c.to_builtin_tensor %540 : !torch.vtensor<[1,12,128,64],f32> -> tensor<1x12x128x64xf32>
    %544 = torch_c.to_builtin_tensor %arg1 : !torch.vtensor<[1,1,128,128],i1> -> tensor<1x1x128x128xi1>
    %545 = dnn.scaled_dot_product_attention %541, %542, %543, %544 {parameter_indices = array<i32: 4, 5, 6, 7>, parameters = [0.000000e+00, false, 1.250000e-01, false]} : (tensor<1x12x128x64xf32>, tensor<1x12x128x64xf32>, tensor<1x12x128x64xf32>, tensor<1x1x128x128xi1>) -> tensor<1x12x128x64xf32>
    %546 = torch_c.from_builtin_tensor %545 : tensor<1x12x128x64xf32> -> !torch.vtensor<[1,12,128,64],f32>
    %547 = torch.aten.transpose.int %546, %int1, %int2 : !torch.vtensor<[1,12,128,64],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,128,12,64],f32>
    %548 = torch.prim.ListConstruct %int1, %int128, %int-1 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %549 = torch.aten.reshape %547, %548 : !torch.vtensor<[1,128,12,64],f32>, !torch.list<int> -> !torch.vtensor<[1,128,768],f32>
    %550 = torch_c.to_builtin_tensor %549 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %551 = dnn.linear %550, %cst_5, %cst_6 : tensor<1x128x768xf32>, tensor<768x768xf32>, tensor<768xf32> -> tensor<1x128x768xf32>
    %552 = torch_c.from_builtin_tensor %551 : tensor<1x128x768xf32> -> !torch.vtensor<[1,128,768],f32>
    %553 = torch.aten.dropout %552, %float1.000000e-01, %false : !torch.vtensor<[1,128,768],f32>, !torch.float, !torch.bool -> !torch.vtensor<[1,128,768],f32>
    %554 = torch_c.to_builtin_tensor %553 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %555 = torch_c.to_builtin_tensor %522 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %556 = dnn.add %554, %555 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<1x128x768xf32>, tensor<1x128x768xf32>) -> tensor<1x128x768xf32>
    %557 = torch_c.from_builtin_tensor %556 : tensor<1x128x768xf32> -> !torch.vtensor<[1,128,768],f32>
    %558 = torch_c.to_builtin_tensor %557 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %559 = dnn.layer_norm %558, %cst_6, %cst_6 {parameter_indices = array<i32: 1, 4, 5>, parameters = [[768], 9.9999999999999998E-13, false]} : (tensor<1x128x768xf32>, tensor<768xf32>, tensor<768xf32>) -> tensor<1x128x768xf32>
    %560 = torch_c.from_builtin_tensor %559 : tensor<1x128x768xf32> -> !torch.vtensor<[1,128,768],f32>
    %561 = torch_c.to_builtin_tensor %560 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %562 = dnn.linear %561, %cst_4, %cst_3 : tensor<1x128x768xf32>, tensor<3072x768xf32>, tensor<3072xf32> -> tensor<1x128x3072xf32>
    %563 = torch_c.from_builtin_tensor %562 : tensor<1x128x3072xf32> -> !torch.vtensor<[1,128,3072],f32>
    %564 = torch_c.to_builtin_tensor %563 : !torch.vtensor<[1,128,3072],f32> -> tensor<1x128x3072xf32>
    %565 = dnn.gelu %564 {parameter_indices = array<i32: 1>, parameters = ["none"]} : (tensor<1x128x3072xf32>) -> tensor<1x128x3072xf32>
    %566 = torch_c.from_builtin_tensor %565 : tensor<1x128x3072xf32> -> !torch.vtensor<[1,128,3072],f32>
    %567 = torch_c.to_builtin_tensor %566 : !torch.vtensor<[1,128,3072],f32> -> tensor<1x128x3072xf32>
    %568 = dnn.linear %567, %cst_2, %cst_6 : tensor<1x128x3072xf32>, tensor<768x3072xf32>, tensor<768xf32> -> tensor<1x128x768xf32>
    %569 = torch_c.from_builtin_tensor %568 : tensor<1x128x768xf32> -> !torch.vtensor<[1,128,768],f32>
    %570 = torch.aten.dropout %569, %float1.000000e-01, %false : !torch.vtensor<[1,128,768],f32>, !torch.float, !torch.bool -> !torch.vtensor<[1,128,768],f32>
    %571 = torch_c.to_builtin_tensor %570 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %572 = torch_c.to_builtin_tensor %560 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %573 = dnn.add %571, %572 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<1x128x768xf32>, tensor<1x128x768xf32>) -> tensor<1x128x768xf32>
    %574 = torch_c.from_builtin_tensor %573 : tensor<1x128x768xf32> -> !torch.vtensor<[1,128,768],f32>
    %575 = torch_c.to_builtin_tensor %574 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %576 = dnn.layer_norm %575, %cst_6, %cst_6 {parameter_indices = array<i32: 1, 4, 5>, parameters = [[768], 9.9999999999999998E-13, false]} : (tensor<1x128x768xf32>, tensor<768xf32>, tensor<768xf32>) -> tensor<1x128x768xf32>
    %577 = torch_c.from_builtin_tensor %576 : tensor<1x128x768xf32> -> !torch.vtensor<[1,128,768],f32>
    %578 = torch_c.to_builtin_tensor %577 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %579 = dnn.linear %578, %cst_5, %cst_6 : tensor<1x128x768xf32>, tensor<768x768xf32>, tensor<768xf32> -> tensor<1x128x768xf32>
    %580 = torch_c.from_builtin_tensor %579 : tensor<1x128x768xf32> -> !torch.vtensor<[1,128,768],f32>
    %581 = torch.prim.ListConstruct %int1, %int128, %int-1, %int64 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %582 = torch.aten.view %580, %581 : !torch.vtensor<[1,128,768],f32>, !torch.list<int> -> !torch.vtensor<[1,128,12,64],f32>
    %583 = torch.aten.transpose.int %582, %int1, %int2 : !torch.vtensor<[1,128,12,64],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,12,128,64],f32>
    %584 = torch_c.to_builtin_tensor %577 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %585 = dnn.linear %584, %cst_5, %cst_6 : tensor<1x128x768xf32>, tensor<768x768xf32>, tensor<768xf32> -> tensor<1x128x768xf32>
    %586 = torch_c.from_builtin_tensor %585 : tensor<1x128x768xf32> -> !torch.vtensor<[1,128,768],f32>
    %587 = torch.prim.ListConstruct %int1, %int128, %int-1, %int64 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %588 = torch.aten.view %586, %587 : !torch.vtensor<[1,128,768],f32>, !torch.list<int> -> !torch.vtensor<[1,128,12,64],f32>
    %589 = torch.aten.transpose.int %588, %int1, %int2 : !torch.vtensor<[1,128,12,64],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,12,128,64],f32>
    %590 = torch_c.to_builtin_tensor %577 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %591 = dnn.linear %590, %cst_5, %cst_6 : tensor<1x128x768xf32>, tensor<768x768xf32>, tensor<768xf32> -> tensor<1x128x768xf32>
    %592 = torch_c.from_builtin_tensor %591 : tensor<1x128x768xf32> -> !torch.vtensor<[1,128,768],f32>
    %593 = torch.prim.ListConstruct %int1, %int128, %int-1, %int64 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %594 = torch.aten.view %592, %593 : !torch.vtensor<[1,128,768],f32>, !torch.list<int> -> !torch.vtensor<[1,128,12,64],f32>
    %595 = torch.aten.transpose.int %594, %int1, %int2 : !torch.vtensor<[1,128,12,64],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,12,128,64],f32>
    %596 = torch_c.to_builtin_tensor %583 : !torch.vtensor<[1,12,128,64],f32> -> tensor<1x12x128x64xf32>
    %597 = torch_c.to_builtin_tensor %589 : !torch.vtensor<[1,12,128,64],f32> -> tensor<1x12x128x64xf32>
    %598 = torch_c.to_builtin_tensor %595 : !torch.vtensor<[1,12,128,64],f32> -> tensor<1x12x128x64xf32>
    %599 = torch_c.to_builtin_tensor %arg1 : !torch.vtensor<[1,1,128,128],i1> -> tensor<1x1x128x128xi1>
    %600 = dnn.scaled_dot_product_attention %596, %597, %598, %599 {parameter_indices = array<i32: 4, 5, 6, 7>, parameters = [0.000000e+00, false, 1.250000e-01, false]} : (tensor<1x12x128x64xf32>, tensor<1x12x128x64xf32>, tensor<1x12x128x64xf32>, tensor<1x1x128x128xi1>) -> tensor<1x12x128x64xf32>
    %601 = torch_c.from_builtin_tensor %600 : tensor<1x12x128x64xf32> -> !torch.vtensor<[1,12,128,64],f32>
    %602 = torch.aten.transpose.int %601, %int1, %int2 : !torch.vtensor<[1,12,128,64],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,128,12,64],f32>
    %603 = torch.prim.ListConstruct %int1, %int128, %int-1 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %604 = torch.aten.reshape %602, %603 : !torch.vtensor<[1,128,12,64],f32>, !torch.list<int> -> !torch.vtensor<[1,128,768],f32>
    %605 = torch_c.to_builtin_tensor %604 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %606 = dnn.linear %605, %cst_5, %cst_6 : tensor<1x128x768xf32>, tensor<768x768xf32>, tensor<768xf32> -> tensor<1x128x768xf32>
    %607 = torch_c.from_builtin_tensor %606 : tensor<1x128x768xf32> -> !torch.vtensor<[1,128,768],f32>
    %608 = torch.aten.dropout %607, %float1.000000e-01, %false : !torch.vtensor<[1,128,768],f32>, !torch.float, !torch.bool -> !torch.vtensor<[1,128,768],f32>
    %609 = torch_c.to_builtin_tensor %608 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %610 = torch_c.to_builtin_tensor %577 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %611 = dnn.add %609, %610 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<1x128x768xf32>, tensor<1x128x768xf32>) -> tensor<1x128x768xf32>
    %612 = torch_c.from_builtin_tensor %611 : tensor<1x128x768xf32> -> !torch.vtensor<[1,128,768],f32>
    %613 = torch_c.to_builtin_tensor %612 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %614 = dnn.layer_norm %613, %cst_6, %cst_6 {parameter_indices = array<i32: 1, 4, 5>, parameters = [[768], 9.9999999999999998E-13, false]} : (tensor<1x128x768xf32>, tensor<768xf32>, tensor<768xf32>) -> tensor<1x128x768xf32>
    %615 = torch_c.from_builtin_tensor %614 : tensor<1x128x768xf32> -> !torch.vtensor<[1,128,768],f32>
    %616 = torch_c.to_builtin_tensor %615 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %617 = dnn.linear %616, %cst_4, %cst_3 : tensor<1x128x768xf32>, tensor<3072x768xf32>, tensor<3072xf32> -> tensor<1x128x3072xf32>
    %618 = torch_c.from_builtin_tensor %617 : tensor<1x128x3072xf32> -> !torch.vtensor<[1,128,3072],f32>
    %619 = torch_c.to_builtin_tensor %618 : !torch.vtensor<[1,128,3072],f32> -> tensor<1x128x3072xf32>
    %620 = dnn.gelu %619 {parameter_indices = array<i32: 1>, parameters = ["none"]} : (tensor<1x128x3072xf32>) -> tensor<1x128x3072xf32>
    %621 = torch_c.from_builtin_tensor %620 : tensor<1x128x3072xf32> -> !torch.vtensor<[1,128,3072],f32>
    %622 = torch_c.to_builtin_tensor %621 : !torch.vtensor<[1,128,3072],f32> -> tensor<1x128x3072xf32>
    %623 = dnn.linear %622, %cst_2, %cst_6 : tensor<1x128x3072xf32>, tensor<768x3072xf32>, tensor<768xf32> -> tensor<1x128x768xf32>
    %624 = torch_c.from_builtin_tensor %623 : tensor<1x128x768xf32> -> !torch.vtensor<[1,128,768],f32>
    %625 = torch.aten.dropout %624, %float1.000000e-01, %false : !torch.vtensor<[1,128,768],f32>, !torch.float, !torch.bool -> !torch.vtensor<[1,128,768],f32>
    %626 = torch_c.to_builtin_tensor %625 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %627 = torch_c.to_builtin_tensor %615 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %628 = dnn.add %626, %627 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<1x128x768xf32>, tensor<1x128x768xf32>) -> tensor<1x128x768xf32>
    %629 = torch_c.from_builtin_tensor %628 : tensor<1x128x768xf32> -> !torch.vtensor<[1,128,768],f32>
    %630 = torch_c.to_builtin_tensor %629 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %631 = dnn.layer_norm %630, %cst_6, %cst_6 {parameter_indices = array<i32: 1, 4, 5>, parameters = [[768], 9.9999999999999998E-13, false]} : (tensor<1x128x768xf32>, tensor<768xf32>, tensor<768xf32>) -> tensor<1x128x768xf32>
    %632 = torch_c.from_builtin_tensor %631 : tensor<1x128x768xf32> -> !torch.vtensor<[1,128,768],f32>
    %633 = torch_c.to_builtin_tensor %632 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %634 = dnn.linear %633, %cst_5, %cst_6 : tensor<1x128x768xf32>, tensor<768x768xf32>, tensor<768xf32> -> tensor<1x128x768xf32>
    %635 = torch_c.from_builtin_tensor %634 : tensor<1x128x768xf32> -> !torch.vtensor<[1,128,768],f32>
    %636 = torch.prim.ListConstruct %int1, %int128, %int-1, %int64 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %637 = torch.aten.view %635, %636 : !torch.vtensor<[1,128,768],f32>, !torch.list<int> -> !torch.vtensor<[1,128,12,64],f32>
    %638 = torch.aten.transpose.int %637, %int1, %int2 : !torch.vtensor<[1,128,12,64],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,12,128,64],f32>
    %639 = torch_c.to_builtin_tensor %632 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %640 = dnn.linear %639, %cst_5, %cst_6 : tensor<1x128x768xf32>, tensor<768x768xf32>, tensor<768xf32> -> tensor<1x128x768xf32>
    %641 = torch_c.from_builtin_tensor %640 : tensor<1x128x768xf32> -> !torch.vtensor<[1,128,768],f32>
    %642 = torch.prim.ListConstruct %int1, %int128, %int-1, %int64 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %643 = torch.aten.view %641, %642 : !torch.vtensor<[1,128,768],f32>, !torch.list<int> -> !torch.vtensor<[1,128,12,64],f32>
    %644 = torch.aten.transpose.int %643, %int1, %int2 : !torch.vtensor<[1,128,12,64],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,12,128,64],f32>
    %645 = torch_c.to_builtin_tensor %632 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %646 = dnn.linear %645, %cst_5, %cst_6 : tensor<1x128x768xf32>, tensor<768x768xf32>, tensor<768xf32> -> tensor<1x128x768xf32>
    %647 = torch_c.from_builtin_tensor %646 : tensor<1x128x768xf32> -> !torch.vtensor<[1,128,768],f32>
    %648 = torch.prim.ListConstruct %int1, %int128, %int-1, %int64 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %649 = torch.aten.view %647, %648 : !torch.vtensor<[1,128,768],f32>, !torch.list<int> -> !torch.vtensor<[1,128,12,64],f32>
    %650 = torch.aten.transpose.int %649, %int1, %int2 : !torch.vtensor<[1,128,12,64],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,12,128,64],f32>
    %651 = torch_c.to_builtin_tensor %638 : !torch.vtensor<[1,12,128,64],f32> -> tensor<1x12x128x64xf32>
    %652 = torch_c.to_builtin_tensor %644 : !torch.vtensor<[1,12,128,64],f32> -> tensor<1x12x128x64xf32>
    %653 = torch_c.to_builtin_tensor %650 : !torch.vtensor<[1,12,128,64],f32> -> tensor<1x12x128x64xf32>
    %654 = torch_c.to_builtin_tensor %arg1 : !torch.vtensor<[1,1,128,128],i1> -> tensor<1x1x128x128xi1>
    %655 = dnn.scaled_dot_product_attention %651, %652, %653, %654 {parameter_indices = array<i32: 4, 5, 6, 7>, parameters = [0.000000e+00, false, 1.250000e-01, false]} : (tensor<1x12x128x64xf32>, tensor<1x12x128x64xf32>, tensor<1x12x128x64xf32>, tensor<1x1x128x128xi1>) -> tensor<1x12x128x64xf32>
    %656 = torch_c.from_builtin_tensor %655 : tensor<1x12x128x64xf32> -> !torch.vtensor<[1,12,128,64],f32>
    %657 = torch.aten.transpose.int %656, %int1, %int2 : !torch.vtensor<[1,12,128,64],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,128,12,64],f32>
    %658 = torch.prim.ListConstruct %int1, %int128, %int-1 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %659 = torch.aten.reshape %657, %658 : !torch.vtensor<[1,128,12,64],f32>, !torch.list<int> -> !torch.vtensor<[1,128,768],f32>
    %660 = torch_c.to_builtin_tensor %659 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %661 = dnn.linear %660, %cst_5, %cst_6 : tensor<1x128x768xf32>, tensor<768x768xf32>, tensor<768xf32> -> tensor<1x128x768xf32>
    %662 = torch_c.from_builtin_tensor %661 : tensor<1x128x768xf32> -> !torch.vtensor<[1,128,768],f32>
    %663 = torch.aten.dropout %662, %float1.000000e-01, %false : !torch.vtensor<[1,128,768],f32>, !torch.float, !torch.bool -> !torch.vtensor<[1,128,768],f32>
    %664 = torch_c.to_builtin_tensor %663 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %665 = torch_c.to_builtin_tensor %632 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %666 = dnn.add %664, %665 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<1x128x768xf32>, tensor<1x128x768xf32>) -> tensor<1x128x768xf32>
    %667 = torch_c.from_builtin_tensor %666 : tensor<1x128x768xf32> -> !torch.vtensor<[1,128,768],f32>
    %668 = torch_c.to_builtin_tensor %667 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %669 = dnn.layer_norm %668, %cst_6, %cst_6 {parameter_indices = array<i32: 1, 4, 5>, parameters = [[768], 9.9999999999999998E-13, false]} : (tensor<1x128x768xf32>, tensor<768xf32>, tensor<768xf32>) -> tensor<1x128x768xf32>
    %670 = torch_c.from_builtin_tensor %669 : tensor<1x128x768xf32> -> !torch.vtensor<[1,128,768],f32>
    %671 = torch_c.to_builtin_tensor %670 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %672 = dnn.linear %671, %cst_4, %cst_3 : tensor<1x128x768xf32>, tensor<3072x768xf32>, tensor<3072xf32> -> tensor<1x128x3072xf32>
    %673 = torch_c.from_builtin_tensor %672 : tensor<1x128x3072xf32> -> !torch.vtensor<[1,128,3072],f32>
    %674 = torch_c.to_builtin_tensor %673 : !torch.vtensor<[1,128,3072],f32> -> tensor<1x128x3072xf32>
    %675 = dnn.gelu %674 {parameter_indices = array<i32: 1>, parameters = ["none"]} : (tensor<1x128x3072xf32>) -> tensor<1x128x3072xf32>
    %676 = torch_c.from_builtin_tensor %675 : tensor<1x128x3072xf32> -> !torch.vtensor<[1,128,3072],f32>
    %677 = torch_c.to_builtin_tensor %676 : !torch.vtensor<[1,128,3072],f32> -> tensor<1x128x3072xf32>
    %678 = dnn.linear %677, %cst_2, %cst_6 : tensor<1x128x3072xf32>, tensor<768x3072xf32>, tensor<768xf32> -> tensor<1x128x768xf32>
    %679 = torch_c.from_builtin_tensor %678 : tensor<1x128x768xf32> -> !torch.vtensor<[1,128,768],f32>
    %680 = torch.aten.dropout %679, %float1.000000e-01, %false : !torch.vtensor<[1,128,768],f32>, !torch.float, !torch.bool -> !torch.vtensor<[1,128,768],f32>
    %681 = torch_c.to_builtin_tensor %680 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %682 = torch_c.to_builtin_tensor %670 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %683 = dnn.add %681, %682 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<1x128x768xf32>, tensor<1x128x768xf32>) -> tensor<1x128x768xf32>
    %684 = torch_c.from_builtin_tensor %683 : tensor<1x128x768xf32> -> !torch.vtensor<[1,128,768],f32>
    %685 = torch_c.to_builtin_tensor %684 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %686 = dnn.layer_norm %685, %cst_6, %cst_6 {parameter_indices = array<i32: 1, 4, 5>, parameters = [[768], 9.9999999999999998E-13, false]} : (tensor<1x128x768xf32>, tensor<768xf32>, tensor<768xf32>) -> tensor<1x128x768xf32>
    %687 = torch_c.from_builtin_tensor %686 : tensor<1x128x768xf32> -> !torch.vtensor<[1,128,768],f32>
    %688 = torch.aten.select.int %687, %int1, %int0 : !torch.vtensor<[1,128,768],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,768],f32>
    %689 = torch_c.to_builtin_tensor %688 : !torch.vtensor<[1,768],f32> -> tensor<1x768xf32>
    %690 = dnn.linear %689, %cst_5, %cst_6 : tensor<1x768xf32>, tensor<768x768xf32>, tensor<768xf32> -> tensor<1x768xf32>
    %691 = torch_c.from_builtin_tensor %690 : tensor<1x768xf32> -> !torch.vtensor<[1,768],f32>
    %692 = torch_c.to_builtin_tensor %691 : !torch.vtensor<[1,768],f32> -> tensor<1x768xf32>
    %693 = dnn.tanh %692 : (tensor<1x768xf32>) -> tensor<1x768xf32>
    return %687 : !torch.vtensor<[1,128,768],f32>
  }
}

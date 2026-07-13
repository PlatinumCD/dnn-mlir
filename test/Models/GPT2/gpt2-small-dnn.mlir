// RUN: dnn-mlir-opt %s | FileCheck %s
//
// Structural Hugging Face GPT-2 Small language model imported through
// Torch-MLIR's FX path. It contains all 12 decoder blocks, sequence length
// 128, the 50,257-token LM head, explicit position IDs, and an explicit 4-D
// causal mask. Learned parameters use deterministic zero splats so the fixture
// remains compact while preserving production tensor shapes.
//
// CHECK-LABEL: func.func @gpt2_small
// CHECK-DAG: dnn.embedding
// CHECK-DAG: dnn.mm
// CHECK-DAG: dnn.matmul
// CHECK-DAG: dnn.add
// CHECK-DAG: dnn.layer_norm
// CHECK-DAG: dnn.gelu
// CHECK-DAG: dnn.scaled_dot_product_attention
// CHECK-NOT: torch.aten.rsqrt
// CHECK-NOT: torch.aten.pow

module {
  func.func @gpt2_small(%arg0: !torch.vtensor<[1,128],si64>, %arg1: !torch.vtensor<[1,128],si64>, %arg2: !torch.vtensor<[1,1,128,128],i1>) -> !torch.vtensor<[1,128,50257],f32> {
    %cst = arith.constant dense<0.000000e+00> : tensor<3072x768xf32>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<3072xf32>
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<768x3072xf32>
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<768x768xf32>
    %cst_3 = arith.constant dense<0.000000e+00> : tensor<2304xf32>
    %cst_4 = arith.constant dense<0.000000e+00> : tensor<768x2304xf32>
    %cst_5 = arith.constant dense<0.000000e+00> : tensor<768xf32>
    %cst_6 = arith.constant dense<0.000000e+00> : tensor<1024x768xf32>
    %cst_7 = arith.constant dense<0.000000e+00> : tensor<50257x768xf32>
    %int1536 = torch.constant.int 1536
    %int0 = torch.constant.int 0
    %int3072 = torch.constant.int 3072
    %int64 = torch.constant.int 64
    %int2 = torch.constant.int 2
    %int2304 = torch.constant.int 2304
    %int768 = torch.constant.int 768
    %int1 = torch.constant.int 1
    %0 = torch.vtensor.literal(dense<0.000000e+00> : tensor<50257x768xf32>) : !torch.vtensor<[50257,768],f32>
    %int-1 = torch.constant.int -1
    %int128 = torch.constant.int 128
    %1 = torch.prim.ListConstruct %int-1, %int128 : (!torch.int, !torch.int) -> !torch.list<int>
    %2 = torch.aten.view %arg0, %1 : !torch.vtensor<[1,128],si64>, !torch.list<int> -> !torch.vtensor<[1,128],si64>
    %3 = torch_c.to_builtin_tensor %2 : !torch.vtensor<[1,128],si64> -> tensor<1x128xi64>
    %4 = dnn.embedding %cst_7, %3 {parameter_indices = array<i32: 2, 3, 4>, parameters = [-1, false, false]} : (tensor<50257x768xf32>, tensor<1x128xi64>) -> tensor<1x128x768xf32>
    %5 = torch_c.from_builtin_tensor %4 : tensor<1x128x768xf32> -> !torch.vtensor<[1,128,768],f32>
    %6 = torch_c.to_builtin_tensor %arg1 : !torch.vtensor<[1,128],si64> -> tensor<1x128xi64>
    %7 = dnn.embedding %cst_6, %6 {parameter_indices = array<i32: 2, 3, 4>, parameters = [-1, false, false]} : (tensor<1024x768xf32>, tensor<1x128xi64>) -> tensor<1x128x768xf32>
    %8 = torch_c.from_builtin_tensor %7 : tensor<1x128x768xf32> -> !torch.vtensor<[1,128,768],f32>
    %9 = torch_c.to_builtin_tensor %5 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %10 = torch_c.to_builtin_tensor %8 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %11 = dnn.add %9, %10 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<1x128x768xf32>, tensor<1x128x768xf32>) -> tensor<1x128x768xf32>
    %12 = torch_c.from_builtin_tensor %11 : tensor<1x128x768xf32> -> !torch.vtensor<[1,128,768],f32>
    %13 = torch_c.to_builtin_tensor %12 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %14 = dnn.layer_norm %13, %cst_5, %cst_5 {parameter_indices = array<i32: 1, 4, 5>, parameters = [[768], 1.000000e-05, false]} : (tensor<1x128x768xf32>, tensor<768xf32>, tensor<768xf32>) -> tensor<1x128x768xf32>
    %15 = torch_c.from_builtin_tensor %14 : tensor<1x128x768xf32> -> !torch.vtensor<[1,128,768],f32>
    %16 = torch.prim.ListConstruct %int-1, %int768 : (!torch.int, !torch.int) -> !torch.list<int>
    %17 = torch.aten.view %15, %16 : !torch.vtensor<[1,128,768],f32>, !torch.list<int> -> !torch.vtensor<[128,768],f32>
    %18 = torch_c.to_builtin_tensor %17 : !torch.vtensor<[128,768],f32> -> tensor<128x768xf32>
    %19 = dnn.mm %18, %cst_4 : tensor<128x768xf32>, tensor<768x2304xf32> -> tensor<128x2304xf32>
    %20 = torch_c.from_builtin_tensor %19 : tensor<128x2304xf32> -> !torch.vtensor<[128,2304],f32>
    %21 = torch.aten.mul.Scalar %20, %int1 : !torch.vtensor<[128,2304],f32>, !torch.int -> !torch.vtensor<[128,2304],f32>
    %22 = torch_c.to_builtin_tensor %21 : !torch.vtensor<[128,2304],f32> -> tensor<128x2304xf32>
    %23 = dnn.add %22, %cst_3 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<128x2304xf32>, tensor<2304xf32>) -> tensor<128x2304xf32>
    %24 = torch_c.from_builtin_tensor %23 : tensor<128x2304xf32> -> !torch.vtensor<[128,2304],f32>
    %25 = torch.prim.ListConstruct %int1, %int128, %int2304 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %26 = torch.aten.view %24, %25 : !torch.vtensor<[128,2304],f32>, !torch.list<int> -> !torch.vtensor<[1,128,2304],f32>
    %27 = torch.aten.slice.Tensor %26, %int2, %int0, %int768, %int1 : !torch.vtensor<[1,128,2304],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,128,768],f32>
    %28 = torch.aten.slice.Tensor %26, %int2, %int768, %int1536, %int1 : !torch.vtensor<[1,128,2304],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,128,768],f32>
    %29 = torch.aten.slice.Tensor %26, %int2, %int1536, %int2304, %int1 : !torch.vtensor<[1,128,2304],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,128,768],f32>
    %30 = torch.prim.ListConstruct %int1, %int128, %int-1, %int64 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %31 = torch.aten.view %28, %30 : !torch.vtensor<[1,128,768],f32>, !torch.list<int> -> !torch.vtensor<[1,128,12,64],f32>
    %32 = torch.aten.transpose.int %31, %int1, %int2 : !torch.vtensor<[1,128,12,64],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,12,128,64],f32>
    %33 = torch.prim.ListConstruct %int1, %int128, %int-1, %int64 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %34 = torch.aten.view %29, %33 : !torch.vtensor<[1,128,768],f32>, !torch.list<int> -> !torch.vtensor<[1,128,12,64],f32>
    %35 = torch.aten.transpose.int %34, %int1, %int2 : !torch.vtensor<[1,128,12,64],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,12,128,64],f32>
    %36 = torch.prim.ListConstruct %int1, %int128, %int-1, %int64 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %37 = torch.aten.view %27, %36 : !torch.vtensor<[1,128,768],f32>, !torch.list<int> -> !torch.vtensor<[1,128,12,64],f32>
    %38 = torch.aten.transpose.int %37, %int1, %int2 : !torch.vtensor<[1,128,12,64],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,12,128,64],f32>
    %39 = torch_c.to_builtin_tensor %38 : !torch.vtensor<[1,12,128,64],f32> -> tensor<1x12x128x64xf32>
    %40 = torch_c.to_builtin_tensor %32 : !torch.vtensor<[1,12,128,64],f32> -> tensor<1x12x128x64xf32>
    %41 = torch_c.to_builtin_tensor %35 : !torch.vtensor<[1,12,128,64],f32> -> tensor<1x12x128x64xf32>
    %42 = torch_c.to_builtin_tensor %arg2 : !torch.vtensor<[1,1,128,128],i1> -> tensor<1x1x128x128xi1>
    %43 = dnn.scaled_dot_product_attention %39, %40, %41, %42 {parameter_indices = array<i32: 4, 5, 6, 7>, parameters = [0.000000e+00, false, 1.250000e-01, false]} : (tensor<1x12x128x64xf32>, tensor<1x12x128x64xf32>, tensor<1x12x128x64xf32>, tensor<1x1x128x128xi1>) -> tensor<1x12x128x64xf32>
    %44 = torch_c.from_builtin_tensor %43 : tensor<1x12x128x64xf32> -> !torch.vtensor<[1,12,128,64],f32>
    %45 = torch.aten.transpose.int %44, %int1, %int2 : !torch.vtensor<[1,12,128,64],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,128,12,64],f32>
    %46 = torch.prim.ListConstruct %int1, %int128, %int-1 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %47 = torch.aten.view %45, %46 : !torch.vtensor<[1,128,12,64],f32>, !torch.list<int> -> !torch.vtensor<[1,128,768],f32>
    %48 = torch.prim.ListConstruct %int-1, %int768 : (!torch.int, !torch.int) -> !torch.list<int>
    %49 = torch.aten.view %47, %48 : !torch.vtensor<[1,128,768],f32>, !torch.list<int> -> !torch.vtensor<[128,768],f32>
    %50 = torch_c.to_builtin_tensor %49 : !torch.vtensor<[128,768],f32> -> tensor<128x768xf32>
    %51 = dnn.mm %50, %cst_2 : tensor<128x768xf32>, tensor<768x768xf32> -> tensor<128x768xf32>
    %52 = torch_c.from_builtin_tensor %51 : tensor<128x768xf32> -> !torch.vtensor<[128,768],f32>
    %53 = torch.aten.mul.Scalar %52, %int1 : !torch.vtensor<[128,768],f32>, !torch.int -> !torch.vtensor<[128,768],f32>
    %54 = torch_c.to_builtin_tensor %53 : !torch.vtensor<[128,768],f32> -> tensor<128x768xf32>
    %55 = dnn.add %54, %cst_5 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<128x768xf32>, tensor<768xf32>) -> tensor<128x768xf32>
    %56 = torch_c.from_builtin_tensor %55 : tensor<128x768xf32> -> !torch.vtensor<[128,768],f32>
    %57 = torch.prim.ListConstruct %int1, %int128, %int768 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %58 = torch.aten.view %56, %57 : !torch.vtensor<[128,768],f32>, !torch.list<int> -> !torch.vtensor<[1,128,768],f32>
    %59 = torch_c.to_builtin_tensor %58 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %60 = torch_c.to_builtin_tensor %12 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %61 = dnn.add %59, %60 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<1x128x768xf32>, tensor<1x128x768xf32>) -> tensor<1x128x768xf32>
    %62 = torch_c.from_builtin_tensor %61 : tensor<1x128x768xf32> -> !torch.vtensor<[1,128,768],f32>
    %63 = torch_c.to_builtin_tensor %62 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %64 = dnn.layer_norm %63, %cst_5, %cst_5 {parameter_indices = array<i32: 1, 4, 5>, parameters = [[768], 1.000000e-05, false]} : (tensor<1x128x768xf32>, tensor<768xf32>, tensor<768xf32>) -> tensor<1x128x768xf32>
    %65 = torch_c.from_builtin_tensor %64 : tensor<1x128x768xf32> -> !torch.vtensor<[1,128,768],f32>
    %66 = torch.prim.ListConstruct %int-1, %int768 : (!torch.int, !torch.int) -> !torch.list<int>
    %67 = torch.aten.view %65, %66 : !torch.vtensor<[1,128,768],f32>, !torch.list<int> -> !torch.vtensor<[128,768],f32>
    %68 = torch_c.to_builtin_tensor %67 : !torch.vtensor<[128,768],f32> -> tensor<128x768xf32>
    %69 = dnn.mm %68, %cst_1 : tensor<128x768xf32>, tensor<768x3072xf32> -> tensor<128x3072xf32>
    %70 = torch_c.from_builtin_tensor %69 : tensor<128x3072xf32> -> !torch.vtensor<[128,3072],f32>
    %71 = torch.aten.mul.Scalar %70, %int1 : !torch.vtensor<[128,3072],f32>, !torch.int -> !torch.vtensor<[128,3072],f32>
    %72 = torch_c.to_builtin_tensor %71 : !torch.vtensor<[128,3072],f32> -> tensor<128x3072xf32>
    %73 = dnn.add %72, %cst_0 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<128x3072xf32>, tensor<3072xf32>) -> tensor<128x3072xf32>
    %74 = torch_c.from_builtin_tensor %73 : tensor<128x3072xf32> -> !torch.vtensor<[128,3072],f32>
    %75 = torch.prim.ListConstruct %int1, %int128, %int3072 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %76 = torch.aten.view %74, %75 : !torch.vtensor<[128,3072],f32>, !torch.list<int> -> !torch.vtensor<[1,128,3072],f32>
    %77 = torch_c.to_builtin_tensor %76 : !torch.vtensor<[1,128,3072],f32> -> tensor<1x128x3072xf32>
    %78 = dnn.gelu %77 {parameter_indices = array<i32: 1>, parameters = ["tanh"]} : (tensor<1x128x3072xf32>) -> tensor<1x128x3072xf32>
    %79 = torch_c.from_builtin_tensor %78 : tensor<1x128x3072xf32> -> !torch.vtensor<[1,128,3072],f32>
    %80 = torch.prim.ListConstruct %int-1, %int3072 : (!torch.int, !torch.int) -> !torch.list<int>
    %81 = torch.aten.view %79, %80 : !torch.vtensor<[1,128,3072],f32>, !torch.list<int> -> !torch.vtensor<[128,3072],f32>
    %82 = torch_c.to_builtin_tensor %81 : !torch.vtensor<[128,3072],f32> -> tensor<128x3072xf32>
    %83 = dnn.mm %82, %cst : tensor<128x3072xf32>, tensor<3072x768xf32> -> tensor<128x768xf32>
    %84 = torch_c.from_builtin_tensor %83 : tensor<128x768xf32> -> !torch.vtensor<[128,768],f32>
    %85 = torch.aten.mul.Scalar %84, %int1 : !torch.vtensor<[128,768],f32>, !torch.int -> !torch.vtensor<[128,768],f32>
    %86 = torch_c.to_builtin_tensor %85 : !torch.vtensor<[128,768],f32> -> tensor<128x768xf32>
    %87 = dnn.add %86, %cst_5 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<128x768xf32>, tensor<768xf32>) -> tensor<128x768xf32>
    %88 = torch_c.from_builtin_tensor %87 : tensor<128x768xf32> -> !torch.vtensor<[128,768],f32>
    %89 = torch.prim.ListConstruct %int1, %int128, %int768 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %90 = torch.aten.view %88, %89 : !torch.vtensor<[128,768],f32>, !torch.list<int> -> !torch.vtensor<[1,128,768],f32>
    %91 = torch_c.to_builtin_tensor %62 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %92 = torch_c.to_builtin_tensor %90 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %93 = dnn.add %91, %92 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<1x128x768xf32>, tensor<1x128x768xf32>) -> tensor<1x128x768xf32>
    %94 = torch_c.from_builtin_tensor %93 : tensor<1x128x768xf32> -> !torch.vtensor<[1,128,768],f32>
    %95 = torch_c.to_builtin_tensor %94 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %96 = dnn.layer_norm %95, %cst_5, %cst_5 {parameter_indices = array<i32: 1, 4, 5>, parameters = [[768], 1.000000e-05, false]} : (tensor<1x128x768xf32>, tensor<768xf32>, tensor<768xf32>) -> tensor<1x128x768xf32>
    %97 = torch_c.from_builtin_tensor %96 : tensor<1x128x768xf32> -> !torch.vtensor<[1,128,768],f32>
    %98 = torch.prim.ListConstruct %int-1, %int768 : (!torch.int, !torch.int) -> !torch.list<int>
    %99 = torch.aten.view %97, %98 : !torch.vtensor<[1,128,768],f32>, !torch.list<int> -> !torch.vtensor<[128,768],f32>
    %100 = torch_c.to_builtin_tensor %99 : !torch.vtensor<[128,768],f32> -> tensor<128x768xf32>
    %101 = dnn.mm %100, %cst_4 : tensor<128x768xf32>, tensor<768x2304xf32> -> tensor<128x2304xf32>
    %102 = torch_c.from_builtin_tensor %101 : tensor<128x2304xf32> -> !torch.vtensor<[128,2304],f32>
    %103 = torch.aten.mul.Scalar %102, %int1 : !torch.vtensor<[128,2304],f32>, !torch.int -> !torch.vtensor<[128,2304],f32>
    %104 = torch_c.to_builtin_tensor %103 : !torch.vtensor<[128,2304],f32> -> tensor<128x2304xf32>
    %105 = dnn.add %104, %cst_3 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<128x2304xf32>, tensor<2304xf32>) -> tensor<128x2304xf32>
    %106 = torch_c.from_builtin_tensor %105 : tensor<128x2304xf32> -> !torch.vtensor<[128,2304],f32>
    %107 = torch.prim.ListConstruct %int1, %int128, %int2304 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %108 = torch.aten.view %106, %107 : !torch.vtensor<[128,2304],f32>, !torch.list<int> -> !torch.vtensor<[1,128,2304],f32>
    %109 = torch.aten.slice.Tensor %108, %int2, %int0, %int768, %int1 : !torch.vtensor<[1,128,2304],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,128,768],f32>
    %110 = torch.aten.slice.Tensor %108, %int2, %int768, %int1536, %int1 : !torch.vtensor<[1,128,2304],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,128,768],f32>
    %111 = torch.aten.slice.Tensor %108, %int2, %int1536, %int2304, %int1 : !torch.vtensor<[1,128,2304],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,128,768],f32>
    %112 = torch.prim.ListConstruct %int1, %int128, %int-1, %int64 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %113 = torch.aten.view %110, %112 : !torch.vtensor<[1,128,768],f32>, !torch.list<int> -> !torch.vtensor<[1,128,12,64],f32>
    %114 = torch.aten.transpose.int %113, %int1, %int2 : !torch.vtensor<[1,128,12,64],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,12,128,64],f32>
    %115 = torch.prim.ListConstruct %int1, %int128, %int-1, %int64 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %116 = torch.aten.view %111, %115 : !torch.vtensor<[1,128,768],f32>, !torch.list<int> -> !torch.vtensor<[1,128,12,64],f32>
    %117 = torch.aten.transpose.int %116, %int1, %int2 : !torch.vtensor<[1,128,12,64],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,12,128,64],f32>
    %118 = torch.prim.ListConstruct %int1, %int128, %int-1, %int64 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %119 = torch.aten.view %109, %118 : !torch.vtensor<[1,128,768],f32>, !torch.list<int> -> !torch.vtensor<[1,128,12,64],f32>
    %120 = torch.aten.transpose.int %119, %int1, %int2 : !torch.vtensor<[1,128,12,64],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,12,128,64],f32>
    %121 = torch_c.to_builtin_tensor %120 : !torch.vtensor<[1,12,128,64],f32> -> tensor<1x12x128x64xf32>
    %122 = torch_c.to_builtin_tensor %114 : !torch.vtensor<[1,12,128,64],f32> -> tensor<1x12x128x64xf32>
    %123 = torch_c.to_builtin_tensor %117 : !torch.vtensor<[1,12,128,64],f32> -> tensor<1x12x128x64xf32>
    %124 = torch_c.to_builtin_tensor %arg2 : !torch.vtensor<[1,1,128,128],i1> -> tensor<1x1x128x128xi1>
    %125 = dnn.scaled_dot_product_attention %121, %122, %123, %124 {parameter_indices = array<i32: 4, 5, 6, 7>, parameters = [0.000000e+00, false, 1.250000e-01, false]} : (tensor<1x12x128x64xf32>, tensor<1x12x128x64xf32>, tensor<1x12x128x64xf32>, tensor<1x1x128x128xi1>) -> tensor<1x12x128x64xf32>
    %126 = torch_c.from_builtin_tensor %125 : tensor<1x12x128x64xf32> -> !torch.vtensor<[1,12,128,64],f32>
    %127 = torch.aten.transpose.int %126, %int1, %int2 : !torch.vtensor<[1,12,128,64],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,128,12,64],f32>
    %128 = torch.prim.ListConstruct %int1, %int128, %int-1 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %129 = torch.aten.view %127, %128 : !torch.vtensor<[1,128,12,64],f32>, !torch.list<int> -> !torch.vtensor<[1,128,768],f32>
    %130 = torch.prim.ListConstruct %int-1, %int768 : (!torch.int, !torch.int) -> !torch.list<int>
    %131 = torch.aten.view %129, %130 : !torch.vtensor<[1,128,768],f32>, !torch.list<int> -> !torch.vtensor<[128,768],f32>
    %132 = torch_c.to_builtin_tensor %131 : !torch.vtensor<[128,768],f32> -> tensor<128x768xf32>
    %133 = dnn.mm %132, %cst_2 : tensor<128x768xf32>, tensor<768x768xf32> -> tensor<128x768xf32>
    %134 = torch_c.from_builtin_tensor %133 : tensor<128x768xf32> -> !torch.vtensor<[128,768],f32>
    %135 = torch.aten.mul.Scalar %134, %int1 : !torch.vtensor<[128,768],f32>, !torch.int -> !torch.vtensor<[128,768],f32>
    %136 = torch_c.to_builtin_tensor %135 : !torch.vtensor<[128,768],f32> -> tensor<128x768xf32>
    %137 = dnn.add %136, %cst_5 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<128x768xf32>, tensor<768xf32>) -> tensor<128x768xf32>
    %138 = torch_c.from_builtin_tensor %137 : tensor<128x768xf32> -> !torch.vtensor<[128,768],f32>
    %139 = torch.prim.ListConstruct %int1, %int128, %int768 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %140 = torch.aten.view %138, %139 : !torch.vtensor<[128,768],f32>, !torch.list<int> -> !torch.vtensor<[1,128,768],f32>
    %141 = torch_c.to_builtin_tensor %140 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %142 = torch_c.to_builtin_tensor %94 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %143 = dnn.add %141, %142 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<1x128x768xf32>, tensor<1x128x768xf32>) -> tensor<1x128x768xf32>
    %144 = torch_c.from_builtin_tensor %143 : tensor<1x128x768xf32> -> !torch.vtensor<[1,128,768],f32>
    %145 = torch_c.to_builtin_tensor %144 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %146 = dnn.layer_norm %145, %cst_5, %cst_5 {parameter_indices = array<i32: 1, 4, 5>, parameters = [[768], 1.000000e-05, false]} : (tensor<1x128x768xf32>, tensor<768xf32>, tensor<768xf32>) -> tensor<1x128x768xf32>
    %147 = torch_c.from_builtin_tensor %146 : tensor<1x128x768xf32> -> !torch.vtensor<[1,128,768],f32>
    %148 = torch.prim.ListConstruct %int-1, %int768 : (!torch.int, !torch.int) -> !torch.list<int>
    %149 = torch.aten.view %147, %148 : !torch.vtensor<[1,128,768],f32>, !torch.list<int> -> !torch.vtensor<[128,768],f32>
    %150 = torch_c.to_builtin_tensor %149 : !torch.vtensor<[128,768],f32> -> tensor<128x768xf32>
    %151 = dnn.mm %150, %cst_1 : tensor<128x768xf32>, tensor<768x3072xf32> -> tensor<128x3072xf32>
    %152 = torch_c.from_builtin_tensor %151 : tensor<128x3072xf32> -> !torch.vtensor<[128,3072],f32>
    %153 = torch.aten.mul.Scalar %152, %int1 : !torch.vtensor<[128,3072],f32>, !torch.int -> !torch.vtensor<[128,3072],f32>
    %154 = torch_c.to_builtin_tensor %153 : !torch.vtensor<[128,3072],f32> -> tensor<128x3072xf32>
    %155 = dnn.add %154, %cst_0 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<128x3072xf32>, tensor<3072xf32>) -> tensor<128x3072xf32>
    %156 = torch_c.from_builtin_tensor %155 : tensor<128x3072xf32> -> !torch.vtensor<[128,3072],f32>
    %157 = torch.prim.ListConstruct %int1, %int128, %int3072 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %158 = torch.aten.view %156, %157 : !torch.vtensor<[128,3072],f32>, !torch.list<int> -> !torch.vtensor<[1,128,3072],f32>
    %159 = torch_c.to_builtin_tensor %158 : !torch.vtensor<[1,128,3072],f32> -> tensor<1x128x3072xf32>
    %160 = dnn.gelu %159 {parameter_indices = array<i32: 1>, parameters = ["tanh"]} : (tensor<1x128x3072xf32>) -> tensor<1x128x3072xf32>
    %161 = torch_c.from_builtin_tensor %160 : tensor<1x128x3072xf32> -> !torch.vtensor<[1,128,3072],f32>
    %162 = torch.prim.ListConstruct %int-1, %int3072 : (!torch.int, !torch.int) -> !torch.list<int>
    %163 = torch.aten.view %161, %162 : !torch.vtensor<[1,128,3072],f32>, !torch.list<int> -> !torch.vtensor<[128,3072],f32>
    %164 = torch_c.to_builtin_tensor %163 : !torch.vtensor<[128,3072],f32> -> tensor<128x3072xf32>
    %165 = dnn.mm %164, %cst : tensor<128x3072xf32>, tensor<3072x768xf32> -> tensor<128x768xf32>
    %166 = torch_c.from_builtin_tensor %165 : tensor<128x768xf32> -> !torch.vtensor<[128,768],f32>
    %167 = torch.aten.mul.Scalar %166, %int1 : !torch.vtensor<[128,768],f32>, !torch.int -> !torch.vtensor<[128,768],f32>
    %168 = torch_c.to_builtin_tensor %167 : !torch.vtensor<[128,768],f32> -> tensor<128x768xf32>
    %169 = dnn.add %168, %cst_5 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<128x768xf32>, tensor<768xf32>) -> tensor<128x768xf32>
    %170 = torch_c.from_builtin_tensor %169 : tensor<128x768xf32> -> !torch.vtensor<[128,768],f32>
    %171 = torch.prim.ListConstruct %int1, %int128, %int768 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %172 = torch.aten.view %170, %171 : !torch.vtensor<[128,768],f32>, !torch.list<int> -> !torch.vtensor<[1,128,768],f32>
    %173 = torch_c.to_builtin_tensor %144 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %174 = torch_c.to_builtin_tensor %172 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %175 = dnn.add %173, %174 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<1x128x768xf32>, tensor<1x128x768xf32>) -> tensor<1x128x768xf32>
    %176 = torch_c.from_builtin_tensor %175 : tensor<1x128x768xf32> -> !torch.vtensor<[1,128,768],f32>
    %177 = torch_c.to_builtin_tensor %176 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %178 = dnn.layer_norm %177, %cst_5, %cst_5 {parameter_indices = array<i32: 1, 4, 5>, parameters = [[768], 1.000000e-05, false]} : (tensor<1x128x768xf32>, tensor<768xf32>, tensor<768xf32>) -> tensor<1x128x768xf32>
    %179 = torch_c.from_builtin_tensor %178 : tensor<1x128x768xf32> -> !torch.vtensor<[1,128,768],f32>
    %180 = torch.prim.ListConstruct %int-1, %int768 : (!torch.int, !torch.int) -> !torch.list<int>
    %181 = torch.aten.view %179, %180 : !torch.vtensor<[1,128,768],f32>, !torch.list<int> -> !torch.vtensor<[128,768],f32>
    %182 = torch_c.to_builtin_tensor %181 : !torch.vtensor<[128,768],f32> -> tensor<128x768xf32>
    %183 = dnn.mm %182, %cst_4 : tensor<128x768xf32>, tensor<768x2304xf32> -> tensor<128x2304xf32>
    %184 = torch_c.from_builtin_tensor %183 : tensor<128x2304xf32> -> !torch.vtensor<[128,2304],f32>
    %185 = torch.aten.mul.Scalar %184, %int1 : !torch.vtensor<[128,2304],f32>, !torch.int -> !torch.vtensor<[128,2304],f32>
    %186 = torch_c.to_builtin_tensor %185 : !torch.vtensor<[128,2304],f32> -> tensor<128x2304xf32>
    %187 = dnn.add %186, %cst_3 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<128x2304xf32>, tensor<2304xf32>) -> tensor<128x2304xf32>
    %188 = torch_c.from_builtin_tensor %187 : tensor<128x2304xf32> -> !torch.vtensor<[128,2304],f32>
    %189 = torch.prim.ListConstruct %int1, %int128, %int2304 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %190 = torch.aten.view %188, %189 : !torch.vtensor<[128,2304],f32>, !torch.list<int> -> !torch.vtensor<[1,128,2304],f32>
    %191 = torch.aten.slice.Tensor %190, %int2, %int0, %int768, %int1 : !torch.vtensor<[1,128,2304],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,128,768],f32>
    %192 = torch.aten.slice.Tensor %190, %int2, %int768, %int1536, %int1 : !torch.vtensor<[1,128,2304],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,128,768],f32>
    %193 = torch.aten.slice.Tensor %190, %int2, %int1536, %int2304, %int1 : !torch.vtensor<[1,128,2304],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,128,768],f32>
    %194 = torch.prim.ListConstruct %int1, %int128, %int-1, %int64 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %195 = torch.aten.view %192, %194 : !torch.vtensor<[1,128,768],f32>, !torch.list<int> -> !torch.vtensor<[1,128,12,64],f32>
    %196 = torch.aten.transpose.int %195, %int1, %int2 : !torch.vtensor<[1,128,12,64],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,12,128,64],f32>
    %197 = torch.prim.ListConstruct %int1, %int128, %int-1, %int64 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %198 = torch.aten.view %193, %197 : !torch.vtensor<[1,128,768],f32>, !torch.list<int> -> !torch.vtensor<[1,128,12,64],f32>
    %199 = torch.aten.transpose.int %198, %int1, %int2 : !torch.vtensor<[1,128,12,64],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,12,128,64],f32>
    %200 = torch.prim.ListConstruct %int1, %int128, %int-1, %int64 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %201 = torch.aten.view %191, %200 : !torch.vtensor<[1,128,768],f32>, !torch.list<int> -> !torch.vtensor<[1,128,12,64],f32>
    %202 = torch.aten.transpose.int %201, %int1, %int2 : !torch.vtensor<[1,128,12,64],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,12,128,64],f32>
    %203 = torch_c.to_builtin_tensor %202 : !torch.vtensor<[1,12,128,64],f32> -> tensor<1x12x128x64xf32>
    %204 = torch_c.to_builtin_tensor %196 : !torch.vtensor<[1,12,128,64],f32> -> tensor<1x12x128x64xf32>
    %205 = torch_c.to_builtin_tensor %199 : !torch.vtensor<[1,12,128,64],f32> -> tensor<1x12x128x64xf32>
    %206 = torch_c.to_builtin_tensor %arg2 : !torch.vtensor<[1,1,128,128],i1> -> tensor<1x1x128x128xi1>
    %207 = dnn.scaled_dot_product_attention %203, %204, %205, %206 {parameter_indices = array<i32: 4, 5, 6, 7>, parameters = [0.000000e+00, false, 1.250000e-01, false]} : (tensor<1x12x128x64xf32>, tensor<1x12x128x64xf32>, tensor<1x12x128x64xf32>, tensor<1x1x128x128xi1>) -> tensor<1x12x128x64xf32>
    %208 = torch_c.from_builtin_tensor %207 : tensor<1x12x128x64xf32> -> !torch.vtensor<[1,12,128,64],f32>
    %209 = torch.aten.transpose.int %208, %int1, %int2 : !torch.vtensor<[1,12,128,64],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,128,12,64],f32>
    %210 = torch.prim.ListConstruct %int1, %int128, %int-1 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %211 = torch.aten.view %209, %210 : !torch.vtensor<[1,128,12,64],f32>, !torch.list<int> -> !torch.vtensor<[1,128,768],f32>
    %212 = torch.prim.ListConstruct %int-1, %int768 : (!torch.int, !torch.int) -> !torch.list<int>
    %213 = torch.aten.view %211, %212 : !torch.vtensor<[1,128,768],f32>, !torch.list<int> -> !torch.vtensor<[128,768],f32>
    %214 = torch_c.to_builtin_tensor %213 : !torch.vtensor<[128,768],f32> -> tensor<128x768xf32>
    %215 = dnn.mm %214, %cst_2 : tensor<128x768xf32>, tensor<768x768xf32> -> tensor<128x768xf32>
    %216 = torch_c.from_builtin_tensor %215 : tensor<128x768xf32> -> !torch.vtensor<[128,768],f32>
    %217 = torch.aten.mul.Scalar %216, %int1 : !torch.vtensor<[128,768],f32>, !torch.int -> !torch.vtensor<[128,768],f32>
    %218 = torch_c.to_builtin_tensor %217 : !torch.vtensor<[128,768],f32> -> tensor<128x768xf32>
    %219 = dnn.add %218, %cst_5 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<128x768xf32>, tensor<768xf32>) -> tensor<128x768xf32>
    %220 = torch_c.from_builtin_tensor %219 : tensor<128x768xf32> -> !torch.vtensor<[128,768],f32>
    %221 = torch.prim.ListConstruct %int1, %int128, %int768 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %222 = torch.aten.view %220, %221 : !torch.vtensor<[128,768],f32>, !torch.list<int> -> !torch.vtensor<[1,128,768],f32>
    %223 = torch_c.to_builtin_tensor %222 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %224 = torch_c.to_builtin_tensor %176 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %225 = dnn.add %223, %224 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<1x128x768xf32>, tensor<1x128x768xf32>) -> tensor<1x128x768xf32>
    %226 = torch_c.from_builtin_tensor %225 : tensor<1x128x768xf32> -> !torch.vtensor<[1,128,768],f32>
    %227 = torch_c.to_builtin_tensor %226 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %228 = dnn.layer_norm %227, %cst_5, %cst_5 {parameter_indices = array<i32: 1, 4, 5>, parameters = [[768], 1.000000e-05, false]} : (tensor<1x128x768xf32>, tensor<768xf32>, tensor<768xf32>) -> tensor<1x128x768xf32>
    %229 = torch_c.from_builtin_tensor %228 : tensor<1x128x768xf32> -> !torch.vtensor<[1,128,768],f32>
    %230 = torch.prim.ListConstruct %int-1, %int768 : (!torch.int, !torch.int) -> !torch.list<int>
    %231 = torch.aten.view %229, %230 : !torch.vtensor<[1,128,768],f32>, !torch.list<int> -> !torch.vtensor<[128,768],f32>
    %232 = torch_c.to_builtin_tensor %231 : !torch.vtensor<[128,768],f32> -> tensor<128x768xf32>
    %233 = dnn.mm %232, %cst_1 : tensor<128x768xf32>, tensor<768x3072xf32> -> tensor<128x3072xf32>
    %234 = torch_c.from_builtin_tensor %233 : tensor<128x3072xf32> -> !torch.vtensor<[128,3072],f32>
    %235 = torch.aten.mul.Scalar %234, %int1 : !torch.vtensor<[128,3072],f32>, !torch.int -> !torch.vtensor<[128,3072],f32>
    %236 = torch_c.to_builtin_tensor %235 : !torch.vtensor<[128,3072],f32> -> tensor<128x3072xf32>
    %237 = dnn.add %236, %cst_0 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<128x3072xf32>, tensor<3072xf32>) -> tensor<128x3072xf32>
    %238 = torch_c.from_builtin_tensor %237 : tensor<128x3072xf32> -> !torch.vtensor<[128,3072],f32>
    %239 = torch.prim.ListConstruct %int1, %int128, %int3072 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %240 = torch.aten.view %238, %239 : !torch.vtensor<[128,3072],f32>, !torch.list<int> -> !torch.vtensor<[1,128,3072],f32>
    %241 = torch_c.to_builtin_tensor %240 : !torch.vtensor<[1,128,3072],f32> -> tensor<1x128x3072xf32>
    %242 = dnn.gelu %241 {parameter_indices = array<i32: 1>, parameters = ["tanh"]} : (tensor<1x128x3072xf32>) -> tensor<1x128x3072xf32>
    %243 = torch_c.from_builtin_tensor %242 : tensor<1x128x3072xf32> -> !torch.vtensor<[1,128,3072],f32>
    %244 = torch.prim.ListConstruct %int-1, %int3072 : (!torch.int, !torch.int) -> !torch.list<int>
    %245 = torch.aten.view %243, %244 : !torch.vtensor<[1,128,3072],f32>, !torch.list<int> -> !torch.vtensor<[128,3072],f32>
    %246 = torch_c.to_builtin_tensor %245 : !torch.vtensor<[128,3072],f32> -> tensor<128x3072xf32>
    %247 = dnn.mm %246, %cst : tensor<128x3072xf32>, tensor<3072x768xf32> -> tensor<128x768xf32>
    %248 = torch_c.from_builtin_tensor %247 : tensor<128x768xf32> -> !torch.vtensor<[128,768],f32>
    %249 = torch.aten.mul.Scalar %248, %int1 : !torch.vtensor<[128,768],f32>, !torch.int -> !torch.vtensor<[128,768],f32>
    %250 = torch_c.to_builtin_tensor %249 : !torch.vtensor<[128,768],f32> -> tensor<128x768xf32>
    %251 = dnn.add %250, %cst_5 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<128x768xf32>, tensor<768xf32>) -> tensor<128x768xf32>
    %252 = torch_c.from_builtin_tensor %251 : tensor<128x768xf32> -> !torch.vtensor<[128,768],f32>
    %253 = torch.prim.ListConstruct %int1, %int128, %int768 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %254 = torch.aten.view %252, %253 : !torch.vtensor<[128,768],f32>, !torch.list<int> -> !torch.vtensor<[1,128,768],f32>
    %255 = torch_c.to_builtin_tensor %226 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %256 = torch_c.to_builtin_tensor %254 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %257 = dnn.add %255, %256 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<1x128x768xf32>, tensor<1x128x768xf32>) -> tensor<1x128x768xf32>
    %258 = torch_c.from_builtin_tensor %257 : tensor<1x128x768xf32> -> !torch.vtensor<[1,128,768],f32>
    %259 = torch_c.to_builtin_tensor %258 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %260 = dnn.layer_norm %259, %cst_5, %cst_5 {parameter_indices = array<i32: 1, 4, 5>, parameters = [[768], 1.000000e-05, false]} : (tensor<1x128x768xf32>, tensor<768xf32>, tensor<768xf32>) -> tensor<1x128x768xf32>
    %261 = torch_c.from_builtin_tensor %260 : tensor<1x128x768xf32> -> !torch.vtensor<[1,128,768],f32>
    %262 = torch.prim.ListConstruct %int-1, %int768 : (!torch.int, !torch.int) -> !torch.list<int>
    %263 = torch.aten.view %261, %262 : !torch.vtensor<[1,128,768],f32>, !torch.list<int> -> !torch.vtensor<[128,768],f32>
    %264 = torch_c.to_builtin_tensor %263 : !torch.vtensor<[128,768],f32> -> tensor<128x768xf32>
    %265 = dnn.mm %264, %cst_4 : tensor<128x768xf32>, tensor<768x2304xf32> -> tensor<128x2304xf32>
    %266 = torch_c.from_builtin_tensor %265 : tensor<128x2304xf32> -> !torch.vtensor<[128,2304],f32>
    %267 = torch.aten.mul.Scalar %266, %int1 : !torch.vtensor<[128,2304],f32>, !torch.int -> !torch.vtensor<[128,2304],f32>
    %268 = torch_c.to_builtin_tensor %267 : !torch.vtensor<[128,2304],f32> -> tensor<128x2304xf32>
    %269 = dnn.add %268, %cst_3 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<128x2304xf32>, tensor<2304xf32>) -> tensor<128x2304xf32>
    %270 = torch_c.from_builtin_tensor %269 : tensor<128x2304xf32> -> !torch.vtensor<[128,2304],f32>
    %271 = torch.prim.ListConstruct %int1, %int128, %int2304 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %272 = torch.aten.view %270, %271 : !torch.vtensor<[128,2304],f32>, !torch.list<int> -> !torch.vtensor<[1,128,2304],f32>
    %273 = torch.aten.slice.Tensor %272, %int2, %int0, %int768, %int1 : !torch.vtensor<[1,128,2304],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,128,768],f32>
    %274 = torch.aten.slice.Tensor %272, %int2, %int768, %int1536, %int1 : !torch.vtensor<[1,128,2304],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,128,768],f32>
    %275 = torch.aten.slice.Tensor %272, %int2, %int1536, %int2304, %int1 : !torch.vtensor<[1,128,2304],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,128,768],f32>
    %276 = torch.prim.ListConstruct %int1, %int128, %int-1, %int64 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %277 = torch.aten.view %274, %276 : !torch.vtensor<[1,128,768],f32>, !torch.list<int> -> !torch.vtensor<[1,128,12,64],f32>
    %278 = torch.aten.transpose.int %277, %int1, %int2 : !torch.vtensor<[1,128,12,64],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,12,128,64],f32>
    %279 = torch.prim.ListConstruct %int1, %int128, %int-1, %int64 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %280 = torch.aten.view %275, %279 : !torch.vtensor<[1,128,768],f32>, !torch.list<int> -> !torch.vtensor<[1,128,12,64],f32>
    %281 = torch.aten.transpose.int %280, %int1, %int2 : !torch.vtensor<[1,128,12,64],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,12,128,64],f32>
    %282 = torch.prim.ListConstruct %int1, %int128, %int-1, %int64 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %283 = torch.aten.view %273, %282 : !torch.vtensor<[1,128,768],f32>, !torch.list<int> -> !torch.vtensor<[1,128,12,64],f32>
    %284 = torch.aten.transpose.int %283, %int1, %int2 : !torch.vtensor<[1,128,12,64],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,12,128,64],f32>
    %285 = torch_c.to_builtin_tensor %284 : !torch.vtensor<[1,12,128,64],f32> -> tensor<1x12x128x64xf32>
    %286 = torch_c.to_builtin_tensor %278 : !torch.vtensor<[1,12,128,64],f32> -> tensor<1x12x128x64xf32>
    %287 = torch_c.to_builtin_tensor %281 : !torch.vtensor<[1,12,128,64],f32> -> tensor<1x12x128x64xf32>
    %288 = torch_c.to_builtin_tensor %arg2 : !torch.vtensor<[1,1,128,128],i1> -> tensor<1x1x128x128xi1>
    %289 = dnn.scaled_dot_product_attention %285, %286, %287, %288 {parameter_indices = array<i32: 4, 5, 6, 7>, parameters = [0.000000e+00, false, 1.250000e-01, false]} : (tensor<1x12x128x64xf32>, tensor<1x12x128x64xf32>, tensor<1x12x128x64xf32>, tensor<1x1x128x128xi1>) -> tensor<1x12x128x64xf32>
    %290 = torch_c.from_builtin_tensor %289 : tensor<1x12x128x64xf32> -> !torch.vtensor<[1,12,128,64],f32>
    %291 = torch.aten.transpose.int %290, %int1, %int2 : !torch.vtensor<[1,12,128,64],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,128,12,64],f32>
    %292 = torch.prim.ListConstruct %int1, %int128, %int-1 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %293 = torch.aten.view %291, %292 : !torch.vtensor<[1,128,12,64],f32>, !torch.list<int> -> !torch.vtensor<[1,128,768],f32>
    %294 = torch.prim.ListConstruct %int-1, %int768 : (!torch.int, !torch.int) -> !torch.list<int>
    %295 = torch.aten.view %293, %294 : !torch.vtensor<[1,128,768],f32>, !torch.list<int> -> !torch.vtensor<[128,768],f32>
    %296 = torch_c.to_builtin_tensor %295 : !torch.vtensor<[128,768],f32> -> tensor<128x768xf32>
    %297 = dnn.mm %296, %cst_2 : tensor<128x768xf32>, tensor<768x768xf32> -> tensor<128x768xf32>
    %298 = torch_c.from_builtin_tensor %297 : tensor<128x768xf32> -> !torch.vtensor<[128,768],f32>
    %299 = torch.aten.mul.Scalar %298, %int1 : !torch.vtensor<[128,768],f32>, !torch.int -> !torch.vtensor<[128,768],f32>
    %300 = torch_c.to_builtin_tensor %299 : !torch.vtensor<[128,768],f32> -> tensor<128x768xf32>
    %301 = dnn.add %300, %cst_5 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<128x768xf32>, tensor<768xf32>) -> tensor<128x768xf32>
    %302 = torch_c.from_builtin_tensor %301 : tensor<128x768xf32> -> !torch.vtensor<[128,768],f32>
    %303 = torch.prim.ListConstruct %int1, %int128, %int768 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %304 = torch.aten.view %302, %303 : !torch.vtensor<[128,768],f32>, !torch.list<int> -> !torch.vtensor<[1,128,768],f32>
    %305 = torch_c.to_builtin_tensor %304 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %306 = torch_c.to_builtin_tensor %258 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %307 = dnn.add %305, %306 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<1x128x768xf32>, tensor<1x128x768xf32>) -> tensor<1x128x768xf32>
    %308 = torch_c.from_builtin_tensor %307 : tensor<1x128x768xf32> -> !torch.vtensor<[1,128,768],f32>
    %309 = torch_c.to_builtin_tensor %308 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %310 = dnn.layer_norm %309, %cst_5, %cst_5 {parameter_indices = array<i32: 1, 4, 5>, parameters = [[768], 1.000000e-05, false]} : (tensor<1x128x768xf32>, tensor<768xf32>, tensor<768xf32>) -> tensor<1x128x768xf32>
    %311 = torch_c.from_builtin_tensor %310 : tensor<1x128x768xf32> -> !torch.vtensor<[1,128,768],f32>
    %312 = torch.prim.ListConstruct %int-1, %int768 : (!torch.int, !torch.int) -> !torch.list<int>
    %313 = torch.aten.view %311, %312 : !torch.vtensor<[1,128,768],f32>, !torch.list<int> -> !torch.vtensor<[128,768],f32>
    %314 = torch_c.to_builtin_tensor %313 : !torch.vtensor<[128,768],f32> -> tensor<128x768xf32>
    %315 = dnn.mm %314, %cst_1 : tensor<128x768xf32>, tensor<768x3072xf32> -> tensor<128x3072xf32>
    %316 = torch_c.from_builtin_tensor %315 : tensor<128x3072xf32> -> !torch.vtensor<[128,3072],f32>
    %317 = torch.aten.mul.Scalar %316, %int1 : !torch.vtensor<[128,3072],f32>, !torch.int -> !torch.vtensor<[128,3072],f32>
    %318 = torch_c.to_builtin_tensor %317 : !torch.vtensor<[128,3072],f32> -> tensor<128x3072xf32>
    %319 = dnn.add %318, %cst_0 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<128x3072xf32>, tensor<3072xf32>) -> tensor<128x3072xf32>
    %320 = torch_c.from_builtin_tensor %319 : tensor<128x3072xf32> -> !torch.vtensor<[128,3072],f32>
    %321 = torch.prim.ListConstruct %int1, %int128, %int3072 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %322 = torch.aten.view %320, %321 : !torch.vtensor<[128,3072],f32>, !torch.list<int> -> !torch.vtensor<[1,128,3072],f32>
    %323 = torch_c.to_builtin_tensor %322 : !torch.vtensor<[1,128,3072],f32> -> tensor<1x128x3072xf32>
    %324 = dnn.gelu %323 {parameter_indices = array<i32: 1>, parameters = ["tanh"]} : (tensor<1x128x3072xf32>) -> tensor<1x128x3072xf32>
    %325 = torch_c.from_builtin_tensor %324 : tensor<1x128x3072xf32> -> !torch.vtensor<[1,128,3072],f32>
    %326 = torch.prim.ListConstruct %int-1, %int3072 : (!torch.int, !torch.int) -> !torch.list<int>
    %327 = torch.aten.view %325, %326 : !torch.vtensor<[1,128,3072],f32>, !torch.list<int> -> !torch.vtensor<[128,3072],f32>
    %328 = torch_c.to_builtin_tensor %327 : !torch.vtensor<[128,3072],f32> -> tensor<128x3072xf32>
    %329 = dnn.mm %328, %cst : tensor<128x3072xf32>, tensor<3072x768xf32> -> tensor<128x768xf32>
    %330 = torch_c.from_builtin_tensor %329 : tensor<128x768xf32> -> !torch.vtensor<[128,768],f32>
    %331 = torch.aten.mul.Scalar %330, %int1 : !torch.vtensor<[128,768],f32>, !torch.int -> !torch.vtensor<[128,768],f32>
    %332 = torch_c.to_builtin_tensor %331 : !torch.vtensor<[128,768],f32> -> tensor<128x768xf32>
    %333 = dnn.add %332, %cst_5 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<128x768xf32>, tensor<768xf32>) -> tensor<128x768xf32>
    %334 = torch_c.from_builtin_tensor %333 : tensor<128x768xf32> -> !torch.vtensor<[128,768],f32>
    %335 = torch.prim.ListConstruct %int1, %int128, %int768 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %336 = torch.aten.view %334, %335 : !torch.vtensor<[128,768],f32>, !torch.list<int> -> !torch.vtensor<[1,128,768],f32>
    %337 = torch_c.to_builtin_tensor %308 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %338 = torch_c.to_builtin_tensor %336 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %339 = dnn.add %337, %338 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<1x128x768xf32>, tensor<1x128x768xf32>) -> tensor<1x128x768xf32>
    %340 = torch_c.from_builtin_tensor %339 : tensor<1x128x768xf32> -> !torch.vtensor<[1,128,768],f32>
    %341 = torch_c.to_builtin_tensor %340 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %342 = dnn.layer_norm %341, %cst_5, %cst_5 {parameter_indices = array<i32: 1, 4, 5>, parameters = [[768], 1.000000e-05, false]} : (tensor<1x128x768xf32>, tensor<768xf32>, tensor<768xf32>) -> tensor<1x128x768xf32>
    %343 = torch_c.from_builtin_tensor %342 : tensor<1x128x768xf32> -> !torch.vtensor<[1,128,768],f32>
    %344 = torch.prim.ListConstruct %int-1, %int768 : (!torch.int, !torch.int) -> !torch.list<int>
    %345 = torch.aten.view %343, %344 : !torch.vtensor<[1,128,768],f32>, !torch.list<int> -> !torch.vtensor<[128,768],f32>
    %346 = torch_c.to_builtin_tensor %345 : !torch.vtensor<[128,768],f32> -> tensor<128x768xf32>
    %347 = dnn.mm %346, %cst_4 : tensor<128x768xf32>, tensor<768x2304xf32> -> tensor<128x2304xf32>
    %348 = torch_c.from_builtin_tensor %347 : tensor<128x2304xf32> -> !torch.vtensor<[128,2304],f32>
    %349 = torch.aten.mul.Scalar %348, %int1 : !torch.vtensor<[128,2304],f32>, !torch.int -> !torch.vtensor<[128,2304],f32>
    %350 = torch_c.to_builtin_tensor %349 : !torch.vtensor<[128,2304],f32> -> tensor<128x2304xf32>
    %351 = dnn.add %350, %cst_3 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<128x2304xf32>, tensor<2304xf32>) -> tensor<128x2304xf32>
    %352 = torch_c.from_builtin_tensor %351 : tensor<128x2304xf32> -> !torch.vtensor<[128,2304],f32>
    %353 = torch.prim.ListConstruct %int1, %int128, %int2304 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %354 = torch.aten.view %352, %353 : !torch.vtensor<[128,2304],f32>, !torch.list<int> -> !torch.vtensor<[1,128,2304],f32>
    %355 = torch.aten.slice.Tensor %354, %int2, %int0, %int768, %int1 : !torch.vtensor<[1,128,2304],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,128,768],f32>
    %356 = torch.aten.slice.Tensor %354, %int2, %int768, %int1536, %int1 : !torch.vtensor<[1,128,2304],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,128,768],f32>
    %357 = torch.aten.slice.Tensor %354, %int2, %int1536, %int2304, %int1 : !torch.vtensor<[1,128,2304],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,128,768],f32>
    %358 = torch.prim.ListConstruct %int1, %int128, %int-1, %int64 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %359 = torch.aten.view %356, %358 : !torch.vtensor<[1,128,768],f32>, !torch.list<int> -> !torch.vtensor<[1,128,12,64],f32>
    %360 = torch.aten.transpose.int %359, %int1, %int2 : !torch.vtensor<[1,128,12,64],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,12,128,64],f32>
    %361 = torch.prim.ListConstruct %int1, %int128, %int-1, %int64 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %362 = torch.aten.view %357, %361 : !torch.vtensor<[1,128,768],f32>, !torch.list<int> -> !torch.vtensor<[1,128,12,64],f32>
    %363 = torch.aten.transpose.int %362, %int1, %int2 : !torch.vtensor<[1,128,12,64],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,12,128,64],f32>
    %364 = torch.prim.ListConstruct %int1, %int128, %int-1, %int64 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %365 = torch.aten.view %355, %364 : !torch.vtensor<[1,128,768],f32>, !torch.list<int> -> !torch.vtensor<[1,128,12,64],f32>
    %366 = torch.aten.transpose.int %365, %int1, %int2 : !torch.vtensor<[1,128,12,64],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,12,128,64],f32>
    %367 = torch_c.to_builtin_tensor %366 : !torch.vtensor<[1,12,128,64],f32> -> tensor<1x12x128x64xf32>
    %368 = torch_c.to_builtin_tensor %360 : !torch.vtensor<[1,12,128,64],f32> -> tensor<1x12x128x64xf32>
    %369 = torch_c.to_builtin_tensor %363 : !torch.vtensor<[1,12,128,64],f32> -> tensor<1x12x128x64xf32>
    %370 = torch_c.to_builtin_tensor %arg2 : !torch.vtensor<[1,1,128,128],i1> -> tensor<1x1x128x128xi1>
    %371 = dnn.scaled_dot_product_attention %367, %368, %369, %370 {parameter_indices = array<i32: 4, 5, 6, 7>, parameters = [0.000000e+00, false, 1.250000e-01, false]} : (tensor<1x12x128x64xf32>, tensor<1x12x128x64xf32>, tensor<1x12x128x64xf32>, tensor<1x1x128x128xi1>) -> tensor<1x12x128x64xf32>
    %372 = torch_c.from_builtin_tensor %371 : tensor<1x12x128x64xf32> -> !torch.vtensor<[1,12,128,64],f32>
    %373 = torch.aten.transpose.int %372, %int1, %int2 : !torch.vtensor<[1,12,128,64],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,128,12,64],f32>
    %374 = torch.prim.ListConstruct %int1, %int128, %int-1 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %375 = torch.aten.view %373, %374 : !torch.vtensor<[1,128,12,64],f32>, !torch.list<int> -> !torch.vtensor<[1,128,768],f32>
    %376 = torch.prim.ListConstruct %int-1, %int768 : (!torch.int, !torch.int) -> !torch.list<int>
    %377 = torch.aten.view %375, %376 : !torch.vtensor<[1,128,768],f32>, !torch.list<int> -> !torch.vtensor<[128,768],f32>
    %378 = torch_c.to_builtin_tensor %377 : !torch.vtensor<[128,768],f32> -> tensor<128x768xf32>
    %379 = dnn.mm %378, %cst_2 : tensor<128x768xf32>, tensor<768x768xf32> -> tensor<128x768xf32>
    %380 = torch_c.from_builtin_tensor %379 : tensor<128x768xf32> -> !torch.vtensor<[128,768],f32>
    %381 = torch.aten.mul.Scalar %380, %int1 : !torch.vtensor<[128,768],f32>, !torch.int -> !torch.vtensor<[128,768],f32>
    %382 = torch_c.to_builtin_tensor %381 : !torch.vtensor<[128,768],f32> -> tensor<128x768xf32>
    %383 = dnn.add %382, %cst_5 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<128x768xf32>, tensor<768xf32>) -> tensor<128x768xf32>
    %384 = torch_c.from_builtin_tensor %383 : tensor<128x768xf32> -> !torch.vtensor<[128,768],f32>
    %385 = torch.prim.ListConstruct %int1, %int128, %int768 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %386 = torch.aten.view %384, %385 : !torch.vtensor<[128,768],f32>, !torch.list<int> -> !torch.vtensor<[1,128,768],f32>
    %387 = torch_c.to_builtin_tensor %386 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %388 = torch_c.to_builtin_tensor %340 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %389 = dnn.add %387, %388 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<1x128x768xf32>, tensor<1x128x768xf32>) -> tensor<1x128x768xf32>
    %390 = torch_c.from_builtin_tensor %389 : tensor<1x128x768xf32> -> !torch.vtensor<[1,128,768],f32>
    %391 = torch_c.to_builtin_tensor %390 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %392 = dnn.layer_norm %391, %cst_5, %cst_5 {parameter_indices = array<i32: 1, 4, 5>, parameters = [[768], 1.000000e-05, false]} : (tensor<1x128x768xf32>, tensor<768xf32>, tensor<768xf32>) -> tensor<1x128x768xf32>
    %393 = torch_c.from_builtin_tensor %392 : tensor<1x128x768xf32> -> !torch.vtensor<[1,128,768],f32>
    %394 = torch.prim.ListConstruct %int-1, %int768 : (!torch.int, !torch.int) -> !torch.list<int>
    %395 = torch.aten.view %393, %394 : !torch.vtensor<[1,128,768],f32>, !torch.list<int> -> !torch.vtensor<[128,768],f32>
    %396 = torch_c.to_builtin_tensor %395 : !torch.vtensor<[128,768],f32> -> tensor<128x768xf32>
    %397 = dnn.mm %396, %cst_1 : tensor<128x768xf32>, tensor<768x3072xf32> -> tensor<128x3072xf32>
    %398 = torch_c.from_builtin_tensor %397 : tensor<128x3072xf32> -> !torch.vtensor<[128,3072],f32>
    %399 = torch.aten.mul.Scalar %398, %int1 : !torch.vtensor<[128,3072],f32>, !torch.int -> !torch.vtensor<[128,3072],f32>
    %400 = torch_c.to_builtin_tensor %399 : !torch.vtensor<[128,3072],f32> -> tensor<128x3072xf32>
    %401 = dnn.add %400, %cst_0 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<128x3072xf32>, tensor<3072xf32>) -> tensor<128x3072xf32>
    %402 = torch_c.from_builtin_tensor %401 : tensor<128x3072xf32> -> !torch.vtensor<[128,3072],f32>
    %403 = torch.prim.ListConstruct %int1, %int128, %int3072 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %404 = torch.aten.view %402, %403 : !torch.vtensor<[128,3072],f32>, !torch.list<int> -> !torch.vtensor<[1,128,3072],f32>
    %405 = torch_c.to_builtin_tensor %404 : !torch.vtensor<[1,128,3072],f32> -> tensor<1x128x3072xf32>
    %406 = dnn.gelu %405 {parameter_indices = array<i32: 1>, parameters = ["tanh"]} : (tensor<1x128x3072xf32>) -> tensor<1x128x3072xf32>
    %407 = torch_c.from_builtin_tensor %406 : tensor<1x128x3072xf32> -> !torch.vtensor<[1,128,3072],f32>
    %408 = torch.prim.ListConstruct %int-1, %int3072 : (!torch.int, !torch.int) -> !torch.list<int>
    %409 = torch.aten.view %407, %408 : !torch.vtensor<[1,128,3072],f32>, !torch.list<int> -> !torch.vtensor<[128,3072],f32>
    %410 = torch_c.to_builtin_tensor %409 : !torch.vtensor<[128,3072],f32> -> tensor<128x3072xf32>
    %411 = dnn.mm %410, %cst : tensor<128x3072xf32>, tensor<3072x768xf32> -> tensor<128x768xf32>
    %412 = torch_c.from_builtin_tensor %411 : tensor<128x768xf32> -> !torch.vtensor<[128,768],f32>
    %413 = torch.aten.mul.Scalar %412, %int1 : !torch.vtensor<[128,768],f32>, !torch.int -> !torch.vtensor<[128,768],f32>
    %414 = torch_c.to_builtin_tensor %413 : !torch.vtensor<[128,768],f32> -> tensor<128x768xf32>
    %415 = dnn.add %414, %cst_5 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<128x768xf32>, tensor<768xf32>) -> tensor<128x768xf32>
    %416 = torch_c.from_builtin_tensor %415 : tensor<128x768xf32> -> !torch.vtensor<[128,768],f32>
    %417 = torch.prim.ListConstruct %int1, %int128, %int768 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %418 = torch.aten.view %416, %417 : !torch.vtensor<[128,768],f32>, !torch.list<int> -> !torch.vtensor<[1,128,768],f32>
    %419 = torch_c.to_builtin_tensor %390 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %420 = torch_c.to_builtin_tensor %418 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %421 = dnn.add %419, %420 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<1x128x768xf32>, tensor<1x128x768xf32>) -> tensor<1x128x768xf32>
    %422 = torch_c.from_builtin_tensor %421 : tensor<1x128x768xf32> -> !torch.vtensor<[1,128,768],f32>
    %423 = torch_c.to_builtin_tensor %422 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %424 = dnn.layer_norm %423, %cst_5, %cst_5 {parameter_indices = array<i32: 1, 4, 5>, parameters = [[768], 1.000000e-05, false]} : (tensor<1x128x768xf32>, tensor<768xf32>, tensor<768xf32>) -> tensor<1x128x768xf32>
    %425 = torch_c.from_builtin_tensor %424 : tensor<1x128x768xf32> -> !torch.vtensor<[1,128,768],f32>
    %426 = torch.prim.ListConstruct %int-1, %int768 : (!torch.int, !torch.int) -> !torch.list<int>
    %427 = torch.aten.view %425, %426 : !torch.vtensor<[1,128,768],f32>, !torch.list<int> -> !torch.vtensor<[128,768],f32>
    %428 = torch_c.to_builtin_tensor %427 : !torch.vtensor<[128,768],f32> -> tensor<128x768xf32>
    %429 = dnn.mm %428, %cst_4 : tensor<128x768xf32>, tensor<768x2304xf32> -> tensor<128x2304xf32>
    %430 = torch_c.from_builtin_tensor %429 : tensor<128x2304xf32> -> !torch.vtensor<[128,2304],f32>
    %431 = torch.aten.mul.Scalar %430, %int1 : !torch.vtensor<[128,2304],f32>, !torch.int -> !torch.vtensor<[128,2304],f32>
    %432 = torch_c.to_builtin_tensor %431 : !torch.vtensor<[128,2304],f32> -> tensor<128x2304xf32>
    %433 = dnn.add %432, %cst_3 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<128x2304xf32>, tensor<2304xf32>) -> tensor<128x2304xf32>
    %434 = torch_c.from_builtin_tensor %433 : tensor<128x2304xf32> -> !torch.vtensor<[128,2304],f32>
    %435 = torch.prim.ListConstruct %int1, %int128, %int2304 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %436 = torch.aten.view %434, %435 : !torch.vtensor<[128,2304],f32>, !torch.list<int> -> !torch.vtensor<[1,128,2304],f32>
    %437 = torch.aten.slice.Tensor %436, %int2, %int0, %int768, %int1 : !torch.vtensor<[1,128,2304],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,128,768],f32>
    %438 = torch.aten.slice.Tensor %436, %int2, %int768, %int1536, %int1 : !torch.vtensor<[1,128,2304],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,128,768],f32>
    %439 = torch.aten.slice.Tensor %436, %int2, %int1536, %int2304, %int1 : !torch.vtensor<[1,128,2304],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,128,768],f32>
    %440 = torch.prim.ListConstruct %int1, %int128, %int-1, %int64 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %441 = torch.aten.view %438, %440 : !torch.vtensor<[1,128,768],f32>, !torch.list<int> -> !torch.vtensor<[1,128,12,64],f32>
    %442 = torch.aten.transpose.int %441, %int1, %int2 : !torch.vtensor<[1,128,12,64],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,12,128,64],f32>
    %443 = torch.prim.ListConstruct %int1, %int128, %int-1, %int64 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %444 = torch.aten.view %439, %443 : !torch.vtensor<[1,128,768],f32>, !torch.list<int> -> !torch.vtensor<[1,128,12,64],f32>
    %445 = torch.aten.transpose.int %444, %int1, %int2 : !torch.vtensor<[1,128,12,64],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,12,128,64],f32>
    %446 = torch.prim.ListConstruct %int1, %int128, %int-1, %int64 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %447 = torch.aten.view %437, %446 : !torch.vtensor<[1,128,768],f32>, !torch.list<int> -> !torch.vtensor<[1,128,12,64],f32>
    %448 = torch.aten.transpose.int %447, %int1, %int2 : !torch.vtensor<[1,128,12,64],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,12,128,64],f32>
    %449 = torch_c.to_builtin_tensor %448 : !torch.vtensor<[1,12,128,64],f32> -> tensor<1x12x128x64xf32>
    %450 = torch_c.to_builtin_tensor %442 : !torch.vtensor<[1,12,128,64],f32> -> tensor<1x12x128x64xf32>
    %451 = torch_c.to_builtin_tensor %445 : !torch.vtensor<[1,12,128,64],f32> -> tensor<1x12x128x64xf32>
    %452 = torch_c.to_builtin_tensor %arg2 : !torch.vtensor<[1,1,128,128],i1> -> tensor<1x1x128x128xi1>
    %453 = dnn.scaled_dot_product_attention %449, %450, %451, %452 {parameter_indices = array<i32: 4, 5, 6, 7>, parameters = [0.000000e+00, false, 1.250000e-01, false]} : (tensor<1x12x128x64xf32>, tensor<1x12x128x64xf32>, tensor<1x12x128x64xf32>, tensor<1x1x128x128xi1>) -> tensor<1x12x128x64xf32>
    %454 = torch_c.from_builtin_tensor %453 : tensor<1x12x128x64xf32> -> !torch.vtensor<[1,12,128,64],f32>
    %455 = torch.aten.transpose.int %454, %int1, %int2 : !torch.vtensor<[1,12,128,64],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,128,12,64],f32>
    %456 = torch.prim.ListConstruct %int1, %int128, %int-1 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %457 = torch.aten.view %455, %456 : !torch.vtensor<[1,128,12,64],f32>, !torch.list<int> -> !torch.vtensor<[1,128,768],f32>
    %458 = torch.prim.ListConstruct %int-1, %int768 : (!torch.int, !torch.int) -> !torch.list<int>
    %459 = torch.aten.view %457, %458 : !torch.vtensor<[1,128,768],f32>, !torch.list<int> -> !torch.vtensor<[128,768],f32>
    %460 = torch_c.to_builtin_tensor %459 : !torch.vtensor<[128,768],f32> -> tensor<128x768xf32>
    %461 = dnn.mm %460, %cst_2 : tensor<128x768xf32>, tensor<768x768xf32> -> tensor<128x768xf32>
    %462 = torch_c.from_builtin_tensor %461 : tensor<128x768xf32> -> !torch.vtensor<[128,768],f32>
    %463 = torch.aten.mul.Scalar %462, %int1 : !torch.vtensor<[128,768],f32>, !torch.int -> !torch.vtensor<[128,768],f32>
    %464 = torch_c.to_builtin_tensor %463 : !torch.vtensor<[128,768],f32> -> tensor<128x768xf32>
    %465 = dnn.add %464, %cst_5 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<128x768xf32>, tensor<768xf32>) -> tensor<128x768xf32>
    %466 = torch_c.from_builtin_tensor %465 : tensor<128x768xf32> -> !torch.vtensor<[128,768],f32>
    %467 = torch.prim.ListConstruct %int1, %int128, %int768 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %468 = torch.aten.view %466, %467 : !torch.vtensor<[128,768],f32>, !torch.list<int> -> !torch.vtensor<[1,128,768],f32>
    %469 = torch_c.to_builtin_tensor %468 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %470 = torch_c.to_builtin_tensor %422 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %471 = dnn.add %469, %470 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<1x128x768xf32>, tensor<1x128x768xf32>) -> tensor<1x128x768xf32>
    %472 = torch_c.from_builtin_tensor %471 : tensor<1x128x768xf32> -> !torch.vtensor<[1,128,768],f32>
    %473 = torch_c.to_builtin_tensor %472 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %474 = dnn.layer_norm %473, %cst_5, %cst_5 {parameter_indices = array<i32: 1, 4, 5>, parameters = [[768], 1.000000e-05, false]} : (tensor<1x128x768xf32>, tensor<768xf32>, tensor<768xf32>) -> tensor<1x128x768xf32>
    %475 = torch_c.from_builtin_tensor %474 : tensor<1x128x768xf32> -> !torch.vtensor<[1,128,768],f32>
    %476 = torch.prim.ListConstruct %int-1, %int768 : (!torch.int, !torch.int) -> !torch.list<int>
    %477 = torch.aten.view %475, %476 : !torch.vtensor<[1,128,768],f32>, !torch.list<int> -> !torch.vtensor<[128,768],f32>
    %478 = torch_c.to_builtin_tensor %477 : !torch.vtensor<[128,768],f32> -> tensor<128x768xf32>
    %479 = dnn.mm %478, %cst_1 : tensor<128x768xf32>, tensor<768x3072xf32> -> tensor<128x3072xf32>
    %480 = torch_c.from_builtin_tensor %479 : tensor<128x3072xf32> -> !torch.vtensor<[128,3072],f32>
    %481 = torch.aten.mul.Scalar %480, %int1 : !torch.vtensor<[128,3072],f32>, !torch.int -> !torch.vtensor<[128,3072],f32>
    %482 = torch_c.to_builtin_tensor %481 : !torch.vtensor<[128,3072],f32> -> tensor<128x3072xf32>
    %483 = dnn.add %482, %cst_0 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<128x3072xf32>, tensor<3072xf32>) -> tensor<128x3072xf32>
    %484 = torch_c.from_builtin_tensor %483 : tensor<128x3072xf32> -> !torch.vtensor<[128,3072],f32>
    %485 = torch.prim.ListConstruct %int1, %int128, %int3072 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %486 = torch.aten.view %484, %485 : !torch.vtensor<[128,3072],f32>, !torch.list<int> -> !torch.vtensor<[1,128,3072],f32>
    %487 = torch_c.to_builtin_tensor %486 : !torch.vtensor<[1,128,3072],f32> -> tensor<1x128x3072xf32>
    %488 = dnn.gelu %487 {parameter_indices = array<i32: 1>, parameters = ["tanh"]} : (tensor<1x128x3072xf32>) -> tensor<1x128x3072xf32>
    %489 = torch_c.from_builtin_tensor %488 : tensor<1x128x3072xf32> -> !torch.vtensor<[1,128,3072],f32>
    %490 = torch.prim.ListConstruct %int-1, %int3072 : (!torch.int, !torch.int) -> !torch.list<int>
    %491 = torch.aten.view %489, %490 : !torch.vtensor<[1,128,3072],f32>, !torch.list<int> -> !torch.vtensor<[128,3072],f32>
    %492 = torch_c.to_builtin_tensor %491 : !torch.vtensor<[128,3072],f32> -> tensor<128x3072xf32>
    %493 = dnn.mm %492, %cst : tensor<128x3072xf32>, tensor<3072x768xf32> -> tensor<128x768xf32>
    %494 = torch_c.from_builtin_tensor %493 : tensor<128x768xf32> -> !torch.vtensor<[128,768],f32>
    %495 = torch.aten.mul.Scalar %494, %int1 : !torch.vtensor<[128,768],f32>, !torch.int -> !torch.vtensor<[128,768],f32>
    %496 = torch_c.to_builtin_tensor %495 : !torch.vtensor<[128,768],f32> -> tensor<128x768xf32>
    %497 = dnn.add %496, %cst_5 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<128x768xf32>, tensor<768xf32>) -> tensor<128x768xf32>
    %498 = torch_c.from_builtin_tensor %497 : tensor<128x768xf32> -> !torch.vtensor<[128,768],f32>
    %499 = torch.prim.ListConstruct %int1, %int128, %int768 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %500 = torch.aten.view %498, %499 : !torch.vtensor<[128,768],f32>, !torch.list<int> -> !torch.vtensor<[1,128,768],f32>
    %501 = torch_c.to_builtin_tensor %472 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %502 = torch_c.to_builtin_tensor %500 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %503 = dnn.add %501, %502 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<1x128x768xf32>, tensor<1x128x768xf32>) -> tensor<1x128x768xf32>
    %504 = torch_c.from_builtin_tensor %503 : tensor<1x128x768xf32> -> !torch.vtensor<[1,128,768],f32>
    %505 = torch_c.to_builtin_tensor %504 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %506 = dnn.layer_norm %505, %cst_5, %cst_5 {parameter_indices = array<i32: 1, 4, 5>, parameters = [[768], 1.000000e-05, false]} : (tensor<1x128x768xf32>, tensor<768xf32>, tensor<768xf32>) -> tensor<1x128x768xf32>
    %507 = torch_c.from_builtin_tensor %506 : tensor<1x128x768xf32> -> !torch.vtensor<[1,128,768],f32>
    %508 = torch.prim.ListConstruct %int-1, %int768 : (!torch.int, !torch.int) -> !torch.list<int>
    %509 = torch.aten.view %507, %508 : !torch.vtensor<[1,128,768],f32>, !torch.list<int> -> !torch.vtensor<[128,768],f32>
    %510 = torch_c.to_builtin_tensor %509 : !torch.vtensor<[128,768],f32> -> tensor<128x768xf32>
    %511 = dnn.mm %510, %cst_4 : tensor<128x768xf32>, tensor<768x2304xf32> -> tensor<128x2304xf32>
    %512 = torch_c.from_builtin_tensor %511 : tensor<128x2304xf32> -> !torch.vtensor<[128,2304],f32>
    %513 = torch.aten.mul.Scalar %512, %int1 : !torch.vtensor<[128,2304],f32>, !torch.int -> !torch.vtensor<[128,2304],f32>
    %514 = torch_c.to_builtin_tensor %513 : !torch.vtensor<[128,2304],f32> -> tensor<128x2304xf32>
    %515 = dnn.add %514, %cst_3 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<128x2304xf32>, tensor<2304xf32>) -> tensor<128x2304xf32>
    %516 = torch_c.from_builtin_tensor %515 : tensor<128x2304xf32> -> !torch.vtensor<[128,2304],f32>
    %517 = torch.prim.ListConstruct %int1, %int128, %int2304 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %518 = torch.aten.view %516, %517 : !torch.vtensor<[128,2304],f32>, !torch.list<int> -> !torch.vtensor<[1,128,2304],f32>
    %519 = torch.aten.slice.Tensor %518, %int2, %int0, %int768, %int1 : !torch.vtensor<[1,128,2304],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,128,768],f32>
    %520 = torch.aten.slice.Tensor %518, %int2, %int768, %int1536, %int1 : !torch.vtensor<[1,128,2304],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,128,768],f32>
    %521 = torch.aten.slice.Tensor %518, %int2, %int1536, %int2304, %int1 : !torch.vtensor<[1,128,2304],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,128,768],f32>
    %522 = torch.prim.ListConstruct %int1, %int128, %int-1, %int64 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %523 = torch.aten.view %520, %522 : !torch.vtensor<[1,128,768],f32>, !torch.list<int> -> !torch.vtensor<[1,128,12,64],f32>
    %524 = torch.aten.transpose.int %523, %int1, %int2 : !torch.vtensor<[1,128,12,64],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,12,128,64],f32>
    %525 = torch.prim.ListConstruct %int1, %int128, %int-1, %int64 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %526 = torch.aten.view %521, %525 : !torch.vtensor<[1,128,768],f32>, !torch.list<int> -> !torch.vtensor<[1,128,12,64],f32>
    %527 = torch.aten.transpose.int %526, %int1, %int2 : !torch.vtensor<[1,128,12,64],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,12,128,64],f32>
    %528 = torch.prim.ListConstruct %int1, %int128, %int-1, %int64 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %529 = torch.aten.view %519, %528 : !torch.vtensor<[1,128,768],f32>, !torch.list<int> -> !torch.vtensor<[1,128,12,64],f32>
    %530 = torch.aten.transpose.int %529, %int1, %int2 : !torch.vtensor<[1,128,12,64],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,12,128,64],f32>
    %531 = torch_c.to_builtin_tensor %530 : !torch.vtensor<[1,12,128,64],f32> -> tensor<1x12x128x64xf32>
    %532 = torch_c.to_builtin_tensor %524 : !torch.vtensor<[1,12,128,64],f32> -> tensor<1x12x128x64xf32>
    %533 = torch_c.to_builtin_tensor %527 : !torch.vtensor<[1,12,128,64],f32> -> tensor<1x12x128x64xf32>
    %534 = torch_c.to_builtin_tensor %arg2 : !torch.vtensor<[1,1,128,128],i1> -> tensor<1x1x128x128xi1>
    %535 = dnn.scaled_dot_product_attention %531, %532, %533, %534 {parameter_indices = array<i32: 4, 5, 6, 7>, parameters = [0.000000e+00, false, 1.250000e-01, false]} : (tensor<1x12x128x64xf32>, tensor<1x12x128x64xf32>, tensor<1x12x128x64xf32>, tensor<1x1x128x128xi1>) -> tensor<1x12x128x64xf32>
    %536 = torch_c.from_builtin_tensor %535 : tensor<1x12x128x64xf32> -> !torch.vtensor<[1,12,128,64],f32>
    %537 = torch.aten.transpose.int %536, %int1, %int2 : !torch.vtensor<[1,12,128,64],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,128,12,64],f32>
    %538 = torch.prim.ListConstruct %int1, %int128, %int-1 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %539 = torch.aten.view %537, %538 : !torch.vtensor<[1,128,12,64],f32>, !torch.list<int> -> !torch.vtensor<[1,128,768],f32>
    %540 = torch.prim.ListConstruct %int-1, %int768 : (!torch.int, !torch.int) -> !torch.list<int>
    %541 = torch.aten.view %539, %540 : !torch.vtensor<[1,128,768],f32>, !torch.list<int> -> !torch.vtensor<[128,768],f32>
    %542 = torch_c.to_builtin_tensor %541 : !torch.vtensor<[128,768],f32> -> tensor<128x768xf32>
    %543 = dnn.mm %542, %cst_2 : tensor<128x768xf32>, tensor<768x768xf32> -> tensor<128x768xf32>
    %544 = torch_c.from_builtin_tensor %543 : tensor<128x768xf32> -> !torch.vtensor<[128,768],f32>
    %545 = torch.aten.mul.Scalar %544, %int1 : !torch.vtensor<[128,768],f32>, !torch.int -> !torch.vtensor<[128,768],f32>
    %546 = torch_c.to_builtin_tensor %545 : !torch.vtensor<[128,768],f32> -> tensor<128x768xf32>
    %547 = dnn.add %546, %cst_5 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<128x768xf32>, tensor<768xf32>) -> tensor<128x768xf32>
    %548 = torch_c.from_builtin_tensor %547 : tensor<128x768xf32> -> !torch.vtensor<[128,768],f32>
    %549 = torch.prim.ListConstruct %int1, %int128, %int768 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %550 = torch.aten.view %548, %549 : !torch.vtensor<[128,768],f32>, !torch.list<int> -> !torch.vtensor<[1,128,768],f32>
    %551 = torch_c.to_builtin_tensor %550 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %552 = torch_c.to_builtin_tensor %504 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %553 = dnn.add %551, %552 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<1x128x768xf32>, tensor<1x128x768xf32>) -> tensor<1x128x768xf32>
    %554 = torch_c.from_builtin_tensor %553 : tensor<1x128x768xf32> -> !torch.vtensor<[1,128,768],f32>
    %555 = torch_c.to_builtin_tensor %554 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %556 = dnn.layer_norm %555, %cst_5, %cst_5 {parameter_indices = array<i32: 1, 4, 5>, parameters = [[768], 1.000000e-05, false]} : (tensor<1x128x768xf32>, tensor<768xf32>, tensor<768xf32>) -> tensor<1x128x768xf32>
    %557 = torch_c.from_builtin_tensor %556 : tensor<1x128x768xf32> -> !torch.vtensor<[1,128,768],f32>
    %558 = torch.prim.ListConstruct %int-1, %int768 : (!torch.int, !torch.int) -> !torch.list<int>
    %559 = torch.aten.view %557, %558 : !torch.vtensor<[1,128,768],f32>, !torch.list<int> -> !torch.vtensor<[128,768],f32>
    %560 = torch_c.to_builtin_tensor %559 : !torch.vtensor<[128,768],f32> -> tensor<128x768xf32>
    %561 = dnn.mm %560, %cst_1 : tensor<128x768xf32>, tensor<768x3072xf32> -> tensor<128x3072xf32>
    %562 = torch_c.from_builtin_tensor %561 : tensor<128x3072xf32> -> !torch.vtensor<[128,3072],f32>
    %563 = torch.aten.mul.Scalar %562, %int1 : !torch.vtensor<[128,3072],f32>, !torch.int -> !torch.vtensor<[128,3072],f32>
    %564 = torch_c.to_builtin_tensor %563 : !torch.vtensor<[128,3072],f32> -> tensor<128x3072xf32>
    %565 = dnn.add %564, %cst_0 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<128x3072xf32>, tensor<3072xf32>) -> tensor<128x3072xf32>
    %566 = torch_c.from_builtin_tensor %565 : tensor<128x3072xf32> -> !torch.vtensor<[128,3072],f32>
    %567 = torch.prim.ListConstruct %int1, %int128, %int3072 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %568 = torch.aten.view %566, %567 : !torch.vtensor<[128,3072],f32>, !torch.list<int> -> !torch.vtensor<[1,128,3072],f32>
    %569 = torch_c.to_builtin_tensor %568 : !torch.vtensor<[1,128,3072],f32> -> tensor<1x128x3072xf32>
    %570 = dnn.gelu %569 {parameter_indices = array<i32: 1>, parameters = ["tanh"]} : (tensor<1x128x3072xf32>) -> tensor<1x128x3072xf32>
    %571 = torch_c.from_builtin_tensor %570 : tensor<1x128x3072xf32> -> !torch.vtensor<[1,128,3072],f32>
    %572 = torch.prim.ListConstruct %int-1, %int3072 : (!torch.int, !torch.int) -> !torch.list<int>
    %573 = torch.aten.view %571, %572 : !torch.vtensor<[1,128,3072],f32>, !torch.list<int> -> !torch.vtensor<[128,3072],f32>
    %574 = torch_c.to_builtin_tensor %573 : !torch.vtensor<[128,3072],f32> -> tensor<128x3072xf32>
    %575 = dnn.mm %574, %cst : tensor<128x3072xf32>, tensor<3072x768xf32> -> tensor<128x768xf32>
    %576 = torch_c.from_builtin_tensor %575 : tensor<128x768xf32> -> !torch.vtensor<[128,768],f32>
    %577 = torch.aten.mul.Scalar %576, %int1 : !torch.vtensor<[128,768],f32>, !torch.int -> !torch.vtensor<[128,768],f32>
    %578 = torch_c.to_builtin_tensor %577 : !torch.vtensor<[128,768],f32> -> tensor<128x768xf32>
    %579 = dnn.add %578, %cst_5 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<128x768xf32>, tensor<768xf32>) -> tensor<128x768xf32>
    %580 = torch_c.from_builtin_tensor %579 : tensor<128x768xf32> -> !torch.vtensor<[128,768],f32>
    %581 = torch.prim.ListConstruct %int1, %int128, %int768 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %582 = torch.aten.view %580, %581 : !torch.vtensor<[128,768],f32>, !torch.list<int> -> !torch.vtensor<[1,128,768],f32>
    %583 = torch_c.to_builtin_tensor %554 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %584 = torch_c.to_builtin_tensor %582 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %585 = dnn.add %583, %584 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<1x128x768xf32>, tensor<1x128x768xf32>) -> tensor<1x128x768xf32>
    %586 = torch_c.from_builtin_tensor %585 : tensor<1x128x768xf32> -> !torch.vtensor<[1,128,768],f32>
    %587 = torch_c.to_builtin_tensor %586 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %588 = dnn.layer_norm %587, %cst_5, %cst_5 {parameter_indices = array<i32: 1, 4, 5>, parameters = [[768], 1.000000e-05, false]} : (tensor<1x128x768xf32>, tensor<768xf32>, tensor<768xf32>) -> tensor<1x128x768xf32>
    %589 = torch_c.from_builtin_tensor %588 : tensor<1x128x768xf32> -> !torch.vtensor<[1,128,768],f32>
    %590 = torch.prim.ListConstruct %int-1, %int768 : (!torch.int, !torch.int) -> !torch.list<int>
    %591 = torch.aten.view %589, %590 : !torch.vtensor<[1,128,768],f32>, !torch.list<int> -> !torch.vtensor<[128,768],f32>
    %592 = torch_c.to_builtin_tensor %591 : !torch.vtensor<[128,768],f32> -> tensor<128x768xf32>
    %593 = dnn.mm %592, %cst_4 : tensor<128x768xf32>, tensor<768x2304xf32> -> tensor<128x2304xf32>
    %594 = torch_c.from_builtin_tensor %593 : tensor<128x2304xf32> -> !torch.vtensor<[128,2304],f32>
    %595 = torch.aten.mul.Scalar %594, %int1 : !torch.vtensor<[128,2304],f32>, !torch.int -> !torch.vtensor<[128,2304],f32>
    %596 = torch_c.to_builtin_tensor %595 : !torch.vtensor<[128,2304],f32> -> tensor<128x2304xf32>
    %597 = dnn.add %596, %cst_3 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<128x2304xf32>, tensor<2304xf32>) -> tensor<128x2304xf32>
    %598 = torch_c.from_builtin_tensor %597 : tensor<128x2304xf32> -> !torch.vtensor<[128,2304],f32>
    %599 = torch.prim.ListConstruct %int1, %int128, %int2304 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %600 = torch.aten.view %598, %599 : !torch.vtensor<[128,2304],f32>, !torch.list<int> -> !torch.vtensor<[1,128,2304],f32>
    %601 = torch.aten.slice.Tensor %600, %int2, %int0, %int768, %int1 : !torch.vtensor<[1,128,2304],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,128,768],f32>
    %602 = torch.aten.slice.Tensor %600, %int2, %int768, %int1536, %int1 : !torch.vtensor<[1,128,2304],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,128,768],f32>
    %603 = torch.aten.slice.Tensor %600, %int2, %int1536, %int2304, %int1 : !torch.vtensor<[1,128,2304],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,128,768],f32>
    %604 = torch.prim.ListConstruct %int1, %int128, %int-1, %int64 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %605 = torch.aten.view %602, %604 : !torch.vtensor<[1,128,768],f32>, !torch.list<int> -> !torch.vtensor<[1,128,12,64],f32>
    %606 = torch.aten.transpose.int %605, %int1, %int2 : !torch.vtensor<[1,128,12,64],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,12,128,64],f32>
    %607 = torch.prim.ListConstruct %int1, %int128, %int-1, %int64 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %608 = torch.aten.view %603, %607 : !torch.vtensor<[1,128,768],f32>, !torch.list<int> -> !torch.vtensor<[1,128,12,64],f32>
    %609 = torch.aten.transpose.int %608, %int1, %int2 : !torch.vtensor<[1,128,12,64],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,12,128,64],f32>
    %610 = torch.prim.ListConstruct %int1, %int128, %int-1, %int64 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %611 = torch.aten.view %601, %610 : !torch.vtensor<[1,128,768],f32>, !torch.list<int> -> !torch.vtensor<[1,128,12,64],f32>
    %612 = torch.aten.transpose.int %611, %int1, %int2 : !torch.vtensor<[1,128,12,64],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,12,128,64],f32>
    %613 = torch_c.to_builtin_tensor %612 : !torch.vtensor<[1,12,128,64],f32> -> tensor<1x12x128x64xf32>
    %614 = torch_c.to_builtin_tensor %606 : !torch.vtensor<[1,12,128,64],f32> -> tensor<1x12x128x64xf32>
    %615 = torch_c.to_builtin_tensor %609 : !torch.vtensor<[1,12,128,64],f32> -> tensor<1x12x128x64xf32>
    %616 = torch_c.to_builtin_tensor %arg2 : !torch.vtensor<[1,1,128,128],i1> -> tensor<1x1x128x128xi1>
    %617 = dnn.scaled_dot_product_attention %613, %614, %615, %616 {parameter_indices = array<i32: 4, 5, 6, 7>, parameters = [0.000000e+00, false, 1.250000e-01, false]} : (tensor<1x12x128x64xf32>, tensor<1x12x128x64xf32>, tensor<1x12x128x64xf32>, tensor<1x1x128x128xi1>) -> tensor<1x12x128x64xf32>
    %618 = torch_c.from_builtin_tensor %617 : tensor<1x12x128x64xf32> -> !torch.vtensor<[1,12,128,64],f32>
    %619 = torch.aten.transpose.int %618, %int1, %int2 : !torch.vtensor<[1,12,128,64],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,128,12,64],f32>
    %620 = torch.prim.ListConstruct %int1, %int128, %int-1 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %621 = torch.aten.view %619, %620 : !torch.vtensor<[1,128,12,64],f32>, !torch.list<int> -> !torch.vtensor<[1,128,768],f32>
    %622 = torch.prim.ListConstruct %int-1, %int768 : (!torch.int, !torch.int) -> !torch.list<int>
    %623 = torch.aten.view %621, %622 : !torch.vtensor<[1,128,768],f32>, !torch.list<int> -> !torch.vtensor<[128,768],f32>
    %624 = torch_c.to_builtin_tensor %623 : !torch.vtensor<[128,768],f32> -> tensor<128x768xf32>
    %625 = dnn.mm %624, %cst_2 : tensor<128x768xf32>, tensor<768x768xf32> -> tensor<128x768xf32>
    %626 = torch_c.from_builtin_tensor %625 : tensor<128x768xf32> -> !torch.vtensor<[128,768],f32>
    %627 = torch.aten.mul.Scalar %626, %int1 : !torch.vtensor<[128,768],f32>, !torch.int -> !torch.vtensor<[128,768],f32>
    %628 = torch_c.to_builtin_tensor %627 : !torch.vtensor<[128,768],f32> -> tensor<128x768xf32>
    %629 = dnn.add %628, %cst_5 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<128x768xf32>, tensor<768xf32>) -> tensor<128x768xf32>
    %630 = torch_c.from_builtin_tensor %629 : tensor<128x768xf32> -> !torch.vtensor<[128,768],f32>
    %631 = torch.prim.ListConstruct %int1, %int128, %int768 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %632 = torch.aten.view %630, %631 : !torch.vtensor<[128,768],f32>, !torch.list<int> -> !torch.vtensor<[1,128,768],f32>
    %633 = torch_c.to_builtin_tensor %632 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %634 = torch_c.to_builtin_tensor %586 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %635 = dnn.add %633, %634 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<1x128x768xf32>, tensor<1x128x768xf32>) -> tensor<1x128x768xf32>
    %636 = torch_c.from_builtin_tensor %635 : tensor<1x128x768xf32> -> !torch.vtensor<[1,128,768],f32>
    %637 = torch_c.to_builtin_tensor %636 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %638 = dnn.layer_norm %637, %cst_5, %cst_5 {parameter_indices = array<i32: 1, 4, 5>, parameters = [[768], 1.000000e-05, false]} : (tensor<1x128x768xf32>, tensor<768xf32>, tensor<768xf32>) -> tensor<1x128x768xf32>
    %639 = torch_c.from_builtin_tensor %638 : tensor<1x128x768xf32> -> !torch.vtensor<[1,128,768],f32>
    %640 = torch.prim.ListConstruct %int-1, %int768 : (!torch.int, !torch.int) -> !torch.list<int>
    %641 = torch.aten.view %639, %640 : !torch.vtensor<[1,128,768],f32>, !torch.list<int> -> !torch.vtensor<[128,768],f32>
    %642 = torch_c.to_builtin_tensor %641 : !torch.vtensor<[128,768],f32> -> tensor<128x768xf32>
    %643 = dnn.mm %642, %cst_1 : tensor<128x768xf32>, tensor<768x3072xf32> -> tensor<128x3072xf32>
    %644 = torch_c.from_builtin_tensor %643 : tensor<128x3072xf32> -> !torch.vtensor<[128,3072],f32>
    %645 = torch.aten.mul.Scalar %644, %int1 : !torch.vtensor<[128,3072],f32>, !torch.int -> !torch.vtensor<[128,3072],f32>
    %646 = torch_c.to_builtin_tensor %645 : !torch.vtensor<[128,3072],f32> -> tensor<128x3072xf32>
    %647 = dnn.add %646, %cst_0 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<128x3072xf32>, tensor<3072xf32>) -> tensor<128x3072xf32>
    %648 = torch_c.from_builtin_tensor %647 : tensor<128x3072xf32> -> !torch.vtensor<[128,3072],f32>
    %649 = torch.prim.ListConstruct %int1, %int128, %int3072 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %650 = torch.aten.view %648, %649 : !torch.vtensor<[128,3072],f32>, !torch.list<int> -> !torch.vtensor<[1,128,3072],f32>
    %651 = torch_c.to_builtin_tensor %650 : !torch.vtensor<[1,128,3072],f32> -> tensor<1x128x3072xf32>
    %652 = dnn.gelu %651 {parameter_indices = array<i32: 1>, parameters = ["tanh"]} : (tensor<1x128x3072xf32>) -> tensor<1x128x3072xf32>
    %653 = torch_c.from_builtin_tensor %652 : tensor<1x128x3072xf32> -> !torch.vtensor<[1,128,3072],f32>
    %654 = torch.prim.ListConstruct %int-1, %int3072 : (!torch.int, !torch.int) -> !torch.list<int>
    %655 = torch.aten.view %653, %654 : !torch.vtensor<[1,128,3072],f32>, !torch.list<int> -> !torch.vtensor<[128,3072],f32>
    %656 = torch_c.to_builtin_tensor %655 : !torch.vtensor<[128,3072],f32> -> tensor<128x3072xf32>
    %657 = dnn.mm %656, %cst : tensor<128x3072xf32>, tensor<3072x768xf32> -> tensor<128x768xf32>
    %658 = torch_c.from_builtin_tensor %657 : tensor<128x768xf32> -> !torch.vtensor<[128,768],f32>
    %659 = torch.aten.mul.Scalar %658, %int1 : !torch.vtensor<[128,768],f32>, !torch.int -> !torch.vtensor<[128,768],f32>
    %660 = torch_c.to_builtin_tensor %659 : !torch.vtensor<[128,768],f32> -> tensor<128x768xf32>
    %661 = dnn.add %660, %cst_5 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<128x768xf32>, tensor<768xf32>) -> tensor<128x768xf32>
    %662 = torch_c.from_builtin_tensor %661 : tensor<128x768xf32> -> !torch.vtensor<[128,768],f32>
    %663 = torch.prim.ListConstruct %int1, %int128, %int768 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %664 = torch.aten.view %662, %663 : !torch.vtensor<[128,768],f32>, !torch.list<int> -> !torch.vtensor<[1,128,768],f32>
    %665 = torch_c.to_builtin_tensor %636 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %666 = torch_c.to_builtin_tensor %664 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %667 = dnn.add %665, %666 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<1x128x768xf32>, tensor<1x128x768xf32>) -> tensor<1x128x768xf32>
    %668 = torch_c.from_builtin_tensor %667 : tensor<1x128x768xf32> -> !torch.vtensor<[1,128,768],f32>
    %669 = torch_c.to_builtin_tensor %668 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %670 = dnn.layer_norm %669, %cst_5, %cst_5 {parameter_indices = array<i32: 1, 4, 5>, parameters = [[768], 1.000000e-05, false]} : (tensor<1x128x768xf32>, tensor<768xf32>, tensor<768xf32>) -> tensor<1x128x768xf32>
    %671 = torch_c.from_builtin_tensor %670 : tensor<1x128x768xf32> -> !torch.vtensor<[1,128,768],f32>
    %672 = torch.prim.ListConstruct %int-1, %int768 : (!torch.int, !torch.int) -> !torch.list<int>
    %673 = torch.aten.view %671, %672 : !torch.vtensor<[1,128,768],f32>, !torch.list<int> -> !torch.vtensor<[128,768],f32>
    %674 = torch_c.to_builtin_tensor %673 : !torch.vtensor<[128,768],f32> -> tensor<128x768xf32>
    %675 = dnn.mm %674, %cst_4 : tensor<128x768xf32>, tensor<768x2304xf32> -> tensor<128x2304xf32>
    %676 = torch_c.from_builtin_tensor %675 : tensor<128x2304xf32> -> !torch.vtensor<[128,2304],f32>
    %677 = torch.aten.mul.Scalar %676, %int1 : !torch.vtensor<[128,2304],f32>, !torch.int -> !torch.vtensor<[128,2304],f32>
    %678 = torch_c.to_builtin_tensor %677 : !torch.vtensor<[128,2304],f32> -> tensor<128x2304xf32>
    %679 = dnn.add %678, %cst_3 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<128x2304xf32>, tensor<2304xf32>) -> tensor<128x2304xf32>
    %680 = torch_c.from_builtin_tensor %679 : tensor<128x2304xf32> -> !torch.vtensor<[128,2304],f32>
    %681 = torch.prim.ListConstruct %int1, %int128, %int2304 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %682 = torch.aten.view %680, %681 : !torch.vtensor<[128,2304],f32>, !torch.list<int> -> !torch.vtensor<[1,128,2304],f32>
    %683 = torch.aten.slice.Tensor %682, %int2, %int0, %int768, %int1 : !torch.vtensor<[1,128,2304],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,128,768],f32>
    %684 = torch.aten.slice.Tensor %682, %int2, %int768, %int1536, %int1 : !torch.vtensor<[1,128,2304],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,128,768],f32>
    %685 = torch.aten.slice.Tensor %682, %int2, %int1536, %int2304, %int1 : !torch.vtensor<[1,128,2304],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,128,768],f32>
    %686 = torch.prim.ListConstruct %int1, %int128, %int-1, %int64 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %687 = torch.aten.view %684, %686 : !torch.vtensor<[1,128,768],f32>, !torch.list<int> -> !torch.vtensor<[1,128,12,64],f32>
    %688 = torch.aten.transpose.int %687, %int1, %int2 : !torch.vtensor<[1,128,12,64],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,12,128,64],f32>
    %689 = torch.prim.ListConstruct %int1, %int128, %int-1, %int64 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %690 = torch.aten.view %685, %689 : !torch.vtensor<[1,128,768],f32>, !torch.list<int> -> !torch.vtensor<[1,128,12,64],f32>
    %691 = torch.aten.transpose.int %690, %int1, %int2 : !torch.vtensor<[1,128,12,64],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,12,128,64],f32>
    %692 = torch.prim.ListConstruct %int1, %int128, %int-1, %int64 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %693 = torch.aten.view %683, %692 : !torch.vtensor<[1,128,768],f32>, !torch.list<int> -> !torch.vtensor<[1,128,12,64],f32>
    %694 = torch.aten.transpose.int %693, %int1, %int2 : !torch.vtensor<[1,128,12,64],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,12,128,64],f32>
    %695 = torch_c.to_builtin_tensor %694 : !torch.vtensor<[1,12,128,64],f32> -> tensor<1x12x128x64xf32>
    %696 = torch_c.to_builtin_tensor %688 : !torch.vtensor<[1,12,128,64],f32> -> tensor<1x12x128x64xf32>
    %697 = torch_c.to_builtin_tensor %691 : !torch.vtensor<[1,12,128,64],f32> -> tensor<1x12x128x64xf32>
    %698 = torch_c.to_builtin_tensor %arg2 : !torch.vtensor<[1,1,128,128],i1> -> tensor<1x1x128x128xi1>
    %699 = dnn.scaled_dot_product_attention %695, %696, %697, %698 {parameter_indices = array<i32: 4, 5, 6, 7>, parameters = [0.000000e+00, false, 1.250000e-01, false]} : (tensor<1x12x128x64xf32>, tensor<1x12x128x64xf32>, tensor<1x12x128x64xf32>, tensor<1x1x128x128xi1>) -> tensor<1x12x128x64xf32>
    %700 = torch_c.from_builtin_tensor %699 : tensor<1x12x128x64xf32> -> !torch.vtensor<[1,12,128,64],f32>
    %701 = torch.aten.transpose.int %700, %int1, %int2 : !torch.vtensor<[1,12,128,64],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,128,12,64],f32>
    %702 = torch.prim.ListConstruct %int1, %int128, %int-1 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %703 = torch.aten.view %701, %702 : !torch.vtensor<[1,128,12,64],f32>, !torch.list<int> -> !torch.vtensor<[1,128,768],f32>
    %704 = torch.prim.ListConstruct %int-1, %int768 : (!torch.int, !torch.int) -> !torch.list<int>
    %705 = torch.aten.view %703, %704 : !torch.vtensor<[1,128,768],f32>, !torch.list<int> -> !torch.vtensor<[128,768],f32>
    %706 = torch_c.to_builtin_tensor %705 : !torch.vtensor<[128,768],f32> -> tensor<128x768xf32>
    %707 = dnn.mm %706, %cst_2 : tensor<128x768xf32>, tensor<768x768xf32> -> tensor<128x768xf32>
    %708 = torch_c.from_builtin_tensor %707 : tensor<128x768xf32> -> !torch.vtensor<[128,768],f32>
    %709 = torch.aten.mul.Scalar %708, %int1 : !torch.vtensor<[128,768],f32>, !torch.int -> !torch.vtensor<[128,768],f32>
    %710 = torch_c.to_builtin_tensor %709 : !torch.vtensor<[128,768],f32> -> tensor<128x768xf32>
    %711 = dnn.add %710, %cst_5 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<128x768xf32>, tensor<768xf32>) -> tensor<128x768xf32>
    %712 = torch_c.from_builtin_tensor %711 : tensor<128x768xf32> -> !torch.vtensor<[128,768],f32>
    %713 = torch.prim.ListConstruct %int1, %int128, %int768 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %714 = torch.aten.view %712, %713 : !torch.vtensor<[128,768],f32>, !torch.list<int> -> !torch.vtensor<[1,128,768],f32>
    %715 = torch_c.to_builtin_tensor %714 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %716 = torch_c.to_builtin_tensor %668 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %717 = dnn.add %715, %716 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<1x128x768xf32>, tensor<1x128x768xf32>) -> tensor<1x128x768xf32>
    %718 = torch_c.from_builtin_tensor %717 : tensor<1x128x768xf32> -> !torch.vtensor<[1,128,768],f32>
    %719 = torch_c.to_builtin_tensor %718 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %720 = dnn.layer_norm %719, %cst_5, %cst_5 {parameter_indices = array<i32: 1, 4, 5>, parameters = [[768], 1.000000e-05, false]} : (tensor<1x128x768xf32>, tensor<768xf32>, tensor<768xf32>) -> tensor<1x128x768xf32>
    %721 = torch_c.from_builtin_tensor %720 : tensor<1x128x768xf32> -> !torch.vtensor<[1,128,768],f32>
    %722 = torch.prim.ListConstruct %int-1, %int768 : (!torch.int, !torch.int) -> !torch.list<int>
    %723 = torch.aten.view %721, %722 : !torch.vtensor<[1,128,768],f32>, !torch.list<int> -> !torch.vtensor<[128,768],f32>
    %724 = torch_c.to_builtin_tensor %723 : !torch.vtensor<[128,768],f32> -> tensor<128x768xf32>
    %725 = dnn.mm %724, %cst_1 : tensor<128x768xf32>, tensor<768x3072xf32> -> tensor<128x3072xf32>
    %726 = torch_c.from_builtin_tensor %725 : tensor<128x3072xf32> -> !torch.vtensor<[128,3072],f32>
    %727 = torch.aten.mul.Scalar %726, %int1 : !torch.vtensor<[128,3072],f32>, !torch.int -> !torch.vtensor<[128,3072],f32>
    %728 = torch_c.to_builtin_tensor %727 : !torch.vtensor<[128,3072],f32> -> tensor<128x3072xf32>
    %729 = dnn.add %728, %cst_0 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<128x3072xf32>, tensor<3072xf32>) -> tensor<128x3072xf32>
    %730 = torch_c.from_builtin_tensor %729 : tensor<128x3072xf32> -> !torch.vtensor<[128,3072],f32>
    %731 = torch.prim.ListConstruct %int1, %int128, %int3072 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %732 = torch.aten.view %730, %731 : !torch.vtensor<[128,3072],f32>, !torch.list<int> -> !torch.vtensor<[1,128,3072],f32>
    %733 = torch_c.to_builtin_tensor %732 : !torch.vtensor<[1,128,3072],f32> -> tensor<1x128x3072xf32>
    %734 = dnn.gelu %733 {parameter_indices = array<i32: 1>, parameters = ["tanh"]} : (tensor<1x128x3072xf32>) -> tensor<1x128x3072xf32>
    %735 = torch_c.from_builtin_tensor %734 : tensor<1x128x3072xf32> -> !torch.vtensor<[1,128,3072],f32>
    %736 = torch.prim.ListConstruct %int-1, %int3072 : (!torch.int, !torch.int) -> !torch.list<int>
    %737 = torch.aten.view %735, %736 : !torch.vtensor<[1,128,3072],f32>, !torch.list<int> -> !torch.vtensor<[128,3072],f32>
    %738 = torch_c.to_builtin_tensor %737 : !torch.vtensor<[128,3072],f32> -> tensor<128x3072xf32>
    %739 = dnn.mm %738, %cst : tensor<128x3072xf32>, tensor<3072x768xf32> -> tensor<128x768xf32>
    %740 = torch_c.from_builtin_tensor %739 : tensor<128x768xf32> -> !torch.vtensor<[128,768],f32>
    %741 = torch.aten.mul.Scalar %740, %int1 : !torch.vtensor<[128,768],f32>, !torch.int -> !torch.vtensor<[128,768],f32>
    %742 = torch_c.to_builtin_tensor %741 : !torch.vtensor<[128,768],f32> -> tensor<128x768xf32>
    %743 = dnn.add %742, %cst_5 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<128x768xf32>, tensor<768xf32>) -> tensor<128x768xf32>
    %744 = torch_c.from_builtin_tensor %743 : tensor<128x768xf32> -> !torch.vtensor<[128,768],f32>
    %745 = torch.prim.ListConstruct %int1, %int128, %int768 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %746 = torch.aten.view %744, %745 : !torch.vtensor<[128,768],f32>, !torch.list<int> -> !torch.vtensor<[1,128,768],f32>
    %747 = torch_c.to_builtin_tensor %718 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %748 = torch_c.to_builtin_tensor %746 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %749 = dnn.add %747, %748 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<1x128x768xf32>, tensor<1x128x768xf32>) -> tensor<1x128x768xf32>
    %750 = torch_c.from_builtin_tensor %749 : tensor<1x128x768xf32> -> !torch.vtensor<[1,128,768],f32>
    %751 = torch_c.to_builtin_tensor %750 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %752 = dnn.layer_norm %751, %cst_5, %cst_5 {parameter_indices = array<i32: 1, 4, 5>, parameters = [[768], 1.000000e-05, false]} : (tensor<1x128x768xf32>, tensor<768xf32>, tensor<768xf32>) -> tensor<1x128x768xf32>
    %753 = torch_c.from_builtin_tensor %752 : tensor<1x128x768xf32> -> !torch.vtensor<[1,128,768],f32>
    %754 = torch.prim.ListConstruct %int-1, %int768 : (!torch.int, !torch.int) -> !torch.list<int>
    %755 = torch.aten.view %753, %754 : !torch.vtensor<[1,128,768],f32>, !torch.list<int> -> !torch.vtensor<[128,768],f32>
    %756 = torch_c.to_builtin_tensor %755 : !torch.vtensor<[128,768],f32> -> tensor<128x768xf32>
    %757 = dnn.mm %756, %cst_4 : tensor<128x768xf32>, tensor<768x2304xf32> -> tensor<128x2304xf32>
    %758 = torch_c.from_builtin_tensor %757 : tensor<128x2304xf32> -> !torch.vtensor<[128,2304],f32>
    %759 = torch.aten.mul.Scalar %758, %int1 : !torch.vtensor<[128,2304],f32>, !torch.int -> !torch.vtensor<[128,2304],f32>
    %760 = torch_c.to_builtin_tensor %759 : !torch.vtensor<[128,2304],f32> -> tensor<128x2304xf32>
    %761 = dnn.add %760, %cst_3 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<128x2304xf32>, tensor<2304xf32>) -> tensor<128x2304xf32>
    %762 = torch_c.from_builtin_tensor %761 : tensor<128x2304xf32> -> !torch.vtensor<[128,2304],f32>
    %763 = torch.prim.ListConstruct %int1, %int128, %int2304 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %764 = torch.aten.view %762, %763 : !torch.vtensor<[128,2304],f32>, !torch.list<int> -> !torch.vtensor<[1,128,2304],f32>
    %765 = torch.aten.slice.Tensor %764, %int2, %int0, %int768, %int1 : !torch.vtensor<[1,128,2304],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,128,768],f32>
    %766 = torch.aten.slice.Tensor %764, %int2, %int768, %int1536, %int1 : !torch.vtensor<[1,128,2304],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,128,768],f32>
    %767 = torch.aten.slice.Tensor %764, %int2, %int1536, %int2304, %int1 : !torch.vtensor<[1,128,2304],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,128,768],f32>
    %768 = torch.prim.ListConstruct %int1, %int128, %int-1, %int64 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %769 = torch.aten.view %766, %768 : !torch.vtensor<[1,128,768],f32>, !torch.list<int> -> !torch.vtensor<[1,128,12,64],f32>
    %770 = torch.aten.transpose.int %769, %int1, %int2 : !torch.vtensor<[1,128,12,64],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,12,128,64],f32>
    %771 = torch.prim.ListConstruct %int1, %int128, %int-1, %int64 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %772 = torch.aten.view %767, %771 : !torch.vtensor<[1,128,768],f32>, !torch.list<int> -> !torch.vtensor<[1,128,12,64],f32>
    %773 = torch.aten.transpose.int %772, %int1, %int2 : !torch.vtensor<[1,128,12,64],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,12,128,64],f32>
    %774 = torch.prim.ListConstruct %int1, %int128, %int-1, %int64 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %775 = torch.aten.view %765, %774 : !torch.vtensor<[1,128,768],f32>, !torch.list<int> -> !torch.vtensor<[1,128,12,64],f32>
    %776 = torch.aten.transpose.int %775, %int1, %int2 : !torch.vtensor<[1,128,12,64],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,12,128,64],f32>
    %777 = torch_c.to_builtin_tensor %776 : !torch.vtensor<[1,12,128,64],f32> -> tensor<1x12x128x64xf32>
    %778 = torch_c.to_builtin_tensor %770 : !torch.vtensor<[1,12,128,64],f32> -> tensor<1x12x128x64xf32>
    %779 = torch_c.to_builtin_tensor %773 : !torch.vtensor<[1,12,128,64],f32> -> tensor<1x12x128x64xf32>
    %780 = torch_c.to_builtin_tensor %arg2 : !torch.vtensor<[1,1,128,128],i1> -> tensor<1x1x128x128xi1>
    %781 = dnn.scaled_dot_product_attention %777, %778, %779, %780 {parameter_indices = array<i32: 4, 5, 6, 7>, parameters = [0.000000e+00, false, 1.250000e-01, false]} : (tensor<1x12x128x64xf32>, tensor<1x12x128x64xf32>, tensor<1x12x128x64xf32>, tensor<1x1x128x128xi1>) -> tensor<1x12x128x64xf32>
    %782 = torch_c.from_builtin_tensor %781 : tensor<1x12x128x64xf32> -> !torch.vtensor<[1,12,128,64],f32>
    %783 = torch.aten.transpose.int %782, %int1, %int2 : !torch.vtensor<[1,12,128,64],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,128,12,64],f32>
    %784 = torch.prim.ListConstruct %int1, %int128, %int-1 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %785 = torch.aten.view %783, %784 : !torch.vtensor<[1,128,12,64],f32>, !torch.list<int> -> !torch.vtensor<[1,128,768],f32>
    %786 = torch.prim.ListConstruct %int-1, %int768 : (!torch.int, !torch.int) -> !torch.list<int>
    %787 = torch.aten.view %785, %786 : !torch.vtensor<[1,128,768],f32>, !torch.list<int> -> !torch.vtensor<[128,768],f32>
    %788 = torch_c.to_builtin_tensor %787 : !torch.vtensor<[128,768],f32> -> tensor<128x768xf32>
    %789 = dnn.mm %788, %cst_2 : tensor<128x768xf32>, tensor<768x768xf32> -> tensor<128x768xf32>
    %790 = torch_c.from_builtin_tensor %789 : tensor<128x768xf32> -> !torch.vtensor<[128,768],f32>
    %791 = torch.aten.mul.Scalar %790, %int1 : !torch.vtensor<[128,768],f32>, !torch.int -> !torch.vtensor<[128,768],f32>
    %792 = torch_c.to_builtin_tensor %791 : !torch.vtensor<[128,768],f32> -> tensor<128x768xf32>
    %793 = dnn.add %792, %cst_5 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<128x768xf32>, tensor<768xf32>) -> tensor<128x768xf32>
    %794 = torch_c.from_builtin_tensor %793 : tensor<128x768xf32> -> !torch.vtensor<[128,768],f32>
    %795 = torch.prim.ListConstruct %int1, %int128, %int768 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %796 = torch.aten.view %794, %795 : !torch.vtensor<[128,768],f32>, !torch.list<int> -> !torch.vtensor<[1,128,768],f32>
    %797 = torch_c.to_builtin_tensor %796 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %798 = torch_c.to_builtin_tensor %750 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %799 = dnn.add %797, %798 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<1x128x768xf32>, tensor<1x128x768xf32>) -> tensor<1x128x768xf32>
    %800 = torch_c.from_builtin_tensor %799 : tensor<1x128x768xf32> -> !torch.vtensor<[1,128,768],f32>
    %801 = torch_c.to_builtin_tensor %800 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %802 = dnn.layer_norm %801, %cst_5, %cst_5 {parameter_indices = array<i32: 1, 4, 5>, parameters = [[768], 1.000000e-05, false]} : (tensor<1x128x768xf32>, tensor<768xf32>, tensor<768xf32>) -> tensor<1x128x768xf32>
    %803 = torch_c.from_builtin_tensor %802 : tensor<1x128x768xf32> -> !torch.vtensor<[1,128,768],f32>
    %804 = torch.prim.ListConstruct %int-1, %int768 : (!torch.int, !torch.int) -> !torch.list<int>
    %805 = torch.aten.view %803, %804 : !torch.vtensor<[1,128,768],f32>, !torch.list<int> -> !torch.vtensor<[128,768],f32>
    %806 = torch_c.to_builtin_tensor %805 : !torch.vtensor<[128,768],f32> -> tensor<128x768xf32>
    %807 = dnn.mm %806, %cst_1 : tensor<128x768xf32>, tensor<768x3072xf32> -> tensor<128x3072xf32>
    %808 = torch_c.from_builtin_tensor %807 : tensor<128x3072xf32> -> !torch.vtensor<[128,3072],f32>
    %809 = torch.aten.mul.Scalar %808, %int1 : !torch.vtensor<[128,3072],f32>, !torch.int -> !torch.vtensor<[128,3072],f32>
    %810 = torch_c.to_builtin_tensor %809 : !torch.vtensor<[128,3072],f32> -> tensor<128x3072xf32>
    %811 = dnn.add %810, %cst_0 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<128x3072xf32>, tensor<3072xf32>) -> tensor<128x3072xf32>
    %812 = torch_c.from_builtin_tensor %811 : tensor<128x3072xf32> -> !torch.vtensor<[128,3072],f32>
    %813 = torch.prim.ListConstruct %int1, %int128, %int3072 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %814 = torch.aten.view %812, %813 : !torch.vtensor<[128,3072],f32>, !torch.list<int> -> !torch.vtensor<[1,128,3072],f32>
    %815 = torch_c.to_builtin_tensor %814 : !torch.vtensor<[1,128,3072],f32> -> tensor<1x128x3072xf32>
    %816 = dnn.gelu %815 {parameter_indices = array<i32: 1>, parameters = ["tanh"]} : (tensor<1x128x3072xf32>) -> tensor<1x128x3072xf32>
    %817 = torch_c.from_builtin_tensor %816 : tensor<1x128x3072xf32> -> !torch.vtensor<[1,128,3072],f32>
    %818 = torch.prim.ListConstruct %int-1, %int3072 : (!torch.int, !torch.int) -> !torch.list<int>
    %819 = torch.aten.view %817, %818 : !torch.vtensor<[1,128,3072],f32>, !torch.list<int> -> !torch.vtensor<[128,3072],f32>
    %820 = torch_c.to_builtin_tensor %819 : !torch.vtensor<[128,3072],f32> -> tensor<128x3072xf32>
    %821 = dnn.mm %820, %cst : tensor<128x3072xf32>, tensor<3072x768xf32> -> tensor<128x768xf32>
    %822 = torch_c.from_builtin_tensor %821 : tensor<128x768xf32> -> !torch.vtensor<[128,768],f32>
    %823 = torch.aten.mul.Scalar %822, %int1 : !torch.vtensor<[128,768],f32>, !torch.int -> !torch.vtensor<[128,768],f32>
    %824 = torch_c.to_builtin_tensor %823 : !torch.vtensor<[128,768],f32> -> tensor<128x768xf32>
    %825 = dnn.add %824, %cst_5 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<128x768xf32>, tensor<768xf32>) -> tensor<128x768xf32>
    %826 = torch_c.from_builtin_tensor %825 : tensor<128x768xf32> -> !torch.vtensor<[128,768],f32>
    %827 = torch.prim.ListConstruct %int1, %int128, %int768 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %828 = torch.aten.view %826, %827 : !torch.vtensor<[128,768],f32>, !torch.list<int> -> !torch.vtensor<[1,128,768],f32>
    %829 = torch_c.to_builtin_tensor %800 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %830 = torch_c.to_builtin_tensor %828 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %831 = dnn.add %829, %830 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<1x128x768xf32>, tensor<1x128x768xf32>) -> tensor<1x128x768xf32>
    %832 = torch_c.from_builtin_tensor %831 : tensor<1x128x768xf32> -> !torch.vtensor<[1,128,768],f32>
    %833 = torch_c.to_builtin_tensor %832 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %834 = dnn.layer_norm %833, %cst_5, %cst_5 {parameter_indices = array<i32: 1, 4, 5>, parameters = [[768], 1.000000e-05, false]} : (tensor<1x128x768xf32>, tensor<768xf32>, tensor<768xf32>) -> tensor<1x128x768xf32>
    %835 = torch_c.from_builtin_tensor %834 : tensor<1x128x768xf32> -> !torch.vtensor<[1,128,768],f32>
    %836 = torch.prim.ListConstruct %int-1, %int768 : (!torch.int, !torch.int) -> !torch.list<int>
    %837 = torch.aten.view %835, %836 : !torch.vtensor<[1,128,768],f32>, !torch.list<int> -> !torch.vtensor<[128,768],f32>
    %838 = torch_c.to_builtin_tensor %837 : !torch.vtensor<[128,768],f32> -> tensor<128x768xf32>
    %839 = dnn.mm %838, %cst_4 : tensor<128x768xf32>, tensor<768x2304xf32> -> tensor<128x2304xf32>
    %840 = torch_c.from_builtin_tensor %839 : tensor<128x2304xf32> -> !torch.vtensor<[128,2304],f32>
    %841 = torch.aten.mul.Scalar %840, %int1 : !torch.vtensor<[128,2304],f32>, !torch.int -> !torch.vtensor<[128,2304],f32>
    %842 = torch_c.to_builtin_tensor %841 : !torch.vtensor<[128,2304],f32> -> tensor<128x2304xf32>
    %843 = dnn.add %842, %cst_3 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<128x2304xf32>, tensor<2304xf32>) -> tensor<128x2304xf32>
    %844 = torch_c.from_builtin_tensor %843 : tensor<128x2304xf32> -> !torch.vtensor<[128,2304],f32>
    %845 = torch.prim.ListConstruct %int1, %int128, %int2304 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %846 = torch.aten.view %844, %845 : !torch.vtensor<[128,2304],f32>, !torch.list<int> -> !torch.vtensor<[1,128,2304],f32>
    %847 = torch.aten.slice.Tensor %846, %int2, %int0, %int768, %int1 : !torch.vtensor<[1,128,2304],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,128,768],f32>
    %848 = torch.aten.slice.Tensor %846, %int2, %int768, %int1536, %int1 : !torch.vtensor<[1,128,2304],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,128,768],f32>
    %849 = torch.aten.slice.Tensor %846, %int2, %int1536, %int2304, %int1 : !torch.vtensor<[1,128,2304],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,128,768],f32>
    %850 = torch.prim.ListConstruct %int1, %int128, %int-1, %int64 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %851 = torch.aten.view %848, %850 : !torch.vtensor<[1,128,768],f32>, !torch.list<int> -> !torch.vtensor<[1,128,12,64],f32>
    %852 = torch.aten.transpose.int %851, %int1, %int2 : !torch.vtensor<[1,128,12,64],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,12,128,64],f32>
    %853 = torch.prim.ListConstruct %int1, %int128, %int-1, %int64 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %854 = torch.aten.view %849, %853 : !torch.vtensor<[1,128,768],f32>, !torch.list<int> -> !torch.vtensor<[1,128,12,64],f32>
    %855 = torch.aten.transpose.int %854, %int1, %int2 : !torch.vtensor<[1,128,12,64],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,12,128,64],f32>
    %856 = torch.prim.ListConstruct %int1, %int128, %int-1, %int64 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %857 = torch.aten.view %847, %856 : !torch.vtensor<[1,128,768],f32>, !torch.list<int> -> !torch.vtensor<[1,128,12,64],f32>
    %858 = torch.aten.transpose.int %857, %int1, %int2 : !torch.vtensor<[1,128,12,64],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,12,128,64],f32>
    %859 = torch_c.to_builtin_tensor %858 : !torch.vtensor<[1,12,128,64],f32> -> tensor<1x12x128x64xf32>
    %860 = torch_c.to_builtin_tensor %852 : !torch.vtensor<[1,12,128,64],f32> -> tensor<1x12x128x64xf32>
    %861 = torch_c.to_builtin_tensor %855 : !torch.vtensor<[1,12,128,64],f32> -> tensor<1x12x128x64xf32>
    %862 = torch_c.to_builtin_tensor %arg2 : !torch.vtensor<[1,1,128,128],i1> -> tensor<1x1x128x128xi1>
    %863 = dnn.scaled_dot_product_attention %859, %860, %861, %862 {parameter_indices = array<i32: 4, 5, 6, 7>, parameters = [0.000000e+00, false, 1.250000e-01, false]} : (tensor<1x12x128x64xf32>, tensor<1x12x128x64xf32>, tensor<1x12x128x64xf32>, tensor<1x1x128x128xi1>) -> tensor<1x12x128x64xf32>
    %864 = torch_c.from_builtin_tensor %863 : tensor<1x12x128x64xf32> -> !torch.vtensor<[1,12,128,64],f32>
    %865 = torch.aten.transpose.int %864, %int1, %int2 : !torch.vtensor<[1,12,128,64],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,128,12,64],f32>
    %866 = torch.prim.ListConstruct %int1, %int128, %int-1 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %867 = torch.aten.view %865, %866 : !torch.vtensor<[1,128,12,64],f32>, !torch.list<int> -> !torch.vtensor<[1,128,768],f32>
    %868 = torch.prim.ListConstruct %int-1, %int768 : (!torch.int, !torch.int) -> !torch.list<int>
    %869 = torch.aten.view %867, %868 : !torch.vtensor<[1,128,768],f32>, !torch.list<int> -> !torch.vtensor<[128,768],f32>
    %870 = torch_c.to_builtin_tensor %869 : !torch.vtensor<[128,768],f32> -> tensor<128x768xf32>
    %871 = dnn.mm %870, %cst_2 : tensor<128x768xf32>, tensor<768x768xf32> -> tensor<128x768xf32>
    %872 = torch_c.from_builtin_tensor %871 : tensor<128x768xf32> -> !torch.vtensor<[128,768],f32>
    %873 = torch.aten.mul.Scalar %872, %int1 : !torch.vtensor<[128,768],f32>, !torch.int -> !torch.vtensor<[128,768],f32>
    %874 = torch_c.to_builtin_tensor %873 : !torch.vtensor<[128,768],f32> -> tensor<128x768xf32>
    %875 = dnn.add %874, %cst_5 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<128x768xf32>, tensor<768xf32>) -> tensor<128x768xf32>
    %876 = torch_c.from_builtin_tensor %875 : tensor<128x768xf32> -> !torch.vtensor<[128,768],f32>
    %877 = torch.prim.ListConstruct %int1, %int128, %int768 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %878 = torch.aten.view %876, %877 : !torch.vtensor<[128,768],f32>, !torch.list<int> -> !torch.vtensor<[1,128,768],f32>
    %879 = torch_c.to_builtin_tensor %878 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %880 = torch_c.to_builtin_tensor %832 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %881 = dnn.add %879, %880 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<1x128x768xf32>, tensor<1x128x768xf32>) -> tensor<1x128x768xf32>
    %882 = torch_c.from_builtin_tensor %881 : tensor<1x128x768xf32> -> !torch.vtensor<[1,128,768],f32>
    %883 = torch_c.to_builtin_tensor %882 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %884 = dnn.layer_norm %883, %cst_5, %cst_5 {parameter_indices = array<i32: 1, 4, 5>, parameters = [[768], 1.000000e-05, false]} : (tensor<1x128x768xf32>, tensor<768xf32>, tensor<768xf32>) -> tensor<1x128x768xf32>
    %885 = torch_c.from_builtin_tensor %884 : tensor<1x128x768xf32> -> !torch.vtensor<[1,128,768],f32>
    %886 = torch.prim.ListConstruct %int-1, %int768 : (!torch.int, !torch.int) -> !torch.list<int>
    %887 = torch.aten.view %885, %886 : !torch.vtensor<[1,128,768],f32>, !torch.list<int> -> !torch.vtensor<[128,768],f32>
    %888 = torch_c.to_builtin_tensor %887 : !torch.vtensor<[128,768],f32> -> tensor<128x768xf32>
    %889 = dnn.mm %888, %cst_1 : tensor<128x768xf32>, tensor<768x3072xf32> -> tensor<128x3072xf32>
    %890 = torch_c.from_builtin_tensor %889 : tensor<128x3072xf32> -> !torch.vtensor<[128,3072],f32>
    %891 = torch.aten.mul.Scalar %890, %int1 : !torch.vtensor<[128,3072],f32>, !torch.int -> !torch.vtensor<[128,3072],f32>
    %892 = torch_c.to_builtin_tensor %891 : !torch.vtensor<[128,3072],f32> -> tensor<128x3072xf32>
    %893 = dnn.add %892, %cst_0 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<128x3072xf32>, tensor<3072xf32>) -> tensor<128x3072xf32>
    %894 = torch_c.from_builtin_tensor %893 : tensor<128x3072xf32> -> !torch.vtensor<[128,3072],f32>
    %895 = torch.prim.ListConstruct %int1, %int128, %int3072 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %896 = torch.aten.view %894, %895 : !torch.vtensor<[128,3072],f32>, !torch.list<int> -> !torch.vtensor<[1,128,3072],f32>
    %897 = torch_c.to_builtin_tensor %896 : !torch.vtensor<[1,128,3072],f32> -> tensor<1x128x3072xf32>
    %898 = dnn.gelu %897 {parameter_indices = array<i32: 1>, parameters = ["tanh"]} : (tensor<1x128x3072xf32>) -> tensor<1x128x3072xf32>
    %899 = torch_c.from_builtin_tensor %898 : tensor<1x128x3072xf32> -> !torch.vtensor<[1,128,3072],f32>
    %900 = torch.prim.ListConstruct %int-1, %int3072 : (!torch.int, !torch.int) -> !torch.list<int>
    %901 = torch.aten.view %899, %900 : !torch.vtensor<[1,128,3072],f32>, !torch.list<int> -> !torch.vtensor<[128,3072],f32>
    %902 = torch_c.to_builtin_tensor %901 : !torch.vtensor<[128,3072],f32> -> tensor<128x3072xf32>
    %903 = dnn.mm %902, %cst : tensor<128x3072xf32>, tensor<3072x768xf32> -> tensor<128x768xf32>
    %904 = torch_c.from_builtin_tensor %903 : tensor<128x768xf32> -> !torch.vtensor<[128,768],f32>
    %905 = torch.aten.mul.Scalar %904, %int1 : !torch.vtensor<[128,768],f32>, !torch.int -> !torch.vtensor<[128,768],f32>
    %906 = torch_c.to_builtin_tensor %905 : !torch.vtensor<[128,768],f32> -> tensor<128x768xf32>
    %907 = dnn.add %906, %cst_5 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<128x768xf32>, tensor<768xf32>) -> tensor<128x768xf32>
    %908 = torch_c.from_builtin_tensor %907 : tensor<128x768xf32> -> !torch.vtensor<[128,768],f32>
    %909 = torch.prim.ListConstruct %int1, %int128, %int768 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %910 = torch.aten.view %908, %909 : !torch.vtensor<[128,768],f32>, !torch.list<int> -> !torch.vtensor<[1,128,768],f32>
    %911 = torch_c.to_builtin_tensor %882 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %912 = torch_c.to_builtin_tensor %910 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %913 = dnn.add %911, %912 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<1x128x768xf32>, tensor<1x128x768xf32>) -> tensor<1x128x768xf32>
    %914 = torch_c.from_builtin_tensor %913 : tensor<1x128x768xf32> -> !torch.vtensor<[1,128,768],f32>
    %915 = torch_c.to_builtin_tensor %914 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %916 = dnn.layer_norm %915, %cst_5, %cst_5 {parameter_indices = array<i32: 1, 4, 5>, parameters = [[768], 1.000000e-05, false]} : (tensor<1x128x768xf32>, tensor<768xf32>, tensor<768xf32>) -> tensor<1x128x768xf32>
    %917 = torch_c.from_builtin_tensor %916 : tensor<1x128x768xf32> -> !torch.vtensor<[1,128,768],f32>
    %918 = torch.prim.ListConstruct %int-1, %int768 : (!torch.int, !torch.int) -> !torch.list<int>
    %919 = torch.aten.view %917, %918 : !torch.vtensor<[1,128,768],f32>, !torch.list<int> -> !torch.vtensor<[128,768],f32>
    %920 = torch_c.to_builtin_tensor %919 : !torch.vtensor<[128,768],f32> -> tensor<128x768xf32>
    %921 = dnn.mm %920, %cst_4 : tensor<128x768xf32>, tensor<768x2304xf32> -> tensor<128x2304xf32>
    %922 = torch_c.from_builtin_tensor %921 : tensor<128x2304xf32> -> !torch.vtensor<[128,2304],f32>
    %923 = torch.aten.mul.Scalar %922, %int1 : !torch.vtensor<[128,2304],f32>, !torch.int -> !torch.vtensor<[128,2304],f32>
    %924 = torch_c.to_builtin_tensor %923 : !torch.vtensor<[128,2304],f32> -> tensor<128x2304xf32>
    %925 = dnn.add %924, %cst_3 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<128x2304xf32>, tensor<2304xf32>) -> tensor<128x2304xf32>
    %926 = torch_c.from_builtin_tensor %925 : tensor<128x2304xf32> -> !torch.vtensor<[128,2304],f32>
    %927 = torch.prim.ListConstruct %int1, %int128, %int2304 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %928 = torch.aten.view %926, %927 : !torch.vtensor<[128,2304],f32>, !torch.list<int> -> !torch.vtensor<[1,128,2304],f32>
    %929 = torch.aten.slice.Tensor %928, %int2, %int0, %int768, %int1 : !torch.vtensor<[1,128,2304],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,128,768],f32>
    %930 = torch.aten.slice.Tensor %928, %int2, %int768, %int1536, %int1 : !torch.vtensor<[1,128,2304],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,128,768],f32>
    %931 = torch.aten.slice.Tensor %928, %int2, %int1536, %int2304, %int1 : !torch.vtensor<[1,128,2304],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,128,768],f32>
    %932 = torch.prim.ListConstruct %int1, %int128, %int-1, %int64 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %933 = torch.aten.view %930, %932 : !torch.vtensor<[1,128,768],f32>, !torch.list<int> -> !torch.vtensor<[1,128,12,64],f32>
    %934 = torch.aten.transpose.int %933, %int1, %int2 : !torch.vtensor<[1,128,12,64],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,12,128,64],f32>
    %935 = torch.prim.ListConstruct %int1, %int128, %int-1, %int64 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %936 = torch.aten.view %931, %935 : !torch.vtensor<[1,128,768],f32>, !torch.list<int> -> !torch.vtensor<[1,128,12,64],f32>
    %937 = torch.aten.transpose.int %936, %int1, %int2 : !torch.vtensor<[1,128,12,64],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,12,128,64],f32>
    %938 = torch.prim.ListConstruct %int1, %int128, %int-1, %int64 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %939 = torch.aten.view %929, %938 : !torch.vtensor<[1,128,768],f32>, !torch.list<int> -> !torch.vtensor<[1,128,12,64],f32>
    %940 = torch.aten.transpose.int %939, %int1, %int2 : !torch.vtensor<[1,128,12,64],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,12,128,64],f32>
    %941 = torch_c.to_builtin_tensor %940 : !torch.vtensor<[1,12,128,64],f32> -> tensor<1x12x128x64xf32>
    %942 = torch_c.to_builtin_tensor %934 : !torch.vtensor<[1,12,128,64],f32> -> tensor<1x12x128x64xf32>
    %943 = torch_c.to_builtin_tensor %937 : !torch.vtensor<[1,12,128,64],f32> -> tensor<1x12x128x64xf32>
    %944 = torch_c.to_builtin_tensor %arg2 : !torch.vtensor<[1,1,128,128],i1> -> tensor<1x1x128x128xi1>
    %945 = dnn.scaled_dot_product_attention %941, %942, %943, %944 {parameter_indices = array<i32: 4, 5, 6, 7>, parameters = [0.000000e+00, false, 1.250000e-01, false]} : (tensor<1x12x128x64xf32>, tensor<1x12x128x64xf32>, tensor<1x12x128x64xf32>, tensor<1x1x128x128xi1>) -> tensor<1x12x128x64xf32>
    %946 = torch_c.from_builtin_tensor %945 : tensor<1x12x128x64xf32> -> !torch.vtensor<[1,12,128,64],f32>
    %947 = torch.aten.transpose.int %946, %int1, %int2 : !torch.vtensor<[1,12,128,64],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,128,12,64],f32>
    %948 = torch.prim.ListConstruct %int1, %int128, %int-1 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %949 = torch.aten.view %947, %948 : !torch.vtensor<[1,128,12,64],f32>, !torch.list<int> -> !torch.vtensor<[1,128,768],f32>
    %950 = torch.prim.ListConstruct %int-1, %int768 : (!torch.int, !torch.int) -> !torch.list<int>
    %951 = torch.aten.view %949, %950 : !torch.vtensor<[1,128,768],f32>, !torch.list<int> -> !torch.vtensor<[128,768],f32>
    %952 = torch_c.to_builtin_tensor %951 : !torch.vtensor<[128,768],f32> -> tensor<128x768xf32>
    %953 = dnn.mm %952, %cst_2 : tensor<128x768xf32>, tensor<768x768xf32> -> tensor<128x768xf32>
    %954 = torch_c.from_builtin_tensor %953 : tensor<128x768xf32> -> !torch.vtensor<[128,768],f32>
    %955 = torch.aten.mul.Scalar %954, %int1 : !torch.vtensor<[128,768],f32>, !torch.int -> !torch.vtensor<[128,768],f32>
    %956 = torch_c.to_builtin_tensor %955 : !torch.vtensor<[128,768],f32> -> tensor<128x768xf32>
    %957 = dnn.add %956, %cst_5 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<128x768xf32>, tensor<768xf32>) -> tensor<128x768xf32>
    %958 = torch_c.from_builtin_tensor %957 : tensor<128x768xf32> -> !torch.vtensor<[128,768],f32>
    %959 = torch.prim.ListConstruct %int1, %int128, %int768 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %960 = torch.aten.view %958, %959 : !torch.vtensor<[128,768],f32>, !torch.list<int> -> !torch.vtensor<[1,128,768],f32>
    %961 = torch_c.to_builtin_tensor %960 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %962 = torch_c.to_builtin_tensor %914 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %963 = dnn.add %961, %962 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<1x128x768xf32>, tensor<1x128x768xf32>) -> tensor<1x128x768xf32>
    %964 = torch_c.from_builtin_tensor %963 : tensor<1x128x768xf32> -> !torch.vtensor<[1,128,768],f32>
    %965 = torch_c.to_builtin_tensor %964 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %966 = dnn.layer_norm %965, %cst_5, %cst_5 {parameter_indices = array<i32: 1, 4, 5>, parameters = [[768], 1.000000e-05, false]} : (tensor<1x128x768xf32>, tensor<768xf32>, tensor<768xf32>) -> tensor<1x128x768xf32>
    %967 = torch_c.from_builtin_tensor %966 : tensor<1x128x768xf32> -> !torch.vtensor<[1,128,768],f32>
    %968 = torch.prim.ListConstruct %int-1, %int768 : (!torch.int, !torch.int) -> !torch.list<int>
    %969 = torch.aten.view %967, %968 : !torch.vtensor<[1,128,768],f32>, !torch.list<int> -> !torch.vtensor<[128,768],f32>
    %970 = torch_c.to_builtin_tensor %969 : !torch.vtensor<[128,768],f32> -> tensor<128x768xf32>
    %971 = dnn.mm %970, %cst_1 : tensor<128x768xf32>, tensor<768x3072xf32> -> tensor<128x3072xf32>
    %972 = torch_c.from_builtin_tensor %971 : tensor<128x3072xf32> -> !torch.vtensor<[128,3072],f32>
    %973 = torch.aten.mul.Scalar %972, %int1 : !torch.vtensor<[128,3072],f32>, !torch.int -> !torch.vtensor<[128,3072],f32>
    %974 = torch_c.to_builtin_tensor %973 : !torch.vtensor<[128,3072],f32> -> tensor<128x3072xf32>
    %975 = dnn.add %974, %cst_0 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<128x3072xf32>, tensor<3072xf32>) -> tensor<128x3072xf32>
    %976 = torch_c.from_builtin_tensor %975 : tensor<128x3072xf32> -> !torch.vtensor<[128,3072],f32>
    %977 = torch.prim.ListConstruct %int1, %int128, %int3072 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %978 = torch.aten.view %976, %977 : !torch.vtensor<[128,3072],f32>, !torch.list<int> -> !torch.vtensor<[1,128,3072],f32>
    %979 = torch_c.to_builtin_tensor %978 : !torch.vtensor<[1,128,3072],f32> -> tensor<1x128x3072xf32>
    %980 = dnn.gelu %979 {parameter_indices = array<i32: 1>, parameters = ["tanh"]} : (tensor<1x128x3072xf32>) -> tensor<1x128x3072xf32>
    %981 = torch_c.from_builtin_tensor %980 : tensor<1x128x3072xf32> -> !torch.vtensor<[1,128,3072],f32>
    %982 = torch.prim.ListConstruct %int-1, %int3072 : (!torch.int, !torch.int) -> !torch.list<int>
    %983 = torch.aten.view %981, %982 : !torch.vtensor<[1,128,3072],f32>, !torch.list<int> -> !torch.vtensor<[128,3072],f32>
    %984 = torch_c.to_builtin_tensor %983 : !torch.vtensor<[128,3072],f32> -> tensor<128x3072xf32>
    %985 = dnn.mm %984, %cst : tensor<128x3072xf32>, tensor<3072x768xf32> -> tensor<128x768xf32>
    %986 = torch_c.from_builtin_tensor %985 : tensor<128x768xf32> -> !torch.vtensor<[128,768],f32>
    %987 = torch.aten.mul.Scalar %986, %int1 : !torch.vtensor<[128,768],f32>, !torch.int -> !torch.vtensor<[128,768],f32>
    %988 = torch_c.to_builtin_tensor %987 : !torch.vtensor<[128,768],f32> -> tensor<128x768xf32>
    %989 = dnn.add %988, %cst_5 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<128x768xf32>, tensor<768xf32>) -> tensor<128x768xf32>
    %990 = torch_c.from_builtin_tensor %989 : tensor<128x768xf32> -> !torch.vtensor<[128,768],f32>
    %991 = torch.prim.ListConstruct %int1, %int128, %int768 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %992 = torch.aten.view %990, %991 : !torch.vtensor<[128,768],f32>, !torch.list<int> -> !torch.vtensor<[1,128,768],f32>
    %993 = torch_c.to_builtin_tensor %964 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %994 = torch_c.to_builtin_tensor %992 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %995 = dnn.add %993, %994 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<1x128x768xf32>, tensor<1x128x768xf32>) -> tensor<1x128x768xf32>
    %996 = torch_c.from_builtin_tensor %995 : tensor<1x128x768xf32> -> !torch.vtensor<[1,128,768],f32>
    %997 = torch_c.to_builtin_tensor %996 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %998 = dnn.layer_norm %997, %cst_5, %cst_5 {parameter_indices = array<i32: 1, 4, 5>, parameters = [[768], 1.000000e-05, false]} : (tensor<1x128x768xf32>, tensor<768xf32>, tensor<768xf32>) -> tensor<1x128x768xf32>
    %999 = torch_c.from_builtin_tensor %998 : tensor<1x128x768xf32> -> !torch.vtensor<[1,128,768],f32>
    %1000 = torch.prim.ListConstruct %int-1, %int128, %int768 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %1001 = torch.aten.view %999, %1000 : !torch.vtensor<[1,128,768],f32>, !torch.list<int> -> !torch.vtensor<[1,128,768],f32>
    %1002 = torch.aten.transpose.int %0, %int0, %int1 : !torch.vtensor<[50257,768],f32>, !torch.int, !torch.int -> !torch.vtensor<[768,50257],f32>
    %1003 = torch_c.to_builtin_tensor %1001 : !torch.vtensor<[1,128,768],f32> -> tensor<1x128x768xf32>
    %1004 = torch_c.to_builtin_tensor %1002 : !torch.vtensor<[768,50257],f32> -> tensor<768x50257xf32>
    %1005 = dnn.matmul %1003, %1004 : tensor<1x128x768xf32>, tensor<768x50257xf32> -> tensor<1x128x50257xf32>
    %1006 = torch_c.from_builtin_tensor %1005 : tensor<1x128x50257xf32> -> !torch.vtensor<[1,128,50257],f32>
    return %1006 : !torch.vtensor<[1,128,50257],f32>
  }
}

// RUN: dnn-mlir-opt %s | FileCheck %s
//
// Structural torchvision ViT-B/16 inference graph imported through
// Torch-MLIR's FX path without decomposition, then converted with the
// Torch-to-DNN pass. Parameter tensors use deterministic splat values.
//
// CHECK-LABEL: func.func @vit_b_16
// CHECK-DAG: dnn.convolution
// CHECK-DAG: dnn.linear
// CHECK-DAG: dnn.add
// CHECK-DAG: dnn.gelu
// CHECK-DAG: dnn.layer_norm
// CHECK-DAG: dnn.scaled_dot_product_attention
// CHECK-NOT: torch.aten.layer_norm
// CHECK-NOT: torch.aten.scaled_dot_product_attention

module {
  func.func @vit_b_16(%arg0: !torch.vtensor<[1,3,224,224],f32>) -> !torch.vtensor<[1,1000],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %cst = arith.constant dense<0.000000e+00> : tensor<1000xf32>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<1000x768xf32>
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<768x3072xf32>
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<3072xf32>
    %cst_3 = arith.constant dense<0.000000e+00> : tensor<3072x768xf32>
    %cst_4 = arith.constant dense<0.000000e+00> : tensor<768x768xf32>
    %cst_5 = arith.constant dense<0.000000e+00> : tensor<2304xf32>
    %cst_6 = arith.constant dense<0.000000e+00> : tensor<2304x768xf32>
    %cst_7 = arith.constant dense<0.000000e+00> : tensor<1x197x768xf32>
    %cst_8 = arith.constant dense<0.000000e+00> : tensor<768xf32>
    %cst_9 = arith.constant dense<0.000000e+00> : tensor<768x3x16x16xf32>
    %int64 = torch.constant.int 64
    %int12 = torch.constant.int 12
    %int197 = torch.constant.int 197
    %int-2 = torch.constant.int -2
    %int3 = torch.constant.int 3
    %float0.000000e00 = torch.constant.float 0.000000e+00
    %false = torch.constant.bool false
    %int-1 = torch.constant.int -1
    %0 = torch.vtensor.literal(dense<0.000000e+00> : tensor<1x1x768xf32>) : !torch.vtensor<[1,1,768],f32>
    %int2 = torch.constant.int 2
    %int196 = torch.constant.int 196
    %int768 = torch.constant.int 768
    %int1 = torch.constant.int 1
    %int0 = torch.constant.int 0
    %1 = torch_c.to_builtin_tensor %arg0 : !torch.vtensor<[1,3,224,224],f32> -> tensor<1x3x224x224xf32>
    %2 = dnn.convolution %1, %cst_9, %cst_8 {kind = "aten.conv2d", parameter_indices = array<i32: 3, 4, 5, 6>, parameters = [[16, 16], [0, 0], [1, 1], 1]} : (tensor<1x3x224x224xf32>, tensor<768x3x16x16xf32>, tensor<768xf32>) -> tensor<1x768x14x14xf32>
    %3 = torch_c.from_builtin_tensor %2 : tensor<1x768x14x14xf32> -> !torch.vtensor<[1,768,14,14],f32>
    %4 = torch.prim.ListConstruct %int1, %int768, %int196 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %5 = torch.aten.reshape %3, %4 : !torch.vtensor<[1,768,14,14],f32>, !torch.list<int> -> !torch.vtensor<[1,768,196],f32>
    %6 = torch.prim.ListConstruct %int0, %int2, %int1 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %7 = torch.aten.permute %5, %6 : !torch.vtensor<[1,768,196],f32>, !torch.list<int> -> !torch.vtensor<[1,196,768],f32>
    %8 = torch.prim.ListConstruct %int1, %int-1, %int-1 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %9 = torch.aten.expand %0, %8, %false : !torch.vtensor<[1,1,768],f32>, !torch.list<int>, !torch.bool -> !torch.vtensor<[1,1,768],f32>
    %10 = torch.prim.ListConstruct %9, %7 : (!torch.vtensor<[1,1,768],f32>, !torch.vtensor<[1,196,768],f32>) -> !torch.list<vtensor>
    %11 = torch.aten.cat %10, %int1 : !torch.list<vtensor>, !torch.int -> !torch.vtensor<[1,197,768],f32>
    %12 = torch_c.to_builtin_tensor %11 : !torch.vtensor<[1,197,768],f32> -> tensor<1x197x768xf32>
    %13 = dnn.add %12, %cst_7 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %14 = torch_c.from_builtin_tensor %13 : tensor<1x197x768xf32> -> !torch.vtensor<[1,197,768],f32>
    %15 = torch.aten.dropout %14, %float0.000000e00, %false : !torch.vtensor<[1,197,768],f32>, !torch.float, !torch.bool -> !torch.vtensor<[1,197,768],f32>
    %16 = torch_c.to_builtin_tensor %15 : !torch.vtensor<[1,197,768],f32> -> tensor<1x197x768xf32>
    %17 = dnn.layer_norm %16, %cst_8, %cst_8 {parameter_indices = array<i32: 1, 4, 5>, parameters = [[768], 9.9999999999999995E-7, false]} : (tensor<1x197x768xf32>, tensor<768xf32>, tensor<768xf32>) -> tensor<1x197x768xf32>
    %18 = torch_c.from_builtin_tensor %17 : tensor<1x197x768xf32> -> !torch.vtensor<[1,197,768],f32>
    %19 = torch.aten.transpose.int %18, %int1, %int0 : !torch.vtensor<[1,197,768],f32>, !torch.int, !torch.int -> !torch.vtensor<[197,1,768],f32>
    %20 = torch_c.to_builtin_tensor %19 : !torch.vtensor<[197,1,768],f32> -> tensor<197x1x768xf32>
    %21 = dnn.linear %20, %cst_6, %cst_5 : tensor<197x1x768xf32>, tensor<2304x768xf32>, tensor<2304xf32> -> tensor<197x1x2304xf32>
    %22 = torch_c.from_builtin_tensor %21 : tensor<197x1x2304xf32> -> !torch.vtensor<[197,1,2304],f32>
    %23 = torch.prim.ListConstruct %int3, %int768 : (!torch.int, !torch.int) -> !torch.list<int>
    %24 = torch.aten.unflatten.int %22, %int-1, %23 : !torch.vtensor<[197,1,2304],f32>, !torch.int, !torch.list<int> -> !torch.vtensor<[197,1,3,768],f32>
    %25 = torch.aten.unsqueeze %24, %int0 : !torch.vtensor<[197,1,3,768],f32>, !torch.int -> !torch.vtensor<[1,197,1,3,768],f32>
    %26 = torch.aten.transpose.int %25, %int0, %int-2 : !torch.vtensor<[1,197,1,3,768],f32>, !torch.int, !torch.int -> !torch.vtensor<[3,197,1,1,768],f32>
    %27 = torch.aten.squeeze.dim %26, %int-2 : !torch.vtensor<[3,197,1,1,768],f32>, !torch.int -> !torch.vtensor<[3,197,1,768],f32>
    %28 = torch.aten.contiguous %27, %int0 : !torch.vtensor<[3,197,1,768],f32>, !torch.int -> !torch.vtensor<[3,197,1,768],f32>
    %29 = torch.aten.select.int %28, %int0, %int0 : !torch.vtensor<[3,197,1,768],f32>, !torch.int, !torch.int -> !torch.vtensor<[197,1,768],f32>
    %30 = torch.aten.select.int %28, %int0, %int1 : !torch.vtensor<[3,197,1,768],f32>, !torch.int, !torch.int -> !torch.vtensor<[197,1,768],f32>
    %31 = torch.aten.select.int %28, %int0, %int2 : !torch.vtensor<[3,197,1,768],f32>, !torch.int, !torch.int -> !torch.vtensor<[197,1,768],f32>
    %32 = torch.prim.ListConstruct %int197, %int12, %int64 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %33 = torch.aten.view %29, %32 : !torch.vtensor<[197,1,768],f32>, !torch.list<int> -> !torch.vtensor<[197,12,64],f32>
    %34 = torch.aten.transpose.int %33, %int0, %int1 : !torch.vtensor<[197,12,64],f32>, !torch.int, !torch.int -> !torch.vtensor<[12,197,64],f32>
    %35 = torch.prim.ListConstruct %int197, %int12, %int64 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %36 = torch.aten.view %30, %35 : !torch.vtensor<[197,1,768],f32>, !torch.list<int> -> !torch.vtensor<[197,12,64],f32>
    %37 = torch.aten.transpose.int %36, %int0, %int1 : !torch.vtensor<[197,12,64],f32>, !torch.int, !torch.int -> !torch.vtensor<[12,197,64],f32>
    %38 = torch.prim.ListConstruct %int197, %int12, %int64 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %39 = torch.aten.view %31, %38 : !torch.vtensor<[197,1,768],f32>, !torch.list<int> -> !torch.vtensor<[197,12,64],f32>
    %40 = torch.aten.transpose.int %39, %int0, %int1 : !torch.vtensor<[197,12,64],f32>, !torch.int, !torch.int -> !torch.vtensor<[12,197,64],f32>
    %41 = torch.prim.ListConstruct %int1, %int12, %int197, %int64 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %42 = torch.aten.view %34, %41 : !torch.vtensor<[12,197,64],f32>, !torch.list<int> -> !torch.vtensor<[1,12,197,64],f32>
    %43 = torch.prim.ListConstruct %int1, %int12, %int197, %int64 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %44 = torch.aten.view %37, %43 : !torch.vtensor<[12,197,64],f32>, !torch.list<int> -> !torch.vtensor<[1,12,197,64],f32>
    %45 = torch.prim.ListConstruct %int1, %int12, %int197, %int64 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %46 = torch.aten.view %40, %45 : !torch.vtensor<[12,197,64],f32>, !torch.list<int> -> !torch.vtensor<[1,12,197,64],f32>
    %47 = torch_c.to_builtin_tensor %42 : !torch.vtensor<[1,12,197,64],f32> -> tensor<1x12x197x64xf32>
    %48 = torch_c.to_builtin_tensor %44 : !torch.vtensor<[1,12,197,64],f32> -> tensor<1x12x197x64xf32>
    %49 = torch_c.to_builtin_tensor %46 : !torch.vtensor<[1,12,197,64],f32> -> tensor<1x12x197x64xf32>
    %50 = dnn.scaled_dot_product_attention %47, %48, %49 {parameter_indices = array<i32: 3, 4, 5, 6, 7>, parameters = [unit, 0.000000e+00, false, unit, false]} : (tensor<1x12x197x64xf32>, tensor<1x12x197x64xf32>, tensor<1x12x197x64xf32>) -> tensor<1x12x197x64xf32>
    %51 = torch_c.from_builtin_tensor %50 : tensor<1x12x197x64xf32> -> !torch.vtensor<[1,12,197,64],f32>
    %52 = torch.prim.ListConstruct %int2, %int0, %int1, %int3 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %53 = torch.aten.permute %51, %52 : !torch.vtensor<[1,12,197,64],f32>, !torch.list<int> -> !torch.vtensor<[197,1,12,64],f32>
    %54 = torch.prim.ListConstruct %int197, %int768 : (!torch.int, !torch.int) -> !torch.list<int>
    %55 = torch.aten.view %53, %54 : !torch.vtensor<[197,1,12,64],f32>, !torch.list<int> -> !torch.vtensor<[197,768],f32>
    %56 = torch_c.to_builtin_tensor %55 : !torch.vtensor<[197,768],f32> -> tensor<197x768xf32>
    %57 = dnn.linear %56, %cst_4, %cst_8 : tensor<197x768xf32>, tensor<768x768xf32>, tensor<768xf32> -> tensor<197x768xf32>
    %58 = torch_c.from_builtin_tensor %57 : tensor<197x768xf32> -> !torch.vtensor<[197,768],f32>
    %59 = torch.prim.ListConstruct %int197, %int1, %int768 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %60 = torch.aten.view %58, %59 : !torch.vtensor<[197,768],f32>, !torch.list<int> -> !torch.vtensor<[197,1,768],f32>
    %61 = torch.aten.transpose.int %60, %int1, %int0 : !torch.vtensor<[197,1,768],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,197,768],f32>
    %62 = torch.aten.dropout %61, %float0.000000e00, %false : !torch.vtensor<[1,197,768],f32>, !torch.float, !torch.bool -> !torch.vtensor<[1,197,768],f32>
    %63 = torch_c.to_builtin_tensor %62 : !torch.vtensor<[1,197,768],f32> -> tensor<1x197x768xf32>
    %64 = torch_c.to_builtin_tensor %15 : !torch.vtensor<[1,197,768],f32> -> tensor<1x197x768xf32>
    %65 = dnn.add %63, %64 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %66 = torch_c.from_builtin_tensor %65 : tensor<1x197x768xf32> -> !torch.vtensor<[1,197,768],f32>
    %67 = torch_c.to_builtin_tensor %66 : !torch.vtensor<[1,197,768],f32> -> tensor<1x197x768xf32>
    %68 = dnn.layer_norm %67, %cst_8, %cst_8 {parameter_indices = array<i32: 1, 4, 5>, parameters = [[768], 9.9999999999999995E-7, false]} : (tensor<1x197x768xf32>, tensor<768xf32>, tensor<768xf32>) -> tensor<1x197x768xf32>
    %69 = torch_c.from_builtin_tensor %68 : tensor<1x197x768xf32> -> !torch.vtensor<[1,197,768],f32>
    %70 = torch_c.to_builtin_tensor %69 : !torch.vtensor<[1,197,768],f32> -> tensor<1x197x768xf32>
    %71 = dnn.linear %70, %cst_3, %cst_2 : tensor<1x197x768xf32>, tensor<3072x768xf32>, tensor<3072xf32> -> tensor<1x197x3072xf32>
    %72 = torch_c.from_builtin_tensor %71 : tensor<1x197x3072xf32> -> !torch.vtensor<[1,197,3072],f32>
    %73 = torch_c.to_builtin_tensor %72 : !torch.vtensor<[1,197,3072],f32> -> tensor<1x197x3072xf32>
    %74 = dnn.gelu %73 {parameter_indices = array<i32: 1>, parameters = ["none"]} : (tensor<1x197x3072xf32>) -> tensor<1x197x3072xf32>
    %75 = torch_c.from_builtin_tensor %74 : tensor<1x197x3072xf32> -> !torch.vtensor<[1,197,3072],f32>
    %76 = torch.aten.dropout %75, %float0.000000e00, %false : !torch.vtensor<[1,197,3072],f32>, !torch.float, !torch.bool -> !torch.vtensor<[1,197,3072],f32>
    %77 = torch_c.to_builtin_tensor %76 : !torch.vtensor<[1,197,3072],f32> -> tensor<1x197x3072xf32>
    %78 = dnn.linear %77, %cst_1, %cst_8 : tensor<1x197x3072xf32>, tensor<768x3072xf32>, tensor<768xf32> -> tensor<1x197x768xf32>
    %79 = torch_c.from_builtin_tensor %78 : tensor<1x197x768xf32> -> !torch.vtensor<[1,197,768],f32>
    %80 = torch.aten.dropout %79, %float0.000000e00, %false : !torch.vtensor<[1,197,768],f32>, !torch.float, !torch.bool -> !torch.vtensor<[1,197,768],f32>
    %81 = torch_c.to_builtin_tensor %66 : !torch.vtensor<[1,197,768],f32> -> tensor<1x197x768xf32>
    %82 = torch_c.to_builtin_tensor %80 : !torch.vtensor<[1,197,768],f32> -> tensor<1x197x768xf32>
    %83 = dnn.add %81, %82 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %84 = torch_c.from_builtin_tensor %83 : tensor<1x197x768xf32> -> !torch.vtensor<[1,197,768],f32>
    %85 = torch_c.to_builtin_tensor %84 : !torch.vtensor<[1,197,768],f32> -> tensor<1x197x768xf32>
    %86 = dnn.layer_norm %85, %cst_8, %cst_8 {parameter_indices = array<i32: 1, 4, 5>, parameters = [[768], 9.9999999999999995E-7, false]} : (tensor<1x197x768xf32>, tensor<768xf32>, tensor<768xf32>) -> tensor<1x197x768xf32>
    %87 = torch_c.from_builtin_tensor %86 : tensor<1x197x768xf32> -> !torch.vtensor<[1,197,768],f32>
    %88 = torch.aten.transpose.int %87, %int1, %int0 : !torch.vtensor<[1,197,768],f32>, !torch.int, !torch.int -> !torch.vtensor<[197,1,768],f32>
    %89 = torch_c.to_builtin_tensor %88 : !torch.vtensor<[197,1,768],f32> -> tensor<197x1x768xf32>
    %90 = dnn.linear %89, %cst_6, %cst_5 : tensor<197x1x768xf32>, tensor<2304x768xf32>, tensor<2304xf32> -> tensor<197x1x2304xf32>
    %91 = torch_c.from_builtin_tensor %90 : tensor<197x1x2304xf32> -> !torch.vtensor<[197,1,2304],f32>
    %92 = torch.prim.ListConstruct %int3, %int768 : (!torch.int, !torch.int) -> !torch.list<int>
    %93 = torch.aten.unflatten.int %91, %int-1, %92 : !torch.vtensor<[197,1,2304],f32>, !torch.int, !torch.list<int> -> !torch.vtensor<[197,1,3,768],f32>
    %94 = torch.aten.unsqueeze %93, %int0 : !torch.vtensor<[197,1,3,768],f32>, !torch.int -> !torch.vtensor<[1,197,1,3,768],f32>
    %95 = torch.aten.transpose.int %94, %int0, %int-2 : !torch.vtensor<[1,197,1,3,768],f32>, !torch.int, !torch.int -> !torch.vtensor<[3,197,1,1,768],f32>
    %96 = torch.aten.squeeze.dim %95, %int-2 : !torch.vtensor<[3,197,1,1,768],f32>, !torch.int -> !torch.vtensor<[3,197,1,768],f32>
    %97 = torch.aten.contiguous %96, %int0 : !torch.vtensor<[3,197,1,768],f32>, !torch.int -> !torch.vtensor<[3,197,1,768],f32>
    %98 = torch.aten.select.int %97, %int0, %int0 : !torch.vtensor<[3,197,1,768],f32>, !torch.int, !torch.int -> !torch.vtensor<[197,1,768],f32>
    %99 = torch.aten.select.int %97, %int0, %int1 : !torch.vtensor<[3,197,1,768],f32>, !torch.int, !torch.int -> !torch.vtensor<[197,1,768],f32>
    %100 = torch.aten.select.int %97, %int0, %int2 : !torch.vtensor<[3,197,1,768],f32>, !torch.int, !torch.int -> !torch.vtensor<[197,1,768],f32>
    %101 = torch.prim.ListConstruct %int197, %int12, %int64 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %102 = torch.aten.view %98, %101 : !torch.vtensor<[197,1,768],f32>, !torch.list<int> -> !torch.vtensor<[197,12,64],f32>
    %103 = torch.aten.transpose.int %102, %int0, %int1 : !torch.vtensor<[197,12,64],f32>, !torch.int, !torch.int -> !torch.vtensor<[12,197,64],f32>
    %104 = torch.prim.ListConstruct %int197, %int12, %int64 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %105 = torch.aten.view %99, %104 : !torch.vtensor<[197,1,768],f32>, !torch.list<int> -> !torch.vtensor<[197,12,64],f32>
    %106 = torch.aten.transpose.int %105, %int0, %int1 : !torch.vtensor<[197,12,64],f32>, !torch.int, !torch.int -> !torch.vtensor<[12,197,64],f32>
    %107 = torch.prim.ListConstruct %int197, %int12, %int64 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %108 = torch.aten.view %100, %107 : !torch.vtensor<[197,1,768],f32>, !torch.list<int> -> !torch.vtensor<[197,12,64],f32>
    %109 = torch.aten.transpose.int %108, %int0, %int1 : !torch.vtensor<[197,12,64],f32>, !torch.int, !torch.int -> !torch.vtensor<[12,197,64],f32>
    %110 = torch.prim.ListConstruct %int1, %int12, %int197, %int64 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %111 = torch.aten.view %103, %110 : !torch.vtensor<[12,197,64],f32>, !torch.list<int> -> !torch.vtensor<[1,12,197,64],f32>
    %112 = torch.prim.ListConstruct %int1, %int12, %int197, %int64 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %113 = torch.aten.view %106, %112 : !torch.vtensor<[12,197,64],f32>, !torch.list<int> -> !torch.vtensor<[1,12,197,64],f32>
    %114 = torch.prim.ListConstruct %int1, %int12, %int197, %int64 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %115 = torch.aten.view %109, %114 : !torch.vtensor<[12,197,64],f32>, !torch.list<int> -> !torch.vtensor<[1,12,197,64],f32>
    %116 = torch_c.to_builtin_tensor %111 : !torch.vtensor<[1,12,197,64],f32> -> tensor<1x12x197x64xf32>
    %117 = torch_c.to_builtin_tensor %113 : !torch.vtensor<[1,12,197,64],f32> -> tensor<1x12x197x64xf32>
    %118 = torch_c.to_builtin_tensor %115 : !torch.vtensor<[1,12,197,64],f32> -> tensor<1x12x197x64xf32>
    %119 = dnn.scaled_dot_product_attention %116, %117, %118 {parameter_indices = array<i32: 3, 4, 5, 6, 7>, parameters = [unit, 0.000000e+00, false, unit, false]} : (tensor<1x12x197x64xf32>, tensor<1x12x197x64xf32>, tensor<1x12x197x64xf32>) -> tensor<1x12x197x64xf32>
    %120 = torch_c.from_builtin_tensor %119 : tensor<1x12x197x64xf32> -> !torch.vtensor<[1,12,197,64],f32>
    %121 = torch.prim.ListConstruct %int2, %int0, %int1, %int3 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %122 = torch.aten.permute %120, %121 : !torch.vtensor<[1,12,197,64],f32>, !torch.list<int> -> !torch.vtensor<[197,1,12,64],f32>
    %123 = torch.prim.ListConstruct %int197, %int768 : (!torch.int, !torch.int) -> !torch.list<int>
    %124 = torch.aten.view %122, %123 : !torch.vtensor<[197,1,12,64],f32>, !torch.list<int> -> !torch.vtensor<[197,768],f32>
    %125 = torch_c.to_builtin_tensor %124 : !torch.vtensor<[197,768],f32> -> tensor<197x768xf32>
    %126 = dnn.linear %125, %cst_4, %cst_8 : tensor<197x768xf32>, tensor<768x768xf32>, tensor<768xf32> -> tensor<197x768xf32>
    %127 = torch_c.from_builtin_tensor %126 : tensor<197x768xf32> -> !torch.vtensor<[197,768],f32>
    %128 = torch.prim.ListConstruct %int197, %int1, %int768 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %129 = torch.aten.view %127, %128 : !torch.vtensor<[197,768],f32>, !torch.list<int> -> !torch.vtensor<[197,1,768],f32>
    %130 = torch.aten.transpose.int %129, %int1, %int0 : !torch.vtensor<[197,1,768],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,197,768],f32>
    %131 = torch.aten.dropout %130, %float0.000000e00, %false : !torch.vtensor<[1,197,768],f32>, !torch.float, !torch.bool -> !torch.vtensor<[1,197,768],f32>
    %132 = torch_c.to_builtin_tensor %131 : !torch.vtensor<[1,197,768],f32> -> tensor<1x197x768xf32>
    %133 = torch_c.to_builtin_tensor %84 : !torch.vtensor<[1,197,768],f32> -> tensor<1x197x768xf32>
    %134 = dnn.add %132, %133 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %135 = torch_c.from_builtin_tensor %134 : tensor<1x197x768xf32> -> !torch.vtensor<[1,197,768],f32>
    %136 = torch_c.to_builtin_tensor %135 : !torch.vtensor<[1,197,768],f32> -> tensor<1x197x768xf32>
    %137 = dnn.layer_norm %136, %cst_8, %cst_8 {parameter_indices = array<i32: 1, 4, 5>, parameters = [[768], 9.9999999999999995E-7, false]} : (tensor<1x197x768xf32>, tensor<768xf32>, tensor<768xf32>) -> tensor<1x197x768xf32>
    %138 = torch_c.from_builtin_tensor %137 : tensor<1x197x768xf32> -> !torch.vtensor<[1,197,768],f32>
    %139 = torch_c.to_builtin_tensor %138 : !torch.vtensor<[1,197,768],f32> -> tensor<1x197x768xf32>
    %140 = dnn.linear %139, %cst_3, %cst_2 : tensor<1x197x768xf32>, tensor<3072x768xf32>, tensor<3072xf32> -> tensor<1x197x3072xf32>
    %141 = torch_c.from_builtin_tensor %140 : tensor<1x197x3072xf32> -> !torch.vtensor<[1,197,3072],f32>
    %142 = torch_c.to_builtin_tensor %141 : !torch.vtensor<[1,197,3072],f32> -> tensor<1x197x3072xf32>
    %143 = dnn.gelu %142 {parameter_indices = array<i32: 1>, parameters = ["none"]} : (tensor<1x197x3072xf32>) -> tensor<1x197x3072xf32>
    %144 = torch_c.from_builtin_tensor %143 : tensor<1x197x3072xf32> -> !torch.vtensor<[1,197,3072],f32>
    %145 = torch.aten.dropout %144, %float0.000000e00, %false : !torch.vtensor<[1,197,3072],f32>, !torch.float, !torch.bool -> !torch.vtensor<[1,197,3072],f32>
    %146 = torch_c.to_builtin_tensor %145 : !torch.vtensor<[1,197,3072],f32> -> tensor<1x197x3072xf32>
    %147 = dnn.linear %146, %cst_1, %cst_8 : tensor<1x197x3072xf32>, tensor<768x3072xf32>, tensor<768xf32> -> tensor<1x197x768xf32>
    %148 = torch_c.from_builtin_tensor %147 : tensor<1x197x768xf32> -> !torch.vtensor<[1,197,768],f32>
    %149 = torch.aten.dropout %148, %float0.000000e00, %false : !torch.vtensor<[1,197,768],f32>, !torch.float, !torch.bool -> !torch.vtensor<[1,197,768],f32>
    %150 = torch_c.to_builtin_tensor %135 : !torch.vtensor<[1,197,768],f32> -> tensor<1x197x768xf32>
    %151 = torch_c.to_builtin_tensor %149 : !torch.vtensor<[1,197,768],f32> -> tensor<1x197x768xf32>
    %152 = dnn.add %150, %151 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %153 = torch_c.from_builtin_tensor %152 : tensor<1x197x768xf32> -> !torch.vtensor<[1,197,768],f32>
    %154 = torch_c.to_builtin_tensor %153 : !torch.vtensor<[1,197,768],f32> -> tensor<1x197x768xf32>
    %155 = dnn.layer_norm %154, %cst_8, %cst_8 {parameter_indices = array<i32: 1, 4, 5>, parameters = [[768], 9.9999999999999995E-7, false]} : (tensor<1x197x768xf32>, tensor<768xf32>, tensor<768xf32>) -> tensor<1x197x768xf32>
    %156 = torch_c.from_builtin_tensor %155 : tensor<1x197x768xf32> -> !torch.vtensor<[1,197,768],f32>
    %157 = torch.aten.transpose.int %156, %int1, %int0 : !torch.vtensor<[1,197,768],f32>, !torch.int, !torch.int -> !torch.vtensor<[197,1,768],f32>
    %158 = torch_c.to_builtin_tensor %157 : !torch.vtensor<[197,1,768],f32> -> tensor<197x1x768xf32>
    %159 = dnn.linear %158, %cst_6, %cst_5 : tensor<197x1x768xf32>, tensor<2304x768xf32>, tensor<2304xf32> -> tensor<197x1x2304xf32>
    %160 = torch_c.from_builtin_tensor %159 : tensor<197x1x2304xf32> -> !torch.vtensor<[197,1,2304],f32>
    %161 = torch.prim.ListConstruct %int3, %int768 : (!torch.int, !torch.int) -> !torch.list<int>
    %162 = torch.aten.unflatten.int %160, %int-1, %161 : !torch.vtensor<[197,1,2304],f32>, !torch.int, !torch.list<int> -> !torch.vtensor<[197,1,3,768],f32>
    %163 = torch.aten.unsqueeze %162, %int0 : !torch.vtensor<[197,1,3,768],f32>, !torch.int -> !torch.vtensor<[1,197,1,3,768],f32>
    %164 = torch.aten.transpose.int %163, %int0, %int-2 : !torch.vtensor<[1,197,1,3,768],f32>, !torch.int, !torch.int -> !torch.vtensor<[3,197,1,1,768],f32>
    %165 = torch.aten.squeeze.dim %164, %int-2 : !torch.vtensor<[3,197,1,1,768],f32>, !torch.int -> !torch.vtensor<[3,197,1,768],f32>
    %166 = torch.aten.contiguous %165, %int0 : !torch.vtensor<[3,197,1,768],f32>, !torch.int -> !torch.vtensor<[3,197,1,768],f32>
    %167 = torch.aten.select.int %166, %int0, %int0 : !torch.vtensor<[3,197,1,768],f32>, !torch.int, !torch.int -> !torch.vtensor<[197,1,768],f32>
    %168 = torch.aten.select.int %166, %int0, %int1 : !torch.vtensor<[3,197,1,768],f32>, !torch.int, !torch.int -> !torch.vtensor<[197,1,768],f32>
    %169 = torch.aten.select.int %166, %int0, %int2 : !torch.vtensor<[3,197,1,768],f32>, !torch.int, !torch.int -> !torch.vtensor<[197,1,768],f32>
    %170 = torch.prim.ListConstruct %int197, %int12, %int64 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %171 = torch.aten.view %167, %170 : !torch.vtensor<[197,1,768],f32>, !torch.list<int> -> !torch.vtensor<[197,12,64],f32>
    %172 = torch.aten.transpose.int %171, %int0, %int1 : !torch.vtensor<[197,12,64],f32>, !torch.int, !torch.int -> !torch.vtensor<[12,197,64],f32>
    %173 = torch.prim.ListConstruct %int197, %int12, %int64 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %174 = torch.aten.view %168, %173 : !torch.vtensor<[197,1,768],f32>, !torch.list<int> -> !torch.vtensor<[197,12,64],f32>
    %175 = torch.aten.transpose.int %174, %int0, %int1 : !torch.vtensor<[197,12,64],f32>, !torch.int, !torch.int -> !torch.vtensor<[12,197,64],f32>
    %176 = torch.prim.ListConstruct %int197, %int12, %int64 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %177 = torch.aten.view %169, %176 : !torch.vtensor<[197,1,768],f32>, !torch.list<int> -> !torch.vtensor<[197,12,64],f32>
    %178 = torch.aten.transpose.int %177, %int0, %int1 : !torch.vtensor<[197,12,64],f32>, !torch.int, !torch.int -> !torch.vtensor<[12,197,64],f32>
    %179 = torch.prim.ListConstruct %int1, %int12, %int197, %int64 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %180 = torch.aten.view %172, %179 : !torch.vtensor<[12,197,64],f32>, !torch.list<int> -> !torch.vtensor<[1,12,197,64],f32>
    %181 = torch.prim.ListConstruct %int1, %int12, %int197, %int64 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %182 = torch.aten.view %175, %181 : !torch.vtensor<[12,197,64],f32>, !torch.list<int> -> !torch.vtensor<[1,12,197,64],f32>
    %183 = torch.prim.ListConstruct %int1, %int12, %int197, %int64 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %184 = torch.aten.view %178, %183 : !torch.vtensor<[12,197,64],f32>, !torch.list<int> -> !torch.vtensor<[1,12,197,64],f32>
    %185 = torch_c.to_builtin_tensor %180 : !torch.vtensor<[1,12,197,64],f32> -> tensor<1x12x197x64xf32>
    %186 = torch_c.to_builtin_tensor %182 : !torch.vtensor<[1,12,197,64],f32> -> tensor<1x12x197x64xf32>
    %187 = torch_c.to_builtin_tensor %184 : !torch.vtensor<[1,12,197,64],f32> -> tensor<1x12x197x64xf32>
    %188 = dnn.scaled_dot_product_attention %185, %186, %187 {parameter_indices = array<i32: 3, 4, 5, 6, 7>, parameters = [unit, 0.000000e+00, false, unit, false]} : (tensor<1x12x197x64xf32>, tensor<1x12x197x64xf32>, tensor<1x12x197x64xf32>) -> tensor<1x12x197x64xf32>
    %189 = torch_c.from_builtin_tensor %188 : tensor<1x12x197x64xf32> -> !torch.vtensor<[1,12,197,64],f32>
    %190 = torch.prim.ListConstruct %int2, %int0, %int1, %int3 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %191 = torch.aten.permute %189, %190 : !torch.vtensor<[1,12,197,64],f32>, !torch.list<int> -> !torch.vtensor<[197,1,12,64],f32>
    %192 = torch.prim.ListConstruct %int197, %int768 : (!torch.int, !torch.int) -> !torch.list<int>
    %193 = torch.aten.view %191, %192 : !torch.vtensor<[197,1,12,64],f32>, !torch.list<int> -> !torch.vtensor<[197,768],f32>
    %194 = torch_c.to_builtin_tensor %193 : !torch.vtensor<[197,768],f32> -> tensor<197x768xf32>
    %195 = dnn.linear %194, %cst_4, %cst_8 : tensor<197x768xf32>, tensor<768x768xf32>, tensor<768xf32> -> tensor<197x768xf32>
    %196 = torch_c.from_builtin_tensor %195 : tensor<197x768xf32> -> !torch.vtensor<[197,768],f32>
    %197 = torch.prim.ListConstruct %int197, %int1, %int768 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %198 = torch.aten.view %196, %197 : !torch.vtensor<[197,768],f32>, !torch.list<int> -> !torch.vtensor<[197,1,768],f32>
    %199 = torch.aten.transpose.int %198, %int1, %int0 : !torch.vtensor<[197,1,768],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,197,768],f32>
    %200 = torch.aten.dropout %199, %float0.000000e00, %false : !torch.vtensor<[1,197,768],f32>, !torch.float, !torch.bool -> !torch.vtensor<[1,197,768],f32>
    %201 = torch_c.to_builtin_tensor %200 : !torch.vtensor<[1,197,768],f32> -> tensor<1x197x768xf32>
    %202 = torch_c.to_builtin_tensor %153 : !torch.vtensor<[1,197,768],f32> -> tensor<1x197x768xf32>
    %203 = dnn.add %201, %202 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %204 = torch_c.from_builtin_tensor %203 : tensor<1x197x768xf32> -> !torch.vtensor<[1,197,768],f32>
    %205 = torch_c.to_builtin_tensor %204 : !torch.vtensor<[1,197,768],f32> -> tensor<1x197x768xf32>
    %206 = dnn.layer_norm %205, %cst_8, %cst_8 {parameter_indices = array<i32: 1, 4, 5>, parameters = [[768], 9.9999999999999995E-7, false]} : (tensor<1x197x768xf32>, tensor<768xf32>, tensor<768xf32>) -> tensor<1x197x768xf32>
    %207 = torch_c.from_builtin_tensor %206 : tensor<1x197x768xf32> -> !torch.vtensor<[1,197,768],f32>
    %208 = torch_c.to_builtin_tensor %207 : !torch.vtensor<[1,197,768],f32> -> tensor<1x197x768xf32>
    %209 = dnn.linear %208, %cst_3, %cst_2 : tensor<1x197x768xf32>, tensor<3072x768xf32>, tensor<3072xf32> -> tensor<1x197x3072xf32>
    %210 = torch_c.from_builtin_tensor %209 : tensor<1x197x3072xf32> -> !torch.vtensor<[1,197,3072],f32>
    %211 = torch_c.to_builtin_tensor %210 : !torch.vtensor<[1,197,3072],f32> -> tensor<1x197x3072xf32>
    %212 = dnn.gelu %211 {parameter_indices = array<i32: 1>, parameters = ["none"]} : (tensor<1x197x3072xf32>) -> tensor<1x197x3072xf32>
    %213 = torch_c.from_builtin_tensor %212 : tensor<1x197x3072xf32> -> !torch.vtensor<[1,197,3072],f32>
    %214 = torch.aten.dropout %213, %float0.000000e00, %false : !torch.vtensor<[1,197,3072],f32>, !torch.float, !torch.bool -> !torch.vtensor<[1,197,3072],f32>
    %215 = torch_c.to_builtin_tensor %214 : !torch.vtensor<[1,197,3072],f32> -> tensor<1x197x3072xf32>
    %216 = dnn.linear %215, %cst_1, %cst_8 : tensor<1x197x3072xf32>, tensor<768x3072xf32>, tensor<768xf32> -> tensor<1x197x768xf32>
    %217 = torch_c.from_builtin_tensor %216 : tensor<1x197x768xf32> -> !torch.vtensor<[1,197,768],f32>
    %218 = torch.aten.dropout %217, %float0.000000e00, %false : !torch.vtensor<[1,197,768],f32>, !torch.float, !torch.bool -> !torch.vtensor<[1,197,768],f32>
    %219 = torch_c.to_builtin_tensor %204 : !torch.vtensor<[1,197,768],f32> -> tensor<1x197x768xf32>
    %220 = torch_c.to_builtin_tensor %218 : !torch.vtensor<[1,197,768],f32> -> tensor<1x197x768xf32>
    %221 = dnn.add %219, %220 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %222 = torch_c.from_builtin_tensor %221 : tensor<1x197x768xf32> -> !torch.vtensor<[1,197,768],f32>
    %223 = torch_c.to_builtin_tensor %222 : !torch.vtensor<[1,197,768],f32> -> tensor<1x197x768xf32>
    %224 = dnn.layer_norm %223, %cst_8, %cst_8 {parameter_indices = array<i32: 1, 4, 5>, parameters = [[768], 9.9999999999999995E-7, false]} : (tensor<1x197x768xf32>, tensor<768xf32>, tensor<768xf32>) -> tensor<1x197x768xf32>
    %225 = torch_c.from_builtin_tensor %224 : tensor<1x197x768xf32> -> !torch.vtensor<[1,197,768],f32>
    %226 = torch.aten.transpose.int %225, %int1, %int0 : !torch.vtensor<[1,197,768],f32>, !torch.int, !torch.int -> !torch.vtensor<[197,1,768],f32>
    %227 = torch_c.to_builtin_tensor %226 : !torch.vtensor<[197,1,768],f32> -> tensor<197x1x768xf32>
    %228 = dnn.linear %227, %cst_6, %cst_5 : tensor<197x1x768xf32>, tensor<2304x768xf32>, tensor<2304xf32> -> tensor<197x1x2304xf32>
    %229 = torch_c.from_builtin_tensor %228 : tensor<197x1x2304xf32> -> !torch.vtensor<[197,1,2304],f32>
    %230 = torch.prim.ListConstruct %int3, %int768 : (!torch.int, !torch.int) -> !torch.list<int>
    %231 = torch.aten.unflatten.int %229, %int-1, %230 : !torch.vtensor<[197,1,2304],f32>, !torch.int, !torch.list<int> -> !torch.vtensor<[197,1,3,768],f32>
    %232 = torch.aten.unsqueeze %231, %int0 : !torch.vtensor<[197,1,3,768],f32>, !torch.int -> !torch.vtensor<[1,197,1,3,768],f32>
    %233 = torch.aten.transpose.int %232, %int0, %int-2 : !torch.vtensor<[1,197,1,3,768],f32>, !torch.int, !torch.int -> !torch.vtensor<[3,197,1,1,768],f32>
    %234 = torch.aten.squeeze.dim %233, %int-2 : !torch.vtensor<[3,197,1,1,768],f32>, !torch.int -> !torch.vtensor<[3,197,1,768],f32>
    %235 = torch.aten.contiguous %234, %int0 : !torch.vtensor<[3,197,1,768],f32>, !torch.int -> !torch.vtensor<[3,197,1,768],f32>
    %236 = torch.aten.select.int %235, %int0, %int0 : !torch.vtensor<[3,197,1,768],f32>, !torch.int, !torch.int -> !torch.vtensor<[197,1,768],f32>
    %237 = torch.aten.select.int %235, %int0, %int1 : !torch.vtensor<[3,197,1,768],f32>, !torch.int, !torch.int -> !torch.vtensor<[197,1,768],f32>
    %238 = torch.aten.select.int %235, %int0, %int2 : !torch.vtensor<[3,197,1,768],f32>, !torch.int, !torch.int -> !torch.vtensor<[197,1,768],f32>
    %239 = torch.prim.ListConstruct %int197, %int12, %int64 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %240 = torch.aten.view %236, %239 : !torch.vtensor<[197,1,768],f32>, !torch.list<int> -> !torch.vtensor<[197,12,64],f32>
    %241 = torch.aten.transpose.int %240, %int0, %int1 : !torch.vtensor<[197,12,64],f32>, !torch.int, !torch.int -> !torch.vtensor<[12,197,64],f32>
    %242 = torch.prim.ListConstruct %int197, %int12, %int64 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %243 = torch.aten.view %237, %242 : !torch.vtensor<[197,1,768],f32>, !torch.list<int> -> !torch.vtensor<[197,12,64],f32>
    %244 = torch.aten.transpose.int %243, %int0, %int1 : !torch.vtensor<[197,12,64],f32>, !torch.int, !torch.int -> !torch.vtensor<[12,197,64],f32>
    %245 = torch.prim.ListConstruct %int197, %int12, %int64 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %246 = torch.aten.view %238, %245 : !torch.vtensor<[197,1,768],f32>, !torch.list<int> -> !torch.vtensor<[197,12,64],f32>
    %247 = torch.aten.transpose.int %246, %int0, %int1 : !torch.vtensor<[197,12,64],f32>, !torch.int, !torch.int -> !torch.vtensor<[12,197,64],f32>
    %248 = torch.prim.ListConstruct %int1, %int12, %int197, %int64 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %249 = torch.aten.view %241, %248 : !torch.vtensor<[12,197,64],f32>, !torch.list<int> -> !torch.vtensor<[1,12,197,64],f32>
    %250 = torch.prim.ListConstruct %int1, %int12, %int197, %int64 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %251 = torch.aten.view %244, %250 : !torch.vtensor<[12,197,64],f32>, !torch.list<int> -> !torch.vtensor<[1,12,197,64],f32>
    %252 = torch.prim.ListConstruct %int1, %int12, %int197, %int64 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %253 = torch.aten.view %247, %252 : !torch.vtensor<[12,197,64],f32>, !torch.list<int> -> !torch.vtensor<[1,12,197,64],f32>
    %254 = torch_c.to_builtin_tensor %249 : !torch.vtensor<[1,12,197,64],f32> -> tensor<1x12x197x64xf32>
    %255 = torch_c.to_builtin_tensor %251 : !torch.vtensor<[1,12,197,64],f32> -> tensor<1x12x197x64xf32>
    %256 = torch_c.to_builtin_tensor %253 : !torch.vtensor<[1,12,197,64],f32> -> tensor<1x12x197x64xf32>
    %257 = dnn.scaled_dot_product_attention %254, %255, %256 {parameter_indices = array<i32: 3, 4, 5, 6, 7>, parameters = [unit, 0.000000e+00, false, unit, false]} : (tensor<1x12x197x64xf32>, tensor<1x12x197x64xf32>, tensor<1x12x197x64xf32>) -> tensor<1x12x197x64xf32>
    %258 = torch_c.from_builtin_tensor %257 : tensor<1x12x197x64xf32> -> !torch.vtensor<[1,12,197,64],f32>
    %259 = torch.prim.ListConstruct %int2, %int0, %int1, %int3 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %260 = torch.aten.permute %258, %259 : !torch.vtensor<[1,12,197,64],f32>, !torch.list<int> -> !torch.vtensor<[197,1,12,64],f32>
    %261 = torch.prim.ListConstruct %int197, %int768 : (!torch.int, !torch.int) -> !torch.list<int>
    %262 = torch.aten.view %260, %261 : !torch.vtensor<[197,1,12,64],f32>, !torch.list<int> -> !torch.vtensor<[197,768],f32>
    %263 = torch_c.to_builtin_tensor %262 : !torch.vtensor<[197,768],f32> -> tensor<197x768xf32>
    %264 = dnn.linear %263, %cst_4, %cst_8 : tensor<197x768xf32>, tensor<768x768xf32>, tensor<768xf32> -> tensor<197x768xf32>
    %265 = torch_c.from_builtin_tensor %264 : tensor<197x768xf32> -> !torch.vtensor<[197,768],f32>
    %266 = torch.prim.ListConstruct %int197, %int1, %int768 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %267 = torch.aten.view %265, %266 : !torch.vtensor<[197,768],f32>, !torch.list<int> -> !torch.vtensor<[197,1,768],f32>
    %268 = torch.aten.transpose.int %267, %int1, %int0 : !torch.vtensor<[197,1,768],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,197,768],f32>
    %269 = torch.aten.dropout %268, %float0.000000e00, %false : !torch.vtensor<[1,197,768],f32>, !torch.float, !torch.bool -> !torch.vtensor<[1,197,768],f32>
    %270 = torch_c.to_builtin_tensor %269 : !torch.vtensor<[1,197,768],f32> -> tensor<1x197x768xf32>
    %271 = torch_c.to_builtin_tensor %222 : !torch.vtensor<[1,197,768],f32> -> tensor<1x197x768xf32>
    %272 = dnn.add %270, %271 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %273 = torch_c.from_builtin_tensor %272 : tensor<1x197x768xf32> -> !torch.vtensor<[1,197,768],f32>
    %274 = torch_c.to_builtin_tensor %273 : !torch.vtensor<[1,197,768],f32> -> tensor<1x197x768xf32>
    %275 = dnn.layer_norm %274, %cst_8, %cst_8 {parameter_indices = array<i32: 1, 4, 5>, parameters = [[768], 9.9999999999999995E-7, false]} : (tensor<1x197x768xf32>, tensor<768xf32>, tensor<768xf32>) -> tensor<1x197x768xf32>
    %276 = torch_c.from_builtin_tensor %275 : tensor<1x197x768xf32> -> !torch.vtensor<[1,197,768],f32>
    %277 = torch_c.to_builtin_tensor %276 : !torch.vtensor<[1,197,768],f32> -> tensor<1x197x768xf32>
    %278 = dnn.linear %277, %cst_3, %cst_2 : tensor<1x197x768xf32>, tensor<3072x768xf32>, tensor<3072xf32> -> tensor<1x197x3072xf32>
    %279 = torch_c.from_builtin_tensor %278 : tensor<1x197x3072xf32> -> !torch.vtensor<[1,197,3072],f32>
    %280 = torch_c.to_builtin_tensor %279 : !torch.vtensor<[1,197,3072],f32> -> tensor<1x197x3072xf32>
    %281 = dnn.gelu %280 {parameter_indices = array<i32: 1>, parameters = ["none"]} : (tensor<1x197x3072xf32>) -> tensor<1x197x3072xf32>
    %282 = torch_c.from_builtin_tensor %281 : tensor<1x197x3072xf32> -> !torch.vtensor<[1,197,3072],f32>
    %283 = torch.aten.dropout %282, %float0.000000e00, %false : !torch.vtensor<[1,197,3072],f32>, !torch.float, !torch.bool -> !torch.vtensor<[1,197,3072],f32>
    %284 = torch_c.to_builtin_tensor %283 : !torch.vtensor<[1,197,3072],f32> -> tensor<1x197x3072xf32>
    %285 = dnn.linear %284, %cst_1, %cst_8 : tensor<1x197x3072xf32>, tensor<768x3072xf32>, tensor<768xf32> -> tensor<1x197x768xf32>
    %286 = torch_c.from_builtin_tensor %285 : tensor<1x197x768xf32> -> !torch.vtensor<[1,197,768],f32>
    %287 = torch.aten.dropout %286, %float0.000000e00, %false : !torch.vtensor<[1,197,768],f32>, !torch.float, !torch.bool -> !torch.vtensor<[1,197,768],f32>
    %288 = torch_c.to_builtin_tensor %273 : !torch.vtensor<[1,197,768],f32> -> tensor<1x197x768xf32>
    %289 = torch_c.to_builtin_tensor %287 : !torch.vtensor<[1,197,768],f32> -> tensor<1x197x768xf32>
    %290 = dnn.add %288, %289 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %291 = torch_c.from_builtin_tensor %290 : tensor<1x197x768xf32> -> !torch.vtensor<[1,197,768],f32>
    %292 = torch_c.to_builtin_tensor %291 : !torch.vtensor<[1,197,768],f32> -> tensor<1x197x768xf32>
    %293 = dnn.layer_norm %292, %cst_8, %cst_8 {parameter_indices = array<i32: 1, 4, 5>, parameters = [[768], 9.9999999999999995E-7, false]} : (tensor<1x197x768xf32>, tensor<768xf32>, tensor<768xf32>) -> tensor<1x197x768xf32>
    %294 = torch_c.from_builtin_tensor %293 : tensor<1x197x768xf32> -> !torch.vtensor<[1,197,768],f32>
    %295 = torch.aten.transpose.int %294, %int1, %int0 : !torch.vtensor<[1,197,768],f32>, !torch.int, !torch.int -> !torch.vtensor<[197,1,768],f32>
    %296 = torch_c.to_builtin_tensor %295 : !torch.vtensor<[197,1,768],f32> -> tensor<197x1x768xf32>
    %297 = dnn.linear %296, %cst_6, %cst_5 : tensor<197x1x768xf32>, tensor<2304x768xf32>, tensor<2304xf32> -> tensor<197x1x2304xf32>
    %298 = torch_c.from_builtin_tensor %297 : tensor<197x1x2304xf32> -> !torch.vtensor<[197,1,2304],f32>
    %299 = torch.prim.ListConstruct %int3, %int768 : (!torch.int, !torch.int) -> !torch.list<int>
    %300 = torch.aten.unflatten.int %298, %int-1, %299 : !torch.vtensor<[197,1,2304],f32>, !torch.int, !torch.list<int> -> !torch.vtensor<[197,1,3,768],f32>
    %301 = torch.aten.unsqueeze %300, %int0 : !torch.vtensor<[197,1,3,768],f32>, !torch.int -> !torch.vtensor<[1,197,1,3,768],f32>
    %302 = torch.aten.transpose.int %301, %int0, %int-2 : !torch.vtensor<[1,197,1,3,768],f32>, !torch.int, !torch.int -> !torch.vtensor<[3,197,1,1,768],f32>
    %303 = torch.aten.squeeze.dim %302, %int-2 : !torch.vtensor<[3,197,1,1,768],f32>, !torch.int -> !torch.vtensor<[3,197,1,768],f32>
    %304 = torch.aten.contiguous %303, %int0 : !torch.vtensor<[3,197,1,768],f32>, !torch.int -> !torch.vtensor<[3,197,1,768],f32>
    %305 = torch.aten.select.int %304, %int0, %int0 : !torch.vtensor<[3,197,1,768],f32>, !torch.int, !torch.int -> !torch.vtensor<[197,1,768],f32>
    %306 = torch.aten.select.int %304, %int0, %int1 : !torch.vtensor<[3,197,1,768],f32>, !torch.int, !torch.int -> !torch.vtensor<[197,1,768],f32>
    %307 = torch.aten.select.int %304, %int0, %int2 : !torch.vtensor<[3,197,1,768],f32>, !torch.int, !torch.int -> !torch.vtensor<[197,1,768],f32>
    %308 = torch.prim.ListConstruct %int197, %int12, %int64 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %309 = torch.aten.view %305, %308 : !torch.vtensor<[197,1,768],f32>, !torch.list<int> -> !torch.vtensor<[197,12,64],f32>
    %310 = torch.aten.transpose.int %309, %int0, %int1 : !torch.vtensor<[197,12,64],f32>, !torch.int, !torch.int -> !torch.vtensor<[12,197,64],f32>
    %311 = torch.prim.ListConstruct %int197, %int12, %int64 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %312 = torch.aten.view %306, %311 : !torch.vtensor<[197,1,768],f32>, !torch.list<int> -> !torch.vtensor<[197,12,64],f32>
    %313 = torch.aten.transpose.int %312, %int0, %int1 : !torch.vtensor<[197,12,64],f32>, !torch.int, !torch.int -> !torch.vtensor<[12,197,64],f32>
    %314 = torch.prim.ListConstruct %int197, %int12, %int64 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %315 = torch.aten.view %307, %314 : !torch.vtensor<[197,1,768],f32>, !torch.list<int> -> !torch.vtensor<[197,12,64],f32>
    %316 = torch.aten.transpose.int %315, %int0, %int1 : !torch.vtensor<[197,12,64],f32>, !torch.int, !torch.int -> !torch.vtensor<[12,197,64],f32>
    %317 = torch.prim.ListConstruct %int1, %int12, %int197, %int64 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %318 = torch.aten.view %310, %317 : !torch.vtensor<[12,197,64],f32>, !torch.list<int> -> !torch.vtensor<[1,12,197,64],f32>
    %319 = torch.prim.ListConstruct %int1, %int12, %int197, %int64 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %320 = torch.aten.view %313, %319 : !torch.vtensor<[12,197,64],f32>, !torch.list<int> -> !torch.vtensor<[1,12,197,64],f32>
    %321 = torch.prim.ListConstruct %int1, %int12, %int197, %int64 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %322 = torch.aten.view %316, %321 : !torch.vtensor<[12,197,64],f32>, !torch.list<int> -> !torch.vtensor<[1,12,197,64],f32>
    %323 = torch_c.to_builtin_tensor %318 : !torch.vtensor<[1,12,197,64],f32> -> tensor<1x12x197x64xf32>
    %324 = torch_c.to_builtin_tensor %320 : !torch.vtensor<[1,12,197,64],f32> -> tensor<1x12x197x64xf32>
    %325 = torch_c.to_builtin_tensor %322 : !torch.vtensor<[1,12,197,64],f32> -> tensor<1x12x197x64xf32>
    %326 = dnn.scaled_dot_product_attention %323, %324, %325 {parameter_indices = array<i32: 3, 4, 5, 6, 7>, parameters = [unit, 0.000000e+00, false, unit, false]} : (tensor<1x12x197x64xf32>, tensor<1x12x197x64xf32>, tensor<1x12x197x64xf32>) -> tensor<1x12x197x64xf32>
    %327 = torch_c.from_builtin_tensor %326 : tensor<1x12x197x64xf32> -> !torch.vtensor<[1,12,197,64],f32>
    %328 = torch.prim.ListConstruct %int2, %int0, %int1, %int3 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %329 = torch.aten.permute %327, %328 : !torch.vtensor<[1,12,197,64],f32>, !torch.list<int> -> !torch.vtensor<[197,1,12,64],f32>
    %330 = torch.prim.ListConstruct %int197, %int768 : (!torch.int, !torch.int) -> !torch.list<int>
    %331 = torch.aten.view %329, %330 : !torch.vtensor<[197,1,12,64],f32>, !torch.list<int> -> !torch.vtensor<[197,768],f32>
    %332 = torch_c.to_builtin_tensor %331 : !torch.vtensor<[197,768],f32> -> tensor<197x768xf32>
    %333 = dnn.linear %332, %cst_4, %cst_8 : tensor<197x768xf32>, tensor<768x768xf32>, tensor<768xf32> -> tensor<197x768xf32>
    %334 = torch_c.from_builtin_tensor %333 : tensor<197x768xf32> -> !torch.vtensor<[197,768],f32>
    %335 = torch.prim.ListConstruct %int197, %int1, %int768 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %336 = torch.aten.view %334, %335 : !torch.vtensor<[197,768],f32>, !torch.list<int> -> !torch.vtensor<[197,1,768],f32>
    %337 = torch.aten.transpose.int %336, %int1, %int0 : !torch.vtensor<[197,1,768],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,197,768],f32>
    %338 = torch.aten.dropout %337, %float0.000000e00, %false : !torch.vtensor<[1,197,768],f32>, !torch.float, !torch.bool -> !torch.vtensor<[1,197,768],f32>
    %339 = torch_c.to_builtin_tensor %338 : !torch.vtensor<[1,197,768],f32> -> tensor<1x197x768xf32>
    %340 = torch_c.to_builtin_tensor %291 : !torch.vtensor<[1,197,768],f32> -> tensor<1x197x768xf32>
    %341 = dnn.add %339, %340 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %342 = torch_c.from_builtin_tensor %341 : tensor<1x197x768xf32> -> !torch.vtensor<[1,197,768],f32>
    %343 = torch_c.to_builtin_tensor %342 : !torch.vtensor<[1,197,768],f32> -> tensor<1x197x768xf32>
    %344 = dnn.layer_norm %343, %cst_8, %cst_8 {parameter_indices = array<i32: 1, 4, 5>, parameters = [[768], 9.9999999999999995E-7, false]} : (tensor<1x197x768xf32>, tensor<768xf32>, tensor<768xf32>) -> tensor<1x197x768xf32>
    %345 = torch_c.from_builtin_tensor %344 : tensor<1x197x768xf32> -> !torch.vtensor<[1,197,768],f32>
    %346 = torch_c.to_builtin_tensor %345 : !torch.vtensor<[1,197,768],f32> -> tensor<1x197x768xf32>
    %347 = dnn.linear %346, %cst_3, %cst_2 : tensor<1x197x768xf32>, tensor<3072x768xf32>, tensor<3072xf32> -> tensor<1x197x3072xf32>
    %348 = torch_c.from_builtin_tensor %347 : tensor<1x197x3072xf32> -> !torch.vtensor<[1,197,3072],f32>
    %349 = torch_c.to_builtin_tensor %348 : !torch.vtensor<[1,197,3072],f32> -> tensor<1x197x3072xf32>
    %350 = dnn.gelu %349 {parameter_indices = array<i32: 1>, parameters = ["none"]} : (tensor<1x197x3072xf32>) -> tensor<1x197x3072xf32>
    %351 = torch_c.from_builtin_tensor %350 : tensor<1x197x3072xf32> -> !torch.vtensor<[1,197,3072],f32>
    %352 = torch.aten.dropout %351, %float0.000000e00, %false : !torch.vtensor<[1,197,3072],f32>, !torch.float, !torch.bool -> !torch.vtensor<[1,197,3072],f32>
    %353 = torch_c.to_builtin_tensor %352 : !torch.vtensor<[1,197,3072],f32> -> tensor<1x197x3072xf32>
    %354 = dnn.linear %353, %cst_1, %cst_8 : tensor<1x197x3072xf32>, tensor<768x3072xf32>, tensor<768xf32> -> tensor<1x197x768xf32>
    %355 = torch_c.from_builtin_tensor %354 : tensor<1x197x768xf32> -> !torch.vtensor<[1,197,768],f32>
    %356 = torch.aten.dropout %355, %float0.000000e00, %false : !torch.vtensor<[1,197,768],f32>, !torch.float, !torch.bool -> !torch.vtensor<[1,197,768],f32>
    %357 = torch_c.to_builtin_tensor %342 : !torch.vtensor<[1,197,768],f32> -> tensor<1x197x768xf32>
    %358 = torch_c.to_builtin_tensor %356 : !torch.vtensor<[1,197,768],f32> -> tensor<1x197x768xf32>
    %359 = dnn.add %357, %358 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %360 = torch_c.from_builtin_tensor %359 : tensor<1x197x768xf32> -> !torch.vtensor<[1,197,768],f32>
    %361 = torch_c.to_builtin_tensor %360 : !torch.vtensor<[1,197,768],f32> -> tensor<1x197x768xf32>
    %362 = dnn.layer_norm %361, %cst_8, %cst_8 {parameter_indices = array<i32: 1, 4, 5>, parameters = [[768], 9.9999999999999995E-7, false]} : (tensor<1x197x768xf32>, tensor<768xf32>, tensor<768xf32>) -> tensor<1x197x768xf32>
    %363 = torch_c.from_builtin_tensor %362 : tensor<1x197x768xf32> -> !torch.vtensor<[1,197,768],f32>
    %364 = torch.aten.transpose.int %363, %int1, %int0 : !torch.vtensor<[1,197,768],f32>, !torch.int, !torch.int -> !torch.vtensor<[197,1,768],f32>
    %365 = torch_c.to_builtin_tensor %364 : !torch.vtensor<[197,1,768],f32> -> tensor<197x1x768xf32>
    %366 = dnn.linear %365, %cst_6, %cst_5 : tensor<197x1x768xf32>, tensor<2304x768xf32>, tensor<2304xf32> -> tensor<197x1x2304xf32>
    %367 = torch_c.from_builtin_tensor %366 : tensor<197x1x2304xf32> -> !torch.vtensor<[197,1,2304],f32>
    %368 = torch.prim.ListConstruct %int3, %int768 : (!torch.int, !torch.int) -> !torch.list<int>
    %369 = torch.aten.unflatten.int %367, %int-1, %368 : !torch.vtensor<[197,1,2304],f32>, !torch.int, !torch.list<int> -> !torch.vtensor<[197,1,3,768],f32>
    %370 = torch.aten.unsqueeze %369, %int0 : !torch.vtensor<[197,1,3,768],f32>, !torch.int -> !torch.vtensor<[1,197,1,3,768],f32>
    %371 = torch.aten.transpose.int %370, %int0, %int-2 : !torch.vtensor<[1,197,1,3,768],f32>, !torch.int, !torch.int -> !torch.vtensor<[3,197,1,1,768],f32>
    %372 = torch.aten.squeeze.dim %371, %int-2 : !torch.vtensor<[3,197,1,1,768],f32>, !torch.int -> !torch.vtensor<[3,197,1,768],f32>
    %373 = torch.aten.contiguous %372, %int0 : !torch.vtensor<[3,197,1,768],f32>, !torch.int -> !torch.vtensor<[3,197,1,768],f32>
    %374 = torch.aten.select.int %373, %int0, %int0 : !torch.vtensor<[3,197,1,768],f32>, !torch.int, !torch.int -> !torch.vtensor<[197,1,768],f32>
    %375 = torch.aten.select.int %373, %int0, %int1 : !torch.vtensor<[3,197,1,768],f32>, !torch.int, !torch.int -> !torch.vtensor<[197,1,768],f32>
    %376 = torch.aten.select.int %373, %int0, %int2 : !torch.vtensor<[3,197,1,768],f32>, !torch.int, !torch.int -> !torch.vtensor<[197,1,768],f32>
    %377 = torch.prim.ListConstruct %int197, %int12, %int64 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %378 = torch.aten.view %374, %377 : !torch.vtensor<[197,1,768],f32>, !torch.list<int> -> !torch.vtensor<[197,12,64],f32>
    %379 = torch.aten.transpose.int %378, %int0, %int1 : !torch.vtensor<[197,12,64],f32>, !torch.int, !torch.int -> !torch.vtensor<[12,197,64],f32>
    %380 = torch.prim.ListConstruct %int197, %int12, %int64 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %381 = torch.aten.view %375, %380 : !torch.vtensor<[197,1,768],f32>, !torch.list<int> -> !torch.vtensor<[197,12,64],f32>
    %382 = torch.aten.transpose.int %381, %int0, %int1 : !torch.vtensor<[197,12,64],f32>, !torch.int, !torch.int -> !torch.vtensor<[12,197,64],f32>
    %383 = torch.prim.ListConstruct %int197, %int12, %int64 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %384 = torch.aten.view %376, %383 : !torch.vtensor<[197,1,768],f32>, !torch.list<int> -> !torch.vtensor<[197,12,64],f32>
    %385 = torch.aten.transpose.int %384, %int0, %int1 : !torch.vtensor<[197,12,64],f32>, !torch.int, !torch.int -> !torch.vtensor<[12,197,64],f32>
    %386 = torch.prim.ListConstruct %int1, %int12, %int197, %int64 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %387 = torch.aten.view %379, %386 : !torch.vtensor<[12,197,64],f32>, !torch.list<int> -> !torch.vtensor<[1,12,197,64],f32>
    %388 = torch.prim.ListConstruct %int1, %int12, %int197, %int64 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %389 = torch.aten.view %382, %388 : !torch.vtensor<[12,197,64],f32>, !torch.list<int> -> !torch.vtensor<[1,12,197,64],f32>
    %390 = torch.prim.ListConstruct %int1, %int12, %int197, %int64 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %391 = torch.aten.view %385, %390 : !torch.vtensor<[12,197,64],f32>, !torch.list<int> -> !torch.vtensor<[1,12,197,64],f32>
    %392 = torch_c.to_builtin_tensor %387 : !torch.vtensor<[1,12,197,64],f32> -> tensor<1x12x197x64xf32>
    %393 = torch_c.to_builtin_tensor %389 : !torch.vtensor<[1,12,197,64],f32> -> tensor<1x12x197x64xf32>
    %394 = torch_c.to_builtin_tensor %391 : !torch.vtensor<[1,12,197,64],f32> -> tensor<1x12x197x64xf32>
    %395 = dnn.scaled_dot_product_attention %392, %393, %394 {parameter_indices = array<i32: 3, 4, 5, 6, 7>, parameters = [unit, 0.000000e+00, false, unit, false]} : (tensor<1x12x197x64xf32>, tensor<1x12x197x64xf32>, tensor<1x12x197x64xf32>) -> tensor<1x12x197x64xf32>
    %396 = torch_c.from_builtin_tensor %395 : tensor<1x12x197x64xf32> -> !torch.vtensor<[1,12,197,64],f32>
    %397 = torch.prim.ListConstruct %int2, %int0, %int1, %int3 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %398 = torch.aten.permute %396, %397 : !torch.vtensor<[1,12,197,64],f32>, !torch.list<int> -> !torch.vtensor<[197,1,12,64],f32>
    %399 = torch.prim.ListConstruct %int197, %int768 : (!torch.int, !torch.int) -> !torch.list<int>
    %400 = torch.aten.view %398, %399 : !torch.vtensor<[197,1,12,64],f32>, !torch.list<int> -> !torch.vtensor<[197,768],f32>
    %401 = torch_c.to_builtin_tensor %400 : !torch.vtensor<[197,768],f32> -> tensor<197x768xf32>
    %402 = dnn.linear %401, %cst_4, %cst_8 : tensor<197x768xf32>, tensor<768x768xf32>, tensor<768xf32> -> tensor<197x768xf32>
    %403 = torch_c.from_builtin_tensor %402 : tensor<197x768xf32> -> !torch.vtensor<[197,768],f32>
    %404 = torch.prim.ListConstruct %int197, %int1, %int768 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %405 = torch.aten.view %403, %404 : !torch.vtensor<[197,768],f32>, !torch.list<int> -> !torch.vtensor<[197,1,768],f32>
    %406 = torch.aten.transpose.int %405, %int1, %int0 : !torch.vtensor<[197,1,768],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,197,768],f32>
    %407 = torch.aten.dropout %406, %float0.000000e00, %false : !torch.vtensor<[1,197,768],f32>, !torch.float, !torch.bool -> !torch.vtensor<[1,197,768],f32>
    %408 = torch_c.to_builtin_tensor %407 : !torch.vtensor<[1,197,768],f32> -> tensor<1x197x768xf32>
    %409 = torch_c.to_builtin_tensor %360 : !torch.vtensor<[1,197,768],f32> -> tensor<1x197x768xf32>
    %410 = dnn.add %408, %409 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %411 = torch_c.from_builtin_tensor %410 : tensor<1x197x768xf32> -> !torch.vtensor<[1,197,768],f32>
    %412 = torch_c.to_builtin_tensor %411 : !torch.vtensor<[1,197,768],f32> -> tensor<1x197x768xf32>
    %413 = dnn.layer_norm %412, %cst_8, %cst_8 {parameter_indices = array<i32: 1, 4, 5>, parameters = [[768], 9.9999999999999995E-7, false]} : (tensor<1x197x768xf32>, tensor<768xf32>, tensor<768xf32>) -> tensor<1x197x768xf32>
    %414 = torch_c.from_builtin_tensor %413 : tensor<1x197x768xf32> -> !torch.vtensor<[1,197,768],f32>
    %415 = torch_c.to_builtin_tensor %414 : !torch.vtensor<[1,197,768],f32> -> tensor<1x197x768xf32>
    %416 = dnn.linear %415, %cst_3, %cst_2 : tensor<1x197x768xf32>, tensor<3072x768xf32>, tensor<3072xf32> -> tensor<1x197x3072xf32>
    %417 = torch_c.from_builtin_tensor %416 : tensor<1x197x3072xf32> -> !torch.vtensor<[1,197,3072],f32>
    %418 = torch_c.to_builtin_tensor %417 : !torch.vtensor<[1,197,3072],f32> -> tensor<1x197x3072xf32>
    %419 = dnn.gelu %418 {parameter_indices = array<i32: 1>, parameters = ["none"]} : (tensor<1x197x3072xf32>) -> tensor<1x197x3072xf32>
    %420 = torch_c.from_builtin_tensor %419 : tensor<1x197x3072xf32> -> !torch.vtensor<[1,197,3072],f32>
    %421 = torch.aten.dropout %420, %float0.000000e00, %false : !torch.vtensor<[1,197,3072],f32>, !torch.float, !torch.bool -> !torch.vtensor<[1,197,3072],f32>
    %422 = torch_c.to_builtin_tensor %421 : !torch.vtensor<[1,197,3072],f32> -> tensor<1x197x3072xf32>
    %423 = dnn.linear %422, %cst_1, %cst_8 : tensor<1x197x3072xf32>, tensor<768x3072xf32>, tensor<768xf32> -> tensor<1x197x768xf32>
    %424 = torch_c.from_builtin_tensor %423 : tensor<1x197x768xf32> -> !torch.vtensor<[1,197,768],f32>
    %425 = torch.aten.dropout %424, %float0.000000e00, %false : !torch.vtensor<[1,197,768],f32>, !torch.float, !torch.bool -> !torch.vtensor<[1,197,768],f32>
    %426 = torch_c.to_builtin_tensor %411 : !torch.vtensor<[1,197,768],f32> -> tensor<1x197x768xf32>
    %427 = torch_c.to_builtin_tensor %425 : !torch.vtensor<[1,197,768],f32> -> tensor<1x197x768xf32>
    %428 = dnn.add %426, %427 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %429 = torch_c.from_builtin_tensor %428 : tensor<1x197x768xf32> -> !torch.vtensor<[1,197,768],f32>
    %430 = torch_c.to_builtin_tensor %429 : !torch.vtensor<[1,197,768],f32> -> tensor<1x197x768xf32>
    %431 = dnn.layer_norm %430, %cst_8, %cst_8 {parameter_indices = array<i32: 1, 4, 5>, parameters = [[768], 9.9999999999999995E-7, false]} : (tensor<1x197x768xf32>, tensor<768xf32>, tensor<768xf32>) -> tensor<1x197x768xf32>
    %432 = torch_c.from_builtin_tensor %431 : tensor<1x197x768xf32> -> !torch.vtensor<[1,197,768],f32>
    %433 = torch.aten.transpose.int %432, %int1, %int0 : !torch.vtensor<[1,197,768],f32>, !torch.int, !torch.int -> !torch.vtensor<[197,1,768],f32>
    %434 = torch_c.to_builtin_tensor %433 : !torch.vtensor<[197,1,768],f32> -> tensor<197x1x768xf32>
    %435 = dnn.linear %434, %cst_6, %cst_5 : tensor<197x1x768xf32>, tensor<2304x768xf32>, tensor<2304xf32> -> tensor<197x1x2304xf32>
    %436 = torch_c.from_builtin_tensor %435 : tensor<197x1x2304xf32> -> !torch.vtensor<[197,1,2304],f32>
    %437 = torch.prim.ListConstruct %int3, %int768 : (!torch.int, !torch.int) -> !torch.list<int>
    %438 = torch.aten.unflatten.int %436, %int-1, %437 : !torch.vtensor<[197,1,2304],f32>, !torch.int, !torch.list<int> -> !torch.vtensor<[197,1,3,768],f32>
    %439 = torch.aten.unsqueeze %438, %int0 : !torch.vtensor<[197,1,3,768],f32>, !torch.int -> !torch.vtensor<[1,197,1,3,768],f32>
    %440 = torch.aten.transpose.int %439, %int0, %int-2 : !torch.vtensor<[1,197,1,3,768],f32>, !torch.int, !torch.int -> !torch.vtensor<[3,197,1,1,768],f32>
    %441 = torch.aten.squeeze.dim %440, %int-2 : !torch.vtensor<[3,197,1,1,768],f32>, !torch.int -> !torch.vtensor<[3,197,1,768],f32>
    %442 = torch.aten.contiguous %441, %int0 : !torch.vtensor<[3,197,1,768],f32>, !torch.int -> !torch.vtensor<[3,197,1,768],f32>
    %443 = torch.aten.select.int %442, %int0, %int0 : !torch.vtensor<[3,197,1,768],f32>, !torch.int, !torch.int -> !torch.vtensor<[197,1,768],f32>
    %444 = torch.aten.select.int %442, %int0, %int1 : !torch.vtensor<[3,197,1,768],f32>, !torch.int, !torch.int -> !torch.vtensor<[197,1,768],f32>
    %445 = torch.aten.select.int %442, %int0, %int2 : !torch.vtensor<[3,197,1,768],f32>, !torch.int, !torch.int -> !torch.vtensor<[197,1,768],f32>
    %446 = torch.prim.ListConstruct %int197, %int12, %int64 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %447 = torch.aten.view %443, %446 : !torch.vtensor<[197,1,768],f32>, !torch.list<int> -> !torch.vtensor<[197,12,64],f32>
    %448 = torch.aten.transpose.int %447, %int0, %int1 : !torch.vtensor<[197,12,64],f32>, !torch.int, !torch.int -> !torch.vtensor<[12,197,64],f32>
    %449 = torch.prim.ListConstruct %int197, %int12, %int64 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %450 = torch.aten.view %444, %449 : !torch.vtensor<[197,1,768],f32>, !torch.list<int> -> !torch.vtensor<[197,12,64],f32>
    %451 = torch.aten.transpose.int %450, %int0, %int1 : !torch.vtensor<[197,12,64],f32>, !torch.int, !torch.int -> !torch.vtensor<[12,197,64],f32>
    %452 = torch.prim.ListConstruct %int197, %int12, %int64 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %453 = torch.aten.view %445, %452 : !torch.vtensor<[197,1,768],f32>, !torch.list<int> -> !torch.vtensor<[197,12,64],f32>
    %454 = torch.aten.transpose.int %453, %int0, %int1 : !torch.vtensor<[197,12,64],f32>, !torch.int, !torch.int -> !torch.vtensor<[12,197,64],f32>
    %455 = torch.prim.ListConstruct %int1, %int12, %int197, %int64 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %456 = torch.aten.view %448, %455 : !torch.vtensor<[12,197,64],f32>, !torch.list<int> -> !torch.vtensor<[1,12,197,64],f32>
    %457 = torch.prim.ListConstruct %int1, %int12, %int197, %int64 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %458 = torch.aten.view %451, %457 : !torch.vtensor<[12,197,64],f32>, !torch.list<int> -> !torch.vtensor<[1,12,197,64],f32>
    %459 = torch.prim.ListConstruct %int1, %int12, %int197, %int64 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %460 = torch.aten.view %454, %459 : !torch.vtensor<[12,197,64],f32>, !torch.list<int> -> !torch.vtensor<[1,12,197,64],f32>
    %461 = torch_c.to_builtin_tensor %456 : !torch.vtensor<[1,12,197,64],f32> -> tensor<1x12x197x64xf32>
    %462 = torch_c.to_builtin_tensor %458 : !torch.vtensor<[1,12,197,64],f32> -> tensor<1x12x197x64xf32>
    %463 = torch_c.to_builtin_tensor %460 : !torch.vtensor<[1,12,197,64],f32> -> tensor<1x12x197x64xf32>
    %464 = dnn.scaled_dot_product_attention %461, %462, %463 {parameter_indices = array<i32: 3, 4, 5, 6, 7>, parameters = [unit, 0.000000e+00, false, unit, false]} : (tensor<1x12x197x64xf32>, tensor<1x12x197x64xf32>, tensor<1x12x197x64xf32>) -> tensor<1x12x197x64xf32>
    %465 = torch_c.from_builtin_tensor %464 : tensor<1x12x197x64xf32> -> !torch.vtensor<[1,12,197,64],f32>
    %466 = torch.prim.ListConstruct %int2, %int0, %int1, %int3 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %467 = torch.aten.permute %465, %466 : !torch.vtensor<[1,12,197,64],f32>, !torch.list<int> -> !torch.vtensor<[197,1,12,64],f32>
    %468 = torch.prim.ListConstruct %int197, %int768 : (!torch.int, !torch.int) -> !torch.list<int>
    %469 = torch.aten.view %467, %468 : !torch.vtensor<[197,1,12,64],f32>, !torch.list<int> -> !torch.vtensor<[197,768],f32>
    %470 = torch_c.to_builtin_tensor %469 : !torch.vtensor<[197,768],f32> -> tensor<197x768xf32>
    %471 = dnn.linear %470, %cst_4, %cst_8 : tensor<197x768xf32>, tensor<768x768xf32>, tensor<768xf32> -> tensor<197x768xf32>
    %472 = torch_c.from_builtin_tensor %471 : tensor<197x768xf32> -> !torch.vtensor<[197,768],f32>
    %473 = torch.prim.ListConstruct %int197, %int1, %int768 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %474 = torch.aten.view %472, %473 : !torch.vtensor<[197,768],f32>, !torch.list<int> -> !torch.vtensor<[197,1,768],f32>
    %475 = torch.aten.transpose.int %474, %int1, %int0 : !torch.vtensor<[197,1,768],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,197,768],f32>
    %476 = torch.aten.dropout %475, %float0.000000e00, %false : !torch.vtensor<[1,197,768],f32>, !torch.float, !torch.bool -> !torch.vtensor<[1,197,768],f32>
    %477 = torch_c.to_builtin_tensor %476 : !torch.vtensor<[1,197,768],f32> -> tensor<1x197x768xf32>
    %478 = torch_c.to_builtin_tensor %429 : !torch.vtensor<[1,197,768],f32> -> tensor<1x197x768xf32>
    %479 = dnn.add %477, %478 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %480 = torch_c.from_builtin_tensor %479 : tensor<1x197x768xf32> -> !torch.vtensor<[1,197,768],f32>
    %481 = torch_c.to_builtin_tensor %480 : !torch.vtensor<[1,197,768],f32> -> tensor<1x197x768xf32>
    %482 = dnn.layer_norm %481, %cst_8, %cst_8 {parameter_indices = array<i32: 1, 4, 5>, parameters = [[768], 9.9999999999999995E-7, false]} : (tensor<1x197x768xf32>, tensor<768xf32>, tensor<768xf32>) -> tensor<1x197x768xf32>
    %483 = torch_c.from_builtin_tensor %482 : tensor<1x197x768xf32> -> !torch.vtensor<[1,197,768],f32>
    %484 = torch_c.to_builtin_tensor %483 : !torch.vtensor<[1,197,768],f32> -> tensor<1x197x768xf32>
    %485 = dnn.linear %484, %cst_3, %cst_2 : tensor<1x197x768xf32>, tensor<3072x768xf32>, tensor<3072xf32> -> tensor<1x197x3072xf32>
    %486 = torch_c.from_builtin_tensor %485 : tensor<1x197x3072xf32> -> !torch.vtensor<[1,197,3072],f32>
    %487 = torch_c.to_builtin_tensor %486 : !torch.vtensor<[1,197,3072],f32> -> tensor<1x197x3072xf32>
    %488 = dnn.gelu %487 {parameter_indices = array<i32: 1>, parameters = ["none"]} : (tensor<1x197x3072xf32>) -> tensor<1x197x3072xf32>
    %489 = torch_c.from_builtin_tensor %488 : tensor<1x197x3072xf32> -> !torch.vtensor<[1,197,3072],f32>
    %490 = torch.aten.dropout %489, %float0.000000e00, %false : !torch.vtensor<[1,197,3072],f32>, !torch.float, !torch.bool -> !torch.vtensor<[1,197,3072],f32>
    %491 = torch_c.to_builtin_tensor %490 : !torch.vtensor<[1,197,3072],f32> -> tensor<1x197x3072xf32>
    %492 = dnn.linear %491, %cst_1, %cst_8 : tensor<1x197x3072xf32>, tensor<768x3072xf32>, tensor<768xf32> -> tensor<1x197x768xf32>
    %493 = torch_c.from_builtin_tensor %492 : tensor<1x197x768xf32> -> !torch.vtensor<[1,197,768],f32>
    %494 = torch.aten.dropout %493, %float0.000000e00, %false : !torch.vtensor<[1,197,768],f32>, !torch.float, !torch.bool -> !torch.vtensor<[1,197,768],f32>
    %495 = torch_c.to_builtin_tensor %480 : !torch.vtensor<[1,197,768],f32> -> tensor<1x197x768xf32>
    %496 = torch_c.to_builtin_tensor %494 : !torch.vtensor<[1,197,768],f32> -> tensor<1x197x768xf32>
    %497 = dnn.add %495, %496 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %498 = torch_c.from_builtin_tensor %497 : tensor<1x197x768xf32> -> !torch.vtensor<[1,197,768],f32>
    %499 = torch_c.to_builtin_tensor %498 : !torch.vtensor<[1,197,768],f32> -> tensor<1x197x768xf32>
    %500 = dnn.layer_norm %499, %cst_8, %cst_8 {parameter_indices = array<i32: 1, 4, 5>, parameters = [[768], 9.9999999999999995E-7, false]} : (tensor<1x197x768xf32>, tensor<768xf32>, tensor<768xf32>) -> tensor<1x197x768xf32>
    %501 = torch_c.from_builtin_tensor %500 : tensor<1x197x768xf32> -> !torch.vtensor<[1,197,768],f32>
    %502 = torch.aten.transpose.int %501, %int1, %int0 : !torch.vtensor<[1,197,768],f32>, !torch.int, !torch.int -> !torch.vtensor<[197,1,768],f32>
    %503 = torch_c.to_builtin_tensor %502 : !torch.vtensor<[197,1,768],f32> -> tensor<197x1x768xf32>
    %504 = dnn.linear %503, %cst_6, %cst_5 : tensor<197x1x768xf32>, tensor<2304x768xf32>, tensor<2304xf32> -> tensor<197x1x2304xf32>
    %505 = torch_c.from_builtin_tensor %504 : tensor<197x1x2304xf32> -> !torch.vtensor<[197,1,2304],f32>
    %506 = torch.prim.ListConstruct %int3, %int768 : (!torch.int, !torch.int) -> !torch.list<int>
    %507 = torch.aten.unflatten.int %505, %int-1, %506 : !torch.vtensor<[197,1,2304],f32>, !torch.int, !torch.list<int> -> !torch.vtensor<[197,1,3,768],f32>
    %508 = torch.aten.unsqueeze %507, %int0 : !torch.vtensor<[197,1,3,768],f32>, !torch.int -> !torch.vtensor<[1,197,1,3,768],f32>
    %509 = torch.aten.transpose.int %508, %int0, %int-2 : !torch.vtensor<[1,197,1,3,768],f32>, !torch.int, !torch.int -> !torch.vtensor<[3,197,1,1,768],f32>
    %510 = torch.aten.squeeze.dim %509, %int-2 : !torch.vtensor<[3,197,1,1,768],f32>, !torch.int -> !torch.vtensor<[3,197,1,768],f32>
    %511 = torch.aten.contiguous %510, %int0 : !torch.vtensor<[3,197,1,768],f32>, !torch.int -> !torch.vtensor<[3,197,1,768],f32>
    %512 = torch.aten.select.int %511, %int0, %int0 : !torch.vtensor<[3,197,1,768],f32>, !torch.int, !torch.int -> !torch.vtensor<[197,1,768],f32>
    %513 = torch.aten.select.int %511, %int0, %int1 : !torch.vtensor<[3,197,1,768],f32>, !torch.int, !torch.int -> !torch.vtensor<[197,1,768],f32>
    %514 = torch.aten.select.int %511, %int0, %int2 : !torch.vtensor<[3,197,1,768],f32>, !torch.int, !torch.int -> !torch.vtensor<[197,1,768],f32>
    %515 = torch.prim.ListConstruct %int197, %int12, %int64 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %516 = torch.aten.view %512, %515 : !torch.vtensor<[197,1,768],f32>, !torch.list<int> -> !torch.vtensor<[197,12,64],f32>
    %517 = torch.aten.transpose.int %516, %int0, %int1 : !torch.vtensor<[197,12,64],f32>, !torch.int, !torch.int -> !torch.vtensor<[12,197,64],f32>
    %518 = torch.prim.ListConstruct %int197, %int12, %int64 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %519 = torch.aten.view %513, %518 : !torch.vtensor<[197,1,768],f32>, !torch.list<int> -> !torch.vtensor<[197,12,64],f32>
    %520 = torch.aten.transpose.int %519, %int0, %int1 : !torch.vtensor<[197,12,64],f32>, !torch.int, !torch.int -> !torch.vtensor<[12,197,64],f32>
    %521 = torch.prim.ListConstruct %int197, %int12, %int64 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %522 = torch.aten.view %514, %521 : !torch.vtensor<[197,1,768],f32>, !torch.list<int> -> !torch.vtensor<[197,12,64],f32>
    %523 = torch.aten.transpose.int %522, %int0, %int1 : !torch.vtensor<[197,12,64],f32>, !torch.int, !torch.int -> !torch.vtensor<[12,197,64],f32>
    %524 = torch.prim.ListConstruct %int1, %int12, %int197, %int64 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %525 = torch.aten.view %517, %524 : !torch.vtensor<[12,197,64],f32>, !torch.list<int> -> !torch.vtensor<[1,12,197,64],f32>
    %526 = torch.prim.ListConstruct %int1, %int12, %int197, %int64 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %527 = torch.aten.view %520, %526 : !torch.vtensor<[12,197,64],f32>, !torch.list<int> -> !torch.vtensor<[1,12,197,64],f32>
    %528 = torch.prim.ListConstruct %int1, %int12, %int197, %int64 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %529 = torch.aten.view %523, %528 : !torch.vtensor<[12,197,64],f32>, !torch.list<int> -> !torch.vtensor<[1,12,197,64],f32>
    %530 = torch_c.to_builtin_tensor %525 : !torch.vtensor<[1,12,197,64],f32> -> tensor<1x12x197x64xf32>
    %531 = torch_c.to_builtin_tensor %527 : !torch.vtensor<[1,12,197,64],f32> -> tensor<1x12x197x64xf32>
    %532 = torch_c.to_builtin_tensor %529 : !torch.vtensor<[1,12,197,64],f32> -> tensor<1x12x197x64xf32>
    %533 = dnn.scaled_dot_product_attention %530, %531, %532 {parameter_indices = array<i32: 3, 4, 5, 6, 7>, parameters = [unit, 0.000000e+00, false, unit, false]} : (tensor<1x12x197x64xf32>, tensor<1x12x197x64xf32>, tensor<1x12x197x64xf32>) -> tensor<1x12x197x64xf32>
    %534 = torch_c.from_builtin_tensor %533 : tensor<1x12x197x64xf32> -> !torch.vtensor<[1,12,197,64],f32>
    %535 = torch.prim.ListConstruct %int2, %int0, %int1, %int3 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %536 = torch.aten.permute %534, %535 : !torch.vtensor<[1,12,197,64],f32>, !torch.list<int> -> !torch.vtensor<[197,1,12,64],f32>
    %537 = torch.prim.ListConstruct %int197, %int768 : (!torch.int, !torch.int) -> !torch.list<int>
    %538 = torch.aten.view %536, %537 : !torch.vtensor<[197,1,12,64],f32>, !torch.list<int> -> !torch.vtensor<[197,768],f32>
    %539 = torch_c.to_builtin_tensor %538 : !torch.vtensor<[197,768],f32> -> tensor<197x768xf32>
    %540 = dnn.linear %539, %cst_4, %cst_8 : tensor<197x768xf32>, tensor<768x768xf32>, tensor<768xf32> -> tensor<197x768xf32>
    %541 = torch_c.from_builtin_tensor %540 : tensor<197x768xf32> -> !torch.vtensor<[197,768],f32>
    %542 = torch.prim.ListConstruct %int197, %int1, %int768 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %543 = torch.aten.view %541, %542 : !torch.vtensor<[197,768],f32>, !torch.list<int> -> !torch.vtensor<[197,1,768],f32>
    %544 = torch.aten.transpose.int %543, %int1, %int0 : !torch.vtensor<[197,1,768],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,197,768],f32>
    %545 = torch.aten.dropout %544, %float0.000000e00, %false : !torch.vtensor<[1,197,768],f32>, !torch.float, !torch.bool -> !torch.vtensor<[1,197,768],f32>
    %546 = torch_c.to_builtin_tensor %545 : !torch.vtensor<[1,197,768],f32> -> tensor<1x197x768xf32>
    %547 = torch_c.to_builtin_tensor %498 : !torch.vtensor<[1,197,768],f32> -> tensor<1x197x768xf32>
    %548 = dnn.add %546, %547 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %549 = torch_c.from_builtin_tensor %548 : tensor<1x197x768xf32> -> !torch.vtensor<[1,197,768],f32>
    %550 = torch_c.to_builtin_tensor %549 : !torch.vtensor<[1,197,768],f32> -> tensor<1x197x768xf32>
    %551 = dnn.layer_norm %550, %cst_8, %cst_8 {parameter_indices = array<i32: 1, 4, 5>, parameters = [[768], 9.9999999999999995E-7, false]} : (tensor<1x197x768xf32>, tensor<768xf32>, tensor<768xf32>) -> tensor<1x197x768xf32>
    %552 = torch_c.from_builtin_tensor %551 : tensor<1x197x768xf32> -> !torch.vtensor<[1,197,768],f32>
    %553 = torch_c.to_builtin_tensor %552 : !torch.vtensor<[1,197,768],f32> -> tensor<1x197x768xf32>
    %554 = dnn.linear %553, %cst_3, %cst_2 : tensor<1x197x768xf32>, tensor<3072x768xf32>, tensor<3072xf32> -> tensor<1x197x3072xf32>
    %555 = torch_c.from_builtin_tensor %554 : tensor<1x197x3072xf32> -> !torch.vtensor<[1,197,3072],f32>
    %556 = torch_c.to_builtin_tensor %555 : !torch.vtensor<[1,197,3072],f32> -> tensor<1x197x3072xf32>
    %557 = dnn.gelu %556 {parameter_indices = array<i32: 1>, parameters = ["none"]} : (tensor<1x197x3072xf32>) -> tensor<1x197x3072xf32>
    %558 = torch_c.from_builtin_tensor %557 : tensor<1x197x3072xf32> -> !torch.vtensor<[1,197,3072],f32>
    %559 = torch.aten.dropout %558, %float0.000000e00, %false : !torch.vtensor<[1,197,3072],f32>, !torch.float, !torch.bool -> !torch.vtensor<[1,197,3072],f32>
    %560 = torch_c.to_builtin_tensor %559 : !torch.vtensor<[1,197,3072],f32> -> tensor<1x197x3072xf32>
    %561 = dnn.linear %560, %cst_1, %cst_8 : tensor<1x197x3072xf32>, tensor<768x3072xf32>, tensor<768xf32> -> tensor<1x197x768xf32>
    %562 = torch_c.from_builtin_tensor %561 : tensor<1x197x768xf32> -> !torch.vtensor<[1,197,768],f32>
    %563 = torch.aten.dropout %562, %float0.000000e00, %false : !torch.vtensor<[1,197,768],f32>, !torch.float, !torch.bool -> !torch.vtensor<[1,197,768],f32>
    %564 = torch_c.to_builtin_tensor %549 : !torch.vtensor<[1,197,768],f32> -> tensor<1x197x768xf32>
    %565 = torch_c.to_builtin_tensor %563 : !torch.vtensor<[1,197,768],f32> -> tensor<1x197x768xf32>
    %566 = dnn.add %564, %565 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %567 = torch_c.from_builtin_tensor %566 : tensor<1x197x768xf32> -> !torch.vtensor<[1,197,768],f32>
    %568 = torch_c.to_builtin_tensor %567 : !torch.vtensor<[1,197,768],f32> -> tensor<1x197x768xf32>
    %569 = dnn.layer_norm %568, %cst_8, %cst_8 {parameter_indices = array<i32: 1, 4, 5>, parameters = [[768], 9.9999999999999995E-7, false]} : (tensor<1x197x768xf32>, tensor<768xf32>, tensor<768xf32>) -> tensor<1x197x768xf32>
    %570 = torch_c.from_builtin_tensor %569 : tensor<1x197x768xf32> -> !torch.vtensor<[1,197,768],f32>
    %571 = torch.aten.transpose.int %570, %int1, %int0 : !torch.vtensor<[1,197,768],f32>, !torch.int, !torch.int -> !torch.vtensor<[197,1,768],f32>
    %572 = torch_c.to_builtin_tensor %571 : !torch.vtensor<[197,1,768],f32> -> tensor<197x1x768xf32>
    %573 = dnn.linear %572, %cst_6, %cst_5 : tensor<197x1x768xf32>, tensor<2304x768xf32>, tensor<2304xf32> -> tensor<197x1x2304xf32>
    %574 = torch_c.from_builtin_tensor %573 : tensor<197x1x2304xf32> -> !torch.vtensor<[197,1,2304],f32>
    %575 = torch.prim.ListConstruct %int3, %int768 : (!torch.int, !torch.int) -> !torch.list<int>
    %576 = torch.aten.unflatten.int %574, %int-1, %575 : !torch.vtensor<[197,1,2304],f32>, !torch.int, !torch.list<int> -> !torch.vtensor<[197,1,3,768],f32>
    %577 = torch.aten.unsqueeze %576, %int0 : !torch.vtensor<[197,1,3,768],f32>, !torch.int -> !torch.vtensor<[1,197,1,3,768],f32>
    %578 = torch.aten.transpose.int %577, %int0, %int-2 : !torch.vtensor<[1,197,1,3,768],f32>, !torch.int, !torch.int -> !torch.vtensor<[3,197,1,1,768],f32>
    %579 = torch.aten.squeeze.dim %578, %int-2 : !torch.vtensor<[3,197,1,1,768],f32>, !torch.int -> !torch.vtensor<[3,197,1,768],f32>
    %580 = torch.aten.contiguous %579, %int0 : !torch.vtensor<[3,197,1,768],f32>, !torch.int -> !torch.vtensor<[3,197,1,768],f32>
    %581 = torch.aten.select.int %580, %int0, %int0 : !torch.vtensor<[3,197,1,768],f32>, !torch.int, !torch.int -> !torch.vtensor<[197,1,768],f32>
    %582 = torch.aten.select.int %580, %int0, %int1 : !torch.vtensor<[3,197,1,768],f32>, !torch.int, !torch.int -> !torch.vtensor<[197,1,768],f32>
    %583 = torch.aten.select.int %580, %int0, %int2 : !torch.vtensor<[3,197,1,768],f32>, !torch.int, !torch.int -> !torch.vtensor<[197,1,768],f32>
    %584 = torch.prim.ListConstruct %int197, %int12, %int64 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %585 = torch.aten.view %581, %584 : !torch.vtensor<[197,1,768],f32>, !torch.list<int> -> !torch.vtensor<[197,12,64],f32>
    %586 = torch.aten.transpose.int %585, %int0, %int1 : !torch.vtensor<[197,12,64],f32>, !torch.int, !torch.int -> !torch.vtensor<[12,197,64],f32>
    %587 = torch.prim.ListConstruct %int197, %int12, %int64 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %588 = torch.aten.view %582, %587 : !torch.vtensor<[197,1,768],f32>, !torch.list<int> -> !torch.vtensor<[197,12,64],f32>
    %589 = torch.aten.transpose.int %588, %int0, %int1 : !torch.vtensor<[197,12,64],f32>, !torch.int, !torch.int -> !torch.vtensor<[12,197,64],f32>
    %590 = torch.prim.ListConstruct %int197, %int12, %int64 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %591 = torch.aten.view %583, %590 : !torch.vtensor<[197,1,768],f32>, !torch.list<int> -> !torch.vtensor<[197,12,64],f32>
    %592 = torch.aten.transpose.int %591, %int0, %int1 : !torch.vtensor<[197,12,64],f32>, !torch.int, !torch.int -> !torch.vtensor<[12,197,64],f32>
    %593 = torch.prim.ListConstruct %int1, %int12, %int197, %int64 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %594 = torch.aten.view %586, %593 : !torch.vtensor<[12,197,64],f32>, !torch.list<int> -> !torch.vtensor<[1,12,197,64],f32>
    %595 = torch.prim.ListConstruct %int1, %int12, %int197, %int64 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %596 = torch.aten.view %589, %595 : !torch.vtensor<[12,197,64],f32>, !torch.list<int> -> !torch.vtensor<[1,12,197,64],f32>
    %597 = torch.prim.ListConstruct %int1, %int12, %int197, %int64 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %598 = torch.aten.view %592, %597 : !torch.vtensor<[12,197,64],f32>, !torch.list<int> -> !torch.vtensor<[1,12,197,64],f32>
    %599 = torch_c.to_builtin_tensor %594 : !torch.vtensor<[1,12,197,64],f32> -> tensor<1x12x197x64xf32>
    %600 = torch_c.to_builtin_tensor %596 : !torch.vtensor<[1,12,197,64],f32> -> tensor<1x12x197x64xf32>
    %601 = torch_c.to_builtin_tensor %598 : !torch.vtensor<[1,12,197,64],f32> -> tensor<1x12x197x64xf32>
    %602 = dnn.scaled_dot_product_attention %599, %600, %601 {parameter_indices = array<i32: 3, 4, 5, 6, 7>, parameters = [unit, 0.000000e+00, false, unit, false]} : (tensor<1x12x197x64xf32>, tensor<1x12x197x64xf32>, tensor<1x12x197x64xf32>) -> tensor<1x12x197x64xf32>
    %603 = torch_c.from_builtin_tensor %602 : tensor<1x12x197x64xf32> -> !torch.vtensor<[1,12,197,64],f32>
    %604 = torch.prim.ListConstruct %int2, %int0, %int1, %int3 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %605 = torch.aten.permute %603, %604 : !torch.vtensor<[1,12,197,64],f32>, !torch.list<int> -> !torch.vtensor<[197,1,12,64],f32>
    %606 = torch.prim.ListConstruct %int197, %int768 : (!torch.int, !torch.int) -> !torch.list<int>
    %607 = torch.aten.view %605, %606 : !torch.vtensor<[197,1,12,64],f32>, !torch.list<int> -> !torch.vtensor<[197,768],f32>
    %608 = torch_c.to_builtin_tensor %607 : !torch.vtensor<[197,768],f32> -> tensor<197x768xf32>
    %609 = dnn.linear %608, %cst_4, %cst_8 : tensor<197x768xf32>, tensor<768x768xf32>, tensor<768xf32> -> tensor<197x768xf32>
    %610 = torch_c.from_builtin_tensor %609 : tensor<197x768xf32> -> !torch.vtensor<[197,768],f32>
    %611 = torch.prim.ListConstruct %int197, %int1, %int768 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %612 = torch.aten.view %610, %611 : !torch.vtensor<[197,768],f32>, !torch.list<int> -> !torch.vtensor<[197,1,768],f32>
    %613 = torch.aten.transpose.int %612, %int1, %int0 : !torch.vtensor<[197,1,768],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,197,768],f32>
    %614 = torch.aten.dropout %613, %float0.000000e00, %false : !torch.vtensor<[1,197,768],f32>, !torch.float, !torch.bool -> !torch.vtensor<[1,197,768],f32>
    %615 = torch_c.to_builtin_tensor %614 : !torch.vtensor<[1,197,768],f32> -> tensor<1x197x768xf32>
    %616 = torch_c.to_builtin_tensor %567 : !torch.vtensor<[1,197,768],f32> -> tensor<1x197x768xf32>
    %617 = dnn.add %615, %616 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %618 = torch_c.from_builtin_tensor %617 : tensor<1x197x768xf32> -> !torch.vtensor<[1,197,768],f32>
    %619 = torch_c.to_builtin_tensor %618 : !torch.vtensor<[1,197,768],f32> -> tensor<1x197x768xf32>
    %620 = dnn.layer_norm %619, %cst_8, %cst_8 {parameter_indices = array<i32: 1, 4, 5>, parameters = [[768], 9.9999999999999995E-7, false]} : (tensor<1x197x768xf32>, tensor<768xf32>, tensor<768xf32>) -> tensor<1x197x768xf32>
    %621 = torch_c.from_builtin_tensor %620 : tensor<1x197x768xf32> -> !torch.vtensor<[1,197,768],f32>
    %622 = torch_c.to_builtin_tensor %621 : !torch.vtensor<[1,197,768],f32> -> tensor<1x197x768xf32>
    %623 = dnn.linear %622, %cst_3, %cst_2 : tensor<1x197x768xf32>, tensor<3072x768xf32>, tensor<3072xf32> -> tensor<1x197x3072xf32>
    %624 = torch_c.from_builtin_tensor %623 : tensor<1x197x3072xf32> -> !torch.vtensor<[1,197,3072],f32>
    %625 = torch_c.to_builtin_tensor %624 : !torch.vtensor<[1,197,3072],f32> -> tensor<1x197x3072xf32>
    %626 = dnn.gelu %625 {parameter_indices = array<i32: 1>, parameters = ["none"]} : (tensor<1x197x3072xf32>) -> tensor<1x197x3072xf32>
    %627 = torch_c.from_builtin_tensor %626 : tensor<1x197x3072xf32> -> !torch.vtensor<[1,197,3072],f32>
    %628 = torch.aten.dropout %627, %float0.000000e00, %false : !torch.vtensor<[1,197,3072],f32>, !torch.float, !torch.bool -> !torch.vtensor<[1,197,3072],f32>
    %629 = torch_c.to_builtin_tensor %628 : !torch.vtensor<[1,197,3072],f32> -> tensor<1x197x3072xf32>
    %630 = dnn.linear %629, %cst_1, %cst_8 : tensor<1x197x3072xf32>, tensor<768x3072xf32>, tensor<768xf32> -> tensor<1x197x768xf32>
    %631 = torch_c.from_builtin_tensor %630 : tensor<1x197x768xf32> -> !torch.vtensor<[1,197,768],f32>
    %632 = torch.aten.dropout %631, %float0.000000e00, %false : !torch.vtensor<[1,197,768],f32>, !torch.float, !torch.bool -> !torch.vtensor<[1,197,768],f32>
    %633 = torch_c.to_builtin_tensor %618 : !torch.vtensor<[1,197,768],f32> -> tensor<1x197x768xf32>
    %634 = torch_c.to_builtin_tensor %632 : !torch.vtensor<[1,197,768],f32> -> tensor<1x197x768xf32>
    %635 = dnn.add %633, %634 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %636 = torch_c.from_builtin_tensor %635 : tensor<1x197x768xf32> -> !torch.vtensor<[1,197,768],f32>
    %637 = torch_c.to_builtin_tensor %636 : !torch.vtensor<[1,197,768],f32> -> tensor<1x197x768xf32>
    %638 = dnn.layer_norm %637, %cst_8, %cst_8 {parameter_indices = array<i32: 1, 4, 5>, parameters = [[768], 9.9999999999999995E-7, false]} : (tensor<1x197x768xf32>, tensor<768xf32>, tensor<768xf32>) -> tensor<1x197x768xf32>
    %639 = torch_c.from_builtin_tensor %638 : tensor<1x197x768xf32> -> !torch.vtensor<[1,197,768],f32>
    %640 = torch.aten.transpose.int %639, %int1, %int0 : !torch.vtensor<[1,197,768],f32>, !torch.int, !torch.int -> !torch.vtensor<[197,1,768],f32>
    %641 = torch_c.to_builtin_tensor %640 : !torch.vtensor<[197,1,768],f32> -> tensor<197x1x768xf32>
    %642 = dnn.linear %641, %cst_6, %cst_5 : tensor<197x1x768xf32>, tensor<2304x768xf32>, tensor<2304xf32> -> tensor<197x1x2304xf32>
    %643 = torch_c.from_builtin_tensor %642 : tensor<197x1x2304xf32> -> !torch.vtensor<[197,1,2304],f32>
    %644 = torch.prim.ListConstruct %int3, %int768 : (!torch.int, !torch.int) -> !torch.list<int>
    %645 = torch.aten.unflatten.int %643, %int-1, %644 : !torch.vtensor<[197,1,2304],f32>, !torch.int, !torch.list<int> -> !torch.vtensor<[197,1,3,768],f32>
    %646 = torch.aten.unsqueeze %645, %int0 : !torch.vtensor<[197,1,3,768],f32>, !torch.int -> !torch.vtensor<[1,197,1,3,768],f32>
    %647 = torch.aten.transpose.int %646, %int0, %int-2 : !torch.vtensor<[1,197,1,3,768],f32>, !torch.int, !torch.int -> !torch.vtensor<[3,197,1,1,768],f32>
    %648 = torch.aten.squeeze.dim %647, %int-2 : !torch.vtensor<[3,197,1,1,768],f32>, !torch.int -> !torch.vtensor<[3,197,1,768],f32>
    %649 = torch.aten.contiguous %648, %int0 : !torch.vtensor<[3,197,1,768],f32>, !torch.int -> !torch.vtensor<[3,197,1,768],f32>
    %650 = torch.aten.select.int %649, %int0, %int0 : !torch.vtensor<[3,197,1,768],f32>, !torch.int, !torch.int -> !torch.vtensor<[197,1,768],f32>
    %651 = torch.aten.select.int %649, %int0, %int1 : !torch.vtensor<[3,197,1,768],f32>, !torch.int, !torch.int -> !torch.vtensor<[197,1,768],f32>
    %652 = torch.aten.select.int %649, %int0, %int2 : !torch.vtensor<[3,197,1,768],f32>, !torch.int, !torch.int -> !torch.vtensor<[197,1,768],f32>
    %653 = torch.prim.ListConstruct %int197, %int12, %int64 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %654 = torch.aten.view %650, %653 : !torch.vtensor<[197,1,768],f32>, !torch.list<int> -> !torch.vtensor<[197,12,64],f32>
    %655 = torch.aten.transpose.int %654, %int0, %int1 : !torch.vtensor<[197,12,64],f32>, !torch.int, !torch.int -> !torch.vtensor<[12,197,64],f32>
    %656 = torch.prim.ListConstruct %int197, %int12, %int64 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %657 = torch.aten.view %651, %656 : !torch.vtensor<[197,1,768],f32>, !torch.list<int> -> !torch.vtensor<[197,12,64],f32>
    %658 = torch.aten.transpose.int %657, %int0, %int1 : !torch.vtensor<[197,12,64],f32>, !torch.int, !torch.int -> !torch.vtensor<[12,197,64],f32>
    %659 = torch.prim.ListConstruct %int197, %int12, %int64 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %660 = torch.aten.view %652, %659 : !torch.vtensor<[197,1,768],f32>, !torch.list<int> -> !torch.vtensor<[197,12,64],f32>
    %661 = torch.aten.transpose.int %660, %int0, %int1 : !torch.vtensor<[197,12,64],f32>, !torch.int, !torch.int -> !torch.vtensor<[12,197,64],f32>
    %662 = torch.prim.ListConstruct %int1, %int12, %int197, %int64 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %663 = torch.aten.view %655, %662 : !torch.vtensor<[12,197,64],f32>, !torch.list<int> -> !torch.vtensor<[1,12,197,64],f32>
    %664 = torch.prim.ListConstruct %int1, %int12, %int197, %int64 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %665 = torch.aten.view %658, %664 : !torch.vtensor<[12,197,64],f32>, !torch.list<int> -> !torch.vtensor<[1,12,197,64],f32>
    %666 = torch.prim.ListConstruct %int1, %int12, %int197, %int64 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %667 = torch.aten.view %661, %666 : !torch.vtensor<[12,197,64],f32>, !torch.list<int> -> !torch.vtensor<[1,12,197,64],f32>
    %668 = torch_c.to_builtin_tensor %663 : !torch.vtensor<[1,12,197,64],f32> -> tensor<1x12x197x64xf32>
    %669 = torch_c.to_builtin_tensor %665 : !torch.vtensor<[1,12,197,64],f32> -> tensor<1x12x197x64xf32>
    %670 = torch_c.to_builtin_tensor %667 : !torch.vtensor<[1,12,197,64],f32> -> tensor<1x12x197x64xf32>
    %671 = dnn.scaled_dot_product_attention %668, %669, %670 {parameter_indices = array<i32: 3, 4, 5, 6, 7>, parameters = [unit, 0.000000e+00, false, unit, false]} : (tensor<1x12x197x64xf32>, tensor<1x12x197x64xf32>, tensor<1x12x197x64xf32>) -> tensor<1x12x197x64xf32>
    %672 = torch_c.from_builtin_tensor %671 : tensor<1x12x197x64xf32> -> !torch.vtensor<[1,12,197,64],f32>
    %673 = torch.prim.ListConstruct %int2, %int0, %int1, %int3 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %674 = torch.aten.permute %672, %673 : !torch.vtensor<[1,12,197,64],f32>, !torch.list<int> -> !torch.vtensor<[197,1,12,64],f32>
    %675 = torch.prim.ListConstruct %int197, %int768 : (!torch.int, !torch.int) -> !torch.list<int>
    %676 = torch.aten.view %674, %675 : !torch.vtensor<[197,1,12,64],f32>, !torch.list<int> -> !torch.vtensor<[197,768],f32>
    %677 = torch_c.to_builtin_tensor %676 : !torch.vtensor<[197,768],f32> -> tensor<197x768xf32>
    %678 = dnn.linear %677, %cst_4, %cst_8 : tensor<197x768xf32>, tensor<768x768xf32>, tensor<768xf32> -> tensor<197x768xf32>
    %679 = torch_c.from_builtin_tensor %678 : tensor<197x768xf32> -> !torch.vtensor<[197,768],f32>
    %680 = torch.prim.ListConstruct %int197, %int1, %int768 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %681 = torch.aten.view %679, %680 : !torch.vtensor<[197,768],f32>, !torch.list<int> -> !torch.vtensor<[197,1,768],f32>
    %682 = torch.aten.transpose.int %681, %int1, %int0 : !torch.vtensor<[197,1,768],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,197,768],f32>
    %683 = torch.aten.dropout %682, %float0.000000e00, %false : !torch.vtensor<[1,197,768],f32>, !torch.float, !torch.bool -> !torch.vtensor<[1,197,768],f32>
    %684 = torch_c.to_builtin_tensor %683 : !torch.vtensor<[1,197,768],f32> -> tensor<1x197x768xf32>
    %685 = torch_c.to_builtin_tensor %636 : !torch.vtensor<[1,197,768],f32> -> tensor<1x197x768xf32>
    %686 = dnn.add %684, %685 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %687 = torch_c.from_builtin_tensor %686 : tensor<1x197x768xf32> -> !torch.vtensor<[1,197,768],f32>
    %688 = torch_c.to_builtin_tensor %687 : !torch.vtensor<[1,197,768],f32> -> tensor<1x197x768xf32>
    %689 = dnn.layer_norm %688, %cst_8, %cst_8 {parameter_indices = array<i32: 1, 4, 5>, parameters = [[768], 9.9999999999999995E-7, false]} : (tensor<1x197x768xf32>, tensor<768xf32>, tensor<768xf32>) -> tensor<1x197x768xf32>
    %690 = torch_c.from_builtin_tensor %689 : tensor<1x197x768xf32> -> !torch.vtensor<[1,197,768],f32>
    %691 = torch_c.to_builtin_tensor %690 : !torch.vtensor<[1,197,768],f32> -> tensor<1x197x768xf32>
    %692 = dnn.linear %691, %cst_3, %cst_2 : tensor<1x197x768xf32>, tensor<3072x768xf32>, tensor<3072xf32> -> tensor<1x197x3072xf32>
    %693 = torch_c.from_builtin_tensor %692 : tensor<1x197x3072xf32> -> !torch.vtensor<[1,197,3072],f32>
    %694 = torch_c.to_builtin_tensor %693 : !torch.vtensor<[1,197,3072],f32> -> tensor<1x197x3072xf32>
    %695 = dnn.gelu %694 {parameter_indices = array<i32: 1>, parameters = ["none"]} : (tensor<1x197x3072xf32>) -> tensor<1x197x3072xf32>
    %696 = torch_c.from_builtin_tensor %695 : tensor<1x197x3072xf32> -> !torch.vtensor<[1,197,3072],f32>
    %697 = torch.aten.dropout %696, %float0.000000e00, %false : !torch.vtensor<[1,197,3072],f32>, !torch.float, !torch.bool -> !torch.vtensor<[1,197,3072],f32>
    %698 = torch_c.to_builtin_tensor %697 : !torch.vtensor<[1,197,3072],f32> -> tensor<1x197x3072xf32>
    %699 = dnn.linear %698, %cst_1, %cst_8 : tensor<1x197x3072xf32>, tensor<768x3072xf32>, tensor<768xf32> -> tensor<1x197x768xf32>
    %700 = torch_c.from_builtin_tensor %699 : tensor<1x197x768xf32> -> !torch.vtensor<[1,197,768],f32>
    %701 = torch.aten.dropout %700, %float0.000000e00, %false : !torch.vtensor<[1,197,768],f32>, !torch.float, !torch.bool -> !torch.vtensor<[1,197,768],f32>
    %702 = torch_c.to_builtin_tensor %687 : !torch.vtensor<[1,197,768],f32> -> tensor<1x197x768xf32>
    %703 = torch_c.to_builtin_tensor %701 : !torch.vtensor<[1,197,768],f32> -> tensor<1x197x768xf32>
    %704 = dnn.add %702, %703 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %705 = torch_c.from_builtin_tensor %704 : tensor<1x197x768xf32> -> !torch.vtensor<[1,197,768],f32>
    %706 = torch_c.to_builtin_tensor %705 : !torch.vtensor<[1,197,768],f32> -> tensor<1x197x768xf32>
    %707 = dnn.layer_norm %706, %cst_8, %cst_8 {parameter_indices = array<i32: 1, 4, 5>, parameters = [[768], 9.9999999999999995E-7, false]} : (tensor<1x197x768xf32>, tensor<768xf32>, tensor<768xf32>) -> tensor<1x197x768xf32>
    %708 = torch_c.from_builtin_tensor %707 : tensor<1x197x768xf32> -> !torch.vtensor<[1,197,768],f32>
    %709 = torch.aten.transpose.int %708, %int1, %int0 : !torch.vtensor<[1,197,768],f32>, !torch.int, !torch.int -> !torch.vtensor<[197,1,768],f32>
    %710 = torch_c.to_builtin_tensor %709 : !torch.vtensor<[197,1,768],f32> -> tensor<197x1x768xf32>
    %711 = dnn.linear %710, %cst_6, %cst_5 : tensor<197x1x768xf32>, tensor<2304x768xf32>, tensor<2304xf32> -> tensor<197x1x2304xf32>
    %712 = torch_c.from_builtin_tensor %711 : tensor<197x1x2304xf32> -> !torch.vtensor<[197,1,2304],f32>
    %713 = torch.prim.ListConstruct %int3, %int768 : (!torch.int, !torch.int) -> !torch.list<int>
    %714 = torch.aten.unflatten.int %712, %int-1, %713 : !torch.vtensor<[197,1,2304],f32>, !torch.int, !torch.list<int> -> !torch.vtensor<[197,1,3,768],f32>
    %715 = torch.aten.unsqueeze %714, %int0 : !torch.vtensor<[197,1,3,768],f32>, !torch.int -> !torch.vtensor<[1,197,1,3,768],f32>
    %716 = torch.aten.transpose.int %715, %int0, %int-2 : !torch.vtensor<[1,197,1,3,768],f32>, !torch.int, !torch.int -> !torch.vtensor<[3,197,1,1,768],f32>
    %717 = torch.aten.squeeze.dim %716, %int-2 : !torch.vtensor<[3,197,1,1,768],f32>, !torch.int -> !torch.vtensor<[3,197,1,768],f32>
    %718 = torch.aten.contiguous %717, %int0 : !torch.vtensor<[3,197,1,768],f32>, !torch.int -> !torch.vtensor<[3,197,1,768],f32>
    %719 = torch.aten.select.int %718, %int0, %int0 : !torch.vtensor<[3,197,1,768],f32>, !torch.int, !torch.int -> !torch.vtensor<[197,1,768],f32>
    %720 = torch.aten.select.int %718, %int0, %int1 : !torch.vtensor<[3,197,1,768],f32>, !torch.int, !torch.int -> !torch.vtensor<[197,1,768],f32>
    %721 = torch.aten.select.int %718, %int0, %int2 : !torch.vtensor<[3,197,1,768],f32>, !torch.int, !torch.int -> !torch.vtensor<[197,1,768],f32>
    %722 = torch.prim.ListConstruct %int197, %int12, %int64 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %723 = torch.aten.view %719, %722 : !torch.vtensor<[197,1,768],f32>, !torch.list<int> -> !torch.vtensor<[197,12,64],f32>
    %724 = torch.aten.transpose.int %723, %int0, %int1 : !torch.vtensor<[197,12,64],f32>, !torch.int, !torch.int -> !torch.vtensor<[12,197,64],f32>
    %725 = torch.prim.ListConstruct %int197, %int12, %int64 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %726 = torch.aten.view %720, %725 : !torch.vtensor<[197,1,768],f32>, !torch.list<int> -> !torch.vtensor<[197,12,64],f32>
    %727 = torch.aten.transpose.int %726, %int0, %int1 : !torch.vtensor<[197,12,64],f32>, !torch.int, !torch.int -> !torch.vtensor<[12,197,64],f32>
    %728 = torch.prim.ListConstruct %int197, %int12, %int64 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %729 = torch.aten.view %721, %728 : !torch.vtensor<[197,1,768],f32>, !torch.list<int> -> !torch.vtensor<[197,12,64],f32>
    %730 = torch.aten.transpose.int %729, %int0, %int1 : !torch.vtensor<[197,12,64],f32>, !torch.int, !torch.int -> !torch.vtensor<[12,197,64],f32>
    %731 = torch.prim.ListConstruct %int1, %int12, %int197, %int64 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %732 = torch.aten.view %724, %731 : !torch.vtensor<[12,197,64],f32>, !torch.list<int> -> !torch.vtensor<[1,12,197,64],f32>
    %733 = torch.prim.ListConstruct %int1, %int12, %int197, %int64 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %734 = torch.aten.view %727, %733 : !torch.vtensor<[12,197,64],f32>, !torch.list<int> -> !torch.vtensor<[1,12,197,64],f32>
    %735 = torch.prim.ListConstruct %int1, %int12, %int197, %int64 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %736 = torch.aten.view %730, %735 : !torch.vtensor<[12,197,64],f32>, !torch.list<int> -> !torch.vtensor<[1,12,197,64],f32>
    %737 = torch_c.to_builtin_tensor %732 : !torch.vtensor<[1,12,197,64],f32> -> tensor<1x12x197x64xf32>
    %738 = torch_c.to_builtin_tensor %734 : !torch.vtensor<[1,12,197,64],f32> -> tensor<1x12x197x64xf32>
    %739 = torch_c.to_builtin_tensor %736 : !torch.vtensor<[1,12,197,64],f32> -> tensor<1x12x197x64xf32>
    %740 = dnn.scaled_dot_product_attention %737, %738, %739 {parameter_indices = array<i32: 3, 4, 5, 6, 7>, parameters = [unit, 0.000000e+00, false, unit, false]} : (tensor<1x12x197x64xf32>, tensor<1x12x197x64xf32>, tensor<1x12x197x64xf32>) -> tensor<1x12x197x64xf32>
    %741 = torch_c.from_builtin_tensor %740 : tensor<1x12x197x64xf32> -> !torch.vtensor<[1,12,197,64],f32>
    %742 = torch.prim.ListConstruct %int2, %int0, %int1, %int3 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %743 = torch.aten.permute %741, %742 : !torch.vtensor<[1,12,197,64],f32>, !torch.list<int> -> !torch.vtensor<[197,1,12,64],f32>
    %744 = torch.prim.ListConstruct %int197, %int768 : (!torch.int, !torch.int) -> !torch.list<int>
    %745 = torch.aten.view %743, %744 : !torch.vtensor<[197,1,12,64],f32>, !torch.list<int> -> !torch.vtensor<[197,768],f32>
    %746 = torch_c.to_builtin_tensor %745 : !torch.vtensor<[197,768],f32> -> tensor<197x768xf32>
    %747 = dnn.linear %746, %cst_4, %cst_8 : tensor<197x768xf32>, tensor<768x768xf32>, tensor<768xf32> -> tensor<197x768xf32>
    %748 = torch_c.from_builtin_tensor %747 : tensor<197x768xf32> -> !torch.vtensor<[197,768],f32>
    %749 = torch.prim.ListConstruct %int197, %int1, %int768 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %750 = torch.aten.view %748, %749 : !torch.vtensor<[197,768],f32>, !torch.list<int> -> !torch.vtensor<[197,1,768],f32>
    %751 = torch.aten.transpose.int %750, %int1, %int0 : !torch.vtensor<[197,1,768],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,197,768],f32>
    %752 = torch.aten.dropout %751, %float0.000000e00, %false : !torch.vtensor<[1,197,768],f32>, !torch.float, !torch.bool -> !torch.vtensor<[1,197,768],f32>
    %753 = torch_c.to_builtin_tensor %752 : !torch.vtensor<[1,197,768],f32> -> tensor<1x197x768xf32>
    %754 = torch_c.to_builtin_tensor %705 : !torch.vtensor<[1,197,768],f32> -> tensor<1x197x768xf32>
    %755 = dnn.add %753, %754 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %756 = torch_c.from_builtin_tensor %755 : tensor<1x197x768xf32> -> !torch.vtensor<[1,197,768],f32>
    %757 = torch_c.to_builtin_tensor %756 : !torch.vtensor<[1,197,768],f32> -> tensor<1x197x768xf32>
    %758 = dnn.layer_norm %757, %cst_8, %cst_8 {parameter_indices = array<i32: 1, 4, 5>, parameters = [[768], 9.9999999999999995E-7, false]} : (tensor<1x197x768xf32>, tensor<768xf32>, tensor<768xf32>) -> tensor<1x197x768xf32>
    %759 = torch_c.from_builtin_tensor %758 : tensor<1x197x768xf32> -> !torch.vtensor<[1,197,768],f32>
    %760 = torch_c.to_builtin_tensor %759 : !torch.vtensor<[1,197,768],f32> -> tensor<1x197x768xf32>
    %761 = dnn.linear %760, %cst_3, %cst_2 : tensor<1x197x768xf32>, tensor<3072x768xf32>, tensor<3072xf32> -> tensor<1x197x3072xf32>
    %762 = torch_c.from_builtin_tensor %761 : tensor<1x197x3072xf32> -> !torch.vtensor<[1,197,3072],f32>
    %763 = torch_c.to_builtin_tensor %762 : !torch.vtensor<[1,197,3072],f32> -> tensor<1x197x3072xf32>
    %764 = dnn.gelu %763 {parameter_indices = array<i32: 1>, parameters = ["none"]} : (tensor<1x197x3072xf32>) -> tensor<1x197x3072xf32>
    %765 = torch_c.from_builtin_tensor %764 : tensor<1x197x3072xf32> -> !torch.vtensor<[1,197,3072],f32>
    %766 = torch.aten.dropout %765, %float0.000000e00, %false : !torch.vtensor<[1,197,3072],f32>, !torch.float, !torch.bool -> !torch.vtensor<[1,197,3072],f32>
    %767 = torch_c.to_builtin_tensor %766 : !torch.vtensor<[1,197,3072],f32> -> tensor<1x197x3072xf32>
    %768 = dnn.linear %767, %cst_1, %cst_8 : tensor<1x197x3072xf32>, tensor<768x3072xf32>, tensor<768xf32> -> tensor<1x197x768xf32>
    %769 = torch_c.from_builtin_tensor %768 : tensor<1x197x768xf32> -> !torch.vtensor<[1,197,768],f32>
    %770 = torch.aten.dropout %769, %float0.000000e00, %false : !torch.vtensor<[1,197,768],f32>, !torch.float, !torch.bool -> !torch.vtensor<[1,197,768],f32>
    %771 = torch_c.to_builtin_tensor %756 : !torch.vtensor<[1,197,768],f32> -> tensor<1x197x768xf32>
    %772 = torch_c.to_builtin_tensor %770 : !torch.vtensor<[1,197,768],f32> -> tensor<1x197x768xf32>
    %773 = dnn.add %771, %772 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %774 = torch_c.from_builtin_tensor %773 : tensor<1x197x768xf32> -> !torch.vtensor<[1,197,768],f32>
    %775 = torch_c.to_builtin_tensor %774 : !torch.vtensor<[1,197,768],f32> -> tensor<1x197x768xf32>
    %776 = dnn.layer_norm %775, %cst_8, %cst_8 {parameter_indices = array<i32: 1, 4, 5>, parameters = [[768], 9.9999999999999995E-7, false]} : (tensor<1x197x768xf32>, tensor<768xf32>, tensor<768xf32>) -> tensor<1x197x768xf32>
    %777 = torch_c.from_builtin_tensor %776 : tensor<1x197x768xf32> -> !torch.vtensor<[1,197,768],f32>
    %778 = torch.aten.transpose.int %777, %int1, %int0 : !torch.vtensor<[1,197,768],f32>, !torch.int, !torch.int -> !torch.vtensor<[197,1,768],f32>
    %779 = torch_c.to_builtin_tensor %778 : !torch.vtensor<[197,1,768],f32> -> tensor<197x1x768xf32>
    %780 = dnn.linear %779, %cst_6, %cst_5 : tensor<197x1x768xf32>, tensor<2304x768xf32>, tensor<2304xf32> -> tensor<197x1x2304xf32>
    %781 = torch_c.from_builtin_tensor %780 : tensor<197x1x2304xf32> -> !torch.vtensor<[197,1,2304],f32>
    %782 = torch.prim.ListConstruct %int3, %int768 : (!torch.int, !torch.int) -> !torch.list<int>
    %783 = torch.aten.unflatten.int %781, %int-1, %782 : !torch.vtensor<[197,1,2304],f32>, !torch.int, !torch.list<int> -> !torch.vtensor<[197,1,3,768],f32>
    %784 = torch.aten.unsqueeze %783, %int0 : !torch.vtensor<[197,1,3,768],f32>, !torch.int -> !torch.vtensor<[1,197,1,3,768],f32>
    %785 = torch.aten.transpose.int %784, %int0, %int-2 : !torch.vtensor<[1,197,1,3,768],f32>, !torch.int, !torch.int -> !torch.vtensor<[3,197,1,1,768],f32>
    %786 = torch.aten.squeeze.dim %785, %int-2 : !torch.vtensor<[3,197,1,1,768],f32>, !torch.int -> !torch.vtensor<[3,197,1,768],f32>
    %787 = torch.aten.contiguous %786, %int0 : !torch.vtensor<[3,197,1,768],f32>, !torch.int -> !torch.vtensor<[3,197,1,768],f32>
    %788 = torch.aten.select.int %787, %int0, %int0 : !torch.vtensor<[3,197,1,768],f32>, !torch.int, !torch.int -> !torch.vtensor<[197,1,768],f32>
    %789 = torch.aten.select.int %787, %int0, %int1 : !torch.vtensor<[3,197,1,768],f32>, !torch.int, !torch.int -> !torch.vtensor<[197,1,768],f32>
    %790 = torch.aten.select.int %787, %int0, %int2 : !torch.vtensor<[3,197,1,768],f32>, !torch.int, !torch.int -> !torch.vtensor<[197,1,768],f32>
    %791 = torch.prim.ListConstruct %int197, %int12, %int64 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %792 = torch.aten.view %788, %791 : !torch.vtensor<[197,1,768],f32>, !torch.list<int> -> !torch.vtensor<[197,12,64],f32>
    %793 = torch.aten.transpose.int %792, %int0, %int1 : !torch.vtensor<[197,12,64],f32>, !torch.int, !torch.int -> !torch.vtensor<[12,197,64],f32>
    %794 = torch.prim.ListConstruct %int197, %int12, %int64 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %795 = torch.aten.view %789, %794 : !torch.vtensor<[197,1,768],f32>, !torch.list<int> -> !torch.vtensor<[197,12,64],f32>
    %796 = torch.aten.transpose.int %795, %int0, %int1 : !torch.vtensor<[197,12,64],f32>, !torch.int, !torch.int -> !torch.vtensor<[12,197,64],f32>
    %797 = torch.prim.ListConstruct %int197, %int12, %int64 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %798 = torch.aten.view %790, %797 : !torch.vtensor<[197,1,768],f32>, !torch.list<int> -> !torch.vtensor<[197,12,64],f32>
    %799 = torch.aten.transpose.int %798, %int0, %int1 : !torch.vtensor<[197,12,64],f32>, !torch.int, !torch.int -> !torch.vtensor<[12,197,64],f32>
    %800 = torch.prim.ListConstruct %int1, %int12, %int197, %int64 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %801 = torch.aten.view %793, %800 : !torch.vtensor<[12,197,64],f32>, !torch.list<int> -> !torch.vtensor<[1,12,197,64],f32>
    %802 = torch.prim.ListConstruct %int1, %int12, %int197, %int64 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %803 = torch.aten.view %796, %802 : !torch.vtensor<[12,197,64],f32>, !torch.list<int> -> !torch.vtensor<[1,12,197,64],f32>
    %804 = torch.prim.ListConstruct %int1, %int12, %int197, %int64 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %805 = torch.aten.view %799, %804 : !torch.vtensor<[12,197,64],f32>, !torch.list<int> -> !torch.vtensor<[1,12,197,64],f32>
    %806 = torch_c.to_builtin_tensor %801 : !torch.vtensor<[1,12,197,64],f32> -> tensor<1x12x197x64xf32>
    %807 = torch_c.to_builtin_tensor %803 : !torch.vtensor<[1,12,197,64],f32> -> tensor<1x12x197x64xf32>
    %808 = torch_c.to_builtin_tensor %805 : !torch.vtensor<[1,12,197,64],f32> -> tensor<1x12x197x64xf32>
    %809 = dnn.scaled_dot_product_attention %806, %807, %808 {parameter_indices = array<i32: 3, 4, 5, 6, 7>, parameters = [unit, 0.000000e+00, false, unit, false]} : (tensor<1x12x197x64xf32>, tensor<1x12x197x64xf32>, tensor<1x12x197x64xf32>) -> tensor<1x12x197x64xf32>
    %810 = torch_c.from_builtin_tensor %809 : tensor<1x12x197x64xf32> -> !torch.vtensor<[1,12,197,64],f32>
    %811 = torch.prim.ListConstruct %int2, %int0, %int1, %int3 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %812 = torch.aten.permute %810, %811 : !torch.vtensor<[1,12,197,64],f32>, !torch.list<int> -> !torch.vtensor<[197,1,12,64],f32>
    %813 = torch.prim.ListConstruct %int197, %int768 : (!torch.int, !torch.int) -> !torch.list<int>
    %814 = torch.aten.view %812, %813 : !torch.vtensor<[197,1,12,64],f32>, !torch.list<int> -> !torch.vtensor<[197,768],f32>
    %815 = torch_c.to_builtin_tensor %814 : !torch.vtensor<[197,768],f32> -> tensor<197x768xf32>
    %816 = dnn.linear %815, %cst_4, %cst_8 : tensor<197x768xf32>, tensor<768x768xf32>, tensor<768xf32> -> tensor<197x768xf32>
    %817 = torch_c.from_builtin_tensor %816 : tensor<197x768xf32> -> !torch.vtensor<[197,768],f32>
    %818 = torch.prim.ListConstruct %int197, %int1, %int768 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %819 = torch.aten.view %817, %818 : !torch.vtensor<[197,768],f32>, !torch.list<int> -> !torch.vtensor<[197,1,768],f32>
    %820 = torch.aten.transpose.int %819, %int1, %int0 : !torch.vtensor<[197,1,768],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,197,768],f32>
    %821 = torch.aten.dropout %820, %float0.000000e00, %false : !torch.vtensor<[1,197,768],f32>, !torch.float, !torch.bool -> !torch.vtensor<[1,197,768],f32>
    %822 = torch_c.to_builtin_tensor %821 : !torch.vtensor<[1,197,768],f32> -> tensor<1x197x768xf32>
    %823 = torch_c.to_builtin_tensor %774 : !torch.vtensor<[1,197,768],f32> -> tensor<1x197x768xf32>
    %824 = dnn.add %822, %823 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %825 = torch_c.from_builtin_tensor %824 : tensor<1x197x768xf32> -> !torch.vtensor<[1,197,768],f32>
    %826 = torch_c.to_builtin_tensor %825 : !torch.vtensor<[1,197,768],f32> -> tensor<1x197x768xf32>
    %827 = dnn.layer_norm %826, %cst_8, %cst_8 {parameter_indices = array<i32: 1, 4, 5>, parameters = [[768], 9.9999999999999995E-7, false]} : (tensor<1x197x768xf32>, tensor<768xf32>, tensor<768xf32>) -> tensor<1x197x768xf32>
    %828 = torch_c.from_builtin_tensor %827 : tensor<1x197x768xf32> -> !torch.vtensor<[1,197,768],f32>
    %829 = torch_c.to_builtin_tensor %828 : !torch.vtensor<[1,197,768],f32> -> tensor<1x197x768xf32>
    %830 = dnn.linear %829, %cst_3, %cst_2 : tensor<1x197x768xf32>, tensor<3072x768xf32>, tensor<3072xf32> -> tensor<1x197x3072xf32>
    %831 = torch_c.from_builtin_tensor %830 : tensor<1x197x3072xf32> -> !torch.vtensor<[1,197,3072],f32>
    %832 = torch_c.to_builtin_tensor %831 : !torch.vtensor<[1,197,3072],f32> -> tensor<1x197x3072xf32>
    %833 = dnn.gelu %832 {parameter_indices = array<i32: 1>, parameters = ["none"]} : (tensor<1x197x3072xf32>) -> tensor<1x197x3072xf32>
    %834 = torch_c.from_builtin_tensor %833 : tensor<1x197x3072xf32> -> !torch.vtensor<[1,197,3072],f32>
    %835 = torch.aten.dropout %834, %float0.000000e00, %false : !torch.vtensor<[1,197,3072],f32>, !torch.float, !torch.bool -> !torch.vtensor<[1,197,3072],f32>
    %836 = torch_c.to_builtin_tensor %835 : !torch.vtensor<[1,197,3072],f32> -> tensor<1x197x3072xf32>
    %837 = dnn.linear %836, %cst_1, %cst_8 : tensor<1x197x3072xf32>, tensor<768x3072xf32>, tensor<768xf32> -> tensor<1x197x768xf32>
    %838 = torch_c.from_builtin_tensor %837 : tensor<1x197x768xf32> -> !torch.vtensor<[1,197,768],f32>
    %839 = torch.aten.dropout %838, %float0.000000e00, %false : !torch.vtensor<[1,197,768],f32>, !torch.float, !torch.bool -> !torch.vtensor<[1,197,768],f32>
    %840 = torch_c.to_builtin_tensor %825 : !torch.vtensor<[1,197,768],f32> -> tensor<1x197x768xf32>
    %841 = torch_c.to_builtin_tensor %839 : !torch.vtensor<[1,197,768],f32> -> tensor<1x197x768xf32>
    %842 = dnn.add %840, %841 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %843 = torch_c.from_builtin_tensor %842 : tensor<1x197x768xf32> -> !torch.vtensor<[1,197,768],f32>
    %844 = torch_c.to_builtin_tensor %843 : !torch.vtensor<[1,197,768],f32> -> tensor<1x197x768xf32>
    %845 = dnn.layer_norm %844, %cst_8, %cst_8 {parameter_indices = array<i32: 1, 4, 5>, parameters = [[768], 9.9999999999999995E-7, false]} : (tensor<1x197x768xf32>, tensor<768xf32>, tensor<768xf32>) -> tensor<1x197x768xf32>
    %846 = torch_c.from_builtin_tensor %845 : tensor<1x197x768xf32> -> !torch.vtensor<[1,197,768],f32>
    %847 = torch.aten.select.int %846, %int1, %int0 : !torch.vtensor<[1,197,768],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,768],f32>
    %848 = torch_c.to_builtin_tensor %847 : !torch.vtensor<[1,768],f32> -> tensor<1x768xf32>
    %849 = dnn.linear %848, %cst_0, %cst : tensor<1x768xf32>, tensor<1000x768xf32>, tensor<1000xf32> -> tensor<1x1000xf32>
    %850 = torch_c.from_builtin_tensor %849 : tensor<1x1000xf32> -> !torch.vtensor<[1,1000],f32>
    return %850 : !torch.vtensor<[1,1000],f32>
  }
}

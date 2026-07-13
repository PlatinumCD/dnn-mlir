// RUN: dnn-mlir-opt %s | FileCheck %s
//
// BERT Base encoder after captures=all and the remaining Torch-to-Linalg
// backend lowering. Transformer blocks and embedding lookup remain high-level
// DNN operations; uncaptured tensor layout manipulation uses standard MLIR
// dialects.
//
// CHECK-LABEL: func.func @bert_base_encoder
// CHECK-DAG: dnn.linear
// CHECK-DAG: dnn.add
// CHECK-DAG: dnn.embedding
// CHECK-DAG: dnn.layer_norm
// CHECK-DAG: dnn.scaled_dot_product_attention
// CHECK-DAG: dnn.gelu
// CHECK-DAG: dnn.tanh
// CHECK-DAG: linalg.generic
// CHECK-DAG: linalg.transpose
// CHECK-NOT: torch.

#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @bert_base_encoder(%arg0: tensor<1x128xi64>, %arg1: tensor<1x1x128x128xi1>) -> tensor<1x128x768xf32> {
    %cst = arith.constant dense<0> : tensor<1x128xi64>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<768xf32>
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<768x768xf32>
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<3072x768xf32>
    %cst_3 = arith.constant dense<0.000000e+00> : tensor<3072xf32>
    %cst_4 = arith.constant dense<0.000000e+00> : tensor<768x3072xf32>
    %cst_5 = arith.constant dense<0.000000e+00> : tensor<30522x768xf32>
    %cst_6 = arith.constant dense<0.000000e+00> : tensor<2x768xf32>
    %cst_7 = arith.constant dense<0.000000e+00> : tensor<512x768xf32>
    %c0_i64 = arith.constant 0 : i64
    %c512 = arith.constant 512 : index
    %0 = tensor.empty() : tensor<1x128xi64>
    %1 = linalg.fill ins(%c0_i64 : i64) outs(%0 : tensor<1x128xi64>) -> tensor<1x128xi64>
    %2 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%cst : tensor<1x128xi64>) outs(%1 : tensor<1x128xi64>) {
    ^bb0(%in: i64, %out: i64):
      %133 = arith.index_cast %in : i64 to index
      %134 = arith.cmpi slt, %133, %c512 : index
      cf.assert %134, "index must be smaller than dim size"
      %135 = arith.cmpi sge, %in, %c0_i64 : i64
      cf.assert %135, "index must be larger or equal to 0"
      linalg.yield %c0_i64 : i64
    } -> tensor<1x128xi64>
    %3 = dnn.embedding %cst_5, %arg0 {parameter_indices = array<i32: 2, 3, 4>, parameters = [0, false, false]} : (tensor<30522x768xf32>, tensor<1x128xi64>) -> tensor<1x128x768xf32>
    %4 = dnn.embedding %cst_6, %2 {parameter_indices = array<i32: 2, 3, 4>, parameters = [-1, false, false]} : (tensor<2x768xf32>, tensor<1x128xi64>) -> tensor<1x128x768xf32>
    %5 = dnn.add %3, %4 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<1x128x768xf32>, tensor<1x128x768xf32>) -> tensor<1x128x768xf32>
    %6 = dnn.embedding %cst_7, %cst {parameter_indices = array<i32: 2, 3, 4>, parameters = [-1, false, false]} : (tensor<512x768xf32>, tensor<1x128xi64>) -> tensor<1x128x768xf32>
    %7 = dnn.add %5, %6 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<1x128x768xf32>, tensor<1x128x768xf32>) -> tensor<1x128x768xf32>
    %8 = dnn.layer_norm %7, %cst_0, %cst_0 {parameter_indices = array<i32: 1, 4, 5>, parameters = [[768], 9.9999999999999998E-13, false]} : (tensor<1x128x768xf32>, tensor<768xf32>, tensor<768xf32>) -> tensor<1x128x768xf32>
    %9 = dnn.linear %8, %cst_1, %cst_0 : tensor<1x128x768xf32>, tensor<768x768xf32>, tensor<768xf32> -> tensor<1x128x768xf32>
    %expanded = tensor.expand_shape %9 [[0], [1], [2, 3]] output_shape [1, 128, 12, 64] : tensor<1x128x768xf32> into tensor<1x128x12x64xf32>
    %10 = tensor.empty() : tensor<1x12x128x64xf32>
    %transposed = linalg.transpose ins(%expanded : tensor<1x128x12x64xf32>) outs(%10 : tensor<1x12x128x64xf32>) permutation = [0, 2, 1, 3] 
    %11 = dnn.scaled_dot_product_attention %transposed, %transposed, %transposed, %arg1 {parameter_indices = array<i32: 4, 5, 6, 7>, parameters = [0.000000e+00, false, 1.250000e-01, false]} : (tensor<1x12x128x64xf32>, tensor<1x12x128x64xf32>, tensor<1x12x128x64xf32>, tensor<1x1x128x128xi1>) -> tensor<1x12x128x64xf32>
    %12 = tensor.empty() : tensor<1x128x12x64xf32>
    %transposed_8 = linalg.transpose ins(%11 : tensor<1x12x128x64xf32>) outs(%12 : tensor<1x128x12x64xf32>) permutation = [0, 2, 1, 3] 
    %collapsed = tensor.collapse_shape %transposed_8 [[0], [1], [2, 3]] : tensor<1x128x12x64xf32> into tensor<1x128x768xf32>
    %13 = dnn.linear %collapsed, %cst_1, %cst_0 : tensor<1x128x768xf32>, tensor<768x768xf32>, tensor<768xf32> -> tensor<1x128x768xf32>
    %14 = dnn.add %13, %8 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<1x128x768xf32>, tensor<1x128x768xf32>) -> tensor<1x128x768xf32>
    %15 = dnn.layer_norm %14, %cst_0, %cst_0 {parameter_indices = array<i32: 1, 4, 5>, parameters = [[768], 9.9999999999999998E-13, false]} : (tensor<1x128x768xf32>, tensor<768xf32>, tensor<768xf32>) -> tensor<1x128x768xf32>
    %16 = dnn.linear %15, %cst_2, %cst_3 : tensor<1x128x768xf32>, tensor<3072x768xf32>, tensor<3072xf32> -> tensor<1x128x3072xf32>
    %17 = dnn.gelu %16 {parameter_indices = array<i32: 1>, parameters = ["none"]} : (tensor<1x128x3072xf32>) -> tensor<1x128x3072xf32>
    %18 = dnn.linear %17, %cst_4, %cst_0 : tensor<1x128x3072xf32>, tensor<768x3072xf32>, tensor<768xf32> -> tensor<1x128x768xf32>
    %19 = dnn.add %18, %15 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<1x128x768xf32>, tensor<1x128x768xf32>) -> tensor<1x128x768xf32>
    %20 = dnn.layer_norm %19, %cst_0, %cst_0 {parameter_indices = array<i32: 1, 4, 5>, parameters = [[768], 9.9999999999999998E-13, false]} : (tensor<1x128x768xf32>, tensor<768xf32>, tensor<768xf32>) -> tensor<1x128x768xf32>
    %21 = dnn.linear %20, %cst_1, %cst_0 : tensor<1x128x768xf32>, tensor<768x768xf32>, tensor<768xf32> -> tensor<1x128x768xf32>
    %expanded_9 = tensor.expand_shape %21 [[0], [1], [2, 3]] output_shape [1, 128, 12, 64] : tensor<1x128x768xf32> into tensor<1x128x12x64xf32>
    %transposed_10 = linalg.transpose ins(%expanded_9 : tensor<1x128x12x64xf32>) outs(%10 : tensor<1x12x128x64xf32>) permutation = [0, 2, 1, 3] 
    %22 = dnn.scaled_dot_product_attention %transposed_10, %transposed_10, %transposed_10, %arg1 {parameter_indices = array<i32: 4, 5, 6, 7>, parameters = [0.000000e+00, false, 1.250000e-01, false]} : (tensor<1x12x128x64xf32>, tensor<1x12x128x64xf32>, tensor<1x12x128x64xf32>, tensor<1x1x128x128xi1>) -> tensor<1x12x128x64xf32>
    %transposed_11 = linalg.transpose ins(%22 : tensor<1x12x128x64xf32>) outs(%12 : tensor<1x128x12x64xf32>) permutation = [0, 2, 1, 3] 
    %collapsed_12 = tensor.collapse_shape %transposed_11 [[0], [1], [2, 3]] : tensor<1x128x12x64xf32> into tensor<1x128x768xf32>
    %23 = dnn.linear %collapsed_12, %cst_1, %cst_0 : tensor<1x128x768xf32>, tensor<768x768xf32>, tensor<768xf32> -> tensor<1x128x768xf32>
    %24 = dnn.add %23, %20 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<1x128x768xf32>, tensor<1x128x768xf32>) -> tensor<1x128x768xf32>
    %25 = dnn.layer_norm %24, %cst_0, %cst_0 {parameter_indices = array<i32: 1, 4, 5>, parameters = [[768], 9.9999999999999998E-13, false]} : (tensor<1x128x768xf32>, tensor<768xf32>, tensor<768xf32>) -> tensor<1x128x768xf32>
    %26 = dnn.linear %25, %cst_2, %cst_3 : tensor<1x128x768xf32>, tensor<3072x768xf32>, tensor<3072xf32> -> tensor<1x128x3072xf32>
    %27 = dnn.gelu %26 {parameter_indices = array<i32: 1>, parameters = ["none"]} : (tensor<1x128x3072xf32>) -> tensor<1x128x3072xf32>
    %28 = dnn.linear %27, %cst_4, %cst_0 : tensor<1x128x3072xf32>, tensor<768x3072xf32>, tensor<768xf32> -> tensor<1x128x768xf32>
    %29 = dnn.add %28, %25 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<1x128x768xf32>, tensor<1x128x768xf32>) -> tensor<1x128x768xf32>
    %30 = dnn.layer_norm %29, %cst_0, %cst_0 {parameter_indices = array<i32: 1, 4, 5>, parameters = [[768], 9.9999999999999998E-13, false]} : (tensor<1x128x768xf32>, tensor<768xf32>, tensor<768xf32>) -> tensor<1x128x768xf32>
    %31 = dnn.linear %30, %cst_1, %cst_0 : tensor<1x128x768xf32>, tensor<768x768xf32>, tensor<768xf32> -> tensor<1x128x768xf32>
    %expanded_13 = tensor.expand_shape %31 [[0], [1], [2, 3]] output_shape [1, 128, 12, 64] : tensor<1x128x768xf32> into tensor<1x128x12x64xf32>
    %transposed_14 = linalg.transpose ins(%expanded_13 : tensor<1x128x12x64xf32>) outs(%10 : tensor<1x12x128x64xf32>) permutation = [0, 2, 1, 3] 
    %32 = dnn.scaled_dot_product_attention %transposed_14, %transposed_14, %transposed_14, %arg1 {parameter_indices = array<i32: 4, 5, 6, 7>, parameters = [0.000000e+00, false, 1.250000e-01, false]} : (tensor<1x12x128x64xf32>, tensor<1x12x128x64xf32>, tensor<1x12x128x64xf32>, tensor<1x1x128x128xi1>) -> tensor<1x12x128x64xf32>
    %transposed_15 = linalg.transpose ins(%32 : tensor<1x12x128x64xf32>) outs(%12 : tensor<1x128x12x64xf32>) permutation = [0, 2, 1, 3] 
    %collapsed_16 = tensor.collapse_shape %transposed_15 [[0], [1], [2, 3]] : tensor<1x128x12x64xf32> into tensor<1x128x768xf32>
    %33 = dnn.linear %collapsed_16, %cst_1, %cst_0 : tensor<1x128x768xf32>, tensor<768x768xf32>, tensor<768xf32> -> tensor<1x128x768xf32>
    %34 = dnn.add %33, %30 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<1x128x768xf32>, tensor<1x128x768xf32>) -> tensor<1x128x768xf32>
    %35 = dnn.layer_norm %34, %cst_0, %cst_0 {parameter_indices = array<i32: 1, 4, 5>, parameters = [[768], 9.9999999999999998E-13, false]} : (tensor<1x128x768xf32>, tensor<768xf32>, tensor<768xf32>) -> tensor<1x128x768xf32>
    %36 = dnn.linear %35, %cst_2, %cst_3 : tensor<1x128x768xf32>, tensor<3072x768xf32>, tensor<3072xf32> -> tensor<1x128x3072xf32>
    %37 = dnn.gelu %36 {parameter_indices = array<i32: 1>, parameters = ["none"]} : (tensor<1x128x3072xf32>) -> tensor<1x128x3072xf32>
    %38 = dnn.linear %37, %cst_4, %cst_0 : tensor<1x128x3072xf32>, tensor<768x3072xf32>, tensor<768xf32> -> tensor<1x128x768xf32>
    %39 = dnn.add %38, %35 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<1x128x768xf32>, tensor<1x128x768xf32>) -> tensor<1x128x768xf32>
    %40 = dnn.layer_norm %39, %cst_0, %cst_0 {parameter_indices = array<i32: 1, 4, 5>, parameters = [[768], 9.9999999999999998E-13, false]} : (tensor<1x128x768xf32>, tensor<768xf32>, tensor<768xf32>) -> tensor<1x128x768xf32>
    %41 = dnn.linear %40, %cst_1, %cst_0 : tensor<1x128x768xf32>, tensor<768x768xf32>, tensor<768xf32> -> tensor<1x128x768xf32>
    %expanded_17 = tensor.expand_shape %41 [[0], [1], [2, 3]] output_shape [1, 128, 12, 64] : tensor<1x128x768xf32> into tensor<1x128x12x64xf32>
    %transposed_18 = linalg.transpose ins(%expanded_17 : tensor<1x128x12x64xf32>) outs(%10 : tensor<1x12x128x64xf32>) permutation = [0, 2, 1, 3] 
    %42 = dnn.scaled_dot_product_attention %transposed_18, %transposed_18, %transposed_18, %arg1 {parameter_indices = array<i32: 4, 5, 6, 7>, parameters = [0.000000e+00, false, 1.250000e-01, false]} : (tensor<1x12x128x64xf32>, tensor<1x12x128x64xf32>, tensor<1x12x128x64xf32>, tensor<1x1x128x128xi1>) -> tensor<1x12x128x64xf32>
    %transposed_19 = linalg.transpose ins(%42 : tensor<1x12x128x64xf32>) outs(%12 : tensor<1x128x12x64xf32>) permutation = [0, 2, 1, 3] 
    %collapsed_20 = tensor.collapse_shape %transposed_19 [[0], [1], [2, 3]] : tensor<1x128x12x64xf32> into tensor<1x128x768xf32>
    %43 = dnn.linear %collapsed_20, %cst_1, %cst_0 : tensor<1x128x768xf32>, tensor<768x768xf32>, tensor<768xf32> -> tensor<1x128x768xf32>
    %44 = dnn.add %43, %40 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<1x128x768xf32>, tensor<1x128x768xf32>) -> tensor<1x128x768xf32>
    %45 = dnn.layer_norm %44, %cst_0, %cst_0 {parameter_indices = array<i32: 1, 4, 5>, parameters = [[768], 9.9999999999999998E-13, false]} : (tensor<1x128x768xf32>, tensor<768xf32>, tensor<768xf32>) -> tensor<1x128x768xf32>
    %46 = dnn.linear %45, %cst_2, %cst_3 : tensor<1x128x768xf32>, tensor<3072x768xf32>, tensor<3072xf32> -> tensor<1x128x3072xf32>
    %47 = dnn.gelu %46 {parameter_indices = array<i32: 1>, parameters = ["none"]} : (tensor<1x128x3072xf32>) -> tensor<1x128x3072xf32>
    %48 = dnn.linear %47, %cst_4, %cst_0 : tensor<1x128x3072xf32>, tensor<768x3072xf32>, tensor<768xf32> -> tensor<1x128x768xf32>
    %49 = dnn.add %48, %45 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<1x128x768xf32>, tensor<1x128x768xf32>) -> tensor<1x128x768xf32>
    %50 = dnn.layer_norm %49, %cst_0, %cst_0 {parameter_indices = array<i32: 1, 4, 5>, parameters = [[768], 9.9999999999999998E-13, false]} : (tensor<1x128x768xf32>, tensor<768xf32>, tensor<768xf32>) -> tensor<1x128x768xf32>
    %51 = dnn.linear %50, %cst_1, %cst_0 : tensor<1x128x768xf32>, tensor<768x768xf32>, tensor<768xf32> -> tensor<1x128x768xf32>
    %expanded_21 = tensor.expand_shape %51 [[0], [1], [2, 3]] output_shape [1, 128, 12, 64] : tensor<1x128x768xf32> into tensor<1x128x12x64xf32>
    %transposed_22 = linalg.transpose ins(%expanded_21 : tensor<1x128x12x64xf32>) outs(%10 : tensor<1x12x128x64xf32>) permutation = [0, 2, 1, 3] 
    %52 = dnn.scaled_dot_product_attention %transposed_22, %transposed_22, %transposed_22, %arg1 {parameter_indices = array<i32: 4, 5, 6, 7>, parameters = [0.000000e+00, false, 1.250000e-01, false]} : (tensor<1x12x128x64xf32>, tensor<1x12x128x64xf32>, tensor<1x12x128x64xf32>, tensor<1x1x128x128xi1>) -> tensor<1x12x128x64xf32>
    %transposed_23 = linalg.transpose ins(%52 : tensor<1x12x128x64xf32>) outs(%12 : tensor<1x128x12x64xf32>) permutation = [0, 2, 1, 3] 
    %collapsed_24 = tensor.collapse_shape %transposed_23 [[0], [1], [2, 3]] : tensor<1x128x12x64xf32> into tensor<1x128x768xf32>
    %53 = dnn.linear %collapsed_24, %cst_1, %cst_0 : tensor<1x128x768xf32>, tensor<768x768xf32>, tensor<768xf32> -> tensor<1x128x768xf32>
    %54 = dnn.add %53, %50 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<1x128x768xf32>, tensor<1x128x768xf32>) -> tensor<1x128x768xf32>
    %55 = dnn.layer_norm %54, %cst_0, %cst_0 {parameter_indices = array<i32: 1, 4, 5>, parameters = [[768], 9.9999999999999998E-13, false]} : (tensor<1x128x768xf32>, tensor<768xf32>, tensor<768xf32>) -> tensor<1x128x768xf32>
    %56 = dnn.linear %55, %cst_2, %cst_3 : tensor<1x128x768xf32>, tensor<3072x768xf32>, tensor<3072xf32> -> tensor<1x128x3072xf32>
    %57 = dnn.gelu %56 {parameter_indices = array<i32: 1>, parameters = ["none"]} : (tensor<1x128x3072xf32>) -> tensor<1x128x3072xf32>
    %58 = dnn.linear %57, %cst_4, %cst_0 : tensor<1x128x3072xf32>, tensor<768x3072xf32>, tensor<768xf32> -> tensor<1x128x768xf32>
    %59 = dnn.add %58, %55 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<1x128x768xf32>, tensor<1x128x768xf32>) -> tensor<1x128x768xf32>
    %60 = dnn.layer_norm %59, %cst_0, %cst_0 {parameter_indices = array<i32: 1, 4, 5>, parameters = [[768], 9.9999999999999998E-13, false]} : (tensor<1x128x768xf32>, tensor<768xf32>, tensor<768xf32>) -> tensor<1x128x768xf32>
    %61 = dnn.linear %60, %cst_1, %cst_0 : tensor<1x128x768xf32>, tensor<768x768xf32>, tensor<768xf32> -> tensor<1x128x768xf32>
    %expanded_25 = tensor.expand_shape %61 [[0], [1], [2, 3]] output_shape [1, 128, 12, 64] : tensor<1x128x768xf32> into tensor<1x128x12x64xf32>
    %transposed_26 = linalg.transpose ins(%expanded_25 : tensor<1x128x12x64xf32>) outs(%10 : tensor<1x12x128x64xf32>) permutation = [0, 2, 1, 3] 
    %62 = dnn.scaled_dot_product_attention %transposed_26, %transposed_26, %transposed_26, %arg1 {parameter_indices = array<i32: 4, 5, 6, 7>, parameters = [0.000000e+00, false, 1.250000e-01, false]} : (tensor<1x12x128x64xf32>, tensor<1x12x128x64xf32>, tensor<1x12x128x64xf32>, tensor<1x1x128x128xi1>) -> tensor<1x12x128x64xf32>
    %transposed_27 = linalg.transpose ins(%62 : tensor<1x12x128x64xf32>) outs(%12 : tensor<1x128x12x64xf32>) permutation = [0, 2, 1, 3] 
    %collapsed_28 = tensor.collapse_shape %transposed_27 [[0], [1], [2, 3]] : tensor<1x128x12x64xf32> into tensor<1x128x768xf32>
    %63 = dnn.linear %collapsed_28, %cst_1, %cst_0 : tensor<1x128x768xf32>, tensor<768x768xf32>, tensor<768xf32> -> tensor<1x128x768xf32>
    %64 = dnn.add %63, %60 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<1x128x768xf32>, tensor<1x128x768xf32>) -> tensor<1x128x768xf32>
    %65 = dnn.layer_norm %64, %cst_0, %cst_0 {parameter_indices = array<i32: 1, 4, 5>, parameters = [[768], 9.9999999999999998E-13, false]} : (tensor<1x128x768xf32>, tensor<768xf32>, tensor<768xf32>) -> tensor<1x128x768xf32>
    %66 = dnn.linear %65, %cst_2, %cst_3 : tensor<1x128x768xf32>, tensor<3072x768xf32>, tensor<3072xf32> -> tensor<1x128x3072xf32>
    %67 = dnn.gelu %66 {parameter_indices = array<i32: 1>, parameters = ["none"]} : (tensor<1x128x3072xf32>) -> tensor<1x128x3072xf32>
    %68 = dnn.linear %67, %cst_4, %cst_0 : tensor<1x128x3072xf32>, tensor<768x3072xf32>, tensor<768xf32> -> tensor<1x128x768xf32>
    %69 = dnn.add %68, %65 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<1x128x768xf32>, tensor<1x128x768xf32>) -> tensor<1x128x768xf32>
    %70 = dnn.layer_norm %69, %cst_0, %cst_0 {parameter_indices = array<i32: 1, 4, 5>, parameters = [[768], 9.9999999999999998E-13, false]} : (tensor<1x128x768xf32>, tensor<768xf32>, tensor<768xf32>) -> tensor<1x128x768xf32>
    %71 = dnn.linear %70, %cst_1, %cst_0 : tensor<1x128x768xf32>, tensor<768x768xf32>, tensor<768xf32> -> tensor<1x128x768xf32>
    %expanded_29 = tensor.expand_shape %71 [[0], [1], [2, 3]] output_shape [1, 128, 12, 64] : tensor<1x128x768xf32> into tensor<1x128x12x64xf32>
    %transposed_30 = linalg.transpose ins(%expanded_29 : tensor<1x128x12x64xf32>) outs(%10 : tensor<1x12x128x64xf32>) permutation = [0, 2, 1, 3] 
    %72 = dnn.scaled_dot_product_attention %transposed_30, %transposed_30, %transposed_30, %arg1 {parameter_indices = array<i32: 4, 5, 6, 7>, parameters = [0.000000e+00, false, 1.250000e-01, false]} : (tensor<1x12x128x64xf32>, tensor<1x12x128x64xf32>, tensor<1x12x128x64xf32>, tensor<1x1x128x128xi1>) -> tensor<1x12x128x64xf32>
    %transposed_31 = linalg.transpose ins(%72 : tensor<1x12x128x64xf32>) outs(%12 : tensor<1x128x12x64xf32>) permutation = [0, 2, 1, 3] 
    %collapsed_32 = tensor.collapse_shape %transposed_31 [[0], [1], [2, 3]] : tensor<1x128x12x64xf32> into tensor<1x128x768xf32>
    %73 = dnn.linear %collapsed_32, %cst_1, %cst_0 : tensor<1x128x768xf32>, tensor<768x768xf32>, tensor<768xf32> -> tensor<1x128x768xf32>
    %74 = dnn.add %73, %70 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<1x128x768xf32>, tensor<1x128x768xf32>) -> tensor<1x128x768xf32>
    %75 = dnn.layer_norm %74, %cst_0, %cst_0 {parameter_indices = array<i32: 1, 4, 5>, parameters = [[768], 9.9999999999999998E-13, false]} : (tensor<1x128x768xf32>, tensor<768xf32>, tensor<768xf32>) -> tensor<1x128x768xf32>
    %76 = dnn.linear %75, %cst_2, %cst_3 : tensor<1x128x768xf32>, tensor<3072x768xf32>, tensor<3072xf32> -> tensor<1x128x3072xf32>
    %77 = dnn.gelu %76 {parameter_indices = array<i32: 1>, parameters = ["none"]} : (tensor<1x128x3072xf32>) -> tensor<1x128x3072xf32>
    %78 = dnn.linear %77, %cst_4, %cst_0 : tensor<1x128x3072xf32>, tensor<768x3072xf32>, tensor<768xf32> -> tensor<1x128x768xf32>
    %79 = dnn.add %78, %75 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<1x128x768xf32>, tensor<1x128x768xf32>) -> tensor<1x128x768xf32>
    %80 = dnn.layer_norm %79, %cst_0, %cst_0 {parameter_indices = array<i32: 1, 4, 5>, parameters = [[768], 9.9999999999999998E-13, false]} : (tensor<1x128x768xf32>, tensor<768xf32>, tensor<768xf32>) -> tensor<1x128x768xf32>
    %81 = dnn.linear %80, %cst_1, %cst_0 : tensor<1x128x768xf32>, tensor<768x768xf32>, tensor<768xf32> -> tensor<1x128x768xf32>
    %expanded_33 = tensor.expand_shape %81 [[0], [1], [2, 3]] output_shape [1, 128, 12, 64] : tensor<1x128x768xf32> into tensor<1x128x12x64xf32>
    %transposed_34 = linalg.transpose ins(%expanded_33 : tensor<1x128x12x64xf32>) outs(%10 : tensor<1x12x128x64xf32>) permutation = [0, 2, 1, 3] 
    %82 = dnn.scaled_dot_product_attention %transposed_34, %transposed_34, %transposed_34, %arg1 {parameter_indices = array<i32: 4, 5, 6, 7>, parameters = [0.000000e+00, false, 1.250000e-01, false]} : (tensor<1x12x128x64xf32>, tensor<1x12x128x64xf32>, tensor<1x12x128x64xf32>, tensor<1x1x128x128xi1>) -> tensor<1x12x128x64xf32>
    %transposed_35 = linalg.transpose ins(%82 : tensor<1x12x128x64xf32>) outs(%12 : tensor<1x128x12x64xf32>) permutation = [0, 2, 1, 3] 
    %collapsed_36 = tensor.collapse_shape %transposed_35 [[0], [1], [2, 3]] : tensor<1x128x12x64xf32> into tensor<1x128x768xf32>
    %83 = dnn.linear %collapsed_36, %cst_1, %cst_0 : tensor<1x128x768xf32>, tensor<768x768xf32>, tensor<768xf32> -> tensor<1x128x768xf32>
    %84 = dnn.add %83, %80 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<1x128x768xf32>, tensor<1x128x768xf32>) -> tensor<1x128x768xf32>
    %85 = dnn.layer_norm %84, %cst_0, %cst_0 {parameter_indices = array<i32: 1, 4, 5>, parameters = [[768], 9.9999999999999998E-13, false]} : (tensor<1x128x768xf32>, tensor<768xf32>, tensor<768xf32>) -> tensor<1x128x768xf32>
    %86 = dnn.linear %85, %cst_2, %cst_3 : tensor<1x128x768xf32>, tensor<3072x768xf32>, tensor<3072xf32> -> tensor<1x128x3072xf32>
    %87 = dnn.gelu %86 {parameter_indices = array<i32: 1>, parameters = ["none"]} : (tensor<1x128x3072xf32>) -> tensor<1x128x3072xf32>
    %88 = dnn.linear %87, %cst_4, %cst_0 : tensor<1x128x3072xf32>, tensor<768x3072xf32>, tensor<768xf32> -> tensor<1x128x768xf32>
    %89 = dnn.add %88, %85 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<1x128x768xf32>, tensor<1x128x768xf32>) -> tensor<1x128x768xf32>
    %90 = dnn.layer_norm %89, %cst_0, %cst_0 {parameter_indices = array<i32: 1, 4, 5>, parameters = [[768], 9.9999999999999998E-13, false]} : (tensor<1x128x768xf32>, tensor<768xf32>, tensor<768xf32>) -> tensor<1x128x768xf32>
    %91 = dnn.linear %90, %cst_1, %cst_0 : tensor<1x128x768xf32>, tensor<768x768xf32>, tensor<768xf32> -> tensor<1x128x768xf32>
    %expanded_37 = tensor.expand_shape %91 [[0], [1], [2, 3]] output_shape [1, 128, 12, 64] : tensor<1x128x768xf32> into tensor<1x128x12x64xf32>
    %transposed_38 = linalg.transpose ins(%expanded_37 : tensor<1x128x12x64xf32>) outs(%10 : tensor<1x12x128x64xf32>) permutation = [0, 2, 1, 3] 
    %92 = dnn.scaled_dot_product_attention %transposed_38, %transposed_38, %transposed_38, %arg1 {parameter_indices = array<i32: 4, 5, 6, 7>, parameters = [0.000000e+00, false, 1.250000e-01, false]} : (tensor<1x12x128x64xf32>, tensor<1x12x128x64xf32>, tensor<1x12x128x64xf32>, tensor<1x1x128x128xi1>) -> tensor<1x12x128x64xf32>
    %transposed_39 = linalg.transpose ins(%92 : tensor<1x12x128x64xf32>) outs(%12 : tensor<1x128x12x64xf32>) permutation = [0, 2, 1, 3] 
    %collapsed_40 = tensor.collapse_shape %transposed_39 [[0], [1], [2, 3]] : tensor<1x128x12x64xf32> into tensor<1x128x768xf32>
    %93 = dnn.linear %collapsed_40, %cst_1, %cst_0 : tensor<1x128x768xf32>, tensor<768x768xf32>, tensor<768xf32> -> tensor<1x128x768xf32>
    %94 = dnn.add %93, %90 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<1x128x768xf32>, tensor<1x128x768xf32>) -> tensor<1x128x768xf32>
    %95 = dnn.layer_norm %94, %cst_0, %cst_0 {parameter_indices = array<i32: 1, 4, 5>, parameters = [[768], 9.9999999999999998E-13, false]} : (tensor<1x128x768xf32>, tensor<768xf32>, tensor<768xf32>) -> tensor<1x128x768xf32>
    %96 = dnn.linear %95, %cst_2, %cst_3 : tensor<1x128x768xf32>, tensor<3072x768xf32>, tensor<3072xf32> -> tensor<1x128x3072xf32>
    %97 = dnn.gelu %96 {parameter_indices = array<i32: 1>, parameters = ["none"]} : (tensor<1x128x3072xf32>) -> tensor<1x128x3072xf32>
    %98 = dnn.linear %97, %cst_4, %cst_0 : tensor<1x128x3072xf32>, tensor<768x3072xf32>, tensor<768xf32> -> tensor<1x128x768xf32>
    %99 = dnn.add %98, %95 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<1x128x768xf32>, tensor<1x128x768xf32>) -> tensor<1x128x768xf32>
    %100 = dnn.layer_norm %99, %cst_0, %cst_0 {parameter_indices = array<i32: 1, 4, 5>, parameters = [[768], 9.9999999999999998E-13, false]} : (tensor<1x128x768xf32>, tensor<768xf32>, tensor<768xf32>) -> tensor<1x128x768xf32>
    %101 = dnn.linear %100, %cst_1, %cst_0 : tensor<1x128x768xf32>, tensor<768x768xf32>, tensor<768xf32> -> tensor<1x128x768xf32>
    %expanded_41 = tensor.expand_shape %101 [[0], [1], [2, 3]] output_shape [1, 128, 12, 64] : tensor<1x128x768xf32> into tensor<1x128x12x64xf32>
    %transposed_42 = linalg.transpose ins(%expanded_41 : tensor<1x128x12x64xf32>) outs(%10 : tensor<1x12x128x64xf32>) permutation = [0, 2, 1, 3] 
    %102 = dnn.scaled_dot_product_attention %transposed_42, %transposed_42, %transposed_42, %arg1 {parameter_indices = array<i32: 4, 5, 6, 7>, parameters = [0.000000e+00, false, 1.250000e-01, false]} : (tensor<1x12x128x64xf32>, tensor<1x12x128x64xf32>, tensor<1x12x128x64xf32>, tensor<1x1x128x128xi1>) -> tensor<1x12x128x64xf32>
    %transposed_43 = linalg.transpose ins(%102 : tensor<1x12x128x64xf32>) outs(%12 : tensor<1x128x12x64xf32>) permutation = [0, 2, 1, 3] 
    %collapsed_44 = tensor.collapse_shape %transposed_43 [[0], [1], [2, 3]] : tensor<1x128x12x64xf32> into tensor<1x128x768xf32>
    %103 = dnn.linear %collapsed_44, %cst_1, %cst_0 : tensor<1x128x768xf32>, tensor<768x768xf32>, tensor<768xf32> -> tensor<1x128x768xf32>
    %104 = dnn.add %103, %100 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<1x128x768xf32>, tensor<1x128x768xf32>) -> tensor<1x128x768xf32>
    %105 = dnn.layer_norm %104, %cst_0, %cst_0 {parameter_indices = array<i32: 1, 4, 5>, parameters = [[768], 9.9999999999999998E-13, false]} : (tensor<1x128x768xf32>, tensor<768xf32>, tensor<768xf32>) -> tensor<1x128x768xf32>
    %106 = dnn.linear %105, %cst_2, %cst_3 : tensor<1x128x768xf32>, tensor<3072x768xf32>, tensor<3072xf32> -> tensor<1x128x3072xf32>
    %107 = dnn.gelu %106 {parameter_indices = array<i32: 1>, parameters = ["none"]} : (tensor<1x128x3072xf32>) -> tensor<1x128x3072xf32>
    %108 = dnn.linear %107, %cst_4, %cst_0 : tensor<1x128x3072xf32>, tensor<768x3072xf32>, tensor<768xf32> -> tensor<1x128x768xf32>
    %109 = dnn.add %108, %105 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<1x128x768xf32>, tensor<1x128x768xf32>) -> tensor<1x128x768xf32>
    %110 = dnn.layer_norm %109, %cst_0, %cst_0 {parameter_indices = array<i32: 1, 4, 5>, parameters = [[768], 9.9999999999999998E-13, false]} : (tensor<1x128x768xf32>, tensor<768xf32>, tensor<768xf32>) -> tensor<1x128x768xf32>
    %111 = dnn.linear %110, %cst_1, %cst_0 : tensor<1x128x768xf32>, tensor<768x768xf32>, tensor<768xf32> -> tensor<1x128x768xf32>
    %expanded_45 = tensor.expand_shape %111 [[0], [1], [2, 3]] output_shape [1, 128, 12, 64] : tensor<1x128x768xf32> into tensor<1x128x12x64xf32>
    %transposed_46 = linalg.transpose ins(%expanded_45 : tensor<1x128x12x64xf32>) outs(%10 : tensor<1x12x128x64xf32>) permutation = [0, 2, 1, 3] 
    %112 = dnn.scaled_dot_product_attention %transposed_46, %transposed_46, %transposed_46, %arg1 {parameter_indices = array<i32: 4, 5, 6, 7>, parameters = [0.000000e+00, false, 1.250000e-01, false]} : (tensor<1x12x128x64xf32>, tensor<1x12x128x64xf32>, tensor<1x12x128x64xf32>, tensor<1x1x128x128xi1>) -> tensor<1x12x128x64xf32>
    %transposed_47 = linalg.transpose ins(%112 : tensor<1x12x128x64xf32>) outs(%12 : tensor<1x128x12x64xf32>) permutation = [0, 2, 1, 3] 
    %collapsed_48 = tensor.collapse_shape %transposed_47 [[0], [1], [2, 3]] : tensor<1x128x12x64xf32> into tensor<1x128x768xf32>
    %113 = dnn.linear %collapsed_48, %cst_1, %cst_0 : tensor<1x128x768xf32>, tensor<768x768xf32>, tensor<768xf32> -> tensor<1x128x768xf32>
    %114 = dnn.add %113, %110 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<1x128x768xf32>, tensor<1x128x768xf32>) -> tensor<1x128x768xf32>
    %115 = dnn.layer_norm %114, %cst_0, %cst_0 {parameter_indices = array<i32: 1, 4, 5>, parameters = [[768], 9.9999999999999998E-13, false]} : (tensor<1x128x768xf32>, tensor<768xf32>, tensor<768xf32>) -> tensor<1x128x768xf32>
    %116 = dnn.linear %115, %cst_2, %cst_3 : tensor<1x128x768xf32>, tensor<3072x768xf32>, tensor<3072xf32> -> tensor<1x128x3072xf32>
    %117 = dnn.gelu %116 {parameter_indices = array<i32: 1>, parameters = ["none"]} : (tensor<1x128x3072xf32>) -> tensor<1x128x3072xf32>
    %118 = dnn.linear %117, %cst_4, %cst_0 : tensor<1x128x3072xf32>, tensor<768x3072xf32>, tensor<768xf32> -> tensor<1x128x768xf32>
    %119 = dnn.add %118, %115 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<1x128x768xf32>, tensor<1x128x768xf32>) -> tensor<1x128x768xf32>
    %120 = dnn.layer_norm %119, %cst_0, %cst_0 {parameter_indices = array<i32: 1, 4, 5>, parameters = [[768], 9.9999999999999998E-13, false]} : (tensor<1x128x768xf32>, tensor<768xf32>, tensor<768xf32>) -> tensor<1x128x768xf32>
    %121 = dnn.linear %120, %cst_1, %cst_0 : tensor<1x128x768xf32>, tensor<768x768xf32>, tensor<768xf32> -> tensor<1x128x768xf32>
    %expanded_49 = tensor.expand_shape %121 [[0], [1], [2, 3]] output_shape [1, 128, 12, 64] : tensor<1x128x768xf32> into tensor<1x128x12x64xf32>
    %transposed_50 = linalg.transpose ins(%expanded_49 : tensor<1x128x12x64xf32>) outs(%10 : tensor<1x12x128x64xf32>) permutation = [0, 2, 1, 3] 
    %122 = dnn.scaled_dot_product_attention %transposed_50, %transposed_50, %transposed_50, %arg1 {parameter_indices = array<i32: 4, 5, 6, 7>, parameters = [0.000000e+00, false, 1.250000e-01, false]} : (tensor<1x12x128x64xf32>, tensor<1x12x128x64xf32>, tensor<1x12x128x64xf32>, tensor<1x1x128x128xi1>) -> tensor<1x12x128x64xf32>
    %transposed_51 = linalg.transpose ins(%122 : tensor<1x12x128x64xf32>) outs(%12 : tensor<1x128x12x64xf32>) permutation = [0, 2, 1, 3] 
    %collapsed_52 = tensor.collapse_shape %transposed_51 [[0], [1], [2, 3]] : tensor<1x128x12x64xf32> into tensor<1x128x768xf32>
    %123 = dnn.linear %collapsed_52, %cst_1, %cst_0 : tensor<1x128x768xf32>, tensor<768x768xf32>, tensor<768xf32> -> tensor<1x128x768xf32>
    %124 = dnn.add %123, %120 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<1x128x768xf32>, tensor<1x128x768xf32>) -> tensor<1x128x768xf32>
    %125 = dnn.layer_norm %124, %cst_0, %cst_0 {parameter_indices = array<i32: 1, 4, 5>, parameters = [[768], 9.9999999999999998E-13, false]} : (tensor<1x128x768xf32>, tensor<768xf32>, tensor<768xf32>) -> tensor<1x128x768xf32>
    %126 = dnn.linear %125, %cst_2, %cst_3 : tensor<1x128x768xf32>, tensor<3072x768xf32>, tensor<3072xf32> -> tensor<1x128x3072xf32>
    %127 = dnn.gelu %126 {parameter_indices = array<i32: 1>, parameters = ["none"]} : (tensor<1x128x3072xf32>) -> tensor<1x128x3072xf32>
    %128 = dnn.linear %127, %cst_4, %cst_0 : tensor<1x128x3072xf32>, tensor<768x3072xf32>, tensor<768xf32> -> tensor<1x128x768xf32>
    %129 = dnn.add %128, %125 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<1x128x768xf32>, tensor<1x128x768xf32>) -> tensor<1x128x768xf32>
    %130 = dnn.layer_norm %129, %cst_0, %cst_0 {parameter_indices = array<i32: 1, 4, 5>, parameters = [[768], 9.9999999999999998E-13, false]} : (tensor<1x128x768xf32>, tensor<768xf32>, tensor<768xf32>) -> tensor<1x128x768xf32>
    %extracted_slice = tensor.extract_slice %130[0, 0, 0] [1, 1, 768] [1, 1, 1] : tensor<1x128x768xf32> to tensor<1x1x768xf32>
    %collapsed_53 = tensor.collapse_shape %extracted_slice [[0, 1], [2]] : tensor<1x1x768xf32> into tensor<1x768xf32>
    %131 = dnn.linear %collapsed_53, %cst_1, %cst_0 : tensor<1x768xf32>, tensor<768x768xf32>, tensor<768xf32> -> tensor<1x768xf32>
    %132 = dnn.tanh %131 : (tensor<1x768xf32>) -> tensor<1x768xf32>
    return %130 : tensor<1x128x768xf32>
  }
}

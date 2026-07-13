// RUN: dnn-mlir-opt %s | FileCheck %s
//
// GPT-2 Small after captures=all and the remaining Torch-to-Linalg backend
// lowering. Neural-network operations remain in DNN, while layout operations
// use standard tensor and Linalg named operations. Decomposed FX LayerNorm and
// gelu_new subgraphs have been reconstructed before backend lowering.
//
// CHECK-LABEL: func.func @gpt2_small
// CHECK-DAG: dnn.embedding
// CHECK-DAG: dnn.mm
// CHECK-DAG: dnn.matmul
// CHECK-DAG: dnn.add
// CHECK-DAG: dnn.layer_norm
// CHECK-DAG: dnn.gelu
// CHECK-DAG: dnn.scaled_dot_product_attention
// CHECK-DAG: linalg.transpose
// CHECK-NOT: linalg.generic
// CHECK-NOT: linalg.batch_matmul
// CHECK-NOT: torch.

module {
  func.func @gpt2_small(%arg0: tensor<1x128xi64>, %arg1: tensor<1x128xi64>, %arg2: tensor<1x1x128x128xi1>) -> tensor<1x128x50257xf32> {
    %cst = arith.constant dense<0.000000e+00> : tensor<50257x768xf32>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<1024x768xf32>
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<768xf32>
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<768x2304xf32>
    %cst_3 = arith.constant dense<0.000000e+00> : tensor<2304xf32>
    %cst_4 = arith.constant dense<0.000000e+00> : tensor<768x768xf32>
    %cst_5 = arith.constant dense<0.000000e+00> : tensor<768x3072xf32>
    %cst_6 = arith.constant dense<0.000000e+00> : tensor<3072xf32>
    %cst_7 = arith.constant dense<0.000000e+00> : tensor<3072x768xf32>
    %cst_8 = arith.constant dense<0.000000e+00> : tensor<768x50257xf32>
    %0 = dnn.embedding %cst, %arg0 {parameter_indices = array<i32: 2, 3, 4>, parameters = [-1, false, false]} : (tensor<50257x768xf32>, tensor<1x128xi64>) -> tensor<1x128x768xf32>
    %1 = dnn.embedding %cst_0, %arg1 {parameter_indices = array<i32: 2, 3, 4>, parameters = [-1, false, false]} : (tensor<1024x768xf32>, tensor<1x128xi64>) -> tensor<1x128x768xf32>
    %2 = dnn.add %0, %1 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<1x128x768xf32>, tensor<1x128x768xf32>) -> tensor<1x128x768xf32>
    %3 = dnn.layer_norm %2, %cst_1, %cst_1 {parameter_indices = array<i32: 1, 4, 5>, parameters = [[768], 1.000000e-05, false]} : (tensor<1x128x768xf32>, tensor<768xf32>, tensor<768xf32>) -> tensor<1x128x768xf32>
    %collapsed = tensor.collapse_shape %3 [[0, 1], [2]] : tensor<1x128x768xf32> into tensor<128x768xf32>
    %4 = dnn.mm %collapsed, %cst_2 : tensor<128x768xf32>, tensor<768x2304xf32> -> tensor<128x2304xf32>
    %5 = dnn.add %4, %cst_3 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<128x2304xf32>, tensor<2304xf32>) -> tensor<128x2304xf32>
    %expanded = tensor.expand_shape %5 [[0, 1], [2]] output_shape [1, 128, 2304] : tensor<128x2304xf32> into tensor<1x128x2304xf32>
    %extracted_slice = tensor.extract_slice %expanded[0, 0, 0] [1, 128, 768] [1, 1, 1] : tensor<1x128x2304xf32> to tensor<1x128x768xf32>
    %extracted_slice_9 = tensor.extract_slice %expanded[0, 0, 768] [1, 128, 768] [1, 1, 1] : tensor<1x128x2304xf32> to tensor<1x128x768xf32>
    %extracted_slice_10 = tensor.extract_slice %expanded[0, 0, 1536] [1, 128, 768] [1, 1, 1] : tensor<1x128x2304xf32> to tensor<1x128x768xf32>
    %expanded_11 = tensor.expand_shape %extracted_slice_9 [[0], [1], [2, 3]] output_shape [1, 128, 12, 64] : tensor<1x128x768xf32> into tensor<1x128x12x64xf32>
    %6 = tensor.empty() : tensor<1x12x128x64xf32>
    %transposed = linalg.transpose ins(%expanded_11 : tensor<1x128x12x64xf32>) outs(%6 : tensor<1x12x128x64xf32>) permutation = [0, 2, 1, 3] 
    %expanded_12 = tensor.expand_shape %extracted_slice_10 [[0], [1], [2, 3]] output_shape [1, 128, 12, 64] : tensor<1x128x768xf32> into tensor<1x128x12x64xf32>
    %transposed_13 = linalg.transpose ins(%expanded_12 : tensor<1x128x12x64xf32>) outs(%6 : tensor<1x12x128x64xf32>) permutation = [0, 2, 1, 3] 
    %expanded_14 = tensor.expand_shape %extracted_slice [[0], [1], [2, 3]] output_shape [1, 128, 12, 64] : tensor<1x128x768xf32> into tensor<1x128x12x64xf32>
    %transposed_15 = linalg.transpose ins(%expanded_14 : tensor<1x128x12x64xf32>) outs(%6 : tensor<1x12x128x64xf32>) permutation = [0, 2, 1, 3] 
    %7 = dnn.scaled_dot_product_attention %transposed_15, %transposed, %transposed_13, %arg2 {parameter_indices = array<i32: 4, 5, 6, 7>, parameters = [0.000000e+00, false, 1.250000e-01, false]} : (tensor<1x12x128x64xf32>, tensor<1x12x128x64xf32>, tensor<1x12x128x64xf32>, tensor<1x1x128x128xi1>) -> tensor<1x12x128x64xf32>
    %8 = tensor.empty() : tensor<1x128x12x64xf32>
    %transposed_16 = linalg.transpose ins(%7 : tensor<1x12x128x64xf32>) outs(%8 : tensor<1x128x12x64xf32>) permutation = [0, 2, 1, 3] 
    %collapsed_17 = tensor.collapse_shape %transposed_16 [[0, 1], [2, 3]] : tensor<1x128x12x64xf32> into tensor<128x768xf32>
    %9 = dnn.mm %collapsed_17, %cst_4 : tensor<128x768xf32>, tensor<768x768xf32> -> tensor<128x768xf32>
    %10 = dnn.add %9, %cst_1 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<128x768xf32>, tensor<768xf32>) -> tensor<128x768xf32>
    %expanded_18 = tensor.expand_shape %10 [[0, 1], [2]] output_shape [1, 128, 768] : tensor<128x768xf32> into tensor<1x128x768xf32>
    %11 = dnn.add %expanded_18, %2 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<1x128x768xf32>, tensor<1x128x768xf32>) -> tensor<1x128x768xf32>
    %12 = dnn.layer_norm %11, %cst_1, %cst_1 {parameter_indices = array<i32: 1, 4, 5>, parameters = [[768], 1.000000e-05, false]} : (tensor<1x128x768xf32>, tensor<768xf32>, tensor<768xf32>) -> tensor<1x128x768xf32>
    %collapsed_19 = tensor.collapse_shape %12 [[0, 1], [2]] : tensor<1x128x768xf32> into tensor<128x768xf32>
    %13 = dnn.mm %collapsed_19, %cst_5 : tensor<128x768xf32>, tensor<768x3072xf32> -> tensor<128x3072xf32>
    %14 = dnn.add %13, %cst_6 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<128x3072xf32>, tensor<3072xf32>) -> tensor<128x3072xf32>
    %expanded_20 = tensor.expand_shape %14 [[0, 1], [2]] output_shape [1, 128, 3072] : tensor<128x3072xf32> into tensor<1x128x3072xf32>
    %15 = dnn.gelu %expanded_20 {parameter_indices = array<i32: 1>, parameters = ["tanh"]} : (tensor<1x128x3072xf32>) -> tensor<1x128x3072xf32>
    %collapsed_21 = tensor.collapse_shape %15 [[0, 1], [2]] : tensor<1x128x3072xf32> into tensor<128x3072xf32>
    %16 = dnn.mm %collapsed_21, %cst_7 : tensor<128x3072xf32>, tensor<3072x768xf32> -> tensor<128x768xf32>
    %17 = dnn.add %16, %cst_1 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<128x768xf32>, tensor<768xf32>) -> tensor<128x768xf32>
    %expanded_22 = tensor.expand_shape %17 [[0, 1], [2]] output_shape [1, 128, 768] : tensor<128x768xf32> into tensor<1x128x768xf32>
    %18 = dnn.add %11, %expanded_22 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<1x128x768xf32>, tensor<1x128x768xf32>) -> tensor<1x128x768xf32>
    %19 = dnn.layer_norm %18, %cst_1, %cst_1 {parameter_indices = array<i32: 1, 4, 5>, parameters = [[768], 1.000000e-05, false]} : (tensor<1x128x768xf32>, tensor<768xf32>, tensor<768xf32>) -> tensor<1x128x768xf32>
    %collapsed_23 = tensor.collapse_shape %19 [[0, 1], [2]] : tensor<1x128x768xf32> into tensor<128x768xf32>
    %20 = dnn.mm %collapsed_23, %cst_2 : tensor<128x768xf32>, tensor<768x2304xf32> -> tensor<128x2304xf32>
    %21 = dnn.add %20, %cst_3 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<128x2304xf32>, tensor<2304xf32>) -> tensor<128x2304xf32>
    %expanded_24 = tensor.expand_shape %21 [[0, 1], [2]] output_shape [1, 128, 2304] : tensor<128x2304xf32> into tensor<1x128x2304xf32>
    %extracted_slice_25 = tensor.extract_slice %expanded_24[0, 0, 0] [1, 128, 768] [1, 1, 1] : tensor<1x128x2304xf32> to tensor<1x128x768xf32>
    %extracted_slice_26 = tensor.extract_slice %expanded_24[0, 0, 768] [1, 128, 768] [1, 1, 1] : tensor<1x128x2304xf32> to tensor<1x128x768xf32>
    %extracted_slice_27 = tensor.extract_slice %expanded_24[0, 0, 1536] [1, 128, 768] [1, 1, 1] : tensor<1x128x2304xf32> to tensor<1x128x768xf32>
    %expanded_28 = tensor.expand_shape %extracted_slice_26 [[0], [1], [2, 3]] output_shape [1, 128, 12, 64] : tensor<1x128x768xf32> into tensor<1x128x12x64xf32>
    %transposed_29 = linalg.transpose ins(%expanded_28 : tensor<1x128x12x64xf32>) outs(%6 : tensor<1x12x128x64xf32>) permutation = [0, 2, 1, 3] 
    %expanded_30 = tensor.expand_shape %extracted_slice_27 [[0], [1], [2, 3]] output_shape [1, 128, 12, 64] : tensor<1x128x768xf32> into tensor<1x128x12x64xf32>
    %transposed_31 = linalg.transpose ins(%expanded_30 : tensor<1x128x12x64xf32>) outs(%6 : tensor<1x12x128x64xf32>) permutation = [0, 2, 1, 3] 
    %expanded_32 = tensor.expand_shape %extracted_slice_25 [[0], [1], [2, 3]] output_shape [1, 128, 12, 64] : tensor<1x128x768xf32> into tensor<1x128x12x64xf32>
    %transposed_33 = linalg.transpose ins(%expanded_32 : tensor<1x128x12x64xf32>) outs(%6 : tensor<1x12x128x64xf32>) permutation = [0, 2, 1, 3] 
    %22 = dnn.scaled_dot_product_attention %transposed_33, %transposed_29, %transposed_31, %arg2 {parameter_indices = array<i32: 4, 5, 6, 7>, parameters = [0.000000e+00, false, 1.250000e-01, false]} : (tensor<1x12x128x64xf32>, tensor<1x12x128x64xf32>, tensor<1x12x128x64xf32>, tensor<1x1x128x128xi1>) -> tensor<1x12x128x64xf32>
    %transposed_34 = linalg.transpose ins(%22 : tensor<1x12x128x64xf32>) outs(%8 : tensor<1x128x12x64xf32>) permutation = [0, 2, 1, 3] 
    %collapsed_35 = tensor.collapse_shape %transposed_34 [[0, 1], [2, 3]] : tensor<1x128x12x64xf32> into tensor<128x768xf32>
    %23 = dnn.mm %collapsed_35, %cst_4 : tensor<128x768xf32>, tensor<768x768xf32> -> tensor<128x768xf32>
    %24 = dnn.add %23, %cst_1 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<128x768xf32>, tensor<768xf32>) -> tensor<128x768xf32>
    %expanded_36 = tensor.expand_shape %24 [[0, 1], [2]] output_shape [1, 128, 768] : tensor<128x768xf32> into tensor<1x128x768xf32>
    %25 = dnn.add %expanded_36, %18 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<1x128x768xf32>, tensor<1x128x768xf32>) -> tensor<1x128x768xf32>
    %26 = dnn.layer_norm %25, %cst_1, %cst_1 {parameter_indices = array<i32: 1, 4, 5>, parameters = [[768], 1.000000e-05, false]} : (tensor<1x128x768xf32>, tensor<768xf32>, tensor<768xf32>) -> tensor<1x128x768xf32>
    %collapsed_37 = tensor.collapse_shape %26 [[0, 1], [2]] : tensor<1x128x768xf32> into tensor<128x768xf32>
    %27 = dnn.mm %collapsed_37, %cst_5 : tensor<128x768xf32>, tensor<768x3072xf32> -> tensor<128x3072xf32>
    %28 = dnn.add %27, %cst_6 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<128x3072xf32>, tensor<3072xf32>) -> tensor<128x3072xf32>
    %expanded_38 = tensor.expand_shape %28 [[0, 1], [2]] output_shape [1, 128, 3072] : tensor<128x3072xf32> into tensor<1x128x3072xf32>
    %29 = dnn.gelu %expanded_38 {parameter_indices = array<i32: 1>, parameters = ["tanh"]} : (tensor<1x128x3072xf32>) -> tensor<1x128x3072xf32>
    %collapsed_39 = tensor.collapse_shape %29 [[0, 1], [2]] : tensor<1x128x3072xf32> into tensor<128x3072xf32>
    %30 = dnn.mm %collapsed_39, %cst_7 : tensor<128x3072xf32>, tensor<3072x768xf32> -> tensor<128x768xf32>
    %31 = dnn.add %30, %cst_1 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<128x768xf32>, tensor<768xf32>) -> tensor<128x768xf32>
    %expanded_40 = tensor.expand_shape %31 [[0, 1], [2]] output_shape [1, 128, 768] : tensor<128x768xf32> into tensor<1x128x768xf32>
    %32 = dnn.add %25, %expanded_40 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<1x128x768xf32>, tensor<1x128x768xf32>) -> tensor<1x128x768xf32>
    %33 = dnn.layer_norm %32, %cst_1, %cst_1 {parameter_indices = array<i32: 1, 4, 5>, parameters = [[768], 1.000000e-05, false]} : (tensor<1x128x768xf32>, tensor<768xf32>, tensor<768xf32>) -> tensor<1x128x768xf32>
    %collapsed_41 = tensor.collapse_shape %33 [[0, 1], [2]] : tensor<1x128x768xf32> into tensor<128x768xf32>
    %34 = dnn.mm %collapsed_41, %cst_2 : tensor<128x768xf32>, tensor<768x2304xf32> -> tensor<128x2304xf32>
    %35 = dnn.add %34, %cst_3 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<128x2304xf32>, tensor<2304xf32>) -> tensor<128x2304xf32>
    %expanded_42 = tensor.expand_shape %35 [[0, 1], [2]] output_shape [1, 128, 2304] : tensor<128x2304xf32> into tensor<1x128x2304xf32>
    %extracted_slice_43 = tensor.extract_slice %expanded_42[0, 0, 0] [1, 128, 768] [1, 1, 1] : tensor<1x128x2304xf32> to tensor<1x128x768xf32>
    %extracted_slice_44 = tensor.extract_slice %expanded_42[0, 0, 768] [1, 128, 768] [1, 1, 1] : tensor<1x128x2304xf32> to tensor<1x128x768xf32>
    %extracted_slice_45 = tensor.extract_slice %expanded_42[0, 0, 1536] [1, 128, 768] [1, 1, 1] : tensor<1x128x2304xf32> to tensor<1x128x768xf32>
    %expanded_46 = tensor.expand_shape %extracted_slice_44 [[0], [1], [2, 3]] output_shape [1, 128, 12, 64] : tensor<1x128x768xf32> into tensor<1x128x12x64xf32>
    %transposed_47 = linalg.transpose ins(%expanded_46 : tensor<1x128x12x64xf32>) outs(%6 : tensor<1x12x128x64xf32>) permutation = [0, 2, 1, 3] 
    %expanded_48 = tensor.expand_shape %extracted_slice_45 [[0], [1], [2, 3]] output_shape [1, 128, 12, 64] : tensor<1x128x768xf32> into tensor<1x128x12x64xf32>
    %transposed_49 = linalg.transpose ins(%expanded_48 : tensor<1x128x12x64xf32>) outs(%6 : tensor<1x12x128x64xf32>) permutation = [0, 2, 1, 3] 
    %expanded_50 = tensor.expand_shape %extracted_slice_43 [[0], [1], [2, 3]] output_shape [1, 128, 12, 64] : tensor<1x128x768xf32> into tensor<1x128x12x64xf32>
    %transposed_51 = linalg.transpose ins(%expanded_50 : tensor<1x128x12x64xf32>) outs(%6 : tensor<1x12x128x64xf32>) permutation = [0, 2, 1, 3] 
    %36 = dnn.scaled_dot_product_attention %transposed_51, %transposed_47, %transposed_49, %arg2 {parameter_indices = array<i32: 4, 5, 6, 7>, parameters = [0.000000e+00, false, 1.250000e-01, false]} : (tensor<1x12x128x64xf32>, tensor<1x12x128x64xf32>, tensor<1x12x128x64xf32>, tensor<1x1x128x128xi1>) -> tensor<1x12x128x64xf32>
    %transposed_52 = linalg.transpose ins(%36 : tensor<1x12x128x64xf32>) outs(%8 : tensor<1x128x12x64xf32>) permutation = [0, 2, 1, 3] 
    %collapsed_53 = tensor.collapse_shape %transposed_52 [[0, 1], [2, 3]] : tensor<1x128x12x64xf32> into tensor<128x768xf32>
    %37 = dnn.mm %collapsed_53, %cst_4 : tensor<128x768xf32>, tensor<768x768xf32> -> tensor<128x768xf32>
    %38 = dnn.add %37, %cst_1 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<128x768xf32>, tensor<768xf32>) -> tensor<128x768xf32>
    %expanded_54 = tensor.expand_shape %38 [[0, 1], [2]] output_shape [1, 128, 768] : tensor<128x768xf32> into tensor<1x128x768xf32>
    %39 = dnn.add %expanded_54, %32 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<1x128x768xf32>, tensor<1x128x768xf32>) -> tensor<1x128x768xf32>
    %40 = dnn.layer_norm %39, %cst_1, %cst_1 {parameter_indices = array<i32: 1, 4, 5>, parameters = [[768], 1.000000e-05, false]} : (tensor<1x128x768xf32>, tensor<768xf32>, tensor<768xf32>) -> tensor<1x128x768xf32>
    %collapsed_55 = tensor.collapse_shape %40 [[0, 1], [2]] : tensor<1x128x768xf32> into tensor<128x768xf32>
    %41 = dnn.mm %collapsed_55, %cst_5 : tensor<128x768xf32>, tensor<768x3072xf32> -> tensor<128x3072xf32>
    %42 = dnn.add %41, %cst_6 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<128x3072xf32>, tensor<3072xf32>) -> tensor<128x3072xf32>
    %expanded_56 = tensor.expand_shape %42 [[0, 1], [2]] output_shape [1, 128, 3072] : tensor<128x3072xf32> into tensor<1x128x3072xf32>
    %43 = dnn.gelu %expanded_56 {parameter_indices = array<i32: 1>, parameters = ["tanh"]} : (tensor<1x128x3072xf32>) -> tensor<1x128x3072xf32>
    %collapsed_57 = tensor.collapse_shape %43 [[0, 1], [2]] : tensor<1x128x3072xf32> into tensor<128x3072xf32>
    %44 = dnn.mm %collapsed_57, %cst_7 : tensor<128x3072xf32>, tensor<3072x768xf32> -> tensor<128x768xf32>
    %45 = dnn.add %44, %cst_1 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<128x768xf32>, tensor<768xf32>) -> tensor<128x768xf32>
    %expanded_58 = tensor.expand_shape %45 [[0, 1], [2]] output_shape [1, 128, 768] : tensor<128x768xf32> into tensor<1x128x768xf32>
    %46 = dnn.add %39, %expanded_58 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<1x128x768xf32>, tensor<1x128x768xf32>) -> tensor<1x128x768xf32>
    %47 = dnn.layer_norm %46, %cst_1, %cst_1 {parameter_indices = array<i32: 1, 4, 5>, parameters = [[768], 1.000000e-05, false]} : (tensor<1x128x768xf32>, tensor<768xf32>, tensor<768xf32>) -> tensor<1x128x768xf32>
    %collapsed_59 = tensor.collapse_shape %47 [[0, 1], [2]] : tensor<1x128x768xf32> into tensor<128x768xf32>
    %48 = dnn.mm %collapsed_59, %cst_2 : tensor<128x768xf32>, tensor<768x2304xf32> -> tensor<128x2304xf32>
    %49 = dnn.add %48, %cst_3 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<128x2304xf32>, tensor<2304xf32>) -> tensor<128x2304xf32>
    %expanded_60 = tensor.expand_shape %49 [[0, 1], [2]] output_shape [1, 128, 2304] : tensor<128x2304xf32> into tensor<1x128x2304xf32>
    %extracted_slice_61 = tensor.extract_slice %expanded_60[0, 0, 0] [1, 128, 768] [1, 1, 1] : tensor<1x128x2304xf32> to tensor<1x128x768xf32>
    %extracted_slice_62 = tensor.extract_slice %expanded_60[0, 0, 768] [1, 128, 768] [1, 1, 1] : tensor<1x128x2304xf32> to tensor<1x128x768xf32>
    %extracted_slice_63 = tensor.extract_slice %expanded_60[0, 0, 1536] [1, 128, 768] [1, 1, 1] : tensor<1x128x2304xf32> to tensor<1x128x768xf32>
    %expanded_64 = tensor.expand_shape %extracted_slice_62 [[0], [1], [2, 3]] output_shape [1, 128, 12, 64] : tensor<1x128x768xf32> into tensor<1x128x12x64xf32>
    %transposed_65 = linalg.transpose ins(%expanded_64 : tensor<1x128x12x64xf32>) outs(%6 : tensor<1x12x128x64xf32>) permutation = [0, 2, 1, 3] 
    %expanded_66 = tensor.expand_shape %extracted_slice_63 [[0], [1], [2, 3]] output_shape [1, 128, 12, 64] : tensor<1x128x768xf32> into tensor<1x128x12x64xf32>
    %transposed_67 = linalg.transpose ins(%expanded_66 : tensor<1x128x12x64xf32>) outs(%6 : tensor<1x12x128x64xf32>) permutation = [0, 2, 1, 3] 
    %expanded_68 = tensor.expand_shape %extracted_slice_61 [[0], [1], [2, 3]] output_shape [1, 128, 12, 64] : tensor<1x128x768xf32> into tensor<1x128x12x64xf32>
    %transposed_69 = linalg.transpose ins(%expanded_68 : tensor<1x128x12x64xf32>) outs(%6 : tensor<1x12x128x64xf32>) permutation = [0, 2, 1, 3] 
    %50 = dnn.scaled_dot_product_attention %transposed_69, %transposed_65, %transposed_67, %arg2 {parameter_indices = array<i32: 4, 5, 6, 7>, parameters = [0.000000e+00, false, 1.250000e-01, false]} : (tensor<1x12x128x64xf32>, tensor<1x12x128x64xf32>, tensor<1x12x128x64xf32>, tensor<1x1x128x128xi1>) -> tensor<1x12x128x64xf32>
    %transposed_70 = linalg.transpose ins(%50 : tensor<1x12x128x64xf32>) outs(%8 : tensor<1x128x12x64xf32>) permutation = [0, 2, 1, 3] 
    %collapsed_71 = tensor.collapse_shape %transposed_70 [[0, 1], [2, 3]] : tensor<1x128x12x64xf32> into tensor<128x768xf32>
    %51 = dnn.mm %collapsed_71, %cst_4 : tensor<128x768xf32>, tensor<768x768xf32> -> tensor<128x768xf32>
    %52 = dnn.add %51, %cst_1 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<128x768xf32>, tensor<768xf32>) -> tensor<128x768xf32>
    %expanded_72 = tensor.expand_shape %52 [[0, 1], [2]] output_shape [1, 128, 768] : tensor<128x768xf32> into tensor<1x128x768xf32>
    %53 = dnn.add %expanded_72, %46 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<1x128x768xf32>, tensor<1x128x768xf32>) -> tensor<1x128x768xf32>
    %54 = dnn.layer_norm %53, %cst_1, %cst_1 {parameter_indices = array<i32: 1, 4, 5>, parameters = [[768], 1.000000e-05, false]} : (tensor<1x128x768xf32>, tensor<768xf32>, tensor<768xf32>) -> tensor<1x128x768xf32>
    %collapsed_73 = tensor.collapse_shape %54 [[0, 1], [2]] : tensor<1x128x768xf32> into tensor<128x768xf32>
    %55 = dnn.mm %collapsed_73, %cst_5 : tensor<128x768xf32>, tensor<768x3072xf32> -> tensor<128x3072xf32>
    %56 = dnn.add %55, %cst_6 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<128x3072xf32>, tensor<3072xf32>) -> tensor<128x3072xf32>
    %expanded_74 = tensor.expand_shape %56 [[0, 1], [2]] output_shape [1, 128, 3072] : tensor<128x3072xf32> into tensor<1x128x3072xf32>
    %57 = dnn.gelu %expanded_74 {parameter_indices = array<i32: 1>, parameters = ["tanh"]} : (tensor<1x128x3072xf32>) -> tensor<1x128x3072xf32>
    %collapsed_75 = tensor.collapse_shape %57 [[0, 1], [2]] : tensor<1x128x3072xf32> into tensor<128x3072xf32>
    %58 = dnn.mm %collapsed_75, %cst_7 : tensor<128x3072xf32>, tensor<3072x768xf32> -> tensor<128x768xf32>
    %59 = dnn.add %58, %cst_1 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<128x768xf32>, tensor<768xf32>) -> tensor<128x768xf32>
    %expanded_76 = tensor.expand_shape %59 [[0, 1], [2]] output_shape [1, 128, 768] : tensor<128x768xf32> into tensor<1x128x768xf32>
    %60 = dnn.add %53, %expanded_76 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<1x128x768xf32>, tensor<1x128x768xf32>) -> tensor<1x128x768xf32>
    %61 = dnn.layer_norm %60, %cst_1, %cst_1 {parameter_indices = array<i32: 1, 4, 5>, parameters = [[768], 1.000000e-05, false]} : (tensor<1x128x768xf32>, tensor<768xf32>, tensor<768xf32>) -> tensor<1x128x768xf32>
    %collapsed_77 = tensor.collapse_shape %61 [[0, 1], [2]] : tensor<1x128x768xf32> into tensor<128x768xf32>
    %62 = dnn.mm %collapsed_77, %cst_2 : tensor<128x768xf32>, tensor<768x2304xf32> -> tensor<128x2304xf32>
    %63 = dnn.add %62, %cst_3 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<128x2304xf32>, tensor<2304xf32>) -> tensor<128x2304xf32>
    %expanded_78 = tensor.expand_shape %63 [[0, 1], [2]] output_shape [1, 128, 2304] : tensor<128x2304xf32> into tensor<1x128x2304xf32>
    %extracted_slice_79 = tensor.extract_slice %expanded_78[0, 0, 0] [1, 128, 768] [1, 1, 1] : tensor<1x128x2304xf32> to tensor<1x128x768xf32>
    %extracted_slice_80 = tensor.extract_slice %expanded_78[0, 0, 768] [1, 128, 768] [1, 1, 1] : tensor<1x128x2304xf32> to tensor<1x128x768xf32>
    %extracted_slice_81 = tensor.extract_slice %expanded_78[0, 0, 1536] [1, 128, 768] [1, 1, 1] : tensor<1x128x2304xf32> to tensor<1x128x768xf32>
    %expanded_82 = tensor.expand_shape %extracted_slice_80 [[0], [1], [2, 3]] output_shape [1, 128, 12, 64] : tensor<1x128x768xf32> into tensor<1x128x12x64xf32>
    %transposed_83 = linalg.transpose ins(%expanded_82 : tensor<1x128x12x64xf32>) outs(%6 : tensor<1x12x128x64xf32>) permutation = [0, 2, 1, 3] 
    %expanded_84 = tensor.expand_shape %extracted_slice_81 [[0], [1], [2, 3]] output_shape [1, 128, 12, 64] : tensor<1x128x768xf32> into tensor<1x128x12x64xf32>
    %transposed_85 = linalg.transpose ins(%expanded_84 : tensor<1x128x12x64xf32>) outs(%6 : tensor<1x12x128x64xf32>) permutation = [0, 2, 1, 3] 
    %expanded_86 = tensor.expand_shape %extracted_slice_79 [[0], [1], [2, 3]] output_shape [1, 128, 12, 64] : tensor<1x128x768xf32> into tensor<1x128x12x64xf32>
    %transposed_87 = linalg.transpose ins(%expanded_86 : tensor<1x128x12x64xf32>) outs(%6 : tensor<1x12x128x64xf32>) permutation = [0, 2, 1, 3] 
    %64 = dnn.scaled_dot_product_attention %transposed_87, %transposed_83, %transposed_85, %arg2 {parameter_indices = array<i32: 4, 5, 6, 7>, parameters = [0.000000e+00, false, 1.250000e-01, false]} : (tensor<1x12x128x64xf32>, tensor<1x12x128x64xf32>, tensor<1x12x128x64xf32>, tensor<1x1x128x128xi1>) -> tensor<1x12x128x64xf32>
    %transposed_88 = linalg.transpose ins(%64 : tensor<1x12x128x64xf32>) outs(%8 : tensor<1x128x12x64xf32>) permutation = [0, 2, 1, 3] 
    %collapsed_89 = tensor.collapse_shape %transposed_88 [[0, 1], [2, 3]] : tensor<1x128x12x64xf32> into tensor<128x768xf32>
    %65 = dnn.mm %collapsed_89, %cst_4 : tensor<128x768xf32>, tensor<768x768xf32> -> tensor<128x768xf32>
    %66 = dnn.add %65, %cst_1 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<128x768xf32>, tensor<768xf32>) -> tensor<128x768xf32>
    %expanded_90 = tensor.expand_shape %66 [[0, 1], [2]] output_shape [1, 128, 768] : tensor<128x768xf32> into tensor<1x128x768xf32>
    %67 = dnn.add %expanded_90, %60 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<1x128x768xf32>, tensor<1x128x768xf32>) -> tensor<1x128x768xf32>
    %68 = dnn.layer_norm %67, %cst_1, %cst_1 {parameter_indices = array<i32: 1, 4, 5>, parameters = [[768], 1.000000e-05, false]} : (tensor<1x128x768xf32>, tensor<768xf32>, tensor<768xf32>) -> tensor<1x128x768xf32>
    %collapsed_91 = tensor.collapse_shape %68 [[0, 1], [2]] : tensor<1x128x768xf32> into tensor<128x768xf32>
    %69 = dnn.mm %collapsed_91, %cst_5 : tensor<128x768xf32>, tensor<768x3072xf32> -> tensor<128x3072xf32>
    %70 = dnn.add %69, %cst_6 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<128x3072xf32>, tensor<3072xf32>) -> tensor<128x3072xf32>
    %expanded_92 = tensor.expand_shape %70 [[0, 1], [2]] output_shape [1, 128, 3072] : tensor<128x3072xf32> into tensor<1x128x3072xf32>
    %71 = dnn.gelu %expanded_92 {parameter_indices = array<i32: 1>, parameters = ["tanh"]} : (tensor<1x128x3072xf32>) -> tensor<1x128x3072xf32>
    %collapsed_93 = tensor.collapse_shape %71 [[0, 1], [2]] : tensor<1x128x3072xf32> into tensor<128x3072xf32>
    %72 = dnn.mm %collapsed_93, %cst_7 : tensor<128x3072xf32>, tensor<3072x768xf32> -> tensor<128x768xf32>
    %73 = dnn.add %72, %cst_1 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<128x768xf32>, tensor<768xf32>) -> tensor<128x768xf32>
    %expanded_94 = tensor.expand_shape %73 [[0, 1], [2]] output_shape [1, 128, 768] : tensor<128x768xf32> into tensor<1x128x768xf32>
    %74 = dnn.add %67, %expanded_94 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<1x128x768xf32>, tensor<1x128x768xf32>) -> tensor<1x128x768xf32>
    %75 = dnn.layer_norm %74, %cst_1, %cst_1 {parameter_indices = array<i32: 1, 4, 5>, parameters = [[768], 1.000000e-05, false]} : (tensor<1x128x768xf32>, tensor<768xf32>, tensor<768xf32>) -> tensor<1x128x768xf32>
    %collapsed_95 = tensor.collapse_shape %75 [[0, 1], [2]] : tensor<1x128x768xf32> into tensor<128x768xf32>
    %76 = dnn.mm %collapsed_95, %cst_2 : tensor<128x768xf32>, tensor<768x2304xf32> -> tensor<128x2304xf32>
    %77 = dnn.add %76, %cst_3 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<128x2304xf32>, tensor<2304xf32>) -> tensor<128x2304xf32>
    %expanded_96 = tensor.expand_shape %77 [[0, 1], [2]] output_shape [1, 128, 2304] : tensor<128x2304xf32> into tensor<1x128x2304xf32>
    %extracted_slice_97 = tensor.extract_slice %expanded_96[0, 0, 0] [1, 128, 768] [1, 1, 1] : tensor<1x128x2304xf32> to tensor<1x128x768xf32>
    %extracted_slice_98 = tensor.extract_slice %expanded_96[0, 0, 768] [1, 128, 768] [1, 1, 1] : tensor<1x128x2304xf32> to tensor<1x128x768xf32>
    %extracted_slice_99 = tensor.extract_slice %expanded_96[0, 0, 1536] [1, 128, 768] [1, 1, 1] : tensor<1x128x2304xf32> to tensor<1x128x768xf32>
    %expanded_100 = tensor.expand_shape %extracted_slice_98 [[0], [1], [2, 3]] output_shape [1, 128, 12, 64] : tensor<1x128x768xf32> into tensor<1x128x12x64xf32>
    %transposed_101 = linalg.transpose ins(%expanded_100 : tensor<1x128x12x64xf32>) outs(%6 : tensor<1x12x128x64xf32>) permutation = [0, 2, 1, 3] 
    %expanded_102 = tensor.expand_shape %extracted_slice_99 [[0], [1], [2, 3]] output_shape [1, 128, 12, 64] : tensor<1x128x768xf32> into tensor<1x128x12x64xf32>
    %transposed_103 = linalg.transpose ins(%expanded_102 : tensor<1x128x12x64xf32>) outs(%6 : tensor<1x12x128x64xf32>) permutation = [0, 2, 1, 3] 
    %expanded_104 = tensor.expand_shape %extracted_slice_97 [[0], [1], [2, 3]] output_shape [1, 128, 12, 64] : tensor<1x128x768xf32> into tensor<1x128x12x64xf32>
    %transposed_105 = linalg.transpose ins(%expanded_104 : tensor<1x128x12x64xf32>) outs(%6 : tensor<1x12x128x64xf32>) permutation = [0, 2, 1, 3] 
    %78 = dnn.scaled_dot_product_attention %transposed_105, %transposed_101, %transposed_103, %arg2 {parameter_indices = array<i32: 4, 5, 6, 7>, parameters = [0.000000e+00, false, 1.250000e-01, false]} : (tensor<1x12x128x64xf32>, tensor<1x12x128x64xf32>, tensor<1x12x128x64xf32>, tensor<1x1x128x128xi1>) -> tensor<1x12x128x64xf32>
    %transposed_106 = linalg.transpose ins(%78 : tensor<1x12x128x64xf32>) outs(%8 : tensor<1x128x12x64xf32>) permutation = [0, 2, 1, 3] 
    %collapsed_107 = tensor.collapse_shape %transposed_106 [[0, 1], [2, 3]] : tensor<1x128x12x64xf32> into tensor<128x768xf32>
    %79 = dnn.mm %collapsed_107, %cst_4 : tensor<128x768xf32>, tensor<768x768xf32> -> tensor<128x768xf32>
    %80 = dnn.add %79, %cst_1 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<128x768xf32>, tensor<768xf32>) -> tensor<128x768xf32>
    %expanded_108 = tensor.expand_shape %80 [[0, 1], [2]] output_shape [1, 128, 768] : tensor<128x768xf32> into tensor<1x128x768xf32>
    %81 = dnn.add %expanded_108, %74 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<1x128x768xf32>, tensor<1x128x768xf32>) -> tensor<1x128x768xf32>
    %82 = dnn.layer_norm %81, %cst_1, %cst_1 {parameter_indices = array<i32: 1, 4, 5>, parameters = [[768], 1.000000e-05, false]} : (tensor<1x128x768xf32>, tensor<768xf32>, tensor<768xf32>) -> tensor<1x128x768xf32>
    %collapsed_109 = tensor.collapse_shape %82 [[0, 1], [2]] : tensor<1x128x768xf32> into tensor<128x768xf32>
    %83 = dnn.mm %collapsed_109, %cst_5 : tensor<128x768xf32>, tensor<768x3072xf32> -> tensor<128x3072xf32>
    %84 = dnn.add %83, %cst_6 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<128x3072xf32>, tensor<3072xf32>) -> tensor<128x3072xf32>
    %expanded_110 = tensor.expand_shape %84 [[0, 1], [2]] output_shape [1, 128, 3072] : tensor<128x3072xf32> into tensor<1x128x3072xf32>
    %85 = dnn.gelu %expanded_110 {parameter_indices = array<i32: 1>, parameters = ["tanh"]} : (tensor<1x128x3072xf32>) -> tensor<1x128x3072xf32>
    %collapsed_111 = tensor.collapse_shape %85 [[0, 1], [2]] : tensor<1x128x3072xf32> into tensor<128x3072xf32>
    %86 = dnn.mm %collapsed_111, %cst_7 : tensor<128x3072xf32>, tensor<3072x768xf32> -> tensor<128x768xf32>
    %87 = dnn.add %86, %cst_1 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<128x768xf32>, tensor<768xf32>) -> tensor<128x768xf32>
    %expanded_112 = tensor.expand_shape %87 [[0, 1], [2]] output_shape [1, 128, 768] : tensor<128x768xf32> into tensor<1x128x768xf32>
    %88 = dnn.add %81, %expanded_112 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<1x128x768xf32>, tensor<1x128x768xf32>) -> tensor<1x128x768xf32>
    %89 = dnn.layer_norm %88, %cst_1, %cst_1 {parameter_indices = array<i32: 1, 4, 5>, parameters = [[768], 1.000000e-05, false]} : (tensor<1x128x768xf32>, tensor<768xf32>, tensor<768xf32>) -> tensor<1x128x768xf32>
    %collapsed_113 = tensor.collapse_shape %89 [[0, 1], [2]] : tensor<1x128x768xf32> into tensor<128x768xf32>
    %90 = dnn.mm %collapsed_113, %cst_2 : tensor<128x768xf32>, tensor<768x2304xf32> -> tensor<128x2304xf32>
    %91 = dnn.add %90, %cst_3 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<128x2304xf32>, tensor<2304xf32>) -> tensor<128x2304xf32>
    %expanded_114 = tensor.expand_shape %91 [[0, 1], [2]] output_shape [1, 128, 2304] : tensor<128x2304xf32> into tensor<1x128x2304xf32>
    %extracted_slice_115 = tensor.extract_slice %expanded_114[0, 0, 0] [1, 128, 768] [1, 1, 1] : tensor<1x128x2304xf32> to tensor<1x128x768xf32>
    %extracted_slice_116 = tensor.extract_slice %expanded_114[0, 0, 768] [1, 128, 768] [1, 1, 1] : tensor<1x128x2304xf32> to tensor<1x128x768xf32>
    %extracted_slice_117 = tensor.extract_slice %expanded_114[0, 0, 1536] [1, 128, 768] [1, 1, 1] : tensor<1x128x2304xf32> to tensor<1x128x768xf32>
    %expanded_118 = tensor.expand_shape %extracted_slice_116 [[0], [1], [2, 3]] output_shape [1, 128, 12, 64] : tensor<1x128x768xf32> into tensor<1x128x12x64xf32>
    %transposed_119 = linalg.transpose ins(%expanded_118 : tensor<1x128x12x64xf32>) outs(%6 : tensor<1x12x128x64xf32>) permutation = [0, 2, 1, 3] 
    %expanded_120 = tensor.expand_shape %extracted_slice_117 [[0], [1], [2, 3]] output_shape [1, 128, 12, 64] : tensor<1x128x768xf32> into tensor<1x128x12x64xf32>
    %transposed_121 = linalg.transpose ins(%expanded_120 : tensor<1x128x12x64xf32>) outs(%6 : tensor<1x12x128x64xf32>) permutation = [0, 2, 1, 3] 
    %expanded_122 = tensor.expand_shape %extracted_slice_115 [[0], [1], [2, 3]] output_shape [1, 128, 12, 64] : tensor<1x128x768xf32> into tensor<1x128x12x64xf32>
    %transposed_123 = linalg.transpose ins(%expanded_122 : tensor<1x128x12x64xf32>) outs(%6 : tensor<1x12x128x64xf32>) permutation = [0, 2, 1, 3] 
    %92 = dnn.scaled_dot_product_attention %transposed_123, %transposed_119, %transposed_121, %arg2 {parameter_indices = array<i32: 4, 5, 6, 7>, parameters = [0.000000e+00, false, 1.250000e-01, false]} : (tensor<1x12x128x64xf32>, tensor<1x12x128x64xf32>, tensor<1x12x128x64xf32>, tensor<1x1x128x128xi1>) -> tensor<1x12x128x64xf32>
    %transposed_124 = linalg.transpose ins(%92 : tensor<1x12x128x64xf32>) outs(%8 : tensor<1x128x12x64xf32>) permutation = [0, 2, 1, 3] 
    %collapsed_125 = tensor.collapse_shape %transposed_124 [[0, 1], [2, 3]] : tensor<1x128x12x64xf32> into tensor<128x768xf32>
    %93 = dnn.mm %collapsed_125, %cst_4 : tensor<128x768xf32>, tensor<768x768xf32> -> tensor<128x768xf32>
    %94 = dnn.add %93, %cst_1 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<128x768xf32>, tensor<768xf32>) -> tensor<128x768xf32>
    %expanded_126 = tensor.expand_shape %94 [[0, 1], [2]] output_shape [1, 128, 768] : tensor<128x768xf32> into tensor<1x128x768xf32>
    %95 = dnn.add %expanded_126, %88 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<1x128x768xf32>, tensor<1x128x768xf32>) -> tensor<1x128x768xf32>
    %96 = dnn.layer_norm %95, %cst_1, %cst_1 {parameter_indices = array<i32: 1, 4, 5>, parameters = [[768], 1.000000e-05, false]} : (tensor<1x128x768xf32>, tensor<768xf32>, tensor<768xf32>) -> tensor<1x128x768xf32>
    %collapsed_127 = tensor.collapse_shape %96 [[0, 1], [2]] : tensor<1x128x768xf32> into tensor<128x768xf32>
    %97 = dnn.mm %collapsed_127, %cst_5 : tensor<128x768xf32>, tensor<768x3072xf32> -> tensor<128x3072xf32>
    %98 = dnn.add %97, %cst_6 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<128x3072xf32>, tensor<3072xf32>) -> tensor<128x3072xf32>
    %expanded_128 = tensor.expand_shape %98 [[0, 1], [2]] output_shape [1, 128, 3072] : tensor<128x3072xf32> into tensor<1x128x3072xf32>
    %99 = dnn.gelu %expanded_128 {parameter_indices = array<i32: 1>, parameters = ["tanh"]} : (tensor<1x128x3072xf32>) -> tensor<1x128x3072xf32>
    %collapsed_129 = tensor.collapse_shape %99 [[0, 1], [2]] : tensor<1x128x3072xf32> into tensor<128x3072xf32>
    %100 = dnn.mm %collapsed_129, %cst_7 : tensor<128x3072xf32>, tensor<3072x768xf32> -> tensor<128x768xf32>
    %101 = dnn.add %100, %cst_1 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<128x768xf32>, tensor<768xf32>) -> tensor<128x768xf32>
    %expanded_130 = tensor.expand_shape %101 [[0, 1], [2]] output_shape [1, 128, 768] : tensor<128x768xf32> into tensor<1x128x768xf32>
    %102 = dnn.add %95, %expanded_130 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<1x128x768xf32>, tensor<1x128x768xf32>) -> tensor<1x128x768xf32>
    %103 = dnn.layer_norm %102, %cst_1, %cst_1 {parameter_indices = array<i32: 1, 4, 5>, parameters = [[768], 1.000000e-05, false]} : (tensor<1x128x768xf32>, tensor<768xf32>, tensor<768xf32>) -> tensor<1x128x768xf32>
    %collapsed_131 = tensor.collapse_shape %103 [[0, 1], [2]] : tensor<1x128x768xf32> into tensor<128x768xf32>
    %104 = dnn.mm %collapsed_131, %cst_2 : tensor<128x768xf32>, tensor<768x2304xf32> -> tensor<128x2304xf32>
    %105 = dnn.add %104, %cst_3 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<128x2304xf32>, tensor<2304xf32>) -> tensor<128x2304xf32>
    %expanded_132 = tensor.expand_shape %105 [[0, 1], [2]] output_shape [1, 128, 2304] : tensor<128x2304xf32> into tensor<1x128x2304xf32>
    %extracted_slice_133 = tensor.extract_slice %expanded_132[0, 0, 0] [1, 128, 768] [1, 1, 1] : tensor<1x128x2304xf32> to tensor<1x128x768xf32>
    %extracted_slice_134 = tensor.extract_slice %expanded_132[0, 0, 768] [1, 128, 768] [1, 1, 1] : tensor<1x128x2304xf32> to tensor<1x128x768xf32>
    %extracted_slice_135 = tensor.extract_slice %expanded_132[0, 0, 1536] [1, 128, 768] [1, 1, 1] : tensor<1x128x2304xf32> to tensor<1x128x768xf32>
    %expanded_136 = tensor.expand_shape %extracted_slice_134 [[0], [1], [2, 3]] output_shape [1, 128, 12, 64] : tensor<1x128x768xf32> into tensor<1x128x12x64xf32>
    %transposed_137 = linalg.transpose ins(%expanded_136 : tensor<1x128x12x64xf32>) outs(%6 : tensor<1x12x128x64xf32>) permutation = [0, 2, 1, 3] 
    %expanded_138 = tensor.expand_shape %extracted_slice_135 [[0], [1], [2, 3]] output_shape [1, 128, 12, 64] : tensor<1x128x768xf32> into tensor<1x128x12x64xf32>
    %transposed_139 = linalg.transpose ins(%expanded_138 : tensor<1x128x12x64xf32>) outs(%6 : tensor<1x12x128x64xf32>) permutation = [0, 2, 1, 3] 
    %expanded_140 = tensor.expand_shape %extracted_slice_133 [[0], [1], [2, 3]] output_shape [1, 128, 12, 64] : tensor<1x128x768xf32> into tensor<1x128x12x64xf32>
    %transposed_141 = linalg.transpose ins(%expanded_140 : tensor<1x128x12x64xf32>) outs(%6 : tensor<1x12x128x64xf32>) permutation = [0, 2, 1, 3] 
    %106 = dnn.scaled_dot_product_attention %transposed_141, %transposed_137, %transposed_139, %arg2 {parameter_indices = array<i32: 4, 5, 6, 7>, parameters = [0.000000e+00, false, 1.250000e-01, false]} : (tensor<1x12x128x64xf32>, tensor<1x12x128x64xf32>, tensor<1x12x128x64xf32>, tensor<1x1x128x128xi1>) -> tensor<1x12x128x64xf32>
    %transposed_142 = linalg.transpose ins(%106 : tensor<1x12x128x64xf32>) outs(%8 : tensor<1x128x12x64xf32>) permutation = [0, 2, 1, 3] 
    %collapsed_143 = tensor.collapse_shape %transposed_142 [[0, 1], [2, 3]] : tensor<1x128x12x64xf32> into tensor<128x768xf32>
    %107 = dnn.mm %collapsed_143, %cst_4 : tensor<128x768xf32>, tensor<768x768xf32> -> tensor<128x768xf32>
    %108 = dnn.add %107, %cst_1 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<128x768xf32>, tensor<768xf32>) -> tensor<128x768xf32>
    %expanded_144 = tensor.expand_shape %108 [[0, 1], [2]] output_shape [1, 128, 768] : tensor<128x768xf32> into tensor<1x128x768xf32>
    %109 = dnn.add %expanded_144, %102 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<1x128x768xf32>, tensor<1x128x768xf32>) -> tensor<1x128x768xf32>
    %110 = dnn.layer_norm %109, %cst_1, %cst_1 {parameter_indices = array<i32: 1, 4, 5>, parameters = [[768], 1.000000e-05, false]} : (tensor<1x128x768xf32>, tensor<768xf32>, tensor<768xf32>) -> tensor<1x128x768xf32>
    %collapsed_145 = tensor.collapse_shape %110 [[0, 1], [2]] : tensor<1x128x768xf32> into tensor<128x768xf32>
    %111 = dnn.mm %collapsed_145, %cst_5 : tensor<128x768xf32>, tensor<768x3072xf32> -> tensor<128x3072xf32>
    %112 = dnn.add %111, %cst_6 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<128x3072xf32>, tensor<3072xf32>) -> tensor<128x3072xf32>
    %expanded_146 = tensor.expand_shape %112 [[0, 1], [2]] output_shape [1, 128, 3072] : tensor<128x3072xf32> into tensor<1x128x3072xf32>
    %113 = dnn.gelu %expanded_146 {parameter_indices = array<i32: 1>, parameters = ["tanh"]} : (tensor<1x128x3072xf32>) -> tensor<1x128x3072xf32>
    %collapsed_147 = tensor.collapse_shape %113 [[0, 1], [2]] : tensor<1x128x3072xf32> into tensor<128x3072xf32>
    %114 = dnn.mm %collapsed_147, %cst_7 : tensor<128x3072xf32>, tensor<3072x768xf32> -> tensor<128x768xf32>
    %115 = dnn.add %114, %cst_1 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<128x768xf32>, tensor<768xf32>) -> tensor<128x768xf32>
    %expanded_148 = tensor.expand_shape %115 [[0, 1], [2]] output_shape [1, 128, 768] : tensor<128x768xf32> into tensor<1x128x768xf32>
    %116 = dnn.add %109, %expanded_148 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<1x128x768xf32>, tensor<1x128x768xf32>) -> tensor<1x128x768xf32>
    %117 = dnn.layer_norm %116, %cst_1, %cst_1 {parameter_indices = array<i32: 1, 4, 5>, parameters = [[768], 1.000000e-05, false]} : (tensor<1x128x768xf32>, tensor<768xf32>, tensor<768xf32>) -> tensor<1x128x768xf32>
    %collapsed_149 = tensor.collapse_shape %117 [[0, 1], [2]] : tensor<1x128x768xf32> into tensor<128x768xf32>
    %118 = dnn.mm %collapsed_149, %cst_2 : tensor<128x768xf32>, tensor<768x2304xf32> -> tensor<128x2304xf32>
    %119 = dnn.add %118, %cst_3 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<128x2304xf32>, tensor<2304xf32>) -> tensor<128x2304xf32>
    %expanded_150 = tensor.expand_shape %119 [[0, 1], [2]] output_shape [1, 128, 2304] : tensor<128x2304xf32> into tensor<1x128x2304xf32>
    %extracted_slice_151 = tensor.extract_slice %expanded_150[0, 0, 0] [1, 128, 768] [1, 1, 1] : tensor<1x128x2304xf32> to tensor<1x128x768xf32>
    %extracted_slice_152 = tensor.extract_slice %expanded_150[0, 0, 768] [1, 128, 768] [1, 1, 1] : tensor<1x128x2304xf32> to tensor<1x128x768xf32>
    %extracted_slice_153 = tensor.extract_slice %expanded_150[0, 0, 1536] [1, 128, 768] [1, 1, 1] : tensor<1x128x2304xf32> to tensor<1x128x768xf32>
    %expanded_154 = tensor.expand_shape %extracted_slice_152 [[0], [1], [2, 3]] output_shape [1, 128, 12, 64] : tensor<1x128x768xf32> into tensor<1x128x12x64xf32>
    %transposed_155 = linalg.transpose ins(%expanded_154 : tensor<1x128x12x64xf32>) outs(%6 : tensor<1x12x128x64xf32>) permutation = [0, 2, 1, 3] 
    %expanded_156 = tensor.expand_shape %extracted_slice_153 [[0], [1], [2, 3]] output_shape [1, 128, 12, 64] : tensor<1x128x768xf32> into tensor<1x128x12x64xf32>
    %transposed_157 = linalg.transpose ins(%expanded_156 : tensor<1x128x12x64xf32>) outs(%6 : tensor<1x12x128x64xf32>) permutation = [0, 2, 1, 3] 
    %expanded_158 = tensor.expand_shape %extracted_slice_151 [[0], [1], [2, 3]] output_shape [1, 128, 12, 64] : tensor<1x128x768xf32> into tensor<1x128x12x64xf32>
    %transposed_159 = linalg.transpose ins(%expanded_158 : tensor<1x128x12x64xf32>) outs(%6 : tensor<1x12x128x64xf32>) permutation = [0, 2, 1, 3] 
    %120 = dnn.scaled_dot_product_attention %transposed_159, %transposed_155, %transposed_157, %arg2 {parameter_indices = array<i32: 4, 5, 6, 7>, parameters = [0.000000e+00, false, 1.250000e-01, false]} : (tensor<1x12x128x64xf32>, tensor<1x12x128x64xf32>, tensor<1x12x128x64xf32>, tensor<1x1x128x128xi1>) -> tensor<1x12x128x64xf32>
    %transposed_160 = linalg.transpose ins(%120 : tensor<1x12x128x64xf32>) outs(%8 : tensor<1x128x12x64xf32>) permutation = [0, 2, 1, 3] 
    %collapsed_161 = tensor.collapse_shape %transposed_160 [[0, 1], [2, 3]] : tensor<1x128x12x64xf32> into tensor<128x768xf32>
    %121 = dnn.mm %collapsed_161, %cst_4 : tensor<128x768xf32>, tensor<768x768xf32> -> tensor<128x768xf32>
    %122 = dnn.add %121, %cst_1 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<128x768xf32>, tensor<768xf32>) -> tensor<128x768xf32>
    %expanded_162 = tensor.expand_shape %122 [[0, 1], [2]] output_shape [1, 128, 768] : tensor<128x768xf32> into tensor<1x128x768xf32>
    %123 = dnn.add %expanded_162, %116 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<1x128x768xf32>, tensor<1x128x768xf32>) -> tensor<1x128x768xf32>
    %124 = dnn.layer_norm %123, %cst_1, %cst_1 {parameter_indices = array<i32: 1, 4, 5>, parameters = [[768], 1.000000e-05, false]} : (tensor<1x128x768xf32>, tensor<768xf32>, tensor<768xf32>) -> tensor<1x128x768xf32>
    %collapsed_163 = tensor.collapse_shape %124 [[0, 1], [2]] : tensor<1x128x768xf32> into tensor<128x768xf32>
    %125 = dnn.mm %collapsed_163, %cst_5 : tensor<128x768xf32>, tensor<768x3072xf32> -> tensor<128x3072xf32>
    %126 = dnn.add %125, %cst_6 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<128x3072xf32>, tensor<3072xf32>) -> tensor<128x3072xf32>
    %expanded_164 = tensor.expand_shape %126 [[0, 1], [2]] output_shape [1, 128, 3072] : tensor<128x3072xf32> into tensor<1x128x3072xf32>
    %127 = dnn.gelu %expanded_164 {parameter_indices = array<i32: 1>, parameters = ["tanh"]} : (tensor<1x128x3072xf32>) -> tensor<1x128x3072xf32>
    %collapsed_165 = tensor.collapse_shape %127 [[0, 1], [2]] : tensor<1x128x3072xf32> into tensor<128x3072xf32>
    %128 = dnn.mm %collapsed_165, %cst_7 : tensor<128x3072xf32>, tensor<3072x768xf32> -> tensor<128x768xf32>
    %129 = dnn.add %128, %cst_1 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<128x768xf32>, tensor<768xf32>) -> tensor<128x768xf32>
    %expanded_166 = tensor.expand_shape %129 [[0, 1], [2]] output_shape [1, 128, 768] : tensor<128x768xf32> into tensor<1x128x768xf32>
    %130 = dnn.add %123, %expanded_166 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<1x128x768xf32>, tensor<1x128x768xf32>) -> tensor<1x128x768xf32>
    %131 = dnn.layer_norm %130, %cst_1, %cst_1 {parameter_indices = array<i32: 1, 4, 5>, parameters = [[768], 1.000000e-05, false]} : (tensor<1x128x768xf32>, tensor<768xf32>, tensor<768xf32>) -> tensor<1x128x768xf32>
    %collapsed_167 = tensor.collapse_shape %131 [[0, 1], [2]] : tensor<1x128x768xf32> into tensor<128x768xf32>
    %132 = dnn.mm %collapsed_167, %cst_2 : tensor<128x768xf32>, tensor<768x2304xf32> -> tensor<128x2304xf32>
    %133 = dnn.add %132, %cst_3 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<128x2304xf32>, tensor<2304xf32>) -> tensor<128x2304xf32>
    %expanded_168 = tensor.expand_shape %133 [[0, 1], [2]] output_shape [1, 128, 2304] : tensor<128x2304xf32> into tensor<1x128x2304xf32>
    %extracted_slice_169 = tensor.extract_slice %expanded_168[0, 0, 0] [1, 128, 768] [1, 1, 1] : tensor<1x128x2304xf32> to tensor<1x128x768xf32>
    %extracted_slice_170 = tensor.extract_slice %expanded_168[0, 0, 768] [1, 128, 768] [1, 1, 1] : tensor<1x128x2304xf32> to tensor<1x128x768xf32>
    %extracted_slice_171 = tensor.extract_slice %expanded_168[0, 0, 1536] [1, 128, 768] [1, 1, 1] : tensor<1x128x2304xf32> to tensor<1x128x768xf32>
    %expanded_172 = tensor.expand_shape %extracted_slice_170 [[0], [1], [2, 3]] output_shape [1, 128, 12, 64] : tensor<1x128x768xf32> into tensor<1x128x12x64xf32>
    %transposed_173 = linalg.transpose ins(%expanded_172 : tensor<1x128x12x64xf32>) outs(%6 : tensor<1x12x128x64xf32>) permutation = [0, 2, 1, 3] 
    %expanded_174 = tensor.expand_shape %extracted_slice_171 [[0], [1], [2, 3]] output_shape [1, 128, 12, 64] : tensor<1x128x768xf32> into tensor<1x128x12x64xf32>
    %transposed_175 = linalg.transpose ins(%expanded_174 : tensor<1x128x12x64xf32>) outs(%6 : tensor<1x12x128x64xf32>) permutation = [0, 2, 1, 3] 
    %expanded_176 = tensor.expand_shape %extracted_slice_169 [[0], [1], [2, 3]] output_shape [1, 128, 12, 64] : tensor<1x128x768xf32> into tensor<1x128x12x64xf32>
    %transposed_177 = linalg.transpose ins(%expanded_176 : tensor<1x128x12x64xf32>) outs(%6 : tensor<1x12x128x64xf32>) permutation = [0, 2, 1, 3] 
    %134 = dnn.scaled_dot_product_attention %transposed_177, %transposed_173, %transposed_175, %arg2 {parameter_indices = array<i32: 4, 5, 6, 7>, parameters = [0.000000e+00, false, 1.250000e-01, false]} : (tensor<1x12x128x64xf32>, tensor<1x12x128x64xf32>, tensor<1x12x128x64xf32>, tensor<1x1x128x128xi1>) -> tensor<1x12x128x64xf32>
    %transposed_178 = linalg.transpose ins(%134 : tensor<1x12x128x64xf32>) outs(%8 : tensor<1x128x12x64xf32>) permutation = [0, 2, 1, 3] 
    %collapsed_179 = tensor.collapse_shape %transposed_178 [[0, 1], [2, 3]] : tensor<1x128x12x64xf32> into tensor<128x768xf32>
    %135 = dnn.mm %collapsed_179, %cst_4 : tensor<128x768xf32>, tensor<768x768xf32> -> tensor<128x768xf32>
    %136 = dnn.add %135, %cst_1 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<128x768xf32>, tensor<768xf32>) -> tensor<128x768xf32>
    %expanded_180 = tensor.expand_shape %136 [[0, 1], [2]] output_shape [1, 128, 768] : tensor<128x768xf32> into tensor<1x128x768xf32>
    %137 = dnn.add %expanded_180, %130 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<1x128x768xf32>, tensor<1x128x768xf32>) -> tensor<1x128x768xf32>
    %138 = dnn.layer_norm %137, %cst_1, %cst_1 {parameter_indices = array<i32: 1, 4, 5>, parameters = [[768], 1.000000e-05, false]} : (tensor<1x128x768xf32>, tensor<768xf32>, tensor<768xf32>) -> tensor<1x128x768xf32>
    %collapsed_181 = tensor.collapse_shape %138 [[0, 1], [2]] : tensor<1x128x768xf32> into tensor<128x768xf32>
    %139 = dnn.mm %collapsed_181, %cst_5 : tensor<128x768xf32>, tensor<768x3072xf32> -> tensor<128x3072xf32>
    %140 = dnn.add %139, %cst_6 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<128x3072xf32>, tensor<3072xf32>) -> tensor<128x3072xf32>
    %expanded_182 = tensor.expand_shape %140 [[0, 1], [2]] output_shape [1, 128, 3072] : tensor<128x3072xf32> into tensor<1x128x3072xf32>
    %141 = dnn.gelu %expanded_182 {parameter_indices = array<i32: 1>, parameters = ["tanh"]} : (tensor<1x128x3072xf32>) -> tensor<1x128x3072xf32>
    %collapsed_183 = tensor.collapse_shape %141 [[0, 1], [2]] : tensor<1x128x3072xf32> into tensor<128x3072xf32>
    %142 = dnn.mm %collapsed_183, %cst_7 : tensor<128x3072xf32>, tensor<3072x768xf32> -> tensor<128x768xf32>
    %143 = dnn.add %142, %cst_1 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<128x768xf32>, tensor<768xf32>) -> tensor<128x768xf32>
    %expanded_184 = tensor.expand_shape %143 [[0, 1], [2]] output_shape [1, 128, 768] : tensor<128x768xf32> into tensor<1x128x768xf32>
    %144 = dnn.add %137, %expanded_184 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<1x128x768xf32>, tensor<1x128x768xf32>) -> tensor<1x128x768xf32>
    %145 = dnn.layer_norm %144, %cst_1, %cst_1 {parameter_indices = array<i32: 1, 4, 5>, parameters = [[768], 1.000000e-05, false]} : (tensor<1x128x768xf32>, tensor<768xf32>, tensor<768xf32>) -> tensor<1x128x768xf32>
    %collapsed_185 = tensor.collapse_shape %145 [[0, 1], [2]] : tensor<1x128x768xf32> into tensor<128x768xf32>
    %146 = dnn.mm %collapsed_185, %cst_2 : tensor<128x768xf32>, tensor<768x2304xf32> -> tensor<128x2304xf32>
    %147 = dnn.add %146, %cst_3 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<128x2304xf32>, tensor<2304xf32>) -> tensor<128x2304xf32>
    %expanded_186 = tensor.expand_shape %147 [[0, 1], [2]] output_shape [1, 128, 2304] : tensor<128x2304xf32> into tensor<1x128x2304xf32>
    %extracted_slice_187 = tensor.extract_slice %expanded_186[0, 0, 0] [1, 128, 768] [1, 1, 1] : tensor<1x128x2304xf32> to tensor<1x128x768xf32>
    %extracted_slice_188 = tensor.extract_slice %expanded_186[0, 0, 768] [1, 128, 768] [1, 1, 1] : tensor<1x128x2304xf32> to tensor<1x128x768xf32>
    %extracted_slice_189 = tensor.extract_slice %expanded_186[0, 0, 1536] [1, 128, 768] [1, 1, 1] : tensor<1x128x2304xf32> to tensor<1x128x768xf32>
    %expanded_190 = tensor.expand_shape %extracted_slice_188 [[0], [1], [2, 3]] output_shape [1, 128, 12, 64] : tensor<1x128x768xf32> into tensor<1x128x12x64xf32>
    %transposed_191 = linalg.transpose ins(%expanded_190 : tensor<1x128x12x64xf32>) outs(%6 : tensor<1x12x128x64xf32>) permutation = [0, 2, 1, 3] 
    %expanded_192 = tensor.expand_shape %extracted_slice_189 [[0], [1], [2, 3]] output_shape [1, 128, 12, 64] : tensor<1x128x768xf32> into tensor<1x128x12x64xf32>
    %transposed_193 = linalg.transpose ins(%expanded_192 : tensor<1x128x12x64xf32>) outs(%6 : tensor<1x12x128x64xf32>) permutation = [0, 2, 1, 3] 
    %expanded_194 = tensor.expand_shape %extracted_slice_187 [[0], [1], [2, 3]] output_shape [1, 128, 12, 64] : tensor<1x128x768xf32> into tensor<1x128x12x64xf32>
    %transposed_195 = linalg.transpose ins(%expanded_194 : tensor<1x128x12x64xf32>) outs(%6 : tensor<1x12x128x64xf32>) permutation = [0, 2, 1, 3] 
    %148 = dnn.scaled_dot_product_attention %transposed_195, %transposed_191, %transposed_193, %arg2 {parameter_indices = array<i32: 4, 5, 6, 7>, parameters = [0.000000e+00, false, 1.250000e-01, false]} : (tensor<1x12x128x64xf32>, tensor<1x12x128x64xf32>, tensor<1x12x128x64xf32>, tensor<1x1x128x128xi1>) -> tensor<1x12x128x64xf32>
    %transposed_196 = linalg.transpose ins(%148 : tensor<1x12x128x64xf32>) outs(%8 : tensor<1x128x12x64xf32>) permutation = [0, 2, 1, 3] 
    %collapsed_197 = tensor.collapse_shape %transposed_196 [[0, 1], [2, 3]] : tensor<1x128x12x64xf32> into tensor<128x768xf32>
    %149 = dnn.mm %collapsed_197, %cst_4 : tensor<128x768xf32>, tensor<768x768xf32> -> tensor<128x768xf32>
    %150 = dnn.add %149, %cst_1 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<128x768xf32>, tensor<768xf32>) -> tensor<128x768xf32>
    %expanded_198 = tensor.expand_shape %150 [[0, 1], [2]] output_shape [1, 128, 768] : tensor<128x768xf32> into tensor<1x128x768xf32>
    %151 = dnn.add %expanded_198, %144 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<1x128x768xf32>, tensor<1x128x768xf32>) -> tensor<1x128x768xf32>
    %152 = dnn.layer_norm %151, %cst_1, %cst_1 {parameter_indices = array<i32: 1, 4, 5>, parameters = [[768], 1.000000e-05, false]} : (tensor<1x128x768xf32>, tensor<768xf32>, tensor<768xf32>) -> tensor<1x128x768xf32>
    %collapsed_199 = tensor.collapse_shape %152 [[0, 1], [2]] : tensor<1x128x768xf32> into tensor<128x768xf32>
    %153 = dnn.mm %collapsed_199, %cst_5 : tensor<128x768xf32>, tensor<768x3072xf32> -> tensor<128x3072xf32>
    %154 = dnn.add %153, %cst_6 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<128x3072xf32>, tensor<3072xf32>) -> tensor<128x3072xf32>
    %expanded_200 = tensor.expand_shape %154 [[0, 1], [2]] output_shape [1, 128, 3072] : tensor<128x3072xf32> into tensor<1x128x3072xf32>
    %155 = dnn.gelu %expanded_200 {parameter_indices = array<i32: 1>, parameters = ["tanh"]} : (tensor<1x128x3072xf32>) -> tensor<1x128x3072xf32>
    %collapsed_201 = tensor.collapse_shape %155 [[0, 1], [2]] : tensor<1x128x3072xf32> into tensor<128x3072xf32>
    %156 = dnn.mm %collapsed_201, %cst_7 : tensor<128x3072xf32>, tensor<3072x768xf32> -> tensor<128x768xf32>
    %157 = dnn.add %156, %cst_1 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<128x768xf32>, tensor<768xf32>) -> tensor<128x768xf32>
    %expanded_202 = tensor.expand_shape %157 [[0, 1], [2]] output_shape [1, 128, 768] : tensor<128x768xf32> into tensor<1x128x768xf32>
    %158 = dnn.add %151, %expanded_202 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<1x128x768xf32>, tensor<1x128x768xf32>) -> tensor<1x128x768xf32>
    %159 = dnn.layer_norm %158, %cst_1, %cst_1 {parameter_indices = array<i32: 1, 4, 5>, parameters = [[768], 1.000000e-05, false]} : (tensor<1x128x768xf32>, tensor<768xf32>, tensor<768xf32>) -> tensor<1x128x768xf32>
    %collapsed_203 = tensor.collapse_shape %159 [[0, 1], [2]] : tensor<1x128x768xf32> into tensor<128x768xf32>
    %160 = dnn.mm %collapsed_203, %cst_2 : tensor<128x768xf32>, tensor<768x2304xf32> -> tensor<128x2304xf32>
    %161 = dnn.add %160, %cst_3 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<128x2304xf32>, tensor<2304xf32>) -> tensor<128x2304xf32>
    %expanded_204 = tensor.expand_shape %161 [[0, 1], [2]] output_shape [1, 128, 2304] : tensor<128x2304xf32> into tensor<1x128x2304xf32>
    %extracted_slice_205 = tensor.extract_slice %expanded_204[0, 0, 0] [1, 128, 768] [1, 1, 1] : tensor<1x128x2304xf32> to tensor<1x128x768xf32>
    %extracted_slice_206 = tensor.extract_slice %expanded_204[0, 0, 768] [1, 128, 768] [1, 1, 1] : tensor<1x128x2304xf32> to tensor<1x128x768xf32>
    %extracted_slice_207 = tensor.extract_slice %expanded_204[0, 0, 1536] [1, 128, 768] [1, 1, 1] : tensor<1x128x2304xf32> to tensor<1x128x768xf32>
    %expanded_208 = tensor.expand_shape %extracted_slice_206 [[0], [1], [2, 3]] output_shape [1, 128, 12, 64] : tensor<1x128x768xf32> into tensor<1x128x12x64xf32>
    %transposed_209 = linalg.transpose ins(%expanded_208 : tensor<1x128x12x64xf32>) outs(%6 : tensor<1x12x128x64xf32>) permutation = [0, 2, 1, 3] 
    %expanded_210 = tensor.expand_shape %extracted_slice_207 [[0], [1], [2, 3]] output_shape [1, 128, 12, 64] : tensor<1x128x768xf32> into tensor<1x128x12x64xf32>
    %transposed_211 = linalg.transpose ins(%expanded_210 : tensor<1x128x12x64xf32>) outs(%6 : tensor<1x12x128x64xf32>) permutation = [0, 2, 1, 3] 
    %expanded_212 = tensor.expand_shape %extracted_slice_205 [[0], [1], [2, 3]] output_shape [1, 128, 12, 64] : tensor<1x128x768xf32> into tensor<1x128x12x64xf32>
    %transposed_213 = linalg.transpose ins(%expanded_212 : tensor<1x128x12x64xf32>) outs(%6 : tensor<1x12x128x64xf32>) permutation = [0, 2, 1, 3] 
    %162 = dnn.scaled_dot_product_attention %transposed_213, %transposed_209, %transposed_211, %arg2 {parameter_indices = array<i32: 4, 5, 6, 7>, parameters = [0.000000e+00, false, 1.250000e-01, false]} : (tensor<1x12x128x64xf32>, tensor<1x12x128x64xf32>, tensor<1x12x128x64xf32>, tensor<1x1x128x128xi1>) -> tensor<1x12x128x64xf32>
    %transposed_214 = linalg.transpose ins(%162 : tensor<1x12x128x64xf32>) outs(%8 : tensor<1x128x12x64xf32>) permutation = [0, 2, 1, 3] 
    %collapsed_215 = tensor.collapse_shape %transposed_214 [[0, 1], [2, 3]] : tensor<1x128x12x64xf32> into tensor<128x768xf32>
    %163 = dnn.mm %collapsed_215, %cst_4 : tensor<128x768xf32>, tensor<768x768xf32> -> tensor<128x768xf32>
    %164 = dnn.add %163, %cst_1 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<128x768xf32>, tensor<768xf32>) -> tensor<128x768xf32>
    %expanded_216 = tensor.expand_shape %164 [[0, 1], [2]] output_shape [1, 128, 768] : tensor<128x768xf32> into tensor<1x128x768xf32>
    %165 = dnn.add %expanded_216, %158 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<1x128x768xf32>, tensor<1x128x768xf32>) -> tensor<1x128x768xf32>
    %166 = dnn.layer_norm %165, %cst_1, %cst_1 {parameter_indices = array<i32: 1, 4, 5>, parameters = [[768], 1.000000e-05, false]} : (tensor<1x128x768xf32>, tensor<768xf32>, tensor<768xf32>) -> tensor<1x128x768xf32>
    %collapsed_217 = tensor.collapse_shape %166 [[0, 1], [2]] : tensor<1x128x768xf32> into tensor<128x768xf32>
    %167 = dnn.mm %collapsed_217, %cst_5 : tensor<128x768xf32>, tensor<768x3072xf32> -> tensor<128x3072xf32>
    %168 = dnn.add %167, %cst_6 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<128x3072xf32>, tensor<3072xf32>) -> tensor<128x3072xf32>
    %expanded_218 = tensor.expand_shape %168 [[0, 1], [2]] output_shape [1, 128, 3072] : tensor<128x3072xf32> into tensor<1x128x3072xf32>
    %169 = dnn.gelu %expanded_218 {parameter_indices = array<i32: 1>, parameters = ["tanh"]} : (tensor<1x128x3072xf32>) -> tensor<1x128x3072xf32>
    %collapsed_219 = tensor.collapse_shape %169 [[0, 1], [2]] : tensor<1x128x3072xf32> into tensor<128x3072xf32>
    %170 = dnn.mm %collapsed_219, %cst_7 : tensor<128x3072xf32>, tensor<3072x768xf32> -> tensor<128x768xf32>
    %171 = dnn.add %170, %cst_1 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<128x768xf32>, tensor<768xf32>) -> tensor<128x768xf32>
    %expanded_220 = tensor.expand_shape %171 [[0, 1], [2]] output_shape [1, 128, 768] : tensor<128x768xf32> into tensor<1x128x768xf32>
    %172 = dnn.add %165, %expanded_220 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<1x128x768xf32>, tensor<1x128x768xf32>) -> tensor<1x128x768xf32>
    %173 = dnn.layer_norm %172, %cst_1, %cst_1 {parameter_indices = array<i32: 1, 4, 5>, parameters = [[768], 1.000000e-05, false]} : (tensor<1x128x768xf32>, tensor<768xf32>, tensor<768xf32>) -> tensor<1x128x768xf32>
    %174 = dnn.matmul %173, %cst_8 : tensor<1x128x768xf32>, tensor<768x50257xf32> -> tensor<1x128x50257xf32>
    return %174 : tensor<1x128x50257xf32>
  }
}

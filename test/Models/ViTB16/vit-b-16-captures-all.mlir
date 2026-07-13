// RUN: dnn-mlir-opt %s | FileCheck %s
//
// ViT-B/16 after captures=all and the remaining Torch-to-Linalg backend
// lowering. Transformer computation remains high-level in DNN; standard
// tensor reshaping and transposition remain in MLIR infrastructure dialects.
//
// CHECK-LABEL: func.func @vit_b_16
// CHECK-DAG: dnn.convolution
// CHECK-DAG: dnn.linear
// CHECK-DAG: dnn.add
// CHECK-DAG: dnn.gelu
// CHECK-DAG: dnn.layer_norm
// CHECK-DAG: dnn.scaled_dot_product_attention
// CHECK-DAG: linalg.transpose
// CHECK-NOT: linalg.batch_matmul
// CHECK-NOT: linalg.generic
// CHECK-NOT: torch.aten

module {
  func.func @vit_b_16(%arg0: tensor<1x3x224x224xf32>) -> tensor<1x1000xf32> {
    %cst = arith.constant dense<0.000000e+00> : tensor<768x3x16x16xf32>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<768xf32>
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<1x197x768xf32>
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<2304x768xf32>
    %cst_3 = arith.constant dense<0.000000e+00> : tensor<2304xf32>
    %cst_4 = arith.constant dense<0.000000e+00> : tensor<768x768xf32>
    %cst_5 = arith.constant dense<0.000000e+00> : tensor<3072x768xf32>
    %cst_6 = arith.constant dense<0.000000e+00> : tensor<3072xf32>
    %cst_7 = arith.constant dense<0.000000e+00> : tensor<768x3072xf32>
    %cst_8 = arith.constant dense<0.000000e+00> : tensor<1000x768xf32>
    %cst_9 = arith.constant dense<0.000000e+00> : tensor<1000xf32>
    %cst_10 = arith.constant dense<0.000000e+00> : tensor<1x1x768xf32>
    %0 = dnn.convolution %arg0, %cst, %cst_0 {kind = "aten.conv2d", parameter_indices = array<i32: 3, 4, 5, 6>, parameters = [[16, 16], [0, 0], [1, 1], 1]} : (tensor<1x3x224x224xf32>, tensor<768x3x16x16xf32>, tensor<768xf32>) -> tensor<1x768x14x14xf32>
    %collapsed = tensor.collapse_shape %0 [[0], [1], [2, 3]] : tensor<1x768x14x14xf32> into tensor<1x768x196xf32>
    %1 = tensor.empty() : tensor<1x196x768xf32>
    %transposed = linalg.transpose ins(%collapsed : tensor<1x768x196xf32>) outs(%1 : tensor<1x196x768xf32>) permutation = [0, 2, 1] 
    %concat = tensor.concat dim(1) %cst_10, %transposed : (tensor<1x1x768xf32>, tensor<1x196x768xf32>) -> tensor<1x197x768xf32>
    %2 = dnn.add %concat, %cst_1 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %3 = dnn.layer_norm %2, %cst_0, %cst_0 {parameter_indices = array<i32: 1, 4, 5>, parameters = [[768], 9.9999999999999995E-7, false]} : (tensor<1x197x768xf32>, tensor<768xf32>, tensor<768xf32>) -> tensor<1x197x768xf32>
    %4 = tensor.empty() : tensor<197x1x768xf32>
    %transposed_11 = linalg.transpose ins(%3 : tensor<1x197x768xf32>) outs(%4 : tensor<197x1x768xf32>) permutation = [1, 0, 2] 
    %5 = dnn.linear %transposed_11, %cst_2, %cst_3 : tensor<197x1x768xf32>, tensor<2304x768xf32>, tensor<2304xf32> -> tensor<197x1x2304xf32>
    %expanded = tensor.expand_shape %5 [[0, 1], [2], [3, 4]] output_shape [1, 197, 1, 3, 768] : tensor<197x1x2304xf32> into tensor<1x197x1x3x768xf32>
    %6 = tensor.empty() : tensor<3x197x1x1x768xf32>
    %transposed_12 = linalg.transpose ins(%expanded : tensor<1x197x1x3x768xf32>) outs(%6 : tensor<3x197x1x1x768xf32>) permutation = [3, 1, 2, 0, 4] 
    %collapsed_13 = tensor.collapse_shape %transposed_12 [[0], [1], [2, 3], [4]] : tensor<3x197x1x1x768xf32> into tensor<3x197x1x768xf32>
    %extracted_slice = tensor.extract_slice %collapsed_13[0, 0, 0, 0] [1, 197, 1, 768] [1, 1, 1, 1] : tensor<3x197x1x768xf32> to tensor<1x197x1x768xf32>
    %extracted_slice_14 = tensor.extract_slice %collapsed_13[1, 0, 0, 0] [1, 197, 1, 768] [1, 1, 1, 1] : tensor<3x197x1x768xf32> to tensor<1x197x1x768xf32>
    %extracted_slice_15 = tensor.extract_slice %collapsed_13[2, 0, 0, 0] [1, 197, 1, 768] [1, 1, 1, 1] : tensor<3x197x1x768xf32> to tensor<1x197x1x768xf32>
    %collapsed_16 = tensor.collapse_shape %extracted_slice [[0, 1], [2, 3]] : tensor<1x197x1x768xf32> into tensor<197x768xf32>
    %expanded_17 = tensor.expand_shape %collapsed_16 [[0], [1, 2]] output_shape [197, 12, 64] : tensor<197x768xf32> into tensor<197x12x64xf32>
    %7 = tensor.empty() : tensor<12x197x64xf32>
    %transposed_18 = linalg.transpose ins(%expanded_17 : tensor<197x12x64xf32>) outs(%7 : tensor<12x197x64xf32>) permutation = [1, 0, 2] 
    %collapsed_19 = tensor.collapse_shape %extracted_slice_14 [[0, 1], [2, 3]] : tensor<1x197x1x768xf32> into tensor<197x768xf32>
    %expanded_20 = tensor.expand_shape %collapsed_19 [[0], [1, 2]] output_shape [197, 12, 64] : tensor<197x768xf32> into tensor<197x12x64xf32>
    %transposed_21 = linalg.transpose ins(%expanded_20 : tensor<197x12x64xf32>) outs(%7 : tensor<12x197x64xf32>) permutation = [1, 0, 2] 
    %collapsed_22 = tensor.collapse_shape %extracted_slice_15 [[0, 1], [2, 3]] : tensor<1x197x1x768xf32> into tensor<197x768xf32>
    %expanded_23 = tensor.expand_shape %collapsed_22 [[0], [1, 2]] output_shape [197, 12, 64] : tensor<197x768xf32> into tensor<197x12x64xf32>
    %transposed_24 = linalg.transpose ins(%expanded_23 : tensor<197x12x64xf32>) outs(%7 : tensor<12x197x64xf32>) permutation = [1, 0, 2] 
    %expanded_25 = tensor.expand_shape %transposed_18 [[0, 1], [2], [3]] output_shape [1, 12, 197, 64] : tensor<12x197x64xf32> into tensor<1x12x197x64xf32>
    %expanded_26 = tensor.expand_shape %transposed_21 [[0, 1], [2], [3]] output_shape [1, 12, 197, 64] : tensor<12x197x64xf32> into tensor<1x12x197x64xf32>
    %expanded_27 = tensor.expand_shape %transposed_24 [[0, 1], [2], [3]] output_shape [1, 12, 197, 64] : tensor<12x197x64xf32> into tensor<1x12x197x64xf32>
    %8 = dnn.scaled_dot_product_attention %expanded_25, %expanded_26, %expanded_27 {parameter_indices = array<i32: 3, 4, 5, 6, 7>, parameters = [unit, 0.000000e+00, false, unit, false]} : (tensor<1x12x197x64xf32>, tensor<1x12x197x64xf32>, tensor<1x12x197x64xf32>) -> tensor<1x12x197x64xf32>
    %9 = tensor.empty() : tensor<197x1x12x64xf32>
    %transposed_28 = linalg.transpose ins(%8 : tensor<1x12x197x64xf32>) outs(%9 : tensor<197x1x12x64xf32>) permutation = [2, 0, 1, 3] 
    %collapsed_29 = tensor.collapse_shape %transposed_28 [[0], [1, 2, 3]] : tensor<197x1x12x64xf32> into tensor<197x768xf32>
    %10 = dnn.linear %collapsed_29, %cst_4, %cst_0 : tensor<197x768xf32>, tensor<768x768xf32>, tensor<768xf32> -> tensor<197x768xf32>
    %expanded_30 = tensor.expand_shape %10 [[0], [1, 2]] output_shape [197, 1, 768] : tensor<197x768xf32> into tensor<197x1x768xf32>
    %11 = tensor.empty() : tensor<1x197x768xf32>
    %transposed_31 = linalg.transpose ins(%expanded_30 : tensor<197x1x768xf32>) outs(%11 : tensor<1x197x768xf32>) permutation = [1, 0, 2] 
    %12 = dnn.add %transposed_31, %2 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %13 = dnn.layer_norm %12, %cst_0, %cst_0 {parameter_indices = array<i32: 1, 4, 5>, parameters = [[768], 9.9999999999999995E-7, false]} : (tensor<1x197x768xf32>, tensor<768xf32>, tensor<768xf32>) -> tensor<1x197x768xf32>
    %14 = dnn.linear %13, %cst_5, %cst_6 : tensor<1x197x768xf32>, tensor<3072x768xf32>, tensor<3072xf32> -> tensor<1x197x3072xf32>
    %15 = dnn.gelu %14 {parameter_indices = array<i32: 1>, parameters = ["none"]} : (tensor<1x197x3072xf32>) -> tensor<1x197x3072xf32>
    %16 = dnn.linear %15, %cst_7, %cst_0 : tensor<1x197x3072xf32>, tensor<768x3072xf32>, tensor<768xf32> -> tensor<1x197x768xf32>
    %17 = dnn.add %12, %16 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %18 = dnn.layer_norm %17, %cst_0, %cst_0 {parameter_indices = array<i32: 1, 4, 5>, parameters = [[768], 9.9999999999999995E-7, false]} : (tensor<1x197x768xf32>, tensor<768xf32>, tensor<768xf32>) -> tensor<1x197x768xf32>
    %transposed_32 = linalg.transpose ins(%18 : tensor<1x197x768xf32>) outs(%4 : tensor<197x1x768xf32>) permutation = [1, 0, 2] 
    %19 = dnn.linear %transposed_32, %cst_2, %cst_3 : tensor<197x1x768xf32>, tensor<2304x768xf32>, tensor<2304xf32> -> tensor<197x1x2304xf32>
    %expanded_33 = tensor.expand_shape %19 [[0, 1], [2], [3, 4]] output_shape [1, 197, 1, 3, 768] : tensor<197x1x2304xf32> into tensor<1x197x1x3x768xf32>
    %transposed_34 = linalg.transpose ins(%expanded_33 : tensor<1x197x1x3x768xf32>) outs(%6 : tensor<3x197x1x1x768xf32>) permutation = [3, 1, 2, 0, 4] 
    %collapsed_35 = tensor.collapse_shape %transposed_34 [[0], [1], [2, 3], [4]] : tensor<3x197x1x1x768xf32> into tensor<3x197x1x768xf32>
    %extracted_slice_36 = tensor.extract_slice %collapsed_35[0, 0, 0, 0] [1, 197, 1, 768] [1, 1, 1, 1] : tensor<3x197x1x768xf32> to tensor<1x197x1x768xf32>
    %extracted_slice_37 = tensor.extract_slice %collapsed_35[1, 0, 0, 0] [1, 197, 1, 768] [1, 1, 1, 1] : tensor<3x197x1x768xf32> to tensor<1x197x1x768xf32>
    %extracted_slice_38 = tensor.extract_slice %collapsed_35[2, 0, 0, 0] [1, 197, 1, 768] [1, 1, 1, 1] : tensor<3x197x1x768xf32> to tensor<1x197x1x768xf32>
    %collapsed_39 = tensor.collapse_shape %extracted_slice_36 [[0, 1], [2, 3]] : tensor<1x197x1x768xf32> into tensor<197x768xf32>
    %expanded_40 = tensor.expand_shape %collapsed_39 [[0], [1, 2]] output_shape [197, 12, 64] : tensor<197x768xf32> into tensor<197x12x64xf32>
    %transposed_41 = linalg.transpose ins(%expanded_40 : tensor<197x12x64xf32>) outs(%7 : tensor<12x197x64xf32>) permutation = [1, 0, 2] 
    %collapsed_42 = tensor.collapse_shape %extracted_slice_37 [[0, 1], [2, 3]] : tensor<1x197x1x768xf32> into tensor<197x768xf32>
    %expanded_43 = tensor.expand_shape %collapsed_42 [[0], [1, 2]] output_shape [197, 12, 64] : tensor<197x768xf32> into tensor<197x12x64xf32>
    %transposed_44 = linalg.transpose ins(%expanded_43 : tensor<197x12x64xf32>) outs(%7 : tensor<12x197x64xf32>) permutation = [1, 0, 2] 
    %collapsed_45 = tensor.collapse_shape %extracted_slice_38 [[0, 1], [2, 3]] : tensor<1x197x1x768xf32> into tensor<197x768xf32>
    %expanded_46 = tensor.expand_shape %collapsed_45 [[0], [1, 2]] output_shape [197, 12, 64] : tensor<197x768xf32> into tensor<197x12x64xf32>
    %transposed_47 = linalg.transpose ins(%expanded_46 : tensor<197x12x64xf32>) outs(%7 : tensor<12x197x64xf32>) permutation = [1, 0, 2] 
    %expanded_48 = tensor.expand_shape %transposed_41 [[0, 1], [2], [3]] output_shape [1, 12, 197, 64] : tensor<12x197x64xf32> into tensor<1x12x197x64xf32>
    %expanded_49 = tensor.expand_shape %transposed_44 [[0, 1], [2], [3]] output_shape [1, 12, 197, 64] : tensor<12x197x64xf32> into tensor<1x12x197x64xf32>
    %expanded_50 = tensor.expand_shape %transposed_47 [[0, 1], [2], [3]] output_shape [1, 12, 197, 64] : tensor<12x197x64xf32> into tensor<1x12x197x64xf32>
    %20 = dnn.scaled_dot_product_attention %expanded_48, %expanded_49, %expanded_50 {parameter_indices = array<i32: 3, 4, 5, 6, 7>, parameters = [unit, 0.000000e+00, false, unit, false]} : (tensor<1x12x197x64xf32>, tensor<1x12x197x64xf32>, tensor<1x12x197x64xf32>) -> tensor<1x12x197x64xf32>
    %transposed_51 = linalg.transpose ins(%20 : tensor<1x12x197x64xf32>) outs(%9 : tensor<197x1x12x64xf32>) permutation = [2, 0, 1, 3] 
    %collapsed_52 = tensor.collapse_shape %transposed_51 [[0], [1, 2, 3]] : tensor<197x1x12x64xf32> into tensor<197x768xf32>
    %21 = dnn.linear %collapsed_52, %cst_4, %cst_0 : tensor<197x768xf32>, tensor<768x768xf32>, tensor<768xf32> -> tensor<197x768xf32>
    %expanded_53 = tensor.expand_shape %21 [[0], [1, 2]] output_shape [197, 1, 768] : tensor<197x768xf32> into tensor<197x1x768xf32>
    %transposed_54 = linalg.transpose ins(%expanded_53 : tensor<197x1x768xf32>) outs(%11 : tensor<1x197x768xf32>) permutation = [1, 0, 2] 
    %22 = dnn.add %transposed_54, %17 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %23 = dnn.layer_norm %22, %cst_0, %cst_0 {parameter_indices = array<i32: 1, 4, 5>, parameters = [[768], 9.9999999999999995E-7, false]} : (tensor<1x197x768xf32>, tensor<768xf32>, tensor<768xf32>) -> tensor<1x197x768xf32>
    %24 = dnn.linear %23, %cst_5, %cst_6 : tensor<1x197x768xf32>, tensor<3072x768xf32>, tensor<3072xf32> -> tensor<1x197x3072xf32>
    %25 = dnn.gelu %24 {parameter_indices = array<i32: 1>, parameters = ["none"]} : (tensor<1x197x3072xf32>) -> tensor<1x197x3072xf32>
    %26 = dnn.linear %25, %cst_7, %cst_0 : tensor<1x197x3072xf32>, tensor<768x3072xf32>, tensor<768xf32> -> tensor<1x197x768xf32>
    %27 = dnn.add %22, %26 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %28 = dnn.layer_norm %27, %cst_0, %cst_0 {parameter_indices = array<i32: 1, 4, 5>, parameters = [[768], 9.9999999999999995E-7, false]} : (tensor<1x197x768xf32>, tensor<768xf32>, tensor<768xf32>) -> tensor<1x197x768xf32>
    %transposed_55 = linalg.transpose ins(%28 : tensor<1x197x768xf32>) outs(%4 : tensor<197x1x768xf32>) permutation = [1, 0, 2] 
    %29 = dnn.linear %transposed_55, %cst_2, %cst_3 : tensor<197x1x768xf32>, tensor<2304x768xf32>, tensor<2304xf32> -> tensor<197x1x2304xf32>
    %expanded_56 = tensor.expand_shape %29 [[0, 1], [2], [3, 4]] output_shape [1, 197, 1, 3, 768] : tensor<197x1x2304xf32> into tensor<1x197x1x3x768xf32>
    %transposed_57 = linalg.transpose ins(%expanded_56 : tensor<1x197x1x3x768xf32>) outs(%6 : tensor<3x197x1x1x768xf32>) permutation = [3, 1, 2, 0, 4] 
    %collapsed_58 = tensor.collapse_shape %transposed_57 [[0], [1], [2, 3], [4]] : tensor<3x197x1x1x768xf32> into tensor<3x197x1x768xf32>
    %extracted_slice_59 = tensor.extract_slice %collapsed_58[0, 0, 0, 0] [1, 197, 1, 768] [1, 1, 1, 1] : tensor<3x197x1x768xf32> to tensor<1x197x1x768xf32>
    %extracted_slice_60 = tensor.extract_slice %collapsed_58[1, 0, 0, 0] [1, 197, 1, 768] [1, 1, 1, 1] : tensor<3x197x1x768xf32> to tensor<1x197x1x768xf32>
    %extracted_slice_61 = tensor.extract_slice %collapsed_58[2, 0, 0, 0] [1, 197, 1, 768] [1, 1, 1, 1] : tensor<3x197x1x768xf32> to tensor<1x197x1x768xf32>
    %collapsed_62 = tensor.collapse_shape %extracted_slice_59 [[0, 1], [2, 3]] : tensor<1x197x1x768xf32> into tensor<197x768xf32>
    %expanded_63 = tensor.expand_shape %collapsed_62 [[0], [1, 2]] output_shape [197, 12, 64] : tensor<197x768xf32> into tensor<197x12x64xf32>
    %transposed_64 = linalg.transpose ins(%expanded_63 : tensor<197x12x64xf32>) outs(%7 : tensor<12x197x64xf32>) permutation = [1, 0, 2] 
    %collapsed_65 = tensor.collapse_shape %extracted_slice_60 [[0, 1], [2, 3]] : tensor<1x197x1x768xf32> into tensor<197x768xf32>
    %expanded_66 = tensor.expand_shape %collapsed_65 [[0], [1, 2]] output_shape [197, 12, 64] : tensor<197x768xf32> into tensor<197x12x64xf32>
    %transposed_67 = linalg.transpose ins(%expanded_66 : tensor<197x12x64xf32>) outs(%7 : tensor<12x197x64xf32>) permutation = [1, 0, 2] 
    %collapsed_68 = tensor.collapse_shape %extracted_slice_61 [[0, 1], [2, 3]] : tensor<1x197x1x768xf32> into tensor<197x768xf32>
    %expanded_69 = tensor.expand_shape %collapsed_68 [[0], [1, 2]] output_shape [197, 12, 64] : tensor<197x768xf32> into tensor<197x12x64xf32>
    %transposed_70 = linalg.transpose ins(%expanded_69 : tensor<197x12x64xf32>) outs(%7 : tensor<12x197x64xf32>) permutation = [1, 0, 2] 
    %expanded_71 = tensor.expand_shape %transposed_64 [[0, 1], [2], [3]] output_shape [1, 12, 197, 64] : tensor<12x197x64xf32> into tensor<1x12x197x64xf32>
    %expanded_72 = tensor.expand_shape %transposed_67 [[0, 1], [2], [3]] output_shape [1, 12, 197, 64] : tensor<12x197x64xf32> into tensor<1x12x197x64xf32>
    %expanded_73 = tensor.expand_shape %transposed_70 [[0, 1], [2], [3]] output_shape [1, 12, 197, 64] : tensor<12x197x64xf32> into tensor<1x12x197x64xf32>
    %30 = dnn.scaled_dot_product_attention %expanded_71, %expanded_72, %expanded_73 {parameter_indices = array<i32: 3, 4, 5, 6, 7>, parameters = [unit, 0.000000e+00, false, unit, false]} : (tensor<1x12x197x64xf32>, tensor<1x12x197x64xf32>, tensor<1x12x197x64xf32>) -> tensor<1x12x197x64xf32>
    %transposed_74 = linalg.transpose ins(%30 : tensor<1x12x197x64xf32>) outs(%9 : tensor<197x1x12x64xf32>) permutation = [2, 0, 1, 3] 
    %collapsed_75 = tensor.collapse_shape %transposed_74 [[0], [1, 2, 3]] : tensor<197x1x12x64xf32> into tensor<197x768xf32>
    %31 = dnn.linear %collapsed_75, %cst_4, %cst_0 : tensor<197x768xf32>, tensor<768x768xf32>, tensor<768xf32> -> tensor<197x768xf32>
    %expanded_76 = tensor.expand_shape %31 [[0], [1, 2]] output_shape [197, 1, 768] : tensor<197x768xf32> into tensor<197x1x768xf32>
    %transposed_77 = linalg.transpose ins(%expanded_76 : tensor<197x1x768xf32>) outs(%11 : tensor<1x197x768xf32>) permutation = [1, 0, 2] 
    %32 = dnn.add %transposed_77, %27 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %33 = dnn.layer_norm %32, %cst_0, %cst_0 {parameter_indices = array<i32: 1, 4, 5>, parameters = [[768], 9.9999999999999995E-7, false]} : (tensor<1x197x768xf32>, tensor<768xf32>, tensor<768xf32>) -> tensor<1x197x768xf32>
    %34 = dnn.linear %33, %cst_5, %cst_6 : tensor<1x197x768xf32>, tensor<3072x768xf32>, tensor<3072xf32> -> tensor<1x197x3072xf32>
    %35 = dnn.gelu %34 {parameter_indices = array<i32: 1>, parameters = ["none"]} : (tensor<1x197x3072xf32>) -> tensor<1x197x3072xf32>
    %36 = dnn.linear %35, %cst_7, %cst_0 : tensor<1x197x3072xf32>, tensor<768x3072xf32>, tensor<768xf32> -> tensor<1x197x768xf32>
    %37 = dnn.add %32, %36 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %38 = dnn.layer_norm %37, %cst_0, %cst_0 {parameter_indices = array<i32: 1, 4, 5>, parameters = [[768], 9.9999999999999995E-7, false]} : (tensor<1x197x768xf32>, tensor<768xf32>, tensor<768xf32>) -> tensor<1x197x768xf32>
    %transposed_78 = linalg.transpose ins(%38 : tensor<1x197x768xf32>) outs(%4 : tensor<197x1x768xf32>) permutation = [1, 0, 2] 
    %39 = dnn.linear %transposed_78, %cst_2, %cst_3 : tensor<197x1x768xf32>, tensor<2304x768xf32>, tensor<2304xf32> -> tensor<197x1x2304xf32>
    %expanded_79 = tensor.expand_shape %39 [[0, 1], [2], [3, 4]] output_shape [1, 197, 1, 3, 768] : tensor<197x1x2304xf32> into tensor<1x197x1x3x768xf32>
    %transposed_80 = linalg.transpose ins(%expanded_79 : tensor<1x197x1x3x768xf32>) outs(%6 : tensor<3x197x1x1x768xf32>) permutation = [3, 1, 2, 0, 4] 
    %collapsed_81 = tensor.collapse_shape %transposed_80 [[0], [1], [2, 3], [4]] : tensor<3x197x1x1x768xf32> into tensor<3x197x1x768xf32>
    %extracted_slice_82 = tensor.extract_slice %collapsed_81[0, 0, 0, 0] [1, 197, 1, 768] [1, 1, 1, 1] : tensor<3x197x1x768xf32> to tensor<1x197x1x768xf32>
    %extracted_slice_83 = tensor.extract_slice %collapsed_81[1, 0, 0, 0] [1, 197, 1, 768] [1, 1, 1, 1] : tensor<3x197x1x768xf32> to tensor<1x197x1x768xf32>
    %extracted_slice_84 = tensor.extract_slice %collapsed_81[2, 0, 0, 0] [1, 197, 1, 768] [1, 1, 1, 1] : tensor<3x197x1x768xf32> to tensor<1x197x1x768xf32>
    %collapsed_85 = tensor.collapse_shape %extracted_slice_82 [[0, 1], [2, 3]] : tensor<1x197x1x768xf32> into tensor<197x768xf32>
    %expanded_86 = tensor.expand_shape %collapsed_85 [[0], [1, 2]] output_shape [197, 12, 64] : tensor<197x768xf32> into tensor<197x12x64xf32>
    %transposed_87 = linalg.transpose ins(%expanded_86 : tensor<197x12x64xf32>) outs(%7 : tensor<12x197x64xf32>) permutation = [1, 0, 2] 
    %collapsed_88 = tensor.collapse_shape %extracted_slice_83 [[0, 1], [2, 3]] : tensor<1x197x1x768xf32> into tensor<197x768xf32>
    %expanded_89 = tensor.expand_shape %collapsed_88 [[0], [1, 2]] output_shape [197, 12, 64] : tensor<197x768xf32> into tensor<197x12x64xf32>
    %transposed_90 = linalg.transpose ins(%expanded_89 : tensor<197x12x64xf32>) outs(%7 : tensor<12x197x64xf32>) permutation = [1, 0, 2] 
    %collapsed_91 = tensor.collapse_shape %extracted_slice_84 [[0, 1], [2, 3]] : tensor<1x197x1x768xf32> into tensor<197x768xf32>
    %expanded_92 = tensor.expand_shape %collapsed_91 [[0], [1, 2]] output_shape [197, 12, 64] : tensor<197x768xf32> into tensor<197x12x64xf32>
    %transposed_93 = linalg.transpose ins(%expanded_92 : tensor<197x12x64xf32>) outs(%7 : tensor<12x197x64xf32>) permutation = [1, 0, 2] 
    %expanded_94 = tensor.expand_shape %transposed_87 [[0, 1], [2], [3]] output_shape [1, 12, 197, 64] : tensor<12x197x64xf32> into tensor<1x12x197x64xf32>
    %expanded_95 = tensor.expand_shape %transposed_90 [[0, 1], [2], [3]] output_shape [1, 12, 197, 64] : tensor<12x197x64xf32> into tensor<1x12x197x64xf32>
    %expanded_96 = tensor.expand_shape %transposed_93 [[0, 1], [2], [3]] output_shape [1, 12, 197, 64] : tensor<12x197x64xf32> into tensor<1x12x197x64xf32>
    %40 = dnn.scaled_dot_product_attention %expanded_94, %expanded_95, %expanded_96 {parameter_indices = array<i32: 3, 4, 5, 6, 7>, parameters = [unit, 0.000000e+00, false, unit, false]} : (tensor<1x12x197x64xf32>, tensor<1x12x197x64xf32>, tensor<1x12x197x64xf32>) -> tensor<1x12x197x64xf32>
    %transposed_97 = linalg.transpose ins(%40 : tensor<1x12x197x64xf32>) outs(%9 : tensor<197x1x12x64xf32>) permutation = [2, 0, 1, 3] 
    %collapsed_98 = tensor.collapse_shape %transposed_97 [[0], [1, 2, 3]] : tensor<197x1x12x64xf32> into tensor<197x768xf32>
    %41 = dnn.linear %collapsed_98, %cst_4, %cst_0 : tensor<197x768xf32>, tensor<768x768xf32>, tensor<768xf32> -> tensor<197x768xf32>
    %expanded_99 = tensor.expand_shape %41 [[0], [1, 2]] output_shape [197, 1, 768] : tensor<197x768xf32> into tensor<197x1x768xf32>
    %transposed_100 = linalg.transpose ins(%expanded_99 : tensor<197x1x768xf32>) outs(%11 : tensor<1x197x768xf32>) permutation = [1, 0, 2] 
    %42 = dnn.add %transposed_100, %37 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %43 = dnn.layer_norm %42, %cst_0, %cst_0 {parameter_indices = array<i32: 1, 4, 5>, parameters = [[768], 9.9999999999999995E-7, false]} : (tensor<1x197x768xf32>, tensor<768xf32>, tensor<768xf32>) -> tensor<1x197x768xf32>
    %44 = dnn.linear %43, %cst_5, %cst_6 : tensor<1x197x768xf32>, tensor<3072x768xf32>, tensor<3072xf32> -> tensor<1x197x3072xf32>
    %45 = dnn.gelu %44 {parameter_indices = array<i32: 1>, parameters = ["none"]} : (tensor<1x197x3072xf32>) -> tensor<1x197x3072xf32>
    %46 = dnn.linear %45, %cst_7, %cst_0 : tensor<1x197x3072xf32>, tensor<768x3072xf32>, tensor<768xf32> -> tensor<1x197x768xf32>
    %47 = dnn.add %42, %46 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %48 = dnn.layer_norm %47, %cst_0, %cst_0 {parameter_indices = array<i32: 1, 4, 5>, parameters = [[768], 9.9999999999999995E-7, false]} : (tensor<1x197x768xf32>, tensor<768xf32>, tensor<768xf32>) -> tensor<1x197x768xf32>
    %transposed_101 = linalg.transpose ins(%48 : tensor<1x197x768xf32>) outs(%4 : tensor<197x1x768xf32>) permutation = [1, 0, 2] 
    %49 = dnn.linear %transposed_101, %cst_2, %cst_3 : tensor<197x1x768xf32>, tensor<2304x768xf32>, tensor<2304xf32> -> tensor<197x1x2304xf32>
    %expanded_102 = tensor.expand_shape %49 [[0, 1], [2], [3, 4]] output_shape [1, 197, 1, 3, 768] : tensor<197x1x2304xf32> into tensor<1x197x1x3x768xf32>
    %transposed_103 = linalg.transpose ins(%expanded_102 : tensor<1x197x1x3x768xf32>) outs(%6 : tensor<3x197x1x1x768xf32>) permutation = [3, 1, 2, 0, 4] 
    %collapsed_104 = tensor.collapse_shape %transposed_103 [[0], [1], [2, 3], [4]] : tensor<3x197x1x1x768xf32> into tensor<3x197x1x768xf32>
    %extracted_slice_105 = tensor.extract_slice %collapsed_104[0, 0, 0, 0] [1, 197, 1, 768] [1, 1, 1, 1] : tensor<3x197x1x768xf32> to tensor<1x197x1x768xf32>
    %extracted_slice_106 = tensor.extract_slice %collapsed_104[1, 0, 0, 0] [1, 197, 1, 768] [1, 1, 1, 1] : tensor<3x197x1x768xf32> to tensor<1x197x1x768xf32>
    %extracted_slice_107 = tensor.extract_slice %collapsed_104[2, 0, 0, 0] [1, 197, 1, 768] [1, 1, 1, 1] : tensor<3x197x1x768xf32> to tensor<1x197x1x768xf32>
    %collapsed_108 = tensor.collapse_shape %extracted_slice_105 [[0, 1], [2, 3]] : tensor<1x197x1x768xf32> into tensor<197x768xf32>
    %expanded_109 = tensor.expand_shape %collapsed_108 [[0], [1, 2]] output_shape [197, 12, 64] : tensor<197x768xf32> into tensor<197x12x64xf32>
    %transposed_110 = linalg.transpose ins(%expanded_109 : tensor<197x12x64xf32>) outs(%7 : tensor<12x197x64xf32>) permutation = [1, 0, 2] 
    %collapsed_111 = tensor.collapse_shape %extracted_slice_106 [[0, 1], [2, 3]] : tensor<1x197x1x768xf32> into tensor<197x768xf32>
    %expanded_112 = tensor.expand_shape %collapsed_111 [[0], [1, 2]] output_shape [197, 12, 64] : tensor<197x768xf32> into tensor<197x12x64xf32>
    %transposed_113 = linalg.transpose ins(%expanded_112 : tensor<197x12x64xf32>) outs(%7 : tensor<12x197x64xf32>) permutation = [1, 0, 2] 
    %collapsed_114 = tensor.collapse_shape %extracted_slice_107 [[0, 1], [2, 3]] : tensor<1x197x1x768xf32> into tensor<197x768xf32>
    %expanded_115 = tensor.expand_shape %collapsed_114 [[0], [1, 2]] output_shape [197, 12, 64] : tensor<197x768xf32> into tensor<197x12x64xf32>
    %transposed_116 = linalg.transpose ins(%expanded_115 : tensor<197x12x64xf32>) outs(%7 : tensor<12x197x64xf32>) permutation = [1, 0, 2] 
    %expanded_117 = tensor.expand_shape %transposed_110 [[0, 1], [2], [3]] output_shape [1, 12, 197, 64] : tensor<12x197x64xf32> into tensor<1x12x197x64xf32>
    %expanded_118 = tensor.expand_shape %transposed_113 [[0, 1], [2], [3]] output_shape [1, 12, 197, 64] : tensor<12x197x64xf32> into tensor<1x12x197x64xf32>
    %expanded_119 = tensor.expand_shape %transposed_116 [[0, 1], [2], [3]] output_shape [1, 12, 197, 64] : tensor<12x197x64xf32> into tensor<1x12x197x64xf32>
    %50 = dnn.scaled_dot_product_attention %expanded_117, %expanded_118, %expanded_119 {parameter_indices = array<i32: 3, 4, 5, 6, 7>, parameters = [unit, 0.000000e+00, false, unit, false]} : (tensor<1x12x197x64xf32>, tensor<1x12x197x64xf32>, tensor<1x12x197x64xf32>) -> tensor<1x12x197x64xf32>
    %transposed_120 = linalg.transpose ins(%50 : tensor<1x12x197x64xf32>) outs(%9 : tensor<197x1x12x64xf32>) permutation = [2, 0, 1, 3] 
    %collapsed_121 = tensor.collapse_shape %transposed_120 [[0], [1, 2, 3]] : tensor<197x1x12x64xf32> into tensor<197x768xf32>
    %51 = dnn.linear %collapsed_121, %cst_4, %cst_0 : tensor<197x768xf32>, tensor<768x768xf32>, tensor<768xf32> -> tensor<197x768xf32>
    %expanded_122 = tensor.expand_shape %51 [[0], [1, 2]] output_shape [197, 1, 768] : tensor<197x768xf32> into tensor<197x1x768xf32>
    %transposed_123 = linalg.transpose ins(%expanded_122 : tensor<197x1x768xf32>) outs(%11 : tensor<1x197x768xf32>) permutation = [1, 0, 2] 
    %52 = dnn.add %transposed_123, %47 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %53 = dnn.layer_norm %52, %cst_0, %cst_0 {parameter_indices = array<i32: 1, 4, 5>, parameters = [[768], 9.9999999999999995E-7, false]} : (tensor<1x197x768xf32>, tensor<768xf32>, tensor<768xf32>) -> tensor<1x197x768xf32>
    %54 = dnn.linear %53, %cst_5, %cst_6 : tensor<1x197x768xf32>, tensor<3072x768xf32>, tensor<3072xf32> -> tensor<1x197x3072xf32>
    %55 = dnn.gelu %54 {parameter_indices = array<i32: 1>, parameters = ["none"]} : (tensor<1x197x3072xf32>) -> tensor<1x197x3072xf32>
    %56 = dnn.linear %55, %cst_7, %cst_0 : tensor<1x197x3072xf32>, tensor<768x3072xf32>, tensor<768xf32> -> tensor<1x197x768xf32>
    %57 = dnn.add %52, %56 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %58 = dnn.layer_norm %57, %cst_0, %cst_0 {parameter_indices = array<i32: 1, 4, 5>, parameters = [[768], 9.9999999999999995E-7, false]} : (tensor<1x197x768xf32>, tensor<768xf32>, tensor<768xf32>) -> tensor<1x197x768xf32>
    %transposed_124 = linalg.transpose ins(%58 : tensor<1x197x768xf32>) outs(%4 : tensor<197x1x768xf32>) permutation = [1, 0, 2] 
    %59 = dnn.linear %transposed_124, %cst_2, %cst_3 : tensor<197x1x768xf32>, tensor<2304x768xf32>, tensor<2304xf32> -> tensor<197x1x2304xf32>
    %expanded_125 = tensor.expand_shape %59 [[0, 1], [2], [3, 4]] output_shape [1, 197, 1, 3, 768] : tensor<197x1x2304xf32> into tensor<1x197x1x3x768xf32>
    %transposed_126 = linalg.transpose ins(%expanded_125 : tensor<1x197x1x3x768xf32>) outs(%6 : tensor<3x197x1x1x768xf32>) permutation = [3, 1, 2, 0, 4] 
    %collapsed_127 = tensor.collapse_shape %transposed_126 [[0], [1], [2, 3], [4]] : tensor<3x197x1x1x768xf32> into tensor<3x197x1x768xf32>
    %extracted_slice_128 = tensor.extract_slice %collapsed_127[0, 0, 0, 0] [1, 197, 1, 768] [1, 1, 1, 1] : tensor<3x197x1x768xf32> to tensor<1x197x1x768xf32>
    %extracted_slice_129 = tensor.extract_slice %collapsed_127[1, 0, 0, 0] [1, 197, 1, 768] [1, 1, 1, 1] : tensor<3x197x1x768xf32> to tensor<1x197x1x768xf32>
    %extracted_slice_130 = tensor.extract_slice %collapsed_127[2, 0, 0, 0] [1, 197, 1, 768] [1, 1, 1, 1] : tensor<3x197x1x768xf32> to tensor<1x197x1x768xf32>
    %collapsed_131 = tensor.collapse_shape %extracted_slice_128 [[0, 1], [2, 3]] : tensor<1x197x1x768xf32> into tensor<197x768xf32>
    %expanded_132 = tensor.expand_shape %collapsed_131 [[0], [1, 2]] output_shape [197, 12, 64] : tensor<197x768xf32> into tensor<197x12x64xf32>
    %transposed_133 = linalg.transpose ins(%expanded_132 : tensor<197x12x64xf32>) outs(%7 : tensor<12x197x64xf32>) permutation = [1, 0, 2] 
    %collapsed_134 = tensor.collapse_shape %extracted_slice_129 [[0, 1], [2, 3]] : tensor<1x197x1x768xf32> into tensor<197x768xf32>
    %expanded_135 = tensor.expand_shape %collapsed_134 [[0], [1, 2]] output_shape [197, 12, 64] : tensor<197x768xf32> into tensor<197x12x64xf32>
    %transposed_136 = linalg.transpose ins(%expanded_135 : tensor<197x12x64xf32>) outs(%7 : tensor<12x197x64xf32>) permutation = [1, 0, 2] 
    %collapsed_137 = tensor.collapse_shape %extracted_slice_130 [[0, 1], [2, 3]] : tensor<1x197x1x768xf32> into tensor<197x768xf32>
    %expanded_138 = tensor.expand_shape %collapsed_137 [[0], [1, 2]] output_shape [197, 12, 64] : tensor<197x768xf32> into tensor<197x12x64xf32>
    %transposed_139 = linalg.transpose ins(%expanded_138 : tensor<197x12x64xf32>) outs(%7 : tensor<12x197x64xf32>) permutation = [1, 0, 2] 
    %expanded_140 = tensor.expand_shape %transposed_133 [[0, 1], [2], [3]] output_shape [1, 12, 197, 64] : tensor<12x197x64xf32> into tensor<1x12x197x64xf32>
    %expanded_141 = tensor.expand_shape %transposed_136 [[0, 1], [2], [3]] output_shape [1, 12, 197, 64] : tensor<12x197x64xf32> into tensor<1x12x197x64xf32>
    %expanded_142 = tensor.expand_shape %transposed_139 [[0, 1], [2], [3]] output_shape [1, 12, 197, 64] : tensor<12x197x64xf32> into tensor<1x12x197x64xf32>
    %60 = dnn.scaled_dot_product_attention %expanded_140, %expanded_141, %expanded_142 {parameter_indices = array<i32: 3, 4, 5, 6, 7>, parameters = [unit, 0.000000e+00, false, unit, false]} : (tensor<1x12x197x64xf32>, tensor<1x12x197x64xf32>, tensor<1x12x197x64xf32>) -> tensor<1x12x197x64xf32>
    %transposed_143 = linalg.transpose ins(%60 : tensor<1x12x197x64xf32>) outs(%9 : tensor<197x1x12x64xf32>) permutation = [2, 0, 1, 3] 
    %collapsed_144 = tensor.collapse_shape %transposed_143 [[0], [1, 2, 3]] : tensor<197x1x12x64xf32> into tensor<197x768xf32>
    %61 = dnn.linear %collapsed_144, %cst_4, %cst_0 : tensor<197x768xf32>, tensor<768x768xf32>, tensor<768xf32> -> tensor<197x768xf32>
    %expanded_145 = tensor.expand_shape %61 [[0], [1, 2]] output_shape [197, 1, 768] : tensor<197x768xf32> into tensor<197x1x768xf32>
    %transposed_146 = linalg.transpose ins(%expanded_145 : tensor<197x1x768xf32>) outs(%11 : tensor<1x197x768xf32>) permutation = [1, 0, 2] 
    %62 = dnn.add %transposed_146, %57 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %63 = dnn.layer_norm %62, %cst_0, %cst_0 {parameter_indices = array<i32: 1, 4, 5>, parameters = [[768], 9.9999999999999995E-7, false]} : (tensor<1x197x768xf32>, tensor<768xf32>, tensor<768xf32>) -> tensor<1x197x768xf32>
    %64 = dnn.linear %63, %cst_5, %cst_6 : tensor<1x197x768xf32>, tensor<3072x768xf32>, tensor<3072xf32> -> tensor<1x197x3072xf32>
    %65 = dnn.gelu %64 {parameter_indices = array<i32: 1>, parameters = ["none"]} : (tensor<1x197x3072xf32>) -> tensor<1x197x3072xf32>
    %66 = dnn.linear %65, %cst_7, %cst_0 : tensor<1x197x3072xf32>, tensor<768x3072xf32>, tensor<768xf32> -> tensor<1x197x768xf32>
    %67 = dnn.add %62, %66 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %68 = dnn.layer_norm %67, %cst_0, %cst_0 {parameter_indices = array<i32: 1, 4, 5>, parameters = [[768], 9.9999999999999995E-7, false]} : (tensor<1x197x768xf32>, tensor<768xf32>, tensor<768xf32>) -> tensor<1x197x768xf32>
    %transposed_147 = linalg.transpose ins(%68 : tensor<1x197x768xf32>) outs(%4 : tensor<197x1x768xf32>) permutation = [1, 0, 2] 
    %69 = dnn.linear %transposed_147, %cst_2, %cst_3 : tensor<197x1x768xf32>, tensor<2304x768xf32>, tensor<2304xf32> -> tensor<197x1x2304xf32>
    %expanded_148 = tensor.expand_shape %69 [[0, 1], [2], [3, 4]] output_shape [1, 197, 1, 3, 768] : tensor<197x1x2304xf32> into tensor<1x197x1x3x768xf32>
    %transposed_149 = linalg.transpose ins(%expanded_148 : tensor<1x197x1x3x768xf32>) outs(%6 : tensor<3x197x1x1x768xf32>) permutation = [3, 1, 2, 0, 4] 
    %collapsed_150 = tensor.collapse_shape %transposed_149 [[0], [1], [2, 3], [4]] : tensor<3x197x1x1x768xf32> into tensor<3x197x1x768xf32>
    %extracted_slice_151 = tensor.extract_slice %collapsed_150[0, 0, 0, 0] [1, 197, 1, 768] [1, 1, 1, 1] : tensor<3x197x1x768xf32> to tensor<1x197x1x768xf32>
    %extracted_slice_152 = tensor.extract_slice %collapsed_150[1, 0, 0, 0] [1, 197, 1, 768] [1, 1, 1, 1] : tensor<3x197x1x768xf32> to tensor<1x197x1x768xf32>
    %extracted_slice_153 = tensor.extract_slice %collapsed_150[2, 0, 0, 0] [1, 197, 1, 768] [1, 1, 1, 1] : tensor<3x197x1x768xf32> to tensor<1x197x1x768xf32>
    %collapsed_154 = tensor.collapse_shape %extracted_slice_151 [[0, 1], [2, 3]] : tensor<1x197x1x768xf32> into tensor<197x768xf32>
    %expanded_155 = tensor.expand_shape %collapsed_154 [[0], [1, 2]] output_shape [197, 12, 64] : tensor<197x768xf32> into tensor<197x12x64xf32>
    %transposed_156 = linalg.transpose ins(%expanded_155 : tensor<197x12x64xf32>) outs(%7 : tensor<12x197x64xf32>) permutation = [1, 0, 2] 
    %collapsed_157 = tensor.collapse_shape %extracted_slice_152 [[0, 1], [2, 3]] : tensor<1x197x1x768xf32> into tensor<197x768xf32>
    %expanded_158 = tensor.expand_shape %collapsed_157 [[0], [1, 2]] output_shape [197, 12, 64] : tensor<197x768xf32> into tensor<197x12x64xf32>
    %transposed_159 = linalg.transpose ins(%expanded_158 : tensor<197x12x64xf32>) outs(%7 : tensor<12x197x64xf32>) permutation = [1, 0, 2] 
    %collapsed_160 = tensor.collapse_shape %extracted_slice_153 [[0, 1], [2, 3]] : tensor<1x197x1x768xf32> into tensor<197x768xf32>
    %expanded_161 = tensor.expand_shape %collapsed_160 [[0], [1, 2]] output_shape [197, 12, 64] : tensor<197x768xf32> into tensor<197x12x64xf32>
    %transposed_162 = linalg.transpose ins(%expanded_161 : tensor<197x12x64xf32>) outs(%7 : tensor<12x197x64xf32>) permutation = [1, 0, 2] 
    %expanded_163 = tensor.expand_shape %transposed_156 [[0, 1], [2], [3]] output_shape [1, 12, 197, 64] : tensor<12x197x64xf32> into tensor<1x12x197x64xf32>
    %expanded_164 = tensor.expand_shape %transposed_159 [[0, 1], [2], [3]] output_shape [1, 12, 197, 64] : tensor<12x197x64xf32> into tensor<1x12x197x64xf32>
    %expanded_165 = tensor.expand_shape %transposed_162 [[0, 1], [2], [3]] output_shape [1, 12, 197, 64] : tensor<12x197x64xf32> into tensor<1x12x197x64xf32>
    %70 = dnn.scaled_dot_product_attention %expanded_163, %expanded_164, %expanded_165 {parameter_indices = array<i32: 3, 4, 5, 6, 7>, parameters = [unit, 0.000000e+00, false, unit, false]} : (tensor<1x12x197x64xf32>, tensor<1x12x197x64xf32>, tensor<1x12x197x64xf32>) -> tensor<1x12x197x64xf32>
    %transposed_166 = linalg.transpose ins(%70 : tensor<1x12x197x64xf32>) outs(%9 : tensor<197x1x12x64xf32>) permutation = [2, 0, 1, 3] 
    %collapsed_167 = tensor.collapse_shape %transposed_166 [[0], [1, 2, 3]] : tensor<197x1x12x64xf32> into tensor<197x768xf32>
    %71 = dnn.linear %collapsed_167, %cst_4, %cst_0 : tensor<197x768xf32>, tensor<768x768xf32>, tensor<768xf32> -> tensor<197x768xf32>
    %expanded_168 = tensor.expand_shape %71 [[0], [1, 2]] output_shape [197, 1, 768] : tensor<197x768xf32> into tensor<197x1x768xf32>
    %transposed_169 = linalg.transpose ins(%expanded_168 : tensor<197x1x768xf32>) outs(%11 : tensor<1x197x768xf32>) permutation = [1, 0, 2] 
    %72 = dnn.add %transposed_169, %67 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %73 = dnn.layer_norm %72, %cst_0, %cst_0 {parameter_indices = array<i32: 1, 4, 5>, parameters = [[768], 9.9999999999999995E-7, false]} : (tensor<1x197x768xf32>, tensor<768xf32>, tensor<768xf32>) -> tensor<1x197x768xf32>
    %74 = dnn.linear %73, %cst_5, %cst_6 : tensor<1x197x768xf32>, tensor<3072x768xf32>, tensor<3072xf32> -> tensor<1x197x3072xf32>
    %75 = dnn.gelu %74 {parameter_indices = array<i32: 1>, parameters = ["none"]} : (tensor<1x197x3072xf32>) -> tensor<1x197x3072xf32>
    %76 = dnn.linear %75, %cst_7, %cst_0 : tensor<1x197x3072xf32>, tensor<768x3072xf32>, tensor<768xf32> -> tensor<1x197x768xf32>
    %77 = dnn.add %72, %76 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %78 = dnn.layer_norm %77, %cst_0, %cst_0 {parameter_indices = array<i32: 1, 4, 5>, parameters = [[768], 9.9999999999999995E-7, false]} : (tensor<1x197x768xf32>, tensor<768xf32>, tensor<768xf32>) -> tensor<1x197x768xf32>
    %transposed_170 = linalg.transpose ins(%78 : tensor<1x197x768xf32>) outs(%4 : tensor<197x1x768xf32>) permutation = [1, 0, 2] 
    %79 = dnn.linear %transposed_170, %cst_2, %cst_3 : tensor<197x1x768xf32>, tensor<2304x768xf32>, tensor<2304xf32> -> tensor<197x1x2304xf32>
    %expanded_171 = tensor.expand_shape %79 [[0, 1], [2], [3, 4]] output_shape [1, 197, 1, 3, 768] : tensor<197x1x2304xf32> into tensor<1x197x1x3x768xf32>
    %transposed_172 = linalg.transpose ins(%expanded_171 : tensor<1x197x1x3x768xf32>) outs(%6 : tensor<3x197x1x1x768xf32>) permutation = [3, 1, 2, 0, 4] 
    %collapsed_173 = tensor.collapse_shape %transposed_172 [[0], [1], [2, 3], [4]] : tensor<3x197x1x1x768xf32> into tensor<3x197x1x768xf32>
    %extracted_slice_174 = tensor.extract_slice %collapsed_173[0, 0, 0, 0] [1, 197, 1, 768] [1, 1, 1, 1] : tensor<3x197x1x768xf32> to tensor<1x197x1x768xf32>
    %extracted_slice_175 = tensor.extract_slice %collapsed_173[1, 0, 0, 0] [1, 197, 1, 768] [1, 1, 1, 1] : tensor<3x197x1x768xf32> to tensor<1x197x1x768xf32>
    %extracted_slice_176 = tensor.extract_slice %collapsed_173[2, 0, 0, 0] [1, 197, 1, 768] [1, 1, 1, 1] : tensor<3x197x1x768xf32> to tensor<1x197x1x768xf32>
    %collapsed_177 = tensor.collapse_shape %extracted_slice_174 [[0, 1], [2, 3]] : tensor<1x197x1x768xf32> into tensor<197x768xf32>
    %expanded_178 = tensor.expand_shape %collapsed_177 [[0], [1, 2]] output_shape [197, 12, 64] : tensor<197x768xf32> into tensor<197x12x64xf32>
    %transposed_179 = linalg.transpose ins(%expanded_178 : tensor<197x12x64xf32>) outs(%7 : tensor<12x197x64xf32>) permutation = [1, 0, 2] 
    %collapsed_180 = tensor.collapse_shape %extracted_slice_175 [[0, 1], [2, 3]] : tensor<1x197x1x768xf32> into tensor<197x768xf32>
    %expanded_181 = tensor.expand_shape %collapsed_180 [[0], [1, 2]] output_shape [197, 12, 64] : tensor<197x768xf32> into tensor<197x12x64xf32>
    %transposed_182 = linalg.transpose ins(%expanded_181 : tensor<197x12x64xf32>) outs(%7 : tensor<12x197x64xf32>) permutation = [1, 0, 2] 
    %collapsed_183 = tensor.collapse_shape %extracted_slice_176 [[0, 1], [2, 3]] : tensor<1x197x1x768xf32> into tensor<197x768xf32>
    %expanded_184 = tensor.expand_shape %collapsed_183 [[0], [1, 2]] output_shape [197, 12, 64] : tensor<197x768xf32> into tensor<197x12x64xf32>
    %transposed_185 = linalg.transpose ins(%expanded_184 : tensor<197x12x64xf32>) outs(%7 : tensor<12x197x64xf32>) permutation = [1, 0, 2] 
    %expanded_186 = tensor.expand_shape %transposed_179 [[0, 1], [2], [3]] output_shape [1, 12, 197, 64] : tensor<12x197x64xf32> into tensor<1x12x197x64xf32>
    %expanded_187 = tensor.expand_shape %transposed_182 [[0, 1], [2], [3]] output_shape [1, 12, 197, 64] : tensor<12x197x64xf32> into tensor<1x12x197x64xf32>
    %expanded_188 = tensor.expand_shape %transposed_185 [[0, 1], [2], [3]] output_shape [1, 12, 197, 64] : tensor<12x197x64xf32> into tensor<1x12x197x64xf32>
    %80 = dnn.scaled_dot_product_attention %expanded_186, %expanded_187, %expanded_188 {parameter_indices = array<i32: 3, 4, 5, 6, 7>, parameters = [unit, 0.000000e+00, false, unit, false]} : (tensor<1x12x197x64xf32>, tensor<1x12x197x64xf32>, tensor<1x12x197x64xf32>) -> tensor<1x12x197x64xf32>
    %transposed_189 = linalg.transpose ins(%80 : tensor<1x12x197x64xf32>) outs(%9 : tensor<197x1x12x64xf32>) permutation = [2, 0, 1, 3] 
    %collapsed_190 = tensor.collapse_shape %transposed_189 [[0], [1, 2, 3]] : tensor<197x1x12x64xf32> into tensor<197x768xf32>
    %81 = dnn.linear %collapsed_190, %cst_4, %cst_0 : tensor<197x768xf32>, tensor<768x768xf32>, tensor<768xf32> -> tensor<197x768xf32>
    %expanded_191 = tensor.expand_shape %81 [[0], [1, 2]] output_shape [197, 1, 768] : tensor<197x768xf32> into tensor<197x1x768xf32>
    %transposed_192 = linalg.transpose ins(%expanded_191 : tensor<197x1x768xf32>) outs(%11 : tensor<1x197x768xf32>) permutation = [1, 0, 2] 
    %82 = dnn.add %transposed_192, %77 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %83 = dnn.layer_norm %82, %cst_0, %cst_0 {parameter_indices = array<i32: 1, 4, 5>, parameters = [[768], 9.9999999999999995E-7, false]} : (tensor<1x197x768xf32>, tensor<768xf32>, tensor<768xf32>) -> tensor<1x197x768xf32>
    %84 = dnn.linear %83, %cst_5, %cst_6 : tensor<1x197x768xf32>, tensor<3072x768xf32>, tensor<3072xf32> -> tensor<1x197x3072xf32>
    %85 = dnn.gelu %84 {parameter_indices = array<i32: 1>, parameters = ["none"]} : (tensor<1x197x3072xf32>) -> tensor<1x197x3072xf32>
    %86 = dnn.linear %85, %cst_7, %cst_0 : tensor<1x197x3072xf32>, tensor<768x3072xf32>, tensor<768xf32> -> tensor<1x197x768xf32>
    %87 = dnn.add %82, %86 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %88 = dnn.layer_norm %87, %cst_0, %cst_0 {parameter_indices = array<i32: 1, 4, 5>, parameters = [[768], 9.9999999999999995E-7, false]} : (tensor<1x197x768xf32>, tensor<768xf32>, tensor<768xf32>) -> tensor<1x197x768xf32>
    %transposed_193 = linalg.transpose ins(%88 : tensor<1x197x768xf32>) outs(%4 : tensor<197x1x768xf32>) permutation = [1, 0, 2] 
    %89 = dnn.linear %transposed_193, %cst_2, %cst_3 : tensor<197x1x768xf32>, tensor<2304x768xf32>, tensor<2304xf32> -> tensor<197x1x2304xf32>
    %expanded_194 = tensor.expand_shape %89 [[0, 1], [2], [3, 4]] output_shape [1, 197, 1, 3, 768] : tensor<197x1x2304xf32> into tensor<1x197x1x3x768xf32>
    %transposed_195 = linalg.transpose ins(%expanded_194 : tensor<1x197x1x3x768xf32>) outs(%6 : tensor<3x197x1x1x768xf32>) permutation = [3, 1, 2, 0, 4] 
    %collapsed_196 = tensor.collapse_shape %transposed_195 [[0], [1], [2, 3], [4]] : tensor<3x197x1x1x768xf32> into tensor<3x197x1x768xf32>
    %extracted_slice_197 = tensor.extract_slice %collapsed_196[0, 0, 0, 0] [1, 197, 1, 768] [1, 1, 1, 1] : tensor<3x197x1x768xf32> to tensor<1x197x1x768xf32>
    %extracted_slice_198 = tensor.extract_slice %collapsed_196[1, 0, 0, 0] [1, 197, 1, 768] [1, 1, 1, 1] : tensor<3x197x1x768xf32> to tensor<1x197x1x768xf32>
    %extracted_slice_199 = tensor.extract_slice %collapsed_196[2, 0, 0, 0] [1, 197, 1, 768] [1, 1, 1, 1] : tensor<3x197x1x768xf32> to tensor<1x197x1x768xf32>
    %collapsed_200 = tensor.collapse_shape %extracted_slice_197 [[0, 1], [2, 3]] : tensor<1x197x1x768xf32> into tensor<197x768xf32>
    %expanded_201 = tensor.expand_shape %collapsed_200 [[0], [1, 2]] output_shape [197, 12, 64] : tensor<197x768xf32> into tensor<197x12x64xf32>
    %transposed_202 = linalg.transpose ins(%expanded_201 : tensor<197x12x64xf32>) outs(%7 : tensor<12x197x64xf32>) permutation = [1, 0, 2] 
    %collapsed_203 = tensor.collapse_shape %extracted_slice_198 [[0, 1], [2, 3]] : tensor<1x197x1x768xf32> into tensor<197x768xf32>
    %expanded_204 = tensor.expand_shape %collapsed_203 [[0], [1, 2]] output_shape [197, 12, 64] : tensor<197x768xf32> into tensor<197x12x64xf32>
    %transposed_205 = linalg.transpose ins(%expanded_204 : tensor<197x12x64xf32>) outs(%7 : tensor<12x197x64xf32>) permutation = [1, 0, 2] 
    %collapsed_206 = tensor.collapse_shape %extracted_slice_199 [[0, 1], [2, 3]] : tensor<1x197x1x768xf32> into tensor<197x768xf32>
    %expanded_207 = tensor.expand_shape %collapsed_206 [[0], [1, 2]] output_shape [197, 12, 64] : tensor<197x768xf32> into tensor<197x12x64xf32>
    %transposed_208 = linalg.transpose ins(%expanded_207 : tensor<197x12x64xf32>) outs(%7 : tensor<12x197x64xf32>) permutation = [1, 0, 2] 
    %expanded_209 = tensor.expand_shape %transposed_202 [[0, 1], [2], [3]] output_shape [1, 12, 197, 64] : tensor<12x197x64xf32> into tensor<1x12x197x64xf32>
    %expanded_210 = tensor.expand_shape %transposed_205 [[0, 1], [2], [3]] output_shape [1, 12, 197, 64] : tensor<12x197x64xf32> into tensor<1x12x197x64xf32>
    %expanded_211 = tensor.expand_shape %transposed_208 [[0, 1], [2], [3]] output_shape [1, 12, 197, 64] : tensor<12x197x64xf32> into tensor<1x12x197x64xf32>
    %90 = dnn.scaled_dot_product_attention %expanded_209, %expanded_210, %expanded_211 {parameter_indices = array<i32: 3, 4, 5, 6, 7>, parameters = [unit, 0.000000e+00, false, unit, false]} : (tensor<1x12x197x64xf32>, tensor<1x12x197x64xf32>, tensor<1x12x197x64xf32>) -> tensor<1x12x197x64xf32>
    %transposed_212 = linalg.transpose ins(%90 : tensor<1x12x197x64xf32>) outs(%9 : tensor<197x1x12x64xf32>) permutation = [2, 0, 1, 3] 
    %collapsed_213 = tensor.collapse_shape %transposed_212 [[0], [1, 2, 3]] : tensor<197x1x12x64xf32> into tensor<197x768xf32>
    %91 = dnn.linear %collapsed_213, %cst_4, %cst_0 : tensor<197x768xf32>, tensor<768x768xf32>, tensor<768xf32> -> tensor<197x768xf32>
    %expanded_214 = tensor.expand_shape %91 [[0], [1, 2]] output_shape [197, 1, 768] : tensor<197x768xf32> into tensor<197x1x768xf32>
    %transposed_215 = linalg.transpose ins(%expanded_214 : tensor<197x1x768xf32>) outs(%11 : tensor<1x197x768xf32>) permutation = [1, 0, 2] 
    %92 = dnn.add %transposed_215, %87 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %93 = dnn.layer_norm %92, %cst_0, %cst_0 {parameter_indices = array<i32: 1, 4, 5>, parameters = [[768], 9.9999999999999995E-7, false]} : (tensor<1x197x768xf32>, tensor<768xf32>, tensor<768xf32>) -> tensor<1x197x768xf32>
    %94 = dnn.linear %93, %cst_5, %cst_6 : tensor<1x197x768xf32>, tensor<3072x768xf32>, tensor<3072xf32> -> tensor<1x197x3072xf32>
    %95 = dnn.gelu %94 {parameter_indices = array<i32: 1>, parameters = ["none"]} : (tensor<1x197x3072xf32>) -> tensor<1x197x3072xf32>
    %96 = dnn.linear %95, %cst_7, %cst_0 : tensor<1x197x3072xf32>, tensor<768x3072xf32>, tensor<768xf32> -> tensor<1x197x768xf32>
    %97 = dnn.add %92, %96 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %98 = dnn.layer_norm %97, %cst_0, %cst_0 {parameter_indices = array<i32: 1, 4, 5>, parameters = [[768], 9.9999999999999995E-7, false]} : (tensor<1x197x768xf32>, tensor<768xf32>, tensor<768xf32>) -> tensor<1x197x768xf32>
    %transposed_216 = linalg.transpose ins(%98 : tensor<1x197x768xf32>) outs(%4 : tensor<197x1x768xf32>) permutation = [1, 0, 2] 
    %99 = dnn.linear %transposed_216, %cst_2, %cst_3 : tensor<197x1x768xf32>, tensor<2304x768xf32>, tensor<2304xf32> -> tensor<197x1x2304xf32>
    %expanded_217 = tensor.expand_shape %99 [[0, 1], [2], [3, 4]] output_shape [1, 197, 1, 3, 768] : tensor<197x1x2304xf32> into tensor<1x197x1x3x768xf32>
    %transposed_218 = linalg.transpose ins(%expanded_217 : tensor<1x197x1x3x768xf32>) outs(%6 : tensor<3x197x1x1x768xf32>) permutation = [3, 1, 2, 0, 4] 
    %collapsed_219 = tensor.collapse_shape %transposed_218 [[0], [1], [2, 3], [4]] : tensor<3x197x1x1x768xf32> into tensor<3x197x1x768xf32>
    %extracted_slice_220 = tensor.extract_slice %collapsed_219[0, 0, 0, 0] [1, 197, 1, 768] [1, 1, 1, 1] : tensor<3x197x1x768xf32> to tensor<1x197x1x768xf32>
    %extracted_slice_221 = tensor.extract_slice %collapsed_219[1, 0, 0, 0] [1, 197, 1, 768] [1, 1, 1, 1] : tensor<3x197x1x768xf32> to tensor<1x197x1x768xf32>
    %extracted_slice_222 = tensor.extract_slice %collapsed_219[2, 0, 0, 0] [1, 197, 1, 768] [1, 1, 1, 1] : tensor<3x197x1x768xf32> to tensor<1x197x1x768xf32>
    %collapsed_223 = tensor.collapse_shape %extracted_slice_220 [[0, 1], [2, 3]] : tensor<1x197x1x768xf32> into tensor<197x768xf32>
    %expanded_224 = tensor.expand_shape %collapsed_223 [[0], [1, 2]] output_shape [197, 12, 64] : tensor<197x768xf32> into tensor<197x12x64xf32>
    %transposed_225 = linalg.transpose ins(%expanded_224 : tensor<197x12x64xf32>) outs(%7 : tensor<12x197x64xf32>) permutation = [1, 0, 2] 
    %collapsed_226 = tensor.collapse_shape %extracted_slice_221 [[0, 1], [2, 3]] : tensor<1x197x1x768xf32> into tensor<197x768xf32>
    %expanded_227 = tensor.expand_shape %collapsed_226 [[0], [1, 2]] output_shape [197, 12, 64] : tensor<197x768xf32> into tensor<197x12x64xf32>
    %transposed_228 = linalg.transpose ins(%expanded_227 : tensor<197x12x64xf32>) outs(%7 : tensor<12x197x64xf32>) permutation = [1, 0, 2] 
    %collapsed_229 = tensor.collapse_shape %extracted_slice_222 [[0, 1], [2, 3]] : tensor<1x197x1x768xf32> into tensor<197x768xf32>
    %expanded_230 = tensor.expand_shape %collapsed_229 [[0], [1, 2]] output_shape [197, 12, 64] : tensor<197x768xf32> into tensor<197x12x64xf32>
    %transposed_231 = linalg.transpose ins(%expanded_230 : tensor<197x12x64xf32>) outs(%7 : tensor<12x197x64xf32>) permutation = [1, 0, 2] 
    %expanded_232 = tensor.expand_shape %transposed_225 [[0, 1], [2], [3]] output_shape [1, 12, 197, 64] : tensor<12x197x64xf32> into tensor<1x12x197x64xf32>
    %expanded_233 = tensor.expand_shape %transposed_228 [[0, 1], [2], [3]] output_shape [1, 12, 197, 64] : tensor<12x197x64xf32> into tensor<1x12x197x64xf32>
    %expanded_234 = tensor.expand_shape %transposed_231 [[0, 1], [2], [3]] output_shape [1, 12, 197, 64] : tensor<12x197x64xf32> into tensor<1x12x197x64xf32>
    %100 = dnn.scaled_dot_product_attention %expanded_232, %expanded_233, %expanded_234 {parameter_indices = array<i32: 3, 4, 5, 6, 7>, parameters = [unit, 0.000000e+00, false, unit, false]} : (tensor<1x12x197x64xf32>, tensor<1x12x197x64xf32>, tensor<1x12x197x64xf32>) -> tensor<1x12x197x64xf32>
    %transposed_235 = linalg.transpose ins(%100 : tensor<1x12x197x64xf32>) outs(%9 : tensor<197x1x12x64xf32>) permutation = [2, 0, 1, 3] 
    %collapsed_236 = tensor.collapse_shape %transposed_235 [[0], [1, 2, 3]] : tensor<197x1x12x64xf32> into tensor<197x768xf32>
    %101 = dnn.linear %collapsed_236, %cst_4, %cst_0 : tensor<197x768xf32>, tensor<768x768xf32>, tensor<768xf32> -> tensor<197x768xf32>
    %expanded_237 = tensor.expand_shape %101 [[0], [1, 2]] output_shape [197, 1, 768] : tensor<197x768xf32> into tensor<197x1x768xf32>
    %transposed_238 = linalg.transpose ins(%expanded_237 : tensor<197x1x768xf32>) outs(%11 : tensor<1x197x768xf32>) permutation = [1, 0, 2] 
    %102 = dnn.add %transposed_238, %97 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %103 = dnn.layer_norm %102, %cst_0, %cst_0 {parameter_indices = array<i32: 1, 4, 5>, parameters = [[768], 9.9999999999999995E-7, false]} : (tensor<1x197x768xf32>, tensor<768xf32>, tensor<768xf32>) -> tensor<1x197x768xf32>
    %104 = dnn.linear %103, %cst_5, %cst_6 : tensor<1x197x768xf32>, tensor<3072x768xf32>, tensor<3072xf32> -> tensor<1x197x3072xf32>
    %105 = dnn.gelu %104 {parameter_indices = array<i32: 1>, parameters = ["none"]} : (tensor<1x197x3072xf32>) -> tensor<1x197x3072xf32>
    %106 = dnn.linear %105, %cst_7, %cst_0 : tensor<1x197x3072xf32>, tensor<768x3072xf32>, tensor<768xf32> -> tensor<1x197x768xf32>
    %107 = dnn.add %102, %106 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %108 = dnn.layer_norm %107, %cst_0, %cst_0 {parameter_indices = array<i32: 1, 4, 5>, parameters = [[768], 9.9999999999999995E-7, false]} : (tensor<1x197x768xf32>, tensor<768xf32>, tensor<768xf32>) -> tensor<1x197x768xf32>
    %transposed_239 = linalg.transpose ins(%108 : tensor<1x197x768xf32>) outs(%4 : tensor<197x1x768xf32>) permutation = [1, 0, 2] 
    %109 = dnn.linear %transposed_239, %cst_2, %cst_3 : tensor<197x1x768xf32>, tensor<2304x768xf32>, tensor<2304xf32> -> tensor<197x1x2304xf32>
    %expanded_240 = tensor.expand_shape %109 [[0, 1], [2], [3, 4]] output_shape [1, 197, 1, 3, 768] : tensor<197x1x2304xf32> into tensor<1x197x1x3x768xf32>
    %transposed_241 = linalg.transpose ins(%expanded_240 : tensor<1x197x1x3x768xf32>) outs(%6 : tensor<3x197x1x1x768xf32>) permutation = [3, 1, 2, 0, 4] 
    %collapsed_242 = tensor.collapse_shape %transposed_241 [[0], [1], [2, 3], [4]] : tensor<3x197x1x1x768xf32> into tensor<3x197x1x768xf32>
    %extracted_slice_243 = tensor.extract_slice %collapsed_242[0, 0, 0, 0] [1, 197, 1, 768] [1, 1, 1, 1] : tensor<3x197x1x768xf32> to tensor<1x197x1x768xf32>
    %extracted_slice_244 = tensor.extract_slice %collapsed_242[1, 0, 0, 0] [1, 197, 1, 768] [1, 1, 1, 1] : tensor<3x197x1x768xf32> to tensor<1x197x1x768xf32>
    %extracted_slice_245 = tensor.extract_slice %collapsed_242[2, 0, 0, 0] [1, 197, 1, 768] [1, 1, 1, 1] : tensor<3x197x1x768xf32> to tensor<1x197x1x768xf32>
    %collapsed_246 = tensor.collapse_shape %extracted_slice_243 [[0, 1], [2, 3]] : tensor<1x197x1x768xf32> into tensor<197x768xf32>
    %expanded_247 = tensor.expand_shape %collapsed_246 [[0], [1, 2]] output_shape [197, 12, 64] : tensor<197x768xf32> into tensor<197x12x64xf32>
    %transposed_248 = linalg.transpose ins(%expanded_247 : tensor<197x12x64xf32>) outs(%7 : tensor<12x197x64xf32>) permutation = [1, 0, 2] 
    %collapsed_249 = tensor.collapse_shape %extracted_slice_244 [[0, 1], [2, 3]] : tensor<1x197x1x768xf32> into tensor<197x768xf32>
    %expanded_250 = tensor.expand_shape %collapsed_249 [[0], [1, 2]] output_shape [197, 12, 64] : tensor<197x768xf32> into tensor<197x12x64xf32>
    %transposed_251 = linalg.transpose ins(%expanded_250 : tensor<197x12x64xf32>) outs(%7 : tensor<12x197x64xf32>) permutation = [1, 0, 2] 
    %collapsed_252 = tensor.collapse_shape %extracted_slice_245 [[0, 1], [2, 3]] : tensor<1x197x1x768xf32> into tensor<197x768xf32>
    %expanded_253 = tensor.expand_shape %collapsed_252 [[0], [1, 2]] output_shape [197, 12, 64] : tensor<197x768xf32> into tensor<197x12x64xf32>
    %transposed_254 = linalg.transpose ins(%expanded_253 : tensor<197x12x64xf32>) outs(%7 : tensor<12x197x64xf32>) permutation = [1, 0, 2] 
    %expanded_255 = tensor.expand_shape %transposed_248 [[0, 1], [2], [3]] output_shape [1, 12, 197, 64] : tensor<12x197x64xf32> into tensor<1x12x197x64xf32>
    %expanded_256 = tensor.expand_shape %transposed_251 [[0, 1], [2], [3]] output_shape [1, 12, 197, 64] : tensor<12x197x64xf32> into tensor<1x12x197x64xf32>
    %expanded_257 = tensor.expand_shape %transposed_254 [[0, 1], [2], [3]] output_shape [1, 12, 197, 64] : tensor<12x197x64xf32> into tensor<1x12x197x64xf32>
    %110 = dnn.scaled_dot_product_attention %expanded_255, %expanded_256, %expanded_257 {parameter_indices = array<i32: 3, 4, 5, 6, 7>, parameters = [unit, 0.000000e+00, false, unit, false]} : (tensor<1x12x197x64xf32>, tensor<1x12x197x64xf32>, tensor<1x12x197x64xf32>) -> tensor<1x12x197x64xf32>
    %transposed_258 = linalg.transpose ins(%110 : tensor<1x12x197x64xf32>) outs(%9 : tensor<197x1x12x64xf32>) permutation = [2, 0, 1, 3] 
    %collapsed_259 = tensor.collapse_shape %transposed_258 [[0], [1, 2, 3]] : tensor<197x1x12x64xf32> into tensor<197x768xf32>
    %111 = dnn.linear %collapsed_259, %cst_4, %cst_0 : tensor<197x768xf32>, tensor<768x768xf32>, tensor<768xf32> -> tensor<197x768xf32>
    %expanded_260 = tensor.expand_shape %111 [[0], [1, 2]] output_shape [197, 1, 768] : tensor<197x768xf32> into tensor<197x1x768xf32>
    %transposed_261 = linalg.transpose ins(%expanded_260 : tensor<197x1x768xf32>) outs(%11 : tensor<1x197x768xf32>) permutation = [1, 0, 2] 
    %112 = dnn.add %transposed_261, %107 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %113 = dnn.layer_norm %112, %cst_0, %cst_0 {parameter_indices = array<i32: 1, 4, 5>, parameters = [[768], 9.9999999999999995E-7, false]} : (tensor<1x197x768xf32>, tensor<768xf32>, tensor<768xf32>) -> tensor<1x197x768xf32>
    %114 = dnn.linear %113, %cst_5, %cst_6 : tensor<1x197x768xf32>, tensor<3072x768xf32>, tensor<3072xf32> -> tensor<1x197x3072xf32>
    %115 = dnn.gelu %114 {parameter_indices = array<i32: 1>, parameters = ["none"]} : (tensor<1x197x3072xf32>) -> tensor<1x197x3072xf32>
    %116 = dnn.linear %115, %cst_7, %cst_0 : tensor<1x197x3072xf32>, tensor<768x3072xf32>, tensor<768xf32> -> tensor<1x197x768xf32>
    %117 = dnn.add %112, %116 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %118 = dnn.layer_norm %117, %cst_0, %cst_0 {parameter_indices = array<i32: 1, 4, 5>, parameters = [[768], 9.9999999999999995E-7, false]} : (tensor<1x197x768xf32>, tensor<768xf32>, tensor<768xf32>) -> tensor<1x197x768xf32>
    %transposed_262 = linalg.transpose ins(%118 : tensor<1x197x768xf32>) outs(%4 : tensor<197x1x768xf32>) permutation = [1, 0, 2] 
    %119 = dnn.linear %transposed_262, %cst_2, %cst_3 : tensor<197x1x768xf32>, tensor<2304x768xf32>, tensor<2304xf32> -> tensor<197x1x2304xf32>
    %expanded_263 = tensor.expand_shape %119 [[0, 1], [2], [3, 4]] output_shape [1, 197, 1, 3, 768] : tensor<197x1x2304xf32> into tensor<1x197x1x3x768xf32>
    %transposed_264 = linalg.transpose ins(%expanded_263 : tensor<1x197x1x3x768xf32>) outs(%6 : tensor<3x197x1x1x768xf32>) permutation = [3, 1, 2, 0, 4] 
    %collapsed_265 = tensor.collapse_shape %transposed_264 [[0], [1], [2, 3], [4]] : tensor<3x197x1x1x768xf32> into tensor<3x197x1x768xf32>
    %extracted_slice_266 = tensor.extract_slice %collapsed_265[0, 0, 0, 0] [1, 197, 1, 768] [1, 1, 1, 1] : tensor<3x197x1x768xf32> to tensor<1x197x1x768xf32>
    %extracted_slice_267 = tensor.extract_slice %collapsed_265[1, 0, 0, 0] [1, 197, 1, 768] [1, 1, 1, 1] : tensor<3x197x1x768xf32> to tensor<1x197x1x768xf32>
    %extracted_slice_268 = tensor.extract_slice %collapsed_265[2, 0, 0, 0] [1, 197, 1, 768] [1, 1, 1, 1] : tensor<3x197x1x768xf32> to tensor<1x197x1x768xf32>
    %collapsed_269 = tensor.collapse_shape %extracted_slice_266 [[0, 1], [2, 3]] : tensor<1x197x1x768xf32> into tensor<197x768xf32>
    %expanded_270 = tensor.expand_shape %collapsed_269 [[0], [1, 2]] output_shape [197, 12, 64] : tensor<197x768xf32> into tensor<197x12x64xf32>
    %transposed_271 = linalg.transpose ins(%expanded_270 : tensor<197x12x64xf32>) outs(%7 : tensor<12x197x64xf32>) permutation = [1, 0, 2] 
    %collapsed_272 = tensor.collapse_shape %extracted_slice_267 [[0, 1], [2, 3]] : tensor<1x197x1x768xf32> into tensor<197x768xf32>
    %expanded_273 = tensor.expand_shape %collapsed_272 [[0], [1, 2]] output_shape [197, 12, 64] : tensor<197x768xf32> into tensor<197x12x64xf32>
    %transposed_274 = linalg.transpose ins(%expanded_273 : tensor<197x12x64xf32>) outs(%7 : tensor<12x197x64xf32>) permutation = [1, 0, 2] 
    %collapsed_275 = tensor.collapse_shape %extracted_slice_268 [[0, 1], [2, 3]] : tensor<1x197x1x768xf32> into tensor<197x768xf32>
    %expanded_276 = tensor.expand_shape %collapsed_275 [[0], [1, 2]] output_shape [197, 12, 64] : tensor<197x768xf32> into tensor<197x12x64xf32>
    %transposed_277 = linalg.transpose ins(%expanded_276 : tensor<197x12x64xf32>) outs(%7 : tensor<12x197x64xf32>) permutation = [1, 0, 2] 
    %expanded_278 = tensor.expand_shape %transposed_271 [[0, 1], [2], [3]] output_shape [1, 12, 197, 64] : tensor<12x197x64xf32> into tensor<1x12x197x64xf32>
    %expanded_279 = tensor.expand_shape %transposed_274 [[0, 1], [2], [3]] output_shape [1, 12, 197, 64] : tensor<12x197x64xf32> into tensor<1x12x197x64xf32>
    %expanded_280 = tensor.expand_shape %transposed_277 [[0, 1], [2], [3]] output_shape [1, 12, 197, 64] : tensor<12x197x64xf32> into tensor<1x12x197x64xf32>
    %120 = dnn.scaled_dot_product_attention %expanded_278, %expanded_279, %expanded_280 {parameter_indices = array<i32: 3, 4, 5, 6, 7>, parameters = [unit, 0.000000e+00, false, unit, false]} : (tensor<1x12x197x64xf32>, tensor<1x12x197x64xf32>, tensor<1x12x197x64xf32>) -> tensor<1x12x197x64xf32>
    %transposed_281 = linalg.transpose ins(%120 : tensor<1x12x197x64xf32>) outs(%9 : tensor<197x1x12x64xf32>) permutation = [2, 0, 1, 3] 
    %collapsed_282 = tensor.collapse_shape %transposed_281 [[0], [1, 2, 3]] : tensor<197x1x12x64xf32> into tensor<197x768xf32>
    %121 = dnn.linear %collapsed_282, %cst_4, %cst_0 : tensor<197x768xf32>, tensor<768x768xf32>, tensor<768xf32> -> tensor<197x768xf32>
    %expanded_283 = tensor.expand_shape %121 [[0], [1, 2]] output_shape [197, 1, 768] : tensor<197x768xf32> into tensor<197x1x768xf32>
    %transposed_284 = linalg.transpose ins(%expanded_283 : tensor<197x1x768xf32>) outs(%11 : tensor<1x197x768xf32>) permutation = [1, 0, 2] 
    %122 = dnn.add %transposed_284, %117 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %123 = dnn.layer_norm %122, %cst_0, %cst_0 {parameter_indices = array<i32: 1, 4, 5>, parameters = [[768], 9.9999999999999995E-7, false]} : (tensor<1x197x768xf32>, tensor<768xf32>, tensor<768xf32>) -> tensor<1x197x768xf32>
    %124 = dnn.linear %123, %cst_5, %cst_6 : tensor<1x197x768xf32>, tensor<3072x768xf32>, tensor<3072xf32> -> tensor<1x197x3072xf32>
    %125 = dnn.gelu %124 {parameter_indices = array<i32: 1>, parameters = ["none"]} : (tensor<1x197x3072xf32>) -> tensor<1x197x3072xf32>
    %126 = dnn.linear %125, %cst_7, %cst_0 : tensor<1x197x3072xf32>, tensor<768x3072xf32>, tensor<768xf32> -> tensor<1x197x768xf32>
    %127 = dnn.add %122, %126 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
    %128 = dnn.layer_norm %127, %cst_0, %cst_0 {parameter_indices = array<i32: 1, 4, 5>, parameters = [[768], 9.9999999999999995E-7, false]} : (tensor<1x197x768xf32>, tensor<768xf32>, tensor<768xf32>) -> tensor<1x197x768xf32>
    %extracted_slice_285 = tensor.extract_slice %128[0, 0, 0] [1, 1, 768] [1, 1, 1] : tensor<1x197x768xf32> to tensor<1x1x768xf32>
    %collapsed_286 = tensor.collapse_shape %extracted_slice_285 [[0, 1], [2]] : tensor<1x1x768xf32> into tensor<1x768xf32>
    %129 = dnn.linear %collapsed_286, %cst_8, %cst_9 : tensor<1x768xf32>, tensor<1000x768xf32>, tensor<1000xf32> -> tensor<1x1000xf32>
    return %129 : tensor<1x1000xf32>
  }
}

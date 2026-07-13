// RUN: dnn-mlir-opt %s -o /dev/null
//
// ResNet18 after captures=all and the remaining Torch-to-Linalg backend
// lowering. Supported operations remain high-level DNN operations.

module {
  func.func @resnet18(%arg0: tensor<1x3x224x224xf32>) -> tensor<1x1000xf32> {
    %cst = arith.constant dense<0.000000e+00> : tensor<64x3x7x7xf32>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<64xf32>
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<64x64x3x3xf32>
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<128x64x3x3xf32>
    %cst_3 = arith.constant dense<0.000000e+00> : tensor<128xf32>
    %cst_4 = arith.constant dense<0.000000e+00> : tensor<128x128x3x3xf32>
    %cst_5 = arith.constant dense<0.000000e+00> : tensor<128x64x1x1xf32>
    %cst_6 = arith.constant dense<0.000000e+00> : tensor<256x128x3x3xf32>
    %cst_7 = arith.constant dense<0.000000e+00> : tensor<256xf32>
    %cst_8 = arith.constant dense<0.000000e+00> : tensor<256x256x3x3xf32>
    %cst_9 = arith.constant dense<0.000000e+00> : tensor<256x128x1x1xf32>
    %cst_10 = arith.constant dense<0.000000e+00> : tensor<512x256x3x3xf32>
    %cst_11 = arith.constant dense<0.000000e+00> : tensor<512xf32>
    %cst_12 = arith.constant dense<0.000000e+00> : tensor<512x512x3x3xf32>
    %cst_13 = arith.constant dense<0.000000e+00> : tensor<512x256x1x1xf32>
    %cst_14 = arith.constant dense<0.000000e+00> : tensor<1000x512xf32>
    %cst_15 = arith.constant dense<0.000000e+00> : tensor<1000xf32>
    %0 = dnn.convolution %arg0, %cst {kind = "aten.conv2d", parameter_indices = array<i32: 2, 3, 4, 5, 6>, parameters = [unit, [2, 2], [3, 3], [1, 1], 1]} : (tensor<1x3x224x224xf32>, tensor<64x3x7x7xf32>) -> tensor<1x64x112x112xf32>
    %1:3 = dnn.batch_norm %0, %cst_0, %cst_0, %cst_0, %cst_0 {kind = "aten._native_batch_norm_legit_no_training", parameter_indices = array<i32: 5, 6>, parameters = [1.000000e-01, 1.000000e-05]} : (tensor<1x64x112x112xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> (tensor<1x64x112x112xf32>, tensor<0xf32>, tensor<0xf32>)
    %2 = dnn.relu %1#0 : (tensor<1x64x112x112xf32>) -> tensor<1x64x112x112xf32>
    %3 = dnn.max_pool2d %2 {parameter_indices = array<i32: 1, 2, 3, 4, 5>, parameters = [[3, 3], [2, 2], [1, 1], [1, 1], false]} : (tensor<1x64x112x112xf32>) -> tensor<1x64x56x56xf32>
    %4 = dnn.convolution %3, %cst_1 {kind = "aten.conv2d", parameter_indices = array<i32: 2, 3, 4, 5, 6>, parameters = [unit, [1, 1], [1, 1], [1, 1], 1]} : (tensor<1x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<1x64x56x56xf32>
    %5:3 = dnn.batch_norm %4, %cst_0, %cst_0, %cst_0, %cst_0 {kind = "aten._native_batch_norm_legit_no_training", parameter_indices = array<i32: 5, 6>, parameters = [1.000000e-01, 1.000000e-05]} : (tensor<1x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> (tensor<1x64x56x56xf32>, tensor<0xf32>, tensor<0xf32>)
    %6 = dnn.relu %5#0 : (tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
    %7 = dnn.convolution %6, %cst_1 {kind = "aten.conv2d", parameter_indices = array<i32: 2, 3, 4, 5, 6>, parameters = [unit, [1, 1], [1, 1], [1, 1], 1]} : (tensor<1x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<1x64x56x56xf32>
    %8:3 = dnn.batch_norm %7, %cst_0, %cst_0, %cst_0, %cst_0 {kind = "aten._native_batch_norm_legit_no_training", parameter_indices = array<i32: 5, 6>, parameters = [1.000000e-01, 1.000000e-05]} : (tensor<1x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> (tensor<1x64x56x56xf32>, tensor<0xf32>, tensor<0xf32>)
    %9 = dnn.add %8#0, %3 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<1x64x56x56xf32>, tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
    %10 = dnn.relu %9 : (tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
    %11 = dnn.convolution %10, %cst_1 {kind = "aten.conv2d", parameter_indices = array<i32: 2, 3, 4, 5, 6>, parameters = [unit, [1, 1], [1, 1], [1, 1], 1]} : (tensor<1x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<1x64x56x56xf32>
    %12:3 = dnn.batch_norm %11, %cst_0, %cst_0, %cst_0, %cst_0 {kind = "aten._native_batch_norm_legit_no_training", parameter_indices = array<i32: 5, 6>, parameters = [1.000000e-01, 1.000000e-05]} : (tensor<1x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> (tensor<1x64x56x56xf32>, tensor<0xf32>, tensor<0xf32>)
    %13 = dnn.relu %12#0 : (tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
    %14 = dnn.convolution %13, %cst_1 {kind = "aten.conv2d", parameter_indices = array<i32: 2, 3, 4, 5, 6>, parameters = [unit, [1, 1], [1, 1], [1, 1], 1]} : (tensor<1x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<1x64x56x56xf32>
    %15:3 = dnn.batch_norm %14, %cst_0, %cst_0, %cst_0, %cst_0 {kind = "aten._native_batch_norm_legit_no_training", parameter_indices = array<i32: 5, 6>, parameters = [1.000000e-01, 1.000000e-05]} : (tensor<1x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> (tensor<1x64x56x56xf32>, tensor<0xf32>, tensor<0xf32>)
    %16 = dnn.add %15#0, %10 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<1x64x56x56xf32>, tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
    %17 = dnn.relu %16 : (tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
    %18 = dnn.convolution %17, %cst_2 {kind = "aten.conv2d", parameter_indices = array<i32: 2, 3, 4, 5, 6>, parameters = [unit, [2, 2], [1, 1], [1, 1], 1]} : (tensor<1x64x56x56xf32>, tensor<128x64x3x3xf32>) -> tensor<1x128x28x28xf32>
    %19:3 = dnn.batch_norm %18, %cst_3, %cst_3, %cst_3, %cst_3 {kind = "aten._native_batch_norm_legit_no_training", parameter_indices = array<i32: 5, 6>, parameters = [1.000000e-01, 1.000000e-05]} : (tensor<1x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) -> (tensor<1x128x28x28xf32>, tensor<0xf32>, tensor<0xf32>)
    %20 = dnn.relu %19#0 : (tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %21 = dnn.convolution %20, %cst_4 {kind = "aten.conv2d", parameter_indices = array<i32: 2, 3, 4, 5, 6>, parameters = [unit, [1, 1], [1, 1], [1, 1], 1]} : (tensor<1x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<1x128x28x28xf32>
    %22:3 = dnn.batch_norm %21, %cst_3, %cst_3, %cst_3, %cst_3 {kind = "aten._native_batch_norm_legit_no_training", parameter_indices = array<i32: 5, 6>, parameters = [1.000000e-01, 1.000000e-05]} : (tensor<1x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) -> (tensor<1x128x28x28xf32>, tensor<0xf32>, tensor<0xf32>)
    %23 = dnn.convolution %17, %cst_5 {kind = "aten.conv2d", parameter_indices = array<i32: 2, 3, 4, 5, 6>, parameters = [unit, [2, 2], [0, 0], [1, 1], 1]} : (tensor<1x64x56x56xf32>, tensor<128x64x1x1xf32>) -> tensor<1x128x28x28xf32>
    %24:3 = dnn.batch_norm %23, %cst_3, %cst_3, %cst_3, %cst_3 {kind = "aten._native_batch_norm_legit_no_training", parameter_indices = array<i32: 5, 6>, parameters = [1.000000e-01, 1.000000e-05]} : (tensor<1x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) -> (tensor<1x128x28x28xf32>, tensor<0xf32>, tensor<0xf32>)
    %25 = dnn.add %22#0, %24#0 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<1x128x28x28xf32>, tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %26 = dnn.relu %25 : (tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %27 = dnn.convolution %26, %cst_4 {kind = "aten.conv2d", parameter_indices = array<i32: 2, 3, 4, 5, 6>, parameters = [unit, [1, 1], [1, 1], [1, 1], 1]} : (tensor<1x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<1x128x28x28xf32>
    %28:3 = dnn.batch_norm %27, %cst_3, %cst_3, %cst_3, %cst_3 {kind = "aten._native_batch_norm_legit_no_training", parameter_indices = array<i32: 5, 6>, parameters = [1.000000e-01, 1.000000e-05]} : (tensor<1x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) -> (tensor<1x128x28x28xf32>, tensor<0xf32>, tensor<0xf32>)
    %29 = dnn.relu %28#0 : (tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %30 = dnn.convolution %29, %cst_4 {kind = "aten.conv2d", parameter_indices = array<i32: 2, 3, 4, 5, 6>, parameters = [unit, [1, 1], [1, 1], [1, 1], 1]} : (tensor<1x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<1x128x28x28xf32>
    %31:3 = dnn.batch_norm %30, %cst_3, %cst_3, %cst_3, %cst_3 {kind = "aten._native_batch_norm_legit_no_training", parameter_indices = array<i32: 5, 6>, parameters = [1.000000e-01, 1.000000e-05]} : (tensor<1x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) -> (tensor<1x128x28x28xf32>, tensor<0xf32>, tensor<0xf32>)
    %32 = dnn.add %31#0, %26 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<1x128x28x28xf32>, tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %33 = dnn.relu %32 : (tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %34 = dnn.convolution %33, %cst_6 {kind = "aten.conv2d", parameter_indices = array<i32: 2, 3, 4, 5, 6>, parameters = [unit, [2, 2], [1, 1], [1, 1], 1]} : (tensor<1x128x28x28xf32>, tensor<256x128x3x3xf32>) -> tensor<1x256x14x14xf32>
    %35:3 = dnn.batch_norm %34, %cst_7, %cst_7, %cst_7, %cst_7 {kind = "aten._native_batch_norm_legit_no_training", parameter_indices = array<i32: 5, 6>, parameters = [1.000000e-01, 1.000000e-05]} : (tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> (tensor<1x256x14x14xf32>, tensor<0xf32>, tensor<0xf32>)
    %36 = dnn.relu %35#0 : (tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %37 = dnn.convolution %36, %cst_8 {kind = "aten.conv2d", parameter_indices = array<i32: 2, 3, 4, 5, 6>, parameters = [unit, [1, 1], [1, 1], [1, 1], 1]} : (tensor<1x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<1x256x14x14xf32>
    %38:3 = dnn.batch_norm %37, %cst_7, %cst_7, %cst_7, %cst_7 {kind = "aten._native_batch_norm_legit_no_training", parameter_indices = array<i32: 5, 6>, parameters = [1.000000e-01, 1.000000e-05]} : (tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> (tensor<1x256x14x14xf32>, tensor<0xf32>, tensor<0xf32>)
    %39 = dnn.convolution %33, %cst_9 {kind = "aten.conv2d", parameter_indices = array<i32: 2, 3, 4, 5, 6>, parameters = [unit, [2, 2], [0, 0], [1, 1], 1]} : (tensor<1x128x28x28xf32>, tensor<256x128x1x1xf32>) -> tensor<1x256x14x14xf32>
    %40:3 = dnn.batch_norm %39, %cst_7, %cst_7, %cst_7, %cst_7 {kind = "aten._native_batch_norm_legit_no_training", parameter_indices = array<i32: 5, 6>, parameters = [1.000000e-01, 1.000000e-05]} : (tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> (tensor<1x256x14x14xf32>, tensor<0xf32>, tensor<0xf32>)
    %41 = dnn.add %38#0, %40#0 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<1x256x14x14xf32>, tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %42 = dnn.relu %41 : (tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %43 = dnn.convolution %42, %cst_8 {kind = "aten.conv2d", parameter_indices = array<i32: 2, 3, 4, 5, 6>, parameters = [unit, [1, 1], [1, 1], [1, 1], 1]} : (tensor<1x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<1x256x14x14xf32>
    %44:3 = dnn.batch_norm %43, %cst_7, %cst_7, %cst_7, %cst_7 {kind = "aten._native_batch_norm_legit_no_training", parameter_indices = array<i32: 5, 6>, parameters = [1.000000e-01, 1.000000e-05]} : (tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> (tensor<1x256x14x14xf32>, tensor<0xf32>, tensor<0xf32>)
    %45 = dnn.relu %44#0 : (tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %46 = dnn.convolution %45, %cst_8 {kind = "aten.conv2d", parameter_indices = array<i32: 2, 3, 4, 5, 6>, parameters = [unit, [1, 1], [1, 1], [1, 1], 1]} : (tensor<1x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<1x256x14x14xf32>
    %47:3 = dnn.batch_norm %46, %cst_7, %cst_7, %cst_7, %cst_7 {kind = "aten._native_batch_norm_legit_no_training", parameter_indices = array<i32: 5, 6>, parameters = [1.000000e-01, 1.000000e-05]} : (tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> (tensor<1x256x14x14xf32>, tensor<0xf32>, tensor<0xf32>)
    %48 = dnn.add %47#0, %42 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<1x256x14x14xf32>, tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %49 = dnn.relu %48 : (tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %50 = dnn.convolution %49, %cst_10 {kind = "aten.conv2d", parameter_indices = array<i32: 2, 3, 4, 5, 6>, parameters = [unit, [2, 2], [1, 1], [1, 1], 1]} : (tensor<1x256x14x14xf32>, tensor<512x256x3x3xf32>) -> tensor<1x512x7x7xf32>
    %51:3 = dnn.batch_norm %50, %cst_11, %cst_11, %cst_11, %cst_11 {kind = "aten._native_batch_norm_legit_no_training", parameter_indices = array<i32: 5, 6>, parameters = [1.000000e-01, 1.000000e-05]} : (tensor<1x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) -> (tensor<1x512x7x7xf32>, tensor<0xf32>, tensor<0xf32>)
    %52 = dnn.relu %51#0 : (tensor<1x512x7x7xf32>) -> tensor<1x512x7x7xf32>
    %53 = dnn.convolution %52, %cst_12 {kind = "aten.conv2d", parameter_indices = array<i32: 2, 3, 4, 5, 6>, parameters = [unit, [1, 1], [1, 1], [1, 1], 1]} : (tensor<1x512x7x7xf32>, tensor<512x512x3x3xf32>) -> tensor<1x512x7x7xf32>
    %54:3 = dnn.batch_norm %53, %cst_11, %cst_11, %cst_11, %cst_11 {kind = "aten._native_batch_norm_legit_no_training", parameter_indices = array<i32: 5, 6>, parameters = [1.000000e-01, 1.000000e-05]} : (tensor<1x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) -> (tensor<1x512x7x7xf32>, tensor<0xf32>, tensor<0xf32>)
    %55 = dnn.convolution %49, %cst_13 {kind = "aten.conv2d", parameter_indices = array<i32: 2, 3, 4, 5, 6>, parameters = [unit, [2, 2], [0, 0], [1, 1], 1]} : (tensor<1x256x14x14xf32>, tensor<512x256x1x1xf32>) -> tensor<1x512x7x7xf32>
    %56:3 = dnn.batch_norm %55, %cst_11, %cst_11, %cst_11, %cst_11 {kind = "aten._native_batch_norm_legit_no_training", parameter_indices = array<i32: 5, 6>, parameters = [1.000000e-01, 1.000000e-05]} : (tensor<1x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) -> (tensor<1x512x7x7xf32>, tensor<0xf32>, tensor<0xf32>)
    %57 = dnn.add %54#0, %56#0 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<1x512x7x7xf32>, tensor<1x512x7x7xf32>) -> tensor<1x512x7x7xf32>
    %58 = dnn.relu %57 : (tensor<1x512x7x7xf32>) -> tensor<1x512x7x7xf32>
    %59 = dnn.convolution %58, %cst_12 {kind = "aten.conv2d", parameter_indices = array<i32: 2, 3, 4, 5, 6>, parameters = [unit, [1, 1], [1, 1], [1, 1], 1]} : (tensor<1x512x7x7xf32>, tensor<512x512x3x3xf32>) -> tensor<1x512x7x7xf32>
    %60:3 = dnn.batch_norm %59, %cst_11, %cst_11, %cst_11, %cst_11 {kind = "aten._native_batch_norm_legit_no_training", parameter_indices = array<i32: 5, 6>, parameters = [1.000000e-01, 1.000000e-05]} : (tensor<1x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) -> (tensor<1x512x7x7xf32>, tensor<0xf32>, tensor<0xf32>)
    %61 = dnn.relu %60#0 : (tensor<1x512x7x7xf32>) -> tensor<1x512x7x7xf32>
    %62 = dnn.convolution %61, %cst_12 {kind = "aten.conv2d", parameter_indices = array<i32: 2, 3, 4, 5, 6>, parameters = [unit, [1, 1], [1, 1], [1, 1], 1]} : (tensor<1x512x7x7xf32>, tensor<512x512x3x3xf32>) -> tensor<1x512x7x7xf32>
    %63:3 = dnn.batch_norm %62, %cst_11, %cst_11, %cst_11, %cst_11 {kind = "aten._native_batch_norm_legit_no_training", parameter_indices = array<i32: 5, 6>, parameters = [1.000000e-01, 1.000000e-05]} : (tensor<1x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) -> (tensor<1x512x7x7xf32>, tensor<0xf32>, tensor<0xf32>)
    %64 = dnn.add %63#0, %58 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<1x512x7x7xf32>, tensor<1x512x7x7xf32>) -> tensor<1x512x7x7xf32>
    %65 = dnn.relu %64 : (tensor<1x512x7x7xf32>) -> tensor<1x512x7x7xf32>
    %66 = dnn.adaptive_avg_pool2d %65 {parameter_indices = array<i32: 1>, parameters = [[1, 1]]} : (tensor<1x512x7x7xf32>) -> tensor<1x512x1x1xf32>
    %collapsed = tensor.collapse_shape %66 [[0], [1, 2, 3]] : tensor<1x512x1x1xf32> into tensor<1x512xf32>
    %67 = dnn.linear %collapsed, %cst_14, %cst_15 : tensor<1x512xf32>, tensor<1000x512xf32>, tensor<1000xf32> -> tensor<1x1000xf32>
    return %67 : tensor<1x1000xf32>
  }
}

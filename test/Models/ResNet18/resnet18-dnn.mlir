// RUN: dnn-mlir-opt %s -o /dev/null
//
// Structural ResNet18 inference graph imported through Torch-MLIR's FX path
// with batch-normalization decomposition disabled, then converted with the
// Torch-to-DNN pass. Parameter tensors use deterministic splat values.

module {
  func.func @resnet18(%arg0: !torch.vtensor<[1,3,224,224],f32>) -> !torch.vtensor<[1,1000],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %cst = arith.constant dense<0.000000e+00> : tensor<1000xf32>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<1000x512xf32>
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<512x256x1x1xf32>
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<512x512x3x3xf32>
    %cst_3 = arith.constant dense<0.000000e+00> : tensor<512xf32>
    %cst_4 = arith.constant dense<0.000000e+00> : tensor<512x256x3x3xf32>
    %cst_5 = arith.constant dense<0.000000e+00> : tensor<256x128x1x1xf32>
    %cst_6 = arith.constant dense<0.000000e+00> : tensor<256x256x3x3xf32>
    %cst_7 = arith.constant dense<0.000000e+00> : tensor<256xf32>
    %cst_8 = arith.constant dense<0.000000e+00> : tensor<256x128x3x3xf32>
    %cst_9 = arith.constant dense<0.000000e+00> : tensor<128x64x1x1xf32>
    %cst_10 = arith.constant dense<0.000000e+00> : tensor<128x128x3x3xf32>
    %cst_11 = arith.constant dense<0.000000e+00> : tensor<128xf32>
    %cst_12 = arith.constant dense<0.000000e+00> : tensor<128x64x3x3xf32>
    %cst_13 = arith.constant dense<0.000000e+00> : tensor<64x64x3x3xf32>
    %cst_14 = arith.constant dense<0.000000e+00> : tensor<64xf32>
    %cst_15 = arith.constant dense<0.000000e+00> : tensor<64x3x7x7xf32>
    %int512 = torch.constant.int 512
    %int1 = torch.constant.int 1
    %0 = torch_c.to_builtin_tensor %arg0 : !torch.vtensor<[1,3,224,224],f32> -> tensor<1x3x224x224xf32>
    %1 = dnn.convolution %0, %cst_15 {kind = "aten.conv2d", parameter_indices = array<i32: 2, 3, 4, 5, 6>, parameters = [unit, [2, 2], [3, 3], [1, 1], 1]} : (tensor<1x3x224x224xf32>, tensor<64x3x7x7xf32>) -> tensor<1x64x112x112xf32>
    %2 = torch_c.from_builtin_tensor %1 : tensor<1x64x112x112xf32> -> !torch.vtensor<[1,64,112,112],f32>
    %3 = torch_c.to_builtin_tensor %2 : !torch.vtensor<[1,64,112,112],f32> -> tensor<1x64x112x112xf32>
    %4:3 = dnn.batch_norm %3, %cst_14, %cst_14, %cst_14, %cst_14 {kind = "aten._native_batch_norm_legit_no_training", parameter_indices = array<i32: 5, 6>, parameters = [1.000000e-01, 1.000000e-05]} : (tensor<1x64x112x112xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> (tensor<1x64x112x112xf32>, tensor<0xf32>, tensor<0xf32>)
    %5 = torch_c.from_builtin_tensor %4#0 : tensor<1x64x112x112xf32> -> !torch.vtensor<[1,64,112,112],f32>
    %6 = torch_c.to_builtin_tensor %5 : !torch.vtensor<[1,64,112,112],f32> -> tensor<1x64x112x112xf32>
    %7 = dnn.relu %6 : (tensor<1x64x112x112xf32>) -> tensor<1x64x112x112xf32>
    %8 = torch_c.from_builtin_tensor %7 : tensor<1x64x112x112xf32> -> !torch.vtensor<[1,64,112,112],f32>
    %9 = torch_c.to_builtin_tensor %8 : !torch.vtensor<[1,64,112,112],f32> -> tensor<1x64x112x112xf32>
    %10 = dnn.max_pool2d %9 {parameter_indices = array<i32: 1, 2, 3, 4, 5>, parameters = [[3, 3], [2, 2], [1, 1], [1, 1], false]} : (tensor<1x64x112x112xf32>) -> tensor<1x64x56x56xf32>
    %11 = torch_c.from_builtin_tensor %10 : tensor<1x64x56x56xf32> -> !torch.vtensor<[1,64,56,56],f32>
    %12 = torch_c.to_builtin_tensor %11 : !torch.vtensor<[1,64,56,56],f32> -> tensor<1x64x56x56xf32>
    %13 = dnn.convolution %12, %cst_13 {kind = "aten.conv2d", parameter_indices = array<i32: 2, 3, 4, 5, 6>, parameters = [unit, [1, 1], [1, 1], [1, 1], 1]} : (tensor<1x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<1x64x56x56xf32>
    %14 = torch_c.from_builtin_tensor %13 : tensor<1x64x56x56xf32> -> !torch.vtensor<[1,64,56,56],f32>
    %15 = torch_c.to_builtin_tensor %14 : !torch.vtensor<[1,64,56,56],f32> -> tensor<1x64x56x56xf32>
    %16:3 = dnn.batch_norm %15, %cst_14, %cst_14, %cst_14, %cst_14 {kind = "aten._native_batch_norm_legit_no_training", parameter_indices = array<i32: 5, 6>, parameters = [1.000000e-01, 1.000000e-05]} : (tensor<1x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> (tensor<1x64x56x56xf32>, tensor<0xf32>, tensor<0xf32>)
    %17 = torch_c.from_builtin_tensor %16#0 : tensor<1x64x56x56xf32> -> !torch.vtensor<[1,64,56,56],f32>
    %18 = torch_c.to_builtin_tensor %17 : !torch.vtensor<[1,64,56,56],f32> -> tensor<1x64x56x56xf32>
    %19 = dnn.relu %18 : (tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
    %20 = torch_c.from_builtin_tensor %19 : tensor<1x64x56x56xf32> -> !torch.vtensor<[1,64,56,56],f32>
    %21 = torch_c.to_builtin_tensor %20 : !torch.vtensor<[1,64,56,56],f32> -> tensor<1x64x56x56xf32>
    %22 = dnn.convolution %21, %cst_13 {kind = "aten.conv2d", parameter_indices = array<i32: 2, 3, 4, 5, 6>, parameters = [unit, [1, 1], [1, 1], [1, 1], 1]} : (tensor<1x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<1x64x56x56xf32>
    %23 = torch_c.from_builtin_tensor %22 : tensor<1x64x56x56xf32> -> !torch.vtensor<[1,64,56,56],f32>
    %24 = torch_c.to_builtin_tensor %23 : !torch.vtensor<[1,64,56,56],f32> -> tensor<1x64x56x56xf32>
    %25:3 = dnn.batch_norm %24, %cst_14, %cst_14, %cst_14, %cst_14 {kind = "aten._native_batch_norm_legit_no_training", parameter_indices = array<i32: 5, 6>, parameters = [1.000000e-01, 1.000000e-05]} : (tensor<1x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> (tensor<1x64x56x56xf32>, tensor<0xf32>, tensor<0xf32>)
    %26 = torch_c.from_builtin_tensor %25#0 : tensor<1x64x56x56xf32> -> !torch.vtensor<[1,64,56,56],f32>
    %27 = torch_c.to_builtin_tensor %26 : !torch.vtensor<[1,64,56,56],f32> -> tensor<1x64x56x56xf32>
    %28 = torch_c.to_builtin_tensor %11 : !torch.vtensor<[1,64,56,56],f32> -> tensor<1x64x56x56xf32>
    %29 = dnn.add %27, %28 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<1x64x56x56xf32>, tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
    %30 = torch_c.from_builtin_tensor %29 : tensor<1x64x56x56xf32> -> !torch.vtensor<[1,64,56,56],f32>
    %31 = torch_c.to_builtin_tensor %30 : !torch.vtensor<[1,64,56,56],f32> -> tensor<1x64x56x56xf32>
    %32 = dnn.relu %31 : (tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
    %33 = torch_c.from_builtin_tensor %32 : tensor<1x64x56x56xf32> -> !torch.vtensor<[1,64,56,56],f32>
    %34 = torch_c.to_builtin_tensor %33 : !torch.vtensor<[1,64,56,56],f32> -> tensor<1x64x56x56xf32>
    %35 = dnn.convolution %34, %cst_13 {kind = "aten.conv2d", parameter_indices = array<i32: 2, 3, 4, 5, 6>, parameters = [unit, [1, 1], [1, 1], [1, 1], 1]} : (tensor<1x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<1x64x56x56xf32>
    %36 = torch_c.from_builtin_tensor %35 : tensor<1x64x56x56xf32> -> !torch.vtensor<[1,64,56,56],f32>
    %37 = torch_c.to_builtin_tensor %36 : !torch.vtensor<[1,64,56,56],f32> -> tensor<1x64x56x56xf32>
    %38:3 = dnn.batch_norm %37, %cst_14, %cst_14, %cst_14, %cst_14 {kind = "aten._native_batch_norm_legit_no_training", parameter_indices = array<i32: 5, 6>, parameters = [1.000000e-01, 1.000000e-05]} : (tensor<1x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> (tensor<1x64x56x56xf32>, tensor<0xf32>, tensor<0xf32>)
    %39 = torch_c.from_builtin_tensor %38#0 : tensor<1x64x56x56xf32> -> !torch.vtensor<[1,64,56,56],f32>
    %40 = torch_c.to_builtin_tensor %39 : !torch.vtensor<[1,64,56,56],f32> -> tensor<1x64x56x56xf32>
    %41 = dnn.relu %40 : (tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
    %42 = torch_c.from_builtin_tensor %41 : tensor<1x64x56x56xf32> -> !torch.vtensor<[1,64,56,56],f32>
    %43 = torch_c.to_builtin_tensor %42 : !torch.vtensor<[1,64,56,56],f32> -> tensor<1x64x56x56xf32>
    %44 = dnn.convolution %43, %cst_13 {kind = "aten.conv2d", parameter_indices = array<i32: 2, 3, 4, 5, 6>, parameters = [unit, [1, 1], [1, 1], [1, 1], 1]} : (tensor<1x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<1x64x56x56xf32>
    %45 = torch_c.from_builtin_tensor %44 : tensor<1x64x56x56xf32> -> !torch.vtensor<[1,64,56,56],f32>
    %46 = torch_c.to_builtin_tensor %45 : !torch.vtensor<[1,64,56,56],f32> -> tensor<1x64x56x56xf32>
    %47:3 = dnn.batch_norm %46, %cst_14, %cst_14, %cst_14, %cst_14 {kind = "aten._native_batch_norm_legit_no_training", parameter_indices = array<i32: 5, 6>, parameters = [1.000000e-01, 1.000000e-05]} : (tensor<1x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> (tensor<1x64x56x56xf32>, tensor<0xf32>, tensor<0xf32>)
    %48 = torch_c.from_builtin_tensor %47#0 : tensor<1x64x56x56xf32> -> !torch.vtensor<[1,64,56,56],f32>
    %49 = torch_c.to_builtin_tensor %48 : !torch.vtensor<[1,64,56,56],f32> -> tensor<1x64x56x56xf32>
    %50 = torch_c.to_builtin_tensor %33 : !torch.vtensor<[1,64,56,56],f32> -> tensor<1x64x56x56xf32>
    %51 = dnn.add %49, %50 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<1x64x56x56xf32>, tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
    %52 = torch_c.from_builtin_tensor %51 : tensor<1x64x56x56xf32> -> !torch.vtensor<[1,64,56,56],f32>
    %53 = torch_c.to_builtin_tensor %52 : !torch.vtensor<[1,64,56,56],f32> -> tensor<1x64x56x56xf32>
    %54 = dnn.relu %53 : (tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
    %55 = torch_c.from_builtin_tensor %54 : tensor<1x64x56x56xf32> -> !torch.vtensor<[1,64,56,56],f32>
    %56 = torch_c.to_builtin_tensor %55 : !torch.vtensor<[1,64,56,56],f32> -> tensor<1x64x56x56xf32>
    %57 = dnn.convolution %56, %cst_12 {kind = "aten.conv2d", parameter_indices = array<i32: 2, 3, 4, 5, 6>, parameters = [unit, [2, 2], [1, 1], [1, 1], 1]} : (tensor<1x64x56x56xf32>, tensor<128x64x3x3xf32>) -> tensor<1x128x28x28xf32>
    %58 = torch_c.from_builtin_tensor %57 : tensor<1x128x28x28xf32> -> !torch.vtensor<[1,128,28,28],f32>
    %59 = torch_c.to_builtin_tensor %58 : !torch.vtensor<[1,128,28,28],f32> -> tensor<1x128x28x28xf32>
    %60:3 = dnn.batch_norm %59, %cst_11, %cst_11, %cst_11, %cst_11 {kind = "aten._native_batch_norm_legit_no_training", parameter_indices = array<i32: 5, 6>, parameters = [1.000000e-01, 1.000000e-05]} : (tensor<1x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) -> (tensor<1x128x28x28xf32>, tensor<0xf32>, tensor<0xf32>)
    %61 = torch_c.from_builtin_tensor %60#0 : tensor<1x128x28x28xf32> -> !torch.vtensor<[1,128,28,28],f32>
    %62 = torch_c.to_builtin_tensor %61 : !torch.vtensor<[1,128,28,28],f32> -> tensor<1x128x28x28xf32>
    %63 = dnn.relu %62 : (tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %64 = torch_c.from_builtin_tensor %63 : tensor<1x128x28x28xf32> -> !torch.vtensor<[1,128,28,28],f32>
    %65 = torch_c.to_builtin_tensor %64 : !torch.vtensor<[1,128,28,28],f32> -> tensor<1x128x28x28xf32>
    %66 = dnn.convolution %65, %cst_10 {kind = "aten.conv2d", parameter_indices = array<i32: 2, 3, 4, 5, 6>, parameters = [unit, [1, 1], [1, 1], [1, 1], 1]} : (tensor<1x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<1x128x28x28xf32>
    %67 = torch_c.from_builtin_tensor %66 : tensor<1x128x28x28xf32> -> !torch.vtensor<[1,128,28,28],f32>
    %68 = torch_c.to_builtin_tensor %67 : !torch.vtensor<[1,128,28,28],f32> -> tensor<1x128x28x28xf32>
    %69:3 = dnn.batch_norm %68, %cst_11, %cst_11, %cst_11, %cst_11 {kind = "aten._native_batch_norm_legit_no_training", parameter_indices = array<i32: 5, 6>, parameters = [1.000000e-01, 1.000000e-05]} : (tensor<1x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) -> (tensor<1x128x28x28xf32>, tensor<0xf32>, tensor<0xf32>)
    %70 = torch_c.from_builtin_tensor %69#0 : tensor<1x128x28x28xf32> -> !torch.vtensor<[1,128,28,28],f32>
    %71 = torch_c.to_builtin_tensor %55 : !torch.vtensor<[1,64,56,56],f32> -> tensor<1x64x56x56xf32>
    %72 = dnn.convolution %71, %cst_9 {kind = "aten.conv2d", parameter_indices = array<i32: 2, 3, 4, 5, 6>, parameters = [unit, [2, 2], [0, 0], [1, 1], 1]} : (tensor<1x64x56x56xf32>, tensor<128x64x1x1xf32>) -> tensor<1x128x28x28xf32>
    %73 = torch_c.from_builtin_tensor %72 : tensor<1x128x28x28xf32> -> !torch.vtensor<[1,128,28,28],f32>
    %74 = torch_c.to_builtin_tensor %73 : !torch.vtensor<[1,128,28,28],f32> -> tensor<1x128x28x28xf32>
    %75:3 = dnn.batch_norm %74, %cst_11, %cst_11, %cst_11, %cst_11 {kind = "aten._native_batch_norm_legit_no_training", parameter_indices = array<i32: 5, 6>, parameters = [1.000000e-01, 1.000000e-05]} : (tensor<1x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) -> (tensor<1x128x28x28xf32>, tensor<0xf32>, tensor<0xf32>)
    %76 = torch_c.from_builtin_tensor %75#0 : tensor<1x128x28x28xf32> -> !torch.vtensor<[1,128,28,28],f32>
    %77 = torch_c.to_builtin_tensor %70 : !torch.vtensor<[1,128,28,28],f32> -> tensor<1x128x28x28xf32>
    %78 = torch_c.to_builtin_tensor %76 : !torch.vtensor<[1,128,28,28],f32> -> tensor<1x128x28x28xf32>
    %79 = dnn.add %77, %78 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<1x128x28x28xf32>, tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %80 = torch_c.from_builtin_tensor %79 : tensor<1x128x28x28xf32> -> !torch.vtensor<[1,128,28,28],f32>
    %81 = torch_c.to_builtin_tensor %80 : !torch.vtensor<[1,128,28,28],f32> -> tensor<1x128x28x28xf32>
    %82 = dnn.relu %81 : (tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %83 = torch_c.from_builtin_tensor %82 : tensor<1x128x28x28xf32> -> !torch.vtensor<[1,128,28,28],f32>
    %84 = torch_c.to_builtin_tensor %83 : !torch.vtensor<[1,128,28,28],f32> -> tensor<1x128x28x28xf32>
    %85 = dnn.convolution %84, %cst_10 {kind = "aten.conv2d", parameter_indices = array<i32: 2, 3, 4, 5, 6>, parameters = [unit, [1, 1], [1, 1], [1, 1], 1]} : (tensor<1x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<1x128x28x28xf32>
    %86 = torch_c.from_builtin_tensor %85 : tensor<1x128x28x28xf32> -> !torch.vtensor<[1,128,28,28],f32>
    %87 = torch_c.to_builtin_tensor %86 : !torch.vtensor<[1,128,28,28],f32> -> tensor<1x128x28x28xf32>
    %88:3 = dnn.batch_norm %87, %cst_11, %cst_11, %cst_11, %cst_11 {kind = "aten._native_batch_norm_legit_no_training", parameter_indices = array<i32: 5, 6>, parameters = [1.000000e-01, 1.000000e-05]} : (tensor<1x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) -> (tensor<1x128x28x28xf32>, tensor<0xf32>, tensor<0xf32>)
    %89 = torch_c.from_builtin_tensor %88#0 : tensor<1x128x28x28xf32> -> !torch.vtensor<[1,128,28,28],f32>
    %90 = torch_c.to_builtin_tensor %89 : !torch.vtensor<[1,128,28,28],f32> -> tensor<1x128x28x28xf32>
    %91 = dnn.relu %90 : (tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %92 = torch_c.from_builtin_tensor %91 : tensor<1x128x28x28xf32> -> !torch.vtensor<[1,128,28,28],f32>
    %93 = torch_c.to_builtin_tensor %92 : !torch.vtensor<[1,128,28,28],f32> -> tensor<1x128x28x28xf32>
    %94 = dnn.convolution %93, %cst_10 {kind = "aten.conv2d", parameter_indices = array<i32: 2, 3, 4, 5, 6>, parameters = [unit, [1, 1], [1, 1], [1, 1], 1]} : (tensor<1x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<1x128x28x28xf32>
    %95 = torch_c.from_builtin_tensor %94 : tensor<1x128x28x28xf32> -> !torch.vtensor<[1,128,28,28],f32>
    %96 = torch_c.to_builtin_tensor %95 : !torch.vtensor<[1,128,28,28],f32> -> tensor<1x128x28x28xf32>
    %97:3 = dnn.batch_norm %96, %cst_11, %cst_11, %cst_11, %cst_11 {kind = "aten._native_batch_norm_legit_no_training", parameter_indices = array<i32: 5, 6>, parameters = [1.000000e-01, 1.000000e-05]} : (tensor<1x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) -> (tensor<1x128x28x28xf32>, tensor<0xf32>, tensor<0xf32>)
    %98 = torch_c.from_builtin_tensor %97#0 : tensor<1x128x28x28xf32> -> !torch.vtensor<[1,128,28,28],f32>
    %99 = torch_c.to_builtin_tensor %98 : !torch.vtensor<[1,128,28,28],f32> -> tensor<1x128x28x28xf32>
    %100 = torch_c.to_builtin_tensor %83 : !torch.vtensor<[1,128,28,28],f32> -> tensor<1x128x28x28xf32>
    %101 = dnn.add %99, %100 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<1x128x28x28xf32>, tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %102 = torch_c.from_builtin_tensor %101 : tensor<1x128x28x28xf32> -> !torch.vtensor<[1,128,28,28],f32>
    %103 = torch_c.to_builtin_tensor %102 : !torch.vtensor<[1,128,28,28],f32> -> tensor<1x128x28x28xf32>
    %104 = dnn.relu %103 : (tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %105 = torch_c.from_builtin_tensor %104 : tensor<1x128x28x28xf32> -> !torch.vtensor<[1,128,28,28],f32>
    %106 = torch_c.to_builtin_tensor %105 : !torch.vtensor<[1,128,28,28],f32> -> tensor<1x128x28x28xf32>
    %107 = dnn.convolution %106, %cst_8 {kind = "aten.conv2d", parameter_indices = array<i32: 2, 3, 4, 5, 6>, parameters = [unit, [2, 2], [1, 1], [1, 1], 1]} : (tensor<1x128x28x28xf32>, tensor<256x128x3x3xf32>) -> tensor<1x256x14x14xf32>
    %108 = torch_c.from_builtin_tensor %107 : tensor<1x256x14x14xf32> -> !torch.vtensor<[1,256,14,14],f32>
    %109 = torch_c.to_builtin_tensor %108 : !torch.vtensor<[1,256,14,14],f32> -> tensor<1x256x14x14xf32>
    %110:3 = dnn.batch_norm %109, %cst_7, %cst_7, %cst_7, %cst_7 {kind = "aten._native_batch_norm_legit_no_training", parameter_indices = array<i32: 5, 6>, parameters = [1.000000e-01, 1.000000e-05]} : (tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> (tensor<1x256x14x14xf32>, tensor<0xf32>, tensor<0xf32>)
    %111 = torch_c.from_builtin_tensor %110#0 : tensor<1x256x14x14xf32> -> !torch.vtensor<[1,256,14,14],f32>
    %112 = torch_c.to_builtin_tensor %111 : !torch.vtensor<[1,256,14,14],f32> -> tensor<1x256x14x14xf32>
    %113 = dnn.relu %112 : (tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %114 = torch_c.from_builtin_tensor %113 : tensor<1x256x14x14xf32> -> !torch.vtensor<[1,256,14,14],f32>
    %115 = torch_c.to_builtin_tensor %114 : !torch.vtensor<[1,256,14,14],f32> -> tensor<1x256x14x14xf32>
    %116 = dnn.convolution %115, %cst_6 {kind = "aten.conv2d", parameter_indices = array<i32: 2, 3, 4, 5, 6>, parameters = [unit, [1, 1], [1, 1], [1, 1], 1]} : (tensor<1x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<1x256x14x14xf32>
    %117 = torch_c.from_builtin_tensor %116 : tensor<1x256x14x14xf32> -> !torch.vtensor<[1,256,14,14],f32>
    %118 = torch_c.to_builtin_tensor %117 : !torch.vtensor<[1,256,14,14],f32> -> tensor<1x256x14x14xf32>
    %119:3 = dnn.batch_norm %118, %cst_7, %cst_7, %cst_7, %cst_7 {kind = "aten._native_batch_norm_legit_no_training", parameter_indices = array<i32: 5, 6>, parameters = [1.000000e-01, 1.000000e-05]} : (tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> (tensor<1x256x14x14xf32>, tensor<0xf32>, tensor<0xf32>)
    %120 = torch_c.from_builtin_tensor %119#0 : tensor<1x256x14x14xf32> -> !torch.vtensor<[1,256,14,14],f32>
    %121 = torch_c.to_builtin_tensor %105 : !torch.vtensor<[1,128,28,28],f32> -> tensor<1x128x28x28xf32>
    %122 = dnn.convolution %121, %cst_5 {kind = "aten.conv2d", parameter_indices = array<i32: 2, 3, 4, 5, 6>, parameters = [unit, [2, 2], [0, 0], [1, 1], 1]} : (tensor<1x128x28x28xf32>, tensor<256x128x1x1xf32>) -> tensor<1x256x14x14xf32>
    %123 = torch_c.from_builtin_tensor %122 : tensor<1x256x14x14xf32> -> !torch.vtensor<[1,256,14,14],f32>
    %124 = torch_c.to_builtin_tensor %123 : !torch.vtensor<[1,256,14,14],f32> -> tensor<1x256x14x14xf32>
    %125:3 = dnn.batch_norm %124, %cst_7, %cst_7, %cst_7, %cst_7 {kind = "aten._native_batch_norm_legit_no_training", parameter_indices = array<i32: 5, 6>, parameters = [1.000000e-01, 1.000000e-05]} : (tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> (tensor<1x256x14x14xf32>, tensor<0xf32>, tensor<0xf32>)
    %126 = torch_c.from_builtin_tensor %125#0 : tensor<1x256x14x14xf32> -> !torch.vtensor<[1,256,14,14],f32>
    %127 = torch_c.to_builtin_tensor %120 : !torch.vtensor<[1,256,14,14],f32> -> tensor<1x256x14x14xf32>
    %128 = torch_c.to_builtin_tensor %126 : !torch.vtensor<[1,256,14,14],f32> -> tensor<1x256x14x14xf32>
    %129 = dnn.add %127, %128 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<1x256x14x14xf32>, tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %130 = torch_c.from_builtin_tensor %129 : tensor<1x256x14x14xf32> -> !torch.vtensor<[1,256,14,14],f32>
    %131 = torch_c.to_builtin_tensor %130 : !torch.vtensor<[1,256,14,14],f32> -> tensor<1x256x14x14xf32>
    %132 = dnn.relu %131 : (tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %133 = torch_c.from_builtin_tensor %132 : tensor<1x256x14x14xf32> -> !torch.vtensor<[1,256,14,14],f32>
    %134 = torch_c.to_builtin_tensor %133 : !torch.vtensor<[1,256,14,14],f32> -> tensor<1x256x14x14xf32>
    %135 = dnn.convolution %134, %cst_6 {kind = "aten.conv2d", parameter_indices = array<i32: 2, 3, 4, 5, 6>, parameters = [unit, [1, 1], [1, 1], [1, 1], 1]} : (tensor<1x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<1x256x14x14xf32>
    %136 = torch_c.from_builtin_tensor %135 : tensor<1x256x14x14xf32> -> !torch.vtensor<[1,256,14,14],f32>
    %137 = torch_c.to_builtin_tensor %136 : !torch.vtensor<[1,256,14,14],f32> -> tensor<1x256x14x14xf32>
    %138:3 = dnn.batch_norm %137, %cst_7, %cst_7, %cst_7, %cst_7 {kind = "aten._native_batch_norm_legit_no_training", parameter_indices = array<i32: 5, 6>, parameters = [1.000000e-01, 1.000000e-05]} : (tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> (tensor<1x256x14x14xf32>, tensor<0xf32>, tensor<0xf32>)
    %139 = torch_c.from_builtin_tensor %138#0 : tensor<1x256x14x14xf32> -> !torch.vtensor<[1,256,14,14],f32>
    %140 = torch_c.to_builtin_tensor %139 : !torch.vtensor<[1,256,14,14],f32> -> tensor<1x256x14x14xf32>
    %141 = dnn.relu %140 : (tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %142 = torch_c.from_builtin_tensor %141 : tensor<1x256x14x14xf32> -> !torch.vtensor<[1,256,14,14],f32>
    %143 = torch_c.to_builtin_tensor %142 : !torch.vtensor<[1,256,14,14],f32> -> tensor<1x256x14x14xf32>
    %144 = dnn.convolution %143, %cst_6 {kind = "aten.conv2d", parameter_indices = array<i32: 2, 3, 4, 5, 6>, parameters = [unit, [1, 1], [1, 1], [1, 1], 1]} : (tensor<1x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<1x256x14x14xf32>
    %145 = torch_c.from_builtin_tensor %144 : tensor<1x256x14x14xf32> -> !torch.vtensor<[1,256,14,14],f32>
    %146 = torch_c.to_builtin_tensor %145 : !torch.vtensor<[1,256,14,14],f32> -> tensor<1x256x14x14xf32>
    %147:3 = dnn.batch_norm %146, %cst_7, %cst_7, %cst_7, %cst_7 {kind = "aten._native_batch_norm_legit_no_training", parameter_indices = array<i32: 5, 6>, parameters = [1.000000e-01, 1.000000e-05]} : (tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> (tensor<1x256x14x14xf32>, tensor<0xf32>, tensor<0xf32>)
    %148 = torch_c.from_builtin_tensor %147#0 : tensor<1x256x14x14xf32> -> !torch.vtensor<[1,256,14,14],f32>
    %149 = torch_c.to_builtin_tensor %148 : !torch.vtensor<[1,256,14,14],f32> -> tensor<1x256x14x14xf32>
    %150 = torch_c.to_builtin_tensor %133 : !torch.vtensor<[1,256,14,14],f32> -> tensor<1x256x14x14xf32>
    %151 = dnn.add %149, %150 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<1x256x14x14xf32>, tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %152 = torch_c.from_builtin_tensor %151 : tensor<1x256x14x14xf32> -> !torch.vtensor<[1,256,14,14],f32>
    %153 = torch_c.to_builtin_tensor %152 : !torch.vtensor<[1,256,14,14],f32> -> tensor<1x256x14x14xf32>
    %154 = dnn.relu %153 : (tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %155 = torch_c.from_builtin_tensor %154 : tensor<1x256x14x14xf32> -> !torch.vtensor<[1,256,14,14],f32>
    %156 = torch_c.to_builtin_tensor %155 : !torch.vtensor<[1,256,14,14],f32> -> tensor<1x256x14x14xf32>
    %157 = dnn.convolution %156, %cst_4 {kind = "aten.conv2d", parameter_indices = array<i32: 2, 3, 4, 5, 6>, parameters = [unit, [2, 2], [1, 1], [1, 1], 1]} : (tensor<1x256x14x14xf32>, tensor<512x256x3x3xf32>) -> tensor<1x512x7x7xf32>
    %158 = torch_c.from_builtin_tensor %157 : tensor<1x512x7x7xf32> -> !torch.vtensor<[1,512,7,7],f32>
    %159 = torch_c.to_builtin_tensor %158 : !torch.vtensor<[1,512,7,7],f32> -> tensor<1x512x7x7xf32>
    %160:3 = dnn.batch_norm %159, %cst_3, %cst_3, %cst_3, %cst_3 {kind = "aten._native_batch_norm_legit_no_training", parameter_indices = array<i32: 5, 6>, parameters = [1.000000e-01, 1.000000e-05]} : (tensor<1x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) -> (tensor<1x512x7x7xf32>, tensor<0xf32>, tensor<0xf32>)
    %161 = torch_c.from_builtin_tensor %160#0 : tensor<1x512x7x7xf32> -> !torch.vtensor<[1,512,7,7],f32>
    %162 = torch_c.to_builtin_tensor %161 : !torch.vtensor<[1,512,7,7],f32> -> tensor<1x512x7x7xf32>
    %163 = dnn.relu %162 : (tensor<1x512x7x7xf32>) -> tensor<1x512x7x7xf32>
    %164 = torch_c.from_builtin_tensor %163 : tensor<1x512x7x7xf32> -> !torch.vtensor<[1,512,7,7],f32>
    %165 = torch_c.to_builtin_tensor %164 : !torch.vtensor<[1,512,7,7],f32> -> tensor<1x512x7x7xf32>
    %166 = dnn.convolution %165, %cst_2 {kind = "aten.conv2d", parameter_indices = array<i32: 2, 3, 4, 5, 6>, parameters = [unit, [1, 1], [1, 1], [1, 1], 1]} : (tensor<1x512x7x7xf32>, tensor<512x512x3x3xf32>) -> tensor<1x512x7x7xf32>
    %167 = torch_c.from_builtin_tensor %166 : tensor<1x512x7x7xf32> -> !torch.vtensor<[1,512,7,7],f32>
    %168 = torch_c.to_builtin_tensor %167 : !torch.vtensor<[1,512,7,7],f32> -> tensor<1x512x7x7xf32>
    %169:3 = dnn.batch_norm %168, %cst_3, %cst_3, %cst_3, %cst_3 {kind = "aten._native_batch_norm_legit_no_training", parameter_indices = array<i32: 5, 6>, parameters = [1.000000e-01, 1.000000e-05]} : (tensor<1x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) -> (tensor<1x512x7x7xf32>, tensor<0xf32>, tensor<0xf32>)
    %170 = torch_c.from_builtin_tensor %169#0 : tensor<1x512x7x7xf32> -> !torch.vtensor<[1,512,7,7],f32>
    %171 = torch_c.to_builtin_tensor %155 : !torch.vtensor<[1,256,14,14],f32> -> tensor<1x256x14x14xf32>
    %172 = dnn.convolution %171, %cst_1 {kind = "aten.conv2d", parameter_indices = array<i32: 2, 3, 4, 5, 6>, parameters = [unit, [2, 2], [0, 0], [1, 1], 1]} : (tensor<1x256x14x14xf32>, tensor<512x256x1x1xf32>) -> tensor<1x512x7x7xf32>
    %173 = torch_c.from_builtin_tensor %172 : tensor<1x512x7x7xf32> -> !torch.vtensor<[1,512,7,7],f32>
    %174 = torch_c.to_builtin_tensor %173 : !torch.vtensor<[1,512,7,7],f32> -> tensor<1x512x7x7xf32>
    %175:3 = dnn.batch_norm %174, %cst_3, %cst_3, %cst_3, %cst_3 {kind = "aten._native_batch_norm_legit_no_training", parameter_indices = array<i32: 5, 6>, parameters = [1.000000e-01, 1.000000e-05]} : (tensor<1x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) -> (tensor<1x512x7x7xf32>, tensor<0xf32>, tensor<0xf32>)
    %176 = torch_c.from_builtin_tensor %175#0 : tensor<1x512x7x7xf32> -> !torch.vtensor<[1,512,7,7],f32>
    %177 = torch_c.to_builtin_tensor %170 : !torch.vtensor<[1,512,7,7],f32> -> tensor<1x512x7x7xf32>
    %178 = torch_c.to_builtin_tensor %176 : !torch.vtensor<[1,512,7,7],f32> -> tensor<1x512x7x7xf32>
    %179 = dnn.add %177, %178 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<1x512x7x7xf32>, tensor<1x512x7x7xf32>) -> tensor<1x512x7x7xf32>
    %180 = torch_c.from_builtin_tensor %179 : tensor<1x512x7x7xf32> -> !torch.vtensor<[1,512,7,7],f32>
    %181 = torch_c.to_builtin_tensor %180 : !torch.vtensor<[1,512,7,7],f32> -> tensor<1x512x7x7xf32>
    %182 = dnn.relu %181 : (tensor<1x512x7x7xf32>) -> tensor<1x512x7x7xf32>
    %183 = torch_c.from_builtin_tensor %182 : tensor<1x512x7x7xf32> -> !torch.vtensor<[1,512,7,7],f32>
    %184 = torch_c.to_builtin_tensor %183 : !torch.vtensor<[1,512,7,7],f32> -> tensor<1x512x7x7xf32>
    %185 = dnn.convolution %184, %cst_2 {kind = "aten.conv2d", parameter_indices = array<i32: 2, 3, 4, 5, 6>, parameters = [unit, [1, 1], [1, 1], [1, 1], 1]} : (tensor<1x512x7x7xf32>, tensor<512x512x3x3xf32>) -> tensor<1x512x7x7xf32>
    %186 = torch_c.from_builtin_tensor %185 : tensor<1x512x7x7xf32> -> !torch.vtensor<[1,512,7,7],f32>
    %187 = torch_c.to_builtin_tensor %186 : !torch.vtensor<[1,512,7,7],f32> -> tensor<1x512x7x7xf32>
    %188:3 = dnn.batch_norm %187, %cst_3, %cst_3, %cst_3, %cst_3 {kind = "aten._native_batch_norm_legit_no_training", parameter_indices = array<i32: 5, 6>, parameters = [1.000000e-01, 1.000000e-05]} : (tensor<1x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) -> (tensor<1x512x7x7xf32>, tensor<0xf32>, tensor<0xf32>)
    %189 = torch_c.from_builtin_tensor %188#0 : tensor<1x512x7x7xf32> -> !torch.vtensor<[1,512,7,7],f32>
    %190 = torch_c.to_builtin_tensor %189 : !torch.vtensor<[1,512,7,7],f32> -> tensor<1x512x7x7xf32>
    %191 = dnn.relu %190 : (tensor<1x512x7x7xf32>) -> tensor<1x512x7x7xf32>
    %192 = torch_c.from_builtin_tensor %191 : tensor<1x512x7x7xf32> -> !torch.vtensor<[1,512,7,7],f32>
    %193 = torch_c.to_builtin_tensor %192 : !torch.vtensor<[1,512,7,7],f32> -> tensor<1x512x7x7xf32>
    %194 = dnn.convolution %193, %cst_2 {kind = "aten.conv2d", parameter_indices = array<i32: 2, 3, 4, 5, 6>, parameters = [unit, [1, 1], [1, 1], [1, 1], 1]} : (tensor<1x512x7x7xf32>, tensor<512x512x3x3xf32>) -> tensor<1x512x7x7xf32>
    %195 = torch_c.from_builtin_tensor %194 : tensor<1x512x7x7xf32> -> !torch.vtensor<[1,512,7,7],f32>
    %196 = torch_c.to_builtin_tensor %195 : !torch.vtensor<[1,512,7,7],f32> -> tensor<1x512x7x7xf32>
    %197:3 = dnn.batch_norm %196, %cst_3, %cst_3, %cst_3, %cst_3 {kind = "aten._native_batch_norm_legit_no_training", parameter_indices = array<i32: 5, 6>, parameters = [1.000000e-01, 1.000000e-05]} : (tensor<1x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) -> (tensor<1x512x7x7xf32>, tensor<0xf32>, tensor<0xf32>)
    %198 = torch_c.from_builtin_tensor %197#0 : tensor<1x512x7x7xf32> -> !torch.vtensor<[1,512,7,7],f32>
    %199 = torch_c.to_builtin_tensor %198 : !torch.vtensor<[1,512,7,7],f32> -> tensor<1x512x7x7xf32>
    %200 = torch_c.to_builtin_tensor %183 : !torch.vtensor<[1,512,7,7],f32> -> tensor<1x512x7x7xf32>
    %201 = dnn.add %199, %200 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<1x512x7x7xf32>, tensor<1x512x7x7xf32>) -> tensor<1x512x7x7xf32>
    %202 = torch_c.from_builtin_tensor %201 : tensor<1x512x7x7xf32> -> !torch.vtensor<[1,512,7,7],f32>
    %203 = torch_c.to_builtin_tensor %202 : !torch.vtensor<[1,512,7,7],f32> -> tensor<1x512x7x7xf32>
    %204 = dnn.relu %203 : (tensor<1x512x7x7xf32>) -> tensor<1x512x7x7xf32>
    %205 = torch_c.from_builtin_tensor %204 : tensor<1x512x7x7xf32> -> !torch.vtensor<[1,512,7,7],f32>
    %206 = torch_c.to_builtin_tensor %205 : !torch.vtensor<[1,512,7,7],f32> -> tensor<1x512x7x7xf32>
    %207 = dnn.adaptive_avg_pool2d %206 {parameter_indices = array<i32: 1>, parameters = [[1, 1]]} : (tensor<1x512x7x7xf32>) -> tensor<1x512x1x1xf32>
    %208 = torch_c.from_builtin_tensor %207 : tensor<1x512x1x1xf32> -> !torch.vtensor<[1,512,1,1],f32>
    %209 = torch.prim.ListConstruct %int1, %int512 : (!torch.int, !torch.int) -> !torch.list<int>
    %210 = torch.aten.view %208, %209 : !torch.vtensor<[1,512,1,1],f32>, !torch.list<int> -> !torch.vtensor<[1,512],f32>
    %211 = torch_c.to_builtin_tensor %210 : !torch.vtensor<[1,512],f32> -> tensor<1x512xf32>
    %212 = dnn.linear %211, %cst_0, %cst : tensor<1x512xf32>, tensor<1000x512xf32>, tensor<1000xf32> -> tensor<1x1000xf32>
    %213 = torch_c.from_builtin_tensor %212 : tensor<1x1000xf32> -> !torch.vtensor<[1,1000],f32>
    return %213 : !torch.vtensor<[1,1000],f32>
  }
}

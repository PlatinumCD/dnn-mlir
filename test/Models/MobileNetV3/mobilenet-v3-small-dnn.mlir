// RUN: dnn-mlir-opt %s | FileCheck %s
//
// Structural MobileNetV3 Small inference graph imported through Torch-MLIR's
// FX path with batch-normalization decomposition disabled, then converted with
// the Torch-to-DNN pass. Parameter tensors use deterministic splat values.
//
// CHECK-LABEL: func.func @mobilenet_v3_small
// CHECK-DAG: dnn.convolution
// CHECK-DAG: parameters = [unit, [2, 2], [1, 1], [1, 1], 16]
// CHECK-DAG: dnn.batch_norm
// CHECK-DAG: dnn.hardswish
// CHECK-DAG: dnn.hardsigmoid
// CHECK-DAG: dnn.adaptive_avg_pool2d
// CHECK-DAG: dnn.mul
// CHECK-DAG: dnn.add
// CHECK-DAG: dnn.linear

module {
  func.func @mobilenet_v3_small(%arg0: !torch.vtensor<[1,3,224,224],f32>) -> !torch.vtensor<[1,1000],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %cst = arith.constant dense<0.000000e+00> : tensor<1000xf32>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<1000x1024xf32>
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<1024xf32>
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<1024x576xf32>
    %cst_3 = arith.constant dense<0.000000e+00> : tensor<96x576x1x1xf32>
    %cst_4 = arith.constant dense<0.000000e+00> : tensor<576x144x1x1xf32>
    %cst_5 = arith.constant dense<0.000000e+00> : tensor<144x576x1x1xf32>
    %cst_6 = arith.constant dense<0.000000e+00> : tensor<576x1x5x5xf32>
    %cst_7 = arith.constant dense<0.000000e+00> : tensor<576xf32>
    %cst_8 = arith.constant dense<0.000000e+00> : tensor<576x96x1x1xf32>
    %cst_9 = arith.constant dense<0.000000e+00> : tensor<96x288x1x1xf32>
    %cst_10 = arith.constant dense<0.000000e+00> : tensor<288x72x1x1xf32>
    %cst_11 = arith.constant dense<0.000000e+00> : tensor<72x288x1x1xf32>
    %cst_12 = arith.constant dense<0.000000e+00> : tensor<288x1x5x5xf32>
    %cst_13 = arith.constant dense<0.000000e+00> : tensor<288xf32>
    %cst_14 = arith.constant dense<0.000000e+00> : tensor<288x48x1x1xf32>
    %cst_15 = arith.constant dense<0.000000e+00> : tensor<48x144x1x1xf32>
    %cst_16 = arith.constant dense<0.000000e+00> : tensor<144x40x1x1xf32>
    %cst_17 = arith.constant dense<0.000000e+00> : tensor<40x144x1x1xf32>
    %cst_18 = arith.constant dense<0.000000e+00> : tensor<144x1x5x5xf32>
    %cst_19 = arith.constant dense<0.000000e+00> : tensor<144xf32>
    %cst_20 = arith.constant dense<0.000000e+00> : tensor<144x48x1x1xf32>
    %cst_21 = arith.constant dense<0.000000e+00> : tensor<48xf32>
    %cst_22 = arith.constant dense<0.000000e+00> : tensor<48x120x1x1xf32>
    %cst_23 = arith.constant dense<0.000000e+00> : tensor<120x32x1x1xf32>
    %cst_24 = arith.constant dense<0.000000e+00> : tensor<32xf32>
    %cst_25 = arith.constant dense<0.000000e+00> : tensor<32x120x1x1xf32>
    %cst_26 = arith.constant dense<0.000000e+00> : tensor<120x1x5x5xf32>
    %cst_27 = arith.constant dense<0.000000e+00> : tensor<120xf32>
    %cst_28 = arith.constant dense<0.000000e+00> : tensor<120x40x1x1xf32>
    %cst_29 = arith.constant dense<0.000000e+00> : tensor<40x240x1x1xf32>
    %cst_30 = arith.constant dense<0.000000e+00> : tensor<240x64x1x1xf32>
    %cst_31 = arith.constant dense<0.000000e+00> : tensor<64xf32>
    %cst_32 = arith.constant dense<0.000000e+00> : tensor<64x240x1x1xf32>
    %cst_33 = arith.constant dense<0.000000e+00> : tensor<240x1x5x5xf32>
    %cst_34 = arith.constant dense<0.000000e+00> : tensor<240xf32>
    %cst_35 = arith.constant dense<0.000000e+00> : tensor<240x40x1x1xf32>
    %cst_36 = arith.constant dense<0.000000e+00> : tensor<40xf32>
    %cst_37 = arith.constant dense<0.000000e+00> : tensor<40x96x1x1xf32>
    %cst_38 = arith.constant dense<0.000000e+00> : tensor<24x96x1x1xf32>
    %cst_39 = arith.constant dense<0.000000e+00> : tensor<96x1x5x5xf32>
    %cst_40 = arith.constant dense<0.000000e+00> : tensor<96xf32>
    %cst_41 = arith.constant dense<0.000000e+00> : tensor<96x24x1x1xf32>
    %cst_42 = arith.constant dense<0.000000e+00> : tensor<24x88x1x1xf32>
    %cst_43 = arith.constant dense<0.000000e+00> : tensor<88x1x3x3xf32>
    %cst_44 = arith.constant dense<0.000000e+00> : tensor<88xf32>
    %cst_45 = arith.constant dense<0.000000e+00> : tensor<88x24x1x1xf32>
    %cst_46 = arith.constant dense<0.000000e+00> : tensor<24xf32>
    %cst_47 = arith.constant dense<0.000000e+00> : tensor<24x72x1x1xf32>
    %cst_48 = arith.constant dense<0.000000e+00> : tensor<72x1x3x3xf32>
    %cst_49 = arith.constant dense<0.000000e+00> : tensor<72xf32>
    %cst_50 = arith.constant dense<0.000000e+00> : tensor<72x16x1x1xf32>
    %cst_51 = arith.constant dense<0.000000e+00> : tensor<16x16x1x1xf32>
    %cst_52 = arith.constant dense<0.000000e+00> : tensor<16x8x1x1xf32>
    %cst_53 = arith.constant dense<0.000000e+00> : tensor<8xf32>
    %cst_54 = arith.constant dense<0.000000e+00> : tensor<8x16x1x1xf32>
    %cst_55 = arith.constant dense<0.000000e+00> : tensor<16x1x3x3xf32>
    %cst_56 = arith.constant dense<0.000000e+00> : tensor<16xf32>
    %cst_57 = arith.constant dense<0.000000e+00> : tensor<16x3x3x3xf32>
    %int576 = torch.constant.int 576
    %int1 = torch.constant.int 1
    %0 = torch_c.to_builtin_tensor %arg0 : !torch.vtensor<[1,3,224,224],f32> -> tensor<1x3x224x224xf32>
    %1 = dnn.convolution %0, %cst_57 {kind = "aten.conv2d", parameter_indices = array<i32: 2, 3, 4, 5, 6>, parameters = [unit, [2, 2], [1, 1], [1, 1], 1]} : (tensor<1x3x224x224xf32>, tensor<16x3x3x3xf32>) -> tensor<1x16x112x112xf32>
    %2 = torch_c.from_builtin_tensor %1 : tensor<1x16x112x112xf32> -> !torch.vtensor<[1,16,112,112],f32>
    %3 = torch_c.to_builtin_tensor %2 : !torch.vtensor<[1,16,112,112],f32> -> tensor<1x16x112x112xf32>
    %4:3 = dnn.batch_norm %3, %cst_56, %cst_56, %cst_56, %cst_56 {kind = "aten._native_batch_norm_legit_no_training", parameter_indices = array<i32: 5, 6>, parameters = [1.000000e-02, 1.000000e-03]} : (tensor<1x16x112x112xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>) -> (tensor<1x16x112x112xf32>, tensor<0xf32>, tensor<0xf32>)
    %5 = torch_c.from_builtin_tensor %4#0 : tensor<1x16x112x112xf32> -> !torch.vtensor<[1,16,112,112],f32>
    %6 = torch_c.to_builtin_tensor %5 : !torch.vtensor<[1,16,112,112],f32> -> tensor<1x16x112x112xf32>
    %7 = dnn.hardswish %6 : (tensor<1x16x112x112xf32>) -> tensor<1x16x112x112xf32>
    %8 = torch_c.from_builtin_tensor %7 : tensor<1x16x112x112xf32> -> !torch.vtensor<[1,16,112,112],f32>
    %9 = torch_c.to_builtin_tensor %8 : !torch.vtensor<[1,16,112,112],f32> -> tensor<1x16x112x112xf32>
    %10 = dnn.convolution %9, %cst_55 {kind = "aten.conv2d", parameter_indices = array<i32: 2, 3, 4, 5, 6>, parameters = [unit, [2, 2], [1, 1], [1, 1], 16]} : (tensor<1x16x112x112xf32>, tensor<16x1x3x3xf32>) -> tensor<1x16x56x56xf32>
    %11 = torch_c.from_builtin_tensor %10 : tensor<1x16x56x56xf32> -> !torch.vtensor<[1,16,56,56],f32>
    %12 = torch_c.to_builtin_tensor %11 : !torch.vtensor<[1,16,56,56],f32> -> tensor<1x16x56x56xf32>
    %13:3 = dnn.batch_norm %12, %cst_56, %cst_56, %cst_56, %cst_56 {kind = "aten._native_batch_norm_legit_no_training", parameter_indices = array<i32: 5, 6>, parameters = [1.000000e-02, 1.000000e-03]} : (tensor<1x16x56x56xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>) -> (tensor<1x16x56x56xf32>, tensor<0xf32>, tensor<0xf32>)
    %14 = torch_c.from_builtin_tensor %13#0 : tensor<1x16x56x56xf32> -> !torch.vtensor<[1,16,56,56],f32>
    %15 = torch_c.to_builtin_tensor %14 : !torch.vtensor<[1,16,56,56],f32> -> tensor<1x16x56x56xf32>
    %16 = dnn.relu %15 : (tensor<1x16x56x56xf32>) -> tensor<1x16x56x56xf32>
    %17 = torch_c.from_builtin_tensor %16 : tensor<1x16x56x56xf32> -> !torch.vtensor<[1,16,56,56],f32>
    %18 = torch_c.to_builtin_tensor %17 : !torch.vtensor<[1,16,56,56],f32> -> tensor<1x16x56x56xf32>
    %19 = dnn.adaptive_avg_pool2d %18 {parameter_indices = array<i32: 1>, parameters = [[1, 1]]} : (tensor<1x16x56x56xf32>) -> tensor<1x16x1x1xf32>
    %20 = torch_c.from_builtin_tensor %19 : tensor<1x16x1x1xf32> -> !torch.vtensor<[1,16,1,1],f32>
    %21 = torch_c.to_builtin_tensor %20 : !torch.vtensor<[1,16,1,1],f32> -> tensor<1x16x1x1xf32>
    %22 = dnn.convolution %21, %cst_54, %cst_53 {kind = "aten.conv2d", parameter_indices = array<i32: 3, 4, 5, 6>, parameters = [[1, 1], [0, 0], [1, 1], 1]} : (tensor<1x16x1x1xf32>, tensor<8x16x1x1xf32>, tensor<8xf32>) -> tensor<1x8x1x1xf32>
    %23 = torch_c.from_builtin_tensor %22 : tensor<1x8x1x1xf32> -> !torch.vtensor<[1,8,1,1],f32>
    %24 = torch_c.to_builtin_tensor %23 : !torch.vtensor<[1,8,1,1],f32> -> tensor<1x8x1x1xf32>
    %25 = dnn.relu %24 : (tensor<1x8x1x1xf32>) -> tensor<1x8x1x1xf32>
    %26 = torch_c.from_builtin_tensor %25 : tensor<1x8x1x1xf32> -> !torch.vtensor<[1,8,1,1],f32>
    %27 = torch_c.to_builtin_tensor %26 : !torch.vtensor<[1,8,1,1],f32> -> tensor<1x8x1x1xf32>
    %28 = dnn.convolution %27, %cst_52, %cst_56 {kind = "aten.conv2d", parameter_indices = array<i32: 3, 4, 5, 6>, parameters = [[1, 1], [0, 0], [1, 1], 1]} : (tensor<1x8x1x1xf32>, tensor<16x8x1x1xf32>, tensor<16xf32>) -> tensor<1x16x1x1xf32>
    %29 = torch_c.from_builtin_tensor %28 : tensor<1x16x1x1xf32> -> !torch.vtensor<[1,16,1,1],f32>
    %30 = torch_c.to_builtin_tensor %29 : !torch.vtensor<[1,16,1,1],f32> -> tensor<1x16x1x1xf32>
    %31 = dnn.hardsigmoid %30 : (tensor<1x16x1x1xf32>) -> tensor<1x16x1x1xf32>
    %32 = torch_c.from_builtin_tensor %31 : tensor<1x16x1x1xf32> -> !torch.vtensor<[1,16,1,1],f32>
    %33 = torch_c.to_builtin_tensor %32 : !torch.vtensor<[1,16,1,1],f32> -> tensor<1x16x1x1xf32>
    %34 = torch_c.to_builtin_tensor %17 : !torch.vtensor<[1,16,56,56],f32> -> tensor<1x16x56x56xf32>
    %35 = dnn.mul %33, %34 : (tensor<1x16x1x1xf32>, tensor<1x16x56x56xf32>) -> tensor<1x16x56x56xf32>
    %36 = torch_c.from_builtin_tensor %35 : tensor<1x16x56x56xf32> -> !torch.vtensor<[1,16,56,56],f32>
    %37 = torch_c.to_builtin_tensor %36 : !torch.vtensor<[1,16,56,56],f32> -> tensor<1x16x56x56xf32>
    %38 = dnn.convolution %37, %cst_51 {kind = "aten.conv2d", parameter_indices = array<i32: 2, 3, 4, 5, 6>, parameters = [unit, [1, 1], [0, 0], [1, 1], 1]} : (tensor<1x16x56x56xf32>, tensor<16x16x1x1xf32>) -> tensor<1x16x56x56xf32>
    %39 = torch_c.from_builtin_tensor %38 : tensor<1x16x56x56xf32> -> !torch.vtensor<[1,16,56,56],f32>
    %40 = torch_c.to_builtin_tensor %39 : !torch.vtensor<[1,16,56,56],f32> -> tensor<1x16x56x56xf32>
    %41:3 = dnn.batch_norm %40, %cst_56, %cst_56, %cst_56, %cst_56 {kind = "aten._native_batch_norm_legit_no_training", parameter_indices = array<i32: 5, 6>, parameters = [1.000000e-02, 1.000000e-03]} : (tensor<1x16x56x56xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>) -> (tensor<1x16x56x56xf32>, tensor<0xf32>, tensor<0xf32>)
    %42 = torch_c.from_builtin_tensor %41#0 : tensor<1x16x56x56xf32> -> !torch.vtensor<[1,16,56,56],f32>
    %43 = torch_c.to_builtin_tensor %42 : !torch.vtensor<[1,16,56,56],f32> -> tensor<1x16x56x56xf32>
    %44 = dnn.convolution %43, %cst_50 {kind = "aten.conv2d", parameter_indices = array<i32: 2, 3, 4, 5, 6>, parameters = [unit, [1, 1], [0, 0], [1, 1], 1]} : (tensor<1x16x56x56xf32>, tensor<72x16x1x1xf32>) -> tensor<1x72x56x56xf32>
    %45 = torch_c.from_builtin_tensor %44 : tensor<1x72x56x56xf32> -> !torch.vtensor<[1,72,56,56],f32>
    %46 = torch_c.to_builtin_tensor %45 : !torch.vtensor<[1,72,56,56],f32> -> tensor<1x72x56x56xf32>
    %47:3 = dnn.batch_norm %46, %cst_49, %cst_49, %cst_49, %cst_49 {kind = "aten._native_batch_norm_legit_no_training", parameter_indices = array<i32: 5, 6>, parameters = [1.000000e-02, 1.000000e-03]} : (tensor<1x72x56x56xf32>, tensor<72xf32>, tensor<72xf32>, tensor<72xf32>, tensor<72xf32>) -> (tensor<1x72x56x56xf32>, tensor<0xf32>, tensor<0xf32>)
    %48 = torch_c.from_builtin_tensor %47#0 : tensor<1x72x56x56xf32> -> !torch.vtensor<[1,72,56,56],f32>
    %49 = torch_c.to_builtin_tensor %48 : !torch.vtensor<[1,72,56,56],f32> -> tensor<1x72x56x56xf32>
    %50 = dnn.relu %49 : (tensor<1x72x56x56xf32>) -> tensor<1x72x56x56xf32>
    %51 = torch_c.from_builtin_tensor %50 : tensor<1x72x56x56xf32> -> !torch.vtensor<[1,72,56,56],f32>
    %52 = torch_c.to_builtin_tensor %51 : !torch.vtensor<[1,72,56,56],f32> -> tensor<1x72x56x56xf32>
    %53 = dnn.convolution %52, %cst_48 {kind = "aten.conv2d", parameter_indices = array<i32: 2, 3, 4, 5, 6>, parameters = [unit, [2, 2], [1, 1], [1, 1], 72]} : (tensor<1x72x56x56xf32>, tensor<72x1x3x3xf32>) -> tensor<1x72x28x28xf32>
    %54 = torch_c.from_builtin_tensor %53 : tensor<1x72x28x28xf32> -> !torch.vtensor<[1,72,28,28],f32>
    %55 = torch_c.to_builtin_tensor %54 : !torch.vtensor<[1,72,28,28],f32> -> tensor<1x72x28x28xf32>
    %56:3 = dnn.batch_norm %55, %cst_49, %cst_49, %cst_49, %cst_49 {kind = "aten._native_batch_norm_legit_no_training", parameter_indices = array<i32: 5, 6>, parameters = [1.000000e-02, 1.000000e-03]} : (tensor<1x72x28x28xf32>, tensor<72xf32>, tensor<72xf32>, tensor<72xf32>, tensor<72xf32>) -> (tensor<1x72x28x28xf32>, tensor<0xf32>, tensor<0xf32>)
    %57 = torch_c.from_builtin_tensor %56#0 : tensor<1x72x28x28xf32> -> !torch.vtensor<[1,72,28,28],f32>
    %58 = torch_c.to_builtin_tensor %57 : !torch.vtensor<[1,72,28,28],f32> -> tensor<1x72x28x28xf32>
    %59 = dnn.relu %58 : (tensor<1x72x28x28xf32>) -> tensor<1x72x28x28xf32>
    %60 = torch_c.from_builtin_tensor %59 : tensor<1x72x28x28xf32> -> !torch.vtensor<[1,72,28,28],f32>
    %61 = torch_c.to_builtin_tensor %60 : !torch.vtensor<[1,72,28,28],f32> -> tensor<1x72x28x28xf32>
    %62 = dnn.convolution %61, %cst_47 {kind = "aten.conv2d", parameter_indices = array<i32: 2, 3, 4, 5, 6>, parameters = [unit, [1, 1], [0, 0], [1, 1], 1]} : (tensor<1x72x28x28xf32>, tensor<24x72x1x1xf32>) -> tensor<1x24x28x28xf32>
    %63 = torch_c.from_builtin_tensor %62 : tensor<1x24x28x28xf32> -> !torch.vtensor<[1,24,28,28],f32>
    %64 = torch_c.to_builtin_tensor %63 : !torch.vtensor<[1,24,28,28],f32> -> tensor<1x24x28x28xf32>
    %65:3 = dnn.batch_norm %64, %cst_46, %cst_46, %cst_46, %cst_46 {kind = "aten._native_batch_norm_legit_no_training", parameter_indices = array<i32: 5, 6>, parameters = [1.000000e-02, 1.000000e-03]} : (tensor<1x24x28x28xf32>, tensor<24xf32>, tensor<24xf32>, tensor<24xf32>, tensor<24xf32>) -> (tensor<1x24x28x28xf32>, tensor<0xf32>, tensor<0xf32>)
    %66 = torch_c.from_builtin_tensor %65#0 : tensor<1x24x28x28xf32> -> !torch.vtensor<[1,24,28,28],f32>
    %67 = torch_c.to_builtin_tensor %66 : !torch.vtensor<[1,24,28,28],f32> -> tensor<1x24x28x28xf32>
    %68 = dnn.convolution %67, %cst_45 {kind = "aten.conv2d", parameter_indices = array<i32: 2, 3, 4, 5, 6>, parameters = [unit, [1, 1], [0, 0], [1, 1], 1]} : (tensor<1x24x28x28xf32>, tensor<88x24x1x1xf32>) -> tensor<1x88x28x28xf32>
    %69 = torch_c.from_builtin_tensor %68 : tensor<1x88x28x28xf32> -> !torch.vtensor<[1,88,28,28],f32>
    %70 = torch_c.to_builtin_tensor %69 : !torch.vtensor<[1,88,28,28],f32> -> tensor<1x88x28x28xf32>
    %71:3 = dnn.batch_norm %70, %cst_44, %cst_44, %cst_44, %cst_44 {kind = "aten._native_batch_norm_legit_no_training", parameter_indices = array<i32: 5, 6>, parameters = [1.000000e-02, 1.000000e-03]} : (tensor<1x88x28x28xf32>, tensor<88xf32>, tensor<88xf32>, tensor<88xf32>, tensor<88xf32>) -> (tensor<1x88x28x28xf32>, tensor<0xf32>, tensor<0xf32>)
    %72 = torch_c.from_builtin_tensor %71#0 : tensor<1x88x28x28xf32> -> !torch.vtensor<[1,88,28,28],f32>
    %73 = torch_c.to_builtin_tensor %72 : !torch.vtensor<[1,88,28,28],f32> -> tensor<1x88x28x28xf32>
    %74 = dnn.relu %73 : (tensor<1x88x28x28xf32>) -> tensor<1x88x28x28xf32>
    %75 = torch_c.from_builtin_tensor %74 : tensor<1x88x28x28xf32> -> !torch.vtensor<[1,88,28,28],f32>
    %76 = torch_c.to_builtin_tensor %75 : !torch.vtensor<[1,88,28,28],f32> -> tensor<1x88x28x28xf32>
    %77 = dnn.convolution %76, %cst_43 {kind = "aten.conv2d", parameter_indices = array<i32: 2, 3, 4, 5, 6>, parameters = [unit, [1, 1], [1, 1], [1, 1], 88]} : (tensor<1x88x28x28xf32>, tensor<88x1x3x3xf32>) -> tensor<1x88x28x28xf32>
    %78 = torch_c.from_builtin_tensor %77 : tensor<1x88x28x28xf32> -> !torch.vtensor<[1,88,28,28],f32>
    %79 = torch_c.to_builtin_tensor %78 : !torch.vtensor<[1,88,28,28],f32> -> tensor<1x88x28x28xf32>
    %80:3 = dnn.batch_norm %79, %cst_44, %cst_44, %cst_44, %cst_44 {kind = "aten._native_batch_norm_legit_no_training", parameter_indices = array<i32: 5, 6>, parameters = [1.000000e-02, 1.000000e-03]} : (tensor<1x88x28x28xf32>, tensor<88xf32>, tensor<88xf32>, tensor<88xf32>, tensor<88xf32>) -> (tensor<1x88x28x28xf32>, tensor<0xf32>, tensor<0xf32>)
    %81 = torch_c.from_builtin_tensor %80#0 : tensor<1x88x28x28xf32> -> !torch.vtensor<[1,88,28,28],f32>
    %82 = torch_c.to_builtin_tensor %81 : !torch.vtensor<[1,88,28,28],f32> -> tensor<1x88x28x28xf32>
    %83 = dnn.relu %82 : (tensor<1x88x28x28xf32>) -> tensor<1x88x28x28xf32>
    %84 = torch_c.from_builtin_tensor %83 : tensor<1x88x28x28xf32> -> !torch.vtensor<[1,88,28,28],f32>
    %85 = torch_c.to_builtin_tensor %84 : !torch.vtensor<[1,88,28,28],f32> -> tensor<1x88x28x28xf32>
    %86 = dnn.convolution %85, %cst_42 {kind = "aten.conv2d", parameter_indices = array<i32: 2, 3, 4, 5, 6>, parameters = [unit, [1, 1], [0, 0], [1, 1], 1]} : (tensor<1x88x28x28xf32>, tensor<24x88x1x1xf32>) -> tensor<1x24x28x28xf32>
    %87 = torch_c.from_builtin_tensor %86 : tensor<1x24x28x28xf32> -> !torch.vtensor<[1,24,28,28],f32>
    %88 = torch_c.to_builtin_tensor %87 : !torch.vtensor<[1,24,28,28],f32> -> tensor<1x24x28x28xf32>
    %89:3 = dnn.batch_norm %88, %cst_46, %cst_46, %cst_46, %cst_46 {kind = "aten._native_batch_norm_legit_no_training", parameter_indices = array<i32: 5, 6>, parameters = [1.000000e-02, 1.000000e-03]} : (tensor<1x24x28x28xf32>, tensor<24xf32>, tensor<24xf32>, tensor<24xf32>, tensor<24xf32>) -> (tensor<1x24x28x28xf32>, tensor<0xf32>, tensor<0xf32>)
    %90 = torch_c.from_builtin_tensor %89#0 : tensor<1x24x28x28xf32> -> !torch.vtensor<[1,24,28,28],f32>
    %91 = torch_c.to_builtin_tensor %90 : !torch.vtensor<[1,24,28,28],f32> -> tensor<1x24x28x28xf32>
    %92 = torch_c.to_builtin_tensor %66 : !torch.vtensor<[1,24,28,28],f32> -> tensor<1x24x28x28xf32>
    %93 = dnn.add %91, %92 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<1x24x28x28xf32>, tensor<1x24x28x28xf32>) -> tensor<1x24x28x28xf32>
    %94 = torch_c.from_builtin_tensor %93 : tensor<1x24x28x28xf32> -> !torch.vtensor<[1,24,28,28],f32>
    %95 = torch_c.to_builtin_tensor %94 : !torch.vtensor<[1,24,28,28],f32> -> tensor<1x24x28x28xf32>
    %96 = dnn.convolution %95, %cst_41 {kind = "aten.conv2d", parameter_indices = array<i32: 2, 3, 4, 5, 6>, parameters = [unit, [1, 1], [0, 0], [1, 1], 1]} : (tensor<1x24x28x28xf32>, tensor<96x24x1x1xf32>) -> tensor<1x96x28x28xf32>
    %97 = torch_c.from_builtin_tensor %96 : tensor<1x96x28x28xf32> -> !torch.vtensor<[1,96,28,28],f32>
    %98 = torch_c.to_builtin_tensor %97 : !torch.vtensor<[1,96,28,28],f32> -> tensor<1x96x28x28xf32>
    %99:3 = dnn.batch_norm %98, %cst_40, %cst_40, %cst_40, %cst_40 {kind = "aten._native_batch_norm_legit_no_training", parameter_indices = array<i32: 5, 6>, parameters = [1.000000e-02, 1.000000e-03]} : (tensor<1x96x28x28xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>) -> (tensor<1x96x28x28xf32>, tensor<0xf32>, tensor<0xf32>)
    %100 = torch_c.from_builtin_tensor %99#0 : tensor<1x96x28x28xf32> -> !torch.vtensor<[1,96,28,28],f32>
    %101 = torch_c.to_builtin_tensor %100 : !torch.vtensor<[1,96,28,28],f32> -> tensor<1x96x28x28xf32>
    %102 = dnn.hardswish %101 : (tensor<1x96x28x28xf32>) -> tensor<1x96x28x28xf32>
    %103 = torch_c.from_builtin_tensor %102 : tensor<1x96x28x28xf32> -> !torch.vtensor<[1,96,28,28],f32>
    %104 = torch_c.to_builtin_tensor %103 : !torch.vtensor<[1,96,28,28],f32> -> tensor<1x96x28x28xf32>
    %105 = dnn.convolution %104, %cst_39 {kind = "aten.conv2d", parameter_indices = array<i32: 2, 3, 4, 5, 6>, parameters = [unit, [2, 2], [2, 2], [1, 1], 96]} : (tensor<1x96x28x28xf32>, tensor<96x1x5x5xf32>) -> tensor<1x96x14x14xf32>
    %106 = torch_c.from_builtin_tensor %105 : tensor<1x96x14x14xf32> -> !torch.vtensor<[1,96,14,14],f32>
    %107 = torch_c.to_builtin_tensor %106 : !torch.vtensor<[1,96,14,14],f32> -> tensor<1x96x14x14xf32>
    %108:3 = dnn.batch_norm %107, %cst_40, %cst_40, %cst_40, %cst_40 {kind = "aten._native_batch_norm_legit_no_training", parameter_indices = array<i32: 5, 6>, parameters = [1.000000e-02, 1.000000e-03]} : (tensor<1x96x14x14xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>) -> (tensor<1x96x14x14xf32>, tensor<0xf32>, tensor<0xf32>)
    %109 = torch_c.from_builtin_tensor %108#0 : tensor<1x96x14x14xf32> -> !torch.vtensor<[1,96,14,14],f32>
    %110 = torch_c.to_builtin_tensor %109 : !torch.vtensor<[1,96,14,14],f32> -> tensor<1x96x14x14xf32>
    %111 = dnn.hardswish %110 : (tensor<1x96x14x14xf32>) -> tensor<1x96x14x14xf32>
    %112 = torch_c.from_builtin_tensor %111 : tensor<1x96x14x14xf32> -> !torch.vtensor<[1,96,14,14],f32>
    %113 = torch_c.to_builtin_tensor %112 : !torch.vtensor<[1,96,14,14],f32> -> tensor<1x96x14x14xf32>
    %114 = dnn.adaptive_avg_pool2d %113 {parameter_indices = array<i32: 1>, parameters = [[1, 1]]} : (tensor<1x96x14x14xf32>) -> tensor<1x96x1x1xf32>
    %115 = torch_c.from_builtin_tensor %114 : tensor<1x96x1x1xf32> -> !torch.vtensor<[1,96,1,1],f32>
    %116 = torch_c.to_builtin_tensor %115 : !torch.vtensor<[1,96,1,1],f32> -> tensor<1x96x1x1xf32>
    %117 = dnn.convolution %116, %cst_38, %cst_46 {kind = "aten.conv2d", parameter_indices = array<i32: 3, 4, 5, 6>, parameters = [[1, 1], [0, 0], [1, 1], 1]} : (tensor<1x96x1x1xf32>, tensor<24x96x1x1xf32>, tensor<24xf32>) -> tensor<1x24x1x1xf32>
    %118 = torch_c.from_builtin_tensor %117 : tensor<1x24x1x1xf32> -> !torch.vtensor<[1,24,1,1],f32>
    %119 = torch_c.to_builtin_tensor %118 : !torch.vtensor<[1,24,1,1],f32> -> tensor<1x24x1x1xf32>
    %120 = dnn.relu %119 : (tensor<1x24x1x1xf32>) -> tensor<1x24x1x1xf32>
    %121 = torch_c.from_builtin_tensor %120 : tensor<1x24x1x1xf32> -> !torch.vtensor<[1,24,1,1],f32>
    %122 = torch_c.to_builtin_tensor %121 : !torch.vtensor<[1,24,1,1],f32> -> tensor<1x24x1x1xf32>
    %123 = dnn.convolution %122, %cst_41, %cst_40 {kind = "aten.conv2d", parameter_indices = array<i32: 3, 4, 5, 6>, parameters = [[1, 1], [0, 0], [1, 1], 1]} : (tensor<1x24x1x1xf32>, tensor<96x24x1x1xf32>, tensor<96xf32>) -> tensor<1x96x1x1xf32>
    %124 = torch_c.from_builtin_tensor %123 : tensor<1x96x1x1xf32> -> !torch.vtensor<[1,96,1,1],f32>
    %125 = torch_c.to_builtin_tensor %124 : !torch.vtensor<[1,96,1,1],f32> -> tensor<1x96x1x1xf32>
    %126 = dnn.hardsigmoid %125 : (tensor<1x96x1x1xf32>) -> tensor<1x96x1x1xf32>
    %127 = torch_c.from_builtin_tensor %126 : tensor<1x96x1x1xf32> -> !torch.vtensor<[1,96,1,1],f32>
    %128 = torch_c.to_builtin_tensor %127 : !torch.vtensor<[1,96,1,1],f32> -> tensor<1x96x1x1xf32>
    %129 = torch_c.to_builtin_tensor %112 : !torch.vtensor<[1,96,14,14],f32> -> tensor<1x96x14x14xf32>
    %130 = dnn.mul %128, %129 : (tensor<1x96x1x1xf32>, tensor<1x96x14x14xf32>) -> tensor<1x96x14x14xf32>
    %131 = torch_c.from_builtin_tensor %130 : tensor<1x96x14x14xf32> -> !torch.vtensor<[1,96,14,14],f32>
    %132 = torch_c.to_builtin_tensor %131 : !torch.vtensor<[1,96,14,14],f32> -> tensor<1x96x14x14xf32>
    %133 = dnn.convolution %132, %cst_37 {kind = "aten.conv2d", parameter_indices = array<i32: 2, 3, 4, 5, 6>, parameters = [unit, [1, 1], [0, 0], [1, 1], 1]} : (tensor<1x96x14x14xf32>, tensor<40x96x1x1xf32>) -> tensor<1x40x14x14xf32>
    %134 = torch_c.from_builtin_tensor %133 : tensor<1x40x14x14xf32> -> !torch.vtensor<[1,40,14,14],f32>
    %135 = torch_c.to_builtin_tensor %134 : !torch.vtensor<[1,40,14,14],f32> -> tensor<1x40x14x14xf32>
    %136:3 = dnn.batch_norm %135, %cst_36, %cst_36, %cst_36, %cst_36 {kind = "aten._native_batch_norm_legit_no_training", parameter_indices = array<i32: 5, 6>, parameters = [1.000000e-02, 1.000000e-03]} : (tensor<1x40x14x14xf32>, tensor<40xf32>, tensor<40xf32>, tensor<40xf32>, tensor<40xf32>) -> (tensor<1x40x14x14xf32>, tensor<0xf32>, tensor<0xf32>)
    %137 = torch_c.from_builtin_tensor %136#0 : tensor<1x40x14x14xf32> -> !torch.vtensor<[1,40,14,14],f32>
    %138 = torch_c.to_builtin_tensor %137 : !torch.vtensor<[1,40,14,14],f32> -> tensor<1x40x14x14xf32>
    %139 = dnn.convolution %138, %cst_35 {kind = "aten.conv2d", parameter_indices = array<i32: 2, 3, 4, 5, 6>, parameters = [unit, [1, 1], [0, 0], [1, 1], 1]} : (tensor<1x40x14x14xf32>, tensor<240x40x1x1xf32>) -> tensor<1x240x14x14xf32>
    %140 = torch_c.from_builtin_tensor %139 : tensor<1x240x14x14xf32> -> !torch.vtensor<[1,240,14,14],f32>
    %141 = torch_c.to_builtin_tensor %140 : !torch.vtensor<[1,240,14,14],f32> -> tensor<1x240x14x14xf32>
    %142:3 = dnn.batch_norm %141, %cst_34, %cst_34, %cst_34, %cst_34 {kind = "aten._native_batch_norm_legit_no_training", parameter_indices = array<i32: 5, 6>, parameters = [1.000000e-02, 1.000000e-03]} : (tensor<1x240x14x14xf32>, tensor<240xf32>, tensor<240xf32>, tensor<240xf32>, tensor<240xf32>) -> (tensor<1x240x14x14xf32>, tensor<0xf32>, tensor<0xf32>)
    %143 = torch_c.from_builtin_tensor %142#0 : tensor<1x240x14x14xf32> -> !torch.vtensor<[1,240,14,14],f32>
    %144 = torch_c.to_builtin_tensor %143 : !torch.vtensor<[1,240,14,14],f32> -> tensor<1x240x14x14xf32>
    %145 = dnn.hardswish %144 : (tensor<1x240x14x14xf32>) -> tensor<1x240x14x14xf32>
    %146 = torch_c.from_builtin_tensor %145 : tensor<1x240x14x14xf32> -> !torch.vtensor<[1,240,14,14],f32>
    %147 = torch_c.to_builtin_tensor %146 : !torch.vtensor<[1,240,14,14],f32> -> tensor<1x240x14x14xf32>
    %148 = dnn.convolution %147, %cst_33 {kind = "aten.conv2d", parameter_indices = array<i32: 2, 3, 4, 5, 6>, parameters = [unit, [1, 1], [2, 2], [1, 1], 240]} : (tensor<1x240x14x14xf32>, tensor<240x1x5x5xf32>) -> tensor<1x240x14x14xf32>
    %149 = torch_c.from_builtin_tensor %148 : tensor<1x240x14x14xf32> -> !torch.vtensor<[1,240,14,14],f32>
    %150 = torch_c.to_builtin_tensor %149 : !torch.vtensor<[1,240,14,14],f32> -> tensor<1x240x14x14xf32>
    %151:3 = dnn.batch_norm %150, %cst_34, %cst_34, %cst_34, %cst_34 {kind = "aten._native_batch_norm_legit_no_training", parameter_indices = array<i32: 5, 6>, parameters = [1.000000e-02, 1.000000e-03]} : (tensor<1x240x14x14xf32>, tensor<240xf32>, tensor<240xf32>, tensor<240xf32>, tensor<240xf32>) -> (tensor<1x240x14x14xf32>, tensor<0xf32>, tensor<0xf32>)
    %152 = torch_c.from_builtin_tensor %151#0 : tensor<1x240x14x14xf32> -> !torch.vtensor<[1,240,14,14],f32>
    %153 = torch_c.to_builtin_tensor %152 : !torch.vtensor<[1,240,14,14],f32> -> tensor<1x240x14x14xf32>
    %154 = dnn.hardswish %153 : (tensor<1x240x14x14xf32>) -> tensor<1x240x14x14xf32>
    %155 = torch_c.from_builtin_tensor %154 : tensor<1x240x14x14xf32> -> !torch.vtensor<[1,240,14,14],f32>
    %156 = torch_c.to_builtin_tensor %155 : !torch.vtensor<[1,240,14,14],f32> -> tensor<1x240x14x14xf32>
    %157 = dnn.adaptive_avg_pool2d %156 {parameter_indices = array<i32: 1>, parameters = [[1, 1]]} : (tensor<1x240x14x14xf32>) -> tensor<1x240x1x1xf32>
    %158 = torch_c.from_builtin_tensor %157 : tensor<1x240x1x1xf32> -> !torch.vtensor<[1,240,1,1],f32>
    %159 = torch_c.to_builtin_tensor %158 : !torch.vtensor<[1,240,1,1],f32> -> tensor<1x240x1x1xf32>
    %160 = dnn.convolution %159, %cst_32, %cst_31 {kind = "aten.conv2d", parameter_indices = array<i32: 3, 4, 5, 6>, parameters = [[1, 1], [0, 0], [1, 1], 1]} : (tensor<1x240x1x1xf32>, tensor<64x240x1x1xf32>, tensor<64xf32>) -> tensor<1x64x1x1xf32>
    %161 = torch_c.from_builtin_tensor %160 : tensor<1x64x1x1xf32> -> !torch.vtensor<[1,64,1,1],f32>
    %162 = torch_c.to_builtin_tensor %161 : !torch.vtensor<[1,64,1,1],f32> -> tensor<1x64x1x1xf32>
    %163 = dnn.relu %162 : (tensor<1x64x1x1xf32>) -> tensor<1x64x1x1xf32>
    %164 = torch_c.from_builtin_tensor %163 : tensor<1x64x1x1xf32> -> !torch.vtensor<[1,64,1,1],f32>
    %165 = torch_c.to_builtin_tensor %164 : !torch.vtensor<[1,64,1,1],f32> -> tensor<1x64x1x1xf32>
    %166 = dnn.convolution %165, %cst_30, %cst_34 {kind = "aten.conv2d", parameter_indices = array<i32: 3, 4, 5, 6>, parameters = [[1, 1], [0, 0], [1, 1], 1]} : (tensor<1x64x1x1xf32>, tensor<240x64x1x1xf32>, tensor<240xf32>) -> tensor<1x240x1x1xf32>
    %167 = torch_c.from_builtin_tensor %166 : tensor<1x240x1x1xf32> -> !torch.vtensor<[1,240,1,1],f32>
    %168 = torch_c.to_builtin_tensor %167 : !torch.vtensor<[1,240,1,1],f32> -> tensor<1x240x1x1xf32>
    %169 = dnn.hardsigmoid %168 : (tensor<1x240x1x1xf32>) -> tensor<1x240x1x1xf32>
    %170 = torch_c.from_builtin_tensor %169 : tensor<1x240x1x1xf32> -> !torch.vtensor<[1,240,1,1],f32>
    %171 = torch_c.to_builtin_tensor %170 : !torch.vtensor<[1,240,1,1],f32> -> tensor<1x240x1x1xf32>
    %172 = torch_c.to_builtin_tensor %155 : !torch.vtensor<[1,240,14,14],f32> -> tensor<1x240x14x14xf32>
    %173 = dnn.mul %171, %172 : (tensor<1x240x1x1xf32>, tensor<1x240x14x14xf32>) -> tensor<1x240x14x14xf32>
    %174 = torch_c.from_builtin_tensor %173 : tensor<1x240x14x14xf32> -> !torch.vtensor<[1,240,14,14],f32>
    %175 = torch_c.to_builtin_tensor %174 : !torch.vtensor<[1,240,14,14],f32> -> tensor<1x240x14x14xf32>
    %176 = dnn.convolution %175, %cst_29 {kind = "aten.conv2d", parameter_indices = array<i32: 2, 3, 4, 5, 6>, parameters = [unit, [1, 1], [0, 0], [1, 1], 1]} : (tensor<1x240x14x14xf32>, tensor<40x240x1x1xf32>) -> tensor<1x40x14x14xf32>
    %177 = torch_c.from_builtin_tensor %176 : tensor<1x40x14x14xf32> -> !torch.vtensor<[1,40,14,14],f32>
    %178 = torch_c.to_builtin_tensor %177 : !torch.vtensor<[1,40,14,14],f32> -> tensor<1x40x14x14xf32>
    %179:3 = dnn.batch_norm %178, %cst_36, %cst_36, %cst_36, %cst_36 {kind = "aten._native_batch_norm_legit_no_training", parameter_indices = array<i32: 5, 6>, parameters = [1.000000e-02, 1.000000e-03]} : (tensor<1x40x14x14xf32>, tensor<40xf32>, tensor<40xf32>, tensor<40xf32>, tensor<40xf32>) -> (tensor<1x40x14x14xf32>, tensor<0xf32>, tensor<0xf32>)
    %180 = torch_c.from_builtin_tensor %179#0 : tensor<1x40x14x14xf32> -> !torch.vtensor<[1,40,14,14],f32>
    %181 = torch_c.to_builtin_tensor %180 : !torch.vtensor<[1,40,14,14],f32> -> tensor<1x40x14x14xf32>
    %182 = torch_c.to_builtin_tensor %137 : !torch.vtensor<[1,40,14,14],f32> -> tensor<1x40x14x14xf32>
    %183 = dnn.add %181, %182 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<1x40x14x14xf32>, tensor<1x40x14x14xf32>) -> tensor<1x40x14x14xf32>
    %184 = torch_c.from_builtin_tensor %183 : tensor<1x40x14x14xf32> -> !torch.vtensor<[1,40,14,14],f32>
    %185 = torch_c.to_builtin_tensor %184 : !torch.vtensor<[1,40,14,14],f32> -> tensor<1x40x14x14xf32>
    %186 = dnn.convolution %185, %cst_35 {kind = "aten.conv2d", parameter_indices = array<i32: 2, 3, 4, 5, 6>, parameters = [unit, [1, 1], [0, 0], [1, 1], 1]} : (tensor<1x40x14x14xf32>, tensor<240x40x1x1xf32>) -> tensor<1x240x14x14xf32>
    %187 = torch_c.from_builtin_tensor %186 : tensor<1x240x14x14xf32> -> !torch.vtensor<[1,240,14,14],f32>
    %188 = torch_c.to_builtin_tensor %187 : !torch.vtensor<[1,240,14,14],f32> -> tensor<1x240x14x14xf32>
    %189:3 = dnn.batch_norm %188, %cst_34, %cst_34, %cst_34, %cst_34 {kind = "aten._native_batch_norm_legit_no_training", parameter_indices = array<i32: 5, 6>, parameters = [1.000000e-02, 1.000000e-03]} : (tensor<1x240x14x14xf32>, tensor<240xf32>, tensor<240xf32>, tensor<240xf32>, tensor<240xf32>) -> (tensor<1x240x14x14xf32>, tensor<0xf32>, tensor<0xf32>)
    %190 = torch_c.from_builtin_tensor %189#0 : tensor<1x240x14x14xf32> -> !torch.vtensor<[1,240,14,14],f32>
    %191 = torch_c.to_builtin_tensor %190 : !torch.vtensor<[1,240,14,14],f32> -> tensor<1x240x14x14xf32>
    %192 = dnn.hardswish %191 : (tensor<1x240x14x14xf32>) -> tensor<1x240x14x14xf32>
    %193 = torch_c.from_builtin_tensor %192 : tensor<1x240x14x14xf32> -> !torch.vtensor<[1,240,14,14],f32>
    %194 = torch_c.to_builtin_tensor %193 : !torch.vtensor<[1,240,14,14],f32> -> tensor<1x240x14x14xf32>
    %195 = dnn.convolution %194, %cst_33 {kind = "aten.conv2d", parameter_indices = array<i32: 2, 3, 4, 5, 6>, parameters = [unit, [1, 1], [2, 2], [1, 1], 240]} : (tensor<1x240x14x14xf32>, tensor<240x1x5x5xf32>) -> tensor<1x240x14x14xf32>
    %196 = torch_c.from_builtin_tensor %195 : tensor<1x240x14x14xf32> -> !torch.vtensor<[1,240,14,14],f32>
    %197 = torch_c.to_builtin_tensor %196 : !torch.vtensor<[1,240,14,14],f32> -> tensor<1x240x14x14xf32>
    %198:3 = dnn.batch_norm %197, %cst_34, %cst_34, %cst_34, %cst_34 {kind = "aten._native_batch_norm_legit_no_training", parameter_indices = array<i32: 5, 6>, parameters = [1.000000e-02, 1.000000e-03]} : (tensor<1x240x14x14xf32>, tensor<240xf32>, tensor<240xf32>, tensor<240xf32>, tensor<240xf32>) -> (tensor<1x240x14x14xf32>, tensor<0xf32>, tensor<0xf32>)
    %199 = torch_c.from_builtin_tensor %198#0 : tensor<1x240x14x14xf32> -> !torch.vtensor<[1,240,14,14],f32>
    %200 = torch_c.to_builtin_tensor %199 : !torch.vtensor<[1,240,14,14],f32> -> tensor<1x240x14x14xf32>
    %201 = dnn.hardswish %200 : (tensor<1x240x14x14xf32>) -> tensor<1x240x14x14xf32>
    %202 = torch_c.from_builtin_tensor %201 : tensor<1x240x14x14xf32> -> !torch.vtensor<[1,240,14,14],f32>
    %203 = torch_c.to_builtin_tensor %202 : !torch.vtensor<[1,240,14,14],f32> -> tensor<1x240x14x14xf32>
    %204 = dnn.adaptive_avg_pool2d %203 {parameter_indices = array<i32: 1>, parameters = [[1, 1]]} : (tensor<1x240x14x14xf32>) -> tensor<1x240x1x1xf32>
    %205 = torch_c.from_builtin_tensor %204 : tensor<1x240x1x1xf32> -> !torch.vtensor<[1,240,1,1],f32>
    %206 = torch_c.to_builtin_tensor %205 : !torch.vtensor<[1,240,1,1],f32> -> tensor<1x240x1x1xf32>
    %207 = dnn.convolution %206, %cst_32, %cst_31 {kind = "aten.conv2d", parameter_indices = array<i32: 3, 4, 5, 6>, parameters = [[1, 1], [0, 0], [1, 1], 1]} : (tensor<1x240x1x1xf32>, tensor<64x240x1x1xf32>, tensor<64xf32>) -> tensor<1x64x1x1xf32>
    %208 = torch_c.from_builtin_tensor %207 : tensor<1x64x1x1xf32> -> !torch.vtensor<[1,64,1,1],f32>
    %209 = torch_c.to_builtin_tensor %208 : !torch.vtensor<[1,64,1,1],f32> -> tensor<1x64x1x1xf32>
    %210 = dnn.relu %209 : (tensor<1x64x1x1xf32>) -> tensor<1x64x1x1xf32>
    %211 = torch_c.from_builtin_tensor %210 : tensor<1x64x1x1xf32> -> !torch.vtensor<[1,64,1,1],f32>
    %212 = torch_c.to_builtin_tensor %211 : !torch.vtensor<[1,64,1,1],f32> -> tensor<1x64x1x1xf32>
    %213 = dnn.convolution %212, %cst_30, %cst_34 {kind = "aten.conv2d", parameter_indices = array<i32: 3, 4, 5, 6>, parameters = [[1, 1], [0, 0], [1, 1], 1]} : (tensor<1x64x1x1xf32>, tensor<240x64x1x1xf32>, tensor<240xf32>) -> tensor<1x240x1x1xf32>
    %214 = torch_c.from_builtin_tensor %213 : tensor<1x240x1x1xf32> -> !torch.vtensor<[1,240,1,1],f32>
    %215 = torch_c.to_builtin_tensor %214 : !torch.vtensor<[1,240,1,1],f32> -> tensor<1x240x1x1xf32>
    %216 = dnn.hardsigmoid %215 : (tensor<1x240x1x1xf32>) -> tensor<1x240x1x1xf32>
    %217 = torch_c.from_builtin_tensor %216 : tensor<1x240x1x1xf32> -> !torch.vtensor<[1,240,1,1],f32>
    %218 = torch_c.to_builtin_tensor %217 : !torch.vtensor<[1,240,1,1],f32> -> tensor<1x240x1x1xf32>
    %219 = torch_c.to_builtin_tensor %202 : !torch.vtensor<[1,240,14,14],f32> -> tensor<1x240x14x14xf32>
    %220 = dnn.mul %218, %219 : (tensor<1x240x1x1xf32>, tensor<1x240x14x14xf32>) -> tensor<1x240x14x14xf32>
    %221 = torch_c.from_builtin_tensor %220 : tensor<1x240x14x14xf32> -> !torch.vtensor<[1,240,14,14],f32>
    %222 = torch_c.to_builtin_tensor %221 : !torch.vtensor<[1,240,14,14],f32> -> tensor<1x240x14x14xf32>
    %223 = dnn.convolution %222, %cst_29 {kind = "aten.conv2d", parameter_indices = array<i32: 2, 3, 4, 5, 6>, parameters = [unit, [1, 1], [0, 0], [1, 1], 1]} : (tensor<1x240x14x14xf32>, tensor<40x240x1x1xf32>) -> tensor<1x40x14x14xf32>
    %224 = torch_c.from_builtin_tensor %223 : tensor<1x40x14x14xf32> -> !torch.vtensor<[1,40,14,14],f32>
    %225 = torch_c.to_builtin_tensor %224 : !torch.vtensor<[1,40,14,14],f32> -> tensor<1x40x14x14xf32>
    %226:3 = dnn.batch_norm %225, %cst_36, %cst_36, %cst_36, %cst_36 {kind = "aten._native_batch_norm_legit_no_training", parameter_indices = array<i32: 5, 6>, parameters = [1.000000e-02, 1.000000e-03]} : (tensor<1x40x14x14xf32>, tensor<40xf32>, tensor<40xf32>, tensor<40xf32>, tensor<40xf32>) -> (tensor<1x40x14x14xf32>, tensor<0xf32>, tensor<0xf32>)
    %227 = torch_c.from_builtin_tensor %226#0 : tensor<1x40x14x14xf32> -> !torch.vtensor<[1,40,14,14],f32>
    %228 = torch_c.to_builtin_tensor %227 : !torch.vtensor<[1,40,14,14],f32> -> tensor<1x40x14x14xf32>
    %229 = torch_c.to_builtin_tensor %184 : !torch.vtensor<[1,40,14,14],f32> -> tensor<1x40x14x14xf32>
    %230 = dnn.add %228, %229 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<1x40x14x14xf32>, tensor<1x40x14x14xf32>) -> tensor<1x40x14x14xf32>
    %231 = torch_c.from_builtin_tensor %230 : tensor<1x40x14x14xf32> -> !torch.vtensor<[1,40,14,14],f32>
    %232 = torch_c.to_builtin_tensor %231 : !torch.vtensor<[1,40,14,14],f32> -> tensor<1x40x14x14xf32>
    %233 = dnn.convolution %232, %cst_28 {kind = "aten.conv2d", parameter_indices = array<i32: 2, 3, 4, 5, 6>, parameters = [unit, [1, 1], [0, 0], [1, 1], 1]} : (tensor<1x40x14x14xf32>, tensor<120x40x1x1xf32>) -> tensor<1x120x14x14xf32>
    %234 = torch_c.from_builtin_tensor %233 : tensor<1x120x14x14xf32> -> !torch.vtensor<[1,120,14,14],f32>
    %235 = torch_c.to_builtin_tensor %234 : !torch.vtensor<[1,120,14,14],f32> -> tensor<1x120x14x14xf32>
    %236:3 = dnn.batch_norm %235, %cst_27, %cst_27, %cst_27, %cst_27 {kind = "aten._native_batch_norm_legit_no_training", parameter_indices = array<i32: 5, 6>, parameters = [1.000000e-02, 1.000000e-03]} : (tensor<1x120x14x14xf32>, tensor<120xf32>, tensor<120xf32>, tensor<120xf32>, tensor<120xf32>) -> (tensor<1x120x14x14xf32>, tensor<0xf32>, tensor<0xf32>)
    %237 = torch_c.from_builtin_tensor %236#0 : tensor<1x120x14x14xf32> -> !torch.vtensor<[1,120,14,14],f32>
    %238 = torch_c.to_builtin_tensor %237 : !torch.vtensor<[1,120,14,14],f32> -> tensor<1x120x14x14xf32>
    %239 = dnn.hardswish %238 : (tensor<1x120x14x14xf32>) -> tensor<1x120x14x14xf32>
    %240 = torch_c.from_builtin_tensor %239 : tensor<1x120x14x14xf32> -> !torch.vtensor<[1,120,14,14],f32>
    %241 = torch_c.to_builtin_tensor %240 : !torch.vtensor<[1,120,14,14],f32> -> tensor<1x120x14x14xf32>
    %242 = dnn.convolution %241, %cst_26 {kind = "aten.conv2d", parameter_indices = array<i32: 2, 3, 4, 5, 6>, parameters = [unit, [1, 1], [2, 2], [1, 1], 120]} : (tensor<1x120x14x14xf32>, tensor<120x1x5x5xf32>) -> tensor<1x120x14x14xf32>
    %243 = torch_c.from_builtin_tensor %242 : tensor<1x120x14x14xf32> -> !torch.vtensor<[1,120,14,14],f32>
    %244 = torch_c.to_builtin_tensor %243 : !torch.vtensor<[1,120,14,14],f32> -> tensor<1x120x14x14xf32>
    %245:3 = dnn.batch_norm %244, %cst_27, %cst_27, %cst_27, %cst_27 {kind = "aten._native_batch_norm_legit_no_training", parameter_indices = array<i32: 5, 6>, parameters = [1.000000e-02, 1.000000e-03]} : (tensor<1x120x14x14xf32>, tensor<120xf32>, tensor<120xf32>, tensor<120xf32>, tensor<120xf32>) -> (tensor<1x120x14x14xf32>, tensor<0xf32>, tensor<0xf32>)
    %246 = torch_c.from_builtin_tensor %245#0 : tensor<1x120x14x14xf32> -> !torch.vtensor<[1,120,14,14],f32>
    %247 = torch_c.to_builtin_tensor %246 : !torch.vtensor<[1,120,14,14],f32> -> tensor<1x120x14x14xf32>
    %248 = dnn.hardswish %247 : (tensor<1x120x14x14xf32>) -> tensor<1x120x14x14xf32>
    %249 = torch_c.from_builtin_tensor %248 : tensor<1x120x14x14xf32> -> !torch.vtensor<[1,120,14,14],f32>
    %250 = torch_c.to_builtin_tensor %249 : !torch.vtensor<[1,120,14,14],f32> -> tensor<1x120x14x14xf32>
    %251 = dnn.adaptive_avg_pool2d %250 {parameter_indices = array<i32: 1>, parameters = [[1, 1]]} : (tensor<1x120x14x14xf32>) -> tensor<1x120x1x1xf32>
    %252 = torch_c.from_builtin_tensor %251 : tensor<1x120x1x1xf32> -> !torch.vtensor<[1,120,1,1],f32>
    %253 = torch_c.to_builtin_tensor %252 : !torch.vtensor<[1,120,1,1],f32> -> tensor<1x120x1x1xf32>
    %254 = dnn.convolution %253, %cst_25, %cst_24 {kind = "aten.conv2d", parameter_indices = array<i32: 3, 4, 5, 6>, parameters = [[1, 1], [0, 0], [1, 1], 1]} : (tensor<1x120x1x1xf32>, tensor<32x120x1x1xf32>, tensor<32xf32>) -> tensor<1x32x1x1xf32>
    %255 = torch_c.from_builtin_tensor %254 : tensor<1x32x1x1xf32> -> !torch.vtensor<[1,32,1,1],f32>
    %256 = torch_c.to_builtin_tensor %255 : !torch.vtensor<[1,32,1,1],f32> -> tensor<1x32x1x1xf32>
    %257 = dnn.relu %256 : (tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %258 = torch_c.from_builtin_tensor %257 : tensor<1x32x1x1xf32> -> !torch.vtensor<[1,32,1,1],f32>
    %259 = torch_c.to_builtin_tensor %258 : !torch.vtensor<[1,32,1,1],f32> -> tensor<1x32x1x1xf32>
    %260 = dnn.convolution %259, %cst_23, %cst_27 {kind = "aten.conv2d", parameter_indices = array<i32: 3, 4, 5, 6>, parameters = [[1, 1], [0, 0], [1, 1], 1]} : (tensor<1x32x1x1xf32>, tensor<120x32x1x1xf32>, tensor<120xf32>) -> tensor<1x120x1x1xf32>
    %261 = torch_c.from_builtin_tensor %260 : tensor<1x120x1x1xf32> -> !torch.vtensor<[1,120,1,1],f32>
    %262 = torch_c.to_builtin_tensor %261 : !torch.vtensor<[1,120,1,1],f32> -> tensor<1x120x1x1xf32>
    %263 = dnn.hardsigmoid %262 : (tensor<1x120x1x1xf32>) -> tensor<1x120x1x1xf32>
    %264 = torch_c.from_builtin_tensor %263 : tensor<1x120x1x1xf32> -> !torch.vtensor<[1,120,1,1],f32>
    %265 = torch_c.to_builtin_tensor %264 : !torch.vtensor<[1,120,1,1],f32> -> tensor<1x120x1x1xf32>
    %266 = torch_c.to_builtin_tensor %249 : !torch.vtensor<[1,120,14,14],f32> -> tensor<1x120x14x14xf32>
    %267 = dnn.mul %265, %266 : (tensor<1x120x1x1xf32>, tensor<1x120x14x14xf32>) -> tensor<1x120x14x14xf32>
    %268 = torch_c.from_builtin_tensor %267 : tensor<1x120x14x14xf32> -> !torch.vtensor<[1,120,14,14],f32>
    %269 = torch_c.to_builtin_tensor %268 : !torch.vtensor<[1,120,14,14],f32> -> tensor<1x120x14x14xf32>
    %270 = dnn.convolution %269, %cst_22 {kind = "aten.conv2d", parameter_indices = array<i32: 2, 3, 4, 5, 6>, parameters = [unit, [1, 1], [0, 0], [1, 1], 1]} : (tensor<1x120x14x14xf32>, tensor<48x120x1x1xf32>) -> tensor<1x48x14x14xf32>
    %271 = torch_c.from_builtin_tensor %270 : tensor<1x48x14x14xf32> -> !torch.vtensor<[1,48,14,14],f32>
    %272 = torch_c.to_builtin_tensor %271 : !torch.vtensor<[1,48,14,14],f32> -> tensor<1x48x14x14xf32>
    %273:3 = dnn.batch_norm %272, %cst_21, %cst_21, %cst_21, %cst_21 {kind = "aten._native_batch_norm_legit_no_training", parameter_indices = array<i32: 5, 6>, parameters = [1.000000e-02, 1.000000e-03]} : (tensor<1x48x14x14xf32>, tensor<48xf32>, tensor<48xf32>, tensor<48xf32>, tensor<48xf32>) -> (tensor<1x48x14x14xf32>, tensor<0xf32>, tensor<0xf32>)
    %274 = torch_c.from_builtin_tensor %273#0 : tensor<1x48x14x14xf32> -> !torch.vtensor<[1,48,14,14],f32>
    %275 = torch_c.to_builtin_tensor %274 : !torch.vtensor<[1,48,14,14],f32> -> tensor<1x48x14x14xf32>
    %276 = dnn.convolution %275, %cst_20 {kind = "aten.conv2d", parameter_indices = array<i32: 2, 3, 4, 5, 6>, parameters = [unit, [1, 1], [0, 0], [1, 1], 1]} : (tensor<1x48x14x14xf32>, tensor<144x48x1x1xf32>) -> tensor<1x144x14x14xf32>
    %277 = torch_c.from_builtin_tensor %276 : tensor<1x144x14x14xf32> -> !torch.vtensor<[1,144,14,14],f32>
    %278 = torch_c.to_builtin_tensor %277 : !torch.vtensor<[1,144,14,14],f32> -> tensor<1x144x14x14xf32>
    %279:3 = dnn.batch_norm %278, %cst_19, %cst_19, %cst_19, %cst_19 {kind = "aten._native_batch_norm_legit_no_training", parameter_indices = array<i32: 5, 6>, parameters = [1.000000e-02, 1.000000e-03]} : (tensor<1x144x14x14xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144xf32>) -> (tensor<1x144x14x14xf32>, tensor<0xf32>, tensor<0xf32>)
    %280 = torch_c.from_builtin_tensor %279#0 : tensor<1x144x14x14xf32> -> !torch.vtensor<[1,144,14,14],f32>
    %281 = torch_c.to_builtin_tensor %280 : !torch.vtensor<[1,144,14,14],f32> -> tensor<1x144x14x14xf32>
    %282 = dnn.hardswish %281 : (tensor<1x144x14x14xf32>) -> tensor<1x144x14x14xf32>
    %283 = torch_c.from_builtin_tensor %282 : tensor<1x144x14x14xf32> -> !torch.vtensor<[1,144,14,14],f32>
    %284 = torch_c.to_builtin_tensor %283 : !torch.vtensor<[1,144,14,14],f32> -> tensor<1x144x14x14xf32>
    %285 = dnn.convolution %284, %cst_18 {kind = "aten.conv2d", parameter_indices = array<i32: 2, 3, 4, 5, 6>, parameters = [unit, [1, 1], [2, 2], [1, 1], 144]} : (tensor<1x144x14x14xf32>, tensor<144x1x5x5xf32>) -> tensor<1x144x14x14xf32>
    %286 = torch_c.from_builtin_tensor %285 : tensor<1x144x14x14xf32> -> !torch.vtensor<[1,144,14,14],f32>
    %287 = torch_c.to_builtin_tensor %286 : !torch.vtensor<[1,144,14,14],f32> -> tensor<1x144x14x14xf32>
    %288:3 = dnn.batch_norm %287, %cst_19, %cst_19, %cst_19, %cst_19 {kind = "aten._native_batch_norm_legit_no_training", parameter_indices = array<i32: 5, 6>, parameters = [1.000000e-02, 1.000000e-03]} : (tensor<1x144x14x14xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144xf32>) -> (tensor<1x144x14x14xf32>, tensor<0xf32>, tensor<0xf32>)
    %289 = torch_c.from_builtin_tensor %288#0 : tensor<1x144x14x14xf32> -> !torch.vtensor<[1,144,14,14],f32>
    %290 = torch_c.to_builtin_tensor %289 : !torch.vtensor<[1,144,14,14],f32> -> tensor<1x144x14x14xf32>
    %291 = dnn.hardswish %290 : (tensor<1x144x14x14xf32>) -> tensor<1x144x14x14xf32>
    %292 = torch_c.from_builtin_tensor %291 : tensor<1x144x14x14xf32> -> !torch.vtensor<[1,144,14,14],f32>
    %293 = torch_c.to_builtin_tensor %292 : !torch.vtensor<[1,144,14,14],f32> -> tensor<1x144x14x14xf32>
    %294 = dnn.adaptive_avg_pool2d %293 {parameter_indices = array<i32: 1>, parameters = [[1, 1]]} : (tensor<1x144x14x14xf32>) -> tensor<1x144x1x1xf32>
    %295 = torch_c.from_builtin_tensor %294 : tensor<1x144x1x1xf32> -> !torch.vtensor<[1,144,1,1],f32>
    %296 = torch_c.to_builtin_tensor %295 : !torch.vtensor<[1,144,1,1],f32> -> tensor<1x144x1x1xf32>
    %297 = dnn.convolution %296, %cst_17, %cst_36 {kind = "aten.conv2d", parameter_indices = array<i32: 3, 4, 5, 6>, parameters = [[1, 1], [0, 0], [1, 1], 1]} : (tensor<1x144x1x1xf32>, tensor<40x144x1x1xf32>, tensor<40xf32>) -> tensor<1x40x1x1xf32>
    %298 = torch_c.from_builtin_tensor %297 : tensor<1x40x1x1xf32> -> !torch.vtensor<[1,40,1,1],f32>
    %299 = torch_c.to_builtin_tensor %298 : !torch.vtensor<[1,40,1,1],f32> -> tensor<1x40x1x1xf32>
    %300 = dnn.relu %299 : (tensor<1x40x1x1xf32>) -> tensor<1x40x1x1xf32>
    %301 = torch_c.from_builtin_tensor %300 : tensor<1x40x1x1xf32> -> !torch.vtensor<[1,40,1,1],f32>
    %302 = torch_c.to_builtin_tensor %301 : !torch.vtensor<[1,40,1,1],f32> -> tensor<1x40x1x1xf32>
    %303 = dnn.convolution %302, %cst_16, %cst_19 {kind = "aten.conv2d", parameter_indices = array<i32: 3, 4, 5, 6>, parameters = [[1, 1], [0, 0], [1, 1], 1]} : (tensor<1x40x1x1xf32>, tensor<144x40x1x1xf32>, tensor<144xf32>) -> tensor<1x144x1x1xf32>
    %304 = torch_c.from_builtin_tensor %303 : tensor<1x144x1x1xf32> -> !torch.vtensor<[1,144,1,1],f32>
    %305 = torch_c.to_builtin_tensor %304 : !torch.vtensor<[1,144,1,1],f32> -> tensor<1x144x1x1xf32>
    %306 = dnn.hardsigmoid %305 : (tensor<1x144x1x1xf32>) -> tensor<1x144x1x1xf32>
    %307 = torch_c.from_builtin_tensor %306 : tensor<1x144x1x1xf32> -> !torch.vtensor<[1,144,1,1],f32>
    %308 = torch_c.to_builtin_tensor %307 : !torch.vtensor<[1,144,1,1],f32> -> tensor<1x144x1x1xf32>
    %309 = torch_c.to_builtin_tensor %292 : !torch.vtensor<[1,144,14,14],f32> -> tensor<1x144x14x14xf32>
    %310 = dnn.mul %308, %309 : (tensor<1x144x1x1xf32>, tensor<1x144x14x14xf32>) -> tensor<1x144x14x14xf32>
    %311 = torch_c.from_builtin_tensor %310 : tensor<1x144x14x14xf32> -> !torch.vtensor<[1,144,14,14],f32>
    %312 = torch_c.to_builtin_tensor %311 : !torch.vtensor<[1,144,14,14],f32> -> tensor<1x144x14x14xf32>
    %313 = dnn.convolution %312, %cst_15 {kind = "aten.conv2d", parameter_indices = array<i32: 2, 3, 4, 5, 6>, parameters = [unit, [1, 1], [0, 0], [1, 1], 1]} : (tensor<1x144x14x14xf32>, tensor<48x144x1x1xf32>) -> tensor<1x48x14x14xf32>
    %314 = torch_c.from_builtin_tensor %313 : tensor<1x48x14x14xf32> -> !torch.vtensor<[1,48,14,14],f32>
    %315 = torch_c.to_builtin_tensor %314 : !torch.vtensor<[1,48,14,14],f32> -> tensor<1x48x14x14xf32>
    %316:3 = dnn.batch_norm %315, %cst_21, %cst_21, %cst_21, %cst_21 {kind = "aten._native_batch_norm_legit_no_training", parameter_indices = array<i32: 5, 6>, parameters = [1.000000e-02, 1.000000e-03]} : (tensor<1x48x14x14xf32>, tensor<48xf32>, tensor<48xf32>, tensor<48xf32>, tensor<48xf32>) -> (tensor<1x48x14x14xf32>, tensor<0xf32>, tensor<0xf32>)
    %317 = torch_c.from_builtin_tensor %316#0 : tensor<1x48x14x14xf32> -> !torch.vtensor<[1,48,14,14],f32>
    %318 = torch_c.to_builtin_tensor %317 : !torch.vtensor<[1,48,14,14],f32> -> tensor<1x48x14x14xf32>
    %319 = torch_c.to_builtin_tensor %274 : !torch.vtensor<[1,48,14,14],f32> -> tensor<1x48x14x14xf32>
    %320 = dnn.add %318, %319 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<1x48x14x14xf32>, tensor<1x48x14x14xf32>) -> tensor<1x48x14x14xf32>
    %321 = torch_c.from_builtin_tensor %320 : tensor<1x48x14x14xf32> -> !torch.vtensor<[1,48,14,14],f32>
    %322 = torch_c.to_builtin_tensor %321 : !torch.vtensor<[1,48,14,14],f32> -> tensor<1x48x14x14xf32>
    %323 = dnn.convolution %322, %cst_14 {kind = "aten.conv2d", parameter_indices = array<i32: 2, 3, 4, 5, 6>, parameters = [unit, [1, 1], [0, 0], [1, 1], 1]} : (tensor<1x48x14x14xf32>, tensor<288x48x1x1xf32>) -> tensor<1x288x14x14xf32>
    %324 = torch_c.from_builtin_tensor %323 : tensor<1x288x14x14xf32> -> !torch.vtensor<[1,288,14,14],f32>
    %325 = torch_c.to_builtin_tensor %324 : !torch.vtensor<[1,288,14,14],f32> -> tensor<1x288x14x14xf32>
    %326:3 = dnn.batch_norm %325, %cst_13, %cst_13, %cst_13, %cst_13 {kind = "aten._native_batch_norm_legit_no_training", parameter_indices = array<i32: 5, 6>, parameters = [1.000000e-02, 1.000000e-03]} : (tensor<1x288x14x14xf32>, tensor<288xf32>, tensor<288xf32>, tensor<288xf32>, tensor<288xf32>) -> (tensor<1x288x14x14xf32>, tensor<0xf32>, tensor<0xf32>)
    %327 = torch_c.from_builtin_tensor %326#0 : tensor<1x288x14x14xf32> -> !torch.vtensor<[1,288,14,14],f32>
    %328 = torch_c.to_builtin_tensor %327 : !torch.vtensor<[1,288,14,14],f32> -> tensor<1x288x14x14xf32>
    %329 = dnn.hardswish %328 : (tensor<1x288x14x14xf32>) -> tensor<1x288x14x14xf32>
    %330 = torch_c.from_builtin_tensor %329 : tensor<1x288x14x14xf32> -> !torch.vtensor<[1,288,14,14],f32>
    %331 = torch_c.to_builtin_tensor %330 : !torch.vtensor<[1,288,14,14],f32> -> tensor<1x288x14x14xf32>
    %332 = dnn.convolution %331, %cst_12 {kind = "aten.conv2d", parameter_indices = array<i32: 2, 3, 4, 5, 6>, parameters = [unit, [2, 2], [2, 2], [1, 1], 288]} : (tensor<1x288x14x14xf32>, tensor<288x1x5x5xf32>) -> tensor<1x288x7x7xf32>
    %333 = torch_c.from_builtin_tensor %332 : tensor<1x288x7x7xf32> -> !torch.vtensor<[1,288,7,7],f32>
    %334 = torch_c.to_builtin_tensor %333 : !torch.vtensor<[1,288,7,7],f32> -> tensor<1x288x7x7xf32>
    %335:3 = dnn.batch_norm %334, %cst_13, %cst_13, %cst_13, %cst_13 {kind = "aten._native_batch_norm_legit_no_training", parameter_indices = array<i32: 5, 6>, parameters = [1.000000e-02, 1.000000e-03]} : (tensor<1x288x7x7xf32>, tensor<288xf32>, tensor<288xf32>, tensor<288xf32>, tensor<288xf32>) -> (tensor<1x288x7x7xf32>, tensor<0xf32>, tensor<0xf32>)
    %336 = torch_c.from_builtin_tensor %335#0 : tensor<1x288x7x7xf32> -> !torch.vtensor<[1,288,7,7],f32>
    %337 = torch_c.to_builtin_tensor %336 : !torch.vtensor<[1,288,7,7],f32> -> tensor<1x288x7x7xf32>
    %338 = dnn.hardswish %337 : (tensor<1x288x7x7xf32>) -> tensor<1x288x7x7xf32>
    %339 = torch_c.from_builtin_tensor %338 : tensor<1x288x7x7xf32> -> !torch.vtensor<[1,288,7,7],f32>
    %340 = torch_c.to_builtin_tensor %339 : !torch.vtensor<[1,288,7,7],f32> -> tensor<1x288x7x7xf32>
    %341 = dnn.adaptive_avg_pool2d %340 {parameter_indices = array<i32: 1>, parameters = [[1, 1]]} : (tensor<1x288x7x7xf32>) -> tensor<1x288x1x1xf32>
    %342 = torch_c.from_builtin_tensor %341 : tensor<1x288x1x1xf32> -> !torch.vtensor<[1,288,1,1],f32>
    %343 = torch_c.to_builtin_tensor %342 : !torch.vtensor<[1,288,1,1],f32> -> tensor<1x288x1x1xf32>
    %344 = dnn.convolution %343, %cst_11, %cst_49 {kind = "aten.conv2d", parameter_indices = array<i32: 3, 4, 5, 6>, parameters = [[1, 1], [0, 0], [1, 1], 1]} : (tensor<1x288x1x1xf32>, tensor<72x288x1x1xf32>, tensor<72xf32>) -> tensor<1x72x1x1xf32>
    %345 = torch_c.from_builtin_tensor %344 : tensor<1x72x1x1xf32> -> !torch.vtensor<[1,72,1,1],f32>
    %346 = torch_c.to_builtin_tensor %345 : !torch.vtensor<[1,72,1,1],f32> -> tensor<1x72x1x1xf32>
    %347 = dnn.relu %346 : (tensor<1x72x1x1xf32>) -> tensor<1x72x1x1xf32>
    %348 = torch_c.from_builtin_tensor %347 : tensor<1x72x1x1xf32> -> !torch.vtensor<[1,72,1,1],f32>
    %349 = torch_c.to_builtin_tensor %348 : !torch.vtensor<[1,72,1,1],f32> -> tensor<1x72x1x1xf32>
    %350 = dnn.convolution %349, %cst_10, %cst_13 {kind = "aten.conv2d", parameter_indices = array<i32: 3, 4, 5, 6>, parameters = [[1, 1], [0, 0], [1, 1], 1]} : (tensor<1x72x1x1xf32>, tensor<288x72x1x1xf32>, tensor<288xf32>) -> tensor<1x288x1x1xf32>
    %351 = torch_c.from_builtin_tensor %350 : tensor<1x288x1x1xf32> -> !torch.vtensor<[1,288,1,1],f32>
    %352 = torch_c.to_builtin_tensor %351 : !torch.vtensor<[1,288,1,1],f32> -> tensor<1x288x1x1xf32>
    %353 = dnn.hardsigmoid %352 : (tensor<1x288x1x1xf32>) -> tensor<1x288x1x1xf32>
    %354 = torch_c.from_builtin_tensor %353 : tensor<1x288x1x1xf32> -> !torch.vtensor<[1,288,1,1],f32>
    %355 = torch_c.to_builtin_tensor %354 : !torch.vtensor<[1,288,1,1],f32> -> tensor<1x288x1x1xf32>
    %356 = torch_c.to_builtin_tensor %339 : !torch.vtensor<[1,288,7,7],f32> -> tensor<1x288x7x7xf32>
    %357 = dnn.mul %355, %356 : (tensor<1x288x1x1xf32>, tensor<1x288x7x7xf32>) -> tensor<1x288x7x7xf32>
    %358 = torch_c.from_builtin_tensor %357 : tensor<1x288x7x7xf32> -> !torch.vtensor<[1,288,7,7],f32>
    %359 = torch_c.to_builtin_tensor %358 : !torch.vtensor<[1,288,7,7],f32> -> tensor<1x288x7x7xf32>
    %360 = dnn.convolution %359, %cst_9 {kind = "aten.conv2d", parameter_indices = array<i32: 2, 3, 4, 5, 6>, parameters = [unit, [1, 1], [0, 0], [1, 1], 1]} : (tensor<1x288x7x7xf32>, tensor<96x288x1x1xf32>) -> tensor<1x96x7x7xf32>
    %361 = torch_c.from_builtin_tensor %360 : tensor<1x96x7x7xf32> -> !torch.vtensor<[1,96,7,7],f32>
    %362 = torch_c.to_builtin_tensor %361 : !torch.vtensor<[1,96,7,7],f32> -> tensor<1x96x7x7xf32>
    %363:3 = dnn.batch_norm %362, %cst_40, %cst_40, %cst_40, %cst_40 {kind = "aten._native_batch_norm_legit_no_training", parameter_indices = array<i32: 5, 6>, parameters = [1.000000e-02, 1.000000e-03]} : (tensor<1x96x7x7xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>) -> (tensor<1x96x7x7xf32>, tensor<0xf32>, tensor<0xf32>)
    %364 = torch_c.from_builtin_tensor %363#0 : tensor<1x96x7x7xf32> -> !torch.vtensor<[1,96,7,7],f32>
    %365 = torch_c.to_builtin_tensor %364 : !torch.vtensor<[1,96,7,7],f32> -> tensor<1x96x7x7xf32>
    %366 = dnn.convolution %365, %cst_8 {kind = "aten.conv2d", parameter_indices = array<i32: 2, 3, 4, 5, 6>, parameters = [unit, [1, 1], [0, 0], [1, 1], 1]} : (tensor<1x96x7x7xf32>, tensor<576x96x1x1xf32>) -> tensor<1x576x7x7xf32>
    %367 = torch_c.from_builtin_tensor %366 : tensor<1x576x7x7xf32> -> !torch.vtensor<[1,576,7,7],f32>
    %368 = torch_c.to_builtin_tensor %367 : !torch.vtensor<[1,576,7,7],f32> -> tensor<1x576x7x7xf32>
    %369:3 = dnn.batch_norm %368, %cst_7, %cst_7, %cst_7, %cst_7 {kind = "aten._native_batch_norm_legit_no_training", parameter_indices = array<i32: 5, 6>, parameters = [1.000000e-02, 1.000000e-03]} : (tensor<1x576x7x7xf32>, tensor<576xf32>, tensor<576xf32>, tensor<576xf32>, tensor<576xf32>) -> (tensor<1x576x7x7xf32>, tensor<0xf32>, tensor<0xf32>)
    %370 = torch_c.from_builtin_tensor %369#0 : tensor<1x576x7x7xf32> -> !torch.vtensor<[1,576,7,7],f32>
    %371 = torch_c.to_builtin_tensor %370 : !torch.vtensor<[1,576,7,7],f32> -> tensor<1x576x7x7xf32>
    %372 = dnn.hardswish %371 : (tensor<1x576x7x7xf32>) -> tensor<1x576x7x7xf32>
    %373 = torch_c.from_builtin_tensor %372 : tensor<1x576x7x7xf32> -> !torch.vtensor<[1,576,7,7],f32>
    %374 = torch_c.to_builtin_tensor %373 : !torch.vtensor<[1,576,7,7],f32> -> tensor<1x576x7x7xf32>
    %375 = dnn.convolution %374, %cst_6 {kind = "aten.conv2d", parameter_indices = array<i32: 2, 3, 4, 5, 6>, parameters = [unit, [1, 1], [2, 2], [1, 1], 576]} : (tensor<1x576x7x7xf32>, tensor<576x1x5x5xf32>) -> tensor<1x576x7x7xf32>
    %376 = torch_c.from_builtin_tensor %375 : tensor<1x576x7x7xf32> -> !torch.vtensor<[1,576,7,7],f32>
    %377 = torch_c.to_builtin_tensor %376 : !torch.vtensor<[1,576,7,7],f32> -> tensor<1x576x7x7xf32>
    %378:3 = dnn.batch_norm %377, %cst_7, %cst_7, %cst_7, %cst_7 {kind = "aten._native_batch_norm_legit_no_training", parameter_indices = array<i32: 5, 6>, parameters = [1.000000e-02, 1.000000e-03]} : (tensor<1x576x7x7xf32>, tensor<576xf32>, tensor<576xf32>, tensor<576xf32>, tensor<576xf32>) -> (tensor<1x576x7x7xf32>, tensor<0xf32>, tensor<0xf32>)
    %379 = torch_c.from_builtin_tensor %378#0 : tensor<1x576x7x7xf32> -> !torch.vtensor<[1,576,7,7],f32>
    %380 = torch_c.to_builtin_tensor %379 : !torch.vtensor<[1,576,7,7],f32> -> tensor<1x576x7x7xf32>
    %381 = dnn.hardswish %380 : (tensor<1x576x7x7xf32>) -> tensor<1x576x7x7xf32>
    %382 = torch_c.from_builtin_tensor %381 : tensor<1x576x7x7xf32> -> !torch.vtensor<[1,576,7,7],f32>
    %383 = torch_c.to_builtin_tensor %382 : !torch.vtensor<[1,576,7,7],f32> -> tensor<1x576x7x7xf32>
    %384 = dnn.adaptive_avg_pool2d %383 {parameter_indices = array<i32: 1>, parameters = [[1, 1]]} : (tensor<1x576x7x7xf32>) -> tensor<1x576x1x1xf32>
    %385 = torch_c.from_builtin_tensor %384 : tensor<1x576x1x1xf32> -> !torch.vtensor<[1,576,1,1],f32>
    %386 = torch_c.to_builtin_tensor %385 : !torch.vtensor<[1,576,1,1],f32> -> tensor<1x576x1x1xf32>
    %387 = dnn.convolution %386, %cst_5, %cst_19 {kind = "aten.conv2d", parameter_indices = array<i32: 3, 4, 5, 6>, parameters = [[1, 1], [0, 0], [1, 1], 1]} : (tensor<1x576x1x1xf32>, tensor<144x576x1x1xf32>, tensor<144xf32>) -> tensor<1x144x1x1xf32>
    %388 = torch_c.from_builtin_tensor %387 : tensor<1x144x1x1xf32> -> !torch.vtensor<[1,144,1,1],f32>
    %389 = torch_c.to_builtin_tensor %388 : !torch.vtensor<[1,144,1,1],f32> -> tensor<1x144x1x1xf32>
    %390 = dnn.relu %389 : (tensor<1x144x1x1xf32>) -> tensor<1x144x1x1xf32>
    %391 = torch_c.from_builtin_tensor %390 : tensor<1x144x1x1xf32> -> !torch.vtensor<[1,144,1,1],f32>
    %392 = torch_c.to_builtin_tensor %391 : !torch.vtensor<[1,144,1,1],f32> -> tensor<1x144x1x1xf32>
    %393 = dnn.convolution %392, %cst_4, %cst_7 {kind = "aten.conv2d", parameter_indices = array<i32: 3, 4, 5, 6>, parameters = [[1, 1], [0, 0], [1, 1], 1]} : (tensor<1x144x1x1xf32>, tensor<576x144x1x1xf32>, tensor<576xf32>) -> tensor<1x576x1x1xf32>
    %394 = torch_c.from_builtin_tensor %393 : tensor<1x576x1x1xf32> -> !torch.vtensor<[1,576,1,1],f32>
    %395 = torch_c.to_builtin_tensor %394 : !torch.vtensor<[1,576,1,1],f32> -> tensor<1x576x1x1xf32>
    %396 = dnn.hardsigmoid %395 : (tensor<1x576x1x1xf32>) -> tensor<1x576x1x1xf32>
    %397 = torch_c.from_builtin_tensor %396 : tensor<1x576x1x1xf32> -> !torch.vtensor<[1,576,1,1],f32>
    %398 = torch_c.to_builtin_tensor %397 : !torch.vtensor<[1,576,1,1],f32> -> tensor<1x576x1x1xf32>
    %399 = torch_c.to_builtin_tensor %382 : !torch.vtensor<[1,576,7,7],f32> -> tensor<1x576x7x7xf32>
    %400 = dnn.mul %398, %399 : (tensor<1x576x1x1xf32>, tensor<1x576x7x7xf32>) -> tensor<1x576x7x7xf32>
    %401 = torch_c.from_builtin_tensor %400 : tensor<1x576x7x7xf32> -> !torch.vtensor<[1,576,7,7],f32>
    %402 = torch_c.to_builtin_tensor %401 : !torch.vtensor<[1,576,7,7],f32> -> tensor<1x576x7x7xf32>
    %403 = dnn.convolution %402, %cst_3 {kind = "aten.conv2d", parameter_indices = array<i32: 2, 3, 4, 5, 6>, parameters = [unit, [1, 1], [0, 0], [1, 1], 1]} : (tensor<1x576x7x7xf32>, tensor<96x576x1x1xf32>) -> tensor<1x96x7x7xf32>
    %404 = torch_c.from_builtin_tensor %403 : tensor<1x96x7x7xf32> -> !torch.vtensor<[1,96,7,7],f32>
    %405 = torch_c.to_builtin_tensor %404 : !torch.vtensor<[1,96,7,7],f32> -> tensor<1x96x7x7xf32>
    %406:3 = dnn.batch_norm %405, %cst_40, %cst_40, %cst_40, %cst_40 {kind = "aten._native_batch_norm_legit_no_training", parameter_indices = array<i32: 5, 6>, parameters = [1.000000e-02, 1.000000e-03]} : (tensor<1x96x7x7xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>) -> (tensor<1x96x7x7xf32>, tensor<0xf32>, tensor<0xf32>)
    %407 = torch_c.from_builtin_tensor %406#0 : tensor<1x96x7x7xf32> -> !torch.vtensor<[1,96,7,7],f32>
    %408 = torch_c.to_builtin_tensor %407 : !torch.vtensor<[1,96,7,7],f32> -> tensor<1x96x7x7xf32>
    %409 = torch_c.to_builtin_tensor %364 : !torch.vtensor<[1,96,7,7],f32> -> tensor<1x96x7x7xf32>
    %410 = dnn.add %408, %409 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<1x96x7x7xf32>, tensor<1x96x7x7xf32>) -> tensor<1x96x7x7xf32>
    %411 = torch_c.from_builtin_tensor %410 : tensor<1x96x7x7xf32> -> !torch.vtensor<[1,96,7,7],f32>
    %412 = torch_c.to_builtin_tensor %411 : !torch.vtensor<[1,96,7,7],f32> -> tensor<1x96x7x7xf32>
    %413 = dnn.convolution %412, %cst_8 {kind = "aten.conv2d", parameter_indices = array<i32: 2, 3, 4, 5, 6>, parameters = [unit, [1, 1], [0, 0], [1, 1], 1]} : (tensor<1x96x7x7xf32>, tensor<576x96x1x1xf32>) -> tensor<1x576x7x7xf32>
    %414 = torch_c.from_builtin_tensor %413 : tensor<1x576x7x7xf32> -> !torch.vtensor<[1,576,7,7],f32>
    %415 = torch_c.to_builtin_tensor %414 : !torch.vtensor<[1,576,7,7],f32> -> tensor<1x576x7x7xf32>
    %416:3 = dnn.batch_norm %415, %cst_7, %cst_7, %cst_7, %cst_7 {kind = "aten._native_batch_norm_legit_no_training", parameter_indices = array<i32: 5, 6>, parameters = [1.000000e-02, 1.000000e-03]} : (tensor<1x576x7x7xf32>, tensor<576xf32>, tensor<576xf32>, tensor<576xf32>, tensor<576xf32>) -> (tensor<1x576x7x7xf32>, tensor<0xf32>, tensor<0xf32>)
    %417 = torch_c.from_builtin_tensor %416#0 : tensor<1x576x7x7xf32> -> !torch.vtensor<[1,576,7,7],f32>
    %418 = torch_c.to_builtin_tensor %417 : !torch.vtensor<[1,576,7,7],f32> -> tensor<1x576x7x7xf32>
    %419 = dnn.hardswish %418 : (tensor<1x576x7x7xf32>) -> tensor<1x576x7x7xf32>
    %420 = torch_c.from_builtin_tensor %419 : tensor<1x576x7x7xf32> -> !torch.vtensor<[1,576,7,7],f32>
    %421 = torch_c.to_builtin_tensor %420 : !torch.vtensor<[1,576,7,7],f32> -> tensor<1x576x7x7xf32>
    %422 = dnn.convolution %421, %cst_6 {kind = "aten.conv2d", parameter_indices = array<i32: 2, 3, 4, 5, 6>, parameters = [unit, [1, 1], [2, 2], [1, 1], 576]} : (tensor<1x576x7x7xf32>, tensor<576x1x5x5xf32>) -> tensor<1x576x7x7xf32>
    %423 = torch_c.from_builtin_tensor %422 : tensor<1x576x7x7xf32> -> !torch.vtensor<[1,576,7,7],f32>
    %424 = torch_c.to_builtin_tensor %423 : !torch.vtensor<[1,576,7,7],f32> -> tensor<1x576x7x7xf32>
    %425:3 = dnn.batch_norm %424, %cst_7, %cst_7, %cst_7, %cst_7 {kind = "aten._native_batch_norm_legit_no_training", parameter_indices = array<i32: 5, 6>, parameters = [1.000000e-02, 1.000000e-03]} : (tensor<1x576x7x7xf32>, tensor<576xf32>, tensor<576xf32>, tensor<576xf32>, tensor<576xf32>) -> (tensor<1x576x7x7xf32>, tensor<0xf32>, tensor<0xf32>)
    %426 = torch_c.from_builtin_tensor %425#0 : tensor<1x576x7x7xf32> -> !torch.vtensor<[1,576,7,7],f32>
    %427 = torch_c.to_builtin_tensor %426 : !torch.vtensor<[1,576,7,7],f32> -> tensor<1x576x7x7xf32>
    %428 = dnn.hardswish %427 : (tensor<1x576x7x7xf32>) -> tensor<1x576x7x7xf32>
    %429 = torch_c.from_builtin_tensor %428 : tensor<1x576x7x7xf32> -> !torch.vtensor<[1,576,7,7],f32>
    %430 = torch_c.to_builtin_tensor %429 : !torch.vtensor<[1,576,7,7],f32> -> tensor<1x576x7x7xf32>
    %431 = dnn.adaptive_avg_pool2d %430 {parameter_indices = array<i32: 1>, parameters = [[1, 1]]} : (tensor<1x576x7x7xf32>) -> tensor<1x576x1x1xf32>
    %432 = torch_c.from_builtin_tensor %431 : tensor<1x576x1x1xf32> -> !torch.vtensor<[1,576,1,1],f32>
    %433 = torch_c.to_builtin_tensor %432 : !torch.vtensor<[1,576,1,1],f32> -> tensor<1x576x1x1xf32>
    %434 = dnn.convolution %433, %cst_5, %cst_19 {kind = "aten.conv2d", parameter_indices = array<i32: 3, 4, 5, 6>, parameters = [[1, 1], [0, 0], [1, 1], 1]} : (tensor<1x576x1x1xf32>, tensor<144x576x1x1xf32>, tensor<144xf32>) -> tensor<1x144x1x1xf32>
    %435 = torch_c.from_builtin_tensor %434 : tensor<1x144x1x1xf32> -> !torch.vtensor<[1,144,1,1],f32>
    %436 = torch_c.to_builtin_tensor %435 : !torch.vtensor<[1,144,1,1],f32> -> tensor<1x144x1x1xf32>
    %437 = dnn.relu %436 : (tensor<1x144x1x1xf32>) -> tensor<1x144x1x1xf32>
    %438 = torch_c.from_builtin_tensor %437 : tensor<1x144x1x1xf32> -> !torch.vtensor<[1,144,1,1],f32>
    %439 = torch_c.to_builtin_tensor %438 : !torch.vtensor<[1,144,1,1],f32> -> tensor<1x144x1x1xf32>
    %440 = dnn.convolution %439, %cst_4, %cst_7 {kind = "aten.conv2d", parameter_indices = array<i32: 3, 4, 5, 6>, parameters = [[1, 1], [0, 0], [1, 1], 1]} : (tensor<1x144x1x1xf32>, tensor<576x144x1x1xf32>, tensor<576xf32>) -> tensor<1x576x1x1xf32>
    %441 = torch_c.from_builtin_tensor %440 : tensor<1x576x1x1xf32> -> !torch.vtensor<[1,576,1,1],f32>
    %442 = torch_c.to_builtin_tensor %441 : !torch.vtensor<[1,576,1,1],f32> -> tensor<1x576x1x1xf32>
    %443 = dnn.hardsigmoid %442 : (tensor<1x576x1x1xf32>) -> tensor<1x576x1x1xf32>
    %444 = torch_c.from_builtin_tensor %443 : tensor<1x576x1x1xf32> -> !torch.vtensor<[1,576,1,1],f32>
    %445 = torch_c.to_builtin_tensor %444 : !torch.vtensor<[1,576,1,1],f32> -> tensor<1x576x1x1xf32>
    %446 = torch_c.to_builtin_tensor %429 : !torch.vtensor<[1,576,7,7],f32> -> tensor<1x576x7x7xf32>
    %447 = dnn.mul %445, %446 : (tensor<1x576x1x1xf32>, tensor<1x576x7x7xf32>) -> tensor<1x576x7x7xf32>
    %448 = torch_c.from_builtin_tensor %447 : tensor<1x576x7x7xf32> -> !torch.vtensor<[1,576,7,7],f32>
    %449 = torch_c.to_builtin_tensor %448 : !torch.vtensor<[1,576,7,7],f32> -> tensor<1x576x7x7xf32>
    %450 = dnn.convolution %449, %cst_3 {kind = "aten.conv2d", parameter_indices = array<i32: 2, 3, 4, 5, 6>, parameters = [unit, [1, 1], [0, 0], [1, 1], 1]} : (tensor<1x576x7x7xf32>, tensor<96x576x1x1xf32>) -> tensor<1x96x7x7xf32>
    %451 = torch_c.from_builtin_tensor %450 : tensor<1x96x7x7xf32> -> !torch.vtensor<[1,96,7,7],f32>
    %452 = torch_c.to_builtin_tensor %451 : !torch.vtensor<[1,96,7,7],f32> -> tensor<1x96x7x7xf32>
    %453:3 = dnn.batch_norm %452, %cst_40, %cst_40, %cst_40, %cst_40 {kind = "aten._native_batch_norm_legit_no_training", parameter_indices = array<i32: 5, 6>, parameters = [1.000000e-02, 1.000000e-03]} : (tensor<1x96x7x7xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>) -> (tensor<1x96x7x7xf32>, tensor<0xf32>, tensor<0xf32>)
    %454 = torch_c.from_builtin_tensor %453#0 : tensor<1x96x7x7xf32> -> !torch.vtensor<[1,96,7,7],f32>
    %455 = torch_c.to_builtin_tensor %454 : !torch.vtensor<[1,96,7,7],f32> -> tensor<1x96x7x7xf32>
    %456 = torch_c.to_builtin_tensor %411 : !torch.vtensor<[1,96,7,7],f32> -> tensor<1x96x7x7xf32>
    %457 = dnn.add %455, %456 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<1x96x7x7xf32>, tensor<1x96x7x7xf32>) -> tensor<1x96x7x7xf32>
    %458 = torch_c.from_builtin_tensor %457 : tensor<1x96x7x7xf32> -> !torch.vtensor<[1,96,7,7],f32>
    %459 = torch_c.to_builtin_tensor %458 : !torch.vtensor<[1,96,7,7],f32> -> tensor<1x96x7x7xf32>
    %460 = dnn.convolution %459, %cst_8 {kind = "aten.conv2d", parameter_indices = array<i32: 2, 3, 4, 5, 6>, parameters = [unit, [1, 1], [0, 0], [1, 1], 1]} : (tensor<1x96x7x7xf32>, tensor<576x96x1x1xf32>) -> tensor<1x576x7x7xf32>
    %461 = torch_c.from_builtin_tensor %460 : tensor<1x576x7x7xf32> -> !torch.vtensor<[1,576,7,7],f32>
    %462 = torch_c.to_builtin_tensor %461 : !torch.vtensor<[1,576,7,7],f32> -> tensor<1x576x7x7xf32>
    %463:3 = dnn.batch_norm %462, %cst_7, %cst_7, %cst_7, %cst_7 {kind = "aten._native_batch_norm_legit_no_training", parameter_indices = array<i32: 5, 6>, parameters = [1.000000e-02, 1.000000e-03]} : (tensor<1x576x7x7xf32>, tensor<576xf32>, tensor<576xf32>, tensor<576xf32>, tensor<576xf32>) -> (tensor<1x576x7x7xf32>, tensor<0xf32>, tensor<0xf32>)
    %464 = torch_c.from_builtin_tensor %463#0 : tensor<1x576x7x7xf32> -> !torch.vtensor<[1,576,7,7],f32>
    %465 = torch_c.to_builtin_tensor %464 : !torch.vtensor<[1,576,7,7],f32> -> tensor<1x576x7x7xf32>
    %466 = dnn.hardswish %465 : (tensor<1x576x7x7xf32>) -> tensor<1x576x7x7xf32>
    %467 = torch_c.from_builtin_tensor %466 : tensor<1x576x7x7xf32> -> !torch.vtensor<[1,576,7,7],f32>
    %468 = torch_c.to_builtin_tensor %467 : !torch.vtensor<[1,576,7,7],f32> -> tensor<1x576x7x7xf32>
    %469 = dnn.adaptive_avg_pool2d %468 {parameter_indices = array<i32: 1>, parameters = [[1, 1]]} : (tensor<1x576x7x7xf32>) -> tensor<1x576x1x1xf32>
    %470 = torch_c.from_builtin_tensor %469 : tensor<1x576x1x1xf32> -> !torch.vtensor<[1,576,1,1],f32>
    %471 = torch.prim.ListConstruct %int1, %int576 : (!torch.int, !torch.int) -> !torch.list<int>
    %472 = torch.aten.view %470, %471 : !torch.vtensor<[1,576,1,1],f32>, !torch.list<int> -> !torch.vtensor<[1,576],f32>
    %473 = torch_c.to_builtin_tensor %472 : !torch.vtensor<[1,576],f32> -> tensor<1x576xf32>
    %474 = dnn.linear %473, %cst_2, %cst_1 : tensor<1x576xf32>, tensor<1024x576xf32>, tensor<1024xf32> -> tensor<1x1024xf32>
    %475 = torch_c.from_builtin_tensor %474 : tensor<1x1024xf32> -> !torch.vtensor<[1,1024],f32>
    %476 = torch_c.to_builtin_tensor %475 : !torch.vtensor<[1,1024],f32> -> tensor<1x1024xf32>
    %477 = dnn.hardswish %476 : (tensor<1x1024xf32>) -> tensor<1x1024xf32>
    %478 = torch_c.from_builtin_tensor %477 : tensor<1x1024xf32> -> !torch.vtensor<[1,1024],f32>
    %479 = torch_c.to_builtin_tensor %478 : !torch.vtensor<[1,1024],f32> -> tensor<1x1024xf32>
    %480 = dnn.linear %479, %cst_0, %cst : tensor<1x1024xf32>, tensor<1000x1024xf32>, tensor<1000xf32> -> tensor<1x1000xf32>
    %481 = torch_c.from_builtin_tensor %480 : tensor<1x1000xf32> -> !torch.vtensor<[1,1000],f32>
    return %481 : !torch.vtensor<[1,1000],f32>
  }
}

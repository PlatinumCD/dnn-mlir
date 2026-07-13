// RUN: dnn-mlir-opt %s | FileCheck %s
//
// MobileNetV3 Small after captures=all and the remaining Torch-to-Linalg
// backend lowering. Every computational operation is preserved in DNN; only
// tensor shape restructuring remains outside the dialect.
//
// CHECK-LABEL: func.func @mobilenet_v3_small
// CHECK-DAG: dnn.convolution
// CHECK-DAG: dnn.batch_norm
// CHECK-DAG: dnn.hardswish
// CHECK-DAG: dnn.hardsigmoid
// CHECK-DAG: dnn.adaptive_avg_pool2d
// CHECK-DAG: dnn.mul
// CHECK-DAG: dnn.add
// CHECK-DAG: dnn.linear
// CHECK-NOT: linalg.
// CHECK-NOT: torch.aten

module {
  func.func @mobilenet_v3_small(%arg0: tensor<1x3x224x224xf32>) -> tensor<1x1000xf32> {
    %cst = arith.constant dense<0.000000e+00> : tensor<16x3x3x3xf32>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<16xf32>
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<16x1x3x3xf32>
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<8x16x1x1xf32>
    %cst_3 = arith.constant dense<0.000000e+00> : tensor<8xf32>
    %cst_4 = arith.constant dense<0.000000e+00> : tensor<16x8x1x1xf32>
    %cst_5 = arith.constant dense<0.000000e+00> : tensor<16x16x1x1xf32>
    %cst_6 = arith.constant dense<0.000000e+00> : tensor<72x16x1x1xf32>
    %cst_7 = arith.constant dense<0.000000e+00> : tensor<72xf32>
    %cst_8 = arith.constant dense<0.000000e+00> : tensor<72x1x3x3xf32>
    %cst_9 = arith.constant dense<0.000000e+00> : tensor<24x72x1x1xf32>
    %cst_10 = arith.constant dense<0.000000e+00> : tensor<24xf32>
    %cst_11 = arith.constant dense<0.000000e+00> : tensor<88x24x1x1xf32>
    %cst_12 = arith.constant dense<0.000000e+00> : tensor<88xf32>
    %cst_13 = arith.constant dense<0.000000e+00> : tensor<88x1x3x3xf32>
    %cst_14 = arith.constant dense<0.000000e+00> : tensor<24x88x1x1xf32>
    %cst_15 = arith.constant dense<0.000000e+00> : tensor<96x24x1x1xf32>
    %cst_16 = arith.constant dense<0.000000e+00> : tensor<96xf32>
    %cst_17 = arith.constant dense<0.000000e+00> : tensor<96x1x5x5xf32>
    %cst_18 = arith.constant dense<0.000000e+00> : tensor<24x96x1x1xf32>
    %cst_19 = arith.constant dense<0.000000e+00> : tensor<40x96x1x1xf32>
    %cst_20 = arith.constant dense<0.000000e+00> : tensor<40xf32>
    %cst_21 = arith.constant dense<0.000000e+00> : tensor<240x40x1x1xf32>
    %cst_22 = arith.constant dense<0.000000e+00> : tensor<240xf32>
    %cst_23 = arith.constant dense<0.000000e+00> : tensor<240x1x5x5xf32>
    %cst_24 = arith.constant dense<0.000000e+00> : tensor<64x240x1x1xf32>
    %cst_25 = arith.constant dense<0.000000e+00> : tensor<64xf32>
    %cst_26 = arith.constant dense<0.000000e+00> : tensor<240x64x1x1xf32>
    %cst_27 = arith.constant dense<0.000000e+00> : tensor<40x240x1x1xf32>
    %cst_28 = arith.constant dense<0.000000e+00> : tensor<120x40x1x1xf32>
    %cst_29 = arith.constant dense<0.000000e+00> : tensor<120xf32>
    %cst_30 = arith.constant dense<0.000000e+00> : tensor<120x1x5x5xf32>
    %cst_31 = arith.constant dense<0.000000e+00> : tensor<32x120x1x1xf32>
    %cst_32 = arith.constant dense<0.000000e+00> : tensor<32xf32>
    %cst_33 = arith.constant dense<0.000000e+00> : tensor<120x32x1x1xf32>
    %cst_34 = arith.constant dense<0.000000e+00> : tensor<48x120x1x1xf32>
    %cst_35 = arith.constant dense<0.000000e+00> : tensor<48xf32>
    %cst_36 = arith.constant dense<0.000000e+00> : tensor<144x48x1x1xf32>
    %cst_37 = arith.constant dense<0.000000e+00> : tensor<144xf32>
    %cst_38 = arith.constant dense<0.000000e+00> : tensor<144x1x5x5xf32>
    %cst_39 = arith.constant dense<0.000000e+00> : tensor<40x144x1x1xf32>
    %cst_40 = arith.constant dense<0.000000e+00> : tensor<144x40x1x1xf32>
    %cst_41 = arith.constant dense<0.000000e+00> : tensor<48x144x1x1xf32>
    %cst_42 = arith.constant dense<0.000000e+00> : tensor<288x48x1x1xf32>
    %cst_43 = arith.constant dense<0.000000e+00> : tensor<288xf32>
    %cst_44 = arith.constant dense<0.000000e+00> : tensor<288x1x5x5xf32>
    %cst_45 = arith.constant dense<0.000000e+00> : tensor<72x288x1x1xf32>
    %cst_46 = arith.constant dense<0.000000e+00> : tensor<288x72x1x1xf32>
    %cst_47 = arith.constant dense<0.000000e+00> : tensor<96x288x1x1xf32>
    %cst_48 = arith.constant dense<0.000000e+00> : tensor<576x96x1x1xf32>
    %cst_49 = arith.constant dense<0.000000e+00> : tensor<576xf32>
    %cst_50 = arith.constant dense<0.000000e+00> : tensor<576x1x5x5xf32>
    %cst_51 = arith.constant dense<0.000000e+00> : tensor<144x576x1x1xf32>
    %cst_52 = arith.constant dense<0.000000e+00> : tensor<576x144x1x1xf32>
    %cst_53 = arith.constant dense<0.000000e+00> : tensor<96x576x1x1xf32>
    %cst_54 = arith.constant dense<0.000000e+00> : tensor<1024x576xf32>
    %cst_55 = arith.constant dense<0.000000e+00> : tensor<1024xf32>
    %cst_56 = arith.constant dense<0.000000e+00> : tensor<1000x1024xf32>
    %cst_57 = arith.constant dense<0.000000e+00> : tensor<1000xf32>
    %0 = dnn.convolution %arg0, %cst {kind = "aten.conv2d", parameter_indices = array<i32: 2, 3, 4, 5, 6>, parameters = [unit, [2, 2], [1, 1], [1, 1], 1]} : (tensor<1x3x224x224xf32>, tensor<16x3x3x3xf32>) -> tensor<1x16x112x112xf32>
    %1:3 = dnn.batch_norm %0, %cst_0, %cst_0, %cst_0, %cst_0 {kind = "aten._native_batch_norm_legit_no_training", parameter_indices = array<i32: 5, 6>, parameters = [1.000000e-02, 1.000000e-03]} : (tensor<1x16x112x112xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>) -> (tensor<1x16x112x112xf32>, tensor<0xf32>, tensor<0xf32>)
    %2 = dnn.hardswish %1#0 : (tensor<1x16x112x112xf32>) -> tensor<1x16x112x112xf32>
    %3 = dnn.convolution %2, %cst_1 {kind = "aten.conv2d", parameter_indices = array<i32: 2, 3, 4, 5, 6>, parameters = [unit, [2, 2], [1, 1], [1, 1], 16]} : (tensor<1x16x112x112xf32>, tensor<16x1x3x3xf32>) -> tensor<1x16x56x56xf32>
    %4:3 = dnn.batch_norm %3, %cst_0, %cst_0, %cst_0, %cst_0 {kind = "aten._native_batch_norm_legit_no_training", parameter_indices = array<i32: 5, 6>, parameters = [1.000000e-02, 1.000000e-03]} : (tensor<1x16x56x56xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>) -> (tensor<1x16x56x56xf32>, tensor<0xf32>, tensor<0xf32>)
    %5 = dnn.relu %4#0 : (tensor<1x16x56x56xf32>) -> tensor<1x16x56x56xf32>
    %6 = dnn.adaptive_avg_pool2d %5 {parameter_indices = array<i32: 1>, parameters = [[1, 1]]} : (tensor<1x16x56x56xf32>) -> tensor<1x16x1x1xf32>
    %7 = dnn.convolution %6, %cst_2, %cst_3 {kind = "aten.conv2d", parameter_indices = array<i32: 3, 4, 5, 6>, parameters = [[1, 1], [0, 0], [1, 1], 1]} : (tensor<1x16x1x1xf32>, tensor<8x16x1x1xf32>, tensor<8xf32>) -> tensor<1x8x1x1xf32>
    %8 = dnn.relu %7 : (tensor<1x8x1x1xf32>) -> tensor<1x8x1x1xf32>
    %9 = dnn.convolution %8, %cst_4, %cst_0 {kind = "aten.conv2d", parameter_indices = array<i32: 3, 4, 5, 6>, parameters = [[1, 1], [0, 0], [1, 1], 1]} : (tensor<1x8x1x1xf32>, tensor<16x8x1x1xf32>, tensor<16xf32>) -> tensor<1x16x1x1xf32>
    %10 = dnn.hardsigmoid %9 : (tensor<1x16x1x1xf32>) -> tensor<1x16x1x1xf32>
    %11 = dnn.mul %10, %5 : (tensor<1x16x1x1xf32>, tensor<1x16x56x56xf32>) -> tensor<1x16x56x56xf32>
    %12 = dnn.convolution %11, %cst_5 {kind = "aten.conv2d", parameter_indices = array<i32: 2, 3, 4, 5, 6>, parameters = [unit, [1, 1], [0, 0], [1, 1], 1]} : (tensor<1x16x56x56xf32>, tensor<16x16x1x1xf32>) -> tensor<1x16x56x56xf32>
    %13:3 = dnn.batch_norm %12, %cst_0, %cst_0, %cst_0, %cst_0 {kind = "aten._native_batch_norm_legit_no_training", parameter_indices = array<i32: 5, 6>, parameters = [1.000000e-02, 1.000000e-03]} : (tensor<1x16x56x56xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>) -> (tensor<1x16x56x56xf32>, tensor<0xf32>, tensor<0xf32>)
    %14 = dnn.convolution %13#0, %cst_6 {kind = "aten.conv2d", parameter_indices = array<i32: 2, 3, 4, 5, 6>, parameters = [unit, [1, 1], [0, 0], [1, 1], 1]} : (tensor<1x16x56x56xf32>, tensor<72x16x1x1xf32>) -> tensor<1x72x56x56xf32>
    %15:3 = dnn.batch_norm %14, %cst_7, %cst_7, %cst_7, %cst_7 {kind = "aten._native_batch_norm_legit_no_training", parameter_indices = array<i32: 5, 6>, parameters = [1.000000e-02, 1.000000e-03]} : (tensor<1x72x56x56xf32>, tensor<72xf32>, tensor<72xf32>, tensor<72xf32>, tensor<72xf32>) -> (tensor<1x72x56x56xf32>, tensor<0xf32>, tensor<0xf32>)
    %16 = dnn.relu %15#0 : (tensor<1x72x56x56xf32>) -> tensor<1x72x56x56xf32>
    %17 = dnn.convolution %16, %cst_8 {kind = "aten.conv2d", parameter_indices = array<i32: 2, 3, 4, 5, 6>, parameters = [unit, [2, 2], [1, 1], [1, 1], 72]} : (tensor<1x72x56x56xf32>, tensor<72x1x3x3xf32>) -> tensor<1x72x28x28xf32>
    %18:3 = dnn.batch_norm %17, %cst_7, %cst_7, %cst_7, %cst_7 {kind = "aten._native_batch_norm_legit_no_training", parameter_indices = array<i32: 5, 6>, parameters = [1.000000e-02, 1.000000e-03]} : (tensor<1x72x28x28xf32>, tensor<72xf32>, tensor<72xf32>, tensor<72xf32>, tensor<72xf32>) -> (tensor<1x72x28x28xf32>, tensor<0xf32>, tensor<0xf32>)
    %19 = dnn.relu %18#0 : (tensor<1x72x28x28xf32>) -> tensor<1x72x28x28xf32>
    %20 = dnn.convolution %19, %cst_9 {kind = "aten.conv2d", parameter_indices = array<i32: 2, 3, 4, 5, 6>, parameters = [unit, [1, 1], [0, 0], [1, 1], 1]} : (tensor<1x72x28x28xf32>, tensor<24x72x1x1xf32>) -> tensor<1x24x28x28xf32>
    %21:3 = dnn.batch_norm %20, %cst_10, %cst_10, %cst_10, %cst_10 {kind = "aten._native_batch_norm_legit_no_training", parameter_indices = array<i32: 5, 6>, parameters = [1.000000e-02, 1.000000e-03]} : (tensor<1x24x28x28xf32>, tensor<24xf32>, tensor<24xf32>, tensor<24xf32>, tensor<24xf32>) -> (tensor<1x24x28x28xf32>, tensor<0xf32>, tensor<0xf32>)
    %22 = dnn.convolution %21#0, %cst_11 {kind = "aten.conv2d", parameter_indices = array<i32: 2, 3, 4, 5, 6>, parameters = [unit, [1, 1], [0, 0], [1, 1], 1]} : (tensor<1x24x28x28xf32>, tensor<88x24x1x1xf32>) -> tensor<1x88x28x28xf32>
    %23:3 = dnn.batch_norm %22, %cst_12, %cst_12, %cst_12, %cst_12 {kind = "aten._native_batch_norm_legit_no_training", parameter_indices = array<i32: 5, 6>, parameters = [1.000000e-02, 1.000000e-03]} : (tensor<1x88x28x28xf32>, tensor<88xf32>, tensor<88xf32>, tensor<88xf32>, tensor<88xf32>) -> (tensor<1x88x28x28xf32>, tensor<0xf32>, tensor<0xf32>)
    %24 = dnn.relu %23#0 : (tensor<1x88x28x28xf32>) -> tensor<1x88x28x28xf32>
    %25 = dnn.convolution %24, %cst_13 {kind = "aten.conv2d", parameter_indices = array<i32: 2, 3, 4, 5, 6>, parameters = [unit, [1, 1], [1, 1], [1, 1], 88]} : (tensor<1x88x28x28xf32>, tensor<88x1x3x3xf32>) -> tensor<1x88x28x28xf32>
    %26:3 = dnn.batch_norm %25, %cst_12, %cst_12, %cst_12, %cst_12 {kind = "aten._native_batch_norm_legit_no_training", parameter_indices = array<i32: 5, 6>, parameters = [1.000000e-02, 1.000000e-03]} : (tensor<1x88x28x28xf32>, tensor<88xf32>, tensor<88xf32>, tensor<88xf32>, tensor<88xf32>) -> (tensor<1x88x28x28xf32>, tensor<0xf32>, tensor<0xf32>)
    %27 = dnn.relu %26#0 : (tensor<1x88x28x28xf32>) -> tensor<1x88x28x28xf32>
    %28 = dnn.convolution %27, %cst_14 {kind = "aten.conv2d", parameter_indices = array<i32: 2, 3, 4, 5, 6>, parameters = [unit, [1, 1], [0, 0], [1, 1], 1]} : (tensor<1x88x28x28xf32>, tensor<24x88x1x1xf32>) -> tensor<1x24x28x28xf32>
    %29:3 = dnn.batch_norm %28, %cst_10, %cst_10, %cst_10, %cst_10 {kind = "aten._native_batch_norm_legit_no_training", parameter_indices = array<i32: 5, 6>, parameters = [1.000000e-02, 1.000000e-03]} : (tensor<1x24x28x28xf32>, tensor<24xf32>, tensor<24xf32>, tensor<24xf32>, tensor<24xf32>) -> (tensor<1x24x28x28xf32>, tensor<0xf32>, tensor<0xf32>)
    %30 = dnn.add %29#0, %21#0 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<1x24x28x28xf32>, tensor<1x24x28x28xf32>) -> tensor<1x24x28x28xf32>
    %31 = dnn.convolution %30, %cst_15 {kind = "aten.conv2d", parameter_indices = array<i32: 2, 3, 4, 5, 6>, parameters = [unit, [1, 1], [0, 0], [1, 1], 1]} : (tensor<1x24x28x28xf32>, tensor<96x24x1x1xf32>) -> tensor<1x96x28x28xf32>
    %32:3 = dnn.batch_norm %31, %cst_16, %cst_16, %cst_16, %cst_16 {kind = "aten._native_batch_norm_legit_no_training", parameter_indices = array<i32: 5, 6>, parameters = [1.000000e-02, 1.000000e-03]} : (tensor<1x96x28x28xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>) -> (tensor<1x96x28x28xf32>, tensor<0xf32>, tensor<0xf32>)
    %33 = dnn.hardswish %32#0 : (tensor<1x96x28x28xf32>) -> tensor<1x96x28x28xf32>
    %34 = dnn.convolution %33, %cst_17 {kind = "aten.conv2d", parameter_indices = array<i32: 2, 3, 4, 5, 6>, parameters = [unit, [2, 2], [2, 2], [1, 1], 96]} : (tensor<1x96x28x28xf32>, tensor<96x1x5x5xf32>) -> tensor<1x96x14x14xf32>
    %35:3 = dnn.batch_norm %34, %cst_16, %cst_16, %cst_16, %cst_16 {kind = "aten._native_batch_norm_legit_no_training", parameter_indices = array<i32: 5, 6>, parameters = [1.000000e-02, 1.000000e-03]} : (tensor<1x96x14x14xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>) -> (tensor<1x96x14x14xf32>, tensor<0xf32>, tensor<0xf32>)
    %36 = dnn.hardswish %35#0 : (tensor<1x96x14x14xf32>) -> tensor<1x96x14x14xf32>
    %37 = dnn.adaptive_avg_pool2d %36 {parameter_indices = array<i32: 1>, parameters = [[1, 1]]} : (tensor<1x96x14x14xf32>) -> tensor<1x96x1x1xf32>
    %38 = dnn.convolution %37, %cst_18, %cst_10 {kind = "aten.conv2d", parameter_indices = array<i32: 3, 4, 5, 6>, parameters = [[1, 1], [0, 0], [1, 1], 1]} : (tensor<1x96x1x1xf32>, tensor<24x96x1x1xf32>, tensor<24xf32>) -> tensor<1x24x1x1xf32>
    %39 = dnn.relu %38 : (tensor<1x24x1x1xf32>) -> tensor<1x24x1x1xf32>
    %40 = dnn.convolution %39, %cst_15, %cst_16 {kind = "aten.conv2d", parameter_indices = array<i32: 3, 4, 5, 6>, parameters = [[1, 1], [0, 0], [1, 1], 1]} : (tensor<1x24x1x1xf32>, tensor<96x24x1x1xf32>, tensor<96xf32>) -> tensor<1x96x1x1xf32>
    %41 = dnn.hardsigmoid %40 : (tensor<1x96x1x1xf32>) -> tensor<1x96x1x1xf32>
    %42 = dnn.mul %41, %36 : (tensor<1x96x1x1xf32>, tensor<1x96x14x14xf32>) -> tensor<1x96x14x14xf32>
    %43 = dnn.convolution %42, %cst_19 {kind = "aten.conv2d", parameter_indices = array<i32: 2, 3, 4, 5, 6>, parameters = [unit, [1, 1], [0, 0], [1, 1], 1]} : (tensor<1x96x14x14xf32>, tensor<40x96x1x1xf32>) -> tensor<1x40x14x14xf32>
    %44:3 = dnn.batch_norm %43, %cst_20, %cst_20, %cst_20, %cst_20 {kind = "aten._native_batch_norm_legit_no_training", parameter_indices = array<i32: 5, 6>, parameters = [1.000000e-02, 1.000000e-03]} : (tensor<1x40x14x14xf32>, tensor<40xf32>, tensor<40xf32>, tensor<40xf32>, tensor<40xf32>) -> (tensor<1x40x14x14xf32>, tensor<0xf32>, tensor<0xf32>)
    %45 = dnn.convolution %44#0, %cst_21 {kind = "aten.conv2d", parameter_indices = array<i32: 2, 3, 4, 5, 6>, parameters = [unit, [1, 1], [0, 0], [1, 1], 1]} : (tensor<1x40x14x14xf32>, tensor<240x40x1x1xf32>) -> tensor<1x240x14x14xf32>
    %46:3 = dnn.batch_norm %45, %cst_22, %cst_22, %cst_22, %cst_22 {kind = "aten._native_batch_norm_legit_no_training", parameter_indices = array<i32: 5, 6>, parameters = [1.000000e-02, 1.000000e-03]} : (tensor<1x240x14x14xf32>, tensor<240xf32>, tensor<240xf32>, tensor<240xf32>, tensor<240xf32>) -> (tensor<1x240x14x14xf32>, tensor<0xf32>, tensor<0xf32>)
    %47 = dnn.hardswish %46#0 : (tensor<1x240x14x14xf32>) -> tensor<1x240x14x14xf32>
    %48 = dnn.convolution %47, %cst_23 {kind = "aten.conv2d", parameter_indices = array<i32: 2, 3, 4, 5, 6>, parameters = [unit, [1, 1], [2, 2], [1, 1], 240]} : (tensor<1x240x14x14xf32>, tensor<240x1x5x5xf32>) -> tensor<1x240x14x14xf32>
    %49:3 = dnn.batch_norm %48, %cst_22, %cst_22, %cst_22, %cst_22 {kind = "aten._native_batch_norm_legit_no_training", parameter_indices = array<i32: 5, 6>, parameters = [1.000000e-02, 1.000000e-03]} : (tensor<1x240x14x14xf32>, tensor<240xf32>, tensor<240xf32>, tensor<240xf32>, tensor<240xf32>) -> (tensor<1x240x14x14xf32>, tensor<0xf32>, tensor<0xf32>)
    %50 = dnn.hardswish %49#0 : (tensor<1x240x14x14xf32>) -> tensor<1x240x14x14xf32>
    %51 = dnn.adaptive_avg_pool2d %50 {parameter_indices = array<i32: 1>, parameters = [[1, 1]]} : (tensor<1x240x14x14xf32>) -> tensor<1x240x1x1xf32>
    %52 = dnn.convolution %51, %cst_24, %cst_25 {kind = "aten.conv2d", parameter_indices = array<i32: 3, 4, 5, 6>, parameters = [[1, 1], [0, 0], [1, 1], 1]} : (tensor<1x240x1x1xf32>, tensor<64x240x1x1xf32>, tensor<64xf32>) -> tensor<1x64x1x1xf32>
    %53 = dnn.relu %52 : (tensor<1x64x1x1xf32>) -> tensor<1x64x1x1xf32>
    %54 = dnn.convolution %53, %cst_26, %cst_22 {kind = "aten.conv2d", parameter_indices = array<i32: 3, 4, 5, 6>, parameters = [[1, 1], [0, 0], [1, 1], 1]} : (tensor<1x64x1x1xf32>, tensor<240x64x1x1xf32>, tensor<240xf32>) -> tensor<1x240x1x1xf32>
    %55 = dnn.hardsigmoid %54 : (tensor<1x240x1x1xf32>) -> tensor<1x240x1x1xf32>
    %56 = dnn.mul %55, %50 : (tensor<1x240x1x1xf32>, tensor<1x240x14x14xf32>) -> tensor<1x240x14x14xf32>
    %57 = dnn.convolution %56, %cst_27 {kind = "aten.conv2d", parameter_indices = array<i32: 2, 3, 4, 5, 6>, parameters = [unit, [1, 1], [0, 0], [1, 1], 1]} : (tensor<1x240x14x14xf32>, tensor<40x240x1x1xf32>) -> tensor<1x40x14x14xf32>
    %58:3 = dnn.batch_norm %57, %cst_20, %cst_20, %cst_20, %cst_20 {kind = "aten._native_batch_norm_legit_no_training", parameter_indices = array<i32: 5, 6>, parameters = [1.000000e-02, 1.000000e-03]} : (tensor<1x40x14x14xf32>, tensor<40xf32>, tensor<40xf32>, tensor<40xf32>, tensor<40xf32>) -> (tensor<1x40x14x14xf32>, tensor<0xf32>, tensor<0xf32>)
    %59 = dnn.add %58#0, %44#0 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<1x40x14x14xf32>, tensor<1x40x14x14xf32>) -> tensor<1x40x14x14xf32>
    %60 = dnn.convolution %59, %cst_21 {kind = "aten.conv2d", parameter_indices = array<i32: 2, 3, 4, 5, 6>, parameters = [unit, [1, 1], [0, 0], [1, 1], 1]} : (tensor<1x40x14x14xf32>, tensor<240x40x1x1xf32>) -> tensor<1x240x14x14xf32>
    %61:3 = dnn.batch_norm %60, %cst_22, %cst_22, %cst_22, %cst_22 {kind = "aten._native_batch_norm_legit_no_training", parameter_indices = array<i32: 5, 6>, parameters = [1.000000e-02, 1.000000e-03]} : (tensor<1x240x14x14xf32>, tensor<240xf32>, tensor<240xf32>, tensor<240xf32>, tensor<240xf32>) -> (tensor<1x240x14x14xf32>, tensor<0xf32>, tensor<0xf32>)
    %62 = dnn.hardswish %61#0 : (tensor<1x240x14x14xf32>) -> tensor<1x240x14x14xf32>
    %63 = dnn.convolution %62, %cst_23 {kind = "aten.conv2d", parameter_indices = array<i32: 2, 3, 4, 5, 6>, parameters = [unit, [1, 1], [2, 2], [1, 1], 240]} : (tensor<1x240x14x14xf32>, tensor<240x1x5x5xf32>) -> tensor<1x240x14x14xf32>
    %64:3 = dnn.batch_norm %63, %cst_22, %cst_22, %cst_22, %cst_22 {kind = "aten._native_batch_norm_legit_no_training", parameter_indices = array<i32: 5, 6>, parameters = [1.000000e-02, 1.000000e-03]} : (tensor<1x240x14x14xf32>, tensor<240xf32>, tensor<240xf32>, tensor<240xf32>, tensor<240xf32>) -> (tensor<1x240x14x14xf32>, tensor<0xf32>, tensor<0xf32>)
    %65 = dnn.hardswish %64#0 : (tensor<1x240x14x14xf32>) -> tensor<1x240x14x14xf32>
    %66 = dnn.adaptive_avg_pool2d %65 {parameter_indices = array<i32: 1>, parameters = [[1, 1]]} : (tensor<1x240x14x14xf32>) -> tensor<1x240x1x1xf32>
    %67 = dnn.convolution %66, %cst_24, %cst_25 {kind = "aten.conv2d", parameter_indices = array<i32: 3, 4, 5, 6>, parameters = [[1, 1], [0, 0], [1, 1], 1]} : (tensor<1x240x1x1xf32>, tensor<64x240x1x1xf32>, tensor<64xf32>) -> tensor<1x64x1x1xf32>
    %68 = dnn.relu %67 : (tensor<1x64x1x1xf32>) -> tensor<1x64x1x1xf32>
    %69 = dnn.convolution %68, %cst_26, %cst_22 {kind = "aten.conv2d", parameter_indices = array<i32: 3, 4, 5, 6>, parameters = [[1, 1], [0, 0], [1, 1], 1]} : (tensor<1x64x1x1xf32>, tensor<240x64x1x1xf32>, tensor<240xf32>) -> tensor<1x240x1x1xf32>
    %70 = dnn.hardsigmoid %69 : (tensor<1x240x1x1xf32>) -> tensor<1x240x1x1xf32>
    %71 = dnn.mul %70, %65 : (tensor<1x240x1x1xf32>, tensor<1x240x14x14xf32>) -> tensor<1x240x14x14xf32>
    %72 = dnn.convolution %71, %cst_27 {kind = "aten.conv2d", parameter_indices = array<i32: 2, 3, 4, 5, 6>, parameters = [unit, [1, 1], [0, 0], [1, 1], 1]} : (tensor<1x240x14x14xf32>, tensor<40x240x1x1xf32>) -> tensor<1x40x14x14xf32>
    %73:3 = dnn.batch_norm %72, %cst_20, %cst_20, %cst_20, %cst_20 {kind = "aten._native_batch_norm_legit_no_training", parameter_indices = array<i32: 5, 6>, parameters = [1.000000e-02, 1.000000e-03]} : (tensor<1x40x14x14xf32>, tensor<40xf32>, tensor<40xf32>, tensor<40xf32>, tensor<40xf32>) -> (tensor<1x40x14x14xf32>, tensor<0xf32>, tensor<0xf32>)
    %74 = dnn.add %73#0, %59 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<1x40x14x14xf32>, tensor<1x40x14x14xf32>) -> tensor<1x40x14x14xf32>
    %75 = dnn.convolution %74, %cst_28 {kind = "aten.conv2d", parameter_indices = array<i32: 2, 3, 4, 5, 6>, parameters = [unit, [1, 1], [0, 0], [1, 1], 1]} : (tensor<1x40x14x14xf32>, tensor<120x40x1x1xf32>) -> tensor<1x120x14x14xf32>
    %76:3 = dnn.batch_norm %75, %cst_29, %cst_29, %cst_29, %cst_29 {kind = "aten._native_batch_norm_legit_no_training", parameter_indices = array<i32: 5, 6>, parameters = [1.000000e-02, 1.000000e-03]} : (tensor<1x120x14x14xf32>, tensor<120xf32>, tensor<120xf32>, tensor<120xf32>, tensor<120xf32>) -> (tensor<1x120x14x14xf32>, tensor<0xf32>, tensor<0xf32>)
    %77 = dnn.hardswish %76#0 : (tensor<1x120x14x14xf32>) -> tensor<1x120x14x14xf32>
    %78 = dnn.convolution %77, %cst_30 {kind = "aten.conv2d", parameter_indices = array<i32: 2, 3, 4, 5, 6>, parameters = [unit, [1, 1], [2, 2], [1, 1], 120]} : (tensor<1x120x14x14xf32>, tensor<120x1x5x5xf32>) -> tensor<1x120x14x14xf32>
    %79:3 = dnn.batch_norm %78, %cst_29, %cst_29, %cst_29, %cst_29 {kind = "aten._native_batch_norm_legit_no_training", parameter_indices = array<i32: 5, 6>, parameters = [1.000000e-02, 1.000000e-03]} : (tensor<1x120x14x14xf32>, tensor<120xf32>, tensor<120xf32>, tensor<120xf32>, tensor<120xf32>) -> (tensor<1x120x14x14xf32>, tensor<0xf32>, tensor<0xf32>)
    %80 = dnn.hardswish %79#0 : (tensor<1x120x14x14xf32>) -> tensor<1x120x14x14xf32>
    %81 = dnn.adaptive_avg_pool2d %80 {parameter_indices = array<i32: 1>, parameters = [[1, 1]]} : (tensor<1x120x14x14xf32>) -> tensor<1x120x1x1xf32>
    %82 = dnn.convolution %81, %cst_31, %cst_32 {kind = "aten.conv2d", parameter_indices = array<i32: 3, 4, 5, 6>, parameters = [[1, 1], [0, 0], [1, 1], 1]} : (tensor<1x120x1x1xf32>, tensor<32x120x1x1xf32>, tensor<32xf32>) -> tensor<1x32x1x1xf32>
    %83 = dnn.relu %82 : (tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %84 = dnn.convolution %83, %cst_33, %cst_29 {kind = "aten.conv2d", parameter_indices = array<i32: 3, 4, 5, 6>, parameters = [[1, 1], [0, 0], [1, 1], 1]} : (tensor<1x32x1x1xf32>, tensor<120x32x1x1xf32>, tensor<120xf32>) -> tensor<1x120x1x1xf32>
    %85 = dnn.hardsigmoid %84 : (tensor<1x120x1x1xf32>) -> tensor<1x120x1x1xf32>
    %86 = dnn.mul %85, %80 : (tensor<1x120x1x1xf32>, tensor<1x120x14x14xf32>) -> tensor<1x120x14x14xf32>
    %87 = dnn.convolution %86, %cst_34 {kind = "aten.conv2d", parameter_indices = array<i32: 2, 3, 4, 5, 6>, parameters = [unit, [1, 1], [0, 0], [1, 1], 1]} : (tensor<1x120x14x14xf32>, tensor<48x120x1x1xf32>) -> tensor<1x48x14x14xf32>
    %88:3 = dnn.batch_norm %87, %cst_35, %cst_35, %cst_35, %cst_35 {kind = "aten._native_batch_norm_legit_no_training", parameter_indices = array<i32: 5, 6>, parameters = [1.000000e-02, 1.000000e-03]} : (tensor<1x48x14x14xf32>, tensor<48xf32>, tensor<48xf32>, tensor<48xf32>, tensor<48xf32>) -> (tensor<1x48x14x14xf32>, tensor<0xf32>, tensor<0xf32>)
    %89 = dnn.convolution %88#0, %cst_36 {kind = "aten.conv2d", parameter_indices = array<i32: 2, 3, 4, 5, 6>, parameters = [unit, [1, 1], [0, 0], [1, 1], 1]} : (tensor<1x48x14x14xf32>, tensor<144x48x1x1xf32>) -> tensor<1x144x14x14xf32>
    %90:3 = dnn.batch_norm %89, %cst_37, %cst_37, %cst_37, %cst_37 {kind = "aten._native_batch_norm_legit_no_training", parameter_indices = array<i32: 5, 6>, parameters = [1.000000e-02, 1.000000e-03]} : (tensor<1x144x14x14xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144xf32>) -> (tensor<1x144x14x14xf32>, tensor<0xf32>, tensor<0xf32>)
    %91 = dnn.hardswish %90#0 : (tensor<1x144x14x14xf32>) -> tensor<1x144x14x14xf32>
    %92 = dnn.convolution %91, %cst_38 {kind = "aten.conv2d", parameter_indices = array<i32: 2, 3, 4, 5, 6>, parameters = [unit, [1, 1], [2, 2], [1, 1], 144]} : (tensor<1x144x14x14xf32>, tensor<144x1x5x5xf32>) -> tensor<1x144x14x14xf32>
    %93:3 = dnn.batch_norm %92, %cst_37, %cst_37, %cst_37, %cst_37 {kind = "aten._native_batch_norm_legit_no_training", parameter_indices = array<i32: 5, 6>, parameters = [1.000000e-02, 1.000000e-03]} : (tensor<1x144x14x14xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144xf32>) -> (tensor<1x144x14x14xf32>, tensor<0xf32>, tensor<0xf32>)
    %94 = dnn.hardswish %93#0 : (tensor<1x144x14x14xf32>) -> tensor<1x144x14x14xf32>
    %95 = dnn.adaptive_avg_pool2d %94 {parameter_indices = array<i32: 1>, parameters = [[1, 1]]} : (tensor<1x144x14x14xf32>) -> tensor<1x144x1x1xf32>
    %96 = dnn.convolution %95, %cst_39, %cst_20 {kind = "aten.conv2d", parameter_indices = array<i32: 3, 4, 5, 6>, parameters = [[1, 1], [0, 0], [1, 1], 1]} : (tensor<1x144x1x1xf32>, tensor<40x144x1x1xf32>, tensor<40xf32>) -> tensor<1x40x1x1xf32>
    %97 = dnn.relu %96 : (tensor<1x40x1x1xf32>) -> tensor<1x40x1x1xf32>
    %98 = dnn.convolution %97, %cst_40, %cst_37 {kind = "aten.conv2d", parameter_indices = array<i32: 3, 4, 5, 6>, parameters = [[1, 1], [0, 0], [1, 1], 1]} : (tensor<1x40x1x1xf32>, tensor<144x40x1x1xf32>, tensor<144xf32>) -> tensor<1x144x1x1xf32>
    %99 = dnn.hardsigmoid %98 : (tensor<1x144x1x1xf32>) -> tensor<1x144x1x1xf32>
    %100 = dnn.mul %99, %94 : (tensor<1x144x1x1xf32>, tensor<1x144x14x14xf32>) -> tensor<1x144x14x14xf32>
    %101 = dnn.convolution %100, %cst_41 {kind = "aten.conv2d", parameter_indices = array<i32: 2, 3, 4, 5, 6>, parameters = [unit, [1, 1], [0, 0], [1, 1], 1]} : (tensor<1x144x14x14xf32>, tensor<48x144x1x1xf32>) -> tensor<1x48x14x14xf32>
    %102:3 = dnn.batch_norm %101, %cst_35, %cst_35, %cst_35, %cst_35 {kind = "aten._native_batch_norm_legit_no_training", parameter_indices = array<i32: 5, 6>, parameters = [1.000000e-02, 1.000000e-03]} : (tensor<1x48x14x14xf32>, tensor<48xf32>, tensor<48xf32>, tensor<48xf32>, tensor<48xf32>) -> (tensor<1x48x14x14xf32>, tensor<0xf32>, tensor<0xf32>)
    %103 = dnn.add %102#0, %88#0 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<1x48x14x14xf32>, tensor<1x48x14x14xf32>) -> tensor<1x48x14x14xf32>
    %104 = dnn.convolution %103, %cst_42 {kind = "aten.conv2d", parameter_indices = array<i32: 2, 3, 4, 5, 6>, parameters = [unit, [1, 1], [0, 0], [1, 1], 1]} : (tensor<1x48x14x14xf32>, tensor<288x48x1x1xf32>) -> tensor<1x288x14x14xf32>
    %105:3 = dnn.batch_norm %104, %cst_43, %cst_43, %cst_43, %cst_43 {kind = "aten._native_batch_norm_legit_no_training", parameter_indices = array<i32: 5, 6>, parameters = [1.000000e-02, 1.000000e-03]} : (tensor<1x288x14x14xf32>, tensor<288xf32>, tensor<288xf32>, tensor<288xf32>, tensor<288xf32>) -> (tensor<1x288x14x14xf32>, tensor<0xf32>, tensor<0xf32>)
    %106 = dnn.hardswish %105#0 : (tensor<1x288x14x14xf32>) -> tensor<1x288x14x14xf32>
    %107 = dnn.convolution %106, %cst_44 {kind = "aten.conv2d", parameter_indices = array<i32: 2, 3, 4, 5, 6>, parameters = [unit, [2, 2], [2, 2], [1, 1], 288]} : (tensor<1x288x14x14xf32>, tensor<288x1x5x5xf32>) -> tensor<1x288x7x7xf32>
    %108:3 = dnn.batch_norm %107, %cst_43, %cst_43, %cst_43, %cst_43 {kind = "aten._native_batch_norm_legit_no_training", parameter_indices = array<i32: 5, 6>, parameters = [1.000000e-02, 1.000000e-03]} : (tensor<1x288x7x7xf32>, tensor<288xf32>, tensor<288xf32>, tensor<288xf32>, tensor<288xf32>) -> (tensor<1x288x7x7xf32>, tensor<0xf32>, tensor<0xf32>)
    %109 = dnn.hardswish %108#0 : (tensor<1x288x7x7xf32>) -> tensor<1x288x7x7xf32>
    %110 = dnn.adaptive_avg_pool2d %109 {parameter_indices = array<i32: 1>, parameters = [[1, 1]]} : (tensor<1x288x7x7xf32>) -> tensor<1x288x1x1xf32>
    %111 = dnn.convolution %110, %cst_45, %cst_7 {kind = "aten.conv2d", parameter_indices = array<i32: 3, 4, 5, 6>, parameters = [[1, 1], [0, 0], [1, 1], 1]} : (tensor<1x288x1x1xf32>, tensor<72x288x1x1xf32>, tensor<72xf32>) -> tensor<1x72x1x1xf32>
    %112 = dnn.relu %111 : (tensor<1x72x1x1xf32>) -> tensor<1x72x1x1xf32>
    %113 = dnn.convolution %112, %cst_46, %cst_43 {kind = "aten.conv2d", parameter_indices = array<i32: 3, 4, 5, 6>, parameters = [[1, 1], [0, 0], [1, 1], 1]} : (tensor<1x72x1x1xf32>, tensor<288x72x1x1xf32>, tensor<288xf32>) -> tensor<1x288x1x1xf32>
    %114 = dnn.hardsigmoid %113 : (tensor<1x288x1x1xf32>) -> tensor<1x288x1x1xf32>
    %115 = dnn.mul %114, %109 : (tensor<1x288x1x1xf32>, tensor<1x288x7x7xf32>) -> tensor<1x288x7x7xf32>
    %116 = dnn.convolution %115, %cst_47 {kind = "aten.conv2d", parameter_indices = array<i32: 2, 3, 4, 5, 6>, parameters = [unit, [1, 1], [0, 0], [1, 1], 1]} : (tensor<1x288x7x7xf32>, tensor<96x288x1x1xf32>) -> tensor<1x96x7x7xf32>
    %117:3 = dnn.batch_norm %116, %cst_16, %cst_16, %cst_16, %cst_16 {kind = "aten._native_batch_norm_legit_no_training", parameter_indices = array<i32: 5, 6>, parameters = [1.000000e-02, 1.000000e-03]} : (tensor<1x96x7x7xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>) -> (tensor<1x96x7x7xf32>, tensor<0xf32>, tensor<0xf32>)
    %118 = dnn.convolution %117#0, %cst_48 {kind = "aten.conv2d", parameter_indices = array<i32: 2, 3, 4, 5, 6>, parameters = [unit, [1, 1], [0, 0], [1, 1], 1]} : (tensor<1x96x7x7xf32>, tensor<576x96x1x1xf32>) -> tensor<1x576x7x7xf32>
    %119:3 = dnn.batch_norm %118, %cst_49, %cst_49, %cst_49, %cst_49 {kind = "aten._native_batch_norm_legit_no_training", parameter_indices = array<i32: 5, 6>, parameters = [1.000000e-02, 1.000000e-03]} : (tensor<1x576x7x7xf32>, tensor<576xf32>, tensor<576xf32>, tensor<576xf32>, tensor<576xf32>) -> (tensor<1x576x7x7xf32>, tensor<0xf32>, tensor<0xf32>)
    %120 = dnn.hardswish %119#0 : (tensor<1x576x7x7xf32>) -> tensor<1x576x7x7xf32>
    %121 = dnn.convolution %120, %cst_50 {kind = "aten.conv2d", parameter_indices = array<i32: 2, 3, 4, 5, 6>, parameters = [unit, [1, 1], [2, 2], [1, 1], 576]} : (tensor<1x576x7x7xf32>, tensor<576x1x5x5xf32>) -> tensor<1x576x7x7xf32>
    %122:3 = dnn.batch_norm %121, %cst_49, %cst_49, %cst_49, %cst_49 {kind = "aten._native_batch_norm_legit_no_training", parameter_indices = array<i32: 5, 6>, parameters = [1.000000e-02, 1.000000e-03]} : (tensor<1x576x7x7xf32>, tensor<576xf32>, tensor<576xf32>, tensor<576xf32>, tensor<576xf32>) -> (tensor<1x576x7x7xf32>, tensor<0xf32>, tensor<0xf32>)
    %123 = dnn.hardswish %122#0 : (tensor<1x576x7x7xf32>) -> tensor<1x576x7x7xf32>
    %124 = dnn.adaptive_avg_pool2d %123 {parameter_indices = array<i32: 1>, parameters = [[1, 1]]} : (tensor<1x576x7x7xf32>) -> tensor<1x576x1x1xf32>
    %125 = dnn.convolution %124, %cst_51, %cst_37 {kind = "aten.conv2d", parameter_indices = array<i32: 3, 4, 5, 6>, parameters = [[1, 1], [0, 0], [1, 1], 1]} : (tensor<1x576x1x1xf32>, tensor<144x576x1x1xf32>, tensor<144xf32>) -> tensor<1x144x1x1xf32>
    %126 = dnn.relu %125 : (tensor<1x144x1x1xf32>) -> tensor<1x144x1x1xf32>
    %127 = dnn.convolution %126, %cst_52, %cst_49 {kind = "aten.conv2d", parameter_indices = array<i32: 3, 4, 5, 6>, parameters = [[1, 1], [0, 0], [1, 1], 1]} : (tensor<1x144x1x1xf32>, tensor<576x144x1x1xf32>, tensor<576xf32>) -> tensor<1x576x1x1xf32>
    %128 = dnn.hardsigmoid %127 : (tensor<1x576x1x1xf32>) -> tensor<1x576x1x1xf32>
    %129 = dnn.mul %128, %123 : (tensor<1x576x1x1xf32>, tensor<1x576x7x7xf32>) -> tensor<1x576x7x7xf32>
    %130 = dnn.convolution %129, %cst_53 {kind = "aten.conv2d", parameter_indices = array<i32: 2, 3, 4, 5, 6>, parameters = [unit, [1, 1], [0, 0], [1, 1], 1]} : (tensor<1x576x7x7xf32>, tensor<96x576x1x1xf32>) -> tensor<1x96x7x7xf32>
    %131:3 = dnn.batch_norm %130, %cst_16, %cst_16, %cst_16, %cst_16 {kind = "aten._native_batch_norm_legit_no_training", parameter_indices = array<i32: 5, 6>, parameters = [1.000000e-02, 1.000000e-03]} : (tensor<1x96x7x7xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>) -> (tensor<1x96x7x7xf32>, tensor<0xf32>, tensor<0xf32>)
    %132 = dnn.add %131#0, %117#0 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<1x96x7x7xf32>, tensor<1x96x7x7xf32>) -> tensor<1x96x7x7xf32>
    %133 = dnn.convolution %132, %cst_48 {kind = "aten.conv2d", parameter_indices = array<i32: 2, 3, 4, 5, 6>, parameters = [unit, [1, 1], [0, 0], [1, 1], 1]} : (tensor<1x96x7x7xf32>, tensor<576x96x1x1xf32>) -> tensor<1x576x7x7xf32>
    %134:3 = dnn.batch_norm %133, %cst_49, %cst_49, %cst_49, %cst_49 {kind = "aten._native_batch_norm_legit_no_training", parameter_indices = array<i32: 5, 6>, parameters = [1.000000e-02, 1.000000e-03]} : (tensor<1x576x7x7xf32>, tensor<576xf32>, tensor<576xf32>, tensor<576xf32>, tensor<576xf32>) -> (tensor<1x576x7x7xf32>, tensor<0xf32>, tensor<0xf32>)
    %135 = dnn.hardswish %134#0 : (tensor<1x576x7x7xf32>) -> tensor<1x576x7x7xf32>
    %136 = dnn.convolution %135, %cst_50 {kind = "aten.conv2d", parameter_indices = array<i32: 2, 3, 4, 5, 6>, parameters = [unit, [1, 1], [2, 2], [1, 1], 576]} : (tensor<1x576x7x7xf32>, tensor<576x1x5x5xf32>) -> tensor<1x576x7x7xf32>
    %137:3 = dnn.batch_norm %136, %cst_49, %cst_49, %cst_49, %cst_49 {kind = "aten._native_batch_norm_legit_no_training", parameter_indices = array<i32: 5, 6>, parameters = [1.000000e-02, 1.000000e-03]} : (tensor<1x576x7x7xf32>, tensor<576xf32>, tensor<576xf32>, tensor<576xf32>, tensor<576xf32>) -> (tensor<1x576x7x7xf32>, tensor<0xf32>, tensor<0xf32>)
    %138 = dnn.hardswish %137#0 : (tensor<1x576x7x7xf32>) -> tensor<1x576x7x7xf32>
    %139 = dnn.adaptive_avg_pool2d %138 {parameter_indices = array<i32: 1>, parameters = [[1, 1]]} : (tensor<1x576x7x7xf32>) -> tensor<1x576x1x1xf32>
    %140 = dnn.convolution %139, %cst_51, %cst_37 {kind = "aten.conv2d", parameter_indices = array<i32: 3, 4, 5, 6>, parameters = [[1, 1], [0, 0], [1, 1], 1]} : (tensor<1x576x1x1xf32>, tensor<144x576x1x1xf32>, tensor<144xf32>) -> tensor<1x144x1x1xf32>
    %141 = dnn.relu %140 : (tensor<1x144x1x1xf32>) -> tensor<1x144x1x1xf32>
    %142 = dnn.convolution %141, %cst_52, %cst_49 {kind = "aten.conv2d", parameter_indices = array<i32: 3, 4, 5, 6>, parameters = [[1, 1], [0, 0], [1, 1], 1]} : (tensor<1x144x1x1xf32>, tensor<576x144x1x1xf32>, tensor<576xf32>) -> tensor<1x576x1x1xf32>
    %143 = dnn.hardsigmoid %142 : (tensor<1x576x1x1xf32>) -> tensor<1x576x1x1xf32>
    %144 = dnn.mul %143, %138 : (tensor<1x576x1x1xf32>, tensor<1x576x7x7xf32>) -> tensor<1x576x7x7xf32>
    %145 = dnn.convolution %144, %cst_53 {kind = "aten.conv2d", parameter_indices = array<i32: 2, 3, 4, 5, 6>, parameters = [unit, [1, 1], [0, 0], [1, 1], 1]} : (tensor<1x576x7x7xf32>, tensor<96x576x1x1xf32>) -> tensor<1x96x7x7xf32>
    %146:3 = dnn.batch_norm %145, %cst_16, %cst_16, %cst_16, %cst_16 {kind = "aten._native_batch_norm_legit_no_training", parameter_indices = array<i32: 5, 6>, parameters = [1.000000e-02, 1.000000e-03]} : (tensor<1x96x7x7xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>) -> (tensor<1x96x7x7xf32>, tensor<0xf32>, tensor<0xf32>)
    %147 = dnn.add %146#0, %132 {parameter_indices = array<i32: 2>, parameters = [1]} : (tensor<1x96x7x7xf32>, tensor<1x96x7x7xf32>) -> tensor<1x96x7x7xf32>
    %148 = dnn.convolution %147, %cst_48 {kind = "aten.conv2d", parameter_indices = array<i32: 2, 3, 4, 5, 6>, parameters = [unit, [1, 1], [0, 0], [1, 1], 1]} : (tensor<1x96x7x7xf32>, tensor<576x96x1x1xf32>) -> tensor<1x576x7x7xf32>
    %149:3 = dnn.batch_norm %148, %cst_49, %cst_49, %cst_49, %cst_49 {kind = "aten._native_batch_norm_legit_no_training", parameter_indices = array<i32: 5, 6>, parameters = [1.000000e-02, 1.000000e-03]} : (tensor<1x576x7x7xf32>, tensor<576xf32>, tensor<576xf32>, tensor<576xf32>, tensor<576xf32>) -> (tensor<1x576x7x7xf32>, tensor<0xf32>, tensor<0xf32>)
    %150 = dnn.hardswish %149#0 : (tensor<1x576x7x7xf32>) -> tensor<1x576x7x7xf32>
    %151 = dnn.adaptive_avg_pool2d %150 {parameter_indices = array<i32: 1>, parameters = [[1, 1]]} : (tensor<1x576x7x7xf32>) -> tensor<1x576x1x1xf32>
    %collapsed = tensor.collapse_shape %151 [[0], [1, 2, 3]] : tensor<1x576x1x1xf32> into tensor<1x576xf32>
    %152 = dnn.linear %collapsed, %cst_54, %cst_55 : tensor<1x576xf32>, tensor<1024x576xf32>, tensor<1024xf32> -> tensor<1x1024xf32>
    %153 = dnn.hardswish %152 : (tensor<1x1024xf32>) -> tensor<1x1024xf32>
    %154 = dnn.linear %153, %cst_56, %cst_57 : tensor<1x1024xf32>, tensor<1000x1024xf32>, tensor<1000xf32> -> tensor<1x1000xf32>
    return %154 : tensor<1x1000xf32>
  }
}

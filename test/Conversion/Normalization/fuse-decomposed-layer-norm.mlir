// RUN: dnn-mlir-opt --convert-torch-to-dnn %s | FileCheck %s
// RUN: dnn-mlir-opt --dnn-backend-to-linalg-on-tensors-backend-pipeline='captures=dnn.layer_norm' %s | FileCheck %s --check-prefix=PIPE

func.func @fuse_layer_norm(
    %input: !torch.vtensor<[1,4,8],f32>,
    %weight: !torch.vtensor<[8],f32>,
    %bias: !torch.vtensor<[8],f32>) -> !torch.vtensor<[1,4,8],f32> {
  %dim0 = torch.constant.int 1
  %dim1 = torch.constant.int 4
  %width = torch.constant.int 8
  %none = torch.constant.none
  %true = torch.constant.bool true
  %alpha = torch.constant.int 1
  %epsilon = torch.constant.float 1.000000e-05
  %axis = torch.constant.int 2
  %axes = torch.prim.ListConstruct %axis : (!torch.int) -> !torch.list<int>
  %sum = torch.aten.sum.dim_IntList %input, %axes, %true, %none
      : !torch.vtensor<[1,4,8],f32>, !torch.list<int>, !torch.bool,
        !torch.none -> !torch.vtensor<[1,4,1],f32>
  %mean = torch.aten.div.Scalar %sum, %width
      : !torch.vtensor<[1,4,1],f32>, !torch.int
        -> !torch.vtensor<[1,4,1],f32>
  %shape = torch.prim.ListConstruct %dim0, %dim1, %width
      : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
  %mean_broadcast = torch.aten.broadcast_to %mean, %shape
      : !torch.vtensor<[1,4,1],f32>, !torch.list<int>
        -> !torch.vtensor<[1,4,8],f32>
  %centered = torch.aten.sub.Tensor %input, %mean_broadcast, %alpha
      : !torch.vtensor<[1,4,8],f32>, !torch.vtensor<[1,4,8],f32>,
        !torch.int -> !torch.vtensor<[1,4,8],f32>
  %square = torch.aten.mul.Tensor %centered, %centered
      : !torch.vtensor<[1,4,8],f32>, !torch.vtensor<[1,4,8],f32>
        -> !torch.vtensor<[1,4,8],f32>
  %variance_sum = torch.aten.sum.dim_IntList %square, %axes, %true, %none
      : !torch.vtensor<[1,4,8],f32>, !torch.list<int>, !torch.bool,
        !torch.none -> !torch.vtensor<[1,4,1],f32>
  %variance = torch.aten.div.Scalar %variance_sum, %width
      : !torch.vtensor<[1,4,1],f32>, !torch.int
        -> !torch.vtensor<[1,4,1],f32>
  %adjusted = torch.aten.add.Scalar %variance, %epsilon, %alpha
      : !torch.vtensor<[1,4,1],f32>, !torch.float, !torch.int
        -> !torch.vtensor<[1,4,1],f32>
  %inverse = torch.aten.rsqrt %adjusted
      : !torch.vtensor<[1,4,1],f32> -> !torch.vtensor<[1,4,1],f32>
  %inverse_broadcast = torch.aten.broadcast_to %inverse, %shape
      : !torch.vtensor<[1,4,1],f32>, !torch.list<int>
        -> !torch.vtensor<[1,4,8],f32>
  %normalized = torch.aten.mul.Tensor %centered, %inverse_broadcast
      : !torch.vtensor<[1,4,8],f32>, !torch.vtensor<[1,4,8],f32>
        -> !torch.vtensor<[1,4,8],f32>
  %weighted = torch.aten.mul.Tensor %normalized, %weight
      : !torch.vtensor<[1,4,8],f32>, !torch.vtensor<[8],f32>
        -> !torch.vtensor<[1,4,8],f32>
  %result = torch.aten.add.Tensor %weighted, %bias, %alpha
      : !torch.vtensor<[1,4,8],f32>, !torch.vtensor<[8],f32>, !torch.int
        -> !torch.vtensor<[1,4,8],f32>
  return %result : !torch.vtensor<[1,4,8],f32>
}

// A similar affine arithmetic tail without the reduction/variance structure
// must remain ordinary arithmetic.
func.func @reject_non_layer_norm_subgraph(
    %input: !torch.vtensor<[1,4,8],f32>,
    %weight: !torch.vtensor<[8],f32>,
    %bias: !torch.vtensor<[8],f32>) -> !torch.vtensor<[1,4,8],f32> {
  %alpha = torch.constant.int 1
  %centered = torch.aten.sub.Tensor %input, %input, %alpha
      : !torch.vtensor<[1,4,8],f32>, !torch.vtensor<[1,4,8],f32>,
        !torch.int -> !torch.vtensor<[1,4,8],f32>
  %normalized = torch.aten.mul.Tensor %centered, %centered
      : !torch.vtensor<[1,4,8],f32>, !torch.vtensor<[1,4,8],f32>
        -> !torch.vtensor<[1,4,8],f32>
  %weighted = torch.aten.mul.Tensor %normalized, %weight
      : !torch.vtensor<[1,4,8],f32>, !torch.vtensor<[8],f32>
        -> !torch.vtensor<[1,4,8],f32>
  %result = torch.aten.add.Tensor %weighted, %bias, %alpha
      : !torch.vtensor<[1,4,8],f32>, !torch.vtensor<[8],f32>, !torch.int
        -> !torch.vtensor<[1,4,8],f32>
  return %result : !torch.vtensor<[1,4,8],f32>
}

// CHECK-LABEL: func.func @fuse_layer_norm
// CHECK: dnn.layer_norm
// CHECK-SAME: parameters =
// CHECK-SAME: 8
// CHECK-SAME: 1.000000e-05
// CHECK-SAME: false
// CHECK-NOT: torch.aten.sum
// CHECK-NOT: torch.aten.rsqrt
// CHECK-LABEL: func.func @reject_non_layer_norm_subgraph
// CHECK-NOT: dnn.layer_norm

// PIPE-LABEL: func.func @fuse_layer_norm
// PIPE: dnn.layer_norm
// PIPE-NOT: linalg.generic
// PIPE-LABEL: func.func @reject_non_layer_norm_subgraph
// PIPE-NOT: dnn.layer_norm

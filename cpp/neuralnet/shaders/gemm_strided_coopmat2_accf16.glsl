// GLSL Compute Shader: Strided batched GEMM -- FP16 with NV cooperative matrix2, FP16 accumulate.
// For 1x1 convolutions and transformer projections.
// C[batch][N_real][M] += A[batch][K][M] contracted with B[batch][K][N] over K.
//
// Uses KataGo's normal unpacked FP16 strided layout:
//   A: [batch, K, M] as scalar float16_t, M-contiguous
//   B: [batch, K, N] as scalar float16_t, N-contiguous
//   C: [batch, N_real, M] as scalar float16_t, M-contiguous

#version 460

#extension GL_EXT_control_flow_attributes : enable
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_KHR_memory_scope_semantics : require
#extension GL_KHR_cooperative_matrix : require
#extension GL_NV_cooperative_matrix2 : require

#ifdef KATAGO_VULKAN_RTE_F16
#extension GL_EXT_spirv_intrinsics : require
spirv_execution_mode(extensions = ["SPV_KHR_float_controls"], capabilities = [4467], 4462, 16);
#endif

layout(constant_id = 0) const uint BLOCK_SIZE = 128;
layout(constant_id = 1) const uint BM = 64;
layout(constant_id = 2) const uint BN = 64;
layout(constant_id = 3) const uint BK = 32;
layout(constant_id = 4) const uint ALIGNED = 0;
layout(constant_id = 5) const uint ADD_TO_OUTPUT = 0;
layout(constant_id = 6) const uint PACKED_B = 0;
layout(constant_id = 7) const uint K_ALIGNED = 0;
layout(constant_id = 8) const uint K_SPEC = 1;
const uint PACKED_B_PAD_SCALARS = 8u;

layout(local_size_x_id = 0, local_size_y = 1, local_size_z = 1) in;

layout(push_constant) uniform PushConstants {
  int M;        // paddedSpatialSize
  int N;        // outChannelsPadded
  int N_real;   // outChannels, used for C layout bounds
  int strideA;  // vec4 units per batch in A
  int strideB;  // vec4 units per batch in B, zero when filter is reused
  int strideC;  // vec4 units per batch in C, real N
}
pc;

layout(binding = 0) readonly buffer ABufH {
  float16_t data[];
}
aBufH;
layout(binding = 1) readonly buffer BBufH {
  float16_t data[];
}
bBufH;
layout(binding = 2) buffer CBufH {
  float16_t data[];
}
cBufH;

void main() {
  const uint ir = gl_WorkGroupID.x;
  const uint ic = gl_WorkGroupID.y;
  const uint batch = gl_GlobalInvocationID.z;

  const uint M = uint(pc.M);
  const uint N = uint(pc.N);
  const uint NReal = uint(pc.N_real);
  const uint K = K_SPEC;

  const uint aBase = batch * uint(pc.strideA) * 4u;
  const uint bBase = batch * uint(pc.strideB) * 4u;
  const uint cBase = batch * uint(pc.strideC) * 4u;
  const uint packedBRowStride = BK + PACKED_B_PAD_SCALARS;
  const uint kBlocks = (K + BK - 1u) / BK;

  tensorLayoutNV<2> layoutA = createTensorLayoutNV(2);
  tensorLayoutNV<2, gl_CooperativeMatrixClampModeConstantNV> layoutAClamp =
    createTensorLayoutNV(2, gl_CooperativeMatrixClampModeConstantNV);
  tensorLayoutNV<2> layoutB = createTensorLayoutNV(2);
  tensorLayoutNV<2, gl_CooperativeMatrixClampModeConstantNV> layoutBClamp =
    createTensorLayoutNV(2, gl_CooperativeMatrixClampModeConstantNV);
  tensorLayoutNV<2> layoutC = createTensorLayoutNV(2);
  tensorLayoutNV<2, gl_CooperativeMatrixClampModeConstantNV> layoutCClamp =
    createTensorLayoutNV(2, gl_CooperativeMatrixClampModeConstantNV);

  // INVARIANT (NV_cooperative_matrix2 spec): every tensorLayoutNV<N> below must
  // have canonical stride order -- dim0 is the outer (larger-stride) axis and
  // the innermost dim (dim(N-1)) has stride 1. The spec requires
  // stride[i] >= stride[i+1] * ceil(dim[i+1] / blockSize[i+1]). Violating it
  // silently corrupts partial-tile results: OOB rows under
  // gl_CooperativeMatrixClampModeConstantNV return garbage instead of zero,
  // which manifests as rmse=inf/hundreds on partial-tile candidates while
  // full-tile candidates keep passing (making the bug look like a partial-tile
  // compute issue). If a coopmat orientation demands the opposite axis order
  // (e.g. matUseA is [M,K] but physical layout has M contiguous), permute via
  // `transposeView` on the load/store -- DO NOT swap the dimension/stride
  // arguments to setTensorLayout*NV.
  layoutA = setTensorLayoutDimensionNV(layoutA, K, M);
  layoutA = setTensorLayoutStrideNV(layoutA, M, 1);
  layoutAClamp = setTensorLayoutDimensionNV(layoutAClamp, K, M);
  layoutAClamp = setTensorLayoutStrideNV(layoutAClamp, M, 1);
  if(PACKED_B != 0u) {
    // Packed B is tile-local logical (K,N) over physical [N][BK+pad] with BK
    // inner. Canonical: dim0=BN outer, dim1=BK inner (stride 1).
    layoutB = setTensorLayoutDimensionNV(layoutB, BN, BK);
    layoutB = setTensorLayoutStrideNV(layoutB, packedBRowStride, 1);
    layoutBClamp = setTensorLayoutDimensionNV(layoutBClamp, BN, BK);
    layoutBClamp = setTensorLayoutStrideNV(layoutBClamp, packedBRowStride, 1);
  } else {
    // B physical [K][N] with N inner (unit-stride). Canonical: dim0=K outer,
    // dim1=N inner. Note: this maps to coopmat matUseB[K,N] directly with no
    // transpose view.
    layoutB = setTensorLayoutDimensionNV(layoutB, K, N);
    layoutB = setTensorLayoutStrideNV(layoutB, N, 1);
    layoutBClamp = setTensorLayoutDimensionNV(layoutBClamp, K, N);
    layoutBClamp = setTensorLayoutStrideNV(layoutBClamp, N, 1);
  }
  // C physical [NReal][M] with M inner (unit-stride). Canonical: dim0=NReal
  // outer, dim1=M inner. Coopmat matUseAccumulator is [M,N], so store through
  // the transpose view.
  layoutC = setTensorLayoutDimensionNV(layoutC, NReal, M);
  layoutC = setTensorLayoutStrideNV(layoutC, M, 1);
  layoutCClamp = setTensorLayoutDimensionNV(layoutCClamp, NReal, M);
  layoutCClamp = setTensorLayoutStrideNV(layoutCClamp, M, 1);
  tensorViewNV<2, false, 1, 0> transposeView = createTensorViewNV(2, false, 1, 0);

  coopmat<float16_t, gl_ScopeWorkgroup, BM, BN, gl_MatrixUseAccumulator> acc =
    coopmat<float16_t, gl_ScopeWorkgroup, BM, BN, gl_MatrixUseAccumulator>(0.0hf);

  if(ADD_TO_OUTPUT != 0u) {
    coopmat<float16_t, gl_ScopeWorkgroup, BM, BN, gl_MatrixUseAccumulator> c0;
    coopMatLoadTensorNV(
      c0, cBufH.data, cBase, sliceTensorLayoutNV(layoutCClamp, ic * BN, BN, ir * BM, BM), transposeView);
    acc = c0;
  }

  for(uint kb = 0; kb < K; kb += BK) {
    coopmat<float16_t, gl_ScopeWorkgroup, BM, BK, gl_MatrixUseA> a;
    coopmat<float16_t, gl_ScopeWorkgroup, BK, BN, gl_MatrixUseB> b;
    const uint bPackedTileBase = bBase + (ic * kBlocks + kb / BK) * BN * packedBRowStride;
    const bool fullKTile = (K_ALIGNED != 0u || kb + BK <= K);

    // Canonical A layout: dim0=K outer, dim1=M inner (unit stride). Coopmat
    // matUseA is [M,K], so load through the transpose view to permute
    // dimensions.
    if(ALIGNED != 0u && fullKTile) {
      coopMatLoadTensorNV(a, aBufH.data, aBase, sliceTensorLayoutNV(layoutA, kb, BK, ir * BM, BM), transposeView);
      if(PACKED_B != 0u)
        coopMatLoadTensorNV(
          b, bBufH.data, bPackedTileBase, sliceTensorLayoutNV(layoutB, 0u, BN, 0u, BK), transposeView);
      else
        coopMatLoadTensorNV(b, bBufH.data, bBase, sliceTensorLayoutNV(layoutB, kb, BK, ic * BN, BN));
    } else {
      coopMatLoadTensorNV(a, aBufH.data, aBase, sliceTensorLayoutNV(layoutAClamp, kb, BK, ir * BM, BM), transposeView);
      if(PACKED_B != 0u)
        coopMatLoadTensorNV(
          b, bBufH.data, bPackedTileBase, sliceTensorLayoutNV(layoutBClamp, 0u, BN, 0u, BK), transposeView);
      else
        coopMatLoadTensorNV(b, bBufH.data, bBase, sliceTensorLayoutNV(layoutBClamp, kb, BK, ic * BN, BN));
    }
    acc = coopMatMulAdd(a, b, acc);
  }

  if(ALIGNED != 0u && ic * BN + BN <= NReal)
    coopMatStoreTensorNV(acc, cBufH.data, cBase, sliceTensorLayoutNV(layoutC, ic * BN, BN, ir * BM, BM), transposeView);
  else
    coopMatStoreTensorNV(
      acc, cBufH.data, cBase, sliceTensorLayoutNV(layoutCClamp, ic * BN, BN, ir * BM, BM), transposeView);
}

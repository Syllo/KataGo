// GLSL Compute Shader: Winograd batched GEMM -- FP16 with FP32 NV cooperative-matrix2 accumulation.
// C[batch][N][M] = A[batch][K][M] contracted with B[batch][K][N] over K.
//
// Uses VK_NV_cooperative_matrix2 workgroup-scope fragments and tensor layouts
// over KataGo's packed tile-local Winograd A/B layouts:
//   A: [batch, M tile, K tile, M, K/4+pad]
//   B: [batch, N tile, K tile, K, N/4+pad]
//   C: [batch, N, M] as scalar float16_t, M-contiguous
//
// Push-constant strides are preserved from the existing GEMM ABI in vec4 units.
// Tensor-layout bases and strides are scalar half-addressed.

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
layout(constant_id = 4) const uint K_SPEC = 1;

// Must match WINOGRAD_COOPMAT2_PACKED_PAD_WORDS. Tried 0, 1, 2, and 4; 2 and 4
// were equally best, so use the smaller 2-word pad.
const uint PACKED_A_PAD_WORDS = 2u;
const uint PACKED_B_PAD_WORDS = 2u;

layout(local_size_x_id = 0, local_size_y = 1, local_size_z = 1) in;

layout(push_constant) uniform PushConstants {
  int M;        // numTilesPaddedRun
  int N;        // numOutChannelsPadded
  int strideA;  // vec4 units between Winograd batches in A
  int strideB;  // vec4 units between Winograd batches in B
  int strideC;  // vec4 units between Winograd batches in C
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
layout(binding = 2) writeonly buffer CBufH {
  float16_t data[];
}
cBufH;

void main() {
  const uint ir = gl_WorkGroupID.x;
  const uint ic = gl_WorkGroupID.y;
  const uint batch = gl_GlobalInvocationID.z;

  const uint M = uint(pc.M);
  const uint N = uint(pc.N);
  const uint K = K_SPEC;

  const uint aBaseScalar = batch * uint(pc.strideA) * 4u;
  const uint bBaseScalar = batch * uint(pc.strideB) * 4u;
  const uint cBase = batch * uint(pc.strideC) * 4u;

  tensorLayoutNV<2> layoutA = createTensorLayoutNV(2);
  tensorLayoutNV<2> layoutB = createTensorLayoutNV(2);
  tensorLayoutNV<2> layoutC = createTensorLayoutNV(2);

  const uint packedARowWords = BK / 4u + PACKED_A_PAD_WORDS;
  const uint packedARowStride = packedARowWords * 4u;
  const uint packedBRowWords = BN / 4u + PACKED_B_PAD_WORDS;
  const uint packedBRowStride = packedBRowWords * 4u;
  const uint kBlocks = (K + BK - 1u) / BK;

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
  // Packed A is tile-local logical (M,K) over physical [M][K/4+pad].
  layoutA = setTensorLayoutDimensionNV(layoutA, BM, BK);
  layoutA = setTensorLayoutStrideNV(layoutA, packedARowStride, 1);
  // Packed B is tile-local logical (K,N) over physical [K][N/4+pad].
  layoutB = setTensorLayoutDimensionNV(layoutB, BK, BN);
  layoutB = setTensorLayoutStrideNV(layoutB, packedBRowStride, 1);

  // C physical [N][M] with M inner (unit-stride). Canonical: dim0=N outer,
  // dim1=M inner. Coopmat matUseAccumulator is [M,N], so store through the
  // transpose view.
  layoutC = setTensorLayoutDimensionNV(layoutC, N, M);
  layoutC = setTensorLayoutStrideNV(layoutC, M, 1);
  tensorViewNV<2, false, 1, 0> transposeView = createTensorViewNV(2, false, 1, 0);

  coopmat<float, gl_ScopeWorkgroup, BM, BN, gl_MatrixUseAccumulator> acc =
    coopmat<float, gl_ScopeWorkgroup, BM, BN, gl_MatrixUseAccumulator>(0.0);

  for(uint kb = 0; kb < K; kb += BK) {
    coopmat<float16_t, gl_ScopeWorkgroup, BM, BK, gl_MatrixUseA> a;
    coopmat<float16_t, gl_ScopeWorkgroup, BK, BN, gl_MatrixUseB> b;
    const uint aPackedTileBase = aBaseScalar + (ir * kBlocks + kb / BK) * BM * packedARowStride;
    const uint bPackedTileBase = bBaseScalar + (ic * kBlocks + kb / BK) * BK * packedBRowStride;

    coopMatLoadTensorNV(a, aBufH.data, aPackedTileBase, sliceTensorLayoutNV(layoutA, 0u, BM, 0u, BK));
    coopMatLoadTensorNV(b, bBufH.data, bPackedTileBase, sliceTensorLayoutNV(layoutB, 0u, BK, 0u, BN));
    acc = coopMatMulAdd(a, b, acc);
  }

  coopmat<float16_t, gl_ScopeWorkgroup, BM, BN, gl_MatrixUseAccumulator> matC =
    coopmat<float16_t, gl_ScopeWorkgroup, BM, BN, gl_MatrixUseAccumulator>(acc);
  coopMatStoreTensorNV(matC, cBufH.data, cBase, sliceTensorLayoutNV(layoutC, ic * BN, BN, ir * BM, BM), transposeView);
}

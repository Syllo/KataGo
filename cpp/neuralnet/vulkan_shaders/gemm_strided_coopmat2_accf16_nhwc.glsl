// Native-NHWC strided GEMM using NV cooperative matrix2 and FP16 accumulation.
// A/C are physically and logically [M,K]/[M,N]. Packed B is [BK][BN+8].
#version 460

#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_KHR_memory_scope_semantics : require
#extension GL_KHR_cooperative_matrix : require
#extension GL_NV_cooperative_matrix2 : require

#ifdef KATAGO_VULKAN_RTE_F16
#extension GL_EXT_spirv_intrinsics : require
spirv_execution_mode(extensions = ["SPV_KHR_float_controls"], capabilities = [4467], 4462, 16);
#endif

layout(constant_id = 0) const uint WORKGROUP_SIZE = 128;
layout(constant_id = 1) const uint BM = 64;
layout(constant_id = 2) const uint BN = 64;
layout(constant_id = 3) const uint BK = 32;
layout(constant_id = 4) const uint ALIGNED = 0;
layout(constant_id = 5) const uint ADD_TO_OUTPUT = 0;
layout(constant_id = 7) const uint K_ALIGNED = 0;
layout(constant_id = 8) const uint K_SPEC = 1;
layout(local_size_x_id = 0, local_size_y = 1, local_size_z = 1) in;

const uint PACKED_B_PAD_SCALARS = 8u;
layout(push_constant) uniform PushConstants {
  int M, N, N_real, strideA, strideB, strideC;
}
pc;
layout(binding = 0) readonly buffer ABuf {
  float16_t data[];
}
aBuf;
layout(binding = 1) readonly buffer BBuf {
  float16_t data[];
}
bBuf;
layout(binding = 2) buffer CBuf {
  float16_t data[];
}
cBuf;

void main() {
  const uint ir = gl_WorkGroupID.x, ic = gl_WorkGroupID.y, batch = gl_WorkGroupID.z;
  const uint M = uint(pc.M), N = uint(pc.N), NReal = uint(pc.N_real), K = K_SPEC;
  const uint aBase = batch * uint(pc.strideA) * 4u;
  const uint bBase = batch * uint(pc.strideB) * 4u;
  const uint cBase = batch * uint(pc.strideC) * 4u;
  const uint packedBStride = BN + PACKED_B_PAD_SCALARS;
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

  // All layouts are canonical: outer dimension first and unit-stride inner.
  layoutA = setTensorLayoutDimensionNV(layoutA, M, K);
  layoutA = setTensorLayoutStrideNV(layoutA, K, 1);
  layoutAClamp = setTensorLayoutDimensionNV(layoutAClamp, M, K);
  layoutAClamp = setTensorLayoutStrideNV(layoutAClamp, K, 1);
  layoutB = setTensorLayoutDimensionNV(layoutB, BK, BN);
  layoutB = setTensorLayoutStrideNV(layoutB, packedBStride, 1);
  layoutBClamp = setTensorLayoutDimensionNV(layoutBClamp, BK, BN);
  layoutBClamp = setTensorLayoutStrideNV(layoutBClamp, packedBStride, 1);
  layoutC = setTensorLayoutDimensionNV(layoutC, M, NReal);
  layoutC = setTensorLayoutStrideNV(layoutC, NReal, 1);
  layoutCClamp = setTensorLayoutDimensionNV(layoutCClamp, M, NReal);
  layoutCClamp = setTensorLayoutStrideNV(layoutCClamp, NReal, 1);
  layoutAClamp = setTensorLayoutClampValueNV(layoutAClamp, 0u);
  layoutBClamp = setTensorLayoutClampValueNV(layoutBClamp, 0u);
  layoutCClamp = setTensorLayoutClampValueNV(layoutCClamp, 0u);
  coopmat<float16_t, gl_ScopeWorkgroup, BM, BN, gl_MatrixUseAccumulator> acc =
    coopmat<float16_t, gl_ScopeWorkgroup, BM, BN, gl_MatrixUseAccumulator>(0.0hf);
  if(ADD_TO_OUTPUT != 0u)
    coopMatLoadTensorNV(acc, cBuf.data, cBase, sliceTensorLayoutNV(layoutCClamp, ir * BM, BM, ic * BN, BN));

  for(uint kb = 0u; kb < K; kb += BK) {
    coopmat<float16_t, gl_ScopeWorkgroup, BM, BK, gl_MatrixUseA> a;
    coopmat<float16_t, gl_ScopeWorkgroup, BK, BN, gl_MatrixUseB> b;
    const uint bTileBase = bBase + (ic * kBlocks + kb / BK) * BK * packedBStride;
    const bool full = ALIGNED != 0u && (K_ALIGNED != 0u || kb + BK <= K);
    if(full) {
      coopMatLoadTensorNV(a, aBuf.data, aBase, sliceTensorLayoutNV(layoutA, ir * BM, BM, kb, BK));
      coopMatLoadTensorNV(b, bBuf.data, bTileBase, sliceTensorLayoutNV(layoutB, 0u, BK, 0u, BN));
    } else {
      coopMatLoadTensorNV(a, aBuf.data, aBase, sliceTensorLayoutNV(layoutAClamp, ir * BM, BM, kb, BK));
      coopMatLoadTensorNV(b, bBuf.data, bTileBase, sliceTensorLayoutNV(layoutBClamp, 0u, BK, 0u, BN));
    }
    acc = coopMatMulAdd(a, b, acc);
  }
  if(ALIGNED != 0u && ic * BN + BN <= NReal)
    coopMatStoreTensorNV(acc, cBuf.data, cBase, sliceTensorLayoutNV(layoutC, ir * BM, BM, ic * BN, BN));
  else
    coopMatStoreTensorNV(acc, cBuf.data, cBase, sliceTensorLayoutNV(layoutCClamp, ir * BM, BM, ic * BN, BN));
}

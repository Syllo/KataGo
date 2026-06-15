// Strided 1x1 GEMM for native NHWC activations with FP32 cooperative-matrix accumulation. A is [batch, M, K], B uses
// the normal packed coopmat layout, and C is [batch, M, N_real].
//
// NHWC makes eight consecutive K values a contiguous 16-byte word. Stage A
// as packed half8 words in [M][K/8] order, so a workgroup performs one uvec4
// load/store per eight channels and coopMatLoad reads it directly as RowMajor.
// This is the same layout used by the validated NHWC vec8 implicit-3x3 path.
#version 450

#extension GL_EXT_control_flow_attributes : enable
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_KHR_memory_scope_semantics : require
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_cooperative_matrix : require

#ifdef KATAGO_VULKAN_RTE_F16
#extension GL_EXT_spirv_intrinsics : require
spirv_execution_mode(extensions = ["SPV_KHR_float_controls"], capabilities = [4467], 4462, 16);
#endif

layout(constant_id = 0) const uint WORKGROUP_SIZE = 128;
layout(constant_id = 1) const uint BM = 64;
layout(constant_id = 2) const uint BN = 64;
layout(constant_id = 3) const uint BK = 16;
layout(constant_id = 4) const uint SGM = 32;
layout(constant_id = 5) const uint SGN = 32;
layout(constant_id = 6) const uint TM = 16;
layout(constant_id = 7) const uint TN = 16;
layout(constant_id = 8) const uint TK = 16;
layout(constant_id = 9) const uint SUBGROUP_SIZE = 32;
layout(constant_id = 10) const uint ALIGNED = 0;
layout(constant_id = 11) const uint ADD_TO_OUTPUT = 0;
layout(constant_id = 14) const uint K_SPEC = 1;
layout(local_size_x_id = 0, local_size_y = 1, local_size_z = 1) in;

#define SHMEM_B_STRIDE (BN + 8u)
#define SHMEM_A_PAD_V8 1u
#define SHMEM_A_STRIDE_V8 ((BK / 8u) + SHMEM_A_PAD_V8)
layout(push_constant) uniform PushConstants {
  int M, N, N_real, strideA, strideB, strideC;
}
pc;
// K-aligned layers use the packed view directly. The scalar alias constructs
// the final half8 for channel-tail layers without reading across an M row.
layout(binding = 0) readonly buffer ABufV8 {
  uvec4 data[];
}
aBufV8;
layout(binding = 0) readonly buffer ABufH {
  float16_t data[];
}
aBufH;
layout(binding = 1) readonly buffer BBufV4H {
  f16vec4 data[];
}
bBufV4H;
layout(binding = 2) buffer CBufH {
  float16_t data[];
}
cBufH;
shared uvec4 buf_a[BM * SHMEM_A_STRIDE_V8];
shared float16_t buf_b[BK * SHMEM_B_STRIDE];
shared float coopStage[(WORKGROUP_SIZE / SUBGROUP_SIZE) * TM * TN];

void storeC(uint i, float v) {
  if(ADD_TO_OUTPUT != 0u)
    v += float(cBufH.data[i]);
  cBufH.data[i] = float16_t(v);
}

uint packHalfPair(float16_t lo, float16_t hi) {
  return packHalf2x16(vec2(float(lo), float(hi)));
}

uvec4 loadATail(uint aBase, uint gm, uint gk, uint K) {
  float16_t h0 = gk + 0u < K ? aBufH.data[aBase + gm * K + gk + 0u] : 0.0hf;
  float16_t h1 = gk + 1u < K ? aBufH.data[aBase + gm * K + gk + 1u] : 0.0hf;
  float16_t h2 = gk + 2u < K ? aBufH.data[aBase + gm * K + gk + 2u] : 0.0hf;
  float16_t h3 = gk + 3u < K ? aBufH.data[aBase + gm * K + gk + 3u] : 0.0hf;
  float16_t h4 = gk + 4u < K ? aBufH.data[aBase + gm * K + gk + 4u] : 0.0hf;
  float16_t h5 = gk + 5u < K ? aBufH.data[aBase + gm * K + gk + 5u] : 0.0hf;
  float16_t h6 = gk + 6u < K ? aBufH.data[aBase + gm * K + gk + 6u] : 0.0hf;
  float16_t h7 = gk + 7u < K ? aBufH.data[aBase + gm * K + gk + 7u] : 0.0hf;
  return uvec4(packHalfPair(h0, h1), packHalfPair(h2, h3), packHalfPair(h4, h5), packHalfPair(h6, h7));
}

void main() {
  const uint ir = gl_WorkGroupID.x, ic = gl_WorkGroupID.y, batch = gl_GlobalInvocationID.z;
  const uint tid = gl_LocalInvocationID.x, lane = gl_SubgroupInvocationID, subgroup_i = gl_SubgroupID;
  const uint cmsPerRow = SGM / TM, cmsPerCol = SGN / TN;
  const uint subgroup_r = subgroup_i % (BM / SGM), subgroup_c = subgroup_i / (BM / SGM);
  // strideA/C retain the common vec4-unit ABI. NHWC makes K contiguous at each M.
  const uint aBase = batch * uint(pc.strideA) * 4u;
  const uint bBase = batch * uint(pc.strideB);
  const uint cBase = batch * uint(pc.strideC) * 4u;
  const uint kBlocks = (K_SPEC + BK - 1u) / BK;
  coopmat<float, gl_ScopeSubgroup, TM, TN, gl_MatrixUseAccumulator> sums[(SGM / TM) * (SGN / TN)];
  [[unroll]] for(uint i = 0; i < cmsPerRow * cmsPerCol; i++)
    sums[i] = coopmat<float, gl_ScopeSubgroup, TM, TN, gl_MatrixUseAccumulator>(0.0);
  const uint aPacksPerRow = BK / 8u;
  const uint aCountV8 = BM * aPacksPerRow, bCount = (BK * SHMEM_B_STRIDE) / 4u;
  const uint A_PER_THREAD_V8 = (aCountV8 + WORKGROUP_SIZE - 1u) / WORKGROUP_SIZE;
  const uint B_PER_THREAD = (bCount + WORKGROUP_SIZE - 1u) / WORKGROUP_SIZE;
  uvec4 aPrefetch[(BM * (BK / 8u) + WORKGROUP_SIZE - 1u) / WORKGROUP_SIZE];
  f16vec4 bPrefetch[((BK * SHMEM_B_STRIDE) / 4u + WORKGROUP_SIZE - 1u) / WORKGROUP_SIZE];
#define LOAD_A(blk) \
  { \
    [[unroll]] for(uint s = 0u; s < A_PER_THREAD_V8; s++) { \
      const uint i = tid + s * WORKGROUP_SIZE; \
      if(i < aCountV8) { \
        const uint kk8 = i % aPacksPerRow, mm = i / aPacksPerRow; \
        const uint gk = (blk) + kk8 * 8u, gm = ir * BM + mm; \
        const bool validM = ALIGNED != 0u || gm < uint(pc.M); \
        uvec4 packed = uvec4(0); \
        if(validM) { \
          if((K_SPEC % 8u) == 0u) { \
            if(gk < K_SPEC) \
              packed = aBufV8.data[(aBase + gm * K_SPEC + gk) / 8u]; \
          } else if(gk < K_SPEC) { \
            packed = loadATail(aBase, gm, gk, K_SPEC); \
          } \
        } \
        aPrefetch[s] = packed; \
      } \
    } \
  }
#define LOAD_B(blk) \
  { \
    uint tileBase = (ic * kBlocks + (blk) / BK) * bCount; \
    [[unroll]] for(uint s = 0u; s < B_PER_THREAD; s++) { \
      uint i = tid + s * WORKGROUP_SIZE; \
      if(i < bCount) \
        bPrefetch[s] = bBufV4H.data[bBase + tileBase + i]; \
    } \
  }
#define COMMIT_A \
  { \
    [[unroll]] for(uint s = 0u; s < A_PER_THREAD_V8; s++) { \
      uint i = tid + s * WORKGROUP_SIZE; \
      if(i < aCountV8) { \
        uint kk8 = i % aPacksPerRow, mm = i / aPacksPerRow; \
        buf_a[mm * SHMEM_A_STRIDE_V8 + kk8] = aPrefetch[s]; \
      } \
    } \
  }
#define COMMIT_B \
  { \
    [[unroll]] for(uint s = 0u; s < B_PER_THREAD; s++) { \
      uint i = tid + s * WORKGROUP_SIZE; \
      if(i < bCount) { \
        f16vec4 v = bPrefetch[s]; \
        uint d = i * 4u; \
        buf_b[d] = v.x; \
        buf_b[d + 1u] = v.y; \
        buf_b[d + 2u] = v.z; \
        buf_b[d + 3u] = v.w; \
      } \
    } \
  }
  LOAD_A(0u);
  LOAD_B(0u);
  COMMIT_A;
  COMMIT_B;
  barrier();
  for(uint block = 0u; block < K_SPEC; block += BK) {
    uint next = block + BK;
    bool hasNext = next < K_SPEC;
    if(hasNext) {
      LOAD_A(next);
      LOAD_B(next);
    }
    [[unroll]] for(uint ko = 0u; ko < BK; ko += TK) {
      coopmat<float16_t, gl_ScopeSubgroup, TM, TK, gl_MatrixUseA> ca[SGM / TM];
      coopmat<float16_t, gl_ScopeSubgroup, TK, TN, gl_MatrixUseB> cb[SGN / TN];
      [[unroll]] for(uint r = 0u; r < cmsPerRow; r++)
        coopMatLoad(
          ca[r],
          buf_a,
          (subgroup_r * SGM + r * TM) * SHMEM_A_STRIDE_V8 + ko / 8u,
          SHMEM_A_STRIDE_V8,
          gl_CooperativeMatrixLayoutRowMajor);
      [[unroll]] for(uint c = 0u; c < cmsPerCol; c++)
        coopMatLoad(
          cb[c], buf_b, ko * SHMEM_B_STRIDE + subgroup_c * SGN + c * TN, SHMEM_B_STRIDE, gl_CooperativeMatrixLayoutRowMajor);
      [[unroll]] for(uint r = 0u; r < cmsPerRow; r++) [[unroll]]
        for(uint c = 0u; c < cmsPerCol; c++)
          sums[r * cmsPerCol + c] = coopMatMulAdd(ca[r], cb[c], sums[r * cmsPerCol + c]);
    }
    if(hasNext) {
      barrier();
      COMMIT_A;
      COMMIT_B;
      barrier();
    }
  }
#undef LOAD_A
#undef LOAD_B
#undef COMMIT_A
#undef COMMIT_B
  uint stageBase = subgroup_i * TM * TN;
  [[unroll]] for(uint r = 0u; r < cmsPerRow; r++) [[unroll]]
    for(uint c = 0u; c < cmsPerCol; c++) {
      uint m0 = ir * BM + subgroup_r * SGM + r * TM, n0 = ic * BN + subgroup_c * SGN + c * TN;
      uint si = r * cmsPerCol + c;
      if(m0 + TM <= uint(pc.M) && n0 + TN <= uint(pc.N_real)) {
        coopmat<float, gl_ScopeSubgroup, TM, TN, gl_MatrixUseAccumulator> result = sums[si];
        // NHWC C is [M,N], so it is directly row-major.
        if(ADD_TO_OUTPUT != 0u) {
          coopmat<float16_t, gl_ScopeSubgroup, TM, TN, gl_MatrixUseAccumulator> old;
          coopMatLoad(
            old, cBufH.data, cBase + m0 * uint(pc.N_real) + n0, uint(pc.N_real), gl_CooperativeMatrixLayoutRowMajor);
          result = result + coopmat<float, gl_ScopeSubgroup, TM, TN, gl_MatrixUseAccumulator>(old);
        }
        coopmat<float16_t, gl_ScopeSubgroup, TM, TN, gl_MatrixUseAccumulator> outFrag =
          coopmat<float16_t, gl_ScopeSubgroup, TM, TN, gl_MatrixUseAccumulator>(result);
        coopMatStore(
          outFrag, cBufH.data, cBase + m0 * uint(pc.N_real) + n0, uint(pc.N_real), gl_CooperativeMatrixLayoutRowMajor);
      } else {
        coopMatStore(sums[si], coopStage, stageBase, TN, gl_CooperativeMatrixLayoutRowMajor);
        controlBarrier(gl_ScopeSubgroup, gl_ScopeSubgroup, gl_StorageSemanticsShared, gl_SemanticsAcquireRelease);
        for(uint i = lane; i < TM * TN; i += SUBGROUP_SIZE) {
          uint ml = i % TM, nl = i / TM, m = m0 + ml, n = n0 + nl;
          if(m < uint(pc.M) && n < uint(pc.N_real))
            storeC(cBase + m * uint(pc.N_real) + n, coopStage[stageBase + ml * TN + nl]);
        }
        controlBarrier(gl_ScopeSubgroup, gl_ScopeSubgroup, gl_StorageSemanticsShared, gl_SemanticsAcquireRelease);
      }
    }
}

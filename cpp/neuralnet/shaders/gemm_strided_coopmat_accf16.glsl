// GLSL Compute Shader: Strided batched GEMM — FP16 with KHR cooperative matrix (coopmat1, FP16 accumulate).
// For 1x1 convolutions and transformer projections.
// C[batch][N_real][M] += A[batch][K][M] contracted with B[batch][K][N] over K.
//
// Uses VK_KHR_cooperative_matrix subgroup-scope fragments (ggml mul_mm.comp COOPMAT shape).
// The A/B operands are staged into shared memory, then coopMatLoad'd as fragments
// and accumulated with coopMatMulAdd into an FP16 accumulator.
//
// Memory layout (KataGo stores A/B K-outer, free dim contiguous):
//   A: [batch, K, M/8] as raw half8/uvec4 (pack8-M) — 8 consecutive M at a fixed K.
//   B: [batch, K, N/4] as f16vec4   (pack4-N) — 4 consecutive N at a fixed K.
//   C: [batch, N_real, M] as float16_t scalar (M-contiguous).
// A is staged as packed half8 words in K-major order and loaded as ColumnMajor so
// coopMatLoad interprets the packed M dimension as fragment rows. B is staged in
// the legacy [free-dim][K] scalar layout so packed host weights stay unchanged.
// M always ÷8; N always ÷8; K unconstrained.
//
// TM/TN/TK are the device's reported coopmat fragment shape (MSize/NSize/KSize) and are
// set by the host from the enumerated VkCooperativeMatrixPropertiesKHR — NOT free knobs.
//
// Dispatch: global(ceil(M/BM), ceil(N/BN), batch)

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

layout(constant_id = 0) const uint BLOCK_SIZE = 128;
layout(constant_id = 1) const uint BM = 64;
layout(constant_id = 2) const uint BN = 64;
layout(constant_id = 3) const uint BK = 16;
layout(constant_id = 4) const uint WM = 32;
layout(constant_id = 5) const uint WN = 32;
layout(constant_id = 6) const uint TM = 16;  // coopmat MSize
layout(constant_id = 7) const uint TN = 16;  // coopmat NSize
layout(constant_id = 8) const uint TK = 16;  // coopmat KSize
layout(constant_id = 9) const uint WARP = 32;
// ALIGNED != 0 means padded dims M/N are guaranteed multiples of BM/BN, so M/N
// bounds checks on A/B loads compile away. K_ALIGNED separately means K is a
// multiple of BK, so A-side K-tail guards can compile away. Packed B zero-fills
// K tails on the host. The N_real C-store guard stays regardless: real
// outChannels < padded outChannels is what N_real exists to protect.
layout(constant_id = 10) const uint ALIGNED = 0;
layout(constant_id = 11) const uint ADD_TO_OUTPUT = 0;
layout(constant_id = 12) const uint PACKED_B = 0;
layout(constant_id = 13) const uint K_ALIGNED = 0;
layout(constant_id = 14) const uint K_SPEC = 1;

layout(local_size_x_id = 0, local_size_y = 1, local_size_z = 1) in;

// Shared B is scalar f16, [N][K]. Match ggml's coopmat padding budget
// (+4 f16vec2 slots, i.e. +8 half scalars) to reduce bank conflicts.
#define SHMEM_STRIDE (BK + 8u)

// Shared A is packed half8 [K][M/8 + pad] in raw uvec4 words.
#define SHMEM_A_PAD_V8 2u
#define SHMEM_A_STRIDE_V8 ((BM / 8u) + SHMEM_A_PAD_V8)

layout(push_constant) uniform PushConstants {
  int M;        // paddedSpatialSize (always ÷8)
  int N;        // outChannelsPadded (always ÷8)
  int N_real;   // outChannels (real, unpadded) — C-store guard
  int strideA;  // K*(M/4) — vec4 units per batch in A
  int strideB;  // K*(N/4) — vec4 units per batch in B
  int strideC;  // N_real*(M/4) — vec4 units per batch in C (real N, not padded)
}
pc;

// A: raw pack8-M half8 view [batch, K, M/8]. strideA is in vec4 units; /2 gives half8 words.
layout(binding = 0) readonly buffer ABufV8H {
  uvec4 data[];
}
aBufV8H;
// B: pack4-N f16vec4 view [batch, K, N/4]. strideB is in vec4 units.
layout(binding = 1) readonly buffer BBufV4H {
  f16vec4 data[];
}
bBufV4H;
// C: scalar float16_t [batch, N_real, M] (M-contiguous). strideC is in vec4 units.
layout(binding = 2) buffer CBufH {
  float16_t data[];
}
cBufH;

void storeC(uint idx, float16_t val) {
  if(ADD_TO_OUTPUT != 0u)
    val += cBufH.data[idx];
  cBufH.data[idx] = val;
}

shared uvec4 buf_a[BK * SHMEM_A_STRIDE_V8];  // [K][M/8 + pad], M half8-inner
shared float16_t buf_b[BN * SHMEM_STRIDE];   // [N][K], K inner
// Store scratch: one TM*TN fragment per warp, staged then scalar-copied to C with guards.
shared float16_t coopStage[(BLOCK_SIZE / WARP) * TM * TN];

void main() {
  const uint ic = gl_WorkGroupID.y;
  const uint ir = gl_WorkGroupID.x;
  const uint batch = gl_GlobalInvocationID.z;

  const uint cmsPerRow = WM / TM;  // coopmat fragments along M within a warp tile
  const uint cmsPerCol = WN / TN;  // coopmat fragments along N within a warp tile

  // Subgroup identity from the real built-ins (coopmat fragments are subgroup-scoped).
  const uint warp_i = gl_SubgroupID;
  const uint lane = gl_SubgroupInvocationID;
  const uint warp_r = warp_i % (BM / WM);
  const uint warp_c = warp_i / (BM / WM);

  const uint tid = gl_LocalInvocationID.x;

  // Base offsets.
  const uint aBase_v8 = (batch * uint(pc.strideA)) / 2u;  // raw half8 units
  const uint bBase_v4 = batch * uint(pc.strideB);         // f16vec4 units
  const uint cBase = batch * uint(pc.strideC) * 4u;       // float16_t units (1 vec4 = 4 f16)
  const uint kBlocks = (K_SPEC + BK - 1u) / BK;

  // Accumulators: cmsPerRow x cmsPerCol FP16 fragments per warp.
  coopmat<float16_t, gl_ScopeSubgroup, TM, TN, gl_MatrixUseAccumulator> sums[(WM / TM) * (WN / TN)];
  [[unroll]] for(uint i = 0; i < cmsPerRow * cmsPerCol; i++)
    sums[i] = coopmat<float16_t, gl_ScopeSubgroup, TM, TN, gl_MatrixUseAccumulator>(0.0hf);

  const uint aCount = BK * (BM / 8u);
  const uint bPackedCount = (BN * SHMEM_STRIDE) / 4u;
  const uint bUnpackedCount = BK * (BN / 4u);
  const uint bCount = (PACKED_B != 0u) ? bPackedCount : bUnpackedCount;
  const uint A_PER_THREAD = (aCount + BLOCK_SIZE - 1u) / BLOCK_SIZE;
  const uint B_PER_THREAD = (bCount + BLOCK_SIZE - 1u) / BLOCK_SIZE;

  uvec4 aPrefetch[(BK * (BM / 8u) + BLOCK_SIZE - 1u) / BLOCK_SIZE];
  f16vec4 bPrefetch[((BN * SHMEM_STRIDE) / 4u + BLOCK_SIZE - 1u) / BLOCK_SIZE];

#define LOAD_A_TO_REGS(blk) \
  { \
    [[unroll]] for(uint s = 0; s < A_PER_THREAD; s++) { \
      const uint i = tid + s * BLOCK_SIZE; \
      if(i < aCount) { \
        const uint kk = i / (BM / 8u); \
        const uint mm8 = i % (BM / 8u); \
        const uint gk = (blk) + kk; \
        const uint gm8 = (ir * BM) / 8u + mm8; \
        const bool validK = (K_ALIGNED != 0u || gk < K_SPEC); \
        const bool validM = (ALIGNED != 0u || gm8 * 8u < uint(pc.M)); \
        aPrefetch[s] = (validK && validM) ? aBufV8H.data[aBase_v8 + gk * (uint(pc.M) / 8u) + gm8] : uvec4(0); \
      } \
    } \
  }
#define LOAD_B_TO_REGS(blk) \
  { \
    if(PACKED_B != 0u) { \
      const uint bTileBase = (ic * kBlocks + (blk) / BK) * bPackedCount; \
      [[unroll]] for(uint s = 0; s < B_PER_THREAD; s++) { \
        const uint i = tid + s * BLOCK_SIZE; \
        if(i < bPackedCount) \
          bPrefetch[s] = bBufV4H.data[bBase_v4 + bTileBase + i]; \
      } \
    } else { \
      [[unroll]] for(uint s = 0; s < B_PER_THREAD; s++) { \
        const uint i = tid + s * BLOCK_SIZE; \
        if(i < bUnpackedCount) { \
          const uint kk = i / (BN / 4u); \
          const uint n4i = i % (BN / 4u); \
          const uint gk = (blk) + kk; \
          const uint gn4 = (ic * BN) / 4u + n4i; \
          const bool validK = (K_ALIGNED != 0u || gk < K_SPEC); \
          const bool validN = (ALIGNED != 0u || gn4 * 4u < uint(pc.N)); \
          bPrefetch[s] = (validK && validN) ? bBufV4H.data[bBase_v4 + gk * (uint(pc.N) / 4u) + gn4] : f16vec4(0.0hf); \
        } \
      } \
    } \
  }
#define COMMIT_A_TO_SHARED() \
  { \
    [[unroll]] for(uint s = 0; s < A_PER_THREAD; s++) { \
      const uint i = tid + s * BLOCK_SIZE; \
      if(i < aCount) { \
        const uint kk = i / (BM / 8u); \
        const uint mm8 = i % (BM / 8u); \
        buf_a[kk * SHMEM_A_STRIDE_V8 + mm8] = aPrefetch[s]; \
      } \
    } \
  }
#define COMMIT_B_TO_SHARED() \
  { \
    if(PACKED_B != 0u) { \
      [[unroll]] for(uint s = 0; s < B_PER_THREAD; s++) { \
        const uint i = tid + s * BLOCK_SIZE; \
        if(i < bPackedCount) { \
          f16vec4 b4 = bPrefetch[s]; \
          const uint dst = i * 4u; \
          buf_b[dst + 0u] = b4.x; \
          buf_b[dst + 1u] = b4.y; \
          buf_b[dst + 2u] = b4.z; \
          buf_b[dst + 3u] = b4.w; \
        } \
      } \
    } else { \
      [[unroll]] for(uint s = 0; s < B_PER_THREAD; s++) { \
        const uint i = tid + s * BLOCK_SIZE; \
        if(i < bUnpackedCount) { \
          const uint kk = i / (BN / 4u); \
          const uint n4i = i % (BN / 4u); \
          f16vec4 b4 = bPrefetch[s]; \
          const uint nRow = n4i * 4u; \
          [[unroll]] for(uint j = 0; j < 4u; j++) \
            buf_b[(nRow + j) * SHMEM_STRIDE + kk] = b4[j]; \
        } \
      } \
    } \
  }

  LOAD_A_TO_REGS(0u);
  LOAD_B_TO_REGS(0u);
  COMMIT_A_TO_SHARED();
  COMMIT_B_TO_SHARED();
  barrier();

  for(uint block = 0; block < K_SPEC; block += BK) {
    const uint nextBlock = block + BK;
    const bool hasNext = nextBlock < K_SPEC;
    if(hasNext) {
      LOAD_A_TO_REGS(nextBlock);
      LOAD_B_TO_REGS(nextBlock);
    }

    // Coopmat inner loop: contract the BK-slab TK at a time.
    // Register-block: load each distinct A (cmsPerRow) and B (cmsPerCol) fragment
    // from shared once per kOff, then the full outer product — avoids reloading
    // the B fragment inside the cmRow loop.
    [[unroll]] for(uint kOff = 0; kOff < BK; kOff += TK) {
      coopmat<float16_t, gl_ScopeSubgroup, TM, TK, gl_MatrixUseA> cache_a[WM / TM];
      coopmat<float16_t, gl_ScopeSubgroup, TK, TN, gl_MatrixUseB> cache_b[WN / TN];
      [[unroll]] for(uint cmRow = 0; cmRow < cmsPerRow; cmRow++) {
        // A fragment TM(M) x TK(K): buf_a is [K][M/8] half8-packed -> ColumnMajor.
        const uint mBase = warp_r * WM + cmRow * TM;
        const uint aOff = kOff * SHMEM_A_STRIDE_V8 + mBase / 8u;
        coopMatLoad(cache_a[cmRow], buf_a, aOff, SHMEM_A_STRIDE_V8, gl_CooperativeMatrixLayoutColumnMajor);
      }
      [[unroll]] for(uint cmCol = 0; cmCol < cmsPerCol; cmCol++) {
        // B fragment TK(K) x TN(N): buf_b is [N][K] row-major -> read as ColumnMajor
        // (rows=K contiguous within a slot, cols=N stride SHMEM_STRIDE apart).
        const uint bOff = (warp_c * WN + cmCol * TN) * SHMEM_STRIDE + kOff;
        coopMatLoad(cache_b[cmCol], buf_b, bOff, SHMEM_STRIDE, gl_CooperativeMatrixLayoutColumnMajor);
      }
      [[unroll]] for(uint cmRow = 0; cmRow < cmsPerRow; cmRow++) {
        [[unroll]] for(uint cmCol = 0; cmCol < cmsPerCol; cmCol++) {
          const uint s = cmRow * cmsPerCol + cmCol;
          sums[s] = coopMatMulAdd(cache_a[cmRow], cache_b[cmCol], sums[s]);
        }
      }
    }

    if(hasNext) {
      barrier();
      COMMIT_A_TO_SHARED();
      COMMIT_B_TO_SHARED();
      barrier();
    }
  }
#undef LOAD_A_TO_REGS
#undef LOAD_B_TO_REGS
#undef COMMIT_A_TO_SHARED
#undef COMMIT_B_TO_SHARED

  // Store directly for full real output fragments. Tail fragments still stage
  // through subgroup scratch for guards.
  // C[batch, N_real, M] as scalar float16_t. C index: cBase + n*M + m.
  const uint stageBase = warp_i * TM * TN;
  [[unroll]] for(uint cmRow = 0; cmRow < cmsPerRow; cmRow++) {
    [[unroll]] for(uint cmCol = 0; cmCol < cmsPerCol; cmCol++) {
      const uint s = cmRow * cmsPerCol + cmCol;
      const uint mTileBase = ir * BM + warp_r * WM + cmRow * TM;
      const uint nTileBase = ic * BN + warp_c * WN + cmCol * TN;
      if(mTileBase + TM <= uint(pc.M) && nTileBase + TN <= uint(pc.N_real)) {
        coopmat<float16_t, gl_ScopeSubgroup, TM, TN, gl_MatrixUseAccumulator> outSum = sums[s];
        if(ADD_TO_OUTPUT != 0u) {
          coopmat<float16_t, gl_ScopeSubgroup, TM, TN, gl_MatrixUseAccumulator> addFrag;
          coopMatLoad(
            addFrag,
            cBufH.data,
            cBase + nTileBase * uint(pc.M) + mTileBase,
            uint(pc.M),
            gl_CooperativeMatrixLayoutColumnMajor);
          outSum = outSum + addFrag;
        }
        coopMatStore(
          outSum,
          cBufH.data,
          cBase + nTileBase * uint(pc.M) + mTileBase,
          uint(pc.M),
          gl_CooperativeMatrixLayoutColumnMajor);
      } else {
        coopMatStore(sums[s], coopStage, stageBase, TN, gl_CooperativeMatrixLayoutRowMajor);
        controlBarrier(gl_ScopeSubgroup, gl_ScopeSubgroup, gl_StorageSemanticsShared, gl_SemanticsAcquireRelease);
        // Linearize in column-major order so adjacent lanes write adjacent M values.
        for(uint idx = lane; idx < TM * TN; idx += WARP) {
          const uint mLoc = idx % TM;
          const uint nLoc = idx / TM;
          const uint m = mTileBase + mLoc;
          const uint n = nTileBase + nLoc;
          // N_real guard required regardless of ALIGNED — real outC < padded N.
          if(m < uint(pc.M) && n < uint(pc.N_real))
            storeC(cBase + n * uint(pc.M) + m, coopStage[stageBase + mLoc * TN + nLoc]);
        }
        controlBarrier(gl_ScopeSubgroup, gl_ScopeSubgroup, gl_StorageSemanticsShared, gl_SemanticsAcquireRelease);
      }
    }
  }
}

// GLSL Compute Shader: Winograd batched GEMM — FP16 with KHR cooperative matrix (coopmat1, FP16 accumulate).
// C[batch][N][M] += A[batch][K][M] contracted with B[batch][K][N] over K.
//
// Uses VK_KHR_cooperative_matrix subgroup-scope fragments (ggml mul_mm.comp COOPMAT shape).
// The A/B operands are staged into shared memory as native half8 words, then coopMatLoad'd
// with layout/stride flags that interpret the packed source tiles as fragments.
//
// Memory layout:
//   A: [batch, M-tile, K-block, BM, BK/4+pad] as packed half4 words.
//   B: [batch, N-tile, K-block, BK, BN/4+pad] as packed half4 words.
//   C: [batch, N, M]   as float16_t scalar (M-contiguous).
// Shared A is row-major [M][K/4+pad], while B stays [K][N/4+pad]; storage uses
// half8 vectors over those even half4 strides.
// K always ÷8 for half8 A transport; N and M always ÷8.
//
// TM/TN/TK are the device's reported coopmat fragment shape (MSize/NSize/KSize) and are
// set by the host from the enumerated VkCooperativeMatrixPropertiesKHR — NOT free knobs.
//
// Dispatch: global(ceil(M/BM), ceil(N/BN), numBatch=inTileXYSize)

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

// Workgroup / subgroup / fragment tile dimensions — all specialization constants.
layout(constant_id = 0) const uint WORKGROUP_SIZE = 128;
layout(constant_id = 1) const uint BM = 64;
layout(constant_id = 2) const uint BN = 64;
layout(constant_id = 3) const uint BK = 16;
layout(constant_id = 4) const uint SGM = 32;
layout(constant_id = 5) const uint SGN = 32;
layout(constant_id = 6) const uint TM = 16;  // coopmat MSize
layout(constant_id = 7) const uint TN = 16;  // coopmat NSize
layout(constant_id = 8) const uint TK = 16;  // coopmat KSize
layout(constant_id = 9) const uint SUBGROUP_SIZE = 32;
layout(constant_id = 10) const uint K_SPEC = 1;

layout(local_size_x_id = 0, local_size_y = 1, local_size_z = 1) in;

layout(push_constant) uniform PushConstants {
  int M;        // numTilesPaddedRun (always ÷8)
  int N;        // numOutChannelsPadded (always ÷8)
  int strideA;  // packed A vec4 units between Winograd batches
  int strideB;  // packed B vec4 units between Winograd batches
  int strideC;  // (N/4)*M — vec4 units between Winograd batches in C
}
pc;

// A: packed raw half8 view over [batch, M-tile, K-block, BM, BK/4+pad].
// strideA is in half4 words; row strides are even, so /2 gives half8 words.
layout(binding = 0) readonly buffer ABufV8H {
  uvec4 data[];
}
aBufV8H;
// B: packed raw half8 view over [batch, N-tile, K-block, BK, BN/4+pad].
// strideB is in half4 words; all packed tile strides are even, so /2 gives half8 words.
layout(binding = 1) readonly buffer BBufV8H {
  uvec4 data[];
}
bBufV8H;
// C: scalar float16_t [batch, N, M] (M-contiguous). strideC is in vec4 units.
layout(binding = 2) writeonly buffer CBufH {
  float16_t data[];
}
cBufH;

// Packed shared A is [M][K/4 + pad], B is [K][N/4 + pad], stored as half8 words.
#define SHMEM_A_PAD_WORDS 2u
#define SHMEM_B_PAD_WORDS 2u
#define SHMEM_A_STRIDE ((BK / 4u) + SHMEM_A_PAD_WORDS)
#define SHMEM_B_STRIDE ((BN / 4u) + SHMEM_B_PAD_WORDS)
#define SHMEM_A_STRIDE_V8 (SHMEM_A_STRIDE / 2u)
#define SHMEM_B_STRIDE_V8 (SHMEM_B_STRIDE / 2u)

shared uvec4 buf_a[BM * SHMEM_A_STRIDE_V8];  // [M][K/8 + pad/2], K half8-contiguous
shared uvec4 buf_b[BK * SHMEM_B_STRIDE_V8];  // [K][N/8 + pad/2], N half8-contiguous

void main() {
  const uint ic = gl_WorkGroupID.y;
  const uint ir = gl_WorkGroupID.x;
  const uint batch = gl_GlobalInvocationID.z;

  const uint cmsPerRow = SGM / TM;  // coopmat fragments along M within a subgroup tile
  const uint cmsPerCol = SGN / TN;  // coopmat fragments along N within a subgroup tile

  // Subgroup identity from the real built-ins (coopmat fragments are subgroup-scoped).
  const uint subgroup_i = gl_SubgroupID;
  const uint subgroup_r = subgroup_i % (BM / SGM);
  const uint subgroup_c = subgroup_i / (BM / SGM);

  const uint tid = gl_LocalInvocationID.x;

  // Base offsets.
  const uint aBase_v8 = (batch * uint(pc.strideA)) / 2u;  // raw half8 units
  const uint bBase_v8 = (batch * uint(pc.strideB)) / 2u;  // raw half8 units
  const uint cBase = batch * uint(pc.strideC) * 4u;       // float16_t units (1 vec4 = 4 f16)

  // Accumulators: cmsPerRow x cmsPerCol FP16 fragments per subgroup.
  coopmat<float16_t, gl_ScopeSubgroup, TM, TN, gl_MatrixUseAccumulator> sums[(SGM / TM) * (SGN / TN)];
  [[unroll]] for(uint i = 0; i < cmsPerRow * cmsPerCol; i++)
    sums[i] = coopmat<float16_t, gl_ScopeSubgroup, TM, TN, gl_MatrixUseAccumulator>(0.0hf);

  const uint kBlocks = (K_SPEC + BK - 1u) / BK;
  const uint aCountV8 = BM * (BK / 8u);
  const uint bCountV8 = BK * SHMEM_B_STRIDE_V8;
  // Per-thread register prefetch buffers: enough words for this thread's slice of
  // the A/B global loads (ceil(count/WORKGROUP_SIZE)). Spec-const sized; glslang
  // allocates for the compile-time max.
  const uint A_PER_THREAD = (aCountV8 + WORKGROUP_SIZE - 1u) / WORKGROUP_SIZE;
  const uint B_PER_THREAD = (bCountV8 + WORKGROUP_SIZE - 1u) / WORKGROUP_SIZE;

  // Global-load software pipeline (Turing has no cp.async; prefetch via registers).
  // Load block n+1 from global into registers while the MMAs for block n run over
  // shared, then commit the registers to shared. Hides global-load latency behind
  // compute without growing the shared footprint (so occupancy is unchanged).
  uvec4 aPrefetch[(BM * (BK / 8u) + WORKGROUP_SIZE - 1u) / WORKGROUP_SIZE];
  uvec4 bPrefetch[(BK * SHMEM_B_STRIDE_V8 + WORKGROUP_SIZE - 1u) / WORKGROUP_SIZE];

  // Load a K-block's A tile from global into the register prefetch buffer.
  // (Body inlined at each call site below — GLSL has no closures over shared/regs.)
#define LOAD_A_TO_REGS(blk) \
  { \
    const uint aTileBaseV8 = aBase_v8 + (ir * kBlocks + (blk) / BK) * BM * SHMEM_A_STRIDE_V8; \
    [[unroll]] for(uint s = 0; s < A_PER_THREAD; s++) { \
      const uint i = tid + s * WORKGROUP_SIZE; \
      if(i < aCountV8) { \
        const uint mRow = i / (BK / 8u); \
        const uint kk8 = i % (BK / 8u); \
        aPrefetch[s] = aBufV8H.data[aTileBaseV8 + mRow * SHMEM_A_STRIDE_V8 + kk8]; \
      } \
    } \
  }
#define LOAD_B_TO_REGS(blk) \
  { \
    const uint bTileBaseV8 = bBase_v8 + (ic * kBlocks + (blk) / BK) * bCountV8; \
    [[unroll]] for(uint s = 0; s < B_PER_THREAD; s++) { \
      const uint i = tid + s * WORKGROUP_SIZE; \
      if(i < bCountV8) \
        bPrefetch[s] = bBufV8H.data[bTileBaseV8 + i]; \
    } \
  }
#define COMMIT_A_TO_SHARED() \
  { \
    [[unroll]] for(uint s = 0; s < A_PER_THREAD; s++) { \
      const uint i = tid + s * WORKGROUP_SIZE; \
      if(i < aCountV8) { \
        const uint mRow = i / (BK / 8u); \
        const uint kk8 = i % (BK / 8u); \
        buf_a[mRow * SHMEM_A_STRIDE_V8 + kk8] = aPrefetch[s]; \
      } \
    } \
  }
#define COMMIT_B_TO_SHARED() \
  { \
    [[unroll]] for(uint s = 0; s < B_PER_THREAD; s++) { \
      const uint i = tid + s * WORKGROUP_SIZE; \
      if(i < bCountV8) { \
        buf_b[i] = bPrefetch[s]; \
      } \
    } \
  }

  // Prologue: prefetch + commit block 0.
  LOAD_A_TO_REGS(0u);
  LOAD_B_TO_REGS(0u);
  COMMIT_A_TO_SHARED();
  COMMIT_B_TO_SHARED();
  barrier();

  for(uint block = 0; block < K_SPEC; block += BK) {
    // Prefetch the next K-block into registers (global loads in flight during MMA).
    const uint nextBlock = block + BK;
    const bool hasNext = nextBlock < K_SPEC;
    if(hasNext) {
      LOAD_A_TO_REGS(nextBlock);
      LOAD_B_TO_REGS(nextBlock);
    }

    // Coopmat inner loop: contract the BK-slab TK at a time.
    // Register-block the fragments: load each distinct A fragment (cmsPerRow) and
    // B fragment (cmsPerCol) from shared ONCE per kOff, then do the full
    // cmsPerRow x cmsPerCol outer product.
    [[unroll]] for(uint kOff = 0; kOff < BK; kOff += TK) {
      coopmat<float16_t, gl_ScopeSubgroup, TM, TK, gl_MatrixUseA> cache_a[SGM / TM];
      coopmat<float16_t, gl_ScopeSubgroup, TK, TN, gl_MatrixUseB> cache_b[SGN / TN];
      [[unroll]] for(uint cmRow = 0; cmRow < cmsPerRow; cmRow++) {
        const uint mBase = subgroup_r * SGM + cmRow * TM;
        const uint aOff = mBase * SHMEM_A_STRIDE_V8 + kOff / 8u;
        coopMatLoad(cache_a[cmRow], buf_a, aOff, SHMEM_A_STRIDE_V8, gl_CooperativeMatrixLayoutRowMajor);
      }
      [[unroll]] for(uint cmCol = 0; cmCol < cmsPerCol; cmCol++) {
        // B fragment TK(K) x TN(N): buf_b is logical [K][N/4] half4 data in half8 storage.
        const uint nBase = subgroup_c * SGN + cmCol * TN;
        const uint bOff = kOff * SHMEM_B_STRIDE_V8 + nBase / 8u;
        coopMatLoad(cache_b[cmCol], buf_b, bOff, SHMEM_B_STRIDE_V8, gl_CooperativeMatrixLayoutRowMajor);
      }
      [[unroll]] for(uint cmRow = 0; cmRow < cmsPerRow; cmRow++) {
        [[unroll]] for(uint cmCol = 0; cmCol < cmsPerCol; cmCol++) {
          const uint s = cmRow * cmsPerCol + cmCol;
          sums[s] = coopMatMulAdd(cache_a[cmRow], cache_b[cmCol], sums[s]);
        }
      }
    }

    // Commit the prefetched next block to shared for the following iteration.
    if(hasNext) {
      barrier();  // WAR: all subgroups done reading this block's shared before overwrite.
      COMMIT_A_TO_SHARED();
      COMMIT_B_TO_SHARED();
      barrier();  // RAW: shared visible before next block's coopMatLoad.
    }
  }
#undef LOAD_A_TO_REGS
#undef LOAD_B_TO_REGS
#undef COMMIT_A_TO_SHARED
#undef COMMIT_B_TO_SHARED

  [[unroll]] for(uint cmRow = 0; cmRow < cmsPerRow; cmRow++) {
    [[unroll]] for(uint cmCol = 0; cmCol < cmsPerCol; cmCol++) {
      const uint s = cmRow * cmsPerCol + cmCol;
      const uint mTileBase = ir * BM + subgroup_r * SGM + cmRow * TM;
      const uint nTileBase = ic * BN + subgroup_c * SGN + cmCol * TN;
      coopMatStore(
        sums[s],
        cBufH.data,
        cBase + nTileBase * uint(pc.M) + mTileBase,
        uint(pc.M),
        gl_CooperativeMatrixLayoutColumnMajor);
    }
  }
}

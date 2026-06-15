// GLSL Compute Shader: Direct 3x3/5x5 convolution as implicit GEMM, FP16 operands,
// KHR subgroup cooperative matrix (coopmat1), FP32 accumulate,
// aligned-only C store, **NHWC activations with a vec8 channel-contiguous
// im2col gather**.
//
// ---------------------------------------------------------------------------
// Why this shader exists (and why the previous three NHWC attempts failed)
// ---------------------------------------------------------------------------
// §10.16-§10.20 of VULKAN_CONV3X3_IMPLICIT_GEMM_STATUS.md record three NHWC
// tile -- scalar float16_t [K][BM + pad] -- and therefore had to *transpose*
// during staging. That transpose is what cost the time, and it cancelled the
// entire point of NHWC:
//
//   1. Scalar gather. The NHWC-native staging in §10.17 mapped lanes as
//      kk = i % BK, mm = i / BK, so consecutive lanes read consecutive
//      *channels* -- contiguous in NHWC -- but read them ONE float16_t AT A
//      TIME. A subgroup issued 32 separate 2-byte loads covering 64 contiguous
//      bytes. NHWC's whole structural advantage is that 8 consecutive channels
//      are 16 contiguous bytes, i.e. exactly one uvec4. Loading them scalar
//      throws that away and keeps the address arithmetic 8x over.
//
//   2. Transposed shared store => bank conflicts. Writing a channel-major
//      gather into a [K][M]-shaped tile means lane L writes
//      buf_a[(kk+L) * (BM+8) + mm]. With BM=64 the stride between lanes is 72
//      halves = 144 bytes, so all 32 lanes of a subgroup hit the same handful of
//      with mm = tid % BM, i.e. unit stride -- conflict-free.
//
//      mm = tid % BM being thread-constant across the A_PER_THREAD s-loop
//      (WORKGROUP_SIZE is a multiple of BM), which lets SLAB_INSIDE_TAP hoist
//      (y, x, xy, valid) to once per (thread, slab). §10.17's mapping made
//      mm = i / BK vary with s, so the im2col coordinate math -- two integer
//      divides, four bounds compares -- ran per element again. That silently
//      undid commit 7e632b94, the single largest win in this whole effort.
//
//   §10.20 then repacked B and flipped A to [M][K], which fixed the transpose
//   but kept the *scalar* gather, so it only recovered ~90 -> ~85 ms.
//
// ---------------------------------------------------------------------------
// What this shader does differently
// ---------------------------------------------------------------------------
// Shared A is packed half8 [M][K/8 + pad] in raw uvec4 words -- the exact
// layout `winograd_gemm_coopmat1_accf32.glsl` already uses for its A operand in this
// same tree, loaded with a plain RowMajor coopMatLoad. That layout is the
// natural target for NHWC, because a uvec4 of 8 consecutive channels in memory
// IS 8 consecutive K values in the GEMM. So the gather becomes:
//
//   * ONE uvec4 global load per 8 K-values (128B per 8 lanes, perfectly
//     coalesced -- lanes 0..7 cover exactly one 128-byte transaction),
//   * ONE uvec4 shared store, unit-stride in the K/8 axis, no transpose,
//   * mm = i / (BK/8) is thread-constant across the s-loop again (WORKGROUP_SIZE
//     is a multiple of BK/8), so ALL the SLAB_INSIDE_TAP hoisting is restored:
//     tap, dy, dx, y, x, xy, valid, and the mask fetch are computed once per
//     (thread, slab), and the channel offset kk8*8 is a hoisted constant.
//
// B keeps the proven [N][BK + pad] scalar shared layout, loaded ColumnMajor --
// (the filter is prepacked on the CPU at pipeline-build time).
//
// (A 64*(8+1)*16 = 9216 B + B 128*72*2 = 18432 B = 27648 B, matching the
// baseline's measured 27,648 B), identical compute core, but the gather does
// 1/8 the global load instructions and 1/8 the index arithmetic of §10.17.
//
// Requires inChannels % 8 == 0 so a uvec4 never straddles the channel-count
// boundary, and (for the hoisted path) inChannels % BK == 0. Both hold for the
// dominant trunk convs (C=256 or 384). The host gates on this; see the
// nhwcVec8 selection in vulkanlayers.h. This also satisfies the NVIDIA
// convolution-performance-guide alignment requirement noted in §10.18.
//
// GEMM shape: C[oc][m] += A_im2col[K][m] . B[K][oc]
//   K = CONV_SIZE^2 * inChannels (the convolution tap folded into the reduction axis)
//   M = paddedSpatialSize (per batch item), N = outChannelsPadded
// Memory layout:
//   Input  In:  [batch, paddedSpatialSize, ic]   NHWC (channel contiguous)
//   Filter B:   tile-local [Ntile][Kblock][BN][BK + pad]
//   Output C:   [batch, paddedSpatialSize, oc]   NHWC (channel contiguous)
//   Scale/Bias: [ic]     Mask: [batch, paddedSpatialSize]

#version 450

#extension GL_EXT_control_flow_attributes : enable
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_KHR_memory_scope_semantics : require
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_cooperative_matrix : require

#ifdef KATAGO_VULKAN_ACCUMULATE_FP16
#define ACCUM_TYPE float16_t
#define ACCUM_ZERO 0.0hf
#else
#define ACCUM_TYPE float
#define ACCUM_ZERO 0.0
#endif

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
layout(constant_id = 6) const uint TM = 16;  // coopmat MSize (device fragment shape)
layout(constant_id = 7) const uint TN = 16;  // coopmat NSize
layout(constant_id = 8) const uint TK = 16;  // coopmat KSize
layout(constant_id = 9) const uint SUBGROUP_SIZE = 32;
layout(constant_id = 11) const uint ADD_TO_OUTPUT = 0;
// K_SPEC = the full flattened K = CONV_SIZE^2 * inChannels, spec'd at pipeline build.
layout(constant_id = 14) const uint K_SPEC = 9;
// SLAB_INSIDE_TAP: 1 -> host guarantees inChannels is a multiple of BK, so every
// BK-slab lives entirely inside ONE convolution tap, enabling the hoisted gather.
layout(constant_id = 17) const uint SLAB_INSIDE_TAP = 0;
// The 3x3 pipeline uses this default; the 5x5 pipeline specializes it to 5.
layout(constant_id = 18) const uint CONV_SIZE = 3u;
const uint CONV_TAPS = CONV_SIZE * CONV_SIZE;
const int CONV_RADIUS = int(CONV_SIZE / 2u);
layout(local_size_x_id = 0, local_size_y = 1, local_size_z = 1) in;

// Shared B: match ggml/coopmat1 GEMM ([N][K], K inner, +8 half pad). Unchanged
#define SHMEM_STRIDE (BK + 8u)

// Shared A is packed half8 [M][K/8 + pad] in raw uvec4 words, exactly as
// winograd_gemm_coopmat1_accf32.glsl stages its A operand. The +1 uvec4 of padding
// (= 8 halves) keeps the row stride odd in uvec4 units so the coopmat
// RowMajor load's per-row starts spread across shared banks.
#define SHMEM_A_PAD_V8 1u
#define SHMEM_A_STRIDE_V8 ((BK / 8u) + SHMEM_A_PAD_V8)

layout(push_constant) uniform PushConstants {
  int nnXLen;
  int nnYLen;
  int numOutChannels;        // real, unpadded (C-store guard)
  int numOutChannelsPadded;  // must be a multiple of BN
  int numInChannels;         // real, unpadded (K = 9 * numInChannels)
  int paddedSpatialSize;     // M per batch item
  int strideB;               // vec4 units per batch in B; zero if filter reused
  int strideC;               // vec4 units per batch in C (real N, not padded)
  int batchOffset;           // added to gl_WorkGroupID.z to get the batch index
}
pc;

// Binding 0 is the NHWC activation tensor. It is declared twice, as a raw
// uvec4 (half8) view and a scalar float16_t view over the same buffer: the
// vec8 gather uses the packed view, while the fused-BN path and any
// non-multiple-of-8 tail would use the scalar one. Aliasing two block
// declarations onto one binding is legal SPIR-V and is verified to compile.
layout(binding = 0) readonly buffer InBufV8 {
  uvec4 data[];
}
inBufV8;
layout(binding = 0) readonly buffer InBufH {
  float16_t data[];
}
inBufH;
layout(binding = 1) readonly buffer FilterBufH {
  f16vec4 data[];
}
filterBufV4H;
layout(binding = 2) buffer OutBufH {
  float16_t data[];
}
outBufH;
// These bindings remain for the common six-binding descriptor layout. BN, act,
// and mask are intentionally handled by the preceding separate kernel.
layout(binding = 3) readonly buffer ScaleBufH {
  float16_t data[];
}
scaleBufH;
layout(binding = 4) readonly buffer BiasBufH {
  float16_t data[];
}
biasBufH;
layout(binding = 5) readonly buffer MaskBufH {
  float16_t data[];
}
maskBufH;

// exactly, including 0,1,2,12,3 test order and both MISH stability cutoffs.
int conv3x3SpatialY(uint m) {
  return int(m) / pc.nnXLen;
}

int conv3x3SpatialX(uint m, int yBase) {
  return int(m) - yBase * pc.nnXLen;
}

shared uvec4 buf_a[BM * SHMEM_A_STRIDE_V8];  // [M][K/8 + pad], K half8-contiguous
shared float16_t buf_b[BN * SHMEM_STRIDE];   // [N][BK + pad], K inner

void main() {
  const uint ir = gl_WorkGroupID.x;  // tile index along M (spatial)
  const uint ic = gl_WorkGroupID.y;  // tile index along N (output channels)
  const uint n = gl_WorkGroupID.z + uint(pc.batchOffset);
  const uint mIrLocal = ir;

  const uint cmsPerRow = SGM / TM;
  const uint cmsPerCol = SGN / TN;

  const uint subgroup_i = gl_SubgroupID;
  const uint subgroup_r = subgroup_i % (BM / SGM);
  const uint subgroup_c = subgroup_i / (BM / SGM);
  const uint tid = gl_LocalInvocationID.x;

  const uint bBase_v4 = n * uint(pc.strideB);
  const uint cBase = n * uint(pc.strideC) * 4u;

  const uint NReal = uint(pc.numOutChannels);
  const uint inChannels = uint(pc.numInChannels);
  const uint spatialReal = uint(pc.nnXLen * pc.nnYLen);

  const uint kBlocks = (K_SPEC + BK - 1u) / BK;

  // Accumulators: cmsPerRow x cmsPerCol fragments per subgroup.
  coopmat<ACCUM_TYPE, gl_ScopeSubgroup, TM, TN, gl_MatrixUseAccumulator> sums[(SGM / TM) * (SGN / TN)];
  [[unroll]] for(uint i = 0; i < cmsPerRow * cmsPerCol; i++)
    sums[i] = coopmat<ACCUM_TYPE, gl_ScopeSubgroup, TM, TN, gl_MatrixUseAccumulator>(ACCUM_ZERO);

  // A is counted in uvec4 (half8) units now: BM * (BK/8) packs per slab, i.e.
  // 1/8 the element count -- and therefore 1/8 the loop trips and 1/8 the
  // address arithmetic -- of the scalar NHWC staging in §10.17.
  const uint aPackPerRow = BK / 8u;  // uvec4 words per M row per slab
  const uint aCountV8 = BM * aPackPerRow;
  const uint bPackedCount = (BN * SHMEM_STRIDE) / 4u;
  const uint A_PER_THREAD_V8 = (aCountV8 + WORKGROUP_SIZE - 1u) / WORKGROUP_SIZE;
  const uint B_PER_THREAD_PACKED = (bPackedCount + WORKGROUP_SIZE - 1u) / WORKGROUP_SIZE;

  // --- A staging: vec8 NHWC im2col gather.
  //
  // Lane mapping: kk8 = i % (BK/8) selects which group of 8 channels, mm =
  // i / (BK/8) selects the spatial position. Because WORKGROUP_SIZE is a multiple
  // of (BK/8), BOTH are thread-constant across the s-loop, which is what
  // restores the SLAB_INSIDE_TAP hoisting that §10.17 lost. Consecutive lanes
  // walk kk8 first, so lanes 0..(BK/8 - 1) read 16*(BK/8) contiguous bytes of
  // the NHWC channel axis -- one fully-coalesced transaction per 8 lanes at
  // BK=64.
  //
  // Out-of-bounds (off-board tap, or padding M) yields uvec4(0), which is eight
  // exact +0.0h halves, so tails contribute nothing to the dot product -- the
  // same guarantee the scalar path gets from `value = 0.0`.
#define LOAD_A_TO_REGS(blk) \
  { \
    if(SLAB_INSIDE_TAP != 0u) { \
      /* Fast path: BK divides inChannels, so blk maps to exactly one convolution tap. */ \
      /* Everything below the s-loop is computed once per (thread, slab). */ \
      const uint tap = (blk) / inChannels; \
      const uint ichSlabBase = (blk) - tap * inChannels; \
      const int dy = int(tap / CONV_SIZE) - CONV_RADIUS; \
      const int dx = int(tap % CONV_SIZE) - CONV_RADIUS; \
      const uint kk8 = tid % aPackPerRow; \
      const uint mmBase = tid / aPackPerRow; \
      const uint mmStep = WORKGROUP_SIZE / aPackPerRow; \
      /* Channel base for this lane's 8-channel group, hoisted out of the loop. */ \
      const uint ichBase = ichSlabBase + kk8 * 8u; \
      const bool kValid = tap < CONV_TAPS && ((blk) + kk8 * 8u) < K_SPEC && (ichBase + 8u) <= inChannels; \
      [[unroll]] for(uint s = 0; s < A_PER_THREAD_V8; s++) { \
        const uint mm = mmBase + s * mmStep; \
        if(mm < BM) { \
          const uint m = mIrLocal * BM + mm; \
          const int yBase = conv3x3SpatialY(m); \
          const int y = yBase + dy; \
          const int x = conv3x3SpatialX(m, yBase) + dx; \
          const bool valid = kValid && m < spatialReal && y >= 0 && y < pc.nnYLen && x >= 0 && x < pc.nnXLen; \
          uvec4 packed = uvec4(0); \
          if(valid) { \
            const uint xy = uint(y * pc.nnXLen + x); \
            /* NHWC: 8 consecutive channels are 8 consecutive halves = 1 uvec4. */ \
            const uint elemIdx = (n * uint(pc.paddedSpatialSize) + xy) * inChannels + ichBase; \
            packed = inBufV8.data[elemIdx / 8u]; \
          } \
          buf_a[mm * SHMEM_A_STRIDE_V8 + kk8] = packed; \
        } \
      } \
    } else { \
      /* Slow path: slab may straddle a tap boundary, so the tap must be */ \
      /* resolved per 8-channel group. Still one uvec4 per 8 K values. */ \
      [[unroll]] for(uint s = 0; s < A_PER_THREAD_V8; s++) { \
        const uint i = tid + s * WORKGROUP_SIZE; \
        if(i < aCountV8) { \
          const uint kk8 = i % aPackPerRow; \
          const uint mm = i / aPackPerRow; \
          const uint m = mIrLocal * BM + mm; \
          const uint gk = (blk) + kk8 * 8u; \
          uvec4 packed = uvec4(0); \
          { \
            const uint tap = gk / inChannels; \
            const uint ich = gk - tap * inChannels; \
            const int dy = int(tap / CONV_SIZE) - CONV_RADIUS; \
            const int dx = int(tap % CONV_SIZE) - CONV_RADIUS; \
            /* A uvec4 must not straddle the channel-count boundary; the host */ \
            /* gates this shader on inChannels % 8 == 0 so it never does. */ \
            const bool kValid = tap < CONV_TAPS && gk < K_SPEC && (ich + 8u) <= inChannels; \
            if(kValid && m < spatialReal) { \
              const int yBase = conv3x3SpatialY(m); \
              const int y = yBase + dy; \
              const int x = conv3x3SpatialX(m, yBase) + dx; \
              if(y >= 0 && y < pc.nnYLen && x >= 0 && x < pc.nnXLen) { \
                const uint xy = uint(y * pc.nnXLen + x); \
                const uint elemIdx = (n * uint(pc.paddedSpatialSize) + xy) * inChannels + ich; \
                packed = inBufV8.data[elemIdx / 8u]; \
              } \
            } \
          } \
          buf_a[mm * SHMEM_A_STRIDE_V8 + kk8] = packed; \
        } \
      } \
    } \
  }

// --- B staging: B is prepacked as tile-local [Ntile][Kblock][BN][BK + pad].
#define LOAD_B_TO_REGS(blk) \
  { \
    const uint bTileBase = (ic * kBlocks + (blk) / BK) * bPackedCount; \
    [[unroll]] for(uint s = 0; s < B_PER_THREAD_PACKED; s++) { \
      const uint i = tid + s * WORKGROUP_SIZE; \
      if(i < bPackedCount) { \
        const f16vec4 b4 = filterBufV4H.data[bBase_v4 + bTileBase + i]; \
        const uint dst = i * 4u; \
        buf_b[dst + 0u] = b4.x; \
        buf_b[dst + 1u] = b4.y; \
        buf_b[dst + 2u] = b4.z; \
        buf_b[dst + 3u] = b4.w; \
      } \
    } \
  }

  // Prime the pipeline.
  LOAD_A_TO_REGS(0u);
  LOAD_B_TO_REGS(0u);
  barrier();

  for(uint block = 0; block < K_SPEC; block += BK) {
    const uint nextBlock = block + BK;
    const bool hasNext = nextBlock < K_SPEC;
    [[unroll]] for(uint kOff = 0; kOff < BK; kOff += TK) {
      coopmat<float16_t, gl_ScopeSubgroup, TM, TK, gl_MatrixUseA> cache_a[SGM / TM];
      coopmat<float16_t, gl_ScopeSubgroup, TK, TN, gl_MatrixUseB> cache_b[SGN / TN];
      [[unroll]] for(uint cmRow = 0; cmRow < cmsPerRow; cmRow++) {
        // A fragment TM(M) x TK(K): buf_a is packed half8 [M][K/8], K
        // half8-contiguous, so this is a plain RowMajor load with the stride in
        // uvec4 words -- identical to winograd_gemm_coopmat1_accf32.glsl's A load. No
        // ColumnMajor swap and no transpose anywhere in this shader.
        const uint mBase = subgroup_r * SGM + cmRow * TM;
        const uint aOff = mBase * SHMEM_A_STRIDE_V8 + kOff / 8u;
        coopMatLoad(cache_a[cmRow], buf_a, aOff, SHMEM_A_STRIDE_V8, gl_CooperativeMatrixLayoutRowMajor);
      }
      [[unroll]] for(uint cmCol = 0; cmCol < cmsPerCol; cmCol++) {
        // B fragment TK(K) x TN(N): buf_b is [N][K] scalar (K inner), read
        const uint bOff = (subgroup_c * SGN + cmCol * TN) * SHMEM_STRIDE + kOff;
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
      LOAD_A_TO_REGS(nextBlock);
      LOAD_B_TO_REGS(nextBlock);
      barrier();
    }
  }
#undef LOAD_A_TO_REGS
#undef LOAD_B_TO_REGS

  // C store: aligned-only, no coopStage. NHWC is channel-contiguous, so use a
  // row-major store with row stride NReal.
  [[unroll]] for(uint cmRow = 0; cmRow < cmsPerRow; cmRow++) {
    [[unroll]] for(uint cmCol = 0; cmCol < cmsPerCol; cmCol++) {
      const uint s = cmRow * cmsPerCol + cmCol;
      const uint mTileBase = mIrLocal * BM + subgroup_r * SGM + cmRow * TM;
      const uint nTileBase = ic * BN + subgroup_c * SGN + cmCol * TN;
      const uint cOff = cBase + mTileBase * NReal + nTileBase;
      coopmat<ACCUM_TYPE, gl_ScopeSubgroup, TM, TN, gl_MatrixUseAccumulator> outSum = sums[s];
      if(ADD_TO_OUTPUT != 0u) {
        coopmat<float16_t, gl_ScopeSubgroup, TM, TN, gl_MatrixUseAccumulator> addFrag;
        coopMatLoad(addFrag, outBufH.data, cOff, NReal, gl_CooperativeMatrixLayoutRowMajor);
        outSum = outSum + coopmat<ACCUM_TYPE, gl_ScopeSubgroup, TM, TN, gl_MatrixUseAccumulator>(addFrag);
      }
      // Narrow once, at the end of the full K reduction.
      coopmat<float16_t, gl_ScopeSubgroup, TM, TN, gl_MatrixUseAccumulator> outFrag =
        coopmat<float16_t, gl_ScopeSubgroup, TM, TN, gl_MatrixUseAccumulator>(outSum);
      coopMatStore(outFrag, outBufH.data, cOff, NReal, gl_CooperativeMatrixLayoutRowMajor);
    }
  }
}

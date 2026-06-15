// Native NHWC 3x3/5x5 implicit GEMM using workgroup-scope coopmat2.
//
// A (the im2col'd activations) is gathered from canonical physical [M,K] NHWC
// storage into a shared tile, then read with a plain workgroup-scope coopMatLoad
// -- the same structure llama.cpp's conv2d_mm.comp COOPMAT2 branch uses. The
// im2col address math runs once per thread while staging, not per matrix element,
// which is what the earlier decode-callback formulation could not do. Packed B
// stays physical [N,K] in global memory and is read through a transpose view into
// logical [K,N]; it needs no shared tile.
#version 460

#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types : require
#extension GL_EXT_control_flow_attributes : require
#extension GL_KHR_memory_scope_semantics : require
#extension GL_KHR_cooperative_matrix : require
#extension GL_NV_cooperative_matrix2 : require

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

// The workgroup-scope fragment is the whole BM x BN tile, so this variant has
// no warp/fragment tiling constants. Slot 17 selects the hoisted gather when a
// BK slab cannot cross a convolution-tap boundary.
layout(constant_id = 0) const uint BLOCK_SIZE = 128;
layout(constant_id = 1) const uint BM = 64;
layout(constant_id = 2) const uint BN = 64;
layout(constant_id = 3) const uint BK = 32;
layout(constant_id = 11) const uint ADD_TO_OUTPUT = 0;
// Full flattened K = CONV_SIZE^2 * inChannels, before packStridedGemmBWeights adds its
// ordinary final-K-block padding.
layout(constant_id = 14) const uint K_SPEC = 9;
// 1 when inChannels is divisible by BK, so every slab belongs to one tap.
layout(constant_id = 17) const uint SLAB_INSIDE_TAP = 0;
// The 3x3 pipeline uses this default; the 5x5 pipeline specializes it to 5.
layout(constant_id = 18) const uint CONV_SIZE = 3u;
const uint CONV_TAPS = CONV_SIZE * CONV_SIZE;
const int CONV_RADIUS = int(CONV_SIZE / 2u);
layout(local_size_x_id = 0, local_size_y = 1, local_size_z = 1) in;

const uint B_PAD = 8u;
layout(push_constant) uniform PushConstants {
  int nnXLen, nnYLen, numOutChannels, numOutChannelsPadded, numInChannels, paddedSpatialSize;
  int strideB, strideC, batchOffset;
} pc;
layout(binding = 0) readonly buffer InV8 { uvec4 data[]; } inV8;
layout(binding = 1) readonly buffer Filter { float16_t data[]; } filterBuf;
layout(binding = 2) buffer Out { float16_t data[]; } outBuf;
layout(binding = 3) readonly buffer Unused3 { float16_t data[]; } unused3;
layout(binding = 4) readonly buffer Unused4 { float16_t data[]; } unused4;
layout(binding = 5) readonly buffer Unused5 { float16_t data[]; } unused5;

// Shared A: scalar [M][BK + pad], K inner, RowMajor -- the layout the
// workgroup-scope coopMatLoad expects. The +8 half pad keeps the row stride off a
// power of two so the collective load's per-row starts spread across shared banks.
// Global reads stay uvec4 (8 channel-contiguous halves per coalesced transaction);
// each is unpacked into 8 consecutive scalar slots.
#define A_PAD 8u
#define A_STRIDE (BK + A_PAD)
shared float16_t buf_a[BM * A_STRIDE];

void main() {
  const uint ir = gl_WorkGroupID.x;
  const uint ic = gl_WorkGroupID.y;
  const uint batch = gl_WorkGroupID.z + uint(pc.batchOffset);
  const uint tid = gl_LocalInvocationID.x;
  const uint inC = uint(pc.numInChannels);
  const int nnX = pc.nnXLen;
  const int nnY = pc.nnYLen;
  const uint spatialReal = uint(nnX * nnY);
  const uint bBase = batch * uint(pc.strideB) * 4u;
  const uint cBase = batch * uint(pc.strideC) * 4u;
  const uint bStride = BK + B_PAD;
  const uint kBlocks = (K_SPEC + BK - 1u) / BK;

  // A gather lane mapping (thread-constant across the s-loop): kk8 selects the
  // 8-channel group within the BK slab, mm the spatial row within the BM tile.
  // BLOCK_SIZE % (BK/8) == 0 is host-enforced so mmStep tiles [0,BM) exactly.
  const uint aPackPerRow = BK / 8u;         // uvec4 words per M row per slab
  const uint aCountV8 = BM * aPackPerRow;
  const uint A_PER_THREAD_V8 = (aCountV8 + BLOCK_SIZE - 1u) / BLOCK_SIZE;
  const uint kk8 = tid % aPackPerRow;
  const uint mmBase = tid / aPackPerRow;
  const uint mmStep = BLOCK_SIZE / aPackPerRow;
  // Global A loads are staged in registers one slab ahead. The scalar shared
  // layout is retained because it is what the workgroup-scope coopMatLoad
  // expects; the commit happens after the preceding slab's final barrier.
  uvec4 aPrefetch[A_PER_THREAD_V8];

  tensorLayoutNV<2> layoutB = createTensorLayoutNV(2);
  tensorLayoutNV<2> layoutC = createTensorLayoutNV(2);
  layoutB = setTensorLayoutDimensionNV(layoutB, BN, BK);
  layoutB = setTensorLayoutStrideNV(layoutB, bStride, 1u);
  layoutC = setTensorLayoutDimensionNV(layoutC, uint(pc.paddedSpatialSize), uint(pc.numOutChannels));
  layoutC = setTensorLayoutStrideNV(layoutC, uint(pc.numOutChannels), 1u);
  tensorViewNV<2, false, 1, 0> transposeView = createTensorViewNV(2, false, 1, 0);

  coopmat<ACCUM_TYPE, gl_ScopeWorkgroup, BM, BN, gl_MatrixUseAccumulator> acc =
    coopmat<ACCUM_TYPE, gl_ScopeWorkgroup, BM, BN, gl_MatrixUseAccumulator>(ACCUM_ZERO);
  if(ADD_TO_OUTPUT != 0u) {
    coopmat<float16_t, gl_ScopeWorkgroup, BM, BN, gl_MatrixUseAccumulator> old;
    coopMatLoadTensorNV(
      old, outBuf.data, cBase, sliceTensorLayoutNV(layoutC, int(ir * BM), BM, int(ic * BN), BN));
    acc = coopmat<ACCUM_TYPE, gl_ScopeWorkgroup, BM, BN, gl_MatrixUseAccumulator>(old);
  }

// Gather one vec8 per thread into registers. K is flattened across the 3x3
// taps. The fast path hoists the tap lookup per BK slab; the general path keeps
// the vec8 group lookup so slabs that cross tap boundaries remain correct.
#define LOAD_A_TO_PREFETCH(BLK) \
  { \
    const uint flatIchBase = (BLK) + kk8 * 8u; \
    uint tap = flatIchBase / inC; \
    uint ichBase = flatIchBase - tap * inC; \
    if(SLAB_INSIDE_TAP != 0u) { \
      tap = (BLK) / inC; \
      ichBase = (BLK) - tap * inC + kk8 * 8u; \
    } \
    const int dy = int(tap / CONV_SIZE) - CONV_RADIUS; \
    const int dx = int(tap % CONV_SIZE) - CONV_RADIUS; \
    const bool kValid = tap < CONV_TAPS && flatIchBase + 8u <= K_SPEC && ichBase + 8u <= inC; \
    [[unroll]] for(uint s = 0u; s < A_PER_THREAD_V8; s++) { \
      const uint mm = mmBase + s * mmStep; \
      if(mm < BM) { \
        const uint m = ir * BM + mm; \
        const int yBase = int(m) / nnX; \
        const int y = yBase + dy; \
        const int x = (int(m) - yBase * nnX) + dx; \
        const bool valid = kValid && m < spatialReal && y >= 0 && y < nnY && x >= 0 && x < nnX; \
        uvec4 packed = uvec4(0); \
        if(valid) { \
          const uint xy = uint(y * nnX + x); \
          const uint elemIdx = (batch * uint(pc.paddedSpatialSize) + xy) * inC + ichBase; \
          packed = inV8.data[elemIdx / 8u]; \
        } \
        aPrefetch[s] = packed; \
      } \
    } \
  }

// Commit the prior slab after its consumers have reached the trailing barrier.
#define COMMIT_A_TO_SHARED() \
  { \
    [[unroll]] for(uint s = 0u; s < A_PER_THREAD_V8; s++) { \
      const uint mm = mmBase + s * mmStep; \
      if(mm < BM) { \
        const uvec4 packed = aPrefetch[s]; \
        const uint dst = mm * A_STRIDE + kk8 * 8u; \
        buf_a[dst + 0u] = float16_t(unpackFloat2x16(packed.x).x); \
        buf_a[dst + 1u] = float16_t(unpackFloat2x16(packed.x).y); \
        buf_a[dst + 2u] = float16_t(unpackFloat2x16(packed.y).x); \
        buf_a[dst + 3u] = float16_t(unpackFloat2x16(packed.y).y); \
        buf_a[dst + 4u] = float16_t(unpackFloat2x16(packed.z).x); \
        buf_a[dst + 5u] = float16_t(unpackFloat2x16(packed.z).y); \
        buf_a[dst + 6u] = float16_t(unpackFloat2x16(packed.w).x); \
        buf_a[dst + 7u] = float16_t(unpackFloat2x16(packed.w).y); \
      } \
    } \
  }

  LOAD_A_TO_PREFETCH(0u)
  [[dont_unroll]] for(uint blk = 0u; blk < K_SPEC; blk += BK) {
    COMMIT_A_TO_SHARED()
    barrier();
    const uint nextBlk = blk + BK;
    if(nextBlk < K_SPEC)
      LOAD_A_TO_PREFETCH(nextBlk)
    coopmat<float16_t, gl_ScopeWorkgroup, BM, BK, gl_MatrixUseA> a;
    coopmat<float16_t, gl_ScopeWorkgroup, BK, BN, gl_MatrixUseB> b;
    coopMatLoad(a, buf_a, 0u, A_STRIDE, gl_CooperativeMatrixLayoutRowMajor);
    coopMatLoadTensorNV(
      b, filterBuf.data, bBase + (ic * kBlocks + blk / BK) * BN * bStride,
      sliceTensorLayoutNV(layoutB, 0, BN, 0, BK), transposeView);
    acc = coopMatMulAdd(a, b, acc);
    barrier();
  }

#undef COMMIT_A_TO_SHARED
#undef LOAD_A_TO_PREFETCH

  coopmat<float16_t, gl_ScopeWorkgroup, BM, BN, gl_MatrixUseAccumulator> result =
    coopmat<float16_t, gl_ScopeWorkgroup, BM, BN, gl_MatrixUseAccumulator>(acc);
  coopMatStoreTensorNV(
    result, outBuf.data, cBase, sliceTensorLayoutNV(layoutC, int(ir * BM), BM, int(ic * BN), BN));
}

// GLSL Compute Shader: Winograd batched GEMM — FP16 with FP32 DOT2 accumulation (OpFDot2MixAcc32VALVE)
// C[batch][N][M] += A[batch][K][M] contracted with B[batch][K][N] over K.
//
// Uses SPV_VALVE_mixed_float_dot_product for v_dot2_f32_f16 (f16vec2·f16vec2→f32).
// Warp-level register tiling (ggml mul_mm.comp shape) with the accumulation done
// by the native v_dot2 instruction: each dot_product() folds an f16vec4·f16vec4
// (4 K-values) into an FP32 accumulator with two v_dot2 calls.
//
// Memory layout note:
//   A: [batch, M-tile, K-block, M, K/4+pad] as f16vec4 — 4 consecutive K at a fixed M.
//   B: [batch, K, N/4] as f16vec4 — 4 consecutive N at a fixed K.
//   C: [batch, N, M]   as float16_t scalar (M-contiguous).
// v_dot2 needs consecutive K per output row in registers, so row-major A copies
// directly into shared memory laid out [M][K] (K inner).
//
// K always ÷4; N and M always ÷8.
// Dispatch: global(ceil(M/BM), ceil(N/BN), numBatch=inTileXYSize)

#version 450

#extension GL_EXT_control_flow_attributes : enable
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_spirv_intrinsics : require

#ifdef KATAGO_VULKAN_RTE_F16
#extension GL_EXT_spirv_intrinsics : require
spirv_execution_mode(extensions = ["SPV_KHR_float_controls"], capabilities = [4467], 4462, 16);
#endif

// OpFDot2MixAcc32VALVE: f16vec2·f16vec2 + f32 → f32
spirv_instruction(
  extensions = ["SPV_VALVE_mixed_float_dot_product"],
  capabilities = [6912],
  id = 6916) float v_dot2_f32_f16(f16vec2 a, f16vec2 b, float acc);

float dot_product(f16vec4 a, f16vec4 b, float acc) {
  return v_dot2_f32_f16(a.zw, b.zw, v_dot2_f32_f16(a.xy, b.xy, acc));
}

// Workgroup / warp tile dimensions — all specialization constants.
layout(constant_id = 0) const uint BLOCK_SIZE = 128;
layout(constant_id = 1) const uint BM = 64;
layout(constant_id = 2) const uint BN = 64;
layout(constant_id = 3) const uint WM = 32;
layout(constant_id = 4) const uint WN = 32;
layout(constant_id = 5) const uint WMITER = 2;
layout(constant_id = 6) const uint TM = 4;
layout(constant_id = 7) const uint TN = 2;
layout(constant_id = 8) const uint WARP = 32;
layout(constant_id = 9) const uint K_SPEC = 1;

layout(local_size_x_id = 0, local_size_y = 1, local_size_z = 1) in;

#define BK 32
#define BK_STEP 4
#define A_PAD_WORDS 1
#define A_ROW_WORDS ((BK / 4) + A_PAD_WORDS)
// Shared tiles are scalar f16, [free-dim][K]; +4 pad per row eases bank conflicts.
#define SHMEM_STRIDE (BK + 4)

layout(push_constant) uniform PushConstants {
  int M;        // numTilesPaddedRun (always ÷8)
  int N;        // numOutChannelsPadded (always ÷8)
  int strideA;  // row-major packed A vec4 units between Winograd batches
  int strideB;  // K*(N/4) — vec4 units between Winograd batches in B
  int strideC;  // (N/4)*M — vec4 units between Winograd batches in C
}
pc;

// A: row-major packed f16vec4 view [batch, M-tile, K-block, M, K/4+pad].
layout(binding = 0) readonly buffer ABufV4H {
  f16vec4 data[];
}
aBufV4H;
// B: pack4-N f16vec4 view [batch, K, N/4]. strideB is in vec4 units.
layout(binding = 1) readonly buffer BBufV4H {
  f16vec4 data[];
}
bBufV4H;
// C: scalar float16_t [batch, N, M] (M-contiguous). strideC is in vec4 units.
layout(binding = 2) writeonly buffer CBufH {
  float16_t data[];
}
cBufH;

shared float16_t buf_a[BM * SHMEM_STRIDE];  // [M][K], K inner
shared float16_t buf_b[BN * SHMEM_STRIDE];  // [N][K], K inner

void main() {
  const uint ic = gl_WorkGroupID.y;
  const uint ir = gl_WorkGroupID.x;
  const uint batch = gl_GlobalInvocationID.z;

  const uint WNITER = (WM * WN) / (WARP * TM * TN * WMITER);
  const uint WSUBM = WM / WMITER;
  const uint WSUBN = WN / WNITER;

  const uint warp_i = gl_LocalInvocationID.x / WARP;
  const uint tiw = gl_LocalInvocationID.x % WARP;
  const uint tiwr = tiw % (WSUBM / TM);
  const uint tiwc = tiw / (WSUBM / TM);
  const uint warp_r = warp_i % (BM / WM);
  const uint warp_c = warp_i / (BM / WM);

  const uint tid = gl_LocalInvocationID.x;

  // Base offsets.
  const uint aBase_v4 = batch * uint(pc.strideA);    // f16vec4 units
  const uint bBase_v4 = batch * uint(pc.strideB);    // f16vec4 units
  const uint cBase = batch * uint(pc.strideC) * 4u;  // float16_t units (1 vec4 = 4 f16)
  const uint kBlocks = (K_SPEC + BK - 1u) / BK;

  // Accumulator: WMITER*TM rows × WNITER*TN cols as vec2 pairs.
  vec2 sums[WMITER * TM * WNITER * TN / 2];
  [[unroll]] for(uint i = 0; i < WMITER * TM * WNITER * TN / 2; i++)
    sums[i] = vec2(0.0);

  f16vec4 cache_a[WMITER * TM];
  f16vec4 cache_b;

  const uint aCount = BM * (BK / 4u);
  const uint bCount = BK * (BN / 4u);
  const uint A_PER_THREAD = (aCount + BLOCK_SIZE - 1u) / BLOCK_SIZE;
  const uint B_PER_THREAD = (bCount + BLOCK_SIZE - 1u) / BLOCK_SIZE;

  // Global-load software pipeline: prefetch block n+1 while block n computes.
  f16vec4 aPrefetch[(BM * (BK / 4u) + BLOCK_SIZE - 1u) / BLOCK_SIZE];
  f16vec4 bPrefetch[(BK * (BN / 4u) + BLOCK_SIZE - 1u) / BLOCK_SIZE];

#define LOAD_A_TO_REGS(blk) \
  { \
    const uint aTileBase = aBase_v4 + (ir * kBlocks + (blk) / BK) * BM * A_ROW_WORDS; \
    [[unroll]] for(uint s = 0; s < A_PER_THREAD; s++) { \
      const uint i = tid + s * BLOCK_SIZE; \
      if(i < aCount) { \
        const uint mRow = i / (BK / 4u); \
        const uint kk4 = i % (BK / 4u); \
        aPrefetch[s] = aBufV4H.data[aTileBase + mRow * A_ROW_WORDS + kk4]; \
      } \
    } \
  }
#define LOAD_B_TO_REGS(blk) \
  { \
    [[unroll]] for(uint s = 0; s < B_PER_THREAD; s++) { \
      const uint i = tid + s * BLOCK_SIZE; \
      if(i < bCount) { \
        const uint kk = i / (BN / 4u); \
        const uint n4i = i % (BN / 4u); \
        const uint gk = (blk) + kk; \
        const uint gn4 = (ic * BN) / 4u + n4i; \
        bPrefetch[s] = bBufV4H.data[bBase_v4 + gk * (uint(pc.N) / 4u) + gn4]; \
      } \
    } \
  }
#define COMMIT_A_TO_SHARED() \
  { \
    [[unroll]] for(uint s = 0; s < A_PER_THREAD; s++) { \
      const uint i = tid + s * BLOCK_SIZE; \
      if(i < aCount) { \
        const uint mRow = i / (BK / 4u); \
        const uint kk4 = i % (BK / 4u); \
        f16vec4 a4 = aPrefetch[s]; \
        const uint dst = mRow * SHMEM_STRIDE + kk4 * 4u; \
        [[unroll]] for(uint j = 0; j < 4u; j++) \
          buf_a[dst + j] = a4[j]; \
      } \
    } \
  }
#define COMMIT_B_TO_SHARED() \
  { \
    [[unroll]] for(uint s = 0; s < B_PER_THREAD; s++) { \
      const uint i = tid + s * BLOCK_SIZE; \
      if(i < bCount) { \
        const uint kk = i / (BN / 4u); \
        const uint n4i = i % (BN / 4u); \
        f16vec4 b4 = bPrefetch[s]; \
        const uint nRow = n4i * 4u; \
        [[unroll]] for(uint j = 0; j < 4u; j++) \
          buf_b[(nRow + j) * SHMEM_STRIDE + kk] = b4[j]; \
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

    // Inner accumulation: BK/BK_STEP steps, each covering 4 K-values via one v_dot2 pair.
    [[unroll]] for(uint i = 0; i < BK / BK_STEP; i++) {
      const uint kOff = BK_STEP * i;
      [[unroll]] for(uint wsir = 0; wsir < WMITER; wsir++) {
        [[unroll]] for(uint j = 0; j < TM; j++) {
          const uint row = warp_r * WM + wsir * WSUBM + tiwr * TM + j;
          const uint b = row * SHMEM_STRIDE + kOff;
          cache_a[wsir * TM + j] = f16vec4(buf_a[b], buf_a[b + 1u], buf_a[b + 2u], buf_a[b + 3u]);
        }
      }
      [[unroll]] for(uint wsic = 0; wsic < WNITER; wsic++) {
        [[unroll]] for(uint cc = 0; cc < TN; cc++) {
          const uint col = warp_c * WN + wsic * WSUBN + tiwc * TN + cc;
          const uint b = col * SHMEM_STRIDE + kOff;
          cache_b = f16vec4(buf_b[b], buf_b[b + 1u], buf_b[b + 2u], buf_b[b + 3u]);
          [[unroll]] for(uint wsir = 0; wsir < WMITER; wsir++) {
            [[unroll]] for(uint cr = 0; cr < TM / 2; cr++) {
              const uint sums_idx = (wsic * TN + cc) * WMITER * (TM / 2) + wsir * (TM / 2) + cr;
              sums[sums_idx].x = dot_product(cache_a[wsir * TM + 2u * cr], cache_b, sums[sums_idx].x);
              sums[sums_idx].y = dot_product(cache_a[wsir * TM + 2u * cr + 1u], cache_b, sums[sums_idx].y);
            }
          }
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

  // Write output: C[batch, N, M] as scalar float16_t. C index: cBase + n*M + m.
  const uint dr = ir * BM + warp_r * WM;
  const uint dc = ic * BN + warp_c * WN;

  [[unroll]] for(uint wsic = 0; wsic < WNITER; wsic++) {
    [[unroll]] for(uint wsir = 0; wsir < WMITER; wsir++) {
      const uint dr_warp = dr + wsir * WSUBM + tiwr * TM;
      const uint dc_warp = dc + wsic * WSUBN + tiwc * TN;
      [[unroll]] for(uint cc = 0; cc < TN; cc++) {
        [[unroll]] for(uint cr = 0; cr < TM / 2; cr++) {
          const uint sums_idx = (wsic * TN + cc) * WMITER * (TM / 2) + wsir * (TM / 2) + cr;
          const uint m0 = dr_warp + 2u * cr;
          const uint n0 = dc_warp + cc;
          cBufH.data[cBase + n0 * uint(pc.M) + m0] = float16_t(sums[sums_idx].x);
          cBufH.data[cBase + n0 * uint(pc.M) + m0 + 1u] = float16_t(sums[sums_idx].y);
        }
      }
    }
  }
}

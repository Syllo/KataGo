// GLSL Compute Shader: Strided batched GEMM — FP16 with DOT2 (OpFDot2MixAcc32VALVE)
// For 1x1 convolutions and transformer projections.
// C[batch][N_real][M] += A[batch][K][M] contracted with B[batch][K][N] over K.
//
// Uses SPV_VALVE_mixed_float_dot_product for v_dot2_f32_f16 (f16vec2·f16vec2→f32).
// Warp-level register tiling (ggml mul_mm.comp shape) with accumulation done by the
// native v_dot2 instruction: dot_product() folds an f16vec4·f16vec4 (4 K-values)
// into an FP32 accumulator with two v_dot2 calls.
//
// Memory layout (KataGo stores A/B K-outer, free dim contiguous):
//   A: [batch, K, M/8] as f16mat2x4 (pack8-M) — 8 consecutive M at a fixed K.
//   B: [batch, K, N/4] as f16vec4   (pack4-N) — 4 consecutive N at a fixed K.
//   C: [batch, N_real, M] as float16_t scalar (M-contiguous).
// The cooperative load transposes the contiguous free dimension into shared memory
// laid out [free-dim][K] (K inner) so v_dot2 sees consecutive K per output row.
// M always ÷8; N always ÷8; K unconstrained.
//
// Dispatch: global(ceil(M/BM), ceil(N/BN), batch)

#version 450

#extension GL_EXT_control_flow_attributes : enable
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_spirv_intrinsics : require

#ifdef KATAGO_VULKAN_RTE_F16
#extension GL_EXT_spirv_intrinsics : require
spirv_execution_mode(extensions = ["SPV_KHR_float_controls"], capabilities = [4467], 4462, 16);
#endif

spirv_instruction(
  extensions = ["SPV_VALVE_mixed_float_dot_product"],
  capabilities = [6912],
  id = 6916) float v_dot2_f32_f16(f16vec2 a, f16vec2 b, float acc);

float dot_product(f16vec4 a, f16vec4 b, float acc) {
  return v_dot2_f32_f16(a.zw, b.zw, v_dot2_f32_f16(a.xy, b.xy, acc));
}

layout(constant_id = 0) const uint BLOCK_SIZE = 128;
layout(constant_id = 1) const uint BM = 64;
layout(constant_id = 2) const uint BN = 64;
layout(constant_id = 3) const uint WM = 32;
layout(constant_id = 4) const uint WN = 32;
layout(constant_id = 5) const uint WMITER = 2;
layout(constant_id = 6) const uint TM = 4;
layout(constant_id = 7) const uint TN = 2;
layout(constant_id = 8) const uint WARP = 32;
// ALIGNED != 0 means padded dims M/N are guaranteed multiples of BM/BN, so M/N
// bounds checks on A/B loads compile away. K_ALIGNED separately means K is a
// multiple of BK, so A-side K-tail guards can compile away. Packed B zero-fills
// K tails on the host. The N_real C-store guard stays regardless: real
// outChannels < padded outChannels is what N_real exists to protect.
layout(constant_id = 9) const uint ALIGNED = 0;
layout(constant_id = 10) const uint ADD_TO_OUTPUT = 0;
layout(constant_id = 11) const uint PACKED_B = 0;
layout(constant_id = 12) const uint K_ALIGNED = 0;
layout(constant_id = 13) const uint K_SPEC = 1;

layout(local_size_x_id = 0, local_size_y = 1, local_size_z = 1) in;

#define BK 32
#define BK_STEP 4
// Shared tiles are scalar f16, [free-dim][K]; +4 pad per row eases bank conflicts.
#define SHMEM_STRIDE (BK + 4)

layout(push_constant) uniform PushConstants {
  int M;        // paddedSpatialSize (always ÷8)
  int N;        // outChannelsPadded (always ÷8)
  int N_real;   // outChannels (real, unpadded) — C-store guard
  int strideA;  // K*(M/4) — vec4 units per batch in A
  int strideB;  // K*(N/4) — vec4 units per batch in B
  int strideC;  // N_real*(M/4) — vec4 units per batch in C (real N, not padded)
}
pc;

// A: pack8-M f16mat2x4 view [batch, K, M/8]. strideA is in vec4 units; 1 mat2x4 = 2 vec4.
layout(binding = 0) readonly buffer ABufV8H {
  f16mat2x4 data[];
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

void storeC(uint idx, float val) {
  if(ADD_TO_OUTPUT != 0u)
    val += float(cBufH.data[idx]);
  cBufH.data[idx] = float16_t(val);
}

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
  const uint aBase_v8 = (batch * uint(pc.strideA)) / 2u;  // f16mat2x4 units (1 mat2x4 = 2 vec4)
  const uint bBase_v4 = batch * uint(pc.strideB);         // f16vec4 units
  const uint cBase = batch * uint(pc.strideC) * 4u;       // float16_t units (1 vec4 = 4 f16)
  const uint kBlocks = (K_SPEC + BK - 1u) / BK;

  vec2 sums[WMITER * TM * WNITER * TN / 2];
  [[unroll]] for(uint i = 0; i < WMITER * TM * WNITER * TN / 2; i++)
    sums[i] = vec2(0.0);

  f16vec4 cache_a[WMITER * TM];
  f16vec4 cache_b;

  // Per-thread element counts for the cooperative loads (match the loop iteration
  // spaces below). A is pack8-M f16mat2x4; B has a packed and an unpacked path, so
  // the B register buffer is sized for whichever path moves more elements per thread.
  const uint aCount = BK * (BM / 8u);                // f16mat2x4 elements
  const uint bTileWords = (BN * SHMEM_STRIDE) / 4u;  // f16vec4 words (PACKED_B path)
  const uint bCount = BK * (BN / 4u);                // f16vec4 elements (unpacked path)
  const uint A_PER_THREAD = (aCount + BLOCK_SIZE - 1u) / BLOCK_SIZE;
  const uint BP_PER_THREAD = (bTileWords + BLOCK_SIZE - 1u) / BLOCK_SIZE;
  const uint BU_PER_THREAD = (bCount + BLOCK_SIZE - 1u) / BLOCK_SIZE;

  // Global-load software pipeline: prefetch block n+1 into registers while block n
  // computes. Ping-pong through registers (not doubled shared memory), so the shared
  // footprint — and thus occupancy — is unchanged. Mirrors winograd_gemm_dot2.glsl.
  f16mat2x4 aPrefetch[(BK * (BM / 8u) + BLOCK_SIZE - 1u) / BLOCK_SIZE];
  // Max of the packed and unpacked B per-thread counts, evaluated as a constant.
  f16vec4 bPrefetch
    [(((BN * SHMEM_STRIDE) / 4u > BK * (BN / 4u) ? (BN * SHMEM_STRIDE) / 4u : BK * (BN / 4u)) + BLOCK_SIZE - 1u) /
     BLOCK_SIZE];

  // Read pack8-M f16mat2x4 (8 M at fixed K) into registers.
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
        aPrefetch[s] = (validK && validM) ? aBufV8H.data[aBase_v8 + gk * (uint(pc.M) / 8u) + gm8] \
                                          : f16mat2x4(f16vec4(0.0hf), f16vec4(0.0hf)); \
      } \
    } \
  }
  // Transpose-scatter the 8 M-values of each mat2x4 into buf_a[M][K] (K inner).
#define COMMIT_A_TO_SHARED() \
  { \
    [[unroll]] for(uint s = 0; s < A_PER_THREAD; s++) { \
      const uint i = tid + s * BLOCK_SIZE; \
      if(i < aCount) { \
        const uint kk = i / (BM / 8u); \
        const uint mm8 = i % (BM / 8u); \
        f16mat2x4 a8 = aPrefetch[s]; \
        const uint mRow = mm8 * 8u; \
        [[unroll]] for(uint j = 0; j < 8u; j++) \
          buf_a[(mRow + j) * SHMEM_STRIDE + kk] = a8[j / 4u][j % 4u]; \
      } \
    } \
  }
  // Read B into registers: contiguous pre-packed tile (PACKED_B) or pack4-N scatter.
#define LOAD_B_TO_REGS(blk) \
  { \
    if(PACKED_B != 0u) { \
      const uint bTileBase = (ic * kBlocks + (blk) / BK) * bTileWords; \
      [[unroll]] for(uint s = 0; s < BP_PER_THREAD; s++) { \
        const uint i = tid + s * BLOCK_SIZE; \
        if(i < bTileWords) \
          bPrefetch[s] = bBufV4H.data[bBase_v4 + bTileBase + i]; \
      } \
    } else { \
      [[unroll]] for(uint s = 0; s < BU_PER_THREAD; s++) { \
        const uint i = tid + s * BLOCK_SIZE; \
        if(i < bCount) { \
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
#define COMMIT_B_TO_SHARED() \
  { \
    if(PACKED_B != 0u) { \
      [[unroll]] for(uint s = 0; s < BP_PER_THREAD; s++) { \
        const uint i = tid + s * BLOCK_SIZE; \
        if(i < bTileWords) { \
          f16vec4 b4 = bPrefetch[s]; \
          const uint dst = i * 4u; \
          buf_b[dst + 0u] = b4.x; \
          buf_b[dst + 1u] = b4.y; \
          buf_b[dst + 2u] = b4.z; \
          buf_b[dst + 3u] = b4.w; \
        } \
      } \
    } else { \
      [[unroll]] for(uint s = 0; s < BU_PER_THREAD; s++) { \
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
          // N_real guard is required regardless of ALIGNED — real outC < padded N is what
          // N_real exists to protect. When ALIGNED, the M guard drops (M is padded to BM).
          if(ALIGNED != 0u) {
            if(n0 < uint(pc.N_real)) {
              storeC(cBase + n0 * uint(pc.M) + m0, sums[sums_idx].x);
              storeC(cBase + n0 * uint(pc.M) + m0 + 1u, sums[sums_idx].y);
            }
          } else {
            if(m0 < uint(pc.M) && n0 < uint(pc.N_real))
              storeC(cBase + n0 * uint(pc.M) + m0, sums[sums_idx].x);
            if(m0 + 1u < uint(pc.M) && n0 < uint(pc.N_real))
              storeC(cBase + n0 * uint(pc.M) + m0 + 1u, sums[sums_idx].y);
          }
        }
      }
    }
  }
}

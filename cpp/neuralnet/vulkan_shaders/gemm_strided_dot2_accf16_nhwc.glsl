// GLSL Compute Shader: Strided batched GEMM for native NHWC — FP16 with DOT2
// (OpFDot2MixAcc16VALVE). For 1x1 convolutions and transformer projections.
// C[batch][M][N_real] += A[batch][M][K] contracted with B[batch][K][N] over K.
//
// Uses SPV_VALVE_mixed_float_dot_product for v_dot2_f16_f16 (f16vec2·f16vec2→f16).
// Subgroup-level register tiling (ggml mul_mm.comp shape) with accumulation done by the
// native v_dot2 instruction: dot_product() folds an f16vec4·f16vec4 (4 K-values)
// into an FP16 accumulator with two v_dot2 calls. Compute loop, register tiling and
// only the A/C global-memory addressing differs (see
// gemm_strided_coopmat1_accf16_nhwc.glsl for the NHWC A/C layout this borrows).
//
// Memory layout (NHWC keeps the channel/K dimension innermost):
//   A: [batch, M, K] — K contiguous at each M, read as packed half8 (uvec4) words.
//      B's layout does not depend on activation layout.
//   C: [batch, M, N_real] as float16_t scalar (N contiguous).
// Because A is already K-inner in NHWC, no transpose-on-commit is needed: the eight
// K-values of an aligned load land directly in the same shared row that the compute
// [M][K] shared layout). Likewise C is written directly row-major, no swap needed.
// M unconstrained; N always ÷8; K unconstrained.
//
// Dispatch: global(ceil(M/BM), ceil(N/BN), batch)

#version 450

#extension GL_EXT_control_flow_attributes : enable
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
// unpackFloat2x16 (used to unpack the pack8-K uvec4 A words into f16vec2 pairs)
// is defined by the base explicit-arithmetic-types extension, not the _float16
// subset above.
#extension GL_EXT_shader_explicit_arithmetic_types : require
#extension GL_EXT_spirv_intrinsics : require

#ifdef KATAGO_VULKAN_RTE_F16
#extension GL_EXT_spirv_intrinsics : require
spirv_execution_mode(extensions = ["SPV_KHR_float_controls"], capabilities = [4467], 4462, 16);
#endif

spirv_instruction(extensions = ["SPV_VALVE_mixed_float_dot_product"], capabilities = [6913], id = 6917) float16_t
  v_dot2_f16_f16(f16vec2 a, f16vec2 b, float16_t acc);

float16_t dot_product(f16vec4 a, f16vec4 b, float16_t acc) {
  return v_dot2_f16_f16(a.zw, b.zw, v_dot2_f16_f16(a.xy, b.xy, acc));
}

layout(constant_id = 0) const uint WORKGROUP_SIZE = 128;
layout(constant_id = 1) const uint BM = 64;
layout(constant_id = 2) const uint BN = 64;
layout(constant_id = 3) const uint SGM = 32;
layout(constant_id = 4) const uint SGN = 32;
layout(constant_id = 5) const uint SGMITER = 2;
layout(constant_id = 6) const uint TM = 4;
layout(constant_id = 7) const uint TN = 2;
layout(constant_id = 8) const uint SUBGROUP_SIZE = 32;
// ALIGNED != 0 means padded dims M/N are guaranteed multiples of BM/BN, so M/N
// bounds checks on A/B loads compile away. K_ALIGNED separately means K is a
// multiple of BK, so B-side K-tail guards can compile away (A's K-tail guard
// is handled inline below, gated on the compile-time-folded K_SPEC % 8u check).
// Packed B zero-fills K tails on the host. The N_real C-store guard stays
// regardless: real outChannels < padded outChannels is what N_real protects.
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
  int M;        // paddedSpatialSize
  int N;        // outChannelsPadded (always ÷8)
  int N_real;   // outChannels (real, unpadded) — C-store guard
  int strideA;  // K*(M/4) — vec4 units per batch in A (layout-neutral element count)
  int strideB;  // K*(N/4) — vec4 units per batch in B
  int strideC;  // N_real*(M/4) — vec4 units per batch in C (real N, not padded)
}
pc;

// A: NHWC packed half8 view [batch, M, K/8], K contiguous. The scalar alias
// constructs the final half8 for K-tail loads without reading across an M row.
layout(binding = 0) readonly buffer ABufV8 {
  uvec4 data[];
}
aBufV8;
layout(binding = 0) readonly buffer ABufH {
  float16_t data[];
}
aBufH;
// B: pack4-N f16vec4 view [batch, K, N/4]. strideB is in vec4 units. Unchanged
layout(binding = 1) readonly buffer BBufV4H {
  f16vec4 data[];
}
bBufV4H;
// C: scalar float16_t [batch, M, N_real] (N contiguous). strideC is in vec4 units.
layout(binding = 2) buffer CBufH {
  float16_t data[];
}
cBufH;

void storeC(uint idx, float16_t val) {
  if(ADD_TO_OUTPUT != 0u)
    val += cBufH.data[idx];
  cBufH.data[idx] = val;
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

shared float16_t buf_a[BM * SHMEM_STRIDE];  // [M][K], K inner — matches physical A directly.
shared float16_t buf_b[BN * SHMEM_STRIDE];  // [N][K], K inner.

void main() {
  const uint ic = gl_WorkGroupID.y;
  const uint ir = gl_WorkGroupID.x;
  const uint batch = gl_GlobalInvocationID.z;

  const uint SGNITER = (SGM * SGN) / (SUBGROUP_SIZE * TM * TN * SGMITER);
  const uint SGSUBM = SGM / SGMITER;
  const uint SGSUBN = SGN / SGNITER;

  const uint subgroup_i = gl_LocalInvocationID.x / SUBGROUP_SIZE;
  const uint tiw = gl_LocalInvocationID.x % SUBGROUP_SIZE;
  const uint tiwr = tiw % (SGSUBM / TM);
  const uint tiwc = tiw / (SGSUBM / TM);
  const uint subgroup_r = subgroup_i % (BM / SGM);
  const uint subgroup_c = subgroup_i / (BM / SGM);

  const uint tid = gl_LocalInvocationID.x;

  // Base offsets. strideA/C retain the common vec4-unit ABI; NHWC's A/C are
  // [batch, M, K]/[batch, M, N_real] with K/N contiguous respectively.
  const uint aBase = batch * uint(pc.strideA) * 4u;  // float16_t units
  const uint bBase_v4 = batch * uint(pc.strideB);    // f16vec4 units
  const uint cBase = batch * uint(pc.strideC) * 4u;  // float16_t units
  const uint kBlocks = (K_SPEC + BK - 1u) / BK;

  f16vec2 sums[SGMITER * TM * SGNITER * TN / 2];
  [[unroll]] for(uint i = 0; i < SGMITER * TM * SGNITER * TN / 2; i++)
    sums[i] = f16vec2(0.0hf, 0.0hf);

  f16vec4 cache_a[SGMITER * TM];
  f16vec4 cache_b;

  // Per-thread element counts. A is now pack8-K (aCountV8 counts uvec4 words: one
  const uint aPacksPerRow = BK / 8u;
  const uint aCountV8 = BM * aPacksPerRow;           // uvec4 (half8) elements
  const uint bTileWords = (BN * SHMEM_STRIDE) / 4u;  // f16vec4 words (PACKED_B path)
  const uint bCount = BK * (BN / 4u);                // f16vec4 elements (unpacked path)
  const uint A_PER_THREAD_V8 = (aCountV8 + WORKGROUP_SIZE - 1u) / WORKGROUP_SIZE;
  const uint BP_PER_THREAD = (bTileWords + WORKGROUP_SIZE - 1u) / WORKGROUP_SIZE;
  const uint BU_PER_THREAD = (bCount + WORKGROUP_SIZE - 1u) / WORKGROUP_SIZE;

  // Global-load software pipeline: prefetch block n+1 into registers while block n
  // computes. Ping-pong through registers (not doubled shared memory), so the shared
  uvec4 aPrefetch[(BM * (BK / 8u) + WORKGROUP_SIZE - 1u) / WORKGROUP_SIZE];
  f16vec4 bPrefetch
    [(((BN * SHMEM_STRIDE) / 4u > BK * (BN / 4u) ? (BN * SHMEM_STRIDE) / 4u : BK * (BN / 4u)) + WORKGROUP_SIZE - 1u) /
     WORKGROUP_SIZE];

  // Read pack8-K half8 (8 consecutive K at a fixed M) into registers. Unlike the
  // already belong to the same shared row that the compute loop reads from.
#define LOAD_A_TO_REGS(blk) \
  { \
    [[unroll]] for(uint s = 0; s < A_PER_THREAD_V8; s++) { \
      const uint i = tid + s * WORKGROUP_SIZE; \
      if(i < aCountV8) { \
        const uint kk8 = i % aPacksPerRow; \
        const uint mm = i / aPacksPerRow; \
        const uint gk = (blk) + kk8 * 8u; \
        const uint gm = ir * BM + mm; \
        const bool validM = (ALIGNED != 0u || gm < uint(pc.M)); \
        uvec4 packed = uvec4(0); \
        if(validM) { \
          if((K_SPEC % 8u) == 0u) { \
            if((K_ALIGNED != 0u || gk < K_SPEC)) \
              packed = aBufV8.data[(aBase + gm * K_SPEC + gk) / 8u]; \
          } else if(gk < K_SPEC) { \
            packed = loadATail(aBase, gm, gk, K_SPEC); \
          } \
        } \
        aPrefetch[s] = packed; \
      } \
    } \
  }
  // Unpack the pack8-K word and scatter its 8 halves into the single shared row
  // segment [mm][kk8*8 .. kk8*8+7] — a direct store, not a transpose.
#define COMMIT_A_TO_SHARED() \
  { \
    [[unroll]] for(uint s = 0; s < A_PER_THREAD_V8; s++) { \
      const uint i = tid + s * WORKGROUP_SIZE; \
      if(i < aCountV8) { \
        const uint kk8 = i % aPacksPerRow; \
        const uint mm = i / aPacksPerRow; \
        const uvec4 packed = aPrefetch[s]; \
        const f16vec2 p0 = unpackFloat2x16(packed.x); \
        const f16vec2 p1 = unpackFloat2x16(packed.y); \
        const f16vec2 p2 = unpackFloat2x16(packed.z); \
        const f16vec2 p3 = unpackFloat2x16(packed.w); \
        const uint base = mm * SHMEM_STRIDE + kk8 * 8u; \
        buf_a[base + 0u] = p0.x; \
        buf_a[base + 1u] = p0.y; \
        buf_a[base + 2u] = p1.x; \
        buf_a[base + 3u] = p1.y; \
        buf_a[base + 4u] = p2.x; \
        buf_a[base + 5u] = p2.y; \
        buf_a[base + 6u] = p3.x; \
        buf_a[base + 7u] = p3.y; \
      } \
    } \
  }
  // Read B into registers: contiguous pre-packed tile (PACKED_B) or pack4-N scatter.
#define LOAD_B_TO_REGS(blk) \
  { \
    if(PACKED_B != 0u) { \
      const uint bTileBase = (ic * kBlocks + (blk) / BK) * bTileWords; \
      [[unroll]] for(uint s = 0; s < BP_PER_THREAD; s++) { \
        const uint i = tid + s * WORKGROUP_SIZE; \
        if(i < bTileWords) \
          bPrefetch[s] = bBufV4H.data[bBase_v4 + bTileBase + i]; \
      } \
    } else { \
      [[unroll]] for(uint s = 0; s < BU_PER_THREAD; s++) { \
        const uint i = tid + s * WORKGROUP_SIZE; \
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
        const uint i = tid + s * WORKGROUP_SIZE; \
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
        const uint i = tid + s * WORKGROUP_SIZE; \
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
      [[unroll]] for(uint wsir = 0; wsir < SGMITER; wsir++) {
        [[unroll]] for(uint j = 0; j < TM; j++) {
          const uint row = subgroup_r * SGM + wsir * SGSUBM + tiwr * TM + j;
          const uint b = row * SHMEM_STRIDE + kOff;
          cache_a[wsir * TM + j] = f16vec4(buf_a[b], buf_a[b + 1u], buf_a[b + 2u], buf_a[b + 3u]);
        }
      }
      [[unroll]] for(uint wsic = 0; wsic < SGNITER; wsic++) {
        [[unroll]] for(uint cc = 0; cc < TN; cc++) {
          const uint col = subgroup_c * SGN + wsic * SGSUBN + tiwc * TN + cc;
          const uint b = col * SHMEM_STRIDE + kOff;
          cache_b = f16vec4(buf_b[b], buf_b[b + 1u], buf_b[b + 2u], buf_b[b + 3u]);
          [[unroll]] for(uint wsir = 0; wsir < SGMITER; wsir++) {
            [[unroll]] for(uint cr = 0; cr < TM / 2; cr++) {
              const uint sums_idx = (wsic * TN + cc) * SGMITER * (TM / 2) + wsir * (TM / 2) + cr;
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

  const uint dr = ir * BM + subgroup_r * SGM;
  const uint dc = ic * BN + subgroup_c * SGN;

  [[unroll]] for(uint wsic = 0; wsic < SGNITER; wsic++) {
    [[unroll]] for(uint wsir = 0; wsir < SGMITER; wsir++) {
      const uint dr_subgroup = dr + wsir * SGSUBM + tiwr * TM;
      const uint dc_subgroup = dc + wsic * SGSUBN + tiwc * TN;
      [[unroll]] for(uint cc = 0; cc < TN; cc++) {
        [[unroll]] for(uint cr = 0; cr < TM / 2; cr++) {
          const uint sums_idx = (wsic * TN + cc) * SGMITER * (TM / 2) + wsir * (TM / 2) + cr;
          const uint m0 = dr_subgroup + 2u * cr;
          const uint n0 = dc_subgroup + cc;
          // NHWC C is [M, N_real], so N contiguous — the two accumulated m0/m0+1
          // rows land N_real apart, not adjacent (that adjacency belonged to the
          // ALIGNED — real outC < padded N is what N_real exists to protect.
          if(ALIGNED != 0u) {
            if(n0 < uint(pc.N_real)) {
              storeC(cBase + m0 * uint(pc.N_real) + n0, sums[sums_idx].x);
              storeC(cBase + (m0 + 1u) * uint(pc.N_real) + n0, sums[sums_idx].y);
            }
          } else {
            if(m0 < uint(pc.M) && n0 < uint(pc.N_real))
              storeC(cBase + m0 * uint(pc.N_real) + n0, sums[sums_idx].x);
            if(m0 + 1u < uint(pc.M) && n0 < uint(pc.N_real))
              storeC(cBase + (m0 + 1u) * uint(pc.N_real) + n0, sums[sums_idx].y);
          }
        }
      }
    }
  }
}

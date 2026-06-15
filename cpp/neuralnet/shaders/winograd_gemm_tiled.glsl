// GLSL Compute Shader: Winograd batched GEMM — FP32 or FP16 storage + FP32 shared-memory tiles
// C[batch][N][M/4] += A[batch][M-tile][K-block][M][K/4+pad] contracted with B[batch][K][N/4].
//
// Buffer views:
//   FP32: A/B/C as vec4.
//   FP16: A/B/C as f16vec4, widened/narrowed at global I/O.
// Shared-memory tiles and accumulation are FP32 in both modes.
// K is always a multiple of 4; N and M are always multiples of 8.
//
// Dispatch: global(ceil(M/TILE_M), ceil(N/TILE_N), numBatch=inTileXYSize)

#version 450
#extension GL_EXT_shader_16bit_storage : enable
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : enable

#ifdef KATAGO_VULKAN_RTE_F16
#extension GL_EXT_spirv_intrinsics : require
spirv_execution_mode(extensions = ["SPV_KHR_float_controls"], capabilities = [4467], 4462, 16);
#endif

layout(constant_id = 0) const int WG_X = 8;
layout(constant_id = 1) const int WG_Y = 8;
layout(constant_id = 2) const int TILE_K = 8;
layout(constant_id = 3) const int RN = 4;
layout(constant_id = 4) const int USE_FP16_STORAGE = 0;
layout(constant_id = 5) const int K_SPEC = 4;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z = 1) in;

const int TILE_M = WG_X * 4;
const int TILE_N = WG_Y * RN;
const int TILE_K4 = TILE_K / 4;
const int A_PAD_WORDS = 1;
const int A_ROW_WORDS = TILE_K4 + A_PAD_WORDS;

layout(push_constant) uniform PushConstants {
  int M;        // numTilesPaddedRun (always ÷8)
  int N;        // numOutChannelsPadded (always ÷8)
  int strideA;  // row-major packed A vec4 units between Winograd batches
  int strideB;  // K*(N/4) — vec4 units between Winograd batches in B
  int strideC;  // (N/4)*M — vec4 units between Winograd batches in C
}
pc;

layout(binding = 0) readonly buffer ABufV4 {
  vec4 data[];
}
aBufV4;
layout(binding = 1) readonly buffer BBufV4 {
  vec4 data[];
}
bBufV4;
layout(binding = 2) writeonly buffer CBufV4 {
  vec4 data[];
}
cBufV4;
layout(binding = 3) readonly buffer ABufV4H {
  f16vec4 data[];
}
aBufV4H;
layout(binding = 4) readonly buffer BBufV4H {
  f16vec4 data[];
}
bBufV4H;
layout(binding = 5) writeonly buffer CBufV4H {
  f16vec4 data[];
}
cBufV4H;

// Load 4 consecutive K-values for one M row inside the current packed A tile.
vec4 ldA4(int idx) {
  if(USE_FP16_STORAGE == 1)
    return vec4(aBufV4H.data[idx]);
  return aBufV4.data[idx];
}

// Load 4 consecutive N-values for fixed k.
vec4 ldB4(int base_v4, int k, int n4) {
  if(USE_FP16_STORAGE == 1)
    return vec4(bBufV4H.data[base_v4 + k * (pc.N / 4) + n4]);
  return bBufV4.data[base_v4 + k * (pc.N / 4) + n4];
}

// Store 4 consecutive M-values for fixed n.
void stC4(int base_v4, int n, int m0_4, vec4 v) {
  int idx = base_v4 + n * (pc.M / 4) + m0_4;
  if(USE_FP16_STORAGE == 1)
    cBufV4H.data[idx] = f16vec4(v);
  else
    cBufV4.data[idx] = v;
}

// Shared A is K-major [K][M+pad]. Global A remains row-major packed for the
// transform writer, but GEMM wants contiguous M values in the hot inner loop.
const int TILE_N4 = WG_Y * RN / 4;  // TILE_N / 4  (RN always ÷4)
const int A_SHMEM_STRIDE = TILE_M + 4;
shared float tileA[TILE_K * A_SHMEM_STRIDE];
shared vec4 tileB[TILE_K * (TILE_N4 + 1)];

void main() {
  int lx = int(gl_LocalInvocationID.x);
  int ly = int(gl_LocalInvocationID.y);
  int batch = int(gl_GlobalInvocationID.z);

  int mBase = int(gl_WorkGroupID.x) * TILE_M;
  int nBase = int(gl_WorkGroupID.y) * TILE_N;
  int m0 = mBase + lx * 4;
  int myN = nBase + ly * RN;

  int aBase_v4 = batch * pc.strideA;  // vec4 units
  int bBase_v4 = batch * pc.strideB;
  int cBase_v4 = batch * pc.strideC;
  int kBlocks = (K_SPEC + TILE_K - 1) / TILE_K;

  vec4 acc[RN];
  for(int j = 0; j < RN; j++)
    acc[j] = vec4(0.0);

  int T = WG_X * WG_Y;
  int tid = ly * WG_X + lx;

  for(int kBase = 0; kBase < K_SPEC; kBase += TILE_K) {
    // Cooperative A-tile load: TILE_M*(TILE_K/4) vec4s, transposed into scalar LDS [K][M].
    int aTileBase = aBase_v4 + (int(gl_WorkGroupID.x) * kBlocks + kBase / TILE_K) * TILE_M * A_ROW_WORDS;
    for(int i = tid; i < TILE_M * TILE_K4; i += T) {
      int mm = i / TILE_K4;
      int kk4 = i - mm * TILE_K4;
      int gm = mBase + mm;
      vec4 a4 = (gm < pc.M && kBase + kk4 * 4 < K_SPEC) ? ldA4(aTileBase + mm * A_ROW_WORDS + kk4) : vec4(0.0);
      int dst = kk4 * 4 * A_SHMEM_STRIDE + mm;
      tileA[dst + 0 * A_SHMEM_STRIDE] = a4.x;
      tileA[dst + 1 * A_SHMEM_STRIDE] = a4.y;
      tileA[dst + 2 * A_SHMEM_STRIDE] = a4.z;
      tileA[dst + 3 * A_SHMEM_STRIDE] = a4.w;
    }
    // Cooperative B-tile load: TILE_K*TILE_N4 vec4s directly into LDS.
    for(int i = tid; i < TILE_K * TILE_N4; i += T) {
      int kk = i / TILE_N4;
      int n4i = i - kk * TILE_N4;
      int gk = kBase + kk;
      int gn4 = nBase / 4 + n4i;
      tileB[kk * (TILE_N4 + 1) + n4i] = (gk < K_SPEC && gn4 * 4 < pc.N) ? ldB4(bBase_v4, gk, gn4) : vec4(0.0);
    }
    barrier();

    for(int kk = 0; kk < TILE_K; kk++) {
      int m = lx * 4;
      vec4 a = vec4(
        tileA[kk * A_SHMEM_STRIDE + m + 0],
        tileA[kk * A_SHMEM_STRIDE + m + 1],
        tileA[kk * A_SHMEM_STRIDE + m + 2],
        tileA[kk * A_SHMEM_STRIDE + m + 3]);
      int bOff = kk * (TILE_N4 + 1) + ly * (RN / 4);
      for(int j4 = 0; j4 < RN / 4; j4++) {
        vec4 b = tileB[bOff + j4];
        acc[j4 * 4 + 0] += a * b.x;
        acc[j4 * 4 + 1] += a * b.y;
        acc[j4 * 4 + 2] += a * b.z;
        acc[j4 * 4 + 3] += a * b.w;
      }
    }
    barrier();
  }

  for(int j = 0; j < RN; j++) {
    int n = myN + j;
    if(n < pc.N && m0 < pc.M)
      stC4(cBase_v4, n, m0 / 4, acc[j]);
  }
}

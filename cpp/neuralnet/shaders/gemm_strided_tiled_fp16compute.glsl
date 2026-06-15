// GLSL Compute Shader: Strided batched GEMM -- FP16 storage + FP16 shared-memory tiles and accumulation
// A: [batch, K, M] -- M is contiguous (paddedSpatialSize, always /8)
// B: [batch, K, N] -- N is contiguous (outChannelsPadded, always /8)
// C: [batch, N, M] -- M is contiguous
//
// This is the experimental full-FP16-compute sibling of gemm_strided_tiled.glsl.
// It is selected only for FP16 storage; global I/O, shared-memory tiles, and
// accumulators all use 16-bit float types.
//
// M must be /8, NN=WG_Y*RN must be /4 (host guarantees both); K is unconstrained.
// Dispatch: global(ceil(M/(WG_X*4)), ceil(N/(WG_Y*RN)), batch)

#version 450
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require

#ifdef KATAGO_VULKAN_RTE_F16
#extension GL_EXT_spirv_intrinsics : require
spirv_execution_mode(extensions = ["SPV_KHR_float_controls"], capabilities = [4467], 4462, 16);
#endif

layout(constant_id = 0) const int WG_X = 8;
layout(constant_id = 1) const int WG_Y = 8;
layout(constant_id = 2) const int TILE_K = 8;
layout(constant_id = 3) const int RN = 4;
layout(constant_id = 4) const int ADD_TO_OUTPUT = 0;
layout(constant_id = 5) const int K_SPEC = 1;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z = 1) in;

layout(push_constant) uniform PushConstants {
  int M;        // paddedSpatialSize (/8)
  int N;        // outChannelsPadded (/8) -- used for strides and loop bounds
  int N_real;   // outChannels (real, unpadded) -- used only for the C-store guard
  int strideA;  // K*(M/4) -- vec4 units per batch in A; shader converts base to mat2x4 units via /2.
  int strideB;  // K*(N/4) -- vec4 units per batch in B (0 if filter reused)
  int strideC;  // N_real*(M/4) -- vec4 units per batch in C (real N, not padded)
}
pc;

layout(binding = 0) readonly buffer ABufV8H {
  f16mat2x4 data[];
}
aBufV8H;
layout(binding = 1) readonly buffer BBufV4H {
  f16vec4 data[];
}
bBufV4H;
layout(binding = 2) buffer CBufV4H {
  f16vec4 data[];
}
cBufV4H;

// Load 8 consecutive M-values for fixed k.
// base_v8 = batch * (strideA/2) in mat2x4 units (because 1 mat2x4 == 2 vec4).
f16mat2x4 ldA8(int base_v8, int k, int m8) {
  return aBufV8H.data[base_v8 + k * (pc.M / 8) + m8];
}

// Load 4 consecutive N-values for fixed k.
f16vec4 ldB4(int base_v4, int k, int n4) {
  return bBufV4H.data[base_v4 + k * (pc.N / 4) + n4];
}

// Store 4 consecutive M-values for fixed n. Uses N_real row-stride (unpadded).
void stC4(int base_v4, int n, int m0_4, f16vec4 v) {
  int idx = base_v4 + n * (pc.M / 4) + m0_4;
  if(ADD_TO_OUTPUT != 0)
    v += cBufV4H.data[idx];
  cBufV4H.data[idx] = v;
}

const int MM = WG_X * 4;
const int MM4 = MM / 4;
const int MM8 = MM / 8;
const int NN = WG_Y * RN;
shared f16vec4 tileA[TILE_K * (MM4 + 1)];
shared float16_t tileB[TILE_K * (NN + 1)];

void main() {
  int lx = int(gl_LocalInvocationID.x);
  int ly = int(gl_LocalInvocationID.y);
  int batch = int(gl_GlobalInvocationID.z);

  int mBase = int(gl_WorkGroupID.x) * MM;
  int nBase = int(gl_WorkGroupID.y) * NN;
  int m0 = mBase + lx * 4;
  int myN = nBase + ly * RN;

  int aBase_v4 = batch * pc.strideA;  // vec4 units from host
  int aBase_v8 = aBase_v4 / 2;        // mat2x4 units (1 mat2x4 == 2 vec4)
  int bBase_v4 = batch * pc.strideB;
  int cBase_v4 = batch * pc.strideC;

  f16vec4 acc[RN];
  for(int j = 0; j < RN; j++)
    acc[j] = f16vec4(0.0hf);

  int T = WG_X * WG_Y;
  int tid = ly * WG_X + lx;

  for(int kBase = 0; kBase < K_SPEC; kBase += TILE_K) {
    // Cooperative A-tile load: TILE_K*MM8 mat2x4s (8 M-values each), unpacked into vec4 LDS.
    for(int i = tid; i < TILE_K * MM8; i += T) {
      int kk = i / MM8;
      int mm8 = i - kk * MM8;
      int gk = kBase + kk;
      int gm8 = mBase / 8 + mm8;
      int mm4 = mm8 * 2;
      f16mat2x4 a8 =
        (gk < K_SPEC && gm8 * 8 < pc.M) ? ldA8(aBase_v8, gk, gm8) : f16mat2x4(f16vec4(0.0hf), f16vec4(0.0hf));
      tileA[kk * (MM4 + 1) + mm4 + 0] = a8[0];
      tileA[kk * (MM4 + 1) + mm4 + 1] = a8[1];
    }
    // Cooperative B-tile load: TILE_K*(NN/4) vec4s -> 4 tileB columns each.
    for(int i = tid; i < TILE_K * (NN / 4); i += T) {
      int kk = i / (NN / 4);
      int n4i = i - kk * (NN / 4);
      int gk = kBase + kk;
      int gn4 = nBase / 4 + n4i;
      f16vec4 b4 = (gk < K_SPEC && gn4 * 4 < pc.N) ? ldB4(bBase_v4, gk, gn4) : f16vec4(0.0hf);
      tileB[kk * (NN + 1) + n4i * 4 + 0] = b4.x;
      tileB[kk * (NN + 1) + n4i * 4 + 1] = b4.y;
      tileB[kk * (NN + 1) + n4i * 4 + 2] = b4.z;
      tileB[kk * (NN + 1) + n4i * 4 + 3] = b4.w;
    }
    barrier();

    for(int kk = 0; kk < TILE_K; kk++) {
      f16vec4 a = tileA[kk * (MM4 + 1) + lx];
      int bOff = kk * (NN + 1) + ly * RN;
      for(int j = 0; j < RN; j++)
        acc[j] += a * tileB[bOff + j];
    }
    barrier();
  }

  for(int j = 0; j < RN; j++) {
    int n = myN + j;
    if(n < pc.N_real && m0 < pc.M)
      stC4(cBase_v4, n, m0 / 4, acc[j]);
  }
}

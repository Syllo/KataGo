// GLSL Compute Shader: native-NHWC strided batched GEMM.
// A: [batch, M, K] -- K is contiguous (input channels)
// B: [batch, K, N] -- N is contiguous (outChannelsPadded, always /8)
// C: [batch, M, N_real]
//
// FP32 and FP16 storage use the same tile geometry. Global FP16 I/O is
// widened/narrowed at the boundary; shared tiles and accumulation remain FP32.
// layout is [batch, K, M] / [batch, N, M].
//
// M must be /4, N must be /8, and K is unconstrained. Each workgroup covers
// (WG_X * 4) M rows by (WG_Y * RN) N columns.

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
layout(constant_id = 4) const int ADD_TO_OUTPUT = 0;
layout(constant_id = 5) const int USE_FP16_STORAGE = 0;
layout(constant_id = 6) const int K_SPEC = 1;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z = 1) in;

layout(push_constant) uniform PushConstants {
  int M;
  int N;
  int N_real;
  int strideA;  // K*M/4 vec4 units per batch
  int strideB;  // K*N/4 vec4 units per batch (zero for shared filters)
  int strideC;  // M*N_real/4 vec4 units per batch
}
pc;

layout(binding = 0) readonly buffer ABuf {
  float data[];
}
aBuf;
layout(binding = 1) readonly buffer BBufV4 {
  vec4 data[];
}
bBufV4;
layout(binding = 2) buffer CBuf {
  float data[];
}
cBuf;
layout(binding = 3) readonly buffer ABufH {
  float16_t data[];
}
aBufH;
layout(binding = 4) readonly buffer BBufV4H {
  f16vec4 data[];
}
bBufV4H;
layout(binding = 5) buffer CBufH {
  float16_t data[];
}
cBufH;
// Vector-typed aliases of bindings 3 and 5, bound only by the FP16 storage
// pipeline. In NHWC the contiguous axis of A and C is the channel axis, so a
// four-channel slab is one 64-bit access rather than four separately addressed
// scalar ones. Arithmetic remains FP32.
layout(binding = 6) readonly buffer ABufV4H {
  f16vec4 data[];
}
aBufV4H;
layout(binding = 7) buffer CBufV4H {
  f16vec4 data[];
}
cBufV4H;

float ldA(int base, int m, int k) {
  int idx = base + m * K_SPEC + k;
  return USE_FP16_STORAGE == 1 ? float(aBufH.data[idx]) : aBuf.data[idx];
}

// Four contiguous K values (input channels) of one NHWC row.
vec4 ldA4(int base, int m, int k) {
  return vec4(aBufV4H.data[(base + m * K_SPEC + k) / 4]);
}

vec4 ldB4(int base_v4, int k, int n4) {
  int idx = base_v4 + k * (pc.N / 4) + n4;
  return USE_FP16_STORAGE == 1 ? vec4(bBufV4H.data[idx]) : bBufV4.data[idx];
}

void stC(int base, int m, int n, float v) {
  int idx = base + m * pc.N_real + n;
  if(USE_FP16_STORAGE == 1) {
    if(ADD_TO_OUTPUT != 0)
      v += float(cBufH.data[idx]);
    cBufH.data[idx] = float16_t(v);
  } else {
    if(ADD_TO_OUTPUT != 0)
      v += cBuf.data[idx];
    cBuf.data[idx] = v;
  }
}

// Four contiguous N values (output channels) of one NHWC row.
void stC4(int base, int m, int n, vec4 v) {
  int idx4 = (base + m * pc.N_real + n) / 4;
  if(ADD_TO_OUTPUT != 0)
    v += vec4(cBufV4H.data[idx4]);
  cBufV4H.data[idx4] = f16vec4(v);
}

// A is gathered into vec4s holding four adjacent M rows for one k, which is the
// orientation the inner loop reads. The +1 row padding avoids shared-memory bank
// conflicts; B remains scalar because RN need not be /4.
const int MM = WG_X * 4;
const int NN = WG_Y * RN;
shared vec4 tileA[TILE_K * (WG_X + 1)];
shared float tileB[TILE_K * (NN + 1)];

void main() {
  int lx = int(gl_LocalInvocationID.x);
  int ly = int(gl_LocalInvocationID.y);
  int batch = int(gl_GlobalInvocationID.z);

  int mBase = int(gl_WorkGroupID.x) * MM;
  int nBase = int(gl_WorkGroupID.y) * NN;
  int m0 = mBase + lx * 4;
  int myN = nBase + ly * RN;

  // The common GEMM ABI expresses batch strides in vec4 units. NHWC A/C are
  // scalar-addressed, so convert their bases back to scalar elements here.
  int aBase = batch * pc.strideA * 4;
  int bBase_v4 = batch * pc.strideB;
  int cBase = batch * pc.strideC * 4;

  vec4 acc[RN];
  for(int j = 0; j < RN; j++)
    acc[j] = vec4(0.0);

  int threadCount = WG_X * WG_Y;
  int tid = ly * WG_X + lx;

  for(int kBase = 0; kBase < K_SPEC; kBase += TILE_K) {
    // Stage the A tile K-major as vec4 groups of four adjacent M rows, which is
    // what the inner loop reads. The gather direction differs by storage format:
    // in NHWC the contiguous global axis is K, so when a four-K slab is aligned
    // and in range, one lane reads it with a single 64-bit load and scatters it
    // across four K-major rows at a fixed M slot. Otherwise fall back to walking
    // M for a fixed k, which costs four strided scalar loads per entry.
    if(USE_FP16_STORAGE == 1 && TILE_K % 4 == 0 && K_SPEC % 4 == 0) {
      const int tileK4 = TILE_K / 4;
      for(int i = tid; i < MM * tileK4; i += threadCount) {
        int localM = i / tileK4;
        int localK = (i - localM * tileK4) * 4;
        int gm = mBase + localM;
        int gk = kBase + localK;
        vec4 a = (gm < pc.M && gk + 3 < K_SPEC) ? ldA4(aBase, gm, gk) : vec4(0.0);
        int slot = localM >> 2;
        int lane = localM & 3;
        tileA[(localK + 0) * (WG_X + 1) + slot][lane] = a.x;
        tileA[(localK + 1) * (WG_X + 1) + slot][lane] = a.y;
        tileA[(localK + 2) * (WG_X + 1) + slot][lane] = a.z;
        tileA[(localK + 3) * (WG_X + 1) + slot][lane] = a.w;
      }
    } else {
      for(int i = tid; i < TILE_K * WG_X; i += threadCount) {
        int kk = i / WG_X;
        int mx = i - kk * WG_X;
        int gk = kBase + kk;
        int gm0 = mBase + mx * 4;
        vec4 a = vec4(0.0);
        if(gk < K_SPEC) {
          if(gm0 + 0 < pc.M)
            a.x = ldA(aBase, gm0 + 0, gk);
          if(gm0 + 1 < pc.M)
            a.y = ldA(aBase, gm0 + 1, gk);
          if(gm0 + 2 < pc.M)
            a.z = ldA(aBase, gm0 + 2, gk);
          if(gm0 + 3 < pc.M)
            a.w = ldA(aBase, gm0 + 3, gk);
        }
        tileA[kk * (WG_X + 1) + mx] = a;
      }
    }

    // B is K-major with four contiguous N values per vec4, matching the
    // original strided tiled kernel's storage and filter packing.
    for(int i = tid; i < TILE_K * (NN / 4); i += threadCount) {
      int kk = i / (NN / 4);
      int n4i = i - kk * (NN / 4);
      int gk = kBase + kk;
      int gn4 = nBase / 4 + n4i;
      vec4 b = (gk < K_SPEC && gn4 * 4 < pc.N) ? ldB4(bBase_v4, gk, gn4) : vec4(0.0);
      tileB[kk * (NN + 1) + n4i * 4 + 0] = b.x;
      tileB[kk * (NN + 1) + n4i * 4 + 1] = b.y;
      tileB[kk * (NN + 1) + n4i * 4 + 2] = b.z;
      tileB[kk * (NN + 1) + n4i * 4 + 3] = b.w;
    }
    barrier();

    for(int kk = 0; kk < TILE_K; kk++) {
      vec4 a = tileA[kk * (WG_X + 1) + lx];
      int bOff = kk * (NN + 1) + ly * RN;
      for(int j = 0; j < RN; j++)
        acc[j] += a * tileB[bOff + j];
    }
    barrier();
  }

  if(USE_FP16_STORAGE == 1 && RN % 4 == 0 && pc.N_real % 4 == 0) {
    // Accumulators are M-major vec4s across four spatial rows, but NHWC wants
    // contiguous channels. Transpose each 4x4 register block on store so one
    // 64-bit access replaces four scalar stores strided by N_real.
    for(int mi = 0; mi < 4; mi++) {
      int m = m0 + mi;
      if(m >= pc.M)
        continue;
      for(int j = 0; j < RN; j += 4) {
        int n = myN + j;
        if(n + 3 < pc.N_real)
          stC4(cBase, m, n, vec4(acc[j + 0][mi], acc[j + 1][mi], acc[j + 2][mi], acc[j + 3][mi]));
      }
    }
  } else {
    for(int j = 0; j < RN; j++) {
      int n = myN + j;
      if(n < pc.N_real) {
        if(m0 + 0 < pc.M)
          stC(cBase, m0 + 0, n, acc[j].x);
        if(m0 + 1 < pc.M)
          stC(cBase, m0 + 1, n, acc[j].y);
        if(m0 + 2 < pc.M)
          stC(cBase, m0 + 2, n, acc[j].z);
        if(m0 + 3 < pc.M)
          stC(cBase, m0 + 3, n, acc[j].w);
      }
    }
  }
}

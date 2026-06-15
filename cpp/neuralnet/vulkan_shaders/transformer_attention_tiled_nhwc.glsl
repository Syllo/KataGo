// GLSL Compute Shader: Tiled transformer attention, NHWC Q/K/V/output layout.

#version 450
#extension GL_EXT_shader_16bit_storage : enable
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : enable

#ifdef KATAGO_VULKAN_RTE_F16
#extension GL_EXT_spirv_intrinsics : require
spirv_execution_mode(extensions = ["SPV_KHR_float_controls"], capabilities = [4467], 4462, 16);
#endif

layout(constant_id = 0) const int ATTN_BLOCK_Q = 32;
layout(constant_id = 1) const int ATTN_BLOCK_KV = 32;
layout(constant_id = 2) const int Q_PER_THREAD = 1;
layout(constant_id = 3) const int ATTN_HEAD_DIM = 64;
layout(constant_id = 4) const int ATTN_V_HEAD_DIM = 64;
layout(constant_id = 5) const int USE_FP16_STORAGE = 0;
layout(constant_id = 6) const int APPLY_ROPE = 0;
layout(constant_id = 7) const int LEARNABLE_ROPE = 0;
layout(constant_id = 8) const int SEQ_LEN_SPEC = 1;
layout(local_size_x_id = 0, local_size_y = 1, local_size_z = 1) in;

layout(push_constant) uniform PushConstants {
  int numHeads;
  int numKVHeads;
  float scale;
  int numBH;
}
pc;

layout(binding = 0) readonly buffer QBuf {
  float data[];
}
qBuf;
layout(binding = 1) readonly buffer KBuf {
  float data[];
}
kBuf;
layout(binding = 2) readonly buffer VBuf {
  float data[];
}
vBuf;
layout(binding = 3) writeonly buffer OutBuf {
  float data[];
}
outBuf;
layout(binding = 4) readonly buffer MaskBuf {
  float data[];
}
maskBuf;
layout(binding = 5) readonly buffer QBufH {
  float16_t data[];
}
qBufH;
layout(binding = 6) readonly buffer KBufH {
  float16_t data[];
}
kBufH;
layout(binding = 7) readonly buffer VBufH {
  float16_t data[];
}
vBufH;
layout(binding = 8) writeonly buffer OutBufH {
  float16_t data[];
}
outBufH;
layout(binding = 9) readonly buffer MaskBufH {
  float16_t data[];
}
maskBufH;
layout(binding = 10) readonly buffer CosBuf {
  float data[];
}
cosBuf;
layout(binding = 11) readonly buffer SinBuf {
  float data[];
}
sinBuf;

shared float kTile[ATTN_BLOCK_KV * ATTN_HEAD_DIM];
shared float vTile[ATTN_BLOCK_KV * ATTN_V_HEAD_DIM];
shared float kMaskTile[ATTN_BLOCK_KV];
shared float blockMaskMax[1];

// FA4 conditional rescaling, expressed in base-2 units. The running origin is
// allowed to lag the true maximum by at most 8, bounding weights by 2^8 while
// avoiding most full value-accumulator rescale operations.
const float LOG2_E = 1.4426950408889634;
const float SOFTMAX_RESCALE_THRESHOLD = 8.0;

float ldMask(int n, int pos) {
  int idx = n * SEQ_LEN_SPEC + pos;
  return USE_FP16_STORAGE == 1 ? float(float16_t(maskBufH.data[idx])) : maskBuf.data[idx];
}
float ldQ(int n, int h, int d, int pos) {
  int idx = (n * SEQ_LEN_SPEC + pos) * (pc.numHeads * ATTN_HEAD_DIM) + h * ATTN_HEAD_DIM + d;
  return USE_FP16_STORAGE == 1 ? float(float16_t(qBufH.data[idx])) : qBuf.data[idx];
}
float ldK(int n, int h, int d, int pos) {
  int idx = (n * SEQ_LEN_SPEC + pos) * (pc.numKVHeads * ATTN_HEAD_DIM) + h * ATTN_HEAD_DIM + d;
  return USE_FP16_STORAGE == 1 ? float(float16_t(kBufH.data[idx])) : kBuf.data[idx];
}
float ldV(int n, int h, int d, int pos) {
  int idx = (n * SEQ_LEN_SPEC + pos) * (pc.numKVHeads * ATTN_V_HEAD_DIM) + h * ATTN_V_HEAD_DIM + d;
  return USE_FP16_STORAGE == 1 ? float(float16_t(vBufH.data[idx])) : vBuf.data[idx];
}
void stOut(int n, int h, int d, int pos, float v) {
  int idx = (n * SEQ_LEN_SPEC + pos) * (pc.numHeads * ATTN_V_HEAD_DIM) + h * ATTN_V_HEAD_DIM + d;
  if(USE_FP16_STORAGE == 1)
    outBufH.data[idx] = float16_t(v);
  else
    outBuf.data[idx] = v;
}
int ropeTableIdx(int kvh, int pairIdx, int pos) {
  return LEARNABLE_ROPE == 1 ? (kvh * (ATTN_HEAD_DIM / 2) + pairIdx) * SEQ_LEN_SPEC + pos
                             : pairIdx * SEQ_LEN_SPEC + pos;
}

void main() {
  int localIdx = int(gl_LocalInvocationID.x);
  int qBlockStart = int(gl_WorkGroupID.x) * (ATTN_BLOCK_Q * Q_PER_THREAD);
  int bh = int(gl_GlobalInvocationID.y);
  if(bh >= pc.numBH)
    return;
  int n = bh / pc.numHeads;
  int h = bh % pc.numHeads;
  int kvh = pc.numKVHeads == pc.numHeads ? h : h / (pc.numHeads / pc.numKVHeads);

  float q[Q_PER_THREAD * ATTN_HEAD_DIM];
  float qMask[Q_PER_THREAD];
  float runMax[Q_PER_THREAD];
  float runSum[Q_PER_THREAD];
  float acc[Q_PER_THREAD * ATTN_V_HEAD_DIM];

  for(int qi = 0; qi < Q_PER_THREAD; qi++) {
    int qPos = qBlockStart + qi * ATTN_BLOCK_Q + localIdx;
    qMask[qi] = 0.0;
    if(qPos < SEQ_LEN_SPEC) {
      qMask[qi] = ldMask(n, qPos);
      if(qMask[qi] != 0.0) {
        for(int d = 0; d < ATTN_HEAD_DIM; d++)
          q[qi * ATTN_HEAD_DIM + d] = ldQ(n, h, d, qPos);
        if(APPLY_ROPE == 1) {
          for(int pairIdx = 0; pairIdx < ATTN_HEAD_DIM / 2; pairIdx++) {
            int d0 = pairIdx * 2;
            int d1 = d0 + 1;
            float x0 = q[qi * ATTN_HEAD_DIM + d0];
            float x1 = q[qi * ATTN_HEAD_DIM + d1];
            int tableIdx = ropeTableIdx(kvh, pairIdx, qPos);
            float cv = cosBuf.data[tableIdx];
            float sv = sinBuf.data[tableIdx];
            q[qi * ATTN_HEAD_DIM + d0] = x0 * cv - x1 * sv;
            q[qi * ATTN_HEAD_DIM + d1] = x0 * sv + x1 * cv;
          }
        }
      }
    }
    runMax[qi] = -1e30;
    runSum[qi] = 0.0;
    for(int d = 0; d < ATTN_V_HEAD_DIM; d++)
      acc[qi * ATTN_V_HEAD_DIM + d] = 0.0;
  }

  for(int kvStart = 0; kvStart < SEQ_LEN_SPEC; kvStart += ATTN_BLOCK_KV) {
    if(localIdx < ATTN_BLOCK_KV) {
      int pos = kvStart + localIdx;
      kMaskTile[localIdx] = pos < SEQ_LEN_SPEC ? ldMask(n, pos) : 0.0;
    }
    barrier();
    if(localIdx == 0) {
      float m = 0.0;
      for(int kv = 0; kv < ATTN_BLOCK_KV; kv++)
        m = max(m, kMaskTile[kv]);
      blockMaskMax[0] = m;
    }
    barrier();
    if(blockMaskMax[0] == 0.0)
      continue;

    for(int t = localIdx; t < ATTN_BLOCK_KV * ATTN_HEAD_DIM; t += ATTN_BLOCK_Q) {
      int row = t / ATTN_HEAD_DIM;
      int d = t % ATTN_HEAD_DIM;
      int pos = kvStart + row;
      kTile[t] = pos < SEQ_LEN_SPEC ? ldK(n, kvh, d, pos) : 0.0;
    }
    for(int t = localIdx; t < ATTN_BLOCK_KV * ATTN_V_HEAD_DIM; t += ATTN_BLOCK_Q) {
      int row = t / ATTN_V_HEAD_DIM;
      int d = t % ATTN_V_HEAD_DIM;
      int pos = kvStart + row;
      vTile[t] = pos < SEQ_LEN_SPEC ? ldV(n, kvh, d, pos) : 0.0;
    }
    barrier();

    if(APPLY_ROPE == 1) {
      for(int t = localIdx; t < ATTN_BLOCK_KV * (ATTN_HEAD_DIM / 2); t += ATTN_BLOCK_Q) {
        int row = t / (ATTN_HEAD_DIM / 2);
        int pairIdx = t % (ATTN_HEAD_DIM / 2);
        int pos = kvStart + row;
        if(pos < SEQ_LEN_SPEC && kMaskTile[row] != 0.0) {
          int d0 = pairIdx * 2;
          int d1 = d0 + 1;
          float x0 = kTile[row * ATTN_HEAD_DIM + d0];
          float x1 = kTile[row * ATTN_HEAD_DIM + d1];
          int tableIdx = ropeTableIdx(kvh, pairIdx, pos);
          float cv = cosBuf.data[tableIdx];
          float sv = sinBuf.data[tableIdx];
          kTile[row * ATTN_HEAD_DIM + d0] = x0 * cv - x1 * sv;
          kTile[row * ATTN_HEAD_DIM + d1] = x0 * sv + x1 * cv;
        }
      }
      barrier();
    }

    for(int qi = 0; qi < Q_PER_THREAD; qi++) {
      int qPos = qBlockStart + qi * ATTN_BLOCK_Q + localIdx;
      if(qPos < SEQ_LEN_SPEC && qMask[qi] != 0.0) {
        for(int kv = 0; kv < ATTN_BLOCK_KV; kv++) {
          int pos = kvStart + kv;
          if(pos >= SEQ_LEN_SPEC || kMaskTile[kv] == 0.0)
            continue;
          float score = 0.0;
          int d = 0;
          // NHWC head channels are contiguous, so expose four-channel dot
          // products to the compiler while retaining a generic scalar tail.
          for(; d + 3 < ATTN_HEAD_DIM; d += 4) {
            vec4 qv = vec4(
              q[qi * ATTN_HEAD_DIM + d],
              q[qi * ATTN_HEAD_DIM + d + 1],
              q[qi * ATTN_HEAD_DIM + d + 2],
              q[qi * ATTN_HEAD_DIM + d + 3]);
            vec4 kvv = vec4(
              kTile[kv * ATTN_HEAD_DIM + d],
              kTile[kv * ATTN_HEAD_DIM + d + 1],
              kTile[kv * ATTN_HEAD_DIM + d + 2],
              kTile[kv * ATTN_HEAD_DIM + d + 3]);
            score += dot(qv, kvv);
          }
          for(; d < ATTN_HEAD_DIM; d++)
            score += q[qi * ATTN_HEAD_DIM + d] * kTile[kv * ATTN_HEAD_DIM + d];
          score *= pc.scale * LOG2_E;
          if(score > runMax[qi] + SOFTMAX_RESCALE_THRESHOLD) {
            float rescale = exp2(runMax[qi] - score);
            runSum[qi] *= rescale;
            for(int d = 0; d < ATTN_V_HEAD_DIM; d++)
              acc[qi * ATTN_V_HEAD_DIM + d] *= rescale;
            runMax[qi] = score;
          }
          float weight = exp2(score - runMax[qi]);
          runSum[qi] += weight;
          for(int d = 0; d < ATTN_V_HEAD_DIM; d++)
            acc[qi * ATTN_V_HEAD_DIM + d] += weight * vTile[kv * ATTN_V_HEAD_DIM + d];
        }
      }
    }
    barrier();
  }

  for(int qi = 0; qi < Q_PER_THREAD; qi++) {
    int qPos = qBlockStart + qi * ATTN_BLOCK_Q + localIdx;
    if(qPos < SEQ_LEN_SPEC) {
      float invSum = runSum[qi] > 0.0 ? 1.0 / runSum[qi] : 0.0;
      for(int d = 0; d < ATTN_V_HEAD_DIM; d++)
        stOut(n, h, d, qPos, acc[qi * ATTN_V_HEAD_DIM + d] * invSum);
    }
  }
}

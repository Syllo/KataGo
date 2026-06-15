// GLSL Compute Shader: Tiled flash-attention style attention
// Ports OpenCLKernels::transformerScaledDotProductAttention
// Uses online softmax (streaming) and shared-memory tiles.
// Dispatch: global(ATTN_BLOCK_Q * ceil(seqLen/(ATTN_BLOCK_Q*Q_PER_THREAD)), N*numHeads, 1)
// Constraint: ATTN_BLOCK_KV <= ATTN_BLOCK_Q (one kMaskTile writer per localIdx).

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

layout(local_size_x_id = 0) in;  // ATTN_BLOCK_Q
layout(local_size_y = 1) in;
layout(local_size_z = 1) in;

layout(push_constant) uniform PushConstants {
  int numHeads;
  int numKVHeads;
  float scale;
  int numBH;  // batchSize * numHeads (true thread-grid extent in y)
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

// Shared memory tiles for K and V — sized by specialization constants.
shared float kTile[ATTN_BLOCK_KV * ATTN_HEAD_DIM];
shared float vTile[ATTN_BLOCK_KV * ATTN_V_HEAD_DIM];
shared float kMaskTile[ATTN_BLOCK_KV];
shared float blockMaskMax[1];

float ldMask(int n, int pos) {
  int idx = n * SEQ_LEN_SPEC + pos;
  return USE_FP16_STORAGE == 1 ? float(float16_t(maskBufH.data[idx])) : maskBuf.data[idx];
}
float ldQ(int bh, int d, int pos) {
  int idx = (bh * ATTN_HEAD_DIM + d) * SEQ_LEN_SPEC + pos;
  return USE_FP16_STORAGE == 1 ? float(float16_t(qBufH.data[idx])) : qBuf.data[idx];
}
float ldK(int kvBH, int d, int pos) {
  int idx = (kvBH * ATTN_HEAD_DIM + d) * SEQ_LEN_SPEC + pos;
  return USE_FP16_STORAGE == 1 ? float(float16_t(kBufH.data[idx])) : kBuf.data[idx];
}
float ldV(int kvBH, int d, int pos) {
  int idx = (kvBH * ATTN_V_HEAD_DIM + d) * SEQ_LEN_SPEC + pos;
  return USE_FP16_STORAGE == 1 ? float(float16_t(vBufH.data[idx])) : vBuf.data[idx];
}
void stOut(int bh, int d, int pos, float v) {
  int idx = (bh * ATTN_V_HEAD_DIM + d) * SEQ_LEN_SPEC + pos;
  if(USE_FP16_STORAGE == 1)
    outBufH.data[idx] = float16_t(v);
  else
    outBuf.data[idx] = v;
}

int ropeTableIdx(int kvh, int pairIdx, int pos) {
  if(LEARNABLE_ROPE == 1)
    return (kvh * (ATTN_HEAD_DIM / 2) + pairIdx) * SEQ_LEN_SPEC + pos;
  return pairIdx * SEQ_LEN_SPEC + pos;
}

void main() {
  int localIdx = int(gl_LocalInvocationID.x);
  int qBlockStart = int(gl_WorkGroupID.x) * (ATTN_BLOCK_Q * Q_PER_THREAD);
  int bh = int(gl_GlobalInvocationID.y);
  if(bh >= pc.numBH)
    return;

  int n = bh / pc.numHeads;
  int h = bh % pc.numHeads;
  int kvh = (pc.numKVHeads == pc.numHeads) ? h : h / (pc.numHeads / pc.numKVHeads);
  int kvBH = n * pc.numKVHeads + kvh;

  // Private state for Q_PER_THREAD positions each.
  // Thread qi handles query position: qBlockStart + qi * ATTN_BLOCK_Q + localIdx
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
          q[qi * ATTN_HEAD_DIM + d] = ldQ(bh, d, qPos);
        if(APPLY_ROPE == 1) {
          for(int pairIdx = 0; pairIdx < ATTN_HEAD_DIM / 2; pairIdx++) {
            int d0 = pairIdx * 2;
            int d1 = d0 + 1;
            float x0 = q[qi * ATTN_HEAD_DIM + d0];
            float x1 = q[qi * ATTN_HEAD_DIM + d1];
            int tableIdx = ropeTableIdx(kvh, pairIdx, qPos);
            float cosVal = cosBuf.data[tableIdx];
            float sinVal = sinBuf.data[tableIdx];
            q[qi * ATTN_HEAD_DIM + d0] = x0 * cosVal - x1 * sinVal;
            q[qi * ATTN_HEAD_DIM + d1] = x0 * sinVal + x1 * cosVal;
          }
        }
      }
    }
    runMax[qi] = -1e30;
    runSum[qi] = 0.0;
    for(int d = 0; d < ATTN_V_HEAD_DIM; d++)
      acc[qi * ATTN_V_HEAD_DIM + d] = 0.0;
  }

  // Iterate over KV tiles.
  for(int kvStart = 0; kvStart < SEQ_LEN_SPEC; kvStart += ATTN_BLOCK_KV) {
    // Load K mask cooperatively first so fully-masked tiles can skip K/V loads.
    if(localIdx < ATTN_BLOCK_KV) {
      int gkv = kvStart + localIdx;
      kMaskTile[localIdx] = (gkv < SEQ_LEN_SPEC) ? ldMask(n, gkv) : 0.0;
    }
    barrier();
    if(localIdx == 0) {
      float m = 0.0;
      for(int kv = 0; kv < ATTN_BLOCK_KV; kv++) {
        m = max(m, kMaskTile[kv]);
      }
      blockMaskMax[0] = m;
    }
    barrier();
    if(blockMaskMax[0] == 0.0) {
      continue;
    }

    // Load K tile cooperatively.
    for(int t = localIdx; t < ATTN_BLOCK_KV * ATTN_HEAD_DIM; t += ATTN_BLOCK_Q) {
      int tkv = t / ATTN_HEAD_DIM;
      int td = t % ATTN_HEAD_DIM;
      int gkv = kvStart + tkv;
      kTile[tkv * ATTN_HEAD_DIM + td] = (gkv < SEQ_LEN_SPEC) ? ldK(kvBH, td, gkv) : 0.0;
    }
    // Load V tile cooperatively.
    for(int t = localIdx; t < ATTN_BLOCK_KV * ATTN_V_HEAD_DIM; t += ATTN_BLOCK_Q) {
      int tkv = t / ATTN_V_HEAD_DIM;
      int td = t % ATTN_V_HEAD_DIM;
      int gkv = kvStart + tkv;
      vTile[tkv * ATTN_V_HEAD_DIM + td] = (gkv < SEQ_LEN_SPEC) ? ldV(kvBH, td, gkv) : 0.0;
    }
    barrier();

    // K RoPE rotation after full cooperative tile load.
    if(APPLY_ROPE == 1) {
      for(int t = localIdx; t < ATTN_BLOCK_KV * (ATTN_HEAD_DIM / 2); t += ATTN_BLOCK_Q) {
        int tkv = t / (ATTN_HEAD_DIM / 2);
        int pairIdx = t % (ATTN_HEAD_DIM / 2);
        int gkv = kvStart + tkv;
        if(gkv < SEQ_LEN_SPEC && kMaskTile[tkv] != 0.0) {
          int d0 = pairIdx * 2;
          int d1 = d0 + 1;
          float x0 = kTile[tkv * ATTN_HEAD_DIM + d0];
          float x1 = kTile[tkv * ATTN_HEAD_DIM + d1];
          int tableIdx = ropeTableIdx(kvh, pairIdx, gkv);
          float cosVal = cosBuf.data[tableIdx];
          float sinVal = sinBuf.data[tableIdx];
          kTile[tkv * ATTN_HEAD_DIM + d0] = x0 * cosVal - x1 * sinVal;
          kTile[tkv * ATTN_HEAD_DIM + d1] = x0 * sinVal + x1 * cosVal;
        }
      }
      barrier();
    }

    // Each thread processes Q_PER_THREAD query positions against the shared KV tile.
    // The KV tile load cost is amortised across all Q_PER_THREAD positions.
    for(int qi = 0; qi < Q_PER_THREAD; qi++) {
      int qPos = qBlockStart + qi * ATTN_BLOCK_Q + localIdx;
      if(qPos < SEQ_LEN_SPEC && qMask[qi] != 0.0) {
        for(int kv = 0; kv < ATTN_BLOCK_KV; kv++) {
          int gkv = kvStart + kv;
          if(gkv >= SEQ_LEN_SPEC || kMaskTile[kv] == 0.0)
            continue;

          float score = 0.0;
          for(int d = 0; d < ATTN_HEAD_DIM; d++)
            score += q[qi * ATTN_HEAD_DIM + d] * kTile[kv * ATTN_HEAD_DIM + d];
          score *= pc.scale;

          // Online softmax update.
          float newMax = max(runMax[qi], score);
          float rescale = exp(runMax[qi] - newMax);
          runSum[qi] = runSum[qi] * rescale + exp(score - newMax);
          runMax[qi] = newMax;
          float weight = exp(score - runMax[qi]);
          for(int d = 0; d < ATTN_V_HEAD_DIM; d++)
            acc[qi * ATTN_V_HEAD_DIM + d] =
              acc[qi * ATTN_V_HEAD_DIM + d] * rescale + weight * vTile[kv * ATTN_V_HEAD_DIM + d];
        }
      }
    }
    barrier();
  }

  // Write output for all Q_PER_THREAD positions.
  for(int qi = 0; qi < Q_PER_THREAD; qi++) {
    int qPos = qBlockStart + qi * ATTN_BLOCK_Q + localIdx;
    if(qPos < SEQ_LEN_SPEC) {
      float invSum = (runSum[qi] > 0.0) ? 1.0 / runSum[qi] : 0.0;
      for(int d = 0; d < ATTN_V_HEAD_DIM; d++)
        stOut(bh, d, qPos, acc[qi * ATTN_V_HEAD_DIM + d] * invSum);
    }
  }
}

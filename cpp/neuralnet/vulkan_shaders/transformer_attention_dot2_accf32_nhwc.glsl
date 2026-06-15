// NHWC transformer attention using FP16 DOT2 with FP32 accumulation.
//
// One invocation owns one Q row. Q and the output accumulator remain private
// for the entire dispatch; each KV tile stages packed K rows and V transposed
// to [dimension][kv], so both Q*K and P*V reduce in groups of four FP16 values
// with OpFDot2MixAcc32VALVE. Online softmax follows the same conditional FA4
// rescaling policy as the coopmat1 kernel.

#version 450

#extension GL_EXT_control_flow_attributes : enable
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_spirv_intrinsics : require

#ifdef KATAGO_VULKAN_RTE_F16
spirv_execution_mode(extensions = ["SPV_KHR_float_controls"], capabilities = [4467], 4462, 16);
#endif

spirv_instruction(
  extensions = ["SPV_VALVE_mixed_float_dot_product"],
  capabilities = [6912],
  id = 6916) float v_dot2_f32_f16(f16vec2 a, f16vec2 b, float acc);

float dotPackedV8(uvec4 a, uvec4 b, float acc) {
  acc = v_dot2_f32_f16(unpackFloat2x16(a.x), unpackFloat2x16(b.x), acc);
  acc = v_dot2_f32_f16(unpackFloat2x16(a.y), unpackFloat2x16(b.y), acc);
  acc = v_dot2_f32_f16(unpackFloat2x16(a.z), unpackFloat2x16(b.z), acc);
  return v_dot2_f32_f16(unpackFloat2x16(a.w), unpackFloat2x16(b.w), acc);
}

float dotV4(f16vec4 a, f16vec4 b, float acc) {
  return v_dot2_f32_f16(a.zw, b.zw, v_dot2_f32_f16(a.xy, b.xy, acc));
}

layout(constant_id = 0) const uint WORKGROUP_SIZE = 128;
layout(constant_id = 1) const uint BLOCK_Q = 128;
layout(constant_id = 2) const uint BLOCK_KV = 32;
layout(constant_id = 3) const uint HEAD_DIM = 32;
layout(constant_id = 4) const uint V_HEAD_DIM = 32;
layout(constant_id = 5) const uint APPLY_ROPE = 0;
layout(constant_id = 6) const uint LEARNABLE_ROPE = 0;
layout(constant_id = 7) const uint SEQ_LEN_SPEC = 1;
layout(local_size_x_id = 0, local_size_y = 1, local_size_z = 1) in;

layout(push_constant) uniform PushConstants {
  int numHeads;
  int numKVHeads;
  float scale;
  int numBH;
}
pc;

layout(binding = 0) readonly buffer QBuf {
  uvec4 data[];
}
qBuf;
layout(binding = 1) readonly buffer KBuf {
  uvec4 data[];
}
kBuf;
layout(binding = 2) readonly buffer VBuf {
  uvec4 data[];
}
vBuf;
layout(binding = 3) writeonly buffer OutBuf {
  uvec4 data[];
}
outBuf;
layout(binding = 4) readonly buffer MaskBuf {
  float16_t data[];
}
maskBuf;
layout(binding = 5) readonly buffer CosBuf {
  float data[];
}
cosBuf;
layout(binding = 6) readonly buffer SinBuf {
  float data[];
}
sinBuf;

const uint HEAD_COLS_V8 = HEAD_DIM / 8u;
const uint V_COLS_V8 = V_HEAD_DIM / 8u;
const uint K_STRIDE_V8 = HEAD_COLS_V8 + 1u;
const uint V_STRIDE = BLOCK_KV + 4u;

const float LOG2_E = 1.4426950408889634;
const float SOFTMAX_RESCALE_THRESHOLD = 8.0;
const float NEG_BIG = -1.7014118e38;

shared uvec4 kTile[BLOCK_KV * K_STRIDE_V8];
shared float16_t vTile[V_HEAD_DIM * V_STRIDE];
shared float kvMask[BLOCK_KV];

uint qIndex8(uint n, uint h, uint pos) {
  return ((n * SEQ_LEN_SPEC + pos) * (uint(pc.numHeads) * HEAD_DIM) + h * HEAD_DIM) / 8u;
}
uint kIndex8(uint n, uint h, uint pos) {
  return ((n * SEQ_LEN_SPEC + pos) * (uint(pc.numKVHeads) * HEAD_DIM) + h * HEAD_DIM) / 8u;
}
uint vIndex8(uint n, uint h, uint pos) {
  return ((n * SEQ_LEN_SPEC + pos) * (uint(pc.numKVHeads) * V_HEAD_DIM) + h * V_HEAD_DIM) / 8u;
}
uint outIndex8(uint n, uint h, uint pos) {
  return ((n * SEQ_LEN_SPEC + pos) * (uint(pc.numHeads) * V_HEAD_DIM) + h * V_HEAD_DIM) / 8u;
}
uint ropeTableIdx(uint kvh, uint pairIdx, uint pos) {
  return LEARNABLE_ROPE != 0u ? (kvh * (HEAD_DIM / 2u) + pairIdx) * SEQ_LEN_SPEC + pos : pairIdx * SEQ_LEN_SPEC + pos;
}

uint ropeScalePair(uint packed, uint kvh, uint pairIdx, uint pos, float s) {
  vec2 value = unpackHalf2x16(packed);
  if(APPLY_ROPE != 0u) {
    const uint ri = ropeTableIdx(kvh, pairIdx, pos);
    const float cv = cosBuf.data[ri];
    const float sv = sinBuf.data[ri];
    value = vec2(value.x * cv - value.y * sv, value.x * sv + value.y * cv);
  }
  return packHalf2x16(value * s);
}

uvec4 ropeScaleV8(uvec4 raw, uint kvh, uint d8, uint pos, float s) {
  const uint pairBase = d8 * 4u;
  return uvec4(
    ropeScalePair(raw.x, kvh, pairBase + 0u, pos, s),
    ropeScalePair(raw.y, kvh, pairBase + 1u, pos, s),
    ropeScalePair(raw.z, kvh, pairBase + 2u, pos, s),
    ropeScalePair(raw.w, kvh, pairBase + 3u, pos, s));
}

void main() {
  const uint tid = gl_LocalInvocationID.x;
  const uint qStart = gl_WorkGroupID.x * BLOCK_Q;
  const uint bh = gl_WorkGroupID.y;
  if(bh >= uint(pc.numBH))
    return;
  const uint n = bh / uint(pc.numHeads);
  const uint h = bh - n * uint(pc.numHeads);
  const uint headsPerKv = uint(pc.numHeads / pc.numKVHeads);
  const uint kvh = h / headsPerKv;
  const uint myPos = qStart + tid;
  const bool qActive = myPos < SEQ_LEN_SPEC && float(maskBuf.data[n * SEQ_LEN_SPEC + myPos]) != 0.0;

  uvec4 qPacked[HEAD_COLS_V8];
  const uint qBase8 = qActive ? qIndex8(n, h, myPos) : 0u;
  const float qScale = pc.scale * LOG2_E;
  [[unroll]] for(uint d8 = 0u; d8 < HEAD_COLS_V8; d8++)
    qPacked[d8] = qActive ? ropeScaleV8(qBuf.data[qBase8 + d8], kvh, d8, myPos, qScale) : uvec4(0u);

  float outAcc[V_HEAD_DIM];
  [[unroll]] for(uint d = 0u; d < V_HEAD_DIM; d++)
    outAcc[d] = 0.0;
  float rowSum = 0.0;
  float origin = NEG_BIG;

  for(uint kvStart = 0u; kvStart < SEQ_LEN_SPEC; kvStart += BLOCK_KV) {
    for(uint kv = tid; kv < BLOCK_KV; kv += WORKGROUP_SIZE) {
      const uint pos = kvStart + kv;
      kvMask[kv] = pos < SEQ_LEN_SPEC ? float(maskBuf.data[n * SEQ_LEN_SPEC + pos]) : 0.0;
    }
    barrier();

    bool anyActive = false;
    for(uint kv = 0u; kv < BLOCK_KV; kv++)
      anyActive = anyActive || kvMask[kv] != 0.0;

    if(anyActive) {
      for(uint i = tid; i < BLOCK_KV * HEAD_COLS_V8; i += WORKGROUP_SIZE) {
        const uint kv = i / HEAD_COLS_V8;
        const uint d8 = i - kv * HEAD_COLS_V8;
        const uint pos = kvStart + kv;
        uvec4 packed = uvec4(0u);
        if(pos < SEQ_LEN_SPEC && kvMask[kv] != 0.0) {
          const uvec4 raw = kBuf.data[kIndex8(n, kvh, pos) + d8];
          packed = APPLY_ROPE != 0u ? ropeScaleV8(raw, kvh, d8, pos, 1.0) : raw;
        }
        kTile[kv * K_STRIDE_V8 + d8] = packed;
      }
      for(uint i = tid; i < BLOCK_KV * V_COLS_V8; i += WORKGROUP_SIZE) {
        const uint kv = i / V_COLS_V8;
        const uint d8 = i - kv * V_COLS_V8;
        const uint pos = kvStart + kv;
        uvec4 packed = uvec4(0u);
        if(pos < SEQ_LEN_SPEC && kvMask[kv] != 0.0)
          packed = vBuf.data[vIndex8(n, kvh, pos) + d8];
        const f16vec2 p0 = unpackFloat2x16(packed.x);
        const f16vec2 p1 = unpackFloat2x16(packed.y);
        const f16vec2 p2 = unpackFloat2x16(packed.z);
        const f16vec2 p3 = unpackFloat2x16(packed.w);
        const uint d = d8 * 8u;
        vTile[(d + 0u) * V_STRIDE + kv] = p0.x;
        vTile[(d + 1u) * V_STRIDE + kv] = p0.y;
        vTile[(d + 2u) * V_STRIDE + kv] = p1.x;
        vTile[(d + 3u) * V_STRIDE + kv] = p1.y;
        vTile[(d + 4u) * V_STRIDE + kv] = p2.x;
        vTile[(d + 5u) * V_STRIDE + kv] = p2.y;
        vTile[(d + 6u) * V_STRIDE + kv] = p3.x;
        vTile[(d + 7u) * V_STRIDE + kv] = p3.y;
      }
      barrier();

      float scores[BLOCK_KV];
      float rowMax = NEG_BIG;
      [[unroll]] for(uint kv = 0u; kv < BLOCK_KV; kv++) {
        float score = NEG_BIG;
        if(qActive && kvMask[kv] != 0.0 && kvStart + kv < SEQ_LEN_SPEC) {
          score = 0.0;
          [[unroll]] for(uint d8 = 0u; d8 < HEAD_COLS_V8; d8++)
            score = dotPackedV8(qPacked[d8], kTile[kv * K_STRIDE_V8 + d8], score);
          rowMax = max(rowMax, score);
        }
        scores[kv] = score;
      }

      if(rowMax > origin + SOFTMAX_RESCALE_THRESHOLD) {
        const float rescale = exp2(origin - rowMax);
        origin = rowMax;
        rowSum *= rescale;
        [[unroll]] for(uint d = 0u; d < V_HEAD_DIM; d++)
          outAcc[d] *= rescale;
      }

      f16vec4 probabilities[BLOCK_KV / 4u];
      [[unroll]] for(uint kv4 = 0u; kv4 < BLOCK_KV / 4u; kv4++) {
        const uint kv = kv4 * 4u;
        float p0 = qActive && scores[kv + 0u] != NEG_BIG ? exp2(scores[kv + 0u] - origin) : 0.0;
        float p1 = qActive && scores[kv + 1u] != NEG_BIG ? exp2(scores[kv + 1u] - origin) : 0.0;
        float p2 = qActive && scores[kv + 2u] != NEG_BIG ? exp2(scores[kv + 2u] - origin) : 0.0;
        float p3 = qActive && scores[kv + 3u] != NEG_BIG ? exp2(scores[kv + 3u] - origin) : 0.0;
        rowSum += p0 + p1 + p2 + p3;
        probabilities[kv4] = f16vec4(p0, p1, p2, p3);
      }

      [[unroll]] for(uint d = 0u; d < V_HEAD_DIM; d++) {
        [[unroll]] for(uint kv4 = 0u; kv4 < BLOCK_KV / 4u; kv4++) {
          const uint kv = kv4 * 4u;
          const uint base = d * V_STRIDE + kv;
          const f16vec4 vv = f16vec4(vTile[base], vTile[base + 1u], vTile[base + 2u], vTile[base + 3u]);
          outAcc[d] = dotV4(probabilities[kv4], vv, outAcc[d]);
        }
      }
    }
    barrier();
  }

  if(myPos < SEQ_LEN_SPEC) {
    const float invSum = rowSum > 0.0 ? 1.0 / rowSum : 0.0;
    const uint outBase8 = outIndex8(n, h, myPos);
    [[unroll]] for(uint d8 = 0u; d8 < V_COLS_V8; d8++) {
      const uint d = d8 * 8u;
      outBuf.data[outBase8 + d8] = uvec4(
        packHalf2x16(vec2(outAcc[d + 0u], outAcc[d + 1u]) * invSum),
        packHalf2x16(vec2(outAcc[d + 2u], outAcc[d + 3u]) * invSum),
        packHalf2x16(vec2(outAcc[d + 4u], outAcc[d + 5u]) * invSum),
        packHalf2x16(vec2(outAcc[d + 6u], outAcc[d + 7u]) * invSum));
    }
  }
}

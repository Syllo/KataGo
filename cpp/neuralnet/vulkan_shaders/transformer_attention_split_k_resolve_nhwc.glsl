// Resolve pass for split-K coopmat1 attention (transformer_attention_coopmat1_nhwc.glsl
// with KV_CHUNK_COUNT > 1). One workgroup per (pos, batchHead); one lane per
// half2 pair of V_HEAD_DIM. Combines the KV_CHUNK_COUNT partial
// (un-normalized O, running-max origin, running-sum) triples written by the
// split-K attention kernel with the standard FlashAttention online-softmax
// merge, then writes the final normalized fp16 row.
//
// V_HEAD_DIM is a plain spec constant used directly to size the shared
// array below -- the same pattern as kvMask[BLOCK_KV] in
// transformer_attention_coopmat1_nhwc.glsl, which is known to be safe on the
// T4 SPIR-V translator. (The crash that pattern avoids is a *derived*
// file-scope const -- e.g. a bool folding several spec constants together --
// used for shared-array sizing; a single spec constant used as-is is fine.)

#version 450

#extension GL_EXT_control_flow_attributes : enable

layout(constant_id = 0) const uint NUM_PAIRS = 16;  // V_HEAD_DIM / 2, also the workgroup size
layout(constant_id = 1) const uint V_HEAD_DIM = 32;
layout(local_size_x_id = 0, local_size_y = 1, local_size_z = 1) in;

layout(push_constant) uniform PushConstants {
  int numHeads;
  int seqLenSpec;
  int kvChunkCount;
}
pc;

layout(binding = 0) readonly buffer PartialsBuf {
  float data[];
}
partialsBuf;
layout(binding = 1) readonly buffer StatsBuf {
  float data[];
}
statsBuf;
layout(binding = 2) writeonly buffer OutBuf {
  uint data[];
}
outBuf;

// -FLT_MAX/2 rather than -inf, so max(origin_c) - origin_c cannot produce NaN
// when a chunk was entirely masked out.
const float NEG_BIG = -1.7014118e38;

// The row-wide softmax merge is computed once and shared with the lanes that
// independently merge their own output pair.
shared vec2 mergedStats;  // (rowOrigin, invRowSum)

void main() {
  const uint pos = gl_WorkGroupID.x;
  const uint bh = gl_WorkGroupID.y;
  if(pos >= uint(pc.seqLenSpec))
    return;
  const uint kvChunkCount = uint(pc.kvChunkCount);
  const uint statsRowBase = (bh * kvChunkCount) * uint(pc.seqLenSpec) + pos;
  const uint partialsRowBase = (bh * kvChunkCount) * uint(pc.seqLenSpec) * V_HEAD_DIM + pos * V_HEAD_DIM;
  const uint chunkStatsStride = uint(pc.seqLenSpec);
  const uint chunkPartialsStride = uint(pc.seqLenSpec) * V_HEAD_DIM;

  if(gl_LocalInvocationID.x == 0u) {
    float rowOrigin = NEG_BIG;
    for(uint c = 0u; c < kvChunkCount; c++)
      rowOrigin = max(rowOrigin, statsBuf.data[(statsRowBase + c * chunkStatsStride) * 2u + 0u]);
    float rowSum = 0.0;
    for(uint c = 0u; c < kvChunkCount; c++) {
      const float origin_c = statsBuf.data[(statsRowBase + c * chunkStatsStride) * 2u + 0u];
      const float sum_c = statsBuf.data[(statsRowBase + c * chunkStatsStride) * 2u + 1u];
      rowSum += sum_c * exp2(origin_c - rowOrigin);
    }
    mergedStats = vec2(rowOrigin, rowSum > 0.0 ? 1.0 / rowSum : 0.0);
  }
  barrier();
  const float rowOrigin = mergedStats.x;
  const float invSum = mergedStats.y;

  // Each lane merges and directly writes its own half2 pair. V_HEAD_DIM is
  // even, and the workgroup has V_HEAD_DIM/2 lanes.
  const uint d0 = gl_LocalInvocationID.x * 2u;
  vec2 o = vec2(0.0);
  for(uint c = 0u; c < kvChunkCount; c++) {
    const float origin_c = statsBuf.data[(statsRowBase + c * chunkStatsStride) * 2u + 0u];
    const float scale_c = exp2(origin_c - rowOrigin);
    o.x += partialsBuf.data[partialsRowBase + c * chunkPartialsStride + d0] * scale_c;
    o.y += partialsBuf.data[partialsRowBase + c * chunkPartialsStride + d0 + 1u] * scale_c;
  }

  const uint n = bh / uint(pc.numHeads);
  const uint h = bh - n * uint(pc.numHeads);
  const uint rowWordBase =
    (n * uint(pc.seqLenSpec) + pos) * (uint(pc.numHeads) * V_HEAD_DIM / 2u) + h * (V_HEAD_DIM / 2u);
  outBuf.data[rowWordBase + gl_LocalInvocationID.x] = packHalf2x16(o * invSum);
}

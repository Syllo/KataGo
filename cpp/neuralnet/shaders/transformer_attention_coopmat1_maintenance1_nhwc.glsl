











































#version 450

#extension GL_EXT_control_flow_attributes : enable
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_KHR_memory_scope_semantics : require
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_KHR_cooperative_matrix : require

#extension GL_EXT_cooperative_matrix_maintenance1 : require







layout(constant_id = 0) const uint BLOCK_SIZE = 128;
layout(constant_id = 1) const uint BLOCK_Q = 128;
layout(constant_id = 2) const uint BLOCK_KV = 32;
layout(constant_id = 3) const uint TM = 16;
layout(constant_id = 4) const uint TN = 16;
layout(constant_id = 5) const uint TK = 16;
layout(constant_id = 6) const uint WARP = 32;
layout(constant_id = 7) const uint HEAD_DIM = 32;
layout(constant_id = 8) const uint V_HEAD_DIM = 32;
layout(constant_id = 9) const uint APPLY_ROPE = 0;
layout(constant_id = 10) const uint LEARNABLE_ROPE = 0;
layout(constant_id = 11) const uint SEQ_LEN_SPEC = 1;


layout(constant_id = 12) const uint DIRECT_KV = 0;
layout(local_size_x_id = 0, local_size_y = 1, local_size_z = 1) in;

layout(push_constant) uniform PushConstants {
  int numHeads;
  int numKVHeads;
  float scale;
  int numBH;
} pc;




layout(binding = 0) readonly buffer QBuf { uvec4 data[]; } qBuf;
layout(binding = 1) readonly buffer KBuf { uvec4 data[]; } kBuf;
layout(binding = 2) readonly buffer VBuf { uvec4 data[]; } vBuf;
layout(binding = 3) writeonly buffer OutBuf { uvec4 data[]; } outBuf;
layout(binding = 4) readonly buffer MaskBuf { float16_t data[]; } maskBuf;
layout(binding = 5) readonly buffer CosBuf { float data[]; } cosBuf;
layout(binding = 6) readonly buffer SinBuf { float data[]; } sinBuf;

const uint HEAD_PAD = ( (HEAD_DIM + TK - 1u) / TK) * TK;
const uint V_HEAD_PAD = ( (V_HEAD_DIM + TN - 1u) / TN) * TN;



const uint Q_COLS_V8 = HEAD_PAD / 8u;
// P never lands in shared memory here: probFrag converts straight to the P*V A
// operand, so aPacked stages Q alone and needs no BLOCK_KV-wide P tile.
const uint A_COLS_V8 = Q_COLS_V8;
const uint A_STRIDE_V8 = A_COLS_V8 + 1u;


const uint K_COLS_V8 = HEAD_PAD / 8u;
const uint V_COLS_V8 = V_HEAD_PAD / 8u;



const uint KV_COLS_V8 = DIRECT_KV != 0u
  ? K_COLS_V8
  : (K_COLS_V8 > V_COLS_V8 ? K_COLS_V8 : V_COLS_V8);
const uint KV_STRIDE_V8 = KV_COLS_V8 + 1u;


const uint KV_TILE_ROWS = DIRECT_KV != 0u && APPLY_ROPE == 0u ? 1u : BLOCK_KV;






// The maintenance1 reductions operate directly on cooperative-matrix rows, so
// scratch is only the final V/output spill, not a BLOCK_KV-wide score tile.
const uint SC_STRIDE = V_HEAD_PAD + 1u;

const uint NUM_WARPS = BLOCK_SIZE / WARP;
const uint NUM_Q_STRIPS = BLOCK_Q / TM;
const uint STRIPS_PER_WARP = NUM_Q_STRIPS / NUM_WARPS;
const uint V_FRAGS = V_HEAD_PAD / TN;
const uint K_FRAGS = HEAD_PAD / TK;

const float LOG2_E = 1.4426950408889634;
const float SOFTMAX_RESCALE_THRESHOLD = 8.0;

const float NEG_BIG = - 1.7014118e38;


float maintenanceRowMax(const in float a, const in float b) { return max(a, b); }
float maintenanceRowSum(const in float a, const in float b) { return a + b; }






shared uvec4 aPacked[BLOCK_Q * A_STRIDE_V8];
shared uvec4 kvTile[KV_TILE_ROWS * KV_STRIDE_V8];
shared float scratch[BLOCK_Q * SC_STRIDE];
shared float kvMask[BLOCK_KV];




shared float maintenanceRowReduce[BLOCK_Q];
shared float maintenanceRowOrigin[BLOCK_Q];
shared float maintenanceRowActive[BLOCK_Q];






uint maintenanceRowBase;
uint maintenanceColBase;

float maintenanceMaskScore(const in uint row, const in uint col, const in float value) {
  const uint globalRow = maintenanceRowBase + row;
  const uint globalCol = maintenanceColBase + col;
  return maintenanceRowActive[globalRow] != 0.0 && kvMask[globalCol] != 0.0 && value == value
    ? clamp(value, NEG_BIG, - NEG_BIG) : NEG_BIG;
}
float maintenanceExpScore(const in uint row, const in uint col, const in float value) {
  const uint globalRow = maintenanceRowBase + row;
  const uint globalCol = maintenanceColBase + col;
  return maintenanceRowActive[globalRow] != 0.0 && kvMask[globalCol] != 0.0
    ? exp2(value - maintenanceRowOrigin[globalRow]) : 0.0;
}


uint qIndex8(uint n, uint h, uint pos) {
  return( (n * SEQ_LEN_SPEC + pos) * (uint(pc.numHeads) * HEAD_DIM) + h * HEAD_DIM) / 8u;
}
uint kIndex8(uint n, uint h, uint pos) {
  return( (n * SEQ_LEN_SPEC + pos) * (uint(pc.numKVHeads) * HEAD_DIM) + h * HEAD_DIM) / 8u;
}
uint vIndex8(uint n, uint h, uint pos) {
  return( (n * SEQ_LEN_SPEC + pos) * (uint(pc.numKVHeads) * V_HEAD_DIM) + h * V_HEAD_DIM) / 8u;
}
uint outIndex8(uint n, uint h, uint pos) {
  return( (n * SEQ_LEN_SPEC + pos) * (uint(pc.numHeads) * V_HEAD_DIM) + h * V_HEAD_DIM) / 8u;
}
uint ropeTableIdx(uint kvh, uint pairIdx, uint pos) {
  return LEARNABLE_ROPE != 0u
    ? (kvh * (HEAD_DIM / 2u) + pairIdx) * SEQ_LEN_SPEC + pos
    : pairIdx * SEQ_LEN_SPEC + pos;
}



uint ropeScalePair(uint packed, uint kvh, uint pairIdx, uint pos, float s) {
  vec2 v = unpackHalf2x16(packed);
  if (APPLY_ROPE != 0u) {
    const uint ri = ropeTableIdx(kvh, pairIdx, pos);
    const float cv = cosBuf.data[ri];
    const float sv = sinBuf.data[ri];
    v = vec2(v.x * cv - v.y * sv, v.x * sv + v.y * cv);
  }
  return packHalf2x16(v * s);
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















  const uint warpIdx = gl_SubgroupID;
  const uint laneID = gl_SubgroupInvocationID;
  const uint tid = warpIdx * WARP + laneID;
  const uint qStart = gl_WorkGroupID.x * BLOCK_Q;
  const uint bh = gl_WorkGroupID.y;
  if (bh >= uint(pc.numBH))
    return;
  const uint n = bh / uint(pc.numHeads);
  const uint h = bh - n * uint(pc.numHeads);
  const uint headsPerKv = uint(pc.numHeads / pc.numKVHeads);
  const uint kvh = h / headsPerKv;

  const uint myPos = qStart + tid;
  const float myQMask = (myPos < SEQ_LEN_SPEC)
    ? float(maskBuf.data[n * SEQ_LEN_SPEC + myPos]) : 0.0;
  const bool qActive = myPos < SEQ_LEN_SPEC && myQMask != 0.0;
  float rowSum = 0.0;
  float origin = NEG_BIG;

  maintenanceRowActive[tid] = qActive ? 1.0 : 0.0;






  const float qScale = pc.scale * LOG2_E;
  const uint qBase8 = qActive ? qIndex8(n, h, myPos) : 0u;
  [[unroll]] for (uint d8 = 0u; d8 < Q_COLS_V8; d8 ++) {
    uvec4 packed = uvec4(0u);
    if (qActive && d8 * 8u < HEAD_DIM)
      packed = ropeScaleV8(qBuf.data[qBase8 + d8], kvh, d8, myPos, qScale);
    aPacked[tid * A_STRIDE_V8 + d8] = packed;
  }



  subgroupBarrier();


  coopmat < float16_t, gl_ScopeSubgroup, TM, TK, gl_MatrixUseA > qFrag[STRIPS_PER_WARP * K_FRAGS];
  [[unroll]] for (uint s = 0u; s < STRIPS_PER_WARP; s ++) {
    const uint qBase = (warpIdx * STRIPS_PER_WARP + s) * TM;
    [[unroll]] for (uint k = 0u; k < K_FRAGS; k ++)
      coopMatLoad(qFrag[s * K_FRAGS + k], aPacked,
        qBase * A_STRIDE_V8 + k * TK / 8u, A_STRIDE_V8,
        gl_CooperativeMatrixLayoutRowMajor);
  }


  coopmat < float, gl_ScopeSubgroup, TM, TN, gl_MatrixUseAccumulator > outAcc[STRIPS_PER_WARP * V_FRAGS];
  [[unroll]] for (uint i = 0u; i < STRIPS_PER_WARP * V_FRAGS; i ++)
    outAcc[i] = coopmat < float, gl_ScopeSubgroup, TM, TN, gl_MatrixUseAccumulator > (0.0);







  const uint kRowStrideV8 = uint(pc.numKVHeads) * HEAD_DIM / 8u;
  const uint vRowStrideV8 = uint(pc.numKVHeads) * V_HEAD_DIM / 8u;
  const uint kNBaseV8 = n * SEQ_LEN_SPEC * kRowStrideV8 + kvh * HEAD_DIM / 8u;
  const uint vNBaseV8 = n * SEQ_LEN_SPEC * vRowStrideV8 + kvh * V_HEAD_DIM / 8u;
  const uint maskNBase = n * SEQ_LEN_SPEC;







  for (uint kvStart = 0u; kvStart < SEQ_LEN_SPEC; kvStart += BLOCK_KV) {




    for (uint kv = tid; kv < BLOCK_KV; kv += BLOCK_SIZE) {
      const uint pos = kvStart + kv;
      kvMask[kv] = pos < SEQ_LEN_SPEC ? float(maskBuf.data[maskNBase + pos]) : 0.0;
    }
    barrier();


    bool anyActive = false;
    for (uint kv = 0u; kv < BLOCK_KV; kv ++)
      anyActive = anyActive || kvMask[kv] != 0.0;



    if (anyActive) {


    if (DIRECT_KV == 0u || APPLY_ROPE != 0u) {
      for (uint i = tid; i < BLOCK_KV * K_COLS_V8; i += BLOCK_SIZE) {
        const uint kv = i / K_COLS_V8;
        const uint d8 = i - kv * K_COLS_V8;
        const uint pos = kvStart + kv;
        uvec4 packed = uvec4(0u);
        if (pos < SEQ_LEN_SPEC && kvMask[kv] != 0.0 && d8 * 8u < HEAD_DIM) {
          const uvec4 raw = kBuf.data[kNBaseV8 + pos * kRowStrideV8 + d8];
          packed = APPLY_ROPE != 0u ? ropeScaleV8(raw, kvh, d8, pos, 1.0) : raw;
        }
        kvTile[kv * KV_STRIDE_V8 + d8] = packed;
      }
    }


    if (DIRECT_KV == 0u || APPLY_ROPE != 0u)
      barrier();









    coopmat < float, gl_ScopeSubgroup, TM, TN, gl_MatrixUseAccumulator > scoreFrag[STRIPS_PER_WARP * (BLOCK_KV / TN)];
    coopmat < float, gl_ScopeSubgroup, TM, TN, gl_MatrixUseAccumulator > probFrag[STRIPS_PER_WARP * (BLOCK_KV / TN)];

    [[unroll]] for (uint s = 0u; s < STRIPS_PER_WARP; s ++) {
      const uint qBase = (warpIdx * STRIPS_PER_WARP + s) * TM;
      [[unroll]] for (uint c = 0u; c < BLOCK_KV / TN; c ++) {
        coopmat < float, gl_ScopeSubgroup, TM, TN, gl_MatrixUseAccumulator > sf =
          coopmat < float, gl_ScopeSubgroup, TM, TN, gl_MatrixUseAccumulator > (0.0);
        [[unroll]] for (uint k = 0u; k < K_FRAGS; k ++) {
          coopmat < float16_t, gl_ScopeSubgroup, TK, TN, gl_MatrixUseB > kf;
          if (DIRECT_KV != 0u && APPLY_ROPE == 0u)
            coopMatLoad(kf, kBuf.data,
              kNBaseV8 + (kvStart + c * TN) * kRowStrideV8 + k * TK / 8u,
              kRowStrideV8, gl_CooperativeMatrixLayoutColumnMajor);
          else
            coopMatLoad(kf, kvTile, (c * TN) * KV_STRIDE_V8 + k * TK / 8u, KV_STRIDE_V8,
              gl_CooperativeMatrixLayoutColumnMajor);
          sf = coopMatMulAdd(qFrag[s * K_FRAGS + k], kf, sf);
        }

        maintenanceRowBase = qBase;
        maintenanceColBase = c * TN;
        coopMatPerElementEXT(scoreFrag[s * (BLOCK_KV / TN) + c], sf, maintenanceMaskScore);




      }
    }



    float rowMax = NEG_BIG;
    [[unroll]] for (uint c = 0u; c < BLOCK_KV / TN; c ++) {
      [[unroll]] for (uint s = 0u; s < STRIPS_PER_WARP; s ++) {
        const uint qBase = (warpIdx * STRIPS_PER_WARP + s) * TM;
        coopmat < float, gl_ScopeSubgroup, TM, TN, gl_MatrixUseAccumulator > rowMaxFrag;
        coopMatReduceEXT(rowMaxFrag, scoreFrag[s * (BLOCK_KV / TN) + c],
          gl_CooperativeMatrixReduceRowEXT, maintenanceRowMax);
        [[unroll]] for (uint e = 0u; e < rowMaxFrag.length(); e ++) {
          const uvec2 coord = coopMatGetCoordinateEXT(rowMaxFrag, e);
          if (coord.y == 0u)
            maintenanceRowReduce[qBase + coord.x] = rowMaxFrag[e];
        }
      }
      subgroupBarrier();
      rowMax = max(rowMax, maintenanceRowReduce[tid]);
    }
    const float warpMax = subgroupMax(rowMax);
    if (warpMax > origin + SOFTMAX_RESCALE_THRESHOLD) {
      const float rescale = exp2(origin - warpMax);
      origin = warpMax;
      rowSum *= rescale;
      [[unroll]] for (uint i = 0u; i < STRIPS_PER_WARP * V_FRAGS; i ++)
        outAcc[i] = outAcc[i] * rescale;
    }
    maintenanceRowOrigin[tid] = origin;




    if (DIRECT_KV == 0u)
      barrier();
    else
      subgroupBarrier();



    if (DIRECT_KV == 0u) {
      for (uint i = tid; i < BLOCK_KV * V_COLS_V8; i += BLOCK_SIZE) {
        const uint kv = i / V_COLS_V8;
        const uint d8 = i - kv * V_COLS_V8;
        const uint pos = kvStart + kv;
        uvec4 packed = uvec4(0u);
        if (pos < SEQ_LEN_SPEC && kvMask[kv] != 0.0 && d8 * 8u < V_HEAD_DIM)
          packed = vBuf.data[vNBaseV8 + pos * vRowStrideV8 + d8];
        kvTile[kv * KV_STRIDE_V8 + d8] = packed;
      }
    }





































    [[unroll]] for (uint c = 0u; c < BLOCK_KV / TN; c ++) {
      [[unroll]] for (uint s = 0u; s < STRIPS_PER_WARP; s ++) {
        const uint qBase = (warpIdx * STRIPS_PER_WARP + s) * TM;
        maintenanceRowBase = qBase;
        maintenanceColBase = c * TN;
        coopMatPerElementEXT(probFrag[s * (BLOCK_KV / TN) + c],
          scoreFrag[s * (BLOCK_KV / TN) + c], maintenanceExpScore);
        coopmat < float, gl_ScopeSubgroup, TM, TN, gl_MatrixUseAccumulator > rowSumFrag;
        coopMatReduceEXT(rowSumFrag, probFrag[s * (BLOCK_KV / TN) + c],
          gl_CooperativeMatrixReduceRowEXT, maintenanceRowSum);
        [[unroll]] for (uint e = 0u; e < rowSumFrag.length(); e ++) {
          const uvec2 coord = coopMatGetCoordinateEXT(rowSumFrag, e);
          if (coord.y == 0u)
            maintenanceRowReduce[qBase + coord.x] = rowSumFrag[e];
        }
      }
      subgroupBarrier();
      rowSum += maintenanceRowReduce[tid];
    }






















    if (DIRECT_KV == 0u)
      barrier();
    else
      subgroupBarrier();



    [[unroll]] for (uint s = 0u; s < STRIPS_PER_WARP; s ++) {
      const uint qBase = (warpIdx * STRIPS_PER_WARP + s) * TM;
      [[unroll]] for (uint k = 0u; k < BLOCK_KV / TK; k ++) {

        coopmat < float16_t, gl_ScopeSubgroup, TM, TK, gl_MatrixUseA > pf =
          coopmat < float16_t, gl_ScopeSubgroup, TM, TK, gl_MatrixUseA > (
            probFrag[s * (BLOCK_KV / TN) + k]);





        [[unroll]] for (uint j = 0u; j < V_FRAGS; j ++) {
          coopmat < float16_t, gl_ScopeSubgroup, TK, TN, gl_MatrixUseB > vf;
          if (DIRECT_KV != 0u)
            coopMatLoad(vf, vBuf.data,
              vNBaseV8 + (kvStart + k * TK) * vRowStrideV8 + j * TN / 8u,
              vRowStrideV8, gl_CooperativeMatrixLayoutRowMajor);
          else
            coopMatLoad(vf, kvTile, (k * TK) * KV_STRIDE_V8 + j * TN / 8u, KV_STRIDE_V8,
              gl_CooperativeMatrixLayoutRowMajor);
          outAcc[s * V_FRAGS + j] = coopMatMulAdd(pf, vf, outAcc[s * V_FRAGS + j]);
        }
      }
    }
    }



    barrier();
  }






  [[unroll]] for (uint s = 0u; s < STRIPS_PER_WARP; s ++) {
    const uint qBase = (warpIdx * STRIPS_PER_WARP + s) * TM;
    [[unroll]] for (uint j = 0u; j < V_FRAGS; j ++) {
      coopMatStore(outAcc[s * V_FRAGS + j], scratch,
        qBase * SC_STRIDE + j * TN, SC_STRIDE,
        gl_CooperativeMatrixLayoutRowMajor);
    }
  }


  subgroupBarrier();

  if (myPos < SEQ_LEN_SPEC) {
    const float invSum = rowSum > 0.0 ? 1.0 / rowSum : 0.0;
    const uint outBase8 = outIndex8(n, h, myPos);
    const uint row = tid * SC_STRIDE;
    [[unroll]] for (uint d8 = 0u; d8 < V_COLS_V8; d8 ++) {
      if (d8 * 8u >= V_HEAD_DIM)
        break;
      const uint d = d8 * 8u;
      outBuf.data[outBase8 + d8] = uvec4(
        packHalf2x16(vec2(scratch[row + d + 0u], scratch[row + d + 1u]) * invSum),
        packHalf2x16(vec2(scratch[row + d + 2u], scratch[row + d + 3u]) * invSum),
        packHalf2x16(vec2(scratch[row + d + 4u], scratch[row + d + 5u]) * invSum),
        packHalf2x16(vec2(scratch[row + d + 6u], scratch[row + d + 7u]) * invSum));
    }
  }
}

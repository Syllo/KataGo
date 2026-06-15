// Portable KHR cooperative-matrix transformer attention for NHWC tensors.
//
// Single-pass FlashAttention-2-style kernel with a persistent register-resident
// fp32 output accumulator, in the style of FlashAttention-4 (arXiv 2603.05451):
// one lane per Q row, so softmax state is lane-private, and the O rescale is
// skipped unless the running max grows by more than SOFTMAX_RESCALE_THRESHOLD.
//
// Design invariants:
//   BLOCK_Q == BLOCK_SIZE      -- one lane per Q row.
//   BLOCK_Q / NUM_WARPS == WARP -- so a warp's coopmat fragments cover exactly
//                                  the Q rows owned by that warp's lanes, which
//                                  is what makes the warp-uniform softmax origin
//                                  consistent with the fragment rescale.
//   HEAD_DIM % 8 == 0, V_HEAD_DIM % 8 == 0, BLOCK_KV % 8 == 0
//                                  -- so every global/shared access is uvec4
//                                     (half8) wide.
//
// Shared memory plan (all operands are uvec4-packed half8, the layout the T4
// cooperative-matrix load path wants):
//   aPacked  A-operand staging. Holds Q during the prologue, then P inside the
//            KV loop -- Q is dead once qFrag is in registers, so they share.
//   kvTile   B operand staging, first for QK^T's K (ColumnMajor), then reused
//            for P*V's V (RowMajor). K is dead once QK^T has finished.
//   scratch  fp32, dual use: the score tile inside the KV loop, and the
//            accumulator spill in the epilogue. The two never overlap in time.
//
// Synchronization: aPacked and scratch are declared shared but are in fact
// subgroup-local, which is what the BLOCK_Q / NUM_WARPS == WARP invariant buys.
// STRIPS_PER_WARP * TM == WARP, so subgroup w's coopmat fragments cover rows
// [w*WARP, (w+1)*WARP) -- exactly the rows owned by its own lanes, because tid
// is *constructed* as gl_SubgroupID*WARP + gl_SubgroupInvocationID rather than
// taken from gl_LocalInvocationID (the spec guarantees no relationship between
// those two; see the note in main()). Handoffs through those two arrays are
// therefore intra-subgroup and need only subgroupBarrier(), which the KHR
// subgroup extension defines as both an execution and a memory-ordering barrier
// within the subgroup. kvMask, and the staged K/V operands when present, are
// genuinely cross-subgroup and need a full barrier().
//
// Per active KV iteration the staged path has 5 barrier() calls: post-kvMask,
// post-K staging, K-to-V reuse after QK^T, V/P handoff, and end-of-body.
// Direct no-RoPE has 2 barrier() calls because it has no K/V handoff, and both
// score/P handoffs remain subgroup-local. A fully masked iteration takes 2
// barrier(). The prologue and epilogue use one subgroupBarrier() each.

#version 450

#extension GL_EXT_control_flow_attributes : enable
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_KHR_memory_scope_semantics : require
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_KHR_cooperative_matrix : require

#ifdef KATAGO_VULKAN_RTE_F16
#extension GL_EXT_spirv_intrinsics : require
spirv_execution_mode(extensions = ["SPV_KHR_float_controls"], capabilities = [4467], 4462, 16);
#endif

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
// ggml-style fast path: every KV tile is in bounds, so K/V can be loaded
// directly from global packed-half storage rather than staged through shared.
layout(constant_id = 12) const uint DIRECT_KV = 0;
layout(local_size_x_id = 0, local_size_y = 1, local_size_z = 1) in;

layout(push_constant) uniform PushConstants {
  int numHeads;
  int numKVHeads;
  float scale;
  int numBH;
} pc;

// Q/K/V/out are addressed exclusively as uvec4 (half8). The scalar fp16 views
// are gone; every head dim in play is a multiple of 8 (enforced by
// AttentionCoopmat1Nhwc::isConfigSupported).
layout(binding = 0) readonly buffer QBuf { uvec4 data[]; } qBuf;
layout(binding = 1) readonly buffer KBuf { uvec4 data[]; } kBuf;
layout(binding = 2) readonly buffer VBuf { uvec4 data[]; } vBuf;
layout(binding = 3) writeonly buffer OutBuf { uvec4 data[]; } outBuf;
layout(binding = 4) readonly buffer MaskBuf { float16_t data[]; } maskBuf;
layout(binding = 5) readonly buffer CosBuf { float data[]; } cosBuf;
layout(binding = 6) readonly buffer SinBuf { float data[]; } sinBuf;

const uint HEAD_PAD = ((HEAD_DIM + TK - 1u) / TK) * TK;
const uint V_HEAD_PAD = ((V_HEAD_DIM + TN - 1u) / TN) * TN;

// uvec4 column counts for the shared A-operand staging. Q needs HEAD_PAD/8,
// P needs BLOCK_KV/8; the array is sized to the larger since they share it.
const uint Q_COLS_V8 = HEAD_PAD / 8u;
const uint P_COLS_V8 = BLOCK_KV / 8u;
const uint A_COLS_V8 = Q_COLS_V8 > P_COLS_V8 ? Q_COLS_V8 : P_COLS_V8;
const uint A_STRIDE_V8 = A_COLS_V8 + 1u;  // +1 uvec4 for bank padding

// B-operand strides, also in uvec4 units.
const uint K_COLS_V8 = HEAD_PAD / 8u;
const uint V_COLS_V8 = V_HEAD_PAD / 8u;
// The staged path uses one physical tile for K then V. Direct V loads only
// need K's width for its still-staged RoPE K operand; direct no-RoPE needs no
// staging and leaves one minimal row.
const uint KV_COLS_V8 = DIRECT_KV != 0u
  ? K_COLS_V8
  : (K_COLS_V8 > V_COLS_V8 ? K_COLS_V8 : V_COLS_V8);
const uint KV_STRIDE_V8 = KV_COLS_V8 + 1u;
// Direct KV still stages every K row when RoPE is fused here. Only direct
// no-RoPE can use the one-row placeholder.
const uint KV_TILE_ROWS = DIRECT_KV != 0u && APPLY_ROPE == 0u ? 1u : BLOCK_KV;

// fp32 scratch row stride, in floats. Must cover the score tile (BLOCK_KV
// columns) and the epilogue accumulator spill (V_HEAD_PAD columns). The odd
// +1 makes `lane tid reads row tid` bank-conflict free: (tid*stride + c) % 32
// == (tid + c) % 32.
const uint SC_COLS = BLOCK_KV > V_HEAD_PAD ? BLOCK_KV : V_HEAD_PAD;
const uint SC_STRIDE = SC_COLS + 1u;

const uint NUM_WARPS = BLOCK_SIZE / WARP;
const uint NUM_Q_STRIPS = BLOCK_Q / TM;
const uint STRIPS_PER_WARP = NUM_Q_STRIPS / NUM_WARPS;
const uint V_FRAGS = V_HEAD_PAD / TN;
const uint K_FRAGS = HEAD_PAD / TK;

const float LOG2_E = 1.4426950408889634;
const float SOFTMAX_RESCALE_THRESHOLD = 8.0;
// -FLT_MAX/2 rather than -inf, so that oldOrigin - newOrigin cannot produce NaN.
const float NEG_BIG = -1.7014118e38;

// A operands (Q, then P) share one buffer; K then V share kvTile; and scratch
// is separate. With bq=128/bkv=32/headDim=32 the staged layout is 29824 bytes.
// DIRECT_KV shrinks kvTile to one padding row only when K is also loaded
// globally (the no-RoPE path).
shared uvec4 aPacked[BLOCK_Q * A_STRIDE_V8];
shared uvec4 kvTile[KV_TILE_ROWS * KV_STRIDE_V8];
shared float scratch[BLOCK_Q * SC_STRIDE];
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
  return LEARNABLE_ROPE != 0u
    ? (kvh * (HEAD_DIM / 2u) + pairIdx) * SEQ_LEN_SPEC + pos
    : pairIdx * SEQ_LEN_SPEC + pos;
}

// One packed half2 is exactly one RoPE pair (d0, d0+1), so a uvec4 covers four
// consecutive pairs. Rotate and scale in fp32, then repack.
uint ropeScalePair(uint packed, uint kvh, uint pairIdx, uint pos, float s) {
  vec2 v = unpackHalf2x16(packed);
  if(APPLY_ROPE != 0u) {
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
  // Row ownership is built from the subgroup builtins, NOT from
  // gl_LocalInvocationID. The Vulkan spec is explicit that "there is no direct
  // relationship between SubgroupLocalInvocationId and LocalInvocationId or
  // LocalInvocationIndex", and that with full subgroups an application should
  // construct its own dense index as
  //     index = SubgroupLocalInvocationId + SubgroupId * SubgroupSize
  // which is what tid is below. The host requests a fixed subgroup size and
  // requireFullSubgroups (see AttentionCoopmat1Nhwc::build), so every subgroup
  // is full and this index is dense over [0, BLOCK_SIZE).
  //
  // This is what makes subgroup w own exactly rows [w*WARP, (w+1)*WARP): with
  // STRIPS_PER_WARP * TM == WARP its coopmat fragments cover that same range,
  // so aPacked and scratch are genuinely subgroup-local and the handoffs through
  // them need only subgroupBarrier(). Using gl_LocalInvocationID.x here would
  // make that ownership argument depend on an unspecified driver mapping.
  const uint warpIdx = gl_SubgroupID;
  const uint laneID = gl_SubgroupInvocationID;
  const uint tid = warpIdx * WARP + laneID;
  const uint qStart = gl_WorkGroupID.x * BLOCK_Q;
  const uint bh = gl_WorkGroupID.y;
  if(bh >= uint(pc.numBH))
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

  // ---- Prologue: lane tid owns Q row tid, so it loads, rotates, scales and
  // packs its own row with no shared staging and no barriers. Folding
  // scale*log2(e) into Q here removes a per-element multiply over the whole
  // score tile on every KV block.
  const float qScale = pc.scale * LOG2_E;
  const uint qBase8 = qActive ? qIndex8(n, h, myPos) : 0u;
  [[unroll]] for(uint d8 = 0u; d8 < Q_COLS_V8; d8++) {
    uvec4 packed = uvec4(0u);
    if(qActive && d8 * 8u < HEAD_DIM)
      packed = ropeScaleV8(qBuf.data[qBase8 + d8], kvh, d8, myPos, qScale);
    aPacked[tid * A_STRIDE_V8 + d8] = packed;
  }
  // Lane tid wrote row tid; subgroup w reads back only rows
  // [w*WARP, (w+1)*WARP), i.e. exactly its own lanes' rows. Intra-subgroup
  // handoff, so no barrier().
  subgroupBarrier();

  // Persistent Q A-fragments. Once these are in registers aPacked is free for P.
  coopmat<float16_t, gl_ScopeSubgroup, TM, TK, gl_MatrixUseA> qFrag[STRIPS_PER_WARP * K_FRAGS];
  [[unroll]] for(uint s = 0u; s < STRIPS_PER_WARP; s++) {
    const uint qBase = (warpIdx * STRIPS_PER_WARP + s) * TM;
    [[unroll]] for(uint k = 0u; k < K_FRAGS; k++)
      coopMatLoad(qFrag[s * K_FRAGS + k], aPacked,
        qBase * A_STRIDE_V8 + k * TK / 8u, A_STRIDE_V8,
        gl_CooperativeMatrixLayoutRowMajor);
  }

  // Persistent output accumulator in registers.
  coopmat<float, gl_ScopeSubgroup, TM, TN, gl_MatrixUseAccumulator> outAcc[STRIPS_PER_WARP * V_FRAGS];
  [[unroll]] for(uint i = 0u; i < STRIPS_PER_WARP * V_FRAGS; i++)
    outAcc[i] = coopmat<float, gl_ScopeSubgroup, TM, TN, gl_MatrixUseAccumulator>(0.0);

  // Hoist push-constant math out of the KV loop. These are all loop-invariant
  // (n, kvh fixed per workgroup; pc.numKVHeads/HEAD_DIM/V_HEAD_DIM constant),
  // so evaluating them once here removes the corresponding IMAD/UDIV chain
  // from every KV iteration's index computation. The K/V row strides also
  // feed the direct coopMatLoad path (previously spelled inline at each call
  // site).
  const uint kRowStrideV8 = uint(pc.numKVHeads) * HEAD_DIM / 8u;
  const uint vRowStrideV8 = uint(pc.numKVHeads) * V_HEAD_DIM / 8u;
  const uint kNBaseV8 = n * SEQ_LEN_SPEC * kRowStrideV8 + kvh * HEAD_DIM / 8u;
  const uint vNBaseV8 = n * SEQ_LEN_SPEC * vRowStrideV8 + kvh * V_HEAD_DIM / 8u;
  const uint maskNBase = n * SEQ_LEN_SPEC;

  // =========================================================================
  // KV loop. Three workgroup barriers on the active path: after the kvMask
  // write, after the K/V staging writes, and at the end of the body to protect
  // kTile/vTile/kvMask from the next iteration. A fully masked iteration takes
  // only the first and last. Everything else is intra-subgroup (header note).
  // =========================================================================
  for(uint kvStart = 0u; kvStart < SEQ_LEN_SPEC; kvStart += BLOCK_KV) {
    // Strided, not a single `if(tid < BLOCK_KV)` write: the tuner proposes
    // BLOCK_KV up to 128, which can exceed BLOCK_SIZE on a device whose shared
    // memory admits such a tile, and a single conditional write would leave the
    // tail of kvMask unwritten.
    for(uint kv = tid; kv < BLOCK_KV; kv += BLOCK_SIZE) {
      const uint pos = kvStart + kv;
      kvMask[kv] = pos < SEQ_LEN_SPEC ? float(maskBuf.data[maskNBase + pos]) : 0.0;
    }
    barrier();
    // Every thread reads all of kvMask, so anyActive is workgroup-uniform.
    // Broadcast reads, no bank conflict.
    bool anyActive = false;
    for(uint kv = 0u; kv < BLOCK_KV; kv++)
      anyActive = anyActive || kvMask[kv] != 0.0;

    // Guarding the body rather than `continue`-ing lets the single end-of-body
    // barrier cover the skip path too, instead of needing its own.
    if(anyActive) {
    // ---- Stage K only when it cannot be read directly. The staged tile is
    // reused for V after QK^T, once every subgroup is done reading K.
    if(DIRECT_KV == 0u || APPLY_ROPE != 0u) {
      for(uint i = tid; i < BLOCK_KV * K_COLS_V8; i += BLOCK_SIZE) {
        const uint kv = i / K_COLS_V8;
        const uint d8 = i - kv * K_COLS_V8;
        const uint pos = kvStart + kv;
        uvec4 packed = uvec4(0u);
        if(pos < SEQ_LEN_SPEC && kvMask[kv] != 0.0 && d8 * 8u < HEAD_DIM) {
          const uvec4 raw = kBuf.data[kNBaseV8 + pos * kRowStrideV8 + d8];
          packed = APPLY_ROPE != 0u ? ropeScaleV8(raw, kvh, d8, pos, 1.0) : raw;
        }
        kvTile[kv * KV_STRIDE_V8 + d8] = packed;
      }
    }
    // Only staged K requires a cross-warp handoff. In the direct no-RoPE path
    // this removes the staging barrier entirely.
    if(DIRECT_KV == 0u || APPLY_ROPE != 0u)
      barrier();

    // ---- QK^T -> fp32 scores in scratch. Q is A (RowMajor: the reduction
    // index d is contiguous). K is B (ColumnMajor: element (d, kv) sits at
    // kv*KV_STRIDE_V8 + d, so again d is contiguous, which is what allows a
    // packed uvec4 array to back a B operand). Q already carries the scale.
    [[unroll]] for(uint s = 0u; s < STRIPS_PER_WARP; s++) {
      const uint qBase = (warpIdx * STRIPS_PER_WARP + s) * TM;
      [[unroll]] for(uint c = 0u; c < BLOCK_KV / TN; c++) {
        coopmat<float, gl_ScopeSubgroup, TM, TN, gl_MatrixUseAccumulator> sf =
          coopmat<float, gl_ScopeSubgroup, TM, TN, gl_MatrixUseAccumulator>(0.0);
        [[unroll]] for(uint k = 0u; k < K_FRAGS; k++) {
          coopmat<float16_t, gl_ScopeSubgroup, TK, TN, gl_MatrixUseB> kf;
          if(DIRECT_KV != 0u && APPLY_ROPE == 0u)
            coopMatLoad(kf, kBuf.data,
              kNBaseV8 + (kvStart + c * TN) * kRowStrideV8 + k * TK / 8u,
              kRowStrideV8, gl_CooperativeMatrixLayoutColumnMajor);
          else
            coopMatLoad(kf, kvTile, (c * TN) * KV_STRIDE_V8 + k * TK / 8u, KV_STRIDE_V8,
              gl_CooperativeMatrixLayoutColumnMajor);
          sf = coopMatMulAdd(qFrag[s * K_FRAGS + k], kf, sf);
        }
        coopMatStore(sf, scratch, qBase * SC_STRIDE + c * TN, SC_STRIDE,
          gl_CooperativeMatrixLayoutRowMajor);
      }
    }
    // K is read by every subgroup. Before staged V overwrites kvTile, make
    // every K read complete. Direct-V paths retain the cheaper subgroup score
    // handoff because they never reuse this storage in the iteration.
    if(DIRECT_KV == 0u)
      barrier();
    else
      subgroupBarrier();

    // ---- V takes over the staged K tile. This occurs before softmax so its
    // global loads can overlap other lanes' lane-private score processing.
    if(DIRECT_KV == 0u) {
      for(uint i = tid; i < BLOCK_KV * V_COLS_V8; i += BLOCK_SIZE) {
        const uint kv = i / V_COLS_V8;
        const uint d8 = i - kv * V_COLS_V8;
        const uint pos = kvStart + kv;
        uvec4 packed = uvec4(0u);
        if(pos < SEQ_LEN_SPEC && kvMask[kv] != 0.0 && d8 * 8u < V_HEAD_DIM)
          packed = vBuf.data[vNBaseV8 + pos * vRowStrideV8 + d8];
        kvTile[kv * KV_STRIDE_V8 + d8] = packed;
      }
    }

    // ---- Online softmax. Lane-private state, warp-uniform origin: the O
    // accumulator is a coopmat fragment, so its rescale cannot be per-row, and
    // BLOCK_Q/NUM_WARPS == WARP means a warp's fragments cover exactly its own
    // lanes' rows. subgroupMax must be reached by every lane, so it sits
    // outside the mask test.
    float rowMax = NEG_BIG;
    if(qActive) {
      for(uint kv = 0u; kv < BLOCK_KV; kv++) {
        const uint pos = kvStart + kv;
        if(pos < SEQ_LEN_SPEC && kvMask[kv] != 0.0)
          rowMax = max(rowMax, scratch[tid * SC_STRIDE + kv]);
      }
    }
    const float warpMax = subgroupMax(rowMax);
    // Conditional rescaling (FA-4 eq. 6): only re-base when the max grows by
    // more than the threshold, and let the final divide by rowSum fix up the
    // rest. The warp rescales if any of its lanes needs it.
    if(warpMax > origin + SOFTMAX_RESCALE_THRESHOLD) {
      const float rescale = exp2(origin - warpMax);
      origin = warpMax;
      rowSum *= rescale;
      [[unroll]] for(uint i = 0u; i < STRIPS_PER_WARP * V_FRAGS; i++)
        outAcc[i] = outAcc[i] * rescale;
    }

    // ---- P = exp2(S - origin), written straight into packed A-operand form.
    // The row is lane-private, so this needs no intermediate shared buffer.
    // Note this overwrites the aPacked rows whose Q values were consumed into
    // qFrag registers back in the prologue -- safe because no other warp ever
    // reads this warp's rows.
    [[unroll]] for(uint kv8 = 0u; kv8 < P_COLS_V8; kv8++) {
      uint w[4];
      [[unroll]] for(uint t = 0u; t < 4u; t++) {
        float p0 = 0.0;
        float p1 = 0.0;
        if(qActive) {
          const uint kv = kv8 * 8u + t * 2u;
          const uint pos = kvStart + kv;
          if(pos < SEQ_LEN_SPEC && kvMask[kv] != 0.0)
            p0 = exp2(scratch[tid * SC_STRIDE + kv] - origin);
          if(pos + 1u < SEQ_LEN_SPEC && kvMask[kv + 1u] != 0.0)
            p1 = exp2(scratch[tid * SC_STRIDE + kv + 1u] - origin);
          rowSum += p0 + p1;
        }
        w[t] = packHalf2x16(vec2(p0, p1));
      }
      aPacked[tid * A_STRIDE_V8 + kv8] = uvec4(w[0], w[1], w[2], w[3]);
    }
    // Staged V is read by every subgroup, so the existing P handoff also
    // makes V visible. Direct V remains an intra-subgroup P handoff only.
    if(DIRECT_KV == 0u)
      barrier();
    else
      subgroupBarrier();

    // ---- P*V into the persistent accumulator. P is A (RowMajor), V is B
    // (RowMajor: element (kv, d) at kv*KV_STRIDE_V8 + d, d contiguous).
    [[unroll]] for(uint s = 0u; s < STRIPS_PER_WARP; s++) {
      const uint qBase = (warpIdx * STRIPS_PER_WARP + s) * TM;
      [[unroll]] for(uint k = 0u; k < BLOCK_KV / TK; k++) {
        coopmat<float16_t, gl_ScopeSubgroup, TM, TK, gl_MatrixUseA> pf;
        coopMatLoad(pf, aPacked, qBase * A_STRIDE_V8 + k * TK / 8u, A_STRIDE_V8,
          gl_CooperativeMatrixLayoutRowMajor);
        [[unroll]] for(uint j = 0u; j < V_FRAGS; j++) {
          coopmat<float16_t, gl_ScopeSubgroup, TK, TN, gl_MatrixUseB> vf;
          if(DIRECT_KV != 0u)
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
    }  // if(anyActive)

    // Cross-warp, and reached on both paths: keeps the next iteration's kvMask
    // and kvTile staging writes from racing this iteration's reads.
    barrier();
  }

  // =========================================================================
  // Epilogue: spill the accumulators over the (now dead) score tile, then each
  // lane normalizes and writes its own row. Every column 0..V_HEAD_PAD-1 and
  // every row is covered by a store, so no zero-fill pass is needed.
  // =========================================================================
  [[unroll]] for(uint s = 0u; s < STRIPS_PER_WARP; s++) {
    const uint qBase = (warpIdx * STRIPS_PER_WARP + s) * TM;
    [[unroll]] for(uint j = 0u; j < V_FRAGS; j++) {
      coopMatStore(outAcc[s * V_FRAGS + j], scratch,
        qBase * SC_STRIDE + j * TN, SC_STRIDE,
        gl_CooperativeMatrixLayoutRowMajor);
    }
  }
  // Subgroup w spilled rows [w*WARP, (w+1)*WARP); lane tid reads back row tid
  // from that same subgroup. Intra-subgroup handoff, so no barrier().
  subgroupBarrier();

  if(myPos < SEQ_LEN_SPEC) {
    const float invSum = rowSum > 0.0 ? 1.0 / rowSum : 0.0;
    const uint outBase8 = outIndex8(n, h, myPos);
    const uint row = tid * SC_STRIDE;
    [[unroll]] for(uint d8 = 0u; d8 < V_COLS_V8; d8++) {
      if(d8 * 8u >= V_HEAD_DIM)
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

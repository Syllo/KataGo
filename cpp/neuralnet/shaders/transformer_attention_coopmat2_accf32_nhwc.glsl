// Workgroup-scope cooperative-matrix2 NHWC attention, FP16 inputs / FP32 accumulators.
#version 460

#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int32 : require
#extension GL_KHR_memory_scope_semantics : require
#extension GL_KHR_cooperative_matrix : require
#extension GL_NV_cooperative_matrix2 : require
#extension GL_EXT_buffer_reference : require

#ifdef KATAGO_VULKAN_RTE_F16
#extension GL_EXT_spirv_intrinsics : require
spirv_execution_mode(extensions = ["SPV_KHR_float_controls"], capabilities = [4467], 4462, 16);
#endif

layout(constant_id = 0) const uint BLOCK_SIZE = 128;
layout(constant_id = 1) const uint BLOCK_Q = 128;
layout(constant_id = 2) const uint BLOCK_KV = 32;
layout(constant_id = 3) const uint HEAD_DIM = 32;
layout(constant_id = 4) const uint V_HEAD_DIM = 32;
layout(constant_id = 5) const uint APPLY_ROPE = 0;
layout(constant_id = 6) const uint LEARNABLE_ROPE = 0;
layout(constant_id = 7) const uint SEQ_LEN_SPEC = 1;
// All Q/K/V/output matrix tiles are in bounds when enabled. This selects
// Undefined clamp mode, avoiding the tensor-address bounds machinery.
layout(constant_id = 8) const uint ALIGNED_TILES = 0;
layout(local_size_x_id = 0, local_size_y = 1, local_size_z = 1) in;

layout(push_constant) uniform PushConstants { int numHeads; int numKVHeads; float scale; int numBH; } pc;
layout(binding = 0) readonly buffer QBuf { float16_t data[]; } qBuf;
layout(binding = 1) readonly buffer KBuf { float16_t data[]; } kBuf;
layout(binding = 2) readonly buffer VBuf { float16_t data[]; } vBuf;
layout(binding = 3) writeonly buffer OutBuf { float16_t data[]; } outBuf;
layout(binding = 4) readonly buffer MaskBuf { float16_t data[]; } maskBuf;
layout(binding = 5) readonly buffer CosBuf { float data[]; } cosBuf;
layout(binding = 6) readonly buffer SinBuf { float data[]; } sinBuf;

layout(buffer_reference, std430, buffer_reference_align = 4) readonly buffer RopePair { f16vec2 pair; };

// Global tensors use canonical row-major layouts: the outer stride equals the
// inner dimension and the inner stride is one. This follows the
// NV_cooperative_matrix2 canonical-stride rule:
// https://github.com/KhronosGroup/GLSL/blob/main/extensions/nv/GLSL_NV_cooperative_matrix2.txt
const float LOG2_E = 1.4426950408889634;
const float NEG_BIG = -1.7014118e38;

// Native per-element cooperative-matrix callbacks execute once per matrix
// element. Keep the binary mask in workgroup memory so those callbacks do not
// repeatedly fetch the same two values from global memory. The mask is not
// assumed to be a contiguous valid prefix: KataGo can supply arbitrary board
// masks.
shared uint qMask[BLOCK_Q];
shared uint kvMask[BLOCK_KV];

float maxReduce(const in float x, const in float y) { return max(x,y); }
float sumReduce(const in float x, const in float y) { return x+y; }
float smearReduce(const in float x, const in float y) { return x; }
float maxPerElement(const in uint32_t row, const in uint32_t col, const in float x, const in float y) { return max(x,y); }
float exp2PerElement(const in uint32_t row, const in uint32_t col, const in float x) { return exp2(x); }
float inversePerElement(const in uint32_t row, const in uint32_t col, const in float x) { return x > 0.0 ? 1.0/x : 0.0; }

float maskScore(
  const in uint32_t row, const in uint32_t col, const in float x,
  const in uint32_t n, const in uint32_t qStart, const in uint32_t kvStart) {
  const uint qPos=qStart+row, kvPos=kvStart+col;
  return qPos<SEQ_LEN_SPEC && kvPos<SEQ_LEN_SPEC &&
    qMask[row]!=0u && kvMask[col]!=0u ? x : NEG_BIG;
}

float maskedExp2(
  const in uint32_t row, const in uint32_t col, const in float x,
  const in uint32_t n, const in uint32_t qStart, const in uint32_t kvStart) {
  const uint qPos=qStart+row, kvPos=kvStart+col;
  return qPos<SEQ_LEN_SPEC && kvPos<SEQ_LEN_SPEC &&
    qMask[row]!=0u && kvMask[col]!=0u ? exp2(x) : 0.0;
}

float16_t ropeLoad(const in RopePair raw, const in uint blockCoords[2], const in uint coordInBlock[2]) {
  const uint bh = gl_WorkGroupID.y;
  const uint h = bh % uint(pc.numHeads);
  const uint kvh = h / uint(pc.numHeads / pc.numKVHeads);
  const uint pairs = HEAD_DIM / 2u;
  const uint pairIdx = blockCoords[1] % pairs;
  const uint tableIdx = LEARNABLE_ROPE != 0u
    ? (kvh * SEQ_LEN_SPEC + blockCoords[0]) * pairs + pairIdx
    : blockCoords[0] * pairs + pairIdx;
  const vec2 x = vec2(raw.pair);
  const float c = cosBuf.data[tableIdx], s = sinBuf.data[tableIdx];
  const vec2 y = vec2(x.x*c-x.y*s, x.x*s+x.y*c);
  return float16_t(coordInBlock[1] == 0u ? y.x : y.y);
}

void main() {
  const uint tid=gl_LocalInvocationID.x, qStart=gl_WorkGroupID.x*BLOCK_Q, bh=gl_WorkGroupID.y;
  if(bh>=uint(pc.numBH)) return;
  const uint n=bh/uint(pc.numHeads), h=bh-n*uint(pc.numHeads), kvh=h/uint(pc.numHeads/pc.numKVHeads);
  // Q rows may outnumber workgroup invocations. Fill their mask in a strided
  // loop so BLOCK_Q can be tuned independently of BLOCK_SIZE.
  for(uint row=tid; row<BLOCK_Q; row+=BLOCK_SIZE) {
    const uint pos=qStart+row;
    qMask[row]=pos<SEQ_LEN_SPEC && float(maskBuf.data[n*SEQ_LEN_SPEC+pos])!=0.0 ? 1u : 0u;
  }
  barrier();
  // V is not RoPE-rotated. Load it directly from its canonical global
  // [position][KV-head][value-dimension] layout; invalid columns have zero
  // probability, and the clamp handles the final partial KV tile.
  tensorLayoutNV<2> vLayout = createTensorLayoutNV(2);
  tensorLayoutNV<2, gl_CooperativeMatrixClampModeConstantNV> vLayoutClamp =
    createTensorLayoutNV(2, gl_CooperativeMatrixClampModeConstantNV);
  vLayout = setTensorLayoutDimensionNV(vLayout, SEQ_LEN_SPEC, uint(pc.numKVHeads) * V_HEAD_DIM);
  vLayout = setTensorLayoutStrideNV(vLayout, uint(pc.numKVHeads) * V_HEAD_DIM, 1u);
  vLayoutClamp = setTensorLayoutDimensionNV(vLayoutClamp, SEQ_LEN_SPEC, uint(pc.numKVHeads) * V_HEAD_DIM);
  vLayoutClamp = setTensorLayoutStrideNV(vLayoutClamp, uint(pc.numKVHeads) * V_HEAD_DIM, 1u);
  vLayoutClamp = setTensorLayoutClampValueNV(vLayoutClamp, 0u);
  tensorLayoutNV<2> outLayout = createTensorLayoutNV(2);
  tensorLayoutNV<2, gl_CooperativeMatrixClampModeConstantNV> outLayoutClamp =
    createTensorLayoutNV(2, gl_CooperativeMatrixClampModeConstantNV);
  outLayout = setTensorLayoutDimensionNV(outLayout, SEQ_LEN_SPEC, uint(pc.numHeads) * V_HEAD_DIM);
  outLayout = setTensorLayoutStrideNV(outLayout, uint(pc.numHeads) * V_HEAD_DIM, 1u);
  outLayoutClamp = setTensorLayoutDimensionNV(outLayoutClamp, SEQ_LEN_SPEC, uint(pc.numHeads) * V_HEAD_DIM);
  outLayoutClamp = setTensorLayoutStrideNV(outLayoutClamp, uint(pc.numHeads) * V_HEAD_DIM, 1u);
  outLayoutClamp = setTensorLayoutClampValueNV(outLayoutClamp, 0u);
  tensorLayoutNV<2> qLayout = createTensorLayoutNV(2);
  tensorLayoutNV<2, gl_CooperativeMatrixClampModeConstantNV> qLayoutClamp =
    createTensorLayoutNV(2, gl_CooperativeMatrixClampModeConstantNV);
  if(APPLY_ROPE != 0u) qLayout = setTensorLayoutBlockSizeNV(qLayout, 1u, 2u);
  qLayout = setTensorLayoutDimensionNV(qLayout, SEQ_LEN_SPEC, uint(pc.numHeads) * HEAD_DIM);
  qLayout = setTensorLayoutStrideNV(qLayout, (APPLY_ROPE != 0u ? uint(pc.numHeads) * HEAD_DIM / 2u : uint(pc.numHeads) * HEAD_DIM), 1u);
  if(APPLY_ROPE != 0u) qLayoutClamp = setTensorLayoutBlockSizeNV(qLayoutClamp, 1u, 2u);
  qLayoutClamp = setTensorLayoutDimensionNV(qLayoutClamp, SEQ_LEN_SPEC, uint(pc.numHeads) * HEAD_DIM);
  qLayoutClamp = setTensorLayoutStrideNV(qLayoutClamp, (APPLY_ROPE != 0u ? uint(pc.numHeads) * HEAD_DIM / 2u : uint(pc.numHeads) * HEAD_DIM), 1u);
  qLayoutClamp = setTensorLayoutClampValueNV(qLayoutClamp, 0u);
  tensorLayoutNV<2> kLayout = createTensorLayoutNV(2);
  tensorLayoutNV<2, gl_CooperativeMatrixClampModeConstantNV> kLayoutClamp =
    createTensorLayoutNV(2, gl_CooperativeMatrixClampModeConstantNV);
  if(APPLY_ROPE != 0u) kLayout = setTensorLayoutBlockSizeNV(kLayout, 1u, 2u);
  kLayout = setTensorLayoutDimensionNV(kLayout, SEQ_LEN_SPEC, uint(pc.numKVHeads) * HEAD_DIM);
  kLayout = setTensorLayoutStrideNV(kLayout, (APPLY_ROPE != 0u ? uint(pc.numKVHeads) * HEAD_DIM / 2u : uint(pc.numKVHeads) * HEAD_DIM), 1u);
  if(APPLY_ROPE != 0u) kLayoutClamp = setTensorLayoutBlockSizeNV(kLayoutClamp, 1u, 2u);
  kLayoutClamp = setTensorLayoutDimensionNV(kLayoutClamp, SEQ_LEN_SPEC, uint(pc.numKVHeads) * HEAD_DIM);
  kLayoutClamp = setTensorLayoutStrideNV(kLayoutClamp, (APPLY_ROPE != 0u ? uint(pc.numKVHeads) * HEAD_DIM / 2u : uint(pc.numKVHeads) * HEAD_DIM), 1u);
  kLayoutClamp = setTensorLayoutClampValueNV(kLayoutClamp, 0u);
  tensorViewNV<2, false, 1, 0> transposeView = createTensorViewNV(2, false, 1, 0);
  coopmat<float16_t,gl_ScopeWorkgroup,BLOCK_Q,HEAD_DIM,gl_MatrixUseA> q;
  const uint qBase = n * SEQ_LEN_SPEC * uint(pc.numHeads) * HEAD_DIM;
  if(ALIGNED_TILES != 0u) {
    if(APPLY_ROPE != 0u) coopMatLoadTensorNV(q, qBuf.data, qBase, sliceTensorLayoutNV(qLayout, qStart, BLOCK_Q, h * HEAD_DIM, HEAD_DIM), ropeLoad);
    else coopMatLoadTensorNV(q, qBuf.data, qBase, sliceTensorLayoutNV(qLayout, qStart, BLOCK_Q, h * HEAD_DIM, HEAD_DIM));
  } else {
    if(APPLY_ROPE != 0u) coopMatLoadTensorNV(q, qBuf.data, qBase, sliceTensorLayoutNV(qLayoutClamp, qStart, BLOCK_Q, h * HEAD_DIM, HEAD_DIM), ropeLoad);
    else coopMatLoadTensorNV(q, qBuf.data, qBase, sliceTensorLayoutNV(qLayoutClamp, qStart, BLOCK_Q, h * HEAD_DIM, HEAD_DIM));
  }
  q = q * float16_t(pc.scale * LOG2_E);
  coopmat<float,gl_ScopeWorkgroup,BLOCK_Q,V_HEAD_DIM,gl_MatrixUseAccumulator> outAcc=
    coopmat<float,gl_ScopeWorkgroup,BLOCK_Q,V_HEAD_DIM,gl_MatrixUseAccumulator>(0.0);
  coopmat<float,gl_ScopeWorkgroup,BLOCK_Q,BLOCK_KV,gl_MatrixUseAccumulator> rowSum=
    coopmat<float,gl_ScopeWorkgroup,BLOCK_Q,BLOCK_KV,gl_MatrixUseAccumulator>(0.0);
  coopmat<float,gl_ScopeWorkgroup,BLOCK_Q,BLOCK_KV,gl_MatrixUseAccumulator> origin=
    coopmat<float,gl_ScopeWorkgroup,BLOCK_Q,BLOCK_KV,gl_MatrixUseAccumulator>(NEG_BIG);
  for(uint kvStart=0u;kvStart<SEQ_LEN_SPEC;kvStart+=BLOCK_KV) {
    for(uint col=tid; col<BLOCK_KV; col+=BLOCK_SIZE) {
      const uint pos=kvStart+col;
      kvMask[col]=pos<SEQ_LEN_SPEC && float(maskBuf.data[n*SEQ_LEN_SPEC+pos])!=0.0 ? 1u : 0u;
    }
    barrier();
    coopmat<float16_t,gl_ScopeWorkgroup,HEAD_DIM,BLOCK_KV,gl_MatrixUseB> k;
    coopmat<float,gl_ScopeWorkgroup,BLOCK_Q,BLOCK_KV,gl_MatrixUseAccumulator> scores=coopmat<float,gl_ScopeWorkgroup,BLOCK_Q,BLOCK_KV,gl_MatrixUseAccumulator>(0.0);
    const uint kBase = n * SEQ_LEN_SPEC * uint(pc.numKVHeads) * HEAD_DIM;
    if(ALIGNED_TILES != 0u) {
      if(APPLY_ROPE != 0u) coopMatLoadTensorNV(k, kBuf.data, kBase, sliceTensorLayoutNV(kLayout, kvStart, BLOCK_KV, kvh * HEAD_DIM, HEAD_DIM), transposeView, ropeLoad);
      else coopMatLoadTensorNV(k, kBuf.data, kBase, sliceTensorLayoutNV(kLayout, kvStart, BLOCK_KV, kvh * HEAD_DIM, HEAD_DIM), transposeView);
    } else {
      if(APPLY_ROPE != 0u) coopMatLoadTensorNV(k, kBuf.data, kBase, sliceTensorLayoutNV(kLayoutClamp, kvStart, BLOCK_KV, kvh * HEAD_DIM, HEAD_DIM), transposeView, ropeLoad);
      else coopMatLoadTensorNV(k, kBuf.data, kBase, sliceTensorLayoutNV(kLayoutClamp, kvStart, BLOCK_KV, kvh * HEAD_DIM, HEAD_DIM), transposeView);
    }
    scores=coopMatMulAdd(q,k,scores);
    coopMatPerElementNV(scores,scores,maskScore,n,qStart,kvStart);

    coopmat<float,gl_ScopeWorkgroup,BLOCK_Q,BLOCK_KV,gl_MatrixUseAccumulator> tileMax;
    coopMatReduceNV(tileMax,scores,gl_CooperativeMatrixReduceRowNV,maxReduce);
    coopmat<float,gl_ScopeWorkgroup,BLOCK_Q,BLOCK_KV,gl_MatrixUseAccumulator> oldOrigin=origin;
    coopMatPerElementNV(origin,tileMax,maxPerElement,oldOrigin);

    coopmat<float,gl_ScopeWorkgroup,BLOCK_Q,BLOCK_KV,gl_MatrixUseAccumulator> probsF,rescale,tileSum;
    coopMatPerElementNV(probsF,scores-origin,maskedExp2,n,qStart,kvStart);
    coopMatPerElementNV(rescale,oldOrigin-origin,exp2PerElement);
    coopMatReduceNV(tileSum,probsF,gl_CooperativeMatrixReduceRowNV,sumReduce);
    rowSum=rescale*rowSum+tileSum;

    coopmat<float,gl_ScopeWorkgroup,BLOCK_Q,V_HEAD_DIM,gl_MatrixUseAccumulator> rescaleOut;
    coopMatReduceNV(rescaleOut,rescale,gl_CooperativeMatrixReduceRowNV,smearReduce);
    outAcc*=rescaleOut;

    coopmat<float16_t,gl_ScopeWorkgroup,BLOCK_Q,BLOCK_KV,gl_MatrixUseA> probs=
      coopmat<float16_t,gl_ScopeWorkgroup,BLOCK_Q,BLOCK_KV,gl_MatrixUseA>(probsF);
    coopmat<float16_t,gl_ScopeWorkgroup,BLOCK_KV,V_HEAD_DIM,gl_MatrixUseB> values;
    const uint vBase = n * SEQ_LEN_SPEC * uint(pc.numKVHeads) * V_HEAD_DIM;
    if(ALIGNED_TILES != 0u)
      coopMatLoadTensorNV(values, vBuf.data, vBase,
        sliceTensorLayoutNV(vLayout, kvStart, BLOCK_KV, kvh * V_HEAD_DIM, V_HEAD_DIM));
    else
      coopMatLoadTensorNV(values, vBuf.data, vBase,
        sliceTensorLayoutNV(vLayoutClamp, kvStart, BLOCK_KV, kvh * V_HEAD_DIM, V_HEAD_DIM));
    outAcc=coopMatMulAdd(probs,values,outAcc);
    // kvMask is read by the per-element callbacks above. Do not let an early
    // invocation overwrite it for the next tile before every invocation has
    // completed those reads.
    barrier();
  }
  coopmat<float,gl_ScopeWorkgroup,BLOCK_Q,BLOCK_KV,gl_MatrixUseAccumulator> invSum;
  coopMatPerElementNV(invSum,rowSum,inversePerElement);
  coopmat<float,gl_ScopeWorkgroup,BLOCK_Q,V_HEAD_DIM,gl_MatrixUseAccumulator> invSumOut;
  coopMatReduceNV(invSumOut,invSum,gl_CooperativeMatrixReduceRowNV,smearReduce);
  coopmat<float16_t,gl_ScopeWorkgroup,BLOCK_Q,V_HEAD_DIM,gl_MatrixUseAccumulator> outputMatrix=
    coopmat<float16_t,gl_ScopeWorkgroup,BLOCK_Q,V_HEAD_DIM,gl_MatrixUseAccumulator>(outAcc*invSumOut);
  const uint outBase = n * SEQ_LEN_SPEC * uint(pc.numHeads) * V_HEAD_DIM;
  if(ALIGNED_TILES != 0u)
    coopMatStoreTensorNV(outputMatrix, outBuf.data, outBase,
      sliceTensorLayoutNV(outLayout, qStart, BLOCK_Q, h * V_HEAD_DIM, V_HEAD_DIM));
  else
    coopMatStoreTensorNV(outputMatrix, outBuf.data, outBase,
      sliceTensorLayoutNV(outLayoutClamp, qStart, BLOCK_Q, h * V_HEAD_DIM, V_HEAD_DIM));
}

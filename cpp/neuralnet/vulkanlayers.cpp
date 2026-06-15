#ifdef USE_VULKAN_BACKEND

#include "../neuralnet/vulkanlayers.h"
#include <algorithm>
#include <cmath>
#include <memory>
#include "../core/test.h"

using namespace VulkanHelpers;
using std::vector;

// Out-of-line definitions for the transformer / nested-bottleneck blocks, trunk,
// and policy/value heads declared in vulkanlayers.h. The leaf layers (ConvLayer,
// BatchNormLayer, MatMulLayer, MatBiasLayer, NormActConv) and the residual /
// global-pooling blocks are inline in the header.

// ============================================================================
// SGFMetadataEncoderVk
// ============================================================================

SGFMetadataEncoderVk::SGFMetadataEncoderVk(const VulkanLayerContext& ctx, const SGFMetadataEncoderDesc* desc)
  : mul1(ctx, &desc->mul1),
    bias1(ctx, &desc->bias1, &desc->act1),
    mul2(ctx, &desc->mul2),
    bias2(ctx, &desc->bias2, &desc->act2),
    mul3(ctx, &desc->mul3) {}

void SGFMetadataEncoderVk::dispatch(
  const CmdCtx& ctx,
  ScratchBuffers* scratch,
  VulkanBuffer* input,
  VulkanBuffer* output,
  int maxBatchSize) const {
  int maxMid = std::max(mul1.outChannels, mul2.outChannels);
  SizedBuf<VulkanBuffer*> buf1(scratch->allocator.get(), scratch->getBufSizeFloat(maxMid));
  SizedBuf<VulkanBuffer*> buf2(scratch->allocator.get(), scratch->getBufSizeFloat(maxMid));
  mul1.dispatch(ctx, input, buf1.buf, maxBatchSize);
  barrierFor(ctx, buf1.buf, "mul1", "bias1");

  bias1.dispatch(ctx, buf1.buf, maxBatchSize);
  barrierFor(ctx, buf1.buf, "bias1", "mul2");

  mul2.dispatch(ctx, buf1.buf, buf2.buf, maxBatchSize);
  barrierFor(ctx, buf2.buf, "mul2", "bias2");

  bias2.dispatch(ctx, buf2.buf, maxBatchSize);
  barrierFor(ctx, buf2.buf, "bias2", "mul3");

  mul3.dispatch(ctx, buf2.buf, output, maxBatchSize);
  barrierFor(ctx, output, "mul3", "addChannelBiases");
}

// ============================================================================
// RMSNormVk
// ============================================================================

RMSNormVk::RMSNormVk(
  const VulkanLayerContext& ctx,
  const RMSNormLayerDesc* desc,
  int activation_,
  int paddedSpatialSize_,
  bool useFP16)
  : numChannels(desc->numChannels),
    epsilon(desc->epsilon),
    spatial(desc->spatial),
    activation(activation_),
    paddedSpatialSize(paddedSpatialSize_),
    device(ctx.device) {
  // gamma/beta always FP32 (used by rmsnorm shader which reads them as FP32)
  gammaBuf = makeWeightBuf(ctx.device, ctx.queue, ctx.queueMutex, ctx.commandPool, ctx.memProps, desc->gamma, false);
  betaBuf = makeWeightBuf(ctx.device, ctx.queue, ctx.queueMutex, ctx.commandPool, ctx.memProps, desc->beta, false);
  if(activation_ != ACTIVATION_IDENTITY) {
    // scale=1, bias=0 buffers for the post-normalisation activation pass
    vector<float> ones(numChannels, 1.0f);
    vector<float> zeros(numChannels, 0.0f);
    actScaleBuf = makeWeightBuf(ctx.device, ctx.queue, ctx.queueMutex, ctx.commandPool, ctx.memProps, ones, useFP16);
    actBiasBuf = makeWeightBuf(ctx.device, ctx.queue, ctx.queueMutex, ctx.commandPool, ctx.memProps, zeros, useFP16);
  }
  if(activation_ == ACTIVATION_SILU) {
    siluKernel = VulkanKernels::ScaleBiasMaskActInplaceNhwc::build(
      ctx.device, ctx.pipelineCache, useFP16, ACTIVATION_SILU, numChannels % 4 == 0);
  }
  // Fixed constants from the RMSNorm shader specs: per-position local_size_x
  // is 64 for both subgroup variants; WG_C_SIZE differs (32/8).
  // Pass 2's local_size_x is the tunable tile, so its eligibility must be
  // evaluated per-tile below.
  const uint32_t rmsNormPositionLocalSize = 64u;
  const uint32_t rmsNormPositionWgCSize = 32u;
  const bool useTransformerRMSNormSubgroup = canUseRMSNormSubgroupVariant(
    ctx.supportsSubgroupShuffleCompute,
    ctx.canRequireReportedSubgroupSize,
    ctx.subgroupSize,
    rmsNormPositionLocalSize,
    rmsNormPositionWgCSize);

  // RMSNorm path-specific kernels: build only the set that this instance dispatches.
  if(!spatial) {
    rmsNormKernel = VulkanKernels::TransformerRMSNormNhwc::build(
      ctx.device,
      ctx.pipelineCache,
      useFP16,
      useTransformerRMSNormSubgroup,
      useTransformerRMSNormSubgroup ? ctx.subgroupSize : 0u);
  } else {
    // Three-pass spatial path. Pass 2's WG size == tile; the subgroup variant
    // is only eligible when the tile is a multiple of the subgroup size (which
    // e.g. rules out tile=16 on a subgroup-size-32 device).
    const int32_t tile = ctx.tuneParams.spatialRMSNormNhwcTile;
    const bool useSpatialRMSNormPass2Subgroup = canUseRMSNormSubgroupVariant(
      ctx.supportsSubgroupShuffleCompute,
      ctx.canRequireReportedSubgroupSize,
      ctx.subgroupSize,
      (uint32_t)tile,
      /*wgChannelGroup=*/1u);
    spatialKernels = VulkanKernels::SpatialRMSNormNhwc::build(
      ctx.device,
      ctx.pipelineCache,
      useFP16,
      ctx.tuneParams,
      useSpatialRMSNormPass2Subgroup,
      useSpatialRMSNormPass2Subgroup ? ctx.subgroupSize : 0u);
  }
}

std::vector<VkDeviceSize> RMSNormVk::permanentScratchSlotSizes(int maxBatchSize) const {
  if(!spatial)
    return {};
  const int tile = (int)spatialKernels[0].localSizeX;
  const int chwSize = numChannels * paddedSpatialSize;
  const int numCHWWorkgroups = (chwSize + tile - 1) / tile;
  // slot0: numCHWWorkgroups partial sums per batch element (FP32 always)
  VkDeviceSize slot0 = (VkDeviceSize)maxBatchSize * numCHWWorkgroups * sizeof(float);
  // slot1: one scalar per batch element (FP32 always)
  VkDeviceSize slot1 = (VkDeviceSize)maxBatchSize * sizeof(float);
  return {slot0, slot1};
}

void RMSNormVk::dispatch(
  const CmdCtx& ctx,
  VulkanBuffer* input,
  VulkanBuffer* output,
  VulkanBuffer* mask,
  VulkanBuffer* maskSum,
  ScratchBuffers* scratch,
  int maxBatchSize) const {
  if(!spatial) {
    VulkanKernels::TransformerRMSNormNhwc::PC pc = {numChannels, paddedSpatialSize, epsilon};
    VulkanKernels::TransformerRMSNormNhwc::dispatch(
      ctx, rmsNormKernel, input, output, gammaBuf.get(), betaBuf.get(), mask, pc, maxBatchSize);
  } else {
    // Spatial 3-pass RMSNorm — uses permanent scratch slots 0 (pass1 partials) and 1 (pass2 scalar)
    VulkanBuffer* ws0 = scratch->permanentSlot(0);
    VulkanBuffer* ws1 = scratch->permanentSlot(1);
    VulkanKernels::SpatialRMSNormNhwc::PC pc = {numChannels, paddedSpatialSize, epsilon};
    VulkanKernels::SpatialRMSNormNhwc::dispatch(
      ctx, spatialKernels, input, output, gammaBuf.get(), betaBuf.get(), mask, maskSum, ws0, ws1, pc, maxBatchSize);

    warBarrierFor(ctx, ws0, "release of permanent scratch slot");

    warBarrierFor(ctx, ws1, "release of permanent scratch slot");
  }
  // Apply SILU activation in-place on output if needed
  if(activation == ACTIVATION_SILU) {
    // The RMSNorm pass above wrote output; make it visible before the in-place SILU reads it.
    barrierFor(ctx, output, "RMSNorm", "the in-place SILU activation");

    VulkanKernels::ScaleBiasMaskActInplaceNhwc::PC pc = {numChannels, paddedSpatialSize, maxBatchSize};
    VulkanKernels::ScaleBiasMaskActInplaceNhwc::dispatch(
      ctx, siluKernel, output, actScaleBuf.get(), actBiasBuf.get(), mask, pc);
  }
  barrierFor(ctx, output, "RMSNorm/SILU", "the next consumer");
}

// ============================================================================
// Transformer RMSNorm (per-position, no-bias form of RMSNormVk)
// ============================================================================

RMSNormVk::RMSNormVk(
  const VulkanLayerContext& ctx,
  const TransformerRMSNormDesc* desc,
  int paddedSpatialSize_,
  bool useFP16)
  : numChannels(desc->numChannels),
    epsilon(desc->epsilon),
    spatial(false),
    activation(ACTIVATION_IDENTITY),
    paddedSpatialSize(paddedSpatialSize_),
    device(ctx.device) {
  // The transformer RMSNorm has no beta: gamma = the weight, beta = zeros.
  gammaBuf = makeWeightBuf(ctx.device, ctx.queue, ctx.queueMutex, ctx.commandPool, ctx.memProps, desc->weight, false);
  vector<float> zeros(numChannels, 0.0f);
  betaBuf = makeWeightBuf(ctx.device, ctx.queue, ctx.queueMutex, ctx.commandPool, ctx.memProps, zeros, false);
  const uint32_t rmsNormPositionLocalSize = 64u;
  const uint32_t rmsNormPositionWgCSize = 32u;
  const bool useTransformerRMSNormSubgroup = canUseRMSNormSubgroupVariant(
    ctx.supportsSubgroupShuffleCompute,
    ctx.canRequireReportedSubgroupSize,
    ctx.subgroupSize,
    rmsNormPositionLocalSize,
    rmsNormPositionWgCSize);
  rmsNormKernel = VulkanKernels::TransformerRMSNormNhwc::build(
    ctx.device,
    ctx.pipelineCache,
    useFP16,
    useTransformerRMSNormSubgroup,
    useTransformerRMSNormSubgroup ? ctx.subgroupSize : 0u);
}

// ============================================================================
// TransformerMatMulLayerVk
// ============================================================================

TransformerMatMulLayerVk::TransformerMatMulLayerVk(
  const VulkanLayerContext& ctx,
  const MatMulLayerDesc* desc,
  int paddedSpatialSize_,
  bool useFP16,
  bool addToOutput_,
  int padNumGroups,
  int padTrueChPerGroup,
  int padPaddedChPerGroup,
  bool padOutput)
  : inChannels(
      padNumGroups > 0 && padPaddedChPerGroup > padTrueChPerGroup && !padOutput ? padNumGroups * padPaddedChPerGroup
                                                                                : desc->inChannels),
    outChannels(
      padNumGroups > 0 && padPaddedChPerGroup > padTrueChPerGroup && padOutput ? padNumGroups * padPaddedChPerGroup
                                                                               : desc->outChannels),
    paddedSpatialSize(paddedSpatialSize_),
    addToOutput(addToOutput_) {
  ConvLayerDesc convDesc;
  convDesc.name = desc->name;
  convDesc.convYSize = 1;
  convDesc.convXSize = 1;
  convDesc.inChannels = inChannels;
  convDesc.outChannels = outChannels;
  convDesc.dilationY = 1;
  convDesc.dilationX = 1;
  convDesc.weights.resize((size_t)inChannels * outChannels);
  const bool doPad = padNumGroups > 0 && padPaddedChPerGroup > padTrueChPerGroup;
  const int srcOutChannels = desc->outChannels;
  for(int ic = 0; ic < inChannels; ic++) {
    for(int oc = 0; oc < outChannels; oc++) {
      float w = 0.0f;
      if(!doPad) {
        w = desc->weights[(size_t)ic * srcOutChannels + oc];
      } else if(padOutput) {
        const int group = oc / padPaddedChPerGroup;
        const int d = oc - group * padPaddedChPerGroup;
        if(d < padTrueChPerGroup)
          w = desc->weights[(size_t)ic * srcOutChannels + (group * padTrueChPerGroup + d)];
      } else {
        const int group = ic / padPaddedChPerGroup;
        const int d = ic - group * padPaddedChPerGroup;
        if(d < padTrueChPerGroup)
          w = desc->weights[(size_t)(group * padTrueChPerGroup + d) * srcOutChannels + oc];
      }
      convDesc.weights[(size_t)oc * inChannels + ic] = w;
    }
  }
  nhwcConv = std::make_unique<ConvLayer>(ctx, &convDesc, paddedSpatialSize, 1, paddedSpatialSize, useFP16, addToOutput);
}

void TransformerMatMulLayerVk::dispatch(
  const CmdCtx& ctx,
  ScratchBuffers* scratch,
  VulkanBuffer* input,
  VulkanBuffer* output,
  int maxBatchSize) const {
  nhwcConv->dispatch(ctx, scratch, input, output, maxBatchSize);
}

// ============================================================================
// TransformerAttentionBlockVk
// ============================================================================

namespace {
  bool coopmat2AttentionShapeSupported(
    const std::vector<Coopmat2FlexShape>& shapes,
    int workgroupSize,
    int blockQ,
    int blockKV,
    int headDim,
    int vHeadDim) {
    bool qk = false, pv = false;
    for(const Coopmat2FlexShape& s: shapes) {
      if(
        (int)s.workgroupInvocations != workgroupSize || s.mGranularity == 0 || s.nGranularity == 0 ||
        s.kGranularity == 0)
        continue;
      qk = qk || (blockQ % (int)s.mGranularity == 0 && blockKV % (int)s.nGranularity == 0 &&
                  headDim % (int)s.kGranularity == 0);
      pv = pv || (blockQ % (int)s.mGranularity == 0 && vHeadDim % (int)s.nGranularity == 0 &&
                  blockKV % (int)s.kGranularity == 0);
    }
    return qk && pv;
  }

  // Residual-add tail shared by the transformer attention and FFN blocks: the
  // projection writes its result into trunk, which is also being read as the
  // residual source, so fence the previous trunk consumer's reads first (WAR)
  // and then make the write visible to the next block / trunk consumer.
  void dispatchResidualAddToTrunk(
    const CmdCtx& ctx,
    ScratchBuffers* scratch,
    const TransformerMatMulLayerVk& proj,
    VulkanBuffer* src,
    VulkanBuffer* trunk,
    int maxBatchSize) {
    cmdComputeWARBarrier(ctx.cmd, trunk->buffer);
    proj.dispatch(ctx, scratch, src, trunk, maxBatchSize);
    cmdComputeBarrier(ctx.cmd, trunk->buffer);
  }

  TransformerAttentionBlockVk::AttentionVariant selectAttentionVariant(
    const VulkanLayerContext& ctx,
    const TransformerAttentionDesc* desc,
    int paddedSpatialSize,
    bool useFP16,
    int headDim,
    int vHeadDim) {
    using Variant = TransformerAttentionBlockVk::AttentionVariant;
    if(!useFP16)
      return Variant::Tiled;
    const VulkanTuneParams& tp = ctx.tuneParams;

    // Each cooperative tier is usable if it was tuned, measured, and fits this
    // model's shape/shared-memory budget. Pick the fastest measured coopmat
    // tier; DOT2 is the fallback when no coopmat tier remains.
    struct Candidate {
      Variant variant;
      int32_t timeUs;
      bool usable;
    };
    Candidate coop2 = {Variant::Coopmat2AccF32, tp.attnNhwcCoopmat2TimeUs, false};
    if(
      ctx.supportsCoopmat2Attention && tp.hasKernelTuned(VulkanTuner::TUNED_ATTN_COOPMAT2) &&
      tp.attnNhwcCoopmat2TimeUs > 0 &&
      coopmat2AttentionShapeSupported(
        ctx.coopmat2FlexShapes,
        tp.attnNhwcCoopmat2WorkgroupSize,
        tp.attnNhwcCoopmat2BlockQ,
        tp.attnNhwcCoopmat2BlockKV,
        headDim,
        vHeadDim) &&
      VulkanKernels::AttentionCoopmat2AccF32Nhwc::isConfigSupported(
        tp.attnNhwcCoopmat2WorkgroupSize, tp.attnNhwcCoopmat2BlockQ, tp.attnNhwcCoopmat2BlockKV, headDim, vHeadDim) &&
      VulkanKernels::AttentionCoopmat2AccF32Nhwc::sharedBytes(
        tp.attnNhwcCoopmat2BlockQ, tp.attnNhwcCoopmat2BlockKV, headDim, vHeadDim) +
          ctx.coopmat2ReservedSharedBytes <=
        ctx.maxComputeSharedMemorySize)
      coop2.usable = true;

    Candidate maint = {Variant::CoopmatMaintenance1AccF32, tp.attnNhwcCoopmatMaintenance1TimeUs, false};
    if(
      ctx.supportsCoopmatMaintenance1 && tp.hasKernelTuned(VulkanTuner::TUNED_ATTN_MAINTENANCE1) &&
      tp.attnNhwcCoopmatMaintenance1TimeUs > 0 &&
      tp.attnNhwcCoopmatMaintenance1SubgroupSize == (int32_t)ctx.subgroupSize &&
      coopmatShapeSupported(
        ctx.coopmatShapes,
        tp.attnNhwcCoopmatMaintenance1TM,
        tp.attnNhwcCoopmatMaintenance1TN,
        tp.attnNhwcCoopmatMaintenance1TK) &&
      VulkanKernels::AttentionCoopmat1Nhwc::isConfigSupported(
        tp.attnNhwcCoopmatMaintenance1WorkgroupSize,
        tp.attnNhwcCoopmatMaintenance1BlockQ,
        tp.attnNhwcCoopmatMaintenance1BlockKV,
        tp.attnNhwcCoopmatMaintenance1TM,
        tp.attnNhwcCoopmatMaintenance1TN,
        tp.attnNhwcCoopmatMaintenance1TK,
        tp.attnNhwcCoopmatMaintenance1SubgroupSize,
        headDim,
        vHeadDim) &&
      VulkanKernels::AttentionCoopmat1Nhwc::sharedBytes(
        tp.attnNhwcCoopmatMaintenance1BlockQ,
        tp.attnNhwcCoopmatMaintenance1BlockKV,
        tp.attnNhwcCoopmatMaintenance1TK,
        tp.attnNhwcCoopmatMaintenance1TN,
        headDim,
        vHeadDim,
        desc->useRope,
        tp.attnNhwcCoopmatMaintenance1DirectKV != 0 && paddedSpatialSize % tp.attnNhwcCoopmatMaintenance1BlockKV == 0,
        true) <= ctx.maxComputeSharedMemorySize)
      maint.usable = true;

    Candidate coop1 = {Variant::Coopmat1AccF32, tp.attnNhwcCoopmat1TimeUs, false};
    if(
      ctx.supportsCoopmat1F16 && tp.hasKernelTuned(VulkanTuner::TUNED_ATTN_COOPMAT1) && tp.attnNhwcCoopmat1TimeUs > 0 &&
      tp.attnNhwcCoopmat1SubgroupSize == (int32_t)ctx.subgroupSize &&
      coopmatShapeSupported(ctx.coopmatShapes, tp.attnNhwcCoopmat1TM, tp.attnNhwcCoopmat1TN, tp.attnNhwcCoopmat1TK) &&
      VulkanKernels::AttentionCoopmat1Nhwc::isConfigSupported(
        tp.attnNhwcCoopmat1WorkgroupSize,
        tp.attnNhwcCoopmat1BlockQ,
        tp.attnNhwcCoopmat1BlockKV,
        tp.attnNhwcCoopmat1TM,
        tp.attnNhwcCoopmat1TN,
        tp.attnNhwcCoopmat1TK,
        tp.attnNhwcCoopmat1SubgroupSize,
        headDim,
        vHeadDim) &&
      VulkanKernels::AttentionCoopmat1Nhwc::sharedBytes(
        tp.attnNhwcCoopmat1BlockQ,
        tp.attnNhwcCoopmat1BlockKV,
        tp.attnNhwcCoopmat1TK,
        tp.attnNhwcCoopmat1TN,
        headDim,
        vHeadDim,
        desc->useRope,
        tp.attnNhwcCoopmat1DirectKV != 0 && paddedSpatialSize % tp.attnNhwcCoopmat1BlockKV == 0) <=
        ctx.maxComputeSharedMemorySize)
      coop1.usable = true;

    const Candidate* best = nullptr;
    for(const Candidate* c: {&coop2, &maint, &coop1})
      if(c->usable && (best == nullptr || c->timeUs < best->timeUs))
        best = c;
    if(best != nullptr)
      return best->variant;

    const bool dot2Usable =
      ctx.supportsDot2F16 && tp.hasKernelTuned(VulkanTuner::TUNED_ATTN_DOT2) && tp.attnNhwcDot2TimeUs > 0 &&
      VulkanKernels::AttentionDot2AccF32Nhwc::isConfigSupported(
        tp.attnNhwcDot2WorkgroupSize, tp.attnNhwcDot2BlockQ, tp.attnNhwcDot2BlockKV, headDim, vHeadDim) &&
      VulkanKernels::AttentionDot2AccF32Nhwc::sharedBytes(tp.attnNhwcDot2BlockKV, headDim, vHeadDim) <=
        ctx.maxComputeSharedMemorySize;
    return dot2Usable ? Variant::Dot2AccF32 : Variant::Tiled;
  }

  AttentionPadDims resolveAttentionPadDims(
    const VulkanLayerContext& ctx,
    const TransformerAttentionDesc* desc,
    int paddedSpatialSize,
    bool useFP16) {
    AttentionPadDims d;
    d.qHeadDim = desc->qHeadDim;
    d.vHeadDim = desc->vHeadDim;
    d.padded = false;
    if(useFP16) {
      const int qp = roundUpToMultipleInt(desc->qHeadDim, 8);
      const int vp = roundUpToMultipleInt(desc->vHeadDim, 8);
      if(qp != desc->qHeadDim || vp != desc->vHeadDim) {
        const TransformerAttentionBlockVk::AttentionVariant variant =
          selectAttentionVariant(ctx, desc, paddedSpatialSize, useFP16, qp, vp);
        if(variant != TransformerAttentionBlockVk::AttentionVariant::Tiled) {
          d.qHeadDim = qp;
          d.vHeadDim = vp;
          d.padded = true;
        }
      }
    }
    return d;
  }

  void padRopeTables(
    vector<float>& cosTable,
    vector<float>& sinTable,
    int numTableHeads,
    int trueNumPairs,
    int paddedNumPairs,
    int paddedSpatialSize) {
    if(paddedNumPairs <= trueNumPairs)
      return;
    vector<float> cosPadded((size_t)numTableHeads * paddedNumPairs * paddedSpatialSize, 0.0f);
    vector<float> sinPadded((size_t)numTableHeads * paddedNumPairs * paddedSpatialSize, 0.0f);
    for(int h = 0; h < numTableHeads; h++) {
      for(int p = 0; p < trueNumPairs; p++) {
        for(int xy = 0; xy < paddedSpatialSize; xy++) {
          const size_t src = ((size_t)h * trueNumPairs + p) * paddedSpatialSize + xy;
          const size_t dst = ((size_t)h * paddedNumPairs + p) * paddedSpatialSize + xy;
          cosPadded[dst] = cosTable[src];
          sinPadded[dst] = sinTable[src];
        }
      }
      for(int p = trueNumPairs; p < paddedNumPairs; p++) {
        for(int xy = 0; xy < paddedSpatialSize; xy++) {
          const size_t dst = ((size_t)h * paddedNumPairs + p) * paddedSpatialSize + xy;
          cosPadded[dst] = 1.0f;
          sinPadded[dst] = 0.0f;
        }
      }
    }
    cosTable.swap(cosPadded);
    sinTable.swap(sinPadded);
  }
}  // namespace

TransformerAttentionBlockVk::TransformerAttentionBlockVk(
  const VulkanLayerContext& ctx,
  const TransformerAttentionDesc* desc,
  int nnX,
  int nnY,
  int paddedSpatialSize_,
  bool useFP16)
  : numHeads(desc->numHeads),
    numKVHeads(desc->numKVHeads),
    qHeadDim(desc->qHeadDim),
    vHeadDim(desc->vHeadDim),
    padDims(resolveAttentionPadDims(ctx, desc, paddedSpatialSize_, useFP16)),
    qHeadDimPadded(padDims.qHeadDim),
    vHeadDimPadded(padDims.vHeadDim),
    paddedHeads(padDims.padded),
    useRope(desc->useRope),
    learnableRope(desc->learnableRope),
    inChannels(desc->qProj.inChannels),
    paddedSpatialSize(paddedSpatialSize_),
    attentionVariant(
      selectAttentionVariant(ctx, desc, paddedSpatialSize_, useFP16, padDims.qHeadDim, padDims.vHeadDim)),
    preLN(ctx, &desc->preLN, paddedSpatialSize_, useFP16),
    qProj(
      ctx,
      &desc->qProj,
      paddedSpatialSize_,
      useFP16,
      false,
      paddedHeads ? numHeads : 0,
      paddedHeads ? qHeadDim : 0,
      paddedHeads ? qHeadDimPadded : 0,
      true),
    kProj(
      ctx,
      &desc->kProj,
      paddedSpatialSize_,
      useFP16,
      false,
      paddedHeads ? numKVHeads : 0,
      paddedHeads ? qHeadDim : 0,
      paddedHeads ? qHeadDimPadded : 0,
      true),
    vProj(
      ctx,
      &desc->vProj,
      paddedSpatialSize_,
      useFP16,
      false,
      paddedHeads ? numKVHeads : 0,
      paddedHeads ? vHeadDim : 0,
      paddedHeads ? vHeadDimPadded : 0,
      true),
    outProj(
      ctx,
      &desc->outProj,
      paddedSpatialSize_,
      useFP16,
      true,
      paddedHeads ? numHeads : 0,
      paddedHeads ? vHeadDim : 0,
      paddedHeads ? vHeadDimPadded : 0,
      false),
    device(ctx.device) {
  testAssert(numKVHeads > 0 && numHeads % numKVHeads == 0);
  testAssert(ctx.tuneParams.attnNhwcBlockKV <= ctx.tuneParams.attnNhwcBlockQ);
  if(useRope) {
    vector<float> cosVec, sinVec;
    desc->computeRopeCosSin(nnX, nnY, paddedSpatialSize_, cosVec, sinVec);
    // The accelerated attention kernels operate on the padded head dim, so
    // extend the pair count to match and give the padding pairs an identity
    // rotation (cos=1, sin=0). Their Q/K operands are zero there anyway.
    if(paddedHeads)
      padRopeTables(
        cosVec, sinVec, learnableRope ? numKVHeads : 1, qHeadDim / 2, qHeadDimPadded / 2, paddedSpatialSize_);
    // Coopmat2 tensor-load callbacks vary RoPE pair
    // fastest, so transpose the otherwise pair-major tables once at model
    // construction to make those reads contiguous. Other attention variants
    // continue using the original pair-major table layout.
    if(attentionVariant == AttentionVariant::Coopmat2AccF32) {
      const int tableHeads = learnableRope ? numKVHeads : 1;
      const int numPairs = qHeadDimPadded / 2;
      auto transposeRopeTable = [&](vector<float>& table) {
        vector<float> transposed(table.size());
        for(int h = 0; h < tableHeads; h++)
          for(int pair = 0; pair < numPairs; pair++)
            for(int pos = 0; pos < paddedSpatialSize_; pos++)
              transposed[((size_t)h * paddedSpatialSize_ + pos) * numPairs + pair] =
                table[((size_t)h * numPairs + pair) * paddedSpatialSize_ + pos];
        table.swap(transposed);
      };
      transposeRopeTable(cosVec);
      transposeRopeTable(sinVec);
    }
    ropeCosTable = makeWeightBuf(ctx.device, ctx.queue, ctx.queueMutex, ctx.commandPool, ctx.memProps, cosVec, false);
    ropeSinTable = makeWeightBuf(ctx.device, ctx.queue, ctx.queueMutex, ctx.commandPool, ctx.memProps, sinVec, false);
  }
  // Attention is specialized for this block's (seqLen,qHeadDim,vHeadDim).
  // Map the selected maintenance1 result onto the shared coopmat1 builder
  // interface. This is block-local: the persisted portable fields remain
  // untouched and can still be selected on another device/configuration.
  VulkanTuneParams attentionTune = ctx.tuneParams;
  const bool useMaintenance1 = attentionVariant == AttentionVariant::CoopmatMaintenance1AccF32;
  if(useMaintenance1) {
    attentionTune.attnNhwcCoopmat1WorkgroupSize = attentionTune.attnNhwcCoopmatMaintenance1WorkgroupSize;
    attentionTune.attnNhwcCoopmat1BlockQ = attentionTune.attnNhwcCoopmatMaintenance1BlockQ;
    attentionTune.attnNhwcCoopmat1BlockKV = attentionTune.attnNhwcCoopmatMaintenance1BlockKV;
    attentionTune.attnNhwcCoopmat1TM = attentionTune.attnNhwcCoopmatMaintenance1TM;
    attentionTune.attnNhwcCoopmat1TN = attentionTune.attnNhwcCoopmatMaintenance1TN;
    attentionTune.attnNhwcCoopmat1TK = attentionTune.attnNhwcCoopmatMaintenance1TK;
    attentionTune.attnNhwcCoopmat1SubgroupSize = attentionTune.attnNhwcCoopmatMaintenance1SubgroupSize;
    attentionTune.attnNhwcCoopmat1DirectKV = attentionTune.attnNhwcCoopmatMaintenance1DirectKV;
    if(attentionTune.hasKernelTuned(VulkanTuner::TUNED_ATTN_MAINTENANCE1_SPLITK))
      attentionTune.markKernelTuned({VulkanTuner::TUNED_ATTN_COOPMAT1_SPLITK});
    else
      attentionTune.clearKernelTuned({VulkanTuner::TUNED_ATTN_COOPMAT1_SPLITK});
    attentionTune.attnNhwcCoopmat1SplitKWorkgroupSize = attentionTune.attnNhwcCoopmatMaintenance1SplitKWorkgroupSize;
    attentionTune.attnNhwcCoopmat1SplitKBlockQ = attentionTune.attnNhwcCoopmatMaintenance1SplitKBlockQ;
    attentionTune.attnNhwcCoopmat1SplitKBlockKV = attentionTune.attnNhwcCoopmatMaintenance1SplitKBlockKV;
    attentionTune.attnNhwcCoopmat1SplitKTM = attentionTune.attnNhwcCoopmatMaintenance1SplitKTM;
    attentionTune.attnNhwcCoopmat1SplitKTN = attentionTune.attnNhwcCoopmatMaintenance1SplitKTN;
    attentionTune.attnNhwcCoopmat1SplitKTK = attentionTune.attnNhwcCoopmatMaintenance1SplitKTK;
    attentionTune.attnNhwcCoopmat1SplitKSubgroupSize = attentionTune.attnNhwcCoopmatMaintenance1SplitKSubgroupSize;
    attentionTune.attnNhwcCoopmat1SplitKDirectKV = attentionTune.attnNhwcCoopmatMaintenance1SplitKDirectKV;
    attentionTune.attnNhwcCoopmat1SplitKKVChunkCount = attentionTune.attnNhwcCoopmatMaintenance1SplitKKVChunkCount;
    attentionTune.attnNhwcCoopmat1SplitKCutoffBatch = attentionTune.attnNhwcCoopmatMaintenance1SplitKCutoffBatch;
  }
  if(attentionVariant == AttentionVariant::Coopmat1AccF32 || useMaintenance1) {
    attentionKernel = VulkanKernels::AttentionCoopmat1Nhwc::build(
      ctx.device,
      ctx.pipelineCache,
      attentionTune,
      paddedSpatialSize_,
      qHeadDimPadded,
      vHeadDimPadded,
      useRope,
      learnableRope,
      ctx.subgroupSize,
      /*kvChunkCount=*/1,
      useMaintenance1);

    // A tune file can be reused at a smaller board size. Clamp to the number
    // of KV tiles so every dispatched z-slice produces resolver input.
    const int kvTiles = (paddedSpatialSize_ + attentionTune.attnNhwcCoopmat1SplitKBlockKV - 1) /
                        attentionTune.attnNhwcCoopmat1SplitKBlockKV;
    const int kvChunkCount = std::min(attentionTune.attnNhwcCoopmat1SplitKKVChunkCount, kvTiles);
    // The split-K tile is tuned independently of the regular one, so it needs
    // its own resource check. Leaving splitKResolveKernel null makes dispatch()
    // fall back to the single-pass path rather than failing pipeline creation.
    const bool splitKFits = VulkanKernels::AttentionCoopmat1Nhwc::isConfigSupported(
                              attentionTune.attnNhwcCoopmat1SplitKWorkgroupSize,
                              attentionTune.attnNhwcCoopmat1SplitKBlockQ,
                              attentionTune.attnNhwcCoopmat1SplitKBlockKV,
                              attentionTune.attnNhwcCoopmat1SplitKTM,
                              attentionTune.attnNhwcCoopmat1SplitKTN,
                              attentionTune.attnNhwcCoopmat1SplitKTK,
                              attentionTune.attnNhwcCoopmat1SplitKSubgroupSize,
                              qHeadDimPadded,
                              vHeadDimPadded) &&
                            attentionTune.attnNhwcCoopmat1SplitKSubgroupSize == (int32_t)ctx.subgroupSize &&
                            VulkanKernels::AttentionCoopmat1Nhwc::sharedBytes(
                              attentionTune.attnNhwcCoopmat1SplitKBlockQ,
                              attentionTune.attnNhwcCoopmat1SplitKBlockKV,
                              attentionTune.attnNhwcCoopmat1SplitKTK,
                              attentionTune.attnNhwcCoopmat1SplitKTN,
                              qHeadDimPadded,
                              vHeadDimPadded,
                              useRope,
                              attentionTune.attnNhwcCoopmat1SplitKDirectKV != 0 &&
                                paddedSpatialSize_ % attentionTune.attnNhwcCoopmat1SplitKBlockKV == 0,
                              useMaintenance1) <= ctx.maxComputeSharedMemorySize;
    if(
      attentionTune.hasKernelTuned(VulkanTuner::TUNED_ATTN_COOPMAT1_SPLITK) &&
      attentionTune.attnNhwcCoopmat1SplitKCutoffBatch > 0 && kvChunkCount > 1 && splitKFits) {
      splitKCutoffBatch = attentionTune.attnNhwcCoopmat1SplitKCutoffBatch;
      splitKAttentionKernel = VulkanKernels::AttentionCoopmat1Nhwc::build(
        ctx.device,
        ctx.pipelineCache,
        attentionTune,
        paddedSpatialSize_,
        qHeadDimPadded,
        vHeadDimPadded,
        useRope,
        learnableRope,
        ctx.subgroupSize,
        kvChunkCount,
        useMaintenance1);
      splitKResolveKernel =
        VulkanKernels::AttentionSplitKResolveNhwc::build(ctx.device, ctx.pipelineCache, vHeadDimPadded);
    }
  } else if(attentionVariant == AttentionVariant::Coopmat2AccF32) {
    attentionKernel = VulkanKernels::AttentionCoopmat2AccF32Nhwc::build(
      ctx.device,
      ctx.pipelineCache,
      ctx.tuneParams,
      paddedSpatialSize_,
      qHeadDimPadded,
      vHeadDimPadded,
      useRope,
      learnableRope);
  } else if(attentionVariant == AttentionVariant::Dot2AccF32) {
    attentionKernel = VulkanKernels::AttentionDot2AccF32Nhwc::build(
      ctx.device,
      ctx.pipelineCache,
      ctx.tuneParams,
      paddedSpatialSize_,
      qHeadDimPadded,
      vHeadDimPadded,
      useRope,
      learnableRope);
  } else
    attentionKernel = VulkanKernels::AttentionTiledNhwc::build(
      ctx.device,
      ctx.pipelineCache,
      useFP16,
      ctx.tuneParams,
      paddedSpatialSize_,
      qHeadDimPadded,
      vHeadDimPadded,
      useRope,
      learnableRope);
}

void TransformerAttentionBlockVk::dispatch(
  const CmdCtx& ctx,
  ScratchBuffers* scratch,
  VulkanBuffer* trunk,
  VulkanBuffer* trunkScratch,
  VulkanBuffer* mask,
  [[maybe_unused]] VulkanBuffer* /*maskSum*/,
  int maxBatchSize) const {
  int seqLen = paddedSpatialSize;
  int qTotalDim = numHeads * qHeadDimPadded;
  int kTotalDim = numKVHeads * qHeadDimPadded;
  int vTotalDim = numKVHeads * vHeadDimPadded;

  // Step 1: RMSNorm trunk -> trunkScratch (barrier emitted by the layer)
  preLN.dispatch(ctx, trunk, trunkScratch, mask, nullptr, scratch, maxBatchSize);

  // Step 2: Q/K/V projections
  SizedBuf<VulkanBuffer*> qBuf(scratch->allocator.get(), scratch->getBufSizeXY(qTotalDim));
  SizedBuf<VulkanBuffer*> kBuf(scratch->allocator.get(), scratch->getBufSizeXY(kTotalDim));
  SizedBuf<VulkanBuffer*> vBuf(scratch->allocator.get(), scratch->getBufSizeXY(vTotalDim));

  qProj.dispatch(ctx, scratch, trunkScratch, qBuf.buf, maxBatchSize);
  kProj.dispatch(ctx, scratch, trunkScratch, kBuf.buf, maxBatchSize);
  vProj.dispatch(ctx, scratch, trunkScratch, vBuf.buf, maxBatchSize);
  barrierFor(ctx, qBuf.buf, "qProj", "RoPE/attention");

  barrierFor(ctx, kBuf.buf, "kProj", "RoPE/attention");

  barrierFor(ctx, vBuf.buf, "vProj", "attention");

  // Step 3: Scaled dot-product attention (tiled flash-attention; the per-block
  // ComputeKernel was specialised for this block's seqLen/qHeadDim/vHeadDim at ctor.)
  SizedBuf<VulkanBuffer*> attnOut(scratch->allocator.get(), scratch->getBufSizeXY(numHeads * vHeadDimPadded));
  float scale = 1.0f / sqrtf((float)qHeadDim);
  VulkanKernels::AttentionTiled::PC pcAttn = {numHeads, numKVHeads, scale, maxBatchSize * numHeads};
  if(
    attentionVariant == AttentionVariant::Coopmat1AccF32 ||
    attentionVariant == AttentionVariant::CoopmatMaintenance1AccF32) {
    if(splitKResolveKernel.pipeline != VK_NULL_HANDLE && maxBatchSize <= splitKCutoffBatch) {
      const auto* launch = std::get_if<LaunchProfile::AttentionAccelQ>(&splitKAttentionKernel.launch.data);
      int kvChunkCount = (int)launch->kvChunkCount;
      int numBH = maxBatchSize * numHeads;
      size_t partialsSize =
        VulkanKernels::AttentionCoopmat1Nhwc::partialsBytes(seqLen, numBH, kvChunkCount, vHeadDimPadded);
      size_t statsSize = VulkanKernels::AttentionCoopmat1Nhwc::statsBytes(seqLen, numBH, kvChunkCount);
      SizedBuf<VulkanBuffer*> partialsBuf(scratch->allocator.get(), partialsSize);
      SizedBuf<VulkanBuffer*> statsBuf(scratch->allocator.get(), statsSize);
      VulkanKernels::AttentionCoopmat1Nhwc::dispatch(
        ctx,
        splitKAttentionKernel,
        qBuf.buf,
        kBuf.buf,
        vBuf.buf,
        attnOut.buf,
        mask,
        useRope ? ropeCosTable.get() : nullptr,
        useRope ? ropeSinTable.get() : nullptr,
        seqLen,
        pcAttn,
        partialsBuf.buf,
        statsBuf.buf);
      barrierFor(ctx, partialsBuf.buf, "split-K attention", "resolve");
      barrierFor(ctx, statsBuf.buf, "split-K attention", "resolve");
      VulkanKernels::AttentionSplitKResolveNhwc::PC resolvePC = {numHeads, seqLen, kvChunkCount};
      VulkanKernels::AttentionSplitKResolveNhwc::dispatch(
        ctx, splitKResolveKernel, partialsBuf.buf, statsBuf.buf, attnOut.buf, seqLen, numBH, resolvePC);
    } else {
      VulkanKernels::AttentionCoopmat1Nhwc::dispatch(
        ctx,
        attentionKernel,
        qBuf.buf,
        kBuf.buf,
        vBuf.buf,
        attnOut.buf,
        mask,
        useRope ? ropeCosTable.get() : nullptr,
        useRope ? ropeSinTable.get() : nullptr,
        seqLen,
        pcAttn);
    }
  } else if(attentionVariant == AttentionVariant::Coopmat2AccF32)
    VulkanKernels::AttentionCoopmat2AccF32Nhwc::dispatch(
      ctx,
      attentionKernel,
      qBuf.buf,
      kBuf.buf,
      vBuf.buf,
      attnOut.buf,
      mask,
      useRope ? ropeCosTable.get() : nullptr,
      useRope ? ropeSinTable.get() : nullptr,
      seqLen,
      pcAttn);
  else if(attentionVariant == AttentionVariant::Dot2AccF32)
    VulkanKernels::AttentionDot2AccF32Nhwc::dispatch(
      ctx,
      attentionKernel,
      qBuf.buf,
      kBuf.buf,
      vBuf.buf,
      attnOut.buf,
      mask,
      useRope ? ropeCosTable.get() : nullptr,
      useRope ? ropeSinTable.get() : nullptr,
      seqLen,
      pcAttn);
  else
    VulkanKernels::AttentionTiledNhwc::dispatch(
      ctx,
      attentionKernel,
      qBuf.buf,
      kBuf.buf,
      vBuf.buf,
      attnOut.buf,
      mask,
      useRope ? ropeCosTable.get() : nullptr,
      useRope ? ropeSinTable.get() : nullptr,
      seqLen,
      pcAttn);
  barrierFor(ctx, attnOut.buf, "attention", "outProj");

  // Step 4: Output projection + residual add into trunk.
  dispatchResidualAddToTrunk(ctx, scratch, outProj, attnOut.buf, trunk, maxBatchSize);
}

// ============================================================================
// TransformerFFNBlockVk
// ============================================================================

TransformerFFNBlockVk::TransformerFFNBlockVk(
  const VulkanLayerContext& ctx,
  const TransformerFFNDesc* desc,
  int paddedSpatialSize_,
  bool useFP16)
  : numChannels(desc->numChannels),
    ffnChannels(desc->ffnChannels),
    paddedSpatialSize(paddedSpatialSize_),
    preLN(ctx, &desc->preLN, paddedSpatialSize_, useFP16),
    linear1(ctx, &desc->linear1, paddedSpatialSize_, useFP16),
    linear2(ctx, &desc->linear2, paddedSpatialSize_, useFP16, true),
    device(ctx.device) {
  if(!desc->useSwiGLU)
    throw StringError("Vulkan backend: non-SwiGLU transformer FFN not supported");
  linearGate = std::make_unique<TransformerMatMulLayerVk>(ctx, &desc->linearGate, paddedSpatialSize_, useFP16);
  swiGLUKernel = VulkanKernels::SwiGLU::build(ctx.device, ctx.pipelineCache, useFP16, ctx.tuneParams);
}

void TransformerFFNBlockVk::dispatch(
  const CmdCtx& ctx,
  ScratchBuffers* scratch,
  VulkanBuffer* trunk,
  VulkanBuffer* trunkScratch,
  VulkanBuffer* mask,
  [[maybe_unused]] VulkanBuffer* /*maskSum*/,
  int maxBatchSize) const {
  // Step 1: RMSNorm (barrier emitted by the layer)
  preLN.dispatch(ctx, trunk, trunkScratch, mask, nullptr, scratch, maxBatchSize);

  // Step 2: linear1 + gate projections
  SizedBuf<VulkanBuffer*> ffnBuf(scratch->allocator.get(), scratch->getBufSizeXY(ffnChannels));
  SizedBuf<VulkanBuffer*> gateBuf(scratch->allocator.get(), scratch->getBufSizeXY(ffnChannels));
  linear1.dispatch(ctx, scratch, trunkScratch, ffnBuf.buf, maxBatchSize);
  linearGate->dispatch(ctx, scratch, trunkScratch, gateBuf.buf, maxBatchSize);
  barrierFor(ctx, ffnBuf.buf, "linear1", "SwiGLU");

  barrierFor(ctx, gateBuf.buf, "linearGate", "SwiGLU");

  // Step 3: SwiGLU in-place into ffnBuf
  int totalSize = maxBatchSize * ffnChannels * paddedSpatialSize;
  VulkanKernels::SwiGLU::PC pc = {totalSize};
  VulkanKernels::SwiGLU::dispatch(ctx, swiGLUKernel, ffnBuf.buf, gateBuf.buf, ffnBuf.buf, pc);
  barrierFor(ctx, ffnBuf.buf, "SwiGLU", "linear2");

  // Step 4: linear2 + residual add into trunk.
  dispatchResidualAddToTrunk(ctx, scratch, linear2, ffnBuf.buf, trunk, maxBatchSize);
}

// ============================================================================
// NestedBottleneckResidualBlockVk + BlockStackVk
//
// Defined together because NestedBottleneck owns a BlockStackVk and BlockStackVk
// constructs NestedBottleneck (mutual dependency resolved by ordering the ctor /
// dispatch bodies after BlockStackVk's full definition).
// ============================================================================

NestedBottleneckResidualBlockVk::NestedBottleneckResidualBlockVk(
  const VulkanLayerContext& ctx,
  const NestedBottleneckResidualBlockDesc* desc,
  int nnX,
  int nnY,
  int paddedSpatialSize_,
  bool useFP16)
  : normActConv1(ctx, &desc->preBN, &desc->preActivation, &desc->preConv, nnX, nnY, paddedSpatialSize_, useFP16),
    normActConv2(
      ctx,
      &desc->postBN,
      &desc->postActivation,
      &desc->postConv,
      nnX,
      nnY,
      paddedSpatialSize_,
      useFP16,
      true),
    paddedSpatialSize(paddedSpatialSize_),
    device(ctx.device) {
  blocks = std::make_unique<BlockStackVk>(
    ctx, desc->blocks, desc->numBlocks, desc->preConv.outChannels, nnX, nnY, paddedSpatialSize_, useFP16);
  addPointwiseKernel = VulkanKernels::AddPointwise::build(ctx.device, ctx.pipelineCache, useFP16);
}

void NestedBottleneckResidualBlockVk::dispatch(
  const CmdCtx& ctx,
  ScratchBuffers* scratch,
  VulkanBuffer* trunk,
  VulkanBuffer* trunkScratch,
  VulkanBuffer* mask,
  VulkanBuffer* maskSum,
  int maxBatchSize) const {
  SizedBuf<VulkanBuffer*> mid(scratch->allocator.get(), scratch->getBufSizeXY(normActConv1.outChannels));
  SizedBuf<VulkanBuffer*> midScratch(scratch->allocator.get(), scratch->getBufSizeXY(normActConv1.outChannels));
  normActConv1.dispatch(ctx, trunk, trunkScratch, mid.buf, mask, scratch, maxBatchSize);
  barrierFor(ctx, mid.buf, "normActConv1", "blocks (BlockStack)");

  blocks->dispatch(ctx, scratch, mid.buf, midScratch.buf, mask, maskSum, maxBatchSize);
  barrierFor(ctx, mid.buf, "blocks", "normActConv2");

  if(normActConv2.fusesResidualAdd) {
    warBarrierFor(ctx, trunk, "residual add");
    normActConv2.dispatch(ctx, mid.buf, mid.buf, trunk, mask, scratch, maxBatchSize);
  } else {
    normActConv2.dispatch(ctx, mid.buf, mid.buf, trunkScratch, mask, scratch, maxBatchSize);
    barrierFor(ctx, trunkScratch, "normActConv2", "addPointwise (residual)");

    // Add residual.
    int totalElts = maxBatchSize * normActConv2.outChannels * paddedSpatialSize;
    {
      VulkanKernels::AddPointwise::PC pc = {totalElts};
      VulkanKernels::AddPointwise::dispatch(ctx, addPointwiseKernel, trunk, trunkScratch, pc);
    }
  }
  barrierFor(ctx, trunk, "addPointwise (residual)", "the next block / trunk consumer");
}

BlockStackVk::BlockStackVk(
  const VulkanLayerContext& ctx,
  const std::vector<std::pair<int, unique_ptr_void>>& descBlocks,
  int nBlocks,
  int trunkChannels,
  int nnX,
  int nnY,
  int paddedSpatialSize,
  bool useFP16)
  : numBlocks(nBlocks), trunkNumChannels(trunkChannels) {
  testAssert((int)descBlocks.size() == numBlocks);
  for(int i = 0; i < numBlocks; i++) {
    if(descBlocks[i].first == ORDINARY_BLOCK_KIND) {
      auto* desc = (ResidualBlockDesc*)descBlocks[i].second.get();
      blocks.push_back(std::make_unique<ResidualBlockVk>(ctx, desc, nnX, nnY, paddedSpatialSize, useFP16));
    } else if(descBlocks[i].first == GLOBAL_POOLING_BLOCK_KIND) {
      auto* desc = (GlobalPoolingResidualBlockDesc*)descBlocks[i].second.get();
      blocks.push_back(std::make_unique<GlobalPoolingResidualBlockVk>(ctx, desc, nnX, nnY, paddedSpatialSize, useFP16));
    } else if(descBlocks[i].first == NESTED_BOTTLENECK_BLOCK_KIND) {
      auto* desc = (NestedBottleneckResidualBlockDesc*)descBlocks[i].second.get();
      blocks.push_back(
        std::make_unique<NestedBottleneckResidualBlockVk>(ctx, desc, nnX, nnY, paddedSpatialSize, useFP16));
    } else if(descBlocks[i].first == TRANSFORMER_ATTENTION_BLOCK_KIND) {
      auto* desc = (TransformerAttentionDesc*)descBlocks[i].second.get();
      blocks.push_back(std::make_unique<TransformerAttentionBlockVk>(ctx, desc, nnX, nnY, paddedSpatialSize, useFP16));
    } else if(descBlocks[i].first == TRANSFORMER_FFN_BLOCK_KIND) {
      auto* desc = (TransformerFFNDesc*)descBlocks[i].second.get();
      blocks.push_back(std::make_unique<TransformerFFNBlockVk>(ctx, desc, paddedSpatialSize, useFP16));
    } else {
      ASSERT_UNREACHABLE;
    }
  }
}

void BlockStackVk::dispatch(
  const CmdCtx& ctx,
  ScratchBuffers* scratch,
  VulkanBuffer* trunk,
  VulkanBuffer* trunkScratch,
  VulkanBuffer* mask,
  VulkanBuffer* maskSum,
  int maxBatchSize) const {
  // Each block's dispatch() barriers its own trunk output (producer site), so the
  // next block can read trunk directly without an additional barrier here.
  for(auto& block: blocks)
    block->dispatch(ctx, scratch, trunk, trunkScratch, mask, maskSum, maxBatchSize);
}

std::vector<VkDeviceSize> BlockStackVk::permanentScratchSlotSizes(int maxBatchSize, size_t elemBytes) const {
  vector<VkDeviceSize> slots;
  for(auto& block: blocks)
    mergePermanentScratchSlots(slots, block->permanentScratchSlotSizes(maxBatchSize, elemBytes));
  return slots;
}

std::vector<VkDeviceSize> NestedBottleneckResidualBlockVk::permanentScratchSlotSizes(int maxBatchSize, size_t elemBytes)
  const {
  vector<VkDeviceSize> slots = normActConv1.permanentScratchSlotSizes(maxBatchSize, elemBytes);
  mergePermanentScratchSlots(slots, blocks->permanentScratchSlotSizes(maxBatchSize, elemBytes));
  mergePermanentScratchSlots(slots, normActConv2.permanentScratchSlotSizes(maxBatchSize, elemBytes));
  return slots;
}

// ============================================================================
// TrunkVk
// ============================================================================

std::vector<VkDeviceSize> TrunkVk::permanentScratchSlotSizes(int maxBatchSize, size_t elemBytes) const {
  vector<VkDeviceSize> slots = initialConv->permanentScratchSlotSizes(maxBatchSize, elemBytes);
  mergePermanentScratchSlots(slots, blocks->permanentScratchSlotSizes(maxBatchSize, elemBytes));
  if(trunkTipRMSNorm)
    mergePermanentScratchSlots(slots, trunkTipRMSNorm->permanentScratchSlotSizes(maxBatchSize));
  return slots;
}

TrunkVk::TrunkVk(
  const VulkanLayerContext& ctx,
  const TrunkDesc* desc,
  int nnX,
  int nnY,
  int paddedSpatialSize_,
  bool useFP16)
  : trunkNumChannels(desc->trunkNumChannels),
    trunkNormKind(desc->trunkNormKind),
    paddedSpatialSize(paddedSpatialSize_),
    device(ctx.device) {
  // The model boundary can physically pad an NHWC input channel tail. Allow
  // the initial convolution to select the vec8 implicit-GEMM path; the model
  // queries the selected physical width before allocating/uploading inputs.
  initialConv =
    std::make_unique<ConvLayer>(ctx, &desc->initialConv, nnX, nnY, paddedSpatialSize_, useFP16, false, true);
  initialMatMul = std::make_unique<MatMulLayer>(ctx, &desc->initialMatMul);
  if(desc->metaEncoderVersion > 0) {
    sgfMetadataEncoder = std::make_unique<SGFMetadataEncoderVk>(ctx, &desc->sgfMetadataEncoder);
  }
  blocks = std::make_unique<BlockStackVk>(
    ctx, desc->blocks, desc->numBlocks, desc->trunkNumChannels, nnX, nnY, paddedSpatialSize_, useFP16);
  if(desc->trunkNormKind == TRUNK_NORM_KIND_STANDARD) {
    trunkTipBN = std::make_unique<BatchNormLayer>(ctx, &desc->trunkTipBN, &desc->trunkTipActivation, useFP16);
  } else {
    trunkTipRMSNorm = std::make_unique<RMSNormVk>(
      ctx, &desc->trunkTipRMSNorm, desc->trunkTipActivation.activation, paddedSpatialSize_, useFP16);
  }
  addChannelBiasesNhwcKernel = VulkanKernels::AddChannelBiasesNhwc::build(ctx.device, ctx.pipelineCache, useFP16);
}

void TrunkVk::dispatch(
  const CmdCtx& ctx,
  ScratchBuffers* scratch,
  VulkanBuffer* input,
  VulkanBuffer* inputGlobal,
  VulkanBuffer* inputMeta,
  VulkanBuffer* trunk,
  VulkanBuffer* mask,
  VulkanBuffer* maskSum,
  int maxBatchSize) const {
  // getBufSizeXYFloat: always FP32-sized — initialMatMul and SGFMetadataEncoder
  // always write FP32, and bias application reads the bias as FP32.
  SizedBuf<VulkanBuffer*> trunkScratch(scratch->allocator.get(), scratch->getBufSizeXYFloat(trunkNumChannels));

  // Initial conv: input -> trunk
  initialConv->dispatch(ctx, scratch, input, trunk, maxBatchSize);
  barrierFor(ctx, trunk, "initialConv", "addChannelBiases (global)");

  // Initial MatMul: inputGlobal [N, inGlobalC] -> trunkScratch [N, trunkC]
  initialMatMul->dispatch(ctx, inputGlobal, trunkScratch.buf, maxBatchSize);
  barrierFor(ctx, trunkScratch.buf, "initialMatMul", "addChannelBiases (global)");

  // Add global features via broadcast
  {
    dispatchAddChannelBiasesNhwc(
      ctx, addChannelBiasesNhwcKernel, trunk, trunkScratch.buf, trunkNumChannels, paddedSpatialSize, maxBatchSize);
  }
  barrierFor(ctx, trunk, "addChannelBiases (global)", "sgfMetadata/blocks");

  // SGF metadata encoder (optional). It barriers its own output (trunkScratch) internally.
  if(sgfMetadataEncoder && inputMeta) {
    sgfMetadataEncoder->dispatch(ctx, scratch, inputMeta, trunkScratch.buf, maxBatchSize);
    {
      dispatchAddChannelBiasesNhwc(
        ctx, addChannelBiasesNhwcKernel, trunk, trunkScratch.buf, trunkNumChannels, paddedSpatialSize, maxBatchSize);
    }
    barrierFor(ctx, trunk, "addChannelBiases (meta)", "blocks");
  }

  // Block stack. Each block barriers its own trunk output, so the trunk tip
  // below can read trunk directly.
  blocks->dispatch(ctx, scratch, trunk, trunkScratch.buf, mask, maskSum, maxBatchSize);

  // Trunk tip normalization (in-place on trunk). Each branch barriers its own
  // trunk output for the policy/value heads (RMSNormVk does so internally).
  if(trunkNormKind == TRUNK_NORM_KIND_STANDARD) {
    trunkTipBN->dispatch(ctx, scratch, trunk, trunk, mask, paddedSpatialSize, maxBatchSize);
    barrierFor(ctx, trunk, "trunkTipBN", "policyHead/valueHead");

  } else {
    trunkTipRMSNorm->dispatch(ctx, trunk, trunk, mask, maskSum, scratch, maxBatchSize);
  }
}

// ============================================================================
// PolicyHeadVk
// ============================================================================

std::vector<VkDeviceSize> PolicyHeadVk::permanentScratchSlotSizes(int maxBatchSize, size_t elemBytes) const {
  vector<VkDeviceSize> slots = p1Conv->permanentScratchSlotSizes(maxBatchSize, elemBytes);
  mergePermanentScratchSlots(slots, g1Conv->permanentScratchSlotSizes(maxBatchSize, elemBytes));
  mergePermanentScratchSlots(slots, p2Conv->permanentScratchSlotSizes(maxBatchSize, elemBytes));
  return slots;
}

PolicyHeadVk::PolicyHeadVk(
  const VulkanLayerContext& ctx,
  const PolicyHeadDesc* desc,
  int nnX,
  int nnY,
  int paddedSpatialSize_,
  bool useFP16)
  : modelVersion(desc->modelVersion),
    paddedSpatialSize(paddedSpatialSize_),
    p1Channels(desc->p1Conv.outChannels),
    g1Channels(desc->g1Conv.outChannels),
    p2Channels(desc->p2Conv.outChannels),
    device(ctx.device) {
  p1Conv = std::make_unique<ConvLayer>(ctx, &desc->p1Conv, nnX, nnY, paddedSpatialSize_, useFP16);
  g1Conv = std::make_unique<ConvLayer>(ctx, &desc->g1Conv, nnX, nnY, paddedSpatialSize_, useFP16);
  g1BN = std::make_unique<BatchNormLayer>(ctx, &desc->g1BN, &desc->g1Activation, useFP16);
  gpoolToBiasMul = std::make_unique<MatMulLayer>(ctx, &desc->gpoolToBiasMul);
  p1BN = std::make_unique<BatchNormLayer>(ctx, &desc->p1BN, &desc->p1Activation, useFP16);
  p2Conv = std::make_unique<ConvLayer>(ctx, &desc->p2Conv, nnX, nnY, paddedSpatialSize_, useFP16);
  gpoolToPassMul = std::make_unique<MatMulLayer>(ctx, &desc->gpoolToPassMul);
  if(modelVersion >= 15) {
    gpoolToPassBias = std::make_unique<MatBiasLayer>(ctx, &desc->gpoolToPassBias, &desc->passActivation);
    gpoolToPassMul2 = std::make_unique<MatMulLayer>(ctx, &desc->gpoolToPassMul2);
  }
  gpoolNhwcKernel = VulkanKernels::GPoolReductionNhwc::build(ctx.device, ctx.pipelineCache, useFP16, ctx.tuneParams);
  addChannelBiasesNhwcKernel = VulkanKernels::AddChannelBiasesNhwc::build(ctx.device, ctx.pipelineCache, useFP16);
}

void PolicyHeadVk::dispatch(
  const CmdCtx& ctx,
  ScratchBuffers* scratch,
  VulkanBuffer* trunk,
  VulkanBuffer* mask,
  VulkanBuffer* maskSum,
  VulkanBuffer* policyPass,
  VulkanBuffer* policy,
  int maxBatchSize) const {
  SizedBuf<VulkanBuffer*> p1Out(scratch->allocator.get(), scratch->getBufSizeXY(p1Channels));
  SizedBuf<VulkanBuffer*> gpoolOut(scratch->allocator.get(), scratch->getBufSizeXY(g1Channels));
  SizedBuf<VulkanBuffer*> gpoolConcat(scratch->allocator.get(), scratch->getBufSizeFloat(g1Channels * 3));
  SizedBuf<VulkanBuffer*> gpoolBias(scratch->allocator.get(), scratch->getBufSizeFloat(p1Channels));
  SizedBuf<VulkanBuffer*> p1Pass(scratch->allocator.get(), scratch->getBufSizeFloat(p1Channels));

  p1Conv->dispatch(ctx, scratch, trunk, p1Out.buf, maxBatchSize);
  g1Conv->dispatch(ctx, scratch, trunk, gpoolOut.buf, maxBatchSize);
  barrierFor(ctx, gpoolOut.buf, "g1Conv", "g1BN");

  g1BN->dispatch(ctx, scratch, gpoolOut.buf, gpoolOut.buf, mask, paddedSpatialSize, maxBatchSize);
  barrierFor(ctx, gpoolOut.buf, "g1BN", "pooling");

  VulkanKernels::GPoolReductionNhwc::PC gpoolPc = {g1Channels, paddedSpatialSize};
  VulkanKernels::GPoolReductionNhwc::dispatch(
    ctx, gpoolNhwcKernel, gpoolOut.buf, gpoolConcat.buf, mask, maskSum, gpoolPc, maxBatchSize);

  barrierFor(ctx, gpoolConcat.buf, "pooling", "gpoolToBiasMul/gpoolToPassMul");

  gpoolToBiasMul->dispatch(ctx, gpoolConcat.buf, gpoolBias.buf, maxBatchSize);
  barrierFor(ctx, gpoolBias.buf, "gpoolToBiasMul", "p1 bias application");

  // p1Conv wrote p1Out above; applyConv does not barrier its own output.
  barrierFor(ctx, p1Out.buf, "p1Conv", "addChannelBiases");

  {
    VulkanKernels::AddChannelBiasesNhwc::PC biasPc = {maxBatchSize * p1Channels, p1Channels, paddedSpatialSize};
    VulkanKernels::AddChannelBiasesNhwc::dispatch(ctx, addChannelBiasesNhwcKernel, p1Out.buf, gpoolBias.buf, biasPc);
  }
  barrierFor(ctx, p1Out.buf, "addChannelBiases", "p1BN");

  p1BN->dispatch(ctx, scratch, p1Out.buf, p1Out.buf, mask, paddedSpatialSize, maxBatchSize);
  barrierFor(ctx, p1Out.buf, "p1BN", "p2Conv");

  // p2Conv writes policy; its barrier is the caller's responsibility (Model::apply).
  p2Conv->dispatch(ctx, scratch, p1Out.buf, policy, maxBatchSize);

  if(modelVersion >= 15) {
    gpoolToPassMul->dispatch(ctx, gpoolConcat.buf, p1Pass.buf, maxBatchSize);
    barrierFor(ctx, p1Pass.buf, "gpoolToPassMul", "gpoolToPassBias");

    gpoolToPassBias->dispatch(ctx, p1Pass.buf, maxBatchSize);
    barrierFor(ctx, p1Pass.buf, "gpoolToPassBias", "gpoolToPassMul2");

    // gpoolToPassMul2 writes policyPass; its barrier is the caller's responsibility (Model::apply).
    gpoolToPassMul2->dispatch(ctx, p1Pass.buf, policyPass, maxBatchSize);
  } else {
    // gpoolToPassMul writes policyPass; its barrier is the caller's responsibility (Model::apply).
    gpoolToPassMul->dispatch(ctx, gpoolConcat.buf, policyPass, maxBatchSize);
  }
}

// ============================================================================
// ValueHeadVk
// ============================================================================

std::vector<VkDeviceSize> ValueHeadVk::permanentScratchSlotSizes(int maxBatchSize, size_t elemBytes) const {
  vector<VkDeviceSize> slots = v1Conv->permanentScratchSlotSizes(maxBatchSize, elemBytes);
  if(vOwnershipConv)
    mergePermanentScratchSlots(slots, vOwnershipConv->permanentScratchSlotSizes(maxBatchSize, elemBytes));
  return slots;
}

ValueHeadVk::ValueHeadVk(
  const VulkanLayerContext& ctx,
  const ValueHeadDesc* desc,
  int nnX,
  int nnY,
  int paddedSpatialSize_,
  bool useFP16)
  : paddedSpatialSize(paddedSpatialSize_), v1Channels(desc->v1Conv.outChannels), v2Channels(desc->v2Mul.outChannels) {
  v1Conv = std::make_unique<ConvLayer>(ctx, &desc->v1Conv, nnX, nnY, paddedSpatialSize_, useFP16);
  v1BN = std::make_unique<BatchNormLayer>(ctx, &desc->v1BN, &desc->v1Activation, useFP16);
  v2Mul = std::make_unique<MatMulLayer>(ctx, &desc->v2Mul);
  v2Bias = std::make_unique<MatBiasLayer>(ctx, &desc->v2Bias, &desc->v2Activation);
  v3Mul = std::make_unique<MatMulLayer>(ctx, &desc->v3Mul);
  // v3Bias uses IDENTITY activation
  {
    ActivationLayerDesc identAct;
    identAct.activation = ACTIVATION_IDENTITY;
    v3Bias = std::make_unique<MatBiasLayer>(ctx, &desc->v3Bias, &identAct);
  }
  sv3Mul = std::make_unique<MatMulLayer>(ctx, &desc->sv3Mul);
  {
    ActivationLayerDesc identAct;
    identAct.activation = ACTIVATION_IDENTITY;
    sv3Bias = std::make_unique<MatBiasLayer>(ctx, &desc->sv3Bias, &identAct);
  }
  vOwnershipConv = std::make_unique<ConvLayer>(ctx, &desc->vOwnershipConv, nnX, nnY, paddedSpatialSize_, useFP16);
  device = ctx.device;
  valueHeadPoolNhwcKernel =
    VulkanKernels::ValueHeadPoolNhwc::build(ctx.device, ctx.pipelineCache, useFP16, ctx.tuneParams);
}

void ValueHeadVk::dispatch(
  const CmdCtx& ctx,
  ScratchBuffers* scratch,
  VulkanBuffer* trunk,
  VulkanBuffer* mask,
  VulkanBuffer* maskSum,
  VulkanBuffer* value,
  VulkanBuffer* scoreValue,
  VulkanBuffer* ownership,
  int maxBatchSize) const {
  SizedBuf<VulkanBuffer*> v1Out(scratch->allocator.get(), scratch->getBufSizeXY(v1Channels));
  SizedBuf<VulkanBuffer*> v1Mean(scratch->allocator.get(), scratch->getBufSizeFloat(v1Channels * 3));
  SizedBuf<VulkanBuffer*> v2Out(scratch->allocator.get(), scratch->getBufSizeFloat(v2Channels));

  v1Conv->dispatch(ctx, scratch, trunk, v1Out.buf, maxBatchSize);
  barrierFor(ctx, v1Out.buf, "v1Conv", "v1BN");

  v1BN->dispatch(ctx, scratch, v1Out.buf, v1Out.buf, mask, paddedSpatialSize, maxBatchSize);
  barrierFor(ctx, v1Out.buf, "v1BN", "valueHeadPool and vOwnershipConv");

  VulkanKernels::ValueHeadPoolNhwc::PC pc = {v1Channels, paddedSpatialSize};
  VulkanKernels::ValueHeadPoolNhwc::dispatch(
    ctx, valueHeadPoolNhwcKernel, v1Out.buf, v1Mean.buf, maskSum, pc, maxBatchSize);

  barrierFor(ctx, v1Mean.buf, "valueHeadPool", "v2Mul");

  v2Mul->dispatch(ctx, v1Mean.buf, v2Out.buf, maxBatchSize);
  barrierFor(ctx, v2Out.buf, "v2Mul", "v2Bias");

  v2Bias->dispatch(ctx, v2Out.buf, maxBatchSize);
  barrierFor(ctx, v2Out.buf, "v2Bias", "v3Mul and sv3Mul");

  v3Mul->dispatch(ctx, v2Out.buf, value, maxBatchSize);
  barrierFor(ctx, value, "v3Mul", "v3Bias (in-place)");

  // v3Bias writes value; its output barrier (for download) is the caller's responsibility (Model/getOutput).
  v3Bias->dispatch(ctx, value, maxBatchSize);

  sv3Mul->dispatch(ctx, v2Out.buf, scoreValue, maxBatchSize);
  barrierFor(ctx, scoreValue, "sv3Mul", "sv3Bias (in-place)");

  // sv3Bias writes scoreValue; its output barrier (for download) is the caller's responsibility (Model/getOutput).
  sv3Bias->dispatch(ctx, scoreValue, maxBatchSize);

  // vOwnershipConv reads v1Out (still live) and writes ownership; ownership's
  // output barrier (for download) is the caller's responsibility (Model/getOutput).
  vOwnershipConv->dispatch(ctx, scratch, v1Out.buf, ownership, maxBatchSize);
}

#endif  // USE_VULKAN_BACKEND

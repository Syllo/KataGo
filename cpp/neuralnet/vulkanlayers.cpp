#ifdef USE_VULKAN_BACKEND

#include "../neuralnet/vulkanlayers.h"
#include <algorithm>
#include <cmath>
#include <memory>
#include <numeric>
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
  : name(desc->name),
    mul1(ctx, &desc->mul1),
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
  // mark output buf1 of mul1 input dependence for bias1
  cmdComputeBarrier(ctx.cmd, buf1.buf->buffer);
  bias1.dispatch(ctx, buf1.buf, maxBatchSize);
  // mark output buf1 of bias1 input dependence for mul2
  cmdComputeBarrier(ctx.cmd, buf1.buf->buffer);
  mul2.dispatch(ctx, buf1.buf, buf2.buf, maxBatchSize);
  // mark output buf2 of mul2 input dependence for bias2
  cmdComputeBarrier(ctx.cmd, buf2.buf->buffer);
  bias2.dispatch(ctx, buf2.buf, maxBatchSize);
  // mark output buf2 of bias2 input dependence for mul3
  cmdComputeBarrier(ctx.cmd, buf2.buf->buffer);
  mul3.dispatch(ctx, buf2.buf, output, maxBatchSize);
  // mark output output of mul3 input dependence for the consumer (addChannelBiases)
  cmdComputeBarrier(ctx.cmd, output->buffer);
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
  : name(desc->name),
    numChannels(desc->numChannels),
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
    siluKernel = VulkanKernels::ScaleBiasMaskAct::build(ctx.device, ctx.pipelineCache, useFP16, ACTIVATION_SILU);
  }
  const bool canUseSubgroupShuffle = ctx.supportsSubgroupShuffleCompute && ctx.subgroupSize > 0;
  const bool useTransformerRMSNormSubgroup =
    canUseSubgroupShuffle && ctx.subgroupSize >= 8 && (ctx.subgroupSize % 8) == 0;
  const bool useSpatialRMSNormPass2Subgroup = canUseSubgroupShuffle;
  // RMSNorm path-specific kernels: build only the set that this instance dispatches.
  if(!spatial) {
    rmsNormKernel =
      VulkanKernels::TransformerRMSNorm::build(ctx.device, ctx.pipelineCache, useFP16, useTransformerRMSNormSubgroup);
  } else {
    // Three-pass spatial path.
    spatialKernels = VulkanKernels::SpatialRMSNorm::build(
      ctx.device, ctx.pipelineCache, useFP16, ctx.tuneParams.spatialRMSNormTile, useSpatialRMSNormPass2Subgroup);
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
    VulkanKernels::TransformerRMSNorm::PC pc = {numChannels, paddedSpatialSize, epsilon};
    VulkanKernels::TransformerRMSNorm::dispatch(
      ctx, rmsNormKernel, input, output, gammaBuf.get(), betaBuf.get(), mask, pc, maxBatchSize);
  } else {
    // Spatial 3-pass RMSNorm — uses permanent scratch slots 0 (pass1 partials) and 1 (pass2 scalar)
    VulkanBuffer* ws0 = scratch->permanentSlot(0);
    VulkanBuffer* ws1 = scratch->permanentSlot(1);
    VulkanKernels::SpatialRMSNorm::PC pc = {numChannels, paddedSpatialSize, epsilon};
    VulkanKernels::SpatialRMSNorm::dispatch(
      ctx, spatialKernels, input, output, gammaBuf.get(), betaBuf.get(), mask, maskSum, ws0, ws1, pc, maxBatchSize);

    // WAR: release permanent scratch slots — next user may write them immediately
    cmdComputeWARBarrier(ctx.cmd, ws0->buffer);
    cmdComputeWARBarrier(ctx.cmd, ws1->buffer);
  }
  // Apply SILU activation in-place on output if needed
  if(activation == ACTIVATION_SILU) {
    // The RMSNorm pass above wrote output; make it visible before the in-place SILU reads it.
    // mark output output of RMSNorm input dependence for the in-place SILU activation
    cmdComputeBarrier(ctx.cmd, output->buffer);
    VulkanKernels::ScaleBiasMaskAct::PC pc = {numChannels, paddedSpatialSize, maxBatchSize};
    VulkanKernels::ScaleBiasMaskAct::dispatch(
      ctx, siluKernel, output, output, actScaleBuf.get(), actBiasBuf.get(), mask, pc);
  }
  // mark output output of RMSNorm/SILU input dependence for the next consumer
  cmdComputeBarrier(ctx.cmd, output->buffer);
}

// ============================================================================
// TransformerRMSNormLayerVk
// ============================================================================

TransformerRMSNormLayerVk::TransformerRMSNormLayerVk(
  const VulkanLayerContext& ctx,
  const TransformerRMSNormDesc* desc,
  int paddedSpatialSize_,
  bool useFP16)
  : name(desc->name),
    numChannels(desc->numChannels),
    epsilon(desc->epsilon),
    paddedSpatialSize(paddedSpatialSize_),
    device(ctx.device) {
  weightBuf = makeWeightBuf(ctx.device, ctx.queue, ctx.queueMutex, ctx.commandPool, ctx.memProps, desc->weight, false);
  vector<float> zeros(numChannels, 0.0f);
  zeroBetaBuf = makeWeightBuf(ctx.device, ctx.queue, ctx.queueMutex, ctx.commandPool, ctx.memProps, zeros, false);
  const bool useTransformerRMSNormSubgroup =
    ctx.supportsSubgroupShuffleCompute && ctx.subgroupSize >= 8 && (ctx.subgroupSize % 8) == 0;
  rmsNormKernel =
    VulkanKernels::TransformerRMSNorm::build(ctx.device, ctx.pipelineCache, useFP16, useTransformerRMSNormSubgroup);
}

void TransformerRMSNormLayerVk::dispatch(
  const CmdCtx& ctx,
  VulkanBuffer* input,
  VulkanBuffer* output,
  VulkanBuffer* mask,
  int maxBatchSize) const {
  VulkanKernels::TransformerRMSNorm::PC pc = {numChannels, paddedSpatialSize, epsilon};
  VulkanKernels::TransformerRMSNorm::dispatch(
    ctx, rmsNormKernel, input, output, weightBuf.get(), zeroBetaBuf.get(), mask, pc, maxBatchSize);
}

// ============================================================================
// TransformerMatMulLayerVk
// ============================================================================

TransformerMatMulLayerVk::TransformerMatMulLayerVk(
  const VulkanLayerContext& ctx,
  const MatMulLayerDesc* desc,
  int paddedSpatialSize_,
  bool useFP16,
  bool addToOutput_)
  : name(desc->name),
    inChannels(desc->inChannels),
    outChannels(desc->outChannels),
    outChannelsPadded(
      // Pad outC to the lcm of every enabled strided variant's requirement
      // (see mergedStridedPaddingContract). K stays real (previous layer's outC).
      roundUpToMultipleInt(desc->outChannels, mergedStridedPaddingContract(ctx, useFP16).n)),
    paddedSpatialSize(paddedSpatialSize_),
    addToOutput(addToOutput_),
    device(ctx.device),
    gemmStridedVariant(GemmStridedVariant::Tiled),
    gemmStridedBStride(0) {
  gemmStridedVariant = resolveGemmStridedVariant(ctx, useFP16);
  // Zero-fill padding so kernels that read the padded region see zeros.
  std::vector<float> paddedW(inChannels * outChannelsPadded, 0.0f);
  for(int ic = 0; ic < inChannels; ic++)
    for(int oc = 0; oc < outChannels; oc++)
      paddedW[ic * outChannelsPadded + oc] = desc->weights[ic * outChannels + oc];
  if(gemmStridedVariantUsesPackedB(gemmStridedVariant)) {
    const int packedBN = gemmStridedPackedBBN(ctx.tuneParams, gemmStridedVariant);
    const int packedBK = gemmStridedPackedBBK(ctx.tuneParams, gemmStridedVariant);
    const int packedPadScalars = gemmStridedPackedBPadScalars(gemmStridedVariant);
    std::vector<float> packedW =
      packStridedGemmBWeights(paddedW, 1, outChannelsPadded, inChannels, packedBN, packedBK, packedPadScalars);
    filterBuf = makeWeightBuf(ctx.device, ctx.queue, ctx.queueMutex, ctx.commandPool, ctx.memProps, packedW, useFP16);
  } else {
    filterBuf = makeWeightBuf(ctx.device, ctx.queue, ctx.queueMutex, ctx.commandPool, ctx.memProps, paddedW, useFP16);
  }
  if(gemmStridedVariant == GemmStridedVariant::Coopmat2AccF16) {
    const auto cm2Tile = VulkanKernels::GemmStridedCoopmat2AccF16::tileAlignmentReq(ctx.tuneParams);
    int32_t aligned = (paddedSpatialSize % cm2Tile.m == 0 && outChannelsPadded % cm2Tile.n == 0) ? 1 : 0;
    int32_t kAligned = (inChannels % cm2Tile.k == 0) ? 1 : 0;
    gemmStridedKernel = VulkanKernels::GemmStridedCoopmat2AccF16::build(
      ctx.device,
      ctx.pipelineCache,
      ctx.tuneParams.stridedCoopmat2AccF16BlockSize,
      ctx.tuneParams.stridedCoopmat2AccF16BM,
      ctx.tuneParams.stridedCoopmat2AccF16BN,
      ctx.tuneParams.stridedCoopmat2AccF16BK,
      aligned,
      1,
      kAligned,
      inChannels,
      addToOutput);
  } else if(gemmStridedVariant == GemmStridedVariant::Coopmat2) {
    const auto cm2Tile = VulkanKernels::GemmStridedCoopmat2::tileAlignmentReq(ctx.tuneParams);
    int32_t aligned = (paddedSpatialSize % cm2Tile.m == 0 && outChannelsPadded % cm2Tile.n == 0) ? 1 : 0;
    int32_t kAligned = (inChannels % cm2Tile.k == 0) ? 1 : 0;
    gemmStridedKernel = VulkanKernels::GemmStridedCoopmat2::build(
      ctx.device,
      ctx.pipelineCache,
      ctx.tuneParams.stridedCoopmat2BlockSize,
      ctx.tuneParams.stridedCoopmat2BM,
      ctx.tuneParams.stridedCoopmat2BN,
      ctx.tuneParams.stridedCoopmat2BK,
      aligned,
      1,
      kAligned,
      inChannels,
      addToOutput);
  } else if(gemmStridedVariant == GemmStridedVariant::CoopmatAccF16) {
    const auto cmTile = VulkanKernels::GemmStridedCoopmatAccF16::tileAlignmentReq(ctx.tuneParams);
    int32_t aligned = (paddedSpatialSize % cmTile.m == 0 && outChannelsPadded % cmTile.n == 0) ? 1 : 0;
    int32_t kAligned = (inChannels % cmTile.k == 0) ? 1 : 0;
    gemmStridedKernel = VulkanKernels::GemmStridedCoopmatAccF16::build(
      ctx.device,
      ctx.pipelineCache,
      ctx.tuneParams.stridedCoopmatAccF16BlockSize,
      ctx.tuneParams.stridedCoopmatAccF16BM,
      ctx.tuneParams.stridedCoopmatAccF16BN,
      ctx.tuneParams.stridedCoopmatAccF16BK,
      ctx.tuneParams.stridedCoopmatAccF16WM,
      ctx.tuneParams.stridedCoopmatAccF16WN,
      ctx.tuneParams.stridedCoopmatAccF16TM,
      ctx.tuneParams.stridedCoopmatAccF16TN,
      ctx.tuneParams.stridedCoopmatAccF16TK,
      ctx.tuneParams.stridedCoopmatAccF16Warp,
      aligned,
      1,
      kAligned,
      inChannels,
      ctx.subgroupSize,
      addToOutput);
  } else if(gemmStridedVariant == GemmStridedVariant::Coopmat) {
    const auto cmTile = VulkanKernels::GemmStridedCoopmat::tileAlignmentReq(ctx.tuneParams);
    int32_t aligned = (paddedSpatialSize % cmTile.m == 0 && outChannelsPadded % cmTile.n == 0) ? 1 : 0;
    int32_t kAligned = (inChannels % cmTile.k == 0) ? 1 : 0;
    gemmStridedKernel = VulkanKernels::GemmStridedCoopmat::build(
      ctx.device,
      ctx.pipelineCache,
      ctx.tuneParams.stridedCoopmatBlockSize,
      ctx.tuneParams.stridedCoopmatBM,
      ctx.tuneParams.stridedCoopmatBN,
      ctx.tuneParams.stridedCoopmatBK,
      ctx.tuneParams.stridedCoopmatWM,
      ctx.tuneParams.stridedCoopmatWN,
      ctx.tuneParams.stridedCoopmatTM,
      ctx.tuneParams.stridedCoopmatTN,
      ctx.tuneParams.stridedCoopmatTK,
      ctx.tuneParams.stridedCoopmatWarp,
      aligned,
      1,
      kAligned,
      inChannels,
      ctx.subgroupSize,
      addToOutput);
  } else if(gemmStridedVariant == GemmStridedVariant::Dot2) {
    // Strided path: N (outChannelsPadded) is DOT2-aligned by
    // mergedStridedPaddingContract; M (paddedSpatialSize) is globally pinned
    // to VULKAN_SPATIAL_ALIGN and NOT per-variant re-padded; K stays real
    // (kPaddable:false). ALIGNED signals M/N safety only; K_ALIGNED handles
    // the K-tail guard.
    const auto dot2Tile = VulkanKernels::GemmStridedDot2::tileAlignmentReq(ctx.tuneParams);
    int32_t aligned = (paddedSpatialSize % dot2Tile.m == 0 && outChannelsPadded % dot2Tile.n == 0) ? 1 : 0;
    int32_t kAligned = (inChannels % dot2Tile.k == 0) ? 1 : 0;
    gemmStridedKernel = VulkanKernels::GemmStridedDot2::build(
      ctx.device,
      ctx.pipelineCache,
      ctx.tuneParams.stridedDot2BlockSize,
      ctx.tuneParams.stridedDot2BM,
      ctx.tuneParams.stridedDot2BN,
      ctx.tuneParams.stridedDot2WM,
      ctx.tuneParams.stridedDot2WN,
      ctx.tuneParams.stridedDot2WMIter,
      ctx.tuneParams.stridedDot2TM,
      ctx.tuneParams.stridedDot2TN,
      ctx.tuneParams.stridedDot2Warp,
      aligned,
      1,
      kAligned,
      inChannels,
      addToOutput);
  } else if(gemmStridedVariant == GemmStridedVariant::Dot2AccF16) {
    const auto dot2Tile = VulkanKernels::GemmStridedDot2AccF16::tileAlignmentReq(ctx.tuneParams);
    int32_t aligned = (paddedSpatialSize % dot2Tile.m == 0 && outChannelsPadded % dot2Tile.n == 0) ? 1 : 0;
    int32_t kAligned = (inChannels % dot2Tile.k == 0) ? 1 : 0;
    gemmStridedKernel = VulkanKernels::GemmStridedDot2AccF16::build(
      ctx.device,
      ctx.pipelineCache,
      ctx.tuneParams.stridedDot2AccF16BlockSize,
      ctx.tuneParams.stridedDot2AccF16BM,
      ctx.tuneParams.stridedDot2AccF16BN,
      ctx.tuneParams.stridedDot2AccF16WM,
      ctx.tuneParams.stridedDot2AccF16WN,
      ctx.tuneParams.stridedDot2AccF16WMIter,
      ctx.tuneParams.stridedDot2AccF16TM,
      ctx.tuneParams.stridedDot2AccF16TN,
      ctx.tuneParams.stridedDot2AccF16Warp,
      aligned,
      1,
      kAligned,
      inChannels,
      addToOutput);
  } else {
    gemmStridedKernel = VulkanKernels::GemmStridedTiled::build(
      ctx.device,
      ctx.pipelineCache,
      useFP16,
      ctx.tuneParams.gemmStridedTiledLocalSizeX,
      ctx.tuneParams.gemmStridedTiledLocalSizeY,
      ctx.tuneParams.gemmStridedTiledTileK,
      ctx.tuneParams.gemmStridedTiledRN,
      inChannels,
      addToOutput);
  }
}

void TransformerMatMulLayerVk::dispatch(const CmdCtx& ctx, VulkanBuffer* input, VulkanBuffer* output, int maxBatchSize)
  const {
  VulkanKernels::GemmStridedTiled::PC pc = {
    paddedSpatialSize,
    outChannelsPadded,
    outChannels,
    inChannels * (paddedSpatialSize / 4),
    gemmStridedBStride,
    outChannels * (paddedSpatialSize / 4)};  // strideC uses real N
  if(gemmStridedVariant == GemmStridedVariant::Coopmat2AccF16) {
    VulkanKernels::GemmStridedCoopmat2AccF16::dispatch(
      ctx, gemmStridedKernel, input, filterBuf.get(), output, inChannels, pc, maxBatchSize);
  } else if(gemmStridedVariant == GemmStridedVariant::Coopmat2) {
    VulkanKernels::GemmStridedCoopmat2::dispatch(
      ctx, gemmStridedKernel, input, filterBuf.get(), output, inChannels, pc, maxBatchSize);
  } else if(gemmStridedVariant == GemmStridedVariant::CoopmatAccF16) {
    VulkanKernels::GemmStridedCoopmatAccF16::dispatch(
      ctx, gemmStridedKernel, input, filterBuf.get(), output, inChannels, pc, maxBatchSize);
  } else if(gemmStridedVariant == GemmStridedVariant::Coopmat) {
    VulkanKernels::GemmStridedCoopmat::dispatch(
      ctx, gemmStridedKernel, input, filterBuf.get(), output, inChannels, pc, maxBatchSize);
  } else if(gemmStridedVariant == GemmStridedVariant::Dot2) {
    VulkanKernels::GemmStridedDot2::dispatch(
      ctx, gemmStridedKernel, input, filterBuf.get(), output, inChannels, pc, maxBatchSize);
  } else if(gemmStridedVariant == GemmStridedVariant::Dot2AccF16) {
    VulkanKernels::GemmStridedDot2AccF16::dispatch(
      ctx, gemmStridedKernel, input, filterBuf.get(), output, inChannels, pc, maxBatchSize);
  } else {
    VulkanKernels::GemmStridedTiled::dispatch(
      ctx, gemmStridedKernel, input, filterBuf.get(), output, inChannels, pc, maxBatchSize);
  }
}

// ============================================================================
// TransformerAttentionBlockVk
// ============================================================================

TransformerAttentionBlockVk::TransformerAttentionBlockVk(
  const VulkanLayerContext& ctx,
  const TransformerAttentionDesc* desc,
  int nnX,
  int nnY,
  int paddedSpatialSize_,
  bool useFP16)
  : name(desc->name),
    numHeads(desc->numHeads),
    numKVHeads(desc->numKVHeads),
    qHeadDim(desc->qHeadDim),
    vHeadDim(desc->vHeadDim),
    useRope(desc->useRope),
    learnableRope(desc->learnableRope),
    inChannels(desc->qProj.inChannels),
    paddedSpatialSize(paddedSpatialSize_),
    preLN(ctx, &desc->preLN, paddedSpatialSize_, useFP16),
    qProj(ctx, &desc->qProj, paddedSpatialSize_, useFP16),
    kProj(ctx, &desc->kProj, paddedSpatialSize_, useFP16),
    vProj(ctx, &desc->vProj, paddedSpatialSize_, useFP16),
    outProj(ctx, &desc->outProj, paddedSpatialSize_, useFP16, true),
    device(ctx.device) {
  testAssert(numKVHeads > 0 && numHeads % numKVHeads == 0);
  testAssert((int)ctx.tuneParams.attnBlockKV <= (int)ctx.tuneParams.attnBlockQ);
  if(useRope) {
    vector<float> cosVec, sinVec;
    desc->computeRopeCosSin(nnX, nnY, paddedSpatialSize_, cosVec, sinVec);
    ropeCosTable = makeWeightBuf(ctx.device, ctx.queue, ctx.queueMutex, ctx.commandPool, ctx.memProps, cosVec, false);
    ropeSinTable = makeWeightBuf(ctx.device, ctx.queue, ctx.queueMutex, ctx.commandPool, ctx.memProps, sinVec, false);
  }
  // Tiled flash-attention specialised for this block's (seqLen, qHeadDim, vHeadDim).
  attentionKernel = VulkanKernels::AttentionTiled::build(
    ctx.device,
    ctx.pipelineCache,
    useFP16,
    (int)ctx.tuneParams.attnBlockQ,
    (int)ctx.tuneParams.attnBlockKV,
    (int)ctx.tuneParams.attnQPerThread,
    paddedSpatialSize_,
    qHeadDim,
    vHeadDim,
    useRope,
    learnableRope);
}

void TransformerAttentionBlockVk::dispatch(
  const CmdCtx& ctx,
  ScratchBuffers* scratch,
  VulkanBuffer* trunk,
  VulkanBuffer* trunkScratch,
  VulkanBuffer* mask,
  int maxBatchSize) const {
  int seqLen = paddedSpatialSize;
  int qTotalDim = numHeads * qHeadDim;
  int kTotalDim = numKVHeads * qHeadDim;
  int vTotalDim = numKVHeads * vHeadDim;

  // Step 1: RMSNorm trunk -> trunkScratch
  preLN.dispatch(ctx, trunk, trunkScratch, mask, maxBatchSize);
  // mark output trunkScratch of preLN input dependence for qProj/kProj/vProj
  cmdComputeBarrier(ctx.cmd, trunkScratch->buffer);

  // Step 2: Q/K/V projections
  SizedBuf<VulkanBuffer*> qBuf(scratch->allocator.get(), scratch->getBufSizeXY(qTotalDim));
  SizedBuf<VulkanBuffer*> kBuf(scratch->allocator.get(), scratch->getBufSizeXY(kTotalDim));
  SizedBuf<VulkanBuffer*> vBuf(scratch->allocator.get(), scratch->getBufSizeXY(vTotalDim));

  qProj.dispatch(ctx, trunkScratch, qBuf.buf, maxBatchSize);
  kProj.dispatch(ctx, trunkScratch, kBuf.buf, maxBatchSize);
  vProj.dispatch(ctx, trunkScratch, vBuf.buf, maxBatchSize);
  // mark output qBuf of qProj input dependence for RoPE/attention
  cmdComputeBarrier(ctx.cmd, qBuf.buf->buffer);
  // mark output kBuf of kProj input dependence for RoPE/attention
  cmdComputeBarrier(ctx.cmd, kBuf.buf->buffer);
  // mark output vBuf of vProj input dependence for attention
  cmdComputeBarrier(ctx.cmd, vBuf.buf->buffer);

  // Step 3: Scaled dot-product attention (tiled flash-attention; the per-block
  // ComputeKernel was specialised for this block's seqLen/qHeadDim/vHeadDim at ctor.)
  SizedBuf<VulkanBuffer*> attnOut(scratch->allocator.get(), scratch->getBufSizeXY(numHeads * vHeadDim));
  float scale = 1.0f / sqrtf((float)qHeadDim);
  VulkanKernels::AttentionTiled::PC pcAttn = {numHeads, numKVHeads, scale, maxBatchSize * numHeads};
  VulkanKernels::AttentionTiled::dispatch(
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
  // mark output attnOut of attention input dependence for outProj
  cmdComputeBarrier(ctx.cmd, attnOut.buf->buffer);

  // Step 4: Output projection + residual add into trunk.
  cmdComputeWARBarrier(ctx.cmd, trunk->buffer);
  outProj.dispatch(ctx, attnOut.buf, trunk, maxBatchSize);
  // mark output trunk of outProj(residual) input dependence for the next block / trunk consumer
  cmdComputeBarrier(ctx.cmd, trunk->buffer);
}

// ============================================================================
// TransformerFFNBlockVk
// ============================================================================

TransformerFFNBlockVk::TransformerFFNBlockVk(
  const VulkanLayerContext& ctx,
  const TransformerFFNDesc* desc,
  int paddedSpatialSize_,
  bool useFP16)
  : name(desc->name),
    numChannels(desc->numChannels),
    ffnChannels(desc->ffnChannels),
    paddedSpatialSize(paddedSpatialSize_),
    preLN(ctx, &desc->preLN, paddedSpatialSize_, useFP16),
    linear1(ctx, &desc->linear1, paddedSpatialSize_, useFP16),
    linear2(ctx, &desc->linear2, paddedSpatialSize_, useFP16, true),
    device(ctx.device) {
  if(!desc->useSwiGLU)
    throw StringError("Vulkan backend: non-SwiGLU transformer FFN not supported");
  linearGate = std::make_unique<TransformerMatMulLayerVk>(ctx, &desc->linearGate, paddedSpatialSize_, useFP16);
  swiGLUKernel = VulkanKernels::SwiGLU::build(ctx.device, ctx.pipelineCache, useFP16);
}

void TransformerFFNBlockVk::dispatch(
  const CmdCtx& ctx,
  ScratchBuffers* scratch,
  VulkanBuffer* trunk,
  VulkanBuffer* trunkScratch,
  VulkanBuffer* mask,
  int maxBatchSize) const {
  // Step 1: RMSNorm
  preLN.dispatch(ctx, trunk, trunkScratch, mask, maxBatchSize);
  // mark output trunkScratch of preLN input dependence for linear1/linearGate
  cmdComputeBarrier(ctx.cmd, trunkScratch->buffer);

  // Step 2: linear1 + gate projections
  SizedBuf<VulkanBuffer*> ffnBuf(scratch->allocator.get(), scratch->getBufSizeXY(ffnChannels));
  SizedBuf<VulkanBuffer*> gateBuf(scratch->allocator.get(), scratch->getBufSizeXY(ffnChannels));
  linear1.dispatch(ctx, trunkScratch, ffnBuf.buf, maxBatchSize);
  linearGate->dispatch(ctx, trunkScratch, gateBuf.buf, maxBatchSize);
  // mark output ffnBuf of linear1 input dependence for SwiGLU
  cmdComputeBarrier(ctx.cmd, ffnBuf.buf->buffer);
  // mark output gateBuf of linearGate input dependence for SwiGLU
  cmdComputeBarrier(ctx.cmd, gateBuf.buf->buffer);

  // Step 3: SwiGLU in-place into ffnBuf
  int totalSize = maxBatchSize * ffnChannels * paddedSpatialSize;
  VulkanKernels::SwiGLU::PC pc = {totalSize};
  VulkanKernels::SwiGLU::dispatch(ctx, swiGLUKernel, ffnBuf.buf, gateBuf.buf, ffnBuf.buf, pc);
  // mark output ffnBuf of SwiGLU input dependence for linear2
  cmdComputeBarrier(ctx.cmd, ffnBuf.buf->buffer);

  // Step 4: linear2 + residual add into trunk.
  cmdComputeWARBarrier(ctx.cmd, trunk->buffer);
  linear2.dispatch(ctx, ffnBuf.buf, trunk, maxBatchSize);
  // mark output trunk of linear2(residual) input dependence for the next block / trunk consumer
  cmdComputeBarrier(ctx.cmd, trunk->buffer);
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
  : name(desc->name),
    normActConv1(ctx, &desc->preBN, &desc->preActivation, &desc->preConv, nnX, nnY, paddedSpatialSize_, useFP16),
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
  // mark output mid of normActConv1 input dependence for blocks(BlockStack)
  cmdComputeBarrier(ctx.cmd, mid.buf->buffer);
  blocks->dispatch(ctx, scratch, mid.buf, midScratch.buf, mask, maskSum, maxBatchSize);
  // mark output mid of blocks input dependence for normActConv2
  cmdComputeBarrier(ctx.cmd, mid.buf->buffer);
  if(normActConv2.fusesResidualAdd) {
    cmdComputeWARBarrier(ctx.cmd, trunk->buffer);
    normActConv2.dispatch(ctx, mid.buf, mid.buf, trunk, mask, scratch, maxBatchSize);
  } else {
    normActConv2.dispatch(ctx, mid.buf, mid.buf, trunkScratch, mask, scratch, maxBatchSize);
    // mark output trunkScratch of normActConv2 input dependence for addPointwise(residual)
    cmdComputeBarrier(ctx.cmd, trunkScratch->buffer);
    // Add residual.
    int totalElts = maxBatchSize * normActConv2.outChannels * paddedSpatialSize;
    {
      VulkanKernels::AddPointwise::PC pc = {totalElts};
      VulkanKernels::AddPointwise::dispatch(ctx, addPointwiseKernel, trunk, trunkScratch, pc);
    }
  }
  // mark output trunk of addPointwise(residual) input dependence for the next block / trunk consumer
  cmdComputeBarrier(ctx.cmd, trunk->buffer);
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
  : name(desc->name),
    modelVersion(desc->modelVersion),
    trunkNumChannels(desc->trunkNumChannels),
    trunkNormKind(desc->trunkNormKind),
    paddedSpatialSize(paddedSpatialSize_),
    device(ctx.device) {
  initialConv = std::make_unique<ConvLayer>(ctx, &desc->initialConv, nnX, nnY, paddedSpatialSize_, useFP16);
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
  addChannelBiasesKernel = VulkanKernels::AddChannelBiases::build(ctx.device, ctx.pipelineCache, useFP16);
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
  // always write FP32, and addChannelBiasesNCHW reads the bias as FP32.
  SizedBuf<VulkanBuffer*> trunkScratch(scratch->allocator.get(), scratch->getBufSizeXYFloat(trunkNumChannels));

  // Initial conv: input -> trunk
  initialConv->dispatch(ctx, scratch, input, trunk, maxBatchSize);
  // mark output trunk of initialConv input dependence for addChannelBiases(global)
  cmdComputeBarrier(ctx.cmd, trunk->buffer);

  // Initial MatMul: inputGlobal [N, inGlobalC] -> trunkScratch [N, trunkC]
  initialMatMul->dispatch(ctx, inputGlobal, trunkScratch.buf, maxBatchSize);
  // mark output trunkScratch of initialMatMul input dependence for addChannelBiases(global)
  cmdComputeBarrier(ctx.cmd, trunkScratch.buf->buffer);

  // Add global features via broadcast
  {
    VulkanKernels::AddChannelBiases::PC pc = {maxBatchSize * trunkNumChannels, paddedSpatialSize};
    VulkanKernels::AddChannelBiases::dispatch(ctx, addChannelBiasesKernel, trunk, trunkScratch.buf, pc);
  }
  // mark output trunk of addChannelBiases(global) input dependence for sgfMetadata/blocks
  cmdComputeBarrier(ctx.cmd, trunk->buffer);

  // SGF metadata encoder (optional). It barriers its own output (trunkScratch) internally.
  if(sgfMetadataEncoder && inputMeta) {
    sgfMetadataEncoder->dispatch(ctx, scratch, inputMeta, trunkScratch.buf, maxBatchSize);
    {
      VulkanKernels::AddChannelBiases::PC pc = {maxBatchSize * trunkNumChannels, paddedSpatialSize};
      VulkanKernels::AddChannelBiases::dispatch(ctx, addChannelBiasesKernel, trunk, trunkScratch.buf, pc);
    }
    // mark output trunk of addChannelBiases(meta) input dependence for blocks
    cmdComputeBarrier(ctx.cmd, trunk->buffer);
  }

  // Block stack. Each block barriers its own trunk output, so the trunk tip
  // below can read trunk directly.
  blocks->dispatch(ctx, scratch, trunk, trunkScratch.buf, mask, maskSum, maxBatchSize);

  // Trunk tip normalization (in-place on trunk). Each branch barriers its own
  // trunk output for the policy/value heads (RMSNormVk does so internally).
  if(trunkNormKind == TRUNK_NORM_KIND_STANDARD) {
    trunkTipBN->dispatch(ctx, trunk, trunk, mask, paddedSpatialSize, maxBatchSize);
    // mark output trunk of trunkTipBN input dependence for policyHead/valueHead
    cmdComputeBarrier(ctx.cmd, trunk->buffer);
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
  : name(desc->name),
    modelVersion(desc->modelVersion),
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
  gpoolKernel =
    VulkanKernels::GPoolReduction::build(ctx.device, ctx.pipelineCache, useFP16, ctx.tuneParams.gpoolXystride);
  addChannelBiasesKernel = VulkanKernels::AddChannelBiases::build(ctx.device, ctx.pipelineCache, useFP16);
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
  // mark output gpoolOut of g1Conv input dependence for g1BN
  cmdComputeBarrier(ctx.cmd, gpoolOut.buf->buffer);
  g1BN->dispatch(ctx, gpoolOut.buf, gpoolOut.buf, mask, paddedSpatialSize, maxBatchSize);
  // mark output gpoolOut of g1BN input dependence for gpoolNCHW(pool)
  cmdComputeBarrier(ctx.cmd, gpoolOut.buf->buffer);

  {
    VulkanKernels::GPoolReduction::PC pc = {g1Channels, paddedSpatialSize};
    VulkanKernels::GPoolReduction::dispatch(
      ctx, gpoolKernel, gpoolOut.buf, gpoolConcat.buf, mask, maskSum, pc, maxBatchSize);
  }
  // mark output gpoolConcat of gpoolNCHW input dependence for gpoolToBiasMul/gpoolToPassMul
  cmdComputeBarrier(ctx.cmd, gpoolConcat.buf->buffer);

  gpoolToBiasMul->dispatch(ctx, gpoolConcat.buf, gpoolBias.buf, maxBatchSize);
  // mark output gpoolBias of gpoolToBiasMul input dependence for p1 bias application
  cmdComputeBarrier(ctx.cmd, gpoolBias.buf->buffer);
  // p1Conv wrote p1Out above; applyConv does not barrier its own output.
  // mark output p1Out of p1Conv input dependence for addChannelBiases
  cmdComputeBarrier(ctx.cmd, p1Out.buf->buffer);
  {
    VulkanKernels::AddChannelBiases::PC pc = {maxBatchSize * p1Channels, paddedSpatialSize};
    VulkanKernels::AddChannelBiases::dispatch(ctx, addChannelBiasesKernel, p1Out.buf, gpoolBias.buf, pc);
  }
  // mark output p1Out of addChannelBiases input dependence for p1BN
  cmdComputeBarrier(ctx.cmd, p1Out.buf->buffer);
  p1BN->dispatch(ctx, p1Out.buf, p1Out.buf, mask, paddedSpatialSize, maxBatchSize);
  // mark output p1Out of p1BN input dependence for p2Conv
  cmdComputeBarrier(ctx.cmd, p1Out.buf->buffer);
  // p2Conv writes policy; its barrier is the caller's responsibility (Model::apply).
  p2Conv->dispatch(ctx, scratch, p1Out.buf, policy, maxBatchSize);

  if(modelVersion >= 15) {
    gpoolToPassMul->dispatch(ctx, gpoolConcat.buf, p1Pass.buf, maxBatchSize);
    // mark output p1Pass of gpoolToPassMul input dependence for gpoolToPassBias
    cmdComputeBarrier(ctx.cmd, p1Pass.buf->buffer);
    gpoolToPassBias->dispatch(ctx, p1Pass.buf, maxBatchSize);
    // mark output p1Pass of gpoolToPassBias input dependence for gpoolToPassMul2
    cmdComputeBarrier(ctx.cmd, p1Pass.buf->buffer);
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
  bool useFP16) {
  name = desc->name;
  modelVersion = desc->modelVersion;
  paddedSpatialSize = paddedSpatialSize_;
  v1Channels = desc->v1Conv.outChannels;
  v2Channels = desc->v2Mul.outChannels;
  valueChannels = desc->v3Mul.outChannels;
  scoreValueChannels = desc->sv3Mul.outChannels;
  ownershipChannels = desc->vOwnershipConv.outChannels;

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
  valueHeadPoolKernel =
    VulkanKernels::ValueHeadPool::build(ctx.device, ctx.pipelineCache, useFP16, ctx.tuneParams.valueHeadPoolXystride);
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
  // mark output v1Out of v1Conv input dependence for v1BN
  cmdComputeBarrier(ctx.cmd, v1Out.buf->buffer);
  v1BN->dispatch(ctx, v1Out.buf, v1Out.buf, mask, paddedSpatialSize, maxBatchSize);
  // mark output v1Out of v1BN input dependence for valueHeadPool and vOwnershipConv
  cmdComputeBarrier(ctx.cmd, v1Out.buf->buffer);

  VulkanKernels::ValueHeadPool::PC pc = {v1Channels, paddedSpatialSize};
  VulkanKernels::ValueHeadPool::dispatch(ctx, valueHeadPoolKernel, v1Out.buf, v1Mean.buf, maskSum, pc, maxBatchSize);
  // mark output v1Mean of valueHeadPool input dependence for v2Mul
  cmdComputeBarrier(ctx.cmd, v1Mean.buf->buffer);

  v2Mul->dispatch(ctx, v1Mean.buf, v2Out.buf, maxBatchSize);
  // mark output v2Out of v2Mul input dependence for v2Bias
  cmdComputeBarrier(ctx.cmd, v2Out.buf->buffer);
  v2Bias->dispatch(ctx, v2Out.buf, maxBatchSize);
  // mark output v2Out of v2Bias input dependence for v3Mul and sv3Mul
  cmdComputeBarrier(ctx.cmd, v2Out.buf->buffer);
  v3Mul->dispatch(ctx, v2Out.buf, value, maxBatchSize);
  // mark output value of v3Mul input dependence for v3Bias (in-place)
  cmdComputeBarrier(ctx.cmd, value->buffer);
  // v3Bias writes value; its output barrier (for download) is the caller's responsibility (Model/getOutput).
  v3Bias->dispatch(ctx, value, maxBatchSize);

  sv3Mul->dispatch(ctx, v2Out.buf, scoreValue, maxBatchSize);
  // mark output scoreValue of sv3Mul input dependence for sv3Bias (in-place)
  cmdComputeBarrier(ctx.cmd, scoreValue->buffer);
  // sv3Bias writes scoreValue; its output barrier (for download) is the caller's responsibility (Model/getOutput).
  sv3Bias->dispatch(ctx, scoreValue, maxBatchSize);

  // vOwnershipConv reads v1Out (still live) and writes ownership; ownership's
  // output barrier (for download) is the caller's responsibility (Model/getOutput).
  vOwnershipConv->dispatch(ctx, scratch, v1Out.buf, ownership, maxBatchSize);
}

#endif  // USE_VULKAN_BACKEND

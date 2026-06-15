#ifndef NEURALNET_VULKAN_LAYERS_H_
#define NEURALNET_VULKAN_LAYERS_H_

#ifdef USE_VULKAN_BACKEND

#include <algorithm>
#include <array>
#include <initializer_list>
#include <limits>
#include <memory>
#include <numeric>
#include <string>
#include <vector>
#include "../core/test.h"
#include "../neuralnet/nninterface.h"
#include "../neuralnet/vulkanhelpers.h"
#include "../neuralnet/vulkanincludes.h"
#include "../neuralnet/vulkankernels.h"

// Forward-declared in vulkankernels.h. Full definition lives in vulkanbackend.h.
struct DispatchProfiler;

// NN layer / block / head primitives for the Vulkan backend.
//
// These global-scope structs compose the VulkanKernels:: compute primitives into
// the layer tree of a KataGo network: leaf layers (conv, batchnorm, matmul/bias),
// the residual / global-pooling / transformer blocks that stack them, and the
// trunk + policy/value heads. The orchestration layer (Model, ComputeHandle,
// ComputeContext, getOutput) in vulkanbackend.cpp builds and dispatches them.
//
// Leaf-layer and small-block bodies are inline here; the larger transformer /
// block / head / trunk bodies are defined in vulkanlayers.cpp.

// VulkanLayerContext: device-side resources every layer constructor takes,
// bundled into one value. Carries the per-call tuneParams + the shared pipeline
// cache, so a layer builds its own ComputeKernels for that config.
struct VulkanLayerContext {
  VkDevice device;
  VkQueue queue;
  std::mutex& queueMutex;
  VkCommandPool commandPool;
  const VkPhysicalDeviceMemoryProperties& memProps;
  const VulkanTuneParams& tuneParams;
  bool supportsSubgroupShuffleCompute;
  uint32_t subgroupSize;
  bool supportsFP16Compute;
  bool supportsDot2F16;
  bool supportsDot2F16AccF16;
  bool supportsCoopmat1F16;
  bool supportsCoopmat1F16AccF16;
  bool supportsCoopmat2F16;
  bool supportsCoopmat2F16AccF16;
  // Device-reported usable coopmat shapes (f16/f16->f32/f32, subgroup scope).
  // The coopmat TM/TN/TK must match one of these. Empty iff !supportsCoopmat1F16.
  std::vector<CoopmatShape> coopmatShapes;
  std::vector<CoopmatShape> coopmatAccF16Shapes;
  std::vector<Coopmat2FlexShape> coopmat2FlexShapes;
  std::vector<Coopmat2FlexShape> coopmat2AccF16FlexShapes;
  uint32_t coopmat2ReservedSharedBytes;
  VkPipelineCache pipelineCache;

  VulkanLayerContext() = delete;
  VulkanLayerContext(const VulkanLayerContext&) = delete;
  VulkanLayerContext& operator=(const VulkanLayerContext&) = delete;

  VulkanLayerContext(
    VkDevice device_,
    VkQueue queue_,
    std::mutex& queueMutex_,
    VkCommandPool commandPool_,
    const VkPhysicalDeviceMemoryProperties& memProps_,
    const VulkanTuneParams& tuneParams_,
    bool supportsSubgroupShuffleCompute_,
    uint32_t subgroupSize_,
    bool supportsFP16Compute_,
    bool supportsDot2F16_,
    bool supportsDot2F16AccF16_,
    bool supportsCoopmat1F16_,
    bool supportsCoopmat1F16AccF16_,
    bool supportsCoopmat2F16_,
    bool supportsCoopmat2F16AccF16_,
    std::vector<CoopmatShape> coopmatShapes_,
    std::vector<CoopmatShape> coopmatAccF16Shapes_,
    std::vector<Coopmat2FlexShape> coopmat2FlexShapes_,
    std::vector<Coopmat2FlexShape> coopmat2AccF16FlexShapes_,
    uint32_t coopmat2ReservedSharedBytes_,
    VkPipelineCache pipelineCache_)
    : device(device_),
      queue(queue_),
      queueMutex(queueMutex_),
      commandPool(commandPool_),
      memProps(memProps_),
      tuneParams(tuneParams_),
      supportsSubgroupShuffleCompute(supportsSubgroupShuffleCompute_),
      subgroupSize(subgroupSize_),
      supportsFP16Compute(supportsFP16Compute_),
      supportsDot2F16(supportsDot2F16_),
      supportsDot2F16AccF16(supportsDot2F16AccF16_),
      supportsCoopmat1F16(supportsCoopmat1F16_),
      supportsCoopmat1F16AccF16(supportsCoopmat1F16AccF16_),
      supportsCoopmat2F16(supportsCoopmat2F16_),
      supportsCoopmat2F16AccF16(supportsCoopmat2F16AccF16_),
      coopmatShapes(std::move(coopmatShapes_)),
      coopmatAccF16Shapes(std::move(coopmatAccF16Shapes_)),
      coopmat2FlexShapes(std::move(coopmat2FlexShapes_)),
      coopmat2AccF16FlexShapes(std::move(coopmat2AccF16FlexShapes_)),
      coopmat2ReservedSharedBytes(coopmat2ReservedSharedBytes_),
      pipelineCache(pipelineCache_) {}
};

enum class GemmStridedVariant { Tiled, Dot2, Dot2AccF16, Coopmat, CoopmatAccF16, Coopmat2, Coopmat2AccF16 };
enum class WinogradGemmVariant { Tiled, Dot2, Dot2AccF16, Coopmat, CoopmatAccF16, Coopmat2, Coopmat2AccF16 };

inline bool gemmStridedVariantUsesPackedB(GemmStridedVariant variant) {
  return variant == GemmStridedVariant::Dot2 || variant == GemmStridedVariant::Dot2AccF16 ||
         variant == GemmStridedVariant::Coopmat || variant == GemmStridedVariant::CoopmatAccF16 ||
         variant == GemmStridedVariant::Coopmat2 || variant == GemmStridedVariant::Coopmat2AccF16;
}

inline int gemmStridedPackedBBN(const VulkanTuneParams& tuneParams, GemmStridedVariant variant) {
  if(variant == GemmStridedVariant::Dot2 || variant == GemmStridedVariant::Dot2AccF16)
    return tuneParams.stridedDot2BN;
  if(variant == GemmStridedVariant::Coopmat)
    return tuneParams.stridedCoopmatBN;
  if(variant == GemmStridedVariant::CoopmatAccF16)
    return tuneParams.stridedCoopmatAccF16BN;
  if(variant == GemmStridedVariant::Coopmat2)
    return tuneParams.stridedCoopmat2BN;
  if(variant == GemmStridedVariant::Coopmat2AccF16)
    return tuneParams.stridedCoopmat2AccF16BN;
  return 0;
}

inline int gemmStridedPackedBBK(const VulkanTuneParams& tuneParams, GemmStridedVariant variant) {
  if(variant == GemmStridedVariant::Dot2 || variant == GemmStridedVariant::Dot2AccF16)
    return VulkanKernels::DOT2_BK;
  if(variant == GemmStridedVariant::Coopmat)
    return tuneParams.stridedCoopmatBK;
  if(variant == GemmStridedVariant::CoopmatAccF16)
    return tuneParams.stridedCoopmatAccF16BK;
  if(variant == GemmStridedVariant::Coopmat2)
    return tuneParams.stridedCoopmat2BK;
  if(variant == GemmStridedVariant::Coopmat2AccF16)
    return tuneParams.stridedCoopmat2AccF16BK;
  return 0;
}

inline int gemmStridedPackedBPadScalars(GemmStridedVariant variant) {
  if(variant == GemmStridedVariant::Dot2 || variant == GemmStridedVariant::Dot2AccF16)
    return STRIDED_DOT2_PACKED_B_PAD_SCALARS;
  if(variant == GemmStridedVariant::Coopmat)
    return STRIDED_COOPMAT_PACKED_B_PAD_SCALARS;
  if(variant == GemmStridedVariant::CoopmatAccF16)
    return STRIDED_COOPMAT_PACKED_B_PAD_SCALARS;
  if(variant == GemmStridedVariant::Coopmat2)
    return STRIDED_COOPMAT2_PACKED_B_PAD_SCALARS;
  if(variant == GemmStridedVariant::Coopmat2AccF16)
    return STRIDED_COOPMAT2_PACKED_B_PAD_SCALARS;
  return 0;
}

inline int winogradPackedABM(const VulkanTuneParams& tuneParams, WinogradGemmVariant variant) {
  if(variant == WinogradGemmVariant::Dot2 || variant == WinogradGemmVariant::Dot2AccF16)
    return tuneParams.dot2BM;
  if(variant == WinogradGemmVariant::Coopmat)
    return tuneParams.coopmatBM;
  if(variant == WinogradGemmVariant::CoopmatAccF16)
    return tuneParams.coopmatAccF16BM;
  if(variant == WinogradGemmVariant::Coopmat2)
    return tuneParams.coopmat2BM;
  if(variant == WinogradGemmVariant::Coopmat2AccF16)
    return tuneParams.coopmat2AccF16BM;
  return tuneParams.winogradGemmM;
}

inline int winogradPackedABK(const VulkanTuneParams& tuneParams, WinogradGemmVariant variant) {
  if(variant == WinogradGemmVariant::Dot2 || variant == WinogradGemmVariant::Dot2AccF16)
    return VulkanKernels::DOT2_BK;
  if(variant == WinogradGemmVariant::Coopmat)
    return tuneParams.coopmatBK;
  if(variant == WinogradGemmVariant::CoopmatAccF16)
    return tuneParams.coopmatAccF16BK;
  if(variant == WinogradGemmVariant::Coopmat2)
    return tuneParams.coopmat2BK;
  if(variant == WinogradGemmVariant::Coopmat2AccF16)
    return tuneParams.coopmat2AccF16BK;
  return tuneParams.winogradGemmK;
}

inline int winogradPackedAPadWords(WinogradGemmVariant variant) {
  if(variant == WinogradGemmVariant::Coopmat || variant == WinogradGemmVariant::CoopmatAccF16)
    return WINOGRAD_COOPMAT_PACKED_PAD_WORDS;
  if(variant == WinogradGemmVariant::Coopmat2 || variant == WinogradGemmVariant::Coopmat2AccF16)
    return WINOGRAD_COOPMAT2_PACKED_PAD_WORDS;
  return WINOGRAD_ROW_MAJOR_A_PAD_WORDS;
}

inline int winogradCoopmatPackedAPadWords(WinogradGemmVariant variant) {
  return winogradPackedAPadWords(variant);
}

inline GemmStridedVariant resolveGemmStridedVariant(const VulkanLayerContext& ctx, bool fp16) {
  // Coopmat and Dot2 are mutually-exclusively enabled by the tuner mode-select
  // (only the winning variant's enable flag is set), so the order here just
  // reflects that at most one is active.
  if(fp16 && ctx.supportsCoopmat2F16AccF16 && ctx.tuneParams.enableGemmStridedCoopmat2AccF16 != 0)
    return GemmStridedVariant::Coopmat2AccF16;
  if(fp16 && ctx.supportsCoopmat2F16 && ctx.tuneParams.enableGemmStridedCoopmat2 != 0)
    return GemmStridedVariant::Coopmat2;
  if(fp16 && ctx.supportsCoopmat1F16AccF16 && ctx.tuneParams.enableGemmStridedCoopmatAccF16 != 0)
    return GemmStridedVariant::CoopmatAccF16;
  if(fp16 && ctx.supportsCoopmat1F16 && ctx.tuneParams.enableGemmStridedCoopmat != 0)
    return GemmStridedVariant::Coopmat;
  if(fp16 && ctx.supportsDot2F16AccF16 && ctx.tuneParams.enableGemmStridedDot2AccF16 != 0)
    return GemmStridedVariant::Dot2AccF16;
  if(fp16 && ctx.supportsDot2F16 && ctx.tuneParams.enableGemmStridedDot2 != 0)
    return GemmStridedVariant::Dot2;
  return GemmStridedVariant::Tiled;
}

inline WinogradGemmVariant resolveWinogradGemmVariant(const VulkanLayerContext& ctx, bool fp16) {
  if(fp16 && ctx.supportsCoopmat2F16AccF16 && ctx.tuneParams.enableWinogradGemmCoopmat2AccF16 != 0)
    return WinogradGemmVariant::Coopmat2AccF16;
  if(fp16 && ctx.supportsCoopmat2F16 && ctx.tuneParams.enableWinogradGemmCoopmat2 != 0)
    return WinogradGemmVariant::Coopmat2;
  if(fp16 && ctx.supportsCoopmat1F16AccF16 && ctx.tuneParams.enableWinogradGemmCoopmatAccF16 != 0)
    return WinogradGemmVariant::CoopmatAccF16;
  if(fp16 && ctx.supportsCoopmat1F16 && ctx.tuneParams.enableWinogradGemmCoopmat != 0)
    return WinogradGemmVariant::Coopmat;
  if(fp16 && ctx.supportsDot2F16AccF16 && ctx.tuneParams.enableWinogradGemmDot2AccF16 != 0)
    return WinogradGemmVariant::Dot2AccF16;
  if(fp16 && ctx.supportsDot2F16 && ctx.tuneParams.enableWinogradGemmDot2 != 0)
    return WinogradGemmVariant::Dot2;
  return WinogradGemmVariant::Tiled;
}

inline LayerPaddingContract selectedWinogradPaddingContract(
  const VulkanTuneParams& tuneParams,
  WinogradGemmVariant variant) {
  switch(variant) {
    case WinogradGemmVariant::Coopmat2AccF16:
      return VulkanKernels::WinogradGemmCoopmat2AccF16::layerPaddingContract(tuneParams);
    case WinogradGemmVariant::Coopmat2:
      return VulkanKernels::WinogradGemmCoopmat2::layerPaddingContract(tuneParams);
    case WinogradGemmVariant::CoopmatAccF16:
      return VulkanKernels::WinogradGemmCoopmatAccF16::layerPaddingContract(tuneParams);
    case WinogradGemmVariant::Coopmat:
      return VulkanKernels::WinogradGemmCoopmat::layerPaddingContract(tuneParams);
    case WinogradGemmVariant::Dot2:
      return VulkanKernels::WinogradGemmDot2::layerPaddingContract(tuneParams);
    case WinogradGemmVariant::Dot2AccF16:
      return VulkanKernels::WinogradGemmDot2AccF16::layerPaddingContract(tuneParams);
    case WinogradGemmVariant::Tiled:
      return VulkanKernels::WinogradGemm::layerPaddingContract(tuneParams);
    default:
      ASSERT_UNREACHABLE;
  }
}

// Lcm-merge a set of LayerPaddingContracts into a single contract. Used when
// a strided layer might select any of several kernel variants at runtime. K is
// merged only when the contributing contract declares kPaddable=true; the
// merged kPaddable stays true only while every merged contract was itself
// kPaddable.
inline LayerPaddingContract mergePaddingContracts(std::initializer_list<LayerPaddingContract> contracts) {
  LayerPaddingContract merged;  // defaults {1,1,1,true}
  for(const auto& r: contracts) {
    merged.m = std::lcm(merged.m, r.m);
    merged.n = std::lcm(merged.n, r.n);
    if(r.kPaddable)
      merged.k = std::lcm(merged.k, r.k);
    merged.kPaddable = merged.kPaddable && r.kPaddable;
  }
  return merged;
}

// mergedStridedPaddingContract: strided path applies N-padding only.
//
// Consumers should read .n only. Every strided kernel declares {m=1, k=1,
// kPaddable=false} in its contract, so the generic lcm-merge naturally
// returns {1, mergedN, 1, false} — .m and .k are informational (always 1).
// Runtime strided M is paddedSpatialSize (VULKAN_SPATIAL_ALIGN, globally
// pinned) and K is unpaddable across layers.
inline LayerPaddingContract mergedStridedPaddingContract(const VulkanLayerContext& ctx, bool fp16) {
  LayerPaddingContract merged = VulkanKernels::GemmStridedTiled::layerPaddingContract(ctx.tuneParams);
  if(fp16 && ctx.supportsDot2F16 && ctx.tuneParams.enableGemmStridedDot2 != 0)
    merged = mergePaddingContracts({merged, VulkanKernels::GemmStridedDot2::layerPaddingContract(ctx.tuneParams)});
  if(fp16 && ctx.supportsDot2F16AccF16 && ctx.tuneParams.enableGemmStridedDot2AccF16 != 0)
    merged =
      mergePaddingContracts({merged, VulkanKernels::GemmStridedDot2AccF16::layerPaddingContract(ctx.tuneParams)});
  if(fp16 && ctx.supportsCoopmat1F16 && ctx.tuneParams.enableGemmStridedCoopmat != 0)
    merged = mergePaddingContracts({merged, VulkanKernels::GemmStridedCoopmat::layerPaddingContract(ctx.tuneParams)});
  if(fp16 && ctx.supportsCoopmat1F16AccF16 && ctx.tuneParams.enableGemmStridedCoopmatAccF16 != 0)
    merged =
      mergePaddingContracts({merged, VulkanKernels::GemmStridedCoopmatAccF16::layerPaddingContract(ctx.tuneParams)});
  if(fp16 && ctx.supportsCoopmat2F16 && ctx.tuneParams.enableGemmStridedCoopmat2 != 0)
    merged = mergePaddingContracts({merged, VulkanKernels::GemmStridedCoopmat2::layerPaddingContract(ctx.tuneParams)});
  if(fp16 && ctx.supportsCoopmat2F16AccF16 && ctx.tuneParams.enableGemmStridedCoopmat2AccF16 != 0)
    merged =
      mergePaddingContracts({merged, VulkanKernels::GemmStridedCoopmat2AccF16::layerPaddingContract(ctx.tuneParams)});
  return merged;
}

// BatchNorm + activation + mask. Owns one scaleBiasMaskAct kernel for its
// activation and dispatches itself via dispatch() (input -> output, in place ok).
struct BatchNormLayer {
  const std::string name;
  const int numChannels;
  const int activation;
  VBuf mergedScaleBuf;
  VBuf mergedBiasBuf;
  VkDevice device;
  ComputeKernel scaleBiasMaskActKernel;

  BatchNormLayer(
    const VulkanLayerContext& ctx,
    const BatchNormLayerDesc* desc,
    const ActivationLayerDesc* actDesc,
    bool useFP16)
    : name(desc->name), numChannels(desc->numChannels), activation(actDesc->activation), device(ctx.device) {
    mergedScaleBuf =
      makeWeightBuf(ctx.device, ctx.queue, ctx.queueMutex, ctx.commandPool, ctx.memProps, desc->mergedScale, useFP16);
    mergedBiasBuf =
      makeWeightBuf(ctx.device, ctx.queue, ctx.queueMutex, ctx.commandPool, ctx.memProps, desc->mergedBias, useFP16);
    scaleBiasMaskActKernel = VulkanKernels::ScaleBiasMaskAct::build(ctx.device, ctx.pipelineCache, useFP16, activation);
  }

  BatchNormLayer() = delete;
  BatchNormLayer(const BatchNormLayer&) = delete;
  BatchNormLayer& operator=(const BatchNormLayer&) = delete;

  ~BatchNormLayer() { scaleBiasMaskActKernel.destroy(device); }

  void dispatch(
    const CmdCtx& ctx,
    VulkanBuffer* input,
    VulkanBuffer* output,
    VulkanBuffer* mask,
    int paddedSpatialSize,
    int maxBatchSize) const {
    VulkanKernels::ScaleBiasMaskAct::PC pc = {numChannels, paddedSpatialSize, maxBatchSize};
    VulkanKernels::ScaleBiasMaskAct::dispatch(
      ctx, scaleBiasMaskActKernel, input, output, mergedScaleBuf.get(), mergedBiasBuf.get(), mask, pc);
  }
};

// FP32 fully-connected layer (MatMul). Owns a gemmDirectFP32 kernel.
struct MatMulLayer {
  const std::string name;
  const int inChannels;
  const int outChannels;
  VBuf weightBuf;  // [outChannels, inChannels] FP32
  VkDevice device;
  ComputeKernel gemmDirectKernel;

  MatMulLayer(const VulkanLayerContext& ctx, const MatMulLayerDesc* desc)
    : name(desc->name), inChannels(desc->inChannels), outChannels(desc->outChannels), device(ctx.device) {
    testAssert(desc->weights.size() == (size_t)inChannels * (size_t)outChannels);
    std::vector<float> transW(inChannels * outChannels);
    for(int oc = 0; oc < outChannels; oc++)
      for(int ic = 0; ic < inChannels; ic++)
        transW[oc * inChannels + ic] = desc->weights[ic * outChannels + oc];
    weightBuf = makeWeightBuf(ctx.device, ctx.queue, ctx.queueMutex, ctx.commandPool, ctx.memProps, transW, false);
    gemmDirectKernel = VulkanKernels::GemmDirectFP32::build(
      ctx.device,
      ctx.pipelineCache,
      ctx.tuneParams.gemmDirectLocalSizeX,
      ctx.tuneParams.gemmDirectLocalSizeY,
      inChannels);
  }

  MatMulLayer() = delete;
  MatMulLayer(const MatMulLayer&) = delete;
  MatMulLayer& operator=(const MatMulLayer&) = delete;
  ~MatMulLayer() { gemmDirectKernel.destroy(device); }

  void dispatch(const CmdCtx& ctx, VulkanBuffer* input, VulkanBuffer* output, int maxBatchSize) const {
    VulkanKernels::GemmDirectFP32::PC pc = {maxBatchSize, outChannels};
    VulkanKernels::GemmDirectFP32::dispatch(ctx, gemmDirectKernel, input, weightBuf.get(), output, inChannels, pc);
  }
};

// FP32 bias + activation layer. Owns an addCBiasActNC kernel.
struct MatBiasLayer {
  const std::string name;
  const int numChannels;
  const int activation;
  VBuf biasBuf;
  VkDevice device;
  ComputeKernel addCBiasActKernel;

  MatBiasLayer(const VulkanLayerContext& ctx, const MatBiasLayerDesc* desc, const ActivationLayerDesc* actDesc)
    : name(desc->name),
      numChannels(desc->numChannels),
      activation(actDesc ? actDesc->activation : ACTIVATION_IDENTITY),
      device(ctx.device) {
    biasBuf = makeWeightBuf(ctx.device, ctx.queue, ctx.queueMutex, ctx.commandPool, ctx.memProps, desc->weights, false);
    addCBiasActKernel = VulkanKernels::AddCBiasActNC::build(ctx.device, ctx.pipelineCache, activation);
  }

  MatBiasLayer() = delete;
  MatBiasLayer(const MatBiasLayer&) = delete;
  MatBiasLayer& operator=(const MatBiasLayer&) = delete;
  ~MatBiasLayer() { addCBiasActKernel.destroy(device); }

  void dispatch(const CmdCtx& ctx, VulkanBuffer* data, int maxBatchSize) const {
    VulkanKernels::AddCBiasActNC::PC pc = {numChannels};
    VulkanKernels::AddCBiasActNC::dispatch(ctx, addCBiasActKernel, data, biasBuf.get(), pc, maxBatchSize);
  }
};

// A convolution layer: owns the compute pipelines for its dispatch path (1x1 ->
// direct strided GEMM; 3x3/5x5 -> winograd transform+winogradGemm+untransform;
// else -> conv2dDirect), built in its ctor from ctx.tuneParams + ctx.pipelineCache.
struct ConvLayer {
  const std::string name;
  const int convYSize, convXSize;
  const int inChannels, outChannels;
  const int nnXLen, nnYLen, paddedSpatialSize;

  // For 3x3/5x5 Winograd
  int numTilesX, numTilesY;
  int inTileXYSize, outTileXYSize;
  int inTileXSize, inTileYSize;
  int outTileXSize, outTileYSize;
  int numInChannelsPadded, numOutChannelsPadded;
  int numTilesPadded;
  // M-dimension pad multiple for the winograd path (numTilesPaddedRun = roundUp(ntxty, this)).
  // Cached from the selected Winograd GEMM variant at ctor time (post-tune).
  int winogradMAlignment;
  int winogradPackedABM;
  int winogradPackedABK;
  int winogradPackedAPadWords;
  int winogradCoopmatBM;
  int winogradCoopmatBK;
  int winogradCoopmatBStride;
  int winogradTransformOffset;
  // For 1x1 GemmStrided path: outChannels padded to ÷8.
  int outChannelsPadded1x1;
  int gemmStridedBStride;

  VBuf filterBuf;

  VkDevice device;
  ComputeKernel gemmStridedKernel;
  ComputeKernel winogradTransformKernel;
  ComputeKernel winogradGemmKernel;
  ComputeKernel winogradUntransformKernel;
  ComputeKernel conv2dDirectKernel;
  GemmStridedVariant gemmStridedVariant;
  WinogradGemmVariant winogradGemmVariant;

  size_t workspaceElts1;
  size_t workspaceElts2;

  ConvLayer(
    const VulkanLayerContext& ctx,
    const ConvLayerDesc* desc,
    int nnX,
    int nnY,
    int paddedSpatialSize_,
    bool useFP16,
    bool addToOutputOnStore = false)
    : name(desc->name),
      convYSize(desc->convYSize),
      convXSize(desc->convXSize),
      inChannels(desc->inChannels),
      outChannels(desc->outChannels),
      nnXLen(nnX),
      nnYLen(nnY),
      paddedSpatialSize(paddedSpatialSize_),
      device(ctx.device) {
    if(desc->dilationX != 1 || desc->dilationY != 1)
      throw StringError("Vulkan backend: convolution dilation != 1 not supported");

    workspaceElts1 = 0;
    workspaceElts2 = 0;
    numTilesX = numTilesY = 0;
    inTileXYSize = outTileXYSize = 0;
    inTileXSize = inTileYSize = 0;
    outTileXSize = outTileYSize = 0;
    numInChannelsPadded = numOutChannelsPadded = 0;
    numTilesPadded = 0;
    winogradMAlignment = (int)ctx.tuneParams.winogradGemmM;
    winogradPackedABM = 0;
    winogradPackedABK = 0;
    winogradPackedAPadWords = 0;
    winogradCoopmatBM = 0;
    winogradCoopmatBK = 0;
    winogradCoopmatBStride = 0;
    winogradTransformOffset = 0;
    numTilesPadded = 0;
    outChannelsPadded1x1 = 0;
    gemmStridedBStride = 0;
    gemmStridedVariant = GemmStridedVariant::Tiled;
    winogradGemmVariant = WinogradGemmVariant::Tiled;
    const bool isGemmStrided1x1 = convXSize == 1 && convYSize == 1;

    if(isGemmStrided1x1) {
      gemmStridedVariant = resolveGemmStridedVariant(ctx, useFP16);
      // Pad outC to the lcm of every enabled strided variant's requirement (see
      // mergedStridedPaddingContract). K stays real (previous layer's outC).
      // Zero-fill padding so kernels that read the padded region see zeros.
      LayerPaddingContract stridedContract = mergedStridedPaddingContract(ctx, useFP16);
      outChannelsPadded1x1 = roundUpToMultipleInt(outChannels, stridedContract.n);
      std::vector<float> transW(inChannels * outChannelsPadded1x1, 0.0f);
      for(int oc = 0; oc < outChannels; oc++)
        for(int ic = 0; ic < inChannels; ic++)
          transW[ic * outChannelsPadded1x1 + oc] = desc->weights[oc * inChannels + ic];
      if(gemmStridedVariantUsesPackedB(gemmStridedVariant)) {
        const int packedBN = gemmStridedPackedBBN(ctx.tuneParams, gemmStridedVariant);
        const int packedBK = gemmStridedPackedBBK(ctx.tuneParams, gemmStridedVariant);
        const int packedPadScalars = gemmStridedPackedBPadScalars(gemmStridedVariant);
        std::vector<float> packedW =
          packStridedGemmBWeights(transW, 1, outChannelsPadded1x1, inChannels, packedBN, packedBK, packedPadScalars);
        filterBuf =
          makeWeightBuf(ctx.device, ctx.queue, ctx.queueMutex, ctx.commandPool, ctx.memProps, packedW, useFP16);
      } else {
        filterBuf =
          makeWeightBuf(ctx.device, ctx.queue, ctx.queueMutex, ctx.commandPool, ctx.memProps, transW, useFP16);
      }
      if(gemmStridedVariant == GemmStridedVariant::Coopmat2AccF16) {
        const auto cm2Tile = VulkanKernels::GemmStridedCoopmat2AccF16::tileAlignmentReq(ctx.tuneParams);
        int32_t aligned = (paddedSpatialSize % cm2Tile.m == 0 && outChannelsPadded1x1 % cm2Tile.n == 0) ? 1 : 0;
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
          addToOutputOnStore);
      } else if(gemmStridedVariant == GemmStridedVariant::Coopmat2) {
        const auto cm2Tile = VulkanKernels::GemmStridedCoopmat2::tileAlignmentReq(ctx.tuneParams);
        int32_t aligned = (paddedSpatialSize % cm2Tile.m == 0 && outChannelsPadded1x1 % cm2Tile.n == 0) ? 1 : 0;
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
          addToOutputOnStore);
      } else if(gemmStridedVariant == GemmStridedVariant::CoopmatAccF16) {
        const auto cmTile = VulkanKernels::GemmStridedCoopmatAccF16::tileAlignmentReq(ctx.tuneParams);
        int32_t aligned = (paddedSpatialSize % cmTile.m == 0 && outChannelsPadded1x1 % cmTile.n == 0) ? 1 : 0;
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
          addToOutputOnStore);
      } else if(gemmStridedVariant == GemmStridedVariant::Coopmat) {
        // Same padding contract as DOT2: N (outChannelsPadded1x1) is aligned by
        // mergedStridedPaddingContract; M is VULKAN_SPATIAL_ALIGN-pinned; K real.
        // ALIGNED reflects M/N safety only; K_ALIGNED handles the K-tail guard.
        const auto cmTile = VulkanKernels::GemmStridedCoopmat::tileAlignmentReq(ctx.tuneParams);
        int32_t aligned = (paddedSpatialSize % cmTile.m == 0 && outChannelsPadded1x1 % cmTile.n == 0) ? 1 : 0;
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
          addToOutputOnStore);
      } else if(gemmStridedVariant == GemmStridedVariant::Dot2) {
        // 1x1 strided path: N (outChannelsPadded1x1) is DOT2-aligned by
        // mergedStridedPaddingContract; M (paddedSpatialSize) is globally
        // pinned to VULKAN_SPATIAL_ALIGN and NOT per-variant re-padded; K
        // stays real (kPaddable:false). ALIGNED reflects M/N safety only;
        // K_ALIGNED handles the K-tail guard.
        const auto dot2Tile = VulkanKernels::GemmStridedDot2::tileAlignmentReq(ctx.tuneParams);
        int32_t aligned = (paddedSpatialSize % dot2Tile.m == 0 && outChannelsPadded1x1 % dot2Tile.n == 0) ? 1 : 0;
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
          addToOutputOnStore);
      } else if(gemmStridedVariant == GemmStridedVariant::Dot2AccF16) {
        const auto dot2Tile = VulkanKernels::GemmStridedDot2AccF16::tileAlignmentReq(ctx.tuneParams);
        int32_t aligned = (paddedSpatialSize % dot2Tile.m == 0 && outChannelsPadded1x1 % dot2Tile.n == 0) ? 1 : 0;
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
          addToOutputOnStore);
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
          addToOutputOnStore);
      }
    } else if((convXSize == 3 && convYSize == 3) || (convXSize == 5 && convYSize == 5)) {
      outTileXSize = (convXSize == 3) ? winograd3x3OutTileFor(ctx.tuneParams) : 2;
      outTileYSize = outTileXSize;
      inTileXSize = outTileXSize + convXSize - 1;
      inTileYSize = inTileXSize;
      winogradTransformOffset = -(convXSize / 2);
      winogradGemmVariant = resolveWinogradGemmVariant(ctx, useFP16);
      winogradPackedABM = ::winogradPackedABM(ctx.tuneParams, winogradGemmVariant);
      winogradPackedABK = ::winogradPackedABK(ctx.tuneParams, winogradGemmVariant);
      winogradPackedAPadWords = ::winogradPackedAPadWords(winogradGemmVariant);

      // Pad only to the selected Winograd GEMM variant. The tuner mode-select
      // keeps variants mutually exclusive at runtime, and hand-edited configs
      // follow resolveWinogradGemmVariant's deterministic priority.
      LayerPaddingContract winoContract = selectedWinogradPaddingContract(ctx.tuneParams, winogradGemmVariant);
      winogradMAlignment = winoContract.m;
      numInChannelsPadded = roundUpToMultipleInt(inChannels, winoContract.k);
      numOutChannelsPadded = roundUpToMultipleInt(outChannels, winoContract.n);

      numTilesX = (nnX + outTileXSize - 1) / outTileXSize;
      numTilesY = (nnY + outTileYSize - 1) / outTileYSize;
      inTileXYSize = inTileXSize * inTileYSize;
      outTileXYSize = outTileXSize * outTileYSize;

      numTilesPadded = roundUpToMultipleInt(numTilesX * numTilesY * 1, winoContract.m);

      auto transform3x3_4 = [](float& a0, float& a1, float& a2, float& a3) {
        float z0 = a0, z1 = a1, z2 = a2;
        a0 = z0;
        a1 = 0.5f * (z0 + z1 + z2);
        a2 = 0.5f * (z0 - z1 + z2);
        a3 = z2;
      };
      auto transform3x3_6 = [](float* a) {
        float z0 = a[0], z1 = a[1], z2 = a[2];
        a[0] = 0.25f * z0;
        a[1] = (float)((1.0 / 6.0) * (-z0 - z1 - z2));
        a[2] = (float)((1.0 / 6.0) * (-z0 + z1 - z2));
        a[3] = (float)((1.0 / 24.0) * (z0 + 2.0 * z1 + 4.0 * z2));
        a[4] = (float)((1.0 / 24.0) * (z0 - 2.0 * z1 + 4.0 * z2));
        a[5] = z2;
      };
      auto transform5x5_6 = [](float* a) {
        float z0 = a[0], z1 = a[1], z2 = a[2], z3 = a[3], z4 = a[4];
        a[0] = 0.25f * z0;
        a[1] = (float)((1.0 / 6.0) * (-z0 - z1 - z2 - z3 - z4));
        a[2] = (float)((1.0 / 6.0) * (-z0 + z1 - z2 + z3 - z4));
        a[3] = (float)((1.0 / 24.0) * (z0 + 2.0 * z1 + 4.0 * z2 + 8.0 * z3 + 16.0 * z4));
        a[4] = (float)((1.0 / 24.0) * (z0 - 2.0 * z1 + 4.0 * z2 - 8.0 * z3 + 16.0 * z4));
        a[5] = z4;
      };

      std::vector<float> transWeights(inTileXYSize * numInChannelsPadded * numOutChannelsPadded, 0.0f);
      for(int oc = 0; oc < numOutChannelsPadded; oc++) {
        for(int ic = 0; ic < numInChannelsPadded; ic++) {
          float tmp[6][6] = {};
          if(oc < outChannels && ic < inChannels) {
            for(int sy = 0; sy < convYSize; sy++)
              for(int sx = 0; sx < convXSize; sx++)
                tmp[sy][sx] = desc->weights[((oc * inChannels + ic) * convYSize + sy) * convXSize + sx];
          }
          if(convXSize == 3) {
            if(inTileXSize == 4) {
              for(int sy = 0; sy < convYSize; sy++)
                transform3x3_4(tmp[sy][0], tmp[sy][1], tmp[sy][2], tmp[sy][3]);
              for(int sx = 0; sx < inTileXSize; sx++)
                transform3x3_4(tmp[0][sx], tmp[1][sx], tmp[2][sx], tmp[3][sx]);
            } else if(inTileXSize == 6) {
              for(int sy = 0; sy < convYSize; sy++)
                transform3x3_6(tmp[sy]);
              for(int sx = 0; sx < inTileXSize; sx++) {
                float col[6] = {tmp[0][sx], tmp[1][sx], tmp[2][sx], tmp[3][sx], tmp[4][sx], tmp[5][sx]};
                transform3x3_6(col);
                for(int sy2 = 0; sy2 < inTileYSize; sy2++)
                  tmp[sy2][sx] = col[sy2];
              }
            } else {
              throw StringError(
                "Vulkan backend: unsupported 3x3 Winograd input tile size " + Global::intToString(inTileXSize));
            }
          } else {
            for(int sy = 0; sy < convYSize; sy++)
              transform5x5_6(tmp[sy]);
            for(int sx = 0; sx < inTileXSize; sx++) {
              float col[6] = {tmp[0][sx], tmp[1][sx], tmp[2][sx], tmp[3][sx], tmp[4][sx], tmp[5][sx]};
              transform5x5_6(col);
              for(int sy2 = 0; sy2 < inTileYSize; sy2++)
                tmp[sy2][sx] = col[sy2];
            }
          }
          for(int subY = 0; subY < inTileYSize; subY++) {
            for(int subX = 0; subX < inTileXSize; subX++) {
              int tidx = (subY * inTileXSize + subX) * numInChannelsPadded * numOutChannelsPadded +
                         ic * numOutChannelsPadded + oc;
              transWeights[tidx] = tmp[subY][subX];
            }
          }
        }
      }

      if(winogradGemmVariant == WinogradGemmVariant::CoopmatAccF16) {
        winogradCoopmatBM = ctx.tuneParams.coopmatAccF16BM;
        winogradCoopmatBK = ctx.tuneParams.coopmatAccF16BK;
        size_t packedStrideB = winogradCoopmatPackedBStrideWords(
          numOutChannelsPadded, numInChannelsPadded, ctx.tuneParams.coopmatAccF16BN, ctx.tuneParams.coopmatAccF16BK);
        testAssert(packedStrideB <= (size_t)std::numeric_limits<int>::max());
        winogradCoopmatBStride = (int)packedStrideB;
        std::vector<float> packedWeights = packWinogradCoopmatBWeights(
          transWeights,
          inTileXYSize,
          numOutChannelsPadded,
          numInChannelsPadded,
          ctx.tuneParams.coopmatAccF16BN,
          ctx.tuneParams.coopmatAccF16BK);
        filterBuf =
          makeWeightBuf(ctx.device, ctx.queue, ctx.queueMutex, ctx.commandPool, ctx.memProps, packedWeights, useFP16);
      } else if(winogradGemmVariant == WinogradGemmVariant::Coopmat) {
        winogradCoopmatBM = ctx.tuneParams.coopmatBM;
        winogradCoopmatBK = ctx.tuneParams.coopmatBK;
        size_t packedStrideB = winogradCoopmatPackedBStrideWords(
          numOutChannelsPadded, numInChannelsPadded, ctx.tuneParams.coopmatBN, ctx.tuneParams.coopmatBK);
        testAssert(packedStrideB <= (size_t)std::numeric_limits<int>::max());
        winogradCoopmatBStride = (int)packedStrideB;
        std::vector<float> packedWeights = packWinogradCoopmatBWeights(
          transWeights,
          inTileXYSize,
          numOutChannelsPadded,
          numInChannelsPadded,
          ctx.tuneParams.coopmatBN,
          ctx.tuneParams.coopmatBK);
        filterBuf =
          makeWeightBuf(ctx.device, ctx.queue, ctx.queueMutex, ctx.commandPool, ctx.memProps, packedWeights, useFP16);
      } else if(winogradGemmVariant == WinogradGemmVariant::Coopmat2AccF16) {
        winogradCoopmatBM = ctx.tuneParams.coopmat2AccF16BM;
        winogradCoopmatBK = ctx.tuneParams.coopmat2AccF16BK;
        size_t packedStrideB = winogradCoopmatPackedBStrideWords(
          numOutChannelsPadded,
          numInChannelsPadded,
          ctx.tuneParams.coopmat2AccF16BN,
          ctx.tuneParams.coopmat2AccF16BK,
          WINOGRAD_COOPMAT2_PACKED_PAD_WORDS);
        testAssert(packedStrideB <= (size_t)std::numeric_limits<int>::max());
        winogradCoopmatBStride = (int)packedStrideB;
        std::vector<float> packedWeights = packWinogradCoopmatBWeights(
          transWeights,
          inTileXYSize,
          numOutChannelsPadded,
          numInChannelsPadded,
          ctx.tuneParams.coopmat2AccF16BN,
          ctx.tuneParams.coopmat2AccF16BK,
          WINOGRAD_COOPMAT2_PACKED_PAD_WORDS);
        filterBuf =
          makeWeightBuf(ctx.device, ctx.queue, ctx.queueMutex, ctx.commandPool, ctx.memProps, packedWeights, useFP16);
      } else if(winogradGemmVariant == WinogradGemmVariant::Coopmat2) {
        winogradCoopmatBM = ctx.tuneParams.coopmat2BM;
        winogradCoopmatBK = ctx.tuneParams.coopmat2BK;
        size_t packedStrideB = winogradCoopmatPackedBStrideWords(
          numOutChannelsPadded,
          numInChannelsPadded,
          ctx.tuneParams.coopmat2BN,
          ctx.tuneParams.coopmat2BK,
          WINOGRAD_COOPMAT2_PACKED_PAD_WORDS);
        testAssert(packedStrideB <= (size_t)std::numeric_limits<int>::max());
        winogradCoopmatBStride = (int)packedStrideB;
        std::vector<float> packedWeights = packWinogradCoopmatBWeights(
          transWeights,
          inTileXYSize,
          numOutChannelsPadded,
          numInChannelsPadded,
          ctx.tuneParams.coopmat2BN,
          ctx.tuneParams.coopmat2BK,
          WINOGRAD_COOPMAT2_PACKED_PAD_WORDS);
        filterBuf =
          makeWeightBuf(ctx.device, ctx.queue, ctx.queueMutex, ctx.commandPool, ctx.memProps, packedWeights, useFP16);
      } else {
        filterBuf =
          makeWeightBuf(ctx.device, ctx.queue, ctx.queueMutex, ctx.commandPool, ctx.memProps, transWeights, useFP16);
      }

      workspaceElts1 = static_cast<size_t>(inTileXYSize) * numInChannelsPadded;
      workspaceElts2 = static_cast<size_t>(inTileXYSize) * numOutChannelsPadded;

      const int inTile = inTileXSize;
      const int outTile = outTileXSize;
      const int convSz = convXSize;
      const int offset = winogradTransformOffset;
      winogradTransformKernel = VulkanKernels::WinogradTransform::build(
        ctx.device,
        ctx.pipelineCache,
        useFP16,
        inTile,
        outTile,
        convSz,
        offset,
        ctx.tuneParams.winogradTransformLocalSizeX,
        ctx.tuneParams.winogradTransformLocalSizeY,
        winogradPackedABM,
        winogradPackedABK,
        winogradPackedAPadWords);
      if(winogradGemmVariant == WinogradGemmVariant::Coopmat2AccF16) {
        winogradGemmKernel = VulkanKernels::WinogradGemmCoopmat2AccF16::build(
          ctx.device,
          ctx.pipelineCache,
          ctx.tuneParams.coopmat2AccF16BlockSize,
          ctx.tuneParams.coopmat2AccF16BM,
          ctx.tuneParams.coopmat2AccF16BN,
          ctx.tuneParams.coopmat2AccF16BK,
          numInChannelsPadded);
      } else if(winogradGemmVariant == WinogradGemmVariant::Coopmat2) {
        winogradGemmKernel = VulkanKernels::WinogradGemmCoopmat2::build(
          ctx.device,
          ctx.pipelineCache,
          ctx.tuneParams.coopmat2BlockSize,
          ctx.tuneParams.coopmat2BM,
          ctx.tuneParams.coopmat2BN,
          ctx.tuneParams.coopmat2BK,
          numInChannelsPadded);
      } else if(winogradGemmVariant == WinogradGemmVariant::CoopmatAccF16) {
        winogradGemmKernel = VulkanKernels::WinogradGemmCoopmatAccF16::build(
          ctx.device,
          ctx.pipelineCache,
          ctx.tuneParams.coopmatAccF16BlockSize,
          ctx.tuneParams.coopmatAccF16BM,
          ctx.tuneParams.coopmatAccF16BN,
          ctx.tuneParams.coopmatAccF16BK,
          ctx.tuneParams.coopmatAccF16WM,
          ctx.tuneParams.coopmatAccF16WN,
          ctx.tuneParams.coopmatAccF16TM,
          ctx.tuneParams.coopmatAccF16TN,
          ctx.tuneParams.coopmatAccF16TK,
          ctx.tuneParams.coopmatAccF16Warp,
          numInChannelsPadded,
          ctx.subgroupSize);
      } else if(winogradGemmVariant == WinogradGemmVariant::Coopmat) {
        winogradGemmKernel = VulkanKernels::WinogradGemmCoopmat::build(
          ctx.device,
          ctx.pipelineCache,
          ctx.tuneParams.coopmatBlockSize,
          ctx.tuneParams.coopmatBM,
          ctx.tuneParams.coopmatBN,
          ctx.tuneParams.coopmatBK,
          ctx.tuneParams.coopmatWM,
          ctx.tuneParams.coopmatWN,
          ctx.tuneParams.coopmatTM,
          ctx.tuneParams.coopmatTN,
          ctx.tuneParams.coopmatTK,
          ctx.tuneParams.coopmatWarp,
          numInChannelsPadded,
          ctx.subgroupSize);
      } else if(winogradGemmVariant == WinogradGemmVariant::Dot2) {
        winogradGemmKernel = VulkanKernels::WinogradGemmDot2::build(
          ctx.device,
          ctx.pipelineCache,
          ctx.tuneParams.dot2BlockSize,
          ctx.tuneParams.dot2BM,
          ctx.tuneParams.dot2BN,
          ctx.tuneParams.dot2WM,
          ctx.tuneParams.dot2WN,
          ctx.tuneParams.dot2WMIter,
          ctx.tuneParams.dot2TM,
          ctx.tuneParams.dot2TN,
          ctx.tuneParams.dot2Warp,
          numInChannelsPadded);
      } else if(winogradGemmVariant == WinogradGemmVariant::Dot2AccF16) {
        winogradGemmKernel = VulkanKernels::WinogradGemmDot2AccF16::build(
          ctx.device,
          ctx.pipelineCache,
          ctx.tuneParams.dot2AccF16BlockSize,
          ctx.tuneParams.dot2AccF16BM,
          ctx.tuneParams.dot2AccF16BN,
          ctx.tuneParams.dot2AccF16WM,
          ctx.tuneParams.dot2AccF16WN,
          ctx.tuneParams.dot2AccF16WMIter,
          ctx.tuneParams.dot2AccF16TM,
          ctx.tuneParams.dot2AccF16TN,
          ctx.tuneParams.dot2AccF16Warp,
          numInChannelsPadded);
      } else {
        winogradGemmKernel = VulkanKernels::WinogradGemm::build(
          ctx.device,
          ctx.pipelineCache,
          useFP16,
          ctx.tuneParams.winogradGemmM,
          ctx.tuneParams.winogradGemmN,
          ctx.tuneParams.winogradGemmK,
          ctx.tuneParams.winogradGemmRN,
          numInChannelsPadded);
      }
      winogradUntransformKernel = VulkanKernels::WinogradUntransform::build(
        ctx.device,
        ctx.pipelineCache,
        useFP16,
        inTile,
        outTile,
        convSz,
        ctx.tuneParams.winogradUntransformLocalSizeX,
        ctx.tuneParams.winogradUntransformLocalSizeY,
        addToOutputOnStore);
    } else {
      filterBuf =
        makeWeightBuf(ctx.device, ctx.queue, ctx.queueMutex, ctx.commandPool, ctx.memProps, desc->weights, useFP16);
      conv2dDirectKernel = VulkanKernels::Conv2dDirect::build(ctx.device, ctx.pipelineCache, useFP16);
    }
  }

  ~ConvLayer() {
    gemmStridedKernel.destroy(device);
    winogradTransformKernel.destroy(device);
    winogradGemmKernel.destroy(device);
    winogradUntransformKernel.destroy(device);
    conv2dDirectKernel.destroy(device);
  }

  ConvLayer() = delete;
  ConvLayer(const ConvLayer&) = delete;
  ConvLayer& operator=(const ConvLayer&) = delete;

  std::vector<VkDeviceSize> permanentScratchSlotSizes(int maxBatchSize, size_t elemBytes) const {
    if(numTilesX == 0)
      return {};
    int ntxtyPadded = roundUpToMultipleInt(maxBatchSize * numTilesX * numTilesY, winogradMAlignment);
    testAssert(winogradPackedABM > 0 && winogradPackedABK > 0 && winogradPackedAPadWords >= 0);
    size_t packedStrideA = winogradPackedRowMajorAStrideWords(
      ntxtyPadded, numInChannelsPadded, winogradPackedABM, winogradPackedABK, winogradPackedAPadWords);
    VkDeviceSize ws0Bytes = (VkDeviceSize)inTileXYSize * (VkDeviceSize)packedStrideA * 4 * elemBytes;
    return {
      ws0Bytes,
      (VkDeviceSize)ntxtyPadded * workspaceElts2 * elemBytes,
    };
  }

  void dispatch(
    const CmdCtx& ctx,
    ScratchBuffers* scratch,
    VulkanBuffer* input,
    VulkanBuffer* output,
    int maxBatchSize,
    const BatchNormLayer* fusedBN = nullptr,
    VulkanBuffer* mask = nullptr,
    const ComputeKernel* fusedTransformKernel = nullptr) const {
    const ConvLayer& conv = *this;
    if(conv.convXSize == 1 && conv.convYSize == 1) {
      testAssert(fusedBN == nullptr);
      testAssert(conv.paddedSpatialSize % 4 == 0);
      testAssert(conv.outChannelsPadded1x1 % 8 == 0);
      VulkanKernels::GemmStridedTiled::PC pc = {
        conv.paddedSpatialSize,
        conv.outChannelsPadded1x1,
        conv.outChannels,
        conv.inChannels * (conv.paddedSpatialSize / 4),
        conv.gemmStridedBStride,
        conv.outChannels * (conv.paddedSpatialSize / 4)};  // strideC uses real N
      if(conv.gemmStridedVariant == GemmStridedVariant::Coopmat2AccF16) {
        VulkanKernels::GemmStridedCoopmat2AccF16::dispatch(
          ctx, conv.gemmStridedKernel, input, conv.filterBuf.get(), output, conv.inChannels, pc, maxBatchSize);
      } else if(conv.gemmStridedVariant == GemmStridedVariant::Coopmat2) {
        VulkanKernels::GemmStridedCoopmat2::dispatch(
          ctx, conv.gemmStridedKernel, input, conv.filterBuf.get(), output, conv.inChannels, pc, maxBatchSize);
      } else if(conv.gemmStridedVariant == GemmStridedVariant::CoopmatAccF16) {
        VulkanKernels::GemmStridedCoopmatAccF16::dispatch(
          ctx, conv.gemmStridedKernel, input, conv.filterBuf.get(), output, conv.inChannels, pc, maxBatchSize);
      } else if(conv.gemmStridedVariant == GemmStridedVariant::Coopmat) {
        VulkanKernels::GemmStridedCoopmat::dispatch(
          ctx, conv.gemmStridedKernel, input, conv.filterBuf.get(), output, conv.inChannels, pc, maxBatchSize);
      } else if(conv.gemmStridedVariant == GemmStridedVariant::Dot2) {
        VulkanKernels::GemmStridedDot2::dispatch(
          ctx, conv.gemmStridedKernel, input, conv.filterBuf.get(), output, conv.inChannels, pc, maxBatchSize);
      } else if(conv.gemmStridedVariant == GemmStridedVariant::Dot2AccF16) {
        VulkanKernels::GemmStridedDot2AccF16::dispatch(
          ctx, conv.gemmStridedKernel, input, conv.filterBuf.get(), output, conv.inChannels, pc, maxBatchSize);
      } else {
        VulkanKernels::GemmStridedTiled::dispatch(
          ctx, conv.gemmStridedKernel, input, conv.filterBuf.get(), output, conv.inChannels, pc, maxBatchSize);
      }
    } else if((conv.convXSize == 3 && conv.convYSize == 3) || (conv.convXSize == 5 && conv.convYSize == 5)) {
      VulkanBuffer* ws0 = scratch->permanentSlot(0);
      VulkanBuffer* ws1 = scratch->permanentSlot(1);
      int ntxty = maxBatchSize * conv.numTilesX * conv.numTilesY;
      int numTilesPaddedRun = roundUpToMultipleInt(ntxty, conv.winogradMAlignment);

      if(fusedBN != nullptr) {
        testAssert(fusedTransformKernel != nullptr);
        VulkanKernels::WinogradBNActTransform::PC pc = {
          conv.nnXLen,
          conv.nnYLen,
          conv.numTilesX,
          conv.numTilesY,
          conv.inChannels,
          conv.numInChannelsPadded,
          ntxty,
          numTilesPaddedRun,
          conv.paddedSpatialSize};
        VulkanKernels::WinogradBNActTransform::dispatch(
          ctx,
          *fusedTransformKernel,
          input,
          ws0,
          fusedBN->mergedScaleBuf.get(),
          fusedBN->mergedBiasBuf.get(),
          mask,
          pc);
      } else {
        VulkanKernels::WinogradTransform::PC pc = {
          conv.nnXLen,
          conv.nnYLen,
          conv.numTilesX,
          conv.numTilesY,
          conv.inChannels,
          conv.numInChannelsPadded,
          ntxty,
          numTilesPaddedRun,
          conv.paddedSpatialSize};
        VulkanKernels::WinogradTransform::dispatch(ctx, conv.winogradTransformKernel, input, ws0, pc);
      }
      VulkanHelpers::cmdComputeBarrier(ctx.cmd, ws0->buffer);

      // strideA is the row-major packed A tile stride in vec4 units.
      // strideB is variant-specific: normal Winograd GEMM uses K*(N/4), coopmat uses packed tile words.
      // strideC in vec4 units = (N/4)*M.  M ÷4 by numTilesPaddedRun roundUp.
      testAssert(conv.numInChannelsPadded % 4 == 0);
      testAssert(conv.numOutChannelsPadded % 4 == 0);
      testAssert(numTilesPaddedRun % 4 == 0);
      testAssert(conv.winogradPackedABM > 0 && conv.winogradPackedABK > 0 && conv.winogradPackedAPadWords >= 0);
      size_t packedStrideA = winogradPackedRowMajorAStrideWords(
        numTilesPaddedRun,
        conv.numInChannelsPadded,
        conv.winogradPackedABM,
        conv.winogradPackedABK,
        conv.winogradPackedAPadWords);
      testAssert(packedStrideA <= (size_t)std::numeric_limits<int>::max());
      int strideA = (int)packedStrideA;
      int strideB = conv.numInChannelsPadded * (conv.numOutChannelsPadded / 4);
      if(
        conv.winogradGemmVariant == WinogradGemmVariant::Coopmat ||
        conv.winogradGemmVariant == WinogradGemmVariant::CoopmatAccF16 ||
        conv.winogradGemmVariant == WinogradGemmVariant::Coopmat2AccF16 ||
        conv.winogradGemmVariant == WinogradGemmVariant::Coopmat2) {
        testAssert(conv.winogradCoopmatBStride > 0);
        strideB = conv.winogradCoopmatBStride;
      }
      VulkanKernels::WinogradGemm::PC pcGemm = {
        numTilesPaddedRun,
        conv.numOutChannelsPadded,
        strideA,                                               // strideA, vec4 units
        strideB,                                               // strideB, vec4 units
        (conv.numOutChannelsPadded / 4) * numTilesPaddedRun};  // strideC, vec4 units
      if(conv.winogradGemmVariant == WinogradGemmVariant::Coopmat2AccF16) {
        VulkanKernels::WinogradGemmCoopmat2AccF16::dispatch(
          ctx,
          conv.winogradGemmKernel,
          ws0,
          conv.filterBuf.get(),
          ws1,
          conv.numInChannelsPadded,
          pcGemm,
          conv.inTileXYSize);
      } else if(conv.winogradGemmVariant == WinogradGemmVariant::Coopmat2) {
        VulkanKernels::WinogradGemmCoopmat2::dispatch(
          ctx,
          conv.winogradGemmKernel,
          ws0,
          conv.filterBuf.get(),
          ws1,
          conv.numInChannelsPadded,
          pcGemm,
          conv.inTileXYSize);
      } else if(conv.winogradGemmVariant == WinogradGemmVariant::CoopmatAccF16) {
        VulkanKernels::WinogradGemmCoopmatAccF16::dispatch(
          ctx,
          conv.winogradGemmKernel,
          ws0,
          conv.filterBuf.get(),
          ws1,
          conv.numInChannelsPadded,
          pcGemm,
          conv.inTileXYSize);
      } else if(conv.winogradGemmVariant == WinogradGemmVariant::Coopmat) {
        VulkanKernels::WinogradGemmCoopmat::dispatch(
          ctx,
          conv.winogradGemmKernel,
          ws0,
          conv.filterBuf.get(),
          ws1,
          conv.numInChannelsPadded,
          pcGemm,
          conv.inTileXYSize);
      } else if(conv.winogradGemmVariant == WinogradGemmVariant::Dot2) {
        VulkanKernels::WinogradGemmDot2::dispatch(
          ctx,
          conv.winogradGemmKernel,
          ws0,
          conv.filterBuf.get(),
          ws1,
          conv.numInChannelsPadded,
          pcGemm,
          conv.inTileXYSize);
      } else if(conv.winogradGemmVariant == WinogradGemmVariant::Dot2AccF16) {
        VulkanKernels::WinogradGemmDot2AccF16::dispatch(
          ctx,
          conv.winogradGemmKernel,
          ws0,
          conv.filterBuf.get(),
          ws1,
          conv.numInChannelsPadded,
          pcGemm,
          conv.inTileXYSize);
      } else {
        VulkanKernels::WinogradGemm::dispatch(
          ctx,
          conv.winogradGemmKernel,
          ws0,
          conv.filterBuf.get(),
          ws1,
          conv.numInChannelsPadded,
          pcGemm,
          conv.inTileXYSize);
      }
      VulkanHelpers::cmdComputeBarrier(ctx.cmd, ws1->buffer);

      VulkanKernels::WinogradUntransform::PC pcUnt = {
        conv.nnXLen,
        conv.nnYLen,
        conv.numTilesX,
        conv.numTilesY,
        conv.outChannels,
        conv.numOutChannelsPadded,
        numTilesPaddedRun,
        conv.paddedSpatialSize};
      VulkanKernels::WinogradUntransform::dispatch(
        ctx, conv.winogradUntransformKernel, ws1, output, pcUnt, maxBatchSize);
      VulkanHelpers::cmdComputeWARBarrier(ctx.cmd, ws0->buffer);
      VulkanHelpers::cmdComputeWARBarrier(ctx.cmd, ws1->buffer);
    } else {
      testAssert(fusedBN == nullptr);
      VulkanKernels::Conv2dDirect::PC pc = {
        conv.nnXLen,
        conv.nnYLen,
        conv.outChannels,
        conv.inChannels,
        conv.convXSize / 2,
        conv.convYSize / 2,
        conv.paddedSpatialSize,
        maxBatchSize};
      VulkanKernels::Conv2dDirect::dispatch(ctx, conv.conv2dDirectKernel, input, conv.filterBuf.get(), output, pc);
    }
  }
};

// Abstract interface for all block types in a BlockStack.
struct VulkanBlockVk {
  virtual ~VulkanBlockVk() = default;
  virtual void dispatch(
    const CmdCtx& ctx,
    ScratchBuffers* scratch,
    VulkanBuffer* trunk,
    VulkanBuffer* trunkScratch,
    VulkanBuffer* mask,
    VulkanBuffer* maskSum,
    int maxBatchSize) const = 0;
  virtual std::vector<VkDeviceSize> permanentScratchSlotSizes(int maxBatchSize, size_t elemBytes) const = 0;
};

inline void mergePermanentScratchSlots(std::vector<VkDeviceSize>& dst, const std::vector<VkDeviceSize>& src) {
  for(size_t i = 0; i < src.size(); i++) {
    if(i >= dst.size())
      dst.push_back(src[i]);
    else
      dst[i] = std::max(dst[i], src[i]);
  }
}

struct NormActConv {
  const BatchNormLayer norm;
  const ConvLayer conv;
  const int inChannels;
  const int outChannels;
  VkDevice device;
  ComputeKernel fusedTransformKernel;
  bool canFuse;
  bool fusesResidualAdd;

  NormActConv(
    const VulkanLayerContext& ctx,
    const BatchNormLayerDesc* normDesc,
    const ActivationLayerDesc* actDesc,
    const ConvLayerDesc* convDesc,
    int nnX,
    int nnY,
    int paddedSpatialSize,
    bool useFP16,
    bool addToOutputOnStore = false)
    : norm(ctx, normDesc, actDesc, useFP16),
      conv(ctx, convDesc, nnX, nnY, paddedSpatialSize, useFP16, addToOutputOnStore),
      inChannels(norm.numChannels),
      outChannels(conv.outChannels),
      device(ctx.device),
      canFuse((conv.convXSize == 3 && conv.convYSize == 3) || (conv.convXSize == 5 && conv.convYSize == 5)),
      fusesResidualAdd(addToOutputOnStore && ((conv.convXSize == 1 && conv.convYSize == 1) || canFuse)) {
    static_assert(
      VulkanKernels::GemmStridedTiled::kSingleWriterStore && VulkanKernels::GemmStridedDot2::kSingleWriterStore &&
        VulkanKernels::GemmStridedCoopmat::kSingleWriterStore &&
        VulkanKernels::GemmStridedCoopmatAccF16::kSingleWriterStore &&
        VulkanKernels::GemmStridedCoopmat2::kSingleWriterStore &&
        VulkanKernels::GemmStridedCoopmat2AccF16::kSingleWriterStore,
      "ADD_TO_OUTPUT residual fusion requires single-writer C stores (no split-K)");
    assert(norm.numChannels == conv.inChannels);
    if(canFuse) {
      fusedTransformKernel = VulkanKernels::WinogradBNActTransform::build(
        ctx.device,
        ctx.pipelineCache,
        useFP16,
        conv.inTileXSize,
        conv.outTileXSize,
        conv.convXSize,
        conv.winogradTransformOffset,
        norm.activation,
        ctx.tuneParams.winogradBNActTransformLocalSizeX,
        ctx.tuneParams.winogradBNActTransformLocalSizeY,
        conv.winogradPackedABM,
        conv.winogradPackedABK,
        conv.winogradPackedAPadWords);
    }
  }

  NormActConv() = delete;
  NormActConv(const NormActConv&) = delete;
  NormActConv& operator=(const NormActConv&) = delete;
  ~NormActConv() { fusedTransformKernel.destroy(device); }

  std::vector<VkDeviceSize> permanentScratchSlotSizes(int maxBatchSize, size_t elemBytes) const {
    return conv.permanentScratchSlotSizes(maxBatchSize, elemBytes);
  }

  void dispatch(
    const CmdCtx& ctx,
    VulkanBuffer* input,
    VulkanBuffer* inputScratchOrInput,
    VulkanBuffer* output,
    VulkanBuffer* mask,
    ScratchBuffers* scratch,
    int maxBatchSize) const {
    if(canFuse) {
      conv.dispatch(ctx, scratch, input, output, maxBatchSize, &norm, mask, &fusedTransformKernel);
    } else {
      norm.dispatch(ctx, input, inputScratchOrInput, mask, conv.paddedSpatialSize, maxBatchSize);
      VulkanHelpers::cmdComputeBarrier(ctx.cmd, inputScratchOrInput->buffer);
      conv.dispatch(ctx, scratch, inputScratchOrInput, output, maxBatchSize);
    }
  }
};

struct ResidualBlockVk : VulkanBlockVk {
  const std::string name;
  const NormActConv normActConv1;
  const NormActConv normActConv2;
  const int paddedSpatialSize;
  VkDevice device;
  ComputeKernel addPointwiseKernel;

  ResidualBlockVk(
    const VulkanLayerContext& ctx,
    const ResidualBlockDesc* desc,
    int nnX,
    int nnY,
    int paddedSpatialSize_,
    bool useFP16)
    : name(desc->name),
      normActConv1(ctx, &desc->preBN, &desc->preActivation, &desc->regularConv, nnX, nnY, paddedSpatialSize_, useFP16),
      normActConv2(
        ctx,
        &desc->midBN,
        &desc->midActivation,
        &desc->finalConv,
        nnX,
        nnY,
        paddedSpatialSize_,
        useFP16,
        true),
      paddedSpatialSize(paddedSpatialSize_),
      device(ctx.device) {
    addPointwiseKernel = VulkanKernels::AddPointwise::build(ctx.device, ctx.pipelineCache, useFP16);
  }

  ResidualBlockVk() = delete;
  ResidualBlockVk(const ResidualBlockVk&) = delete;
  ResidualBlockVk& operator=(const ResidualBlockVk&) = delete;
  ~ResidualBlockVk() { addPointwiseKernel.destroy(device); }

  std::vector<VkDeviceSize> permanentScratchSlotSizes(int maxBatchSize, size_t elemBytes) const override {
    std::vector<VkDeviceSize> slots = normActConv1.permanentScratchSlotSizes(maxBatchSize, elemBytes);
    mergePermanentScratchSlots(slots, normActConv2.permanentScratchSlotSizes(maxBatchSize, elemBytes));
    return slots;
  }

  void dispatch(
    const CmdCtx& ctx,
    ScratchBuffers* scratch,
    VulkanBuffer* trunk,
    VulkanBuffer* trunkScratch,
    VulkanBuffer* mask,
    int maxBatchSize) const {
    SizedBuf<VulkanBuffer*> mid(scratch->allocator.get(), scratch->getBufSizeXY(normActConv1.outChannels));
    normActConv1.dispatch(ctx, trunk, trunkScratch, mid.buf, mask, scratch, maxBatchSize);
    VulkanHelpers::cmdComputeBarrier(ctx.cmd, mid.buf->buffer);
    if(normActConv2.fusesResidualAdd) {
      VulkanHelpers::cmdComputeWARBarrier(ctx.cmd, trunk->buffer);
      normActConv2.dispatch(ctx, mid.buf, mid.buf, trunk, mask, scratch, maxBatchSize);
    } else {
      normActConv2.dispatch(ctx, mid.buf, mid.buf, trunkScratch, mask, scratch, maxBatchSize);
      VulkanHelpers::cmdComputeBarrier(ctx.cmd, trunkScratch->buffer);
      int totalElts = maxBatchSize * normActConv2.outChannels * paddedSpatialSize;
      {
        VulkanKernels::AddPointwise::PC pc = {totalElts};
        VulkanKernels::AddPointwise::dispatch(ctx, addPointwiseKernel, trunk, trunkScratch, pc);
      }
    }
    VulkanHelpers::cmdComputeBarrier(ctx.cmd, trunk->buffer);
  }

  void dispatch(
    const CmdCtx& ctx,
    ScratchBuffers* scratch,
    VulkanBuffer* trunk,
    VulkanBuffer* trunkScratch,
    VulkanBuffer* mask,
    VulkanBuffer* /*maskSum*/,
    int maxBatchSize) const override {
    dispatch(ctx, scratch, trunk, trunkScratch, mask, maxBatchSize);
  }
};

struct GlobalPoolingResidualBlockVk : VulkanBlockVk {
  const std::string name;
  const BatchNormLayer preBN;
  const ConvLayer regularConv;
  const ConvLayer gpoolConv;
  const BatchNormLayer gpoolBN;
  const MatMulLayer gpoolToBiasMul;
  const NormActConv normActConv2;

  const int paddedSpatialSize;
  const int regularChannels;
  const int gpoolChannels;
  VkDevice device;
  ComputeKernel gpoolKernel;
  ComputeKernel addChannelBiasesKernel;
  ComputeKernel addPointwiseKernel;

  GlobalPoolingResidualBlockVk(
    const VulkanLayerContext& ctx,
    const GlobalPoolingResidualBlockDesc* desc,
    int nnX,
    int nnY,
    int paddedSpatialSize_,
    bool useFP16)
    : name(desc->name),
      preBN(ctx, &desc->preBN, &desc->preActivation, useFP16),
      regularConv(ctx, &desc->regularConv, nnX, nnY, paddedSpatialSize_, useFP16),
      gpoolConv(ctx, &desc->gpoolConv, nnX, nnY, paddedSpatialSize_, useFP16),
      gpoolBN(ctx, &desc->gpoolBN, &desc->gpoolActivation, useFP16),
      gpoolToBiasMul(ctx, &desc->gpoolToBiasMul),
      normActConv2(
        ctx,
        &desc->midBN,
        &desc->midActivation,
        &desc->finalConv,
        nnX,
        nnY,
        paddedSpatialSize_,
        useFP16,
        true),
      paddedSpatialSize(paddedSpatialSize_),
      regularChannels(desc->regularConv.outChannels),
      gpoolChannels(desc->gpoolConv.outChannels),
      device(ctx.device) {
    gpoolKernel =
      VulkanKernels::GPoolReduction::build(ctx.device, ctx.pipelineCache, useFP16, ctx.tuneParams.gpoolXystride);
    addChannelBiasesKernel = VulkanKernels::AddChannelBiases::build(ctx.device, ctx.pipelineCache, useFP16);
    addPointwiseKernel = VulkanKernels::AddPointwise::build(ctx.device, ctx.pipelineCache, useFP16);
  }

  GlobalPoolingResidualBlockVk() = delete;
  GlobalPoolingResidualBlockVk(const GlobalPoolingResidualBlockVk&) = delete;
  GlobalPoolingResidualBlockVk& operator=(const GlobalPoolingResidualBlockVk&) = delete;

  ~GlobalPoolingResidualBlockVk() {
    gpoolKernel.destroy(device);
    addChannelBiasesKernel.destroy(device);
    addPointwiseKernel.destroy(device);
  }

  std::vector<VkDeviceSize> permanentScratchSlotSizes(int maxBatchSize, size_t elemBytes) const override {
    std::vector<VkDeviceSize> slots = regularConv.permanentScratchSlotSizes(maxBatchSize, elemBytes);
    mergePermanentScratchSlots(slots, gpoolConv.permanentScratchSlotSizes(maxBatchSize, elemBytes));
    mergePermanentScratchSlots(slots, normActConv2.permanentScratchSlotSizes(maxBatchSize, elemBytes));
    return slots;
  }

  void dispatch(
    const CmdCtx& ctx,
    ScratchBuffers* scratch,
    VulkanBuffer* trunk,
    VulkanBuffer* trunkScratch,
    VulkanBuffer* mask,
    VulkanBuffer* maskSum,
    int maxBatchSize) const override {
    SizedBuf<VulkanBuffer*> regularOut(scratch->allocator.get(), scratch->getBufSizeXY(regularChannels));
    SizedBuf<VulkanBuffer*> gpoolOut(scratch->allocator.get(), scratch->getBufSizeXY(gpoolChannels));
    SizedBuf<VulkanBuffer*> gpoolConcat(scratch->allocator.get(), scratch->getBufSizeFloat(gpoolChannels * 3));
    SizedBuf<VulkanBuffer*> gpoolBias(scratch->allocator.get(), scratch->getBufSizeFloat(regularChannels));

    preBN.dispatch(ctx, trunk, trunkScratch, mask, paddedSpatialSize, maxBatchSize);
    VulkanHelpers::cmdComputeBarrier(ctx.cmd, trunkScratch->buffer);
    regularConv.dispatch(ctx, scratch, trunkScratch, regularOut.buf, maxBatchSize);
    gpoolConv.dispatch(ctx, scratch, trunkScratch, gpoolOut.buf, maxBatchSize);
    VulkanHelpers::cmdComputeBarrier(ctx.cmd, gpoolOut.buf->buffer);
    gpoolBN.dispatch(ctx, gpoolOut.buf, gpoolOut.buf, mask, paddedSpatialSize, maxBatchSize);
    VulkanHelpers::cmdComputeBarrier(ctx.cmd, gpoolOut.buf->buffer);

    {
      VulkanKernels::GPoolReduction::PC pc = {gpoolChannels, paddedSpatialSize};
      VulkanKernels::GPoolReduction::dispatch(
        ctx, gpoolKernel, gpoolOut.buf, gpoolConcat.buf, mask, maskSum, pc, maxBatchSize);
    }
    VulkanHelpers::cmdComputeBarrier(ctx.cmd, gpoolConcat.buf->buffer);

    gpoolToBiasMul.dispatch(ctx, gpoolConcat.buf, gpoolBias.buf, maxBatchSize);
    VulkanHelpers::cmdComputeBarrier(ctx.cmd, gpoolBias.buf->buffer);
    VulkanHelpers::cmdComputeBarrier(ctx.cmd, regularOut.buf->buffer);

    VulkanKernels::AddChannelBiases::PC pcBias = {maxBatchSize * regularChannels, paddedSpatialSize};
    VulkanKernels::AddChannelBiases::dispatch(ctx, addChannelBiasesKernel, regularOut.buf, gpoolBias.buf, pcBias);
    VulkanHelpers::cmdComputeBarrier(ctx.cmd, regularOut.buf->buffer);

    if(normActConv2.fusesResidualAdd) {
      VulkanHelpers::cmdComputeWARBarrier(ctx.cmd, trunk->buffer);
      normActConv2.dispatch(ctx, regularOut.buf, regularOut.buf, trunk, mask, scratch, maxBatchSize);
    } else {
      normActConv2.dispatch(ctx, regularOut.buf, regularOut.buf, trunkScratch, mask, scratch, maxBatchSize);
      VulkanHelpers::cmdComputeBarrier(ctx.cmd, trunkScratch->buffer);

      int totalElts = maxBatchSize * normActConv2.outChannels * paddedSpatialSize;
      {
        VulkanKernels::AddPointwise::PC pc = {totalElts};
        VulkanKernels::AddPointwise::dispatch(ctx, addPointwiseKernel, trunk, trunkScratch, pc);
      }
    }
    VulkanHelpers::cmdComputeBarrier(ctx.cmd, trunk->buffer);
  }
};

// ---------------------------------------------------------------------------
// Transformer / nested-bottleneck blocks, trunk, and heads.
//
// Bodies are defined out-of-line in vulkanlayers.cpp. These are declared here so
// the Model orchestration in vulkanbackend.cpp can hold and construct them.
// ---------------------------------------------------------------------------

struct BlockStackVk;

struct NestedBottleneckResidualBlockVk : VulkanBlockVk {
  const std::string name;
  const NormActConv normActConv1;
  std::unique_ptr<BlockStackVk> blocks;
  const NormActConv normActConv2;
  const int paddedSpatialSize;
  // Owned residual-add kernel.
  VkDevice device;
  ComputeKernel addPointwiseKernel;

  NestedBottleneckResidualBlockVk(
    const VulkanLayerContext& ctx,
    const NestedBottleneckResidualBlockDesc* desc,
    int nnX,
    int nnY,
    int paddedSpatialSize_,
    bool useFP16);  // defined after BlockStackVk

  NestedBottleneckResidualBlockVk() = delete;
  NestedBottleneckResidualBlockVk(const NestedBottleneckResidualBlockVk&) = delete;
  NestedBottleneckResidualBlockVk& operator=(const NestedBottleneckResidualBlockVk&) = delete;

  ~NestedBottleneckResidualBlockVk() { addPointwiseKernel.destroy(device); }

  void dispatch(
    const CmdCtx& ctx,
    ScratchBuffers* scratch,
    VulkanBuffer* trunk,
    VulkanBuffer* trunkScratch,
    VulkanBuffer* mask,
    VulkanBuffer* maskSum,
    int maxBatchSize) const override;  // defined after BlockStackVk

  std::vector<VkDeviceSize> permanentScratchSlotSizes(int maxBatchSize, size_t elemBytes)
    const override;  // defined after BlockStackVk
};

struct SGFMetadataEncoderVk {
  const std::string name;
  const MatMulLayer mul1;
  const MatBiasLayer bias1;
  const MatMulLayer mul2;
  const MatBiasLayer bias2;
  const MatMulLayer mul3;

  SGFMetadataEncoderVk(const VulkanLayerContext& ctx, const SGFMetadataEncoderDesc* desc);

  SGFMetadataEncoderVk() = delete;
  SGFMetadataEncoderVk(const SGFMetadataEncoderVk&) = delete;
  SGFMetadataEncoderVk& operator=(const SGFMetadataEncoderVk&) = delete;

  void dispatch(const CmdCtx& ctx, ScratchBuffers* scratch, VulkanBuffer* input, VulkanBuffer* output, int maxBatchSize)
    const;
};

// RMSNormVk: apply RMSNorm (spatial or non-spatial)
struct RMSNormVk {
  const std::string name;
  const int numChannels;
  const float epsilon;
  const bool spatial;
  const int activation;
  const int paddedSpatialSize;
  VBuf gammaBuf;
  VBuf betaBuf;
  VBuf actScaleBuf;  // all-ones, FP16 if useFP16 — only allocated when activation != IDENTITY
  VBuf actBiasBuf;   // all-zeros, FP16 if useFP16 — only allocated when activation != IDENTITY
  // Owned SILU scale-bias-mask-activation kernel, built only when activation
  // is SILU (the only post-RMSNorm activation that this dispatches through
  // ScaleBiasMaskAct). Empty for IDENTITY or other activations.
  VkDevice device;
  ComputeKernel siluKernel;
  // Owned per-position RMSNorm kernel (built for the non-spatial path) and the
  // three spatial-RMSNorm pass kernels as one array (built for the spatial path;
  // pass2 has no FP16 mode). Only the relevant set is populated; destroy() is
  // null-safe.
  ComputeKernel rmsNormKernel;
  std::array<ComputeKernel, 3> spatialKernels;

  RMSNormVk(
    const VulkanLayerContext& ctx,
    const RMSNormLayerDesc* desc,
    int activation_,
    int paddedSpatialSize_,
    bool useFP16);

  RMSNormVk() = delete;
  RMSNormVk(const RMSNormVk&) = delete;
  RMSNormVk& operator=(const RMSNormVk&) = delete;

  ~RMSNormVk() {
    siluKernel.destroy(device);
    rmsNormKernel.destroy(device);
    for(auto& k: spatialKernels)
      k.destroy(device);
  }

  // Spatial RMSNorm uses two permanent scratch slots (pass1 partials, pass2 scalar).
  // Non-spatial uses no permanent scratch.
  std::vector<VkDeviceSize> permanentScratchSlotSizes(int maxBatchSize) const;

  void dispatch(
    const CmdCtx& ctx,
    VulkanBuffer* input,
    VulkanBuffer* output,
    VulkanBuffer* mask,
    VulkanBuffer* maskSum,
    ScratchBuffers* scratch,
    int maxBatchSize) const;
};

// TransformerRMSNormLayerVk: per-position RMSNorm, no bias (zeros beta)
struct TransformerRMSNormLayerVk {
  const std::string name;
  const int numChannels;
  const float epsilon;
  const int paddedSpatialSize;
  VBuf weightBuf;
  VBuf zeroBetaBuf;
  // Owned per-position RMSNorm kernel. This shader has non-trivial inline
  // build (spec constants + FP16-aware bindingMap), built here per-instance
  // and shared by all transformer blocks through the device pipeline cache.
  VkDevice device;
  ComputeKernel rmsNormKernel;

  TransformerRMSNormLayerVk(
    const VulkanLayerContext& ctx,
    const TransformerRMSNormDesc* desc,
    int paddedSpatialSize_,
    bool useFP16);

  TransformerRMSNormLayerVk() = delete;
  TransformerRMSNormLayerVk(const TransformerRMSNormLayerVk&) = delete;
  TransformerRMSNormLayerVk& operator=(const TransformerRMSNormLayerVk&) = delete;

  ~TransformerRMSNormLayerVk() { rmsNormKernel.destroy(device); }

  void dispatch(const CmdCtx& ctx, VulkanBuffer* input, VulkanBuffer* output, VulkanBuffer* mask, int maxBatchSize)
    const;
};

// TransformerMatMulLayerVk: per-position linear projection (1x1 spatial conv)
struct TransformerMatMulLayerVk {
  const std::string name;
  const int inChannels;
  const int outChannels;
  const int outChannelsPadded;  // roundUp(outChannels, 4) for vec4 B-tile alignment
  const int paddedSpatialSize;
  const bool addToOutput;
  VBuf filterBuf;
  // Owned strided-GEMM kernel for this per-position projection.
  VkDevice device;
  ComputeKernel gemmStridedKernel;
  GemmStridedVariant gemmStridedVariant;
  int gemmStridedBStride;

  TransformerMatMulLayerVk(
    const VulkanLayerContext& ctx,
    const MatMulLayerDesc* desc,
    int paddedSpatialSize_,
    bool useFP16,
    bool addToOutput_ = false);

  TransformerMatMulLayerVk() = delete;
  TransformerMatMulLayerVk(const TransformerMatMulLayerVk&) = delete;
  TransformerMatMulLayerVk& operator=(const TransformerMatMulLayerVk&) = delete;

  ~TransformerMatMulLayerVk() { gemmStridedKernel.destroy(device); }

  // Apply as 1x1 spatial convolution: [N, inC, padXY] -> [N, outC, padXY]
  void dispatch(const CmdCtx& ctx, VulkanBuffer* input, VulkanBuffer* output, int maxBatchSize) const;
};

struct TransformerAttentionBlockVk : VulkanBlockVk {
  const std::string name;
  const int numHeads;
  const int numKVHeads;
  const int qHeadDim;
  const int vHeadDim;
  const bool useRope;
  const bool learnableRope;
  const int inChannels;
  const int paddedSpatialSize;

  TransformerRMSNormLayerVk preLN;
  TransformerMatMulLayerVk qProj;
  TransformerMatMulLayerVk kProj;
  TransformerMatMulLayerVk vProj;
  TransformerMatMulLayerVk outProj;

  // RoPE tables (always FP32)
  VBuf ropeCosTable;
  VBuf ropeSinTable;
  VkDevice device;
  // Owned per-(qHeadDim,vHeadDim) tiled attention kernel, built
  // once at model-load time; the shared pipeline cache amortizes compilation.
  ComputeKernel attentionKernel;

  TransformerAttentionBlockVk(
    const VulkanLayerContext& ctx,
    const TransformerAttentionDesc* desc,
    int nnX,
    int nnY,
    int paddedSpatialSize_,
    bool useFP16);

  TransformerAttentionBlockVk() = delete;
  TransformerAttentionBlockVk(const TransformerAttentionBlockVk&) = delete;
  TransformerAttentionBlockVk& operator=(const TransformerAttentionBlockVk&) = delete;

  ~TransformerAttentionBlockVk() { attentionKernel.destroy(device); }

  std::vector<VkDeviceSize> permanentScratchSlotSizes(
    [[maybe_unused]] int maxBatchSize,
    [[maybe_unused]] size_t elemBytes) const override {
    return {};  // transformer blocks use only dynamic SizedBuf allocations
  }

  void dispatch(
    const CmdCtx& ctx,
    ScratchBuffers* scratch,
    VulkanBuffer* trunk,
    VulkanBuffer* trunkScratch,
    VulkanBuffer* mask,
    int maxBatchSize) const;

  void dispatch(
    const CmdCtx& ctx,
    ScratchBuffers* scratch,
    VulkanBuffer* trunk,
    VulkanBuffer* trunkScratch,
    VulkanBuffer* mask,
    VulkanBuffer* /*maskSum*/,
    int maxBatchSize) const override {
    dispatch(ctx, scratch, trunk, trunkScratch, mask, maxBatchSize);
  }
};

struct TransformerFFNBlockVk : VulkanBlockVk {
  const std::string name;
  const int numChannels;
  const int ffnChannels;
  const int paddedSpatialSize;

  TransformerRMSNormLayerVk preLN;
  TransformerMatMulLayerVk linear1;
  std::unique_ptr<TransformerMatMulLayerVk> linearGate;
  TransformerMatMulLayerVk linear2;
  VkDevice device;
  // Owned SwiGLU activation kernel.
  ComputeKernel swiGLUKernel;

  TransformerFFNBlockVk(
    const VulkanLayerContext& ctx,
    const TransformerFFNDesc* desc,
    int paddedSpatialSize_,
    bool useFP16);

  TransformerFFNBlockVk() = delete;
  TransformerFFNBlockVk(const TransformerFFNBlockVk&) = delete;
  TransformerFFNBlockVk& operator=(const TransformerFFNBlockVk&) = delete;

  ~TransformerFFNBlockVk() { swiGLUKernel.destroy(device); }

  std::vector<VkDeviceSize> permanentScratchSlotSizes(
    [[maybe_unused]] int maxBatchSize,
    [[maybe_unused]] size_t elemBytes) const override {
    return {};  // transformer blocks use only dynamic SizedBuf allocations
  }

  void dispatch(
    const CmdCtx& ctx,
    ScratchBuffers* scratch,
    VulkanBuffer* trunk,
    VulkanBuffer* trunkScratch,
    VulkanBuffer* mask,
    int maxBatchSize) const;

  void dispatch(
    const CmdCtx& ctx,
    ScratchBuffers* scratch,
    VulkanBuffer* trunk,
    VulkanBuffer* trunkScratch,
    VulkanBuffer* mask,
    VulkanBuffer* /*maskSum*/,
    int maxBatchSize) const override {
    dispatch(ctx, scratch, trunk, trunkScratch, mask, maxBatchSize);
  }
};

struct BlockStackVk {
  int numBlocks;
  int trunkNumChannels;
  std::vector<std::unique_ptr<VulkanBlockVk>> blocks;

  BlockStackVk() = delete;
  BlockStackVk(const BlockStackVk&) = delete;
  BlockStackVk& operator=(const BlockStackVk&) = delete;

  BlockStackVk(
    const VulkanLayerContext& ctx,
    const std::vector<std::pair<int, unique_ptr_void>>& descBlocks,
    int nBlocks,
    int trunkChannels,
    int nnX,
    int nnY,
    int paddedSpatialSize,
    bool useFP16);

  void dispatch(
    const CmdCtx& ctx,
    ScratchBuffers* scratch,
    VulkanBuffer* trunk,
    VulkanBuffer* trunkScratch,
    VulkanBuffer* mask,
    VulkanBuffer* maskSum,
    int maxBatchSize) const;

  std::vector<VkDeviceSize> permanentScratchSlotSizes(int maxBatchSize, size_t elemBytes) const;
};

struct TrunkVk {
  const std::string name;
  const int modelVersion;
  const int trunkNumChannels;
  const int trunkNormKind;
  const int paddedSpatialSize;

  std::unique_ptr<ConvLayer> initialConv;
  std::unique_ptr<MatMulLayer> initialMatMul;
  std::unique_ptr<SGFMetadataEncoderVk> sgfMetadataEncoder;
  std::unique_ptr<BlockStackVk> blocks;
  std::unique_ptr<BatchNormLayer> trunkTipBN;
  std::unique_ptr<RMSNormVk> trunkTipRMSNorm;
  // Owned channel-bias add for the initialMatMul / SGF metadata residuals.
  VkDevice device;
  ComputeKernel addChannelBiasesKernel;

  TrunkVk() = delete;
  TrunkVk(const TrunkVk&) = delete;
  TrunkVk& operator=(const TrunkVk&) = delete;

  ~TrunkVk() { addChannelBiasesKernel.destroy(device); }

  std::vector<VkDeviceSize> permanentScratchSlotSizes(int maxBatchSize, size_t elemBytes) const;

  TrunkVk(const VulkanLayerContext& ctx, const TrunkDesc* desc, int nnX, int nnY, int paddedSpatialSize_, bool useFP16);

  void dispatch(
    const CmdCtx& ctx,
    ScratchBuffers* scratch,
    VulkanBuffer* input,
    VulkanBuffer* inputGlobal,
    VulkanBuffer* inputMeta,
    VulkanBuffer* trunk,
    VulkanBuffer* mask,
    VulkanBuffer* maskSum,
    int maxBatchSize) const;
};

struct PolicyHeadVk {
  const std::string name;
  const int modelVersion;
  const int paddedSpatialSize;
  const int p1Channels;
  const int g1Channels;
  const int p2Channels;

  std::unique_ptr<ConvLayer> p1Conv;
  std::unique_ptr<ConvLayer> g1Conv;
  std::unique_ptr<BatchNormLayer> g1BN;
  std::unique_ptr<MatMulLayer> gpoolToBiasMul;
  std::unique_ptr<BatchNormLayer> p1BN;
  std::unique_ptr<ConvLayer> p2Conv;
  std::unique_ptr<MatMulLayer> gpoolToPassMul;
  std::unique_ptr<MatBiasLayer> gpoolToPassBias;
  std::unique_ptr<MatMulLayer> gpoolToPassMul2;
  // Owned global-pool and channel-bias add kernels.
  VkDevice device;
  ComputeKernel gpoolKernel;
  ComputeKernel addChannelBiasesKernel;

  PolicyHeadVk() = delete;
  PolicyHeadVk(const PolicyHeadVk&) = delete;
  PolicyHeadVk& operator=(const PolicyHeadVk&) = delete;

  ~PolicyHeadVk() {
    gpoolKernel.destroy(device);
    addChannelBiasesKernel.destroy(device);
  }

  std::vector<VkDeviceSize> permanentScratchSlotSizes(int maxBatchSize, size_t elemBytes) const;

  PolicyHeadVk(
    const VulkanLayerContext& ctx,
    const PolicyHeadDesc* desc,
    int nnX,
    int nnY,
    int paddedSpatialSize_,
    bool useFP16);

  void dispatch(
    const CmdCtx& ctx,
    ScratchBuffers* scratch,
    VulkanBuffer* trunk,
    VulkanBuffer* mask,
    VulkanBuffer* maskSum,
    VulkanBuffer* policyPass,
    VulkanBuffer* policy,
    int maxBatchSize) const;
};

struct ValueHeadVk {
  std::string name;
  int modelVersion;
  int paddedSpatialSize;
  int v1Channels;
  int v2Channels;
  int valueChannels;
  int scoreValueChannels;
  int ownershipChannels;

  std::unique_ptr<ConvLayer> v1Conv;
  std::unique_ptr<BatchNormLayer> v1BN;
  std::unique_ptr<MatMulLayer> v2Mul;
  std::unique_ptr<MatBiasLayer> v2Bias;
  std::unique_ptr<MatMulLayer> v3Mul;
  std::unique_ptr<MatBiasLayer> v3Bias;
  std::unique_ptr<MatMulLayer> sv3Mul;
  std::unique_ptr<MatBiasLayer> sv3Bias;
  std::unique_ptr<ConvLayer> vOwnershipConv;
  // Owned value-head pool kernel.
  VkDevice device;
  ComputeKernel valueHeadPoolKernel;

  ValueHeadVk() = delete;
  ValueHeadVk(const ValueHeadVk&) = delete;
  ValueHeadVk& operator=(const ValueHeadVk&) = delete;

  ~ValueHeadVk() { valueHeadPoolKernel.destroy(device); }

  std::vector<VkDeviceSize> permanentScratchSlotSizes(int maxBatchSize, size_t elemBytes) const;

  ValueHeadVk(
    const VulkanLayerContext& ctx,
    const ValueHeadDesc* desc,
    int nnX,
    int nnY,
    int paddedSpatialSize_,
    bool useFP16);

  void dispatch(
    const CmdCtx& ctx,
    ScratchBuffers* scratch,
    VulkanBuffer* trunk,
    VulkanBuffer* mask,
    VulkanBuffer* maskSum,
    VulkanBuffer* value,
    VulkanBuffer* scoreValue,
    VulkanBuffer* ownership,
    int maxBatchSize) const;
};

#endif  // USE_VULKAN_BACKEND
#endif  // NEURALNET_VULKAN_LAYERS_H_

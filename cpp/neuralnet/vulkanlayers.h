#ifndef NEURALNET_VULKAN_LAYERS_H_
#define NEURALNET_VULKAN_LAYERS_H_

#ifdef USE_VULKAN_BACKEND

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdlib>
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
  // True iff the device supports VK_EXT_subgroup_size_control + compute-stage
  // full subgroups at exactly `subgroupSize`. Shaders that rely on
  // requireFullSubgroups + fixed subgroup size (e.g. the RMSNorm subgroup
  // variants) must gate on this.
  bool canRequireReportedSubgroupSize;
  bool supportsFP16Compute;
  bool supportsDot2F16;
  bool supportsDot2F16AccF16;
  bool supportsCoopmat1F16;
  bool supportsCoopmat1F16AccF16;
  bool supportsCoopmatMaintenance1;
  bool supportsCoopmat2F16;
  bool supportsCoopmat2F16AccF16;
  // Device-reported usable coopmat shapes (f16/f16->f32/f32, subgroup scope).
  // The coopmat TM/TN/TK must match one of these. Empty iff !supportsCoopmat1F16.
  std::vector<CoopmatShape> coopmatShapes;
  std::vector<CoopmatShape> coopmatAccF16Shapes;
  std::vector<Coopmat2FlexShape> coopmat2FlexShapes;
  std::vector<Coopmat2FlexShape> coopmat2AccF16FlexShapes;
  uint32_t coopmat2ReservedSharedBytes;
  uint32_t maxComputeSharedMemorySize;
  VkPipelineCache pipelineCache;
  // Vulkan graphs are always channel-last. Transformer blocks use the same
  // layout as convolutional blocks.
  const bool useNhwc = true;

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
    bool canRequireReportedSubgroupSize_,
    bool supportsFP16Compute_,
    bool supportsDot2F16_,
    bool supportsDot2F16AccF16_,
    bool supportsCoopmat1F16_,
    bool supportsCoopmat1F16AccF16_,
    bool supportsCoopmatMaintenance1_,
    bool supportsCoopmat2F16_,
    bool supportsCoopmat2F16AccF16_,
    std::vector<CoopmatShape> coopmatShapes_,
    std::vector<CoopmatShape> coopmatAccF16Shapes_,
    std::vector<Coopmat2FlexShape> coopmat2FlexShapes_,
    std::vector<Coopmat2FlexShape> coopmat2AccF16FlexShapes_,
    uint32_t coopmat2ReservedSharedBytes_,
    uint32_t maxComputeSharedMemorySize_,
    VkPipelineCache pipelineCache_)
    : device(device_),
      queue(queue_),
      queueMutex(queueMutex_),
      commandPool(commandPool_),
      memProps(memProps_),
      tuneParams(tuneParams_),
      supportsSubgroupShuffleCompute(supportsSubgroupShuffleCompute_),
      subgroupSize(subgroupSize_),
      canRequireReportedSubgroupSize(canRequireReportedSubgroupSize_),
      supportsFP16Compute(supportsFP16Compute_),
      supportsDot2F16(supportsDot2F16_),
      supportsDot2F16AccF16(supportsDot2F16AccF16_),
      supportsCoopmat1F16(supportsCoopmat1F16_),
      supportsCoopmat1F16AccF16(supportsCoopmat1F16AccF16_),
      supportsCoopmatMaintenance1(supportsCoopmatMaintenance1_),
      supportsCoopmat2F16(supportsCoopmat2F16_),
      supportsCoopmat2F16AccF16(supportsCoopmat2F16AccF16_),
      coopmatShapes(std::move(coopmatShapes_)),
      coopmatAccF16Shapes(std::move(coopmatAccF16Shapes_)),
      coopmat2FlexShapes(std::move(coopmat2FlexShapes_)),
      coopmat2AccF16FlexShapes(std::move(coopmat2AccF16FlexShapes_)),
      coopmat2ReservedSharedBytes(coopmat2ReservedSharedBytes_),
      maxComputeSharedMemorySize(maxComputeSharedMemorySize_),
      pipelineCache(pipelineCache_) {}
};

enum class GemmStridedVariant { Tiled, Dot2, Dot2AccF16, Coopmat1, Coopmat1AccF16, Coopmat2, Coopmat2AccF16 };
enum class WinogradGemmVariant { Tiled, Dot2, Dot2AccF16, Coopmat1, Coopmat1AccF16, Coopmat2, Coopmat2AccF16 };

inline constexpr bool vulkanUseNhwc() { return true; }

// Add an FP32 [N, C] bias to a native NHWC activation. The caller is
// responsible for making srcDst/bias visible before this call and for making
// srcDst visible to its later consumer.
inline void dispatchAddChannelBiasesNhwc(
  const CmdCtx& ctx,
  const ComputeKernel& addChannelBiasesNhwcKernel,
  VulkanBuffer* srcDst,
  VulkanBuffer* bias,
  int numChannels,
  int paddedSpatialSize,
  int maxBatchSize) {
  VulkanKernels::AddChannelBiasesNhwc::PC nativePc = {
    maxBatchSize * numChannels, numChannels, paddedSpatialSize};
  VulkanKernels::AddChannelBiasesNhwc::dispatch(ctx, addChannelBiasesNhwcKernel, srcDst, bias, nativePc);
}

inline bool coopmatShapeSupported(const std::vector<CoopmatShape>& shapes, int m, int n, int k) {
  for(const CoopmatShape& shape: shapes) {
    if(shape.m == (uint32_t)m && shape.n == (uint32_t)n && shape.k == (uint32_t)k)
      return true;
  }
  return false;
}

inline bool gemmStridedVariantUsesPackedB(GemmStridedVariant variant) {
  return variant == GemmStridedVariant::Dot2 || variant == GemmStridedVariant::Dot2AccF16 ||
         variant == GemmStridedVariant::Coopmat1 || variant == GemmStridedVariant::Coopmat1AccF16 ||
         variant == GemmStridedVariant::Coopmat2 || variant == GemmStridedVariant::Coopmat2AccF16;
}

inline int gemmStridedPackedBBN(const VulkanTuneParams& tuneParams, GemmStridedVariant variant) {
  if(variant == GemmStridedVariant::Dot2)
    return tuneParams.nhwcStridedDot2BN;
  if(variant == GemmStridedVariant::Dot2AccF16)
    return tuneParams.nhwcStridedDot2AccF16BN;
  if(variant == GemmStridedVariant::Coopmat1)
    return tuneParams.nhwcStridedCoopmat1BN;
  if(variant == GemmStridedVariant::Coopmat1AccF16)
    return tuneParams.nhwcStridedCoopmat1AccF16BN;
  if(variant == GemmStridedVariant::Coopmat2)
    return tuneParams.nhwcStridedCoopmat2BN;
  if(variant == GemmStridedVariant::Coopmat2AccF16)
    return tuneParams.nhwcStridedCoopmat2AccF16BN;
  return 0;
}

inline int gemmStridedPackedBBK(const VulkanTuneParams& tuneParams, GemmStridedVariant variant) {
  if(variant == GemmStridedVariant::Dot2 || variant == GemmStridedVariant::Dot2AccF16)
    return VulkanKernels::DOT2_BK;
  if(variant == GemmStridedVariant::Coopmat1)
    return tuneParams.nhwcStridedCoopmat1BK;
  if(variant == GemmStridedVariant::Coopmat1AccF16)
    return tuneParams.nhwcStridedCoopmat1AccF16BK;
  if(variant == GemmStridedVariant::Coopmat2)
    return tuneParams.nhwcStridedCoopmat2BK;
  if(variant == GemmStridedVariant::Coopmat2AccF16)
    return tuneParams.nhwcStridedCoopmat2AccF16BK;
  return 0;
}

inline int gemmStridedPackedBPadScalars(GemmStridedVariant variant) {
  if(variant == GemmStridedVariant::Dot2 || variant == GemmStridedVariant::Dot2AccF16)
    return STRIDED_DOT2_PACKED_B_PAD_SCALARS;
  if(variant == GemmStridedVariant::Coopmat1)
    return STRIDED_COOPMAT_PACKED_B_PAD_SCALARS;
  if(variant == GemmStridedVariant::Coopmat1AccF16)
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
  if(variant == WinogradGemmVariant::Coopmat1)
    return tuneParams.coopmat1BM;
  if(variant == WinogradGemmVariant::Coopmat1AccF16)
    return tuneParams.coopmat1AccF16BM;
  if(variant == WinogradGemmVariant::Coopmat2)
    return tuneParams.coopmat2BM;
  if(variant == WinogradGemmVariant::Coopmat2AccF16)
    return tuneParams.coopmat2AccF16BM;
  return tuneParams.winogradGemmM;
}

inline int winogradPackedABK(const VulkanTuneParams& tuneParams, WinogradGemmVariant variant) {
  if(variant == WinogradGemmVariant::Dot2 || variant == WinogradGemmVariant::Dot2AccF16)
    return VulkanKernels::DOT2_BK;
  if(variant == WinogradGemmVariant::Coopmat1)
    return tuneParams.coopmat1BK;
  if(variant == WinogradGemmVariant::Coopmat1AccF16)
    return tuneParams.coopmat1AccF16BK;
  if(variant == WinogradGemmVariant::Coopmat2)
    return tuneParams.coopmat2BK;
  if(variant == WinogradGemmVariant::Coopmat2AccF16)
    return tuneParams.coopmat2AccF16BK;
  return tuneParams.winogradGemmK;
}

inline int winogradPackedAPadWords(WinogradGemmVariant variant) {
  if(variant == WinogradGemmVariant::Coopmat1 || variant == WinogradGemmVariant::Coopmat1AccF16)
    return WINOGRAD_COOPMAT_PACKED_PAD_WORDS;
  if(variant == WinogradGemmVariant::Coopmat2 || variant == WinogradGemmVariant::Coopmat2AccF16)
    return WINOGRAD_COOPMAT2_PACKED_PAD_WORDS;
  return WINOGRAD_ROW_MAJOR_A_PAD_WORDS;
}

inline int winogradCoopmatPackedAPadWords(WinogradGemmVariant variant) {
  return winogradPackedAPadWords(variant);
}

inline GemmStridedVariant resolveGemmStridedVariant(const VulkanLayerContext& ctx, bool fp16) {
  const bool selectedAccF16IsAvailable = ctx.tuneParams.nhwcGemmUseCoopmatAccF16 != 0 &&
    (ctx.supportsCoopmat1F16AccF16 || ctx.supportsCoopmat2F16AccF16);
  if(fp16 && !selectedAccF16IsAvailable && ctx.supportsCoopmat2F16 &&
     (ctx.tuneParams.nhwcGemmF32UseCoopmat2 != 0 || !ctx.supportsCoopmat1F16) &&
     ctx.tuneParams.gemmStridedNhwcCoopmat2TunerValueValid != 0)
    return GemmStridedVariant::Coopmat2;
  if(fp16 && !selectedAccF16IsAvailable && ctx.supportsCoopmat1F16 &&
     ctx.tuneParams.gemmStridedNhwcCoopmat1TunerValueValid != 0)
    return GemmStridedVariant::Coopmat1;
  if(fp16 && ctx.supportsCoopmat2F16AccF16 &&
     (ctx.tuneParams.nhwcGemmUseCoopmat2 != 0 || !ctx.supportsCoopmat1F16AccF16) &&
     ctx.tuneParams.gemmStridedNhwcCoopmat2AccF16TunerValueValid != 0)
    return GemmStridedVariant::Coopmat2AccF16;
  if(fp16 && ctx.supportsCoopmat1F16AccF16 &&
     ctx.tuneParams.gemmStridedNhwcCoopmat1AccF16TunerValueValid != 0)
    return GemmStridedVariant::Coopmat1AccF16;
  if(fp16 && ctx.supportsDot2F16AccF16 && ctx.tuneParams.nhwcGemmUseDot2AccF16 != 0 &&
     ctx.tuneParams.gemmStridedNhwcDot2AccF16TunerValueValid != 0)
    return GemmStridedVariant::Dot2AccF16;
  if(fp16 && ctx.supportsDot2F16 && ctx.tuneParams.nhwcGemmUseDot2 != 0 &&
     ctx.tuneParams.gemmStridedNhwcDot2TunerValueValid != 0)
    return GemmStridedVariant::Dot2;
  return GemmStridedVariant::Tiled;
}

inline WinogradGemmVariant resolveWinogradGemmVariant(const VulkanLayerContext& ctx, bool fp16) {
  if(fp16 && ctx.supportsCoopmat2F16AccF16 && ctx.tuneParams.enableWinogradGemmCoopmat2AccF16 != 0)
    return WinogradGemmVariant::Coopmat2AccF16;
  if(fp16 && ctx.supportsCoopmat2F16 && ctx.tuneParams.enableWinogradGemmCoopmat2 != 0)
    return WinogradGemmVariant::Coopmat2;
  if(fp16 && ctx.supportsCoopmat1F16AccF16 && ctx.tuneParams.enableWinogradGemmCoopmat1AccF16 != 0)
    return WinogradGemmVariant::Coopmat1AccF16;
  if(fp16 && ctx.supportsCoopmat1F16 && ctx.tuneParams.enableWinogradGemmCoopmat1 != 0)
    return WinogradGemmVariant::Coopmat1;
  if(fp16 && ctx.supportsDot2F16AccF16 && ctx.tuneParams.enableWinogradGemmDot2AccF16 != 0)
    return WinogradGemmVariant::Dot2AccF16;
  if(fp16 && ctx.supportsDot2F16 && ctx.tuneParams.enableWinogradGemmDot2 != 0)
    return WinogradGemmVariant::Dot2;
  return WinogradGemmVariant::Tiled;
}

inline WinogradGemmVariant resolveNhwcWinogradGemmVariant(const VulkanLayerContext& ctx, bool fp16) {
  return resolveWinogradGemmVariant(ctx, fp16);
}

inline LayerPaddingContract selectedWinogradPaddingContract(
  const VulkanTuneParams& tuneParams,
  WinogradGemmVariant variant) {
  switch(variant) {
    case WinogradGemmVariant::Coopmat2AccF16:
      return VulkanKernels::WinogradGemmCoopmat2AccF16::layerPaddingContract(tuneParams);
    case WinogradGemmVariant::Coopmat2:
      return VulkanKernels::WinogradGemmCoopmat2::layerPaddingContract(tuneParams);
    case WinogradGemmVariant::Coopmat1AccF16:
      return VulkanKernels::WinogradGemmCoopmat1AccF16::layerPaddingContract(tuneParams);
    case WinogradGemmVariant::Coopmat1:
      return VulkanKernels::WinogradGemmCoopmat1::layerPaddingContract(tuneParams);
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
  if(fp16 && ctx.tuneParams.nhwcGemmUseCoopmatAccF16 == 0 && ctx.supportsCoopmat2F16 &&
       ctx.tuneParams.nhwcGemmF32UseCoopmat2 != 0 &&
       ctx.tuneParams.gemmStridedNhwcCoopmat2TunerValueValid != 0)
      return {1, ctx.tuneParams.nhwcStridedCoopmat2BN, 1, false};
    if(fp16 && ctx.tuneParams.nhwcGemmUseCoopmatAccF16 == 0 && ctx.supportsCoopmat1F16 &&
       ctx.tuneParams.gemmStridedNhwcCoopmat1TunerValueValid != 0)
      return {1, ctx.tuneParams.nhwcStridedCoopmat1BN, 1, false};
    if(
      fp16 && ctx.supportsCoopmat2F16AccF16 && ctx.tuneParams.nhwcGemmUseCoopmat2 != 0 &&
      ctx.tuneParams.gemmStridedNhwcCoopmat2AccF16TunerValueValid != 0)
      return {1, ctx.tuneParams.nhwcStridedCoopmat2AccF16BN, 1, false};
    if(
      fp16 && ctx.supportsCoopmat1F16AccF16 &&
      ctx.tuneParams.gemmStridedNhwcCoopmat1AccF16TunerValueValid != 0)
      return {1, ctx.tuneParams.nhwcStridedCoopmat1AccF16BN, 1, false};
    if(
      fp16 && ctx.supportsDot2F16AccF16 && ctx.tuneParams.nhwcGemmUseDot2AccF16 != 0 &&
      ctx.tuneParams.gemmStridedNhwcDot2AccF16TunerValueValid != 0)
      return {1, ctx.tuneParams.nhwcStridedDot2AccF16BN, 1, false};
    if(
      fp16 && ctx.supportsDot2F16 && ctx.tuneParams.nhwcGemmUseDot2 != 0 &&
      ctx.tuneParams.gemmStridedNhwcDot2TunerValueValid != 0)
    return {1, ctx.tuneParams.nhwcStridedDot2BN, 1, false};
  return VulkanKernels::GemmStridedTiledNhwc::layerPaddingContract(ctx.tuneParams);
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
  ComputeKernel scaleBiasMaskActNhwcKernel;

  BatchNormLayer(
    const VulkanLayerContext& ctx,
    const BatchNormLayerDesc* desc,
    const ActivationLayerDesc* actDesc,
    bool useFP16)
    : name(desc->name),
      numChannels(desc->numChannels),
      activation(actDesc->activation),
      device(ctx.device) {
    mergedScaleBuf =
      makeWeightBuf(ctx.device, ctx.queue, ctx.queueMutex, ctx.commandPool, ctx.memProps, desc->mergedScale, useFP16);
    mergedBiasBuf =
      makeWeightBuf(ctx.device, ctx.queue, ctx.queueMutex, ctx.commandPool, ctx.memProps, desc->mergedBias, useFP16);
    scaleBiasMaskActNhwcKernel = VulkanKernels::ScaleBiasMaskActNhwc::build(
      ctx.device, ctx.pipelineCache, useFP16, activation, numChannels % 4 == 0);
  }

  BatchNormLayer() = delete;
  BatchNormLayer(const BatchNormLayer&) = delete;
  BatchNormLayer& operator=(const BatchNormLayer&) = delete;

  ~BatchNormLayer() {
    scaleBiasMaskActNhwcKernel.destroy(device);
  }

  void dispatch(
    const CmdCtx& ctx,
    ScratchBuffers* scratch,
    VulkanBuffer* input,
    VulkanBuffer* output,
    VulkanBuffer* mask,
    int paddedSpatialSize,
    int maxBatchSize) const {
    VulkanKernels::ScaleBiasMaskActNhwc::PC pc = {numChannels, paddedSpatialSize, maxBatchSize};
    (void)scratch;

    // Every spatial graph buffer is already NHWC. Mask is one value per xy
    // and is layout-independent.
    VulkanKernels::ScaleBiasMaskActNhwc::dispatch(
      ctx, scaleBiasMaskActNhwcKernel, input, output, mergedScaleBuf.get(), mergedBiasBuf.get(), mask, pc);
    return;

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
  // NHWC Winograd untransform store channel-vectorization width: 4 when the
  // *unpadded* outChannels is divisible by 4 (the store's contiguous NHWC
  // channel run is bounded by the physical, not padded, tensor), else 1
  // (scalar fallback). Derived once at ctor time and passed to both build()
  // and dispatch() so the spec constant the kernel was compiled with can
  // never drift from the dispatch grid.
  int nhwcWinogradUntransformOcVec;
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
  ComputeKernel conv2dDirectNhwcKernel;
  ComputeKernel conv3x3ImplicitGemmNhwcVec8Kernel;
  GemmStridedVariant gemmStridedVariant;
  WinogradGemmVariant winogradGemmVariant;

  bool useConv3x3ImplicitGemmNhwcVec8;
  bool useConv3x3WinogradNhwc;
  bool useConv2dDirectNhwc;
  bool useGemmStridedTiledNhwc;
  bool useGemmStridedCoopmatNhwc;
  bool useGemmStridedCoopmat2Nhwc;
  bool useGemmStridedCoopmatAccF16Nhwc;
  bool useGemmStridedCoopmat2AccF16Nhwc;
  bool useGemmStridedDot2Nhwc;
  bool useGemmStridedDot2AccF16Nhwc;
  bool useConv3x3ImplicitGemmCoopmat2NhwcVec8;
  bool useConv3x3ImplicitGemmAccF16NhwcVec8;
  bool gemmStridedAddsToOutput;
  bool conv3x3ImplicitGemmAddsToOutput;
  // Physical NHWC channel stride consumed by this convolution. This differs
  // from inChannels only when a graph boundary explicitly supplies a
  // zero-padded channel tail for the specialized 3x3 kernel.
  int inChannelsPhysical;
  int inChannelsPaddedConv3x3;
  int outChannelsPaddedConv3x3;

  size_t workspaceElts1;
  size_t workspaceElts2;

  ConvLayer(
    const VulkanLayerContext& ctx,
    const ConvLayerDesc* desc,
    int nnX,
    int nnY,
    int paddedSpatialSize_,
    bool useFP16,
    bool addToOutputOnStore = false,
    bool inputMayBeChannelPadded = false)
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
    useConv3x3ImplicitGemmNhwcVec8 = false;
    useConv3x3WinogradNhwc = false;
    useConv2dDirectNhwc = false;
    useGemmStridedTiledNhwc = false;
    useGemmStridedCoopmatNhwc = false;
    useGemmStridedCoopmat2Nhwc = false;
    useGemmStridedCoopmatAccF16Nhwc = false;
    useGemmStridedCoopmat2AccF16Nhwc = false;
    useGemmStridedDot2Nhwc = false;
    useGemmStridedDot2AccF16Nhwc = false;
    useConv3x3ImplicitGemmCoopmat2NhwcVec8 = false;
    useConv3x3ImplicitGemmAccF16NhwcVec8 = false;
    gemmStridedAddsToOutput = false;
    conv3x3ImplicitGemmAddsToOutput = false;
    inChannelsPhysical = inChannels;
    inChannelsPaddedConv3x3 = 0;
    outChannelsPaddedConv3x3 = 0;
    const bool isGemmStrided1x1 = convXSize == 1 && convYSize == 1;
    inChannelsPaddedConv3x3 = roundUpToMultipleInt(inChannels, 8);
    const bool isImplicitSquareConv =
      (convXSize == 3 && convYSize == 3) || (convXSize == 5 && convYSize == 5);
    const bool implicitConvIs5x5 = convXSize == 5 && convYSize == 5;
    const bool conv3x3Coopmat1F32Available = ctx.supportsCoopmat1F16 &&
      (implicitConvIs5x5 ? ctx.tuneParams.conv5x5NhwcCoopmat1TunerValueValid :
                           ctx.tuneParams.conv3x3NhwcCoopmat1TunerValueValid) != 0;
    const bool conv3x3Coopmat2F32Available = ctx.supportsCoopmat2F16 &&
      (implicitConvIs5x5 ? ctx.tuneParams.conv5x5NhwcCoopmat2TunerValueValid :
                           ctx.tuneParams.conv3x3NhwcCoopmat2TunerValueValid) != 0;
    const bool conv3x3Coopmat1AccF16Available = ctx.supportsCoopmat1F16AccF16 &&
      (implicitConvIs5x5 ? ctx.tuneParams.conv5x5NhwcCoopmat1AccF16TunerValueValid :
                           ctx.tuneParams.conv3x3NhwcCoopmat1AccF16TunerValueValid) != 0;
    const bool conv3x3Coopmat2AccF16Available = ctx.supportsCoopmat2F16AccF16 &&
      (implicitConvIs5x5 ? ctx.tuneParams.conv5x5NhwcCoopmat2AccF16TunerValueValid :
                           ctx.tuneParams.conv3x3NhwcCoopmat2AccF16TunerValueValid) != 0;

    // The cached selection is the normal unfiltered FP32 winner. When a GEMM
    // variant filter has left only one family or accumulator type available,
    // use it instead of falling back to Winograd.
    bool conv3x3Coopmat2Selected = implicitConvIs5x5 ?
      ctx.tuneParams.nhwcConv5x5UseCoopmat2 != 0 : ctx.tuneParams.nhwcConv3x3UseCoopmat2 != 0;
    const bool conv3x3Coopmat1Available = conv3x3Coopmat1F32Available || conv3x3Coopmat1AccF16Available;
    const bool conv3x3Coopmat2Available = conv3x3Coopmat2F32Available || conv3x3Coopmat2AccF16Available;
    if(!conv3x3Coopmat1Available && conv3x3Coopmat2Available)
      conv3x3Coopmat2Selected = true;
    else if(conv3x3Coopmat1Available && !conv3x3Coopmat2Available)
      conv3x3Coopmat2Selected = false;
    const bool conv3x3AccF16Selected = conv3x3Coopmat2Selected ?
      !conv3x3Coopmat2F32Available && conv3x3Coopmat2AccF16Available :
      !conv3x3Coopmat1F32Available && conv3x3Coopmat1AccF16Available;

    const int nativeBlockSize = conv3x3Coopmat2Selected ?
      (conv3x3AccF16Selected ? (implicitConvIs5x5 ? ctx.tuneParams.conv5x5NhwcCoopmat2AccF16BlockSize :
                                                        ctx.tuneParams.conv3x3NhwcCoopmat2AccF16BlockSize) :
        (implicitConvIs5x5 ? ctx.tuneParams.conv5x5NhwcCoopmat2BlockSize : ctx.tuneParams.conv3x3NhwcCoopmat2BlockSize)) :
      (conv3x3AccF16Selected ? (implicitConvIs5x5 ? ctx.tuneParams.conv5x5NhwcCoopmat1AccF16BlockSize :
                                                        ctx.tuneParams.conv3x3NhwcCoopmat1AccF16BlockSize) :
        (implicitConvIs5x5 ? ctx.tuneParams.conv5x5NhwcCoopmat1BlockSize : ctx.tuneParams.conv3x3NhwcCoopmat1BlockSize));
    const int nativeBM = conv3x3Coopmat2Selected ?
      (conv3x3AccF16Selected ? (implicitConvIs5x5 ? ctx.tuneParams.conv5x5NhwcCoopmat2AccF16BM :
                                                        ctx.tuneParams.conv3x3NhwcCoopmat2AccF16BM) :
        (implicitConvIs5x5 ? ctx.tuneParams.conv5x5NhwcCoopmat2BM : ctx.tuneParams.conv3x3NhwcCoopmat2BM)) :
      (conv3x3AccF16Selected ? (implicitConvIs5x5 ? ctx.tuneParams.conv5x5NhwcCoopmat1AccF16BM :
                                                        ctx.tuneParams.conv3x3NhwcCoopmat1AccF16BM) :
        (implicitConvIs5x5 ? ctx.tuneParams.conv5x5NhwcCoopmat1BM : ctx.tuneParams.conv3x3NhwcCoopmat1BM));
    const int nativeBN = conv3x3Coopmat2Selected ?
      (conv3x3AccF16Selected ? (implicitConvIs5x5 ? ctx.tuneParams.conv5x5NhwcCoopmat2AccF16BN :
                                                        ctx.tuneParams.conv3x3NhwcCoopmat2AccF16BN) :
        (implicitConvIs5x5 ? ctx.tuneParams.conv5x5NhwcCoopmat2BN : ctx.tuneParams.conv3x3NhwcCoopmat2BN)) :
      (conv3x3AccF16Selected ? (implicitConvIs5x5 ? ctx.tuneParams.conv5x5NhwcCoopmat1AccF16BN :
                                                        ctx.tuneParams.conv3x3NhwcCoopmat1AccF16BN) :
        (implicitConvIs5x5 ? ctx.tuneParams.conv5x5NhwcCoopmat1BN : ctx.tuneParams.conv3x3NhwcCoopmat1BN));
    const int nativeBK = conv3x3Coopmat2Selected ?
      (conv3x3AccF16Selected ? (implicitConvIs5x5 ? ctx.tuneParams.conv5x5NhwcCoopmat2AccF16BK :
                                                        ctx.tuneParams.conv3x3NhwcCoopmat2AccF16BK) :
        (implicitConvIs5x5 ? ctx.tuneParams.conv5x5NhwcCoopmat2BK : ctx.tuneParams.conv3x3NhwcCoopmat2BK)) :
      (conv3x3AccF16Selected ? (implicitConvIs5x5 ? ctx.tuneParams.conv5x5NhwcCoopmat1AccF16BK :
                                                        ctx.tuneParams.conv3x3NhwcCoopmat1AccF16BK) :
        (implicitConvIs5x5 ? ctx.tuneParams.conv5x5NhwcCoopmat1BK : ctx.tuneParams.conv3x3NhwcCoopmat1BK));
    const int nativeWM = conv3x3AccF16Selected ? (implicitConvIs5x5 ? ctx.tuneParams.conv5x5NhwcCoopmat1AccF16WM :
                                                                        ctx.tuneParams.conv3x3NhwcCoopmat1AccF16WM) :
      (implicitConvIs5x5 ? ctx.tuneParams.conv5x5NhwcCoopmat1WM : ctx.tuneParams.conv3x3NhwcCoopmat1WM);
    const int nativeWN = conv3x3AccF16Selected ? (implicitConvIs5x5 ? ctx.tuneParams.conv5x5NhwcCoopmat1AccF16WN :
                                                                        ctx.tuneParams.conv3x3NhwcCoopmat1AccF16WN) :
      (implicitConvIs5x5 ? ctx.tuneParams.conv5x5NhwcCoopmat1WN : ctx.tuneParams.conv3x3NhwcCoopmat1WN);
    const int nativeTM = conv3x3AccF16Selected ? (implicitConvIs5x5 ? ctx.tuneParams.conv5x5NhwcCoopmat1AccF16TM :
                                                                        ctx.tuneParams.conv3x3NhwcCoopmat1AccF16TM) :
      (implicitConvIs5x5 ? ctx.tuneParams.conv5x5NhwcCoopmat1TM : ctx.tuneParams.conv3x3NhwcCoopmat1TM);
    const int nativeTN = conv3x3AccF16Selected ? (implicitConvIs5x5 ? ctx.tuneParams.conv5x5NhwcCoopmat1AccF16TN :
                                                                        ctx.tuneParams.conv3x3NhwcCoopmat1AccF16TN) :
      (implicitConvIs5x5 ? ctx.tuneParams.conv5x5NhwcCoopmat1TN : ctx.tuneParams.conv3x3NhwcCoopmat1TN);
    const int nativeTK = conv3x3AccF16Selected ? (implicitConvIs5x5 ? ctx.tuneParams.conv5x5NhwcCoopmat1AccF16TK :
                                                                        ctx.tuneParams.conv3x3NhwcCoopmat1AccF16TK) :
      (implicitConvIs5x5 ? ctx.tuneParams.conv5x5NhwcCoopmat1TK : ctx.tuneParams.conv3x3NhwcCoopmat1TK);
    const int nativeWarp = conv3x3AccF16Selected ? (implicitConvIs5x5 ? ctx.tuneParams.conv5x5NhwcCoopmat1AccF16Warp :
                                                                          ctx.tuneParams.conv3x3NhwcCoopmat1AccF16Warp) :
      (implicitConvIs5x5 ? ctx.tuneParams.conv5x5NhwcCoopmat1Warp : ctx.tuneParams.conv3x3NhwcCoopmat1Warp);
    const bool conv3x3SelectedTileSupported = conv3x3Coopmat2Selected ?
      (conv3x3AccF16Selected ?
        VulkanKernels::Conv3x3ImplicitGemmCoopmat2AccF16NhwcVec8::isConfigSupported(nativeBlockSize, nativeBM, nativeBN, nativeBK) :
        VulkanKernels::Conv3x3ImplicitGemmCoopmat2AccF32NhwcVec8::isConfigSupported(nativeBlockSize, nativeBM, nativeBN, nativeBK)) :
      (conv3x3AccF16Selected ?
        (VulkanKernels::Conv3x3ImplicitGemmCoopmat1AccF16NhwcVec8::isConfigSupported(
           nativeBlockSize,nativeBM,nativeBN,nativeBK,nativeWM,nativeWN,nativeTM,nativeTN,nativeTK,nativeWarp) &&
         coopmatShapeSupported(ctx.coopmatAccF16Shapes,nativeTM,nativeTN,nativeTK)) :
        (VulkanKernels::Conv3x3ImplicitGemmCoopmat1AccF32NhwcVec8::isConfigSupported(
           nativeBlockSize,nativeBM,nativeBN,nativeBK,nativeWM,nativeWN,nativeTM,nativeTN,nativeTK,nativeWarp) &&
         coopmatShapeSupported(ctx.coopmatShapes,nativeTM,nativeTN,nativeTK))) &&
      ctx.subgroupSize == (uint32_t)nativeWarp;
    const bool conv3x3SelectedTileAligned = paddedSpatialSize % nativeBM == 0 && outChannels % nativeBN == 0;
    const bool canUseConv3x3ImplicitGemmNhwcVec8Base =
      isImplicitSquareConv && useFP16 &&
      (conv3x3Coopmat2Selected ? conv3x3Coopmat2Available : conv3x3Coopmat1Available) &&
      (inChannelsPaddedConv3x3 % 8) == 0 &&
      // Most native graph tensors have an exact physical channel width. The
      // initial graph boundary may instead promise a zero-padded tail so that
      // inputs such as C=22 can use a physical C=24 vector stride.
      (inChannels == inChannelsPaddedConv3x3 || inputMayBeChannelPadded) &&
      conv3x3SelectedTileSupported && conv3x3SelectedTileAligned;
    const bool canUseConv3x3ImplicitGemmNhwcVec8 = canUseConv3x3ImplicitGemmNhwcVec8Base;
    // NHWC Winograd fallback (F(2,3)/F(4,3)/F(2,5)): the fast non-coopmat path
    // for 3x3/5x5 layers. It normally only applies when the coopmat vec8
    // implicit GEMM kernel isn't selected (checked first in the branch chain
    // below).
    const bool canUseConv3x3WinogradNhwc =
      isImplicitSquareConv && !canUseConv3x3ImplicitGemmNhwcVec8;

    // 1x1 convolutions are layout-invariant GEMMs and keep their existing
    // strided-GEMM path. The NHWC direct fallback is for spatial convolutions
    // that cannot use the specialized 3x3 kernel.
    if(!isGemmStrided1x1 && !canUseConv3x3ImplicitGemmNhwcVec8 &&
       !canUseConv3x3WinogradNhwc) {
      useConv2dDirectNhwc = true;
      filterBuf =
        makeWeightBuf(ctx.device, ctx.queue, ctx.queueMutex, ctx.commandPool, ctx.memProps, desc->weights, useFP16);
      conv2dDirectNhwcKernel = VulkanKernels::Conv2dDirectNhwc::build(ctx.device, ctx.pipelineCache, useFP16);
    } else if(isGemmStrided1x1) {
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
        const int packedBN = gemmStridedVariant == GemmStridedVariant::Coopmat1
          ? ctx.tuneParams.nhwcStridedCoopmat1BN
          : gemmStridedVariant == GemmStridedVariant::Coopmat2
          ? ctx.tuneParams.nhwcStridedCoopmat2BN
          : gemmStridedVariant == GemmStridedVariant::Coopmat1AccF16
          ? ctx.tuneParams.nhwcStridedCoopmat1AccF16BN
          : gemmStridedVariant == GemmStridedVariant::Dot2
          ? ctx.tuneParams.nhwcStridedDot2BN
          : gemmStridedVariant == GemmStridedVariant::Dot2AccF16
          ? ctx.tuneParams.nhwcStridedDot2AccF16BN
          : gemmStridedPackedBBN(ctx.tuneParams, gemmStridedVariant);
        const int packedBK = gemmStridedVariant == GemmStridedVariant::Coopmat1
          ? ctx.tuneParams.nhwcStridedCoopmat1BK
          : gemmStridedVariant == GemmStridedVariant::Coopmat2
          ? ctx.tuneParams.nhwcStridedCoopmat2BK
          : gemmStridedVariant == GemmStridedVariant::Coopmat1AccF16
          ? ctx.tuneParams.nhwcStridedCoopmat1AccF16BK
          : (gemmStridedVariant == GemmStridedVariant::Dot2 || gemmStridedVariant == GemmStridedVariant::Dot2AccF16)
          ? (int)VulkanKernels::DOT2_BK
          : gemmStridedPackedBBK(ctx.tuneParams, gemmStridedVariant);
        const int packedPadScalars = gemmStridedPackedBPadScalars(gemmStridedVariant);
        const bool rowMajorPackedB =
          gemmStridedVariant == GemmStridedVariant::Coopmat1 ||
           gemmStridedVariant == GemmStridedVariant::Coopmat1AccF16 ||
           gemmStridedVariant == GemmStridedVariant::Coopmat2 ||
           gemmStridedVariant == GemmStridedVariant::Coopmat2AccF16;
        std::vector<float> packedW = rowMajorPackedB
          ? packStridedGemmBWeightsRowMajor(
              transW, 1, outChannelsPadded1x1, inChannels, packedBN, packedBK, packedPadScalars)
          : packStridedGemmBWeights(
              transW, 1, outChannelsPadded1x1, inChannels, packedBN, packedBK, packedPadScalars);
        filterBuf =
          makeWeightBuf(ctx.device, ctx.queue, ctx.queueMutex, ctx.commandPool, ctx.memProps, packedW, useFP16);
      } else {
        filterBuf =
          makeWeightBuf(ctx.device, ctx.queue, ctx.queueMutex, ctx.commandPool, ctx.memProps, transW, useFP16);
      }
      if(gemmStridedVariant == GemmStridedVariant::Coopmat2AccF16) {
        const int32_t tileM = ctx.tuneParams.nhwcStridedCoopmat2AccF16BM;
        const int32_t tileN = ctx.tuneParams.nhwcStridedCoopmat2AccF16BN;
        const int32_t tileK = ctx.tuneParams.nhwcStridedCoopmat2AccF16BK;
        int32_t aligned = (paddedSpatialSize % tileM == 0 && outChannelsPadded1x1 % tileN == 0) ? 1 : 0;
        int32_t kAligned = (inChannels % tileK == 0) ? 1 : 0;
        useGemmStridedCoopmat2AccF16Nhwc = true;
        gemmStridedAddsToOutput = addToOutputOnStore;
        gemmStridedKernel = VulkanKernels::GemmStridedCoopmat2AccF16Nhwc::build(
          ctx.device, ctx.pipelineCache, ctx.tuneParams.nhwcStridedCoopmat2AccF16BlockSize,
          tileM, tileN, tileK, aligned, kAligned, inChannels, addToOutputOnStore);
      } else if(gemmStridedVariant == GemmStridedVariant::Coopmat2) {
        const int32_t tileM = ctx.tuneParams.nhwcStridedCoopmat2BM;
        const int32_t tileN = ctx.tuneParams.nhwcStridedCoopmat2BN;
        const int32_t tileK = ctx.tuneParams.nhwcStridedCoopmat2BK;
        int32_t aligned = (paddedSpatialSize % tileM == 0 && outChannelsPadded1x1 % tileN == 0) ? 1 : 0;
        int32_t kAligned = (inChannels % tileK == 0) ? 1 : 0;
        useGemmStridedCoopmat2Nhwc = true;
        gemmStridedAddsToOutput = addToOutputOnStore;
        gemmStridedKernel = VulkanKernels::GemmStridedCoopmat2Nhwc::build(
          ctx.device, ctx.pipelineCache, ctx.tuneParams.nhwcStridedCoopmat2BlockSize,
          tileM, tileN, tileK, aligned, kAligned, inChannels, addToOutputOnStore);
      } else if(gemmStridedVariant == GemmStridedVariant::Coopmat1AccF16) {
        const int32_t tileM = ctx.tuneParams.nhwcStridedCoopmat1AccF16BM;
        const int32_t tileN = ctx.tuneParams.nhwcStridedCoopmat1AccF16BN;
        int32_t aligned = (paddedSpatialSize % tileM == 0 && outChannelsPadded1x1 % tileN == 0) ? 1 : 0;
        useGemmStridedCoopmatAccF16Nhwc = true;
        gemmStridedAddsToOutput = addToOutputOnStore;
        gemmStridedKernel = VulkanKernels::GemmStridedCoopmat1AccF16Nhwc::build(
          ctx.device, ctx.pipelineCache,
          ctx.tuneParams.nhwcStridedCoopmat1AccF16BlockSize,
          ctx.tuneParams.nhwcStridedCoopmat1AccF16BM,
          ctx.tuneParams.nhwcStridedCoopmat1AccF16BN,
          ctx.tuneParams.nhwcStridedCoopmat1AccF16BK,
          ctx.tuneParams.nhwcStridedCoopmat1AccF16WM,
          ctx.tuneParams.nhwcStridedCoopmat1AccF16WN,
          ctx.tuneParams.nhwcStridedCoopmat1AccF16TM,
          ctx.tuneParams.nhwcStridedCoopmat1AccF16TN,
          ctx.tuneParams.nhwcStridedCoopmat1AccF16TK,
          ctx.tuneParams.nhwcStridedCoopmat1AccF16Warp,
          aligned, inChannels, ctx.subgroupSize, addToOutputOnStore);
      } else if(gemmStridedVariant == GemmStridedVariant::Coopmat1) {
        // Same padding contract as DOT2: N (outChannelsPadded1x1) is aligned by
        // mergedStridedPaddingContract; M is VULKAN_SPATIAL_ALIGN-pinned; K real.
        // ALIGNED reflects M/N safety only; K_ALIGNED handles the K-tail guard.
        const int32_t tileM = ctx.tuneParams.nhwcStridedCoopmat1BM;
        const int32_t tileN = ctx.tuneParams.nhwcStridedCoopmat1BN;
        int32_t aligned = (paddedSpatialSize % tileM == 0 && outChannelsPadded1x1 % tileN == 0) ? 1 : 0;
        useGemmStridedCoopmatNhwc = true;
        gemmStridedAddsToOutput = addToOutputOnStore;
        gemmStridedKernel = VulkanKernels::GemmStridedCoopmat1Nhwc::build(
          ctx.device, ctx.pipelineCache,
          ctx.tuneParams.nhwcStridedCoopmat1BlockSize, tileM, tileN, ctx.tuneParams.nhwcStridedCoopmat1BK,
          ctx.tuneParams.nhwcStridedCoopmat1WM, ctx.tuneParams.nhwcStridedCoopmat1WN,
          ctx.tuneParams.nhwcStridedCoopmat1TM, ctx.tuneParams.nhwcStridedCoopmat1TN,
          ctx.tuneParams.nhwcStridedCoopmat1TK, ctx.tuneParams.nhwcStridedCoopmat1Warp,
          aligned, inChannels, ctx.subgroupSize, addToOutputOnStore);
      } else if(gemmStridedVariant == GemmStridedVariant::Dot2) {
        // 1x1 strided path: N (outChannelsPadded1x1) is DOT2-aligned by
        // mergedStridedPaddingContract; M (paddedSpatialSize) is globally
        // pinned to VULKAN_SPATIAL_ALIGN and NOT per-variant re-padded; K
        // stays real (kPaddable:false). ALIGNED reflects M/N safety only;
        // K_ALIGNED handles the K-tail guard.
        useGemmStridedDot2Nhwc = true;
        gemmStridedAddsToOutput = addToOutputOnStore;

          const int32_t tileM = ctx.tuneParams.nhwcStridedDot2BM;
          const int32_t tileN = ctx.tuneParams.nhwcStridedDot2BN;
          int32_t aligned = (paddedSpatialSize % tileM == 0 && outChannelsPadded1x1 % tileN == 0) ? 1 : 0;
          int32_t kAligned = (inChannels % (int)VulkanKernels::DOT2_BK == 0) ? 1 : 0;
          gemmStridedKernel = VulkanKernels::GemmStridedDot2Nhwc::build(
            ctx.device,
            ctx.pipelineCache,
            ctx.tuneParams.nhwcStridedDot2BlockSize,
            ctx.tuneParams.nhwcStridedDot2BM,
            ctx.tuneParams.nhwcStridedDot2BN,
            ctx.tuneParams.nhwcStridedDot2WM,
            ctx.tuneParams.nhwcStridedDot2WN,
            ctx.tuneParams.nhwcStridedDot2WMIter,
            ctx.tuneParams.nhwcStridedDot2TM,
            ctx.tuneParams.nhwcStridedDot2TN,
            ctx.tuneParams.nhwcStridedDot2Warp,
            aligned,
            1,
            kAligned,
            inChannels,
            addToOutputOnStore);

      } else if(gemmStridedVariant == GemmStridedVariant::Dot2AccF16) {
        useGemmStridedDot2AccF16Nhwc = true;
        gemmStridedAddsToOutput = addToOutputOnStore;

          const int32_t tileM = ctx.tuneParams.nhwcStridedDot2AccF16BM;
          const int32_t tileN = ctx.tuneParams.nhwcStridedDot2AccF16BN;
          int32_t aligned = (paddedSpatialSize % tileM == 0 && outChannelsPadded1x1 % tileN == 0) ? 1 : 0;
          int32_t kAligned = (inChannels % (int)VulkanKernels::DOT2_BK == 0) ? 1 : 0;
          gemmStridedKernel = VulkanKernels::GemmStridedDot2AccF16Nhwc::build(
            ctx.device,
            ctx.pipelineCache,
            ctx.tuneParams.nhwcStridedDot2AccF16BlockSize,
            ctx.tuneParams.nhwcStridedDot2AccF16BM,
            ctx.tuneParams.nhwcStridedDot2AccF16BN,
            ctx.tuneParams.nhwcStridedDot2AccF16WM,
            ctx.tuneParams.nhwcStridedDot2AccF16WN,
            ctx.tuneParams.nhwcStridedDot2AccF16WMIter,
            ctx.tuneParams.nhwcStridedDot2AccF16TM,
            ctx.tuneParams.nhwcStridedDot2AccF16TN,
            ctx.tuneParams.nhwcStridedDot2AccF16Warp,
            aligned,
            1,
            kAligned,
            inChannels,
            addToOutputOnStore);

      } else {
        useGemmStridedTiledNhwc = true;
        gemmStridedKernel = VulkanKernels::GemmStridedTiledNhwc::build(
          ctx.device, ctx.pipelineCache, useFP16,
          ctx.tuneParams.gemmStridedTiledNhwcLocalSizeX,
          ctx.tuneParams.gemmStridedTiledNhwcLocalSizeY,
          ctx.tuneParams.gemmStridedTiledNhwcTileK,
          ctx.tuneParams.gemmStridedTiledNhwcRN,
          inChannels, addToOutputOnStore);
      }
    } else if(canUseConv3x3ImplicitGemmNhwcVec8) {
      useConv3x3ImplicitGemmNhwcVec8 = true;
      useConv3x3ImplicitGemmCoopmat2NhwcVec8 = conv3x3Coopmat2Selected;
      useConv3x3ImplicitGemmAccF16NhwcVec8 = conv3x3AccF16Selected;
      inChannelsPhysical = inChannelsPaddedConv3x3;
      conv3x3ImplicitGemmAddsToOutput = addToOutputOnStore;
      outChannelsPaddedConv3x3 = outChannels;
      const int convTaps = convXSize * convYSize;
      const int flattenedK = convTaps * inChannelsPaddedConv3x3;
      std::vector<float> filterWeights((size_t)flattenedK * outChannelsPaddedConv3x3, 0.0f);
      for(int tap = 0; tap < convTaps; tap++) {
        const int fy = tap / convXSize;
        const int fx = tap % convXSize;
        for(int ic = 0; ic < inChannels; ic++) {
          const int k = tap * inChannelsPaddedConv3x3 + ic;
          for(int oc = 0; oc < outChannels; oc++) {
            const size_t src = ((size_t)(oc * inChannels + ic) * convYSize + fy) * convXSize + fx;
            filterWeights[(size_t)k * outChannelsPaddedConv3x3 + oc] = desc->weights[src];
          }
        }
      }
      const int packedK = flattenedK;
      filterWeights = packStridedGemmBWeights(
        filterWeights,
        1,
        outChannelsPaddedConv3x3,
        packedK,
        nativeBN,
        nativeBK,
        STRIDED_COOPMAT_PACKED_B_PAD_SCALARS);
      filterBuf =
        makeWeightBuf(ctx.device, ctx.queue, ctx.queueMutex, ctx.commandPool, ctx.memProps, filterWeights, useFP16);
      if(useConv3x3ImplicitGemmCoopmat2NhwcVec8) {
        conv3x3ImplicitGemmNhwcVec8Kernel = useConv3x3ImplicitGemmAccF16NhwcVec8 ?
          VulkanKernels::Conv3x3ImplicitGemmCoopmat2AccF16NhwcVec8::build(
            ctx.device,ctx.pipelineCache,nativeBlockSize,nativeBM,nativeBN,nativeBK,convXSize,packedK,addToOutputOnStore) :
          VulkanKernels::Conv3x3ImplicitGemmCoopmat2AccF32NhwcVec8::build(
            ctx.device,ctx.pipelineCache,nativeBlockSize,nativeBM,nativeBN,nativeBK,convXSize,packedK,addToOutputOnStore);
      } else {
        conv3x3ImplicitGemmNhwcVec8Kernel = useConv3x3ImplicitGemmAccF16NhwcVec8 ?
          VulkanKernels::Conv3x3ImplicitGemmCoopmat1AccF16NhwcVec8::build(
            ctx.device,ctx.pipelineCache,nativeBlockSize,nativeBM,nativeBN,nativeBK,nativeWM,nativeWN,nativeTM,nativeTN,nativeTK,
            nativeWarp,convXSize,flattenedK,addToOutputOnStore,(inChannelsPaddedConv3x3 % nativeBK) == 0,ctx.subgroupSize) :
          VulkanKernels::Conv3x3ImplicitGemmCoopmat1AccF32NhwcVec8::build(
            ctx.device,ctx.pipelineCache,nativeBlockSize,nativeBM,nativeBN,nativeBK,nativeWM,nativeWN,nativeTM,nativeTN,nativeTK,
            nativeWarp,convXSize,flattenedK,addToOutputOnStore,(inChannelsPaddedConv3x3 % nativeBK) == 0,ctx.subgroupSize);
      }
    } else if(canUseConv3x3WinogradNhwc) {
      // NHWC Winograd fallback. GEMM, host weight pre-transform, and packed-A
      // layout are layout-neutral; transform and untransform use dedicated
      // NHWC tuning fields.
      useConv3x3WinogradNhwc = true;
      outTileXSize = (convXSize == 3) ? nhwcWinograd3x3OutTileFor(ctx.tuneParams) : 2;
      outTileYSize = outTileXSize;
      inTileXSize = outTileXSize + convXSize - 1;
      inTileYSize = inTileXSize;
      winogradTransformOffset = -(convXSize / 2);
      winogradGemmVariant = resolveNhwcWinogradGemmVariant(ctx, useFP16);
      winogradPackedABM = ::winogradPackedABM(ctx.tuneParams, winogradGemmVariant);
      winogradPackedABK = ::winogradPackedABK(ctx.tuneParams, winogradGemmVariant);
      winogradPackedAPadWords = ::winogradPackedAPadWords(winogradGemmVariant);

      LayerPaddingContract winoContract = selectedWinogradPaddingContract(ctx.tuneParams, winogradGemmVariant);
      winogradMAlignment = winoContract.m;
      numInChannelsPadded = roundUpToMultipleInt(inChannels, winoContract.k);
      numOutChannelsPadded = roundUpToMultipleInt(outChannels, winoContract.n);

      numTilesX = (nnX + outTileXSize - 1) / outTileXSize;
      numTilesY = (nnY + outTileYSize - 1) / outTileYSize;
      inTileXYSize = inTileXSize * inTileYSize;
      outTileXYSize = outTileXSize * outTileYSize;

      numTilesPadded = roundUpToMultipleInt(numTilesX * numTilesY * 1, winoContract.m);

      auto transform3x3_4Nhwc = [](float& a0, float& a1, float& a2, float& a3) {
        float z0 = a0, z1 = a1, z2 = a2;
        a0 = z0;
        a1 = 0.5f * (z0 + z1 + z2);
        a2 = 0.5f * (z0 - z1 + z2);
        a3 = z2;
      };
      auto transform3x3_6Nhwc = [](float* a) {
        float z0 = a[0], z1 = a[1], z2 = a[2];
        a[0] = 0.25f * z0;
        a[1] = (float)((1.0 / 6.0) * (-z0 - z1 - z2));
        a[2] = (float)((1.0 / 6.0) * (-z0 + z1 - z2));
        a[3] = (float)((1.0 / 24.0) * (z0 + 2.0 * z1 + 4.0 * z2));
        a[4] = (float)((1.0 / 24.0) * (z0 - 2.0 * z1 + 4.0 * z2));
        a[5] = z2;
      };
      auto transform5x5_6Nhwc = [](float* a) {
        float z0 = a[0], z1 = a[1], z2 = a[2], z3 = a[3], z4 = a[4];
        a[0] = 0.25f * z0;
        a[1] = (float)((1.0 / 6.0) * (-z0 - z1 - z2 - z3 - z4));
        a[2] = (float)((1.0 / 6.0) * (-z0 + z1 - z2 + z3 - z4));
        a[3] = (float)((1.0 / 24.0) * (z0 + 2.0 * z1 + 4.0 * z2 + 8.0 * z3 + 16.0 * z4));
        a[4] = (float)((1.0 / 24.0) * (z0 - 2.0 * z1 + 4.0 * z2 - 8.0 * z3 + 16.0 * z4));
        a[5] = z4;
      };

      std::vector<float> transWeightsNhwc(inTileXYSize * numInChannelsPadded * numOutChannelsPadded, 0.0f);
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
                transform3x3_4Nhwc(tmp[sy][0], tmp[sy][1], tmp[sy][2], tmp[sy][3]);
              for(int sx = 0; sx < inTileXSize; sx++)
                transform3x3_4Nhwc(tmp[0][sx], tmp[1][sx], tmp[2][sx], tmp[3][sx]);
            } else if(inTileXSize == 6) {
              for(int sy = 0; sy < convYSize; sy++)
                transform3x3_6Nhwc(tmp[sy]);
              for(int sx = 0; sx < inTileXSize; sx++) {
                float col[6] = {tmp[0][sx], tmp[1][sx], tmp[2][sx], tmp[3][sx], tmp[4][sx], tmp[5][sx]};
                transform3x3_6Nhwc(col);
                for(int sy2 = 0; sy2 < inTileYSize; sy2++)
                  tmp[sy2][sx] = col[sy2];
              }
            } else {
              throw StringError(
                "Vulkan backend: unsupported 3x3 Winograd input tile size " + Global::intToString(inTileXSize));
            }
          } else {
            for(int sy = 0; sy < convYSize; sy++)
              transform5x5_6Nhwc(tmp[sy]);
            for(int sx = 0; sx < inTileXSize; sx++) {
              float col[6] = {tmp[0][sx], tmp[1][sx], tmp[2][sx], tmp[3][sx], tmp[4][sx], tmp[5][sx]};
              transform5x5_6Nhwc(col);
              for(int sy2 = 0; sy2 < inTileYSize; sy2++)
                tmp[sy2][sx] = col[sy2];
            }
          }
          for(int subY = 0; subY < inTileYSize; subY++) {
            for(int subX = 0; subX < inTileXSize; subX++) {
              int tidx = (subY * inTileXSize + subX) * numInChannelsPadded * numOutChannelsPadded +
                         ic * numOutChannelsPadded + oc;
              transWeightsNhwc[tidx] = tmp[subY][subX];
            }
          }
        }
      }

      if(winogradGemmVariant == WinogradGemmVariant::Coopmat1AccF16) {
        winogradCoopmatBM = ctx.tuneParams.coopmat1AccF16BM;
        winogradCoopmatBK = ctx.tuneParams.coopmat1AccF16BK;
        size_t packedStrideB = winogradCoopmatPackedBStrideWords(
          numOutChannelsPadded, numInChannelsPadded, ctx.tuneParams.coopmat1AccF16BN, ctx.tuneParams.coopmat1AccF16BK);
        testAssert(packedStrideB <= (size_t)std::numeric_limits<int>::max());
        winogradCoopmatBStride = (int)packedStrideB;
        std::vector<float> packedWeights = packWinogradCoopmatBWeights(
          transWeightsNhwc,
          inTileXYSize,
          numOutChannelsPadded,
          numInChannelsPadded,
          ctx.tuneParams.coopmat1AccF16BN,
          ctx.tuneParams.coopmat1AccF16BK);
        filterBuf =
          makeWeightBuf(ctx.device, ctx.queue, ctx.queueMutex, ctx.commandPool, ctx.memProps, packedWeights, useFP16);
      } else if(winogradGemmVariant == WinogradGemmVariant::Coopmat1) {
        winogradCoopmatBM = ctx.tuneParams.coopmat1BM;
        winogradCoopmatBK = ctx.tuneParams.coopmat1BK;
        size_t packedStrideB = winogradCoopmatPackedBStrideWords(
          numOutChannelsPadded, numInChannelsPadded, ctx.tuneParams.coopmat1BN, ctx.tuneParams.coopmat1BK);
        testAssert(packedStrideB <= (size_t)std::numeric_limits<int>::max());
        winogradCoopmatBStride = (int)packedStrideB;
        std::vector<float> packedWeights = packWinogradCoopmatBWeights(
          transWeightsNhwc,
          inTileXYSize,
          numOutChannelsPadded,
          numInChannelsPadded,
          ctx.tuneParams.coopmat1BN,
          ctx.tuneParams.coopmat1BK);
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
          transWeightsNhwc,
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
          transWeightsNhwc,
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
          makeWeightBuf(ctx.device, ctx.queue, ctx.queueMutex, ctx.commandPool, ctx.memProps, transWeightsNhwc, useFP16);
      }

      workspaceElts1 = static_cast<size_t>(inTileXYSize) * numInChannelsPadded;
      workspaceElts2 = static_cast<size_t>(inTileXYSize) * numOutChannelsPadded;

      {
        const int inTile = inTileXSize;
        const int outTile = outTileXSize;
        const int convSz = convXSize;
        const int offset = winogradTransformOffset;
        winogradTransformKernel = VulkanKernels::WinogradTransformNhwc::build(
          ctx.device,
          ctx.pipelineCache,
          useFP16,
          inTile,
          outTile,
          convSz,
          offset,
          ctx.tuneParams.nhwcWinogradTransformLocalSizeX,
          ctx.tuneParams.nhwcWinogradTransformLocalSizeY,
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
        } else if(winogradGemmVariant == WinogradGemmVariant::Coopmat1AccF16) {
          winogradGemmKernel = VulkanKernels::WinogradGemmCoopmat1AccF16::build(
            ctx.device,
            ctx.pipelineCache,
            ctx.tuneParams.coopmat1AccF16BlockSize,
            ctx.tuneParams.coopmat1AccF16BM,
            ctx.tuneParams.coopmat1AccF16BN,
            ctx.tuneParams.coopmat1AccF16BK,
            ctx.tuneParams.coopmat1AccF16WM,
            ctx.tuneParams.coopmat1AccF16WN,
            ctx.tuneParams.coopmat1AccF16TM,
            ctx.tuneParams.coopmat1AccF16TN,
            ctx.tuneParams.coopmat1AccF16TK,
            ctx.tuneParams.coopmat1AccF16Warp,
            numInChannelsPadded,
            ctx.subgroupSize);
        } else if(winogradGemmVariant == WinogradGemmVariant::Coopmat1) {
          winogradGemmKernel = VulkanKernels::WinogradGemmCoopmat1::build(
            ctx.device,
            ctx.pipelineCache,
            ctx.tuneParams.coopmat1BlockSize,
            ctx.tuneParams.coopmat1BM,
            ctx.tuneParams.coopmat1BN,
            ctx.tuneParams.coopmat1BK,
            ctx.tuneParams.coopmat1WM,
            ctx.tuneParams.coopmat1WN,
            ctx.tuneParams.coopmat1TM,
            ctx.tuneParams.coopmat1TN,
            ctx.tuneParams.coopmat1TK,
            ctx.tuneParams.coopmat1Warp,
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
        // outChannels here is the unpadded physical output stride (the C
        // tensor's contiguous NHWC channel run), not numOutChannelsPadded --
        // padding is a GEMM-internal concern and irrelevant to whether the
        // output store can address 4 real channels at once.
        nhwcWinogradUntransformOcVec = (outChannels % 4 == 0) ? 4 : 1;
        winogradUntransformKernel = VulkanKernels::WinogradUntransformNhwc::build(
          ctx.device,
          ctx.pipelineCache,
          useFP16,
          inTile,
          outTile,
          convSz,
          ctx.tuneParams.nhwcWinogradUntransformLocalSizeX,
          ctx.tuneParams.nhwcWinogradUntransformLocalSizeY,
          addToOutputOnStore,
          nhwcWinogradUntransformOcVec);
      }
    } else {
      throw StringError("Vulkan backend: no NHWC convolution implementation selected");
    }
  }

  ~ConvLayer() {
    gemmStridedKernel.destroy(device);
    winogradTransformKernel.destroy(device);
    winogradGemmKernel.destroy(device);
    winogradUntransformKernel.destroy(device);
    conv2dDirectKernel.destroy(device);
    conv2dDirectNhwcKernel.destroy(device);
    conv3x3ImplicitGemmNhwcVec8Kernel.destroy(device);
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
    const BatchNormLayer* fusedBN = nullptr) const {
    const ConvLayer& conv = *this;
    if(conv.useConv2dDirectNhwc) {
      testAssert(fusedBN == nullptr && scratch != nullptr);
      // Native NHWC graph: input and output are graph-owned NHWC buffers.
      VulkanKernels::Conv2dDirectNhwc::PC pc = {
        conv.nnXLen, conv.nnYLen, conv.outChannels, conv.inChannels,
        conv.convXSize / 2, conv.convYSize / 2, conv.paddedSpatialSize, maxBatchSize};
      VulkanKernels::Conv2dDirectNhwc::dispatch(
        ctx, conv.conv2dDirectNhwcKernel, input, conv.filterBuf.get(), output, pc);
      return;
    } else if(conv.convXSize == 1 && conv.convYSize == 1) {
      testAssert(fusedBN == nullptr);
      testAssert(conv.paddedSpatialSize % 4 == 0);
      testAssert(conv.outChannelsPadded1x1 % 8 == 0);
      VulkanKernels::GemmStridedTiledNhwc::PC pc = {
        conv.paddedSpatialSize,
        conv.outChannelsPadded1x1,
        conv.outChannels,
        conv.inChannels * (conv.paddedSpatialSize / 4),
        conv.gemmStridedBStride,
        conv.outChannels * (conv.paddedSpatialSize / 4)};  // strideC uses real N
      if(conv.useGemmStridedCoopmat2Nhwc) {
        VulkanKernels::GemmStridedCoopmat2Nhwc::dispatch(
          ctx, conv.gemmStridedKernel, input, conv.filterBuf.get(), output, conv.inChannels, pc, maxBatchSize);
      } else if(conv.useGemmStridedCoopmat2AccF16Nhwc) {
        VulkanKernels::GemmStridedCoopmat2AccF16Nhwc::dispatch(
          ctx, conv.gemmStridedKernel, input, conv.filterBuf.get(), output, conv.inChannels, pc, maxBatchSize);
      } else if(conv.useGemmStridedTiledNhwc) {
        VulkanKernels::GemmStridedTiledNhwc::dispatch(
          ctx, conv.gemmStridedKernel, input, conv.filterBuf.get(), output, conv.inChannels, pc, maxBatchSize);
      } else if(conv.useGemmStridedCoopmatNhwc) {
        VulkanKernels::GemmStridedCoopmat1Nhwc::dispatch(
          ctx, conv.gemmStridedKernel, input, conv.filterBuf.get(), output, conv.inChannels, pc, maxBatchSize);
      } else if(conv.useGemmStridedCoopmatAccF16Nhwc) {
        testAssert(scratch != nullptr);
        VulkanKernels::GemmStridedCoopmat1AccF16Nhwc::dispatch(
          ctx, conv.gemmStridedKernel, input, conv.filterBuf.get(), output, conv.inChannels, pc, maxBatchSize);
      } else if(conv.gemmStridedVariant == GemmStridedVariant::Coopmat2AccF16) {
        VulkanKernels::GemmStridedCoopmat2AccF16Nhwc::dispatch(
          ctx, conv.gemmStridedKernel, input, conv.filterBuf.get(), output, conv.inChannels, pc, maxBatchSize);
      } else if(conv.gemmStridedVariant == GemmStridedVariant::Coopmat2) {
        VulkanKernels::GemmStridedCoopmat2Nhwc::dispatch(
          ctx, conv.gemmStridedKernel, input, conv.filterBuf.get(), output, conv.inChannels, pc, maxBatchSize);
      } else if(conv.gemmStridedVariant == GemmStridedVariant::Coopmat1AccF16) {
        VulkanKernels::GemmStridedCoopmat1AccF16Nhwc::dispatch(
          ctx, conv.gemmStridedKernel, input, conv.filterBuf.get(), output, conv.inChannels, pc, maxBatchSize);
      } else if(conv.gemmStridedVariant == GemmStridedVariant::Coopmat1) {
        VulkanKernels::GemmStridedCoopmat1Nhwc::dispatch(
          ctx, conv.gemmStridedKernel, input, conv.filterBuf.get(), output, conv.inChannels, pc, maxBatchSize);
      } else if(conv.gemmStridedVariant == GemmStridedVariant::Dot2) {
        VulkanKernels::GemmStridedDot2Nhwc::dispatch(
          ctx, conv.gemmStridedKernel, input, conv.filterBuf.get(), output, conv.inChannels, pc, maxBatchSize);
      } else if(conv.gemmStridedVariant == GemmStridedVariant::Dot2AccF16) {
        VulkanKernels::GemmStridedDot2AccF16Nhwc::dispatch(
          ctx, conv.gemmStridedKernel, input, conv.filterBuf.get(), output, conv.inChannels, pc, maxBatchSize);
      } else {
        VulkanKernels::GemmStridedTiledNhwc::dispatch(
          ctx, conv.gemmStridedKernel, input, conv.filterBuf.get(), output, conv.inChannels, pc, maxBatchSize);
      }
    } else if(conv.useConv3x3ImplicitGemmNhwcVec8) {
      // Dispatch the specialized NHWC convolution.
      testAssert(scratch != nullptr);
#if 1
      // The specialized shader needs padded input channels.  Convolutional
      // trunk channels are normally already multiples of eight; retain the
      // older packing fallback only for the exceptional channel-tail case.
      if(conv.inChannelsPhysical == conv.inChannelsPaddedConv3x3) {
        testAssert((conv.outChannels * conv.paddedSpatialSize) % 4 == 0);
        VulkanKernels::Conv3x3ImplicitGemmCoopmat1AccF32NhwcVec8::PC pc = {
          conv.nnXLen, conv.nnYLen, conv.outChannels, conv.outChannelsPaddedConv3x3,
          conv.inChannelsPaddedConv3x3, conv.paddedSpatialSize, 0,
          (conv.outChannels * conv.paddedSpatialSize) / 4, 0};
        if(conv.useConv3x3ImplicitGemmCoopmat2NhwcVec8) {
          if(conv.useConv3x3ImplicitGemmAccF16NhwcVec8)
            VulkanKernels::Conv3x3ImplicitGemmCoopmat2AccF16NhwcVec8::dispatch(
              ctx, conv.conv3x3ImplicitGemmNhwcVec8Kernel, input, conv.filterBuf.get(), output, conv.convXSize, pc, maxBatchSize);
          else
            VulkanKernels::Conv3x3ImplicitGemmCoopmat2AccF32NhwcVec8::dispatch(
              ctx, conv.conv3x3ImplicitGemmNhwcVec8Kernel, input, conv.filterBuf.get(), output, conv.convXSize, pc, maxBatchSize);
        } else if(conv.useConv3x3ImplicitGemmAccF16NhwcVec8)
          VulkanKernels::Conv3x3ImplicitGemmCoopmat1AccF16NhwcVec8::dispatch(
            ctx, conv.conv3x3ImplicitGemmNhwcVec8Kernel, input, conv.filterBuf.get(), output, conv.convXSize, pc, maxBatchSize);
        else
          VulkanKernels::Conv3x3ImplicitGemmCoopmat1AccF32NhwcVec8::dispatch(
            ctx, conv.conv3x3ImplicitGemmNhwcVec8Kernel, input, conv.filterBuf.get(), output, conv.convXSize, pc, maxBatchSize);
        return;
      }
#endif
      testAssert(false);
    } else if(conv.useConv3x3WinogradNhwc) {
      // Normalization and activation are applied by the caller before this
      // channel-innermost Winograd convolution.
      testAssert(fusedBN == nullptr);
      VulkanBuffer* ws0 = scratch->permanentSlot(0);
      VulkanBuffer* ws1 = scratch->permanentSlot(1);
      int ntxty = maxBatchSize * conv.numTilesX * conv.numTilesY;
      int numTilesPaddedRun = roundUpToMultipleInt(ntxty, conv.winogradMAlignment);

      VulkanKernels::WinogradTransformNhwc::PC pcTransform = {
        conv.nnXLen,
        conv.nnYLen,
        conv.numTilesX,
        conv.numTilesY,
        conv.inChannels,
        conv.numInChannelsPadded,
        ntxty,
        numTilesPaddedRun,
        conv.paddedSpatialSize};
      VulkanKernels::WinogradTransformNhwc::dispatch(ctx, conv.winogradTransformKernel, input, ws0, pcTransform);
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
        conv.winogradGemmVariant == WinogradGemmVariant::Coopmat1 ||
        conv.winogradGemmVariant == WinogradGemmVariant::Coopmat1AccF16 ||
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
      } else if(conv.winogradGemmVariant == WinogradGemmVariant::Coopmat1AccF16) {
        VulkanKernels::WinogradGemmCoopmat1AccF16::dispatch(
          ctx,
          conv.winogradGemmKernel,
          ws0,
          conv.filterBuf.get(),
          ws1,
          conv.numInChannelsPadded,
          pcGemm,
          conv.inTileXYSize);
      } else if(conv.winogradGemmVariant == WinogradGemmVariant::Coopmat1) {
        VulkanKernels::WinogradGemmCoopmat1::dispatch(
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

      VulkanKernels::WinogradUntransformNhwc::PC pcUnt = {
        conv.nnXLen,
        conv.nnYLen,
        conv.numTilesX,
        conv.numTilesY,
        conv.outChannels,
        conv.numOutChannelsPadded,
        numTilesPaddedRun,
        conv.paddedSpatialSize};
      VulkanKernels::WinogradUntransformNhwc::dispatch(
        ctx, conv.winogradUntransformKernel, ws1, output, pcUnt, maxBatchSize, conv.nhwcWinogradUntransformOcVec);
      VulkanHelpers::cmdComputeWARBarrier(ctx.cmd, ws0->buffer);
      VulkanHelpers::cmdComputeWARBarrier(ctx.cmd, ws1->buffer);
    } else {
      testAssert(false && "Vulkan ConvLayer has no selected NHWC dispatch");
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
      fusesResidualAdd(
        addToOutputOnStore &&
        !conv.useConv2dDirectNhwc &&
        ((conv.convXSize == 1 && conv.convYSize == 1) || conv.useConv3x3ImplicitGemmNhwcVec8 ||
         conv.useConv3x3WinogradNhwc)) {
    static_assert(
      VulkanKernels::GemmStridedTiledNhwc::kSingleWriterStore &&
        VulkanKernels::GemmStridedCoopmat1Nhwc::kSingleWriterStore &&
        VulkanKernels::GemmStridedCoopmat2Nhwc::kSingleWriterStore &&
        VulkanKernels::GemmStridedCoopmat2AccF16Nhwc::kSingleWriterStore &&
        VulkanKernels::GemmStridedDot2Nhwc::kSingleWriterStore &&
        VulkanKernels::GemmStridedDot2AccF16Nhwc::kSingleWriterStore &&
        VulkanKernels::Conv3x3ImplicitGemmCoopmat1AccF32NhwcVec8::kSingleWriterStore &&
        VulkanKernels::Conv3x3ImplicitGemmCoopmat2AccF32NhwcVec8::kSingleWriterStore &&
        VulkanKernels::Conv3x3ImplicitGemmCoopmat1AccF16NhwcVec8::kSingleWriterStore &&
        VulkanKernels::Conv3x3ImplicitGemmCoopmat2AccF16NhwcVec8::kSingleWriterStore,
      "ADD_TO_OUTPUT residual fusion requires single-writer C stores (no split-K)");
    assert(norm.numChannels == conv.inChannels);
  }

  NormActConv() = delete;
  NormActConv(const NormActConv&) = delete;
  NormActConv& operator=(const NormActConv&) = delete;

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
    norm.dispatch(ctx, scratch, input, inputScratchOrInput, mask, conv.paddedSpatialSize, maxBatchSize);
    VulkanHelpers::cmdComputeBarrier(ctx.cmd, inputScratchOrInput->buffer);
    conv.dispatch(ctx, scratch, inputScratchOrInput, output, maxBatchSize);
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
  const bool useNhwc;
  VkDevice device;
  ComputeKernel gpoolKernel;
  ComputeKernel gpoolNhwcKernel;
  ComputeKernel addChannelBiasesKernel;
  ComputeKernel scaleBiasMaskActNhwcDynamicBiasKernel;
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
      useNhwc(ctx.useNhwc),
      device(ctx.device) {
    gpoolNhwcKernel =
  VulkanKernels::GPoolReductionNhwc::build(
    ctx.device, ctx.pipelineCache, useFP16, ctx.tuneParams.gpoolNhwcXystride);
scaleBiasMaskActNhwcDynamicBiasKernel = VulkanKernels::ScaleBiasMaskActInplaceNhwc::build(
  ctx.device,
  ctx.pipelineCache,
  useFP16,
  normActConv2.norm.activation,
  regularChannels % 4 == 0,
  true);

    addPointwiseKernel = VulkanKernels::AddPointwise::build(ctx.device, ctx.pipelineCache, useFP16);
  }

  GlobalPoolingResidualBlockVk() = delete;
  GlobalPoolingResidualBlockVk(const GlobalPoolingResidualBlockVk&) = delete;
  GlobalPoolingResidualBlockVk& operator=(const GlobalPoolingResidualBlockVk&) = delete;

  ~GlobalPoolingResidualBlockVk() {
    gpoolKernel.destroy(device);
    gpoolNhwcKernel.destroy(device);
    addChannelBiasesKernel.destroy(device);
    scaleBiasMaskActNhwcDynamicBiasKernel.destroy(device);
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

    preBN.dispatch(ctx, scratch, trunk, trunkScratch, mask, paddedSpatialSize, maxBatchSize);
    VulkanHelpers::cmdComputeBarrier(ctx.cmd, trunkScratch->buffer);
    regularConv.dispatch(ctx, scratch, trunkScratch, regularOut.buf, maxBatchSize);
    gpoolConv.dispatch(ctx, scratch, trunkScratch, gpoolOut.buf, maxBatchSize);
    VulkanHelpers::cmdComputeBarrier(ctx.cmd, gpoolOut.buf->buffer);
    gpoolBN.dispatch(ctx, scratch, gpoolOut.buf, gpoolOut.buf, mask, paddedSpatialSize, maxBatchSize);
    VulkanHelpers::cmdComputeBarrier(ctx.cmd, gpoolOut.buf->buffer);

    // gpoolConv and gpoolBN already produce NHWC. The pooled result
    // has one spatial element, so [N,1,3*C] is byte-identical to the flat
    // FP32 [N,3*C] vector consumed by MatMulLayer.
    VulkanKernels::GPoolReductionNhwc::PC gpoolPC = {gpoolChannels, paddedSpatialSize};
    VulkanKernels::GPoolReductionNhwc::dispatch(
      ctx, gpoolNhwcKernel, gpoolOut.buf, gpoolConcat.buf, mask, maskSum, gpoolPC, maxBatchSize);

    VulkanHelpers::cmdComputeBarrier(ctx.cmd, gpoolConcat.buf->buffer);

    gpoolToBiasMul.dispatch(ctx, gpoolConcat.buf, gpoolBias.buf, maxBatchSize);
    VulkanHelpers::cmdComputeBarrier(ctx.cmd, gpoolBias.buf->buffer);
    VulkanHelpers::cmdComputeBarrier(ctx.cmd, regularOut.buf->buffer);

    // Fold the dynamic global-pooling bias into the immediately following
    // BN/activation pass. This removes one materialization and dispatch.
    VulkanKernels::ScaleBiasMaskActNhwc::PC scaleBiasPC = {regularChannels, paddedSpatialSize, maxBatchSize};
    VulkanKernels::ScaleBiasMaskActInplaceNhwc::dispatch(
      ctx,
      scaleBiasMaskActNhwcDynamicBiasKernel,
      regularOut.buf,
      normActConv2.norm.mergedScaleBuf.get(),
      normActConv2.norm.mergedBiasBuf.get(),
      mask,
      scaleBiasPC,
      gpoolBias.buf);

    VulkanHelpers::cmdComputeBarrier(ctx.cmd, regularOut.buf->buffer);

    if(normActConv2.fusesResidualAdd) {
      VulkanHelpers::cmdComputeWARBarrier(ctx.cmd, trunk->buffer);
      normActConv2.conv.dispatch(ctx, scratch, regularOut.buf, trunk, maxBatchSize);
    } else {
      normActConv2.conv.dispatch(ctx, scratch, regularOut.buf, trunkScratch, maxBatchSize);
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
  const bool useNhwc;
  VBuf gammaBuf;
  VBuf betaBuf;
  VBuf actScaleBuf;  // all-ones, FP16 if useFP16 — only allocated when activation != IDENTITY
  VBuf actBiasBuf;   // all-zeros, FP16 if useFP16 — only allocated when activation != IDENTITY
  // Owned SILU scale-bias-mask-activation kernel, built only when activation
  // is SILU (the only post-RMSNorm activation that this dispatches through
  // ScaleBiasMaskActInplaceNhwc). Empty for IDENTITY or other activations.
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
  const bool useNhwc;
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
  const bool useNhwc;
  std::unique_ptr<ConvLayer> nhwcConv;
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
  void dispatch(
    const CmdCtx& ctx, ScratchBuffers* scratch, VulkanBuffer* input, VulkanBuffer* output, int maxBatchSize) const;
};

struct TransformerAttentionBlockVk : VulkanBlockVk {
  enum class AttentionVariant { Tiled, Coopmat1AccF32, CoopmatMaintenance1AccF32, Coopmat2AccF32, Dot2AccF32 };
  const std::string name;
  const int numHeads;
  const int numKVHeads;
  const int qHeadDim;
  const int vHeadDim;
  const bool useRope;
  const bool learnableRope;
  const int inChannels;
  const int paddedSpatialSize;
  const bool useNhwc;
  const AttentionVariant attentionVariant;

  TransformerRMSNormLayerVk preLN;
  TransformerMatMulLayerVk qProj;
  TransformerMatMulLayerVk kProj;
  TransformerMatMulLayerVk vProj;
  TransformerMatMulLayerVk outProj;

  // RoPE tables (always FP32)
  VBuf ropeCosTable;
  VBuf ropeSinTable;
  VkDevice device;
  // Owned per-(qHeadDim,vHeadDim) attention kernel.
  // attentionKernel is the always-unsplit path. splitKAttentionKernel and
  // splitKResolveKernel are built only when split-K is viable for this block.
  ComputeKernel attentionKernel;
  ComputeKernel splitKAttentionKernel;
  ComputeKernel attentionPrepareQKKernel;
  ComputeKernel splitKResolveKernel;
  int splitKCutoffBatch = 0;

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

  ~TransformerAttentionBlockVk() {
    attentionKernel.destroy(device);
    splitKAttentionKernel.destroy(device);
    attentionPrepareQKKernel.destroy(device);
    splitKResolveKernel.destroy(device);
  }

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
  const bool useNhwc;

  std::unique_ptr<ConvLayer> initialConv;
  std::unique_ptr<MatMulLayer> initialMatMul;
  std::unique_ptr<SGFMetadataEncoderVk> sgfMetadataEncoder;
  std::unique_ptr<BlockStackVk> blocks;
  std::unique_ptr<BatchNormLayer> trunkTipBN;
  std::unique_ptr<RMSNormVk> trunkTipRMSNorm;
  // Owned channel-bias add for the initialMatMul / SGF metadata residuals.
  VkDevice device;
  ComputeKernel addChannelBiasesKernel;
  ComputeKernel addChannelBiasesNhwcKernel;

  TrunkVk() = delete;
  TrunkVk(const TrunkVk&) = delete;
  TrunkVk& operator=(const TrunkVk&) = delete;

  ~TrunkVk() {
    addChannelBiasesKernel.destroy(device);
    addChannelBiasesNhwcKernel.destroy(device);
  }

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
  const bool useNhwc;

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
  ComputeKernel gpoolNhwcKernel;
  ComputeKernel addChannelBiasesKernel;
  ComputeKernel addChannelBiasesNhwcKernel;

  PolicyHeadVk() = delete;
  PolicyHeadVk(const PolicyHeadVk&) = delete;
  PolicyHeadVk& operator=(const PolicyHeadVk&) = delete;

  ~PolicyHeadVk() {
    gpoolKernel.destroy(device);
    gpoolNhwcKernel.destroy(device);
    addChannelBiasesKernel.destroy(device);
    addChannelBiasesNhwcKernel.destroy(device);
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
  bool useNhwc;

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
  ComputeKernel valueHeadPoolNhwcKernel;

  ValueHeadVk() = delete;
  ValueHeadVk(const ValueHeadVk&) = delete;
  ValueHeadVk& operator=(const ValueHeadVk&) = delete;

  ~ValueHeadVk() {
    valueHeadPoolKernel.destroy(device);
    valueHeadPoolNhwcKernel.destroy(device);
  }

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

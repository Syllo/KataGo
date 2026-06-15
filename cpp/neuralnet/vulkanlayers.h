#ifndef NEURALNET_VULKAN_LAYERS_H_
#define NEURALNET_VULKAN_LAYERS_H_

#ifdef USE_VULKAN_BACKEND

#include <algorithm>
#include <array>
#include <cstdlib>
#include <initializer_list>
#include <limits>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>
#include "../core/test.h"
#include "../neuralnet/vulkanhelpers.h"
#include "../neuralnet/vulkankernels.h"
// For the k*TileFields pointer-to-member groups, which resolveNativeConv3x3Tile
// reads directly rather than re-spelling each tile member.
#include "../neuralnet/vulkantuner_shared.h"

// Forward-declared in vulkankernels.h. Full definition lives in vulkanbackend.h.
namespace VulkanProfiler {
  struct DispatchProfiler;
}

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
  // Held by value so the constructor can mask out filter-disabled accelerator
  // variants from tunedKernelMask, matching what the runtime may actually
  // select. All resolve* helpers read this copy.
  VulkanTuneParams tuneParams;
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
  // True when the coopmat2 attention shader is usable (coopmat2 base plus the
  // reductions/conversions/per-element-operations/block-loads feature bits it
  // needs). False on devices lacking any of those; coopmat1 is the fallback.
  bool supportsCoopmat2Attention;
  // Device-reported usable coopmat shapes (f16/f16->f32/f32, subgroup scope).
  // The coopmat TM/TN/TK must match one of these. Empty iff !supportsCoopmat1F16.
  std::vector<CoopmatShape> coopmatShapes;
  std::vector<CoopmatShape> coopmatAccF16Shapes;
  std::vector<Coopmat2FlexShape> coopmat2FlexShapes;
  std::vector<Coopmat2FlexShape> coopmat2AccF16FlexShapes;
  uint32_t coopmat2ReservedSharedBytes;
  uint32_t maxComputeSharedMemorySize;
  VkPipelineCache pipelineCache;

  VulkanLayerContext() = delete;
  VulkanLayerContext(const VulkanLayerContext&) = delete;
  VulkanLayerContext& operator=(const VulkanLayerContext&) = delete;

  // Build the layer context from a device's reported info plus the tuner's
  // params. The disabled-accel mask is folded into the tuneParams copy so the
  // resolve* helpers can never select a filter-disabled variant.
  static VulkanLayerContext fromDevice(
    VkDevice device_,
    VkQueue queue_,
    std::mutex& queueMutex_,
    VkCommandPool commandPool_,
    const VulkanDeviceInfo& devInfo,
    const VulkanTuneParams& tuneParams_,
    VkPipelineCache pipelineCache_) {
    return VulkanLayerContext(device_, queue_, queueMutex_, commandPool_, devInfo, tuneParams_, pipelineCache_);
  }

 private:
  VulkanLayerContext(
    VkDevice device_,
    VkQueue queue_,
    std::mutex& queueMutex_,
    VkCommandPool commandPool_,
    const VulkanDeviceInfo& devInfo,
    const VulkanTuneParams& tuneParams_,
    VkPipelineCache pipelineCache_)
    : device(device_),
      queue(queue_),
      queueMutex(queueMutex_),
      commandPool(commandPool_),
      memProps(devInfo.memoryProperties),
      tuneParams(tuneParams_),
      supportsSubgroupShuffleCompute(devInfo.supportsSubgroupShuffleCompute),
      subgroupSize(devInfo.subgroupSize),
      canRequireReportedSubgroupSize(devInfo.canRequireComputeSubgroupSize(devInfo.subgroupSize)),
      supportsFP16Compute(devInfo.supportsFP16Compute),
      supportsDot2F16(devInfo.supportsDot2F16),
      supportsDot2F16AccF16(devInfo.supportsDot2F16AccF16),
      supportsCoopmat1F16(devInfo.supportsCoopmat1F16),
      supportsCoopmat1F16AccF16(devInfo.supportsCoopmat1F16AccF16),
      supportsCoopmatMaintenance1(devInfo.supportsCoopmatMaintenance1),
      supportsCoopmat2F16(devInfo.supportsCoopmat2F16),
      supportsCoopmat2F16AccF16(devInfo.supportsCoopmat2F16AccF16),
      supportsCoopmat2Attention(devInfo.supportsCoopmat2Attention),
      coopmatShapes(devInfo.coopmatShapes),
      coopmatAccF16Shapes(devInfo.coopmatAccF16Shapes),
      coopmat2FlexShapes(devInfo.coopmat2FlexShapes),
      coopmat2AccF16FlexShapes(devInfo.coopmat2AccF16FlexShapes),
      coopmat2ReservedSharedBytes(devInfo.coopmat2ReservedSharedBytes),
      maxComputeSharedMemorySize(devInfo.properties.limits.maxComputeSharedMemorySize),
      pipelineCache(pipelineCache_) {
    tuneParams.tunedKernelMask &= ~devInfo.disabledAccelVariantMask;
  }
};

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
  VulkanKernels::AddChannelBiasesNhwc::PC nativePc = {maxBatchSize * numChannels, numChannels, paddedSpatialSize};
  VulkanKernels::AddChannelBiasesNhwc::dispatch(ctx, addChannelBiasesNhwcKernel, srcDst, bias, nativePc);
}

// Fence so writes to buf by `producer` are visible to `consumer` (the standard
// output/consumer ordering used between layer dispatches). The two name args are
// documentation only; they state the dependency the barrier enforces.
inline void barrierFor(const CmdCtx& ctx, VulkanBuffer* buf, const char* producer, const char* consumer) {
  (void)producer;
  (void)consumer;
  VulkanHelpers::cmdComputeBarrier(ctx.cmd, buf->buffer);
}

// Fence so reads of buf by its previous consumer finish before a new write
// (write-after-read), e.g. for in-place residual adds.
inline void warBarrierFor(const CmdCtx& ctx, VulkanBuffer* buf, const char* reason) {
  (void)reason;
  VulkanHelpers::cmdComputeWARBarrier(ctx.cmd, buf->buffer);
}

// Whether the device reports a usable coopmat fragment shape (m,n,k).
inline bool coopmatShapeSupported(const std::vector<CoopmatShape>& shapes, int m, int n, int k) {
  for(const CoopmatShape& shape: shapes) {
    if(shape.m == (uint32_t)m && shape.n == (uint32_t)n && shape.k == (uint32_t)k)
      return true;
  }
  return false;
}

// --- Strided / Winograd packed-tile geometry lookups ---
//
// The layer selects a GEMM variant once (resolveGemmStridedVariant /
// resolveWinogradGemmVariant) and then reads the packed-tile geometry for that
// single variant from the per-variant tables below, so the A/B packing performed
// in ConvLayer's ctor can never disagree with the kernel that is actually
// dispatched. BN/BK are the B-tile dims, BM the A-tile M dim, and the pad
// scalars/words are the per-tile padding both the packer and the shader apply.

// Index of a TUNED_GEMM_STRIDED_* / TUNED_WINOGRAD_* bit into its table below.
// The ids occupy contiguous bit ranges (strided 5..11, winograd 12..18) in
// declaration order; the static_asserts below pin that contract.
inline int gemmVariantTableIndex(int64_t variant, int firstBit) {
  int idx = 0;
  int64_t v = variant;
  while((v & 1) == 0) {
    v >>= 1;
    idx++;
  }
  return idx - firstBit;
}

struct StridedGemmVariantInfo {
  const int32_t VulkanTuneParams::* bnField;  // B-tile N dim source; nullptr for the raw-weights tiled path
  const int32_t VulkanTuneParams::* bkField;  // B-tile K dim source; nullptr when BK is the fixed DOT2_BK
  int padScalars;                             // per-B-tile pad width in scalars
  bool usesPackedB;                           // consumes a pre-packed B tensor
  bool rowMajor;                              // coopmat packs rows row-major, dot2 column-major
};
static const StridedGemmVariantInfo kStridedGemmVariantInfo[] = {
  // TUNED_GEMM_STRIDED_TILED
  {nullptr, nullptr, 0, false, false},
  // TUNED_GEMM_STRIDED_DOT2
  {&VulkanTuneParams::nhwcStridedDot2BN, nullptr, STRIDED_DOT2_PACKED_B_PAD_SCALARS, true, false},
  // TUNED_GEMM_STRIDED_DOT2_ACCF16
  {&VulkanTuneParams::nhwcStridedDot2AccF16BN, nullptr, STRIDED_DOT2_PACKED_B_PAD_SCALARS, true, false},
  // TUNED_GEMM_STRIDED_COOPMAT1
  {&VulkanTuneParams::nhwcStridedCoopmat1BN,
   &VulkanTuneParams::nhwcStridedCoopmat1BK,
   STRIDED_COOPMAT_PACKED_B_PAD_SCALARS,
   true,
   true},
  // TUNED_GEMM_STRIDED_COOPMAT1_ACCF16
  {&VulkanTuneParams::nhwcStridedCoopmat1AccF16BN,
   &VulkanTuneParams::nhwcStridedCoopmat1AccF16BK,
   STRIDED_COOPMAT_PACKED_B_PAD_SCALARS,
   true,
   true},
  // TUNED_GEMM_STRIDED_COOPMAT2
  {&VulkanTuneParams::nhwcStridedCoopmat2BN,
   &VulkanTuneParams::nhwcStridedCoopmat2BK,
   STRIDED_COOPMAT2_PACKED_B_PAD_SCALARS,
   true,
   true},
  // TUNED_GEMM_STRIDED_COOPMAT2_ACCF16
  {&VulkanTuneParams::nhwcStridedCoopmat2AccF16BN,
   &VulkanTuneParams::nhwcStridedCoopmat2AccF16BK,
   STRIDED_COOPMAT2_PACKED_B_PAD_SCALARS,
   true,
   true},
};
static_assert(
  VulkanTuner::TUNED_GEMM_STRIDED_TILED == ((int64_t)1 << 5) &&
    VulkanTuner::TUNED_GEMM_STRIDED_COOPMAT2_ACCF16 == ((int64_t)1 << 11),
  "strided GEMM variant ids must occupy bits 5..11 for kStridedGemmVariantInfo");

inline const StridedGemmVariantInfo& stridedGemmVariantInfo(int64_t variant) {
  return kStridedGemmVariantInfo[gemmVariantTableIndex(variant, 5)];
}

// True when this strided variant consumes a pre-packed B tensor (dot2/coopmat),
// rather than the raw transposed weights the tiled path reads.
inline bool gemmStridedVariantUsesPackedB(int64_t variant) {
  return stridedGemmVariantInfo(variant).usesPackedB;
}

// True when the packed B rows are stored row-major (coopmat) rather than
// column-major (dot2).
inline bool gemmStridedVariantUsesRowMajorPackedB(int64_t variant) {
  return stridedGemmVariantInfo(variant).rowMajor;
}

// B-tile N dim for the selected strided variant (used to pad outChannels).
inline int gemmStridedPackedBBN(const VulkanTuneParams& tuneParams, int64_t variant) {
  const StridedGemmVariantInfo& info = stridedGemmVariantInfo(variant);
  return info.bnField != nullptr ? tuneParams.*(info.bnField) : 0;
}

// B-tile K dim for the selected strided variant (DOT2 uses the fixed DOT2_BK).
inline int gemmStridedPackedBBK(const VulkanTuneParams& tuneParams, int64_t variant) {
  const StridedGemmVariantInfo& info = stridedGemmVariantInfo(variant);
  if(!info.usesPackedB)
    return 0;
  return info.bkField != nullptr ? tuneParams.*(info.bkField) : (int)VulkanKernels::DOT2_BK;
}

// Per-B-tile pad width (in scalars) for the selected strided variant.
inline int gemmStridedPackedBPadScalars(int64_t variant) {
  return stridedGemmVariantInfo(variant).padScalars;
}

struct WinogradGemmVariantInfo {
  const int32_t VulkanTuneParams::* bmField;  // A-tile M dim source; nullptr -> winogradGemmM
  const int32_t VulkanTuneParams::* bkField;  // A-tile K dim source; nullptr -> winogradGemmK or DOT2_BK
  int padWords;                               // per-A-tile pad width in words
  bool bkIsDot2;                              // A-tile K is the fixed DOT2_BK, not a field
  // This variant's kernel's M/N/K padding contract, so the contract lookup is a
  // table read rather than a parallel switch that could disagree with the row.
  LayerPaddingContract (*contract)(const VulkanTuneParams&);
};
static const WinogradGemmVariantInfo kWinogradGemmVariantInfo[] = {
  // TUNED_WINOGRAD_TILED
  {nullptr, nullptr, WINOGRAD_ROW_MAJOR_A_PAD_WORDS, false, &VulkanKernels::WinogradGemm::layerPaddingContract},
  // TUNED_WINOGRAD_DOT2
  {&VulkanTuneParams::dot2BM, nullptr, WINOGRAD_ROW_MAJOR_A_PAD_WORDS, true,
   &VulkanKernels::WinogradGemmDot2::layerPaddingContract},
  // TUNED_WINOGRAD_DOT2_ACCF16
  {&VulkanTuneParams::dot2AccF16BM, nullptr, WINOGRAD_ROW_MAJOR_A_PAD_WORDS, true,
   &VulkanKernels::WinogradGemmDot2AccF16::layerPaddingContract},
  // TUNED_WINOGRAD_COOPMAT1
  {&VulkanTuneParams::coopmat1BM, &VulkanTuneParams::coopmat1BK, WINOGRAD_COOPMAT_PACKED_PAD_WORDS, false,
   &VulkanKernels::WinogradGemmCoopmat1::layerPaddingContract},
  // TUNED_WINOGRAD_COOPMAT1_ACCF16
  {&VulkanTuneParams::coopmat1AccF16BM, &VulkanTuneParams::coopmat1AccF16BK, WINOGRAD_COOPMAT_PACKED_PAD_WORDS, false,
   &VulkanKernels::WinogradGemmCoopmat1AccF16::layerPaddingContract},
  // TUNED_WINOGRAD_COOPMAT2
  {&VulkanTuneParams::coopmat2BM, &VulkanTuneParams::coopmat2BK, WINOGRAD_COOPMAT2_PACKED_PAD_WORDS, false,
   &VulkanKernels::WinogradGemmCoopmat2::layerPaddingContract},
  // TUNED_WINOGRAD_COOPMAT2_ACCF16
  {&VulkanTuneParams::coopmat2AccF16BM, &VulkanTuneParams::coopmat2AccF16BK, WINOGRAD_COOPMAT2_PACKED_PAD_WORDS, false,
   &VulkanKernels::WinogradGemmCoopmat2AccF16::layerPaddingContract},
};
static_assert(
  VulkanTuner::TUNED_WINOGRAD_TILED == ((int64_t)1 << 12) &&
    VulkanTuner::TUNED_WINOGRAD_COOPMAT2_ACCF16 == ((int64_t)1 << 18),
  "winograd GEMM variant ids must occupy bits 12..18 for kWinogradGemmVariantInfo");

inline const WinogradGemmVariantInfo& winogradGemmVariantInfo(int64_t variant) {
  return kWinogradGemmVariantInfo[gemmVariantTableIndex(variant, 12)];
}

// A-tile M dim for the selected Winograd GEMM variant (fallback: winogradGemmM).
inline int winogradPackedABM(const VulkanTuneParams& tuneParams, int64_t variant) {
  const WinogradGemmVariantInfo& info = winogradGemmVariantInfo(variant);
  return info.bmField != nullptr ? tuneParams.*(info.bmField) : tuneParams.winogradGemmM;
}

// A-tile K dim for the selected Winograd GEMM variant (fallback: winogradGemmK).
inline int winogradPackedABK(const VulkanTuneParams& tuneParams, int64_t variant) {
  const WinogradGemmVariantInfo& info = winogradGemmVariantInfo(variant);
  if(info.bkIsDot2)
    return (int)VulkanKernels::DOT2_BK;
  return info.bkField != nullptr ? tuneParams.*(info.bkField) : tuneParams.winogradGemmK;
}

// Per-A-tile pad width (in words) for the selected Winograd GEMM variant.
inline int winogradPackedAPadWords(int64_t variant) {
  return winogradGemmVariantInfo(variant).padWords;
}

// ---- Generic accelerator-tier selection ---------------------------------
//
// Every accelerated kernel family (strided GEMM, Winograd GEMM, 3x3/5x5
// implicit-GEMM conv, attention) picks its variant by the same policy:
//
//   * Ordered tiers: coopmat first, then dot2, then the non-accelerated
//     fallback. The first tier holding any usable variant wins outright -- a
//     slower coopmat variant still beats a faster dot2 one, because the tier
//     order encodes an architectural preference, not a speed ranking.
//   * Within a tier: every usable variant's measured time is scaled by its
//     selection penalty and the smallest result wins. A penalty above 1.0 is
//     how a variant that costs something other than time (precision, for FP16
//     accumulation) is made to earn its selection.
//   * "Usable" always means tuned AND measured (timeUs > 0) AND supported by
//     the device -- an unmeasured variant is never selectable.
//
// selectVariantTier below is the single implementation of that policy. Each
// family just builds its candidate table and calls it, so the runtime layer
// selection, the tuner's diagnostics, and the FLOP estimator cannot drift
// apart. Add a tier by adding a row, not a branch.

// Multiplier applied to a variant's measured time before ranking it against the
// others in its tier. 1.0 means "rank on raw measured time".
constexpr double VARIANT_SELECT_PENALTY_NONE = 1.0;
// FP16 accumulation trades precision for speed, so it is only selected when it
// is at least 25% faster than FP32 accumulation -- expressed here as a 1.25x
// penalty on its measured time. Constexpr for now; this is the natural knob to
// promote to a tune-file field if it ever wants to be configurable per device.
constexpr double VARIANT_SELECT_PENALTY_ACCF16 = 1.25;

// One selectable variant within a tier.
struct VariantChoice {
  int64_t bit;     // the TUNED_* variant bit to return if this wins
  int32_t timeUs;  // measured weighted model time; <= 0 means not measured
  bool usable;     // device support + tuned bit + any family-specific gating
  // Ranking penalty; VARIANT_SELECT_PENALTY_ACCF16 for FP16-accumulation
  // flavors, VARIANT_SELECT_PENALTY_NONE (the default) otherwise.
  double selectPenalty = VARIANT_SELECT_PENALTY_NONE;
};

// Resolve one tier: the usable, measured variant with the smallest penalized
// time. Returns 0 when the tier has no usable measured variant.
inline int64_t selectWithinTier(ArrayView<VariantChoice> choices) {
  const VariantChoice* best = nullptr;
  double bestCost = 0.0;
  for(const VariantChoice& c: choices) {
    if(!c.usable || c.timeUs <= 0)
      continue;
    const double cost = (double)c.timeUs * c.selectPenalty;
    // Strictly-smaller cost wins. On an exact tie the smaller penalty wins, so a
    // penalized flavor never displaces an unpenalized one it merely ties: the
    // FP16 rule is "at least 25% faster", not "25% faster or equal". Making the
    // tie-break explicit also keeps the result independent of row order.
    const bool better = best == nullptr || cost < bestCost ||
                        (cost == bestCost && c.selectPenalty < best->selectPenalty);
    if(better) {
      best = &c;
      bestCost = cost;
    }
  }
  return best != nullptr ? best->bit : 0;
}

// Walk tiers in preference order and return the first tier's winner, or
// fallbackBit when no tier has a usable measured variant.
inline int64_t
selectVariantTier(std::initializer_list<ArrayView<VariantChoice>> tiers, int64_t fallbackBit) {
  for(ArrayView<VariantChoice> tier: tiers) {
    const int64_t bit = selectWithinTier(tier);
    if(bit != 0)
      return bit;
  }
  return fallbackBit;
}

// Pick the fastest 1x1-strided GEMM variant available for the device: prefer
// the coopmat tier over dot2 over tiled, and FP16 accumulation only when it
// beats FP32 accumulation by the VARIANT_SELECT_PENALTY_ACCF16 margin.
inline int64_t resolveGemmStridedVariant(const VulkanLayerContext& ctx, bool fp16) {
  if(!fp16)
    return VulkanTuner::TUNED_GEMM_STRIDED_TILED;
  const VulkanTuneParams& tp = ctx.tuneParams;
  auto tuned = [&](int64_t bit) { return tp.hasKernelTuned(bit); };

  const VariantChoice coopTier[] = {
    {VulkanTuner::TUNED_GEMM_STRIDED_COOPMAT2,
     tp.nhwcGemmCoopmat2F32TimeUs,
     ctx.supportsCoopmat2F16 && tuned(VulkanTuner::TUNED_GEMM_STRIDED_COOPMAT2),
     VARIANT_SELECT_PENALTY_NONE},
    {VulkanTuner::TUNED_GEMM_STRIDED_COOPMAT1,
     tp.nhwcGemmCoopmat1F32TimeUs,
     ctx.supportsCoopmat1F16 && tuned(VulkanTuner::TUNED_GEMM_STRIDED_COOPMAT1),
     VARIANT_SELECT_PENALTY_NONE},
    {VulkanTuner::TUNED_GEMM_STRIDED_COOPMAT2_ACCF16,
     tp.nhwcGemmCoopmat2F16TimeUs,
     ctx.supportsCoopmat2F16AccF16 && tuned(VulkanTuner::TUNED_GEMM_STRIDED_COOPMAT2_ACCF16),
     VARIANT_SELECT_PENALTY_ACCF16},
    {VulkanTuner::TUNED_GEMM_STRIDED_COOPMAT1_ACCF16,
     tp.nhwcGemmCoopmat1F16TimeUs,
     ctx.supportsCoopmat1F16AccF16 && tuned(VulkanTuner::TUNED_GEMM_STRIDED_COOPMAT1_ACCF16),
     VARIANT_SELECT_PENALTY_ACCF16},
  };
  const VariantChoice dot2Tier[] = {
    {VulkanTuner::TUNED_GEMM_STRIDED_DOT2,
     tp.nhwcGemmDot2F32TimeUs,
     ctx.supportsDot2F16 && tuned(VulkanTuner::TUNED_GEMM_STRIDED_DOT2),
     VARIANT_SELECT_PENALTY_NONE},
    {VulkanTuner::TUNED_GEMM_STRIDED_DOT2_ACCF16,
     tp.nhwcGemmDot2F16TimeUs,
     ctx.supportsDot2F16AccF16 && tuned(VulkanTuner::TUNED_GEMM_STRIDED_DOT2_ACCF16),
     VARIANT_SELECT_PENALTY_ACCF16},
  };
  return selectVariantTier({coopTier, dot2Tier}, VulkanTuner::TUNED_GEMM_STRIDED_TILED);
}

// Pick the fastest Winograd GEMM variant (coopmat2/coopmat1/dot2/tiled, each
// with FP32 and FP16 accumulation) under the same tier + penalty policy.
inline int64_t resolveWinogradGemmVariant(
  const VulkanTuneParams& tp,
  bool fp16,
  bool supportsCoopmat1F16,
  bool supportsCoopmat1F16AccF16,
  bool supportsCoopmat2F16,
  bool supportsCoopmat2F16AccF16,
  bool supportsDot2F16,
  bool supportsDot2F16AccF16) {
  if(!fp16)
    return VulkanTuner::TUNED_WINOGRAD_TILED;
  auto tuned = [&](int64_t bit) { return tp.hasKernelTuned(bit); };

  const VariantChoice coopTier[] = {
    {VulkanTuner::TUNED_WINOGRAD_COOPMAT2,
     tp.winogradGemmCoopmat2F32TimeUs,
     supportsCoopmat2F16 && tuned(VulkanTuner::TUNED_WINOGRAD_COOPMAT2),
     VARIANT_SELECT_PENALTY_NONE},
    {VulkanTuner::TUNED_WINOGRAD_COOPMAT1,
     tp.winogradGemmCoopmat1F32TimeUs,
     supportsCoopmat1F16 && tuned(VulkanTuner::TUNED_WINOGRAD_COOPMAT1),
     VARIANT_SELECT_PENALTY_NONE},
    {VulkanTuner::TUNED_WINOGRAD_COOPMAT2_ACCF16,
     tp.winogradGemmCoopmat2F16TimeUs,
     supportsCoopmat2F16AccF16 && tuned(VulkanTuner::TUNED_WINOGRAD_COOPMAT2_ACCF16),
     VARIANT_SELECT_PENALTY_ACCF16},
    {VulkanTuner::TUNED_WINOGRAD_COOPMAT1_ACCF16,
     tp.winogradGemmCoopmat1F16TimeUs,
     supportsCoopmat1F16AccF16 && tuned(VulkanTuner::TUNED_WINOGRAD_COOPMAT1_ACCF16),
     VARIANT_SELECT_PENALTY_ACCF16},
  };
  const VariantChoice dot2Tier[] = {
    {VulkanTuner::TUNED_WINOGRAD_DOT2,
     tp.winogradGemmDot2F32TimeUs,
     supportsDot2F16 && tuned(VulkanTuner::TUNED_WINOGRAD_DOT2),
     VARIANT_SELECT_PENALTY_NONE},
    {VulkanTuner::TUNED_WINOGRAD_DOT2_ACCF16,
     tp.winogradGemmDot2F16TimeUs,
     supportsDot2F16AccF16 && tuned(VulkanTuner::TUNED_WINOGRAD_DOT2_ACCF16),
     VARIANT_SELECT_PENALTY_ACCF16},
  };
  return selectVariantTier({coopTier, dot2Tier}, VulkanTuner::TUNED_WINOGRAD_TILED);
}

// Device-info overload: applies the filter-disabled variant mask to a scratch
// copy (hidden variants are never selectable), exactly like the runtime's
// filtered tune params, and returns the selected kernel bit.
inline int64_t resolveWinogradGemmVariant(const VulkanDeviceInfo& deviceInfo, const VulkanTuneParams& tp, bool fp16) {
  VulkanTuneParams effective = tp;
  effective.tunedKernelMask &= ~deviceInfo.disabledAccelVariantMask;
  return resolveWinogradGemmVariant(
    effective,
    fp16,
    deviceInfo.supportsCoopmat1F16,
    deviceInfo.supportsCoopmat1F16AccF16,
    deviceInfo.supportsCoopmat2F16,
    deviceInfo.supportsCoopmat2F16AccF16,
    deviceInfo.supportsDot2F16,
    deviceInfo.supportsDot2F16AccF16);
}

inline int64_t resolveWinogradGemmVariant(const VulkanLayerContext& ctx, bool fp16) {
  return resolveWinogradGemmVariant(
    ctx.tuneParams,
    fp16,
    ctx.supportsCoopmat1F16,
    ctx.supportsCoopmat1F16AccF16,
    ctx.supportsCoopmat2F16,
    ctx.supportsCoopmat2F16AccF16,
    ctx.supportsDot2F16,
    ctx.supportsDot2F16AccF16);
}

// The Winograd GEMM variant's M/N/K padding contract, read from the variant's
// row in kWinogradGemmVariantInfo (so the contract and the tile geometry above
// can never come from different variants).
inline LayerPaddingContract selectedWinogradPaddingContract(const VulkanTuneParams& tuneParams, int64_t variant) {
  return winogradGemmVariantInfo(variant).contract(tuneParams);
}

// mergedStridedPaddingContract: strided path applies N-padding only.
//
// Consumers should read .n only. Every strided kernel declares {m=1, k=1,
// kPaddable=false} in its contract, so the generic lcm-merge naturally
// returns {1, mergedN, 1, false} — .m and .k are informational (always 1).
// Runtime strided M is paddedSpatialSize (VULKAN_SPATIAL_ALIGN, globally
// pinned) and K is unpaddable across layers.
//
// The variant is resolved once (resolveGemmStridedVariant) and the N pad is
// looked up from that single decision, so this contract can never disagree
// with the kernel the layer will actually dispatch.
inline LayerPaddingContract mergedStridedPaddingContract(const VulkanLayerContext& ctx, bool fp16) {
  const int64_t variant = resolveGemmStridedVariant(ctx, fp16);
  const StridedGemmVariantInfo& info = stridedGemmVariantInfo(variant);
  // Accelerated variants pad N to their B-tile N dim; the tiled fallback has no
  // pre-packed B and declares its own contract.
  if(info.bnField == nullptr)
    return VulkanKernels::GemmStridedTiledNhwc::layerPaddingContract(ctx.tuneParams);
  return {1, ctx.tuneParams.*(info.bnField), 1, false};
}

// Pick the implicit-GEMM 3x3/5x5 conv variant: one coopmat tier holding all
// four flavors (coopmat1/coopmat2 x accF32/accF16), under the same fastest-wins
// + accF16-penalty policy as the GEMM families. Returns 0 when no variant is
// usable, in which case the caller falls back to the Winograd conv path.
//
// Note there is no dot2 tier here: dot2 has no implicit-GEMM conv shader, so
// the fallback below the coopmat tier is Winograd, not dot2.
inline int64_t resolveNativeConvVariant(const VulkanLayerContext& ctx, bool is5x5) {
  const VulkanTuneParams& tp = ctx.tuneParams;
  auto pick = [is5x5](int64_t b3, int64_t b5) { return is5x5 ? b5 : b3; };
  const int64_t cm1Bit = pick(VulkanTuner::TUNED_CONV3X3_COOPMAT1, VulkanTuner::TUNED_CONV5X5_COOPMAT1);
  const int64_t cm2Bit = pick(VulkanTuner::TUNED_CONV3X3_COOPMAT2, VulkanTuner::TUNED_CONV5X5_COOPMAT2);
  const int64_t cm1F16Bit =
    pick(VulkanTuner::TUNED_CONV3X3_COOPMAT1_ACCF16, VulkanTuner::TUNED_CONV5X5_COOPMAT1_ACCF16);
  const int64_t cm2F16Bit =
    pick(VulkanTuner::TUNED_CONV3X3_COOPMAT2_ACCF16, VulkanTuner::TUNED_CONV5X5_COOPMAT2_ACCF16);

  const VariantChoice coopTier[] = {
    {cm2Bit,
     is5x5 ? tp.conv5x5NhwcCoopmat2TimeUs : tp.conv3x3NhwcCoopmat2TimeUs,
     ctx.supportsCoopmat2F16 && tp.hasKernelTuned(cm2Bit),
     VARIANT_SELECT_PENALTY_NONE},
    {cm1Bit,
     is5x5 ? tp.conv5x5NhwcCoopmat1TimeUs : tp.conv3x3NhwcCoopmat1TimeUs,
     ctx.supportsCoopmat1F16 && tp.hasKernelTuned(cm1Bit),
     VARIANT_SELECT_PENALTY_NONE},
    {cm2F16Bit,
     is5x5 ? tp.conv5x5NhwcCoopmat2AccF16TimeUs : tp.conv3x3NhwcCoopmat2AccF16TimeUs,
     ctx.supportsCoopmat2F16AccF16 && tp.hasKernelTuned(cm2F16Bit),
     VARIANT_SELECT_PENALTY_ACCF16},
    {cm1F16Bit,
     is5x5 ? tp.conv5x5NhwcCoopmat1AccF16TimeUs : tp.conv3x3NhwcCoopmat1AccF16TimeUs,
     ctx.supportsCoopmat1F16AccF16 && tp.hasKernelTuned(cm1F16Bit),
     VARIANT_SELECT_PENALTY_ACCF16},
  };
  return selectWithinTier(coopTier);
}

// Tile geometry for the native 3x3/5x5 implicit-GEMM convolution, selected
// from the tune params once for the chosen coopmat family / accumulator type
// instead of a forest of per-field ternaries at each use site.
struct NativeConv3x3Tile {
  int workgroupSize;
  int bm;
  int bn;
  int bk;
  // Only the coopmat1 path consumes these (coopmat2 derives nothing from
  // per-subgroup tiling); they are still populated for all selections so the
  // struct stays uniform.
  int sgm;
  int sgn;
  int tm;
  int tn;
  int tk;
  int subgroupSize;
};

inline NativeConv3x3Tile
resolveNativeConv3x3Tile(const VulkanTuneParams& tuneParams, bool coopmat2Selected, bool accF16Selected, bool is5x5) {
  // The four (family, accumulator) combinations differ only in which .inc field
  // group they read; the groups are already declared in tile-member order in
  // vulkantuner_shared.h with static_asserts pinning their sizes. Read straight
  // out of the selected group instead of spelling out four brace-lists of
  // per-member ternaries.
  //
  // coopmat2 has no per-subgroup tiling, so its 4-field group fills only the
  // leading workgroup/BM/BN/BK members and the rest stay 0.
  NativeConv3x3Tile tile = {};
  int32_t* const members[] = {
    &tile.workgroupSize, &tile.bm, &tile.bn, &tile.bk, &tile.sgm,
    &tile.sgn, &tile.tm, &tile.tn, &tile.tk, &tile.subgroupSize};

  auto readGroup = [&](auto& group) {
    static_assert(
      std::extent_v<std::remove_reference_t<decltype(group)>> <= std::size(members),
      "tile field group larger than NativeConv3x3Tile");
    for(size_t i = 0; i < std::size(group); i++)
      *members[i] = tuneParams.*(group[i]);
  };

  if(coopmat2Selected) {
    if(accF16Selected)
      readGroup(
        is5x5 ? VulkanTuner::kConv5x5Coopmat2AccF16TileFields : VulkanTuner::kConv3x3Coopmat2AccF16TileFields);
    else
      readGroup(is5x5 ? VulkanTuner::kConv5x5Coopmat2TileFields : VulkanTuner::kConv3x3Coopmat2TileFields);
  } else if(accF16Selected) {
    readGroup(is5x5 ? VulkanTuner::kConv5x5Coopmat1AccF16TileFields : VulkanTuner::kConv3x3Coopmat1AccF16TileFields);
  } else {
    readGroup(is5x5 ? VulkanTuner::kConv5x5Coopmat1TileFields : VulkanTuner::kConv3x3Coopmat1TileFields);
  }
  return tile;
}

// BatchNorm + activation + mask. Owns one scaleBiasMaskAct kernel for its
// activation and dispatches itself via dispatch() (input -> output, in place ok).
struct BatchNormLayer {
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
    : numChannels(desc->numChannels), activation(actDesc->activation), device(ctx.device) {
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

  ~BatchNormLayer() { scaleBiasMaskActNhwcKernel.destroy(device); }

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
  const int inChannels;
  const int outChannels;
  VBuf weightBuf;  // [outChannels, inChannels] FP32
  VkDevice device;
  ComputeKernel gemmDirectKernel;

  MatMulLayer(const VulkanLayerContext& ctx, const MatMulLayerDesc* desc)
    : inChannels(desc->inChannels), outChannels(desc->outChannels), device(ctx.device) {
    testAssert(desc->weights.size() == (size_t)inChannels * (size_t)outChannels);
    std::vector<float> transW(inChannels * outChannels);
    for(int oc = 0; oc < outChannels; oc++)
      for(int ic = 0; ic < inChannels; ic++)
        transW[oc * inChannels + ic] = desc->weights[ic * outChannels + oc];
    weightBuf = makeWeightBuf(ctx.device, ctx.queue, ctx.queueMutex, ctx.commandPool, ctx.memProps, transW, false);
    gemmDirectKernel = VulkanKernels::GemmDirectFP32::build(ctx.device, ctx.pipelineCache, ctx.tuneParams, inChannels);
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
  const int numChannels;
  const int activation;
  VBuf biasBuf;
  VkDevice device;
  ComputeKernel addCBiasActKernel;

  MatBiasLayer(const VulkanLayerContext& ctx, const MatBiasLayerDesc* desc, const ActivationLayerDesc* actDesc)
    : numChannels(desc->numChannels),
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
  const int convYSize, convXSize;
  const int inChannels, outChannels;
  const int nnXLen, nnYLen, paddedSpatialSize;

  // For 3x3/5x5 Winograd
  int numTilesX, numTilesY;
  int inTileXYSize;
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

  VBuf filterBuf;

  VkDevice device;
  ComputeKernel gemmStridedKernel;
  ComputeKernel winogradTransformKernel;
  ComputeKernel winogradGemmKernel;
  ComputeKernel winogradUntransformKernel;
  ComputeKernel conv2dDirectNhwcKernel;
  ComputeKernel conv3x3ImplicitGemmNhwcVec8Kernel;
  int64_t gemmStridedVariant;
  int64_t winogradGemmVariant;

  bool useConv3x3ImplicitGemmNhwcVec8;
  bool useConv3x3WinogradNhwc;
  bool useConv2dDirectNhwc;
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
    : convYSize(desc->convYSize),
      convXSize(desc->convXSize),
      inChannels(desc->inChannels),
      outChannels(desc->outChannels),
      nnXLen(nnX),
      nnYLen(nnY),
      paddedSpatialSize(paddedSpatialSize_),
      device(ctx.device) {
    if(desc->dilationX != 1 || desc->dilationY != 1)
      throw StringError("Vulkan backend: convolution dilation != 1 not supported");

    workspaceElts2 = 0;
    numTilesX = numTilesY = 0;
    inTileXYSize = 0;
    inTileXSize = inTileYSize = 0;
    outTileXSize = outTileYSize = 0;
    numInChannelsPadded = numOutChannelsPadded = 0;
    numTilesPadded = 0;
    winogradMAlignment = (int)ctx.tuneParams.winogradGemmM;
    winogradPackedABM = 0;
    winogradPackedABK = 0;
    winogradPackedAPadWords = 0;
    winogradCoopmatBStride = 0;
    winogradTransformOffset = 0;
    numTilesPadded = 0;
    outChannelsPadded1x1 = 0;
    gemmStridedVariant = VulkanTuner::TUNED_GEMM_STRIDED_TILED;
    winogradGemmVariant = VulkanTuner::TUNED_WINOGRAD_TILED;
    useConv3x3ImplicitGemmNhwcVec8 = false;
    useConv3x3WinogradNhwc = false;
    useConv2dDirectNhwc = false;
    useConv3x3ImplicitGemmCoopmat2NhwcVec8 = false;
    useConv3x3ImplicitGemmAccF16NhwcVec8 = false;
    gemmStridedAddsToOutput = false;
    conv3x3ImplicitGemmAddsToOutput = false;
    inChannelsPhysical = inChannels;
    inChannelsPaddedConv3x3 = 0;
    outChannelsPaddedConv3x3 = 0;
    const bool isGemmStrided1x1 = convXSize == 1 && convYSize == 1;
    inChannelsPaddedConv3x3 = roundUpToMultipleInt(inChannels, 8);
    const bool isImplicitSquareConv = (convXSize == 3 && convYSize == 3) || (convXSize == 5 && convYSize == 5);
    const bool implicitConvIs5x5 = convXSize == 5 && convYSize == 5;
    // The implicit-GEMM conv variant is selected by the same tier + penalty
    // policy as every other accelerated family (see selectVariantTier): the
    // fastest measured coopmat variant wins, with FP16 accumulation taken only
    // when it clears the margin against FP32 accumulation.
    const int64_t convVariant = resolveNativeConvVariant(ctx, implicitConvIs5x5);
    const bool conv3x3Coopmat2Selected = convVariant == VulkanTuner::TUNED_CONV3X3_COOPMAT2 ||
                                         convVariant == VulkanTuner::TUNED_CONV5X5_COOPMAT2 ||
                                         convVariant == VulkanTuner::TUNED_CONV3X3_COOPMAT2_ACCF16 ||
                                         convVariant == VulkanTuner::TUNED_CONV5X5_COOPMAT2_ACCF16;
    const bool conv3x3AccF16Selected = convVariant == VulkanTuner::TUNED_CONV3X3_COOPMAT1_ACCF16 ||
                                       convVariant == VulkanTuner::TUNED_CONV5X5_COOPMAT1_ACCF16 ||
                                       convVariant == VulkanTuner::TUNED_CONV3X3_COOPMAT2_ACCF16 ||
                                       convVariant == VulkanTuner::TUNED_CONV5X5_COOPMAT2_ACCF16;

    const NativeConv3x3Tile nativeTile =
      resolveNativeConv3x3Tile(ctx.tuneParams, conv3x3Coopmat2Selected, conv3x3AccF16Selected, implicitConvIs5x5);
    const int nativeWorkgroupSize = nativeTile.workgroupSize;
    const int nativeBM = nativeTile.bm;
    const int nativeBN = nativeTile.bn;
    const int nativeBK = nativeTile.bk;
    const int nativeSGM = nativeTile.sgm;
    const int nativeSGN = nativeTile.sgn;
    const int nativeTM = nativeTile.tm;
    const int nativeTN = nativeTile.tn;
    const int nativeTK = nativeTile.tk;
    const int nativeSubgroupSize = nativeTile.subgroupSize;
    const bool conv3x3SelectedTileSupported =
      conv3x3Coopmat2Selected
        ? (conv3x3AccF16Selected ? VulkanKernels::Conv3x3ImplicitGemmCoopmat2AccF16NhwcVec8::isConfigSupported(
                                     nativeWorkgroupSize, nativeBM, nativeBN, nativeBK)
                                 : VulkanKernels::Conv3x3ImplicitGemmCoopmat2AccF32NhwcVec8::isConfigSupported(
                                     nativeWorkgroupSize, nativeBM, nativeBN, nativeBK))
        : (conv3x3AccF16Selected ? (VulkanKernels::Conv3x3ImplicitGemmCoopmat1AccF16NhwcVec8::isConfigSupported(
                                      nativeWorkgroupSize,
                                      nativeBM,
                                      nativeBN,
                                      nativeBK,
                                      nativeSGM,
                                      nativeSGN,
                                      nativeTM,
                                      nativeTN,
                                      nativeTK,
                                      nativeSubgroupSize) &&
                                    coopmatShapeSupported(ctx.coopmatAccF16Shapes, nativeTM, nativeTN, nativeTK))
                                 : (VulkanKernels::Conv3x3ImplicitGemmCoopmat1AccF32NhwcVec8::isConfigSupported(
                                      nativeWorkgroupSize,
                                      nativeBM,
                                      nativeBN,
                                      nativeBK,
                                      nativeSGM,
                                      nativeSGN,
                                      nativeTM,
                                      nativeTN,
                                      nativeTK,
                                      nativeSubgroupSize) &&
                                    coopmatShapeSupported(ctx.coopmatShapes, nativeTM, nativeTN, nativeTK))) &&
            ctx.subgroupSize == (uint32_t)nativeSubgroupSize;
    const bool conv3x3SelectedTileAligned = paddedSpatialSize % nativeBM == 0 && outChannels % nativeBN == 0;
    const bool canUseConv3x3ImplicitGemmNhwcVec8Base =
      isImplicitSquareConv && useFP16 &&
      // A nonzero variant means some coopmat flavor was tuned, measured, and is
      // device-supported; zero means fall back to the Winograd conv path.
      convVariant != 0 && (inChannelsPaddedConv3x3 % 8) == 0 &&
      // Most native graph tensors have an exact physical channel width. The
      // initial graph boundary may instead promise a zero-padded tail so that
      // inputs such as C=22 can use a physical C=24 vector stride.
      (inChannels == inChannelsPaddedConv3x3 || inputMayBeChannelPadded) && conv3x3SelectedTileSupported &&
      conv3x3SelectedTileAligned;
    const bool canUseConv3x3ImplicitGemmNhwcVec8 = canUseConv3x3ImplicitGemmNhwcVec8Base;
    // NHWC Winograd fallback (F(2,3)/F(4,3)/F(2,5)): the fast non-coopmat path
    // for 3x3/5x5 layers. It normally only applies when the coopmat vec8
    // implicit GEMM kernel isn't selected (checked first in the branch chain
    // below).
    const bool canUseConv3x3WinogradNhwc = isImplicitSquareConv && !canUseConv3x3ImplicitGemmNhwcVec8;

    // 1x1 convolutions are layout-invariant GEMMs and keep their existing
    // strided-GEMM path. The NHWC direct fallback is for spatial convolutions
    // that cannot use the specialized 3x3 kernel.
    if(!isGemmStrided1x1 && !canUseConv3x3ImplicitGemmNhwcVec8 && !canUseConv3x3WinogradNhwc) {
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
        const int packedBN = gemmStridedPackedBBN(ctx.tuneParams, gemmStridedVariant);
        const int packedBK = gemmStridedPackedBBK(ctx.tuneParams, gemmStridedVariant);
        const int packedPadScalars = gemmStridedPackedBPadScalars(gemmStridedVariant);
        const bool rowMajorPackedB = gemmStridedVariantUsesRowMajorPackedB(gemmStridedVariant);
        std::vector<float> packedW =
          rowMajorPackedB ? packStridedGemmBWeightsRowMajor(
                              transW, 1, outChannelsPadded1x1, inChannels, packedBN, packedBK, packedPadScalars)
                          : packStridedGemmBWeights(
                              transW, 1, outChannelsPadded1x1, inChannels, packedBN, packedBK, packedPadScalars);
        filterBuf =
          makeWeightBuf(ctx.device, ctx.queue, ctx.queueMutex, ctx.commandPool, ctx.memProps, packedW, useFP16);
      } else {
        filterBuf =
          makeWeightBuf(ctx.device, ctx.queue, ctx.queueMutex, ctx.commandPool, ctx.memProps, transW, useFP16);
      }
      if(gemmStridedVariant == VulkanTuner::TUNED_GEMM_STRIDED_COOPMAT2_ACCF16) {
        const int32_t tileM = ctx.tuneParams.nhwcStridedCoopmat2AccF16BM;
        const int32_t tileN = ctx.tuneParams.nhwcStridedCoopmat2AccF16BN;
        const int32_t tileK = ctx.tuneParams.nhwcStridedCoopmat2AccF16BK;
        int32_t aligned = (paddedSpatialSize % tileM == 0 && outChannelsPadded1x1 % tileN == 0) ? 1 : 0;
        int32_t kAligned = (inChannels % tileK == 0) ? 1 : 0;
        gemmStridedAddsToOutput = addToOutputOnStore;
        gemmStridedKernel = VulkanKernels::GemmStridedCoopmat2AccF16Nhwc::build(
          ctx.device, ctx.pipelineCache, ctx.tuneParams, aligned, kAligned, inChannels, addToOutputOnStore);
      } else if(gemmStridedVariant == VulkanTuner::TUNED_GEMM_STRIDED_COOPMAT2) {
        const int32_t tileM = ctx.tuneParams.nhwcStridedCoopmat2BM;
        const int32_t tileN = ctx.tuneParams.nhwcStridedCoopmat2BN;
        const int32_t tileK = ctx.tuneParams.nhwcStridedCoopmat2BK;
        int32_t aligned = (paddedSpatialSize % tileM == 0 && outChannelsPadded1x1 % tileN == 0) ? 1 : 0;
        int32_t kAligned = (inChannels % tileK == 0) ? 1 : 0;
        gemmStridedAddsToOutput = addToOutputOnStore;
        gemmStridedKernel = VulkanKernels::GemmStridedCoopmat2Nhwc::build(
          ctx.device, ctx.pipelineCache, ctx.tuneParams, aligned, kAligned, inChannels, addToOutputOnStore);
      } else if(gemmStridedVariant == VulkanTuner::TUNED_GEMM_STRIDED_COOPMAT1_ACCF16) {
        const int32_t tileM = ctx.tuneParams.nhwcStridedCoopmat1AccF16BM;
        const int32_t tileN = ctx.tuneParams.nhwcStridedCoopmat1AccF16BN;
        int32_t aligned = (paddedSpatialSize % tileM == 0 && outChannelsPadded1x1 % tileN == 0) ? 1 : 0;
        gemmStridedAddsToOutput = addToOutputOnStore;
        gemmStridedKernel = VulkanKernels::GemmStridedCoopmat1AccF16Nhwc::build(
          ctx.device, ctx.pipelineCache, ctx.tuneParams, aligned, inChannels, ctx.subgroupSize, addToOutputOnStore);
      } else if(gemmStridedVariant == VulkanTuner::TUNED_GEMM_STRIDED_COOPMAT1) {
        // Same padding contract as DOT2: N (outChannelsPadded1x1) is aligned by
        // mergedStridedPaddingContract; M is VULKAN_SPATIAL_ALIGN-pinned; K real.
        // ALIGNED reflects M/N safety only; K_ALIGNED handles the K-tail guard.
        const int32_t tileM = ctx.tuneParams.nhwcStridedCoopmat1BM;
        const int32_t tileN = ctx.tuneParams.nhwcStridedCoopmat1BN;
        int32_t aligned = (paddedSpatialSize % tileM == 0 && outChannelsPadded1x1 % tileN == 0) ? 1 : 0;
        gemmStridedAddsToOutput = addToOutputOnStore;
        gemmStridedKernel = VulkanKernels::GemmStridedCoopmat1Nhwc::build(
          ctx.device, ctx.pipelineCache, ctx.tuneParams, aligned, inChannels, ctx.subgroupSize, addToOutputOnStore);
      } else if(gemmStridedVariant == VulkanTuner::TUNED_GEMM_STRIDED_DOT2) {
        // 1x1 strided path: N (outChannelsPadded1x1) is DOT2-aligned by
        // mergedStridedPaddingContract; M (paddedSpatialSize) is globally
        // pinned to VULKAN_SPATIAL_ALIGN and NOT per-variant re-padded; K
        // stays real (kPaddable:false). ALIGNED reflects M/N safety only;
        // K_ALIGNED handles the K-tail guard.
        gemmStridedAddsToOutput = addToOutputOnStore;

        const int32_t tileM = ctx.tuneParams.nhwcStridedDot2BM;
        const int32_t tileN = ctx.tuneParams.nhwcStridedDot2BN;
        int32_t aligned = (paddedSpatialSize % tileM == 0 && outChannelsPadded1x1 % tileN == 0) ? 1 : 0;
        int32_t kAligned = (inChannels % (int)VulkanKernels::DOT2_BK == 0) ? 1 : 0;
        gemmStridedKernel = VulkanKernels::GemmStridedDot2Nhwc::build(
          ctx.device,
          ctx.pipelineCache,
          ctx.tuneParams,
          aligned,
          VulkanKernels::DOT2_PACKED_B,
          kAligned,
          inChannels,
          addToOutputOnStore,
          ctx.canRequireReportedSubgroupSize ? ctx.subgroupSize : 0u);

      } else if(gemmStridedVariant == VulkanTuner::TUNED_GEMM_STRIDED_DOT2_ACCF16) {
        gemmStridedAddsToOutput = addToOutputOnStore;

        const int32_t tileM = ctx.tuneParams.nhwcStridedDot2AccF16BM;
        const int32_t tileN = ctx.tuneParams.nhwcStridedDot2AccF16BN;
        int32_t aligned = (paddedSpatialSize % tileM == 0 && outChannelsPadded1x1 % tileN == 0) ? 1 : 0;
        int32_t kAligned = (inChannels % (int)VulkanKernels::DOT2_BK == 0) ? 1 : 0;
        gemmStridedKernel = VulkanKernels::GemmStridedDot2AccF16Nhwc::build(
          ctx.device,
          ctx.pipelineCache,
          ctx.tuneParams,
          aligned,
          VulkanKernels::DOT2_PACKED_B,
          kAligned,
          inChannels,
          addToOutputOnStore,
          ctx.canRequireReportedSubgroupSize ? ctx.subgroupSize : 0u);

      } else {
        gemmStridedKernel = VulkanKernels::GemmStridedTiledNhwc::build(
          ctx.device, ctx.pipelineCache, useFP16, ctx.tuneParams, inChannels, addToOutputOnStore);
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
        filterWeights, 1, outChannelsPaddedConv3x3, packedK, nativeBN, nativeBK, STRIDED_COOPMAT_PACKED_B_PAD_SCALARS);
      filterBuf =
        makeWeightBuf(ctx.device, ctx.queue, ctx.queueMutex, ctx.commandPool, ctx.memProps, filterWeights, useFP16);
      if(useConv3x3ImplicitGemmCoopmat2NhwcVec8) {
        conv3x3ImplicitGemmNhwcVec8Kernel =
          useConv3x3ImplicitGemmAccF16NhwcVec8
            ? VulkanKernels::Conv3x3ImplicitGemmCoopmat2AccF16NhwcVec8::build(
                ctx.device, ctx.pipelineCache, ctx.tuneParams, convXSize, packedK, addToOutputOnStore)
            : VulkanKernels::Conv3x3ImplicitGemmCoopmat2AccF32NhwcVec8::build(
                ctx.device, ctx.pipelineCache, ctx.tuneParams, convXSize, packedK, addToOutputOnStore);
      } else {
        conv3x3ImplicitGemmNhwcVec8Kernel = useConv3x3ImplicitGemmAccF16NhwcVec8
                                              ? VulkanKernels::Conv3x3ImplicitGemmCoopmat1AccF16NhwcVec8::build(
                                                  ctx.device,
                                                  ctx.pipelineCache,
                                                  ctx.tuneParams,
                                                  convXSize,
                                                  flattenedK,
                                                  addToOutputOnStore,
                                                  (inChannelsPaddedConv3x3 % nativeBK) == 0,
                                                  ctx.subgroupSize)
                                              : VulkanKernels::Conv3x3ImplicitGemmCoopmat1AccF32NhwcVec8::build(
                                                  ctx.device,
                                                  ctx.pipelineCache,
                                                  ctx.tuneParams,
                                                  convXSize,
                                                  flattenedK,
                                                  addToOutputOnStore,
                                                  (inChannelsPaddedConv3x3 % nativeBK) == 0,
                                                  ctx.subgroupSize);
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
      winogradGemmVariant = resolveWinogradGemmVariant(ctx, useFP16);
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

      std::pair<int, VBuf> packedFilter = makeWinogradFilterBuf(ctx, transWeightsNhwc, useFP16);
      winogradCoopmatBStride = packedFilter.first;
      filterBuf = std::move(packedFilter.second);

      workspaceElts2 = static_cast<size_t>(inTileXYSize) * numOutChannelsPadded;

      {
        const int convSz = convXSize;
        const int offset = winogradTransformOffset;
        winogradTransformKernel = VulkanKernels::WinogradTransformNhwc::build(
          ctx.device,
          ctx.pipelineCache,
          useFP16,
          ctx.tuneParams,
          convSz,
          offset,
          winogradPackedABM,
          winogradPackedABK,
          winogradPackedAPadWords);
        winogradGemmKernel = buildWinogradGemmKernel(ctx, useFP16);
        // outChannels here is the unpadded physical output stride (the C
        // tensor's contiguous NHWC channel run), not numOutChannelsPadded --
        // padding is a GEMM-internal concern and irrelevant to whether the
        // output store can address 4 real channels at once.
        nhwcWinogradUntransformOcVec = (outChannels % 4 == 0) ? 4 : 1;
        winogradUntransformKernel = VulkanKernels::WinogradUntransformNhwc::build(
          ctx.device,
          ctx.pipelineCache,
          useFP16,
          ctx.tuneParams,
          convSz,
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
    conv2dDirectNhwcKernel.destroy(device);
    conv3x3ImplicitGemmNhwcVec8Kernel.destroy(device);
  }

  // Upload the Winograd B tensor for the selected Winograd GEMM variant.
  // Coopmat variants upload the packed B layout and return its stride;
  // all other variants upload the raw transformed weights with a zero stride.
  std::pair<int, VBuf>
  makeWinogradFilterBuf(const VulkanLayerContext& ctx, const std::vector<float>& transWeightsNhwc, bool useFP16) const {
    const bool coopmat2 = winogradGemmVariant == VulkanTuner::TUNED_WINOGRAD_COOPMAT2 ||
                          winogradGemmVariant == VulkanTuner::TUNED_WINOGRAD_COOPMAT2_ACCF16;
    if(
      coopmat2 || winogradGemmVariant == VulkanTuner::TUNED_WINOGRAD_COOPMAT1 ||
      winogradGemmVariant == VulkanTuner::TUNED_WINOGRAD_COOPMAT1_ACCF16) {
      const bool accF16 = winogradGemmVariant == VulkanTuner::TUNED_WINOGRAD_COOPMAT2_ACCF16 ||
                          winogradGemmVariant == VulkanTuner::TUNED_WINOGRAD_COOPMAT1_ACCF16;
      const int bn = coopmat2 ? (accF16 ? ctx.tuneParams.coopmat2AccF16BN : ctx.tuneParams.coopmat2BN)
                              : (accF16 ? ctx.tuneParams.coopmat1AccF16BN : ctx.tuneParams.coopmat1BN);
      const int bk = coopmat2 ? (accF16 ? ctx.tuneParams.coopmat2AccF16BK : ctx.tuneParams.coopmat2BK)
                              : (accF16 ? ctx.tuneParams.coopmat1AccF16BK : ctx.tuneParams.coopmat1BK);
      // Coopmat2 has an independently tuned pad width; coopmat1 uses the shared
      // default (both are currently 2 words, but remain separate constants).
      size_t packedStrideB;
      std::vector<float> packedWeights;
      if(coopmat2) {
        packedStrideB = winogradCoopmatPackedBStrideWords(
          numOutChannelsPadded, numInChannelsPadded, bn, bk, WINOGRAD_COOPMAT2_PACKED_PAD_WORDS);
        packedWeights = packWinogradCoopmatBWeights(
          transWeightsNhwc,
          inTileXYSize,
          numOutChannelsPadded,
          numInChannelsPadded,
          bn,
          bk,
          WINOGRAD_COOPMAT2_PACKED_PAD_WORDS);
      } else {
        packedStrideB = winogradCoopmatPackedBStrideWords(numOutChannelsPadded, numInChannelsPadded, bn, bk);
        packedWeights = packWinogradCoopmatBWeights(
          transWeightsNhwc, inTileXYSize, numOutChannelsPadded, numInChannelsPadded, bn, bk);
      }
      testAssert(packedStrideB <= (size_t)std::numeric_limits<int>::max());
      return {
        (int)packedStrideB,
        makeWeightBuf(ctx.device, ctx.queue, ctx.queueMutex, ctx.commandPool, ctx.memProps, packedWeights, useFP16)};
    }
    return {
      0,
      makeWeightBuf(ctx.device, ctx.queue, ctx.queueMutex, ctx.commandPool, ctx.memProps, transWeightsNhwc, useFP16)};
  }

  // Build the Winograd GEMM pipeline for the selected variant.
  ComputeKernel buildWinogradGemmKernel(const VulkanLayerContext& ctx, bool useFP16) const {
    switch(winogradGemmVariant) {
      case VulkanTuner::TUNED_WINOGRAD_COOPMAT2_ACCF16:
        return VulkanKernels::WinogradGemmCoopmat2AccF16::build(
          ctx.device, ctx.pipelineCache, ctx.tuneParams, numInChannelsPadded);
      case VulkanTuner::TUNED_WINOGRAD_COOPMAT2:
        return VulkanKernels::WinogradGemmCoopmat2::build(
          ctx.device, ctx.pipelineCache, ctx.tuneParams, numInChannelsPadded);
      case VulkanTuner::TUNED_WINOGRAD_COOPMAT1_ACCF16:
        return VulkanKernels::WinogradGemmCoopmat1AccF16::build(
          ctx.device, ctx.pipelineCache, ctx.tuneParams, numInChannelsPadded, ctx.subgroupSize);
      case VulkanTuner::TUNED_WINOGRAD_COOPMAT1:
        return VulkanKernels::WinogradGemmCoopmat1::build(
          ctx.device, ctx.pipelineCache, ctx.tuneParams, numInChannelsPadded, ctx.subgroupSize);
      case VulkanTuner::TUNED_WINOGRAD_DOT2:
        return VulkanKernels::WinogradGemmDot2::build(
          ctx.device,
          ctx.pipelineCache,
          ctx.tuneParams,
          numInChannelsPadded,
          ctx.canRequireReportedSubgroupSize ? ctx.subgroupSize : 0u);
      case VulkanTuner::TUNED_WINOGRAD_DOT2_ACCF16:
        return VulkanKernels::WinogradGemmDot2AccF16::build(
          ctx.device,
          ctx.pipelineCache,
          ctx.tuneParams,
          numInChannelsPadded,
          ctx.canRequireReportedSubgroupSize ? ctx.subgroupSize : 0u);
      default:
        return VulkanKernels::WinogradGemm::build(
          ctx.device, ctx.pipelineCache, useFP16, ctx.tuneParams, numInChannelsPadded);
    }
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
    int maxBatchSize) const {
    const ConvLayer& conv = *this;
    if(conv.useConv2dDirectNhwc) {
      testAssert(scratch != nullptr);
      // Native NHWC graph: input and output are graph-owned NHWC buffers.
      VulkanKernels::Conv2dDirectNhwc::PC pc = {
        conv.nnXLen,
        conv.nnYLen,
        conv.outChannels,
        conv.inChannels,
        conv.convXSize / 2,
        conv.convYSize / 2,
        conv.paddedSpatialSize,
        maxBatchSize};
      VulkanKernels::Conv2dDirectNhwc::dispatch(
        ctx, conv.conv2dDirectNhwcKernel, input, conv.filterBuf.get(), output, pc);
      return;
    } else if(conv.convXSize == 1 && conv.convYSize == 1) {
      testAssert(conv.paddedSpatialSize % 4 == 0);
      testAssert(conv.outChannelsPadded1x1 % 8 == 0);
      VulkanKernels::GemmStridedTiledNhwc::PC pc = {
        conv.paddedSpatialSize,
        conv.outChannelsPadded1x1,
        conv.outChannels,
        conv.inChannels * (conv.paddedSpatialSize / 4),
        0,
        conv.outChannels * (conv.paddedSpatialSize / 4)};  // strideC uses real N
      switch(conv.gemmStridedVariant) {
        case VulkanTuner::TUNED_GEMM_STRIDED_COOPMAT2:
          VulkanKernels::GemmStridedCoopmat2Nhwc::dispatch(
            ctx, conv.gemmStridedKernel, input, conv.filterBuf.get(), output, conv.inChannels, pc, maxBatchSize);
          break;
        case VulkanTuner::TUNED_GEMM_STRIDED_COOPMAT2_ACCF16:
          VulkanKernels::GemmStridedCoopmat2AccF16Nhwc::dispatch(
            ctx, conv.gemmStridedKernel, input, conv.filterBuf.get(), output, conv.inChannels, pc, maxBatchSize);
          break;
        case VulkanTuner::TUNED_GEMM_STRIDED_COOPMAT1:
          VulkanKernels::GemmStridedCoopmat1Nhwc::dispatch(
            ctx, conv.gemmStridedKernel, input, conv.filterBuf.get(), output, conv.inChannels, pc, maxBatchSize);
          break;
        case VulkanTuner::TUNED_GEMM_STRIDED_COOPMAT1_ACCF16:
          testAssert(scratch != nullptr);
          VulkanKernels::GemmStridedCoopmat1AccF16Nhwc::dispatch(
            ctx, conv.gemmStridedKernel, input, conv.filterBuf.get(), output, conv.inChannels, pc, maxBatchSize);
          break;
        case VulkanTuner::TUNED_GEMM_STRIDED_DOT2:
          VulkanKernels::GemmStridedDot2Nhwc::dispatch(
            ctx, conv.gemmStridedKernel, input, conv.filterBuf.get(), output, conv.inChannels, pc, maxBatchSize);
          break;
        case VulkanTuner::TUNED_GEMM_STRIDED_DOT2_ACCF16:
          VulkanKernels::GemmStridedDot2AccF16Nhwc::dispatch(
            ctx, conv.gemmStridedKernel, input, conv.filterBuf.get(), output, conv.inChannels, pc, maxBatchSize);
          break;
        case VulkanTuner::TUNED_GEMM_STRIDED_TILED:
          VulkanKernels::GemmStridedTiledNhwc::dispatch(
            ctx, conv.gemmStridedKernel, input, conv.filterBuf.get(), output, conv.inChannels, pc, maxBatchSize);
          break;
        default:
          ASSERT_UNREACHABLE;
      }
    } else if(conv.useConv3x3ImplicitGemmNhwcVec8) {
      // Dispatch the specialized NHWC convolution.
      testAssert(scratch != nullptr);
      // The specialized shader needs padded input channels.  Convolutional
      // trunk channels are normally already multiples of eight; the older
      // packing fallback only applies to the exceptional channel-tail case.
      if(conv.inChannelsPhysical == conv.inChannelsPaddedConv3x3) {
        testAssert((conv.outChannels * conv.paddedSpatialSize) % 4 == 0);
        VulkanKernels::Conv3x3ImplicitGemmCoopmat1AccF32NhwcVec8::PC pc = {
          conv.nnXLen,
          conv.nnYLen,
          conv.outChannels,
          conv.outChannelsPaddedConv3x3,
          conv.inChannelsPaddedConv3x3,
          conv.paddedSpatialSize,
          0,
          (conv.outChannels * conv.paddedSpatialSize) / 4,
          0};
        if(conv.useConv3x3ImplicitGemmCoopmat2NhwcVec8) {
          if(conv.useConv3x3ImplicitGemmAccF16NhwcVec8)
            VulkanKernels::Conv3x3ImplicitGemmCoopmat2AccF16NhwcVec8::dispatch(
              ctx,
              conv.conv3x3ImplicitGemmNhwcVec8Kernel,
              input,
              conv.filterBuf.get(),
              output,
              conv.convXSize,
              pc,
              maxBatchSize);
          else
            VulkanKernels::Conv3x3ImplicitGemmCoopmat2AccF32NhwcVec8::dispatch(
              ctx,
              conv.conv3x3ImplicitGemmNhwcVec8Kernel,
              input,
              conv.filterBuf.get(),
              output,
              conv.convXSize,
              pc,
              maxBatchSize);
        } else if(conv.useConv3x3ImplicitGemmAccF16NhwcVec8)
          VulkanKernels::Conv3x3ImplicitGemmCoopmat1AccF16NhwcVec8::dispatch(
            ctx,
            conv.conv3x3ImplicitGemmNhwcVec8Kernel,
            input,
            conv.filterBuf.get(),
            output,
            conv.convXSize,
            pc,
            maxBatchSize);
        else
          VulkanKernels::Conv3x3ImplicitGemmCoopmat1AccF32NhwcVec8::dispatch(
            ctx,
            conv.conv3x3ImplicitGemmNhwcVec8Kernel,
            input,
            conv.filterBuf.get(),
            output,
            conv.convXSize,
            pc,
            maxBatchSize);
        return;
      }
      testAssert(false);
    } else if(conv.useConv3x3WinogradNhwc) {
      // Normalization and activation are applied by the caller before this
      // channel-innermost Winograd convolution.
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
        conv.winogradGemmVariant == VulkanTuner::TUNED_WINOGRAD_COOPMAT1 ||
        conv.winogradGemmVariant == VulkanTuner::TUNED_WINOGRAD_COOPMAT1_ACCF16 ||
        conv.winogradGemmVariant == VulkanTuner::TUNED_WINOGRAD_COOPMAT2_ACCF16 ||
        conv.winogradGemmVariant == VulkanTuner::TUNED_WINOGRAD_COOPMAT2) {
        testAssert(conv.winogradCoopmatBStride > 0);
        strideB = conv.winogradCoopmatBStride;
      }
      VulkanKernels::WinogradGemm::PC pcGemm = {
        numTilesPaddedRun,
        conv.numOutChannelsPadded,
        strideA,                                               // strideA, vec4 units
        strideB,                                               // strideB, vec4 units
        (conv.numOutChannelsPadded / 4) * numTilesPaddedRun};  // strideC, vec4 units
      if(conv.winogradGemmVariant == VulkanTuner::TUNED_WINOGRAD_COOPMAT2_ACCF16) {
        VulkanKernels::WinogradGemmCoopmat2AccF16::dispatch(
          ctx,
          conv.winogradGemmKernel,
          ws0,
          conv.filterBuf.get(),
          ws1,
          conv.numInChannelsPadded,
          pcGemm,
          conv.inTileXYSize);
      } else if(conv.winogradGemmVariant == VulkanTuner::TUNED_WINOGRAD_COOPMAT2) {
        VulkanKernels::WinogradGemmCoopmat2::dispatch(
          ctx,
          conv.winogradGemmKernel,
          ws0,
          conv.filterBuf.get(),
          ws1,
          conv.numInChannelsPadded,
          pcGemm,
          conv.inTileXYSize);
      } else if(conv.winogradGemmVariant == VulkanTuner::TUNED_WINOGRAD_COOPMAT1_ACCF16) {
        VulkanKernels::WinogradGemmCoopmat1AccF16::dispatch(
          ctx,
          conv.winogradGemmKernel,
          ws0,
          conv.filterBuf.get(),
          ws1,
          conv.numInChannelsPadded,
          pcGemm,
          conv.inTileXYSize);
      } else if(conv.winogradGemmVariant == VulkanTuner::TUNED_WINOGRAD_COOPMAT1) {
        VulkanKernels::WinogradGemmCoopmat1::dispatch(
          ctx,
          conv.winogradGemmKernel,
          ws0,
          conv.filterBuf.get(),
          ws1,
          conv.numInChannelsPadded,
          pcGemm,
          conv.inTileXYSize);
      } else if(conv.winogradGemmVariant == VulkanTuner::TUNED_WINOGRAD_DOT2) {
        VulkanKernels::WinogradGemmDot2::dispatch(
          ctx,
          conv.winogradGemmKernel,
          ws0,
          conv.filterBuf.get(),
          ws1,
          conv.numInChannelsPadded,
          pcGemm,
          conv.inTileXYSize);
      } else if(conv.winogradGemmVariant == VulkanTuner::TUNED_WINOGRAD_DOT2_ACCF16) {
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

// Fold src's per-slot byte sizes into dst by element-wise max (slots are shared
// across blocks that execute sequentially).
inline void mergePermanentScratchSlots(std::vector<VkDeviceSize>& dst, const std::vector<VkDeviceSize>& src) {
  for(size_t i = 0; i < src.size(); i++) {
    if(i >= dst.size())
      dst.push_back(src[i]);
    else
      dst[i] = std::max(dst[i], src[i]);
  }
}

// A normalization + activation followed by a convolution, the reusable building
// block of every residual-style block. May fuse a residual add into the conv's
// store (single-writer kernels only).
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
        addToOutputOnStore && !conv.useConv2dDirectNhwc &&
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

// A standard residual block: trunk -> norm/act/conv -> norm/act/conv (with the
// second conv fusing the residual add) -> add back into the trunk.
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
    : normActConv1(ctx, &desc->preBN, &desc->preActivation, &desc->regularConv, nnX, nnY, paddedSpatialSize_, useFP16),
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
    [[maybe_unused]] VulkanBuffer* /*maskSum*/,
    int maxBatchSize) const override {
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
};

// A residual block that also computes a global-pooled side branch, folds it into
// a dynamic bias via a MatMul, and adds it back before the second conv.
struct GlobalPoolingResidualBlockVk : VulkanBlockVk {
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
  ComputeKernel gpoolNhwcKernel;
  ComputeKernel scaleBiasMaskActNhwcDynamicBiasKernel;
  ComputeKernel addPointwiseKernel;

  GlobalPoolingResidualBlockVk(
    const VulkanLayerContext& ctx,
    const GlobalPoolingResidualBlockDesc* desc,
    int nnX,
    int nnY,
    int paddedSpatialSize_,
    bool useFP16)
    : preBN(ctx, &desc->preBN, &desc->preActivation, useFP16),
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
    gpoolNhwcKernel = VulkanKernels::GPoolReductionNhwc::build(ctx.device, ctx.pipelineCache, useFP16, ctx.tuneParams);
    scaleBiasMaskActNhwcDynamicBiasKernel = VulkanKernels::ScaleBiasMaskActInplaceNhwc::build(
      ctx.device, ctx.pipelineCache, useFP16, normActConv2.norm.activation, regularChannels % 4 == 0, true);

    addPointwiseKernel = VulkanKernels::AddPointwise::build(ctx.device, ctx.pipelineCache, useFP16);
  }

  GlobalPoolingResidualBlockVk() = delete;
  GlobalPoolingResidualBlockVk(const GlobalPoolingResidualBlockVk&) = delete;
  GlobalPoolingResidualBlockVk& operator=(const GlobalPoolingResidualBlockVk&) = delete;

  ~GlobalPoolingResidualBlockVk() {
    gpoolNhwcKernel.destroy(device);
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

// A nested-bottleneck residual block: outer norm/act/conv around an inner
// BlockStack of (usually smaller-channel) blocks, with a fused residual add.
struct NestedBottleneckResidualBlockVk : VulkanBlockVk {
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

// The SGF metadata encoder head: a small three-MatMul feedforward network that
// turns the trunk's global-pooled summary into the SGF-annotation metadata vector.
struct SGFMetadataEncoderVk {
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

  // Per-position RMSNorm with no bias (transformer blocks): the same kernel and
  // PC as the non-spatial form, with the weight as gamma and zeros as beta.
  RMSNormVk(const VulkanLayerContext& ctx, const TransformerRMSNormDesc* desc, int paddedSpatialSize_, bool useFP16);

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

// TransformerMatMulLayerVk: per-position linear projection (1x1 spatial conv)
struct TransformerMatMulLayerVk {
  const int inChannels;
  const int outChannels;
  const int paddedSpatialSize;
  const bool addToOutput;
  std::unique_ptr<ConvLayer> nhwcConv;

  TransformerMatMulLayerVk(
    const VulkanLayerContext& ctx,
    const MatMulLayerDesc* desc,
    int paddedSpatialSize_,
    bool useFP16,
    bool addToOutput_ = false,
    int padNumGroups = 0,
    int padTrueChPerGroup = 0,
    int padPaddedChPerGroup = 0,
    bool padOutput = true);

  TransformerMatMulLayerVk() = delete;
  TransformerMatMulLayerVk(const TransformerMatMulLayerVk&) = delete;
  TransformerMatMulLayerVk& operator=(const TransformerMatMulLayerVk&) = delete;

  // Apply as 1x1 spatial convolution: [N, inC, padXY] -> [N, outC, padXY]
  void dispatch(const CmdCtx& ctx, ScratchBuffers* scratch, VulkanBuffer* input, VulkanBuffer* output, int maxBatchSize)
    const;
};

struct AttentionPadDims {
  int qHeadDim = 0;
  int vHeadDim = 0;
  bool padded = false;
};

// A multi-head self/cross attention block (RMSNorm -> Q/K/V/out projections ->
// the selected attention kernel -> residual add), plus optional RoPE.
struct TransformerAttentionBlockVk : VulkanBlockVk {
  enum class AttentionVariant { Tiled, Coopmat1AccF32, CoopmatMaintenance1AccF32, Coopmat2AccF32, Dot2AccF32 };
  const int numHeads;
  const int numKVHeads;
  const int qHeadDim;
  const int vHeadDim;
  const AttentionPadDims padDims;
  const int qHeadDimPadded;
  const int vHeadDimPadded;
  const bool paddedHeads;
  const bool useRope;
  const bool learnableRope;
  const int inChannels;
  const int paddedSpatialSize;
  const AttentionVariant attentionVariant;

  RMSNormVk preLN;
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
    [[maybe_unused]] VulkanBuffer* /*maskSum*/,
    int maxBatchSize) const override;  // defined in vulkanlayers.cpp
};

// A transformer feed-forward block (RMSNorm -> linear1 -> SwiGLU gate -> linear2
// -> residual add). Gate is optional; when absent it's a plain two-MatMul MLP.
struct TransformerFFNBlockVk : VulkanBlockVk {
  const int numChannels;
  const int ffnChannels;
  const int paddedSpatialSize;

  RMSNormVk preLN;
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
    [[maybe_unused]] VulkanBuffer* /*maskSum*/,
    int maxBatchSize) const override;  // defined in vulkanlayers.cpp
};

// An ordered sequence of blocks (residual / bottleneck / attention / FFN) that
// makes up the trunk's main body. Dispatches each block in turn.
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

// The network trunk: initial conv / matmul / metadata encoder, the block stack,
// and the final tip normalization (BN or RMSNorm).
struct TrunkVk {
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
  ComputeKernel addChannelBiasesNhwcKernel;

  TrunkVk() = delete;
  TrunkVk(const TrunkVk&) = delete;
  TrunkVk& operator=(const TrunkVk&) = delete;

  ~TrunkVk() { addChannelBiasesNhwcKernel.destroy(device); }

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

// The policy head: conv -> global pool -> MatMul -> policy output (and the pass
// logit), fed from the trunk's post-tip output.
struct PolicyHeadVk {
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
  ComputeKernel gpoolNhwcKernel;
  ComputeKernel addChannelBiasesNhwcKernel;

  PolicyHeadVk() = delete;
  PolicyHeadVk(const PolicyHeadVk&) = delete;
  PolicyHeadVk& operator=(const PolicyHeadVk&) = delete;

  ~PolicyHeadVk() {
    gpoolNhwcKernel.destroy(device);
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

// The value head: conv -> pool -> MatMuls producing value, score-value, and
// ownership outputs.
struct ValueHeadVk {
  const int paddedSpatialSize;
  const int v1Channels;
  const int v2Channels;

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
  ComputeKernel valueHeadPoolNhwcKernel;

  ValueHeadVk() = delete;
  ValueHeadVk(const ValueHeadVk&) = delete;
  ValueHeadVk& operator=(const ValueHeadVk&) = delete;

  ~ValueHeadVk() { valueHeadPoolNhwcKernel.destroy(device); }

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

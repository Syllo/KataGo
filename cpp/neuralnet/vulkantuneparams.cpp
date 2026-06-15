#ifdef USE_VULKAN_BACKEND

// VulkanTuneParams reconcile / load / save / validate / summary and the top-level
// `katago tuner` command. Split out of vulkantuner.cpp so that file holds only the
// tuning machinery; the VulkanKernels::*::bench implementations live in
// vulkantunebench.cpp. The bench-support helpers (coopmatShapeIsSupported,
// makeWinogradTransformBenchLayout, WinogradTransformBenchLayout) stay in
// vulkantuner.cpp and are declared in vulkantuner.h; the cross-TU tuner internals
// shared with the tuning machinery are in vulkantuner_shared.h.

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <initializer_list>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>
#include "../command/commandline.h"
#include "../core/fileutils.h"
#include "../core/makedir.h"
#include "../dataio/homedata.h"
#include "../neuralnet/desc.h"
#include "../neuralnet/vulkanbackend.h"
#include "../neuralnet/vulkanhelpers.h"
#include "../neuralnet/vulkankernels.h"
#include "../neuralnet/vulkanlayers.h"
#include "../neuralnet/vulkantuner.h"
#include "../neuralnet/vulkantuner_shared.h"
#include "../program/setup.h"

using namespace std;
using namespace VulkanHelpers;

namespace {

  bool isPow2(int32_t v) {
    return v > 0 && (v & (v - 1)) == 0;
  }

  string uniqueTempFileNameForAtomicSave(const string& filename) {
    static const uint64_t randBase = std::random_device{}();
    static std::atomic<uint64_t> counter{0};
    return Global::strprintf(
      "%s.tmp_%llx_%llu", filename.c_str(), (unsigned long long)randBase, (unsigned long long)counter.fetch_add(1));
  }

  void finishAtomicSave(const string& filename, const string& tmpFilename, ofstream& out) {
    out.close();
    if(out.fail()) {
      FileUtils::tryRemoveFile(tmpFilename);
      throw IOError("Could not write complete temp file " + tmpFilename + " while saving " + filename);
    }
    try {
      FileUtils::rename(tmpFilename, filename);
    } catch(const IOError&) {
      FileUtils::tryRemoveFile(tmpFilename);
      throw;
    }
  }

  // Keep this in lockstep with VULKAN_TUNEPARAMS_VERSION_LINE: the cache
  // filename encodes this number (tune<N>_...) and the file content declares
  // the same VERSION=<N> line.
  constexpr int VULKAN_TUNER_VERSION = 1;
  constexpr string_view VULKAN_TUNEPARAMS_VERSION_LINE = "VERSION=1";

}  // namespace

// Layout-signature and invalid-bit helpers shared with the tuning machinery
// (declared in vulkantuner_shared.h).
namespace VulkanTuner {

  std::string winogradTransformLayoutSignature(
    const VulkanTuneParams& config,
    const VulkanDeviceInfo& deviceInfo,
    bool fp16Storage,
    bool fp16Compute) {
    VulkanTuner::WinogradTransformBenchLayout layout =
      VulkanTuner::makeWinogradTransformBenchLayout(config, deviceInfo, fp16Storage, fp16Compute);
    std::ostringstream out;
    out << (fp16Storage ? 1 : 0) << "-" << layout.mAlignment << "-" << layout.nAlignment << "-" << layout.kAlignment
        << "-" << layout.packedBM << "-" << layout.packedBK << "-" << layout.packedAPadWords;
    return out.str();
  }

  void markNhwcWinogradTransformsTunedForCurrentLayout(
    VulkanTuneParams& config,
    const VulkanDeviceInfo& deviceInfo,
    bool fp16Storage,
    bool fp16Compute) {
    config.nhwcWinogradTransformTunedLayout =
      winogradTransformLayoutSignature(config, deviceInfo, fp16Storage, fp16Compute);
  }

}  // namespace VulkanTuner

// One row per tunable kernel/variant. This is the single place mapping a
// TunedKernel bit to its display label and reconcile gating; the per-kernel
// reason checks and field resets live next to their consumers below.
namespace VulkanTuner {

  struct KernelFamilyInfo {
    int64_t bit;
    const char* label;
    bool gated;  // true => reconcile validates it only when hasKernelTuned()
  };

  static const std::vector<KernelFamilyInfo>& kernelFamilyInfos() {
    static const std::vector<KernelFamilyInfo> infos = {
      {TUNED_GEMM_DIRECT, "gemmDirect", false},
      {TUNED_NCHW_TO_NHWC, "nchwToNhwc", false},
      {TUNED_GPOOL_NHWC, "gpoolNhwc", false},
      {TUNED_VALUE_HEAD_POOL_NHWC, "valueHeadPoolNhwc", false},
      {TUNED_SPATIAL_RMSNORM_NHWC, "spatialRMSNormNhwc", false},
      {TUNED_GEMM_STRIDED_TILED, "gemmStridedTiledNhwc", false},
      {TUNED_GEMM_STRIDED_DOT2, "gemmStridedNhwcDot2", true},
      {TUNED_GEMM_STRIDED_DOT2_ACCF16, "gemmStridedNhwcDot2AccF16", true},
      {TUNED_GEMM_STRIDED_COOPMAT1, "gemmStridedNhwcCoopmat1", true},
      {TUNED_GEMM_STRIDED_COOPMAT1_ACCF16, "gemmStridedNhwcCoopmat1AccF16", true},
      {TUNED_GEMM_STRIDED_COOPMAT2, "gemmStridedNhwcCoopmat2", true},
      {TUNED_GEMM_STRIDED_COOPMAT2_ACCF16, "gemmStridedNhwcCoopmat2AccF16", true},
      {TUNED_WINOGRAD_TILED, "winogradGemmTiled", false},
      {TUNED_WINOGRAD_DOT2, "winogradGemmDot2", true},
      {TUNED_WINOGRAD_DOT2_ACCF16, "winogradGemmDot2AccF16", true},
      {TUNED_WINOGRAD_COOPMAT1, "winogradGemmCoopmat1", true},
      {TUNED_WINOGRAD_COOPMAT1_ACCF16, "winogradGemmCoopmat1AccF16", true},
      {TUNED_WINOGRAD_COOPMAT2, "winogradGemmCoopmat2", true},
      {TUNED_WINOGRAD_COOPMAT2_ACCF16, "winogradGemmCoopmat2AccF16", true},
      {TUNED_CONV3X3_COOPMAT1, "nhwcConv3x3Coopmat1", true},
      {TUNED_CONV3X3_COOPMAT1_ACCF16, "nhwcConv3x3Coopmat1AccF16", true},
      {TUNED_CONV3X3_COOPMAT2, "nhwcConv3x3Coopmat2", true},
      {TUNED_CONV3X3_COOPMAT2_ACCF16, "nhwcConv3x3Coopmat2AccF16", true},
      {TUNED_CONV3X3_WINOGRAD, "conv3x3Winograd", false},
      {TUNED_CONV5X5_COOPMAT1, "nhwcConv5x5Coopmat1", true},
      {TUNED_CONV5X5_COOPMAT1_ACCF16, "nhwcConv5x5Coopmat1AccF16", true},
      {TUNED_CONV5X5_COOPMAT2, "nhwcConv5x5Coopmat2", true},
      {TUNED_CONV5X5_COOPMAT2_ACCF16, "nhwcConv5x5Coopmat2AccF16", true},
      {TUNED_CONV5X5_WINOGRAD, "conv5x5Winograd", false},
      {TUNED_ATTN_TILED, "attnTiledNhwc", false},
      {TUNED_ATTN_COOPMAT1, "attnNhwcCoopmat1", true},
      {TUNED_ATTN_COOPMAT1_SPLITK, "attnNhwcCoopmat1SplitK", true},
      {TUNED_ATTN_MAINTENANCE1, "attnNhwcCoopmatMaintenance1", true},
      {TUNED_ATTN_MAINTENANCE1_SPLITK, "attnNhwcCoopmatMaintenance1SplitK", true},
      {TUNED_ATTN_COOPMAT2, "attnNhwcCoopmat2", true},
      {TUNED_ATTN_DOT2, "attnNhwcDot2", true},
      {TUNED_SWIGLU, "swiGLU", false},
      {TUNED_WINOGRAD_TRANSFORM, "nhwcWinogradTransform", false},
      {TUNED_WINOGRAD_UNTRANSFORM, "nhwcWinogradUntransform", false},
    };
    return infos;
  }

}  // namespace VulkanTuner

bool VulkanTuner::nhwcWinogradTransformsTunedForCurrentLayout(
  const VulkanTuneParams& config,
  const VulkanDeviceInfo& deviceInfo,
  bool fp16Storage,
  bool fp16Compute) {
  return config.nhwcWinogradTransformTunedLayout ==
         winogradTransformLayoutSignature(config, deviceInfo, fp16Storage, fp16Compute);
}

// Record "attempted but no valid candidate" for the accelerator gemm, conv and
// attention tiers: a tuned bit whose measured time score is missing. These bits
// are recomputed from the TimeUs fields after tuning (and derived at load time
// in reconcile), so invalidKernelMask stays consistent with the measured scores.
int64_t VulkanTuner::computeInvalidVariantBitsFromTimeUs(const VulkanTuneParams& cfg) {
  const int64_t m = cfg.tunedKernelMask;
  // One (tuned bit, measured-time field) pair per accelerator variant. The 8
  // native-conv variants are included now that conv selection goes through the
  // shared measured-time tier policy: a conv variant that was tuned but never
  // produced a usable measurement is exactly the tuned-but-invalid state this
  // mask records, and leaving it out would let an unmeasured conv variant look
  // selectable. The split-K attention variants share the non-split tier's TimeUs
  // field, so ATTN_COOPMAT1_TIMEUS backs both TUNED_ATTN_COOPMAT1 and
  // TUNED_ATTN_COOPMAT1_SPLITK (and likewise for maintenance1).
  static const struct {
    int64_t bit;
    const int32_t VulkanTuneParams::* timeUsField;
  } invalidVariants[] = {
    {TUNED_GEMM_STRIDED_COOPMAT1, &VulkanTuneParams::nhwcGemmCoopmat1F32TimeUs},
    {TUNED_GEMM_STRIDED_COOPMAT1_ACCF16, &VulkanTuneParams::nhwcGemmCoopmat1F16TimeUs},
    {TUNED_GEMM_STRIDED_COOPMAT2, &VulkanTuneParams::nhwcGemmCoopmat2F32TimeUs},
    {TUNED_GEMM_STRIDED_COOPMAT2_ACCF16, &VulkanTuneParams::nhwcGemmCoopmat2F16TimeUs},
    {TUNED_GEMM_STRIDED_DOT2, &VulkanTuneParams::nhwcGemmDot2F32TimeUs},
    {TUNED_GEMM_STRIDED_DOT2_ACCF16, &VulkanTuneParams::nhwcGemmDot2F16TimeUs},
    {TUNED_WINOGRAD_COOPMAT1, &VulkanTuneParams::winogradGemmCoopmat1F32TimeUs},
    {TUNED_WINOGRAD_COOPMAT1_ACCF16, &VulkanTuneParams::winogradGemmCoopmat1F16TimeUs},
    {TUNED_WINOGRAD_COOPMAT2, &VulkanTuneParams::winogradGemmCoopmat2F32TimeUs},
    {TUNED_WINOGRAD_COOPMAT2_ACCF16, &VulkanTuneParams::winogradGemmCoopmat2F16TimeUs},
    {TUNED_WINOGRAD_DOT2, &VulkanTuneParams::winogradGemmDot2F32TimeUs},
    {TUNED_WINOGRAD_DOT2_ACCF16, &VulkanTuneParams::winogradGemmDot2F16TimeUs},
    {TUNED_CONV3X3_COOPMAT1, &VulkanTuneParams::conv3x3NhwcCoopmat1TimeUs},
    {TUNED_CONV3X3_COOPMAT1_ACCF16, &VulkanTuneParams::conv3x3NhwcCoopmat1AccF16TimeUs},
    {TUNED_CONV3X3_COOPMAT2, &VulkanTuneParams::conv3x3NhwcCoopmat2TimeUs},
    {TUNED_CONV3X3_COOPMAT2_ACCF16, &VulkanTuneParams::conv3x3NhwcCoopmat2AccF16TimeUs},
    {TUNED_CONV5X5_COOPMAT1, &VulkanTuneParams::conv5x5NhwcCoopmat1TimeUs},
    {TUNED_CONV5X5_COOPMAT1_ACCF16, &VulkanTuneParams::conv5x5NhwcCoopmat1AccF16TimeUs},
    {TUNED_CONV5X5_COOPMAT2, &VulkanTuneParams::conv5x5NhwcCoopmat2TimeUs},
    {TUNED_CONV5X5_COOPMAT2_ACCF16, &VulkanTuneParams::conv5x5NhwcCoopmat2AccF16TimeUs},
    {TUNED_ATTN_COOPMAT1, &VulkanTuneParams::attnNhwcCoopmat1TimeUs},
    {TUNED_ATTN_COOPMAT1_SPLITK, &VulkanTuneParams::attnNhwcCoopmat1TimeUs},
    {TUNED_ATTN_MAINTENANCE1, &VulkanTuneParams::attnNhwcCoopmatMaintenance1TimeUs},
    {TUNED_ATTN_MAINTENANCE1_SPLITK, &VulkanTuneParams::attnNhwcCoopmatMaintenance1TimeUs},
    {TUNED_ATTN_COOPMAT2, &VulkanTuneParams::attnNhwcCoopmat2TimeUs},
    {TUNED_ATTN_DOT2, &VulkanTuneParams::attnNhwcDot2TimeUs},
  };
  int64_t inv = 0;
  for(const auto& [bit, timeUsField]: invalidVariants)
    if((m & bit) && cfg.*(timeUsField) <= 0)
      inv |= bit;
  return inv;
}

bool VulkanTuner::reconcileTuneParamsForDevice(
  VulkanTuneParams& cfg,
  const VulkanDeviceInfo& info,
  Logger* logger,
  bool fp16Storage,
  bool fp16Compute) {
  static const VulkanTuneParams kDefaults;
  const bool fp16 = fp16Storage && fp16Compute;
  const int64_t dm = info.disabledAccelVariantMask;
  bool disabledAny = false;

  auto disable = [&](std::string_view which, const string& why, std::initializer_list<int64_t> bits) {
    if(logger != nullptr)
      logger->write(
        string("Vulkan tune file enables ") + string(which) + " but " + why + "; resetting to defaults and re-tuning.");
    cfg.clearKernelTuned(bits);
    disabledAny = true;
  };
  // Copy a set of fields back to their default values (from a default-constructed
  // VulkanTuneParams, which is the source of truth for the compiled-in defaults).
  auto resetFields = [&](std::initializer_list<int32_t VulkanTuneParams::*> fields) {
    for(auto f: fields)
      cfg.*f = kDefaults.*f;
  };
  // Gate a coopmat2 tile on its family's support flag, then delegate the
  // device-validity checks to the shared coopmat2TileSupported.
  auto coopmat2TileSupported = [&](
                                 bool supports,
                                 const vector<Coopmat2FlexShape>& shapes,
                                 int32_t workgroupSize,
                                 int32_t bm,
                                 int32_t bn,
                                 int32_t bk) {
    return supports && VulkanTuner::coopmat2TileSupported(info, shapes, Coopmat2Tile{workgroupSize, bm, bn, bk});
  };
  // Empty string means the tier is usable on this device; otherwise a reason to
  // invalidate it. accF16 selects the FP16-accumulation coopmat1 family.
  auto coopmat1Reason = [&](int32_t subgroupSize, int32_t tm, int32_t tn, int32_t tk, bool accF16) -> string {
    if(accF16 ? !info.supportsCoopmat1F16AccF16 : !info.supportsCoopmat1F16)
      return string("this GPU does not support coopmat1") + (accF16 ? " with FP16 accumulation" : "");
    if(subgroupSize != (int32_t)info.subgroupSize)
      return "its subgroup size does not match runtime subgroup size";
    if(!coopmatShapeIsSupported(accF16 ? info.coopmatAccF16Shapes : info.coopmatShapes, tm, tn, tk))
      return "its coopmat fragment shape is not one this GPU reports";
    return "";
  };
  // Dot2 kernels partition the workgroup by a logical SUBGROUP_SIZE spec constant
  // and do not use subgroup intrinsics, so only the device's dot2 support gates
  // them here. The subgroup-size check lives in dot2SubgroupReason below, because
  // the attention-dot2 kernel (which also calls dot2Reason) has no subgroup.
  auto dot2Reason = [&](bool accF16) -> string {
    if(accF16 ? !info.supportsDot2F16AccF16 : !info.supportsDot2F16)
      return string("this GPU does not support dot2") + (accF16 ? " with FP16 accumulation" : "");
    return "";
  };
  // When the device can pin a compute subgroup size (VK_EXT_subgroup_size_control),
  // the dot2 GEMM pipelines are built with requiredSubgroupSize == info.subgroupSize
  // and the SUBGROUP_SIZE spec constant set to the same value, so a tuned config
  // whose subgroup size disagrees would not match the pinned runtime subgroup.
  // On a device that cannot pin, the pipelines are built un-pinned with the
  // constant-32 fallback, so any stored subgroup size remains valid.
  auto dot2SubgroupReason = [&](int32_t subgroupSize) -> string {
    if(info.canRequireComputeSubgroupSize(info.subgroupSize) && subgroupSize != (int32_t)info.subgroupSize)
      return "its subgroup size does not match the pinned runtime subgroup size";
    return "";
  };

  // -------------------------------------------------------------------------
  // Always-present structural fields: used at runtime even when the matching
  // kernel was never tuned, so validate them regardless of the tuned bit.
  // -------------------------------------------------------------------------
  auto checkAlwaysPresent = [&](
                              std::string_view which,
                              const string& why,
                              std::initializer_list<int64_t> bits,
                              std::initializer_list<int32_t VulkanTuneParams::*> fields) {
    if(why.empty())
      return;
    disable(which, why, bits);
    resetFields(fields);
  };
  auto layoutValid = VulkanKernels::LayoutTransform::isConfigSupported(
    cfg.nchwToNhwcSmallTile, cfg.nchwToNhwcLargeTile, cfg.nchwToNhwcTileCrossover);
  if(!layoutValid)
    checkAlwaysPresent(
      "nchwToNhwc",
      "its tile config is not supported",
      {TUNED_NCHW_TO_NHWC},
      {
#define VULKAN_TUNE_PARAM_KERNEL_NCHW_TO_NHWC
#include "vulkantuneparams_fields.inc"
      });
  if(!VulkanKernels::WinogradGemm::isConfigSupported(
       cfg.winogradGemmM, cfg.winogradGemmN, cfg.winogradGemmK, cfg.winogradGemmRN))
    checkAlwaysPresent(
      "winogradGemmTiled",
      "its tile config is not supported",
      {TUNED_WINOGRAD_TILED},
      {
#define VULKAN_TUNE_PARAM_KERNEL_WINOGRAD_GEMM_TILED
#include "vulkantuneparams_fields.inc"
      });
  if(!VulkanKernels::GemmStridedTiledNhwc::isConfigSupported(
       cfg.gemmStridedTiledNhwcLocalSizeX,
       cfg.gemmStridedTiledNhwcLocalSizeY,
       cfg.gemmStridedTiledNhwcTileK,
       cfg.gemmStridedTiledNhwcRN))
    checkAlwaysPresent(
      "gemmStridedTiledNhwc",
      "its workgroup config is not supported",
      {TUNED_GEMM_STRIDED_TILED},
      {
#define VULKAN_TUNE_PARAM_KERNEL_GEMM_STRIDED_TILED
#include "vulkantuneparams_fields.inc"
      });
  if(!VulkanKernels::GemmDirectFP32::isConfigSupported(cfg.gemmDirectLocalSizeX, cfg.gemmDirectLocalSizeY))
    checkAlwaysPresent(
      "gemmDirect",
      "its workgroup config is not supported",
      {TUNED_GEMM_DIRECT},
      {
#define VULKAN_TUNE_PARAM_KERNEL_GEMM_DIRECT
#include "vulkantuneparams_fields.inc"
      });
  if(!VulkanKernels::WinogradTransformNhwc::isConfigSupported(
       cfg.nhwcWinogradTransformLocalSizeX, cfg.nhwcWinogradTransformLocalSizeY))
    resetFields({
#define VULKAN_TUNE_PARAM_KERNEL_WINOGRAD_TRANSFORM_NHWC
#include "vulkantuneparams_fields.inc"
    });
  if(!VulkanKernels::WinogradUntransformNhwc::isConfigSupported(
       cfg.nhwcWinogradUntransformLocalSizeX, cfg.nhwcWinogradUntransformLocalSizeY))
    resetFields({
#define VULKAN_TUNE_PARAM_KERNEL_WINOGRAD_UNTRANSFORM_NHWC
#include "vulkantuneparams_fields.inc"
    });
  if(cfg.nhwcWinograd3x3OutTile != 2 && cfg.nhwcWinograd3x3OutTile != 4)
    resetFields({
#define VULKAN_TUNE_PARAM_KERNEL_WINOGRAD_3X3_OUT_TILE
#include "vulkantuneparams_fields.inc"
    });
  if(!isPow2(cfg.gpoolNhwcXystride))
    checkAlwaysPresent(
      "gpoolNhwc",
      "its stride is not a power of two",
      {TUNED_GPOOL_NHWC},
      {
#define VULKAN_TUNE_PARAM_KERNEL_GPOOL_NHWC
#include "vulkantuneparams_fields.inc"
      });
  if(!isPow2(cfg.valueHeadPoolNhwcXystride))
    checkAlwaysPresent(
      "valueHeadPoolNhwc",
      "its stride is not a power of two",
      {TUNED_VALUE_HEAD_POOL_NHWC},
      {
#define VULKAN_TUNE_PARAM_KERNEL_VALUE_HEAD_POOL_NHWC
#include "vulkantuneparams_fields.inc"
      });
  if(!VulkanKernels::SpatialRMSNormNhwc::isConfigSupported(cfg.spatialRMSNormNhwcTile))
    checkAlwaysPresent(
      "spatialRMSNormNhwc",
      "its tile is not supported",
      {TUNED_SPATIAL_RMSNORM_NHWC},
      {
#define VULKAN_TUNE_PARAM_KERNEL_SPATIAL_RMSNORM_NHWC
#include "vulkantuneparams_fields.inc"
      });
  if(!VulkanKernels::SwiGLU::isConfigSupported(cfg.swiGLULocalSizeX))
    checkAlwaysPresent(
      "swiGLU",
      "its workgroup is not supported",
      {TUNED_SWIGLU},
      {
#define VULKAN_TUNE_PARAM_KERNEL_SWIGLU
#include "vulkantuneparams_fields.inc"
      });
  if(!VulkanKernels::AttentionTiled::isConfigSupported(cfg.attnNhwcBlockQ, cfg.attnNhwcBlockKV, cfg.attnNhwcQPerThread))
    checkAlwaysPresent(
      "attnTiledNhwc",
      "its block config is not supported",
      {TUNED_ATTN_TILED},
      {
#define VULKAN_TUNE_PARAM_KERNEL_ATTN_TILED
#include "vulkantuneparams_fields.inc"
      });

  // -------------------------------------------------------------------------
  // Accelerated tiers. Without fp16 none of the accelerator tiers are
  // selectable at runtime, so clear every one of them. The tiled/winograd
  // fallback tiers are valid without fp16 and must be preserved.
  // -------------------------------------------------------------------------
  const int64_t acceleratedBits = (winogradGemmFlavorBits() & ~TUNED_WINOGRAD_TILED) |
                                  (stridedGemmFlavorBits() & ~TUNED_GEMM_STRIDED_TILED) |
                                  (conv3x3FlavorBits() & ~TUNED_CONV3X3_WINOGRAD) |
                                  (conv5x5FlavorBits() & ~TUNED_CONV5X5_WINOGRAD) | attentionVariantBits();
  if(!fp16 && cfg.anyOfKernelTuned({acceleratedBits})) {
    if(logger != nullptr)
      logger->write(
        "Vulkan tune file enables accelerated fp16 tiers but this run is not fp16; resetting them to defaults and "
        "re-tuning.");
    cfg.clearKernelTuned({acceleratedBits});
    resetFields({
#define VULKAN_TUNE_PARAM_ALL_ACCELERATED
#include "vulkantuneparams_fields.inc"
    });
    disabledAny = true;
    cfg.nhwcWinogradTransformTunedLayout.clear();
  }

  // Hide the filter-disabled variants for the per-tier validation and the
  // Winograd variant resolution below, so both match what the runtime will
  // actually select. Their tuned/invalid bits are restored untouched at the end.
  const int64_t disabledTunedBits = cfg.tunedKernelMask & dm;
  const int64_t disabledInvalidBits = cfg.invalidKernelMask & dm;
  cfg.tunedKernelMask &= ~dm;
  cfg.invalidKernelMask &= ~dm;

  // -------------------------------------------------------------------------
  // Tuned accelerated tiers (only when fp16). Each is validated for structural
  // support and against the device's reported capabilities.
  if(fp16) {
    // Table-driven per-kernel validation for the uniform gated tiers
    // (coopmat1/2 and dot2 GEMMs). The attention tiers have coupled split-K /
    // maintenance1 semantics and stay hand-written.
    auto reconcileReasonForKernel = [&](int64_t bit) -> string {
      const string tileReason = "its tile config is not supported";
      switch(bit) {
        case TUNED_WINOGRAD_COOPMAT1: {
          string why = coopmat1Reason(cfg.coopmat1SubgroupSize, cfg.coopmat1TM, cfg.coopmat1TN, cfg.coopmat1TK, false);
          if(
            why.empty() && !VulkanKernels::WinogradGemmCoopmat1::isConfigSupported(
                             cfg.coopmat1WorkgroupSize,
                             cfg.coopmat1BM,
                             cfg.coopmat1BN,
                             cfg.coopmat1BK,
                             cfg.coopmat1SGM,
                             cfg.coopmat1SGN,
                             cfg.coopmat1TM,
                             cfg.coopmat1TN,
                             cfg.coopmat1TK,
                             cfg.coopmat1SubgroupSize))
            why = tileReason;
          return why;
        }
        case TUNED_CONV3X3_COOPMAT1: {
          string why = coopmat1Reason(
            cfg.conv3x3NhwcCoopmat1SubgroupSize,
            cfg.conv3x3NhwcCoopmat1TM,
            cfg.conv3x3NhwcCoopmat1TN,
            cfg.conv3x3NhwcCoopmat1TK,
            false);
          if(
            why.empty() && !VulkanKernels::Conv3x3ImplicitGemmCoopmat1AccF32NhwcVec8::isConfigSupported(
                             cfg.conv3x3NhwcCoopmat1WorkgroupSize,
                             cfg.conv3x3NhwcCoopmat1BM,
                             cfg.conv3x3NhwcCoopmat1BN,
                             cfg.conv3x3NhwcCoopmat1BK,
                             cfg.conv3x3NhwcCoopmat1SGM,
                             cfg.conv3x3NhwcCoopmat1SGN,
                             cfg.conv3x3NhwcCoopmat1TM,
                             cfg.conv3x3NhwcCoopmat1TN,
                             cfg.conv3x3NhwcCoopmat1TK,
                             cfg.conv3x3NhwcCoopmat1SubgroupSize))
            why = tileReason;
          return why;
        }
        case TUNED_CONV5X5_COOPMAT1: {
          string why = coopmat1Reason(
            cfg.conv5x5NhwcCoopmat1SubgroupSize,
            cfg.conv5x5NhwcCoopmat1TM,
            cfg.conv5x5NhwcCoopmat1TN,
            cfg.conv5x5NhwcCoopmat1TK,
            false);
          if(
            why.empty() && !VulkanKernels::Conv3x3ImplicitGemmCoopmat1AccF32NhwcVec8::isConfigSupported(
                             cfg.conv5x5NhwcCoopmat1WorkgroupSize,
                             cfg.conv5x5NhwcCoopmat1BM,
                             cfg.conv5x5NhwcCoopmat1BN,
                             cfg.conv5x5NhwcCoopmat1BK,
                             cfg.conv5x5NhwcCoopmat1SGM,
                             cfg.conv5x5NhwcCoopmat1SGN,
                             cfg.conv5x5NhwcCoopmat1TM,
                             cfg.conv5x5NhwcCoopmat1TN,
                             cfg.conv5x5NhwcCoopmat1TK,
                             cfg.conv5x5NhwcCoopmat1SubgroupSize))
            why = tileReason;
          return why;
        }
        case TUNED_GEMM_STRIDED_COOPMAT1: {
          string why = coopmat1Reason(
            cfg.nhwcStridedCoopmat1SubgroupSize,
            cfg.nhwcStridedCoopmat1TM,
            cfg.nhwcStridedCoopmat1TN,
            cfg.nhwcStridedCoopmat1TK,
            false);
          if(
            why.empty() && !VulkanKernels::GemmStridedCoopmat1Nhwc::isConfigSupported(
                             cfg.nhwcStridedCoopmat1WorkgroupSize,
                             cfg.nhwcStridedCoopmat1BM,
                             cfg.nhwcStridedCoopmat1BN,
                             cfg.nhwcStridedCoopmat1BK,
                             cfg.nhwcStridedCoopmat1SGM,
                             cfg.nhwcStridedCoopmat1SGN,
                             cfg.nhwcStridedCoopmat1TM,
                             cfg.nhwcStridedCoopmat1TN,
                             cfg.nhwcStridedCoopmat1TK,
                             cfg.nhwcStridedCoopmat1SubgroupSize))
            why = tileReason;
          return why;
        }
        case TUNED_WINOGRAD_COOPMAT1_ACCF16: {
          string why = coopmat1Reason(
            cfg.coopmat1AccF16SubgroupSize, cfg.coopmat1AccF16TM, cfg.coopmat1AccF16TN, cfg.coopmat1AccF16TK, true);
          if(
            why.empty() && !VulkanKernels::WinogradGemmCoopmat1AccF16::isConfigSupported(
                             cfg.coopmat1AccF16WorkgroupSize,
                             cfg.coopmat1AccF16BM,
                             cfg.coopmat1AccF16BN,
                             cfg.coopmat1AccF16BK,
                             cfg.coopmat1AccF16SGM,
                             cfg.coopmat1AccF16SGN,
                             cfg.coopmat1AccF16TM,
                             cfg.coopmat1AccF16TN,
                             cfg.coopmat1AccF16TK,
                             cfg.coopmat1AccF16SubgroupSize))
            why = tileReason;
          return why;
        }
        case TUNED_CONV3X3_COOPMAT1_ACCF16: {
          string why = coopmat1Reason(
            cfg.conv3x3NhwcCoopmat1AccF16SubgroupSize,
            cfg.conv3x3NhwcCoopmat1AccF16TM,
            cfg.conv3x3NhwcCoopmat1AccF16TN,
            cfg.conv3x3NhwcCoopmat1AccF16TK,
            true);
          if(
            why.empty() && !VulkanKernels::Conv3x3ImplicitGemmCoopmat1AccF16NhwcVec8::isConfigSupported(
                             cfg.conv3x3NhwcCoopmat1AccF16WorkgroupSize,
                             cfg.conv3x3NhwcCoopmat1AccF16BM,
                             cfg.conv3x3NhwcCoopmat1AccF16BN,
                             cfg.conv3x3NhwcCoopmat1AccF16BK,
                             cfg.conv3x3NhwcCoopmat1AccF16SGM,
                             cfg.conv3x3NhwcCoopmat1AccF16SGN,
                             cfg.conv3x3NhwcCoopmat1AccF16TM,
                             cfg.conv3x3NhwcCoopmat1AccF16TN,
                             cfg.conv3x3NhwcCoopmat1AccF16TK,
                             cfg.conv3x3NhwcCoopmat1AccF16SubgroupSize))
            why = tileReason;
          return why;
        }
        case TUNED_CONV5X5_COOPMAT1_ACCF16: {
          string why = coopmat1Reason(
            cfg.conv5x5NhwcCoopmat1AccF16SubgroupSize,
            cfg.conv5x5NhwcCoopmat1AccF16TM,
            cfg.conv5x5NhwcCoopmat1AccF16TN,
            cfg.conv5x5NhwcCoopmat1AccF16TK,
            true);
          if(
            why.empty() && !VulkanKernels::Conv3x3ImplicitGemmCoopmat1AccF16NhwcVec8::isConfigSupported(
                             cfg.conv5x5NhwcCoopmat1AccF16WorkgroupSize,
                             cfg.conv5x5NhwcCoopmat1AccF16BM,
                             cfg.conv5x5NhwcCoopmat1AccF16BN,
                             cfg.conv5x5NhwcCoopmat1AccF16BK,
                             cfg.conv5x5NhwcCoopmat1AccF16SGM,
                             cfg.conv5x5NhwcCoopmat1AccF16SGN,
                             cfg.conv5x5NhwcCoopmat1AccF16TM,
                             cfg.conv5x5NhwcCoopmat1AccF16TN,
                             cfg.conv5x5NhwcCoopmat1AccF16TK,
                             cfg.conv5x5NhwcCoopmat1AccF16SubgroupSize))
            why = tileReason;
          return why;
        }
        case TUNED_GEMM_STRIDED_COOPMAT1_ACCF16: {
          string why = coopmat1Reason(
            cfg.nhwcStridedCoopmat1AccF16SubgroupSize,
            cfg.nhwcStridedCoopmat1AccF16TM,
            cfg.nhwcStridedCoopmat1AccF16TN,
            cfg.nhwcStridedCoopmat1AccF16TK,
            true);
          if(
            why.empty() && !VulkanKernels::GemmStridedCoopmat1AccF16Nhwc::isConfigSupported(
                             cfg.nhwcStridedCoopmat1AccF16WorkgroupSize,
                             cfg.nhwcStridedCoopmat1AccF16BM,
                             cfg.nhwcStridedCoopmat1AccF16BN,
                             cfg.nhwcStridedCoopmat1AccF16BK,
                             cfg.nhwcStridedCoopmat1AccF16SGM,
                             cfg.nhwcStridedCoopmat1AccF16SGN,
                             cfg.nhwcStridedCoopmat1AccF16TM,
                             cfg.nhwcStridedCoopmat1AccF16TN,
                             cfg.nhwcStridedCoopmat1AccF16TK,
                             cfg.nhwcStridedCoopmat1AccF16SubgroupSize))
            why = tileReason;
          return why;
        }
        case TUNED_WINOGRAD_COOPMAT2: {
          string why;
          if(!coopmat2TileSupported(
               info.supportsCoopmat2F16,
               info.coopmat2FlexShapes,
               cfg.coopmat2WorkgroupSize,
               cfg.coopmat2BM,
               cfg.coopmat2BN,
               cfg.coopmat2BK))
            why = "this GPU does not support this coopmat2 tile";
          if(
            why.empty() && !VulkanKernels::WinogradGemmCoopmat2::isConfigSupported(
                             cfg.coopmat2WorkgroupSize, cfg.coopmat2BM, cfg.coopmat2BN, cfg.coopmat2BK))
            why = tileReason;
          return why;
        }
        case TUNED_CONV3X3_COOPMAT2: {
          string why;
          if(!coopmat2TileSupported(
               info.supportsCoopmat2F16,
               info.coopmat2FlexShapes,
               cfg.conv3x3NhwcCoopmat2WorkgroupSize,
               cfg.conv3x3NhwcCoopmat2BM,
               cfg.conv3x3NhwcCoopmat2BN,
               cfg.conv3x3NhwcCoopmat2BK))
            why = "this GPU does not support this coopmat2 tile";
          else if(
            (size_t)info.coopmat2ReservedSharedBytes +
              VulkanKernels::Conv3x3ImplicitGemmCoopmat2AccF32NhwcVec8::sharedBytes(
                cfg.conv3x3NhwcCoopmat2BM, cfg.conv3x3NhwcCoopmat2BK) >
            info.properties.limits.maxComputeSharedMemorySize)
            why = "its shared-A tile plus reserved coopmat memory exceeds this GPU's limit";
          if(
            why.empty() && !VulkanKernels::Conv3x3ImplicitGemmCoopmat2AccF32NhwcVec8::isConfigSupported(
                             cfg.conv3x3NhwcCoopmat2WorkgroupSize,
                             cfg.conv3x3NhwcCoopmat2BM,
                             cfg.conv3x3NhwcCoopmat2BN,
                             cfg.conv3x3NhwcCoopmat2BK))
            why = tileReason;
          return why;
        }
        case TUNED_CONV5X5_COOPMAT2: {
          string why;
          if(!coopmat2TileSupported(
               info.supportsCoopmat2F16,
               info.coopmat2FlexShapes,
               cfg.conv5x5NhwcCoopmat2WorkgroupSize,
               cfg.conv5x5NhwcCoopmat2BM,
               cfg.conv5x5NhwcCoopmat2BN,
               cfg.conv5x5NhwcCoopmat2BK))
            why = "this GPU does not support this coopmat2 tile";
          else if(
            (size_t)info.coopmat2ReservedSharedBytes +
              VulkanKernels::Conv3x3ImplicitGemmCoopmat2AccF32NhwcVec8::sharedBytes(
                cfg.conv5x5NhwcCoopmat2BM, cfg.conv5x5NhwcCoopmat2BK) >
            info.properties.limits.maxComputeSharedMemorySize)
            why = "its shared-A tile plus reserved coopmat memory exceeds this GPU's limit";
          if(
            why.empty() && !VulkanKernels::Conv3x3ImplicitGemmCoopmat2AccF32NhwcVec8::isConfigSupported(
                             cfg.conv5x5NhwcCoopmat2WorkgroupSize,
                             cfg.conv5x5NhwcCoopmat2BM,
                             cfg.conv5x5NhwcCoopmat2BN,
                             cfg.conv5x5NhwcCoopmat2BK))
            why = tileReason;
          return why;
        }
        case TUNED_GEMM_STRIDED_COOPMAT2: {
          string why;
          if(!coopmat2TileSupported(
               info.supportsCoopmat2F16,
               info.coopmat2FlexShapes,
               cfg.nhwcStridedCoopmat2WorkgroupSize,
               cfg.nhwcStridedCoopmat2BM,
               cfg.nhwcStridedCoopmat2BN,
               cfg.nhwcStridedCoopmat2BK))
            why = "this GPU does not support this coopmat2 tile";
          if(
            why.empty() && !VulkanKernels::GemmStridedCoopmat2Nhwc::isConfigSupported(
                             cfg.nhwcStridedCoopmat2WorkgroupSize,
                             cfg.nhwcStridedCoopmat2BM,
                             cfg.nhwcStridedCoopmat2BN,
                             cfg.nhwcStridedCoopmat2BK))
            why = tileReason;
          return why;
        }
        case TUNED_WINOGRAD_COOPMAT2_ACCF16: {
          string why;
          if(!coopmat2TileSupported(
               info.supportsCoopmat2F16AccF16,
               info.coopmat2AccF16FlexShapes,
               cfg.coopmat2AccF16WorkgroupSize,
               cfg.coopmat2AccF16BM,
               cfg.coopmat2AccF16BN,
               cfg.coopmat2AccF16BK))
            why = "this GPU does not support this coopmat2 tile";
          if(
            why.empty() &&
            !VulkanKernels::WinogradGemmCoopmat2AccF16::isConfigSupported(
              cfg.coopmat2AccF16WorkgroupSize, cfg.coopmat2AccF16BM, cfg.coopmat2AccF16BN, cfg.coopmat2AccF16BK))
            why = tileReason;
          return why;
        }
        case TUNED_CONV3X3_COOPMAT2_ACCF16: {
          string why;
          if(!coopmat2TileSupported(
               info.supportsCoopmat2F16AccF16,
               info.coopmat2AccF16FlexShapes,
               cfg.conv3x3NhwcCoopmat2AccF16WorkgroupSize,
               cfg.conv3x3NhwcCoopmat2AccF16BM,
               cfg.conv3x3NhwcCoopmat2AccF16BN,
               cfg.conv3x3NhwcCoopmat2AccF16BK))
            why = "this GPU does not support this coopmat2 tile";
          else if(
            (size_t)info.coopmat2ReservedSharedBytes +
              VulkanKernels::Conv3x3ImplicitGemmCoopmat2AccF16NhwcVec8::sharedBytes(
                cfg.conv3x3NhwcCoopmat2AccF16BM, cfg.conv3x3NhwcCoopmat2AccF16BK) >
            info.properties.limits.maxComputeSharedMemorySize)
            why = "its shared-A tile plus reserved coopmat memory exceeds this GPU's limit";
          if(
            why.empty() && !VulkanKernels::Conv3x3ImplicitGemmCoopmat2AccF16NhwcVec8::isConfigSupported(
                             cfg.conv3x3NhwcCoopmat2AccF16WorkgroupSize,
                             cfg.conv3x3NhwcCoopmat2AccF16BM,
                             cfg.conv3x3NhwcCoopmat2AccF16BN,
                             cfg.conv3x3NhwcCoopmat2AccF16BK))
            why = tileReason;
          return why;
        }
        case TUNED_CONV5X5_COOPMAT2_ACCF16: {
          string why;
          if(!coopmat2TileSupported(
               info.supportsCoopmat2F16AccF16,
               info.coopmat2AccF16FlexShapes,
               cfg.conv5x5NhwcCoopmat2AccF16WorkgroupSize,
               cfg.conv5x5NhwcCoopmat2AccF16BM,
               cfg.conv5x5NhwcCoopmat2AccF16BN,
               cfg.conv5x5NhwcCoopmat2AccF16BK))
            why = "this GPU does not support this coopmat2 tile";
          else if(
            (size_t)info.coopmat2ReservedSharedBytes +
              VulkanKernels::Conv3x3ImplicitGemmCoopmat2AccF16NhwcVec8::sharedBytes(
                cfg.conv5x5NhwcCoopmat2AccF16BM, cfg.conv5x5NhwcCoopmat2AccF16BK) >
            info.properties.limits.maxComputeSharedMemorySize)
            why = "its shared-A tile plus reserved coopmat memory exceeds this GPU's limit";
          if(
            why.empty() && !VulkanKernels::Conv3x3ImplicitGemmCoopmat2AccF16NhwcVec8::isConfigSupported(
                             cfg.conv5x5NhwcCoopmat2AccF16WorkgroupSize,
                             cfg.conv5x5NhwcCoopmat2AccF16BM,
                             cfg.conv5x5NhwcCoopmat2AccF16BN,
                             cfg.conv5x5NhwcCoopmat2AccF16BK))
            why = tileReason;
          return why;
        }
        case TUNED_GEMM_STRIDED_COOPMAT2_ACCF16: {
          string why;
          if(!coopmat2TileSupported(
               info.supportsCoopmat2F16AccF16,
               info.coopmat2AccF16FlexShapes,
               cfg.nhwcStridedCoopmat2AccF16WorkgroupSize,
               cfg.nhwcStridedCoopmat2AccF16BM,
               cfg.nhwcStridedCoopmat2AccF16BN,
               cfg.nhwcStridedCoopmat2AccF16BK))
            why = "this GPU does not support this coopmat2 tile";
          if(
            why.empty() && !VulkanKernels::GemmStridedCoopmat2AccF16Nhwc::isConfigSupported(
                             cfg.nhwcStridedCoopmat2AccF16WorkgroupSize,
                             cfg.nhwcStridedCoopmat2AccF16BM,
                             cfg.nhwcStridedCoopmat2AccF16BN,
                             cfg.nhwcStridedCoopmat2AccF16BK))
            why = tileReason;
          return why;
        }
        case TUNED_WINOGRAD_DOT2: {
          string why = dot2Reason(false);
          if(why.empty())
            why = dot2SubgroupReason(cfg.dot2SubgroupSize);
          if(
            why.empty() && !VulkanKernels::WinogradGemmDot2::isConfigSupported(
                             cfg.dot2WorkgroupSize,
                             cfg.dot2BM,
                             cfg.dot2BN,
                             cfg.dot2SGM,
                             cfg.dot2SGN,
                             cfg.dot2SGMIter,
                             cfg.dot2TM,
                             cfg.dot2TN,
                             cfg.dot2SubgroupSize))
            why = tileReason;
          return why;
        }
        case TUNED_GEMM_STRIDED_DOT2: {
          string why = dot2Reason(false);
          if(why.empty())
            why = dot2SubgroupReason(cfg.nhwcStridedDot2SubgroupSize);
          if(
            why.empty() && !VulkanKernels::GemmStridedDot2Nhwc::isConfigSupported(
                             cfg.nhwcStridedDot2WorkgroupSize,
                             cfg.nhwcStridedDot2BM,
                             cfg.nhwcStridedDot2BN,
                             cfg.nhwcStridedDot2SGM,
                             cfg.nhwcStridedDot2SGN,
                             cfg.nhwcStridedDot2SGMIter,
                             cfg.nhwcStridedDot2TM,
                             cfg.nhwcStridedDot2TN,
                             cfg.nhwcStridedDot2SubgroupSize))
            why = tileReason;
          return why;
        }
        case TUNED_WINOGRAD_DOT2_ACCF16: {
          string why = dot2Reason(true);
          if(why.empty())
            why = dot2SubgroupReason(cfg.dot2AccF16SubgroupSize);
          if(
            why.empty() && !VulkanKernels::WinogradGemmDot2AccF16::isConfigSupported(
                             cfg.dot2AccF16WorkgroupSize,
                             cfg.dot2AccF16BM,
                             cfg.dot2AccF16BN,
                             cfg.dot2AccF16SGM,
                             cfg.dot2AccF16SGN,
                             cfg.dot2AccF16SGMIter,
                             cfg.dot2AccF16TM,
                             cfg.dot2AccF16TN,
                             cfg.dot2AccF16SubgroupSize))
            why = tileReason;
          return why;
        }
        case TUNED_GEMM_STRIDED_DOT2_ACCF16: {
          string why = dot2Reason(true);
          if(why.empty())
            why = dot2SubgroupReason(cfg.nhwcStridedDot2AccF16SubgroupSize);
          if(
            why.empty() && !VulkanKernels::GemmStridedDot2AccF16Nhwc::isConfigSupported(
                             cfg.nhwcStridedDot2AccF16WorkgroupSize,
                             cfg.nhwcStridedDot2AccF16BM,
                             cfg.nhwcStridedDot2AccF16BN,
                             cfg.nhwcStridedDot2AccF16SGM,
                             cfg.nhwcStridedDot2AccF16SGN,
                             cfg.nhwcStridedDot2AccF16SGMIter,
                             cfg.nhwcStridedDot2AccF16TM,
                             cfg.nhwcStridedDot2AccF16TN,
                             cfg.nhwcStridedDot2AccF16SubgroupSize))
            why = tileReason;
          return why;
        }
        default:
          testAssert(false && "reconcileReasonForKernel: unexpected gated kernel");
          return "";
      }
    };
    auto resetFieldsForKernel = [&](int64_t bit) {
      switch(bit) {
        case TUNED_WINOGRAD_COOPMAT1:
          resetFields({
#define VULKAN_TUNE_PARAM_KERNEL_WINOGRAD_GEMM_COOPMAT1_FULL
#include "vulkantuneparams_fields.inc"
          });
          break;
        case TUNED_CONV3X3_COOPMAT1:
          resetFields({
#define VULKAN_TUNE_PARAM_KERNEL_CONV3X3_COOPMAT1_FULL
#include "vulkantuneparams_fields.inc"
          });
          break;
        case TUNED_CONV5X5_COOPMAT1:
          resetFields({
#define VULKAN_TUNE_PARAM_KERNEL_CONV5X5_COOPMAT1_FULL
#include "vulkantuneparams_fields.inc"
          });
          break;
        case TUNED_GEMM_STRIDED_COOPMAT1:
          resetFields({
#define VULKAN_TUNE_PARAM_KERNEL_GEMM_STRIDED_COOPMAT1_FULL
#include "vulkantuneparams_fields.inc"
          });
          break;
        case TUNED_WINOGRAD_COOPMAT1_ACCF16:
          resetFields({
#define VULKAN_TUNE_PARAM_KERNEL_WINOGRAD_GEMM_COOPMAT1_ACCF16_FULL
#include "vulkantuneparams_fields.inc"
          });
          break;
        case TUNED_CONV3X3_COOPMAT1_ACCF16:
          resetFields({
#define VULKAN_TUNE_PARAM_KERNEL_CONV3X3_COOPMAT1_ACCF16_FULL
#include "vulkantuneparams_fields.inc"
          });
          break;
        case TUNED_CONV5X5_COOPMAT1_ACCF16:
          resetFields({
#define VULKAN_TUNE_PARAM_KERNEL_CONV5X5_COOPMAT1_ACCF16_FULL
#include "vulkantuneparams_fields.inc"
          });
          break;
        case TUNED_GEMM_STRIDED_COOPMAT1_ACCF16:
          resetFields({
#define VULKAN_TUNE_PARAM_KERNEL_GEMM_STRIDED_COOPMAT1_ACCF16_FULL
#include "vulkantuneparams_fields.inc"
          });
          break;
        case TUNED_WINOGRAD_COOPMAT2:
          resetFields({
#define VULKAN_TUNE_PARAM_KERNEL_WINOGRAD_GEMM_COOPMAT2_FULL
#include "vulkantuneparams_fields.inc"
          });
          break;
        case TUNED_CONV3X3_COOPMAT2:
          resetFields({
#define VULKAN_TUNE_PARAM_KERNEL_CONV3X3_COOPMAT2_FULL
#include "vulkantuneparams_fields.inc"
          });
          break;
        case TUNED_CONV5X5_COOPMAT2:
          resetFields({
#define VULKAN_TUNE_PARAM_KERNEL_CONV5X5_COOPMAT2_FULL
#include "vulkantuneparams_fields.inc"
          });
          break;
        case TUNED_GEMM_STRIDED_COOPMAT2:
          resetFields({
#define VULKAN_TUNE_PARAM_KERNEL_GEMM_STRIDED_COOPMAT2_FULL
#include "vulkantuneparams_fields.inc"
          });
          break;
        case TUNED_WINOGRAD_COOPMAT2_ACCF16:
          resetFields({
#define VULKAN_TUNE_PARAM_KERNEL_WINOGRAD_GEMM_COOPMAT2_ACCF16_FULL
#include "vulkantuneparams_fields.inc"
          });
          break;
        case TUNED_CONV3X3_COOPMAT2_ACCF16:
          resetFields({
#define VULKAN_TUNE_PARAM_KERNEL_CONV3X3_COOPMAT2_ACCF16_FULL
#include "vulkantuneparams_fields.inc"
          });
          break;
        case TUNED_CONV5X5_COOPMAT2_ACCF16:
          resetFields({
#define VULKAN_TUNE_PARAM_KERNEL_CONV5X5_COOPMAT2_ACCF16_FULL
#include "vulkantuneparams_fields.inc"
          });
          break;
        case TUNED_GEMM_STRIDED_COOPMAT2_ACCF16:
          resetFields({
#define VULKAN_TUNE_PARAM_KERNEL_GEMM_STRIDED_COOPMAT2_ACCF16_FULL
#include "vulkantuneparams_fields.inc"
          });
          break;
        case TUNED_WINOGRAD_DOT2:
          resetFields({
#define VULKAN_TUNE_PARAM_KERNEL_WINOGRAD_GEMM_DOT2_FULL
#include "vulkantuneparams_fields.inc"
          });
          break;
        case TUNED_GEMM_STRIDED_DOT2:
          resetFields({
#define VULKAN_TUNE_PARAM_KERNEL_GEMM_STRIDED_DOT2_FULL
#include "vulkantuneparams_fields.inc"
          });
          break;
        case TUNED_WINOGRAD_DOT2_ACCF16:
          resetFields({
#define VULKAN_TUNE_PARAM_KERNEL_WINOGRAD_GEMM_DOT2_ACCF16_FULL
#include "vulkantuneparams_fields.inc"
          });
          break;
        case TUNED_GEMM_STRIDED_DOT2_ACCF16:
          resetFields({
#define VULKAN_TUNE_PARAM_KERNEL_GEMM_STRIDED_DOT2_ACCF16_FULL
#include "vulkantuneparams_fields.inc"
          });
          break;
        default:
          testAssert(false && "resetFieldsForKernel: unexpected gated kernel");
          break;
      }
    };

    // Uniform gated accelerated tiers, driven by the kernel-family table.
    for(const auto& k: kernelFamilyInfos()) {
      if(!k.gated)
        continue;
      if(
        k.bit == TUNED_ATTN_COOPMAT1 || k.bit == TUNED_ATTN_COOPMAT1_SPLITK || k.bit == TUNED_ATTN_MAINTENANCE1 ||
        k.bit == TUNED_ATTN_MAINTENANCE1_SPLITK || k.bit == TUNED_ATTN_COOPMAT2 || k.bit == TUNED_ATTN_DOT2)
        continue;
      if(!cfg.hasKernelTuned(k.bit))
        continue;
      string why = reconcileReasonForKernel(k.bit);
      if(!why.empty()) {
        disable(k.label, why, {k.bit});
        resetFieldsForKernel(k.bit);
      }
    }

    // ---- attention: coopmat1 (+split-K), maintenance1 (+split-K), coopmat2, dot2 ----
    if(cfg.hasKernelTuned(TUNED_ATTN_COOPMAT1)) {
      string why = coopmat1Reason(
        cfg.attnNhwcCoopmat1SubgroupSize,
        cfg.attnNhwcCoopmat1TM,
        cfg.attnNhwcCoopmat1TN,
        cfg.attnNhwcCoopmat1TK,
        false);
      if(
        why.empty() && !VulkanKernels::AttentionCoopmat1Nhwc::isConfigSupported(
                         cfg.attnNhwcCoopmat1WorkgroupSize,
                         cfg.attnNhwcCoopmat1BlockQ,
                         cfg.attnNhwcCoopmat1BlockKV,
                         cfg.attnNhwcCoopmat1TM,
                         cfg.attnNhwcCoopmat1TN,
                         cfg.attnNhwcCoopmat1TK,
                         cfg.attnNhwcCoopmat1SubgroupSize,
                         64,
                         64))
        why = "its tile config is not supported";
      if(!why.empty()) {
        disable("attnNhwcCoopmat1", why, {TUNED_ATTN_COOPMAT1, TUNED_ATTN_COOPMAT1_SPLITK});
        resetFields({
#define VULKAN_TUNE_PARAM_KERNEL_ATTN_COOPMAT1_FULL
#define VULKAN_TUNE_PARAM_KERNEL_ATTN_COOPMAT1_SPLITK
#include "vulkantuneparams_fields.inc"
        });
      }
    } else if(cfg.hasKernelTuned(TUNED_ATTN_COOPMAT1_SPLITK)) {
      string why = coopmat1Reason(
        cfg.attnNhwcCoopmat1SplitKSubgroupSize,
        cfg.attnNhwcCoopmat1SplitKTM,
        cfg.attnNhwcCoopmat1SplitKTN,
        cfg.attnNhwcCoopmat1SplitKTK,
        false);
      if(
        why.empty() && !VulkanKernels::AttentionCoopmat1Nhwc::isConfigSupported(
                         cfg.attnNhwcCoopmat1SplitKWorkgroupSize,
                         cfg.attnNhwcCoopmat1SplitKBlockQ,
                         cfg.attnNhwcCoopmat1SplitKBlockKV,
                         cfg.attnNhwcCoopmat1SplitKTM,
                         cfg.attnNhwcCoopmat1SplitKTN,
                         cfg.attnNhwcCoopmat1SplitKTK,
                         cfg.attnNhwcCoopmat1SplitKSubgroupSize,
                         64,
                         64))
        why = "its tile config is not supported";
      if(!why.empty()) {
        disable("attnNhwcCoopmat1SplitK", why, {TUNED_ATTN_COOPMAT1_SPLITK});
        resetFields({
#define VULKAN_TUNE_PARAM_KERNEL_ATTN_COOPMAT1_SPLITK
#include "vulkantuneparams_fields.inc"
        });
      }
    }
    if(cfg.hasKernelTuned(TUNED_ATTN_MAINTENANCE1)) {
      string why;
      if(!info.supportsCoopmatMaintenance1)
        why = "this GPU does not support cooperative matrix maintenance1";
      else if(cfg.attnNhwcCoopmatMaintenance1SubgroupSize != (int32_t)info.subgroupSize)
        why = "its subgroup size does not match runtime subgroup size";
      else if(!coopmatShapeIsSupported(
                info.coopmatShapes,
                cfg.attnNhwcCoopmatMaintenance1TM,
                cfg.attnNhwcCoopmatMaintenance1TN,
                cfg.attnNhwcCoopmatMaintenance1TK))
        why = "its coopmat fragment shape is not one this GPU reports";
      if(
        why.empty() && !VulkanKernels::AttentionCoopmat1Nhwc::isConfigSupported(
                         cfg.attnNhwcCoopmatMaintenance1WorkgroupSize,
                         cfg.attnNhwcCoopmatMaintenance1BlockQ,
                         cfg.attnNhwcCoopmatMaintenance1BlockKV,
                         cfg.attnNhwcCoopmatMaintenance1TM,
                         cfg.attnNhwcCoopmatMaintenance1TN,
                         cfg.attnNhwcCoopmatMaintenance1TK,
                         cfg.attnNhwcCoopmatMaintenance1SubgroupSize,
                         64,
                         64))
        why = "its tile config is not supported";
      if(!why.empty()) {
        disable("attnNhwcCoopmatMaintenance1", why, {TUNED_ATTN_MAINTENANCE1, TUNED_ATTN_MAINTENANCE1_SPLITK});
        resetFields({
#define VULKAN_TUNE_PARAM_KERNEL_ATTN_MAINTENANCE1_FULL
#define VULKAN_TUNE_PARAM_KERNEL_ATTN_MAINTENANCE1_SPLITK
#include "vulkantuneparams_fields.inc"
        });
      }
    } else if(cfg.hasKernelTuned(TUNED_ATTN_MAINTENANCE1_SPLITK)) {
      string why;
      if(!info.supportsCoopmatMaintenance1)
        why = "this GPU does not support cooperative matrix maintenance1";
      else if(cfg.attnNhwcCoopmatMaintenance1SplitKSubgroupSize != (int32_t)info.subgroupSize)
        why = "its subgroup size does not match runtime subgroup size";
      else if(!coopmatShapeIsSupported(
                info.coopmatShapes,
                cfg.attnNhwcCoopmatMaintenance1SplitKTM,
                cfg.attnNhwcCoopmatMaintenance1SplitKTN,
                cfg.attnNhwcCoopmatMaintenance1SplitKTK))
        why = "its coopmat fragment shape is not one this GPU reports";
      if(
        why.empty() && !VulkanKernels::AttentionCoopmat1Nhwc::isConfigSupported(
                         cfg.attnNhwcCoopmatMaintenance1SplitKWorkgroupSize,
                         cfg.attnNhwcCoopmatMaintenance1SplitKBlockQ,
                         cfg.attnNhwcCoopmatMaintenance1SplitKBlockKV,
                         cfg.attnNhwcCoopmatMaintenance1SplitKTM,
                         cfg.attnNhwcCoopmatMaintenance1SplitKTN,
                         cfg.attnNhwcCoopmatMaintenance1SplitKTK,
                         cfg.attnNhwcCoopmatMaintenance1SplitKSubgroupSize,
                         64,
                         64))
        why = "its tile config is not supported";
      if(!why.empty()) {
        disable("attnNhwcCoopmatMaintenance1SplitK", why, {TUNED_ATTN_MAINTENANCE1_SPLITK});
        resetFields({
#define VULKAN_TUNE_PARAM_KERNEL_ATTN_MAINTENANCE1_SPLITK
#include "vulkantuneparams_fields.inc"
        });
      }
    }
    if(cfg.hasKernelTuned(TUNED_ATTN_COOPMAT2)) {
      string why;
      if(!info.supportsCoopmat2Attention)
        why = "this GPU does not support coopmat2 attention";
      else {
        bool granularityOk = false;
        for(const auto& s: info.coopmat2FlexShapes) {
          if(
            cfg.attnNhwcCoopmat2WorkgroupSize == (int32_t)s.workgroupInvocations && s.mGranularity > 0 &&
            s.nGranularity > 0 && s.kGranularity > 0 && cfg.attnNhwcCoopmat2BlockQ % (int32_t)s.mGranularity == 0 &&
            cfg.attnNhwcCoopmat2BlockKV % (int32_t)s.nGranularity == 0 &&
            cfg.attnNhwcCoopmat2BlockKV % (int32_t)s.kGranularity == 0) {
            granularityOk = true;
            break;
          }
        }
        if(!granularityOk)
          why = "its block/granularity config is not one this GPU reports";
      }
      if(
        why.empty() &&
        !VulkanKernels::AttentionCoopmat2AccF32Nhwc::isConfigSupported(
          cfg.attnNhwcCoopmat2WorkgroupSize, cfg.attnNhwcCoopmat2BlockQ, cfg.attnNhwcCoopmat2BlockKV, 64, 64))
        why = "its tile config is not supported";
      if(!why.empty()) {
        disable("attnNhwcCoopmat2", why, {TUNED_ATTN_COOPMAT2});
        resetFields({
#define VULKAN_TUNE_PARAM_KERNEL_ATTN_COOPMAT2_FULL
#include "vulkantuneparams_fields.inc"
        });
      }
    }
    if(cfg.hasKernelTuned(TUNED_ATTN_DOT2)) {
      string why = dot2Reason(false);
      if(
        why.empty() && !VulkanKernels::AttentionDot2AccF32Nhwc::isConfigSupported(
                         cfg.attnNhwcDot2WorkgroupSize, cfg.attnNhwcDot2BlockQ, cfg.attnNhwcDot2BlockKV, 64, 64))
        why = "its tile config is not supported";
      if(!why.empty()) {
        disable("attnNhwcDot2", why, {TUNED_ATTN_DOT2});
        resetFields({
#define VULKAN_TUNE_PARAM_KERNEL_ATTN_DOT2_FULL
#include "vulkantuneparams_fields.inc"
        });
      }
    }
  }

  // -------------------------------------------------------------------------
  // Derived entries. The Winograd conv path needs a valid Winograd GEMM variant
  // on this device; the transform/untransform workgroup sizes are tied to that
  // variant's layout, so a stale layout signature forces a transform re-tune.
  // -------------------------------------------------------------------------
  if(
    (cfg.hasKernelTuned(TUNED_CONV3X3_WINOGRAD) || cfg.hasKernelTuned(TUNED_CONV5X5_WINOGRAD)) ||
    !cfg.nhwcWinogradTransformTunedLayout.empty()) {
    const bool winogradGemmValid = cfg.hasKernelTuned(TUNED_WINOGRAD_TILED) ||
                                   (fp16 && resolveWinogradGemmVariant(info, cfg, fp16) != TUNED_WINOGRAD_TILED);
    if(!winogradGemmValid) {
      disable(
        "nhwcWinograd",
        "no valid Winograd GEMM variant remains for this device",
        {TUNED_CONV3X3_WINOGRAD, TUNED_CONV5X5_WINOGRAD, TUNED_WINOGRAD_TRANSFORM, TUNED_WINOGRAD_UNTRANSFORM});
      cfg.nhwcWinogradTransformTunedLayout.clear();
    } else if(
      cfg.nhwcWinogradTransformTunedLayout == winogradTransformLayoutSignature(cfg, info, fp16Storage, fp16Compute)) {
      // Layout is current, so the recorded transforms are still valid. Adopt the
      // layout signature as their tuned state (files written before the
      // per-kernel bits existed carry no bits, only the signature).
      cfg.markKernelTuned({TUNED_WINOGRAD_TRANSFORM, TUNED_WINOGRAD_UNTRANSFORM});
    } else if(!cfg.nhwcWinogradTransformTunedLayout.empty()) {
      if(logger != nullptr)
        logger->write(
          "Vulkan tune file Winograd transforms were tuned for a different layout; clearing to re-tune them.");
      cfg.clearKernelTuned({TUNED_WINOGRAD_TRANSFORM, TUNED_WINOGRAD_UNTRANSFORM});
      cfg.nhwcWinogradTransformTunedLayout.clear();
      disabledAny = true;
    }
  }

  // Restore the filter-disabled variants untouched (their tuned/invalid state
  // is preserved for a later unfiltered run), derive the invalid bits from any
  // tuned-but-unmeasured score (disabled bits are still hidden here, so they are
  // left alone), and re-apply the invariant that invalidKernelMask is a subset
  // of tunedKernelMask.
  cfg.invalidKernelMask |= computeInvalidVariantBitsFromTimeUs(cfg);
  cfg.tunedKernelMask |= disabledTunedBits;
  cfg.invalidKernelMask |= disabledInvalidBits;
  cfg.invalidKernelMask &= cfg.tunedKernelMask;

  return disabledAny;
}

// ============================================================================
// VulkanTuneParams persistence
// ============================================================================

bool VulkanTuneParams::hasKernelTuned(int64_t flag) const {
  return (tunedKernelMask & flag) != 0;
}

bool VulkanTuneParams::allOfKernelTuned(std::initializer_list<int64_t> flags) const {
  for(int64_t flag: flags)
    if((tunedKernelMask & flag) == 0)
      return false;
  return true;
}

bool VulkanTuneParams::anyOfKernelTuned(std::initializer_list<int64_t> flags) const {
  for(int64_t flag: flags)
    if((tunedKernelMask & flag) != 0)
      return true;
  return false;
}

void VulkanTuneParams::markKernelTuned(std::initializer_list<int64_t> flags) {
  for(int64_t flag: flags)
    tunedKernelMask |= flag;
}

void VulkanTuneParams::clearKernelTuned(std::initializer_list<int64_t> flags) {
  for(int64_t flag: flags) {
    tunedKernelMask &= ~flag;
    invalidKernelMask &= ~flag;
  }
}

namespace VulkanTuner {

  bool operator==(const VulkanTuneParams& a, const VulkanTuneParams& b) {
#define VULKAN_TUNE_PARAM_ALL
#define VULKAN_TUNE_PARAM_FIELD(TYPE, NAME, DEFAULT, CANDIDATES) \
  if(a.NAME != b.NAME) \
    return false;
#include "vulkantuneparams_fields.inc"
    return true;
  }

}  // namespace VulkanTuner

bool VulkanTuneParams::isValid() const {
  return VulkanKernels::LayoutTransform::isConfigSupported(
           nchwToNhwcSmallTile, nchwToNhwcLargeTile, nchwToNhwcTileCrossover) &&
         VulkanKernels::WinogradGemm::isConfigSupported(winogradGemmM, winogradGemmN, winogradGemmK, winogradGemmRN) &&
         VulkanKernels::GemmStridedTiledNhwc::isConfigSupported(
           gemmStridedTiledNhwcLocalSizeX,
           gemmStridedTiledNhwcLocalSizeY,
           gemmStridedTiledNhwcTileK,
           gemmStridedTiledNhwcRN) &&
         winogradGemmCoopmat1F32TimeUs >= 0 && winogradGemmCoopmat2F32TimeUs >= 0 &&
         winogradGemmCoopmat1F16TimeUs >= 0 && winogradGemmCoopmat2F16TimeUs >= 0 && winogradGemmDot2F32TimeUs >= 0 &&
         winogradGemmDot2F16TimeUs >= 0 && (nhwcWinograd3x3OutTile == 2 || nhwcWinograd3x3OutTile == 4) &&
         VulkanKernels::WinogradTransformNhwc::isConfigSupported(
           nhwcWinogradTransformLocalSizeX, nhwcWinogradTransformLocalSizeY) &&
         VulkanKernels::WinogradUntransformNhwc::isConfigSupported(
           nhwcWinogradUntransformLocalSizeX, nhwcWinogradUntransformLocalSizeY) &&
         VulkanKernels::GemmDirectFP32::isConfigSupported(gemmDirectLocalSizeX, gemmDirectLocalSizeY) &&
         VulkanKernels::AttentionTiled::isConfigSupported(attnNhwcBlockQ, attnNhwcBlockKV, attnNhwcQPerThread) &&
         VulkanKernels::AttentionCoopmat1Nhwc::isModeConfigSupported(
           attnNhwcCoopmatMaintenance1DirectKV,
           /*kvChunkCount=*/1,  // maintenance1's regular path is always built with a single KV chunk
           attnNhwcCoopmatMaintenance1SplitKDirectKV,
           attnNhwcCoopmatMaintenance1SplitKKVChunkCount,
           attnNhwcCoopmatMaintenance1SplitKCutoffBatch) &&
         attnNhwcCoopmatMaintenance1TimeUs >= 0 && attnNhwcCoopmatMaintenance1TN == attnNhwcCoopmatMaintenance1TK &&
         attnNhwcCoopmatMaintenance1SplitKTN == attnNhwcCoopmatMaintenance1SplitKTK &&
         VulkanKernels::AttentionCoopmat1Nhwc::isConfigSupported(
           attnNhwcCoopmatMaintenance1WorkgroupSize,
           attnNhwcCoopmatMaintenance1BlockQ,
           attnNhwcCoopmatMaintenance1BlockKV,
           attnNhwcCoopmatMaintenance1TM,
           attnNhwcCoopmatMaintenance1TN,
           attnNhwcCoopmatMaintenance1TK,
           attnNhwcCoopmatMaintenance1SubgroupSize,
           64,
           64) &&
         VulkanKernels::AttentionCoopmat1Nhwc::isConfigSupported(
           attnNhwcCoopmatMaintenance1SplitKWorkgroupSize,
           attnNhwcCoopmatMaintenance1SplitKBlockQ,
           attnNhwcCoopmatMaintenance1SplitKBlockKV,
           attnNhwcCoopmatMaintenance1SplitKTM,
           attnNhwcCoopmatMaintenance1SplitKTN,
           attnNhwcCoopmatMaintenance1SplitKTK,
           attnNhwcCoopmatMaintenance1SplitKSubgroupSize,
           64,
           64) &&
         VulkanKernels::AttentionCoopmat1Nhwc::isModeConfigSupported(
           attnNhwcCoopmat1DirectKV,
           /*kvChunkCount=*/1,  // the regular coopmat1 path is always built with a single KV chunk
           attnNhwcCoopmat1SplitKDirectKV,
           attnNhwcCoopmat1SplitKKVChunkCount,
           attnNhwcCoopmat1SplitKCutoffBatch) &&
         VulkanKernels::AttentionCoopmat1Nhwc::isConfigSupported(
           attnNhwcCoopmat1SplitKWorkgroupSize,
           attnNhwcCoopmat1SplitKBlockQ,
           attnNhwcCoopmat1SplitKBlockKV,
           attnNhwcCoopmat1SplitKTM,
           attnNhwcCoopmat1SplitKTN,
           attnNhwcCoopmat1SplitKTK,
           attnNhwcCoopmat1SplitKSubgroupSize,
           64,
           64) &&
         attnNhwcCoopmat1TimeUs >= 0 &&
         VulkanKernels::AttentionCoopmat1Nhwc::isConfigSupported(
           attnNhwcCoopmat1WorkgroupSize,
           attnNhwcCoopmat1BlockQ,
           attnNhwcCoopmat1BlockKV,
           attnNhwcCoopmat1TM,
           attnNhwcCoopmat1TN,
           attnNhwcCoopmat1TK,
           attnNhwcCoopmat1SubgroupSize,
           64,
           64) &&
         attnNhwcCoopmat2TimeUs >= 0 &&
         VulkanKernels::AttentionCoopmat2AccF32Nhwc::isConfigSupported(
           attnNhwcCoopmat2WorkgroupSize, attnNhwcCoopmat2BlockQ, attnNhwcCoopmat2BlockKV, 64, 64) &&
         attnNhwcDot2TimeUs >= 0 &&
         VulkanKernels::AttentionDot2AccF32Nhwc::isConfigSupported(
           attnNhwcDot2WorkgroupSize, attnNhwcDot2BlockQ, attnNhwcDot2BlockKV, 64, 64) &&
         isPow2(gpoolNhwcXystride) && isPow2(valueHeadPoolNhwcXystride) &&
         VulkanKernels::SpatialRMSNormNhwc::isConfigSupported(spatialRMSNormNhwcTile) &&
         VulkanKernels::SwiGLU::isConfigSupported(swiGLULocalSizeX) &&
         VulkanKernels::WinogradGemmDot2::isConfigSupported(
           dot2WorkgroupSize, dot2BM, dot2BN, dot2SGM, dot2SGN, dot2SGMIter, dot2TM, dot2TN, dot2SubgroupSize) &&
         VulkanKernels::WinogradGemmDot2AccF16::isConfigSupported(
           dot2AccF16WorkgroupSize,
           dot2AccF16BM,
           dot2AccF16BN,
           dot2AccF16SGM,
           dot2AccF16SGN,
           dot2AccF16SGMIter,
           dot2AccF16TM,
           dot2AccF16TN,
           dot2AccF16SubgroupSize) &&
         VulkanKernels::WinogradGemmCoopmat1::isConfigSupported(
           coopmat1WorkgroupSize,
           coopmat1BM,
           coopmat1BN,
           coopmat1BK,
           coopmat1SGM,
           coopmat1SGN,
           coopmat1TM,
           coopmat1TN,
           coopmat1TK,
           coopmat1SubgroupSize) &&
         VulkanKernels::Conv3x3ImplicitGemmCoopmat1AccF32NhwcVec8::isConfigSupported(
           conv3x3NhwcCoopmat1WorkgroupSize,
           conv3x3NhwcCoopmat1BM,
           conv3x3NhwcCoopmat1BN,
           conv3x3NhwcCoopmat1BK,
           conv3x3NhwcCoopmat1SGM,
           conv3x3NhwcCoopmat1SGN,
           conv3x3NhwcCoopmat1TM,
           conv3x3NhwcCoopmat1TN,
           conv3x3NhwcCoopmat1TK,
           conv3x3NhwcCoopmat1SubgroupSize) &&
         VulkanKernels::Conv3x3ImplicitGemmCoopmat1AccF32NhwcVec8::isConfigSupported(
           conv5x5NhwcCoopmat1WorkgroupSize,
           conv5x5NhwcCoopmat1BM,
           conv5x5NhwcCoopmat1BN,
           conv5x5NhwcCoopmat1BK,
           conv5x5NhwcCoopmat1SGM,
           conv5x5NhwcCoopmat1SGN,
           conv5x5NhwcCoopmat1TM,
           conv5x5NhwcCoopmat1TN,
           conv5x5NhwcCoopmat1TK,
           conv5x5NhwcCoopmat1SubgroupSize) &&
         VulkanKernels::WinogradGemmCoopmat1AccF16::isConfigSupported(
           coopmat1AccF16WorkgroupSize,
           coopmat1AccF16BM,
           coopmat1AccF16BN,
           coopmat1AccF16BK,
           coopmat1AccF16SGM,
           coopmat1AccF16SGN,
           coopmat1AccF16TM,
           coopmat1AccF16TN,
           coopmat1AccF16TK,
           coopmat1AccF16SubgroupSize) &&
         VulkanKernels::GemmStridedCoopmat1Nhwc::isConfigSupported(
           nhwcStridedCoopmat1WorkgroupSize,
           nhwcStridedCoopmat1BM,
           nhwcStridedCoopmat1BN,
           nhwcStridedCoopmat1BK,
           nhwcStridedCoopmat1SGM,
           nhwcStridedCoopmat1SGN,
           nhwcStridedCoopmat1TM,
           nhwcStridedCoopmat1TN,
           nhwcStridedCoopmat1TK,
           nhwcStridedCoopmat1SubgroupSize) &&
         VulkanKernels::GemmStridedCoopmat1AccF16Nhwc::isConfigSupported(
           nhwcStridedCoopmat1AccF16WorkgroupSize,
           nhwcStridedCoopmat1AccF16BM,
           nhwcStridedCoopmat1AccF16BN,
           nhwcStridedCoopmat1AccF16BK,
           nhwcStridedCoopmat1AccF16SGM,
           nhwcStridedCoopmat1AccF16SGN,
           nhwcStridedCoopmat1AccF16TM,
           nhwcStridedCoopmat1AccF16TN,
           nhwcStridedCoopmat1AccF16TK,
           nhwcStridedCoopmat1AccF16SubgroupSize) &&
         VulkanKernels::GemmStridedCoopmat2Nhwc::isConfigSupported(
           nhwcStridedCoopmat2WorkgroupSize, nhwcStridedCoopmat2BM, nhwcStridedCoopmat2BN, nhwcStridedCoopmat2BK) &&
         VulkanKernels::GemmStridedCoopmat2AccF16Nhwc::isConfigSupported(
           nhwcStridedCoopmat2AccF16WorkgroupSize,
           nhwcStridedCoopmat2AccF16BM,
           nhwcStridedCoopmat2AccF16BN,
           nhwcStridedCoopmat2AccF16BK) &&
         VulkanKernels::Conv3x3ImplicitGemmCoopmat2AccF32NhwcVec8::isConfigSupported(
           conv3x3NhwcCoopmat2WorkgroupSize, conv3x3NhwcCoopmat2BM, conv3x3NhwcCoopmat2BN, conv3x3NhwcCoopmat2BK) &&
         VulkanKernels::Conv3x3ImplicitGemmCoopmat2AccF32NhwcVec8::isConfigSupported(
           conv5x5NhwcCoopmat2WorkgroupSize, conv5x5NhwcCoopmat2BM, conv5x5NhwcCoopmat2BN, conv5x5NhwcCoopmat2BK) &&
         nhwcGemmCoopmat1F32TimeUs >= 0 && nhwcGemmCoopmat2F32TimeUs >= 0 && nhwcGemmCoopmat1F16TimeUs >= 0 &&
         nhwcGemmCoopmat2F16TimeUs >= 0 &&
         VulkanKernels::GemmStridedDot2Nhwc::isConfigSupported(
           nhwcStridedDot2WorkgroupSize,
           nhwcStridedDot2BM,
           nhwcStridedDot2BN,
           nhwcStridedDot2SGM,
           nhwcStridedDot2SGN,
           nhwcStridedDot2SGMIter,
           nhwcStridedDot2TM,
           nhwcStridedDot2TN,
           nhwcStridedDot2SubgroupSize) &&
         VulkanKernels::GemmStridedDot2AccF16Nhwc::isConfigSupported(
           nhwcStridedDot2AccF16WorkgroupSize,
           nhwcStridedDot2AccF16BM,
           nhwcStridedDot2AccF16BN,
           nhwcStridedDot2AccF16SGM,
           nhwcStridedDot2AccF16SGN,
           nhwcStridedDot2AccF16SGMIter,
           nhwcStridedDot2AccF16TM,
           nhwcStridedDot2AccF16TN,
           nhwcStridedDot2AccF16SubgroupSize) &&
         nhwcGemmDot2F32TimeUs >= 0 && nhwcGemmDot2F16TimeUs >= 0 && conv3x3NhwcCoopmat1TimeUs >= 0 &&
         conv3x3NhwcCoopmat2TimeUs >= 0 && conv5x5NhwcCoopmat1TimeUs >= 0 && conv5x5NhwcCoopmat2TimeUs >= 0 &&
         conv3x3NhwcCoopmat1AccF16TimeUs >= 0 && conv3x3NhwcCoopmat2AccF16TimeUs >= 0 &&
         conv5x5NhwcCoopmat1AccF16TimeUs >= 0 && conv5x5NhwcCoopmat2AccF16TimeUs >= 0 &&
         VulkanKernels::WinogradGemmCoopmat2::isConfigSupported(
           coopmat2WorkgroupSize, coopmat2BM, coopmat2BN, coopmat2BK) &&
         VulkanKernels::WinogradGemmCoopmat2AccF16::isConfigSupported(
           coopmat2AccF16WorkgroupSize, coopmat2AccF16BM, coopmat2AccF16BN, coopmat2AccF16BK);
}

void VulkanTuneParams::save(const string& filename, const VulkanTuneParams& config) {
  string tmpFilename = uniqueTempFileNameForAtomicSave(filename);
  ofstream out;
  FileUtils::open(out, tmpFilename, std::ios::out | std::ios::trunc);
  out << VULKAN_TUNEPARAMS_VERSION_LINE << "\n";
#define VULKAN_TUNE_PARAM_ALL
#define VULKAN_TUNE_PARAM_FIELD(TYPE, NAME, DEFAULT, CANDIDATES) out << #NAME "=" << config.NAME << "\n";
#include "vulkantuneparams_fields.inc"
  finishAtomicSave(filename, tmpFilename, out);
}

namespace {

  struct TuneParamsLoadState {
    uint32_t nhwcTiledGemmFields = 0;
    uint32_t nchwToNhwcFields = 0;
  };

  // Parse one field value from a save-file token. The structural string is
  // assigned verbatim, the int64_t masks via stringToInt64, and every other
  // field is int32_t via stringToInt (replicating the previous load behavior).
  template<typename T>
  T parseTuneValue(const string& valueStr);
  template<>
  string parseTuneValue<string>(const string& valueStr) {
    return valueStr;
  }
  template<>
  int64_t parseTuneValue<int64_t>(const string& valueStr) {
    return Global::stringToInt64(valueStr);
  }
  template<>
  int32_t parseTuneValue<int32_t>(const string& valueStr) {
    return (int32_t)Global::stringToInt(valueStr);
  }

  // Track which of the presence-checked fields appeared, so the tail of load()
  // can clear a kernel's tuned bit when one of its field groups is incomplete.
  void recordFieldSeen(TuneParamsLoadState& loadState, string_view key) {
    if(key == "nchwToNhwcSmallTile")
      loadState.nchwToNhwcFields |= 1u << 0;
    else if(key == "nchwToNhwcLargeTile")
      loadState.nchwToNhwcFields |= 1u << 1;
    else if(key == "nchwToNhwcTileCrossover")
      loadState.nchwToNhwcFields |= 1u << 2;
    else if(key == "gemmStridedTiledNhwcLocalSizeX")
      loadState.nhwcTiledGemmFields |= 1u << 0;
    else if(key == "gemmStridedTiledNhwcLocalSizeY")
      loadState.nhwcTiledGemmFields |= 1u << 1;
    else if(key == "gemmStridedTiledNhwcTileK")
      loadState.nhwcTiledGemmFields |= 1u << 2;
    else if(key == "gemmStridedTiledNhwcRN")
      loadState.nhwcTiledGemmFields |= 1u << 3;
  }

  void
  fillFromDesc(const string& filename, const string& desc, VulkanTuneParams& config, TuneParamsLoadState& loadState) {
    istringstream in(desc);
    string token;
    while(in >> token) {
      size_t eq = token.find('=');
      if(eq == string::npos)
        throw IOError("VulkanTuneParams::load: bad token '" + token + "' in " + filename);
      string key = token.substr(0, eq);
      string valueStr = token.substr(eq + 1);
#define VULKAN_TUNE_PARAM_ALL
#define VULKAN_TUNE_PARAM_FIELD(TYPE, NAME, DEFAULT, CANDIDATES) \
  if(key == #NAME) { \
    config.NAME = parseTuneValue<TYPE>(valueStr); \
    recordFieldSeen(loadState, #NAME); \
  } else
#include "vulkantuneparams_fields.inc"
      throw IOError("VulkanTuneParams::load: unknown key '" + key + "' in " + filename);
    }
  }

}  // namespace

VulkanTuneParams VulkanTuneParams::load(const string& filename) {
  vector<string> lines = FileUtils::readFileLines(filename, '\n');
  vector<string> filteredLines;
  for(size_t i = 0; i < lines.size(); i++) {
    string line = Global::trim(Global::stripComments(lines[i]));
    if(line.length() > 0)
      filteredLines.push_back(line);
  }
  if(filteredLines.empty())
    throw IOError("VulkanTuneParams::load: no params in file " + filename);
  if(filteredLines[0] != string(VULKAN_TUNEPARAMS_VERSION_LINE))
    throw IOError(
      "VulkanTuneParams::load: expected first line " + string(VULKAN_TUNEPARAMS_VERSION_LINE) + " in " + filename);
  // Structural validation is done by fillFromDesc (rejects unknown keys) and by
  // reconcileTuneParamsForDevice at the call site (resets anything unsupported
  // and clears its tuned bit).
  // We deliberately do NOT hardcode an expected line count: adding a new tuner
  // field is a routine change, and coupling load() to save()'s exact line count
  // was silently rejecting every self-produced tune file whenever the schema
  // grew, forcing a full retune from defaults on every launch.

  VulkanTuneParams config;
  TuneParamsLoadState loadState;
  for(size_t i = 1; i < filteredLines.size(); i++)
    fillFromDesc(filename, filteredLines[i], config, loadState);

  // These two fallback kernels are "always present" and validated structurally in
  // reconcile, but reconcile only checks that the loaded VALUES are supported, not
  // that the FILE contained all of the group's fields. A stale or hand-edited file
  // missing one field leaves it at its default, producing a mixed config that can
  // still pass reconcile's isConfigSupported() and so would otherwise be trusted as
  // tuned. Clearing the tuned bit here forces a re-tune with a complete field set.
  if(loadState.nhwcTiledGemmFields != 0xFu) {
    config.clearKernelTuned({VulkanTuner::TUNED_GEMM_STRIDED_TILED});
  }
  if(loadState.nchwToNhwcFields != 0x7u) {
    config.clearKernelTuned({VulkanTuner::TUNED_NCHW_TO_NHWC});
  }
  // Enforce the (tunedKernelMask, invalidKernelMask) invariant on load: an
  // invalid bit without its tuned bit is meaningless.
  config.invalidKernelMask &= config.tunedKernelMask;
  return config;
}

// ============================================================================
// File path / naming
// ============================================================================

// Returns true if the model has a transformer attention block anywhere in its
// trunk (including inside nested bottleneck blocks).
namespace VulkanTuner {

  bool modelHasTransformer(const ModelDesc* desc) {
    std::function<bool(const vector<pair<int, unique_ptr_void>>&)> scan =
      [&](const vector<pair<int, unique_ptr_void>>& blocks) -> bool {
      for(const auto& kv: blocks) {
        if(kv.first == TRANSFORMER_ATTENTION_BLOCK_KIND)
          return true;
        if(kv.first == NESTED_BOTTLENECK_BLOCK_KIND) {
          const auto* nbt = static_cast<const NestedBottleneckResidualBlockDesc*>(kv.second.get());
          if(scan(nbt->blocks))
            return true;
        }
      }
      return false;
    };
    return scan(desc->trunk.blocks);
  }

}  // namespace VulkanTuner

int64_t VulkanTuner::requiredKernelMask(
  const ModelDesc* modelDesc,
  const VulkanDeviceInfo& deviceInfo,
  bool fp16Storage,
  bool fp16Compute,
  bool full) {
  if(modelDesc == nullptr)
    return 0;
  const bool hasTransformer = modelHasTransformer(modelDesc);
  bool has3x3Convolution = false;
  bool has5x5Convolution = false;
  bool hasStridedGemm = hasTransformer;
  modelDesc->iterConvLayers([&](const ConvLayerDesc& conv) noexcept {
    has3x3Convolution = has3x3Convolution || (conv.convXSize == 3 && conv.convYSize == 3);
    has5x5Convolution = has5x5Convolution || (conv.convXSize == 5 && conv.convYSize == 5);
    hasStridedGemm = hasStridedGemm || (conv.convXSize == 1 && conv.convYSize == 1);
  });
  const bool fp16 = fp16Storage && fp16Compute;
  const int64_t dm = deviceInfo.disabledAccelVariantMask;
  const bool coopmat1 = deviceInfo.supportsCoopmat1F16 && (dm & coopmat1AccF32Bits()) == 0;
  const bool coopmat1AccF16 = deviceInfo.supportsCoopmat1F16AccF16 && (dm & coopmat1AccF16Bits()) == 0;
  const bool coopmat2 = deviceInfo.supportsCoopmat2F16 && (dm & coopmat2AccF32Bits()) == 0;
  const bool coopmat2AccF16 = deviceInfo.supportsCoopmat2F16AccF16 && (dm & coopmat2AccF16Bits()) == 0;
  const bool dot2 = deviceInfo.supportsDot2F16 && (dm & dot2AccF32Bits()) == 0;
  const bool dot2AccF16 = deviceInfo.supportsDot2F16AccF16 && (dm & dot2AccF16Bits()) == 0;
  const bool coopFamily = coopmat1 || coopmat1AccF16 || coopmat2 || coopmat2AccF16;
  const bool dot2Family = dot2 || dot2AccF16;
  const bool coopmat2Attention = coopmat2 && deviceInfo.supportsCoopmat2Attention;
  const bool coopmatMaintenance1 = coopmat1 && deviceInfo.supportsCoopmatMaintenance1;

  int64_t mask = TUNED_GEMM_DIRECT | TUNED_NCHW_TO_NHWC | TUNED_GPOOL_NHWC | TUNED_VALUE_HEAD_POOL_NHWC;
  if(hasTransformer) {
    mask |= TUNED_SWIGLU | TUNED_SPATIAL_RMSNORM_NHWC;
    // Accelerated attention tiers: the coopmat/coopmat2 families are required
    // whenever supported; DOT2 is only required when no coopmat tier remains.
    // The tiled fallback is only required when no accelerated tier remains (or
    // for a full tune, which benchmarks the fallback so it stays usable).
    const bool coopAttention = fp16 && (coopmat1 || coopmat2Attention || coopmatMaintenance1);
    const bool dot2Attention = fp16 && !coopmat1 && !coopmat2Attention && dot2;
    if(!(coopAttention || dot2Attention) || full)
      mask |= TUNED_ATTN_TILED;
    if(coopAttention) {
      if(coopmat1)
        mask |= TUNED_ATTN_COOPMAT1 | TUNED_ATTN_COOPMAT1_SPLITK;
      if(coopmat2Attention)
        mask |= TUNED_ATTN_COOPMAT2;
      if(coopmatMaintenance1)
        mask |= TUNED_ATTN_MAINTENANCE1 | TUNED_ATTN_MAINTENANCE1_SPLITK;
    } else if(dot2) {
      mask |= TUNED_ATTN_DOT2;
    }
  }

  // An accelerated strided-GEMM variant (coopmat1/2 or dot2, any accumulator)
  // supersedes the tiled fallback. The tiled tier is therefore only required
  // on devices/filters where no such variant remains. A full tune additionally
  // benchmarks the fallback tier so it stays usable.
  if(hasStridedGemm) {
    if(fp16) {
      if(coopFamily) {
        if(coopmat1)
          mask |= TUNED_GEMM_STRIDED_COOPMAT1;
        if(coopmat1AccF16)
          mask |= TUNED_GEMM_STRIDED_COOPMAT1_ACCF16;
        if(coopmat2)
          mask |= TUNED_GEMM_STRIDED_COOPMAT2;
        if(coopmat2AccF16)
          mask |= TUNED_GEMM_STRIDED_COOPMAT2_ACCF16;
      } else if(dot2Family) {
        if(dot2)
          mask |= TUNED_GEMM_STRIDED_DOT2;
        if(dot2AccF16)
          mask |= TUNED_GEMM_STRIDED_DOT2_ACCF16;
      }
    }
    if(!(fp16 && (coopFamily || dot2Family)) || full)
      mask |= TUNED_GEMM_STRIDED_TILED;
  }

  // Native implicit-im2col convolutions use the coopmat family for both
  // accumulator types; when no coopmat family remains, or for a full tune,
  // they fall back to Winograd.
  if(has3x3Convolution) {
    if(fp16 && coopFamily) {
      if(coopmat1)
        mask |= TUNED_CONV3X3_COOPMAT1;
      if(coopmat1AccF16)
        mask |= TUNED_CONV3X3_COOPMAT1_ACCF16;
      if(coopmat2)
        mask |= TUNED_CONV3X3_COOPMAT2;
      if(coopmat2AccF16)
        mask |= TUNED_CONV3X3_COOPMAT2_ACCF16;
    }
    if(!(fp16 && coopFamily) || full)
      mask |= TUNED_CONV3X3_WINOGRAD;
  }
  if(has5x5Convolution) {
    if(fp16 && coopFamily) {
      if(coopmat1)
        mask |= TUNED_CONV5X5_COOPMAT1;
      if(coopmat1AccF16)
        mask |= TUNED_CONV5X5_COOPMAT1_ACCF16;
      if(coopmat2)
        mask |= TUNED_CONV5X5_COOPMAT2;
      if(coopmat2AccF16)
        mask |= TUNED_CONV5X5_COOPMAT2_ACCF16;
    }
    if(!(fp16 && coopFamily) || full)
      mask |= TUNED_CONV5X5_WINOGRAD;
  }

  // The Winograd conv path runs on the selected Winograd GEMM variant. Require
  // the flavors of the highest available accelerator tier (coopmat > dot2 >
  // tiled), which is what the runtime resolver can actually select.
  if((mask & (TUNED_CONV3X3_WINOGRAD | TUNED_CONV5X5_WINOGRAD)) != 0) {
    // The NHWC transform/untransform kernels feed the winograd conv path, so
    // they are required (and re-tuned on layout change) exactly when that path
    // is selected.
    mask |= TUNED_WINOGRAD_TRANSFORM | TUNED_WINOGRAD_UNTRANSFORM;
    if(fp16 && coopFamily) {
      if(coopmat1)
        mask |= TUNED_WINOGRAD_COOPMAT1;
      if(coopmat1AccF16)
        mask |= TUNED_WINOGRAD_COOPMAT1_ACCF16;
      if(coopmat2)
        mask |= TUNED_WINOGRAD_COOPMAT2;
      if(coopmat2AccF16)
        mask |= TUNED_WINOGRAD_COOPMAT2_ACCF16;
    } else if(fp16 && dot2Family) {
      if(dot2)
        mask |= TUNED_WINOGRAD_DOT2;
      if(dot2AccF16)
        mask |= TUNED_WINOGRAD_DOT2_ACCF16;
    } else {
      mask |= TUNED_WINOGRAD_TILED;
    }
  }
  return mask;
}

string VulkanTuner::defaultDirectory(const string& homeDataDirOverride) {
  string dir = HomeData::getHomeDataDir(true, homeDataDirOverride);
  dir += "/vulkantuning";
  MakeDir::make(dir);
  return dir;
}

string VulkanTuner::defaultFileName(
  const string& gpuName,
  const string& tuneDeviceKey,
  int nnXLen,
  int nnYLen,
  const ModelDesc* modelDesc,
  bool fp16) {
  string gpuNameForFile;
  for(size_t i = 0; i < gpuName.length(); i++) {
    char c = gpuName[i];
    if(contains("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789", c))
      gpuNameForFile += c;
  }
  bool hasTransformer = modelHasTransformer(modelDesc);
  string_view fp16Suffix = fp16 ? "_fp16" : "";

  int trunkC = modelDesc->trunk.trunkNumChannels;
  int mv = modelDesc->modelVersion;
  if(hasTransformer) {
    // Find the first attention block's head dim for the filename.
    // Recurse into nested-bottleneck blocks so nested-only transformers
    // still get an accurate head-dim in the cache key.
    std::function<int(const vector<pair<int, unique_ptr_void>>&)> findHeadDim =
      [&](const vector<pair<int, unique_ptr_void>>& blocks) -> int {
      for(const auto& kv: blocks) {
        if(kv.first == TRANSFORMER_ATTENTION_BLOCK_KIND)
          return static_cast<const TransformerAttentionDesc*>(kv.second.get())->qHeadDim;
        if(kv.first == NESTED_BOTTLENECK_BLOCK_KIND) {
          const auto* nbt = static_cast<const NestedBottleneckResidualBlockDesc*>(kv.second.get());
          int hd = findHeadDim(nbt->blocks);
          if(hd != 0)
            return hd;
        }
      }
      return 0;
    };
    int headDim = findHeadDim(modelDesc->trunk.blocks);
    return Global::strprintf(
      "tune%d_gpu%s_dev-%s_x%d_y%d_c%d_h%d_mv%d%s.txt",
      VULKAN_TUNER_VERSION,
      gpuNameForFile.c_str(),
      tuneDeviceKey.c_str(),
      nnXLen,
      nnYLen,
      trunkC,
      headDim,
      mv,
      fp16Suffix.data());
  }
  return Global::strprintf(
    "tune%d_gpu%s_dev-%s_x%d_y%d_c%d_mv%d%s.txt",
    VULKAN_TUNER_VERSION,
    gpuNameForFile.c_str(),
    tuneDeviceKey.c_str(),
    nnXLen,
    nnYLen,
    trunkC,
    mv,
    fp16Suffix.data());
}

void VulkanTuner::appendTuneParamsSummary(
  std::ostream& out,
  const VulkanTuneParams& config,
  const ModelDesc* modelDesc) {
  const char sep = 'x';
  auto appendTuple = [&out](std::initializer_list<int32_t> vals) {
    bool first = true;
    for(int32_t v: vals) {
      if(!first)
        out << sep;
      out << v;
      first = false;
    }
  };

  // Only tuned ("active") kernels are listed: each kernel's tile/config line is
  // gated by its TunedKernel bit, and each *TimesUs score line by the variants it
  // summarizes. tunedKernelMask is always emitted as the authoritative bitmask;
  // nhwcWinograd3x3OutTile is a structural layout parameter, not a kernel.
  out << "tunedKernelMask=" << config.tunedKernelMask;
  out << " nhwcWinograd3x3OutTile=" << config.nhwcWinograd3x3OutTile;

  if(config.hasKernelTuned(TUNED_WINOGRAD_TILED)) {
    out << " winogradGemmTiled=";
    appendTuple({config.winogradGemmM, config.winogradGemmN, config.winogradGemmK, config.winogradGemmRN});
  }
  if(config.hasKernelTuned(TUNED_NCHW_TO_NHWC)) {
    out << " nchwToNhwc=";
    appendTuple({config.nchwToNhwcSmallTile, config.nchwToNhwcLargeTile, config.nchwToNhwcTileCrossover});
  }
  if(config.hasKernelTuned(TUNED_GEMM_STRIDED_TILED)) {
    out << " gemmStridedTiledNhwc=";
    appendTuple(
      {config.gemmStridedTiledNhwcLocalSizeX,
       config.gemmStridedTiledNhwcLocalSizeY,
       config.gemmStridedTiledNhwcTileK,
       config.gemmStridedTiledNhwcRN});
  }
  if(config.hasKernelTuned(TUNED_WINOGRAD_TRANSFORM)) {
    out << " nhwcWinogradTransformWG=";
    appendTuple({config.nhwcWinogradTransformLocalSizeX, config.nhwcWinogradTransformLocalSizeY});
  }
  if(config.hasKernelTuned(TUNED_WINOGRAD_UNTRANSFORM)) {
    out << " nhwcWinogradUntransformWG=";
    appendTuple({config.nhwcWinogradUntransformLocalSizeX, config.nhwcWinogradUntransformLocalSizeY});
  }
  if(
    config.anyOfKernelTuned(
      {TUNED_WINOGRAD_COOPMAT1,
       TUNED_WINOGRAD_COOPMAT2,
       TUNED_WINOGRAD_COOPMAT1_ACCF16,
       TUNED_WINOGRAD_COOPMAT2_ACCF16,
       TUNED_WINOGRAD_DOT2,
       TUNED_WINOGRAD_DOT2_ACCF16})) {
    out << " winogradGemmTimesUs=" << config.winogradGemmCoopmat1F32TimeUs << ","
        << config.winogradGemmCoopmat2F32TimeUs << "," << config.winogradGemmCoopmat1F16TimeUs << ","
        << config.winogradGemmCoopmat2F16TimeUs << "," << config.winogradGemmDot2F32TimeUs << ","
        << config.winogradGemmDot2F16TimeUs;
  }
  if(config.hasKernelTuned(TUNED_WINOGRAD_DOT2)) {
    out << " winogradDot2Tile=";
    appendTuple(
      {config.dot2WorkgroupSize,
       config.dot2BM,
       config.dot2BN,
       config.dot2SGM,
       config.dot2SGN,
       config.dot2SGMIter,
       config.dot2TM,
       config.dot2TN,
       config.dot2SubgroupSize});
  }
  if(config.hasKernelTuned(TUNED_WINOGRAD_DOT2_ACCF16)) {
    out << " winogradDot2AccF16Tile=";
    appendTuple(
      {config.dot2AccF16WorkgroupSize,
       config.dot2AccF16BM,
       config.dot2AccF16BN,
       config.dot2AccF16SGM,
       config.dot2AccF16SGN,
       config.dot2AccF16SGMIter,
       config.dot2AccF16TM,
       config.dot2AccF16TN,
       config.dot2AccF16SubgroupSize});
  }
  if(config.hasKernelTuned(TUNED_WINOGRAD_COOPMAT1)) {
    out << " winogradCoopmat1Tile=";
    appendTuple(
      {config.coopmat1WorkgroupSize,
       config.coopmat1BM,
       config.coopmat1BN,
       config.coopmat1BK,
       config.coopmat1SGM,
       config.coopmat1SGN,
       config.coopmat1TM,
       config.coopmat1TN,
       config.coopmat1TK,
       config.coopmat1SubgroupSize});
  }
  if(config.hasKernelTuned(TUNED_CONV3X3_COOPMAT1)) {
    out << " conv3x3NhwcCoopmatTile=";
    appendTuple(
      {config.conv3x3NhwcCoopmat1WorkgroupSize,
       config.conv3x3NhwcCoopmat1BM,
       config.conv3x3NhwcCoopmat1BN,
       config.conv3x3NhwcCoopmat1BK,
       config.conv3x3NhwcCoopmat1SGM,
       config.conv3x3NhwcCoopmat1SGN,
       config.conv3x3NhwcCoopmat1TM,
       config.conv3x3NhwcCoopmat1TN,
       config.conv3x3NhwcCoopmat1TK,
       config.conv3x3NhwcCoopmat1SubgroupSize});
  }
  if(config.hasKernelTuned(TUNED_CONV5X5_COOPMAT1)) {
    out << " conv5x5NhwcCoopmatTile=";
    appendTuple(
      {config.conv5x5NhwcCoopmat1WorkgroupSize,
       config.conv5x5NhwcCoopmat1BM,
       config.conv5x5NhwcCoopmat1BN,
       config.conv5x5NhwcCoopmat1BK,
       config.conv5x5NhwcCoopmat1SGM,
       config.conv5x5NhwcCoopmat1SGN,
       config.conv5x5NhwcCoopmat1TM,
       config.conv5x5NhwcCoopmat1TN,
       config.conv5x5NhwcCoopmat1TK,
       config.conv5x5NhwcCoopmat1SubgroupSize});
  }
  if(config.hasKernelTuned(TUNED_WINOGRAD_COOPMAT1_ACCF16)) {
    out << " winogradCoopmat1AccF16Tile=";
    appendTuple(
      {config.coopmat1AccF16WorkgroupSize,
       config.coopmat1AccF16BM,
       config.coopmat1AccF16BN,
       config.coopmat1AccF16BK,
       config.coopmat1AccF16SGM,
       config.coopmat1AccF16SGN,
       config.coopmat1AccF16TM,
       config.coopmat1AccF16TN,
       config.coopmat1AccF16TK,
       config.coopmat1AccF16SubgroupSize});
  }
  if(config.hasKernelTuned(TUNED_WINOGRAD_COOPMAT2)) {
    out << " winogradCoopmat2Tile=";
    appendTuple({config.coopmat2WorkgroupSize, config.coopmat2BM, config.coopmat2BN, config.coopmat2BK});
  }
  if(config.hasKernelTuned(TUNED_WINOGRAD_COOPMAT2_ACCF16)) {
    out << " winogradCoopmat2AccF16Tile=";
    appendTuple(
      {config.coopmat2AccF16WorkgroupSize, config.coopmat2AccF16BM, config.coopmat2AccF16BN, config.coopmat2AccF16BK});
  }
  if(config.hasKernelTuned(TUNED_GEMM_DIRECT)) {
    out << " gemmDirect=";
    appendTuple({config.gemmDirectLocalSizeX, config.gemmDirectLocalSizeY});
  }
  if(config.hasKernelTuned(TUNED_GPOOL_NHWC)) {
    out << " gpoolNhwc=" << config.gpoolNhwcXystride;
  }
  if(config.hasKernelTuned(TUNED_VALUE_HEAD_POOL_NHWC)) {
    out << " valueHeadPoolNhwc=" << config.valueHeadPoolNhwcXystride;
  }
  if(config.hasKernelTuned(TUNED_GEMM_STRIDED_COOPMAT1)) {
    out << " gemmStridedNhwcCoopmat1=";
    appendTuple(
      {config.nhwcStridedCoopmat1WorkgroupSize,
       config.nhwcStridedCoopmat1BM,
       config.nhwcStridedCoopmat1BN,
       config.nhwcStridedCoopmat1BK});
  }
  if(config.hasKernelTuned(TUNED_GEMM_STRIDED_COOPMAT1_ACCF16)) {
    out << " gemmStridedNhwcCoopmat1AccF16=";
    appendTuple(
      {config.nhwcStridedCoopmat1AccF16WorkgroupSize,
       config.nhwcStridedCoopmat1AccF16BM,
       config.nhwcStridedCoopmat1AccF16BN,
       config.nhwcStridedCoopmat1AccF16BK});
  }
  if(config.hasKernelTuned(TUNED_GEMM_STRIDED_COOPMAT2)) {
    out << " gemmStridedNhwcCoopmat2=";
    appendTuple(
      {config.nhwcStridedCoopmat2WorkgroupSize,
       config.nhwcStridedCoopmat2BM,
       config.nhwcStridedCoopmat2BN,
       config.nhwcStridedCoopmat2BK});
  }
  if(
    config.anyOfKernelTuned(
      {TUNED_GEMM_STRIDED_COOPMAT1,
       TUNED_GEMM_STRIDED_COOPMAT2,
       TUNED_GEMM_STRIDED_COOPMAT1_ACCF16,
       TUNED_GEMM_STRIDED_COOPMAT2_ACCF16})) {
    out << " nhwcGemmTimesUs=" << config.nhwcGemmCoopmat1F32TimeUs << "," << config.nhwcGemmCoopmat2F32TimeUs << ","
        << config.nhwcGemmCoopmat1F16TimeUs << "," << config.nhwcGemmCoopmat2F16TimeUs;
  }
  if(config.hasKernelTuned(TUNED_GEMM_STRIDED_COOPMAT2_ACCF16)) {
    out << " nhwcGemmCoopmat2AccF16Tile=";
    appendTuple(
      {config.nhwcStridedCoopmat2AccF16WorkgroupSize,
       config.nhwcStridedCoopmat2AccF16BM,
       config.nhwcStridedCoopmat2AccF16BN,
       config.nhwcStridedCoopmat2AccF16BK});
  }
  if(config.hasKernelTuned(TUNED_GEMM_STRIDED_DOT2)) {
    out << " nhwcGemmDot2Tile=";
    appendTuple(
      {config.nhwcStridedDot2WorkgroupSize,
       config.nhwcStridedDot2BM,
       config.nhwcStridedDot2BN,
       config.nhwcStridedDot2SGM,
       config.nhwcStridedDot2SGN,
       config.nhwcStridedDot2SGMIter,
       config.nhwcStridedDot2TM,
       config.nhwcStridedDot2TN,
       config.nhwcStridedDot2SubgroupSize});
  }
  if(config.anyOfKernelTuned({TUNED_GEMM_STRIDED_DOT2, TUNED_GEMM_STRIDED_DOT2_ACCF16})) {
    out << " nhwcGemmDot2TimesUs=" << config.nhwcGemmDot2F32TimeUs << "," << config.nhwcGemmDot2F16TimeUs;
  }
  if(config.hasKernelTuned(TUNED_GEMM_STRIDED_DOT2_ACCF16)) {
    out << " nhwcGemmDot2AccF16Tile=";
    appendTuple(
      {config.nhwcStridedDot2AccF16WorkgroupSize,
       config.nhwcStridedDot2AccF16BM,
       config.nhwcStridedDot2AccF16BN,
       config.nhwcStridedDot2AccF16SGM,
       config.nhwcStridedDot2AccF16SGN,
       config.nhwcStridedDot2AccF16SGMIter,
       config.nhwcStridedDot2AccF16TM,
       config.nhwcStridedDot2AccF16TN,
       config.nhwcStridedDot2AccF16SubgroupSize});
  }
  if(config.anyOfKernelTuned(
       {TUNED_CONV3X3_COOPMAT1, TUNED_CONV3X3_COOPMAT2, TUNED_CONV3X3_COOPMAT1_ACCF16,
        TUNED_CONV3X3_COOPMAT2_ACCF16})) {
    out << " nhwcConv3x3TimesUs=" << config.conv3x3NhwcCoopmat1TimeUs << "," << config.conv3x3NhwcCoopmat2TimeUs << ","
        << config.conv3x3NhwcCoopmat1AccF16TimeUs << "," << config.conv3x3NhwcCoopmat2AccF16TimeUs;
  }
  if(config.hasKernelTuned(TUNED_CONV3X3_COOPMAT2)) {
    out << " nhwcConv3x3Coopmat2Tile=";
    appendTuple(
      {config.conv3x3NhwcCoopmat2WorkgroupSize,
       config.conv3x3NhwcCoopmat2BM,
       config.conv3x3NhwcCoopmat2BN,
       config.conv3x3NhwcCoopmat2BK});
  }
  if(config.anyOfKernelTuned({TUNED_CONV5X5_COOPMAT1, TUNED_CONV5X5_COOPMAT2})) {
    out << " nhwcConv5x5TimesUs=" << config.conv5x5NhwcCoopmat1TimeUs << "," << config.conv5x5NhwcCoopmat2TimeUs;
  }
  if(config.hasKernelTuned(TUNED_CONV5X5_COOPMAT2)) {
    out << " nhwcConv5x5Coopmat2Tile=";
    appendTuple(
      {config.conv5x5NhwcCoopmat2WorkgroupSize,
       config.conv5x5NhwcCoopmat2BM,
       config.conv5x5NhwcCoopmat2BN,
       config.conv5x5NhwcCoopmat2BK});
  }

  bool hasTransformer = (modelDesc == nullptr) || modelHasTransformer(modelDesc);
  if(hasTransformer) {
    if(config.hasKernelTuned(TUNED_ATTN_TILED)) {
      out << " attnTiledNhwc=";
      appendTuple({config.attnNhwcBlockQ, config.attnNhwcBlockKV, config.attnNhwcQPerThread});
    }
    if(config.anyOfKernelTuned({TUNED_ATTN_COOPMAT1, TUNED_ATTN_COOPMAT2, TUNED_ATTN_MAINTENANCE1, TUNED_ATTN_DOT2})) {
      out << " attnNhwcTimesUs=" << config.attnNhwcCoopmat1TimeUs << "," << config.attnNhwcCoopmat2TimeUs << ","
          << config.attnNhwcCoopmatMaintenance1TimeUs << "," << config.attnNhwcDot2TimeUs;
    }
    if(config.hasKernelTuned(TUNED_ATTN_COOPMAT1)) {
      out << " attnNhwcCoopmat1Tile=";
      appendTuple(
        {config.attnNhwcCoopmat1WorkgroupSize,
         config.attnNhwcCoopmat1BlockQ,
         config.attnNhwcCoopmat1BlockKV,
         config.attnNhwcCoopmat1TM,
         config.attnNhwcCoopmat1TN,
         config.attnNhwcCoopmat1TK,
         config.attnNhwcCoopmat1SubgroupSize});
      out << " attnNhwcCoopmat1KV=" << (config.attnNhwcCoopmat1DirectKV != 0 ? "direct" : "staged");
    }
    if(config.hasKernelTuned(TUNED_ATTN_COOPMAT1_SPLITK)) {
      out << " attnNhwcCoopmat1SplitK=";
      appendTuple(
        {config.attnNhwcCoopmat1SplitKWorkgroupSize,
         config.attnNhwcCoopmat1SplitKBlockQ,
         config.attnNhwcCoopmat1SplitKBlockKV,
         config.attnNhwcCoopmat1SplitKTM,
         config.attnNhwcCoopmat1SplitKTN,
         config.attnNhwcCoopmat1SplitKTK,
         config.attnNhwcCoopmat1SplitKSubgroupSize});
      out << " KV=" << (config.attnNhwcCoopmat1SplitKDirectKV != 0 ? "direct" : "staged")
          << " chunks=" << config.attnNhwcCoopmat1SplitKKVChunkCount
          << " cutoff=" << config.attnNhwcCoopmat1SplitKCutoffBatch;
    }
    if(config.hasKernelTuned(TUNED_ATTN_MAINTENANCE1)) {
      out << " attnNhwcCoopmatMaintenance1Tile=";
      appendTuple(
        {config.attnNhwcCoopmatMaintenance1WorkgroupSize,
         config.attnNhwcCoopmatMaintenance1BlockQ,
         config.attnNhwcCoopmatMaintenance1BlockKV,
         config.attnNhwcCoopmatMaintenance1TM,
         config.attnNhwcCoopmatMaintenance1TN,
         config.attnNhwcCoopmatMaintenance1TK,
         config.attnNhwcCoopmatMaintenance1SubgroupSize});
      out << " attnNhwcCoopmatMaintenance1KV="
          << (config.attnNhwcCoopmatMaintenance1DirectKV != 0 ? "direct" : "staged");
    }
    if(config.hasKernelTuned(TUNED_ATTN_MAINTENANCE1_SPLITK)) {
      out << " attnNhwcCoopmatMaintenance1SplitK=";
      appendTuple(
        {config.attnNhwcCoopmatMaintenance1SplitKWorkgroupSize,
         config.attnNhwcCoopmatMaintenance1SplitKBlockQ,
         config.attnNhwcCoopmatMaintenance1SplitKBlockKV,
         config.attnNhwcCoopmatMaintenance1SplitKTM,
         config.attnNhwcCoopmatMaintenance1SplitKTN,
         config.attnNhwcCoopmatMaintenance1SplitKTK,
         config.attnNhwcCoopmatMaintenance1SplitKSubgroupSize});
      out << " KV=" << (config.attnNhwcCoopmatMaintenance1SplitKDirectKV != 0 ? "direct" : "staged")
          << " chunks=" << config.attnNhwcCoopmatMaintenance1SplitKKVChunkCount
          << " cutoff=" << config.attnNhwcCoopmatMaintenance1SplitKCutoffBatch;
    }
    if(config.hasKernelTuned(TUNED_ATTN_COOPMAT2)) {
      out << " attnNhwcCoopmat2Tile=";
      appendTuple(
        {config.attnNhwcCoopmat2WorkgroupSize, config.attnNhwcCoopmat2BlockQ, config.attnNhwcCoopmat2BlockKV});
    }
    if(config.hasKernelTuned(TUNED_ATTN_DOT2)) {
      out << " attnNhwcDot2Tile=";
      appendTuple({config.attnNhwcDot2WorkgroupSize, config.attnNhwcDot2BlockQ, config.attnNhwcDot2BlockKV});
    }
    if(config.hasKernelTuned(TUNED_SPATIAL_RMSNORM_NHWC)) {
      out << " spatialRMSNormNhwc=" << config.spatialRMSNormNhwcTile;
    }
    if(config.hasKernelTuned(TUNED_SWIGLU)) {
      out << " swiGLU=" << config.swiGLULocalSizeX;
    }
  }
}

// Command-line entry point for `katago tuner` (Vulkan backend).
// command/tune.cpp forwards here so all Vulkan tuning code lives in one place.
// ============================================================================

int VulkanTuner::runTuneCommand(const vector<string>& args) {
  ConfigParser cfg;
  string modelFile;
  string outputFileFromArg;
  string gpuIdxsStr;
  vector<int> gpuIdxs;
  int nnXLen;
  int nnYLen;
  int batchSize;
  int winograd3x3OutTile;
  int benchIters;
  bool full;
  bool verboseTuner;
  bool batchSizeWasExplicit = false;
  bool batchSizeFromNumSearchThreads = false;
  try {
    KataGoCommandLine cmd("Perform GPU tuning for Vulkan.");
    cmd.addConfigFileArg(KataGoCommandLine::defaultGtpConfigFileName(), "gtp_example.cfg");
    cmd.addModelFileArg();

    TCLAP::ValueArg<string> outputFileArg(
      "", "output", "Filename to output tuning configuration to", false, string(), "FILE");
    TCLAP::ValueArg<string> gpuIdxsArg(
      "", "gpus", "Specific GPU/device number(s) to tune, comma-separated (default all)", false, string(), "GPUS");
    TCLAP::ValueArg<int> nnXLenArg(
      "", "xsize", "Width of board to tune for", false, VulkanTuner::DEFAULT_X_SIZE, "INT");
    TCLAP::ValueArg<int> nnYLenArg(
      "", "ysize", "Height of board to tune for", false, VulkanTuner::DEFAULT_Y_SIZE, "INT");
    TCLAP::ValueArg<int> batchSizeArg(
      "", "batchsize", "Batch size to tune for", false, VulkanTuner::DEFAULT_BATCH_SIZE, "INT");
    TCLAP::ValueArg<int> winograd3x3TileSizeArg(
      "",
      "winograd3x3tilesize",
      "3x3 Winograd output tile size to tune for: 2 for F(2,3), 4 for F(4,3)",
      false,
      VulkanTuner::DEFAULT_WINOGRAD_3X3_TILE_SIZE,
      "2_OR_4");
    TCLAP::ValueArg<int> benchItersArg(
      "",
      "benchiters",
      "Number of dispatch iterations per micro-benchmark candidate (default 500; reduce for faster but noisier tuning)",
      false,
      500,
      "INT");
    TCLAP::SwitchArg fullArg("", "full", "Tune all model-, hardware-, and filter-applicable Vulkan kernel families");
    TCLAP::SwitchArg verboseTunerArg(
      "", "verboseTuner", "Verbosely print out tuner results even if they don't improve the best");

    cmd.setShortUsageArgLimit();
    cmd.addOverrideConfigArg();

    cmd.add(outputFileArg);
    cmd.add(gpuIdxsArg);
    cmd.add(nnXLenArg);
    cmd.add(nnYLenArg);
    cmd.add(batchSizeArg);
    cmd.add(winograd3x3TileSizeArg);
    cmd.add(benchItersArg);
    cmd.add(fullArg);
    cmd.add(verboseTunerArg);
    cmd.parseArgs(args);

    modelFile = cmd.getModelFile();
    outputFileFromArg = outputFileArg.getValue();
    gpuIdxsStr = gpuIdxsArg.getValue();
    nnXLen = nnXLenArg.getValue();
    nnYLen = nnYLenArg.getValue();
    winograd3x3OutTile = winograd3x3TileSizeArg.getValue();
    benchIters = benchItersArg.getValue();
    full = fullArg.getValue();
    verboseTuner = verboseTunerArg.getValue();

    if(gpuIdxsStr.size() > 0) {
      vector<string> pieces = Global::split(gpuIdxsStr, ',');
      int parsed;
      for(size_t i = 0; i < pieces.size(); i++) {
        if(!Global::tryStringToInt(Global::trim(pieces[i]), parsed) || parsed < 0 || contains(gpuIdxs, parsed)) {
          cerr << "Error: bad/duplicate -gpus value: " << pieces[i] << endl;
          return 1;
        }
        gpuIdxs.push_back(parsed);
      }
    }
    if(winograd3x3OutTile != 2 && winograd3x3OutTile != 4) {
      cerr << "Error: --winograd3x3tilesize must be 2 or 4" << endl;
      return 1;
    }
    cmd.getConfigAllowEmpty(cfg);

    if(batchSizeArg.isSet()) {
      batchSize = batchSizeArg.getValue();
      batchSizeWasExplicit = true;
    } else if(cfg.contains("numSearchThreads")) {
      // Search batches typically range from 1 up to about numSearchThreads,
      // so tune at the upper end of the expected runtime range.
      batchSize = std::max(1, cfg.getInt("numSearchThreads", 1, 65536));
      batchSizeFromNumSearchThreads = true;
    } else
      batchSize = VulkanTuner::DEFAULT_BATCH_SIZE;
  } catch(TCLAP::ArgException& e) {
    cerr << "Error: " << e.error() << " for argument " << e.argId() << endl;
    return 1;
  }

  string homeDataDirOverride = Setup::loadHomeDataDirOverride(cfg);
  const bool logToStdoutDefault = true;
  Logger logger(&cfg, logToStdoutDefault);

  logger.write("Loading model...");
  ModelDesc modelDesc;
  string expectedSha256 = "";
  ModelDesc::loadFromFileMaybeGZipped(modelFile, modelDesc, expectedSha256);

  // --output specifies a single file. Refuse to silently overwrite it from
  // multiple devices' sweeps.
  if(!outputFileFromArg.empty() && gpuIdxs.size() > 1) {
    cerr << "Error: --output is incompatible with --gpus listing more than one device "
            "(the same file would be overwritten by each sweep). Run the tuner per-GPU "
            "or omit --output to use the default per-(GPU, board, model) cache path."
         << endl;
    return 1;
  }

  // Empty list → let VulkanDevicesContext resolve to the system's default GPU.
  if(gpuIdxs.empty())
    gpuIdxs.push_back(-1);

  logger.write("Tuner starting...");
  logger.write("Tuner: using 3x3 Winograd output tile size " + Global::intToString(winograd3x3OutTile));
  if(batchSizeWasExplicit)
    logger.write("Tuner: using explicit batchsize from CLI: " + Global::intToString(batchSize));
  else if(batchSizeFromNumSearchThreads)
    logger.write("Tuner: batchsize not provided; using numSearchThreads = " + Global::intToString(batchSize));
  else
    logger.write("Tuner: batchsize not provided; using default batchsize " + Global::intToString(batchSize));

  // One ComputeContext for the whole sweep. tuneSelf iterates every owned
  // device and dedups by name, so multi-GPU passes don't need a host-side loop.
  ComputeContext* ctx = VulkanTuner::createComputeContextForVulkanTuner(
    gpuIdxs, &logger, nnXLen, nnYLen, homeDataDirOverride, enabled_t::Auto);
  NeuralNet::tuneVulkanComputeContext(
    ctx, &modelDesc, batchSize, winograd3x3OutTile, benchIters, verboseTuner, full, outputFileFromArg);
  NeuralNet::freeComputeContext(ctx);
  return 0;
}

#endif  // USE_VULKAN_BACKEND

#ifdef USE_VULKAN_BACKEND

#include "../neuralnet/vulkantuner.h"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <initializer_list>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <string_view>
#include "../command/commandline.h"
#include "../core/fileutils.h"
#include "../core/makedir.h"
#include "../dataio/homedata.h"
#include "../neuralnet/activations.h"
#include "../neuralnet/desc.h"
#include "../neuralnet/vulkanbackend.h"
#include "../neuralnet/vulkanhelpers.h"
#include "../neuralnet/vulkankernels.h"
#include "../program/setup.h"
#include "vulkanshaders_generated.h"

using namespace std;
using namespace VulkanHelpers;

// ============================================================================
// TU-local helpers
// ============================================================================

namespace {

  string_view getenvStringView(const char* name) {
    const char* value = std::getenv(name);
    return value == nullptr ? string_view() : string_view(value, std::strlen(value));
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

  // Keep in lockstep with VULKAN_TUNEPARAMS_VERSION_LINE: the cache filename
  // encodes this number (tune<N>_...), and the file content declares the same
  // VERSION=<N> line.
  constexpr int VULKAN_TUNER_VERSION = 1;
  constexpr string_view VULKAN_TUNEPARAMS_VERSION_LINE = "VERSION=1";

  // Time-budget for all kernel bench runs.
  // Probe with a few iterations to estimate per-dispatch cost, then clamp the
  // timed run so each candidate takes at most BENCH_TARGET_SECONDS wall-clock
  // time. On a fast discrete GPU the ceiling (iters) is reached; on a slow iGPU
  // only a handful of iterations run, keeping auto-tune startup manageable.
  constexpr double BENCH_TARGET_SECONDS = 1.0;
  constexpr double BENCH_WARMUP_SECONDS = 0.050;
  constexpr int BENCH_PROBE_ITERS = 5;
  constexpr int BENCH_WARMUP_MAX_ITERS = 2048;
  constexpr int GEMM_VARIANT_SELECTION_RANK_MAX = 8;

  struct GemmVariantSelectionField {
    int32_t VulkanTuneParams::* enabled;
    int32_t VulkanTuneParams::* valueValid;
    int32_t VulkanTuneParams::* selectionRank;
    bool usable = true;
  };

  bool gemmVariantSelectionRankValid(int32_t rank) {
    return rank >= 1 && rank <= GEMM_VARIANT_SELECTION_RANK_MAX;
  }

  bool disableTiledFp16ComputeGemmVariants(VulkanTuneParams& config) {
    bool changed = false;
    auto setField = [&](int32_t& field, int32_t value) {
      if(field != value) {
        field = value;
        changed = true;
      }
    };
    setField(config.enableWinogradGemmTiledFp16Compute, 0);
    setField(config.enableGemmStridedTiledFp16Compute, 0);
    setField(config.winogradGemmTiledFp16ComputeTunerValueValid, 0);
    setField(config.gemmStridedTiledFp16ComputeTunerValueValid, 0);
    setField(config.winogradGemmTiledFp16ComputeSelectionRank, GEMM_VARIANT_SELECTION_RANK_MAX);
    setField(config.gemmStridedTiledFp16ComputeSelectionRank, GEMM_VARIANT_SELECTION_RANK_MAX);
    return changed;
  }

  std::vector<int32_t VulkanTuneParams::*> sortedGemmVariantSelectionRanks(
    const VulkanTuneParams& config,
    std::initializer_list<int32_t VulkanTuneParams::*> ranks) {
    std::vector<int32_t VulkanTuneParams::*> order(ranks.begin(), ranks.end());
    std::stable_sort(order.begin(), order.end(), [&](auto a, auto b) {
      int32_t rankA = gemmVariantSelectionRankValid(config.*a) ? config.*a : GEMM_VARIANT_SELECTION_RANK_MAX + 1;
      int32_t rankB = gemmVariantSelectionRankValid(config.*b) ? config.*b : GEMM_VARIANT_SELECTION_RANK_MAX + 1;
      return rankA < rankB;
    });
    return order;
  }

  void assignGemmVariantSelectionRanks(
    VulkanTuneParams& config,
    const std::vector<int32_t VulkanTuneParams::*>& order) {
    for(size_t i = 0; i < order.size(); i++)
      config.*order[i] = (int32_t)i + 1;
  }

  void moveGemmVariantSelectionRankToFront(
    VulkanTuneParams& config,
    std::initializer_list<int32_t VulkanTuneParams::*> ranks,
    int32_t VulkanTuneParams::* selectedRank) {
    std::vector<int32_t VulkanTuneParams::*> order = sortedGemmVariantSelectionRanks(config, ranks);
    auto selectedIt = std::find(order.begin(), order.end(), selectedRank);
    if(selectedIt == order.end())
      return;
    int32_t VulkanTuneParams::* selected = *selectedIt;
    order.erase(selectedIt);
    order.insert(order.begin(), selected);
    assignGemmVariantSelectionRanks(config, order);
  }

  void preferGemmVariantSelectionRank(
    VulkanTuneParams& config,
    std::initializer_list<int32_t VulkanTuneParams::*> ranks,
    int32_t VulkanTuneParams::* preferredRank,
    int32_t VulkanTuneParams::* otherRank) {
    std::vector<int32_t VulkanTuneParams::*> order = sortedGemmVariantSelectionRanks(config, ranks);
    auto preferredIt = std::find(order.begin(), order.end(), preferredRank);
    auto otherIt = std::find(order.begin(), order.end(), otherRank);
    if(preferredIt == order.end() || otherIt == order.end() || preferredIt == otherIt)
      return;
    if(preferredIt > otherIt) {
      int32_t VulkanTuneParams::* preferred = *preferredIt;
      order.erase(preferredIt);
      otherIt = std::find(order.begin(), order.end(), otherRank);
      order.insert(otherIt, preferred);
    }
    assignGemmVariantSelectionRanks(config, order);
  }

  void promoteWinogradGemmSelectionRank(VulkanTuneParams& config, int32_t VulkanTuneParams::* selectedRank) {
    moveGemmVariantSelectionRankToFront(
      config,
      {&VulkanTuneParams::winogradGemmTiledSelectionRank,
       &VulkanTuneParams::winogradGemmCoopmat2AccF16SelectionRank,
       &VulkanTuneParams::winogradGemmCoopmat2SelectionRank,
       &VulkanTuneParams::winogradGemmCoopmatAccF16SelectionRank,
       &VulkanTuneParams::winogradGemmCoopmatSelectionRank,
       &VulkanTuneParams::winogradGemmDot2AccF16SelectionRank,
       &VulkanTuneParams::winogradGemmDot2SelectionRank},
      selectedRank);
  }

  void promoteGemmStridedSelectionRank(VulkanTuneParams& config, int32_t VulkanTuneParams::* selectedRank) {
    moveGemmVariantSelectionRankToFront(
      config,
      {&VulkanTuneParams::gemmStridedTiledSelectionRank,
       &VulkanTuneParams::gemmStridedCoopmat2AccF16SelectionRank,
       &VulkanTuneParams::gemmStridedCoopmat2SelectionRank,
       &VulkanTuneParams::gemmStridedCoopmatAccF16SelectionRank,
       &VulkanTuneParams::gemmStridedCoopmatSelectionRank,
       &VulkanTuneParams::gemmStridedDot2AccF16SelectionRank,
       &VulkanTuneParams::gemmStridedDot2SelectionRank},
      selectedRank);
  }

  void preferWinogradGemmSelectionRank(
    VulkanTuneParams& config,
    int32_t VulkanTuneParams::* preferredRank,
    int32_t VulkanTuneParams::* otherRank) {
    preferGemmVariantSelectionRank(
      config,
      {&VulkanTuneParams::winogradGemmTiledSelectionRank,
       &VulkanTuneParams::winogradGemmCoopmat2AccF16SelectionRank,
       &VulkanTuneParams::winogradGemmCoopmat2SelectionRank,
       &VulkanTuneParams::winogradGemmCoopmatAccF16SelectionRank,
       &VulkanTuneParams::winogradGemmCoopmatSelectionRank,
       &VulkanTuneParams::winogradGemmDot2AccF16SelectionRank,
       &VulkanTuneParams::winogradGemmDot2SelectionRank},
      preferredRank,
      otherRank);
  }

  void preferGemmStridedSelectionRank(
    VulkanTuneParams& config,
    int32_t VulkanTuneParams::* preferredRank,
    int32_t VulkanTuneParams::* otherRank) {
    preferGemmVariantSelectionRank(
      config,
      {&VulkanTuneParams::gemmStridedTiledSelectionRank,
       &VulkanTuneParams::gemmStridedCoopmat2AccF16SelectionRank,
       &VulkanTuneParams::gemmStridedCoopmat2SelectionRank,
       &VulkanTuneParams::gemmStridedCoopmatAccF16SelectionRank,
       &VulkanTuneParams::gemmStridedCoopmatSelectionRank,
       &VulkanTuneParams::gemmStridedDot2AccF16SelectionRank,
       &VulkanTuneParams::gemmStridedDot2SelectionRank},
      preferredRank,
      otherRank);
  }

  bool selectGemmVariantByRank(VulkanTuneParams& config, std::initializer_list<GemmVariantSelectionField> fields) {
    int32_t VulkanTuneParams::* selectedEnabled = nullptr;
    int32_t bestRank = GEMM_VARIANT_SELECTION_RANK_MAX + 1;
    for(const GemmVariantSelectionField& field: fields) {
      const int32_t rank = config.*field.selectionRank;
      if(
        field.usable && (field.valueValid == nullptr || (config.*field.valueValid) != 0) &&
        gemmVariantSelectionRankValid(rank) && rank < bestRank) {
        selectedEnabled = field.enabled;
        bestRank = rank;
      }
    }

    bool changed = false;
    for(const GemmVariantSelectionField& field: fields) {
      if(field.enabled == nullptr)
        continue;
      const int32_t normalized = field.enabled == selectedEnabled ? 1 : 0;
      if(config.*field.enabled != normalized) {
        config.*field.enabled = normalized;
        changed = true;
      }
    }
    return changed;
  }

  static constexpr string_view REQUIRED_VULKAN_FP16_FEATURES_AVAILABLE_REASON =
    "required Vulkan features are available";

  static string_view
  missingVulkanFp16BaseReason(const VulkanDeviceInfo* info, bool fp16StorageEnabled, bool fp16ComputeEnabled) {
    if(info == nullptr)
      return "no Vulkan device";
    const bool deviceSupportsFp16Storage = info->detectedSupportsFP16Storage || info->supportsFP16Storage;
    const bool deviceSupportsFp16Compute = info->detectedSupportsFP16Compute || info->supportsFP16Compute;
    if(!deviceSupportsFp16Storage && !deviceSupportsFp16Compute)
      return "device lacks both fp16 storage and shaderFloat16";
    if(!deviceSupportsFp16Storage)
      return "device does not support fp16 storage";
    if(!deviceSupportsFp16Compute)
      return "device does not support shaderFloat16";
    if(!fp16StorageEnabled && !fp16ComputeEnabled)
      return "fp16 storage and shaderFloat16 are not enabled for this run";
    if(!fp16StorageEnabled)
      return "fp16 storage is not enabled for this run";
    if(!fp16ComputeEnabled)
      return "shaderFloat16 is not enabled for this run";
    return REQUIRED_VULKAN_FP16_FEATURES_AVAILABLE_REASON;
  }

  static void appendMissingVulkanFp16FeatureReason(
    std::ostream& out,
    string_view featureName,
    const VulkanDeviceInfo* info,
    bool featureEnabled,
    bool featureUnfilteredSupported,
    bool featureFilterDisabled,
    bool fp16StorageEnabled,
    bool fp16ComputeEnabled) {
    string_view baseReason = missingVulkanFp16BaseReason(info, fp16StorageEnabled, fp16ComputeEnabled);
    if(baseReason != REQUIRED_VULKAN_FP16_FEATURES_AVAILABLE_REASON) {
      out << baseReason;
      return;
    }
    if(featureFilterDisabled) {
      out << featureName << " disabled by Vulkan GEMM variant filter";
      return;
    }
    if(!featureUnfilteredSupported) {
      out << "device does not support " << featureName;
      return;
    }
    if(!featureEnabled) {
      out << featureName << " is not enabled for this run";
      return;
    }
    out << baseReason;
  }

  // Run a time-budgeted bench: first do an untimed, capped time-based warmup
  // so fast kernels get enough sustained work to settle clocks, then probe
  // BENCH_PROBE_ITERS dispatches to estimate per-iteration cost and time the
  // clamped run. Returns {totalSeconds, effectiveIters}, or {-1, 0} on failure.
  struct BenchResult {
    double seconds;
    int iters;
  };
  inline BenchResult
  timeBudgetedBench(VulkanTuner::TuningContext& s, int maxIters, const std::function<void()>& recordOne) {
    s.warmupDispatches(BENCH_WARMUP_SECONDS, BENCH_WARMUP_MAX_ITERS, recordOne);
    double probeSeconds = s.timeDispatches(BENCH_PROBE_ITERS, recordOne);
    if(probeSeconds <= 0.0)
      return {-1.0, 0};
    int budgetIters = (int)(BENCH_TARGET_SECONDS / (probeSeconds / BENCH_PROBE_ITERS));
    int effectiveIters = std::clamp(budgetIters, BENCH_PROBE_ITERS, maxIters);
    double seconds = s.timeDispatches(effectiveIters, recordOne);
    return {seconds, effectiveIters};
  }

  struct WinogradTransformBenchLayout {
    int mAlignment;
    int nAlignment;
    int kAlignment;
    int packedBM;
    int packedBK;
    int packedAPadWords;
  };

  inline LayerPaddingContract selectedWinogradBenchContract(const VulkanTuneParams& cfg, bool fp16Storage) {
    if(fp16Storage && cfg.enableWinogradGemmCoopmat2AccF16 != 0)
      return VulkanKernels::WinogradGemmCoopmat2AccF16::layerPaddingContract(cfg);
    if(fp16Storage && cfg.enableWinogradGemmCoopmat2 != 0)
      return VulkanKernels::WinogradGemmCoopmat2::layerPaddingContract(cfg);
    if(fp16Storage && cfg.enableWinogradGemmCoopmatAccF16 != 0)
      return VulkanKernels::WinogradGemmCoopmatAccF16::layerPaddingContract(cfg);
    if(fp16Storage && cfg.enableWinogradGemmCoopmat != 0)
      return VulkanKernels::WinogradGemmCoopmat::layerPaddingContract(cfg);
    if(fp16Storage && cfg.enableWinogradGemmDot2AccF16 != 0)
      return VulkanKernels::WinogradGemmDot2AccF16::layerPaddingContract(cfg);
    if(fp16Storage && cfg.enableWinogradGemmDot2 != 0)
      return VulkanKernels::WinogradGemmDot2::layerPaddingContract(cfg);
    return VulkanKernels::WinogradGemm::layerPaddingContract(cfg);
  }

  inline WinogradTransformBenchLayout makeWinogradTransformBenchLayout(const VulkanTuneParams& cfg, bool fp16Storage) {
    WinogradTransformBenchLayout layout;
    LayerPaddingContract contract = selectedWinogradBenchContract(cfg, fp16Storage);
    layout.mAlignment = std::max(1, contract.m);
    layout.nAlignment = std::max(1, contract.n);
    layout.kAlignment = std::max(1, contract.kPaddable ? contract.k : 1);
    layout.packedBM = std::max(1, (int)cfg.winogradGemmM);
    layout.packedBK = std::max(1, (int)cfg.winogradGemmK);
    layout.packedAPadWords = WINOGRAD_ROW_MAJOR_A_PAD_WORDS;
    if(fp16Storage && cfg.enableWinogradGemmCoopmat2AccF16 != 0) {
      layout.packedBM = std::max(1, (int)cfg.coopmat2AccF16BM);
      layout.packedBK = std::max(1, (int)cfg.coopmat2AccF16BK);
      layout.packedAPadWords = WINOGRAD_COOPMAT2_PACKED_PAD_WORDS;
    } else if(fp16Storage && cfg.enableWinogradGemmCoopmat2 != 0) {
      layout.packedBM = std::max(1, (int)cfg.coopmat2BM);
      layout.packedBK = std::max(1, (int)cfg.coopmat2BK);
      layout.packedAPadWords = WINOGRAD_COOPMAT2_PACKED_PAD_WORDS;
    } else if(fp16Storage && cfg.enableWinogradGemmCoopmatAccF16 != 0) {
      layout.packedBM = std::max(1, (int)cfg.coopmatAccF16BM);
      layout.packedBK = std::max(1, (int)cfg.coopmatAccF16BK);
      layout.packedAPadWords = WINOGRAD_COOPMAT_PACKED_PAD_WORDS;
    } else if(fp16Storage && cfg.enableWinogradGemmCoopmat != 0) {
      layout.packedBM = std::max(1, (int)cfg.coopmatBM);
      layout.packedBK = std::max(1, (int)cfg.coopmatBK);
      layout.packedAPadWords = WINOGRAD_COOPMAT_PACKED_PAD_WORDS;
    } else if(fp16Storage && cfg.enableWinogradGemmDot2AccF16 != 0) {
      layout.packedBM = std::max(1, (int)cfg.dot2AccF16BM);
      layout.packedBK = VulkanKernels::DOT2_BK;
    } else if(fp16Storage && cfg.enableWinogradGemmDot2 != 0) {
      layout.packedBM = std::max(1, (int)cfg.dot2BM);
      layout.packedBK = VulkanKernels::DOT2_BK;
    }
    return layout;
  }

}  // namespace

std::string VulkanTuner::winogradTransformLayoutSignature(const VulkanTuneParams& config, bool fp16Storage) {
  WinogradTransformBenchLayout layout = makeWinogradTransformBenchLayout(config, fp16Storage);
  std::ostringstream out;
  out << (fp16Storage ? 1 : 0) << "-" << layout.mAlignment << "-" << layout.kAlignment << "-" << layout.packedBM << "-"
      << layout.packedBK << "-" << layout.packedAPadWords;
  return out.str();
}

bool VulkanTuner::winogradTransformsTunedForCurrentLayout(const VulkanTuneParams& config, bool fp16Storage) {
  return config.winogradTransformTunedLayout == winogradTransformLayoutSignature(config, fp16Storage);
}

void VulkanTuner::markWinogradTransformsTunedForCurrentLayout(VulkanTuneParams& config, bool fp16Storage) {
  config.winogradTransformTunedLayout = winogradTransformLayoutSignature(config, fp16Storage);
}

bool VulkanTuner::gemmVariantFilterTunedForCurrentHardware(
  const VulkanTuneParams& config,
  const VulkanDeviceInfo& deviceInfo) {
  return config.gemmVariantFilterTunedSignature == deviceInfo.gemmVariantFilterSignature;
}

// ============================================================================
// VulkanTuneParams persistence
// ============================================================================

bool VulkanTuneParams::operator==(const VulkanTuneParams& other) const {
  return winogradGemmM == other.winogradGemmM && winogradGemmN == other.winogradGemmN &&
         winogradGemmK == other.winogradGemmK && winogradGemmRN == other.winogradGemmRN &&
         winogradGemmTiledSelectionRank == other.winogradGemmTiledSelectionRank &&
         enableWinogradGemmTiledFp16Compute == other.enableWinogradGemmTiledFp16Compute &&
         enableGemmStridedTiledFp16Compute == other.enableGemmStridedTiledFp16Compute &&
         winogradGemmTiledFp16ComputeTunerValueValid == other.winogradGemmTiledFp16ComputeTunerValueValid &&
         gemmStridedTiledFp16ComputeTunerValueValid == other.gemmStridedTiledFp16ComputeTunerValueValid &&
         winogradGemmTiledFp16ComputeSelectionRank == other.winogradGemmTiledFp16ComputeSelectionRank &&
         gemmStridedTiledFp16ComputeSelectionRank == other.gemmStridedTiledFp16ComputeSelectionRank &&
         winogradGemmFp16ComputeM == other.winogradGemmFp16ComputeM &&
         winogradGemmFp16ComputeN == other.winogradGemmFp16ComputeN &&
         winogradGemmFp16ComputeK == other.winogradGemmFp16ComputeK &&
         winogradGemmFp16ComputeRN == other.winogradGemmFp16ComputeRN &&
         winograd3x3OutTile == other.winograd3x3OutTile && attnBlockQ == other.attnBlockQ &&
         attnBlockKV == other.attnBlockKV && attnQPerThread == other.attnQPerThread &&
         winogradTransformLocalSizeX == other.winogradTransformLocalSizeX &&
         winogradTransformLocalSizeY == other.winogradTransformLocalSizeY &&
         winogradTransformTunedLayout == other.winogradTransformTunedLayout &&
         gemmVariantFilterTunedSignature == other.gemmVariantFilterTunedSignature &&
         winogradBNActTransformLocalSizeX == other.winogradBNActTransformLocalSizeX &&
         winogradBNActTransformLocalSizeY == other.winogradBNActTransformLocalSizeY &&
         winogradUntransformLocalSizeX == other.winogradUntransformLocalSizeX &&
         winogradUntransformLocalSizeY == other.winogradUntransformLocalSizeY &&
         gemmStridedTiledLocalSizeX == other.gemmStridedTiledLocalSizeX &&
         gemmStridedTiledLocalSizeY == other.gemmStridedTiledLocalSizeY &&
         gemmStridedTiledTileK == other.gemmStridedTiledTileK && gemmStridedTiledRN == other.gemmStridedTiledRN &&
         gemmStridedTiledSelectionRank == other.gemmStridedTiledSelectionRank &&
         gemmStridedFp16ComputeLocalSizeX == other.gemmStridedFp16ComputeLocalSizeX &&
         gemmStridedFp16ComputeLocalSizeY == other.gemmStridedFp16ComputeLocalSizeY &&
         gemmStridedFp16ComputeTileK == other.gemmStridedFp16ComputeTileK &&
         gemmStridedFp16ComputeRN == other.gemmStridedFp16ComputeRN &&
         gemmDirectLocalSizeX == other.gemmDirectLocalSizeX && gemmDirectLocalSizeY == other.gemmDirectLocalSizeY &&
         gpoolXystride == other.gpoolXystride && valueHeadPoolXystride == other.valueHeadPoolXystride &&
         spatialRMSNormTile == other.spatialRMSNormTile && enableWinogradGemmDot2 == other.enableWinogradGemmDot2 &&
         enableGemmStridedDot2 == other.enableGemmStridedDot2 &&
         enableWinogradGemmDot2AccF16 == other.enableWinogradGemmDot2AccF16 &&
         enableGemmStridedDot2AccF16 == other.enableGemmStridedDot2AccF16 &&
         winogradGemmDot2TunerValueValid == other.winogradGemmDot2TunerValueValid &&
         gemmStridedDot2TunerValueValid == other.gemmStridedDot2TunerValueValid &&
         winogradGemmDot2AccF16TunerValueValid == other.winogradGemmDot2AccF16TunerValueValid &&
         gemmStridedDot2AccF16TunerValueValid == other.gemmStridedDot2AccF16TunerValueValid &&
         winogradGemmDot2SelectionRank == other.winogradGemmDot2SelectionRank &&
         gemmStridedDot2SelectionRank == other.gemmStridedDot2SelectionRank &&
         winogradGemmDot2AccF16SelectionRank == other.winogradGemmDot2AccF16SelectionRank &&
         gemmStridedDot2AccF16SelectionRank == other.gemmStridedDot2AccF16SelectionRank &&
         dot2BlockSize == other.dot2BlockSize && dot2BM == other.dot2BM && dot2BN == other.dot2BN &&
         dot2WM == other.dot2WM && dot2WN == other.dot2WN && dot2WMIter == other.dot2WMIter && dot2TM == other.dot2TM &&
         dot2TN == other.dot2TN && dot2Warp == other.dot2Warp && stridedDot2BlockSize == other.stridedDot2BlockSize &&
         stridedDot2BM == other.stridedDot2BM && stridedDot2BN == other.stridedDot2BN &&
         stridedDot2WM == other.stridedDot2WM && stridedDot2WN == other.stridedDot2WN &&
         stridedDot2WMIter == other.stridedDot2WMIter && stridedDot2TM == other.stridedDot2TM &&
         stridedDot2TN == other.stridedDot2TN && stridedDot2Warp == other.stridedDot2Warp &&
         dot2AccF16BlockSize == other.dot2AccF16BlockSize && dot2AccF16BM == other.dot2AccF16BM &&
         dot2AccF16BN == other.dot2AccF16BN && dot2AccF16WM == other.dot2AccF16WM &&
         dot2AccF16WN == other.dot2AccF16WN && dot2AccF16WMIter == other.dot2AccF16WMIter &&
         dot2AccF16TM == other.dot2AccF16TM && dot2AccF16TN == other.dot2AccF16TN &&
         dot2AccF16Warp == other.dot2AccF16Warp && stridedDot2AccF16BlockSize == other.stridedDot2AccF16BlockSize &&
         stridedDot2AccF16BM == other.stridedDot2AccF16BM && stridedDot2AccF16BN == other.stridedDot2AccF16BN &&
         stridedDot2AccF16WM == other.stridedDot2AccF16WM && stridedDot2AccF16WN == other.stridedDot2AccF16WN &&
         stridedDot2AccF16WMIter == other.stridedDot2AccF16WMIter && stridedDot2AccF16TM == other.stridedDot2AccF16TM &&
         stridedDot2AccF16TN == other.stridedDot2AccF16TN && stridedDot2AccF16Warp == other.stridedDot2AccF16Warp &&
         enableWinogradGemmCoopmat == other.enableWinogradGemmCoopmat &&
         enableGemmStridedCoopmat == other.enableGemmStridedCoopmat &&
         winogradGemmCoopmatTunerValueValid == other.winogradGemmCoopmatTunerValueValid &&
         gemmStridedCoopmatTunerValueValid == other.gemmStridedCoopmatTunerValueValid &&
         winogradGemmCoopmatSelectionRank == other.winogradGemmCoopmatSelectionRank &&
         gemmStridedCoopmatSelectionRank == other.gemmStridedCoopmatSelectionRank &&
         coopmatBlockSize == other.coopmatBlockSize && coopmatBM == other.coopmatBM && coopmatBN == other.coopmatBN &&
         coopmatBK == other.coopmatBK && coopmatWM == other.coopmatWM && coopmatWN == other.coopmatWN &&
         coopmatTM == other.coopmatTM && coopmatTN == other.coopmatTN && coopmatTK == other.coopmatTK &&
         coopmatWarp == other.coopmatWarp && stridedCoopmatBlockSize == other.stridedCoopmatBlockSize &&
         stridedCoopmatBM == other.stridedCoopmatBM && stridedCoopmatBN == other.stridedCoopmatBN &&
         stridedCoopmatBK == other.stridedCoopmatBK && stridedCoopmatWM == other.stridedCoopmatWM &&
         stridedCoopmatWN == other.stridedCoopmatWN && stridedCoopmatTM == other.stridedCoopmatTM &&
         stridedCoopmatTN == other.stridedCoopmatTN && stridedCoopmatTK == other.stridedCoopmatTK &&
         stridedCoopmatWarp == other.stridedCoopmatWarp &&
         enableWinogradGemmCoopmatAccF16 == other.enableWinogradGemmCoopmatAccF16 &&
         enableGemmStridedCoopmatAccF16 == other.enableGemmStridedCoopmatAccF16 &&
         winogradGemmCoopmatAccF16TunerValueValid == other.winogradGemmCoopmatAccF16TunerValueValid &&
         gemmStridedCoopmatAccF16TunerValueValid == other.gemmStridedCoopmatAccF16TunerValueValid &&
         winogradGemmCoopmatAccF16SelectionRank == other.winogradGemmCoopmatAccF16SelectionRank &&
         gemmStridedCoopmatAccF16SelectionRank == other.gemmStridedCoopmatAccF16SelectionRank &&
         coopmatAccF16BlockSize == other.coopmatAccF16BlockSize && coopmatAccF16BM == other.coopmatAccF16BM &&
         coopmatAccF16BN == other.coopmatAccF16BN && coopmatAccF16BK == other.coopmatAccF16BK &&
         coopmatAccF16WM == other.coopmatAccF16WM && coopmatAccF16WN == other.coopmatAccF16WN &&
         coopmatAccF16TM == other.coopmatAccF16TM && coopmatAccF16TN == other.coopmatAccF16TN &&
         coopmatAccF16TK == other.coopmatAccF16TK && coopmatAccF16Warp == other.coopmatAccF16Warp &&
         stridedCoopmatAccF16BlockSize == other.stridedCoopmatAccF16BlockSize &&
         stridedCoopmatAccF16BM == other.stridedCoopmatAccF16BM &&
         stridedCoopmatAccF16BN == other.stridedCoopmatAccF16BN &&
         stridedCoopmatAccF16BK == other.stridedCoopmatAccF16BK &&
         stridedCoopmatAccF16WM == other.stridedCoopmatAccF16WM &&
         stridedCoopmatAccF16WN == other.stridedCoopmatAccF16WN &&
         stridedCoopmatAccF16TM == other.stridedCoopmatAccF16TM &&
         stridedCoopmatAccF16TN == other.stridedCoopmatAccF16TN &&
         stridedCoopmatAccF16TK == other.stridedCoopmatAccF16TK &&
         stridedCoopmatAccF16Warp == other.stridedCoopmatAccF16Warp &&
         enableWinogradGemmCoopmat2 == other.enableWinogradGemmCoopmat2 &&
         enableGemmStridedCoopmat2 == other.enableGemmStridedCoopmat2 &&
         winogradGemmCoopmat2TunerValueValid == other.winogradGemmCoopmat2TunerValueValid &&
         gemmStridedCoopmat2TunerValueValid == other.gemmStridedCoopmat2TunerValueValid &&
         winogradGemmCoopmat2SelectionRank == other.winogradGemmCoopmat2SelectionRank &&
         gemmStridedCoopmat2SelectionRank == other.gemmStridedCoopmat2SelectionRank &&
         coopmat2BlockSize == other.coopmat2BlockSize && coopmat2BM == other.coopmat2BM &&
         coopmat2BN == other.coopmat2BN && coopmat2BK == other.coopmat2BK &&
         stridedCoopmat2BlockSize == other.stridedCoopmat2BlockSize && stridedCoopmat2BM == other.stridedCoopmat2BM &&
         stridedCoopmat2BN == other.stridedCoopmat2BN && stridedCoopmat2BK == other.stridedCoopmat2BK &&
         enableWinogradGemmCoopmat2AccF16 == other.enableWinogradGemmCoopmat2AccF16 &&
         enableGemmStridedCoopmat2AccF16 == other.enableGemmStridedCoopmat2AccF16 &&
         winogradGemmCoopmat2AccF16TunerValueValid == other.winogradGemmCoopmat2AccF16TunerValueValid &&
         gemmStridedCoopmat2AccF16TunerValueValid == other.gemmStridedCoopmat2AccF16TunerValueValid &&
         winogradGemmCoopmat2AccF16SelectionRank == other.winogradGemmCoopmat2AccF16SelectionRank &&
         gemmStridedCoopmat2AccF16SelectionRank == other.gemmStridedCoopmat2AccF16SelectionRank &&
         coopmat2AccF16BlockSize == other.coopmat2AccF16BlockSize && coopmat2AccF16BM == other.coopmat2AccF16BM &&
         coopmat2AccF16BN == other.coopmat2AccF16BN && coopmat2AccF16BK == other.coopmat2AccF16BK &&
         stridedCoopmat2AccF16BlockSize == other.stridedCoopmat2AccF16BlockSize &&
         stridedCoopmat2AccF16BM == other.stridedCoopmat2AccF16BM &&
         stridedCoopmat2AccF16BN == other.stridedCoopmat2AccF16BN &&
         stridedCoopmat2AccF16BK == other.stridedCoopmat2AccF16BK;
}

bool VulkanTuneParams::isValid() const {
  return (winograd3x3OutTile == 2 || winograd3x3OutTile == 4) &&
         VulkanKernels::WinogradTransform::isConfigSupported(
           winogradTransformLocalSizeX, winogradTransformLocalSizeY) &&
         VulkanKernels::WinogradBNActTransform::isConfigSupported(
           winogradBNActTransformLocalSizeX, winogradBNActTransformLocalSizeY) &&
         VulkanKernels::WinogradUntransform::isConfigSupported(
           winogradUntransformLocalSizeX, winogradUntransformLocalSizeY) &&
         VulkanKernels::WinogradGemm::isConfigSupported(winogradGemmM, winogradGemmN, winogradGemmK, winogradGemmRN) &&
         gemmVariantSelectionRankValid(winogradGemmTiledSelectionRank) &&
         (enableWinogradGemmTiledFp16Compute == 0 || enableWinogradGemmTiledFp16Compute == 1) &&
         (enableGemmStridedTiledFp16Compute == 0 || enableGemmStridedTiledFp16Compute == 1) &&
         (winogradGemmTiledFp16ComputeTunerValueValid == 0 || winogradGemmTiledFp16ComputeTunerValueValid == 1) &&
         (gemmStridedTiledFp16ComputeTunerValueValid == 0 || gemmStridedTiledFp16ComputeTunerValueValid == 1) &&
         gemmVariantSelectionRankValid(winogradGemmTiledFp16ComputeSelectionRank) &&
         gemmVariantSelectionRankValid(gemmStridedTiledFp16ComputeSelectionRank) &&
         VulkanKernels::GemmStridedTiled::isConfigSupported(
           gemmStridedTiledLocalSizeX, gemmStridedTiledLocalSizeY, gemmStridedTiledTileK, gemmStridedTiledRN) &&
         gemmVariantSelectionRankValid(gemmStridedTiledSelectionRank) &&
         VulkanKernels::GemmDirectFP32::isConfigSupported(gemmDirectLocalSizeX, gemmDirectLocalSizeY) &&
         VulkanKernels::AttentionTiled::isConfigSupported(attnBlockQ, attnBlockKV, attnQPerThread) &&
         VulkanKernels::GPoolReduction::isConfigSupported(gpoolXystride) &&
         VulkanKernels::ValueHeadPool::isConfigSupported(valueHeadPoolXystride) &&
         VulkanKernels::SpatialRMSNorm::isConfigSupported(spatialRMSNormTile) &&
         (enableWinogradGemmDot2 == 0 || enableWinogradGemmDot2 == 1) &&
         (enableGemmStridedDot2 == 0 || enableGemmStridedDot2 == 1) &&
         (enableWinogradGemmDot2AccF16 == 0 || enableWinogradGemmDot2AccF16 == 1) &&
         (enableGemmStridedDot2AccF16 == 0 || enableGemmStridedDot2AccF16 == 1) &&
         (winogradGemmDot2TunerValueValid == 0 || winogradGemmDot2TunerValueValid == 1) &&
         (gemmStridedDot2TunerValueValid == 0 || gemmStridedDot2TunerValueValid == 1) &&
         (winogradGemmDot2AccF16TunerValueValid == 0 || winogradGemmDot2AccF16TunerValueValid == 1) &&
         (gemmStridedDot2AccF16TunerValueValid == 0 || gemmStridedDot2AccF16TunerValueValid == 1) &&
         gemmVariantSelectionRankValid(winogradGemmDot2SelectionRank) &&
         gemmVariantSelectionRankValid(gemmStridedDot2SelectionRank) &&
         gemmVariantSelectionRankValid(winogradGemmDot2AccF16SelectionRank) &&
         gemmVariantSelectionRankValid(gemmStridedDot2AccF16SelectionRank) &&
         VulkanKernels::WinogradGemmDot2::isConfigSupported(
           dot2BlockSize, dot2BM, dot2BN, dot2WM, dot2WN, dot2WMIter, dot2TM, dot2TN, dot2Warp) &&
         VulkanKernels::GemmStridedDot2::isConfigSupported(
           stridedDot2BlockSize,
           stridedDot2BM,
           stridedDot2BN,
           stridedDot2WM,
           stridedDot2WN,
           stridedDot2WMIter,
           stridedDot2TM,
           stridedDot2TN,
           stridedDot2Warp) &&
         VulkanKernels::WinogradGemmDot2AccF16::isConfigSupported(
           dot2AccF16BlockSize,
           dot2AccF16BM,
           dot2AccF16BN,
           dot2AccF16WM,
           dot2AccF16WN,
           dot2AccF16WMIter,
           dot2AccF16TM,
           dot2AccF16TN,
           dot2AccF16Warp) &&
         VulkanKernels::GemmStridedDot2AccF16::isConfigSupported(
           stridedDot2AccF16BlockSize,
           stridedDot2AccF16BM,
           stridedDot2AccF16BN,
           stridedDot2AccF16WM,
           stridedDot2AccF16WN,
           stridedDot2AccF16WMIter,
           stridedDot2AccF16TM,
           stridedDot2AccF16TN,
           stridedDot2AccF16Warp) &&
         (enableWinogradGemmCoopmat == 0 || enableWinogradGemmCoopmat == 1) &&
         (enableGemmStridedCoopmat == 0 || enableGemmStridedCoopmat == 1) &&
         (winogradGemmCoopmatTunerValueValid == 0 || winogradGemmCoopmatTunerValueValid == 1) &&
         (gemmStridedCoopmatTunerValueValid == 0 || gemmStridedCoopmatTunerValueValid == 1) &&
         gemmVariantSelectionRankValid(winogradGemmCoopmatSelectionRank) &&
         gemmVariantSelectionRankValid(gemmStridedCoopmatSelectionRank) &&
         VulkanKernels::WinogradGemmCoopmat::isConfigSupported(
           coopmatBlockSize,
           coopmatBM,
           coopmatBN,
           coopmatBK,
           coopmatWM,
           coopmatWN,
           coopmatTM,
           coopmatTN,
           coopmatTK,
           coopmatWarp) &&
         VulkanKernels::GemmStridedCoopmat::isConfigSupported(
           stridedCoopmatBlockSize,
           stridedCoopmatBM,
           stridedCoopmatBN,
           stridedCoopmatBK,
           stridedCoopmatWM,
           stridedCoopmatWN,
           stridedCoopmatTM,
           stridedCoopmatTN,
           stridedCoopmatTK,
           stridedCoopmatWarp) &&
         (enableWinogradGemmCoopmatAccF16 == 0 || enableWinogradGemmCoopmatAccF16 == 1) &&
         (enableGemmStridedCoopmatAccF16 == 0 || enableGemmStridedCoopmatAccF16 == 1) &&
         (winogradGemmCoopmatAccF16TunerValueValid == 0 || winogradGemmCoopmatAccF16TunerValueValid == 1) &&
         (gemmStridedCoopmatAccF16TunerValueValid == 0 || gemmStridedCoopmatAccF16TunerValueValid == 1) &&
         gemmVariantSelectionRankValid(winogradGemmCoopmatAccF16SelectionRank) &&
         gemmVariantSelectionRankValid(gemmStridedCoopmatAccF16SelectionRank) &&
         VulkanKernels::WinogradGemmCoopmatAccF16::isConfigSupported(
           coopmatAccF16BlockSize,
           coopmatAccF16BM,
           coopmatAccF16BN,
           coopmatAccF16BK,
           coopmatAccF16WM,
           coopmatAccF16WN,
           coopmatAccF16TM,
           coopmatAccF16TN,
           coopmatAccF16TK,
           coopmatAccF16Warp) &&
         VulkanKernels::GemmStridedCoopmatAccF16::isConfigSupported(
           stridedCoopmatAccF16BlockSize,
           stridedCoopmatAccF16BM,
           stridedCoopmatAccF16BN,
           stridedCoopmatAccF16BK,
           stridedCoopmatAccF16WM,
           stridedCoopmatAccF16WN,
           stridedCoopmatAccF16TM,
           stridedCoopmatAccF16TN,
           stridedCoopmatAccF16TK,
           stridedCoopmatAccF16Warp) &&
         (enableWinogradGemmCoopmat2 == 0 || enableWinogradGemmCoopmat2 == 1) &&
         (enableGemmStridedCoopmat2 == 0 || enableGemmStridedCoopmat2 == 1) &&
         (winogradGemmCoopmat2TunerValueValid == 0 || winogradGemmCoopmat2TunerValueValid == 1) &&
         (gemmStridedCoopmat2TunerValueValid == 0 || gemmStridedCoopmat2TunerValueValid == 1) &&
         gemmVariantSelectionRankValid(winogradGemmCoopmat2SelectionRank) &&
         gemmVariantSelectionRankValid(gemmStridedCoopmat2SelectionRank) &&
         VulkanKernels::WinogradGemmCoopmat2::isConfigSupported(
           coopmat2BlockSize, coopmat2BM, coopmat2BN, coopmat2BK) &&
         VulkanKernels::GemmStridedCoopmat2::isConfigSupported(
           stridedCoopmat2BlockSize, stridedCoopmat2BM, stridedCoopmat2BN, stridedCoopmat2BK) &&
         (enableWinogradGemmCoopmat2AccF16 == 0 || enableWinogradGemmCoopmat2AccF16 == 1) &&
         (enableGemmStridedCoopmat2AccF16 == 0 || enableGemmStridedCoopmat2AccF16 == 1) &&
         (winogradGemmCoopmat2AccF16TunerValueValid == 0 || winogradGemmCoopmat2AccF16TunerValueValid == 1) &&
         (gemmStridedCoopmat2AccF16TunerValueValid == 0 || gemmStridedCoopmat2AccF16TunerValueValid == 1) &&
         gemmVariantSelectionRankValid(winogradGemmCoopmat2AccF16SelectionRank) &&
         gemmVariantSelectionRankValid(gemmStridedCoopmat2AccF16SelectionRank) &&
         VulkanKernels::WinogradGemmCoopmat2AccF16::isConfigSupported(
           coopmat2AccF16BlockSize, coopmat2AccF16BM, coopmat2AccF16BN, coopmat2AccF16BK) &&
         VulkanKernels::GemmStridedCoopmat2AccF16::isConfigSupported(
           stridedCoopmat2AccF16BlockSize, stridedCoopmat2AccF16BM, stridedCoopmat2AccF16BN, stridedCoopmat2AccF16BK);
}

bool VulkanTuneParams::normalizeGemmVariantSelections() {
  bool changed = disableTiledFp16ComputeGemmVariants(*this);
  changed |= selectGemmVariantByRank(
    *this,
    {{&VulkanTuneParams::enableWinogradGemmCoopmat2AccF16,
      &VulkanTuneParams::winogradGemmCoopmat2AccF16TunerValueValid,
      &VulkanTuneParams::winogradGemmCoopmat2AccF16SelectionRank},
     {&VulkanTuneParams::enableWinogradGemmCoopmat2,
      &VulkanTuneParams::winogradGemmCoopmat2TunerValueValid,
      &VulkanTuneParams::winogradGemmCoopmat2SelectionRank},
     {&VulkanTuneParams::enableWinogradGemmCoopmatAccF16,
      &VulkanTuneParams::winogradGemmCoopmatAccF16TunerValueValid,
      &VulkanTuneParams::winogradGemmCoopmatAccF16SelectionRank},
     {&VulkanTuneParams::enableWinogradGemmCoopmat,
      &VulkanTuneParams::winogradGemmCoopmatTunerValueValid,
      &VulkanTuneParams::winogradGemmCoopmatSelectionRank},
     {&VulkanTuneParams::enableWinogradGemmDot2AccF16,
      &VulkanTuneParams::winogradGemmDot2AccF16TunerValueValid,
      &VulkanTuneParams::winogradGemmDot2AccF16SelectionRank},
     {&VulkanTuneParams::enableWinogradGemmDot2,
      &VulkanTuneParams::winogradGemmDot2TunerValueValid,
      &VulkanTuneParams::winogradGemmDot2SelectionRank},
     {nullptr, nullptr, &VulkanTuneParams::winogradGemmTiledSelectionRank}});
  changed |= selectGemmVariantByRank(
    *this,
    {{&VulkanTuneParams::enableGemmStridedCoopmat2AccF16,
      &VulkanTuneParams::gemmStridedCoopmat2AccF16TunerValueValid,
      &VulkanTuneParams::gemmStridedCoopmat2AccF16SelectionRank},
     {&VulkanTuneParams::enableGemmStridedCoopmat2,
      &VulkanTuneParams::gemmStridedCoopmat2TunerValueValid,
      &VulkanTuneParams::gemmStridedCoopmat2SelectionRank},
     {&VulkanTuneParams::enableGemmStridedCoopmatAccF16,
      &VulkanTuneParams::gemmStridedCoopmatAccF16TunerValueValid,
      &VulkanTuneParams::gemmStridedCoopmatAccF16SelectionRank},
     {&VulkanTuneParams::enableGemmStridedCoopmat,
      &VulkanTuneParams::gemmStridedCoopmatTunerValueValid,
      &VulkanTuneParams::gemmStridedCoopmatSelectionRank},
     {&VulkanTuneParams::enableGemmStridedDot2AccF16,
      &VulkanTuneParams::gemmStridedDot2AccF16TunerValueValid,
      &VulkanTuneParams::gemmStridedDot2AccF16SelectionRank},
     {&VulkanTuneParams::enableGemmStridedDot2,
      &VulkanTuneParams::gemmStridedDot2TunerValueValid,
      &VulkanTuneParams::gemmStridedDot2SelectionRank},
     {nullptr, nullptr, &VulkanTuneParams::gemmStridedTiledSelectionRank}});
  return changed;
}

bool VulkanTuneParams::normalizeGemmVariantSelectionsForDevice(
  const VulkanDeviceInfo& deviceInfo,
  bool fp16Storage,
  bool fp16Compute) {
  auto supports = [&](bool featureSupported, int32_t VulkanTuneParams::* tunerValueValid) {
    return fp16Storage && fp16Compute && featureSupported && this->*tunerValueValid != 0;
  };

  bool changed = disableTiledFp16ComputeGemmVariants(*this);
  changed |= selectGemmVariantByRank(
    *this,
    {{&VulkanTuneParams::enableWinogradGemmCoopmat2AccF16,
      &VulkanTuneParams::winogradGemmCoopmat2AccF16TunerValueValid,
      &VulkanTuneParams::winogradGemmCoopmat2AccF16SelectionRank,
      supports(deviceInfo.supportsCoopmat2F16AccF16, &VulkanTuneParams::winogradGemmCoopmat2AccF16TunerValueValid)},
     {&VulkanTuneParams::enableWinogradGemmCoopmat2,
      &VulkanTuneParams::winogradGemmCoopmat2TunerValueValid,
      &VulkanTuneParams::winogradGemmCoopmat2SelectionRank,
      supports(deviceInfo.supportsCoopmat2F16, &VulkanTuneParams::winogradGemmCoopmat2TunerValueValid)},
     {&VulkanTuneParams::enableWinogradGemmCoopmatAccF16,
      &VulkanTuneParams::winogradGemmCoopmatAccF16TunerValueValid,
      &VulkanTuneParams::winogradGemmCoopmatAccF16SelectionRank,
      supports(deviceInfo.supportsCoopmat1F16AccF16, &VulkanTuneParams::winogradGemmCoopmatAccF16TunerValueValid)},
     {&VulkanTuneParams::enableWinogradGemmCoopmat,
      &VulkanTuneParams::winogradGemmCoopmatTunerValueValid,
      &VulkanTuneParams::winogradGemmCoopmatSelectionRank,
      supports(deviceInfo.supportsCoopmat1F16, &VulkanTuneParams::winogradGemmCoopmatTunerValueValid)},
     {&VulkanTuneParams::enableWinogradGemmDot2AccF16,
      &VulkanTuneParams::winogradGemmDot2AccF16TunerValueValid,
      &VulkanTuneParams::winogradGemmDot2AccF16SelectionRank,
      supports(deviceInfo.supportsDot2F16AccF16, &VulkanTuneParams::winogradGemmDot2AccF16TunerValueValid)},
     {&VulkanTuneParams::enableWinogradGemmDot2,
      &VulkanTuneParams::winogradGemmDot2TunerValueValid,
      &VulkanTuneParams::winogradGemmDot2SelectionRank,
      supports(deviceInfo.supportsDot2F16, &VulkanTuneParams::winogradGemmDot2TunerValueValid)},
     {nullptr, nullptr, &VulkanTuneParams::winogradGemmTiledSelectionRank, true}});
  changed |= selectGemmVariantByRank(
    *this,
    {{&VulkanTuneParams::enableGemmStridedCoopmat2AccF16,
      &VulkanTuneParams::gemmStridedCoopmat2AccF16TunerValueValid,
      &VulkanTuneParams::gemmStridedCoopmat2AccF16SelectionRank,
      supports(deviceInfo.supportsCoopmat2F16AccF16, &VulkanTuneParams::gemmStridedCoopmat2AccF16TunerValueValid)},
     {&VulkanTuneParams::enableGemmStridedCoopmat2,
      &VulkanTuneParams::gemmStridedCoopmat2TunerValueValid,
      &VulkanTuneParams::gemmStridedCoopmat2SelectionRank,
      supports(deviceInfo.supportsCoopmat2F16, &VulkanTuneParams::gemmStridedCoopmat2TunerValueValid)},
     {&VulkanTuneParams::enableGemmStridedCoopmatAccF16,
      &VulkanTuneParams::gemmStridedCoopmatAccF16TunerValueValid,
      &VulkanTuneParams::gemmStridedCoopmatAccF16SelectionRank,
      supports(deviceInfo.supportsCoopmat1F16AccF16, &VulkanTuneParams::gemmStridedCoopmatAccF16TunerValueValid)},
     {&VulkanTuneParams::enableGemmStridedCoopmat,
      &VulkanTuneParams::gemmStridedCoopmatTunerValueValid,
      &VulkanTuneParams::gemmStridedCoopmatSelectionRank,
      supports(deviceInfo.supportsCoopmat1F16, &VulkanTuneParams::gemmStridedCoopmatTunerValueValid)},
     {&VulkanTuneParams::enableGemmStridedDot2AccF16,
      &VulkanTuneParams::gemmStridedDot2AccF16TunerValueValid,
      &VulkanTuneParams::gemmStridedDot2AccF16SelectionRank,
      supports(deviceInfo.supportsDot2F16AccF16, &VulkanTuneParams::gemmStridedDot2AccF16TunerValueValid)},
     {&VulkanTuneParams::enableGemmStridedDot2,
      &VulkanTuneParams::gemmStridedDot2TunerValueValid,
      &VulkanTuneParams::gemmStridedDot2SelectionRank,
      supports(deviceInfo.supportsDot2F16, &VulkanTuneParams::gemmStridedDot2TunerValueValid)},
     {nullptr, nullptr, &VulkanTuneParams::gemmStridedTiledSelectionRank, true}});
  return changed;
}

void VulkanTuneParams::inferLegacyGemmVariantTuningValidity() {
  winogradGemmTiledFp16ComputeTunerValueValid = 0;
  gemmStridedTiledFp16ComputeTunerValueValid = 0;

  winogradGemmDot2TunerValueValid = enableWinogradGemmDot2 != 0 ? 1 : 0;
  gemmStridedDot2TunerValueValid = enableGemmStridedDot2 != 0 ? 1 : 0;
  winogradGemmDot2AccF16TunerValueValid = enableWinogradGemmDot2AccF16 != 0 ? 1 : 0;
  gemmStridedDot2AccF16TunerValueValid = enableGemmStridedDot2AccF16 != 0 ? 1 : 0;

  winogradGemmCoopmatTunerValueValid = enableWinogradGemmCoopmat != 0 ? 1 : 0;
  gemmStridedCoopmatTunerValueValid = enableGemmStridedCoopmat != 0 ? 1 : 0;
  winogradGemmCoopmatAccF16TunerValueValid = enableWinogradGemmCoopmatAccF16 != 0 ? 1 : 0;
  gemmStridedCoopmatAccF16TunerValueValid = enableGemmStridedCoopmatAccF16 != 0 ? 1 : 0;

  winogradGemmCoopmat2TunerValueValid = enableWinogradGemmCoopmat2 != 0 ? 1 : 0;
  gemmStridedCoopmat2TunerValueValid = enableGemmStridedCoopmat2 != 0 ? 1 : 0;
  winogradGemmCoopmat2AccF16TunerValueValid = enableWinogradGemmCoopmat2AccF16 != 0 ? 1 : 0;
  gemmStridedCoopmat2AccF16TunerValueValid = enableGemmStridedCoopmat2AccF16 != 0 ? 1 : 0;
}

void VulkanTuneParams::inferLegacyGemmVariantSelectionRanks() {
  auto firstEnabledRank =
    [&](std::initializer_list<GemmVariantSelectionField> fields, int32_t VulkanTuneParams::* fallbackRank) {
      int32_t VulkanTuneParams::* selectedRank = nullptr;
      int32_t bestRank = GEMM_VARIANT_SELECTION_RANK_MAX + 1;
      for(const GemmVariantSelectionField& field: fields) {
        const int32_t rank = this->*field.selectionRank;
        if((this->*field.enabled) != 0 && gemmVariantSelectionRankValid(rank) && rank < bestRank) {
          selectedRank = field.selectionRank;
          bestRank = rank;
        }
      }
      return selectedRank != nullptr ? selectedRank : fallbackRank;
    };

  promoteWinogradGemmSelectionRank(
    *this,
    firstEnabledRank(
      {{&VulkanTuneParams::enableWinogradGemmCoopmat2AccF16,
        &VulkanTuneParams::winogradGemmCoopmat2AccF16TunerValueValid,
        &VulkanTuneParams::winogradGemmCoopmat2AccF16SelectionRank},
       {&VulkanTuneParams::enableWinogradGemmCoopmat2,
        &VulkanTuneParams::winogradGemmCoopmat2TunerValueValid,
        &VulkanTuneParams::winogradGemmCoopmat2SelectionRank},
       {&VulkanTuneParams::enableWinogradGemmCoopmatAccF16,
        &VulkanTuneParams::winogradGemmCoopmatAccF16TunerValueValid,
        &VulkanTuneParams::winogradGemmCoopmatAccF16SelectionRank},
       {&VulkanTuneParams::enableWinogradGemmCoopmat,
        &VulkanTuneParams::winogradGemmCoopmatTunerValueValid,
        &VulkanTuneParams::winogradGemmCoopmatSelectionRank},
       {&VulkanTuneParams::enableWinogradGemmDot2AccF16,
        &VulkanTuneParams::winogradGemmDot2AccF16TunerValueValid,
        &VulkanTuneParams::winogradGemmDot2AccF16SelectionRank},
       {&VulkanTuneParams::enableWinogradGemmDot2,
        &VulkanTuneParams::winogradGemmDot2TunerValueValid,
        &VulkanTuneParams::winogradGemmDot2SelectionRank}},
      &VulkanTuneParams::winogradGemmTiledSelectionRank));
  promoteGemmStridedSelectionRank(
    *this,
    firstEnabledRank(
      {{&VulkanTuneParams::enableGemmStridedCoopmat2AccF16,
        &VulkanTuneParams::gemmStridedCoopmat2AccF16TunerValueValid,
        &VulkanTuneParams::gemmStridedCoopmat2AccF16SelectionRank},
       {&VulkanTuneParams::enableGemmStridedCoopmat2,
        &VulkanTuneParams::gemmStridedCoopmat2TunerValueValid,
        &VulkanTuneParams::gemmStridedCoopmat2SelectionRank},
       {&VulkanTuneParams::enableGemmStridedCoopmatAccF16,
        &VulkanTuneParams::gemmStridedCoopmatAccF16TunerValueValid,
        &VulkanTuneParams::gemmStridedCoopmatAccF16SelectionRank},
       {&VulkanTuneParams::enableGemmStridedCoopmat,
        &VulkanTuneParams::gemmStridedCoopmatTunerValueValid,
        &VulkanTuneParams::gemmStridedCoopmatSelectionRank},
       {&VulkanTuneParams::enableGemmStridedDot2AccF16,
        &VulkanTuneParams::gemmStridedDot2AccF16TunerValueValid,
        &VulkanTuneParams::gemmStridedDot2AccF16SelectionRank},
       {&VulkanTuneParams::enableGemmStridedDot2,
        &VulkanTuneParams::gemmStridedDot2TunerValueValid,
        &VulkanTuneParams::gemmStridedDot2SelectionRank}},
      &VulkanTuneParams::gemmStridedTiledSelectionRank));
}

void VulkanTuneParams::save(const string& filename, const VulkanTuneParams& config) {
  string tmpFilename = uniqueTempFileNameForAtomicSave(filename);
  ofstream out;
  FileUtils::open(out, tmpFilename, std::ios::out | std::ios::trunc);
  out << VULKAN_TUNEPARAMS_VERSION_LINE << "\n";
  out << "#winograd3x3\n";
  out << "winograd3x3OutTile=" << config.winograd3x3OutTile << "\n";
  out << "#winogradGemm\n";
  out << "winogradGemmM=" << config.winogradGemmM << " winogradGemmN=" << config.winogradGemmN
      << " winogradGemmK=" << config.winogradGemmK << " winogradGemmRN=" << config.winogradGemmRN
      << " winogradGemmTiledSelectionRank=" << config.winogradGemmTiledSelectionRank << "\n";
  out << "#fp16ComputeTiled\n";
  out << "winogradGemmTiledFp16ComputeTunerValueValid=" << config.winogradGemmTiledFp16ComputeTunerValueValid
      << " gemmStridedTiledFp16ComputeTunerValueValid=" << config.gemmStridedTiledFp16ComputeTunerValueValid
      << " winogradGemmTiledFp16ComputeSelectionRank=" << config.winogradGemmTiledFp16ComputeSelectionRank
      << " gemmStridedTiledFp16ComputeSelectionRank=" << config.gemmStridedTiledFp16ComputeSelectionRank
      << " gemmVariantFilterTunedSignature=" << config.gemmVariantFilterTunedSignature << "\n";
  out << "#winogradGemmFp16Compute\n";
  out << "winogradGemmFp16ComputeM=" << config.winogradGemmFp16ComputeM
      << " winogradGemmFp16ComputeN=" << config.winogradGemmFp16ComputeN
      << " winogradGemmFp16ComputeK=" << config.winogradGemmFp16ComputeK
      << " winogradGemmFp16ComputeRN=" << config.winogradGemmFp16ComputeRN << "\n";
  out << "#attention\n";
  out << "attnBlockQ=" << config.attnBlockQ << " attnBlockKV=" << config.attnBlockKV
      << " attnQPerThread=" << config.attnQPerThread << "\n";
  out << "#winogradTransformTunedLayout\n";
  out << "winogradTransformTunedLayout=" << config.winogradTransformTunedLayout << "\n";
  out << "#winogradTransformLocalSize\n";
  out << "winogradTransformLocalSizeX=" << config.winogradTransformLocalSizeX
      << " winogradTransformLocalSizeY=" << config.winogradTransformLocalSizeY << "\n";
  out << "#winogradBNActTransformLocalSize\n";
  out << "winogradBNActTransformLocalSizeX=" << config.winogradBNActTransformLocalSizeX
      << " winogradBNActTransformLocalSizeY=" << config.winogradBNActTransformLocalSizeY << "\n";
  out << "#winogradUntransformLocalSize\n";
  out << "winogradUntransformLocalSizeX=" << config.winogradUntransformLocalSizeX
      << " winogradUntransformLocalSizeY=" << config.winogradUntransformLocalSizeY << "\n";
  out << "#gemmStridedTiledLocalSize\n";
  out << "gemmStridedTiledLocalSizeX=" << config.gemmStridedTiledLocalSizeX
      << " gemmStridedTiledLocalSizeY=" << config.gemmStridedTiledLocalSizeY
      << " gemmStridedTiledTileK=" << config.gemmStridedTiledTileK
      << " gemmStridedTiledRN=" << config.gemmStridedTiledRN
      << " gemmStridedTiledSelectionRank=" << config.gemmStridedTiledSelectionRank << "\n";
  out << "#gemmStridedFp16Compute\n";
  out << "gemmStridedFp16ComputeLocalSizeX=" << config.gemmStridedFp16ComputeLocalSizeX
      << " gemmStridedFp16ComputeLocalSizeY=" << config.gemmStridedFp16ComputeLocalSizeY
      << " gemmStridedFp16ComputeTileK=" << config.gemmStridedFp16ComputeTileK
      << " gemmStridedFp16ComputeRN=" << config.gemmStridedFp16ComputeRN << "\n";
  out << "#gemmDirectLocalSize\n";
  out << "gemmDirectLocalSizeX=" << config.gemmDirectLocalSizeX
      << " gemmDirectLocalSizeY=" << config.gemmDirectLocalSizeY << "\n";
  out << "#reductionTiles\n";
  out << "gpoolXystride=" << config.gpoolXystride << " valueHeadPoolXystride=" << config.valueHeadPoolXystride
      << " spatialRMSNormTile=" << config.spatialRMSNormTile << "\n";
  out << "#dot2\n";
  out << "winogradGemmDot2TunerValueValid=" << config.winogradGemmDot2TunerValueValid
      << " gemmStridedDot2TunerValueValid=" << config.gemmStridedDot2TunerValueValid
      << " winogradGemmDot2AccF16TunerValueValid=" << config.winogradGemmDot2AccF16TunerValueValid
      << " gemmStridedDot2AccF16TunerValueValid=" << config.gemmStridedDot2AccF16TunerValueValid
      << " winogradGemmDot2SelectionRank=" << config.winogradGemmDot2SelectionRank
      << " gemmStridedDot2SelectionRank=" << config.gemmStridedDot2SelectionRank
      << " winogradGemmDot2AccF16SelectionRank=" << config.winogradGemmDot2AccF16SelectionRank
      << " gemmStridedDot2AccF16SelectionRank=" << config.gemmStridedDot2AccF16SelectionRank << "\n";
  out << "#winogradDot2tile\n";
  out << "dot2BlockSize=" << config.dot2BlockSize << " dot2BM=" << config.dot2BM << " dot2BN=" << config.dot2BN
      << " dot2WM=" << config.dot2WM << " dot2WN=" << config.dot2WN << " dot2WMIter=" << config.dot2WMIter
      << " dot2TM=" << config.dot2TM << " dot2TN=" << config.dot2TN << " dot2Warp=" << config.dot2Warp << "\n";
  out << "#stridedDot2tile\n";
  out << "stridedDot2BlockSize=" << config.stridedDot2BlockSize << " stridedDot2BM=" << config.stridedDot2BM
      << " stridedDot2BN=" << config.stridedDot2BN << " stridedDot2WM=" << config.stridedDot2WM
      << " stridedDot2WN=" << config.stridedDot2WN << " stridedDot2WMIter=" << config.stridedDot2WMIter
      << " stridedDot2TM=" << config.stridedDot2TM << " stridedDot2TN=" << config.stridedDot2TN
      << " stridedDot2Warp=" << config.stridedDot2Warp << "\n";
  out << "#winogradDot2AccF16tile\n";
  out << "dot2AccF16BlockSize=" << config.dot2AccF16BlockSize << " dot2AccF16BM=" << config.dot2AccF16BM
      << " dot2AccF16BN=" << config.dot2AccF16BN << " dot2AccF16WM=" << config.dot2AccF16WM
      << " dot2AccF16WN=" << config.dot2AccF16WN << " dot2AccF16WMIter=" << config.dot2AccF16WMIter
      << " dot2AccF16TM=" << config.dot2AccF16TM << " dot2AccF16TN=" << config.dot2AccF16TN
      << " dot2AccF16Warp=" << config.dot2AccF16Warp << "\n";
  out << "#stridedDot2AccF16tile\n";
  out << "stridedDot2AccF16BlockSize=" << config.stridedDot2AccF16BlockSize
      << " stridedDot2AccF16BM=" << config.stridedDot2AccF16BM << " stridedDot2AccF16BN=" << config.stridedDot2AccF16BN
      << " stridedDot2AccF16WM=" << config.stridedDot2AccF16WM << " stridedDot2AccF16WN=" << config.stridedDot2AccF16WN
      << " stridedDot2AccF16WMIter=" << config.stridedDot2AccF16WMIter
      << " stridedDot2AccF16TM=" << config.stridedDot2AccF16TM << " stridedDot2AccF16TN=" << config.stridedDot2AccF16TN
      << " stridedDot2AccF16Warp=" << config.stridedDot2AccF16Warp << "\n";
  out << "#coopmat\n";
  out << "winogradGemmCoopmatTunerValueValid=" << config.winogradGemmCoopmatTunerValueValid
      << " gemmStridedCoopmatTunerValueValid=" << config.gemmStridedCoopmatTunerValueValid
      << " winogradGemmCoopmatSelectionRank=" << config.winogradGemmCoopmatSelectionRank
      << " gemmStridedCoopmatSelectionRank=" << config.gemmStridedCoopmatSelectionRank << "\n";
  out << "#winogradCoopmatTile\n";
  out << "coopmatBlockSize=" << config.coopmatBlockSize << " coopmatBM=" << config.coopmatBM
      << " coopmatBN=" << config.coopmatBN << " coopmatBK=" << config.coopmatBK << " coopmatWM=" << config.coopmatWM
      << " coopmatWN=" << config.coopmatWN << " coopmatTM=" << config.coopmatTM << " coopmatTN=" << config.coopmatTN
      << " coopmatTK=" << config.coopmatTK << " coopmatWarp=" << config.coopmatWarp << "\n";
  out << "#stridedCoopmatTile\n";
  out << "stridedCoopmatBlockSize=" << config.stridedCoopmatBlockSize << " stridedCoopmatBM=" << config.stridedCoopmatBM
      << " stridedCoopmatBN=" << config.stridedCoopmatBN << " stridedCoopmatBK=" << config.stridedCoopmatBK
      << " stridedCoopmatWM=" << config.stridedCoopmatWM << " stridedCoopmatWN=" << config.stridedCoopmatWN
      << " stridedCoopmatTM=" << config.stridedCoopmatTM << " stridedCoopmatTN=" << config.stridedCoopmatTN
      << " stridedCoopmatTK=" << config.stridedCoopmatTK << " stridedCoopmatWarp=" << config.stridedCoopmatWarp << "\n";
  out << "#coopmatAccF16\n";
  out << "winogradGemmCoopmatAccF16TunerValueValid=" << config.winogradGemmCoopmatAccF16TunerValueValid
      << " gemmStridedCoopmatAccF16TunerValueValid=" << config.gemmStridedCoopmatAccF16TunerValueValid
      << " winogradGemmCoopmatAccF16SelectionRank=" << config.winogradGemmCoopmatAccF16SelectionRank
      << " gemmStridedCoopmatAccF16SelectionRank=" << config.gemmStridedCoopmatAccF16SelectionRank << "\n";
  out << "#winogradCoopmatAccF16Tile\n";
  out << "coopmatAccF16BlockSize=" << config.coopmatAccF16BlockSize << " coopmatAccF16BM=" << config.coopmatAccF16BM
      << " coopmatAccF16BN=" << config.coopmatAccF16BN << " coopmatAccF16BK=" << config.coopmatAccF16BK
      << " coopmatAccF16WM=" << config.coopmatAccF16WM << " coopmatAccF16WN=" << config.coopmatAccF16WN
      << " coopmatAccF16TM=" << config.coopmatAccF16TM << " coopmatAccF16TN=" << config.coopmatAccF16TN
      << " coopmatAccF16TK=" << config.coopmatAccF16TK << " coopmatAccF16Warp=" << config.coopmatAccF16Warp << "\n";
  out << "#stridedCoopmatAccF16Tile\n";
  out << "stridedCoopmatAccF16BlockSize=" << config.stridedCoopmatAccF16BlockSize
      << " stridedCoopmatAccF16BM=" << config.stridedCoopmatAccF16BM
      << " stridedCoopmatAccF16BN=" << config.stridedCoopmatAccF16BN
      << " stridedCoopmatAccF16BK=" << config.stridedCoopmatAccF16BK
      << " stridedCoopmatAccF16WM=" << config.stridedCoopmatAccF16WM
      << " stridedCoopmatAccF16WN=" << config.stridedCoopmatAccF16WN
      << " stridedCoopmatAccF16TM=" << config.stridedCoopmatAccF16TM
      << " stridedCoopmatAccF16TN=" << config.stridedCoopmatAccF16TN
      << " stridedCoopmatAccF16TK=" << config.stridedCoopmatAccF16TK
      << " stridedCoopmatAccF16Warp=" << config.stridedCoopmatAccF16Warp << "\n";
  out << "#coopmat2\n";
  out << "winogradGemmCoopmat2TunerValueValid=" << config.winogradGemmCoopmat2TunerValueValid
      << " gemmStridedCoopmat2TunerValueValid=" << config.gemmStridedCoopmat2TunerValueValid
      << " winogradGemmCoopmat2SelectionRank=" << config.winogradGemmCoopmat2SelectionRank
      << " gemmStridedCoopmat2SelectionRank=" << config.gemmStridedCoopmat2SelectionRank << "\n";
  out << "#winogradCoopmat2Tile\n";
  out << "coopmat2BlockSize=" << config.coopmat2BlockSize << " coopmat2BM=" << config.coopmat2BM
      << " coopmat2BN=" << config.coopmat2BN << " coopmat2BK=" << config.coopmat2BK << "\n";
  out << "#stridedCoopmat2Tile\n";
  out << "stridedCoopmat2BlockSize=" << config.stridedCoopmat2BlockSize
      << " stridedCoopmat2BM=" << config.stridedCoopmat2BM << " stridedCoopmat2BN=" << config.stridedCoopmat2BN
      << " stridedCoopmat2BK=" << config.stridedCoopmat2BK << "\n";
  out << "#coopmat2AccF16\n";
  out << "winogradGemmCoopmat2AccF16TunerValueValid=" << config.winogradGemmCoopmat2AccF16TunerValueValid
      << " gemmStridedCoopmat2AccF16TunerValueValid=" << config.gemmStridedCoopmat2AccF16TunerValueValid
      << " winogradGemmCoopmat2AccF16SelectionRank=" << config.winogradGemmCoopmat2AccF16SelectionRank
      << " gemmStridedCoopmat2AccF16SelectionRank=" << config.gemmStridedCoopmat2AccF16SelectionRank << "\n";
  out << "#winogradCoopmat2AccF16Tile\n";
  out << "coopmat2AccF16BlockSize=" << config.coopmat2AccF16BlockSize << " coopmat2AccF16BM=" << config.coopmat2AccF16BM
      << " coopmat2AccF16BN=" << config.coopmat2AccF16BN << " coopmat2AccF16BK=" << config.coopmat2AccF16BK << "\n";
  out << "#stridedCoopmat2AccF16Tile\n";
  out << "stridedCoopmat2AccF16BlockSize=" << config.stridedCoopmat2AccF16BlockSize
      << " stridedCoopmat2AccF16BM=" << config.stridedCoopmat2AccF16BM
      << " stridedCoopmat2AccF16BN=" << config.stridedCoopmat2AccF16BN
      << " stridedCoopmat2AccF16BK=" << config.stridedCoopmat2AccF16BK << "\n";
  finishAtomicSave(filename, tmpFilename, out);
}

// Parse "key=value key=value ..." into a single config field by name.
namespace {

  void fillFromDesc(
    const string& filename,
    const string& desc,
    VulkanTuneParams& config,
    bool& sawTunerValueValid,
    bool& sawSelectionRank) {
    istringstream in(desc);
    string token;
    while(in >> token) {
      size_t eq = token.find('=');
      if(eq == string::npos)
        throw IOError("VulkanTuneParams::load: bad token '" + token + "' in " + filename);
      string key = token.substr(0, eq);
      string valueStr = token.substr(eq + 1);
      if(key == "winogradTransformTunedLayout") {
        config.winogradTransformTunedLayout = valueStr;
        continue;
      }
      if(key == "gemmVariantFilterTunedSignature") {
        config.gemmVariantFilterTunedSignature = valueStr;
        continue;
      }
      int value = Global::stringToInt(valueStr);
      if(key == "winogradGemmM")
        config.winogradGemmM = value;
      else if(key == "winograd3x3OutTile")
        config.winograd3x3OutTile = value;
      else if(key == "winogradGemmN")
        config.winogradGemmN = value;
      else if(key == "winogradGemmK")
        config.winogradGemmK = value;
      else if(key == "winogradGemmRN")
        config.winogradGemmRN = value;
      else if(key == "winogradGemmTiledSelectionRank") {
        config.winogradGemmTiledSelectionRank = value;
        sawSelectionRank = true;
      } else if(key == "enableWinogradGemmTiledFp16Compute")
        config.enableWinogradGemmTiledFp16Compute = value;
      else if(key == "enableGemmStridedTiledFp16Compute")
        config.enableGemmStridedTiledFp16Compute = value;
      else if(key == "winogradGemmTiledFp16ComputeTunerValueValid") {
        config.winogradGemmTiledFp16ComputeTunerValueValid = value;
        sawTunerValueValid = true;
      } else if(key == "gemmStridedTiledFp16ComputeTunerValueValid") {
        config.gemmStridedTiledFp16ComputeTunerValueValid = value;
        sawTunerValueValid = true;
      } else if(key == "winogradGemmTiledFp16ComputeSelectionRank") {
        config.winogradGemmTiledFp16ComputeSelectionRank = value;
        sawSelectionRank = true;
      } else if(key == "gemmStridedTiledFp16ComputeSelectionRank") {
        config.gemmStridedTiledFp16ComputeSelectionRank = value;
        sawSelectionRank = true;
      } else if(key == "winogradGemmFp16ComputeM")
        config.winogradGemmFp16ComputeM = value;
      else if(key == "winogradGemmFp16ComputeN")
        config.winogradGemmFp16ComputeN = value;
      else if(key == "winogradGemmFp16ComputeK")
        config.winogradGemmFp16ComputeK = value;
      else if(key == "winogradGemmFp16ComputeRN")
        config.winogradGemmFp16ComputeRN = value;
      else if(key == "attnBlockQ")
        config.attnBlockQ = value;
      else if(key == "attnBlockKV")
        config.attnBlockKV = value;
      else if(key == "attnQPerThread")
        config.attnQPerThread = value;
      else if(key == "winogradTransformLocalSizeX")
        config.winogradTransformLocalSizeX = value;
      else if(key == "winogradTransformLocalSizeY")
        config.winogradTransformLocalSizeY = value;
      else if(key == "winogradBNActTransformLocalSizeX")
        config.winogradBNActTransformLocalSizeX = value;
      else if(key == "winogradBNActTransformLocalSizeY")
        config.winogradBNActTransformLocalSizeY = value;
      else if(key == "winogradUntransformLocalSizeX")
        config.winogradUntransformLocalSizeX = value;
      else if(key == "winogradUntransformLocalSizeY")
        config.winogradUntransformLocalSizeY = value;
      else if(key == "gemmStridedTiledLocalSizeX")
        config.gemmStridedTiledLocalSizeX = value;
      else if(key == "gemmStridedTiledLocalSizeY")
        config.gemmStridedTiledLocalSizeY = value;
      else if(key == "gemmStridedTiledTileK")
        config.gemmStridedTiledTileK = value;
      else if(key == "gemmStridedTiledRN")
        config.gemmStridedTiledRN = value;
      else if(key == "gemmStridedTiledSelectionRank") {
        config.gemmStridedTiledSelectionRank = value;
        sawSelectionRank = true;
      } else if(key == "gemmStridedFp16ComputeLocalSizeX")
        config.gemmStridedFp16ComputeLocalSizeX = value;
      else if(key == "gemmStridedFp16ComputeLocalSizeY")
        config.gemmStridedFp16ComputeLocalSizeY = value;
      else if(key == "gemmStridedFp16ComputeTileK")
        config.gemmStridedFp16ComputeTileK = value;
      else if(key == "gemmStridedFp16ComputeRN")
        config.gemmStridedFp16ComputeRN = value;
      else if(key == "gemmDirectLocalSizeX")
        config.gemmDirectLocalSizeX = value;
      else if(key == "gemmDirectLocalSizeY")
        config.gemmDirectLocalSizeY = value;
      else if(key == "gpoolXystride")
        config.gpoolXystride = value;
      else if(key == "valueHeadPoolXystride")
        config.valueHeadPoolXystride = value;
      else if(key == "spatialRMSNormTile")
        config.spatialRMSNormTile = value;
      else if(key == "enableWinogradGemmDot2")
        config.enableWinogradGemmDot2 = value;
      else if(key == "enableGemmStridedDot2")
        config.enableGemmStridedDot2 = value;
      else if(key == "enableWinogradGemmDot2AccF16")
        config.enableWinogradGemmDot2AccF16 = value;
      else if(key == "enableGemmStridedDot2AccF16")
        config.enableGemmStridedDot2AccF16 = value;
      else if(key == "winogradGemmDot2TunerValueValid") {
        config.winogradGemmDot2TunerValueValid = value;
        sawTunerValueValid = true;
      } else if(key == "gemmStridedDot2TunerValueValid") {
        config.gemmStridedDot2TunerValueValid = value;
        sawTunerValueValid = true;
      } else if(key == "winogradGemmDot2AccF16TunerValueValid") {
        config.winogradGemmDot2AccF16TunerValueValid = value;
        sawTunerValueValid = true;
      } else if(key == "gemmStridedDot2AccF16TunerValueValid") {
        config.gemmStridedDot2AccF16TunerValueValid = value;
        sawTunerValueValid = true;
      } else if(key == "winogradGemmDot2SelectionRank") {
        config.winogradGemmDot2SelectionRank = value;
        sawSelectionRank = true;
      } else if(key == "gemmStridedDot2SelectionRank") {
        config.gemmStridedDot2SelectionRank = value;
        sawSelectionRank = true;
      } else if(key == "winogradGemmDot2AccF16SelectionRank") {
        config.winogradGemmDot2AccF16SelectionRank = value;
        sawSelectionRank = true;
      } else if(key == "gemmStridedDot2AccF16SelectionRank") {
        config.gemmStridedDot2AccF16SelectionRank = value;
        sawSelectionRank = true;
      } else if(key == "dot2BlockSize")
        config.dot2BlockSize = value;
      else if(key == "dot2BM")
        config.dot2BM = value;
      else if(key == "dot2BN")
        config.dot2BN = value;
      else if(key == "dot2WM")
        config.dot2WM = value;
      else if(key == "dot2WN")
        config.dot2WN = value;
      else if(key == "dot2WMIter")
        config.dot2WMIter = value;
      else if(key == "dot2TM")
        config.dot2TM = value;
      else if(key == "dot2TN")
        config.dot2TN = value;
      else if(key == "dot2Warp")
        config.dot2Warp = value;
      else if(key == "stridedDot2BlockSize")
        config.stridedDot2BlockSize = value;
      else if(key == "stridedDot2BM")
        config.stridedDot2BM = value;
      else if(key == "stridedDot2BN")
        config.stridedDot2BN = value;
      else if(key == "stridedDot2WM")
        config.stridedDot2WM = value;
      else if(key == "stridedDot2WN")
        config.stridedDot2WN = value;
      else if(key == "stridedDot2WMIter")
        config.stridedDot2WMIter = value;
      else if(key == "stridedDot2TM")
        config.stridedDot2TM = value;
      else if(key == "stridedDot2TN")
        config.stridedDot2TN = value;
      else if(key == "stridedDot2Warp")
        config.stridedDot2Warp = value;
      else if(key == "dot2AccF16BlockSize")
        config.dot2AccF16BlockSize = value;
      else if(key == "dot2AccF16BM")
        config.dot2AccF16BM = value;
      else if(key == "dot2AccF16BN")
        config.dot2AccF16BN = value;
      else if(key == "dot2AccF16WM")
        config.dot2AccF16WM = value;
      else if(key == "dot2AccF16WN")
        config.dot2AccF16WN = value;
      else if(key == "dot2AccF16WMIter")
        config.dot2AccF16WMIter = value;
      else if(key == "dot2AccF16TM")
        config.dot2AccF16TM = value;
      else if(key == "dot2AccF16TN")
        config.dot2AccF16TN = value;
      else if(key == "dot2AccF16Warp")
        config.dot2AccF16Warp = value;
      else if(key == "stridedDot2AccF16BlockSize")
        config.stridedDot2AccF16BlockSize = value;
      else if(key == "stridedDot2AccF16BM")
        config.stridedDot2AccF16BM = value;
      else if(key == "stridedDot2AccF16BN")
        config.stridedDot2AccF16BN = value;
      else if(key == "stridedDot2AccF16WM")
        config.stridedDot2AccF16WM = value;
      else if(key == "stridedDot2AccF16WN")
        config.stridedDot2AccF16WN = value;
      else if(key == "stridedDot2AccF16WMIter")
        config.stridedDot2AccF16WMIter = value;
      else if(key == "stridedDot2AccF16TM")
        config.stridedDot2AccF16TM = value;
      else if(key == "stridedDot2AccF16TN")
        config.stridedDot2AccF16TN = value;
      else if(key == "stridedDot2AccF16Warp")
        config.stridedDot2AccF16Warp = value;
      else if(key == "enableWinogradGemmCoopmat")
        config.enableWinogradGemmCoopmat = value;
      else if(key == "enableGemmStridedCoopmat")
        config.enableGemmStridedCoopmat = value;
      else if(key == "winogradGemmCoopmatTunerValueValid") {
        config.winogradGemmCoopmatTunerValueValid = value;
        sawTunerValueValid = true;
      } else if(key == "gemmStridedCoopmatTunerValueValid") {
        config.gemmStridedCoopmatTunerValueValid = value;
        sawTunerValueValid = true;
      } else if(key == "winogradGemmCoopmatSelectionRank") {
        config.winogradGemmCoopmatSelectionRank = value;
        sawSelectionRank = true;
      } else if(key == "gemmStridedCoopmatSelectionRank") {
        config.gemmStridedCoopmatSelectionRank = value;
        sawSelectionRank = true;
      } else if(key == "coopmatBlockSize")
        config.coopmatBlockSize = value;
      else if(key == "coopmatBM")
        config.coopmatBM = value;
      else if(key == "coopmatBN")
        config.coopmatBN = value;
      else if(key == "coopmatBK")
        config.coopmatBK = value;
      else if(key == "coopmatWM")
        config.coopmatWM = value;
      else if(key == "coopmatWN")
        config.coopmatWN = value;
      else if(key == "coopmatTM")
        config.coopmatTM = value;
      else if(key == "coopmatTN")
        config.coopmatTN = value;
      else if(key == "coopmatTK")
        config.coopmatTK = value;
      else if(key == "coopmatWarp")
        config.coopmatWarp = value;
      else if(key == "stridedCoopmatBlockSize")
        config.stridedCoopmatBlockSize = value;
      else if(key == "stridedCoopmatBM")
        config.stridedCoopmatBM = value;
      else if(key == "stridedCoopmatBN")
        config.stridedCoopmatBN = value;
      else if(key == "stridedCoopmatBK")
        config.stridedCoopmatBK = value;
      else if(key == "stridedCoopmatWM")
        config.stridedCoopmatWM = value;
      else if(key == "stridedCoopmatWN")
        config.stridedCoopmatWN = value;
      else if(key == "stridedCoopmatTM")
        config.stridedCoopmatTM = value;
      else if(key == "stridedCoopmatTN")
        config.stridedCoopmatTN = value;
      else if(key == "stridedCoopmatTK")
        config.stridedCoopmatTK = value;
      else if(key == "stridedCoopmatWarp")
        config.stridedCoopmatWarp = value;
      else if(key == "enableWinogradGemmCoopmatAccF16")
        config.enableWinogradGemmCoopmatAccF16 = value;
      else if(key == "enableGemmStridedCoopmatAccF16")
        config.enableGemmStridedCoopmatAccF16 = value;
      else if(key == "winogradGemmCoopmatAccF16TunerValueValid") {
        config.winogradGemmCoopmatAccF16TunerValueValid = value;
        sawTunerValueValid = true;
      } else if(key == "gemmStridedCoopmatAccF16TunerValueValid") {
        config.gemmStridedCoopmatAccF16TunerValueValid = value;
        sawTunerValueValid = true;
      } else if(key == "winogradGemmCoopmatAccF16SelectionRank") {
        config.winogradGemmCoopmatAccF16SelectionRank = value;
        sawSelectionRank = true;
      } else if(key == "gemmStridedCoopmatAccF16SelectionRank") {
        config.gemmStridedCoopmatAccF16SelectionRank = value;
        sawSelectionRank = true;
      } else if(key == "coopmatAccF16BlockSize")
        config.coopmatAccF16BlockSize = value;
      else if(key == "coopmatAccF16BM")
        config.coopmatAccF16BM = value;
      else if(key == "coopmatAccF16BN")
        config.coopmatAccF16BN = value;
      else if(key == "coopmatAccF16BK")
        config.coopmatAccF16BK = value;
      else if(key == "coopmatAccF16WM")
        config.coopmatAccF16WM = value;
      else if(key == "coopmatAccF16WN")
        config.coopmatAccF16WN = value;
      else if(key == "coopmatAccF16TM")
        config.coopmatAccF16TM = value;
      else if(key == "coopmatAccF16TN")
        config.coopmatAccF16TN = value;
      else if(key == "coopmatAccF16TK")
        config.coopmatAccF16TK = value;
      else if(key == "coopmatAccF16Warp")
        config.coopmatAccF16Warp = value;
      else if(key == "stridedCoopmatAccF16BlockSize")
        config.stridedCoopmatAccF16BlockSize = value;
      else if(key == "stridedCoopmatAccF16BM")
        config.stridedCoopmatAccF16BM = value;
      else if(key == "stridedCoopmatAccF16BN")
        config.stridedCoopmatAccF16BN = value;
      else if(key == "stridedCoopmatAccF16BK")
        config.stridedCoopmatAccF16BK = value;
      else if(key == "stridedCoopmatAccF16WM")
        config.stridedCoopmatAccF16WM = value;
      else if(key == "stridedCoopmatAccF16WN")
        config.stridedCoopmatAccF16WN = value;
      else if(key == "stridedCoopmatAccF16TM")
        config.stridedCoopmatAccF16TM = value;
      else if(key == "stridedCoopmatAccF16TN")
        config.stridedCoopmatAccF16TN = value;
      else if(key == "stridedCoopmatAccF16TK")
        config.stridedCoopmatAccF16TK = value;
      else if(key == "stridedCoopmatAccF16Warp")
        config.stridedCoopmatAccF16Warp = value;
      else if(key == "enableWinogradGemmCoopmat2")
        config.enableWinogradGemmCoopmat2 = value;
      else if(key == "enableGemmStridedCoopmat2")
        config.enableGemmStridedCoopmat2 = value;
      else if(key == "winogradGemmCoopmat2TunerValueValid") {
        config.winogradGemmCoopmat2TunerValueValid = value;
        sawTunerValueValid = true;
      } else if(key == "gemmStridedCoopmat2TunerValueValid") {
        config.gemmStridedCoopmat2TunerValueValid = value;
        sawTunerValueValid = true;
      } else if(key == "winogradGemmCoopmat2SelectionRank") {
        config.winogradGemmCoopmat2SelectionRank = value;
        sawSelectionRank = true;
      } else if(key == "gemmStridedCoopmat2SelectionRank") {
        config.gemmStridedCoopmat2SelectionRank = value;
        sawSelectionRank = true;
      } else if(key == "coopmat2BlockSize")
        config.coopmat2BlockSize = value;
      else if(key == "coopmat2BM")
        config.coopmat2BM = value;
      else if(key == "coopmat2BN")
        config.coopmat2BN = value;
      else if(key == "coopmat2BK")
        config.coopmat2BK = value;
      else if(key == "stridedCoopmat2BlockSize")
        config.stridedCoopmat2BlockSize = value;
      else if(key == "stridedCoopmat2BM")
        config.stridedCoopmat2BM = value;
      else if(key == "stridedCoopmat2BN")
        config.stridedCoopmat2BN = value;
      else if(key == "stridedCoopmat2BK")
        config.stridedCoopmat2BK = value;
      else if(key == "enableWinogradGemmCoopmat2AccF16")
        config.enableWinogradGemmCoopmat2AccF16 = value;
      else if(key == "enableGemmStridedCoopmat2AccF16")
        config.enableGemmStridedCoopmat2AccF16 = value;
      else if(key == "winogradGemmCoopmat2AccF16TunerValueValid") {
        config.winogradGemmCoopmat2AccF16TunerValueValid = value;
        sawTunerValueValid = true;
      } else if(key == "gemmStridedCoopmat2AccF16TunerValueValid") {
        config.gemmStridedCoopmat2AccF16TunerValueValid = value;
        sawTunerValueValid = true;
      } else if(key == "winogradGemmCoopmat2AccF16SelectionRank") {
        config.winogradGemmCoopmat2AccF16SelectionRank = value;
        sawSelectionRank = true;
      } else if(key == "gemmStridedCoopmat2AccF16SelectionRank") {
        config.gemmStridedCoopmat2AccF16SelectionRank = value;
        sawSelectionRank = true;
      } else if(key == "coopmat2AccF16BlockSize")
        config.coopmat2AccF16BlockSize = value;
      else if(key == "coopmat2AccF16BM")
        config.coopmat2AccF16BM = value;
      else if(key == "coopmat2AccF16BN")
        config.coopmat2AccF16BN = value;
      else if(key == "coopmat2AccF16BK")
        config.coopmat2AccF16BK = value;
      else if(key == "stridedCoopmat2AccF16BlockSize")
        config.stridedCoopmat2AccF16BlockSize = value;
      else if(key == "stridedCoopmat2AccF16BM")
        config.stridedCoopmat2AccF16BM = value;
      else if(key == "stridedCoopmat2AccF16BN")
        config.stridedCoopmat2AccF16BN = value;
      else if(key == "stridedCoopmat2AccF16BK")
        config.stridedCoopmat2AccF16BK = value;
      else
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
  if(filteredLines.size() != 31)
    throw IOError("VulkanTuneParams::load: unexpected number of parameter lines in " + filename);

  VulkanTuneParams config;
  bool sawTunerValueValid = false;
  bool sawSelectionRank = false;
  for(size_t i = 1; i < filteredLines.size(); i++)
    fillFromDesc(filename, filteredLines[i], config, sawTunerValueValid, sawSelectionRank);
  if(!sawTunerValueValid)
    config.inferLegacyGemmVariantTuningValidity();
  if(!sawSelectionRank)
    config.inferLegacyGemmVariantSelectionRanks();
  config.normalizeGemmVariantSelections();
  return config;
}

// ============================================================================
// File path / naming
// ============================================================================

// Returns true if the model has a transformer attention block anywhere in its
// trunk (including inside nested bottleneck blocks).
namespace {

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

}  // namespace

string VulkanTuner::defaultDirectory(bool makeDir, const string& homeDataDirOverride) {
  string dir = HomeData::getHomeDataDir(true, homeDataDirOverride);
  dir += "/vulkantuning";
  if(makeDir)
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
  auto appendTuple = [&out, sep](std::initializer_list<int32_t> vals) {
    bool first = true;
    for(int32_t v: vals) {
      if(!first)
        out << sep;
      out << v;
      first = false;
    }
  };

  out << "winogradGemm=";
  appendTuple({config.winogradGemmM, config.winogradGemmN, config.winogradGemmK, config.winogradGemmRN});
  out << " winogradGemmFp16Compute=" << (config.enableWinogradGemmTiledFp16Compute != 0 ? "on" : "off")
      << " winogradGemmFp16ComputeTile=";
  appendTuple(
    {config.winogradGemmFp16ComputeM,
     config.winogradGemmFp16ComputeN,
     config.winogradGemmFp16ComputeK,
     config.winogradGemmFp16ComputeRN});
  out << " winograd3x3OutTile=" << config.winograd3x3OutTile;
  out << " winogradTransformWG=";
  appendTuple({config.winogradTransformLocalSizeX, config.winogradTransformLocalSizeY});
  out << " winogradBNActTransformWG=";
  appendTuple({config.winogradBNActTransformLocalSizeX, config.winogradBNActTransformLocalSizeY});
  out << " winogradUntransformWG=";
  appendTuple({config.winogradUntransformLocalSizeX, config.winogradUntransformLocalSizeY});
  out << " gemmStridedTiled=";
  appendTuple(
    {config.gemmStridedTiledLocalSizeX,
     config.gemmStridedTiledLocalSizeY,
     config.gemmStridedTiledTileK,
     config.gemmStridedTiledRN});
  out << " gemmStridedFp16Compute=" << (config.enableGemmStridedTiledFp16Compute != 0 ? "on" : "off")
      << " gemmStridedFp16ComputeTile=";
  appendTuple(
    {config.gemmStridedFp16ComputeLocalSizeX,
     config.gemmStridedFp16ComputeLocalSizeY,
     config.gemmStridedFp16ComputeTileK,
     config.gemmStridedFp16ComputeRN});
  out << " winogradDot2=" << (config.enableWinogradGemmDot2 != 0 ? "on" : "off")
      << " winogradDot2AccF16=" << (config.enableWinogradGemmDot2AccF16 != 0 ? "on" : "off") << " winogradDot2Tile=";
  appendTuple(
    {config.dot2BlockSize,
     config.dot2BM,
     config.dot2BN,
     config.dot2WM,
     config.dot2WN,
     config.dot2WMIter,
     config.dot2TM,
     config.dot2TN,
     config.dot2Warp});
  out << " winogradDot2AccF16Tile=";
  appendTuple(
    {config.dot2AccF16BlockSize,
     config.dot2AccF16BM,
     config.dot2AccF16BN,
     config.dot2AccF16WM,
     config.dot2AccF16WN,
     config.dot2AccF16WMIter,
     config.dot2AccF16TM,
     config.dot2AccF16TN,
     config.dot2AccF16Warp});
  out << " gemmStridedDot2=" << (config.enableGemmStridedDot2 != 0 ? "on" : "off")
      << " gemmStridedDot2AccF16=" << (config.enableGemmStridedDot2AccF16 != 0 ? "on" : "off")
      << " gemmStridedDot2Tile=";
  appendTuple(
    {config.stridedDot2BlockSize,
     config.stridedDot2BM,
     config.stridedDot2BN,
     config.stridedDot2WM,
     config.stridedDot2WN,
     config.stridedDot2WMIter,
     config.stridedDot2TM,
     config.stridedDot2TN,
     config.stridedDot2Warp});
  out << " gemmStridedDot2AccF16Tile=";
  appendTuple(
    {config.stridedDot2AccF16BlockSize,
     config.stridedDot2AccF16BM,
     config.stridedDot2AccF16BN,
     config.stridedDot2AccF16WM,
     config.stridedDot2AccF16WN,
     config.stridedDot2AccF16WMIter,
     config.stridedDot2AccF16TM,
     config.stridedDot2AccF16TN,
     config.stridedDot2AccF16Warp});
  out << " winogradCoopmat=" << (config.enableWinogradGemmCoopmat != 0 ? "on" : "off") << " winogradCoopmatTile=";
  appendTuple(
    {config.coopmatBlockSize,
     config.coopmatBM,
     config.coopmatBN,
     config.coopmatBK,
     config.coopmatWM,
     config.coopmatWN,
     config.coopmatTM,
     config.coopmatTN,
     config.coopmatTK,
     config.coopmatWarp});
  out << " gemmStridedCoopmat=" << (config.enableGemmStridedCoopmat != 0 ? "on" : "off") << " gemmStridedCoopmatTile=";
  appendTuple(
    {config.stridedCoopmatBlockSize,
     config.stridedCoopmatBM,
     config.stridedCoopmatBN,
     config.stridedCoopmatBK,
     config.stridedCoopmatWM,
     config.stridedCoopmatWN,
     config.stridedCoopmatTM,
     config.stridedCoopmatTN,
     config.stridedCoopmatTK,
     config.stridedCoopmatWarp});
  out << " winogradCoopmatAccF16=" << (config.enableWinogradGemmCoopmatAccF16 != 0 ? "on" : "off")
      << " winogradCoopmatAccF16Tile=";
  appendTuple(
    {config.coopmatAccF16BlockSize,
     config.coopmatAccF16BM,
     config.coopmatAccF16BN,
     config.coopmatAccF16BK,
     config.coopmatAccF16WM,
     config.coopmatAccF16WN,
     config.coopmatAccF16TM,
     config.coopmatAccF16TN,
     config.coopmatAccF16TK,
     config.coopmatAccF16Warp});
  out << " gemmStridedCoopmatAccF16=" << (config.enableGemmStridedCoopmatAccF16 != 0 ? "on" : "off")
      << " gemmStridedCoopmatAccF16Tile=";
  appendTuple(
    {config.stridedCoopmatAccF16BlockSize,
     config.stridedCoopmatAccF16BM,
     config.stridedCoopmatAccF16BN,
     config.stridedCoopmatAccF16BK,
     config.stridedCoopmatAccF16WM,
     config.stridedCoopmatAccF16WN,
     config.stridedCoopmatAccF16TM,
     config.stridedCoopmatAccF16TN,
     config.stridedCoopmatAccF16TK,
     config.stridedCoopmatAccF16Warp});
  out << " winogradCoopmat2=" << (config.enableWinogradGemmCoopmat2 != 0 ? "on" : "off") << " winogradCoopmat2Tile=";
  appendTuple({config.coopmat2BlockSize, config.coopmat2BM, config.coopmat2BN, config.coopmat2BK});
  out << " gemmStridedCoopmat2=" << (config.enableGemmStridedCoopmat2 != 0 ? "on" : "off")
      << " gemmStridedCoopmat2Tile=";
  appendTuple(
    {config.stridedCoopmat2BlockSize, config.stridedCoopmat2BM, config.stridedCoopmat2BN, config.stridedCoopmat2BK});
  out << " winogradCoopmat2AccF16=" << (config.enableWinogradGemmCoopmat2AccF16 != 0 ? "on" : "off")
      << " winogradCoopmat2AccF16Tile=";
  appendTuple(
    {config.coopmat2AccF16BlockSize, config.coopmat2AccF16BM, config.coopmat2AccF16BN, config.coopmat2AccF16BK});
  out << " gemmStridedCoopmat2AccF16=" << (config.enableGemmStridedCoopmat2AccF16 != 0 ? "on" : "off")
      << " gemmStridedCoopmat2AccF16Tile=";
  appendTuple(
    {config.stridedCoopmat2AccF16BlockSize,
     config.stridedCoopmat2AccF16BM,
     config.stridedCoopmat2AccF16BN,
     config.stridedCoopmat2AccF16BK});
  out << " gemmDirect=";
  appendTuple({config.gemmDirectLocalSizeX, config.gemmDirectLocalSizeY});
  out << " gpool=" << config.gpoolXystride << " valueHeadPool=" << config.valueHeadPoolXystride;
  bool hasTransformer = (modelDesc == nullptr) || modelHasTransformer(modelDesc);
  if(hasTransformer) {
    out << " attnTiled=";
    appendTuple({config.attnBlockQ, config.attnBlockKV, config.attnQPerThread});
    out << " spatialRMSNorm=" << config.spatialRMSNormTile;
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
    TCLAP::SwitchArg fullArg("", "full", "Test more possible configurations");
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
    // The generic tuner uses fixed candidate sets; --full is accepted for CLI
    // compatibility but no longer widens the search.
    (void)full;

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
  ComputeContext* ctx = NeuralNet::createComputeContextForVulkanTuner(
    gpuIdxs, &logger, nnXLen, nnYLen, homeDataDirOverride, enabled_t::Auto);
  NeuralNet::tuneVulkanComputeContext(
    ctx, &modelDesc, batchSize, winograd3x3OutTile, benchIters, verboseTuner, outputFileFromArg);
  NeuralNet::freeComputeContext(ctx);
  return 0;
}

// Tuning machinery (Vulkan-API-dependent): TuningContext lifecycle, the search
// loop, the kernel factory, and the VulkanKernels::*::bench definitions.
// Folded in from the former vulkantunable.cpp; TuningContext::makeInputBufFP/
// downloadFloatsFP live in vulkanbackend.cpp where its base class does.
// ============================================================================

namespace VulkanTuner {

  // ---- Shared helpers ----

  // Deterministic pseudo-random fill in [-0.5, 0.5). Seeded from problem dims so
  // a candidate's output is reproducible across configs (RMSE comparison is valid).
  // Visible across this TU (used by VulkanKernels::*::bench definitions below the
  // VulkanTuner namespace block).
  static void fillRandom(vector<float>& v, uint32_t seed) {
    for(auto& x: v) {
      seed = seed * 1664525u + 1013904223u;
      x = ((seed >> 8) & 0xFFFF) / 65536.0f - 0.5f;
    }
  }

  namespace {

    // Normalized RMSE of candidate output vs reference. +inf on shape/NaN mismatch.
    double normalizedRmse(const vector<float>& ref, const vector<float>& got) {
      if(ref.size() != got.size())
        return std::numeric_limits<double>::infinity();
      double squerr = 0.0, sqmag = 0.0;
      for(size_t i = 0; i < ref.size(); i++) {
        if(!isfinite(ref[i]) || !isfinite(got[i]))
          return std::numeric_limits<double>::infinity();
        double diff = (double)ref[i] - (double)got[i];
        squerr += diff * diff;
        sqmag += (double)ref[i] * (double)ref[i];
      }
      return sqrt(squerr / (sqmag + 1e-30));
    }

    constexpr int GEMM_STRIDED_MODE_MAX_CASES = 6;
    constexpr int WINOGRAD_MODE_MAX_CASES = 6;
    constexpr double MODE_SELECT_MIN_SPEEDUP = 1.05;

    struct GemmStridedModeCase {
      int M;
      int N;
      int K;
      int occurrences;
      double weightedWork;
    };

    static void addGemmStridedModeCase(std::vector<GemmStridedModeCase>& cases, int M, int rawN, int rawK) {
      if(M <= 0 || rawN <= 0 || rawK <= 0)
        return;
      int N = roundUpToMultipleInt(rawN, 8);
      int K = rawK;
      for(auto& c: cases) {
        if(c.M == M && c.N == N && c.K == K) {
          c.occurrences += 1;
          return;
        }
      }
      cases.push_back({M, N, K, 1, 0.0});
    }

    static void collectTransformerGemmStridedCases(
      const std::vector<std::pair<int, unique_ptr_void>>& blocks,
      int paddedSpatialSize,
      std::vector<GemmStridedModeCase>& outCases) {
      for(const auto& kv: blocks) {
        if(kv.first == TRANSFORMER_ATTENTION_BLOCK_KIND) {
          const auto* attn = static_cast<const TransformerAttentionDesc*>(kv.second.get());
          addGemmStridedModeCase(outCases, paddedSpatialSize, attn->qProj.outChannels, attn->qProj.inChannels);
          addGemmStridedModeCase(outCases, paddedSpatialSize, attn->kProj.outChannels, attn->kProj.inChannels);
          addGemmStridedModeCase(outCases, paddedSpatialSize, attn->vProj.outChannels, attn->vProj.inChannels);
          addGemmStridedModeCase(outCases, paddedSpatialSize, attn->outProj.outChannels, attn->outProj.inChannels);
        } else if(kv.first == TRANSFORMER_FFN_BLOCK_KIND) {
          const auto* ffn = static_cast<const TransformerFFNDesc*>(kv.second.get());
          addGemmStridedModeCase(outCases, paddedSpatialSize, ffn->linear1.outChannels, ffn->linear1.inChannels);
          if(ffn->useSwiGLU)
            addGemmStridedModeCase(
              outCases, paddedSpatialSize, ffn->linearGate.outChannels, ffn->linearGate.inChannels);
          addGemmStridedModeCase(outCases, paddedSpatialSize, ffn->linear2.outChannels, ffn->linear2.inChannels);
        } else if(kv.first == NESTED_BOTTLENECK_BLOCK_KIND) {
          const auto* nbt = static_cast<const NestedBottleneckResidualBlockDesc*>(kv.second.get());
          collectTransformerGemmStridedCases(nbt->blocks, paddedSpatialSize, outCases);
        }
      }
    }

    static std::vector<GemmStridedModeCase>
    collectGemmStridedModeCases(const ModelDesc* modelDesc, int nnXLen, int nnYLen) {
      std::vector<GemmStridedModeCase> cases;
      if(modelDesc == nullptr)
        return cases;
      int paddedSpatialSize = roundUpToMultipleInt(nnXLen * nnYLen, VulkanKernels::VULKAN_SPATIAL_ALIGN);

      modelDesc->iterConvLayers([&](const ConvLayerDesc& conv) {
        if(conv.convXSize == 1 && conv.convYSize == 1)
          addGemmStridedModeCase(cases, paddedSpatialSize, conv.outChannels, conv.inChannels);
      });
      collectTransformerGemmStridedCases(modelDesc->trunk.blocks, paddedSpatialSize, cases);

      for(auto& c: cases)
        c.weightedWork = (double)c.occurrences * (double)c.N * (double)c.K;

      std::sort(cases.begin(), cases.end(), [](const GemmStridedModeCase& a, const GemmStridedModeCase& b) {
        if(a.weightedWork != b.weightedWork)
          return a.weightedWork > b.weightedWork;
        if(a.N != b.N)
          return a.N > b.N;
        if(a.K != b.K)
          return a.K > b.K;
        return a.M > b.M;
      });
      if(cases.size() > GEMM_STRIDED_MODE_MAX_CASES)
        cases.resize(GEMM_STRIDED_MODE_MAX_CASES);
      return cases;
    }

    struct WinogradModeCase {
      int M;
      int N;
      int K;
      int numBatches;
      int occurrences;
      double weightedWork;
    };

    static void addWinogradModeCase(std::vector<WinogradModeCase>& cases, int M, int N, int K, int numBatches) {
      if(M <= 0 || N <= 0 || K <= 0 || numBatches <= 0)
        return;
      for(auto& c: cases) {
        if(c.M == M && c.N == N && c.K == K && c.numBatches == numBatches) {
          c.occurrences += 1;
          return;
        }
      }
      cases.push_back({M, N, K, numBatches, 1, 0.0});
    }

    static std::vector<WinogradModeCase> collectWinogradModeCases(
      const ModelDesc* modelDesc,
      int batchSize,
      int nnXLen,
      int nnYLen,
      const VulkanTuneParams& cfg) {
      std::vector<WinogradModeCase> cases;
      if(modelDesc == nullptr)
        return cases;

      modelDesc->iterConvLayers([&](const ConvLayerDesc& conv) {
        bool is3x3 = conv.convXSize == 3 && conv.convYSize == 3;
        bool is5x5 = conv.convXSize == 5 && conv.convYSize == 5;
        if(!is3x3 && !is5x5)
          return;
        if(conv.dilationX != 1 || conv.dilationY != 1)
          return;
        int outTile = is3x3 ? winograd3x3OutTileFor(cfg) : 2;
        int inTile = outTile + conv.convXSize - 1;
        int numTilesX = (nnXLen + outTile - 1) / outTile;
        int numTilesY = (nnYLen + outTile - 1) / outTile;
        int mBase = std::max(1, batchSize) * numTilesX * numTilesY;
        int M = mBase;
        int N = conv.outChannels;
        int K = conv.inChannels;
        int numBatches = inTile * inTile;
        addWinogradModeCase(cases, M, N, K, numBatches);
      });

      for(auto& c: cases)
        c.weightedWork = (double)c.occurrences * (double)c.numBatches * (double)c.M * (double)c.N * (double)c.K;

      std::sort(cases.begin(), cases.end(), [](const WinogradModeCase& a, const WinogradModeCase& b) {
        if(a.weightedWork != b.weightedWork)
          return a.weightedWork > b.weightedWork;
        if(a.numBatches != b.numBatches)
          return a.numBatches > b.numBatches;
        if(a.N != b.N)
          return a.N > b.N;
        if(a.K != b.K)
          return a.K > b.K;
        return a.M > b.M;
      });
      if(cases.size() > WINOGRAD_MODE_MAX_CASES)
        cases.resize(WINOGRAD_MODE_MAX_CASES);
      return cases;
    }

    static double dispatchWeightedTime(const KernelBench& b, int occurrences) {
      return (double)occurrences / b.kernelsPerSecond;
    }

    static void appendMatmulSizes(std::ostream& out, const std::vector<GemmStridedModeCase>& cases) {
      out << " matmul_sizes=[";
      for(size_t i = 0; i < cases.size(); i++) {
        const GemmStridedModeCase& c = cases[i];
        out << (i == 0 ? "" : "; ") << "A[" << c.M << "][" << c.K << "] * B[" << c.K << "][" << c.N << "]";
      }
      out << "]";
    }

    static void appendMatmulSizes(std::ostream& out, const std::vector<WinogradModeCase>& cases) {
      out << " matmul_sizes=[";
      for(size_t i = 0; i < cases.size(); i++) {
        const WinogradModeCase& c = cases[i];
        out << (i == 0 ? "" : "; ") << "A[" << c.M << "][" << c.K << "] * B[" << c.K << "][" << c.N << "]";
      }
      out << "]";
    }

    // A workgroup narrower than one subgroup leaves lanes idle every dispatch and
    // is not expected to win the lowest-time selection, so benching it is usually
    // pure waste. We prune those candidates from a sweep, but only when the sweep
    // can otherwise reach a full subgroup, so a kernel whose entire candidate set
    // is sub-subgroup by design (e.g. a reduction pinned below the warp width)
    // never has all its candidates pruned.
    //
    // effectiveSubgroupFloor returns the thread count below which a candidate may
    // be skipped, or 0 to disable pruning entirely. It walks the Cartesian product
    // of the sweep's params and returns subgroupSize only if some *valid* reachable
    // config (cfg.isValid() && kernel.validate) meets or exceeds it. A config that
    // fails validation cannot be the survivor that justifies pruning smaller ones.
    // Returns 0 when the device reports no subgroup size, the kernel opts out
    // (workgroupThreads()==0 for every config), or no valid config reaches a
    // subgroup.
    static uint32_t effectiveSubgroupFloor(
      const TunableKernel& kernel,
      const VulkanTuneParams& seed,
      ArrayView<TunableParam> params,
      const VkPhysicalDeviceLimits& limits,
      uint32_t subgroupSize) {
      if(subgroupSize == 0)
        return 0;
      size_t nParams = params.size();
      std::vector<size_t> idx(nParams, 0);
      bool done = (nParams == 0);
      bool anyReachesSubgroup = false;
      while(!done) {
        VulkanTuneParams cfg = seed;
        for(size_t i = 0; i < nParams; i++)
          cfg.*(params[i].field) = params[i].candidates[idx[i]];
        uint32_t threads = kernel.workgroupThreads(cfg);
        if(threads >= subgroupSize && cfg.isValid() && kernel.validate(cfg, limits)) {
          anyReachesSubgroup = true;
          break;
        }
        size_t pos = 0;
        while(pos < nParams) {
          idx[pos]++;
          if(idx[pos] < params[pos].candidates.size())
            break;
          idx[pos] = 0;
          pos++;
        }
        if(pos == nParams)
          done = true;
      }
      return anyReachesSubgroup ? subgroupSize : 0;
    }

    static bool workgroupBelowFloor(const TunableKernel& kernel, const VulkanTuneParams& cfg, uint32_t subgroupFloor) {
      if(subgroupFloor == 0)
        return false;
      uint32_t threads = kernel.workgroupThreads(cfg);
      return threads > 0 && threads < subgroupFloor;
    }

    static bool challengerWinsModeSelect(double challengerWeightedTime, double incumbentWeightedTime) {
      return challengerWeightedTime * MODE_SELECT_MIN_SPEEDUP < incumbentWeightedTime;
    }

    static void
    appendKernelPerf(std::ostream& out, const TunableKernel& kernel, const VulkanTuneParams& cfg, double kps) {
      out << " kps=" << kps;
      if(kps > 0.0 && std::isfinite(kps))
        out << " sec_per_kernel=" << (1.0 / kps);
      double tflops = kernel.estimatedTflops(cfg, kps);
      if(tflops > 0.0)
        out << " tflops=" << tflops;
    }

    static double aggregateTflops(double totalFlops, double totalSeconds) {
      return totalFlops > 0.0 && totalSeconds > 0.0 ? totalFlops / totalSeconds / 1e12 : 0.0;
    }

    static void appendTflopsField(std::ostream& out, string_view name, double tflops) {
      out << " " << name << "=";
      if(tflops > 0.0)
        out << tflops;
      else
        out << "n/a";
    }

    static void appendPositiveField(std::ostream& out, string_view name, double value) {
      out << " " << name << "=";
      if(value > 0.0 && std::isfinite(value))
        out << value;
      else
        out << "n/a";
    }

    struct ModeSelectStats {
      bool allOk = true;
      double weightedTime = 0.0;
      double weightedFlops = 0.0;

      double recordCase(bool ok, const KernelBench& bench, double flopsPerDispatch, int occurrences) {
        if(!ok) {
          allOk = false;
          return 0.0;
        }
        double caseWeightedTime = dispatchWeightedTime(bench, occurrences);
        weightedTime += caseWeightedTime;
        weightedFlops += flopsPerDispatch * (double)occurrences;
        return caseWeightedTime;
      }

      double tflops() const { return aggregateTflops(weightedFlops, weightedTime); }
    };

    static double modeSelectSpeedup(const ModeSelectStats& incumbent, const ModeSelectStats& challenger) {
      return incumbent.weightedTime > 0.0 && challenger.weightedTime > 0.0
               ? incumbent.weightedTime / challenger.weightedTime
               : 0.0;
    }

    template<typename CaseT, typename AppendExtraFieldsFn>
    static bool appendModeSelectResult(
      std::ostream& out,
      string_view gemmName,
      string_view incumbentName,
      string_view challengerName,
      const std::vector<CaseT>& cases,
      const ModeSelectStats& incumbent,
      const ModeSelectStats& challenger,
      double maxRmse,
      bool appendRequiredMinSpeedup,
      bool appendRejectReason,
      AppendExtraFieldsFn appendExtraFields) {
      bool challengerSelected = challenger.allOk && incumbent.allOk &&
                                challengerWinsModeSelect(challenger.weightedTime, incumbent.weightedTime);
      string_view winnerName = challengerSelected ? challengerName : incumbentName;
      string_view otherName = challengerSelected ? incumbentName : challengerName;
      const ModeSelectStats& winner = challengerSelected ? challenger : incumbent;
      const ModeSelectStats& other = challengerSelected ? incumbent : challenger;
      out << "VulkanTuner: " << gemmName << " mode-select " << (challengerSelected ? "selected" : "kept")
          << " winner=" << winnerName << " winner_weighted_time=" << winner.weightedTime << " challenger=" << otherName
          << " challenger_weighted_time=" << other.weightedTime;
      appendMatmulSizes(out, cases);
      appendExtraFields(out);
      if(appendRequiredMinSpeedup)
        out << " required_min_speedup=" << MODE_SELECT_MIN_SPEEDUP;
      appendTflopsField(out, "winner_tflops", winner.tflops());
      appendTflopsField(out, "challenger_tflops", other.tflops());
      if(!challengerSelected && appendRejectReason) {
        if(!incumbent.allOk)
          out << " reason=winner_failed_or_invalid";
        else if(!challenger.allOk)
          out << " reason=challenger_failed_or_invalid";
        else if(challenger.weightedTime < incumbent.weightedTime)
          out << " reason=below_min_speedup";
      }
      out << " max_rmse=" << maxRmse << endl;
      return challengerSelected;
    }

    constexpr size_t TUNER_CONFIRM_TOP_K = 6;
    constexpr size_t TUNER_CONFIRM_PREFIX_K = 6;
    constexpr int TUNER_CONFIRM_REPEATS = 3;

    static double medianValue(std::vector<double> samples) {
      if(samples.empty())
        return 0.0;
      std::sort(samples.begin(), samples.end());
      size_t mid = samples.size() / 2;
      if(samples.size() % 2 == 0)
        return 0.5 * (samples[mid - 1] + samples[mid]);
      return samples[mid];
    }

    template<typename BetterIndexFn>
    static std::vector<size_t>
    makeConfirmationCandidateIndexes(size_t candidateCount, size_t prefixCount, BetterIndexFn betterIndex) {
      std::vector<size_t> ranked(candidateCount);
      std::iota(ranked.begin(), ranked.end(), 0);
      std::stable_sort(ranked.begin(), ranked.end(), betterIndex);

      std::vector<size_t> selected;
      selected.reserve(std::min(candidateCount, TUNER_CONFIRM_TOP_K + prefixCount));
      auto addIfNew = [&](size_t idx) {
        if(!contains(selected, idx))
          selected.push_back(idx);
      };
      for(size_t i = 0; i < ranked.size() && i < TUNER_CONFIRM_TOP_K; i++)
        addIfNew(ranked[i]);
      // Prefix candidates are from the already-screened list, so invalid or
      // failed-to-bench curated defaults are not forced into confirmation.
      for(size_t i = 0; i < candidateCount && i < prefixCount; i++)
        addIfNew(i);
      return selected;
    }

    struct ConfirmationSample {
      bool ok;
      double metric;
      double tflops;
      bool rmseChecked;
      bool rmseOk;
    };

    struct ConfirmationResult {
      size_t candidateIndex;
      double metric;
      double tflops;
      size_t rmseChecked;
      size_t rmseOk;
      std::vector<double> metricSamples;
    };

    // Raw confirmation-sample logging is opt-in: it multiplies the length of every
    // confirm line, which is too noisy for the default "just tune my tiles" run.
    // Gated by KATAGO_VULKAN_VERBOSE_TUNER, read once per process (getenv walks the
    // environment each call), matching the KATAGO_VULKAN_PROFILE_KERNELS gate.
    static bool verboseTunerEnabled() {
      static const bool cached = []() {
        string_view v = getenvStringView("KATAGO_VULKAN_VERBOSE_TUNER");
        return !v.empty() && v[0] != '0';
      }();
      return cached;
    }

    static void appendSamplesField(std::ostream& out, string_view name, const std::vector<double>& samples) {
      if(!verboseTunerEnabled())
        return;
      out << " " << name << "=[";
      for(size_t i = 0; i < samples.size(); i++)
        out << (i ? "," : "") << samples[i];
      out << "]";
    }

    template<typename ConfirmOneFn>
    static std::vector<ConfirmationResult> confirmCandidateMetrics(
      const std::vector<size_t>& confirmIndexes,
      ConfirmOneFn confirmOne) {
      std::vector<std::vector<double>> metrics(confirmIndexes.size());
      std::vector<std::vector<double>> tflops(confirmIndexes.size());
      std::vector<size_t> rmseChecked(confirmIndexes.size(), 0);
      std::vector<size_t> rmseOk(confirmIndexes.size(), 0);
      for(int repeat = 0; repeat < TUNER_CONFIRM_REPEATS; repeat++) {
        for(size_t pos = 0; pos < confirmIndexes.size(); pos++) {
          ConfirmationSample sample = confirmOne(confirmIndexes[pos]);
          if(sample.rmseChecked)
            rmseChecked[pos]++;
          if(sample.rmseOk)
            rmseOk[pos]++;
          if(sample.ok) {
            metrics[pos].push_back(sample.metric);
            if(sample.tflops > 0.0)
              tflops[pos].push_back(sample.tflops);
          }
        }
      }

      std::vector<ConfirmationResult> results;
      results.reserve(confirmIndexes.size());
      for(size_t pos = 0; pos < confirmIndexes.size(); pos++) {
        results.push_back(
          {confirmIndexes[pos],
           medianValue(metrics[pos]),
           medianValue(tflops[pos]),
           rmseChecked[pos],
           rmseOk[pos],
           metrics[pos]});
      }
      return results;
    }

  }  // namespace

  // ---- Kernel factory ----

  std::vector<std::unique_ptr<TunableKernel>>
  makeMicroKernels(const ModelDesc* modelDesc, int nnXLen, int nnYLen, int batchSize, const VulkanTuneParams& cfg) {
    std::vector<std::unique_ptr<TunableKernel>> kernels;
    const TrunkDesc& trunk = modelDesc->trunk;

    // GEMM (Winograd trunk matmul): M ~ tile positions, N/K ~ channels, inTile^2 Winograd positions.
    // The tiled shader requires M, N ÷8 (8-wide A loads); round representative bench dims up
    // to ÷8 multiples so the bench exercises the same aligned path.
    int winograd3x3OutTile = winograd3x3OutTileFor(cfg);
    int winograd3x3InTile = winograd3x3OutTile + 2;
    int channels = roundUpToMultipleInt(
      std::max({trunk.trunkNumChannels, trunk.midNumChannels, modelDesc->maxConvChannels(3, 3), 64}), 8);
    int numTilesX = (nnXLen + winograd3x3OutTile - 1) / winograd3x3OutTile;
    int numTilesY = (nnYLen + winograd3x3OutTile - 1) / winograd3x3OutTile;
    int numTiles = roundUpToMultipleInt(std::max(1, batchSize) * numTilesX * numTilesY, 8);
    kernels.push_back(
      std::make_unique<VulkanKernels::WinogradGemm>(
        numTiles, channels, channels, winograd3x3InTile * winograd3x3InTile));

    int wgChannels = trunk.trunkNumChannels > 0 ? trunk.trunkNumChannels : 64;
    // Winograd output untransform.
    kernels.push_back(std::make_unique<VulkanKernels::WinogradUntransform>(batchSize, wgChannels, nnXLen, nnYLen));

    // GemmStridedTiled (1x1 conv).
    // The tiled GEMM path requires M and N to be ÷8; the runtime rounds M to
    // VULKAN_SPATIAL_ALIGN and pads N to a ÷8 multiple. Round bench dims to match.
    int trunkC = roundUpToMultipleInt(trunk.trunkNumChannels > 0 ? trunk.trunkNumChannels : 64, 8);
    int paddedSpatialSize = roundUpToMultipleInt(nnXLen * nnYLen, VulkanKernels::VULKAN_SPATIAL_ALIGN);
    kernels.push_back(std::make_unique<VulkanKernels::GemmStridedTiled>(batchSize, paddedSpatialSize, trunkC, trunkC));

    // GemmDirect (FC head).
    int gpoolC3 = trunk.gpoolNumChannels > 0 ? trunk.gpoolNumChannels * 3 : 192;
    kernels.push_back(std::make_unique<VulkanKernels::GemmDirectFP32>(batchSize > 0 ? batchSize : 1, gpoolC3, gpoolC3));

    // Reduction (gpool representative).
    int gpoolC = trunk.gpoolNumChannels > 0 ? trunk.gpoolNumChannels : 32;
    kernels.push_back(std::make_unique<VulkanKernels::GPoolReduction>(batchSize, gpoolC, nnXLen * nnYLen));

    // Value-head pooling (use gpoolNumChannels as a proxy for the head channel count).
    int v1C = trunk.gpoolNumChannels > 0 ? trunk.gpoolNumChannels : 32;
    kernels.push_back(std::make_unique<VulkanKernels::ValueHeadPool>(batchSize, v1C, nnXLen * nnYLen));

    // Transformer kernels — scan blocks recursively (transformers may live inside
    // nested bottleneck blocks) to find representative attention dimensions and
    // whether any attention block uses fixed/learnable RoPE.
    int headDim = 0, vHeadDim = 0, numHeads = 0, numKVHeads = 0;
    int ropeHeadDim = 0, ropeVHeadDim = 0, ropeNumHeads = 0, ropeNumKVHeads = 0;
    bool useRope = false;
    bool learnableRope = false;
    std::function<void(const vector<pair<int, unique_ptr_void>>&)> findAttn =
      [&](const vector<pair<int, unique_ptr_void>>& blks) {
        for(const auto& kv: blks) {
          if(kv.first == TRANSFORMER_ATTENTION_BLOCK_KIND) {
            const auto* attn = static_cast<const TransformerAttentionDesc*>(kv.second.get());
            if(headDim == 0) {
              headDim = attn->qHeadDim;
              vHeadDim = attn->vHeadDim;
              numHeads = attn->numHeads;
              numKVHeads = attn->numKVHeads;
            }
            if(attn->useRope) {
              useRope = true;
              if(ropeHeadDim == 0) {
                ropeHeadDim = attn->qHeadDim;
                ropeVHeadDim = attn->vHeadDim;
                ropeNumHeads = attn->numHeads;
                ropeNumKVHeads = attn->numKVHeads;
              }
            }
            learnableRope = learnableRope || (attn->useRope && attn->learnableRope);
          } else if(kv.first == NESTED_BOTTLENECK_BLOCK_KIND) {
            const auto* nbt = static_cast<const NestedBottleneckResidualBlockDesc*>(kv.second.get());
            findAttn(nbt->blocks);
          }
        }
      };
    findAttn(trunk.blocks);
    if(headDim > 0) {
      int benchHeadDim = (useRope && ropeHeadDim > 0) ? ropeHeadDim : headDim;
      int benchVHeadDim = (useRope && ropeVHeadDim > 0) ? ropeVHeadDim : vHeadDim;
      int benchNumHeads = (useRope && ropeNumHeads > 0) ? ropeNumHeads : numHeads;
      int benchNumKVHeads = (useRope && ropeNumKVHeads > 0) ? ropeNumKVHeads : numKVHeads;
      if(benchVHeadDim <= 0)
        benchVHeadDim = benchHeadDim;
      if(benchNumKVHeads <= 0)
        benchNumKVHeads = benchNumHeads;
      int seqLen = nnXLen * nnYLen;
      kernels.push_back(
        std::make_unique<VulkanKernels::AttentionTiled>(
          batchSize, seqLen, benchHeadDim, benchVHeadDim, benchNumHeads, benchNumKVHeads, useRope, learnableRope));
      int rmsChannels = trunk.trunkNumChannels > 0 ? trunk.trunkNumChannels : 64;
      kernels.push_back(std::make_unique<VulkanKernels::SpatialRMSNorm>(batchSize, rmsChannels, seqLen));
    }

    return kernels;
  }

  // ---- Generic search ----

  namespace {

    void appendTunedCandidateValues(std::ostream& out, const TunedCandidate& cand) {
      for(size_t i = 0; i < cand.values.size(); i++)
        out << (i ? "," : "") << cand.values[i].value;
    }

    bool shouldPrintTunerResult(bool verboseTuner, bool improvesBest, bool ok) {
      return verboseTuner || improvesBest || !ok;
    }

    struct AggregateCandidateMetrics {
      double weightedTime = 0.0;
      double tflops = 0.0;
      double maxRmse = 0.0;
      std::string rejectReason;
    };

    bool rejectAggregateCandidate(AggregateCandidateMetrics& metrics, string_view reason) {
      metrics.rejectReason = string(reason);
      return false;
    }

    template<typename CandidateT>
    struct ScreenedAggregateCandidate {
      CandidateT candidate;
      double screenWeightedTime;
      double maxRmse;
    };

    template<typename CandidateT>
    struct AggregateCandidateSweepResult {
      bool found = false;
      CandidateT candidate;
      double weightedTime = std::numeric_limits<double>::infinity();
      double tflops = 0.0;
      double maxRmse = 0.0;
    };

    namespace {

      struct TunerStreamText {
        constexpr TunerStreamText(
          string_view firstPart,
          string_view secondPart = string_view(),
          string_view thirdPart = string_view())
          : first(firstPart), second(secondPart), third(thirdPart) {}

        string_view first;
        string_view second;
        string_view third;
      };

      std::ostream& operator<<(std::ostream& out, const TunerStreamText& text) {
        return out << text.first << text.second << text.third;
      }

    }  // namespace

    template<typename CandidateT, typename BenchCandidateFn, typename CandidateToStringFn, typename CandidateEqualFn>
    AggregateCandidateSweepResult<CandidateT> tuneAggregateCandidateSweep(
      TunerStreamText candidateLogName,
      TunerStreamText confirmLogName,
      TunerStreamText selectedLogName,
      string_view candidateFieldName,
      TunerStreamText noValidMessage,
      const std::vector<CandidateT>& candidates,
      size_t confirmPrefixCount,
      bool printMaxRmse,
      bool printConfirmRmse,
      bool dedupeConfirmCandidates,
      std::ostream& out,
      bool verboseTuner,
      BenchCandidateFn benchCandidate,
      CandidateToStringFn candidateToString,
      CandidateEqualFn candidateEqual) {
      AggregateCandidateSweepResult<CandidateT> result;
      if(!candidates.empty())
        result.candidate = candidates[0];

      std::vector<ScreenedAggregateCandidate<CandidateT>> screenedCandidates;
      screenedCandidates.reserve(candidates.size());
      for(const CandidateT& candidate: candidates) {
        AggregateCandidateMetrics metrics;
        bool ok = benchCandidate(candidate, metrics);
        bool improvesBest = ok && metrics.weightedTime < result.weightedTime;
        if(shouldPrintTunerResult(verboseTuner, improvesBest, ok)) {
          out << "VulkanTuner:   " << candidateLogName << " candidate " << candidateFieldName << "="
              << candidateToString(candidate);
          if(ok) {
            out << " weighted_time=" << metrics.weightedTime;
            if(metrics.tflops > 0.0)
              out << " tflops=" << metrics.tflops;
            if(printMaxRmse)
              out << " max_rmse=" << metrics.maxRmse;
          } else if(printMaxRmse && metrics.maxRmse > 0.0) {
            out << " max_rmse=" << metrics.maxRmse;
          }
          out << " result=" << (ok ? (improvesBest ? "new_best" : "accepted") : "rejected");
          if(!ok && !metrics.rejectReason.empty())
            out << " reason=" << metrics.rejectReason;
          out << endl;
        }
        if(ok)
          screenedCandidates.push_back({candidate, metrics.weightedTime, metrics.maxRmse});
        if(improvesBest) {
          result.found = true;
          result.candidate = candidate;
          result.weightedTime = metrics.weightedTime;
          result.tflops = metrics.tflops;
          result.maxRmse = metrics.maxRmse;
        }
      }

      if(!screenedCandidates.empty()) {
        std::vector<size_t> confirmIndexes =
          makeConfirmationCandidateIndexes(screenedCandidates.size(), confirmPrefixCount, [&](size_t a, size_t b) {
            return screenedCandidates[a].screenWeightedTime < screenedCandidates[b].screenWeightedTime;
          });
        std::vector<ScreenedAggregateCandidate<CandidateT>> confirmCandidates;
        confirmCandidates.reserve(confirmIndexes.size());
        auto addConfirmCandidate = [&](const ScreenedAggregateCandidate<CandidateT>& candidate) {
          if(dedupeConfirmCandidates) {
            for(const ScreenedAggregateCandidate<CandidateT>& existing: confirmCandidates)
              if(candidateEqual(existing.candidate, candidate.candidate))
                return;
          }
          confirmCandidates.push_back(candidate);
        };
        for(size_t idx: confirmIndexes)
          addConfirmCandidate(screenedCandidates[idx]);

        out << "VulkanTuner: " << confirmLogName << " confirm candidates=" << confirmCandidates.size()
            << " repeats=" << TUNER_CONFIRM_REPEATS << endl;
        std::vector<size_t> dedupedIndexes(confirmCandidates.size());
        std::iota(dedupedIndexes.begin(), dedupedIndexes.end(), 0);
        std::vector<ConfirmationResult> confirmResults =
          confirmCandidateMetrics(dedupedIndexes, [&](size_t candidateIndex) {
            const ScreenedAggregateCandidate<CandidateT>& candidate = confirmCandidates[candidateIndex];
            AggregateCandidateMetrics metrics;
            bool ok = benchCandidate(candidate.candidate, metrics);
            return ConfirmationSample{
              ok, ok ? metrics.weightedTime : 0.0, ok ? metrics.tflops : 0.0, printConfirmRmse, printConfirmRmse && ok};
          });

        bool foundConfirmed = false;
        double confirmedBestWeightedTime = std::numeric_limits<double>::infinity();
        double confirmedBestTflops = 0.0;
        double confirmedBestMaxRmse = 0.0;
        CandidateT confirmedBestCandidate = result.candidate;
        for(size_t pos = 0; pos < confirmResults.size(); pos++) {
          const ConfirmationResult& confirmResult = confirmResults[pos];
          const ScreenedAggregateCandidate<CandidateT>& candidate = confirmCandidates[confirmResult.candidateIndex];
          const double confirmedWeightedTime = confirmResult.metric;
          const double confirmedTflops = confirmResult.tflops;
          const bool improvesConfirmed =
            confirmedWeightedTime > 0.0 && confirmedWeightedTime < confirmedBestWeightedTime;
          if(shouldPrintTunerResult(verboseTuner, improvesConfirmed, confirmedWeightedTime > 0.0)) {
            out << "VulkanTuner:   " << candidateLogName << " confirm candidate#" << pos << " " << candidateFieldName
                << "=" << candidateToString(candidate.candidate);
            if(confirmedWeightedTime > 0.0) {
              out << " weighted_time=" << confirmedWeightedTime;
              if(confirmedTflops > 0.0)
                out << " tflops=" << confirmedTflops;
            }
            if(!confirmResult.metricSamples.empty())
              appendSamplesField(out, "weighted_time_samples", confirmResult.metricSamples);
            if(printConfirmRmse)
              out << " rmse_ok=" << confirmResult.rmseOk << "/" << confirmResult.rmseChecked;
            out << " result="
                << (confirmedWeightedTime > 0.0 ? (improvesConfirmed ? "new_best" : "accepted") : "rejected") << endl;
          }
          if(improvesConfirmed) {
            foundConfirmed = true;
            confirmedBestWeightedTime = confirmedWeightedTime;
            confirmedBestTflops = confirmedTflops;
            confirmedBestMaxRmse = candidate.maxRmse;
            confirmedBestCandidate = candidate.candidate;
          }
        }

        if(foundConfirmed) {
          result.found = true;
          result.candidate = confirmedBestCandidate;
          result.weightedTime = confirmedBestWeightedTime;
          result.tflops = confirmedBestTflops;
          result.maxRmse = confirmedBestMaxRmse;
        }
      }

      if(result.found) {
        out << "VulkanTuner: " << selectedLogName << " selected " << candidateFieldName << "="
            << candidateToString(result.candidate) << " weighted_time=" << result.weightedTime;
        if(result.tflops > 0.0)
          out << " tflops=" << result.tflops;
        if(printMaxRmse)
          out << " max_rmse=" << result.maxRmse;
        out << endl;
      } else {
        out << noValidMessage << endl;
      }
      return result;
    }

    bool confirmTunedCandidates(
      TunableKernel& kernel,
      const VulkanTuneParams& seed,
      TuningContext& ctx,
      int iters,
      const KernelBench& ref,
      std::vector<TunedCandidate>& scored,
      std::ostream& out,
      bool verboseTuner) {
      if(scored.empty())
        return false;

      const double errorTol = 0.02;
      std::vector<TunedCandidate> candidates = scored;
      std::vector<size_t> confirmIndexes =
        makeConfirmationCandidateIndexes(candidates.size(), 0, [&](size_t a, size_t b) {
          return candidates[a].kernelsPerSecond > candidates[b].kernelsPerSecond;
        });
      if(confirmIndexes.empty())
        return false;

      out << "VulkanTuner: " << kernel.name() << " confirm candidates=" << confirmIndexes.size()
          << " repeats=" << TUNER_CONFIRM_REPEATS << endl;
      std::vector<ConfirmationResult> confirmResults =
        confirmCandidateMetrics(confirmIndexes, [&](size_t candidateIndex) {
          TunedCandidate& cand = candidates[candidateIndex];
          VulkanTuneParams cfg = seed;
          cand.applyTo(cfg);
          KernelBench b = kernel.bench(ctx, cfg, iters);
          if(!(b.ok && b.kernelsPerSecond > 0.0))
            return ConfirmationSample{false, 0.0, 0.0, false, false};
          double rmse = normalizedRmse(ref.output, b.output);
          bool rmseOk = rmse <= errorTol;
          return ConfirmationSample{rmseOk, b.kernelsPerSecond, 0.0, true, rmseOk};
        });

      std::vector<TunedCandidate> confirmed;
      confirmed.reserve(confirmIndexes.size());
      double bestConfirmedKps = -1.0;
      for(size_t pos = 0; pos < confirmResults.size(); pos++) {
        const ConfirmationResult& result = confirmResults[pos];
        TunedCandidate cand = candidates[result.candidateIndex];
        VulkanTuneParams cfg = seed;
        cand.applyTo(cfg);
        const double confirmedMedianKps = result.metric;
        const bool accepted = confirmedMedianKps > 0.0;
        const bool improvesConfirmed = accepted && confirmedMedianKps > bestConfirmedKps;
        if(shouldPrintTunerResult(verboseTuner, improvesConfirmed, accepted)) {
          out << "VulkanTuner:   " << kernel.name() << " confirm candidate#" << pos << " params=[";
          appendTunedCandidateValues(out, cand);
          out << "]";
          if(accepted)
            appendKernelPerf(out, kernel, cfg, confirmedMedianKps);
          if(!result.metricSamples.empty())
            appendSamplesField(out, "kps_samples", result.metricSamples);
          out << " rmse_ok=" << result.rmseOk << "/" << result.rmseChecked
              << " result=" << (accepted ? (improvesConfirmed ? "new_best" : "accepted") : "rejected") << endl;
        }
        if(accepted) {
          cand.kernelsPerSecond = confirmedMedianKps;
          confirmed.push_back(std::move(cand));
          if(improvesConfirmed)
            bestConfirmedKps = confirmedMedianKps;
        }
      }

      if(confirmed.empty())
        return false;
      std::sort(confirmed.begin(), confirmed.end(), [](const TunedCandidate& a, const TunedCandidate& b) {
        return a.kernelsPerSecond > b.kernelsPerSecond;
      });
      scored = std::move(confirmed);
      return true;
    }

  }  // namespace

  std::vector<TunedCandidate> tuneOne(
    TunableKernel& kernel,
    const VulkanTuneParams& seed,
    TuningContext& ctx,
    int iters,
    std::ostream& out,
    bool verboseTuner) {
    const double errorTol = 0.02;
    const VkPhysicalDeviceLimits& limits = ctx.dev->info.properties.limits;
    ArrayView<TunableParam> params = kernel.params();
    const uint32_t subgroupFloor = effectiveSubgroupFloor(kernel, seed, params, limits, ctx.dev->info.subgroupSize);

    std::vector<TunedCandidate> scored;
    double bestKernelsPerSecond = -1.0;
    size_t belowSubgroupSkipped = 0;

    // Reference output on the seed config, for RMSE gating.
    KernelBench ref = kernel.bench(ctx, seed, iters);
    if(!ref.ok) {
      out << "VulkanTuner: " << kernel.name() << " reference bench failed; keeping seed params." << endl;
      return scored;
    }

    // Odometer over the Cartesian product of params[i].candidates.
    size_t nParams = params.size();
    std::vector<size_t> idx(nParams, 0);
    bool done = (nParams == 0);
    while(!done) {
      VulkanTuneParams cfg = seed;
      TunedCandidate cand;
      for(size_t i = 0; i < nParams; i++) {
        int32_t v = params[i].candidates[idx[i]];
        cfg.*(params[i].field) = v;
        cand.values.push_back({params[i].field, v});
      }

      if(cfg.isValid() && kernel.validate(cfg, limits)) {
        if(workgroupBelowFloor(kernel, cfg, subgroupFloor)) {
          belowSubgroupSkipped++;
        } else {
          KernelBench b = kernel.bench(ctx, cfg, iters);
          if(b.ok) {
            double rmse = normalizedRmse(ref.output, b.output);
            bool within = rmse <= errorTol;
            bool improvesBest = within && b.kernelsPerSecond > bestKernelsPerSecond;
            if(shouldPrintTunerResult(verboseTuner, improvesBest, within)) {
              out << "VulkanTuner:   " << kernel.name() << " candidate params=[";
              for(size_t i = 0; i < cand.values.size(); i++)
                out << (i ? "," : "") << cand.values[i].value;
              out << "]";
              appendKernelPerf(out, kernel, cfg, b.kernelsPerSecond);
              out << " rmse=" << rmse;
              if(!within)
                out << " result=rejected";
              else if(improvesBest)
                out << " result=new_best";
              else
                out << " result=accepted";
              out << endl;
            }
            if(within) {
              cand.kernelsPerSecond = b.kernelsPerSecond;
              scored.push_back(std::move(cand));
              if(improvesBest)
                bestKernelsPerSecond = b.kernelsPerSecond;
            }
          } else if(shouldPrintTunerResult(verboseTuner, false, false)) {
            out << "VulkanTuner:   " << kernel.name() << " candidate params=[";
            for(size_t i = 0; i < cand.values.size(); i++)
              out << (i ? "," : "") << cand.values[i].value;
            out << "] result=rejected reason=bench_failed" << endl;
          }
        }
      }

      // Advance the odometer.
      size_t pos = 0;
      while(pos < nParams) {
        idx[pos]++;
        if(idx[pos] < params[pos].candidates.size())
          break;
        idx[pos] = 0;
        pos++;
      }
      if(pos == nParams)
        done = true;
    }

    if(belowSubgroupSkipped > 0)
      out << "VulkanTuner:   " << kernel.name() << " skipped " << belowSubgroupSkipped
          << " candidate(s) with workgroup < subgroup size (" << subgroupFloor << ")" << endl;

    std::sort(scored.begin(), scored.end(), [](const TunedCandidate& a, const TunedCandidate& b) {
      return a.kernelsPerSecond > b.kernelsPerSecond;
    });
    confirmTunedCandidates(kernel, seed, ctx, iters, ref, scored, out, verboseTuner);
    std::sort(scored.begin(), scored.end(), [](const TunedCandidate& a, const TunedCandidate& b) {
      return a.kernelsPerSecond > b.kernelsPerSecond;
    });
    if(scored.size() > (size_t)kernel.topK())
      scored.resize((size_t)kernel.topK());
    if(!scored.empty()) {
      out << "VulkanTuner: " << kernel.name() << " tuning selected params=[";
      for(size_t i = 0; i < scored[0].values.size(); i++)
        out << (i ? "," : "") << scored[0].values[i].value;
      VulkanTuneParams selectedCfg = seed;
      scored[0].applyTo(selectedCfg);
      out << "]";
      appendKernelPerf(out, kernel, selectedCfg, scored[0].kernelsPerSecond);
      out << endl;
    }
    return scored;
  }

  namespace {

    template<typename KernelT, typename CaseT, typename MakeProblemFn, typename MakeKernelFn, typename PrintCaseFn>
    bool tuneTiledMultiShape(
      string_view kernelName,
      const std::vector<CaseT>& modeCases,
      const VulkanTuneParams& seed,
      TuningContext& ctx,
      int iters,
      std::ostream& out,
      bool verboseTuner,
      VulkanTuneParams& tunedOut,
      MakeProblemFn&& makeProblem,
      MakeKernelFn&& makeKernel,
      PrintCaseFn&& printCase) {
      const double errorTol = 0.02;
      const VkPhysicalDeviceLimits& limits = ctx.dev->info.properties.limits;
      if(modeCases.empty())
        return false;

      out << "VulkanTuner:   " << kernelName << " multi-shape tuning cases:";
      for(const auto& cs: modeCases)
        printCase(out, cs);
      out << endl;

      ArrayView<TunableParam> params = KernelT::sharedParams();
      size_t nParams = params.size();
      std::vector<size_t> idx(nParams, 0);
      bool done = (nParams == 0);
      auto seedProblem = makeProblem(seed, modeCases[0]);
      KernelT subgroupKernel = makeKernel(seedProblem);
      const uint32_t subgroupFloor =
        effectiveSubgroupFloor(subgroupKernel, seed, params, limits, ctx.dev->info.subgroupSize);
      size_t belowSubgroupSkipped = 0;
      int itersPerCase = std::max(1, iters / std::max(1, (int)modeCases.size()));

      using ProblemT = decltype(makeProblem(seed, modeCases[0]));
      struct CaseRef {
        ProblemT problem;
        KernelBench bench;
      };
      std::vector<CaseRef> refs;
      auto getRef = [&](const ProblemT& problem) -> const KernelBench* {
        for(const CaseRef& ref: refs)
          if(ref.problem == problem)
            return &ref.bench;
        KernelT refKernel = makeKernel(problem);
        refs.push_back({problem, refKernel.bench(ctx, seed, itersPerCase)});
        return &refs.back().bench;
      };

      struct MultiShapeParamCandidate {
        VulkanTuneParams cfg;
        std::vector<int32_t> values;
      };
      std::vector<MultiShapeParamCandidate> candidates;

      auto benchMultiShapeCandidate = [&](const MultiShapeParamCandidate& cand, AggregateCandidateMetrics& metrics) {
        metrics.weightedTime = 0.0;
        metrics.tflops = 0.0;
        metrics.maxRmse = 0.0;
        double aggregateFlops = 0.0;
        double aggregateSeconds = 0.0;
        for(const CaseT& cs: modeCases) {
          ProblemT problem = makeProblem(cand.cfg, cs);
          KernelT probe = makeKernel(problem);
          if(!probe.validate(cand.cfg, limits))
            return rejectAggregateCandidate(metrics, "validate_failed");
          KernelBench b = probe.bench(ctx, cand.cfg, itersPerCase);
          if(!b.ok || b.kernelsPerSecond <= 0.0)
            return rejectAggregateCandidate(metrics, "bench_failed");

          const KernelBench* ref = getRef(problem);
          if(ref == nullptr || !ref->ok || ref->kernelsPerSecond <= 0.0)
            return rejectAggregateCandidate(metrics, "reference_failed");
          double rmse = normalizedRmse(ref->output, b.output);
          metrics.maxRmse = std::max(metrics.maxRmse, rmse);
          if(rmse > errorTol)
            return rejectAggregateCandidate(metrics, "rmse_exceeded");

          const double caseWeightedTime = dispatchWeightedTime(b, cs.occurrences);
          metrics.weightedTime += caseWeightedTime;
          aggregateFlops += probe.estimatedFlopsPerDispatch(cand.cfg) * (double)cs.occurrences;
          aggregateSeconds += caseWeightedTime;
        }
        metrics.tflops = aggregateTflops(aggregateFlops, aggregateSeconds);
        return metrics.weightedTime > 0.0;
      };

      while(!done) {
        VulkanTuneParams cfg = seed;
        std::vector<int32_t> values;
        values.reserve(nParams);
        for(size_t i = 0; i < nParams; i++) {
          int32_t v = params[i].candidates[idx[i]];
          cfg.*(params[i].field) = v;
          values.push_back(v);
        }

        ProblemT firstProblem = makeProblem(cfg, modeCases[0]);
        KernelT firstKernel = makeKernel(firstProblem);
        if(cfg.isValid() && firstKernel.validate(cfg, limits)) {
          if(workgroupBelowFloor(firstKernel, cfg, subgroupFloor)) {
            belowSubgroupSkipped++;
          } else {
            candidates.push_back({cfg, values});
          }
        }

        size_t pos = 0;
        while(pos < nParams) {
          idx[pos]++;
          if(idx[pos] < params[pos].candidates.size())
            break;
          idx[pos] = 0;
          pos++;
        }
        if(pos == nParams)
          done = true;
      }

      if(belowSubgroupSkipped > 0)
        out << "VulkanTuner:   " << kernelName << " multi-shape skipped " << belowSubgroupSkipped
            << " candidate(s) with workgroup < subgroup size (" << subgroupFloor << ")" << endl;

      auto candidateToString = [](const MultiShapeParamCandidate& cand) {
        std::ostringstream ss;
        ss << "[";
        for(size_t i = 0; i < cand.values.size(); i++)
          ss << (i ? "," : "") << cand.values[i];
        ss << "]";
        return ss.str();
      };
      auto candidatesEqual = [](const MultiShapeParamCandidate& a, const MultiShapeParamCandidate& b) {
        return a.values == b.values;
      };
      AggregateCandidateSweepResult<MultiShapeParamCandidate> result = tuneAggregateCandidateSweep(
        TunerStreamText{kernelName, " multi-shape"},
        TunerStreamText{kernelName, " multi-shape"},
        TunerStreamText{kernelName, " multi-shape tuning"},
        "params",
        TunerStreamText{
          "VulkanTuner: ", kernelName, " multi-shape tuning found no valid candidate; keeping seed params."},
        candidates,
        0,
        false,
        false,
        false,
        out,
        verboseTuner,
        benchMultiShapeCandidate,
        candidateToString,
        candidatesEqual);
      if(!result.found)
        return false;
      tunedOut = result.candidate.cfg;
      return true;
    }

    struct GemmStridedTiledProblem {
      int batchSize;
      int m;
      int n;
      int k;
      bool operator==(const GemmStridedTiledProblem& other) const {
        return batchSize == other.batchSize && m == other.m && n == other.n && k == other.k;
      }
    };

    template<typename KernelT>
    bool tuneGemmStridedTiledMultiShape(
      string_view kernelName,
      const ModelDesc* modelDesc,
      int batchSize,
      int nnXLen,
      int nnYLen,
      const VulkanTuneParams& seed,
      TuningContext& ctx,
      int iters,
      std::ostream& out,
      bool verboseTuner,
      VulkanTuneParams& tunedOut) {
      std::vector<GemmStridedModeCase> modeCases = collectGemmStridedModeCases(modelDesc, nnXLen, nnYLen);
      if(modeCases.empty()) {
        const TrunkDesc& trunk = modelDesc->trunk;
        int trunkC = roundUpToMultipleInt(trunk.trunkNumChannels > 0 ? trunk.trunkNumChannels : 64, 8);
        int paddedSpatialSize = roundUpToMultipleInt(nnXLen * nnYLen, VulkanKernels::VULKAN_SPATIAL_ALIGN);
        modeCases.push_back({paddedSpatialSize, trunkC, trunkC, 1, (double)trunkC * (double)trunkC});
      }

      auto makeProblem = [&](const VulkanTuneParams& /*cfg*/, const GemmStridedModeCase& cs) {
        return GemmStridedTiledProblem{std::max(1, batchSize), cs.M, cs.N, cs.K};
      };
      auto makeKernel = [](const GemmStridedTiledProblem& problem) {
        return KernelT(problem.batchSize, problem.m, problem.n, problem.k);
      };
      auto printCase = [](std::ostream& caseOut, const GemmStridedModeCase& cs) {
        caseOut << " [M=" << cs.M << " N=" << cs.N << " K=" << cs.K << " x" << cs.occurrences
                << " work=" << cs.weightedWork << "]";
      };
      return tuneTiledMultiShape<KernelT>(
        kernelName, modeCases, seed, ctx, iters, out, verboseTuner, tunedOut, makeProblem, makeKernel, printCase);
    }

    struct WinogradGemmTiledProblem {
      int m;
      int n;
      int k;
      int numBatches;
      bool operator==(const WinogradGemmTiledProblem& other) const {
        return m == other.m && n == other.n && k == other.k && numBatches == other.numBatches;
      }
    };

    template<typename KernelT>
    bool tuneWinogradGemmTiledMultiShape(
      string_view kernelName,
      const ModelDesc* modelDesc,
      int batchSize,
      int nnXLen,
      int nnYLen,
      const VulkanTuneParams& seed,
      TuningContext& ctx,
      int iters,
      std::ostream& out,
      bool verboseTuner,
      VulkanTuneParams& tunedOut) {
      std::vector<WinogradModeCase> modeCases = collectWinogradModeCases(modelDesc, batchSize, nnXLen, nnYLen, seed);
      if(modeCases.empty()) {
        const TrunkDesc& trunk = modelDesc->trunk;
        int outTile = winograd3x3OutTileFor(seed);
        int inTile = outTile + 2;
        int channels = roundUpToMultipleInt(
          std::max({trunk.trunkNumChannels, trunk.midNumChannels, modelDesc->maxConvChannels(3, 3), 64}), 8);
        int numTilesX = (nnXLen + outTile - 1) / outTile;
        int numTilesY = (nnYLen + outTile - 1) / outTile;
        int numTiles = std::max(1, batchSize) * numTilesX * numTilesY;
        modeCases.push_back(
          {numTiles,
           channels,
           channels,
           inTile * inTile,
           1,
           (double)inTile * (double)inTile * (double)numTiles * (double)channels * (double)channels});
      }

      auto makeProblem = [](const VulkanTuneParams& cfg, const WinogradModeCase& cs) {
        const auto pad = padDims(cs.M, cs.N, cs.K, KernelT::layerPaddingContract(cfg));
        return WinogradGemmTiledProblem{pad.m, pad.n, pad.k, cs.numBatches};
      };
      auto makeKernel = [](const WinogradGemmTiledProblem& problem) {
        return KernelT(problem.m, problem.n, problem.k, problem.numBatches);
      };
      auto printCase = [](std::ostream& caseOut, const WinogradModeCase& cs) {
        caseOut << " [M=" << cs.M << " N=" << cs.N << " K=" << cs.K << " transform_planes=" << cs.numBatches << " x"
                << cs.occurrences << " work=" << cs.weightedWork << "]";
      };
      return tuneTiledMultiShape<KernelT>(
        kernelName, modeCases, seed, ctx, iters, out, verboseTuner, tunedOut, makeProblem, makeKernel, printCase);
    }

    struct Dot2Tile {
      int32_t blockSize, bm, bn, wm, wn, wmIter, tm, tn, warp;
    };

    static const Dot2Tile kDot2Candidates[] = {
      {128, 64, 64, 32, 32, 2, 4, 2, 32},    // ggml F16 default
      {32, 32, 32, 32, 32, 2, 2, 2, 32},     // small tile (small matmuls)
      {128, 128, 128, 64, 64, 2, 4, 4, 32},  // large tile (big trunk GEMMs)
      {128, 128, 64, 64, 32, 2, 4, 2, 32},   // tall tile
      {128, 64, 128, 32, 64, 2, 4, 4, 32},   // wide tile
      // blockSize=64 (2-warp) thin tiles. Halving one tile dimension roughly doubles
      // the workgroup count, which helps two device classes: the strided path (gz =
      // batchSize, often 1, so a 64x64 tile underfills a wide GPU) and weak iGPUs/APUs
      // (few CUs, register/LDS-starved — more co-resident 2-warp workgroups hide
      // latency better than one 4-warp tile). Both bm/bn divide the sweep LCM (128),
      // so adding them does not change the LCM-padded bench dims of the other tiles.
      {64, 32, 64, 32, 32, 2, 4, 2, 32},  // thin-M tile
      {64, 64, 32, 32, 32, 2, 4, 2, 32},  // thin-N tile
    };

    std::string dot2TileToString(const Dot2Tile& t) {
      return Global::strprintf(
        "[%d,%d,%d,%d,%d,%d,%d,%d,%d]", t.blockSize, t.bm, t.bn, t.wm, t.wn, t.wmIter, t.tm, t.tn, t.warp);
    }

    bool dot2TilesEqual(const Dot2Tile& a, const Dot2Tile& b) {
      return a.blockSize == b.blockSize && a.bm == b.bm && a.bn == b.bn && a.wm == b.wm && a.wn == b.wn &&
             a.wmIter == b.wmIter && a.tm == b.tm && a.tn == b.tn && a.warp == b.warp;
    }

    void applyWinogradDot2Tile(VulkanTuneParams& cfg, const Dot2Tile& t) {
      cfg.dot2BlockSize = t.blockSize;
      cfg.dot2BM = t.bm;
      cfg.dot2BN = t.bn;
      cfg.dot2WM = t.wm;
      cfg.dot2WN = t.wn;
      cfg.dot2WMIter = t.wmIter;
      cfg.dot2TM = t.tm;
      cfg.dot2TN = t.tn;
      cfg.dot2Warp = t.warp;
    }

    void applyWinogradDot2AccF16Tile(VulkanTuneParams& cfg, const Dot2Tile& t) {
      cfg.dot2AccF16BlockSize = t.blockSize;
      cfg.dot2AccF16BM = t.bm;
      cfg.dot2AccF16BN = t.bn;
      cfg.dot2AccF16WM = t.wm;
      cfg.dot2AccF16WN = t.wn;
      cfg.dot2AccF16WMIter = t.wmIter;
      cfg.dot2AccF16TM = t.tm;
      cfg.dot2AccF16TN = t.tn;
      cfg.dot2AccF16Warp = t.warp;
    }

    void applyStridedDot2Tile(VulkanTuneParams& cfg, const Dot2Tile& t) {
      cfg.stridedDot2BlockSize = t.blockSize;
      cfg.stridedDot2BM = t.bm;
      cfg.stridedDot2BN = t.bn;
      cfg.stridedDot2WM = t.wm;
      cfg.stridedDot2WN = t.wn;
      cfg.stridedDot2WMIter = t.wmIter;
      cfg.stridedDot2TM = t.tm;
      cfg.stridedDot2TN = t.tn;
      cfg.stridedDot2Warp = t.warp;
    }

    void applyStridedDot2AccF16Tile(VulkanTuneParams& cfg, const Dot2Tile& t) {
      cfg.stridedDot2AccF16BlockSize = t.blockSize;
      cfg.stridedDot2AccF16BM = t.bm;
      cfg.stridedDot2AccF16BN = t.bn;
      cfg.stridedDot2AccF16WM = t.wm;
      cfg.stridedDot2AccF16WN = t.wn;
      cfg.stridedDot2AccF16WMIter = t.wmIter;
      cfg.stridedDot2AccF16TM = t.tm;
      cfg.stridedDot2AccF16TN = t.tn;
      cfg.stridedDot2AccF16Warp = t.warp;
    }

    void tuneWinogradDot2TileSweep(
      TuningContext& ctx,
      const ModelDesc* modelDesc,
      int problemBatchSize,
      int nnXLen,
      int nnYLen,
      int iters,
      std::ostream& out,
      bool verboseTuner,
      bool accF16,
      VulkanTuneParams& tunedConfig) {
      std::vector<WinogradModeCase> sweepCases =
        collectWinogradModeCases(modelDesc, problemBatchSize, nnXLen, nnYLen, tunedConfig);
      if(sweepCases.empty())
        return;

      struct CachedWinogradRef {
        int m, n, k, numBatches;
        KernelBench bench;
      };
      std::vector<CachedWinogradRef> tiledRefs;
      auto getTiledRef = [&](int m, int n, int k, int numBatches) -> const KernelBench* {
        for(const CachedWinogradRef& ref: tiledRefs)
          if(ref.m == m && ref.n == n && ref.k == k && ref.numBatches == numBatches)
            return &ref.bench;
        VulkanKernels::WinogradGemm tiledRefProbe(m, n, k, numBatches);
        tiledRefs.push_back({m, n, k, numBatches, tiledRefProbe.bench(ctx, tunedConfig, iters)});
        return &tiledRefs.back().bench;
      };

      auto benchWinogradDot2Tile = [&](const Dot2Tile& t, AggregateCandidateMetrics& metrics) {
        if(!VulkanKernels::WinogradGemmDot2::isConfigSupported(
             t.blockSize, t.bm, t.bn, t.wm, t.wn, t.wmIter, t.tm, t.tn, t.warp))
          return rejectAggregateCandidate(metrics, "unsupported_tile");
        VulkanTuneParams trial = tunedConfig;
        if(accF16)
          applyWinogradDot2AccF16Tile(trial, t);
        else
          applyWinogradDot2Tile(trial, t);

        metrics.weightedTime = 0.0;
        metrics.tflops = 0.0;
        metrics.maxRmse = 0.0;
        double aggregateFlops = 0.0;
        double aggregateSeconds = 0.0;
        for(const WinogradModeCase& cs: sweepCases) {
          const auto dot2Pad = padDims(
            cs.M,
            cs.N,
            cs.K,
            accF16 ? VulkanKernels::WinogradGemmDot2AccF16::layerPaddingContract(trial)
                   : VulkanKernels::WinogradGemmDot2::layerPaddingContract(trial));
          KernelBench dot2Bench;
          double dot2FlopsPerDispatch = 0.0;
          if(accF16) {
            VulkanKernels::WinogradGemmDot2AccF16 dot2Probe(dot2Pad.m, dot2Pad.n, dot2Pad.k, cs.numBatches);
            if(!dot2Probe.validate(trial, ctx.dev->info.properties.limits))
              return rejectAggregateCandidate(metrics, "validate_failed");
            dot2Bench = dot2Probe.bench(ctx, trial, iters);
            dot2FlopsPerDispatch = dot2Probe.estimatedFlopsPerDispatch(trial);
          } else {
            VulkanKernels::WinogradGemmDot2 dot2Probe(dot2Pad.m, dot2Pad.n, dot2Pad.k, cs.numBatches);
            if(!dot2Probe.validate(trial, ctx.dev->info.properties.limits))
              return rejectAggregateCandidate(metrics, "validate_failed");
            dot2Bench = dot2Probe.bench(ctx, trial, iters);
            dot2FlopsPerDispatch = dot2Probe.estimatedFlopsPerDispatch(trial);
          }
          if(!(dot2Bench.ok && dot2Bench.kernelsPerSecond > 0.0))
            return rejectAggregateCandidate(metrics, "bench_failed");

          const KernelBench* tiledRef = getTiledRef(dot2Pad.m, dot2Pad.n, dot2Pad.k, cs.numBatches);
          if(!tiledRef->ok)
            return rejectAggregateCandidate(metrics, "reference_failed");
          double rmse = normalizedRmse(tiledRef->output, dot2Bench.output);
          metrics.maxRmse = std::max(metrics.maxRmse, rmse);
          if(rmse > 0.02)
            return rejectAggregateCandidate(metrics, "rmse_exceeded");

          const double caseWeightedTime = dispatchWeightedTime(dot2Bench, cs.occurrences);
          metrics.weightedTime += caseWeightedTime;
          aggregateFlops += dot2FlopsPerDispatch * (double)cs.occurrences;
          aggregateSeconds += caseWeightedTime;
        }
        metrics.tflops = aggregateTflops(aggregateFlops, aggregateSeconds);
        return metrics.weightedTime > 0.0;
      };

      std::vector<Dot2Tile> candidates(
        kDot2Candidates, kDot2Candidates + sizeof(kDot2Candidates) / sizeof(kDot2Candidates[0]));
      string_view label = accF16 ? "winogradDot2AccF16" : "winogradDot2";
      out << "VulkanTuner: " << label << " tile sweep candidates=" << candidates.size()
          << " cases=" << sweepCases.size() << endl;
      AggregateCandidateSweepResult<Dot2Tile> result = tuneAggregateCandidateSweep(
        TunerStreamText{label},
        TunerStreamText{label, " tile sweep"},
        TunerStreamText{label, " tile sweep"},
        "tile",
        TunerStreamText{"VulkanTuner: ", label, " tile sweep found no valid tile; keeping defaults."},
        candidates,
        TUNER_CONFIRM_PREFIX_K,
        true,
        true,
        false,
        out,
        verboseTuner,
        benchWinogradDot2Tile,
        dot2TileToString,
        dot2TilesEqual);
      if(result.found) {
        if(accF16) {
          applyWinogradDot2AccF16Tile(tunedConfig, result.candidate);
          tunedConfig.winogradGemmDot2AccF16TunerValueValid = 1;
        } else {
          applyWinogradDot2Tile(tunedConfig, result.candidate);
          tunedConfig.winogradGemmDot2TunerValueValid = 1;
        }
      }
    }

    void tuneGemmStridedDot2TileSweep(
      TuningContext& ctx,
      const ModelDesc* modelDesc,
      int problemBatchSize,
      int nnXLen,
      int nnYLen,
      int iters,
      std::ostream& out,
      bool verboseTuner,
      bool accF16,
      VulkanTuneParams& tunedConfig) {
      std::vector<GemmStridedModeCase> modeCases = collectGemmStridedModeCases(modelDesc, nnXLen, nnYLen);
      if(modeCases.empty()) {
        const TrunkDesc& trunk = modelDesc->trunk;
        int trunkC = roundUpToMultipleInt(trunk.trunkNumChannels > 0 ? trunk.trunkNumChannels : 64, 8);
        int paddedSpatialSize = roundUpToMultipleInt(nnXLen * nnYLen, VulkanKernels::VULKAN_SPATIAL_ALIGN);
        modeCases.push_back({paddedSpatialSize, trunkC, trunkC, 1, (double)trunkC * (double)trunkC});
      }

      const VkPhysicalDeviceLimits& limits = ctx.dev->info.properties.limits;
      struct CachedStridedRef {
        int m, n, k;
        KernelBench bench;
      };
      std::vector<CachedStridedRef> tiledRefs;
      auto getTiledRef = [&](int m, int n, int k) -> const KernelBench* {
        for(const CachedStridedRef& ref: tiledRefs)
          if(ref.m == m && ref.n == n && ref.k == k)
            return &ref.bench;
        VulkanKernels::GemmStridedTiled tiledRefProbe(problemBatchSize, m, n, k);
        tiledRefs.push_back({m, n, k, tiledRefProbe.bench(ctx, tunedConfig, iters)});
        return &tiledRefs.back().bench;
      };

      auto benchStridedDot2Tile = [&](const Dot2Tile& t, AggregateCandidateMetrics& metrics) {
        if(!VulkanKernels::GemmStridedDot2::isConfigSupported(
             t.blockSize, t.bm, t.bn, t.wm, t.wn, t.wmIter, t.tm, t.tn, t.warp))
          return rejectAggregateCandidate(metrics, "unsupported_tile");
        VulkanTuneParams trial = tunedConfig;
        if(accF16)
          applyStridedDot2AccF16Tile(trial, t);
        else
          applyStridedDot2Tile(trial, t);

        metrics.weightedTime = 0.0;
        metrics.tflops = 0.0;
        metrics.maxRmse = 0.0;
        double aggregateFlops = 0.0;
        double aggregateSeconds = 0.0;
        for(const GemmStridedModeCase& cs: modeCases) {
          const auto dot2Pad = padDims(
            cs.M,
            cs.N,
            cs.K,
            accF16 ? VulkanKernels::GemmStridedDot2AccF16::layerPaddingContract(trial)
                   : VulkanKernels::GemmStridedDot2::layerPaddingContract(trial));

          KernelBench dot2Bench;
          double dot2FlopsPerDispatch = 0.0;
          if(accF16) {
            VulkanKernels::GemmStridedDot2AccF16 dot2Probe(problemBatchSize, dot2Pad.m, dot2Pad.n, dot2Pad.k);
            if(!dot2Probe.validate(trial, limits))
              return rejectAggregateCandidate(metrics, "validate_failed");
            dot2Bench = dot2Probe.bench(ctx, trial, iters);
            dot2FlopsPerDispatch = dot2Probe.estimatedFlopsPerDispatch(trial);
          } else {
            VulkanKernels::GemmStridedDot2 dot2Probe(problemBatchSize, dot2Pad.m, dot2Pad.n, dot2Pad.k);
            if(!dot2Probe.validate(trial, limits))
              return rejectAggregateCandidate(metrics, "validate_failed");
            dot2Bench = dot2Probe.bench(ctx, trial, iters);
            dot2FlopsPerDispatch = dot2Probe.estimatedFlopsPerDispatch(trial);
          }
          if(!(dot2Bench.ok && dot2Bench.kernelsPerSecond > 0.0))
            return rejectAggregateCandidate(metrics, "bench_failed");

          const KernelBench* tiledRef = getTiledRef(dot2Pad.m, dot2Pad.n, dot2Pad.k);
          if(!tiledRef->ok)
            return rejectAggregateCandidate(metrics, "reference_failed");
          double rmse = normalizedRmse(tiledRef->output, dot2Bench.output);
          metrics.maxRmse = std::max(metrics.maxRmse, rmse);
          if(rmse > 0.02)
            return rejectAggregateCandidate(metrics, "rmse_exceeded");

          const double caseWeightedTime = dispatchWeightedTime(dot2Bench, cs.occurrences);
          metrics.weightedTime += caseWeightedTime;
          aggregateFlops += dot2FlopsPerDispatch * (double)cs.occurrences;
          aggregateSeconds += caseWeightedTime;
        }
        metrics.tflops = aggregateTflops(aggregateFlops, aggregateSeconds);
        return metrics.weightedTime > 0.0;
      };

      std::vector<Dot2Tile> candidates(
        kDot2Candidates, kDot2Candidates + sizeof(kDot2Candidates) / sizeof(kDot2Candidates[0]));
      string_view label = accF16 ? "gemmStridedDot2AccF16" : "gemmStridedDot2";
      out << "VulkanTuner: " << label << " tile sweep candidates=" << candidates.size() << " cases=" << modeCases.size()
          << endl;
      AggregateCandidateSweepResult<Dot2Tile> result = tuneAggregateCandidateSweep(
        TunerStreamText{label},
        TunerStreamText{label, " tile sweep"},
        TunerStreamText{label, " tile sweep"},
        "tile",
        TunerStreamText{"VulkanTuner: ", label, " tile sweep found no valid tile; keeping defaults."},
        candidates,
        TUNER_CONFIRM_PREFIX_K,
        true,
        true,
        false,
        out,
        verboseTuner,
        benchStridedDot2Tile,
        dot2TileToString,
        dot2TilesEqual);
      if(result.found) {
        if(accF16) {
          applyStridedDot2AccF16Tile(tunedConfig, result.candidate);
          tunedConfig.gemmStridedDot2AccF16TunerValueValid = 1;
        } else {
          applyStridedDot2Tile(tunedConfig, result.candidate);
          tunedConfig.gemmStridedDot2TunerValueValid = 1;
        }
      }
    }

    void tuneGemmStridedDot2ModeSelect(
      TuningContext& ctx,
      const ModelDesc* modelDesc,
      int problemBatchSize,
      int nnXLen,
      int nnYLen,
      int iters,
      std::ostream& out,
      bool verboseTuner,
      VulkanTuneParams& tunedConfig) {
      std::vector<GemmStridedModeCase> modeCases = collectGemmStridedModeCases(modelDesc, nnXLen, nnYLen);
      if(modeCases.empty())
        return;

      ModeSelectStats tiledStats;
      ModeSelectStats dot2Stats;
      double maxRmse = 0.0;

      out << endl << "VulkanTuner: gemmStrided dot2 mode-select cases (" << modeCases.size() << ") ..." << endl;
      for(size_t i = 0; i < modeCases.size(); i++) {
        const GemmStridedModeCase& cs = modeCases[i];
        const auto tiledPad =
          padDims(cs.M, cs.N, cs.K, VulkanKernels::GemmStridedTiled::layerPaddingContract(tunedConfig));
        const auto dot2Pad =
          padDims(cs.M, cs.N, cs.K, VulkanKernels::GemmStridedDot2::layerPaddingContract(tunedConfig));

        VulkanKernels::GemmStridedTiled tiledSpeedProbe(problemBatchSize, tiledPad.m, tiledPad.n, tiledPad.k);
        KernelBench tiledSpeedBench = tiledSpeedProbe.bench(ctx, tunedConfig, iters);
        const bool tiledOk = tiledSpeedBench.ok && tiledSpeedBench.kernelsPerSecond > 0.0;
        double tiledTflops = 0.0;
        if(tiledOk)
          tiledTflops = tiledSpeedProbe.estimatedTflops(tunedConfig, tiledSpeedBench.kernelsPerSecond);
        double tiledCaseWeightedTime = tiledStats.recordCase(
          tiledOk, tiledSpeedBench, tiledSpeedProbe.estimatedFlopsPerDispatch(tunedConfig), cs.occurrences);

        VulkanKernels::GemmStridedDot2 dot2Probe(problemBatchSize, dot2Pad.m, dot2Pad.n, dot2Pad.k);
        KernelBench dot2Bench = dot2Probe.bench(ctx, tunedConfig, iters);
        double dot2Tflops = dot2Probe.estimatedTflops(tunedConfig, dot2Bench.kernelsPerSecond);
        bool dot2Ok = dot2Bench.ok && dot2Bench.kernelsPerSecond > 0.0;
        double dot2CaseWeightedTime = 0.0;
        double rmse = std::numeric_limits<double>::infinity();
        if(dot2Ok) {
          KernelBench tiledRefBench = tiledSpeedBench;
          if(tiledPad.m != dot2Pad.m || tiledPad.n != dot2Pad.n || tiledPad.k != dot2Pad.k) {
            VulkanKernels::GemmStridedTiled tiledRefProbe(problemBatchSize, dot2Pad.m, dot2Pad.n, dot2Pad.k);
            tiledRefBench = tiledRefProbe.bench(ctx, tunedConfig, iters);
          }
          dot2Ok = tiledRefBench.ok;
          if(dot2Ok) {
            rmse = normalizedRmse(tiledRefBench.output, dot2Bench.output);
            maxRmse = std::max(maxRmse, rmse);
            dot2Ok = rmse <= 0.02;
          }
        }
        if(dot2Ok) {
          dot2CaseWeightedTime =
            dot2Stats.recordCase(dot2Ok, dot2Bench, dot2Probe.estimatedFlopsPerDispatch(tunedConfig), cs.occurrences);
        } else
          dot2Stats.allOk = false;

        if(shouldPrintTunerResult(verboseTuner, false, tiledOk && dot2Ok)) {
          out << "VulkanTuner:   gemmStrided dot2 case#" << i;
          if(tiledOk) {
            out << " tiled_kps=" << tiledSpeedBench.kernelsPerSecond;
            if(tiledTflops > 0.0)
              out << " tiled_tflops=" << tiledTflops;
            out << " tiled_weighted_time=" << tiledCaseWeightedTime << " tiled_dims=" << tiledPad.m << "x" << tiledPad.n
                << "x" << tiledPad.k;
          } else
            out << " tiled=failed";
          if(dot2Bench.ok && dot2Bench.kernelsPerSecond > 0.0) {
            out << " dot2_kps=" << dot2Bench.kernelsPerSecond;
            if(dot2Tflops > 0.0)
              out << " dot2_tflops=" << dot2Tflops;
            out << " dot2_weighted_time=" << dot2CaseWeightedTime << " dot2_dims=" << dot2Pad.m << "x" << dot2Pad.n
                << "x" << dot2Pad.k;
          } else
            out << " dot2=failed";
          out << " rmse=" << (std::isfinite(rmse) ? Global::doubleToString(rmse) : string("n/a"))
              << " dot2_result=" << (dot2Ok ? "accepted" : "rejected") << endl;
        }
      }

      bool selected = appendModeSelectResult(
        out,
        "gemmStrided",
        "gemmStridedTiled",
        "gemmStridedDot2",
        modeCases,
        tiledStats,
        dot2Stats,
        maxRmse,
        true,
        true,
        [&](std::ostream& resultOut) {
          appendPositiveField(resultOut, "dot2_speedup", modeSelectSpeedup(tiledStats, dot2Stats));
        });
      if(selected) {
        tunedConfig.enableGemmStridedDot2 = 1;
        tunedConfig.enableGemmStridedDot2AccF16 = 0;
        preferGemmStridedSelectionRank(
          tunedConfig,
          &VulkanTuneParams::gemmStridedDot2SelectionRank,
          &VulkanTuneParams::gemmStridedTiledSelectionRank);
      } else
        preferGemmStridedSelectionRank(
          tunedConfig,
          &VulkanTuneParams::gemmStridedTiledSelectionRank,
          &VulkanTuneParams::gemmStridedDot2SelectionRank);
    }

    void tuneWinogradGemmDot2ModeSelect(
      TuningContext& ctx,
      const ModelDesc* modelDesc,
      int problemBatchSize,
      int nnXLen,
      int nnYLen,
      int iters,
      std::ostream& out,
      bool verboseTuner,
      VulkanTuneParams& tunedConfig) {
      std::vector<WinogradModeCase> wCases =
        collectWinogradModeCases(modelDesc, problemBatchSize, nnXLen, nnYLen, tunedConfig);
      if(wCases.empty())
        return;

      ModeSelectStats tiledStats;
      ModeSelectStats dot2Stats;
      double maxRmse = 0.0;

      out << endl << "VulkanTuner: winogradGemm dot2 mode-select cases (" << wCases.size() << ") ..." << endl;
      for(size_t i = 0; i < wCases.size(); i++) {
        const WinogradModeCase& cs = wCases[i];
        const auto tiledPad = padDims(cs.M, cs.N, cs.K, VulkanKernels::WinogradGemm::layerPaddingContract(tunedConfig));
        const auto dot2Pad =
          padDims(cs.M, cs.N, cs.K, VulkanKernels::WinogradGemmDot2::layerPaddingContract(tunedConfig));

        VulkanKernels::WinogradGemm tiledSpeedProbe(tiledPad.m, tiledPad.n, tiledPad.k, cs.numBatches);
        KernelBench tiledSpeedBench = tiledSpeedProbe.bench(ctx, tunedConfig, iters);
        const bool tiledOk = tiledSpeedBench.ok && tiledSpeedBench.kernelsPerSecond > 0.0;
        double tiledTflops = 0.0;
        if(tiledOk)
          tiledTflops = tiledSpeedProbe.estimatedTflops(tunedConfig, tiledSpeedBench.kernelsPerSecond);
        double tiledCaseWeightedTime = tiledStats.recordCase(
          tiledOk, tiledSpeedBench, tiledSpeedProbe.estimatedFlopsPerDispatch(tunedConfig), cs.occurrences);

        VulkanKernels::WinogradGemmDot2 dot2Probe(dot2Pad.m, dot2Pad.n, dot2Pad.k, cs.numBatches);
        KernelBench dot2Bench = dot2Probe.bench(ctx, tunedConfig, iters);
        double dot2Tflops = dot2Probe.estimatedTflops(tunedConfig, dot2Bench.kernelsPerSecond);
        bool dot2Ok = dot2Bench.ok && dot2Bench.kernelsPerSecond > 0.0;
        double dot2CaseWeightedTime = 0.0;
        double rmse = std::numeric_limits<double>::infinity();
        if(dot2Ok) {
          KernelBench tiledRefBench = tiledSpeedBench;
          if(tiledPad.m != dot2Pad.m || tiledPad.n != dot2Pad.n || tiledPad.k != dot2Pad.k) {
            VulkanKernels::WinogradGemm tiledRefProbe(dot2Pad.m, dot2Pad.n, dot2Pad.k, cs.numBatches);
            tiledRefBench = tiledRefProbe.bench(ctx, tunedConfig, iters);
          }
          dot2Ok = tiledRefBench.ok;
          if(dot2Ok) {
            rmse = normalizedRmse(tiledRefBench.output, dot2Bench.output);
            maxRmse = std::max(maxRmse, rmse);
            dot2Ok = rmse <= 0.02;
          }
        }
        if(dot2Ok) {
          dot2CaseWeightedTime =
            dot2Stats.recordCase(dot2Ok, dot2Bench, dot2Probe.estimatedFlopsPerDispatch(tunedConfig), cs.occurrences);
        } else
          dot2Stats.allOk = false;

        if(shouldPrintTunerResult(verboseTuner, false, tiledOk && dot2Ok)) {
          out << "VulkanTuner:   winogradGemm dot2 case#" << i;
          if(tiledOk) {
            out << " tiled_kps=" << tiledSpeedBench.kernelsPerSecond;
            if(tiledTflops > 0.0)
              out << " tiled_tflops=" << tiledTflops;
            out << " tiled_weighted_time=" << tiledCaseWeightedTime << " tiled_dims=" << tiledPad.m << "x" << tiledPad.n
                << "x" << tiledPad.k;
          } else
            out << " tiled=failed";
          if(dot2Bench.ok && dot2Bench.kernelsPerSecond > 0.0) {
            out << " dot2_kps=" << dot2Bench.kernelsPerSecond;
            if(dot2Tflops > 0.0)
              out << " dot2_tflops=" << dot2Tflops;
            out << " dot2_weighted_time=" << dot2CaseWeightedTime << " dot2_dims=" << dot2Pad.m << "x" << dot2Pad.n
                << "x" << dot2Pad.k;
          } else
            out << " dot2=failed";
          out << " rmse=" << (std::isfinite(rmse) ? Global::doubleToString(rmse) : string("n/a"))
              << " dot2_result=" << (dot2Ok ? "accepted" : "rejected") << endl;
        }
      }

      bool selected = appendModeSelectResult(
        out,
        "winogradGemm",
        "winogradGemmTiled",
        "winogradGemmDot2",
        wCases,
        tiledStats,
        dot2Stats,
        maxRmse,
        true,
        true,
        [&](std::ostream& resultOut) {
          appendPositiveField(resultOut, "dot2_speedup", modeSelectSpeedup(tiledStats, dot2Stats));
        });
      if(selected) {
        tunedConfig.enableWinogradGemmDot2 = 1;
        tunedConfig.enableWinogradGemmDot2AccF16 = 0;
        preferWinogradGemmSelectionRank(
          tunedConfig,
          &VulkanTuneParams::winogradGemmDot2SelectionRank,
          &VulkanTuneParams::winogradGemmTiledSelectionRank);
      } else
        preferWinogradGemmSelectionRank(
          tunedConfig,
          &VulkanTuneParams::winogradGemmTiledSelectionRank,
          &VulkanTuneParams::winogradGemmDot2SelectionRank);
    }

    void tuneGemmStridedDot2AccF16ModeSelect(
      TuningContext& ctx,
      const ModelDesc* modelDesc,
      int problemBatchSize,
      int nnXLen,
      int nnYLen,
      int iters,
      std::ostream& out,
      bool verboseTuner,
      VulkanTuneParams& tunedConfig) {
      std::vector<GemmStridedModeCase> modeCases = collectGemmStridedModeCases(modelDesc, nnXLen, nnYLen);
      if(modeCases.empty())
        return;

      const bool incumbentDot2 = tunedConfig.enableGemmStridedDot2 != 0;
      string_view incumbentName = incumbentDot2 ? "gemmStridedDot2" : "gemmStridedTiled";

      ModeSelectStats incumbentStats;
      ModeSelectStats accStats;
      double maxRmse = 0.0;

      out << endl << "VulkanTuner: gemmStrided dot2AccF16 mode-select cases (" << modeCases.size() << ") ..." << endl;
      for(size_t i = 0; i < modeCases.size(); i++) {
        const GemmStridedModeCase& cs = modeCases[i];

        KernelBench incumbentBench;
        double incumbentTflops = 0.0;
        double incumbentFlopsPerDispatch = 0.0;
        if(incumbentDot2) {
          const auto dPad =
            padDims(cs.M, cs.N, cs.K, VulkanKernels::GemmStridedDot2::layerPaddingContract(tunedConfig));
          VulkanKernels::GemmStridedDot2 dProbe(problemBatchSize, dPad.m, dPad.n, dPad.k);
          incumbentBench = dProbe.bench(ctx, tunedConfig, iters);
          incumbentTflops = dProbe.estimatedTflops(tunedConfig, incumbentBench.kernelsPerSecond);
          incumbentFlopsPerDispatch = dProbe.estimatedFlopsPerDispatch(tunedConfig);
        } else {
          const auto tPad =
            padDims(cs.M, cs.N, cs.K, VulkanKernels::GemmStridedTiled::layerPaddingContract(tunedConfig));
          VulkanKernels::GemmStridedTiled tProbe(problemBatchSize, tPad.m, tPad.n, tPad.k);
          incumbentBench = tProbe.bench(ctx, tunedConfig, iters);
          incumbentTflops = tProbe.estimatedTflops(tunedConfig, incumbentBench.kernelsPerSecond);
          incumbentFlopsPerDispatch = tProbe.estimatedFlopsPerDispatch(tunedConfig);
        }
        const bool incumbentOk = incumbentBench.ok && incumbentBench.kernelsPerSecond > 0.0;
        double incumbentCaseWeightedTime =
          incumbentStats.recordCase(incumbentOk, incumbentBench, incumbentFlopsPerDispatch, cs.occurrences);

        const auto accPad =
          padDims(cs.M, cs.N, cs.K, VulkanKernels::GemmStridedDot2AccF16::layerPaddingContract(tunedConfig));
        VulkanKernels::GemmStridedDot2AccF16 accProbe(problemBatchSize, accPad.m, accPad.n, accPad.k);
        KernelBench accBench = accProbe.bench(ctx, tunedConfig, iters);
        double accTflops = accProbe.estimatedTflops(tunedConfig, accBench.kernelsPerSecond);
        bool accOk = accBench.ok && accBench.kernelsPerSecond > 0.0;
        double accCaseWeightedTime = 0.0;
        double rmse = std::numeric_limits<double>::infinity();
        if(accOk) {
          VulkanKernels::GemmStridedTiled refProbe(problemBatchSize, accPad.m, accPad.n, accPad.k);
          KernelBench refBench = refProbe.bench(ctx, tunedConfig, iters);
          accOk = refBench.ok;
          if(accOk) {
            rmse = normalizedRmse(refBench.output, accBench.output);
            maxRmse = std::max(maxRmse, rmse);
            accOk = rmse <= 0.02;
          }
        }
        if(accOk) {
          accCaseWeightedTime =
            accStats.recordCase(accOk, accBench, accProbe.estimatedFlopsPerDispatch(tunedConfig), cs.occurrences);
        } else
          accStats.allOk = false;

        if(shouldPrintTunerResult(verboseTuner, false, incumbentOk && accOk)) {
          out << "VulkanTuner:   gemmStrided dot2AccF16 case#" << i;
          if(incumbentOk) {
            out << " " << incumbentName << "_kps=" << incumbentBench.kernelsPerSecond;
            if(incumbentTflops > 0.0)
              out << " " << incumbentName << "_tflops=" << incumbentTflops;
            out << " " << incumbentName << "_weighted_time=" << incumbentCaseWeightedTime;
          } else
            out << " " << incumbentName << "=failed";
          if(accBench.ok && accBench.kernelsPerSecond > 0.0) {
            out << " dot2AccF16_kps=" << accBench.kernelsPerSecond;
            if(accTflops > 0.0)
              out << " dot2AccF16_tflops=" << accTflops;
            out << " dot2AccF16_weighted_time=" << accCaseWeightedTime;
          } else
            out << " dot2AccF16=failed";
          out << " rmse=" << (std::isfinite(rmse) ? Global::doubleToString(rmse) : string("n/a"))
              << " dot2AccF16_result=" << (accOk ? "accepted" : "rejected") << endl;
        }
      }

      bool selected = appendModeSelectResult(
        out,
        "gemmStrided",
        incumbentName,
        "gemmStridedDot2AccF16",
        modeCases,
        incumbentStats,
        accStats,
        maxRmse,
        true,
        true,
        [&](std::ostream& resultOut) {
          appendPositiveField(resultOut, "dot2AccF16_speedup", modeSelectSpeedup(incumbentStats, accStats));
        });
      if(selected) {
        tunedConfig.enableGemmStridedDot2AccF16 = 1;
        tunedConfig.enableGemmStridedDot2 = 0;
        preferGemmStridedSelectionRank(
          tunedConfig,
          &VulkanTuneParams::gemmStridedDot2AccF16SelectionRank,
          incumbentDot2 ? &VulkanTuneParams::gemmStridedDot2SelectionRank
                        : &VulkanTuneParams::gemmStridedTiledSelectionRank);
      } else if(!incumbentDot2) {
        preferGemmStridedSelectionRank(
          tunedConfig,
          &VulkanTuneParams::gemmStridedTiledSelectionRank,
          &VulkanTuneParams::gemmStridedDot2AccF16SelectionRank);
      }
    }

    void tuneWinogradGemmDot2AccF16ModeSelect(
      TuningContext& ctx,
      const ModelDesc* modelDesc,
      int problemBatchSize,
      int nnXLen,
      int nnYLen,
      int iters,
      std::ostream& out,
      bool verboseTuner,
      VulkanTuneParams& tunedConfig) {
      std::vector<WinogradModeCase> wCases =
        collectWinogradModeCases(modelDesc, problemBatchSize, nnXLen, nnYLen, tunedConfig);
      if(wCases.empty())
        return;

      const bool incumbentDot2AccF16 = tunedConfig.enableWinogradGemmDot2AccF16 != 0;
      const bool incumbentDot2 = !incumbentDot2AccF16 && tunedConfig.enableWinogradGemmDot2 != 0;
      string_view incumbentName =
        incumbentDot2AccF16 ? "winogradGemmDot2AccF16" : (incumbentDot2 ? "winogradGemmDot2" : "winogradGemmTiled");

      ModeSelectStats incumbentStats;
      ModeSelectStats accStats;
      double maxRmse = 0.0;

      out << endl << "VulkanTuner: winogradGemm dot2AccF16 mode-select cases (" << wCases.size() << ") ..." << endl;
      for(size_t i = 0; i < wCases.size(); i++) {
        const WinogradModeCase& cs = wCases[i];

        KernelBench incumbentBench;
        double incumbentTflops = 0.0;
        double incumbentFlopsPerDispatch = 0.0;
        if(incumbentDot2AccF16) {
          const auto dPad =
            padDims(cs.M, cs.N, cs.K, VulkanKernels::WinogradGemmDot2AccF16::layerPaddingContract(tunedConfig));
          VulkanKernels::WinogradGemmDot2AccF16 dProbe(dPad.m, dPad.n, dPad.k, cs.numBatches);
          incumbentBench = dProbe.bench(ctx, tunedConfig, iters);
          incumbentTflops = dProbe.estimatedTflops(tunedConfig, incumbentBench.kernelsPerSecond);
          incumbentFlopsPerDispatch = dProbe.estimatedFlopsPerDispatch(tunedConfig);
        } else if(incumbentDot2) {
          const auto dPad =
            padDims(cs.M, cs.N, cs.K, VulkanKernels::WinogradGemmDot2::layerPaddingContract(tunedConfig));
          VulkanKernels::WinogradGemmDot2 dProbe(dPad.m, dPad.n, dPad.k, cs.numBatches);
          incumbentBench = dProbe.bench(ctx, tunedConfig, iters);
          incumbentTflops = dProbe.estimatedTflops(tunedConfig, incumbentBench.kernelsPerSecond);
          incumbentFlopsPerDispatch = dProbe.estimatedFlopsPerDispatch(tunedConfig);
        } else {
          const auto tPad = padDims(cs.M, cs.N, cs.K, VulkanKernels::WinogradGemm::layerPaddingContract(tunedConfig));
          VulkanKernels::WinogradGemm tProbe(tPad.m, tPad.n, tPad.k, cs.numBatches);
          incumbentBench = tProbe.bench(ctx, tunedConfig, iters);
          incumbentTflops = tProbe.estimatedTflops(tunedConfig, incumbentBench.kernelsPerSecond);
          incumbentFlopsPerDispatch = tProbe.estimatedFlopsPerDispatch(tunedConfig);
        }
        const bool incumbentOk = incumbentBench.ok && incumbentBench.kernelsPerSecond > 0.0;
        double incumbentCaseWeightedTime =
          incumbentStats.recordCase(incumbentOk, incumbentBench, incumbentFlopsPerDispatch, cs.occurrences);

        const auto accPad =
          padDims(cs.M, cs.N, cs.K, VulkanKernels::WinogradGemmDot2AccF16::layerPaddingContract(tunedConfig));
        VulkanKernels::WinogradGemmDot2AccF16 accProbe(accPad.m, accPad.n, accPad.k, cs.numBatches);
        KernelBench accBench = accProbe.bench(ctx, tunedConfig, iters);
        double accTflops = accProbe.estimatedTflops(tunedConfig, accBench.kernelsPerSecond);
        bool accOk = accBench.ok && accBench.kernelsPerSecond > 0.0;
        double accCaseWeightedTime = 0.0;
        double rmse = std::numeric_limits<double>::infinity();
        if(accOk) {
          VulkanKernels::WinogradGemm refProbe(accPad.m, accPad.n, accPad.k, cs.numBatches);
          KernelBench refBench = refProbe.bench(ctx, tunedConfig, iters);
          accOk = refBench.ok;
          if(accOk) {
            rmse = normalizedRmse(refBench.output, accBench.output);
            maxRmse = std::max(maxRmse, rmse);
            accOk = rmse <= 0.02;
          }
        }
        if(accOk) {
          accCaseWeightedTime =
            accStats.recordCase(accOk, accBench, accProbe.estimatedFlopsPerDispatch(tunedConfig), cs.occurrences);
        } else
          accStats.allOk = false;

        if(shouldPrintTunerResult(verboseTuner, false, incumbentOk && accOk)) {
          out << "VulkanTuner:   winogradGemm dot2AccF16 case#" << i;
          if(incumbentOk) {
            out << " " << incumbentName << "_kps=" << incumbentBench.kernelsPerSecond;
            if(incumbentTflops > 0.0)
              out << " " << incumbentName << "_tflops=" << incumbentTflops;
            out << " " << incumbentName << "_weighted_time=" << incumbentCaseWeightedTime;
          } else
            out << " " << incumbentName << "=failed";
          if(accBench.ok && accBench.kernelsPerSecond > 0.0) {
            out << " dot2AccF16_kps=" << accBench.kernelsPerSecond;
            if(accTflops > 0.0)
              out << " dot2AccF16_tflops=" << accTflops;
            out << " dot2AccF16_weighted_time=" << accCaseWeightedTime;
          } else
            out << " dot2AccF16=failed";
          out << " rmse=" << (std::isfinite(rmse) ? Global::doubleToString(rmse) : string("n/a"))
              << " dot2AccF16_result=" << (accOk ? "accepted" : "rejected") << endl;
        }
      }

      bool selected = appendModeSelectResult(
        out,
        "winogradGemm",
        incumbentName,
        "winogradGemmDot2AccF16",
        wCases,
        incumbentStats,
        accStats,
        maxRmse,
        true,
        true,
        [&](std::ostream& resultOut) {
          appendPositiveField(resultOut, "dot2AccF16_speedup", modeSelectSpeedup(incumbentStats, accStats));
        });
      if(selected) {
        tunedConfig.enableWinogradGemmDot2AccF16 = 1;
        tunedConfig.enableWinogradGemmDot2 = 0;
        preferWinogradGemmSelectionRank(
          tunedConfig,
          &VulkanTuneParams::winogradGemmDot2AccF16SelectionRank,
          incumbentDot2AccF16 ? &VulkanTuneParams::winogradGemmDot2AccF16SelectionRank
                              : (incumbentDot2 ? &VulkanTuneParams::winogradGemmDot2SelectionRank
                                               : &VulkanTuneParams::winogradGemmTiledSelectionRank));
      } else if(!incumbentDot2AccF16 && !incumbentDot2) {
        preferWinogradGemmSelectionRank(
          tunedConfig,
          &VulkanTuneParams::winogradGemmTiledSelectionRank,
          &VulkanTuneParams::winogradGemmDot2AccF16SelectionRank);
      }
    }

    void tuneDot2ModeSelect(
      TuningContext& ctx,
      const ModelDesc* modelDesc,
      int batchSize,
      int nnXLen,
      int nnYLen,
      int iters,
      std::ostream& out,
      bool verboseTuner,
      VulkanTuneParams& tunedConfig) {
      tunedConfig.enableGemmStridedDot2 = 0;
      tunedConfig.enableWinogradGemmDot2 = 0;
      tunedConfig.enableGemmStridedDot2AccF16 = 0;
      tunedConfig.enableWinogradGemmDot2AccF16 = 0;
      const bool haveDevice = ctx.dev != nullptr;
      const VulkanDeviceInfo* info = haveDevice ? &ctx.dev->info : nullptr;
      const bool supportsFP16Compute = haveDevice && ctx.fp16Compute;
      const bool supportsAnyDot2 = haveDevice && (ctx.dev->info.supportsDot2F16 || ctx.dev->info.supportsDot2F16AccF16);
      const bool unfilteredSupportsAnyDot2 =
        haveDevice && (ctx.dev->info.unfilteredSupportsDot2F16 || ctx.dev->info.unfilteredSupportsDot2F16AccF16);
      const bool filterDisabledAnyDot2 =
        haveDevice && (ctx.dev->info.filterDisabledDot2F16 || ctx.dev->info.filterDisabledDot2F16AccF16);
      const bool supportsDot2AccF32 =
        haveDevice && ctx.dev->info.supportsDot2F16 && ctx.fp16Storage && supportsFP16Compute;
      const bool supportsDot2AccF16 =
        haveDevice && ctx.dev->info.supportsDot2F16AccF16 && ctx.fp16Storage && supportsFP16Compute;
      if(supportsDot2AccF32 || supportsDot2AccF16) {
        out << endl << "VulkanTuner: mode-select dot2 vs current best for winograd/strided GEMM..." << endl;
        const int problemBatchSize = std::max(1, batchSize);
        if(supportsDot2AccF32) {
          if(tunedConfig.gemmStridedDot2TunerValueValid != 0)
            tuneGemmStridedDot2ModeSelect(
              ctx, modelDesc, problemBatchSize, nnXLen, nnYLen, iters, out, verboseTuner, tunedConfig);
          if(tunedConfig.winogradGemmDot2TunerValueValid != 0)
            tuneWinogradGemmDot2ModeSelect(
              ctx, modelDesc, problemBatchSize, nnXLen, nnYLen, iters, out, verboseTuner, tunedConfig);
        }
        if(supportsDot2AccF16) {
          if(tunedConfig.gemmStridedDot2AccF16TunerValueValid != 0)
            tuneGemmStridedDot2AccF16ModeSelect(
              ctx, modelDesc, problemBatchSize, nnXLen, nnYLen, iters, out, verboseTuner, tunedConfig);
          if(tunedConfig.winogradGemmDot2AccF16TunerValueValid != 0)
            tuneWinogradGemmDot2AccF16ModeSelect(
              ctx, modelDesc, problemBatchSize, nnXLen, nnYLen, iters, out, verboseTuner, tunedConfig);
        }
      } else {
        out << endl << "VulkanTuner: skip dot2 mode-select (";
        appendMissingVulkanFp16FeatureReason(
          out,
          "dot2F16/dot2F16AccF16",
          info,
          supportsAnyDot2,
          unfilteredSupportsAnyDot2,
          filterDisabledAnyDot2,
          ctx.fp16Storage,
          supportsFP16Compute);
        out << ")." << endl;
      }
    }

    // ---- Coopmat (VK_KHR_cooperative_matrix) mode-select ----

    struct CoopmatTile {
      int32_t blockSize, bm, bn, bk, wm, wn, tm, tn, tk, warp;
    };

    std::string coopmatTileToString(const CoopmatTile& t) {
      return Global::strprintf(
        "[bs=%d bm=%d bn=%d bk=%d wm=%d wn=%d tm=%d tn=%d tk=%d warp=%d]",
        t.blockSize,
        t.bm,
        t.bn,
        t.bk,
        t.wm,
        t.wn,
        t.tm,
        t.tn,
        t.tk,
        t.warp);
    }

    // Build coopmat tile candidates from the device's reported fragment shapes.
    // TM/TN/TK come verbatim from each shape; only the block/warp tiling is varied.
    // Every candidate satisfies coopmatConfigSupported by construction.
    std::vector<CoopmatTile> makeCoopmatCandidates(const TuningContext& ctx, const std::vector<CoopmatShape>& shapes) {
      std::vector<CoopmatTile> out;
      if(ctx.dev == nullptr)
        return out;
      const uint32_t warp = ctx.dev->info.subgroupSize > 0 ? ctx.dev->info.subgroupSize : 32u;
      for(const auto& shape: shapes) {
        const int32_t tm = (int32_t)shape.m;
        const int32_t tn = (int32_t)shape.n;
        const int32_t tk = (int32_t)shape.k;
        if(tm <= 0 || tn <= 0 || tk <= 0)
          continue;
        // WM/WN warp-tile multipliers over the fragment; BK is a small multiple of TK.
        // Include non-power-of-two CTA multiples so tiles can fit common KataGo
        // dimensions such as M=320/384, N=192/384/512, and K=192 cleanly.
        // validate() and shared-memory limits prune the device-inappropriate cases.
        // (bm/wm)*(bn/wn)*warp == blockSize is enforced by choosing blockSize to match.
        static const int warpTileMul[] = {1, 2, 4, 8, 16, 32, 64};
        static const int bmMulOverWm[] = {1, 2, 3, 4, 5, 6};  // BM = wm * this
        static const int bnMulOverWn[] = {1, 2, 3, 4, 6, 8};
        static const int bkMul[] = {1, 2, 3, 4, 6, 8};
        constexpr int maxBm = 384;
        constexpr int maxBn = 512;
        for(int wmm: warpTileMul)
          for(int wnm: warpTileMul)
            for(int bmm: bmMulOverWm)
              for(int bnm: bnMulOverWn)
                for(int bkm: bkMul) {
                  const int32_t wm = tm * wmm;
                  const int32_t wn = tn * wnm;
                  const int32_t bm = wm * bmm;
                  const int32_t bn = wn * bnm;
                  const int32_t bk = tk * bkm;
                  if(wm > 64 || wn > 64 || bm > maxBm || bn > maxBn)
                    continue;
                  const int32_t blockSize = (bm / wm) * (bn / wn) * (int32_t)warp;
                  CoopmatTile t{blockSize, bm, bn, bk, wm, wn, tm, tn, tk, (int32_t)warp};
                  if(
                    VulkanKernels::WinogradGemmCoopmat::isConfigSupported(
                      t.blockSize, t.bm, t.bn, t.bk, t.wm, t.wn, t.tm, t.tn, t.tk, t.warp))
                    out.push_back(t);
                }
      }
      return out;
    }

    int roundUpCoopmatDim(int v, int tile) {
      return ((v + tile - 1) / tile) * tile;
    }

    struct RankedCoopmatTile {
      CoopmatTile tile;
      double score;
    };

    // Tile-agnostic trim cost model shared by coopmat1 and coopmat2. Reads only the
    // block/tile scalars (bm/bn/bk/blockSize), so it applies to any tile representation.
    double tileTrimScore(
      int bm,
      int bn,
      int bk,
      int blockSize,
      int m,
      int n,
      int k,
      bool padM,
      bool padN,
      bool padK,
      double targetTileArea,
      double targetBk,
      double targetBlockSize) {
      const int paddedM = padM ? roundUpCoopmatDim(m, bm) : m;
      const int paddedN = padN ? roundUpCoopmatDim(n, bn) : n;
      const int paddedK = padK ? roundUpCoopmatDim(k, bk) : k;
      const double baseWork = std::max(1.0, (double)m * (double)n * (double)k);
      const double paddedWork = (double)paddedM * (double)paddedN * (double)paddedK;
      const double paddingRatio = paddedWork / baseWork;
      const double tileArea = std::max(1.0, (double)bm * (double)bn);
      const double areaPenalty = std::abs(std::log2(tileArea / targetTileArea));
      const double bkPenalty = std::abs(std::log2(std::max(1.0, (double)bk) / targetBk));
      const double blockPenalty = std::abs(std::log2(std::max(1.0, (double)blockSize) / targetBlockSize));
      return paddingRatio + 0.06 * areaPenalty + 0.04 * bkPenalty + 0.03 * blockPenalty;
    }

    double coopmatTrimScore(
      const CoopmatTile& t,
      int m,
      int n,
      int k,
      bool padM,
      bool padN,
      bool padK,
      double targetTileArea,
      double targetBk,
      double targetBlockSize) {
      return tileTrimScore(
        t.bm, t.bn, t.bk, t.blockSize, m, n, k, padM, padN, padK, targetTileArea, targetBk, targetBlockSize);
    }

    std::vector<CoopmatTile> trimCoopmatCandidates(
      const std::vector<CoopmatTile>& candidates,
      int m,
      int n,
      int k,
      bool padM,
      bool padN,
      bool padK,
      size_t maxCandidates,
      int minBlockSize,
      int maxBlockSize,
      int minBm,
      int minBn,
      int maxBk,
      double maxPaddingRatio,
      double targetTileArea,
      double targetBk,
      double targetBlockSize) {
      std::vector<RankedCoopmatTile> ranked;
      ranked.reserve(candidates.size());
      for(const CoopmatTile& t: candidates) {
        if(
          t.blockSize < minBlockSize || t.blockSize > maxBlockSize || t.bm < minBm || t.bn < minBn || t.bk > maxBk ||
          t.wm < t.tm * 2 || t.wn < t.tn * 2)
          continue;
        const int paddedM = padM ? roundUpCoopmatDim(m, t.bm) : m;
        const int paddedN = padN ? roundUpCoopmatDim(n, t.bn) : n;
        const int paddedK = padK ? roundUpCoopmatDim(k, t.bk) : k;
        const double baseWork = std::max(1.0, (double)m * (double)n * (double)k);
        const double paddingRatio = ((double)paddedM * (double)paddedN * (double)paddedK) / baseWork;
        if(paddingRatio > maxPaddingRatio)
          continue;
        ranked.push_back(
          {t, coopmatTrimScore(t, m, n, k, padM, padN, padK, targetTileArea, targetBk, targetBlockSize)});
      }

      if(ranked.empty()) {
        for(const CoopmatTile& t: candidates) {
          ranked.push_back(
            {t, coopmatTrimScore(t, m, n, k, padM, padN, padK, targetTileArea, targetBk, targetBlockSize)});
        }
      }

      std::stable_sort(ranked.begin(), ranked.end(), [](const RankedCoopmatTile& a, const RankedCoopmatTile& b) {
        return a.score < b.score;
      });
      if(ranked.size() > maxCandidates)
        ranked.resize(maxCandidates);

      std::vector<CoopmatTile> trimmed;
      trimmed.reserve(ranked.size());
      for(const RankedCoopmatTile& r: ranked)
        trimmed.push_back(r.tile);
      return trimmed;
    }

    void applyWinogradCoopmatTile(VulkanTuneParams& cfg, const CoopmatTile& t) {
      cfg.coopmatBlockSize = t.blockSize;
      cfg.coopmatBM = t.bm;
      cfg.coopmatBN = t.bn;
      cfg.coopmatBK = t.bk;
      cfg.coopmatWM = t.wm;
      cfg.coopmatWN = t.wn;
      cfg.coopmatTM = t.tm;
      cfg.coopmatTN = t.tn;
      cfg.coopmatTK = t.tk;
      cfg.coopmatWarp = t.warp;
    }

    void applyStridedCoopmatTile(VulkanTuneParams& cfg, const CoopmatTile& t) {
      cfg.stridedCoopmatBlockSize = t.blockSize;
      cfg.stridedCoopmatBM = t.bm;
      cfg.stridedCoopmatBN = t.bn;
      cfg.stridedCoopmatBK = t.bk;
      cfg.stridedCoopmatWM = t.wm;
      cfg.stridedCoopmatWN = t.wn;
      cfg.stridedCoopmatTM = t.tm;
      cfg.stridedCoopmatTN = t.tn;
      cfg.stridedCoopmatTK = t.tk;
      cfg.stridedCoopmatWarp = t.warp;
    }

    void applyWinogradCoopmatAccF16Tile(VulkanTuneParams& cfg, const CoopmatTile& t) {
      cfg.coopmatAccF16BlockSize = t.blockSize;
      cfg.coopmatAccF16BM = t.bm;
      cfg.coopmatAccF16BN = t.bn;
      cfg.coopmatAccF16BK = t.bk;
      cfg.coopmatAccF16WM = t.wm;
      cfg.coopmatAccF16WN = t.wn;
      cfg.coopmatAccF16TM = t.tm;
      cfg.coopmatAccF16TN = t.tn;
      cfg.coopmatAccF16TK = t.tk;
      cfg.coopmatAccF16Warp = t.warp;
    }

    void applyStridedCoopmatAccF16Tile(VulkanTuneParams& cfg, const CoopmatTile& t) {
      cfg.stridedCoopmatAccF16BlockSize = t.blockSize;
      cfg.stridedCoopmatAccF16BM = t.bm;
      cfg.stridedCoopmatAccF16BN = t.bn;
      cfg.stridedCoopmatAccF16BK = t.bk;
      cfg.stridedCoopmatAccF16WM = t.wm;
      cfg.stridedCoopmatAccF16WN = t.wn;
      cfg.stridedCoopmatAccF16TM = t.tm;
      cfg.stridedCoopmatAccF16TN = t.tn;
      cfg.stridedCoopmatAccF16TK = t.tk;
      cfg.stridedCoopmatAccF16Warp = t.warp;
    }

    bool coopmatTilesEqual(const CoopmatTile& a, const CoopmatTile& b) {
      return a.blockSize == b.blockSize && a.bm == b.bm && a.bn == b.bn && a.bk == b.bk && a.wm == b.wm &&
             a.wn == b.wn && a.tm == b.tm && a.tn == b.tn && a.tk == b.tk && a.warp == b.warp;
    }

    void appendUniqueCoopmatTile(std::vector<CoopmatTile>& out, const CoopmatTile& tile) {
      for(const CoopmatTile& existing: out)
        if(coopmatTilesEqual(existing, tile))
          return;
      out.push_back(tile);
    }

    void appendNvidiaShmemFp32VisitTiles(std::vector<CoopmatTile>& out, const std::vector<CoopmatTile>& candidates) {
      auto appendMatching = [&](int32_t bm, int32_t bn) {
        for(const CoopmatTile& t: candidates) {
          const int32_t nvidiaWorkgroupSize = t.warp * 8;
          if(nvidiaWorkgroupSize > 256)
            continue;
          if(t.bm == bm && t.bn == bn && t.bk == 16 && t.blockSize == nvidiaWorkgroupSize)
            appendUniqueCoopmatTile(out, t);
        }
      };

      // Mirrors vk_cooperative_matrix_perf's shmem fp16->fp32 sweep:
      // TILE_M/TILE_N = 128/256, TILE_K = 16, and skip 256x256 for fp32 result.
      appendMatching(128, 128);
      appendMatching(128, 256);
      appendMatching(256, 128);
    }

    // Sweep coopmat tiles for the Winograd trunk GEMM over the same aggregate
    // cases used by mode-select, gated on normalized RMSE <= 0.02 vs the tiled
    // reference for every case.
    void tuneWinogradCoopmatTileSweep(
      TuningContext& ctx,
      const ModelDesc* modelDesc,
      int problemBatchSize,
      int nnXLen,
      int nnYLen,
      int iters,
      const std::vector<CoopmatTile>& candidates,
      bool accF16,
      std::ostream& out,
      bool verboseTuner,
      VulkanTuneParams& tunedConfig) {
      if(candidates.empty())
        return;
      std::vector<WinogradModeCase> sweepCases =
        collectWinogradModeCases(modelDesc, problemBatchSize, nnXLen, nnYLen, tunedConfig);
      if(sweepCases.empty())
        return;
      const WinogradModeCase& sc0 = sweepCases[0];
      std::vector<CoopmatTile> sweepCandidates;
      appendNvidiaShmemFp32VisitTiles(sweepCandidates, candidates);
      std::vector<CoopmatTile> trimmedCandidates = trimCoopmatCandidates(
        candidates, sc0.M, sc0.N, sc0.K, true, true, true, 64, 64, 256, 64, 32, 32, 1.60, 32768.0, 16.0, 256.0);
      for(const CoopmatTile& t: trimmedCandidates)
        appendUniqueCoopmatTile(sweepCandidates, t);
      if(sweepCandidates.empty())
        return;

      struct CachedWinogradRef {
        int m, n, k, numBatches;
        KernelBench bench;
      };
      std::vector<CachedWinogradRef> tiledRefs;
      auto getTiledRef = [&](int m, int n, int k, int numBatches) -> const KernelBench* {
        for(const CachedWinogradRef& ref: tiledRefs)
          if(ref.m == m && ref.n == n && ref.k == k && ref.numBatches == numBatches)
            return &ref.bench;
        VulkanKernels::WinogradGemm tiledRefProbe(m, n, k, numBatches);
        tiledRefs.push_back({m, n, k, numBatches, tiledRefProbe.bench(ctx, tunedConfig, iters)});
        return &tiledRefs.back().bench;
      };

      auto benchAggregateTile = [&](const CoopmatTile& t, AggregateCandidateMetrics& metrics) {
        VulkanTuneParams trial = tunedConfig;
        if(accF16)
          applyWinogradCoopmatAccF16Tile(trial, t);
        else
          applyWinogradCoopmatTile(trial, t);
        metrics.weightedTime = 0.0;
        metrics.tflops = 0.0;
        metrics.maxRmse = 0.0;
        double aggregateFlops = 0.0;
        double aggregateSeconds = 0.0;
        for(const WinogradModeCase& cs: sweepCases) {
          const auto pad =
            accF16 ? padDims(cs.M, cs.N, cs.K, VulkanKernels::WinogradGemmCoopmatAccF16::layerPaddingContract(trial))
                   : padDims(cs.M, cs.N, cs.K, VulkanKernels::WinogradGemmCoopmat::layerPaddingContract(trial));
          KernelBench b;
          double flopsPerDispatch = 0.0;
          if(accF16) {
            VulkanKernels::WinogradGemmCoopmatAccF16 probe(pad.m, pad.n, pad.k, cs.numBatches);
            if(!probe.validate(trial, ctx.dev->info.properties.limits))
              return rejectAggregateCandidate(metrics, "validate_failed");
            b = probe.bench(ctx, trial, iters);
            flopsPerDispatch = probe.estimatedFlopsPerDispatch(trial);
          } else {
            VulkanKernels::WinogradGemmCoopmat probe(pad.m, pad.n, pad.k, cs.numBatches);
            if(!probe.validate(trial, ctx.dev->info.properties.limits))
              return rejectAggregateCandidate(metrics, "validate_failed");
            b = probe.bench(ctx, trial, iters);
            flopsPerDispatch = probe.estimatedFlopsPerDispatch(trial);
          }
          if(!(b.ok && b.kernelsPerSecond > 0.0))
            return rejectAggregateCandidate(metrics, "bench_failed");
          const KernelBench* tiledRef = getTiledRef(pad.m, pad.n, pad.k, cs.numBatches);
          if(!tiledRef->ok)
            return rejectAggregateCandidate(metrics, "reference_failed");
          double rmse = normalizedRmse(tiledRef->output, b.output);
          metrics.maxRmse = std::max(metrics.maxRmse, rmse);
          if(rmse > 0.02)
            return rejectAggregateCandidate(metrics, "rmse_exceeded");
          const double caseWeightedTime = dispatchWeightedTime(b, cs.occurrences);
          metrics.weightedTime += caseWeightedTime;
          aggregateFlops += flopsPerDispatch * (double)cs.occurrences;
          aggregateSeconds += caseWeightedTime;
        }
        metrics.tflops = aggregateTflops(aggregateFlops, aggregateSeconds);
        return metrics.weightedTime > 0.0;
      };

      string_view label = accF16 ? "winogradCoopmatAccF16" : "winogradCoopmat";
      out << "VulkanTuner: " << label << " tile sweep candidates=" << sweepCandidates.size()
          << " raw_candidates=" << candidates.size() << " cases=" << sweepCases.size() << " trim_case M=" << sc0.M
          << " N=" << sc0.N << " K=" << sc0.K << " transform_planes=" << sc0.numBatches << endl;
      AggregateCandidateSweepResult<CoopmatTile> result = tuneAggregateCandidateSweep(
        TunerStreamText{label},
        TunerStreamText{label},
        TunerStreamText{label, " tile sweep"},
        "tile",
        TunerStreamText{"VulkanTuner: ", label, " tile sweep found no valid tile; keeping defaults."},
        sweepCandidates,
        TUNER_CONFIRM_PREFIX_K,
        true,
        true,
        true,
        out,
        verboseTuner,
        benchAggregateTile,
        coopmatTileToString,
        coopmatTilesEqual);
      if(result.found) {
        if(accF16) {
          applyWinogradCoopmatAccF16Tile(tunedConfig, result.candidate);
          tunedConfig.winogradGemmCoopmatAccF16TunerValueValid = 1;
        } else {
          applyWinogradCoopmatTile(tunedConfig, result.candidate);
          tunedConfig.winogradGemmCoopmatTunerValueValid = 1;
        }
      }
    }

    // Strided coopmat tile sweep: keep fastest valid+correct tile by aggregate
    // weighted time over the same 1x1/transformer cases used by mode-select.
    void tuneGemmStridedCoopmatTileSweep(
      TuningContext& ctx,
      const ModelDesc* modelDesc,
      int problemBatchSize,
      int nnXLen,
      int nnYLen,
      int iters,
      const std::vector<CoopmatTile>& candidates,
      bool accF16,
      std::ostream& out,
      bool verboseTuner,
      VulkanTuneParams& tunedConfig) {
      if(candidates.empty())
        return;
      std::vector<GemmStridedModeCase> modeCases = collectGemmStridedModeCases(modelDesc, nnXLen, nnYLen);
      if(modeCases.empty()) {
        const TrunkDesc& trunk = modelDesc->trunk;
        int trunkC = roundUpToMultipleInt(trunk.trunkNumChannels > 0 ? trunk.trunkNumChannels : 64, 8);
        int paddedSpatialSize = roundUpToMultipleInt(nnXLen * nnYLen, VulkanKernels::VULKAN_SPATIAL_ALIGN);
        modeCases.push_back({paddedSpatialSize, trunkC, trunkC, 1, (double)trunkC * (double)trunkC});
      }
      const GemmStridedModeCase& cs = modeCases[0];
      // N padded to the candidate's BN; M/K stay real (strided contract m=k=1).
      std::vector<CoopmatTile> sweepCandidates = trimCoopmatCandidates(
        candidates, cs.M, cs.N, cs.K, false, true, false, 64, 32, 256, 32, 32, 32, 1.35, 4096.0, 32.0, 64.0);
      if(sweepCandidates.empty())
        return;

      struct CachedStridedRef {
        int m, n, k;
        KernelBench bench;
      };
      std::vector<CachedStridedRef> tiledRefs;
      auto getTiledRef = [&](int m, int n, int k) -> const KernelBench* {
        for(const CachedStridedRef& ref: tiledRefs)
          if(ref.m == m && ref.n == n && ref.k == k)
            return &ref.bench;
        VulkanKernels::GemmStridedTiled tiledAtDims(problemBatchSize, m, n, k);
        tiledRefs.push_back({m, n, k, tiledAtDims.bench(ctx, tunedConfig, iters)});
        return &tiledRefs.back().bench;
      };

      auto benchAggregateTile = [&](const CoopmatTile& t, AggregateCandidateMetrics& metrics) {
        VulkanTuneParams trial = tunedConfig;
        if(accF16)
          applyStridedCoopmatAccF16Tile(trial, t);
        else
          applyStridedCoopmatTile(trial, t);
        metrics.weightedTime = 0.0;
        metrics.tflops = 0.0;
        metrics.maxRmse = 0.0;
        double aggregateFlops = 0.0;
        double aggregateSeconds = 0.0;
        for(const GemmStridedModeCase& modeCase: modeCases) {
          const auto pad =
            accF16
              ? padDims(
                  modeCase.M,
                  modeCase.N,
                  modeCase.K,
                  VulkanKernels::GemmStridedCoopmatAccF16::layerPaddingContract(trial))
              : padDims(
                  modeCase.M, modeCase.N, modeCase.K, VulkanKernels::GemmStridedCoopmat::layerPaddingContract(trial));
          KernelBench b;
          double flopsPerDispatch = 0.0;
          if(accF16) {
            VulkanKernels::GemmStridedCoopmatAccF16 probe(problemBatchSize, pad.m, pad.n, pad.k);
            if(!probe.validate(trial, ctx.dev->info.properties.limits))
              return rejectAggregateCandidate(metrics, "validate_failed");
            b = probe.bench(ctx, trial, iters);
            flopsPerDispatch = probe.estimatedFlopsPerDispatch(trial);
          } else {
            VulkanKernels::GemmStridedCoopmat probe(problemBatchSize, pad.m, pad.n, pad.k);
            if(!probe.validate(trial, ctx.dev->info.properties.limits))
              return rejectAggregateCandidate(metrics, "validate_failed");
            b = probe.bench(ctx, trial, iters);
            flopsPerDispatch = probe.estimatedFlopsPerDispatch(trial);
          }
          if(!(b.ok && b.kernelsPerSecond > 0.0))
            return rejectAggregateCandidate(metrics, "bench_failed");
          const KernelBench* ref = getTiledRef(pad.m, pad.n, pad.k);
          if(!ref->ok)
            return rejectAggregateCandidate(metrics, "reference_failed");
          double rmse = normalizedRmse(ref->output, b.output);
          metrics.maxRmse = std::max(metrics.maxRmse, rmse);
          if(rmse > 0.02)
            return rejectAggregateCandidate(metrics, "rmse_exceeded");
          const double caseWeightedTime = dispatchWeightedTime(b, modeCase.occurrences);
          metrics.weightedTime += caseWeightedTime;
          aggregateFlops += flopsPerDispatch * (double)modeCase.occurrences;
          aggregateSeconds += caseWeightedTime;
        }
        metrics.tflops = aggregateTflops(aggregateFlops, aggregateSeconds);
        return metrics.weightedTime > 0.0;
      };

      string_view label = accF16 ? "gemmStridedCoopmatAccF16" : "gemmStridedCoopmat";
      out << "VulkanTuner: " << label << " tile sweep candidates=" << sweepCandidates.size()
          << " raw_candidates=" << candidates.size() << " cases=" << modeCases.size() << " trim_case M=" << cs.M
          << " N=" << cs.N << " K=" << cs.K << " batch=" << problemBatchSize << endl;
      AggregateCandidateSweepResult<CoopmatTile> result = tuneAggregateCandidateSweep(
        TunerStreamText{label},
        TunerStreamText{label},
        TunerStreamText{label, " tile sweep"},
        "tile",
        TunerStreamText{"VulkanTuner: ", label, " tile sweep found no valid tile; keeping defaults."},
        sweepCandidates,
        TUNER_CONFIRM_PREFIX_K,
        true,
        true,
        true,
        out,
        verboseTuner,
        benchAggregateTile,
        coopmatTileToString,
        coopmatTilesEqual);
      if(result.found) {
        if(accF16) {
          applyStridedCoopmatAccF16Tile(tunedConfig, result.candidate);
          tunedConfig.gemmStridedCoopmatAccF16TunerValueValid = 1;
        } else {
          applyStridedCoopmatTile(tunedConfig, result.candidate);
          tunedConfig.gemmStridedCoopmatTunerValueValid = 1;
        }
      }
    }

    // Decide whether coopmat beats the current best (tiled/dot2) for each
    // GEMM path, flipping the enable flag only on a faster + RMSE-correct win.
    void tuneWinogradCoopmatModeSelect(
      TuningContext& ctx,
      const ModelDesc* modelDesc,
      int problemBatchSize,
      int nnXLen,
      int nnYLen,
      int iters,
      bool accF16,
      std::ostream& out,
      bool verboseTuner,
      VulkanTuneParams& tunedConfig) {
      std::vector<WinogradModeCase> wCases =
        collectWinogradModeCases(modelDesc, problemBatchSize, nnXLen, nnYLen, tunedConfig);
      if(wCases.empty())
        return;

      const bool incumbentCoopmat2AccF16 = tunedConfig.enableWinogradGemmCoopmat2AccF16 != 0;
      const bool incumbentCoopmat2 = !incumbentCoopmat2AccF16 && tunedConfig.enableWinogradGemmCoopmat2 != 0;
      const bool incumbentCoopmatAccF16 =
        !incumbentCoopmat2AccF16 && !incumbentCoopmat2 && !accF16 && tunedConfig.enableWinogradGemmCoopmatAccF16 != 0;
      const bool incumbentCoopmat = !incumbentCoopmat2AccF16 && !incumbentCoopmat2 && !incumbentCoopmatAccF16 &&
                                    accF16 && tunedConfig.enableWinogradGemmCoopmat != 0;
      const bool incumbentDot2AccF16 = !incumbentCoopmat2AccF16 && !incumbentCoopmat2 && !incumbentCoopmatAccF16 &&
                                       !incumbentCoopmat && tunedConfig.enableWinogradGemmDot2AccF16 != 0;
      const bool incumbentDot2 = !incumbentCoopmat2AccF16 && !incumbentCoopmat2 && !incumbentCoopmatAccF16 &&
                                 !incumbentCoopmat && !incumbentDot2AccF16 && tunedConfig.enableWinogradGemmDot2 != 0;
      string_view incumbentName = incumbentCoopmat2AccF16  ? "winogradGemmCoopmat2AccF16"
                                  : incumbentCoopmat2      ? "winogradGemmCoopmat2"
                                  : incumbentCoopmatAccF16 ? "winogradGemmCoopmatAccF16"
                                  : incumbentCoopmat       ? "winogradGemmCoopmat"
                                  : incumbentDot2AccF16    ? "winogradGemmDot2AccF16"
                                  : incumbentDot2          ? "winogradGemmDot2"
                                                           : "winogradGemmTiled";

      bool incumbentAllOk = true;
      bool coopmatAllOk = true;
      double incumbentWeightedTime = 0.0;
      double coopmatWeightedTime = 0.0;
      double incumbentWeightedFlops = 0.0;
      double coopmatWeightedFlops = 0.0;
      double maxCoopmatSpeedup = 0.0;
      double maxRmse = 0.0;

      string_view challengerName = accF16 ? "winogradGemmCoopmatAccF16" : "winogradGemmCoopmat";
      out << endl
          << "VulkanTuner: winogradGemm " << (accF16 ? "coopmatAccF16" : "coopmat") << " mode-select cases ("
          << wCases.size() << ") ..." << endl;
      for(size_t i = 0; i < wCases.size(); i++) {
        const WinogradModeCase& cs = wCases[i];
        if(verboseTuner)
          out << "VulkanTuner:   raw case#" << i << " M=" << cs.M << " N=" << cs.N << " K=" << cs.K
              << " transform_planes=" << cs.numBatches << " occurrences=" << cs.occurrences
              << " weight=" << (uint64_t)std::llround(cs.weightedWork) << endl;

        KernelBench incumbentBench;
        double incumbentTflops = 0.0;
        double incumbentFlopsPerDispatch = 0.0;
        if(incumbentCoopmat2AccF16) {
          const auto pad =
            padDims(cs.M, cs.N, cs.K, VulkanKernels::WinogradGemmCoopmat2AccF16::layerPaddingContract(tunedConfig));
          VulkanKernels::WinogradGemmCoopmat2AccF16 probe(pad.m, pad.n, pad.k, cs.numBatches);
          incumbentBench = probe.bench(ctx, tunedConfig, iters);
          incumbentTflops = probe.estimatedTflops(tunedConfig, incumbentBench.kernelsPerSecond);
          incumbentFlopsPerDispatch = probe.estimatedFlopsPerDispatch(tunedConfig);
        } else if(incumbentCoopmat2) {
          const auto pad =
            padDims(cs.M, cs.N, cs.K, VulkanKernels::WinogradGemmCoopmat2::layerPaddingContract(tunedConfig));
          VulkanKernels::WinogradGemmCoopmat2 probe(pad.m, pad.n, pad.k, cs.numBatches);
          incumbentBench = probe.bench(ctx, tunedConfig, iters);
          incumbentTflops = probe.estimatedTflops(tunedConfig, incumbentBench.kernelsPerSecond);
          incumbentFlopsPerDispatch = probe.estimatedFlopsPerDispatch(tunedConfig);
        } else if(incumbentCoopmatAccF16) {
          const auto pad =
            padDims(cs.M, cs.N, cs.K, VulkanKernels::WinogradGemmCoopmatAccF16::layerPaddingContract(tunedConfig));
          VulkanKernels::WinogradGemmCoopmatAccF16 probe(pad.m, pad.n, pad.k, cs.numBatches);
          incumbentBench = probe.bench(ctx, tunedConfig, iters);
          incumbentTflops = probe.estimatedTflops(tunedConfig, incumbentBench.kernelsPerSecond);
          incumbentFlopsPerDispatch = probe.estimatedFlopsPerDispatch(tunedConfig);
        } else if(incumbentCoopmat) {
          const auto pad =
            padDims(cs.M, cs.N, cs.K, VulkanKernels::WinogradGemmCoopmat::layerPaddingContract(tunedConfig));
          VulkanKernels::WinogradGemmCoopmat probe(pad.m, pad.n, pad.k, cs.numBatches);
          incumbentBench = probe.bench(ctx, tunedConfig, iters);
          incumbentTflops = probe.estimatedTflops(tunedConfig, incumbentBench.kernelsPerSecond);
          incumbentFlopsPerDispatch = probe.estimatedFlopsPerDispatch(tunedConfig);
        } else if(incumbentDot2AccF16) {
          const auto dPad =
            padDims(cs.M, cs.N, cs.K, VulkanKernels::WinogradGemmDot2AccF16::layerPaddingContract(tunedConfig));
          VulkanKernels::WinogradGemmDot2AccF16 dProbe(dPad.m, dPad.n, dPad.k, cs.numBatches);
          incumbentBench = dProbe.bench(ctx, tunedConfig, iters);
          incumbentTflops = dProbe.estimatedTflops(tunedConfig, incumbentBench.kernelsPerSecond);
          incumbentFlopsPerDispatch = dProbe.estimatedFlopsPerDispatch(tunedConfig);
        } else if(incumbentDot2) {
          const auto dPad =
            padDims(cs.M, cs.N, cs.K, VulkanKernels::WinogradGemmDot2::layerPaddingContract(tunedConfig));
          VulkanKernels::WinogradGemmDot2 dProbe(dPad.m, dPad.n, dPad.k, cs.numBatches);
          incumbentBench = dProbe.bench(ctx, tunedConfig, iters);
          incumbentTflops = dProbe.estimatedTflops(tunedConfig, incumbentBench.kernelsPerSecond);
          incumbentFlopsPerDispatch = dProbe.estimatedFlopsPerDispatch(tunedConfig);
        } else {
          const auto tPad = padDims(cs.M, cs.N, cs.K, VulkanKernels::WinogradGemm::layerPaddingContract(tunedConfig));
          VulkanKernels::WinogradGemm tProbe(tPad.m, tPad.n, tPad.k, cs.numBatches);
          incumbentBench = tProbe.bench(ctx, tunedConfig, iters);
          incumbentTflops = tProbe.estimatedTflops(tunedConfig, incumbentBench.kernelsPerSecond);
          incumbentFlopsPerDispatch = tProbe.estimatedFlopsPerDispatch(tunedConfig);
        }
        const bool incumbentOk = incumbentBench.ok && incumbentBench.kernelsPerSecond > 0.0;
        double incumbentCaseWeightedTime = 0.0;
        if(incumbentOk) {
          incumbentCaseWeightedTime = dispatchWeightedTime(incumbentBench, cs.occurrences);
          incumbentWeightedTime += incumbentCaseWeightedTime;
          incumbentWeightedFlops += incumbentFlopsPerDispatch * (double)cs.occurrences;
        } else
          incumbentAllOk = false;

        const auto cmPad =
          accF16
            ? padDims(cs.M, cs.N, cs.K, VulkanKernels::WinogradGemmCoopmatAccF16::layerPaddingContract(tunedConfig))
            : padDims(cs.M, cs.N, cs.K, VulkanKernels::WinogradGemmCoopmat::layerPaddingContract(tunedConfig));
        KernelBench cmBench;
        double coopmatTflops = 0.0;
        double coopmatFlopsPerDispatch = 0.0;
        if(accF16) {
          VulkanKernels::WinogradGemmCoopmatAccF16 cmProbe(cmPad.m, cmPad.n, cmPad.k, cs.numBatches);
          cmBench = cmProbe.bench(ctx, tunedConfig, iters);
          coopmatTflops = cmProbe.estimatedTflops(tunedConfig, cmBench.kernelsPerSecond);
          coopmatFlopsPerDispatch = cmProbe.estimatedFlopsPerDispatch(tunedConfig);
        } else {
          VulkanKernels::WinogradGemmCoopmat cmProbe(cmPad.m, cmPad.n, cmPad.k, cs.numBatches);
          cmBench = cmProbe.bench(ctx, tunedConfig, iters);
          coopmatTflops = cmProbe.estimatedTflops(tunedConfig, cmBench.kernelsPerSecond);
          coopmatFlopsPerDispatch = cmProbe.estimatedFlopsPerDispatch(tunedConfig);
        }
        bool coopmatOk = cmBench.ok && cmBench.kernelsPerSecond > 0.0;
        double coopmatCaseWeightedTime = 0.0;
        double rmse = std::numeric_limits<double>::infinity();
        if(coopmatOk) {
          VulkanKernels::WinogradGemm tiledRefProbe(cmPad.m, cmPad.n, cmPad.k, cs.numBatches);
          KernelBench tiledRef = tiledRefProbe.bench(ctx, tunedConfig, iters);
          if(tiledRef.ok) {
            rmse = normalizedRmse(tiledRef.output, cmBench.output);
            maxRmse = std::max(maxRmse, rmse);
            coopmatOk = rmse <= 0.02;
          } else
            coopmatOk = false;
        }
        if(coopmatOk) {
          coopmatCaseWeightedTime = dispatchWeightedTime(cmBench, cs.occurrences);
          coopmatWeightedTime += coopmatCaseWeightedTime;
          coopmatWeightedFlops += coopmatFlopsPerDispatch * (double)cs.occurrences;
        } else
          coopmatAllOk = false;
        if(incumbentOk && coopmatOk && coopmatCaseWeightedTime > 0.0)
          maxCoopmatSpeedup = std::max(maxCoopmatSpeedup, incumbentCaseWeightedTime / coopmatCaseWeightedTime);

        if(shouldPrintTunerResult(verboseTuner, false, incumbentOk && coopmatOk)) {
          out << "VulkanTuner:   winogradGemm " << (accF16 ? "coopmatAccF16" : "coopmat") << " case#" << i;
          if(incumbentOk) {
            out << " " << incumbentName << "_kps=" << incumbentBench.kernelsPerSecond;
            if(incumbentTflops > 0.0)
              out << " " << incumbentName << "_tflops=" << incumbentTflops;
            out << " " << incumbentName << "_weighted_time=" << incumbentCaseWeightedTime;
          } else
            out << " " << incumbentName << "=failed";
          if(cmBench.ok && cmBench.kernelsPerSecond > 0.0) {
            out << " " << challengerName << "_kps=" << cmBench.kernelsPerSecond;
            if(coopmatTflops > 0.0)
              out << " " << challengerName << "_tflops=" << coopmatTflops;
            out << " " << challengerName << "_weighted_time=" << coopmatCaseWeightedTime;
          } else
            out << " " << challengerName << "=failed";
          out << " rmse=" << (std::isfinite(rmse) ? Global::doubleToString(rmse) : string("n/a")) << " "
              << challengerName << "_result=" << (coopmatOk ? "accepted" : "rejected") << endl;
        }
      }

      ModeSelectStats incumbentStats{incumbentAllOk, incumbentWeightedTime, incumbentWeightedFlops};
      ModeSelectStats coopmatStats{coopmatAllOk, coopmatWeightedTime, coopmatWeightedFlops};
      bool selected = appendModeSelectResult(
        out,
        "winogradGemm",
        incumbentName,
        challengerName,
        wCases,
        incumbentStats,
        coopmatStats,
        maxRmse,
        true,
        true,
        [&](std::ostream& resultOut) {
          appendPositiveField(resultOut, "coopmat_avg_speedup", modeSelectSpeedup(incumbentStats, coopmatStats));
          appendPositiveField(resultOut, "coopmat_max_speedup", maxCoopmatSpeedup);
        });
      if(selected) {
        if(accF16) {
          tunedConfig.enableWinogradGemmCoopmatAccF16 = 1;
          promoteWinogradGemmSelectionRank(tunedConfig, &VulkanTuneParams::winogradGemmCoopmatAccF16SelectionRank);
        } else {
          tunedConfig.enableWinogradGemmCoopmat = 1;
          promoteWinogradGemmSelectionRank(tunedConfig, &VulkanTuneParams::winogradGemmCoopmatSelectionRank);
        }
        tunedConfig.enableWinogradGemmDot2 = 0;  // mutually exclusive at runtime
        tunedConfig.enableWinogradGemmDot2AccF16 = 0;
        tunedConfig.enableWinogradGemmCoopmat = accF16 ? 0 : tunedConfig.enableWinogradGemmCoopmat;
        tunedConfig.enableWinogradGemmCoopmatAccF16 = accF16 ? tunedConfig.enableWinogradGemmCoopmatAccF16 : 0;
        tunedConfig.enableWinogradGemmCoopmat2 = 0;
        tunedConfig.enableWinogradGemmCoopmat2AccF16 = 0;
        tunedConfig.enableWinogradGemmTiledFp16Compute = 0;
      } else if(
        !incumbentCoopmat2AccF16 && !incumbentCoopmat2 && !incumbentCoopmatAccF16 && !incumbentCoopmat &&
        !incumbentDot2AccF16 && !incumbentDot2)
        promoteWinogradGemmSelectionRank(tunedConfig, &VulkanTuneParams::winogradGemmTiledSelectionRank);
    }

    void tuneGemmStridedCoopmatModeSelect(
      TuningContext& ctx,
      const ModelDesc* modelDesc,
      int problemBatchSize,
      int nnXLen,
      int nnYLen,
      int iters,
      bool accF16,
      std::ostream& out,
      bool verboseTuner,
      VulkanTuneParams& tunedConfig) {
      std::vector<GemmStridedModeCase> modeCases = collectGemmStridedModeCases(modelDesc, nnXLen, nnYLen);
      if(modeCases.empty())
        return;

      const bool incumbentCoopmat2AccF16 = tunedConfig.enableGemmStridedCoopmat2AccF16 != 0;
      const bool incumbentCoopmat2 = !incumbentCoopmat2AccF16 && tunedConfig.enableGemmStridedCoopmat2 != 0;
      const bool incumbentCoopmatAccF16 =
        !incumbentCoopmat2AccF16 && !incumbentCoopmat2 && !accF16 && tunedConfig.enableGemmStridedCoopmatAccF16 != 0;
      const bool incumbentCoopmat = !incumbentCoopmat2AccF16 && !incumbentCoopmat2 && !incumbentCoopmatAccF16 &&
                                    accF16 && tunedConfig.enableGemmStridedCoopmat != 0;
      const bool incumbentDot2AccF16 = !incumbentCoopmat2AccF16 && !incumbentCoopmat2 && !incumbentCoopmatAccF16 &&
                                       !incumbentCoopmat && tunedConfig.enableGemmStridedDot2AccF16 != 0;
      const bool incumbentDot2 = !incumbentCoopmat2AccF16 && !incumbentCoopmat2 && !incumbentCoopmatAccF16 &&
                                 !incumbentCoopmat && !incumbentDot2AccF16 && tunedConfig.enableGemmStridedDot2 != 0;
      string_view incumbentName = incumbentCoopmat2AccF16  ? "gemmStridedCoopmat2AccF16"
                                  : incumbentCoopmat2      ? "gemmStridedCoopmat2"
                                  : incumbentCoopmatAccF16 ? "gemmStridedCoopmatAccF16"
                                  : incumbentCoopmat       ? "gemmStridedCoopmat"
                                  : incumbentDot2AccF16    ? "gemmStridedDot2AccF16"
                                  : incumbentDot2          ? "gemmStridedDot2"
                                                           : "gemmStridedTiled";

      bool incumbentAllOk = true;
      bool coopmatAllOk = true;
      double incumbentWeightedTime = 0.0;
      double coopmatWeightedTime = 0.0;
      double incumbentWeightedFlops = 0.0;
      double coopmatWeightedFlops = 0.0;
      double maxCoopmatSpeedup = 0.0;
      double maxRmse = 0.0;

      string_view challengerName = accF16 ? "gemmStridedCoopmatAccF16" : "gemmStridedCoopmat";
      out << endl
          << "VulkanTuner: gemmStrided " << (accF16 ? "coopmatAccF16" : "coopmat") << " mode-select cases ("
          << modeCases.size() << ") ..." << endl;
      for(size_t i = 0; i < modeCases.size(); i++) {
        const GemmStridedModeCase& cs = modeCases[i];
        if(verboseTuner)
          out << "VulkanTuner:   case#" << i << " M=" << cs.M << " N=" << cs.N << " K=" << cs.K
              << " occurrences=" << cs.occurrences << " weight=" << (uint64_t)std::llround(cs.weightedWork) << endl;

        KernelBench incumbentBench;
        double incumbentTflops = 0.0;
        double incumbentFlopsPerDispatch = 0.0;
        if(incumbentCoopmat2AccF16) {
          const auto pad =
            padDims(cs.M, cs.N, cs.K, VulkanKernels::GemmStridedCoopmat2AccF16::layerPaddingContract(tunedConfig));
          VulkanKernels::GemmStridedCoopmat2AccF16 probe(problemBatchSize, pad.m, pad.n, pad.k);
          incumbentBench = probe.bench(ctx, tunedConfig, iters);
          incumbentTflops = probe.estimatedTflops(tunedConfig, incumbentBench.kernelsPerSecond);
          incumbentFlopsPerDispatch = probe.estimatedFlopsPerDispatch(tunedConfig);
        } else if(incumbentCoopmat2) {
          const auto pad =
            padDims(cs.M, cs.N, cs.K, VulkanKernels::GemmStridedCoopmat2::layerPaddingContract(tunedConfig));
          VulkanKernels::GemmStridedCoopmat2 probe(problemBatchSize, pad.m, pad.n, pad.k);
          incumbentBench = probe.bench(ctx, tunedConfig, iters);
          incumbentTflops = probe.estimatedTflops(tunedConfig, incumbentBench.kernelsPerSecond);
          incumbentFlopsPerDispatch = probe.estimatedFlopsPerDispatch(tunedConfig);
        } else if(incumbentCoopmatAccF16) {
          const auto pad =
            padDims(cs.M, cs.N, cs.K, VulkanKernels::GemmStridedCoopmatAccF16::layerPaddingContract(tunedConfig));
          VulkanKernels::GemmStridedCoopmatAccF16 probe(problemBatchSize, pad.m, pad.n, pad.k);
          incumbentBench = probe.bench(ctx, tunedConfig, iters);
          incumbentTflops = probe.estimatedTflops(tunedConfig, incumbentBench.kernelsPerSecond);
          incumbentFlopsPerDispatch = probe.estimatedFlopsPerDispatch(tunedConfig);
        } else if(incumbentCoopmat) {
          const auto pad =
            padDims(cs.M, cs.N, cs.K, VulkanKernels::GemmStridedCoopmat::layerPaddingContract(tunedConfig));
          VulkanKernels::GemmStridedCoopmat probe(problemBatchSize, pad.m, pad.n, pad.k);
          incumbentBench = probe.bench(ctx, tunedConfig, iters);
          incumbentTflops = probe.estimatedTflops(tunedConfig, incumbentBench.kernelsPerSecond);
          incumbentFlopsPerDispatch = probe.estimatedFlopsPerDispatch(tunedConfig);
        } else if(incumbentDot2AccF16) {
          const auto dPad =
            padDims(cs.M, cs.N, cs.K, VulkanKernels::GemmStridedDot2AccF16::layerPaddingContract(tunedConfig));
          VulkanKernels::GemmStridedDot2AccF16 dProbe(problemBatchSize, dPad.m, dPad.n, dPad.k);
          incumbentBench = dProbe.bench(ctx, tunedConfig, iters);
          incumbentTflops = dProbe.estimatedTflops(tunedConfig, incumbentBench.kernelsPerSecond);
          incumbentFlopsPerDispatch = dProbe.estimatedFlopsPerDispatch(tunedConfig);
        } else if(incumbentDot2) {
          const auto dPad =
            padDims(cs.M, cs.N, cs.K, VulkanKernels::GemmStridedDot2::layerPaddingContract(tunedConfig));
          VulkanKernels::GemmStridedDot2 dProbe(problemBatchSize, dPad.m, dPad.n, dPad.k);
          incumbentBench = dProbe.bench(ctx, tunedConfig, iters);
          incumbentTflops = dProbe.estimatedTflops(tunedConfig, incumbentBench.kernelsPerSecond);
          incumbentFlopsPerDispatch = dProbe.estimatedFlopsPerDispatch(tunedConfig);
        } else {
          const auto tPad =
            padDims(cs.M, cs.N, cs.K, VulkanKernels::GemmStridedTiled::layerPaddingContract(tunedConfig));
          VulkanKernels::GemmStridedTiled tProbe(problemBatchSize, tPad.m, tPad.n, tPad.k);
          incumbentBench = tProbe.bench(ctx, tunedConfig, iters);
          incumbentTflops = tProbe.estimatedTflops(tunedConfig, incumbentBench.kernelsPerSecond);
          incumbentFlopsPerDispatch = tProbe.estimatedFlopsPerDispatch(tunedConfig);
        }
        const bool incumbentOk = incumbentBench.ok && incumbentBench.kernelsPerSecond > 0.0;
        double incumbentCaseWeightedTime = 0.0;
        if(incumbentOk) {
          incumbentCaseWeightedTime = dispatchWeightedTime(incumbentBench, cs.occurrences);
          incumbentWeightedTime += incumbentCaseWeightedTime;
          incumbentWeightedFlops += incumbentFlopsPerDispatch * (double)cs.occurrences;
        } else
          incumbentAllOk = false;

        const auto cmPad =
          accF16 ? padDims(cs.M, cs.N, cs.K, VulkanKernels::GemmStridedCoopmatAccF16::layerPaddingContract(tunedConfig))
                 : padDims(cs.M, cs.N, cs.K, VulkanKernels::GemmStridedCoopmat::layerPaddingContract(tunedConfig));
        KernelBench cmBench;
        double coopmatTflops = 0.0;
        double coopmatFlopsPerDispatch = 0.0;
        if(accF16) {
          VulkanKernels::GemmStridedCoopmatAccF16 cmProbe(problemBatchSize, cmPad.m, cmPad.n, cmPad.k);
          cmBench = cmProbe.bench(ctx, tunedConfig, iters);
          coopmatTflops = cmProbe.estimatedTflops(tunedConfig, cmBench.kernelsPerSecond);
          coopmatFlopsPerDispatch = cmProbe.estimatedFlopsPerDispatch(tunedConfig);
        } else {
          VulkanKernels::GemmStridedCoopmat cmProbe(problemBatchSize, cmPad.m, cmPad.n, cmPad.k);
          cmBench = cmProbe.bench(ctx, tunedConfig, iters);
          coopmatTflops = cmProbe.estimatedTflops(tunedConfig, cmBench.kernelsPerSecond);
          coopmatFlopsPerDispatch = cmProbe.estimatedFlopsPerDispatch(tunedConfig);
        }
        bool coopmatOk = cmBench.ok && cmBench.kernelsPerSecond > 0.0;
        double coopmatCaseWeightedTime = 0.0;
        double rmse = std::numeric_limits<double>::infinity();
        if(coopmatOk) {
          VulkanKernels::GemmStridedTiled tiledRefProbe(problemBatchSize, cmPad.m, cmPad.n, cmPad.k);
          KernelBench tiledRef = tiledRefProbe.bench(ctx, tunedConfig, iters);
          if(tiledRef.ok) {
            rmse = normalizedRmse(tiledRef.output, cmBench.output);
            maxRmse = std::max(maxRmse, rmse);
            coopmatOk = rmse <= 0.02;
          } else
            coopmatOk = false;
        }
        if(coopmatOk) {
          coopmatCaseWeightedTime = dispatchWeightedTime(cmBench, cs.occurrences);
          coopmatWeightedTime += coopmatCaseWeightedTime;
          coopmatWeightedFlops += coopmatFlopsPerDispatch * (double)cs.occurrences;
        } else
          coopmatAllOk = false;
        if(incumbentOk && coopmatOk && coopmatCaseWeightedTime > 0.0)
          maxCoopmatSpeedup = std::max(maxCoopmatSpeedup, incumbentCaseWeightedTime / coopmatCaseWeightedTime);

        if(shouldPrintTunerResult(verboseTuner, false, incumbentOk && coopmatOk)) {
          out << "VulkanTuner:   gemmStrided " << (accF16 ? "coopmatAccF16" : "coopmat") << " case#" << i;
          if(incumbentOk) {
            out << " " << incumbentName << "_kps=" << incumbentBench.kernelsPerSecond;
            if(incumbentTflops > 0.0)
              out << " " << incumbentName << "_tflops=" << incumbentTflops;
            out << " " << incumbentName << "_weighted_time=" << incumbentCaseWeightedTime;
          } else
            out << " " << incumbentName << "=failed";
          if(cmBench.ok && cmBench.kernelsPerSecond > 0.0) {
            out << " " << challengerName << "_kps=" << cmBench.kernelsPerSecond;
            if(coopmatTflops > 0.0)
              out << " " << challengerName << "_tflops=" << coopmatTflops;
            out << " " << challengerName << "_weighted_time=" << coopmatCaseWeightedTime;
          } else
            out << " " << challengerName << "=failed";
          out << " rmse=" << (std::isfinite(rmse) ? Global::doubleToString(rmse) : string("n/a")) << " "
              << challengerName << "_result=" << (coopmatOk ? "accepted" : "rejected") << endl;
        }
      }

      ModeSelectStats incumbentStats{incumbentAllOk, incumbentWeightedTime, incumbentWeightedFlops};
      ModeSelectStats coopmatStats{coopmatAllOk, coopmatWeightedTime, coopmatWeightedFlops};
      bool selected = appendModeSelectResult(
        out,
        "gemmStrided",
        incumbentName,
        challengerName,
        modeCases,
        incumbentStats,
        coopmatStats,
        maxRmse,
        true,
        true,
        [&](std::ostream& resultOut) {
          appendPositiveField(resultOut, "coopmat_avg_speedup", modeSelectSpeedup(incumbentStats, coopmatStats));
          appendPositiveField(resultOut, "coopmat_max_speedup", maxCoopmatSpeedup);
        });
      if(selected) {
        if(accF16) {
          tunedConfig.enableGemmStridedCoopmatAccF16 = 1;
          promoteGemmStridedSelectionRank(tunedConfig, &VulkanTuneParams::gemmStridedCoopmatAccF16SelectionRank);
        } else {
          tunedConfig.enableGemmStridedCoopmat = 1;
          promoteGemmStridedSelectionRank(tunedConfig, &VulkanTuneParams::gemmStridedCoopmatSelectionRank);
        }
        tunedConfig.enableGemmStridedDot2 = 0;  // mutually exclusive at runtime
        tunedConfig.enableGemmStridedDot2AccF16 = 0;
        tunedConfig.enableGemmStridedCoopmat = accF16 ? 0 : tunedConfig.enableGemmStridedCoopmat;
        tunedConfig.enableGemmStridedCoopmatAccF16 = accF16 ? tunedConfig.enableGemmStridedCoopmatAccF16 : 0;
        tunedConfig.enableGemmStridedCoopmat2 = 0;
        tunedConfig.enableGemmStridedCoopmat2AccF16 = 0;
        tunedConfig.enableGemmStridedTiledFp16Compute = 0;
      } else if(
        !incumbentCoopmat2AccF16 && !incumbentCoopmat2 && !incumbentCoopmatAccF16 && !incumbentCoopmat &&
        !incumbentDot2AccF16 && !incumbentDot2)
        promoteGemmStridedSelectionRank(tunedConfig, &VulkanTuneParams::gemmStridedTiledSelectionRank);
    }

    void tuneCoopmatModeSelect(
      TuningContext& ctx,
      const ModelDesc* modelDesc,
      int batchSize,
      int nnXLen,
      int nnYLen,
      int iters,
      std::ostream& out,
      bool verboseTuner,
      VulkanTuneParams& tunedConfig) {
      tunedConfig.enableWinogradGemmCoopmat = 0;
      tunedConfig.enableGemmStridedCoopmat = 0;
      const bool haveDevice = ctx.dev != nullptr;
      const VulkanDeviceInfo* info = haveDevice ? &ctx.dev->info : nullptr;
      const bool supportsFP16Compute = haveDevice && ctx.fp16Compute;
      const bool supportsCoopmat1F16 = haveDevice && ctx.dev->info.supportsCoopmat1F16;
      if(!haveDevice || !supportsCoopmat1F16 || !ctx.fp16Storage || !supportsFP16Compute) {
        out << endl << "VulkanTuner: skip coopmat mode-select (";
        appendMissingVulkanFp16FeatureReason(
          out,
          "coopmat1F16",
          info,
          supportsCoopmat1F16,
          info != nullptr && info->unfilteredSupportsCoopmat1F16,
          info != nullptr && info->filterDisabledCoopmat1F16,
          ctx.fp16Storage,
          supportsFP16Compute);
        out << ")." << endl;
        return;
      }
      std::vector<CoopmatTile> winogradCandidates = makeCoopmatCandidates(ctx, ctx.dev->info.coopmatShapes);
      std::vector<CoopmatTile> stridedCandidates = makeCoopmatCandidates(ctx, ctx.dev->info.coopmatShapes);
      if(winogradCandidates.empty() && stridedCandidates.empty()) {
        out << endl << "VulkanTuner: skip coopmat mode-select (no usable coopmat shapes)." << endl;
        return;
      }
      out << endl << "VulkanTuner: mode-select coopmat vs current best for winograd/strided GEMM..." << endl;
      out << "VulkanTuner: coopmat candidate counts winograd=" << winogradCandidates.size()
          << " gemmStrided=" << stridedCandidates.size() << " device_shapes=" << ctx.dev->info.coopmatShapes.size()
          << endl;
      const int problemBatchSize = std::max(1, batchSize);
      if(tunedConfig.gemmStridedCoopmatTunerValueValid != 0)
        tuneGemmStridedCoopmatModeSelect(
          ctx, modelDesc, problemBatchSize, nnXLen, nnYLen, iters, false, out, verboseTuner, tunedConfig);
      if(tunedConfig.winogradGemmCoopmatTunerValueValid != 0)
        tuneWinogradCoopmatModeSelect(
          ctx, modelDesc, problemBatchSize, nnXLen, nnYLen, iters, false, out, verboseTuner, tunedConfig);
    }

    void tuneCoopmatAccF16ModeSelect(
      TuningContext& ctx,
      const ModelDesc* modelDesc,
      int batchSize,
      int nnXLen,
      int nnYLen,
      int iters,
      std::ostream& out,
      bool verboseTuner,
      VulkanTuneParams& tunedConfig) {
      tunedConfig.enableWinogradGemmCoopmatAccF16 = 0;
      tunedConfig.enableGemmStridedCoopmatAccF16 = 0;
      const bool haveDevice = ctx.dev != nullptr;
      const VulkanDeviceInfo* info = haveDevice ? &ctx.dev->info : nullptr;
      const bool supportsFP16Compute = haveDevice && ctx.fp16Compute;
      const bool supportsCoopmat1F16AccF16 = haveDevice && ctx.dev->info.supportsCoopmat1F16AccF16;
      if(!haveDevice || !supportsCoopmat1F16AccF16 || !ctx.fp16Storage || !supportsFP16Compute) {
        out << endl << "VulkanTuner: skip coopmatAccF16 mode-select (";
        appendMissingVulkanFp16FeatureReason(
          out,
          "coopmat1F16AccF16",
          info,
          supportsCoopmat1F16AccF16,
          info != nullptr && info->unfilteredSupportsCoopmat1F16AccF16,
          info != nullptr && info->filterDisabledCoopmat1F16AccF16,
          ctx.fp16Storage,
          supportsFP16Compute);
        out << ")." << endl;
        return;
      }
      std::vector<CoopmatTile> winogradCandidates = makeCoopmatCandidates(ctx, ctx.dev->info.coopmatAccF16Shapes);
      std::vector<CoopmatTile> stridedCandidates = makeCoopmatCandidates(ctx, ctx.dev->info.coopmatAccF16Shapes);
      if(winogradCandidates.empty() && stridedCandidates.empty()) {
        out << endl << "VulkanTuner: skip coopmatAccF16 mode-select (no usable coopmatAccF16 shapes)." << endl;
        return;
      }
      out << endl << "VulkanTuner: mode-select coopmatAccF16 vs current best for winograd/strided GEMM..." << endl;
      out << "VulkanTuner: coopmatAccF16 candidate counts winograd=" << winogradCandidates.size()
          << " gemmStrided=" << stridedCandidates.size()
          << " device_shapes=" << ctx.dev->info.coopmatAccF16Shapes.size() << endl;
      const int problemBatchSize = std::max(1, batchSize);
      if(tunedConfig.gemmStridedCoopmatAccF16TunerValueValid != 0)
        tuneGemmStridedCoopmatModeSelect(
          ctx, modelDesc, problemBatchSize, nnXLen, nnYLen, iters, true, out, verboseTuner, tunedConfig);
      if(tunedConfig.winogradGemmCoopmatAccF16TunerValueValid != 0)
        tuneWinogradCoopmatModeSelect(
          ctx, modelDesc, problemBatchSize, nnXLen, nnYLen, iters, true, out, verboseTuner, tunedConfig);
    }

    // ---- Coopmat2 (VK_NV_cooperative_matrix2) mode-select ----

    struct Coopmat2Tile {
      int32_t blockSize;
      int32_t bm;
      int32_t bn;
      int32_t bk;
    };

    std::string coopmat2TileToString(const Coopmat2Tile& t) {
      return Global::strprintf("bs=%d BM=%d BN=%d BK=%d", t.blockSize, t.bm, t.bn, t.bk);
    }

    bool coopmat2TilesEqual(const Coopmat2Tile& a, const Coopmat2Tile& b) {
      return a.blockSize == b.blockSize && a.bm == b.bm && a.bn == b.bn && a.bk == b.bk;
    }

    bool coopmat2TileMatchesShape(const Coopmat2Tile& t, const Coopmat2FlexShape& s) {
      return t.blockSize == (int32_t)s.workgroupInvocations && t.bm % (int32_t)s.mGranularity == 0 &&
             t.bn % (int32_t)s.nGranularity == 0 && t.bk % (int32_t)s.kGranularity == 0;
    }

    bool coopmat2TileSupported(
      const VulkanDeviceInfo& info,
      const std::vector<Coopmat2FlexShape>& shapes,
      const Coopmat2Tile& t) {
      if(t.bm <= 0 || t.bn <= 0 || t.bk <= 0 || t.blockSize <= 0)
        return false;
      if(info.coopmat2ReservedSharedBytes > info.properties.limits.maxComputeSharedMemorySize)
        return false;
      if(info.coopmat2MaxWorkgroupSize > 0 && (uint32_t)t.blockSize > info.coopmat2MaxWorkgroupSize)
        return false;
      if(info.coopmat2MaxFlexDimension > 0) {
        uint32_t maxDim = std::max((uint32_t)t.bm, std::max((uint32_t)t.bn, (uint32_t)t.bk));
        if(maxDim > info.coopmat2MaxFlexDimension)
          return false;
      }
      for(const auto& s: shapes)
        if(coopmat2TileMatchesShape(t, s))
          return true;
      return false;
    }

    // Build coopmat2 tile candidates from the device's reported flexible-dimension
    // granularities. BM/BN/BK are generated as multiples of each shape's granularity so
    // every candidate is device-appropriate by construction, and blockSize is the shape's
    // workgroupInvocations. This mirrors makeCoopmatCandidates (coopmat1) instead of
    // hardcoding absolute tile sizes, which do not transfer across GPUs. BK is capped at
    // 32 (BK>=64 never wins, matching coopmat1's maxBk). M/N are capped at 256:
    // workgroup-scope fp16->fp32 microbenchmarks on Turing strongly favor 128x128,
    // with 128x256/256x128 sometimes close enough to be worth letting the tuner see.
    // isConfigSupported / coopmat2TileSupported / maxDim prune the
    // device-inappropriate cases.
    std::vector<Coopmat2Tile> makeCoopmat2Candidates(
      const TuningContext& ctx,
      const std::vector<Coopmat2FlexShape>& shapes) {
      std::vector<Coopmat2Tile> out;
      if(ctx.dev == nullptr)
        return out;
      const VulkanDeviceInfo& info = ctx.dev->info;
      auto add = [&](Coopmat2Tile t) {
        if(!VulkanKernels::WinogradGemmCoopmat2::isConfigSupported(t.blockSize, t.bm, t.bn, t.bk))
          return;
        if(!coopmat2TileSupported(info, shapes, t))
          return;
        for(const Coopmat2Tile& x: out)
          if(x.blockSize == t.blockSize && x.bm == t.bm && x.bn == t.bn && x.bk == t.bk)
            return;
        out.push_back(t);
      };
      // Non-power-of-two CTA multiples so tiles can fit common KataGo dimensions cleanly;
      // BK stays a small multiple, capped at maxBk. trimCoopmat2Candidates ranks these
      // per-problem and keeps only the top-K, so the raw set can be generous.
      static const int bmMul[] = {1, 2, 3, 4, 6, 8};
      static const int bnMul[] = {1, 2, 3, 4, 6, 8};
      static const int bkMul[] = {1, 2, 3, 4};
      constexpr int maxBk = 32;
      for(const Coopmat2FlexShape& s: shapes) {
        int maxDim =
          info.coopmat2MaxFlexDimension > 0 ? (int)std::min<uint32_t>(info.coopmat2MaxFlexDimension, 256) : 256;
        for(int bmm: bmMul) {
          int bm = (int)s.mGranularity * bmm;
          if(bm > maxDim)
            continue;
          for(int bnm: bnMul) {
            int bn = (int)s.nGranularity * bnm;
            if(bn > maxDim)
              continue;
            for(int bkm: bkMul) {
              int bk = (int)s.kGranularity * bkm;
              if(bk > maxDim || bk > maxBk)
                continue;
              add(Coopmat2Tile{(int32_t)s.workgroupInvocations, (int32_t)bm, (int32_t)bn, (int32_t)bk});
            }
          }
        }
      }
      return out;
    }

    // Trim coopmat2 candidates to the most promising top-K for a concrete (M,N,K),
    // mirroring trimCoopmatCandidates (coopmat1) but over the leaner Coopmat2Tile (no
    // wm/wn warp filters, since coopmat2 has none). Shares tileTrimScore.
    std::vector<Coopmat2Tile> trimCoopmat2Candidates(
      const std::vector<Coopmat2Tile>& candidates,
      int m,
      int n,
      int k,
      bool padM,
      bool padN,
      bool padK,
      size_t maxCandidates,
      int minBlockSize,
      int maxBlockSize,
      int minBm,
      int minBn,
      int maxBk,
      double maxPaddingRatio,
      double targetTileArea,
      double targetBk,
      double targetBlockSize) {
      struct RankedCoopmat2Tile {
        Coopmat2Tile tile;
        double score;
      };
      std::vector<RankedCoopmat2Tile> ranked;
      ranked.reserve(candidates.size());
      for(const Coopmat2Tile& t: candidates) {
        if(t.blockSize < minBlockSize || t.blockSize > maxBlockSize || t.bm < minBm || t.bn < minBn || t.bk > maxBk)
          continue;
        const int paddedM = padM ? roundUpCoopmatDim(m, t.bm) : m;
        const int paddedN = padN ? roundUpCoopmatDim(n, t.bn) : n;
        const int paddedK = padK ? roundUpCoopmatDim(k, t.bk) : k;
        const double baseWork = std::max(1.0, (double)m * (double)n * (double)k);
        const double paddingRatio = ((double)paddedM * (double)paddedN * (double)paddedK) / baseWork;
        if(paddingRatio > maxPaddingRatio)
          continue;
        ranked.push_back(
          {t,
           tileTrimScore(
             t.bm, t.bn, t.bk, t.blockSize, m, n, k, padM, padN, padK, targetTileArea, targetBk, targetBlockSize)});
      }

      if(ranked.empty()) {
        for(const Coopmat2Tile& t: candidates)
          ranked.push_back(
            {t,
             tileTrimScore(
               t.bm, t.bn, t.bk, t.blockSize, m, n, k, padM, padN, padK, targetTileArea, targetBk, targetBlockSize)});
      }

      std::stable_sort(ranked.begin(), ranked.end(), [](const RankedCoopmat2Tile& a, const RankedCoopmat2Tile& b) {
        return a.score < b.score;
      });
      if(ranked.size() > maxCandidates)
        ranked.resize(maxCandidates);

      std::vector<Coopmat2Tile> trimmed;
      trimmed.reserve(ranked.size());
      for(const RankedCoopmat2Tile& r: ranked)
        trimmed.push_back(r.tile);
      return trimmed;
    }

    void applyWinogradCoopmat2Tile(VulkanTuneParams& cfg, const Coopmat2Tile& t) {
      cfg.coopmat2BlockSize = t.blockSize;
      cfg.coopmat2BM = t.bm;
      cfg.coopmat2BN = t.bn;
      cfg.coopmat2BK = t.bk;
    }

    void applyStridedCoopmat2Tile(VulkanTuneParams& cfg, const Coopmat2Tile& t) {
      cfg.stridedCoopmat2BlockSize = t.blockSize;
      cfg.stridedCoopmat2BM = t.bm;
      cfg.stridedCoopmat2BN = t.bn;
      cfg.stridedCoopmat2BK = t.bk;
    }

    void applyWinogradCoopmat2AccF16Tile(VulkanTuneParams& cfg, const Coopmat2Tile& t) {
      cfg.coopmat2AccF16BlockSize = t.blockSize;
      cfg.coopmat2AccF16BM = t.bm;
      cfg.coopmat2AccF16BN = t.bn;
      cfg.coopmat2AccF16BK = t.bk;
    }

    void applyStridedCoopmat2AccF16Tile(VulkanTuneParams& cfg, const Coopmat2Tile& t) {
      cfg.stridedCoopmat2AccF16BlockSize = t.blockSize;
      cfg.stridedCoopmat2AccF16BM = t.bm;
      cfg.stridedCoopmat2AccF16BN = t.bn;
      cfg.stridedCoopmat2AccF16BK = t.bk;
    }

    void tuneWinogradCoopmat2TileSweep(
      TuningContext& ctx,
      const ModelDesc* modelDesc,
      int problemBatchSize,
      int nnXLen,
      int nnYLen,
      int iters,
      const std::vector<Coopmat2Tile>& candidates,
      bool accF16,
      std::ostream& out,
      bool verboseTuner,
      VulkanTuneParams& tunedConfig) {
      std::vector<WinogradModeCase> cases =
        collectWinogradModeCases(modelDesc, problemBatchSize, nnXLen, nnYLen, tunedConfig);
      if(cases.empty() || candidates.empty())
        return;
      // Trim to the most promising tiles for this problem, like coopmat1's winograd
      // sweep. Filters are relaxed vs coopmat1 (minBm/minBlockSize small), while
      // targets are centered on the fp16->fp32 workgroup-scope sweet spot observed
      // on Turing-class coopmat2: 128x128 tiles, BK=16, 128-ish invocations.
      const WinogradModeCase& tc0 = cases[0];
      std::vector<Coopmat2Tile> sweepCandidates = trimCoopmat2Candidates(
        candidates, tc0.M, tc0.N, tc0.K, true, true, true, 64, 1, 100000, 8, 8, 32, 1.60, 16384.0, 16.0, 128.0);
      if(sweepCandidates.empty())
        return;

      struct CachedWinogradRef {
        int m, n, k, numBatches;
        KernelBench bench;
      };
      std::vector<CachedWinogradRef> tiledRefs;
      auto getTiledRef = [&](int m, int n, int k, int numBatches) -> const KernelBench* {
        for(const CachedWinogradRef& ref: tiledRefs)
          if(ref.m == m && ref.n == n && ref.k == k && ref.numBatches == numBatches)
            return &ref.bench;
        VulkanKernels::WinogradGemm ref(m, n, k, numBatches);
        tiledRefs.push_back({m, n, k, numBatches, ref.bench(ctx, tunedConfig, iters)});
        return &tiledRefs.back().bench;
      };

      auto benchAggregateTile = [&](const Coopmat2Tile& t, AggregateCandidateMetrics& metrics) {
        VulkanTuneParams trial = tunedConfig;
        if(accF16)
          applyWinogradCoopmat2AccF16Tile(trial, t);
        else
          applyWinogradCoopmat2Tile(trial, t);
        metrics.weightedTime = 0.0;
        metrics.tflops = 0.0;
        metrics.maxRmse = 0.0;
        double aggregateFlops = 0.0;
        double aggregateSeconds = 0.0;
        for(const WinogradModeCase& cs: cases) {
          const auto pad =
            accF16 ? padDims(cs.M, cs.N, cs.K, VulkanKernels::WinogradGemmCoopmat2AccF16::layerPaddingContract(trial))
                   : padDims(cs.M, cs.N, cs.K, VulkanKernels::WinogradGemmCoopmat2::layerPaddingContract(trial));
          KernelBench bench;
          double flopsPerDispatch = 0.0;
          if(accF16) {
            VulkanKernels::WinogradGemmCoopmat2AccF16 probe(pad.m, pad.n, pad.k, cs.numBatches);
            if(!probe.validate(trial, ctx.dev->info.properties.limits))
              return rejectAggregateCandidate(metrics, "validate_failed");
            bench = probe.bench(ctx, trial, iters);
            flopsPerDispatch = probe.estimatedFlopsPerDispatch(trial);
          } else {
            VulkanKernels::WinogradGemmCoopmat2 probe(pad.m, pad.n, pad.k, cs.numBatches);
            if(!probe.validate(trial, ctx.dev->info.properties.limits))
              return rejectAggregateCandidate(metrics, "validate_failed");
            bench = probe.bench(ctx, trial, iters);
            flopsPerDispatch = probe.estimatedFlopsPerDispatch(trial);
          }
          if(!(bench.ok && bench.kernelsPerSecond > 0.0))
            return rejectAggregateCandidate(metrics, "bench_failed");
          const KernelBench* refBench = getTiledRef(pad.m, pad.n, pad.k, cs.numBatches);
          if(!refBench->ok)
            return rejectAggregateCandidate(metrics, "reference_failed");
          double rmse = normalizedRmse(refBench->output, bench.output);
          metrics.maxRmse = std::max(metrics.maxRmse, rmse);
          if(rmse > 0.02)
            return rejectAggregateCandidate(metrics, "rmse_exceeded");
          const double caseWeightedTime = dispatchWeightedTime(bench, cs.occurrences);
          metrics.weightedTime += caseWeightedTime;
          aggregateFlops += flopsPerDispatch * (double)cs.occurrences;
          aggregateSeconds += caseWeightedTime;
        }
        metrics.tflops = aggregateTflops(aggregateFlops, aggregateSeconds);
        return metrics.weightedTime > 0.0;
      };

      string_view label = accF16 ? "winogradCoopmat2AccF16" : "winogradCoopmat2";
      out << "VulkanTuner: " << label << " tile sweep candidates=" << sweepCandidates.size()
          << " cases=" << cases.size() << endl;
      AggregateCandidateSweepResult<Coopmat2Tile> result = tuneAggregateCandidateSweep(
        TunerStreamText{label},
        TunerStreamText{label},
        TunerStreamText{label, " tile sweep"},
        "tile",
        TunerStreamText{"VulkanTuner: ", label, " tile sweep found no valid tile; keeping defaults."},
        sweepCandidates,
        TUNER_CONFIRM_PREFIX_K,
        true,
        true,
        false,
        out,
        verboseTuner,
        benchAggregateTile,
        coopmat2TileToString,
        coopmat2TilesEqual);
      if(result.found) {
        if(accF16) {
          applyWinogradCoopmat2AccF16Tile(tunedConfig, result.candidate);
          tunedConfig.winogradGemmCoopmat2AccF16TunerValueValid = 1;
        } else {
          applyWinogradCoopmat2Tile(tunedConfig, result.candidate);
          tunedConfig.winogradGemmCoopmat2TunerValueValid = 1;
        }
      }
    }

    void tuneGemmStridedCoopmat2TileSweep(
      TuningContext& ctx,
      const ModelDesc* modelDesc,
      int problemBatchSize,
      int nnXLen,
      int nnYLen,
      int iters,
      const std::vector<Coopmat2Tile>& candidates,
      bool accF16,
      std::ostream& out,
      bool verboseTuner,
      VulkanTuneParams& tunedConfig) {
      std::vector<GemmStridedModeCase> cases = collectGemmStridedModeCases(modelDesc, nnXLen, nnYLen);
      if(cases.empty() || candidates.empty())
        return;
      // Trim to the most promising tiles for this problem, like coopmat1's strided sweep:
      // pad only N (strided contract keeps M/K real), relaxed small-tile filters, and
      // targets centered on the same fp16->fp32 workgroup-scope sweet spot as
      // winogradCoopmat2.
      const GemmStridedModeCase& tc0 = cases[0];
      std::vector<Coopmat2Tile> sweepCandidates = trimCoopmat2Candidates(
        candidates, tc0.M, tc0.N, tc0.K, false, true, false, 64, 1, 100000, 8, 8, 32, 1.35, 16384.0, 16.0, 128.0);
      if(sweepCandidates.empty())
        return;

      struct CachedStridedRef {
        int m, n, k;
        KernelBench bench;
      };
      std::vector<CachedStridedRef> tiledRefs;
      auto getTiledRef = [&](int m, int n, int k) -> const KernelBench* {
        for(const CachedStridedRef& ref: tiledRefs)
          if(ref.m == m && ref.n == n && ref.k == k)
            return &ref.bench;
        VulkanKernels::GemmStridedTiled ref(problemBatchSize, m, n, k);
        tiledRefs.push_back({m, n, k, ref.bench(ctx, tunedConfig, iters)});
        return &tiledRefs.back().bench;
      };

      auto benchAggregateTile = [&](const Coopmat2Tile& t, AggregateCandidateMetrics& metrics) {
        VulkanTuneParams trial = tunedConfig;
        if(accF16)
          applyStridedCoopmat2AccF16Tile(trial, t);
        else
          applyStridedCoopmat2Tile(trial, t);
        metrics.weightedTime = 0.0;
        metrics.tflops = 0.0;
        metrics.maxRmse = 0.0;
        double aggregateFlops = 0.0;
        double aggregateSeconds = 0.0;
        for(const GemmStridedModeCase& cs: cases) {
          const auto pad =
            accF16 ? padDims(cs.M, cs.N, cs.K, VulkanKernels::GemmStridedCoopmat2AccF16::layerPaddingContract(trial))
                   : padDims(cs.M, cs.N, cs.K, VulkanKernels::GemmStridedCoopmat2::layerPaddingContract(trial));
          KernelBench bench;
          double flopsPerDispatch = 0.0;
          if(accF16) {
            VulkanKernels::GemmStridedCoopmat2AccF16 probe(problemBatchSize, pad.m, pad.n, pad.k);
            if(!probe.validate(trial, ctx.dev->info.properties.limits))
              return rejectAggregateCandidate(metrics, "validate_failed");
            bench = probe.bench(ctx, trial, iters);
            flopsPerDispatch = probe.estimatedFlopsPerDispatch(trial);
          } else {
            VulkanKernels::GemmStridedCoopmat2 probe(problemBatchSize, pad.m, pad.n, pad.k);
            if(!probe.validate(trial, ctx.dev->info.properties.limits))
              return rejectAggregateCandidate(metrics, "validate_failed");
            bench = probe.bench(ctx, trial, iters);
            flopsPerDispatch = probe.estimatedFlopsPerDispatch(trial);
          }
          if(!(bench.ok && bench.kernelsPerSecond > 0.0))
            return rejectAggregateCandidate(metrics, "bench_failed");
          const KernelBench* refBench = getTiledRef(pad.m, pad.n, pad.k);
          if(!refBench->ok)
            return rejectAggregateCandidate(metrics, "reference_failed");
          double rmse = normalizedRmse(refBench->output, bench.output);
          metrics.maxRmse = std::max(metrics.maxRmse, rmse);
          if(rmse > 0.02)
            return rejectAggregateCandidate(metrics, "rmse_exceeded");
          const double caseWeightedTime = dispatchWeightedTime(bench, cs.occurrences);
          metrics.weightedTime += caseWeightedTime;
          aggregateFlops += flopsPerDispatch * (double)cs.occurrences;
          aggregateSeconds += caseWeightedTime;
        }
        metrics.tflops = aggregateTflops(aggregateFlops, aggregateSeconds);
        return metrics.weightedTime > 0.0;
      };

      string_view label = accF16 ? "gemmStridedCoopmat2AccF16" : "gemmStridedCoopmat2";
      out << "VulkanTuner: " << label << " tile sweep candidates=" << sweepCandidates.size()
          << " cases=" << cases.size() << endl;
      AggregateCandidateSweepResult<Coopmat2Tile> result = tuneAggregateCandidateSweep(
        TunerStreamText{label},
        TunerStreamText{label},
        TunerStreamText{label, " tile sweep"},
        "tile",
        TunerStreamText{"VulkanTuner: ", label, " tile sweep found no valid tile; keeping defaults."},
        sweepCandidates,
        TUNER_CONFIRM_PREFIX_K,
        true,
        true,
        false,
        out,
        verboseTuner,
        benchAggregateTile,
        coopmat2TileToString,
        coopmat2TilesEqual);
      if(result.found) {
        if(accF16) {
          applyStridedCoopmat2AccF16Tile(tunedConfig, result.candidate);
          tunedConfig.gemmStridedCoopmat2AccF16TunerValueValid = 1;
        } else {
          applyStridedCoopmat2Tile(tunedConfig, result.candidate);
          tunedConfig.gemmStridedCoopmat2TunerValueValid = 1;
        }
      }
    }

    void tuneWinogradCoopmat2ModeSelect(
      TuningContext& ctx,
      const ModelDesc* modelDesc,
      int problemBatchSize,
      int nnXLen,
      int nnYLen,
      int iters,
      bool accF16,
      std::ostream& out,
      bool verboseTuner,
      VulkanTuneParams& tunedConfig) {
      std::vector<WinogradModeCase> cases =
        collectWinogradModeCases(modelDesc, problemBatchSize, nnXLen, nnYLen, tunedConfig);
      if(cases.empty())
        return;
      const bool incumbentCoopmat2AccF16 = !accF16 && tunedConfig.enableWinogradGemmCoopmat2AccF16 != 0;
      const bool incumbentCoopmat2 = !incumbentCoopmat2AccF16 && accF16 && tunedConfig.enableWinogradGemmCoopmat2 != 0;
      const bool incumbentCoopmatAccF16 =
        !incumbentCoopmat2AccF16 && !incumbentCoopmat2 && tunedConfig.enableWinogradGemmCoopmatAccF16 != 0;
      const bool incumbentCoopmat = !incumbentCoopmat2AccF16 && !incumbentCoopmat2 && !incumbentCoopmatAccF16 &&
                                    tunedConfig.enableWinogradGemmCoopmat != 0;
      const bool incumbentDot2AccF16 = !incumbentCoopmat2AccF16 && !incumbentCoopmat2 && !incumbentCoopmatAccF16 &&
                                       !incumbentCoopmat && tunedConfig.enableWinogradGemmDot2AccF16 != 0;
      const bool incumbentDot2 = !incumbentCoopmat2AccF16 && !incumbentCoopmat2 && !incumbentCoopmatAccF16 &&
                                 !incumbentCoopmat && !incumbentDot2AccF16 && tunedConfig.enableWinogradGemmDot2 != 0;
      string_view incumbentName = incumbentCoopmat2AccF16  ? "winogradGemmCoopmat2AccF16"
                                  : incumbentCoopmat2      ? "winogradGemmCoopmat2"
                                  : incumbentCoopmatAccF16 ? "winogradGemmCoopmatAccF16"
                                  : incumbentCoopmat       ? "winogradGemmCoopmat"
                                  : incumbentDot2AccF16    ? "winogradGemmDot2AccF16"
                                  : incumbentDot2          ? "winogradGemmDot2"
                                                           : "winogradGemmTiled";
      bool incumbentAllOk = true;
      bool cm2AllOk = true;
      double incumbentWeightedTime = 0.0;
      double cm2WeightedTime = 0.0;
      double incumbentWeightedFlops = 0.0;
      double cm2WeightedFlops = 0.0;
      double maxRmse = 0.0;
      string_view challengerName = accF16 ? "winogradGemmCoopmat2AccF16" : "winogradGemmCoopmat2";
      out << endl
          << "VulkanTuner: winogradGemm " << (accF16 ? "coopmat2AccF16" : "coopmat2") << " mode-select cases ("
          << cases.size() << ") ..." << endl;
      for(size_t i = 0; i < cases.size(); i++) {
        const WinogradModeCase& cs = cases[i];
        KernelBench incumbentBench;
        double incumbentTflops = 0.0;
        double incumbentFlopsPerDispatch = 0.0;
        if(incumbentCoopmat2AccF16) {
          const auto pad =
            padDims(cs.M, cs.N, cs.K, VulkanKernels::WinogradGemmCoopmat2AccF16::layerPaddingContract(tunedConfig));
          VulkanKernels::WinogradGemmCoopmat2AccF16 probe(pad.m, pad.n, pad.k, cs.numBatches);
          incumbentBench = probe.bench(ctx, tunedConfig, iters);
          incumbentTflops = probe.estimatedTflops(tunedConfig, incumbentBench.kernelsPerSecond);
          incumbentFlopsPerDispatch = probe.estimatedFlopsPerDispatch(tunedConfig);
        } else if(incumbentCoopmat2) {
          const auto pad =
            padDims(cs.M, cs.N, cs.K, VulkanKernels::WinogradGemmCoopmat2::layerPaddingContract(tunedConfig));
          VulkanKernels::WinogradGemmCoopmat2 probe(pad.m, pad.n, pad.k, cs.numBatches);
          incumbentBench = probe.bench(ctx, tunedConfig, iters);
          incumbentTflops = probe.estimatedTflops(tunedConfig, incumbentBench.kernelsPerSecond);
          incumbentFlopsPerDispatch = probe.estimatedFlopsPerDispatch(tunedConfig);
        } else if(incumbentCoopmatAccF16) {
          const auto pad =
            padDims(cs.M, cs.N, cs.K, VulkanKernels::WinogradGemmCoopmatAccF16::layerPaddingContract(tunedConfig));
          VulkanKernels::WinogradGemmCoopmatAccF16 probe(pad.m, pad.n, pad.k, cs.numBatches);
          incumbentBench = probe.bench(ctx, tunedConfig, iters);
          incumbentTflops = probe.estimatedTflops(tunedConfig, incumbentBench.kernelsPerSecond);
          incumbentFlopsPerDispatch = probe.estimatedFlopsPerDispatch(tunedConfig);
        } else if(incumbentCoopmat) {
          const auto pad =
            padDims(cs.M, cs.N, cs.K, VulkanKernels::WinogradGemmCoopmat::layerPaddingContract(tunedConfig));
          VulkanKernels::WinogradGemmCoopmat probe(pad.m, pad.n, pad.k, cs.numBatches);
          incumbentBench = probe.bench(ctx, tunedConfig, iters);
          incumbentTflops = probe.estimatedTflops(tunedConfig, incumbentBench.kernelsPerSecond);
          incumbentFlopsPerDispatch = probe.estimatedFlopsPerDispatch(tunedConfig);
        } else if(incumbentDot2AccF16) {
          const auto pad =
            padDims(cs.M, cs.N, cs.K, VulkanKernels::WinogradGemmDot2AccF16::layerPaddingContract(tunedConfig));
          VulkanKernels::WinogradGemmDot2AccF16 probe(pad.m, pad.n, pad.k, cs.numBatches);
          incumbentBench = probe.bench(ctx, tunedConfig, iters);
          incumbentTflops = probe.estimatedTflops(tunedConfig, incumbentBench.kernelsPerSecond);
          incumbentFlopsPerDispatch = probe.estimatedFlopsPerDispatch(tunedConfig);
        } else if(incumbentDot2) {
          const auto pad =
            padDims(cs.M, cs.N, cs.K, VulkanKernels::WinogradGemmDot2::layerPaddingContract(tunedConfig));
          VulkanKernels::WinogradGemmDot2 probe(pad.m, pad.n, pad.k, cs.numBatches);
          incumbentBench = probe.bench(ctx, tunedConfig, iters);
          incumbentTflops = probe.estimatedTflops(tunedConfig, incumbentBench.kernelsPerSecond);
          incumbentFlopsPerDispatch = probe.estimatedFlopsPerDispatch(tunedConfig);
        } else {
          const auto pad = padDims(cs.M, cs.N, cs.K, VulkanKernels::WinogradGemm::layerPaddingContract(tunedConfig));
          VulkanKernels::WinogradGemm probe(pad.m, pad.n, pad.k, cs.numBatches);
          incumbentBench = probe.bench(ctx, tunedConfig, iters);
          incumbentTflops = probe.estimatedTflops(tunedConfig, incumbentBench.kernelsPerSecond);
          incumbentFlopsPerDispatch = probe.estimatedFlopsPerDispatch(tunedConfig);
        }
        bool incumbentOk = incumbentBench.ok && incumbentBench.kernelsPerSecond > 0.0;
        double incumbentCaseWeightedTime = 0.0;
        if(incumbentOk) {
          incumbentCaseWeightedTime = dispatchWeightedTime(incumbentBench, cs.occurrences);
          incumbentWeightedTime += incumbentCaseWeightedTime;
          incumbentWeightedFlops += incumbentFlopsPerDispatch * (double)cs.occurrences;
        } else
          incumbentAllOk = false;

        const auto cm2Pad =
          accF16
            ? padDims(cs.M, cs.N, cs.K, VulkanKernels::WinogradGemmCoopmat2AccF16::layerPaddingContract(tunedConfig))
            : padDims(cs.M, cs.N, cs.K, VulkanKernels::WinogradGemmCoopmat2::layerPaddingContract(tunedConfig));
        KernelBench cm2Bench;
        double cm2Tflops = 0.0;
        double cm2FlopsPerDispatch = 0.0;
        if(accF16) {
          VulkanKernels::WinogradGemmCoopmat2AccF16 cm2Probe(cm2Pad.m, cm2Pad.n, cm2Pad.k, cs.numBatches);
          cm2Bench = cm2Probe.bench(ctx, tunedConfig, iters);
          cm2Tflops = cm2Probe.estimatedTflops(tunedConfig, cm2Bench.kernelsPerSecond);
          cm2FlopsPerDispatch = cm2Probe.estimatedFlopsPerDispatch(tunedConfig);
        } else {
          VulkanKernels::WinogradGemmCoopmat2 cm2Probe(cm2Pad.m, cm2Pad.n, cm2Pad.k, cs.numBatches);
          cm2Bench = cm2Probe.bench(ctx, tunedConfig, iters);
          cm2Tflops = cm2Probe.estimatedTflops(tunedConfig, cm2Bench.kernelsPerSecond);
          cm2FlopsPerDispatch = cm2Probe.estimatedFlopsPerDispatch(tunedConfig);
        }
        bool cm2Ok = cm2Bench.ok && cm2Bench.kernelsPerSecond > 0.0;
        double cm2CaseWeightedTime = 0.0;
        double rmse = std::numeric_limits<double>::infinity();
        if(cm2Ok) {
          VulkanKernels::WinogradGemm ref(cm2Pad.m, cm2Pad.n, cm2Pad.k, cs.numBatches);
          KernelBench refBench = ref.bench(ctx, tunedConfig, iters);
          cm2Ok = refBench.ok;
          if(cm2Ok) {
            rmse = normalizedRmse(refBench.output, cm2Bench.output);
            maxRmse = std::max(maxRmse, rmse);
            cm2Ok = rmse <= 0.02;
          }
        }
        if(cm2Ok) {
          cm2CaseWeightedTime = dispatchWeightedTime(cm2Bench, cs.occurrences);
          cm2WeightedTime += cm2CaseWeightedTime;
          cm2WeightedFlops += cm2FlopsPerDispatch * (double)cs.occurrences;
        } else
          cm2AllOk = false;
        if(shouldPrintTunerResult(verboseTuner, false, incumbentOk && cm2Ok)) {
          out << "VulkanTuner:   winogradGemm " << (accF16 ? "coopmat2AccF16" : "coopmat2") << " case#" << i;
          if(incumbentOk) {
            out << " " << incumbentName << "_kps=" << incumbentBench.kernelsPerSecond;
            if(incumbentTflops > 0.0)
              out << " " << incumbentName << "_tflops=" << incumbentTflops;
            out << " " << incumbentName << "_weighted_time=" << incumbentCaseWeightedTime;
          } else
            out << " " << incumbentName << "=failed";
          if(cm2Bench.ok && cm2Bench.kernelsPerSecond > 0.0) {
            out << " " << challengerName << "_kps=" << cm2Bench.kernelsPerSecond;
            if(cm2Tflops > 0.0)
              out << " " << challengerName << "_tflops=" << cm2Tflops;
            out << " " << challengerName << "_weighted_time=" << cm2CaseWeightedTime;
          } else
            out << " " << challengerName << "=failed";
          out << " rmse=" << (std::isfinite(rmse) ? Global::doubleToString(rmse) : string("n/a")) << " "
              << challengerName << "_result=" << (cm2Ok ? "accepted" : "rejected") << endl;
        }
      }
      ModeSelectStats incumbentStats{incumbentAllOk, incumbentWeightedTime, incumbentWeightedFlops};
      ModeSelectStats cm2Stats{cm2AllOk, cm2WeightedTime, cm2WeightedFlops};
      bool selected = appendModeSelectResult(
        out,
        "winogradGemm",
        incumbentName,
        challengerName,
        cases,
        incumbentStats,
        cm2Stats,
        maxRmse,
        false,
        false,
        [](std::ostream&) {});
      if(selected) {
        if(accF16) {
          tunedConfig.enableWinogradGemmCoopmat2AccF16 = 1;
          promoteWinogradGemmSelectionRank(tunedConfig, &VulkanTuneParams::winogradGemmCoopmat2AccF16SelectionRank);
        } else {
          tunedConfig.enableWinogradGemmCoopmat2 = 1;
          promoteWinogradGemmSelectionRank(tunedConfig, &VulkanTuneParams::winogradGemmCoopmat2SelectionRank);
        }
        tunedConfig.enableWinogradGemmCoopmat2 = accF16 ? 0 : tunedConfig.enableWinogradGemmCoopmat2;
        tunedConfig.enableWinogradGemmCoopmat2AccF16 = accF16 ? tunedConfig.enableWinogradGemmCoopmat2AccF16 : 0;
        tunedConfig.enableWinogradGemmCoopmat = 0;
        tunedConfig.enableWinogradGemmCoopmatAccF16 = 0;
        tunedConfig.enableWinogradGemmDot2 = 0;
        tunedConfig.enableWinogradGemmDot2AccF16 = 0;
        tunedConfig.enableWinogradGemmTiledFp16Compute = 0;
      } else if(
        !incumbentCoopmat2AccF16 && !incumbentCoopmat2 && !incumbentCoopmatAccF16 && !incumbentCoopmat &&
        !incumbentDot2AccF16 && !incumbentDot2)
        promoteWinogradGemmSelectionRank(tunedConfig, &VulkanTuneParams::winogradGemmTiledSelectionRank);
    }

    void tuneGemmStridedCoopmat2ModeSelect(
      TuningContext& ctx,
      const ModelDesc* modelDesc,
      int problemBatchSize,
      int nnXLen,
      int nnYLen,
      int iters,
      bool accF16,
      std::ostream& out,
      bool verboseTuner,
      VulkanTuneParams& tunedConfig) {
      std::vector<GemmStridedModeCase> cases = collectGemmStridedModeCases(modelDesc, nnXLen, nnYLen);
      if(cases.empty())
        return;
      const bool incumbentCoopmat2AccF16 = !accF16 && tunedConfig.enableGemmStridedCoopmat2AccF16 != 0;
      const bool incumbentCoopmat2 = !incumbentCoopmat2AccF16 && accF16 && tunedConfig.enableGemmStridedCoopmat2 != 0;
      const bool incumbentCoopmatAccF16 =
        !incumbentCoopmat2AccF16 && !incumbentCoopmat2 && tunedConfig.enableGemmStridedCoopmatAccF16 != 0;
      const bool incumbentCoopmat = !incumbentCoopmat2AccF16 && !incumbentCoopmat2 && !incumbentCoopmatAccF16 &&
                                    tunedConfig.enableGemmStridedCoopmat != 0;
      const bool incumbentDot2AccF16 = !incumbentCoopmat2AccF16 && !incumbentCoopmat2 && !incumbentCoopmatAccF16 &&
                                       !incumbentCoopmat && tunedConfig.enableGemmStridedDot2AccF16 != 0;
      const bool incumbentDot2 = !incumbentCoopmat2AccF16 && !incumbentCoopmat2 && !incumbentCoopmatAccF16 &&
                                 !incumbentCoopmat && !incumbentDot2AccF16 && tunedConfig.enableGemmStridedDot2 != 0;
      string_view incumbentName = incumbentCoopmat2AccF16  ? "gemmStridedCoopmat2AccF16"
                                  : incumbentCoopmat2      ? "gemmStridedCoopmat2"
                                  : incumbentCoopmatAccF16 ? "gemmStridedCoopmatAccF16"
                                  : incumbentCoopmat       ? "gemmStridedCoopmat"
                                  : incumbentDot2AccF16    ? "gemmStridedDot2AccF16"
                                  : incumbentDot2          ? "gemmStridedDot2"
                                                           : "gemmStridedTiled";
      bool incumbentAllOk = true;
      bool cm2AllOk = true;
      double incumbentWeightedTime = 0.0;
      double cm2WeightedTime = 0.0;
      double incumbentWeightedFlops = 0.0;
      double cm2WeightedFlops = 0.0;
      double maxRmse = 0.0;
      string_view challengerName = accF16 ? "gemmStridedCoopmat2AccF16" : "gemmStridedCoopmat2";
      out << endl
          << "VulkanTuner: gemmStrided " << (accF16 ? "coopmat2AccF16" : "coopmat2") << " mode-select cases ("
          << cases.size() << ") ..." << endl;
      for(size_t i = 0; i < cases.size(); i++) {
        const GemmStridedModeCase& cs = cases[i];
        KernelBench incumbentBench;
        double incumbentTflops = 0.0;
        double incumbentFlopsPerDispatch = 0.0;
        if(incumbentCoopmat2AccF16) {
          const auto pad =
            padDims(cs.M, cs.N, cs.K, VulkanKernels::GemmStridedCoopmat2AccF16::layerPaddingContract(tunedConfig));
          VulkanKernels::GemmStridedCoopmat2AccF16 probe(problemBatchSize, pad.m, pad.n, pad.k);
          incumbentBench = probe.bench(ctx, tunedConfig, iters);
          incumbentTflops = probe.estimatedTflops(tunedConfig, incumbentBench.kernelsPerSecond);
          incumbentFlopsPerDispatch = probe.estimatedFlopsPerDispatch(tunedConfig);
        } else if(incumbentCoopmat2) {
          const auto pad =
            padDims(cs.M, cs.N, cs.K, VulkanKernels::GemmStridedCoopmat2::layerPaddingContract(tunedConfig));
          VulkanKernels::GemmStridedCoopmat2 probe(problemBatchSize, pad.m, pad.n, pad.k);
          incumbentBench = probe.bench(ctx, tunedConfig, iters);
          incumbentTflops = probe.estimatedTflops(tunedConfig, incumbentBench.kernelsPerSecond);
          incumbentFlopsPerDispatch = probe.estimatedFlopsPerDispatch(tunedConfig);
        } else if(incumbentCoopmatAccF16) {
          const auto pad =
            padDims(cs.M, cs.N, cs.K, VulkanKernels::GemmStridedCoopmatAccF16::layerPaddingContract(tunedConfig));
          VulkanKernels::GemmStridedCoopmatAccF16 probe(problemBatchSize, pad.m, pad.n, pad.k);
          incumbentBench = probe.bench(ctx, tunedConfig, iters);
          incumbentTflops = probe.estimatedTflops(tunedConfig, incumbentBench.kernelsPerSecond);
          incumbentFlopsPerDispatch = probe.estimatedFlopsPerDispatch(tunedConfig);
        } else if(incumbentCoopmat) {
          const auto pad =
            padDims(cs.M, cs.N, cs.K, VulkanKernels::GemmStridedCoopmat::layerPaddingContract(tunedConfig));
          VulkanKernels::GemmStridedCoopmat probe(problemBatchSize, pad.m, pad.n, pad.k);
          incumbentBench = probe.bench(ctx, tunedConfig, iters);
          incumbentTflops = probe.estimatedTflops(tunedConfig, incumbentBench.kernelsPerSecond);
          incumbentFlopsPerDispatch = probe.estimatedFlopsPerDispatch(tunedConfig);
        } else if(incumbentDot2AccF16) {
          const auto pad =
            padDims(cs.M, cs.N, cs.K, VulkanKernels::GemmStridedDot2AccF16::layerPaddingContract(tunedConfig));
          VulkanKernels::GemmStridedDot2AccF16 probe(problemBatchSize, pad.m, pad.n, pad.k);
          incumbentBench = probe.bench(ctx, tunedConfig, iters);
          incumbentTflops = probe.estimatedTflops(tunedConfig, incumbentBench.kernelsPerSecond);
          incumbentFlopsPerDispatch = probe.estimatedFlopsPerDispatch(tunedConfig);
        } else if(incumbentDot2) {
          const auto pad = padDims(cs.M, cs.N, cs.K, VulkanKernels::GemmStridedDot2::layerPaddingContract(tunedConfig));
          VulkanKernels::GemmStridedDot2 probe(problemBatchSize, pad.m, pad.n, pad.k);
          incumbentBench = probe.bench(ctx, tunedConfig, iters);
          incumbentTflops = probe.estimatedTflops(tunedConfig, incumbentBench.kernelsPerSecond);
          incumbentFlopsPerDispatch = probe.estimatedFlopsPerDispatch(tunedConfig);
        } else {
          const auto pad =
            padDims(cs.M, cs.N, cs.K, VulkanKernels::GemmStridedTiled::layerPaddingContract(tunedConfig));
          VulkanKernels::GemmStridedTiled probe(problemBatchSize, pad.m, pad.n, pad.k);
          incumbentBench = probe.bench(ctx, tunedConfig, iters);
          incumbentTflops = probe.estimatedTflops(tunedConfig, incumbentBench.kernelsPerSecond);
          incumbentFlopsPerDispatch = probe.estimatedFlopsPerDispatch(tunedConfig);
        }
        bool incumbentOk = incumbentBench.ok && incumbentBench.kernelsPerSecond > 0.0;
        double incumbentCaseWeightedTime = 0.0;
        if(incumbentOk) {
          incumbentCaseWeightedTime = dispatchWeightedTime(incumbentBench, cs.occurrences);
          incumbentWeightedTime += incumbentCaseWeightedTime;
          incumbentWeightedFlops += incumbentFlopsPerDispatch * (double)cs.occurrences;
        } else
          incumbentAllOk = false;

        const auto cm2Pad =
          accF16
            ? padDims(cs.M, cs.N, cs.K, VulkanKernels::GemmStridedCoopmat2AccF16::layerPaddingContract(tunedConfig))
            : padDims(cs.M, cs.N, cs.K, VulkanKernels::GemmStridedCoopmat2::layerPaddingContract(tunedConfig));
        KernelBench cm2Bench;
        double cm2Tflops = 0.0;
        double cm2FlopsPerDispatch = 0.0;
        if(accF16) {
          VulkanKernels::GemmStridedCoopmat2AccF16 cm2Probe(problemBatchSize, cm2Pad.m, cm2Pad.n, cm2Pad.k);
          cm2Bench = cm2Probe.bench(ctx, tunedConfig, iters);
          cm2Tflops = cm2Probe.estimatedTflops(tunedConfig, cm2Bench.kernelsPerSecond);
          cm2FlopsPerDispatch = cm2Probe.estimatedFlopsPerDispatch(tunedConfig);
        } else {
          VulkanKernels::GemmStridedCoopmat2 cm2Probe(problemBatchSize, cm2Pad.m, cm2Pad.n, cm2Pad.k);
          cm2Bench = cm2Probe.bench(ctx, tunedConfig, iters);
          cm2Tflops = cm2Probe.estimatedTflops(tunedConfig, cm2Bench.kernelsPerSecond);
          cm2FlopsPerDispatch = cm2Probe.estimatedFlopsPerDispatch(tunedConfig);
        }
        bool cm2Ok = cm2Bench.ok && cm2Bench.kernelsPerSecond > 0.0;
        double cm2CaseWeightedTime = 0.0;
        double rmse = std::numeric_limits<double>::infinity();
        if(cm2Ok) {
          VulkanKernels::GemmStridedTiled ref(problemBatchSize, cm2Pad.m, cm2Pad.n, cm2Pad.k);
          KernelBench refBench = ref.bench(ctx, tunedConfig, iters);
          cm2Ok = refBench.ok;
          if(cm2Ok) {
            rmse = normalizedRmse(refBench.output, cm2Bench.output);
            maxRmse = std::max(maxRmse, rmse);
            cm2Ok = rmse <= 0.02;
          }
        }
        if(cm2Ok) {
          cm2CaseWeightedTime = dispatchWeightedTime(cm2Bench, cs.occurrences);
          cm2WeightedTime += cm2CaseWeightedTime;
          cm2WeightedFlops += cm2FlopsPerDispatch * (double)cs.occurrences;
        } else
          cm2AllOk = false;
        if(shouldPrintTunerResult(verboseTuner, false, incumbentOk && cm2Ok)) {
          out << "VulkanTuner:   gemmStrided " << (accF16 ? "coopmat2AccF16" : "coopmat2") << " case#" << i;
          if(incumbentOk) {
            out << " " << incumbentName << "_kps=" << incumbentBench.kernelsPerSecond;
            if(incumbentTflops > 0.0)
              out << " " << incumbentName << "_tflops=" << incumbentTflops;
            out << " " << incumbentName << "_weighted_time=" << incumbentCaseWeightedTime;
          } else
            out << " " << incumbentName << "=failed";
          if(cm2Bench.ok && cm2Bench.kernelsPerSecond > 0.0) {
            out << " " << challengerName << "_kps=" << cm2Bench.kernelsPerSecond;
            if(cm2Tflops > 0.0)
              out << " " << challengerName << "_tflops=" << cm2Tflops;
            out << " " << challengerName << "_weighted_time=" << cm2CaseWeightedTime;
          } else
            out << " " << challengerName << "=failed";
          out << " rmse=" << (std::isfinite(rmse) ? Global::doubleToString(rmse) : string("n/a")) << " "
              << challengerName << "_result=" << (cm2Ok ? "accepted" : "rejected") << endl;
        }
      }
      ModeSelectStats incumbentStats{incumbentAllOk, incumbentWeightedTime, incumbentWeightedFlops};
      ModeSelectStats cm2Stats{cm2AllOk, cm2WeightedTime, cm2WeightedFlops};
      bool selected = appendModeSelectResult(
        out,
        "gemmStrided",
        incumbentName,
        challengerName,
        cases,
        incumbentStats,
        cm2Stats,
        maxRmse,
        false,
        false,
        [](std::ostream&) {});
      if(selected) {
        if(accF16) {
          tunedConfig.enableGemmStridedCoopmat2AccF16 = 1;
          promoteGemmStridedSelectionRank(tunedConfig, &VulkanTuneParams::gemmStridedCoopmat2AccF16SelectionRank);
        } else {
          tunedConfig.enableGemmStridedCoopmat2 = 1;
          promoteGemmStridedSelectionRank(tunedConfig, &VulkanTuneParams::gemmStridedCoopmat2SelectionRank);
        }
        tunedConfig.enableGemmStridedCoopmat2 = accF16 ? 0 : tunedConfig.enableGemmStridedCoopmat2;
        tunedConfig.enableGemmStridedCoopmat2AccF16 = accF16 ? tunedConfig.enableGemmStridedCoopmat2AccF16 : 0;
        tunedConfig.enableGemmStridedCoopmat = 0;
        tunedConfig.enableGemmStridedCoopmatAccF16 = 0;
        tunedConfig.enableGemmStridedDot2 = 0;
        tunedConfig.enableGemmStridedDot2AccF16 = 0;
        tunedConfig.enableGemmStridedTiledFp16Compute = 0;
      } else if(
        !incumbentCoopmat2AccF16 && !incumbentCoopmat2 && !incumbentCoopmatAccF16 && !incumbentCoopmat &&
        !incumbentDot2AccF16 && !incumbentDot2)
        promoteGemmStridedSelectionRank(tunedConfig, &VulkanTuneParams::gemmStridedTiledSelectionRank);
    }

    void tuneCoopmat2ModeSelect(
      TuningContext& ctx,
      const ModelDesc* modelDesc,
      int batchSize,
      int nnXLen,
      int nnYLen,
      int iters,
      std::ostream& out,
      bool verboseTuner,
      VulkanTuneParams& tunedConfig) {
      tunedConfig.enableWinogradGemmCoopmat2 = 0;
      tunedConfig.enableGemmStridedCoopmat2 = 0;
      const bool haveDevice = ctx.dev != nullptr;
      const VulkanDeviceInfo* info = haveDevice ? &ctx.dev->info : nullptr;
      const bool supportsFP16Compute = haveDevice && ctx.fp16Compute;
      const bool supportsCoopmat2F16 = haveDevice && ctx.dev->info.supportsCoopmat2F16;
      if(!haveDevice || !supportsCoopmat2F16 || !ctx.fp16Storage || !supportsFP16Compute) {
        out << endl << "VulkanTuner: skip coopmat2 mode-select (";
        appendMissingVulkanFp16FeatureReason(
          out,
          "coopmat2F16",
          info,
          supportsCoopmat2F16,
          info != nullptr && info->unfilteredSupportsCoopmat2F16,
          info != nullptr && info->filterDisabledCoopmat2F16,
          ctx.fp16Storage,
          supportsFP16Compute);
        out << ")." << endl;
        return;
      }
      std::vector<Coopmat2Tile> candidates = makeCoopmat2Candidates(ctx, ctx.dev->info.coopmat2FlexShapes);
      if(candidates.empty()) {
        out << endl
            << "VulkanTuner: skip coopmat2 mode-select (no usable coopmat2 flexible-dimension candidates)." << endl;
        return;
      }
      out << endl << "VulkanTuner: mode-select coopmat2 vs current best for winograd/strided GEMM..." << endl;
      out << "VulkanTuner: coopmat2 candidate count=" << candidates.size()
          << " flex_shapes=" << ctx.dev->info.coopmat2FlexShapes.size() << endl;
      const int problemBatchSize = std::max(1, batchSize);
      if(tunedConfig.gemmStridedCoopmat2TunerValueValid != 0)
        tuneGemmStridedCoopmat2ModeSelect(
          ctx, modelDesc, problemBatchSize, nnXLen, nnYLen, iters, false, out, verboseTuner, tunedConfig);
      if(tunedConfig.winogradGemmCoopmat2TunerValueValid != 0)
        tuneWinogradCoopmat2ModeSelect(
          ctx, modelDesc, problemBatchSize, nnXLen, nnYLen, iters, false, out, verboseTuner, tunedConfig);
    }

    void tuneCoopmat2AccF16ModeSelect(
      TuningContext& ctx,
      const ModelDesc* modelDesc,
      int batchSize,
      int nnXLen,
      int nnYLen,
      int iters,
      std::ostream& out,
      bool verboseTuner,
      VulkanTuneParams& tunedConfig) {
      tunedConfig.enableWinogradGemmCoopmat2AccF16 = 0;
      tunedConfig.enableGemmStridedCoopmat2AccF16 = 0;
      const bool haveDevice = ctx.dev != nullptr;
      const VulkanDeviceInfo* info = haveDevice ? &ctx.dev->info : nullptr;
      const bool supportsFP16Compute = haveDevice && ctx.fp16Compute;
      const bool supportsCoopmat2F16AccF16 = haveDevice && ctx.dev->info.supportsCoopmat2F16AccF16;
      if(!haveDevice || !supportsCoopmat2F16AccF16 || !ctx.fp16Storage || !supportsFP16Compute) {
        out << endl << "VulkanTuner: skip coopmat2AccF16 mode-select (";
        appendMissingVulkanFp16FeatureReason(
          out,
          "coopmat2F16AccF16",
          info,
          supportsCoopmat2F16AccF16,
          info != nullptr && info->unfilteredSupportsCoopmat2F16AccF16,
          info != nullptr && info->filterDisabledCoopmat2F16AccF16,
          ctx.fp16Storage,
          supportsFP16Compute);
        out << ")." << endl;
        return;
      }
      std::vector<Coopmat2Tile> candidates = makeCoopmat2Candidates(ctx, ctx.dev->info.coopmat2AccF16FlexShapes);
      if(candidates.empty()) {
        out << endl
            << "VulkanTuner: skip coopmat2AccF16 mode-select "
               "(no usable coopmat2AccF16 flexible-dimension candidates)."
            << endl;
        return;
      }
      out << endl << "VulkanTuner: mode-select coopmat2AccF16 vs current best for winograd/strided GEMM..." << endl;
      out << "VulkanTuner: coopmat2AccF16 candidate count=" << candidates.size()
          << " flex_shapes=" << ctx.dev->info.coopmat2AccF16FlexShapes.size() << endl;
      const int problemBatchSize = std::max(1, batchSize);
      if(tunedConfig.gemmStridedCoopmat2AccF16TunerValueValid != 0)
        tuneGemmStridedCoopmat2ModeSelect(
          ctx, modelDesc, problemBatchSize, nnXLen, nnYLen, iters, true, out, verboseTuner, tunedConfig);
      if(tunedConfig.winogradGemmCoopmat2AccF16TunerValueValid != 0)
        tuneWinogradCoopmat2ModeSelect(
          ctx, modelDesc, problemBatchSize, nnXLen, nnYLen, iters, true, out, verboseTuner, tunedConfig);
    }

    CoopmatTile currentWinogradCoopmatTile(const VulkanTuneParams& config, bool accF16) {
      return accF16 ? CoopmatTile{
                      config.coopmatAccF16BlockSize,
                      config.coopmatAccF16BM,
                      config.coopmatAccF16BN,
                      config.coopmatAccF16BK,
                      config.coopmatAccF16WM,
                      config.coopmatAccF16WN,
                      config.coopmatAccF16TM,
                      config.coopmatAccF16TN,
                      config.coopmatAccF16TK,
                      config.coopmatAccF16Warp}
                  : CoopmatTile{
                      config.coopmatBlockSize,
                      config.coopmatBM,
                      config.coopmatBN,
                      config.coopmatBK,
                      config.coopmatWM,
                      config.coopmatWN,
                      config.coopmatTM,
                      config.coopmatTN,
                      config.coopmatTK,
                      config.coopmatWarp};
    }

    CoopmatTile currentStridedCoopmatTile(const VulkanTuneParams& config, bool accF16) {
      return accF16 ? CoopmatTile{
                      config.stridedCoopmatAccF16BlockSize,
                      config.stridedCoopmatAccF16BM,
                      config.stridedCoopmatAccF16BN,
                      config.stridedCoopmatAccF16BK,
                      config.stridedCoopmatAccF16WM,
                      config.stridedCoopmatAccF16WN,
                      config.stridedCoopmatAccF16TM,
                      config.stridedCoopmatAccF16TN,
                      config.stridedCoopmatAccF16TK,
                      config.stridedCoopmatAccF16Warp}
                  : CoopmatTile{
                      config.stridedCoopmatBlockSize,
                      config.stridedCoopmatBM,
                      config.stridedCoopmatBN,
                      config.stridedCoopmatBK,
                      config.stridedCoopmatWM,
                      config.stridedCoopmatWN,
                      config.stridedCoopmatTM,
                      config.stridedCoopmatTN,
                      config.stridedCoopmatTK,
                      config.stridedCoopmatWarp};
    }

    Coopmat2Tile currentWinogradCoopmat2Tile(const VulkanTuneParams& config, bool accF16) {
      return accF16
               ? Coopmat2Tile{config.coopmat2AccF16BlockSize, config.coopmat2AccF16BM, config.coopmat2AccF16BN, config.coopmat2AccF16BK}
               : Coopmat2Tile{config.coopmat2BlockSize, config.coopmat2BM, config.coopmat2BN, config.coopmat2BK};
    }

    Coopmat2Tile currentStridedCoopmat2Tile(const VulkanTuneParams& config, bool accF16) {
      return accF16 ? Coopmat2Tile{
                      config.stridedCoopmat2AccF16BlockSize,
                      config.stridedCoopmat2AccF16BM,
                      config.stridedCoopmat2AccF16BN,
                      config.stridedCoopmat2AccF16BK}
                  : Coopmat2Tile{
                      config.stridedCoopmat2BlockSize,
                      config.stridedCoopmat2BM,
                      config.stridedCoopmat2BN,
                      config.stridedCoopmat2BK};
    }

    bool coopmatTileSupportedByDevice(
      const VulkanDeviceInfo& info,
      const std::vector<CoopmatShape>& shapes,
      const CoopmatTile& tile) {
      if(tile.warp != (int32_t)info.subgroupSize)
        return false;
      for(const CoopmatShape& shape: shapes) {
        if(shape.m == (uint32_t)tile.tm && shape.n == (uint32_t)tile.tn && shape.k == (uint32_t)tile.tk)
          return true;
      }
      return false;
    }

    void tuneMissingGemmVariantCandidatesForCurrentHardware(
      TuningContext& ctx,
      const ModelDesc* modelDesc,
      int batchSize,
      int nnXLen,
      int nnYLen,
      int iters,
      std::ostream& out,
      bool verboseTuner,
      VulkanTuneParams& tunedConfig) {
      if(ctx.dev == nullptr || modelDesc == nullptr)
        return;
      disableTiledFp16ComputeGemmVariants(tunedConfig);
      const bool supportsFP16Compute = ctx.fp16Compute;
      const int problemBatchSize = std::max(1, batchSize);

      if(ctx.dev->info.supportsDot2F16 && ctx.fp16Storage && supportsFP16Compute) {
        if(tunedConfig.winogradGemmDot2TunerValueValid == 0)
          tuneWinogradDot2TileSweep(
            ctx, modelDesc, problemBatchSize, nnXLen, nnYLen, iters, out, verboseTuner, false, tunedConfig);
        if(tunedConfig.gemmStridedDot2TunerValueValid == 0)
          tuneGemmStridedDot2TileSweep(
            ctx, modelDesc, problemBatchSize, nnXLen, nnYLen, iters, out, verboseTuner, false, tunedConfig);
      }
      if(ctx.dev->info.supportsDot2F16AccF16 && ctx.fp16Storage && supportsFP16Compute) {
        if(tunedConfig.winogradGemmDot2AccF16TunerValueValid == 0)
          tuneWinogradDot2TileSweep(
            ctx, modelDesc, problemBatchSize, nnXLen, nnYLen, iters, out, verboseTuner, true, tunedConfig);
        if(tunedConfig.gemmStridedDot2AccF16TunerValueValid == 0)
          tuneGemmStridedDot2TileSweep(
            ctx, modelDesc, problemBatchSize, nnXLen, nnYLen, iters, out, verboseTuner, true, tunedConfig);
      }

      if(ctx.dev->info.supportsCoopmat1F16 && ctx.fp16Storage && supportsFP16Compute) {
        std::vector<CoopmatTile> candidates = makeCoopmatCandidates(ctx, ctx.dev->info.coopmatShapes);
        if(
          tunedConfig.winogradGemmCoopmatTunerValueValid != 0 &&
          !coopmatTileSupportedByDevice(
            ctx.dev->info, ctx.dev->info.coopmatShapes, currentWinogradCoopmatTile(tunedConfig, false)))
          tunedConfig.winogradGemmCoopmatTunerValueValid = 0;
        if(
          tunedConfig.gemmStridedCoopmatTunerValueValid != 0 &&
          !coopmatTileSupportedByDevice(
            ctx.dev->info, ctx.dev->info.coopmatShapes, currentStridedCoopmatTile(tunedConfig, false)))
          tunedConfig.gemmStridedCoopmatTunerValueValid = 0;
        if(!candidates.empty()) {
          if(tunedConfig.winogradGemmCoopmatTunerValueValid == 0)
            tuneWinogradCoopmatTileSweep(
              ctx,
              modelDesc,
              problemBatchSize,
              nnXLen,
              nnYLen,
              iters,
              candidates,
              false,
              out,
              verboseTuner,
              tunedConfig);
          if(tunedConfig.gemmStridedCoopmatTunerValueValid == 0)
            tuneGemmStridedCoopmatTileSweep(
              ctx,
              modelDesc,
              problemBatchSize,
              nnXLen,
              nnYLen,
              iters,
              candidates,
              false,
              out,
              verboseTuner,
              tunedConfig);
        }
      }
      if(ctx.dev->info.supportsCoopmat1F16AccF16 && ctx.fp16Storage && supportsFP16Compute) {
        std::vector<CoopmatTile> candidates = makeCoopmatCandidates(ctx, ctx.dev->info.coopmatAccF16Shapes);
        if(
          tunedConfig.winogradGemmCoopmatAccF16TunerValueValid != 0 &&
          !coopmatTileSupportedByDevice(
            ctx.dev->info, ctx.dev->info.coopmatAccF16Shapes, currentWinogradCoopmatTile(tunedConfig, true)))
          tunedConfig.winogradGemmCoopmatAccF16TunerValueValid = 0;
        if(
          tunedConfig.gemmStridedCoopmatAccF16TunerValueValid != 0 &&
          !coopmatTileSupportedByDevice(
            ctx.dev->info, ctx.dev->info.coopmatAccF16Shapes, currentStridedCoopmatTile(tunedConfig, true)))
          tunedConfig.gemmStridedCoopmatAccF16TunerValueValid = 0;
        if(!candidates.empty()) {
          if(tunedConfig.winogradGemmCoopmatAccF16TunerValueValid == 0)
            tuneWinogradCoopmatTileSweep(
              ctx,
              modelDesc,
              problemBatchSize,
              nnXLen,
              nnYLen,
              iters,
              candidates,
              true,
              out,
              verboseTuner,
              tunedConfig);
          if(tunedConfig.gemmStridedCoopmatAccF16TunerValueValid == 0)
            tuneGemmStridedCoopmatTileSweep(
              ctx,
              modelDesc,
              problemBatchSize,
              nnXLen,
              nnYLen,
              iters,
              candidates,
              true,
              out,
              verboseTuner,
              tunedConfig);
        }
      }

      if(ctx.dev->info.supportsCoopmat2F16 && ctx.fp16Storage && supportsFP16Compute) {
        std::vector<Coopmat2Tile> candidates = makeCoopmat2Candidates(ctx, ctx.dev->info.coopmat2FlexShapes);
        if(
          tunedConfig.winogradGemmCoopmat2TunerValueValid != 0 &&
          !coopmat2TileSupported(
            ctx.dev->info, ctx.dev->info.coopmat2FlexShapes, currentWinogradCoopmat2Tile(tunedConfig, false)))
          tunedConfig.winogradGemmCoopmat2TunerValueValid = 0;
        if(
          tunedConfig.gemmStridedCoopmat2TunerValueValid != 0 &&
          !coopmat2TileSupported(
            ctx.dev->info, ctx.dev->info.coopmat2FlexShapes, currentStridedCoopmat2Tile(tunedConfig, false)))
          tunedConfig.gemmStridedCoopmat2TunerValueValid = 0;
        if(!candidates.empty()) {
          if(tunedConfig.winogradGemmCoopmat2TunerValueValid == 0)
            tuneWinogradCoopmat2TileSweep(
              ctx,
              modelDesc,
              problemBatchSize,
              nnXLen,
              nnYLen,
              iters,
              candidates,
              false,
              out,
              verboseTuner,
              tunedConfig);
          if(tunedConfig.gemmStridedCoopmat2TunerValueValid == 0)
            tuneGemmStridedCoopmat2TileSweep(
              ctx,
              modelDesc,
              problemBatchSize,
              nnXLen,
              nnYLen,
              iters,
              candidates,
              false,
              out,
              verboseTuner,
              tunedConfig);
        }
      }
      if(ctx.dev->info.supportsCoopmat2F16AccF16 && ctx.fp16Storage && supportsFP16Compute) {
        std::vector<Coopmat2Tile> candidates = makeCoopmat2Candidates(ctx, ctx.dev->info.coopmat2AccF16FlexShapes);
        if(
          tunedConfig.winogradGemmCoopmat2AccF16TunerValueValid != 0 &&
          !coopmat2TileSupported(
            ctx.dev->info, ctx.dev->info.coopmat2AccF16FlexShapes, currentWinogradCoopmat2Tile(tunedConfig, true)))
          tunedConfig.winogradGemmCoopmat2AccF16TunerValueValid = 0;
        if(
          tunedConfig.gemmStridedCoopmat2AccF16TunerValueValid != 0 &&
          !coopmat2TileSupported(
            ctx.dev->info, ctx.dev->info.coopmat2AccF16FlexShapes, currentStridedCoopmat2Tile(tunedConfig, true)))
          tunedConfig.gemmStridedCoopmat2AccF16TunerValueValid = 0;
        if(!candidates.empty()) {
          if(tunedConfig.winogradGemmCoopmat2AccF16TunerValueValid == 0)
            tuneWinogradCoopmat2TileSweep(
              ctx,
              modelDesc,
              problemBatchSize,
              nnXLen,
              nnYLen,
              iters,
              candidates,
              true,
              out,
              verboseTuner,
              tunedConfig);
          if(tunedConfig.gemmStridedCoopmat2AccF16TunerValueValid == 0)
            tuneGemmStridedCoopmat2TileSweep(
              ctx,
              modelDesc,
              problemBatchSize,
              nnXLen,
              nnYLen,
              iters,
              candidates,
              true,
              out,
              verboseTuner,
              tunedConfig);
        }
      }
    }

    void selectGemmVariantsForCurrentHardware(
      TuningContext& ctx,
      const ModelDesc* modelDesc,
      int batchSize,
      int nnXLen,
      int nnYLen,
      int iters,
      std::ostream& out,
      bool verboseTuner,
      VulkanTuneParams& tunedConfig) {
      const VulkanDeviceInfo* info = ctx.dev != nullptr ? &ctx.dev->info : nullptr;
      if(info != nullptr)
        tunedConfig.normalizeGemmVariantSelectionsForDevice(*info, ctx.fp16Storage, ctx.fp16Compute);
      else
        tunedConfig.normalizeGemmVariantSelections();

      tuneDot2ModeSelect(ctx, modelDesc, batchSize, nnXLen, nnYLen, iters, out, verboseTuner, tunedConfig);
      tuneCoopmatModeSelect(ctx, modelDesc, batchSize, nnXLen, nnYLen, iters, out, verboseTuner, tunedConfig);
      tuneCoopmatAccF16ModeSelect(ctx, modelDesc, batchSize, nnXLen, nnYLen, iters, out, verboseTuner, tunedConfig);
      tuneCoopmat2ModeSelect(ctx, modelDesc, batchSize, nnXLen, nnYLen, iters, out, verboseTuner, tunedConfig);
      tuneCoopmat2AccF16ModeSelect(ctx, modelDesc, batchSize, nnXLen, nnYLen, iters, out, verboseTuner, tunedConfig);
      const bool normalized =
        info != nullptr ? tunedConfig.normalizeGemmVariantSelectionsForDevice(*info, ctx.fp16Storage, ctx.fp16Compute)
                        : tunedConfig.normalizeGemmVariantSelections();
      if(normalized) {
        out
          << endl
          << "VulkanTuner: normalized GEMM variant selections using tuned candidate order and current hardware filter."
          << endl;
      }
    }

  }  // namespace

  bool tuneGemmVariantsForCurrentHardware(
    TuningContext& ctx,
    const ModelDesc* modelDesc,
    int batchSize,
    int nnXLen,
    int nnYLen,
    int benchIters,
    std::ostream& out,
    bool verboseTuner,
    VulkanTuneParams& tunedConfig) {
    if(ctx.dev == nullptr || modelDesc == nullptr)
      return false;
    const VulkanTuneParams before = tunedConfig;
    const int iters = std::max(1, benchIters);
    out << endl << "VulkanTuner: updating GEMM variant candidates for current hardware..." << endl;
    tuneMissingGemmVariantCandidatesForCurrentHardware(
      ctx, modelDesc, batchSize, nnXLen, nnYLen, iters, out, verboseTuner, tunedConfig);
    selectGemmVariantsForCurrentHardware(
      ctx, modelDesc, batchSize, nnXLen, nnYLen, iters, out, verboseTuner, tunedConfig);
    tunedConfig.gemmVariantFilterTunedSignature = ctx.dev->info.gemmVariantFilterSignature;
    return !(tunedConfig == before);
  }

  bool retuneWinogradTransformsForCurrentLayout(
    TuningContext& ctx,
    const ModelDesc* modelDesc,
    int batchSize,
    int nnXLen,
    int nnYLen,
    int benchIters,
    std::ostream& out,
    bool verboseTuner,
    VulkanTuneParams& tunedConfig) {
    if(modelDesc == nullptr)
      return false;
    const int iters = std::max(1, benchIters);
    const TrunkDesc& trunk = modelDesc->trunk;
    int wgChannels = trunk.trunkNumChannels > 0 ? trunk.trunkNumChannels : 64;
    bool tunedTransform = false;
    bool tunedBNActTransform = false;
    const std::string retuneReason = tunedConfig.winogradTransformTunedLayout.empty()
                                       ? "no previous Winograd transform layout tune recorded"
                                       : (winogradTransformsTunedForCurrentLayout(tunedConfig, ctx.fp16Storage)
                                            ? "retune requested for selected Winograd GEMM layout"
                                            : "selected Winograd GEMM layout changed");
    auto printRetuneNotice = [&](const std::string& kernelName) {
      out << endl
          << "VulkanTuner: re-tuning " << kernelName << " for selected Winograd GEMM layout (reason: " << retuneReason
          << ")..." << endl;
    };
    {
      VulkanKernels::WinogradTransform kernel(batchSize, wgChannels, nnXLen, nnYLen);
      printRetuneNotice(kernel.name());
      std::vector<TunedCandidate> top = tuneOne(kernel, tunedConfig, ctx, iters, out, verboseTuner);
      if(!top.empty()) {
        top[0].applyTo(tunedConfig);
        tunedTransform = true;
      }
    }
    {
      VulkanKernels::WinogradBNActTransform kernel(batchSize, wgChannels, nnXLen, nnYLen, ACTIVATION_MISH_SCALE8);
      printRetuneNotice(kernel.name());
      std::vector<TunedCandidate> top = tuneOne(kernel, tunedConfig, ctx, iters, out, verboseTuner);
      if(!top.empty()) {
        top[0].applyTo(tunedConfig);
        tunedBNActTransform = true;
      }
    }
    if(tunedTransform && tunedBNActTransform)
      markWinogradTransformsTunedForCurrentLayout(tunedConfig, ctx.fp16Storage);
    return tunedTransform && tunedBNActTransform;
  }

  // ---- Top-level tuning ----

  void tune(
    TuningContext* ctx,
    const VulkanTuneParams& initialConfig,
    const ModelDesc* modelDesc,
    int batchSize,
    int nnXLen,
    int nnYLen,
    int benchIters,
    std::ostream& out,
    bool verboseTuner,
    VulkanTuneParams& tunedConfig) {
    tunedConfig = initialConfig;
    if(ctx == nullptr)
      return;
    const int iters = std::max(1, benchIters);

    auto kernels = makeMicroKernels(modelDesc, nnXLen, nnYLen, batchSize, tunedConfig);
    for(auto& kernel: kernels) {
      out << endl << "VulkanTuner: tuning " << kernel->name() << "..." << endl;
      if(kernel->name() == "winogradGemmTiled") {
        VulkanTuneParams tunedCandidate = tunedConfig;
        bool tuned = tuneWinogradGemmTiledMultiShape<VulkanKernels::WinogradGemm>(
          "winogradGemmTiled",
          modelDesc,
          batchSize,
          nnXLen,
          nnYLen,
          tunedConfig,
          *ctx,
          iters,
          out,
          verboseTuner,
          tunedCandidate);
        if(tuned) {
          tunedConfig = tunedCandidate;
          continue;
        }
        out << "VulkanTuner: falling back to single-shape winogradGemmTiled tuning." << endl;
      }
      if(kernel->name() == "gemmStridedTiled") {
        VulkanTuneParams tunedCandidate = tunedConfig;
        bool tuned = tuneGemmStridedTiledMultiShape<VulkanKernels::GemmStridedTiled>(
          "gemmStridedTiled",
          modelDesc,
          batchSize,
          nnXLen,
          nnYLen,
          tunedConfig,
          *ctx,
          iters,
          out,
          verboseTuner,
          tunedCandidate);
        if(tuned) {
          tunedConfig = tunedCandidate;
          continue;
        }
        out << "VulkanTuner: falling back to single-shape gemmStridedTiled tuning." << endl;
      }
      // Seed each kernel's sweep from the winners locked in so far, so candidate
      // problem padding that depends on already-tuned params (e.g. Winograd's
      // selected GEMM contract) sees the chosen values.
      std::vector<TunedCandidate> top = tuneOne(*kernel, tunedConfig, *ctx, iters, out, verboseTuner);
      if(!top.empty())
        top[0].applyTo(tunedConfig);
    }

    tuneGemmVariantsForCurrentHardware(
      *ctx, modelDesc, batchSize, nnXLen, nnYLen, iters, out, verboseTuner, tunedConfig);

    // These benches derive their padded tile layout from the currently selected
    // Winograd GEMM variant, so keep them after all Winograd GEMM mode-selects.
    out << endl << "VulkanTuner: tuning winograd transforms for selected GEMM layout..." << endl;
    retuneWinogradTransformsForCurrentLayout(
      *ctx, modelDesc, batchSize, nnXLen, nnYLen, iters, out, verboseTuner, tunedConfig);

    // All three power-of-two reduction kernels (gpool, valueHeadPool, spatialRMSNorm)
    // are now tuned independently above (the last is gated on hasTransformer).
  }

}  // namespace VulkanTuner

// VulkanKernels member impls that need TuningContext (a full type only here).
// Each instantiable kernel (TunableKernel subclass) defines its bench() here.
// ============================================================================

namespace VulkanKernels {

  namespace {

    template<typename BuildKernel, typename DispatchKernel>
    KernelBench benchWinogradGemmVariant(
      VulkanTuner::TuningContext& s,
      int iters,
      int problemM,
      int problemN,
      int problemK,
      int problemNumBatches,
      string_view candidateName,
      BuildKernel&& buildKernel,
      DispatchKernel&& dispatchKernel,
      bool packB = false,
      int packedBM = 0,
      int packedBN = 0,
      int packedBK = 0,
      int packedPadWords = WINOGRAD_COOPMAT_PACKED_PAD_WORDS) {
      KernelBench result;
      if(iters < 1)
        return result;
      try {
        ComputeKernel kernel = buildKernel();

        size_t aElts = (size_t)problemNumBatches * problemK * problemM;
        size_t bElts = (size_t)problemNumBatches * problemK * problemN;
        size_t cElts = (size_t)problemNumBatches * problemN * problemM;
        std::vector<float> aData(aElts), bData(bElts);
        uint32_t seed = (uint32_t)(problemM * 73856093u ^ problemN * 19349663u ^ problemK * 83492791u);
        VulkanTuner::fillRandom(aData, seed);
        VulkanTuner::fillRandom(bData, seed ^ 0x9e3779b9u);

        std::vector<float> aUploadData, bUploadData;
        aUploadData =
          packWinogradRowMajorAData(aData, problemNumBatches, problemM, problemK, packedBM, packedBK, packedPadWords);
        if(packB)
          bUploadData = packWinogradCoopmatBWeights(
            bData, problemNumBatches, problemN, problemK, packedBN, packedBK, packedPadWords);

        VBuf aBuf = s.makeInputBufFP(aUploadData);
        VBuf bBuf = s.makeInputBufFP(packB ? bUploadData : bData);
        VBuf cBuf = makeDeviceBuf(s.device(), s.memProps(), cElts, s.fp16Storage);

        // strideA is the row-major packed A tile stride in vec4 units.
        // strideB is variant-specific: normal Winograd GEMM uses K*(N/4), coopmat uses packed tile words.
        // strideC in vec4 units = (N/4)*M (M ÷4 by the bench roundUp to a ÷4 multiple).
        size_t strideA = winogradPackedRowMajorAStrideWords(problemM, problemK, packedBM, packedBK, packedPadWords);
        size_t strideB = packB
                           ? winogradCoopmatPackedBStrideWords(problemN, problemK, packedBN, packedBK, packedPadWords)
                           : (size_t)problemK * (size_t)(problemN / 4);
        testAssert(strideA <= (size_t)std::numeric_limits<int>::max());
        testAssert(strideB <= (size_t)std::numeric_limits<int>::max());
        WinogradGemm::PC pc = {
          problemM,
          problemN,
          (int)strideA,                // strideA, vec4
          (int)strideB,                // strideB, vec4
          (problemN / 4) * problemM};  // strideC, vec4

        CmdCtx cctx{s.cmd, s.getPushDescFn(), nullptr};
        auto recordOne = [&]() {
          dispatchKernel(cctx, kernel, aBuf.get(), bBuf.get(), cBuf.get(), pc, problemNumBatches);
          VulkanHelpers::cmdComputeBarrier(s.cmd, cBuf->buffer);
        };

        s.beginRecording();
        recordOne();
        s.submitAndWait();
        auto [seconds, effectiveIters] = timeBudgetedBench(s, iters, recordOne);
        if(seconds <= 0.0) {
          kernel.destroy(s.device());
          return result;
        }
        result.output.resize(cElts);
        s.downloadFloatsFP(cBuf.get(), result.output, cElts);
        result.kernelsPerSecond = (double)effectiveIters / seconds;
        result.ok = true;
        kernel.destroy(s.device());
        return result;
      } catch(const std::exception& e) {
        if(s.logger != nullptr)
          s.logger->write(std::string("VulkanTuner: ") + string(candidateName) + " candidate failed: " + e.what());
        return result;
      }
    }

    template<typename BuildKernel, typename DispatchKernel>
    KernelBench benchGemmStridedVariant(
      VulkanTuner::TuningContext& s,
      int iters,
      int problemBatchSize,
      int problemM,
      int problemN,
      int problemK,
      string_view candidateName,
      BuildKernel&& buildKernel,
      DispatchKernel&& dispatchKernel,
      bool packB = false,
      int packedBN = 0,
      int packedBK = 0,
      int packedPadScalars = 0) {
      KernelBench result;
      if(iters < 1)
        return result;
      try {
        ComputeKernel kernel = buildKernel();

        size_t aElts = (size_t)problemBatchSize * problemK * problemM;
        size_t bElts = (size_t)problemBatchSize * problemK * problemN;
        size_t cElts = (size_t)problemBatchSize * problemN * problemM;
        std::vector<float> aData(aElts), bData(bElts);
        uint32_t seed = (uint32_t)(problemM * 73856093u ^ problemN * 19349663u ^ problemK * 83492791u);
        VulkanTuner::fillRandom(aData, seed);
        VulkanTuner::fillRandom(bData, seed ^ 0x9e3779b9u);

        std::vector<float> bUploadData;
        if(packB)
          bUploadData =
            packStridedGemmBWeights(bData, problemBatchSize, problemN, problemK, packedBN, packedBK, packedPadScalars);

        VBuf aBuf = s.makeInputBufFP(aData);
        VBuf bBuf = s.makeInputBufFP(packB ? bUploadData : bData);
        VBuf cBuf = makeDeviceBuf(s.device(), s.memProps(), cElts, s.fp16Storage);
        size_t strideB = packB ? stridedGemmPackedBStrideVec4s(problemN, problemK, packedBN, packedBK, packedPadScalars)
                               : (size_t)problemK * (size_t)(problemN / 4);
        testAssert(strideB <= (size_t)std::numeric_limits<int>::max());
        GemmStridedTiled::PC pc = {
          problemM,
          problemN,
          problemN,  // N_real == N (bench uses aligned N)
          problemK * (problemM / 4),
          (int)strideB,
          problemN * (problemM / 4)};

        auto recordOne = [&]() {
          CmdCtx cctx{s.cmd, s.getPushDescFn(), nullptr};
          dispatchKernel(cctx, kernel, aBuf.get(), bBuf.get(), cBuf.get(), pc, problemBatchSize);
          VulkanHelpers::cmdComputeBarrier(s.cmd, cBuf->buffer);
        };

        s.beginRecording();
        recordOne();
        s.submitAndWait();
        auto [seconds, effectiveIters] = timeBudgetedBench(s, iters, recordOne);
        if(seconds <= 0.0) {
          kernel.destroy(s.device());
          return result;
        }
        result.output.resize(cElts);
        s.downloadFloatsFP(cBuf.get(), result.output, cElts);
        result.kernelsPerSecond = (double)effectiveIters / seconds;
        result.ok = true;
        kernel.destroy(s.device());
        return result;
      } catch(const std::exception& e) {
        if(s.logger != nullptr)
          s.logger->write(std::string("VulkanTuner: ") + string(candidateName) + " candidate failed: " + e.what());
        return result;
      }
    }

  }  // namespace

  KernelBench WinogradGemm::bench(VulkanTuner::TuningContext& s, const VulkanTuneParams& cfg, int iters) const {
    return benchWinogradGemmVariant(
      s,
      iters,
      problemM,
      problemN,
      problemK,
      problemNumBatches,
      "winogradGemm",
      [&]() {
        return WinogradGemm::build(
          s.device(),
          VK_NULL_HANDLE,
          s.fp16Storage,
          (uint32_t)cfg.winogradGemmM,
          (uint32_t)cfg.winogradGemmN,
          (uint32_t)cfg.winogradGemmK,
          (uint32_t)cfg.winogradGemmRN,
          (uint32_t)problemK);
      },
      [&](
        const CmdCtx& cctx,
        const ComputeKernel& kernel,
        VulkanBuffer* aBuf,
        VulkanBuffer* bBuf,
        VulkanBuffer* cBuf,
        const WinogradGemm::PC& pc,
        int numBatches) { WinogradGemm::dispatch(cctx, kernel, aBuf, bBuf, cBuf, problemK, pc, numBatches); },
      false,
      cfg.winogradGemmM,
      0,
      cfg.winogradGemmK,
      WINOGRAD_ROW_MAJOR_A_PAD_WORDS);
  }

  KernelBench WinogradTransform::bench(VulkanTuner::TuningContext& s, const VulkanTuneParams& cfg, int iters) const {
    KernelBench result;
    if(iters < 1)
      return result;
    try {
      const int outTile = winograd3x3OutTileFor(cfg);
      const int inTile = outTile + 2;
      int numTilesX = (problemNnXLen + outTile - 1) / outTile;
      int numTilesY = (problemNnYLen + outTile - 1) / outTile;
      int ntxty = problemBatchSize * numTilesX * numTilesY;
      WinogradTransformBenchLayout layout = makeWinogradTransformBenchLayout(cfg, s.fp16Storage);
      int numTilesPadded = ((ntxty + layout.mAlignment - 1) / layout.mAlignment) * layout.mAlignment;
      int numInChannelsPadded = ((problemInChannels + layout.kAlignment - 1) / layout.kAlignment) * layout.kAlignment;
      int paddedSpatialSize = problemNnXLen * problemNnYLen;

      size_t inputElts = (size_t)problemBatchSize * problemInChannels * paddedSpatialSize;
      size_t packedStrideA = winogradPackedRowMajorAStrideWords(
        numTilesPadded, numInChannelsPadded, layout.packedBM, layout.packedBK, layout.packedAPadWords);
      size_t outputElts = (size_t)inTile * inTile * packedStrideA * 4;

      std::vector<float> inputData(inputElts);
      VulkanTuner::fillRandom(
        inputData,
        (uint32_t)(problemNnXLen * 73856093u ^ problemInChannels * 19349663u ^ problemBatchSize * 83492791u));

      VBuf inputBuf = s.makeInputBufFP(inputData);
      VBuf outputBuf = makeDeviceBuf(s.device(), s.memProps(), outputElts, s.fp16Storage);

      WinogradTransform::PC pc = {
        problemNnXLen,
        problemNnYLen,
        numTilesX,
        numTilesY,
        problemInChannels,
        numInChannelsPadded,
        ntxty,
        numTilesPadded,
        paddedSpatialSize};

      ComputeKernel kernel = WinogradTransform::build(
        s.device(),
        VK_NULL_HANDLE,
        s.fp16Storage,
        inTile,
        outTile,
        3,
        -1,
        cfg.winogradTransformLocalSizeX,
        cfg.winogradTransformLocalSizeY,
        layout.packedBM,
        layout.packedBK,
        layout.packedAPadWords);

      auto recordOne = [&]() {
        CmdCtx cctx{s.cmd, s.getPushDescFn(), nullptr};
        WinogradTransform::dispatch(cctx, kernel, inputBuf.get(), outputBuf.get(), pc);
        VulkanHelpers::cmdComputeBarrier(s.cmd, outputBuf->buffer);
      };

      s.beginRecording();
      recordOne();
      s.submitAndWait();
      auto [seconds, effectiveIters] = timeBudgetedBench(s, iters, recordOne);
      if(seconds <= 0.0) {
        kernel.destroy(s.device());
        return result;
      }
      result.output.resize(outputElts);
      s.downloadFloatsFP(outputBuf.get(), result.output, outputElts);
      result.kernelsPerSecond = (double)effectiveIters / seconds;
      result.ok = true;
      kernel.destroy(s.device());
      return result;
    } catch(const std::exception& e) {
      if(s.logger != nullptr)
        s.logger->write(std::string("VulkanTuner: winograd candidate failed: ") + e.what());
      return result;
    }
  }

  KernelBench WinogradBNActTransform::bench(VulkanTuner::TuningContext& s, const VulkanTuneParams& cfg, int iters)
    const {
    KernelBench result;
    if(iters < 1)
      return result;
    try {
      const int outTile = winograd3x3OutTileFor(cfg);
      const int inTile = outTile + 2;
      int numTilesX = (problemNnXLen + outTile - 1) / outTile;
      int numTilesY = (problemNnYLen + outTile - 1) / outTile;
      int ntxty = problemBatchSize * numTilesX * numTilesY;
      WinogradTransformBenchLayout layout = makeWinogradTransformBenchLayout(cfg, s.fp16Storage);
      int numTilesPadded = ((ntxty + layout.mAlignment - 1) / layout.mAlignment) * layout.mAlignment;
      int numInChannelsPadded = ((problemInChannels + layout.kAlignment - 1) / layout.kAlignment) * layout.kAlignment;
      int paddedSpatialSize = problemNnXLen * problemNnYLen;

      size_t inputElts = (size_t)problemBatchSize * problemInChannels * paddedSpatialSize;
      size_t packedStrideA = winogradPackedRowMajorAStrideWords(
        numTilesPadded, numInChannelsPadded, layout.packedBM, layout.packedBK, layout.packedAPadWords);
      size_t outputElts = (size_t)inTile * inTile * packedStrideA * 4;
      size_t bnElts = (size_t)problemInChannels;
      size_t maskElts = (size_t)problemBatchSize * paddedSpatialSize;

      std::vector<float> inputData(inputElts), scaleData(bnElts), biasData(bnElts), maskData(maskElts);
      VulkanTuner::fillRandom(
        inputData,
        (uint32_t)(problemNnXLen * 73856093u ^ problemInChannels * 19349663u ^ problemBatchSize * 83492791u ^ 11u));
      VulkanTuner::fillRandom(scaleData, (uint32_t)(problemInChannels * 2654435761u ^ 17u));
      VulkanTuner::fillRandom(biasData, (uint32_t)(problemInChannels * 2246822519u ^ 23u));
      VulkanTuner::fillRandom(maskData, (uint32_t)(problemBatchSize * 3266489917u ^ paddedSpatialSize * 668265263u));

      VBuf inputBuf = s.makeInputBufFP(inputData);
      VBuf outputBuf = makeDeviceBuf(s.device(), s.memProps(), outputElts, s.fp16Storage);
      VBuf scaleBuf = s.makeInputBufFP(scaleData);
      VBuf biasBuf = s.makeInputBufFP(biasData);
      VBuf maskBuf = s.makeInputBufFP(maskData);

      WinogradBNActTransform::PC pc = {
        problemNnXLen,
        problemNnYLen,
        numTilesX,
        numTilesY,
        problemInChannels,
        numInChannelsPadded,
        ntxty,
        numTilesPadded,
        paddedSpatialSize};

      ComputeKernel kernel = WinogradBNActTransform::build(
        s.device(),
        VK_NULL_HANDLE,
        s.fp16Storage,
        inTile,
        outTile,
        3,
        -1,
        problemActivation,
        cfg.winogradBNActTransformLocalSizeX,
        cfg.winogradBNActTransformLocalSizeY,
        layout.packedBM,
        layout.packedBK,
        layout.packedAPadWords);

      auto recordOne = [&]() {
        CmdCtx cctx{s.cmd, s.getPushDescFn(), nullptr};
        WinogradBNActTransform::dispatch(
          cctx, kernel, inputBuf.get(), outputBuf.get(), scaleBuf.get(), biasBuf.get(), maskBuf.get(), pc);
        VulkanHelpers::cmdComputeBarrier(s.cmd, outputBuf->buffer);
      };

      s.beginRecording();
      recordOne();
      s.submitAndWait();
      auto [seconds, effectiveIters] = timeBudgetedBench(s, iters, recordOne);
      if(seconds <= 0.0) {
        kernel.destroy(s.device());
        return result;
      }
      result.output.resize(outputElts);
      s.downloadFloatsFP(outputBuf.get(), result.output, outputElts);
      result.kernelsPerSecond = (double)effectiveIters / seconds;
      result.ok = true;
      kernel.destroy(s.device());
      return result;
    } catch(const std::exception& e) {
      if(s.logger != nullptr)
        s.logger->write(std::string("VulkanTuner: winogradBNActTransform candidate failed: ") + e.what());
      return result;
    }
  }

  KernelBench WinogradUntransform::bench(VulkanTuner::TuningContext& s, const VulkanTuneParams& cfg, int iters) const {
    KernelBench result;
    if(iters < 1)
      return result;
    try {
      const int outTile = winograd3x3OutTileFor(cfg);
      const int inTile = outTile + 2;
      int numTilesX = (problemNnXLen + outTile - 1) / outTile;
      int numTilesY = (problemNnYLen + outTile - 1) / outTile;
      int ntxty = problemBatchSize * numTilesX * numTilesY;
      WinogradTransformBenchLayout layout = makeWinogradTransformBenchLayout(cfg, s.fp16Storage);
      int numTilesPadded = ((ntxty + layout.mAlignment - 1) / layout.mAlignment) * layout.mAlignment;
      int numOutChannelsPadded = ((problemOutChannels + layout.nAlignment - 1) / layout.nAlignment) * layout.nAlignment;
      int paddedSpatialSize = problemNnXLen * problemNnYLen;

      size_t inputElts = (size_t)inTile * inTile * numOutChannelsPadded * numTilesPadded;
      size_t outputElts = (size_t)problemBatchSize * problemOutChannels * paddedSpatialSize;

      std::vector<float> inputData(inputElts);
      VulkanTuner::fillRandom(
        inputData,
        (uint32_t)(problemNnXLen * 73856093u ^ problemOutChannels * 19349663u ^ problemBatchSize * 83492791u ^ 29u));

      VBuf inputBuf = s.makeInputBufFP(inputData);
      VBuf outputBuf = makeDeviceBuf(s.device(), s.memProps(), outputElts, s.fp16Storage);

      WinogradUntransform::PC pc = {
        problemNnXLen,
        problemNnYLen,
        numTilesX,
        numTilesY,
        problemOutChannels,
        numOutChannelsPadded,
        numTilesPadded,
        paddedSpatialSize};

      ComputeKernel kernel = WinogradUntransform::build(
        s.device(),
        VK_NULL_HANDLE,
        s.fp16Storage,
        inTile,
        outTile,
        3,
        cfg.winogradUntransformLocalSizeX,
        cfg.winogradUntransformLocalSizeY,
        false);

      auto recordOne = [&]() {
        CmdCtx cctx{s.cmd, s.getPushDescFn(), nullptr};
        WinogradUntransform::dispatch(cctx, kernel, inputBuf.get(), outputBuf.get(), pc, problemBatchSize);
        VulkanHelpers::cmdComputeBarrier(s.cmd, outputBuf->buffer);
      };

      s.beginRecording();
      recordOne();
      s.submitAndWait();
      auto [seconds, effectiveIters] = timeBudgetedBench(s, iters, recordOne);
      if(seconds <= 0.0) {
        kernel.destroy(s.device());
        return result;
      }
      result.output.resize(outputElts);
      s.downloadFloatsFP(outputBuf.get(), result.output, outputElts);
      result.kernelsPerSecond = (double)effectiveIters / seconds;
      result.ok = true;
      kernel.destroy(s.device());
      return result;
    } catch(const std::exception& e) {
      if(s.logger != nullptr)
        s.logger->write(std::string("VulkanTuner: winogradUntransform candidate failed: ") + e.what());
      return result;
    }
  }

  KernelBench GemmStridedTiled::bench(VulkanTuner::TuningContext& s, const VulkanTuneParams& cfg, int iters) const {
    return benchGemmStridedVariant(
      s,
      iters,
      problemBatchSize,
      problemM,
      problemN,
      problemK,
      "gemmStridedTiled",
      [&]() {
        return GemmStridedTiled::build(
          s.device(),
          VK_NULL_HANDLE,
          s.fp16Storage,
          (uint32_t)cfg.gemmStridedTiledLocalSizeX,
          (uint32_t)cfg.gemmStridedTiledLocalSizeY,
          (uint32_t)cfg.gemmStridedTiledTileK,
          (uint32_t)cfg.gemmStridedTiledRN,
          (uint32_t)problemK);
      },
      [&](
        const CmdCtx& cctx,
        const ComputeKernel& kernel,
        VulkanBuffer* aBuf,
        VulkanBuffer* bBuf,
        VulkanBuffer* cBuf,
        const GemmStridedTiled::PC& pc,
        int batchSize) { GemmStridedTiled::dispatch(cctx, kernel, aBuf, bBuf, cBuf, problemK, pc, batchSize); });
  }

  KernelBench WinogradGemmDot2::bench(VulkanTuner::TuningContext& s, const VulkanTuneParams& cfg, int iters) const {
    if(s.dev == nullptr || !s.dev->info.supportsDot2F16 || !s.fp16Storage)
      return KernelBench();
    return benchWinogradGemmVariant(
      s,
      iters,
      problemM,
      problemN,
      problemK,
      problemNumBatches,
      "winogradGemmDot2",
      [&]() {
        return WinogradGemmDot2::build(
          s.device(),
          VK_NULL_HANDLE,
          cfg.dot2BlockSize,
          cfg.dot2BM,
          cfg.dot2BN,
          cfg.dot2WM,
          cfg.dot2WN,
          cfg.dot2WMIter,
          cfg.dot2TM,
          cfg.dot2TN,
          cfg.dot2Warp,
          problemK);
      },
      [&](
        const CmdCtx& cctx,
        const ComputeKernel& kernel,
        VulkanBuffer* aBuf,
        VulkanBuffer* bBuf,
        VulkanBuffer* cBuf,
        const WinogradGemm::PC& pc,
        int numBatches) { WinogradGemmDot2::dispatch(cctx, kernel, aBuf, bBuf, cBuf, problemK, pc, numBatches); },
      false,
      cfg.dot2BM,
      0,
      VulkanKernels::DOT2_BK,
      WINOGRAD_ROW_MAJOR_A_PAD_WORDS);
  }

  KernelBench GemmStridedDot2::bench(VulkanTuner::TuningContext& s, const VulkanTuneParams& cfg, int iters) const {
    if(s.dev == nullptr || !s.dev->info.supportsDot2F16 || !s.fp16Storage)
      return KernelBench();
    // Compute aligned from the actual bench dims. Match runtime: ALIGNED signals
    // M/N safety only, while K_ALIGNED separately specializes the K-tail guard.
    const auto tile = GemmStridedDot2::tileAlignmentReq(cfg);
    const int32_t aligned = (problemM % tile.m == 0 && problemN % tile.n == 0) ? 1 : 0;
    const int32_t kAligned = (problemK % tile.k == 0) ? 1 : 0;
    return benchGemmStridedVariant(
      s,
      iters,
      problemBatchSize,
      problemM,
      problemN,
      problemK,
      "gemmStridedDot2",
      [&]() {
        return GemmStridedDot2::build(
          s.device(),
          VK_NULL_HANDLE,
          cfg.stridedDot2BlockSize,
          cfg.stridedDot2BM,
          cfg.stridedDot2BN,
          cfg.stridedDot2WM,
          cfg.stridedDot2WN,
          cfg.stridedDot2WMIter,
          cfg.stridedDot2TM,
          cfg.stridedDot2TN,
          cfg.stridedDot2Warp,
          aligned,
          1,
          kAligned,
          problemK);
      },
      [&](
        const CmdCtx& cctx,
        const ComputeKernel& kernel,
        VulkanBuffer* aBuf,
        VulkanBuffer* bBuf,
        VulkanBuffer* cBuf,
        const GemmStridedTiled::PC& pc,
        int batchSize) { GemmStridedDot2::dispatch(cctx, kernel, aBuf, bBuf, cBuf, problemK, pc, batchSize); },
      true,
      cfg.stridedDot2BN,
      VulkanKernels::DOT2_BK,
      STRIDED_DOT2_PACKED_B_PAD_SCALARS);
  }

  KernelBench WinogradGemmDot2AccF16::bench(VulkanTuner::TuningContext& s, const VulkanTuneParams& cfg, int iters)
    const {
    if(s.dev == nullptr || !s.dev->info.supportsDot2F16AccF16 || !s.fp16Storage)
      return KernelBench();
    return benchWinogradGemmVariant(
      s,
      iters,
      problemM,
      problemN,
      problemK,
      problemNumBatches,
      "winogradGemmDot2AccF16",
      [&]() {
        return WinogradGemmDot2AccF16::build(
          s.device(),
          VK_NULL_HANDLE,
          cfg.dot2AccF16BlockSize,
          cfg.dot2AccF16BM,
          cfg.dot2AccF16BN,
          cfg.dot2AccF16WM,
          cfg.dot2AccF16WN,
          cfg.dot2AccF16WMIter,
          cfg.dot2AccF16TM,
          cfg.dot2AccF16TN,
          cfg.dot2AccF16Warp,
          problemK);
      },
      [&](
        const CmdCtx& cctx,
        const ComputeKernel& kernel,
        VulkanBuffer* aBuf,
        VulkanBuffer* bBuf,
        VulkanBuffer* cBuf,
        const WinogradGemm::PC& pc,
        int numBatches) { WinogradGemmDot2AccF16::dispatch(cctx, kernel, aBuf, bBuf, cBuf, problemK, pc, numBatches); },
      false,
      cfg.dot2AccF16BM,
      0,
      VulkanKernels::DOT2_BK,
      WINOGRAD_ROW_MAJOR_A_PAD_WORDS);
  }

  KernelBench GemmStridedDot2AccF16::bench(VulkanTuner::TuningContext& s, const VulkanTuneParams& cfg, int iters)
    const {
    if(s.dev == nullptr || !s.dev->info.supportsDot2F16AccF16 || !s.fp16Storage)
      return KernelBench();
    const auto tile = GemmStridedDot2AccF16::tileAlignmentReq(cfg);
    const int32_t aligned = (problemM % tile.m == 0 && problemN % tile.n == 0) ? 1 : 0;
    const int32_t kAligned = (problemK % tile.k == 0) ? 1 : 0;
    return benchGemmStridedVariant(
      s,
      iters,
      problemBatchSize,
      problemM,
      problemN,
      problemK,
      "gemmStridedDot2AccF16",
      [&]() {
        return GemmStridedDot2AccF16::build(
          s.device(),
          VK_NULL_HANDLE,
          cfg.stridedDot2AccF16BlockSize,
          cfg.stridedDot2AccF16BM,
          cfg.stridedDot2AccF16BN,
          cfg.stridedDot2AccF16WM,
          cfg.stridedDot2AccF16WN,
          cfg.stridedDot2AccF16WMIter,
          cfg.stridedDot2AccF16TM,
          cfg.stridedDot2AccF16TN,
          cfg.stridedDot2AccF16Warp,
          aligned,
          1,
          kAligned,
          problemK);
      },
      [&](
        const CmdCtx& cctx,
        const ComputeKernel& kernel,
        VulkanBuffer* aBuf,
        VulkanBuffer* bBuf,
        VulkanBuffer* cBuf,
        const GemmStridedTiled::PC& pc,
        int batchSize) { GemmStridedDot2AccF16::dispatch(cctx, kernel, aBuf, bBuf, cBuf, problemK, pc, batchSize); },
      true,
      cfg.stridedDot2AccF16BN,
      VulkanKernels::DOT2_BK,
      STRIDED_DOT2_PACKED_B_PAD_SCALARS);
  }

  namespace {
    // The coopmat TM/TN/TK must be a device-reported shape. bench() enforces this
    // (validate() only sees VkPhysicalDeviceLimits, not the shape list).
    bool coopmatShapeIsSupported(const std::vector<CoopmatShape>& shapes, int32_t tm, int32_t tn, int32_t tk) {
      for(const auto& shape: shapes)
        if(shape.m == (uint32_t)tm && shape.n == (uint32_t)tn && shape.k == (uint32_t)tk)
          return true;
      return false;
    }
  }  // namespace

  KernelBench WinogradGemmCoopmat::bench(VulkanTuner::TuningContext& s, const VulkanTuneParams& cfg, int iters) const {
    if(s.dev == nullptr || !s.dev->info.supportsCoopmat1F16 || !s.fp16Storage)
      return KernelBench();
    if(!coopmatShapeIsSupported(s.dev->info.coopmatShapes, cfg.coopmatTM, cfg.coopmatTN, cfg.coopmatTK))
      return KernelBench();
    return benchWinogradGemmVariant(
      s,
      iters,
      problemM,
      problemN,
      problemK,
      problemNumBatches,
      "winogradGemmCoopmat",
      [&]() {
        return WinogradGemmCoopmat::build(
          s.device(),
          VK_NULL_HANDLE,
          cfg.coopmatBlockSize,
          cfg.coopmatBM,
          cfg.coopmatBN,
          cfg.coopmatBK,
          cfg.coopmatWM,
          cfg.coopmatWN,
          cfg.coopmatTM,
          cfg.coopmatTN,
          cfg.coopmatTK,
          cfg.coopmatWarp,
          problemK,
          s.dev->info.subgroupSize);
      },
      [&](
        const CmdCtx& cctx,
        const ComputeKernel& kernel,
        VulkanBuffer* aBuf,
        VulkanBuffer* bBuf,
        VulkanBuffer* cBuf,
        const WinogradGemm::PC& pc,
        int numBatches) { WinogradGemmCoopmat::dispatch(cctx, kernel, aBuf, bBuf, cBuf, problemK, pc, numBatches); },
      true,
      cfg.coopmatBM,
      cfg.coopmatBN,
      cfg.coopmatBK);
  }

  KernelBench GemmStridedCoopmat::bench(VulkanTuner::TuningContext& s, const VulkanTuneParams& cfg, int iters) const {
    if(s.dev == nullptr || !s.dev->info.supportsCoopmat1F16 || !s.fp16Storage)
      return KernelBench();
    if(!coopmatShapeIsSupported(
         s.dev->info.coopmatShapes, cfg.stridedCoopmatTM, cfg.stridedCoopmatTN, cfg.stridedCoopmatTK))
      return KernelBench();
    // Strided K is not padded across layers, so ALIGNED signals M/N safety only;
    // K_ALIGNED separately specializes the K-tail guard.
    const auto tile = GemmStridedCoopmat::tileAlignmentReq(cfg);
    const int32_t aligned = (problemM % tile.m == 0 && problemN % tile.n == 0) ? 1 : 0;
    const int32_t kAligned = (problemK % tile.k == 0) ? 1 : 0;
    return benchGemmStridedVariant(
      s,
      iters,
      problemBatchSize,
      problemM,
      problemN,
      problemK,
      "gemmStridedCoopmat",
      [&]() {
        return GemmStridedCoopmat::build(
          s.device(),
          VK_NULL_HANDLE,
          cfg.stridedCoopmatBlockSize,
          cfg.stridedCoopmatBM,
          cfg.stridedCoopmatBN,
          cfg.stridedCoopmatBK,
          cfg.stridedCoopmatWM,
          cfg.stridedCoopmatWN,
          cfg.stridedCoopmatTM,
          cfg.stridedCoopmatTN,
          cfg.stridedCoopmatTK,
          cfg.stridedCoopmatWarp,
          aligned,
          1,
          kAligned,
          problemK,
          s.dev->info.subgroupSize);
      },
      [&](
        const CmdCtx& cctx,
        const ComputeKernel& kernel,
        VulkanBuffer* aBuf,
        VulkanBuffer* bBuf,
        VulkanBuffer* cBuf,
        const GemmStridedTiled::PC& pc,
        int batchSize) { GemmStridedCoopmat::dispatch(cctx, kernel, aBuf, bBuf, cBuf, problemK, pc, batchSize); },
      true,
      cfg.stridedCoopmatBN,
      cfg.stridedCoopmatBK,
      STRIDED_COOPMAT_PACKED_B_PAD_SCALARS);
  }

  KernelBench WinogradGemmCoopmatAccF16::bench(VulkanTuner::TuningContext& s, const VulkanTuneParams& cfg, int iters)
    const {
    if(s.dev == nullptr || !s.dev->info.supportsCoopmat1F16AccF16 || !s.fp16Storage)
      return KernelBench();
    if(!coopmatShapeIsSupported(
         s.dev->info.coopmatAccF16Shapes, cfg.coopmatAccF16TM, cfg.coopmatAccF16TN, cfg.coopmatAccF16TK))
      return KernelBench();
    return benchWinogradGemmVariant(
      s,
      iters,
      problemM,
      problemN,
      problemK,
      problemNumBatches,
      "winogradGemmCoopmatAccF16",
      [&]() {
        return WinogradGemmCoopmatAccF16::build(
          s.device(),
          VK_NULL_HANDLE,
          cfg.coopmatAccF16BlockSize,
          cfg.coopmatAccF16BM,
          cfg.coopmatAccF16BN,
          cfg.coopmatAccF16BK,
          cfg.coopmatAccF16WM,
          cfg.coopmatAccF16WN,
          cfg.coopmatAccF16TM,
          cfg.coopmatAccF16TN,
          cfg.coopmatAccF16TK,
          cfg.coopmatAccF16Warp,
          problemK,
          s.dev->info.subgroupSize);
      },
      [&](
        const CmdCtx& cctx,
        const ComputeKernel& kernel,
        VulkanBuffer* aBuf,
        VulkanBuffer* bBuf,
        VulkanBuffer* cBuf,
        const WinogradGemm::PC& pc,
        int numBatches) {
        WinogradGemmCoopmatAccF16::dispatch(cctx, kernel, aBuf, bBuf, cBuf, problemK, pc, numBatches);
      },
      true,
      cfg.coopmatAccF16BM,
      cfg.coopmatAccF16BN,
      cfg.coopmatAccF16BK);
  }

  KernelBench GemmStridedCoopmatAccF16::bench(VulkanTuner::TuningContext& s, const VulkanTuneParams& cfg, int iters)
    const {
    if(s.dev == nullptr || !s.dev->info.supportsCoopmat1F16AccF16 || !s.fp16Storage)
      return KernelBench();
    if(!coopmatShapeIsSupported(
         s.dev->info.coopmatAccF16Shapes,
         cfg.stridedCoopmatAccF16TM,
         cfg.stridedCoopmatAccF16TN,
         cfg.stridedCoopmatAccF16TK))
      return KernelBench();
    const auto tile = GemmStridedCoopmatAccF16::tileAlignmentReq(cfg);
    const int32_t aligned = (problemM % tile.m == 0 && problemN % tile.n == 0) ? 1 : 0;
    const int32_t kAligned = (problemK % tile.k == 0) ? 1 : 0;
    return benchGemmStridedVariant(
      s,
      iters,
      problemBatchSize,
      problemM,
      problemN,
      problemK,
      "gemmStridedCoopmatAccF16",
      [&]() {
        return GemmStridedCoopmatAccF16::build(
          s.device(),
          VK_NULL_HANDLE,
          cfg.stridedCoopmatAccF16BlockSize,
          cfg.stridedCoopmatAccF16BM,
          cfg.stridedCoopmatAccF16BN,
          cfg.stridedCoopmatAccF16BK,
          cfg.stridedCoopmatAccF16WM,
          cfg.stridedCoopmatAccF16WN,
          cfg.stridedCoopmatAccF16TM,
          cfg.stridedCoopmatAccF16TN,
          cfg.stridedCoopmatAccF16TK,
          cfg.stridedCoopmatAccF16Warp,
          aligned,
          1,
          kAligned,
          problemK,
          s.dev->info.subgroupSize);
      },
      [&](
        const CmdCtx& cctx,
        const ComputeKernel& kernel,
        VulkanBuffer* aBuf,
        VulkanBuffer* bBuf,
        VulkanBuffer* cBuf,
        const GemmStridedTiled::PC& pc,
        int batchSize) { GemmStridedCoopmatAccF16::dispatch(cctx, kernel, aBuf, bBuf, cBuf, problemK, pc, batchSize); },
      true,
      cfg.stridedCoopmatAccF16BN,
      cfg.stridedCoopmatAccF16BK,
      STRIDED_COOPMAT_PACKED_B_PAD_SCALARS);
  }

  namespace {
    bool coopmat2TileIsSupported(
      const VulkanTuner::TuningContext& s,
      const std::vector<Coopmat2FlexShape>& shapes,
      int32_t blockSize,
      int32_t bm,
      int32_t bn,
      int32_t bk) {
      if(s.dev == nullptr)
        return false;
      const VulkanDeviceInfo& info = s.dev->info;
      if(info.coopmat2ReservedSharedBytes > info.properties.limits.maxComputeSharedMemorySize)
        return false;
      if(info.coopmat2MaxWorkgroupSize > 0 && (uint32_t)blockSize > info.coopmat2MaxWorkgroupSize)
        return false;
      if(info.coopmat2MaxFlexDimension > 0) {
        uint32_t maxDim = std::max((uint32_t)bm, std::max((uint32_t)bn, (uint32_t)bk));
        if(maxDim > info.coopmat2MaxFlexDimension)
          return false;
      }
      for(const auto& shape: shapes) {
        if(
          blockSize == (int32_t)shape.workgroupInvocations && bm % (int32_t)shape.mGranularity == 0 &&
          bn % (int32_t)shape.nGranularity == 0 && bk % (int32_t)shape.kGranularity == 0)
          return true;
      }
      return false;
    }
  }  // namespace

  KernelBench WinogradGemmCoopmat2::bench(VulkanTuner::TuningContext& s, const VulkanTuneParams& cfg, int iters) const {
    if(s.dev == nullptr || !s.dev->info.supportsCoopmat2F16 || !s.fp16Storage)
      return KernelBench();
    if(!coopmat2TileIsSupported(
         s, s.dev->info.coopmat2FlexShapes, cfg.coopmat2BlockSize, cfg.coopmat2BM, cfg.coopmat2BN, cfg.coopmat2BK))
      return KernelBench();
    return benchWinogradGemmVariant(
      s,
      iters,
      problemM,
      problemN,
      problemK,
      problemNumBatches,
      "winogradGemmCoopmat2",
      [&]() {
        return WinogradGemmCoopmat2::build(
          s.device(), VK_NULL_HANDLE, cfg.coopmat2BlockSize, cfg.coopmat2BM, cfg.coopmat2BN, cfg.coopmat2BK, problemK);
      },
      [&](
        const CmdCtx& cctx,
        const ComputeKernel& kernel,
        VulkanBuffer* aBuf,
        VulkanBuffer* bBuf,
        VulkanBuffer* cBuf,
        const WinogradGemm::PC& pc,
        int numBatches) { WinogradGemmCoopmat2::dispatch(cctx, kernel, aBuf, bBuf, cBuf, problemK, pc, numBatches); },
      true,
      cfg.coopmat2BM,
      cfg.coopmat2BN,
      cfg.coopmat2BK,
      WINOGRAD_COOPMAT2_PACKED_PAD_WORDS);
  }

  KernelBench GemmStridedCoopmat2::bench(VulkanTuner::TuningContext& s, const VulkanTuneParams& cfg, int iters) const {
    if(s.dev == nullptr || !s.dev->info.supportsCoopmat2F16 || !s.fp16Storage)
      return KernelBench();
    if(!coopmat2TileIsSupported(
         s,
         s.dev->info.coopmat2FlexShapes,
         cfg.stridedCoopmat2BlockSize,
         cfg.stridedCoopmat2BM,
         cfg.stridedCoopmat2BN,
         cfg.stridedCoopmat2BK))
      return KernelBench();
    const auto tile = GemmStridedCoopmat2::tileAlignmentReq(cfg);
    const int32_t aligned = (problemM % tile.m == 0 && problemN % tile.n == 0) ? 1 : 0;
    const int32_t kAligned = (problemK % tile.k == 0) ? 1 : 0;
    return benchGemmStridedVariant(
      s,
      iters,
      problemBatchSize,
      problemM,
      problemN,
      problemK,
      "gemmStridedCoopmat2",
      [&]() {
        return GemmStridedCoopmat2::build(
          s.device(),
          VK_NULL_HANDLE,
          cfg.stridedCoopmat2BlockSize,
          cfg.stridedCoopmat2BM,
          cfg.stridedCoopmat2BN,
          cfg.stridedCoopmat2BK,
          aligned,
          1,
          kAligned,
          problemK);
      },
      [&](
        const CmdCtx& cctx,
        const ComputeKernel& kernel,
        VulkanBuffer* aBuf,
        VulkanBuffer* bBuf,
        VulkanBuffer* cBuf,
        const GemmStridedTiled::PC& pc,
        int batchSize) { GemmStridedCoopmat2::dispatch(cctx, kernel, aBuf, bBuf, cBuf, problemK, pc, batchSize); },
      true,
      cfg.stridedCoopmat2BN,
      cfg.stridedCoopmat2BK,
      STRIDED_COOPMAT2_PACKED_B_PAD_SCALARS);
  }

  KernelBench WinogradGemmCoopmat2AccF16::bench(VulkanTuner::TuningContext& s, const VulkanTuneParams& cfg, int iters)
    const {
    if(s.dev == nullptr || !s.dev->info.supportsCoopmat2F16AccF16 || !s.fp16Storage)
      return KernelBench();
    if(!coopmat2TileIsSupported(
         s,
         s.dev->info.coopmat2AccF16FlexShapes,
         cfg.coopmat2AccF16BlockSize,
         cfg.coopmat2AccF16BM,
         cfg.coopmat2AccF16BN,
         cfg.coopmat2AccF16BK))
      return KernelBench();
    return benchWinogradGemmVariant(
      s,
      iters,
      problemM,
      problemN,
      problemK,
      problemNumBatches,
      "winogradGemmCoopmat2AccF16",
      [&]() {
        return WinogradGemmCoopmat2AccF16::build(
          s.device(),
          VK_NULL_HANDLE,
          cfg.coopmat2AccF16BlockSize,
          cfg.coopmat2AccF16BM,
          cfg.coopmat2AccF16BN,
          cfg.coopmat2AccF16BK,
          problemK);
      },
      [&](
        const CmdCtx& cctx,
        const ComputeKernel& kernel,
        VulkanBuffer* aBuf,
        VulkanBuffer* bBuf,
        VulkanBuffer* cBuf,
        const WinogradGemm::PC& pc,
        int numBatches) {
        WinogradGemmCoopmat2AccF16::dispatch(cctx, kernel, aBuf, bBuf, cBuf, problemK, pc, numBatches);
      },
      true,
      cfg.coopmat2AccF16BM,
      cfg.coopmat2AccF16BN,
      cfg.coopmat2AccF16BK,
      WINOGRAD_COOPMAT2_PACKED_PAD_WORDS);
  }

  KernelBench GemmStridedCoopmat2AccF16::bench(VulkanTuner::TuningContext& s, const VulkanTuneParams& cfg, int iters)
    const {
    if(s.dev == nullptr || !s.dev->info.supportsCoopmat2F16AccF16 || !s.fp16Storage)
      return KernelBench();
    if(!coopmat2TileIsSupported(
         s,
         s.dev->info.coopmat2AccF16FlexShapes,
         cfg.stridedCoopmat2AccF16BlockSize,
         cfg.stridedCoopmat2AccF16BM,
         cfg.stridedCoopmat2AccF16BN,
         cfg.stridedCoopmat2AccF16BK))
      return KernelBench();
    const auto tile = GemmStridedCoopmat2AccF16::tileAlignmentReq(cfg);
    const int32_t aligned = (problemM % tile.m == 0 && problemN % tile.n == 0) ? 1 : 0;
    const int32_t kAligned = (problemK % tile.k == 0) ? 1 : 0;
    return benchGemmStridedVariant(
      s,
      iters,
      problemBatchSize,
      problemM,
      problemN,
      problemK,
      "gemmStridedCoopmat2AccF16",
      [&]() {
        return GemmStridedCoopmat2AccF16::build(
          s.device(),
          VK_NULL_HANDLE,
          cfg.stridedCoopmat2AccF16BlockSize,
          cfg.stridedCoopmat2AccF16BM,
          cfg.stridedCoopmat2AccF16BN,
          cfg.stridedCoopmat2AccF16BK,
          aligned,
          1,
          kAligned,
          problemK);
      },
      [&](
        const CmdCtx& cctx,
        const ComputeKernel& kernel,
        VulkanBuffer* aBuf,
        VulkanBuffer* bBuf,
        VulkanBuffer* cBuf,
        const GemmStridedTiled::PC& pc,
        int batchSize) {
        GemmStridedCoopmat2AccF16::dispatch(cctx, kernel, aBuf, bBuf, cBuf, problemK, pc, batchSize);
      },
      true,
      cfg.stridedCoopmat2AccF16BN,
      cfg.stridedCoopmat2AccF16BK,
      STRIDED_COOPMAT2_PACKED_B_PAD_SCALARS);
  }

  KernelBench GemmDirectFP32::bench(VulkanTuner::TuningContext& s, const VulkanTuneParams& cfg, int iters) const {
    KernelBench result;
    if(iters < 1)
      return result;
    try {
      ComputeKernel kernel = GemmDirectFP32::build(
        s.device(),
        VK_NULL_HANDLE,
        (uint32_t)cfg.gemmDirectLocalSizeX,
        (uint32_t)cfg.gemmDirectLocalSizeY,
        (uint32_t)problemK);
      size_t aElts = (size_t)problemM * problemK;
      size_t bElts = (size_t)problemN * problemK;
      size_t cElts = (size_t)problemM * problemN;
      std::vector<float> aData(aElts), bData(bElts);
      uint32_t seed = (uint32_t)(problemM * 73856093u ^ problemN * 19349663u ^ problemK * 83492791u ^ 12345u);
      VulkanTuner::fillRandom(aData, seed);
      VulkanTuner::fillRandom(bData, seed ^ 0x9e3779b9u);
      VBuf aBuf = s.makeInputBufFP(aData);
      VBuf bBuf = s.makeInputBufFP(bData);
      VBuf cBuf = makeDeviceBuf(s.device(), s.memProps(), cElts, false);  // GemmDirectFP32 output is always FP32
      GemmDirectFP32::PC pc = {problemM, problemN};
      auto recordOne = [&]() {
        CmdCtx cctx{s.cmd, s.getPushDescFn(), nullptr};
        GemmDirectFP32::dispatch(cctx, kernel, aBuf.get(), bBuf.get(), cBuf.get(), problemK, pc);
        VulkanHelpers::cmdComputeBarrier(s.cmd, cBuf->buffer);
      };
      s.beginRecording();
      recordOne();
      s.submitAndWait();
      auto [seconds, effectiveIters] = timeBudgetedBench(s, iters, recordOne);
      if(seconds <= 0.0) {
        kernel.destroy(s.device());
        return result;
      }
      result.output.resize(cElts);
      s.downloadFloats(cBuf.get(), result.output, cElts);
      result.kernelsPerSecond = (double)effectiveIters / seconds;
      result.ok = true;
      kernel.destroy(s.device());
      return result;
    } catch(const std::exception& e) {
      if(s.logger != nullptr)
        s.logger->write(std::string("VulkanTuner: gemmDirect candidate failed: ") + e.what());
      return result;
    }
  }

  KernelBench GPoolReduction::bench(VulkanTuner::TuningContext& s, const VulkanTuneParams& cfg, int iters) const {
    KernelBench result;
    if(iters < 1)
      return result;
    try {
      ComputeKernel kernel = GPoolReduction::build(s.device(), VK_NULL_HANDLE, s.fp16Storage, cfg.gpoolXystride);

      size_t inputElts = (size_t)problemBatchSize * problemChannels * problemXySize;
      size_t outputElts = (size_t)problemBatchSize * problemChannels * 3;  // mean, mean*sqrt, max
      size_t maskElts = (size_t)problemBatchSize * problemXySize;
      size_t maskSumElts = (size_t)problemBatchSize;

      std::vector<float> inputData(inputElts), maskData(maskElts, 1.0f), maskSumData(maskSumElts);
      for(int n = 0; n < problemBatchSize; n++)
        maskSumData[n] = (float)problemXySize;
      VulkanTuner::fillRandom(inputData, (uint32_t)(problemChannels * 73856093u ^ problemXySize * 19349663u));

      VBuf inputBuf = s.makeInputBufFP(inputData);
      VBuf outputBuf = makeDeviceBuf(s.device(), s.memProps(), outputElts, false);  // gpool output is always FP32
      VBuf maskBuf = s.makeInputBufFP(maskData);
      VBuf maskSumBuf = s.makeInputBuf(maskSumData);

      GPoolReduction::PC pc = {problemChannels, problemXySize};
      auto recordOne = [&]() {
        CmdCtx cctx{s.cmd, s.getPushDescFn(), nullptr};
        GPoolReduction::dispatch(
          cctx, kernel, inputBuf.get(), outputBuf.get(), maskBuf.get(), maskSumBuf.get(), pc, problemBatchSize);
        VulkanHelpers::cmdComputeBarrier(s.cmd, outputBuf->buffer);
      };

      s.beginRecording();
      recordOne();
      s.submitAndWait();
      auto [seconds, effectiveIters] = timeBudgetedBench(s, iters, recordOne);
      if(seconds <= 0.0) {
        kernel.destroy(s.device());
        return result;
      }
      result.output.resize(outputElts);
      s.downloadFloats(outputBuf.get(), result.output, outputElts);
      result.kernelsPerSecond = (double)effectiveIters / seconds;
      result.ok = true;
      kernel.destroy(s.device());
      return result;
    } catch(const std::exception& e) {
      if(s.logger != nullptr)
        s.logger->write(std::string("VulkanTuner: gpoolReduction candidate failed: ") + e.what());
      return result;
    }
  }

  KernelBench ValueHeadPool::bench(VulkanTuner::TuningContext& s, const VulkanTuneParams& cfg, int iters) const {
    KernelBench result;
    if(iters < 1)
      return result;
    try {
      ComputeKernel kernel = ValueHeadPool::build(s.device(), VK_NULL_HANDLE, s.fp16Storage, cfg.valueHeadPoolXystride);

      size_t inputElts = (size_t)problemBatchSize * problemChannels * problemXySize;
      size_t outputElts = (size_t)problemBatchSize * problemChannels * 3;  // mean, mean*t*0.1, mean*(t^2*0.01-0.1)
      size_t maskSumElts = (size_t)problemBatchSize;

      std::vector<float> inputData(inputElts), maskSumData(maskSumElts);
      for(int n = 0; n < problemBatchSize; n++)
        maskSumData[n] = (float)problemXySize;
      VulkanTuner::fillRandom(inputData, (uint32_t)(problemChannels * 47093279u ^ problemXySize * 30000023u));

      VBuf inputBuf = s.makeInputBufFP(inputData);
      VBuf outputBuf = makeDeviceBuf(s.device(), s.memProps(), outputElts, false);  // output is always FP32
      VBuf maskSumBuf = s.makeInputBuf(maskSumData);

      ValueHeadPool::PC pc = {problemChannels, problemXySize};
      auto recordOne = [&]() {
        CmdCtx cctx{s.cmd, s.getPushDescFn(), nullptr};
        ValueHeadPool::dispatch(cctx, kernel, inputBuf.get(), outputBuf.get(), maskSumBuf.get(), pc, problemBatchSize);
        VulkanHelpers::cmdComputeBarrier(s.cmd, outputBuf->buffer);
      };

      s.beginRecording();
      recordOne();
      s.submitAndWait();
      auto [seconds, effectiveIters] = timeBudgetedBench(s, iters, recordOne);
      if(seconds <= 0.0) {
        kernel.destroy(s.device());
        return result;
      }
      result.output.resize(outputElts);
      s.downloadFloats(outputBuf.get(), result.output, outputElts);
      result.kernelsPerSecond = (double)effectiveIters / seconds;
      result.ok = true;
      kernel.destroy(s.device());
      return result;
    } catch(const std::exception& e) {
      if(s.logger != nullptr)
        s.logger->write(std::string("VulkanTuner: valueHeadPool candidate failed: ") + e.what());
      return result;
    }
  }

  KernelBench SpatialRMSNorm::bench(VulkanTuner::TuningContext& s, const VulkanTuneParams& cfg, int iters) const {
    KernelBench result;
    if(iters < 1)
      return result;
    try {
      const int tile = cfg.spatialRMSNormTile;
      const int chwSize = problemChannels * problemXySize;
      const int numCHWWorkgroups = (chwSize + tile - 1) / tile;

      const bool useSpatialRMSNormPass2Subgroup = s.dev != nullptr && s.dev->info.supportsSubgroupShuffleCompute;
      std::array<ComputeKernel, 3> kernels =
        SpatialRMSNorm::build(s.device(), VK_NULL_HANDLE, s.fp16Storage, tile, useSpatialRMSNormPass2Subgroup);

      size_t inputElts = (size_t)problemBatchSize * problemChannels * problemXySize;
      size_t maskElts = (size_t)problemBatchSize * problemXySize;
      size_t maskSumElts = (size_t)problemBatchSize;
      size_t gammaBetaElts = (size_t)problemChannels;
      size_t partialElts = (size_t)problemBatchSize * numCHWWorkgroups;
      size_t scalarElts = (size_t)problemBatchSize;

      std::vector<float> inputData(inputElts), maskData(maskElts, 1.0f), maskSumData(maskSumElts);
      std::vector<float> gammaData(gammaBetaElts, 1.0f), betaData(gammaBetaElts, 0.0f);
      for(int n = 0; n < problemBatchSize; n++)
        maskSumData[n] = (float)problemXySize;
      VulkanTuner::fillRandom(inputData, (uint32_t)(problemChannels * 11400714u ^ problemXySize * 2654435u));

      VBuf inputBuf = s.makeInputBufFP(inputData);
      VBuf outputBuf = makeDeviceBuf(s.device(), s.memProps(), inputElts, s.fp16Storage);
      VBuf maskBuf = s.makeInputBufFP(maskData);
      VBuf maskSumBuf = s.makeInputBuf(maskSumData);
      VBuf gammaBuf = s.makeInputBuf(gammaData);
      VBuf betaBuf = s.makeInputBuf(betaData);
      VBuf partialsBuf = makeDeviceBuf(s.device(), s.memProps(), partialElts, false);
      VBuf scalarBuf = makeDeviceBuf(s.device(), s.memProps(), scalarElts, false);

      auto recordOne = [&]() {
        CmdCtx cctx{s.cmd, s.getPushDescFn(), nullptr};
        SpatialRMSNorm::PC pc = {problemChannels, problemXySize, 1e-5f};
        SpatialRMSNorm::dispatch(
          cctx,
          kernels,
          inputBuf.get(),
          outputBuf.get(),
          gammaBuf.get(),
          betaBuf.get(),
          maskBuf.get(),
          maskSumBuf.get(),
          partialsBuf.get(),
          scalarBuf.get(),
          pc,
          problemBatchSize);
        VulkanHelpers::cmdComputeBarrier(s.cmd, outputBuf->buffer);
      };

      s.beginRecording();
      recordOne();
      s.submitAndWait();
      auto [seconds, effectiveIters] = timeBudgetedBench(s, iters, recordOne);
      if(seconds <= 0.0) {
        for(auto& k: kernels)
          k.destroy(s.device());
        return result;
      }
      result.output.resize(inputElts);
      s.downloadFloatsFP(outputBuf.get(), result.output, inputElts);
      result.kernelsPerSecond = (double)effectiveIters / seconds;
      result.ok = true;
      for(auto& k: kernels)
        k.destroy(s.device());
      return result;
    } catch(const std::exception& e) {
      if(s.logger != nullptr)
        s.logger->write(std::string("VulkanTuner: spatialRMSNorm candidate failed: ") + e.what());
      return result;
    }
  }

  KernelBench AttentionTiled::bench(VulkanTuner::TuningContext& s, const VulkanTuneParams& cfg, int iters) const {
    KernelBench result;
    if(iters < 1)
      return result;
    try {
      ComputeKernel kernel = AttentionTiled::build(
        s.device(),
        VK_NULL_HANDLE,
        s.fp16Storage,
        cfg.attnBlockQ,
        cfg.attnBlockKV,
        cfg.attnQPerThread,
        problemNumTokens,
        problemHeadDim,
        problemVHeadDim,
        problemUseRope,
        problemLearnableRope);

      size_t qElts = (size_t)problemBatchSize * problemNumHeads * problemHeadDim * problemNumTokens;
      size_t kElts = (size_t)problemBatchSize * problemNumKVHeads * problemHeadDim * problemNumTokens;
      size_t vElts = (size_t)problemBatchSize * problemNumKVHeads * problemVHeadDim * problemNumTokens;
      size_t outElts = (size_t)problemBatchSize * problemNumHeads * problemVHeadDim * problemNumTokens;
      size_t maskElts = (size_t)problemBatchSize * problemNumTokens;
      int numPairs = problemHeadDim / 2;

      std::vector<float> qData(qElts), kData(kElts), vData(vElts), maskData(maskElts, 1.0f);
      uint32_t seed =
        (uint32_t)(problemNumTokens * 73856093u ^ problemHeadDim * 19349663u ^ problemNumHeads * 83492791u);
      VulkanTuner::fillRandom(qData, seed);
      VulkanTuner::fillRandom(kData, seed ^ 0x1111u);
      VulkanTuner::fillRandom(vData, seed ^ 0x2222u);

      VBuf qBuf = s.makeInputBufFP(qData);
      VBuf kBuf = s.makeInputBufFP(kData);
      VBuf vBuf = s.makeInputBufFP(vData);
      VBuf outBuf = makeDeviceBuf(s.device(), s.memProps(), outElts, s.fp16Storage);
      // mask is bound at both the FP32 and FP16 slots via bindingMap, so its
      // storage width must match s.fp16Storage — use the FP-aware upload.
      VBuf maskBuf = s.makeInputBufFP(maskData);
      VBuf cosBuf;
      VBuf sinBuf;
      if(problemUseRope) {
        size_t tableElts =
          (size_t)(problemLearnableRope ? problemNumKVHeads : 1) * (size_t)numPairs * (size_t)problemNumTokens;
        std::vector<float> cosData(tableElts, 1.0f), sinData(tableElts, 0.0f);
        cosBuf = s.makeInputBuf(cosData);
        sinBuf = s.makeInputBuf(sinData);
      }

      AttentionTiled::PC pc = {
        problemNumHeads, problemNumKVHeads, 1.0f / sqrtf((float)problemHeadDim), problemBatchSize * problemNumHeads};
      auto recordOne = [&]() {
        CmdCtx cctx{s.cmd, s.getPushDescFn(), nullptr};
        AttentionTiled::dispatch(
          cctx,
          kernel,
          qBuf.get(),
          kBuf.get(),
          vBuf.get(),
          outBuf.get(),
          maskBuf.get(),
          problemUseRope ? cosBuf.get() : nullptr,
          problemUseRope ? sinBuf.get() : nullptr,
          problemNumTokens,
          pc);
        VulkanHelpers::cmdComputeBarrier(s.cmd, outBuf->buffer);
      };

      s.beginRecording();
      recordOne();
      s.submitAndWait();
      auto [seconds, effectiveIters] = timeBudgetedBench(s, iters, recordOne);
      if(seconds <= 0.0) {
        kernel.destroy(s.device());
        return result;
      }
      result.output.resize(outElts);
      s.downloadFloatsFP(outBuf.get(), result.output, outElts);
      result.kernelsPerSecond = (double)effectiveIters / seconds;
      result.ok = true;
      kernel.destroy(s.device());
      return result;
    } catch(const std::exception& e) {
      if(s.logger != nullptr)
        s.logger->write(std::string("VulkanTuner: attnTiled candidate failed: ") + e.what());
      return result;
    }
  }

}  // namespace VulkanKernels

#endif  // USE_VULKAN_BACKEND

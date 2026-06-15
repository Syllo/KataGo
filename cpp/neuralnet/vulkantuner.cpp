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
#include <memory>
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

  bool isPow2(int32_t v) {
    return v < 0 ? isPow2(-v) : (v > 0 && (v & (v - 1)) == 0);
  }

  string_view getenvStringView(const char* name) {
    const char* value = std::getenv(name);
    return value == nullptr ? string_view() : string_view(value, std::strlen(value));
  }

  constexpr bool tuningUsesNhwc() { return true; }

  struct ScopedTuningLayout {
    explicit ScopedTuningLayout(bool) {}
  };

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

  // Time-budget for all kernel bench runs.
  // Probe with a few iterations to estimate per-dispatch cost, then clamp the
  // timed run so each candidate takes at most BENCH_TARGET_SECONDS wall-clock
  // time. On a fast discrete GPU the ceiling (iters) is reached; on a slow iGPU
  // only a handful of iterations run, keeping auto-tune startup manageable.
  constexpr double BENCH_TARGET_SECONDS = 1.0;
  // Broad multi-shape sweeps multiply the per-benchmark budget by every
  // candidate, layer shape, and tested batch size. Use a shorter, still
  // time-based sample to rank them, then use BENCH_TARGET_SECONDS to confirm
  // the finalists.
  constexpr double BENCH_SCREEN_TARGET_SECONDS = 0.2;
  constexpr double BENCH_WARMUP_SECONDS = 0.050;
  constexpr int BENCH_PROBE_ITERS = 5;
  constexpr int BENCH_WARMUP_MAX_ITERS = 2048;


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
    int budgetIters = (int)(s.benchTargetSeconds / (probeSeconds / BENCH_PROBE_ITERS));
    int effectiveIters = std::clamp(budgetIters, BENCH_PROBE_ITERS, maxIters);
    double seconds = s.timeDispatches(effectiveIters, recordOne);
    return {seconds, effectiveIters};
  }

  class ScopedTuningBenchTarget {
   public:
    ScopedTuningBenchTarget(VulkanTuner::TuningContext& ctx, double target)
      : ctx(ctx), oldTarget(ctx.benchTargetSeconds) {
      ctx.benchTargetSeconds = target;
    }
    ~ScopedTuningBenchTarget() { ctx.benchTargetSeconds = oldTarget; }

   private:
    VulkanTuner::TuningContext& ctx;
    double oldTarget;
  };

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
    if(fp16Storage && cfg.enableWinogradGemmCoopmat1AccF16 != 0)
      return VulkanKernels::WinogradGemmCoopmat1AccF16::layerPaddingContract(cfg);
    if(fp16Storage && cfg.enableWinogradGemmCoopmat1 != 0)
      return VulkanKernels::WinogradGemmCoopmat1::layerPaddingContract(cfg);
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
    } else if(fp16Storage && cfg.enableWinogradGemmCoopmat1AccF16 != 0) {
      layout.packedBM = std::max(1, (int)cfg.coopmat1AccF16BM);
      layout.packedBK = std::max(1, (int)cfg.coopmat1AccF16BK);
      layout.packedAPadWords = WINOGRAD_COOPMAT_PACKED_PAD_WORDS;
    } else if(fp16Storage && cfg.enableWinogradGemmCoopmat1 != 0) {
      layout.packedBM = std::max(1, (int)cfg.coopmat1BM);
      layout.packedBK = std::max(1, (int)cfg.coopmat1BK);
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

bool VulkanTuner::nhwcWinogradTransformsTunedForCurrentLayout(const VulkanTuneParams& config, bool fp16Storage) {
  return config.nhwcWinogradTransformTunedLayout == winogradTransformLayoutSignature(config, fp16Storage);
}

void VulkanTuner::markNhwcWinogradTransformsTunedForCurrentLayout(VulkanTuneParams& config, bool fp16Storage) {
  config.nhwcWinogradTransformTunedLayout = winogradTransformLayoutSignature(config, fp16Storage);
}

bool VulkanTuner::gemmVariantFilterTunedForCurrentHardware(
  const VulkanTuneParams& config,
  const VulkanDeviceInfo& deviceInfo) {
  return config.gemmVariantFilterTunedSignature == deviceInfo.gemmVariantFilterSignature;
}

bool VulkanTuner::gemmVariantTuningCompleteForCurrentHardware(
  const VulkanTuneParams& config,
  const VulkanDeviceInfo& info,
  bool fp16Storage,
  bool fp16Compute,
  int32_t requiredMask,
  bool useNhwc) {
  (void)useNhwc;
  if(!fp16Storage || !fp16Compute)
    return true;
  // Tiles may remain valid across selection-policy changes, but the chosen
  // accelerator/accumulation mode must be recomputed for a new signature.
  if(!gemmVariantFilterTunedForCurrentHardware(config, info))
    return false;
  const bool haveCoop = info.supportsCoopmat1F16 || info.supportsCoopmat1F16AccF16 ||
    info.supportsCoopmat2F16 || info.supportsCoopmat2F16AccF16;
  const bool winogradCoopComplete =
    (!info.supportsCoopmat1F16 ||
     (config.winogradGemmCoopmat1TunerValueValid != 0 && config.winogradGemmCoopmat1F32TimeUs > 0)) &&
    (!info.supportsCoopmat1F16AccF16 ||
     (config.winogradGemmCoopmat1AccF16TunerValueValid != 0 && config.winogradGemmCoopmat1F16TimeUs > 0)) &&
    (!info.supportsCoopmat2F16 ||
     (config.winogradGemmCoopmat2TunerValueValid != 0 && config.winogradGemmCoopmat2F32TimeUs > 0)) &&
    (!info.supportsCoopmat2F16AccF16 ||
     (config.winogradGemmCoopmat2AccF16TunerValueValid != 0 && config.winogradGemmCoopmat2F16TimeUs > 0));
  const bool winogradDot2Complete =
    (!info.supportsDot2F16 ||
     (config.winogradGemmDot2TunerValueValid != 0 && config.winogradGemmDot2F32TimeUs > 0)) &&
    (!info.supportsDot2F16AccF16 ||
     (config.winogradGemmDot2AccF16TunerValueValid != 0 && config.winogradGemmDot2F16TimeUs > 0));
  {
    if((requiredMask & TUNED_NHWC_GEMM) != 0) {
      const bool haveNhwcCoop = info.supportsCoopmat1F16 || info.supportsCoopmat1F16AccF16 ||
        info.supportsCoopmat2F16 || info.supportsCoopmat2F16AccF16;
      if(haveNhwcCoop) {
        if((info.supportsCoopmat1F16 &&
            (config.gemmStridedNhwcCoopmat1TunerValueValid == 0 || config.nhwcGemmCoopmat1F32TimeUs <= 0)) ||
           (info.supportsCoopmat1F16AccF16 &&
            (config.gemmStridedNhwcCoopmat1AccF16TunerValueValid == 0 || config.nhwcGemmCoopmat1F16TimeUs <= 0)) ||
           (info.supportsCoopmat2F16 &&
            (config.gemmStridedNhwcCoopmat2TunerValueValid == 0 || config.nhwcGemmCoopmat2F32TimeUs <= 0)) ||
           (info.supportsCoopmat2F16AccF16 &&
            (config.gemmStridedNhwcCoopmat2AccF16TunerValueValid == 0 || config.nhwcGemmCoopmat2F16TimeUs <= 0)))
          return false;
      } else if(
        (info.supportsDot2F16 &&
         (config.gemmStridedNhwcDot2TunerValueValid == 0 || config.nhwcGemmDot2F32TimeUs <= 0)) ||
        (info.supportsDot2F16AccF16 &&
         (config.gemmStridedNhwcDot2AccF16TunerValueValid == 0 || config.nhwcGemmDot2F16TimeUs <= 0)))
        return false;
    }
    if((requiredMask & TUNED_NHWC_CONV3X3) != 0 &&
       ((info.supportsCoopmat1F16 && config.conv3x3NhwcCoopmat1TunerValueValid == 0) ||
        (info.supportsCoopmat2F16 && config.conv3x3NhwcCoopmat2TunerValueValid == 0) ||
        (info.supportsCoopmat1F16AccF16 && config.conv3x3NhwcCoopmat1AccF16TunerValueValid == 0) ||
        (info.supportsCoopmat2F16AccF16 && config.conv3x3NhwcCoopmat2AccF16TunerValueValid == 0)))
      return false;
    if((requiredMask & TUNED_NHWC_CONV5X5) != 0 &&
       ((info.supportsCoopmat1F16 && config.conv5x5NhwcCoopmat1TunerValueValid == 0) ||
        (info.supportsCoopmat2F16 && config.conv5x5NhwcCoopmat2TunerValueValid == 0) ||
        (info.supportsCoopmat1F16AccF16 && config.conv5x5NhwcCoopmat1AccF16TunerValueValid == 0) ||
        (info.supportsCoopmat2F16AccF16 && config.conv5x5NhwcCoopmat2AccF16TunerValueValid == 0)))
      return false;
    if((requiredMask & TUNED_NHWC_WINOGRAD) != 0 &&
       (haveCoop ? !winogradCoopComplete : !winogradDot2Complete))
      return false;
    return true;
  }
}

// ============================================================================
// VulkanTuneParams persistence
// ============================================================================

bool VulkanTuneParams::operator==(const VulkanTuneParams& other) const {
  return
    nhwcWinogradTransformTunedLayout == other.nhwcWinogradTransformTunedLayout &&
    gemmVariantFilterTunedSignature == other.gemmVariantFilterTunedSignature &&
    tunedKernelMask == other.tunedKernelMask &&
    winogradGemmCoopmat1F32TimeUs == other.winogradGemmCoopmat1F32TimeUs &&
    winogradGemmCoopmat2F32TimeUs == other.winogradGemmCoopmat2F32TimeUs &&
    winogradGemmCoopmat1F16TimeUs == other.winogradGemmCoopmat1F16TimeUs &&
    winogradGemmCoopmat2F16TimeUs == other.winogradGemmCoopmat2F16TimeUs &&
    winogradGemmDot2F32TimeUs == other.winogradGemmDot2F32TimeUs &&
    winogradGemmDot2F16TimeUs == other.winogradGemmDot2F16TimeUs &&
    nchwToNhwcSmallTile == other.nchwToNhwcSmallTile &&
    nchwToNhwcLargeTile == other.nchwToNhwcLargeTile &&
    nchwToNhwcTileCrossover == other.nchwToNhwcTileCrossover &&
    nhwcWinogradTransformLocalSizeX == other.nhwcWinogradTransformLocalSizeX &&
    nhwcWinogradTransformLocalSizeY == other.nhwcWinogradTransformLocalSizeY &&
    nhwcWinogradUntransformLocalSizeX == other.nhwcWinogradUntransformLocalSizeX &&
    nhwcWinogradUntransformLocalSizeY == other.nhwcWinogradUntransformLocalSizeY &&
    nhwcWinograd3x3OutTile == other.nhwcWinograd3x3OutTile &&
    conv3x3NhwcWinogradTunerValueValid == other.conv3x3NhwcWinogradTunerValueValid &&
    conv5x5NhwcWinogradTunerValueValid == other.conv5x5NhwcWinogradTunerValueValid &&
    gemmStridedTiledNhwcLocalSizeX == other.gemmStridedTiledNhwcLocalSizeX &&
    gemmStridedTiledNhwcLocalSizeY == other.gemmStridedTiledNhwcLocalSizeY &&
    gemmStridedTiledNhwcTileK == other.gemmStridedTiledNhwcTileK &&
    gemmStridedTiledNhwcRN == other.gemmStridedTiledNhwcRN &&
    gemmDirectLocalSizeX == other.gemmDirectLocalSizeX &&
    gemmDirectLocalSizeY == other.gemmDirectLocalSizeY &&
    winogradGemmM == other.winogradGemmM &&
    winogradGemmN == other.winogradGemmN &&
    winogradGemmK == other.winogradGemmK &&
    winogradGemmRN == other.winogradGemmRN &&
    enableWinogradGemmDot2 == other.enableWinogradGemmDot2 &&
    enableWinogradGemmDot2AccF16 == other.enableWinogradGemmDot2AccF16 &&
    winogradGemmDot2TunerValueValid == other.winogradGemmDot2TunerValueValid &&
    winogradGemmDot2AccF16TunerValueValid == other.winogradGemmDot2AccF16TunerValueValid &&
    dot2BlockSize == other.dot2BlockSize &&
    dot2BM == other.dot2BM &&
    dot2BN == other.dot2BN &&
    dot2WM == other.dot2WM &&
    dot2WN == other.dot2WN &&
    dot2WMIter == other.dot2WMIter &&
    dot2TM == other.dot2TM &&
    dot2TN == other.dot2TN &&
    dot2Warp == other.dot2Warp &&
    dot2AccF16BlockSize == other.dot2AccF16BlockSize &&
    dot2AccF16BM == other.dot2AccF16BM &&
    dot2AccF16BN == other.dot2AccF16BN &&
    dot2AccF16WM == other.dot2AccF16WM &&
    dot2AccF16WN == other.dot2AccF16WN &&
    dot2AccF16WMIter == other.dot2AccF16WMIter &&
    dot2AccF16TM == other.dot2AccF16TM &&
    dot2AccF16TN == other.dot2AccF16TN &&
    dot2AccF16Warp == other.dot2AccF16Warp &&
    enableWinogradGemmCoopmat1 == other.enableWinogradGemmCoopmat1 &&
    winogradGemmCoopmat1TunerValueValid == other.winogradGemmCoopmat1TunerValueValid &&
    coopmat1BlockSize == other.coopmat1BlockSize &&
    coopmat1BM == other.coopmat1BM &&
    coopmat1BN == other.coopmat1BN &&
    coopmat1BK == other.coopmat1BK &&
    coopmat1WM == other.coopmat1WM &&
    coopmat1WN == other.coopmat1WN &&
    coopmat1TM == other.coopmat1TM &&
    coopmat1TN == other.coopmat1TN &&
    coopmat1TK == other.coopmat1TK &&
    coopmat1Warp == other.coopmat1Warp &&
    conv3x3NhwcCoopmat1TunerValueValid == other.conv3x3NhwcCoopmat1TunerValueValid &&
    conv3x3NhwcCoopmat1BlockSize == other.conv3x3NhwcCoopmat1BlockSize &&
    conv3x3NhwcCoopmat1BM == other.conv3x3NhwcCoopmat1BM &&
    conv3x3NhwcCoopmat1BN == other.conv3x3NhwcCoopmat1BN &&
    conv3x3NhwcCoopmat1BK == other.conv3x3NhwcCoopmat1BK &&
    conv3x3NhwcCoopmat1WM == other.conv3x3NhwcCoopmat1WM &&
    conv3x3NhwcCoopmat1WN == other.conv3x3NhwcCoopmat1WN &&
    conv3x3NhwcCoopmat1TM == other.conv3x3NhwcCoopmat1TM &&
    conv3x3NhwcCoopmat1TN == other.conv3x3NhwcCoopmat1TN &&
    conv3x3NhwcCoopmat1TK == other.conv3x3NhwcCoopmat1TK &&
    conv3x3NhwcCoopmat1Warp == other.conv3x3NhwcCoopmat1Warp &&
    conv5x5NhwcCoopmat1TunerValueValid == other.conv5x5NhwcCoopmat1TunerValueValid &&
    conv5x5NhwcCoopmat1BlockSize == other.conv5x5NhwcCoopmat1BlockSize &&
    conv5x5NhwcCoopmat1BM == other.conv5x5NhwcCoopmat1BM &&
    conv5x5NhwcCoopmat1BN == other.conv5x5NhwcCoopmat1BN &&
    conv5x5NhwcCoopmat1BK == other.conv5x5NhwcCoopmat1BK &&
    conv5x5NhwcCoopmat1WM == other.conv5x5NhwcCoopmat1WM &&
    conv5x5NhwcCoopmat1WN == other.conv5x5NhwcCoopmat1WN &&
    conv5x5NhwcCoopmat1TM == other.conv5x5NhwcCoopmat1TM &&
    conv5x5NhwcCoopmat1TN == other.conv5x5NhwcCoopmat1TN &&
    conv5x5NhwcCoopmat1TK == other.conv5x5NhwcCoopmat1TK &&
    conv5x5NhwcCoopmat1Warp == other.conv5x5NhwcCoopmat1Warp &&
    conv3x3NhwcCoopmat1AccF16TunerValueValid == other.conv3x3NhwcCoopmat1AccF16TunerValueValid &&
    conv3x3NhwcCoopmat1AccF16BlockSize == other.conv3x3NhwcCoopmat1AccF16BlockSize &&
    conv3x3NhwcCoopmat1AccF16BM == other.conv3x3NhwcCoopmat1AccF16BM &&
    conv3x3NhwcCoopmat1AccF16BN == other.conv3x3NhwcCoopmat1AccF16BN &&
    conv3x3NhwcCoopmat1AccF16BK == other.conv3x3NhwcCoopmat1AccF16BK &&
    conv3x3NhwcCoopmat1AccF16WM == other.conv3x3NhwcCoopmat1AccF16WM &&
    conv3x3NhwcCoopmat1AccF16WN == other.conv3x3NhwcCoopmat1AccF16WN &&
    conv3x3NhwcCoopmat1AccF16TM == other.conv3x3NhwcCoopmat1AccF16TM &&
    conv3x3NhwcCoopmat1AccF16TN == other.conv3x3NhwcCoopmat1AccF16TN &&
    conv3x3NhwcCoopmat1AccF16TK == other.conv3x3NhwcCoopmat1AccF16TK &&
    conv3x3NhwcCoopmat1AccF16Warp == other.conv3x3NhwcCoopmat1AccF16Warp &&
    conv5x5NhwcCoopmat1AccF16TunerValueValid == other.conv5x5NhwcCoopmat1AccF16TunerValueValid &&
    conv5x5NhwcCoopmat1AccF16BlockSize == other.conv5x5NhwcCoopmat1AccF16BlockSize &&
    conv5x5NhwcCoopmat1AccF16BM == other.conv5x5NhwcCoopmat1AccF16BM &&
    conv5x5NhwcCoopmat1AccF16BN == other.conv5x5NhwcCoopmat1AccF16BN &&
    conv5x5NhwcCoopmat1AccF16BK == other.conv5x5NhwcCoopmat1AccF16BK &&
    conv5x5NhwcCoopmat1AccF16WM == other.conv5x5NhwcCoopmat1AccF16WM &&
    conv5x5NhwcCoopmat1AccF16WN == other.conv5x5NhwcCoopmat1AccF16WN &&
    conv5x5NhwcCoopmat1AccF16TM == other.conv5x5NhwcCoopmat1AccF16TM &&
    conv5x5NhwcCoopmat1AccF16TN == other.conv5x5NhwcCoopmat1AccF16TN &&
    conv5x5NhwcCoopmat1AccF16TK == other.conv5x5NhwcCoopmat1AccF16TK &&
    conv5x5NhwcCoopmat1AccF16Warp == other.conv5x5NhwcCoopmat1AccF16Warp &&
    enableWinogradGemmCoopmat1AccF16 == other.enableWinogradGemmCoopmat1AccF16 &&
    winogradGemmCoopmat1AccF16TunerValueValid == other.winogradGemmCoopmat1AccF16TunerValueValid &&
    coopmat1AccF16BlockSize == other.coopmat1AccF16BlockSize &&
    coopmat1AccF16BM == other.coopmat1AccF16BM &&
    coopmat1AccF16BN == other.coopmat1AccF16BN &&
    coopmat1AccF16BK == other.coopmat1AccF16BK &&
    coopmat1AccF16WM == other.coopmat1AccF16WM &&
    coopmat1AccF16WN == other.coopmat1AccF16WN &&
    coopmat1AccF16TM == other.coopmat1AccF16TM &&
    coopmat1AccF16TN == other.coopmat1AccF16TN &&
    coopmat1AccF16TK == other.coopmat1AccF16TK &&
    coopmat1AccF16Warp == other.coopmat1AccF16Warp &&
    gemmStridedNhwcCoopmat1TunerValueValid == other.gemmStridedNhwcCoopmat1TunerValueValid &&
    nhwcStridedCoopmat1BlockSize == other.nhwcStridedCoopmat1BlockSize &&
    nhwcStridedCoopmat1BM == other.nhwcStridedCoopmat1BM &&
    nhwcStridedCoopmat1BN == other.nhwcStridedCoopmat1BN &&
    nhwcStridedCoopmat1BK == other.nhwcStridedCoopmat1BK &&
    nhwcStridedCoopmat1WM == other.nhwcStridedCoopmat1WM &&
    nhwcStridedCoopmat1WN == other.nhwcStridedCoopmat1WN &&
    nhwcStridedCoopmat1TM == other.nhwcStridedCoopmat1TM &&
    nhwcStridedCoopmat1TN == other.nhwcStridedCoopmat1TN &&
    nhwcStridedCoopmat1TK == other.nhwcStridedCoopmat1TK &&
    nhwcStridedCoopmat1Warp == other.nhwcStridedCoopmat1Warp &&
    gemmStridedNhwcCoopmat1AccF16TunerValueValid == other.gemmStridedNhwcCoopmat1AccF16TunerValueValid &&
    nhwcStridedCoopmat1AccF16BlockSize == other.nhwcStridedCoopmat1AccF16BlockSize &&
    nhwcStridedCoopmat1AccF16BM == other.nhwcStridedCoopmat1AccF16BM &&
    nhwcStridedCoopmat1AccF16BN == other.nhwcStridedCoopmat1AccF16BN &&
    nhwcStridedCoopmat1AccF16BK == other.nhwcStridedCoopmat1AccF16BK &&
    nhwcStridedCoopmat1AccF16WM == other.nhwcStridedCoopmat1AccF16WM &&
    nhwcStridedCoopmat1AccF16WN == other.nhwcStridedCoopmat1AccF16WN &&
    nhwcStridedCoopmat1AccF16TM == other.nhwcStridedCoopmat1AccF16TM &&
    nhwcStridedCoopmat1AccF16TN == other.nhwcStridedCoopmat1AccF16TN &&
    nhwcStridedCoopmat1AccF16TK == other.nhwcStridedCoopmat1AccF16TK &&
    nhwcStridedCoopmat1AccF16Warp == other.nhwcStridedCoopmat1AccF16Warp &&
    gemmStridedNhwcCoopmat2TunerValueValid == other.gemmStridedNhwcCoopmat2TunerValueValid &&
    nhwcStridedCoopmat2BlockSize == other.nhwcStridedCoopmat2BlockSize &&
    nhwcStridedCoopmat2BM == other.nhwcStridedCoopmat2BM &&
    nhwcStridedCoopmat2BN == other.nhwcStridedCoopmat2BN &&
    nhwcStridedCoopmat2BK == other.nhwcStridedCoopmat2BK &&
    gemmStridedNhwcCoopmat2AccF16TunerValueValid == other.gemmStridedNhwcCoopmat2AccF16TunerValueValid &&
    nhwcStridedCoopmat2AccF16BlockSize == other.nhwcStridedCoopmat2AccF16BlockSize &&
    nhwcStridedCoopmat2AccF16BM == other.nhwcStridedCoopmat2AccF16BM &&
    nhwcStridedCoopmat2AccF16BN == other.nhwcStridedCoopmat2AccF16BN &&
    nhwcStridedCoopmat2AccF16BK == other.nhwcStridedCoopmat2AccF16BK &&
    nhwcGemmF32UseCoopmat2 == other.nhwcGemmF32UseCoopmat2 &&
    nhwcGemmUseCoopmat2 == other.nhwcGemmUseCoopmat2 &&
    nhwcGemmUseCoopmatAccF16 == other.nhwcGemmUseCoopmatAccF16 &&
    nhwcGemmCoopmat1F32TimeUs == other.nhwcGemmCoopmat1F32TimeUs &&
    nhwcGemmCoopmat2F32TimeUs == other.nhwcGemmCoopmat2F32TimeUs &&
    nhwcGemmCoopmat1F16TimeUs == other.nhwcGemmCoopmat1F16TimeUs &&
    nhwcGemmCoopmat2F16TimeUs == other.nhwcGemmCoopmat2F16TimeUs &&
    gemmStridedNhwcDot2TunerValueValid == other.gemmStridedNhwcDot2TunerValueValid &&
    nhwcStridedDot2BlockSize == other.nhwcStridedDot2BlockSize &&
    nhwcStridedDot2BM == other.nhwcStridedDot2BM &&
    nhwcStridedDot2BN == other.nhwcStridedDot2BN &&
    nhwcStridedDot2WM == other.nhwcStridedDot2WM &&
    nhwcStridedDot2WN == other.nhwcStridedDot2WN &&
    nhwcStridedDot2WMIter == other.nhwcStridedDot2WMIter &&
    nhwcStridedDot2TM == other.nhwcStridedDot2TM &&
    nhwcStridedDot2TN == other.nhwcStridedDot2TN &&
    nhwcStridedDot2Warp == other.nhwcStridedDot2Warp &&
    nhwcGemmUseDot2 == other.nhwcGemmUseDot2 &&
    gemmStridedNhwcDot2AccF16TunerValueValid == other.gemmStridedNhwcDot2AccF16TunerValueValid &&
    nhwcStridedDot2AccF16BlockSize == other.nhwcStridedDot2AccF16BlockSize &&
    nhwcStridedDot2AccF16BM == other.nhwcStridedDot2AccF16BM &&
    nhwcStridedDot2AccF16BN == other.nhwcStridedDot2AccF16BN &&
    nhwcStridedDot2AccF16WM == other.nhwcStridedDot2AccF16WM &&
    nhwcStridedDot2AccF16WN == other.nhwcStridedDot2AccF16WN &&
    nhwcStridedDot2AccF16WMIter == other.nhwcStridedDot2AccF16WMIter &&
    nhwcStridedDot2AccF16TM == other.nhwcStridedDot2AccF16TM &&
    nhwcStridedDot2AccF16TN == other.nhwcStridedDot2AccF16TN &&
    nhwcStridedDot2AccF16Warp == other.nhwcStridedDot2AccF16Warp &&
    nhwcGemmUseDot2AccF16 == other.nhwcGemmUseDot2AccF16 &&
    nhwcGemmDot2F32TimeUs == other.nhwcGemmDot2F32TimeUs &&
    nhwcGemmDot2F16TimeUs == other.nhwcGemmDot2F16TimeUs &&
    conv3x3NhwcCoopmat2TunerValueValid == other.conv3x3NhwcCoopmat2TunerValueValid &&
    conv3x3NhwcCoopmat2BlockSize == other.conv3x3NhwcCoopmat2BlockSize &&
    conv3x3NhwcCoopmat2BM == other.conv3x3NhwcCoopmat2BM &&
    conv3x3NhwcCoopmat2BN == other.conv3x3NhwcCoopmat2BN &&
    conv3x3NhwcCoopmat2BK == other.conv3x3NhwcCoopmat2BK &&
    nhwcConv3x3UseCoopmat2 == other.nhwcConv3x3UseCoopmat2 &&
    conv5x5NhwcCoopmat2TunerValueValid == other.conv5x5NhwcCoopmat2TunerValueValid &&
    conv5x5NhwcCoopmat2BlockSize == other.conv5x5NhwcCoopmat2BlockSize &&
    conv5x5NhwcCoopmat2BM == other.conv5x5NhwcCoopmat2BM &&
    conv5x5NhwcCoopmat2BN == other.conv5x5NhwcCoopmat2BN &&
    conv5x5NhwcCoopmat2BK == other.conv5x5NhwcCoopmat2BK &&
    nhwcConv5x5UseCoopmat2 == other.nhwcConv5x5UseCoopmat2 &&
    conv3x3NhwcCoopmat2AccF16TunerValueValid == other.conv3x3NhwcCoopmat2AccF16TunerValueValid &&
    conv3x3NhwcCoopmat2AccF16BlockSize == other.conv3x3NhwcCoopmat2AccF16BlockSize &&
    conv3x3NhwcCoopmat2AccF16BM == other.conv3x3NhwcCoopmat2AccF16BM &&
    conv3x3NhwcCoopmat2AccF16BN == other.conv3x3NhwcCoopmat2AccF16BN &&
    conv3x3NhwcCoopmat2AccF16BK == other.conv3x3NhwcCoopmat2AccF16BK &&
    conv5x5NhwcCoopmat2AccF16TunerValueValid == other.conv5x5NhwcCoopmat2AccF16TunerValueValid &&
    conv5x5NhwcCoopmat2AccF16BlockSize == other.conv5x5NhwcCoopmat2AccF16BlockSize &&
    conv5x5NhwcCoopmat2AccF16BM == other.conv5x5NhwcCoopmat2AccF16BM &&
    conv5x5NhwcCoopmat2AccF16BN == other.conv5x5NhwcCoopmat2AccF16BN &&
    conv5x5NhwcCoopmat2AccF16BK == other.conv5x5NhwcCoopmat2AccF16BK &&
    enableWinogradGemmCoopmat2 == other.enableWinogradGemmCoopmat2 &&
    winogradGemmCoopmat2TunerValueValid == other.winogradGemmCoopmat2TunerValueValid &&
    coopmat2BlockSize == other.coopmat2BlockSize &&
    coopmat2BM == other.coopmat2BM &&
    coopmat2BN == other.coopmat2BN &&
    coopmat2BK == other.coopmat2BK &&
    enableWinogradGemmCoopmat2AccF16 == other.enableWinogradGemmCoopmat2AccF16 &&
    winogradGemmCoopmat2AccF16TunerValueValid == other.winogradGemmCoopmat2AccF16TunerValueValid &&
    coopmat2AccF16BlockSize == other.coopmat2AccF16BlockSize &&
    coopmat2AccF16BM == other.coopmat2AccF16BM &&
    coopmat2AccF16BN == other.coopmat2AccF16BN &&
    coopmat2AccF16BK == other.coopmat2AccF16BK &&
    gpoolNhwcXystride == other.gpoolNhwcXystride &&
    valueHeadPoolNhwcXystride == other.valueHeadPoolNhwcXystride &&
    spatialRMSNormNhwcTile == other.spatialRMSNormNhwcTile &&
    swiGLUTunerValueValid == other.swiGLUTunerValueValid &&
    swiGLULocalSizeX == other.swiGLULocalSizeX &&
    attnBlockQ == other.attnBlockQ &&
    attnBlockKV == other.attnBlockKV &&
    attnQPerThread == other.attnQPerThread &&
    attnNhwcBlockQ == other.attnNhwcBlockQ &&
    attnNhwcBlockKV == other.attnNhwcBlockKV &&
    attnNhwcQPerThread == other.attnNhwcQPerThread &&
    attnNhwcVariantPolicyVersion == other.attnNhwcVariantPolicyVersion &&
    attnNhwcCoopmat1TunerValueValid == other.attnNhwcCoopmat1TunerValueValid &&
    attnNhwcUseCoopmat1 == other.attnNhwcUseCoopmat1 &&
    attnNhwcTiledTimeUs == other.attnNhwcTiledTimeUs &&
    attnNhwcCoopmat1TimeUs == other.attnNhwcCoopmat1TimeUs &&
    attnNhwcCoopmat1BlockSize == other.attnNhwcCoopmat1BlockSize &&
    attnNhwcCoopmat1BlockQ == other.attnNhwcCoopmat1BlockQ &&
    attnNhwcCoopmat1BlockKV == other.attnNhwcCoopmat1BlockKV &&
    attnNhwcCoopmat1TM == other.attnNhwcCoopmat1TM &&
    attnNhwcCoopmat1TN == other.attnNhwcCoopmat1TN &&
    attnNhwcCoopmat1TK == other.attnNhwcCoopmat1TK &&
    attnNhwcCoopmat1Warp == other.attnNhwcCoopmat1Warp &&
    attnNhwcCoopmat1DirectKV == other.attnNhwcCoopmat1DirectKV &&
    attnNhwcCoopmat1KVChunkCount == other.attnNhwcCoopmat1KVChunkCount &&
    attnNhwcCoopmat1SplitKTunerValueValid == other.attnNhwcCoopmat1SplitKTunerValueValid &&
    attnNhwcCoopmat1SplitKBlockSize == other.attnNhwcCoopmat1SplitKBlockSize &&
    attnNhwcCoopmat1SplitKBlockQ == other.attnNhwcCoopmat1SplitKBlockQ &&
    attnNhwcCoopmat1SplitKBlockKV == other.attnNhwcCoopmat1SplitKBlockKV &&
    attnNhwcCoopmat1SplitKTM == other.attnNhwcCoopmat1SplitKTM &&
    attnNhwcCoopmat1SplitKTN == other.attnNhwcCoopmat1SplitKTN &&
    attnNhwcCoopmat1SplitKTK == other.attnNhwcCoopmat1SplitKTK &&
    attnNhwcCoopmat1SplitKWarp == other.attnNhwcCoopmat1SplitKWarp &&
    attnNhwcCoopmat1SplitKDirectKV == other.attnNhwcCoopmat1SplitKDirectKV &&
    attnNhwcCoopmat1SplitKKVChunkCount == other.attnNhwcCoopmat1SplitKKVChunkCount &&
    attnNhwcCoopmat1SplitKCutoffBatch == other.attnNhwcCoopmat1SplitKCutoffBatch &&
    attnNhwcCoopmatMaintenance1TunerValueValid == other.attnNhwcCoopmatMaintenance1TunerValueValid &&
    attnNhwcUseCoopmatMaintenance1 == other.attnNhwcUseCoopmatMaintenance1 &&
    attnNhwcCoopmatMaintenance1TimeUs == other.attnNhwcCoopmatMaintenance1TimeUs &&
    attnNhwcCoopmatMaintenance1BlockSize == other.attnNhwcCoopmatMaintenance1BlockSize &&
    attnNhwcCoopmatMaintenance1BlockQ == other.attnNhwcCoopmatMaintenance1BlockQ &&
    attnNhwcCoopmatMaintenance1BlockKV == other.attnNhwcCoopmatMaintenance1BlockKV &&
    attnNhwcCoopmatMaintenance1TM == other.attnNhwcCoopmatMaintenance1TM &&
    attnNhwcCoopmatMaintenance1TN == other.attnNhwcCoopmatMaintenance1TN &&
    attnNhwcCoopmatMaintenance1TK == other.attnNhwcCoopmatMaintenance1TK &&
    attnNhwcCoopmatMaintenance1Warp == other.attnNhwcCoopmatMaintenance1Warp &&
    attnNhwcCoopmatMaintenance1DirectKV == other.attnNhwcCoopmatMaintenance1DirectKV &&
    attnNhwcCoopmatMaintenance1SplitKTunerValueValid == other.attnNhwcCoopmatMaintenance1SplitKTunerValueValid &&
    attnNhwcCoopmatMaintenance1SplitKBlockSize == other.attnNhwcCoopmatMaintenance1SplitKBlockSize &&
    attnNhwcCoopmatMaintenance1SplitKBlockQ == other.attnNhwcCoopmatMaintenance1SplitKBlockQ &&
    attnNhwcCoopmatMaintenance1SplitKBlockKV == other.attnNhwcCoopmatMaintenance1SplitKBlockKV &&
    attnNhwcCoopmatMaintenance1SplitKTM == other.attnNhwcCoopmatMaintenance1SplitKTM &&
    attnNhwcCoopmatMaintenance1SplitKTN == other.attnNhwcCoopmatMaintenance1SplitKTN &&
    attnNhwcCoopmatMaintenance1SplitKTK == other.attnNhwcCoopmatMaintenance1SplitKTK &&
    attnNhwcCoopmatMaintenance1SplitKWarp == other.attnNhwcCoopmatMaintenance1SplitKWarp &&
    attnNhwcCoopmatMaintenance1SplitKDirectKV == other.attnNhwcCoopmatMaintenance1SplitKDirectKV &&
    attnNhwcCoopmatMaintenance1SplitKKVChunkCount == other.attnNhwcCoopmatMaintenance1SplitKKVChunkCount &&
    attnNhwcCoopmatMaintenance1SplitKCutoffBatch == other.attnNhwcCoopmatMaintenance1SplitKCutoffBatch &&
    attnNhwcCoopmat2TunerValueValid == other.attnNhwcCoopmat2TunerValueValid &&
    attnNhwcUseCoopmat2 == other.attnNhwcUseCoopmat2 &&
    attnNhwcCoopmat2TimeUs == other.attnNhwcCoopmat2TimeUs &&
    attnNhwcCoopmat2BlockSize == other.attnNhwcCoopmat2BlockSize &&
    attnNhwcCoopmat2BlockQ == other.attnNhwcCoopmat2BlockQ &&
    attnNhwcCoopmat2BlockKV == other.attnNhwcCoopmat2BlockKV &&
    attnNhwcDot2TunerValueValid == other.attnNhwcDot2TunerValueValid &&
    attnNhwcUseDot2 == other.attnNhwcUseDot2 &&
    attnNhwcDot2TimeUs == other.attnNhwcDot2TimeUs &&
    attnNhwcDot2BlockSize == other.attnNhwcDot2BlockSize &&
    attnNhwcDot2BlockQ == other.attnNhwcDot2BlockQ &&
    attnNhwcDot2BlockKV == other.attnNhwcDot2BlockKV;
}
bool VulkanTuneParams::isValid() const {
  return VulkanKernels::LayoutTransform::isConfigSupported(
           nchwToNhwcSmallTile, nchwToNhwcLargeTile, nchwToNhwcTileCrossover)
         && VulkanKernels::WinogradGemm::isConfigSupported(winogradGemmM, winogradGemmN, winogradGemmK, winogradGemmRN)
         && VulkanKernels::GemmStridedTiledNhwc::isConfigSupported(
           gemmStridedTiledNhwcLocalSizeX, gemmStridedTiledNhwcLocalSizeY, gemmStridedTiledNhwcTileK, gemmStridedTiledNhwcRN)
         && winogradGemmCoopmat1F32TimeUs >= 0
         && winogradGemmCoopmat2F32TimeUs >= 0
         && winogradGemmCoopmat1F16TimeUs >= 0
         && winogradGemmCoopmat2F16TimeUs >= 0
         && winogradGemmDot2F32TimeUs >= 0
         && winogradGemmDot2F16TimeUs >= 0
         && (nhwcWinograd3x3OutTile == 2 || nhwcWinograd3x3OutTile == 4)
         && (conv3x3NhwcWinogradTunerValueValid == 0 || conv3x3NhwcWinogradTunerValueValid == 1)
         && (conv5x5NhwcWinogradTunerValueValid == 0 || conv5x5NhwcWinogradTunerValueValid == 1)
         && VulkanKernels::WinogradTransformNhwc::isConfigSupported(
           nhwcWinogradTransformLocalSizeX, nhwcWinogradTransformLocalSizeY)
         && VulkanKernels::WinogradUntransformNhwc::isConfigSupported(
           nhwcWinogradUntransformLocalSizeX, nhwcWinogradUntransformLocalSizeY)
         && VulkanKernels::GemmDirectFP32::isConfigSupported(gemmDirectLocalSizeX, gemmDirectLocalSizeY)
         && VulkanKernels::AttentionTiled::isConfigSupported(attnBlockQ, attnBlockKV, attnQPerThread)
         && VulkanKernels::AttentionTiled::isConfigSupported(
           attnNhwcBlockQ, attnNhwcBlockKV, attnNhwcQPerThread)
         && attnNhwcVariantPolicyVersion == 4
         && (attnNhwcCoopmat1TunerValueValid == 0 || attnNhwcCoopmat1TunerValueValid == 1)
         && (attnNhwcUseCoopmat1 == 0 || attnNhwcUseCoopmat1 == 1)
         && (attnNhwcCoopmatMaintenance1TunerValueValid == 0 || attnNhwcCoopmatMaintenance1TunerValueValid == 1)
         && (attnNhwcUseCoopmatMaintenance1 == 0 || attnNhwcUseCoopmatMaintenance1 == 1)
         && (attnNhwcCoopmatMaintenance1DirectKV == 0 || attnNhwcCoopmatMaintenance1DirectKV == 1)
         && (attnNhwcCoopmatMaintenance1SplitKTunerValueValid == 0 || attnNhwcCoopmatMaintenance1SplitKTunerValueValid == 1)
         && (attnNhwcCoopmatMaintenance1SplitKDirectKV == 0 || attnNhwcCoopmatMaintenance1SplitKDirectKV == 1)
         && attnNhwcCoopmatMaintenance1TimeUs >= 0
         && attnNhwcCoopmatMaintenance1SplitKKVChunkCount >= 1
         && attnNhwcCoopmatMaintenance1SplitKKVChunkCount <= 8
         && attnNhwcCoopmatMaintenance1SplitKCutoffBatch >= 0
         && attnNhwcCoopmatMaintenance1TN == attnNhwcCoopmatMaintenance1TK
         && attnNhwcCoopmatMaintenance1SplitKTN == attnNhwcCoopmatMaintenance1SplitKTK
         && VulkanKernels::AttentionCoopmat1Nhwc::isConfigSupported(
           attnNhwcCoopmatMaintenance1BlockSize, attnNhwcCoopmatMaintenance1BlockQ,
           attnNhwcCoopmatMaintenance1BlockKV, attnNhwcCoopmatMaintenance1TM,
           attnNhwcCoopmatMaintenance1TN, attnNhwcCoopmatMaintenance1TK,
           attnNhwcCoopmatMaintenance1Warp, 64, 64)
         && VulkanKernels::AttentionCoopmat1Nhwc::isConfigSupported(
           attnNhwcCoopmatMaintenance1SplitKBlockSize, attnNhwcCoopmatMaintenance1SplitKBlockQ,
           attnNhwcCoopmatMaintenance1SplitKBlockKV, attnNhwcCoopmatMaintenance1SplitKTM,
           attnNhwcCoopmatMaintenance1SplitKTN, attnNhwcCoopmatMaintenance1SplitKTK,
           attnNhwcCoopmatMaintenance1SplitKWarp, 64, 64)
         && (attnNhwcCoopmat1DirectKV == 0 || attnNhwcCoopmat1DirectKV == 1)
         && attnNhwcCoopmat1KVChunkCount >= 1
         && attnNhwcCoopmat1KVChunkCount <= 8
         && (attnNhwcCoopmat1SplitKTunerValueValid == 0 || attnNhwcCoopmat1SplitKTunerValueValid == 1)
         && (attnNhwcCoopmat1SplitKDirectKV == 0 || attnNhwcCoopmat1SplitKDirectKV == 1)
         && attnNhwcCoopmat1SplitKKVChunkCount >= 1
         && attnNhwcCoopmat1SplitKKVChunkCount <= 8
         && attnNhwcCoopmat1SplitKCutoffBatch >= 0
         && VulkanKernels::AttentionCoopmat1Nhwc::isConfigSupported(
           attnNhwcCoopmat1SplitKBlockSize,
           attnNhwcCoopmat1SplitKBlockQ,
           attnNhwcCoopmat1SplitKBlockKV,
           attnNhwcCoopmat1SplitKTM,
           attnNhwcCoopmat1SplitKTN,
           attnNhwcCoopmat1SplitKTK,
           attnNhwcCoopmat1SplitKWarp,
           64,
           64)
         && attnNhwcTiledTimeUs >= 0
         && attnNhwcCoopmat1TimeUs >= 0
         && VulkanKernels::AttentionCoopmat1Nhwc::isConfigSupported(
           attnNhwcCoopmat1BlockSize,
           attnNhwcCoopmat1BlockQ,
           attnNhwcCoopmat1BlockKV,
           attnNhwcCoopmat1TM,
           attnNhwcCoopmat1TN,
           attnNhwcCoopmat1TK,
           attnNhwcCoopmat1Warp,
           64,
           64)
         && (attnNhwcCoopmat2TunerValueValid == 0 || attnNhwcCoopmat2TunerValueValid == 1)
         && (attnNhwcUseCoopmat2 == 0 || attnNhwcUseCoopmat2 == 1)
         && attnNhwcCoopmat2TimeUs >= 0
         && VulkanKernels::AttentionCoopmat2AccF32Nhwc::isConfigSupported(
           attnNhwcCoopmat2BlockSize, attnNhwcCoopmat2BlockQ, attnNhwcCoopmat2BlockKV, 64, 64)
         && (attnNhwcDot2TunerValueValid == 0 || attnNhwcDot2TunerValueValid == 1)
         && (attnNhwcUseDot2 == 0 || attnNhwcUseDot2 == 1)
         && attnNhwcUseCoopmat1 + attnNhwcUseCoopmatMaintenance1 + attnNhwcUseCoopmat2 + attnNhwcUseDot2 <= 1
         && attnNhwcDot2TimeUs >= 0
         && VulkanKernels::AttentionDot2AccF32Nhwc::isConfigSupported(
           attnNhwcDot2BlockSize,
           attnNhwcDot2BlockQ,
           attnNhwcDot2BlockKV,
           64,
           64)
         && isPow2(gpoolNhwcXystride)
         && isPow2(valueHeadPoolNhwcXystride)
         && VulkanKernels::SpatialRMSNormNhwc::isConfigSupported(spatialRMSNormNhwcTile)
         && (swiGLUTunerValueValid == 0 || swiGLUTunerValueValid == 1)
         && VulkanKernels::SwiGLU::isConfigSupported(swiGLULocalSizeX)
         && (enableWinogradGemmDot2 == 0 || enableWinogradGemmDot2 == 1)
         && (enableWinogradGemmDot2AccF16 == 0 || enableWinogradGemmDot2AccF16 == 1)
         && (winogradGemmDot2TunerValueValid == 0 || winogradGemmDot2TunerValueValid == 1)
         && (winogradGemmDot2AccF16TunerValueValid == 0 || winogradGemmDot2AccF16TunerValueValid == 1)
         && VulkanKernels::WinogradGemmDot2::isConfigSupported(
           dot2BlockSize, dot2BM, dot2BN, dot2WM, dot2WN, dot2WMIter, dot2TM, dot2TN, dot2Warp)
         && VulkanKernels::WinogradGemmDot2AccF16::isConfigSupported(
           dot2AccF16BlockSize,
           dot2AccF16BM,
           dot2AccF16BN,
           dot2AccF16WM,
           dot2AccF16WN,
           dot2AccF16WMIter,
           dot2AccF16TM,
           dot2AccF16TN,
           dot2AccF16Warp)
         && (enableWinogradGemmCoopmat1 == 0 || enableWinogradGemmCoopmat1 == 1)
         && (winogradGemmCoopmat1TunerValueValid == 0 || winogradGemmCoopmat1TunerValueValid == 1)
         && VulkanKernels::WinogradGemmCoopmat1::isConfigSupported(
           coopmat1BlockSize,
           coopmat1BM,
           coopmat1BN,
           coopmat1BK,
           coopmat1WM,
           coopmat1WN,
           coopmat1TM,
           coopmat1TN,
           coopmat1TK,
           coopmat1Warp)
         && (conv3x3NhwcCoopmat1TunerValueValid == 0 || conv3x3NhwcCoopmat1TunerValueValid == 1)
         && VulkanKernels::Conv3x3ImplicitGemmCoopmat1AccF32NhwcVec8::isConfigSupported(
           conv3x3NhwcCoopmat1BlockSize,
           conv3x3NhwcCoopmat1BM,
           conv3x3NhwcCoopmat1BN,
           conv3x3NhwcCoopmat1BK,
           conv3x3NhwcCoopmat1WM,
           conv3x3NhwcCoopmat1WN,
           conv3x3NhwcCoopmat1TM,
           conv3x3NhwcCoopmat1TN,
           conv3x3NhwcCoopmat1TK,
           conv3x3NhwcCoopmat1Warp)
         && (conv5x5NhwcCoopmat1TunerValueValid == 0 || conv5x5NhwcCoopmat1TunerValueValid == 1)
         && VulkanKernels::Conv3x3ImplicitGemmCoopmat1AccF32NhwcVec8::isConfigSupported(
           conv5x5NhwcCoopmat1BlockSize,
           conv5x5NhwcCoopmat1BM,
           conv5x5NhwcCoopmat1BN,
           conv5x5NhwcCoopmat1BK,
           conv5x5NhwcCoopmat1WM,
           conv5x5NhwcCoopmat1WN,
           conv5x5NhwcCoopmat1TM,
           conv5x5NhwcCoopmat1TN,
           conv5x5NhwcCoopmat1TK,
           conv5x5NhwcCoopmat1Warp)
         && (enableWinogradGemmCoopmat1AccF16 == 0 || enableWinogradGemmCoopmat1AccF16 == 1)
         && (winogradGemmCoopmat1AccF16TunerValueValid == 0 || winogradGemmCoopmat1AccF16TunerValueValid == 1)
         && VulkanKernels::WinogradGemmCoopmat1AccF16::isConfigSupported(
           coopmat1AccF16BlockSize,
           coopmat1AccF16BM,
           coopmat1AccF16BN,
           coopmat1AccF16BK,
           coopmat1AccF16WM,
           coopmat1AccF16WN,
           coopmat1AccF16TM,
           coopmat1AccF16TN,
           coopmat1AccF16TK,
           coopmat1AccF16Warp)
         && (gemmStridedNhwcCoopmat1TunerValueValid == 0 || gemmStridedNhwcCoopmat1TunerValueValid == 1)
         && VulkanKernels::GemmStridedCoopmat1Nhwc::isConfigSupported(
           nhwcStridedCoopmat1BlockSize,
           nhwcStridedCoopmat1BM,
           nhwcStridedCoopmat1BN,
           nhwcStridedCoopmat1BK,
           nhwcStridedCoopmat1WM,
           nhwcStridedCoopmat1WN,
           nhwcStridedCoopmat1TM,
           nhwcStridedCoopmat1TN,
           nhwcStridedCoopmat1TK,
           nhwcStridedCoopmat1Warp)
         && (gemmStridedNhwcCoopmat1AccF16TunerValueValid == 0 ||
          gemmStridedNhwcCoopmat1AccF16TunerValueValid == 1)
         && VulkanKernels::GemmStridedCoopmat1AccF16Nhwc::isConfigSupported(
           nhwcStridedCoopmat1AccF16BlockSize,
           nhwcStridedCoopmat1AccF16BM,
           nhwcStridedCoopmat1AccF16BN,
           nhwcStridedCoopmat1AccF16BK,
           nhwcStridedCoopmat1AccF16WM,
           nhwcStridedCoopmat1AccF16WN,
           nhwcStridedCoopmat1AccF16TM,
           nhwcStridedCoopmat1AccF16TN,
           nhwcStridedCoopmat1AccF16TK,
           nhwcStridedCoopmat1AccF16Warp)
         && (gemmStridedNhwcCoopmat2TunerValueValid == 0 || gemmStridedNhwcCoopmat2TunerValueValid == 1)
         && VulkanKernels::GemmStridedCoopmat2Nhwc::isConfigSupported(
           nhwcStridedCoopmat2BlockSize, nhwcStridedCoopmat2BM,
           nhwcStridedCoopmat2BN, nhwcStridedCoopmat2BK)
         && (gemmStridedNhwcCoopmat2AccF16TunerValueValid == 0 || gemmStridedNhwcCoopmat2AccF16TunerValueValid == 1)
         && VulkanKernels::GemmStridedCoopmat2AccF16Nhwc::isConfigSupported(
           nhwcStridedCoopmat2AccF16BlockSize, nhwcStridedCoopmat2AccF16BM,
           nhwcStridedCoopmat2AccF16BN, nhwcStridedCoopmat2AccF16BK)
         && (conv3x3NhwcCoopmat2TunerValueValid == 0 || conv3x3NhwcCoopmat2TunerValueValid == 1)
         && VulkanKernels::Conv3x3ImplicitGemmCoopmat2AccF32NhwcVec8::isConfigSupported(
           conv3x3NhwcCoopmat2BlockSize, conv3x3NhwcCoopmat2BM,
           conv3x3NhwcCoopmat2BN, conv3x3NhwcCoopmat2BK)
         && (conv5x5NhwcCoopmat2TunerValueValid == 0 || conv5x5NhwcCoopmat2TunerValueValid == 1)
         && VulkanKernels::Conv3x3ImplicitGemmCoopmat2AccF32NhwcVec8::isConfigSupported(
           conv5x5NhwcCoopmat2BlockSize, conv5x5NhwcCoopmat2BM,
           conv5x5NhwcCoopmat2BN, conv5x5NhwcCoopmat2BK)
         && (nhwcGemmF32UseCoopmat2 == 0 || nhwcGemmF32UseCoopmat2 == 1)
         && (nhwcGemmUseCoopmat2 == 0 || nhwcGemmUseCoopmat2 == 1)
         && (nhwcGemmUseCoopmatAccF16 == 0 || nhwcGemmUseCoopmatAccF16 == 1)
         && nhwcGemmCoopmat1F32TimeUs >= 0
         && nhwcGemmCoopmat2F32TimeUs >= 0
         && nhwcGemmCoopmat1F16TimeUs >= 0
         && nhwcGemmCoopmat2F16TimeUs >= 0
         && (gemmStridedNhwcDot2TunerValueValid == 0 || gemmStridedNhwcDot2TunerValueValid == 1)
         && VulkanKernels::GemmStridedDot2Nhwc::isConfigSupported(
           nhwcStridedDot2BlockSize, nhwcStridedDot2BM, nhwcStridedDot2BN, nhwcStridedDot2WM,
           nhwcStridedDot2WN, nhwcStridedDot2WMIter, nhwcStridedDot2TM, nhwcStridedDot2TN,
           nhwcStridedDot2Warp)
         && (nhwcGemmUseDot2 == 0 || nhwcGemmUseDot2 == 1)
         && (gemmStridedNhwcDot2AccF16TunerValueValid == 0 || gemmStridedNhwcDot2AccF16TunerValueValid == 1)
         && VulkanKernels::GemmStridedDot2AccF16Nhwc::isConfigSupported(
           nhwcStridedDot2AccF16BlockSize, nhwcStridedDot2AccF16BM, nhwcStridedDot2AccF16BN,
           nhwcStridedDot2AccF16WM, nhwcStridedDot2AccF16WN, nhwcStridedDot2AccF16WMIter,
           nhwcStridedDot2AccF16TM, nhwcStridedDot2AccF16TN, nhwcStridedDot2AccF16Warp)
         && (nhwcGemmUseDot2AccF16 == 0 || nhwcGemmUseDot2AccF16 == 1)
         && nhwcGemmDot2F32TimeUs >= 0
         && nhwcGemmDot2F16TimeUs >= 0
         && (nhwcConv3x3UseCoopmat2 == 0 || nhwcConv3x3UseCoopmat2 == 1)
         && (nhwcConv5x5UseCoopmat2 == 0 || nhwcConv5x5UseCoopmat2 == 1)
         && (enableWinogradGemmCoopmat2 == 0 || enableWinogradGemmCoopmat2 == 1)
         && (winogradGemmCoopmat2TunerValueValid == 0 || winogradGemmCoopmat2TunerValueValid == 1)
         && VulkanKernels::WinogradGemmCoopmat2::isConfigSupported(
           coopmat2BlockSize, coopmat2BM, coopmat2BN, coopmat2BK)
         && (enableWinogradGemmCoopmat2AccF16 == 0 || enableWinogradGemmCoopmat2AccF16 == 1)
         && (winogradGemmCoopmat2AccF16TunerValueValid == 0 || winogradGemmCoopmat2AccF16TunerValueValid == 1)
         && VulkanKernels::WinogradGemmCoopmat2AccF16::isConfigSupported(
           coopmat2AccF16BlockSize, coopmat2AccF16BM, coopmat2AccF16BN, coopmat2AccF16BK);
}

bool VulkanTuneParams::normalizeGemmVariantSelectionsForDevice(
  const VulkanDeviceInfo& deviceInfo,
  bool fp16Storage,
  bool fp16Compute) {
  bool changed = false;
  struct Candidate {
    int32_t* enabled;
    int32_t valueValid;
    bool featureSupported;
    int32_t timeUs;
    bool accF16;
  };
  auto selectBest = [&](std::initializer_list<Candidate> coopmat, std::initializer_list<Candidate> dot2) {
    auto eligible = [&](const Candidate& candidate) {
      return fp16Storage && fp16Compute && candidate.featureSupported && candidate.valueValid != 0 &&
        candidate.timeUs > 0;
    };
    auto bestForAccumulation = [&](std::initializer_list<Candidate> tier, bool accF16) {
      const Candidate* best = nullptr;
      for(const Candidate& candidate: tier) {
        if(eligible(candidate) && candidate.accF16 == accF16 && (best == nullptr || candidate.timeUs < best->timeUs))
          best = &candidate;
      }
      return best;
    };
    auto chooseFromTier = [&](std::initializer_list<Candidate> tier) {
      const Candidate* f32 = bestForAccumulation(tier, false);
      const Candidate* f16 = bestForAccumulation(tier, true);
      if(f16 != nullptr && (f32 == nullptr || (int64_t)f16->timeUs * 5 < (int64_t)f32->timeUs * 4))
        return f16;
      return f32;
    };

    // Coopmat has priority over Dot2, and Dot2 over tiled. Within the active
    // accelerator tier, select the fastest implementation for each
    // accumulation type and require a 25% gain before accepting FP16.
    const Candidate* selected = chooseFromTier(coopmat);
    if(selected == nullptr)
      selected = chooseFromTier(dot2);
    auto apply = [&](std::initializer_list<Candidate> tier) {
      for(const Candidate& candidate: tier) {
        const int32_t desired = &candidate == selected ? 1 : 0;
        if(*candidate.enabled != desired) {
          *candidate.enabled = desired;
          changed = true;
        }
      }
    };
    apply(coopmat);
    apply(dot2);
  };
  selectBest({
    {&enableWinogradGemmCoopmat2AccF16, winogradGemmCoopmat2AccF16TunerValueValid,
     deviceInfo.supportsCoopmat2F16AccF16, winogradGemmCoopmat2F16TimeUs, true},
    {&enableWinogradGemmCoopmat2, winogradGemmCoopmat2TunerValueValid,
     deviceInfo.supportsCoopmat2F16, winogradGemmCoopmat2F32TimeUs, false},
    {&enableWinogradGemmCoopmat1AccF16, winogradGemmCoopmat1AccF16TunerValueValid,
     deviceInfo.supportsCoopmat1F16AccF16, winogradGemmCoopmat1F16TimeUs, true},
    {&enableWinogradGemmCoopmat1, winogradGemmCoopmat1TunerValueValid,
     deviceInfo.supportsCoopmat1F16, winogradGemmCoopmat1F32TimeUs, false},
  }, {
    {&enableWinogradGemmDot2AccF16, winogradGemmDot2AccF16TunerValueValid,
     deviceInfo.supportsDot2F16AccF16, winogradGemmDot2F16TimeUs, true},
    {&enableWinogradGemmDot2, winogradGemmDot2TunerValueValid,
     deviceInfo.supportsDot2F16, winogradGemmDot2F32TimeUs, false},
  });
  return changed;
}

void VulkanTuneParams::save(const string& filename, const VulkanTuneParams& config) {
  string tmpFilename = uniqueTempFileNameForAtomicSave(filename);
  ofstream out;
  FileUtils::open(out, tmpFilename, std::ios::out | std::ios::trunc);
  out << VULKAN_TUNEPARAMS_VERSION_LINE << "\n";
  out << "nhwcWinogradTransformTunedLayout=" << config.nhwcWinogradTransformTunedLayout << "\n";
  out << "gemmVariantFilterTunedSignature=" << config.gemmVariantFilterTunedSignature << "\n";
  out << "tunedKernelMask=" << config.tunedKernelMask << "\n";
  out << "winogradGemmCoopmat1F32TimeUs=" << config.winogradGemmCoopmat1F32TimeUs << "\n";
  out << "winogradGemmCoopmat2F32TimeUs=" << config.winogradGemmCoopmat2F32TimeUs << "\n";
  out << "winogradGemmCoopmat1F16TimeUs=" << config.winogradGemmCoopmat1F16TimeUs << "\n";
  out << "winogradGemmCoopmat2F16TimeUs=" << config.winogradGemmCoopmat2F16TimeUs << "\n";
  out << "winogradGemmDot2F32TimeUs=" << config.winogradGemmDot2F32TimeUs << "\n";
  out << "winogradGemmDot2F16TimeUs=" << config.winogradGemmDot2F16TimeUs << "\n";
  out << "nchwToNhwcSmallTile=" << config.nchwToNhwcSmallTile << "\n";
  out << "nchwToNhwcLargeTile=" << config.nchwToNhwcLargeTile << "\n";
  out << "nchwToNhwcTileCrossover=" << config.nchwToNhwcTileCrossover << "\n";
  out << "nhwcWinogradTransformLocalSizeX=" << config.nhwcWinogradTransformLocalSizeX << "\n";
  out << "nhwcWinogradTransformLocalSizeY=" << config.nhwcWinogradTransformLocalSizeY << "\n";
  out << "nhwcWinogradUntransformLocalSizeX=" << config.nhwcWinogradUntransformLocalSizeX << "\n";
  out << "nhwcWinogradUntransformLocalSizeY=" << config.nhwcWinogradUntransformLocalSizeY << "\n";
  out << "nhwcWinograd3x3OutTile=" << config.nhwcWinograd3x3OutTile << "\n";
  out << "conv3x3NhwcWinogradTunerValueValid=" << config.conv3x3NhwcWinogradTunerValueValid << "\n";
  out << "conv5x5NhwcWinogradTunerValueValid=" << config.conv5x5NhwcWinogradTunerValueValid << "\n";
  out << "gemmStridedTiledNhwcLocalSizeX=" << config.gemmStridedTiledNhwcLocalSizeX << "\n";
  out << "gemmStridedTiledNhwcLocalSizeY=" << config.gemmStridedTiledNhwcLocalSizeY << "\n";
  out << "gemmStridedTiledNhwcTileK=" << config.gemmStridedTiledNhwcTileK << "\n";
  out << "gemmStridedTiledNhwcRN=" << config.gemmStridedTiledNhwcRN << "\n";
  out << "gemmDirectLocalSizeX=" << config.gemmDirectLocalSizeX << "\n";
  out << "gemmDirectLocalSizeY=" << config.gemmDirectLocalSizeY << "\n";
  out << "winogradGemmM=" << config.winogradGemmM << "\n";
  out << "winogradGemmN=" << config.winogradGemmN << "\n";
  out << "winogradGemmK=" << config.winogradGemmK << "\n";
  out << "winogradGemmRN=" << config.winogradGemmRN << "\n";
  out << "enableWinogradGemmDot2=" << config.enableWinogradGemmDot2 << "\n";
  out << "enableWinogradGemmDot2AccF16=" << config.enableWinogradGemmDot2AccF16 << "\n";
  out << "winogradGemmDot2TunerValueValid=" << config.winogradGemmDot2TunerValueValid << "\n";
  out << "winogradGemmDot2AccF16TunerValueValid=" << config.winogradGemmDot2AccF16TunerValueValid << "\n";
  out << "dot2BlockSize=" << config.dot2BlockSize << "\n";
  out << "dot2BM=" << config.dot2BM << "\n";
  out << "dot2BN=" << config.dot2BN << "\n";
  out << "dot2WM=" << config.dot2WM << "\n";
  out << "dot2WN=" << config.dot2WN << "\n";
  out << "dot2WMIter=" << config.dot2WMIter << "\n";
  out << "dot2TM=" << config.dot2TM << "\n";
  out << "dot2TN=" << config.dot2TN << "\n";
  out << "dot2Warp=" << config.dot2Warp << "\n";
  out << "dot2AccF16BlockSize=" << config.dot2AccF16BlockSize << "\n";
  out << "dot2AccF16BM=" << config.dot2AccF16BM << "\n";
  out << "dot2AccF16BN=" << config.dot2AccF16BN << "\n";
  out << "dot2AccF16WM=" << config.dot2AccF16WM << "\n";
  out << "dot2AccF16WN=" << config.dot2AccF16WN << "\n";
  out << "dot2AccF16WMIter=" << config.dot2AccF16WMIter << "\n";
  out << "dot2AccF16TM=" << config.dot2AccF16TM << "\n";
  out << "dot2AccF16TN=" << config.dot2AccF16TN << "\n";
  out << "dot2AccF16Warp=" << config.dot2AccF16Warp << "\n";
  out << "enableWinogradGemmCoopmat1=" << config.enableWinogradGemmCoopmat1 << "\n";
  out << "winogradGemmCoopmat1TunerValueValid=" << config.winogradGemmCoopmat1TunerValueValid << "\n";
  out << "coopmat1BlockSize=" << config.coopmat1BlockSize << "\n";
  out << "coopmat1BM=" << config.coopmat1BM << "\n";
  out << "coopmat1BN=" << config.coopmat1BN << "\n";
  out << "coopmat1BK=" << config.coopmat1BK << "\n";
  out << "coopmat1WM=" << config.coopmat1WM << "\n";
  out << "coopmat1WN=" << config.coopmat1WN << "\n";
  out << "coopmat1TM=" << config.coopmat1TM << "\n";
  out << "coopmat1TN=" << config.coopmat1TN << "\n";
  out << "coopmat1TK=" << config.coopmat1TK << "\n";
  out << "coopmat1Warp=" << config.coopmat1Warp << "\n";
  out << "conv3x3NhwcCoopmat1TunerValueValid=" << config.conv3x3NhwcCoopmat1TunerValueValid << "\n";
  out << "conv3x3NhwcCoopmat1BlockSize=" << config.conv3x3NhwcCoopmat1BlockSize << "\n";
  out << "conv3x3NhwcCoopmat1BM=" << config.conv3x3NhwcCoopmat1BM << "\n";
  out << "conv3x3NhwcCoopmat1BN=" << config.conv3x3NhwcCoopmat1BN << "\n";
  out << "conv3x3NhwcCoopmat1BK=" << config.conv3x3NhwcCoopmat1BK << "\n";
  out << "conv3x3NhwcCoopmat1WM=" << config.conv3x3NhwcCoopmat1WM << "\n";
  out << "conv3x3NhwcCoopmat1WN=" << config.conv3x3NhwcCoopmat1WN << "\n";
  out << "conv3x3NhwcCoopmat1TM=" << config.conv3x3NhwcCoopmat1TM << "\n";
  out << "conv3x3NhwcCoopmat1TN=" << config.conv3x3NhwcCoopmat1TN << "\n";
  out << "conv3x3NhwcCoopmat1TK=" << config.conv3x3NhwcCoopmat1TK << "\n";
  out << "conv3x3NhwcCoopmat1Warp=" << config.conv3x3NhwcCoopmat1Warp << "\n";
  out << "conv5x5NhwcCoopmat1TunerValueValid=" << config.conv5x5NhwcCoopmat1TunerValueValid << "\n";
  out << "conv5x5NhwcCoopmat1BlockSize=" << config.conv5x5NhwcCoopmat1BlockSize << "\n";
  out << "conv5x5NhwcCoopmat1BM=" << config.conv5x5NhwcCoopmat1BM << "\n";
  out << "conv5x5NhwcCoopmat1BN=" << config.conv5x5NhwcCoopmat1BN << "\n";
  out << "conv5x5NhwcCoopmat1BK=" << config.conv5x5NhwcCoopmat1BK << "\n";
  out << "conv5x5NhwcCoopmat1WM=" << config.conv5x5NhwcCoopmat1WM << "\n";
  out << "conv5x5NhwcCoopmat1WN=" << config.conv5x5NhwcCoopmat1WN << "\n";
  out << "conv5x5NhwcCoopmat1TM=" << config.conv5x5NhwcCoopmat1TM << "\n";
  out << "conv5x5NhwcCoopmat1TN=" << config.conv5x5NhwcCoopmat1TN << "\n";
  out << "conv5x5NhwcCoopmat1TK=" << config.conv5x5NhwcCoopmat1TK << "\n";
  out << "conv5x5NhwcCoopmat1Warp=" << config.conv5x5NhwcCoopmat1Warp << "\n";
  out << "conv3x3NhwcCoopmat1AccF16TunerValueValid=" << config.conv3x3NhwcCoopmat1AccF16TunerValueValid << "\n";
  out << "conv3x3NhwcCoopmat1AccF16BlockSize=" << config.conv3x3NhwcCoopmat1AccF16BlockSize << "\n";
  out << "conv3x3NhwcCoopmat1AccF16BM=" << config.conv3x3NhwcCoopmat1AccF16BM << "\n";
  out << "conv3x3NhwcCoopmat1AccF16BN=" << config.conv3x3NhwcCoopmat1AccF16BN << "\n";
  out << "conv3x3NhwcCoopmat1AccF16BK=" << config.conv3x3NhwcCoopmat1AccF16BK << "\n";
  out << "conv3x3NhwcCoopmat1AccF16WM=" << config.conv3x3NhwcCoopmat1AccF16WM << "\n";
  out << "conv3x3NhwcCoopmat1AccF16WN=" << config.conv3x3NhwcCoopmat1AccF16WN << "\n";
  out << "conv3x3NhwcCoopmat1AccF16TM=" << config.conv3x3NhwcCoopmat1AccF16TM << "\n";
  out << "conv3x3NhwcCoopmat1AccF16TN=" << config.conv3x3NhwcCoopmat1AccF16TN << "\n";
  out << "conv3x3NhwcCoopmat1AccF16TK=" << config.conv3x3NhwcCoopmat1AccF16TK << "\n";
  out << "conv3x3NhwcCoopmat1AccF16Warp=" << config.conv3x3NhwcCoopmat1AccF16Warp << "\n";
  out << "conv5x5NhwcCoopmat1AccF16TunerValueValid=" << config.conv5x5NhwcCoopmat1AccF16TunerValueValid << "\n";
  out << "conv5x5NhwcCoopmat1AccF16BlockSize=" << config.conv5x5NhwcCoopmat1AccF16BlockSize << "\n";
  out << "conv5x5NhwcCoopmat1AccF16BM=" << config.conv5x5NhwcCoopmat1AccF16BM << "\n";
  out << "conv5x5NhwcCoopmat1AccF16BN=" << config.conv5x5NhwcCoopmat1AccF16BN << "\n";
  out << "conv5x5NhwcCoopmat1AccF16BK=" << config.conv5x5NhwcCoopmat1AccF16BK << "\n";
  out << "conv5x5NhwcCoopmat1AccF16WM=" << config.conv5x5NhwcCoopmat1AccF16WM << "\n";
  out << "conv5x5NhwcCoopmat1AccF16WN=" << config.conv5x5NhwcCoopmat1AccF16WN << "\n";
  out << "conv5x5NhwcCoopmat1AccF16TM=" << config.conv5x5NhwcCoopmat1AccF16TM << "\n";
  out << "conv5x5NhwcCoopmat1AccF16TN=" << config.conv5x5NhwcCoopmat1AccF16TN << "\n";
  out << "conv5x5NhwcCoopmat1AccF16TK=" << config.conv5x5NhwcCoopmat1AccF16TK << "\n";
  out << "conv5x5NhwcCoopmat1AccF16Warp=" << config.conv5x5NhwcCoopmat1AccF16Warp << "\n";
  out << "enableWinogradGemmCoopmat1AccF16=" << config.enableWinogradGemmCoopmat1AccF16 << "\n";
  out << "winogradGemmCoopmat1AccF16TunerValueValid=" << config.winogradGemmCoopmat1AccF16TunerValueValid << "\n";
  out << "coopmat1AccF16BlockSize=" << config.coopmat1AccF16BlockSize << "\n";
  out << "coopmat1AccF16BM=" << config.coopmat1AccF16BM << "\n";
  out << "coopmat1AccF16BN=" << config.coopmat1AccF16BN << "\n";
  out << "coopmat1AccF16BK=" << config.coopmat1AccF16BK << "\n";
  out << "coopmat1AccF16WM=" << config.coopmat1AccF16WM << "\n";
  out << "coopmat1AccF16WN=" << config.coopmat1AccF16WN << "\n";
  out << "coopmat1AccF16TM=" << config.coopmat1AccF16TM << "\n";
  out << "coopmat1AccF16TN=" << config.coopmat1AccF16TN << "\n";
  out << "coopmat1AccF16TK=" << config.coopmat1AccF16TK << "\n";
  out << "coopmat1AccF16Warp=" << config.coopmat1AccF16Warp << "\n";
  out << "gemmStridedNhwcCoopmat1TunerValueValid=" << config.gemmStridedNhwcCoopmat1TunerValueValid << "\n";
  out << "nhwcStridedCoopmat1BlockSize=" << config.nhwcStridedCoopmat1BlockSize << "\n";
  out << "nhwcStridedCoopmat1BM=" << config.nhwcStridedCoopmat1BM << "\n";
  out << "nhwcStridedCoopmat1BN=" << config.nhwcStridedCoopmat1BN << "\n";
  out << "nhwcStridedCoopmat1BK=" << config.nhwcStridedCoopmat1BK << "\n";
  out << "nhwcStridedCoopmat1WM=" << config.nhwcStridedCoopmat1WM << "\n";
  out << "nhwcStridedCoopmat1WN=" << config.nhwcStridedCoopmat1WN << "\n";
  out << "nhwcStridedCoopmat1TM=" << config.nhwcStridedCoopmat1TM << "\n";
  out << "nhwcStridedCoopmat1TN=" << config.nhwcStridedCoopmat1TN << "\n";
  out << "nhwcStridedCoopmat1TK=" << config.nhwcStridedCoopmat1TK << "\n";
  out << "nhwcStridedCoopmat1Warp=" << config.nhwcStridedCoopmat1Warp << "\n";
  out << "gemmStridedNhwcCoopmat1AccF16TunerValueValid=" << config.gemmStridedNhwcCoopmat1AccF16TunerValueValid << "\n";
  out << "nhwcStridedCoopmat1AccF16BlockSize=" << config.nhwcStridedCoopmat1AccF16BlockSize << "\n";
  out << "nhwcStridedCoopmat1AccF16BM=" << config.nhwcStridedCoopmat1AccF16BM << "\n";
  out << "nhwcStridedCoopmat1AccF16BN=" << config.nhwcStridedCoopmat1AccF16BN << "\n";
  out << "nhwcStridedCoopmat1AccF16BK=" << config.nhwcStridedCoopmat1AccF16BK << "\n";
  out << "nhwcStridedCoopmat1AccF16WM=" << config.nhwcStridedCoopmat1AccF16WM << "\n";
  out << "nhwcStridedCoopmat1AccF16WN=" << config.nhwcStridedCoopmat1AccF16WN << "\n";
  out << "nhwcStridedCoopmat1AccF16TM=" << config.nhwcStridedCoopmat1AccF16TM << "\n";
  out << "nhwcStridedCoopmat1AccF16TN=" << config.nhwcStridedCoopmat1AccF16TN << "\n";
  out << "nhwcStridedCoopmat1AccF16TK=" << config.nhwcStridedCoopmat1AccF16TK << "\n";
  out << "nhwcStridedCoopmat1AccF16Warp=" << config.nhwcStridedCoopmat1AccF16Warp << "\n";
  out << "gemmStridedNhwcCoopmat2TunerValueValid=" << config.gemmStridedNhwcCoopmat2TunerValueValid << "\n";
  out << "nhwcStridedCoopmat2BlockSize=" << config.nhwcStridedCoopmat2BlockSize << "\n";
  out << "nhwcStridedCoopmat2BM=" << config.nhwcStridedCoopmat2BM << "\n";
  out << "nhwcStridedCoopmat2BN=" << config.nhwcStridedCoopmat2BN << "\n";
  out << "nhwcStridedCoopmat2BK=" << config.nhwcStridedCoopmat2BK << "\n";
  out << "gemmStridedNhwcCoopmat2AccF16TunerValueValid=" << config.gemmStridedNhwcCoopmat2AccF16TunerValueValid << "\n";
  out << "nhwcStridedCoopmat2AccF16BlockSize=" << config.nhwcStridedCoopmat2AccF16BlockSize << "\n";
  out << "nhwcStridedCoopmat2AccF16BM=" << config.nhwcStridedCoopmat2AccF16BM << "\n";
  out << "nhwcStridedCoopmat2AccF16BN=" << config.nhwcStridedCoopmat2AccF16BN << "\n";
  out << "nhwcStridedCoopmat2AccF16BK=" << config.nhwcStridedCoopmat2AccF16BK << "\n";
  out << "nhwcGemmF32UseCoopmat2=" << config.nhwcGemmF32UseCoopmat2 << "\n";
  out << "nhwcGemmUseCoopmat2=" << config.nhwcGemmUseCoopmat2 << "\n";
  out << "nhwcGemmUseCoopmatAccF16=" << config.nhwcGemmUseCoopmatAccF16 << "\n";
  out << "nhwcGemmCoopmat1F32TimeUs=" << config.nhwcGemmCoopmat1F32TimeUs << "\n";
  out << "nhwcGemmCoopmat2F32TimeUs=" << config.nhwcGemmCoopmat2F32TimeUs << "\n";
  out << "nhwcGemmCoopmat1F16TimeUs=" << config.nhwcGemmCoopmat1F16TimeUs << "\n";
  out << "nhwcGemmCoopmat2F16TimeUs=" << config.nhwcGemmCoopmat2F16TimeUs << "\n";
  out << "gemmStridedNhwcDot2TunerValueValid=" << config.gemmStridedNhwcDot2TunerValueValid << "\n";
  out << "nhwcStridedDot2BlockSize=" << config.nhwcStridedDot2BlockSize << "\n";
  out << "nhwcStridedDot2BM=" << config.nhwcStridedDot2BM << "\n";
  out << "nhwcStridedDot2BN=" << config.nhwcStridedDot2BN << "\n";
  out << "nhwcStridedDot2WM=" << config.nhwcStridedDot2WM << "\n";
  out << "nhwcStridedDot2WN=" << config.nhwcStridedDot2WN << "\n";
  out << "nhwcStridedDot2WMIter=" << config.nhwcStridedDot2WMIter << "\n";
  out << "nhwcStridedDot2TM=" << config.nhwcStridedDot2TM << "\n";
  out << "nhwcStridedDot2TN=" << config.nhwcStridedDot2TN << "\n";
  out << "nhwcStridedDot2Warp=" << config.nhwcStridedDot2Warp << "\n";
  out << "nhwcGemmUseDot2=" << config.nhwcGemmUseDot2 << "\n";
  out << "gemmStridedNhwcDot2AccF16TunerValueValid=" << config.gemmStridedNhwcDot2AccF16TunerValueValid << "\n";
  out << "nhwcStridedDot2AccF16BlockSize=" << config.nhwcStridedDot2AccF16BlockSize << "\n";
  out << "nhwcStridedDot2AccF16BM=" << config.nhwcStridedDot2AccF16BM << "\n";
  out << "nhwcStridedDot2AccF16BN=" << config.nhwcStridedDot2AccF16BN << "\n";
  out << "nhwcStridedDot2AccF16WM=" << config.nhwcStridedDot2AccF16WM << "\n";
  out << "nhwcStridedDot2AccF16WN=" << config.nhwcStridedDot2AccF16WN << "\n";
  out << "nhwcStridedDot2AccF16WMIter=" << config.nhwcStridedDot2AccF16WMIter << "\n";
  out << "nhwcStridedDot2AccF16TM=" << config.nhwcStridedDot2AccF16TM << "\n";
  out << "nhwcStridedDot2AccF16TN=" << config.nhwcStridedDot2AccF16TN << "\n";
  out << "nhwcStridedDot2AccF16Warp=" << config.nhwcStridedDot2AccF16Warp << "\n";
  out << "nhwcGemmUseDot2AccF16=" << config.nhwcGemmUseDot2AccF16 << "\n";
  out << "nhwcGemmDot2F32TimeUs=" << config.nhwcGemmDot2F32TimeUs << "\n";
  out << "nhwcGemmDot2F16TimeUs=" << config.nhwcGemmDot2F16TimeUs << "\n";
  out << "conv3x3NhwcCoopmat2TunerValueValid=" << config.conv3x3NhwcCoopmat2TunerValueValid << "\n";
  out << "conv3x3NhwcCoopmat2BlockSize=" << config.conv3x3NhwcCoopmat2BlockSize << "\n";
  out << "conv3x3NhwcCoopmat2BM=" << config.conv3x3NhwcCoopmat2BM << "\n";
  out << "conv3x3NhwcCoopmat2BN=" << config.conv3x3NhwcCoopmat2BN << "\n";
  out << "conv3x3NhwcCoopmat2BK=" << config.conv3x3NhwcCoopmat2BK << "\n";
  out << "nhwcConv3x3UseCoopmat2=" << config.nhwcConv3x3UseCoopmat2 << "\n";
  out << "conv5x5NhwcCoopmat2TunerValueValid=" << config.conv5x5NhwcCoopmat2TunerValueValid << "\n";
  out << "conv5x5NhwcCoopmat2BlockSize=" << config.conv5x5NhwcCoopmat2BlockSize << "\n";
  out << "conv5x5NhwcCoopmat2BM=" << config.conv5x5NhwcCoopmat2BM << "\n";
  out << "conv5x5NhwcCoopmat2BN=" << config.conv5x5NhwcCoopmat2BN << "\n";
  out << "conv5x5NhwcCoopmat2BK=" << config.conv5x5NhwcCoopmat2BK << "\n";
  out << "nhwcConv5x5UseCoopmat2=" << config.nhwcConv5x5UseCoopmat2 << "\n";
  out << "conv3x3NhwcCoopmat2AccF16TunerValueValid=" << config.conv3x3NhwcCoopmat2AccF16TunerValueValid << "\n";
  out << "conv3x3NhwcCoopmat2AccF16BlockSize=" << config.conv3x3NhwcCoopmat2AccF16BlockSize << "\n";
  out << "conv3x3NhwcCoopmat2AccF16BM=" << config.conv3x3NhwcCoopmat2AccF16BM << "\n";
  out << "conv3x3NhwcCoopmat2AccF16BN=" << config.conv3x3NhwcCoopmat2AccF16BN << "\n";
  out << "conv3x3NhwcCoopmat2AccF16BK=" << config.conv3x3NhwcCoopmat2AccF16BK << "\n";
  out << "conv5x5NhwcCoopmat2AccF16TunerValueValid=" << config.conv5x5NhwcCoopmat2AccF16TunerValueValid << "\n";
  out << "conv5x5NhwcCoopmat2AccF16BlockSize=" << config.conv5x5NhwcCoopmat2AccF16BlockSize << "\n";
  out << "conv5x5NhwcCoopmat2AccF16BM=" << config.conv5x5NhwcCoopmat2AccF16BM << "\n";
  out << "conv5x5NhwcCoopmat2AccF16BN=" << config.conv5x5NhwcCoopmat2AccF16BN << "\n";
  out << "conv5x5NhwcCoopmat2AccF16BK=" << config.conv5x5NhwcCoopmat2AccF16BK << "\n";
  out << "enableWinogradGemmCoopmat2=" << config.enableWinogradGemmCoopmat2 << "\n";
  out << "winogradGemmCoopmat2TunerValueValid=" << config.winogradGemmCoopmat2TunerValueValid << "\n";
  out << "coopmat2BlockSize=" << config.coopmat2BlockSize << "\n";
  out << "coopmat2BM=" << config.coopmat2BM << "\n";
  out << "coopmat2BN=" << config.coopmat2BN << "\n";
  out << "coopmat2BK=" << config.coopmat2BK << "\n";
  out << "enableWinogradGemmCoopmat2AccF16=" << config.enableWinogradGemmCoopmat2AccF16 << "\n";
  out << "winogradGemmCoopmat2AccF16TunerValueValid=" << config.winogradGemmCoopmat2AccF16TunerValueValid << "\n";
  out << "coopmat2AccF16BlockSize=" << config.coopmat2AccF16BlockSize << "\n";
  out << "coopmat2AccF16BM=" << config.coopmat2AccF16BM << "\n";
  out << "coopmat2AccF16BN=" << config.coopmat2AccF16BN << "\n";
  out << "coopmat2AccF16BK=" << config.coopmat2AccF16BK << "\n";
  out << "gpoolNhwcXystride=" << config.gpoolNhwcXystride << "\n";
  out << "valueHeadPoolNhwcXystride=" << config.valueHeadPoolNhwcXystride << "\n";
  out << "spatialRMSNormNhwcTile=" << config.spatialRMSNormNhwcTile << "\n";
  out << "swiGLUTunerValueValid=" << config.swiGLUTunerValueValid << "\n";
  out << "swiGLULocalSizeX=" << config.swiGLULocalSizeX << "\n";
  out << "attnBlockQ=" << config.attnBlockQ << "\n";
  out << "attnBlockKV=" << config.attnBlockKV << "\n";
  out << "attnQPerThread=" << config.attnQPerThread << "\n";
  out << "attnNhwcBlockQ=" << config.attnNhwcBlockQ << "\n";
  out << "attnNhwcBlockKV=" << config.attnNhwcBlockKV << "\n";
  out << "attnNhwcQPerThread=" << config.attnNhwcQPerThread << "\n";
  out << "attnNhwcVariantPolicyVersion=" << config.attnNhwcVariantPolicyVersion << "\n";
  out << "attnNhwcCoopmat1TunerValueValid=" << config.attnNhwcCoopmat1TunerValueValid << "\n";
  out << "attnNhwcUseCoopmat1=" << config.attnNhwcUseCoopmat1 << "\n";
  out << "attnNhwcTiledTimeUs=" << config.attnNhwcTiledTimeUs << "\n";
  out << "attnNhwcCoopmat1TimeUs=" << config.attnNhwcCoopmat1TimeUs << "\n";
  out << "attnNhwcCoopmat1BlockSize=" << config.attnNhwcCoopmat1BlockSize << "\n";
  out << "attnNhwcCoopmat1BlockQ=" << config.attnNhwcCoopmat1BlockQ << "\n";
  out << "attnNhwcCoopmat1BlockKV=" << config.attnNhwcCoopmat1BlockKV << "\n";
  out << "attnNhwcCoopmat1TM=" << config.attnNhwcCoopmat1TM << "\n";
  out << "attnNhwcCoopmat1TN=" << config.attnNhwcCoopmat1TN << "\n";
  out << "attnNhwcCoopmat1TK=" << config.attnNhwcCoopmat1TK << "\n";
  out << "attnNhwcCoopmat1Warp=" << config.attnNhwcCoopmat1Warp << "\n";
  out << "attnNhwcCoopmat1DirectKV=" << config.attnNhwcCoopmat1DirectKV << "\n";
  out << "attnNhwcCoopmat1KVChunkCount=" << config.attnNhwcCoopmat1KVChunkCount << "\n";
  out << "attnNhwcCoopmat1SplitKTunerValueValid=" << config.attnNhwcCoopmat1SplitKTunerValueValid << "\n";
  out << "attnNhwcCoopmat1SplitKBlockSize=" << config.attnNhwcCoopmat1SplitKBlockSize << "\n";
  out << "attnNhwcCoopmat1SplitKBlockQ=" << config.attnNhwcCoopmat1SplitKBlockQ << "\n";
  out << "attnNhwcCoopmat1SplitKBlockKV=" << config.attnNhwcCoopmat1SplitKBlockKV << "\n";
  out << "attnNhwcCoopmat1SplitKTM=" << config.attnNhwcCoopmat1SplitKTM << "\n";
  out << "attnNhwcCoopmat1SplitKTN=" << config.attnNhwcCoopmat1SplitKTN << "\n";
  out << "attnNhwcCoopmat1SplitKTK=" << config.attnNhwcCoopmat1SplitKTK << "\n";
  out << "attnNhwcCoopmat1SplitKWarp=" << config.attnNhwcCoopmat1SplitKWarp << "\n";
  out << "attnNhwcCoopmat1SplitKDirectKV=" << config.attnNhwcCoopmat1SplitKDirectKV << "\n";
  out << "attnNhwcCoopmat1SplitKKVChunkCount=" << config.attnNhwcCoopmat1SplitKKVChunkCount << "\n";
  out << "attnNhwcCoopmat1SplitKCutoffBatch=" << config.attnNhwcCoopmat1SplitKCutoffBatch << "\n";
  out << "attnNhwcCoopmatMaintenance1TunerValueValid=" << config.attnNhwcCoopmatMaintenance1TunerValueValid << "\n";
  out << "attnNhwcUseCoopmatMaintenance1=" << config.attnNhwcUseCoopmatMaintenance1 << "\n";
  out << "attnNhwcCoopmatMaintenance1TimeUs=" << config.attnNhwcCoopmatMaintenance1TimeUs << "\n";
  out << "attnNhwcCoopmatMaintenance1BlockSize=" << config.attnNhwcCoopmatMaintenance1BlockSize << "\n";
  out << "attnNhwcCoopmatMaintenance1BlockQ=" << config.attnNhwcCoopmatMaintenance1BlockQ << "\n";
  out << "attnNhwcCoopmatMaintenance1BlockKV=" << config.attnNhwcCoopmatMaintenance1BlockKV << "\n";
  out << "attnNhwcCoopmatMaintenance1TM=" << config.attnNhwcCoopmatMaintenance1TM << "\n";
  out << "attnNhwcCoopmatMaintenance1TN=" << config.attnNhwcCoopmatMaintenance1TN << "\n";
  out << "attnNhwcCoopmatMaintenance1TK=" << config.attnNhwcCoopmatMaintenance1TK << "\n";
  out << "attnNhwcCoopmatMaintenance1Warp=" << config.attnNhwcCoopmatMaintenance1Warp << "\n";
  out << "attnNhwcCoopmatMaintenance1DirectKV=" << config.attnNhwcCoopmatMaintenance1DirectKV << "\n";
  out << "attnNhwcCoopmatMaintenance1SplitKTunerValueValid=" << config.attnNhwcCoopmatMaintenance1SplitKTunerValueValid << "\n";
  out << "attnNhwcCoopmatMaintenance1SplitKBlockSize=" << config.attnNhwcCoopmatMaintenance1SplitKBlockSize << "\n";
  out << "attnNhwcCoopmatMaintenance1SplitKBlockQ=" << config.attnNhwcCoopmatMaintenance1SplitKBlockQ << "\n";
  out << "attnNhwcCoopmatMaintenance1SplitKBlockKV=" << config.attnNhwcCoopmatMaintenance1SplitKBlockKV << "\n";
  out << "attnNhwcCoopmatMaintenance1SplitKTM=" << config.attnNhwcCoopmatMaintenance1SplitKTM << "\n";
  out << "attnNhwcCoopmatMaintenance1SplitKTN=" << config.attnNhwcCoopmatMaintenance1SplitKTN << "\n";
  out << "attnNhwcCoopmatMaintenance1SplitKTK=" << config.attnNhwcCoopmatMaintenance1SplitKTK << "\n";
  out << "attnNhwcCoopmatMaintenance1SplitKWarp=" << config.attnNhwcCoopmatMaintenance1SplitKWarp << "\n";
  out << "attnNhwcCoopmatMaintenance1SplitKDirectKV=" << config.attnNhwcCoopmatMaintenance1SplitKDirectKV << "\n";
  out << "attnNhwcCoopmatMaintenance1SplitKKVChunkCount=" << config.attnNhwcCoopmatMaintenance1SplitKKVChunkCount << "\n";
  out << "attnNhwcCoopmatMaintenance1SplitKCutoffBatch=" << config.attnNhwcCoopmatMaintenance1SplitKCutoffBatch << "\n";
  out << "attnNhwcCoopmat2TunerValueValid=" << config.attnNhwcCoopmat2TunerValueValid << "\n";
  out << "attnNhwcUseCoopmat2=" << config.attnNhwcUseCoopmat2 << "\n";
  out << "attnNhwcCoopmat2TimeUs=" << config.attnNhwcCoopmat2TimeUs << "\n";
  out << "attnNhwcCoopmat2BlockSize=" << config.attnNhwcCoopmat2BlockSize << "\n";
  out << "attnNhwcCoopmat2BlockQ=" << config.attnNhwcCoopmat2BlockQ << "\n";
  out << "attnNhwcCoopmat2BlockKV=" << config.attnNhwcCoopmat2BlockKV << "\n";
  out << "attnNhwcDot2TunerValueValid=" << config.attnNhwcDot2TunerValueValid << "\n";
  out << "attnNhwcUseDot2=" << config.attnNhwcUseDot2 << "\n";
  out << "attnNhwcDot2TimeUs=" << config.attnNhwcDot2TimeUs << "\n";
  out << "attnNhwcDot2BlockSize=" << config.attnNhwcDot2BlockSize << "\n";
  out << "attnNhwcDot2BlockQ=" << config.attnNhwcDot2BlockQ << "\n";
  out << "attnNhwcDot2BlockKV=" << config.attnNhwcDot2BlockKV << "\n";
  finishAtomicSave(filename, tmpFilename, out);
}

namespace {

  struct TuneParamsLoadState {
    uint32_t nhwcTiledGemmFields = 0;
    uint32_t nchwToNhwcFields = 0;
  };

  void fillFromDesc(
    const string& filename,
    const string& desc,
    VulkanTuneParams& config,
    TuneParamsLoadState& loadState) {
    istringstream in(desc);
    string token;
    while(in >> token) {
      size_t eq = token.find('=');
      if(eq == string::npos)
        throw IOError("VulkanTuneParams::load: bad token '" + token + "' in " + filename);
      string key = token.substr(0, eq);
      string valueStr = token.substr(eq + 1);
      if(key == "nhwcWinogradTransformTunedLayout")
        config.nhwcWinogradTransformTunedLayout = valueStr;
      else if(key == "gemmVariantFilterTunedSignature")
        config.gemmVariantFilterTunedSignature = valueStr;
      else {
        int value = Global::stringToInt(valueStr);
        if(key == "tunedKernelMask")
          config.tunedKernelMask = value;
        else if(key == "winogradGemmCoopmat1F32TimeUs")
          config.winogradGemmCoopmat1F32TimeUs = value;
        else if(key == "winogradGemmCoopmat2F32TimeUs")
          config.winogradGemmCoopmat2F32TimeUs = value;
        else if(key == "winogradGemmCoopmat1F16TimeUs")
          config.winogradGemmCoopmat1F16TimeUs = value;
        else if(key == "winogradGemmCoopmat2F16TimeUs")
          config.winogradGemmCoopmat2F16TimeUs = value;
        else if(key == "winogradGemmDot2F32TimeUs")
          config.winogradGemmDot2F32TimeUs = value;
        else if(key == "winogradGemmDot2F16TimeUs")
          config.winogradGemmDot2F16TimeUs = value;
        else if(key == "nchwToNhwcSmallTile") {
          config.nchwToNhwcSmallTile = value;
          loadState.nchwToNhwcFields |= 1u << 0;
        }
        else if(key == "nchwToNhwcLargeTile") {
          config.nchwToNhwcLargeTile = value;
          loadState.nchwToNhwcFields |= 1u << 1;
        }
        else if(key == "nchwToNhwcTileCrossover") {
          config.nchwToNhwcTileCrossover = value;
          loadState.nchwToNhwcFields |= 1u << 2;
        }
        else if(key == "nhwcWinogradTransformLocalSizeX")
          config.nhwcWinogradTransformLocalSizeX = value;
        else if(key == "nhwcWinogradTransformLocalSizeY")
          config.nhwcWinogradTransformLocalSizeY = value;
        else if(key == "nhwcWinogradUntransformLocalSizeX")
          config.nhwcWinogradUntransformLocalSizeX = value;
        else if(key == "nhwcWinogradUntransformLocalSizeY")
          config.nhwcWinogradUntransformLocalSizeY = value;
        else if(key == "nhwcWinograd3x3OutTile")
          config.nhwcWinograd3x3OutTile = value;
        else if(key == "conv3x3NhwcWinogradTunerValueValid")
          config.conv3x3NhwcWinogradTunerValueValid = value;
        else if(key == "conv5x5NhwcWinogradTunerValueValid")
          config.conv5x5NhwcWinogradTunerValueValid = value;
        else if(key == "gemmStridedTiledNhwcLocalSizeX") {
          config.gemmStridedTiledNhwcLocalSizeX = value;
          loadState.nhwcTiledGemmFields |= 1u << 0;
        }
        else if(key == "gemmStridedTiledNhwcLocalSizeY") {
          config.gemmStridedTiledNhwcLocalSizeY = value;
          loadState.nhwcTiledGemmFields |= 1u << 1;
        }
        else if(key == "gemmStridedTiledNhwcTileK") {
          config.gemmStridedTiledNhwcTileK = value;
          loadState.nhwcTiledGemmFields |= 1u << 2;
        }
        else if(key == "gemmStridedTiledNhwcRN") {
          config.gemmStridedTiledNhwcRN = value;
          loadState.nhwcTiledGemmFields |= 1u << 3;
        }
        else if(key == "gemmDirectLocalSizeX")
          config.gemmDirectLocalSizeX = value;
        else if(key == "gemmDirectLocalSizeY")
          config.gemmDirectLocalSizeY = value;
        else if(key == "winogradGemmM")
          config.winogradGemmM = value;
        else if(key == "winogradGemmN")
          config.winogradGemmN = value;
        else if(key == "winogradGemmK")
          config.winogradGemmK = value;
        else if(key == "winogradGemmRN")
          config.winogradGemmRN = value;
        else if(key == "enableWinogradGemmDot2")
          config.enableWinogradGemmDot2 = value;
        else if(key == "enableWinogradGemmDot2AccF16")
          config.enableWinogradGemmDot2AccF16 = value;
        else if(key == "winogradGemmDot2TunerValueValid")
          config.winogradGemmDot2TunerValueValid = value;
        else if(key == "winogradGemmDot2AccF16TunerValueValid")
          config.winogradGemmDot2AccF16TunerValueValid = value;
        else if(key == "dot2BlockSize")
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
        else if(key == "enableWinogradGemmCoopmat1")
          config.enableWinogradGemmCoopmat1 = value;
        else if(key == "winogradGemmCoopmat1TunerValueValid")
          config.winogradGemmCoopmat1TunerValueValid = value;
        else if(key == "coopmat1BlockSize")
          config.coopmat1BlockSize = value;
        else if(key == "coopmat1BM")
          config.coopmat1BM = value;
        else if(key == "coopmat1BN")
          config.coopmat1BN = value;
        else if(key == "coopmat1BK")
          config.coopmat1BK = value;
        else if(key == "coopmat1WM")
          config.coopmat1WM = value;
        else if(key == "coopmat1WN")
          config.coopmat1WN = value;
        else if(key == "coopmat1TM")
          config.coopmat1TM = value;
        else if(key == "coopmat1TN")
          config.coopmat1TN = value;
        else if(key == "coopmat1TK")
          config.coopmat1TK = value;
        else if(key == "coopmat1Warp")
          config.coopmat1Warp = value;
        else if(key == "conv3x3NhwcCoopmat1TunerValueValid")
          config.conv3x3NhwcCoopmat1TunerValueValid = value;
        else if(key == "conv3x3NhwcCoopmat1BlockSize")
          config.conv3x3NhwcCoopmat1BlockSize = value;
        else if(key == "conv3x3NhwcCoopmat1BM")
          config.conv3x3NhwcCoopmat1BM = value;
        else if(key == "conv3x3NhwcCoopmat1BN")
          config.conv3x3NhwcCoopmat1BN = value;
        else if(key == "conv3x3NhwcCoopmat1BK")
          config.conv3x3NhwcCoopmat1BK = value;
        else if(key == "conv3x3NhwcCoopmat1WM")
          config.conv3x3NhwcCoopmat1WM = value;
        else if(key == "conv3x3NhwcCoopmat1WN")
          config.conv3x3NhwcCoopmat1WN = value;
        else if(key == "conv3x3NhwcCoopmat1TM")
          config.conv3x3NhwcCoopmat1TM = value;
        else if(key == "conv3x3NhwcCoopmat1TN")
          config.conv3x3NhwcCoopmat1TN = value;
        else if(key == "conv3x3NhwcCoopmat1TK")
          config.conv3x3NhwcCoopmat1TK = value;
        else if(key == "conv3x3NhwcCoopmat1Warp")
          config.conv3x3NhwcCoopmat1Warp = value;
        else if(key == "conv5x5NhwcCoopmat1TunerValueValid")
          config.conv5x5NhwcCoopmat1TunerValueValid = value;
        else if(key == "conv5x5NhwcCoopmat1BlockSize")
          config.conv5x5NhwcCoopmat1BlockSize = value;
        else if(key == "conv5x5NhwcCoopmat1BM")
          config.conv5x5NhwcCoopmat1BM = value;
        else if(key == "conv5x5NhwcCoopmat1BN")
          config.conv5x5NhwcCoopmat1BN = value;
        else if(key == "conv5x5NhwcCoopmat1BK")
          config.conv5x5NhwcCoopmat1BK = value;
        else if(key == "conv5x5NhwcCoopmat1WM")
          config.conv5x5NhwcCoopmat1WM = value;
        else if(key == "conv5x5NhwcCoopmat1WN")
          config.conv5x5NhwcCoopmat1WN = value;
        else if(key == "conv5x5NhwcCoopmat1TM")
          config.conv5x5NhwcCoopmat1TM = value;
        else if(key == "conv5x5NhwcCoopmat1TN")
          config.conv5x5NhwcCoopmat1TN = value;
        else if(key == "conv5x5NhwcCoopmat1TK")
          config.conv5x5NhwcCoopmat1TK = value;
        else if(key == "conv5x5NhwcCoopmat1Warp")
          config.conv5x5NhwcCoopmat1Warp = value;
        else if(key == "conv3x3NhwcCoopmat1AccF16TunerValueValid")
          config.conv3x3NhwcCoopmat1AccF16TunerValueValid = value;
        else if(key == "conv3x3NhwcCoopmat1AccF16BlockSize")
          config.conv3x3NhwcCoopmat1AccF16BlockSize = value;
        else if(key == "conv3x3NhwcCoopmat1AccF16BM")
          config.conv3x3NhwcCoopmat1AccF16BM = value;
        else if(key == "conv3x3NhwcCoopmat1AccF16BN")
          config.conv3x3NhwcCoopmat1AccF16BN = value;
        else if(key == "conv3x3NhwcCoopmat1AccF16BK")
          config.conv3x3NhwcCoopmat1AccF16BK = value;
        else if(key == "conv3x3NhwcCoopmat1AccF16WM")
          config.conv3x3NhwcCoopmat1AccF16WM = value;
        else if(key == "conv3x3NhwcCoopmat1AccF16WN")
          config.conv3x3NhwcCoopmat1AccF16WN = value;
        else if(key == "conv3x3NhwcCoopmat1AccF16TM")
          config.conv3x3NhwcCoopmat1AccF16TM = value;
        else if(key == "conv3x3NhwcCoopmat1AccF16TN")
          config.conv3x3NhwcCoopmat1AccF16TN = value;
        else if(key == "conv3x3NhwcCoopmat1AccF16TK")
          config.conv3x3NhwcCoopmat1AccF16TK = value;
        else if(key == "conv3x3NhwcCoopmat1AccF16Warp")
          config.conv3x3NhwcCoopmat1AccF16Warp = value;
        else if(key == "conv5x5NhwcCoopmat1AccF16TunerValueValid")
          config.conv5x5NhwcCoopmat1AccF16TunerValueValid = value;
        else if(key == "conv5x5NhwcCoopmat1AccF16BlockSize")
          config.conv5x5NhwcCoopmat1AccF16BlockSize = value;
        else if(key == "conv5x5NhwcCoopmat1AccF16BM")
          config.conv5x5NhwcCoopmat1AccF16BM = value;
        else if(key == "conv5x5NhwcCoopmat1AccF16BN")
          config.conv5x5NhwcCoopmat1AccF16BN = value;
        else if(key == "conv5x5NhwcCoopmat1AccF16BK")
          config.conv5x5NhwcCoopmat1AccF16BK = value;
        else if(key == "conv5x5NhwcCoopmat1AccF16WM")
          config.conv5x5NhwcCoopmat1AccF16WM = value;
        else if(key == "conv5x5NhwcCoopmat1AccF16WN")
          config.conv5x5NhwcCoopmat1AccF16WN = value;
        else if(key == "conv5x5NhwcCoopmat1AccF16TM")
          config.conv5x5NhwcCoopmat1AccF16TM = value;
        else if(key == "conv5x5NhwcCoopmat1AccF16TN")
          config.conv5x5NhwcCoopmat1AccF16TN = value;
        else if(key == "conv5x5NhwcCoopmat1AccF16TK")
          config.conv5x5NhwcCoopmat1AccF16TK = value;
        else if(key == "conv5x5NhwcCoopmat1AccF16Warp")
          config.conv5x5NhwcCoopmat1AccF16Warp = value;
        else if(key == "enableWinogradGemmCoopmat1AccF16")
          config.enableWinogradGemmCoopmat1AccF16 = value;
        else if(key == "winogradGemmCoopmat1AccF16TunerValueValid")
          config.winogradGemmCoopmat1AccF16TunerValueValid = value;
        else if(key == "coopmat1AccF16BlockSize")
          config.coopmat1AccF16BlockSize = value;
        else if(key == "coopmat1AccF16BM")
          config.coopmat1AccF16BM = value;
        else if(key == "coopmat1AccF16BN")
          config.coopmat1AccF16BN = value;
        else if(key == "coopmat1AccF16BK")
          config.coopmat1AccF16BK = value;
        else if(key == "coopmat1AccF16WM")
          config.coopmat1AccF16WM = value;
        else if(key == "coopmat1AccF16WN")
          config.coopmat1AccF16WN = value;
        else if(key == "coopmat1AccF16TM")
          config.coopmat1AccF16TM = value;
        else if(key == "coopmat1AccF16TN")
          config.coopmat1AccF16TN = value;
        else if(key == "coopmat1AccF16TK")
          config.coopmat1AccF16TK = value;
        else if(key == "coopmat1AccF16Warp")
          config.coopmat1AccF16Warp = value;
        else if(key == "gemmStridedNhwcCoopmat1TunerValueValid")
          config.gemmStridedNhwcCoopmat1TunerValueValid = value;
        else if(key == "nhwcStridedCoopmat1BlockSize")
          config.nhwcStridedCoopmat1BlockSize = value;
        else if(key == "nhwcStridedCoopmat1BM")
          config.nhwcStridedCoopmat1BM = value;
        else if(key == "nhwcStridedCoopmat1BN")
          config.nhwcStridedCoopmat1BN = value;
        else if(key == "nhwcStridedCoopmat1BK")
          config.nhwcStridedCoopmat1BK = value;
        else if(key == "nhwcStridedCoopmat1WM")
          config.nhwcStridedCoopmat1WM = value;
        else if(key == "nhwcStridedCoopmat1WN")
          config.nhwcStridedCoopmat1WN = value;
        else if(key == "nhwcStridedCoopmat1TM")
          config.nhwcStridedCoopmat1TM = value;
        else if(key == "nhwcStridedCoopmat1TN")
          config.nhwcStridedCoopmat1TN = value;
        else if(key == "nhwcStridedCoopmat1TK")
          config.nhwcStridedCoopmat1TK = value;
        else if(key == "nhwcStridedCoopmat1Warp")
          config.nhwcStridedCoopmat1Warp = value;
        else if(key == "gemmStridedNhwcCoopmat1AccF16TunerValueValid")
          config.gemmStridedNhwcCoopmat1AccF16TunerValueValid = value;
        else if(key == "nhwcStridedCoopmat1AccF16BlockSize")
          config.nhwcStridedCoopmat1AccF16BlockSize = value;
        else if(key == "nhwcStridedCoopmat1AccF16BM")
          config.nhwcStridedCoopmat1AccF16BM = value;
        else if(key == "nhwcStridedCoopmat1AccF16BN")
          config.nhwcStridedCoopmat1AccF16BN = value;
        else if(key == "nhwcStridedCoopmat1AccF16BK")
          config.nhwcStridedCoopmat1AccF16BK = value;
        else if(key == "nhwcStridedCoopmat1AccF16WM")
          config.nhwcStridedCoopmat1AccF16WM = value;
        else if(key == "nhwcStridedCoopmat1AccF16WN")
          config.nhwcStridedCoopmat1AccF16WN = value;
        else if(key == "nhwcStridedCoopmat1AccF16TM")
          config.nhwcStridedCoopmat1AccF16TM = value;
        else if(key == "nhwcStridedCoopmat1AccF16TN")
          config.nhwcStridedCoopmat1AccF16TN = value;
        else if(key == "nhwcStridedCoopmat1AccF16TK")
          config.nhwcStridedCoopmat1AccF16TK = value;
        else if(key == "nhwcStridedCoopmat1AccF16Warp")
          config.nhwcStridedCoopmat1AccF16Warp = value;
        else if(key == "gemmStridedNhwcCoopmat2TunerValueValid")
          config.gemmStridedNhwcCoopmat2TunerValueValid = value;
        else if(key == "nhwcStridedCoopmat2BlockSize")
          config.nhwcStridedCoopmat2BlockSize = value;
        else if(key == "nhwcStridedCoopmat2BM")
          config.nhwcStridedCoopmat2BM = value;
        else if(key == "nhwcStridedCoopmat2BN")
          config.nhwcStridedCoopmat2BN = value;
        else if(key == "nhwcStridedCoopmat2BK")
          config.nhwcStridedCoopmat2BK = value;
        else if(key == "gemmStridedNhwcCoopmat2AccF16TunerValueValid")
          config.gemmStridedNhwcCoopmat2AccF16TunerValueValid = value;
        else if(key == "nhwcStridedCoopmat2AccF16BlockSize")
          config.nhwcStridedCoopmat2AccF16BlockSize = value;
        else if(key == "nhwcStridedCoopmat2AccF16BM")
          config.nhwcStridedCoopmat2AccF16BM = value;
        else if(key == "nhwcStridedCoopmat2AccF16BN")
          config.nhwcStridedCoopmat2AccF16BN = value;
        else if(key == "nhwcStridedCoopmat2AccF16BK")
          config.nhwcStridedCoopmat2AccF16BK = value;
        else if(key == "nhwcGemmF32UseCoopmat2")
          config.nhwcGemmF32UseCoopmat2 = value;
        else if(key == "nhwcGemmUseCoopmat2")
          config.nhwcGemmUseCoopmat2 = value;
        else if(key == "nhwcGemmUseCoopmatAccF16")
          config.nhwcGemmUseCoopmatAccF16 = value;
        else if(key == "nhwcGemmCoopmat1F32TimeUs")
          config.nhwcGemmCoopmat1F32TimeUs = value;
        else if(key == "nhwcGemmCoopmat2F32TimeUs")
          config.nhwcGemmCoopmat2F32TimeUs = value;
        else if(key == "nhwcGemmCoopmat1F16TimeUs")
          config.nhwcGemmCoopmat1F16TimeUs = value;
        else if(key == "nhwcGemmCoopmat2F16TimeUs")
          config.nhwcGemmCoopmat2F16TimeUs = value;
        else if(key == "gemmStridedNhwcDot2TunerValueValid")
          config.gemmStridedNhwcDot2TunerValueValid = value;
        else if(key == "nhwcStridedDot2BlockSize")
          config.nhwcStridedDot2BlockSize = value;
        else if(key == "nhwcStridedDot2BM")
          config.nhwcStridedDot2BM = value;
        else if(key == "nhwcStridedDot2BN")
          config.nhwcStridedDot2BN = value;
        else if(key == "nhwcStridedDot2WM")
          config.nhwcStridedDot2WM = value;
        else if(key == "nhwcStridedDot2WN")
          config.nhwcStridedDot2WN = value;
        else if(key == "nhwcStridedDot2WMIter")
          config.nhwcStridedDot2WMIter = value;
        else if(key == "nhwcStridedDot2TM")
          config.nhwcStridedDot2TM = value;
        else if(key == "nhwcStridedDot2TN")
          config.nhwcStridedDot2TN = value;
        else if(key == "nhwcStridedDot2Warp")
          config.nhwcStridedDot2Warp = value;
        else if(key == "nhwcGemmUseDot2")
          config.nhwcGemmUseDot2 = value;
        else if(key == "gemmStridedNhwcDot2AccF16TunerValueValid")
          config.gemmStridedNhwcDot2AccF16TunerValueValid = value;
        else if(key == "nhwcStridedDot2AccF16BlockSize")
          config.nhwcStridedDot2AccF16BlockSize = value;
        else if(key == "nhwcStridedDot2AccF16BM")
          config.nhwcStridedDot2AccF16BM = value;
        else if(key == "nhwcStridedDot2AccF16BN")
          config.nhwcStridedDot2AccF16BN = value;
        else if(key == "nhwcStridedDot2AccF16WM")
          config.nhwcStridedDot2AccF16WM = value;
        else if(key == "nhwcStridedDot2AccF16WN")
          config.nhwcStridedDot2AccF16WN = value;
        else if(key == "nhwcStridedDot2AccF16WMIter")
          config.nhwcStridedDot2AccF16WMIter = value;
        else if(key == "nhwcStridedDot2AccF16TM")
          config.nhwcStridedDot2AccF16TM = value;
        else if(key == "nhwcStridedDot2AccF16TN")
          config.nhwcStridedDot2AccF16TN = value;
        else if(key == "nhwcStridedDot2AccF16Warp")
          config.nhwcStridedDot2AccF16Warp = value;
        else if(key == "nhwcGemmUseDot2AccF16")
          config.nhwcGemmUseDot2AccF16 = value;
        else if(key == "nhwcGemmDot2F32TimeUs")
          config.nhwcGemmDot2F32TimeUs = value;
        else if(key == "nhwcGemmDot2F16TimeUs")
          config.nhwcGemmDot2F16TimeUs = value;
        else if(key == "conv3x3NhwcCoopmat2TunerValueValid")
          config.conv3x3NhwcCoopmat2TunerValueValid = value;
        else if(key == "conv3x3NhwcCoopmat2BlockSize")
          config.conv3x3NhwcCoopmat2BlockSize = value;
        else if(key == "conv3x3NhwcCoopmat2BM")
          config.conv3x3NhwcCoopmat2BM = value;
        else if(key == "conv3x3NhwcCoopmat2BN")
          config.conv3x3NhwcCoopmat2BN = value;
        else if(key == "conv3x3NhwcCoopmat2BK")
          config.conv3x3NhwcCoopmat2BK = value;
        else if(key == "nhwcConv3x3UseCoopmat2")
          config.nhwcConv3x3UseCoopmat2 = value;
        else if(key == "conv5x5NhwcCoopmat2TunerValueValid")
          config.conv5x5NhwcCoopmat2TunerValueValid = value;
        else if(key == "conv5x5NhwcCoopmat2BlockSize")
          config.conv5x5NhwcCoopmat2BlockSize = value;
        else if(key == "conv5x5NhwcCoopmat2BM")
          config.conv5x5NhwcCoopmat2BM = value;
        else if(key == "conv5x5NhwcCoopmat2BN")
          config.conv5x5NhwcCoopmat2BN = value;
        else if(key == "conv5x5NhwcCoopmat2BK")
          config.conv5x5NhwcCoopmat2BK = value;
        else if(key == "nhwcConv5x5UseCoopmat2")
          config.nhwcConv5x5UseCoopmat2 = value;
        else if(key == "conv3x3NhwcCoopmat2AccF16TunerValueValid")
          config.conv3x3NhwcCoopmat2AccF16TunerValueValid = value;
        else if(key == "conv3x3NhwcCoopmat2AccF16BlockSize")
          config.conv3x3NhwcCoopmat2AccF16BlockSize = value;
        else if(key == "conv3x3NhwcCoopmat2AccF16BM")
          config.conv3x3NhwcCoopmat2AccF16BM = value;
        else if(key == "conv3x3NhwcCoopmat2AccF16BN")
          config.conv3x3NhwcCoopmat2AccF16BN = value;
        else if(key == "conv3x3NhwcCoopmat2AccF16BK")
          config.conv3x3NhwcCoopmat2AccF16BK = value;
        else if(key == "conv5x5NhwcCoopmat2AccF16TunerValueValid")
          config.conv5x5NhwcCoopmat2AccF16TunerValueValid = value;
        else if(key == "conv5x5NhwcCoopmat2AccF16BlockSize")
          config.conv5x5NhwcCoopmat2AccF16BlockSize = value;
        else if(key == "conv5x5NhwcCoopmat2AccF16BM")
          config.conv5x5NhwcCoopmat2AccF16BM = value;
        else if(key == "conv5x5NhwcCoopmat2AccF16BN")
          config.conv5x5NhwcCoopmat2AccF16BN = value;
        else if(key == "conv5x5NhwcCoopmat2AccF16BK")
          config.conv5x5NhwcCoopmat2AccF16BK = value;
        else if(key == "enableWinogradGemmCoopmat2")
          config.enableWinogradGemmCoopmat2 = value;
        else if(key == "winogradGemmCoopmat2TunerValueValid")
          config.winogradGemmCoopmat2TunerValueValid = value;
        else if(key == "coopmat2BlockSize")
          config.coopmat2BlockSize = value;
        else if(key == "coopmat2BM")
          config.coopmat2BM = value;
        else if(key == "coopmat2BN")
          config.coopmat2BN = value;
        else if(key == "coopmat2BK")
          config.coopmat2BK = value;
        else if(key == "enableWinogradGemmCoopmat2AccF16")
          config.enableWinogradGemmCoopmat2AccF16 = value;
        else if(key == "winogradGemmCoopmat2AccF16TunerValueValid")
          config.winogradGemmCoopmat2AccF16TunerValueValid = value;
        else if(key == "coopmat2AccF16BlockSize")
          config.coopmat2AccF16BlockSize = value;
        else if(key == "coopmat2AccF16BM")
          config.coopmat2AccF16BM = value;
        else if(key == "coopmat2AccF16BN")
          config.coopmat2AccF16BN = value;
        else if(key == "coopmat2AccF16BK")
          config.coopmat2AccF16BK = value;
        else if(key == "gpoolNhwcXystride")
          config.gpoolNhwcXystride = value;
        else if(key == "valueHeadPoolNhwcXystride")
          config.valueHeadPoolNhwcXystride = value;
        else if(key == "spatialRMSNormNhwcTile")
          config.spatialRMSNormNhwcTile = value;
        else if(key == "swiGLUTunerValueValid")
          config.swiGLUTunerValueValid = value;
        else if(key == "swiGLULocalSizeX")
          config.swiGLULocalSizeX = value;
        else if(key == "attnBlockQ")
          config.attnBlockQ = value;
        else if(key == "attnBlockKV")
          config.attnBlockKV = value;
        else if(key == "attnQPerThread")
          config.attnQPerThread = value;
        else if(key == "attnNhwcBlockQ")
          config.attnNhwcBlockQ = value;
        else if(key == "attnNhwcBlockKV")
          config.attnNhwcBlockKV = value;
        else if(key == "attnNhwcQPerThread")
          config.attnNhwcQPerThread = value;
        else if(key == "attnNhwcVariantPolicyVersion")
          config.attnNhwcVariantPolicyVersion = value;
        else if(key == "attnNhwcCoopmat1TunerValueValid")
          config.attnNhwcCoopmat1TunerValueValid = value;
        else if(key == "attnNhwcUseCoopmat1")
          config.attnNhwcUseCoopmat1 = value;
        else if(key == "attnNhwcTiledTimeUs")
          config.attnNhwcTiledTimeUs = value;
        else if(key == "attnNhwcCoopmat1TimeUs")
          config.attnNhwcCoopmat1TimeUs = value;
        else if(key == "attnNhwcCoopmat1BlockSize")
          config.attnNhwcCoopmat1BlockSize = value;
        else if(key == "attnNhwcCoopmat1BlockQ")
          config.attnNhwcCoopmat1BlockQ = value;
        else if(key == "attnNhwcCoopmat1BlockKV")
          config.attnNhwcCoopmat1BlockKV = value;
        else if(key == "attnNhwcCoopmat1TM")
          config.attnNhwcCoopmat1TM = value;
        else if(key == "attnNhwcCoopmat1TN")
          config.attnNhwcCoopmat1TN = value;
        else if(key == "attnNhwcCoopmat1TK")
          config.attnNhwcCoopmat1TK = value;
        else if(key == "attnNhwcCoopmat1Warp")
          config.attnNhwcCoopmat1Warp = value;
        else if(key == "attnNhwcCoopmat1DirectKV")
          config.attnNhwcCoopmat1DirectKV = value;
        else if(key == "attnNhwcCoopmat1KVChunkCount")
          config.attnNhwcCoopmat1KVChunkCount = value;
        else if(key == "attnNhwcCoopmat1SplitKTunerValueValid")
          config.attnNhwcCoopmat1SplitKTunerValueValid = value;
        else if(key == "attnNhwcCoopmat1SplitKBlockSize")
          config.attnNhwcCoopmat1SplitKBlockSize = value;
        else if(key == "attnNhwcCoopmat1SplitKBlockQ")
          config.attnNhwcCoopmat1SplitKBlockQ = value;
        else if(key == "attnNhwcCoopmat1SplitKBlockKV")
          config.attnNhwcCoopmat1SplitKBlockKV = value;
        else if(key == "attnNhwcCoopmat1SplitKTM")
          config.attnNhwcCoopmat1SplitKTM = value;
        else if(key == "attnNhwcCoopmat1SplitKTN")
          config.attnNhwcCoopmat1SplitKTN = value;
        else if(key == "attnNhwcCoopmat1SplitKTK")
          config.attnNhwcCoopmat1SplitKTK = value;
        else if(key == "attnNhwcCoopmat1SplitKWarp")
          config.attnNhwcCoopmat1SplitKWarp = value;
        else if(key == "attnNhwcCoopmat1SplitKDirectKV")
          config.attnNhwcCoopmat1SplitKDirectKV = value;
        else if(key == "attnNhwcCoopmat1SplitKKVChunkCount")
          config.attnNhwcCoopmat1SplitKKVChunkCount = value;
        else if(key == "attnNhwcCoopmat1SplitKCutoffBatch")
          config.attnNhwcCoopmat1SplitKCutoffBatch = value;
        else if(key == "attnNhwcCoopmatMaintenance1TunerValueValid")
          config.attnNhwcCoopmatMaintenance1TunerValueValid = value;
        else if(key == "attnNhwcUseCoopmatMaintenance1")
          config.attnNhwcUseCoopmatMaintenance1 = value;
        else if(key == "attnNhwcCoopmatMaintenance1TimeUs")
          config.attnNhwcCoopmatMaintenance1TimeUs = value;
        else if(key == "attnNhwcCoopmatMaintenance1BlockSize")
          config.attnNhwcCoopmatMaintenance1BlockSize = value;
        else if(key == "attnNhwcCoopmatMaintenance1BlockQ")
          config.attnNhwcCoopmatMaintenance1BlockQ = value;
        else if(key == "attnNhwcCoopmatMaintenance1BlockKV")
          config.attnNhwcCoopmatMaintenance1BlockKV = value;
        else if(key == "attnNhwcCoopmatMaintenance1TM")
          config.attnNhwcCoopmatMaintenance1TM = value;
        else if(key == "attnNhwcCoopmatMaintenance1TN")
          config.attnNhwcCoopmatMaintenance1TN = value;
        else if(key == "attnNhwcCoopmatMaintenance1TK")
          config.attnNhwcCoopmatMaintenance1TK = value;
        else if(key == "attnNhwcCoopmatMaintenance1Warp")
          config.attnNhwcCoopmatMaintenance1Warp = value;
        else if(key == "attnNhwcCoopmatMaintenance1DirectKV")
          config.attnNhwcCoopmatMaintenance1DirectKV = value;
        else if(key == "attnNhwcCoopmatMaintenance1SplitKTunerValueValid")
          config.attnNhwcCoopmatMaintenance1SplitKTunerValueValid = value;
        else if(key == "attnNhwcCoopmatMaintenance1SplitKBlockSize")
          config.attnNhwcCoopmatMaintenance1SplitKBlockSize = value;
        else if(key == "attnNhwcCoopmatMaintenance1SplitKBlockQ")
          config.attnNhwcCoopmatMaintenance1SplitKBlockQ = value;
        else if(key == "attnNhwcCoopmatMaintenance1SplitKBlockKV")
          config.attnNhwcCoopmatMaintenance1SplitKBlockKV = value;
        else if(key == "attnNhwcCoopmatMaintenance1SplitKTM")
          config.attnNhwcCoopmatMaintenance1SplitKTM = value;
        else if(key == "attnNhwcCoopmatMaintenance1SplitKTN")
          config.attnNhwcCoopmatMaintenance1SplitKTN = value;
        else if(key == "attnNhwcCoopmatMaintenance1SplitKTK")
          config.attnNhwcCoopmatMaintenance1SplitKTK = value;
        else if(key == "attnNhwcCoopmatMaintenance1SplitKWarp")
          config.attnNhwcCoopmatMaintenance1SplitKWarp = value;
        else if(key == "attnNhwcCoopmatMaintenance1SplitKDirectKV")
          config.attnNhwcCoopmatMaintenance1SplitKDirectKV = value;
        else if(key == "attnNhwcCoopmatMaintenance1SplitKKVChunkCount")
          config.attnNhwcCoopmatMaintenance1SplitKKVChunkCount = value;
        else if(key == "attnNhwcCoopmatMaintenance1SplitKCutoffBatch")
          config.attnNhwcCoopmatMaintenance1SplitKCutoffBatch = value;
        else if(key == "attnNhwcCoopmat2TunerValueValid")
          config.attnNhwcCoopmat2TunerValueValid = value;
        else if(key == "attnNhwcUseCoopmat2")
          config.attnNhwcUseCoopmat2 = value;
        else if(key == "attnNhwcCoopmat2TimeUs")
          config.attnNhwcCoopmat2TimeUs = value;
        else if(key == "attnNhwcCoopmat2BlockSize")
          config.attnNhwcCoopmat2BlockSize = value;
        else if(key == "attnNhwcCoopmat2BlockQ")
          config.attnNhwcCoopmat2BlockQ = value;
        else if(key == "attnNhwcCoopmat2BlockKV")
          config.attnNhwcCoopmat2BlockKV = value;
        else if(key == "attnNhwcDot2TunerValueValid")
          config.attnNhwcDot2TunerValueValid = value;
        else if(key == "attnNhwcUseDot2")
          config.attnNhwcUseDot2 = value;
        else if(key == "attnNhwcDot2TimeUs")
          config.attnNhwcDot2TimeUs = value;
        else if(key == "attnNhwcDot2BlockSize")
          config.attnNhwcDot2BlockSize = value;
        else if(key == "attnNhwcDot2BlockQ")
          config.attnNhwcDot2BlockQ = value;
        else if(key == "attnNhwcDot2BlockKV")
          config.attnNhwcDot2BlockKV = value;
        else
          throw IOError("VulkanTuneParams::load: unknown key '" + key + "' in " + filename);
      }
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
  // isValid() at the call site (validates every essential field's value range).
  // We deliberately do NOT hardcode an expected line count: adding a new tuner
  // field is a routine change, and coupling load() to save()'s exact line count
  // was silently rejecting every self-produced tune file whenever the schema
  // grew, forcing a full retune from defaults on every launch.

  VulkanTuneParams config;
  TuneParamsLoadState loadState;
  for(size_t i = 1; i < filteredLines.size(); i++)
    fillFromDesc(filename, filteredLines[i], config, loadState);

  if(loadState.nhwcTiledGemmFields != 0xFu)
    config.tunedKernelMask &= ~VulkanTuner::TUNED_NHWC_STRIDED;
  if(loadState.nchwToNhwcFields != 0x7u) {
    config.tunedKernelMask &= ~VulkanTuner::TUNED_NCHW_TO_NHWC;
  }
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

int32_t VulkanTuner::requiredKernelMask(
  const ModelDesc* modelDesc,
  const VulkanDeviceInfo& deviceInfo,
  bool fp16Storage,
  bool fp16Compute,
  bool useNhwc,
  bool full) {
  if(modelDesc == nullptr)
    return 0;
  const bool hasTransformer = modelHasTransformer(modelDesc);
  (void)useNhwc;
  (void)full;
  bool hasWinogradConvolution = false;
  bool has3x3Convolution = false;
  bool has5x5Convolution = false;
  bool hasStridedGemm = hasTransformer;
  modelDesc->iterConvLayers([&](const ConvLayerDesc& conv) noexcept {
    has3x3Convolution = has3x3Convolution || (conv.convXSize == 3 && conv.convYSize == 3);
    has5x5Convolution = has5x5Convolution || (conv.convXSize == 5 && conv.convYSize == 5);
    hasWinogradConvolution = hasWinogradConvolution ||
      ((conv.convXSize == 3 && conv.convYSize == 3) || (conv.convXSize == 5 && conv.convYSize == 5));
    hasStridedGemm = hasStridedGemm || (conv.convXSize == 1 && conv.convYSize == 1);
  });
  int32_t mask = TUNED_GEMM_DIRECT | TUNED_NCHW_TO_NHWC;
  if(hasTransformer)
    mask |= TUNED_SWIGLU;
  {
    mask |= TUNED_NHWC_REDUCTIONS;
    if(hasTransformer)
      mask |= TUNED_NHWC_TRANSFORMER;
    if(hasTransformer && fp16Storage && fp16Compute &&
       (deviceInfo.supportsCoopmat1F16 || deviceInfo.supportsCoopmat2F16 || deviceInfo.supportsDot2F16))
      mask |= TUNED_NHWC_ATTENTION_VARIANT;
    if(hasStridedGemm)
      mask |= TUNED_NHWC_STRIDED;
    if(hasStridedGemm && fp16Storage && fp16Compute &&
       (deviceInfo.supportsCoopmat1F16 || deviceInfo.supportsCoopmat1F16AccF16 ||
        deviceInfo.supportsCoopmat2F16 || deviceInfo.supportsCoopmat2F16AccF16))
      mask |= TUNED_NHWC_GEMM;
    if(has3x3Convolution)
      mask |= TUNED_NHWC_CONV3X3;
    if(has5x5Convolution)
      mask |= TUNED_NHWC_CONV5X5;
    // Native implicit-im2col is available for both accumulator types. If no
    // cooperative-matrix family remains after filtering, these convolutions
    // use Winograd instead.
    if((has3x3Convolution || has5x5Convolution) &&
       (!deviceInfo.supportsCoopmat1F16 && !deviceInfo.supportsCoopmat1F16AccF16 &&
        !deviceInfo.supportsCoopmat2F16 && !deviceInfo.supportsCoopmat2F16AccF16))
      mask |= TUNED_NHWC_WINOGRAD;
  }
  return mask;
}

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
  out << " winogradTransformWG=";
  out << " winogradBNActTransformWG=";
  out << " winogradUntransformWG=";
  out << " nchwToNhwc=";
  appendTuple(
    {config.nchwToNhwcSmallTile,
     config.nchwToNhwcLargeTile,
     config.nchwToNhwcTileCrossover});
  out << " gemmStridedTiledNhwc=";
  appendTuple(
    {config.gemmStridedTiledNhwcLocalSizeX,
     config.gemmStridedTiledNhwcLocalSizeY,
     config.gemmStridedTiledNhwcTileK,
     config.gemmStridedTiledNhwcRN});
  out << " nhwcWinogradTransformWG=";
  appendTuple({config.nhwcWinogradTransformLocalSizeX, config.nhwcWinogradTransformLocalSizeY});
  out << " nhwcWinogradUntransformWG=";
  appendTuple({config.nhwcWinogradUntransformLocalSizeX, config.nhwcWinogradUntransformLocalSizeY});
  out << " nhwcWinograd3x3OutTile=" << config.nhwcWinograd3x3OutTile;
  out << " winogradDot2=" << (config.enableWinogradGemmDot2 != 0 ? "on" : "off")
      << " winogradDot2AccF16=" << (config.enableWinogradGemmDot2AccF16 != 0 ? "on" : "off")
      << " winogradGemmTimesUs=" << config.winogradGemmCoopmat1F32TimeUs << ","
      << config.winogradGemmCoopmat2F32TimeUs << "," << config.winogradGemmCoopmat1F16TimeUs << ","
      << config.winogradGemmCoopmat2F16TimeUs << "," << config.winogradGemmDot2F32TimeUs << ","
      << config.winogradGemmDot2F16TimeUs << " winogradDot2Tile=";
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
  out << " gemmStridedDot2AccF16Tile=";
  out << " winogradCoopmat1=" << (config.enableWinogradGemmCoopmat1 != 0 ? "on" : "off") << " winogradCoopmat1Tile=";
  appendTuple(
    {config.coopmat1BlockSize,
     config.coopmat1BM,
     config.coopmat1BN,
     config.coopmat1BK,
     config.coopmat1WM,
     config.coopmat1WN,
     config.coopmat1TM,
     config.coopmat1TN,
     config.coopmat1TK,
     config.coopmat1Warp});
  out << " conv3x3NhwcCoopmatTile=";
  appendTuple(
    {config.conv3x3NhwcCoopmat1BlockSize,
     config.conv3x3NhwcCoopmat1BM,
     config.conv3x3NhwcCoopmat1BN,
     config.conv3x3NhwcCoopmat1BK,
     config.conv3x3NhwcCoopmat1WM,
     config.conv3x3NhwcCoopmat1WN,
     config.conv3x3NhwcCoopmat1TM,
     config.conv3x3NhwcCoopmat1TN,
     config.conv3x3NhwcCoopmat1TK,
     config.conv3x3NhwcCoopmat1Warp});
  out << " conv5x5NhwcCoopmatTile=";
  appendTuple(
    {config.conv5x5NhwcCoopmat1BlockSize,
     config.conv5x5NhwcCoopmat1BM,
     config.conv5x5NhwcCoopmat1BN,
     config.conv5x5NhwcCoopmat1BK,
     config.conv5x5NhwcCoopmat1WM,
     config.conv5x5NhwcCoopmat1WN,
     config.conv5x5NhwcCoopmat1TM,
     config.conv5x5NhwcCoopmat1TN,
     config.conv5x5NhwcCoopmat1TK,
     config.conv5x5NhwcCoopmat1Warp});
  out << " winogradCoopmat1AccF16=" << (config.enableWinogradGemmCoopmat1AccF16 != 0 ? "on" : "off")
      << " winogradCoopmat1AccF16Tile=";
  appendTuple(
    {config.coopmat1AccF16BlockSize,
     config.coopmat1AccF16BM,
     config.coopmat1AccF16BN,
     config.coopmat1AccF16BK,
     config.coopmat1AccF16WM,
     config.coopmat1AccF16WN,
     config.coopmat1AccF16TM,
     config.coopmat1AccF16TN,
     config.coopmat1AccF16TK,
     config.coopmat1AccF16Warp});
  out << " winogradCoopmat2=" << (config.enableWinogradGemmCoopmat2 != 0 ? "on" : "off") << " winogradCoopmat2Tile=";
  appendTuple({config.coopmat2BlockSize, config.coopmat2BM, config.coopmat2BN, config.coopmat2BK});
  out << " winogradCoopmat2AccF16=" << (config.enableWinogradGemmCoopmat2AccF16 != 0 ? "on" : "off")
      << " winogradCoopmat2AccF16Tile=";
  appendTuple(
    {config.coopmat2AccF16BlockSize, config.coopmat2AccF16BM, config.coopmat2AccF16BN, config.coopmat2AccF16BK});
  out << " gemmDirect=";
  appendTuple({config.gemmDirectLocalSizeX, config.gemmDirectLocalSizeY});
  out << " gpoolNhwc=" << config.gpoolNhwcXystride
      << " valueHeadPoolNhwc=" << config.valueHeadPoolNhwcXystride
      << " tunedKernelMask=" << config.tunedKernelMask;
  out << " gemmStridedNhwcCoopmat1="
      << (config.gemmStridedNhwcCoopmat1TunerValueValid != 0 ? "ready" : "untuned") << " tile=";
  appendTuple(
    {config.nhwcStridedCoopmat1BlockSize,
     config.nhwcStridedCoopmat1BM,
     config.nhwcStridedCoopmat1BN,
     config.nhwcStridedCoopmat1BK});
  out << " gemmStridedNhwcCoopmat1AccF16="
      << (config.gemmStridedNhwcCoopmat1AccF16TunerValueValid != 0 ? "ready" : "untuned") << " tile=";
  appendTuple(
    {config.nhwcStridedCoopmat1AccF16BlockSize,
     config.nhwcStridedCoopmat1AccF16BM,
     config.nhwcStridedCoopmat1AccF16BN,
     config.nhwcStridedCoopmat1AccF16BK});
  out << " gemmStridedNhwcCoopmat2="
      << (config.gemmStridedNhwcCoopmat2TunerValueValid != 0 ? "ready" : "untuned") << " tile=";
  appendTuple({config.nhwcStridedCoopmat2BlockSize,config.nhwcStridedCoopmat2BM,
    config.nhwcStridedCoopmat2BN,config.nhwcStridedCoopmat2BK});
  out << " nhwcGemmF32=" << (config.nhwcGemmF32UseCoopmat2 != 0 ? "coopmat2" : "coopmat1")
      << " nhwcGemmAccF16=" << (config.nhwcGemmUseCoopmat2 != 0 ? "coopmat2" : "coopmat1")
      << " nhwcGemmAccumulation=" << (config.nhwcGemmUseCoopmatAccF16 != 0 ? "fp16" : "fp32")
      << " nhwcGemmTimesUs=" << config.nhwcGemmCoopmat1F32TimeUs << ","
      << config.nhwcGemmCoopmat2F32TimeUs << "," << config.nhwcGemmCoopmat1F16TimeUs << ","
      << config.nhwcGemmCoopmat2F16TimeUs
      << " nhwcGemmCoopmat2AccF16Tile=";
  appendTuple({config.nhwcStridedCoopmat2AccF16BlockSize,config.nhwcStridedCoopmat2AccF16BM,
    config.nhwcStridedCoopmat2AccF16BN,config.nhwcStridedCoopmat2AccF16BK});
  out << " nhwcGemmDot2=" << (config.nhwcGemmUseDot2 != 0 ? "on" : "off") << " nhwcGemmDot2Tile=";
  appendTuple({config.nhwcStridedDot2BlockSize, config.nhwcStridedDot2BM, config.nhwcStridedDot2BN,
    config.nhwcStridedDot2WM, config.nhwcStridedDot2WN});
  out << " nhwcGemmDot2AccF16=" << (config.nhwcGemmUseDot2AccF16 != 0 ? "on" : "off")
      << " nhwcGemmDot2TimesUs=" << config.nhwcGemmDot2F32TimeUs << "," << config.nhwcGemmDot2F16TimeUs
      << " nhwcGemmDot2AccF16Tile=";
  appendTuple({config.nhwcStridedDot2AccF16BlockSize, config.nhwcStridedDot2AccF16BM,
    config.nhwcStridedDot2AccF16BN, config.nhwcStridedDot2AccF16WM, config.nhwcStridedDot2AccF16WN});
  out << " nhwcConv3x3=" << (config.nhwcConv3x3UseCoopmat2 != 0 ? "coopmat2" : "coopmat1")
      << " nhwcConv3x3Coopmat2Tile=";
  appendTuple({config.conv3x3NhwcCoopmat2BlockSize,config.conv3x3NhwcCoopmat2BM,
    config.conv3x3NhwcCoopmat2BN,config.conv3x3NhwcCoopmat2BK});
  out << " nhwcConv5x5=" << (config.nhwcConv5x5UseCoopmat2 != 0 ? "coopmat2" : "coopmat1")
      << " nhwcConv5x5Coopmat2Tile=";
  appendTuple({config.conv5x5NhwcCoopmat2BlockSize,config.conv5x5NhwcCoopmat2BM,
    config.conv5x5NhwcCoopmat2BN,config.conv5x5NhwcCoopmat2BK});
  bool hasTransformer = (modelDesc == nullptr) || modelHasTransformer(modelDesc);
  if(hasTransformer) {
    out << " attnTiled=";
    appendTuple({config.attnBlockQ, config.attnBlockKV, config.attnQPerThread});
    out << " attnTiledNhwc=";
    appendTuple({config.attnNhwcBlockQ, config.attnNhwcBlockKV, config.attnNhwcQPerThread});
    out << " attnNhwcVariant=" << (config.attnNhwcUseCoopmat2 != 0 ? "coopmat2" :
                                     (config.attnNhwcUseCoopmat1 != 0 ? "coopmat1" :
                                     (config.attnNhwcUseDot2 != 0 ? "dot2accf32" : "tiled"))
                                     )
        << " attnNhwcCoopmat1Tile=";
    appendTuple({config.attnNhwcCoopmat1BlockSize, config.attnNhwcCoopmat1BlockQ,
      config.attnNhwcCoopmat1BlockKV, config.attnNhwcCoopmat1TM,
      config.attnNhwcCoopmat1TN, config.attnNhwcCoopmat1TK});
    out << " attnNhwcCoopmat1KV=" << (config.attnNhwcCoopmat1DirectKV != 0 ? "direct" : "staged");
    out << " attnNhwcCoopmat1SplitK="
        << (config.attnNhwcCoopmat1SplitKCutoffBatch > 0 ? "on" : "off") << " tile=";
    appendTuple({config.attnNhwcCoopmat1SplitKBlockSize, config.attnNhwcCoopmat1SplitKBlockQ,
      config.attnNhwcCoopmat1SplitKBlockKV, config.attnNhwcCoopmat1SplitKTM,
      config.attnNhwcCoopmat1SplitKTN, config.attnNhwcCoopmat1SplitKTK});
    out << " chunks=" << config.attnNhwcCoopmat1SplitKKVChunkCount
        << " cutoff=" << config.attnNhwcCoopmat1SplitKCutoffBatch;
    out << " attnNhwcCoopmat2Tile=";
    appendTuple({config.attnNhwcCoopmat2BlockSize, config.attnNhwcCoopmat2BlockQ,
      config.attnNhwcCoopmat2BlockKV});
    out << " attnNhwcDot2Tile=";
    appendTuple({config.attnNhwcDot2BlockSize, config.attnNhwcDot2BlockQ,
      config.attnNhwcDot2BlockKV});
    out << " spatialRMSNormNhwc=" << config.spatialRMSNormNhwcTile
        << " swiGLU=" << (config.swiGLUTunerValueValid != 0 ? Global::intToString(config.swiGLULocalSizeX) : string("untuned"));
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
    TCLAP::SwitchArg fullArg(
      "", "full", "Tune all model-, hardware-, and filter-applicable Vulkan kernel families");
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
  ComputeContext* ctx = NeuralNet::createComputeContextForVulkanTuner(
    gpuIdxs, &logger, nnXLen, nnYLen, homeDataDirOverride, enabled_t::Auto);
  NeuralNet::tuneVulkanComputeContext(
    ctx, &modelDesc, batchSize, winograd3x3OutTile, benchIters, verboseTuner, full, outputFileFromArg);
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
    constexpr int CONV3X3_NHWC_MODE_MAX_CASES = 6;
    constexpr double MODE_SELECT_MIN_SPEEDUP = 1.0;

    struct GemmStridedModeCase {
      int M;
      int N;
      int K;
      int occurrences;
      int overwriteOccurrences;
      int addToOutputOccurrences;
      double weightedWork;
    };

    static void addGemmStridedModeCase(
      std::vector<GemmStridedModeCase>& cases, int M, int rawN, int rawK, bool addToOutput) {
      if(M <= 0 || rawN <= 0 || rawK <= 0)
        return;
      int N = roundUpToMultipleInt(rawN, 8);
      int K = rawK;
      for(auto& c: cases) {
        if(c.M == M && c.N == N && c.K == K) {
          c.occurrences += 1;
          if(addToOutput)
            c.addToOutputOccurrences += 1;
          else
            c.overwriteOccurrences += 1;
          return;
        }
      }
      cases.push_back({M, N, K, 1, addToOutput ? 0 : 1, addToOutput ? 1 : 0, 0.0});
    }

    static void collectResidualAddConvLayers(
      const std::vector<std::pair<int, unique_ptr_void>>& blocks,
      std::vector<const ConvLayerDesc*>& addToOutputConvs) {
      for(const auto& kv: blocks) {
        if(kv.first == ORDINARY_BLOCK_KIND) {
          const auto* block = static_cast<const ResidualBlockDesc*>(kv.second.get());
          addToOutputConvs.push_back(&block->finalConv);
        } else if(kv.first == GLOBAL_POOLING_BLOCK_KIND) {
          const auto* block = static_cast<const GlobalPoolingResidualBlockDesc*>(kv.second.get());
          addToOutputConvs.push_back(&block->finalConv);
        } else if(kv.first == NESTED_BOTTLENECK_BLOCK_KIND) {
          const auto* block = static_cast<const NestedBottleneckResidualBlockDesc*>(kv.second.get());
          addToOutputConvs.push_back(&block->postConv);
          collectResidualAddConvLayers(block->blocks, addToOutputConvs);
        }
      }
    }

    static void collectTransformerGemmStridedCases(
      const std::vector<std::pair<int, unique_ptr_void>>& blocks,
      int paddedSpatialSize,
      std::vector<GemmStridedModeCase>& outCases) {
      for(const auto& kv: blocks) {
        if(kv.first == TRANSFORMER_ATTENTION_BLOCK_KIND) {
          const auto* attn = static_cast<const TransformerAttentionDesc*>(kv.second.get());
          addGemmStridedModeCase(outCases, paddedSpatialSize, attn->qProj.outChannels, attn->qProj.inChannels, false);
          addGemmStridedModeCase(outCases, paddedSpatialSize, attn->kProj.outChannels, attn->kProj.inChannels, false);
          addGemmStridedModeCase(outCases, paddedSpatialSize, attn->vProj.outChannels, attn->vProj.inChannels, false);
          addGemmStridedModeCase(outCases, paddedSpatialSize, attn->outProj.outChannels, attn->outProj.inChannels, true);
        } else if(kv.first == TRANSFORMER_FFN_BLOCK_KIND) {
          const auto* ffn = static_cast<const TransformerFFNDesc*>(kv.second.get());
          addGemmStridedModeCase(outCases, paddedSpatialSize, ffn->linear1.outChannels, ffn->linear1.inChannels, false);
          if(ffn->useSwiGLU)
            addGemmStridedModeCase(
              outCases, paddedSpatialSize, ffn->linearGate.outChannels, ffn->linearGate.inChannels, false);
          addGemmStridedModeCase(outCases, paddedSpatialSize, ffn->linear2.outChannels, ffn->linear2.inChannels, true);
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
      std::vector<const ConvLayerDesc*> addToOutputConvs;
      collectResidualAddConvLayers(modelDesc->trunk.blocks, addToOutputConvs);

      modelDesc->iterConvLayers([&](const ConvLayerDesc& conv) {
        if(conv.convXSize == 1 && conv.convYSize == 1)
          addGemmStridedModeCase(
            cases, paddedSpatialSize, conv.outChannels, conv.inChannels, contains(addToOutputConvs, &conv));
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
        int outTile = is3x3 ? nhwcWinograd3x3OutTileFor(cfg) : 2;
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

    struct Conv3x3NhwcModeCase {
      int inChannels;
      int outChannels;
      int occurrences;
      double weightedWork;
    };

    static std::vector<Conv3x3NhwcModeCase> collectConv3x3NhwcModeCases(const ModelDesc* modelDesc, int convSize = 3) {
      std::vector<Conv3x3NhwcModeCase> cases;
      if(modelDesc == nullptr || (convSize != 3 && convSize != 5))
        return cases;
      modelDesc->iterConvLayers([&](const ConvLayerDesc& conv) {
        if(
          conv.convXSize != convSize || conv.convYSize != convSize || conv.dilationX != 1 || conv.dilationY != 1 ||
          conv.inChannels <= 0 || conv.outChannels <= 0)
          return;
        // Native NHWC pads the input channel tail to a vec8 stride before
        // dispatching this kernel. Tune that physical width as well, so an
        // initial layer such as 22 -> 96 is represented as K=convTaps*24
        // rather than being silently omitted from coverage.
        const int paddedInChannels = roundUpToMultipleInt(conv.inChannels, 8);
        for(auto& cs: cases) {
          if(cs.inChannels == paddedInChannels && cs.outChannels == conv.outChannels) {
            cs.occurrences++;
            return;
          }
        }
        cases.push_back({paddedInChannels, conv.outChannels, 1, 0.0});
      });
      for(auto& cs: cases)
        cs.weightedWork = (double)cs.occurrences * cs.inChannels * cs.outChannels;
      std::sort(cases.begin(), cases.end(), [](const Conv3x3NhwcModeCase& a, const Conv3x3NhwcModeCase& b) {
        if(a.weightedWork != b.weightedWork)
          return a.weightedWork > b.weightedWork;
        if(a.outChannels != b.outChannels)
          return a.outChannels > b.outChannels;
        return a.inChannels > b.inChannels;
      });
      return cases;
    }

    static double dispatchWeightedTime(const KernelBench& b, int occurrences) {
      return (double)occurrences / b.kernelsPerSecond;
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

    static int32_t weightedTimeToMicros(const ModeSelectStats& stats) {
      if(!stats.allOk || !(stats.weightedTime > 0.0) || !std::isfinite(stats.weightedTime))
        return 0;
      const double micros = std::round(stats.weightedTime * 1.0e6);
      return (int32_t)std::max(1.0, std::min(micros, (double)std::numeric_limits<int32_t>::max()));
    }

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
      bool preferChallengerFamily,
      bool appendRequiredMinSpeedup,
      bool appendRejectReason,
      AppendExtraFieldsFn appendExtraFields) {
      // Preserve the accelerator-family order (Coopmat, then Dot2, then
      // tiled). Within an accelerator family, FP16 accumulation must buy
      // enough speed to justify its reduced precision versus FP32.
      const bool challengerAccF16 = challengerName.find("AccF16") != string_view::npos;
      const bool incumbentAccF32 = incumbentName.find("AccF16") == string_view::npos;
      const bool compareAcceleratorAccumulation = challengerAccF16 && incumbentAccF32 &&
        ((challengerName.find("Coopmat") != string_view::npos && incumbentName.find("Coopmat") != string_view::npos) ||
         (challengerName.find("Dot2") != string_view::npos && incumbentName.find("Dot2") != string_view::npos));
      const double requiredMinSpeedup = compareAcceleratorAccumulation ? 1.25 : MODE_SELECT_MIN_SPEEDUP;
      const bool requiresAccF16Gain = compareAcceleratorAccumulation;
      bool challengerSelected = challenger.allOk &&
        ((!requiresAccF16Gain && preferChallengerFamily) ||
         (incumbent.allOk && challenger.weightedTime * requiredMinSpeedup < incumbent.weightedTime));
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
        out << " required_min_speedup=" << requiredMinSpeedup;
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

    int paddedSpatialSize = roundUpToMultipleInt(nnXLen * nnYLen, VulkanKernels::VULKAN_SPATIAL_ALIGN);
    int maxChannels = std::max(
      {modelDesc->numInputChannels,
       trunk.trunkNumChannels,
       trunk.midNumChannels,
       modelDesc->maxConvChannels(3, 3),
       22});
    std::vector<int> layoutChannels = {8, 16, 22, std::max(32, maxChannels)};
    std::sort(layoutChannels.begin(), layoutChannels.end());
    layoutChannels.erase(std::unique(layoutChannels.begin(), layoutChannels.end()), layoutChannels.end());
    kernels.push_back(std::make_unique<VulkanKernels::NchwToNhwcTuner>(
      std::max(1, batchSize), paddedSpatialSize, layoutChannels));

    if(tuningUsesNhwc()) {
      int gpoolC3 = trunk.gpoolNumChannels > 0 ? trunk.gpoolNumChannels * 3 : 192;
      kernels.push_back(std::make_unique<VulkanKernels::GemmDirectFP32>(batchSize > 0 ? batchSize : 1, gpoolC3, gpoolC3));
      int gpoolC = trunk.gpoolNumChannels > 0 ? trunk.gpoolNumChannels : 32;
      kernels.push_back(std::make_unique<VulkanKernels::GPoolReductionNhwc>(batchSize, gpoolC, nnXLen * nnYLen));
      kernels.push_back(std::make_unique<VulkanKernels::ValueHeadPoolNhwc>(batchSize, gpoolC, nnXLen * nnYLen));
      // NHWC Winograd transform/untransform (non-coopmat fallback conv path).
      // Only actually benched when TUNED_NHWC_WINOGRAD is in the missing mask
      // (requiredKernelMask omits it on coopmat-capable devices, where this
      // path is never selected), so it's safe to always add these here.
      int wgChannelsNhwc = trunk.trunkNumChannels > 0 ? trunk.trunkNumChannels : 64;
      int wgInChannelsNhwc = std::max({modelDesc->numInputChannels, trunk.trunkNumChannels, trunk.midNumChannels, 64});
      kernels.push_back(
        std::make_unique<VulkanKernels::WinogradTransformNhwc>(batchSize, wgInChannelsNhwc, nnXLen, nnYLen));
      kernels.push_back(
        std::make_unique<VulkanKernels::WinogradUntransformNhwc>(batchSize, wgChannelsNhwc, nnXLen, nnYLen));
      if(!modelHasTransformer(modelDesc))
        return kernels;
    }

    // GEMM (Winograd trunk matmul): M ~ tile positions, N/K ~ channels, inTile^2 Winograd positions.
    // The tiled shader requires M, N ÷8 (8-wide A loads); round representative bench dims up
    // to ÷8 multiples so the bench exercises the same aligned path.
    int winograd3x3OutTile = nhwcWinograd3x3OutTileFor(cfg);
    int winograd3x3InTile = winograd3x3OutTile + 2;
    int channels = roundUpToMultipleInt(
      std::max({trunk.trunkNumChannels, trunk.midNumChannels, modelDesc->maxConvChannels(3, 3), 64}), 8);
    int numTilesX = (nnXLen + winograd3x3OutTile - 1) / winograd3x3OutTile;
    int numTilesY = (nnYLen + winograd3x3OutTile - 1) / winograd3x3OutTile;
    int numTiles = roundUpToMultipleInt(std::max(1, batchSize) * numTilesX * numTilesY, 8);
    kernels.push_back(
      std::make_unique<VulkanKernels::WinogradGemm>(
        numTiles, channels, channels, winograd3x3InTile * winograd3x3InTile));

    // GemmDirect (FC head).
    int gpoolC3 = trunk.gpoolNumChannels > 0 ? trunk.gpoolNumChannels * 3 : 192;
    kernels.push_back(std::make_unique<VulkanKernels::GemmDirectFP32>(batchSize > 0 ? batchSize : 1, gpoolC3, gpoolC3));

    // Transformer kernels — scan blocks recursively (transformers may live inside
    // nested bottleneck blocks) to find representative attention dimensions and
    // whether any attention block uses fixed/learnable RoPE.
    int headDim = 0, vHeadDim = 0, numHeads = 0, numKVHeads = 0;
    int swiGLUChannels = 0;
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
          } else if(kv.first == TRANSFORMER_FFN_BLOCK_KIND) {
            const auto* ffn = static_cast<const TransformerFFNDesc*>(kv.second.get());
            if(ffn->useSwiGLU)
              swiGLUChannels = std::max(swiGLUChannels, ffn->ffnChannels);
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
      int seqLen = paddedSpatialSize;
      kernels.push_back(
        std::make_unique<VulkanKernels::AttentionTiled>(
          batchSize, seqLen, benchHeadDim, benchVHeadDim, benchNumHeads, benchNumKVHeads, useRope, learnableRope,
          tuningUsesNhwc(), nnXLen * nnYLen));
      int rmsChannels = trunk.trunkNumChannels > 0 ? trunk.trunkNumChannels : 64;
      kernels.push_back(std::make_unique<VulkanKernels::SpatialRMSNormNhwc>(batchSize, rmsChannels, seqLen));
    }
    if(swiGLUChannels > 0)
      kernels.push_back(std::make_unique<VulkanKernels::SwiGLU>(batchSize, swiGLUChannels, paddedSpatialSize));

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
      TuningContext& ctx,
      BenchCandidateFn benchCandidate,
      CandidateToStringFn candidateToString,
      CandidateEqualFn candidateEqual) {
      AggregateCandidateSweepResult<CandidateT> result;
      if(!candidates.empty())
        result.candidate = candidates[0];

      std::vector<ScreenedAggregateCandidate<CandidateT>> screenedCandidates;
      screenedCandidates.reserve(candidates.size());
      {
        ScopedTuningBenchTarget screenTarget(ctx, BENCH_SCREEN_TARGET_SECONDS);
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
        ScopedTuningBenchTarget confirmTarget(ctx, BENCH_TARGET_SECONDS);
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
        ctx,
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
        modeCases.push_back({paddedSpatialSize, trunkC, trunkC, 1, 1, 0, (double)trunkC * (double)trunkC});
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
        int outTile = nhwcWinograd3x3OutTileFor(seed);
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
        ctx,
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

      tunedConfig.winogradGemmDot2F32TimeUs = weightedTimeToMicros(dot2Stats);
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
        true,
        [&](std::ostream& resultOut) {
          appendPositiveField(resultOut, "dot2_speedup", modeSelectSpeedup(tiledStats, dot2Stats));
        });
      if(selected) {
        tunedConfig.enableWinogradGemmDot2 = 1;
        tunedConfig.enableWinogradGemmDot2AccF16 = 0;
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

      tunedConfig.winogradGemmDot2F16TimeUs = weightedTimeToMicros(accStats);
      bool selected = appendModeSelectResult(
        out,
        "winogradGemm",
        incumbentName,
        "winogradGemmDot2AccF16",
        wCases,
        incumbentStats,
        accStats,
        maxRmse,
        !incumbentDot2AccF16 && !incumbentDot2,
        true,
        true,
        [&](std::ostream& resultOut) {
          appendPositiveField(resultOut, "dot2AccF16_speedup", modeSelectSpeedup(incumbentStats, accStats));
        });
      if(selected) {
        tunedConfig.enableWinogradGemmDot2AccF16 = 1;
        tunedConfig.enableWinogradGemmDot2 = 0;
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
                    VulkanKernels::WinogradGemmCoopmat1::isConfigSupported(
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
      cfg.coopmat1BlockSize = t.blockSize;
      cfg.coopmat1BM = t.bm;
      cfg.coopmat1BN = t.bn;
      cfg.coopmat1BK = t.bk;
      cfg.coopmat1WM = t.wm;
      cfg.coopmat1WN = t.wn;
      cfg.coopmat1TM = t.tm;
      cfg.coopmat1TN = t.tn;
      cfg.coopmat1TK = t.tk;
      cfg.coopmat1Warp = t.warp;
    }

    void applyConv3x3NhwcCoopmatTile(VulkanTuneParams& cfg, const CoopmatTile& t) {
      cfg.conv3x3NhwcCoopmat1BlockSize = t.blockSize;
      cfg.conv3x3NhwcCoopmat1BM = t.bm;
      cfg.conv3x3NhwcCoopmat1BN = t.bn;
      cfg.conv3x3NhwcCoopmat1BK = t.bk;
      cfg.conv3x3NhwcCoopmat1WM = t.wm;
      cfg.conv3x3NhwcCoopmat1WN = t.wn;
      cfg.conv3x3NhwcCoopmat1TM = t.tm;
      cfg.conv3x3NhwcCoopmat1TN = t.tn;
      cfg.conv3x3NhwcCoopmat1TK = t.tk;
      cfg.conv3x3NhwcCoopmat1Warp = t.warp;
    }

    void applyConv5x5NhwcCoopmatTile(VulkanTuneParams& cfg, const CoopmatTile& t) {
      cfg.conv5x5NhwcCoopmat1BlockSize = t.blockSize;
      cfg.conv5x5NhwcCoopmat1BM = t.bm;
      cfg.conv5x5NhwcCoopmat1BN = t.bn;
      cfg.conv5x5NhwcCoopmat1BK = t.bk;
      cfg.conv5x5NhwcCoopmat1WM = t.wm;
      cfg.conv5x5NhwcCoopmat1WN = t.wn;
      cfg.conv5x5NhwcCoopmat1TM = t.tm;
      cfg.conv5x5NhwcCoopmat1TN = t.tn;
      cfg.conv5x5NhwcCoopmat1TK = t.tk;
      cfg.conv5x5NhwcCoopmat1Warp = t.warp;
    }

    void applyWinogradCoopmatAccF16Tile(VulkanTuneParams& cfg, const CoopmatTile& t) {
      cfg.coopmat1AccF16BlockSize = t.blockSize;
      cfg.coopmat1AccF16BM = t.bm;
      cfg.coopmat1AccF16BN = t.bn;
      cfg.coopmat1AccF16BK = t.bk;
      cfg.coopmat1AccF16WM = t.wm;
      cfg.coopmat1AccF16WN = t.wn;
      cfg.coopmat1AccF16TM = t.tm;
      cfg.coopmat1AccF16TN = t.tn;
      cfg.coopmat1AccF16TK = t.tk;
      cfg.coopmat1AccF16Warp = t.warp;
    }

    void applyNhwcStridedCoopmatAccF16Tile(VulkanTuneParams& cfg, const CoopmatTile& t) {
      cfg.nhwcStridedCoopmat1AccF16BlockSize = t.blockSize;
      cfg.nhwcStridedCoopmat1AccF16BM = t.bm;
      cfg.nhwcStridedCoopmat1AccF16BN = t.bn;
      cfg.nhwcStridedCoopmat1AccF16BK = t.bk;
      cfg.nhwcStridedCoopmat1AccF16WM = t.wm;
      cfg.nhwcStridedCoopmat1AccF16WN = t.wn;
      cfg.nhwcStridedCoopmat1AccF16TM = t.tm;
      cfg.nhwcStridedCoopmat1AccF16TN = t.tn;
      cfg.nhwcStridedCoopmat1AccF16TK = t.tk;
      cfg.nhwcStridedCoopmat1AccF16Warp = t.warp;
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
            accF16 ? padDims(cs.M, cs.N, cs.K, VulkanKernels::WinogradGemmCoopmat1AccF16::layerPaddingContract(trial))
                   : padDims(cs.M, cs.N, cs.K, VulkanKernels::WinogradGemmCoopmat1::layerPaddingContract(trial));
          KernelBench b;
          double flopsPerDispatch = 0.0;
          if(accF16) {
            VulkanKernels::WinogradGemmCoopmat1AccF16 probe(pad.m, pad.n, pad.k, cs.numBatches);
            if(!probe.validate(trial, ctx.dev->info.properties.limits))
              return rejectAggregateCandidate(metrics, "validate_failed");
            b = probe.bench(ctx, trial, iters);
            flopsPerDispatch = probe.estimatedFlopsPerDispatch(trial);
          } else {
            VulkanKernels::WinogradGemmCoopmat1 probe(pad.m, pad.n, pad.k, cs.numBatches);
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

      string_view label = accF16 ? "winogradCoopmat1AccF16" : "winogradCoopmat1";
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
        ctx,
        benchAggregateTile,
        coopmatTileToString,
        coopmatTilesEqual);
      if(result.found) {
        if(accF16) {
          applyWinogradCoopmatAccF16Tile(tunedConfig, result.candidate);
          tunedConfig.winogradGemmCoopmat1AccF16TunerValueValid = 1;
        } else {
          applyWinogradCoopmatTile(tunedConfig, result.candidate);
          tunedConfig.winogradGemmCoopmat1TunerValueValid = 1;
        }
      }
    }

    template<typename Ops>
    void tuneNhwcConvSweep(
      TuningContext& ctx,
      const ModelDesc* modelDesc,
      int problemBatchSize,
      int nnXLen,
      int nnYLen,
      int convSize,
      int iters,
      const std::vector<typename Ops::Tile>& candidates,
      std::ostream& out,
      bool verboseTuner,
      VulkanTuneParams& tunedConfig) {
      using Tile = typename Ops::Tile;
      const int paddedSpatialSize = roundUpToMultipleInt(nnXLen * nnYLen, VulkanKernels::VULKAN_SPATIAL_ALIGN);
      std::vector<Conv3x3NhwcModeCase> coverageCases = collectConv3x3NhwcModeCases(modelDesc, convSize);
      if(coverageCases.empty() || candidates.empty()) {
        Ops::markConvValid(tunedConfig, convSize);
        return;
      }
      std::vector<Conv3x3NhwcModeCase> modeCases = coverageCases;
      if(modeCases.size() > CONV3X3_NHWC_MODE_MAX_CASES)
        modeCases.resize(CONV3X3_NHWC_MODE_MAX_CASES);

      const VkPhysicalDeviceLimits& limits = ctx.dev->info.properties.limits;
      std::vector<Tile> eligible;
      for(const Tile& t: candidates) {
        if(!Ops::convEligible(ctx, t, paddedSpatialSize, limits))
          continue;
        bool coversAllCases = true;
        for(const Conv3x3NhwcModeCase& cs: coverageCases) {
          if(cs.outChannels % Ops::bn(t) != 0) {
            coversAllCases = false;
            break;
          }
        }
        if(coversAllCases)
          eligible.push_back(t);
      }
      if(eligible.empty()) {
        out << "VulkanTuner: " << Ops::convLabel(convSize) << " tile sweep found no tile covering all native "
               << convSize << "x" << convSize << " layers; keeping "
               "defaults."
            << endl;
        Ops::markConvValid(tunedConfig, convSize);
        return;
      }

      const Conv3x3NhwcModeCase& trimCase = modeCases[0];
      std::vector<Tile> sweepCandidates =
        Ops::trimConv(eligible, paddedSpatialSize, trimCase.outChannels, convSize * convSize * trimCase.inChannels);
      if(sweepCandidates.empty()) {
        Ops::markConvValid(tunedConfig, convSize);
        return;
      }

      struct CachedConvRef {
        int batchSize;
        int inChannels;
        int outChannels;
        KernelBench bench;
      };
      std::vector<CachedConvRef> refs;
      auto getReference = [&](int batchSize, const Conv3x3NhwcModeCase& cs) -> const KernelBench* {
        for(const CachedConvRef& ref: refs) {
          if(ref.batchSize == batchSize && ref.inChannels == cs.inChannels && ref.outChannels == cs.outChannels)
            return &ref.bench;
        }
        KernelBench reference = Ops::benchConvReference(
          ctx,
          tunedConfig,
          sweepCandidates,
          iters,
          batchSize,
          nnXLen,
          nnYLen,
          paddedSpatialSize,
          convSize,
          cs.inChannels,
          cs.outChannels);
        refs.push_back({batchSize, cs.inChannels, cs.outChannels, std::move(reference)});
        return &refs.back().bench;
      };

      auto benchTile = [&](const Tile& t, AggregateCandidateMetrics& metrics) {
        VulkanTuneParams trial = tunedConfig;
        Ops::applyConv(trial, t, convSize);
        metrics.weightedTime = 0.0;
        metrics.tflops = 0.0;
        metrics.maxRmse = 0.0;
        double aggregateFlops = 0.0;
        double aggregateSeconds = 0.0;
        for(const Conv3x3NhwcModeCase& cs: modeCases) {
          const int batches[2] = {1, std::max(1, problemBatchSize)};
          const int batchCount = batches[0] == batches[1] ? 1 : 2;
          for(int batchIdx = 0; batchIdx < batchCount; batchIdx++) {
            const int batchSize = batches[batchIdx];
            KernelBench b = Ops::benchConv(
              ctx, trial, iters, batchSize, nnXLen, nnYLen, paddedSpatialSize, convSize, cs.inChannels, cs.outChannels);
            if(!(b.ok && b.kernelsPerSecond > 0.0))
              return rejectAggregateCandidate(metrics, "bench_failed");
            const KernelBench* ref = getReference(batchSize, cs);
            if(ref == nullptr || !ref->ok)
              return rejectAggregateCandidate(metrics, "reference_failed");
            const double rmse = normalizedRmse(ref->output, b.output);
            metrics.maxRmse = std::max(metrics.maxRmse, rmse);
            if(rmse > 0.02)
              return rejectAggregateCandidate(metrics, "rmse_exceeded");

            const double latencyWeight = batchSize == 1 && problemBatchSize > 1 ? 4.0 : 1.0;
            const double caseSeconds = latencyWeight * (double)cs.occurrences / b.kernelsPerSecond;
            metrics.weightedTime += caseSeconds;
            const double flops =
              2.0 * batchSize * nnXLen * nnYLen * cs.outChannels * (double)(convSize * convSize) * cs.inChannels * cs.occurrences;
            aggregateFlops += latencyWeight * flops;
            aggregateSeconds += caseSeconds;
          }
        }
        metrics.tflops = aggregateTflops(aggregateFlops, aggregateSeconds);
        return metrics.weightedTime > 0.0;
      };

      out << "VulkanTuner: " << Ops::convLabel(convSize) << " tile sweep candidates=" << sweepCandidates.size()
          << " raw_candidates=" << candidates.size() << " cases=" << modeCases.size()
          << " batches=[1," << std::max(1, problemBatchSize) << "] batch1_weight=4"
          << " trim_case M=" << paddedSpatialSize << " N=" << trimCase.outChannels
          << " K=" << convSize * convSize * trimCase.inChannels << endl;
      AggregateCandidateSweepResult<Tile> result = tuneAggregateCandidateSweep(
        TunerStreamText{Ops::convLabel(convSize)},
        TunerStreamText{Ops::convLabel(convSize)},
        TunerStreamText{Ops::convLabel(convSize), " tile sweep"},
        "tile",
        TunerStreamText{"VulkanTuner: ", Ops::convLabel(convSize), " tile sweep found no valid tile; keeping defaults."},
        sweepCandidates,
        TUNER_CONFIRM_PREFIX_K,
        true,
        true,
        true,
        out,
        verboseTuner,
        ctx,
        benchTile,
        Ops::format,
        Ops::equal);
      if(result.found)
        Ops::saveConv(tunedConfig, result.candidate, convSize);
      Ops::markConvValid(tunedConfig, convSize);
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
        !incumbentCoopmat2AccF16 && !incumbentCoopmat2 && !accF16 && tunedConfig.enableWinogradGemmCoopmat1AccF16 != 0;
      const bool incumbentCoopmat = !incumbentCoopmat2AccF16 && !incumbentCoopmat2 && !incumbentCoopmatAccF16 &&
                                    accF16 && tunedConfig.enableWinogradGemmCoopmat1 != 0;
      const bool incumbentDot2AccF16 = !incumbentCoopmat2AccF16 && !incumbentCoopmat2 && !incumbentCoopmatAccF16 &&
                                       !incumbentCoopmat && tunedConfig.enableWinogradGemmDot2AccF16 != 0;
      const bool incumbentDot2 = !incumbentCoopmat2AccF16 && !incumbentCoopmat2 && !incumbentCoopmatAccF16 &&
                                 !incumbentCoopmat && !incumbentDot2AccF16 && tunedConfig.enableWinogradGemmDot2 != 0;
      string_view incumbentName = incumbentCoopmat2AccF16  ? "winogradGemmCoopmat2AccF16"
                                  : incumbentCoopmat2      ? "winogradGemmCoopmat2"
                                  : incumbentCoopmatAccF16 ? "winogradGemmCoopmat1AccF16"
                                  : incumbentCoopmat       ? "winogradGemmCoopmat1"
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

      string_view challengerName = accF16 ? "winogradGemmCoopmat1AccF16" : "winogradGemmCoopmat1";
      out << endl
          << "VulkanTuner: winogradGemm " << (accF16 ? "coopmat1AccF16" : "coopmat1") << " mode-select cases ("
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
            padDims(cs.M, cs.N, cs.K, VulkanKernels::WinogradGemmCoopmat1AccF16::layerPaddingContract(tunedConfig));
          VulkanKernels::WinogradGemmCoopmat1AccF16 probe(pad.m, pad.n, pad.k, cs.numBatches);
          incumbentBench = probe.bench(ctx, tunedConfig, iters);
          incumbentTflops = probe.estimatedTflops(tunedConfig, incumbentBench.kernelsPerSecond);
          incumbentFlopsPerDispatch = probe.estimatedFlopsPerDispatch(tunedConfig);
        } else if(incumbentCoopmat) {
          const auto pad =
            padDims(cs.M, cs.N, cs.K, VulkanKernels::WinogradGemmCoopmat1::layerPaddingContract(tunedConfig));
          VulkanKernels::WinogradGemmCoopmat1 probe(pad.m, pad.n, pad.k, cs.numBatches);
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
            ? padDims(cs.M, cs.N, cs.K, VulkanKernels::WinogradGemmCoopmat1AccF16::layerPaddingContract(tunedConfig))
            : padDims(cs.M, cs.N, cs.K, VulkanKernels::WinogradGemmCoopmat1::layerPaddingContract(tunedConfig));
        KernelBench cmBench;
        double coopmatTflops = 0.0;
        double coopmatFlopsPerDispatch = 0.0;
        if(accF16) {
          VulkanKernels::WinogradGemmCoopmat1AccF16 cmProbe(cmPad.m, cmPad.n, cmPad.k, cs.numBatches);
          cmBench = cmProbe.bench(ctx, tunedConfig, iters);
          coopmatTflops = cmProbe.estimatedTflops(tunedConfig, cmBench.kernelsPerSecond);
          coopmatFlopsPerDispatch = cmProbe.estimatedFlopsPerDispatch(tunedConfig);
        } else {
          VulkanKernels::WinogradGemmCoopmat1 cmProbe(cmPad.m, cmPad.n, cmPad.k, cs.numBatches);
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
          out << "VulkanTuner:   winogradGemm " << (accF16 ? "coopmat1AccF16" : "coopmat1") << " case#" << i;
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
      (accF16 ? tunedConfig.winogradGemmCoopmat1F16TimeUs : tunedConfig.winogradGemmCoopmat1F32TimeUs) =
        weightedTimeToMicros(coopmatStats);
      bool selected = appendModeSelectResult(
        out,
        "winogradGemm",
        incumbentName,
        challengerName,
        wCases,
        incumbentStats,
        coopmatStats,
        maxRmse,
        !incumbentCoopmat2AccF16 && !incumbentCoopmat2 && !incumbentCoopmatAccF16 && !incumbentCoopmat,
        true,
        true,
        [&](std::ostream& resultOut) {
          appendPositiveField(resultOut, "coopmat_avg_speedup", modeSelectSpeedup(incumbentStats, coopmatStats));
          appendPositiveField(resultOut, "coopmat_max_speedup", maxCoopmatSpeedup);
        });
      if(selected) {
        if(accF16)
          tunedConfig.enableWinogradGemmCoopmat1AccF16 = 1;
        else
          tunedConfig.enableWinogradGemmCoopmat1 = 1;
        tunedConfig.enableWinogradGemmDot2 = 0;  // mutually exclusive at runtime
        tunedConfig.enableWinogradGemmDot2AccF16 = 0;
        tunedConfig.enableWinogradGemmCoopmat1 = accF16 ? 0 : tunedConfig.enableWinogradGemmCoopmat1;
        tunedConfig.enableWinogradGemmCoopmat1AccF16 = accF16 ? tunedConfig.enableWinogradGemmCoopmat1AccF16 : 0;
        tunedConfig.enableWinogradGemmCoopmat2 = 0;
        tunedConfig.enableWinogradGemmCoopmat2AccF16 = 0;
      }
    }

    // ---- Coopmat2 (VK_NV_cooperative_matrix2) tuning ----

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
    // device-inappropriate cases. maxBk defaults to 32 but callers may raise it:
    // the shaders that stage through shared memory pay for BK in occupancy, while
    // the decode-based direct convolution uses no shared memory at all, so for it
    // a larger BK only means fewer, larger coopMatMulAdds and fewer A re-slices.
    std::vector<Coopmat2Tile> makeCoopmat2Candidates(
      const TuningContext& ctx,
      const std::vector<Coopmat2FlexShape>& shapes,
      int maxBk = 32) {
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

    void applyWinogradCoopmat2AccF16Tile(VulkanTuneParams& cfg, const Coopmat2Tile& t) {
      cfg.coopmat2AccF16BlockSize = t.blockSize;
      cfg.coopmat2AccF16BM = t.bm;
      cfg.coopmat2AccF16BN = t.bn;
      cfg.coopmat2AccF16BK = t.bk;
    }

    struct NhwcCoopmat1F32Ops {
      using Tile = CoopmatTile;
      static string_view gemmLabel(){return "gemmStridedCoopmat1AccF32Nhwc";}
      static int bm(const Tile& t){return t.bm;} static int bn(const Tile& t){return t.bn;}
      static int bk(const Tile& t){return t.bk;} static int block(const Tile& t){return t.blockSize;}
      static std::string format(const Tile& t){return coopmatTileToString(t);}
      static bool equal(const Tile& a,const Tile& b){return coopmatTilesEqual(a,b);}
      static std::vector<Tile> trim(const std::vector<Tile>& c,int m,int n,int k){
        std::vector<Tile> trimmed;
        for(const Tile& t: trimCoopmatCandidates(c,m,n,k,false,true,false,64,32,256,32,32,32,1.35,4096.0,32.0,64.0))
          if(VulkanKernels::GemmStridedCoopmat1Nhwc::isConfigSupported(
               t.blockSize,t.bm,t.bn,t.bk,t.wm,t.wn,t.tm,t.tn,t.tk,t.warp)) trimmed.push_back(t);
        return trimmed;
      }
      static void applyGemm(VulkanTuneParams& c,const Tile& t){
        c.nhwcStridedCoopmat1BlockSize=t.blockSize;c.nhwcStridedCoopmat1BM=t.bm;
        c.nhwcStridedCoopmat1BN=t.bn;c.nhwcStridedCoopmat1BK=t.bk;c.nhwcStridedCoopmat1WM=t.wm;
        c.nhwcStridedCoopmat1WN=t.wn;c.nhwcStridedCoopmat1TM=t.tm;c.nhwcStridedCoopmat1TN=t.tn;
        c.nhwcStridedCoopmat1TK=t.tk;c.nhwcStridedCoopmat1Warp=t.warp;
      }
      static void saveGemm(VulkanTuneParams& c,const Tile& t){applyGemm(c,t);c.gemmStridedNhwcCoopmat1TunerValueValid=1;}
      static int32_t& timeUs(VulkanTuneParams& c){return c.nhwcGemmCoopmat1F32TimeUs;}
      static bool splitOutputModes(){return true;}
      static KernelBench benchGemm(
        TuningContext& x,const VulkanTuneParams& c,int it,int b,int m,int n,int k,bool addToOutput){
        return VulkanKernels::GemmStridedCoopmat1Nhwc::bench(x,c,it,b,m,n,k,addToOutput);}
      static string_view convLabel(int convSize){return convSize == 5 ? "conv5x5NhwcCoopmat" : "conv3x3NhwcCoopmat";}
      static std::vector<Tile> trimConv(const std::vector<Tile>& c,int m,int n,int k){
        return trimCoopmatCandidates(c,m,n,k,false,false,false,64,32,256,32,32,64,1.01,4096.0,32.0,128.0);}
      static bool convEligible(TuningContext& x,const Tile& t,int spatial,const VkPhysicalDeviceLimits& limits){
        return VulkanKernels::Conv3x3ImplicitGemmCoopmat1AccF32NhwcVec8::isConfigSupported(
                 t.blockSize,t.bm,t.bn,t.bk,t.wm,t.wn,t.tm,t.tn,t.tk,t.warp) &&
               t.warp==(int32_t)x.dev->info.subgroupSize && (uint32_t)t.blockSize<=limits.maxComputeWorkGroupInvocations &&
               (uint32_t)t.blockSize<=limits.maxComputeWorkGroupSize[0] && spatial%t.bm==0 &&
               VulkanKernels::Conv3x3ImplicitGemmCoopmat1AccF32NhwcVec8::sharedBytes(t.bm,t.bn,t.bk)<=limits.maxComputeSharedMemorySize;
      }
      static void applyConv(VulkanTuneParams& c,const Tile& t,int convSize){
        if(convSize == 5) applyConv5x5NhwcCoopmatTile(c,t); else applyConv3x3NhwcCoopmatTile(c,t);}
      static void saveConv(VulkanTuneParams& c,const Tile& t,int convSize){applyConv(c,t,convSize);}
      static void markConvValid(VulkanTuneParams& c,int convSize){
        if(convSize == 5) c.conv5x5NhwcCoopmat1TunerValueValid=1; else c.conv3x3NhwcCoopmat1TunerValueValid=1;}
      static KernelBench benchConv(TuningContext& x,const VulkanTuneParams& c,int it,int b,int nx,int ny,int s,int convSize,int in,int out){
        return VulkanKernels::Conv3x3ImplicitGemmCoopmat1AccF32NhwcVec8::bench(x,c,it,b,nx,ny,s,convSize,in,out);}
      static KernelBench benchConvReference(TuningContext& x,const VulkanTuneParams& c,const std::vector<Tile>& tiles,int it,int b,int nx,int ny,int s,int convSize,int in,int out){
        for(const Tile& tile:tiles) { VulkanTuneParams trial=c;applyConv(trial,tile,convSize);
          KernelBench ref=benchConv(x,trial,it,b,nx,ny,s,convSize,in,out);if(ref.ok)return ref; }
        return KernelBench();}
    };
    struct NhwcCoopmat2F32Ops {
      using Tile = Coopmat2Tile;
      static string_view gemmLabel(){return "gemmStridedCoopmat2AccF32Nhwc";}
      static int bm(const Tile& t){return t.bm;} static int bn(const Tile& t){return t.bn;}
      static int bk(const Tile& t){return t.bk;} static int block(const Tile& t){return t.blockSize;}
      static std::string format(const Tile& t){return coopmat2TileToString(t);}
      static bool equal(const Tile& a,const Tile& b){return coopmat2TilesEqual(a,b);}
      static std::vector<Tile> trim(const std::vector<Tile>& c,int m,int n,int k){
        std::vector<Tile> trimmed;
        for(const Tile& t: trimCoopmat2Candidates(c,m,n,k,false,true,false,64,1,100000,8,8,32,1.35,4096.0,32.0,64.0))
          if(VulkanKernels::GemmStridedCoopmat2Nhwc::isConfigSupported(t.blockSize,t.bm,t.bn,t.bk)) trimmed.push_back(t);
        return trimmed;
      }
      static void applyGemm(VulkanTuneParams& c,const Tile& t){
        c.nhwcStridedCoopmat2BlockSize=t.blockSize;c.nhwcStridedCoopmat2BM=t.bm;
        c.nhwcStridedCoopmat2BN=t.bn;c.nhwcStridedCoopmat2BK=t.bk;
      }
      static void saveGemm(VulkanTuneParams& c,const Tile& t){applyGemm(c,t);c.gemmStridedNhwcCoopmat2TunerValueValid=1;}
      static int32_t& timeUs(VulkanTuneParams& c){return c.nhwcGemmCoopmat2F32TimeUs;}
      static bool splitOutputModes(){return true;}
      static KernelBench benchGemm(
        TuningContext& x,const VulkanTuneParams& c,int it,int b,int m,int n,int k,bool addToOutput){
        return VulkanKernels::GemmStridedCoopmat2Nhwc::bench(x,c,it,b,m,n,k,addToOutput);}
      static string_view convLabel(int convSize){return convSize == 5 ? "conv5x5NhwcCoopmat2" : "conv3x3NhwcCoopmat2";}
      static std::vector<Tile> trimConv(const std::vector<Tile>& c,int m,int n,int k){
        return trimCoopmat2Candidates(c,m,n,k,false,false,false,64,1,100000,8,8,64,1.01,4096.0,32.0,128.0);}
      static bool convEligible(TuningContext& x,const Tile& t,int spatial,const VkPhysicalDeviceLimits& limits){
        const size_t shared=VulkanKernels::Conv3x3ImplicitGemmCoopmat2AccF32NhwcVec8::sharedBytes(t.bm,t.bk);
        return VulkanKernels::Conv3x3ImplicitGemmCoopmat2AccF32NhwcVec8::isConfigSupported(t.blockSize,t.bm,t.bn,t.bk) &&
          coopmat2TileSupported(x.dev->info,x.dev->info.coopmat2FlexShapes,t) &&
          (size_t)x.dev->info.coopmat2ReservedSharedBytes+shared<=limits.maxComputeSharedMemorySize &&
          (uint32_t)t.blockSize<=limits.maxComputeWorkGroupInvocations && (uint32_t)t.blockSize<=limits.maxComputeWorkGroupSize[0] && spatial%t.bm==0;
      }
      static void applyConv(VulkanTuneParams& c,const Tile& t,int convSize){
        if(convSize==5){c.conv5x5NhwcCoopmat2BlockSize=t.blockSize;c.conv5x5NhwcCoopmat2BM=t.bm;c.conv5x5NhwcCoopmat2BN=t.bn;c.conv5x5NhwcCoopmat2BK=t.bk;}
        else {c.conv3x3NhwcCoopmat2BlockSize=t.blockSize;c.conv3x3NhwcCoopmat2BM=t.bm;c.conv3x3NhwcCoopmat2BN=t.bn;c.conv3x3NhwcCoopmat2BK=t.bk;}}
      static void saveConv(VulkanTuneParams& c,const Tile& t,int convSize){applyConv(c,t,convSize);}
      static void markConvValid(VulkanTuneParams& c,int convSize){if(convSize==5)c.conv5x5NhwcCoopmat2TunerValueValid=1;else c.conv3x3NhwcCoopmat2TunerValueValid=1;}
      static KernelBench benchConv(TuningContext& x,const VulkanTuneParams& c,int it,int b,int nx,int ny,int s,int convSize,int in,int out){
        return VulkanKernels::Conv3x3ImplicitGemmCoopmat2AccF32NhwcVec8::bench(x,c,it,b,nx,ny,s,convSize,in,out);}
      static KernelBench benchConvReference(TuningContext& x,const VulkanTuneParams& c,const std::vector<Tile>& tiles,int it,int b,int nx,int ny,int s,int convSize,int in,int out){
        if(x.dev->info.supportsCoopmat1F16){KernelBench ref=VulkanKernels::Conv3x3ImplicitGemmCoopmat1AccF32NhwcVec8::bench(x,c,it,b,nx,ny,s,convSize,in,out);if(ref.ok)return ref;}
        for(const Tile& tile:tiles){VulkanTuneParams trial=c;applyConv(trial,tile,convSize);KernelBench ref=benchConv(x,trial,it,b,nx,ny,s,convSize,in,out);if(ref.ok)return ref;}return KernelBench();}
    };

    struct NhwcCoopmat1Ops {
      using Tile = CoopmatTile;
      static string_view gemmLabel(){return "gemmStridedCoopmat1AccF16Nhwc";}
      static int bm(const Tile& t){return t.bm;} static int bn(const Tile& t){return t.bn;}
      static int bk(const Tile& t){return t.bk;} static int block(const Tile& t){return t.blockSize;}
      static std::string format(const Tile& t){return coopmatTileToString(t);}
      static bool equal(const Tile& a,const Tile& b){return coopmatTilesEqual(a,b);}
      static std::vector<Tile> trim(const std::vector<Tile>& c,int m,int n,int k){
        return trimCoopmatCandidates(c,m,n,k,false,true,false,64,32,256,32,32,32,1.35,4096.0,32.0,64.0);}
      static void applyGemm(VulkanTuneParams& c,const Tile& t){applyNhwcStridedCoopmatAccF16Tile(c,t);}
      static void saveGemm(VulkanTuneParams& c,const Tile& t){
        c.nhwcStridedCoopmat1AccF16BlockSize=t.blockSize;c.nhwcStridedCoopmat1AccF16BM=t.bm;
        c.nhwcStridedCoopmat1AccF16BN=t.bn;c.nhwcStridedCoopmat1AccF16BK=t.bk;c.nhwcStridedCoopmat1AccF16WM=t.wm;
        c.nhwcStridedCoopmat1AccF16WN=t.wn;c.nhwcStridedCoopmat1AccF16TM=t.tm;c.nhwcStridedCoopmat1AccF16TN=t.tn;
        c.nhwcStridedCoopmat1AccF16TK=t.tk;c.nhwcStridedCoopmat1AccF16Warp=t.warp;
        c.gemmStridedNhwcCoopmat1AccF16TunerValueValid=1;}
      static int32_t& timeUs(VulkanTuneParams& c){return c.nhwcGemmCoopmat1F16TimeUs;}
      static bool splitOutputModes(){return true;}
      static KernelBench benchGemm(
        TuningContext& x,const VulkanTuneParams& c,int it,int b,int m,int n,int k,bool addToOutput){
        return VulkanKernels::GemmStridedCoopmat1AccF16Nhwc::bench(x,c,it,b,m,n,k,addToOutput);}
      static string_view convLabel(int convSize){return convSize == 5 ? "conv5x5NhwcCoopmatAccF16" : "conv3x3NhwcCoopmatAccF16";}
      static std::vector<Tile> trimConv(const std::vector<Tile>& c,int m,int n,int k){
        return trimCoopmatCandidates(c,m,n,k,false,false,false,64,32,256,32,32,64,1.01,4096.0,32.0,128.0);}
      static bool convEligible(TuningContext& x,const Tile& t,int spatial,const VkPhysicalDeviceLimits& limits){
        return VulkanKernels::Conv3x3ImplicitGemmCoopmat1AccF16NhwcVec8::isConfigSupported(
                 t.blockSize,t.bm,t.bn,t.bk,t.wm,t.wn,t.tm,t.tn,t.tk,t.warp) &&
               t.warp==(int32_t)x.dev->info.subgroupSize &&
               (uint32_t)t.blockSize<=limits.maxComputeWorkGroupInvocations &&
               (uint32_t)t.blockSize<=limits.maxComputeWorkGroupSize[0] && spatial%t.bm==0 &&
               VulkanKernels::Conv3x3ImplicitGemmCoopmat1AccF16NhwcVec8::sharedBytes(t.bm,t.bn,t.bk)<=
                 limits.maxComputeSharedMemorySize;
      }
      static void applyConv(VulkanTuneParams& c,const Tile& t,int convSize){
        if(convSize == 5) {
          c.conv5x5NhwcCoopmat1AccF16BlockSize=t.blockSize;c.conv5x5NhwcCoopmat1AccF16BM=t.bm;
          c.conv5x5NhwcCoopmat1AccF16BN=t.bn;c.conv5x5NhwcCoopmat1AccF16BK=t.bk;
          c.conv5x5NhwcCoopmat1AccF16WM=t.wm;c.conv5x5NhwcCoopmat1AccF16WN=t.wn;
          c.conv5x5NhwcCoopmat1AccF16TM=t.tm;c.conv5x5NhwcCoopmat1AccF16TN=t.tn;
          c.conv5x5NhwcCoopmat1AccF16TK=t.tk;c.conv5x5NhwcCoopmat1AccF16Warp=t.warp;
        } else {
          c.conv3x3NhwcCoopmat1AccF16BlockSize=t.blockSize;c.conv3x3NhwcCoopmat1AccF16BM=t.bm;
          c.conv3x3NhwcCoopmat1AccF16BN=t.bn;c.conv3x3NhwcCoopmat1AccF16BK=t.bk;
          c.conv3x3NhwcCoopmat1AccF16WM=t.wm;c.conv3x3NhwcCoopmat1AccF16WN=t.wn;
          c.conv3x3NhwcCoopmat1AccF16TM=t.tm;c.conv3x3NhwcCoopmat1AccF16TN=t.tn;
          c.conv3x3NhwcCoopmat1AccF16TK=t.tk;c.conv3x3NhwcCoopmat1AccF16Warp=t.warp;
        }}
      static void saveConv(VulkanTuneParams& c,const Tile& t,int convSize){applyConv(c,t,convSize);}
      static void markConvValid(VulkanTuneParams& c,int convSize){
        if(convSize == 5) c.conv5x5NhwcCoopmat1AccF16TunerValueValid=1; else c.conv3x3NhwcCoopmat1AccF16TunerValueValid=1;}
      static KernelBench benchConv(TuningContext& x,const VulkanTuneParams& c,int it,int b,int nx,int ny,int s,int convSize,int in,int out){
        return VulkanKernels::Conv3x3ImplicitGemmCoopmat1AccF16NhwcVec8::bench(x,c,it,b,nx,ny,s,convSize,in,out);}
      static KernelBench benchConvReference(
        TuningContext& x,const VulkanTuneParams& c,const std::vector<Tile>& tiles,int it,int b,int nx,int ny,int s,int convSize,int in,int out) {
        for(const Tile& tile:tiles) {
          VulkanTuneParams trial=c;applyConv(trial,tile,convSize);
          KernelBench ref=benchConv(x,trial,it,b,nx,ny,s,convSize,in,out);
          if(ref.ok)return ref;
        }
        return KernelBench();
      }
    };
    struct NhwcCoopmat2Ops {
      using Tile = Coopmat2Tile;
      static string_view gemmLabel(){return "gemmStridedCoopmat2AccF16Nhwc";}
      static int bm(const Tile& t){return t.bm;} static int bn(const Tile& t){return t.bn;}
      static int bk(const Tile& t){return t.bk;} static int block(const Tile& t){return t.blockSize;}
      static std::string format(const Tile& t){return coopmat2TileToString(t);}
      static bool equal(const Tile& a,const Tile& b){return coopmat2TilesEqual(a,b);}
      static std::vector<Tile> trim(const std::vector<Tile>& c,int m,int n,int k){
        return trimCoopmat2Candidates(c,m,n,k,false,true,false,64,1,100000,8,8,32,1.35,4096.0,32.0,64.0);}
      static void applyGemm(VulkanTuneParams& c,const Tile& t){
        c.nhwcStridedCoopmat2AccF16BlockSize=t.blockSize;c.nhwcStridedCoopmat2AccF16BM=t.bm;
        c.nhwcStridedCoopmat2AccF16BN=t.bn;c.nhwcStridedCoopmat2AccF16BK=t.bk;}
      static void saveGemm(VulkanTuneParams& c,const Tile& t){applyGemm(c,t);c.gemmStridedNhwcCoopmat2AccF16TunerValueValid=1;}
      static int32_t& timeUs(VulkanTuneParams& c){return c.nhwcGemmCoopmat2F16TimeUs;}
      static bool splitOutputModes(){return true;}
      static KernelBench benchGemm(
        TuningContext& x,const VulkanTuneParams& c,int it,int b,int m,int n,int k,bool addToOutput){
        return VulkanKernels::GemmStridedCoopmat2AccF16Nhwc::bench(x,c,it,b,m,n,k,addToOutput);}
      static string_view convLabel(int convSize){return convSize == 5 ? "conv5x5NhwcCoopmat2AccF16" : "conv3x3NhwcCoopmat2AccF16";}
      static std::vector<Tile> trimConv(const std::vector<Tile>& c,int m,int n,int k){
        // maxBk 64: A stages through a [BM][BK+8] shared tile, so larger BK does
        // cost shared memory now -- but convEligible enforces the device limit, and
        // fewer, wider slabs still tend to win. Let the sweep judge up to 64.
        return trimCoopmat2Candidates(c,m,n,k,false,false,false,64,1,100000,8,8,64,1.01,4096.0,32.0,128.0);}
      static bool convEligible(TuningContext& x,const Tile& t,int spatial,const VkPhysicalDeviceLimits& limits){
        // The device flex-shape check is load-bearing, not redundant with
        // isConfigSupported: the workgroup-scope fragment IS the BM x BN tile, so a
        // tile the device does not report silently produces wrong results rather
        // than failing. This variant accumulates in FP16.
        // The shared A tile (sharedBytes) adds to the coopmat reserved shared, and
        // the sum must fit the device limit.
        const size_t userShared =
          VulkanKernels::Conv3x3ImplicitGemmCoopmat2AccF16NhwcVec8::sharedBytes(t.bm,t.bk);
        const size_t totalShared = (size_t)x.dev->info.coopmat2ReservedSharedBytes + userShared;
        return VulkanKernels::Conv3x3ImplicitGemmCoopmat2AccF16NhwcVec8::isConfigSupported(
                 t.blockSize,t.bm,t.bn,t.bk) &&
               coopmat2TileSupported(x.dev->info,x.dev->info.coopmat2AccF16FlexShapes,t) &&
               totalShared<=limits.maxComputeSharedMemorySize &&
               (uint32_t)t.blockSize<=limits.maxComputeWorkGroupInvocations &&
               (uint32_t)t.blockSize<=limits.maxComputeWorkGroupSize[0] && spatial%t.bm==0;
      }
      static void applyConv(VulkanTuneParams& c,const Tile& t,int convSize){
        if(convSize == 5) {
          c.conv5x5NhwcCoopmat2AccF16BlockSize=t.blockSize;c.conv5x5NhwcCoopmat2AccF16BM=t.bm;
          c.conv5x5NhwcCoopmat2AccF16BN=t.bn;c.conv5x5NhwcCoopmat2AccF16BK=t.bk;
        } else {
          c.conv3x3NhwcCoopmat2AccF16BlockSize=t.blockSize;c.conv3x3NhwcCoopmat2AccF16BM=t.bm;
          c.conv3x3NhwcCoopmat2AccF16BN=t.bn;c.conv3x3NhwcCoopmat2AccF16BK=t.bk;
        }}
      static void saveConv(VulkanTuneParams& c,const Tile& t,int convSize){applyConv(c,t,convSize);}
      static void markConvValid(VulkanTuneParams& c,int convSize){
        if(convSize == 5) c.conv5x5NhwcCoopmat2AccF16TunerValueValid=1; else c.conv3x3NhwcCoopmat2AccF16TunerValueValid=1;}
      static KernelBench benchConv(TuningContext& x,const VulkanTuneParams& c,int it,int b,int nx,int ny,int s,int convSize,int in,int out){
        return VulkanKernels::Conv3x3ImplicitGemmCoopmat2AccF16NhwcVec8::bench(x,c,it,b,nx,ny,s,convSize,in,out);}
      static KernelBench benchConvReference(
        TuningContext& x,const VulkanTuneParams& c,const std::vector<Tile>& tiles,int it,int b,int nx,int ny,int s,int convSize,int in,int out) {
        if(x.dev->info.supportsCoopmat1F16AccF16) {
          KernelBench ref=VulkanKernels::Conv3x3ImplicitGemmCoopmat1AccF16NhwcVec8::bench(x,c,it,b,nx,ny,s,convSize,in,out);
          if(ref.ok)return ref;
        }
        for(const Tile& tile:tiles) {
          VulkanTuneParams trial=c;applyConv(trial,tile,convSize);
          KernelBench ref=benchConv(x,trial,it,b,nx,ny,s,convSize,in,out);
          if(ref.ok)return ref;
        }
        return KernelBench();
      }
    };

    // NHWC DOT2/DOT2AccF16 Ops for tuneGemmStridedSweep. DOT2's BK is a fixed
    // shader #define (VulkanKernels::DOT2_BK), not a tuned field, so bk() just
    // reports that constant. Unlike the coopmat Ops structs above, dot2 is not
    // used for conv3x3/conv5x5 (no dot2 convolution kernel exists), so this
    // struct omits the whole convLabel/trimConv/convEligible/applyConv family —
    // tuneGemmStridedSweep only calls gemmLabel/bn/trim/applyGemm/saveGemm/benchGemm/
    // format/equal, all of which are provided.
    struct NhwcDot2Ops {
      using Tile = Dot2Tile;
      static string_view gemmLabel(){return "gemmStridedDot2AccF32Nhwc";}
      static int bm(const Tile& t){return t.bm;} static int bn(const Tile& t){return t.bn;}
      static int bk(const Tile&){return (int)VulkanKernels::DOT2_BK;} static int block(const Tile& t){return t.blockSize;}
      static std::string format(const Tile& t){return dot2TileToString(t);}
      static bool equal(const Tile& a,const Tile& b){return dot2TilesEqual(a,b);}
      static std::vector<Tile> trim(const std::vector<Tile>& c,int,int,int){
        std::vector<Tile> trimmed;
        for(const Tile& t: c)
          if(VulkanKernels::GemmStridedDot2Nhwc::isConfigSupported(
               t.blockSize,t.bm,t.bn,t.wm,t.wn,t.wmIter,t.tm,t.tn,t.warp))
            trimmed.push_back(t);
        return trimmed;
      }
      static void applyGemm(VulkanTuneParams& c,const Tile& t){
        c.nhwcStridedDot2BlockSize=t.blockSize;c.nhwcStridedDot2BM=t.bm;c.nhwcStridedDot2BN=t.bn;
        c.nhwcStridedDot2WM=t.wm;c.nhwcStridedDot2WN=t.wn;c.nhwcStridedDot2WMIter=t.wmIter;
        c.nhwcStridedDot2TM=t.tm;c.nhwcStridedDot2TN=t.tn;c.nhwcStridedDot2Warp=t.warp;}
      static void saveGemm(VulkanTuneParams& c,const Tile& t){applyGemm(c,t);c.gemmStridedNhwcDot2TunerValueValid=1;}
      static int32_t& timeUs(VulkanTuneParams& c){return c.nhwcGemmDot2F32TimeUs;}
      static bool splitOutputModes(){return true;}
      static KernelBench benchGemm(
        TuningContext& x,const VulkanTuneParams& c,int it,int b,int m,int n,int k,bool addToOutput){
        return VulkanKernels::GemmStridedDot2Nhwc::bench(x,c,it,b,m,n,k,addToOutput);}
    };
    struct NhwcDot2AccF16Ops {
      using Tile = Dot2Tile;
      static string_view gemmLabel(){return "gemmStridedDot2AccF16Nhwc";}
      static int bm(const Tile& t){return t.bm;} static int bn(const Tile& t){return t.bn;}
      static int bk(const Tile&){return (int)VulkanKernels::DOT2_BK;} static int block(const Tile& t){return t.blockSize;}
      static std::string format(const Tile& t){return dot2TileToString(t);}
      static bool equal(const Tile& a,const Tile& b){return dot2TilesEqual(a,b);}
      static std::vector<Tile> trim(const std::vector<Tile>& c,int,int,int){
        std::vector<Tile> trimmed;
        for(const Tile& t: c)
          if(VulkanKernels::GemmStridedDot2AccF16Nhwc::isConfigSupported(
               t.blockSize,t.bm,t.bn,t.wm,t.wn,t.wmIter,t.tm,t.tn,t.warp))
            trimmed.push_back(t);
        return trimmed;
      }
      static void applyGemm(VulkanTuneParams& c,const Tile& t){
        c.nhwcStridedDot2AccF16BlockSize=t.blockSize;c.nhwcStridedDot2AccF16BM=t.bm;c.nhwcStridedDot2AccF16BN=t.bn;
        c.nhwcStridedDot2AccF16WM=t.wm;c.nhwcStridedDot2AccF16WN=t.wn;c.nhwcStridedDot2AccF16WMIter=t.wmIter;
        c.nhwcStridedDot2AccF16TM=t.tm;c.nhwcStridedDot2AccF16TN=t.tn;c.nhwcStridedDot2AccF16Warp=t.warp;}
      static void saveGemm(VulkanTuneParams& c,const Tile& t){
        applyGemm(c,t);c.gemmStridedNhwcDot2AccF16TunerValueValid=1;}
      static int32_t& timeUs(VulkanTuneParams& c){return c.nhwcGemmDot2F16TimeUs;}
      static bool splitOutputModes(){return true;}
      static KernelBench benchGemm(
        TuningContext& x,const VulkanTuneParams& c,int it,int b,int m,int n,int k,bool addToOutput){
        return VulkanKernels::GemmStridedDot2AccF16Nhwc::bench(x,c,it,b,m,n,k,addToOutput);}
    };

    template<typename Ops>
    void tuneGemmStridedSweep(
      TuningContext& ctx,const ModelDesc* modelDesc,int problemBatchSize,int nnXLen,int nnYLen,int iters,
      const std::vector<typename Ops::Tile>& candidates,std::ostream& out,bool verbose,VulkanTuneParams& cfg) {
      using Tile=typename Ops::Tile;
      std::vector<GemmStridedModeCase> cases=collectGemmStridedModeCases(modelDesc,nnXLen,nnYLen);
      if(cases.empty()||candidates.empty()) return;
      std::vector<Tile> sweep=Ops::trim(candidates,cases[0].M,cases[0].N,cases[0].K);
      struct CachedRef { int m,n,k; KernelBench bench; };
      std::vector<CachedRef> refs;
      auto getRef=[&](int m,int n,int k)->const KernelBench* {
        for(const CachedRef& ref:refs)
          if(ref.m==m&&ref.n==n&&ref.k==k)return &ref.bench;
        VulkanKernels::GemmStridedTiledNhwc probe(problemBatchSize,m,n,k);
        refs.push_back({m,n,k,probe.bench(ctx,cfg,iters)});
        return &refs.back().bench;
      };
      auto bench=[&](const Tile& t,AggregateCandidateMetrics& metrics){
        VulkanTuneParams trial=cfg;Ops::applyGemm(trial,t);metrics={};double flops=0.0,seconds=0.0;
        for(const auto& cs:cases){
          const int n=roundUpCoopmatDim(cs.N,Ops::bn(t));
          const KernelBench* ref=getRef(cs.M,n,cs.K);
          if(ref==nullptr||!ref->ok) return rejectAggregateCandidate(metrics,"reference_failed");
          const int modeOccurrences[2]={
            Ops::splitOutputModes() ? cs.overwriteOccurrences : cs.occurrences,
            Ops::splitOutputModes() ? cs.addToOutputOccurrences : 0};
          for(int mode=0;mode<2;mode++) {
            const int occurrences=modeOccurrences[mode];
            if(occurrences<=0) continue;
            KernelBench b=Ops::benchGemm(ctx,trial,iters,problemBatchSize,cs.M,n,cs.K,mode!=0);
            if(!(b.ok&&b.kernelsPerSecond>0.0)) return rejectAggregateCandidate(metrics,"bench_failed");
            double rmse=normalizedRmse(ref->output,b.output);metrics.maxRmse=std::max(metrics.maxRmse,rmse);
            if(rmse>0.02) return rejectAggregateCandidate(metrics,"rmse_exceeded");
            double wt=dispatchWeightedTime(b,occurrences);metrics.weightedTime+=wt;seconds+=wt;
            flops+=2.0*problemBatchSize*cs.M*n*cs.K*occurrences;
          }
        } metrics.tflops=aggregateTflops(flops,seconds);return metrics.weightedTime>0.0;};
      const string_view label=Ops::gemmLabel();
      auto result=tuneAggregateCandidateSweep(TunerStreamText{label},TunerStreamText{label},
        TunerStreamText{label," tile sweep"},"tile",TunerStreamText{"VulkanTuner: strided GEMM found no valid tile."},
        sweep,TUNER_CONFIRM_PREFIX_K,true,true,true,out,verbose,ctx,bench,Ops::format,Ops::equal);
      if(result.found) {
        Ops::saveGemm(cfg,result.candidate);
        ModeSelectStats stats{true,result.weightedTime,0.0};
        Ops::timeUs(cfg)=weightedTimeToMicros(stats);
      }
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
        ctx,
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
        !incumbentCoopmat2AccF16 && !incumbentCoopmat2 && tunedConfig.enableWinogradGemmCoopmat1AccF16 != 0;
      const bool incumbentCoopmat = !incumbentCoopmat2AccF16 && !incumbentCoopmat2 && !incumbentCoopmatAccF16 &&
                                    tunedConfig.enableWinogradGemmCoopmat1 != 0;
      const bool incumbentDot2AccF16 = !incumbentCoopmat2AccF16 && !incumbentCoopmat2 && !incumbentCoopmatAccF16 &&
                                       !incumbentCoopmat && tunedConfig.enableWinogradGemmDot2AccF16 != 0;
      const bool incumbentDot2 = !incumbentCoopmat2AccF16 && !incumbentCoopmat2 && !incumbentCoopmatAccF16 &&
                                 !incumbentCoopmat && !incumbentDot2AccF16 && tunedConfig.enableWinogradGemmDot2 != 0;
      string_view incumbentName = incumbentCoopmat2AccF16  ? "winogradGemmCoopmat2AccF16"
                                  : incumbentCoopmat2      ? "winogradGemmCoopmat2"
                                  : incumbentCoopmatAccF16 ? "winogradGemmCoopmat1AccF16"
                                  : incumbentCoopmat       ? "winogradGemmCoopmat1"
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
            padDims(cs.M, cs.N, cs.K, VulkanKernels::WinogradGemmCoopmat1AccF16::layerPaddingContract(tunedConfig));
          VulkanKernels::WinogradGemmCoopmat1AccF16 probe(pad.m, pad.n, pad.k, cs.numBatches);
          incumbentBench = probe.bench(ctx, tunedConfig, iters);
          incumbentTflops = probe.estimatedTflops(tunedConfig, incumbentBench.kernelsPerSecond);
          incumbentFlopsPerDispatch = probe.estimatedFlopsPerDispatch(tunedConfig);
        } else if(incumbentCoopmat) {
          const auto pad =
            padDims(cs.M, cs.N, cs.K, VulkanKernels::WinogradGemmCoopmat1::layerPaddingContract(tunedConfig));
          VulkanKernels::WinogradGemmCoopmat1 probe(pad.m, pad.n, pad.k, cs.numBatches);
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
      (accF16 ? tunedConfig.winogradGemmCoopmat2F16TimeUs : tunedConfig.winogradGemmCoopmat2F32TimeUs) =
        weightedTimeToMicros(cm2Stats);
      bool selected = appendModeSelectResult(
        out,
        "winogradGemm",
        incumbentName,
        challengerName,
        cases,
        incumbentStats,
        cm2Stats,
        maxRmse,
        !incumbentCoopmat2AccF16 && !incumbentCoopmat2 && !incumbentCoopmatAccF16 && !incumbentCoopmat,
        false,
        false,
        [](std::ostream&) {});
      if(selected) {
        if(accF16)
          tunedConfig.enableWinogradGemmCoopmat2AccF16 = 1;
        else
          tunedConfig.enableWinogradGemmCoopmat2 = 1;
        tunedConfig.enableWinogradGemmCoopmat2 = accF16 ? 0 : tunedConfig.enableWinogradGemmCoopmat2;
        tunedConfig.enableWinogradGemmCoopmat2AccF16 = accF16 ? tunedConfig.enableWinogradGemmCoopmat2AccF16 : 0;
        tunedConfig.enableWinogradGemmCoopmat1 = 0;
        tunedConfig.enableWinogradGemmCoopmat1AccF16 = 0;
        tunedConfig.enableWinogradGemmDot2 = 0;
        tunedConfig.enableWinogradGemmDot2AccF16 = 0;
      }
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
      const bool supportsFP16Compute = ctx.fp16Compute;
      const int problemBatchSize = std::max(1, batchSize);
      const bool fp16Ready = ctx.fp16Storage && supportsFP16Compute;

      if(tuningUsesNhwc()) {
        const bool supportsNhwcCoop = fp16Ready &&
          (ctx.dev->info.supportsCoopmat1F16 || ctx.dev->info.supportsCoopmat1F16AccF16 ||
           ctx.dev->info.supportsCoopmat2F16 || ctx.dev->info.supportsCoopmat2F16AccF16);
        const bool supportsNhwcDot2 = fp16Ready &&
          (ctx.dev->info.supportsDot2F16 || ctx.dev->info.supportsDot2F16AccF16);
        // Sweep tiled NHWC strided GEMM when its marker is missing.
        if(!supportsNhwcCoop && !supportsNhwcDot2 &&
           (tunedConfig.tunedKernelMask & TUNED_NHWC_STRIDED) == 0) {
          out << endl << "VulkanTuner: tuning gemmStridedTiledNhwc..." << endl;
          VulkanTuneParams tunedCandidate = tunedConfig;
          bool tuned = tuneGemmStridedTiledMultiShape<VulkanKernels::GemmStridedTiledNhwc>(
            "gemmStridedTiledNhwc",
            modelDesc, problemBatchSize, nnXLen, nnYLen,
            tunedConfig, ctx, iters, out, verboseTuner, tunedCandidate);
          if(tuned)
            tunedConfig = tunedCandidate;
        }
        if(ctx.dev->info.supportsCoopmat1F16 && ctx.fp16Storage && supportsFP16Compute &&
           tunedConfig.gemmStridedNhwcCoopmat1TunerValueValid == 0) {
          tunedConfig.nhwcGemmCoopmat1F32TimeUs = 0;
          std::vector<CoopmatTile> candidates = makeCoopmatCandidates(ctx, ctx.dev->info.coopmatShapes);
          tuneGemmStridedSweep<NhwcCoopmat1F32Ops>(ctx,modelDesc,problemBatchSize,nnXLen,nnYLen,iters,
            candidates,out,verboseTuner,tunedConfig);
        }
        if(ctx.dev->info.supportsCoopmat1F16AccF16 && ctx.fp16Storage && supportsFP16Compute &&
           tunedConfig.gemmStridedNhwcCoopmat1AccF16TunerValueValid == 0) {
          tunedConfig.nhwcGemmCoopmat1F16TimeUs = 0;
          std::vector<CoopmatTile> candidates = makeCoopmatCandidates(ctx, ctx.dev->info.coopmatAccF16Shapes);
          tuneGemmStridedSweep<NhwcCoopmat1Ops>(ctx,modelDesc,problemBatchSize,nnXLen,nnYLen,iters,
            candidates,out,verboseTuner,tunedConfig);
        }
        if(ctx.dev->info.supportsCoopmat2F16 && ctx.fp16Storage && supportsFP16Compute &&
           tunedConfig.gemmStridedNhwcCoopmat2TunerValueValid == 0) {
          tunedConfig.nhwcGemmCoopmat2F32TimeUs = 0;
          std::vector<Coopmat2Tile> candidates=makeCoopmat2Candidates(ctx,ctx.dev->info.coopmat2FlexShapes);
          tuneGemmStridedSweep<NhwcCoopmat2F32Ops>(ctx,modelDesc,problemBatchSize,nnXLen,nnYLen,iters,
            candidates,out,verboseTuner,tunedConfig);
        }
        if(ctx.dev->info.supportsCoopmat2F16AccF16 && ctx.fp16Storage && supportsFP16Compute &&
           tunedConfig.gemmStridedNhwcCoopmat2AccF16TunerValueValid == 0) {
          tunedConfig.nhwcGemmCoopmat2F16TimeUs = 0;
          std::vector<Coopmat2Tile> candidates=makeCoopmat2Candidates(ctx,ctx.dev->info.coopmat2AccF16FlexShapes);
          tuneGemmStridedSweep<NhwcCoopmat2Ops>(ctx,modelDesc,problemBatchSize,nnXLen,nnYLen,iters,
            candidates,out,verboseTuner,tunedConfig);
        }
        const bool haveNhwcCoop =
          (ctx.dev->info.supportsCoopmat1F16 &&
           tunedConfig.gemmStridedNhwcCoopmat1TunerValueValid != 0) ||
          (ctx.dev->info.supportsCoopmat1F16AccF16 &&
           tunedConfig.gemmStridedNhwcCoopmat1AccF16TunerValueValid != 0) ||
          (ctx.dev->info.supportsCoopmat2F16 &&
           tunedConfig.gemmStridedNhwcCoopmat2TunerValueValid != 0) ||
          (ctx.dev->info.supportsCoopmat2F16AccF16 &&
           tunedConfig.gemmStridedNhwcCoopmat2AccF16TunerValueValid != 0);
        if(!haveNhwcCoop && ctx.dev->info.supportsDot2F16 && fp16Ready &&
           tunedConfig.gemmStridedNhwcDot2TunerValueValid == 0) {
          tunedConfig.nhwcGemmDot2F32TimeUs = 0;
          std::vector<Dot2Tile> candidates(
            kDot2Candidates, kDot2Candidates + sizeof(kDot2Candidates) / sizeof(kDot2Candidates[0]));
          tuneGemmStridedSweep<NhwcDot2Ops>(ctx,modelDesc,problemBatchSize,nnXLen,nnYLen,iters,
            candidates,out,verboseTuner,tunedConfig);
        }
        if(!haveNhwcCoop && ctx.dev->info.supportsDot2F16AccF16 && fp16Ready &&
           tunedConfig.gemmStridedNhwcDot2AccF16TunerValueValid == 0) {
          tunedConfig.nhwcGemmDot2F16TimeUs = 0;
          std::vector<Dot2Tile> candidates(
            kDot2Candidates, kDot2Candidates + sizeof(kDot2Candidates) / sizeof(kDot2Candidates[0]));
          tuneGemmStridedSweep<NhwcDot2AccF16Ops>(ctx,modelDesc,problemBatchSize,nnXLen,nnYLen,iters,
            candidates,out,verboseTuner,tunedConfig);
        }
        const bool haveNhwcDot2 =
          (ctx.dev->info.supportsDot2F16 && tunedConfig.gemmStridedNhwcDot2TunerValueValid != 0) ||
          (ctx.dev->info.supportsDot2F16AccF16 && tunedConfig.gemmStridedNhwcDot2AccF16TunerValueValid != 0);
        if((supportsNhwcCoop || supportsNhwcDot2) && !haveNhwcCoop && !haveNhwcDot2 &&
           !collectGemmStridedModeCases(modelDesc, nnXLen, nnYLen).empty()) {
          VulkanTuneParams candidate = tunedConfig;
          if(tuneGemmStridedTiledMultiShape<VulkanKernels::GemmStridedTiledNhwc>(
               "gemmStridedTiledNhwc", modelDesc, problemBatchSize, nnXLen, nnYLen,
               tunedConfig, ctx, iters, out, verboseTuner, candidate))
            tunedConfig = candidate;
        }
        if(tunedConfig.conv3x3NhwcCoopmat1TunerValueValid == 0 && ctx.dev->info.supportsCoopmat1F16 &&
           ctx.fp16Storage && supportsFP16Compute) {
          std::vector<CoopmatTile> candidates = makeCoopmatCandidates(ctx, ctx.dev->info.coopmatShapes);
          tuneNhwcConvSweep<NhwcCoopmat1F32Ops>(
            ctx,
            modelDesc,
            problemBatchSize,
            nnXLen,
            nnYLen,
            3,
            iters,
            candidates,
            out,
            verboseTuner,
            tunedConfig);
        }
        if(tunedConfig.conv3x3NhwcCoopmat2TunerValueValid == 0 && ctx.dev->info.supportsCoopmat2F16 &&
           ctx.fp16Storage && supportsFP16Compute) {
          std::vector<Coopmat2Tile> candidates=makeCoopmat2Candidates(ctx,ctx.dev->info.coopmat2FlexShapes,64);
          tuneNhwcConvSweep<NhwcCoopmat2F32Ops>(
            ctx,modelDesc,problemBatchSize,nnXLen,nnYLen,3,iters,candidates,out,verboseTuner,tunedConfig);
        }
        if(tunedConfig.conv3x3NhwcCoopmat1AccF16TunerValueValid == 0 && ctx.dev->info.supportsCoopmat1F16AccF16 &&
           ctx.fp16Storage && supportsFP16Compute) {
          std::vector<CoopmatTile> candidates = makeCoopmatCandidates(ctx, ctx.dev->info.coopmatAccF16Shapes);
          tuneNhwcConvSweep<NhwcCoopmat1Ops>(
            ctx,modelDesc,problemBatchSize,nnXLen,nnYLen,3,iters,candidates,out,verboseTuner,tunedConfig);
        }
        if(tunedConfig.conv3x3NhwcCoopmat2AccF16TunerValueValid == 0 && ctx.dev->info.supportsCoopmat2F16AccF16 &&
          ctx.fp16Storage && supportsFP16Compute) {
          std::vector<Coopmat2Tile> candidates=makeCoopmat2Candidates(ctx,ctx.dev->info.coopmat2AccF16FlexShapes,64);
          tuneNhwcConvSweep<NhwcCoopmat2Ops>(
            ctx,modelDesc,problemBatchSize,nnXLen,nnYLen,3,iters,candidates,out,verboseTuner,tunedConfig);
        }
        if(tunedConfig.conv5x5NhwcCoopmat1TunerValueValid == 0 && ctx.dev->info.supportsCoopmat1F16 &&
          ctx.fp16Storage && supportsFP16Compute) {
          std::vector<CoopmatTile> candidates = makeCoopmatCandidates(ctx, ctx.dev->info.coopmatShapes);
          tuneNhwcConvSweep<NhwcCoopmat1F32Ops>(
            ctx,modelDesc,problemBatchSize,nnXLen,nnYLen,5,iters,candidates,out,verboseTuner,tunedConfig);
        }
        if(tunedConfig.conv5x5NhwcCoopmat2TunerValueValid == 0 && ctx.dev->info.supportsCoopmat2F16 &&
          ctx.fp16Storage && supportsFP16Compute) {
          std::vector<Coopmat2Tile> candidates=makeCoopmat2Candidates(ctx,ctx.dev->info.coopmat2FlexShapes,64);
          tuneNhwcConvSweep<NhwcCoopmat2F32Ops>(
            ctx,modelDesc,problemBatchSize,nnXLen,nnYLen,5,iters,candidates,out,verboseTuner,tunedConfig);
        }
        if(tunedConfig.conv5x5NhwcCoopmat1AccF16TunerValueValid == 0 && ctx.dev->info.supportsCoopmat1F16AccF16 &&
           ctx.fp16Storage && supportsFP16Compute) {
          std::vector<CoopmatTile> candidates = makeCoopmatCandidates(ctx, ctx.dev->info.coopmatAccF16Shapes);
          tuneNhwcConvSweep<NhwcCoopmat1Ops>(
            ctx,modelDesc,problemBatchSize,nnXLen,nnYLen,5,iters,candidates,out,verboseTuner,tunedConfig);
        }
        if(tunedConfig.conv5x5NhwcCoopmat2AccF16TunerValueValid == 0 && ctx.dev->info.supportsCoopmat2F16AccF16 &&
           ctx.fp16Storage && supportsFP16Compute) {
          std::vector<Coopmat2Tile> candidates=makeCoopmat2Candidates(ctx,ctx.dev->info.coopmat2AccF16FlexShapes,64);
          tuneNhwcConvSweep<NhwcCoopmat2Ops>(
            ctx,modelDesc,problemBatchSize,nnXLen,nnYLen,5,iters,candidates,out,verboseTuner,tunedConfig);
        }
        const bool needsImplicit3 = !collectConv3x3NhwcModeCases(modelDesc, 3).empty();
        const bool needsImplicit5 = !collectConv3x3NhwcModeCases(modelDesc, 5).empty();
        const bool haveImplicit3 = !needsImplicit3 ||
          (fp16Ready &&
           ((ctx.dev->info.supportsCoopmat1F16 && tunedConfig.conv3x3NhwcCoopmat1TunerValueValid != 0) ||
            (ctx.dev->info.supportsCoopmat2F16 && tunedConfig.conv3x3NhwcCoopmat2TunerValueValid != 0) ||
            (ctx.dev->info.supportsCoopmat1F16AccF16 &&
             tunedConfig.conv3x3NhwcCoopmat1AccF16TunerValueValid != 0) ||
            (ctx.dev->info.supportsCoopmat2F16AccF16 &&
             tunedConfig.conv3x3NhwcCoopmat2AccF16TunerValueValid != 0)));
        const bool haveImplicit5 = !needsImplicit5 ||
          (fp16Ready &&
           ((ctx.dev->info.supportsCoopmat1F16 && tunedConfig.conv5x5NhwcCoopmat1TunerValueValid != 0) ||
            (ctx.dev->info.supportsCoopmat2F16 && tunedConfig.conv5x5NhwcCoopmat2TunerValueValid != 0) ||
            (ctx.dev->info.supportsCoopmat1F16AccF16 &&
             tunedConfig.conv5x5NhwcCoopmat1AccF16TunerValueValid != 0) ||
            (ctx.dev->info.supportsCoopmat2F16AccF16 &&
             tunedConfig.conv5x5NhwcCoopmat2AccF16TunerValueValid != 0)));
        const bool implicitCoopAvailable = haveImplicit3 && haveImplicit5;
        if(!implicitCoopAvailable) {
          const bool needsWinograd =
            !collectWinogradModeCases(modelDesc, problemBatchSize, nnXLen, nnYLen, tunedConfig).empty();
          const bool supportsWinogradCoop = fp16Ready &&
            (ctx.dev->info.supportsCoopmat1F16 || ctx.dev->info.supportsCoopmat1F16AccF16 ||
             ctx.dev->info.supportsCoopmat2F16 || ctx.dev->info.supportsCoopmat2F16AccF16);
          if(needsWinograd && supportsWinogradCoop) {
            if(ctx.dev->info.supportsCoopmat1F16 && tunedConfig.winogradGemmCoopmat1TunerValueValid == 0) {
              tunedConfig.winogradGemmCoopmat1F32TimeUs = 0;
              auto candidates = makeCoopmatCandidates(ctx, ctx.dev->info.coopmatShapes);
              tuneWinogradCoopmatTileSweep(
                ctx, modelDesc, problemBatchSize, nnXLen, nnYLen, iters, candidates, false, out, verboseTuner, tunedConfig);
            }
            if(ctx.dev->info.supportsCoopmat1F16AccF16 &&
               tunedConfig.winogradGemmCoopmat1AccF16TunerValueValid == 0) {
              tunedConfig.winogradGemmCoopmat1F16TimeUs = 0;
              auto candidates = makeCoopmatCandidates(ctx, ctx.dev->info.coopmatAccF16Shapes);
              tuneWinogradCoopmatTileSweep(
                ctx, modelDesc, problemBatchSize, nnXLen, nnYLen, iters, candidates, true, out, verboseTuner, tunedConfig);
            }
            if(ctx.dev->info.supportsCoopmat2F16 && tunedConfig.winogradGemmCoopmat2TunerValueValid == 0) {
              tunedConfig.winogradGemmCoopmat2F32TimeUs = 0;
              auto candidates = makeCoopmat2Candidates(ctx, ctx.dev->info.coopmat2FlexShapes);
              tuneWinogradCoopmat2TileSweep(
                ctx, modelDesc, problemBatchSize, nnXLen, nnYLen, iters, candidates, false, out, verboseTuner, tunedConfig);
            }
            if(ctx.dev->info.supportsCoopmat2F16AccF16 &&
               tunedConfig.winogradGemmCoopmat2AccF16TunerValueValid == 0) {
              tunedConfig.winogradGemmCoopmat2F16TimeUs = 0;
              auto candidates = makeCoopmat2Candidates(ctx, ctx.dev->info.coopmat2AccF16FlexShapes);
              tuneWinogradCoopmat2TileSweep(
                ctx, modelDesc, problemBatchSize, nnXLen, nnYLen, iters, candidates, true, out, verboseTuner, tunedConfig);
            }
          }
          const bool haveWinogradCoop =
            (ctx.dev->info.supportsCoopmat1F16 && tunedConfig.winogradGemmCoopmat1TunerValueValid != 0) ||
            (ctx.dev->info.supportsCoopmat1F16AccF16 &&
             tunedConfig.winogradGemmCoopmat1AccF16TunerValueValid != 0) ||
            (ctx.dev->info.supportsCoopmat2F16 && tunedConfig.winogradGemmCoopmat2TunerValueValid != 0) ||
            (ctx.dev->info.supportsCoopmat2F16AccF16 &&
             tunedConfig.winogradGemmCoopmat2AccF16TunerValueValid != 0);
          if(needsWinograd && !haveWinogradCoop && supportsNhwcDot2) {
            if(ctx.dev->info.supportsDot2F16 && tunedConfig.winogradGemmDot2TunerValueValid == 0) {
              tunedConfig.winogradGemmDot2F32TimeUs = 0;
              tuneWinogradDot2TileSweep(
                ctx, modelDesc, problemBatchSize, nnXLen, nnYLen, iters, out, verboseTuner, false, tunedConfig);
            }
            if(ctx.dev->info.supportsDot2F16AccF16 && tunedConfig.winogradGemmDot2AccF16TunerValueValid == 0) {
              tunedConfig.winogradGemmDot2F16TimeUs = 0;
              tuneWinogradDot2TileSweep(
                ctx, modelDesc, problemBatchSize, nnXLen, nnYLen, iters, out, verboseTuner, true, tunedConfig);
            }
          }
          const bool haveWinogradDot2 =
            (ctx.dev->info.supportsDot2F16 && tunedConfig.winogradGemmDot2TunerValueValid != 0) ||
            (ctx.dev->info.supportsDot2F16AccF16 && tunedConfig.winogradGemmDot2AccF16TunerValueValid != 0);
          if(needsWinograd && !haveWinogradCoop && !haveWinogradDot2) {
            VulkanTuneParams candidate = tunedConfig;
            if(tuneWinogradGemmTiledMultiShape<VulkanKernels::WinogradGemm>(
                 "winogradGemmTiled", modelDesc, problemBatchSize, nnXLen, nnYLen,
                 tunedConfig, ctx, iters, out, verboseTuner, candidate))
              tunedConfig = candidate;
          }
        }
        return;
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
      const bool fp16Ready = info != nullptr && ctx.fp16Storage && ctx.fp16Compute;
      auto clearSelections = [&]() {
        tunedConfig.enableWinogradGemmCoopmat1 = 0;
        tunedConfig.enableWinogradGemmCoopmat1AccF16 = 0;
        tunedConfig.enableWinogradGemmCoopmat2 = 0;
        tunedConfig.enableWinogradGemmCoopmat2AccF16 = 0;
        tunedConfig.enableWinogradGemmDot2 = 0;
        tunedConfig.enableWinogradGemmDot2AccF16 = 0;
      };
      clearSelections();

      const int problemBatchSize = std::max(1, batchSize);
      auto selectWinogradTier = [&]() {
        const bool cm1 = fp16Ready && info->supportsCoopmat1F16 &&
          tunedConfig.winogradGemmCoopmat1TunerValueValid != 0;
        const bool cm1f16 = fp16Ready && info->supportsCoopmat1F16AccF16 &&
          tunedConfig.winogradGemmCoopmat1AccF16TunerValueValid != 0;
        const bool cm2 = fp16Ready && info->supportsCoopmat2F16 &&
          tunedConfig.winogradGemmCoopmat2TunerValueValid != 0;
        const bool cm2f16 = fp16Ready && info->supportsCoopmat2F16AccF16 &&
          tunedConfig.winogradGemmCoopmat2AccF16TunerValueValid != 0;
        if(cm1 || cm1f16 || cm2 || cm2f16) {
          const bool scoresComplete = (!cm1 || tunedConfig.winogradGemmCoopmat1F32TimeUs > 0) &&
            (!cm1f16 || tunedConfig.winogradGemmCoopmat1F16TimeUs > 0) &&
            (!cm2 || tunedConfig.winogradGemmCoopmat2F32TimeUs > 0) &&
            (!cm2f16 || tunedConfig.winogradGemmCoopmat2F16TimeUs > 0);
          if(!scoresComplete) {
            if(cm1) tuneWinogradCoopmatModeSelect(
              ctx, modelDesc, problemBatchSize, nnXLen, nnYLen, iters, false, out, verboseTuner, tunedConfig);
            if(cm1f16) tuneWinogradCoopmatModeSelect(
              ctx, modelDesc, problemBatchSize, nnXLen, nnYLen, iters, true, out, verboseTuner, tunedConfig);
            if(cm2) tuneWinogradCoopmat2ModeSelect(
              ctx, modelDesc, problemBatchSize, nnXLen, nnYLen, iters, false, out, verboseTuner, tunedConfig);
            if(cm2f16) tuneWinogradCoopmat2ModeSelect(
              ctx, modelDesc, problemBatchSize, nnXLen, nnYLen, iters, true, out, verboseTuner, tunedConfig);
          }
          return;
        }
        const bool dot = fp16Ready && info->supportsDot2F16 && tunedConfig.winogradGemmDot2TunerValueValid != 0;
        const bool dotf16 = fp16Ready && info->supportsDot2F16AccF16 &&
          tunedConfig.winogradGemmDot2AccF16TunerValueValid != 0;
        const bool scoresComplete = (!dot || tunedConfig.winogradGemmDot2F32TimeUs > 0) &&
          (!dotf16 || tunedConfig.winogradGemmDot2F16TimeUs > 0);
        if(!scoresComplete) {
          if(dot) tuneWinogradGemmDot2ModeSelect(
            ctx, modelDesc, problemBatchSize, nnXLen, nnYLen, iters, out, verboseTuner, tunedConfig);
          if(dotf16) tuneWinogradGemmDot2AccF16ModeSelect(
            ctx, modelDesc, problemBatchSize, nnXLen, nnYLen, iters, out, verboseTuner, tunedConfig);
        }
      };

      if(tuningUsesNhwc()) {
        tunedConfig.nhwcGemmF32UseCoopmat2 = 0;
        tunedConfig.nhwcGemmUseCoopmat2 = 0;
        tunedConfig.nhwcGemmUseCoopmatAccF16 = 0;
        tunedConfig.nhwcGemmUseDot2 = 0;
        tunedConfig.nhwcGemmUseDot2AccF16 = 0;
        const bool cm1 = fp16Ready && info->supportsCoopmat1F16AccF16 &&
          tunedConfig.gemmStridedNhwcCoopmat1AccF16TunerValueValid != 0;
        const bool cm2 = fp16Ready && info->supportsCoopmat2F16AccF16 &&
          tunedConfig.gemmStridedNhwcCoopmat2AccF16TunerValueValid != 0;
        const bool cm1f32 = fp16Ready && info->supportsCoopmat1F16 &&
          tunedConfig.gemmStridedNhwcCoopmat1TunerValueValid != 0;
        const bool cm2f32 = fp16Ready && info->supportsCoopmat2F16 &&
          tunedConfig.gemmStridedNhwcCoopmat2TunerValueValid != 0;
        std::vector<GemmStridedModeCase> cases = collectGemmStridedModeCases(modelDesc, nnXLen, nnYLen);
        auto storeTimeUs = [](double seconds) {
          if(!(seconds > 0.0) || !std::isfinite(seconds))
            return int32_t(0);
          const double micros = std::round(seconds * 1.0e6);
          return (int32_t)std::max(1.0, std::min(micros, (double)std::numeric_limits<int32_t>::max()));
        };
        auto scoreCandidate = [&](bool available, int32_t& storedTimeUs, const std::function<double(bool&)>& measure) {
          if(!available)
            return int32_t(0);
          if(storedTimeUs > 0)
            return storedTimeUs;
          bool ok = false;
          const double seconds = measure(ok);
          if(ok)
            storedTimeUs = storeTimeUs(seconds);
          return storedTimeUs;
        };
        auto measureF32Coop = [&](bool coop2, bool& ok) {
          double total = 0.0; ok = true;
          for(const auto& cs: cases) {
            const int occurrences[2] = {cs.overwriteOccurrences, cs.addToOutputOccurrences};
            const int n = roundUpCoopmatDim(
              cs.N, coop2 ? tunedConfig.nhwcStridedCoopmat2BN : tunedConfig.nhwcStridedCoopmat1BN);
            for(int mode = 0; mode < 2; mode++) {
              if(occurrences[mode] <= 0) continue;
              KernelBench b = coop2 ? VulkanKernels::GemmStridedCoopmat2Nhwc::bench(
                ctx, tunedConfig, iters, batchSize, cs.M, n, cs.K, mode != 0) :
                VulkanKernels::GemmStridedCoopmat1Nhwc::bench(
                  ctx, tunedConfig, iters, batchSize, cs.M, n, cs.K, mode != 0);
              if(b.ok && b.kernelsPerSecond > 0.0) total += occurrences[mode] / b.kernelsPerSecond;
              else ok = false;
            }
          }
          return total;
        };
        auto measureF16Coop = [&](bool coop2, bool& ok) {
          double total = 0.0;
          ok = true;
          for(const auto& cs: cases) {
            const int occurrences[2] = {cs.overwriteOccurrences, cs.addToOutputOccurrences};
            for(int mode = 0; mode < 2; mode++) {
              if(occurrences[mode] <= 0) continue;
              const int n = roundUpCoopmatDim(
                cs.N, coop2 ? tunedConfig.nhwcStridedCoopmat2AccF16BN :
                                tunedConfig.nhwcStridedCoopmat1AccF16BN);
              KernelBench b = coop2 ? VulkanKernels::GemmStridedCoopmat2AccF16Nhwc::bench(
                ctx, tunedConfig, iters, batchSize, cs.M, n, cs.K, mode != 0) :
                VulkanKernels::GemmStridedCoopmat1AccF16Nhwc::bench(
                  ctx, tunedConfig, iters, batchSize, cs.M, n, cs.K, mode != 0);
              if(b.ok && b.kernelsPerSecond > 0.0) total += occurrences[mode] / b.kernelsPerSecond;
              else ok = false;
            }
          }
          return total;
        };
        const int32_t cm1F32Score = scoreCandidate(
          cm1f32, tunedConfig.nhwcGemmCoopmat1F32TimeUs,
          [&](bool& ok) { return measureF32Coop(false, ok); });
        const int32_t cm2F32Score = scoreCandidate(
          cm2f32, tunedConfig.nhwcGemmCoopmat2F32TimeUs,
          [&](bool& ok) { return measureF32Coop(true, ok); });
        const int32_t cm1F16Score = scoreCandidate(
          cm1, tunedConfig.nhwcGemmCoopmat1F16TimeUs,
          [&](bool& ok) { return measureF16Coop(false, ok); });
        const int32_t cm2F16Score = scoreCandidate(
          cm2, tunedConfig.nhwcGemmCoopmat2F16TimeUs,
          [&](bool& ok) { return measureF16Coop(true, ok); });
        tunedConfig.nhwcGemmF32UseCoopmat2 =
          cm2F32Score > 0 && (cm1F32Score == 0 || cm2F32Score < cm1F32Score);
        tunedConfig.nhwcGemmUseCoopmat2 =
          cm2F16Score > 0 && (cm1F16Score == 0 || cm2F16Score < cm1F16Score);

        if(cm1f32 || cm2f32 || cm1 || cm2) {
          const int32_t f32Score = tunedConfig.nhwcGemmF32UseCoopmat2 != 0 ? cm2F32Score : cm1F32Score;
          const int32_t f16Score = tunedConfig.nhwcGemmUseCoopmat2 != 0 ? cm2F16Score : cm1F16Score;
          tunedConfig.nhwcGemmUseCoopmatAccF16 = f16Score > 0 &&
            (f32Score == 0 || (int64_t)f16Score * 5 < (int64_t)f32Score * 4);
          const bool selectedF16 = tunedConfig.nhwcGemmUseCoopmatAccF16 != 0;
          out << "VulkanTuner: NHWC 1x1 coopmat winner="
              << ((selectedF16 ? tunedConfig.nhwcGemmUseCoopmat2 : tunedConfig.nhwcGemmF32UseCoopmat2) ?
                    "coopmat2" : "coopmat1")
              << " accumulation=" << (selectedF16 ? "fp16" : "fp32")
              << " scores_us=[" << cm1F32Score << "," << cm2F32Score << ","
              << cm1F16Score << "," << cm2F16Score << "]" << endl;
        } else {
          const bool dot = fp16Ready && info->supportsDot2F16 &&
            tunedConfig.gemmStridedNhwcDot2TunerValueValid != 0;
          const bool dotf16 = fp16Ready && info->supportsDot2F16AccF16 &&
            tunedConfig.gemmStridedNhwcDot2AccF16TunerValueValid != 0;
          auto dotTime = [&](bool accF16, bool& ok) {
            double total = 0.0; ok = true;
            for(const auto& cs: cases) {
              const int occurrences[2] = {cs.overwriteOccurrences, cs.addToOutputOccurrences};
              for(int mode = 0; mode < 2; mode++) {
                if(occurrences[mode] <= 0) continue;
                const int n = roundUpCoopmatDim(
                  cs.N, accF16 ? tunedConfig.nhwcStridedDot2AccF16BN : tunedConfig.nhwcStridedDot2BN);
                KernelBench b = accF16
                  ? VulkanKernels::GemmStridedDot2AccF16Nhwc::bench(
                      ctx, tunedConfig, iters, batchSize, cs.M, n, cs.K, mode != 0)
                  : VulkanKernels::GemmStridedDot2Nhwc::bench(
                      ctx, tunedConfig, iters, batchSize, cs.M, n, cs.K, mode != 0);
                if(b.ok && b.kernelsPerSecond > 0.0) total += occurrences[mode] / b.kernelsPerSecond;
                else ok = false;
              }
            }
            return total;
          };
          const int32_t dotF32Score = scoreCandidate(
            dot, tunedConfig.nhwcGemmDot2F32TimeUs,
            [&](bool& ok) { return dotTime(false, ok); });
          const int32_t dotF16Score = scoreCandidate(
            dotf16, tunedConfig.nhwcGemmDot2F16TimeUs,
            [&](bool& ok) { return dotTime(true, ok); });
          if(dotF16Score > 0 &&
             (dotF32Score == 0 || (int64_t)dotF16Score * 5 < (int64_t)dotF32Score * 4))
            tunedConfig.nhwcGemmUseDot2AccF16 = 1;
          else if(dotF32Score > 0)
            tunedConfig.nhwcGemmUseDot2 = 1;
          out << "VulkanTuner: NHWC 1x1 dot2 accumulation="
              << (tunedConfig.nhwcGemmUseDot2AccF16 != 0 ? "fp16" : "fp32")
              << " scores_us=[" << dotF32Score << "," << dotF16Score << "]" << endl;
        }

        auto selectConvMode = [&](int convSize, int valid1, int valid2, int32_t& useCoopmat2) {
          const bool have1 = fp16Ready && info->supportsCoopmat1F16 && valid1 != 0;
          const bool have2 = fp16Ready && info->supportsCoopmat2F16 && valid2 != 0;
          useCoopmat2 = have2 && !have1 ? 1 : 0;
          if(!have1 || !have2)
            return;
          auto convCases = collectConv3x3NhwcModeCases(modelDesc, convSize);
          const int spatial = roundUpToMultipleInt(nnXLen * nnYLen, VulkanKernels::VULKAN_SPATIAL_ALIGN);
          double t1 = 0, t2 = 0; bool ok1 = true, ok2 = true;
          for(const auto& cs: convCases) {
            const int batches[2] = {1, std::max(1, batchSize)};
            const int count = batches[0] == batches[1] ? 1 : 2;
            for(int bi = 0; bi < count; bi++) {
              const int bs = batches[bi];
              const double w = (bs == 1 && batchSize > 1 ? 4.0 : 1.0) * cs.occurrences;
              KernelBench b1 = VulkanKernels::Conv3x3ImplicitGemmCoopmat1AccF32NhwcVec8::bench(
                ctx,tunedConfig,iters,bs,nnXLen,nnYLen,spatial,convSize,cs.inChannels,cs.outChannels);
              KernelBench b2 = VulkanKernels::Conv3x3ImplicitGemmCoopmat2AccF32NhwcVec8::bench(
                ctx,tunedConfig,iters,bs,nnXLen,nnYLen,spatial,convSize,cs.inChannels,cs.outChannels);
              if(b1.ok && b1.kernelsPerSecond > 0) t1 += w / b1.kernelsPerSecond; else ok1 = false;
              if(b2.ok && b2.kernelsPerSecond > 0) {
                t2 += w / b2.kernelsPerSecond;
                if(!b1.ok || normalizedRmse(b1.output,b2.output) > 0.02) ok2 = false;
              } else ok2 = false;
            }
          }
          useCoopmat2 = (ok2 && (!ok1 || challengerWinsModeSelect(t2,t1))) ? 1 : 0;
        };
        selectConvMode(3, tunedConfig.conv3x3NhwcCoopmat1TunerValueValid,
          tunedConfig.conv3x3NhwcCoopmat2TunerValueValid, tunedConfig.nhwcConv3x3UseCoopmat2);
        selectConvMode(5, tunedConfig.conv5x5NhwcCoopmat1TunerValueValid,
          tunedConfig.conv5x5NhwcCoopmat2TunerValueValid, tunedConfig.nhwcConv5x5UseCoopmat2);
        selectWinogradTier();
        tunedConfig.normalizeGemmVariantSelectionsForDevice(*info, ctx.fp16Storage, ctx.fp16Compute);
        return;
      }
      selectWinogradTier();
      tunedConfig.normalizeGemmVariantSelectionsForDevice(*info, ctx.fp16Storage, ctx.fp16Compute);
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

  bool retuneNhwcWinogradTransformsForCurrentLayout(
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
    const std::string retuneReason = tunedConfig.nhwcWinogradTransformTunedLayout.empty()
      ? "no previous NHWC Winograd transform layout tune recorded"
      : (nhwcWinogradTransformsTunedForCurrentLayout(tunedConfig, ctx.fp16Storage)
           ? "retune requested for selected Winograd GEMM layout"
           : "selected Winograd GEMM layout changed");
    auto tuneKernel = [&](auto& kernel) {
      out << endl << "VulkanTuner: re-tuning " << kernel.name()
          << " for selected Winograd GEMM layout (reason: " << retuneReason << ")..." << endl;
      std::vector<TunedCandidate> top = tuneOne(kernel, tunedConfig, ctx, iters, out, verboseTuner);
      if(top.empty())
        return false;
      top[0].applyTo(tunedConfig);
      return true;
    };
    VulkanKernels::WinogradTransformNhwc transform(batchSize, wgChannels, nnXLen, nnYLen);
    VulkanKernels::WinogradUntransformNhwc untransform(batchSize, wgChannels, nnXLen, nnYLen);
    const bool tunedTransform = tuneKernel(transform);
    const bool tunedUntransform = tuneKernel(untransform);
    if(tunedTransform && tunedUntransform)
      markNhwcWinogradTransformsTunedForCurrentLayout(tunedConfig, ctx.fp16Storage);
    return tunedTransform && tunedUntransform;
  }

  namespace {

    struct AttentionTuneProblem {
      int headDim = 0;
      int vHeadDim = 0;
      int numHeads = 0;
      int numKVHeads = 0;
      bool useRope = false;
      bool learnableRope = false;
    };

    AttentionTuneProblem findAttentionTuneProblem(const ModelDesc* modelDesc) {
      AttentionTuneProblem problem;
      std::function<void(const vector<pair<int, unique_ptr_void>>&)> scan =
        [&](const vector<pair<int, unique_ptr_void>>& blocks) {
          for(const auto& kv: blocks) {
            if(kv.first == TRANSFORMER_ATTENTION_BLOCK_KIND && problem.headDim == 0) {
              const auto* attn = static_cast<const TransformerAttentionDesc*>(kv.second.get());
              problem.headDim = attn->qHeadDim;
              problem.vHeadDim = attn->vHeadDim;
              problem.numHeads = attn->numHeads;
              problem.numKVHeads = attn->numKVHeads;
              problem.useRope = attn->useRope;
              problem.learnableRope = attn->learnableRope;
            } else if(kv.first == NESTED_BOTTLENECK_BLOCK_KIND) {
              const auto* nbt = static_cast<const NestedBottleneckResidualBlockDesc*>(kv.second.get());
              scan(nbt->blocks);
            }
          }
        };
      scan(modelDesc->trunk.blocks);
      return problem;
    }

    struct AttentionCoopmatCandidate {
      int32_t blockSize, blockQ, blockKV, tm, tn, tk, warp;
      int32_t directKV;
      int32_t kvChunkCount;
    };

    string attentionCandidateString(const AttentionCoopmatCandidate& c) {
      return Global::strprintf(
        "[bs=%d bq=%d bkv=%d tm=%d tn=%d tk=%d warp=%d kv=%s chunks=%d]",
        c.blockSize, c.blockQ, c.blockKV, c.tm, c.tn, c.tk, c.warp,
        c.directKV != 0 ? "direct" : "staged", c.kvChunkCount);
    }

    int32_t gcdPositive(int32_t a, int32_t b) {
      while(b != 0) {
        int32_t t = a % b;
        a = b;
        b = t;
      }
      return a;
    }

    bool tuneAttentionCoopmat1Nhwc(
      TuningContext& ctx,
      const ModelDesc* modelDesc,
      int batchSize,
      int nnXLen,
      int nnYLen,
      int iters,
      std::ostream& out,
      bool verboseTuner,
      VulkanTuneParams& tunedConfig,
      bool maintenance1 = false) {
      tunedConfig.attnNhwcCoopmat1TunerValueValid = 1;
      tunedConfig.attnNhwcVariantPolicyVersion = 4;
      tunedConfig.attnNhwcUseCoopmat1 = 0;
      tunedConfig.attnNhwcCoopmat1DirectKV = 0;
      tunedConfig.attnNhwcCoopmat1KVChunkCount = 1;
      tunedConfig.attnNhwcCoopmat1SplitKTunerValueValid = 0;
      tunedConfig.attnNhwcCoopmat1SplitKCutoffBatch = 0;
      tunedConfig.attnNhwcUseDot2 = 0;
      if(ctx.dev == nullptr || !ctx.fp16Storage || !ctx.fp16Compute || !ctx.dev->info.supportsCoopmat1F16)
        return false;
      AttentionTuneProblem p = findAttentionTuneProblem(modelDesc);
      if(p.headDim <= 0)
        return false;
      const int validTokens = nnXLen * nnYLen;
      const int seqLen = roundUpToMultipleInt(validTokens, VulkanKernels::VULKAN_SPATIAL_ALIGN);
      VulkanKernels::AttentionTiled scalar(
        batchSize, seqLen, p.headDim, p.vHeadDim, p.numHeads, p.numKVHeads,
        p.useRope, p.learnableRope, true, validTokens);
      auto makeCoopBench = [&](int benchBatchSize) -> std::unique_ptr<VulkanKernels::AttentionCoopmat1Bench> {
        if(maintenance1)
          return std::make_unique<VulkanKernels::AttentionCoopmat1Maintenance1Bench>(
            benchBatchSize, seqLen, p.headDim, p.vHeadDim, p.numHeads, p.numKVHeads,
            p.useRope, p.learnableRope, validTokens);
        return std::make_unique<VulkanKernels::AttentionCoopmat1Bench>(
          benchBatchSize, seqLen, p.headDim, p.vHeadDim, p.numHeads, p.numKVHeads,
          p.useRope, p.learnableRope, validTokens);
      };
      std::unique_ptr<VulkanKernels::AttentionCoopmat1Bench> coop = makeCoopBench(batchSize);

      KernelBench scalarRef = scalar.bench(ctx, tunedConfig, iters);
      if(!scalarRef.ok || scalarRef.kernelsPerSecond <= 0.0) {
        out << "VulkanTuner: attention variant reference bench failed; keeping tiled attention." << endl;
        return false;
      }
      std::vector<AttentionCoopmatCandidate> candidates;
      const int32_t warp = (int32_t)ctx.dev->info.subgroupSize;
      for(const CoopmatShape& shape: ctx.dev->info.coopmatShapes) {
        int32_t tm = (int32_t)shape.m, tn = (int32_t)shape.n, tk = (int32_t)shape.k;
        if(tm <= 0 || tn <= 0 || tk <= 0 || warp <= 0)
          continue;
        // maintenance1 converts each TN-wide probability fragment straight to
        // the TK-wide P×V A operand, so unlike the portable spill/reload path
        // it deliberately supports only square N/K fragment widths.
        if(maintenance1 && tn != tk)
          continue;
        int32_t kvMultiple = tn / gcdPositive(tn, tk) * tk;
        for(int targetKV: {16, 32, 64, 128}) {
          int32_t bkv = ((targetKV + kvMultiple - 1) / kvMultiple) * kvMultiple;
          if(bkv > 128)
            continue;
          for(int warpMul: {2, 4, 8}) {
            int32_t blockSize = warp * warpMul;
            int32_t bq = blockSize;
            for(int32_t directKV: {0, 1}) {
              if(directKV != 0 && seqLen % bkv != 0)
                continue;
              AttentionCoopmatCandidate c{blockSize, bq, bkv, tm, tn, tk, warp, directKV, 1};
              VulkanTuneParams cfg = tunedConfig;
              cfg.attnNhwcCoopmat1BlockSize = c.blockSize;
              cfg.attnNhwcCoopmat1BlockQ = c.blockQ;
              cfg.attnNhwcCoopmat1BlockKV = c.blockKV;
              cfg.attnNhwcCoopmat1TM = c.tm;
              cfg.attnNhwcCoopmat1TN = c.tn;
              cfg.attnNhwcCoopmat1TK = c.tk;
              cfg.attnNhwcCoopmat1Warp = c.warp;
              cfg.attnNhwcCoopmat1DirectKV = c.directKV;
              if(!coop->validate(cfg, ctx.dev->info.properties.limits))
                continue;
              bool duplicate = false;
              for(const auto& old: candidates)
                duplicate = duplicate ||
                  (old.blockSize == c.blockSize && old.blockQ == c.blockQ && old.blockKV == c.blockKV &&
                   old.tm == c.tm && old.tn == c.tn && old.tk == c.tk && old.warp == c.warp &&
                   old.directKV == c.directKV);
              if(!duplicate)
                candidates.push_back(c);
            }
          }
        }
      }

      struct Scored {
        AttentionCoopmatCandidate candidate;
        double weightedTime;  // lower = better
        double rmse;
      };
      std::vector<Scored> scored;
      const double errorTol = 0.02;
      out << "VulkanTuner: attention coopmat1 candidates=" << candidates.size() << endl;
      for(const AttentionCoopmatCandidate& c: candidates) {
        VulkanTuneParams cfg = tunedConfig;
        cfg.attnNhwcCoopmat1BlockSize = c.blockSize;
        cfg.attnNhwcCoopmat1BlockQ = c.blockQ;
        cfg.attnNhwcCoopmat1BlockKV = c.blockKV;
        cfg.attnNhwcCoopmat1TM = c.tm;
        cfg.attnNhwcCoopmat1TN = c.tn;
        cfg.attnNhwcCoopmat1TK = c.tk;
        cfg.attnNhwcCoopmat1Warp = c.warp;
        cfg.attnNhwcCoopmat1DirectKV = c.directKV;
        cfg.attnNhwcCoopmat1KVChunkCount = c.kvChunkCount;
        KernelBench b = coop->bench(ctx, cfg, iters);
        double rmse = b.ok ? normalizedRmse(scalarRef.output, b.output) : std::numeric_limits<double>::infinity();
        bool accepted = b.ok && b.kernelsPerSecond > 0.0 && std::isfinite(rmse) && rmse <= errorTol;
        double weightedTime = std::numeric_limits<double>::infinity();
        if(accepted) {
          weightedTime = 1.0 / b.kernelsPerSecond;
        }
        if(verboseTuner || !accepted) {
          out << "VulkanTuner:   attnCoopmat1Nhwc candidate tile=" << attentionCandidateString(c);
          if(b.ok && b.kernelsPerSecond > 0.0)
            out << " kps=" << b.kernelsPerSecond << " sec_per_kernel=" << 1.0 / b.kernelsPerSecond;
          if(std::isfinite(weightedTime))
            out << " weighted_us=" << (int)(weightedTime * 1.0e6);
          out << " rmse=" << (std::isfinite(rmse) ? Global::doubleToString(rmse) : string("n/a"))
              << " result=" << (accepted ? "accepted" : "rejected") << endl;
        }
        if(accepted)
          scored.push_back({c, weightedTime, rmse});
      }
      std::sort(scored.begin(), scored.end(), [](const Scored& a, const Scored& b) { return a.weightedTime < b.weightedTime; });
      if(scored.size() > 3)
        scored.resize(3);

      std::vector<double> scalarSamples;
      for(int repeat = 0; repeat < TUNER_CONFIRM_REPEATS; repeat++) {
        KernelBench b = scalar.bench(ctx, tunedConfig, iters);
        if(b.ok && b.kernelsPerSecond > 0.0)
          scalarSamples.push_back(b.kernelsPerSecond);
      }
      double scalarKps = medianValue(scalarSamples);
      double bestWeightedTime = std::numeric_limits<double>::infinity();
      AttentionCoopmatCandidate best{};
      double bestRmse = 0.0;
      for(const Scored& candidate: scored) {
        std::vector<double> wSamples;
        size_t rmseOk = 0;
        VulkanTuneParams cfg = tunedConfig;
        cfg.attnNhwcCoopmat1BlockSize = candidate.candidate.blockSize;
        cfg.attnNhwcCoopmat1BlockQ = candidate.candidate.blockQ;
        cfg.attnNhwcCoopmat1BlockKV = candidate.candidate.blockKV;
        cfg.attnNhwcCoopmat1TM = candidate.candidate.tm;
        cfg.attnNhwcCoopmat1TN = candidate.candidate.tn;
        cfg.attnNhwcCoopmat1TK = candidate.candidate.tk;
        cfg.attnNhwcCoopmat1Warp = candidate.candidate.warp;
        cfg.attnNhwcCoopmat1DirectKV = candidate.candidate.directKV;
        cfg.attnNhwcCoopmat1KVChunkCount = candidate.candidate.kvChunkCount;
        for(int repeat = 0; repeat < TUNER_CONFIRM_REPEATS; repeat++) {
          KernelBench b = coop->bench(ctx, cfg, iters);
          if(!b.ok || b.kernelsPerSecond <= 0.0)
            continue;
          double rmse = normalizedRmse(scalarRef.output, b.output);
          if(!(std::isfinite(rmse) && rmse <= errorTol))
            continue;
          double wt = 1.0 / b.kernelsPerSecond;
          wSamples.push_back(wt);
          rmseOk++;
        }
        double wt = medianValue(wSamples);
        if(verboseTuner || rmseOk != TUNER_CONFIRM_REPEATS)
          out << "VulkanTuner:   attnCoopmat1Nhwc confirm tile=" << attentionCandidateString(candidate.candidate)
              << " us=" << (wt > 0.0 ? (int)(wt * 1.0e6) : 0)
              << " rmse_ok=" << rmseOk << "/" << TUNER_CONFIRM_REPEATS << endl;
        if(wt > 0.0 && wt < bestWeightedTime) {
          bestWeightedTime = wt;
          best = candidate.candidate;
          bestRmse = candidate.rmse;
        }
      }

      double bestCoopKps = bestWeightedTime < std::numeric_limits<double>::infinity() ? 1.0 / bestWeightedTime : 0.0;
      tunedConfig.attnNhwcTiledTimeUs = scalarKps > 0.0 ? (int32_t)std::llround(1.0e6 / scalarKps) : 0;
      tunedConfig.attnNhwcCoopmat1TimeUs = bestCoopKps > 0.0 ? (int32_t)std::llround(bestWeightedTime * 1.0e6) : 0;
      if(bestCoopKps > 0.0) {
        tunedConfig.attnNhwcUseCoopmat1 = 1;
        tunedConfig.attnNhwcCoopmat1BlockSize = best.blockSize;
        tunedConfig.attnNhwcCoopmat1BlockQ = best.blockQ;
        tunedConfig.attnNhwcCoopmat1BlockKV = best.blockKV;
        tunedConfig.attnNhwcCoopmat1TM = best.tm;
        tunedConfig.attnNhwcCoopmat1TN = best.tn;
        tunedConfig.attnNhwcCoopmat1TK = best.tk;
        tunedConfig.attnNhwcCoopmat1Warp = best.warp;
        tunedConfig.attnNhwcCoopmat1DirectKV = best.directKV;
        tunedConfig.attnNhwcCoopmat1KVChunkCount = 1;
        out << "VulkanTuner:   attnCoopmat1Nhwc winner tile=" << attentionCandidateString(best)
            << " us=" << (int)(bestWeightedTime * 1.0e6) << endl;
        // A valid split-K entry may deliberately retain chunkCount=1 when no
        // split candidate is viable or faster for this device.
        tunedConfig.attnNhwcCoopmat1SplitKTunerValueValid = 1;

        // Split-K is a separate batch-1 role. Sweep both its tile and chunk
        // count independently so it cannot compromise the regular path.
        VulkanKernels::AttentionTiled scalarB1(
          1, seqLen, p.headDim, p.vHeadDim, p.numHeads, p.numKVHeads,
          p.useRope, p.learnableRope, true, validTokens);
        std::unique_ptr<VulkanKernels::AttentionCoopmat1Bench> coopB1 = makeCoopBench(1);
        KernelBench scalarRefB1 = scalarB1.bench(ctx, tunedConfig, iters);
        struct SplitScored { AttentionCoopmatCandidate candidate; double time; double rmse; };
        std::vector<SplitScored> splitScored;
        if(verboseTuner)
          out << "VulkanTuner: attention coopmat1 split-k candidates at batch=1" << endl;
        if(scalarRefB1.ok && scalarRefB1.kernelsPerSecond > 0.0) {
          for(const AttentionCoopmatCandidate& base: candidates) {
            const int kvTiles = (seqLen + base.blockKV - 1) / base.blockKV;
            for(int chunks: {2, 3, 4}) {
              if(chunks > kvTiles)
                continue;
              AttentionCoopmatCandidate split = base;
              split.kvChunkCount = chunks;
              VulkanTuneParams splitCfg = tunedConfig;
              splitCfg.attnNhwcCoopmat1BlockSize = split.blockSize;
              splitCfg.attnNhwcCoopmat1BlockQ = split.blockQ;
              splitCfg.attnNhwcCoopmat1BlockKV = split.blockKV;
              splitCfg.attnNhwcCoopmat1TM = split.tm;
              splitCfg.attnNhwcCoopmat1TN = split.tn;
              splitCfg.attnNhwcCoopmat1TK = split.tk;
              splitCfg.attnNhwcCoopmat1Warp = split.warp;
              splitCfg.attnNhwcCoopmat1DirectKV = split.directKV;
              splitCfg.attnNhwcCoopmat1KVChunkCount = split.kvChunkCount;
              if(!coopB1->validate(splitCfg, ctx.dev->info.properties.limits))
                continue;
              KernelBench result = coopB1->bench(ctx, splitCfg, iters);
              const double rmse = result.ok ? normalizedRmse(scalarRefB1.output, result.output)
                                            : std::numeric_limits<double>::infinity();
              const bool accepted = result.ok && result.kernelsPerSecond > 0.0 &&
                std::isfinite(rmse) && rmse <= errorTol;
              if(verboseTuner || !accepted) {
                out << "VulkanTuner:   attnCoopmat1Nhwc split-k candidate tile="
                    << attentionCandidateString(split);
                if(result.ok && result.kernelsPerSecond > 0.0)
                  out << " kps=" << result.kernelsPerSecond
                      << " sec_per_kernel=" << 1.0 / result.kernelsPerSecond;
                out << " rmse=" << (std::isfinite(rmse) ? Global::doubleToString(rmse) : string("n/a"))
                    << " result=" << (accepted ? "accepted" : "rejected") << endl;
              }
              if(accepted)
                splitScored.push_back({split, 1.0 / result.kernelsPerSecond, rmse});
            }
          }
        }
        std::sort(splitScored.begin(), splitScored.end(), [](const SplitScored& a, const SplitScored& b) {
          return a.time < b.time;
        });
        if(splitScored.size() > 3)
          splitScored.resize(3);
        double bestSplitTime = std::numeric_limits<double>::infinity();
        AttentionCoopmatCandidate bestSplit{};
        for(const SplitScored& candidate: splitScored) {
          std::vector<double> samples;
          size_t rmseOk = 0;
          VulkanTuneParams splitCfg = tunedConfig;
          splitCfg.attnNhwcCoopmat1BlockSize = candidate.candidate.blockSize;
          splitCfg.attnNhwcCoopmat1BlockQ = candidate.candidate.blockQ;
          splitCfg.attnNhwcCoopmat1BlockKV = candidate.candidate.blockKV;
          splitCfg.attnNhwcCoopmat1TM = candidate.candidate.tm;
          splitCfg.attnNhwcCoopmat1TN = candidate.candidate.tn;
          splitCfg.attnNhwcCoopmat1TK = candidate.candidate.tk;
          splitCfg.attnNhwcCoopmat1Warp = candidate.candidate.warp;
          splitCfg.attnNhwcCoopmat1DirectKV = candidate.candidate.directKV;
          splitCfg.attnNhwcCoopmat1KVChunkCount = candidate.candidate.kvChunkCount;
          for(int repeat = 0; repeat < TUNER_CONFIRM_REPEATS; repeat++) {
            KernelBench result = coopB1->bench(ctx, splitCfg, iters);
            const double rmse = result.ok ? normalizedRmse(scalarRefB1.output, result.output)
                                          : std::numeric_limits<double>::infinity();
            if(result.ok && result.kernelsPerSecond > 0.0 && std::isfinite(rmse) && rmse <= errorTol) {
              samples.push_back(1.0 / result.kernelsPerSecond);
              rmseOk++;
              bestRmse = std::max(bestRmse, rmse);
            }
          }
          const double time = medianValue(samples);
          if(verboseTuner || rmseOk != TUNER_CONFIRM_REPEATS)
            out << "VulkanTuner:   attnCoopmat1Nhwc split-k confirm tile="
                << attentionCandidateString(candidate.candidate)
                << " us=" << (time > 0.0 ? (int)(time * 1.0e6) : 0)
                << " rmse_ok=" << rmseOk << "/" << TUNER_CONFIRM_REPEATS << endl;
          if(time > 0.0 && time < bestSplitTime) {
            bestSplitTime = time;
            bestSplit = candidate.candidate;
          }
        }
        if(bestSplitTime < std::numeric_limits<double>::infinity()) {
          tunedConfig.attnNhwcCoopmat1SplitKTunerValueValid = 1;
          tunedConfig.attnNhwcCoopmat1SplitKBlockSize = bestSplit.blockSize;
          tunedConfig.attnNhwcCoopmat1SplitKBlockQ = bestSplit.blockQ;
          tunedConfig.attnNhwcCoopmat1SplitKBlockKV = bestSplit.blockKV;
          tunedConfig.attnNhwcCoopmat1SplitKTM = bestSplit.tm;
          tunedConfig.attnNhwcCoopmat1SplitKTN = bestSplit.tn;
          tunedConfig.attnNhwcCoopmat1SplitKTK = bestSplit.tk;
          tunedConfig.attnNhwcCoopmat1SplitKWarp = bestSplit.warp;
          tunedConfig.attnNhwcCoopmat1SplitKDirectKV = bestSplit.directKV;
          tunedConfig.attnNhwcCoopmat1SplitKKVChunkCount = bestSplit.kvChunkCount;
          out << "VulkanTuner:   attnCoopmat1Nhwc split-k winner tile="
              << attentionCandidateString(bestSplit)
              << " us=" << (int)(bestSplitTime * 1.0e6) << endl;

          // The two roles are tuned independently above: regular attention at
          // the requested batch size, and split-K at batch one. Now measure
          // those fixed winners at every intermediate batch to choose the
          // largest contiguous prefix where split-K wins. A cutoff rather than
          // a per-batch table keeps the saved configuration compact and makes
          // reuse at smaller batch sizes safe.
          VulkanTuneParams regularCfg = tunedConfig;
          VulkanTuneParams splitCfg = tunedConfig;
          splitCfg.attnNhwcCoopmat1BlockSize = bestSplit.blockSize;
          splitCfg.attnNhwcCoopmat1BlockQ = bestSplit.blockQ;
          splitCfg.attnNhwcCoopmat1BlockKV = bestSplit.blockKV;
          splitCfg.attnNhwcCoopmat1TM = bestSplit.tm;
          splitCfg.attnNhwcCoopmat1TN = bestSplit.tn;
          splitCfg.attnNhwcCoopmat1TK = bestSplit.tk;
          splitCfg.attnNhwcCoopmat1Warp = bestSplit.warp;
          splitCfg.attnNhwcCoopmat1DirectKV = bestSplit.directKV;
          splitCfg.attnNhwcCoopmat1KVChunkCount = bestSplit.kvChunkCount;
          bool splitKWonEverySmallerBatch = true;
          for(int crossoverBatch = 1; crossoverBatch <= batchSize; crossoverBatch++) {
            std::unique_ptr<VulkanKernels::AttentionCoopmat1Bench> regularBench = makeCoopBench(crossoverBatch);
            std::unique_ptr<VulkanKernels::AttentionCoopmat1Bench> splitBench = makeCoopBench(crossoverBatch);
            std::vector<double> regularSamples;
            std::vector<double> splitSamples;
            for(int repeat = 0; repeat < TUNER_CONFIRM_REPEATS; repeat++) {
              KernelBench regularResult = regularBench->bench(ctx, regularCfg, iters);
              if(regularResult.ok && regularResult.kernelsPerSecond > 0.0)
                regularSamples.push_back(1.0 / regularResult.kernelsPerSecond);
              KernelBench splitResult = splitBench->bench(ctx, splitCfg, iters);
              if(splitResult.ok && splitResult.kernelsPerSecond > 0.0)
                splitSamples.push_back(1.0 / splitResult.kernelsPerSecond);
            }
            const double regularTime = medianValue(regularSamples);
            const double splitTime = medianValue(splitSamples);
            const bool splitKWins = splitTime > 0.0 && regularTime > 0.0 && splitTime < regularTime;
            if(verboseTuner || regularTime <= 0.0 || splitTime <= 0.0)
              out << "VulkanTuner:   attnCoopmat1Nhwc crossover batch=" << crossoverBatch
                  << " regular_us=" << (regularTime > 0.0 ? (int)(regularTime * 1.0e6) : 0)
                  << " split_k_us=" << (splitTime > 0.0 ? (int)(splitTime * 1.0e6) : 0)
                  << " selected=" << (splitKWonEverySmallerBatch && splitKWins ? "split-k" : "regular")
                  << endl;
            if(splitKWonEverySmallerBatch && splitKWins)
              tunedConfig.attnNhwcCoopmat1SplitKCutoffBatch = crossoverBatch;
            else
              splitKWonEverySmallerBatch = false;
          }
          out << "VulkanTuner:   attnCoopmat1Nhwc split-k cutoff batch="
              << tunedConfig.attnNhwcCoopmat1SplitKCutoffBatch << endl;
        }
      }
      out << "VulkanTuner: attention NHWC selected="
          << (tunedConfig.attnNhwcUseCoopmat1 != 0 ? "coopmat1" : "tiled")
          << " tiled_us=" << tunedConfig.attnNhwcTiledTimeUs
          << " coopmat1_us=" << tunedConfig.attnNhwcCoopmat1TimeUs
          << " max_rmse=" << bestRmse << endl;
      return tunedConfig.attnNhwcUseCoopmat1 != 0;
    }

    // Reuse the candidate sweep while running the maintenance1 SPIR-V. The
    // temporary base fields are never persisted: copy its independently timed
    // winners into the maintenance1 namespace and restore portable coopmat1.
    bool tuneAttentionCoopmatMaintenance1Nhwc(
      TuningContext& ctx, const ModelDesc* modelDesc, int batchSize, int nnXLen, int nnYLen,
      int iters, std::ostream& out, bool verboseTuner, VulkanTuneParams& cfg) {
      if(ctx.dev == nullptr || !ctx.dev->info.supportsCoopmatMaintenance1)
        return false;
      const VulkanTuneParams saved = cfg;
      tuneAttentionCoopmat1Nhwc(ctx, modelDesc, batchSize, nnXLen, nnYLen, iters, out, verboseTuner, cfg, true);
      const VulkanTuneParams winner = cfg;
      cfg = saved;
      cfg.attnNhwcCoopmatMaintenance1TunerValueValid = 1;
      cfg.attnNhwcCoopmatMaintenance1TimeUs = winner.attnNhwcCoopmat1TimeUs;
      cfg.attnNhwcCoopmatMaintenance1BlockSize = winner.attnNhwcCoopmat1BlockSize;
      cfg.attnNhwcCoopmatMaintenance1BlockQ = winner.attnNhwcCoopmat1BlockQ;
      cfg.attnNhwcCoopmatMaintenance1BlockKV = winner.attnNhwcCoopmat1BlockKV;
      cfg.attnNhwcCoopmatMaintenance1TM = winner.attnNhwcCoopmat1TM;
      cfg.attnNhwcCoopmatMaintenance1TN = winner.attnNhwcCoopmat1TN;
      cfg.attnNhwcCoopmatMaintenance1TK = winner.attnNhwcCoopmat1TK;
      cfg.attnNhwcCoopmatMaintenance1Warp = winner.attnNhwcCoopmat1Warp;
      cfg.attnNhwcCoopmatMaintenance1DirectKV = winner.attnNhwcCoopmat1DirectKV;
      cfg.attnNhwcCoopmatMaintenance1SplitKTunerValueValid = winner.attnNhwcCoopmat1SplitKTunerValueValid;
      cfg.attnNhwcCoopmatMaintenance1SplitKBlockSize = winner.attnNhwcCoopmat1SplitKBlockSize;
      cfg.attnNhwcCoopmatMaintenance1SplitKBlockQ = winner.attnNhwcCoopmat1SplitKBlockQ;
      cfg.attnNhwcCoopmatMaintenance1SplitKBlockKV = winner.attnNhwcCoopmat1SplitKBlockKV;
      cfg.attnNhwcCoopmatMaintenance1SplitKTM = winner.attnNhwcCoopmat1SplitKTM;
      cfg.attnNhwcCoopmatMaintenance1SplitKTN = winner.attnNhwcCoopmat1SplitKTN;
      cfg.attnNhwcCoopmatMaintenance1SplitKTK = winner.attnNhwcCoopmat1SplitKTK;
      cfg.attnNhwcCoopmatMaintenance1SplitKWarp = winner.attnNhwcCoopmat1SplitKWarp;
      cfg.attnNhwcCoopmatMaintenance1SplitKDirectKV = winner.attnNhwcCoopmat1SplitKDirectKV;
      cfg.attnNhwcCoopmatMaintenance1SplitKKVChunkCount = winner.attnNhwcCoopmat1SplitKKVChunkCount;
      cfg.attnNhwcCoopmatMaintenance1SplitKCutoffBatch = winner.attnNhwcCoopmat1SplitKCutoffBatch;
      out << "VulkanTuner:   attnCoopmatMaintenance1Nhwc winner us="
          << cfg.attnNhwcCoopmatMaintenance1TimeUs << " cutoff="
          << cfg.attnNhwcCoopmatMaintenance1SplitKCutoffBatch << endl;
      return cfg.attnNhwcCoopmatMaintenance1TimeUs > 0;
    }

    bool tuneAttentionCoopmat2AccF32Nhwc(
      TuningContext& ctx, const ModelDesc* modelDesc, int batchSize, int nnXLen, int nnYLen,
      int iters, std::ostream& out, bool verboseTuner, VulkanTuneParams& tunedConfig) {
      tunedConfig.attnNhwcCoopmat2TunerValueValid = 1;
      tunedConfig.attnNhwcVariantPolicyVersion = 4;
      if(ctx.dev == nullptr || !ctx.fp16Storage || !ctx.fp16Compute || !ctx.dev->info.supportsCoopmat2F16)
        return false;
      AttentionTuneProblem p=findAttentionTuneProblem(modelDesc);
      if(p.headDim<=0) return false;
      const int validTokens=nnXLen*nnYLen;
      const int seqLen=roundUpToMultipleInt(validTokens,VulkanKernels::VULKAN_SPATIAL_ALIGN);
      VulkanKernels::AttentionTiled scalar(batchSize,seqLen,p.headDim,p.vHeadDim,p.numHeads,p.numKVHeads,p.useRope,p.learnableRope,true,validTokens);
      VulkanKernels::AttentionCoopmat2AccF32Bench coop(batchSize,seqLen,p.headDim,p.vHeadDim,p.numHeads,p.numKVHeads,p.useRope,p.learnableRope,validTokens);
      KernelBench ref=scalar.bench(ctx,tunedConfig,iters);
      if(!ref.ok || ref.kernelsPerSecond<=0) return false;
      auto matches=[&](int bs,int bq,int bkv) {
        bool qk=false, pv=false;
        for(const Coopmat2FlexShape& s:ctx.dev->info.coopmat2FlexShapes) {
          if((uint32_t)bs!=s.workgroupInvocations ||
             s.mGranularity==0 || s.nGranularity==0 || s.kGranularity==0) continue;
          qk = qk || (bq%(int)s.mGranularity==0 && bkv%(int)s.nGranularity==0 &&
                      p.headDim%(int)s.kGranularity==0);
          pv = pv || (bq%(int)s.mGranularity==0 && p.vHeadDim%(int)s.nGranularity==0 &&
                      bkv%(int)s.kGranularity==0);
        }
        return qk && pv;
      };
      struct C {int bs,bq,bkv; double kps,rmse;};
      std::vector<C> tried, good;
      static const int bqMultipliers[] = {4,8,12,16};
      const int maxBq=std::min(256,seqLen);
      for(const Coopmat2FlexShape& shape:ctx.dev->info.coopmat2FlexShapes) {
        const int bs=(int)shape.workgroupInvocations;
        std::vector<int> bqs={bs};
        for(int mul:bqMultipliers) bqs.push_back((int)shape.mGranularity*mul);
        for(int bq:bqs) for(int bkv:{8,16,32,64}) {
          if(bq<=0 || bq>maxBq || !matches(bs,bq,bkv)) continue;
          bool duplicate=false;
          for(const C& c:tried) if(c.bs==bs && c.bq==bq && c.bkv==bkv) { duplicate=true; break; }
          if(duplicate) continue;
          tried.push_back({bs,bq,bkv,0.0,0.0});
          VulkanTuneParams cfg=tunedConfig; cfg.attnNhwcCoopmat2BlockSize=bs; cfg.attnNhwcCoopmat2BlockQ=bq; cfg.attnNhwcCoopmat2BlockKV=bkv;
          if(!coop.validate(cfg,ctx.dev->info.properties.limits) ||
             VulkanKernels::AttentionCoopmat2AccF32Nhwc::sharedBytes(bq,bkv,p.headDim,p.vHeadDim)+ctx.dev->info.coopmat2ReservedSharedBytes>ctx.dev->info.properties.limits.maxComputeSharedMemorySize) continue;
          KernelBench b=coop.bench(ctx,cfg,iters); double rmse=b.ok?normalizedRmse(ref.output,b.output):std::numeric_limits<double>::infinity();
          const bool accept=b.ok&&b.kernelsPerSecond>0&&std::isfinite(rmse)&&rmse<=0.02;
          if(verboseTuner||accept) out << "VulkanTuner: attnCoopmat2 native tile=[bs="<<bs<<" bq="<<bq<<" bkv="<<bkv<<"] kps="<<b.kernelsPerSecond<<" rmse="<<rmse<<" result="<<(accept?"accepted":"rejected")<<endl;
          if(accept) good.push_back({bs,bq,bkv,b.kernelsPerSecond,rmse});
        }
      }
      std::sort(good.begin(),good.end(),[](const C&a,const C&b){return a.kps>b.kps;});
      tunedConfig.attnNhwcCoopmat2TimeUs=good.empty()?0:(int32_t)std::llround(1e6/good[0].kps);
      if(!good.empty()) { tunedConfig.attnNhwcCoopmat2BlockSize=good[0].bs; tunedConfig.attnNhwcCoopmat2BlockQ=good[0].bq; tunedConfig.attnNhwcCoopmat2BlockKV=good[0].bkv; }
      return !good.empty();
    }

    struct AttentionDot2Candidate {
      int32_t blockSize, blockQ, blockKV;
    };

    string attentionDot2CandidateString(const AttentionDot2Candidate& c) {
      return Global::strprintf("[bs=%d bq=%d bkv=%d]", c.blockSize, c.blockQ, c.blockKV);
    }

    bool tuneAttentionDot2AccF32Nhwc(
      TuningContext& ctx,
      const ModelDesc* modelDesc,
      int batchSize,
      int nnXLen,
      int nnYLen,
      int iters,
      std::ostream& out,
      bool verboseTuner,
      VulkanTuneParams& tunedConfig) {
      tunedConfig.attnNhwcDot2TunerValueValid = 1;
      tunedConfig.attnNhwcVariantPolicyVersion = 4;
      tunedConfig.attnNhwcUseCoopmat1 = 0;
      tunedConfig.attnNhwcUseDot2 = 0;
      if(ctx.dev == nullptr || !ctx.fp16Storage || !ctx.fp16Compute || !ctx.dev->info.supportsDot2F16)
        return false;
      AttentionTuneProblem p = findAttentionTuneProblem(modelDesc);
      if(p.headDim <= 0)
        return false;
      const int validTokens = nnXLen * nnYLen;
      const int seqLen = roundUpToMultipleInt(validTokens, VulkanKernels::VULKAN_SPATIAL_ALIGN);
      VulkanKernels::AttentionTiled scalar(
        batchSize, seqLen, p.headDim, p.vHeadDim, p.numHeads, p.numKVHeads,
        p.useRope, p.learnableRope, true, validTokens);
      VulkanKernels::AttentionDot2AccF32Bench dot2(
        batchSize, seqLen, p.headDim, p.vHeadDim, p.numHeads, p.numKVHeads,
        p.useRope, p.learnableRope, validTokens);

      KernelBench scalarRef = scalar.bench(ctx, tunedConfig, iters);
      if(!scalarRef.ok || scalarRef.kernelsPerSecond <= 0.0) {
        out << "VulkanTuner: attention DOT2 reference bench failed; keeping tiled attention." << endl;
        return false;
      }

      std::vector<AttentionDot2Candidate> candidates;
      for(int32_t blockSize: {32, 64, 128, 256}) {
        for(int32_t blockKV: {16, 32, 64}) {
          AttentionDot2Candidate c{blockSize, blockSize, blockKV};
          VulkanTuneParams cfg = tunedConfig;
          cfg.attnNhwcDot2BlockSize = c.blockSize;
          cfg.attnNhwcDot2BlockQ = c.blockQ;
          cfg.attnNhwcDot2BlockKV = c.blockKV;
          if(dot2.validate(cfg, ctx.dev->info.properties.limits))
            candidates.push_back(c);
        }
      }

      struct Scored {
        AttentionDot2Candidate candidate;
        double kps;
        double rmse;
      };
      std::vector<Scored> scored;
      const double errorTol = 0.02;
      out << "VulkanTuner: attention DOT2 AccF32 candidates=" << candidates.size() << endl;
      for(const AttentionDot2Candidate& c: candidates) {
        VulkanTuneParams cfg = tunedConfig;
        cfg.attnNhwcDot2BlockSize = c.blockSize;
        cfg.attnNhwcDot2BlockQ = c.blockQ;
        cfg.attnNhwcDot2BlockKV = c.blockKV;
        KernelBench b = dot2.bench(ctx, cfg, iters);
        double rmse = b.ok ? normalizedRmse(scalarRef.output, b.output) : std::numeric_limits<double>::infinity();
        bool accepted = b.ok && b.kernelsPerSecond > 0.0 && std::isfinite(rmse) && rmse <= errorTol;
        if(verboseTuner || accepted || !b.ok) {
          out << "VulkanTuner:   attnDot2AccF32Nhwc candidate tile=" << attentionDot2CandidateString(c);
          if(b.ok && b.kernelsPerSecond > 0.0)
            out << " kps=" << b.kernelsPerSecond << " sec_per_kernel=" << 1.0 / b.kernelsPerSecond;
          out << " rmse=" << (std::isfinite(rmse) ? Global::doubleToString(rmse) : string("n/a"))
              << " result=" << (accepted ? "accepted" : "rejected") << endl;
        }
        if(accepted)
          scored.push_back({c, b.kernelsPerSecond, rmse});
      }
      std::sort(scored.begin(), scored.end(), [](const Scored& a, const Scored& b) { return a.kps > b.kps; });
      if(scored.size() > 3)
        scored.resize(3);

      std::vector<double> scalarSamples;
      for(int repeat = 0; repeat < TUNER_CONFIRM_REPEATS; repeat++) {
        KernelBench b = scalar.bench(ctx, tunedConfig, iters);
        if(b.ok && b.kernelsPerSecond > 0.0)
          scalarSamples.push_back(b.kernelsPerSecond);
      }
      double scalarKps = medianValue(scalarSamples);
      double bestDot2Kps = 0.0;
      AttentionDot2Candidate best{};
      double bestRmse = 0.0;
      for(const Scored& candidate: scored) {
        std::vector<double> samples;
        size_t rmseOk = 0;
        VulkanTuneParams cfg = tunedConfig;
        cfg.attnNhwcDot2BlockSize = candidate.candidate.blockSize;
        cfg.attnNhwcDot2BlockQ = candidate.candidate.blockQ;
        cfg.attnNhwcDot2BlockKV = candidate.candidate.blockKV;
        for(int repeat = 0; repeat < TUNER_CONFIRM_REPEATS; repeat++) {
          KernelBench b = dot2.bench(ctx, cfg, iters);
          if(!b.ok || b.kernelsPerSecond <= 0.0)
            continue;
          double rmse = normalizedRmse(scalarRef.output, b.output);
          if(std::isfinite(rmse) && rmse <= errorTol) {
            samples.push_back(b.kernelsPerSecond);
            rmseOk++;
          }
        }
        double kps = medianValue(samples);
        out << "VulkanTuner:   attnDot2AccF32Nhwc confirm tile="
            << attentionDot2CandidateString(candidate.candidate)
            << " kps=" << kps << " rmse_ok=" << rmseOk << "/" << TUNER_CONFIRM_REPEATS << endl;
        if(kps > bestDot2Kps) {
          bestDot2Kps = kps;
          best = candidate.candidate;
          bestRmse = candidate.rmse;
        }
      }

      tunedConfig.attnNhwcTiledTimeUs = scalarKps > 0.0 ? (int32_t)std::llround(1.0e6 / scalarKps) : 0;
      tunedConfig.attnNhwcDot2TimeUs = bestDot2Kps > 0.0 ? (int32_t)std::llround(1.0e6 / bestDot2Kps) : 0;
      if(bestDot2Kps > 0.0) {
        tunedConfig.attnNhwcUseDot2 = 1;
        tunedConfig.attnNhwcDot2BlockSize = best.blockSize;
        tunedConfig.attnNhwcDot2BlockQ = best.blockQ;
        tunedConfig.attnNhwcDot2BlockKV = best.blockKV;
      }
      out << "VulkanTuner: attention NHWC selected="
          << (tunedConfig.attnNhwcUseDot2 != 0 ? "dot2accf32" : "tiled")
          << " tiled_us=" << tunedConfig.attnNhwcTiledTimeUs
          << " dot2_us=" << tunedConfig.attnNhwcDot2TimeUs
          << " max_rmse=" << bestRmse << endl;
      return tunedConfig.attnNhwcUseDot2 != 0;
    }

  }  // namespace

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
    bool full,
    VulkanTuneParams& tunedConfig) {
    tunedConfig = initialConfig;
    if(ctx == nullptr || ctx->dev == nullptr || modelDesc == nullptr)
      return;
    const int iters = std::max(1, benchIters);

    (void)full;
    std::vector<bool> layouts = {true};
    for(bool useNhwc: layouts) {
      ScopedTuningLayout layoutScope(useNhwc);
      const int32_t required = requiredKernelMask(
        modelDesc, ctx->dev->info, ctx->fp16Storage, ctx->fp16Compute, useNhwc, false);
      int32_t missing = required & ~tunedConfig.tunedKernelMask;
      if((required & TUNED_SWIGLU) != 0 && tunedConfig.swiGLUTunerValueValid == 0)
        missing |= TUNED_SWIGLU;
      if(useNhwc && (required & TUNED_NHWC_ATTENTION_VARIANT) != 0) {
        const bool attentionTierMissing =
          (ctx->dev->info.supportsCoopmat1F16 && tunedConfig.attnNhwcCoopmat1TunerValueValid == 0) ||
          (ctx->dev->info.supportsCoopmat1F16 && tunedConfig.attnNhwcCoopmat1SplitKTunerValueValid == 0) ||
          (ctx->dev->info.supportsCoopmat2F16 && tunedConfig.attnNhwcCoopmat2TunerValueValid == 0) ||
          (!ctx->dev->info.supportsCoopmat1F16 && !ctx->dev->info.supportsCoopmat2F16 &&
           ctx->dev->info.supportsDot2F16 && tunedConfig.attnNhwcDot2TunerValueValid == 0);
        if(attentionTierMissing || (ctx->dev->info.supportsCoopmatMaintenance1 &&
             (tunedConfig.attnNhwcCoopmatMaintenance1TunerValueValid == 0 ||
              tunedConfig.attnNhwcCoopmatMaintenance1SplitKTunerValueValid == 0)) ||
           tunedConfig.attnNhwcVariantPolicyVersion != 4 ||
           !gemmVariantFilterTunedForCurrentHardware(tunedConfig, ctx->dev->info))
          missing |= TUNED_NHWC_ATTENTION_VARIANT;
      }
      if(useNhwc && (required & TUNED_NHWC_GEMM) != 0 &&
         (tunedConfig.tunedKernelMask & TUNED_NHWC_GEMM_RUNTIME_MODEL) == 0)
        missing |= TUNED_NHWC_GEMM_RUNTIME_MODEL;
      const bool fp16Ready = ctx->fp16Storage && ctx->fp16Compute;
      const bool haveCoopFamily = fp16Ready &&
        (ctx->dev->info.supportsCoopmat1F16 || ctx->dev->info.supportsCoopmat1F16AccF16 ||
         ctx->dev->info.supportsCoopmat2F16 || ctx->dev->info.supportsCoopmat2F16AccF16);
      if(useNhwc) {
        const bool haveNhwcGemmCoop = fp16Ready &&
          (ctx->dev->info.supportsCoopmat1F16 || ctx->dev->info.supportsCoopmat1F16AccF16 ||
           ctx->dev->info.supportsCoopmat2F16 || ctx->dev->info.supportsCoopmat2F16AccF16);
        if((required & TUNED_NHWC_GEMM) != 0) {
          if(haveNhwcGemmCoop) {
            if((ctx->dev->info.supportsCoopmat1F16 &&
                (tunedConfig.gemmStridedNhwcCoopmat1TunerValueValid == 0 ||
                 tunedConfig.nhwcGemmCoopmat1F32TimeUs <= 0)) ||
               (ctx->dev->info.supportsCoopmat1F16AccF16 &&
                (tunedConfig.gemmStridedNhwcCoopmat1AccF16TunerValueValid == 0 ||
                 tunedConfig.nhwcGemmCoopmat1F16TimeUs <= 0)) ||
               (ctx->dev->info.supportsCoopmat2F16 &&
                (tunedConfig.gemmStridedNhwcCoopmat2TunerValueValid == 0 ||
                 tunedConfig.nhwcGemmCoopmat2F32TimeUs <= 0)) ||
               (ctx->dev->info.supportsCoopmat2F16AccF16 &&
                (tunedConfig.gemmStridedNhwcCoopmat2AccF16TunerValueValid == 0 ||
                 tunedConfig.nhwcGemmCoopmat2F16TimeUs <= 0)))
              missing |= TUNED_NHWC_GEMM;
          } else if(
            (ctx->dev->info.supportsDot2F16 &&
             (tunedConfig.gemmStridedNhwcDot2TunerValueValid == 0 || tunedConfig.nhwcGemmDot2F32TimeUs <= 0)) ||
            (ctx->dev->info.supportsDot2F16AccF16 &&
             (tunedConfig.gemmStridedNhwcDot2AccF16TunerValueValid == 0 || tunedConfig.nhwcGemmDot2F16TimeUs <= 0)))
            missing |= TUNED_NHWC_GEMM;
        }
        if((required & TUNED_NHWC_CONV3X3) != 0 &&
           ((ctx->dev->info.supportsCoopmat1F16 && tunedConfig.conv3x3NhwcCoopmat1TunerValueValid == 0) ||
            (ctx->dev->info.supportsCoopmat2F16 && tunedConfig.conv3x3NhwcCoopmat2TunerValueValid == 0) ||
            (ctx->dev->info.supportsCoopmat1F16AccF16 && tunedConfig.conv3x3NhwcCoopmat1AccF16TunerValueValid == 0) ||
            (ctx->dev->info.supportsCoopmat2F16AccF16 && tunedConfig.conv3x3NhwcCoopmat2AccF16TunerValueValid == 0)))
          missing |= TUNED_NHWC_CONV3X3;
        if((required & TUNED_NHWC_CONV5X5) != 0 &&
           ((ctx->dev->info.supportsCoopmat1F16 && tunedConfig.conv5x5NhwcCoopmat1TunerValueValid == 0) ||
            (ctx->dev->info.supportsCoopmat2F16 && tunedConfig.conv5x5NhwcCoopmat2TunerValueValid == 0) ||
            (ctx->dev->info.supportsCoopmat1F16AccF16 && tunedConfig.conv5x5NhwcCoopmat1AccF16TunerValueValid == 0) ||
            (ctx->dev->info.supportsCoopmat2F16AccF16 && tunedConfig.conv5x5NhwcCoopmat2AccF16TunerValueValid == 0)))
          missing |= TUNED_NHWC_CONV5X5;
      }
      if(useNhwc && (required & TUNED_NHWC_WINOGRAD) != 0 &&
         (tunedConfig.conv3x3NhwcWinogradTunerValueValid == 0 || tunedConfig.conv5x5NhwcWinogradTunerValueValid == 0))
        missing |= TUNED_NHWC_WINOGRAD;
      if(useNhwc && (required & TUNED_NHWC_WINOGRAD) != 0) {
        if(haveCoopFamily) {
          if((ctx->dev->info.supportsCoopmat1F16 &&
              (tunedConfig.winogradGemmCoopmat1TunerValueValid == 0 || tunedConfig.winogradGemmCoopmat1F32TimeUs <= 0)) ||
             (ctx->dev->info.supportsCoopmat1F16AccF16 &&
              (tunedConfig.winogradGemmCoopmat1AccF16TunerValueValid == 0 || tunedConfig.winogradGemmCoopmat1F16TimeUs <= 0)) ||
             (ctx->dev->info.supportsCoopmat2F16 &&
              (tunedConfig.winogradGemmCoopmat2TunerValueValid == 0 || tunedConfig.winogradGemmCoopmat2F32TimeUs <= 0)) ||
             (ctx->dev->info.supportsCoopmat2F16AccF16 &&
              (tunedConfig.winogradGemmCoopmat2AccF16TunerValueValid == 0 || tunedConfig.winogradGemmCoopmat2F16TimeUs <= 0)))
            missing |= TUNED_NHWC_WINOGRAD;
        } else if(
          (ctx->dev->info.supportsDot2F16 &&
           (tunedConfig.winogradGemmDot2TunerValueValid == 0 || tunedConfig.winogradGemmDot2F32TimeUs <= 0)) ||
          (ctx->dev->info.supportsDot2F16AccF16 &&
           (tunedConfig.winogradGemmDot2AccF16TunerValueValid == 0 || tunedConfig.winogradGemmDot2F16TimeUs <= 0)))
          missing |= TUNED_NHWC_WINOGRAD;
      }
      const bool gemmSelectionPolicyChanged =
        !gemmVariantFilterTunedForCurrentHardware(tunedConfig, ctx->dev->info);
      if(missing == 0 && !gemmSelectionPolicyChanged)
        continue;

      auto kernels = makeMicroKernels(modelDesc, nnXLen, nnYLen, batchSize, tunedConfig);
      for(auto& kernel: kernels) {
        const string name = kernel->name();
        if(name == "winogradTransformNhwc" || name == "winogradUntransformNhwc")
          continue;
        const bool acceleratedGemmAvailable = ctx->fp16Storage && ctx->fp16Compute &&
          (ctx->dev->info.supportsCoopmat1F16 || ctx->dev->info.supportsCoopmat1F16AccF16 ||
           ctx->dev->info.supportsCoopmat2F16 || ctx->dev->info.supportsCoopmat2F16AccF16 ||
           ctx->dev->info.supportsDot2F16 || ctx->dev->info.supportsDot2F16AccF16);
        const bool stridedTiledKernel = name == "gemmStridedTiledNhwc";
        if(acceleratedGemmAvailable && (name == "winogradGemmTiled" || stridedTiledKernel))
          continue;
        int32_t family = TUNED_NHWC_REDUCTIONS;
        if(name == "gemmDirect")
          family = TUNED_GEMM_DIRECT;
        else if(name == "nchwToNhwc")
          family = TUNED_NCHW_TO_NHWC;
        else if(name == "gemmStridedTiledNhwc")
          family = TUNED_NHWC_STRIDED;
        else if(name == "attnTiledNhwc" || name == "spatialRMSNormNhwc")
          family = TUNED_NHWC_TRANSFORMER;
        else if(name == "swiGLU")
          family = TUNED_SWIGLU;
        else if(name == "gpoolReductionNhwc" || name == "valueHeadPoolNhwc")
          family = TUNED_NHWC_REDUCTIONS;
        else if(name == "winogradTransformNhwc" || name == "winogradUntransformNhwc")
          family = TUNED_NHWC_WINOGRAD;
        if((missing & family) == 0)
          continue;

        out << endl << "VulkanTuner: tuning " << name << "..." << endl;
        if(name == "winogradGemmTiled") {
          VulkanTuneParams tunedCandidate = tunedConfig;
          bool tuned = tuneWinogradGemmTiledMultiShape<VulkanKernels::WinogradGemm>(
            name, modelDesc, batchSize, nnXLen, nnYLen, tunedConfig, *ctx, iters, out, verboseTuner, tunedCandidate);
          if(tuned) {
            tunedConfig = tunedCandidate;
            continue;
          }
          out << "VulkanTuner: falling back to single-shape winogradGemmTiled tuning." << endl;
        }
        if(name == "gemmStridedTiledNhwc") {
          VulkanTuneParams tunedCandidate = tunedConfig;
          bool tuned = tuneGemmStridedTiledMultiShape<VulkanKernels::GemmStridedTiledNhwc>(
            name, modelDesc, batchSize, nnXLen, nnYLen, tunedConfig, *ctx, iters, out, verboseTuner, tunedCandidate);
          if(tuned) {
            tunedConfig = tunedCandidate;
            continue;
          }
          out << "VulkanTuner: falling back to single-shape gemmStridedTiledNhwc tuning." << endl;
        }
        std::vector<TunedCandidate> top = tuneOne(*kernel, tunedConfig, *ctx, iters, out, verboseTuner);
        if(!top.empty())
          top[0].applyTo(tunedConfig);
      }

      if(useNhwc && (missing & TUNED_NHWC_ATTENTION_VARIANT) != 0) {
        // This is a composite tier: enabling an additional accelerator family
        // must benchmark only that missing family, then compare its saved time
        // against the already-tuned families. Do not discard a valid coopmat1
        // measurement merely because coopmat2 became eligible.
        const bool attentionPolicyChanged = tunedConfig.attnNhwcVariantPolicyVersion != 4;
        const bool coopmat1Missing = ctx->dev->info.supportsCoopmat1F16 &&
          (attentionPolicyChanged || tunedConfig.attnNhwcCoopmat1TunerValueValid == 0 ||
           tunedConfig.attnNhwcCoopmat1SplitKTunerValueValid == 0);
        const bool coopmat2Missing = ctx->dev->info.supportsCoopmat2F16 &&
          (attentionPolicyChanged || tunedConfig.attnNhwcCoopmat2TunerValueValid == 0);
        if(coopmat1Missing) {
          out << endl << "VulkanTuner: tuning attention NHWC coopmat1 tier..." << endl;
          tuneAttentionCoopmat1Nhwc(
            *ctx, modelDesc, batchSize, nnXLen, nnYLen, iters, out, verboseTuner, tunedConfig);
        }
        const bool maintenance1Missing = ctx->dev->info.supportsCoopmatMaintenance1 &&
          (attentionPolicyChanged || tunedConfig.attnNhwcCoopmatMaintenance1TunerValueValid == 0 ||
           tunedConfig.attnNhwcCoopmatMaintenance1SplitKTunerValueValid == 0);
        if(maintenance1Missing) {
          out << endl << "VulkanTuner: tuning attention NHWC coopmat maintenance1 tier..." << endl;
          tuneAttentionCoopmatMaintenance1Nhwc(
            *ctx, modelDesc, batchSize, nnXLen, nnYLen, iters, out, verboseTuner, tunedConfig);
        }
        if(coopmat2Missing) {
          out << endl << "VulkanTuner: tuning attention NHWC coopmat2 tier..." << endl;
          tuneAttentionCoopmat2AccF32Nhwc(
            *ctx, modelDesc, batchSize, nnXLen, nnYLen, iters, out, verboseTuner, tunedConfig);
        }
        if(ctx->dev->info.supportsCoopmat1F16 || ctx->dev->info.supportsCoopmat2F16) {
          const bool one = ctx->dev->info.supportsCoopmat1F16 && tunedConfig.attnNhwcCoopmat1TimeUs > 0;
          const bool two = ctx->dev->info.supportsCoopmat2F16 && tunedConfig.attnNhwcCoopmat2TimeUs > 0;
          const bool maint = ctx->dev->info.supportsCoopmatMaintenance1 && tunedConfig.attnNhwcCoopmatMaintenance1TimeUs > 0;
          const int best = std::min(one ? tunedConfig.attnNhwcCoopmat1TimeUs : INT_MAX,
                                    std::min(two ? tunedConfig.attnNhwcCoopmat2TimeUs : INT_MAX,
                                             maint ? tunedConfig.attnNhwcCoopmatMaintenance1TimeUs : INT_MAX));
          tunedConfig.attnNhwcUseCoopmatMaintenance1 = maint && tunedConfig.attnNhwcCoopmatMaintenance1TimeUs == best;
          tunedConfig.attnNhwcUseCoopmat2 = two && !tunedConfig.attnNhwcUseCoopmatMaintenance1 && tunedConfig.attnNhwcCoopmat2TimeUs == best;
          tunedConfig.attnNhwcUseCoopmat1 = one && !tunedConfig.attnNhwcUseCoopmatMaintenance1 && !tunedConfig.attnNhwcUseCoopmat2;
          tunedConfig.attnNhwcUseDot2 = 0;
        }
        else if(ctx->dev->info.supportsDot2F16) {
          out << endl << "VulkanTuner: tuning attention NHWC DOT2 AccF32 tier..." << endl;
          tuneAttentionDot2AccF32Nhwc(
            *ctx, modelDesc, batchSize, nnXLen, nnYLen, iters, out, verboseTuner, tunedConfig);
        }
      }

      const int32_t gemmFamilies =
        TUNED_NHWC_GEMM | TUNED_NHWC_GEMM_RUNTIME_MODEL | TUNED_NHWC_CONV3X3 |
        TUNED_NHWC_CONV5X5 | TUNED_NHWC_STRIDED | TUNED_NHWC_WINOGRAD;
      if((missing & gemmFamilies) != 0 ||
         (ctx->dev != nullptr && !gemmVariantFilterTunedForCurrentHardware(tunedConfig, ctx->dev->info)))
        tuneGemmVariantsForCurrentHardware(
          *ctx, modelDesc, batchSize, nnXLen, nnYLen, iters, out, verboseTuner, tunedConfig);

      if(useNhwc && (missing & TUNED_NHWC_WINOGRAD) != 0) {
        retuneNhwcWinogradTransformsForCurrentLayout(
          *ctx, modelDesc, batchSize, nnXLen, nnYLen, iters, out, verboseTuner, tunedConfig);
      }
      tunedConfig.tunedKernelMask |= required;
      if((required & TUNED_SWIGLU) != 0)
        tunedConfig.swiGLUTunerValueValid = 1;
      if(useNhwc && (required & TUNED_NHWC_GEMM) != 0)
        tunedConfig.tunedKernelMask |= TUNED_NHWC_GEMM_RUNTIME_MODEL;
      // WinogradTransformNhwc/WinogradUntransformNhwc are tuned above via the
      // generic tuneOne() loop, so mark the per-conv-size valid flags here once
      // the family has been swept (or was skipped because this device can
      // never select the path).
      if(useNhwc && (required & TUNED_NHWC_WINOGRAD) != 0) {
        tunedConfig.conv3x3NhwcWinogradTunerValueValid = 1;
        tunedConfig.conv5x5NhwcWinogradTunerValueValid = 1;
      }
    }
  }

}  // namespace VulkanTuner

// VulkanKernels member impls that need TuningContext (a full type only here).
// Each instantiable kernel (TunableKernel subclass) defines its bench() here.
// ============================================================================

namespace VulkanKernels {

  KernelBench SwiGLU::bench(VulkanTuner::TuningContext& s, const VulkanTuneParams& cfg, int iters) const {
    KernelBench result;
    if(iters < 1 || !isConfigSupported(cfg.swiGLULocalSizeX))
      return result;
    try {
      const size_t elts = (size_t)problemBatchSize * problemChannels * problemXySize;
      if(elts == 0 || elts % ELEMENTS_PER_VEC != 0)
        return result;
      std::vector<float> mainData(elts), gateData(elts);
      VulkanTuner::fillRandom(mainData, (uint32_t)(problemChannels * 73856093u ^ problemXySize * 19349663u));
      VulkanTuner::fillRandom(gateData, (uint32_t)(problemChannels * 83492791u ^ problemBatchSize * 2654435761u));
      VBuf mainBuf = s.makeInputBufFP(mainData);
      VBuf gateBuf = s.makeInputBufFP(gateData);
      ComputeKernel kernel = build(s.device(), VK_NULL_HANDLE, s.fp16Storage, cfg.swiGLULocalSizeX);
      PC pc = {(int)elts};
      auto recordOne = [&]() {
        CmdCtx cctx{s.cmd, s.getPushDescFn(), nullptr};
        dispatch(cctx, kernel, mainBuf.get(), gateBuf.get(), mainBuf.get(), pc);
        VulkanHelpers::cmdComputeBarrier(s.cmd, mainBuf->buffer);
      };
      s.beginRecording();
      recordOne();
      s.submitAndWait();
      auto [seconds, effectiveIters] = timeBudgetedBench(s, iters, recordOne);
      if(seconds > 0.0) {
        result.kernelsPerSecond = (double)effectiveIters / seconds;
        result.ok = true;
      }
      kernel.destroy(s.device());
      return result;
    } catch(const std::exception& e) {
      if(s.logger != nullptr)
        s.logger->write(std::string("VulkanTuner: swiGLU candidate failed: ") + e.what());
      return result;
    }
  }

  namespace {
    void fillAttentionRopeTables(
      std::vector<float>& cosData, std::vector<float>& sinData,
      int tableHeads, int numPairs, int numTokens) {
      for(int h = 0; h < tableHeads; h++)
        for(int pair = 0; pair < numPairs; pair++)
          for(int pos = 0; pos < numTokens; pos++) {
            const size_t i = ((size_t)h * numPairs + pair) * numTokens + pos;
            const float angle = 0.013f * (float)(pair + 1) * (float)(pos + 1) + 0.071f * (float)h;
            cosData[i] = std::cos(angle);
            sinData[i] = std::sin(angle);
          }
    }

    void transposeAttentionRopeTables(
      std::vector<float>& cosData, std::vector<float>& sinData,
      int tableHeads, int numPairs, int numTokens) {
      auto transpose = [&](std::vector<float>& table) {
        std::vector<float> transposed(table.size());
        for(int h = 0; h < tableHeads; h++)
          for(int pair = 0; pair < numPairs; pair++)
            for(int pos = 0; pos < numTokens; pos++)
              transposed[((size_t)h * numTokens + pos) * numPairs + pair] =
                table[((size_t)h * numPairs + pair) * numTokens + pos];
        table.swap(transposed);
      };
      transpose(cosData);
      transpose(sinData);
    }
  }

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
        size_t bElts = (size_t)problemK * problemN;
        size_t cElts = (size_t)problemBatchSize * problemN * problemM;
        std::vector<float> aData(aElts), bData(bElts);
        uint32_t seed = (uint32_t)(problemM * 73856093u ^ problemN * 19349663u ^ problemK * 83492791u);
        VulkanTuner::fillRandom(aData, seed);
        VulkanTuner::fillRandom(bData, seed ^ 0x9e3779b9u);

        std::vector<float> bUploadData;
        if(packB)
          bUploadData =
            packStridedGemmBWeights(bData, 1, problemN, problemK, packedBN, packedBK, packedPadScalars);

        VBuf aBuf = s.makeInputBufFP(aData);
        VBuf bBuf = s.makeInputBufFP(packB ? bUploadData : bData);
        VBuf cBuf = s.makeInputBufFP(std::vector<float>(cElts, 0.0f));
        GemmStridedPC pc = {
          problemM,
          problemN,
          problemN,  // N_real == N (bench uses aligned N)
          problemK * (problemM / 4),
          0,  // Runtime convolution and transformer weights are shared across the batch.
          problemN * (problemM / 4)};

        auto recordOne = [&]() {
          CmdCtx cctx{s.cmd, s.getPushDescFn(), nullptr};
          dispatchKernel(cctx, kernel, aBuf.get(), bBuf.get(), cBuf.get(), pc, problemBatchSize);
          VulkanHelpers::cmdComputeBarrier(s.cmd, cBuf->buffer);
        };

        s.beginRecording();
        recordOne();
        s.submitAndWait();
        result.output.resize(cElts);
        s.downloadFloatsFP(cBuf.get(), result.output, cElts);
        auto [seconds, effectiveIters] = timeBudgetedBench(s, iters, recordOne);
        if(seconds <= 0.0) {
          kernel.destroy(s.device());
          return result;
        }
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

    // Build a canonical [batch, K, M] reference ordering, transpose only the uploaded A buffer to NHWC
    // [batch, M, K], then transpose C back before RMSE comparison. This times
    // exactly the native runtime kernel without letting layout change the
    // correctness oracle.
    //
    // packedBN==0 skips the B-packing step and uploads the plain [K, N] tile;
    // the tiled NHWC kernel consumes the plain K-major B layout.
    template<typename BuildKernel, typename DispatchKernel>
    KernelBench benchGemmStridedNhwcVariant(
      VulkanTuner::TuningContext& s,
      int iters,
      int problemBatchSize,
      int problemM,
      int problemN,
      int problemK,
      string_view candidateName,
      BuildKernel&& buildKernel,
      DispatchKernel&& dispatchKernel,
      int packedBN,
      int packedBK,
      int packedPadScalars,
      bool rowMajorPackedB = false) {
      KernelBench result;
      if(iters < 1)
        return result;
      try {
        ComputeKernel kernel = buildKernel();
        const size_t aElts = (size_t)problemBatchSize * problemK * problemM;
        const size_t bElts = (size_t)problemK * problemN;
        const size_t cElts = (size_t)problemBatchSize * problemN * problemM;
        std::vector<float> aCanonical(aElts), aNhwc(aElts), bData(bElts);
        uint32_t seed = (uint32_t)(problemM * 73856093u ^ problemN * 19349663u ^ problemK * 83492791u);
        VulkanTuner::fillRandom(aCanonical, seed);
        VulkanTuner::fillRandom(bData, seed ^ 0x9e3779b9u);
        for(int batch = 0; batch < problemBatchSize; batch++)
          for(int m = 0; m < problemM; m++)
            for(int k = 0; k < problemK; k++)
              aNhwc[((size_t)batch * problemM + m) * problemK + k] =
                aCanonical[((size_t)batch * problemK + k) * problemM + m];

        std::vector<float> bUploadData;
        if(packedBN > 0)
          bUploadData = rowMajorPackedB
            ? packStridedGemmBWeightsRowMajor(
                bData, 1, problemN, problemK, packedBN, packedBK, packedPadScalars)
            : packStridedGemmBWeights(
                bData, 1, problemN, problemK, packedBN, packedBK, packedPadScalars);
        VBuf aBuf = s.makeInputBufFP(aNhwc);
        VBuf bBuf = s.makeInputBufFP(packedBN > 0 ? bUploadData : bData);
        VBuf cBuf = s.makeInputBufFP(std::vector<float>(cElts, 0.0f));
        GemmStridedPC pc = {
          problemM,
          problemN,
          problemN,
          problemK * (problemM / 4),
          0,  // Runtime convolution weights are shared across the batch.
          problemN * (problemM / 4)};
        auto recordOne = [&]() {
          CmdCtx cctx{s.cmd, s.getPushDescFn(), nullptr};
          dispatchKernel(cctx, kernel, aBuf.get(), bBuf.get(), cBuf.get(), pc, problemBatchSize);
          VulkanHelpers::cmdComputeBarrier(s.cmd, cBuf->buffer);
        };
        s.beginRecording();
        recordOne();
        s.submitAndWait();
        std::vector<float> cNhwc(cElts);
        s.downloadFloatsFP(cBuf.get(), cNhwc, cElts);
        result.output.resize(cElts);
        for(int batch = 0; batch < problemBatchSize; batch++)
          for(int m = 0; m < problemM; m++)
            for(int n = 0; n < problemN; n++)
              result.output[((size_t)batch * problemN + n) * problemM + m] =
                cNhwc[((size_t)batch * problemM + m) * problemN + n];
        auto [seconds, effectiveIters] = timeBudgetedBench(s, iters, recordOne);
        if(seconds <= 0.0) {
          kernel.destroy(s.device());
          return result;
        }
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

  namespace {
    KernelBench benchLayoutTransformDirection(
      VulkanTuner::TuningContext& s,
      const VulkanTuneParams& cfg,
      int iters,
      int problemBatchSize,
      int problemSpatialSize,
      const std::vector<int>& problemChannels) {
      KernelBench result;
      if(iters < 1 || problemBatchSize < 1 || problemSpatialSize < 1 || problemChannels.empty())
        return result;

      LayoutTransformKernels kernels;
      try {
        kernels = NchwToNhwc::build(s.device(), VK_NULL_HANDLE, s.fp16Storage, cfg);

        struct CaseBuffers {
          int channels;
          size_t elements;
          VBuf input;
          VBuf output;
        };
        std::vector<CaseBuffers> cases;
        cases.reserve(problemChannels.size());
        for(int channels: problemChannels) {
          const size_t elements = (size_t)problemBatchSize * (size_t)channels * (size_t)problemSpatialSize;
          std::vector<float> inputData(elements);
          VulkanTuner::fillRandom(
            inputData,
            (uint32_t)(channels * 73856093u ^ problemSpatialSize * 19349663u ^ problemBatchSize * 83492791u));
          cases.push_back(
            {channels,
             elements,
             s.makeInputBufFP(inputData),
             makeDeviceBuf(s.device(), s.memProps(), elements, s.fp16Storage)});
        }

        auto recordOne = [&]() {
          CmdCtx cctx{s.cmd, s.getPushDescFn(), nullptr};
          for(CaseBuffers& cs: cases) {
            NchwToNhwc::PC pc = {cs.channels, cs.channels, problemSpatialSize, problemBatchSize};
            NchwToNhwc::dispatch(cctx, kernels, cs.input.get(), cs.output.get(), pc);
            VulkanHelpers::cmdComputeBarrier(s.cmd, cs.output->buffer);
          }
        };

        s.beginRecording();
        recordOne();
        s.submitAndWait();
        auto [seconds, effectiveIters] = timeBudgetedBench(s, iters, recordOne);
        if(seconds <= 0.0) {
          kernels.destroy(s.device());
          return result;
        }

        size_t totalElements = 0;
        for(const CaseBuffers& cs: cases)
          totalElements += cs.elements;
        result.output.reserve(totalElements);
        for(CaseBuffers& cs: cases) {
          std::vector<float> output(cs.elements);
          s.downloadFloatsFP(cs.output.get(), output, cs.elements);
          result.output.insert(result.output.end(), output.begin(), output.end());
        }
        result.kernelsPerSecond = (double)effectiveIters / seconds;
        result.ok = true;
        kernels.destroy(s.device());
        return result;
      } catch(const std::exception& e) {
        kernels.destroy(s.device());
        if(s.logger != nullptr)
          s.logger->write(
            std::string("VulkanTuner: nchwToNhwc candidate failed: ") + e.what());
        return result;
      }
    }
  }

  KernelBench NchwToNhwcTuner::bench(
    VulkanTuner::TuningContext& s, const VulkanTuneParams& cfg, int iters) const {
    return benchLayoutTransformDirection(
      s, cfg, iters, problemBatchSize, problemSpatialSize, problemChannels);
  }

  KernelBench WinogradTransformNhwc::bench(VulkanTuner::TuningContext& s, const VulkanTuneParams& cfg, int iters) const {
    KernelBench result;
    if(iters < 1)
      return result;
    try {
      const int outTile = nhwcWinograd3x3OutTileFor(cfg);
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
        (uint32_t)(problemNnXLen * 73856093u ^ problemInChannels * 19349663u ^ problemBatchSize * 83492791u ^ 41u));

      VBuf inputBuf = s.makeInputBufFP(inputData);
      VBuf outputBuf = makeDeviceBuf(s.device(), s.memProps(), outputElts, s.fp16Storage);

      WinogradTransformNhwc::PC pc = {
        problemNnXLen,
        problemNnYLen,
        numTilesX,
        numTilesY,
        problemInChannels,
        numInChannelsPadded,
        ntxty,
        numTilesPadded,
        paddedSpatialSize};

      ComputeKernel kernel = WinogradTransformNhwc::build(
        s.device(),
        VK_NULL_HANDLE,
        s.fp16Storage,
        inTile,
        outTile,
        3,
        -1,
        cfg.nhwcWinogradTransformLocalSizeX,
        cfg.nhwcWinogradTransformLocalSizeY,
        layout.packedBM,
        layout.packedBK,
        layout.packedAPadWords);

      auto recordOne = [&]() {
        CmdCtx cctx{s.cmd, s.getPushDescFn(), nullptr};
        WinogradTransformNhwc::dispatch(cctx, kernel, inputBuf.get(), outputBuf.get(), pc);
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
        s.logger->write(std::string("VulkanTuner: winogradNhwc candidate failed: ") + e.what());
      return result;
    }
  }

  KernelBench WinogradUntransformNhwc::bench(VulkanTuner::TuningContext& s, const VulkanTuneParams& cfg, int iters) const {
    KernelBench result;
    if(iters < 1)
      return result;
    try {
      const int outTile = nhwcWinograd3x3OutTileFor(cfg);
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
        (uint32_t)(problemNnXLen * 73856093u ^ problemOutChannels * 19349663u ^ problemBatchSize * 83492791u ^ 53u));

      VBuf inputBuf = s.makeInputBufFP(inputData);
      VBuf outputBuf = makeDeviceBuf(s.device(), s.memProps(), outputElts, s.fp16Storage);

      WinogradUntransformNhwc::PC pc = {
        problemNnXLen,
        problemNnYLen,
        numTilesX,
        numTilesY,
        problemOutChannels,
        numOutChannelsPadded,
        numTilesPadded,
        paddedSpatialSize};

      // ocVec is derived from the unpadded problemOutChannels, matching the
      // real construction site in vulkanlayers.h -- padding is irrelevant to
      // whether the store can address 4 contiguous physical channels.
      const int ocVec = (problemOutChannels % 4 == 0) ? 4 : 1;
      ComputeKernel kernel = WinogradUntransformNhwc::build(
        s.device(),
        VK_NULL_HANDLE,
        s.fp16Storage,
        inTile,
        outTile,
        3,
        cfg.nhwcWinogradUntransformLocalSizeX,
        cfg.nhwcWinogradUntransformLocalSizeY,
        false,
        ocVec);

      auto recordOne = [&]() {
        CmdCtx cctx{s.cmd, s.getPushDescFn(), nullptr};
        WinogradUntransformNhwc::dispatch(cctx, kernel, inputBuf.get(), outputBuf.get(), pc, problemBatchSize, ocVec);
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
        s.logger->write(std::string("VulkanTuner: winogradUntransformNhwc candidate failed: ") + e.what());
      return result;
    }
  }

  KernelBench GemmStridedTiledNhwc::bench(VulkanTuner::TuningContext& s, const VulkanTuneParams& cfg, int iters) const {
    return benchGemmStridedNhwcVariant(
      s,
      iters,
      problemBatchSize,
      problemM,
      problemN,
      problemK,
      "gemmStridedTiledNhwc",
      [&]() {
        return GemmStridedTiledNhwc::build(
          s.device(),
          VK_NULL_HANDLE,
          s.fp16Storage,
          (uint32_t)cfg.gemmStridedTiledNhwcLocalSizeX,
          (uint32_t)cfg.gemmStridedTiledNhwcLocalSizeY,
          (uint32_t)cfg.gemmStridedTiledNhwcTileK,
          (uint32_t)cfg.gemmStridedTiledNhwcRN,
          (uint32_t)problemK);
      },
      [&](
        const CmdCtx& cctx,
        const ComputeKernel& kernel,
        VulkanBuffer* aBuf,
        VulkanBuffer* bBuf,
        VulkanBuffer* cBuf,
        const GemmStridedTiledNhwc::PC& pc,
        int batchSize) { GemmStridedTiledNhwc::dispatch(cctx, kernel, aBuf, bBuf, cBuf, problemK, pc, batchSize); },
      0,  // No B repack; the tiled NHWC shader consumes the canonical B layout.
      0,
      0);
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

  KernelBench Conv3x3ImplicitGemmCoopmat1AccF32NhwcVec8::bench(
    VulkanTuner::TuningContext& s,
    const VulkanTuneParams& cfg,
    int iters,
    int batchSize,
    int nnXLen,
    int nnYLen,
    int paddedSpatialSize,
    int convSize,
    int inChannels,
    int outChannels) {
    KernelBench result;
    if(
      iters < 1 || batchSize < 1 || nnXLen < 1 || nnYLen < 1 || inChannels < 1 || outChannels < 1 ||
      (convSize != 3 && convSize != 5) || s.dev == nullptr || !s.dev->info.supportsCoopmat1F16 || !s.fp16Storage)
      return result;
    const bool is5x5 = convSize == 5;
    const int blockSize = is5x5 ? cfg.conv5x5NhwcCoopmat1BlockSize : cfg.conv3x3NhwcCoopmat1BlockSize;
    const int bm = is5x5 ? cfg.conv5x5NhwcCoopmat1BM : cfg.conv3x3NhwcCoopmat1BM;
    const int bn = is5x5 ? cfg.conv5x5NhwcCoopmat1BN : cfg.conv3x3NhwcCoopmat1BN;
    const int bk = is5x5 ? cfg.conv5x5NhwcCoopmat1BK : cfg.conv3x3NhwcCoopmat1BK;
    const int wm = is5x5 ? cfg.conv5x5NhwcCoopmat1WM : cfg.conv3x3NhwcCoopmat1WM;
    const int wn = is5x5 ? cfg.conv5x5NhwcCoopmat1WN : cfg.conv3x3NhwcCoopmat1WN;
    const int tm = is5x5 ? cfg.conv5x5NhwcCoopmat1TM : cfg.conv3x3NhwcCoopmat1TM;
    const int tn = is5x5 ? cfg.conv5x5NhwcCoopmat1TN : cfg.conv3x3NhwcCoopmat1TN;
    const int tk = is5x5 ? cfg.conv5x5NhwcCoopmat1TK : cfg.conv3x3NhwcCoopmat1TK;
    const int warp = is5x5 ? cfg.conv5x5NhwcCoopmat1Warp : cfg.conv3x3NhwcCoopmat1Warp;
    if(!coopmatShapeIsSupported(
         s.dev->info.coopmatShapes,
         tm, tn, tk))
      return result;
    if(!isConfigSupported(
         blockSize, bm, bn, bk, wm, wn, tm, tn, tk, warp))
      return result;
    const VkPhysicalDeviceLimits& limits = s.dev->info.properties.limits;
    if(
      (uint32_t)blockSize > limits.maxComputeWorkGroupInvocations ||
      (uint32_t)blockSize > limits.maxComputeWorkGroupSize[0] ||
      sharedBytes(bm, bn, bk) >
        limits.maxComputeSharedMemorySize ||
      paddedSpatialSize % bm != 0 || outChannels % bn != 0 || inChannels % 8 != 0)
      return result;

    try {
      const int flattenedK = convSize * convSize * inChannels;
      ComputeKernel kernel = build(
        s.device(),
        VK_NULL_HANDLE,
        blockSize, bm, bn, bk, wm, wn, tm, tn, tk, warp,
        convSize,
        flattenedK,
        false,
        inChannels % bk == 0,
        s.dev->info.subgroupSize);

      const size_t inputElts = (size_t)batchSize * paddedSpatialSize * inChannels;
      const size_t filterElts = (size_t)flattenedK * outChannels;
      const size_t outputElts = (size_t)batchSize * paddedSpatialSize * outChannels;
      std::vector<float> inputData(inputElts), filterData(filterElts);
      const uint32_t seed =
        (uint32_t)(nnXLen * 73856093u ^ nnYLen * 19349663u ^ inChannels * 83492791u ^ outChannels * 2654435761u);
      VulkanTuner::fillRandom(inputData, seed);
      VulkanTuner::fillRandom(filterData, seed ^ 0x9e3779b9u);
      std::vector<float> packedFilter = packStridedGemmBWeights(
        filterData,
        1,
        outChannels,
        flattenedK,
        bn,
        bk,
        STRIDED_COOPMAT_PACKED_B_PAD_SCALARS);

      VBuf inputBuf = s.makeInputBufFP(inputData);
      VBuf filterBuf = s.makeInputBufFP(packedFilter);
      VBuf outputBuf = makeDeviceBuf(s.device(), s.memProps(), outputElts, s.fp16Storage);
      PC pc = {
        nnXLen,
        nnYLen,
        outChannels,
        outChannels,
        inChannels,
        paddedSpatialSize,
        0,
        outChannels * (paddedSpatialSize / 4),
        0};
      auto recordOne = [&]() {
        CmdCtx cctx{s.cmd, s.getPushDescFn(), nullptr};
        dispatch(cctx, kernel, inputBuf.get(), filterBuf.get(), outputBuf.get(), convSize, pc, batchSize);
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
        s.logger->write(std::string("VulkanTuner: implicitNhwcCoopmat candidate failed: ") + e.what());
      return result;
    }
  }

  KernelBench Conv3x3ImplicitGemmCoopmat1AccF16NhwcVec8::bench(
    VulkanTuner::TuningContext& s, const VulkanTuneParams& cfg, int iters, int batchSize,
    int nnXLen, int nnYLen, int paddedSpatialSize, int convSize, int inChannels, int outChannels) {
    KernelBench result;
    const bool is5x5 = convSize == 5;
    const int blockSize = is5x5 ? cfg.conv5x5NhwcCoopmat1AccF16BlockSize : cfg.conv3x3NhwcCoopmat1AccF16BlockSize;
    const int bm = is5x5 ? cfg.conv5x5NhwcCoopmat1AccF16BM : cfg.conv3x3NhwcCoopmat1AccF16BM;
    const int bn = is5x5 ? cfg.conv5x5NhwcCoopmat1AccF16BN : cfg.conv3x3NhwcCoopmat1AccF16BN;
    const int bk = is5x5 ? cfg.conv5x5NhwcCoopmat1AccF16BK : cfg.conv3x3NhwcCoopmat1AccF16BK;
    const int wm = is5x5 ? cfg.conv5x5NhwcCoopmat1AccF16WM : cfg.conv3x3NhwcCoopmat1AccF16WM;
    const int wn = is5x5 ? cfg.conv5x5NhwcCoopmat1AccF16WN : cfg.conv3x3NhwcCoopmat1AccF16WN;
    const int tm = is5x5 ? cfg.conv5x5NhwcCoopmat1AccF16TM : cfg.conv3x3NhwcCoopmat1AccF16TM;
    const int tn = is5x5 ? cfg.conv5x5NhwcCoopmat1AccF16TN : cfg.conv3x3NhwcCoopmat1AccF16TN;
    const int tk = is5x5 ? cfg.conv5x5NhwcCoopmat1AccF16TK : cfg.conv3x3NhwcCoopmat1AccF16TK;
    const int warp = is5x5 ? cfg.conv5x5NhwcCoopmat1AccF16Warp : cfg.conv3x3NhwcCoopmat1AccF16Warp;
    if(iters < 1 || batchSize < 1 || nnXLen < 1 || nnYLen < 1 || inChannels < 1 || outChannels < 1 ||
       (convSize != 3 && convSize != 5) || s.dev == nullptr || !s.dev->info.supportsCoopmat1F16AccF16 || !s.fp16Storage ||
       !coopmatShapeIsSupported(s.dev->info.coopmatAccF16Shapes,tm,tn,tk) ||
       !isConfigSupported(blockSize,bm,bn,bk,wm,wn,tm,tn,tk,warp) || paddedSpatialSize % bm != 0 ||
       outChannels % bn != 0 || inChannels % 8 != 0)
      return result;
    const VkPhysicalDeviceLimits& limits = s.dev->info.properties.limits;
    if((uint32_t)blockSize > limits.maxComputeWorkGroupInvocations ||
       (uint32_t)blockSize > limits.maxComputeWorkGroupSize[0] ||
       sharedBytes(bm,bn,bk) > limits.maxComputeSharedMemorySize)
      return result;
    try {
      const int flattenedK = convSize * convSize * inChannels;
      ComputeKernel kernel = build(s.device(),VK_NULL_HANDLE,blockSize,bm,bn,bk,wm,wn,tm,tn,tk,warp,
        convSize,flattenedK,false,inChannels % bk == 0,s.dev->info.subgroupSize);
      const size_t inputElts=(size_t)batchSize*paddedSpatialSize*inChannels;
      const size_t outputElts=(size_t)batchSize*paddedSpatialSize*outChannels;
      std::vector<float> inputData(inputElts),filterData((size_t)flattenedK*outChannels);
      const uint32_t seed=(uint32_t)(nnXLen*73856093u ^ nnYLen*19349663u ^ inChannels*83492791u ^ outChannels*2654435761u);
      VulkanTuner::fillRandom(inputData,seed); VulkanTuner::fillRandom(filterData,seed^0x9e3779b9u);
      std::vector<float> packedFilter=packStridedGemmBWeights(filterData,1,outChannels,flattenedK,bn,bk,
        STRIDED_COOPMAT_PACKED_B_PAD_SCALARS);
      VBuf inputBuf=s.makeInputBufFP(inputData),filterBuf=s.makeInputBufFP(packedFilter);
      VBuf outputBuf=makeDeviceBuf(s.device(),s.memProps(),outputElts,s.fp16Storage);
      PC pc={nnXLen,nnYLen,outChannels,outChannels,inChannels,paddedSpatialSize,0,outChannels*(paddedSpatialSize/4),0};
      auto recordOne=[&](){CmdCtx cctx{s.cmd,s.getPushDescFn(),nullptr};
        dispatch(cctx,kernel,inputBuf.get(),filterBuf.get(),outputBuf.get(),convSize,pc,batchSize);
        VulkanHelpers::cmdComputeBarrier(s.cmd,outputBuf->buffer);};
      s.beginRecording(); recordOne(); s.submitAndWait(); auto [seconds,effectiveIters]=timeBudgetedBench(s,iters,recordOne);
      if(seconds<=0.0){kernel.destroy(s.device());return result;}
      result.output.resize(outputElts);s.downloadFloatsFP(outputBuf.get(),result.output,outputElts);
      result.kernelsPerSecond=(double)effectiveIters/seconds;result.ok=true;kernel.destroy(s.device());
    } catch(const std::exception& e) {
      if(s.logger) s.logger->write(std::string("VulkanTuner: implicitNhwcCoopmatAccF16 candidate failed: ")+e.what());
    }
    return result;
  }

  KernelBench Conv3x3ImplicitGemmCoopmat2AccF32NhwcVec8::bench(
    VulkanTuner::TuningContext& s, const VulkanTuneParams& cfg, int iters, int batchSize,
    int nnXLen, int nnYLen, int paddedSpatialSize, int convSize, int inChannels, int outChannels) {
    KernelBench result;
    const bool is5x5 = convSize == 5;
    const int blockSize = is5x5 ? cfg.conv5x5NhwcCoopmat2BlockSize : cfg.conv3x3NhwcCoopmat2BlockSize;
    const int bm = is5x5 ? cfg.conv5x5NhwcCoopmat2BM : cfg.conv3x3NhwcCoopmat2BM;
    const int bn = is5x5 ? cfg.conv5x5NhwcCoopmat2BN : cfg.conv3x3NhwcCoopmat2BN;
    const int bk = is5x5 ? cfg.conv5x5NhwcCoopmat2BK : cfg.conv3x3NhwcCoopmat2BK;
    if(iters<1 || batchSize<1 || s.dev==nullptr || !s.dev->info.supportsCoopmat2F16 || !s.fp16Storage ||
       (convSize != 3 && convSize != 5) || inChannels%8!=0 || paddedSpatialSize%bm!=0 ||
       outChannels%bn!=0 || !isConfigSupported(blockSize,bm,bn,bk)) return result;
    try {
      const int K=convSize*convSize*inChannels;
      const int packedK=K;
      ComputeKernel kernel=build(s.device(),VK_NULL_HANDLE,blockSize,bm,bn,bk,
        convSize,packedK,false);
      const size_t aElts=(size_t)batchSize*paddedSpatialSize*inChannels;
      const size_t cElts=(size_t)batchSize*paddedSpatialSize*outChannels;
      std::vector<float> a(aElts),b((size_t)K*outChannels);
      const uint32_t seed=(uint32_t)(nnXLen*73856093u ^ nnYLen*19349663u ^
        inChannels*83492791u ^ outChannels*2654435761u);
      VulkanTuner::fillRandom(a,seed);
      VulkanTuner::fillRandom(b,seed^0x9e3779b9u);
      b=packStridedGemmBWeights(b,1,outChannels,packedK,bn,bk,STRIDED_COOPMAT2_PACKED_B_PAD_SCALARS);
      VBuf aBuf=s.makeInputBufFP(a),bBuf=s.makeInputBufFP(b);
      VBuf cBuf=makeDeviceBuf(s.device(),s.memProps(),cElts,s.fp16Storage);
      PC pc={nnXLen,nnYLen,outChannels,outChannels,inChannels,paddedSpatialSize,0,
        outChannels*(paddedSpatialSize/4),0};
      auto record=[&](){CmdCtx c{s.cmd,s.getPushDescFn(),nullptr};dispatch(c,kernel,aBuf.get(),bBuf.get(),cBuf.get(),convSize,pc,batchSize);
        VulkanHelpers::cmdComputeBarrier(s.cmd,cBuf->buffer);};
      s.beginRecording();record();s.submitAndWait();
      auto timing=timeBudgetedBench(s,iters,record);
      if(timing.seconds<=0.0){kernel.destroy(s.device());return result;}
      result.output.resize(cElts);s.downloadFloatsFP(cBuf.get(),result.output,cElts);
      result.kernelsPerSecond=(double)timing.iters/timing.seconds;result.ok=true;kernel.destroy(s.device());
    } catch(const std::exception& e) {
      if(s.logger) s.logger->write(std::string("VulkanTuner: implicitNhwcCoopmat2 candidate failed: ")+e.what());
    }
    return result;
  }

  KernelBench Conv3x3ImplicitGemmCoopmat2AccF16NhwcVec8::bench(
    VulkanTuner::TuningContext& s, const VulkanTuneParams& cfg, int iters, int batchSize,
    int nnXLen, int nnYLen, int paddedSpatialSize, int convSize, int inChannels, int outChannels) {
    KernelBench result;
    const bool is5x5=convSize==5;
    const int blockSize=is5x5?cfg.conv5x5NhwcCoopmat2AccF16BlockSize:cfg.conv3x3NhwcCoopmat2AccF16BlockSize;
    const int bm=is5x5?cfg.conv5x5NhwcCoopmat2AccF16BM:cfg.conv3x3NhwcCoopmat2AccF16BM;
    const int bn=is5x5?cfg.conv5x5NhwcCoopmat2AccF16BN:cfg.conv3x3NhwcCoopmat2AccF16BN;
    const int bk=is5x5?cfg.conv5x5NhwcCoopmat2AccF16BK:cfg.conv3x3NhwcCoopmat2AccF16BK;
    if(iters<1 || batchSize<1 || s.dev==nullptr || !s.dev->info.supportsCoopmat2F16AccF16 || !s.fp16Storage ||
       (convSize!=3 && convSize!=5) || inChannels%8!=0 || paddedSpatialSize%bm!=0 || outChannels%bn!=0 ||
       !isConfigSupported(blockSize,bm,bn,bk)) return result;
    const VkPhysicalDeviceLimits& limits=s.dev->info.properties.limits;
    if((uint32_t)blockSize>limits.maxComputeWorkGroupInvocations || (uint32_t)blockSize>limits.maxComputeWorkGroupSize[0] ||
       (size_t)s.dev->info.coopmat2ReservedSharedBytes+sharedBytes(bm,bk)>limits.maxComputeSharedMemorySize) return result;
    try {
      const int K=convSize*convSize*inChannels;
      ComputeKernel kernel=build(s.device(),VK_NULL_HANDLE,blockSize,bm,bn,bk,convSize,K,false);
      const size_t aElts=(size_t)batchSize*paddedSpatialSize*inChannels;
      const size_t cElts=(size_t)batchSize*paddedSpatialSize*outChannels;
      std::vector<float> a(aElts),b((size_t)K*outChannels);
      const uint32_t seed=(uint32_t)(nnXLen*73856093u ^ nnYLen*19349663u ^ inChannels*83492791u ^ outChannels*2654435761u);
      VulkanTuner::fillRandom(a,seed);VulkanTuner::fillRandom(b,seed^0x9e3779b9u);
      b=packStridedGemmBWeights(b,1,outChannels,K,bn,bk,STRIDED_COOPMAT2_PACKED_B_PAD_SCALARS);
      VBuf aBuf=s.makeInputBufFP(a),bBuf=s.makeInputBufFP(b);
      VBuf cBuf=makeDeviceBuf(s.device(),s.memProps(),cElts,s.fp16Storage);
      PC pc={nnXLen,nnYLen,outChannels,outChannels,inChannels,paddedSpatialSize,0,outChannels*(paddedSpatialSize/4),0};
      auto record=[&](){CmdCtx c{s.cmd,s.getPushDescFn(),nullptr};dispatch(c,kernel,aBuf.get(),bBuf.get(),cBuf.get(),convSize,pc,batchSize);
        VulkanHelpers::cmdComputeBarrier(s.cmd,cBuf->buffer);};
      s.beginRecording();record();s.submitAndWait();auto timing=timeBudgetedBench(s,iters,record);
      if(timing.seconds<=0.0){kernel.destroy(s.device());return result;}
      result.output.resize(cElts);s.downloadFloatsFP(cBuf.get(),result.output,cElts);
      result.kernelsPerSecond=(double)timing.iters/timing.seconds;result.ok=true;kernel.destroy(s.device());
    } catch(const std::exception& e) {
      if(s.logger) s.logger->write(std::string("VulkanTuner: implicitNhwcCoopmat2AccF16 candidate failed: ")+e.what());
    }
    return result;
  }

  KernelBench WinogradGemmCoopmat1::bench(VulkanTuner::TuningContext& s, const VulkanTuneParams& cfg, int iters) const {
    if(s.dev == nullptr || !s.dev->info.supportsCoopmat1F16 || !s.fp16Storage)
      return KernelBench();
    if(!coopmatShapeIsSupported(s.dev->info.coopmatShapes, cfg.coopmat1TM, cfg.coopmat1TN, cfg.coopmat1TK))
      return KernelBench();
    return benchWinogradGemmVariant(
      s,
      iters,
      problemM,
      problemN,
      problemK,
      problemNumBatches,
      "winogradGemmCoopmat1",
      [&]() {
        return WinogradGemmCoopmat1::build(
          s.device(),
          VK_NULL_HANDLE,
          cfg.coopmat1BlockSize,
          cfg.coopmat1BM,
          cfg.coopmat1BN,
          cfg.coopmat1BK,
          cfg.coopmat1WM,
          cfg.coopmat1WN,
          cfg.coopmat1TM,
          cfg.coopmat1TN,
          cfg.coopmat1TK,
          cfg.coopmat1Warp,
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
        int numBatches) { WinogradGemmCoopmat1::dispatch(cctx, kernel, aBuf, bBuf, cBuf, problemK, pc, numBatches); },
      true,
      cfg.coopmat1BM,
      cfg.coopmat1BN,
      cfg.coopmat1BK);
  }
  KernelBench WinogradGemmCoopmat1AccF16::bench(VulkanTuner::TuningContext& s, const VulkanTuneParams& cfg, int iters)
    const {
    if(s.dev == nullptr || !s.dev->info.supportsCoopmat1F16AccF16 || !s.fp16Storage)
      return KernelBench();
    if(!coopmatShapeIsSupported(
         s.dev->info.coopmatAccF16Shapes, cfg.coopmat1AccF16TM, cfg.coopmat1AccF16TN, cfg.coopmat1AccF16TK))
      return KernelBench();
    return benchWinogradGemmVariant(
      s,
      iters,
      problemM,
      problemN,
      problemK,
      problemNumBatches,
      "winogradGemmCoopmat1AccF16",
      [&]() {
        return WinogradGemmCoopmat1AccF16::build(
          s.device(),
          VK_NULL_HANDLE,
          cfg.coopmat1AccF16BlockSize,
          cfg.coopmat1AccF16BM,
          cfg.coopmat1AccF16BN,
          cfg.coopmat1AccF16BK,
          cfg.coopmat1AccF16WM,
          cfg.coopmat1AccF16WN,
          cfg.coopmat1AccF16TM,
          cfg.coopmat1AccF16TN,
          cfg.coopmat1AccF16TK,
          cfg.coopmat1AccF16Warp,
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
        WinogradGemmCoopmat1AccF16::dispatch(cctx, kernel, aBuf, bBuf, cBuf, problemK, pc, numBatches);
      },
      true,
      cfg.coopmat1AccF16BM,
      cfg.coopmat1AccF16BN,
      cfg.coopmat1AccF16BK);
  }
  KernelBench GemmStridedCoopmat1Nhwc::bench(
    VulkanTuner::TuningContext& s,
    const VulkanTuneParams& cfg,
    int iters,
    int batchSize,
    int M,
    int N,
    int K,
    bool addToOutput) {
    if(s.dev == nullptr || !s.dev->info.supportsCoopmat1F16 || !s.fp16Storage)
      return KernelBench();
    if(!coopmatShapeIsSupported(
         s.dev->info.coopmatShapes,
         cfg.nhwcStridedCoopmat1TM,
         cfg.nhwcStridedCoopmat1TN,
         cfg.nhwcStridedCoopmat1TK))
      return KernelBench();
    if(!isConfigSupported(
         cfg.nhwcStridedCoopmat1BlockSize, cfg.nhwcStridedCoopmat1BM, cfg.nhwcStridedCoopmat1BN,
         cfg.nhwcStridedCoopmat1BK, cfg.nhwcStridedCoopmat1WM, cfg.nhwcStridedCoopmat1WN,
         cfg.nhwcStridedCoopmat1TM, cfg.nhwcStridedCoopmat1TN, cfg.nhwcStridedCoopmat1TK,
         cfg.nhwcStridedCoopmat1Warp))
      return KernelBench();
    if(sharedBytes(
         cfg.nhwcStridedCoopmat1BlockSize, cfg.nhwcStridedCoopmat1BM, cfg.nhwcStridedCoopmat1BN,
         cfg.nhwcStridedCoopmat1BK, cfg.nhwcStridedCoopmat1TM, cfg.nhwcStridedCoopmat1TN,
         cfg.nhwcStridedCoopmat1Warp) > s.dev->info.properties.limits.maxComputeSharedMemorySize)
      return KernelBench();
    const int aligned = (M % cfg.nhwcStridedCoopmat1BM == 0 && N % cfg.nhwcStridedCoopmat1BN == 0) ? 1 : 0;
    return benchGemmStridedNhwcVariant(
      s, iters, batchSize, M, N, K, "gemmStridedCoopmat1Nhwc",
      [&]() { return build(
        s.device(), VK_NULL_HANDLE, cfg.nhwcStridedCoopmat1BlockSize, cfg.nhwcStridedCoopmat1BM,
        cfg.nhwcStridedCoopmat1BN, cfg.nhwcStridedCoopmat1BK, cfg.nhwcStridedCoopmat1WM,
        cfg.nhwcStridedCoopmat1WN, cfg.nhwcStridedCoopmat1TM, cfg.nhwcStridedCoopmat1TN,
        cfg.nhwcStridedCoopmat1TK, cfg.nhwcStridedCoopmat1Warp, aligned, K,
        s.dev->info.subgroupSize, addToOutput); },
      [&](const CmdCtx& cctx, const ComputeKernel& kernel, VulkanBuffer* aBuf, VulkanBuffer* bBuf,
          VulkanBuffer* cBuf, const GemmStridedPC& pc, int dispatchBatchSize) {
        dispatch(cctx, kernel, aBuf, bBuf, cBuf, K, pc, dispatchBatchSize);
      },
      cfg.nhwcStridedCoopmat1BN, cfg.nhwcStridedCoopmat1BK,
      STRIDED_COOPMAT_PACKED_B_PAD_SCALARS, true);
  }

  KernelBench GemmStridedCoopmat1AccF16Nhwc::bench(
    VulkanTuner::TuningContext& s,
    const VulkanTuneParams& cfg,
    int iters,
    int batchSize,
    int M,
    int N,
    int K,
    bool addToOutput) {
    if(s.dev == nullptr || !s.dev->info.supportsCoopmat1F16AccF16 || !s.fp16Storage)
      return KernelBench();
    if(!coopmatShapeIsSupported(
         s.dev->info.coopmatAccF16Shapes,
         cfg.nhwcStridedCoopmat1AccF16TM,
         cfg.nhwcStridedCoopmat1AccF16TN,
         cfg.nhwcStridedCoopmat1AccF16TK))
      return KernelBench();
    if(!GemmStridedCoopmat1AccF16Nhwc::isConfigSupported(
         cfg.nhwcStridedCoopmat1AccF16BlockSize,
         cfg.nhwcStridedCoopmat1AccF16BM,
         cfg.nhwcStridedCoopmat1AccF16BN,
         cfg.nhwcStridedCoopmat1AccF16BK,
         cfg.nhwcStridedCoopmat1AccF16WM,
         cfg.nhwcStridedCoopmat1AccF16WN,
         cfg.nhwcStridedCoopmat1AccF16TM,
         cfg.nhwcStridedCoopmat1AccF16TN,
         cfg.nhwcStridedCoopmat1AccF16TK,
         cfg.nhwcStridedCoopmat1AccF16Warp))
      return KernelBench();
    if(GemmStridedCoopmat1AccF16Nhwc::sharedBytes(
         cfg.nhwcStridedCoopmat1AccF16BlockSize,
         cfg.nhwcStridedCoopmat1AccF16BM,
         cfg.nhwcStridedCoopmat1AccF16BN,
         cfg.nhwcStridedCoopmat1AccF16BK,
         cfg.nhwcStridedCoopmat1AccF16TM,
         cfg.nhwcStridedCoopmat1AccF16TN,
         cfg.nhwcStridedCoopmat1AccF16Warp) > s.dev->info.properties.limits.maxComputeSharedMemorySize)
      return KernelBench();
    const int32_t aligned =
      (M % cfg.nhwcStridedCoopmat1AccF16BM == 0 && N % cfg.nhwcStridedCoopmat1AccF16BN == 0) ? 1 : 0;
    return benchGemmStridedNhwcVariant(
      s,
      iters,
      batchSize,
      M,
      N,
      K,
      "gemmStridedCoopmat1AccF16Nhwc",
      [&]() {
        return GemmStridedCoopmat1AccF16Nhwc::build(
          s.device(),
          VK_NULL_HANDLE,
          cfg.nhwcStridedCoopmat1AccF16BlockSize,
          cfg.nhwcStridedCoopmat1AccF16BM,
          cfg.nhwcStridedCoopmat1AccF16BN,
          cfg.nhwcStridedCoopmat1AccF16BK,
          cfg.nhwcStridedCoopmat1AccF16WM,
          cfg.nhwcStridedCoopmat1AccF16WN,
          cfg.nhwcStridedCoopmat1AccF16TM,
          cfg.nhwcStridedCoopmat1AccF16TN,
          cfg.nhwcStridedCoopmat1AccF16TK,
          cfg.nhwcStridedCoopmat1AccF16Warp,
          aligned,
          K,
          s.dev->info.subgroupSize,
          addToOutput);
      },
      [&](const CmdCtx& cctx,
          const ComputeKernel& kernel,
          VulkanBuffer* aBuf,
          VulkanBuffer* bBuf,
          VulkanBuffer* cBuf,
          const GemmStridedPC& pc,
          int dispatchBatchSize) {
        GemmStridedCoopmat1AccF16Nhwc::dispatch(cctx, kernel, aBuf, bBuf, cBuf, K, pc, dispatchBatchSize);
      },
      cfg.nhwcStridedCoopmat1AccF16BN,
      cfg.nhwcStridedCoopmat1AccF16BK,
      STRIDED_COOPMAT_PACKED_B_PAD_SCALARS,
      true);
  }

  KernelBench GemmStridedCoopmat2Nhwc::bench(
    VulkanTuner::TuningContext& s, const VulkanTuneParams& cfg, int iters,
    int batchSize, int M, int N, int K, bool addToOutput) {
    if(s.dev == nullptr || !s.dev->info.supportsCoopmat2F16 || !s.fp16Storage ||
       !isConfigSupported(cfg.nhwcStridedCoopmat2BlockSize, cfg.nhwcStridedCoopmat2BM,
         cfg.nhwcStridedCoopmat2BN, cfg.nhwcStridedCoopmat2BK))
      return KernelBench();
    const int aligned = (M % cfg.nhwcStridedCoopmat2BM == 0 && N % cfg.nhwcStridedCoopmat2BN == 0) ? 1 : 0;
    const int kAligned = K % cfg.nhwcStridedCoopmat2BK == 0 ? 1 : 0;
    return benchGemmStridedNhwcVariant(
      s, iters, batchSize, M, N, K, "gemmStridedCoopmat2Nhwc",
      [&]() { return build(s.device(), VK_NULL_HANDLE, cfg.nhwcStridedCoopmat2BlockSize,
        cfg.nhwcStridedCoopmat2BM, cfg.nhwcStridedCoopmat2BN, cfg.nhwcStridedCoopmat2BK,
        aligned, kAligned, K, addToOutput); },
      [&](const CmdCtx& c, const ComputeKernel& kernel, VulkanBuffer* a, VulkanBuffer* b, VulkanBuffer* out,
          const GemmStridedPC& pc, int batches) { dispatch(c, kernel, a, b, out, K, pc, batches); },
      cfg.nhwcStridedCoopmat2BN, cfg.nhwcStridedCoopmat2BK,
      STRIDED_COOPMAT2_PACKED_B_PAD_SCALARS, true);
  }

  KernelBench GemmStridedCoopmat2AccF16Nhwc::bench(
    VulkanTuner::TuningContext& s, const VulkanTuneParams& cfg, int iters,
    int batchSize, int M, int N, int K, bool addToOutput) {
    if(s.dev == nullptr || !s.dev->info.supportsCoopmat2F16AccF16 || !s.fp16Storage ||
       !isConfigSupported(cfg.nhwcStridedCoopmat2AccF16BlockSize,cfg.nhwcStridedCoopmat2AccF16BM,
         cfg.nhwcStridedCoopmat2AccF16BN,cfg.nhwcStridedCoopmat2AccF16BK))
      return KernelBench();
    const int aligned=(M%cfg.nhwcStridedCoopmat2AccF16BM==0 && N%cfg.nhwcStridedCoopmat2AccF16BN==0)?1:0;
    const int kAligned=K%cfg.nhwcStridedCoopmat2AccF16BK==0?1:0;
    return benchGemmStridedNhwcVariant(s,iters,batchSize,M,N,K,"gemmStridedCoopmat2AccF16Nhwc",
      [&](){return build(s.device(),VK_NULL_HANDLE,cfg.nhwcStridedCoopmat2AccF16BlockSize,
        cfg.nhwcStridedCoopmat2AccF16BM,cfg.nhwcStridedCoopmat2AccF16BN,
        cfg.nhwcStridedCoopmat2AccF16BK,aligned,kAligned,K,addToOutput);},
      [&](const CmdCtx& c,const ComputeKernel& kernel,VulkanBuffer* a,VulkanBuffer* b,VulkanBuffer* out,
          const GemmStridedPC& pc,int batches){dispatch(c,kernel,a,b,out,K,pc,batches);},
      cfg.nhwcStridedCoopmat2AccF16BN,cfg.nhwcStridedCoopmat2AccF16BK,
      STRIDED_COOPMAT2_PACKED_B_PAD_SCALARS, true);
  }

  // NHWC K-inner A and N-contiguous C use the transpose-based correctness oracle.
  KernelBench GemmStridedDot2Nhwc::bench(
    VulkanTuner::TuningContext& s,
    const VulkanTuneParams& cfg,
    int iters,
    int batchSize,
    int M,
    int N,
    int K,
    bool addToOutput) {
    if(s.dev == nullptr || !s.dev->info.supportsDot2F16 || !s.fp16Storage)
      return KernelBench();
    if(!GemmStridedDot2Nhwc::isConfigSupported(
         cfg.nhwcStridedDot2BlockSize,
         cfg.nhwcStridedDot2BM,
         cfg.nhwcStridedDot2BN,
         cfg.nhwcStridedDot2WM,
         cfg.nhwcStridedDot2WN,
         cfg.nhwcStridedDot2WMIter,
         cfg.nhwcStridedDot2TM,
         cfg.nhwcStridedDot2TN,
         cfg.nhwcStridedDot2Warp))
      return KernelBench();
    // DOT2's BK is a fixed shader #define (VulkanKernels::DOT2_BK), not a tuned
    // field, so alignment reads the NHWC BM/BN fields directly.
    const int32_t aligned = (M % cfg.nhwcStridedDot2BM == 0 && N % cfg.nhwcStridedDot2BN == 0) ? 1 : 0;
    const int32_t kAligned = (K % VulkanKernels::DOT2_BK == 0) ? 1 : 0;
    return benchGemmStridedNhwcVariant(
      s,
      iters,
      batchSize,
      M,
      N,
      K,
      "gemmStridedDot2Nhwc",
      [&]() {
        return GemmStridedDot2Nhwc::build(
          s.device(),
          VK_NULL_HANDLE,
          cfg.nhwcStridedDot2BlockSize,
          cfg.nhwcStridedDot2BM,
          cfg.nhwcStridedDot2BN,
          cfg.nhwcStridedDot2WM,
          cfg.nhwcStridedDot2WN,
          cfg.nhwcStridedDot2WMIter,
          cfg.nhwcStridedDot2TM,
          cfg.nhwcStridedDot2TN,
          cfg.nhwcStridedDot2Warp,
          aligned,
          1,
          kAligned,
          K,
          addToOutput);
      },
      [&](const CmdCtx& cctx,
          const ComputeKernel& kernel,
          VulkanBuffer* aBuf,
          VulkanBuffer* bBuf,
          VulkanBuffer* cBuf,
          const GemmStridedPC& pc,
          int dispatchBatchSize) {
        GemmStridedDot2Nhwc::dispatch(cctx, kernel, aBuf, bBuf, cBuf, K, pc, dispatchBatchSize);
      },
      cfg.nhwcStridedDot2BN,
      VulkanKernels::DOT2_BK,
      STRIDED_DOT2_PACKED_B_PAD_SCALARS);
  }

  // Native-NHWC companion to GemmStridedDot2AccF16::bench.
  KernelBench GemmStridedDot2AccF16Nhwc::bench(
    VulkanTuner::TuningContext& s,
    const VulkanTuneParams& cfg,
    int iters,
    int batchSize,
    int M,
    int N,
    int K,
    bool addToOutput) {
    if(s.dev == nullptr || !s.dev->info.supportsDot2F16AccF16 || !s.fp16Storage)
      return KernelBench();
    if(!GemmStridedDot2AccF16Nhwc::isConfigSupported(
         cfg.nhwcStridedDot2AccF16BlockSize,
         cfg.nhwcStridedDot2AccF16BM,
         cfg.nhwcStridedDot2AccF16BN,
         cfg.nhwcStridedDot2AccF16WM,
         cfg.nhwcStridedDot2AccF16WN,
         cfg.nhwcStridedDot2AccF16WMIter,
         cfg.nhwcStridedDot2AccF16TM,
         cfg.nhwcStridedDot2AccF16TN,
         cfg.nhwcStridedDot2AccF16Warp))
      return KernelBench();
    const int32_t aligned =
      (M % cfg.nhwcStridedDot2AccF16BM == 0 && N % cfg.nhwcStridedDot2AccF16BN == 0) ? 1 : 0;
    const int32_t kAligned = (K % VulkanKernels::DOT2_BK == 0) ? 1 : 0;
    return benchGemmStridedNhwcVariant(
      s,
      iters,
      batchSize,
      M,
      N,
      K,
      "gemmStridedDot2AccF16Nhwc",
      [&]() {
        return GemmStridedDot2AccF16Nhwc::build(
          s.device(),
          VK_NULL_HANDLE,
          cfg.nhwcStridedDot2AccF16BlockSize,
          cfg.nhwcStridedDot2AccF16BM,
          cfg.nhwcStridedDot2AccF16BN,
          cfg.nhwcStridedDot2AccF16WM,
          cfg.nhwcStridedDot2AccF16WN,
          cfg.nhwcStridedDot2AccF16WMIter,
          cfg.nhwcStridedDot2AccF16TM,
          cfg.nhwcStridedDot2AccF16TN,
          cfg.nhwcStridedDot2AccF16Warp,
          aligned,
          1,
          kAligned,
          K,
          addToOutput);
      },
      [&](const CmdCtx& cctx,
          const ComputeKernel& kernel,
          VulkanBuffer* aBuf,
          VulkanBuffer* bBuf,
          VulkanBuffer* cBuf,
          const GemmStridedPC& pc,
          int dispatchBatchSize) {
        GemmStridedDot2AccF16Nhwc::dispatch(cctx, kernel, aBuf, bBuf, cBuf, K, pc, dispatchBatchSize);
      },
      cfg.nhwcStridedDot2AccF16BN,
      VulkanKernels::DOT2_BK,
      STRIDED_DOT2_PACKED_B_PAD_SCALARS);
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

  KernelBench GPoolReductionNhwc::bench(VulkanTuner::TuningContext& s, const VulkanTuneParams& cfg, int iters) const {
    KernelBench result;
    if(iters < 1)
      return result;
    try {
      int xyStride = cfg.gpoolNhwcXystride;
      ComputeKernel kernel = GPoolReductionNhwc::build(s.device(), VK_NULL_HANDLE, s.fp16Storage, xyStride);

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

      GPoolReductionNhwc::PC pc = {problemChannels, problemXySize};
      auto recordOne = [&]() {
        CmdCtx cctx{s.cmd, s.getPushDescFn(), nullptr};
        GPoolReductionNhwc::dispatch(
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
        s.logger->write(std::string("VulkanTuner: gpoolReductionNhwc candidate failed: ") + e.what());
      return result;
    }
  }

  KernelBench ValueHeadPoolNhwc::bench(VulkanTuner::TuningContext& s, const VulkanTuneParams& cfg, int iters) const {
    KernelBench result;
    if(iters < 1)
      return result;
    try {
      int xyStride = cfg.valueHeadPoolNhwcXystride;
      ComputeKernel kernel = ValueHeadPoolNhwc::build(s.device(), VK_NULL_HANDLE, s.fp16Storage, xyStride);

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

      ValueHeadPoolNhwc::PC pc = {problemChannels, problemXySize};
      auto recordOne = [&]() {
        CmdCtx cctx{s.cmd, s.getPushDescFn(), nullptr};
        ValueHeadPoolNhwc::dispatch(
          cctx, kernel, inputBuf.get(), outputBuf.get(), maskSumBuf.get(), pc, problemBatchSize);
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
        s.logger->write(std::string("VulkanTuner: valueHeadPoolNhwc candidate failed: ") + e.what());
      return result;
    }
  }

  KernelBench SpatialRMSNormNhwc::bench(VulkanTuner::TuningContext& s, const VulkanTuneParams& cfg, int iters) const {
    KernelBench result;
    if(iters < 1)
      return result;
    try {
      const int tile = cfg.spatialRMSNormNhwcTile;
      const int spatialSize = problemChannels * problemXySize;
      const int numSpatialWorkgroups = (spatialSize + tile - 1) / tile;

      const bool canRequireSize = s.dev != nullptr &&
        s.dev->info.canRequireComputeSubgroupSize(s.dev->info.subgroupSize);
      const bool useSpatialRMSNormPass2Subgroup = s.dev != nullptr &&
        canUseRMSNormSubgroupVariant(
          s.dev->info.supportsSubgroupShuffleCompute,
          canRequireSize,
          s.dev->info.subgroupSize,
          (uint32_t)tile,
          /*wgChannelGroup=*/1u);
      const uint32_t rmsNormPass2SubgroupSize =
        useSpatialRMSNormPass2Subgroup ? s.dev->info.subgroupSize : 0u;
      std::array<ComputeKernel, 3> kernels = SpatialRMSNormNhwc::build(
        s.device(), VK_NULL_HANDLE, s.fp16Storage, tile, useSpatialRMSNormPass2Subgroup,
        rmsNormPass2SubgroupSize);

      size_t inputElts = (size_t)problemBatchSize * problemChannels * problemXySize;
      size_t maskElts = (size_t)problemBatchSize * problemXySize;
      size_t maskSumElts = (size_t)problemBatchSize;
      size_t gammaBetaElts = (size_t)problemChannels;
      size_t partialElts = (size_t)problemBatchSize * numSpatialWorkgroups;
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
        SpatialRMSNormNhwc::PC pc = {problemChannels, problemXySize, 1e-5f};
        SpatialRMSNormNhwc::dispatch(
          cctx, kernels, inputBuf.get(), outputBuf.get(), gammaBuf.get(), betaBuf.get(), maskBuf.get(),
          maskSumBuf.get(), partialsBuf.get(), scalarBuf.get(), pc, problemBatchSize);
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

  KernelBench AttentionCoopmat1Bench::bench(
    VulkanTuner::TuningContext& s, const VulkanTuneParams& cfg, int iters) const {
    KernelBench result;
    if(iters < 1 || s.dev == nullptr || !s.fp16Storage || !s.fp16Compute ||
       !s.dev->info.supportsCoopmat1F16)
      return result;
    if(!coopmatShapeIsSupported(
         s.dev->info.coopmatShapes,
         cfg.attnNhwcCoopmat1TM,
         cfg.attnNhwcCoopmat1TN,
         cfg.attnNhwcCoopmat1TK) ||
       cfg.attnNhwcCoopmat1Warp != (int32_t)s.dev->info.subgroupSize ||
       !validate(cfg, s.dev->info.properties.limits))
      return result;
    try {
      int kvChunkCount = cfg.attnNhwcCoopmat1KVChunkCount;
      ComputeKernel kernel = AttentionCoopmat1Nhwc::build(
        s.device(),
        VK_NULL_HANDLE,
        cfg.attnNhwcCoopmat1BlockSize,
        cfg.attnNhwcCoopmat1BlockQ,
        cfg.attnNhwcCoopmat1BlockKV,
        cfg.attnNhwcCoopmat1TM,
        cfg.attnNhwcCoopmat1TN,
        cfg.attnNhwcCoopmat1TK,
        cfg.attnNhwcCoopmat1Warp,
        problemNumTokens,
        problemHeadDim,
        problemVHeadDim,
        problemUseRope,
        problemLearnableRope,
        cfg.attnNhwcCoopmat1DirectKV != 0 &&
          problemNumTokens % cfg.attnNhwcCoopmat1BlockKV == 0,
        s.dev->info.subgroupSize,
        kvChunkCount,
        usesMaintenance1());

      int numBH = problemBatchSize * problemNumHeads;
      size_t qElts = (size_t)problemBatchSize * problemNumHeads * problemHeadDim * problemNumTokens;
      size_t kElts = (size_t)problemBatchSize * problemNumKVHeads * problemHeadDim * problemNumTokens;
      size_t vElts = (size_t)problemBatchSize * problemNumKVHeads * problemVHeadDim * problemNumTokens;
      size_t outElts = (size_t)problemBatchSize * problemNumHeads * problemVHeadDim * problemNumTokens;
      size_t maskElts = (size_t)problemBatchSize * problemNumTokens;
      int numPairs = problemHeadDim / 2;
      std::vector<float> qData(qElts), kData(kElts), vData(vElts), maskData(maskElts, 0.0f);
      for(int n = 0; n < problemBatchSize; n++)
        for(int pos = 0; pos < problemValidTokens; pos++)
          maskData[(size_t)n * problemNumTokens + pos] = 1.0f;
      uint32_t seed =
        (uint32_t)(problemNumTokens * 73856093u ^ problemHeadDim * 19349663u ^ problemNumHeads * 83492791u);
      VulkanTuner::fillRandom(qData, seed);
      VulkanTuner::fillRandom(kData, seed ^ 0x1111u);
      VulkanTuner::fillRandom(vData, seed ^ 0x2222u);
      VBuf qBuf = s.makeInputBufFP(qData);
      VBuf kBuf = s.makeInputBufFP(kData);
      VBuf vBuf = s.makeInputBufFP(vData);
      VBuf outBuf = makeDeviceBuf(s.device(), s.memProps(), outElts, true);
      VBuf maskBuf = s.makeInputBufFP(maskData);
      VBuf cosBuf;
      VBuf sinBuf;
      if(problemUseRope) {
        size_t tableElts =
          (size_t)(problemLearnableRope ? problemNumKVHeads : 1) * (size_t)numPairs * (size_t)problemNumTokens;
        std::vector<float> cosData(tableElts, 1.0f), sinData(tableElts, 0.0f);
        fillAttentionRopeTables(cosData, sinData, problemLearnableRope ? problemNumKVHeads : 1, numPairs, problemNumTokens);
        cosBuf = s.makeInputBuf(cosData);
        sinBuf = s.makeInputBuf(sinData);
      }
      VBuf partialsBuf;
      VBuf statsBuf;
      ComputeKernel resolveKernel{};
      if(kvChunkCount > 1) {
        size_t partialsElts = AttentionCoopmat1Nhwc::partialsBytes(problemNumTokens, numBH, kvChunkCount, problemVHeadDim) / sizeof(float);
        size_t statsElts = AttentionCoopmat1Nhwc::statsBytes(problemNumTokens, numBH, kvChunkCount) / sizeof(float);
        partialsBuf = makeDeviceBuf(s.device(), s.memProps(), partialsElts, false);
        statsBuf = makeDeviceBuf(s.device(), s.memProps(), statsElts, false);
        resolveKernel = AttentionSplitKResolveNhwc::build(s.device(), VK_NULL_HANDLE, problemVHeadDim);
      }
      AttentionTiled::PC pc = {
        problemNumHeads,
        problemNumKVHeads,
        1.0f / sqrtf((float)problemHeadDim),
        numBH,
      };
      auto recordOne = [&]() {
        CmdCtx cctx{s.cmd, s.getPushDescFn(), nullptr};
        AttentionCoopmat1Nhwc::dispatch(
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
          pc,
          kvChunkCount > 1 ? partialsBuf.get() : nullptr,
          kvChunkCount > 1 ? statsBuf.get() : nullptr);
        if(kvChunkCount > 1) {
          VulkanHelpers::cmdComputeBarrier(s.cmd, partialsBuf->buffer);
          VulkanHelpers::cmdComputeBarrier(s.cmd, statsBuf->buffer);
          AttentionSplitKResolveNhwc::PC resolvePC = {problemNumHeads, problemNumTokens, kvChunkCount};
          AttentionSplitKResolveNhwc::dispatch(
            cctx, resolveKernel, partialsBuf.get(), statsBuf.get(), outBuf.get(),
            problemNumTokens, numBH, resolvePC);
        }
        VulkanHelpers::cmdComputeBarrier(s.cmd, outBuf->buffer);
      };
      s.beginRecording();
      recordOne();
      s.submitAndWait();
      auto [seconds, effectiveIters] = timeBudgetedBench(s, iters, recordOne);
      if(seconds <= 0.0) {
        kernel.destroy(s.device());
        if(kvChunkCount > 1)
          resolveKernel.destroy(s.device());
        return result;
      }
      result.output.resize(outElts);
      s.downloadFloatsFP(outBuf.get(), result.output, outElts);
      for(int n = 0; n < problemBatchSize; n++)
        for(int pos = problemValidTokens; pos < problemNumTokens; pos++) {
          size_t base = ((size_t)n * problemNumTokens + pos) * problemNumHeads * problemVHeadDim;
          std::fill(result.output.begin() + base,
                    result.output.begin() + base + (size_t)problemNumHeads * problemVHeadDim,
                    0.0f);
        }
      result.kernelsPerSecond = (double)effectiveIters / seconds;
      result.ok = true;
      kernel.destroy(s.device());
      if(kvChunkCount > 1)
        resolveKernel.destroy(s.device());
      return result;
    } catch(const std::exception& e) {
      if(s.logger != nullptr)
        s.logger->write(std::string("VulkanTuner: attnCoopmat1Nhwc candidate failed: ") + e.what());
      return result;
    }
  }

  KernelBench AttentionCoopmat2AccF32Bench::bench(
    VulkanTuner::TuningContext& s, const VulkanTuneParams& cfg, int iters) const {
    KernelBench result;
    if(iters < 1 || s.dev == nullptr || !s.fp16Storage || !s.fp16Compute ||
       !s.dev->info.supportsCoopmat2F16 || !validate(cfg, s.dev->info.properties.limits))
      return result;
    try {
      ComputeKernel kernel = AttentionCoopmat2AccF32Nhwc::build(
        s.device(), VK_NULL_HANDLE, cfg.attnNhwcCoopmat2BlockSize, cfg.attnNhwcCoopmat2BlockQ,
        cfg.attnNhwcCoopmat2BlockKV, problemNumTokens, problemHeadDim, problemVHeadDim,
        problemUseRope, problemLearnableRope);
      const size_t qElts=(size_t)problemBatchSize*problemNumHeads*problemHeadDim*problemNumTokens;
      const size_t kElts=(size_t)problemBatchSize*problemNumKVHeads*problemHeadDim*problemNumTokens;
      const size_t vElts=(size_t)problemBatchSize*problemNumKVHeads*problemVHeadDim*problemNumTokens;
      const size_t outElts=(size_t)problemBatchSize*problemNumHeads*problemVHeadDim*problemNumTokens;
      std::vector<float> qData(qElts),kData(kElts),vData(vElts),maskData((size_t)problemBatchSize*problemNumTokens,0.0f);
      for(int n=0;n<problemBatchSize;n++) for(int p=0;p<problemValidTokens;p++) maskData[(size_t)n*problemNumTokens+p]=1.0f;
      uint32_t seed=(uint32_t)(problemNumTokens*73856093u^problemHeadDim*19349663u^problemNumHeads*83492791u);
      VulkanTuner::fillRandom(qData,seed); VulkanTuner::fillRandom(kData,seed^0x1111u); VulkanTuner::fillRandom(vData,seed^0x2222u);
      VBuf qBuf=s.makeInputBufFP(qData), kBuf=s.makeInputBufFP(kData), vBuf=s.makeInputBufFP(vData);
      VBuf outBuf=makeDeviceBuf(s.device(),s.memProps(),outElts,true), maskBuf=s.makeInputBufFP(maskData), cosBuf, sinBuf;
      if(problemUseRope) {
        const size_t tableElts=(size_t)(problemLearnableRope?problemNumKVHeads:1)*(problemHeadDim/2)*problemNumTokens;
        std::vector<float> cosData(tableElts,1.0f),sinData(tableElts,0.0f);
        fillAttentionRopeTables(cosData, sinData, problemLearnableRope ? problemNumKVHeads : 1, problemHeadDim/2, problemNumTokens);
        transposeAttentionRopeTables(
          cosData, sinData, problemLearnableRope ? problemNumKVHeads : 1,
          problemHeadDim / 2, problemNumTokens);
        cosBuf=s.makeInputBuf(cosData); sinBuf=s.makeInputBuf(sinData);
      }
      AttentionTiled::PC pc={problemNumHeads,problemNumKVHeads,1.0f/sqrtf((float)problemHeadDim),problemBatchSize*problemNumHeads};
      auto recordOne=[&]() { CmdCtx cctx{s.cmd,s.getPushDescFn(),nullptr};
        AttentionCoopmat2AccF32Nhwc::dispatch(
        cctx,kernel,qBuf.get(),kBuf.get(),vBuf.get(),outBuf.get(),maskBuf.get(),
        problemUseRope ? cosBuf.get() : nullptr, problemUseRope ? sinBuf.get() : nullptr,problemNumTokens,pc);
        VulkanHelpers::cmdComputeBarrier(s.cmd,outBuf->buffer); };
      s.beginRecording(); recordOne(); s.submitAndWait();
      auto [seconds,effectiveIters]=timeBudgetedBench(s,iters,recordOne);
      if(seconds<=0.0) { kernel.destroy(s.device()); return result; }
      result.output.resize(outElts); s.downloadFloatsFP(outBuf.get(),result.output,outElts);
      for(int n=0;n<problemBatchSize;n++) for(int p=problemValidTokens;p<problemNumTokens;p++) {
        const size_t base=((size_t)n*problemNumTokens+p)*problemNumHeads*problemVHeadDim;
        std::fill(result.output.begin()+base,result.output.begin()+base+(size_t)problemNumHeads*problemVHeadDim,0.0f);
      }
      result.kernelsPerSecond=(double)effectiveIters/seconds; result.ok=true; kernel.destroy(s.device()); return result;
    } catch(const std::exception& e) {
      if(s.logger!=nullptr) s.logger->write(std::string("VulkanTuner: attnCoopmat2Nhwc candidate failed: ")+e.what());
      return result;
    }
  }

  KernelBench AttentionDot2AccF32Bench::bench(
    VulkanTuner::TuningContext& s, const VulkanTuneParams& cfg, int iters) const {
    KernelBench result;
    if(iters < 1 || s.dev == nullptr || !s.fp16Storage || !s.fp16Compute ||
       !s.dev->info.supportsDot2F16 || !validate(cfg, s.dev->info.properties.limits))
      return result;
    try {
      ComputeKernel kernel = AttentionDot2AccF32Nhwc::build(
        s.device(),
        VK_NULL_HANDLE,
        cfg.attnNhwcDot2BlockSize,
        cfg.attnNhwcDot2BlockQ,
        cfg.attnNhwcDot2BlockKV,
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
      std::vector<float> qData(qElts), kData(kElts), vData(vElts), maskData(maskElts, 0.0f);
      for(int n = 0; n < problemBatchSize; n++)
        for(int pos = 0; pos < problemValidTokens; pos++)
          maskData[(size_t)n * problemNumTokens + pos] = 1.0f;
      uint32_t seed =
        (uint32_t)(problemNumTokens * 73856093u ^ problemHeadDim * 19349663u ^ problemNumHeads * 83492791u);
      VulkanTuner::fillRandom(qData, seed);
      VulkanTuner::fillRandom(kData, seed ^ 0x1111u);
      VulkanTuner::fillRandom(vData, seed ^ 0x2222u);
      VBuf qBuf = s.makeInputBufFP(qData);
      VBuf kBuf = s.makeInputBufFP(kData);
      VBuf vBuf = s.makeInputBufFP(vData);
      VBuf outBuf = makeDeviceBuf(s.device(), s.memProps(), outElts, true);
      VBuf maskBuf = s.makeInputBufFP(maskData);
      VBuf cosBuf;
      VBuf sinBuf;
      if(problemUseRope) {
        size_t tableElts =
          (size_t)(problemLearnableRope ? problemNumKVHeads : 1) * (size_t)numPairs * (size_t)problemNumTokens;
        std::vector<float> cosData(tableElts, 1.0f), sinData(tableElts, 0.0f);
        fillAttentionRopeTables(cosData, sinData, problemLearnableRope ? problemNumKVHeads : 1, numPairs, problemNumTokens);
        cosBuf = s.makeInputBuf(cosData);
        sinBuf = s.makeInputBuf(sinData);
      }
      AttentionTiled::PC pc = {
        problemNumHeads,
        problemNumKVHeads,
        1.0f / sqrtf((float)problemHeadDim),
        problemBatchSize * problemNumHeads,
      };
      auto recordOne = [&]() {
        CmdCtx cctx{s.cmd, s.getPushDescFn(), nullptr};
        AttentionDot2AccF32Nhwc::dispatch(
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
      for(int n = 0; n < problemBatchSize; n++)
        for(int pos = problemValidTokens; pos < problemNumTokens; pos++) {
          size_t base = ((size_t)n * problemNumTokens + pos) * problemNumHeads * problemVHeadDim;
          std::fill(result.output.begin() + base,
                    result.output.begin() + base + (size_t)problemNumHeads * problemVHeadDim,
                    0.0f);
        }
      result.kernelsPerSecond = (double)effectiveIters / seconds;
      result.ok = true;
      kernel.destroy(s.device());
      return result;
    } catch(const std::exception& e) {
      if(s.logger != nullptr)
        s.logger->write(std::string("VulkanTuner: attnDot2AccF32Nhwc candidate failed: ") + e.what());
      return result;
    }
  }

  KernelBench AttentionTiled::bench(VulkanTuner::TuningContext& s, const VulkanTuneParams& cfg, int iters) const {
    KernelBench result;
    if(iters < 1)
      return result;
    try {
      ComputeKernel kernel = problemNhwc ? AttentionTiledNhwc::build(
        s.device(), VK_NULL_HANDLE, s.fp16Storage,
        cfg.attnNhwcBlockQ, cfg.attnNhwcBlockKV, cfg.attnNhwcQPerThread,
        problemNumTokens, problemHeadDim, problemVHeadDim, problemUseRope, problemLearnableRope)
        : AttentionTiled::build(
        s.device(),
        VK_NULL_HANDLE,
        s.fp16Storage,
        problemNhwc ? cfg.attnNhwcBlockQ : cfg.attnBlockQ,
        problemNhwc ? cfg.attnNhwcBlockKV : cfg.attnBlockKV,
        problemNhwc ? cfg.attnNhwcQPerThread : cfg.attnQPerThread,
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

      std::vector<float> qData(qElts), kData(kElts), vData(vElts), maskData(maskElts, 0.0f);
      for(int n = 0; n < problemBatchSize; n++)
        for(int pos = 0; pos < problemValidTokens; pos++)
          maskData[(size_t)n * problemNumTokens + pos] = 1.0f;
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
        fillAttentionRopeTables(cosData, sinData, problemLearnableRope ? problemNumKVHeads : 1, numPairs, problemNumTokens);
        cosBuf = s.makeInputBuf(cosData);
        sinBuf = s.makeInputBuf(sinData);
      }

      AttentionTiled::PC pc = {
        problemNumHeads, problemNumKVHeads, 1.0f / sqrtf((float)problemHeadDim), problemBatchSize * problemNumHeads};
      auto recordOne = [&]() {
        CmdCtx cctx{s.cmd, s.getPushDescFn(), nullptr};
        if(problemNhwc)
          AttentionTiledNhwc::dispatch(
            cctx, kernel, qBuf.get(), kBuf.get(), vBuf.get(), outBuf.get(), maskBuf.get(),
            problemUseRope ? cosBuf.get() : nullptr, problemUseRope ? sinBuf.get() : nullptr,
            problemNumTokens, pc);
        else
          AttentionTiled::dispatch(
            cctx, kernel, qBuf.get(), kBuf.get(), vBuf.get(), outBuf.get(), maskBuf.get(),
            problemUseRope ? cosBuf.get() : nullptr, problemUseRope ? sinBuf.get() : nullptr,
            problemNumTokens, pc);
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
      if(problemNhwc) {
        for(int n = 0; n < problemBatchSize; n++)
          for(int pos = problemValidTokens; pos < problemNumTokens; pos++) {
            size_t base = ((size_t)n * problemNumTokens + pos) * problemNumHeads * problemVHeadDim;
            std::fill(result.output.begin() + base,
                      result.output.begin() + base + (size_t)problemNumHeads * problemVHeadDim,
                      0.0f);
          }
      }
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
